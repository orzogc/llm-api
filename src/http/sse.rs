//! Server-Sent Events parsing (WHATWG `text/event-stream`), on top of the
//! transport's byte stream.

use crate::error::{BodyKind, Error};

/// One complete logical SSE event (joined `data:` lines).
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SseEvent {
    /// The `event:` field, if any.
    pub event: Option<String>,
    /// Joined `data:` lines (separated by `\n`, trailing newline removed).
    pub data: String,
    /// The last seen `id:` field, if any (persists across events per spec).
    pub id: Option<String>,
    /// The `retry:` field, if any.
    pub retry: Option<u64>,
}

impl SseEvent {
    /// An event with the given type and data (test/construction helper).
    #[must_use]
    pub fn new(event: Option<&str>, data: impl Into<String>) -> Self {
        Self {
            event: event.map(str::to_owned),
            data: data.into(),
            id: None,
            retry: None,
        }
    }
}

/// Incremental SSE parser. Feed raw bytes with [`SseParser::push`]; call
/// [`SseParser::finish`] exactly once at end of stream.
///
/// Enforces the `max_sse_event` size cap (§ 12): one logical event larger
/// than the cap fails with [`Error::BodyTooLarge`].
#[derive(Debug)]
pub struct SseParser {
    max_event: usize,
    buf: Vec<u8>,
    data: String,
    event: Option<String>,
    id: Option<String>,
    retry: Option<u64>,
    bom_checked: bool,
}

impl SseParser {
    /// A parser enforcing the given per-event byte cap
    /// (`usize::MAX` ≈ unlimited).
    #[must_use]
    pub fn new(max_event: usize) -> Self {
        Self {
            max_event,
            buf: Vec::new(),
            data: String::new(),
            event: None,
            id: None,
            retry: None,
            bom_checked: false,
        }
    }

    /// Feeds bytes; returns the events completed by them.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<SseEvent>, Error> {
        self.buf.extend_from_slice(bytes);
        if !self.bom_checked && self.buf.len() >= 3 {
            if self.buf.starts_with(&[0xEF, 0xBB, 0xBF]) {
                self.buf.drain(..3);
            }
            self.bom_checked = true;
        }
        self.check_cap()?;
        self.drain_lines(false)
    }

    /// Signals end of stream. Complete trailing lines are processed; an
    /// unterminated partial line — and any half-built event — is discarded,
    /// per spec (protocol-level truncation detection is the stream parser's
    /// job, § 11).
    pub fn finish(&mut self) -> Result<Vec<SseEvent>, Error> {
        let events = self.drain_lines(true)?;
        self.buf.clear();
        self.data.clear();
        self.event = None;
        Ok(events)
    }

    fn check_cap(&self) -> Result<(), Error> {
        // Guard both the joined data and the raw buffer (a stream that
        // never sends a newline must not grow memory unboundedly).
        let current = self.data.len().max(self.buf.len());
        if current > self.max_event {
            let prefix = if self.data.len() >= self.buf.len() {
                bytes::Bytes::copy_from_slice(self.data.as_bytes())
            } else {
                bytes::Bytes::copy_from_slice(&self.buf)
            };
            return Err(Error::BodyTooLarge {
                kind: BodyKind::SseEvent,
                limit: self.max_event,
                status: None,
                headers: None,
                prefix,
            });
        }
        Ok(())
    }

    fn drain_lines(&mut self, eof: bool) -> Result<Vec<SseEvent>, Error> {
        let mut events = Vec::new();
        let mut start = 0;
        let mut i = 0;
        // Extract complete lines first (avoids borrowing `buf` while
        // mutating other fields).
        let mut lines: Vec<String> = Vec::new();
        while i < self.buf.len() {
            match self.buf[i] {
                b'\n' => {
                    lines.push(String::from_utf8_lossy(&self.buf[start..i]).into_owned());
                    i += 1;
                    start = i;
                }
                b'\r' => {
                    if i + 1 < self.buf.len() {
                        lines.push(String::from_utf8_lossy(&self.buf[start..i]).into_owned());
                        i += if self.buf[i + 1] == b'\n' { 2 } else { 1 };
                        start = i;
                    } else if eof {
                        lines.push(String::from_utf8_lossy(&self.buf[start..i]).into_owned());
                        i += 1;
                        start = i;
                    } else {
                        // A trailing `\r` may be half of a `\r\n` split
                        // across chunks; wait for the next byte.
                        break;
                    }
                }
                _ => i += 1,
            }
        }
        self.buf.drain(..start);
        for line in lines {
            if let Some(event) = self.process_line(&line)? {
                events.push(event);
            }
        }
        Ok(events)
    }

    fn process_line(&mut self, line: &str) -> Result<Option<SseEvent>, Error> {
        if line.is_empty() {
            // Dispatch. Per spec: no dispatch when the data buffer is
            // empty; the event type buffer still resets.
            if self.data.is_empty() {
                self.event = None;
                return Ok(None);
            }
            let mut data = std::mem::take(&mut self.data);
            if data.ends_with('\n') {
                data.pop();
            }
            let event = SseEvent {
                event: self.event.take(),
                data,
                id: self.id.clone(),
                retry: self.retry,
            };
            return Ok(Some(event));
        }
        if line.starts_with(':') {
            return Ok(None);
        }
        let (name, value) = match line.split_once(':') {
            Some((name, value)) => (name, value.strip_prefix(' ').unwrap_or(value)),
            None => (line, ""),
        };
        match name {
            "data" => {
                self.data.push_str(value);
                self.data.push('\n');
                self.check_cap()?;
            }
            "event" => self.event = Some(value.to_owned()),
            "id" => {
                if !value.contains('\0') {
                    self.id = Some(value.to_owned());
                }
            }
            "retry" => {
                if let Ok(ms) = value.parse::<u64>() {
                    self.retry = Some(ms);
                }
            }
            _ => {}
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(parser: &mut SseParser, chunks: &[&str]) -> Vec<SseEvent> {
        let mut events = Vec::new();
        for c in chunks {
            events.extend(parser.push(c.as_bytes()).unwrap());
        }
        events.extend(parser.finish().unwrap());
        events
    }

    #[test]
    fn basic_events() {
        let mut p = SseParser::new(usize::MAX);
        let events = collect(&mut p, &["data: hello\n\ndata: world\n\n"]);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "hello");
        assert_eq!(events[1].data, "world");
    }

    #[test]
    fn multi_line_data_joins_with_newline() {
        let mut p = SseParser::new(usize::MAX);
        let events = collect(&mut p, &["data: a\ndata: b\n\n"]);
        assert_eq!(events[0].data, "a\nb");
    }

    #[test]
    fn event_names_and_comments() {
        let mut p = SseParser::new(usize::MAX);
        let events = collect(
            &mut p,
            &[": comment\nevent: message_start\ndata: {}\n\ndata: next\n\n"],
        );
        assert_eq!(events[0].event.as_deref(), Some("message_start"));
        assert_eq!(events[0].data, "{}");
        // Event type resets after dispatch.
        assert_eq!(events[1].event, None);
    }

    #[test]
    fn chunk_splits_anywhere() {
        let mut p = SseParser::new(usize::MAX);
        let events = collect(&mut p, &["da", "ta: he", "llo\n", "\nda", "ta: x\n\n"]);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "hello");
        assert_eq!(events[1].data, "x");
    }

    #[test]
    fn crlf_and_cr_line_endings() {
        let mut p = SseParser::new(usize::MAX);
        let events = collect(&mut p, &["data: a\r\n\r\n", "data: b\r\rdata: c\n\n"]);
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].data, "a");
        assert_eq!(events[1].data, "b");
        assert_eq!(events[2].data, "c");
    }

    #[test]
    fn crlf_split_across_chunks() {
        let mut p = SseParser::new(usize::MAX);
        let mut events = p.push(b"data: a\r").unwrap();
        assert!(events.is_empty());
        events.extend(p.push(b"\n\r\n").unwrap());
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "a");
    }

    #[test]
    fn empty_data_no_dispatch() {
        let mut p = SseParser::new(usize::MAX);
        let events = collect(&mut p, &["event: ping\n\n"]);
        assert!(events.is_empty());
    }

    #[test]
    fn field_without_colon_and_no_space() {
        let mut p = SseParser::new(usize::MAX);
        let events = collect(&mut p, &["data\n\n", "data:tight\n\n"]);
        // "data" alone appends an empty line: dispatches as empty string.
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "");
        assert_eq!(events[1].data, "tight");
    }

    #[test]
    fn bom_stripped() {
        let mut p = SseParser::new(usize::MAX);
        let mut input = vec![0xEF, 0xBB, 0xBF];
        input.extend_from_slice(b"data: x\n\n");
        let events = p.push(&input).unwrap();
        assert_eq!(events[0].data, "x");
    }

    #[test]
    fn unterminated_tail_discarded_at_finish() {
        let mut p = SseParser::new(usize::MAX);
        p.push(b"data: complete\n\ndata: partial").unwrap();
        let events = p.finish().unwrap();
        assert!(
            events.is_empty(),
            "partial event must be discarded per spec"
        );
    }

    #[test]
    fn size_cap_enforced() {
        let mut p = SseParser::new(8);
        let err = p.push(b"data: 123456789012345\n\n").unwrap_err();
        assert!(matches!(
            err,
            Error::BodyTooLarge {
                kind: BodyKind::SseEvent,
                limit: 8,
                ..
            }
        ));

        // A no-newline flood also trips the cap.
        let mut p2 = SseParser::new(8);
        assert!(p2.push(b"0123456789abcdef").is_err());
    }

    #[test]
    fn id_and_retry_parsed() {
        let mut p = SseParser::new(usize::MAX);
        let events = collect(&mut p, &["id: 7\nretry: 100\ndata: x\n\ndata: y\n\n"]);
        assert_eq!(events[0].id.as_deref(), Some("7"));
        assert_eq!(events[0].retry, Some(100));
        // id persists across events per spec.
        assert_eq!(events[1].id.as_deref(), Some("7"));
    }
}
