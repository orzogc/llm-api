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
        // Strip a single leading UTF-8 BOM (spec: only at the very start of
        // the stream; a BOM anywhere else is content). Decided by prefix
        // comparison so chunk boundaries cannot change the outcome:
        // - Any diverging byte proves the stream does not start with a BOM
        //   and settles the check for good.
        // - Until settled, `buf` is a strict 1-2 byte BOM prefix; deferring
        //   is safe because `drain_lines` splits only on `\n`/`\r` and
        //   0xEF/0xBB are neither, so the prefix cannot be consumed as (part
        //   of) a line. At EOF such a leftover yields no line and `finish`
        //   discards it with the rest of the buffer.
        if !self.bom_checked {
            const BOM: [u8; 3] = [0xEF, 0xBB, 0xBF];
            let n = self.buf.len().min(BOM.len());
            if self.buf[..n] != BOM[..n] {
                self.bom_checked = true;
            } else if n == BOM.len() {
                self.buf.drain(..n);
                self.bom_checked = true;
            }
            // else: an empty buffer or a partial BOM prefix — undecided,
            // wait for more bytes.
        }
        // Drain complete lines first (the per-event data cap is enforced
        // inside `process_line`), then check what remains: `buf` now holds
        // only the current unfinished line, so this is a flood guard against
        // a stream that never sends a newline — it keeps accumulating across
        // pushes — and never trips on a chunk that carries several small
        // complete events. Known trade-off: a giant non-`data` line
        // (`event:`, `id:`, comment) that arrives complete within one chunk
        // is consumed into parser state unchecked — the § 12 cap is defined
        // over joined `data:` lines only.
        let events = self.drain_lines(false)?;
        self.check_cap()?;
        Ok(events)
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
        //
        // Invariant: `data` is either empty or ends with exactly one
        // trailing `\n` that dispatch removes, so `len - 1` is the final
        // event length dispatch would yield right now (interior `\n`
        // between joined lines is real content). `buf` holds an unfinished
        // line with no newline pending removal, so its length is not
        // adjusted.
        debug_assert!(self.data.is_empty() || self.data.ends_with('\n'));
        let data_len = self.data.len().saturating_sub(1);
        let current = data_len.max(self.buf.len());
        if current > self.max_event {
            let src = if data_len >= self.buf.len() {
                self.data.as_bytes()
            } else {
                &self.buf
            };
            // `prefix` carries at most `limit` bytes of what was read.
            // When `src` is `data`, `limit < data_len < data.len()` holds
            // here, so the pending trailing `\n` never enters the prefix.
            let end = src.len().min(self.max_event);
            return Err(Error::BodyTooLarge {
                kind: BodyKind::SseEvent,
                limit: self.max_event,
                status: None,
                headers: None,
                prefix: bytes::Bytes::copy_from_slice(&src[..end]),
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

    /// Parses `input` delivered as two chunks split at `at`, then finished.
    fn parse_split(input: &[u8], at: usize) -> Vec<SseEvent> {
        let mut p = SseParser::new(usize::MAX);
        let mut events = p.push(&input[..at]).unwrap();
        events.extend(p.push(&input[at..]).unwrap());
        events.extend(p.finish().unwrap());
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
    fn bom_stripped_across_chunk_boundaries() {
        // The leading BOM is stripped however it is split across chunks,
        // including byte-by-byte delivery.
        let mut input = vec![0xEF, 0xBB, 0xBF];
        input.extend_from_slice(b"data: x\n\n");
        for at in 0..=input.len() {
            let events = parse_split(&input, at);
            assert_eq!(events.len(), 1, "split at {at}");
            assert_eq!(events[0].data, "x", "split at {at}");
        }

        let mut p = SseParser::new(usize::MAX);
        let mut events = Vec::new();
        for b in &input {
            events.extend(p.push(&[*b]).unwrap());
        }
        events.extend(p.finish().unwrap());
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "x");
    }

    #[test]
    fn mid_stream_bom_is_content_regardless_of_chunking() {
        // The stream starts with a comment, so the later BOM is content: it
        // turns the following field name into "\u{FEFF}data", which is
        // ignored, and no event dispatches. Every split point must agree
        // with the unsplit parse — draining the short ":\n" chunk used to
        // leave the BOM check pending and mis-strip the next chunk's
        // mid-stream BOM.
        let mut input = b":\n".to_vec();
        input.extend_from_slice(&[0xEF, 0xBB, 0xBF]);
        input.extend_from_slice(b"data:x\n\n");
        let whole = parse_split(&input, input.len());
        assert!(whole.is_empty(), "mid-stream BOM must not be stripped");
        for at in 0..=input.len() {
            assert_eq!(parse_split(&input, at), whole, "split at {at}");
        }
    }

    #[test]
    fn mid_stream_bom_after_events_is_content() {
        // A non-BOM stream with a BOM between events: the BOM-prefixed
        // `data:` line is ignored on every chunking.
        let mut input = b"data:a\n\n".to_vec();
        input.extend_from_slice(&[0xEF, 0xBB, 0xBF]);
        input.extend_from_slice(b"data:b\n\n");
        let whole = parse_split(&input, input.len());
        assert_eq!(whole.len(), 1);
        assert_eq!(whole[0].data, "a");
        for at in 0..=input.len() {
            assert_eq!(parse_split(&input, at), whole, "split at {at}");
        }
    }

    #[test]
    fn bom_only_and_partial_bom_streams_end_cleanly() {
        // A stream that is exactly one BOM: no events, no error.
        let mut p = SseParser::new(usize::MAX);
        assert!(p.push(&[0xEF, 0xBB, 0xBF]).unwrap().is_empty());
        assert!(p.finish().unwrap().is_empty());

        // EOF while the check is still undecided on a 1-2 byte BOM prefix:
        // the leftover yields no line and is discarded.
        for prefix in [&[0xEF][..], &[0xEF, 0xBB][..]] {
            let mut p = SseParser::new(usize::MAX);
            assert!(p.push(prefix).unwrap().is_empty());
            assert!(p.finish().unwrap().is_empty());
        }
    }

    #[test]
    fn partial_bom_prefix_that_diverges_settles_the_check() {
        // [0xEF] alone is undecided; the following `\n` proves the stream
        // does not start with a BOM and settles the check. A later BOM is
        // then content, and the parser keeps working normally.
        let mut p = SseParser::new(usize::MAX);
        assert!(p.push(&[0xEF]).unwrap().is_empty());
        assert!(p.push(b"\n").unwrap().is_empty());
        let mut rest = vec![0xEF, 0xBB, 0xBF];
        rest.extend_from_slice(b"data:x\n\ndata:y\n\n");
        let mut events = p.push(&rest).unwrap();
        events.extend(p.finish().unwrap());
        // "\u{FEFF}data:x" is an ignored field; the clean event parses.
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "y");
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
    fn size_cap_ignores_chunk_size() {
        // One chunk carrying several small events must not trip the cap:
        // the limit is per logical event, not per network chunk.
        let mut p = SseParser::new(10);
        let events = p.push(b"data:a\n\ndata:b\n\ndata:c\n\n").unwrap();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].data, "a");
        assert_eq!(events[2].data, "c");
    }

    #[test]
    fn size_cap_result_is_chunking_independent() {
        let input = b"data: hello\n\ndata: world\n\n";
        let mut whole = SseParser::new(16);
        let all_at_once = collect(&mut whole, &[std::str::from_utf8(input).unwrap()]);

        // Byte-by-byte delivery of the same stream yields the same events.
        let mut split = SseParser::new(16);
        let mut one_by_one = Vec::new();
        for b in input {
            one_by_one.extend(split.push(&[*b]).unwrap());
        }
        one_by_one.extend(split.finish().unwrap());
        assert_eq!(all_at_once, one_by_one);
    }

    #[test]
    fn size_cap_ignores_comment_and_keepalive_overhead() {
        // A long comment line does not count against the event's data cap.
        let mut p = SseParser::new(8);
        let events = p
            .push(b": a keepalive comment much longer than the cap\ndata:a\n\n")
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "a");
    }

    #[test]
    fn size_cap_prefix_is_truncated_to_limit() {
        let mut p = SseParser::new(8);
        let err = p.push(&[b'x'; 30]).unwrap_err();
        match err {
            Error::BodyTooLarge { prefix, limit, .. } => {
                assert_eq!(limit, 8);
                assert_eq!(prefix.len(), 8, "prefix must not exceed the limit");
            }
            other => panic!("unexpected error: {other:?}"),
        }

        let mut p2 = SseParser::new(8);
        let err2 = p2.push(b"data: 0123456789abcdef\n").unwrap_err();
        match err2 {
            Error::BodyTooLarge { prefix, .. } => assert!(prefix.len() <= 8),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn size_cap_counts_final_event_length_exactly() {
        // A single-line payload exactly at the cap passes and dispatches:
        // the pending trailing `\n` (removed at dispatch) must not count.
        let mut p = SseParser::new(1);
        let events = p.push(b"data:a\n\n").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "a");

        // One byte over the cap fails; the prefix carries the payload
        // without the pending trailing newline.
        let mut p2 = SseParser::new(1);
        let err = p2.push(b"data:ab\n\n").unwrap_err();
        match err {
            Error::BodyTooLarge { prefix, limit, .. } => {
                assert_eq!(limit, 1);
                assert_eq!(&prefix[..], b"a");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn size_cap_counts_joined_multi_line_length_exactly() {
        // "a\nb" is exactly 3 bytes: the interior newline between joined
        // `data:` lines is real content and counts.
        let mut p = SseParser::new(3);
        let events = p.push(b"data:a\ndata:b\n\n").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "a\nb");

        // "ab\nc" is 4 bytes: over the cap.
        let mut p2 = SseParser::new(3);
        assert!(p2.push(b"data:ab\ndata:c\n\n").is_err());
    }

    #[test]
    fn size_cap_zero_allows_empty_payload() {
        let mut p = SseParser::new(0);
        let events = p.push(b"data:\n\n").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "");
    }

    #[test]
    fn size_cap_flood_accumulates_across_pushes() {
        // An unterminated line growing across pushes still trips the guard.
        let mut p = SseParser::new(8);
        assert!(p.push(b"01234").is_ok());
        assert!(p.push(b"56789").is_err());
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
