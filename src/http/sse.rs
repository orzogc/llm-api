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
/// than the cap fails with [`Error::BodyTooLarge`]. The cap also bounds
/// the current unfinished raw line (flood guard against a stream that
/// never sends a newline), so a cap smaller than a line's raw length —
/// protocol overhead included — is chunking-sensitive: byte-wise delivery
/// can trip the guard on a line that a single-chunk push parses whole.
///
/// The cap bounds everything the parser accumulates and retains — the
/// joined data, the unfinished line, and the stored `event:`/`id:` values
/// (`id` persists across events) — not the instantaneous peak, which also
/// spans the chunk currently being pushed: the caller chooses chunk
/// granularity, and the transport's own allocation of that chunk is
/// outside the parser's control. A cap error discards the buffered input
/// and the half-built event (the error's `prefix` is the only retained
/// copy); subsequent pushes start at a fresh line boundary.
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

    /// Feeds bytes; returns the events completed by them. On error the
    /// events already parsed from this push are dropped; the
    /// crate-internal `SseParser::push_partial` keeps them.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<SseEvent>, Error> {
        match self.push_partial(bytes) {
            (events, None) => Ok(events),
            (_, Some(e)) => Err(e),
        }
    }

    /// Like [`SseParser::push`], but also delivers the events parsed
    /// before an error together with it — a protocol terminator fused
    /// into the same chunk as an oversized frame must still reach the
    /// stream parser (§ 9).
    pub(crate) fn push_partial(&mut self, bytes: &[u8]) -> (Vec<SseEvent>, Option<Error>) {
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
        // Drain complete lines first (the per-event data cap and the
        // stored `event:`/`id:` metadata cap are enforced inside
        // `process_line`; complete comment/unknown-field lines retain
        // nothing), then check what remains: `buf` now holds only the
        // current unfinished line, so this is a flood guard against a
        // stream that never sends a newline — it keeps accumulating across
        // pushes — and never trips on a chunk that carries several small
        // complete events. When the cap is smaller than an unfinished
        // line's raw length (field name and colon included), the outcome
        // depends on chunking: byte-wise delivery trips this guard before
        // the line completes while a single-chunk push parses it whole —
        // conservative, and required for flood protection.
        let (events, failed) = self.drain_lines(false);
        if let Some(e) = failed {
            return (events, Some(self.fail_reset(e)));
        }
        if let Err(e) = self.check_cap() {
            return (events, Some(self.fail_reset(e)));
        }
        (events, None)
    }

    /// Signals end of stream. Complete trailing lines are processed; an
    /// unterminated partial line — and any half-built event — is discarded,
    /// per spec (protocol-level truncation detection is the stream parser's
    /// job, § 11). On error the events already parsed are dropped; the
    /// crate-internal `SseParser::finish_partial` keeps them.
    pub fn finish(&mut self) -> Result<Vec<SseEvent>, Error> {
        match self.finish_partial() {
            (events, None) => Ok(events),
            (_, Some(e)) => Err(e),
        }
    }

    /// Like [`SseParser::finish`], but also delivers the events parsed
    /// before an error together with it.
    pub(crate) fn finish_partial(&mut self) -> (Vec<SseEvent>, Option<Error>) {
        let (events, failed) = self.drain_lines(true);
        if let Some(e) = failed {
            return (events, Some(self.fail_reset(e)));
        }
        self.buf.clear();
        self.data.clear();
        self.event = None;
        (events, None)
    }

    /// Applies the post-error contract: buffered input and the half-built
    /// event are discarded — the error's `prefix` is the only retained
    /// copy — so subsequent pushes start at a fresh line boundary. `id`
    /// and `retry` persist, as they do across [`SseParser::finish`].
    fn fail_reset(&mut self, error: Error) -> Error {
        self.buf = Vec::new();
        self.data = String::new();
        self.event = None;
        error
    }

    fn check_cap(&self) -> Result<(), Error> {
        // While the BOM check is undecided, `buf` is a 0-2 byte BOM prefix
        // and no line was ever drained, so `data` is empty. The prefix is
        // transport decoration, not event payload: counting it would make
        // caps of 0/1 fail on byte-wise delivery of a BOM that a whole-
        // stream push accepts. Exempt the window (≤ 2 bytes, so no flood
        // risk); the push that settles the check re-enters with
        // `bom_checked` set and counts the bytes normally if they turn out
        // to be content.
        if !self.bom_checked {
            debug_assert!(self.data.is_empty() && self.buf.len() <= 2);
            return Ok(());
        }
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

    /// The `BodyTooLarge` error for an over-cap `data:` line, built
    /// without materializing the oversized joined data: byte-for-byte
    /// what [`SseParser::check_cap`] would produce after the copy —
    /// `prefix` carries the first `limit` bytes of the prospective joined
    /// data (the pending trailing `\n` never enters:
    /// `max_event < final_len` holds when this is called).
    fn data_cap_error(&self, value: &str) -> Error {
        let final_len = self.data.len() + value.len();
        let end = self.max_event.min(final_len);
        let from_data = end.min(self.data.len());
        let mut prefix = Vec::with_capacity(end);
        prefix.extend_from_slice(&self.data.as_bytes()[..from_data]);
        prefix.extend_from_slice(&value.as_bytes()[..end - from_data]);
        Error::BodyTooLarge {
            kind: BodyKind::SseEvent,
            limit: self.max_event,
            status: None,
            headers: None,
            prefix: bytes::Bytes::from(prefix),
        }
    }

    /// A metadata-line overflow: an `event:`/`id:` value larger than the
    /// cap would otherwise persist unbounded in parser state. `prefix`
    /// carries the value's first `limit` bytes; nothing was stored.
    fn metadata_cap_error(&self, value: &str) -> Error {
        let end = self.max_event.min(value.len());
        Error::BodyTooLarge {
            kind: BodyKind::SseEvent,
            limit: self.max_event,
            status: None,
            headers: None,
            prefix: bytes::Bytes::copy_from_slice(&value.as_bytes()[..end]),
        }
    }

    fn drain_lines(&mut self, eof: bool) -> (Vec<SseEvent>, Option<Error>) {
        // Process complete lines straight out of the taken buffer: no
        // staging copies (only an invalid-UTF-8 line allocates, for the
        // lossy replacement). On success the buffer is moved back with
        // the processed region drained, leaving the unfinished tail; on
        // error it is dropped — the events parsed so far are still
        // returned, and the caller's `fail_reset` discards the rest.
        let owned = std::mem::take(&mut self.buf);
        let mut events = Vec::new();
        let mut failed = None;
        let mut pos = 0;
        while let Some((range, next)) = next_line(&owned, pos, eof) {
            pos = next;
            let line = String::from_utf8_lossy(&owned[range]);
            match self.process_line(&line) {
                Ok(Some(event)) => events.push(event),
                Ok(None) => {}
                Err(e) => {
                    failed = Some(e);
                    break;
                }
            }
        }
        if failed.is_none() {
            self.buf = owned;
            self.buf.drain(..pos);
        }
        (events, failed)
    }

    /// Processes one complete line.
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
                // After appending, `data` would be `data + value + "\n"`
                // and dispatch removes that trailing `\n`, so the final
                // event would be `data.len() + value.len()` bytes.
                // Checked *before* copying: an over-cap value never
                // inflates `data`, and the error is byte-identical to the
                // post-copy check it replaces. The unfinished tail is
                // deliberately not consulted here — the flood guard at
                // the end of the push covers it — so events completed by
                // a chunk still parse ahead of a trailing flood and reach
                // `push_partial` callers.
                if self.data.len() + value.len() > self.max_event {
                    return Err(self.data_cap_error(value));
                }
                self.data.push_str(value);
                self.data.push('\n');
            }
            "event" => {
                // Stored metadata folds into the § 12 cap: `event`/`id`
                // values persist in parser state (`id` across events), so
                // an oversized value is a cap error like oversized data —
                // never unbounded retained state.
                if value.len() > self.max_event {
                    return Err(self.metadata_cap_error(value));
                }
                self.event = Some(value.to_owned());
            }
            "id" => {
                if value.len() > self.max_event {
                    return Err(self.metadata_cap_error(value));
                }
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

/// Finds the next complete line in `buf` at `pos`: the line's byte range
/// plus the position after its terminator (`\n`, lone `\r`, or `\r\n`).
/// `None` at end of input — or at a trailing `\r` that may be half of a
/// `\r\n` split across chunks, unless `eof` says no byte can follow.
fn next_line(buf: &[u8], pos: usize, eof: bool) -> Option<(std::ops::Range<usize>, usize)> {
    let mut i = pos;
    while i < buf.len() {
        match buf[i] {
            b'\n' => return Some((pos..i, i + 1)),
            b'\r' => {
                return if i + 1 < buf.len() {
                    Some((pos..i, i + if buf[i + 1] == b'\n' { 2 } else { 1 }))
                } else if eof {
                    Some((pos..i, i + 1))
                } else {
                    None
                };
            }
            _ => i += 1,
        }
    }
    None
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
    fn metadata_lines_fold_into_the_cap() {
        // Stored `event:`/`id:` values count against the cap even when
        // the line arrives complete within one chunk — parser-retained
        // state is bounded regardless of chunking.
        let mut p = SseParser::new(8);
        let err = p.push(b"event: 0123456789abcdef\ndata:a\n\n").unwrap_err();
        match err {
            Error::BodyTooLarge { prefix, limit, .. } => {
                assert_eq!(limit, 8);
                assert_eq!(prefix.len(), 8, "prefix must not exceed the limit");
            }
            other => panic!("unexpected error: {other:?}"),
        }
        // Error recovery: nothing was stored, later pushes parse afresh.
        let events = collect(&mut p, &["data:ok\n\n"]);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "ok");

        let mut p = SseParser::new(8);
        let err = p.push(b"id: 0123456789abcdef\n").unwrap_err();
        assert!(matches!(err, Error::BodyTooLarge { .. }));
        let events = collect(&mut p, &["id: 7\ndata:a\n\n"]);
        assert_eq!(events[0].id.as_deref(), Some("7"));

        // A value exactly at the cap still parses.
        let mut p = SseParser::new(8);
        let events = collect(&mut p, &["event: 12345678\ndata:a\n\n"]);
        assert_eq!(events[0].event.as_deref(), Some("12345678"));
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
    fn size_cap_below_raw_line_length_is_chunking_sensitive() {
        // Pinned trade-off (see `push` and the struct doc): a cap smaller
        // than a line's raw length rejects byte-wise delivery — the flood
        // guard must bound the unfinished line before it completes — while
        // a single-chunk push parses the same bytes whole.
        let mut whole = SseParser::new(1);
        let events = whole.push(b"data:a\n\n").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "a");

        let mut bytewise = SseParser::new(1);
        let mut tripped = false;
        for byte in b"data:a\n\n" {
            if bytewise.push(&[*byte]).is_err() {
                tripped = true;
                break;
            }
        }
        assert!(tripped, "byte-wise delivery must trip the flood guard");
    }

    #[test]
    fn size_cap_zero_exempts_undecided_bom_prefix() {
        // A pure-BOM stream delivered byte by byte at cap 0: the undecided
        // 1-2 byte prefix is transport decoration and must not count, so
        // every push succeeds with no events — same as one whole push.
        let mut p = SseParser::new(0);
        for b in [0xEF, 0xBB, 0xBF] {
            assert!(p.push(&[b]).unwrap().is_empty());
        }
        assert!(p.finish().unwrap().is_empty());

        let mut whole = SseParser::new(0);
        assert!(whole.push(&[0xEF, 0xBB, 0xBF]).unwrap().is_empty());
        assert!(whole.finish().unwrap().is_empty());

        // EOF while still undecided on a partial prefix also stays clean.
        for prefix in [&[0xEF][..], &[0xEF, 0xBB][..]] {
            let mut p = SseParser::new(0);
            assert!(p.push(prefix).unwrap().is_empty());
            assert!(p.finish().unwrap().is_empty());
        }
    }

    #[test]
    fn size_cap_zero_unaffected_by_leading_bom() {
        // After a byte-wise BOM, cap accounting matches the BOM-less stream:
        // an empty payload still fits a zero cap and dispatches.
        let mut p = SseParser::new(0);
        for b in [0xEF, 0xBB, 0xBF] {
            assert!(p.push(&[b]).unwrap().is_empty());
        }
        let events = p.push(b"data:\n\n").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "");
    }

    #[test]
    fn size_cap_counts_settled_bom_lookalike_as_content() {
        // A diverging byte proves the 1-byte prefix was content all along;
        // the settling push counts it: [EF, x] is a 2-byte unfinished line
        // over cap 1, identically whether split or pushed whole.
        let mut split = SseParser::new(1);
        assert!(split.push(&[0xEF]).unwrap().is_empty());
        let err = split.push(b"x").unwrap_err();
        assert!(matches!(
            err,
            Error::BodyTooLarge {
                kind: BodyKind::SseEvent,
                limit: 1,
                ..
            }
        ));

        let mut whole = SseParser::new(1);
        let err = whole.push(&[0xEF, b'x']).unwrap_err();
        assert!(matches!(
            err,
            Error::BodyTooLarge {
                kind: BodyKind::SseEvent,
                limit: 1,
                ..
            }
        ));
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

    #[test]
    fn size_cap_oversized_data_error_prefix_spans_joined_lines() {
        // The pre-copy check builds the same prefix the post-copy check
        // used to: the first `limit` bytes of the prospective joined data
        // (previous lines, interior `\n`, then the offending value).
        let mut p = SseParser::new(4);
        let err = p.push(b"data:ab\ndata:cde\n\n").unwrap_err();
        match err {
            Error::BodyTooLarge { prefix, limit, .. } => {
                assert_eq!(limit, 4);
                assert_eq!(&prefix[..], b"ab\nc");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn size_cap_flood_tail_fails_push_despite_complete_events() {
        // A chunk carrying a small complete event followed by an over-cap
        // unfinished line: the public `push` fails (its events are
        // dropped; `push_partial` keeps them) and the prefix comes from
        // the raw tail via the end-of-push flood guard.
        let mut p = SseParser::new(8);
        let mut input = b"data:a\n\n".to_vec();
        input.extend_from_slice(&[b'x'; 20]);
        let err = p.push(&input).unwrap_err();
        match err {
            Error::BodyTooLarge {
                kind: BodyKind::SseEvent,
                prefix,
                ..
            } => assert_eq!(&prefix[..], &[b'x'; 8]),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn invalid_utf8_line_is_replaced_lossily() {
        let mut p = SseParser::new(usize::MAX);
        let events = p.push(b"data: h\xFFi\n\n").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "h\u{FFFD}i");
    }

    /// Asserts the post-error contract: nothing buffered remains (the
    /// flood allocation is dropped, not merely truncated) and a clean
    /// event parses from a fresh line boundary.
    fn assert_reset_and_recovered(p: &mut SseParser) {
        assert_eq!(p.buf.capacity(), 0, "buffered input must be discarded");
        assert!(p.data.is_empty(), "half-built event data must be discarded");
        assert!(p.event.is_none(), "half-built event type must be discarded");
        let events = p
            .push(b"data: ok\n\n")
            .expect("clean push parses after reset");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "ok");
    }

    #[test]
    fn cap_error_discards_flood_buffer_and_recovers() {
        // Flood-guard path: a no-newline chunk far over the cap. The
        // error's prefix is built before the reset and is unaffected.
        let mut p = SseParser::new(8);
        let err = p.push(&vec![b'a'; 100 * 1024]).unwrap_err();
        match err {
            Error::BodyTooLarge { prefix, limit, .. } => {
                assert_eq!(limit, 8);
                assert_eq!(&prefix[..], &[b'a'; 8]);
            }
            other => panic!("unexpected error: {other:?}"),
        }
        assert_reset_and_recovered(&mut p);
    }

    #[test]
    fn cap_error_discards_tail_flood_after_events_and_recovers() {
        // Complete event + over-cap unfinished tail in one push: the
        // flood guard trips at the end of the push; the retained state is
        // discarded all the same.
        let mut p = SseParser::new(8);
        let mut input = b"data:a\n\n".to_vec();
        input.extend_from_slice(&vec![b'x'; 50 * 1024]);
        let err = p.push(&input).unwrap_err();
        match err {
            Error::BodyTooLarge { prefix, .. } => assert_eq!(&prefix[..], &[b'x'; 8]),
            other => panic!("unexpected error: {other:?}"),
        }
        assert_reset_and_recovered(&mut p);
    }

    #[test]
    fn cap_error_discards_data_state_and_recovers() {
        // Data pre-check path: a complete oversized `data:` line. The
        // over-cap value never entered `data`; the reset still applies.
        let mut p = SseParser::new(8);
        let line = format!("data: {}\n\n", "b".repeat(64));
        let err = p.push(line.as_bytes()).unwrap_err();
        match err {
            Error::BodyTooLarge { prefix, .. } => assert_eq!(&prefix[..], &[b'b'; 8]),
            other => panic!("unexpected error: {other:?}"),
        }
        assert_reset_and_recovered(&mut p);
    }

    #[test]
    fn finish_cap_error_resets_too() {
        // A trailing `\r` becomes a line terminator only at EOF, so the
        // joined data first exceeds the cap inside `finish`: it fails and
        // applies the same reset.
        let mut p = SseParser::new(11);
        assert!(p.push(b"data:abcdefgh\n").unwrap().is_empty());
        assert!(p.push(b"data:abc\r").unwrap().is_empty());
        let err = p.finish().unwrap_err();
        assert!(matches!(
            err,
            Error::BodyTooLarge {
                kind: BodyKind::SseEvent,
                limit: 11,
                ..
            }
        ));
        assert_eq!(p.buf.capacity(), 0);
        assert!(p.data.is_empty());
    }

    #[test]
    fn push_partial_delivers_events_parsed_before_error() {
        // A complete terminator-style event fused with a complete
        // oversized frame: the parsed event rides alongside the error.
        let mut p = SseParser::new(20);
        let mut input = b"data: [DONE]\n\n".to_vec();
        input.extend_from_slice(format!("data: {}\n\n", "y".repeat(64)).as_bytes());
        let (events, err) = p.push_partial(&input);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "[DONE]");
        assert!(matches!(err, Some(Error::BodyTooLarge { .. })));

        // Fused with an over-cap unfinished flood instead: the complete
        // event still parses ahead of the end-of-push flood guard.
        let mut p = SseParser::new(20);
        let mut input = b"data: [DONE]\n\n".to_vec();
        input.extend_from_slice(&[b'x'; 64]);
        let (events, err) = p.push_partial(&input);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "[DONE]");
        assert!(matches!(err, Some(Error::BodyTooLarge { .. })));

        // The public wrapper drops the events, unchanged.
        let mut p = SseParser::new(20);
        let mut input = b"data: [DONE]\n\n".to_vec();
        input.extend_from_slice(&[b'x'; 64]);
        assert!(p.push(&input).is_err());
    }
}
