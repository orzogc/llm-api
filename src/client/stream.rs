//! The streaming call handle (§ 9 / § 12): SSE pump, warning attachment
//! and accumulation.

use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures_core::Stream;

use crate::convert::{ConversionWarning, WarningCode};
use crate::error::{Error, Result};
use crate::format::{ApiFormat, StreamParser};
use crate::http::{BodyStream, SseEvent, SseParser};
use crate::ir::{Accumulator, Response, StreamEvent, StreamItem};

/// Pump state: `Pumping` reads the body, `Draining` serves the remaining
/// queue (body finished or a fatal error was queued), `Done` yields `None`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Pumping,
    Draining,
    Done,
}

/// A live streaming call, returned by [`super::Client::stream`].
///
/// Implements [`futures_core::Stream`] with `Item = Result<StreamItem>`:
/// body bytes feed the SSE parser, each complete SSE event feeds the
/// format's [`StreamParser`], and the resulting items are yielded one by
/// one. Warning attachment (§ 9): the warnings of one parse call ride the
/// first item that call emits; when a call emits none they are held for the
/// next item, and warnings still held at end of stream with no item to
/// carry them are delivered on one synthetic
/// `StreamItem { event: Unknown, raw: None }`.
///
/// A transport error, an SSE event over
/// [`super::Limits::max_sse_event`] (reported with the response status and
/// headers filled in) or a parser failure is yielded as one `Err`, after
/// which the stream terminates — but only before the protocol terminator.
/// Past the parser's [`StreamEvent::MessageStop`] the response is
/// complete: such failures downgrade to one cosmetic
/// [`WarningCode::PostTerminalStreamFailure`] warning riding the
/// end-of-stream carrier and the stream ends cleanly; SSE events decoded
/// before an in-chunk failure are still delivered. Items produced by
/// [`StreamParser::finish`] — including the synthetic warning carrier —
/// have `raw: None` (there is no source SSE event for them).
pub struct StreamHandle {
    status: u16,
    headers: http::HeaderMap,
    build_warnings: Vec<ConversionWarning>,
    include_raw: bool,
    body: BodyStream,
    sse: SseParser,
    parser: Box<dyn StreamParser>,
    /// Canonical id of the chat endpoint's format, for warnings the
    /// handle produces itself.
    format_id: String,
    queue: VecDeque<Result<StreamItem>>,
    pending_warnings: Vec<ConversionWarning>,
    /// The parser delivered its protocol terminator (`MessageStop`): the
    /// response is complete, and later pump-level failures downgrade to
    /// a warning instead of failing the stream.
    protocol_terminal: bool,
    state: State,
}

impl StreamHandle {
    /// Wraps an accepted (2xx) streaming response. `format` supplies the
    /// stream parser and the canonical id carried by handle-produced
    /// warnings.
    pub(crate) fn new(
        status: u16,
        headers: http::HeaderMap,
        build_warnings: Vec<ConversionWarning>,
        body: BodyStream,
        sse: SseParser,
        format: &dyn ApiFormat,
        include_raw: bool,
    ) -> Self {
        Self {
            status,
            headers,
            build_warnings,
            include_raw,
            body,
            sse,
            parser: format.stream_parser(),
            format_id: format.id().to_owned(),
            queue: VecDeque::new(),
            pending_warnings: Vec::new(),
            protocol_terminal: false,
            state: State::Pumping,
        }
    }

    /// Request-build warnings (the streaming counterpart of
    /// [`Response::warnings`]'s build-side prefix).
    #[must_use]
    pub fn warnings(&self) -> &[ConversionWarning] {
        &self.build_warnings
    }

    /// HTTP status of the streaming response.
    #[must_use]
    pub fn status(&self) -> u16 {
        self.status
    }

    /// HTTP headers of the streaming response.
    #[must_use]
    pub fn headers(&self) -> &http::HeaderMap {
        &self.headers
    }

    /// Consumes the stream and accumulates it into a [`Response`] using the
    /// core [`Accumulator`], seeded with the response status, headers and
    /// the request-build warnings. Fails with the first stream error, or
    /// with `Error::Parse` when the stream ends without its protocol
    /// terminator.
    pub async fn collect(mut self) -> Result<Response> {
        let mut acc = Accumulator::new();
        acc.set_status(self.status);
        acc.set_headers(self.headers.clone());
        acc.extend_warnings(std::mem::take(&mut self.build_warnings));
        while let Some(item) = std::future::poll_fn(|cx| Pin::new(&mut self).poll_next(cx)).await {
            acc.push(&item?)?;
        }
        acc.finish()
    }

    /// Queues a fatal error and stops pumping. `BodyTooLarge` errors from
    /// the SSE parser get the response status/headers filled in; `Api`
    /// errors built by a stream parser from an in-stream error event get
    /// the response headers and `retry_after` filled in.
    fn fatal(&mut self, mut error: Error) {
        if let Error::BodyTooLarge {
            status, headers, ..
        } = &mut error
        {
            status.get_or_insert(self.status);
            if headers.is_none() {
                *headers = Some(self.headers.clone());
            }
        }
        if let Error::Api {
            headers,
            retry_after,
            ..
        } = &mut error
        {
            // An empty header map is the parser-constructed sentinel: a
            // real HTTP response always carries headers, and stream parsers
            // only see the event body. A non-empty map is preserved — a
            // third-party parser may have set it deliberately. `status` is
            // left alone (200 could be either the sentinel or genuine).
            if headers.is_empty() {
                *headers = self.headers.clone();
            }
            if retry_after.is_none() {
                *retry_after = crate::error::retry_after_from_headers(&self.headers);
            }
        }
        self.queue.push_back(Err(error));
        self.state = State::Draining;
    }

    /// Feeds complete SSE events to the stream parser; stops at and
    /// returns the parser's first error, leaving later events unfed.
    fn feed(&mut self, events: &[SseEvent]) -> Option<Error> {
        for event in events {
            match self.parser.parse(event) {
                Ok((parsed, warnings)) => self.emit(parsed, warnings, Some(event.data.as_str())),
                Err(e) => return Some(e),
            }
        }
        None
    }

    /// Records a pump-level failure that arrived after the protocol
    /// terminator: the completed response stands and the failure rides
    /// the § 9 warning machinery instead (delivered by the end-of-stream
    /// carrier when no later item picks it up).
    fn post_terminal_warning(&mut self, error: &Error) {
        self.pending_warnings.push(ConversionWarning::from_format(
            WarningCode::PostTerminalStreamFailure,
            self.format_id.clone(),
            "",
            format!("stream failure after the protocol terminator: {error}"),
        ));
    }

    /// Routes a pump-level failure (transport, SSE decode or parser
    /// error): fatal before the protocol terminator; past it the failure
    /// downgrades to a [`WarningCode::PostTerminalStreamFailure`] warning
    /// and the stream ends cleanly.
    fn pump_failure(&mut self, error: Error) {
        if !self.protocol_terminal {
            self.fatal(error);
            return;
        }
        self.post_terminal_warning(&error);
        self.on_eof();
    }

    /// Queues the items of one parse call, applying the § 9 warning and raw
    /// attachment rules. `raw` is the source SSE event's data, absent for
    /// events produced by [`StreamParser::finish`].
    fn emit(
        &mut self,
        events: Vec<StreamEvent>,
        warnings: Vec<ConversionWarning>,
        raw: Option<&str>,
    ) {
        let mut carried = std::mem::take(&mut self.pending_warnings);
        carried.extend(warnings);
        if events.is_empty() {
            self.pending_warnings = carried;
            return;
        }
        for event in events {
            if matches!(event, StreamEvent::MessageStop) {
                self.protocol_terminal = true;
            }
            let raw = raw
                .filter(|_| self.include_raw || matches!(event, StreamEvent::Unknown))
                .map(str::to_owned);
            // The first item of the batch takes all carried warnings.
            self.queue.push_back(Ok(StreamItem {
                event,
                raw,
                warnings: std::mem::take(&mut carried),
            }));
        }
    }

    /// End of the byte stream: flush the SSE parser, finish the stream
    /// parser, and synthesize a carrier item for warnings that would
    /// otherwise be lost. Failures here are fatal before the protocol
    /// terminator and downgrade to a warning past it.
    fn on_eof(&mut self) {
        let (events, sse_err) = self.sse.finish_partial();
        let parser_err = self.feed(&events);
        if let Some(e) = parser_err.or(sse_err) {
            if self.protocol_terminal {
                self.post_terminal_warning(&e);
            } else {
                self.fatal(e);
            }
        }
        if self.state != State::Draining {
            match self.parser.finish() {
                Ok((events, warnings)) => self.emit(events, warnings, None),
                Err(e) => {
                    if self.protocol_terminal {
                        self.post_terminal_warning(&e);
                    } else {
                        self.fatal(e);
                    }
                }
            }
        }
        if self.state != State::Draining && !self.pending_warnings.is_empty() {
            let warnings = std::mem::take(&mut self.pending_warnings);
            self.queue.push_back(Ok(StreamItem {
                event: StreamEvent::Unknown,
                raw: None,
                warnings,
            }));
        }
        self.state = State::Draining;
    }
}

impl Stream for StreamHandle {
    type Item = Result<StreamItem>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(item) = this.queue.pop_front() {
                if item.is_err() {
                    // An error terminates the stream (§ 9): subsequent
                    // polls yield `None`.
                    this.state = State::Done;
                }
                return Poll::Ready(Some(item));
            }
            match this.state {
                State::Done => return Poll::Ready(None),
                State::Draining => {
                    this.state = State::Done;
                    return Poll::Ready(None);
                }
                State::Pumping => {}
            }
            match this.body.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Ok(bytes))) => {
                    // Feed the events decoded before any in-chunk failure
                    // first: a terminator fused into the same chunk must
                    // count before the failure is routed. The parser's
                    // error precedes the SSE one chronologically (it
                    // struck an earlier event).
                    let (events, sse_err) = this.sse.push_partial(&bytes);
                    let parser_err = this.feed(&events);
                    if let Some(e) = parser_err.or(sse_err) {
                        this.pump_failure(e);
                    }
                }
                Poll::Ready(Some(Err(e))) => this.pump_failure(Error::Transport(e)),
                Poll::Ready(None) => this.on_eof(),
            }
        }
    }
}

impl std::fmt::Debug for StreamHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamHandle")
            .field("status", &self.status)
            .field("headers", &self.headers)
            .field("build_warnings", &self.build_warnings)
            .field("include_raw", &self.include_raw)
            .field("queued", &self.queue.len())
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}
