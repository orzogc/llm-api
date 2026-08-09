//! The streaming call handle (§ 9 / § 12): SSE pump, warning attachment
//! and accumulation.

use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures_core::Stream;

use crate::convert::ConversionWarning;
use crate::error::{Error, Result};
use crate::format::StreamParser;
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
/// which the stream terminates. Items produced by
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
    queue: VecDeque<Result<StreamItem>>,
    pending_warnings: Vec<ConversionWarning>,
    state: State,
}

impl StreamHandle {
    /// Wraps an accepted (2xx) streaming response.
    pub(crate) fn new(
        status: u16,
        headers: http::HeaderMap,
        build_warnings: Vec<ConversionWarning>,
        body: BodyStream,
        sse: SseParser,
        parser: Box<dyn StreamParser>,
        include_raw: bool,
    ) -> Self {
        Self {
            status,
            headers,
            build_warnings,
            include_raw,
            body,
            sse,
            parser,
            queue: VecDeque::new(),
            pending_warnings: Vec::new(),
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
    /// the SSE parser get the response status/headers filled in.
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
        self.queue.push_back(Err(error));
        self.state = State::Draining;
    }

    /// Feeds complete SSE events to the stream parser.
    fn feed(&mut self, events: &[SseEvent]) {
        for event in events {
            if self.state == State::Draining {
                break;
            }
            match self.parser.parse(event) {
                Ok((parsed, warnings)) => self.emit(parsed, warnings, Some(event.data.as_str())),
                Err(e) => self.fatal(e),
            }
        }
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
    /// otherwise be lost.
    fn on_eof(&mut self) {
        match self.sse.finish() {
            Ok(events) => self.feed(&events),
            Err(e) => self.fatal(e),
        }
        if self.state != State::Draining {
            match self.parser.finish() {
                Ok((events, warnings)) => self.emit(events, warnings, None),
                Err(e) => self.fatal(e),
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
                Poll::Ready(Some(Ok(bytes))) => match this.sse.push(&bytes) {
                    Ok(events) => this.feed(&events),
                    Err(e) => this.fatal(e),
                },
                Poll::Ready(Some(Err(e))) => this.fatal(Error::Transport(e)),
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
