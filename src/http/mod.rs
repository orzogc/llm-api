//! The pluggable HTTP transport (§ 10).
//!
//! The library performs no IO by itself; any HTTP stack can be plugged in
//! via [`HttpClient`]. A `reqwest`-based default is feature-gated.

mod key;
#[cfg(feature = "reqwest")]
mod reqwest_client;
pub mod sse;

pub use key::ApiKey;
pub use sse::{SseEvent, SseParser};

use std::fmt;
use std::future::Future;
use std::pin::Pin;

use bytes::Bytes;
use futures_core::Stream;

/// The response body as a byte stream.
pub type BodyStream = Pin<Box<dyn Stream<Item = Result<Bytes, HttpError>> + Send>>;

/// Transport-level failure (before/while producing a response).
#[non_exhaustive]
#[derive(Debug)]
pub struct HttpError {
    /// Coarse classification.
    pub kind: HttpErrorKind,
    /// Underlying error, if any.
    pub source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl HttpError {
    /// An error with just a kind.
    #[must_use]
    pub fn new(kind: HttpErrorKind) -> Self {
        Self { kind, source: None }
    }

    /// An error wrapping an underlying source.
    #[must_use]
    pub fn with_source(
        kind: HttpErrorKind,
        source: impl Into<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self {
            kind,
            source: Some(source.into()),
        }
    }
}

impl fmt::Display for HttpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.kind {
            HttpErrorKind::Connect => "connection failed",
            HttpErrorKind::Timeout => "timed out",
            HttpErrorKind::Body => "body stream failed",
            HttpErrorKind::Protocol => "protocol error",
            HttpErrorKind::Other => "transport error",
        };
        match &self.source {
            Some(source) => write!(f, "{kind}: {source}"),
            None => f.write_str(kind),
        }
    }
}

/// Coarse transport error classification.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HttpErrorKind {
    /// Failed to establish a connection.
    Connect,
    /// The request or body timed out.
    Timeout,
    /// The body stream failed mid-transfer.
    Body,
    /// HTTP protocol violation.
    Protocol,
    /// Anything else.
    Other,
}

/// The authentication header the transport injects at send time.
///
/// The injected header value is `prefix + key`. This is passed to the
/// transport separately from the request so the API key never sits in an
/// `http::Request` that user code might log.
#[derive(Clone, Debug)]
pub struct AuthHeader {
    /// Header name (e.g. `authorization`, `x-api-key`).
    pub name: http::HeaderName,
    /// Value prefix, e.g. `"Bearer "`.
    pub prefix: Option<String>,
    /// The key itself.
    pub value: ApiKey,
}

impl AuthHeader {
    /// Builds the full header value (`prefix + key`). The result should be
    /// marked sensitive by transports that support it.
    pub fn header_value(&self) -> Result<http::HeaderValue, http::header::InvalidHeaderValue> {
        let mut s = String::new();
        if let Some(prefix) = &self.prefix {
            s.push_str(prefix);
        }
        s.push_str(self.value.expose());
        let mut value = http::HeaderValue::from_str(&s)?;
        value.set_sensitive(true);
        Ok(value)
    }
}

/// A pluggable HTTP client (§ 10).
///
/// Contract: the client sends the request **as-is**; its sole permitted
/// modification is injecting the provided [`AuthHeader`] at send time.
/// Every other header is the format layer's job. The request carries the
/// final URL — the client does no URL joining.
///
/// Chunk sizing: deliver the response body in moderately sized chunks
/// (network-buffer granularity). The size caps
/// ([`crate::client::Limits`]) bound the data the library accumulates,
/// while its peak memory scales with the largest single chunk — a
/// transport that yields a whole body as one chunk forfeits that
/// bounding.
pub trait HttpClient: Send + Sync {
    /// Sends the request, returning response head plus body stream.
    fn send(
        &self,
        request: http::Request<Bytes>,
        auth: Option<AuthHeader>,
    ) -> Pin<Box<dyn Future<Output = Result<http::Response<BodyStream>, HttpError>> + Send + '_>>;
}

impl std::error::Error for HttpError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_deref().map(|e| e as _)
    }
}
