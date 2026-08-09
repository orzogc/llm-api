//! Unified error types for the whole crate.

use std::fmt;
use std::time::Duration;

use bytes::Bytes;
use serde_json::Value;

use crate::convert::ConversionWarning;
use crate::http::HttpError;
use crate::ir::Role;

/// Convenience alias used across the crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Top-level error type returned by client calls and conversion entry points.
#[non_exhaustive]
#[derive(Debug)]
pub enum Error {
    /// The HTTP transport failed before a response was produced.
    Transport(HttpError),
    /// The upstream API returned a non-2xx response.
    #[non_exhaustive]
    Api {
        /// HTTP status code.
        status: u16,
        /// Coarse, provider-independent classification.
        kind: ApiErrorKind,
        /// Best-effort human-readable message extracted from the body.
        message: String,
        /// Raw response body (may be non-JSON / non-UTF-8), possibly a prefix.
        raw: Bytes,
        /// `true` when `raw` is only a prefix (`max_error_body` was hit).
        truncated: bool,
        /// Present when the body parsed as JSON.
        parsed: Option<Value>,
        /// Parsed `Retry-After` header, if present.
        retry_after: Option<Duration>,
        /// Response headers.
        headers: http::HeaderMap,
    },
    /// Structural conversion failure or a strict-mode escalation.
    Conversion(ConversionError),
    /// A request hook returned an error.
    Hook(HookError),
    /// A 2xx body or stream could not be parsed.
    #[non_exhaustive]
    Parse {
        /// Description of the parse failure.
        message: String,
        /// The offending raw data.
        raw: Bytes,
    },
    /// A response body or SSE event exceeded a configured size cap.
    #[non_exhaustive]
    BodyTooLarge {
        /// Which kind of payload exceeded the cap.
        kind: BodyKind,
        /// The configured cap in bytes.
        limit: usize,
        /// HTTP status, where already known.
        status: Option<u16>,
        /// Response headers, where already known.
        headers: Option<http::HeaderMap>,
        /// The prefix read before the cap was hit.
        prefix: Bytes,
    },
    /// The capability is not supported by the configured format/endpoint.
    NotSupported(&'static str),
}

/// Coarse classification of upstream API errors.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApiErrorKind {
    /// The request was malformed or rejected by validation.
    InvalidRequest,
    /// Authentication failed (missing/invalid API key).
    Auth,
    /// The key is valid but not allowed to perform the operation.
    PermissionDenied,
    /// The requested resource (e.g. model) does not exist.
    NotFound,
    /// Rate limit exceeded.
    RateLimit,
    /// The provider is temporarily overloaded.
    Overloaded,
    /// Provider-side internal error.
    ServerError,
    /// Anything else; carries the provider's own error type string.
    Other(String),
}

impl ApiErrorKind {
    /// Default status-code based classification, used when a format has no
    /// more precise mapping for the error body.
    #[must_use]
    pub fn from_status(status: u16) -> Self {
        match status {
            401 => Self::Auth,
            403 => Self::PermissionDenied,
            404 => Self::NotFound,
            429 => Self::RateLimit,
            503 | 529 => Self::Overloaded,
            400..=499 => Self::InvalidRequest,
            500..=599 => Self::ServerError,
            _ => Self::Other(status.to_string()),
        }
    }
}

/// Which payload kind exceeded a size cap.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyKind {
    /// A 2xx (success) response body.
    SuccessBody,
    /// One complete logical SSE event (joined `data:` lines).
    SseEvent,
}

/// Structural failure while converting between the IR and a provider format.
///
/// These are *errors*, not warnings: data the target format structurally
/// requires is missing, or the IR itself is invalid. Lenient mode drops
/// extras with warnings but never fabricates required data (§ 6 of the
/// design).
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum ConversionError {
    /// The target format requires data the IR does not carry
    /// (Anthropic `max_tokens`, a tool-call id, a resolvable
    /// `functionResponse.name`, …).
    #[non_exhaustive]
    MissingRequired {
        /// What is missing.
        what: String,
        /// JSON-Pointer-style location in the would-be output.
        location: String,
    },
    /// `ToolCall.arguments` had to be embedded as a JSON object but is not
    /// parseable as one. Carries the raw string for diagnosis.
    #[non_exhaustive]
    InvalidToolArguments {
        /// Tool name of the offending call.
        name: String,
        /// The raw argument string as received.
        arguments: String,
        /// Parse failure or "not a JSON object" description.
        reason: String,
        /// Location of the offending block.
        location: String,
    },
    /// A content block is not valid for its message's role (§ 7.4).
    #[non_exhaustive]
    InvalidBlockForRole {
        /// The role of the containing message (or `System` for
        /// `Request.system`).
        role: Role,
        /// Human-readable name of the block kind.
        block: &'static str,
        /// Location of the offending block in the IR.
        location: String,
    },
    /// Strict mode: non-overridden semantic warnings were escalated.
    #[non_exhaustive]
    Strict {
        /// The escalated warnings.
        warnings: Vec<ConversionWarning>,
    },
    /// A user-supplied query key collides with a format-required one
    /// (e.g. `alt` on Google streaming).
    #[non_exhaustive]
    ProtectedQueryKey {
        /// The colliding key.
        key: String,
    },
    /// Any other structural problem.
    #[non_exhaustive]
    Other {
        /// Description.
        message: String,
    },
}

impl ConversionError {
    /// A `MissingRequired` error. Public constructor for third-party
    /// [`crate::ApiFormat`] implementations (the variants are
    /// `#[non_exhaustive]` and cannot be built with literals outside this
    /// crate).
    #[must_use]
    pub fn missing(what: impl Into<String>, location: impl Into<String>) -> Self {
        Self::MissingRequired { what: what.into(), location: location.into() }
    }

    /// An `InvalidToolArguments` error.
    #[must_use]
    pub fn invalid_tool_arguments(
        name: impl Into<String>,
        arguments: impl Into<String>,
        reason: impl Into<String>,
        location: impl Into<String>,
    ) -> Self {
        Self::InvalidToolArguments {
            name: name.into(),
            arguments: arguments.into(),
            reason: reason.into(),
            location: location.into(),
        }
    }

    /// An `InvalidBlockForRole` error.
    #[must_use]
    pub fn invalid_block_for_role(
        role: Role,
        block: &'static str,
        location: impl Into<String>,
    ) -> Self {
        Self::InvalidBlockForRole { role, block, location: location.into() }
    }

    /// An `Other` structural error.
    #[must_use]
    pub fn other(message: impl Into<String>) -> Self {
        Self::Other { message: message.into() }
    }
}

impl Error {
    /// A `Parse` error. Public constructor for third-party
    /// [`crate::ApiFormat`] / [`crate::StreamParser`] implementations.
    #[must_use]
    pub fn parse(message: impl Into<String>, raw: impl Into<Bytes>) -> Self {
        Self::Parse { message: message.into(), raw: raw.into() }
    }

    /// An `Api` error with the given classification; `retry_after` is
    /// extracted from `headers`, `parsed` from `raw` when it is JSON.
    #[must_use]
    pub fn api(
        status: u16,
        kind: ApiErrorKind,
        message: impl Into<String>,
        raw: impl Into<Bytes>,
        headers: http::HeaderMap,
    ) -> Self {
        let raw = raw.into();
        let parsed = serde_json::from_slice(&raw).ok();
        let retry_after = retry_after_from_headers(&headers);
        Self::Api {
            status,
            kind,
            message: message.into(),
            raw,
            truncated: false,
            parsed,
            retry_after,
            headers,
        }
    }
}

impl fmt::Display for ConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingRequired { what, location, .. } => {
                write!(f, "missing required data: {what} (at {location})")
            }
            Self::InvalidToolArguments { name, reason, location, .. } => {
                write!(f, "tool call `{name}`: arguments are not a JSON object: {reason} (at {location})")
            }
            Self::InvalidBlockForRole { role, block, location, .. } => {
                write!(f, "{block} block is not valid in a {role:?} message (at {location})")
            }
            Self::Strict { warnings, .. } => {
                write!(f, "strict mode: {} semantic warning(s) escalated to an error", warnings.len())?;
                if let Some(w) = warnings.first() {
                    write!(f, "; first: {}", w.message)?;
                }
                Ok(())
            }
            Self::ProtectedQueryKey { key, .. } => {
                write!(f, "query key `{key}` is required by the format and cannot be overridden")
            }
            Self::Other { message, .. } => f.write_str(message),
        }
    }
}

impl std::error::Error for ConversionError {}

/// Error returned by a user-supplied request hook.
#[derive(Debug)]
pub struct HookError {
    message: String,
    source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl HookError {
    /// Creates a hook error from a message.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self { message: message.into(), source: None }
    }

    /// Creates a hook error wrapping another error.
    #[must_use]
    pub fn with_source(
        message: impl Into<String>,
        source: impl Into<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self { message: message.into(), source: Some(source.into()) }
    }
}

impl fmt::Display for HookError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for HookError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_deref().map(|e| e as _)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transport(e) => write!(f, "transport error: {e}"),
            Self::Api { status, kind, message, truncated, .. } => {
                write!(f, "API error (HTTP {status}, {kind:?}): {message}")?;
                if *truncated {
                    write!(f, " [body truncated]")?;
                }
                Ok(())
            }
            Self::Conversion(e) => write!(f, "conversion error: {e}"),
            Self::Hook(e) => write!(f, "request hook failed: {e}"),
            Self::Parse { message, .. } => write!(f, "parse error: {message}"),
            Self::BodyTooLarge { kind, limit, .. } => {
                let what = match kind {
                    BodyKind::SuccessBody => "response body",
                    BodyKind::SseEvent => "SSE event",
                };
                write!(f, "{what} exceeded the configured cap of {limit} bytes")
            }
            Self::NotSupported(what) => write!(f, "not supported: {what}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Transport(e) => Some(e),
            Self::Conversion(e) => Some(e),
            Self::Hook(e) => Some(e),
            _ => None,
        }
    }
}

impl From<HttpError> for Error {
    fn from(e: HttpError) -> Self {
        Self::Transport(e)
    }
}

impl From<ConversionError> for Error {
    fn from(e: ConversionError) -> Self {
        Self::Conversion(e)
    }
}

impl From<HookError> for Error {
    fn from(e: HookError) -> Self {
        Self::Hook(e)
    }
}

/// Parses a `Retry-After` header as integer seconds.
///
/// HTTP-date values are not parsed and yield `None` (the providers covered by
/// this crate send integer seconds).
#[must_use]
pub fn retry_after_from_headers(headers: &http::HeaderMap) -> Option<Duration> {
    let v = headers.get(http::header::RETRY_AFTER)?;
    let s = v.to_str().ok()?.trim();
    s.parse::<u64>().ok().map(Duration::from_secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_error_kind_from_status() {
        assert_eq!(ApiErrorKind::from_status(400), ApiErrorKind::InvalidRequest);
        assert_eq!(ApiErrorKind::from_status(401), ApiErrorKind::Auth);
        assert_eq!(ApiErrorKind::from_status(403), ApiErrorKind::PermissionDenied);
        assert_eq!(ApiErrorKind::from_status(404), ApiErrorKind::NotFound);
        assert_eq!(ApiErrorKind::from_status(422), ApiErrorKind::InvalidRequest);
        assert_eq!(ApiErrorKind::from_status(429), ApiErrorKind::RateLimit);
        assert_eq!(ApiErrorKind::from_status(500), ApiErrorKind::ServerError);
        assert_eq!(ApiErrorKind::from_status(503), ApiErrorKind::Overloaded);
        assert_eq!(ApiErrorKind::from_status(529), ApiErrorKind::Overloaded);
    }

    #[test]
    fn retry_after_parses_seconds_only() {
        let mut h = http::HeaderMap::new();
        h.insert(http::header::RETRY_AFTER, http::HeaderValue::from_static("12"));
        assert_eq!(retry_after_from_headers(&h), Some(Duration::from_secs(12)));

        h.insert(
            http::header::RETRY_AFTER,
            http::HeaderValue::from_static("Sun, 06 Nov 1994 08:49:37 GMT"),
        );
        assert_eq!(retry_after_from_headers(&h), None);
    }
}
