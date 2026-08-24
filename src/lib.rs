//! `llm-api`: a unified intermediate representation (IR) for LLM chat APIs,
//! bidirectional conversion between the IR and multiple provider API
//! formats, and a pluggable HTTP transport.
//!
//! Users build requests against the IR and pick the upstream API format at
//! call time. The two non-negotiable differentiators (design § 1):
//!
//! - **Free JSON manipulation** — arbitrary fields of the serialized
//!   request can be set/overridden/deleted at any level via the namespaced
//!   [`ir::Extra`] mechanism (RFC 7396 merge) and request hooks.
//! - **Pluggable HTTP client** — the library performs no IO by itself; any
//!   HTTP stack plugs in via [`http::HttpClient`]. A `reqwest`-based
//!   default is feature-gated (`reqwest`, on by default).
//!
//! With `default-features = false` the crate is a pure data layer: IR
//! types, format types, conversions — no IO, no tokio, no TLS.

pub mod client;
pub mod convert;
mod error;
mod format;
pub mod formats;
pub mod http;
pub mod ir;
pub mod models;
pub mod tokens;

pub use client::{
    CallOptions, Client, EndpointConfig, Limits, Override, ProviderConfig, StreamHandle,
};
pub use convert::{
    ConversionDirection, ConversionWarning, ConvertOptions, OrphanToolCalls, RequestHooks,
    WarningCode, WarningSeverity,
};
pub use error::{
    ApiErrorKind, BodyKind, ConversionError, Error, HookError, Result, retry_after_from_headers,
};
pub use format::{
    AnthropicAuthStyle, AnthropicOptions, ApiFormat, AuthScheme, BuildCtx, BuiltRequest, CallMode,
    EndpointUrl, FormatOptions, GoogleGenerateContentOptions, GoogleSafetySettings,
    OpenAiChatCompletionsOptions, ResponseMeta, StreamParser, build_url, encode_path_segment,
    extract_error_message, finalize_request, generic_api_error, ids, parse_data_url, to_data_url,
};
pub use ir::{
    Accumulator, BlockDelta, CacheHint, ContentBlock, Effort, Extra, FunctionTool, ImageSource,
    Message, OutputFormat, Reasoning, Request, Response, Role, RoundTripMeta, StopReason,
    StreamEvent, StreamItem, Tool, ToolChoice, ToolOutputBlock, Usage,
};
pub use models::Model;
pub use tokens::TokenCount;
