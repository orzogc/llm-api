//! The high-level client and its configuration (§ 12).
//!
//! [`Client`] wraps any [`crate::http::HttpClient`] and drives the § 6/§ 11
//! pipeline end to end: option merging, request building via the endpoint's
//! [`crate::format::ApiFormat`], header layering, auth resolution, size-cap
//! enforcement, response/error parsing and SSE streaming.
//!
//! Configuration is two-level: a [`ProviderConfig`] describes one provider
//! (default model, key, conversion options, limits and one
//! [`EndpointConfig`] per capability, with [`Override`]-based endpoint
//! customization), while [`CallOptions`] merges per-call adjustments on top
//! (`model`/`convert`/`hooks` replace wholesale, headers and query
//! parameters stack). Streaming calls yield a [`StreamHandle`].

mod call;
mod config;
mod stream;

pub use call::Client;
pub use config::{CallOptions, EndpointConfig, Limits, Override, ProviderConfig};
pub use stream::StreamHandle;

// Re-exported for convenience: endpoint URLs are part of endpoint
// configuration even though the type lives in the format layer.
pub use crate::format::EndpointUrl;
