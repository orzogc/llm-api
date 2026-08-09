//! Client configuration (§ 12): provider and endpoint configuration,
//! per-call options and size limits.

use std::fmt;
use std::sync::Arc;

use crate::convert::{ConvertOptions, RequestHooks};
use crate::error::Result;
use crate::format::{ApiFormat, EndpointUrl, FormatOptions};
use crate::http::{ApiKey, AuthHeader};

/// A three-state override for endpoint-level settings (§ 12).
///
/// `Inherit` uses the value from the surrounding layer, `Set` replaces it,
/// `Disable` turns the setting off for the endpoint.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Override<T> {
    /// Use the inherited value.
    Inherit,
    /// Replace the inherited value.
    Set(T),
    /// Turn the setting off: send no auth header / drop the provider
    /// `extra_headers` layer.
    Disable,
}

impl<T> Default for Override<T> {
    /// Defaults to [`Override::Inherit`].
    fn default() -> Self {
        Self::Inherit
    }
}

/// Where and how one capability (chat, model listing, token counting) is
/// served (§ 12).
#[non_exhaustive]
#[derive(Clone)]
pub struct EndpointConfig {
    /// The provider format serving this endpoint.
    pub format: Arc<dyn ApiFormat>,
    /// The endpoint URL (base to join, or a full template).
    pub url: EndpointUrl,
    /// Authentication override: `Inherit` combines the format's default
    /// auth scheme with the provider-level [`ProviderConfig::auth`] key;
    /// `Set` sends the given header as-is; `Disable` sends none.
    pub auth: Override<AuthHeader>,
    /// Header override: `Inherit` applies the provider
    /// [`ProviderConfig::extra_headers`]; `Set` layers the given map on top
    /// of them (same-name override, other names add); `Disable` drops the
    /// provider layer. Format-default headers are never removable here.
    pub headers: Override<http::HeaderMap>,
}

impl EndpointConfig {
    /// An endpoint with inherited auth and headers.
    #[must_use]
    pub fn new(format: Arc<dyn ApiFormat>, url: EndpointUrl) -> Self {
        Self {
            format,
            url,
            auth: Override::Inherit,
            headers: Override::Inherit,
        }
    }

    /// Sets the auth override.
    #[must_use]
    pub fn with_auth(mut self, auth: Override<AuthHeader>) -> Self {
        self.auth = auth;
        self
    }

    /// Sets the header override.
    #[must_use]
    pub fn with_headers(mut self, headers: Override<http::HeaderMap>) -> Self {
        self.headers = headers;
        self
    }
}

impl fmt::Debug for EndpointConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EndpointConfig")
            .field("format", &self.format.id())
            .field("url", &self.url)
            .field("auth", &self.auth)
            .field("headers", &self.headers)
            .finish()
    }
}

/// Size caps bounding memory against misbehaving peers (§ 12).
///
/// Plain byte counts, no magic values (`usize::MAX` ≈ unlimited). A payload
/// strictly larger than its cap fails: a 2xx body or an SSE event with
/// [`crate::error::Error::BodyTooLarge`], an error body with a full
/// [`crate::error::Error::Api`] carrying `truncated: true` and the read
/// prefix as `raw`.
///
/// The defaults are generous and **may be tuned in minor versions** — the
/// semver guarantee is the mechanism, not the numbers.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// Cap on a 2xx response body (decompressed bytes, as delivered by the
    /// transport). Default: 256 MiB.
    pub max_response_body: usize,
    /// Cap on a non-2xx response body; reading stops at the cap. Default:
    /// 8 MiB.
    pub max_error_body: usize,
    /// Cap on one complete logical SSE event (joined `data:` lines).
    /// Default: 64 MiB.
    pub max_sse_event: usize,
}

impl Limits {
    /// Default [`Limits::max_response_body`] (256 MiB).
    pub const DEFAULT_MAX_RESPONSE_BODY: usize = 256 * 1024 * 1024;
    /// Default [`Limits::max_error_body`] (8 MiB).
    pub const DEFAULT_MAX_ERROR_BODY: usize = 8 * 1024 * 1024;
    /// Default [`Limits::max_sse_event`] (64 MiB).
    pub const DEFAULT_MAX_SSE_EVENT: usize = 64 * 1024 * 1024;

    /// The default limits.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the 2xx body cap.
    #[must_use]
    pub fn with_max_response_body(mut self, bytes: usize) -> Self {
        self.max_response_body = bytes;
        self
    }

    /// Sets the error body cap.
    #[must_use]
    pub fn with_max_error_body(mut self, bytes: usize) -> Self {
        self.max_error_body = bytes;
        self
    }

    /// Sets the SSE event cap.
    #[must_use]
    pub fn with_max_sse_event(mut self, bytes: usize) -> Self {
        self.max_sse_event = bytes;
        self
    }
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_response_body: Self::DEFAULT_MAX_RESPONSE_BODY,
            max_error_body: Self::DEFAULT_MAX_ERROR_BODY,
            max_sse_event: Self::DEFAULT_MAX_SSE_EVENT,
        }
    }
}

/// One provider: default model, credentials, conversion behavior and the
/// endpoint set (§ 12).
///
/// Capabilities are decoupled: `models` / `count_tokens` may use their own
/// format, URL, auth and headers. When unset they derive from `chat` —
/// same format and URL, inherited auth and headers — but only when
/// [`ProviderConfig::chat`] uses a base URL; a full chat URL cannot be
/// reliably decomposed, so those capabilities then return
/// [`crate::error::Error::NotSupported`] unless configured explicitly.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ProviderConfig {
    /// Default model name; per-call override via [`CallOptions::model`].
    pub model: String,
    /// Provider-level API key; combined with the format's default auth
    /// scheme when an endpoint's auth is [`Override::Inherit`]. `None`
    /// sends no auth header.
    pub auth: Option<ApiKey>,
    /// Additional headers layered over the format defaults on every call
    /// (same-name override, other names add).
    pub extra_headers: http::HeaderMap,
    /// Additional query parameters appended to every call's URL (later
    /// same-name layers replace: per-call entries win over these).
    pub extra_query: Vec<(String, String)>,
    /// Conversion options; per-call replacement via
    /// [`CallOptions::convert`].
    pub convert: ConvertOptions,
    /// Per-format knobs.
    pub format_options: FormatOptions,
    /// Size caps (§ 12).
    pub limits: Limits,
    /// Request hooks; per-call replacement via [`CallOptions::hooks`].
    pub hooks: RequestHooks,
    /// The chat endpoint (required).
    pub chat: EndpointConfig,
    /// Model-listing endpoint; derived from `chat` when unset (see type
    /// docs).
    pub models: Option<EndpointConfig>,
    /// Token-counting endpoint; derived from `chat` when unset (see type
    /// docs).
    pub count_tokens: Option<EndpointConfig>,
}

impl ProviderConfig {
    /// The common case: one format, one base URL, one model. Fails when
    /// `base_url` is not a valid URI.
    pub fn new(format: Arc<dyn ApiFormat>, base_url: &str, model: &str) -> Result<Self> {
        Ok(Self::from_endpoint(
            EndpointConfig::new(format, EndpointUrl::base(base_url)?),
            model,
        ))
    }

    /// A provider from an explicit chat endpoint (e.g. one with a
    /// [`EndpointUrl::Full`] URL template).
    #[must_use]
    pub fn from_endpoint(chat: EndpointConfig, model: &str) -> Self {
        Self {
            model: model.to_owned(),
            auth: None,
            extra_headers: http::HeaderMap::new(),
            extra_query: Vec::new(),
            convert: ConvertOptions::default(),
            format_options: FormatOptions::default(),
            limits: Limits::default(),
            hooks: RequestHooks::default(),
            chat,
            models: None,
            count_tokens: None,
        }
    }

    /// Sets the provider API key.
    #[must_use]
    pub fn with_auth(mut self, key: impl Into<ApiKey>) -> Self {
        self.auth = Some(key.into());
        self
    }

    /// Sets the default model.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets the conversion options.
    #[must_use]
    pub fn with_convert(mut self, convert: ConvertOptions) -> Self {
        self.convert = convert;
        self
    }

    /// Sets the per-format knobs.
    #[must_use]
    pub fn with_format_options(mut self, options: FormatOptions) -> Self {
        self.format_options = options;
        self
    }

    /// Sets the size caps.
    #[must_use]
    pub fn with_limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }

    /// Sets the request hooks.
    #[must_use]
    pub fn with_hooks(mut self, hooks: RequestHooks) -> Self {
        self.hooks = hooks;
        self
    }

    /// Appends one extra header.
    #[must_use]
    pub fn with_header(mut self, name: http::HeaderName, value: http::HeaderValue) -> Self {
        self.extra_headers.append(name, value);
        self
    }

    /// Replaces the extra headers wholesale.
    #[must_use]
    pub fn with_extra_headers(mut self, headers: http::HeaderMap) -> Self {
        self.extra_headers = headers;
        self
    }

    /// Appends one extra query parameter.
    #[must_use]
    pub fn with_query(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_query.push((key.into(), value.into()));
        self
    }

    /// Replaces the extra query parameters wholesale.
    #[must_use]
    pub fn with_extra_query(mut self, query: Vec<(String, String)>) -> Self {
        self.extra_query = query;
        self
    }

    /// Sets an explicit model-listing endpoint.
    #[must_use]
    pub fn with_models_endpoint(mut self, endpoint: EndpointConfig) -> Self {
        self.models = Some(endpoint);
        self
    }

    /// Sets an explicit token-counting endpoint.
    #[must_use]
    pub fn with_count_tokens_endpoint(mut self, endpoint: EndpointConfig) -> Self {
        self.count_tokens = Some(endpoint);
        self
    }
}

/// Per-call options (§ 12), merged field-wise with the [`ProviderConfig`]:
/// `model`, `convert` and `hooks` replace the provider value wholesale when
/// set (per-call wins); `extra_headers` and `extra_query` stack on top of
/// the provider layers. Format, URL and auth are deliberately not per-call —
/// use another provider config for that.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct CallOptions {
    /// Model override; `None` uses [`ProviderConfig::model`].
    pub model: Option<String>,
    /// Conversion-options override. `Some` replaces the provider options
    /// wholesale; `None` inherits them.
    pub convert: Option<ConvertOptions>,
    /// Hook override. `Some` replaces the provider hooks wholesale
    /// (`Some(RequestHooks::new())` disables them); `None` inherits them.
    pub hooks: Option<RequestHooks>,
    /// Headers layered on top of the provider/endpoint layers (same-name
    /// override, other names add). Auth is injected after all header
    /// layers and cannot be overridden here.
    pub extra_headers: http::HeaderMap,
    /// Query parameters appended after the provider's; same-name keys
    /// replace the provider entry.
    pub extra_query: Vec<(String, String)>,
    /// Populates [`crate::ir::StreamItem::raw`] for every event of a
    /// streaming call (for `Unknown` events it is populated regardless).
    /// Ignored by [`super::Client::count_tokens`].
    pub include_raw: bool,
}

impl CallOptions {
    /// Default options (inherit everything from the provider).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the model override.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the conversion-options override.
    #[must_use]
    pub fn with_convert(mut self, convert: ConvertOptions) -> Self {
        self.convert = Some(convert);
        self
    }

    /// Sets the hook override.
    #[must_use]
    pub fn with_hooks(mut self, hooks: RequestHooks) -> Self {
        self.hooks = Some(hooks);
        self
    }

    /// Appends one extra header.
    #[must_use]
    pub fn with_header(mut self, name: http::HeaderName, value: http::HeaderValue) -> Self {
        self.extra_headers.append(name, value);
        self
    }

    /// Replaces the per-call extra headers wholesale.
    #[must_use]
    pub fn with_extra_headers(mut self, headers: http::HeaderMap) -> Self {
        self.extra_headers = headers;
        self
    }

    /// Appends one extra query parameter.
    #[must_use]
    pub fn with_query(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_query.push((key.into(), value.into()));
        self
    }

    /// Replaces the per-call extra query parameters wholesale.
    #[must_use]
    pub fn with_extra_query(mut self, query: Vec<(String, String)>) -> Self {
        self.extra_query = query;
        self
    }

    /// Sets [`CallOptions::include_raw`].
    #[must_use]
    pub fn with_include_raw(mut self, include_raw: bool) -> Self {
        self.include_raw = include_raw;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::ConversionWarning;
    use crate::error::Error;
    use crate::format::{BuildCtx, BuiltRequest, ResponseMeta, StreamParser};
    use crate::ir::{Request, Response};

    struct NullFormat;

    impl ApiFormat for NullFormat {
        fn id(&self) -> &str {
            "null"
        }

        fn build_request(&self, _: &Request, _: &BuildCtx) -> Result<BuiltRequest> {
            Err(Error::NotSupported("null"))
        }

        fn parse_response(&self, _: &[u8], _: &ResponseMeta) -> Result<Response> {
            Err(Error::NotSupported("null"))
        }

        fn stream_parser(&self) -> Box<dyn StreamParser> {
            unimplemented!("null format has no stream parser")
        }

        fn parse_request(&self, _: &[u8]) -> Result<(Request, Vec<ConversionWarning>)> {
            Err(Error::NotSupported("null"))
        }
    }

    #[test]
    fn limits_defaults() {
        let limits = Limits::default();
        assert_eq!(limits.max_response_body, 256 * 1024 * 1024);
        assert_eq!(limits.max_error_body, 8 * 1024 * 1024);
        assert_eq!(limits.max_sse_event, 64 * 1024 * 1024);
        let tuned = Limits::new()
            .with_max_sse_event(16)
            .with_max_error_body(8)
            .with_max_response_body(4);
        assert_eq!(
            (
                tuned.max_response_body,
                tuned.max_error_body,
                tuned.max_sse_event
            ),
            (4, 8, 16)
        );
    }

    #[test]
    fn override_defaults_to_inherit() {
        assert_eq!(Override::<u8>::default(), Override::Inherit);
    }

    #[test]
    fn call_options_default_inherits_everything() {
        let opts = CallOptions::default();
        assert!(opts.model.is_none());
        assert!(opts.convert.is_none());
        assert!(opts.hooks.is_none());
        assert!(opts.extra_headers.is_empty());
        assert!(opts.extra_query.is_empty());
        assert!(!opts.include_raw);
    }

    #[test]
    fn provider_config_new_rejects_invalid_url() {
        let err =
            ProviderConfig::new(Arc::new(NullFormat), "http://exa mple.com", "m").unwrap_err();
        assert!(matches!(err, Error::Conversion(_)));
    }

    #[test]
    fn provider_config_builders() {
        let provider = ProviderConfig::new(Arc::new(NullFormat), "http://h/v1", "m")
            .unwrap()
            .with_auth("sk-super-secret")
            .with_model("m2")
            .with_query("a", "1")
            .with_header(
                http::HeaderName::from_static("x-p"),
                http::HeaderValue::from_static("1"),
            )
            .with_limits(Limits::new().with_max_error_body(9))
            .with_models_endpoint(EndpointConfig::new(
                Arc::new(NullFormat),
                EndpointUrl::full("http://h/models"),
            ));
        assert_eq!(provider.model, "m2");
        assert_eq!(provider.auth.as_ref().unwrap().expose(), "sk-super-secret");
        assert_eq!(provider.extra_query, vec![("a".to_owned(), "1".to_owned())]);
        assert_eq!(provider.extra_headers.get("x-p").unwrap(), "1");
        assert_eq!(provider.limits.max_error_body, 9);
        assert!(provider.models.is_some());
        assert!(provider.count_tokens.is_none());
        // Debug does not leak the key.
        let debug = format!("{provider:?}");
        assert!(
            !debug.contains("sk-super-secret"),
            "debug output must not leak the API key: {debug}"
        );
    }
}
