//! The dynamic format abstraction (§ 11) and endpoint URL construction
//! (§ 12).
//!
//! Each format module also exposes a typed layer (complete serde types plus
//! IR conversion functions) usable standalone; the client depends only on
//! `dyn ApiFormat`.

use std::collections::BTreeMap;

use bytes::Bytes;
use serde_json::Value;

use crate::convert::{ConvertOptions, ConversionWarning, RequestHooks};
use crate::error::{ConversionError, Error, Result};
use crate::http::SseEvent;
use crate::ir::{Request, Response, StreamEvent};
use crate::models::Model;
use crate::tokens::TokenCount;

/// Canonical format id constants.
pub mod ids {
    /// OpenAI Chat Completions (single implementation covering dialects
    /// such as DeepSeek `reasoning_content`).
    pub const OPENAI_CHAT_COMPLETIONS: &str = "openai_chat_completions";
    /// OpenAI Responses.
    pub const OPENAI_RESPONSES: &str = "openai_responses";
    /// Anthropic Messages.
    pub const ANTHROPIC_MESSAGES: &str = "anthropic_messages";
    /// Google `generateContent`.
    pub const GOOGLE_GENERATE_CONTENT: &str = "google_generate_content";
}

/// Unary vs streaming call — Google's URL differs by mode, and formats set
/// `stream: true` accordingly.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CallMode {
    /// A non-streaming call.
    #[default]
    Unary,
    /// A streaming (SSE) call.
    Streaming,
}

/// Where an endpoint lives (§ 12).
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum EndpointUrl {
    /// Base URL; the format appends its documented per-capability path.
    /// The base's own query string is preserved.
    Base(http::Uri),
    /// Complete request URL, used as-is after substituting the
    /// placeholders the format documents (`{model}`; `{method}` on
    /// Google). Without placeholders the string is used verbatim for every
    /// call mode.
    Full(String),
}

impl EndpointUrl {
    /// A base URL parsed from a string.
    pub fn base(uri: impl AsRef<str>) -> Result<Self, Error> {
        let uri: http::Uri = uri
            .as_ref()
            .parse()
            .map_err(|e| ConversionError::other(format!("invalid base URL: {e}")))?;
        Ok(Self::Base(uri))
    }

    /// A complete URL template.
    #[must_use]
    pub fn full(url: impl Into<String>) -> Self {
        Self::Full(url.into())
    }
}

/// How the transport should inject authentication for a request built by a
/// format: header name plus value prefix. The client combines this with the
/// configured [`crate::http::ApiKey`] into an [`crate::http::AuthHeader`] —
/// the key itself never appears in [`BuiltRequest`].
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthScheme {
    /// Header name.
    pub header: http::HeaderName,
    /// Value prefix, e.g. `"Bearer "`.
    pub prefix: Option<String>,
}

impl AuthScheme {
    /// `Authorization: Bearer <key>`.
    #[must_use]
    pub fn bearer() -> Self {
        Self { header: http::header::AUTHORIZATION, prefix: Some("Bearer ".to_owned()) }
    }

    /// `<name>: <key>` with no prefix.
    #[must_use]
    pub fn header(name: http::HeaderName) -> Self {
        Self { header: name, prefix: None }
    }
}

/// Everything a format needs to build a request (§ 11).
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct BuildCtx {
    /// Endpoint URL for the capability being called.
    pub url: EndpointUrl,
    /// Model name (plain string).
    pub model: String,
    /// Unary vs streaming.
    pub mode: CallMode,
    /// Conversion options.
    pub convert: ConvertOptions,
    /// Per-format knobs.
    pub format_options: FormatOptions,
    /// Request hooks (§ 5); run by the format after the strict gate.
    pub hooks: RequestHooks,
    /// Merged user query parameters (provider `extra_query` then per-call,
    /// later same-name keys already resolved by the client).
    pub extra_query: Vec<(String, String)>,
}

impl BuildCtx {
    /// A context with default options and no hooks or extra query.
    #[must_use]
    pub fn new(url: EndpointUrl, model: impl Into<String>, mode: CallMode) -> Self {
        Self {
            url,
            model: model.into(),
            mode,
            convert: ConvertOptions::default(),
            format_options: FormatOptions::default(),
            hooks: RequestHooks::default(),
            extra_query: Vec::new(),
        }
    }
}

/// A request ready for the transport (§ 11): JSON body + URL + headers +
/// auth spec + warnings.
#[non_exhaustive]
#[derive(Debug)]
pub struct BuiltRequest {
    /// HTTP method.
    pub method: http::Method,
    /// The final URL (path and query fully built; the client does no URL
    /// joining).
    pub url: http::Uri,
    /// Format-default headers (`content-type`, `anthropic-version`, beta
    /// flags, …). The client layers provider/endpoint/call headers on top.
    pub headers: http::HeaderMap,
    /// Serialized body (empty for GET).
    pub body: Bytes,
    /// How to inject the API key; `None` when the format sends none.
    pub auth: Option<AuthScheme>,
    /// Build-side conversion warnings.
    pub warnings: Vec<ConversionWarning>,
}

impl BuiltRequest {
    /// A POST request with a JSON body.
    #[must_use]
    pub fn post_json(url: http::Uri, body: &Value) -> Self {
        let mut headers = http::HeaderMap::new();
        headers.insert(http::header::CONTENT_TYPE, http::HeaderValue::from_static("application/json"));
        Self {
            method: http::Method::POST,
            url,
            headers,
            body: Bytes::from(serde_json::to_vec(body).expect("JSON value serializes")),
            auth: None,
            warnings: Vec::new(),
        }
    }

    /// A GET request with no body.
    #[must_use]
    pub fn get(url: http::Uri) -> Self {
        Self {
            method: http::Method::GET,
            url,
            headers: http::HeaderMap::new(),
            body: Bytes::new(),
            auth: None,
            warnings: Vec::new(),
        }
    }
}

/// Metadata the response parser cannot get from the body.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ResponseMeta {
    /// HTTP status (always 2xx — non-2xx goes to `parse_error`).
    pub status: u16,
    /// Response headers.
    pub headers: http::HeaderMap,
}

impl ResponseMeta {
    /// Metadata from status and headers.
    #[must_use]
    pub fn new(status: u16, headers: http::HeaderMap) -> Self {
        Self { status, headers }
    }
}

/// Per-format configuration knobs (§ 12).
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct FormatOptions {
    /// Anthropic Messages knobs.
    pub anthropic: AnthropicOptions,
    /// OpenAI Chat Completions knobs.
    pub openai_chat_completions: OpenAiChatCompletionsOptions,
    /// Free-form knobs for third-party formats, keyed by format id.
    pub custom: BTreeMap<String, Value>,
}

/// Which auth header the Anthropic format sends.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AnthropicAuthStyle {
    /// `x-api-key: <key>` (official).
    #[default]
    XApiKey,
    /// `Authorization: Bearer <key>` — needed by messages-format providers
    /// such as OpenRouter. Only switches the header; any beta flag a setup
    /// additionally requires is the user's responsibility.
    Bearer,
}

/// Anthropic Messages options (§ 12).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct AnthropicOptions {
    /// Auth header style.
    pub auth_style: AnthropicAuthStyle,
    /// `anthropic-version` header; `None` omits it (some dialect providers
    /// do not want it).
    pub version: Option<String>,
    /// Beta flags, joined into an `anthropic-beta` header.
    pub betas: Vec<String>,
    /// Converts mid-conversation system messages to `user` + warning for
    /// dialect providers without in-array system support (§ 7.1).
    pub downgrade_mid_system: bool,
    /// Merges adjacent same-role messages for dialect providers that do
    /// not auto-merge; signed blocks are merge barriers (§ 7.5).
    pub merge_consecutive_roles: bool,
    /// Fallback for the required `max_tokens` when the IR leaves
    /// `max_output_tokens` unset; without it that is a conversion error.
    pub default_max_tokens: Option<u32>,
}

impl Default for AnthropicOptions {
    fn default() -> Self {
        Self {
            auth_style: AnthropicAuthStyle::default(),
            version: Some("2023-06-01".to_owned()),
            betas: Vec::new(),
            downgrade_mid_system: false,
            merge_consecutive_roles: false,
            default_max_tokens: None,
        }
    }
}

/// OpenAI Chat Completions options (§ 12).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct OpenAiChatCompletionsOptions {
    /// Injects `stream_options: {include_usage: true}` on streaming calls;
    /// disable for dialects that reject it.
    pub inject_include_usage: bool,
}

impl Default for OpenAiChatCompletionsOptions {
    fn default() -> Self {
        Self { inject_include_usage: true }
    }
}

/// An object-safe, third-party-implementable provider format (§ 11).
pub trait ApiFormat: Send + Sync {
    /// The canonical format id (used as `extra` namespace and config key).
    fn id(&self) -> &str;

    /// Builds the chat request: conversion + `extra` merge + strict gate +
    /// hooks, per the § 6 pipeline.
    fn build_request(&self, req: &Request, ctx: &BuildCtx) -> Result<BuiltRequest>;

    /// Parses a 2xx chat response. Parse-side warnings go directly into
    /// `Response.warnings`; the client prepends request-build warnings
    /// afterwards.
    fn parse_response(&self, body: &[u8], meta: &ResponseMeta) -> Result<Response>;

    /// Classifies a non-2xx response, preserving the provider error shape.
    /// The default builds a generic `Error::Api` from status + raw body.
    fn parse_error(&self, status: u16, headers: &http::HeaderMap, body: &[u8]) -> Error {
        generic_api_error(status, headers, body)
    }

    /// One parser instance per stream: block-boundary inference is
    /// stateful.
    fn stream_parser(&self) -> Box<dyn StreamParser>;

    /// Parses a provider-format request into the IR (powers round-trip
    /// tests and future format-to-format conversion).
    fn parse_request(&self, body: &[u8]) -> Result<(Request, Vec<ConversionWarning>)>;

    /// Builds a model-listing request; `cursor` is the pagination cursor
    /// returned by the previous page.
    fn build_models_request(&self, ctx: &BuildCtx, cursor: Option<&str>) -> Result<BuiltRequest> {
        let _ = (ctx, cursor);
        Err(Error::NotSupported("model listing is not supported by this format"))
    }

    /// Parses one model-list page into models plus the next cursor.
    fn parse_models_response(&self, body: &[u8]) -> Result<(Vec<Model>, Option<String>)> {
        let _ = body;
        Err(Error::NotSupported("model listing is not supported by this format"))
    }

    /// Builds a token-count request from the prospective chat request
    /// (§ 12 pipeline: chat JSON first — extra, convert options and hooks
    /// all act on it — then the per-format count adapter reshapes it).
    fn build_count_tokens_request(&self, req: &Request, ctx: &BuildCtx) -> Result<BuiltRequest> {
        let _ = (req, ctx);
        Err(Error::NotSupported("token counting is not supported by this format"))
    }

    /// Parses a token-count response.
    fn parse_count_tokens_response(&self, body: &[u8]) -> Result<TokenCount> {
        let _ = body;
        Err(Error::NotSupported("token counting is not supported by this format"))
    }
}

/// Stateful per-stream event parser (§ 11).
pub trait StreamParser: Send {
    /// Parses one SSE event into unified events plus parse-side warnings
    /// (skipped extra candidates, lossy inference, …).
    fn parse(&mut self, event: &SseEvent)
    -> Result<(Vec<StreamEvent>, Vec<ConversionWarning>)>;

    /// Called exactly once when the byte stream ends: flushes held warnings
    /// and safely finalizable blocks, then validates terminal state — it
    /// never synthesizes `MessageStop` (parsers emit that themselves, § 9).
    /// A stream that showed no protocol terminator returns `Error::Parse`
    /// ("truncated stream") — a silent EOF must not pass a half response
    /// off as complete.
    fn finish(&mut self) -> Result<(Vec<StreamEvent>, Vec<ConversionWarning>)>;
}

/// Finalizes a built request body per the § 6 pipeline, shared by all
/// formats: marks warnings whose pointer the `extra` merge addressed as
/// `overridden`, applies the strict gate, then runs hooks (`on_message`
/// over the serialized message sequence, then `on_request`).
///
/// `messages` lists `(JSON pointer, role)` for each serialized
/// target-format message, in serialized order — after the splits, merges
/// and downgrades of § 7. Top-level system channels are not visited.
pub fn finalize_request(
    body: &mut Value,
    warnings: &mut [ConversionWarning],
    merge_log: &crate::ir::MergeLog,
    strict: bool,
    hooks: &RequestHooks,
    messages: &[(String, crate::ir::Role)],
) -> Result<()> {
    for w in warnings.iter_mut() {
        if w.direction == crate::convert::ConversionDirection::ToFormat {
            w.overridden = merge_log.overrides(&w.location);
        }
    }
    crate::convert::strict_gate(strict, warnings)?;
    if let Some(on_message) = &hooks.on_message {
        for (index, (pointer, role)) in messages.iter().enumerate() {
            // A pointer that no longer resolves (e.g. the request-level
            // `extra` rewrote the message array) is skipped defensively.
            if let Some(value) = body.pointer_mut(pointer) {
                on_message(index, role, value).map_err(Error::Hook)?;
            }
        }
    }
    if let Some(on_request) = &hooks.on_request {
        on_request(body).map_err(Error::Hook)?;
    }
    Ok(())
}

/// Builds a `data:` URL from a media type and base64 payload (OpenAI-family
/// image encoding).
#[must_use]
pub fn to_data_url(media_type: &str, data: &str) -> String {
    format!("data:{media_type};base64,{data}")
}

/// Splits a base64 `data:` URL into `(media_type, data)`. Returns `None`
/// for non-data URLs or non-base64 encodings.
#[must_use]
pub fn parse_data_url(url: &str) -> Option<(String, String)> {
    let rest = url.strip_prefix("data:")?;
    let (media_type, data) = rest.split_once(";base64,")?;
    Some((media_type.to_owned(), data.to_owned()))
}

/// The default non-2xx classifier: extracts a message from common error
/// body shapes, classifies by status, preserves the body.
#[must_use]
pub fn generic_api_error(status: u16, headers: &http::HeaderMap, body: &[u8]) -> Error {
    let parsed: Option<Value> = serde_json::from_slice(body).ok();
    let message = parsed
        .as_ref()
        .and_then(extract_error_message)
        .unwrap_or_else(|| String::from_utf8_lossy(body).chars().take(200).collect());
    Error::Api {
        status,
        kind: crate::error::ApiErrorKind::from_status(status),
        message,
        raw: Bytes::copy_from_slice(body),
        truncated: false,
        parsed,
        retry_after: crate::error::retry_after_from_headers(headers),
        headers: headers.clone(),
    }
}

/// Pulls a human-readable message out of the common provider error shapes:
/// `{"error": {"message": …}}`, `{"error": "…"}`, `{"message": …}`.
#[must_use]
pub fn extract_error_message(body: &Value) -> Option<String> {
    match body.get("error") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Object(o)) => o.get("message").and_then(Value::as_str).map(str::to_owned),
        _ => body.get("message").and_then(Value::as_str).map(str::to_owned),
    }
}

/// Percent-encodes a string as a single URL path segment (RFC 3986
/// unreserved characters kept).
#[must_use]
pub fn encode_path_segment(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn encode_query_component(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Builds the final request URL per the § 12 rules.
///
/// `capability_path` is the format's documented path template and may
/// contain `{model}` / `{method}`; for [`EndpointUrl::Full`] the whole URL
/// is the template and `capability_path` is ignored. `{model}` is
/// percent-encoded as a single path segment after stripping a leading
/// `models/` prefix; Google `tunedModels/…` resource names return
/// [`Error::NotSupported`]. `protected_query` keys are format-required —
/// a user-supplied query with the same key is a `ConversionError`;
/// `extra_query` entries replace same-name keys from the URL itself.
pub fn build_url(
    endpoint: &EndpointUrl,
    capability_path: &str,
    model: &str,
    method: Option<&str>,
    protected_query: &[(&str, &str)],
    extra_query: &[(String, String)],
) -> Result<http::Uri> {
    let substitute = |template: &str| -> Result<String> {
        let mut out = template.to_owned();
        if out.contains("{model}") {
            if model.starts_with("tunedModels/") {
                return Err(Error::NotSupported(
                    "Google tuned models (`tunedModels/…`) are out of scope for v1",
                ));
            }
            let stripped = model.strip_prefix("models/").unwrap_or(model);
            if stripped.is_empty() {
                return Err(ConversionError::missing("model name", "/").into());
            }
            out = out.replace("{model}", &encode_path_segment(stripped));
        }
        if out.contains("{method}") {
            let method = method.ok_or_else(|| {
                ConversionError::other("URL template contains {method} but no method was provided")
            })?;
            out = out.replace("{method}", method);
        }
        Ok(out)
    };

    // Assemble "scheme://authority/path" plus raw query pairs.
    let (base_part, mut query_pairs): (String, Vec<(String, String)>) = match endpoint {
        EndpointUrl::Base(uri) => {
            let scheme = uri
                .scheme_str()
                .ok_or_else(|| ConversionError::other("base URL must include a scheme"))?;
            let authority = uri
                .authority()
                .ok_or_else(|| ConversionError::other("base URL must include a host"))?;
            let base_path = uri.path().trim_end_matches('/');
            let path = substitute(capability_path)?;
            (
                format!("{scheme}://{authority}{base_path}/{path}"),
                parse_query(uri.query().unwrap_or("")),
            )
        }
        EndpointUrl::Full(template) => {
            let full = substitute(template)?;
            match full.split_once('?') {
                Some((base, query)) => (base.to_owned(), parse_query(query)),
                None => (full, Vec::new()),
            }
        }
    };

    // Format-required keys are protected: a user-supplied same-name key —
    // from the URL itself or from extra_query — is an error.
    for (key, value) in protected_query {
        if query_pairs.iter().any(|(k, _)| k == key)
            || extra_query.iter().any(|(k, _)| k == key)
        {
            return Err(ConversionError::ProtectedQueryKey { key: (*key).to_owned() }.into());
        }
        query_pairs.push(((*key).to_owned(), (*value).to_owned()));
    }

    // Later layers replace same-name keys (the caller passes provider-then-
    // per-call entries already in order).
    for (key, value) in extra_query {
        let encoded = (encode_query_component(key), encode_query_component(value));
        if let Some(existing) = query_pairs.iter_mut().find(|(k, _)| *k == encoded.0) {
            existing.1 = encoded.1;
        } else {
            query_pairs.push(encoded);
        }
    }

    let mut url = base_part;
    if !query_pairs.is_empty() {
        let qs: Vec<String> = query_pairs
            .iter()
            .map(|(k, v)| if v.is_empty() { k.clone() } else { format!("{k}={v}") })
            .collect();
        url.push('?');
        url.push_str(&qs.join("&"));
    }
    url.parse::<http::Uri>()
        .map_err(|e| ConversionError::other(format!("constructed URL is invalid: {e}")).into())
}

/// Splits a raw query string into raw (still-encoded) pairs.
fn parse_query(query: &str) -> Vec<(String, String)> {
    if query.is_empty() {
        return Vec::new();
    }
    query
        .split('&')
        .filter(|part| !part.is_empty())
        .map(|part| match part.split_once('=') {
            Some((k, v)) => (k.to_owned(), v.to_owned()),
            None => (part.to_owned(), String::new()),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_join_trims_and_appends() {
        let base = EndpointUrl::base("https://api.example.com/v1/").unwrap();
        let url = build_url(&base, "chat/completions", "m", None, &[], &[]).unwrap();
        assert_eq!(url.to_string(), "https://api.example.com/v1/chat/completions");
    }

    #[test]
    fn base_query_preserved_and_rebuilt() {
        let base = EndpointUrl::base("https://h/v1?key=abc").unwrap();
        let url = build_url(&base, "models", "m", None, &[], &[]).unwrap();
        assert_eq!(url.to_string(), "https://h/v1/models?key=abc");
    }

    #[test]
    fn model_substitution_encodes_and_strips_prefix() {
        let base = EndpointUrl::base("https://gl/v1beta").unwrap();
        let url = build_url(
            &base,
            "models/{model}:{method}",
            "models/gemini-2.5-flash",
            Some("generateContent"),
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(url.to_string(), "https://gl/v1beta/models/gemini-2.5-flash:generateContent");

        // Special characters are percent-encoded as one segment.
        let url2 = build_url(&base, "models/{model}", "a/b c", None, &[], &[]).unwrap();
        assert_eq!(url2.to_string(), "https://gl/v1beta/models/a%2Fb%20c");
    }

    #[test]
    fn tuned_models_not_supported() {
        let base = EndpointUrl::base("https://gl/v1beta").unwrap();
        let err = build_url(&base, "models/{model}", "tunedModels/x", None, &[], &[]).unwrap_err();
        assert!(matches!(err, Error::NotSupported(_)));
    }

    #[test]
    fn full_url_used_verbatim_with_substitution() {
        let full = EndpointUrl::full("https://h/api/custom/{model}:{method}?x=1");
        let url =
            build_url(&full, "ignored", "m", Some("streamGenerateContent"), &[], &[]).unwrap();
        assert_eq!(url.to_string(), "https://h/api/custom/m:streamGenerateContent?x=1");

        // Without placeholders the URL is used as-is.
        let plain = EndpointUrl::full("https://h/chat");
        let url2 = build_url(&plain, "ignored", "m", None, &[], &[]).unwrap();
        assert_eq!(url2.to_string(), "https://h/chat");
    }

    #[test]
    fn protected_query_conflicts_error() {
        let base = EndpointUrl::base("https://h/v1?alt=json").unwrap();
        let err =
            build_url(&base, "p", "m", None, &[("alt", "sse")], &[]).unwrap_err();
        assert!(matches!(
            err,
            Error::Conversion(ConversionError::ProtectedQueryKey { .. })
        ));

        let base2 = EndpointUrl::base("https://h/v1").unwrap();
        let err2 = build_url(
            &base2,
            "p",
            "m",
            None,
            &[("alt", "sse")],
            &[("alt".to_owned(), "json".to_owned())],
        )
        .unwrap_err();
        assert!(matches!(
            err2,
            Error::Conversion(ConversionError::ProtectedQueryKey { .. })
        ));

        // Protected key appended when no conflict.
        let ok = build_url(&base2, "p", "m", None, &[("alt", "sse")], &[]).unwrap();
        assert_eq!(ok.to_string(), "https://h/v1/p?alt=sse");
    }

    #[test]
    fn extra_query_replaces_same_key() {
        let base = EndpointUrl::base("https://h/v1?a=1&b=2").unwrap();
        let url = build_url(
            &base,
            "p",
            "m",
            None,
            &[],
            &[("a".to_owned(), "9".to_owned()), ("c".to_owned(), "3".to_owned())],
        )
        .unwrap();
        assert_eq!(url.to_string(), "https://h/v1/p?a=9&b=2&c=3");
    }

    #[test]
    fn data_url_round_trip() {
        let url = to_data_url("image/png", "AAAA");
        assert_eq!(url, "data:image/png;base64,AAAA");
        assert_eq!(parse_data_url(&url).unwrap(), ("image/png".to_owned(), "AAAA".to_owned()));
        assert!(parse_data_url("https://x/y.png").is_none());
        assert!(parse_data_url("data:text/plain,hello").is_none());
    }

    #[test]
    fn finalize_request_pipeline() {
        use crate::convert::WarningCode;
        use crate::ir::{MergeLog, Role, merge_patch};
        use serde_json::json;

        // Warning at /top_k gets overridden by an extra merge that set it.
        let mut body = json!({"messages": [{"role": "user", "content": "hi"}]});
        let mut log = MergeLog::new();
        let Value::Object(patch) = json!({"top_k": 40}) else { unreachable!() };
        merge_patch(&mut body, &patch, "", &mut log);

        let mut warnings = vec![ConversionWarning::to_format(
            WarningCode::SamplingParameterDropped,
            "f",
            "/top_k",
            "dropped",
        )];
        let hooks = RequestHooks::new()
            .with_on_message(|index, role, value| {
                assert_eq!(index, 0);
                assert_eq!(*role, Role::User);
                value["hooked"] = json!(true);
                Ok(())
            })
            .with_on_request(|value| {
                value["done"] = json!(true);
                Ok(())
            });
        finalize_request(
            &mut body,
            &mut warnings,
            &log,
            true, // strict passes: the only warning is cosmetic anyway
            &hooks,
            &[("/messages/0".to_owned(), Role::User)],
        )
        .unwrap();
        assert!(warnings[0].overridden);
        assert_eq!(body["messages"][0]["hooked"], json!(true));
        assert_eq!(body["done"], json!(true));
    }

    #[test]
    fn finalize_request_hook_error_aborts() {
        let mut body = serde_json::json!({});
        let hooks = RequestHooks::new()
            .with_on_request(|_| Err(crate::error::HookError::new("nope")));
        let err = finalize_request(
            &mut body,
            &mut [],
            &crate::ir::MergeLog::new(),
            false,
            &hooks,
            &[],
        )
        .unwrap_err();
        assert!(matches!(err, Error::Hook(_)));
    }

    #[test]
    fn generic_error_extracts_messages() {
        let headers = http::HeaderMap::new();
        let err = generic_api_error(
            429,
            &headers,
            br#"{"error": {"message": "slow down", "type": "rate_limit"}}"#,
        );
        match err {
            Error::Api { status, kind, message, parsed, .. } => {
                assert_eq!(status, 429);
                assert_eq!(kind, crate::error::ApiErrorKind::RateLimit);
                assert_eq!(message, "slow down");
                assert!(parsed.is_some());
            }
            other => panic!("unexpected: {other:?}"),
        }

        // Non-JSON body degrades to a text prefix.
        let err2 = generic_api_error(502, &headers, b"<html>Bad Gateway</html>");
        match err2 {
            Error::Api { message, parsed, .. } => {
                assert!(message.contains("Bad Gateway"));
                assert!(parsed.is_none());
            }
            other => panic!("unexpected: {other:?}"),
        }
    }
}
