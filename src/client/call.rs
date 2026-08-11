//! The high-level client (§ 12): option merging, header/auth layering,
//! dispatch and the per-capability call paths.

use std::collections::HashSet;
use std::sync::Arc;

use bytes::Bytes;

use crate::convert::{ConversionWarning, ConvertOptions, RequestHooks, WarningCode};
use crate::error::{BodyKind, Error, Result};
use crate::format::{
    ApiFormat, AuthScheme, BuildCtx, BuiltRequest, CallMode, EndpointUrl, ResponseMeta,
};
use crate::http::{ApiKey, AuthHeader, BodyStream, HttpClient, HttpError, SseParser};
use crate::ir::{Request, Response};
use crate::models::Model;
use crate::tokens::TokenCount;

use super::config::{CallOptions, EndpointConfig, Override, ProviderConfig};
use super::stream::StreamHandle;

/// The high-level client (§ 12). Cheap to clone; holds only the transport.
///
/// All calls take a [`ProviderConfig`] (format, URL, credentials, limits)
/// and, where applicable, per-call [`CallOptions`]. The client performs no
/// URL joining and never touches the request body — both are the format
/// layer's output; its own responsibilities are option merging, header
/// layering, auth resolution, size-cap enforcement and moving bytes.
#[derive(Clone)]
pub struct Client {
    http: Arc<dyn HttpClient>,
}

impl Client {
    /// A client over an owned transport.
    #[must_use]
    pub fn new(http: impl HttpClient + 'static) -> Self {
        Self {
            http: Arc::new(http),
        }
    }

    /// A client over a shared transport.
    #[must_use]
    pub fn from_arc(http: Arc<dyn HttpClient>) -> Self {
        Self { http }
    }

    /// Performs a non-streaming chat call.
    ///
    /// Pipeline: merge options → `format.build_request` (conversion,
    /// `extra` merge, strict gate, hooks) → header layering (format
    /// defaults < provider `extra_headers` < endpoint override < per-call)
    /// → auth resolution → send. A 2xx body is collected up to
    /// [`super::Limits::max_response_body`] (exceeding it fails with
    /// [`Error::BodyTooLarge`]) and parsed; request-build warnings are
    /// prepended to [`Response::warnings`]. A non-2xx body is collected up
    /// to [`super::Limits::max_error_body`] and classified by
    /// `format.parse_error`; when only a prefix was read (cap hit, or the
    /// connection failed mid-body) the resulting [`Error::Api`] carries
    /// `truncated: true` with the prefix as `raw` — status, headers and
    /// `retry_after` survive.
    pub async fn send(
        &self,
        provider: &ProviderConfig,
        request: &Request,
        opts: &CallOptions,
    ) -> Result<Response> {
        let eff = effective_options(provider, opts);
        let ctx = eff.build_ctx(provider.chat.url.clone(), CallMode::Unary, provider);
        let built = provider.chat.format.build_request(request, &ctx)?;
        let mut d = self
            .dispatch(provider, &provider.chat, &opts.extra_headers, built)
            .await?;
        if !is_success(d.status) {
            return Err(error_from_response(
                provider.chat.format.as_ref(),
                d.status,
                &d.headers,
                &mut d.body,
                provider.limits.max_error_body,
            )
            .await);
        }
        let body = read_success_body(
            &mut d.body,
            provider.limits.max_response_body,
            d.status,
            &d.headers,
        )
        .await?;
        let meta = ResponseMeta::new(d.status, d.headers);
        let mut response =
            provider
                .chat
                .format
                .parse_response_with(&body, &meta, &provider.format_options)?;
        let mut warnings = d.warnings;
        warnings.append(&mut response.warnings);
        response.warnings = warnings;
        Ok(response)
    }

    /// Performs a streaming chat call, returning a [`StreamHandle`] that
    /// yields unified [`crate::ir::StreamItem`]s.
    ///
    /// The request is built with [`CallMode::Streaming`]; a non-2xx
    /// response fails exactly like [`Client::send`]. On 2xx the handle
    /// pumps body bytes → SSE parser (capped by
    /// [`super::Limits::max_sse_event`]) → the format's stream parser; see
    /// [`StreamHandle`] for the item, warning and error semantics.
    pub async fn stream(
        &self,
        provider: &ProviderConfig,
        request: &Request,
        opts: &CallOptions,
    ) -> Result<StreamHandle> {
        let eff = effective_options(provider, opts);
        let ctx = eff.build_ctx(provider.chat.url.clone(), CallMode::Streaming, provider);
        let built = provider.chat.format.build_request(request, &ctx)?;
        let mut d = self
            .dispatch(provider, &provider.chat, &opts.extra_headers, built)
            .await?;
        if !is_success(d.status) {
            return Err(error_from_response(
                provider.chat.format.as_ref(),
                d.status,
                &d.headers,
                &mut d.body,
                provider.limits.max_error_body,
            )
            .await);
        }
        Ok(StreamHandle::new(
            ResponseMeta::new(d.status, d.headers),
            d.warnings,
            d.body,
            SseParser::new(provider.limits.max_sse_event),
            provider.chat.format.as_ref(),
            &provider.format_options,
            opts.include_raw,
        ))
    }

    /// Lists models, auto-paginating to exhaustion (§ 13).
    ///
    /// The endpoint is [`ProviderConfig::models`] when set; otherwise it
    /// derives from the chat endpoint (same format, URL, auth and headers)
    /// — but only when the chat URL is a base URL; with a full
    /// chat URL and no explicit config this returns
    /// [`Error::NotSupported`]. A pagination cursor equal to one already
    /// seen, or more than [`super::Limits::max_model_pages`] pages without
    /// an end, aborts with `Error::Parse` (malformed pagination) instead
    /// of looping forever.
    pub async fn list_models(&self, provider: &ProviderConfig) -> Result<Vec<Model>> {
        let endpoint = resolve_secondary_endpoint(
            provider,
            provider.models.as_ref(),
            "model listing needs an explicit `models` endpoint when the chat URL is a full URL",
        )?;
        let ctx = BuildCtx {
            url: endpoint.url.clone(),
            model: provider.model.clone(),
            mode: CallMode::Unary,
            convert: provider.convert.clone(),
            format_options: provider.format_options.clone(),
            hooks: provider.hooks.clone(),
            extra_query: provider.extra_query.clone(),
        };
        let no_call_headers = http::HeaderMap::new();
        let mut models = Vec::new();
        let mut cursor: Option<String> = None;
        let mut seen = HashSet::new();
        let mut pages = 0usize;
        loop {
            let built = endpoint
                .format
                .build_models_request(&ctx, cursor.as_deref())?;
            let mut d = self
                .dispatch(provider, &endpoint, &no_call_headers, built)
                .await?;
            if !is_success(d.status) {
                return Err(error_from_response(
                    endpoint.format.as_ref(),
                    d.status,
                    &d.headers,
                    &mut d.body,
                    provider.limits.max_error_body,
                )
                .await);
            }
            let body = read_success_body(
                &mut d.body,
                provider.limits.max_response_body,
                d.status,
                &d.headers,
            )
            .await?;
            let (page, next) = endpoint.format.parse_models_response(&body)?;
            models.extend(page);
            pages += 1;
            match next {
                None => break,
                Some(next) => {
                    // The runaway guard: a server minting a fresh cursor on
                    // every page never trips the seen-cursor check.
                    if pages >= provider.limits.max_model_pages {
                        return Err(Error::Parse {
                            message: format!(
                                "malformed pagination: exceeded {} pages without an end",
                                provider.limits.max_model_pages
                            ),
                            raw: Bytes::from(body),
                        });
                    }
                    if !seen.insert(next.clone()) {
                        return Err(Error::Parse {
                            message: format!(
                                "malformed pagination: cursor `{next}` was already seen"
                            ),
                            raw: Bytes::from(body),
                        });
                    }
                    cursor = Some(next);
                }
            }
        }
        Ok(models)
    }

    /// Counts the tokens of the prospective request via the provider's
    /// count endpoint (§ 13).
    ///
    /// The endpoint resolves like [`Client::list_models`]
    /// ([`ProviderConfig::count_tokens`], else derived from a base chat
    /// URL, else [`Error::NotSupported`]). Every body-affecting
    /// [`CallOptions`] field applies with the same merge result as
    /// [`Client::send`]; only `include_raw` is ignored. When the count
    /// endpoint's format is decoupled from the chat format (a different
    /// instance with a different id), one semantic
    /// [`WarningCode::CountTokensApproximate`] warning marks the result as
    /// an approximation. `TokenCount::warnings` carries the request-build
    /// warnings, then that approximation warning, then parse-side
    /// warnings. Under strict, a decoupled count format fails the call
    /// before any IO (§ 13); build-warning escalation happens inside the
    /// format's build as usual.
    pub async fn count_tokens(
        &self,
        provider: &ProviderConfig,
        request: &Request,
        opts: &CallOptions,
    ) -> Result<TokenCount> {
        let endpoint = resolve_secondary_endpoint(
            provider,
            provider.count_tokens.as_ref(),
            "token counting needs an explicit `count_tokens` endpoint when the chat URL is a full URL",
        )?;
        let decoupled = !Arc::ptr_eq(&endpoint.format, &provider.chat.format)
            && endpoint.format.id() != provider.chat.format.id();
        let eff = effective_options(provider, opts);
        let approx_warning = || {
            ConversionWarning::to_format(
                WarningCode::CountTokensApproximate,
                endpoint.format.id(),
                "",
                "token counting uses a format decoupled from the chat format; the result is an approximation",
            )
        };
        // § 13: under strict a decoupled count format fails the call — and
        // does so before any IO; a per-call `convert` with `strict: false`
        // opts into the approximate count.
        if decoupled {
            crate::convert::strict_gate(eff.convert.strict, &[approx_warning()])?;
        }
        let ctx = eff.build_ctx(endpoint.url.clone(), CallMode::Unary, provider);
        let built = endpoint.format.build_count_tokens_request(request, &ctx)?;
        let mut d = self
            .dispatch(provider, &endpoint, &opts.extra_headers, built)
            .await?;
        if !is_success(d.status) {
            return Err(error_from_response(
                endpoint.format.as_ref(),
                d.status,
                &d.headers,
                &mut d.body,
                provider.limits.max_error_body,
            )
            .await);
        }
        let body = read_success_body(
            &mut d.body,
            provider.limits.max_response_body,
            d.status,
            &d.headers,
        )
        .await?;
        let mut count = endpoint.format.parse_count_tokens_response(&body)?;
        let parse_warnings = std::mem::take(&mut count.warnings);
        let mut warnings = d.warnings;
        if decoupled {
            warnings.push(approx_warning());
        }
        warnings.extend(parse_warnings);
        count.warnings = warnings;
        Ok(count)
    }

    /// Layers headers, resolves auth, sends the request and returns the
    /// response head plus body stream.
    async fn dispatch(
        &self,
        provider: &ProviderConfig,
        endpoint: &EndpointConfig,
        call_headers: &http::HeaderMap,
        built: BuiltRequest,
    ) -> Result<Dispatched> {
        let warnings = built.warnings;
        let mut headers = built.headers;
        apply_header_layers(
            &mut headers,
            &provider.extra_headers,
            &endpoint.headers,
            call_headers,
        );
        let auth = resolve_auth(&endpoint.auth, built.auth.as_ref(), provider.auth.as_ref());
        let mut request = http::Request::builder()
            .method(built.method)
            .uri(built.url)
            .body(built.body)
            .expect("method and URI are pre-validated typed values");
        *request.headers_mut() = headers;
        let response = self
            .http
            .send(request, auth)
            .await
            .map_err(Error::Transport)?;
        let (parts, body) = response.into_parts();
        Ok(Dispatched {
            warnings,
            status: parts.status.as_u16(),
            headers: parts.headers,
            body,
        })
    }
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client").finish_non_exhaustive()
    }
}

/// A dispatched request: build warnings plus the response head and body.
struct Dispatched {
    warnings: Vec<ConversionWarning>,
    status: u16,
    headers: http::HeaderMap,
    body: BodyStream,
}

/// The per-call merge result (§ 12): `model`/`convert`/`hooks` replace
/// wholesale (per-call wins), `extra_query` stacks provider entries first
/// (same-name resolution happens in `build_url`, later entries win).
struct EffectiveOptions {
    model: String,
    convert: ConvertOptions,
    hooks: RequestHooks,
    extra_query: Vec<(String, String)>,
}

impl EffectiveOptions {
    fn build_ctx(self, url: EndpointUrl, mode: CallMode, provider: &ProviderConfig) -> BuildCtx {
        BuildCtx {
            url,
            model: self.model,
            mode,
            convert: self.convert,
            format_options: provider.format_options.clone(),
            hooks: self.hooks,
            extra_query: self.extra_query,
        }
    }
}

fn effective_options(provider: &ProviderConfig, opts: &CallOptions) -> EffectiveOptions {
    EffectiveOptions {
        model: opts.model.clone().unwrap_or_else(|| provider.model.clone()),
        convert: opts
            .convert
            .clone()
            .unwrap_or_else(|| provider.convert.clone()),
        hooks: opts.hooks.clone().unwrap_or_else(|| provider.hooks.clone()),
        extra_query: provider
            .extra_query
            .iter()
            .chain(&opts.extra_query)
            .cloned()
            .collect(),
    }
}

/// `true` for 2xx statuses.
fn is_success(status: u16) -> bool {
    (200..300).contains(&status)
}

/// Layers one header map onto another: same-name entries replace (all
/// values), other names add.
fn layer_headers(base: &mut http::HeaderMap, layer: &http::HeaderMap) {
    for name in layer.keys() {
        base.remove(name);
    }
    for (name, value) in layer {
        base.append(name, value.clone());
    }
}

/// Applies the § 12 header precedence on top of the format defaults:
/// provider `extra_headers` (skipped when the endpoint override is
/// `Disable`) < endpoint `Set` map < per-call headers. Format defaults are
/// never removable here; auth is injected by the transport after all
/// layers.
fn apply_header_layers(
    headers: &mut http::HeaderMap,
    provider_headers: &http::HeaderMap,
    endpoint_headers: &Override<http::HeaderMap>,
    call_headers: &http::HeaderMap,
) {
    if !matches!(endpoint_headers, Override::Disable) {
        layer_headers(headers, provider_headers);
    }
    if let Override::Set(map) = endpoint_headers {
        layer_headers(headers, map);
    }
    layer_headers(headers, call_headers);
}

/// Resolves the § 12 auth three-state: `Inherit` combines the built
/// request's auth scheme with the provider key (either absent → no auth);
/// `Set` uses the given header as-is; `Disable` sends none.
fn resolve_auth(
    endpoint_auth: &Override<AuthHeader>,
    scheme: Option<&AuthScheme>,
    key: Option<&ApiKey>,
) -> Option<AuthHeader> {
    match endpoint_auth {
        Override::Set(auth) => Some(auth.clone()),
        Override::Disable => None,
        Override::Inherit => match (scheme, key) {
            (Some(scheme), Some(key)) => Some(AuthHeader {
                name: scheme.header.clone(),
                prefix: scheme.prefix.clone(),
                value: key.clone(),
            }),
            _ => None,
        },
    }
}

/// Resolves a secondary (models / count-tokens) endpoint: the explicit
/// config when present, else derived from the chat endpoint when its URL
/// is a base URL, else `NotSupported`. A derived endpoint is the chat
/// endpoint itself (same format, URL, auth and headers) — only the
/// capability path differs, and that is appended by the format's build
/// functions from the base URL.
fn resolve_secondary_endpoint(
    provider: &ProviderConfig,
    explicit: Option<&EndpointConfig>,
    not_supported: &'static str,
) -> Result<EndpointConfig> {
    if let Some(endpoint) = explicit {
        return Ok(endpoint.clone());
    }
    match &provider.chat.url {
        EndpointUrl::Base(_) => Ok(provider.chat.clone()),
        EndpointUrl::Full(_) => Err(Error::NotSupported(not_supported)),
    }
}

/// How reading a capped body ended.
enum BodyEnd {
    /// The stream completed within the cap.
    Complete,
    /// The cap was exceeded; the buffer holds exactly `cap` bytes.
    CapHit,
    /// The transport failed mid-body; the buffer holds the prefix read.
    Failed(HttpError),
}

/// Collects a body stream into memory (a body strictly larger than `cap`
/// stops early). An oversized chunk is sliced before it is copied, so the
/// buffer never grows past `cap + 1` bytes regardless of chunk size — the
/// transport's own chunk allocation is outside this function's control.
async fn read_capped(body: &mut BodyStream, cap: usize) -> (Vec<u8>, BodyEnd) {
    let mut buf = Vec::new();
    loop {
        match std::future::poll_fn(|cx| body.as_mut().poll_next(cx)).await {
            Some(Ok(chunk)) => {
                // `cap + 1` bytes prove the body exceeds the cap while
                // copying no more than one byte past it (`buf.len() <= cap`
                // here, so `remaining >= 1`; saturating for
                // `cap == usize::MAX`).
                let remaining = cap.saturating_add(1) - buf.len();
                buf.extend_from_slice(&chunk[..remaining.min(chunk.len())]);
                if buf.len() > cap {
                    buf.truncate(cap);
                    return (buf, BodyEnd::CapHit);
                }
            }
            Some(Err(e)) => return (buf, BodyEnd::Failed(e)),
            None => return (buf, BodyEnd::Complete),
        }
    }
}

/// Reads a 2xx body under `limit`; a cap hit fails with `BodyTooLarge`
/// (status/headers/prefix attached), a transport failure with
/// `Error::Transport`.
async fn read_success_body(
    body: &mut BodyStream,
    limit: usize,
    status: u16,
    headers: &http::HeaderMap,
) -> Result<Vec<u8>> {
    let (buf, end) = read_capped(body, limit).await;
    match end {
        BodyEnd::Complete => Ok(buf),
        BodyEnd::CapHit => Err(Error::BodyTooLarge {
            kind: BodyKind::SuccessBody,
            limit,
            status: Some(status),
            headers: Some(headers.clone()),
            prefix: Bytes::from(buf),
        }),
        BodyEnd::Failed(e) => Err(Error::Transport(e)),
    }
}

/// Builds the unified error for a non-2xx response: the body is read up to
/// `max_error_body` (stopping at the cap) and classified by the format's
/// `parse_error`. When only a prefix was read — cap hit or a mid-body
/// transport failure — a resulting [`Error::Api`] is rewritten to
/// `truncated: true` with the prefix as `raw`; status, headers and
/// `retry_after` survive (§ 14: an error body is never downgraded to a
/// bare parse or transport error).
async fn error_from_response(
    format: &dyn ApiFormat,
    status: u16,
    headers: &http::HeaderMap,
    body: &mut BodyStream,
    max_error_body: usize,
) -> Error {
    let (buf, end) = read_capped(body, max_error_body).await;
    let mut error = format.parse_error(status, headers, &buf);
    if !matches!(end, BodyEnd::Complete)
        && let Error::Api { truncated, raw, .. } = &mut error
    {
        *truncated = true;
        *raw = Bytes::from(buf);
    }
    error
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    use crate::http::HttpErrorKind;

    fn name(s: &'static str) -> http::HeaderName {
        http::HeaderName::from_static(s)
    }

    fn value(s: &'static str) -> http::HeaderValue {
        http::HeaderValue::from_static(s)
    }

    #[test]
    fn layer_headers_replaces_same_name_and_adds() {
        let mut base = http::HeaderMap::new();
        base.append(name("x-a"), value("1"));
        base.append(name("x-a"), value("2"));
        base.append(name("x-b"), value("keep"));

        let mut layer = http::HeaderMap::new();
        layer.append(name("x-a"), value("9"));
        layer.append(name("x-c"), value("new"));
        layer.append(name("x-c"), value("new2"));

        layer_headers(&mut base, &layer);
        assert_eq!(base.get_all("x-a").iter().collect::<Vec<_>>(), vec!["9"]);
        assert_eq!(base.get("x-b").unwrap(), "keep");
        assert_eq!(
            base.get_all("x-c").iter().collect::<Vec<_>>(),
            vec!["new", "new2"]
        );
    }

    #[test]
    fn header_layer_precedence_and_disable() {
        let mut fmt = http::HeaderMap::new();
        fmt.insert(name("x-fmt"), value("f"));
        fmt.insert(name("x-a"), value("f"));

        let mut provider = http::HeaderMap::new();
        provider.insert(name("x-a"), value("p"));
        provider.insert(name("x-b"), value("p"));

        let mut endpoint = http::HeaderMap::new();
        endpoint.insert(name("x-b"), value("e"));
        endpoint.insert(name("x-c"), value("e"));

        let mut call = http::HeaderMap::new();
        call.insert(name("x-c"), value("c"));

        let mut headers = fmt.clone();
        apply_header_layers(
            &mut headers,
            &provider,
            &Override::Set(endpoint.clone()),
            &call,
        );
        assert_eq!(headers.get("x-fmt").unwrap(), "f");
        assert_eq!(headers.get("x-a").unwrap(), "p");
        assert_eq!(headers.get("x-b").unwrap(), "e");
        assert_eq!(headers.get("x-c").unwrap(), "c");

        // Disable drops the provider layer only; format defaults stay.
        let mut headers = fmt.clone();
        apply_header_layers(&mut headers, &provider, &Override::Disable, &call);
        assert_eq!(headers.get("x-a").unwrap(), "f");
        assert!(headers.get("x-b").is_none());
        assert_eq!(headers.get("x-c").unwrap(), "c");
    }

    #[test]
    fn resolve_auth_three_states() {
        let scheme = AuthScheme::bearer();
        let key = ApiKey::new("k");

        let inherited = resolve_auth(&Override::Inherit, Some(&scheme), Some(&key)).expect("auth");
        assert_eq!(inherited.name, http::header::AUTHORIZATION);
        assert_eq!(inherited.prefix.as_deref(), Some("Bearer "));
        assert_eq!(inherited.value.expose(), "k");

        assert!(resolve_auth(&Override::Inherit, Some(&scheme), None).is_none());
        assert!(resolve_auth(&Override::Inherit, None, Some(&key)).is_none());
        assert!(resolve_auth(&Override::Disable, Some(&scheme), Some(&key)).is_none());

        let set = AuthHeader {
            name: name("x-api-key"),
            prefix: None,
            value: ApiKey::new("other"),
        };
        let resolved = resolve_auth(&Override::Set(set), Some(&scheme), Some(&key)).expect("auth");
        assert_eq!(resolved.name, "x-api-key");
        assert_eq!(resolved.value.expose(), "other");
    }

    /// A test-only body stream yielding scripted chunks (no `futures-util`
    /// in the library, including its unit tests).
    struct Chunks(VecDeque<Result<Bytes, HttpError>>);

    impl futures_core::Stream for Chunks {
        type Item = Result<Bytes, HttpError>;

        fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Poll::Ready(self.0.pop_front())
        }
    }

    fn body_of(chunks: Vec<Result<Bytes, HttpError>>) -> BodyStream {
        Box::pin(Chunks(chunks.into()))
    }

    #[tokio::test]
    async fn read_capped_complete_cap_and_failure() {
        let mut body = body_of(vec![
            Ok(Bytes::from_static(b"hello ")),
            Ok(Bytes::from_static(b"world")),
        ]);
        let (buf, end) = read_capped(&mut body, 100).await;
        assert_eq!(buf, b"hello world");
        assert!(matches!(end, BodyEnd::Complete));

        // A body of exactly `cap` bytes is complete, not truncated.
        let mut body = body_of(vec![Ok(Bytes::from_static(b"12345678"))]);
        let (buf, end) = read_capped(&mut body, 8).await;
        assert_eq!(buf.len(), 8);
        assert!(matches!(end, BodyEnd::Complete));

        let mut body = body_of(vec![Ok(Bytes::from_static(b"0123456789"))]);
        let (buf, end) = read_capped(&mut body, 4).await;
        assert_eq!(buf, b"0123");
        assert!(matches!(end, BodyEnd::CapHit));

        let mut body = body_of(vec![
            Ok(Bytes::from_static(b"par")),
            Err(HttpError::new(HttpErrorKind::Body)),
        ]);
        let (buf, end) = read_capped(&mut body, 100).await;
        assert_eq!(buf, b"par");
        assert!(matches!(end, BodyEnd::Failed(_)));
    }

    #[tokio::test]
    async fn read_capped_slices_oversized_chunk() {
        // A single chunk far larger than the cap: only `cap` bytes are
        // kept, and the chunk was sliced to `cap + 1` bytes before the
        // copy — the buffer never held the whole chunk.
        let mut body = body_of(vec![Ok(Bytes::from(vec![b'x'; 1000]))]);
        let (buf, end) = read_capped(&mut body, 8).await;
        assert!(matches!(end, BodyEnd::CapHit));
        assert_eq!(buf, vec![b'x'; 8]);
        assert!(
            buf.capacity() < 1000,
            "an oversized chunk must not be copied whole (capacity {})",
            buf.capacity()
        );
    }

    #[tokio::test]
    async fn read_capped_cap_boundary_across_chunks() {
        // `cap + 1` reached exactly by a later chunk: CapHit with exactly
        // `cap` bytes kept.
        let mut body = body_of(vec![
            Ok(Bytes::from_static(b"12345")),
            Ok(Bytes::from_static(b"678")), // exactly `cap` so far
            Ok(Bytes::new()),               // an empty chunk is a no-op
            Ok(Bytes::from_static(b"9")),   // crosses to `cap + 1`
        ]);
        let (buf, end) = read_capped(&mut body, 8).await;
        assert_eq!(buf, b"12345678");
        assert!(matches!(end, BodyEnd::CapHit));

        // Crossing the cap mid-chunk keeps the prefix up to the cap.
        let mut body = body_of(vec![
            Ok(Bytes::from_static(b"12345")),
            Ok(Bytes::from_static(b"6789")),
        ]);
        let (buf, end) = read_capped(&mut body, 8).await;
        assert_eq!(buf, b"12345678");
        assert!(matches!(end, BodyEnd::CapHit));
    }
}
