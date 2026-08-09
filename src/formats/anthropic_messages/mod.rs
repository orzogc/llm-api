//! Anthropic Messages (`POST /v1/messages`).
//!
//! Format id: `anthropic_messages`. This module owns the complete wire
//! types ([`types`]) and the bidirectional IR conversions, exposed through
//! the [`AnthropicMessages`] unit struct's [`ApiFormat`] implementation —
//! usable standalone without the client.
//!
//! Mapping highlights (design § 4–§ 9, § 7.1–7.5 and the implementation
//! contract):
//!
//! - `max_tokens` is required upstream: an unset `Request.max_output_tokens`
//!   without `AnthropicOptions.default_max_tokens` is a conversion error in
//!   both modes.
//! - `Request.system` plus the leading run of `System` messages combine
//!   into the top-level `system` field (string form for exactly one plain
//!   text segment). In-array `system` messages parsed from a wire request
//!   carry a round-trip marker and keep their in-array placement when
//!   re-serialized; mid-conversation `System` messages stay in-array
//!   (`AnthropicOptions.downgrade_mid_system` converts them to `user`).
//!   `Developer` messages always downgrade to `user` with a warning.
//! - A top-level `system` array whose entries are all `text` parses into
//!   `Request.system`; with any non-text entry the whole array parses as a
//!   marker-less leading `System` message (non-text entries as `Opaque`
//!   blocks), which the combine rule hoists back on re-serialization.
//! - A wire message with an unknown role parses to a lone `Opaque` block
//!   holding the whole message; the build side re-emits it verbatim as a
//!   standalone wire message, excluded from every merge.
//! - Thinking: `enabled: true` → `{type: "adaptive"}`, `enabled: false` →
//!   `{type: "disabled"}`, `include_thoughts` → `thinking.display`;
//!   `budget_tokens` is unmodeled and round-trips via the reasoning
//!   `extra` (namespace `anthropic_messages`, merged into `/thinking`).
//! - `redacted_thinking` blocks parse to `Thinking` with the payload in
//!   `signature` and `{"redacted": true}` in the block's format namespace;
//!   the marker is consumed on re-serialization.
//! - Unmodeled content blocks (`document`, `server_tool_use`, tool-result
//!   variants, …) are preserved in place as `Opaque` nodes.

use serde_json::Value;

use crate::convert::{ConversionWarning, WarningCode, strict_gate};
use crate::error::{ApiErrorKind, ConversionError, Error, Result};
use crate::format::{
    AnthropicAuthStyle, AnthropicOptions, ApiFormat, AuthScheme, BuildCtx, BuiltRequest, CallMode,
    ResponseMeta, StreamParser, build_url, finalize_request, generic_api_error, ids,
};
use crate::ir::{Request, Response, RoundTripMeta, escape_pointer_token};
use crate::models::{Model, system_time_from_rfc3339};
use crate::tokens::TokenCount;

mod from_ir;
mod stream;
mod to_ir;
pub mod types;

/// Canonical id of this format.
pub(crate) const FORMAT_ID: &str = ids::ANTHROPIC_MESSAGES;

/// The Anthropic Messages format (`anthropic_messages`).
#[derive(Clone, Copy, Debug, Default)]
pub struct AnthropicMessages;

/// Fields the `count_tokens` endpoint accepts (official API reference);
/// everything else is filtered by the count adapter.
const COUNT_TOKENS_ACCEPTED: &[&str] = &[
    "model",
    "messages",
    "system",
    "tools",
    "tool_choice",
    "thinking",
    "cache_control",
    "context_management",
    "mcp_servers",
    "output_config",
    "output_format",
    "speed",
];

impl ApiFormat for AnthropicMessages {
    fn id(&self) -> &str {
        FORMAT_ID
    }

    fn build_request(&self, req: &Request, ctx: &BuildCtx) -> Result<BuiltRequest> {
        let streaming = ctx.mode == CallMode::Streaming;
        let mut chat = from_ir::build_chat_body(req, ctx, streaming)?;
        finalize_request(
            &mut chat.body,
            &mut chat.warnings,
            &chat.merge_log,
            ctx.convert.strict,
            &ctx.hooks,
            &chat.message_pointers,
        )?;
        let url = build_url(
            &ctx.url,
            "messages",
            &ctx.model,
            None,
            &[],
            &ctx.extra_query,
        )?;
        let mut built = BuiltRequest::post_json(url, &chat.body);
        apply_default_headers(&ctx.format_options.anthropic, &mut built.headers)?;
        built.auth = Some(auth_scheme(&ctx.format_options.anthropic));
        built.warnings = chat.warnings;
        Ok(built)
    }

    fn parse_response(&self, body: &[u8], meta: &ResponseMeta) -> Result<Response> {
        to_ir::parse_response_body(body, meta)
    }

    fn parse_error(&self, status: u16, headers: &http::HeaderMap, body: &[u8]) -> Error {
        let Ok(envelope) = serde_json::from_slice::<types::ErrorEnvelope>(body) else {
            return generic_api_error(status, headers, body);
        };
        let kind = match &envelope.error.kind {
            Some(t) => map_error_kind(t),
            None => ApiErrorKind::from_status(status),
        };
        Error::Api {
            status,
            kind,
            message: envelope
                .error
                .message
                .clone()
                .unwrap_or_else(|| format!("HTTP {status}")),
            raw: bytes::Bytes::copy_from_slice(body),
            truncated: false,
            parsed: serde_json::from_slice(body).ok(),
            retry_after: crate::error::retry_after_from_headers(headers),
            headers: headers.clone(),
        }
    }

    fn stream_parser(&self) -> Box<dyn StreamParser> {
        Box::new(stream::AnthropicStreamParser::new())
    }

    fn parse_request(&self, body: &[u8]) -> Result<(Request, Vec<ConversionWarning>)> {
        to_ir::parse_request_body(body)
    }

    fn build_models_request(&self, ctx: &BuildCtx, cursor: Option<&str>) -> Result<BuiltRequest> {
        let mut extra_query = ctx.extra_query.clone();
        if let Some(cursor) = cursor {
            extra_query.push(("after_id".to_owned(), cursor.to_owned()));
        }
        let url = build_url(&ctx.url, "models", &ctx.model, None, &[], &extra_query)?;
        let mut built = BuiltRequest::get(url);
        apply_default_headers(&ctx.format_options.anthropic, &mut built.headers)?;
        built.auth = Some(auth_scheme(&ctx.format_options.anthropic));
        Ok(built)
    }

    fn parse_models_response(&self, body: &[u8]) -> Result<(Vec<Model>, Option<String>)> {
        let page: types::ModelsPage = serde_json::from_slice(body).map_err(|e| Error::Parse {
            message: format!("invalid Anthropic models page: {e}"),
            raw: bytes::Bytes::copy_from_slice(body),
        })?;
        let mut models = Vec::new();
        for entry in &page.data {
            // Entries without an id are skipped defensively.
            let Some(id) = entry.get("id").and_then(Value::as_str) else {
                continue;
            };
            let mut model = Model::new(id, entry.clone());
            model.display_name = entry
                .get("display_name")
                .and_then(Value::as_str)
                .map(str::to_owned);
            model.created = entry
                .get("created_at")
                .and_then(Value::as_str)
                .and_then(system_time_from_rfc3339);
            models.push(model);
        }
        // has_more without a usable cursor would silently truncate the
        // paginated listing — malformed pagination is a hard parse error.
        let next = match page.has_more {
            Some(true) => match page.last_id {
                Some(id) if !id.is_empty() => Some(id),
                _ => {
                    return Err(Error::Parse {
                        message: "malformed pagination: has_more=true without last_id".to_owned(),
                        raw: bytes::Bytes::copy_from_slice(body),
                    });
                }
            },
            _ => None,
        };
        Ok((models, next))
    }

    fn build_count_tokens_request(&self, req: &Request, ctx: &BuildCtx) -> Result<BuiltRequest> {
        // Build the full chat body first — extra, convert options and hooks
        // all act on it exactly as for `send` (§ 13 pipeline), always in
        // unary mode.
        let mut chat = from_ir::build_chat_body(req, ctx, false)?;
        finalize_request(
            &mut chat.body,
            &mut chat.warnings,
            &chat.merge_log,
            ctx.convert.strict,
            &ctx.hooks,
            &chat.message_pointers,
        )?;
        // Adapt: drop keys the count endpoint does not accept. Dropping a
        // key the converter generated is silent; dropping injected data
        // makes the count inexact — semantic warning, error under strict.
        let mut adapter_warnings: Vec<ConversionWarning> = Vec::new();
        if let Some(obj) = chat.body.as_object_mut() {
            let keys: Vec<String> = obj.keys().cloned().collect();
            for key in keys {
                if COUNT_TOKENS_ACCEPTED.contains(&key.as_str()) {
                    continue;
                }
                obj.remove(&key);
                if !chat.generated_keys.contains(&key) {
                    adapter_warnings.push(ConversionWarning::to_format(
                        WarningCode::CountTokensFieldDropped,
                        FORMAT_ID,
                        format!("/{}", escape_pointer_token(&key)),
                        format!(
                            "`{key}` is not accepted by the count_tokens endpoint; the count \
                             may not be exact"
                        ),
                    ));
                }
            }
        }
        strict_gate(ctx.convert.strict, &adapter_warnings)?;
        let url = build_url(
            &ctx.url,
            "messages/count_tokens",
            &ctx.model,
            None,
            &[],
            &ctx.extra_query,
        )?;
        let mut built = BuiltRequest::post_json(url, &chat.body);
        apply_default_headers(&ctx.format_options.anthropic, &mut built.headers)?;
        built.auth = Some(auth_scheme(&ctx.format_options.anthropic));
        built.warnings = chat.warnings;
        built.warnings.extend(adapter_warnings);
        Ok(built)
    }

    fn parse_count_tokens_response(&self, body: &[u8]) -> Result<TokenCount> {
        let raw: Value = serde_json::from_slice(body).map_err(|e| Error::Parse {
            message: format!("invalid count_tokens response: {e}"),
            raw: bytes::Bytes::copy_from_slice(body),
        })?;
        let wire: types::CountTokensResponse =
            serde_json::from_value(raw.clone()).map_err(|e| Error::Parse {
                message: format!("invalid count_tokens response: {e}"),
                raw: bytes::Bytes::copy_from_slice(body),
            })?;
        let mut count = TokenCount::new(wire.input_tokens);
        count.raw = Some(raw);
        Ok(count)
    }
}

/// Adds the format-default headers (`anthropic-version` unless disabled,
/// `anthropic-beta` when betas are configured). `content-type` is set by
/// [`BuiltRequest::post_json`].
fn apply_default_headers(opts: &AnthropicOptions, headers: &mut http::HeaderMap) -> Result<()> {
    if let Some(version) = &opts.version {
        let value = http::HeaderValue::from_str(version).map_err(|_| {
            ConversionError::other(format!("invalid anthropic-version value: {version:?}"))
        })?;
        headers.insert(http::HeaderName::from_static("anthropic-version"), value);
    }
    if !opts.betas.is_empty() {
        let joined = opts.betas.join(",");
        let value = http::HeaderValue::from_str(&joined).map_err(|_| {
            ConversionError::other(format!("invalid anthropic-beta value: {joined:?}"))
        })?;
        headers.insert(http::HeaderName::from_static("anthropic-beta"), value);
    }
    Ok(())
}

/// The auth scheme selected by `AnthropicOptions.auth_style`.
fn auth_scheme(opts: &AnthropicOptions) -> AuthScheme {
    match opts.auth_style {
        AnthropicAuthStyle::XApiKey => {
            AuthScheme::header(http::HeaderName::from_static("x-api-key"))
        }
        AnthropicAuthStyle::Bearer => AuthScheme::bearer(),
    }
}

/// Maps an Anthropic `error.type` to the unified [`ApiErrorKind`].
pub(crate) fn map_error_kind(kind: &str) -> ApiErrorKind {
    match kind {
        "invalid_request_error" => ApiErrorKind::InvalidRequest,
        "authentication_error" => ApiErrorKind::Auth,
        "permission_error" => ApiErrorKind::PermissionDenied,
        "not_found_error" => ApiErrorKind::NotFound,
        "rate_limit_error" => ApiErrorKind::RateLimit,
        "overloaded_error" => ApiErrorKind::Overloaded,
        "api_error" => ApiErrorKind::ServerError,
        other => ApiErrorKind::Other(other.to_owned()),
    }
}

/// Round-trip marker attached to in-array `system` messages on parse, so
/// re-serialization keeps their placement instead of hoisting a leading run
/// into the top-level `system` field (§ 7.1). Written through the
/// versioned [`RoundTripMeta`] payload; a missing or malformed marker
/// degrades to canonical placement (hoisting).
pub(crate) fn in_array_system_meta() -> RoundTripMeta {
    let mut fields = serde_json::Map::new();
    fields.insert(IN_ARRAY_SYSTEM_KEY.to_owned(), Value::Bool(true));
    RoundTripMeta::v1(fields)
}

/// Reads the in-array system marker back, defensively.
pub(crate) fn has_in_array_system_meta(meta: Option<&RoundTripMeta>) -> bool {
    meta.and_then(|m| m.v1_field(IN_ARRAY_SYSTEM_KEY))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// v1 `RoundTripMeta` key marking an in-array `system` message.
const IN_ARRAY_SYSTEM_KEY: &str = "anthropic_in_array_system";
