//! OpenAI Responses format (`openai_responses`).
//!
//! Bidirectional conversion between the IR and the `POST /v1/responses`
//! wire format, the semantic-event SSE stream parser, model listing
//! (`GET /v1/models`, single page) and input-token counting
//! (`POST /v1/responses/input_tokens`).
//!
//! # Request mapping highlights (design § 4, § 7)
//!
//! - `Request.system` → top-level `instructions` (text blocks joined with
//!   `"\n\n"`; cache hints warn — put system content in the message array
//!   if breakpoints are needed). Block-level `extra` on `Request.system`
//!   has no channel here (`instructions` is a plain string) and drops with
//!   an `ExtraDropped` warning; use `Request.extra` to touch
//!   `instructions` itself.
//! - Messages → `input` items. User/system/developer messages become
//!   `message` items with `input_text` / `input_image` parts; assistant
//!   messages explode into top-level items (`Thinking` → `reasoning`,
//!   `Text` → `message` with `output_text` / `refusal` parts, `ToolCall`
//!   → `function_call`), order preserved; each `ToolResult` of a `Tool`
//!   message becomes one `function_call_output` item. Parsing groups
//!   consecutive assistant-side items back into one assistant message and
//!   gives every `function_call_output` its own `Tool` message. A
//!   `message` item with an unknown role parses to an `Opaque` block
//!   (`MalformedField` warning) and is re-emitted verbatim, in place.
//! - `max_output_tokens` / `temperature` / `top_p` / `metadata` /
//!   `prompt_cache_key` map directly; `stop_sequences` is dropped with a
//!   semantic warning; `top_k` / `seed` / penalties drop with cosmetic
//!   warnings.
//! - Reasoning (§ 4.7): `enabled: false` → `reasoning.effort: "none"`,
//!   `effort` → `reasoning.effort`, `include_thoughts: true` →
//!   `reasoning.summary: "auto"` (`false` omits `summary`). Other summary
//!   modes ride `Reasoning.extra` (e.g. `{"summary": "detailed"}`).
//! - Tools (§ 4.5): flat `{type: "function", name, description,
//!   parameters, strict}`; `parameters` and `strict` are
//!   required-but-nullable upstream and always serialize (`null` when
//!   unset). Hosted/built-in tools are `Tool::Opaque`.
//! - Cache hints (§ 4.8): input content parts get
//!   `prompt_cache_breakpoint: {"mode": "explicit"}` (hint TTLs warn —
//!   OpenAI TTLs are request-level `prompt_cache_options.ttl`, settable
//!   via `extra`). Hints on `ToolCall` / `ToolResult` blocks, assistant
//!   text and nested tool-output blocks drop with cosmetic warnings.
//!
//! # `extra` namespace conventions
//!
//! Parsing stores provider structure in each node's
//! `extra["openai_responses"]` namespace so same-format round-trips are
//! lossless:
//!
//! - `Thinking` blocks (from `reasoning` items): `Thinking.text` is the
//!   raw chain of thought — the `reasoning_text` parts of the item's
//!   `content` array joined with `"\n\n"` — falling back to the joined
//!   summary text; `encrypted_content` maps to `Thinking.signature`. The
//!   namespace keeps `id`, the original `summary` and `content` arrays and
//!   unknown item fields (`status`, …) for verbatim reconstruction; a
//!   native block without those arrays re-emits its text through
//!   `content` (the raw-CoT channel), as does `thinking_as_text`.
//! - `ToolCall` blocks (from `function_call` items): the item `id` plus
//!   unknown item fields; `call_id` maps to the block id.
//! - `Text` blocks of assistant messages use the reserved keys of
//!   [`text_block_reserved_key`]: `id` / `status` / `phase` are
//!   item-level fields, `item` nests unknown item-level fields, `refusal`
//!   marks refusal parts; every other key (`annotations`, `logprobs`, …)
//!   is part-level and merges back into the content part.
//! - A thinking block is *native* here iff its `extra` carries this
//!   namespace, or it has a `signature` and no namespace at all
//!   (optimistic replay). Foreign thinking drops with a warning unless
//!   `ConvertOptions.thinking_as_text` re-emits its text as summary text.
//!
//! # Notes
//!
//! - `reasoning.encrypted_content` is populated by default on
//!   `POST /v1/responses` per current official docs; for `store: false` /
//!   ZDR setups the `include` flag is still relevant. The library never
//!   injects it — request it via
//!   `Request.extra["openai_responses"] = {"include":
//!   ["reasoning.encrypted_content"]}` when needed.
//! - Per-image `detail` and other unmodeled part options ride block
//!   `extra`; `prompt_cache_options`, `store`, `include`, `truncation`,
//!   `service_tier`, … ride `Request.extra`.
//! - Canonicalizations (§ 1): string shorthands (`input`, message
//!   `content`, single-text `function_call_output.output`) parse into
//!   their array forms; absent required-but-nullable tool fields
//!   serialize as explicit `null`; `model` / `stream` are configuration,
//!   not IR data, and are consumed on parse; `is_error: Some(false)` on a
//!   tool result drops silently (equivalent to absent).

mod from_ir;
mod stream;
mod to_ir;
pub mod types;

use std::collections::BTreeSet;

use serde_json::Value;

use crate::convert::{ConversionWarning, ConvertOptions, WarningCode, strict_gate};
use crate::error::{ApiErrorKind, Error, Result, retry_after_from_headers};
use crate::format::{
    ApiFormat, AuthScheme, BuildCtx, BuiltRequest, CallMode, ResponseMeta, StreamParser, build_url,
    finalize_request, ids,
};
use crate::ir::{Request, Response, escape_pointer_token};
use crate::models::{Model, system_time_from_unix_seconds};
use crate::tokens::TokenCount;

pub use stream::ResponsesStreamParser;
pub use to_ir::{request_to_ir, response_to_ir};

/// This format's canonical id (`extra` namespace key).
pub(crate) const FORMAT: &str = ids::OPENAI_RESPONSES;

/// Reserved keys of the `openai_responses` namespace on assistant `Text`
/// blocks (see the module docs): they carry item-level data of the
/// containing `message` item, while all other keys are part-level.
pub mod text_block_reserved_key {
    /// The containing item's `id`; blocks sharing it re-group into one
    /// `message` item on serialization.
    pub const ID: &str = "id";
    /// The containing item's `status`.
    pub const STATUS: &str = "status";
    /// The containing item's `phase`.
    pub const PHASE: &str = "phase";
    /// Unknown item-level fields, nested as an object.
    pub const ITEM: &str = "item";
    /// Marks the block as a `refusal` part (§ 9).
    pub const REFUSAL: &str = "refusal";
}

/// Top-level request keys accepted by `POST /v1/responses/input_tokens`
/// (official reference; everything else must be stripped).
const COUNT_TOKENS_ACCEPTED: &[&str] = &[
    "model",
    "input",
    "instructions",
    "conversation",
    "previous_response_id",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "text",
    "reasoning",
    "personality",
    "truncation",
];

/// The OpenAI Responses format (typed layer entry point and
/// [`ApiFormat`] implementation).
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAiResponses;

/// Converts an IR request into the Responses request body (typed layer).
///
/// Runs the § 6 pipeline without hooks: node and request `extra` merges,
/// `overridden` marking and the strict gate. `model` is optional because
/// it comes from configuration, not the IR.
pub fn request_from_ir(
    req: &Request,
    model: Option<&str>,
    mode: CallMode,
    options: &ConvertOptions,
) -> Result<(Value, Vec<ConversionWarning>)> {
    let mut built = from_ir::build_body(req, model, mode, options)?;
    finalize_request(
        &mut built.body,
        &mut built.warnings,
        &built.merge_log,
        options.strict,
        &crate::convert::RequestHooks::default(),
        &built.messages,
    )?;
    Ok((built.body, built.warnings))
}

impl ApiFormat for OpenAiResponses {
    fn id(&self) -> &str {
        FORMAT
    }

    fn build_request(&self, req: &Request, ctx: &BuildCtx) -> Result<BuiltRequest> {
        let mut built = from_ir::build_body(req, Some(&ctx.model), ctx.mode, &ctx.convert)?;
        finalize_request(
            &mut built.body,
            &mut built.warnings,
            &built.merge_log,
            ctx.convert.strict,
            &ctx.hooks,
            &built.messages,
        )?;
        let url = build_url(
            &ctx.url,
            "responses",
            &ctx.model,
            None,
            &[],
            &ctx.extra_query,
        )?;
        let mut request = BuiltRequest::post_json(url, &built.body);
        request.auth = Some(AuthScheme::bearer());
        request.warnings = built.warnings;
        Ok(request)
    }

    fn parse_response(&self, body: &[u8], meta: &ResponseMeta) -> Result<Response> {
        response_to_ir(body, meta)
    }

    fn parse_error(&self, status: u16, headers: &http::HeaderMap, body: &[u8]) -> Error {
        let parsed: Option<Value> = serde_json::from_slice(body).ok();
        let detail = parsed
            .as_ref()
            .and_then(|v| serde_json::from_value::<types::ErrorBody>(v.clone()).ok())
            .and_then(|b| b.error);
        let kind = match detail.as_ref().and_then(|d| d.error_type.as_deref()) {
            Some("invalid_request_error") => ApiErrorKind::InvalidRequest,
            Some("authentication_error") => ApiErrorKind::Auth,
            Some("permission_error") => ApiErrorKind::PermissionDenied,
            Some("not_found_error") => ApiErrorKind::NotFound,
            Some("rate_limit_error" | "insufficient_quota" | "tokens") => ApiErrorKind::RateLimit,
            Some("overloaded_error") => ApiErrorKind::Overloaded,
            Some("server_error") => ApiErrorKind::ServerError,
            _ => ApiErrorKind::from_status(status),
        };
        let message = detail
            .as_ref()
            .and_then(|d| d.message.clone())
            .unwrap_or_else(|| String::from_utf8_lossy(body).chars().take(200).collect());
        Error::Api {
            status,
            kind,
            message,
            raw: bytes::Bytes::copy_from_slice(body),
            truncated: false,
            parsed,
            retry_after: retry_after_from_headers(headers),
            headers: headers.clone(),
        }
    }

    fn stream_parser(&self) -> Box<dyn StreamParser> {
        Box::new(ResponsesStreamParser::new())
    }

    fn parse_request(&self, body: &[u8]) -> Result<(Request, Vec<ConversionWarning>)> {
        request_to_ir(body)
    }

    fn build_models_request(&self, ctx: &BuildCtx, _cursor: Option<&str>) -> Result<BuiltRequest> {
        // OpenAI model listing is a single page; the cursor is ignored.
        let url = build_url(&ctx.url, "models", &ctx.model, None, &[], &ctx.extra_query)?;
        let mut request = BuiltRequest::get(url);
        request.auth = Some(AuthScheme::bearer());
        Ok(request)
    }

    fn parse_models_response(&self, body: &[u8]) -> Result<(Vec<Model>, Option<String>)> {
        let list: types::ModelList = serde_json::from_slice(body).map_err(|e| Error::Parse {
            message: format!("invalid model list: {e}"),
            raw: bytes::Bytes::copy_from_slice(body),
        })?;
        let mut models = Vec::with_capacity(list.data.len());
        for entry in list.data {
            // Entries without a string id cannot be addressed and are
            // skipped; `created` degrades to `None` silently (§ 13).
            let Some(id) = entry.get("id").and_then(Value::as_str) else {
                continue;
            };
            let mut model = Model::new(id, entry.clone());
            model.created = entry
                .get("created")
                .and_then(Value::as_i64)
                .and_then(system_time_from_unix_seconds);
            models.push(model);
        }
        Ok((models, None))
    }

    fn build_count_tokens_request(&self, req: &Request, ctx: &BuildCtx) -> Result<BuiltRequest> {
        // § 12/§ 13 pipeline: build the prospective chat JSON first (extra,
        // convert options and hooks all act on it, exactly as for `send`),
        // then reshape it for the count endpoint.
        let mut built = from_ir::build_body(req, Some(&ctx.model), CallMode::Unary, &ctx.convert)?;
        finalize_request(
            &mut built.body,
            &mut built.warnings,
            &built.merge_log,
            ctx.convert.strict,
            &ctx.hooks,
            &built.messages,
        )?;
        let generated: BTreeSet<String> = built.generated_keys;
        let mut warnings = built.warnings;
        if let Value::Object(map) = &mut built.body {
            let keys: Vec<String> = map.keys().cloned().collect();
            for key in keys {
                if COUNT_TOKENS_ACCEPTED.contains(&key.as_str()) {
                    continue;
                }
                map.remove(&key);
                // Generated keys the endpoint ignores by design drop
                // silently; anything else came from `extra`, hooks or a
                // dialect and makes the count inexact (§ 13).
                if !generated.contains(&key) {
                    warnings.push(ConversionWarning::to_format(
                        WarningCode::CountTokensFieldDropped,
                        FORMAT,
                        format!("/{}", escape_pointer_token(&key)),
                        format!(
                            "`{key}` is not accepted by `responses/input_tokens` and was dropped; \
                             the count may not be exact"
                        ),
                    ));
                }
            }
        }
        strict_gate(ctx.convert.strict, &warnings)?;
        let url = build_url(
            &ctx.url,
            "responses/input_tokens",
            &ctx.model,
            None,
            &[],
            &ctx.extra_query,
        )?;
        let mut request = BuiltRequest::post_json(url, &built.body);
        request.auth = Some(AuthScheme::bearer());
        request.warnings = warnings;
        Ok(request)
    }

    fn parse_count_tokens_response(&self, body: &[u8]) -> Result<TokenCount> {
        let raw: Value = serde_json::from_slice(body).map_err(|e| Error::Parse {
            message: format!("invalid input_tokens response: {e}"),
            raw: bytes::Bytes::copy_from_slice(body),
        })?;
        let wire: types::InputTokenCount =
            serde_json::from_value(raw.clone()).map_err(|e| Error::Parse {
                message: format!("invalid input_tokens response: {e}"),
                raw: bytes::Bytes::copy_from_slice(body),
            })?;
        let mut count = TokenCount::new(wire.input_tokens);
        count.raw = Some(raw);
        Ok(count)
    }
}
