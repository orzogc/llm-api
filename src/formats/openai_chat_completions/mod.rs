//! OpenAI Chat Completions format (`openai_chat_completions`).
//!
//! Bidirectional conversion between the IR and the
//! `POST /v1/chat/completions` wire format — a single implementation
//! covering dialects such as DeepSeek `reasoning_content` — plus the
//! chunk-stream SSE parser and model listing (`GET /v1/models`, single
//! page). Chat Completions has no token-counting endpoint, so the
//! [`ApiFormat`] count-token methods keep their `NotSupported` defaults.
//!
//! # Request mapping highlights (design § 4, § 7)
//!
//! - `Request.system` is inserted at the front of `messages` as a `system`
//!   message; in-array system/developer messages pass through natively
//!   (`ConvertOptions.downgrade_developer` rewrites developer → `user`).
//!   Parsing keeps leading `system` messages in-array — no hoisting.
//! - `max_output_tokens` → `max_completion_tokens` (legacy `max_tokens`
//!   parses into the same IR field); `stop_sequences` → `stop` (always the
//!   array form); `temperature` / `top_p` / `seed` / penalties /
//!   `metadata` / `prompt_cache_key` map verbatim; `top_k` drops with a
//!   cosmetic warning.
//! - Reasoning (§ 4.7): `effort` → `reasoning_effort` (the full tier set
//!   is accepted); `enabled: false` → `reasoning_effort: "none"`;
//!   `include_thoughts` has no channel (cosmetic warning); a non-empty
//!   `Reasoning.extra` namespace has no landing spot (`ExtraDropped`) —
//!   use `Request.extra` for top-level dialect knobs.
//! - Tools (§ 4.5): nested `{type: "function", function: {name,
//!   description, parameters, strict}}`; `parameters` is omitted when
//!   unset (officially an empty parameter list). `custom` tools and other
//!   kinds round-trip as `Tool::Opaque`. `ToolChoice::Tool` maps to
//!   `{type: "function", function: {name}}`.
//! - Assistant messages: native thinking → the configured
//!   `reasoning_field` (default `reasoning_content`, see below), text /
//!   refusal-marked blocks → `content`, `ToolCall` blocks →
//!   `tool_calls[]` (an id is required — `ConversionError` when absent).
//!   The wire message holds one field per channel, so only canonical block
//!   order (thinking → content → tool calls) survives; serializing an
//!   interleaved sequence warns `BlockOrderLost` (semantic). Assistant
//!   images drop with a semantic warning (parse-only channel, § 7.4).
//! - `Tool` messages: one `role: "tool"` wire message per `ToolResult`
//!   (`tool_call_id` required). Content is text-only: images drop with
//!   `ToolResultImageDropped`; a single plain non-empty text uses the
//!   string shorthand, several keep a text-part array, the empty list
//!   encodes as `content: ""` (§ 7.2). `ToolResult.name` maps to the
//!   dialect-accepted `name` field.
//! - Images (§ 4.3): `Url` → `image_url.url` verbatim; `Base64` → a
//!   `data:` URL; `FileId` has no image channel (the CC `file` part is a
//!   document channel) and drops with `ImageSourceUnsupported`. `detail`
//!   rides block `extra` as `{"image_url": {"detail": …}}`.
//! - Cache hints (§ 4.8): input content parts get
//!   `prompt_cache_breakpoint: {"mode": "explicit"}`; hint TTLs warn
//!   (`CacheTtlDropped`). Hints on `ToolCall` / `ToolResult` blocks,
//!   assistant text and nested tool-output blocks drop with cosmetic
//!   warnings. `Request.cache_key` → `prompt_cache_key`.
//! - Streaming builds set `stream: true` and inject
//!   `stream_options: {"include_usage": true}` per
//!   [`OpenAiChatCompletionsOptions::inject_include_usage`].
//!
//! # Thinking provenance (§ 4.4, implementation contract)
//!
//! The thinking channel here is a plaintext wire field, by default
//! `reasoning_content` (DeepSeek dialect). Dialects using another name
//! (`reasoning`, `thinking`, …) configure it via
//! [`OpenAiChatCompletionsOptions::reasoning_field`] (§ 12); the
//! configured name is the single authority on both sides — under a custom
//! name a wire `reasoning_content` is an ordinary unknown field. A
//! `Thinking` block is native iff its `extra` carries a non-empty
//! namespace for this format, or no non-empty format namespace at all —
//! plaintext thinking with no provenance is native to Chat Completions.
//! Native
//! blocks emit their text (several join with `"\n\n"` — the wire field is
//! a single string — adding a cosmetic `ThinkingBlocksJoined` warning);
//! a signature has no channel and drops with
//! `ThinkingSignatureDropped`. Foreign blocks drop with `ThinkingDropped`
//! unless `ConvertOptions.thinking_as_text` re-emits their text.
//! `ConvertOptions.fill_missing_thinking` genuinely helps here — the
//! filled plaintext block is native and reaches the wire.
//!
//! # `extra` namespace conventions
//!
//! Parsing stores provider structure in each node's
//! `extra["openai_chat_completions"]` namespace so same-format round-trips
//! reconstruct the wire shape faithfully:
//!
//! - `ToolCall` blocks use the reserved key of [`tool_call_reserved_key`]:
//!   `type` records a non-`function` call kind (`custom` calls map
//!   name/arguments to `custom.name` / `custom.input`; unknown kinds and
//!   non-string `type` values mirror the whole entry). Unknown entry
//!   fields sit at the top, `function` unknowns nest under `"function"`.
//! - Assistant `Text` blocks use the reserved keys of
//!   [`text_block_reserved_key`] (library-internal `__llm_api_`-prefixed
//!   markers); all other keys are part-level and merge into the serialized
//!   content part. `__llm_api_refusal` marks refusal content (the
//!   `refusal` message field or `refusal` parts) parsed into `Text` blocks
//!   with `{"__llm_api_refusal": true}` (§ 9); such blocks serialize back
//!   as `refusal` content parts. `__llm_api_message` nests message-level
//!   fields the streaming parser folded from unknown `delta` fields;
//!   serialization merges the nested object into the containing wire
//!   message — several holding blocks merge in block order (later keys
//!   win, RFC 7396), and the message's own `extra` merges last, staying
//!   authoritative.
//! - Unknown wire-message fields (`name`, `audio`, `annotations`, legacy
//!   `function_call`, …) ride the message extra; on `tool` messages they
//!   ride the `ToolResult` block extra (each wire message is its own IR
//!   `Tool` message). Thinking-block extras merge into the containing
//!   assistant message on serialization (the channel is a plain string
//!   field), so sibling dialect fields such as `reasoning_details` can be
//!   attached.
//! - Legacy `function`-role messages parse to an IR `Tool` message holding
//!   one whole-message `Opaque` block; unknown roles do the same under
//!   `User` with a warning. A message holding exactly one such Opaque node
//!   (an object with a `role`) re-emits verbatim.
//!
//! # Canonicalizations (§ 1)
//!
//! String content shorthands re-serialize as a string only for a single
//! plain, non-empty `Text` block (no cache hint, no namespace extra) —
//! everything else uses the part-array form; a string `stop` becomes a
//! one-element array; legacy `max_tokens` becomes `max_completion_tokens`;
//! `model`, `stream` and `stream_options` are configuration, not IR data,
//! and are consumed on parse (`stream_options` members warn
//! `StreamOptionsDropped` except a literal `include_usage: true`, the only
//! value the build side re-injects); explicit `null` fields
//! (e.g. assistant `content: null`) canonicalize to absent; a
//! message-level `refusal` field becomes a `refusal` content part; a
//! typeless `tool_calls[]` entry carrying only a `custom` payload counts
//! as a custom call and re-serializes with an explicit `type: "custom"`
//! (like the `function` default).
//! Streaming `logprobs` and chunk envelope fields (`system_fingerprint`,
//! `obfuscation`, …) are known-ignorable and not surfaced — use the
//! `include_raw` call option for the raw payloads.

mod from_ir;
mod stream;
mod to_ir;
pub mod types;

use serde_json::Value;

use crate::convert::{ConversionWarning, ConvertOptions};
use crate::error::{ApiErrorKind, Error, Result, retry_after_from_headers};
use crate::format::{
    ApiFormat, AuthScheme, BuildCtx, BuiltRequest, CallMode, FormatOptions,
    OpenAiChatCompletionsOptions, ResponseMeta, StreamParser, build_url, finalize_request, ids,
};
use crate::ir::{Request, Response};
use crate::models::{Model, system_time_from_unix_seconds};

pub use stream::ChatCompletionsStreamParser;
pub use to_ir::{request_to_ir, response_to_ir};

/// This format's canonical id (`extra` namespace key).
pub(crate) const FORMAT: &str = ids::OPENAI_CHAT_COMPLETIONS;

/// The default wire field of the plaintext thinking channel (§ 12,
/// [`OpenAiChatCompletionsOptions::reasoning_field`]).
pub(crate) const DEFAULT_REASONING_FIELD: &str = "reasoning_content";

/// Modeled assistant-message / stream-delta wire keys the configured
/// `reasoning_field` may not collide with: the typed message fields plus
/// the deprecated channels the wire reserves (`function_call`, `audio`).
const RESERVED_MESSAGE_KEYS: &[&str] = &[
    "role",
    "content",
    "refusal",
    "tool_calls",
    "tool_call_id",
    "name",
    "function_call",
    "audio",
];

/// Validates [`OpenAiChatCompletionsOptions::reasoning_field`]: the
/// thinking channel needs a dedicated, non-empty field name. Shared by
/// the build, parse and stream entry points so all three fail alike.
pub(crate) fn validate_reasoning_field(field: &str) -> Result<()> {
    if field.is_empty() {
        return Err(crate::error::ConversionError::other(
            "`reasoning_field` is empty; the Chat Completions thinking channel needs a wire \
             field name (default `reasoning_content`)",
        )
        .into());
    }
    if RESERVED_MESSAGE_KEYS.contains(&field) {
        return Err(crate::error::ConversionError::other(format!(
            "`reasoning_field` `{field}` collides with the modeled Chat Completions wire \
             field of the same name; the thinking channel needs a dedicated field \
             (default `reasoning_content`)"
        ))
        .into());
    }
    Ok(())
}

/// Reserved keys of the `openai_chat_completions` namespace on `ToolCall`
/// blocks (see the module docs).
pub mod tool_call_reserved_key {
    /// The `tool_calls[]` entry `type` when it is not `function`
    /// (`custom`, dialect kinds, non-string values kept verbatim);
    /// selects the serialized payload shape — only `"function"` /
    /// `"custom"` rebuild a payload object, every other value re-emits
    /// via the mirrored namespace.
    pub const TYPE: &str = "type";
}

/// Reserved keys of the `openai_chat_completions` namespace on assistant
/// `Text` blocks (see the module docs): they carry data of levels above
/// the content part, while all other keys are part-level. Both are
/// library-internal `__llm_api_`-prefixed markers, so a wire field that
/// happens to be named `refusal` or `message` mirrors into the namespace
/// as plain unknown data and replays verbatim.
pub mod text_block_reserved_key {
    /// Marks the block as a `refusal` part (§ 9).
    pub const REFUSAL: &str = crate::ir::REFUSAL_MARKER;
    /// Message-level fields, nested as an object. The streaming parser
    /// folds unknown `delta` fields here (the `delta` object is
    /// message-shaped); serialization merges the object into the
    /// containing wire message — several holding blocks merge in block
    /// order (later keys win, RFC 7396), and the message's own `extra`
    /// merges last, staying authoritative.
    pub const MESSAGE: &str = "__llm_api_message";
}

/// The OpenAI Chat Completions format (typed layer entry point and
/// [`ApiFormat`] implementation).
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAiChatCompletions;

/// Converts an IR request into the Chat Completions request body (typed
/// layer).
///
/// Runs the § 6 pipeline without hooks: node and request `extra` merges,
/// `overridden` marking and the strict gate. `model` is optional because
/// it comes from configuration, not the IR; `options` controls the
/// streaming `stream_options` injection.
pub fn request_from_ir(
    req: &Request,
    model: Option<&str>,
    mode: CallMode,
    convert: &ConvertOptions,
    options: &OpenAiChatCompletionsOptions,
) -> Result<(Value, Vec<ConversionWarning>)> {
    let mut built = from_ir::build_body(req, model, mode, convert, options)?;
    finalize_request(
        &mut built.body,
        &mut built.warnings,
        &built.merge_log,
        convert.strict,
        &crate::convert::RequestHooks::default(),
        &built.messages,
    )?;
    Ok((built.body, built.warnings))
}

impl ApiFormat for OpenAiChatCompletions {
    fn id(&self) -> &str {
        FORMAT
    }

    fn build_request(&self, req: &Request, ctx: &BuildCtx) -> Result<BuiltRequest> {
        let mut built = from_ir::build_body(
            req,
            Some(&ctx.model),
            ctx.mode,
            &ctx.convert,
            &ctx.format_options.openai_chat_completions,
        )?;
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
            "chat/completions",
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

    fn parse_response_with(
        &self,
        body: &[u8],
        meta: &ResponseMeta,
        options: &FormatOptions,
    ) -> Result<Response> {
        to_ir::response_to_ir_with(body, meta, &options.openai_chat_completions)
    }

    fn parse_error(&self, status: u16, headers: &http::HeaderMap, body: &[u8]) -> Error {
        let parsed: Option<Value> = serde_json::from_slice(body).ok();
        let detail = parsed
            .as_ref()
            .and_then(|v| serde_json::from_value::<types::ErrorBody>(v.clone()).ok())
            .and_then(|b| b.error);
        let code = detail
            .as_ref()
            .and_then(|d| d.code.as_ref())
            .and_then(Value::as_str);
        // Codes refine types: OpenAI reports auth failures as
        // `invalid_request_error` with `code: "invalid_api_key"`.
        let kind = match code {
            Some("invalid_api_key" | "invalid_organization" | "account_deactivated") => {
                ApiErrorKind::Auth
            }
            Some("model_not_found") => ApiErrorKind::NotFound,
            Some("insufficient_quota" | "rate_limit_exceeded") => ApiErrorKind::RateLimit,
            _ => match detail.as_ref().and_then(|d| d.error_type.as_deref()) {
                Some("invalid_request_error") => ApiErrorKind::InvalidRequest,
                Some("authentication_error") => ApiErrorKind::Auth,
                Some("permission_error") => ApiErrorKind::PermissionDenied,
                Some("not_found_error") => ApiErrorKind::NotFound,
                Some("rate_limit_error" | "insufficient_quota" | "tokens") => {
                    ApiErrorKind::RateLimit
                }
                Some("overloaded_error") => ApiErrorKind::Overloaded,
                Some("server_error") => ApiErrorKind::ServerError,
                _ => ApiErrorKind::from_status(status),
            },
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
        Box::new(ChatCompletionsStreamParser::new())
    }

    fn stream_parser_with(&self, options: &FormatOptions) -> Box<dyn StreamParser> {
        Box::new(ChatCompletionsStreamParser::with_options(
            &options.openai_chat_completions,
        ))
    }

    fn parse_request(&self, body: &[u8]) -> Result<(Request, Vec<ConversionWarning>)> {
        request_to_ir(body)
    }

    fn parse_request_with(
        &self,
        body: &[u8],
        options: &FormatOptions,
    ) -> Result<(Request, Vec<ConversionWarning>)> {
        to_ir::request_to_ir_with(body, &options.openai_chat_completions)
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
}
