//! Google `generateContent` format (Generative Language API, `v1beta`).
//!
//! Canonical format id: [`crate::ids::GOOGLE_GENERATE_CONTENT`]
//! (`google_generate_content`). The typed layer lives in [`types`] plus the
//! conversion entry points re-exported here; [`GoogleGenerateContent`] is the
//! dynamic [`ApiFormat`] implementation.
//!
//! Mapping highlights (see `docs/design.md` § 4–§ 9 for the full rules):
//!
//! - Chat path `models/{model}:generateContent` /
//!   `:streamGenerateContent?alt=sse`; auth header `x-goog-api-key`.
//! - `Request.system` and the leading run of system messages become
//!   `systemInstruction`; mid-conversation system and developer messages
//!   downgrade to `user` with a warning. A parsed `systemInstruction` maps to
//!   `Request.system` when text-only; with any out-of-schema non-text part it
//!   parses as a leading `System` message (text + same-format opaque blocks)
//!   instead, which the hoist rule serializes back — nothing is lost.
//! - Adjacent `User`/`Tool` messages merge into one `user` turn and adjacent
//!   `Assistant` messages into one `model` turn — Google requires role
//!   alternation; parsed mixed turns are split back apart carrying
//!   turn-group metadata. Wire turns are created lazily: a non-empty IR
//!   message whose blocks all drop (foreign opaque/thinking) is omitted
//!   entirely with an `EmptyMessageDropped` warning instead of emitting an
//!   empty `parts` turn, and same-side neighbours merge across the gap; an
//!   IR message with zero blocks still serializes as an empty turn
//!   (faithful replay of the wire's own empty-content form).
//! - A `tools` list whose entries were all dropped (foreign opaque tools)
//!   omits the `tools` key — the `OpaqueDropped` warnings disclose the
//!   drops; an explicitly empty IR tool list replays as `"tools": []`.
//! - `ToolResult.is_error: true` maps to the documented
//!   `functionResponse.response` failure key `{"error": …}` (and back);
//!   `is_error: false` canonicalizes to the plain `{"output": …}` encoding.
//! - Thought parts round-trip as `Thinking` blocks whose `extra` namespace
//!   records the wire `thought: true` marker; a `thoughtSignature` riding a
//!   `functionCall` part rides the `ToolCall` block's
//!   `extra["google_generate_content"]["thoughtSignature"]`.
//! - `thinkingBudget` is deliberately not modeled (design § 4.7); rewrite
//!   `generationConfig.thinkingConfig` via `extra` where needed.
//! - Usage: `input_tokens` = `promptTokenCount + toolUsePromptTokenCount`
//!   (live-verified: the tool-use term is excluded from `promptTokenCount`
//!   but included in `totalTokenCount`, so folding it in keeps
//!   input + output = total); a malformed `usageMetadata` degrades to no
//!   usage with a `MalformedField` warning instead of failing the billed
//!   response or chunk.
//! - Streaming: an error envelope (`data: {"error": {...}}`) on the 2xx
//!   channel raises [`Error::Api`] (`error.code` → status when plausible,
//!   gRPC `error.status` → kind); a chunk carrying no modeled signal at all
//!   surfaces as an `Unknown` event with one `MalformedField` warning per
//!   stream instead of being consumed silently.
//! - Known representational limits: `extra` set on a *nested* tool-output
//!   text block has no wire location on Google (the text is flattened into
//!   `response`) and drops with an `ExtraDropped` warning; explicit
//!   `thought: false` and content `role` defaults canonicalize to their
//!   absent forms; a `tools` entry combining `functionDeclarations` with
//!   other members (hosted tools or unknown siblings — assumed to be
//!   independent tool kinds, per the `Tool` object's one-field-per-kind
//!   design) splits on round-trip into a `functionDeclarations` entry plus
//!   one entry carrying the remaining members verbatim — an
//!   upstream-equivalent form (the official tool-combination examples list
//!   the combined tools as separate entries).

pub mod types;

mod from_ir;
mod stream;
mod to_ir;

pub use from_ir::request_from_ir;
pub use stream::GoogleStreamParser;
pub use to_ir::{request_to_ir, response_to_ir};

use bytes::Bytes;
use serde_json::{Value, json};

use crate::convert::ConversionWarning;
use crate::error::{ApiErrorKind, Error, Result};
use crate::format::{
    ApiFormat, AuthScheme, BuildCtx, BuiltRequest, CallMode, ResponseMeta, StreamParser, build_url,
    finalize_request, generic_api_error, ids,
};
use crate::ir::{Request, Response};
use crate::models::Model;
use crate::tokens::TokenCount;

/// The Google `generateContent` format (dynamic layer).
#[derive(Clone, Copy, Debug, Default)]
pub struct GoogleGenerateContent;

/// The default auth scheme: `x-goog-api-key: <key>` (header, not query, to
/// keep keys out of logs).
fn auth_scheme() -> AuthScheme {
    AuthScheme::header(http::HeaderName::from_static("x-goog-api-key"))
}

/// Maps a gRPC-style `error.status` code onto [`ApiErrorKind`]; unknown
/// codes fall back to HTTP-status classification.
fn api_error_kind(grpc_status: &str, http_status: u16) -> ApiErrorKind {
    match grpc_status {
        "INVALID_ARGUMENT" | "FAILED_PRECONDITION" | "OUT_OF_RANGE" => ApiErrorKind::InvalidRequest,
        "UNAUTHENTICATED" => ApiErrorKind::Auth,
        "PERMISSION_DENIED" => ApiErrorKind::PermissionDenied,
        "NOT_FOUND" => ApiErrorKind::NotFound,
        "RESOURCE_EXHAUSTED" => ApiErrorKind::RateLimit,
        "UNAVAILABLE" => ApiErrorKind::Overloaded,
        "INTERNAL" => ApiErrorKind::ServerError,
        _ => ApiErrorKind::from_status(http_status),
    }
}

/// Refines an already-classified [`Error::Api`] in place: when the parsed
/// body carries a gRPC `error.status` string, it overrides the plain
/// HTTP-status classification.
fn refine_api_error_kind(error: &mut Error, http_status: u16) {
    if let Error::Api {
        kind,
        parsed: Some(parsed),
        ..
    } = error
        && let Some(grpc) = parsed
            .get("error")
            .and_then(|e| e.get("status"))
            .and_then(Value::as_str)
    {
        *kind = api_error_kind(grpc, http_status);
    }
}

/// Classifies an error envelope (`data: {"error": {...}}`) received on the
/// 2xx stream channel, reusing the non-2xx pipeline: `error.code` supplies
/// the status when it is a plausible HTTP status (100..=599; otherwise the
/// transport's 200 is kept), message extraction and body preservation come
/// from [`generic_api_error`], and the gRPC `error.status` drives the kind
/// exactly as in [`ApiFormat::parse_error`].
pub(crate) fn stream_error(data: &str) -> Error {
    let status = serde_json::from_str::<Value>(data)
        .ok()
        .as_ref()
        .and_then(|v| v.get("error"))
        .and_then(|e| e.get("code"))
        .and_then(Value::as_u64)
        .and_then(|c| u16::try_from(c).ok())
        .filter(|c| (100..=599).contains(c))
        .unwrap_or(200);
    let mut error = generic_api_error(status, &http::HeaderMap::new(), data.as_bytes());
    refine_api_error_kind(&mut error, status);
    error
}

impl ApiFormat for GoogleGenerateContent {
    fn id(&self) -> &str {
        ids::GOOGLE_GENERATE_CONTENT
    }

    fn build_request(&self, req: &Request, ctx: &BuildCtx) -> Result<BuiltRequest> {
        let from_ir::BuiltBody {
            mut body,
            mut warnings,
            merge_log,
            message_pointers,
        } = from_ir::build_body(req, &ctx.convert)?;
        finalize_request(
            &mut body,
            &mut warnings,
            &merge_log,
            ctx.convert.strict,
            &ctx.hooks,
            &message_pointers,
        )?;
        let (method, protected): (&str, &[(&str, &str)]) = match ctx.mode {
            CallMode::Streaming => ("streamGenerateContent", &[("alt", "sse")]),
            _ => ("generateContent", &[]),
        };
        let url = build_url(
            &ctx.url,
            "models/{model}:{method}",
            &ctx.model,
            Some(method),
            protected,
            &ctx.extra_query,
        )?;
        let mut built = BuiltRequest::post_json(url, &body);
        built.auth = Some(auth_scheme());
        built.warnings = warnings;
        Ok(built)
    }

    fn parse_response(&self, body: &[u8], meta: &ResponseMeta) -> Result<Response> {
        response_to_ir(body, meta)
    }

    fn parse_error(&self, status: u16, headers: &http::HeaderMap, body: &[u8]) -> Error {
        let mut error = generic_api_error(status, headers, body);
        refine_api_error_kind(&mut error, status);
        error
    }

    fn stream_parser(&self) -> Box<dyn StreamParser> {
        Box::new(GoogleStreamParser::new())
    }

    fn parse_request(&self, body: &[u8]) -> Result<(Request, Vec<ConversionWarning>)> {
        request_to_ir(body)
    }

    fn build_models_request(&self, ctx: &BuildCtx, cursor: Option<&str>) -> Result<BuiltRequest> {
        // pageSize/pageToken are pagination-mechanism keys owned by the
        // library (§ 13); user query keys with the same names conflict.
        let mut protected: Vec<(&str, &str)> = vec![("pageSize", "1000")];
        if let Some(cursor) = cursor {
            protected.push(("pageToken", cursor));
        }
        let url = build_url(
            &ctx.url,
            "models",
            &ctx.model,
            None,
            &protected,
            &ctx.extra_query,
        )?;
        let mut built = BuiltRequest::get(url);
        built.auth = Some(auth_scheme());
        Ok(built)
    }

    fn parse_models_response(&self, body: &[u8]) -> Result<(Vec<Model>, Option<String>)> {
        let wire: types::ListModelsResponse =
            serde_json::from_slice(body).map_err(|e| Error::Parse {
                message: format!("invalid Google models response: {e}"),
                raw: Bytes::copy_from_slice(body),
            })?;
        let mut models = Vec::new();
        for entry in wire.models {
            let Some(name) = entry.get("name").and_then(Value::as_str) else {
                // An entry without a resource name cannot be addressed;
                // skip it defensively.
                continue;
            };
            let id = name.strip_prefix("models/").unwrap_or(name).to_owned();
            let display_name = entry
                .get("displayName")
                .and_then(Value::as_str)
                .map(str::to_owned);
            let mut model = Model::new(id, entry.clone());
            model.display_name = display_name;
            models.push(model);
        }
        let next = wire.next_page_token.filter(|t| !t.is_empty());
        Ok((models, next))
    }

    fn build_count_tokens_request(&self, req: &Request, ctx: &BuildCtx) -> Result<BuiltRequest> {
        // § 13 pipeline: the prospective chat body is built first — extra,
        // convert options and hooks all act on it exactly as for `send` —
        // then wrapped. The count endpoint accepts the entire
        // `GenerateContentRequest`, so nothing is dropped and the count is
        // exact for whatever the chat call would have sent.
        let from_ir::BuiltBody {
            mut body,
            mut warnings,
            merge_log,
            message_pointers,
        } = from_ir::build_body(req, &ctx.convert)?;
        finalize_request(
            &mut body,
            &mut warnings,
            &merge_log,
            ctx.convert.strict,
            &ctx.hooks,
            &message_pointers,
        )?;
        let url = build_url(
            &ctx.url,
            "models/{model}:countTokens",
            &ctx.model,
            None,
            &[],
            &ctx.extra_query,
        )?;
        // The nested GenerateContentRequest requires an explicit model
        // resource name (the URL path does not reach it).
        let stripped = ctx.model.strip_prefix("models/").unwrap_or(&ctx.model);
        if let Some(obj) = body.as_object_mut() {
            obj.insert("model".to_owned(), json!(format!("models/{stripped}")));
        }
        let wrapped = json!({ "generateContentRequest": body });
        let mut built = BuiltRequest::post_json(url, &wrapped);
        built.auth = Some(auth_scheme());
        built.warnings = warnings;
        Ok(built)
    }

    fn parse_count_tokens_response(&self, body: &[u8]) -> Result<TokenCount> {
        let raw: Value = serde_json::from_slice(body).map_err(|e| Error::Parse {
            message: format!("invalid Google countTokens response: {e}"),
            raw: Bytes::copy_from_slice(body),
        })?;
        let wire: types::CountTokensResponse =
            serde_json::from_value(raw.clone()).map_err(|e| Error::Parse {
                message: format!("invalid Google countTokens response: {e}"),
                raw: Bytes::copy_from_slice(body),
            })?;
        let total = match wire.total_tokens {
            // proto3 JSON omits zero-valued fields: an absent totalTokens
            // is the wire encoding of zero, not malformed data (same
            // reading as § 8 usage counts).
            None => 0,
            Some(t) => u64::try_from(t).map_err(|_| Error::Parse {
                message: format!("Google countTokens response has a negative totalTokens: {t}"),
                raw: Bytes::copy_from_slice(body),
            })?,
        };
        let mut count = TokenCount::new(total);
        count.raw = Some(raw);
        Ok(count)
    }
}
