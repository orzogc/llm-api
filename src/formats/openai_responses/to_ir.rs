//! OpenAI Responses → IR parsing (request → IR and response → IR).

use bytes::Bytes;
use serde_json::{Map, Value, json};

use crate::convert::{ConversionWarning, WarningCode};
use crate::error::{ApiErrorKind, Error, Result, retry_after_from_headers};
use crate::format::{ResponseMeta, parse_data_url};
use crate::ir::{
    CacheHint, ContentBlock, Effort, Extra, FunctionTool, ImageSource, Message, OutputFormat,
    Reasoning, Request, Response, Role, StopReason, Tool, ToolChoice, ToolOutputBlock, Usage,
    normalize_stop_reason,
};

use super::FORMAT;
use super::types;

/// Parse-side warning shorthand.
fn warn(
    code: WarningCode,
    location: impl Into<String>,
    message: impl Into<String>,
) -> ConversionWarning {
    ConversionWarning::from_format(code, FORMAT, location, message)
}

fn parse_failure(what: &str, error: impl std::fmt::Display, body: &[u8]) -> Error {
    Error::Parse {
        message: format!("invalid Responses {what}: {error}"),
        raw: Bytes::copy_from_slice(body),
    }
}

/// Parses a Responses request body into the IR (§ 11 `parse_request`).
///
/// Canonicalizations (§ 1): the string forms of `input` and message
/// `content` become item/part arrays, `model` and `stream` are dropped
/// (they are configuration, not IR data), and unmapped-but-valid fields
/// (`instructions` echoed as an array, unmodeled `tool_choice` shapes,
/// `text.verbosity`, …) move into the request `extra` namespace so they
/// round-trip verbatim.
pub fn request_to_ir(body: &[u8]) -> Result<(Request, Vec<ConversionWarning>)> {
    let wire: types::Request =
        serde_json::from_slice(body).map_err(|e| parse_failure("request", e, body))?;
    let mut warnings = Vec::new();
    let mut ns: Map<String, Value> = Map::new();
    let mut req = Request::new();

    match wire.instructions {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) => req.system = Some(vec![ContentBlock::text(s)]),
        Some(other) => {
            warnings.push(warn(
                WarningCode::MalformedField,
                "/instructions",
                "non-string `instructions` kept verbatim in the request extra",
            ));
            ns.insert("instructions".to_owned(), other);
        }
    }
    match wire.input {
        None => {}
        Some(types::RequestInput::Text(s)) => req.messages.push(Message::user_text(s)),
        Some(types::RequestInput::Items(items)) => {
            req.messages = parse_input_items(&items, &mut warnings);
        }
    }
    req.max_output_tokens = wire.max_output_tokens;
    req.temperature = wire.temperature;
    req.top_p = wire.top_p;
    req.metadata = wire.metadata;
    req.cache_key = wire.prompt_cache_key;
    if let Some(rc) = &wire.reasoning {
        req.reasoning = Some(reasoning_to_ir(rc));
    }
    if let Some(tc) = wire.text {
        text_to_ir(tc, &mut req, &mut ns);
    }
    if let Some(tools) = &wire.tools {
        req.tools = Some(
            tools
                .iter()
                .enumerate()
                .map(|(i, t)| tool_to_ir(t, i, &mut warnings))
                .collect(),
        );
    }
    if let Some(choice) = wire.tool_choice {
        tool_choice_to_ir(choice, &mut req, &mut ns);
    }
    req.parallel_tool_calls = wire.parallel_tool_calls;
    // `wire.model` and `wire.stream` are deliberately consumed: the model
    // comes from configuration and streaming from the call mode.
    ns.extend(wire.extra);
    req.extra = Extra::from_unknown(FORMAT, ns);
    Ok((req, warnings))
}

/// Parses a 2xx Response body into the IR (§ 8). `status: "failed"`
/// becomes an [`Error::Api`] built from `response.error`.
pub fn response_to_ir(body: &[u8], meta: &ResponseMeta) -> Result<Response> {
    let raw: Value =
        serde_json::from_slice(body).map_err(|e| parse_failure("response", e, body))?;
    // Two-stage parse: `usage` is split off and parsed leniently below so
    // a malformed usage object degrades to `None` with a warning instead
    // of failing the whole (already billed) 2xx response.
    let mut envelope = raw.clone();
    let usage_value = envelope.as_object_mut().and_then(|o| o.remove("usage"));
    let wire: types::Response =
        serde_json::from_value(envelope).map_err(|e| parse_failure("response", e, body))?;
    if wire.status.as_deref() == Some("failed") {
        return Err(failed_response_error(
            meta.status,
            &meta.headers,
            wire.error.as_ref().and_then(|e| e.code.as_deref()),
            wire.error.as_ref().and_then(|e| e.message.as_deref()),
            body,
            Some(raw),
        ));
    }
    let mut warnings = Vec::new();
    let mut blocks = Vec::new();
    let mut has_pending_tool_call = false;
    for (i, item) in wire.output.iter().enumerate() {
        if awaits_tool_output(item) {
            has_pending_tool_call = true;
        }
        blocks.extend(assistant_item_to_blocks(
            item,
            &format!("/output/{i}"),
            &mut warnings,
        ));
    }
    let usage = usage_lenient(usage_value.as_ref(), &mut warnings);
    let message = Message::assistant(blocks);
    let stop_reason = derive_stop_reason(
        wire.status.as_deref(),
        wire.incomplete_details
            .as_ref()
            .and_then(|d| d.reason.as_deref()),
        has_pending_tool_call,
    );
    let stop_reason = normalize_stop_reason(&message, stop_reason);
    let mut response = Response::new(message);
    response.id = wire.id;
    response.model = wire.model;
    response.stop_reason = stop_reason;
    response.usage = usage;
    response.status = meta.status;
    response.headers = meta.headers.clone();
    response.raw = Some(raw);
    response.warnings = warnings;
    Ok(response)
}

/// Whether an output item is a call awaiting developer-supplied output —
/// `function_call` or `custom_tool_call` ("analogous to `function_call`"
/// per the official reference). These drive the § 8 `EndTurn` → `ToolUse`
/// upgrade; other waiting-type items (`mcp_approval_request`,
/// `local_shell_call`, …) stay `Opaque` and do not participate.
pub(crate) fn awaits_tool_output(item: &Value) -> bool {
    matches!(
        item.get("type").and_then(Value::as_str),
        Some("function_call" | "custom_tool_call")
    )
}

/// § 8 stop-reason derivation for the Responses API: `status` +
/// `incomplete_details.reason` + the presence of output items awaiting
/// developer-supplied output (see [`awaits_tool_output`]). The core
/// `normalize_stop_reason` still runs afterwards.
pub(crate) fn derive_stop_reason(
    status: Option<&str>,
    incomplete_reason: Option<&str>,
    has_pending_tool_call: bool,
) -> Option<StopReason> {
    let base = match status {
        Some("completed") => Some(StopReason::EndTurn),
        Some("incomplete") => Some(match incomplete_reason {
            Some("max_output_tokens") => StopReason::MaxTokens,
            Some("content_filter") => StopReason::ContentFilter,
            Some(other) => StopReason::Other(other.to_owned()),
            None => StopReason::Other("incomplete".to_owned()),
        }),
        Some("in_progress" | "queued") | None => None,
        Some(other) => Some(StopReason::Other(other.to_owned())),
    };
    if base == Some(StopReason::EndTurn) && has_pending_tool_call {
        Some(StopReason::ToolUse)
    } else {
        base
    }
}

/// Builds the [`Error::Api`] for a Response whose `status` is `failed`
/// (§ 8), also used by the streaming `response.failed` / `error` events.
pub(crate) fn failed_response_error(
    status: u16,
    headers: &http::HeaderMap,
    code: Option<&str>,
    message: Option<&str>,
    body: &[u8],
    parsed: Option<Value>,
) -> Error {
    let kind = match code {
        Some("server_error") => ApiErrorKind::ServerError,
        Some("rate_limit_exceeded") => ApiErrorKind::RateLimit,
        Some("invalid_prompt") => ApiErrorKind::InvalidRequest,
        Some(other) => ApiErrorKind::Other(other.to_owned()),
        None => ApiErrorKind::Other("failed".to_owned()),
    };
    Error::Api {
        status,
        kind,
        message: message.map_or_else(
            || "the response status is `failed`".to_owned(),
            str::to_owned,
        ),
        raw: Bytes::copy_from_slice(body),
        truncated: false,
        parsed,
        retry_after: retry_after_from_headers(headers),
        headers: headers.clone(),
    }
}

/// Maps the wire usage object to the unified [`Usage`] (§ 8):
/// `input_tokens` already includes cached tokens. `raw` is the wire value
/// verbatim (never a re-serialization). The core-field unwraps are
/// unreachable zeros — [`usage_lenient`] rejects usage without them.
pub(crate) fn usage_to_ir(u: &types::Usage, raw: Value) -> Usage {
    Usage {
        input_tokens: u.input_tokens.unwrap_or(0),
        output_tokens: u.output_tokens.unwrap_or(0),
        total_tokens: u.total_tokens,
        cache_read_tokens: u
            .input_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens),
        cache_write_tokens: u
            .input_tokens_details
            .as_ref()
            .and_then(|d| d.cache_write_tokens),
        reasoning_tokens: u
            .output_tokens_details
            .as_ref()
            .and_then(|d| d.reasoning_tokens),
        raw: Some(raw),
    }
}

/// Lenient usage parse (§ 8 degradation): absent or `null` is `None`
/// silently; any other shape that fails the typed parse — or an object
/// missing the wire-required `input_tokens`/`output_tokens` — degrades to
/// `None` with a `MalformedField` warning at `/usage`. A billed 2xx
/// response or stream never fails — and a core count never silently reads
/// as zero — over its usage object.
pub(crate) fn usage_lenient(
    value: Option<&Value>,
    warnings: &mut Vec<ConversionWarning>,
) -> Option<Usage> {
    let value = value?;
    if value.is_null() {
        return None;
    }
    match serde_json::from_value::<types::Usage>(value.clone()) {
        Ok(u) if u.input_tokens.is_some() && u.output_tokens.is_some() => {
            Some(usage_to_ir(&u, value.clone()))
        }
        Ok(_) => {
            warnings.push(warn(
                WarningCode::MalformedField,
                "/usage",
                "`usage` lacks the required `input_tokens`/`output_tokens` and was \
                 dropped; core counts are never silently zeroed",
            ));
            None
        }
        Err(e) => {
            warnings.push(warn(
                WarningCode::MalformedField,
                "/usage",
                format!("`usage` failed to parse and was dropped: {e}"),
            ));
            None
        }
    }
}

/// Which IR home an input item belongs to.
enum ItemKind {
    /// A `message` item with role `user` / `system` / `developer`.
    InputMessage,
    /// A `function_call_output` item → its own `Tool` message.
    FunctionCallOutput,
    /// Everything else: assistant messages, `reasoning`, `function_call`,
    /// unknown-role `message` items and unmodeled items (grouped into the
    /// current assistant run; unmapped ones stay `Opaque`).
    AssistantSide,
}

fn classify_item(item: &Value) -> ItemKind {
    let ty = item.get("type").and_then(Value::as_str);
    let role = item.get("role").and_then(Value::as_str);
    match ty {
        Some("message") | None => match role {
            Some("user" | "system" | "developer") => ItemKind::InputMessage,
            _ => ItemKind::AssistantSide,
        },
        Some("function_call_output") => ItemKind::FunctionCallOutput,
        _ => ItemKind::AssistantSide,
    }
}

/// Parses an `input` item array into IR messages: user/system/developer
/// message items map 1:1, each `function_call_output` becomes its own
/// `Tool` message, and consecutive assistant-side items group into one
/// assistant message (implementation contract, "Turn grouping").
pub(crate) fn parse_input_items(
    items: &[Value],
    warnings: &mut Vec<ConversionWarning>,
) -> Vec<Message> {
    let mut messages: Vec<Message> = Vec::new();
    let mut run: Vec<ContentBlock> = Vec::new();
    let flush = |run: &mut Vec<ContentBlock>, messages: &mut Vec<Message>| {
        if !run.is_empty() {
            messages.push(Message::assistant(std::mem::take(run)));
        }
    };
    for (i, item) in items.iter().enumerate() {
        let ptr = format!("/input/{i}");
        match classify_item(item) {
            ItemKind::InputMessage => match input_message_to_ir(item, &ptr, warnings) {
                Some(msg) => {
                    flush(&mut run, &mut messages);
                    messages.push(msg);
                }
                None => run.push(ContentBlock::opaque(FORMAT, item.clone())),
            },
            ItemKind::FunctionCallOutput => {
                match function_call_output_to_ir(item, &ptr, warnings) {
                    Some(msg) => {
                        flush(&mut run, &mut messages);
                        messages.push(msg);
                    }
                    None => run.push(ContentBlock::opaque(FORMAT, item.clone())),
                }
            }
            ItemKind::AssistantSide => {
                run.extend(assistant_item_to_blocks(item, &ptr, warnings));
            }
        }
    }
    flush(&mut run, &mut messages);
    messages
}

/// Parses a user/system/developer `message` item. Returns `None` (caller
/// keeps the item verbatim as `Opaque`) when the typed parse fails.
fn input_message_to_ir(
    item: &Value,
    ptr: &str,
    warnings: &mut Vec<ConversionWarning>,
) -> Option<Message> {
    let wire: types::MessageItem = match serde_json::from_value(item.clone()) {
        Ok(w) => w,
        Err(e) => {
            warnings.push(warn(
                WarningCode::MalformedField,
                ptr.to_owned(),
                format!("message item failed to parse and was kept verbatim: {e}"),
            ));
            return None;
        }
    };
    let role = match wire.role.as_str() {
        "user" => Role::User,
        "system" => Role::System,
        "developer" => Role::Developer,
        _ => return None,
    };
    let content = match wire.content {
        types::MessageContent::Text(s) => vec![ContentBlock::text(s)],
        types::MessageContent::Parts(parts) => parts
            .iter()
            .enumerate()
            .map(|(pi, p)| input_part_to_block(p, &format!("{ptr}/content/{pi}"), warnings))
            .collect(),
    };
    let mut msg = Message::new(role, content);
    msg.extra = Extra::from_unknown(FORMAT, wire.extra);
    Some(msg)
}

/// Splits a `prompt_cache_breakpoint` value into the IR cache hint plus
/// the raw value to preserve when it differs from the canonical
/// `{"mode": "explicit"}` shape.
fn parse_breakpoint(value: Option<Value>) -> (Option<CacheHint>, Option<Value>) {
    match value {
        None => (None, None),
        Some(v) => {
            let canonical = v == json!({"mode": "explicit"});
            (Some(CacheHint::new()), (!canonical).then_some(v))
        }
    }
}

/// Parses one input content part (user/system/developer messages).
fn input_part_to_block(
    part: &Value,
    ptr: &str,
    warnings: &mut Vec<ConversionWarning>,
) -> ContentBlock {
    let malformed = |warnings: &mut Vec<ConversionWarning>, e: &dyn std::fmt::Display| {
        warnings.push(warn(
            WarningCode::MalformedField,
            ptr.to_owned(),
            format!("content part failed to parse and was kept verbatim: {e}"),
        ));
    };
    match part.get("type").and_then(Value::as_str) {
        Some("input_text") => match serde_json::from_value::<types::InputTextPart>(part.clone()) {
            Ok(w) => {
                let (cache, pcb) = parse_breakpoint(w.prompt_cache_breakpoint);
                let mut ns = w.extra;
                if let Some(pcb) = pcb {
                    ns.insert("prompt_cache_breakpoint".to_owned(), pcb);
                }
                ContentBlock::Text {
                    text: w.text,
                    cache,
                    extra: Extra::from_unknown(FORMAT, ns),
                }
            }
            Err(e) => {
                malformed(warnings, &e);
                ContentBlock::opaque(FORMAT, part.clone())
            }
        },
        Some("input_image") => {
            match serde_json::from_value::<types::InputImagePart>(part.clone()) {
                Ok(w) => {
                    let mut ns = w.extra;
                    let source = if let Some(url) = w.image_url {
                        if let Some(id) = w.file_id {
                            // Both channels set: keep the URL as the source and
                            // preserve the file id verbatim.
                            ns.insert("file_id".to_owned(), Value::from(id));
                        }
                        match parse_data_url(&url) {
                            Some((media_type, data)) => ImageSource::base64(media_type, data),
                            None => ImageSource::url(url),
                        }
                    } else if let Some(id) = w.file_id {
                        ImageSource::file_id(id)
                    } else {
                        malformed(
                            warnings,
                            &"input_image has neither `image_url` nor `file_id`",
                        );
                        return ContentBlock::opaque(FORMAT, part.clone());
                    };
                    let (cache, pcb) = parse_breakpoint(w.prompt_cache_breakpoint);
                    if let Some(pcb) = pcb {
                        ns.insert("prompt_cache_breakpoint".to_owned(), pcb);
                    }
                    ContentBlock::Image {
                        source,
                        cache,
                        extra: Extra::from_unknown(FORMAT, ns),
                    }
                }
                Err(e) => {
                    malformed(warnings, &e);
                    ContentBlock::opaque(FORMAT, part.clone())
                }
            }
        }
        // `input_file` and future part kinds stay verbatim.
        _ => ContentBlock::opaque(FORMAT, part.clone()),
    }
}

/// Parses a `function_call_output` item into a `Tool` message holding one
/// `ToolResult`. `output: ""` parses to the empty content list (§ 7.2).
fn function_call_output_to_ir(
    item: &Value,
    ptr: &str,
    warnings: &mut Vec<ConversionWarning>,
) -> Option<Message> {
    let wire: types::FunctionCallOutputItem = match serde_json::from_value(item.clone()) {
        Ok(w) => w,
        Err(e) => {
            warnings.push(warn(
                WarningCode::MalformedField,
                ptr.to_owned(),
                format!("function_call_output item failed to parse and was kept verbatim: {e}"),
            ));
            return None;
        }
    };
    let content: Vec<ToolOutputBlock> = match wire.output {
        types::FunctionCallOutput::Text(s) if s.is_empty() => Vec::new(),
        types::FunctionCallOutput::Text(s) => vec![ToolOutputBlock::text(s)],
        types::FunctionCallOutput::Parts(parts) => parts
            .iter()
            .enumerate()
            .map(|(pi, p)| tool_output_part_to_block(p, &format!("{ptr}/output/{pi}"), warnings))
            .collect(),
    };
    let block = ContentBlock::ToolResult {
        tool_call_id: Some(wire.call_id),
        name: wire.name,
        content,
        is_error: None,
        cache: None,
        extra: Extra::from_unknown(FORMAT, wire.extra),
    };
    Some(Message::tool(vec![block]))
}

/// Parses one nested tool-output part. Nested cache breakpoints are not
/// mapped to IR hints (the v1 build side drops nested hints, § 4.8); they
/// stay verbatim in the part extra so round-trips are warning-free.
fn tool_output_part_to_block(
    part: &Value,
    ptr: &str,
    warnings: &mut Vec<ConversionWarning>,
) -> ToolOutputBlock {
    match part.get("type").and_then(Value::as_str) {
        Some("input_text") => match serde_json::from_value::<types::InputTextPart>(part.clone()) {
            Ok(w) => {
                let mut ns = w.extra;
                if let Some(pcb) = w.prompt_cache_breakpoint {
                    ns.insert("prompt_cache_breakpoint".to_owned(), pcb);
                }
                ToolOutputBlock::Text {
                    text: w.text,
                    cache: None,
                    extra: Extra::from_unknown(FORMAT, ns),
                }
            }
            Err(e) => {
                warnings.push(warn(
                    WarningCode::MalformedField,
                    ptr.to_owned(),
                    format!("tool-output part failed to parse and was kept verbatim: {e}"),
                ));
                ToolOutputBlock::opaque(FORMAT, part.clone())
            }
        },
        Some("input_image") => {
            match serde_json::from_value::<types::InputImagePart>(part.clone()) {
                Ok(w) => {
                    let mut ns = w.extra;
                    let source = if let Some(url) = w.image_url {
                        if let Some(id) = w.file_id {
                            ns.insert("file_id".to_owned(), Value::from(id));
                        }
                        match parse_data_url(&url) {
                            Some((media_type, data)) => ImageSource::base64(media_type, data),
                            None => ImageSource::url(url),
                        }
                    } else if let Some(id) = w.file_id {
                        ImageSource::file_id(id)
                    } else {
                        return ToolOutputBlock::opaque(FORMAT, part.clone());
                    };
                    if let Some(pcb) = w.prompt_cache_breakpoint {
                        ns.insert("prompt_cache_breakpoint".to_owned(), pcb);
                    }
                    ToolOutputBlock::Image {
                        source,
                        cache: None,
                        extra: Extra::from_unknown(FORMAT, ns),
                    }
                }
                Err(e) => {
                    warnings.push(warn(
                        WarningCode::MalformedField,
                        ptr.to_owned(),
                        format!("tool-output part failed to parse and was kept verbatim: {e}"),
                    ));
                    ToolOutputBlock::opaque(FORMAT, part.clone())
                }
            }
        }
        _ => ToolOutputBlock::opaque(FORMAT, part.clone()),
    }
}

/// Item-level fields of an assistant `message` item, redistributed onto
/// each of its `Text` blocks under the reserved keys of the format
/// namespace (see the module docs).
pub(crate) struct AssistantItemMeta {
    ns: Map<String, Value>,
}

impl AssistantItemMeta {
    /// Meta with no item-level fields.
    pub(crate) fn empty() -> Self {
        Self { ns: Map::new() }
    }

    /// Extracts `id` / `status` / `phase` plus unknown item fields (nested
    /// under the reserved `item` key) from a message item's leftover map.
    pub(crate) fn from_item_extra(extra: &Map<String, Value>) -> Self {
        let mut ns = Map::new();
        let mut unknown = Map::new();
        for (key, value) in extra {
            if value.is_null() {
                continue; // the documented § 1 null loss
            }
            match key.as_str() {
                super::text_block_reserved_key::ID
                | super::text_block_reserved_key::STATUS
                | super::text_block_reserved_key::PHASE => {
                    ns.insert(key.clone(), value.clone());
                }
                _ => {
                    unknown.insert(key.clone(), value.clone());
                }
            }
        }
        if !unknown.is_empty() {
            ns.insert(
                super::text_block_reserved_key::ITEM.to_owned(),
                Value::Object(unknown),
            );
        }
        Self { ns }
    }
}

/// Extracts the item meta and content parts of an assistant `message`
/// item value (used by the stream parser at `output_item.done`). A string
/// content shorthand canonicalizes to a single `output_text` part.
pub(crate) fn assistant_message_meta(item: &Value) -> (AssistantItemMeta, Vec<Value>) {
    match serde_json::from_value::<types::MessageItem>(item.clone()) {
        Ok(wire) => {
            let parts = match wire.content {
                types::MessageContent::Text(s) => {
                    vec![json!({"type": "output_text", "text": s, "annotations": []})]
                }
                types::MessageContent::Parts(parts) => parts,
            };
            (AssistantItemMeta::from_item_extra(&wire.extra), parts)
        }
        Err(_) => (AssistantItemMeta::empty(), Vec::new()),
    }
}

/// Parses one assistant message content part into a `Text` block, or
/// `None` for unmodeled part kinds. `refusal` parts are refusal-marked
/// (§ 9); `input_text` parts (assistant replay shorthand) canonicalize to
/// `output_text` on re-serialization.
pub(crate) fn assistant_text_block_from_part(
    part: &Value,
    meta: &AssistantItemMeta,
) -> Option<ContentBlock> {
    let build = |text: String, mut ns: Map<String, Value>| {
        // Item-level fields win over identically named part fields — the
        // contract pins `extra["openai_responses"]["id"]` as the item id.
        for (key, value) in &meta.ns {
            ns.insert(key.clone(), value.clone());
        }
        ContentBlock::Text {
            text,
            cache: None,
            extra: Extra::from_unknown(FORMAT, ns),
        }
    };
    match part.get("type").and_then(Value::as_str)? {
        "output_text" => {
            let wire: types::OutputTextPart = serde_json::from_value(part.clone()).ok()?;
            let mut ns = Map::new();
            if let Some(annotations) = wire.annotations
                && !annotations.is_empty()
            {
                ns.insert("annotations".to_owned(), Value::Array(annotations));
            }
            ns.extend(wire.extra);
            Some(build(wire.text, ns))
        }
        "refusal" => {
            let wire: types::RefusalPart = serde_json::from_value(part.clone()).ok()?;
            let mut ns = Map::new();
            ns.insert(
                super::text_block_reserved_key::REFUSAL.to_owned(),
                Value::from(true),
            );
            ns.extend(wire.extra);
            Some(build(wire.refusal, ns))
        }
        "input_text" => {
            let wire: types::InputTextPart = serde_json::from_value(part.clone()).ok()?;
            let mut ns = Map::new();
            if let Some(pcb) = wire.prompt_cache_breakpoint {
                ns.insert("prompt_cache_breakpoint".to_owned(), pcb);
            }
            ns.extend(wire.extra);
            Some(build(wire.text, ns))
        }
        _ => None,
    }
}

/// Parses one assistant-side item into IR blocks (shared by request
/// input parsing and response output parsing).
pub(crate) fn assistant_item_to_blocks(
    item: &Value,
    ptr: &str,
    warnings: &mut Vec<ConversionWarning>,
) -> Vec<ContentBlock> {
    match item.get("type").and_then(Value::as_str) {
        Some("message") | None => message_item_to_assistant_blocks(item, ptr, warnings),
        Some("reasoning") => vec![thinking_from_reasoning(item, ptr, warnings)],
        Some("function_call") => vec![function_call_to_block(item, ptr, warnings)],
        // Built-in tool items (`web_search_call`, …) and future kinds stay
        // verbatim, in place.
        _ => vec![ContentBlock::opaque(FORMAT, item.clone())],
    }
}

/// Parses an assistant `message` item into `Text` blocks. An item holding
/// an unmodeled part kind, or one whose role is unknown, is kept whole as
/// `Opaque` so its wire shape (role, part vs. top-level item) survives
/// re-serialization.
fn message_item_to_assistant_blocks(
    item: &Value,
    ptr: &str,
    warnings: &mut Vec<ConversionWarning>,
) -> Vec<ContentBlock> {
    // Known input roles were routed off by `classify_item`, so any
    // non-`assistant` role here is unknown. Rewriting it to `assistant`
    // would silently change semantics — keep the whole item verbatim
    // instead (re-emitted in place, like unmodeled item types).
    if let Some(role) = item.get("role").and_then(Value::as_str)
        && role != "assistant"
    {
        warnings.push(warn(
            WarningCode::MalformedField,
            ptr.to_owned(),
            format!("message item with unknown role `{role}` was kept verbatim"),
        ));
        return vec![ContentBlock::opaque(FORMAT, item.clone())];
    }
    let wire: types::MessageItem = match serde_json::from_value(item.clone()) {
        Ok(w) => w,
        Err(e) => {
            warnings.push(warn(
                WarningCode::MalformedField,
                ptr.to_owned(),
                format!("assistant message item failed to parse and was kept verbatim: {e}"),
            ));
            return vec![ContentBlock::opaque(FORMAT, item.clone())];
        }
    };
    let meta = AssistantItemMeta::from_item_extra(&wire.extra);
    match wire.content {
        types::MessageContent::Text(s) => {
            vec![ContentBlock::Text {
                text: s,
                cache: None,
                extra: Extra::from_unknown(FORMAT, meta.ns.clone()),
            }]
        }
        types::MessageContent::Parts(parts) => {
            let mut blocks = Vec::with_capacity(parts.len());
            for part in &parts {
                match assistant_text_block_from_part(part, &meta) {
                    Some(block) => blocks.push(block),
                    None => return vec![ContentBlock::opaque(FORMAT, item.clone())],
                }
            }
            blocks
        }
    }
}

/// Parses a `reasoning` item into a `Thinking` block.
///
/// `text` is the raw chain of thought — the `reasoning_text` parts of the
/// item's `content` array joined with `"\n\n"` — falling back to the
/// summary text (same join) when no raw reasoning is exposed (§ 4.4:
/// "plaintext chain of thought … or joined summary text"). `signature` is
/// `encrypted_content`. The original `id` / `summary` array (plus unknown
/// fields such as `content` and `status`) ride the block's format
/// namespace for lossless reconstruction (implementation contract).
pub(crate) fn thinking_from_reasoning(
    item: &Value,
    ptr: &str,
    warnings: &mut Vec<ConversionWarning>,
) -> ContentBlock {
    let wire: types::ReasoningItem = match serde_json::from_value(item.clone()) {
        Ok(w) => w,
        Err(e) => {
            warnings.push(warn(
                WarningCode::MalformedField,
                ptr.to_owned(),
                format!("reasoning item failed to parse and was kept verbatim: {e}"),
            ));
            return ContentBlock::opaque(FORMAT, item.clone());
        }
    };
    // Raw reasoning (`content` reasoning_text parts) is the actual chain
    // of thought; the summary is a derived rendering. Prefer the former.
    let raw: String = wire
        .extra
        .get("content")
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter(|p| p.get("type").and_then(Value::as_str) == Some("reasoning_text"))
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n\n")
        })
        .unwrap_or_default();
    let joined = if raw.is_empty() {
        wire.summary
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n\n")
    } else {
        raw
    };
    let text = (!joined.is_empty()).then_some(joined);
    let mut ns = Map::new();
    if let Some(id) = wire.id {
        ns.insert(
            super::text_block_reserved_key::ID.to_owned(),
            Value::from(id),
        );
    }
    ns.insert("summary".to_owned(), Value::Array(wire.summary));
    ns.extend(wire.extra);
    ContentBlock::Thinking {
        text,
        signature: wire.encrypted_content,
        extra: Extra::from_unknown(FORMAT, ns),
    }
}

/// Parses a `function_call` item into a `ToolCall` block: `call_id` maps
/// to the block id, the item `id` (and unknown fields) ride the block's
/// format namespace.
pub(crate) fn function_call_to_block(
    item: &Value,
    ptr: &str,
    warnings: &mut Vec<ConversionWarning>,
) -> ContentBlock {
    let wire: types::FunctionCallItem = match serde_json::from_value(item.clone()) {
        Ok(w) => w,
        Err(e) => {
            warnings.push(warn(
                WarningCode::MalformedField,
                ptr.to_owned(),
                format!("function_call item failed to parse and was kept verbatim: {e}"),
            ));
            return ContentBlock::opaque(FORMAT, item.clone());
        }
    };
    ContentBlock::ToolCall {
        id: Some(wire.call_id),
        name: wire.name,
        arguments: wire.arguments,
        cache: None,
        extra: Extra::from_unknown(FORMAT, wire.extra),
    }
}

/// Maps the wire `reasoning` object to the IR (§ 4.7): `effort` maps
/// tier-for-tier, `summary: "auto"` becomes `include_thoughts: true`, and
/// other summary values plus unknown fields ride the `Reasoning` extra.
fn reasoning_to_ir(rc: &types::ReasoningConfig) -> Reasoning {
    let mut reasoning = Reasoning::new();
    reasoning.effort = rc.effort.as_deref().map(Effort::from_str_lossy);
    let mut ns = Map::new();
    match rc.summary.as_deref() {
        Some("auto") => reasoning.include_thoughts = Some(true),
        Some(other) => {
            ns.insert("summary".to_owned(), Value::from(other));
        }
        None => {}
    }
    ns.extend(rc.extra.clone());
    reasoning.extra = Extra::from_unknown(FORMAT, ns);
    reasoning
}

/// Removes and returns a string value from a map, leaving non-strings in
/// place so they round-trip via the mirrored extra.
fn take_string(map: &mut Map<String, Value>, key: &str) -> Option<String> {
    match map.get(key) {
        Some(Value::String(_)) => match map.remove(key) {
            Some(Value::String(s)) => Some(s),
            _ => None,
        },
        _ => None,
    }
}

/// Maps `text` to the IR: `format` becomes `OutputFormat`; `verbosity`,
/// unmapped format shapes and leftover format keys mirror into the request
/// extra at `text` / `text.format`.
fn text_to_ir(tc: types::TextConfig, req: &mut Request, ns: &mut Map<String, Value>) {
    let mut text_ns = Map::new();
    if let Some(format) = tc.format {
        let mapped = match (
            format.get("type").and_then(Value::as_str),
            format.as_object(),
        ) {
            (Some("json_schema"), Some(obj)) if obj.contains_key("schema") => {
                let mut leftover = obj.clone();
                leftover.remove("type");
                let name = take_string(&mut leftover, "name");
                let description = take_string(&mut leftover, "description");
                let schema = leftover.remove("schema").unwrap_or(Value::Null);
                let strict = match leftover.get("strict") {
                    Some(Value::Bool(b)) => {
                        let b = *b;
                        leftover.remove("strict");
                        Some(b)
                    }
                    _ => None,
                };
                req.output_format = Some(OutputFormat::JsonSchema {
                    name,
                    description,
                    schema,
                    strict,
                });
                if !leftover.is_empty() {
                    text_ns.insert("format".to_owned(), Value::Object(leftover));
                }
                true
            }
            (Some("json_object"), Some(obj)) => {
                req.output_format = Some(OutputFormat::JsonObject);
                let mut leftover = obj.clone();
                leftover.remove("type");
                if !leftover.is_empty() {
                    text_ns.insert("format".to_owned(), Value::Object(leftover));
                }
                true
            }
            _ => false,
        };
        if !mapped {
            // `{"type": "text"}` and unknown shapes round-trip verbatim.
            text_ns.insert("format".to_owned(), format);
        }
    }
    text_ns.extend(tc.extra);
    if !text_ns.is_empty() {
        ns.insert("text".to_owned(), Value::Object(text_ns));
    }
}

/// Maps one wire tool definition to the IR: `function` tools become
/// [`FunctionTool`]s (null `parameters` / `strict` → unset), everything
/// else stays verbatim as `Tool::Opaque`.
fn tool_to_ir(value: &Value, index: usize, warnings: &mut Vec<ConversionWarning>) -> Tool {
    if value.get("type").and_then(Value::as_str) != Some("function") {
        return Tool::opaque(FORMAT, value.clone());
    }
    match serde_json::from_value::<types::FunctionToolDef>(value.clone()) {
        Ok(def) => Tool::Function(FunctionTool {
            name: def.name,
            description: def.description,
            parameters: def.parameters,
            strict: def.strict,
            cache: None,
            extra: Extra::from_unknown(FORMAT, def.extra),
        }),
        Err(e) => {
            warnings.push(warn(
                WarningCode::MalformedField,
                format!("/tools/{index}"),
                format!("function tool failed to parse and was kept verbatim: {e}"),
            ));
            Tool::opaque(FORMAT, value.clone())
        }
    }
}

/// Maps `tool_choice` to the IR: the three string modes and the exact
/// `{type: "function", name}` shape are modeled; every other shape
/// (`allowed_tools`, hosted tools, MCP, …) mirrors into the request extra.
fn tool_choice_to_ir(choice: Value, req: &mut Request, ns: &mut Map<String, Value>) {
    match &choice {
        Value::String(s) => match s.as_str() {
            "auto" => req.tool_choice = Some(ToolChoice::Auto),
            "none" => req.tool_choice = Some(ToolChoice::None),
            "required" => req.tool_choice = Some(ToolChoice::Required),
            _ => {
                ns.insert("tool_choice".to_owned(), choice);
            }
        },
        Value::Object(obj)
            if obj.len() == 2
                && obj.get("type").and_then(Value::as_str) == Some("function")
                && obj.get("name").is_some_and(Value::is_string) =>
        {
            let name = obj.get("name").and_then(Value::as_str).unwrap_or_default();
            req.tool_choice = Some(ToolChoice::tool(name));
        }
        _ => {
            ns.insert("tool_choice".to_owned(), choice);
        }
    }
}
