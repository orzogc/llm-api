//! OpenAI Chat Completions → IR parsing (request → IR and response → IR).

use bytes::Bytes;
use serde_json::{Map, Value, json};

use crate::convert::{ConversionWarning, WarningCode};
use crate::error::{Error, Result};
use crate::format::{ResponseMeta, parse_data_url};
use crate::ir::{
    CacheHint, ContentBlock, Effort, Extra, FunctionTool, ImageSource, Message, OutputFormat,
    Reasoning, Request, Response, Role, StopReason, Tool, ToolChoice, ToolOutputBlock, Usage,
    normalize_stop_reason,
};

use super::{FORMAT, tool_call_reserved_key, types};

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
        message: format!("invalid Chat Completions {what}: {error}"),
        raw: Bytes::copy_from_slice(body),
    }
}

/// Parses a Chat Completions request body into the IR (§ 11
/// `parse_request`).
///
/// Canonicalizations (§ 1): string content shorthands become part arrays
/// unless they stay eligible for the shorthand, a string `stop` becomes a
/// one-element array, legacy `max_tokens` maps to the IR
/// `max_output_tokens` (re-serializing as `max_completion_tokens`), and
/// `model` / `stream` / `stream_options` are configuration, not IR data,
/// and are consumed (`stream_options` members warn `StreamOptionsDropped`
/// except a literal `include_usage: true`, the only value the build side
/// re-injects). Leading `system` messages stay in-array (no hoisting to
/// `Request.system`, implementation contract).
pub fn request_to_ir(body: &[u8]) -> Result<(Request, Vec<ConversionWarning>)> {
    let wire: types::Request =
        serde_json::from_slice(body).map_err(|e| parse_failure("request", e, body))?;
    let mut warnings = Vec::new();
    let mut ns: Map<String, Value> = Map::new();
    let mut req = Request::new();

    req.messages = wire
        .messages
        .iter()
        .enumerate()
        .map(|(i, m)| parse_message(m, i, &mut warnings))
        .collect();

    req.max_output_tokens = wire.max_completion_tokens;
    if let Some(legacy) = wire.max_tokens {
        if req.max_output_tokens.is_none() {
            req.max_output_tokens = Some(legacy);
        } else {
            warnings.push(warn(
                WarningCode::MalformedField,
                "/max_tokens",
                "both `max_completion_tokens` and legacy `max_tokens` are set; \
                 `max_completion_tokens` wins and `max_tokens` was dropped",
            ));
        }
    }
    req.temperature = wire.temperature;
    req.top_p = wire.top_p;
    match wire.stop {
        None => {}
        Some(Value::String(s)) => req.stop_sequences = Some(vec![s]),
        Some(Value::Array(items)) if items.iter().all(Value::is_string) => {
            req.stop_sequences = Some(
                items
                    .into_iter()
                    .filter_map(|v| match v {
                        Value::String(s) => Some(s),
                        _ => None,
                    })
                    .collect(),
            );
        }
        Some(other) => {
            warnings.push(warn(
                WarningCode::MalformedField,
                "/stop",
                "`stop` is neither a string nor an array of strings; kept verbatim in the \
                 request extra",
            ));
            ns.insert("stop".to_owned(), other);
        }
    }
    req.seed = wire.seed;
    req.frequency_penalty = wire.frequency_penalty;
    req.presence_penalty = wire.presence_penalty;
    req.metadata = wire.metadata;
    req.cache_key = wire.prompt_cache_key;
    if let Some(effort) = wire.reasoning_effort.as_deref() {
        req.reasoning = Some(Reasoning::effort(Effort::from_str_lossy(effort)));
    }
    if let Some(rf) = wire.response_format {
        response_format_to_ir(rf, &mut req, &mut ns);
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
    // `model`, `stream` and `stream_options` are deliberately consumed:
    // the model comes from configuration, streaming from the call mode,
    // and `include_usage` from `OpenAiChatCompletionsOptions`. Members
    // the build side cannot reconstruct warn — they are not mirrored into
    // `extra`, since a re-serialized unary body must not carry a bare
    // `stream_options`.
    if let Some(options) = &wire.stream_options {
        warn_dropped_stream_options(options, &mut warnings);
    }
    ns.extend(wire.extra);
    req.extra = Extra::from_unknown(FORMAT, ns);
    Ok((req, warnings))
}

/// Warns about consumed `stream_options` content the build side cannot
/// reconstruct: everything except a literal `include_usage: true` — the
/// only value [`crate::OpenAiChatCompletionsOptions::inject_include_usage`]
/// re-injects on streaming builds. Any other `include_usage` value
/// (`false`, `null`, non-boolean) counts as dropped, since rebuilding it
/// as `true` would flip its meaning.
fn warn_dropped_stream_options(options: &Value, warnings: &mut Vec<ConversionWarning>) {
    let message = match options {
        Value::Object(members) => {
            let dropped: Vec<String> = members
                .iter()
                .filter(|(key, value)| {
                    key.as_str() != "include_usage" || **value != Value::Bool(true)
                })
                .map(|(key, _)| format!("`{key}`"))
                .collect();
            if dropped.is_empty() {
                return;
            }
            // A dropped non-`true` `include_usage` deserves a remedy hint:
            // the default streaming build re-injects the literal `true`.
            let hint = if members
                .get("include_usage")
                .is_some_and(|v| *v != Value::Bool(true))
            {
                "; set `inject_include_usage: false` to stream without the usage chunk"
            } else {
                ""
            };
            format!(
                "`stream_options` member(s) {} were dropped; the build side reconstructs \
                 only a literal `include_usage: true` (per `OpenAiChatCompletionsOptions`){hint}",
                dropped.join(", ")
            )
        }
        _ => "non-object `stream_options` was dropped; the build side reconstructs only \
              a literal `include_usage: true` (per `OpenAiChatCompletionsOptions`)"
            .to_owned(),
    };
    warnings.push(warn(
        WarningCode::StreamOptionsDropped,
        "/stream_options",
        message,
    ));
}

/// Parses a 2xx `chat.completion` body into the IR (§ 8). Reads the first
/// choice; more than one adds a `MultipleCandidates` warning.
pub fn response_to_ir(body: &[u8], meta: &ResponseMeta) -> Result<Response> {
    let raw: Value =
        serde_json::from_slice(body).map_err(|e| parse_failure("response", e, body))?;
    let wire: types::Response =
        serde_json::from_value(raw.clone()).map_err(|e| parse_failure("response", e, body))?;
    let mut warnings = Vec::new();
    if wire.choices.len() > 1 {
        warnings.push(warn(
            WarningCode::MultipleCandidates,
            "/choices",
            format!(
                "the response carries {} choices; only the first was read (the rest remain in \
                 `raw`)",
                wire.choices.len()
            ),
        ));
    }
    let (message, finish_reason) = match wire.choices.first() {
        Some(choice) => parse_choice(choice, &mut warnings),
        None => {
            warnings.push(warn(
                WarningCode::MalformedField,
                "/choices",
                "the response carries no choices; the assistant message is empty",
            ));
            (Message::assistant(Vec::new()), None)
        }
    };
    let stop_reason = finish_reason.as_deref().map(map_finish_reason);
    let stop_reason = normalize_stop_reason(&message, stop_reason);
    let mut response = Response::new(message);
    response.id = wire.id;
    response.model = wire.model;
    response.stop_reason = stop_reason;
    response.usage = wire.usage.as_ref().map(usage_to_ir);
    response.status = meta.status;
    response.headers = meta.headers.clone();
    response.raw = Some(raw);
    response.warnings = warnings;
    Ok(response)
}

/// § 8 finish-reason mapping. The core `normalize_stop_reason` still runs
/// afterwards (non-streaming parses here, streams in the accumulator).
pub(crate) fn map_finish_reason(s: &str) -> StopReason {
    match s {
        "stop" => StopReason::EndTurn,
        "length" => StopReason::MaxTokens,
        "tool_calls" => StopReason::ToolUse,
        "content_filter" => StopReason::ContentFilter,
        other => StopReason::Other(other.to_owned()),
    }
}

/// Parses the first choice into the assistant message plus the raw finish
/// reason.
fn parse_choice(
    choice: &Value,
    warnings: &mut Vec<ConversionWarning>,
) -> (Message, Option<String>) {
    let wire: types::Choice = match serde_json::from_value(choice.clone()) {
        Ok(c) => c,
        Err(e) => {
            warnings.push(warn(
                WarningCode::MalformedField,
                "/choices/0",
                format!("choice failed to parse; the assistant message is empty: {e}"),
            ));
            return (Message::assistant(Vec::new()), None);
        }
    };
    let message = match &wire.message {
        Some(m) => assistant_value_to_message(m, "/choices/0/message", warnings),
        None => {
            warnings.push(warn(
                WarningCode::MalformedField,
                "/choices/0",
                "choice carries no `message`; the assistant message is empty",
            ));
            Message::assistant(Vec::new())
        }
    };
    (message, wire.finish_reason)
}

/// Parses an assistant wire message (response side) into an IR assistant
/// message: `reasoning_content` → `Thinking`, `content` → `Text`,
/// `refusal` → refusal-marked `Text` (§ 9), `tool_calls` → `ToolCall`
/// blocks; unknown message fields (`annotations`, `audio`, …) ride the
/// message extra.
pub(crate) fn assistant_value_to_message(
    value: &Value,
    ptr: &str,
    warnings: &mut Vec<ConversionWarning>,
) -> Message {
    let wire: types::Message = match serde_json::from_value(value.clone()) {
        Ok(m) => m,
        Err(e) => {
            warnings.push(warn(
                WarningCode::MalformedField,
                ptr.to_owned(),
                format!("assistant message failed to parse; it is empty: {e}"),
            ));
            return Message::assistant(Vec::new());
        }
    };
    assistant_wire_to_message(wire, ptr, warnings)
}

/// Builds the IR assistant message from a typed wire message. `role` is
/// forced to assistant; `name` and everything unknown mirrors into the
/// message extra.
fn assistant_wire_to_message(
    wire: types::Message,
    ptr: &str,
    warnings: &mut Vec<ConversionWarning>,
) -> Message {
    let mut ns = Map::new();
    let blocks = assistant_fields_to_blocks(&wire, ptr, &mut ns, warnings);
    if let Some(name) = wire.name {
        ns.insert("name".to_owned(), name);
    }
    if let Some(id) = wire.tool_call_id {
        ns.insert("tool_call_id".to_owned(), id);
    }
    ns.extend(wire.extra);
    let mut msg = Message::assistant(blocks);
    msg.extra = Extra::from_unknown(FORMAT, ns);
    msg
}

/// Shared assistant-side block extraction (request and response parsing):
/// returns the blocks in canonical order — thinking, content parts,
/// message-level refusal, tool calls.
fn assistant_fields_to_blocks(
    wire: &types::Message,
    ptr: &str,
    ns: &mut Map<String, Value>,
    warnings: &mut Vec<ConversionWarning>,
) -> Vec<ContentBlock> {
    let mut blocks: Vec<ContentBlock> = Vec::new();
    match &wire.reasoning_content {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) => {
            // Plaintext thinking with no namespace is native to this
            // format (implementation contract) — no marker needed.
            blocks.push(ContentBlock::thinking(s.clone()));
        }
        Some(other) => {
            warnings.push(warn(
                WarningCode::MalformedField,
                format!("{ptr}/reasoning_content"),
                "non-string `reasoning_content` kept verbatim in the message extra",
            ));
            ns.insert("reasoning_content".to_owned(), other.clone());
        }
    }
    match &wire.content {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) => blocks.push(ContentBlock::text(s.clone())),
        Some(Value::Array(parts)) => {
            for (pi, part) in parts.iter().enumerate() {
                blocks.push(content_part_to_block(
                    part,
                    &format!("{ptr}/content/{pi}"),
                    false,
                    warnings,
                ));
            }
        }
        Some(other) => {
            warnings.push(warn(
                WarningCode::MalformedField,
                format!("{ptr}/content"),
                "assistant `content` is neither a string nor an array; kept verbatim in the \
                 message extra",
            ));
            ns.insert("content".to_owned(), other.clone());
        }
    }
    match &wire.refusal {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) => {
            blocks.push(refusal_text_block(s.clone(), Map::new()));
        }
        Some(other) => {
            warnings.push(warn(
                WarningCode::MalformedField,
                format!("{ptr}/refusal"),
                "non-string `refusal` kept verbatim in the message extra",
            ));
            ns.insert("refusal".to_owned(), other.clone());
        }
    }
    match &wire.tool_calls {
        None | Some(Value::Null) => {}
        Some(Value::Array(entries)) => {
            for (ci, entry) in entries.iter().enumerate() {
                let call_ptr = format!("{ptr}/tool_calls/{ci}");
                if let Some(block) = tool_call_entry_to_block(entry, &call_ptr, warnings) {
                    blocks.push(block);
                }
            }
        }
        Some(other) => {
            warnings.push(warn(
                WarningCode::MalformedField,
                format!("{ptr}/tool_calls"),
                "non-array `tool_calls` kept verbatim in the message extra",
            ));
            ns.insert("tool_calls".to_owned(), other.clone());
        }
    }
    blocks
}

/// A refusal-marked `Text` block (§ 9).
fn refusal_text_block(text: String, mut ns: Map<String, Value>) -> ContentBlock {
    ns.insert("refusal".to_owned(), Value::from(true));
    ContentBlock::Text {
        text,
        cache: None,
        extra: Extra::from_unknown(FORMAT, ns),
    }
}

/// Warns that a tool-call payload member required upstream is absent; the
/// lenient parse substitutes an empty string, which re-serializes as `""`.
fn call_member_or_empty(
    value: Option<String>,
    ptr: &str,
    payload: &str,
    member: &str,
    warnings: &mut Vec<ConversionWarning>,
) -> String {
    value.unwrap_or_else(|| {
        warnings.push(warn(
            WarningCode::MalformedField,
            format!("{ptr}/{payload}/{member}"),
            format!(
                "`{payload}.{member}` is missing; it parses as an empty string and \
                 re-serializes as such"
            ),
        ));
        String::new()
    })
}

/// Parses one `tool_calls[]` entry into a `ToolCall` block. `function`
/// entries map directly; `custom` entries and unknown kinds use the
/// reserved `type` key of the format namespace (see
/// [`super::tool_call_reserved_key`]). Returns `None` (entry skipped, with
/// a warning) for non-object entries. Upstream-required fields that are
/// absent parse leniently — `id` as `None`, payload strings as `""` — and
/// each missing one warns `MalformedField` at its field path.
pub(crate) fn tool_call_entry_to_block(
    entry: &Value,
    ptr: &str,
    warnings: &mut Vec<ConversionWarning>,
) -> Option<ContentBlock> {
    let wire: types::ToolCallEntry = match serde_json::from_value(entry.clone()) {
        Ok(e) => e,
        Err(e) => {
            warnings.push(warn(
                WarningCode::MalformedField,
                ptr.to_owned(),
                format!("tool call entry failed to parse and was skipped: {e}"),
            ));
            return None;
        }
    };
    if wire.id.is_none() {
        warnings.push(warn(
            WarningCode::MalformedField,
            format!("{ptr}/id"),
            "tool call entry has no `id`; the IR keeps `id: None`, and formats that \
             require tool call ids will fail to rebuild the call",
        ));
    }
    let mut ns = Map::new();
    let (name, arguments) = match wire.call_type.as_deref() {
        None | Some("function") => {
            if wire.function.is_none() {
                warnings.push(warn(
                    WarningCode::MalformedField,
                    format!("{ptr}/function"),
                    "`function` tool call carries no `function` payload; `name` and \
                     `arguments` parse as empty strings",
                ));
            }
            let payload = wire.function.unwrap_or_default();
            if !payload.extra.is_empty() {
                ns.insert("function".to_owned(), Value::Object(payload.extra));
            }
            if let Some(custom) = wire.custom {
                ns.insert(
                    "custom".to_owned(),
                    serde_json::to_value(custom).unwrap_or(Value::Null),
                );
            }
            (
                call_member_or_empty(payload.name, ptr, "function", "name", warnings),
                call_member_or_empty(payload.arguments, ptr, "function", "arguments", warnings),
            )
        }
        Some("custom") => {
            ns.insert(
                tool_call_reserved_key::TYPE.to_owned(),
                Value::from("custom"),
            );
            if wire.custom.is_none() {
                warnings.push(warn(
                    WarningCode::MalformedField,
                    format!("{ptr}/custom"),
                    "`custom` tool call carries no `custom` payload; `name` and `input` \
                     parse as empty strings",
                ));
            }
            let payload = wire.custom.unwrap_or_default();
            if !payload.extra.is_empty() {
                ns.insert("custom".to_owned(), Value::Object(payload.extra));
            }
            if let Some(function) = wire.function {
                ns.insert(
                    "function".to_owned(),
                    serde_json::to_value(function).unwrap_or(Value::Null),
                );
            }
            (
                call_member_or_empty(payload.name, ptr, "custom", "name", warnings),
                call_member_or_empty(payload.input, ptr, "custom", "input", warnings),
            )
        }
        Some(other) => {
            // Unknown call kinds: mirror every field except `id` wholesale
            // so serialization can restore the entry verbatim.
            warnings.push(warn(
                WarningCode::MalformedField,
                ptr.to_owned(),
                format!("unknown tool call type `{other}`; the entry was mirrored verbatim"),
            ));
            if let Value::Object(fields) = entry {
                for (key, value) in fields {
                    if key != "id" {
                        ns.insert(key.clone(), value.clone());
                    }
                }
            }
            (String::new(), String::new())
        }
    };
    ns.extend(wire.extra);
    Some(ContentBlock::ToolCall {
        id: wire.id,
        name,
        arguments,
        cache: None,
        extra: Extra::from_unknown(FORMAT, ns),
    })
}

/// Parses one wire message (request side) into an IR message. Unmodeled
/// roles (legacy `function`, dialect roles) and structurally garbage
/// entries keep the whole message verbatim as a lone `Opaque` block.
fn parse_message(value: &Value, index: usize, warnings: &mut Vec<ConversionWarning>) -> Message {
    let ptr = format!("/messages/{index}");
    let wire: types::Message = match serde_json::from_value(value.clone()) {
        Ok(m) => m,
        Err(e) => {
            warnings.push(warn(
                WarningCode::MalformedField,
                ptr,
                format!("message failed to parse and was kept verbatim: {e}"),
            ));
            return Message::user(vec![ContentBlock::opaque(FORMAT, value.clone())]);
        }
    };
    let Some(role) = wire.role.clone() else {
        warnings.push(warn(
            WarningCode::MalformedField,
            ptr,
            "message has no `role`; kept verbatim",
        ));
        return Message::user(vec![ContentBlock::opaque(FORMAT, value.clone())]);
    };
    match role.as_str() {
        "system" => input_message_to_ir(Role::System, &wire, &ptr, warnings),
        "developer" => input_message_to_ir(Role::Developer, &wire, &ptr, warnings),
        "user" => input_message_to_ir(Role::User, &wire, &ptr, warnings),
        "assistant" => assistant_wire_to_message(wire, &ptr, warnings),
        "tool" => tool_message_to_ir(&wire, &ptr, warnings),
        // The legacy `function` role maps to the IR Tool role
        // (implementation contract) as a whole-message Opaque node.
        "function" => Message::tool(vec![ContentBlock::opaque(FORMAT, value.clone())]),
        other => {
            warnings.push(warn(
                WarningCode::MalformedField,
                ptr,
                format!("unknown message role `{other}`; the message was kept verbatim"),
            ));
            Message::user(vec![ContentBlock::opaque(FORMAT, value.clone())])
        }
    }
}

/// Parses a system/developer/user wire message.
fn input_message_to_ir(
    role: Role,
    wire: &types::Message,
    ptr: &str,
    warnings: &mut Vec<ConversionWarning>,
) -> Message {
    let mut ns = Map::new();
    let mut blocks: Vec<ContentBlock> = Vec::new();
    match &wire.content {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) => blocks.push(ContentBlock::text(s.clone())),
        Some(Value::Array(parts)) => {
            for (pi, part) in parts.iter().enumerate() {
                blocks.push(content_part_to_block(
                    part,
                    &format!("{ptr}/content/{pi}"),
                    true,
                    warnings,
                ));
            }
        }
        Some(other) => {
            warnings.push(warn(
                WarningCode::MalformedField,
                format!("{ptr}/content"),
                "`content` is neither a string nor an array; kept verbatim in the message extra",
            ));
            ns.insert("content".to_owned(), other.clone());
        }
    }
    stash_unused_message_fields(wire, &mut ns);
    if let Some(name) = &wire.name {
        ns.insert("name".to_owned(), name.clone());
    }
    ns.extend(wire.extra.clone());
    let mut msg = Message::new(role, blocks);
    msg.extra = Extra::from_unknown(FORMAT, ns);
    msg
}

/// Preserves assistant/tool-specific fields that appeared on a role that
/// does not use them.
fn stash_unused_message_fields(wire: &types::Message, ns: &mut Map<String, Value>) {
    if let Some(v) = &wire.reasoning_content {
        ns.insert("reasoning_content".to_owned(), v.clone());
    }
    if let Some(v) = &wire.refusal {
        ns.insert("refusal".to_owned(), v.clone());
    }
    if let Some(v) = &wire.tool_calls {
        ns.insert("tool_calls".to_owned(), v.clone());
    }
    if let Some(v) = &wire.tool_call_id {
        ns.insert("tool_call_id".to_owned(), v.clone());
    }
}

/// Parses a `role: "tool"` wire message into its own IR `Tool` message
/// holding one `ToolResult` (implementation contract). `content: ""`
/// parses to the empty content list (§ 7.2); unknown wire-message fields
/// ride the block extra so multi-result IR messages keep per-message data.
fn tool_message_to_ir(
    wire: &types::Message,
    ptr: &str,
    warnings: &mut Vec<ConversionWarning>,
) -> Message {
    let mut ns = Map::new();
    let tool_call_id = match &wire.tool_call_id {
        Some(Value::String(s)) => Some(s.clone()),
        None | Some(Value::Null) => None,
        Some(other) => {
            warnings.push(warn(
                WarningCode::MalformedField,
                format!("{ptr}/tool_call_id"),
                "non-string `tool_call_id` kept verbatim in the block extra",
            ));
            ns.insert("tool_call_id".to_owned(), other.clone());
            None
        }
    };
    let name = match &wire.name {
        Some(Value::String(s)) => Some(s.clone()),
        None | Some(Value::Null) => None,
        Some(other) => {
            ns.insert("name".to_owned(), other.clone());
            None
        }
    };
    let mut content: Vec<ToolOutputBlock> = Vec::new();
    match &wire.content {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) if s.is_empty() => {}
        Some(Value::String(s)) => content.push(ToolOutputBlock::text(s.clone())),
        Some(Value::Array(parts)) => {
            for (pi, part) in parts.iter().enumerate() {
                content.push(tool_output_part_to_block(
                    part,
                    &format!("{ptr}/content/{pi}"),
                ));
            }
        }
        Some(other) => {
            warnings.push(warn(
                WarningCode::MalformedField,
                format!("{ptr}/content"),
                "tool `content` is neither a string nor an array; kept verbatim in the block \
                 extra",
            ));
            ns.insert("content".to_owned(), other.clone());
        }
    }
    if let Some(v) = &wire.reasoning_content {
        ns.insert("reasoning_content".to_owned(), v.clone());
    }
    if let Some(v) = &wire.refusal {
        ns.insert("refusal".to_owned(), v.clone());
    }
    if let Some(v) = &wire.tool_calls {
        ns.insert("tool_calls".to_owned(), v.clone());
    }
    ns.extend(wire.extra.clone());
    let block = ContentBlock::ToolResult {
        tool_call_id,
        name,
        content,
        is_error: None,
        cache: None,
        extra: Extra::from_unknown(FORMAT, ns),
    };
    Message::tool(vec![block])
}

/// Parses one nested tool-message content part. Text parts map; nested
/// cache breakpoints stay verbatim in the part extra (the v1 build side
/// drops nested hints, § 4.8) so round-trips are warning-free.
fn tool_output_part_to_block(part: &Value, _ptr: &str) -> ToolOutputBlock {
    if part.get("type").and_then(Value::as_str) == Some("text")
        && let Ok(w) = serde_json::from_value::<types::TextPart>(part.clone())
    {
        let mut ns = w.extra;
        if let Some(pcb) = w.prompt_cache_breakpoint {
            ns.insert("prompt_cache_breakpoint".to_owned(), pcb);
        }
        return ToolOutputBlock::Text {
            text: w.text,
            cache: None,
            extra: Extra::from_unknown(FORMAT, ns),
        };
    }
    ToolOutputBlock::opaque(FORMAT, part.clone())
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

/// Parses one content part. `input_side` maps breakpoints to IR cache
/// hints (user/system/developer); on the assistant side breakpoints stay
/// verbatim in the extra (the build side drops assistant hints, § 4.8).
fn content_part_to_block(
    part: &Value,
    ptr: &str,
    input_side: bool,
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
        Some("text") => match serde_json::from_value::<types::TextPart>(part.clone()) {
            Ok(w) => {
                let mut ns = w.extra;
                let cache = if input_side {
                    let (cache, pcb) = parse_breakpoint(w.prompt_cache_breakpoint);
                    if let Some(pcb) = pcb {
                        ns.insert("prompt_cache_breakpoint".to_owned(), pcb);
                    }
                    cache
                } else {
                    if let Some(pcb) = w.prompt_cache_breakpoint {
                        ns.insert("prompt_cache_breakpoint".to_owned(), pcb);
                    }
                    None
                };
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
        Some("image_url") => match serde_json::from_value::<types::ImagePart>(part.clone()) {
            Ok(w) => {
                let mut ns = w.extra;
                let source = match parse_data_url(&w.image_url.url) {
                    Some((media_type, data)) => ImageSource::base64(media_type, data),
                    None => ImageSource::url(w.image_url.url),
                };
                if !w.image_url.extra.is_empty() {
                    ns.insert("image_url".to_owned(), Value::Object(w.image_url.extra));
                }
                let cache = if input_side {
                    let (cache, pcb) = parse_breakpoint(w.prompt_cache_breakpoint);
                    if let Some(pcb) = pcb {
                        ns.insert("prompt_cache_breakpoint".to_owned(), pcb);
                    }
                    cache
                } else {
                    if let Some(pcb) = w.prompt_cache_breakpoint {
                        ns.insert("prompt_cache_breakpoint".to_owned(), pcb);
                    }
                    None
                };
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
        },
        Some("refusal") => match serde_json::from_value::<types::RefusalPart>(part.clone()) {
            Ok(w) => refusal_text_block(w.refusal, w.extra),
            Err(e) => {
                malformed(warnings, &e);
                ContentBlock::opaque(FORMAT, part.clone())
            }
        },
        // `input_audio`, `file` and future part kinds stay verbatim.
        _ => ContentBlock::opaque(FORMAT, part.clone()),
    }
}

/// Maps the wire usage object to the unified [`Usage`] (§ 8):
/// `prompt_tokens` already includes cached tokens.
pub(crate) fn usage_to_ir(u: &types::Usage) -> Usage {
    Usage {
        input_tokens: u.prompt_tokens.unwrap_or(0),
        output_tokens: u.completion_tokens.unwrap_or(0),
        total_tokens: u.total_tokens,
        cache_read_tokens: u
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens),
        cache_write_tokens: u
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cache_write_tokens),
        reasoning_tokens: u
            .completion_tokens_details
            .as_ref()
            .and_then(|d| d.reasoning_tokens),
        raw: serde_json::to_value(u).ok(),
    }
}

/// Parses a usage object out of a raw JSON value, ignoring non-objects.
pub(crate) fn usage_from_value(value: &Value) -> Option<Usage> {
    if !value.is_object() {
        return None;
    }
    serde_json::from_value::<types::Usage>(value.clone())
        .ok()
        .map(|u| usage_to_ir(&u))
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

/// Maps `response_format` to the IR (§ 4.9): the `json_schema` and
/// `json_object` shapes are modeled; `{"type": "text"}` and unknown shapes
/// mirror into the request extra verbatim.
fn response_format_to_ir(format: Value, req: &mut Request, ns: &mut Map<String, Value>) {
    let mapped = match format.get("type").and_then(Value::as_str) {
        Some("json_schema") => match format.get("json_schema").and_then(Value::as_object) {
            Some(inner) if inner.contains_key("schema") => {
                let mut leftover = inner.clone();
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
                let mut mirror = format.as_object().cloned().unwrap_or_default();
                mirror.remove("type");
                mirror.remove("json_schema");
                if !leftover.is_empty() {
                    mirror.insert("json_schema".to_owned(), Value::Object(leftover));
                }
                if !mirror.is_empty() {
                    ns.insert("response_format".to_owned(), Value::Object(mirror));
                }
                true
            }
            _ => false,
        },
        Some("json_object") => {
            req.output_format = Some(OutputFormat::JsonObject);
            let mut mirror = format.as_object().cloned().unwrap_or_default();
            mirror.remove("type");
            if !mirror.is_empty() {
                ns.insert("response_format".to_owned(), Value::Object(mirror));
            }
            true
        }
        _ => false,
    };
    if !mapped {
        // `{"type": "text"}` and unknown shapes round-trip verbatim.
        ns.insert("response_format".to_owned(), format);
    }
}

/// Maps one wire tool definition to the IR: `function` tools become
/// [`FunctionTool`]s; everything else (`custom` tools, dialect kinds)
/// stays verbatim as `Tool::Opaque`.
fn tool_to_ir(value: &Value, index: usize, warnings: &mut Vec<ConversionWarning>) -> Tool {
    if value.get("type").and_then(Value::as_str) != Some("function") {
        return Tool::opaque(FORMAT, value.clone());
    }
    match serde_json::from_value::<types::FunctionToolDef>(value.clone()) {
        Ok(def) => {
            let mut ns = def.extra;
            if !def.function.extra.is_empty() {
                ns.insert("function".to_owned(), Value::Object(def.function.extra));
            }
            Tool::Function(FunctionTool {
                name: def.function.name,
                description: def.function.description,
                parameters: def.function.parameters,
                strict: def.function.strict,
                cache: None,
                extra: Extra::from_unknown(FORMAT, ns),
            })
        }
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
/// `{type: "function", function: {name}}` shape are modeled; every other
/// shape (`allowed_tools`, `custom`, …) mirrors into the request extra.
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
                && obj
                    .get("function")
                    .and_then(Value::as_object)
                    .is_some_and(|f| {
                        f.len() == 1 && f.get("name").is_some_and(Value::is_string)
                    }) =>
        {
            let name = choice
                .pointer("/function/name")
                .and_then(Value::as_str)
                .unwrap_or_default();
            req.tool_choice = Some(ToolChoice::tool(name));
        }
        _ => {
            ns.insert("tool_choice".to_owned(), choice);
        }
    }
}
