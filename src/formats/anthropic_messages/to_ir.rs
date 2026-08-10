//! Anthropic Messages JSON → IR (parse side): request → IR and
//! response → IR.

use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};

use crate::convert::{ConversionWarning, WarningCode};
use crate::error::{Error, Result};
use crate::format::ResponseMeta;
use crate::ir::{
    CacheHint, ContentBlock, Effort, Extra, ImageSource, Message, OutputFormat, Reasoning, Request,
    Response, Role, StopReason, ToolChoice, ToolOutputBlock, Usage, normalize_stop_reason,
};
use crate::ir::{FunctionTool, Tool};

use super::types::{
    CacheControl, FunctionToolParam, ImageBlock, ImageSourceParam, MessageContent, MessageParam,
    MessagesRequest, MessagesResponse, RedactedThinkingBlock, SystemPrompt, TextBlock,
    ThinkingBlock, ToolResultBlock, ToolResultContent, ToolUseBlock, UsageWire,
};
use super::{FORMAT_ID as FMT, in_array_system_meta};

fn pwarn(
    code: WarningCode,
    location: impl Into<String>,
    text: impl Into<String>,
) -> ConversionWarning {
    ConversionWarning::from_format(code, FMT, location, text)
}

fn parse_error(message: impl Into<String>, raw: &[u8]) -> Error {
    Error::Parse {
        message: message.into(),
        raw: bytes::Bytes::copy_from_slice(raw),
    }
}

/// Reads a block's `type` discriminator.
fn block_type(value: &Value) -> Option<&str> {
    value.get("type").and_then(Value::as_str)
}

/// Builds a block `Extra` from leftover wire fields (null values dropped —
/// the documented § 1 loss).
fn extra_from(map: Map<String, Value>) -> Extra {
    Extra::from_unknown(FMT, map)
}

/// Splits a wire `cache_control` into a modeled [`CacheHint`] (for the
/// documented `ephemeral` type) plus mirrored leftovers in `rest`, so that
/// undocumented cache-control shapes still round-trip via `extra`.
pub(crate) fn split_cache_control(
    cc: Option<CacheControl>,
    rest: &mut Map<String, Value>,
) -> Option<CacheHint> {
    let cc = cc?;
    if cc.kind == "ephemeral" {
        if !cc.extra.is_empty() {
            let filtered: Map<String, Value> =
                cc.extra.into_iter().filter(|(_, v)| !v.is_null()).collect();
            if !filtered.is_empty() {
                rest.insert("cache_control".to_owned(), Value::Object(filtered));
            }
        }
        Some(match cc.ttl {
            Some(ttl) => CacheHint::with_ttl(ttl),
            None => CacheHint::new(),
        })
    } else {
        if let Ok(v) = serde_json::to_value(&cc) {
            rest.insert("cache_control".to_owned(), v);
        }
        None
    }
}

fn image_source_from_wire(source: ImageSourceParam) -> ImageSource {
    match source {
        ImageSourceParam::Base64 { media_type, data } => ImageSource::base64(media_type, data),
        ImageSourceParam::Url { url } => ImageSource::url(url),
        ImageSourceParam::File { file_id } => ImageSource::file_id(file_id),
    }
}

/// Deserializes a known block shape, or falls back to an `Opaque` block with
/// a parse-side `MalformedField` warning.
fn typed_or_opaque<T: DeserializeOwned>(
    value: &Value,
    loc: &str,
    warnings: &mut Vec<ConversionWarning>,
) -> std::result::Result<T, ContentBlock> {
    serde_json::from_value::<T>(value.clone()).map_err(|e| {
        warnings.push(pwarn(
            WarningCode::MalformedField,
            loc,
            format!("known block type failed to parse ({e}); kept as opaque"),
        ));
        ContentBlock::opaque(FMT, value.clone())
    })
}

/// Parses one assistant-side content block (responses and assistant request
/// messages share the same union).
pub(crate) fn parse_assistant_block(
    value: &Value,
    loc: &str,
    warnings: &mut Vec<ConversionWarning>,
) -> ContentBlock {
    match block_type(value) {
        Some("text") => match typed_or_opaque::<TextBlock>(value, loc, warnings) {
            Ok(t) => {
                let mut rest = t.extra;
                let cache = split_cache_control(t.cache_control, &mut rest);
                ContentBlock::Text {
                    text: t.text,
                    cache,
                    extra: extra_from(rest),
                }
            }
            Err(op) => op,
        },
        Some("tool_use") => match typed_or_opaque::<ToolUseBlock>(value, loc, warnings) {
            Ok(t) => {
                if !t.input.is_object() {
                    // Semantic: the value is preserved verbatim, but
                    // `input` requires an object, so even the source
                    // format cannot re-serialize the resulting call.
                    warnings.push(pwarn(
                        WarningCode::MalformedToolCall,
                        format!("{loc}/input"),
                        "tool_use.input is not a JSON object; kept verbatim in \
                         `arguments`, so rebuilding this call for Anthropic will fail",
                    ));
                }
                let mut rest = t.extra;
                let cache = split_cache_control(t.cache_control, &mut rest);
                ContentBlock::ToolCall {
                    id: Some(t.id),
                    name: t.name,
                    arguments: serde_json::to_string(&t.input).unwrap_or_else(|_| "{}".to_owned()),
                    cache,
                    extra: extra_from(rest),
                }
            }
            Err(op) => op,
        },
        Some("thinking") => match typed_or_opaque::<ThinkingBlock>(value, loc, warnings) {
            Ok(t) => ContentBlock::Thinking {
                text: Some(t.thinking),
                signature: t.signature,
                extra: extra_from(t.extra),
            },
            Err(op) => op,
        },
        Some("redacted_thinking") => {
            match typed_or_opaque::<RedactedThinkingBlock>(value, loc, warnings) {
                Ok(t) => {
                    let mut rest = t.extra;
                    // Provenance marker consumed on re-serialization.
                    rest.insert("redacted".to_owned(), Value::Bool(true));
                    ContentBlock::Thinking {
                        text: None,
                        signature: Some(t.data),
                        extra: extra_from(rest),
                    }
                }
                Err(op) => op,
            }
        }
        // Unmodeled union members (server_tool_use, document, compaction, …)
        // stay in place as opaque nodes.
        _ => ContentBlock::opaque(FMT, value.clone()),
    }
}

/// Parses one non-`tool_result` user-side block.
fn parse_user_block(
    value: &Value,
    loc: &str,
    warnings: &mut Vec<ConversionWarning>,
) -> ContentBlock {
    match block_type(value) {
        Some("text") => parse_assistant_block(value, loc, warnings),
        Some("image") => match typed_or_opaque::<ImageBlock>(value, loc, warnings) {
            Ok(img) => {
                let mut rest = img.extra;
                let cache = split_cache_control(img.cache_control, &mut rest);
                ContentBlock::Image {
                    source: image_source_from_wire(img.source),
                    cache,
                    extra: extra_from(rest),
                }
            }
            Err(op) => op,
        },
        _ => ContentBlock::opaque(FMT, value.clone()),
    }
}

/// Parses a `tool_result` block.
fn parse_tool_result(
    value: &Value,
    loc: &str,
    warnings: &mut Vec<ConversionWarning>,
) -> ContentBlock {
    let tr = match typed_or_opaque::<ToolResultBlock>(value, loc, warnings) {
        Ok(tr) => tr,
        Err(op) => return op,
    };
    let mut rest = tr.extra;
    let cache = split_cache_control(tr.cache_control, &mut rest);
    let content: Vec<ToolOutputBlock> = match tr.content {
        None => Vec::new(),
        Some(ToolResultContent::Text(s)) => vec![ToolOutputBlock::text(s)],
        Some(ToolResultContent::Blocks(blocks)) => blocks
            .iter()
            .enumerate()
            .map(|(k, b)| parse_nested_tool_block(b, &format!("{loc}/content/{k}"), warnings))
            .collect(),
    };
    ContentBlock::ToolResult {
        tool_call_id: Some(tr.tool_use_id),
        name: None,
        content,
        is_error: tr.is_error,
        cache,
        extra: extra_from(rest),
    }
}

/// Parses one nested tool-result block. Nested `cache_control` is
/// deliberately *not* modeled as a [`CacheHint`] (the build side drops
/// nested hints in v1, § 4.8) — it stays in the block's `extra` so it
/// round-trips verbatim.
fn parse_nested_tool_block(
    value: &Value,
    _loc: &str,
    _warnings: &mut Vec<ConversionWarning>,
) -> ToolOutputBlock {
    match block_type(value) {
        Some("text") => {
            let Some(obj) = value.as_object() else {
                return ToolOutputBlock::opaque(FMT, value.clone());
            };
            let Some(text) = obj.get("text").and_then(Value::as_str) else {
                return ToolOutputBlock::opaque(FMT, value.clone());
            };
            let rest: Map<String, Value> = obj
                .iter()
                .filter(|(k, _)| k.as_str() != "type" && k.as_str() != "text")
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            ToolOutputBlock::Text {
                text: text.to_owned(),
                cache: None,
                extra: extra_from(rest),
            }
        }
        Some("image") => {
            let source = value
                .get("source")
                .and_then(|s| serde_json::from_value::<ImageSourceParam>(s.clone()).ok());
            let Some(source) = source else {
                return ToolOutputBlock::opaque(FMT, value.clone());
            };
            let rest: Map<String, Value> = value
                .as_object()
                .map(|obj| {
                    obj.iter()
                        .filter(|(k, _)| k.as_str() != "type" && k.as_str() != "source")
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect()
                })
                .unwrap_or_default();
            ToolOutputBlock::Image {
                source: image_source_from_wire(source),
                cache: None,
                extra: extra_from(rest),
            }
        }
        _ => ToolOutputBlock::opaque(FMT, value.clone()),
    }
}

/// Content of a wire message as a block array (string shorthand expanded).
fn content_blocks(content: MessageContent) -> Vec<Value> {
    match content {
        MessageContent::Text(s) => vec![json!({"type": "text", "text": s})],
        MessageContent::Blocks(blocks) => blocks,
    }
}

/// Parses one wire message into one or more IR messages (§ 7.2 turn
/// splitting: a user turn mixing `tool_result` blocks with other content
/// splits into runs sharing a fresh turn-group id). A message with an
/// unknown role is kept verbatim as a lone `Opaque` block; the build side
/// re-emits it untouched.
fn parse_wire_message(
    wm: MessageParam,
    msg_loc: &str,
    group_counter: &mut u64,
    warnings: &mut Vec<ConversionWarning>,
) -> Vec<Message> {
    if !matches!(wm.role.as_str(), "user" | "assistant" | "system") {
        warnings.push(pwarn(
            WarningCode::MalformedField,
            msg_loc,
            format!(
                "unknown message role `{}`; the message was kept verbatim as an opaque block",
                wm.role
            ),
        ));
        // `MessageParam` preserves every field (content untagged, unknown
        // fields flattened), so serializing it back is the wire message.
        let value = serde_json::to_value(&wm).expect("wire message is plain JSON data");
        return vec![Message::user(vec![ContentBlock::opaque(FMT, value)])];
    }
    let MessageParam {
        role,
        content,
        extra: msg_extra,
    } = wm;
    let blocks = content_blocks(content);
    match role.as_str() {
        "assistant" => {
            let content: Vec<ContentBlock> = blocks
                .iter()
                .enumerate()
                .map(|(j, b)| parse_assistant_block(b, &format!("{msg_loc}/content/{j}"), warnings))
                .collect();
            let mut m = Message::assistant(content);
            m.extra = extra_from(msg_extra);
            vec![m]
        }
        "system" => {
            // In-array system content: text parses as Text, everything else
            // is kept opaque so out-of-schema entries survive round-trips.
            let content: Vec<ContentBlock> = blocks
                .iter()
                .enumerate()
                .map(|(j, b)| match block_type(b) {
                    Some("text") => {
                        parse_assistant_block(b, &format!("{msg_loc}/content/{j}"), warnings)
                    }
                    _ => ContentBlock::opaque(FMT, b.clone()),
                })
                .collect();
            let mut m = Message::new(Role::System, content);
            m.round_trip = Some(in_array_system_meta());
            m.extra = extra_from(msg_extra);
            vec![m]
        }
        _ => {
            // Split into runs of tool results vs other content.
            let mut runs: Vec<(bool, Vec<ContentBlock>)> = Vec::new();
            for (j, b) in blocks.iter().enumerate() {
                let loc = format!("{msg_loc}/content/{j}");
                let is_tool = block_type(b) == Some("tool_result");
                let block = if is_tool {
                    parse_tool_result(b, &loc, warnings)
                } else {
                    parse_user_block(b, &loc, warnings)
                };
                match runs.last_mut() {
                    Some((last_is_tool, run)) if *last_is_tool == is_tool => run.push(block),
                    _ => runs.push((is_tool, vec![block])),
                }
            }
            if runs.is_empty() {
                let mut m = Message::user(Vec::new());
                m.extra = extra_from(msg_extra);
                return vec![m];
            }
            let mixed = runs.len() > 1;
            let group = if mixed {
                let g = *group_counter;
                *group_counter += 1;
                Some(g)
            } else {
                None
            };
            let mut out: Vec<Message> = Vec::new();
            for (is_tool, run) in runs {
                let mut m = if is_tool {
                    Message::tool(run)
                } else {
                    Message::user(run)
                };
                if let Some(g) = group {
                    m = m.with_turn_group(g);
                }
                out.push(m);
            }
            // The wire message's own extra attaches to the first produced
            // message (mirrors the build-side first-wire-message rule).
            out[0].extra = extra_from(msg_extra);
            out
        }
    }
}

/// Inserts a non-null value at a mirrored path in the unknown-field root.
fn mirror(root: &mut Map<String, Value>, path: &[&str], value: Value) {
    if value.is_null() {
        return;
    }
    let mut node = root;
    for key in &path[..path.len() - 1] {
        let entry = node
            .entry((*key).to_owned())
            .or_insert_with(|| Value::Object(Map::new()));
        if !entry.is_object() {
            *entry = Value::Object(Map::new());
        }
        node = entry.as_object_mut().expect("just ensured object");
    }
    node.insert(path[path.len() - 1].to_owned(), value);
}

/// Extends a mirrored sub-object with the non-null entries of `map`.
fn mirror_map(root: &mut Map<String, Value>, path: &[&str], map: Map<String, Value>) {
    for (k, v) in map {
        if v.is_null() {
            continue;
        }
        let mut full: Vec<&str> = path.to_vec();
        full.push(&k);
        mirror(root, &full, v);
    }
}

/// Maps an optional wire integer into the IR's `u32` range. A value above
/// `u32::MAX` is not modeled: the IR field stays unset, the original number
/// round-trips verbatim through the unknown-field mirror at `/{key}`, and a
/// `MalformedField` warning discloses the degrade (no silent clamp).
fn u32_or_mirror(
    v: Option<u64>,
    key: &str,
    unknown_root: &mut Map<String, Value>,
    warnings: &mut Vec<ConversionWarning>,
) -> Option<u32> {
    let v = v?;
    match u32::try_from(v) {
        Ok(n) => Some(n),
        Err(_) => {
            warnings.push(pwarn(
                WarningCode::MalformedField,
                format!("/{key}"),
                format!(
                    "`{key}` {v} exceeds the modeled u32 range; the IR field is unset \
                     and the value re-serializes verbatim via `extra`"
                ),
            ));
            mirror(unknown_root, &[key], Value::from(v));
            None
        }
    }
}

/// Parses an Anthropic Messages request body into the IR (§ 11
/// `parse_request`).
pub(crate) fn parse_request_body(body: &[u8]) -> Result<(Request, Vec<ConversionWarning>)> {
    let wire: MessagesRequest = serde_json::from_slice(body)
        .map_err(|e| parse_error(format!("invalid Anthropic Messages request: {e}"), body))?;
    let mut warnings: Vec<ConversionWarning> = Vec::new();
    let mut req = Request::new();
    // Unknown fields to be re-merged at their original paths.
    let mut unknown_root: Map<String, Value> = Map::new();

    // `model` selects configuration, not IR state; `stream` is decided by
    // the call mode — both are dropped (canonicalization).
    req.max_output_tokens = u32_or_mirror(
        wire.max_tokens,
        "max_tokens",
        &mut unknown_root,
        &mut warnings,
    );
    req.temperature = wire.temperature;
    req.top_p = wire.top_p;
    req.top_k = u32_or_mirror(wire.top_k, "top_k", &mut unknown_root, &mut warnings);
    req.stop_sequences = wire.stop_sequences;

    if let Some(metadata) = wire.metadata {
        let mut ir_meta = Map::new();
        for (k, v) in metadata {
            if k == "user_id" {
                ir_meta.insert(k, v);
            } else {
                mirror(&mut unknown_root, &["metadata", k.as_str()], v);
            }
        }
        if !ir_meta.is_empty() {
            req.metadata = Some(ir_meta);
        }
    }

    if let Some(system) = wire.system {
        let blocks: Vec<ContentBlock> = match system {
            SystemPrompt::Text(s) => vec![ContentBlock::text(s)],
            SystemPrompt::Blocks(entries) => entries
                .iter()
                .enumerate()
                .map(|(j, b)| {
                    let loc = format!("/system/{j}");
                    if block_type(b) == Some("text") {
                        parse_assistant_block(b, &loc, &mut warnings)
                    } else {
                        warnings.push(pwarn(
                            WarningCode::MalformedField,
                            loc,
                            "non-text system entry: the whole system channel is parsed as a \
                             leading system message so the entry survives round-trips as an \
                             opaque block",
                        ));
                        ContentBlock::opaque(FMT, b.clone())
                    }
                })
                .collect(),
        };
        // `Request.system` is Text-only (contract pin). An all-text channel
        // parses into it; a channel with any non-text entry (opaque fallback
        // above, or a malformed text entry) instead becomes a marker-less
        // leading System message — the § 7.1 combine rule hoists it back
        // into the top-level `system` field, keeping round-trips idempotent.
        if blocks
            .iter()
            .all(|b| matches!(b, ContentBlock::Text { .. }))
        {
            if !blocks.is_empty() {
                req.system = Some(blocks);
            }
        } else {
            req.messages.push(Message::new(Role::System, blocks));
        }
    }

    let mut group_counter: u64 = 0;
    for (i, wm) in wire.messages.into_iter().enumerate() {
        let msg_loc = format!("/messages/{i}");
        req.messages.extend(parse_wire_message(
            wm,
            &msg_loc,
            &mut group_counter,
            &mut warnings,
        ));
    }

    if let Some(tools) = wire.tools {
        let mut out: Vec<Tool> = Vec::new();
        for (n, tv) in tools.iter().enumerate() {
            let kind = tv.get("type").and_then(Value::as_str);
            let is_custom = kind.is_none() || kind == Some("custom");
            if is_custom && let Ok(ft) = serde_json::from_value::<FunctionToolParam>(tv.clone()) {
                let mut rest = ft.extra;
                let cache = split_cache_control(ft.cache_control, &mut rest);
                out.push(Tool::Function(FunctionTool {
                    name: ft.name,
                    description: ft.description,
                    parameters: Some(ft.input_schema),
                    strict: ft.strict,
                    cache,
                    extra: extra_from(rest),
                }));
            } else {
                if is_custom {
                    warnings.push(pwarn(
                        WarningCode::MalformedField,
                        format!("/tools/{n}"),
                        "custom tool failed to parse; kept as opaque",
                    ));
                }
                out.push(Tool::opaque(FMT, tv.clone()));
            }
        }
        req.tools = Some(out);
    }

    if let Some(tc) = wire.tool_choice {
        let choice = match tc.kind.as_str() {
            "auto" => Some(ToolChoice::Auto),
            "any" => Some(ToolChoice::Required),
            "none" => Some(ToolChoice::None),
            "tool" => tc.name.clone().map(ToolChoice::tool),
            _ => None,
        };
        match choice {
            Some(c) => {
                if tc.kind == "none" {
                    // `none` takes no flag; anything present is unknown data.
                    if let Some(d) = tc.disable_parallel_tool_use {
                        mirror(
                            &mut unknown_root,
                            &["tool_choice", "disable_parallel_tool_use"],
                            Value::Bool(d),
                        );
                    }
                } else if let Some(d) = tc.disable_parallel_tool_use {
                    req.parallel_tool_calls = Some(!d);
                }
                mirror_map(&mut unknown_root, &["tool_choice"], tc.extra);
                req.tool_choice = Some(c);
            }
            None => {
                // Unknown variant (or `tool` without a name): keep verbatim.
                if let Ok(v) = serde_json::to_value(&tc) {
                    mirror(&mut unknown_root, &["tool_choice"], v);
                }
                if tc.kind == "tool" {
                    warnings.push(pwarn(
                        WarningCode::MalformedField,
                        "/tool_choice",
                        "tool_choice of type `tool` has no name; kept in extra",
                    ));
                }
            }
        }
    }

    let mut reasoning = Reasoning::new();
    let mut have_reasoning = false;
    if let Some(thinking) = wire.thinking {
        match thinking.kind.as_str() {
            "disabled" => {
                reasoning.enabled = Some(false);
                have_reasoning = true;
                if let Some(d) = thinking.display {
                    reasoning.extra.set(FMT, "display", d);
                }
            }
            "adaptive" | "enabled" => {
                reasoning.enabled = Some(true);
                have_reasoning = true;
                if thinking.kind == "enabled" {
                    // The manual-budget variant is not modeled; its shape is
                    // restored through the reasoning extra (§ 4.7).
                    reasoning.extra.set(FMT, "type", "enabled");
                    if let Some(b) = thinking.budget_tokens {
                        reasoning.extra.set(FMT, "budget_tokens", b);
                    }
                }
                match thinking.display.as_deref() {
                    Some("summarized") => reasoning.include_thoughts = Some(true),
                    Some("omitted") => reasoning.include_thoughts = Some(false),
                    Some(other) => reasoning.extra.set(FMT, "display", other),
                    None => {}
                }
            }
            _ => {
                // Unknown thinking variant: keep the whole object verbatim.
                if let Ok(v) = serde_json::to_value(&thinking) {
                    mirror(&mut unknown_root, &["thinking"], v);
                }
            }
        }
        if have_reasoning {
            for (k, v) in thinking.extra {
                if !v.is_null() {
                    reasoning.extra.set(FMT, k, v);
                }
            }
        }
    }

    if let Some(oc) = wire.output_config {
        if let Some(effort) = oc.effort {
            reasoning.effort = Some(Effort::from_str_lossy(&effort));
            have_reasoning = true;
        }
        if let Some(format) = oc.format {
            if format.kind == "json_schema" && format.schema.is_some() {
                req.output_format = Some(OutputFormat::json_schema(
                    format.schema.clone().expect("checked"),
                ));
                mirror_map(
                    &mut unknown_root,
                    &["output_config", "format"],
                    format.extra,
                );
            } else {
                if let Ok(v) = serde_json::to_value(&format) {
                    mirror(&mut unknown_root, &["output_config", "format"], v);
                }
                if format.kind == "json_schema" {
                    warnings.push(pwarn(
                        WarningCode::MalformedField,
                        "/output_config/format",
                        "json_schema output format without a schema; kept in extra",
                    ));
                }
            }
        }
        mirror_map(&mut unknown_root, &["output_config"], oc.extra);
    }
    if have_reasoning {
        req.reasoning = Some(reasoning);
    }

    for (k, v) in wire.extra {
        if !v.is_null() {
            unknown_root.insert(k, v);
        }
    }
    req.extra = Extra::from_unknown(FMT, unknown_root);
    Ok((req, warnings))
}

/// Unifies the provider usage object per § 8: `input_tokens` sums uncached
/// input plus cache reads and writes.
pub(crate) fn unify_usage(wire: &UsageWire, raw: Value) -> Usage {
    let mut usage = Usage {
        // Saturating: misbehaving provider data must not overflow (same
        // defense as `visible_output_tokens`'s saturating_sub).
        input_tokens: wire
            .input_tokens
            .unwrap_or(0)
            .saturating_add(wire.cache_creation_input_tokens.unwrap_or(0))
            .saturating_add(wire.cache_read_input_tokens.unwrap_or(0)),
        output_tokens: wire.output_tokens.unwrap_or(0),
        ..Usage::default()
    };
    usage.cache_read_tokens = wire.cache_read_input_tokens;
    usage.cache_write_tokens = wire.cache_creation_input_tokens;
    usage.reasoning_tokens = wire
        .output_tokens_details
        .as_ref()
        .and_then(|d| d.thinking_tokens);
    usage.raw = Some(raw);
    usage
}

/// Parses a 2xx Messages response body into a unified [`Response`].
pub(crate) fn parse_response_body(body: &[u8], meta: &ResponseMeta) -> Result<Response> {
    let raw: Value = serde_json::from_slice(body)
        .map_err(|e| parse_error(format!("invalid Anthropic Messages response: {e}"), body))?;
    let wire: MessagesResponse = serde_json::from_value(raw.clone())
        .map_err(|e| parse_error(format!("invalid Anthropic Messages response: {e}"), body))?;
    let mut warnings: Vec<ConversionWarning> = Vec::new();
    let content: Vec<ContentBlock> = wire
        .content
        .iter()
        .enumerate()
        .map(|(j, b)| parse_assistant_block(b, &format!("/content/{j}"), &mut warnings))
        .collect();
    let message = Message::assistant(content);
    let stop_reason = wire.stop_reason.as_deref().map(StopReason::from_str_lossy);
    let stop_reason = normalize_stop_reason(&message, stop_reason);
    let usage = wire.usage.as_ref().map(|u| {
        let uraw = serde_json::to_value(u).unwrap_or(Value::Null);
        unify_usage(u, uraw)
    });
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
