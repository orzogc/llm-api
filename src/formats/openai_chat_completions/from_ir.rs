//! IR → OpenAI Chat Completions request conversion (build side).

use std::borrow::Cow;

use serde_json::{Map, Value, json};

use crate::convert::{ConversionWarning, ConvertOptions, OrphanToolCalls, WarningCode};
use crate::error::{ConversionError, Result};
use crate::format::{CallMode, OpenAiChatCompletionsOptions, to_data_url};
use crate::ir::{
    CacheHint, ContentBlock, Effort, Extra, ImageSource, MergeLog, Message, OutputFormat,
    Reasoning, Request, Role, Tool, ToolChoice, ToolOutputBlock,
};

use super::{FORMAT, tool_call_reserved_key};

/// Output of the build-side conversion, before
/// [`crate::finalize_request`] runs.
#[derive(Debug)]
pub(crate) struct BuiltBody {
    /// The request JSON (extra merges applied).
    pub body: Value,
    /// Build-side warnings.
    pub warnings: Vec<ConversionWarning>,
    /// Log of the `extra` merge operations (for `overridden` marking).
    pub merge_log: MergeLog,
    /// `(JSON pointer, role)` per serialized wire message, in order.
    pub messages: Vec<(String, Role)>,
}

/// Build-side warning shorthand.
fn warn(
    code: WarningCode,
    location: impl Into<String>,
    message: impl Into<String>,
) -> ConversionWarning {
    ConversionWarning::to_format(code, FORMAT, location, message)
}

/// Converts an IR request into the Chat Completions request body.
pub(crate) fn build_body(
    req: &Request,
    model: Option<&str>,
    mode: CallMode,
    options: &ConvertOptions,
    format_options: &OpenAiChatCompletionsOptions,
) -> Result<BuiltBody> {
    crate::convert::check_finite_sampling(&[
        (req.temperature, "/temperature"),
        (req.top_p, "/top_p"),
        (req.frequency_penalty, "/frequency_penalty"),
        (req.presence_penalty, "/presence_penalty"),
    ])?;
    let mut warnings = Vec::new();
    let mut log = MergeLog::new();
    let messages = preprocess_messages(req, options, &mut warnings);

    let mut body = Map::new();
    if let Some(model) = model
        && !model.is_empty()
    {
        body.insert("model".to_owned(), Value::from(model));
    }

    let mut wire_messages: Vec<Value> = Vec::new();
    let mut pointers: Vec<(String, Role)> = Vec::new();
    if let Some(system) = &req.system {
        let msg = build_system_from_request(system, &mut warnings, &mut log)?;
        wire_messages.push(msg);
        pointers.push(("/messages/0".to_owned(), Role::System));
    }
    for (mi, msg) in messages.iter().enumerate() {
        build_message(
            msg,
            mi,
            options,
            &mut wire_messages,
            &mut pointers,
            &mut warnings,
            &mut log,
        )?;
    }
    body.insert("messages".to_owned(), Value::Array(wire_messages));

    if let Some(v) = req.max_output_tokens {
        body.insert("max_completion_tokens".to_owned(), Value::from(v));
    }
    if let Some(v) = req.temperature {
        body.insert("temperature".to_owned(), Value::from(v));
    }
    if let Some(v) = req.top_p {
        body.insert("top_p".to_owned(), Value::from(v));
    }
    if req.top_k.is_some() {
        warnings.push(warn(
            WarningCode::SamplingParameterDropped,
            "/top_k",
            "`top_k` has no Chat Completions equivalent and was dropped",
        ));
    }
    if let Some(stop) = &req.stop_sequences {
        body.insert("stop".to_owned(), Value::from(stop.clone()));
    }
    if let Some(v) = req.seed {
        body.insert("seed".to_owned(), Value::from(v));
    }
    if let Some(v) = req.frequency_penalty {
        body.insert("frequency_penalty".to_owned(), Value::from(v));
    }
    if let Some(v) = req.presence_penalty {
        body.insert("presence_penalty".to_owned(), Value::from(v));
    }
    if let Some(metadata) = &req.metadata {
        body.insert("metadata".to_owned(), Value::Object(metadata.clone()));
    }
    if let Some(key) = &req.cache_key {
        body.insert("prompt_cache_key".to_owned(), Value::from(key.clone()));
    }
    if let Some(reasoning) = &req.reasoning {
        build_reasoning(reasoning, &mut body, &mut warnings);
    }
    if let Some(format) = &req.output_format {
        body.insert("response_format".to_owned(), build_response_format(format));
    }
    if let Some(tools) = &req.tools {
        let built = build_tools(tools, &mut warnings, &mut log);
        if !built.is_empty() {
            body.insert("tools".to_owned(), Value::Array(built));
        }
    }
    if let Some(choice) = &req.tool_choice {
        body.insert("tool_choice".to_owned(), build_tool_choice(choice));
    }
    if let Some(parallel) = req.parallel_tool_calls {
        let has_tools = req.tools.as_ref().is_some_and(|t| !t.is_empty());
        if has_tools {
            body.insert("parallel_tool_calls".to_owned(), Value::from(parallel));
        } else {
            warnings.push(warn(
                WarningCode::ParallelToolCallsIgnored,
                "/parallel_tool_calls",
                "`parallel_tool_calls` is meaningless without tools and was not emitted",
            ));
        }
    }
    if mode == CallMode::Streaming {
        body.insert("stream".to_owned(), Value::from(true));
        if format_options.inject_include_usage {
            body.insert("stream_options".to_owned(), json!({"include_usage": true}));
        }
    }

    let mut body = Value::Object(body);
    req.extra.merge_into(FORMAT, &mut body, "", &mut log);

    Ok(BuiltBody {
        body,
        warnings,
        merge_log: log,
        messages: pointers,
    })
}

/// Applies the § 7.3 orphan-tool-call policy and missing-thinking handling,
/// cloning the message list only when a policy modifies it.
fn preprocess_messages<'a>(
    req: &'a Request,
    options: &ConvertOptions,
    warnings: &mut Vec<ConversionWarning>,
) -> Cow<'a, [Message]> {
    let mut messages: Cow<'a, [Message]> = Cow::Borrowed(&req.messages);
    let orphans = find_orphans(&messages);
    let last = messages.len().saturating_sub(1);
    for (mi, blocks) in &orphans {
        if *mi < last {
            warnings.push(warn(
                WarningCode::OrphanToolCalls,
                "/messages",
                format!(
                    "message {} contains {} unmatched tool call(s) in the middle of the conversation",
                    mi,
                    blocks.len()
                ),
            ));
        }
    }
    let trailing: Option<&(usize, Vec<usize>)> = orphans.iter().find(|(mi, _)| *mi == last);
    if let Some((mi, blocks)) = trailing {
        match options.orphan_tool_calls {
            OrphanToolCalls::Passthrough => {}
            OrphanToolCalls::DropTrailing => {
                let count = blocks.len();
                let owned = messages.to_mut();
                let msg = &mut owned[*mi];
                let had_calls = msg
                    .content
                    .iter()
                    .filter(|b| matches!(b, ContentBlock::ToolCall { .. }))
                    .count();
                let mut bi = 0usize;
                msg.content.retain(|_| {
                    let drop = blocks.contains(&bi);
                    bi += 1;
                    !drop
                });
                let calls_left = msg
                    .content
                    .iter()
                    .any(|b| matches!(b, ContentBlock::ToolCall { .. }));
                let thinking_left = msg
                    .content
                    .iter()
                    .any(|b| matches!(b, ContentBlock::Thinking { .. }));
                if msg.content.is_empty() {
                    owned.remove(*mi);
                } else if had_calls > 0 && !calls_left && thinking_left {
                    warnings.push(warn(
                        WarningCode::ThinkingOrphaned,
                        "/messages",
                        "dropping the trailing tool calls left a thinking block without them; \
                         the upstream may reject the turn",
                    ));
                }
                warnings.push(warn(
                    WarningCode::OrphanToolCallsDropped,
                    "/messages",
                    format!("removed {count} unmatched trailing tool call(s)"),
                ));
            }
            OrphanToolCalls::SynthesizeError => {
                let mut results = Vec::new();
                for bi in blocks {
                    if let ContentBlock::ToolCall { id, name, .. } = &messages[*mi].content[*bi] {
                        results.push(
                            ContentBlock::tool_result_text(id.clone(), "cancelled")
                                .with_tool_name(name.clone())
                                .with_is_error(true),
                        );
                    }
                }
                let count = results.len();
                messages.to_mut().push(Message::tool(results));
                warnings.push(warn(
                    WarningCode::OrphanToolCallsSynthesized,
                    "/messages",
                    format!("appended {count} synthetic error tool result(s) for unmatched trailing tool call(s)"),
                ));
            }
        }
    }

    if request_enables_thinking(req.reasoning.as_ref()) {
        let offending: Vec<usize> = messages
            .iter()
            .enumerate()
            .filter(|(_, m)| {
                m.role == Role::Assistant
                    && m.content
                        .iter()
                        .any(|b| matches!(b, ContentBlock::ToolCall { .. }))
                    && !m
                        .content
                        .iter()
                        .any(|b| matches!(b, ContentBlock::Thinking { .. }))
            })
            .map(|(i, _)| i)
            .collect();
        for mi in offending {
            if let Some(text) = &options.fill_missing_thinking {
                messages.to_mut()[mi]
                    .content
                    .insert(0, ContentBlock::thinking(text.clone()));
                warnings.push(warn(
                    WarningCode::MissingThinkingFilled,
                    "/messages",
                    format!("inserted a placeholder thinking block into assistant message {mi}"),
                ));
            } else {
                warnings.push(warn(
                    WarningCode::MissingThinkingWithToolCalls,
                    "/messages",
                    format!(
                        "assistant message {mi} carries tool calls but no thinking block while \
                         the request enables thinking; the upstream may reject or degrade the turn"
                    ),
                ));
            }
        }
    }
    messages
}

/// `true` when the request enables thinking per the implementation
/// contract: `reasoning` present and (`enabled == Some(true)`, or `enabled`
/// unset and `effort` set to something other than `Effort::None`).
fn request_enables_thinking(reasoning: Option<&Reasoning>) -> bool {
    reasoning.is_some_and(|r| {
        r.enabled == Some(true)
            || (r.enabled.is_none() && matches!(&r.effort, Some(e) if *e != Effort::None))
    })
}

/// Finds unmatched tool calls: `(message index, offending block indices)`.
fn find_orphans(messages: &[Message]) -> Vec<(usize, Vec<usize>)> {
    let mut out = Vec::new();
    for (i, msg) in messages.iter().enumerate() {
        if msg.role != Role::Assistant {
            continue;
        }
        let mut orphans = Vec::new();
        for (bi, block) in msg.content.iter().enumerate() {
            let ContentBlock::ToolCall { id, name, .. } = block else {
                continue;
            };
            let matched = messages.iter().skip(i + 1).any(|later| {
                later.content.iter().any(|b| {
                    let ContentBlock::ToolResult {
                        tool_call_id,
                        name: result_name,
                        ..
                    } = b
                    else {
                        return false;
                    };
                    match id {
                        Some(call_id) => tool_call_id.as_deref() == Some(call_id),
                        None => result_name.as_deref() == Some(name),
                    }
                })
            });
            if !matched {
                orphans.push(bi);
            }
        }
        if !orphans.is_empty() {
            out.push((i, orphans));
        }
    }
    out
}

/// Builds the `system` wire message inserted at the front of `messages`
/// from `Request.system` (§ 7.1). Text blocks only.
fn build_system_from_request(
    system: &[ContentBlock],
    warnings: &mut Vec<ConversionWarning>,
    log: &mut MergeLog,
) -> Result<Value> {
    for (i, block) in system.iter().enumerate() {
        if !matches!(block, ContentBlock::Text { .. }) {
            return Err(ConversionError::InvalidBlockForRole {
                role: Role::System,
                block: block.kind_name(),
                location: format!("/system/{i}"),
            }
            .into());
        }
    }
    let content = encode_input_content(system, "/messages/0", true, warnings, log)?;
    Ok(json!({"role": "system", "content": content}))
}

/// `true` when a run of blocks is eligible for the string content
/// shorthand: exactly one plain, non-empty `Text` block with no cache hint
/// and no fields in this format's `extra` namespace. (An empty text keeps
/// the array form so the tool-message empty encoding `content: ""` stays
/// unambiguous.)
fn string_shorthand(blocks: &[ContentBlock]) -> Option<&str> {
    if let [
        ContentBlock::Text {
            text,
            cache: None,
            extra,
        },
    ] = blocks
        && !text.is_empty()
        && extra.get(FORMAT).is_none_or(Map::is_empty)
    {
        return Some(text);
    }
    None
}

/// Encodes input-side (system/developer/user) content blocks as the wire
/// `content` value: string shorthand where eligible, otherwise a part
/// array. `allow_images` is `false` for system/developer per § 7.4.
fn encode_input_content(
    blocks: &[ContentBlock],
    msg_ptr: &str,
    system_channel: bool,
    warnings: &mut Vec<ConversionWarning>,
    log: &mut MergeLog,
) -> Result<Value> {
    if let Some(text) = string_shorthand(blocks) {
        return Ok(Value::from(text));
    }
    let mut parts: Vec<Value> = Vec::new();
    for (bi, block) in blocks.iter().enumerate() {
        let part_ptr = format!("{msg_ptr}/content/{}", parts.len());
        match block {
            ContentBlock::Text { text, cache, extra } => {
                let mut part = json!({"type": "text", "text": text});
                apply_breakpoint(&mut part, cache.as_ref(), &part_ptr, warnings);
                extra.merge_into(FORMAT, &mut part, &part_ptr, log);
                parts.push(part);
            }
            ContentBlock::Image {
                source,
                cache,
                extra,
            } if !system_channel => {
                if let Some(part) =
                    build_image_part(source, cache.as_ref(), extra, &part_ptr, warnings, log)
                {
                    parts.push(part);
                }
            }
            ContentBlock::Opaque { format, value } => {
                if format == FORMAT {
                    parts.push(value.clone());
                } else {
                    warnings.push(warn(
                        WarningCode::OpaqueDropped,
                        part_ptr,
                        format!("opaque block belongs to `{format}` and was dropped"),
                    ));
                }
            }
            other => {
                let role = if system_channel {
                    Role::System
                } else {
                    Role::User
                };
                return Err(ConversionError::InvalidBlockForRole {
                    role,
                    block: other.kind_name(),
                    location: format!("{msg_ptr}/content/{bi}"),
                }
                .into());
            }
        }
    }
    Ok(Value::Array(parts))
}

/// Builds an `image_url` part, or drops the block with a warning when the
/// source has no Chat Completions channel (`FileId`, § 4.3).
fn build_image_part(
    source: &ImageSource,
    cache: Option<&CacheHint>,
    extra: &Extra,
    part_ptr: &str,
    warnings: &mut Vec<ConversionWarning>,
    log: &mut MergeLog,
) -> Option<Value> {
    let url = match source {
        ImageSource::Url(url) => url.clone(),
        ImageSource::Base64 {
            media_type, data, ..
        } => to_data_url(media_type, data),
        ImageSource::FileId(_) => {
            warnings.push(warn(
                WarningCode::ImageSourceUnsupported,
                part_ptr.to_owned(),
                "Chat Completions has no image file-id channel (the `file` content part is a \
                 document channel); the image was dropped",
            ));
            return None;
        }
    };
    let mut part = json!({"type": "image_url", "image_url": {"url": url}});
    apply_breakpoint(&mut part, cache, part_ptr, warnings);
    extra.merge_into(FORMAT, &mut part, part_ptr, log);
    Some(part)
}

/// Adds a `prompt_cache_breakpoint` for a cache hint; the TTL has no OpenAI
/// equivalent and warns (§ 4.8).
fn apply_breakpoint(
    part: &mut Value,
    cache: Option<&CacheHint>,
    part_ptr: &str,
    warnings: &mut Vec<ConversionWarning>,
) {
    let Some(hint) = cache else { return };
    part["prompt_cache_breakpoint"] = json!({"mode": "explicit"});
    if hint.ttl.is_some() {
        warnings.push(warn(
            WarningCode::CacheTtlDropped,
            format!("{part_ptr}/prompt_cache_breakpoint"),
            "per-block cache TTLs have no OpenAI equivalent (the breakpoint itself was kept; \
             set `prompt_cache_options.ttl` via `extra` for a request-level TTL)",
        ));
    }
}

/// Serializes one IR message into wire messages.
fn build_message(
    msg: &Message,
    mi: usize,
    options: &ConvertOptions,
    wire: &mut Vec<Value>,
    pointers: &mut Vec<(String, Role)>,
    warnings: &mut Vec<ConversionWarning>,
    log: &mut MergeLog,
) -> Result<()> {
    // A message holding exactly one Opaque node that is itself a wire
    // message (it has a `role`) re-emits verbatim — the parse-side home for
    // legacy `function` messages and unmodeled dialect roles.
    if let [ContentBlock::Opaque { format, value }] = msg.content.as_slice()
        && format == FORMAT
        && value.get("role").is_some_and(Value::is_string)
    {
        let ptr = format!("/messages/{}", wire.len());
        let mut value = value.clone();
        msg.extra.merge_into(FORMAT, &mut value, &ptr, log);
        wire.push(value);
        pointers.push((ptr, msg.role));
        return Ok(());
    }
    match msg.role {
        Role::System | Role::Developer | Role::User => {
            let (wire_role, ir_role) = match msg.role {
                Role::System => ("system", Role::System),
                Role::Developer if options.downgrade_developer => {
                    warnings.push(warn(
                        WarningCode::RoleDowngraded,
                        format!("/messages/{}", wire.len()),
                        "developer message downgraded to `user` (downgrade_developer)",
                    ));
                    ("user", Role::User)
                }
                Role::Developer => ("developer", Role::Developer),
                _ => ("user", Role::User),
            };
            let ptr = format!("/messages/{}", wire.len());
            let system_channel = matches!(msg.role, Role::System | Role::Developer);
            let content = encode_role_content(msg, mi, &ptr, system_channel, warnings, log)?;
            let mut item = json!({"role": wire_role, "content": content});
            msg.extra.merge_into(FORMAT, &mut item, &ptr, log);
            wire.push(item);
            pointers.push((ptr, ir_role));
        }
        Role::Assistant => {
            let ptr = format!("/messages/{}", wire.len());
            let item = build_assistant_message(msg, mi, options, &ptr, warnings, log)?;
            wire.push(item);
            pointers.push((ptr, Role::Assistant));
        }
        Role::Tool => build_tool_messages(msg, mi, wire, pointers, warnings, log)?,
    }
    Ok(())
}

/// Content encoding for system/developer/user messages, with the § 7.4
/// role-validity errors pointing at the IR location.
fn encode_role_content(
    msg: &Message,
    mi: usize,
    msg_ptr: &str,
    system_channel: bool,
    warnings: &mut Vec<ConversionWarning>,
    log: &mut MergeLog,
) -> Result<Value> {
    // Re-check role validity with IR-accurate locations before encoding.
    for (bi, block) in msg.content.iter().enumerate() {
        let ok = match block {
            ContentBlock::Text { .. } | ContentBlock::Opaque { .. } => true,
            ContentBlock::Image { .. } => !system_channel,
            _ => false,
        };
        if !ok {
            return Err(ConversionError::InvalidBlockForRole {
                role: msg.role,
                block: block.kind_name(),
                location: format!("/messages/{mi}/content/{bi}"),
            }
            .into());
        }
    }
    encode_input_content(&msg.content, msg_ptr, system_channel, warnings, log)
}

/// Serializes an assistant message: native thinking → `reasoning_content`,
/// text/refusal blocks → `content`, tool calls → `tool_calls[]`.
///
/// The wire message holds one field per channel, so only canonical block
/// order (thinking → content → tool calls) survives a round trip:
/// serializing an interleaved sequence warns `BlockOrderLost` (semantic),
/// and joining several thinking texts into the single `reasoning_content`
/// string warns `ThinkingBlocksJoined` (cosmetic).
fn build_assistant_message(
    msg: &Message,
    mi: usize,
    options: &ConvertOptions,
    msg_ptr: &str,
    warnings: &mut Vec<ConversionWarning>,
    log: &mut MergeLog,
) -> Result<Value> {
    let mut thinking_texts: Vec<String> = Vec::new();
    let mut text_blocks: Vec<&ContentBlock> = Vec::new();
    let mut calls: Vec<Value> = Vec::new();
    let mut thinking_extras: Vec<&Extra> = Vec::new();

    // Channel-order tracking, in canonical wire order. Only blocks that
    // actually reach a wire channel participate: dropped blocks (foreign
    // thinking/opaque, assistant images) carry their own warnings and
    // have no wire position to lose.
    const THINKING: u8 = 0;
    const CONTENT: u8 = 1;
    const TOOL_CALLS: u8 = 2;
    let mut max_channel = THINKING;
    let mut order_lost = false;
    let mut reached = |channel: u8| {
        if channel < max_channel {
            order_lost = true;
        } else {
            max_channel = channel;
        }
    };

    for (bi, block) in msg.content.iter().enumerate() {
        match block {
            ContentBlock::Text { .. } => {
                reached(CONTENT);
                text_blocks.push(block);
            }
            ContentBlock::Opaque { format, .. } => {
                if format == FORMAT {
                    reached(CONTENT);
                }
                text_blocks.push(block);
            }
            ContentBlock::Thinking {
                text,
                signature,
                extra,
            } => {
                let ptr = format!("{msg_ptr}/reasoning_content");
                if is_native_thinking(extra) {
                    if let Some(text) = text {
                        reached(THINKING);
                        thinking_texts.push(text.clone());
                    }
                    if signature.is_some() {
                        warnings.push(warn(
                            WarningCode::ThinkingSignatureDropped,
                            ptr,
                            "`reasoning_content` is a plaintext channel; the thinking signature \
                             was dropped",
                        ));
                    }
                    thinking_extras.push(extra);
                } else if options.thinking_as_text
                    && let Some(text) = text
                {
                    reached(THINKING);
                    thinking_texts.push(text.clone());
                    if signature.is_some() {
                        warnings.push(warn(
                            WarningCode::ThinkingSignatureDropped,
                            ptr,
                            "thinking_as_text re-emitted foreign thinking as `reasoning_content`; \
                             its signature was dropped",
                        ));
                    }
                } else {
                    warnings.push(warn(
                        WarningCode::ThinkingDropped,
                        ptr,
                        "thinking block is not native to `openai_chat_completions` and was \
                         dropped (set `thinking_as_text` to keep its text)",
                    ));
                }
            }
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
                cache,
                extra,
            } => {
                reached(TOOL_CALLS);
                let ptr = format!("{msg_ptr}/tool_calls/{}", calls.len());
                if cache.is_some() {
                    warnings.push(warn(
                        WarningCode::CacheHintDropped,
                        ptr.clone(),
                        "cache breakpoints exist only on content parts; the hint on this tool \
                         call was dropped",
                    ));
                }
                calls.push(build_tool_call_entry(
                    id.as_deref(),
                    name,
                    arguments,
                    extra,
                    &ptr,
                    log,
                )?);
            }
            ContentBlock::Image { .. } => {
                warnings.push(warn(
                    WarningCode::ImageSourceUnsupported,
                    format!("{msg_ptr}/content"),
                    "assistant images have no Chat Completions channel and were dropped",
                ));
            }
            ContentBlock::ToolResult { .. } => {
                return Err(ConversionError::InvalidBlockForRole {
                    role: Role::Assistant,
                    block: block.kind_name(),
                    location: format!("/messages/{mi}/content/{bi}"),
                }
                .into());
            }
        }
    }

    if order_lost {
        warnings.push(warn(
            WarningCode::BlockOrderLost,
            msg_ptr.to_owned(),
            "assistant blocks interleave across the reasoning_content / content / \
             tool_calls channels; the wire message holds one field per channel, so \
             parsing it back yields canonical order (thinking, content, tool calls) \
             and the original block order is lost",
        ));
    }
    if thinking_texts.len() > 1 {
        warnings.push(warn(
            WarningCode::ThinkingBlocksJoined,
            format!("{msg_ptr}/reasoning_content"),
            format!(
                "{} thinking texts were joined with \"\\n\\n\" into the single \
                 `reasoning_content` string; block boundaries are lost (order kept)",
                thinking_texts.len()
            ),
        ));
    }

    let mut item = json!({"role": "assistant"});
    if let Some(content) = encode_assistant_content(&text_blocks, msg_ptr, warnings, log) {
        item["content"] = content;
    }
    if !thinking_texts.is_empty() {
        item["reasoning_content"] = Value::from(thinking_texts.join("\n\n"));
    }
    if !calls.is_empty() {
        item["tool_calls"] = Value::Array(calls);
    }
    // Thinking-block extras have no dedicated wire object (the channel is a
    // plain string field); they merge into the containing message so
    // sibling dialect fields (e.g. `reasoning_details`) can be attached.
    for extra in thinking_extras {
        extra.merge_into(FORMAT, &mut item, msg_ptr, log);
    }
    msg.extra.merge_into(FORMAT, &mut item, msg_ptr, log);
    Ok(item)
}

/// Encodes assistant text/refusal/opaque blocks as `content`: string
/// shorthand where eligible, part array otherwise, `None` (field omitted)
/// when there are no content-channel blocks.
fn encode_assistant_content(
    blocks: &[&ContentBlock],
    msg_ptr: &str,
    warnings: &mut Vec<ConversionWarning>,
    log: &mut MergeLog,
) -> Option<Value> {
    if blocks.is_empty() {
        return None;
    }
    if let [ContentBlock::Text { text, cache, extra }] = blocks
        && cache.is_none()
        && !text.is_empty()
        && extra.get(FORMAT).is_none_or(Map::is_empty)
    {
        return Some(Value::from(text.as_str()));
    }
    let mut parts: Vec<Value> = Vec::new();
    for block in blocks {
        let part_ptr = format!("{msg_ptr}/content/{}", parts.len());
        match block {
            ContentBlock::Text { text, cache, extra } => {
                if cache.is_some() {
                    warnings.push(warn(
                        WarningCode::CacheHintDropped,
                        part_ptr.clone(),
                        "assistant output parts have no documented cache-breakpoint channel; \
                         the hint was dropped",
                    ));
                }
                let ns = extra.get(FORMAT).cloned().unwrap_or_default();
                let refusal = ns.get("refusal").and_then(Value::as_bool).unwrap_or(false);
                let mut part = if refusal {
                    json!({"type": "refusal", "refusal": text})
                } else {
                    json!({"type": "text", "text": text})
                };
                let mut patch = ns;
                patch.remove("refusal");
                crate::ir::merge_patch(&mut part, &patch, &part_ptr, log);
                parts.push(part);
            }
            ContentBlock::Opaque { format, value } => {
                if format == FORMAT {
                    parts.push(value.clone());
                } else {
                    warnings.push(warn(
                        WarningCode::OpaqueDropped,
                        part_ptr,
                        format!("opaque block belongs to `{format}` and was dropped"),
                    ));
                }
            }
            _ => unreachable!("caller collects only text-channel blocks"),
        }
    }
    Some(Value::Array(parts))
}

/// § 4.4 / contract provenance for the plaintext `reasoning_content`
/// channel: a thinking block is native to Chat Completions iff its `extra`
/// has this format's namespace, or it has no format namespace at all
/// (plaintext thinking is native here; a bare signature rides along
/// optimistically but has no channel and is dropped with a warning).
fn is_native_thinking(extra: &Extra) -> bool {
    let has_own = extra.get(FORMAT).is_some_and(|ns| !ns.is_empty());
    if has_own {
        return true;
    }
    !extra
        .formats()
        .any(|f| extra.get(f).is_some_and(|ns| !ns.is_empty()))
}

/// Builds one `tool_calls[]` entry. The reserved `type` key of the block's
/// format namespace selects the payload shape (see
/// [`super::tool_call_reserved_key`]): only the strings `"function"` /
/// `"custom"` (or an absent / explicit-`null` key — the canonical
/// `function` default) rebuild a payload object from the unified fields;
/// every other value takes the verbatim mirror path.
fn build_tool_call_entry(
    id: Option<&str>,
    name: &str,
    arguments: &str,
    extra: &Extra,
    ptr: &str,
    log: &mut MergeLog,
) -> Result<Value> {
    let call_type = extra
        .get(FORMAT)
        .and_then(|ns| ns.get(tool_call_reserved_key::TYPE))
        // An explicit `null` counts as absent (the canonical form).
        .filter(|t| !t.is_null());
    let mut entry = match call_type.map(Value::as_str) {
        None | Some(Some("function")) => {
            let id = require_call_id(id, name, ptr)?;
            json!({
                "id": id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            })
        }
        Some(Some("custom")) => {
            let id = require_call_id(id, name, ptr)?;
            json!({
                "id": id,
                "type": "custom",
                "custom": {"name": name, "input": arguments},
            })
        }
        // Unknown call kinds and non-string `type` values were mirrored
        // wholesale into the namespace at parse time; the merge below
        // restores them (`id` included, when one existed) without
        // fabricating a payload object.
        Some(_) => match id {
            Some(id) => json!({"id": id}),
            None => json!({}),
        },
    };
    extra.merge_into(FORMAT, &mut entry, ptr, log);
    Ok(entry)
}

fn require_call_id(id: Option<&str>, name: &str, ptr: &str) -> Result<String> {
    id.map(str::to_owned).ok_or_else(|| {
        ConversionError::missing(
            format!("tool call `{name}` requires an `id` on the Chat Completions API"),
            ptr.to_owned(),
        )
        .into()
    })
}

/// Serializes a `Tool` message: one `role: "tool"` wire message per
/// `ToolResult` block (§ 7.2). Message-level extra merges into the first
/// produced wire message (implementation contract).
fn build_tool_messages(
    msg: &Message,
    mi: usize,
    wire: &mut Vec<Value>,
    pointers: &mut Vec<(String, Role)>,
    warnings: &mut Vec<ConversionWarning>,
    log: &mut MergeLog,
) -> Result<()> {
    let first = wire.len();
    for (bi, block) in msg.content.iter().enumerate() {
        let ptr = format!("/messages/{}", wire.len());
        match block {
            ContentBlock::ToolResult {
                tool_call_id,
                name,
                content,
                is_error,
                cache,
                extra,
            } => {
                let call_id = tool_call_id.clone().ok_or_else(|| {
                    ConversionError::missing(
                        "tool result requires `tool_call_id` on the Chat Completions API",
                        ptr.clone(),
                    )
                })?;
                if is_error == &Some(true) {
                    warnings.push(warn(
                        WarningCode::IsErrorDropped,
                        ptr.clone(),
                        "`is_error` is native to Anthropic and Google; the error marker was \
                         dropped",
                    ));
                }
                if cache.is_some() {
                    warnings.push(warn(
                        WarningCode::CacheHintDropped,
                        ptr.clone(),
                        "cache breakpoints exist only on content parts; the hint on this tool \
                         result was dropped",
                    ));
                }
                let content = build_tool_content(content, &ptr, warnings, log);
                let mut item = json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content,
                });
                if let Some(name) = name {
                    item["name"] = Value::from(name.clone());
                }
                extra.merge_into(FORMAT, &mut item, &ptr, log);
                wire.push(item);
                pointers.push((ptr, Role::Tool));
            }
            // Tool-role Opaque nodes are whole wire messages (legacy
            // `function` messages parse this way).
            ContentBlock::Opaque { format, value } => {
                if format == FORMAT {
                    wire.push(value.clone());
                    pointers.push((ptr, Role::Tool));
                } else {
                    warnings.push(warn(
                        WarningCode::OpaqueDropped,
                        ptr,
                        format!("opaque block belongs to `{format}` and was dropped"),
                    ));
                }
            }
            other => {
                return Err(ConversionError::InvalidBlockForRole {
                    role: Role::Tool,
                    block: other.kind_name(),
                    location: format!("/messages/{mi}/content/{bi}"),
                }
                .into());
            }
        }
    }
    if wire.len() > first {
        let ptr = format!("/messages/{first}");
        msg.extra.merge_into(FORMAT, &mut wire[first], &ptr, log);
    }
    Ok(())
}

/// Encodes `ToolResult.content` as the tool message `content`: the empty
/// list becomes `""`, a single plain non-empty text block uses the string
/// shorthand, anything else becomes a text-part array (§ 7.2). Chat
/// Completions tool messages are text-only: images drop with a semantic
/// warning (§ 4.5); nested cache hints drop with a cosmetic warning
/// (§ 4.8, v1 rule).
fn build_tool_content(
    content: &[ToolOutputBlock],
    msg_ptr: &str,
    warnings: &mut Vec<ConversionWarning>,
    log: &mut MergeLog,
) -> Value {
    let nested_hint = |warnings: &mut Vec<ConversionWarning>, ptr: String| {
        warnings.push(warn(
            WarningCode::CacheHintDropped,
            ptr,
            "cache hints on nested tool-output blocks are dropped on every target in v1",
        ));
    };
    if content.is_empty() {
        return Value::from("");
    }
    if let [ToolOutputBlock::Text { text, cache, extra }] = content
        && !text.is_empty()
        && extra.get(FORMAT).is_none_or(Map::is_empty)
    {
        if cache.is_some() {
            nested_hint(warnings, format!("{msg_ptr}/content"));
        }
        return Value::from(text.clone());
    }
    let mut parts: Vec<Value> = Vec::new();
    for block in content {
        let part_ptr = format!("{msg_ptr}/content/{}", parts.len());
        match block {
            ToolOutputBlock::Text { text, cache, extra } => {
                if cache.is_some() {
                    nested_hint(warnings, part_ptr.clone());
                }
                let mut part = json!({"type": "text", "text": text});
                extra.merge_into(FORMAT, &mut part, &part_ptr, log);
                parts.push(part);
            }
            ToolOutputBlock::Image { .. } => {
                warnings.push(warn(
                    WarningCode::ToolResultImageDropped,
                    part_ptr,
                    "Chat Completions tool messages are text-only; the image was dropped",
                ));
            }
            ToolOutputBlock::Opaque { format, value } => {
                if format == FORMAT {
                    parts.push(value.clone());
                } else {
                    warnings.push(warn(
                        WarningCode::OpaqueDropped,
                        part_ptr,
                        format!("opaque tool-output block belongs to `{format}` and was dropped"),
                    ));
                }
            }
        }
    }
    Value::Array(parts)
}

/// Maps `Reasoning` onto `reasoning_effort` (§ 4.7 table). Chat
/// Completions accepts the full tier set, so no `EffortUnsupported` path
/// exists here; `include_thoughts` has no channel (cosmetic warning), and
/// a non-empty `Reasoning.extra` namespace has no landing spot
/// (`ExtraDropped`, semantic — use `Request.extra` for top-level dialect
/// knobs).
fn build_reasoning(
    reasoning: &Reasoning,
    body: &mut Map<String, Value>,
    warnings: &mut Vec<ConversionWarning>,
) {
    let mut effort: Option<String> = reasoning.effort.as_ref().map(|e| e.as_str().to_owned());
    match (reasoning.enabled, &reasoning.effort) {
        (Some(true), Some(Effort::None)) | (Some(false), Some(_)) => {
            if !(reasoning.enabled == Some(false) && reasoning.effort == Some(Effort::None)) {
                warnings.push(warn(
                    WarningCode::ReasoningConflict,
                    "/reasoning_effort",
                    "`reasoning.enabled` conflicts with `reasoning.effort`; `effort` wins",
                ));
            }
        }
        (Some(false), None) => effort = Some("none".to_owned()),
        _ => {}
    }
    if let Some(effort) = effort {
        body.insert("reasoning_effort".to_owned(), Value::from(effort));
    }
    if reasoning.include_thoughts.is_some() {
        warnings.push(warn(
            WarningCode::IncludeThoughtsUnsupported,
            "/reasoning_effort",
            "`include_thoughts` has no Chat Completions channel and was dropped",
        ));
    }
    if reasoning.extra.get(FORMAT).is_some_and(|ns| !ns.is_empty()) {
        warnings.push(warn(
            WarningCode::ExtraDropped,
            "/reasoning_effort",
            "`Reasoning.extra` has no landing spot: `reasoning_effort` is a plain string; \
             use `Request.extra` for top-level fields",
        ));
    }
}

/// Builds `response_format` (§ 4.9 table). A missing schema name
/// synthesizes `"response"` — the field is required upstream.
fn build_response_format(format: &OutputFormat) -> Value {
    match format {
        OutputFormat::JsonSchema {
            name,
            description,
            schema,
            strict,
            ..
        } => {
            let mut inner = json!({
                "name": name.clone().unwrap_or_else(|| "response".to_owned()),
                "schema": schema.clone(),
            });
            if let Some(description) = description {
                inner["description"] = Value::from(description.clone());
            }
            if let Some(strict) = strict {
                inner["strict"] = Value::from(*strict);
            }
            json!({"type": "json_schema", "json_schema": inner})
        }
        OutputFormat::JsonObject => json!({"type": "json_object"}),
    }
}

/// Builds the nested Chat Completions tool array (§ 4.5): `{type:
/// "function", function: {name, description, parameters, strict}}`, with
/// `parameters` omitted when unset (officially "an empty parameter list").
fn build_tools(
    tools: &[Tool],
    warnings: &mut Vec<ConversionWarning>,
    log: &mut MergeLog,
) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    for tool in tools {
        let ptr = format!("/tools/{}", out.len());
        match tool {
            Tool::Function(f) => {
                let mut function = Map::new();
                function.insert("name".to_owned(), Value::from(f.name.clone()));
                if let Some(description) = &f.description {
                    function.insert("description".to_owned(), Value::from(description.clone()));
                }
                if let Some(parameters) = &f.parameters {
                    function.insert("parameters".to_owned(), parameters.clone());
                }
                if let Some(strict) = f.strict {
                    function.insert("strict".to_owned(), Value::from(strict));
                }
                if f.cache.is_some() {
                    warnings.push(warn(
                        WarningCode::CacheHintDropped,
                        ptr.clone(),
                        "tool-definition cache hints are Anthropic-only and were dropped",
                    ));
                }
                let mut value = json!({"type": "function", "function": Value::Object(function)});
                f.extra.merge_into(FORMAT, &mut value, &ptr, log);
                out.push(value);
            }
            Tool::Opaque { format, value } => {
                if format == FORMAT {
                    out.push(value.clone());
                } else {
                    warnings.push(warn(
                        WarningCode::OpaqueDropped,
                        ptr,
                        format!("opaque tool belongs to `{format}` and was dropped"),
                    ));
                }
            }
        }
    }
    out
}

/// Builds `tool_choice` (§ 4.5 table).
fn build_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => Value::from("auto"),
        ToolChoice::None => Value::from("none"),
        ToolChoice::Required => Value::from("required"),
        ToolChoice::Tool { name, .. } => {
            json!({"type": "function", "function": {"name": name}})
        }
    }
}
