//! Google `generateContent` JSON → IR (parse side): request → IR for
//! round-trips and format-to-format conversion, response → IR, plus the
//! part-classification helpers shared with the stream parser.

use serde_json::{Map, Value, json};

use crate::convert::{ConversionWarning, WarningCode};
use crate::error::{Error, Result};
use crate::format::ResponseMeta;
use crate::format::ids::GOOGLE_GENERATE_CONTENT as ID;
use crate::ir::{
    ContentBlock, Effort, Extra, ImageSource, Message, OutputFormat, Reasoning, Request, Response,
    Role, StopReason, Tool, ToolChoice, ToolOutputBlock, Usage, normalize_stop_reason,
};

use super::types;

fn parse_error(message: impl Into<String>, raw: &[u8]) -> Error {
    Error::Parse {
        message: message.into(),
        raw: bytes::Bytes::copy_from_slice(raw),
    }
}

fn warn(
    warnings: &mut Vec<ConversionWarning>,
    code: WarningCode,
    location: impl Into<String>,
    message: impl Into<String>,
) {
    warnings.push(ConversionWarning::from_format(code, ID, location, message));
}

/// Maps a Google `finishReason` onto the unified stop reason (§ 8): `STOP` →
/// `EndTurn`, `MAX_TOKENS` → `MaxTokens`, the safety family (`SAFETY`,
/// `PROHIBITED_CONTENT`, `BLOCKLIST`, `SPII`, `IMAGE_*`) → `ContentFilter`,
/// anything else → `Other` with the original string.
#[must_use]
pub(crate) fn map_finish_reason(reason: &str) -> StopReason {
    match reason {
        "STOP" => StopReason::EndTurn,
        "MAX_TOKENS" => StopReason::MaxTokens,
        "SAFETY" | "PROHIBITED_CONTENT" | "BLOCKLIST" | "SPII" => StopReason::ContentFilter,
        other if other.starts_with("IMAGE_") => StopReason::ContentFilter,
        other => StopReason::Other(other.to_owned()),
    }
}

fn clamp(v: Option<i64>) -> u64 {
    v.map_or(0, |v| u64::try_from(v).unwrap_or(0))
}

fn clamp_opt(v: Option<i64>) -> Option<u64> {
    v.map(|v| u64::try_from(v).unwrap_or(0))
}

/// Converts `usageMetadata` to unified usage (§ 8): `promptTokenCount`
/// already includes cached tokens; `candidatesTokenCount` excludes thoughts,
/// so `output_tokens` adds `thoughtsTokenCount` back in.
#[must_use]
pub(crate) fn usage_from_metadata(u: &types::UsageMetadata) -> Usage {
    let mut usage = Usage {
        input_tokens: clamp(u.prompt_token_count),
        // Saturating so misbehaving provider counts cannot overflow the sum.
        output_tokens: clamp(u.candidates_token_count)
            .saturating_add(clamp(u.thoughts_token_count)),
        total_tokens: clamp_opt(u.total_token_count),
        cache_read_tokens: clamp_opt(u.cached_content_token_count),
        reasoning_tokens: clamp_opt(u.thoughts_token_count),
        ..Usage::default()
    };
    usage.raw = serde_json::to_value(u).ok();
    usage
}

/// A model-side part classified for block building. Text-like parts may
/// continue across stream chunks; everything else arrives whole.
pub(crate) enum AssistantPart {
    /// A `text` part (plain or `thought: true`).
    TextLike {
        /// Whether the part is a thought summary.
        thought: bool,
        /// The text fragment.
        text: String,
        /// Part-level `thoughtSignature`, if any.
        signature: Option<String>,
        /// Unknown part-level fields.
        ns: Map<String, Value>,
    },
    /// A complete block (`functionCall`, media, opaque).
    Complete(ContentBlock),
}

/// Number of `data`-union members set on a part.
fn union_members(part: &types::Part) -> usize {
    usize::from(part.text.is_some())
        + usize::from(part.inline_data.is_some())
        + usize::from(part.function_call.is_some())
        + usize::from(part.function_response.is_some())
        + usize::from(part.file_data.is_some())
}

fn opaque_part(part: &types::Part) -> ContentBlock {
    ContentBlock::opaque(ID, serde_json::to_value(part).expect("part serializes"))
}

/// Builds the block-extra namespace shared by non-text parts: unknown
/// part-level fields plus the `thought` / `thoughtSignature` modifiers.
fn part_ns(part: &types::Part) -> Map<String, Value> {
    let mut ns = part.extra.clone();
    if part.thought == Some(true) {
        ns.insert("thought".to_owned(), json!(true));
    }
    if let Some(sig) = &part.thought_signature {
        ns.insert("thoughtSignature".to_owned(), json!(sig));
    }
    ns
}

/// Builds the `Text` block for a `systemInstruction` text part (wire
/// modifiers such as `thought` ride the block's namespace verbatim).
fn system_text_block(part: &types::Part) -> ContentBlock {
    ContentBlock::Text {
        text: part.text.clone().unwrap_or_default(),
        cache: None,
        extra: Extra::from_unknown(ID, part_ns(part)),
    }
}

/// Classifies a model-content part (§ 8 / § 9 shared logic).
pub(crate) fn classify_assistant_part(
    part: &types::Part,
    location: &str,
    warnings: &mut Vec<ConversionWarning>,
) -> AssistantPart {
    if union_members(part) != 1 {
        // Empty or multi-member parts are malformed unions; keep verbatim.
        return AssistantPart::Complete(opaque_part(part));
    }
    if let Some(fc) = &part.function_call {
        // `name` is required upstream; a missing one degrades to an empty
        // string (which is what re-serialization emits) with a warning.
        let name = fc.name.clone().unwrap_or_else(|| {
            warn(
                warnings,
                WarningCode::MalformedToolCall,
                format!("{location}/functionCall/name"),
                "functionCall.name is missing; treated as an empty string",
            );
            String::new()
        });
        let arguments = match &fc.args {
            None => "{}".to_owned(),
            Some(args @ Value::Object(_)) => {
                serde_json::to_string(args).expect("JSON value serializes")
            }
            Some(other) => {
                warn(
                    warnings,
                    WarningCode::MalformedField,
                    format!("{location}/functionCall/args"),
                    "functionCall.args is not a JSON object",
                );
                serde_json::to_string(other).expect("JSON value serializes")
            }
        };
        let mut ns = part_ns(part);
        if !fc.extra.is_empty() {
            ns.insert("functionCall".to_owned(), Value::Object(fc.extra.clone()));
        }
        return AssistantPart::Complete(ContentBlock::ToolCall {
            id: fc.id.clone(),
            name,
            arguments,
            cache: None,
            extra: Extra::from_unknown(ID, ns),
        });
    }
    if let Some(text) = &part.text {
        return AssistantPart::TextLike {
            thought: part.thought == Some(true),
            text: text.clone(),
            signature: part.thought_signature.clone(),
            ns: part.extra.clone(),
        };
    }
    if let Some(blob) = &part.inline_data {
        return AssistantPart::Complete(image_block(part, blob));
    }
    if let Some(file) = &part.file_data {
        return AssistantPart::Complete(file_image_block(part, file));
    }
    // The remaining single member is `functionResponse`, which is not valid
    // model output; keep it verbatim.
    AssistantPart::Complete(opaque_part(part))
}

/// Builds a `Text` or `Thinking` block from a classified text-like part.
/// `thought: true` parts record the wire marker in the block's namespace so
/// re-serialization treats them as native Google thinking (§ 4.4).
#[must_use]
pub(crate) fn text_like_block(
    thought: bool,
    text: String,
    signature: Option<String>,
    mut ns: Map<String, Value>,
) -> ContentBlock {
    if thought {
        ns.insert("thought".to_owned(), json!(true));
        ContentBlock::Thinking {
            text: Some(text),
            signature,
            extra: Extra::from_unknown(ID, ns),
        }
    } else {
        if let Some(sig) = signature {
            ns.insert("thoughtSignature".to_owned(), json!(sig));
        }
        ContentBlock::Text {
            text,
            cache: None,
            extra: Extra::from_unknown(ID, ns),
        }
    }
}

fn image_block(part: &types::Part, blob: &types::Blob) -> ContentBlock {
    let mut ns = part_ns(part);
    if !blob.extra.is_empty() {
        ns.insert("inlineData".to_owned(), Value::Object(blob.extra.clone()));
    }
    ContentBlock::Image {
        source: ImageSource::base64(blob.mime_type.clone(), blob.data.clone()),
        cache: None,
        extra: Extra::from_unknown(ID, ns),
    }
}

fn file_image_block(part: &types::Part, file: &types::FileData) -> ContentBlock {
    let mut ns = part_ns(part);
    let mut fd = file.extra.clone();
    if let Some(mime) = &file.mime_type {
        fd.insert("mimeType".to_owned(), json!(mime));
    }
    if !fd.is_empty() {
        ns.insert("fileData".to_owned(), Value::Object(fd));
    }
    ContentBlock::Image {
        source: ImageSource::file_id(file.file_uri.clone()),
        cache: None,
        extra: Extra::from_unknown(ID, ns),
    }
}

/// Classifies a user-content part that is not a `functionResponse`.
fn user_block_from_part(part: &types::Part) -> ContentBlock {
    if union_members(part) != 1 {
        return opaque_part(part);
    }
    if let Some(text) = &part.text {
        if part.thought == Some(true) {
            // Thinking is not valid user content (§ 7.4); keep verbatim.
            return opaque_part(part);
        }
        return text_like_block(
            false,
            text.clone(),
            part.thought_signature.clone(),
            part.extra.clone(),
        );
    }
    if let Some(blob) = &part.inline_data {
        return image_block(part, blob);
    }
    if let Some(file) = &part.file_data {
        return file_image_block(part, file);
    }
    // functionCall (invalid in user content) and unknown members stay
    // verbatim.
    opaque_part(part)
}

/// Converts a `functionResponse` part into a `ToolResult` block (§ 7.2).
///
/// Canonical decoding of `response`: an `"error"` key marks the result as an
/// error (`is_error: true`, documented failure key), an `"output"` string
/// becomes text content; all other keys are preserved in the block's `extra`
/// namespace at their mirrored path.
///
/// The upstream-required fields parse leniently when absent — `name` stays
/// `None`, a missing `response` becomes empty content — each with a
/// `MalformedToolResult` warning.
fn tool_result_from_part(
    part: &types::Part,
    location: &str,
    warnings: &mut Vec<ConversionWarning>,
) -> ContentBlock {
    let fr = part
        .function_response
        .as_ref()
        .expect("caller checked functionResponse");
    let mut ns = part_ns(part);
    let mut fr_ns = fr.extra.clone();
    let mut content: Vec<ToolOutputBlock> = Vec::new();
    let mut is_error = None;

    if fr.name.is_none() {
        warn(
            warnings,
            WarningCode::MalformedToolResult,
            format!("{location}/functionResponse/name"),
            "functionResponse.name is missing; the block's name stays unset, so \
             re-serialization needs an id matching an earlier tool call (or fails)",
        );
    }

    match &fr.response {
        Some(Value::Object(obj)) => {
            let mut leftover = obj.clone();
            if let Some(error_value) = leftover.shift_remove("error") {
                is_error = Some(true);
                match error_value {
                    Value::String(text) => content.push(ToolOutputBlock::text(text)),
                    other => {
                        leftover.insert("error".to_owned(), other);
                    }
                }
            } else if matches!(leftover.get("output"), Some(Value::String(_))) {
                let Some(Value::String(text)) = leftover.shift_remove("output") else {
                    unreachable!("checked string above")
                };
                content.push(ToolOutputBlock::text(text));
            }
            if !leftover.is_empty() {
                fr_ns.insert("response".to_owned(), Value::Object(leftover));
            }
        }
        Some(other) => {
            warn(
                warnings,
                WarningCode::MalformedField,
                format!("{location}/functionResponse/response"),
                "functionResponse.response is not a JSON object",
            );
            fr_ns.insert("response".to_owned(), other.clone());
        }
        None => {
            warn(
                warnings,
                WarningCode::MalformedToolResult,
                format!("{location}/functionResponse/response"),
                "functionResponse.response is missing; parsed as empty content, \
                 which re-serializes as an empty object the wire never carried",
            );
        }
    }

    for frp in fr.parts.as_deref().unwrap_or_default() {
        if let Some(blob) = &frp.inline_data {
            let mut img_ns = frp.extra.clone();
            if !blob.extra.is_empty() {
                img_ns.insert("inlineData".to_owned(), Value::Object(blob.extra.clone()));
            }
            content.push(ToolOutputBlock::Image {
                source: ImageSource::base64(blob.mime_type.clone(), blob.data.clone()),
                cache: None,
                extra: Extra::from_unknown(ID, img_ns),
            });
        } else {
            content.push(ToolOutputBlock::opaque(
                ID,
                serde_json::to_value(frp).expect("part serializes"),
            ));
        }
    }

    if !fr_ns.is_empty() {
        ns.insert("functionResponse".to_owned(), Value::Object(fr_ns));
    }
    ContentBlock::ToolResult {
        tool_call_id: fr.id.clone(),
        name: fr.name.clone(),
        content,
        is_error,
        cache: None,
        extra: Extra::from_unknown(ID, ns),
    }
}

/// Splits one wire user content into IR messages: consecutive
/// `functionResponse` parts become a `Tool` message, other runs a `User`
/// message; a mixed turn tags every produced message with the same fresh
/// turn group so serialization can restore the single wire turn (§ 7.2).
fn split_user_content(
    content: &types::Content,
    content_index: usize,
    group: &mut u64,
    warnings: &mut Vec<ConversionWarning>,
) -> Vec<Message> {
    let mut runs: Vec<(bool, Vec<ContentBlock>)> = Vec::new();
    for (pi, part) in content.parts.iter().enumerate() {
        let location = format!("/contents/{content_index}/parts/{pi}");
        let is_tool = part.function_response.is_some() && union_members(part) == 1;
        let block = if is_tool {
            tool_result_from_part(part, &location, warnings)
        } else {
            user_block_from_part(part)
        };
        match runs.last_mut() {
            Some((last_is_tool, blocks)) if *last_is_tool == is_tool => blocks.push(block),
            _ => runs.push((is_tool, vec![block])),
        }
    }
    if runs.is_empty() {
        runs.push((false, Vec::new()));
    }
    let split = runs.len() > 1;
    let g = if split {
        *group += 1;
        *group
    } else {
        0
    };
    let mut messages = Vec::new();
    for (ri, (is_tool, blocks)) in runs.into_iter().enumerate() {
        let mut msg = if is_tool {
            Message::tool(blocks)
        } else {
            Message::user(blocks)
        };
        if split {
            msg = msg.with_turn_group(g);
        }
        if ri == 0 && !content.extra.is_empty() {
            msg.extra = Extra::from_unknown(ID, content.extra.clone());
        }
        messages.push(msg);
    }
    messages
}

/// Parses a model content into one assistant message.
fn assistant_message(
    content: &types::Content,
    content_index: usize,
    warnings: &mut Vec<ConversionWarning>,
) -> Message {
    let mut blocks = Vec::new();
    for (pi, part) in content.parts.iter().enumerate() {
        let location = format!("/contents/{content_index}/parts/{pi}");
        match classify_assistant_part(part, &location, warnings) {
            AssistantPart::TextLike {
                thought,
                text,
                signature,
                ns,
            } => {
                blocks.push(text_like_block(thought, text, signature, ns));
            }
            AssistantPart::Complete(block) => blocks.push(block),
        }
    }
    let mut msg = Message::assistant(blocks);
    if !content.extra.is_empty() {
        msg.extra = Extra::from_unknown(ID, content.extra.clone());
    }
    msg
}

/// Parses a Google request body into the IR (round-trip direction).
pub fn request_to_ir(body: &[u8]) -> Result<(Request, Vec<ConversionWarning>)> {
    let wire: types::GenerateContentRequest = serde_json::from_slice(body)
        .map_err(|e| parse_error(format!("invalid Google request: {e}"), body))?;
    let mut warnings = Vec::new();
    let mut req = Request::new();
    let mut extra_ns = wire.extra;

    // systemInstruction → Request.system (§ 7.1). Request.system is
    // Text-only, so an out-of-schema non-text part forces the *whole*
    // instruction into a leading System message instead (text parts →
    // `Text`, others → same-format `Opaque`, part order kept); § 7.1 hoists
    // leading System messages back into `systemInstruction`, keeping the
    // wire round-trip the identity and losing nothing.
    if let Some(si) = wire.system_instruction {
        let text_only = si
            .parts
            .iter()
            .all(|part| part.text.is_some() && union_members(part) == 1);
        if text_only {
            let system: Vec<ContentBlock> = si.parts.iter().map(system_text_block).collect();
            if !system.is_empty() {
                req.system = Some(system);
            }
            if !si.extra.is_empty() {
                extra_ns.insert("systemInstruction".to_owned(), Value::Object(si.extra));
            }
        } else {
            let blocks: Vec<ContentBlock> = si
                .parts
                .iter()
                .enumerate()
                .map(|(pi, part)| {
                    if part.text.is_some() && union_members(part) == 1 {
                        system_text_block(part)
                    } else {
                        warn(
                            &mut warnings,
                            WarningCode::MalformedField,
                            format!("/systemInstruction/parts/{pi}"),
                            "systemInstruction accepts text parts only; the part was kept as an opaque block and round-trips verbatim",
                        );
                        opaque_part(part)
                    }
                })
                .collect();
            let mut msg = Message::new(Role::System, blocks);
            if !si.extra.is_empty() {
                msg.extra = Extra::from_unknown(ID, si.extra);
            }
            req.messages.push(msg);
        }
    }

    // contents → messages, splitting mixed user turns. The documented role
    // union is closed (`user` / `model`, optional), and non-`model` roles
    // behave as `user` upstream — keep that behavior, warning only when an
    // out-of-schema value was actually present.
    let mut group = 0u64;
    for (ci, content) in wire.contents.iter().enumerate() {
        if content.role.as_deref() == Some("model") {
            req.messages
                .push(assistant_message(content, ci, &mut warnings));
        } else {
            if let Some(role) = content.role.as_deref()
                && role != "user"
            {
                warn(
                    &mut warnings,
                    WarningCode::MalformedField,
                    format!("/contents/{ci}/role"),
                    format!("unknown content role `{role}`; treated as user"),
                );
            }
            req.messages
                .extend(split_user_content(content, ci, &mut group, &mut warnings));
        }
    }

    // tools.
    if let Some(entries) = wire.tools {
        let mut tools = Vec::new();
        for entry in entries {
            if let Some(decls) = entry.function_declarations {
                for decl in decls {
                    tools.push(Tool::Function(crate::ir::FunctionTool {
                        name: decl.name,
                        description: decl.description,
                        parameters: decl.parameters_json_schema,
                        strict: None,
                        cache: None,
                        extra: Extra::from_unknown(ID, decl.extra),
                    }));
                }
            }
            if !entry.extra.is_empty() {
                tools.push(Tool::opaque(ID, Value::Object(entry.extra)));
            }
        }
        req.tools = Some(tools);
    }

    // toolConfig → tool_choice; unmodeled shapes mirror into extra.
    if let Some(tc) = wire.tool_config {
        let mut tc_mirror = tc.extra;
        if let Some(fcc) = tc.function_calling_config {
            let mut fcc_mirror = fcc.extra;
            let allowed = fcc.allowed_function_names;
            match fcc.mode.as_deref() {
                Some("AUTO") if allowed.is_none() => req.tool_choice = Some(ToolChoice::Auto),
                Some("NONE") if allowed.is_none() => req.tool_choice = Some(ToolChoice::None),
                Some("ANY") => match allowed {
                    Some(names) if names.len() == 1 => {
                        req.tool_choice = Some(ToolChoice::tool(names[0].clone()));
                    }
                    Some(names) => {
                        req.tool_choice = Some(ToolChoice::Required);
                        fcc_mirror.insert("allowedFunctionNames".to_owned(), json!(names));
                    }
                    None => req.tool_choice = Some(ToolChoice::Required),
                },
                mode => {
                    // VALIDATED / unknown modes, or a mode with an allowed
                    // list the IR cannot model: mirror verbatim.
                    if let Some(mode) = mode {
                        fcc_mirror.insert("mode".to_owned(), json!(mode));
                    }
                    if let Some(names) = allowed {
                        fcc_mirror.insert("allowedFunctionNames".to_owned(), json!(names));
                    }
                }
            }
            if !fcc_mirror.is_empty() {
                tc_mirror.insert(
                    "functionCallingConfig".to_owned(),
                    Value::Object(fcc_mirror),
                );
            }
        }
        if !tc_mirror.is_empty() {
            extra_ns.insert("toolConfig".to_owned(), Value::Object(tc_mirror));
        }
    }

    // generationConfig.
    if let Some(gc) = wire.generation_config {
        let mut gc_mirror = gc.extra;
        req.max_output_tokens = gc.max_output_tokens;
        req.temperature = gc.temperature;
        req.top_p = gc.top_p;
        req.top_k = gc.top_k;
        req.stop_sequences = gc.stop_sequences;
        req.seed = gc.seed;
        req.frequency_penalty = gc.frequency_penalty;
        req.presence_penalty = gc.presence_penalty;
        match (gc.response_mime_type.as_deref(), gc.response_json_schema) {
            (Some("application/json"), Some(schema)) => {
                req.output_format = Some(OutputFormat::json_schema(schema));
            }
            (Some("application/json"), None) => {
                req.output_format = Some(OutputFormat::json_object());
            }
            (mime, schema) => {
                if let Some(mime) = mime {
                    gc_mirror.insert("responseMimeType".to_owned(), json!(mime));
                }
                if let Some(schema) = schema {
                    gc_mirror.insert("responseJsonSchema".to_owned(), schema);
                }
            }
        }
        if let Some(tc) = gc.thinking_config {
            let mut reasoning = Reasoning::new();
            reasoning.include_thoughts = tc.include_thoughts;
            reasoning.effort = tc.thinking_level.as_deref().map(|level| match level {
                "MINIMAL" => Effort::Minimal,
                "LOW" => Effort::Low,
                "MEDIUM" => Effort::Medium,
                "HIGH" => Effort::High,
                other => Effort::Other(other.to_owned()),
            });
            if reasoning != Reasoning::new() {
                req.reasoning = Some(reasoning);
            }
            if !tc.extra.is_empty() {
                gc_mirror.insert("thinkingConfig".to_owned(), Value::Object(tc.extra));
            }
        }
        if !gc_mirror.is_empty() {
            extra_ns.insert("generationConfig".to_owned(), Value::Object(gc_mirror));
        }
    }

    req.extra = Extra::from_unknown(ID, extra_ns);
    Ok((req, warnings))
}

/// Parses a 2xx Google response body into the unified [`Response`] (§ 8).
pub fn response_to_ir(body: &[u8], meta: &ResponseMeta) -> Result<Response> {
    let raw: Value = serde_json::from_slice(body)
        .map_err(|e| parse_error(format!("invalid Google response: {e}"), body))?;
    let wire: types::GenerateContentResponse = serde_json::from_value(raw.clone())
        .map_err(|e| parse_error(format!("invalid Google response: {e}"), body))?;
    let mut warnings = Vec::new();

    let blocked = wire
        .prompt_feedback
        .as_ref()
        .is_some_and(|pf| pf.block_reason.is_some());
    let (message, stop_reason) = if wire.candidates.is_empty() {
        if !blocked {
            warn(
                &mut warnings,
                WarningCode::MalformedField,
                "/candidates",
                "response carries neither candidates nor a prompt block reason",
            );
        }
        let stop = blocked.then_some(StopReason::ContentFilter);
        (Message::assistant(Vec::new()), stop)
    } else {
        if wire.candidates.len() > 1 {
            warn(
                &mut warnings,
                WarningCode::MultipleCandidates,
                "/candidates",
                format!(
                    "response carries {} candidates; only the first was read (the rest remain in raw)",
                    wire.candidates.len()
                ),
            );
        }
        let candidate = &wire.candidates[0];
        let mut blocks = Vec::new();
        if let Some(content) = &candidate.content {
            for (pi, part) in content.parts.iter().enumerate() {
                let location = format!("/candidates/0/content/parts/{pi}");
                match classify_assistant_part(part, &location, &mut warnings) {
                    AssistantPart::TextLike {
                        thought,
                        text,
                        signature,
                        ns,
                    } => {
                        blocks.push(text_like_block(thought, text, signature, ns));
                    }
                    AssistantPart::Complete(block) => blocks.push(block),
                }
            }
        }
        let stop = candidate.finish_reason.as_deref().map(map_finish_reason);
        (Message::assistant(blocks), stop)
    };

    let stop_reason = normalize_stop_reason(&message, stop_reason);
    let mut response = Response::new(message);
    response.id = wire.response_id;
    response.model = wire.model_version;
    response.stop_reason = stop_reason;
    response.usage = wire.usage_metadata.as_ref().map(usage_from_metadata);
    response.status = meta.status;
    response.headers = meta.headers.clone();
    response.raw = Some(raw);
    response.warnings = warnings;
    Ok(response)
}
