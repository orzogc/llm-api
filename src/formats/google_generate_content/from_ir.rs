//! IR → Google `generateContent` request JSON (build side).
//!
//! Implements the § 6 build pipeline steps 1–3: wire conversion with
//! bottom-up `extra` merges, warning collection with JSON-Pointer locations
//! into the final body, and the § 7 message rules (system hoisting, role
//! downgrades, mandatory user/model turn merging, orphan-tool-call policies,
//! thinking provenance).

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use crate::convert::{ConversionWarning, ConvertOptions, OrphanToolCalls, WarningCode};
use crate::error::{ConversionError, Result};
use crate::format::ids::GOOGLE_GENERATE_CONTENT as ID;
use crate::ir::{
    ContentBlock, Effort, Extra, ImageSource, MergeLog, Message, OutputFormat, Request, Role, Tool,
    ToolChoice, ToolOutputBlock,
};

/// Output of the build pipeline before the strict gate and hooks
/// ([`crate::format::finalize_request`] applies those).
pub(crate) struct BuiltBody {
    /// The request body after all `extra` merges.
    pub body: Value,
    /// Build-side warnings (not yet marked `overridden`).
    pub warnings: Vec<ConversionWarning>,
    /// Log of every `extra` merge, for `overridden` marking.
    pub merge_log: MergeLog,
    /// `(JSON pointer, IR role)` per serialized wire message, in order.
    pub message_pointers: Vec<(String, Role)>,
}

/// Converts an IR request into the Google request JSON, applying the `extra`
/// merge, marking overridden warnings and running the strict gate — the pure
/// typed-layer entry point (no hooks, no URL handling).
pub fn request_from_ir(
    req: &Request,
    options: &ConvertOptions,
) -> Result<(Value, Vec<ConversionWarning>)> {
    let BuiltBody {
        mut body,
        mut warnings,
        merge_log,
        message_pointers,
    } = build_body(req, options)?;
    crate::format::finalize_request(
        &mut body,
        &mut warnings,
        &merge_log,
        options.strict,
        &crate::convert::RequestHooks::default(),
        &message_pointers,
    )?;
    Ok((body, warnings))
}

/// Which wire role a group of adjacent IR messages lands in.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    /// `role: "user"` — `User`/`Tool` messages and downgraded
    /// `System`/`Developer` messages.
    User,
    /// `role: "model"` — `Assistant` messages.
    Model,
}

/// Deferred warning whose final location depends on the assembled body.
struct PendingWarning {
    code: WarningCode,
    /// Index into the (post-policy) message vector, when the warning targets
    /// a message that may still be serialized; `None` falls back to
    /// `/contents`.
    message_index: Option<usize>,
    text: String,
}

fn warn(
    warnings: &mut Vec<ConversionWarning>,
    code: WarningCode,
    location: impl Into<String>,
    message: impl Into<String>,
) {
    warnings.push(ConversionWarning::to_format(code, ID, location, message));
}

/// Builds the request body per the § 6 pipeline (steps 1–3).
pub(crate) fn build_body(req: &Request, options: &ConvertOptions) -> Result<BuiltBody> {
    let mut warnings: Vec<ConversionWarning> = Vec::new();
    let mut pending: Vec<PendingWarning> = Vec::new();
    let mut log = MergeLog::new();

    let mut messages = req.messages.clone();
    apply_orphan_policy(&mut messages, options.orphan_tool_calls, &mut pending);
    apply_missing_thinking(&mut messages, req, options, &mut pending);

    let mut body = Map::new();

    // ---- systemInstruction: Request.system + leading in-array system run.
    let mut sys_parts: Vec<Value> = Vec::new();
    if let Some(system) = &req.system {
        for (bi, block) in system.iter().enumerate() {
            match block {
                ContentBlock::Text {
                    text, cache, extra, ..
                } => {
                    let ptr = format!("/systemInstruction/parts/{}", sys_parts.len());
                    if cache.is_some() {
                        warn(
                            &mut warnings,
                            WarningCode::CacheHintDropped,
                            &ptr,
                            "Google has no cache breakpoint channel; the hint was dropped",
                        );
                    }
                    let mut part = json!({"text": text});
                    extra.merge_into(ID, &mut part, &ptr, &mut log);
                    sys_parts.push(part);
                }
                other => {
                    return Err(ConversionError::InvalidBlockForRole {
                        role: Role::System,
                        block: other.kind_name(),
                        location: format!("/system/{bi}"),
                    }
                    .into());
                }
            }
        }
    }
    let mut hoisted = 0;
    while hoisted < messages.len() && messages[hoisted].role == Role::System {
        let msg = &messages[hoisted];
        for (bi, block) in msg.content.iter().enumerate() {
            serialize_system_block(block, hoisted, bi, &mut sys_parts, &mut warnings, &mut log)?;
        }
        hoisted += 1;
    }
    let mut system_instruction = json!({ "parts": sys_parts });
    for msg in &messages[..hoisted] {
        msg.extra
            .merge_into(ID, &mut system_instruction, "/systemInstruction", &mut log);
    }
    // Emit systemInstruction unless it is exactly the empty default form.
    let keep_system = system_instruction.as_object().is_some_and(|o| {
        !(o.len() == 1
            && o.get("parts")
                .and_then(Value::as_array)
                .is_some_and(Vec::is_empty))
    });

    // ---- contents: group adjacent user-side / model-side messages into
    // alternating wire turns (Google requires alternation, § 7.2).
    let mut contents: Vec<Value> = Vec::new();
    let mut sides: Vec<Side> = Vec::new();
    let mut pointers: Vec<(String, Role)> = Vec::new();
    let mut msg_to_content: HashMap<usize, usize> = HashMap::new();
    // Tool-call names seen earlier in the request, for `functionResponse.name`
    // resolution (§ 7.2).
    let mut call_names: HashMap<String, String> = HashMap::new();

    for (mi, msg) in messages.iter().enumerate().skip(hoisted) {
        let (side, downgraded) = match msg.role {
            Role::Assistant => (Side::Model, false),
            Role::User | Role::Tool => (Side::User, false),
            Role::System | Role::Developer => (Side::User, true),
        };
        if sides.last() != Some(&side) {
            let role = match side {
                Side::User => "user",
                Side::Model => "model",
            };
            contents.push(json!({"role": role, "parts": []}));
            sides.push(side);
            pointers.push((
                format!("/contents/{}", contents.len() - 1),
                match side {
                    Side::User => Role::User,
                    Side::Model => Role::Assistant,
                },
            ));
        }
        let ci = contents.len() - 1;
        msg_to_content.insert(mi, ci);
        if downgraded {
            warn(
                &mut warnings,
                WarningCode::RoleDowngraded,
                format!("/contents/{ci}"),
                format!(
                    "Google has no {} role; the message was downgraded to user",
                    msg.role
                ),
            );
        }
        serialize_message_blocks(
            msg,
            mi,
            ci,
            &mut contents[ci],
            options,
            &mut call_names,
            &mut warnings,
            &mut log,
        )?;
        let base = format!("/contents/{ci}");
        msg.extra.merge_into(ID, &mut contents[ci], &base, &mut log);
    }
    body.insert("contents".to_owned(), Value::Array(contents));

    // ---- tools + toolConfig.
    if let Some(tools) = &req.tools {
        let entries = serialize_tools(tools, &mut warnings, &mut log)?;
        body.insert("tools".to_owned(), Value::Array(entries));
    }
    if let Some(choice) = &req.tool_choice {
        let mut fcc = Map::new();
        let mode = match choice {
            ToolChoice::Auto => "AUTO",
            ToolChoice::None => "NONE",
            ToolChoice::Required | ToolChoice::Tool { .. } => "ANY",
        };
        fcc.insert("mode".to_owned(), json!(mode));
        if let ToolChoice::Tool { name, .. } = choice {
            fcc.insert("allowedFunctionNames".to_owned(), json!([name]));
        }
        body.insert(
            "toolConfig".to_owned(),
            json!({"functionCallingConfig": Value::Object(fcc)}),
        );
    }
    if let Some(parallel) = req.parallel_tool_calls {
        let has_tools = req.tools.as_ref().is_some_and(|t| !t.is_empty());
        if !has_tools {
            warn(
                &mut warnings,
                WarningCode::ParallelToolCallsIgnored,
                "/toolConfig",
                "parallel_tool_calls is meaningless without tools and was not emitted",
            );
        } else if parallel {
            warn(
                &mut warnings,
                WarningCode::ParallelToolCallsIgnored,
                "/toolConfig",
                "Google has no parallel_tool_calls channel; parallel calling is its default",
            );
        } else {
            warn(
                &mut warnings,
                WarningCode::SerialToolCallsDropped,
                "/toolConfig",
                "Google cannot express parallel_tool_calls: false; the serial-execution constraint was dropped",
            );
        }
    }

    if keep_system {
        body.insert("systemInstruction".to_owned(), system_instruction);
    }

    // ---- generationConfig.
    let generation_config = build_generation_config(req, &mut warnings, &mut log);
    if !generation_config
        .as_object()
        .expect("generationConfig is an object")
        .is_empty()
    {
        body.insert("generationConfig".to_owned(), generation_config);
    }

    // ---- fields with no Google channel.
    if let Some(metadata) = &req.metadata {
        let keys: Vec<&str> = metadata.keys().map(String::as_str).collect();
        warn(
            &mut warnings,
            WarningCode::MetadataDropped,
            "/metadata",
            format!(
                "Google has no request metadata channel; dropped keys: {}",
                keys.join(", ")
            ),
        );
    }
    if req.cache_key.is_some() {
        warn(
            &mut warnings,
            WarningCode::CacheKeyDropped,
            "/cacheKey",
            "Google has no prompt cache key channel; cache_key was dropped",
        );
    }

    // ---- request-level extra merge, last (§ 6 pipeline step 2).
    let mut body = Value::Object(body);
    req.extra.merge_into(ID, &mut body, "", &mut log);

    // Resolve deferred warning locations now that contents indexes are known.
    for p in pending {
        let location = p
            .message_index
            .and_then(|mi| msg_to_content.get(&mi))
            .map_or_else(|| "/contents".to_owned(), |ci| format!("/contents/{ci}"));
        warnings.push(ConversionWarning::to_format(p.code, ID, location, p.text));
    }

    Ok(BuiltBody {
        body,
        warnings,
        merge_log: log,
        message_pointers: pointers,
    })
}

/// Serializes one block of a system message into `systemInstruction.parts`.
fn serialize_system_block(
    block: &ContentBlock,
    msg_index: usize,
    block_index: usize,
    sys_parts: &mut Vec<Value>,
    warnings: &mut Vec<ConversionWarning>,
    log: &mut MergeLog,
) -> Result<()> {
    let ptr = format!("/systemInstruction/parts/{}", sys_parts.len());
    match block {
        ContentBlock::Text {
            text, cache, extra, ..
        } => {
            if cache.is_some() {
                warn(
                    warnings,
                    WarningCode::CacheHintDropped,
                    &ptr,
                    "Google has no cache breakpoint channel; the hint was dropped",
                );
            }
            let mut part = json!({"text": text});
            extra.merge_into(ID, &mut part, &ptr, log);
            sys_parts.push(part);
            Ok(())
        }
        ContentBlock::Opaque { format, value, .. } => {
            if format == ID {
                sys_parts.push(value.clone());
            } else {
                warn(
                    warnings,
                    WarningCode::OpaqueDropped,
                    &ptr,
                    format!("opaque block belongs to format `{format}` and was dropped"),
                );
            }
            Ok(())
        }
        other => Err(ConversionError::InvalidBlockForRole {
            role: Role::System,
            block: other.kind_name(),
            location: format!("/messages/{msg_index}/content/{block_index}"),
        }
        .into()),
    }
}

/// `true` when the § 7.4 role × block table allows the combination.
fn block_allowed(role: Role, block: &ContentBlock) -> bool {
    matches!(
        (role, block),
        (_, ContentBlock::Opaque { .. })
            | (
                Role::System | Role::Developer | Role::User | Role::Assistant,
                ContentBlock::Text { .. },
            )
            | (Role::User | Role::Assistant, ContentBlock::Image { .. })
            | (
                Role::Assistant,
                ContentBlock::ToolCall { .. } | ContentBlock::Thinking { .. }
            )
            | (Role::Tool, ContentBlock::ToolResult { .. })
    )
}

/// Serializes the blocks of one IR message into the wire content's `parts`.
#[expect(clippy::too_many_arguments, reason = "internal assembly helper")]
fn serialize_message_blocks(
    msg: &Message,
    msg_index: usize,
    content_index: usize,
    content: &mut Value,
    options: &ConvertOptions,
    call_names: &mut HashMap<String, String>,
    warnings: &mut Vec<ConversionWarning>,
    log: &mut MergeLog,
) -> Result<()> {
    for (bi, block) in msg.content.iter().enumerate() {
        if !block_allowed(msg.role, block) {
            return Err(ConversionError::InvalidBlockForRole {
                role: msg.role,
                block: block.kind_name(),
                location: format!("/messages/{msg_index}/content/{bi}"),
            }
            .into());
        }
        let part_index = content["parts"]
            .as_array()
            .expect("content.parts is an array")
            .len();
        let ptr = format!("/contents/{content_index}/parts/{part_index}");
        if block.cache().is_some() {
            warn(
                warnings,
                WarningCode::CacheHintDropped,
                &ptr,
                "Google has no cache breakpoint channel; the hint was dropped",
            );
        }
        match block {
            ContentBlock::Text { text, extra, .. } => {
                let mut part = json!({"text": text});
                extra.merge_into(ID, &mut part, &ptr, log);
                content["parts"].as_array_mut().expect("parts").push(part);
            }
            ContentBlock::Image { source, extra, .. } => {
                let mut part = serialize_image(source, &ptr, warnings);
                extra.merge_into(ID, &mut part, &ptr, log);
                content["parts"].as_array_mut().expect("parts").push(part);
            }
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
                extra,
                ..
            } => {
                let args: Value = serde_json::from_str(arguments).map_err(|e| {
                    ConversionError::InvalidToolArguments {
                        name: name.clone(),
                        arguments: arguments.clone(),
                        reason: e.to_string(),
                        location: format!("{ptr}/functionCall/args"),
                    }
                })?;
                if !args.is_object() {
                    return Err(ConversionError::InvalidToolArguments {
                        name: name.clone(),
                        arguments: arguments.clone(),
                        reason: "not a JSON object".to_owned(),
                        location: format!("{ptr}/functionCall/args"),
                    }
                    .into());
                }
                if let Some(id) = id {
                    call_names.insert(id.clone(), name.clone());
                }
                let mut fc = Map::new();
                if let Some(id) = id {
                    fc.insert("id".to_owned(), json!(id));
                }
                fc.insert("name".to_owned(), json!(name));
                fc.insert("args".to_owned(), args);
                let mut part = json!({"functionCall": Value::Object(fc)});
                extra.merge_into(ID, &mut part, &ptr, log);
                content["parts"].as_array_mut().expect("parts").push(part);
            }
            ContentBlock::Thinking {
                text,
                signature,
                extra,
                ..
            } => {
                if let Some(mut part) =
                    serialize_thinking(text, signature, extra, options, &ptr, warnings)
                {
                    extra.merge_into(ID, &mut part, &ptr, log);
                    content["parts"].as_array_mut().expect("parts").push(part);
                }
            }
            ContentBlock::ToolResult {
                tool_call_id,
                name,
                content: result,
                is_error,
                extra,
                ..
            } => {
                let mut part = serialize_tool_result(
                    tool_call_id.as_deref(),
                    name.as_deref(),
                    result,
                    *is_error,
                    call_names,
                    &ptr,
                    warnings,
                    log,
                )?;
                extra.merge_into(ID, &mut part, &ptr, log);
                content["parts"].as_array_mut().expect("parts").push(part);
            }
            ContentBlock::Opaque { format, value, .. } => {
                if format == ID {
                    content["parts"]
                        .as_array_mut()
                        .expect("parts")
                        .push(value.clone());
                } else {
                    warn(
                        warnings,
                        WarningCode::OpaqueDropped,
                        &ptr,
                        format!("opaque block belongs to format `{format}` and was dropped"),
                    );
                }
            }
        }
    }
    Ok(())
}

/// Maps an image source per the § 4.3 table.
fn serialize_image(
    source: &ImageSource,
    ptr: &str,
    warnings: &mut Vec<ConversionWarning>,
) -> Value {
    match source {
        ImageSource::Url(url) => {
            warn(
                warnings,
                WarningCode::ImageUrlAsFileUri,
                format!("{ptr}/fileData/fileUri"),
                "URL mapped to fileData.fileUri verbatim; documented URIs are Files API ones and arbitrary URLs may be rejected upstream",
            );
            json!({"fileData": {"fileUri": url}})
        }
        ImageSource::Base64 {
            media_type, data, ..
        } => {
            json!({"inlineData": {"mimeType": media_type, "data": data}})
        }
        ImageSource::FileId(id) => json!({"fileData": {"fileUri": id}}),
    }
}

/// Serializes a thinking block per the § 4.4 provenance rules; `None` means
/// the block was dropped (with a warning).
fn serialize_thinking(
    text: &Option<String>,
    signature: &Option<String>,
    extra: &Extra,
    options: &ConvertOptions,
    ptr: &str,
    warnings: &mut Vec<ConversionWarning>,
) -> Option<Value> {
    let has_any_namespace = extra.formats().next().is_some();
    let native = extra.get(ID).is_some() || (signature.is_some() && !has_any_namespace);
    if native {
        let mut part = Map::new();
        part.insert("text".to_owned(), json!(text.clone().unwrap_or_default()));
        part.insert("thought".to_owned(), json!(true));
        if let Some(sig) = signature {
            part.insert("thoughtSignature".to_owned(), json!(sig));
        }
        return Some(Value::Object(part));
    }
    if options.thinking_as_text
        && let Some(text) = text
    {
        if signature.is_some() {
            warn(
                warnings,
                WarningCode::ThinkingSignatureDropped,
                ptr,
                "thinking_as_text re-encoded a foreign thinking block; its signature was dropped",
            );
        }
        return Some(json!({"text": text, "thought": true}));
    }
    warn(
        warnings,
        WarningCode::ThinkingDropped,
        ptr,
        "thinking block from another provider was dropped (enable thinking_as_text to re-encode plaintext thinking)",
    );
    None
}

/// Serializes a tool result into a `functionResponse` part (§ 7.2).
///
/// Text goes into `response` (`{"output": …}`, or `{"error": …}` when
/// `is_error` — the documented failure key), base64 images into `parts[]`.
#[expect(clippy::too_many_arguments, reason = "internal assembly helper")]
fn serialize_tool_result(
    tool_call_id: Option<&str>,
    name: Option<&str>,
    result: &[ToolOutputBlock],
    is_error: Option<bool>,
    call_names: &HashMap<String, String>,
    ptr: &str,
    warnings: &mut Vec<ConversionWarning>,
    log: &mut MergeLog,
) -> Result<Value> {
    let resolved = name
        .map(str::to_owned)
        .or_else(|| tool_call_id.and_then(|id| call_names.get(id).cloned()));
    let Some(fn_name) = resolved else {
        return Err(ConversionError::MissingRequired {
            what: "functionResponse.name (set ToolResult.name or give the result the id of an earlier tool call)"
                .to_owned(),
            location: format!("{ptr}/functionResponse/name"),
        }
        .into());
    };

    let mut texts: Vec<&str> = Vec::new();
    let mut media: Vec<Value> = Vec::new();
    let mut order_lost = false;
    let mut seen_media = false;
    for block in result {
        match block {
            ToolOutputBlock::Text {
                text, cache, extra, ..
            } => {
                if cache.is_some() {
                    warn(
                        warnings,
                        WarningCode::CacheHintDropped,
                        ptr,
                        "cache hints on nested tool-output blocks are dropped on every target in v1",
                    );
                }
                if extra.get(ID).is_some_and(|ns| !ns.is_empty()) {
                    warn(
                        warnings,
                        WarningCode::ExtraDropped,
                        format!("{ptr}/functionResponse/response"),
                        "`extra` on a nested tool-output text block has no wire location on Google (text is flattened into `response`)",
                    );
                }
                if seen_media {
                    order_lost = true;
                }
                texts.push(text);
            }
            ToolOutputBlock::Image {
                source,
                cache,
                extra,
                ..
            } => {
                if cache.is_some() {
                    warn(
                        warnings,
                        WarningCode::CacheHintDropped,
                        ptr,
                        "cache hints on nested tool-output blocks are dropped on every target in v1",
                    );
                }
                let media_ptr = format!("{ptr}/functionResponse/parts/{}", media.len());
                match source {
                    ImageSource::Base64 {
                        media_type, data, ..
                    } => {
                        let mut frp = json!({"inlineData": {"mimeType": media_type, "data": data}});
                        extra.merge_into(ID, &mut frp, &media_ptr, log);
                        media.push(frp);
                        seen_media = true;
                    }
                    ImageSource::Url(_) | ImageSource::FileId(_) => {
                        warn(
                            warnings,
                            WarningCode::ToolResultImageDropped,
                            media_ptr,
                            "Google functionResponse media accepts only inline base64 (inlineData); URL/file-id images have no zero-IO channel and were dropped",
                        );
                    }
                }
            }
            ToolOutputBlock::Opaque { format, value, .. } => {
                let media_ptr = format!("{ptr}/functionResponse/parts/{}", media.len());
                if format == ID {
                    media.push(value.clone());
                    seen_media = true;
                } else {
                    warn(
                        warnings,
                        WarningCode::OpaqueDropped,
                        media_ptr,
                        format!(
                            "opaque tool-output block belongs to format `{format}` and was dropped"
                        ),
                    );
                }
            }
        }
    }
    if order_lost {
        warn(
            warnings,
            WarningCode::ToolResultOrderLost,
            ptr,
            "Google splits tool-result media (parts[]) from text (response); text following media cannot keep its position",
        );
    }
    let joined = match texts.len() {
        0 => None,
        1 => Some(texts[0].to_owned()),
        _ => {
            warn(
                warnings,
                WarningCode::ToolResultTextJoined,
                format!("{ptr}/functionResponse/response/output"),
                "Google flattens tool-result text into one string; multiple text blocks were joined with \\n\\n",
            );
            Some(texts.join("\n\n"))
        }
    };
    // `is_error: Some(false)` is the default reading of a function response
    // and canonicalizes to the plain output encoding.
    let response = if is_error == Some(true) {
        json!({"error": joined.unwrap_or_default()})
    } else {
        match joined {
            Some(text) => json!({"output": text}),
            None => json!({}),
        }
    };

    let mut fr = Map::new();
    if let Some(id) = tool_call_id {
        fr.insert("id".to_owned(), json!(id));
    }
    fr.insert("name".to_owned(), json!(fn_name));
    fr.insert("response".to_owned(), response);
    if !media.is_empty() {
        fr.insert("parts".to_owned(), Value::Array(media));
    }
    Ok(json!({"functionResponse": Value::Object(fr)}))
}

/// Serializes the IR tool list. Function tools collapse into a single
/// `functionDeclarations` entry placed at the position of the first function
/// tool; opaque Google tools keep their own entries in order.
fn serialize_tools(
    tools: &[Tool],
    warnings: &mut Vec<ConversionWarning>,
    log: &mut MergeLog,
) -> Result<Vec<Value>> {
    let mut entries: Vec<Value> = Vec::new();
    let mut decl_entry: Option<usize> = None;
    for tool in tools {
        match tool {
            Tool::Function(ft) => {
                let ei = match decl_entry {
                    Some(ei) => ei,
                    None => {
                        entries.push(json!({"functionDeclarations": []}));
                        let ei = entries.len() - 1;
                        decl_entry = Some(ei);
                        ei
                    }
                };
                let di = entries[ei]["functionDeclarations"]
                    .as_array()
                    .expect("declarations array")
                    .len();
                let ptr = format!("/tools/{ei}/functionDeclarations/{di}");
                if ft.strict.is_some() {
                    warn(
                        warnings,
                        WarningCode::StrictUnsupported,
                        &ptr,
                        "Google has no per-tool strict flag; schema adherence cannot be requested",
                    );
                }
                if ft.cache.is_some() {
                    warn(
                        warnings,
                        WarningCode::CacheHintDropped,
                        &ptr,
                        "Google tool declarations have no cache_control equivalent",
                    );
                }
                let mut decl = Map::new();
                decl.insert("name".to_owned(), json!(ft.name));
                if let Some(description) = &ft.description {
                    decl.insert("description".to_owned(), json!(description));
                }
                if let Some(parameters) = &ft.parameters {
                    decl.insert("parametersJsonSchema".to_owned(), parameters.clone());
                }
                let mut decl = Value::Object(decl);
                ft.extra.merge_into(ID, &mut decl, &ptr, log);
                entries[ei]["functionDeclarations"]
                    .as_array_mut()
                    .expect("declarations array")
                    .push(decl);
            }
            Tool::Opaque { format, value, .. } => {
                if format == ID {
                    entries.push(value.clone());
                } else {
                    warn(
                        warnings,
                        WarningCode::OpaqueDropped,
                        format!("/tools/{}", entries.len()),
                        format!("opaque tool belongs to format `{format}` and was dropped"),
                    );
                }
            }
        }
    }
    Ok(entries)
}

/// Builds `generationConfig` from the sampling parameters, structured output
/// and reasoning configuration.
fn build_generation_config(
    req: &Request,
    warnings: &mut Vec<ConversionWarning>,
    log: &mut MergeLog,
) -> Value {
    let mut config = Map::new();
    if let Some(v) = req.max_output_tokens {
        config.insert("maxOutputTokens".to_owned(), json!(v));
    }
    if let Some(v) = req.temperature {
        config.insert("temperature".to_owned(), json!(v));
    }
    if let Some(v) = req.top_p {
        config.insert("topP".to_owned(), json!(v));
    }
    if let Some(v) = req.top_k {
        config.insert("topK".to_owned(), json!(v));
    }
    if let Some(v) = &req.stop_sequences {
        config.insert("stopSequences".to_owned(), json!(v));
    }
    if let Some(v) = req.seed {
        config.insert("seed".to_owned(), json!(v));
    }
    if let Some(v) = req.frequency_penalty {
        config.insert("frequencyPenalty".to_owned(), json!(v));
    }
    if let Some(v) = req.presence_penalty {
        config.insert("presencePenalty".to_owned(), json!(v));
    }

    match &req.output_format {
        Some(OutputFormat::JsonSchema {
            name,
            description,
            schema,
            strict,
            ..
        }) => {
            config.insert("responseMimeType".to_owned(), json!("application/json"));
            config.insert("responseJsonSchema".to_owned(), schema.clone());
            let mut dropped = Vec::new();
            if name.is_some() {
                dropped.push("name");
            }
            if description.is_some() {
                dropped.push("description");
            }
            if strict.is_some() {
                dropped.push("strict");
            }
            if !dropped.is_empty() {
                warn(
                    warnings,
                    WarningCode::OutputFormatDetailDropped,
                    "/generationConfig/responseJsonSchema",
                    format!(
                        "Google responseJsonSchema carries only the schema; dropped: {}",
                        dropped.join(", ")
                    ),
                );
            }
        }
        Some(OutputFormat::JsonObject) => {
            config.insert("responseMimeType".to_owned(), json!("application/json"));
        }
        _ => {}
    }

    if let Some(reasoning) = &req.reasoning {
        let mut tc = Map::new();
        let conflict = match (reasoning.enabled, reasoning.effort.as_ref()) {
            (Some(true), Some(Effort::None)) => true,
            (Some(false), Some(effort)) => *effort != Effort::None,
            _ => false,
        };
        if conflict {
            warn(
                warnings,
                WarningCode::ReasoningConflict,
                "/generationConfig/thinkingConfig",
                "reasoning.enabled and reasoning.effort conflict; effort wins",
            );
        }
        match reasoning.effort.as_ref() {
            Some(Effort::Minimal) => {
                tc.insert("thinkingLevel".to_owned(), json!("MINIMAL"));
            }
            Some(Effort::Low) => {
                tc.insert("thinkingLevel".to_owned(), json!("LOW"));
            }
            Some(Effort::Medium) => {
                tc.insert("thinkingLevel".to_owned(), json!("MEDIUM"));
            }
            Some(Effort::High) => {
                tc.insert("thinkingLevel".to_owned(), json!("HIGH"));
            }
            Some(effort @ (Effort::None | Effort::XHigh | Effort::Max)) => {
                warn(
                    warnings,
                    WarningCode::EffortUnsupported,
                    "/generationConfig/thinkingConfig",
                    format!(
                        "Google thinkingLevel accepts minimal/low/medium/high; `{effort}` was dropped"
                    ),
                );
            }
            Some(Effort::Other(s)) => {
                tc.insert("thinkingLevel".to_owned(), json!(s));
            }
            None => {
                if reasoning.enabled == Some(false) {
                    warn(
                        warnings,
                        WarningCode::ReasoningDisableUnsupported,
                        "/generationConfig/thinkingConfig",
                        "Google thinkingConfig cannot disable thinking (thinkingBudget: 0 via extra may work on some models)",
                    );
                }
                // enabled: true / unset is a no-op on Google.
            }
        }
        if let Some(include) = reasoning.include_thoughts {
            tc.insert("includeThoughts".to_owned(), json!(include));
        }
        let mut tc = Value::Object(tc);
        reasoning
            .extra
            .merge_into(ID, &mut tc, "/generationConfig/thinkingConfig", log);
        if tc.as_object().is_some_and(|o| !o.is_empty()) {
            config.insert("thinkingConfig".to_owned(), tc);
        }
    }

    Value::Object(config)
}

/// Applies the § 7.3 orphan-tool-call policy in place.
fn apply_orphan_policy(
    messages: &mut Vec<Message>,
    policy: OrphanToolCalls,
    pending: &mut Vec<PendingWarning>,
) {
    if messages.is_empty() {
        return;
    }
    // A call is matched iff a *later* message contains a ToolResult with the
    // same id (or the same name when the call has no id).
    let mut orphans: Vec<(usize, usize)> = Vec::new();
    for mi in 0..messages.len() {
        for (bi, block) in messages[mi].content.iter().enumerate() {
            let ContentBlock::ToolCall { id, name, .. } = block else {
                continue;
            };
            let matched = messages[mi + 1..].iter().any(|m| {
                m.content.iter().any(|b| match b {
                    ContentBlock::ToolResult {
                        tool_call_id,
                        name: rname,
                        ..
                    } => match id {
                        Some(id) => tool_call_id.as_deref() == Some(id),
                        None => rname.as_deref() == Some(name),
                    },
                    _ => false,
                })
            });
            if !matched {
                orphans.push((mi, bi));
            }
        }
    }
    if orphans.is_empty() {
        return;
    }
    let last = messages.len() - 1;
    let mid: Vec<usize> = {
        let mut seen = Vec::new();
        for (mi, _) in orphans.iter().filter(|(mi, _)| *mi != last) {
            if !seen.contains(mi) {
                seen.push(*mi);
            }
        }
        seen
    };
    for mi in mid {
        pending.push(PendingWarning {
            code: WarningCode::OrphanToolCalls,
            message_index: Some(mi),
            text: "assistant message contains tool calls without matching results; mid-conversation orphans are never repaired".to_owned(),
        });
    }
    let trailing: Vec<usize> = orphans
        .iter()
        .filter(|(mi, _)| *mi == last)
        .map(|(_, bi)| *bi)
        .collect();
    if trailing.is_empty() {
        return;
    }
    match policy {
        OrphanToolCalls::DropTrailing => {
            let count = trailing.len();
            let msg = &mut messages[last];
            let mut bi = 0;
            msg.content.retain(|_| {
                let drop = trailing.contains(&bi);
                bi += 1;
                !drop
            });
            let removed_message = msg.content.is_empty();
            let thinking_orphaned = !removed_message
                && msg
                    .content
                    .iter()
                    .any(|b| matches!(b, ContentBlock::Thinking { .. }))
                && !msg
                    .content
                    .iter()
                    .any(|b| matches!(b, ContentBlock::ToolCall { .. }));
            if removed_message {
                messages.pop();
            }
            pending.push(PendingWarning {
                code: WarningCode::OrphanToolCallsDropped,
                message_index: if removed_message { None } else { Some(last) },
                text: format!("removed {count} unmatched trailing tool call(s) as requested"),
            });
            if thinking_orphaned {
                pending.push(PendingWarning {
                    code: WarningCode::ThinkingOrphaned,
                    message_index: Some(last),
                    text: "dropping trailing tool calls left a thinking block without any tool call; the upstream may reject the turn".to_owned(),
                });
            }
        }
        OrphanToolCalls::SynthesizeError => {
            let mut results = Vec::new();
            for bi in &trailing {
                let ContentBlock::ToolCall { id, name, .. } = &messages[last].content[*bi] else {
                    continue;
                };
                results.push(
                    ContentBlock::tool_result_text(id.clone(), "cancelled")
                        .with_tool_name(name.clone())
                        .with_is_error(true),
                );
            }
            let count = results.len();
            messages.push(Message::tool(results));
            pending.push(PendingWarning {
                code: WarningCode::OrphanToolCallsSynthesized,
                message_index: Some(messages.len() - 1),
                text: format!("appended {count} synthetic error tool result(s) as requested"),
            });
        }
        _ => {}
    }
}

/// Applies the § 7.3 missing-thinking check / fill in place.
fn apply_missing_thinking(
    messages: &mut [Message],
    req: &Request,
    options: &ConvertOptions,
    pending: &mut Vec<PendingWarning>,
) {
    let enables = req.reasoning.as_ref().is_some_and(|r| {
        r.enabled == Some(true)
            || (r.enabled.is_none() && r.effort.as_ref().is_some_and(|e| *e != Effort::None))
    });
    if !enables {
        return;
    }
    for (mi, msg) in messages.iter_mut().enumerate() {
        if msg.role != Role::Assistant {
            continue;
        }
        let has_call = msg
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolCall { .. }));
        let has_thinking = msg
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::Thinking { .. }));
        if !has_call || has_thinking {
            continue;
        }
        if let Some(text) = &options.fill_missing_thinking {
            msg.content.insert(0, ContentBlock::thinking(text.clone()));
            pending.push(PendingWarning {
                code: WarningCode::MissingThinkingFilled,
                message_index: Some(mi),
                text: "inserted a placeholder thinking block before the tool calls".to_owned(),
            });
        } else {
            pending.push(PendingWarning {
                code: WarningCode::MissingThinkingWithToolCalls,
                message_index: Some(mi),
                text: "assistant message carries tool calls but no thinking while the request enables thinking; Google requires thought signatures with function calls on thinking models".to_owned(),
            });
        }
    }
}
