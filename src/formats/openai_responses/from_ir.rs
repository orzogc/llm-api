//! IR → OpenAI Responses request conversion (build side).

use std::borrow::Cow;
use std::collections::BTreeSet;

use serde_json::{Map, Value, json};

use crate::convert::{ConversionWarning, ConvertOptions, OrphanToolCalls, WarningCode};
use crate::error::{ConversionError, Result};
use crate::format::{CallMode, to_data_url};
use crate::ir::{
    CacheHint, ContentBlock, Effort, Extra, ImageSource, MergeLog, Message, OutputFormat,
    Reasoning, Request, Role, Tool, ToolChoice, ToolOutputBlock, escape_pointer_token, merge_patch,
};

use super::FORMAT;
use super::text_block_reserved_key;

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
    /// `(JSON pointer, role)` per serialized `input` item, in order.
    pub messages: Vec<(String, Role)>,
    /// Top-level keys the converter itself generated, snapshotted before
    /// the request-level `extra` merge (token-count adapter input, § 13).
    pub generated_keys: BTreeSet<String>,
}

/// Build-side warning shorthand.
fn warn(
    code: WarningCode,
    location: impl Into<String>,
    message: impl Into<String>,
) -> ConversionWarning {
    ConversionWarning::to_format(code, FORMAT, location, message)
}

/// Converts an IR request into the Responses request body.
pub(crate) fn build_body(
    req: &Request,
    model: Option<&str>,
    mode: CallMode,
    options: &ConvertOptions,
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
    if let Some(system) = &req.system
        && let Some(instructions) = build_instructions(system, &mut warnings)?
    {
        body.insert("instructions".to_owned(), Value::from(instructions));
    }

    let mut items: Vec<Value> = Vec::new();
    let mut pointers: Vec<(String, Role)> = Vec::new();
    for (mi, msg) in messages.iter().enumerate() {
        let before = items.len();
        build_message(
            msg,
            mi,
            options,
            &mut items,
            &mut pointers,
            &mut warnings,
            &mut log,
        )?;
        // A non-empty IR message whose blocks all dropped contributes no
        // wire item and is omitted (per-block warnings explain why); the
        // omission itself is disclosed here. Truly empty IR messages keep
        // their existing shape: user/system/developer replay as an empty
        // `content` item, assistant/tool emit nothing, all silently.
        if items.len() == before && !msg.content.is_empty() {
            warnings.push(warn(
                WarningCode::EmptyMessageDropped,
                "/input",
                format!(
                    "message {mi} serialized to zero wire content (every block was dropped) \
                     and was omitted from `input`"
                ),
            ));
        }
    }
    if !items.is_empty() {
        body.insert("input".to_owned(), Value::Array(items));
    }

    if let Some(v) = req.max_output_tokens {
        body.insert("max_output_tokens".to_owned(), Value::from(v));
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
            "`top_k` has no Responses equivalent and was dropped",
        ));
    }
    if req.stop_sequences.is_some() {
        warnings.push(warn(
            WarningCode::StopSequencesDropped,
            "/stop_sequences",
            "the Responses API has no stop-sequence parameter; `stop_sequences` was dropped",
        ));
    }
    if req.seed.is_some() {
        warnings.push(warn(
            WarningCode::SamplingParameterDropped,
            "/seed",
            "`seed` has no Responses equivalent and was dropped",
        ));
    }
    if req.frequency_penalty.is_some() {
        warnings.push(warn(
            WarningCode::SamplingParameterDropped,
            "/frequency_penalty",
            "`frequency_penalty` has no Responses equivalent and was dropped",
        ));
    }
    if req.presence_penalty.is_some() {
        warnings.push(warn(
            WarningCode::SamplingParameterDropped,
            "/presence_penalty",
            "`presence_penalty` has no Responses equivalent and was dropped",
        ));
    }
    if let Some(metadata) = &req.metadata {
        body.insert("metadata".to_owned(), Value::Object(metadata.clone()));
    }
    if let Some(key) = &req.cache_key {
        body.insert("prompt_cache_key".to_owned(), Value::from(key.clone()));
    }
    if let Some(reasoning) = &req.reasoning {
        build_reasoning(reasoning, &mut body, &mut warnings, &mut log);
    }
    if let Some(format) = &req.output_format {
        body.insert(
            "text".to_owned(),
            json!({ "format": build_text_format(format) }),
        );
    }
    // Whether any tool actually reached the wire — an explicitly empty IR
    // list replays as `"tools": []` (faithful), while a non-empty list
    // whose members were all dropped omits the key; `tool_choice` and
    // `parallel_tool_calls` follow the emitted tools, not the IR list.
    let mut has_tools = false;
    if let Some(tools) = &req.tools {
        let built = build_tools(tools, &mut warnings, &mut log);
        has_tools = !built.is_empty();
        if has_tools || tools.is_empty() {
            body.insert("tools".to_owned(), Value::Array(built));
        }
    }
    if let Some(choice) = &req.tool_choice {
        if has_tools {
            body.insert("tool_choice".to_owned(), build_tool_choice(choice));
        } else {
            warnings.push(warn(
                WarningCode::ToolChoiceIgnored,
                "/tool_choice",
                "`tool_choice` is meaningless without tools and was not emitted",
            ));
        }
    }
    if let Some(parallel) = req.parallel_tool_calls {
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
    }

    let generated_keys: BTreeSet<String> = body.keys().cloned().collect();
    let mut body = Value::Object(body);
    req.extra.merge_into(FORMAT, &mut body, "", &mut log);

    Ok(BuiltBody {
        body,
        warnings,
        merge_log: log,
        messages: pointers,
        generated_keys,
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
                "/input",
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
                        "/input",
                        "dropping the trailing tool calls left a thinking block without them; \
                         the upstream may reject the turn",
                    ));
                }
                warnings.push(warn(
                    WarningCode::OrphanToolCallsDropped,
                    "/input",
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
                    "/input",
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
                // Without `thinking_as_text` the plaintext placeholder is
                // foreign here and drops again at serialization — say so,
                // or the warning reads as a successful repair.
                let mut message =
                    format!("inserted a placeholder thinking block into assistant message {mi}");
                if !options.thinking_as_text {
                    message.push_str(
                        "; this format drops the placeholder during serialization (no unsigned \
                         thinking channel) — set `thinking_as_text` to carry it as text",
                    );
                }
                warnings.push(warn(WarningCode::MissingThinkingFilled, "/input", message));
            } else {
                warnings.push(warn(
                    WarningCode::MissingThinkingWithToolCalls,
                    "/input",
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

/// Joins `Request.system` text blocks into the top-level `instructions`
/// string (§ 7.1). Cache hints warn; only `Text` blocks are allowed.
fn build_instructions(
    system: &[ContentBlock],
    warnings: &mut Vec<ConversionWarning>,
) -> Result<Option<String>> {
    let mut texts: Vec<&str> = Vec::new();
    for (i, block) in system.iter().enumerate() {
        match block {
            ContentBlock::Text {
                text, cache, extra, ..
            } => {
                if cache.is_some() {
                    warnings.push(warn(
                        WarningCode::CacheHintDropped,
                        "/instructions",
                        "`instructions` is a plain string with no cache-breakpoint channel; \
                         put system content in the message array if breakpoints are needed",
                    ));
                }
                if extra.get(FORMAT).is_some_and(|ns| !ns.is_empty()) {
                    warnings.push(warn(
                        WarningCode::ExtraDropped,
                        "/instructions",
                        "block-level `extra` on `Request.system` has no landing spot: \
                         `instructions` is a plain string; use `Request.extra` to touch \
                         `instructions` itself, or put system content in the message array",
                    ));
                }
                texts.push(text);
            }
            other => {
                return Err(ConversionError::InvalidBlockForRole {
                    role: Role::System,
                    block: other.kind_name(),
                    location: format!("/system/{i}"),
                }
                .into());
            }
        }
    }
    if texts.is_empty() {
        Ok(None)
    } else {
        Ok(Some(texts.join("\n\n")))
    }
}

/// Serializes one IR message into `input` items.
fn build_message(
    msg: &Message,
    mi: usize,
    options: &ConvertOptions,
    items: &mut Vec<Value>,
    pointers: &mut Vec<(String, Role)>,
    warnings: &mut Vec<ConversionWarning>,
    log: &mut MergeLog,
) -> Result<()> {
    match msg.role {
        Role::System | Role::Developer | Role::User => {
            let (wire_role, ir_role) = match msg.role {
                Role::System => ("system", Role::System),
                Role::Developer if options.downgrade_developer => {
                    warnings.push(warn(
                        WarningCode::RoleDowngraded,
                        format!("/input/{}", items.len()),
                        "developer message downgraded to `user` (downgrade_developer)",
                    ));
                    ("user", Role::User)
                }
                Role::Developer => ("developer", Role::Developer),
                _ => ("user", Role::User),
            };
            let ptr = format!("/input/{}", items.len());
            if let Some(item) = build_input_message_item(msg, mi, wire_role, &ptr, warnings, log)? {
                items.push(item);
                pointers.push((ptr, ir_role));
            }
        }
        Role::Assistant => build_assistant_items(msg, mi, options, items, pointers, warnings, log)?,
        Role::Tool => build_tool_items(msg, mi, items, pointers, warnings, log)?,
    }
    Ok(())
}

/// Builds a `message` input item for a user/system/developer message.
/// Returns `None` when a non-empty message lost every block (the caller
/// omits the item and warns `EmptyMessageDropped`); a truly empty IR
/// message still replays as an item with an empty `content` array.
fn build_input_message_item(
    msg: &Message,
    mi: usize,
    wire_role: &str,
    item_ptr: &str,
    warnings: &mut Vec<ConversionWarning>,
    log: &mut MergeLog,
) -> Result<Option<Value>> {
    let mut parts: Vec<Value> = Vec::new();
    for (bi, block) in msg.content.iter().enumerate() {
        let part_ptr = format!("{item_ptr}/content/{}", parts.len());
        match block {
            ContentBlock::Text { text, cache, extra } => {
                let mut part = json!({"type": "input_text", "text": text});
                apply_breakpoint(&mut part, cache.as_ref(), &part_ptr, warnings);
                extra.merge_into(FORMAT, &mut part, &part_ptr, log);
                parts.push(part);
            }
            ContentBlock::Image {
                source,
                cache,
                extra,
            } if msg.role == Role::User => {
                let mut part = json!({"type": "input_image"});
                set_image_source(&mut part, source);
                apply_breakpoint(&mut part, cache.as_ref(), &part_ptr, warnings);
                extra.merge_into(FORMAT, &mut part, &part_ptr, log);
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
            other => {
                return Err(ConversionError::InvalidBlockForRole {
                    role: msg.role,
                    block: other.kind_name(),
                    location: format!("/messages/{mi}/content/{bi}"),
                }
                .into());
            }
        }
    }
    if parts.is_empty() && !msg.content.is_empty() {
        return Ok(None);
    }
    let mut item = json!({"type": "message", "role": wire_role, "content": parts});
    msg.extra.merge_into(FORMAT, &mut item, item_ptr, log);
    Ok(Some(item))
}

/// The item-identity grouping key of an assistant `Text` block.
///
/// A stored item `id` is authoritative on its own: parse distributes the
/// same item-level fields to every block of one item, so id-sharing
/// blocks always belong together (splitting hand-built blocks whose
/// other fields diverge would emit two wire items with the same id).
/// Without an id, blocks group only when the remaining item-level
/// reserved fields (`status` / `phase` / `item`) all agree — distinct
/// id-less items with differing metadata keep their boundaries, while
/// metadata-free ones still merge (the documented normalization).
fn item_group_key(extra: &Extra) -> [Option<Value>; 4] {
    match extra.get(FORMAT) {
        Some(ns) => identity_group_key(ns),
        None => Default::default(),
    }
}

/// The grouping key of a reserved-key identity map (a block's format
/// namespace, or a shell's recorded identity) — see [`item_group_key`]
/// for the id-vs-metadata rule.
fn identity_group_key(identity: &Map<String, Value>) -> [Option<Value>; 4] {
    let get = |key: &str| identity.get(key).cloned();
    let id = get(text_block_reserved_key::ID);
    if id.is_some() {
        [id, None, None, None]
    } else {
        [
            None,
            get(text_block_reserved_key::STATUS),
            get(text_block_reserved_key::PHASE),
            get(text_block_reserved_key::ITEM),
        ]
    }
}

/// One member of an assistant `message` item group.
enum GroupEntry<'a> {
    /// A `Text` block of the item.
    Text(&'a ContentBlock),
    /// An inlined unknown-part shell: its content parts re-serialize
    /// verbatim, in place, inside the item.
    Shell {
        /// The shell's content parts.
        parts: Vec<Value>,
        /// The shell's recorded item identity (reserved-key layout).
        identity: Map<String, Value>,
    },
}

/// Splits an own-format `Opaque` value carrying the internal
/// [`super::OPAQUE_SHELL_MARKER`] into its content parts and recorded
/// item identity. Returns `None` for marker-less values — user-built
/// opaques replay verbatim as top-level items. The marker is consumed
/// here; it never reaches the wire.
fn shell_parts_and_identity(value: &Value) -> Option<(Vec<Value>, Map<String, Value>)> {
    let marker = value.get(super::OPAQUE_SHELL_MARKER)?;
    let identity = marker.as_object().cloned().unwrap_or_default();
    let parts = value
        .get("content")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    Some((parts, identity))
}

/// Writes a recorded item identity onto a wire item: `id` / `status` /
/// `phase` directly, the `item` key expanding its nested unknown fields.
/// Plain inserts — item-level fields are parse-side restoration, not
/// user `extra`, so they must not enter the merge log (a conflict
/// warning located at such a field would otherwise mark itself
/// overridden).
fn apply_identity(item: &mut Value, identity: &Map<String, Value>) {
    let Some(obj) = item.as_object_mut() else {
        return;
    };
    for (key, value) in identity {
        if key == text_block_reserved_key::ITEM {
            if let Value::Object(fields) = value {
                for (field, fv) in fields {
                    obj.insert(field.clone(), fv.clone());
                }
            }
        } else {
            obj.insert(key.clone(), value.clone());
        }
    }
}

/// Explodes an assistant message into top-level items (reasoning /
/// message / `function_call` / opaque), preserving block order.
fn build_assistant_items(
    msg: &Message,
    mi: usize,
    options: &ConvertOptions,
    items: &mut Vec<Value>,
    pointers: &mut Vec<(String, Role)>,
    warnings: &mut Vec<ConversionWarning>,
    log: &mut MergeLog,
) -> Result<()> {
    let first_item = items.len();
    // Consecutive Text blocks (and inlined unknown-part shells) with the
    // same item-identity key (see `item_group_key`) group into one
    // assistant `message` item.
    let mut text_group: Vec<GroupEntry> = Vec::new();
    let mut group_key: [Option<Value>; 4] = Default::default();

    let flush = |group: &mut Vec<GroupEntry>,
                 items: &mut Vec<Value>,
                 pointers: &mut Vec<(String, Role)>,
                 warnings: &mut Vec<ConversionWarning>,
                 log: &mut MergeLog| {
        if group.is_empty() {
            return;
        }
        let ptr = format!("/input/{}", items.len());
        let item = build_assistant_message_item(group, &ptr, warnings, log);
        items.push(item);
        pointers.push((ptr, Role::Assistant));
        group.clear();
    };

    for (bi, block) in msg.content.iter().enumerate() {
        match block {
            ContentBlock::Text { extra, .. } => {
                let key = item_group_key(extra);
                if !text_group.is_empty() && key != group_key {
                    flush(&mut text_group, items, pointers, warnings, log);
                }
                group_key = key;
                text_group.push(GroupEntry::Text(block));
            }
            ContentBlock::Thinking {
                text,
                signature,
                extra,
            } => {
                flush(&mut text_group, items, pointers, warnings, log);
                let ptr = format!("/input/{}", items.len());
                if let Some(item) = build_reasoning_item(
                    text.as_deref(),
                    signature.as_deref(),
                    extra,
                    options,
                    &ptr,
                    warnings,
                    log,
                ) {
                    items.push(item);
                    pointers.push((ptr, Role::Assistant));
                }
            }
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
                cache,
                extra,
            } => {
                flush(&mut text_group, items, pointers, warnings, log);
                let ptr = format!("/input/{}", items.len());
                let call_id = id.clone().ok_or_else(|| {
                    ConversionError::missing(
                        format!(
                            "tool call `{name}` requires an id (`call_id`) on the Responses API"
                        ),
                        ptr.clone(),
                    )
                })?;
                if cache.is_some() {
                    warnings.push(warn(
                        WarningCode::CacheHintDropped,
                        ptr.clone(),
                        "cache breakpoints exist only on content parts; the hint on this tool call was dropped",
                    ));
                }
                let mut item = json!({
                    "type": "function_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                });
                extra.merge_into(FORMAT, &mut item, &ptr, log);
                items.push(item);
                pointers.push((ptr, Role::Assistant));
            }
            ContentBlock::Image { .. } => {
                flush(&mut text_group, items, pointers, warnings, log);
                warnings.push(warn(
                    WarningCode::ImageSourceUnsupported,
                    format!("/input/{}", items.len()),
                    "assistant images have no Responses input channel and were dropped",
                ));
            }
            ContentBlock::Opaque { format, value } => {
                if format != FORMAT {
                    flush(&mut text_group, items, pointers, warnings, log);
                    warnings.push(warn(
                        WarningCode::OpaqueDropped,
                        format!("/input/{}", items.len()),
                        format!("opaque block belongs to `{format}` and was dropped"),
                    ));
                } else if let Some((parts, identity)) = shell_parts_and_identity(value) {
                    let key = identity_group_key(&identity);
                    if text_group.is_empty() || key == group_key {
                        // The shell belongs to the current item (or
                        // starts one): its parts re-inline in place,
                        // restoring the original wire item.
                        group_key = key;
                        text_group.push(GroupEntry::Shell { parts, identity });
                    } else {
                        // Neighbour mismatch: the original boundary
                        // cannot be restored — the shell becomes its own
                        // item, marker stripped, identity written back.
                        flush(&mut text_group, items, pointers, warnings, log);
                        let ptr = format!("/input/{}", items.len());
                        warnings.push(warn(
                            WarningCode::ItemBoundaryLost,
                            ptr.clone(),
                            "the unknown-part shell's recorded item identity does not \
                             match its neighbouring blocks; it serializes as a separate \
                             item (sibling items may repeat its wire `id`)",
                        ));
                        let mut item =
                            json!({"type": "message", "role": "assistant", "content": parts});
                        apply_identity(&mut item, &identity);
                        items.push(item);
                        pointers.push((ptr, Role::Assistant));
                    }
                } else {
                    flush(&mut text_group, items, pointers, warnings, log);
                    let ptr = format!("/input/{}", items.len());
                    items.push(value.clone());
                    pointers.push((ptr, Role::Assistant));
                }
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
    flush(&mut text_group, items, pointers, warnings, log);

    // The message-level extra merges into the first produced item.
    if items.len() > first_item {
        let ptr = format!("/input/{first_item}");
        msg.extra
            .merge_into(FORMAT, &mut items[first_item], &ptr, log);
    }
    Ok(())
}

/// The per-field `ExtraDropped` warning for a conflicting item-level
/// value, located at the item-level field that kept the leading member's
/// value (so an `extra` override addressing that field marks it
/// overridden).
fn item_field_dropped(item_ptr: &str, field: &str) -> ConversionWarning {
    warn(
        WarningCode::ExtraDropped,
        format!("{item_ptr}/{}", escape_pointer_token(field)),
        format!(
            "item-level `{field}` on a non-leading member of this item conflicts \
             with the leading member's value and was dropped"
        ),
    )
}

/// Inserts a leading group member's item-level field into the item patch
/// (the `item` key expands its nested unknown fields).
fn fill_item_patch(item_patch: &mut Map<String, Value>, key: &str, value: &Value) {
    if key == text_block_reserved_key::ITEM {
        if let Value::Object(fields) = value {
            item_patch.extend(fields.clone());
        }
    } else {
        item_patch.insert(key.to_owned(), value.clone());
    }
}

/// Folds a non-leading group member's item-level field: silently equal
/// to the leading member's, or dropped with a per-field warning (`item`
/// compares its nested unknown fields one by one).
fn fold_item_conflict(
    key: &str,
    value: &Value,
    first_ns: &Map<String, Value>,
    item_ptr: &str,
    warnings: &mut Vec<ConversionWarning>,
) {
    if key == text_block_reserved_key::ITEM {
        let empty = Map::new();
        let first_item = first_ns
            .get(text_block_reserved_key::ITEM)
            .and_then(Value::as_object)
            .unwrap_or(&empty);
        if let Value::Object(fields) = value {
            for (field, fv) in fields {
                if first_item.get(field) != Some(fv) {
                    warnings.push(item_field_dropped(item_ptr, field));
                }
            }
        }
    } else if first_ns.get(key) != Some(value) {
        warnings.push(item_field_dropped(item_ptr, key));
    }
}

/// Builds an assistant `message` item from a group of `Text` blocks and
/// inlined unknown-part shells.
///
/// Reserved keys of each block's format namespace (see the module docs)
/// select the part shape (`refusal`) and restore item-level fields (`id`,
/// `status`, `phase`, `item`); the remaining keys merge into the part.
/// Shell members contribute their content parts verbatim, in place.
/// Item-level fields come from the leading member; a later member's
/// value that disagrees with it (possible only in id-keyed groups, i.e.
/// hand-built inconsistent metadata under one item id) is dropped with
/// an `ExtraDropped` warning located at the item-level field.
fn build_assistant_message_item(
    group: &[GroupEntry],
    item_ptr: &str,
    warnings: &mut Vec<ConversionWarning>,
    log: &mut MergeLog,
) -> Value {
    let mut content: Vec<Value> = Vec::new();
    let mut item_patch = Map::new();
    let first_ns: Map<String, Value> = match group.first() {
        Some(GroupEntry::Text(ContentBlock::Text { extra, .. })) => {
            extra.get(FORMAT).cloned().unwrap_or_default()
        }
        Some(GroupEntry::Shell { identity, .. }) => identity.clone(),
        _ => Map::new(),
    };
    for (gi, entry) in group.iter().enumerate() {
        let (text, cache, extra) = match entry {
            GroupEntry::Shell { parts, identity } => {
                for (key, value) in identity {
                    if gi == 0 {
                        fill_item_patch(&mut item_patch, key, value);
                    } else {
                        fold_item_conflict(key, value, &first_ns, item_ptr, warnings);
                    }
                }
                content.extend(parts.iter().cloned());
                continue;
            }
            GroupEntry::Text(ContentBlock::Text { text, cache, extra }) => (text, cache, extra),
            GroupEntry::Text(_) => continue,
        };
        let part_ptr = format!("{item_ptr}/content/{}", content.len());
        let ns = extra.get(FORMAT).cloned().unwrap_or_default();
        if cache.is_some() {
            warnings.push(warn(
                WarningCode::CacheHintDropped,
                part_ptr.clone(),
                "assistant output parts have no documented cache-breakpoint channel; the hint was dropped",
            ));
        }
        let refusal = ns
            .get(text_block_reserved_key::REFUSAL)
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let mut part = if refusal {
            json!({"type": "refusal", "refusal": text})
        } else {
            json!({"type": "output_text", "text": text, "annotations": []})
        };
        let mut part_patch = Map::new();
        for (key, value) in &ns {
            match key.as_str() {
                text_block_reserved_key::ID
                | text_block_reserved_key::STATUS
                | text_block_reserved_key::PHASE
                | text_block_reserved_key::ITEM => {
                    if gi == 0 {
                        fill_item_patch(&mut item_patch, key, value);
                    } else {
                        fold_item_conflict(key, value, &first_ns, item_ptr, warnings);
                    }
                }
                text_block_reserved_key::REFUSAL => {}
                _ => {
                    part_patch.insert(key.clone(), value.clone());
                }
            }
        }
        merge_patch(&mut part, &part_patch, &part_ptr, log);
        content.push(part);
    }
    let mut item = json!({"type": "message", "role": "assistant", "content": content});
    // Item-level fields are parse-side restoration, not user `extra`:
    // plain inserts, outside the merge log (see `apply_identity`).
    if let Some(obj) = item.as_object_mut() {
        for (key, value) in &item_patch {
            obj.insert(key.clone(), value.clone());
        }
    }
    item
}

/// Builds a `reasoning` item from a `Thinking` block, or drops the block
/// per the § 4.4 provenance rules. Returns `None` when nothing is emitted.
fn build_reasoning_item(
    text: Option<&str>,
    signature: Option<&str>,
    extra: &Extra,
    options: &ConvertOptions,
    ptr: &str,
    warnings: &mut Vec<ConversionWarning>,
    log: &mut MergeLog,
) -> Option<Value> {
    let native = is_native_thinking(signature, extra);
    if native {
        // A block parsed from this format carries the original `summary`
        // (and, when raw reasoning was exposed, `content`) arrays in its
        // namespace — the merge below restores them verbatim, so the base
        // must not synthesize either. A native block without those arrays
        // (hand-built, or an optimistically replayed foreign signature)
        // re-emits its text through `content` — the official raw-CoT
        // channel (`reasoning_text` parts); `summary` stays the required
        // empty array.
        let has_arrays = extra
            .get(FORMAT)
            .is_some_and(|ns| ns.contains_key("summary") || ns.contains_key("content"));
        let mut item = json!({"type": "reasoning", "summary": []});
        if !has_arrays && let Some(text) = text {
            item["content"] = json!([{"type": "reasoning_text", "text": text}]);
        }
        if let Some(sig) = signature {
            item["encrypted_content"] = Value::from(sig);
        }
        extra.merge_into(FORMAT, &mut item, ptr, log);
        return Some(item);
    }
    if options.thinking_as_text
        && let Some(text) = text
    {
        if signature.is_some() {
            warnings.push(warn(
                WarningCode::ThinkingSignatureDropped,
                ptr.to_owned(),
                "thinking_as_text re-emitted foreign thinking as raw reasoning text; its signature was dropped",
            ));
        }
        return Some(json!({
            "type": "reasoning",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": text}],
        }));
    }
    warnings.push(warn(
        WarningCode::ThinkingDropped,
        ptr.to_owned(),
        "thinking block is not native to `openai_responses` and was dropped \
         (set `thinking_as_text` to keep its text)",
    ));
    None
}

/// § 4.4 / contract provenance: a thinking block is native iff its `extra`
/// has this format's namespace, or it has a signature and no format
/// namespace at all (optimistic replay).
fn is_native_thinking(signature: Option<&str>, extra: &Extra) -> bool {
    let has_own = extra.get(FORMAT).is_some_and(|ns| !ns.is_empty());
    if has_own {
        return true;
    }
    let has_any = extra
        .formats()
        .any(|f| extra.get(f).is_some_and(|ns| !ns.is_empty()));
    signature.is_some() && !has_any
}

/// Serializes a `Tool` message: one `function_call_output` item per
/// `ToolResult` block (§ 7.2).
fn build_tool_items(
    msg: &Message,
    mi: usize,
    items: &mut Vec<Value>,
    pointers: &mut Vec<(String, Role)>,
    warnings: &mut Vec<ConversionWarning>,
    log: &mut MergeLog,
) -> Result<()> {
    let first_item = items.len();
    for (bi, block) in msg.content.iter().enumerate() {
        let ptr = format!("/input/{}", items.len());
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
                        "tool result requires `tool_call_id` (`call_id`) on the Responses API",
                        ptr.clone(),
                    )
                })?;
                if is_error == &Some(true) {
                    warnings.push(warn(
                        WarningCode::IsErrorDropped,
                        ptr.clone(),
                        "`is_error` is native to Anthropic and Google; the error marker was dropped",
                    ));
                }
                let output = build_tool_output(content, cache.as_ref(), &ptr, warnings, log);
                let mut item = json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                });
                if let Some(name) = name {
                    item["name"] = Value::from(name.clone());
                }
                extra.merge_into(FORMAT, &mut item, &ptr, log);
                items.push(item);
                pointers.push((ptr, Role::Tool));
            }
            ContentBlock::Opaque { format, value } => {
                if format == FORMAT {
                    items.push(value.clone());
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
    if items.len() > first_item {
        let ptr = format!("/input/{first_item}");
        msg.extra
            .merge_into(FORMAT, &mut items[first_item], &ptr, log);
    }
    Ok(())
}

/// Encodes `ToolResult.content` as `function_call_output.output`: the empty
/// list becomes `""`, a single plain text block uses the string shorthand,
/// anything else becomes a part array (§ 7.2). A block-level cache hint on
/// the `ToolResult` maps to a `prompt_cache_breakpoint` on the last output
/// part (output parts are input parts, which all accept breakpoints
/// upstream) — forcing the array form past the string shorthand when
/// needed; with no part to carry it (empty output) the hint drops with a
/// cosmetic warning. Nested per-block cache hints are dropped with a
/// cosmetic warning (§ 4.8, v1 rule).
fn build_tool_output(
    content: &[ToolOutputBlock],
    cache: Option<&CacheHint>,
    item_ptr: &str,
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
    let hint_without_part = |warnings: &mut Vec<ConversionWarning>| {
        warnings.push(warn(
            WarningCode::CacheHintDropped,
            item_ptr.to_owned(),
            "the tool result serialized with no output part to carry the cache breakpoint; \
             the hint was dropped",
        ));
    };
    if content.is_empty() {
        if cache.is_some() {
            hint_without_part(warnings);
        }
        return Value::from("");
    }
    if let [
        ToolOutputBlock::Text {
            text,
            cache: nested,
            extra,
        },
    ] = content
        && extra.get(FORMAT).is_none_or(Map::is_empty)
        && cache.is_none()
    {
        if nested.is_some() {
            nested_hint(warnings, format!("{item_ptr}/output"));
        }
        return Value::from(text.clone());
    }
    let mut parts: Vec<Value> = Vec::new();
    for block in content {
        let part_ptr = format!("{item_ptr}/output/{}", parts.len());
        match block {
            ToolOutputBlock::Text { text, cache, extra } => {
                if cache.is_some() {
                    nested_hint(warnings, part_ptr.clone());
                }
                let mut part = json!({"type": "input_text", "text": text});
                extra.merge_into(FORMAT, &mut part, &part_ptr, log);
                parts.push(part);
            }
            ToolOutputBlock::Image {
                source,
                cache,
                extra,
            } => {
                if cache.is_some() {
                    nested_hint(warnings, part_ptr.clone());
                }
                let mut part = json!({"type": "input_image"});
                set_image_source(&mut part, source);
                extra.merge_into(FORMAT, &mut part, &part_ptr, log);
                parts.push(part);
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
    if parts.is_empty() {
        // Every nested block was dropped: fall back to the canonical
        // empty encoding instead of a non-canonical `[]`.
        if cache.is_some() {
            hint_without_part(warnings);
        }
        return Value::from("");
    }
    if cache.is_some() {
        let last = parts.len() - 1;
        let part_ptr = format!("{item_ptr}/output/{last}");
        if parts[last].is_object() {
            apply_breakpoint(&mut parts[last], cache, &part_ptr, warnings);
        } else {
            hint_without_part(warnings);
        }
    }
    Value::Array(parts)
}

/// Writes an [`ImageSource`] into an `input_image` part (§ 4.3 table).
fn set_image_source(part: &mut Value, source: &ImageSource) {
    match source {
        ImageSource::Url(url) => part["image_url"] = Value::from(url.clone()),
        ImageSource::Base64 {
            media_type, data, ..
        } => {
            part["image_url"] = Value::from(to_data_url(media_type, data));
        }
        ImageSource::FileId(id) => part["file_id"] = Value::from(id.clone()),
    }
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

/// Builds the `reasoning` request object (§ 4.7 table) and merges the
/// `Reasoning.extra` namespace into it.
fn build_reasoning(
    reasoning: &Reasoning,
    body: &mut Map<String, Value>,
    warnings: &mut Vec<ConversionWarning>,
    log: &mut MergeLog,
) {
    let mut effort: Option<String> = reasoning.effort.as_ref().map(|e| e.as_str().to_owned());
    match (reasoning.enabled, &reasoning.effort) {
        (Some(true), Some(Effort::None)) | (Some(false), Some(_)) => {
            if !(reasoning.enabled == Some(false) && reasoning.effort == Some(Effort::None)) {
                warnings.push(warn(
                    WarningCode::ReasoningConflict,
                    "/reasoning",
                    "`reasoning.enabled` conflicts with `reasoning.effort`; `effort` wins",
                ));
            }
        }
        (Some(false), None) => effort = Some("none".to_owned()),
        _ => {}
    }
    let mut obj = Map::new();
    if let Some(effort) = effort {
        obj.insert("effort".to_owned(), Value::from(effort));
    }
    if reasoning.include_thoughts == Some(true) {
        obj.insert("summary".to_owned(), Value::from("auto"));
    }
    let has_extra = reasoning.extra.get(FORMAT).is_some_and(|ns| !ns.is_empty());
    if obj.is_empty() && !has_extra {
        return;
    }
    let mut value = Value::Object(obj);
    reasoning
        .extra
        .merge_into(FORMAT, &mut value, "/reasoning", log);
    body.insert("reasoning".to_owned(), value);
}

/// Builds `text.format` (§ 4.9 table). A missing schema name synthesizes
/// `"response"` — the field is required upstream.
fn build_text_format(format: &OutputFormat) -> Value {
    match format {
        OutputFormat::JsonSchema {
            name,
            description,
            schema,
            strict,
            ..
        } => {
            let mut obj = json!({
                "type": "json_schema",
                "name": name.clone().unwrap_or_else(|| "response".to_owned()),
                "schema": schema.clone(),
            });
            if let Some(description) = description {
                obj["description"] = Value::from(description.clone());
            }
            if let Some(strict) = strict {
                obj["strict"] = Value::from(*strict);
            }
            obj
        }
        OutputFormat::JsonObject => json!({"type": "json_object"}),
    }
}

/// Builds the flat Responses tool array (§ 4.5). `parameters` and `strict`
/// are required-but-nullable upstream: both always serialize, as `null`
/// when unset.
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
                let mut obj = Map::new();
                obj.insert("type".to_owned(), Value::from("function"));
                obj.insert("name".to_owned(), Value::from(f.name.clone()));
                if let Some(description) = &f.description {
                    obj.insert("description".to_owned(), Value::from(description.clone()));
                }
                obj.insert(
                    "parameters".to_owned(),
                    f.parameters.clone().unwrap_or(Value::Null),
                );
                obj.insert(
                    "strict".to_owned(),
                    f.strict.map(Value::from).unwrap_or(Value::Null),
                );
                if f.cache.is_some() {
                    warnings.push(warn(
                        WarningCode::CacheHintDropped,
                        ptr.clone(),
                        "tool-definition cache hints are Anthropic-only and were dropped",
                    ));
                }
                let mut value = Value::Object(obj);
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
        ToolChoice::Tool { name, .. } => json!({"type": "function", "name": name}),
    }
}
