//! IR → Anthropic Messages request JSON (build side).

use serde_json::{Map, Value, json};

use crate::convert::{ConversionWarning, ConvertOptions, OrphanToolCalls, WarningCode};
use crate::error::{ConversionError, Error, Result};
use crate::format::BuildCtx;
use crate::ir::{
    CacheHint, ContentBlock, Effort, Extra, MergeLog, Message, Reasoning, Request, Role, Tool,
    ToolChoice, ToolOutputBlock, merge_patch,
};

use super::{FORMAT_ID as FMT, has_in_array_system_meta};

/// Output of the body builder, consumed by `AnthropicMessages::build_request`
/// and the count-tokens adapter.
pub(crate) struct ChatBody {
    /// The assembled request JSON (node-level and request-level `extra`
    /// already merged; hooks and the strict gate not yet run).
    pub body: Value,
    /// Build-side warnings collected so far.
    pub warnings: Vec<ConversionWarning>,
    /// Log of the `extra` merges, for `overridden` marking.
    pub merge_log: MergeLog,
    /// `(pointer, role)` per serialized wire message, for the `on_message`
    /// hook (wire roles mapped to IR roles).
    pub message_pointers: Vec<(String, Role)>,
    /// Top-level keys the converter itself generated (snapshot taken before
    /// the request-level `extra` merge), for the count-tokens adapter.
    pub generated_keys: Vec<String>,
}

/// Wire role of a serialized message.
#[derive(Clone, Copy, PartialEq, Eq)]
enum WireRole {
    User,
    Assistant,
    System,
}

impl WireRole {
    fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::System => "system",
        }
    }

    /// The IR role reported to the `on_message` hook for this wire role.
    fn ir_role(self) -> Role {
        match self {
            Self::User => Role::User,
            Self::Assistant => Role::Assistant,
            Self::System => Role::System,
        }
    }
}

/// One planned wire message: the IR messages (by index) whose blocks it
/// concatenates.
struct WirePlan {
    role: WireRole,
    parts: Vec<usize>,
    /// Emit `parts[0]`'s lone own-format `Opaque` value verbatim as the
    /// whole wire message (unknown-role passthrough); excluded from every
    /// merge.
    verbatim: bool,
}

/// A warning recorded before final wire positions are known; `msg` is an
/// index into the post-transform IR message list.
struct Pending {
    code: WarningCode,
    msg: Option<usize>,
    /// Append `/content/0` to the resolved message pointer.
    at_first_block: bool,
    text: String,
}

fn warn(
    code: WarningCode,
    location: impl Into<String>,
    text: impl Into<String>,
) -> ConversionWarning {
    ConversionWarning::to_format(code, FMT, location, text)
}

/// Builds the Anthropic Messages request body from the IR (§ 6 pipeline
/// steps 1–3; `finalize_request` and URL building happen in the caller).
pub(crate) fn build_chat_body(req: &Request, ctx: &BuildCtx, streaming: bool) -> Result<ChatBody> {
    crate::convert::check_finite_sampling(&[
        (req.temperature, "/temperature"),
        (req.top_p, "/top_p"),
        (req.frequency_penalty, "/frequency_penalty"),
        (req.presence_penalty, "/presence_penalty"),
    ])?;
    let opts = &ctx.format_options.anthropic;
    let convert = &ctx.convert;
    let mut warnings: Vec<ConversionWarning> = Vec::new();
    let mut pending: Vec<Pending> = Vec::new();
    let mut log = MergeLog::new();

    // IR-level transforms (owned copy; the input request is never mutated).
    let mut messages: Vec<Message> = req.messages.clone();
    apply_orphan_policy(&mut messages, convert, &mut pending);
    apply_missing_thinking(&mut messages, req.reasoning.as_ref(), convert, &mut pending);

    // Layout: hoist the leading run of unmarked System messages, plan wire
    // messages, re-merge turn groups, then apply `merge_consecutive_roles`.
    let mut hoisted: Vec<usize> = Vec::new();
    let mut plans: Vec<WirePlan> = Vec::new();
    let mut leading = true;
    for (i, m) in messages.iter().enumerate() {
        // Verbatim wire messages neither hoist nor merge: they become
        // standalone wire messages re-emitting the opaque value untouched.
        if verbatim_wire_message(m).is_some() {
            leading = false;
            plans.push(WirePlan {
                // Placeholder; verbatim plans render from the opaque value
                // and never take part in role-based merging.
                role: WireRole::User,
                parts: vec![i],
                verbatim: true,
            });
            continue;
        }
        if leading && m.role == Role::System && !has_in_array_system_meta(m.round_trip.as_ref()) {
            hoisted.push(i);
            continue;
        }
        leading = false;
        let wire_role = match m.role {
            Role::System => {
                if opts.downgrade_mid_system {
                    pending.push(Pending {
                        code: WarningCode::RoleDowngraded,
                        msg: Some(i),
                        at_first_block: false,
                        text: "system message downgraded to user (downgrade_mid_system)".into(),
                    });
                    WireRole::User
                } else {
                    WireRole::System
                }
            }
            Role::Developer => {
                pending.push(Pending {
                    code: WarningCode::RoleDowngraded,
                    msg: Some(i),
                    at_first_block: false,
                    text: "developer role has no Anthropic equivalent; downgraded to user".into(),
                });
                WireRole::User
            }
            Role::User | Role::Tool => WireRole::User,
            Role::Assistant => WireRole::Assistant,
        };
        // Turn-group re-merge: adjacent Tool/User messages sharing a valid
        // group id rejoin the wire user turn they were split from.
        if wire_role == WireRole::User
            && matches!(m.role, Role::User | Role::Tool)
            && let Some(group) = m.turn_group_id()
            && let Some(last) = plans.last_mut()
            && last.role == WireRole::User
            && !last.verbatim
            && last.parts.last().is_some_and(|&p| {
                matches!(messages[p].role, Role::User | Role::Tool)
                    && messages[p].turn_group_id() == Some(group)
            })
        {
            last.parts.push(i);
            continue;
        }
        plans.push(WirePlan {
            role: wire_role,
            parts: vec![i],
            verbatim: false,
        });
    }

    if opts.merge_consecutive_roles {
        let mut merged: Vec<WirePlan> = Vec::new();
        for plan in plans {
            // Verbatim plans are merge barriers on both sides, silently —
            // they are not same-role neighbours but complete wire messages.
            if let Some(prev) = merged.last_mut()
                && prev.role == plan.role
                && !prev.verbatim
                && !plan.verbatim
            {
                let barrier = prev
                    .parts
                    .iter()
                    .chain(plan.parts.iter())
                    .any(|&p| has_signed_block(&messages[p]));
                if barrier {
                    pending.push(Pending {
                        code: WarningCode::MergeBlockedBySignature,
                        msg: Some(plan.parts[0]),
                        at_first_block: false,
                        text: "adjacent same-role messages not merged: a signed \
                               thinking/tool-call block is a merge barrier"
                            .into(),
                    });
                } else {
                    prev.parts.extend(plan.parts);
                    continue;
                }
            }
            merged.push(plan);
        }
        plans = merged;
    }

    // Required max_tokens.
    let Some(max_tokens) = req.max_output_tokens.or(opts.default_max_tokens) else {
        return Err(ConversionError::missing(
            "max_tokens (set Request.max_output_tokens or AnthropicOptions.default_max_tokens)",
            "/max_tokens",
        )
        .into());
    };

    let mut body = Map::new();
    body.insert("model".to_owned(), Value::from(ctx.model.clone()));
    body.insert("max_tokens".to_owned(), Value::from(max_tokens));

    // Top-level system channel.
    if let Some(system) = build_system(req, &messages, &hoisted, &mut warnings, &mut log)? {
        body.insert("system".to_owned(), system);
    }

    // Messages. Wire indices are assigned at push time: a wire message may
    // be omitted below, shifting every later index.
    let mut ir_to_wire: Vec<Option<usize>> = vec![None; messages.len()];
    let mut message_pointers: Vec<(String, Role)> = Vec::new();
    let mut wire_messages: Vec<Value> = Vec::new();
    for plan in &plans {
        let w = wire_messages.len();
        let msg_loc = format!("/messages/{w}");
        if plan.verbatim {
            let m = &messages[plan.parts[0]];
            let value =
                verbatim_wire_message(m).expect("verbatim plan holds a lone opaque wire message");
            let mut wire_msg = value.clone();
            m.extra.merge_into(FMT, &mut wire_msg, &msg_loc, &mut log);
            ir_to_wire[plan.parts[0]] = Some(w);
            wire_messages.push(wire_msg);
            message_pointers.push((msg_loc, m.role));
            continue;
        }
        let mut content: Vec<Value> = Vec::new();
        // Block warnings are buffered per plan: when the whole wire message
        // is omitted below, their positional locations would point at the
        // next message, so they are re-addressed to the container.
        let mut plan_warnings: Vec<ConversionWarning> = Vec::new();
        for &p in &plan.parts {
            let m = &messages[p];
            for (bi, block) in m.content.iter().enumerate() {
                let block_loc = format!("{msg_loc}/content/{}", content.len());
                let ir_loc = format!("/messages/{p}/content/{bi}");
                if let Some(v) = convert_block(
                    m.role,
                    block,
                    &block_loc,
                    &ir_loc,
                    convert,
                    &mut plan_warnings,
                    &mut log,
                )? {
                    content.push(v);
                }
            }
        }
        // Non-empty IR content that serialized to zero wire content is
        // omitted (disclosed) rather than sent as an empty `content: []`
        // message; truly empty IR messages still serialize as `content: []`.
        if content.is_empty() && plan.parts.iter().any(|&p| !messages[p].content.is_empty()) {
            for pw in &mut plan_warnings {
                pw.location = "/messages".to_owned();
            }
            warnings.append(&mut plan_warnings);
            let dropped: Vec<String> = plan
                .parts
                .iter()
                .filter(|&&p| !messages[p].content.is_empty())
                .map(usize::to_string)
                .collect();
            warnings.push(warn(
                WarningCode::EmptyMessageDropped,
                "/messages",
                format!(
                    "IR message(s) {} serialized to zero wire content (every block \
                     was dropped); the empty wire message was omitted",
                    dropped.join(", ")
                ),
            ));
            continue;
        }
        warnings.append(&mut plan_warnings);
        let mut wire_msg = json!({
            "role": plan.role.as_str(),
            "content": Value::Array(content),
        });
        for &p in &plan.parts {
            messages[p]
                .extra
                .merge_into(FMT, &mut wire_msg, &msg_loc, &mut log);
            ir_to_wire[p] = Some(w);
        }
        wire_messages.push(wire_msg);
        message_pointers.push((msg_loc, plan.role.ir_role()));
    }
    body.insert("messages".to_owned(), Value::Array(wire_messages));

    // Deferred warnings resolve against final wire positions; a message
    // omitted above (or removed by a transform) falls back to the
    // `/messages` container.
    for p in pending {
        let location = match p.msg.and_then(|i| ir_to_wire.get(i).copied().flatten()) {
            Some(w) if p.at_first_block => format!("/messages/{w}/content/0"),
            Some(w) => format!("/messages/{w}"),
            None => "/messages".to_owned(),
        };
        warnings.push(warn(p.code, location, p.text));
    }

    // Metadata: only `user_id` maps; other keys are dropped with a warning.
    if let Some(metadata) = &req.metadata
        && !metadata.is_empty()
    {
        let mut dropped: Vec<&str> = Vec::new();
        let mut out = Map::new();
        for (k, v) in metadata {
            if k == "user_id" {
                out.insert(k.clone(), v.clone());
            } else {
                dropped.push(k);
            }
        }
        if !out.is_empty() {
            body.insert("metadata".to_owned(), Value::Object(out));
        }
        if !dropped.is_empty() {
            warnings.push(warn(
                WarningCode::MetadataDropped,
                "/metadata",
                format!(
                    "Anthropic metadata only accepts user_id; dropped keys: {}",
                    dropped.join(", ")
                ),
            ));
        }
    }

    // Sampling parameters.
    if let Some(stop) = &req.stop_sequences {
        body.insert("stop_sequences".to_owned(), json!(stop));
    }
    if let Some(t) = req.temperature {
        body.insert("temperature".to_owned(), json!(t));
    }
    if let Some(p) = req.top_p {
        body.insert("top_p".to_owned(), json!(p));
    }
    if let Some(k) = req.top_k {
        body.insert("top_k".to_owned(), json!(k));
    }
    if req.seed.is_some() {
        warnings.push(warn(
            WarningCode::SamplingParameterDropped,
            "/seed",
            "seed is not supported by Anthropic Messages; dropped",
        ));
    }
    if req.frequency_penalty.is_some() {
        warnings.push(warn(
            WarningCode::SamplingParameterDropped,
            "/frequency_penalty",
            "frequency_penalty is not supported by Anthropic Messages; dropped",
        ));
    }
    if req.presence_penalty.is_some() {
        warnings.push(warn(
            WarningCode::SamplingParameterDropped,
            "/presence_penalty",
            "presence_penalty is not supported by Anthropic Messages; dropped",
        ));
    }
    if req.cache_key.is_some() {
        warnings.push(warn(
            WarningCode::CacheKeyDropped,
            "/prompt_cache_key",
            "cache_key has no Anthropic equivalent; dropped",
        ));
    }

    if streaming {
        body.insert("stream".to_owned(), Value::Bool(true));
    }

    // Tools. Whether any tool actually reached the wire — an explicitly
    // empty IR list replays as `"tools": []` (faithful), while a non-empty
    // list whose entries were all dropped (foreign `Tool::Opaque`, already
    // disclosed by `OpaqueDropped`) omits the key; the `tool_choice` keys
    // follow the emitted tools, not the IR list.
    let mut has_tools = false;
    if let Some(tools) = &req.tools {
        let mut out: Vec<Value> = Vec::new();
        for tool in tools {
            let loc = format!("/tools/{}", out.len());
            match tool {
                Tool::Function(ft) => {
                    let mut obj = Map::new();
                    obj.insert("name".to_owned(), Value::from(ft.name.clone()));
                    if let Some(d) = &ft.description {
                        obj.insert("description".to_owned(), Value::from(d.clone()));
                    }
                    let schema = match &ft.parameters {
                        Some(s) => s.clone(),
                        None => {
                            warnings.push(warn(
                                WarningCode::EmptyParametersSynthesized,
                                format!("{loc}/input_schema"),
                                "input_schema is required; synthesized an explicit \
                                 empty-parameter schema",
                            ));
                            json!({"type": "object", "properties": {}, "additionalProperties": false})
                        }
                    };
                    obj.insert("input_schema".to_owned(), schema);
                    if let Some(s) = ft.strict {
                        obj.insert("strict".to_owned(), Value::Bool(s));
                    }
                    if let Some(hint) = &ft.cache {
                        obj.insert("cache_control".to_owned(), cache_control_value(hint));
                    }
                    let mut v = Value::Object(obj);
                    ft.extra.merge_into(FMT, &mut v, &loc, &mut log);
                    out.push(v);
                }
                Tool::Opaque { format, value, .. } => {
                    if format == FMT {
                        out.push(value.clone());
                    } else {
                        warnings.push(warn(
                            WarningCode::OpaqueDropped,
                            loc,
                            format!("opaque tool belongs to format `{format}`; dropped"),
                        ));
                    }
                }
            }
        }
        has_tools = !out.is_empty();
        if has_tools || tools.is_empty() {
            body.insert("tools".to_owned(), Value::Array(out));
        }
    }

    // Tool choice and parallel_tool_calls (§ 4.5 combination table).
    build_tool_choice(req, has_tools, &mut body, &mut warnings);

    // Reasoning → thinking / output_config.effort (§ 4.7).
    let mut output_config = Map::new();
    if let Some(reasoning) = &req.reasoning {
        build_reasoning(reasoning, &mut body, &mut output_config, &mut warnings);
    }

    // Structured output (§ 4.9).
    if let Some(format) = &req.output_format {
        match format {
            crate::ir::OutputFormat::JsonSchema {
                name,
                description,
                schema,
                strict,
                ..
            } => {
                let mut detail: Vec<&str> = Vec::new();
                if name.is_some() {
                    detail.push("name");
                }
                if description.is_some() {
                    detail.push("description");
                }
                if strict.is_some() {
                    detail.push("strict");
                }
                if !detail.is_empty() {
                    warnings.push(warn(
                        WarningCode::OutputFormatDetailDropped,
                        "/output_config/format",
                        format!(
                            "Anthropic json_schema output has no channel for: {}",
                            detail.join(", ")
                        ),
                    ));
                }
                output_config.insert(
                    "format".to_owned(),
                    json!({"type": "json_schema", "schema": schema}),
                );
            }
            crate::ir::OutputFormat::JsonObject => {
                warnings.push(warn(
                    WarningCode::JsonModeUnsupported,
                    "/output_config/format",
                    "Anthropic has no schema-less JSON mode; output_format dropped",
                ));
            }
        }
    }
    if !output_config.is_empty() {
        body.insert("output_config".to_owned(), Value::Object(output_config));
    }

    // Node-level extra of the reasoning config merges into the `thinking`
    // object (this format's home for reasoning configuration) — this is how
    // unmodeled fields such as `budget_tokens` are set (§ 4.7).
    let mut body = Value::Object(body);
    if let Some(reasoning) = &req.reasoning
        && reasoning.extra.get(FMT).is_some_and(|ns| !ns.is_empty())
    {
        if body.get("thinking").is_none() {
            body["thinking"] = json!({});
        }
        let target = body.get_mut("thinking").expect("just ensured");
        reasoning
            .extra
            .merge_into(FMT, target, "/thinking", &mut log);
    }

    // Snapshot of converter-generated top-level keys (before the
    // request-level extra merge and hooks), for the count-tokens adapter.
    let generated_keys: Vec<String> = body
        .as_object()
        .expect("body is an object")
        .keys()
        .cloned()
        .collect();

    // Request-level extra merges into the whole body, last.
    req.extra.merge_into(FMT, &mut body, "", &mut log);

    Ok(ChatBody {
        body,
        warnings,
        merge_log: log,
        message_pointers,
        generated_keys,
    })
}

/// Returns the wire-message JSON when the message re-emits verbatim:
/// exactly one own-format `Opaque` block whose value is itself a wire
/// message (it has a string `role` field) — the parse-side home for
/// messages with unknown wire roles. `System` messages are excluded: their
/// canonical home is the top-level `system` channel (§ 7.1), where a
/// role-bearing opaque entry stays a system entry.
fn verbatim_wire_message(message: &Message) -> Option<&Value> {
    if message.role == Role::System {
        return None;
    }
    match message.content.as_slice() {
        [ContentBlock::Opaque { format, value, .. }]
            if format == FMT && value.get("role").is_some_and(Value::is_string) =>
        {
            Some(value)
        }
        _ => None,
    }
}

/// `true` when the message contains a signed block (§ 7.5): a `Thinking`
/// block with a signature, or a `ToolCall` carrying a Google
/// `thoughtSignature` in its `extra`.
fn has_signed_block(message: &Message) -> bool {
    message.content.iter().any(|b| match b {
        ContentBlock::Thinking { signature, .. } => signature.is_some(),
        ContentBlock::ToolCall { extra, .. } => extra
            .get(crate::format::ids::GOOGLE_GENERATE_CONTENT)
            .is_some_and(|ns| ns.contains_key("thoughtSignature")),
        _ => false,
    })
}

/// Applies the § 7.3 orphan-tool-call policy in place.
fn apply_orphan_policy(
    messages: &mut Vec<Message>,
    convert: &ConvertOptions,
    pending: &mut Vec<Pending>,
) {
    if messages.is_empty() {
        return;
    }
    // A ToolCall in message `i` is matched iff a later message contains a
    // ToolResult with the same id (or, when the call has no id, same name).
    let matched = |messages: &[Message], i: usize, id: Option<&str>, name: &str| -> bool {
        messages.iter().skip(i + 1).any(|m| {
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
        })
    };
    let last = messages.len() - 1;
    let mut mid_array: Vec<usize> = Vec::new();
    let mut trailing: Vec<(Option<String>, String)> = Vec::new();
    for (i, m) in messages.iter().enumerate() {
        if m.role != Role::Assistant {
            continue;
        }
        let mut has_orphan = false;
        for b in &m.content {
            if let ContentBlock::ToolCall { id, name, .. } = b
                && !matched(messages, i, id.as_deref(), name)
            {
                has_orphan = true;
                if i == last {
                    trailing.push((id.clone(), name.clone()));
                }
            }
        }
        if has_orphan && i != last {
            mid_array.push(i);
        }
    }
    for i in mid_array {
        pending.push(Pending {
            code: WarningCode::OrphanToolCalls,
            msg: Some(i),
            at_first_block: false,
            text: "assistant message contains tool calls without matching results \
                   mid-conversation; never repaired"
                .into(),
        });
    }
    if trailing.is_empty() {
        return;
    }
    match convert.orphan_tool_calls {
        OrphanToolCalls::Passthrough => {}
        OrphanToolCalls::DropTrailing => {
            let m = &mut messages[last];
            let is_orphan = |b: &ContentBlock| {
                matches!(b, ContentBlock::ToolCall { id, name, .. }
                    if trailing.iter().any(|(tid, tname)| tid == id && tname == name))
            };
            m.content.retain(|b| !is_orphan(b));
            let kept_thinking = m
                .content
                .iter()
                .any(|b| matches!(b, ContentBlock::Thinking { .. }));
            let no_calls_left = !m
                .content
                .iter()
                .any(|b| matches!(b, ContentBlock::ToolCall { .. }));
            let removed = m.content.is_empty();
            if removed {
                messages.pop();
            }
            pending.push(Pending {
                code: WarningCode::OrphanToolCallsDropped,
                msg: if removed { None } else { Some(last) },
                at_first_block: false,
                text: format!("dropped {} unmatched trailing tool call(s)", trailing.len()),
            });
            if !removed && kept_thinking && no_calls_left {
                pending.push(Pending {
                    code: WarningCode::ThinkingOrphaned,
                    msg: Some(last),
                    at_first_block: false,
                    text: "dropping trailing tool calls left a thinking block orphaned".into(),
                });
            }
        }
        OrphanToolCalls::SynthesizeError => {
            let results: Vec<ContentBlock> = trailing
                .iter()
                .map(|(id, name)| {
                    let mut r = ContentBlock::tool_result(
                        id.clone(),
                        vec![ToolOutputBlock::text("cancelled")],
                    )
                    .with_is_error(true);
                    r = r.with_tool_name(name.clone());
                    r
                })
                .collect();
            messages.push(Message::tool(results));
            pending.push(Pending {
                code: WarningCode::OrphanToolCallsSynthesized,
                msg: Some(messages.len() - 1),
                at_first_block: false,
                text: format!(
                    "appended {} synthetic error tool result(s) for unmatched trailing tool calls",
                    trailing.len()
                ),
            });
        }
    }
}

/// § 7.3: warn about (or fill) assistant messages that carry tool calls but
/// no thinking while the request enables thinking.
fn apply_missing_thinking(
    messages: &mut [Message],
    reasoning: Option<&Reasoning>,
    convert: &ConvertOptions,
    pending: &mut Vec<Pending>,
) {
    let enables = reasoning.is_some_and(|r| {
        r.enabled == Some(true)
            || (r.enabled.is_none() && r.effort.as_ref().is_some_and(|e| *e != Effort::None))
    });
    if !enables {
        return;
    }
    for (i, m) in messages.iter_mut().enumerate() {
        if m.role != Role::Assistant {
            continue;
        }
        let has_call = m
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolCall { .. }));
        let has_thinking = m
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::Thinking { .. }));
        if !has_call || has_thinking {
            continue;
        }
        if let Some(text) = &convert.fill_missing_thinking {
            m.content.insert(0, ContentBlock::thinking(text.clone()));
            pending.push(Pending {
                code: WarningCode::MissingThinkingFilled,
                msg: Some(i),
                at_first_block: true,
                text: "inserted a placeholder thinking block before tool calls".into(),
            });
        } else {
            pending.push(Pending {
                code: WarningCode::MissingThinkingWithToolCalls,
                msg: Some(i),
                at_first_block: false,
                text: "assistant message carries tool calls but no thinking block while \
                       the request enables thinking"
                    .into(),
            });
        }
    }
}

/// Builds the top-level `system` value from `Request.system` plus the
/// hoisted leading System messages (§ 7.1). Returns `None` when there is no
/// system content. Non-empty system input whose every block dropped
/// (foreign opaques in hoisted messages) omits the channel with an
/// `EmptyMessageDropped` warning (§ 7.6); block warnings are re-addressed
/// to the `/system` container, whose entries never reached the wire.
fn build_system(
    req: &Request,
    messages: &[Message],
    hoisted: &[usize],
    warnings: &mut Vec<ConversionWarning>,
    log: &mut MergeLog,
) -> Result<Option<Value>> {
    /// One prospective `system` array entry.
    struct Entry<'a> {
        value: Value,
        block_extra: Option<&'a Extra>,
        /// Eligible for the string form: a plain text block with no cache
        /// hint and no extra for this format.
        text_form_ok: bool,
    }
    fn text_entry<'a>(text: &str, cache: Option<&CacheHint>, extra: &'a Extra) -> Entry<'a> {
        let mut value = json!({"type": "text", "text": text});
        if let Some(hint) = cache {
            value["cache_control"] = cache_control_value(hint);
        }
        let text_form_ok = cache.is_none() && extra.get(FMT).is_none_or(Map::is_empty);
        Entry {
            value,
            block_extra: Some(extra),
            text_form_ok,
        }
    }

    let mut entries: Vec<Entry<'_>> = Vec::new();
    let mut msg_extras: Vec<(usize, &Extra)> = Vec::new(); // (first entry index, extra)
    // Block warnings are buffered: when the whole channel is omitted below,
    // their positional locations would point into a `system` array that
    // does not exist, so they are re-addressed to the container.
    let mut sys_warnings: Vec<ConversionWarning> = Vec::new();

    if let Some(system) = &req.system {
        for (bi, block) in system.iter().enumerate() {
            match block {
                ContentBlock::Text {
                    text, cache, extra, ..
                } => {
                    entries.push(text_entry(text, cache.as_ref(), extra));
                }
                other => {
                    // `Request.system` allows only Text blocks (contract
                    // pin); the location is the block's IR coordinate.
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
    for &i in hoisted {
        let m = &messages[i];
        let first_entry = entries.len();
        for (bi, block) in m.content.iter().enumerate() {
            let loc = format!("/system/{}", entries.len());
            match block {
                ContentBlock::Text {
                    text, cache, extra, ..
                } => {
                    entries.push(text_entry(text, cache.as_ref(), extra));
                }
                ContentBlock::Opaque { format, value, .. } => {
                    if format == FMT {
                        entries.push(Entry {
                            value: value.clone(),
                            block_extra: None,
                            text_form_ok: false,
                        });
                    } else {
                        sys_warnings.push(warn(
                            WarningCode::OpaqueDropped,
                            loc,
                            format!("opaque block belongs to format `{format}`; dropped"),
                        ));
                    }
                }
                other => {
                    // A hoisted message's block reports its IR coordinate.
                    return Err(ConversionError::InvalidBlockForRole {
                        role: Role::System,
                        block: other.kind_name(),
                        location: format!("/messages/{i}/content/{bi}"),
                    }
                    .into());
                }
            }
        }
        if entries.len() > first_entry && m.extra.get(FMT).is_some_and(|ns| !ns.is_empty()) {
            msg_extras.push((first_entry, &m.extra));
        }
    }

    if entries.is_empty() {
        // Non-empty input that serialized to nothing omits the channel with
        // a disclosure (§ 7.6); genuinely empty input (`Some(vec![])`, or a
        // hoisted message with no blocks) stays silent — the caller's own
        // data. `Request.system` is Text-only (never drops), so only
        // hoisted messages can lose every block here.
        let input_non_empty = req.system.as_ref().is_some_and(|s| !s.is_empty())
            || hoisted.iter().any(|&i| !messages[i].content.is_empty());
        if input_non_empty {
            for w in &mut sys_warnings {
                w.location = "/system".to_owned();
            }
            sys_warnings.push(warn(
                WarningCode::EmptyMessageDropped,
                "/system",
                "the system channel serialized to zero wire content (every block was \
                 dropped); the `system` key was omitted",
            ));
            // A hoisted message's extra would have merged into its first
            // produced entry; with none, the user-addressed data is lost.
            for &i in hoisted {
                if !messages[i].content.is_empty()
                    && messages[i].extra.get(FMT).is_some_and(|ns| !ns.is_empty())
                {
                    sys_warnings.push(warn(
                        WarningCode::ExtraDropped,
                        "/system",
                        format!(
                            "message-level `extra` of hoisted system message {i} has no wire \
                             location (the `system` channel was omitted) and was dropped"
                        ),
                    ));
                }
            }
        }
        warnings.append(&mut sys_warnings);
        return Ok(None);
    }
    warnings.append(&mut sys_warnings);
    // Exactly one plain text segment (no cache hint, no extra for this
    // format) uses the string form; anything else uses the block array.
    if entries.len() == 1 && entries[0].text_form_ok && msg_extras.is_empty() {
        let text = entries[0].value["text"].clone();
        return Ok(Some(text));
    }
    let mut array: Vec<Value> = Vec::new();
    for (j, entry) in entries.into_iter().enumerate() {
        let mut v = entry.value;
        let loc = format!("/system/{j}");
        if let Some(extra) = entry.block_extra {
            extra.merge_into(FMT, &mut v, &loc, log);
        }
        if let Some((_, extra)) = msg_extras.iter().find(|(first, _)| *first == j) {
            // A hoisted message has no wire node of its own; its
            // message-level extra merges into its first produced block.
            extra.merge_into(FMT, &mut v, &loc, log);
        }
        array.push(v);
    }
    Ok(Some(Value::Array(array)))
}

/// Converts one IR content block for its role (§ 7.4 validity plus the
/// per-block mapping). Returns `None` when the block is dropped with a
/// warning. `loc` points into the would-be wire output (warnings, merges,
/// `MissingRequired`); `ir_loc` is the block's IR coordinate, which is what
/// `InvalidBlockForRole` reports.
fn convert_block(
    role: Role,
    block: &ContentBlock,
    loc: &str,
    ir_loc: &str,
    convert: &ConvertOptions,
    warnings: &mut Vec<ConversionWarning>,
    log: &mut MergeLog,
) -> Result<Option<Value>> {
    let invalid = |block: &ContentBlock| -> Error {
        ConversionError::InvalidBlockForRole {
            role,
            block: block.kind_name(),
            location: ir_loc.to_owned(),
        }
        .into()
    };
    match block {
        ContentBlock::Text {
            text, cache, extra, ..
        } => {
            if role == Role::Tool {
                return Err(invalid(block));
            }
            let mut v = json!({"type": "text", "text": text});
            if let Some(hint) = cache {
                v["cache_control"] = cache_control_value(hint);
            }
            extra.merge_into(FMT, &mut v, loc, log);
            Ok(Some(v))
        }
        ContentBlock::Image {
            source,
            cache,
            extra,
            ..
        } => match role {
            Role::User => {
                let mut v = json!({"type": "image", "source": image_source_value(source)});
                if let Some(hint) = cache {
                    v["cache_control"] = cache_control_value(hint);
                }
                extra.merge_into(FMT, &mut v, loc, log);
                Ok(Some(v))
            }
            Role::Assistant => {
                warnings.push(warn(
                    WarningCode::ImageSourceUnsupported,
                    loc,
                    "assistant images have no channel on Anthropic; dropped",
                ));
                Ok(None)
            }
            _ => Err(invalid(block)),
        },
        ContentBlock::ToolCall {
            id,
            name,
            arguments,
            cache,
            extra,
            ..
        } => {
            if role != Role::Assistant {
                return Err(invalid(block));
            }
            let Some(id) = id else {
                return Err(ConversionError::missing("tool call id (tool_use.id)", loc).into());
            };
            let input = parse_tool_arguments(name, arguments, loc)?;
            let mut v = json!({"type": "tool_use", "id": id, "name": name, "input": input});
            if let Some(hint) = cache {
                v["cache_control"] = cache_control_value(hint);
            }
            extra.merge_into(FMT, &mut v, loc, log);
            Ok(Some(v))
        }
        ContentBlock::ToolResult {
            tool_call_id,
            content,
            is_error,
            cache,
            extra,
            ..
        } => {
            if role != Role::Tool {
                return Err(invalid(block));
            }
            let Some(tool_use_id) = tool_call_id else {
                return Err(ConversionError::missing(
                    "tool result id (tool_result.tool_use_id)",
                    loc,
                )
                .into());
            };
            let mut v = json!({"type": "tool_result", "tool_use_id": tool_use_id});
            build_tool_result_content(content, loc, warnings, log, &mut v);
            if let Some(err) = is_error {
                v["is_error"] = Value::Bool(*err);
            }
            if let Some(hint) = cache {
                v["cache_control"] = cache_control_value(hint);
            }
            extra.merge_into(FMT, &mut v, loc, log);
            Ok(Some(v))
        }
        ContentBlock::Thinking {
            text,
            signature,
            extra,
            ..
        } => {
            if role != Role::Assistant {
                return Err(invalid(block));
            }
            convert_thinking(
                text.as_deref(),
                signature.as_deref(),
                extra,
                loc,
                convert,
                warnings,
                log,
            )
        }
        ContentBlock::Opaque { format, value, .. } => {
            if format == FMT {
                Ok(Some(value.clone()))
            } else {
                warnings.push(warn(
                    WarningCode::OpaqueDropped,
                    loc,
                    format!("opaque block belongs to format `{format}`; dropped"),
                ));
                Ok(None)
            }
        }
    }
}

/// Parses `ToolCall.arguments` into the JSON object Anthropic `input`
/// requires; invalid JSON or a non-object is a `ConversionError` in both
/// modes (§ 4.5).
fn parse_tool_arguments(name: &str, arguments: &str, loc: &str) -> Result<Value> {
    let parsed: Value =
        serde_json::from_str(arguments).map_err(|e| ConversionError::InvalidToolArguments {
            name: name.to_owned(),
            arguments: arguments.to_owned(),
            reason: format!("invalid JSON: {e}"),
            location: loc.to_owned(),
        })?;
    if !parsed.is_object() {
        return Err(ConversionError::InvalidToolArguments {
            name: name.to_owned(),
            arguments: arguments.to_owned(),
            reason: "arguments must be a JSON object".to_owned(),
            location: loc.to_owned(),
        }
        .into());
    }
    Ok(parsed)
}

/// Encodes `ToolResult.content` per § 7.2: empty → omitted, a single plain
/// text block → string shorthand, anything else → block array. Nested cache
/// hints are dropped with a cosmetic warning (v1 pin); foreign opaque blocks
/// are dropped with a semantic warning.
fn build_tool_result_content(
    content: &[ToolOutputBlock],
    loc: &str,
    warnings: &mut Vec<ConversionWarning>,
    log: &mut MergeLog,
    out: &mut Value,
) {
    let nested_cache_warning = |warnings: &mut Vec<ConversionWarning>, k: usize| {
        warnings.push(warn(
            WarningCode::CacheHintDropped,
            format!("{loc}/content/{k}"),
            "cache hints on nested tool-result blocks are dropped in v1",
        ));
    };
    if content.is_empty() {
        return; // Empty result: `content` omitted (§ 7.2 pin).
    }
    if let [
        ToolOutputBlock::Text {
            text, cache, extra, ..
        },
    ] = content
        && extra.get(FMT).is_none_or(Map::is_empty)
    {
        if cache.is_some() {
            nested_cache_warning(warnings, 0);
        }
        out["content"] = Value::from(text.clone());
        return;
    }
    let mut array: Vec<Value> = Vec::new();
    for block in content {
        let k = array.len();
        let block_loc = format!("{loc}/content/{k}");
        match block {
            ToolOutputBlock::Text {
                text, cache, extra, ..
            } => {
                if cache.is_some() {
                    nested_cache_warning(warnings, k);
                }
                let mut v = json!({"type": "text", "text": text});
                extra.merge_into(FMT, &mut v, &block_loc, log);
                array.push(v);
            }
            ToolOutputBlock::Image {
                source,
                cache,
                extra,
                ..
            } => {
                if cache.is_some() {
                    nested_cache_warning(warnings, k);
                }
                let mut v = json!({"type": "image", "source": image_source_value(source)});
                extra.merge_into(FMT, &mut v, &block_loc, log);
                array.push(v);
            }
            ToolOutputBlock::Opaque { format, value, .. } => {
                if format == FMT {
                    array.push(value.clone());
                } else {
                    warnings.push(warn(
                        WarningCode::OpaqueDropped,
                        block_loc,
                        format!("opaque block belongs to format `{format}`; dropped"),
                    ));
                }
            }
        }
    }
    out["content"] = Value::Array(array);
}

/// Converts a `Thinking` block per the § 4.4 provenance rules.
fn convert_thinking(
    text: Option<&str>,
    signature: Option<&str>,
    extra: &Extra,
    loc: &str,
    convert: &ConvertOptions,
    warnings: &mut Vec<ConversionWarning>,
    log: &mut MergeLog,
) -> Result<Option<Value>> {
    // Native iff the block has a non-empty namespace of this format — an
    // empty namespace carries no provenance — or it has a signature and no
    // non-empty format namespace at all (optimistic replay).
    let has_own_ns = extra.get(FMT).is_some_and(|ns| !ns.is_empty());
    let has_any_ns = extra
        .formats()
        .any(|f| extra.get(f).is_some_and(|ns| !ns.is_empty()));
    let native = has_own_ns || (signature.is_some() && !has_any_ns);
    if native {
        let redacted = extra
            .get(FMT)
            .and_then(|ns| ns.get("redacted"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let mut v = if redacted {
            let Some(data) = signature else {
                return Err(ConversionError::missing(
                    "redacted thinking data (Thinking.signature)",
                    loc,
                )
                .into());
            };
            json!({"type": "redacted_thinking", "data": data})
        } else {
            let mut obj = json!({"type": "thinking", "thinking": text.unwrap_or_default()});
            if let Some(sig) = signature {
                obj["signature"] = Value::from(sig);
            }
            obj
        };
        // The `redacted` marker is consumed by reconstruction; merge the
        // rest of the namespace.
        if let Some(ns) = extra.get(FMT) {
            let filtered: Map<String, Value> = ns
                .iter()
                .filter(|(key, _)| key.as_str() != "redacted")
                .map(|(key, val)| (key.clone(), val.clone()))
                .collect();
            if !filtered.is_empty() {
                merge_patch(&mut v, &filtered, loc, log);
            }
        }
        return Ok(Some(v));
    }
    // Foreign: drop, or re-emit as signature-less thinking text when
    // `thinking_as_text` is on.
    if convert.thinking_as_text
        && let Some(text) = text
    {
        if signature.is_some() {
            warnings.push(warn(
                WarningCode::ThinkingSignatureDropped,
                loc,
                "thinking_as_text cannot carry the foreign signature; dropped",
            ));
        }
        return Ok(Some(json!({"type": "thinking", "thinking": text})));
    }
    warnings.push(warn(
        WarningCode::ThinkingDropped,
        loc,
        "foreign thinking block dropped (enable thinking_as_text to keep its text)",
    ));
    Ok(None)
}

/// Builds `tool_choice` (and folds `parallel_tool_calls` into it) per the
/// § 4.5 combination table. `has_tools` is whether any tool reached the
/// wire: without wire tools no `tool_choice` key is emitted at all —
/// including the object synthesized from `parallel_tool_calls: false` —
/// and the ignored settings warn instead.
fn build_tool_choice(
    req: &Request,
    has_tools: bool,
    body: &mut Map<String, Value>,
    warnings: &mut Vec<ConversionWarning>,
) {
    let parallel = req.parallel_tool_calls;
    let parallel_ignored = |warnings: &mut Vec<ConversionWarning>, why: &str| {
        warnings.push(warn(
            WarningCode::ParallelToolCallsIgnored,
            "/tool_choice",
            why,
        ));
    };
    if !has_tools {
        if req.tool_choice.is_some() {
            warnings.push(warn(
                WarningCode::ToolChoiceIgnored,
                "/tool_choice",
                "tool_choice is meaningless without tools; not emitted",
            ));
        }
        if parallel.is_some() {
            parallel_ignored(
                warnings,
                "parallel_tool_calls is meaningless without tools; not emitted",
            );
        }
        return;
    }
    match &req.tool_choice {
        Some(choice) => {
            let mut obj = match choice {
                ToolChoice::Auto => json!({"type": "auto"}),
                ToolChoice::None => json!({"type": "none"}),
                ToolChoice::Required => json!({"type": "any"}),
                ToolChoice::Tool { name, .. } => json!({"type": "tool", "name": name}),
            };
            if let Some(p) = parallel {
                if matches!(choice, ToolChoice::None) {
                    parallel_ignored(
                        warnings,
                        "parallel_tool_calls is meaningless on tool_choice none; not emitted",
                    );
                } else {
                    obj["disable_parallel_tool_use"] = Value::Bool(!p);
                }
            }
            body.insert("tool_choice".to_owned(), obj);
        }
        None => {
            // Some(true) canonicalizes to nothing, no warning: parallel is
            // Anthropic's default (§ 4.5).
            if parallel == Some(false) {
                body.insert(
                    "tool_choice".to_owned(),
                    json!({"type": "auto", "disable_parallel_tool_use": true}),
                );
            }
        }
    }
}

/// Maps the IR reasoning configuration per the § 4.7 table.
///
/// `include_thoughts` maps to `thinking.display`, which only exists on the
/// `adaptive`/`enabled` thinking variants. It is attached to the generated
/// `{type: "adaptive"}` object when `enabled: true` produced one, or when an
/// effort tier that maps to Anthropic (`low`…`max` or a verbatim custom
/// tier) enables thinking; with no thinking basis (or `{type: "disabled"}`)
/// there is no carrier and the flag is dropped with a cosmetic warning —
/// synthesizing an `adaptive` object would turn thinking on as a side
/// effect.
fn build_reasoning(
    reasoning: &Reasoning,
    body: &mut Map<String, Value>,
    output_config: &mut Map<String, Value>,
    warnings: &mut Vec<ConversionWarning>,
) {
    let mut enabled = reasoning.enabled;
    let effort = reasoning.effort.as_ref();
    // Conflict rule: effort wins, cosmetic warning.
    let conflict = match (enabled, effort) {
        (Some(true), Some(Effort::None)) => true,
        (Some(false), Some(e)) => *e != Effort::None,
        _ => false,
    };
    if conflict {
        warnings.push(warn(
            WarningCode::ReasoningConflict,
            "/thinking",
            "Reasoning.enabled conflicts with effort; effort wins",
        ));
        enabled = None;
    }
    let mut effort_out: Option<String> = None;
    match effort {
        Some(e @ (Effort::None | Effort::Minimal)) => {
            warnings.push(warn(
                WarningCode::EffortUnsupported,
                "/output_config/effort",
                format!(
                    "effort `{}` is outside Anthropic's set (low/medium/high/xhigh/max); dropped",
                    e.as_str()
                ),
            ));
        }
        Some(e @ (Effort::Low | Effort::Medium | Effort::High | Effort::XHigh | Effort::Max)) => {
            effort_out = Some(e.as_str().to_owned());
        }
        Some(Effort::Other(s)) => effort_out = Some(s.clone()),
        None => {}
    }
    if let Some(e) = effort_out.clone() {
        output_config.insert("effort".to_owned(), Value::from(e));
    }
    let mut thinking = match enabled {
        Some(true) => Some(json!({"type": "adaptive"})),
        Some(false) => Some(json!({"type": "disabled"})),
        None => None,
    };
    if let Some(include) = reasoning.include_thoughts {
        let display = if include { "summarized" } else { "omitted" };
        match &mut thinking {
            Some(t) if t["type"] == "adaptive" => {
                t["display"] = Value::from(display);
            }
            Some(_) => {
                warnings.push(warn(
                    WarningCode::IncludeThoughtsUnsupported,
                    "/thinking",
                    "include_thoughts has no carrier on thinking {type: disabled}; dropped",
                ));
            }
            None if effort_out.is_some() => {
                // Effort enables thinking; adaptive is the faithful carrier.
                thinking = Some(json!({"type": "adaptive", "display": display}));
            }
            None => {
                warnings.push(warn(
                    WarningCode::IncludeThoughtsUnsupported,
                    "/thinking",
                    "include_thoughts without an enabling thinking configuration has no \
                     carrier; dropped",
                ));
            }
        }
    }
    if let Some(t) = thinking {
        body.insert("thinking".to_owned(), t);
    }
}

/// Serializes a cache hint as `cache_control`.
pub(crate) fn cache_control_value(hint: &CacheHint) -> Value {
    match &hint.ttl {
        Some(ttl) => json!({"type": "ephemeral", "ttl": ttl}),
        None => json!({"type": "ephemeral"}),
    }
}

/// Serializes an IR image source as the Anthropic `source` union.
fn image_source_value(source: &crate::ir::ImageSource) -> Value {
    match source {
        crate::ir::ImageSource::Url(url) => json!({"type": "url", "url": url}),
        crate::ir::ImageSource::Base64 {
            media_type, data, ..
        } => {
            json!({"type": "base64", "media_type": media_type, "data": data})
        }
        crate::ir::ImageSource::FileId(id) => json!({"type": "file", "file_id": id}),
    }
}
