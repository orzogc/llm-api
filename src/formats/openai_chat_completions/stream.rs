//! Streaming parser for `chat.completion.chunk` SSE streams (§ 9, § 11).
//!
//! Chat Completions streams carry per-choice `delta` fragments with no
//! explicit block boundaries; this parser infers them statefully:
//!
//! - `delta.content`, `delta.reasoning_content` and `delta.refusal` are
//!   three text channels; a fragment on a different channel than the last
//!   one closes the current block (`BlockStop { block: None }`) and opens
//!   a new one, so `reasoning_content` ↔ `content` transitions produce
//!   interleaved `Thinking` / `Text` blocks in arrival order.
//! - `delta.tool_calls` fragments group by their `index`: the first
//!   fragment of an index opens a `ToolCall` block (`arguments: ""`),
//!   later `function.arguments` (or dialect `custom.input`) pieces stream
//!   as [`BlockDelta::ToolArguments`], and the block finalizes at close
//!   with the accumulated id/name/arguments plus mirrored unknown
//!   fragment fields. A call that finishes without an `id` or with an
//!   empty `name` warns `MalformedToolCall`, matching the non-streaming
//!   entry parse.
//! - A choice `finish_reason` closes all open blocks and emits
//!   `MessageDelta { stop_reason }`; a non-null chunk `usage` (the
//!   `include_usage` final chunk, or dialects that attach usage to the
//!   finish chunk) emits a `MessageDelta` usage snapshot.
//! - The literal `data: [DONE]` terminator emits `MessageStop`; a stream
//!   that ends without it fails [`StreamParser::finish`] with a
//!   truncated-stream parse error.
//! - Chunks for choice indexes beyond the first surface as
//!   [`StreamEvent::Unknown`] with a `MultipleCandidates` warning once per
//!   stream (§ 8). Chunk envelope noise (`system_fingerprint`,
//!   `obfuscation`, `service_tier`, `logprobs`, null `usage`) is consumed
//!   silently; unknown `delta` fields surface as [`BlockDelta::Other`] on
//!   the current channel block when one is open.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value};

use crate::convert::{ConversionWarning, WarningCode};
use crate::error::{Error, Result};
use crate::format::StreamParser;
use crate::http::SseEvent;
use crate::ir::{BlockDelta, ContentBlock, Extra, StreamEvent};

use super::{FORMAT, to_ir, tool_call_reserved_key};

/// Parse-side warning shorthand.
fn warn(
    code: WarningCode,
    location: impl Into<String>,
    message: impl Into<String>,
) -> ConversionWarning {
    ConversionWarning::from_format(code, FORMAT, location, message)
}

/// Which text channel the current block belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Channel {
    /// `delta.content` → `Text` block.
    Content,
    /// `delta.reasoning_content` → `Thinking` block.
    Reasoning,
    /// `delta.refusal` → refusal-marked `Text` block (§ 9).
    Refusal,
}

/// Which payload object a streamed tool call carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PayloadKind {
    /// A `function` payload (`name` / `arguments`).
    Function,
    /// A `custom` payload (`name` / `input`, dialect: chunks document only
    /// `function`).
    Custom,
}

impl PayloadKind {
    /// The payload object key on the wire entry.
    fn key(self) -> &'static str {
        match self {
            Self::Function => "function",
            Self::Custom => "custom",
        }
    }

    /// The argument-channel member of the payload object.
    fn args_key(self) -> &'static str {
        match self {
            Self::Function => "arguments",
            Self::Custom => "input",
        }
    }
}

/// Accumulated state of one streamed tool call, keyed by its wire `index`.
#[derive(Debug, Default)]
struct ToolCallState {
    /// Unified block index.
    block_index: usize,
    /// Call id (first fragment).
    id: Option<String>,
    /// Accumulated name fragments.
    name: String,
    /// Accumulated argument fragments.
    arguments: String,
    /// The entry `type`, when not `function`.
    call_type: Option<String>,
    /// The payload kind locked by the declared `type` or by the first
    /// payload object; a conflicting later payload cannot be spliced and
    /// mirrors into `extra` instead.
    kind: Option<PayloadKind>,
    /// Unknown entry-level fragment fields, mirrored into the finalized
    /// block extra (`function` / `custom` unknowns nested under their
    /// payload key).
    extra: Map<String, Value>,
}

/// Stateful stream parser for `openai_chat_completions` (one instance per
/// stream). Implements [`StreamParser`].
#[derive(Debug, Default)]
pub struct ChatCompletionsStreamParser {
    started: bool,
    terminal: bool,
    next_index: usize,
    /// The open text channel and its block index.
    channel: Option<(Channel, usize)>,
    /// Open tool calls by wire `index`.
    tool_calls: BTreeMap<u64, ToolCallState>,
    warned_multiple: bool,
    /// Unknown delta field names already warned about.
    warned_unknown: BTreeSet<String>,
}

impl ChatCompletionsStreamParser {
    /// A fresh parser.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn alloc_index(&mut self) -> usize {
        let index = self.next_index;
        self.next_index += 1;
        index
    }

    /// Emits `MessageStart` on the first chunk (id/model from the chunk
    /// envelope), or defensively at the terminator.
    fn ensure_started(&mut self, chunk: Option<&Value>, events: &mut Vec<StreamEvent>) {
        if self.started {
            return;
        }
        self.started = true;
        events.push(StreamEvent::message_start(
            chunk
                .and_then(|c| c.get("id"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            chunk
                .and_then(|c| c.get("model"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            None,
        ));
    }

    /// Closes the open text channel and every open tool call, in block
    /// order. Tool calls finalize with their accumulated state, warning
    /// about unified fields the stream never delivered.
    fn close_open_blocks(
        &mut self,
        events: &mut Vec<StreamEvent>,
        warnings: &mut Vec<ConversionWarning>,
    ) {
        let mut stops: Vec<(usize, Option<ContentBlock>)> = Vec::new();
        if let Some((_, index)) = self.channel.take() {
            stops.push((index, None));
        }
        for (wire_index, state) in std::mem::take(&mut self.tool_calls) {
            let block_index = state.block_index;
            let block = finalize_tool_call(state, wire_index, warnings);
            stops.push((block_index, Some(block)));
        }
        stops.sort_by_key(|(index, _)| *index);
        for (index, block) in stops {
            events.push(StreamEvent::block_stop(index, block));
        }
    }

    /// Routes a fragment on a text channel: switching channels closes the
    /// current block and opens a new one; empty fragments open blocks but
    /// emit no delta.
    fn channel_delta(&mut self, kind: Channel, fragment: &str, events: &mut Vec<StreamEvent>) {
        let index = match self.channel {
            Some((current, index)) if current == kind => index,
            _ => {
                if let Some((_, old)) = self.channel.take() {
                    events.push(StreamEvent::block_stop(old, None));
                }
                let index = self.alloc_index();
                let block = match kind {
                    Channel::Content => ContentBlock::text(""),
                    Channel::Reasoning => ContentBlock::Thinking {
                        text: None,
                        signature: None,
                        extra: Extra::new(),
                    },
                    Channel::Refusal => {
                        let mut ns = Map::new();
                        ns.insert("refusal".to_owned(), Value::from(true));
                        ContentBlock::Text {
                            text: String::new(),
                            cache: None,
                            extra: Extra::from_unknown(FORMAT, ns),
                        }
                    }
                };
                self.channel = Some((kind, index));
                events.push(StreamEvent::block_start(index, block));
                index
            }
        };
        if !fragment.is_empty() {
            let delta = match kind {
                Channel::Content | Channel::Refusal => BlockDelta::Text(fragment.to_owned()),
                Channel::Reasoning => BlockDelta::Thinking(fragment.to_owned()),
            };
            events.push(StreamEvent::block_delta(index, delta));
        }
    }

    /// Handles one `delta.tool_calls[]` fragment.
    fn on_tool_call_fragment(
        &mut self,
        fragment: &Value,
        position: usize,
        events: &mut Vec<StreamEvent>,
        warnings: &mut Vec<ConversionWarning>,
    ) {
        let index = match fragment.get("index").and_then(Value::as_u64) {
            Some(i) => i,
            None => {
                warnings.push(warn(
                    WarningCode::MalformedField,
                    "/choices/0/delta/tool_calls",
                    "tool call fragment has no `index`; its array position was used instead",
                ));
                position as u64
            }
        };
        let is_new = !self.tool_calls.contains_key(&index);
        if is_new {
            let block_index = self.alloc_index();
            self.tool_calls.insert(
                index,
                ToolCallState {
                    block_index,
                    ..ToolCallState::default()
                },
            );
        }
        let state = self.tool_calls.get_mut(&index).expect("inserted above");
        // Id fragments concatenate (dialects may split them).
        if let Some(id) = fragment.get("id").and_then(Value::as_str) {
            match &mut state.id {
                Some(existing) => existing.push_str(id),
                none => *none = Some(id.to_owned()),
            }
        }
        // An explicit `type` locks the payload kind up front (it appears
        // only on a call's first fragment), so a lone conflicting payload
        // object is mirrored, not spliced, like on the non-streaming side.
        if state.kind.is_none() {
            state.kind = match fragment.get("type").and_then(Value::as_str) {
                Some("function") => Some(PayloadKind::Function),
                Some("custom") => Some(PayloadKind::Custom),
                _ => None,
            };
        }
        // Continuation fragments omit `type`; route by the payload object
        // the fragment actually carries.
        let mut deltas: Vec<String> = Vec::new();
        for kind in [PayloadKind::Function, PayloadKind::Custom] {
            match fragment.get(kind.key()) {
                None | Some(Value::Null) => {}
                Some(payload) => {
                    route_payload(kind, payload, index, state, &mut deltas, warnings);
                }
            }
        }
        collect_fragment_extras(fragment, state);
        if is_new {
            let block = ContentBlock::ToolCall {
                id: state.id.clone(),
                name: state.name.clone(),
                arguments: String::new(),
                cache: None,
                extra: Extra::new(),
            };
            events.push(StreamEvent::block_start(state.block_index, block));
        }
        for piece in deltas {
            state.arguments.push_str(&piece);
            events.push(StreamEvent::block_delta(
                state.block_index,
                BlockDelta::ToolArguments(piece),
            ));
        }
    }

    /// Handles the primary choice's `delta` object.
    fn on_delta(
        &mut self,
        delta: &Value,
        events: &mut Vec<StreamEvent>,
        warnings: &mut Vec<ConversionWarning>,
    ) {
        if let Some(s) = delta.get("reasoning_content").and_then(Value::as_str) {
            self.channel_delta(Channel::Reasoning, s, events);
        }
        if let Some(s) = delta.get("content").and_then(Value::as_str) {
            self.channel_delta(Channel::Content, s, events);
        }
        if let Some(s) = delta.get("refusal").and_then(Value::as_str) {
            self.channel_delta(Channel::Refusal, s, events);
        }
        if let Some(entries) = delta.get("tool_calls").and_then(Value::as_array) {
            for (position, fragment) in entries.iter().enumerate() {
                self.on_tool_call_fragment(fragment, position, events, warnings);
            }
        }
        // Unknown delta fields: surface on the current channel block when
        // one is open, otherwise as Unknown (warned once per field name).
        let mut leftover = Map::new();
        if let Some(obj) = delta.as_object() {
            for (key, value) in obj {
                match key.as_str() {
                    "role" | "content" | "reasoning_content" | "refusal" | "tool_calls" => {}
                    _ if value.is_null() => {}
                    _ => {
                        leftover.insert(key.clone(), value.clone());
                    }
                }
            }
        }
        if !leftover.is_empty() {
            match self.channel {
                Some((_, index)) => {
                    events.push(StreamEvent::block_delta(
                        index,
                        BlockDelta::Other(Value::Object(leftover)),
                    ));
                }
                None => {
                    events.push(StreamEvent::Unknown);
                    for key in leftover.keys() {
                        if self.warned_unknown.insert(key.clone()) {
                            warnings.push(warn(
                                WarningCode::UnknownStreamEvent,
                                "/choices/0/delta",
                                format!(
                                    "unrecognized delta field `{key}` could not be attributed \
                                     to a block"
                                ),
                            ));
                        }
                    }
                }
            }
        }
    }
}

/// Routes one payload object (`function` or `custom`) of a tool-call
/// fragment. The call's first payload locks its kind (unless the declared
/// `type` already did): `name` pieces concatenate into the state and
/// argument-channel pieces are collected for delta emission, mirroring the
/// non-streaming entry parse. A payload of the conflicting kind cannot be
/// spliced into the accumulated name/arguments and mirrors verbatim into
/// the state extra with a `MalformedToolCall` warning.
fn route_payload(
    kind: PayloadKind,
    payload: &Value,
    wire_index: u64,
    state: &mut ToolCallState,
    deltas: &mut Vec<String>,
    warnings: &mut Vec<ConversionWarning>,
) {
    let Some(members) = payload.as_object() else {
        warnings.push(warn(
            WarningCode::MalformedToolCall,
            "/choices/0/delta/tool_calls",
            format!(
                "tool call index {wire_index}: non-object `{}` payload fragment was ignored",
                kind.key()
            ),
        ));
        return;
    };
    match state.kind {
        None => state.kind = Some(kind),
        Some(locked) if locked != kind => {
            warnings.push(warn(
                WarningCode::MalformedToolCall,
                "/choices/0/delta/tool_calls",
                format!(
                    "tool call index {wire_index} mixes `{}` and `{}` payloads; the `{}` \
                     fragment was mirrored into the block extra instead of accumulated",
                    locked.key(),
                    kind.key(),
                    kind.key(),
                ),
            ));
            extend_nested_extra(state, kind.key(), members.clone());
            return;
        }
        Some(_) => {}
    }
    let mut unknowns = Map::new();
    for (key, value) in members {
        if key == "name" || key == kind.args_key() {
            match value.as_str() {
                Some(piece) if key == "name" => state.name.push_str(piece),
                Some(piece) => {
                    if !piece.is_empty() {
                        deltas.push(piece.to_owned());
                    }
                }
                None if value.is_null() => {}
                None => {
                    warnings.push(warn(
                        WarningCode::MalformedToolCall,
                        "/choices/0/delta/tool_calls",
                        format!(
                            "tool call index {wire_index}: non-string `{}.{key}` fragment \
                             was ignored",
                            kind.key()
                        ),
                    ));
                }
            }
        } else if !value.is_null() {
            unknowns.insert(key.clone(), value.clone());
        }
    }
    extend_nested_extra(state, kind.key(), unknowns);
}

/// Merges payload members into the object mirrored at `state.extra[key]`.
fn extend_nested_extra(state: &mut ToolCallState, key: &str, members: Map<String, Value>) {
    if members.is_empty() {
        return;
    }
    let nested = state
        .extra
        .entry(key.to_owned())
        .or_insert_with(|| Value::Object(Map::new()));
    if let Some(nested) = nested.as_object_mut() {
        nested.extend(members);
    }
}

/// Copies unmodeled entry-level fragment fields into the tool-call state;
/// payload objects are handled by [`route_payload`] and non-`function`
/// `type`s are recorded for the reserved key.
fn collect_fragment_extras(fragment: &Value, state: &mut ToolCallState) {
    let Some(obj) = fragment.as_object() else {
        return;
    };
    for (key, value) in obj {
        match key.as_str() {
            "index" | "id" | "function" | "custom" => {}
            "type" => {
                if let Some(t) = value.as_str()
                    && t != "function"
                {
                    state.call_type = Some(t.to_owned());
                }
            }
            _ if value.is_null() => {}
            _ => {
                state.extra.insert(key.clone(), value.clone());
            }
        }
    }
}

/// Builds the finalized `ToolCall` block from accumulated fragments. A
/// call that never delivered an `id` or a `name` is degraded the same way
/// the non-streaming entry parse degrades it (`id: None`, empty `name`)
/// and warns `MalformedToolCall` per field.
fn finalize_tool_call(
    state: ToolCallState,
    wire_index: u64,
    warnings: &mut Vec<ConversionWarning>,
) -> ContentBlock {
    if state.id.is_none() {
        warnings.push(warn(
            WarningCode::MalformedToolCall,
            "/choices/0/delta/tool_calls",
            format!(
                "tool call index {wire_index} has no `id`; the IR keeps `id: None`, and \
                 formats that require tool call ids will fail to rebuild the call"
            ),
        ));
    }
    if state.name.is_empty() {
        warnings.push(warn(
            WarningCode::MalformedToolCall,
            "/choices/0/delta/tool_calls",
            format!("tool call index {wire_index} finished with an empty `name`"),
        ));
    }
    let mut ns = state.extra;
    if let Some(t) = state.call_type {
        ns.insert(tool_call_reserved_key::TYPE.to_owned(), Value::from(t));
    }
    ContentBlock::ToolCall {
        id: state.id,
        name: state.name,
        arguments: state.arguments,
        cache: None,
        extra: Extra::from_unknown(FORMAT, ns),
    }
}

impl StreamParser for ChatCompletionsStreamParser {
    fn parse(&mut self, event: &SseEvent) -> Result<(Vec<StreamEvent>, Vec<ConversionWarning>)> {
        let mut events = Vec::new();
        let mut warnings = Vec::new();
        if event.data.trim() == "[DONE]" {
            self.ensure_started(None, &mut events);
            self.close_open_blocks(&mut events, &mut warnings);
            events.push(StreamEvent::MessageStop);
            self.terminal = true;
            return Ok((events, warnings));
        }
        let chunk: Value = match serde_json::from_str(&event.data) {
            Ok(v) => v,
            Err(e) => {
                warnings.push(warn(
                    WarningCode::MalformedField,
                    "",
                    format!("stream chunk is not JSON and was surfaced as unknown: {e}"),
                ));
                events.push(StreamEvent::Unknown);
                return Ok((events, warnings));
            }
        };
        self.ensure_started(Some(&chunk), &mut events);

        let mut finish_reason: Option<String> = None;
        if let Some(choices) = chunk.get("choices").and_then(Value::as_array) {
            for (position, choice) in choices.iter().enumerate() {
                let index = choice
                    .get("index")
                    .and_then(Value::as_u64)
                    .unwrap_or(position as u64);
                if index != 0 {
                    // Multi-candidate streams are unsupported (§ 8).
                    events.push(StreamEvent::Unknown);
                    if !self.warned_multiple {
                        self.warned_multiple = true;
                        warnings.push(warn(
                            WarningCode::MultipleCandidates,
                            "/choices",
                            "chunks for choice indexes beyond the first are surfaced as \
                             unknown events",
                        ));
                    }
                    continue;
                }
                if let Some(delta) = choice.get("delta") {
                    self.on_delta(delta, &mut events, &mut warnings);
                }
                if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                    finish_reason = Some(reason.to_owned());
                }
            }
        }

        let usage = chunk.get("usage").and_then(to_ir::usage_from_value);
        if let Some(reason) = finish_reason {
            self.close_open_blocks(&mut events, &mut warnings);
            events.push(StreamEvent::message_delta(
                Some(to_ir::map_finish_reason(&reason)),
                usage,
            ));
        } else if let Some(usage) = usage {
            // The `include_usage` final chunk (empty `choices`) or a
            // dialect attaching usage mid-stream: a cumulative snapshot.
            events.push(StreamEvent::message_delta(None, Some(usage)));
        }
        Ok((events, warnings))
    }

    fn finish(&mut self) -> Result<(Vec<StreamEvent>, Vec<ConversionWarning>)> {
        if self.terminal {
            Ok((Vec::new(), Vec::new()))
        } else {
            Err(Error::Parse {
                message: "truncated stream: the Chat Completions stream ended without `[DONE]`"
                    .to_owned(),
                raw: bytes::Bytes::new(),
            })
        }
    }
}
