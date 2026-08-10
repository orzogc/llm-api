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
//!   fragment fields. Finalization warns `MalformedToolCall` for a
//!   missing `id` and for payload members the stream never delivered,
//!   matching the non-streaming entry parse. The first explicitly
//!   declared `type` is authoritative: it overrides payload-shape
//!   inference, later conflicting declarations warn and are ignored,
//!   and an unknown or non-string declared `type` switches the call to
//!   mirror mode — its payloads accumulate silently and fold verbatim
//!   into the block extra, like the non-streaming verbatim entry mirror.
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
    /// Whether a payload object of the locked kind was ever accepted.
    saw_payload: bool,
    /// Whether a payload `name` member was ever delivered. An explicit
    /// empty string counts; `null` does not (the non-streaming parse
    /// treats `null` as missing too).
    saw_name: bool,
    /// Whether an argument-channel member (`arguments` / `input`) was
    /// ever delivered; same rules as [`Self::saw_name`].
    saw_arguments: bool,
    /// The first explicitly declared entry `type` (any non-`null` JSON
    /// value). It alone drives the kind/mirror routing and the reserved
    /// `type` key at finalization: later conflicting declarations warn
    /// and are ignored (first wins), and payload-shape inference applies
    /// only when no declaration ever arrives.
    declared_type: Option<Value>,
    /// The payload kind locked by the declared `type` or, absent one, by
    /// the first payload object (a late first declaration re-locks it);
    /// a conflicting payload cannot be spliced and mirrors into `extra`
    /// instead.
    kind: Option<PayloadKind>,
    /// Mirror mode (unknown or non-string declared `type`): payloads
    /// accumulate here for the finalize-time verbatim fold instead of
    /// the unified fields.
    mirror: Option<MirrorState>,
    /// Unknown entry-level fragment fields, mirrored into the finalized
    /// block extra (`function` / `custom` unknowns nested under their
    /// payload key).
    extra: Map<String, Value>,
}

/// Mirror-mode accumulation of a streamed tool call whose declared `type`
/// is unknown or non-string: the unified name/arguments channels stay
/// empty (the non-streaming parse mirrors such entries verbatim into the
/// format namespace), so each payload object accumulates here and folds
/// into the finalized block extra.
#[derive(Debug, Default)]
struct MirrorState {
    /// The `function` payload accumulation.
    function: MirrorPayload,
    /// The `custom` payload accumulation.
    custom: MirrorPayload,
}

impl MirrorState {
    /// The accumulator for one payload kind.
    fn payload_mut(&mut self, kind: PayloadKind) -> &mut MirrorPayload {
        match kind {
            PayloadKind::Function => &mut self.function,
            PayloadKind::Custom => &mut self.custom,
        }
    }
}

/// Accumulates one payload object of a call in mirror mode.
#[derive(Debug, Default)]
struct MirrorPayload {
    /// Whether this payload object ever appeared on the wire.
    seen: bool,
    /// Concatenated `name` pieces (`None`: never delivered).
    name: Option<String>,
    /// Concatenated argument-channel pieces (`None`: never delivered).
    args: Option<String>,
    /// Other members, verbatim (later fragments overwrite by key).
    extra: Map<String, Value>,
}

impl MirrorPayload {
    /// Merges one payload-object fragment: string `name` / argument
    /// pieces concatenate (splicing them into an extra map piecewise
    /// would overwrite earlier pieces — `Map::extend` replaces by key),
    /// non-string and unmodeled members stash verbatim, and `null`
    /// members drop like every fragment field.
    fn absorb(&mut self, kind: PayloadKind, members: Map<String, Value>) {
        self.seen = true;
        for (key, value) in members {
            if value.is_null() {
                continue;
            }
            if let Value::String(piece) = &value {
                let slot = if key == "name" {
                    Some(&mut self.name)
                } else if key == kind.args_key() {
                    Some(&mut self.args)
                } else {
                    None
                };
                if let Some(slot) = slot {
                    match slot {
                        Some(existing) => existing.push_str(piece),
                        none => *none = Some(piece.clone()),
                    }
                    continue;
                }
            }
            self.extra.insert(key, value);
        }
    }

    /// Folds the accumulation into the complete payload object; `None`
    /// when the stream never carried this payload.
    fn fold(self, kind: PayloadKind) -> Option<Value> {
        if !self.seen {
            return None;
        }
        let mut members = self.extra;
        if let Some(name) = self.name {
            members.insert("name".to_owned(), Value::from(name));
        }
        if let Some(args) = self.args {
            members.insert(kind.args_key().to_owned(), Value::from(args));
        }
        Some(Value::Object(members))
    }
}

/// Switches a tool call to mirror mode when a fragment declares an
/// unknown or non-string `type`. Payload state accumulated before the
/// switch (the pathological order: payloads first, the declaration late)
/// moves into the mirror accumulators so the finalize-time fold still
/// produces complete payload objects; argument deltas already emitted
/// are superseded by the authoritative `BlockStop` block (§ 9).
fn enter_mirror(state: &mut ToolCallState) {
    let mut mirror = MirrorState::default();
    for kind in [PayloadKind::Function, PayloadKind::Custom] {
        // Payload objects already mirrored under their key (the
        // mixed-kind conflict path) move over so later fragments of that
        // payload keep merging.
        match state.extra.remove(kind.key()) {
            Some(Value::Object(members)) => mirror.payload_mut(kind).absorb(kind, members),
            Some(other) => {
                // Defensive: a non-object mirror cannot merge; keep it.
                state.extra.insert(kind.key().to_owned(), other);
            }
            None => {}
        }
    }
    if let Some(kind) = state.kind {
        // The modeled accumulation of the locked kind; members the
        // stream never delivered stay absent from the fold.
        let payload = mirror.payload_mut(kind);
        payload.seen |= state.saw_payload;
        if state.saw_name {
            payload.name = Some(std::mem::take(&mut state.name));
        }
        if state.saw_arguments {
            payload.args = Some(std::mem::take(&mut state.arguments));
        }
    }
    state.mirror = Some(mirror);
}

/// Applies the first explicitly declared `type` of a streamed tool call:
/// `function` / `custom` lock (or re-lock) the payload kind, an unknown
/// string switches the call to mirror mode (warned at finalization, like
/// the non-streaming verbatim entry mirror), and a non-string value also
/// mirrors — verbatim and rebuildable, so it warns `MalformedField`
/// (cosmetic) here rather than at finalization.
fn apply_declared_type(
    declared: &Value,
    wire_index: u64,
    state: &mut ToolCallState,
    warnings: &mut Vec<ConversionWarning>,
) {
    match declared.as_str() {
        Some("function") => relock_kind(state, PayloadKind::Function),
        Some("custom") => relock_kind(state, PayloadKind::Custom),
        Some(_) => enter_mirror(state),
        None => {
            warnings.push(warn(
                WarningCode::MalformedField,
                "/choices/0/delta/tool_calls",
                format!(
                    "tool call index {wire_index} declares a non-string `type`; the entry \
                     was mirrored verbatim"
                ),
            ));
            enter_mirror(state);
        }
    }
}

/// Locks the payload kind from a declared known `type`. When payload
/// inference already locked the conflicting kind, the declaration wins
/// (matching the non-streaming parse of the assembled entry): the
/// accumulated payload moves verbatim into the state extra under its key
/// — like the mixed-payload conflict path — the unified channels reset,
/// and finalization then reports the declared kind's missing members
/// exactly as the non-streaming parse would. Argument deltas already
/// emitted are superseded by the authoritative `BlockStop` block (§ 9).
fn relock_kind(state: &mut ToolCallState, declared: PayloadKind) {
    let inferred = match state.kind {
        None => {
            state.kind = Some(declared);
            return;
        }
        Some(kind) if kind == declared => return,
        Some(kind) => kind,
    };
    let mut members = match state.extra.remove(inferred.key()) {
        Some(Value::Object(members)) => members,
        Some(other) => {
            // Defensive: a non-object mirror cannot merge; keep it.
            state.extra.insert(inferred.key().to_owned(), other);
            Map::new()
        }
        None => Map::new(),
    };
    if state.saw_name {
        members.insert(
            "name".to_owned(),
            Value::from(std::mem::take(&mut state.name)),
        );
    }
    if state.saw_arguments {
        members.insert(
            inferred.args_key().to_owned(),
            Value::from(std::mem::take(&mut state.arguments)),
        );
    }
    if state.saw_payload || !members.is_empty() {
        state
            .extra
            .insert(inferred.key().to_owned(), Value::Object(members));
    }
    state.name.clear();
    state.arguments.clear();
    state.saw_payload = false;
    state.saw_name = false;
    state.saw_arguments = false;
    state.kind = Some(declared);
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
        // Payload-kind authority order: the first explicitly declared
        // `type` > later declarations (first wins; a conflicting one
        // warns and is ignored) > payload-shape inference. `null` is the
        // absent canonical form, like every fragment field.
        match fragment.get("type") {
            None | Some(Value::Null) => {}
            Some(declared) => match &state.declared_type {
                None => {
                    state.declared_type = Some(declared.clone());
                    apply_declared_type(declared, index, state, warnings);
                }
                Some(first) if first == declared => {}
                Some(_) => {
                    warnings.push(warn(
                        WarningCode::MalformedToolCall,
                        "/choices/0/delta/tool_calls",
                        format!(
                            "tool call index {index} declares conflicting `type` values; \
                             the first declaration wins and this one was ignored"
                        ),
                    ));
                }
            },
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
/// the state extra with a `MalformedToolCall` warning. In mirror mode
/// (unknown or non-string declared `type`) every payload accumulates
/// silently for the finalize-time fold and emits no argument deltas.
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
    if let Some(mirror) = &mut state.mirror {
        mirror.payload_mut(kind).absorb(kind, members.clone());
        return;
    }
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
    state.saw_payload = true;
    let mut unknowns = Map::new();
    for (key, value) in members {
        if key == "name" || key == kind.args_key() {
            match value.as_str() {
                Some(piece) if key == "name" => {
                    state.saw_name = true;
                    state.name.push_str(piece);
                }
                Some(piece) => {
                    state.saw_arguments = true;
                    if !piece.is_empty() {
                        deltas.push(piece.to_owned());
                    }
                }
                // `null` means "absent on this fragment": the member still
                // counts as never delivered, like the non-streaming parse.
                None if value.is_null() => {}
                None => {
                    // Warned here, so the finalize missing-member check
                    // stays silent for this member.
                    if key == "name" {
                        state.saw_name = true;
                    } else {
                        state.saw_arguments = true;
                    }
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
/// `id`, `type` and the payload objects have dedicated handling
/// (`declared_type` and [`route_payload`]).
fn collect_fragment_extras(fragment: &Value, state: &mut ToolCallState) {
    let Some(obj) = fragment.as_object() else {
        return;
    };
    for (key, value) in obj {
        match key.as_str() {
            "index" | "id" | "type" | "function" | "custom" => {}
            _ if value.is_null() => {}
            _ => {
                state.extra.insert(key.clone(), value.clone());
            }
        }
    }
}

/// Builds the finalized `ToolCall` block from accumulated fragments,
/// warning `MalformedToolCall` with the same coverage as the non-streaming
/// entry parse: a missing `id`, a payload object that never arrived, and
/// payload members the stream never delivered (an explicit empty string is
/// a legitimate delivery and stays silent). A call in mirror mode folds
/// its accumulated payloads verbatim into the format namespace instead —
/// empty unified `name` / `arguments`, one cosmetic `MalformedField`
/// (emitted here for unknown strings, at the declaring fragment for
/// non-string values) — matching the non-streaming verbatim entry mirror.
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
    let mut ns = state.extra;
    if let Some(mirror) = state.mirror {
        // Unknown or non-string declared kind: fold each accumulated
        // payload into one complete object under its key and keep the
        // unified fields empty, exactly like the non-streaming verbatim
        // entry mirror. A non-string declaration already warned when its
        // fragment arrived.
        if let Some(Value::String(declared)) = &state.declared_type {
            warnings.push(warn(
                WarningCode::MalformedField,
                "/choices/0/delta/tool_calls",
                format!(
                    "tool call index {wire_index} has unknown tool call type `{declared}`; \
                     the entry was mirrored verbatim"
                ),
            ));
        }
        for (kind, payload) in [
            (PayloadKind::Function, mirror.function),
            (PayloadKind::Custom, mirror.custom),
        ] {
            if let Some(folded) = payload.fold(kind) {
                ns.insert(kind.key().to_owned(), folded);
            }
        }
        if let Some(declared) = state.declared_type {
            ns.insert(tool_call_reserved_key::TYPE.to_owned(), declared);
        }
        return ContentBlock::ToolCall {
            id: state.id,
            name: String::new(),
            arguments: String::new(),
            cache: None,
            extra: Extra::from_unknown(FORMAT, ns),
        };
    }
    // Parity with the non-streaming lenient parse: report the missing
    // payload and each never-delivered member of the effective kind.
    let kind = state.kind.unwrap_or(PayloadKind::Function);
    let (key, args_key) = (kind.key(), kind.args_key());
    if !state.saw_payload {
        warnings.push(warn(
            WarningCode::MalformedToolCall,
            "/choices/0/delta/tool_calls",
            format!(
                "tool call index {wire_index} carries no `{key}` payload; `name` and \
                 `{args_key}` parse as empty strings"
            ),
        ));
    }
    if !state.saw_name {
        warnings.push(warn(
            WarningCode::MalformedToolCall,
            "/choices/0/delta/tool_calls",
            format!(
                "tool call index {wire_index}: `{key}.name` is missing; it parses as an \
                 empty string and re-serializes as such"
            ),
        ));
    }
    if !state.saw_arguments {
        warnings.push(warn(
            WarningCode::MalformedToolCall,
            "/choices/0/delta/tool_calls",
            format!(
                "tool call index {wire_index}: `{key}.{args_key}` is missing; it parses \
                 as an empty string and re-serializes as such"
            ),
        ));
    }
    // The reserved `type` key is a pure serialization rule computed from
    // the declared type or, absent one, the inferred kind.
    if let Some(declared) = state.declared_type {
        // The canonical `function` declaration needs no reserved key;
        // in this non-mirror path the only other declaration is the
        // string `"custom"`.
        if declared.as_str() != Some("function") {
            ns.insert(tool_call_reserved_key::TYPE.to_owned(), declared);
        }
    } else if kind == PayloadKind::Custom {
        // A typeless call that streamed only a `custom` payload is
        // inferred as custom, like the non-streaming parse — silent
        // canonicalization (§ 1): the rebuilt entry gains
        // `type: "custom"`. (Extreme cross-fragment order — `custom`
        // payload first, `function` payload later — keeps the locked
        // custom kind, whereas the non-streaming parse of the assembled
        // entry would prefer `function`.)
        ns.insert(
            tool_call_reserved_key::TYPE.to_owned(),
            Value::from("custom"),
        );
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
