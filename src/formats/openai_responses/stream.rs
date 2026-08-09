//! Streaming parser for the Responses semantic SSE events (§ 9, § 11).
//!
//! The protocol streams whole output items (`response.output_item.*`) with
//! nested content parts (`response.content_part.*`) and per-kind deltas.
//! This parser flattens the `(output_index, content_index)` /
//! `summary_index` coordinates into the unified, strictly increasing block
//! index space: each message content part becomes one `Text` block, a
//! `reasoning` item becomes one `Thinking` block (summary parts joined
//! with `"\n\n"`), a `function_call` item one `ToolCall` block, and any
//! unmodeled item one `Opaque` block whose sub-events surface as
//! [`BlockDelta::Other`].

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value};

use crate::convert::{ConversionWarning, WarningCode};
use crate::error::{Error, Result};
use crate::format::StreamParser;
use crate::http::SseEvent;
use crate::ir::{BlockDelta, ContentBlock, Extra, StreamEvent};

use super::{FORMAT, to_ir};

/// Parse-side warning shorthand.
fn warn(code: WarningCode, location: impl Into<String>, message: impl Into<String>) -> ConversionWarning {
    ConversionWarning::from_format(code, FORMAT, location, message)
}

/// Per-content-part streaming state of a `message` item.
#[derive(Debug)]
struct PartState {
    /// Unified block index.
    index: usize,
    /// Annotations collected from `output_text.annotation.added` events,
    /// folded into the finalized block when the final part lacks them.
    annotations: Vec<Value>,
    /// Final part payload from `content_part.done`, used when the final
    /// item omits the part.
    final_part: Option<Value>,
}

/// Streaming state of one output item.
#[derive(Debug)]
enum ItemState {
    /// An assistant `message` item; blocks are its content parts.
    Message {
        /// Parts by `content_index`.
        parts: BTreeMap<u64, PartState>,
    },
    /// A `reasoning` item mapped to a single `Thinking` block.
    Reasoning {
        /// Unified block index.
        index: usize,
        /// Whether a summary part was already opened (later parts insert
        /// the `"\n\n"` joiner, matching the non-streaming parse).
        saw_summary_part: bool,
    },
    /// A `function_call` item mapped to a `ToolCall` block.
    FunctionCall {
        /// Unified block index.
        index: usize,
    },
    /// An unmodeled item mapped to an `Opaque` block.
    Opaque {
        /// Unified block index.
        index: usize,
    },
}

impl ItemState {
    /// The block index a sub-event of this item attributes to, if any.
    fn block_index(&self, content_index: Option<u64>) -> Option<usize> {
        match self {
            Self::Message { parts } => content_index.and_then(|ci| parts.get(&ci)).map(|p| p.index),
            Self::Reasoning { index, .. } | Self::FunctionCall { index } | Self::Opaque { index } => {
                Some(*index)
            }
        }
    }
}

/// Stateful stream parser for `openai_responses` (one instance per
/// stream). Implements [`StreamParser`].
#[derive(Debug, Default)]
pub struct ResponsesStreamParser {
    started: bool,
    terminal: bool,
    next_index: usize,
    items: BTreeMap<u64, ItemState>,
    warned_unknown: BTreeSet<String>,
}

impl ResponsesStreamParser {
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

    /// Emits a defensive `MessageStart` if the stream skipped
    /// `response.created`.
    fn ensure_started(&mut self, events: &mut Vec<StreamEvent>) {
        if !self.started {
            self.started = true;
            events.push(StreamEvent::MessageStart { id: None, model: None, usage: None });
        }
    }

    /// Closes every still-open block with `BlockStop { block: None }` (the
    /// accumulated content stands) — used at terminal events for streams
    /// that skipped `output_item.done`.
    fn close_open_blocks(&mut self, events: &mut Vec<StreamEvent>) {
        let mut indices: Vec<usize> = Vec::new();
        for state in self.items.values() {
            match state {
                ItemState::Message { parts } => indices.extend(parts.values().map(|p| p.index)),
                ItemState::Reasoning { index, .. }
                | ItemState::FunctionCall { index }
                | ItemState::Opaque { index } => indices.push(*index),
            }
        }
        indices.sort_unstable();
        for index in indices {
            events.push(StreamEvent::BlockStop { index, block: None });
        }
        self.items.clear();
    }

    fn on_created(&mut self, payload: &Value, events: &mut Vec<StreamEvent>) {
        self.started = true;
        let resp = payload.get("response");
        events.push(StreamEvent::MessageStart {
            id: resp
                .and_then(|r| r.get("id"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            model: resp
                .and_then(|r| r.get("model"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            usage: resp.and_then(|r| r.get("usage")).and_then(to_ir::usage_from_value),
        });
    }

    fn on_item_added(&mut self, payload: &Value, events: &mut Vec<StreamEvent>) {
        let Some(idx) = payload.get("output_index").and_then(Value::as_u64) else { return };
        let item = payload.get("item").cloned().unwrap_or(Value::Null);
        self.ensure_started(events);
        match item.get("type").and_then(Value::as_str) {
            Some("message") => {
                self.items.insert(idx, ItemState::Message { parts: BTreeMap::new() });
            }
            Some("reasoning") => {
                let index = self.alloc_index();
                let mut ns = Map::new();
                if let Some(id) = item.get("id").and_then(Value::as_str) {
                    ns.insert("id".to_owned(), Value::from(id));
                }
                let block = ContentBlock::Thinking {
                    text: None,
                    signature: item
                        .get("encrypted_content")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    extra: Extra::from_unknown(FORMAT, ns),
                };
                self.items.insert(idx, ItemState::Reasoning { index, saw_summary_part: false });
                events.push(StreamEvent::BlockStart { index, block });
            }
            Some("function_call") => {
                let index = self.alloc_index();
                let mut ns = Map::new();
                if let Some(id) = item.get("id").and_then(Value::as_str) {
                    ns.insert("id".to_owned(), Value::from(id));
                }
                let block = ContentBlock::ToolCall {
                    id: item.get("call_id").and_then(Value::as_str).map(str::to_owned),
                    name: item
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    arguments: item
                        .get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    cache: None,
                    extra: Extra::from_unknown(FORMAT, ns),
                };
                self.items.insert(idx, ItemState::FunctionCall { index });
                events.push(StreamEvent::BlockStart { index, block });
            }
            _ => {
                let index = self.alloc_index();
                self.items.insert(idx, ItemState::Opaque { index });
                events
                    .push(StreamEvent::BlockStart { index, block: ContentBlock::opaque(FORMAT, item) });
            }
        }
    }

    fn on_part_added(
        &mut self,
        payload: &Value,
        events: &mut Vec<StreamEvent>,
        warnings: &mut Vec<ConversionWarning>,
    ) {
        let (Some(idx), Some(ci)) = (
            payload.get("output_index").and_then(Value::as_u64),
            payload.get("content_index").and_then(Value::as_u64),
        ) else {
            self.on_unknown("response.content_part.added", payload, events, warnings);
            return;
        };
        self.ensure_started(events);
        let part = payload.get("part").cloned().unwrap_or(Value::Null);
        let mut ns = Map::new();
        if let Some(item_id) = payload.get("item_id").and_then(Value::as_str) {
            ns.insert("id".to_owned(), Value::from(item_id));
        }
        let block = match part.get("type").and_then(Value::as_str) {
            Some("refusal") => {
                ns.insert("refusal".to_owned(), Value::from(true));
                ContentBlock::Text {
                    text: part
                        .get("refusal")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    cache: None,
                    extra: Extra::from_unknown(FORMAT, ns),
                }
            }
            Some("output_text" | "input_text") => ContentBlock::Text {
                text: part.get("text").and_then(Value::as_str).unwrap_or_default().to_owned(),
                cache: None,
                extra: Extra::from_unknown(FORMAT, ns),
            },
            _ => ContentBlock::opaque(FORMAT, part.clone()),
        };
        let index = self.alloc_index();
        let state = self
            .items
            .entry(idx)
            .or_insert_with(|| ItemState::Message { parts: BTreeMap::new() });
        if let ItemState::Message { parts } = state {
            parts.insert(ci, PartState { index, annotations: Vec::new(), final_part: None });
            events.push(StreamEvent::BlockStart { index, block });
        } else if let Some(index) = state.block_index(None) {
            // A part event for a non-message item: surface it on that block.
            events.push(StreamEvent::BlockDelta { index, delta: BlockDelta::Other(payload.clone()) });
        }
    }

    /// Attributes an event to a block via `output_index` /
    /// `content_index`, or reports it as unknown.
    fn attributed_index(&self, payload: &Value) -> Option<usize> {
        let idx = payload.get("output_index").and_then(Value::as_u64)?;
        let state = self.items.get(&idx)?;
        state.block_index(payload.get("content_index").and_then(Value::as_u64))
    }

    fn on_text_delta(
        &mut self,
        event_type: &str,
        payload: &Value,
        events: &mut Vec<StreamEvent>,
        warnings: &mut Vec<ConversionWarning>,
    ) {
        let delta = payload.get("delta").and_then(Value::as_str);
        match (self.attributed_index(payload), delta) {
            (Some(index), Some(delta)) => {
                events.push(StreamEvent::BlockDelta {
                    index,
                    delta: BlockDelta::Text(delta.to_owned()),
                });
            }
            _ => self.on_unknown(event_type, payload, events, warnings),
        }
    }

    fn on_annotation_added(
        &mut self,
        payload: &Value,
        events: &mut Vec<StreamEvent>,
        warnings: &mut Vec<ConversionWarning>,
    ) {
        let idx = payload.get("output_index").and_then(Value::as_u64);
        let ci = payload.get("content_index").and_then(Value::as_u64);
        let part = idx
            .zip(ci)
            .and_then(|(idx, ci)| match self.items.get_mut(&idx) {
                Some(ItemState::Message { parts }) => parts.get_mut(&ci),
                _ => None,
            });
        if let Some(part) = part {
            if let Some(annotation) = payload.get("annotation") {
                part.annotations.push(annotation.clone());
            }
            let index = part.index;
            events.push(StreamEvent::BlockDelta { index, delta: BlockDelta::Other(payload.clone()) });
        } else {
            self.on_unknown("response.output_text.annotation.added", payload, events, warnings);
        }
    }

    fn on_arguments_delta(
        &mut self,
        payload: &Value,
        events: &mut Vec<StreamEvent>,
        warnings: &mut Vec<ConversionWarning>,
    ) {
        let idx = payload.get("output_index").and_then(Value::as_u64);
        let delta = payload.get("delta").and_then(Value::as_str);
        match (idx.and_then(|i| self.items.get(&i)), delta) {
            (Some(ItemState::FunctionCall { index }), Some(delta)) => {
                events.push(StreamEvent::BlockDelta {
                    index: *index,
                    delta: BlockDelta::ToolArguments(delta.to_owned()),
                });
            }
            _ => self.on_unknown("response.function_call_arguments.delta", payload, events, warnings),
        }
    }

    fn on_summary_part_added(
        &mut self,
        payload: &Value,
        events: &mut Vec<StreamEvent>,
        warnings: &mut Vec<ConversionWarning>,
    ) {
        let idx = payload.get("output_index").and_then(Value::as_u64);
        match idx.and_then(|i| self.items.get_mut(&i)) {
            Some(ItemState::Reasoning { index, saw_summary_part }) => {
                let index = *index;
                if *saw_summary_part {
                    // Later summary parts join with a blank line, matching
                    // the non-streaming summary join.
                    events.push(StreamEvent::BlockDelta {
                        index,
                        delta: BlockDelta::Thinking("\n\n".to_owned()),
                    });
                }
                *saw_summary_part = true;
                if let Some(text) = payload
                    .pointer("/part/text")
                    .and_then(Value::as_str)
                    .filter(|t| !t.is_empty())
                {
                    events.push(StreamEvent::BlockDelta {
                        index,
                        delta: BlockDelta::Thinking(text.to_owned()),
                    });
                }
            }
            _ => self.on_unknown("response.reasoning_summary_part.added", payload, events, warnings),
        }
    }

    fn on_summary_text_delta(
        &mut self,
        payload: &Value,
        events: &mut Vec<StreamEvent>,
        warnings: &mut Vec<ConversionWarning>,
    ) {
        let idx = payload.get("output_index").and_then(Value::as_u64);
        let delta = payload.get("delta").and_then(Value::as_str);
        match (idx.and_then(|i| self.items.get(&i)), delta) {
            (Some(ItemState::Reasoning { index, .. }), Some(delta)) => {
                events.push(StreamEvent::BlockDelta {
                    index: *index,
                    delta: BlockDelta::Thinking(delta.to_owned()),
                });
            }
            _ => self.on_unknown("response.reasoning_summary_text.delta", payload, events, warnings),
        }
    }

    fn on_reasoning_text_delta(
        &mut self,
        payload: &Value,
        events: &mut Vec<StreamEvent>,
        warnings: &mut Vec<ConversionWarning>,
    ) {
        // Raw reasoning content (`content` array) is unmodeled: surface it
        // for real-time consumers; `output_item.done` folds the final
        // `content` array into the block extra.
        let idx = payload.get("output_index").and_then(Value::as_u64);
        match idx.and_then(|i| self.items.get(&i)) {
            Some(ItemState::Reasoning { index, .. }) => {
                events.push(StreamEvent::BlockDelta {
                    index: *index,
                    delta: BlockDelta::Other(payload.clone()),
                });
            }
            _ => self.on_unknown("response.reasoning_text.delta", payload, events, warnings),
        }
    }

    fn on_part_done(
        &mut self,
        payload: &Value,
        events: &mut Vec<StreamEvent>,
        warnings: &mut Vec<ConversionWarning>,
    ) {
        let idx = payload.get("output_index").and_then(Value::as_u64);
        let ci = payload.get("content_index").and_then(Value::as_u64);
        let part = idx
            .zip(ci)
            .and_then(|(idx, ci)| match self.items.get_mut(&idx) {
                Some(ItemState::Message { parts }) => parts.get_mut(&ci),
                _ => None,
            });
        match part {
            Some(part) => part.final_part = payload.get("part").cloned(),
            None => self.on_unknown("response.content_part.done", payload, events, warnings),
        }
    }

    fn on_item_done(
        &mut self,
        payload: &Value,
        events: &mut Vec<StreamEvent>,
        warnings: &mut Vec<ConversionWarning>,
    ) {
        let Some(idx) = payload.get("output_index").and_then(Value::as_u64) else {
            self.on_unknown("response.output_item.done", payload, events, warnings);
            return;
        };
        let item = payload.get("item").cloned().unwrap_or(Value::Null);
        let ptr = format!("/output/{idx}");
        self.ensure_started(events);
        let Some(state) = self.items.remove(&idx) else {
            // The stream never announced this item: synthesize start+stop
            // pairs from the final item.
            for block in to_ir::assistant_item_to_blocks(&item, &ptr, warnings) {
                let index = self.alloc_index();
                events.push(StreamEvent::BlockStart { index, block: block.clone() });
                events.push(StreamEvent::BlockStop { index, block: Some(block) });
            }
            return;
        };
        match state {
            ItemState::Message { parts } => {
                let (meta, final_parts) = to_ir::assistant_message_meta(&item);
                for (ci, part_state) in &parts {
                    let source = usize::try_from(*ci)
                        .ok()
                        .and_then(|ci| final_parts.get(ci))
                        .or(part_state.final_part.as_ref());
                    let block = source.map(|p| {
                        let block = to_ir::assistant_text_block_from_part(p, &meta)
                            .unwrap_or_else(|| ContentBlock::opaque(FORMAT, p.clone()));
                        fold_annotations(block, &part_state.annotations)
                    });
                    events.push(StreamEvent::BlockStop { index: part_state.index, block });
                }
                for (ci, part) in final_parts.iter().enumerate() {
                    if parts.contains_key(&(ci as u64)) {
                        continue;
                    }
                    let block = to_ir::assistant_text_block_from_part(part, &meta)
                        .unwrap_or_else(|| ContentBlock::opaque(FORMAT, part.clone()));
                    let index = self.alloc_index();
                    events.push(StreamEvent::BlockStart { index, block: block.clone() });
                    events.push(StreamEvent::BlockStop { index, block: Some(block) });
                }
            }
            ItemState::Reasoning { index, .. } => {
                let block = to_ir::thinking_from_reasoning(&item, &ptr, warnings);
                events.push(StreamEvent::BlockStop { index, block: Some(block) });
            }
            ItemState::FunctionCall { index } => {
                let block = to_ir::function_call_to_block(&item, &ptr, warnings);
                events.push(StreamEvent::BlockStop { index, block: Some(block) });
            }
            ItemState::Opaque { index } => {
                events.push(StreamEvent::BlockStop {
                    index,
                    block: Some(ContentBlock::opaque(FORMAT, item)),
                });
            }
        }
    }

    fn on_terminal(&mut self, payload: &Value, incomplete: bool, events: &mut Vec<StreamEvent>) {
        self.ensure_started(events);
        self.close_open_blocks(events);
        self.terminal = true;
        let resp = payload.get("response");
        let has_function_call = resp
            .and_then(|r| r.get("output"))
            .and_then(Value::as_array)
            .is_some_and(|items| {
                items
                    .iter()
                    .any(|i| i.get("type").and_then(Value::as_str) == Some("function_call"))
            });
        let (status, reason) = if incomplete {
            (
                Some("incomplete"),
                resp.and_then(|r| r.pointer("/incomplete_details/reason"))
                    .and_then(Value::as_str),
            )
        } else {
            (Some("completed"), None)
        };
        let stop_reason = to_ir::derive_stop_reason(status, reason, has_function_call);
        let usage = resp.and_then(|r| r.get("usage")).and_then(to_ir::usage_from_value);
        events.push(StreamEvent::MessageDelta { stop_reason, usage });
        events.push(StreamEvent::MessageStop);
    }

    fn on_unknown(
        &mut self,
        event_type: &str,
        payload: &Value,
        events: &mut Vec<StreamEvent>,
        warnings: &mut Vec<ConversionWarning>,
    ) {
        if let Some(index) = self.attributed_index(payload) {
            // Recognizable coordinates: surface the payload on that block
            // (built-in tool progress, MCP argument streams, …).
            events.push(StreamEvent::BlockDelta { index, delta: BlockDelta::Other(payload.clone()) });
            return;
        }
        events.push(StreamEvent::Unknown);
        if self.warned_unknown.insert(event_type.to_owned()) {
            warnings.push(warn(
                WarningCode::UnknownStreamEvent,
                "/type",
                format!("unrecognized stream event `{event_type}`"),
            ));
        }
    }
}

/// Folds annotations collected from `annotation.added` events into a
/// finalized `Text` block when the final part payload lacks them.
fn fold_annotations(mut block: ContentBlock, annotations: &[Value]) -> ContentBlock {
    if annotations.is_empty() {
        return block;
    }
    if let ContentBlock::Text { extra, .. } = &mut block {
        let ns = extra.namespace_mut(FORMAT);
        if !ns.contains_key("annotations") {
            ns.insert("annotations".to_owned(), Value::Array(annotations.to_vec()));
        }
    }
    block
}

impl StreamParser for ResponsesStreamParser {
    fn parse(&mut self, event: &SseEvent) -> Result<(Vec<StreamEvent>, Vec<ConversionWarning>)> {
        let mut events = Vec::new();
        let mut warnings = Vec::new();
        let payload: Value = match serde_json::from_str(&event.data) {
            Ok(v) => v,
            Err(e) => {
                warnings.push(warn(
                    WarningCode::MalformedField,
                    "",
                    format!("stream event data is not JSON and was surfaced as unknown: {e}"),
                ));
                events.push(StreamEvent::Unknown);
                return Ok((events, warnings));
            }
        };
        // The payload `type` mirrors the SSE event name and is
        // authoritative; fall back to the `event:` field.
        let event_type = payload
            .get("type")
            .and_then(Value::as_str)
            .or(event.event.as_deref())
            .unwrap_or_default()
            .to_owned();
        match event_type.as_str() {
            "response.created" => self.on_created(&payload, &mut events),
            // Lifecycle noise and `.done` duplicates of already-streamed
            // data are consumed silently (contract, "Response parsing").
            "response.queued"
            | "response.in_progress"
            | "response.output_text.done"
            | "response.refusal.done"
            | "response.function_call_arguments.done"
            | "response.reasoning_summary_text.done"
            | "response.reasoning_summary_part.done"
            | "response.reasoning_text.done" => {}
            "response.output_item.added" => self.on_item_added(&payload, &mut events),
            "response.content_part.added" => self.on_part_added(&payload, &mut events, &mut warnings),
            "response.output_text.delta" | "response.refusal.delta" => {
                self.on_text_delta(&event_type, &payload, &mut events, &mut warnings);
            }
            "response.output_text.annotation.added" => {
                self.on_annotation_added(&payload, &mut events, &mut warnings);
            }
            "response.function_call_arguments.delta" => {
                self.on_arguments_delta(&payload, &mut events, &mut warnings);
            }
            "response.reasoning_summary_part.added" => {
                self.on_summary_part_added(&payload, &mut events, &mut warnings);
            }
            "response.reasoning_summary_text.delta" => {
                self.on_summary_text_delta(&payload, &mut events, &mut warnings);
            }
            "response.reasoning_text.delta" => {
                self.on_reasoning_text_delta(&payload, &mut events, &mut warnings);
            }
            "response.content_part.done" => self.on_part_done(&payload, &mut events, &mut warnings),
            "response.output_item.done" => self.on_item_done(&payload, &mut events, &mut warnings),
            "response.completed" => self.on_terminal(&payload, false, &mut events),
            "response.incomplete" => self.on_terminal(&payload, true, &mut events),
            "response.failed" => {
                self.terminal = true;
                let error = payload.pointer("/response/error");
                return Err(to_ir::failed_response_error(
                    200,
                    &http::HeaderMap::new(),
                    error.and_then(|e| e.get("code")).and_then(Value::as_str),
                    error.and_then(|e| e.get("message")).and_then(Value::as_str),
                    event.data.as_bytes(),
                    Some(payload.clone()),
                ));
            }
            "error" => {
                self.terminal = true;
                return Err(to_ir::failed_response_error(
                    200,
                    &http::HeaderMap::new(),
                    payload.get("code").and_then(Value::as_str),
                    payload.get("message").and_then(Value::as_str),
                    event.data.as_bytes(),
                    Some(payload.clone()),
                ));
            }
            other => self.on_unknown(other, &payload, &mut events, &mut warnings),
        }
        Ok((events, warnings))
    }

    fn finish(&mut self) -> Result<(Vec<StreamEvent>, Vec<ConversionWarning>)> {
        if self.terminal {
            Ok((Vec::new(), Vec::new()))
        } else {
            Err(Error::Parse {
                message: "truncated stream: the Responses stream ended without a terminal event"
                    .to_owned(),
                raw: bytes::Bytes::new(),
            })
        }
    }
}
