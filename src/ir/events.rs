//! The unified stream event model and the accumulator (§ 9).

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::convert::ConversionWarning;
use crate::error::Error;

use super::block::ContentBlock;
use super::message::Message;
use super::response::{Response, StopReason, Usage, normalize_stop_reason};

/// One item delivered by a streaming call.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StreamItem {
    /// The unified event.
    pub event: StreamEvent,
    /// Original payload: always `Some` for `Unknown`, `Some` for every event
    /// when the `include_raw` call option is on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
    /// Parse-side warnings, usually empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<ConversionWarning>,
}

impl StreamItem {
    /// Wraps an event with no raw payload or warnings.
    #[must_use]
    pub fn new(event: StreamEvent) -> Self {
        Self { event, raw: None, warnings: Vec::new() }
    }
}

/// Unified block-level stream events — closest to Anthropic's model,
/// mid-granularity; every format maps into it.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// Stream opened; carries envelope data known up front.
    #[non_exhaustive]
    MessageStart {
        /// Provider response id.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// Model name.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        /// Cumulative usage snapshot, if the protocol provides one here.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
    },
    /// A content block opened; `block` may be partial.
    #[non_exhaustive]
    BlockStart {
        /// Block index within the message.
        index: usize,
        /// The (possibly partial) block.
        block: ContentBlock,
    },
    /// Incremental payload for the block at `index`.
    #[non_exhaustive]
    BlockDelta {
        /// Block index within the message.
        index: usize,
        /// The delta payload.
        delta: BlockDelta,
    },
    /// The block at `index` is complete. `block`, when present, is the
    /// parser's finalized block — citations, annotations, final status and
    /// signatures already folded into `extra`; the accumulator replaces its
    /// incrementally built block with it.
    #[non_exhaustive]
    BlockStop {
        /// Block index within the message.
        index: usize,
        /// The parser-finalized block, when the protocol provides one.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        block: Option<ContentBlock>,
    },
    /// Envelope updates (stop reason, cumulative usage snapshot).
    #[non_exhaustive]
    MessageDelta {
        /// Stop reason, once known.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stop_reason: Option<StopReason>,
        /// Complete cumulative usage snapshot (§ 9): the stateful parser
        /// folds provider partials into a running total before emitting.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
    },
    /// The protocol terminator was seen.
    MessageStop,
    /// Unrecognized event; payload in `StreamItem.raw`.
    Unknown,
}

/// Incremental payload kinds.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum BlockDelta {
    /// Text fragment.
    Text(String),
    /// Thinking-text fragment.
    Thinking(String),
    /// Signature fragment.
    Signature(String),
    /// Tool-argument string fragment (exact bytes as received).
    ToolArguments(String),
    /// Recognized but unmodeled delta payload (Anthropic `citations_delta`,
    /// Responses `annotation.added`, …) — surfaced for real-time consumers
    /// and folded into the finalized block at `BlockStop` by the parser.
    Other(Value),
}

/// Folds [`StreamEvent`]s into a [`Response`] (§ 9).
///
/// Blocks are appended strictly in arrival order and same-typed blocks are
/// never merged — interleaved thinking→text→thinking→text sequences survive
/// verbatim. Usage snapshots are cumulative; the latest wins.
#[derive(Debug, Default)]
pub struct Accumulator {
    id: Option<String>,
    model: Option<String>,
    usage: Option<Usage>,
    stop_reason: Option<StopReason>,
    /// `(index, block)` in arrival order.
    blocks: Vec<(usize, ContentBlock)>,
    started: bool,
    stopped: bool,
    status: u16,
    headers: http::HeaderMap,
    warnings: Vec<ConversionWarning>,
}

impl Accumulator {
    /// A fresh accumulator (status defaults to 200 until
    /// [`Accumulator::set_status`] is called).
    #[must_use]
    pub fn new() -> Self {
        Self { status: 200, ..Self::default() }
    }

    /// Records the HTTP status of the streaming response.
    pub fn set_status(&mut self, status: u16) {
        self.status = status;
    }

    /// Records the HTTP headers of the streaming response.
    pub fn set_headers(&mut self, headers: http::HeaderMap) {
        self.headers = headers;
    }

    /// Seeds warnings (used for request-build warnings from the stream
    /// handle, which the accumulated response must also carry).
    pub fn extend_warnings(&mut self, warnings: impl IntoIterator<Item = ConversionWarning>) {
        self.warnings.extend(warnings);
    }

    fn block_mut(&mut self, index: usize) -> Result<&mut ContentBlock, Error> {
        self.blocks
            .iter_mut()
            .find(|(i, _)| *i == index)
            .map(|(_, b)| b)
            .ok_or_else(|| Error::Parse {
                message: format!("stream delta for unknown block index {index}"),
                raw: bytes::Bytes::new(),
            })
    }

    /// Folds one stream item, including its warnings.
    pub fn push(&mut self, item: &StreamItem) -> Result<(), Error> {
        self.warnings.extend(item.warnings.iter().cloned());
        match &item.event {
            StreamEvent::MessageStart { id, model, usage, .. } => {
                self.started = true;
                if id.is_some() {
                    self.id.clone_from(id);
                }
                if model.is_some() {
                    self.model.clone_from(model);
                }
                if usage.is_some() {
                    self.usage.clone_from(usage);
                }
            }
            StreamEvent::BlockStart { index, block, .. } => {
                if self.blocks.iter().any(|(i, _)| i == index) {
                    return Err(Error::Parse {
                        message: format!("duplicate BlockStart for index {index}"),
                        raw: bytes::Bytes::new(),
                    });
                }
                self.blocks.push((*index, block.clone()));
            }
            StreamEvent::BlockDelta { index, delta, .. } => {
                let block = self.block_mut(*index)?;
                apply_delta(block, delta).map_err(|message| Error::Parse {
                    message: format!("block {index}: {message}"),
                    raw: bytes::Bytes::new(),
                })?;
            }
            StreamEvent::BlockStop { index, block, .. } => {
                if let Some(finalized) = block {
                    *self.block_mut(*index)? = finalized.clone();
                } else {
                    // Validate the index exists even without a replacement.
                    self.block_mut(*index)?;
                }
            }
            StreamEvent::MessageDelta { stop_reason, usage, .. } => {
                if stop_reason.is_some() {
                    self.stop_reason.clone_from(stop_reason);
                }
                if usage.is_some() {
                    self.usage.clone_from(usage);
                }
            }
            StreamEvent::MessageStop => self.stopped = true,
            StreamEvent::Unknown => {}
        }
        Ok(())
    }

    /// Finishes accumulation. Fails with `Error::Parse` when the stream
    /// never delivered its `MessageStop` — a partial stream must not pass a
    /// half response off as complete; the events already delivered remain
    /// the caller's partial record.
    pub fn finish(self) -> Result<Response, Error> {
        if !self.stopped {
            return Err(Error::Parse {
                message: "stream ended without MessageStop; accumulated response is partial"
                    .to_owned(),
                raw: bytes::Bytes::new(),
            });
        }
        let message = Message::assistant(self.blocks.into_iter().map(|(_, b)| b).collect());
        let stop_reason = normalize_stop_reason(&message, self.stop_reason);
        let mut response = Response::new(message);
        response.id = self.id;
        response.model = self.model;
        response.stop_reason = stop_reason;
        response.usage = self.usage;
        response.status = self.status;
        response.headers = self.headers;
        response.raw = None;
        response.warnings = self.warnings;
        Ok(response)
    }
}

/// Applies a delta to an incrementally built block; returns a description on
/// a kind mismatch.
fn apply_delta(block: &mut ContentBlock, delta: &BlockDelta) -> Result<(), String> {
    match (block, delta) {
        (ContentBlock::Text { text, .. }, BlockDelta::Text(fragment)) => {
            text.push_str(fragment);
            Ok(())
        }
        (ContentBlock::Thinking { text, .. }, BlockDelta::Thinking(fragment)) => {
            text.get_or_insert_with(String::new).push_str(fragment);
            Ok(())
        }
        (ContentBlock::Thinking { signature, .. }, BlockDelta::Signature(fragment)) => {
            signature.get_or_insert_with(String::new).push_str(fragment);
            Ok(())
        }
        (ContentBlock::ToolCall { arguments, .. }, BlockDelta::ToolArguments(fragment)) => {
            arguments.push_str(fragment);
            Ok(())
        }
        // Unmodeled payloads are folded into the finalized block by the
        // parser at BlockStop; the accumulator ignores them.
        (_, BlockDelta::Other(_)) => Ok(()),
        (block, delta) => Err(format!(
            "delta kind does not match block kind ({} block, {:?} delta)",
            block.kind_name(),
            delta_name(delta)
        )),
    }
}

fn delta_name(delta: &BlockDelta) -> &'static str {
    match delta {
        BlockDelta::Text(_) => "Text",
        BlockDelta::Thinking(_) => "Thinking",
        BlockDelta::Signature(_) => "Signature",
        BlockDelta::ToolArguments(_) => "ToolArguments",
        BlockDelta::Other(_) => "Other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn item(event: StreamEvent) -> StreamItem {
        StreamItem::new(event)
    }

    #[test]
    fn accumulates_interleaved_blocks_in_arrival_order() {
        let mut acc = Accumulator::new();
        let events = [
            StreamEvent::MessageStart { id: Some("m1".into()), model: Some("x".into()), usage: None },
            StreamEvent::BlockStart { index: 0, block: ContentBlock::thinking("") },
            StreamEvent::BlockDelta { index: 0, delta: BlockDelta::Thinking("think ".into()) },
            StreamEvent::BlockStart { index: 1, block: ContentBlock::text("") },
            StreamEvent::BlockDelta { index: 1, delta: BlockDelta::Text("hello".into()) },
            StreamEvent::BlockDelta { index: 0, delta: BlockDelta::Thinking("more".into()) },
            StreamEvent::BlockStart { index: 2, block: ContentBlock::text("") },
            StreamEvent::BlockDelta { index: 2, delta: BlockDelta::Text("!".into()) },
            StreamEvent::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: Some(Usage { input_tokens: 1, output_tokens: 2, ..Usage::default() }),
            },
            StreamEvent::MessageStop,
        ];
        for e in events {
            acc.push(&item(e)).unwrap();
        }
        let resp = acc.finish().unwrap();
        assert_eq!(resp.id.as_deref(), Some("m1"));
        assert_eq!(resp.message.content.len(), 3);
        assert!(matches!(
            &resp.message.content[0],
            ContentBlock::Thinking { text: Some(t), .. } if t == "think more"
        ));
        assert_eq!(resp.text(), "hello!");
        assert_eq!(resp.stop_reason, Some(StopReason::EndTurn));
        assert_eq!(resp.usage.as_ref().unwrap().output_tokens, 2);
        assert!(resp.raw.is_none());
    }

    #[test]
    fn tool_call_arguments_accumulate_and_stop_reason_normalizes() {
        let mut acc = Accumulator::new();
        for e in [
            StreamEvent::MessageStart { id: None, model: None, usage: None },
            StreamEvent::BlockStart { index: 0, block: ContentBlock::tool_call_with_id("c", "f", "") },
            StreamEvent::BlockDelta { index: 0, delta: BlockDelta::ToolArguments("{\"a\":".into()) },
            StreamEvent::BlockDelta { index: 0, delta: BlockDelta::ToolArguments("1}".into()) },
            StreamEvent::BlockStop { index: 0, block: None },
            StreamEvent::MessageDelta { stop_reason: Some(StopReason::EndTurn), usage: None },
            StreamEvent::MessageStop,
        ] {
            acc.push(&item(e)).unwrap();
        }
        let resp = acc.finish().unwrap();
        assert!(matches!(
            &resp.message.content[0],
            ContentBlock::ToolCall { arguments, .. } if arguments == "{\"a\":1}"
        ));
        // EndTurn + tool call normalizes to ToolUse.
        assert_eq!(resp.stop_reason, Some(StopReason::ToolUse));
    }

    #[test]
    fn finalized_block_replaces_built_block() {
        let mut acc = Accumulator::new();
        let finalized = ContentBlock::thinking_signed("full text", "sig");
        for e in [
            StreamEvent::BlockStart { index: 0, block: ContentBlock::thinking("") },
            StreamEvent::BlockDelta { index: 0, delta: BlockDelta::Thinking("partial".into()) },
            StreamEvent::BlockStop { index: 0, block: Some(finalized.clone()) },
            StreamEvent::MessageStop,
        ] {
            acc.push(&item(e)).unwrap();
        }
        let resp = acc.finish().unwrap();
        assert_eq!(resp.message.content[0], finalized);
    }

    #[test]
    fn missing_message_stop_fails() {
        let mut acc = Accumulator::new();
        acc.push(&item(StreamEvent::MessageStart { id: None, model: None, usage: None })).unwrap();
        assert!(matches!(acc.finish(), Err(Error::Parse { .. })));
    }

    #[test]
    fn delta_kind_mismatch_fails() {
        let mut acc = Accumulator::new();
        acc.push(&item(StreamEvent::BlockStart { index: 0, block: ContentBlock::text("") }))
            .unwrap();
        let err = acc.push(&item(StreamEvent::BlockDelta {
            index: 0,
            delta: BlockDelta::Thinking("x".into()),
        }));
        assert!(matches!(err, Err(Error::Parse { .. })));
    }

    #[test]
    fn unknown_index_fails() {
        let mut acc = Accumulator::new();
        let err = acc
            .push(&item(StreamEvent::BlockDelta { index: 7, delta: BlockDelta::Text("x".into()) }));
        assert!(matches!(err, Err(Error::Parse { .. })));
    }

    #[test]
    fn usage_snapshot_latest_wins() {
        let mut acc = Accumulator::new();
        for e in [
            StreamEvent::MessageStart {
                id: None,
                model: None,
                usage: Some(Usage { input_tokens: 10, output_tokens: 0, ..Usage::default() }),
            },
            StreamEvent::MessageDelta {
                stop_reason: None,
                usage: Some(Usage { input_tokens: 10, output_tokens: 7, ..Usage::default() }),
            },
            StreamEvent::MessageStop,
        ] {
            acc.push(&item(e)).unwrap();
        }
        let resp = acc.finish().unwrap();
        assert_eq!(resp.usage.as_ref().unwrap().output_tokens, 7);
    }

    #[test]
    fn stream_event_serde() {
        let e = StreamEvent::BlockDelta { index: 1, delta: BlockDelta::Text("hi".into()) };
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(
            v,
            json!({"type": "block_delta", "index": 1, "delta": {"type": "text", "value": "hi"}})
        );
        assert_eq!(serde_json::from_value::<StreamEvent>(v).unwrap(), e);
        assert_eq!(
            serde_json::to_value(StreamEvent::MessageStop).unwrap(),
            json!({"type": "message_stop"})
        );
    }
}
