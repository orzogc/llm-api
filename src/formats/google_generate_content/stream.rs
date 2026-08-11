//! Streaming parser for `streamGenerateContent?alt=sse` (§ 9).
//!
//! Each SSE event's `data` is a partial `GenerateContentResponse`. Block
//! boundaries are inferred from the parts of the first candidate: same-kind
//! text parts continue the open block (`text` continues `text`, `thought`
//! continues `thought`) while a kind switch, a signature already sealed onto
//! the open block, or any whole part (`functionCall`, media, opaque) closes
//! it. The stream terminates on the first candidate's `finishReason` or on a
//! blocked-prompt chunk; a silent EOF without either is a truncated stream.
//! An error envelope (`data: {"error": {...}}`) raises [`Error::Api`]; a
//! chunk with no modeled signal at all surfaces as an `Unknown` event with
//! a once-per-stream warning.

use serde_json::{Map, Value, json};

use crate::convert::{ConversionWarning, WarningCode};
use crate::error::{Error, Result};
use crate::format::StreamParser;
use crate::format::ids::GOOGLE_GENERATE_CONTENT as ID;
use crate::http::SseEvent;
use crate::ir::{BlockDelta, ContentBlock, Extra, StopReason, StreamEvent, Usage};

use super::to_ir::{AssistantPart, classify_assistant_part, map_finish_reason, text_like_block};
use super::types;

/// An incrementally built text-like block.
struct OpenBlock {
    index: usize,
    thought: bool,
    text: String,
    signature: Option<String>,
    ns: Map<String, Value>,
}

impl OpenBlock {
    /// `true` when a same-kind part may continue this block. A block that
    /// already carries a signature is sealed — signatures are validated
    /// against the exact part they were issued for (§ 7.5).
    fn accepts(&self, thought: bool) -> bool {
        self.thought == thought
            && self.signature.is_none()
            && !self.ns.contains_key("thoughtSignature")
    }

    /// Builds the finalized block (signature and unknown part fields folded
    /// in).
    fn finalize(self) -> ContentBlock {
        text_like_block(self.thought, self.text, self.signature, self.ns)
    }
}

/// Stateful stream parser for the Google format; one instance per stream.
#[derive(Default)]
pub struct GoogleStreamParser {
    started: bool,
    terminated: bool,
    next_index: usize,
    open: Option<OpenBlock>,
    last_usage: Option<Usage>,
    multi_candidate_warned: bool,
    empty_chunk_warned: bool,
}

impl GoogleStreamParser {
    /// A fresh parser.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn close_open(&mut self, events: &mut Vec<StreamEvent>) {
        if let Some(open) = self.open.take() {
            let index = open.index;
            events.push(StreamEvent::BlockStop {
                index,
                block: Some(open.finalize()),
            });
        }
    }

    fn warn_multi_candidate(&mut self, warnings: &mut Vec<ConversionWarning>) {
        if !self.multi_candidate_warned {
            warnings.push(ConversionWarning::from_format(
                WarningCode::MultipleCandidates,
                ID,
                "/candidates",
                "stream carries chunks for more than one candidate; only the first is parsed",
            ));
            self.multi_candidate_warned = true;
        }
    }

    fn open_block(&mut self, thought: bool, events: &mut Vec<StreamEvent>) {
        self.close_open(events);
        let index = self.next_index;
        self.next_index += 1;
        let block = if thought {
            let mut ns = Map::new();
            ns.insert("thought".to_owned(), json!(true));
            ContentBlock::Thinking {
                text: Some(String::new()),
                signature: None,
                extra: Extra::from_unknown(ID, ns),
            }
        } else {
            ContentBlock::text("")
        };
        events.push(StreamEvent::BlockStart { index, block });
        self.open = Some(OpenBlock {
            index,
            thought,
            text: String::new(),
            signature: None,
            ns: Map::new(),
        });
    }

    /// Processes one text-like part: continues the open block or opens a new
    /// one, emitting the corresponding deltas.
    fn text_part(
        &mut self,
        thought: bool,
        text: String,
        signature: Option<String>,
        ns: Map<String, Value>,
        events: &mut Vec<StreamEvent>,
    ) {
        let continue_open = self.open.as_ref().is_some_and(|o| o.accepts(thought));
        if !continue_open {
            self.open_block(thought, events);
        }
        let open = self.open.as_mut().expect("block just ensured open");
        if !text.is_empty() {
            let delta = if thought {
                BlockDelta::Thinking(text.clone())
            } else {
                BlockDelta::Text(text.clone())
            };
            events.push(StreamEvent::BlockDelta {
                index: open.index,
                delta,
            });
            open.text.push_str(&text);
        }
        if let Some(sig) = signature {
            if thought {
                events.push(StreamEvent::BlockDelta {
                    index: open.index,
                    delta: BlockDelta::Signature(sig.clone()),
                });
                open.signature = Some(sig);
            } else {
                // The accumulator accepts Signature deltas only on thinking
                // blocks; surface the payload as Other and fold it into the
                // finalized block.
                events.push(StreamEvent::BlockDelta {
                    index: open.index,
                    delta: BlockDelta::Other(json!({"thoughtSignature": sig})),
                });
                open.ns.insert("thoughtSignature".to_owned(), json!(sig));
            }
        }
        if !ns.is_empty() {
            events.push(StreamEvent::BlockDelta {
                index: open.index,
                delta: BlockDelta::Other(Value::Object(ns.clone())),
            });
            open.ns.extend(ns);
        }
    }

    /// Emits a whole (non-text) block: `BlockStart` with the complete block
    /// plus an immediate `BlockStop`.
    fn whole_block(&mut self, block: ContentBlock, events: &mut Vec<StreamEvent>) {
        self.close_open(events);
        let index = self.next_index;
        self.next_index += 1;
        events.push(StreamEvent::BlockStart { index, block });
        events.push(StreamEvent::BlockStop { index, block: None });
    }

    fn terminate(&mut self, stop_reason: StopReason, events: &mut Vec<StreamEvent>) {
        self.close_open(events);
        events.push(StreamEvent::MessageDelta {
            stop_reason: Some(stop_reason),
            usage: self.last_usage.clone(),
        });
        events.push(StreamEvent::MessageStop);
        self.terminated = true;
    }
}

impl StreamParser for GoogleStreamParser {
    fn parse(&mut self, event: &SseEvent) -> Result<(Vec<StreamEvent>, Vec<ConversionWarning>)> {
        let mut warnings = Vec::new();
        let Ok(chunk) = serde_json::from_str::<types::GenerateContentResponse>(&event.data) else {
            warnings.push(ConversionWarning::from_format(
                WarningCode::UnknownStreamEvent,
                ID,
                "",
                "stream data is not a GenerateContentResponse chunk",
            ));
            return Ok((vec![StreamEvent::Unknown], warnings));
        };

        // Pick the first candidate with index 0 (or no index); everything
        // else is unsupported multi-candidate output (§ 8).
        let mut primary: Option<&types::Candidate> = None;
        let mut extras = false;
        for candidate in &chunk.candidates {
            if candidate.index.unwrap_or(0) == 0 && primary.is_none() {
                primary = Some(candidate);
            } else {
                extras = true;
            }
        }

        if self.terminated {
            // Data after the protocol terminator cannot be attributed.
            if extras {
                self.warn_multi_candidate(&mut warnings);
            } else {
                warnings.push(ConversionWarning::from_format(
                    WarningCode::UnknownStreamEvent,
                    ID,
                    "",
                    "chunk received after the stream terminated",
                ));
            }
            return Ok((vec![StreamEvent::Unknown], warnings));
        }

        // A mid-stream error envelope (`data: {"error": {...}}`) replaces
        // normal flow: raise the API error instead of consuming the chunk
        // silently (first chunk or later; after the terminator the gate
        // above downgrades it like any other unattributable data).
        if chunk.extra.contains_key("error") {
            return Err(super::stream_error(&event.data));
        }

        // A chunk with no modeled signal at all produces no events and
        // would otherwise vanish: surface it (aligned with the non-stream
        // "neither candidates nor a prompt block reason" check). `Unknown`
        // is emitted per chunk so `include_raw` exposes each one; the
        // warning fires once per stream.
        if chunk.candidates.is_empty()
            && chunk.prompt_feedback.is_none()
            && chunk.usage_metadata.is_none()
            && chunk.model_version.is_none()
            && chunk.response_id.is_none()
        {
            if !self.empty_chunk_warned {
                self.empty_chunk_warned = true;
                warnings.push(ConversionWarning::from_format(
                    WarningCode::MalformedField,
                    ID,
                    "/candidates",
                    "chunk carries neither candidates nor a prompt block reason",
                ));
            }
            return Ok((vec![StreamEvent::Unknown], warnings));
        }

        let mut events = Vec::new();
        if let Some(value) = &chunk.usage_metadata
            && let Some(usage) = super::to_ir::usage_from_value(value, &mut warnings)
        {
            self.last_usage = Some(usage);
        }
        if !self.started {
            self.started = true;
            events.push(StreamEvent::MessageStart {
                id: chunk.response_id.clone(),
                model: chunk.model_version.clone(),
                usage: self.last_usage.clone(),
            });
        }

        let blocked = chunk
            .prompt_feedback
            .as_ref()
            .is_some_and(|pf| pf.block_reason.is_some());
        if blocked && chunk.candidates.is_empty() {
            self.terminate(StopReason::ContentFilter, &mut events);
            return Ok((events, warnings));
        }

        if let Some(candidate) = primary {
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
                            self.text_part(thought, text, signature, ns, &mut events);
                        }
                        AssistantPart::Complete(block) => self.whole_block(block, &mut events),
                    }
                }
            }
            if let Some(reason) = &candidate.finish_reason {
                self.terminate(map_finish_reason(reason), &mut events);
            }
        }
        if extras {
            self.warn_multi_candidate(&mut warnings);
            events.push(StreamEvent::Unknown);
        }
        Ok((events, warnings))
    }

    fn finish(&mut self) -> Result<(Vec<StreamEvent>, Vec<ConversionWarning>)> {
        if self.terminated {
            Ok((Vec::new(), Vec::new()))
        } else {
            Err(Error::TruncatedStream {
                message: "no finishReason or blocked-prompt chunk was seen".to_owned(),
                raw: bytes::Bytes::new(),
            })
        }
    }
}
