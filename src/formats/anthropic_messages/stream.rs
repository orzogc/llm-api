//! Anthropic SSE events → unified [`StreamEvent`]s (near-direct mapping,
//! § 9).

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use crate::convert::{ConversionWarning, WarningCode};
use crate::error::{ApiErrorKind, Error, Result};
use crate::format::StreamParser;
use crate::http::SseEvent;
use crate::ir::{BlockDelta, CacheHint, ContentBlock, Extra, StopReason, StreamEvent};

use super::to_ir::unify_usage;
use super::types::{
    ContentBlockDeltaEvent, ContentBlockStartEvent, ContentBlockStopEvent, ErrorEnvelope,
    MessageDeltaEvent, MessageStartEvent, RedactedThinkingBlock, TextBlock, ThinkingBlock,
    ToolUseBlock, UsageWire,
};
use super::{FORMAT_ID as FMT, map_error_kind};

fn pwarn(code: WarningCode, location: &str, text: impl Into<String>) -> ConversionWarning {
    ConversionWarning::from_format(code, FMT, location, text)
}

fn perr(message: impl Into<String>, raw: &str) -> Error {
    Error::Parse {
        message: message.into(),
        raw: bytes::Bytes::copy_from_slice(raw.as_bytes()),
    }
}

/// Per-block accumulation state for finalization at `content_block_stop`.
enum OpenBlock {
    Text {
        text: String,
        extra: Map<String, Value>,
        cache: Option<CacheHint>,
        citations: Vec<Value>,
    },
    Thinking {
        text: String,
        signature: String,
        extra: Map<String, Value>,
    },
    RedactedThinking {
        data: String,
        extra: Map<String, Value>,
    },
    ToolUse {
        id: String,
        name: String,
        json: String,
        start_input: Value,
        extra: Map<String, Value>,
        cache: Option<CacheHint>,
    },
    Opaque {
        value: Value,
        json: String,
    },
}

/// Stateful per-stream parser (one instance per stream, § 11).
#[derive(Default)]
pub(crate) struct AnthropicStreamParser {
    /// Latest raw usage snapshot: seeded by `message_start`, overlaid by
    /// every `message_delta` (whose usage is cumulative but may omit the
    /// input-side fields).
    usage: Map<String, Value>,
    blocks: BTreeMap<usize, OpenBlock>,
    terminated: bool,
}

impl AnthropicStreamParser {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Overlays `delta` usage fields onto the cached snapshot and returns
    /// the unified cumulative usage.
    fn merge_usage(&mut self, delta: Option<&Map<String, Value>>) -> Option<crate::ir::Usage> {
        if let Some(delta) = delta {
            for (k, v) in delta {
                self.usage.insert(k.clone(), v.clone());
            }
        }
        if self.usage.is_empty() {
            return None;
        }
        let raw = Value::Object(self.usage.clone());
        let wire: UsageWire = serde_json::from_value(raw.clone()).unwrap_or_default();
        Some(unify_usage(&wire, raw))
    }

    fn start_block(
        &mut self,
        index: usize,
        content_block: Value,
        warnings: &mut Vec<ConversionWarning>,
    ) -> ContentBlock {
        let loc = format!("/content_block_start/{index}");
        let (open, start) = match content_block.get("type").and_then(Value::as_str) {
            Some("text") => match serde_json::from_value::<TextBlock>(content_block.clone()) {
                Ok(t) => {
                    let mut rest = t.extra;
                    let cache = super::to_ir::split_cache_control(t.cache_control, &mut rest);
                    let start = ContentBlock::Text {
                        text: t.text.clone(),
                        cache: cache.clone(),
                        extra: Extra::from_unknown(FMT, rest.clone()),
                    };
                    (
                        OpenBlock::Text {
                            text: t.text,
                            extra: rest,
                            cache,
                            citations: Vec::new(),
                        },
                        start,
                    )
                }
                Err(_) => opaque_open(content_block, &loc, warnings),
            },
            Some("thinking") => {
                match serde_json::from_value::<ThinkingBlock>(content_block.clone()) {
                    Ok(t) => {
                        // The opening block carries `signature: ""`; an empty
                        // signature is treated as absent until deltas arrive.
                        let signature = t.signature.unwrap_or_default();
                        let start = ContentBlock::Thinking {
                            text: Some(t.thinking.clone()),
                            signature: if signature.is_empty() {
                                None
                            } else {
                                Some(signature.clone())
                            },
                            extra: Extra::from_unknown(FMT, t.extra.clone()),
                        };
                        (
                            OpenBlock::Thinking {
                                text: t.thinking,
                                signature,
                                extra: t.extra,
                            },
                            start,
                        )
                    }
                    Err(_) => opaque_open(content_block, &loc, warnings),
                }
            }
            Some("redacted_thinking") => {
                match serde_json::from_value::<RedactedThinkingBlock>(content_block.clone()) {
                    Ok(t) => {
                        let mut extra = t.extra.clone();
                        extra.insert("redacted".to_owned(), Value::Bool(true));
                        let start = ContentBlock::Thinking {
                            text: None,
                            signature: Some(t.data.clone()),
                            extra: Extra::from_unknown(FMT, extra),
                        };
                        (
                            OpenBlock::RedactedThinking {
                                data: t.data,
                                extra: t.extra,
                            },
                            start,
                        )
                    }
                    Err(_) => opaque_open(content_block, &loc, warnings),
                }
            }
            Some("tool_use") => {
                match serde_json::from_value::<ToolUseBlock>(content_block.clone()) {
                    Ok(t) => {
                        // `input` arrives whole here (not incrementally), so
                        // a non-object value is wire data of the same shape
                        // as in the non-streaming response — warn now, at
                        // the event that carried it. Invalid JSON later
                        // *accumulated* from `input_json_delta` fragments
                        // stays silent by design (§ 4.5 preserves invalid
                        // JSON the model produced; truncation already shows
                        // in the stop reason).
                        if !t.input.is_object() {
                            warnings.push(pwarn(
                                WarningCode::MalformedToolCall,
                                &format!("{loc}/input"),
                                "tool_use.input is not a JSON object; unless \
                                 input_json_delta fragments replace it, the finalized \
                                 call keeps it verbatim in `arguments` and rebuilding \
                                 for Anthropic will fail",
                            ));
                        }
                        let mut rest = t.extra;
                        let cache = super::to_ir::split_cache_control(t.cache_control, &mut rest);
                        let start = ContentBlock::ToolCall {
                            id: Some(t.id.clone()),
                            name: t.name.clone(),
                            arguments: String::new(),
                            cache: cache.clone(),
                            extra: Extra::from_unknown(FMT, rest.clone()),
                        };
                        (
                            OpenBlock::ToolUse {
                                id: t.id,
                                name: t.name,
                                json: String::new(),
                                start_input: t.input,
                                extra: rest,
                                cache,
                            },
                            start,
                        )
                    }
                    Err(_) => opaque_open(content_block, &loc, warnings),
                }
            }
            _ => {
                let start = ContentBlock::opaque(FMT, content_block.clone());
                (
                    OpenBlock::Opaque {
                        value: content_block,
                        json: String::new(),
                    },
                    start,
                )
            }
        };
        self.blocks.insert(index, open);
        start
    }

    fn apply_delta(
        &mut self,
        index: usize,
        delta: &Value,
        raw: &str,
        warnings: &mut Vec<ConversionWarning>,
    ) -> Result<BlockDelta> {
        let Some(open) = self.blocks.get_mut(&index) else {
            return Err(perr(
                format!("content_block_delta for unknown block index {index}"),
                raw,
            ));
        };
        let kind = delta.get("type").and_then(Value::as_str).unwrap_or("");
        let mismatch = |kind: &str| {
            perr(
                format!("delta `{kind}` does not match the open block at index {index}"),
                raw,
            )
        };
        match (open, kind) {
            (OpenBlock::Text { text, .. }, "text_delta") => {
                let fragment = delta.get("text").and_then(Value::as_str).unwrap_or("");
                text.push_str(fragment);
                Ok(BlockDelta::Text(fragment.to_owned()))
            }
            (OpenBlock::Text { citations, .. }, "citations_delta") => {
                // Recognized but unmodeled: surfaced as Other and folded
                // into the finalized block's extra at content_block_stop.
                if let Some(c) = delta.get("citation") {
                    citations.push(c.clone());
                }
                Ok(BlockDelta::Other(delta.clone()))
            }
            (OpenBlock::Thinking { text, .. }, "thinking_delta") => {
                let fragment = delta.get("thinking").and_then(Value::as_str).unwrap_or("");
                text.push_str(fragment);
                Ok(BlockDelta::Thinking(fragment.to_owned()))
            }
            (OpenBlock::Thinking { signature, .. }, "signature_delta") => {
                let fragment = delta.get("signature").and_then(Value::as_str).unwrap_or("");
                signature.push_str(fragment);
                Ok(BlockDelta::Signature(fragment.to_owned()))
            }
            (OpenBlock::ToolUse { json, .. }, "input_json_delta") => {
                let fragment = delta
                    .get("partial_json")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                json.push_str(fragment);
                Ok(BlockDelta::ToolArguments(fragment.to_owned()))
            }
            (OpenBlock::Opaque { json, .. }, "input_json_delta") => {
                // Server-side tool use streams input fragments into an
                // unmodeled block; folded into its `input` at stop.
                let fragment = delta
                    .get("partial_json")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                json.push_str(fragment);
                Ok(BlockDelta::Other(delta.clone()))
            }
            (OpenBlock::Opaque { value, .. }, "compaction_delta") => {
                if value.is_object() {
                    if let Some(fragment) = delta.get("content").and_then(Value::as_str) {
                        let existing = value.get("content").and_then(Value::as_str).unwrap_or("");
                        value["content"] = Value::from(format!("{existing}{fragment}"));
                    }
                    if let Some(enc) = delta.get("encrypted_content") {
                        value["encrypted_content"] = enc.clone();
                    }
                }
                Ok(BlockDelta::Other(delta.clone()))
            }
            (
                _,
                "text_delta" | "thinking_delta" | "signature_delta" | "input_json_delta"
                | "citations_delta" | "compaction_delta",
            ) => Err(mismatch(kind)),
            (_, other) => {
                // Unknown delta kind addressed to a live block: surfaced as
                // Other; nothing known to fold at stop.
                warnings.push(pwarn(
                    WarningCode::MalformedField,
                    "/delta",
                    format!("unknown content_block_delta type `{other}`"),
                ));
                Ok(BlockDelta::Other(delta.clone()))
            }
        }
    }

    fn finalize_block(
        &mut self,
        index: usize,
        raw: &str,
        warnings: &mut Vec<ConversionWarning>,
    ) -> Result<ContentBlock> {
        let Some(open) = self.blocks.remove(&index) else {
            return Err(perr(
                format!("content_block_stop for unknown block index {index}"),
                raw,
            ));
        };
        Ok(match open {
            OpenBlock::Text {
                text,
                mut extra,
                cache,
                citations,
            } => {
                if !citations.is_empty() {
                    extra.insert("citations".to_owned(), Value::Array(citations));
                }
                ContentBlock::Text {
                    text,
                    cache,
                    extra: Extra::from_unknown(FMT, extra),
                }
            }
            OpenBlock::Thinking {
                text,
                signature,
                extra,
            } => ContentBlock::Thinking {
                text: Some(text),
                signature: if signature.is_empty() {
                    None
                } else {
                    Some(signature)
                },
                extra: Extra::from_unknown(FMT, extra),
            },
            OpenBlock::RedactedThinking { data, mut extra } => {
                extra.insert("redacted".to_owned(), Value::Bool(true));
                ContentBlock::Thinking {
                    text: None,
                    signature: Some(data),
                    extra: Extra::from_unknown(FMT, extra),
                }
            }
            OpenBlock::ToolUse {
                id,
                name,
                json,
                start_input,
                extra,
                cache,
            } => {
                // Deliberately no validation here: accumulated
                // `input_json_delta` fragments are the model's own output
                // and invalid JSON is preserved verbatim (§ 4.5); a
                // non-object `start_input` already warned at
                // `content_block_start`.
                let arguments = if json.is_empty() {
                    serde_json::to_string(&start_input).unwrap_or_else(|_| "{}".to_owned())
                } else {
                    json
                };
                ContentBlock::ToolCall {
                    id: Some(id),
                    name,
                    arguments,
                    cache,
                    extra: Extra::from_unknown(FMT, extra),
                }
            }
            OpenBlock::Opaque { mut value, json } => {
                if !json.is_empty() && value.is_object() {
                    match serde_json::from_str::<Value>(&json) {
                        Ok(input) => value["input"] = input,
                        Err(e) => warnings.push(pwarn(
                            WarningCode::MalformedField,
                            "/content_block_stop",
                            format!("accumulated partial_json is not valid JSON: {e}"),
                        )),
                    }
                }
                ContentBlock::opaque(FMT, value)
            }
        })
    }
}

/// Fallback for a known-type block that fails its shape: keep it opaque.
fn opaque_open(
    content_block: Value,
    loc: &str,
    warnings: &mut Vec<ConversionWarning>,
) -> (OpenBlock, ContentBlock) {
    warnings.push(pwarn(
        WarningCode::MalformedField,
        loc,
        "known block type failed to parse; kept as opaque",
    ));
    let start = ContentBlock::opaque(FMT, content_block.clone());
    (
        OpenBlock::Opaque {
            value: content_block,
            json: String::new(),
        },
        start,
    )
}

impl StreamParser for AnthropicStreamParser {
    fn parse(&mut self, event: &SseEvent) -> Result<(Vec<StreamEvent>, Vec<ConversionWarning>)> {
        // Post-terminal gate, before JSON decoding: `message_stop` completed
        // the protocol, so trailing data (a stray event, proxy keep-alive or
        // junk) cannot be attributed and must not turn an already-complete
        // stream into a fatal error (aligned with the Google parser).
        if self.terminated {
            let warnings = vec![pwarn(
                WarningCode::UnknownStreamEvent,
                "",
                "chunk received after the stream terminated",
            )];
            return Ok((vec![StreamEvent::Unknown], warnings));
        }
        let data: Value = serde_json::from_str(&event.data).map_err(|e| {
            perr(
                format!("SSE event data is not valid JSON: {e}"),
                &event.data,
            )
        })?;
        // The protocol names events both in the SSE `event:` field and in
        // the payload's `type`; either is accepted.
        let name = event
            .event
            .as_deref()
            .or_else(|| data.get("type").and_then(Value::as_str))
            .unwrap_or("");
        let mut warnings: Vec<ConversionWarning> = Vec::new();
        let events = match name {
            "ping" => Vec::new(),
            "message_start" => {
                let start: MessageStartEvent = serde_json::from_value(data)
                    .map_err(|e| perr(format!("invalid message_start: {e}"), &event.data))?;
                if let Some(usage) = &start.message.usage
                    && let Ok(Value::Object(map)) = serde_json::to_value(usage)
                {
                    self.usage = map;
                }
                let usage = self.merge_usage(None);
                vec![StreamEvent::MessageStart {
                    id: start.message.id,
                    model: start.message.model,
                    usage,
                }]
            }
            "content_block_start" => {
                let ev: ContentBlockStartEvent = serde_json::from_value(data)
                    .map_err(|e| perr(format!("invalid content_block_start: {e}"), &event.data))?;
                let block = self.start_block(ev.index, ev.content_block, &mut warnings);
                vec![StreamEvent::BlockStart {
                    index: ev.index,
                    block,
                }]
            }
            "content_block_delta" => {
                let ev: ContentBlockDeltaEvent = serde_json::from_value(data)
                    .map_err(|e| perr(format!("invalid content_block_delta: {e}"), &event.data))?;
                let delta = self.apply_delta(ev.index, &ev.delta, &event.data, &mut warnings)?;
                vec![StreamEvent::BlockDelta {
                    index: ev.index,
                    delta,
                }]
            }
            "content_block_stop" => {
                let ev: ContentBlockStopEvent = serde_json::from_value(data)
                    .map_err(|e| perr(format!("invalid content_block_stop: {e}"), &event.data))?;
                let block = self.finalize_block(ev.index, &event.data, &mut warnings)?;
                vec![StreamEvent::BlockStop {
                    index: ev.index,
                    block: Some(block),
                }]
            }
            "message_delta" => {
                let ev: MessageDeltaEvent = serde_json::from_value(data)
                    .map_err(|e| perr(format!("invalid message_delta: {e}"), &event.data))?;
                let usage = self.merge_usage(ev.usage.as_ref());
                let stop_reason = ev
                    .delta
                    .stop_reason
                    .as_deref()
                    .map(StopReason::from_str_lossy);
                vec![StreamEvent::MessageDelta { stop_reason, usage }]
            }
            "message_stop" => {
                self.terminated = true;
                vec![StreamEvent::MessageStop]
            }
            "error" => {
                // A mid-stream error envelope replaces normal flow; the
                // stream arrived on a 2xx response, so the HTTP status is
                // kept as 200 and the provider error type drives `kind`.
                let envelope: ErrorEnvelope = serde_json::from_value(data)
                    .map_err(|e| perr(format!("invalid error event: {e}"), &event.data))?;
                let kind = envelope
                    .error
                    .kind
                    .as_deref()
                    .map_or(ApiErrorKind::ServerError, map_error_kind);
                return Err(Error::Api {
                    status: 200,
                    kind,
                    message: envelope
                        .error
                        .message
                        .unwrap_or_else(|| "stream error".to_owned()),
                    raw: bytes::Bytes::copy_from_slice(event.data.as_bytes()),
                    truncated: false,
                    parsed: serde_json::from_str(&event.data).ok(),
                    retry_after: None,
                    headers: http::HeaderMap::new(),
                });
            }
            other => {
                warnings.push(pwarn(
                    WarningCode::UnknownStreamEvent,
                    "",
                    format!("unknown stream event `{other}`"),
                ));
                vec![StreamEvent::Unknown]
            }
        };
        Ok((events, warnings))
    }

    fn finish(&mut self) -> Result<(Vec<StreamEvent>, Vec<ConversionWarning>)> {
        if self.terminated {
            Ok((Vec::new(), Vec::new()))
        } else {
            Err(Error::Parse {
                message: "truncated stream: no message_stop before EOF".to_owned(),
                raw: bytes::Bytes::new(),
            })
        }
    }
}
