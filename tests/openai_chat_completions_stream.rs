//! Streaming tests for the `openai_chat_completions` format: complete SSE
//! sessions from fixtures, block-boundary inference
//! (`reasoning_content` ↔ `content` transitions, tool-call index
//! grouping), refusal deltas, `include_usage`, truncation, multi-choice
//! chunks and accumulation.
//!
//! Expected events are asserted through their IR JSON representation
//! (`StreamEvent` variants are `#[non_exhaustive]` and cannot be
//! constructed outside the crate).

use serde_json::{Value, json};

use llm_api::formats::openai_chat_completions::OpenAiChatCompletions;
use llm_api::http::{SseEvent, SseParser};
use llm_api::{
    Accumulator, ApiFormat, BlockDelta, ContentBlock, ConversionWarning, Error, StopReason,
    StreamEvent, StreamItem, WarningCode,
};

const F: &str = "openai_chat_completions";

fn fixture(name: &str) -> Vec<u8> {
    let path =
        format!("{}/tests/fixtures/openai_chat_completions/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

fn sse_events(bytes: &[u8]) -> Vec<SseEvent> {
    let mut parser = SseParser::new(usize::MAX);
    let mut events = parser.push(bytes).expect("SSE parses");
    events.extend(parser.finish().expect("SSE finish"));
    events
}

/// Runs a fixture through a fresh stream parser; event parsing must
/// succeed. Returns the unified events, warnings and the `finish` result.
fn run_stream(
    name: &str,
) -> (Vec<StreamEvent>, Vec<ConversionWarning>, llm_api::Result<Vec<StreamEvent>>) {
    let mut parser = OpenAiChatCompletions.stream_parser();
    let mut events = Vec::new();
    let mut warnings = Vec::new();
    for event in sse_events(&fixture(name)) {
        let (evs, ws) = parser.parse(&event).expect("stream event parses");
        events.extend(evs);
        warnings.extend(ws);
    }
    let finish = parser.finish().map(|(evs, ws)| {
        warnings.extend(ws);
        evs
    });
    (events, warnings, finish)
}

/// The IR JSON representation of a stream event.
fn ev(event: &StreamEvent) -> Value {
    serde_json::to_value(event).expect("event serializes")
}

fn accumulate(events: &[StreamEvent]) -> llm_api::Response {
    let mut acc = Accumulator::new();
    for event in events {
        acc.push(&StreamItem::new(event.clone())).expect("accumulates");
    }
    acc.finish().expect("accumulation finishes")
}

fn chunk_event(data: &Value) -> SseEvent {
    SseEvent::new(None, data.to_string())
}

#[test]
fn text_stream_with_include_usage_final_chunk() {
    let (events, warnings, finish) = run_stream("stream_text.sse");
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(finish.unwrap().is_empty());

    assert_eq!(events.len(), 8, "{events:#?}");
    assert_eq!(
        ev(&events[0]),
        json!({"type": "message_start", "id": "chatcmpl-s1", "model": "gpt-4o-mini"})
    );
    // The first chunk's empty content fragment opens the block without a
    // delta; envelope noise (obfuscation, null usage, logprobs) is silent.
    assert_eq!(
        ev(&events[1]),
        json!({"type": "block_start", "index": 0, "block": {"type": "text", "text": ""}})
    );
    assert_eq!(
        ev(&events[2]),
        json!({"type": "block_delta", "index": 0, "delta": {"type": "text", "value": "Hello"}})
    );
    assert_eq!(
        ev(&events[3]),
        json!({"type": "block_delta", "index": 0, "delta": {"type": "text", "value": " world"}})
    );
    // finish_reason closes the block, then reports the stop reason.
    assert_eq!(ev(&events[4]), json!({"type": "block_stop", "index": 0}));
    assert_eq!(
        ev(&events[5]),
        json!({"type": "message_delta", "stop_reason": "end_turn"})
    );
    // The include_usage final chunk (empty choices) carries the snapshot.
    let StreamEvent::MessageDelta { stop_reason, usage, .. } = &events[6] else {
        panic!("expected MessageDelta, got {:?}", events[6]);
    };
    assert!(stop_reason.is_none());
    let usage = usage.as_ref().unwrap();
    assert_eq!(usage.input_tokens, 9);
    assert_eq!(usage.output_tokens, 2);
    assert_eq!(usage.total_tokens, Some(11));
    assert_eq!(usage.cache_read_tokens, Some(4));
    assert_eq!(usage.reasoning_tokens, Some(0));
    assert_eq!(events[7], StreamEvent::MessageStop);

    let resp = accumulate(&events);
    assert_eq!(resp.id.as_deref(), Some("chatcmpl-s1"));
    assert_eq!(resp.text(), "Hello world");
    assert_eq!(resp.stop_reason, Some(StopReason::EndTurn));
    assert_eq!(resp.usage.as_ref().unwrap().input_tokens, 9);
    assert!(resp.raw.is_none());
}

#[test]
fn reasoning_content_transitions_open_new_blocks() {
    let (events, warnings, finish) = run_stream("stream_reasoning_switch.sse");
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(finish.is_ok());

    assert_eq!(events.len(), 18, "{events:#?}");
    assert_eq!(
        ev(&events[1]),
        json!({"type": "block_start", "index": 0, "block": {"type": "thinking"}})
    );
    assert_eq!(
        ev(&events[2]),
        json!({"type": "block_delta", "index": 0, "delta": {"type": "thinking", "value": "Compare the"}})
    );
    assert_eq!(
        ev(&events[3]),
        json!({"type": "block_delta", "index": 0, "delta": {"type": "thinking", "value": " decimals."}})
    );
    // Switch to content: the thinking block closes, a text block opens.
    assert_eq!(ev(&events[4]), json!({"type": "block_stop", "index": 0}));
    assert_eq!(
        ev(&events[5]),
        json!({"type": "block_start", "index": 1, "block": {"type": "text", "text": ""}})
    );
    assert_eq!(
        ev(&events[6]),
        json!({"type": "block_delta", "index": 1, "delta": {"type": "text", "value": "9.8 is"}})
    );
    assert_eq!(
        ev(&events[7]),
        json!({"type": "block_delta", "index": 1, "delta": {"type": "text", "value": " larger."}})
    );
    // Switch back to reasoning: a NEW thinking block (index 2).
    assert_eq!(ev(&events[8]), json!({"type": "block_stop", "index": 1}));
    assert_eq!(
        ev(&events[9]),
        json!({"type": "block_start", "index": 2, "block": {"type": "thinking"}})
    );
    assert_eq!(
        ev(&events[10]),
        json!({"type": "block_delta", "index": 2, "delta": {"type": "thinking", "value": "Double-check."}})
    );
    // And forward to content again (index 3).
    assert_eq!(ev(&events[11]), json!({"type": "block_stop", "index": 2}));
    assert_eq!(
        ev(&events[12]),
        json!({"type": "block_start", "index": 3, "block": {"type": "text", "text": ""}})
    );

    let resp = accumulate(&events);
    // Interleaved thinking→text→thinking→text survives verbatim (§ 9).
    assert_eq!(resp.message.content.len(), 4);
    assert!(matches!(&resp.message.content[0], ContentBlock::Thinking { text: Some(t), .. }
        if t == "Compare the decimals."));
    assert!(matches!(&resp.message.content[1], ContentBlock::Text { text, .. } if text == "9.8 is larger."));
    assert!(matches!(&resp.message.content[2], ContentBlock::Thinking { text: Some(t), .. }
        if t == "Double-check."));
    assert!(matches!(&resp.message.content[3], ContentBlock::Text { text, .. } if text == " Confirmed."));
    assert_eq!(resp.stop_reason, Some(StopReason::EndTurn));
    assert_eq!(resp.usage.as_ref().unwrap().reasoning_tokens, Some(18));
}

#[test]
fn interleaved_tool_call_fragments_group_by_index() {
    let (events, warnings, finish) = run_stream("stream_tool_calls.sse");
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(finish.is_ok());

    assert_eq!(events.len(), 10, "{events:#?}");
    assert_eq!(
        ev(&events[1]),
        json!({
            "type": "block_start", "index": 0,
            "block": {"type": "tool_call", "id": "call_a", "name": "get_weather", "arguments": ""},
        })
    );
    assert_eq!(
        ev(&events[2]),
        json!({"type": "block_delta", "index": 0, "delta": {"type": "tool_arguments", "value": "{\"city\":"}})
    );
    // The third chunk interleaves a new call (index 1) with an argument
    // fragment for index 0.
    assert_eq!(
        ev(&events[3]),
        json!({
            "type": "block_start", "index": 1,
            "block": {"type": "tool_call", "id": "call_b", "name": "get_time", "arguments": ""},
        })
    );
    assert_eq!(
        ev(&events[4]),
        json!({"type": "block_delta", "index": 0, "delta": {"type": "tool_arguments", "value": "\"Paris\"}"}})
    );
    assert_eq!(
        ev(&events[5]),
        json!({"type": "block_delta", "index": 1, "delta": {"type": "tool_arguments", "value": "{\"tz\":\"CET\"}"}})
    );
    // finish_reason finalizes both blocks with the accumulated state.
    assert_eq!(
        ev(&events[6]),
        json!({
            "type": "block_stop", "index": 0,
            "block": {"type": "tool_call", "id": "call_a", "name": "get_weather",
                      "arguments": "{\"city\":\"Paris\"}"},
        })
    );
    assert_eq!(
        ev(&events[7]),
        json!({
            "type": "block_stop", "index": 1,
            "block": {"type": "tool_call", "id": "call_b", "name": "get_time",
                      "arguments": "{\"tz\":\"CET\"}"},
        })
    );
    // The finish chunk carried usage too — one MessageDelta with both.
    let StreamEvent::MessageDelta { stop_reason, usage, .. } = &events[8] else {
        panic!("expected MessageDelta, got {:?}", events[8]);
    };
    assert_eq!(*stop_reason, Some(StopReason::ToolUse));
    assert_eq!(usage.as_ref().unwrap().total_tokens, Some(61));
    assert_eq!(events[9], StreamEvent::MessageStop);

    let resp = accumulate(&events);
    assert_eq!(resp.stop_reason, Some(StopReason::ToolUse));
    assert_eq!(resp.message.content.len(), 2);
    assert!(matches!(&resp.message.content[0], ContentBlock::ToolCall { arguments, .. }
        if arguments == "{\"city\":\"Paris\"}"));
    assert!(matches!(&resp.message.content[1], ContentBlock::ToolCall { id: Some(id), .. }
        if id == "call_b"));
}

#[test]
fn refusal_stream_accumulates_to_refusal_stop() {
    let (events, warnings, finish) = run_stream("stream_refusal.sse");
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(finish.is_ok());

    // The refusal channel opens a refusal-marked text block.
    let StreamEvent::BlockStart { index: 0, block, .. } = &events[1] else {
        panic!("expected BlockStart, got {:?}", events[1]);
    };
    assert!(matches!(block, ContentBlock::Text { .. }));
    assert_eq!(block.extra().unwrap().get(F).unwrap().get("refusal"), Some(&json!(true)));
    assert_eq!(
        ev(&events[2]),
        json!({"type": "block_delta", "index": 0, "delta": {"type": "text", "value": "I cannot"}})
    );

    let resp = accumulate(&events);
    // The finish chunk said "stop"; the refusal-marked block normalizes it.
    assert_eq!(resp.stop_reason, Some(StopReason::Refusal));
    assert_eq!(resp.text(), "I cannot help with that.");
}

#[test]
fn truncated_stream_fails_at_finish() {
    let (events, warnings, finish) = run_stream("stream_truncated.sse");
    assert!(warnings.is_empty());
    match finish {
        Err(Error::Parse { message, .. }) => assert!(message.contains("truncated")),
        other => panic!("expected truncation parse error, got {other:?}"),
    }
    // Accumulation of the partial record also refuses to finish.
    let mut acc = Accumulator::new();
    for event in &events {
        acc.push(&StreamItem::new(event.clone())).unwrap();
    }
    assert!(matches!(acc.finish(), Err(Error::Parse { .. })));
}

#[test]
fn multi_choice_chunks_surface_as_unknown_once_warned() {
    let (events, warnings, finish) = run_stream("stream_multi_choice.sse");
    assert!(finish.is_ok());

    // Three chunks carry choice index 1 → three Unknown events, one
    // warning for the whole stream.
    let unknown = events.iter().filter(|e| matches!(e, StreamEvent::Unknown)).count();
    assert_eq!(unknown, 3, "{events:#?}");
    let multi: Vec<_> =
        warnings.iter().filter(|w| w.code == WarningCode::MultipleCandidates).collect();
    assert_eq!(multi.len(), 1, "{warnings:?}");

    // Choice 0 still parses fully.
    let resp = accumulate(&events);
    assert_eq!(resp.text(), "First answer");
    assert_eq!(resp.stop_reason, Some(StopReason::EndTurn));
}

#[test]
fn malformed_chunk_data_is_unknown_with_warning() {
    let mut parser = OpenAiChatCompletions.stream_parser();
    let (events, warnings) = parser.parse(&SseEvent::new(None, "{not json")).unwrap();
    assert_eq!(events, vec![StreamEvent::Unknown]);
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0].code, WarningCode::MalformedField);
}

#[test]
fn done_only_stream_synthesizes_message_start() {
    let mut parser = OpenAiChatCompletions.stream_parser();
    let (events, warnings) = parser.parse(&SseEvent::new(None, "[DONE]")).unwrap();
    assert!(warnings.is_empty());
    assert_eq!(events.len(), 2);
    assert!(matches!(events[0], StreamEvent::MessageStart { .. }));
    assert_eq!(events[1], StreamEvent::MessageStop);
    assert!(parser.finish().is_ok());
}

#[test]
fn done_without_finish_reason_closes_open_blocks() {
    // A dialect stream that never sends finish_reason: [DONE] still closes
    // the open block so accumulation succeeds.
    let mut parser = OpenAiChatCompletions.stream_parser();
    let mut events = Vec::new();
    let chunk = json!({
        "id": "c", "model": "m",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "partial"}, "finish_reason": null}],
    });
    let (evs, _) = parser.parse(&chunk_event(&chunk)).unwrap();
    events.extend(evs);
    let (evs, _) = parser.parse(&SseEvent::new(None, "[DONE]")).unwrap();
    events.extend(evs);
    assert!(parser.finish().is_ok());
    assert_eq!(
        ev(&events[3]),
        json!({"type": "block_stop", "index": 0})
    );
    let resp = accumulate(&events);
    assert_eq!(resp.text(), "partial");
    assert!(resp.stop_reason.is_none());
}

#[test]
fn tool_call_fragment_without_id_or_name_warns() {
    let mut parser = OpenAiChatCompletions.stream_parser();
    let chunk = json!({
        "id": "c", "model": "m",
        "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "{\"x\":1}"}},
        ]}, "finish_reason": null}],
    });
    let (events, warnings) = parser.parse(&chunk_event(&chunk)).unwrap();
    assert!(warnings.iter().any(|w| w.code == WarningCode::MalformedField));
    // The block still opens (empty name) and the arguments still stream.
    assert_eq!(
        ev(&events[1]),
        json!({"type": "block_start", "index": 0,
               "block": {"type": "tool_call", "name": "", "arguments": ""}})
    );
    assert_eq!(
        ev(&events[2]),
        json!({"type": "block_delta", "index": 0, "delta": {"type": "tool_arguments", "value": "{\"x\":1}"}})
    );
}

#[test]
fn late_name_fragments_fold_into_finalized_block() {
    // Dialects may split the function name across fragments; the
    // finalized block carries the concatenation.
    let mut parser = OpenAiChatCompletions.stream_parser();
    let chunks = [
        json!({"id": "c", "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "call_x", "type": "function", "function": {"name": "get_", "arguments": ""}},
        ]}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"name": "weather", "arguments": "{}"}},
        ]}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
    ];
    let mut events = Vec::new();
    for chunk in &chunks {
        let (evs, ws) = parser.parse(&chunk_event(chunk)).unwrap();
        assert!(ws.is_empty(), "{ws:?}");
        events.extend(evs);
    }
    let stop = events
        .iter()
        .find_map(|e| match e {
            StreamEvent::BlockStop { block: Some(b), .. } => Some(b.clone()),
            _ => None,
        })
        .expect("finalized block");
    assert!(matches!(&stop, ContentBlock::ToolCall { name, arguments, .. }
        if name == "get_weather" && arguments == "{}"));
}

#[test]
fn unknown_delta_fields_attach_to_open_block_or_unknown() {
    let mut parser = OpenAiChatCompletions.stream_parser();
    // No open block: the unknown field surfaces as Unknown + one warning.
    let chunk = json!({"id": "c", "choices": [{"index": 0, "delta": {"function_call": {"name": "legacy"}},
        "finish_reason": null}]});
    let (events, warnings) = parser.parse(&chunk_event(&chunk)).unwrap();
    assert!(events.contains(&StreamEvent::Unknown));
    assert!(warnings.iter().any(|w| w.code == WarningCode::UnknownStreamEvent));
    // Repeats of the same field do not warn again.
    let (_, warnings) = parser.parse(&chunk_event(&chunk)).unwrap();
    assert!(warnings.is_empty());

    // With an open content block the field rides it as an Other delta.
    let mut parser = OpenAiChatCompletions.stream_parser();
    let open = json!({"id": "c", "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": null}]});
    parser.parse(&chunk_event(&open)).unwrap();
    let annotated = json!({"id": "c", "choices": [{"index": 0,
        "delta": {"content": "!", "annotations": [{"type": "url_citation"}]}, "finish_reason": null}]});
    let (events, warnings) = parser.parse(&chunk_event(&annotated)).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    let StreamEvent::BlockDelta { index: 0, delta: BlockDelta::Other(payload), .. } = &events[1]
    else {
        panic!("expected Other delta, got {events:#?}");
    };
    assert_eq!(payload["annotations"][0]["type"], json!("url_citation"));
}

#[test]
fn custom_tool_call_type_survives_streaming() {
    // Undocumented for deltas, but dialects may stream custom calls; the
    // reserved `type` key keeps the kind for re-serialization.
    let mut parser = OpenAiChatCompletions.stream_parser();
    let chunks = [
        json!({"id": "c", "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "call_c", "type": "custom", "custom": {"name": "run_sql"}},
        ]}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
    ];
    let mut events = Vec::new();
    for chunk in &chunks {
        let (evs, _) = parser.parse(&chunk_event(chunk)).unwrap();
        events.extend(evs);
    }
    let stop = events
        .iter()
        .find_map(|e| match e {
            StreamEvent::BlockStop { block: Some(b), .. } => Some(b.clone()),
            _ => None,
        })
        .expect("finalized block");
    let ContentBlock::ToolCall { id, extra, .. } = &stop else { panic!("expected tool call") };
    assert_eq!(id.as_deref(), Some("call_c"));
    let ns = extra.get(F).unwrap();
    assert_eq!(ns.get("type"), Some(&json!("custom")));
    assert_eq!(ns.get("custom"), Some(&json!({"name": "run_sql"})));
}

#[test]
fn content_after_finish_opens_fresh_block() {
    // Defensive: a buggy dialect sending content after finish_reason gets
    // a fresh block instead of a panic or a delta on a closed block.
    let mut parser = OpenAiChatCompletions.stream_parser();
    let mut events = Vec::new();
    for chunk in [
        json!({"id": "c", "choices": [{"index": 0, "delta": {"content": "a"}, "finish_reason": "stop"}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {"content": "b"}, "finish_reason": null}]}),
    ] {
        let (evs, _) = parser.parse(&chunk_event(&chunk)).unwrap();
        events.extend(evs);
    }
    let (evs, _) = parser.parse(&SseEvent::new(None, "[DONE]")).unwrap();
    events.extend(evs);
    let resp = accumulate(&events);
    assert_eq!(resp.message.content.len(), 2);
    assert_eq!(resp.text(), "ab");
}
