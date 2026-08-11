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

use llm_api::formats::openai_chat_completions::{
    OpenAiChatCompletions, request_from_ir, request_to_ir, text_block_reserved_key,
};
use llm_api::http::{SseEvent, SseParser};
use llm_api::{
    Accumulator, ApiFormat, BlockDelta, CallMode, ContentBlock, ConversionWarning, ConvertOptions,
    Error, Extra, Message, OpenAiChatCompletionsOptions, Request, ResponseMeta, StopReason,
    StreamEvent, StreamItem, WarningCode, WarningSeverity,
};

const F: &str = "openai_chat_completions";

fn fixture(name: &str) -> Vec<u8> {
    let path = format!(
        "{}/tests/fixtures/openai_chat_completions/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
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
) -> (
    Vec<StreamEvent>,
    Vec<ConversionWarning>,
    llm_api::Result<Vec<StreamEvent>>,
) {
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
        acc.push(&StreamItem::new(event.clone()))
            .expect("accumulates");
    }
    acc.finish().expect("accumulation finishes")
}

fn chunk_event(data: &Value) -> SseEvent {
    SseEvent::new(None, data.to_string())
}

/// Replays an assistant message as a unary CC request body.
fn replay_message(msg: &Message) -> (Value, Vec<ConversionWarning>) {
    let req = Request::with_messages(vec![msg.clone()]);
    request_from_ir(
        &req,
        None,
        CallMode::Unary,
        &ConvertOptions::default(),
        &OpenAiChatCompletionsOptions::default(),
    )
    .unwrap()
}

/// Runs the chunks plus `[DONE]` through a fresh parser; every parse call
/// must succeed and stay warning-free. Returns the unified events.
fn run_chunks_no_warnings(chunks: &[Value]) -> Vec<StreamEvent> {
    let mut parser = OpenAiChatCompletions.stream_parser();
    let mut events = Vec::new();
    for chunk in chunks {
        let (evs, ws) = parser.parse(&chunk_event(chunk)).unwrap();
        assert!(ws.is_empty(), "{ws:?}");
        events.extend(evs);
    }
    let (evs, ws) = parser.parse(&SseEvent::new(None, "[DONE]")).unwrap();
    assert!(ws.is_empty(), "{ws:?}");
    events.extend(evs);
    events
}

/// The first finalized block carried by a `BlockStop` event.
fn finalized_block(events: &[StreamEvent]) -> ContentBlock {
    events
        .iter()
        .find_map(|e| match e {
            StreamEvent::BlockStop { block: Some(b), .. } => Some(b.clone()),
            _ => None,
        })
        .expect("finalized block")
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
    let StreamEvent::MessageDelta {
        stop_reason, usage, ..
    } = &events[6]
    else {
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
    assert!(
        matches!(&resp.message.content[0], ContentBlock::Thinking { text: Some(t), .. }
        if t == "Compare the decimals.")
    );
    assert!(
        matches!(&resp.message.content[1], ContentBlock::Text { text, .. } if text == "9.8 is larger.")
    );
    assert!(
        matches!(&resp.message.content[2], ContentBlock::Thinking { text: Some(t), .. }
        if t == "Double-check.")
    );
    assert!(
        matches!(&resp.message.content[3], ContentBlock::Text { text, .. } if text == " Confirmed.")
    );
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
    let StreamEvent::MessageDelta {
        stop_reason, usage, ..
    } = &events[8]
    else {
        panic!("expected MessageDelta, got {:?}", events[8]);
    };
    assert_eq!(*stop_reason, Some(StopReason::ToolUse));
    assert_eq!(usage.as_ref().unwrap().total_tokens, Some(61));
    assert_eq!(events[9], StreamEvent::MessageStop);

    let resp = accumulate(&events);
    assert_eq!(resp.stop_reason, Some(StopReason::ToolUse));
    assert_eq!(resp.message.content.len(), 2);
    assert!(
        matches!(&resp.message.content[0], ContentBlock::ToolCall { arguments, .. }
        if arguments == "{\"city\":\"Paris\"}")
    );
    assert!(
        matches!(&resp.message.content[1], ContentBlock::ToolCall { id: Some(id), .. }
        if id == "call_b")
    );
}

#[test]
fn refusal_stream_accumulates_to_refusal_stop() {
    let (events, warnings, finish) = run_stream("stream_refusal.sse");
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(finish.is_ok());

    // The refusal channel opens a refusal-marked text block.
    let StreamEvent::BlockStart {
        index: 0, block, ..
    } = &events[1]
    else {
        panic!("expected BlockStart, got {:?}", events[1]);
    };
    assert!(matches!(block, ContentBlock::Text { .. }));
    assert_eq!(
        block
            .extra()
            .unwrap()
            .get(F)
            .unwrap()
            .get(text_block_reserved_key::REFUSAL),
        Some(&json!(true))
    );
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
        Err(Error::TruncatedStream { message, .. }) => assert!(message.contains("[DONE]")),
        other => panic!("expected a truncated-stream error, got {other:?}"),
    }
    // Accumulation of the partial record also refuses to finish.
    let mut acc = Accumulator::new();
    for event in &events {
        acc.push(&StreamItem::new(event.clone())).unwrap();
    }
    assert!(matches!(acc.finish(), Err(Error::TruncatedStream { .. })));
}

#[test]
fn multi_choice_chunks_surface_as_unknown_once_warned() {
    let (events, warnings, finish) = run_stream("stream_multi_choice.sse");
    assert!(finish.is_ok());

    // Three chunks carry choice index 1 → three Unknown events, one
    // warning for the whole stream.
    let unknown = events
        .iter()
        .filter(|e| matches!(e, StreamEvent::Unknown))
        .count();
    assert_eq!(unknown, 3, "{events:#?}");
    let multi: Vec<_> = warnings
        .iter()
        .filter(|w| w.code == WarningCode::MultipleCandidates)
        .collect();
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
    assert_eq!(ev(&events[3]), json!({"type": "block_stop", "index": 0}));
    let resp = accumulate(&events);
    assert_eq!(resp.text(), "partial");
    assert!(resp.stop_reason.is_none());
}

#[test]
fn tool_call_without_id_or_name_warns_at_finalize() {
    // An opening fragment may legitimately carry only arguments (id/name
    // can arrive in later fragments), so the missing-field verdict lands
    // at finalization, per field, matching the non-streaming entry parse.
    let mut parser = OpenAiChatCompletions.stream_parser();
    let chunk = json!({
        "id": "c", "model": "m",
        "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "{\"x\":1}"}},
        ]}, "finish_reason": null}],
    });
    let (events, warnings) = parser.parse(&chunk_event(&chunk)).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
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

    let finish =
        json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]});
    let (_, warnings) = parser.parse(&chunk_event(&finish)).unwrap();
    assert_eq!(warnings.len(), 2, "{warnings:?}");
    assert!(
        warnings
            .iter()
            .all(|w| w.code == WarningCode::MalformedToolCall)
    );
    assert!(warnings[0].message.contains("no `id`"), "{warnings:?}");
    assert!(
        warnings[1].message.contains("`function.name` is missing"),
        "{warnings:?}"
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
    assert!(
        matches!(&stop, ContentBlock::ToolCall { name, arguments, .. }
        if name == "get_weather" && arguments == "{}")
    );
}

#[test]
fn unknown_delta_fields_attach_to_open_block_or_unknown() {
    let mut parser = OpenAiChatCompletions.stream_parser();
    // No open block: the unknown field surfaces as Unknown + one warning.
    let chunk = json!({"id": "c", "choices": [{"index": 0, "delta": {"function_call": {"name": "legacy"}},
        "finish_reason": null}]});
    let (events, warnings) = parser.parse(&chunk_event(&chunk)).unwrap();
    assert!(events.contains(&StreamEvent::Unknown));
    assert!(
        warnings
            .iter()
            .any(|w| w.code == WarningCode::UnknownStreamEvent)
    );
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
    let StreamEvent::BlockDelta {
        index: 0,
        delta: BlockDelta::Other(payload),
        ..
    } = &events[1]
    else {
        panic!("expected Other delta, got {events:#?}");
    };
    assert_eq!(payload["annotations"][0]["type"], json!("url_citation"));
}

#[test]
fn unknown_delta_fields_fold_into_accumulated_block() {
    // The Other deltas are real-time surfacing; the same fields also fold
    // into the finalized block at close (§ 9), so accumulation keeps
    // them: arrays append per item and strings concatenate across chunks.
    let mut parser = OpenAiChatCompletions.stream_parser();
    let chunks = [
        json!({"id": "c", "model": "m", "choices": [{"index": 0,
            "delta": {"role": "assistant", "content": "hi",
                      "annotations": [{"type": "url_citation", "url_citation": {"url": "https://a"}}],
                      "x_note": "he"},
            "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0,
            "delta": {"content": "!",
                      "annotations": [{"type": "url_citation", "url_citation": {"url": "https://b"}}],
                      "x_note": "llo"},
            "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
    ];
    let mut events = Vec::new();
    for chunk in &chunks {
        let (evs, ws) = parser.parse(&chunk_event(chunk)).unwrap();
        assert!(ws.is_empty(), "{ws:?}");
        events.extend(evs);
    }
    let (evs, ws) = parser.parse(&SseEvent::new(None, "[DONE]")).unwrap();
    assert!(ws.is_empty(), "{ws:?}");
    events.extend(evs);

    // Real-time: one Other delta per carrying chunk, raw (unmerged).
    let others = events
        .iter()
        .filter(|e| {
            matches!(
                e,
                StreamEvent::BlockDelta {
                    delta: BlockDelta::Other(_),
                    ..
                }
            )
        })
        .count();
    assert_eq!(others, 2);

    // The stop hands the accumulator a finalized block with the merge.
    let stop = events
        .iter()
        .find_map(|e| match e {
            StreamEvent::BlockStop { block: Some(b), .. } => Some(b.clone()),
            _ => None,
        })
        .expect("finalized text block");
    let resp = accumulate(&events);
    assert_eq!(resp.message.content.len(), 1);
    assert_eq!(resp.message.content[0], stop);
    let ContentBlock::Text { text, extra, .. } = &resp.message.content[0] else {
        panic!("expected text block: {:?}", resp.message.content);
    };
    assert_eq!(text, "hi!");
    // Message-level by origin (the delta object is message-shaped), so
    // the fold nests under the `__llm_api_message` reserved key.
    assert_eq!(
        Value::Object(extra.get(F).unwrap().clone()),
        json!({text_block_reserved_key::MESSAGE: {
            "annotations": [
                {"type": "url_citation", "url_citation": {"url": "https://a"}},
                {"type": "url_citation", "url_citation": {"url": "https://b"}},
            ],
            "x_note": "hello",
        }})
    );
    assert!(resp.warnings.is_empty(), "{:?}", resp.warnings);
}

#[test]
fn legacy_function_call_delta_folds_verbatim() {
    // The deprecated `function_call` delta is unmodeled; its object
    // fragments merge recursively under the `message` reserved key, so
    // split `arguments` increments reassemble and the field replays on
    // the message level — not inside the content part.
    let events = run_chunks_no_warnings(&[
        json!({"id": "c", "choices": [{"index": 0,
            "delta": {"role": "assistant", "content": "",
                      "function_call": {"name": "legacy", "arguments": "{\"a\""}},
            "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0,
            "delta": {"function_call": {"arguments": ":1}"}},
            "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
    ]);
    let resp = accumulate(&events);
    let ContentBlock::Text { text, extra, .. } = &resp.message.content[0] else {
        panic!("expected text block: {:?}", resp.message.content);
    };
    assert_eq!(text, "");
    let call = json!({"name": "legacy", "arguments": "{\"a\":1}"});
    assert_eq!(
        extra.get(F).unwrap().get(text_block_reserved_key::MESSAGE),
        Some(&json!({"function_call": call}))
    );

    let (body, ws) = replay_message(&resp.message);
    assert!(ws.is_empty(), "{ws:?}");
    let msg = &body["messages"][0];
    assert_eq!(msg["function_call"], call);
    assert_eq!(msg["content"], json!([{"type": "text", "text": ""}]));
}

#[test]
fn folded_fields_replay_matches_non_streaming_parse() {
    // Parity target: replaying the accumulated streamed message equals
    // replaying the non-streaming parse of the same terminal message —
    // the folded fields land on the message level and the `content`
    // string shorthand is preserved (the `message` reserved key holds no
    // part-level data).
    let annotations = json!([{"type": "url_citation", "url_citation": {"url": "https://a"}}]);
    let events = run_chunks_no_warnings(&[
        json!({"id": "c", "choices": [{"index": 0,
            "delta": {"role": "assistant", "content": "hi", "annotations": annotations},
            "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
    ]);
    let streamed = accumulate(&events);
    let (streamed_body, ws) = replay_message(&streamed.message);
    assert!(ws.is_empty(), "{ws:?}");

    let body = json!({"id": "c", "object": "chat.completion", "model": "m",
        "choices": [{"index": 0,
            "message": {"role": "assistant", "content": "hi", "annotations": annotations},
            "finish_reason": "stop"}]});
    let parsed = OpenAiChatCompletions
        .parse_response(
            &serde_json::to_vec(&body).unwrap(),
            &ResponseMeta::new(200, http::HeaderMap::new()),
        )
        .unwrap();
    assert!(parsed.warnings.is_empty(), "{:?}", parsed.warnings);
    let (unary_body, ws) = replay_message(&parsed.message);
    assert!(ws.is_empty(), "{ws:?}");

    assert_eq!(streamed_body, unary_body);
    let msg = &streamed_body["messages"][0];
    assert_eq!(msg["content"], json!("hi"));
    assert_eq!(msg["annotations"], annotations);
}

#[test]
fn folded_replay_round_trips_through_request_parse() {
    // Closing the parity loop: the replayed wire parses back with the
    // folded fields on `Message.extra` (their non-streaming home, blocks
    // clean), and a second replay reproduces the same body.
    let events = run_chunks_no_warnings(&[
        json!({"id": "c", "choices": [{"index": 0,
            "delta": {"role": "assistant", "content": "hi", "x_note": "n"},
            "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
    ]);
    let resp = accumulate(&events);
    let (body, ws) = replay_message(&resp.message);
    assert!(ws.is_empty(), "{ws:?}");
    assert_eq!(body["messages"][0]["x_note"], json!("n"));

    let (req, ws) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert!(ws.is_empty(), "{ws:?}");
    let msg = &req.messages[0];
    assert_eq!(msg.extra.get(F).unwrap().get("x_note"), Some(&json!("n")));
    assert!(
        msg.content
            .iter()
            .all(|b| matches!(b, ContentBlock::Text { extra, .. } if extra.is_empty())),
        "{:?}",
        msg.content
    );

    let (body2, ws) = request_from_ir(
        &req,
        None,
        CallMode::Unary,
        &ConvertOptions::default(),
        &OpenAiChatCompletionsOptions::default(),
    )
    .unwrap();
    assert!(ws.is_empty(), "{ws:?}");
    assert_eq!(body2["messages"], body["messages"]);
}

#[test]
fn multiple_blocks_merge_message_fields_in_block_order() {
    // Several text blocks holding the `message` reserved key merge into
    // the message in block order: later keys win (RFC 7396), disjoint
    // keys accumulate.
    let events = run_chunks_no_warnings(&[
        json!({"id": "c", "choices": [{"index": 0,
            "delta": {"role": "assistant", "content": "a",
                      "x_note": "first", "only_first": "f"},
            "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0,
            "delta": {"refusal": "no"}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0,
            "delta": {"content": "b", "x_note": "second"}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
    ]);
    let resp = accumulate(&events);
    assert_eq!(resp.message.content.len(), 3, "{:#?}", resp.message.content);

    let (body, ws) = replay_message(&resp.message);
    assert!(ws.is_empty(), "{ws:?}");
    let msg = &body["messages"][0];
    assert_eq!(msg["x_note"], json!("second"));
    assert_eq!(msg["only_first"], json!("f"));
    assert_eq!(
        msg["content"],
        json!([
            {"type": "text", "text": "a"},
            {"type": "refusal", "refusal": "no"},
            {"type": "text", "text": "b"},
        ])
    );
}

#[test]
fn message_extra_stays_authoritative_over_folded_fields() {
    // The user's message-level extra merges after the hoisted fields and
    // wins on conflicts (§ 6).
    let events = run_chunks_no_warnings(&[
        json!({"id": "c", "choices": [{"index": 0,
            "delta": {"role": "assistant", "content": "hi", "x_note": "stream"},
            "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
    ]);
    let mut message = accumulate(&events).message;
    let user_ns = json!({"x_note": "user"});
    message.extra = Extra::from_unknown(F, user_ns.as_object().unwrap().clone());

    let (body, ws) = replay_message(&message);
    assert!(ws.is_empty(), "{ws:?}");
    assert_eq!(body["messages"][0]["x_note"], json!("user"));
}

#[test]
fn refusal_channel_folded_fields_replay_on_message_level() {
    // Fields folded while the refusal channel is open hoist to the
    // message level too; the refusal part stays clean.
    let events = run_chunks_no_warnings(&[
        json!({"id": "c", "choices": [{"index": 0,
            "delta": {"role": "assistant", "refusal": "no", "x_note": "n"},
            "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
    ]);
    let resp = accumulate(&events);
    let (body, ws) = replay_message(&resp.message);
    assert!(ws.is_empty(), "{ws:?}");
    let msg = &body["messages"][0];
    assert_eq!(
        msg["content"],
        json!([{"type": "refusal", "refusal": "no"}])
    );
    assert_eq!(msg["x_note"], json!("n"));
}

#[test]
fn reasoning_channel_folded_fields_replay_on_message_level() {
    // Fields folded while the reasoning channel is open stay top-level on
    // the Thinking block extra, which already merges into the containing
    // message on serialization.
    let events = run_chunks_no_warnings(&[
        json!({"id": "c", "choices": [{"index": 0,
            "delta": {"role": "assistant", "reasoning_content": "think", "x_note": "n"},
            "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
    ]);
    let resp = accumulate(&events);
    let ContentBlock::Thinking { extra, .. } = &resp.message.content[0] else {
        panic!("expected thinking block: {:?}", resp.message.content);
    };
    assert_eq!(extra.get(F).unwrap().get("x_note"), Some(&json!("n")));

    let (body, ws) = replay_message(&resp.message);
    assert!(ws.is_empty(), "{ws:?}");
    let msg = &body["messages"][0];
    assert_eq!(msg["reasoning_content"], json!("think"));
    assert_eq!(msg["x_note"], json!("n"));
}

#[test]
fn text_after_tool_call_merges_and_warns_block_order_lost() {
    // The wire holds one `content` string per message: a text fragment
    // arriving after a tool call opened merges into its earlier block —
    // matching the non-streaming parse of the assembled message — and
    // the fold is disclosed once per stream.
    let mut parser = OpenAiChatCompletions.stream_parser();
    let chunks = [
        json!({"id": "c", "choices": [{"index": 0,
            "delta": {"role": "assistant", "content": "a"}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "type": "function",
             "function": {"name": "f", "arguments": "{}"}},
        ]}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0,
            "delta": {"content": "b"}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0,
            "delta": {"content": "c"}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
    ];
    let mut events = Vec::new();
    let mut warnings = Vec::new();
    for chunk in &chunks {
        let (evs, ws) = parser.parse(&chunk_event(chunk)).unwrap();
        events.extend(evs);
        warnings.extend(ws);
    }
    let (evs, _) = parser.parse(&SseEvent::new(None, "[DONE]")).unwrap();
    events.extend(evs);

    let order: Vec<&ConversionWarning> = warnings
        .iter()
        .filter(|w| w.code == WarningCode::BlockOrderLost)
        .collect();
    assert_eq!(order.len(), 1, "once per stream: {warnings:?}");
    assert_eq!(order[0].location, "/choices/0/delta/content");
    assert_eq!(order[0].severity, WarningSeverity::Semantic);

    // No third block: the later text reuses block 0.
    let starts: Vec<usize> = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::BlockStart { index, .. } => Some(*index),
            _ => None,
        })
        .collect();
    assert_eq!(starts, vec![0, 1]);
    let resp = accumulate(&events);
    assert_eq!(resp.message.content.len(), 2);
    assert!(matches!(&resp.message.content[0],
        ContentBlock::Text { text, .. } if text == "abc"));
    assert!(matches!(&resp.message.content[1],
        ContentBlock::ToolCall { name, .. } if name == "f"));
}

#[test]
fn tool_then_text_arrival_order_survives_without_order_warning() {
    // Tool call first, text later: the text opens a new, later block, so
    // arrival order is kept in the accumulated message and nothing folds
    // — no BlockOrderLost here (replaying that order to CC warns on the
    // build side instead).
    let mut parser = OpenAiChatCompletions.stream_parser();
    let chunks = [
        json!({"id": "c", "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [
            {"index": 0, "id": "call_1", "type": "function",
             "function": {"name": "f", "arguments": "{}"}},
        ]}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0,
            "delta": {"content": "x"}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
    ];
    let mut events = Vec::new();
    let mut warnings = Vec::new();
    for chunk in &chunks {
        let (evs, ws) = parser.parse(&chunk_event(chunk)).unwrap();
        events.extend(evs);
        warnings.extend(ws);
    }
    let (evs, _) = parser.parse(&SseEvent::new(None, "[DONE]")).unwrap();
    events.extend(evs);
    assert!(
        warnings
            .iter()
            .all(|w| w.code != WarningCode::BlockOrderLost),
        "{warnings:?}"
    );
    let resp = accumulate(&events);
    assert!(matches!(
        &resp.message.content[0],
        ContentBlock::ToolCall { .. }
    ));
    assert!(matches!(&resp.message.content[1],
        ContentBlock::Text { text, .. } if text == "x"));
}

#[test]
fn empty_content_along_tool_chunks_stays_silent() {
    // Dialects ride `content: ""` on every chunk, including tool-call
    // ones; empty fragments fold nothing, so no BlockOrderLost.
    let mut parser = OpenAiChatCompletions.stream_parser();
    let chunks = [
        json!({"id": "c", "choices": [{"index": 0,
            "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {"content": "", "tool_calls": [
            {"index": 0, "id": "call_1", "type": "function",
             "function": {"name": "f", "arguments": "{\"x\""}},
        ]}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {"content": "", "tool_calls": [
            {"index": 0, "function": {"arguments": ":1}"}},
        ]}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
    ];
    let mut events = Vec::new();
    let mut warnings = Vec::new();
    for chunk in &chunks {
        let (evs, ws) = parser.parse(&chunk_event(chunk)).unwrap();
        events.extend(evs);
        warnings.extend(ws);
    }
    let (evs, _) = parser.parse(&SseEvent::new(None, "[DONE]")).unwrap();
    events.extend(evs);
    assert!(
        warnings
            .iter()
            .all(|w| w.code != WarningCode::BlockOrderLost),
        "{warnings:?}"
    );
    let resp = accumulate(&events);
    assert!(matches!(&resp.message.content[1],
        ContentBlock::ToolCall { arguments, .. } if arguments == "{\"x\":1}"));
}

#[test]
fn post_terminal_chunks_surface_as_unknown() {
    // After `[DONE]`, data chunks and repeated terminators alike cannot
    // be attributed (mirrors the Google gate).
    let mut parser = OpenAiChatCompletions.stream_parser();
    let (events, warnings) = parser.parse(&SseEvent::new(None, "[DONE]")).unwrap();
    assert!(warnings.is_empty());
    assert!(matches!(events.last().unwrap(), StreamEvent::MessageStop));

    let chunk = json!({"id": "c", "choices": [{"index": 0,
        "delta": {"content": "late"}, "finish_reason": null}]});
    let (events, warnings) = parser.parse(&chunk_event(&chunk)).unwrap();
    assert_eq!(events, vec![StreamEvent::Unknown]);
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0].code, WarningCode::UnknownStreamEvent);
    assert!(
        warnings[0].message.contains("after the stream terminated"),
        "{warnings:?}"
    );

    // A repeated terminator takes the same gate — no second MessageStop.
    let (events, warnings) = parser.parse(&SseEvent::new(None, "[DONE]")).unwrap();
    assert_eq!(events, vec![StreamEvent::Unknown]);
    assert_eq!(warnings[0].code, WarningCode::UnknownStreamEvent);
    assert!(parser.finish().is_ok());
}

#[test]
fn custom_tool_call_type_survives_streaming() {
    // Undocumented for deltas (chunk schemas only define `function`), but
    // dialects may stream custom calls; the payload maps like the
    // non-streaming parse — `custom.name` → name, `custom.input` →
    // arguments — and the reserved `type` key keeps the kind. The stream
    // never delivers `custom.input`, which warns at finalization like the
    // non-streaming parse of the same entry.
    let mut parser = OpenAiChatCompletions.stream_parser();
    let chunks = [
        json!({"id": "c", "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "call_c", "type": "custom", "custom": {"name": "run_sql"}},
        ]}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
    ];
    let mut events = Vec::new();
    let mut warnings = Vec::new();
    for chunk in &chunks {
        let (evs, ws) = parser.parse(&chunk_event(chunk)).unwrap();
        warnings.extend(ws);
        events.extend(evs);
    }
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].code, WarningCode::MalformedToolCall);
    assert!(
        warnings[0].message.contains("`custom.input` is missing"),
        "{warnings:?}"
    );
    // The opening block already carries the custom name.
    assert_eq!(
        ev(&events[1]),
        json!({"type": "block_start", "index": 0,
               "block": {"type": "tool_call", "id": "call_c", "name": "run_sql", "arguments": ""}})
    );
    let ContentBlock::ToolCall {
        id,
        name,
        arguments,
        extra,
        ..
    } = &finalized_block(&events)
    else {
        panic!("expected tool call")
    };
    assert_eq!(id.as_deref(), Some("call_c"));
    assert_eq!(name, "run_sql");
    assert!(arguments.is_empty());
    let ns = extra.get(F).unwrap();
    assert_eq!(ns.get("type"), Some(&json!("custom")));
    // `custom.name` is a modeled member — nothing rides the nested extra.
    assert!(!ns.contains_key("custom"), "{ns:?}");
}

#[test]
fn custom_tool_call_fragments_accumulate_and_rebuild() {
    // `custom.name` / `custom.input` accumulate across fragments, `input`
    // pieces stream as ToolArguments deltas, unmodeled payload members
    // nest under `custom`, and the finalized block re-serializes as a
    // proper custom entry.
    let mut parser = OpenAiChatCompletions.stream_parser();
    let chunks = [
        json!({"id": "c", "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "call_c", "type": "custom",
             "custom": {"name": "run_", "input": "SELECT", "vendor": true}},
        ]}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "custom": {"name": "sql", "input": " 1"}},
        ]}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
    ];
    let mut events = Vec::new();
    for chunk in &chunks {
        let (evs, ws) = parser.parse(&chunk_event(chunk)).unwrap();
        assert!(ws.is_empty(), "{ws:?}");
        events.extend(evs);
    }
    let pieces: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::BlockDelta {
                delta: BlockDelta::ToolArguments(piece),
                ..
            } => Some(piece.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(pieces, vec!["SELECT", " 1"]);

    let block = finalized_block(&events);
    let ContentBlock::ToolCall {
        id,
        name,
        arguments,
        extra,
        ..
    } = &block
    else {
        panic!("expected tool call")
    };
    assert_eq!(id.as_deref(), Some("call_c"));
    assert_eq!(name, "run_sql");
    assert_eq!(arguments, "SELECT 1");
    let ns = extra.get(F).unwrap();
    assert_eq!(ns.get("type"), Some(&json!("custom")));
    assert_eq!(ns.get("custom"), Some(&json!({"vendor": true})));

    // Round trip: the accumulated block rebuilds the wire entry.
    let req = Request::with_messages(vec![Message::assistant(vec![block])]);
    let (body, warnings) = request_from_ir(
        &req,
        None,
        CallMode::Unary,
        &ConvertOptions::default(),
        &OpenAiChatCompletionsOptions::default(),
    )
    .unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body["messages"][0]["tool_calls"][0],
        json!({"id": "call_c", "type": "custom",
               "custom": {"name": "run_sql", "input": "SELECT 1", "vendor": true}})
    );
}

#[test]
fn streamed_custom_call_matches_non_streaming_parse() {
    // Invariant: accumulating a complete entry from the stream yields the
    // same block as the non-streaming parse of that entry.
    let entry = json!({"id": "call_c", "type": "custom",
                       "custom": {"name": "run_sql", "input": "SELECT 1", "vendor": {"a": 1}}});
    let mut fragment = entry.clone();
    fragment["index"] = json!(0);
    let mut parser = OpenAiChatCompletions.stream_parser();
    let mut events = Vec::new();
    for chunk in [
        json!({"id": "c", "choices": [{"index": 0,
            "delta": {"tool_calls": [fragment]}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
    ] {
        let (evs, ws) = parser.parse(&chunk_event(&chunk)).unwrap();
        assert!(ws.is_empty(), "{ws:?}");
        events.extend(evs);
    }
    let streamed = finalized_block(&events);

    let body = json!({"messages": [{"role": "assistant", "tool_calls": [entry]}]});
    let (req, ws) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert!(ws.is_empty(), "{ws:?}");
    assert_eq!(streamed, req.messages[0].content[0]);
}

/// Parses a full assistant `tool_calls` entry through the non-streaming
/// request parser, returning the block and the warnings.
fn non_streaming_entry(entry: &Value) -> (ContentBlock, Vec<ConversionWarning>) {
    let body = json!({"messages": [{"role": "assistant", "tool_calls": [entry]}]});
    let (req, ws) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    (req.messages[0].content[0].clone(), ws)
}

/// Streams one `tool_calls` fragment array per chunk, then the finish
/// chunk and `[DONE]`; returns the unified events and warnings.
fn stream_tool_call_chunks(
    fragment_chunks: &[Value],
) -> (Vec<StreamEvent>, Vec<ConversionWarning>) {
    let mut parser = OpenAiChatCompletions.stream_parser();
    let mut events = Vec::new();
    let mut warnings = Vec::new();
    for fragments in fragment_chunks {
        let chunk = json!({"id": "c", "choices": [{"index": 0,
            "delta": {"tool_calls": fragments}, "finish_reason": null}]});
        let (evs, ws) = parser.parse(&chunk_event(&chunk)).unwrap();
        events.extend(evs);
        warnings.extend(ws);
    }
    for terminal in [
        chunk_event(
            &json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
        ),
        SseEvent::new(None, "[DONE]"),
    ] {
        let (evs, ws) = parser.parse(&terminal).unwrap();
        events.extend(evs);
        warnings.extend(ws);
    }
    (events, warnings)
}

/// `code:severity` fingerprints, sorted, for streaming ↔ non-streaming
/// warning-parity assertions.
fn warning_fingerprints(warnings: &[ConversionWarning]) -> Vec<String> {
    let mut out: Vec<String> = warnings
        .iter()
        .map(|w| format!("{:?}:{:?}", w.code, w.severity))
        .collect();
    out.sort();
    out
}

#[test]
fn unknown_call_type_streams_in_mirror_mode() {
    // An unknown declared `type` cannot use the unified name/arguments
    // channels; the accumulated payload folds verbatim into the block
    // extra at finalization — the same block the non-streaming parse
    // produces for the complete entry — and rebuilds verbatim.
    let entry = json!({"id": "x", "type": "dialect_call",
                       "function": {"name": "f", "arguments": "{}", "x": 1}});
    let mut fragment = entry.clone();
    fragment["index"] = json!(0);
    let mut parser = OpenAiChatCompletions.stream_parser();
    let mut events = Vec::new();
    let mut warnings = Vec::new();
    for chunk in [
        json!({"id": "c", "choices": [{"index": 0,
            "delta": {"tool_calls": [fragment]}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
    ] {
        let (evs, ws) = parser.parse(&chunk_event(&chunk)).unwrap();
        events.extend(evs);
        warnings.extend(ws);
    }
    // Mirror mode streams no argument deltas.
    assert!(
        !events.iter().any(|e| matches!(
            e,
            StreamEvent::BlockDelta {
                delta: BlockDelta::ToolArguments(_),
                ..
            }
        )),
        "{events:#?}"
    );
    // One cosmetic mirror warning, like the non-streaming parse.
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].code, WarningCode::MalformedField);
    assert!(
        warnings[0].message.contains("mirrored verbatim"),
        "{warnings:?}"
    );

    let streamed = finalized_block(&events);
    let ContentBlock::ToolCall {
        id,
        name,
        arguments,
        extra,
        ..
    } = &streamed
    else {
        panic!("expected tool call");
    };
    assert_eq!(id.as_deref(), Some("x"));
    assert!(name.is_empty());
    assert!(arguments.is_empty());
    let ns = extra.get(F).unwrap();
    assert_eq!(ns.get("type"), Some(&json!("dialect_call")));
    assert_eq!(
        ns.get("function"),
        Some(&json!({"name": "f", "arguments": "{}", "x": 1}))
    );

    // Invariant: identical to the non-streaming parse of the entry.
    let (block, ws) = non_streaming_entry(&entry);
    assert_eq!(ws.len(), 1, "{ws:?}");
    assert_eq!(ws[0].code, WarningCode::MalformedField);
    assert_eq!(streamed, block);

    // Round trip: the block rebuilds the wire entry verbatim, including
    // the payload `name` / `arguments` members.
    let req = Request::with_messages(vec![Message::assistant(vec![streamed])]);
    let (body, ws) = request_from_ir(
        &req,
        None,
        CallMode::Unary,
        &ConvertOptions::default(),
        &OpenAiChatCompletionsOptions::default(),
    )
    .unwrap();
    assert!(ws.is_empty(), "{ws:?}");
    assert_eq!(body["messages"][0]["tool_calls"][0], entry);
}

#[test]
fn unknown_call_type_fragments_fold_across_chunks() {
    // Mirror-mode accumulation: modeled string members concatenate across
    // fragments and fold once at finalization (a piecewise extra merge
    // would drop earlier pieces).
    let mut parser = OpenAiChatCompletions.stream_parser();
    let chunks = [
        json!({"id": "c", "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "x", "type": "dialect_call",
             "function": {"name": "f", "arguments": "{\"a\":"}},
        ]}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "1}", "x": 1}},
        ]}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
    ];
    let mut events = Vec::new();
    let mut warnings = Vec::new();
    for chunk in &chunks {
        let (evs, ws) = parser.parse(&chunk_event(chunk)).unwrap();
        events.extend(evs);
        warnings.extend(ws);
    }
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].code, WarningCode::MalformedField);

    // Equals the non-streaming parse of the assembled entry.
    let entry = json!({"id": "x", "type": "dialect_call",
                       "function": {"name": "f", "arguments": "{\"a\":1}", "x": 1}});
    let (block, _) = non_streaming_entry(&entry);
    assert_eq!(finalized_block(&events), block);
}

#[test]
fn late_unknown_type_switches_to_mirror() {
    // Pathological order: payloads lock `function` first, the unknown
    // `type` arrives later. The call still folds into mirror form, and
    // the argument deltas already emitted are superseded by the
    // authoritative BlockStop block (§ 9).
    let mut parser = OpenAiChatCompletions.stream_parser();
    let chunks = [
        json!({"id": "c", "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "x", "function": {"name": "f", "arguments": "{}"}},
        ]}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "type": "dialect_call", "function": {"x": 1}},
        ]}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
    ];
    let mut events = Vec::new();
    let mut warnings = Vec::new();
    for chunk in &chunks {
        let (evs, ws) = parser.parse(&chunk_event(chunk)).unwrap();
        events.extend(evs);
        warnings.extend(ws);
    }
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].code, WarningCode::MalformedField);

    let entry = json!({"id": "x", "type": "dialect_call",
                       "function": {"name": "f", "arguments": "{}", "x": 1}});
    let (block, _) = non_streaming_entry(&entry);
    assert_eq!(finalized_block(&events), block);

    // Accumulation adopts the authoritative fold: the pre-switch argument
    // delta does not survive into the unified field.
    let (evs, _) = parser.parse(&SseEvent::new(None, "[DONE]")).unwrap();
    events.extend(evs);
    let resp = accumulate(&events);
    let ContentBlock::ToolCall { arguments, .. } = &resp.message.content[0] else {
        panic!("expected tool call");
    };
    assert!(arguments.is_empty());
}

#[test]
fn typeless_custom_payload_infers_custom_kind() {
    // A typeless call streaming only a `custom` payload is inferred as
    // custom (silent § 1 canonicalization): the reserved `type` key is
    // written so the rebuilt entry declares `type: "custom"`, and the
    // block matches the non-streaming parse of the same entry.
    let entry = json!({"id": "call_c", "custom": {"name": "run_sql", "input": "SELECT 1"}});
    let mut fragment = entry.clone();
    fragment["index"] = json!(0);
    let mut parser = OpenAiChatCompletions.stream_parser();
    let mut events = Vec::new();
    for chunk in [
        json!({"id": "c", "choices": [{"index": 0,
            "delta": {"tool_calls": [fragment]}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
    ] {
        let (evs, ws) = parser.parse(&chunk_event(&chunk)).unwrap();
        assert!(ws.is_empty(), "{ws:?}");
        events.extend(evs);
    }
    let streamed = finalized_block(&events);
    let ContentBlock::ToolCall {
        name,
        arguments,
        extra,
        ..
    } = &streamed
    else {
        panic!("expected tool call");
    };
    assert_eq!(name, "run_sql");
    assert_eq!(arguments, "SELECT 1");
    assert_eq!(extra.get(F).unwrap().get("type"), Some(&json!("custom")));

    // Invariant: matches the non-streaming parse of the typeless entry.
    let (block, ws) = non_streaming_entry(&entry);
    assert!(ws.is_empty(), "{ws:?}");
    assert_eq!(streamed, block);

    // The rebuilt entry carries the inferred type.
    let req = Request::with_messages(vec![Message::assistant(vec![streamed])]);
    let (body, ws) = request_from_ir(
        &req,
        None,
        CallMode::Unary,
        &ConvertOptions::default(),
        &OpenAiChatCompletionsOptions::default(),
    )
    .unwrap();
    assert!(ws.is_empty(), "{ws:?}");
    assert_eq!(
        body["messages"][0]["tool_calls"][0],
        json!({"id": "call_c", "type": "custom",
               "custom": {"name": "run_sql", "input": "SELECT 1"}})
    );
}

#[test]
fn late_function_declaration_overrides_inferred_custom() {
    // Payload inference locked `custom`, then the first explicit
    // declaration disagrees. The declaration is authoritative, matching
    // the non-streaming parse of the assembled entry: the accumulated
    // custom payload mirrors into the namespace and the call finalizes
    // as a `function` call missing its whole payload (three semantic
    // warnings); argument deltas already emitted are superseded by the
    // authoritative BlockStop block (§ 9).
    let (events, warnings) = stream_tool_call_chunks(&[
        json!([{"index": 0, "id": "x", "custom": {"name": "run_sql", "input": "SELECT 1"}}]),
        json!([{"index": 0, "type": "function"}]),
    ]);
    let entry = json!({"id": "x", "type": "function",
                       "custom": {"name": "run_sql", "input": "SELECT 1"}});
    let (block, expected) = non_streaming_entry(&entry);
    assert_eq!(expected.len(), 3, "{expected:?}");
    assert_eq!(finalized_block(&events), block);
    assert_eq!(
        warning_fingerprints(&warnings),
        warning_fingerprints(&expected)
    );

    // Accumulation adopts the authoritative fold: the pre-switch name
    // and argument deltas do not survive into the unified fields.
    let resp = accumulate(&events);
    let ContentBlock::ToolCall {
        name, arguments, ..
    } = &resp.message.content[0]
    else {
        panic!("expected tool call");
    };
    assert!(name.is_empty());
    assert!(arguments.is_empty());
}

#[test]
fn late_custom_declaration_overrides_inferred_function() {
    let (events, warnings) = stream_tool_call_chunks(&[
        json!([{"index": 0, "id": "x", "function": {"name": "f", "arguments": "{}"}}]),
        json!([{"index": 0, "type": "custom"}]),
    ]);
    let entry = json!({"id": "x", "type": "custom",
                       "function": {"name": "f", "arguments": "{}"}});
    let (block, expected) = non_streaming_entry(&entry);
    assert_eq!(finalized_block(&events), block);
    assert_eq!(
        warning_fingerprints(&warnings),
        warning_fingerprints(&expected)
    );
    // The namespace keeps the declared kind and the mirrored payload.
    let ContentBlock::ToolCall { extra, .. } = finalized_block(&events) else {
        panic!("expected tool call");
    };
    let ns = extra.get(F).unwrap();
    assert_eq!(ns.get("type"), Some(&json!("custom")));
    assert_eq!(
        ns.get("function"),
        Some(&json!({"name": "f", "arguments": "{}"}))
    );
}

#[test]
fn late_declaration_matching_inference_is_silent() {
    // A late declaration that agrees with the inferred kind changes
    // nothing and warns nothing.
    let (events, warnings) = stream_tool_call_chunks(&[
        json!([{"index": 0, "id": "x", "custom": {"name": "run_sql", "input": "SELECT 1"}}]),
        json!([{"index": 0, "type": "custom"}]),
    ]);
    assert!(warnings.is_empty(), "{warnings:?}");
    let entry = json!({"id": "x", "type": "custom",
                       "custom": {"name": "run_sql", "input": "SELECT 1"}});
    let (block, ws) = non_streaming_entry(&entry);
    assert!(ws.is_empty(), "{ws:?}");
    assert_eq!(finalized_block(&events), block);

    // Same for the `function` default (no reserved key written).
    let (events, warnings) = stream_tool_call_chunks(&[
        json!([{"index": 0, "id": "x", "function": {"name": "f", "arguments": "{}"}}]),
        json!([{"index": 0, "type": "function"}]),
    ]);
    assert!(warnings.is_empty(), "{warnings:?}");
    let (block, _) = non_streaming_entry(&json!({"id": "x", "type": "function",
        "function": {"name": "f", "arguments": "{}"}}));
    assert_eq!(finalized_block(&events), block);
}

#[test]
fn conflicting_type_declarations_first_wins() {
    // A later declaration that differs from the first is ignored with
    // one semantic warning — adopting it could silently change the call
    // shape mid-stream.
    let base = json!([{"index": 0, "id": "x", "type": "function",
                       "function": {"name": "f", "arguments": "{}"}}]);
    let (function_block, ws) = non_streaming_entry(&json!({"id": "x", "type": "function",
        "function": {"name": "f", "arguments": "{}"}}));
    assert!(ws.is_empty(), "{ws:?}");
    for later in [json!("custom"), json!("dialect_call"), json!(5)] {
        let (events, warnings) =
            stream_tool_call_chunks(&[base.clone(), json!([{"index": 0, "type": later.clone()}])]);
        assert_eq!(warnings.len(), 1, "{later}: {warnings:?}");
        assert_eq!(warnings[0].code, WarningCode::MalformedToolCall);
        assert_eq!(warnings[0].severity, WarningSeverity::Semantic);
        assert!(
            warnings[0].message.contains("first declaration wins"),
            "{warnings:?}"
        );
        // The declared function shape stays untouched (no mirror switch).
        assert_eq!(finalized_block(&events), function_block, "{later}");
    }

    // Repeating the same declaration is a consistent no-op.
    let (events, warnings) =
        stream_tool_call_chunks(&[base.clone(), json!([{"index": 0, "type": "function"}])]);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(finalized_block(&events), function_block);

    // The reverse direction warns identically: a declared custom call
    // ignores a later `function` declaration instead of dropping it
    // silently.
    let (events, warnings) = stream_tool_call_chunks(&[
        json!([{"index": 0, "id": "x", "type": "custom",
                "custom": {"name": "run_sql", "input": "SELECT 1"}}]),
        json!([{"index": 0, "type": "function"}]),
    ]);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].code, WarningCode::MalformedToolCall);
    let (custom_block, ws) = non_streaming_entry(&json!({"id": "x", "type": "custom",
        "custom": {"name": "run_sql", "input": "SELECT 1"}}));
    assert!(ws.is_empty(), "{ws:?}");
    assert_eq!(finalized_block(&events), custom_block);

    // First wins also protects a mirror-mode call: a late known
    // declaration cannot un-mirror it.
    let (events, warnings) = stream_tool_call_chunks(&[
        json!([{"index": 0, "id": "x", "type": "dialect_call",
                "function": {"name": "f", "arguments": "{}"}}]),
        json!([{"index": 0, "type": "function"}]),
    ]);
    assert_eq!(warnings.len(), 2, "{warnings:?}");
    assert_eq!(warnings[0].code, WarningCode::MalformedToolCall);
    assert_eq!(warnings[1].code, WarningCode::MalformedField);
    let (mirror_block, _) = non_streaming_entry(&json!({"id": "x", "type": "dialect_call",
        "function": {"name": "f", "arguments": "{}"}}));
    assert_eq!(finalized_block(&events), mirror_block);
}

#[test]
fn non_string_type_parity_with_non_streaming() {
    // A non-string, non-`null` `type` mirrors the call verbatim on both
    // sides: same block, one cosmetic `MalformedField` each (the mirror
    // re-serializes, so nothing is lost).
    for ty in [json!(5), json!({"a": 1}), json!([1, 2])] {
        for payload in [
            Some(("function", json!({"name": "f", "arguments": "{}", "x": 1}))),
            Some(("custom", json!({"name": "run_sql", "input": "SELECT 1"}))),
            None,
        ] {
            let mut entry = json!({"id": "call_t", "type": ty.clone()});
            if let Some((key, value)) = &payload {
                entry[*key] = value.clone();
            }
            let mut fragment = entry.clone();
            fragment["index"] = json!(0);
            let (events, ws_stream) = stream_tool_call_chunks(&[json!([fragment])]);
            // Mirror mode: no argument deltas.
            assert!(
                !events.iter().any(|e| matches!(
                    e,
                    StreamEvent::BlockDelta {
                        delta: BlockDelta::ToolArguments(_),
                        ..
                    }
                )),
                "{entry}: {events:#?}"
            );
            assert_eq!(ws_stream.len(), 1, "{entry}: {ws_stream:?}");
            assert_eq!(ws_stream[0].code, WarningCode::MalformedField);
            assert_eq!(ws_stream[0].severity, WarningSeverity::Cosmetic);
            assert!(ws_stream[0].message.contains("non-string"), "{ws_stream:?}");

            let (block, ws_plain) = non_streaming_entry(&entry);
            assert_eq!(finalized_block(&events), block, "{entry}");
            assert_eq!(
                warning_fingerprints(&ws_stream),
                warning_fingerprints(&ws_plain),
                "{entry}"
            );
        }
    }
}

#[test]
fn non_string_type_round_trips_verbatim() {
    // The rebuilt entry restores the non-string `type` and its payload
    // verbatim — no fabricated `function` object.
    let entry = json!({"id": "call_t", "type": 5,
                       "function": {"name": "f", "arguments": "{}"}});
    let mut fragment = entry.clone();
    fragment["index"] = json!(0);
    let (events, _) = stream_tool_call_chunks(&[json!([fragment])]);
    let req = Request::with_messages(vec![Message::assistant(vec![finalized_block(&events)])]);
    let (body, ws) = request_from_ir(
        &req,
        None,
        CallMode::Unary,
        &ConvertOptions::default(),
        &OpenAiChatCompletionsOptions::default(),
    )
    .unwrap();
    assert!(ws.is_empty(), "{ws:?}");
    assert_eq!(body["messages"][0]["tool_calls"][0], entry);
}

#[test]
fn late_non_string_type_switches_to_mirror() {
    // Payload inference locked `function`, then the first declaration is
    // non-string: the call switches to mirror mode (one cosmetic warning
    // at the declaring fragment), later payload pieces keep accumulating
    // there, and the fold matches the non-streaming parse of the
    // assembled entry. The pre-switch argument delta is superseded by
    // the authoritative BlockStop block (§ 9).
    let (events, warnings) = stream_tool_call_chunks(&[
        json!([{"index": 0, "id": "x", "function": {"name": "f", "arguments": "{\"a\":"}}]),
        json!([{"index": 0, "type": 5, "function": {"arguments": "1}"}}]),
    ]);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].code, WarningCode::MalformedField);
    assert_eq!(warnings[0].severity, WarningSeverity::Cosmetic);

    let entry = json!({"id": "x", "type": 5,
                       "function": {"name": "f", "arguments": "{\"a\":1}"}});
    let (block, ws) = non_streaming_entry(&entry);
    assert_eq!(warning_fingerprints(&warnings), warning_fingerprints(&ws));
    assert_eq!(finalized_block(&events), block);

    let resp = accumulate(&events);
    let ContentBlock::ToolCall { arguments, .. } = &resp.message.content[0] else {
        panic!("expected tool call");
    };
    assert!(arguments.is_empty());
}

#[test]
fn null_type_stays_absent() {
    // `type: null` canonicalizes to absent on both sides — payload
    // inference applies as if no declaration ever arrived.
    let entry = json!({"id": "c", "type": null,
                       "custom": {"name": "run_sql", "input": "SELECT 1"}});
    let mut fragment = entry.clone();
    fragment["index"] = json!(0);
    let (events, warnings) = stream_tool_call_chunks(&[json!([fragment])]);
    assert!(warnings.is_empty(), "{warnings:?}");
    let (block, ws) = non_streaming_entry(&entry);
    assert!(ws.is_empty(), "{ws:?}");
    assert_eq!(finalized_block(&events), block);
    // Inference still runs: the lone custom payload wrote the reserved
    // key on both sides.
    let ContentBlock::ToolCall { extra, .. } = finalized_block(&events) else {
        panic!("expected tool call");
    };
    assert_eq!(extra.get(F).unwrap().get("type"), Some(&json!("custom")));
}

#[test]
fn never_delivered_payload_members_warn_at_finalize() {
    // Warning parity with the non-streaming lenient parse: each payload
    // member the stream never delivered warns `MalformedToolCall` once;
    // explicit empty strings are legitimate deliveries and stay silent.
    let run = |fragments: Value| -> Vec<ConversionWarning> {
        let mut parser = OpenAiChatCompletions.stream_parser();
        let mut warnings = Vec::new();
        for chunk in [
            json!({"id": "c", "choices": [{"index": 0,
                "delta": {"tool_calls": fragments}, "finish_reason": null}]}),
            json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
        ] {
            let (_, ws) = parser.parse(&chunk_event(&chunk)).unwrap();
            warnings.extend(ws);
        }
        warnings
    };

    // Missing arguments: one warning, same code as the non-streaming
    // parse of the same entry.
    let ws = run(json!([{"index": 0, "id": "c1", "function": {"name": "f"}}]));
    assert_eq!(ws.len(), 1, "{ws:?}");
    assert_eq!(ws[0].code, WarningCode::MalformedToolCall);
    assert!(
        ws[0].message.contains("`function.arguments` is missing"),
        "{ws:?}"
    );

    // Missing name: one warning.
    let ws = run(json!([{"index": 0, "id": "c1", "function": {"arguments": "{}"}}]));
    assert_eq!(ws.len(), 1, "{ws:?}");
    assert!(
        ws[0].message.contains("`function.name` is missing"),
        "{ws:?}"
    );

    // Explicit empty strings are complete deliveries: silent.
    let ws = run(json!([{"index": 0, "id": "c1", "function": {"name": "", "arguments": ""}}]));
    assert!(ws.is_empty(), "{ws:?}");

    // `null` members do not count as delivered (the non-streaming parse
    // treats them as missing too).
    let ws = run(json!([
        {"index": 0, "id": "c1", "function": {"name": "f", "arguments": null}},
    ]));
    assert_eq!(ws.len(), 1, "{ws:?}");
    assert!(
        ws[0].message.contains("`function.arguments` is missing"),
        "{ws:?}"
    );

    // A non-string member warns at the routing site only — finalization
    // does not report the same member a second time.
    let ws = run(json!([
        {"index": 0, "id": "c1", "function": {"name": "f", "arguments": 5}},
    ]));
    assert_eq!(ws.len(), 1, "{ws:?}");
    assert!(ws[0].message.contains("non-string"), "{ws:?}");

    // Custom calls name their own argument channel.
    let ws = run(json!([{"index": 0, "id": "c1", "type": "custom", "custom": {"name": "x"}}]));
    assert_eq!(ws.len(), 1, "{ws:?}");
    assert!(
        ws[0].message.contains("`custom.input` is missing"),
        "{ws:?}"
    );

    // A call that never streams any payload reports the payload and both
    // members, like the non-streaming parse of a bare entry.
    let ws = run(json!([{"index": 0, "id": "c1"}]));
    assert_eq!(ws.len(), 3, "{ws:?}");
    assert!(ws.iter().all(|w| w.code == WarningCode::MalformedToolCall));
    assert!(ws[0].message.contains("no `function` payload"), "{ws:?}");
    assert!(
        ws[1].message.contains("`function.name` is missing"),
        "{ws:?}"
    );
    assert!(
        ws[2].message.contains("`function.arguments` is missing"),
        "{ws:?}"
    );
}

#[test]
fn mixed_payload_kinds_mirror_and_warn() {
    // A call cannot switch payload kinds mid-stream; the conflicting
    // fragment is mirrored verbatim instead of spliced, with a warning.
    let mut parser = OpenAiChatCompletions.stream_parser();
    let first = json!({"id": "c", "choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "id": "call_m", "type": "function",
         "function": {"name": "f", "arguments": "{}"}},
    ]}, "finish_reason": null}]});
    let (mut events, ws) = parser.parse(&chunk_event(&first)).unwrap();
    assert!(ws.is_empty(), "{ws:?}");

    let conflicting = json!({"id": "c", "choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "custom": {"name": "x", "input": "y"}},
    ]}, "finish_reason": null}]});
    let (evs, ws) = parser.parse(&chunk_event(&conflicting)).unwrap();
    events.extend(evs);
    assert_eq!(ws.len(), 1, "{ws:?}");
    assert_eq!(ws[0].code, WarningCode::MalformedToolCall);
    assert!(ws[0].message.contains("mixes"), "{ws:?}");

    let finish =
        json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]});
    let (evs, ws) = parser.parse(&chunk_event(&finish)).unwrap();
    assert!(ws.is_empty(), "{ws:?}");
    events.extend(evs);

    let ContentBlock::ToolCall {
        name,
        arguments,
        extra,
        ..
    } = &finalized_block(&events)
    else {
        panic!("expected tool call")
    };
    // The locked function channels are untouched; the foreign payload
    // rides the extra wholesale.
    assert_eq!(name, "f");
    assert_eq!(arguments, "{}");
    assert_eq!(
        extra.get(F).unwrap().get("custom"),
        Some(&json!({"name": "x", "input": "y"}))
    );
}

#[test]
fn custom_stream_without_id_warns_at_finalize() {
    let mut parser = OpenAiChatCompletions.stream_parser();
    let mut warnings = Vec::new();
    let mut events = Vec::new();
    for chunk in [
        json!({"id": "c", "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "type": "custom", "custom": {"name": "run_sql", "input": "SELECT 1"}},
        ]}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
    ] {
        let (evs, ws) = parser.parse(&chunk_event(&chunk)).unwrap();
        events.extend(evs);
        warnings.extend(ws);
    }
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].code, WarningCode::MalformedToolCall);
    assert!(warnings[0].message.contains("no `id`"), "{warnings:?}");
    assert!(matches!(
        finalized_block(&events),
        ContentBlock::ToolCall { id: None, .. }
    ));
}

#[test]
fn non_string_argument_fragments_warn() {
    // A non-string `function.arguments` (or `custom.input`) piece cannot
    // be spliced into the accumulated string and is dropped with a
    // warning instead of silently.
    let mut parser = OpenAiChatCompletions.stream_parser();
    let chunk = json!({"id": "c", "choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "id": "call_n", "function": {"name": "f", "arguments": {"x": 1}}},
    ]}, "finish_reason": null}]});
    let (_, ws) = parser.parse(&chunk_event(&chunk)).unwrap();
    assert_eq!(ws.len(), 1, "{ws:?}");
    assert_eq!(ws[0].code, WarningCode::MalformedToolCall);
    assert!(ws[0].message.contains("function.arguments"), "{ws:?}");

    let mut parser = OpenAiChatCompletions.stream_parser();
    let chunk = json!({"id": "c", "choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "id": "call_n", "type": "custom", "custom": {"name": "f", "input": 5}},
    ]}, "finish_reason": null}]});
    let (_, ws) = parser.parse(&chunk_event(&chunk)).unwrap();
    assert_eq!(ws.len(), 1, "{ws:?}");
    assert_eq!(ws[0].code, WarningCode::MalformedToolCall);
    assert!(ws[0].message.contains("custom.input"), "{ws:?}");
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

// ------------------------------------- non-string channel values (R1)

/// Runs the chunks plus `[DONE]` through a fresh parser, collecting
/// events and warnings.
fn run_chunks(chunks: &[Value]) -> (Vec<StreamEvent>, Vec<ConversionWarning>) {
    let mut parser = OpenAiChatCompletions.stream_parser();
    let mut events = Vec::new();
    let mut warnings = Vec::new();
    for chunk in chunks {
        let (evs, ws) = parser.parse(&chunk_event(chunk)).unwrap();
        events.extend(evs);
        warnings.extend(ws);
    }
    let (evs, ws) = parser.parse(&SseEvent::new(None, "[DONE]")).unwrap();
    events.extend(evs);
    warnings.extend(ws);
    (events, warnings)
}

#[test]
fn non_string_channel_value_without_block_warns_and_surfaces_unknown() {
    // No block open: like any unknown delta field the value cannot be
    // attributed — MalformedField plus one UnknownStreamEvent per field
    // name, and the chunk surfaces as Unknown (nothing silent).
    for field in ["content", "refusal"] {
        let (events, warnings) = run_chunks(&[
            json!({"id": "c", "choices": [{"index": 0,
                "delta": {"role": "assistant",
                          field: [{"type": "text", "text": "hi"}]},
                "finish_reason": null}]}),
            json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        ]);
        let malformed: Vec<_> = warnings
            .iter()
            .filter(|w| w.code == WarningCode::MalformedField)
            .collect();
        assert_eq!(malformed.len(), 1, "{field}: {warnings:?}");
        assert_eq!(malformed[0].location, format!("/choices/0/delta/{field}"));
        assert!(
            malformed[0]
                .message
                .contains(&format!("non-string `{field}`")),
            "{field}: {warnings:?}"
        );
        let unknown: Vec<_> = warnings
            .iter()
            .filter(|w| w.code == WarningCode::UnknownStreamEvent)
            .collect();
        assert_eq!(unknown.len(), 1, "{field}: {warnings:?}");
        assert!(
            unknown[0].message.contains(&format!("`{field}`")),
            "{field}: {warnings:?}"
        );
        assert!(events.contains(&StreamEvent::Unknown), "{field}");
        let resp = accumulate(&events);
        assert!(resp.message.content.is_empty(), "{field}");
    }
}

#[test]
fn non_string_channel_value_folds_into_open_reasoning_block() {
    // With a block open the value takes the leftover fold — parity with
    // the reasoning_field branch: warned, surfaced as an Other delta,
    // folded into the block extra and replayed on the message level.
    let (events, warnings) = run_chunks(&[
        json!({"id": "c", "choices": [{"index": 0,
            "delta": {"role": "assistant", "reasoning_content": "think"},
            "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0,
            "delta": {"content": [{"type": "text", "text": "hi"}]},
            "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
    ]);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].code, WarningCode::MalformedField);
    assert_eq!(warnings[0].location, "/choices/0/delta/content");
    assert!(
        events.iter().any(|e| matches!(e,
            StreamEvent::BlockDelta { index: 0, delta: BlockDelta::Other(v), .. }
                if v.get("content").is_some())),
        "{events:#?}"
    );

    let resp = accumulate(&events);
    assert_eq!(resp.message.content.len(), 1, "{:#?}", resp.message.content);
    let ContentBlock::Thinking { text, extra, .. } = &resp.message.content[0] else {
        panic!("expected thinking block: {:?}", resp.message.content);
    };
    assert_eq!(text.as_deref(), Some("think"));
    // Thinking-channel leftovers stay top-level in the namespace.
    assert_eq!(
        extra.get(F).unwrap().get("content"),
        Some(&json!([{"type": "text", "text": "hi"}]))
    );

    // Replay: the thinking-block extra merges into the containing message,
    // restoring the dialect's array `content` verbatim.
    let (body, ws) = replay_message(&resp.message);
    assert!(ws.is_empty(), "{ws:?}");
    let msg = &body["messages"][0];
    assert_eq!(msg["reasoning_content"], json!("think"));
    assert_eq!(msg["content"], json!([{"type": "text", "text": "hi"}]));
}

#[test]
fn non_array_tool_calls_delta_warns_and_folds() {
    // A non-array `tool_calls` opens no ToolCall block: it warns and
    // folds like an unknown delta field (here under the `message`
    // reserved key of the open text block), replaying on the message.
    let (events, warnings) = run_chunks(&[
        json!({"id": "c", "choices": [{"index": 0,
            "delta": {"role": "assistant", "content": "hi"}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0,
            "delta": {"tool_calls": {"id": "x"}}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
    ]);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].code, WarningCode::MalformedField);
    assert_eq!(warnings[0].location, "/choices/0/delta/tool_calls");
    assert!(
        warnings[0].message.contains("non-array `tool_calls`"),
        "{warnings:?}"
    );
    assert!(
        events.iter().all(|e| !matches!(
            e,
            StreamEvent::BlockStart {
                block: ContentBlock::ToolCall { .. },
                ..
            }
        )),
        "no tool call block: {events:#?}"
    );

    let resp = accumulate(&events);
    assert_eq!(resp.message.content.len(), 1);
    let ContentBlock::Text { text, extra, .. } = &resp.message.content[0] else {
        panic!("expected text block: {:?}", resp.message.content);
    };
    assert_eq!(text, "hi");
    assert_eq!(
        extra.get(F).unwrap().get(text_block_reserved_key::MESSAGE),
        Some(&json!({"tool_calls": {"id": "x"}}))
    );

    let (body, ws) = replay_message(&resp.message);
    assert!(ws.is_empty(), "{ws:?}");
    assert_eq!(body["messages"][0]["tool_calls"], json!({"id": "x"}));
}

#[test]
fn null_channel_values_stay_silent() {
    // Pin: `null` is the absent canonical form on every channel — no
    // warning, no event.
    let events = run_chunks_no_warnings(&[
        json!({"id": "c", "choices": [{"index": 0,
            "delta": {"role": "assistant", "content": null, "refusal": null,
                      "tool_calls": null},
            "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0,
            "delta": {"content": "hi"}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
    ]);
    let resp = accumulate(&events);
    assert_eq!(resp.message.content.len(), 1);
    assert!(matches!(&resp.message.content[0],
        ContentBlock::Text { text, extra, .. } if text == "hi" && extra.is_empty()));
}

// ----------------------------------------------- malformed usage (R5)

#[test]
fn usage_chunk_missing_core_fields_warns_and_drops_snapshot() {
    // The include_usage final chunk with the wire-required core fields
    // missing: warned, never zeroed.
    let (events, warnings) = run_chunks(&[
        json!({"id": "c", "choices": [{"index": 0,
            "delta": {"role": "assistant", "content": "hi"}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        json!({"id": "c", "choices": [], "usage": {"prompt_tokens": 2}}),
    ]);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].code, WarningCode::MalformedField);
    assert_eq!(warnings[0].location, "/usage");
    assert!(
        events
            .iter()
            .all(|e| !matches!(e, StreamEvent::MessageDelta { usage: Some(_), .. }))
    );
}

#[test]
fn malformed_usage_chunk_warns_and_drops_snapshot() {
    // The include_usage final chunk (empty choices) with a malformed
    // usage: warned, no usage snapshot emitted, stream completes.
    let (events, warnings) = run_chunks(&[
        json!({"id": "c", "choices": [{"index": 0,
            "delta": {"role": "assistant", "content": "hi"}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        json!({"id": "c", "choices": [],
            "usage": {"prompt_tokens": 12.5, "completion_tokens": 3}}),
    ]);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].code, WarningCode::MalformedField);
    assert_eq!(warnings[0].location, "/usage");
    assert!(
        events
            .iter()
            .all(|e| !matches!(e, StreamEvent::MessageDelta { usage: Some(_), .. })),
        "no usage snapshot: {events:#?}"
    );
    let resp = accumulate(&events);
    assert_eq!(resp.text(), "hi");
    assert!(resp.usage.is_none());

    // A finish chunk carrying malformed usage still reports the stop
    // reason; only the usage degrades.
    let (events, warnings) = run_chunks(&[
        json!({"id": "c", "choices": [{"index": 0,
            "delta": {"role": "assistant", "content": "hi"}, "finish_reason": null}]}),
        json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": "busy"}),
    ]);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].code, WarningCode::MalformedField);
    assert_eq!(warnings[0].location, "/usage");
    assert!(
        events.iter().any(|e| matches!(
            e,
            StreamEvent::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                usage: None,
                ..
            }
        )),
        "{events:#?}"
    );
    let resp = accumulate(&events);
    assert_eq!(resp.stop_reason, Some(StopReason::EndTurn));
    assert!(resp.usage.is_none());
}
