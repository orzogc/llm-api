//! Streaming tests for the `openai_responses` format: complete SSE
//! sessions from fixtures, block-index flattening, terminal handling,
//! truncation and accumulation.
//!
//! Expected events are asserted through their IR JSON representation
//! (`StreamEvent` variants are `#[non_exhaustive]` and cannot be
//! constructed outside the crate).

use serde_json::{Value, json};

use llm_api::formats::openai_responses::{
    OpenAiResponses, request_from_ir, text_block_reserved_key,
};
use llm_api::http::{SseEvent, SseParser};
use llm_api::{
    Accumulator, ApiFormat, BlockDelta, CallMode, ContentBlock, ConversionWarning, ConvertOptions,
    Error, Request, StopReason, StreamEvent, StreamItem, WarningCode,
};

const F: &str = "openai_responses";

fn fixture(name: &str) -> Vec<u8> {
    let path = format!(
        "{}/tests/fixtures/openai_responses/{name}",
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
    let mut parser = OpenAiResponses.stream_parser();
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

/// Feeds hand-built JSON frames through a fresh stream parser; event
/// parsing must succeed. Returns the unified events, warnings and the
/// `finish` result.
fn feed_frames(
    frames: &[Value],
) -> (
    Vec<StreamEvent>,
    Vec<ConversionWarning>,
    llm_api::Result<Vec<StreamEvent>>,
) {
    let mut parser = OpenAiResponses.stream_parser();
    let mut events = Vec::new();
    let mut warnings = Vec::new();
    for frame in frames {
        let (evs, ws) = parser
            .parse(&SseEvent::new(frame["type"].as_str(), frame.to_string()))
            .expect("stream event parses");
        events.extend(evs);
        warnings.extend(ws);
    }
    let finish = parser.finish().map(|(evs, ws)| {
        warnings.extend(ws);
        evs
    });
    (events, warnings, finish)
}

#[test]
fn text_stream_with_annotation() {
    let (events, warnings, finish) = run_stream("stream_text_annotation.sse");
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(finish.unwrap().is_empty());

    let annotation = json!({
        "type": "url_citation", "url": "https://example.com", "title": "Example",
        "start_index": 0, "end_index": 5,
    });
    assert_eq!(events.len(), 8, "{events:#?}");
    assert_eq!(
        ev(&events[0]),
        json!({"type": "message_start", "id": "resp_s1", "model": "gpt-5.1"})
    );
    // `response.in_progress` and the `.done` duplicates produce nothing.
    assert_eq!(
        ev(&events[1]),
        json!({
            "type": "block_start", "index": 0,
            "block": {"type": "text", "text": "", "extra": {F: {"id": "msg_s1"}}},
        })
    );
    assert_eq!(
        ev(&events[2]),
        json!({"type": "block_delta", "index": 0, "delta": {"type": "text", "value": "Hello"}})
    );
    assert_eq!(
        ev(&events[3]),
        json!({"type": "block_delta", "index": 0, "delta": {"type": "text", "value": " world"}})
    );
    // The annotation surfaces as an unmodeled delta …
    let StreamEvent::BlockDelta {
        index: 0,
        delta: BlockDelta::Other(payload),
        ..
    } = &events[4]
    else {
        panic!("expected Other delta, got {:?}", events[4]);
    };
    assert_eq!(payload["annotation"], annotation);
    // … and is folded into the finalized block at BlockStop.
    assert_eq!(
        ev(&events[5]),
        json!({
            "type": "block_stop", "index": 0,
            "block": {
                "type": "text", "text": "Hello world",
                "extra": {F: {"annotations": [annotation], "id": "msg_s1", "status": "completed"}},
            },
        })
    );
    let StreamEvent::MessageDelta {
        stop_reason, usage, ..
    } = &events[6]
    else {
        panic!("expected MessageDelta, got {:?}", events[6]);
    };
    assert_eq!(*stop_reason, Some(StopReason::EndTurn));
    let usage = usage.as_ref().unwrap();
    assert_eq!(usage.input_tokens, 10);
    assert_eq!(usage.output_tokens, 2);
    assert_eq!(usage.cache_read_tokens, Some(4));
    assert_eq!(usage.total_tokens, Some(12));
    assert_eq!(events[7], StreamEvent::MessageStop);

    let resp = accumulate(&events);
    assert_eq!(resp.id.as_deref(), Some("resp_s1"));
    assert_eq!(resp.text(), "Hello world");
    assert_eq!(resp.stop_reason, Some(StopReason::EndTurn));
    assert!(resp.raw.is_none());
}

#[test]
fn reasoning_and_tool_call_stream() {
    let (events, warnings, finish) = run_stream("stream_tool_reasoning.sse");
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(finish.is_ok());

    // Thinking block: partial at item.added, summary parts joined with a
    // blank line, finalized from the final item.
    assert_eq!(
        ev(&events[1]),
        json!({
            "type": "block_start", "index": 0,
            "block": {"type": "thinking", "extra": {F: {"id": "rs_s2"}}},
        })
    );
    for (i, text) in [
        (2, "Need the weather."),
        (3, "\n\n"),
        (4, "Calling the tool."),
    ] {
        assert_eq!(
            ev(&events[i]),
            json!({"type": "block_delta", "index": 0, "delta": {"type": "thinking", "value": text}}),
            "event {i}"
        );
    }
    let summary = json!([
        {"type": "summary_text", "text": "Need the weather."},
        {"type": "summary_text", "text": "Calling the tool."},
    ]);
    assert_eq!(
        ev(&events[5]),
        json!({
            "type": "block_stop", "index": 0,
            "block": {
                "type": "thinking",
                "text": "Need the weather.\n\nCalling the tool.",
                "signature": "enc_s2",
                "extra": {F: {"id": "rs_s2", "status": "completed", "summary": summary}},
            },
        })
    );
    // Tool call block with argument fragments.
    assert_eq!(
        ev(&events[6]),
        json!({
            "type": "block_start", "index": 1,
            "block": {
                "type": "tool_call", "id": "call_s2", "name": "get_weather", "arguments": "",
                "extra": {F: {"id": "fc_s2"}},
            },
        })
    );
    assert_eq!(
        ev(&events[7]),
        json!({"type": "block_delta", "index": 1, "delta": {"type": "tool_arguments", "value": "{\"city\":"}})
    );
    assert_eq!(
        ev(&events[8]),
        json!({"type": "block_delta", "index": 1, "delta": {"type": "tool_arguments", "value": "\"Paris\"}"}})
    );
    assert_eq!(
        ev(&events[9]),
        json!({
            "type": "block_stop", "index": 1,
            "block": {
                "type": "tool_call", "id": "call_s2", "name": "get_weather",
                "arguments": "{\"city\":\"Paris\"}",
                "extra": {F: {"id": "fc_s2", "status": "completed"}},
            },
        })
    );
    let StreamEvent::MessageDelta {
        stop_reason, usage, ..
    } = &events[10]
    else {
        panic!("expected MessageDelta, got {:?}", events[10]);
    };
    assert_eq!(*stop_reason, Some(StopReason::ToolUse));
    assert_eq!(usage.as_ref().unwrap().reasoning_tokens, Some(12));
    assert_eq!(events[11], StreamEvent::MessageStop);
    assert_eq!(events.len(), 12);

    let resp = accumulate(&events);
    assert_eq!(resp.stop_reason, Some(StopReason::ToolUse));
    let ContentBlock::Thinking {
        text, signature, ..
    } = &resp.message.content[0]
    else {
        panic!("expected thinking block");
    };
    assert_eq!(
        text.as_deref(),
        Some("Need the weather.\n\nCalling the tool.")
    );
    assert_eq!(signature.as_deref(), Some("enc_s2"));
    let ContentBlock::ToolCall { arguments, .. } = &resp.message.content[1] else {
        panic!("expected tool call block");
    };
    assert_eq!(arguments, "{\"city\":\"Paris\"}");
}

#[test]
fn refusal_stream_accumulates_to_refusal_stop() {
    let (events, warnings, finish) = run_stream("stream_refusal.sse");
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(finish.is_ok());

    // The refusal part opens a refusal-marked text block; refusal deltas
    // are text deltas.
    let StreamEvent::BlockStart { block, .. } = &events[1] else {
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
    // MessageDelta said EndTurn; the refusal-marked block normalizes it.
    assert_eq!(resp.stop_reason, Some(StopReason::Refusal));
    assert_eq!(resp.text(), "I cannot help with that.");
}

#[test]
fn incomplete_stream_finalizes_from_terminal_snapshot_and_maps_reason() {
    let (events, warnings, finish) = run_stream("stream_incomplete.sse");
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(finish.is_ok());

    // No output_item.done was seen, but the terminal event carries the
    // final Response: the open block finalizes from its `output`
    // snapshot exactly as `output_item.done` would have.
    assert_eq!(
        ev(&events[3]),
        json!({
            "type": "block_stop", "index": 0,
            "block": {
                "type": "text", "text": "Once upon a",
                "extra": {F: {"id": "msg_s4", "status": "incomplete"}},
            },
        })
    );
    let StreamEvent::MessageDelta {
        stop_reason, usage, ..
    } = &events[4]
    else {
        panic!("expected MessageDelta, got {:?}", events[4]);
    };
    assert_eq!(*stop_reason, Some(StopReason::MaxTokens));
    assert!(usage.is_some());
    assert_eq!(events[5], StreamEvent::MessageStop);

    let resp = accumulate(&events);
    assert_eq!(resp.stop_reason, Some(StopReason::MaxTokens));
    assert_eq!(resp.text(), "Once upon a");
}

#[test]
fn terminal_snapshot_finalizes_unclosed_tool_item() {
    // A built-in tool item announced as an in_progress skeleton and
    // never `done`: the terminal snapshot's completed version (with the
    // final `action`) finalizes the block.
    let (events, warnings, finish) = feed_frames(&[
        json!({"type": "response.created", "response":
               {"id": "resp_t1", "model": "gpt-5.1", "status": "in_progress", "output": []}}),
        json!({"type": "response.output_item.added", "output_index": 0,
               "item": {"id": "ws_1", "type": "web_search_call", "status": "in_progress"}}),
        json!({"type": "response.completed", "response":
               {"id": "resp_t1", "status": "completed",
                "output": [{"id": "ws_1", "type": "web_search_call", "status": "completed",
                            "action": {"type": "search", "query": "weather paris"}}],
                "usage": {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12}}}),
    ]);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(finish.is_ok());
    assert_eq!(
        ev(&events[2]),
        json!({
            "type": "block_stop", "index": 0,
            "block": {"type": "opaque", "format": F, "value":
                      {"id": "ws_1", "type": "web_search_call", "status": "completed",
                       "action": {"type": "search", "query": "weather paris"}}},
        })
    );

    let resp = accumulate(&events);
    let ContentBlock::Opaque { value, .. } = &resp.message.content[0] else {
        panic!("expected opaque block, got {:?}", resp.message.content);
    };
    assert_eq!(value["status"], json!("completed"));
    assert_eq!(value["action"]["query"], json!("weather paris"));
    assert_eq!(resp.stop_reason, Some(StopReason::EndTurn));
    assert_eq!(resp.usage.as_ref().unwrap().output_tokens, 7);
}

#[test]
fn terminal_snapshot_synthesizes_unannounced_items() {
    // An item only the terminal snapshot ever carried synthesizes
    // start/stop pairs (mirroring output_item.done without .added).
    let (events, warnings, finish) = feed_frames(&[
        json!({"type": "response.created", "response":
               {"id": "resp_t2", "model": "gpt-5.1", "status": "in_progress", "output": []}}),
        json!({"type": "response.completed", "response":
               {"id": "resp_t2", "status": "completed",
                "output": [{"id": "msg_u1", "type": "message", "role": "assistant",
                            "status": "completed", "content":
                            [{"type": "output_text", "text": "surprise", "annotations": []}]}],
                "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}}}),
    ]);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(finish.is_ok());
    assert!(matches!(
        &events[1],
        StreamEvent::BlockStart { index: 0, block: ContentBlock::Text { text, .. }, .. }
            if text == "surprise"
    ));
    assert!(matches!(
        &events[2],
        StreamEvent::BlockStop {
            index: 0,
            block: Some(_),
            ..
        }
    ));

    let resp = accumulate(&events);
    assert_eq!(resp.text(), "surprise");
    let extra = resp.message.content[0].extra().unwrap().get(F).unwrap();
    assert_eq!(extra.get("id"), Some(&json!("msg_u1")));
    assert_eq!(resp.stop_reason, Some(StopReason::EndTurn));
}

#[test]
fn post_terminal_frames_are_gated() {
    let mut parser = OpenAiResponses.stream_parser();
    for frame in [
        json!({"type": "response.created", "response":
               {"id": "resp_t3", "model": "gpt-5.1", "status": "in_progress", "output": []}}),
        json!({"type": "response.completed", "response":
               {"id": "resp_t3", "status": "completed", "output": [],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}}),
    ] {
        parser
            .parse(&SseEvent::new(frame["type"].as_str(), frame.to_string()))
            .expect("stream event parses");
    }
    // Every post-terminal frame — including `response.failed` / `error`,
    // which must not turn the complete response into an `Err` — surfaces
    // as Unknown with a warning.
    for frame in [
        json!({"type": "response.output_text.delta", "item_id": "msg_x",
               "output_index": 0, "content_index": 0, "delta": "late"}),
        json!({"type": "response.failed", "response":
               {"error": {"code": "server_error", "message": "too late"}}}),
        json!({"type": "error", "code": "ERR", "message": "late error"}),
    ] {
        let (events, warnings) = parser
            .parse(&SseEvent::new(frame["type"].as_str(), frame.to_string()))
            .expect("post-terminal frames are not errors");
        assert_eq!(events, vec![StreamEvent::Unknown]);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert_eq!(warnings[0].code, WarningCode::UnknownStreamEvent);
        assert_eq!(
            warnings[0].message,
            "chunk received after the stream terminated"
        );
    }
    assert!(parser.finish().is_ok());
}

#[test]
fn unknown_message_part_wraps_into_item_shell_and_replays() {
    // An unmodeled content part becomes an Opaque wrapping the part in a
    // `message` item shell that records the item identity under the
    // internal marker; replaying the collected message re-emits a valid
    // item with the identity restored and the marker stripped.
    let part = json!({"type": "output_audio", "audio": "QUJD", "format": "wav"});
    let item = json!({"id": "msg_1", "type": "message", "role": "assistant",
                      "status": "completed", "content": [part.clone()]});
    let (events, warnings, finish) = feed_frames(&[
        json!({"type": "response.created", "response":
               {"id": "resp_w1", "model": "gpt-5.1", "status": "in_progress", "output": []}}),
        json!({"type": "response.output_item.added", "output_index": 0,
               "item": {"id": "msg_1", "type": "message", "role": "assistant",
                        "status": "in_progress", "content": []}}),
        json!({"type": "response.content_part.added", "item_id": "msg_1",
               "output_index": 0, "content_index": 0, "part": part.clone()}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": item.clone()}),
        json!({"type": "response.completed", "response":
               {"id": "resp_w1", "status": "completed", "output": [item],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}}),
    ]);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(finish.is_ok());

    let mut shell = json!({"type": "message", "role": "assistant", "content": [part.clone()]});
    shell[llm_api::formats::openai_responses::OPAQUE_SHELL_MARKER] =
        json!({"id": "msg_1", "status": "completed"});
    let resp = accumulate(&events);
    let ContentBlock::Opaque { value, .. } = &resp.message.content[0] else {
        panic!("expected opaque block, got {:?}", resp.message.content);
    };
    assert_eq!(*value, shell);

    let req = Request::with_messages(vec![resp.message.clone()]);
    let (body, build_warnings) = request_from_ir(
        &req,
        Some("gpt-5.1"),
        CallMode::Unary,
        &ConvertOptions::default(),
    )
    .unwrap();
    assert!(build_warnings.is_empty(), "{build_warnings:?}");
    // Identity restored onto the item; the internal marker never
    // reaches the wire.
    assert_eq!(
        body["input"],
        json!([{"type": "message", "role": "assistant", "id": "msg_1",
                "status": "completed", "content": [part]}])
    );
}

#[test]
fn shell_between_same_item_texts_reinlines_into_one_item() {
    // Text + unknown part + Text of one wire item: the shell's recorded
    // identity matches the neighbouring blocks, so the parts re-inline
    // and the original item comes back whole — no split, no duplicate
    // ids, no warnings.
    let done_item = json!({"id": "msg_1", "type": "message", "role": "assistant",
    "status": "completed", "content": [
        {"type": "output_text", "text": "a", "annotations": []},
        {"type": "output_audio", "audio": "QUJD"},
        {"type": "output_text", "text": "b", "annotations": []},
    ]});
    let (events, warnings, finish) = feed_frames(&[
        json!({"type": "response.created", "response":
               {"id": "resp_i1", "model": "gpt-5.1", "status": "in_progress", "output": []}}),
        json!({"type": "response.output_item.added", "output_index": 0,
               "item": {"id": "msg_1", "type": "message", "role": "assistant",
                        "status": "in_progress", "content": []}}),
        json!({"type": "response.content_part.added", "item_id": "msg_1", "output_index": 0,
               "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}}),
        json!({"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0,
               "content_index": 0, "delta": "a"}),
        json!({"type": "response.content_part.added", "item_id": "msg_1", "output_index": 0,
               "content_index": 1, "part": {"type": "output_audio", "audio": "QUJD"}}),
        json!({"type": "response.content_part.added", "item_id": "msg_1", "output_index": 0,
               "content_index": 2, "part": {"type": "output_text", "text": "", "annotations": []}}),
        json!({"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0,
               "content_index": 2, "delta": "b"}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": done_item.clone()}),
        json!({"type": "response.completed", "response":
               {"id": "resp_i1", "status": "completed", "output": [done_item],
                "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}}}),
    ]);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(finish.is_ok());

    let resp = accumulate(&events);
    assert_eq!(resp.message.content.len(), 3);

    let req = Request::with_messages(vec![resp.message.clone()]);
    let (body, build_warnings) = request_from_ir(
        &req,
        Some("gpt-5.1"),
        CallMode::Unary,
        &ConvertOptions::default(),
    )
    .unwrap();
    assert!(build_warnings.is_empty(), "{build_warnings:?}");
    assert_eq!(
        body["input"],
        json!([{
            "type": "message", "role": "assistant", "id": "msg_1", "status": "completed",
            "content": [
                {"type": "output_text", "text": "a", "annotations": []},
                {"type": "output_audio", "audio": "QUJD"},
                {"type": "output_text", "text": "b", "annotations": []},
            ],
        }])
    );
}

#[test]
fn id_less_stream_shell_inlines_with_id_less_texts() {
    // Defensive id-less stream (no item id, no output_item.done): the
    // shell records an empty identity and the Text blocks carry none —
    // both group keys are empty, so the parts still form one item.
    let (events, warnings, finish) = feed_frames(&[
        json!({"type": "response.created", "response":
               {"id": "resp_n1", "model": "gpt-5.1", "status": "in_progress", "output": []}}),
        json!({"type": "response.output_item.added", "output_index": 0,
               "item": {"type": "message", "role": "assistant", "content": []}}),
        json!({"type": "response.content_part.added", "output_index": 0,
               "content_index": 0, "part": {"type": "output_audio", "audio": "QUJD"}}),
        json!({"type": "response.content_part.added", "output_index": 0,
               "content_index": 1, "part": {"type": "output_text", "text": "", "annotations": []}}),
        json!({"type": "response.output_text.delta", "output_index": 0,
               "content_index": 1, "delta": "x"}),
        json!({"type": "response.completed", "response":
               {"id": "resp_n1", "status": "completed", "output": [],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}}),
    ]);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(finish.is_ok());

    let resp = accumulate(&events);
    assert_eq!(resp.message.content.len(), 2);

    let req = Request::with_messages(vec![resp.message.clone()]);
    let (body, build_warnings) = request_from_ir(
        &req,
        Some("gpt-5.1"),
        CallMode::Unary,
        &ConvertOptions::default(),
    )
    .unwrap();
    assert!(build_warnings.is_empty(), "{build_warnings:?}");
    assert_eq!(
        body["input"],
        json!([{
            "type": "message", "role": "assistant",
            "content": [
                {"type": "output_audio", "audio": "QUJD"},
                {"type": "output_text", "text": "x", "annotations": []},
            ],
        }])
    );
}

#[test]
fn terminal_out_of_order_synthesis_warns_block_order_lost() {
    // A snapshot-only item placed before already-streamed content can
    // only be appended after it (block indices are final): one
    // BlockOrderLost warning per terminal.
    let done_item = json!({"id": "msg_b", "type": "message", "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "streamed", "annotations": []}]});
    let missed_item = json!({"id": "msg_a", "type": "message", "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "first", "annotations": []}]});
    let (events, warnings, finish) = feed_frames(&[
        json!({"type": "response.created", "response":
               {"id": "resp_o1", "model": "gpt-5.1", "status": "in_progress", "output": []}}),
        json!({"type": "response.output_item.added", "output_index": 1,
               "item": {"id": "msg_b", "type": "message", "role": "assistant",
                        "status": "in_progress", "content": []}}),
        json!({"type": "response.output_item.done", "output_index": 1, "item": done_item.clone()}),
        json!({"type": "response.completed", "response":
               {"id": "resp_o1", "status": "completed",
                "output": [missed_item, done_item],
                "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}}}),
    ]);
    assert!(finish.is_ok());
    let order: Vec<_> = warnings
        .iter()
        .filter(|w| w.code == WarningCode::BlockOrderLost)
        .collect();
    assert_eq!(order.len(), 1, "{warnings:?}");
    assert_eq!(order[0].location, "/output/0");

    // The synthesized item's text follows the streamed one.
    let resp = accumulate(&events);
    assert_eq!(resp.text(), "streamedfirst");
}

#[test]
fn streamed_multi_part_item_rebuilds_as_one_item() {
    // End-to-end: the parts of one streamed item carry identical item
    // metadata, so the extended grouping key must not split them on
    // re-serialization.
    let item = json!({"id": "msg_m1", "type": "message", "role": "assistant",
    "status": "completed", "content": [
        {"type": "output_text", "text": "Hello ", "annotations": []},
        {"type": "output_text", "text": "world", "annotations": []},
    ]});
    let (events, warnings, finish) = feed_frames(&[
        json!({"type": "response.created", "response":
               {"id": "resp_m1", "model": "gpt-5.1", "status": "in_progress", "output": []}}),
        json!({"type": "response.output_item.added", "output_index": 0,
               "item": {"id": "msg_m1", "type": "message", "role": "assistant",
                        "status": "in_progress", "content": []}}),
        json!({"type": "response.content_part.added", "item_id": "msg_m1",
               "output_index": 0, "content_index": 0,
               "part": {"type": "output_text", "text": "", "annotations": []}}),
        json!({"type": "response.output_text.delta", "item_id": "msg_m1",
               "output_index": 0, "content_index": 0, "delta": "Hello "}),
        json!({"type": "response.content_part.added", "item_id": "msg_m1",
               "output_index": 0, "content_index": 1,
               "part": {"type": "output_text", "text": "", "annotations": []}}),
        json!({"type": "response.output_text.delta", "item_id": "msg_m1",
               "output_index": 0, "content_index": 1, "delta": "world"}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": item.clone()}),
        json!({"type": "response.completed", "response":
               {"id": "resp_m1", "status": "completed", "output": [item],
                "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}}}),
    ]);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(finish.is_ok());

    let resp = accumulate(&events);
    assert_eq!(resp.message.content.len(), 2);

    let req = Request::with_messages(vec![resp.message.clone()]);
    let (body, build_warnings) = request_from_ir(
        &req,
        Some("gpt-5.1"),
        CallMode::Unary,
        &ConvertOptions::default(),
    )
    .unwrap();
    assert!(build_warnings.is_empty(), "{build_warnings:?}");
    assert_eq!(
        body["input"],
        json!([{
            "type": "message", "role": "assistant", "id": "msg_m1", "status": "completed",
            "content": [
                {"type": "output_text", "text": "Hello ", "annotations": []},
                {"type": "output_text", "text": "world", "annotations": []},
            ],
        }])
    );
}

#[test]
fn failed_stream_surfaces_api_error() {
    let mut parser = OpenAiResponses.stream_parser();
    let mut failed = None;
    for event in sse_events(&fixture("stream_failed.sse")) {
        match parser.parse(&event) {
            Ok(_) => {}
            Err(err) => {
                failed = Some(err);
                break;
            }
        }
    }
    match failed.expect("response.failed must error") {
        Error::Api {
            status,
            kind,
            message,
            ..
        } => {
            assert_eq!(status, 200);
            assert_eq!(kind, llm_api::ApiErrorKind::ServerError);
            assert_eq!(message, "The model had a bad day.");
        }
        other => panic!("expected Error::Api, got {other:?}"),
    }
}

#[test]
fn error_event_surfaces_api_error() {
    let mut parser = OpenAiResponses.stream_parser();
    let event = SseEvent::new(
        Some("error"),
        r#"{"type":"error","code":"ERR_SOMETHING","message":"Something went wrong","param":null,"sequence_number":1}"#,
    );
    let err = parser.parse(&event).unwrap_err();
    match err {
        Error::Api { kind, message, .. } => {
            assert_eq!(kind, llm_api::ApiErrorKind::Other("ERR_SOMETHING".into()));
            assert_eq!(message, "Something went wrong");
        }
        other => panic!("expected Error::Api, got {other:?}"),
    }
}

#[test]
fn truncated_stream_fails_at_finish() {
    let (events, warnings, finish) = run_stream("stream_truncated.sse");
    assert!(warnings.is_empty());
    match finish {
        Err(Error::TruncatedStream { message, .. }) => {
            assert!(message.contains("without a terminal event"));
        }
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
fn unknown_events_dedupe_and_opaque_items_take_other_deltas() {
    let (events, warnings, finish) = run_stream("stream_opaque_unknown.sse");
    assert!(finish.is_ok());

    // Two occurrences of the same unknown event: two Unknown events, one
    // warning.
    let unknown_count = events
        .iter()
        .filter(|e| matches!(e, StreamEvent::Unknown))
        .count();
    assert_eq!(unknown_count, 2);
    let unknown_warnings: Vec<_> = warnings
        .iter()
        .filter(|w| w.code == WarningCode::UnknownStreamEvent)
        .collect();
    assert_eq!(unknown_warnings.len(), 1, "{warnings:?}");

    // The web_search_call item is an Opaque block; its status event is an
    // attributed Other delta, not Unknown.
    let StreamEvent::BlockStart {
        index: 0,
        block: ContentBlock::Opaque { format, value, .. },
        ..
    } = &events[3]
    else {
        panic!("expected opaque BlockStart, got {:?}", events[3]);
    };
    assert_eq!(format, F);
    assert_eq!(value["type"], json!("web_search_call"));
    let StreamEvent::BlockDelta {
        index: 0,
        delta: BlockDelta::Other(payload),
        ..
    } = &events[4]
    else {
        panic!("expected Other delta, got {:?}", events[4]);
    };
    assert_eq!(payload["type"], json!("response.web_search_call.searching"));
    let StreamEvent::BlockStop {
        index: 0,
        block: Some(ContentBlock::Opaque { value, .. }),
        ..
    } = &events[5]
    else {
        panic!("expected opaque BlockStop, got {:?}", events[5]);
    };
    assert_eq!(value["status"], json!("completed"));

    let resp = accumulate(&events);
    assert_eq!(resp.message.content.len(), 2);
    assert!(matches!(
        &resp.message.content[0],
        ContentBlock::Opaque { .. }
    ));
    assert_eq!(resp.text(), "Cats are cats.");
    assert_eq!(resp.stop_reason, Some(StopReason::EndTurn));
}

#[test]
fn malformed_event_data_is_unknown_with_warning() {
    let mut parser = OpenAiResponses.stream_parser();
    let (events, warnings) = parser
        .parse(&SseEvent::new(
            Some("response.output_text.delta"),
            "{not json",
        ))
        .unwrap();
    assert_eq!(events, vec![StreamEvent::Unknown]);
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0].code, WarningCode::MalformedField);
}

#[test]
fn payload_type_wins_over_event_name() {
    let mut parser = OpenAiResponses.stream_parser();
    let data = json!({
        "type": "response.created",
        "response": {"id": "resp_x", "model": "gpt-5.1", "output": [], "usage": null},
    });
    let (events, warnings) = parser
        .parse(&SseEvent::new(Some("mislabeled"), data.to_string()))
        .unwrap();
    assert!(warnings.is_empty());
    assert_eq!(events.len(), 1);
    assert_eq!(
        ev(&events[0]),
        json!({"type": "message_start", "id": "resp_x", "model": "gpt-5.1"})
    );
}

#[test]
fn synthesized_blocks_for_item_done_without_added() {
    // Defensive path: an output_item.done for an item the stream never
    // announced synthesizes BlockStart + BlockStop pairs.
    let mut parser = OpenAiResponses.stream_parser();
    let done = json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "id": "msg_x", "type": "message", "status": "completed", "role": "assistant",
            "content": [{"type": "output_text", "text": "surprise", "annotations": []}],
        },
    });
    let (events, _) = parser
        .parse(&SseEvent::new(None, done.to_string()))
        .unwrap();
    // A defensive MessageStart precedes the block events.
    assert!(matches!(events[0], StreamEvent::MessageStart { .. }));
    assert!(matches!(
        &events[1],
        StreamEvent::BlockStart { index: 0, block: ContentBlock::Text { text, .. }, .. } if text == "surprise"
    ));
    assert!(matches!(
        &events[2],
        StreamEvent::BlockStop {
            index: 0,
            block: Some(_),
            ..
        }
    ));
    let value: Value =
        json!({"type": "response.completed", "response": {"output": [], "usage": null}});
    let (events, _) = parser
        .parse(&SseEvent::new(None, value.to_string()))
        .unwrap();
    assert!(events.contains(&StreamEvent::MessageStop));
    assert!(parser.finish().is_ok());
}

#[test]
fn reasoning_text_streams_as_thinking_deltas() {
    // `response.reasoning_text.delta` is official raw chain of thought
    // (`content` reasoning_text parts) and must stream as `Thinking`
    // deltas, with the `"\n\n"` joiner between content parts — not as
    // unmodeled `Other` payloads or unknown events.
    let (events, warnings, finish) = run_stream("stream_reasoning_text.sse");
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(finish.unwrap().is_empty());

    let thinking: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::BlockDelta {
                delta: BlockDelta::Thinking(t),
                ..
            } => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        thinking,
        vec![
            "9.11 vs 9.8: compare ",
            "the tenths digit.",
            "\n\n",
            "8 > 1, so 9.8 wins."
        ]
    );

    let resp = accumulate(&events);
    let ContentBlock::Thinking { text, extra, .. } = &resp.message.content[0] else {
        panic!("expected a thinking block: {:?}", resp.message.content);
    };
    // The incrementally built text matches the finalized block's parse.
    assert_eq!(
        text.as_deref(),
        Some("9.11 vs 9.8: compare the tenths digit.\n\n8 > 1, so 9.8 wins.")
    );
    // The finalized block keeps the `content` array for reconstruction.
    let ns = extra.get(F).expect("namespace stored");
    assert_eq!(
        ns.get("content").and_then(Value::as_array).map(Vec::len),
        Some(2)
    );

    assert_eq!(resp.text(), "9.8 is greater.");
    assert_eq!(resp.stop_reason, Some(StopReason::EndTurn));
    assert_eq!(resp.usage.as_ref().unwrap().reasoning_tokens, Some(18));
}
