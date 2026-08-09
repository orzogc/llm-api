//! Streaming tests for the `openai_responses` format: complete SSE
//! sessions from fixtures, block-index flattening, terminal handling,
//! truncation and accumulation.
//!
//! Expected events are asserted through their IR JSON representation
//! (`StreamEvent` variants are `#[non_exhaustive]` and cannot be
//! constructed outside the crate).

use serde_json::{Value, json};

use llm_api::formats::openai_responses::OpenAiResponses;
use llm_api::http::{SseEvent, SseParser};
use llm_api::{
    Accumulator, ApiFormat, BlockDelta, ContentBlock, ConversionWarning, Error, StopReason,
    StreamEvent, StreamItem, WarningCode,
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
        block.extra().unwrap().get(F).unwrap().get("refusal"),
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
fn incomplete_stream_closes_open_blocks_and_maps_reason() {
    let (events, warnings, finish) = run_stream("stream_incomplete.sse");
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(finish.is_ok());

    // No output_item.done was seen: the terminal event closes the block
    // with `block: None` (the accumulated content stands).
    assert_eq!(ev(&events[3]), json!({"type": "block_stop", "index": 0}));
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
