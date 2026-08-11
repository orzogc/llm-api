//! Streaming tests for `anthropic_messages`: SSE fixtures through the
//! stream parser and the accumulator.

use llm_api::formats::anthropic_messages::AnthropicMessages;
use llm_api::http::{SseEvent, SseParser};
use llm_api::{
    Accumulator, ApiErrorKind, ApiFormat, BlockDelta, BuildCtx, CallMode, ContentBlock,
    ConversionWarning, EndpointUrl, Error, Message, Request, Response, StopReason, StreamEvent,
    StreamItem, WarningCode,
};
use serde_json::{Value, json};

const FMT: &str = "anthropic_messages";

fn fixture(name: &str) -> Vec<u8> {
    let path = format!(
        "{}/tests/fixtures/anthropic_messages/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"))
}

/// Runs a complete SSE fixture through the stream parser.
fn feed(name: &str) -> Result<(Vec<StreamEvent>, Vec<ConversionWarning>), Error> {
    let bytes = fixture(name);
    let mut sse = SseParser::new(usize::MAX);
    let mut parser = AnthropicMessages.stream_parser();
    let mut sse_events = sse.push(&bytes)?;
    sse_events.extend(sse.finish()?);
    let mut events = Vec::new();
    let mut warnings = Vec::new();
    for ev in &sse_events {
        let (evs, ws) = parser.parse(ev)?;
        events.extend(evs);
        warnings.extend(ws);
    }
    let (evs, ws) = parser.finish()?;
    events.extend(evs);
    warnings.extend(ws);
    Ok((events, warnings))
}

fn accumulate(events: &[StreamEvent]) -> Response {
    let mut acc = Accumulator::new();
    for e in events {
        acc.push(&StreamItem::new(e.clone())).unwrap();
    }
    acc.finish().unwrap()
}

#[test]
fn basic_text_stream() {
    let (events, warnings) = feed("stream_basic.sse").unwrap();
    assert!(warnings.is_empty());
    // Ping is consumed silently; the unified sequence is complete.
    assert!(matches!(events[0], StreamEvent::MessageStart { .. }));
    assert!(matches!(events.last(), Some(StreamEvent::MessageStop)));
    match &events[0] {
        StreamEvent::MessageStart {
            id, model, usage, ..
        } => {
            assert_eq!(id.as_deref(), Some("msg_1nZd"));
            assert_eq!(model.as_deref(), Some("claude-sonnet-5"));
            // Input side unified: 25 uncached + 3 write + 2 read.
            let u = usage.as_ref().unwrap();
            assert_eq!(u.input_tokens, 30);
            assert_eq!(u.cache_write_tokens, Some(3));
            assert_eq!(u.cache_read_tokens, Some(2));
        }
        other => panic!("unexpected: {other:?}"),
    }
    // message_delta usage is merged with the cached input-side counts into
    // a complete cumulative snapshot.
    let delta_usage = events
        .iter()
        .find_map(|e| match e {
            StreamEvent::MessageDelta { usage: Some(u), .. } => Some(u.clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(delta_usage.input_tokens, 30);
    assert_eq!(delta_usage.output_tokens, 15);

    let resp = accumulate(&events);
    assert_eq!(resp.text(), "Hello!");
    assert_eq!(resp.stop_reason, Some(StopReason::EndTurn));
    assert_eq!(resp.usage.as_ref().unwrap().output_tokens, 15);
    assert!(resp.raw.is_none());
}

#[test]
fn tool_use_stream_accumulates_fragments() {
    let (events, warnings) = feed("stream_tool_use.sse").unwrap();
    assert!(warnings.is_empty());
    // The tool block opens with empty arguments.
    let start = events
        .iter()
        .find_map(|e| match e {
            StreamEvent::BlockStart {
                index: 1, block, ..
            } => Some(block.clone()),
            _ => None,
        })
        .unwrap();
    assert!(matches!(
        &start,
        ContentBlock::ToolCall { arguments, .. } if arguments.is_empty()
    ));
    // Fragments surface verbatim, including the empty one.
    let fragments: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::BlockDelta {
                index: 1,
                delta: BlockDelta::ToolArguments(s),
                ..
            } => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        fragments,
        vec!["", "{\"location\":", " \"San Francisco,", " CA\"}"]
    );

    let resp = accumulate(&events);
    match &resp.message.content[1] {
        ContentBlock::ToolCall {
            id,
            name,
            arguments,
            ..
        } => {
            assert_eq!(id.as_deref(), Some("toolu_01T1"));
            assert_eq!(name, "get_weather");
            assert_eq!(arguments, "{\"location\": \"San Francisco, CA\"}");
        }
        other => panic!("unexpected: {other:?}"),
    }
    // A tool block with no fragments falls back to its opening input.
    match &resp.message.content[2] {
        ContentBlock::ToolCall {
            name, arguments, ..
        } => {
            assert_eq!(name, "noop");
            assert_eq!(arguments, "{}");
        }
        other => panic!("unexpected: {other:?}"),
    }
    assert_eq!(resp.stop_reason, Some(StopReason::ToolUse));
}

#[test]
fn tool_use_non_object_start_input_warns() {
    // `input` on content_block_start arrives whole (not incrementally); a
    // non-object value warns MalformedToolCall at the start event — once —
    // and the finalized call keeps the value verbatim.
    let mut parser = AnthropicMessages.stream_parser();
    let (_, warnings) = parser
        .parse(&SseEvent::new(
            Some("content_block_start"),
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tu_1","name":"f","input":5}}"#,
        ))
        .unwrap();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].code, WarningCode::MalformedToolCall);
    assert_eq!(warnings[0].location, "/content_block_start/0/input");

    let (events, warnings) = parser
        .parse(&SseEvent::new(
            Some("content_block_stop"),
            r#"{"type":"content_block_stop","index":0}"#,
        ))
        .unwrap();
    assert!(
        warnings.is_empty(),
        "no second warning at stop: {warnings:?}"
    );
    match &events[0] {
        StreamEvent::BlockStop {
            block: Some(ContentBlock::ToolCall { arguments, .. }),
            ..
        } => assert_eq!(arguments, "5"),
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn invalid_accumulated_tool_json_stays_silent() {
    // § 4.5 boundary: invalid JSON assembled from the model's own
    // input_json_delta fragments is preserved verbatim with no warning
    // (truncation already shows in the stop reason); only a non-object
    // start `input` warns. The canonical `{}` start input is warning-free.
    let mut parser = AnthropicMessages.stream_parser();
    let (_, warnings) = parser
        .parse(&SseEvent::new(
            Some("content_block_start"),
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tu_1","name":"f","input":{}}}"#,
        ))
        .unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    let (_, warnings) = parser
        .parse(&SseEvent::new(
            Some("content_block_delta"),
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"a\": tru"}}"#,
        ))
        .unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    let (events, warnings) = parser
        .parse(&SseEvent::new(
            Some("content_block_stop"),
            r#"{"type":"content_block_stop","index":0}"#,
        ))
        .unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    match &events[0] {
        StreamEvent::BlockStop {
            block: Some(ContentBlock::ToolCall { arguments, .. }),
            ..
        } => assert_eq!(arguments, "{\"a\": tru"),
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn interleaved_thinking_stream() {
    let (events, warnings) = feed("stream_thinking_interleaved.sse").unwrap();
    assert!(warnings.is_empty());
    // Signature fragments surface as Signature deltas.
    assert!(events.iter().any(|e| matches!(
        e,
        StreamEvent::BlockDelta { index: 0, delta: BlockDelta::Signature(s), .. } if s == "SIGA"
    )));
    let resp = accumulate(&events);
    let kinds: Vec<&'static str> = resp
        .message
        .content
        .iter()
        .map(ContentBlock::kind_name)
        .collect();
    assert_eq!(kinds, vec!["Thinking", "Text", "Thinking", "Text"]);
    match &resp.message.content[0] {
        ContentBlock::Thinking {
            text, signature, ..
        } => {
            assert_eq!(text.as_deref(), Some("Let me think harder."));
            assert_eq!(signature.as_deref(), Some("SIGA"));
        }
        other => panic!("unexpected: {other:?}"),
    }
    match &resp.message.content[2] {
        ContentBlock::Thinking {
            text,
            signature,
            extra,
            ..
        } => {
            assert_eq!(*text, None);
            assert_eq!(signature.as_deref(), Some("RDATA"));
            assert_eq!(extra.get(FMT).unwrap().get("redacted"), Some(&json!(true)));
        }
        other => panic!("unexpected: {other:?}"),
    }
    assert_eq!(resp.text(), "Part one. Done.");
    assert_eq!(resp.usage.as_ref().unwrap().reasoning_tokens, Some(30));

    // Streamed thinking replays into a request exactly like non-streaming.
    let mut req = Request::with_messages(vec![Message::user_text("q"), resp.message.clone()]);
    req.max_output_tokens = Some(10);
    let ctx = BuildCtx::new(
        EndpointUrl::base("https://api.anthropic.com/v1").unwrap(),
        "claude-sonnet-5",
        CallMode::Unary,
    );
    let built = AnthropicMessages.build_request(&req, &ctx).unwrap();
    let body: Value = serde_json::from_slice(&built.body).unwrap();
    assert_eq!(
        body["messages"][1]["content"],
        json!([
            {"type": "thinking", "thinking": "Let me think harder.", "signature": "SIGA"},
            {"type": "text", "text": "Part one."},
            {"type": "redacted_thinking", "data": "RDATA"},
            {"type": "text", "text": " Done."},
        ])
    );
}

#[test]
fn citations_delta_surfaces_and_folds() {
    let (events, warnings) = feed("stream_citations.sse").unwrap();
    assert!(warnings.is_empty());
    // Surfaced for real-time consumers.
    assert!(events.iter().any(|e| matches!(
        e,
        StreamEvent::BlockDelta { delta: BlockDelta::Other(v), .. }
            if v["type"] == "citations_delta"
    )));
    // Folded into the finalized block at BlockStop.
    let finalized = events
        .iter()
        .find_map(|e| match e {
            StreamEvent::BlockStop {
                index: 0,
                block: Some(b),
                ..
            } => Some(b.clone()),
            _ => None,
        })
        .unwrap();
    match &finalized {
        ContentBlock::Text { text, extra, .. } => {
            assert_eq!(text, "Per the doc, yes.");
            let citations = extra.get(FMT).unwrap().get("citations").unwrap();
            assert_eq!(citations[0]["cited_text"], json!("doc body"));
        }
        other => panic!("unexpected: {other:?}"),
    }
    // The accumulator adopts the finalized block.
    let resp = accumulate(&events);
    assert_eq!(resp.message.content[0], finalized);
}

#[test]
fn server_tool_stream_stays_opaque_and_folds_input() {
    let (events, warnings) = feed("stream_server_tool.sse").unwrap();
    assert!(warnings.is_empty());
    // The unmodeled block opens as Opaque and its input fragments surface
    // as Other deltas.
    assert!(events.iter().any(|e| matches!(
        e,
        StreamEvent::BlockStart {
            index: 0,
            block: ContentBlock::Opaque { .. },
            ..
        }
    )));
    assert!(events.iter().any(|e| matches!(
        e,
        StreamEvent::BlockDelta {
            index: 0,
            delta: BlockDelta::Other(_),
            ..
        }
    )));
    let resp = accumulate(&events);
    match &resp.message.content[0] {
        ContentBlock::Opaque { format, value, .. } => {
            assert_eq!(format, FMT);
            assert_eq!(value["type"], json!("server_tool_use"));
            // Accumulated partial_json folded into the final input.
            assert_eq!(value["input"], json!({"query": "rust"}));
        }
        other => panic!("unexpected: {other:?}"),
    }
    assert!(matches!(
        &resp.message.content[1],
        ContentBlock::Opaque { value, .. } if value["type"] == "web_search_tool_result"
    ));
    let usage = resp.usage.as_ref().unwrap();
    assert_eq!(usage.input_tokens, 10682);
    assert_eq!(usage.output_tokens, 510);
    assert_eq!(
        usage.raw.as_ref().unwrap()["server_tool_use"]["web_search_requests"],
        json!(1)
    );
}

#[test]
fn known_delta_on_opaque_block_surfaces_without_failing() {
    // A known-kind delta addressed to an unmodeled block cannot be folded
    // (the block's shape is unknown; nothing is fabricated) but must not
    // kill the stream: it surfaces as Other with one warning per block.
    let mut parser = AnthropicMessages.stream_parser();
    let (_, ws) = parser
        .parse(&SseEvent::new(
            Some("content_block_start"),
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"future_block","x":1}}"#,
        ))
        .unwrap();
    assert!(ws.is_empty(), "{ws:?}");
    let (events, ws) = parser
        .parse(&SseEvent::new(
            Some("content_block_delta"),
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#,
        ))
        .unwrap();
    assert_eq!(events.len(), 1);
    assert!(matches!(
        &events[0],
        StreamEvent::BlockDelta { index: 0, delta: BlockDelta::Other(v), .. }
            if *v == json!({"type": "text_delta", "text": "hi"})
    ));
    assert_eq!(ws.len(), 1, "{ws:?}");
    assert_eq!(ws[0].code, WarningCode::MalformedField);
    assert_eq!(ws[0].location, "/delta");
    assert!(ws[0].message.contains("text_delta"), "{}", ws[0].message);
    // Further known deltas on the same block still surface, without
    // repeating the warning.
    for delta in [
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"t"}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"s"}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{}}}"#,
    ] {
        let (events, ws) = parser
            .parse(&SseEvent::new(Some("content_block_delta"), delta))
            .unwrap();
        assert!(matches!(
            &events[0],
            StreamEvent::BlockDelta {
                delta: BlockDelta::Other(_),
                ..
            }
        ));
        assert!(ws.is_empty(), "{ws:?}");
    }
    // Nothing was folded into the block.
    let (events, ws) = parser
        .parse(&SseEvent::new(
            Some("content_block_stop"),
            r#"{"type":"content_block_stop","index":0}"#,
        ))
        .unwrap();
    assert!(ws.is_empty(), "{ws:?}");
    match &events[0] {
        StreamEvent::BlockStop {
            block: Some(ContentBlock::Opaque { value, .. }),
            ..
        } => assert_eq!(*value, json!({"type": "future_block", "x": 1})),
        other => panic!("unexpected: {other:?}"),
    }

    // A known-type block whose shape failed to parse opens as Opaque too
    // (with its own warning) and benefits from the same degrade; the
    // dedup set is per block index.
    let (_, ws) = parser
        .parse(&SseEvent::new(
            Some("content_block_start"),
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"text"}}"#,
        ))
        .unwrap();
    assert_eq!(ws.len(), 1, "{ws:?}");
    assert_eq!(ws[0].location, "/content_block_start/1");
    let (events, ws) = parser
        .parse(&SseEvent::new(
            Some("content_block_delta"),
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"x"}}"#,
        ))
        .unwrap();
    assert!(matches!(
        &events[0],
        StreamEvent::BlockDelta {
            delta: BlockDelta::Other(_),
            ..
        }
    ));
    assert_eq!(ws.len(), 1, "{ws:?}");
    assert_eq!(ws[0].code, WarningCode::MalformedField);

    // The stream still terminates cleanly.
    parser
        .parse(&SseEvent::new(
            Some("content_block_stop"),
            r#"{"type":"content_block_stop","index":1}"#,
        ))
        .unwrap();
    let (events, _) = parser
        .parse(&SseEvent::new(
            Some("message_stop"),
            r#"{"type":"message_stop"}"#,
        ))
        .unwrap();
    assert_eq!(events, vec![StreamEvent::MessageStop]);
    assert!(parser.finish().is_ok());
}

#[test]
fn unmodeled_delta_members_surface_as_extra_other() {
    // Members of a recognized delta beyond its consumed keys surface as a
    // trailing Other event (transient: never folded into the block), with
    // one warning per member name per stream.
    let mut parser = AnthropicMessages.stream_parser();
    parser
        .parse(&SseEvent::new(
            Some("content_block_start"),
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}"#,
        ))
        .unwrap();
    let (events, ws) = parser
        .parse(&SseEvent::new(
            Some("content_block_delta"),
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"a","estimated_tokens":42}}"#,
        ))
        .unwrap();
    assert_eq!(events.len(), 2, "{events:?}");
    assert!(matches!(
        &events[0],
        StreamEvent::BlockDelta { index: 0, delta: BlockDelta::Thinking(s), .. } if s == "a"
    ));
    assert!(matches!(
        &events[1],
        StreamEvent::BlockDelta { index: 0, delta: BlockDelta::Other(v), .. }
            if *v == json!({"type": "thinking_delta", "estimated_tokens": 42})
    ));
    assert_eq!(ws.len(), 1, "{ws:?}");
    assert_eq!(ws[0].code, WarningCode::MalformedField);
    assert_eq!(ws[0].location, "/delta");
    assert!(
        ws[0].message.contains("estimated_tokens"),
        "{}",
        ws[0].message
    );
    // The same member on a later delta still surfaces but no longer warns.
    let (events, ws) = parser
        .parse(&SseEvent::new(
            Some("content_block_delta"),
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"b","estimated_tokens":43}}"#,
        ))
        .unwrap();
    assert_eq!(events.len(), 2);
    assert!(matches!(
        &events[1],
        StreamEvent::BlockDelta { delta: BlockDelta::Other(v), .. }
            if v["estimated_tokens"] == json!(43)
    ));
    assert!(ws.is_empty(), "{ws:?}");
    // A delta without extra members behaves exactly as before.
    let (events, ws) = parser
        .parse(&SseEvent::new(
            Some("content_block_delta"),
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"c"}}"#,
        ))
        .unwrap();
    assert_eq!(events.len(), 1, "{events:?}");
    assert!(matches!(
        &events[0],
        StreamEvent::BlockDelta { index: 0, delta: BlockDelta::Thinking(s), .. } if s == "c"
    ));
    assert!(ws.is_empty(), "{ws:?}");
    // A new member name warns once, on any recognized delta kind.
    let (events, ws) = parser
        .parse(&SseEvent::new(
            Some("content_block_delta"),
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"S","foo":1}}"#,
        ))
        .unwrap();
    assert_eq!(events.len(), 2);
    assert!(matches!(
        &events[0],
        StreamEvent::BlockDelta {
            delta: BlockDelta::Signature(s),
            ..
        } if s == "S"
    ));
    assert!(matches!(
        &events[1],
        StreamEvent::BlockDelta { delta: BlockDelta::Other(v), .. }
            if *v == json!({"type": "signature_delta", "foo": 1})
    ));
    assert_eq!(ws.len(), 1, "{ws:?}");
    assert!(ws[0].message.contains("foo"), "{}", ws[0].message);
    // The finalized block carries none of the transient members.
    let (events, ws) = parser
        .parse(&SseEvent::new(
            Some("content_block_stop"),
            r#"{"type":"content_block_stop","index":0}"#,
        ))
        .unwrap();
    assert!(ws.is_empty(), "{ws:?}");
    match &events[0] {
        StreamEvent::BlockStop {
            block:
                Some(ContentBlock::Thinking {
                    text,
                    signature,
                    extra,
                    ..
                }),
            ..
        } => {
            assert_eq!(text.as_deref(), Some("abc"));
            assert_eq!(signature.as_deref(), Some("S"));
            assert!(extra.get(FMT).is_none(), "{extra:?}");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn usage_missing_core_fields_warns_once_and_heals() {
    // A snapshot that never saw the wire-required core fields (dialect
    // stream: no usage on message_start, output-only deltas) is omitted
    // with a single per-stream warning — never zeroed — and a later
    // complete cumulative snapshot heals it.
    let mut parser = AnthropicMessages.stream_parser();
    let (_, ws) = parser
        .parse(&SseEvent::new(
            Some("message_start"),
            json!({"type": "message_start", "message": {"id": "msg_1", "model": "m"}}).to_string(),
        ))
        .unwrap();
    assert!(ws.is_empty(), "{ws:?}");

    let delta = |usage: Value| {
        SseEvent::new(
            Some("message_delta"),
            json!({"type": "message_delta", "delta": {}, "usage": usage}).to_string(),
        )
    };
    let (events, ws) = parser.parse(&delta(json!({"output_tokens": 2}))).unwrap();
    assert!(matches!(
        &events[0],
        StreamEvent::MessageDelta { usage: None, .. }
    ));
    assert_eq!(ws.len(), 1, "{ws:?}");
    assert_eq!(ws[0].code, WarningCode::MalformedField);
    assert_eq!(ws[0].location, "/usage");

    // The second incomplete snapshot stays quiet (once per stream).
    let (events, ws) = parser.parse(&delta(json!({"output_tokens": 3}))).unwrap();
    assert!(matches!(
        &events[0],
        StreamEvent::MessageDelta { usage: None, .. }
    ));
    assert!(ws.is_empty(), "{ws:?}");

    // A complete cumulative snapshot heals the overlay.
    let (events, ws) = parser
        .parse(&delta(json!({"input_tokens": 7, "output_tokens": 4})))
        .unwrap();
    let StreamEvent::MessageDelta {
        usage: Some(usage), ..
    } = &events[0]
    else {
        panic!("expected usage: {events:?}");
    };
    assert_eq!((usage.input_tokens, usage.output_tokens), (7, 4));
    assert!(ws.is_empty(), "{ws:?}");
}

#[test]
fn malformed_usage_degrades_and_heals() {
    // message_start with a usage that fails the wire shape: the stream
    // survives, the event carries no usage and the degrade is disclosed.
    let mut parser = AnthropicMessages.stream_parser();
    let (events, ws) = parser
        .parse(&SseEvent::new(
            Some("message_start"),
            json!({"type": "message_start", "message": {
                "id": "msg_1", "model": "m",
                "usage": {"input_tokens": 12.5},
            }})
            .to_string(),
        ))
        .unwrap();
    assert!(matches!(
        &events[0],
        StreamEvent::MessageStart { usage: None, .. }
    ));
    assert_eq!(ws.len(), 1, "{ws:?}");
    assert_eq!(ws[0].code, WarningCode::MalformedField);
    assert_eq!(ws[0].location, "/usage");

    // The snapshot overlay lets a later valid cumulative usage heal it.
    let (events, ws) = parser
        .parse(&SseEvent::new(
            Some("message_delta"),
            json!({"type": "message_delta", "delta": {},
                   "usage": {"input_tokens": 20, "output_tokens": 5}})
            .to_string(),
        ))
        .unwrap();
    assert!(ws.is_empty(), "{ws:?}");
    match &events[0] {
        StreamEvent::MessageDelta { usage: Some(u), .. } => {
            assert_eq!(u.input_tokens, 20);
            assert_eq!(u.output_tokens, 5);
        }
        other => panic!("unexpected: {other:?}"),
    }

    // A malformed message_delta usage degrades that event only…
    let (events, ws) = parser
        .parse(&SseEvent::new(
            Some("message_delta"),
            json!({"type": "message_delta", "delta": {},
                   "usage": {"output_tokens": "x"}})
            .to_string(),
        ))
        .unwrap();
    assert!(matches!(
        &events[0],
        StreamEvent::MessageDelta { usage: None, .. }
    ));
    assert_eq!(ws.len(), 1, "{ws:?}");
    assert_eq!(ws[0].location, "/usage");

    // …and the next valid cumulative snapshot recovers.
    let (events, ws) = parser
        .parse(&SseEvent::new(
            Some("message_delta"),
            json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                   "usage": {"output_tokens": 7}})
            .to_string(),
        ))
        .unwrap();
    assert!(ws.is_empty(), "{ws:?}");
    match &events[0] {
        StreamEvent::MessageDelta { usage: Some(u), .. } => {
            assert_eq!(u.input_tokens, 20);
            assert_eq!(u.output_tokens, 7);
        }
        other => panic!("unexpected: {other:?}"),
    }
    parser
        .parse(&SseEvent::new(
            Some("message_stop"),
            r#"{"type":"message_stop"}"#,
        ))
        .unwrap();
    assert!(parser.finish().is_ok());

    // A non-object message_start usage cannot seed the snapshot: warned,
    // ignored, stream alive.
    let mut parser2 = AnthropicMessages.stream_parser();
    let (events, ws) = parser2
        .parse(&SseEvent::new(
            Some("message_start"),
            json!({"type": "message_start", "message": {
                "id": "msg_2", "model": "m", "usage": 5,
            }})
            .to_string(),
        ))
        .unwrap();
    assert!(matches!(
        &events[0],
        StreamEvent::MessageStart { usage: None, .. }
    ));
    assert_eq!(ws.len(), 1, "{ws:?}");
    assert!(
        ws[0].message.contains("not a JSON object"),
        "{}",
        ws[0].message
    );
}

#[test]
fn error_event_fails_the_stream() {
    let err = feed("stream_error.sse").unwrap_err();
    match err {
        Error::Api {
            status,
            kind,
            message,
            ..
        } => {
            assert_eq!(status, 200);
            assert_eq!(kind, ApiErrorKind::Overloaded);
            assert_eq!(message, "Overloaded");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn truncated_stream_fails_finish() {
    let err = feed("stream_truncated.sse").unwrap_err();
    match err {
        Error::TruncatedStream { message, .. } => assert!(message.contains("message_stop")),
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn unknown_event_surfaces_with_warning() {
    let mut parser = AnthropicMessages.stream_parser();
    let (events, warnings) = parser
        .parse(&SseEvent::new(
            Some("mystery"),
            r#"{"type": "mystery", "x": 1}"#,
        ))
        .unwrap();
    assert_eq!(events, vec![StreamEvent::Unknown]);
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0].code, WarningCode::UnknownStreamEvent);
}

#[test]
fn protocol_violations_are_parse_errors() {
    // Delta for a block that never started.
    let mut parser = AnthropicMessages.stream_parser();
    let err = parser
        .parse(&SseEvent::new(
            Some("content_block_delta"),
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x"}}"#,
        ))
        .unwrap_err();
    assert!(matches!(err, Error::Parse { .. }));

    // Delta kind mismatched with the open block.
    let mut parser2 = AnthropicMessages.stream_parser();
    parser2
        .parse(&SseEvent::new(
            Some("content_block_start"),
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        ))
        .unwrap();
    let err2 = parser2
        .parse(&SseEvent::new(
            Some("content_block_delta"),
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"x"}}"#,
        ))
        .unwrap_err();
    assert!(matches!(err2, Error::Parse { .. }));

    // Non-JSON event data.
    let mut parser3 = AnthropicMessages.stream_parser();
    assert!(matches!(
        parser3.parse(&SseEvent::new(Some("message_stop"), "not json")),
        Err(Error::Parse { .. })
    ));
}

#[test]
fn event_name_falls_back_to_payload_type() {
    // Some proxies drop the SSE `event:` field; the payload type is used.
    let mut parser = AnthropicMessages.stream_parser();
    let (events, _) = parser
        .parse(&SseEvent::new(None, r#"{"type": "message_stop"}"#))
        .unwrap();
    assert_eq!(events, vec![StreamEvent::MessageStop]);
    assert!(parser.finish().is_ok());
}

#[test]
fn post_terminal_data_does_not_turn_the_stream_fatal() {
    // A complete minimal stream…
    let mut parser = AnthropicMessages.stream_parser();
    let mut events = Vec::new();
    for (name, data) in [
        (
            "message_start",
            json!({"type": "message_start", "message": {
                "id": "msg_1", "model": "claude-sonnet-5",
                "usage": {"input_tokens": 3, "output_tokens": 1},
            }})
            .to_string(),
        ),
        (
            "content_block_start",
            json!({"type": "content_block_start", "index": 0,
                   "content_block": {"type": "text", "text": ""}})
            .to_string(),
        ),
        (
            "content_block_delta",
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "text_delta", "text": "hi"}})
            .to_string(),
        ),
        (
            "content_block_stop",
            json!({"type": "content_block_stop", "index": 0}).to_string(),
        ),
        (
            "message_delta",
            json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                   "usage": {"output_tokens": 2}})
            .to_string(),
        ),
        ("message_stop", json!({"type": "message_stop"}).to_string()),
    ] {
        let (evs, ws) = parser.parse(&SseEvent::new(Some(name), data)).unwrap();
        events.extend(evs);
        assert!(ws.is_empty());
    }

    // …followed by a stray but well-formed event: surfaced as Unknown with
    // a warning instead of being applied to the finished message.
    let (post, ws) = parser
        .parse(&SseEvent::new(
            Some("content_block_delta"),
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "text_delta", "text": "junk"}})
            .to_string(),
        ))
        .unwrap();
    assert_eq!(post, vec![StreamEvent::Unknown]);
    assert_eq!(ws.len(), 1);
    assert_eq!(ws[0].code, WarningCode::UnknownStreamEvent);
    assert!(ws[0].message.contains("after the stream terminated"));

    // …and by non-JSON junk (proxy trailer): not a fatal error either.
    let (post2, ws2) = parser.parse(&SseEvent::new(None, "not json")).unwrap();
    assert_eq!(post2, vec![StreamEvent::Unknown]);
    assert_eq!(ws2[0].code, WarningCode::UnknownStreamEvent);

    // The stream still finishes cleanly and the accumulated response is
    // untouched by the post-terminal data.
    let (fin, fin_ws) = parser.finish().unwrap();
    assert!(fin.is_empty() && fin_ws.is_empty());
    events.extend(post);
    events.extend(post2);
    let resp = accumulate(&events);
    assert_eq!(resp.text(), "hi");
    assert_eq!(resp.stop_reason, Some(StopReason::EndTurn));
    assert_eq!(resp.usage.as_ref().unwrap().output_tokens, 2);
}
