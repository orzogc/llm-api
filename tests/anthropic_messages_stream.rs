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
        Error::Parse { message, .. } => assert!(message.contains("truncated")),
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
