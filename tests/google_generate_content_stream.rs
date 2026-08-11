//! Streaming tests for the `google_generate_content` format: SSE fixtures
//! exercising block-boundary inference (§ 9), blocked prompts,
//! multi-candidate handling, truncated streams and usage authority.

use llm_api::formats::google_generate_content::GoogleGenerateContent;
use llm_api::http::SseParser;
use llm_api::{
    Accumulator, ApiFormat, BlockDelta, ContentBlock, ConversionWarning, Error, StopReason,
    StreamEvent, StreamItem, WarningCode,
};
use serde_json::json;

const FMT: &str = "google_generate_content";

/// Runs a complete SSE byte stream through the SSE parser and the format's
/// stream parser; returns the unified events, the parse warnings and the
/// `finish()` outcome.
fn run_stream(data: &str) -> (Vec<StreamEvent>, Vec<ConversionWarning>, Result<(), Error>) {
    let format = GoogleGenerateContent;
    let mut parser = format.stream_parser();
    let mut sse = SseParser::new(usize::MAX);
    let mut sse_events = sse.push(data.as_bytes()).unwrap();
    sse_events.extend(sse.finish().unwrap());

    let mut events = Vec::new();
    let mut warnings = Vec::new();
    for event in &sse_events {
        let (evs, ws) = parser.parse(event).unwrap();
        events.extend(evs);
        warnings.extend(ws);
    }
    match parser.finish() {
        Ok((evs, ws)) => {
            events.extend(evs);
            warnings.extend(ws);
            (events, warnings, Ok(()))
        }
        Err(e) => (events, warnings, Err(e)),
    }
}

fn accumulate(events: &[StreamEvent]) -> llm_api::Response {
    let mut acc = Accumulator::new();
    for event in events {
        acc.push(&StreamItem::new(event.clone())).unwrap();
    }
    acc.finish().unwrap()
}

#[test]
fn text_stream_builds_one_block() {
    let (events, warnings, finish) = run_stream(include_str!(
        "fixtures/google_generate_content/stream_text.sse"
    ));
    assert!(warnings.is_empty(), "{warnings:?}");
    finish.unwrap();

    // MessageStart carries the envelope and the first usage snapshot.
    let StreamEvent::MessageStart {
        id, model, usage, ..
    } = &events[0]
    else {
        panic!("expected MessageStart, got {:?}", events[0]);
    };
    assert_eq!(id.as_deref(), Some("r-abc123"));
    assert_eq!(model.as_deref(), Some("gemini-2.5-flash"));
    assert_eq!(usage.as_ref().unwrap().input_tokens, 8);

    assert!(matches!(
        &events[1],
        StreamEvent::BlockStart {
            index: 0,
            block: ContentBlock::Text { .. },
            ..
        }
    ));
    let deltas: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::BlockDelta {
                delta: BlockDelta::Text(t),
                ..
            } => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        deltas,
        vec![
            "Once upon",
            " a time, a magic backpack",
            " granted wishes. The end."
        ]
    );

    // The final chunk's usage is authoritative on MessageDelta.
    let StreamEvent::MessageDelta {
        stop_reason, usage, ..
    } = events
        .iter()
        .find(|e| matches!(e, StreamEvent::MessageDelta { .. }))
        .unwrap()
    else {
        unreachable!()
    };
    assert_eq!(*stop_reason, Some(StopReason::EndTurn));
    let usage = usage.as_ref().unwrap();
    assert_eq!(usage.output_tokens, 58);
    assert_eq!(usage.total_tokens, Some(66));
    assert!(matches!(events.last().unwrap(), StreamEvent::MessageStop));

    let resp = accumulate(&events);
    assert_eq!(
        resp.text(),
        "Once upon a time, a magic backpack granted wishes. The end."
    );
    assert_eq!(resp.id.as_deref(), Some("r-abc123"));
    assert_eq!(resp.stop_reason, Some(StopReason::EndTurn));
    assert_eq!(resp.usage.as_ref().unwrap().output_tokens, 58);
    assert!(resp.raw.is_none());
}

#[test]
fn thinking_stream_infers_block_boundaries() {
    let (events, warnings, finish) = run_stream(include_str!(
        "fixtures/google_generate_content/stream_thinking.sse"
    ));
    assert!(warnings.is_empty(), "{warnings:?}");
    finish.unwrap();

    // Block 0: thinking across two chunks, sealed by its signature.
    assert!(matches!(
        &events[1],
        StreamEvent::BlockStart {
            index: 0,
            block: ContentBlock::Thinking { .. },
            ..
        }
    ));
    let sig_deltas: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::BlockDelta {
                delta: BlockDelta::Signature(s),
                ..
            } => Some(s.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(sig_deltas, vec!["c2lnLTE="]);
    let StreamEvent::BlockStop {
        index: 0,
        block: Some(finalized),
        ..
    } = events
        .iter()
        .find(|e| matches!(e, StreamEvent::BlockStop { index: 0, .. }))
        .unwrap()
    else {
        panic!("thinking block must finalize");
    };
    let ContentBlock::Thinking {
        text,
        signature,
        extra,
        ..
    } = finalized
    else {
        panic!("finalized block must be thinking");
    };
    assert_eq!(text.as_deref(), Some("**Plan** check the weather"));
    assert_eq!(signature.as_deref(), Some("c2lnLTE="));
    assert_eq!(
        extra.get(FMT).unwrap().get("thought").unwrap(),
        &json!(true)
    );

    // Block 1: plain text. Block 2: the complete function call.
    assert!(matches!(
        events
            .iter()
            .find(|e| matches!(e, StreamEvent::BlockStart { index: 1, .. }))
            .unwrap(),
        StreamEvent::BlockStart {
            block: ContentBlock::Text { .. },
            ..
        }
    ));
    let StreamEvent::BlockStart {
        block:
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
                extra,
                ..
            },
        ..
    } = events
        .iter()
        .find(|e| matches!(e, StreamEvent::BlockStart { index: 2, .. }))
        .unwrap()
    else {
        panic!("expected complete tool call");
    };
    assert_eq!(id.as_deref(), Some("c1"));
    assert_eq!(name, "get_weather");
    assert_eq!(arguments, r#"{"city":"Paris"}"#);
    assert_eq!(
        extra.get(FMT).unwrap().get("thoughtSignature").unwrap(),
        &json!("c2lnLTI=")
    );

    let resp = accumulate(&events);
    assert_eq!(resp.message.content.len(), 3);
    // STOP + tool call normalizes to ToolUse in the accumulator.
    assert_eq!(resp.stop_reason, Some(StopReason::ToolUse));
    let usage = resp.usage.as_ref().unwrap();
    assert_eq!(usage.input_tokens, 10);
    assert_eq!(usage.output_tokens, 40, "candidates + thoughts");
    assert_eq!(usage.reasoning_tokens, Some(31));
}

#[test]
fn blocked_prompt_stream_terminates_immediately() {
    let (events, warnings, finish) = run_stream(include_str!(
        "fixtures/google_generate_content/stream_blocked.sse"
    ));
    assert!(warnings.is_empty(), "{warnings:?}");
    finish.unwrap();
    assert!(matches!(&events[0], StreamEvent::MessageStart { .. }));
    assert!(matches!(
        &events[1],
        StreamEvent::MessageDelta {
            stop_reason: Some(StopReason::ContentFilter),
            ..
        }
    ));
    assert!(matches!(&events[2], StreamEvent::MessageStop));
    assert_eq!(events.len(), 3);

    let resp = accumulate(&events);
    assert!(resp.message.content.is_empty());
    assert_eq!(resp.stop_reason, Some(StopReason::ContentFilter));
    assert_eq!(resp.usage.as_ref().unwrap().input_tokens, 12);
}

#[test]
fn multi_candidate_chunks_surface_as_unknown_with_one_warning() {
    let (events, warnings, finish) = run_stream(include_str!(
        "fixtures/google_generate_content/stream_multi_candidate.sse"
    ));
    finish.unwrap();
    // The warning fires once per stream even though two chunks carried
    // extra candidates.
    let multi: Vec<_> = warnings
        .iter()
        .filter(|w| w.code == WarningCode::MultipleCandidates)
        .collect();
    assert_eq!(multi.len(), 1);
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, StreamEvent::Unknown))
            .count(),
        2
    );

    // Candidate 0 still parses normally.
    let resp = accumulate(&events);
    assert_eq!(resp.text(), "first!");
    assert_eq!(resp.stop_reason, Some(StopReason::EndTurn));
}

#[test]
fn truncated_stream_fails_finish() {
    let (events, warnings, finish) = run_stream(include_str!(
        "fixtures/google_generate_content/stream_truncated.sse"
    ));
    assert!(warnings.is_empty());
    let err = finish.unwrap_err();
    match err {
        Error::TruncatedStream { message, .. } => assert!(message.contains("finishReason")),
        other => panic!("unexpected: {other:?}"),
    }
    // No terminal events were fabricated.
    assert!(!events.iter().any(|e| matches!(e, StreamEvent::MessageStop)));
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, StreamEvent::MessageDelta { .. }))
    );
}

#[test]
fn signature_seals_a_thought_block() {
    // Two signed thought parts in one chunk must become two blocks — a
    // signature is validated against the exact part it was issued for.
    let stream = concat!(
        r#"data: {"candidates":[{"content":{"parts":[{"text":"a","thought":true,"thoughtSignature":"czE="},{"text":"b","thought":true,"thoughtSignature":"czI="}],"role":"model"},"finishReason":"STOP","index":0}],"responseId":"r-seal"}"#,
        "\n\n",
    );
    let (events, warnings, finish) = run_stream(stream);
    assert!(warnings.is_empty(), "{warnings:?}");
    finish.unwrap();
    let starts = events
        .iter()
        .filter(|e| matches!(e, StreamEvent::BlockStart { .. }))
        .count();
    assert_eq!(starts, 2);

    let resp = accumulate(&events);
    assert_eq!(resp.message.content.len(), 2);
    assert!(matches!(
        &resp.message.content[0],
        ContentBlock::Thinking { signature: Some(s), .. } if s == "czE="
    ));
    assert!(matches!(
        &resp.message.content[1],
        ContentBlock::Thinking { signature: Some(s), .. } if s == "czI="
    ));
}

#[test]
fn plain_text_signature_folds_into_the_finalized_block() {
    // A thoughtSignature can ride a non-thought text part; it surfaces as an
    // Other delta and lands in the finalized block's extra namespace.
    let stream = concat!(
        r#"data: {"candidates":[{"content":{"parts":[{"text":"hello","thoughtSignature":"dHh0"}],"role":"model"},"finishReason":"STOP","index":0}],"responseId":"r-txt"}"#,
        "\n\n",
    );
    let (events, warnings, finish) = run_stream(stream);
    assert!(warnings.is_empty());
    finish.unwrap();
    assert!(events.iter().any(|e| matches!(
        e,
        StreamEvent::BlockDelta { delta: BlockDelta::Other(v), .. }
            if v.get("thoughtSignature").is_some()
    )));
    let resp = accumulate(&events);
    let ContentBlock::Text { text, extra, .. } = &resp.message.content[0] else {
        panic!("expected text");
    };
    assert_eq!(text, "hello");
    assert_eq!(
        extra.get(FMT).unwrap().get("thoughtSignature").unwrap(),
        &json!("dHh0")
    );
}

#[test]
fn interleaved_thought_text_thought_opens_new_blocks() {
    // thought → text → thought must yield three blocks in arrival order.
    let stream = concat!(
        r#"data: {"candidates":[{"content":{"parts":[{"text":"t1","thought":true}],"role":"model"},"index":0}],"responseId":"r-i"}"#,
        "\n\n",
        r#"data: {"candidates":[{"content":{"parts":[{"text":"visible"}],"role":"model"},"index":0}],"responseId":"r-i"}"#,
        "\n\n",
        r#"data: {"candidates":[{"content":{"parts":[{"text":"t2","thought":true}],"role":"model"},"finishReason":"STOP","index":0}],"responseId":"r-i"}"#,
        "\n\n",
    );
    let (events, warnings, finish) = run_stream(stream);
    assert!(warnings.is_empty());
    finish.unwrap();
    let resp = accumulate(&events);
    assert_eq!(resp.message.content.len(), 3);
    assert!(
        matches!(&resp.message.content[0], ContentBlock::Thinking { text: Some(t), .. } if t == "t1")
    );
    assert!(
        matches!(&resp.message.content[1], ContentBlock::Text { text, .. } if text == "visible")
    );
    assert!(
        matches!(&resp.message.content[2], ContentBlock::Thinking { text: Some(t), .. } if t == "t2")
    );
}

#[test]
fn unknown_payloads_and_post_terminal_chunks() {
    let format = GoogleGenerateContent;
    let mut parser = format.stream_parser();

    // Undecodable data surfaces as Unknown with a cosmetic warning.
    let (events, warnings) = parser
        .parse(&llm_api::http::SseEvent::new(None, "not json"))
        .unwrap();
    assert!(matches!(events[..], [StreamEvent::Unknown]));
    assert_eq!(warnings[0].code, WarningCode::UnknownStreamEvent);

    // Terminate the stream, then feed another chunk.
    let terminal = json!({
        "candidates": [{
            "content": {"parts": [{"text": "hi"}], "role": "model"},
            "finishReason": "STOP",
            "index": 0,
        }],
        "responseId": "r-x",
    });
    let (events, _) = parser
        .parse(&llm_api::http::SseEvent::new(None, terminal.to_string()))
        .unwrap();
    assert!(matches!(events.last().unwrap(), StreamEvent::MessageStop));
    let (events, warnings) = parser
        .parse(&llm_api::http::SseEvent::new(None, terminal.to_string()))
        .unwrap();
    assert!(matches!(events[..], [StreamEvent::Unknown]));
    assert_eq!(warnings[0].code, WarningCode::UnknownStreamEvent);
    parser.finish().unwrap();
}

#[test]
fn usage_survives_when_the_final_chunk_omits_it() {
    // The last *seen* usageMetadata is used when the finishing chunk has
    // none.
    let stream = concat!(
        r#"data: {"candidates":[{"content":{"parts":[{"text":"a"}],"role":"model"},"index":0}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":1,"totalTokenCount":4},"responseId":"r-u"}"#,
        "\n\n",
        r#"data: {"candidates":[{"content":{"parts":[{"text":"b"}],"role":"model"},"finishReason":"STOP","index":0}],"responseId":"r-u"}"#,
        "\n\n",
    );
    let (events, _, finish) = run_stream(stream);
    finish.unwrap();
    let resp = accumulate(&events);
    assert_eq!(resp.usage.as_ref().unwrap().input_tokens, 3);
    assert_eq!(resp.usage.as_ref().unwrap().total_tokens, Some(4));
}

#[test]
fn empty_thought_part_still_opens_a_block() {
    // A summary part may arrive with empty text before the fragments.
    let stream = concat!(
        r#"data: {"candidates":[{"content":{"parts":[{"text":"","thought":true}],"role":"model"},"index":0}],"responseId":"r-e"}"#,
        "\n\n",
        r#"data: {"candidates":[{"content":{"parts":[{"text":"later","thought":true}],"role":"model"},"finishReason":"STOP","index":0}],"responseId":"r-e"}"#,
        "\n\n",
    );
    let (events, _, finish) = run_stream(stream);
    finish.unwrap();
    let resp = accumulate(&events);
    assert_eq!(resp.message.content.len(), 1);
    assert!(matches!(
        &resp.message.content[0],
        ContentBlock::Thinking { text: Some(t), .. } if t == "later"
    ));
}
