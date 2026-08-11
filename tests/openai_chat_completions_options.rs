//! `OpenAiChatCompletionsOptions::reasoning_field` (§ 12): the wire field
//! of the CC thinking channel is configurable (default
//! `reasoning_content`; dialects use `reasoning`, `thinking`, …). Covers
//! build/parse/stream symmetry, the demotion of `reasoning_content` under
//! a custom name, round-trips, invalid configurations and the pinned
//! default behavior.

use serde_json::{Value, json};

use llm_api::formats::anthropic_messages::AnthropicMessages;
use llm_api::formats::openai_chat_completions::{OpenAiChatCompletions, request_from_ir};
use llm_api::http::SseEvent;
use llm_api::{
    Accumulator, ApiFormat, BlockDelta, CallMode, ContentBlock, ConversionWarning, ConvertOptions,
    Error, FormatOptions, Message, OpenAiChatCompletionsOptions, Request, ResponseMeta,
    StreamEvent, StreamItem, WarningCode,
};

const F: &str = "openai_chat_completions";

/// CC options with a custom thinking field.
fn cc_options(field: &str) -> OpenAiChatCompletionsOptions {
    let mut options = OpenAiChatCompletionsOptions::default();
    options.reasoning_field = field.to_owned();
    options
}

/// Format options with a custom CC thinking field.
fn format_options(field: &str) -> FormatOptions {
    let mut options = FormatOptions::default();
    options.openai_chat_completions = cc_options(field);
    options
}

/// Builds a unary CC body from the request under the given CC options.
fn build(req: &Request, options: &OpenAiChatCompletionsOptions) -> (Value, Vec<ConversionWarning>) {
    request_from_ir(
        req,
        None,
        CallMode::Unary,
        &ConvertOptions::default(),
        options,
    )
    .unwrap()
}

/// A response body with the given assistant message object.
fn response_body(message: Value) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "id": "c", "object": "chat.completion", "model": "m",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
    }))
    .unwrap()
}

fn meta_ok() -> ResponseMeta {
    ResponseMeta::new(200, http::HeaderMap::new())
}

fn chunk_event(data: &Value) -> SseEvent {
    SseEvent::new(None, data.to_string())
}

// ---------------------------------------------------------------- build

#[test]
fn build_thinking_writes_custom_field() {
    // Parse a default-dialect response to obtain a native Thinking block,
    // then rebuild it under a custom field name.
    let parsed = OpenAiChatCompletions
        .parse_response(
            &response_body(json!({"role": "assistant",
                "reasoning_content": "think", "content": "hi"})),
            &meta_ok(),
        )
        .unwrap();
    let req = Request::with_messages(vec![parsed.message]);
    let (body, warnings) = build(&req, &cc_options("reasoning"));
    assert!(warnings.is_empty(), "{warnings:?}");
    let msg = &body["messages"][0];
    assert_eq!(msg["reasoning"], json!("think"));
    assert!(msg.get("reasoning_content").is_none(), "{msg}");
    assert_eq!(msg["content"], json!("hi"));
}

#[test]
fn build_thinking_as_text_uses_custom_field() {
    // A foreign (Anthropic) thinking block re-emitted via thinking_as_text
    // reaches the configured field; the signature-drop warning names it.
    let anthropic = AnthropicMessages
        .parse_response(
            &serde_json::to_vec(&json!({
                "id": "m", "type": "message", "role": "assistant", "model": "x",
                "content": [{"type": "thinking", "thinking": "t", "signature": "s"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }))
            .unwrap(),
            &meta_ok(),
        )
        .unwrap();
    let req = Request::with_messages(vec![anthropic.message]);
    let mut convert = ConvertOptions::default();
    convert.thinking_as_text = true;
    let (body, warnings) = request_from_ir(
        &req,
        None,
        CallMode::Unary,
        &convert,
        &cc_options("reasoning"),
    )
    .unwrap();
    assert_eq!(body["messages"][0]["reasoning"], json!("t"));
    assert!(
        warnings.iter().any(|w| {
            w.code == WarningCode::ThinkingSignatureDropped && w.message.contains("`reasoning`")
        }),
        "{warnings:?}"
    );
}

// ---------------------------------------------------------------- parse

#[test]
fn parse_response_reads_custom_field() {
    let resp = OpenAiChatCompletions
        .parse_response_with(
            &response_body(json!({"role": "assistant", "reasoning": "think", "content": "hi"})),
            &meta_ok(),
            &format_options("reasoning"),
        )
        .unwrap();
    assert!(resp.warnings.is_empty(), "{:?}", resp.warnings);
    assert_eq!(resp.message.content.len(), 2);
    assert!(matches!(&resp.message.content[0],
        ContentBlock::Thinking { text: Some(t), .. } if t == "think"));
    assert!(matches!(&resp.message.content[1],
        ContentBlock::Text { text, .. } if text == "hi"));
    assert!(resp.message.extra.is_empty(), "{:?}", resp.message.extra);
}

#[test]
fn parse_response_demotes_reasoning_content_under_custom_field() {
    // The configured name is the single authority: a wire
    // `reasoning_content` is an ordinary unknown field.
    let resp = OpenAiChatCompletions
        .parse_response_with(
            &response_body(json!({"role": "assistant",
                "reasoning": "think", "reasoning_content": "legacy", "content": "hi"})),
            &meta_ok(),
            &format_options("reasoning"),
        )
        .unwrap();
    assert!(resp.warnings.is_empty(), "{:?}", resp.warnings);
    assert!(matches!(&resp.message.content[0],
        ContentBlock::Thinking { text: Some(t), .. } if t == "think"));
    assert_eq!(
        resp.message.extra.get(F).unwrap().get("reasoning_content"),
        Some(&json!("legacy"))
    );
}

#[test]
fn parse_request_reads_custom_history() {
    let body = json!({"messages": [
        {"role": "user", "content": "q"},
        {"role": "assistant", "reasoning": "t", "content": "a"},
    ]});
    let (req, warnings) = OpenAiChatCompletions
        .parse_request_with(
            &serde_json::to_vec(&body).unwrap(),
            &format_options("reasoning"),
        )
        .unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(matches!(&req.messages[1].content[0],
        ContentBlock::Thinking { text: Some(t), .. } if t == "t"));
}

#[test]
fn custom_field_non_string_kept_verbatim() {
    // Same malformed handling as the default field: warn and mirror.
    let resp = OpenAiChatCompletions
        .parse_response_with(
            &response_body(json!({"role": "assistant", "reasoning": 42, "content": "hi"})),
            &meta_ok(),
            &format_options("reasoning"),
        )
        .unwrap();
    assert_eq!(resp.warnings.len(), 1, "{:?}", resp.warnings);
    assert_eq!(resp.warnings[0].code, WarningCode::MalformedField);
    assert_eq!(resp.warnings[0].location, "/choices/0/message/reasoning");
    assert!(
        resp.warnings[0].message.contains("`reasoning`"),
        "{:?}",
        resp.warnings
    );
    assert_eq!(resp.message.content.len(), 1, "no thinking block");
    assert_eq!(
        resp.message.extra.get(F).unwrap().get("reasoning"),
        Some(&json!(42))
    );
}

#[test]
fn default_options_keep_dialect_names_unknown() {
    // Pin: without configuration, `reasoning` / `thinking` fields stay
    // ordinary unknown fields on the message extra.
    let resp = OpenAiChatCompletions
        .parse_response(
            &response_body(json!({"role": "assistant",
                "reasoning": "r", "thinking": "t", "content": "hi"})),
            &meta_ok(),
        )
        .unwrap();
    assert!(resp.warnings.is_empty(), "{:?}", resp.warnings);
    assert_eq!(resp.message.content.len(), 1);
    assert!(matches!(
        &resp.message.content[0],
        ContentBlock::Text { .. }
    ));
    let ns = resp.message.extra.get(F).unwrap();
    assert_eq!(ns.get("reasoning"), Some(&json!("r")));
    assert_eq!(ns.get("thinking"), Some(&json!("t")));
}

// ---------------------------------------------------------------- stream

/// Runs chunks plus `[DONE]` through a parser with the given options.
fn run_stream(field: &str, chunks: &[Value]) -> (Vec<StreamEvent>, Vec<ConversionWarning>) {
    let mut parser = OpenAiChatCompletions.stream_parser_with(&format_options(field));
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
    events.extend(parser.finish().unwrap().0);
    (events, warnings)
}

fn accumulate(events: &[StreamEvent], warnings: &[ConversionWarning]) -> llm_api::Response {
    let mut acc = Accumulator::new();
    acc.extend_warnings(warnings.iter().cloned());
    for event in events {
        acc.push(&StreamItem::new(event.clone())).unwrap();
    }
    acc.finish().unwrap()
}

#[test]
fn stream_custom_field_drives_thinking_channel() {
    // The configured field streams as the thinking channel, interleaving
    // with `content` in arrival order exactly like the default field.
    let (events, warnings) = run_stream(
        "reasoning",
        &[
            json!({"id": "c", "choices": [{"index": 0,
                "delta": {"role": "assistant", "reasoning": "a"}, "finish_reason": null}]}),
            json!({"id": "c", "choices": [{"index": 0,
                "delta": {"content": "x"}, "finish_reason": null}]}),
            json!({"id": "c", "choices": [{"index": 0,
                "delta": {"reasoning": "b"}, "finish_reason": null}]}),
            json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        ],
    );
    assert!(warnings.is_empty(), "{warnings:?}");
    let resp = accumulate(&events, &warnings);
    assert_eq!(resp.message.content.len(), 3, "{:#?}", resp.message.content);
    assert!(matches!(&resp.message.content[0],
        ContentBlock::Thinking { text: Some(t), .. } if t == "a"));
    assert!(matches!(&resp.message.content[1],
        ContentBlock::Text { text, .. } if text == "x"));
    assert!(matches!(&resp.message.content[2],
        ContentBlock::Thinking { text: Some(t), .. } if t == "b"));
}

#[test]
fn stream_reasoning_content_is_leftover_under_custom_field() {
    // Under a custom name a `reasoning_content` delta is an ordinary
    // unknown field: surfaced as Other, folded under the `message`
    // reserved key, and replayed on the message level.
    let (events, warnings) = run_stream(
        "reasoning",
        &[
            json!({"id": "c", "choices": [{"index": 0,
                "delta": {"role": "assistant", "content": "hi",
                          "reasoning_content": "legacy"},
                "finish_reason": null}]}),
            json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        ],
    );
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(
        events.iter().any(|e| matches!(e,
            StreamEvent::BlockDelta { delta: BlockDelta::Other(v), .. }
                if v.get("reasoning_content").is_some())),
        "{events:#?}"
    );
    let resp = accumulate(&events, &warnings);
    assert_eq!(resp.message.content.len(), 1, "{:#?}", resp.message.content);
    let ContentBlock::Text { extra, .. } = &resp.message.content[0] else {
        panic!("expected text block");
    };
    assert_eq!(
        extra.get(F).unwrap().get("message"),
        Some(&json!({"reasoning_content": "legacy"}))
    );

    // Replay under the same options: back on the message level, while the
    // thinking channel stays untouched.
    let req = Request::with_messages(vec![resp.message.clone()]);
    let (body, ws) = build(&req, &cc_options("reasoning"));
    assert!(ws.is_empty(), "{ws:?}");
    let msg = &body["messages"][0];
    assert_eq!(msg["reasoning_content"], json!("legacy"));
    assert!(msg.get("reasoning").is_none(), "{msg}");
}

// ------------------------------------------------------------ round trip

#[test]
fn custom_wire_round_trips() {
    // serialize(parse(J)) == J under the same options.
    let body = json!({"messages": [
        {"role": "user", "content": "q"},
        {"role": "assistant", "reasoning": "t", "content": "a"},
    ]});
    let (req, ws) = OpenAiChatCompletions
        .parse_request_with(
            &serde_json::to_vec(&body).unwrap(),
            &format_options("reasoning"),
        )
        .unwrap();
    assert!(ws.is_empty(), "{ws:?}");
    let (rebuilt, ws) = build(&req, &cc_options("reasoning"));
    assert!(ws.is_empty(), "{ws:?}");
    assert_eq!(rebuilt["messages"], body["messages"]);
}

#[test]
fn custom_ir_round_trips() {
    // build(IR) then parse under the same options reproduces the blocks.
    let parsed = OpenAiChatCompletions
        .parse_response(
            &response_body(json!({"role": "assistant",
                "reasoning_content": "think", "content": "hi"})),
            &meta_ok(),
        )
        .unwrap();
    let req = Request::with_messages(vec![parsed.message.clone()]);
    let (body, _) = build(&req, &cc_options("reasoning"));
    let (req2, ws) = OpenAiChatCompletions
        .parse_request_with(
            &serde_json::to_vec(&body).unwrap(),
            &format_options("reasoning"),
        )
        .unwrap();
    assert!(ws.is_empty(), "{ws:?}");
    assert_eq!(req2.messages[0].content, parsed.message.content);
}

// -------------------------------------------------------- invalid config

#[test]
fn invalid_reasoning_field_fails_every_entry_point() {
    for bad in ["content", "tool_calls", "function_call", ""] {
        // Build.
        let req = Request::with_messages(vec![Message::user_text("q")]);
        let err = request_from_ir(
            &req,
            None,
            CallMode::Unary,
            &ConvertOptions::default(),
            &cc_options(bad),
        )
        .unwrap_err();
        assert!(matches!(err, Error::Conversion(_)), "build {bad}: {err}");
        assert!(err.to_string().contains("reasoning_field"), "{err}");

        // Non-streaming parse (response and request alike).
        let err = OpenAiChatCompletions
            .parse_response_with(
                &response_body(json!({"role": "assistant", "content": "hi"})),
                &meta_ok(),
                &format_options(bad),
            )
            .unwrap_err();
        assert!(matches!(err, Error::Conversion(_)), "parse {bad}: {err}");
        assert!(err.to_string().contains("reasoning_field"), "{err}");
        let err = OpenAiChatCompletions
            .parse_request_with(
                &serde_json::to_vec(&json!({"messages": []})).unwrap(),
                &format_options(bad),
            )
            .unwrap_err();
        assert!(matches!(err, Error::Conversion(_)), "request {bad}: {err}");

        // Stream: `stream_parser_with` cannot fail, the first parse does.
        let mut parser = OpenAiChatCompletions.stream_parser_with(&format_options(bad));
        let err = parser
            .parse(&chunk_event(&json!({"id": "c", "choices": []})))
            .unwrap_err();
        assert!(matches!(err, Error::Conversion(_)), "stream {bad}: {err}");
        assert!(err.to_string().contains("reasoning_field"), "{err}");
    }
}

#[test]
fn default_reasoning_field_matches_unconfigured_paths() {
    // The `_with` entry points under default options behave exactly like
    // the plain ones.
    let body = response_body(json!({"role": "assistant",
        "reasoning_content": "t", "content": "hi"}));
    let plain = OpenAiChatCompletions
        .parse_response(&body, &meta_ok())
        .unwrap();
    let with = OpenAiChatCompletions
        .parse_response_with(&body, &meta_ok(), &FormatOptions::default())
        .unwrap();
    assert_eq!(plain.message, with.message);
    assert_eq!(plain.warnings, with.warnings);
}

// ------------------------------------------- non-string thinking deltas

#[test]
fn stream_non_string_thinking_value_warns_and_folds() {
    // A non-string value of the thinking field (custom and default alike)
    // matches the non-streaming parse: MalformedField, then the unknown-
    // field path — folded under the `message` reserved key and replayed
    // on the message level. Nothing is silently dropped.
    for field in ["reasoning", "reasoning_content"] {
        let (events, warnings) = run_stream(
            field,
            &[
                json!({"id": "c", "choices": [{"index": 0,
                    "delta": {"role": "assistant", "content": "hi",
                              field: {"x": 1}},
                    "finish_reason": null}]}),
                json!({"id": "c", "choices": [{"index": 0, "delta": {},
                    "finish_reason": "stop"}]}),
            ],
        );
        let malformed: Vec<_> = warnings
            .iter()
            .filter(|w| w.code == WarningCode::MalformedField)
            .collect();
        assert_eq!(malformed.len(), 1, "{field}: {warnings:?}");
        assert_eq!(malformed[0].location, format!("/choices/0/delta/{field}"));
        assert!(
            malformed[0].message.contains(&format!("`{field}`")),
            "{field}: {warnings:?}"
        );
        assert!(
            events.iter().any(|e| matches!(e,
                StreamEvent::BlockDelta { delta: BlockDelta::Other(v), .. }
                    if v.get(field).is_some())),
            "{field}: {events:#?}"
        );

        let resp = accumulate(&events, &warnings);
        assert_eq!(resp.message.content.len(), 1, "{field}");
        let ContentBlock::Text { extra, .. } = &resp.message.content[0] else {
            panic!("{field}: expected text block");
        };
        assert_eq!(
            extra.get(F).unwrap().get("message"),
            Some(&json!({field: {"x": 1}}))
        );

        // Replay under the same options: message level, no thinking field
        // fabricated from it.
        let req = Request::with_messages(vec![resp.message.clone()]);
        let (body, ws) = build(&req, &cc_options(field));
        assert!(ws.is_empty(), "{field}: {ws:?}");
        assert_eq!(body["messages"][0][field], json!({"x": 1}));
    }
}

#[test]
fn stream_null_thinking_value_stays_silent() {
    // Pin: `null` is the absent canonical form — no event, no warning.
    for field in ["reasoning", "reasoning_content"] {
        let (events, warnings) = run_stream(
            field,
            &[
                json!({"id": "c", "choices": [{"index": 0,
                    "delta": {"role": "assistant", "content": "hi", field: null},
                    "finish_reason": null}]}),
                json!({"id": "c", "choices": [{"index": 0, "delta": {},
                    "finish_reason": "stop"}]}),
            ],
        );
        assert!(warnings.is_empty(), "{field}: {warnings:?}");
        assert!(
            events.iter().all(|e| !matches!(
                e,
                StreamEvent::BlockDelta {
                    delta: BlockDelta::Other(_),
                    ..
                }
            )),
            "{field}: {events:#?}"
        );
        let resp = accumulate(&events, &warnings);
        assert_eq!(resp.message.content.len(), 1, "{field}");
        assert!(matches!(&resp.message.content[0],
            ContentBlock::Text { text, extra, .. } if text == "hi" && extra.is_empty()));
    }
}

// ------------------------------------------------- escaped locations

#[test]
fn warning_locations_escape_the_reasoning_field() {
    // JSON-Pointer locations escape the configured name (RFC 6901); the
    // wire key itself and the warning message stay unescaped.
    for (field, escaped) in [("x/y", "x~1y"), ("a~b", "a~0b")] {
        let resp = OpenAiChatCompletions
            .parse_response_with(
                &response_body(json!({"role": "assistant", field: 42, "content": "hi"})),
                &meta_ok(),
                &format_options(field),
            )
            .unwrap();
        assert_eq!(resp.warnings.len(), 1, "{field}: {:?}", resp.warnings);
        assert_eq!(
            resp.warnings[0].location,
            format!("/choices/0/message/{escaped}")
        );
        assert!(
            resp.warnings[0].message.contains(&format!("`{field}`")),
            "{field}: {:?}",
            resp.warnings
        );

        // Streaming malformed value: same escaping.
        let mut parser = OpenAiChatCompletions.stream_parser_with(&format_options(field));
        let (_, ws) = parser
            .parse(&chunk_event(&json!({"id": "c", "choices": [{"index": 0,
                "delta": {"role": "assistant", "content": "hi", field: 42},
                "finish_reason": null}]})))
            .unwrap();
        assert_eq!(ws.len(), 1, "{field}: {ws:?}");
        assert_eq!(ws[0].location, format!("/choices/0/delta/{escaped}"));

        // The wire key itself is not escaped.
        let parsed = OpenAiChatCompletions
            .parse_response_with(
                &response_body(json!({"role": "assistant", field: "think"})),
                &meta_ok(),
                &format_options(field),
            )
            .unwrap();
        let req = Request::with_messages(vec![parsed.message]);
        let (body, ws) = build(&req, &cc_options(field));
        assert!(ws.is_empty(), "{field}: {ws:?}");
        assert_eq!(body["messages"][0][field], json!("think"));
    }
}

#[test]
fn escaped_location_matches_extra_override() {
    // `Extra::merge_into` escapes merge-log pointers the same way, so a
    // user override of the configured field marks the warning overridden.
    let mut msg = Message::assistant(vec![ContentBlock::thinking_signed("t", "sig")]);
    msg.extra.set(F, "x/y", json!("user"));
    let req = Request::with_messages(vec![msg]);
    let (body, warnings) = build(&req, &cc_options("x/y"));
    let dropped: Vec<_> = warnings
        .iter()
        .filter(|w| w.code == WarningCode::ThinkingSignatureDropped)
        .collect();
    assert_eq!(dropped.len(), 1, "{warnings:?}");
    assert_eq!(dropped[0].location, "/messages/0/x~1y");
    assert!(dropped[0].overridden, "{warnings:?}");
    // The user's merge wins over the built thinking value.
    assert_eq!(body["messages"][0]["x/y"], json!("user"));
}
