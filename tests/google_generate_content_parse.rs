//! Parse-side (wire → IR) tests for the `google_generate_content` format:
//! request round-trip idempotence (§ 1), response parsing (§ 8), model
//! listing, token counting and error classification.

use std::time::Duration;

use llm_api::formats::google_generate_content::{
    GoogleGenerateContent, request_from_ir, request_to_ir, response_to_ir,
};
use llm_api::{
    ApiErrorKind, ApiFormat, ContentBlock, ConvertOptions, Error, ImageSource, ResponseMeta,
    Role, StopReason, Tool, ToolChoice, ToolOutputBlock, WarningCode,
};
use serde_json::{Value, json};

const FMT: &str = "google_generate_content";

fn meta() -> ResponseMeta {
    ResponseMeta::new(200, http::HeaderMap::new())
}

/// parse → serialize must reproduce a canonical body exactly; a second
/// parse → serialize pass must be the identity (§ 1 idempotence).
fn assert_fixed_point(body: &Value) {
    let (ir, _) = request_to_ir(&serde_json::to_vec(body).unwrap()).unwrap();
    let (serialized, _) = request_from_ir(&ir, &ConvertOptions::default()).unwrap();
    assert_eq!(&serialized, body, "canonical body must be a fixed point");
    let (ir2, _) = request_to_ir(&serde_json::to_vec(&serialized).unwrap()).unwrap();
    assert_eq!(ir2, ir, "re-parsing the canonical form must reproduce the IR");
}

// ---------------------------------------------------------------- round trip

#[test]
fn canonical_request_fixture_round_trips_losslessly() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/google_generate_content/request_canonical.json"))
            .unwrap();
    let (ir, parse_warnings) =
        request_to_ir(&serde_json::to_vec(&fixture).unwrap()).unwrap();
    assert!(parse_warnings.is_empty(), "{parse_warnings:?}");
    let (serialized, warnings) = request_from_ir(&ir, &ConvertOptions::default()).unwrap();
    assert!(warnings.is_empty(), "round trip must be warning-free: {warnings:?}");
    assert_eq!(serialized, fixture);

    // Structural expectations on the parsed IR.
    assert_eq!(ir.system.as_ref().unwrap().len(), 1);
    let roles: Vec<Role> = ir.messages.iter().map(|m| m.role).collect();
    assert_eq!(
        roles,
        vec![Role::User, Role::Assistant, Role::Tool, Role::User, Role::Assistant, Role::User]
    );
    // The mixed wire turn split into Tool + User sharing one turn group.
    let g2 = ir.messages[2].turn_group_id();
    let g3 = ir.messages[3].turn_group_id();
    assert!(g2.is_some() && g2 == g3);
    assert_eq!(ir.messages[0].turn_group_id(), None);
    // Content-level unknown fields ride the first message of the turn.
    assert_eq!(
        ir.messages[0].extra.get(FMT).unwrap().get("contentTag").unwrap(),
        &json!("first")
    );
    // Thinking parsed natively (signature + wire marker).
    assert!(matches!(
        &ir.messages[1].content[0],
        ContentBlock::Thinking { text: Some(t), signature: Some(s), .. }
            if t == "Let me look that up." && s == "dGhvdWdodC1zaWc="
    ));
    // Tool-call thought signature rides the block's extra.
    let ContentBlock::ToolCall { id, name, arguments, extra, .. } = &ir.messages[1].content[1]
    else {
        panic!("expected tool call");
    };
    assert_eq!(id.as_deref(), Some("call-1"));
    assert_eq!(name, "get_weather");
    assert_eq!(arguments, r#"{"city":"Paris"}"#);
    assert_eq!(extra.get(FMT).unwrap().get("thoughtSignature").unwrap(), &json!("Y2FsbC1zaWc="));
    // The executable-code part survives as an in-place opaque block.
    assert!(matches!(
        &ir.messages[5].content[2],
        ContentBlock::Opaque { format, value, .. } if format == FMT && value.get("executableCode").is_some()
    ));
    // Tools: one function + one hosted opaque; tool choice restricted.
    let tools = ir.tools.as_ref().unwrap();
    assert!(matches!(&tools[0], Tool::Function(f) if f.name == "get_weather"));
    assert!(matches!(&tools[1], Tool::Opaque { format, .. } if format == FMT));
    assert_eq!(ir.tool_choice, Some(ToolChoice::tool("get_weather")));
    // thinkingBudget (unmodeled) mirrors into the request extra.
    assert_eq!(
        ir.extra.get(FMT).unwrap().get("generationConfig").unwrap(),
        &json!({"thinkingConfig": {"thinkingBudget": 1024}})
    );

    // Second pass: parse(serialize) reproduces the IR and the JSON.
    let (ir2, _) = request_to_ir(&serde_json::to_vec(&serialized).unwrap()).unwrap();
    assert_eq!(ir2, ir);
    let (serialized2, _) = request_from_ir(&ir2, &ConvertOptions::default()).unwrap();
    assert_eq!(serialized2, serialized);
}

#[test]
fn tool_result_response_encodings_round_trip() {
    for response in [
        json!({}),
        json!({"output": ""}),
        json!({"output": "plain"}),
        json!({"error": "boom"}),
        json!({"error": ""}),
        json!({"error": {"code": 1}}),
        json!({"output": 42, "note": "x"}),
        json!({"result": [1, 2, 3]}),
        json!({"error": "boom", "output": "partial"}),
    ] {
        let body = json!({
            "contents": [
                {"role": "user", "parts": [
                    {"functionResponse": {"name": "f", "response": response}}
                ]}
            ]
        });
        assert_fixed_point(&body);
    }
}

#[test]
fn tool_result_decoding_details() {
    let body = json!({
        "contents": [
            {"role": "user", "parts": [
                {"functionResponse": {"id": "c1", "name": "f", "response": {"error": "boom", "hint": "retry"}}},
                {"functionResponse": {"name": "g", "response": {"output": ""}}},
                {"functionResponse": {"name": "h", "response": {}}},
                {"functionResponse": {"name": "i", "response": {"output": "text"},
                    "parts": [{"inlineData": {"mimeType": "image/png", "data": "cGl4"}}]}}
            ]}
        ]
    });
    let (ir, warnings) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(ir.messages.len(), 1);
    assert_eq!(ir.messages[0].role, Role::Tool);
    let blocks = &ir.messages[0].content;

    // "error" key → is_error + text; other keys mirror into extra.
    let ContentBlock::ToolResult { tool_call_id, name, content, is_error, extra, .. } = &blocks[0]
    else {
        panic!("expected tool result");
    };
    assert_eq!(tool_call_id.as_deref(), Some("c1"));
    assert_eq!(name.as_deref(), Some("f"));
    assert_eq!(*is_error, Some(true));
    assert!(matches!(&content[0], ToolOutputBlock::Text { text, .. } if text == "boom"));
    assert_eq!(
        extra.get(FMT).unwrap().get("functionResponse").unwrap(),
        &json!({"response": {"hint": "retry"}})
    );

    // {"output": ""} → one empty text block; {} → empty content.
    let ContentBlock::ToolResult { content, is_error, .. } = &blocks[1] else { panic!() };
    assert!(matches!(&content[0], ToolOutputBlock::Text { text, .. } if text.is_empty()));
    assert_eq!(*is_error, None);
    let ContentBlock::ToolResult { content, .. } = &blocks[2] else { panic!() };
    assert!(content.is_empty());

    // parts[] media parse after the response text (canonical order).
    let ContentBlock::ToolResult { content, .. } = &blocks[3] else { panic!() };
    assert!(matches!(&content[0], ToolOutputBlock::Text { text, .. } if text == "text"));
    assert!(matches!(
        &content[1],
        ToolOutputBlock::Image { source: ImageSource::Base64 { media_type, .. }, .. }
            if media_type == "image/png"
    ));
    assert_fixed_point(&body);
}

#[test]
fn malformed_function_response_value_is_preserved() {
    let body = json!({
        "contents": [
            {"role": "user", "parts": [{"functionResponse": {"name": "f", "response": "not an object"}}]}
        ]
    });
    let (ir, warnings) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0].code, WarningCode::MalformedField);
    let ContentBlock::ToolResult { content, extra, .. } = &ir.messages[0].content[0] else {
        panic!()
    };
    assert!(content.is_empty());
    assert_eq!(
        extra.get(FMT).unwrap().get("functionResponse").unwrap(),
        &json!({"response": "not an object"})
    );
    // Still a fixed point: the scalar is restored by the extra merge.
    let (serialized, _) = request_from_ir(&ir, &ConvertOptions::default()).unwrap();
    assert_eq!(serialized, body);
}

#[test]
fn legacy_function_role_parses_as_tool_turn() {
    let body = json!({
        "contents": [
            {"role": "function", "parts": [{"functionResponse": {"name": "f", "response": {"output": "ok"}}}]}
        ]
    });
    let (ir, _) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert_eq!(ir.messages[0].role, Role::Tool);
    // Canonicalizes to the documented user turn on the way back.
    let (serialized, _) = request_from_ir(&ir, &ConvertOptions::default()).unwrap();
    assert_eq!(serialized["contents"][0]["role"], "user");
}

#[test]
fn function_call_args_edge_cases() {
    // Absent args canonicalize to "{}".
    let body = json!({
        "contents": [{"role": "model", "parts": [{"functionCall": {"name": "f"}}]}]
    });
    let (ir, warnings) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert!(warnings.is_empty());
    assert!(matches!(
        &ir.messages[0].content[0],
        ContentBlock::ToolCall { arguments, .. } if arguments == "{}"
    ));
    let (serialized, _) = request_from_ir(&ir, &ConvertOptions::default()).unwrap();
    assert_eq!(
        serialized["contents"][0]["parts"][0]["functionCall"],
        json!({"name": "f", "args": {}})
    );

    // Non-object args parse leniently with a warning; serializing the
    // resulting IR fails per the § 4.5 arguments contract.
    let body = json!({
        "contents": [{"role": "model", "parts": [{"functionCall": {"name": "f", "args": 5}}]}]
    });
    let (ir, warnings) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert_eq!(warnings[0].code, WarningCode::MalformedField);
    assert!(matches!(
        &ir.messages[0].content[0],
        ContentBlock::ToolCall { arguments, .. } if arguments == "5"
    ));
    assert!(request_from_ir(&ir, &ConvertOptions::default()).is_err());
}

#[test]
fn user_parts_that_are_invalid_for_the_role_stay_opaque() {
    // A functionCall inside a user turn and a thought part inside a user
    // turn are invalid IR combinations (§ 7.4) — they parse as opaque
    // blocks and round-trip verbatim.
    let body = json!({
        "contents": [
            {"role": "user", "parts": [
                {"functionCall": {"name": "f", "args": {}}},
                {"text": "thinking?", "thought": true},
                {"text": "normal"}
            ]}
        ]
    });
    let (ir, warnings) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert!(warnings.is_empty());
    assert!(matches!(&ir.messages[0].content[0], ContentBlock::Opaque { .. }));
    assert!(matches!(&ir.messages[0].content[1], ContentBlock::Opaque { .. }));
    assert!(matches!(&ir.messages[0].content[2], ContentBlock::Text { .. }));
    assert_fixed_point(&body);
}

#[test]
fn unmodeled_tool_config_shapes_mirror_into_extra() {
    // VALIDATED mode is not modeled.
    let body = json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "toolConfig": {"functionCallingConfig": {"mode": "VALIDATED", "allowedFunctionNames": ["a", "b"]}}
    });
    let (ir, _) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert_eq!(ir.tool_choice, None);
    assert_fixed_point(&body);

    // ANY with several allowed names degrades to Required + mirror.
    let body = json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "toolConfig": {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["a", "b"]}}
    });
    let (ir, _) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert_eq!(ir.tool_choice, Some(ToolChoice::Required));
    assert_fixed_point(&body);

    // Plain modes map to the § 4.5 table.
    for (mode, expected) in [
        ("AUTO", ToolChoice::Auto),
        ("NONE", ToolChoice::None),
        ("ANY", ToolChoice::Required),
    ] {
        let body = json!({
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "toolConfig": {"functionCallingConfig": {"mode": mode}}
        });
        let (ir, _) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
        assert_eq!(ir.tool_choice, Some(expected));
        assert_fixed_point(&body);
    }
}

#[test]
fn mixed_tool_entry_canonicalizes_by_splitting() {
    let body = json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "tools": [{"functionDeclarations": [{"name": "f"}], "googleSearch": {}}]
    });
    let (ir, _) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    let tools = ir.tools.as_ref().unwrap();
    assert_eq!(tools.len(), 2);
    let (serialized, _) = request_from_ir(&ir, &ConvertOptions::default()).unwrap();
    assert_eq!(
        serialized["tools"],
        json!([{"functionDeclarations": [{"name": "f"}]}, {"googleSearch": {}}])
    );
    // The canonicalized form is a fixed point.
    assert_fixed_point(&serialized);
}

#[test]
fn unknown_null_fields_canonicalize_to_absent() {
    let body = json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "customTop": null
    });
    let (ir, _) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    let (serialized, _) = request_from_ir(&ir, &ConvertOptions::default()).unwrap();
    assert_eq!(serialized, json!({"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}));
}

#[test]
fn non_json_output_configs_mirror_into_extra() {
    for config in [
        json!({"responseMimeType": "text/x.enum", "responseSchema": {"type": "STRING", "enum": ["A", "B"]}}),
        json!({"responseMimeType": "text/plain"}),
    ] {
        let body = json!({
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "generationConfig": config
        });
        let (ir, _) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
        assert_eq!(ir.output_format, None);
        assert_fixed_point(&body);
    }

    // JSON mode without a schema parses to JsonObject.
    let body = json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "generationConfig": {"responseMimeType": "application/json"}
    });
    let (ir, _) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert!(matches!(ir.output_format, Some(llm_api::OutputFormat::JsonObject)));
    assert_fixed_point(&body);
}

#[test]
fn empty_contents_and_empty_parts_round_trip() {
    for body in [
        json!({"contents": []}),
        json!({"contents": [{"role": "user", "parts": []}]}),
        json!({"contents": [{"role": "model", "parts": []}]}),
    ] {
        assert_fixed_point(&body);
    }
}

// ------------------------------------------------------------------ responses

#[test]
fn text_response_parses() {
    let body = include_str!("fixtures/google_generate_content/response_text.json");
    let resp = response_to_ir(body.as_bytes(), &meta()).unwrap();
    assert!(resp.warnings.is_empty(), "{:?}", resp.warnings);
    assert_eq!(resp.id.as_deref(), Some("resp-123"));
    assert_eq!(resp.model.as_deref(), Some("gemini-2.5-flash"));
    assert_eq!(resp.text(), "AI learns patterns from data to make predictions.");
    assert_eq!(resp.stop_reason, Some(StopReason::EndTurn));
    let usage = resp.usage.as_ref().unwrap();
    assert_eq!(usage.input_tokens, 8);
    assert_eq!(usage.output_tokens, 10);
    assert_eq!(usage.total_tokens, Some(18));
    assert_eq!(usage.reasoning_tokens, None);
    assert!(usage.raw.is_some());
    assert_eq!(resp.status, 200);
    assert!(resp.raw.as_ref().unwrap().get("candidates").is_some());
}

#[test]
fn function_call_response_normalizes_to_tool_use() {
    let body = include_str!("fixtures/google_generate_content/response_function_call.json");
    let resp = response_to_ir(body.as_bytes(), &meta()).unwrap();
    assert!(resp.warnings.is_empty());
    let ContentBlock::ToolCall { id, name, arguments, extra, .. } = &resp.message.content[0]
    else {
        panic!("expected tool call");
    };
    assert_eq!(id.as_deref(), Some("call-9"));
    assert_eq!(name, "set_light_color");
    assert_eq!(arguments, r#"{"rgb_hex":"ff0000"}"#);
    assert_eq!(extra.get(FMT).unwrap().get("thoughtSignature").unwrap(), &json!("c2lnLTk="));
    // Google reports STOP for function calls; § 8 normalizes to ToolUse.
    assert_eq!(resp.stop_reason, Some(StopReason::ToolUse));
    let usage = resp.usage.as_ref().unwrap();
    assert_eq!(usage.input_tokens, 21);
    assert_eq!(usage.output_tokens, 35, "candidates + thoughts");
    assert_eq!(usage.reasoning_tokens, Some(30));
    assert_eq!(usage.visible_output_tokens(), 5);
}

#[test]
fn thinking_response_parses_thought_parts() {
    let body = include_str!("fixtures/google_generate_content/response_thinking.json");
    let resp = response_to_ir(body.as_bytes(), &meta()).unwrap();
    let ContentBlock::Thinking { text, signature, extra, .. } = &resp.message.content[0] else {
        panic!("expected thinking");
    };
    assert!(text.as_ref().unwrap().starts_with("**Considering"));
    assert_eq!(signature.as_deref(), Some("dGhpbmstc2ln"));
    assert_eq!(extra.get(FMT).unwrap().get("thought").unwrap(), &json!(true));
    assert!(matches!(&resp.message.content[1], ContentBlock::Text { .. }));
    assert_eq!(resp.stop_reason, Some(StopReason::EndTurn));
    assert_eq!(resp.usage.as_ref().unwrap().output_tokens, 52);

    // The parsed assistant message replays into a request natively.
    let req = llm_api::Request::with_messages(vec![
        llm_api::Message::user_text("haiku please"),
        resp.message.clone(),
    ]);
    let (serialized, warnings) = request_from_ir(&req, &ConvertOptions::default()).unwrap();
    assert!(warnings.is_empty(), "replay must be warning-free: {warnings:?}");
    assert_eq!(
        serialized["contents"][1]["parts"][0]["thoughtSignature"],
        json!("dGhpbmstc2ln")
    );
    assert_eq!(serialized["contents"][1]["parts"][0]["thought"], json!(true));
}

#[test]
fn blocked_prompt_response_has_no_candidates() {
    let body = include_str!("fixtures/google_generate_content/response_blocked.json");
    let resp = response_to_ir(body.as_bytes(), &meta()).unwrap();
    assert!(resp.warnings.is_empty(), "{:?}", resp.warnings);
    assert!(resp.message.content.is_empty());
    assert_eq!(resp.stop_reason, Some(StopReason::ContentFilter));
    assert_eq!(
        resp.raw.as_ref().unwrap()["promptFeedback"]["blockReason"],
        json!("SAFETY")
    );
    assert_eq!(resp.usage.as_ref().unwrap().input_tokens, 12);
}

#[test]
fn multi_candidate_response_reads_first_and_warns() {
    let body = include_str!("fixtures/google_generate_content/response_multi_candidate.json");
    let resp = response_to_ir(body.as_bytes(), &meta()).unwrap();
    assert_eq!(resp.text(), "First answer.");
    assert_eq!(resp.warnings.len(), 1);
    assert_eq!(resp.warnings[0].code, WarningCode::MultipleCandidates);
    // The second candidate stays reachable through raw.
    assert_eq!(resp.raw.as_ref().unwrap()["candidates"][1]["content"]["parts"][0]["text"], "Second answer.");
}

#[test]
fn finish_reason_mapping_covers_the_families() {
    for (reason, expected) in [
        ("STOP", StopReason::EndTurn),
        ("MAX_TOKENS", StopReason::MaxTokens),
        ("SAFETY", StopReason::ContentFilter),
        ("PROHIBITED_CONTENT", StopReason::ContentFilter),
        ("BLOCKLIST", StopReason::ContentFilter),
        ("SPII", StopReason::ContentFilter),
        ("IMAGE_SAFETY", StopReason::ContentFilter),
        ("IMAGE_OTHER", StopReason::ContentFilter),
        ("RECITATION", StopReason::Other("RECITATION".to_owned())),
        ("MALFORMED_FUNCTION_CALL", StopReason::Other("MALFORMED_FUNCTION_CALL".to_owned())),
        ("MISSING_THOUGHT_SIGNATURE", StopReason::Other("MISSING_THOUGHT_SIGNATURE".to_owned())),
        ("NO_IMAGE", StopReason::Other("NO_IMAGE".to_owned())),
    ] {
        let body = json!({
            "candidates": [{
                "content": {"parts": [{"text": "x"}], "role": "model"},
                "finishReason": reason,
            }]
        });
        let resp = response_to_ir(&serde_json::to_vec(&body).unwrap(), &meta()).unwrap();
        assert_eq!(resp.stop_reason, Some(expected), "for {reason}");
    }
}

#[test]
fn malformed_responses_degrade_gracefully() {
    // Neither candidates nor a block reason.
    let resp = response_to_ir(b"{}", &meta()).unwrap();
    assert!(resp.message.content.is_empty());
    assert_eq!(resp.stop_reason, None);
    assert_eq!(resp.warnings[0].code, WarningCode::MalformedField);

    // Invalid JSON is a parse error.
    assert!(matches!(
        response_to_ir(b"not json", &meta()),
        Err(Error::Parse { .. })
    ));

    // Assistant media output parses as an image block (parse-only channel).
    let body = json!({
        "candidates": [{
            "content": {
                "parts": [{"inlineData": {"mimeType": "image/png", "data": "cGl4"}}],
                "role": "model",
            },
            "finishReason": "STOP",
        }]
    });
    let resp = response_to_ir(&serde_json::to_vec(&body).unwrap(), &meta()).unwrap();
    assert!(matches!(
        &resp.message.content[0],
        ContentBlock::Image { source: ImageSource::Base64 { .. }, .. }
    ));
}

// -------------------------------------------------------- models, countTokens

#[test]
fn models_pages_parse_and_paginate() {
    let format = GoogleGenerateContent;
    let page1 = include_str!("fixtures/google_generate_content/models_page1.json");
    let (models, next) = format.parse_models_response(page1.as_bytes()).unwrap();
    assert_eq!(next.as_deref(), Some("tok-page-2"));
    // The entry without a resource name is skipped.
    assert_eq!(models.len(), 2);
    assert_eq!(models[0].id, "gemini-2.5-flash");
    assert_eq!(models[0].display_name.as_deref(), Some("Gemini 2.5 Flash"));
    assert_eq!(models[0].created, None);
    // The original resource name stays in raw.
    assert_eq!(models[0].raw["name"], "models/gemini-2.5-flash");
    assert_eq!(models[1].id, "gemini-2.5-pro");

    let page2 = include_str!("fixtures/google_generate_content/models_page2.json");
    let (models, next) = format.parse_models_response(page2.as_bytes()).unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(next, None);

    assert!(matches!(
        format.parse_models_response(b"[]"),
        Err(Error::Parse { .. })
    ));
}

#[test]
fn count_tokens_response_parses() {
    let format = GoogleGenerateContent;
    let body = include_str!("fixtures/google_generate_content/count_tokens_response.json");
    let count = format.parse_count_tokens_response(body.as_bytes()).unwrap();
    assert_eq!(count.input_tokens, 268);
    assert_eq!(count.raw.as_ref().unwrap()["promptTokensDetails"][1]["modality"], "IMAGE");
    assert!(count.warnings.is_empty());

    // proto3 omits zero counts: an empty body means zero tokens.
    let count = format.parse_count_tokens_response(b"{}").unwrap();
    assert_eq!(count.input_tokens, 0);
}

// -------------------------------------------------------------------- errors

#[test]
fn error_status_maps_grpc_codes() {
    let format = GoogleGenerateContent;
    let body = include_str!("fixtures/google_generate_content/error_rate_limit.json");
    let mut headers = http::HeaderMap::new();
    headers.insert(http::header::RETRY_AFTER, http::HeaderValue::from_static("7"));
    let err = format.parse_error(429, &headers, body.as_bytes());
    match err {
        Error::Api { status, kind, message, parsed, retry_after, truncated, .. } => {
            assert_eq!(status, 429);
            assert_eq!(kind, ApiErrorKind::RateLimit);
            assert!(message.contains("exhausted"));
            assert!(parsed.is_some());
            assert_eq!(retry_after, Some(Duration::from_secs(7)));
            assert!(!truncated);
        }
        other => panic!("unexpected: {other:?}"),
    }

    for (status, grpc, expected) in [
        (400u16, "INVALID_ARGUMENT", ApiErrorKind::InvalidRequest),
        (400, "FAILED_PRECONDITION", ApiErrorKind::InvalidRequest),
        (401, "UNAUTHENTICATED", ApiErrorKind::Auth),
        (403, "PERMISSION_DENIED", ApiErrorKind::PermissionDenied),
        (404, "NOT_FOUND", ApiErrorKind::NotFound),
        (429, "RESOURCE_EXHAUSTED", ApiErrorKind::RateLimit),
        (503, "UNAVAILABLE", ApiErrorKind::Overloaded),
        (500, "INTERNAL", ApiErrorKind::ServerError),
        // Unknown codes fall back to HTTP-status classification.
        (402, "SOMETHING_NEW", ApiErrorKind::InvalidRequest),
    ] {
        let body =
            json!({"error": {"code": status, "message": "m", "status": grpc}}).to_string();
        let err = format.parse_error(status, &http::HeaderMap::new(), body.as_bytes());
        match err {
            Error::Api { kind, .. } => assert_eq!(kind, expected, "for {grpc}"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    // Non-JSON bodies degrade to generic classification with the raw kept.
    let err = format.parse_error(502, &http::HeaderMap::new(), b"<html>Bad Gateway</html>");
    match err {
        Error::Api { kind, message, parsed, raw, .. } => {
            assert_eq!(kind, ApiErrorKind::ServerError);
            assert!(message.contains("Bad Gateway"));
            assert!(parsed.is_none());
            assert_eq!(&raw[..], b"<html>Bad Gateway</html>");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// ----------------------------------------------------- dynamic-layer plumbing

#[test]
fn api_format_id_and_dispatch() {
    let format = GoogleGenerateContent;
    assert_eq!(format.id(), FMT);
    // parse_request goes through the same typed-layer entry point.
    let body = json!({"contents": [{"role": "user", "parts": [{"text": "hi"}]}]});
    let (ir, _) = format.parse_request(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert_eq!(ir.messages[0].text(), "hi");
    assert!(matches!(
        format.parse_request(b"[1, 2]"),
        Err(Error::Parse { .. })
    ));
}
