//! Parse-side (wire → IR) tests for the `google_generate_content` format:
//! request round-trip idempotence (§ 1), response parsing (§ 8), model
//! listing, token counting and error classification.

use std::time::Duration;

use llm_api::formats::google_generate_content::{
    GoogleGenerateContent, request_from_ir, request_to_ir, response_to_ir,
};
use llm_api::{
    ApiErrorKind, ApiFormat, ContentBlock, ConvertOptions, Error, ImageSource, ResponseMeta, Role,
    StopReason, Tool, ToolChoice, ToolOutputBlock, WarningCode, WarningSeverity,
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
    assert_eq!(
        ir2, ir,
        "re-parsing the canonical form must reproduce the IR"
    );
}

// ---------------------------------------------------------------- round trip

#[test]
fn canonical_request_fixture_round_trips_losslessly() {
    let fixture: Value = serde_json::from_str(include_str!(
        "fixtures/google_generate_content/request_canonical.json"
    ))
    .unwrap();
    let (ir, parse_warnings) = request_to_ir(&serde_json::to_vec(&fixture).unwrap()).unwrap();
    assert!(parse_warnings.is_empty(), "{parse_warnings:?}");
    let (serialized, warnings) = request_from_ir(&ir, &ConvertOptions::default()).unwrap();
    assert!(
        warnings.is_empty(),
        "round trip must be warning-free: {warnings:?}"
    );
    assert_eq!(serialized, fixture);

    // Structural expectations on the parsed IR.
    assert_eq!(ir.system.as_ref().unwrap().len(), 1);
    let roles: Vec<Role> = ir.messages.iter().map(|m| m.role).collect();
    assert_eq!(
        roles,
        vec![
            Role::User,
            Role::Assistant,
            Role::Tool,
            Role::User,
            Role::Assistant,
            Role::User
        ]
    );
    // The mixed wire turn split into Tool + User sharing one turn group.
    let g2 = ir.messages[2].turn_group_id();
    let g3 = ir.messages[3].turn_group_id();
    assert!(g2.is_some() && g2 == g3);
    assert_eq!(ir.messages[0].turn_group_id(), None);
    // Content-level unknown fields ride the first message of the turn.
    assert_eq!(
        ir.messages[0]
            .extra
            .get(FMT)
            .unwrap()
            .get("contentTag")
            .unwrap(),
        &json!("first")
    );
    // Thinking parsed natively (signature + wire marker).
    assert!(matches!(
        &ir.messages[1].content[0],
        ContentBlock::Thinking { text: Some(t), signature: Some(s), .. }
            if t == "Let me look that up." && s == "dGhvdWdodC1zaWc="
    ));
    // Tool-call thought signature rides the block's extra.
    let ContentBlock::ToolCall {
        id,
        name,
        arguments,
        extra,
        ..
    } = &ir.messages[1].content[1]
    else {
        panic!("expected tool call");
    };
    assert_eq!(id.as_deref(), Some("call-1"));
    assert_eq!(name, "get_weather");
    assert_eq!(arguments, r#"{"city":"Paris"}"#);
    assert_eq!(
        extra.get(FMT).unwrap().get("thoughtSignature").unwrap(),
        &json!("Y2FsbC1zaWc=")
    );
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
    let ContentBlock::ToolResult {
        tool_call_id,
        name,
        content,
        is_error,
        extra,
        ..
    } = &blocks[0]
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
    let ContentBlock::ToolResult {
        content, is_error, ..
    } = &blocks[1]
    else {
        panic!()
    };
    assert!(matches!(&content[0], ToolOutputBlock::Text { text, .. } if text.is_empty()));
    assert_eq!(*is_error, None);
    let ContentBlock::ToolResult { content, .. } = &blocks[2] else {
        panic!()
    };
    assert!(content.is_empty());

    // parts[] media parse after the response text (canonical order).
    let ContentBlock::ToolResult { content, .. } = &blocks[3] else {
        panic!()
    };
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
fn function_response_missing_response_warns_and_reads_empty() {
    // `response` is required upstream; its absence warns, parses as empty
    // content, and the rebuilt part carries the canonicalized
    // `"response": {}` the wire never had (so the body is not a fixed
    // point — the disclosure exists precisely because of that).
    let body = json!({
        "contents": [
            {"role": "user", "parts": [{"functionResponse": {"name": "f"}}]}
        ]
    });
    let (ir, warnings) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].code, WarningCode::MalformedToolResult);
    assert_eq!(
        warnings[0].location,
        "/contents/0/parts/0/functionResponse/response"
    );
    let ContentBlock::ToolResult {
        name,
        content,
        is_error,
        ..
    } = &ir.messages[0].content[0]
    else {
        panic!("expected tool result");
    };
    assert_eq!(name.as_deref(), Some("f"));
    assert!(content.is_empty());
    assert_eq!(*is_error, None);
    let (serialized, _) = request_from_ir(&ir, &ConvertOptions::default()).unwrap();
    assert_eq!(
        serialized["contents"][0]["parts"][0]["functionResponse"],
        json!({"name": "f", "response": {}})
    );
}

#[test]
fn function_response_missing_name_warns_and_stays_unset() {
    // `name` is required upstream; its absence warns and the IR keeps
    // `name: None` (nothing fabricated), so serializing without a
    // resolvable id fails per the § 7.2 contract.
    let body = json!({
        "contents": [
            {"role": "user", "parts": [{"functionResponse": {"response": {"output": "ok"}}}]}
        ]
    });
    let (ir, warnings) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].code, WarningCode::MalformedToolResult);
    assert_eq!(
        warnings[0].location,
        "/contents/0/parts/0/functionResponse/name"
    );
    let ContentBlock::ToolResult { name, content, .. } = &ir.messages[0].content[0] else {
        panic!("expected tool result");
    };
    assert_eq!(*name, None);
    assert!(matches!(&content[0], ToolOutputBlock::Text { text, .. } if text == "ok"));
    assert!(request_from_ir(&ir, &ConvertOptions::default()).is_err());

    // With an id matching an earlier call, re-serialization recovers the
    // name from history; the parse-side disclosure still fires.
    let body = json!({
        "contents": [
            {"role": "model", "parts": [{"functionCall": {"id": "c1", "name": "f", "args": {}}}]},
            {"role": "user", "parts": [{"functionResponse": {"id": "c1", "response": {"output": "ok"}}}]}
        ]
    });
    let (ir, warnings) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].code, WarningCode::MalformedToolResult);
    assert_eq!(
        warnings[0].location,
        "/contents/1/parts/0/functionResponse/name"
    );
    let (serialized, _) = request_from_ir(&ir, &ConvertOptions::default()).unwrap();
    assert_eq!(
        serialized["contents"][1]["parts"][0]["functionResponse"],
        json!({"id": "c1", "name": "f", "response": {"output": "ok"}})
    );
}

#[test]
fn function_response_missing_both_fields_warns_per_field() {
    let body = json!({
        "contents": [
            {"role": "user", "parts": [{"functionResponse": {}}]}
        ]
    });
    let (ir, warnings) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert_eq!(warnings.len(), 2, "{warnings:?}");
    assert!(
        warnings
            .iter()
            .all(|w| w.code == WarningCode::MalformedToolResult)
    );
    assert_eq!(
        warnings[0].location,
        "/contents/0/parts/0/functionResponse/name"
    );
    assert_eq!(
        warnings[1].location,
        "/contents/0/parts/0/functionResponse/response"
    );
    let ContentBlock::ToolResult { name, content, .. } = &ir.messages[0].content[0] else {
        panic!("expected tool result");
    };
    assert_eq!(*name, None);
    assert!(content.is_empty());
}

#[test]
fn legacy_function_role_parses_as_tool_turn() {
    let body = json!({
        "contents": [
            {"role": "function", "parts": [{"functionResponse": {"name": "f", "response": {"output": "ok"}}}]}
        ]
    });
    let (ir, warnings) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert_eq!(ir.messages[0].role, Role::Tool);
    // "function" is outside the documented role union → user-side + warning.
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0].code, WarningCode::MalformedField);
    assert_eq!(warnings[0].location, "/contents/0/role");
    // Canonicalizes to the documented user turn on the way back.
    let (serialized, _) = request_from_ir(&ir, &ConvertOptions::default()).unwrap();
    assert_eq!(serialized["contents"][0]["role"], "user");
}

#[test]
fn out_of_schema_content_role_warns_and_stays_user() {
    // The documented role union is closed: "user" / "model", optionally
    // absent. Anything else keeps the user-side behavior, with a warning.
    let body = json!({
        "contents": [{"role": "assistant", "parts": [{"text": "hi"}]}]
    });
    let (ir, warnings) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].code, WarningCode::MalformedField);
    assert_eq!(warnings[0].location, "/contents/0/role");
    assert!(warnings[0].message.contains("assistant"));
    assert_eq!(ir.messages[0].role, Role::User);
    // Canonicalizes to role "user" on the way back.
    let (serialized, _) = request_from_ir(&ir, &ConvertOptions::default()).unwrap();
    assert_eq!(serialized["contents"][0]["role"], "user");

    // Documented values stay warning-free — including the absent default.
    for content in [
        json!({"parts": [{"text": "hi"}]}),
        json!({"role": "user", "parts": [{"text": "hi"}]}),
        json!({"role": "model", "parts": [{"text": "hi"}]}),
    ] {
        let body = json!({"contents": [content]});
        let (_, warnings) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
    }
}

#[test]
fn text_only_system_instruction_stays_in_request_system() {
    let body = json!({
        "systemInstruction": {"parts": [{"text": "a"}, {"text": "b", "partTag": 1}]},
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}]
    });
    let (ir, warnings) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(ir.system.as_ref().unwrap().len(), 2);
    assert!(ir.messages.iter().all(|m| m.role != Role::System));
    assert_fixed_point(&body);
}

#[test]
fn snake_case_request_spellings_parse_into_modeled_fields() {
    // proto3 JSON accepts snake_case next to the canonical camelCase; every
    // modeled multi-word request field is aliased. Minimal check: the field
    // lands in the IR — no extra residue, no warnings.
    let (ir, warnings) = request_to_ir(
        &serde_json::to_vec(&json!({
            "contents": [],
            "generation_config": {"temperature": 0.3},
        }))
        .unwrap(),
    )
    .unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(ir.temperature, Some(0.3));
    assert!(ir.extra.is_empty());

    // The same logical request in either spelling parses to the same IR and
    // canonicalizes to the same camelCase body.
    let camel = json!({
        "systemInstruction": {"parts": [{"text": "sys"}]},
        "contents": [
            {"role": "user", "parts": [
                {"text": "look"},
                {"inlineData": {"mimeType": "image/png", "data": "AAAA"}},
                {"fileData": {"mimeType": "image/png", "fileUri": "https://files/x"}},
            ]},
            {"role": "model", "parts": [
                {"text": "t", "thought": true, "thoughtSignature": "sig"},
                {"functionCall": {"id": "c1", "name": "f", "args": {"a": 1}}},
            ]},
            {"role": "user", "parts": [
                {"functionResponse": {"id": "c1", "name": "f",
                    "response": {"output": "ok"},
                    "parts": [{"inlineData": {"mimeType": "image/png", "data": "BBBB"}}]}},
            ]},
        ],
        "tools": [{"functionDeclarations": [
            {"name": "f", "parametersJsonSchema": {"type": "object"}},
        ]}],
        "toolConfig": {"functionCallingConfig": {
            "mode": "ANY", "allowedFunctionNames": ["f"],
        }},
        "generationConfig": {
            "stopSequences": ["x"],
            "responseMimeType": "application/json",
            "responseJsonSchema": {"type": "object"},
            "maxOutputTokens": 100,
            "temperature": 0.3,
            "topP": 0.9,
            "topK": 40,
            "seed": 7,
            "presencePenalty": 0.1,
            "frequencyPenalty": 0.2,
            "thinkingConfig": {"includeThoughts": true, "thinkingLevel": "LOW"},
        },
    });
    let snake = json!({
        "system_instruction": {"parts": [{"text": "sys"}]},
        "contents": [
            {"role": "user", "parts": [
                {"text": "look"},
                {"inline_data": {"mime_type": "image/png", "data": "AAAA"}},
                {"file_data": {"mime_type": "image/png", "file_uri": "https://files/x"}},
            ]},
            {"role": "model", "parts": [
                {"text": "t", "thought": true, "thought_signature": "sig"},
                {"function_call": {"id": "c1", "name": "f", "args": {"a": 1}}},
            ]},
            {"role": "user", "parts": [
                {"function_response": {"id": "c1", "name": "f",
                    "response": {"output": "ok"},
                    "parts": [{"inline_data": {"mime_type": "image/png", "data": "BBBB"}}]}},
            ]},
        ],
        "tools": [{"function_declarations": [
            {"name": "f", "parameters_json_schema": {"type": "object"}},
        ]}],
        "tool_config": {"function_calling_config": {
            "mode": "ANY", "allowed_function_names": ["f"],
        }},
        "generation_config": {
            "stop_sequences": ["x"],
            "response_mime_type": "application/json",
            "response_json_schema": {"type": "object"},
            "max_output_tokens": 100,
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 40,
            "seed": 7,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
            "thinking_config": {"include_thoughts": true, "thinking_level": "LOW"},
        },
    });
    let (ir_c, ws_c) = request_to_ir(&serde_json::to_vec(&camel).unwrap()).unwrap();
    let (ir_s, ws_s) = request_to_ir(&serde_json::to_vec(&snake).unwrap()).unwrap();
    assert_eq!(ir_s, ir_c);
    assert_eq!(ws_s, ws_c);
    let (out_c, _) = request_from_ir(&ir_c, &ConvertOptions::default()).unwrap();
    let (out_s, _) = request_from_ir(&ir_s, &ConvertOptions::default()).unwrap();
    assert_eq!(out_s, out_c);
    // Serialization always emits the canonical camelCase spelling.
    assert!(out_s.get("generationConfig").is_some());
    assert!(out_s.get("generation_config").is_none());
}

#[test]
fn duplicate_camel_and_snake_spellings_fail_the_parse() {
    // Pinned current behavior: both spellings of one field in the same
    // object hit serde's duplicate-field check — a loud parse error, not a
    // silent pick of either value.
    let body = json!({
        "contents": [],
        "generationConfig": {"temperature": 0.1},
        "generation_config": {"temperature": 0.2},
    });
    let err = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap_err();
    match err {
        Error::Parse { message, .. } => {
            assert!(message.contains("duplicate field"), "{message}");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn system_instruction_with_non_text_parts_parses_as_leading_system_message() {
    // Request.system is Text-only, so an out-of-schema part forces the whole
    // instruction into a leading System message (Text + same-format Opaque,
    // part order kept); § 7.1 hoists it back on serialization — nothing is
    // lost.
    let body = json!({
        "systemInstruction": {
            "parts": [
                {"text": "You are terse.", "partTag": 1},
                {"inlineData": {"mimeType": "image/png", "data": "cGl4"}},
                {"unknownPart": 1},
                {"text": "Answer in French."}
            ],
            "siTag": "s1"
        },
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}]
    });
    let (ir, warnings) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    // One MalformedField per non-text part, and nothing else.
    assert_eq!(warnings.len(), 2, "{warnings:?}");
    assert!(
        warnings
            .iter()
            .all(|w| w.code == WarningCode::MalformedField)
    );
    assert_eq!(warnings[0].location, "/systemInstruction/parts/1");
    assert_eq!(warnings[1].location, "/systemInstruction/parts/2");

    assert_eq!(ir.system, None);
    let sys = &ir.messages[0];
    assert_eq!(sys.role, Role::System);
    assert_eq!(ir.messages[1].role, Role::User);
    assert!(matches!(&sys.content[0], ContentBlock::Text { text, .. } if text == "You are terse."));
    assert!(matches!(
        &sys.content[1],
        ContentBlock::Opaque { format, value, .. }
            if format == FMT && value.get("inlineData").is_some()
    ));
    assert!(matches!(
        &sys.content[2],
        ContentBlock::Opaque { format, value, .. }
            if format == FMT && value.get("unknownPart").is_some()
    ));
    assert!(
        matches!(&sys.content[3], ContentBlock::Text { text, .. } if text == "Answer in French.")
    );
    // Instruction-level unknown fields ride the message extra.
    assert_eq!(
        sys.extra.get(FMT).unwrap().get("siTag").unwrap(),
        &json!("s1")
    );

    // The hoist rule makes the wire round-trip the identity, warning-free.
    let (serialized, ser_warnings) = request_from_ir(&ir, &ConvertOptions::default()).unwrap();
    assert!(ser_warnings.is_empty(), "{ser_warnings:?}");
    assert_eq!(serialized, body);
    let (ir2, _) = request_to_ir(&serde_json::to_vec(&serialized).unwrap()).unwrap();
    assert_eq!(ir2, ir);

    // An instruction with no text part at all survives the same way.
    let body = json!({
        "systemInstruction": {"parts": [{"unknownPart": 1}]},
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}]
    });
    let (ir, _) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert_eq!(ir.system, None);
    assert_eq!(ir.messages[0].role, Role::System);
    let (serialized, _) = request_from_ir(&ir, &ConvertOptions::default()).unwrap();
    assert_eq!(serialized, body);
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

    // Non-object args parse leniently but degrade the call: the verbatim
    // value cannot re-serialize (§ 4.5 arguments contract), so the warning
    // is a semantic MalformedToolCall.
    let body = json!({
        "contents": [{"role": "model", "parts": [{"functionCall": {"name": "f", "args": 5}}]}]
    });
    let (ir, warnings) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].code, WarningCode::MalformedToolCall);
    assert_eq!(warnings[0].severity, WarningSeverity::Semantic);
    assert_eq!(
        warnings[0].location,
        "/contents/0/parts/0/functionCall/args"
    );
    assert!(matches!(
        &ir.messages[0].content[0],
        ContentBlock::ToolCall { arguments, .. } if arguments == "5"
    ));
    assert!(request_from_ir(&ir, &ConvertOptions::default()).is_err());
}

#[test]
fn function_call_missing_name_warns_and_defaults_empty() {
    // `name` is required upstream; a missing one warns instead of being
    // silently defaulted, and parses (and re-serializes) as `""`.
    let body = json!({
        "contents": [{"role": "model", "parts": [{"functionCall": {"args": {}}}]}]
    });
    let (ir, warnings) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].code, WarningCode::MalformedToolCall);
    assert_eq!(
        warnings[0].location,
        "/contents/0/parts/0/functionCall/name"
    );
    assert!(matches!(
        &ir.messages[0].content[0],
        ContentBlock::ToolCall { id: None, name, arguments, .. }
            if name.is_empty() && arguments == "{}"
    ));
    let (serialized, _) = request_from_ir(&ir, &ConvertOptions::default()).unwrap();
    assert_eq!(
        serialized["contents"][0]["parts"][0]["functionCall"],
        json!({"name": "", "args": {}})
    );

    // The response side reports through `Response.warnings`.
    let body = json!({
        "candidates": [{
            "content": {"parts": [{"functionCall": {"args": {"a": 1}}}], "role": "model"},
            "finishReason": "STOP",
        }]
    });
    let resp = response_to_ir(&serde_json::to_vec(&body).unwrap(), &meta()).unwrap();
    assert_eq!(resp.warnings.len(), 1, "{:?}", resp.warnings);
    assert_eq!(resp.warnings[0].code, WarningCode::MalformedToolCall);
    assert_eq!(
        resp.warnings[0].location,
        "/candidates/0/content/parts/0/functionCall/name"
    );
    assert!(matches!(
        &resp.message.content[0],
        ContentBlock::ToolCall { name, .. } if name.is_empty()
    ));
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
    assert!(matches!(
        &ir.messages[0].content[0],
        ContentBlock::Opaque { .. }
    ));
    assert!(matches!(
        &ir.messages[0].content[1],
        ContentBlock::Opaque { .. }
    ));
    assert!(matches!(
        &ir.messages[0].content[2],
        ContentBlock::Text { .. }
    ));
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
    // Splitting is a deliberate, silent canonicalization: the split form is
    // upstream-equivalent (the official tool-combination examples list the
    // combined tools as separate entries), so neither direction warns.
    let body = json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "tools": [{"functionDeclarations": [{"name": "f"}], "googleSearch": {}}]
    });
    let (ir, parse_warnings) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert!(parse_warnings.is_empty(), "{parse_warnings:?}");
    let tools = ir.tools.as_ref().unwrap();
    assert_eq!(tools.len(), 2);
    let (serialized, warnings) = request_from_ir(&ir, &ConvertOptions::default()).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
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
    assert_eq!(
        serialized,
        json!({"contents": [{"role": "user", "parts": [{"text": "hi"}]}]})
    );
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
    assert!(matches!(
        ir.output_format,
        Some(llm_api::OutputFormat::JsonObject)
    ));
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
    assert_eq!(
        resp.text(),
        "AI learns patterns from data to make predictions."
    );
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
    let ContentBlock::ToolCall {
        id,
        name,
        arguments,
        extra,
        ..
    } = &resp.message.content[0]
    else {
        panic!("expected tool call");
    };
    assert_eq!(id.as_deref(), Some("call-9"));
    assert_eq!(name, "set_light_color");
    assert_eq!(arguments, r#"{"rgb_hex":"ff0000"}"#);
    assert_eq!(
        extra.get(FMT).unwrap().get("thoughtSignature").unwrap(),
        &json!("c2lnLTk=")
    );
    // Google reports STOP for function calls; § 8 normalizes to ToolUse.
    assert_eq!(resp.stop_reason, Some(StopReason::ToolUse));
    let usage = resp.usage.as_ref().unwrap();
    assert_eq!(usage.input_tokens, 21);
    assert_eq!(usage.output_tokens, 35, "candidates + thoughts");
    assert_eq!(usage.reasoning_tokens, Some(30));
    assert_eq!(usage.visible_output_tokens(), 5);
}

#[test]
fn usage_folds_tool_use_prompt_tokens_into_input() {
    // Live-verified shape (code_execution): toolUsePromptTokenCount is
    // outside promptTokenCount but inside totalTokenCount, so input folds
    // it in and input + output == total holds.
    let body = json!({
        "candidates": [{
            "content": {"parts": [{"text": "2178588618496"}], "role": "model"},
            "finishReason": "STOP",
        }],
        "usageMetadata": {
            "promptTokenCount": 74,
            "candidatesTokenCount": 103,
            "thoughtsTokenCount": 293,
            "toolUsePromptTokenCount": 155,
            "totalTokenCount": 625,
        }
    });
    let resp = response_to_ir(&serde_json::to_vec(&body).unwrap(), &meta()).unwrap();
    let usage = resp.usage.as_ref().unwrap();
    assert_eq!(usage.input_tokens, 74 + 155);
    assert_eq!(usage.output_tokens, 103 + 293);
    assert_eq!(usage.total_tokens, Some(625));
    assert_eq!(
        usage.input_tokens + usage.output_tokens,
        usage.total_tokens.unwrap()
    );
    // The provider field stays visible in raw.
    assert_eq!(
        usage.raw.as_ref().unwrap()["toolUsePromptTokenCount"],
        json!(155)
    );
}

#[test]
fn usage_boundary_values_clamp_and_saturate() {
    let candidate = json!({
        "content": {"parts": [{"text": "x"}], "role": "model"},
        "finishReason": "STOP",
    });

    // Absurdly large counts must not overflow: the output sum saturates.
    let body = json!({
        "candidates": [candidate.clone()],
        "usageMetadata": {
            "promptTokenCount": 1,
            "candidatesTokenCount": i64::MAX,
            "thoughtsTokenCount": i64::MAX,
        }
    });
    let resp = response_to_ir(&serde_json::to_vec(&body).unwrap(), &meta()).unwrap();
    let usage = resp.usage.as_ref().unwrap();
    let max = u64::try_from(i64::MAX).unwrap();
    assert_eq!(usage.output_tokens, max * 2);
    assert_eq!(usage.reasoning_tokens, Some(max));

    // Negative counts (never sent by well-behaved servers) clamp to zero.
    let body = json!({
        "candidates": [candidate],
        "usageMetadata": {
            "promptTokenCount": -1,
            "candidatesTokenCount": -5,
            "thoughtsTokenCount": 7,
            "cachedContentTokenCount": -3,
            "totalTokenCount": -2,
        }
    });
    let resp = response_to_ir(&serde_json::to_vec(&body).unwrap(), &meta()).unwrap();
    let usage = resp.usage.as_ref().unwrap();
    assert_eq!(usage.input_tokens, 0);
    assert_eq!(usage.output_tokens, 7);
    assert_eq!(usage.reasoning_tokens, Some(7));
    assert_eq!(usage.cache_read_tokens, Some(0));
    assert_eq!(usage.total_tokens, Some(0));
}

#[test]
fn malformed_usage_metadata_degrades_instead_of_failing_the_response() {
    let candidate = json!({
        "content": {"parts": [{"text": "billed output"}], "role": "model"},
        "finishReason": "STOP",
    });
    // A float count cannot be decoded; the already-billed response must
    // still parse — usage degrades to None with a warning, never to
    // silently zeroed counts.
    for usage_metadata in [json!({"promptTokenCount": 3.5}), json!("not an object")] {
        let body = json!({
            "candidates": [candidate.clone()],
            "usageMetadata": usage_metadata,
        });
        let resp = response_to_ir(&serde_json::to_vec(&body).unwrap(), &meta()).unwrap();
        assert_eq!(resp.text(), "billed output");
        assert_eq!(resp.stop_reason, Some(StopReason::EndTurn));
        assert!(resp.usage.is_none());
        let w = resp
            .warnings
            .iter()
            .find(|w| w.code == WarningCode::MalformedField)
            .unwrap_or_else(|| panic!("expected warning, got {:?}", resp.warnings));
        assert_eq!(w.location, "/usageMetadata");
        // The verbatim value stays reachable through the raw body.
        assert!(resp.raw.as_ref().unwrap().get("usageMetadata").is_some());
    }
}

#[test]
fn thinking_response_parses_thought_parts() {
    let body = include_str!("fixtures/google_generate_content/response_thinking.json");
    let resp = response_to_ir(body.as_bytes(), &meta()).unwrap();
    let ContentBlock::Thinking {
        text,
        signature,
        extra,
        ..
    } = &resp.message.content[0]
    else {
        panic!("expected thinking");
    };
    assert!(text.as_ref().unwrap().starts_with("**Considering"));
    assert_eq!(signature.as_deref(), Some("dGhpbmstc2ln"));
    assert_eq!(
        extra.get(FMT).unwrap().get("thought").unwrap(),
        &json!(true)
    );
    assert!(matches!(
        &resp.message.content[1],
        ContentBlock::Text { .. }
    ));
    assert_eq!(resp.stop_reason, Some(StopReason::EndTurn));
    assert_eq!(resp.usage.as_ref().unwrap().output_tokens, 52);

    // The parsed assistant message replays into a request natively.
    let req = llm_api::Request::with_messages(vec![
        llm_api::Message::user_text("haiku please"),
        resp.message.clone(),
    ]);
    let (serialized, warnings) = request_from_ir(&req, &ConvertOptions::default()).unwrap();
    assert!(
        warnings.is_empty(),
        "replay must be warning-free: {warnings:?}"
    );
    assert_eq!(
        serialized["contents"][1]["parts"][0]["thoughtSignature"],
        json!("dGhpbmstc2ln")
    );
    assert_eq!(
        serialized["contents"][1]["parts"][0]["thought"],
        json!(true)
    );
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
    assert_eq!(
        resp.raw.as_ref().unwrap()["candidates"][1]["content"]["parts"][0]["text"],
        "Second answer."
    );
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
        (
            "MALFORMED_FUNCTION_CALL",
            StopReason::Other("MALFORMED_FUNCTION_CALL".to_owned()),
        ),
        (
            "MISSING_THOUGHT_SIGNATURE",
            StopReason::Other("MISSING_THOUGHT_SIGNATURE".to_owned()),
        ),
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
        ContentBlock::Image {
            source: ImageSource::Base64 { .. },
            ..
        }
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
    assert_eq!(
        count.raw.as_ref().unwrap()["promptTokensDetails"][1]["modality"],
        "IMAGE"
    );
    assert!(count.warnings.is_empty());

    // proto3 omits zero counts: an empty body means zero tokens, and an
    // explicit zero parses the same.
    let count = format.parse_count_tokens_response(b"{}").unwrap();
    assert_eq!(count.input_tokens, 0);
    let count = format
        .parse_count_tokens_response(br#"{"totalTokens": 0}"#)
        .unwrap();
    assert_eq!(count.input_tokens, 0);

    // A negative count is malformed — never a silent zero.
    let err = format
        .parse_count_tokens_response(br#"{"totalTokens": -5}"#)
        .unwrap_err();
    match err {
        Error::Parse { message, .. } => assert!(message.contains("negative"), "{message}"),
        other => panic!("unexpected: {other:?}"),
    }
}

// -------------------------------------------------------------------- errors

#[test]
fn error_status_maps_grpc_codes() {
    let format = GoogleGenerateContent;
    let body = include_str!("fixtures/google_generate_content/error_rate_limit.json");
    let mut headers = http::HeaderMap::new();
    headers.insert(
        http::header::RETRY_AFTER,
        http::HeaderValue::from_static("7"),
    );
    let err = format.parse_error(429, &headers, body.as_bytes());
    match err {
        Error::Api {
            status,
            kind,
            message,
            parsed,
            retry_after,
            truncated,
            ..
        } => {
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
        let body = json!({"error": {"code": status, "message": "m", "status": grpc}}).to_string();
        let err = format.parse_error(status, &http::HeaderMap::new(), body.as_bytes());
        match err {
            Error::Api { kind, .. } => assert_eq!(kind, expected, "for {grpc}"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    // Non-JSON bodies degrade to generic classification with the raw kept.
    let err = format.parse_error(502, &http::HeaderMap::new(), b"<html>Bad Gateway</html>");
    match err {
        Error::Api {
            kind,
            message,
            parsed,
            raw,
            ..
        } => {
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
    let (ir, _) = format
        .parse_request(&serde_json::to_vec(&body).unwrap())
        .unwrap();
    assert_eq!(ir.messages[0].text(), "hi");
    assert!(matches!(
        format.parse_request(b"[1, 2]"),
        Err(Error::Parse { .. })
    ));
}
