//! Parse-side tests for `anthropic_messages`: request round-trips, response
//! parsing, models, token counting and error mapping.

use llm_api::formats::anthropic_messages::AnthropicMessages;
use llm_api::{
    ApiErrorKind, ApiFormat, BuildCtx, CallMode, ContentBlock, ConversionError, ConvertOptions,
    Effort, EndpointUrl, Error, Message, OutputFormat, Request, ResponseMeta, Role, StopReason,
    Tool, ToolChoice, ToolOutputBlock, WarningCode, WarningSeverity,
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

fn ctx_for(model: &str) -> BuildCtx {
    BuildCtx::new(
        EndpointUrl::base("https://api.anthropic.com/v1").unwrap(),
        model,
        CallMode::Unary,
    )
}

fn rebuild(req: &Request, model: &str) -> Value {
    let built = AnthropicMessages
        .build_request(req, &ctx_for(model))
        .unwrap();
    serde_json::from_slice(&built.body).unwrap()
}

/// Parses a wire request, re-serializes it and asserts the § 1 guarantee:
/// the canonical fixture is a fixed point of parse→serialize.
fn assert_round_trip(bytes: &[u8]) -> Request {
    let original: Value = serde_json::from_slice(bytes).unwrap();
    let model = original["model"].as_str().unwrap().to_owned();
    let (req, _) = AnthropicMessages.parse_request(bytes).unwrap();
    let first = rebuild(&req, &model);
    assert_eq!(
        first, original,
        "first pass must reproduce the canonical fixture"
    );
    let (req2, _) = AnthropicMessages
        .parse_request(&serde_json::to_vec(&first).unwrap())
        .unwrap();
    let second = rebuild(&req2, &model);
    assert_eq!(second, first, "second pass must be idempotent");
    req
}

#[test]
fn round_trip_full_request() {
    let req = assert_round_trip(&fixture("request_full.json"));
    // Spot-check the modeled fields.
    assert_eq!(req.max_output_tokens, Some(1024));
    assert_eq!(req.temperature, Some(0.5));
    assert_eq!(req.top_p, Some(0.9));
    assert_eq!(req.top_k, Some(40));
    assert_eq!(req.stop_sequences.as_deref(), Some(&["END".to_owned()][..]));
    assert_eq!(req.parallel_tool_calls, Some(false));
    assert_eq!(req.tool_choice, Some(ToolChoice::Auto));
    let metadata = req.metadata.as_ref().unwrap();
    assert_eq!(metadata.get("user_id"), Some(&json!("u-1")));
    assert!(
        !metadata.contains_key("session"),
        "unknown metadata keys ride extra"
    );
    let reasoning = req.reasoning.as_ref().unwrap();
    assert_eq!(reasoning.enabled, Some(true));
    assert_eq!(reasoning.include_thoughts, Some(false));
    assert_eq!(reasoning.effort, Some(Effort::High));
    assert!(matches!(
        req.output_format,
        Some(OutputFormat::JsonSchema { .. })
    ));
    let system = req.system.as_ref().unwrap();
    assert!(matches!(&system[0], ContentBlock::Text { text, .. } if text == "Be helpful."));
    let tools = req.tools.as_ref().unwrap();
    match &tools[0] {
        Tool::Function(f) => {
            assert_eq!(f.name, "get_weather");
            assert_eq!(f.strict, Some(true));
            assert!(f.cache.is_some());
            assert!(f.parameters.is_some());
        }
        other => panic!("unexpected tool: {other:?}"),
    }
    assert!(matches!(&tools[1], Tool::Opaque { format, .. } if format == FMT));
    // Unknown top-level field mirrored into the request extra.
    assert_eq!(
        req.extra.get(FMT).unwrap().get("service_tier"),
        Some(&json!("auto"))
    );
}

#[test]
fn round_trip_system_array() {
    let req = assert_round_trip(&fixture("request_system_array.json"));
    let system = req.system.as_ref().unwrap();
    assert_eq!(system.len(), 2);
    match &system[1] {
        ContentBlock::Text { cache, .. } => {
            assert_eq!(cache.as_ref().unwrap().ttl.as_deref(), Some("1h"));
        }
        other => panic!("unexpected block: {other:?}"),
    }
}

#[test]
fn system_array_with_non_text_entry_parses_as_leading_system_message() {
    // Any non-text entry turns the whole system channel into a marker-less
    // leading System message (order kept, non-text entries opaque); the
    // combine rule hoists it back, so the wire round trip is the identity.
    let body = json!({
        "model": "m", "max_tokens": 5,
        "system": [
            {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
            {"type": "unknown_block", "foo": 1},
            {"type": "text", "text": "b"},
        ],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    });
    let bytes = serde_json::to_vec(&body).unwrap();
    let (req, warnings) = AnthropicMessages.parse_request(&bytes).unwrap();
    assert!(req.system.is_none());
    assert_eq!(req.messages.len(), 2);
    let lead = &req.messages[0];
    assert_eq!(lead.role, Role::System);
    assert!(lead.round_trip.is_none(), "no marker: canonical hoisting");
    assert_eq!(lead.content.len(), 3);
    match &lead.content[0] {
        ContentBlock::Text { text, cache, .. } => {
            assert_eq!(text, "a");
            assert!(cache.is_some());
        }
        other => panic!("unexpected block: {other:?}"),
    }
    assert!(matches!(
        &lead.content[1],
        ContentBlock::Opaque { format, value, .. }
            if format == FMT && value["type"] == "unknown_block"
    ));
    assert!(matches!(&lead.content[2], ContentBlock::Text { text, .. } if text == "b"));
    let w = warnings
        .iter()
        .find(|w| w.code == WarningCode::MalformedField)
        .unwrap();
    assert_eq!(w.location, "/system/1");
    // The fixed-point/idempotence helper asserts the identity round trip.
    assert_round_trip(&bytes);

    // An all-opaque channel takes the same path.
    let only_opaque = json!({
        "model": "m", "max_tokens": 5,
        "system": [{"type": "unknown_block", "foo": 1}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    });
    let req2 = assert_round_trip(&serde_json::to_vec(&only_opaque).unwrap());
    assert!(req2.system.is_none());
    assert_eq!(req2.messages[0].role, Role::System);

    // A system entry that itself carries a `role` field stays a system
    // entry (the verbatim wire-message passthrough excludes System-role
    // messages), so it hoists back instead of becoming a wire message.
    let role_bearing = json!({
        "model": "m", "max_tokens": 5,
        "system": [{"type": "unknown_block", "role": "weird"}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    });
    let req3 = assert_round_trip(&serde_json::to_vec(&role_bearing).unwrap());
    assert_eq!(req3.messages[0].role, Role::System);
}

#[test]
fn system_array_with_non_text_entry_coexists_with_in_array_system() {
    // The synthetic leading System message (no marker) hoists while the
    // wire's own leading in-array system message (marker) stays in place.
    let body = json!({
        "model": "m", "max_tokens": 5,
        "system": [
            {"type": "text", "text": "a"},
            {"type": "unknown_block", "foo": 1},
        ],
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "in-array"}]},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ],
    });
    let req = assert_round_trip(&serde_json::to_vec(&body).unwrap());
    assert!(req.system.is_none());
    assert_eq!(req.messages[0].role, Role::System);
    assert!(req.messages[0].round_trip.is_none());
    assert_eq!(req.messages[1].role, Role::System);
    assert!(req.messages[1].round_trip.is_some());
}

#[test]
fn plain_system_forms_still_parse_into_request_system() {
    // String form.
    let body = json!({
        "model": "m", "max_tokens": 5,
        "system": "Be helpful.",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    });
    let req = assert_round_trip(&serde_json::to_vec(&body).unwrap());
    let system = req.system.as_ref().unwrap();
    assert!(matches!(&system[0], ContentBlock::Text { text, .. } if text == "Be helpful."));
    assert_eq!(req.messages.len(), 1);

    // All-text array form.
    let body2 = json!({
        "model": "m", "max_tokens": 5,
        "system": [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    });
    let req2 = assert_round_trip(&serde_json::to_vec(&body2).unwrap());
    assert_eq!(req2.system.as_ref().unwrap().len(), 2);
    assert_eq!(req2.messages.len(), 1);
}

#[test]
fn round_trip_in_array_system_keeps_placement() {
    let req = assert_round_trip(&fixture("request_in_array_system.json"));
    // The leading in-array system message stays a System message (with the
    // placement marker), not hoisted into Request.system.
    assert!(req.system.is_none());
    assert_eq!(req.messages[0].role, Role::System);
    assert_eq!(req.messages[2].role, Role::System);
}

#[test]
fn round_trip_mixed_tool_turn() {
    let req = assert_round_trip(&fixture("request_mixed_tool_turn.json"));
    // The mixed wire user turn splits into Tool + User messages sharing a
    // turn group.
    assert_eq!(req.messages.len(), 4);
    assert_eq!(req.messages[2].role, Role::Tool);
    assert_eq!(req.messages[3].role, Role::User);
    let g2 = req.messages[2].turn_group_id().unwrap();
    let g3 = req.messages[3].turn_group_id().unwrap();
    assert_eq!(g2, g3);
    // Tool results parsed with ids and content shapes.
    match &req.messages[2].content[0] {
        ContentBlock::ToolResult {
            tool_call_id,
            content,
            ..
        } => {
            assert_eq!(tool_call_id.as_deref(), Some("toolu_1"));
            assert!(matches!(&content[0], ToolOutputBlock::Text { text, .. } if text == "sunny"));
        }
        other => panic!("unexpected block: {other:?}"),
    }
    match &req.messages[2].content[1] {
        ContentBlock::ToolResult {
            content, is_error, ..
        } => {
            assert_eq!(content.len(), 2);
            assert_eq!(*is_error, Some(false));
        }
        other => panic!("unexpected block: {other:?}"),
    }
}

#[test]
fn round_trip_block_zoo() {
    let req = assert_round_trip(&fixture("request_blocks.json"));
    // Unmodeled document block is opaque, in place.
    assert!(matches!(
        &req.messages[0].content[0],
        ContentBlock::Opaque { format, .. } if format == FMT
    ));
    // Thinking and redacted thinking.
    match &req.messages[1].content[0] {
        ContentBlock::Thinking {
            text, signature, ..
        } => {
            assert_eq!(text.as_deref(), Some("reasoning..."));
            assert_eq!(signature.as_deref(), Some("sig-1"));
        }
        other => panic!("unexpected block: {other:?}"),
    }
    match &req.messages[1].content[1] {
        ContentBlock::Thinking {
            text,
            signature,
            extra,
            ..
        } => {
            assert_eq!(*text, None);
            assert_eq!(signature.as_deref(), Some("opaque-blob"));
            assert_eq!(extra.get(FMT).unwrap().get("redacted"), Some(&json!(true)));
        }
        other => panic!("unexpected block: {other:?}"),
    }
    // Tool call keeps `caller` in extra and the arguments as a string.
    match &req.messages[1].content[2] {
        ContentBlock::ToolCall {
            id,
            arguments,
            extra,
            ..
        } => {
            assert_eq!(id.as_deref(), Some("toolu_9"));
            assert_eq!(
                serde_json::from_str::<Value>(arguments).unwrap(),
                json!({"q": "x"})
            );
            assert!(extra.get(FMT).unwrap().contains_key("caller"));
        }
        other => panic!("unexpected block: {other:?}"),
    }
    // Nested cache_control on a tool-result text block is not modeled; it
    // rides the nested block's extra instead.
    match &req.messages[2].content[0] {
        ContentBlock::ToolResult { content, cache, .. } => {
            assert!(
                cache.is_some(),
                "the tool_result's own cache_control is modeled"
            );
            match &content[0] {
                ToolOutputBlock::Text { cache, extra, .. } => {
                    assert!(cache.is_none());
                    assert!(extra.get(FMT).unwrap().contains_key("cache_control"));
                }
                other => panic!("unexpected block: {other:?}"),
            }
            assert!(matches!(&content[2], ToolOutputBlock::Opaque { .. }));
        }
        other => panic!("unexpected block: {other:?}"),
    }
}

#[test]
fn non_canonical_forms_converge() {
    // String message content and single-element tool-result arrays are
    // canonicalized on the first pass; the result is then a fixed point.
    let body = json!({
        "model": "m",
        "max_tokens": 5,
        "system": [{"type": "text", "text": "sys"}],
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t", "content": [{"type": "text", "text": "ok"}]}
            ]},
        ],
    });
    let (req, _) = AnthropicMessages
        .parse_request(&serde_json::to_vec(&body).unwrap())
        .unwrap();
    let first = rebuild(&req, "m");
    assert_eq!(
        first,
        json!({
            "model": "m",
            "max_tokens": 5,
            "system": "sys",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t", "content": "ok"}
                ]},
            ],
        })
    );
    let (req2, _) = AnthropicMessages
        .parse_request(&serde_json::to_vec(&first).unwrap())
        .unwrap();
    assert_eq!(rebuild(&req2, "m"), first);
}

#[test]
fn parse_thinking_config_variants() {
    let body = json!({
        "model": "m", "max_tokens": 5,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "disabled"},
    });
    let (req, _) = AnthropicMessages
        .parse_request(&serde_json::to_vec(&body).unwrap())
        .unwrap();
    assert_eq!(req.reasoning.as_ref().unwrap().enabled, Some(false));

    let body2 = json!({
        "model": "m", "max_tokens": 5,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "adaptive", "display": "summarized"},
    });
    let (req2, _) = AnthropicMessages
        .parse_request(&serde_json::to_vec(&body2).unwrap())
        .unwrap();
    let reasoning = req2.reasoning.as_ref().unwrap();
    assert_eq!(reasoning.enabled, Some(true));
    assert_eq!(reasoning.include_thoughts, Some(true));

    // Unknown thinking variant round-trips through the request extra.
    let body3 = json!({
        "model": "m", "max_tokens": 5,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "thinking": {"type": "hyperdrive", "warp": 9},
    });
    let (req3, _) = AnthropicMessages
        .parse_request(&serde_json::to_vec(&body3).unwrap())
        .unwrap();
    assert!(req3.reasoning.is_none());
    let rebuilt = rebuild(&req3, "m");
    assert_eq!(
        rebuilt["thinking"],
        json!({"type": "hyperdrive", "warp": 9})
    );
}

#[test]
fn disable_parallel_tool_use_false_round_trips() {
    let body = json!({
        "model": "m", "max_tokens": 5,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "tools": [{"name": "f", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "any", "disable_parallel_tool_use": false},
    });
    let (req, _) = AnthropicMessages
        .parse_request(&serde_json::to_vec(&body).unwrap())
        .unwrap();
    assert_eq!(req.tool_choice, Some(ToolChoice::Required));
    assert_eq!(req.parallel_tool_calls, Some(true));
    assert_eq!(rebuild(&req, "m"), body);
}

#[test]
fn parse_rejects_invalid_json() {
    let err = AnthropicMessages.parse_request(b"{not json").unwrap_err();
    assert!(matches!(err, Error::Parse { .. }));
}

#[test]
fn parse_unknown_role_kept_verbatim_and_round_trips() {
    // The whole wire message (role string, shorthand content, unknown
    // fields) is kept as a lone opaque block and re-emitted untouched.
    let body = json!({
        "model": "m", "max_tokens": 5,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "critic", "content": "meh", "weight": 2},
            {"role": "user", "content": [{"type": "text", "text": "next"}]},
        ],
    });
    let (req, warnings) = AnthropicMessages
        .parse_request(&serde_json::to_vec(&body).unwrap())
        .unwrap();
    assert_eq!(req.messages.len(), 3);
    assert_eq!(req.messages[1].role, Role::User);
    match &req.messages[1].content[..] {
        [ContentBlock::Opaque { format, value, .. }] => {
            assert_eq!(format, FMT);
            assert_eq!(
                *value,
                json!({"role": "critic", "content": "meh", "weight": 2})
            );
        }
        other => panic!("unexpected content: {other:?}"),
    }
    let w = warnings
        .iter()
        .find(|w| w.code == WarningCode::MalformedField)
        .unwrap();
    assert_eq!(w.location, "/messages/1");
    // Wire-level round trip is the identity (even the string-form content
    // stays a string — verbatim means verbatim).
    let first = rebuild(&req, "m");
    assert_eq!(first, body);
    let (req2, _) = AnthropicMessages
        .parse_request(&serde_json::to_vec(&first).unwrap())
        .unwrap();
    assert_eq!(rebuild(&req2, "m"), first);
}

fn meta() -> ResponseMeta {
    ResponseMeta::new(200, Default::default())
}

#[test]
fn parse_response_text_and_usage_unification() {
    let raw = fixture("response_text.json");
    let resp = AnthropicMessages.parse_response(&raw, &meta()).unwrap();
    assert_eq!(resp.id.as_deref(), Some("msg_013Zva2CMHLNnXjNJJKqJ2EF"));
    assert_eq!(resp.model.as_deref(), Some("claude-sonnet-5"));
    assert_eq!(resp.stop_reason, Some(StopReason::EndTurn));
    assert_eq!(resp.text(), "Hi! My name is Claude.");
    let usage = resp.usage.as_ref().unwrap();
    // input = uncached 2095 + cache write 2051 + cache read 30.
    assert_eq!(usage.input_tokens, 4176);
    assert_eq!(usage.output_tokens, 503);
    assert_eq!(usage.cache_read_tokens, Some(30));
    assert_eq!(usage.cache_write_tokens, Some(2051));
    assert_eq!(usage.reasoning_tokens, Some(100));
    assert_eq!(usage.visible_output_tokens(), 403);
    assert!(usage.raw.is_some());
    // The full body is preserved.
    let original: Value = serde_json::from_slice(&raw).unwrap();
    assert_eq!(resp.raw.as_ref().unwrap(), &original);
    assert_eq!(resp.status, 200);
}

#[test]
fn usage_input_sum_saturates_instead_of_overflowing() {
    // Misbehaving provider data at u64::MAX must saturate, not panic
    // (debug) or wrap (release).
    let body = json!({
        "id": "m", "type": "message", "role": "assistant", "model": "x",
        "content": [], "stop_reason": "end_turn",
        "usage": {
            "input_tokens": u64::MAX,
            "cache_creation_input_tokens": u64::MAX,
            "cache_read_input_tokens": 7,
            "output_tokens": 1,
        },
    });
    let resp = AnthropicMessages
        .parse_response(&serde_json::to_vec(&body).unwrap(), &meta())
        .unwrap();
    let usage = resp.usage.as_ref().unwrap();
    assert_eq!(usage.input_tokens, u64::MAX);
    assert_eq!(usage.output_tokens, 1);
}

#[test]
fn parse_response_tool_use() {
    let resp = AnthropicMessages
        .parse_response(&fixture("response_tool_use.json"), &meta())
        .unwrap();
    assert_eq!(resp.stop_reason, Some(StopReason::ToolUse));
    match &resp.message.content[1] {
        ContentBlock::ToolCall {
            id,
            name,
            arguments,
            ..
        } => {
            assert_eq!(id.as_deref(), Some("toolu_01T1"));
            assert_eq!(name, "get_weather");
            assert_eq!(
                serde_json::from_str::<Value>(arguments).unwrap(),
                json!({"location": "San Francisco, CA"})
            );
        }
        other => panic!("unexpected block: {other:?}"),
    }
}

#[test]
fn tool_use_non_object_input_warns_and_keeps_value_verbatim() {
    // Request side: the wire value survives verbatim in `arguments`, but
    // the call is degraded — same-format rebuild fails the § 4.5 object
    // contract — so the parse warns MalformedToolCall (semantic).
    let body = json!({
        "model": "m", "max_tokens": 5,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_1", "name": "f", "input": 5}
            ]},
        ],
    });
    let (req, warnings) = AnthropicMessages
        .parse_request(&serde_json::to_vec(&body).unwrap())
        .unwrap();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].code, WarningCode::MalformedToolCall);
    assert_eq!(warnings[0].severity, WarningSeverity::Semantic);
    assert_eq!(warnings[0].location, "/messages/1/content/0/input");
    match &req.messages[1].content[0] {
        ContentBlock::ToolCall { id, arguments, .. } => {
            assert_eq!(id.as_deref(), Some("tu_1"));
            assert_eq!(arguments, "5");
        }
        other => panic!("unexpected block: {other:?}"),
    }
    assert!(
        AnthropicMessages
            .build_request(&req, &ctx_for("m"))
            .is_err()
    );

    // Response side reports through Response.warnings.
    let body = json!({
        "id": "m", "type": "message", "role": "assistant", "model": "x",
        "content": [{"type": "tool_use", "id": "tu_1", "name": "f", "input": [1, 2]}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    });
    let resp = AnthropicMessages
        .parse_response(&serde_json::to_vec(&body).unwrap(), &meta())
        .unwrap();
    assert_eq!(resp.warnings.len(), 1, "{:?}", resp.warnings);
    assert_eq!(resp.warnings[0].code, WarningCode::MalformedToolCall);
    assert_eq!(resp.warnings[0].location, "/content/0/input");
    assert!(matches!(
        &resp.message.content[0],
        ContentBlock::ToolCall { arguments, .. } if arguments == "[1,2]"
    ));

    // An object input stays warning-free.
    let body = json!({
        "id": "m", "type": "message", "role": "assistant", "model": "x",
        "content": [{"type": "tool_use", "id": "tu_1", "name": "f", "input": {"a": 1}}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    });
    let resp = AnthropicMessages
        .parse_response(&serde_json::to_vec(&body).unwrap(), &meta())
        .unwrap();
    assert!(resp.warnings.is_empty(), "{:?}", resp.warnings);
}

#[test]
fn parse_response_thinking_blocks_and_history_replay() {
    let resp = AnthropicMessages
        .parse_response(&fixture("response_thinking.json"), &meta())
        .unwrap();
    assert_eq!(resp.message.content.len(), 3);
    assert_eq!(resp.usage.as_ref().unwrap().reasoning_tokens, Some(150));
    // Feeding the assistant message back as history reconstructs the wire
    // blocks (thinking with signature, redacted_thinking).
    let mut req = Request::with_messages(vec![Message::user_text("q"), resp.message.clone()]);
    req.max_output_tokens = Some(10);
    let rebuilt = rebuild(&req, "claude-sonnet-5");
    assert_eq!(
        rebuilt["messages"][1]["content"],
        json!([
            {"type": "thinking", "thinking": "GCD(1071, 462) = 21.", "signature": "EqQBCgIYAhIM"},
            {"type": "redacted_thinking", "data": "EmwKAhgBEgy3"},
            {"type": "text", "text": "The answer is 21."},
        ])
    );
}

#[test]
fn parse_response_unmodeled_blocks_and_pause_turn() {
    let resp = AnthropicMessages
        .parse_response(&fixture("response_server_tool.json"), &meta())
        .unwrap();
    assert_eq!(resp.stop_reason, Some(StopReason::PauseTurn));
    assert!(matches!(
        &resp.message.content[0],
        ContentBlock::Opaque { format, value, .. } if format == FMT && value["type"] == "server_tool_use"
    ));
    assert!(matches!(
        &resp.message.content[1],
        ContentBlock::Opaque { format, value, .. } if format == FMT && value["type"] == "web_search_tool_result"
    ));
}

#[test]
fn stop_reason_mapping_and_normalization() {
    let make = |stop: &str, content: Value| {
        let body = json!({
            "id": "m", "type": "message", "role": "assistant", "model": "x",
            "content": content, "stop_reason": stop,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        });
        AnthropicMessages
            .parse_response(&serde_json::to_vec(&body).unwrap(), &meta())
            .unwrap()
    };
    assert_eq!(
        make("refusal", json!([])).stop_reason,
        Some(StopReason::Refusal)
    );
    assert_eq!(
        make("model_context_window_exceeded", json!([])).stop_reason,
        Some(StopReason::Other("model_context_window_exceeded".into()))
    );
    assert_eq!(
        make("compaction", json!([])).stop_reason,
        Some(StopReason::Other("compaction".into()))
    );
    // end_turn + tool_use content normalizes to ToolUse.
    let resp = make(
        "end_turn",
        json!([{"type": "tool_use", "id": "t", "name": "f", "input": {}}]),
    );
    assert_eq!(resp.stop_reason, Some(StopReason::ToolUse));
}

#[test]
fn parse_error_mapping() {
    let err =
        AnthropicMessages.parse_error(529, &Default::default(), &fixture("error_overloaded.json"));
    match err {
        Error::Api {
            status,
            kind,
            message,
            parsed,
            retry_after,
            ..
        } => {
            assert_eq!(status, 529);
            assert_eq!(kind, ApiErrorKind::Overloaded);
            assert_eq!(message, "Overloaded");
            assert!(parsed.is_some());
            assert_eq!(retry_after, None);
        }
        other => panic!("unexpected error: {other:?}"),
    }

    // Every documented error.type maps.
    for (t, kind) in [
        ("invalid_request_error", ApiErrorKind::InvalidRequest),
        ("authentication_error", ApiErrorKind::Auth),
        ("permission_error", ApiErrorKind::PermissionDenied),
        ("not_found_error", ApiErrorKind::NotFound),
        ("rate_limit_error", ApiErrorKind::RateLimit),
        ("api_error", ApiErrorKind::ServerError),
        ("overloaded_error", ApiErrorKind::Overloaded),
        ("billing_error", ApiErrorKind::Other("billing_error".into())),
    ] {
        let body = json!({"type": "error", "error": {"type": t, "message": "m"}});
        let err = AnthropicMessages.parse_error(
            400,
            &Default::default(),
            &serde_json::to_vec(&body).unwrap(),
        );
        match err {
            Error::Api { kind: k, .. } => assert_eq!(k, kind, "for {t}"),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    // Retry-After header is surfaced.
    let mut m = ResponseMeta::new(429, Default::default());
    m.headers.insert("retry-after", "12".parse().unwrap());
    let body = json!({"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}});
    let err = AnthropicMessages.parse_error(429, &m.headers, &serde_json::to_vec(&body).unwrap());
    match err {
        Error::Api { retry_after, .. } => {
            assert_eq!(retry_after, Some(std::time::Duration::from_secs(12)));
        }
        other => panic!("unexpected error: {other:?}"),
    }

    // Non-JSON bodies degrade to status classification.
    let err = AnthropicMessages.parse_error(429, &Default::default(), b"<html>too many</html>");
    match err {
        Error::Api { kind, .. } => assert_eq!(kind, ApiErrorKind::RateLimit),
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn models_request_and_response() {
    let built = AnthropicMessages
        .build_models_request(&ctx_for("claude-sonnet-5"), None)
        .unwrap();
    assert_eq!(built.method.as_str(), "GET");
    assert_eq!(built.url.to_string(), "https://api.anthropic.com/v1/models");
    assert_eq!(
        built.headers.get("anthropic-version").unwrap(),
        "2023-06-01"
    );
    assert_eq!(built.auth.as_ref().unwrap().header.as_str(), "x-api-key");

    let with_cursor = AnthropicMessages
        .build_models_request(&ctx_for("claude-sonnet-5"), Some("claude-opus-4-6"))
        .unwrap();
    assert_eq!(
        with_cursor.url.to_string(),
        "https://api.anthropic.com/v1/models?after_id=claude-opus-4-6"
    );

    let (models, next) = AnthropicMessages
        .parse_models_response(&fixture("models_page.json"))
        .unwrap();
    assert_eq!(models.len(), 2);
    assert_eq!(models[0].id, "claude-opus-4-6");
    assert_eq!(models[0].display_name.as_deref(), Some("Claude Opus 4.6"));
    assert!(models[0].created.is_some());
    assert_eq!(models[0].raw["max_tokens"], json!(128000));
    // A malformed timestamp degrades to None without failing the call.
    assert!(models[1].created.is_none());
    assert_eq!(next.as_deref(), Some("claude-sonnet-4-6"));

    // has_more: false ends pagination.
    let last_page = json!({"data": [], "has_more": false, "last_id": "x"});
    let (models2, next2) = AnthropicMessages
        .parse_models_response(&serde_json::to_vec(&last_page).unwrap())
        .unwrap();
    assert!(models2.is_empty());
    assert_eq!(next2, None);
}

#[test]
fn models_page_malformed_pagination_is_a_parse_error() {
    // has_more=true without a usable cursor would silently truncate the
    // listing; it must fail instead of ending pagination.
    for page in [
        json!({"data": [], "has_more": true}),
        json!({"data": [], "has_more": true, "last_id": null}),
        json!({"data": [], "has_more": true, "last_id": ""}),
    ] {
        let err = AnthropicMessages
            .parse_models_response(&serde_json::to_vec(&page).unwrap())
            .unwrap_err();
        match err {
            Error::Parse { message, .. } => {
                assert!(message.contains("pagination"), "message: {message}");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
    // Absent or false has_more keeps ending pagination, cursor ignored.
    for page in [
        json!({"data": []}),
        json!({"data": [], "has_more": false}),
        json!({"data": [], "last_id": "x"}),
    ] {
        let (_, next) = AnthropicMessages
            .parse_models_response(&serde_json::to_vec(&page).unwrap())
            .unwrap();
        assert_eq!(next, None);
    }
}

#[test]
fn count_tokens_adapter() {
    let mut req = Request::with_messages(vec![Message::user_text("Hello, world")]);
    req.max_output_tokens = Some(1024);
    req.temperature = Some(0.5);
    req.seed = Some(1); // chat-build cosmetic warning, kept in the output
    let built = AnthropicMessages
        .build_count_tokens_request(&req, &ctx_for("claude-sonnet-5"))
        .unwrap();
    assert_eq!(
        built.url.to_string(),
        "https://api.anthropic.com/v1/messages/count_tokens"
    );
    let body: Value = serde_json::from_slice(&built.body).unwrap();
    // Converter-generated keys outside the accepted set are filtered
    // silently.
    assert_eq!(
        body,
        json!({
            "model": "claude-sonnet-5",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello, world"}]}],
        })
    );
    assert!(
        built
            .warnings
            .iter()
            .any(|w| w.code == WarningCode::SamplingParameterDropped)
    );
    assert!(
        !built
            .warnings
            .iter()
            .any(|w| w.code == WarningCode::CountTokensFieldDropped)
    );

    // Injected fields the endpoint rejects make the count inexact.
    let mut req2 = req.clone();
    req2.extra.set(FMT, "service_tier", "auto");
    req2.extra
        .set(FMT, "context_management", json!({"edits": []}));
    let built2 = AnthropicMessages
        .build_count_tokens_request(&req2, &ctx_for("claude-sonnet-5"))
        .unwrap();
    let body2: Value = serde_json::from_slice(&built2.body).unwrap();
    assert!(body2.get("service_tier").is_none());
    assert_eq!(body2["context_management"], json!({"edits": []})); // accepted key survives
    let dropped = built2
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::CountTokensFieldDropped)
        .unwrap();
    assert_eq!(dropped.location, "/service_tier");

    // Under strict the inexact count is an error.
    let mut strict_ctx = ctx_for("claude-sonnet-5");
    strict_ctx.convert = ConvertOptions::new().strict(true);
    let err = AnthropicMessages
        .build_count_tokens_request(&req2, &strict_ctx)
        .unwrap_err();
    assert!(matches!(
        err,
        Error::Conversion(ConversionError::Strict { .. })
    ));
}

#[test]
fn count_tokens_response_parses() {
    let body = json!({"input_tokens": 2095, "context_management": {"original_input_tokens": 0}});
    let count = AnthropicMessages
        .parse_count_tokens_response(&serde_json::to_vec(&body).unwrap())
        .unwrap();
    assert_eq!(count.input_tokens, 2095);
    assert_eq!(
        count.raw.as_ref().unwrap()["context_management"]["original_input_tokens"],
        json!(0)
    );
}
