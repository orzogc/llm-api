//! Integration tests for the `openai_chat_completions` format: field
//! mappings, round-trips, hooks, models, error parsing and the
//! not-supported token-count surface.

use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use llm_api::formats::openai_chat_completions::{
    OpenAiChatCompletions, request_from_ir, request_to_ir, response_to_ir,
};
use llm_api::{
    ApiFormat, BuildCtx, BuiltRequest, CacheHint, CallMode, ContentBlock, ConversionDirection,
    ConversionError, ConvertOptions, Effort, EndpointUrl, Error, FunctionTool, ImageSource,
    Message, OpenAiChatCompletionsOptions, OrphanToolCalls, OutputFormat, Reasoning, Request,
    RequestHooks, ResponseMeta, Role, StopReason, Tool, ToolChoice, ToolOutputBlock, WarningCode,
    WarningSeverity,
};

const F: &str = "openai_chat_completions";

fn fixture(name: &str) -> Vec<u8> {
    let path = format!(
        "{}/tests/fixtures/openai_chat_completions/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

fn fixture_json(name: &str) -> Value {
    serde_json::from_slice(&fixture(name)).expect("fixture is JSON")
}

fn ctx(mode: CallMode) -> BuildCtx {
    BuildCtx::new(
        EndpointUrl::base("https://api.openai.com/v1").unwrap(),
        "gpt-4.1",
        mode,
    )
}

fn build(req: &Request) -> BuiltRequest {
    OpenAiChatCompletions
        .build_request(req, &ctx(CallMode::Unary))
        .expect("build succeeds")
}

fn build_err(req: &Request) -> Error {
    OpenAiChatCompletions
        .build_request(req, &ctx(CallMode::Unary))
        .unwrap_err()
}

fn body_of(built: &BuiltRequest) -> Value {
    serde_json::from_slice(&built.body).expect("body is JSON")
}

fn meta_ok() -> ResponseMeta {
    ResponseMeta::new(200, http::HeaderMap::new())
}

fn has_code(warnings: &[llm_api::ConversionWarning], code: &WarningCode) -> bool {
    warnings.iter().any(|w| w.code == *code)
}

fn from_ir_unary(req: &Request) -> (Value, Vec<llm_api::ConversionWarning>) {
    request_from_ir(
        req,
        None,
        CallMode::Unary,
        &ConvertOptions::default(),
        &OpenAiChatCompletionsOptions::default(),
    )
    .expect("conversion succeeds")
}

// ---------------------------------------------------------------- build

#[test]
fn chat_request_url_method_auth() {
    let req = Request::with_messages(vec![Message::user_text("hi")]);
    let built = build(&req);
    assert_eq!(built.method, http::Method::POST);
    assert_eq!(
        built.url.to_string(),
        "https://api.openai.com/v1/chat/completions"
    );
    assert_eq!(
        built.headers.get("content-type").unwrap(),
        "application/json"
    );
    let auth = built.auth.as_ref().expect("bearer auth");
    assert_eq!(auth.header, http::header::AUTHORIZATION);
    assert_eq!(auth.prefix.as_deref(), Some("Bearer "));
    let body = body_of(&built);
    assert_eq!(body["model"], json!("gpt-4.1"));
    assert_eq!(body["messages"], json!([{"role": "user", "content": "hi"}]));
    assert!(body.get("stream").is_none());
    assert!(body.get("stream_options").is_none());
}

#[test]
fn streaming_sets_stream_and_injects_include_usage() {
    let req = Request::with_messages(vec![Message::user_text("hi")]);
    let built = OpenAiChatCompletions
        .build_request(&req, &ctx(CallMode::Streaming))
        .unwrap();
    let body = body_of(&built);
    assert_eq!(body["stream"], json!(true));
    assert_eq!(body["stream_options"], json!({"include_usage": true}));

    // The injection knob turns the injection off; `stream` stays.
    let mut no_usage = ctx(CallMode::Streaming);
    no_usage
        .format_options
        .openai_chat_completions
        .inject_include_usage = false;
    let built = OpenAiChatCompletions
        .build_request(&req, &no_usage)
        .unwrap();
    let body = body_of(&built);
    assert_eq!(body["stream"], json!(true));
    assert!(body.get("stream_options").is_none());

    // The typed layer honors the same options.
    let (body, _) = request_from_ir(
        &req,
        Some("m"),
        CallMode::Streaming,
        &ConvertOptions::default(),
        &OpenAiChatCompletionsOptions::default(),
    )
    .unwrap();
    assert_eq!(body["stream_options"], json!({"include_usage": true}));
}

#[test]
fn sampling_parameters_map_or_warn() {
    let mut req = Request::with_messages(vec![Message::user_text("hi")]);
    req.max_output_tokens = Some(512);
    req.temperature = Some(0.5);
    req.top_p = Some(0.9);
    req.top_k = Some(40);
    req.stop_sequences = Some(vec!["END".into()]);
    req.seed = Some(7);
    req.frequency_penalty = Some(0.1);
    req.presence_penalty = Some(0.2);
    req.metadata = Some(serde_json::Map::from_iter([(
        "trace".to_owned(),
        json!("t1"),
    )]));
    req.cache_key = Some("cache-1".into());
    let built = build(&req);
    let body = body_of(&built);
    assert_eq!(body["max_completion_tokens"], json!(512));
    assert!(
        body.get("max_tokens").is_none(),
        "legacy field must not be emitted"
    );
    assert_eq!(body["temperature"], json!(0.5));
    assert_eq!(body["top_p"], json!(0.9));
    assert_eq!(body["stop"], json!(["END"]));
    assert_eq!(body["seed"], json!(7));
    assert_eq!(body["frequency_penalty"], json!(0.1));
    assert_eq!(body["presence_penalty"], json!(0.2));
    assert_eq!(body["metadata"], json!({"trace": "t1"}));
    assert_eq!(body["prompt_cache_key"], json!("cache-1"));
    assert!(body.get("top_k").is_none());
    let w = built
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::SamplingParameterDropped)
        .unwrap();
    assert_eq!(w.location, "/top_k");
    assert_eq!(w.severity, WarningSeverity::Cosmetic);
    assert_eq!(built.warnings.len(), 1, "{:?}", built.warnings);
}

#[test]
fn non_finite_sampling_values_are_conversion_errors() {
    // JSON has no NaN/±infinity (serde_json would write `null`), so the
    // build fails loudly — regardless of strict mode (default is lenient).
    type SetField = fn(&mut Request, f64);
    let fields: [(SetField, &str); 4] = [
        (|r, v| r.temperature = Some(v), "/temperature"),
        (|r, v| r.top_p = Some(v), "/top_p"),
        (|r, v| r.frequency_penalty = Some(v), "/frequency_penalty"),
        (|r, v| r.presence_penalty = Some(v), "/presence_penalty"),
    ];
    for (set, expected) in fields {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut req = Request::with_messages(vec![Message::user_text("hi")]);
            set(&mut req, bad);
            match build_err(&req) {
                Error::Conversion(ConversionError::NonFiniteNumber { location, .. }) => {
                    assert_eq!(location, expected);
                }
                other => panic!("expected NonFiniteNumber for {bad}, got {other:?}"),
            }
        }
    }

    // Finite values — zeroes and extremes included — pass through verbatim.
    let mut req = Request::with_messages(vec![Message::user_text("hi")]);
    req.temperature = Some(0.0);
    req.top_p = Some(-0.0);
    req.frequency_penalty = Some(f64::MAX);
    req.presence_penalty = Some(f64::MIN_POSITIVE);
    let body = body_of(&build(&req));
    assert_eq!(body["temperature"], json!(0.0));
    assert_eq!(body["top_p"], json!(-0.0));
    assert_eq!(body["frequency_penalty"], json!(f64::MAX));
    assert_eq!(body["presence_penalty"], json!(f64::MIN_POSITIVE));
}

#[test]
fn strict_mode_escalates_unless_overridden() {
    // A FileId image has no CC channel — a semantic loss.
    let mut req = Request::with_messages(vec![Message::user(vec![
        ContentBlock::text("look"),
        ContentBlock::image_file_id("file-1"),
    ])]);
    let built = build(&req);
    let w = built
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::ImageSourceUnsupported)
        .unwrap();
    assert_eq!(w.severity, WarningSeverity::Semantic);
    assert_eq!(w.location, "/messages/0/content/1");

    let mut strict_ctx = ctx(CallMode::Unary);
    strict_ctx.convert.strict = true;
    let err = OpenAiChatCompletions
        .build_request(&req, &strict_ctx)
        .unwrap_err();
    assert!(matches!(
        err,
        Error::Conversion(ConversionError::Strict { .. })
    ));

    // A message-level extra that replaces `content` (an array is a scalar
    // replace under RFC 7396) addresses the pointer's ancestor: the
    // warning is overridden and the strict gate passes.
    req.messages[0]
        .extra
        .set(F, "content", json!([{"type": "text", "text": "replaced"}]));
    let built = OpenAiChatCompletions
        .build_request(&req, &strict_ctx)
        .unwrap();
    let w = built
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::ImageSourceUnsupported)
        .unwrap();
    assert!(w.overridden);
    assert_eq!(
        body_of(&built)["messages"][0]["content"],
        json!([{"type": "text", "text": "replaced"}])
    );
}

#[test]
fn system_inserted_at_front_as_message() {
    let mut req = Request::with_messages(vec![Message::user_text("hi")]);
    req.system = Some(vec![ContentBlock::text("You are terse.")]);
    let built = build(&req);
    assert_eq!(
        body_of(&built)["messages"],
        json!([
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "hi"},
        ])
    );
    assert!(built.warnings.is_empty());

    // Multiple blocks / cache hints force the part-array form; breakpoints
    // land on system parts natively.
    req.system = Some(vec![
        ContentBlock::text("a"),
        ContentBlock::text("b").with_cache(CacheHint::new()),
    ]);
    let built = build(&req);
    assert_eq!(
        body_of(&built)["messages"][0],
        json!({"role": "system", "content": [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b", "prompt_cache_breakpoint": {"mode": "explicit"}},
        ]})
    );
    assert!(built.warnings.is_empty());

    // Non-text system blocks are structural errors, Opaque included.
    req.system = Some(vec![ContentBlock::opaque(
        F,
        json!({"type": "text", "text": "x"}),
    )]);
    assert!(matches!(
        build_err(&req),
        Error::Conversion(ConversionError::InvalidBlockForRole {
            role: Role::System,
            ..
        })
    ));
    req.system = Some(vec![ContentBlock::image_url("https://x/i.png")]);
    assert!(matches!(
        build_err(&req),
        Error::Conversion(ConversionError::InvalidBlockForRole {
            role: Role::System,
            ..
        })
    ));
}

#[test]
fn content_string_shorthand_rules() {
    // Single plain non-empty text → string.
    let built = build(&Request::with_messages(vec![Message::user_text("hi")]));
    assert_eq!(body_of(&built)["messages"][0]["content"], json!("hi"));

    // A cache hint forces the array form (breakpoints need a part).
    let msg = Message::user(vec![
        ContentBlock::text("hi").with_cache(CacheHint::with_ttl("5m")),
    ]);
    let built = build(&Request::with_messages(vec![msg]));
    assert_eq!(
        body_of(&built)["messages"][0]["content"],
        json!([{"type": "text", "text": "hi", "prompt_cache_breakpoint": {"mode": "explicit"}}])
    );
    let ttl = built
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::CacheTtlDropped)
        .unwrap();
    assert_eq!(
        ttl.location,
        "/messages/0/content/0/prompt_cache_breakpoint"
    );
    assert_eq!(ttl.severity, WarningSeverity::Cosmetic);

    // Block extra forces the array form so it has a landing spot.
    let msg = Message::user(vec![ContentBlock::text("hi").with_extra(F, "x", 1)]);
    let built = build(&Request::with_messages(vec![msg]));
    assert_eq!(
        body_of(&built)["messages"][0]["content"],
        json!([{"type": "text", "text": "hi", "x": 1}])
    );

    // An empty single text keeps the array form (the string form would
    // collide with the tool-message empty encoding).
    let built = build(&Request::with_messages(vec![Message::user_text("")]));
    assert_eq!(
        body_of(&built)["messages"][0]["content"],
        json!([{"type": "text", "text": ""}])
    );

    // An empty block list is an empty array.
    let built = build(&Request::with_messages(vec![Message::user(vec![])]));
    assert_eq!(body_of(&built)["messages"][0]["content"], json!([]));
}

#[test]
fn user_images_map_per_source() {
    let msg = Message::user(vec![
        ContentBlock::text("look"),
        ContentBlock::image_url("https://example.com/a.png").with_extra(
            F,
            "image_url",
            json!({"detail": "low"}),
        ),
        ContentBlock::image_base64("image/png", "QUJD").with_cache(CacheHint::new()),
        ContentBlock::opaque(
            F,
            json!({"type": "input_audio", "input_audio": {"data": "QQ==", "format": "wav"}}),
        ),
    ]);
    let built = build(&Request::with_messages(vec![msg]));
    assert_eq!(
        body_of(&built)["messages"][0]["content"],
        json!([
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png", "detail": "low"}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"},
             "prompt_cache_breakpoint": {"mode": "explicit"}},
            {"type": "input_audio", "input_audio": {"data": "QQ==", "format": "wav"}},
        ])
    );
    assert!(built.warnings.is_empty(), "{:?}", built.warnings);

    // Foreign opaque content drops with a semantic warning.
    let msg = Message::user(vec![
        ContentBlock::text("hi"),
        ContentBlock::opaque("google_generate_content", json!({"executableCode": {}})),
    ]);
    let built = build(&Request::with_messages(vec![msg]));
    let w = built
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::OpaqueDropped)
        .unwrap();
    assert_eq!(w.severity, WarningSeverity::Semantic);
    assert_eq!(
        body_of(&built)["messages"][0]["content"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn developer_role_native_and_downgraded() {
    let req = Request::with_messages(vec![Message::developer_text("prefer JSON")]);
    let built = build(&req);
    assert_eq!(body_of(&built)["messages"][0]["role"], json!("developer"));
    assert!(built.warnings.is_empty());

    let mut dctx = ctx(CallMode::Unary);
    dctx.convert.downgrade_developer = true;
    let built = OpenAiChatCompletions.build_request(&req, &dctx).unwrap();
    assert_eq!(body_of(&built)["messages"][0]["role"], json!("user"));
    assert!(has_code(&built.warnings, &WarningCode::RoleDowngraded));
}

#[test]
fn assistant_message_maps_thinking_text_and_tool_calls() {
    let msg = Message::assistant(vec![
        ContentBlock::thinking("Consider the weather."),
        ContentBlock::text("Checking now."),
        ContentBlock::tool_call_with_id("call_1", "get_weather", "{\"city\":\"Paris\"}"),
    ]);
    let tool = Message::tool(vec![ContentBlock::tool_result_text(
        Some("call_1".into()),
        "ok",
    )]);
    let built = build(&Request::with_messages(vec![msg, tool]));
    let body = body_of(&built);
    assert_eq!(
        body["messages"][0],
        json!({
            "role": "assistant",
            "content": "Checking now.",
            "reasoning_content": "Consider the weather.",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"},
            }],
        })
    );
    assert!(built.warnings.is_empty(), "{:?}", built.warnings);
}

#[test]
fn assistant_refusal_blocks_become_refusal_parts() {
    let mut refusal = ContentBlock::text("No.");
    refusal = refusal.with_extra(F, "refusal", true);
    let msg = Message::assistant(vec![ContentBlock::text("Partial answer."), refusal]);
    let built = build(&Request::with_messages(vec![msg]));
    assert_eq!(
        body_of(&built)["messages"][0]["content"],
        json!([
            {"type": "text", "text": "Partial answer."},
            {"type": "refusal", "refusal": "No."},
        ])
    );
}

#[test]
fn assistant_thinking_provenance() {
    // Multiple native plaintext blocks join with a blank line — the lost
    // block boundaries are a cosmetic ThinkingBlocksJoined (order kept,
    // so no BlockOrderLost).
    let msg = Message::assistant(vec![
        ContentBlock::thinking("First."),
        ContentBlock::thinking("Second."),
        ContentBlock::text("Answer."),
    ]);
    let built = build(&Request::with_messages(vec![msg]));
    let body = body_of(&built);
    assert_eq!(
        body["messages"][0]["reasoning_content"],
        json!("First.\n\nSecond.")
    );
    assert_eq!(built.warnings.len(), 1, "{:?}", built.warnings);
    assert_eq!(built.warnings[0].code, WarningCode::ThinkingBlocksJoined);
    assert_eq!(built.warnings[0].severity, WarningSeverity::Cosmetic);
    assert_eq!(built.warnings[0].location, "/messages/0/reasoning_content");

    // A signature has no plaintext channel: text kept, signature dropped.
    let msg = Message::assistant(vec![ContentBlock::thinking_signed("plan", "sig")]);
    let built = build(&Request::with_messages(vec![msg]));
    assert_eq!(
        body_of(&built)["messages"][0]["reasoning_content"],
        json!("plan")
    );
    let w = built
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::ThinkingSignatureDropped)
        .unwrap();
    assert_eq!(w.severity, WarningSeverity::Semantic);

    // Foreign thinking drops…
    let foreign = Message::assistant(vec![
        ContentBlock::thinking("chain").with_extra("anthropic_messages", "redacted", false),
        ContentBlock::text("answer"),
    ]);
    let built = build(&Request::with_messages(vec![foreign.clone()]));
    let body = body_of(&built);
    assert!(body["messages"][0].get("reasoning_content").is_none());
    assert!(has_code(&built.warnings, &WarningCode::ThinkingDropped));

    // …unless thinking_as_text re-emits its text.
    let mut tctx = ctx(CallMode::Unary);
    tctx.convert.thinking_as_text = true;
    let built = OpenAiChatCompletions
        .build_request(&Request::with_messages(vec![foreign]), &tctx)
        .unwrap();
    assert_eq!(
        body_of(&built)["messages"][0]["reasoning_content"],
        json!("chain")
    );
    assert!(!has_code(&built.warnings, &WarningCode::ThinkingDropped));

    // Thinking-block extras merge into the containing message.
    let msg = Message::assistant(vec![
        ContentBlock::thinking("t").with_extra(F, "reasoning_details", json!([{"type": "x"}])),
        ContentBlock::text("a"),
    ]);
    let built = build(&Request::with_messages(vec![msg]));
    let body = body_of(&built);
    assert_eq!(body["messages"][0]["reasoning_content"], json!("t"));
    assert_eq!(
        body["messages"][0]["reasoning_details"],
        json!([{"type": "x"}])
    );
}

#[test]
fn assistant_interleaved_blocks_warn_block_order_lost() {
    // Thinking after content cannot keep its position: the wire message
    // holds one field per channel.
    let msg = Message::assistant(vec![
        ContentBlock::thinking("First."),
        ContentBlock::text("One."),
        ContentBlock::thinking("Second."),
        ContentBlock::text("Two."),
    ]);
    let req = Request::with_messages(vec![msg]);
    let built = build(&req);
    let order: Vec<_> = built
        .warnings
        .iter()
        .filter(|w| w.code == WarningCode::BlockOrderLost)
        .collect();
    assert_eq!(order.len(), 1, "{:?}", built.warnings);
    assert_eq!(order[0].severity, WarningSeverity::Semantic);
    assert_eq!(order[0].location, "/messages/0");
    let joined: Vec<_> = built
        .warnings
        .iter()
        .filter(|w| w.code == WarningCode::ThinkingBlocksJoined)
        .collect();
    assert_eq!(joined.len(), 1);
    assert_eq!(joined[0].severity, WarningSeverity::Cosmetic);
    assert_eq!(built.warnings.len(), 2, "{:?}", built.warnings);
    // Every text still reaches the wire, per channel.
    let body = body_of(&built);
    assert_eq!(
        body["messages"][0]["reasoning_content"],
        json!("First.\n\nSecond.")
    );
    assert_eq!(
        body["messages"][0]["content"],
        json!([
            {"type": "text", "text": "One."},
            {"type": "text", "text": "Two."},
        ])
    );

    // Strict mode escalates the semantic order loss.
    let mut strict_ctx = ctx(CallMode::Unary);
    strict_ctx.convert.strict = true;
    assert!(matches!(
        OpenAiChatCompletions
            .build_request(&req, &strict_ctx)
            .unwrap_err(),
        Error::Conversion(ConversionError::Strict { .. })
    ));
}

#[test]
fn assistant_text_after_tool_call_warns_block_order_lost() {
    let msg = Message::assistant(vec![
        ContentBlock::text("before"),
        ContentBlock::tool_call_with_id("c1", "f", "{}"),
        ContentBlock::text("after"),
    ]);
    let tool = Message::tool(vec![ContentBlock::tool_result_text(
        Some("c1".into()),
        "ok",
    )]);
    let built = build(&Request::with_messages(vec![msg, tool]));
    let w = built
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::BlockOrderLost)
        .unwrap();
    assert_eq!(w.location, "/messages/0");
    // A single thinking text at most: no join warning here.
    assert!(!has_code(
        &built.warnings,
        &WarningCode::ThinkingBlocksJoined
    ));
}

#[test]
fn assistant_canonical_order_and_dropped_blocks_stay_quiet() {
    // Canonical channel order never warns.
    let msg = Message::assistant(vec![
        ContentBlock::thinking("plan"),
        ContentBlock::text("answer"),
        ContentBlock::tool_call_with_id("c1", "f", "{}"),
    ]);
    let tool = Message::tool(vec![ContentBlock::tool_result_text(
        Some("c1".into()),
        "ok",
    )]);
    let built = build(&Request::with_messages(vec![msg, tool]));
    assert!(built.warnings.is_empty(), "{:?}", built.warnings);

    // A dropped foreign thinking block has no wire position: only its own
    // ThinkingDropped fires.
    let msg = Message::assistant(vec![
        ContentBlock::text("a"),
        ContentBlock::thinking("chain").with_extra("anthropic_messages", "x", 1),
        ContentBlock::text("b"),
    ]);
    let built = build(&Request::with_messages(vec![msg.clone()]));
    assert!(has_code(&built.warnings, &WarningCode::ThinkingDropped));
    assert!(!has_code(&built.warnings, &WarningCode::BlockOrderLost));

    // thinking_as_text re-emits it into the thinking channel — now the
    // block genuinely moves and the order warning fires.
    let mut tctx = ctx(CallMode::Unary);
    tctx.convert.thinking_as_text = true;
    let built = OpenAiChatCompletions
        .build_request(&Request::with_messages(vec![msg]), &tctx)
        .unwrap();
    assert!(has_code(&built.warnings, &WarningCode::BlockOrderLost));
}

#[test]
fn parsed_assistant_replays_without_order_warnings() {
    // The parse side rebuilds blocks in canonical channel order, so
    // same-format round trips add no order noise.
    let wire = json!({
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "answer", "reasoning_content": "think",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ],
    });
    let (req, parse_warnings) = request_to_ir(&serde_json::to_vec(&wire).unwrap()).unwrap();
    assert!(parse_warnings.is_empty(), "{parse_warnings:?}");
    let (body, warnings) = from_ir_unary(&req);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(body["messages"], wire["messages"]);
}

#[test]
fn assistant_images_drop_and_tool_results_error() {
    let msg = Message::assistant(vec![
        ContentBlock::text("see"),
        ContentBlock::image_url("https://example.com/x.png"),
    ]);
    let built = build(&Request::with_messages(vec![msg]));
    let w = built
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::ImageSourceUnsupported)
        .unwrap();
    assert_eq!(w.severity, WarningSeverity::Semantic);
    assert_eq!(body_of(&built)["messages"][0]["content"], json!("see"));

    let msg = Message::assistant(vec![ContentBlock::tool_result_text(Some("c".into()), "x")]);
    assert!(matches!(
        build_err(&Request::with_messages(vec![msg])),
        Error::Conversion(ConversionError::InvalidBlockForRole {
            role: Role::Assistant,
            ..
        })
    ));
}

#[test]
fn tool_call_requires_id_and_custom_kind_reconstructs() {
    let msg = Message::assistant(vec![ContentBlock::tool_call("f", "{}")]);
    assert!(matches!(
        build_err(&Request::with_messages(vec![msg])),
        Error::Conversion(ConversionError::MissingRequired { .. })
    ));

    // The reserved `type` key rebuilds custom calls.
    let call = ContentBlock::tool_call_with_id("call_9", "run_sql", "SELECT 1")
        .with_extra(F, "type", "custom");
    let msg = Message::assistant(vec![call]);
    let built = build(&Request::with_messages(vec![msg]));
    assert_eq!(
        body_of(&built)["messages"][0]["tool_calls"][0],
        json!({"id": "call_9", "type": "custom", "custom": {"name": "run_sql", "input": "SELECT 1"}})
    );

    // Cache hints on tool calls drop cosmetically.
    let call = ContentBlock::tool_call_with_id("c", "f", "{}").with_cache(CacheHint::new());
    let built = build(&Request::with_messages(vec![Message::assistant(vec![
        call,
    ])]));
    let w = built
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::CacheHintDropped)
        .unwrap();
    assert_eq!(w.location, "/messages/0/tool_calls/0");
}

#[test]
fn tool_messages_one_per_result() {
    let tool = Message::tool(vec![
        ContentBlock::tool_result_text(Some("c1".into()), "sunny").with_tool_name("get_weather"),
        ContentBlock::tool_result(Some("c2".into()), vec![]),
        ContentBlock::tool_result(
            Some("c3".into()),
            vec![ToolOutputBlock::text("one"), ToolOutputBlock::text("two")],
        ),
    ]);
    let built = build(&Request::with_messages(vec![tool]));
    assert_eq!(
        body_of(&built)["messages"],
        json!([
            {"role": "tool", "tool_call_id": "c1", "content": "sunny", "name": "get_weather"},
            {"role": "tool", "tool_call_id": "c2", "content": ""},
            {"role": "tool", "tool_call_id": "c3", "content": [
                {"type": "text", "text": "one"},
                {"type": "text", "text": "two"},
            ]},
        ])
    );
    assert!(built.warnings.is_empty(), "{:?}", built.warnings);
}

#[test]
fn tool_message_images_drop_and_flags_warn() {
    let tool = Message::tool(vec![
        ContentBlock::tool_result(
            Some("c1".into()),
            vec![
                ToolOutputBlock::text("kept"),
                ToolOutputBlock::image(ImageSource::url("https://x/i.png")),
                ToolOutputBlock::image(ImageSource::base64("image/png", "QQ==")),
            ],
        )
        .with_is_error(true)
        .with_cache(CacheHint::new()),
    ]);
    let built = build(&Request::with_messages(vec![tool]));
    let body = body_of(&built);
    // Text is kept; both images drop with semantic warnings.
    assert_eq!(
        body["messages"][0]["content"],
        json!([{"type": "text", "text": "kept"}])
    );
    let images: Vec<_> = built
        .warnings
        .iter()
        .filter(|w| w.code == WarningCode::ToolResultImageDropped)
        .collect();
    assert_eq!(images.len(), 2);
    assert!(
        images
            .iter()
            .all(|w| w.severity == WarningSeverity::Semantic)
    );
    let e = built
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::IsErrorDropped)
        .unwrap();
    assert_eq!(e.severity, WarningSeverity::Semantic);
    assert!(has_code(&built.warnings, &WarningCode::CacheHintDropped));

    // Nested tool-output cache hints drop cosmetically (v1 rule); a single
    // hinted text still uses the string shorthand (no breakpoint channel).
    let hinted = serde_json::from_value::<ToolOutputBlock>(
        json!({"type": "text", "text": "x", "cache": {}}),
    )
    .unwrap();
    let tool = Message::tool(vec![ContentBlock::tool_result(
        Some("c2".into()),
        vec![hinted],
    )]);
    let built = build(&Request::with_messages(vec![tool]));
    assert_eq!(body_of(&built)["messages"][0]["content"], json!("x"));
    assert!(has_code(&built.warnings, &WarningCode::CacheHintDropped));

    // Missing tool_call_id is structural.
    let tool = Message::tool(vec![ContentBlock::tool_result_text(None, "x")]);
    assert!(matches!(
        build_err(&Request::with_messages(vec![tool])),
        Error::Conversion(ConversionError::MissingRequired { .. })
    ));
}

#[test]
fn role_block_validity_enforced() {
    let cases: Vec<(Message, Role)> = vec![
        (
            Message::user(vec![ContentBlock::tool_call_with_id("c", "f", "{}")]),
            Role::User,
        ),
        (Message::user(vec![ContentBlock::thinking("t")]), Role::User),
        (
            Message::user(vec![ContentBlock::tool_result_text(Some("c".into()), "x")]),
            Role::User,
        ),
        (
            Message::new(
                Role::System,
                vec![ContentBlock::image_url("https://x/i.png")],
            ),
            Role::System,
        ),
        (
            Message::new(
                Role::Developer,
                vec![ContentBlock::image_url("https://x/i.png")],
            ),
            Role::Developer,
        ),
        (Message::tool(vec![ContentBlock::text("plain")]), Role::Tool),
        (Message::tool(vec![ContentBlock::thinking("t")]), Role::Tool),
    ];
    for (msg, role) in cases {
        match build_err(&Request::with_messages(vec![msg])) {
            Error::Conversion(ConversionError::InvalidBlockForRole { role: r, .. }) => {
                assert_eq!(r, role);
            }
            other => panic!("expected InvalidBlockForRole, got {other:?}"),
        }
    }
}

#[test]
fn tools_map_nested_shape() {
    let mut req = Request::with_messages(vec![Message::user_text("hi")]);
    let mut with_extra = FunctionTool::new("ext");
    with_extra
        .extra
        .set(F, "function", json!({"custom_field": 1}));
    with_extra.extra.set(F, "top_level", "x");
    req.tools = Some(vec![
        Tool::function(
            FunctionTool::new("get_weather")
                .with_description("Weather.")
                .with_parameters(json!({"type": "object"}))
                .with_strict(true),
        ),
        Tool::function(FunctionTool::new("bare")),
        Tool::function(with_extra),
        Tool::opaque(F, json!({"type": "custom", "custom": {"name": "run_sql"}})),
    ]);
    let built = build(&req);
    assert_eq!(
        body_of(&built)["tools"],
        json!([
            {"type": "function", "function": {
                "name": "get_weather", "description": "Weather.",
                "parameters": {"type": "object"}, "strict": true,
            }},
            {"type": "function", "function": {"name": "bare"}},
            {"type": "function", "function": {"name": "ext", "custom_field": 1}, "top_level": "x"},
            {"type": "custom", "custom": {"name": "run_sql"}},
        ])
    );
    assert!(built.warnings.is_empty(), "{:?}", built.warnings);

    // Tool cache hints drop; foreign opaque tools drop.
    req.tools = Some(vec![
        Tool::function(FunctionTool::new("t").with_cache(CacheHint::new())),
        Tool::opaque("anthropic_messages", json!({"type": "computer_20250124"})),
    ]);
    let built = build(&req);
    assert_eq!(body_of(&built)["tools"].as_array().unwrap().len(), 1);
    assert!(has_code(&built.warnings, &WarningCode::CacheHintDropped));
    let w = built
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::OpaqueDropped)
        .unwrap();
    assert_eq!(w.severity, WarningSeverity::Semantic);
}

#[test]
fn tool_choice_forms() {
    for (choice, expected) in [
        (ToolChoice::Auto, json!("auto")),
        (ToolChoice::None, json!("none")),
        (ToolChoice::Required, json!("required")),
        (
            ToolChoice::tool("get_weather"),
            json!({"type": "function", "function": {"name": "get_weather"}}),
        ),
    ] {
        let mut req = Request::with_messages(vec![Message::user_text("hi")]);
        req.tool_choice = Some(choice.clone());
        let built = build(&req);
        assert_eq!(body_of(&built)["tool_choice"], expected);
        // And it parses back to the same IR value.
        let (parsed, _) = request_to_ir(&built.body).unwrap();
        assert_eq!(parsed.tool_choice, Some(choice));
    }
}

#[test]
fn parallel_tool_calls_requires_tools() {
    let mut req = Request::with_messages(vec![Message::user_text("hi")]);
    req.parallel_tool_calls = Some(false);
    let built = build(&req);
    assert!(body_of(&built).get("parallel_tool_calls").is_none());
    assert!(has_code(
        &built.warnings,
        &WarningCode::ParallelToolCallsIgnored
    ));

    req.tools = Some(vec![Tool::function(FunctionTool::new("t"))]);
    let built = build(&req);
    assert_eq!(body_of(&built)["parallel_tool_calls"], json!(false));
    assert!(!has_code(
        &built.warnings,
        &WarningCode::ParallelToolCallsIgnored
    ));
}

#[test]
fn reasoning_mapping() {
    // enabled: false -> "none".
    let mut req = Request::with_messages(vec![Message::user_text("hi")]);
    req.reasoning = Some(Reasoning::enabled(false));
    assert_eq!(body_of(&build(&req))["reasoning_effort"], json!("none"));

    // enabled: true alone is a no-op.
    req.reasoning = Some(Reasoning::enabled(true));
    assert!(body_of(&build(&req)).get("reasoning_effort").is_none());

    // The full tier set passes through warning-free.
    for effort in [
        Effort::None,
        Effort::Minimal,
        Effort::Low,
        Effort::Medium,
        Effort::High,
        Effort::XHigh,
        Effort::Max,
        Effort::Other("turbo".into()),
    ] {
        req.reasoning = Some(Reasoning::effort(effort.clone()));
        let built = build(&req);
        assert_eq!(body_of(&built)["reasoning_effort"], json!(effort.as_str()));
        assert!(
            built.warnings.is_empty(),
            "{effort:?}: {:?}",
            built.warnings
        );
    }

    // Conflicts: effort wins with a cosmetic warning.
    let mut r = Reasoning::enabled(true);
    r.effort = Some(Effort::None);
    req.reasoning = Some(r);
    let built = build(&req);
    assert_eq!(body_of(&built)["reasoning_effort"], json!("none"));
    let w = built
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::ReasoningConflict)
        .unwrap();
    assert_eq!(w.severity, WarningSeverity::Cosmetic);

    let mut r = Reasoning::enabled(false);
    r.effort = Some(Effort::High);
    req.reasoning = Some(r);
    let built = build(&req);
    assert_eq!(body_of(&built)["reasoning_effort"], json!("high"));
    assert!(has_code(&built.warnings, &WarningCode::ReasoningConflict));

    // enabled: false + effort None agree — no conflict.
    let mut r = Reasoning::enabled(false);
    r.effort = Some(Effort::None);
    req.reasoning = Some(r);
    let built = build(&req);
    assert_eq!(body_of(&built)["reasoning_effort"], json!("none"));
    assert!(built.warnings.is_empty());

    // include_thoughts has no channel — both values warn cosmetically.
    for value in [true, false] {
        let mut r = Reasoning::effort(Effort::Low);
        r.include_thoughts = Some(value);
        req.reasoning = Some(r);
        let built = build(&req);
        let w = built
            .warnings
            .iter()
            .find(|w| w.code == WarningCode::IncludeThoughtsUnsupported)
            .unwrap();
        assert_eq!(w.severity, WarningSeverity::Cosmetic);
    }

    // Reasoning.extra has no landing spot.
    let mut r = Reasoning::effort(Effort::Low);
    r.extra.set(F, "budget_tokens", 512);
    req.reasoning = Some(r);
    let built = build(&req);
    let w = built
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::ExtraDropped)
        .unwrap();
    assert_eq!(w.severity, WarningSeverity::Semantic);
}

#[test]
fn output_format_mapping() {
    let mut req = Request::with_messages(vec![Message::user_text("hi")]);
    req.output_format = Some(
        OutputFormat::json_schema(json!({"type": "object"}))
            .with_name("result")
            .with_description("d")
            .with_strict(true),
    );
    assert_eq!(
        body_of(&build(&req))["response_format"],
        json!({"type": "json_schema", "json_schema": {
            "name": "result", "schema": {"type": "object"}, "description": "d", "strict": true,
        }})
    );

    // The upstream-required name synthesizes as "response" when unset —
    // without a warning (§ 4.9).
    req.output_format = Some(OutputFormat::json_schema(json!({"type": "object"})));
    let built = build(&req);
    assert_eq!(
        body_of(&built)["response_format"]["json_schema"]["name"],
        json!("response")
    );
    assert!(built.warnings.is_empty());

    req.output_format = Some(OutputFormat::json_object());
    assert_eq!(
        body_of(&build(&req))["response_format"],
        json!({"type": "json_object"})
    );
}

#[test]
fn orphan_tool_call_policies() {
    // Mid-array orphans warn under every policy and are never repaired.
    let mid = Request::with_messages(vec![
        Message::assistant(vec![ContentBlock::tool_call_with_id("c1", "f", "{}")]),
        Message::user_text("continue"),
    ]);
    let built = build(&mid);
    let w = built
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::OrphanToolCalls)
        .unwrap();
    assert_eq!(w.severity, WarningSeverity::Semantic);
    assert_eq!(body_of(&built)["messages"].as_array().unwrap().len(), 2);

    // Passthrough: trailing orphans are sent as-is without warnings.
    let trailing = Request::with_messages(vec![
        Message::user_text("go"),
        Message::assistant(vec![
            ContentBlock::thinking("plan"),
            ContentBlock::tool_call_with_id("c2", "f", "{}"),
        ]),
    ]);
    let built = build(&trailing);
    assert!(built.warnings.is_empty(), "{:?}", built.warnings);
    assert_eq!(body_of(&built)["messages"].as_array().unwrap().len(), 2);

    // DropTrailing removes the calls and flags the now-orphaned thinking.
    let mut dctx = ctx(CallMode::Unary);
    dctx.convert.orphan_tool_calls = OrphanToolCalls::DropTrailing;
    let built = OpenAiChatCompletions
        .build_request(&trailing, &dctx)
        .unwrap();
    let body = body_of(&built);
    assert_eq!(body["messages"].as_array().unwrap().len(), 2);
    assert_eq!(body["messages"][1]["reasoning_content"], json!("plan"));
    assert!(body["messages"][1].get("tool_calls").is_none());
    assert!(has_code(
        &built.warnings,
        &WarningCode::OrphanToolCallsDropped
    ));
    assert!(has_code(&built.warnings, &WarningCode::ThinkingOrphaned));

    // A message left empty by the drop is removed entirely.
    let only_call = Request::with_messages(vec![
        Message::user_text("go"),
        Message::assistant(vec![ContentBlock::tool_call_with_id("c9", "f", "{}")]),
    ]);
    let built = OpenAiChatCompletions
        .build_request(&only_call, &dctx)
        .unwrap();
    assert_eq!(body_of(&built)["messages"].as_array().unwrap().len(), 1);

    // SynthesizeError appends an error tool result per orphan; the error
    // marker itself has no CC channel and warns.
    let mut sctx = ctx(CallMode::Unary);
    sctx.convert.orphan_tool_calls = OrphanToolCalls::SynthesizeError;
    let built = OpenAiChatCompletions
        .build_request(&trailing, &sctx)
        .unwrap();
    let body = body_of(&built);
    assert_eq!(
        body["messages"].as_array().unwrap().last().unwrap(),
        &json!({"role": "tool", "tool_call_id": "c2", "content": "cancelled", "name": "f"})
    );
    assert!(has_code(
        &built.warnings,
        &WarningCode::OrphanToolCallsSynthesized
    ));
    assert!(has_code(&built.warnings, &WarningCode::IsErrorDropped));
}

#[test]
fn missing_thinking_and_fill_helps_on_plaintext_channel() {
    let mut req = Request::with_messages(vec![
        Message::user_text("go"),
        Message::assistant(vec![ContentBlock::tool_call_with_id("c1", "f", "{}")]),
        Message::tool(vec![ContentBlock::tool_result_text(
            Some("c1".into()),
            "ok",
        )]),
    ]);
    req.reasoning = Some(Reasoning::effort(Effort::Medium));
    let built = build(&req);
    let w = built
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::MissingThinkingWithToolCalls)
        .unwrap();
    assert_eq!(w.severity, WarningSeverity::Semantic);

    // Effort::None does not enable thinking; no warning.
    let mut off = req.clone();
    off.reasoning = Some(Reasoning::effort(Effort::None));
    assert!(build(&off).warnings.is_empty());

    // fill_missing_thinking inserts a plaintext block, which is native to
    // this signature-less channel and reaches the wire (§ 7.3).
    let mut fctx = ctx(CallMode::Unary);
    fctx.convert.fill_missing_thinking = Some("tool call".into());
    let built = OpenAiChatCompletions.build_request(&req, &fctx).unwrap();
    assert_eq!(
        body_of(&built)["messages"][1]["reasoning_content"],
        json!("tool call")
    );
    assert!(has_code(
        &built.warnings,
        &WarningCode::MissingThinkingFilled
    ));
    assert!(!has_code(&built.warnings, &WarningCode::ThinkingDropped));
    assert!(!has_code(
        &built.warnings,
        &WarningCode::MissingThinkingWithToolCalls
    ));
}

#[test]
fn hooks_visit_serialized_messages_in_order() {
    let mut req = Request::with_messages(vec![
        Message::user_text("hi"),
        Message::assistant(vec![
            ContentBlock::thinking("plan"),
            ContentBlock::text("text"),
            ContentBlock::tool_call_with_id("c1", "f", "{}"),
        ]),
        Message::tool(vec![
            ContentBlock::tool_result_text(Some("c1".into()), "a"),
            ContentBlock::tool_result_text(Some("c1".into()), "b"),
        ]),
    ]);
    req.system = Some(vec![ContentBlock::text("sys")]);
    let seen: Arc<Mutex<Vec<(usize, Role)>>> = Arc::new(Mutex::new(Vec::new()));
    let seen2 = Arc::clone(&seen);
    let mut hctx = ctx(CallMode::Unary);
    hctx.hooks = RequestHooks::new()
        .with_on_message(move |index, role, value| {
            seen2.lock().unwrap().push((index, *role));
            value["_hooked"] = json!(index);
            Ok(())
        })
        .with_on_request(|value| {
            value["_done"] = json!(true);
            Ok(())
        });
    let built = OpenAiChatCompletions.build_request(&req, &hctx).unwrap();
    let body = body_of(&built);
    // The inserted system message is a real array member and is visited;
    // the tool message split into two wire messages yields two visits.
    assert_eq!(
        *seen.lock().unwrap(),
        vec![
            (0, Role::System),
            (1, Role::User),
            (2, Role::Assistant),
            (3, Role::Tool),
            (4, Role::Tool),
        ]
    );
    for (i, item) in body["messages"].as_array().unwrap().iter().enumerate() {
        assert_eq!(item["_hooked"], json!(i));
    }
    assert_eq!(body["_done"], json!(true));
}

#[test]
fn hook_error_aborts() {
    let mut hctx = ctx(CallMode::Unary);
    hctx.hooks = RequestHooks::new().with_on_request(|_| Err(llm_api::HookError::new("nope")));
    let err = OpenAiChatCompletions
        .build_request(
            &Request::with_messages(vec![Message::user_text("hi")]),
            &hctx,
        )
        .unwrap_err();
    assert!(matches!(err, Error::Hook(_)));
}

#[test]
fn extra_overrides_and_deletes_generated_fields() {
    let mut req = Request::with_messages(vec![Message::user_text("hi")]);
    req.temperature = Some(0.7);
    // RFC 7396: null deletes, scalars replace, new keys are added.
    req.extra.set(F, "temperature", Value::Null);
    req.extra.set(
        F,
        "web_search_options",
        json!({"search_context_size": "low"}),
    );
    req.messages[0].extra.set(F, "name", "alice");
    let built = build(&req);
    let body = body_of(&built);
    assert!(body.get("temperature").is_none(), "extra null must delete");
    assert_eq!(
        body["web_search_options"],
        json!({"search_context_size": "low"})
    );
    assert_eq!(body["messages"][0]["name"], json!("alice"));
}

// ---------------------------------------------------------------- round trips

#[test]
fn canonical_request_round_trip_is_idempotent() {
    let bytes = fixture("request_canonical.json");
    let canonical = fixture_json("request_canonical.json");
    let (req, parse_warnings) = OpenAiChatCompletions.parse_request(&bytes).unwrap();
    assert!(parse_warnings.is_empty(), "{parse_warnings:?}");

    let (body, build_warnings) = request_from_ir(
        &req,
        Some("gpt-4.1"),
        CallMode::Unary,
        &ConvertOptions::default(),
        &OpenAiChatCompletionsOptions::default(),
    )
    .unwrap();
    assert!(build_warnings.is_empty(), "{build_warnings:?}");
    assert_eq!(body, canonical);

    // Idempotence: a second parse/serialize pass is a fixed point.
    let (req2, _) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    let (body2, _) = request_from_ir(
        &req2,
        Some("gpt-4.1"),
        CallMode::Unary,
        &ConvertOptions::default(),
        &OpenAiChatCompletionsOptions::default(),
    )
    .unwrap();
    assert_eq!(body2, canonical);
}

#[test]
fn canonical_request_parses_expected_ir_shape() {
    let (req, _) = OpenAiChatCompletions
        .parse_request(&fixture("request_canonical.json"))
        .unwrap();
    // Leading system stays in-array — no hoisting (implementation contract).
    assert!(req.system.is_none());
    let roles: Vec<Role> = req.messages.iter().map(|m| m.role).collect();
    assert_eq!(
        roles,
        vec![
            Role::System,
            Role::Developer,
            Role::User,
            Role::Assistant,
            Role::Tool,
            Role::Tool,
            Role::Tool, // legacy `function` message
            Role::User,
        ]
    );
    let user = &req.messages[2];
    assert_eq!(user.content.len(), 4);
    assert!(matches!(
        &user.content[0],
        ContentBlock::Text { cache: Some(_), .. }
    ));
    assert!(
        matches!(&user.content[1], ContentBlock::Image { source: ImageSource::Url(u), .. }
        if u == "https://example.com/cat.png")
    );
    assert!(
        matches!(&user.content[2], ContentBlock::Image { source: ImageSource::Base64 { media_type, .. }, .. }
        if media_type == "image/png")
    );
    assert!(matches!(&user.content[3], ContentBlock::Opaque { format, .. } if format == F));

    let assistant = &req.messages[3];
    assert_eq!(assistant.content.len(), 4);
    assert!(
        matches!(&assistant.content[0], ContentBlock::Thinking { text: Some(t), signature: None, .. }
        if t == "The image shows a cat.")
    );
    assert!(
        matches!(&assistant.content[1], ContentBlock::Text { text, .. }
        if text == "A cat. Let me check the weather too.")
    );
    assert!(
        matches!(&assistant.content[2], ContentBlock::ToolCall { id: Some(id), name, .. }
        if id == "call_1" && name == "get_weather")
    );
    let ContentBlock::ToolCall {
        name,
        arguments,
        extra,
        ..
    } = &assistant.content[3]
    else {
        panic!("expected custom tool call");
    };
    assert_eq!(name, "run_sql");
    assert_eq!(arguments, "SELECT 1");
    assert_eq!(extra.get(F).unwrap().get("type"), Some(&json!("custom")));

    // Tool messages carry one result each; empty vs string vs parts hold.
    assert!(
        matches!(&req.messages[4].content[0], ContentBlock::ToolResult { content, name: Some(n), .. }
        if content.len() == 1 && n == "get_weather")
    );
    assert!(
        matches!(&req.messages[5].content[0], ContentBlock::ToolResult { content, name: None, .. }
        if content.len() == 2)
    );
    assert!(
        matches!(&req.messages[6].content[0], ContentBlock::Opaque { format, .. } if format == F)
    );

    // Parameters and knobs.
    assert_eq!(req.max_output_tokens, Some(512));
    assert_eq!(req.stop_sequences, Some(vec!["END".to_owned()]));
    assert_eq!(req.seed, Some(42));
    assert_eq!(req.cache_key.as_deref(), Some("cache-1"));
    assert_eq!(req.reasoning.as_ref().unwrap().effort, Some(Effort::Medium));
    assert!(matches!(
        req.output_format,
        Some(OutputFormat::JsonSchema { .. })
    ));
    assert_eq!(req.tools.as_ref().unwrap().len(), 3);
    assert!(matches!(&req.tools.as_ref().unwrap()[2], Tool::Opaque { format, .. } if format == F));
    assert_eq!(req.tool_choice, Some(ToolChoice::Auto));
    assert_eq!(req.parallel_tool_calls, Some(true));
    // Unknown top-level fields landed in the request extra namespace.
    let ns = req.extra.get(F).unwrap();
    assert_eq!(ns.get("n"), Some(&json!(1)));
    assert_eq!(ns.get("logit_bias"), Some(&json!({"50256": -100})));
    assert_eq!(ns.get("service_tier"), Some(&json!("auto")));
    assert_eq!(ns.get("safety_identifier"), Some(&json!("user-hash-1")));
}

#[test]
fn ir_round_trip_preserves_modeled_fields() {
    let mut req = Request::with_messages(vec![
        Message::user(vec![
            ContentBlock::text("look").with_cache(CacheHint::new()),
            ContentBlock::image_base64("image/png", "QUJD"),
        ]),
        Message::assistant(vec![
            ContentBlock::thinking("plan"),
            ContentBlock::text("answer"),
            ContentBlock::tool_call_with_id("c1", "f", "{\"a\":1}"),
        ]),
        Message::tool(vec![ContentBlock::tool_result_text(
            Some("c1".into()),
            "ok",
        )]),
    ]);
    req.max_output_tokens = Some(9);
    req.temperature = Some(0.1);
    req.stop_sequences = Some(vec!["X".into(), "Y".into()]);
    req.tool_choice = Some(ToolChoice::Required);
    req.tools = Some(vec![Tool::function(
        FunctionTool::new("f").with_strict(true),
    )]);
    req.parallel_tool_calls = Some(true);
    req.reasoning = Some(Reasoning::effort(Effort::Medium));
    req.output_format = Some(OutputFormat::json_object());

    let (body, _) = from_ir_unary(&req);
    let (back, warnings) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");

    assert_eq!(back.messages, req.messages);
    assert_eq!(back.max_output_tokens, req.max_output_tokens);
    assert_eq!(back.temperature, req.temperature);
    assert_eq!(back.stop_sequences, req.stop_sequences);
    assert_eq!(back.tool_choice, req.tool_choice);
    assert_eq!(back.tools, req.tools);
    assert_eq!(back.parallel_tool_calls, req.parallel_tool_calls);
    assert_eq!(back.reasoning, req.reasoning);
    assert_eq!(back.output_format, req.output_format);
}

#[test]
fn request_parse_canonicalizes_shorthands() {
    let (req, warnings) = request_to_ir(
        br#"{"model": "m", "stream": true, "stream_options": {"include_usage": true},
             "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100, "stop": "END"}"#,
    )
    .unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(req.messages, vec![Message::user_text("hi")]);
    assert_eq!(req.max_output_tokens, Some(100));
    assert_eq!(req.stop_sequences, Some(vec!["END".to_owned()]));
    // model/stream/stream_options are configuration, not IR data.
    assert!(req.extra.is_empty());

    let (body, _) = request_from_ir(
        &req,
        Some("m"),
        CallMode::Unary,
        &ConvertOptions::default(),
        &OpenAiChatCompletionsOptions::default(),
    )
    .unwrap();
    assert_eq!(
        body,
        json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 100,
            "stop": ["END"],
        })
    );

    // Both max fields set: the modern one wins with a warning.
    let (req, warnings) =
        request_to_ir(br#"{"messages": [], "max_tokens": 5, "max_completion_tokens": 9}"#).unwrap();
    assert_eq!(req.max_output_tokens, Some(9));
    assert!(has_code(&warnings, &WarningCode::MalformedField));
}

#[test]
fn stream_options_unknown_members_warn_on_parse() {
    // Members beyond `include_usage` have no configuration equivalent and
    // are consumed with a warning listing them.
    let (req, warnings) = request_to_ir(
        br#"{"stream": true,
             "stream_options": {"include_usage": true, "include_obfuscation": false,
                                "vendor_x": 1},
             "messages": [{"role": "user", "content": "hi"}]}"#,
    )
    .unwrap();
    let dropped: Vec<_> = warnings
        .iter()
        .filter(|w| w.code == WarningCode::StreamOptionsDropped)
        .collect();
    assert_eq!(dropped.len(), 1, "{warnings:?}");
    assert_eq!(dropped[0].severity, WarningSeverity::Cosmetic);
    assert_eq!(dropped[0].location, "/stream_options");
    assert_eq!(dropped[0].direction, ConversionDirection::FromFormat);
    assert!(dropped[0].message.contains("`include_obfuscation`"));
    assert!(dropped[0].message.contains("`vendor_x`"));
    // The literal-`true` `include_usage` is covered by configuration and
    // is not listed among the dropped members.
    assert!(!dropped[0].message.contains("`include_usage`"));
    // Not mirrored into extra: a unary rebuild must not carry a bare
    // `stream_options` (rejected upstream without `stream`).
    assert!(req.extra.is_empty());
    let (body, _) = from_ir_unary(&req);
    assert!(body.get("stream_options").is_none());

    // `include_usage` alone is fully covered by configuration — silent.
    let (_, warnings) = request_to_ir(
        br#"{"stream": true, "stream_options": {"include_usage": true},
             "messages": [{"role": "user", "content": "hi"}]}"#,
    )
    .unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");

    // A non-object value cannot be reconstructed either.
    let (_, warnings) =
        request_to_ir(br#"{"stream": true, "stream_options": true, "messages": []}"#).unwrap();
    assert!(has_code(&warnings, &WarningCode::StreamOptionsDropped));
}

#[test]
fn stream_options_include_usage_warns_unless_literal_true() {
    // The build side can only re-inject the literal `true`, so rebuilding
    // any other value would flip its meaning — `false`, `null` and
    // non-boolean values all count as dropped members.
    for value in [json!(false), json!(null), json!(1)] {
        let body = json!({
            "stream": true,
            "stream_options": {"include_usage": value},
            "messages": [{"role": "user", "content": "hi"}],
        });
        let (req, warnings) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
        let dropped: Vec<_> = warnings
            .iter()
            .filter(|w| w.code == WarningCode::StreamOptionsDropped)
            .collect();
        assert_eq!(dropped.len(), 1, "include_usage {value}: {warnings:?}");
        assert_eq!(dropped[0].location, "/stream_options");
        assert!(
            dropped[0].message.contains("`include_usage`"),
            "{}",
            dropped[0].message
        );
        // The remedy for a lost opt-out is the injection toggle.
        assert!(
            dropped[0].message.contains("`inject_include_usage: false`"),
            "{}",
            dropped[0].message
        );
        assert!(req.extra.is_empty());
    }

    // Combined with vendor members: one warning listing everything dropped.
    let (_, warnings) = request_to_ir(
        br#"{"stream": true,
             "stream_options": {"include_usage": false, "vendor_x": 1},
             "messages": [{"role": "user", "content": "hi"}]}"#,
    )
    .unwrap();
    let dropped: Vec<_> = warnings
        .iter()
        .filter(|w| w.code == WarningCode::StreamOptionsDropped)
        .collect();
    assert_eq!(dropped.len(), 1, "{warnings:?}");
    assert!(dropped[0].message.contains("`include_usage`"));
    assert!(dropped[0].message.contains("`vendor_x`"));
}

#[test]
fn request_parse_mirrors_unmodeled_fields_into_extra() {
    let wire = json!({
        "messages": [{"role": "user", "content": "hi", "name": "alice"}],
        "n": 2,
        "logit_bias": {"50256": -100},
        "response_format": {"type": "text"},
        "tool_choice": {"type": "allowed_tools", "allowed_tools": {"mode": "auto", "tools": []}},
        "prediction": {"type": "content", "content": "guess"},
    });
    let (req, warnings) = request_to_ir(&serde_json::to_vec(&wire).unwrap()).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(req.tool_choice.is_none());
    assert!(req.output_format.is_none());
    let ns = req.extra.get(F).unwrap();
    assert_eq!(ns.get("n"), Some(&json!(2)));
    assert_eq!(ns.get("response_format"), Some(&json!({"type": "text"})));
    assert!(ns.contains_key("tool_choice"));
    assert!(ns.contains_key("prediction"));
    // The participant name mirrors on the message.
    assert_eq!(
        req.messages[0].extra.get(F).unwrap().get("name"),
        Some(&json!("alice"))
    );

    // Everything restores verbatim.
    let (body, _) = from_ir_unary(&req);
    assert_eq!(body["n"], wire["n"]);
    assert_eq!(body["response_format"], wire["response_format"]);
    assert_eq!(body["tool_choice"], wire["tool_choice"]);
    assert_eq!(body["prediction"], wire["prediction"]);
    assert_eq!(body["messages"][0]["name"], json!("alice"));

    // Unknown keys nested in known sub-objects mirror at their path.
    let wire = json!({
        "messages": [],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "r", "schema": {"type": "object"}, "vendor_hint": true,
        }},
    });
    let (req, _) = request_to_ir(&serde_json::to_vec(&wire).unwrap()).unwrap();
    assert!(matches!(
        req.output_format,
        Some(OutputFormat::JsonSchema { .. })
    ));
    assert_eq!(
        req.extra.get(F).unwrap().get("response_format"),
        Some(&json!({"json_schema": {"vendor_hint": true}}))
    );
    let (body, _) = from_ir_unary(&req);
    assert_eq!(body["response_format"], wire["response_format"]);
}

#[test]
fn null_valued_unknown_fields_canonicalize_to_absent() {
    // The documented § 1 representational loss.
    let (req, _) = request_to_ir(
        br#"{"messages": [{"role": "user", "content": "hi", "mystery": null}], "top_mystery": null}"#,
    )
    .unwrap();
    assert!(req.extra.is_empty());
    assert!(req.messages[0].extra.is_empty());
    let (body, _) = from_ir_unary(&req);
    assert_eq!(
        body,
        json!({"messages": [{"role": "user", "content": "hi"}]})
    );
}

#[test]
fn legacy_function_role_and_unknown_roles_round_trip() {
    let wire = json!({
        "messages": [
            {"role": "user", "content": "look this up"},
            {"role": "function", "name": "lookup", "content": "result"},
            {"role": "observer", "content": "dialect data"},
        ],
    });
    let (req, warnings) = request_to_ir(&serde_json::to_vec(&wire).unwrap()).unwrap();
    // `function` is a documented legacy role and parses silently to Tool;
    // the unknown role warns.
    assert_eq!(req.messages[1].role, Role::Tool);
    assert!(
        matches!(&req.messages[1].content[0], ContentBlock::Opaque { format, .. } if format == F)
    );
    assert_eq!(req.messages[2].role, Role::User);
    assert!(matches!(
        &req.messages[2].content[0],
        ContentBlock::Opaque { .. }
    ));
    let unknown: Vec<_> = warnings
        .iter()
        .filter(|w| w.code == WarningCode::MalformedField)
        .collect();
    assert_eq!(unknown.len(), 1, "{warnings:?}");

    let (body, build_warnings) = from_ir_unary(&req);
    assert!(build_warnings.is_empty(), "{build_warnings:?}");
    assert_eq!(body["messages"], wire["messages"]);
}

#[test]
fn refusal_field_canonicalizes_to_refusal_part() {
    let wire = json!({
        "messages": [{"role": "assistant", "content": "Partial.", "refusal": "No more."}],
    });
    let (req, _) = request_to_ir(&serde_json::to_vec(&wire).unwrap()).unwrap();
    assert_eq!(req.messages[0].content.len(), 2);
    let ContentBlock::Text { extra, .. } = &req.messages[0].content[1] else {
        panic!("expected refusal-marked text");
    };
    assert_eq!(extra.get(F).unwrap().get("refusal"), Some(&json!(true)));

    let (body, _) = from_ir_unary(&req);
    assert_eq!(
        body["messages"][0]["content"],
        json!([
            {"type": "text", "text": "Partial."},
            {"type": "refusal", "refusal": "No more."},
        ])
    );
    // Second pass is stable.
    let (req2, _) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    let (body2, _) = from_ir_unary(&req2);
    assert_eq!(body2, body);
}

#[test]
fn tool_content_encodings_round_trip() {
    // § 7.2: `""` ↔ empty list, string shorthand ↔ single text, part
    // arrays keep boundaries.
    let wire = json!({
        "messages": [
            {"role": "tool", "tool_call_id": "c1", "content": ""},
            {"role": "tool", "tool_call_id": "c2", "content": "just text"},
            {"role": "tool", "tool_call_id": "c3", "content": [
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"},
            ]},
        ],
    });
    let (req, warnings) = request_to_ir(&serde_json::to_vec(&wire).unwrap()).unwrap();
    assert!(warnings.is_empty());
    let lens: Vec<usize> = req
        .messages
        .iter()
        .map(|m| match &m.content[0] {
            ContentBlock::ToolResult { content, .. } => content.len(),
            other => panic!("expected tool result, got {other:?}"),
        })
        .collect();
    assert_eq!(lens, vec![0, 1, 2]);
    let (body, build_warnings) = from_ir_unary(&req);
    assert!(build_warnings.is_empty(), "{build_warnings:?}");
    assert_eq!(body["messages"], wire["messages"]);
}

#[test]
fn tool_message_without_tool_call_id_warns_malformed_tool_result() {
    // Missing entirely: the IR keeps `None` (rebuilding on this format
    // errors later — the structural requirement) and warns.
    let (req, warnings) =
        request_to_ir(br#"{"messages": [{"role": "tool", "content": "ok"}]}"#).unwrap();
    let ContentBlock::ToolResult { tool_call_id, .. } = &req.messages[0].content[0] else {
        panic!("expected tool result");
    };
    assert!(tool_call_id.is_none());
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].code, WarningCode::MalformedToolResult);
    assert_eq!(warnings[0].severity, WarningSeverity::Semantic);
    assert_eq!(warnings[0].location, "/messages/0/tool_call_id");

    // An explicit null canonicalizes to absent — same degradation.
    let (_, warnings) = request_to_ir(
        br#"{"messages": [{"role": "tool", "tool_call_id": null, "content": "ok"}]}"#,
    )
    .unwrap();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].code, WarningCode::MalformedToolResult);

    // Non-string: kept verbatim in the block extra, the IR still `None`.
    let (req, warnings) =
        request_to_ir(br#"{"messages": [{"role": "tool", "tool_call_id": 7, "content": "ok"}]}"#)
            .unwrap();
    let ContentBlock::ToolResult {
        tool_call_id,
        extra,
        ..
    } = &req.messages[0].content[0]
    else {
        panic!("expected tool result");
    };
    assert!(tool_call_id.is_none());
    assert_eq!(extra.get(F).unwrap().get("tool_call_id"), Some(&json!(7)));
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].code, WarningCode::MalformedToolResult);
    assert_eq!(warnings[0].location, "/messages/0/tool_call_id");
}

#[test]
fn tool_call_missing_fields_warn_and_parse_lenient() {
    // Every upstream-required field that is absent gets its own
    // MalformedToolCall (a degraded unified field, semantic) at the field
    // path; the parse stays lenient (empty strings, `id: None`).
    let parse = |entry: Value| {
        let body = json!({"messages": [{"role": "assistant", "tool_calls": [entry]}]});
        request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap()
    };
    let locations = |warnings: &[llm_api::ConversionWarning]| -> Vec<String> {
        warnings
            .iter()
            .map(|w| {
                assert_eq!(w.code, WarningCode::MalformedToolCall, "{w:?}");
                assert_eq!(w.severity, WarningSeverity::Semantic, "{w:?}");
                w.location.clone()
            })
            .collect()
    };

    // Missing `arguments` only.
    let (req, warnings) = parse(json!({"id": "c", "type": "function", "function": {"name": "f"}}));
    assert_eq!(
        locations(&warnings),
        vec!["/messages/0/tool_calls/0/function/arguments"]
    );
    assert!(matches!(
        &req.messages[0].content[0],
        ContentBlock::ToolCall { id: Some(_), name, arguments, .. }
            if name == "f" && arguments.is_empty()
    ));

    // Missing `name` only (an absent `type` defaults to `function`).
    let (_, warnings) = parse(json!({"id": "c", "function": {"arguments": "{}"}}));
    assert_eq!(
        locations(&warnings),
        vec!["/messages/0/tool_calls/0/function/name"]
    );

    // A bare entry reports the missing payload and both of its members.
    let (req, warnings) = parse(json!({"id": "c"}));
    assert_eq!(
        locations(&warnings),
        vec![
            "/messages/0/tool_calls/0/function",
            "/messages/0/tool_calls/0/function/name",
            "/messages/0/tool_calls/0/function/arguments",
        ]
    );
    assert!(matches!(
        &req.messages[0].content[0],
        ContentBlock::ToolCall { id: Some(id), name, arguments, .. }
            if id == "c" && name.is_empty() && arguments.is_empty()
    ));

    // Custom calls report their own member paths.
    let (_, warnings) = parse(json!({"id": "c", "type": "custom", "custom": {"name": "x"}}));
    assert_eq!(
        locations(&warnings),
        vec!["/messages/0/tool_calls/0/custom/input"]
    );
    let (_, warnings) = parse(json!({"id": "c", "type": "custom"}));
    assert_eq!(
        locations(&warnings),
        vec![
            "/messages/0/tool_calls/0/custom",
            "/messages/0/tool_calls/0/custom/name",
            "/messages/0/tool_calls/0/custom/input",
        ]
    );

    // A missing id parses to `id: None` with a warning (rebuilding such a
    // call errors later — § 4.5).
    let (req, warnings) = parse(json!({
        "type": "function", "function": {"name": "f", "arguments": "{}"}
    }));
    assert_eq!(locations(&warnings), vec!["/messages/0/tool_calls/0/id"]);
    assert!(matches!(
        &req.messages[0].content[0],
        ContentBlock::ToolCall { id: None, .. }
    ));

    // A structurally garbage entry is skipped wholesale — the heaviest
    // loss — with one warning at the entry path.
    let (req, warnings) = parse(json!({"id": 5}));
    assert!(req.messages[0].content.is_empty());
    assert_eq!(locations(&warnings), vec!["/messages/0/tool_calls/0"]);

    // Boundary: an unknown call kind mirrors the entry verbatim — nothing
    // is degraded or lost, so it stays cosmetic MalformedField.
    let (_, warnings) = parse(json!({"id": "c", "type": "browser", "action": "open"}));
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].code, WarningCode::MalformedField);
    assert_eq!(warnings[0].severity, WarningSeverity::Cosmetic);

    // A complete entry — including an *explicit* empty arguments string —
    // stays silent.
    let (_, warnings) = parse(json!({
        "id": "c", "type": "function", "function": {"name": "f", "arguments": ""}
    }));
    assert!(warnings.is_empty(), "{warnings:?}");

    // The response side reports through `Response.warnings`.
    let body = json!({
        "id": "r", "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant",
                        "tool_calls": [{"id": "c", "type": "function",
                                        "function": {"name": "f"}}]},
            "finish_reason": "tool_calls",
        }],
    });
    let resp = response_to_ir(&serde_json::to_vec(&body).unwrap(), &meta_ok()).unwrap();
    let locs: Vec<&str> = resp.warnings.iter().map(|w| w.location.as_str()).collect();
    assert_eq!(
        locs,
        vec!["/choices/0/message/tool_calls/0/function/arguments"]
    );
}

#[test]
fn typeless_custom_entry_infers_custom_kind() {
    // A typeless entry carrying only a `custom` payload parses exactly
    // like one declaring `type: "custom"` — silent § 1 canonicalization;
    // the rebuilt entry gains the explicit `type`.
    let parse = |entry: Value| {
        let body = json!({"messages": [{"role": "assistant", "tool_calls": [entry]}]});
        request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap()
    };
    let typeless =
        json!({"id": "c", "custom": {"name": "run_sql", "input": "SELECT 1", "vendor": 1}});
    let mut explicit = typeless.clone();
    explicit["type"] = json!("custom");
    let (req_typeless, ws) = parse(typeless);
    assert!(ws.is_empty(), "{ws:?}");
    let (req_explicit, ws) = parse(explicit.clone());
    assert!(ws.is_empty(), "{ws:?}");
    assert_eq!(req_typeless.messages, req_explicit.messages);
    let ContentBlock::ToolCall {
        name,
        arguments,
        extra,
        ..
    } = &req_typeless.messages[0].content[0]
    else {
        panic!("expected tool call");
    };
    assert_eq!(name, "run_sql");
    assert_eq!(arguments, "SELECT 1");
    assert_eq!(extra.get(F).unwrap().get("type"), Some(&json!("custom")));

    // Serialization restores the explicit form.
    let (body, ws) = from_ir_unary(&req_typeless);
    assert!(ws.is_empty(), "{ws:?}");
    assert_eq!(body["messages"][0]["tool_calls"][0], explicit);

    // Inferred customs degrade like declared ones: a missing `input`
    // warns at the custom member path.
    let (_, ws) = parse(json!({"id": "c", "custom": {"name": "x"}}));
    assert_eq!(ws.len(), 1, "{ws:?}");
    assert_eq!(ws[0].code, WarningCode::MalformedToolCall);
    assert_eq!(ws[0].location, "/messages/0/tool_calls/0/custom/input");

    // Both payloads without a type keep the `function` reading (the
    // `custom` payload mirrors into the namespace, no inferred type).
    let (req, ws) = parse(json!({"id": "c",
        "function": {"name": "f", "arguments": "{}"},
        "custom": {"name": "x", "input": "y"}}));
    assert!(ws.is_empty(), "{ws:?}");
    let ContentBlock::ToolCall { name, extra, .. } = &req.messages[0].content[0] else {
        panic!("expected tool call");
    };
    assert_eq!(name, "f");
    let ns = extra.get(F).unwrap();
    assert!(!ns.contains_key("type"), "{ns:?}");
    assert_eq!(ns.get("custom"), Some(&json!({"name": "x", "input": "y"})));
}

#[test]
fn non_string_tool_call_type_mirrors_verbatim() {
    // A non-string, non-`null` `type` would fail the typed entry parse
    // wholesale; instead the entry mirrors verbatim like unknown kinds —
    // the mirror re-serializes, so the warning stays cosmetic — and the
    // rebuilt entry is exact, without a fabricated `function` payload.
    let parse = |entry: Value| {
        let body = json!({"messages": [{"role": "assistant", "tool_calls": [entry]}]});
        request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap()
    };
    let entry = json!({"id": "c", "type": 5,
                       "function": {"name": "f", "arguments": "{}"}, "vendor": 1});
    let (req, warnings) = parse(entry.clone());
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].code, WarningCode::MalformedField);
    assert_eq!(warnings[0].severity, WarningSeverity::Cosmetic);
    assert_eq!(warnings[0].location, "/messages/0/tool_calls/0");
    let ContentBlock::ToolCall {
        id,
        name,
        arguments,
        extra,
        ..
    } = &req.messages[0].content[0]
    else {
        panic!("expected tool call");
    };
    assert_eq!(id.as_deref(), Some("c"));
    assert!(name.is_empty());
    assert!(arguments.is_empty());
    let ns = extra.get(F).unwrap();
    assert_eq!(ns.get("type"), Some(&json!(5)));
    assert_eq!(
        ns.get("function"),
        Some(&json!({"name": "f", "arguments": "{}"}))
    );
    assert_eq!(ns.get("vendor"), Some(&json!(1)));

    // Round trip: verbatim.
    let (body, ws) = from_ir_unary(&req);
    assert!(ws.is_empty(), "{ws:?}");
    assert_eq!(body["messages"][0]["tool_calls"][0], entry);

    // Object / array `type` values behave the same; a missing `id` adds
    // the usual semantic id warning, and the mirror branch tolerates the
    // rebuild without one.
    let entry = json!({"type": {"a": 1}, "custom": {"name": "x", "input": "y"}});
    let (req, warnings) = parse(entry.clone());
    assert_eq!(warnings.len(), 2, "{warnings:?}");
    assert_eq!(warnings[0].code, WarningCode::MalformedToolCall);
    assert_eq!(warnings[0].location, "/messages/0/tool_calls/0/id");
    assert_eq!(warnings[1].code, WarningCode::MalformedField);
    let (body, ws) = from_ir_unary(&req);
    assert!(ws.is_empty(), "{ws:?}");
    assert_eq!(body["messages"][0]["tool_calls"][0], entry);

    // Other typed-parse failures (an object `arguments` under a string
    // type) still skip the entry wholesale with a semantic warning.
    let (req, warnings) = parse(json!({"id": "c", "type": "function",
        "function": {"name": "f", "arguments": {"x": 1}}}));
    assert!(req.messages[0].content.is_empty());
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].code, WarningCode::MalformedToolCall);
    assert_eq!(warnings[0].severity, WarningSeverity::Semantic);
}

#[test]
fn non_string_reserved_type_rebuilds_without_payload() {
    // Build side: a non-string reserved `type` in the namespace selects
    // the verbatim mirror branch — no payload object is fabricated
    // around it.
    let call = ContentBlock::tool_call_with_id("c", "", "").with_extra(F, "type", json!(5));
    let req = Request::with_messages(vec![Message::assistant(vec![call])]);
    let (body, ws) = from_ir_unary(&req);
    assert!(ws.is_empty(), "{ws:?}");
    assert_eq!(
        body["messages"][0]["tool_calls"][0],
        json!({"id": "c", "type": 5})
    );
}

// ---------------------------------------------------------------- responses

#[test]
fn response_text_parses_envelope_blocks_and_usage() {
    let resp = response_to_ir(&fixture("response_text.json"), &meta_ok()).unwrap();
    assert_eq!(resp.id.as_deref(), Some("chatcmpl-r1"));
    assert_eq!(resp.model.as_deref(), Some("gpt-4.1"));
    assert_eq!(resp.status, 200);
    assert_eq!(resp.stop_reason, Some(StopReason::EndTurn));
    assert_eq!(resp.message.role, Role::Assistant);
    assert_eq!(resp.message.content.len(), 1);
    assert_eq!(resp.text(), "Hello from Paris.");
    // Message-level unknown fields (annotations) ride the message extra.
    let ns = resp.message.extra.get(F).unwrap();
    assert_eq!(
        ns.get("annotations")
            .and_then(Value::as_array)
            .map(Vec::len),
        Some(1)
    );
    let usage = resp.usage.as_ref().unwrap();
    assert_eq!(usage.input_tokens, 36); // prompt_tokens already includes cached
    assert_eq!(usage.output_tokens, 87);
    assert_eq!(usage.total_tokens, Some(123));
    assert_eq!(usage.cache_read_tokens, Some(12));
    assert_eq!(usage.cache_write_tokens, Some(3));
    assert_eq!(usage.reasoning_tokens, Some(5));
    assert_eq!(usage.visible_output_tokens(), 82);
    assert!(usage.raw.is_some());
    assert!(resp.raw.is_some());
    assert!(resp.warnings.is_empty());
}

#[test]
fn response_tool_calls_parse_and_replay() {
    let resp = response_to_ir(&fixture("response_tool_calls.json"), &meta_ok()).unwrap();
    assert_eq!(resp.stop_reason, Some(StopReason::ToolUse));
    assert_eq!(resp.message.content.len(), 2);
    assert!(
        matches!(&resp.message.content[0], ContentBlock::ToolCall { id: Some(id), name, arguments, .. }
        if id == "call_abc123" && name == "get_current_weather"
           && arguments == "{\"location\": \"Boston, MA\"}")
    );
    let ContentBlock::ToolCall {
        name,
        arguments,
        extra,
        ..
    } = &resp.message.content[1]
    else {
        panic!("expected custom call");
    };
    assert_eq!(name, "run_sql");
    assert_eq!(arguments, "SELECT 1");
    assert_eq!(extra.get(F).unwrap().get("type"), Some(&json!("custom")));

    // § 8 flow: the parsed message re-enters a request as history and
    // reconstructs the wire entries verbatim.
    let req = Request::with_messages(vec![
        Message::user_text("weather?"),
        resp.message.clone(),
        Message::tool(vec![
            ContentBlock::tool_result_text(Some("call_abc123".into()), "21C"),
            ContentBlock::tool_result_text(Some("call_def456".into()), "1 row"),
        ]),
    ]);
    let built = build(&req);
    assert!(built.warnings.is_empty(), "{:?}", built.warnings);
    let body = body_of(&built);
    assert_eq!(
        body["messages"][1]["tool_calls"],
        json!([
            {"id": "call_abc123", "type": "function",
             "function": {"name": "get_current_weather", "arguments": "{\"location\": \"Boston, MA\"}"}},
            {"id": "call_def456", "type": "custom",
             "custom": {"name": "run_sql", "input": "SELECT 1"}},
        ])
    );
    assert!(body["messages"][1].get("content").is_none());
}

#[test]
fn response_reasoning_content_parses_to_thinking_and_replays() {
    let resp = response_to_ir(&fixture("response_reasoning.json"), &meta_ok()).unwrap();
    assert_eq!(resp.message.content.len(), 2);
    let ContentBlock::Thinking {
        text,
        signature,
        extra,
        ..
    } = &resp.message.content[0]
    else {
        panic!("expected thinking block");
    };
    assert!(text.as_deref().unwrap().starts_with("Compare 9.11"));
    assert!(signature.is_none());
    assert!(extra.is_empty(), "plaintext thinking is native — no marker");
    assert_eq!(resp.text(), "9.11 is smaller than 9.8.");
    let usage = resp.usage.as_ref().unwrap();
    assert_eq!(usage.reasoning_tokens, Some(45));
    // Dialect usage fields survive in the raw usage object.
    assert_eq!(
        usage.raw.as_ref().unwrap()["prompt_cache_miss_tokens"],
        json!(20)
    );

    // Replay: the thinking block is native and re-emits as
    // reasoning_content, warning-free.
    let req = Request::with_messages(vec![Message::user_text("compare"), resp.message.clone()]);
    let built = build(&req);
    assert!(built.warnings.is_empty(), "{:?}", built.warnings);
    let body = body_of(&built);
    assert!(
        body["messages"][1]["reasoning_content"]
            .as_str()
            .unwrap()
            .starts_with("Compare 9.11")
    );
}

#[test]
fn response_refusal_normalizes_stop_reason() {
    let resp = response_to_ir(&fixture("response_refusal.json"), &meta_ok()).unwrap();
    // finish_reason was "stop"; the refusal-marked block rewrites it (§ 8).
    assert_eq!(resp.stop_reason, Some(StopReason::Refusal));
    let ContentBlock::Text { text, extra, .. } = &resp.message.content[0] else {
        panic!("expected refusal-marked text block");
    };
    assert_eq!(text, "I cannot help with that.");
    assert_eq!(extra.get(F).unwrap().get("refusal"), Some(&json!(true)));
}

#[test]
fn response_multi_choice_reads_first_and_warns() {
    let resp = response_to_ir(&fixture("response_multi_choice.json"), &meta_ok()).unwrap();
    assert_eq!(resp.text(), "First answer.");
    let w = resp
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::MultipleCandidates)
        .unwrap();
    assert_eq!(w.severity, WarningSeverity::Semantic);
    assert_eq!(w.location, "/choices");
    // The skipped choice stays in raw.
    assert_eq!(
        resp.raw.as_ref().unwrap()["choices"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn response_finish_reasons_map() {
    let body = |reason: &str| {
        json!({
            "id": "x", "object": "chat.completion", "created": 1, "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "t"},
                         "finish_reason": reason}],
        })
    };
    for (reason, expected) in [
        ("stop", StopReason::EndTurn),
        ("length", StopReason::MaxTokens),
        ("content_filter", StopReason::ContentFilter),
        ("function_call", StopReason::Other("function_call".into())),
    ] {
        let resp = response_to_ir(&serde_json::to_vec(&body(reason)).unwrap(), &meta_ok()).unwrap();
        assert_eq!(resp.stop_reason, Some(expected), "reason {reason}");
    }
    // tool_calls → ToolUse (and EndTurn + tool calls normalizes too).
    let tool_body = json!({
        "id": "x", "choices": [{"index": 0, "message": {"role": "assistant", "content": null,
            "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
            "finish_reason": "stop"}],
    });
    let resp = response_to_ir(&serde_json::to_vec(&tool_body).unwrap(), &meta_ok()).unwrap();
    assert_eq!(resp.stop_reason, Some(StopReason::ToolUse));
}

#[test]
fn response_without_choices_is_empty_with_warning() {
    let resp = response_to_ir(br#"{"id": "x", "choices": []}"#, &meta_ok()).unwrap();
    assert!(resp.message.content.is_empty());
    assert!(resp.stop_reason.is_none());
    assert!(has_code(&resp.warnings, &WarningCode::MalformedField));
    assert!(response_to_ir(b"not json", &meta_ok()).is_err());
}

// ---------------------------------------------------------------- models

#[test]
fn models_request_and_response() {
    let built = OpenAiChatCompletions
        .build_models_request(&ctx(CallMode::Unary), None)
        .unwrap();
    assert_eq!(built.method, http::Method::GET);
    assert_eq!(built.url.to_string(), "https://api.openai.com/v1/models");
    assert!(built.auth.is_some());
    assert!(built.body.is_empty());

    let (models, cursor) = OpenAiChatCompletions
        .parse_models_response(&fixture("models_list.json"))
        .unwrap();
    assert!(cursor.is_none(), "OpenAI model listing is a single page");
    assert_eq!(models.len(), 2, "the id-less entry is skipped");
    assert_eq!(models[0].id, "gpt-4.1");
    assert!(models[0].display_name.is_none());
    let created = models[0].created.unwrap();
    assert_eq!(
        created
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        1_686_935_002
    );
    assert_eq!(models[1].id, "deepseek-reasoner");
    assert!(
        OpenAiChatCompletions
            .parse_models_response(b"garbage")
            .is_err()
    );
}

// ---------------------------------------------------------------- count tokens

#[test]
fn count_tokens_is_not_supported() {
    // Chat Completions has no counting endpoint (§ 13); the defaults hold.
    let req = Request::with_messages(vec![Message::user_text("hi")]);
    assert!(matches!(
        OpenAiChatCompletions.build_count_tokens_request(&req, &ctx(CallMode::Unary)),
        Err(Error::NotSupported(_))
    ));
    assert!(matches!(
        OpenAiChatCompletions.parse_count_tokens_response(b"{}"),
        Err(Error::NotSupported(_))
    ));
}

// ---------------------------------------------------------------- errors

#[test]
fn parse_error_maps_types_and_codes() {
    let cases = [
        // OpenAI reports auth failures as invalid_request_error + code.
        (
            401,
            r#"{"error": {"message": "Incorrect API key provided", "type": "invalid_request_error", "param": null, "code": "invalid_api_key"}}"#,
            llm_api::ApiErrorKind::Auth,
        ),
        (
            429,
            r#"{"error": {"message": "You exceeded your current quota", "type": "insufficient_quota", "param": null, "code": "insufficient_quota"}}"#,
            llm_api::ApiErrorKind::RateLimit,
        ),
        (
            404,
            r#"{"error": {"message": "The model does not exist", "type": "invalid_request_error", "code": "model_not_found"}}"#,
            llm_api::ApiErrorKind::NotFound,
        ),
        (
            400,
            r#"{"error": {"message": "bad", "type": "invalid_request_error", "param": "messages"}}"#,
            llm_api::ApiErrorKind::InvalidRequest,
        ),
        (
            401,
            r#"{"error": {"message": "bad key", "type": "authentication_error"}}"#,
            llm_api::ApiErrorKind::Auth,
        ),
        (
            500,
            r#"{"error": {"message": "boom", "type": "server_error"}}"#,
            llm_api::ApiErrorKind::ServerError,
        ),
        // Unknown shapes fall back to status classification.
        (
            429,
            r#"{"error": {"message": "slow", "type": "weird"}}"#,
            llm_api::ApiErrorKind::RateLimit,
        ),
    ];
    for (status, body, expected) in cases {
        let err =
            OpenAiChatCompletions.parse_error(status, &http::HeaderMap::new(), body.as_bytes());
        match err {
            Error::Api { kind, parsed, .. } => {
                assert_eq!(kind, expected, "body: {body}");
                assert!(parsed.is_some());
            }
            other => panic!("expected Error::Api, got {other:?}"),
        }
    }
}

#[test]
fn parse_error_keeps_raw_and_retry_after() {
    let mut headers = http::HeaderMap::new();
    headers.insert(
        http::header::RETRY_AFTER,
        http::HeaderValue::from_static("7"),
    );
    let err = OpenAiChatCompletions.parse_error(429, &headers, b"<html>overloaded</html>");
    match err {
        Error::Api {
            status,
            kind,
            message,
            raw,
            retry_after,
            parsed,
            ..
        } => {
            assert_eq!(status, 429);
            assert_eq!(kind, llm_api::ApiErrorKind::RateLimit);
            assert!(message.contains("overloaded"));
            assert_eq!(&raw[..], b"<html>overloaded</html>");
            assert_eq!(retry_after, Some(std::time::Duration::from_secs(7)));
            assert!(parsed.is_none());
        }
        other => panic!("expected Error::Api, got {other:?}"),
    }
}

#[test]
fn format_id_is_registered_constant() {
    assert_eq!(OpenAiChatCompletions.id(), "openai_chat_completions");
    assert_eq!(
        OpenAiChatCompletions.id(),
        llm_api::ids::OPENAI_CHAT_COMPLETIONS
    );
}
