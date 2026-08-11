//! Integration tests for the `openai_responses` format: field mappings,
//! round-trips, hooks, models, token counting and error parsing.

use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use llm_api::formats::openai_responses::{
    OpenAiResponses, request_from_ir, request_to_ir, response_to_ir, text_block_reserved_key,
};
use llm_api::{
    ApiFormat, BuildCtx, BuiltRequest, CacheHint, CallMode, ContentBlock, ConversionError,
    ConvertOptions, Effort, EndpointUrl, Error, FunctionTool, ImageSource, Message,
    OrphanToolCalls, OutputFormat, Reasoning, Request, RequestHooks, ResponseMeta, Role,
    StopReason, Tool, ToolChoice, ToolOutputBlock, WarningCode, WarningSeverity,
};

const F: &str = "openai_responses";

fn fixture(name: &str) -> Vec<u8> {
    let path = format!(
        "{}/tests/fixtures/openai_responses/{name}",
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
        "gpt-5.1",
        mode,
    )
}

fn build(req: &Request) -> BuiltRequest {
    OpenAiResponses
        .build_request(req, &ctx(CallMode::Unary))
        .expect("build succeeds")
}

fn body_of(built: &BuiltRequest) -> Value {
    serde_json::from_slice(&built.body).expect("body is JSON")
}

fn meta_ok() -> ResponseMeta {
    ResponseMeta::new(200, http::HeaderMap::new())
}

fn codes(warnings: &[llm_api::ConversionWarning]) -> Vec<&WarningCode> {
    warnings.iter().map(|w| &w.code).collect()
}

fn has_code(warnings: &[llm_api::ConversionWarning], code: &WarningCode) -> bool {
    warnings.iter().any(|w| w.code == *code)
}

// ---------------------------------------------------------------- build

#[test]
fn chat_request_url_method_auth() {
    let req = Request::with_messages(vec![Message::user_text("hi")]);
    let built = build(&req);
    assert_eq!(built.method, http::Method::POST);
    assert_eq!(built.url.to_string(), "https://api.openai.com/v1/responses");
    assert_eq!(
        built.headers.get("content-type").unwrap(),
        "application/json"
    );
    let auth = built.auth.as_ref().expect("bearer auth");
    assert_eq!(auth.header, http::header::AUTHORIZATION);
    assert_eq!(auth.prefix.as_deref(), Some("Bearer "));
    let body = body_of(&built);
    assert_eq!(body["model"], json!("gpt-5.1"));
    assert_eq!(
        body["input"],
        json!([{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}])
    );
    assert!(body.get("stream").is_none());
}

#[test]
fn streaming_sets_stream_flag() {
    let req = Request::with_messages(vec![Message::user_text("hi")]);
    let built = OpenAiResponses
        .build_request(&req, &ctx(CallMode::Streaming))
        .unwrap();
    assert_eq!(body_of(&built)["stream"], json!(true));
}

#[test]
fn sampling_parameters_map_or_warn() {
    let mut req = Request::with_messages(vec![Message::user_text("hi")]);
    req.max_output_tokens = Some(512);
    req.temperature = Some(0.5);
    req.top_p = Some(0.9);
    req.top_k = Some(40);
    req.seed = Some(7);
    req.frequency_penalty = Some(0.1);
    req.presence_penalty = Some(0.2);
    req.stop_sequences = Some(vec!["END".into()]);
    req.metadata = Some(serde_json::Map::from_iter([(
        "trace".to_owned(),
        json!("t1"),
    )]));
    req.cache_key = Some("cache-1".into());
    let built = build(&req);
    let body = body_of(&built);
    assert_eq!(body["max_output_tokens"], json!(512));
    assert_eq!(body["temperature"], json!(0.5));
    assert_eq!(body["top_p"], json!(0.9));
    assert_eq!(body["metadata"], json!({"trace": "t1"}));
    assert_eq!(body["prompt_cache_key"], json!("cache-1"));
    for absent in [
        "top_k",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "stop_sequences",
    ] {
        assert!(body.get(absent).is_none(), "`{absent}` must not be emitted");
    }
    // stop_sequences is a semantic loss; the sampling knobs are cosmetic.
    let stop = built
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::StopSequencesDropped)
        .unwrap();
    assert_eq!(stop.severity, WarningSeverity::Semantic);
    assert_eq!(stop.location, "/stop_sequences");
    let sampling: Vec<_> = built
        .warnings
        .iter()
        .filter(|w| w.code == WarningCode::SamplingParameterDropped)
        .collect();
    assert_eq!(sampling.len(), 4);
    assert!(
        sampling
            .iter()
            .all(|w| w.severity == WarningSeverity::Cosmetic)
    );
}

#[test]
fn non_finite_sampling_values_are_conversion_errors() {
    // Checked for all four f64 fields — including the ones Responses drops
    // with a warning: a non-finite value is a caller bug, not a droppable
    // knob.
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
            let err = OpenAiResponses
                .build_request(&req, &ctx(CallMode::Unary))
                .unwrap_err();
            match err {
                Error::Conversion(ConversionError::NonFiniteNumber { location, .. }) => {
                    assert_eq!(location, expected);
                }
                other => panic!("expected NonFiniteNumber for {bad}, got {other:?}"),
            }
        }
    }

    // Finite values keep the existing behavior (mapped or dropped+warned).
    let mut req = Request::with_messages(vec![Message::user_text("hi")]);
    req.temperature = Some(0.0);
    req.top_p = Some(f64::MAX);
    req.frequency_penalty = Some(-0.0);
    let built = build(&req);
    let body = body_of(&built);
    assert_eq!(body["temperature"], json!(0.0));
    assert_eq!(body["top_p"], json!(f64::MAX));
    assert!(body.get("frequency_penalty").is_none());
}

#[test]
fn strict_mode_escalates_unless_overridden() {
    let mut req = Request::with_messages(vec![Message::user_text("hi")]);
    req.stop_sequences = Some(vec!["END".into()]);
    let mut strict_ctx = ctx(CallMode::Unary);
    strict_ctx.convert.strict = true;
    let err = OpenAiResponses
        .build_request(&req, &strict_ctx)
        .unwrap_err();
    assert!(matches!(
        err,
        Error::Conversion(ConversionError::Strict { .. })
    ));

    // An extra that addresses the warning's pointer marks it overridden and
    // the strict gate passes.
    req.extra.set(F, "stop_sequences", json!(["END"]));
    let built = OpenAiResponses.build_request(&req, &strict_ctx).unwrap();
    let w = built
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::StopSequencesDropped)
        .unwrap();
    assert!(w.overridden);
    assert_eq!(body_of(&built)["stop_sequences"], json!(["END"]));
}

#[test]
fn system_joins_into_instructions() {
    let mut req = Request::with_messages(vec![Message::user_text("hi")]);
    req.system = Some(vec![
        ContentBlock::text("You are terse."),
        ContentBlock::text("Answer in French.").with_cache(CacheHint::new()),
    ]);
    let built = build(&req);
    assert_eq!(
        body_of(&built)["instructions"],
        json!("You are terse.\n\nAnswer in French.")
    );
    let w = built
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::CacheHintDropped)
        .unwrap();
    assert_eq!(w.location, "/instructions");

    // Non-text system blocks are structural errors, Opaque included.
    req.system = Some(vec![ContentBlock::opaque(
        F,
        json!({"type": "input_text", "text": "x"}),
    )]);
    let err = build_err(&req);
    assert!(matches!(
        err,
        Error::Conversion(ConversionError::InvalidBlockForRole {
            role: Role::System,
            ..
        })
    ));
}

fn build_err(req: &Request) -> Error {
    OpenAiResponses
        .build_request(req, &ctx(CallMode::Unary))
        .unwrap_err()
}

#[test]
fn user_message_parts_map_images_and_breakpoints() {
    let msg = Message::user(vec![
        ContentBlock::text("look").with_cache(CacheHint::with_ttl("5m")),
        ContentBlock::image_url("https://example.com/a.png").with_extra(F, "detail", "low"),
        ContentBlock::image_base64("image/png", "QUJD"),
        ContentBlock::image_file_id("file-1"),
        ContentBlock::opaque(F, json!({"type": "input_file", "file_id": "file-doc"})),
    ]);
    let built = build(&Request::with_messages(vec![msg]));
    let body = body_of(&built);
    assert_eq!(
        body["input"][0]["content"],
        json!([
            {"type": "input_text", "text": "look", "prompt_cache_breakpoint": {"mode": "explicit"}},
            {"type": "input_image", "image_url": "https://example.com/a.png", "detail": "low"},
            {"type": "input_image", "image_url": "data:image/png;base64,QUJD"},
            {"type": "input_image", "file_id": "file-1"},
            {"type": "input_file", "file_id": "file-doc"},
        ])
    );
    // The breakpoint mapped but its TTL did not.
    let ttl = built
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::CacheTtlDropped)
        .unwrap();
    assert_eq!(ttl.location, "/input/0/content/0/prompt_cache_breakpoint");
}

#[test]
fn developer_role_native_and_downgraded() {
    let req = Request::with_messages(vec![Message::developer_text("prefer JSON")]);
    let built = build(&req);
    assert_eq!(body_of(&built)["input"][0]["role"], json!("developer"));
    assert!(built.warnings.is_empty());

    let mut dctx = ctx(CallMode::Unary);
    dctx.convert.downgrade_developer = true;
    let built = OpenAiResponses.build_request(&req, &dctx).unwrap();
    assert_eq!(body_of(&built)["input"][0]["role"], json!("user"));
    assert!(has_code(&built.warnings, &WarningCode::RoleDowngraded));
}

#[test]
fn assistant_message_explodes_into_items() {
    let msg = Message::assistant(vec![
        ContentBlock::thinking_signed("plan", "enc-1"), // optimistic native
        ContentBlock::text("Working on it."),
        ContentBlock::tool_call_with_id("call_1", "get_weather", "{}"),
        ContentBlock::opaque(F, json!({"type": "web_search_call", "id": "ws_1"})),
    ]);
    let tool = Message::tool(vec![ContentBlock::tool_result_text(
        Some("call_1".into()),
        "ok",
    )]);
    let built = build(&Request::with_messages(vec![msg, tool]));
    let body = body_of(&built);
    assert_eq!(
        body["input"],
        json!([
            {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "plan"}], "encrypted_content": "enc-1"},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Working on it.", "annotations": []}]},
            {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{}"},
            {"type": "web_search_call", "id": "ws_1"},
            {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
        ])
    );
    assert!(built.warnings.is_empty(), "{:?}", built.warnings);
}

#[test]
fn assistant_texts_group_by_item_id_and_refusal_parts_rebuild() {
    let msg = Message::assistant(vec![
        ContentBlock::text("Answer.")
            .with_extra(F, "id", "msg_1")
            .with_extra(F, "status", "completed"),
        ContentBlock::text("No more.")
            .with_extra(F, "id", "msg_1")
            .with_extra(F, text_block_reserved_key::REFUSAL, true),
        ContentBlock::text("Different item.").with_extra(F, "id", "msg_2"),
    ]);
    let built = build(&Request::with_messages(vec![msg]));
    let body = body_of(&built);
    assert_eq!(
        body["input"],
        json!([
            {
                "type": "message", "role": "assistant", "id": "msg_1", "status": "completed",
                "content": [
                    {"type": "output_text", "text": "Answer.", "annotations": []},
                    {"type": "refusal", "refusal": "No more."},
                ]
            },
            {
                "type": "message", "role": "assistant", "id": "msg_2",
                "content": [{"type": "output_text", "text": "Different item.", "annotations": []}]
            },
        ])
    );
}

#[test]
fn no_id_assistant_items_with_distinct_phase_keep_boundaries() {
    // Two id-less assistant message items whose `phase` differs must not
    // collapse into one item — the second phase would be lost. They still
    // merge into one IR assistant message (turn grouping); the block-level
    // item metadata splits them again on serialization.
    let wire = json!({
        "input": [
            {"type": "message", "role": "assistant", "phase": "commentary",
             "content": [{"type": "input_text", "text": "thinking aloud"}]},
            {"type": "message", "role": "assistant", "phase": "final_answer",
             "content": [{"type": "input_text", "text": "the answer"}]},
        ]
    });
    let (req, parse_warnings) = OpenAiResponses
        .parse_request(&serde_json::to_vec(&wire).unwrap())
        .unwrap();
    assert!(parse_warnings.is_empty(), "{parse_warnings:?}");
    assert_eq!(req.messages.len(), 1);
    let (body, build_warnings) = request_from_ir(
        &req,
        Some("gpt-5.1"),
        CallMode::Unary,
        &ConvertOptions::default(),
    )
    .unwrap();
    assert!(build_warnings.is_empty(), "{build_warnings:?}");
    // `input_text` canonicalizes to `output_text` (documented).
    assert_eq!(
        body["input"],
        json!([
            {"type": "message", "role": "assistant", "phase": "commentary",
             "content": [{"type": "output_text", "text": "thinking aloud", "annotations": []}]},
            {"type": "message", "role": "assistant", "phase": "final_answer",
             "content": [{"type": "output_text", "text": "the answer", "annotations": []}]},
        ])
    );
}

#[test]
fn no_id_assistant_items_with_distinct_status_keep_boundaries() {
    let wire = json!({
        "input": [
            {"type": "message", "role": "assistant", "status": "completed", "content": "a"},
            {"type": "message", "role": "assistant", "status": "in_progress", "content": "b"},
        ]
    });
    let (req, parse_warnings) = OpenAiResponses
        .parse_request(&serde_json::to_vec(&wire).unwrap())
        .unwrap();
    assert!(parse_warnings.is_empty(), "{parse_warnings:?}");
    let (body, build_warnings) = request_from_ir(
        &req,
        Some("gpt-5.1"),
        CallMode::Unary,
        &ConvertOptions::default(),
    )
    .unwrap();
    assert!(build_warnings.is_empty(), "{build_warnings:?}");
    let input = body["input"].as_array().unwrap();
    assert_eq!(input.len(), 2, "{input:?}");
    assert_eq!(input[0]["status"], json!("completed"));
    assert_eq!(input[1]["status"], json!("in_progress"));
}

#[test]
fn no_id_metadata_free_assistant_items_still_merge() {
    // Without any item-level metadata the id-less boundary is not
    // representable in the IR: the two items merge into one (documented
    // turn-grouping normalization), warning-free.
    let wire = json!({
        "input": [
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "a", "annotations": []}]},
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "b", "annotations": []}]},
        ]
    });
    let (req, parse_warnings) = OpenAiResponses
        .parse_request(&serde_json::to_vec(&wire).unwrap())
        .unwrap();
    assert!(parse_warnings.is_empty(), "{parse_warnings:?}");
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
        json!([
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "a", "annotations": []},
                {"type": "output_text", "text": "b", "annotations": []},
            ]},
        ])
    );
}

#[test]
fn same_id_blocks_with_conflicting_item_fields_warn_first_wins() {
    // Hand-built blocks sharing an item id but disagreeing on an
    // item-level field: the id keeps them in one item (two wire items
    // with the same id would be invalid), the first block's value wins
    // and the dropped value warns.
    let msg = Message::assistant(vec![
        ContentBlock::text("a")
            .with_extra(F, "id", "msg_1")
            .with_extra(F, "phase", "commentary"),
        ContentBlock::text("b")
            .with_extra(F, "id", "msg_1")
            .with_extra(F, "phase", "final_answer"),
    ]);
    let built = build(&Request::with_messages(vec![msg]));
    let body = body_of(&built);
    let input = body["input"].as_array().unwrap();
    assert_eq!(input.len(), 1, "{input:?}");
    assert_eq!(input[0]["phase"], json!("commentary"));
    let w = built
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::ExtraDropped)
        .expect("conflicting phase warns");
    assert_eq!(w.severity, WarningSeverity::Semantic);
    // The location is the item-level field that kept the leading value,
    // so an `extra` override addressing it marks the warning overridden.
    assert_eq!(w.location, "/input/0/phase");
}

#[test]
fn conflicting_item_field_strict_override_via_item_pointer() {
    let make = || {
        Message::assistant(vec![
            ContentBlock::text("a")
                .with_extra(F, "id", "msg_1")
                .with_extra(F, "phase", "commentary"),
            ContentBlock::text("b")
                .with_extra(F, "id", "msg_1")
                .with_extra(F, "phase", "final_answer"),
        ])
    };
    let mut strict_ctx = ctx(CallMode::Unary);
    strict_ctx.convert.strict = true;

    // Without an override, strict rejects the semantic conflict warning
    // (the item-field restoration itself must not self-mark it).
    let err = OpenAiResponses
        .build_request(&Request::with_messages(vec![make()]), &strict_ctx)
        .unwrap_err();
    assert!(matches!(
        err,
        Error::Conversion(ConversionError::Strict { .. })
    ));

    // A message `extra` that takes over the item-level field lands at
    // /input/0/phase and marks the warning overridden: strict passes.
    let mut msg = make();
    msg.extra.set(F, "phase", json!("final_answer"));
    let built = OpenAiResponses
        .build_request(&Request::with_messages(vec![msg]), &strict_ctx)
        .unwrap();
    let w = built
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::ExtraDropped)
        .unwrap();
    assert!(w.overridden);
    assert_eq!(body_of(&built)["input"][0]["phase"], json!("final_answer"));
}

#[test]
fn conflicting_unknown_item_fields_warn_per_field() {
    // `item` (nested unknown item-level fields) conflicts warn one per
    // field, located at that field on the item.
    let msg = Message::assistant(vec![
        ContentBlock::text("a")
            .with_extra(F, "id", "msg_1")
            .with_extra(
                F,
                "item",
                json!({"caller": {"type": "direct"}, "namespace": "ns_1"}),
            ),
        ContentBlock::text("b")
            .with_extra(F, "id", "msg_1")
            .with_extra(
                F,
                "item",
                json!({"caller": {"type": "program"}, "namespace": "ns_1"}),
            ),
    ]);
    let built = build(&Request::with_messages(vec![msg]));
    let conflicts: Vec<_> = built
        .warnings
        .iter()
        .filter(|w| w.code == WarningCode::ExtraDropped)
        .collect();
    assert_eq!(conflicts.len(), 1, "{conflicts:?}");
    assert_eq!(conflicts[0].location, "/input/0/caller");
    let body = body_of(&built);
    assert_eq!(body["input"][0]["caller"], json!({"type": "direct"}));
    assert_eq!(body["input"][0]["namespace"], json!("ns_1"));
}

#[test]
fn mismatched_shell_splits_with_item_boundary_lost() {
    // An unknown-part shell whose recorded identity matches neither
    // neighbour serializes as its own item (marker stripped, identity
    // restored) and warns.
    let mut shell = json!({"type": "message", "role": "assistant",
        "content": [{"type": "output_audio", "audio": "QUJD"}]});
    shell[llm_api::formats::openai_responses::OPAQUE_SHELL_MARKER] = json!({"id": "msg_2"});
    let msg = Message::assistant(vec![
        ContentBlock::text("a").with_extra(F, "id", "msg_1"),
        ContentBlock::opaque(F, shell),
        ContentBlock::text("b").with_extra(F, "id", "msg_1"),
    ]);
    let built = build(&Request::with_messages(vec![msg]));
    let body = body_of(&built);
    let input = body["input"].as_array().unwrap();
    assert_eq!(input.len(), 3, "{input:?}");
    assert_eq!(input[0]["id"], json!("msg_1"));
    assert_eq!(
        input[1],
        json!({"type": "message", "role": "assistant", "id": "msg_2",
               "content": [{"type": "output_audio", "audio": "QUJD"}]})
    );
    assert_eq!(input[2]["id"], json!("msg_1"));
    let w = built
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::ItemBoundaryLost)
        .expect("mismatched shell warns");
    assert_eq!(w.severity, WarningSeverity::Semantic);
    assert_eq!(w.location, "/input/1");
}

#[test]
fn marker_less_user_opaque_replays_verbatim_as_top_level_item() {
    // A hand-built own-format Opaque without the internal marker keeps
    // the verbatim contract: it flushes the group and serializes as a
    // top-level item, never inlined.
    let user_item = json!({"type": "message", "role": "assistant", "id": "msg_1",
        "content": [{"type": "output_audio", "audio": "QUJD"}]});
    let msg = Message::assistant(vec![
        ContentBlock::text("a").with_extra(F, "id", "msg_1"),
        ContentBlock::opaque(F, user_item.clone()),
        ContentBlock::text("b").with_extra(F, "id", "msg_1"),
    ]);
    let built = build(&Request::with_messages(vec![msg]));
    let body = body_of(&built);
    let input = body["input"].as_array().unwrap();
    assert_eq!(input.len(), 3, "{input:?}");
    assert_eq!(input[1], user_item);
    assert!(
        !built
            .warnings
            .iter()
            .any(|w| w.code == WarningCode::ItemBoundaryLost)
    );
}

#[test]
fn assistant_images_drop_with_semantic_warning() {
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
    let body = body_of(&built);
    assert_eq!(body["input"].as_array().unwrap().len(), 1);
}

#[test]
fn tool_results_encode_output_variants() {
    let tool = Message::tool(vec![
        // Single plain text -> string shorthand.
        ContentBlock::tool_result_text(Some("c1".into()), "sunny"),
        // Empty content -> "".
        ContentBlock::tool_result(Some("c2".into()), vec![]),
        // Mixed content -> part array; images per the § 4.5 table.
        ContentBlock::tool_result(
            Some("c3".into()),
            vec![
                ToolOutputBlock::text("a"),
                ToolOutputBlock::text("b"),
                ToolOutputBlock::image(ImageSource::url("https://example.com/i.png")),
                ToolOutputBlock::image(ImageSource::base64("image/png", "QUJD")),
                ToolOutputBlock::image(ImageSource::file_id("file-9")),
            ],
        )
        .with_tool_name("render"),
    ]);
    let built = build(&Request::with_messages(vec![tool]));
    let body = body_of(&built);
    assert_eq!(
        body["input"],
        json!([
            {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
            {"type": "function_call_output", "call_id": "c2", "output": ""},
            {"type": "function_call_output", "call_id": "c3", "output": [
                {"type": "input_text", "text": "a"},
                {"type": "input_text", "text": "b"},
                {"type": "input_image", "image_url": "https://example.com/i.png"},
                {"type": "input_image", "image_url": "data:image/png;base64,QUJD"},
                {"type": "input_image", "file_id": "file-9"},
            ], "name": "render"},
        ])
    );
}

#[test]
fn tool_result_is_error_and_nested_hints_warn() {
    let tool = Message::tool(vec![
        ContentBlock::tool_result_text(Some("c1".into()), "failed").with_is_error(true),
        ContentBlock::tool_result(
            Some("c2".into()),
            // The IR type has no nested-hint builder (nested hints are
            // dropped on every v1 target); deserialize the persisted IR
            // JSON shape instead.
            vec![
                serde_json::from_value::<ToolOutputBlock>(
                    json!({"type": "text", "text": "x", "cache": {}}),
                )
                .unwrap(),
            ],
        ),
    ]);
    let built = build(&Request::with_messages(vec![tool]));
    let w = built
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::IsErrorDropped)
        .unwrap();
    assert_eq!(w.severity, WarningSeverity::Semantic);
    // The nested tool-output hint drops cosmetically.
    let hints: Vec<_> = built
        .warnings
        .iter()
        .filter(|w| w.code == WarningCode::CacheHintDropped)
        .collect();
    assert_eq!(hints.len(), 1, "{:?}", built.warnings);
    assert_eq!(hints[0].location, "/input/1/output");
    // is_error: Some(false) is equivalent to absent and drops silently.
    let ok = Message::tool(vec![
        ContentBlock::tool_result_text(Some("c1".into()), "fine").with_is_error(false),
    ]);
    let built = build(&Request::with_messages(vec![ok]));
    assert!(built.warnings.is_empty());
}

#[test]
fn tool_result_hint_maps_to_breakpoint_on_last_output_part() {
    // A block-level hint on a `ToolResult` becomes a
    // `prompt_cache_breakpoint` on the last output part (output parts are
    // input parts, which all accept breakpoints upstream). The
    // single-text string shorthand yields to the array form to carry it.
    let tool = Message::tool(vec![
        ContentBlock::tool_result_text(Some("c1".into()), "sunny").with_cache(CacheHint::new()),
        ContentBlock::tool_result(
            Some("c2".into()),
            vec![ToolOutputBlock::text("a"), ToolOutputBlock::text("b")],
        )
        .with_cache(CacheHint::new()),
    ]);
    let built = build(&Request::with_messages(vec![tool]));
    assert!(built.warnings.is_empty(), "{:?}", built.warnings);
    let body = body_of(&built);
    assert_eq!(
        body["input"],
        json!([
            {"type": "function_call_output", "call_id": "c1", "output": [
                {"type": "input_text", "text": "sunny",
                 "prompt_cache_breakpoint": {"mode": "explicit"}},
            ]},
            {"type": "function_call_output", "call_id": "c2", "output": [
                {"type": "input_text", "text": "a"},
                {"type": "input_text", "text": "b",
                 "prompt_cache_breakpoint": {"mode": "explicit"}},
            ]},
        ])
    );

    // Round trip: the breakpoint survives verbatim (as part extra).
    let (req, parse_warnings) = request_to_ir(&built.body).unwrap();
    assert!(parse_warnings.is_empty(), "{parse_warnings:?}");
    let (body2, ws) =
        request_from_ir(&req, None, CallMode::Unary, &ConvertOptions::default()).unwrap();
    assert!(ws.is_empty(), "{ws:?}");
    assert_eq!(body2["input"], body["input"]);
}

#[test]
fn tool_result_hint_ttl_warns_and_empty_output_drops_hint() {
    // The breakpoint maps but a hint TTL has no OpenAI equivalent.
    let ttl = Message::tool(vec![
        ContentBlock::tool_result_text(Some("c1".into()), "r")
            .with_cache(CacheHint::with_ttl("5m")),
    ]);
    let built = build(&Request::with_messages(vec![ttl]));
    let w = built
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::CacheTtlDropped)
        .expect("hint TTL warns");
    assert_eq!(w.location, "/input/0/output/0/prompt_cache_breakpoint");
    assert_eq!(
        body_of(&built)["input"][0]["output"][0]["prompt_cache_breakpoint"],
        json!({"mode": "explicit"})
    );

    // With no output part to carry the breakpoint the hint drops.
    let empty = Message::tool(vec![
        ContentBlock::tool_result(Some("c2".into()), vec![]).with_cache(CacheHint::new()),
    ]);
    let built = build(&Request::with_messages(vec![empty]));
    assert!(has_code(&built.warnings, &WarningCode::CacheHintDropped));
    assert_eq!(body_of(&built)["input"][0]["output"], json!(""));
}

#[test]
fn missing_tool_ids_are_structural_errors() {
    let call = Message::assistant(vec![ContentBlock::tool_call("f", "{}")]);
    assert!(matches!(
        build_err(&Request::with_messages(vec![call])),
        Error::Conversion(ConversionError::MissingRequired { .. })
    ));
    let result = Message::tool(vec![ContentBlock::tool_result_text(None, "x")]);
    assert!(matches!(
        build_err(&Request::with_messages(vec![result])),
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
            Message::new(
                Role::System,
                vec![ContentBlock::image_url("https://x/i.png")],
            ),
            Role::System,
        ),
        (
            Message::assistant(vec![ContentBlock::tool_result_text(Some("c".into()), "x")]),
            Role::Assistant,
        ),
        (Message::tool(vec![ContentBlock::text("plain")]), Role::Tool),
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
fn thinking_provenance_rules() {
    // Foreign namespace -> dropped with a semantic warning.
    let foreign = Message::assistant(vec![
        ContentBlock::thinking_signed("chain", "sig").with_extra(
            "anthropic_messages",
            "redacted",
            false,
        ),
        ContentBlock::text("answer"),
    ]);
    let built = build(&Request::with_messages(vec![foreign.clone()]));
    let body = body_of(&built);
    assert_eq!(body["input"].as_array().unwrap().len(), 1);
    assert!(has_code(&built.warnings, &WarningCode::ThinkingDropped));

    // thinking_as_text re-emits the text as raw reasoning text and warns about
    // the lost signature.
    let mut tctx = ctx(CallMode::Unary);
    tctx.convert.thinking_as_text = true;
    let built = OpenAiResponses
        .build_request(&Request::with_messages(vec![foreign]), &tctx)
        .unwrap();
    let body = body_of(&built);
    assert_eq!(
        body["input"][0],
        json!({"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "chain"}]})
    );
    assert!(has_code(
        &built.warnings,
        &WarningCode::ThinkingSignatureDropped
    ));

    // Optimistic replay: a signature with no namespace is native; no item
    // id is invented.
    let optimistic = Message::assistant(vec![ContentBlock::thinking_signed("plan", "enc")]);
    let built = build(&Request::with_messages(vec![optimistic]));
    let body = body_of(&built);
    assert_eq!(
        body["input"][0],
        json!({"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "plan"}], "encrypted_content": "enc"})
    );

    // Native via the format namespace reconstructs id/summary verbatim.
    let native = Message::assistant(vec![
        ContentBlock::thinking_signed("ignored by summary override", "enc")
            .with_extra(F, "id", "rs_1")
            .with_extra(
                F,
                "summary",
                json!([{"type": "summary_text", "text": "original"}]),
            ),
    ]);
    let built = build(&Request::with_messages(vec![native]));
    let body = body_of(&built);
    assert_eq!(
        body["input"][0],
        json!({
            "type": "reasoning", "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "original"}],
            "encrypted_content": "enc",
        })
    );
}

#[test]
fn tools_map_flat_with_nullable_parameters_and_strict() {
    let mut req = Request::with_messages(vec![Message::user_text("hi")]);
    req.tools = Some(vec![
        Tool::function(
            FunctionTool::new("get_weather")
                .with_description("Weather.")
                .with_parameters(json!({"type": "object"}))
                .with_strict(true),
        ),
        Tool::function(FunctionTool::new("noop").with_strict(false)),
        Tool::function(FunctionTool::new("bare")),
        Tool::opaque(F, json!({"type": "web_search"})),
    ]);
    let built = build(&req);
    let body = body_of(&built);
    assert_eq!(
        body["tools"],
        json!([
            {"type": "function", "name": "get_weather", "description": "Weather.",
             "parameters": {"type": "object"}, "strict": true},
            {"type": "function", "name": "noop", "parameters": null, "strict": false},
            {"type": "function", "name": "bare", "parameters": null, "strict": null},
            {"type": "web_search"},
        ])
    );
    assert!(built.warnings.is_empty());
}

#[test]
fn tool_cache_extra_and_foreign_opaque() {
    let mut req = Request::with_messages(vec![Message::user_text("hi")]);
    let mut tool = FunctionTool::new("t").with_cache(CacheHint::new());
    tool.extra
        .set(F, "output_schema", json!({"type": "string"}));
    req.tools = Some(vec![
        Tool::function(tool),
        Tool::opaque("anthropic_messages", json!({"type": "computer_20250124"})),
    ]);
    let built = build(&req);
    let body = body_of(&built);
    let tools = body["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 1, "foreign opaque tool must be dropped");
    assert_eq!(tools[0]["output_schema"], json!({"type": "string"}));
    assert!(has_code(&built.warnings, &WarningCode::CacheHintDropped));
    let opaque = built
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::OpaqueDropped)
        .unwrap();
    assert_eq!(opaque.severity, WarningSeverity::Semantic);
}

#[test]
fn tool_choice_forms() {
    for (choice, expected) in [
        (ToolChoice::Auto, json!("auto")),
        (ToolChoice::None, json!("none")),
        (ToolChoice::Required, json!("required")),
        (
            ToolChoice::tool("get_weather"),
            json!({"type": "function", "name": "get_weather"}),
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
fn all_foreign_tools_dropped_suppresses_tools_and_parallel() {
    // Every tool is a foreign opaque: the wire has no `tools` key, so
    // `parallel_tool_calls` is not emitted either.
    let mut req = Request::with_messages(vec![Message::user_text("hi")]);
    req.tools = Some(vec![Tool::opaque(
        "anthropic_messages",
        json!({"type": "computer_20250124"}),
    )]);
    req.parallel_tool_calls = Some(true);
    let built = build(&req);
    let body = body_of(&built);
    assert!(body.get("tools").is_none());
    assert!(body.get("parallel_tool_calls").is_none());
    assert!(has_code(&built.warnings, &WarningCode::OpaqueDropped));
    assert!(has_code(
        &built.warnings,
        &WarningCode::ParallelToolCallsIgnored
    ));
}

#[test]
fn explicitly_empty_tools_list_replays_verbatim() {
    // IR `Some(vec![])` is user data and replays as `"tools": []` — it
    // still counts as "no tools" for `parallel_tool_calls`.
    let mut req = Request::with_messages(vec![Message::user_text("hi")]);
    req.tools = Some(vec![]);
    req.parallel_tool_calls = Some(true);
    let built = build(&req);
    let body = body_of(&built);
    assert_eq!(body["tools"], json!([]));
    assert!(body.get("parallel_tool_calls").is_none());
    assert!(has_code(
        &built.warnings,
        &WarningCode::ParallelToolCallsIgnored
    ));

    // And the wire shape round-trips.
    let (parsed, _) = request_to_ir(br#"{"input": "hi", "tools": []}"#).unwrap();
    assert_eq!(parsed.tools, Some(vec![]));
    let (body, ws) =
        request_from_ir(&parsed, None, CallMode::Unary, &ConvertOptions::default()).unwrap();
    assert!(ws.is_empty(), "{ws:?}");
    assert_eq!(body["tools"], json!([]));
}

#[test]
fn reasoning_mapping() {
    // enabled: false -> effort "none".
    let mut req = Request::with_messages(vec![Message::user_text("hi")]);
    req.reasoning = Some(Reasoning::enabled(false));
    assert_eq!(
        body_of(&build(&req))["reasoning"],
        json!({"effort": "none"})
    );

    // enabled: true alone is a no-op.
    req.reasoning = Some(Reasoning::enabled(true));
    assert!(body_of(&build(&req)).get("reasoning").is_none());

    // effort tiers pass through, include_thoughts: true -> summary auto.
    let mut r = Reasoning::effort(Effort::XHigh);
    r.include_thoughts = Some(true);
    req.reasoning = Some(r);
    assert_eq!(
        body_of(&build(&req))["reasoning"],
        json!({"effort": "xhigh", "summary": "auto"})
    );

    // include_thoughts: false omits summary.
    let mut r = Reasoning::effort(Effort::Low);
    r.include_thoughts = Some(false);
    req.reasoning = Some(r);
    assert_eq!(body_of(&build(&req))["reasoning"], json!({"effort": "low"}));

    // Effort::Other passes verbatim, warning-free.
    req.reasoning = Some(Reasoning::effort(Effort::Other("turbo".into())));
    let built = build(&req);
    assert_eq!(body_of(&built)["reasoning"], json!({"effort": "turbo"}));
    assert!(built.warnings.is_empty());

    // Conflicts: effort wins, cosmetic warning.
    let mut r = Reasoning::enabled(true);
    r.effort = Some(Effort::None);
    req.reasoning = Some(r);
    let built = build(&req);
    assert_eq!(body_of(&built)["reasoning"], json!({"effort": "none"}));
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
    assert_eq!(body_of(&built)["reasoning"], json!({"effort": "high"}));
    assert!(has_code(&built.warnings, &WarningCode::ReasoningConflict));

    // Reasoning.extra rides the reasoning object (summary: "detailed").
    let mut r = Reasoning::effort(Effort::Medium);
    r.extra.set(F, "summary", "detailed");
    req.reasoning = Some(r);
    assert_eq!(
        body_of(&build(&req))["reasoning"],
        json!({"effort": "medium", "summary": "detailed"})
    );
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
        body_of(&build(&req))["text"],
        json!({"format": {"type": "json_schema", "name": "result", "schema": {"type": "object"}, "description": "d", "strict": true}})
    );

    // The upstream-required name synthesizes as "response" when unset.
    req.output_format = Some(OutputFormat::json_schema(json!({"type": "object"})));
    assert_eq!(
        body_of(&build(&req))["text"]["format"]["name"],
        json!("response")
    );

    req.output_format = Some(OutputFormat::json_object());
    assert_eq!(
        body_of(&build(&req))["text"],
        json!({"format": {"type": "json_object"}})
    );
}

#[test]
fn tool_call_and_assistant_text_cache_hints_drop() {
    let msg = Message::assistant(vec![
        ContentBlock::text("t").with_cache(CacheHint::new()),
        ContentBlock::tool_call_with_id("c1", "f", "{}").with_cache(CacheHint::new()),
    ]);
    let tool = Message::tool(vec![ContentBlock::tool_result_text(Some("c1".into()), "r")]);
    let built = build(&Request::with_messages(vec![msg, tool]));
    let hints: Vec<_> = built
        .warnings
        .iter()
        .filter(|w| w.code == WarningCode::CacheHintDropped)
        .collect();
    assert_eq!(hints.len(), 2);
    assert!(
        hints
            .iter()
            .all(|w| w.severity == WarningSeverity::Cosmetic)
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
    assert_eq!(body_of(&built)["input"].as_array().unwrap().len(), 2);

    // Passthrough: trailing orphans are sent as-is without warnings.
    let trailing = Request::with_messages(vec![
        Message::user_text("go"),
        Message::assistant(vec![
            ContentBlock::thinking_signed("plan", "enc"),
            ContentBlock::tool_call_with_id("c2", "f", "{}"),
        ]),
    ]);
    let built = build(&trailing);
    assert!(built.warnings.is_empty());
    assert_eq!(body_of(&built)["input"].as_array().unwrap().len(), 3);

    // DropTrailing removes the calls and flags the now-orphaned thinking.
    let mut dctx = ctx(CallMode::Unary);
    dctx.convert.orphan_tool_calls = OrphanToolCalls::DropTrailing;
    let built = OpenAiResponses.build_request(&trailing, &dctx).unwrap();
    let body = body_of(&built);
    assert_eq!(body["input"].as_array().unwrap().len(), 2);
    assert_eq!(body["input"][1]["type"], json!("reasoning"));
    assert!(has_code(
        &built.warnings,
        &WarningCode::OrphanToolCallsDropped
    ));
    assert!(has_code(&built.warnings, &WarningCode::ThinkingOrphaned));

    // SynthesizeError appends an error tool result per orphan; the error
    // marker itself has no Responses channel and warns.
    let mut sctx = ctx(CallMode::Unary);
    sctx.convert.orphan_tool_calls = OrphanToolCalls::SynthesizeError;
    let built = OpenAiResponses.build_request(&trailing, &sctx).unwrap();
    let body = body_of(&built);
    let items = body["input"].as_array().unwrap();
    assert_eq!(
        items.last().unwrap(),
        &json!({"type": "function_call_output", "call_id": "c2", "output": "cancelled", "name": "f"})
    );
    assert!(has_code(
        &built.warnings,
        &WarningCode::OrphanToolCallsSynthesized
    ));
    assert!(has_code(&built.warnings, &WarningCode::IsErrorDropped));
}

#[test]
fn missing_thinking_with_tool_calls() {
    let base = Request::with_messages(vec![
        Message::user_text("go"),
        Message::assistant(vec![ContentBlock::tool_call_with_id("c1", "f", "{}")]),
        Message::tool(vec![ContentBlock::tool_result_text(
            Some("c1".into()),
            "ok",
        )]),
    ]);
    let mut req = base.clone();
    req.reasoning = Some(Reasoning::effort(Effort::Medium));
    let built = build(&req);
    let w = built
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::MissingThinkingWithToolCalls)
        .unwrap();
    assert_eq!(w.severity, WarningSeverity::Semantic);

    // Effort::None does not enable thinking; no warning.
    let mut off = base.clone();
    off.reasoning = Some(Reasoning::effort(Effort::None));
    assert!(build(&off).warnings.is_empty());

    // fill_missing_thinking inserts plaintext thinking; on this
    // signature-validated format it is foreign and still drops — the
    // warning says so instead of reading as a successful repair.
    let mut fctx = ctx(CallMode::Unary);
    fctx.convert.fill_missing_thinking = Some("tool call".into());
    let built = OpenAiResponses.build_request(&req, &fctx).unwrap();
    let filled = built
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::MissingThinkingFilled)
        .expect("filled warning");
    assert_eq!(
        filled.message,
        "inserted a placeholder thinking block into assistant message 1; this format drops the \
         placeholder during serialization (no unsigned thinking channel) — set \
         `thinking_as_text` to carry it as text"
    );
    assert!(has_code(&built.warnings, &WarningCode::ThinkingDropped));
    assert!(!has_code(
        &built.warnings,
        &WarningCode::MissingThinkingWithToolCalls
    ));

    // With thinking_as_text the filled text becomes raw reasoning text
    // and the warning has no drop disclaimer.
    fctx.convert.thinking_as_text = true;
    let built = OpenAiResponses.build_request(&req, &fctx).unwrap();
    let body = body_of(&built);
    assert_eq!(
        body["input"][1],
        json!({"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "tool call"}]})
    );
    assert!(!has_code(&built.warnings, &WarningCode::ThinkingDropped));
    let filled = built
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::MissingThinkingFilled)
        .expect("filled warning");
    assert_eq!(
        filled.message,
        "inserted a placeholder thinking block into assistant message 1"
    );
}

#[test]
fn hooks_visit_serialized_items_in_order() {
    let mut req = Request::with_messages(vec![
        Message::user_text("hi"),
        Message::assistant(vec![
            ContentBlock::thinking_signed("plan", "enc"),
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
    let built = OpenAiResponses.build_request(&req, &hctx).unwrap();
    let body = body_of(&built);
    assert_eq!(
        *seen.lock().unwrap(),
        vec![
            (0, Role::User),
            (1, Role::Assistant),
            (2, Role::Assistant),
            (3, Role::Assistant),
            (4, Role::Tool),
            (5, Role::Tool),
        ],
        "every serialized item is visited; instructions is not"
    );
    for (i, item) in body["input"].as_array().unwrap().iter().enumerate() {
        assert_eq!(item["_hooked"], json!(i));
    }
    assert_eq!(body["_done"], json!(true));
    assert_eq!(body["instructions"], json!("sys"));
}

#[test]
fn hook_error_aborts() {
    let mut hctx = ctx(CallMode::Unary);
    hctx.hooks = RequestHooks::new().with_on_request(|_| Err(llm_api::HookError::new("nope")));
    let err = OpenAiResponses
        .build_request(
            &Request::with_messages(vec![Message::user_text("hi")]),
            &hctx,
        )
        .unwrap_err();
    assert!(matches!(err, Error::Hook(_)));
}

// ---------------------------------------------------------------- round trips

#[test]
fn canonical_request_round_trip_is_idempotent() {
    let bytes = fixture("request_canonical.json");
    let canonical = fixture_json("request_canonical.json");
    let (req, parse_warnings) = OpenAiResponses.parse_request(&bytes).unwrap();
    assert!(parse_warnings.is_empty(), "{parse_warnings:?}");

    let (body, build_warnings) = request_from_ir(
        &req,
        Some("gpt-5.1"),
        CallMode::Unary,
        &ConvertOptions::default(),
    )
    .unwrap();
    assert!(build_warnings.is_empty(), "{build_warnings:?}");
    assert_eq!(body, canonical);

    // Idempotence: a second parse/serialize pass is a fixed point.
    let (req2, _) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    let (body2, _) = request_from_ir(
        &req2,
        Some("gpt-5.1"),
        CallMode::Unary,
        &ConvertOptions::default(),
    )
    .unwrap();
    assert_eq!(body2, canonical);
}

#[test]
fn canonical_request_parses_expected_ir_shape() {
    let (req, _) = OpenAiResponses
        .parse_request(&fixture("request_canonical.json"))
        .unwrap();
    assert_eq!(req.system.as_ref().unwrap().len(), 1);
    let roles: Vec<Role> = req.messages.iter().map(|m| m.role).collect();
    assert_eq!(
        roles,
        vec![
            Role::System,
            Role::Developer,
            Role::User,
            Role::Assistant, // reasoning + message + 2 calls + web_search grouped
            Role::Tool,
            Role::Tool,
            Role::User,
        ]
    );
    let assistant = &req.messages[3];
    assert_eq!(assistant.content.len(), 5);
    assert!(
        matches!(&assistant.content[0], ContentBlock::Thinking { text: Some(t), signature: Some(s), .. }
        if t == "Consider the image." && s == "enc_1")
    );
    assert!(matches!(&assistant.content[1], ContentBlock::Text { text, .. } if text == "A cat."));
    assert!(
        matches!(&assistant.content[2], ContentBlock::ToolCall { id: Some(id), .. } if id == "call_1")
    );
    assert!(matches!(&assistant.content[4], ContentBlock::Opaque { format, .. } if format == F));
    // Tool messages carry one result each; the empty-vs-string encodings hold.
    assert!(
        matches!(&req.messages[4].content[0], ContentBlock::ToolResult { content, .. } if content.len() == 1)
    );
    assert!(
        matches!(&req.messages[5].content[0], ContentBlock::ToolResult { content, name: Some(n), .. }
        if content.len() == 2 && n == "get_radar")
    );
    // Parameters and knobs.
    assert_eq!(req.max_output_tokens, Some(512));
    assert_eq!(req.cache_key.as_deref(), Some("cache-1"));
    let reasoning = req.reasoning.as_ref().unwrap();
    assert_eq!(reasoning.effort, Some(Effort::Medium));
    assert_eq!(reasoning.include_thoughts, Some(true));
    assert!(matches!(
        req.output_format,
        Some(OutputFormat::JsonSchema { .. })
    ));
    assert_eq!(req.tools.as_ref().unwrap().len(), 3);
    assert_eq!(req.tool_choice, Some(ToolChoice::Auto));
    // Unknown top-level fields landed in the request extra namespace.
    let ns = req.extra.get(F).unwrap();
    assert_eq!(ns.get("store"), Some(&json!(false)));
    assert_eq!(
        ns.get("include"),
        Some(&json!(["reasoning.encrypted_content"]))
    );
}

#[test]
fn ir_round_trip_preserves_modeled_fields() {
    let mut req = Request::with_messages(vec![
        Message::user(vec![
            ContentBlock::text("look").with_cache(CacheHint::new()),
            ContentBlock::image_base64("image/png", "QUJD"),
        ]),
        Message::assistant(vec![
            ContentBlock::thinking_signed("plan", "enc"),
            ContentBlock::text("answer"),
            ContentBlock::tool_call_with_id("c1", "f", "{\"a\":1}"),
        ]),
        Message::tool(vec![ContentBlock::tool_result_text(
            Some("c1".into()),
            "ok",
        )]),
    ]);
    req.system = Some(vec![ContentBlock::text("sys")]);
    req.max_output_tokens = Some(9);
    req.temperature = Some(0.1);
    req.tool_choice = Some(ToolChoice::Required);
    req.tools = Some(vec![Tool::function(
        FunctionTool::new("f").with_strict(true),
    )]);
    req.parallel_tool_calls = Some(true);
    let mut reasoning = Reasoning::effort(Effort::Medium);
    reasoning.include_thoughts = Some(true);
    req.reasoning = Some(reasoning.clone());
    req.output_format = Some(OutputFormat::json_object());

    let (body, _) =
        request_from_ir(&req, Some("m"), CallMode::Unary, &ConvertOptions::default()).unwrap();
    let (back, _) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();

    assert_eq!(back.system, req.system);
    assert_eq!(back.messages.len(), 3);
    assert_eq!(back.messages[0], req.messages[0]);
    // Assistant blocks: modeled fields survive; parsing adds the canonical
    // summary array to the thinking block's namespace.
    match (&back.messages[1].content[0], &req.messages[1].content[0]) {
        (
            ContentBlock::Thinking {
                text: t1,
                signature: s1,
                ..
            },
            ContentBlock::Thinking {
                text: t2,
                signature: s2,
                ..
            },
        ) => {
            assert_eq!(t1, t2);
            assert_eq!(s1, s2);
        }
        other => panic!("unexpected blocks: {other:?}"),
    }
    assert!(
        matches!(&back.messages[1].content[1], ContentBlock::Text { text, .. } if text == "answer")
    );
    assert_eq!(back.messages[1].content[2], req.messages[1].content[2]);
    assert_eq!(back.messages[2], req.messages[2]);
    assert_eq!(back.max_output_tokens, req.max_output_tokens);
    assert_eq!(back.temperature, req.temperature);
    assert_eq!(back.tool_choice, req.tool_choice);
    assert_eq!(back.tools, req.tools);
    assert_eq!(back.parallel_tool_calls, req.parallel_tool_calls);
    assert_eq!(back.reasoning, Some(reasoning));
    assert_eq!(back.output_format, req.output_format);
}

#[test]
fn request_parse_canonicalizes_shorthands() {
    let (req, warnings) =
        request_to_ir(br#"{"model": "m", "stream": true, "input": "hi", "instructions": "sys"}"#)
            .unwrap();
    assert!(warnings.is_empty());
    assert_eq!(req.messages, vec![Message::user_text("hi")]);
    assert_eq!(req.system, Some(vec![ContentBlock::text("sys")]));
    // model/stream are configuration, not IR data.
    assert!(req.extra.is_empty());

    let (body, _) =
        request_from_ir(&req, Some("m"), CallMode::Unary, &ConvertOptions::default()).unwrap();
    assert_eq!(
        body,
        json!({
            "model": "m",
            "instructions": "sys",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        })
    );

    // A message-content string canonicalizes to a part array; a message
    // item without `type` gains it.
    let (req, _) = request_to_ir(br#"{"input": [{"role": "user", "content": "yo"}]}"#).unwrap();
    let (body, _) =
        request_from_ir(&req, None, CallMode::Unary, &ConvertOptions::default()).unwrap();
    assert_eq!(
        body["input"][0],
        json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "yo"}]})
    );
}

#[test]
fn request_parse_mirrors_unmodeled_fields_into_extra() {
    let wire = json!({
        "instructions": [{"type": "message", "role": "system", "content": "echoed"}],
        "input": "hi",
        "tool_choice": {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": "f"}]},
        "text": {"format": {"type": "text"}, "verbosity": "low"},
        "service_tier": "flex",
    });
    let (req, warnings) = request_to_ir(&serde_json::to_vec(&wire).unwrap()).unwrap();
    assert!(has_code(&warnings, &WarningCode::MalformedField)); // array instructions
    assert!(req.system.is_none());
    assert!(req.tool_choice.is_none());
    assert!(req.output_format.is_none());
    let ns = req.extra.get(F).unwrap();
    assert!(ns.contains_key("instructions"));
    assert!(ns.contains_key("tool_choice"));
    assert_eq!(
        ns.get("text"),
        Some(&json!({"format": {"type": "text"}, "verbosity": "low"}))
    );
    assert_eq!(ns.get("service_tier"), Some(&json!("flex")));

    // Everything is restored verbatim on re-serialization.
    let (body, _) =
        request_from_ir(&req, None, CallMode::Unary, &ConvertOptions::default()).unwrap();
    assert_eq!(body["instructions"], wire["instructions"]);
    assert_eq!(body["tool_choice"], wire["tool_choice"]);
    assert_eq!(body["text"], wire["text"]);
    assert_eq!(body["service_tier"], wire["service_tier"]);
}

#[test]
fn unknown_message_role_parses_to_opaque_with_warning() {
    // A message item with a role outside user/system/developer/assistant
    // must not be silently rewritten: it stays verbatim as `Opaque`.
    let wire = json!({
        "input": [
            {"type": "message", "role": "critic", "content": [{"type": "input_text", "text": "meh"}]},
        ],
    });
    let (req, warnings) = request_to_ir(&serde_json::to_vec(&wire).unwrap()).unwrap();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].code, WarningCode::MalformedField);
    assert_eq!(warnings[0].severity, WarningSeverity::Cosmetic);
    assert_eq!(warnings[0].location, "/input/0");
    assert_eq!(req.messages.len(), 1);
    assert_eq!(req.messages[0].role, Role::Assistant);
    assert_eq!(
        req.messages[0].content,
        vec![ContentBlock::opaque(F, wire["input"][0].clone())]
    );
}

#[test]
fn unknown_message_role_round_trips_verbatim() {
    // Neither the role nor the content part types get canonicalized.
    let wire = json!({
        "input": [
            {"type": "message", "role": "critic", "content": [
                {"type": "input_text", "text": "meh"},
                {"type": "custom_part", "data": 1},
            ]},
        ],
    });
    let (req, _) = request_to_ir(&serde_json::to_vec(&wire).unwrap()).unwrap();
    let (body, build_warnings) =
        request_from_ir(&req, None, CallMode::Unary, &ConvertOptions::default()).unwrap();
    assert!(build_warnings.is_empty(), "{build_warnings:?}");
    assert_eq!(body, wire);

    // Idempotence: a second parse/serialize pass is a fixed point.
    let (req2, _) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    let (body2, _) =
        request_from_ir(&req2, None, CallMode::Unary, &ConvertOptions::default()).unwrap();
    assert_eq!(body2, wire);
}

#[test]
fn known_message_roles_parse_without_warnings() {
    let wire = json!({
        "input": [
            {"type": "message", "role": "system", "content": [{"type": "input_text", "text": "s"}]},
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "d"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "u"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "a", "annotations": []}]},
        ],
    });
    let (req, warnings) = request_to_ir(&serde_json::to_vec(&wire).unwrap()).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    let roles: Vec<Role> = req.messages.iter().map(|m| m.role).collect();
    assert_eq!(
        roles,
        vec![Role::System, Role::Developer, Role::User, Role::Assistant]
    );
    let (body, build_warnings) =
        request_from_ir(&req, None, CallMode::Unary, &ConvertOptions::default()).unwrap();
    assert!(build_warnings.is_empty(), "{build_warnings:?}");
    assert_eq!(body, wire);
}

#[test]
fn unknown_message_role_preserves_position_among_items() {
    // The opaque item keeps its slot between mapped items, whether next to
    // input messages or inside an assistant run; string content stays a
    // string.
    let wire = json!({
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "q"}]},
            {"type": "message", "role": "critic", "content": "meh"},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "a", "annotations": []}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "next"}]},
        ],
    });
    let (req, warnings) = request_to_ir(&serde_json::to_vec(&wire).unwrap()).unwrap();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].code, WarningCode::MalformedField);
    assert_eq!(warnings[0].location, "/input/1");
    // The critic item joins the adjacent assistant run, in item order.
    let roles: Vec<Role> = req.messages.iter().map(|m| m.role).collect();
    assert_eq!(roles, vec![Role::User, Role::Assistant, Role::User]);
    assert_eq!(
        req.messages[1].content[0],
        ContentBlock::opaque(F, wire["input"][1].clone())
    );
    assert!(matches!(&req.messages[1].content[1], ContentBlock::Text { text, .. } if text == "a"));

    let (body, build_warnings) =
        request_from_ir(&req, None, CallMode::Unary, &ConvertOptions::default()).unwrap();
    assert!(build_warnings.is_empty(), "{build_warnings:?}");
    assert_eq!(body, wire);
}

#[test]
fn tool_result_output_encodings_round_trip() {
    // § 7.2: `""` <-> empty list, string shorthand <-> single text, part
    // arrays keep boundaries; nested breakpoints stay verbatim in extra.
    let wire = json!({
        "input": [
            {"type": "function_call_output", "call_id": "c1", "output": ""},
            {"type": "function_call_output", "call_id": "c2", "output": "just text"},
            {"type": "function_call_output", "call_id": "c3", "output": [
                {"type": "input_text", "text": "a"},
                {"type": "input_text", "text": "b", "prompt_cache_breakpoint": {"mode": "explicit"}},
            ]},
        ],
    });
    let (req, warnings) = request_to_ir(&serde_json::to_vec(&wire).unwrap()).unwrap();
    assert!(warnings.is_empty());
    let contents: Vec<usize> = req
        .messages
        .iter()
        .map(|m| match &m.content[0] {
            ContentBlock::ToolResult { content, .. } => content.len(),
            other => panic!("expected tool result, got {other:?}"),
        })
        .collect();
    assert_eq!(contents, vec![0, 1, 2]);
    // The nested breakpoint is not an IR hint in v1 (build drops nested
    // hints); it round-trips through the part extra instead.
    let ContentBlock::ToolResult { content, .. } = &req.messages[2].content[0] else {
        panic!("expected tool result");
    };
    let ToolOutputBlock::Text { cache, extra, .. } = &content[1] else {
        panic!("expected text output block");
    };
    assert!(cache.is_none());
    assert!(
        extra
            .get(F)
            .unwrap()
            .contains_key("prompt_cache_breakpoint")
    );

    let (body, build_warnings) =
        request_from_ir(&req, None, CallMode::Unary, &ConvertOptions::default()).unwrap();
    assert!(build_warnings.is_empty(), "{build_warnings:?}");
    assert_eq!(body, wire);
}

#[test]
fn nested_unknown_fields_mirror_and_round_trip() {
    let wire = json!({
        "input": "hi",
        "reasoning": {"effort": "low", "context": "all_turns", "mode": "pro"},
    });
    let (req, _) = request_to_ir(&serde_json::to_vec(&wire).unwrap()).unwrap();
    let reasoning = req.reasoning.as_ref().unwrap();
    assert_eq!(reasoning.effort, Some(Effort::Low));
    let ns = reasoning.extra.get(F).unwrap();
    assert_eq!(ns.get("context"), Some(&json!("all_turns")));
    assert_eq!(ns.get("mode"), Some(&json!("pro")));
    let (body, _) =
        request_from_ir(&req, None, CallMode::Unary, &ConvertOptions::default()).unwrap();
    assert_eq!(body["reasoning"], wire["reasoning"]);
}

#[test]
fn foreign_opaque_content_drops_with_warning() {
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
        body_of(&built)["input"][0]["content"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn all_blocks_dropped_assistant_message_omitted_with_warning() {
    // A thinking-only assistant message whose block is foreign here
    // serializes to zero items: the message is omitted and the omission
    // is disclosed on top of the per-block warning.
    let req = Request::with_messages(vec![
        Message::user_text("hi"),
        Message::assistant(vec![ContentBlock::thinking("chain").with_extra(
            "anthropic_messages",
            "redacted",
            false,
        )]),
        Message::user_text("still there?"),
    ]);
    let built = build(&req);
    let body = body_of(&built);
    assert_eq!(body["input"].as_array().unwrap().len(), 2);
    assert!(has_code(&built.warnings, &WarningCode::ThinkingDropped));
    let w = built
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::EmptyMessageDropped)
        .expect("omission disclosed");
    assert_eq!(w.severity, WarningSeverity::Semantic);
    assert_eq!(w.location, "/input");
    assert!(w.message.contains("message 1"), "{}", w.message);
}

#[test]
fn all_blocks_dropped_user_message_omitted_not_empty_item() {
    // A user message holding only foreign opaques must not leave an empty
    // `content` item behind: the item is omitted and warned.
    let req = Request::with_messages(vec![
        Message::user(vec![ContentBlock::opaque(
            "google_generate_content",
            json!({"executableCode": {}}),
        )]),
        Message::user_text("hi"),
    ]);
    let built = build(&req);
    let body = body_of(&built);
    assert_eq!(
        body["input"],
        json!([{"type": "message", "role": "user",
                "content": [{"type": "input_text", "text": "hi"}]}])
    );
    assert!(has_code(&built.warnings, &WarningCode::OpaqueDropped));
    let w = built
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::EmptyMessageDropped)
        .expect("omission disclosed");
    assert_eq!(w.location, "/input");
    assert!(w.message.contains("message 0"), "{}", w.message);
}

#[test]
fn all_blocks_dropped_tool_message_omitted_with_warning() {
    // The same rule on the tool path: a tool message holding only foreign
    // opaques contributes no `function_call_output` item.
    let req = Request::with_messages(vec![Message::tool(vec![ContentBlock::opaque(
        "anthropic_messages",
        json!({"type": "tool_result", "tool_use_id": "t1"}),
    )])]);
    let built = build(&req);
    assert!(body_of(&built).get("input").is_none());
    assert!(has_code(&built.warnings, &WarningCode::OpaqueDropped));
    assert!(has_code(&built.warnings, &WarningCode::EmptyMessageDropped));
}

#[test]
fn truly_empty_messages_keep_their_wire_shape() {
    // User-built zero-block messages are not the dropped-blocks case:
    // user/system/developer replay as an empty `content` item;
    // assistant/tool messages emit nothing — all without warnings.
    let req = Request::with_messages(vec![
        Message::user(vec![]),
        Message::assistant(vec![]),
        Message::tool(vec![]),
    ]);
    let built = build(&req);
    assert!(built.warnings.is_empty(), "{:?}", built.warnings);
    assert_eq!(
        body_of(&built)["input"],
        json!([{"type": "message", "role": "user", "content": []}])
    );
}

// ---------------------------------------------------------------- responses

#[test]
fn response_text_parses_envelope_blocks_and_usage() {
    let resp = response_to_ir(&fixture("response_text.json"), &meta_ok()).unwrap();
    assert_eq!(resp.id.as_deref(), Some("resp_1"));
    assert_eq!(resp.model.as_deref(), Some("gpt-5.1"));
    assert_eq!(resp.status, 200);
    assert_eq!(resp.stop_reason, Some(StopReason::EndTurn));
    assert_eq!(resp.message.role, Role::Assistant);
    assert_eq!(resp.message.content.len(), 1);
    let ContentBlock::Text { text, extra, .. } = &resp.message.content[0] else {
        panic!("expected text block");
    };
    assert_eq!(text, "Hello from Paris.");
    let ns = extra.get(F).unwrap();
    assert_eq!(ns.get("id"), Some(&json!("msg_1")));
    assert_eq!(ns.get("status"), Some(&json!("completed")));
    assert_eq!(
        ns.get("annotations")
            .and_then(Value::as_array)
            .map(Vec::len),
        Some(1)
    );
    let usage = resp.usage.as_ref().unwrap();
    assert_eq!(usage.input_tokens, 36); // already includes cached tokens
    assert_eq!(usage.output_tokens, 87);
    assert_eq!(usage.total_tokens, Some(123));
    assert_eq!(usage.cache_read_tokens, Some(12));
    assert_eq!(usage.cache_write_tokens, Some(3));
    assert_eq!(usage.reasoning_tokens, Some(5));
    assert!(resp.raw.is_some());
    assert!(resp.warnings.is_empty());
}

#[test]
fn response_tool_call_parses_reasoning_and_derives_tool_use() {
    let resp = response_to_ir(&fixture("response_tool_call.json"), &meta_ok()).unwrap();
    assert_eq!(resp.stop_reason, Some(StopReason::ToolUse));
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
    assert_eq!(
        text.as_deref(),
        Some("Need the weather.\n\nCalling the tool.")
    );
    assert_eq!(signature.as_deref(), Some("enc_2"));
    let ns = extra.get(F).unwrap();
    assert_eq!(ns.get("id"), Some(&json!("rs_2")));
    assert_eq!(
        ns.get("summary").and_then(Value::as_array).map(Vec::len),
        Some(2)
    );
    assert_eq!(ns.get("status"), Some(&json!("completed")));
    let ContentBlock::ToolCall {
        id,
        name,
        arguments,
        extra,
        ..
    } = &resp.message.content[1]
    else {
        panic!("expected tool call");
    };
    assert_eq!(id.as_deref(), Some("call_9"));
    assert_eq!(name, "get_weather");
    assert_eq!(arguments, "{\"city\":\"Paris\"}");
    assert_eq!(extra.get(F).unwrap().get("id"), Some(&json!("fc_9")));
}

#[test]
fn response_refusal_normalizes_stop_reason() {
    let resp = response_to_ir(&fixture("response_refusal.json"), &meta_ok()).unwrap();
    assert_eq!(resp.stop_reason, Some(StopReason::Refusal));
    let ContentBlock::Text { text, extra, .. } = &resp.message.content[0] else {
        panic!("expected refusal-marked text block");
    };
    assert_eq!(text, "I cannot help with that.");
    assert_eq!(
        extra.get(F).unwrap().get(text_block_reserved_key::REFUSAL),
        Some(&json!(true))
    );
    assert!(llm_api::ir::is_refusal_block(&resp.message.content[0]));
}

#[test]
fn refusal_part_round_trips_via_marker() {
    // The real refusal channel: a wire `refusal` part parses into a
    // `Text` block carrying the internal `__llm_api_refusal` marker and
    // serializes back into the same `refusal` part.
    let part = json!({"type": "refusal", "refusal": "No."});
    let item = json!({"type": "message", "role": "assistant", "content": [part]});
    let wire = json!({"input": [item]});
    let (req, warnings) = request_to_ir(&serde_json::to_vec(&wire).unwrap()).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    let block = &req.messages[0].content[0];
    let ContentBlock::Text { text, extra, .. } = block else {
        panic!("expected refusal-marked text, got {block:?}");
    };
    assert_eq!(text, "No.");
    assert_eq!(
        extra.get(F).unwrap().get(text_block_reserved_key::REFUSAL),
        Some(&json!(true))
    );
    assert!(llm_api::ir::is_refusal_block(block));

    let (body, ws) =
        request_from_ir(&req, Some("m"), CallMode::Unary, &ConvertOptions::default()).unwrap();
    assert!(ws.is_empty(), "{ws:?}");
    assert_eq!(body["input"], json!([item]));
}

#[test]
fn stray_refusal_field_on_output_text_part_round_trips_verbatim() {
    // A wire `output_text` part with an unknown field named `refusal` is
    // plain unknown data — the internal marker key is `__llm_api_refusal`,
    // so the part must not be rewritten into a refusal part.
    let part = json!({"type": "output_text", "text": "hi", "annotations": [], "refusal": true});
    let item = json!({"type": "message", "role": "assistant", "content": [part]});
    let wire = json!({"input": [item]});
    let (req, warnings) = request_to_ir(&serde_json::to_vec(&wire).unwrap()).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    let block = &req.messages[0].content[0];
    assert!(matches!(block, ContentBlock::Text { .. }));
    assert!(!llm_api::ir::is_refusal_block(block));

    let (body, ws) =
        request_from_ir(&req, Some("m"), CallMode::Unary, &ConvertOptions::default()).unwrap();
    assert!(ws.is_empty(), "{ws:?}");
    assert_eq!(body["input"], json!([item]));

    // A response carrying the same stray field keeps its stop reason.
    let resp_wire = json!({
        "id": "resp_s", "object": "response", "status": "completed", "model": "m",
        "output": [item],
    });
    let resp = response_to_ir(&serde_json::to_vec(&resp_wire).unwrap(), &meta_ok()).unwrap();
    assert_ne!(resp.stop_reason, Some(StopReason::Refusal));
    assert!(!llm_api::ir::is_refusal_block(&resp.message.content[0]));
}

#[test]
fn custom_tool_call_derives_tool_use_stop_reason() {
    // `custom_tool_call` awaits developer-supplied output exactly like
    // `function_call` and upgrades `completed` → `ToolUse`; the item
    // itself stays an `Opaque` block (not a typed `ToolCall`).
    let body = json!({
        "id": "resp_ct", "object": "response", "status": "completed", "model": "m",
        "output": [
            {"type": "message", "role": "assistant", "id": "msg_1", "status": "completed",
             "content": [{"type": "output_text", "text": "running", "annotations": []}]},
            {"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_c1",
             "name": "grep", "input": "-r foo"},
        ],
    });
    let resp = response_to_ir(&serde_json::to_vec(&body).unwrap(), &meta_ok()).unwrap();
    assert_eq!(resp.stop_reason, Some(StopReason::ToolUse));
    assert_eq!(
        resp.message.content[1],
        ContentBlock::opaque(F, body["output"][1].clone())
    );

    // Other waiting-type items do not participate: `EndTurn` stands.
    let body = json!({
        "id": "resp_ap", "object": "response", "status": "completed", "model": "m",
        "output": [
            {"type": "mcp_approval_request", "id": "mcpr_1", "server_label": "s",
             "name": "run", "arguments": "{}"},
        ],
    });
    let resp = response_to_ir(&serde_json::to_vec(&body).unwrap(), &meta_ok()).unwrap();
    assert_eq!(resp.stop_reason, Some(StopReason::EndTurn));
}

#[test]
fn response_incomplete_reasons_map() {
    let resp = response_to_ir(&fixture("response_incomplete.json"), &meta_ok()).unwrap();
    assert_eq!(resp.stop_reason, Some(StopReason::MaxTokens));
    assert_eq!(resp.text(), "Once upon a");

    let resp = response_to_ir(&fixture("response_content_filter.json"), &meta_ok()).unwrap();
    assert_eq!(resp.stop_reason, Some(StopReason::ContentFilter));
    assert!(resp.message.content.is_empty());
}

#[test]
fn malformed_usage_degrades_to_none_with_warning() {
    // A malformed usage object (float tokens) must not fail the billed
    // 2xx response: usage degrades to `None` with a warning.
    let body = json!({
        "id": "resp_u", "object": "response", "status": "completed", "model": "m",
        "output": [{"type": "message", "role": "assistant", "id": "msg_1",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "hi", "annotations": []}]}],
        "usage": {"input_tokens": 12.5, "output_tokens": 3, "total_tokens": 15.5},
    });
    let resp = response_to_ir(&serde_json::to_vec(&body).unwrap(), &meta_ok()).unwrap();
    assert_eq!(resp.text(), "hi");
    assert!(resp.usage.is_none());
    let w = resp
        .warnings
        .iter()
        .find(|w| w.code == WarningCode::MalformedField)
        .expect("usage warning");
    assert_eq!(w.location, "/usage");
    assert!(w.message.contains("usage"), "{}", w.message);
    // The raw body still carries the original usage for inspection.
    assert_eq!(resp.raw.as_ref().unwrap()["usage"]["input_tokens"], 12.5);

    // A well-formed usage keeps parsing as before, warning-free.
    let body = json!({
        "id": "resp_u2", "object": "response", "status": "completed", "model": "m",
        "output": [],
        "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
    });
    let resp = response_to_ir(&serde_json::to_vec(&body).unwrap(), &meta_ok()).unwrap();
    assert!(resp.warnings.is_empty(), "{:?}", resp.warnings);
    assert_eq!(resp.usage.as_ref().unwrap().input_tokens, 3);
}

#[test]
fn response_failed_becomes_api_error() {
    let err = response_to_ir(&fixture("response_failed.json"), &meta_ok()).unwrap_err();
    match err {
        Error::Api {
            status,
            kind,
            message,
            parsed,
            ..
        } => {
            assert_eq!(status, 200);
            assert_eq!(kind, llm_api::ApiErrorKind::ServerError);
            assert_eq!(message, "The model had a bad day.");
            assert!(parsed.is_some());
        }
        other => panic!("expected Error::Api, got {other:?}"),
    }
}

#[test]
fn response_replay_reconstructs_items() {
    // The § 8 flow: a parsed response message re-enters a request as
    // history and reconstructs the provider items verbatim.
    let resp = response_to_ir(&fixture("response_tool_call.json"), &meta_ok()).unwrap();
    let mut req = Request::with_messages(vec![
        Message::user_text("weather?"),
        resp.message.clone(),
        Message::tool(vec![ContentBlock::tool_result_text(
            Some("call_9".into()),
            "21C",
        )]),
    ]);
    req.reasoning = Some(Reasoning::effort(Effort::Medium));
    let built = build(&req);
    let body = body_of(&built);
    assert!(built.warnings.is_empty(), "{:?}", built.warnings);
    assert_eq!(
        body["input"][1],
        json!({
            "type": "reasoning", "id": "rs_2",
            "summary": [
                {"type": "summary_text", "text": "Need the weather."},
                {"type": "summary_text", "text": "Calling the tool."},
            ],
            "encrypted_content": "enc_2", "status": "completed",
        })
    );
    assert_eq!(
        body["input"][2],
        json!({
            "type": "function_call", "call_id": "call_9", "name": "get_weather",
            "arguments": "{\"city\":\"Paris\"}", "id": "fc_9", "status": "completed",
        })
    );

    // An annotated text message replays with its annotations restored
    // into the content part.
    let resp = response_to_ir(&fixture("response_text.json"), &meta_ok()).unwrap();
    let req = Request::with_messages(vec![Message::user_text("hi"), resp.message]);
    let built = build(&req);
    assert_eq!(
        body_of(&built)["input"][1],
        json!({
            "type": "message", "role": "assistant", "id": "msg_1", "status": "completed",
            "content": [{
                "type": "output_text",
                "text": "Hello from Paris.",
                "annotations": [{
                    "type": "url_citation", "url": "https://example.com", "title": "Example",
                    "start_index": 0, "end_index": 5,
                }],
            }],
        })
    );
}

#[test]
fn response_output_unknown_role_message_kept_opaque() {
    let body = json!({
        "id": "resp_x", "object": "response", "status": "completed", "model": "m",
        "output": [
            {"type": "message", "role": "critic", "content": [{"type": "output_text", "text": "meh", "annotations": []}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok", "annotations": []}]},
        ],
    });
    let resp = response_to_ir(&serde_json::to_vec(&body).unwrap(), &meta_ok()).unwrap();
    assert_eq!(resp.message.content.len(), 2);
    assert_eq!(
        resp.message.content[0],
        ContentBlock::opaque(F, body["output"][0].clone())
    );
    assert!(matches!(&resp.message.content[1], ContentBlock::Text { text, .. } if text == "ok"));
    assert_eq!(resp.warnings.len(), 1, "{:?}", resp.warnings);
    assert_eq!(resp.warnings[0].code, WarningCode::MalformedField);
    assert_eq!(resp.warnings[0].location, "/output/0");
}

// ---------------------------------------------------------------- models

#[test]
fn models_request_and_response() {
    let built = OpenAiResponses
        .build_models_request(&ctx(CallMode::Unary), None)
        .unwrap();
    assert_eq!(built.method, http::Method::GET);
    assert_eq!(built.url.to_string(), "https://api.openai.com/v1/models");
    assert!(built.auth.is_some());
    assert!(built.body.is_empty());

    let (models, cursor) = OpenAiResponses
        .parse_models_response(&fixture("models_list.json"))
        .unwrap();
    assert!(cursor.is_none(), "OpenAI model listing is a single page");
    assert_eq!(models.len(), 2, "the id-less entry is skipped");
    assert_eq!(models[0].id, "gpt-5.1");
    assert!(models[0].display_name.is_none());
    let created = models[0].created.unwrap();
    assert_eq!(
        created
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        1_686_935_002
    );
    assert_eq!(models[0].raw["owned_by"], json!("openai"));
}

// ---------------------------------------------------------------- count tokens

#[test]
fn count_tokens_adapter_filters_body() {
    let mut req = Request::with_messages(vec![Message::user_text("count me")]);
    req.system = Some(vec![ContentBlock::text("sys")]);
    req.max_output_tokens = Some(100);
    req.temperature = Some(0.5);
    req.metadata = Some(serde_json::Map::from_iter([("k".to_owned(), json!("v"))]));
    req.cache_key = Some("cache".into());
    req.tools = Some(vec![Tool::function(FunctionTool::new("f"))]);
    req.tool_choice = Some(ToolChoice::Auto);
    req.parallel_tool_calls = Some(true);
    req.reasoning = Some(Reasoning::effort(Effort::Low));
    req.output_format = Some(OutputFormat::json_object());
    // Injected via extra: kept when accepted, dropped-with-warning when not.
    req.extra.set(F, "store", false);
    req.extra.set(F, "personality", "friendly");

    let built = OpenAiResponses
        .build_count_tokens_request(&req, &ctx(CallMode::Unary))
        .unwrap();
    assert_eq!(built.method, http::Method::POST);
    assert_eq!(
        built.url.to_string(),
        "https://api.openai.com/v1/responses/input_tokens"
    );
    let body = body_of(&built);
    let keys: Vec<&str> = body
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    for key in [
        "model",
        "input",
        "instructions",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "text",
        "reasoning",
        "personality",
    ] {
        assert!(keys.contains(&key), "expected `{key}` in {keys:?}");
    }
    for key in [
        "max_output_tokens",
        "temperature",
        "metadata",
        "prompt_cache_key",
        "store",
        "stream",
    ] {
        assert!(!keys.contains(&key), "`{key}` must be stripped");
    }
    // Only the non-generated drop warns.
    let drops: Vec<_> = built
        .warnings
        .iter()
        .filter(|w| w.code == WarningCode::CountTokensFieldDropped)
        .collect();
    assert_eq!(drops.len(), 1, "{:?}", codes(&built.warnings));
    assert_eq!(drops[0].location, "/store");
    assert_eq!(drops[0].severity, WarningSeverity::Semantic);

    // Under strict the inexact count is an error.
    let mut strict_ctx = ctx(CallMode::Unary);
    strict_ctx.convert.strict = true;
    let err = OpenAiResponses
        .build_count_tokens_request(&req, &strict_ctx)
        .unwrap_err();
    assert!(matches!(
        err,
        Error::Conversion(ConversionError::Strict { .. })
    ));
}

#[test]
fn count_tokens_response_parses() {
    let count = OpenAiResponses
        .parse_count_tokens_response(br#"{"object": "response.input_tokens", "input_tokens": 11}"#)
        .unwrap();
    assert_eq!(count.input_tokens, 11);
    assert_eq!(count.raw.unwrap()["object"], json!("response.input_tokens"));
    assert!(
        OpenAiResponses
            .parse_count_tokens_response(b"not json")
            .is_err()
    );
    // A missing count is a loud parse error (the wire field is required),
    // and a negative one fails the u64 parse — never a silent zero.
    assert!(matches!(
        OpenAiResponses.parse_count_tokens_response(br#"{"object": "response.input_tokens"}"#),
        Err(Error::Parse { .. })
    ));
    assert!(matches!(
        OpenAiResponses.parse_count_tokens_response(br#"{"input_tokens": -3}"#),
        Err(Error::Parse { .. })
    ));
}

// ---------------------------------------------------------------- errors

#[test]
fn parse_error_maps_openai_error_types() {
    let cases = [
        (
            401,
            r#"{"error": {"message": "bad key", "type": "authentication_error"}}"#,
            llm_api::ApiErrorKind::Auth,
        ),
        (
            429,
            r#"{"error": {"message": "quota", "type": "insufficient_quota"}}"#,
            llm_api::ApiErrorKind::RateLimit,
        ),
        (
            400,
            r#"{"error": {"message": "bad", "type": "invalid_request_error", "param": "input"}}"#,
            llm_api::ApiErrorKind::InvalidRequest,
        ),
        (
            500,
            r#"{"error": {"message": "boom", "type": "server_error"}}"#,
            llm_api::ApiErrorKind::ServerError,
        ),
        // Unknown type falls back to status classification.
        (
            429,
            r#"{"error": {"message": "slow", "type": "weird"}}"#,
            llm_api::ApiErrorKind::RateLimit,
        ),
        // A specific code refines a generic type (quota errors commonly
        // ride `code` while `type` stays a broad bucket).
        (
            429,
            r#"{"error": {"message": "quota", "type": "invalid_request_error", "code": "insufficient_quota"}}"#,
            llm_api::ApiErrorKind::RateLimit,
        ),
        (
            401,
            r#"{"error": {"message": "key", "type": "invalid_request_error", "code": "invalid_api_key"}}"#,
            llm_api::ApiErrorKind::Auth,
        ),
        // When the type misses the table, the code gets a round against
        // the same table before the status fallback.
        (
            400,
            r#"{"error": {"message": "boom", "type": "weird", "code": "server_error"}}"#,
            llm_api::ApiErrorKind::ServerError,
        ),
        (
            503,
            r#"{"error": {"message": "busy", "code": "overloaded_error"}}"#,
            llm_api::ApiErrorKind::Overloaded,
        ),
    ];
    for (status, body, expected) in cases {
        let err = OpenAiResponses.parse_error(status, &http::HeaderMap::new(), body.as_bytes());
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
    let err = OpenAiResponses.parse_error(429, &headers, b"<html>overloaded</html>");
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
    assert_eq!(OpenAiResponses.id(), "openai_responses");
    assert_eq!(OpenAiResponses.id(), llm_api::ids::OPENAI_RESPONSES);
}

#[test]
fn reasoning_text_content_maps_to_thinking_text() {
    // Raw reasoning (`content` reasoning_text parts) is the chain of
    // thought and takes priority over the summary for `Thinking.text`;
    // both arrays stay in the namespace for reconstruction.
    let body = json!({
        "id": "resp_rt", "object": "response", "status": "completed",
        "model": "deepseek-v4-flash",
        "output": [
            {"id": "rs_1", "type": "reasoning",
             "summary": [{"type": "summary_text", "text": "a summary"}],
             "content": [
                {"type": "reasoning_text", "text": "part one"},
                {"type": "reasoning_text", "text": "part two"}
             ]},
            {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": "answer", "annotations": []}]}
        ],
        "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
    });
    let resp = OpenAiResponses
        .parse_response(&serde_json::to_vec(&body).unwrap(), &meta_ok())
        .unwrap();
    let ContentBlock::Thinking { text, extra, .. } = &resp.message.content[0] else {
        panic!("expected a thinking block: {:?}", resp.message.content);
    };
    assert_eq!(text.as_deref(), Some("part one\n\npart two"));
    let ns = extra.get(F).expect("namespace stored");
    assert_eq!(
        ns.get("content").and_then(Value::as_array).map(Vec::len),
        Some(2)
    );
    assert_eq!(
        ns.get("summary").and_then(Value::as_array).map(Vec::len),
        Some(1)
    );
}

#[test]
fn reasoning_text_request_round_trip_is_idempotent() {
    // An input reasoning item carrying raw `content` restores verbatim:
    // the build base must not synthesize a duplicate channel.
    let wire = json!({
        "model": "gpt-5.1",
        "input": [
            {"type": "reasoning", "summary": [], "id": "rs_1",
             "content": [{"type": "reasoning_text", "text": "cot"}]},
            {"type": "message", "role": "assistant", "id": "msg_1",
             "content": [{"type": "output_text", "text": "hi", "annotations": []}]}
        ]
    });
    let bytes = serde_json::to_vec(&wire).unwrap();
    let (req, parse_warnings) = OpenAiResponses.parse_request(&bytes).unwrap();
    assert!(parse_warnings.is_empty(), "{parse_warnings:?}");
    // The raw reasoning surfaced as the block's text.
    assert!(matches!(&req.messages[0].content[0],
        ContentBlock::Thinking { text: Some(t), .. } if t == "cot"));

    let (body, build_warnings) = request_from_ir(
        &req,
        Some("gpt-5.1"),
        CallMode::Unary,
        &ConvertOptions::default(),
    )
    .unwrap();
    assert!(build_warnings.is_empty(), "{build_warnings:?}");
    assert_eq!(body, wire);

    let (req2, _) = request_to_ir(&serde_json::to_vec(&body).unwrap()).unwrap();
    let (body2, _) = request_from_ir(
        &req2,
        Some("gpt-5.1"),
        CallMode::Unary,
        &ConvertOptions::default(),
    )
    .unwrap();
    assert_eq!(body2, wire);
}
