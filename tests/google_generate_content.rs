//! Build-side (IR → request) tests for the `google_generate_content` format:
//! per-field mappings of design § 4.5–§ 4.9, the § 7 message rules, warnings,
//! strict mode, hooks and endpoint URLs.

use llm_api::formats::google_generate_content::{GoogleGenerateContent, request_from_ir};
use llm_api::{
    ApiFormat, BuildCtx, CacheHint, CallMode, ContentBlock, ConversionError, ConversionWarning,
    ConvertOptions, Effort, EndpointUrl, Error, FunctionTool, GoogleGenerateContentOptions,
    GoogleSafetySettings, ImageSource, Message, OrphanToolCalls, OutputFormat, Reasoning, Request,
    RequestHooks, Role, Tool, ToolChoice, ToolOutputBlock, WarningCode, WarningSeverity,
};
use serde_json::{Value, json};

const FMT: &str = "google_generate_content";

fn ctx(mode: CallMode) -> BuildCtx {
    BuildCtx::new(
        EndpointUrl::base("https://generativelanguage.googleapis.com/v1beta").unwrap(),
        "gemini-2.5-pro",
        mode,
    )
}

fn build(req: &Request) -> (Value, Vec<ConversionWarning>) {
    request_from_ir(
        req,
        &ConvertOptions::default(),
        &GoogleGenerateContentOptions::default(),
    )
    .unwrap()
}

fn codes(warnings: &[ConversionWarning]) -> Vec<WarningCode> {
    warnings.iter().map(|w| w.code.clone()).collect()
}

fn find<'a>(warnings: &'a [ConversionWarning], code: &WarningCode) -> &'a ConversionWarning {
    warnings
        .iter()
        .find(|w| w.code == *code)
        .unwrap_or_else(|| panic!("expected warning {code:?}, got {warnings:?}"))
}

fn user_req(text: &str) -> Request {
    Request::with_messages(vec![Message::user_text(text)])
}

// ---------------------------------------------------------------- URLs, auth

#[test]
fn chat_url_differs_by_call_mode() {
    let format = GoogleGenerateContent;
    let req = user_req("hi");

    let unary = format.build_request(&req, &ctx(CallMode::Unary)).unwrap();
    assert_eq!(
        unary.url.to_string(),
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"
    );
    assert_eq!(unary.method, http::Method::POST);
    assert_eq!(
        unary.headers.get("content-type").unwrap(),
        "application/json"
    );
    let auth = unary.auth.unwrap();
    assert_eq!(auth.header.as_str(), "x-goog-api-key");
    assert_eq!(auth.prefix, None);

    let streaming = format
        .build_request(&req, &ctx(CallMode::Streaming))
        .unwrap();
    assert_eq!(
        streaming.url.to_string(),
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:streamGenerateContent?alt=sse"
    );
    // Streaming is expressed in the URL, not the body.
    let body: Value = serde_json::from_slice(&streaming.body).unwrap();
    assert!(body.get("stream").is_none());
}

#[test]
fn model_prefix_stripped_and_tuned_models_rejected() {
    let format = GoogleGenerateContent;
    let req = user_req("hi");

    let mut c = ctx(CallMode::Unary);
    c.model = "models/gemini-2.5-flash".to_owned();
    let built = format.build_request(&req, &c).unwrap();
    assert!(
        built
            .url
            .to_string()
            .ends_with("/models/gemini-2.5-flash:generateContent")
    );

    c.model = "tunedModels/my-tune".to_owned();
    assert!(matches!(
        format.build_request(&req, &c),
        Err(Error::NotSupported(_))
    ));

    // The `models/` prefix is stripped before the tuned-model check: the
    // prefixed spelling names the same resource and is rejected too.
    c.model = "models/tunedModels/my-tune".to_owned();
    assert!(matches!(
        format.build_request(&req, &c),
        Err(Error::NotSupported(_))
    ));

    c.model = String::new();
    assert!(matches!(
        format.build_request(&req, &c),
        Err(Error::Conversion(_))
    ));
}

#[test]
fn protected_alt_query_conflicts() {
    let format = GoogleGenerateContent;
    let mut c = ctx(CallMode::Streaming);
    c.extra_query.push(("alt".to_owned(), "json".to_owned()));
    let err = format.build_request(&user_req("hi"), &c).unwrap_err();
    assert!(matches!(
        err,
        Error::Conversion(ConversionError::ProtectedQueryKey { .. })
    ));

    // Unary mode does not protect `alt`.
    let mut c = ctx(CallMode::Unary);
    c.extra_query.push(("alt".to_owned(), "json".to_owned()));
    let built = format.build_request(&user_req("hi"), &c).unwrap();
    assert!(built.url.to_string().ends_with(":generateContent?alt=json"));
}

// ------------------------------------------------------- sampling parameters

#[test]
fn sampling_parameters_all_map_natively() {
    let mut req = user_req("hi");
    req.max_output_tokens = Some(1024);
    req.temperature = Some(0.5);
    req.top_p = Some(0.9);
    req.top_k = Some(40);
    req.stop_sequences = Some(vec!["END".to_owned()]);
    req.seed = Some(7);
    req.frequency_penalty = Some(0.25);
    req.presence_penalty = Some(0.5);
    let (body, warnings) = build(&req);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body["generationConfig"],
        json!({
            "maxOutputTokens": 1024,
            "temperature": 0.5,
            "topP": 0.9,
            "topK": 40,
            "stopSequences": ["END"],
            "seed": 7,
            "frequencyPenalty": 0.25,
            "presencePenalty": 0.5,
        })
    );
}

#[test]
fn non_finite_sampling_values_are_conversion_errors() {
    // Google nests sampling under generationConfig; the error points at the
    // final-body location.
    type SetField = fn(&mut Request, f64);
    let fields: [(SetField, &str); 4] = [
        (
            |r, v| r.temperature = Some(v),
            "/generationConfig/temperature",
        ),
        (|r, v| r.top_p = Some(v), "/generationConfig/topP"),
        (
            |r, v| r.frequency_penalty = Some(v),
            "/generationConfig/frequencyPenalty",
        ),
        (
            |r, v| r.presence_penalty = Some(v),
            "/generationConfig/presencePenalty",
        ),
    ];
    for (set, expected) in fields {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut req = user_req("hi");
            set(&mut req, bad);
            let err = request_from_ir(
                &req,
                &ConvertOptions::default(),
                &GoogleGenerateContentOptions::default(),
            )
            .unwrap_err();
            match err {
                Error::Conversion(ConversionError::NonFiniteNumber { location, .. }) => {
                    assert_eq!(location, expected);
                }
                other => panic!("expected NonFiniteNumber for {bad}, got {other:?}"),
            }
        }
    }

    // The count-tokens build shares the chat body pipeline and fails the
    // same way.
    let mut req = user_req("hi");
    req.temperature = Some(f64::NAN);
    let err = GoogleGenerateContent
        .build_count_tokens_request(&req, &ctx(CallMode::Unary))
        .unwrap_err();
    assert!(matches!(
        err,
        Error::Conversion(ConversionError::NonFiniteNumber { .. })
    ));

    // Finite values — zeroes and extremes — keep the existing behavior.
    let mut req = user_req("hi");
    req.temperature = Some(0.0);
    req.top_p = Some(-0.0);
    req.frequency_penalty = Some(f64::MAX);
    req.presence_penalty = Some(f64::MIN_POSITIVE);
    let (body, warnings) = build(&req);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(body["generationConfig"]["temperature"], json!(0.0));
    assert_eq!(body["generationConfig"]["topP"], json!(-0.0));
    assert_eq!(
        body["generationConfig"]["frequencyPenalty"],
        json!(f64::MAX)
    );
    assert_eq!(
        body["generationConfig"]["presencePenalty"],
        json!(f64::MIN_POSITIVE)
    );
}

#[test]
fn metadata_and_cache_key_drop_with_cosmetic_warnings() {
    let mut req = user_req("hi");
    let mut metadata = serde_json::Map::new();
    metadata.insert("session".to_owned(), json!("abc"));
    req.metadata = Some(metadata);
    req.cache_key = Some("cache-1".to_owned());
    let (body, warnings) = build(&req);
    assert!(body.get("metadata").is_none());
    let m = find(&warnings, &WarningCode::MetadataDropped);
    assert_eq!(m.severity, WarningSeverity::Cosmetic);
    assert_eq!(m.location, "/metadata");
    assert!(m.message.contains("session"));
    let c = find(&warnings, &WarningCode::CacheKeyDropped);
    assert_eq!(c.severity, WarningSeverity::Cosmetic);
}

// ------------------------------------------------------------- system rules

#[test]
fn system_field_and_leading_system_messages_hoist_in_order() {
    let mut req = Request::with_messages(vec![
        Message::system_text("first message rule"),
        Message::system_text("second message rule"),
        Message::user_text("hi"),
    ]);
    req.system = Some(vec![ContentBlock::text("request-level rule")]);
    let (body, warnings) = build(&req);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body["systemInstruction"],
        json!({"parts": [
            {"text": "request-level rule"},
            {"text": "first message rule"},
            {"text": "second message rule"},
        ]})
    );
    assert_eq!(
        body["contents"],
        json!([{"role": "user", "parts": [{"text": "hi"}]}])
    );
}

#[test]
fn system_channel_serializing_to_nothing_is_omitted_with_warning() {
    // A hoisted leading system message whose every block is a dropped
    // foreign opaque: `systemInstruction` is omitted and the channel-level
    // omission is disclosed on top of the block-level drop.
    let req = Request::with_messages(vec![
        Message::new(
            Role::System,
            vec![ContentBlock::opaque(
                "openai_responses",
                json!({"type": "note"}),
            )],
        ),
        Message::user_text("hi"),
    ]);
    let (body, warnings) = build(&req);
    assert!(body.get("systemInstruction").is_none(), "{body}");
    let dropped = find(&warnings, &WarningCode::EmptyMessageDropped);
    assert_eq!(dropped.location, "/systemInstruction");
    assert_eq!(dropped.severity, WarningSeverity::Semantic);
    assert!(dropped.message.contains("system channel"), "{warnings:?}");
    find(&warnings, &WarningCode::OpaqueDropped);

    // A hoisted message's google-namespace extra merges into
    // `systemInstruction` and keeps the channel on the wire: no omission,
    // no channel-level warning (current-behavior pin).
    let mut sys_msg = Message::new(
        Role::System,
        vec![ContentBlock::opaque(
            "openai_responses",
            json!({"type": "note"}),
        )],
    );
    sys_msg.extra.set(FMT, "contentTag", "tagged");
    let req = Request::with_messages(vec![sys_msg, Message::user_text("hi")]);
    let (body, warnings) = build(&req);
    assert_eq!(
        body["systemInstruction"],
        json!({"parts": [], "contentTag": "tagged"})
    );
    assert!(
        !codes(&warnings).contains(&WarningCode::EmptyMessageDropped),
        "{warnings:?}"
    );
}

#[test]
fn genuinely_empty_system_input_stays_silent() {
    // `Request.system: Some(vec![])` is the caller's own data: the key is
    // omitted without any warning (current-behavior pin).
    let mut req = user_req("hi");
    req.system = Some(vec![]);
    let (body, warnings) = build(&req);
    assert!(body.get("systemInstruction").is_none(), "{body}");
    assert!(warnings.is_empty(), "{warnings:?}");

    // A zero-block leading system message contributes nothing, silently.
    let req = Request::with_messages(vec![
        Message::new(Role::System, vec![]),
        Message::user_text("hi"),
    ]);
    let (body, warnings) = build(&req);
    assert!(body.get("systemInstruction").is_none(), "{body}");
    assert!(warnings.is_empty(), "{warnings:?}");
}

#[test]
fn mid_system_and_developer_downgrade_to_user() {
    let req = Request::with_messages(vec![
        Message::user_text("hi"),
        Message::system_text("mid-conversation rule"),
        Message::assistant_text("ok"),
        Message::developer_text("dev note"),
    ]);
    let (body, warnings) = build(&req);
    assert!(body.get("systemInstruction").is_none());
    let downgrades: Vec<_> = warnings
        .iter()
        .filter(|w| w.code == WarningCode::RoleDowngraded)
        .collect();
    assert_eq!(downgrades.len(), 2);
    assert!(
        downgrades
            .iter()
            .all(|w| w.severity == WarningSeverity::Semantic)
    );
    // hi + mid-system merge into one user turn; dev note becomes its own
    // user turn after the model turn.
    assert_eq!(
        body["contents"],
        json!([
            {"role": "user", "parts": [{"text": "hi"}, {"text": "mid-conversation rule"}]},
            {"role": "model", "parts": [{"text": "ok"}]},
            {"role": "user", "parts": [{"text": "dev note"}]},
        ])
    );
}

#[test]
fn invalid_system_blocks_error() {
    // Request.system allows Text only — even Opaque is rejected there.
    let mut req = user_req("hi");
    req.system = Some(vec![ContentBlock::opaque(FMT, json!({"text": "x"}))]);
    let err = request_from_ir(
        &req,
        &ConvertOptions::default(),
        &GoogleGenerateContentOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(
        err,
        Error::Conversion(ConversionError::InvalidBlockForRole {
            role: Role::System,
            ..
        })
    ));

    // An image in an in-array system message is invalid per § 7.4.
    let req = Request::with_messages(vec![
        Message::new(
            Role::System,
            vec![ContentBlock::image_url("https://x/y.png")],
        ),
        Message::user_text("hi"),
    ]);
    let err = request_from_ir(
        &req,
        &ConvertOptions::default(),
        &GoogleGenerateContentOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(
        err,
        Error::Conversion(ConversionError::InvalidBlockForRole {
            role: Role::System,
            ..
        })
    ));

    // A Google-owned opaque block in an in-array system message is fine.
    let req = Request::with_messages(vec![
        Message::new(
            Role::System,
            vec![
                ContentBlock::text("rule"),
                ContentBlock::opaque(FMT, json!({"text": "opaque rule"})),
            ],
        ),
        Message::user_text("hi"),
    ]);
    let (body, warnings) = build(&req);
    assert!(warnings.is_empty());
    assert_eq!(
        body["systemInstruction"]["parts"],
        json!([{"text": "rule"}, {"text": "opaque rule"}])
    );
}

// --------------------------------------------------------- role × merge rules

#[test]
fn adjacent_user_and_tool_messages_merge_into_one_turn() {
    let req = Request::with_messages(vec![
        Message::user_text("look at this"),
        Message::assistant(vec![ContentBlock::tool_call_with_id("c1", "lookup", "{}")]),
        Message::tool(vec![ContentBlock::tool_result_text(
            Some("c1".to_owned()),
            "found it",
        )]),
        Message::user_text("thanks"),
    ]);
    let (body, warnings) = build(&req);
    assert!(warnings.is_empty(), "{warnings:?}");
    let contents = body["contents"].as_array().unwrap();
    assert_eq!(contents.len(), 3);
    assert_eq!(contents[2]["role"], "user");
    let parts = contents[2]["parts"].as_array().unwrap();
    assert_eq!(parts.len(), 2);
    assert!(parts[0].get("functionResponse").is_some());
    assert_eq!(parts[1], json!({"text": "thanks"}));
}

#[test]
fn adjacent_assistant_messages_merge_into_one_model_turn() {
    let req = Request::with_messages(vec![
        Message::user_text("hi"),
        Message::assistant_text("part one."),
        Message::assistant_text("part two."),
    ]);
    let (body, _) = build(&req);
    assert_eq!(
        body["contents"],
        json!([
            {"role": "user", "parts": [{"text": "hi"}]},
            {"role": "model", "parts": [{"text": "part one."}, {"text": "part two."}]},
        ])
    );
}

#[test]
fn empty_serialization_message_is_omitted_and_neighbours_merge() {
    // The assistant message consists of one foreign thinking block, which
    // drops: no empty-`parts` model turn may reach the wire, and the two
    // user messages merge across the gap into a single turn.
    let foreign_thinking =
        ContentBlock::thinking_signed("plan", "sig").with_extra("anthropic_messages", "s", 1);
    let req = Request::with_messages(vec![
        Message::user_text("first"),
        Message::assistant(vec![foreign_thinking]),
        Message::user_text("second"),
    ]);
    let (body, warnings) = build(&req);
    assert_eq!(
        body["contents"],
        json!([{"role": "user", "parts": [{"text": "first"}, {"text": "second"}]}])
    );
    let dropped: Vec<_> = warnings
        .iter()
        .filter(|w| w.code == WarningCode::EmptyMessageDropped)
        .collect();
    assert_eq!(dropped.len(), 1, "{warnings:?}");
    assert_eq!(dropped[0].severity, WarningSeverity::Semantic);
    assert_eq!(dropped[0].location, "/contents");
    assert!(dropped[0].message.contains("IR message 1"), "{warnings:?}");
    find(&warnings, &WarningCode::ThinkingDropped);

    // A native (google-marked) thinking block keeps its turn.
    let native = ContentBlock::thinking("plan").with_extra(FMT, "thought", true);
    let req = Request::with_messages(vec![
        Message::user_text("first"),
        Message::assistant(vec![native]),
        Message::user_text("second"),
    ]);
    let (body, warnings) = build(&req);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(body["contents"].as_array().unwrap().len(), 3);
}

#[test]
fn omitted_message_skips_role_downgrade_but_reports_lost_extra() {
    // A mid-conversation system message whose only block is foreign opaque:
    // the message is omitted, so no RoleDowngraded fires — only the drop
    // disclosures. Its google-namespace extra has nowhere to land.
    let mut msg = Message::new(
        Role::System,
        vec![ContentBlock::opaque(
            "openai_responses",
            json!({"type": "note"}),
        )],
    );
    msg.extra.set(FMT, "contentTag", "tagged");
    let req = Request::with_messages(vec![
        Message::user_text("hi"),
        Message::assistant_text("hello"),
        msg,
        Message::user_text("bye"),
    ]);
    let (body, warnings) = build(&req);
    assert_eq!(body["contents"].as_array().unwrap().len(), 3);
    assert_eq!(body["contents"][2]["parts"], json!([{"text": "bye"}]));
    assert!(!codes(&warnings).contains(&WarningCode::RoleDowngraded));
    find(&warnings, &WarningCode::OpaqueDropped);
    find(&warnings, &WarningCode::EmptyMessageDropped);
    let extra = find(&warnings, &WarningCode::ExtraDropped);
    assert!(extra.message.contains("IR message 2"), "{warnings:?}");

    // Without a google-namespace extra there is nothing to report beyond
    // the message drop itself.
    let req = Request::with_messages(vec![
        Message::user_text("hi"),
        Message::assistant(vec![ContentBlock::opaque(
            "openai_responses",
            json!({"type": "note"}),
        )]),
    ]);
    let (_, warnings) = build(&req);
    find(&warnings, &WarningCode::EmptyMessageDropped);
    assert!(!codes(&warnings).contains(&WarningCode::ExtraDropped));
}

#[test]
fn deliberately_empty_ir_messages_keep_their_empty_turn() {
    // Zero-block IR messages are the IR image of the wire's own
    // empty-content form and replay as an empty turn, without warnings.
    let req = Request::with_messages(vec![Message::user_text("hi"), Message::assistant(vec![])]);
    let (body, warnings) = build(&req);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body["contents"],
        json!([
            {"role": "user", "parts": [{"text": "hi"}]},
            {"role": "model", "parts": []},
        ])
    );
}

#[test]
fn role_block_validity_is_enforced() {
    for (role, block) in [
        (Role::User, ContentBlock::tool_call("f", "{}")),
        (Role::User, ContentBlock::thinking("x")),
        (Role::User, ContentBlock::tool_result(None, vec![])),
        (Role::Assistant, ContentBlock::tool_result(None, vec![])),
        (Role::Tool, ContentBlock::text("x")),
        (Role::Tool, ContentBlock::image_url("https://x/y.png")),
    ] {
        let req = Request::with_messages(vec![Message::new(role, vec![block])]);
        let err = request_from_ir(
            &req,
            &ConvertOptions::default(),
            &GoogleGenerateContentOptions::default(),
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                Error::Conversion(ConversionError::InvalidBlockForRole { .. })
            ),
            "{role:?} should reject the block"
        );
    }
}

// ------------------------------------------------------------------- images

#[test]
fn image_sources_map_per_table() {
    let req = Request::with_messages(vec![Message::user(vec![
        ContentBlock::image_url("https://example.com/cat.png"),
        ContentBlock::image_base64("image/png", "aW1n"),
        ContentBlock::image_file_id("https://generativelanguage.googleapis.com/v1beta/files/f1"),
    ])]);
    let (body, warnings) = build(&req);
    let parts = body["contents"][0]["parts"].as_array().unwrap();
    assert_eq!(
        parts[0],
        json!({"fileData": {"fileUri": "https://example.com/cat.png"}})
    );
    assert_eq!(
        parts[1],
        json!({"inlineData": {"mimeType": "image/png", "data": "aW1n"}})
    );
    assert_eq!(
        parts[2],
        json!({"fileData": {"fileUri": "https://generativelanguage.googleapis.com/v1beta/files/f1"}})
    );
    // Only the arbitrary URL warns (cosmetic).
    assert_eq!(codes(&warnings), vec![WarningCode::ImageUrlAsFileUri]);
    assert_eq!(warnings[0].severity, WarningSeverity::Cosmetic);
}

#[test]
fn assistant_images_use_the_native_inline_channel() {
    let req = Request::with_messages(vec![
        Message::user_text("draw"),
        Message::assistant(vec![ContentBlock::image_base64("image/png", "cGl4")]),
    ]);
    let (body, warnings) = build(&req);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body["contents"][1]["parts"][0],
        json!({"inlineData": {"mimeType": "image/png", "data": "cGl4"}})
    );
}

#[test]
fn cache_hints_drop_with_cosmetic_warnings() {
    let req = Request::with_messages(vec![Message::user(vec![
        ContentBlock::text("cached").with_cache(CacheHint::with_ttl("1h")),
    ])]);
    let (body, warnings) = build(&req);
    assert_eq!(body["contents"][0]["parts"][0], json!({"text": "cached"}));
    let w = find(&warnings, &WarningCode::CacheHintDropped);
    assert_eq!(w.severity, WarningSeverity::Cosmetic);
    assert_eq!(w.location, "/contents/0/parts/0");
}

// -------------------------------------------------------------------- tools

#[test]
fn function_tools_map_to_declarations() {
    let mut req = user_req("hi");
    let schema = json!({"type": "object", "properties": {"city": {"type": "string"}}});
    req.tools = Some(vec![
        Tool::function(
            FunctionTool::new("get_weather")
                .with_description("Get the weather.")
                .with_parameters(schema.clone()),
        ),
        Tool::function(FunctionTool::new("no_params")),
        Tool::opaque(FMT, json!({"googleSearch": {}})),
    ]);
    let (body, warnings) = build(&req);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body["tools"],
        json!([
            {"functionDeclarations": [
                {"name": "get_weather", "description": "Get the weather.", "parametersJsonSchema": schema},
                // parameters: None means "no parameters": the field is omitted.
                {"name": "no_params"},
            ]},
            {"googleSearch": {}},
        ])
    );
}

#[test]
fn tool_strict_and_cache_warn_and_foreign_opaque_drops() {
    let mut req = user_req("hi");
    req.tools = Some(vec![
        Tool::function(
            FunctionTool::new("f")
                .with_strict(true)
                .with_cache(CacheHint::new()),
        ),
        Tool::opaque("openai_responses", json!({"type": "web_search"})),
    ]);
    let (body, warnings) = build(&req);
    assert_eq!(body["tools"].as_array().unwrap().len(), 1);
    let strict = find(&warnings, &WarningCode::StrictUnsupported);
    assert_eq!(strict.severity, WarningSeverity::Semantic);
    assert_eq!(strict.location, "/tools/0/functionDeclarations/0");
    find(&warnings, &WarningCode::CacheHintDropped);
    let opaque = find(&warnings, &WarningCode::OpaqueDropped);
    assert_eq!(opaque.severity, WarningSeverity::Semantic);
}

#[test]
fn tools_key_omitted_when_all_entries_drop_but_empty_list_replays() {
    // Every IR tool is foreign opaque: the key is omitted entirely (the
    // OpaqueDropped warnings disclose the drops) — `"tools": []` would
    // misstate the caller's intent.
    let mut req = user_req("hi");
    req.tools = Some(vec![
        Tool::opaque("openai_responses", json!({"type": "web_search"})),
        Tool::opaque("anthropic_messages", json!({"type": "bash"})),
    ]);
    let (body, warnings) = build(&req);
    assert!(body.get("tools").is_none(), "{body}");
    assert_eq!(
        warnings
            .iter()
            .filter(|w| w.code == WarningCode::OpaqueDropped)
            .count(),
        2
    );

    // An explicitly empty IR tool list is replayed faithfully.
    let mut req = user_req("hi");
    req.tools = Some(vec![]);
    let (body, warnings) = build(&req);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(body["tools"], json!([]));
}

#[test]
fn tool_choice_maps_per_table() {
    for (choice, expected) in [
        (
            ToolChoice::Auto,
            json!({"functionCallingConfig": {"mode": "AUTO"}}),
        ),
        (
            ToolChoice::None,
            json!({"functionCallingConfig": {"mode": "NONE"}}),
        ),
        (
            ToolChoice::Required,
            json!({"functionCallingConfig": {"mode": "ANY"}}),
        ),
        (
            ToolChoice::tool("get_weather"),
            json!({"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["get_weather"]}}),
        ),
    ] {
        let mut req = user_req("hi");
        req.tools = Some(vec![Tool::function(FunctionTool::new("get_weather"))]);
        req.tool_choice = Some(choice);
        let (body, warnings) = build(&req);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(body["toolConfig"], expected);
    }
}

#[test]
fn parallel_tool_calls_warn_matrix() {
    // Some(false) with tools: the serial constraint is a semantic loss.
    let mut req = user_req("hi");
    req.tools = Some(vec![Tool::function(FunctionTool::new("f"))]);
    req.parallel_tool_calls = Some(false);
    let (_, warnings) = build(&req);
    assert_eq!(codes(&warnings), vec![WarningCode::SerialToolCallsDropped]);
    assert_eq!(warnings[0].severity, WarningSeverity::Semantic);

    // Some(true) with tools: cosmetic (parallel is Google's default).
    req.parallel_tool_calls = Some(true);
    let (_, warnings) = build(&req);
    assert_eq!(
        codes(&warnings),
        vec![WarningCode::ParallelToolCallsIgnored]
    );
    assert_eq!(warnings[0].severity, WarningSeverity::Cosmetic);

    // Without tools the flag is meaningless for either value.
    req.tools = None;
    req.parallel_tool_calls = Some(false);
    let (_, warnings) = build(&req);
    assert_eq!(
        codes(&warnings),
        vec![WarningCode::ParallelToolCallsIgnored]
    );

    // The gate follows the emitted tools: with every entry dropped,
    // Some(false) is meaningless too (no serial constraint was lost — no
    // wire tool exists to run serially).
    req.tools = Some(vec![Tool::opaque(
        "openai_responses",
        json!({"type": "web_search"}),
    )]);
    let (_, warnings) = build(&req);
    assert_eq!(
        codes(&warnings),
        vec![
            WarningCode::OpaqueDropped,
            WarningCode::ParallelToolCallsIgnored,
        ]
    );
}

#[test]
fn tool_config_requires_wire_tools() {
    // `toolConfig` follows the emitted tools, not the IR list: without any
    // wire tool the key is withheld with a cosmetic warning.
    let expect_ignored = |req: &Request| {
        let (body, warnings) = build(req);
        assert!(body.get("toolConfig").is_none(), "{body}");
        let w = find(&warnings, &WarningCode::ToolChoiceIgnored);
        assert_eq!(w.severity, WarningSeverity::Cosmetic);
        assert_eq!(w.location, "/toolConfig");
        body
    };

    // No IR tools at all.
    let mut req = user_req("hi");
    req.tool_choice = Some(ToolChoice::Required);
    let body = expect_ignored(&req);
    assert!(body.get("tools").is_none());

    // Every tool a dropped foreign opaque: no `tools` key either.
    req.tools = Some(vec![Tool::opaque(
        "openai_responses",
        json!({"type": "web_search"}),
    )]);
    let body = expect_ignored(&req);
    assert!(body.get("tools").is_none(), "{body}");

    // An explicitly empty IR list replays `"tools": []` but still counts
    // as no wire tools.
    req.tools = Some(vec![]);
    let body = expect_ignored(&req);
    assert_eq!(body["tools"], json!([]));
}

#[test]
fn tool_call_arguments_must_parse_to_an_object() {
    for arguments in ["not json", "[1, 2]", "\"str\"", ""] {
        let req = Request::with_messages(vec![Message::assistant(vec![
            ContentBlock::tool_call_with_id("c1", "f", arguments),
        ])]);
        let err = request_from_ir(
            &req,
            &ConvertOptions::default(),
            &GoogleGenerateContentOptions::default(),
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                Error::Conversion(ConversionError::InvalidToolArguments { .. })
            ),
            "arguments {arguments:?} must fail"
        );
    }
}

// -------------------------------------------------------------- tool results

#[test]
fn tool_result_name_resolves_from_earlier_call() {
    let req = Request::with_messages(vec![
        Message::user_text("weather?"),
        Message::assistant(vec![ContentBlock::tool_call_with_id(
            "c1",
            "get_weather",
            r#"{"city":"Paris"}"#,
        )]),
        Message::tool(vec![ContentBlock::tool_result_text(
            Some("c1".to_owned()),
            "22C",
        )]),
    ]);
    let (body, warnings) = build(&req);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body["contents"][2]["parts"][0],
        json!({"functionResponse": {
            "id": "c1",
            "name": "get_weather",
            "response": {"output": "22C"},
        }})
    );
}

#[test]
fn tool_result_without_resolvable_name_errors() {
    let req = Request::with_messages(vec![Message::tool(vec![ContentBlock::tool_result_text(
        Some("unknown".to_owned()),
        "x",
    )])]);
    let err = request_from_ir(
        &req,
        &ConvertOptions::default(),
        &GoogleGenerateContentOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(
        err,
        Error::Conversion(ConversionError::MissingRequired { .. })
    ));
}

#[test]
fn tool_result_is_error_maps_to_the_documented_error_key() {
    let req = Request::with_messages(vec![Message::tool(vec![
        ContentBlock::tool_result_text(None, "boom")
            .with_tool_name("f")
            .with_is_error(true),
    ])]);
    let (body, warnings) = build(&req);
    // The official functionResponse.response contract documents the "error"
    // key for failures, so nothing is lost and no warning is attached.
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body["contents"][0]["parts"][0]["functionResponse"]["response"],
        json!({"error": "boom"})
    );

    // is_error: false canonicalizes to the plain output encoding.
    let req = Request::with_messages(vec![Message::tool(vec![
        ContentBlock::tool_result_text(None, "fine")
            .with_tool_name("f")
            .with_is_error(false),
    ])]);
    let (body, warnings) = build(&req);
    assert!(warnings.is_empty());
    assert_eq!(
        body["contents"][0]["parts"][0]["functionResponse"]["response"],
        json!({"output": "fine"})
    );
}

#[test]
fn tool_result_text_encodings() {
    // Empty content → response: {}.
    let req = Request::with_messages(vec![Message::tool(vec![
        ContentBlock::tool_result(None, vec![]).with_tool_name("f"),
    ])]);
    let (body, warnings) = build(&req);
    assert!(warnings.is_empty());
    assert_eq!(
        body["contents"][0]["parts"][0]["functionResponse"]["response"],
        json!({})
    );

    // One empty text block → {"output": ""} (distinct from empty).
    let req = Request::with_messages(vec![Message::tool(vec![
        ContentBlock::tool_result_text(None, "").with_tool_name("f"),
    ])]);
    let (body, warnings) = build(&req);
    assert!(warnings.is_empty());
    assert_eq!(
        body["contents"][0]["parts"][0]["functionResponse"]["response"],
        json!({"output": ""})
    );

    // Multiple text blocks join with \n\n plus a cosmetic warning.
    let req = Request::with_messages(vec![Message::tool(vec![
        ContentBlock::tool_result(
            None,
            vec![
                ToolOutputBlock::text("line one"),
                ToolOutputBlock::text("line two"),
            ],
        )
        .with_tool_name("f"),
    ])]);
    let (body, warnings) = build(&req);
    assert_eq!(codes(&warnings), vec![WarningCode::ToolResultTextJoined]);
    assert_eq!(warnings[0].severity, WarningSeverity::Cosmetic);
    assert_eq!(
        body["contents"][0]["parts"][0]["functionResponse"]["response"],
        json!({"output": "line one\n\nline two"})
    );
}

#[test]
fn tool_result_images_and_order() {
    // Base64 goes to parts[]; URL and file ids have no channel.
    let req = Request::with_messages(vec![Message::tool(vec![
        ContentBlock::tool_result(
            None,
            vec![
                ToolOutputBlock::text("caption"),
                ToolOutputBlock::image(ImageSource::base64("image/png", "cGl4")),
                ToolOutputBlock::image(ImageSource::url("https://x/y.png")),
            ],
        )
        .with_tool_name("f"),
    ])]);
    let (body, warnings) = build(&req);
    let dropped = find(&warnings, &WarningCode::ToolResultImageDropped);
    assert_eq!(dropped.severity, WarningSeverity::Semantic);
    assert!(!codes(&warnings).contains(&WarningCode::ToolResultOrderLost));
    assert_eq!(
        body["contents"][0]["parts"][0]["functionResponse"],
        json!({
            "name": "f",
            "response": {"output": "caption"},
            "parts": [{"inlineData": {"mimeType": "image/png", "data": "cGl4"}}],
        })
    );

    // Text following an emitted image cannot keep its position.
    let req = Request::with_messages(vec![Message::tool(vec![
        ContentBlock::tool_result(
            None,
            vec![
                ToolOutputBlock::image(ImageSource::base64("image/png", "cGl4")),
                ToolOutputBlock::text("after the image"),
            ],
        )
        .with_tool_name("f"),
    ])]);
    let (_, warnings) = build(&req);
    let lost = find(&warnings, &WarningCode::ToolResultOrderLost);
    assert_eq!(lost.severity, WarningSeverity::Semantic);
}

// ---------------------------------------------------------------- reasoning

#[test]
fn reasoning_effort_maps_to_thinking_level() {
    for (effort, level) in [
        (Effort::Minimal, "MINIMAL"),
        (Effort::Low, "LOW"),
        (Effort::Medium, "MEDIUM"),
        (Effort::High, "HIGH"),
        (Effort::Other("ULTRA".to_owned()), "ULTRA"),
    ] {
        let mut req = user_req("hi");
        req.reasoning = Some(Reasoning::effort(effort));
        let (body, warnings) = build(&req);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(
            body["generationConfig"]["thinkingConfig"],
            json!({"thinkingLevel": level})
        );
    }
    for effort in [Effort::None, Effort::XHigh, Effort::Max] {
        let mut req = user_req("hi");
        req.reasoning = Some(Reasoning::effort(effort));
        let (body, warnings) = build(&req);
        assert!(body.get("generationConfig").is_none());
        let w = find(&warnings, &WarningCode::EffortUnsupported);
        assert_eq!(w.severity, WarningSeverity::Semantic);
        assert_eq!(w.location, "/generationConfig/thinkingConfig");
    }
}

#[test]
fn reasoning_enabled_and_include_thoughts() {
    // enabled: false has no channel — semantic warning.
    let mut req = user_req("hi");
    req.reasoning = Some(Reasoning::enabled(false));
    let (body, warnings) = build(&req);
    assert!(body.get("generationConfig").is_none());
    let w = find(&warnings, &WarningCode::ReasoningDisableUnsupported);
    assert_eq!(w.severity, WarningSeverity::Semantic);

    // enabled: true is a no-op.
    req.reasoning = Some(Reasoning::enabled(true));
    let (body, warnings) = build(&req);
    assert!(warnings.is_empty());
    assert!(body.get("generationConfig").is_none());

    // include_thoughts maps for both values.
    for value in [true, false] {
        let mut reasoning = Reasoning::new();
        reasoning.include_thoughts = Some(value);
        req.reasoning = Some(reasoning);
        let (body, warnings) = build(&req);
        assert!(warnings.is_empty());
        assert_eq!(
            body["generationConfig"]["thinkingConfig"],
            json!({"includeThoughts": value})
        );
    }
}

#[test]
fn reasoning_conflicts_let_effort_win() {
    // enabled: true + effort: none → effort wins (dropped) + conflict note.
    let mut req = user_req("hi");
    let mut reasoning = Reasoning::enabled(true);
    reasoning.effort = Some(Effort::None);
    req.reasoning = Some(reasoning);
    let (body, warnings) = build(&req);
    assert!(body.get("generationConfig").is_none());
    find(&warnings, &WarningCode::ReasoningConflict);
    find(&warnings, &WarningCode::EffortUnsupported);
    assert!(!codes(&warnings).contains(&WarningCode::ReasoningDisableUnsupported));

    // enabled: false + effort: low → effort wins, mapped.
    let mut reasoning = Reasoning::enabled(false);
    reasoning.effort = Some(Effort::Low);
    req.reasoning = Some(reasoning);
    let (body, warnings) = build(&req);
    assert_eq!(
        body["generationConfig"]["thinkingConfig"],
        json!({"thinkingLevel": "LOW"})
    );
    assert_eq!(codes(&warnings), vec![WarningCode::ReasoningConflict]);

    // enabled: false + effort: none is consistent — no conflict warning.
    let mut reasoning = Reasoning::enabled(false);
    reasoning.effort = Some(Effort::None);
    req.reasoning = Some(reasoning);
    let (_, warnings) = build(&req);
    assert_eq!(codes(&warnings), vec![WarningCode::EffortUnsupported]);
}

#[test]
fn extra_rewrites_thinking_config_completely() {
    // The § 4.7 documented pattern: switch thinkingLevel for thinkingBudget.
    let mut req = user_req("hi");
    req.reasoning = Some(Reasoning::effort(Effort::Low));
    req.extra.set(
        FMT,
        "generationConfig",
        json!({"thinkingConfig": {"thinkingLevel": null, "thinkingBudget": 512}}),
    );
    let (body, warnings) = build(&req);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body["generationConfig"]["thinkingConfig"],
        json!({"thinkingBudget": 512})
    );
}

#[test]
fn extra_override_marks_warnings() {
    // Deleting the exact warned pointer marks the warning overridden, which
    // also exempts it from the strict gate.
    let mut req = user_req("hi");
    req.reasoning = Some(Reasoning::effort(Effort::XHigh));
    req.extra
        .set(FMT, "generationConfig", json!({"thinkingConfig": null}));
    let options = ConvertOptions::new().strict(true);
    let (_, warnings) =
        request_from_ir(&req, &options, &GoogleGenerateContentOptions::default()).unwrap();
    let w = find(&warnings, &WarningCode::EffortUnsupported);
    assert!(w.overridden);
}

// -------------------------------------------------------- structured output

#[test]
fn output_format_maps_to_response_mime_and_schema() {
    let schema = json!({"type": "object", "properties": {"a": {"type": "number"}}});
    let mut req = user_req("hi");
    req.output_format = Some(OutputFormat::json_schema(schema.clone()));
    let (body, warnings) = build(&req);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body["generationConfig"]["responseMimeType"],
        "application/json"
    );
    assert_eq!(body["generationConfig"]["responseJsonSchema"], schema);

    req.output_format = Some(OutputFormat::json_object());
    let (body, warnings) = build(&req);
    assert!(warnings.is_empty());
    assert_eq!(
        body["generationConfig"],
        json!({"responseMimeType": "application/json"})
    );

    // name/description/strict have no channel — one cosmetic warning.
    req.output_format = Some(
        OutputFormat::json_schema(schema)
            .with_name("answer")
            .with_description("d")
            .with_strict(true),
    );
    let (_, warnings) = build(&req);
    let w = find(&warnings, &WarningCode::OutputFormatDetailDropped);
    assert_eq!(w.severity, WarningSeverity::Cosmetic);
    assert!(w.message.contains("name") && w.message.contains("strict"));
}

// ----------------------------------------------------------------- thinking

#[test]
fn thinking_provenance_controls_serialization() {
    // Signed, no namespace: optimistic replay — native thought part.
    let req = Request::with_messages(vec![
        Message::user_text("hi"),
        Message::assistant(vec![
            ContentBlock::thinking_signed("planning", "c2ln"),
            ContentBlock::text("done"),
        ]),
    ]);
    let (body, warnings) = build(&req);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body["contents"][1]["parts"][0],
        json!({"text": "planning", "thought": true, "thoughtSignature": "c2ln"})
    );

    // Google-namespaced (as the parser produces): native.
    let block = ContentBlock::thinking("planning").with_extra(FMT, "thought", true);
    let req = Request::with_messages(vec![
        Message::user_text("hi"),
        Message::assistant(vec![block]),
    ]);
    let (body, warnings) = build(&req);
    assert!(warnings.is_empty());
    assert_eq!(
        body["contents"][1]["parts"][0],
        json!({"text": "planning", "thought": true})
    );

    // Foreign namespace: dropped with a semantic warning. The message then
    // serializes to nothing, so no empty model turn reaches the wire.
    let block = ContentBlock::thinking_signed("planning", "sig").with_extra(
        "anthropic_messages",
        "redacted",
        false,
    );
    let req = Request::with_messages(vec![
        Message::user_text("hi"),
        Message::assistant(vec![block]),
    ]);
    let (body, warnings) = build(&req);
    assert_eq!(
        body["contents"],
        json!([{"role": "user", "parts": [{"text": "hi"}]}])
    );
    let w = find(&warnings, &WarningCode::ThinkingDropped);
    assert_eq!(w.severity, WarningSeverity::Semantic);
    find(&warnings, &WarningCode::EmptyMessageDropped);

    // thinking_as_text re-encodes the plaintext and drops the signature.
    let block = ContentBlock::thinking_signed("planning", "sig").with_extra(
        "anthropic_messages",
        "redacted",
        false,
    );
    let req = Request::with_messages(vec![
        Message::user_text("hi"),
        Message::assistant(vec![block]),
    ]);
    let mut options = ConvertOptions::default();
    options.thinking_as_text = true;
    let (body, warnings) =
        request_from_ir(&req, &options, &GoogleGenerateContentOptions::default()).unwrap();
    assert_eq!(
        body["contents"][1]["parts"][0],
        json!({"text": "planning", "thought": true})
    );
    let w = find(&warnings, &WarningCode::ThinkingSignatureDropped);
    assert_eq!(w.severity, WarningSeverity::Semantic);
}

#[test]
fn empty_namespaces_carry_no_thinking_provenance() {
    // An empty foreign namespace (created but never written) carries no
    // provenance: the signed block stays native — no ThinkingDropped.
    let mut block = ContentBlock::thinking_signed("planning", "sig");
    if let ContentBlock::Thinking { extra, .. } = &mut block {
        extra.namespace_mut("anthropic_messages");
    }
    let req = Request::with_messages(vec![
        Message::user_text("hi"),
        Message::assistant(vec![block]),
    ]);
    let (body, warnings) = build(&req);
    assert_eq!(
        body["contents"][1]["parts"][0],
        json!({"text": "planning", "thought": true, "thoughtSignature": "sig"})
    );
    assert!(warnings.is_empty(), "{warnings:?}");

    // An empty own namespace is no different: the optimistic-replay arm
    // (signature, no non-empty namespace) keeps the block native.
    let mut block = ContentBlock::thinking_signed("planning", "sig");
    if let ContentBlock::Thinking { extra, .. } = &mut block {
        extra.namespace_mut(FMT);
    }
    let req = Request::with_messages(vec![
        Message::user_text("hi"),
        Message::assistant(vec![block]),
    ]);
    let (body, warnings) = build(&req);
    assert_eq!(
        body["contents"][1]["parts"][0],
        json!({"text": "planning", "thought": true, "thoughtSignature": "sig"})
    );
    assert!(warnings.is_empty(), "{warnings:?}");

    // Signature-less with only an empty own namespace: no provenance at
    // all, so it is plaintext-only thinking — dropped with a warning.
    let mut block = ContentBlock::thinking("planning");
    if let ContentBlock::Thinking { extra, .. } = &mut block {
        extra.namespace_mut(FMT);
    }
    let req = Request::with_messages(vec![
        Message::user_text("hi"),
        Message::assistant(vec![block, ContentBlock::text("a")]),
    ]);
    let (body, warnings) = build(&req);
    assert_eq!(body["contents"][1]["parts"], json!([{"text": "a"}]));
    let w = find(&warnings, &WarningCode::ThinkingDropped);
    assert_eq!(w.severity, WarningSeverity::Semantic);
}

#[test]
fn tool_call_thought_signature_rides_extra() {
    let block = ContentBlock::tool_call_with_id("c1", "f", "{}").with_extra(
        FMT,
        "thoughtSignature",
        "c2lnLXRj",
    );
    let req = Request::with_messages(vec![
        Message::user_text("hi"),
        Message::assistant(vec![block]),
    ]);
    let (body, warnings) = build(&req);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body["contents"][1]["parts"][0],
        json!({
            "functionCall": {"id": "c1", "name": "f", "args": {}},
            "thoughtSignature": "c2lnLXRj",
        })
    );
}

#[test]
fn missing_thinking_with_tool_calls_warns_or_fills() {
    let mut req = Request::with_messages(vec![
        Message::user_text("hi"),
        Message::assistant(vec![ContentBlock::tool_call_with_id("c1", "f", "{}")]),
        Message::tool(vec![ContentBlock::tool_result_text(
            Some("c1".to_owned()),
            "ok",
        )]),
    ]);
    req.reasoning = Some(Reasoning::effort(Effort::High));
    let (_, warnings) = build(&req);
    let w = find(&warnings, &WarningCode::MissingThinkingWithToolCalls);
    assert_eq!(w.severity, WarningSeverity::Semantic);
    assert_eq!(w.location, "/contents/1");

    // With fill_missing_thinking the placeholder is inserted (and, being
    // plaintext-only, subsequently dropped on this signature-validated
    // target — the design documents that the option cannot help here). The
    // Filled message says so instead of implying a fix.
    let mut options = ConvertOptions::default();
    options.fill_missing_thinking = Some("tool call".to_owned());
    let (_, warnings) =
        request_from_ir(&req, &options, &GoogleGenerateContentOptions::default()).unwrap();
    let filled = find(&warnings, &WarningCode::MissingThinkingFilled);
    assert_eq!(
        filled.message,
        "inserted a placeholder thinking block before the tool calls; this format \
         drops the placeholder during serialization (no unsigned thinking channel) \
         — set `thinking_as_text` to carry it as text"
    );
    find(&warnings, &WarningCode::ThinkingDropped);
    assert!(!codes(&warnings).contains(&WarningCode::MissingThinkingWithToolCalls));

    // With thinking_as_text the placeholder actually reaches the wire, so
    // the message keeps its plain form and nothing is dropped.
    options.thinking_as_text = true;
    let (body, warnings) =
        request_from_ir(&req, &options, &GoogleGenerateContentOptions::default()).unwrap();
    let filled = find(&warnings, &WarningCode::MissingThinkingFilled);
    assert_eq!(
        filled.message,
        "inserted a placeholder thinking block before the tool calls"
    );
    assert!(!codes(&warnings).contains(&WarningCode::ThinkingDropped));
    assert_eq!(
        body["contents"][1]["parts"][0],
        json!({"text": "tool call", "thought": true})
    );

    // No reasoning configured → no warning at all.
    req.reasoning = None;
    let (_, warnings) = build(&req);
    assert!(warnings.is_empty());
}

// ------------------------------------------------------------ orphan policy

fn orphan_request() -> Request {
    Request::with_messages(vec![
        Message::user_text("hi"),
        Message::assistant(vec![
            ContentBlock::thinking_signed("planning", "sig"),
            ContentBlock::tool_call_with_id("c1", "f", "{}"),
        ]),
    ])
}

#[test]
fn orphan_passthrough_sends_as_is() {
    let (body, warnings) = build(&orphan_request());
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(body["contents"][1]["parts"].as_array().unwrap().len(), 2);
}

#[test]
fn orphan_drop_trailing_removes_calls_and_flags_thinking() {
    let mut options = ConvertOptions::default();
    options.orphan_tool_calls = OrphanToolCalls::DropTrailing;
    let (body, warnings) = request_from_ir(
        &orphan_request(),
        &options,
        &GoogleGenerateContentOptions::default(),
    )
    .unwrap();
    let parts = body["contents"][1]["parts"].as_array().unwrap();
    assert_eq!(parts.len(), 1, "only the thinking part remains: {parts:?}");
    find(&warnings, &WarningCode::OrphanToolCallsDropped);
    let w = find(&warnings, &WarningCode::ThinkingOrphaned);
    assert_eq!(w.severity, WarningSeverity::Semantic);

    // A message left empty is removed entirely.
    let req = Request::with_messages(vec![
        Message::user_text("hi"),
        Message::assistant(vec![ContentBlock::tool_call_with_id("c1", "f", "{}")]),
    ]);
    let (body, warnings) =
        request_from_ir(&req, &options, &GoogleGenerateContentOptions::default()).unwrap();
    assert_eq!(body["contents"].as_array().unwrap().len(), 1);
    find(&warnings, &WarningCode::OrphanToolCallsDropped);
}

#[test]
fn orphan_synthesize_error_appends_results() {
    let mut options = ConvertOptions::default();
    options.orphan_tool_calls = OrphanToolCalls::SynthesizeError;
    let (body, warnings) = request_from_ir(
        &orphan_request(),
        &options,
        &GoogleGenerateContentOptions::default(),
    )
    .unwrap();
    find(&warnings, &WarningCode::OrphanToolCallsSynthesized);
    let contents = body["contents"].as_array().unwrap();
    assert_eq!(contents.len(), 3);
    assert_eq!(
        contents[2]["parts"][0],
        json!({"functionResponse": {
            "id": "c1",
            "name": "f",
            "response": {"error": "cancelled"},
        }})
    );
}

#[test]
fn orphan_mid_array_only_warns() {
    let req = Request::with_messages(vec![
        Message::user_text("hi"),
        Message::assistant(vec![ContentBlock::tool_call_with_id("c1", "f", "{}")]),
        Message::user_text("never answered"),
    ]);
    let (body, warnings) = build(&req);
    let w = find(&warnings, &WarningCode::OrphanToolCalls);
    assert_eq!(w.severity, WarningSeverity::Semantic);
    assert_eq!(body["contents"].as_array().unwrap().len(), 3);
}

// -------------------------------------------------------------- extra & hooks

#[test]
fn extra_merges_at_every_level() {
    let mut req = Request::with_messages(vec![Message::user(vec![
        ContentBlock::text("hi").with_extra(FMT, "partMetadata", json!({"k": 1})),
    ])]);
    req.messages[0].extra.set(FMT, "contentTag", "tagged");
    req.extra.set(FMT, "cachedContent", "cachedContents/abc");
    req.extra.set(
        FMT,
        "safetySettings",
        json!([{"category": "X", "threshold": "BLOCK_NONE"}]),
    );
    let (body, warnings) = build(&req);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body["contents"][0],
        json!({
            "role": "user",
            "parts": [{"text": "hi", "partMetadata": {"k": 1}}],
            "contentTag": "tagged",
        })
    );
    assert_eq!(body["cachedContent"], "cachedContents/abc");
    assert_eq!(body["safetySettings"][0]["threshold"], "BLOCK_NONE");
}

#[test]
fn hooks_visit_serialized_contents_only() {
    let format = GoogleGenerateContent;
    let mut req = Request::with_messages(vec![
        Message::system_text("hoisted"),
        Message::user_text("hi"),
        Message::assistant_text("hello"),
        Message::tool(vec![
            ContentBlock::tool_result_text(None, "r").with_tool_name("f"),
        ]),
    ]);
    req.system = Some(vec![ContentBlock::text("sys")]);
    let mut c = ctx(CallMode::Unary);
    c.hooks = RequestHooks::new()
        .with_on_message(|index, role, value| {
            // Serialized sequence: user, model, user (tool results merge into
            // a user turn); the hoisted system content is not visited.
            let expected = match index {
                0 | 2 => Role::User,
                1 => Role::Assistant,
                _ => panic!("unexpected message index {index}"),
            };
            assert_eq!(*role, expected);
            value["hookIndex"] = json!(index);
            Ok(())
        })
        .with_on_request(|value| {
            value["hooked"] = json!(true);
            Ok(())
        });
    let built = format.build_request(&req, &c).unwrap();
    let body: Value = serde_json::from_slice(&built.body).unwrap();
    assert_eq!(body["hooked"], json!(true));
    let contents = body["contents"].as_array().unwrap();
    assert_eq!(contents.len(), 3);
    for (i, content) in contents.iter().enumerate() {
        assert_eq!(content["hookIndex"], json!(i));
    }
    assert!(body["systemInstruction"].get("hookIndex").is_none());
}

#[test]
fn strict_mode_escalates_semantic_warnings() {
    let mut req = user_req("hi");
    req.parallel_tool_calls = Some(false);
    req.tools = Some(vec![Tool::function(FunctionTool::new("f"))]);
    let err = request_from_ir(
        &req,
        &ConvertOptions::new().strict(true),
        &GoogleGenerateContentOptions::default(),
    )
    .unwrap_err();
    match err {
        Error::Conversion(ConversionError::Strict { warnings, .. }) => {
            assert_eq!(codes(&warnings), vec![WarningCode::SerialToolCallsDropped]);
        }
        other => panic!("unexpected error: {other:?}"),
    }

    // Cosmetic warnings pass the strict gate.
    let mut req = user_req("hi");
    req.cache_key = Some("k".to_owned());
    let (_, warnings) = request_from_ir(
        &req,
        &ConvertOptions::new().strict(true),
        &GoogleGenerateContentOptions::default(),
    )
    .unwrap();
    assert_eq!(codes(&warnings), vec![WarningCode::CacheKeyDropped]);
}

// ------------------------------------------------------- models, countTokens

#[test]
fn models_request_paginates_with_protected_query() {
    let format = GoogleGenerateContent;
    let c = ctx(CallMode::Unary);
    let built = format.build_models_request(&c, None).unwrap();
    assert_eq!(built.method, http::Method::GET);
    assert_eq!(
        built.url.to_string(),
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000"
    );
    assert_eq!(built.auth.unwrap().header.as_str(), "x-goog-api-key");

    let built = format.build_models_request(&c, Some("tok-2")).unwrap();
    assert_eq!(
        built.url.to_string(),
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000&pageToken=tok-2"
    );

    // User-supplied pageSize conflicts with the pagination mechanism.
    let mut c = ctx(CallMode::Unary);
    c.extra_query.push(("pageSize".to_owned(), "5".to_owned()));
    assert!(matches!(
        format.build_models_request(&c, None),
        Err(Error::Conversion(ConversionError::ProtectedQueryKey { .. }))
    ));
}

#[test]
fn count_tokens_request_wraps_the_chat_body() {
    let format = GoogleGenerateContent;
    let mut req = user_req("count me");
    req.extra.set(FMT, "cachedContent", "cachedContents/abc");
    let mut c = ctx(CallMode::Unary);
    c.hooks = RequestHooks::new().with_on_request(|value| {
        value["hookMark"] = json!(true);
        Ok(())
    });
    let built = format.build_count_tokens_request(&req, &c).unwrap();
    assert_eq!(
        built.url.to_string(),
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:countTokens"
    );
    assert!(built.warnings.is_empty());
    let body: Value = serde_json::from_slice(&built.body).unwrap();
    let inner = &body["generateContentRequest"];
    assert_eq!(inner["model"], "models/gemini-2.5-pro");
    assert_eq!(inner["contents"][0]["parts"][0]["text"], "count me");
    // extra and hooks act on the chat body exactly as for send.
    assert_eq!(inner["cachedContent"], "cachedContents/abc");
    assert_eq!(inner["hookMark"], json!(true));
    assert_eq!(body.as_object().unwrap().len(), 1);
}

#[test]
fn hook_errors_abort_the_build() {
    let format = GoogleGenerateContent;
    let mut c = ctx(CallMode::Unary);
    c.hooks = RequestHooks::new().with_on_request(|_| Err(llm_api::HookError::new("rejected")));
    let err = format.build_request(&user_req("hi"), &c).unwrap_err();
    assert!(matches!(err, Error::Hook(_)));
}

#[test]
fn empty_and_prefill_conversations_serialize() {
    // Empty request: contents is still present (required upstream).
    let (body, warnings) = build(&Request::new());
    assert!(warnings.is_empty());
    assert_eq!(body, json!({"contents": []}));

    // Trailing assistant prefill passes through as-is.
    let req = Request::with_messages(vec![
        Message::user_text("write a poem"),
        Message::assistant_text("Roses are"),
    ]);
    let (body, _) = build(&req);
    assert_eq!(
        body["contents"][1],
        json!({"role": "model", "parts": [{"text": "Roses are"}]})
    );
}

// ---------------------------------------------------------------- safety settings

#[test]
fn safety_settings_presets() {
    let req = user_req("hi");

    // Default: no `safetySettings` — the provider's defaults apply.
    let (body, warnings) = build(&req);
    assert!(body.get("safetySettings").is_none());
    assert!(warnings.is_empty(), "{warnings:?}");

    let with_preset = |preset: GoogleSafetySettings| {
        let mut opts = GoogleGenerateContentOptions::default();
        opts.safety_settings = preset;
        request_from_ir(&req, &ConvertOptions::default(), &opts).unwrap()
    };

    let (body, warnings) = with_preset(GoogleSafetySettings::DisableAiStudio);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body["safetySettings"],
        json!([
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_JAILBREAK", "threshold": "OFF"},
        ])
    );

    let (body, warnings) = with_preset(GoogleSafetySettings::DisableVertex);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body["safetySettings"],
        json!([
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_IMAGE_HATE", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_IMAGE_HARASSMENT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_JAILBREAK", "threshold": "OFF"},
        ])
    );
}

#[test]
fn safety_settings_extra_overrides_preset() {
    // `request.extra` merges after the generated body: an explicit
    // `safetySettings` replaces the preset array wholesale (RFC 7396).
    let custom = json!([
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"}
    ]);
    let mut req = user_req("hi");
    req.extra.set(FMT, "safetySettings", custom.clone());
    let mut opts = GoogleGenerateContentOptions::default();
    opts.safety_settings = GoogleSafetySettings::DisableAiStudio;
    let (body, _) = request_from_ir(&req, &ConvertOptions::default(), &opts).unwrap();
    assert_eq!(body["safetySettings"], custom);
}

#[test]
fn safety_settings_preset_reaches_chat_and_count_bodies() {
    // The trait path threads `BuildCtx.format_options`, and the count
    // adapter reuses the chat build — the preset lands in both bodies.
    let mut c = ctx(CallMode::Unary);
    c.format_options.google_generate_content.safety_settings =
        GoogleSafetySettings::DisableAiStudio;
    let format = GoogleGenerateContent;
    let req = user_req("hi");

    let built = format.build_request(&req, &c).unwrap();
    let body: Value = serde_json::from_slice(&built.body).unwrap();
    assert_eq!(body["safetySettings"].as_array().unwrap().len(), 6);

    let built = format.build_count_tokens_request(&req, &c).unwrap();
    let body: Value = serde_json::from_slice(&built.body).unwrap();
    assert_eq!(
        body["generateContentRequest"]["safetySettings"]
            .as_array()
            .unwrap()
            .len(),
        6
    );
}

#[test]
fn extra_rewritten_parts_forces_a_new_turn_instead_of_panicking() {
    // README promises arbitrary override/delete via `extra`; a message
    // that deletes or rewrites its shared turn's `parts` must not panic
    // the next same-side message — it opens a fresh turn instead.
    for patch in [json!(null), json!({"custom": true}), json!("scalar")] {
        let mut first = Message::user_text("one");
        first.extra.set(FMT, "parts", patch.clone());
        let req = Request::with_messages(vec![first, Message::user_text("two")]);
        let (body, warnings) = build(&req);
        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 2, "{patch}: {body}");
        // The customized turn keeps the user's shape (null deletes the key).
        if patch.is_null() {
            assert!(contents[0].get("parts").is_none(), "{body}");
        } else {
            assert_eq!(contents[0]["parts"], patch, "{body}");
        }
        // The second message lands intact in a fresh same-role turn.
        assert_eq!(contents[1]["role"], json!("user"), "{body}");
        assert_eq!(contents[1]["parts"][0]["text"], json!("two"), "{body}");
        let w = find(&warnings, &WarningCode::MalformedField);
        assert_eq!(w.location, "/contents/0/parts");
    }
}
