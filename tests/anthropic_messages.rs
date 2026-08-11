//! Build-side (IR → request) tests for the `anthropic_messages` format.

use llm_api::formats::anthropic_messages::AnthropicMessages;
use llm_api::{
    AnthropicAuthStyle, ApiFormat, AuthScheme, BuildCtx, CacheHint, CallMode, ContentBlock,
    ConversionError, ConversionWarning, Effort, EndpointUrl, Error, FunctionTool, HookError,
    Message, OrphanToolCalls, OutputFormat, Reasoning, Request, RequestHooks, Role, Tool,
    ToolChoice, ToolOutputBlock, WarningCode, WarningSeverity,
};
use serde_json::{Value, json};

const FMT: &str = "anthropic_messages";

fn ctx() -> BuildCtx {
    BuildCtx::new(
        EndpointUrl::base("https://api.anthropic.com/v1").unwrap(),
        "claude-sonnet-5",
        CallMode::Unary,
    )
}

fn req(messages: Vec<Message>) -> Request {
    let mut r = Request::with_messages(messages);
    r.max_output_tokens = Some(100);
    r
}

fn build_with(r: &Request, c: &BuildCtx) -> (Value, Vec<ConversionWarning>) {
    let built = AnthropicMessages.build_request(r, c).unwrap();
    (serde_json::from_slice(&built.body).unwrap(), built.warnings)
}

fn build(r: &Request) -> (Value, Vec<ConversionWarning>) {
    build_with(r, &ctx())
}

fn find<'w>(
    warnings: &'w [ConversionWarning],
    code: &WarningCode,
) -> Option<&'w ConversionWarning> {
    warnings.iter().find(|w| w.code == *code)
}

#[test]
fn minimal_request_shape_url_and_headers() {
    let r = req(vec![Message::user_text("hi")]);
    let built = AnthropicMessages.build_request(&r, &ctx()).unwrap();
    assert_eq!(built.method.as_str(), "POST");
    assert_eq!(
        built.url.to_string(),
        "https://api.anthropic.com/v1/messages"
    );
    assert_eq!(
        built.headers.get("content-type").unwrap(),
        "application/json"
    );
    assert_eq!(
        built.headers.get("anthropic-version").unwrap(),
        "2023-06-01"
    );
    assert!(built.headers.get("anthropic-beta").is_none());
    let auth = built.auth.unwrap();
    assert_eq!(auth.header.as_str(), "x-api-key");
    assert_eq!(auth.prefix, None);
    let body: Value = serde_json::from_slice(&built.body).unwrap();
    assert_eq!(
        body,
        json!({
            "model": "claude-sonnet-5",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        })
    );
    assert!(built.warnings.is_empty());
}

#[test]
fn anthropic_options_control_headers_and_auth() {
    let r = req(vec![Message::user_text("hi")]);
    let mut c = ctx();
    c.format_options.anthropic.auth_style = AnthropicAuthStyle::Bearer;
    c.format_options.anthropic.version = None;
    c.format_options.anthropic.betas = vec![
        "interleaved-thinking-2025-05-14".into(),
        "context-1m-2025-08-07".into(),
    ];
    let built = AnthropicMessages.build_request(&r, &c).unwrap();
    assert!(built.headers.get("anthropic-version").is_none());
    assert_eq!(
        built.headers.get("anthropic-beta").unwrap(),
        "interleaved-thinking-2025-05-14,context-1m-2025-08-07"
    );
    assert_eq!(built.auth.unwrap(), AuthScheme::bearer());
}

#[test]
fn max_tokens_is_required() {
    let r = Request::with_messages(vec![Message::user_text("hi")]);
    let err = AnthropicMessages.build_request(&r, &ctx()).unwrap_err();
    assert!(matches!(
        err,
        Error::Conversion(ConversionError::MissingRequired { .. })
    ));
    // Strict off makes no difference: required data is an error in both modes.
    let mut c = ctx();
    c.convert.strict = true;
    assert!(AnthropicMessages.build_request(&r, &c).is_err());

    // A configured fallback fills it; an explicit value wins over it.
    let mut c2 = ctx();
    c2.format_options.anthropic.default_max_tokens = Some(512);
    let (body, _) = build_with(&r, &c2);
    assert_eq!(body["max_tokens"], json!(512));
    let r2 = req(vec![Message::user_text("hi")]);
    let (body2, _) = build_with(&r2, &c2);
    assert_eq!(body2["max_tokens"], json!(100));
}

#[test]
fn streaming_sets_stream_flag() {
    let r = req(vec![Message::user_text("hi")]);
    let mut c = ctx();
    c.mode = CallMode::Streaming;
    let (body, _) = build_with(&r, &c);
    assert_eq!(body["stream"], json!(true));
    let (unary, _) = build(&r);
    assert!(unary.get("stream").is_none());
}

#[test]
fn sampling_parameters_map_or_warn() {
    let mut r = req(vec![Message::user_text("hi")]);
    r.temperature = Some(0.5);
    r.top_p = Some(0.9);
    r.top_k = Some(40);
    r.stop_sequences = Some(vec!["END".into()]);
    r.seed = Some(7);
    r.frequency_penalty = Some(0.1);
    r.presence_penalty = Some(0.2);
    let (body, warnings) = build(&r);
    assert_eq!(body["temperature"], json!(0.5));
    assert_eq!(body["top_p"], json!(0.9));
    assert_eq!(body["top_k"], json!(40));
    assert_eq!(body["stop_sequences"], json!(["END"]));
    assert!(body.get("seed").is_none());
    let dropped: Vec<&str> = warnings
        .iter()
        .filter(|w| w.code == WarningCode::SamplingParameterDropped)
        .map(|w| w.location.as_str())
        .collect();
    assert_eq!(
        dropped,
        vec!["/seed", "/frequency_penalty", "/presence_penalty"]
    );
    assert!(
        warnings
            .iter()
            .all(|w| w.severity == WarningSeverity::Cosmetic)
    );
}

#[test]
fn non_finite_sampling_values_are_conversion_errors() {
    // All four f64 fields are checked — including the penalties Anthropic
    // would otherwise drop with a warning: non-finite is a caller bug. The
    // check runs before the required-`max_tokens` gate.
    type SetField = fn(&mut Request, f64);
    let fields: [(SetField, &str); 4] = [
        (|r, v| r.temperature = Some(v), "/temperature"),
        (|r, v| r.top_p = Some(v), "/top_p"),
        (|r, v| r.frequency_penalty = Some(v), "/frequency_penalty"),
        (|r, v| r.presence_penalty = Some(v), "/presence_penalty"),
    ];
    for (set, expected) in fields {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut r = req(vec![Message::user_text("hi")]);
            set(&mut r, bad);
            let err = AnthropicMessages.build_request(&r, &ctx()).unwrap_err();
            match err {
                Error::Conversion(ConversionError::NonFiniteNumber { location, .. }) => {
                    assert_eq!(location, expected);
                }
                other => panic!("expected NonFiniteNumber for {bad}, got {other:?}"),
            }
        }
    }

    // Finite values — zeroes and extremes — keep the existing behavior.
    let mut r = req(vec![Message::user_text("hi")]);
    r.temperature = Some(-0.0);
    r.top_p = Some(f64::MAX);
    let (body, warnings) = build(&r);
    assert_eq!(body["temperature"], json!(-0.0));
    assert_eq!(body["top_p"], json!(f64::MAX));
    assert!(warnings.is_empty(), "{warnings:?}");
}

#[test]
fn metadata_maps_only_user_id() {
    let mut r = req(vec![Message::user_text("hi")]);
    let mut meta = serde_json::Map::new();
    meta.insert("user_id".into(), json!("u-1"));
    meta.insert("session".into(), json!("s-9"));
    r.metadata = Some(meta);
    let (body, warnings) = build(&r);
    assert_eq!(body["metadata"], json!({"user_id": "u-1"}));
    let w = find(&warnings, &WarningCode::MetadataDropped).unwrap();
    assert_eq!(w.location, "/metadata");
    assert!(w.message.contains("session"));
}

#[test]
fn cache_key_dropped() {
    let mut r = req(vec![Message::user_text("hi")]);
    r.cache_key = Some("k".into());
    let (body, warnings) = build(&r);
    assert!(body.get("prompt_cache_key").is_none());
    assert!(find(&warnings, &WarningCode::CacheKeyDropped).is_some());
}

#[test]
fn system_string_form_and_array_form() {
    // Exactly one plain text segment: string form.
    let r = req(vec![Message::user_text("hi")]).with_system_text("Be helpful.");
    let (body, _) = build(&r);
    assert_eq!(body["system"], json!("Be helpful."));

    // A cache hint forces the array form and becomes cache_control.
    let mut r2 = req(vec![Message::user_text("hi")]);
    r2.system = Some(vec![
        ContentBlock::text("Long.").with_cache(CacheHint::with_ttl("1h")),
    ]);
    let (body2, _) = build(&r2);
    assert_eq!(
        body2["system"],
        json!([{"type": "text", "text": "Long.", "cache_control": {"type": "ephemeral", "ttl": "1h"}}])
    );

    // Block-level extra also forces the array form and is merged.
    let mut r3 = req(vec![Message::user_text("hi")]);
    r3.system = Some(vec![ContentBlock::text("T").with_extra(
        FMT,
        "citations",
        json!([]),
    )]);
    let (body3, _) = build(&r3);
    assert_eq!(
        body3["system"],
        json!([{"type": "text", "text": "T", "citations": []}])
    );
}

#[test]
fn leading_system_messages_hoist_and_mid_stays_in_array() {
    let r = req(vec![
        Message::system_text("lead 1"),
        Message::system_text("lead 2"),
        Message::user_text("hi"),
        Message::system_text("mid"),
        Message::user_text("more"),
    ])
    .with_system_text("first");
    let (body, warnings) = build(&r);
    // Request.system content first, then the leading run, in order.
    assert_eq!(
        body["system"],
        json!([
            {"type": "text", "text": "first"},
            {"type": "text", "text": "lead 1"},
            {"type": "text", "text": "lead 2"},
        ])
    );
    assert_eq!(body["messages"][0]["role"], json!("user"));
    assert_eq!(body["messages"][1]["role"], json!("system"));
    assert_eq!(
        body["messages"][1]["content"],
        json!([{"type": "text", "text": "mid"}])
    );
    assert!(warnings.is_empty());
}

#[test]
fn downgrade_mid_system_and_developer() {
    let r = req(vec![
        Message::user_text("hi"),
        Message::system_text("mid"),
        Message::developer_text("dev"),
    ]);
    let mut c = ctx();
    c.format_options.anthropic.downgrade_mid_system = true;
    let (body, warnings) = build_with(&r, &c);
    assert_eq!(body["messages"][1]["role"], json!("user"));
    assert_eq!(body["messages"][2]["role"], json!("user"));
    let downgrades: Vec<&str> = warnings
        .iter()
        .filter(|w| w.code == WarningCode::RoleDowngraded)
        .map(|w| w.location.as_str())
        .collect();
    assert_eq!(downgrades, vec!["/messages/1", "/messages/2"]);

    // Developer downgrades even without the option; system does not.
    let (body2, warnings2) = build(&r);
    assert_eq!(body2["messages"][1]["role"], json!("system"));
    assert_eq!(body2["messages"][2]["role"], json!("user"));
    assert_eq!(
        warnings2
            .iter()
            .filter(|w| w.code == WarningCode::RoleDowngraded)
            .count(),
        1
    );
}

#[test]
fn request_system_rejects_non_text() {
    let mut r = req(vec![Message::user_text("hi")]);
    r.system = Some(vec![ContentBlock::image_url("https://x/y.png")]);
    let err = AnthropicMessages.build_request(&r, &ctx()).unwrap_err();
    assert!(matches!(
        err,
        Error::Conversion(ConversionError::InvalidBlockForRole {
            role: Role::System,
            ..
        })
    ));
}

#[test]
fn system_channel_serializing_to_nothing_is_omitted_with_warning() {
    // A hoisted leading system message whose every block is a dropped
    // foreign opaque: the `system` key is omitted and the channel-level
    // omission is disclosed on top of the block-level drop, both at the
    // `/system` container (its entries never reached the wire).
    let mut sys_msg = Message::new(
        Role::System,
        vec![ContentBlock::opaque(
            "openai_responses",
            json!({"type": "note"}),
        )],
    );
    sys_msg.extra.set(FMT, "meta_key", 1);
    let r = req(vec![sys_msg, Message::user_text("hi")]);
    let (body, warnings) = build(&r);
    assert!(body.get("system").is_none(), "{body}");
    assert_eq!(
        body["messages"],
        json!([{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
    );
    let dropped = find(&warnings, &WarningCode::EmptyMessageDropped).unwrap();
    assert_eq!(dropped.location, "/system");
    assert_eq!(dropped.severity, WarningSeverity::Semantic);
    assert!(dropped.message.contains("system channel"), "{warnings:?}");
    let opaque = find(&warnings, &WarningCode::OpaqueDropped).unwrap();
    assert_eq!(opaque.location, "/system");
    // The hoisted message's own extra lost its landing spot too.
    let extra = find(&warnings, &WarningCode::ExtraDropped).unwrap();
    assert_eq!(extra.location, "/system");

    // With surviving content the channel is kept and nothing extra warns.
    let r2 = req(vec![
        Message::new(
            Role::System,
            vec![
                ContentBlock::opaque("openai_responses", json!({"type": "note"})),
                ContentBlock::text("rule"),
            ],
        ),
        Message::user_text("hi"),
    ]);
    let (body2, w2) = build(&r2);
    // Exactly one surviving plain text entry: the string form applies.
    assert_eq!(body2["system"], json!("rule"));
    assert!(find(&w2, &WarningCode::EmptyMessageDropped).is_none());
    // The block-level drop keeps its positional location.
    assert_eq!(
        find(&w2, &WarningCode::OpaqueDropped).unwrap().location,
        "/system/0"
    );
}

#[test]
fn genuinely_empty_system_input_stays_silent() {
    // `Request.system: Some(vec![])` is the caller's own data: the key is
    // omitted without any warning (current-behavior pin).
    let mut r = req(vec![Message::user_text("hi")]);
    r.system = Some(vec![]);
    let (body, warnings) = build(&r);
    assert!(body.get("system").is_none(), "{body}");
    assert!(warnings.is_empty(), "{warnings:?}");

    // A zero-block hoisted system message contributes nothing, silently.
    let r2 = req(vec![
        Message::new(Role::System, vec![]),
        Message::user_text("hi"),
    ]);
    let (body2, w2) = build(&r2);
    assert!(body2.get("system").is_none(), "{body2}");
    assert!(w2.is_empty(), "{w2:?}");
}

#[test]
fn role_block_validity_is_structural() {
    // User + ToolCall.
    let r = req(vec![Message::user(vec![ContentBlock::tool_call(
        "f", "{}",
    )])]);
    assert!(matches!(
        AnthropicMessages.build_request(&r, &ctx()).unwrap_err(),
        Error::Conversion(ConversionError::InvalidBlockForRole {
            role: Role::User,
            ..
        })
    ));
    // Assistant + ToolResult.
    let r2 = req(vec![Message::assistant(vec![
        ContentBlock::tool_result_text(Some("t".into()), "x"),
    ])]);
    assert!(matches!(
        AnthropicMessages.build_request(&r2, &ctx()).unwrap_err(),
        Error::Conversion(ConversionError::InvalidBlockForRole {
            role: Role::Assistant,
            ..
        })
    ));
    // Tool + Text.
    let r3 = req(vec![Message::tool(vec![ContentBlock::text("x")])]);
    assert!(matches!(
        AnthropicMessages.build_request(&r3, &ctx()).unwrap_err(),
        Error::Conversion(ConversionError::InvalidBlockForRole {
            role: Role::Tool,
            ..
        })
    ));
}

#[test]
fn invalid_block_for_role_uses_ir_coordinates() {
    // A hoisted leading System message shifts wire indices down; the error
    // still points at the offending block's IR coordinate.
    let r = req(vec![
        Message::system_text("lead"),
        Message::user_text("q"),
        Message::assistant(vec![ContentBlock::tool_result_text(Some("t".into()), "x")]),
    ]);
    match AnthropicMessages.build_request(&r, &ctx()).unwrap_err() {
        Error::Conversion(ConversionError::InvalidBlockForRole { role, location, .. }) => {
            assert_eq!(role, Role::Assistant);
            assert_eq!(location, "/messages/2/content/0");
        }
        other => panic!("unexpected: {other:?}"),
    }

    // Turn-group re-merge packs Tool + User into one wire message; the
    // location stays IR-relative (wire would say /messages/2/content/2).
    let r2 = req(vec![
        Message::user_text("q"),
        Message::assistant(vec![ContentBlock::tool_call_with_id("t1", "f", "{}")]),
        Message::tool(vec![ContentBlock::tool_result_text(
            Some("t1".into()),
            "ok",
        )])
        .with_turn_group(1),
        Message::user(vec![
            ContentBlock::text("continue"),
            ContentBlock::tool_call("g", "{}"),
        ])
        .with_turn_group(1),
    ]);
    match AnthropicMessages.build_request(&r2, &ctx()).unwrap_err() {
        Error::Conversion(ConversionError::InvalidBlockForRole { role, location, .. }) => {
            assert_eq!(role, Role::User);
            assert_eq!(location, "/messages/3/content/1");
        }
        other => panic!("unexpected: {other:?}"),
    }

    // A hoisted System message's own invalid block: IR coordinates too
    // (the wire location would be inside the top-level system array).
    let r3 = req(vec![
        Message::new(
            Role::System,
            vec![
                ContentBlock::text("rules"),
                ContentBlock::image_url("https://x/y.png"),
            ],
        ),
        Message::user_text("q"),
    ]);
    match AnthropicMessages.build_request(&r3, &ctx()).unwrap_err() {
        Error::Conversion(ConversionError::InvalidBlockForRole { role, location, .. }) => {
            assert_eq!(role, Role::System);
            assert_eq!(location, "/messages/0/content/1");
        }
        other => panic!("unexpected: {other:?}"),
    }

    // `Request.system` blocks report their own IR coordinate.
    let mut r4 = req(vec![Message::user_text("q")]);
    r4.system = Some(vec![
        ContentBlock::text("ok"),
        ContentBlock::image_url("https://x/y.png"),
    ]);
    match AnthropicMessages.build_request(&r4, &ctx()).unwrap_err() {
        Error::Conversion(ConversionError::InvalidBlockForRole { location, .. }) => {
            assert_eq!(location, "/system/1");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn image_sources_map_and_assistant_images_drop() {
    let r = req(vec![Message::user(vec![
        ContentBlock::image_url("https://x/y.png"),
        ContentBlock::image_base64("image/png", "AAAA").with_cache(CacheHint::new()),
        ContentBlock::image_file_id("file_1"),
    ])]);
    let (body, warnings) = build(&r);
    assert_eq!(
        body["messages"][0]["content"],
        json!([
            {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
             "cache_control": {"type": "ephemeral"}},
            {"type": "image", "source": {"type": "file", "file_id": "file_1"}},
        ])
    );
    assert!(warnings.is_empty());

    let r2 = req(vec![Message::assistant(vec![
        ContentBlock::text("look"),
        ContentBlock::image_url("https://x/y.png"),
    ])]);
    let (body2, warnings2) = build(&r2);
    assert_eq!(
        body2["messages"][0]["content"],
        json!([{"type": "text", "text": "look"}])
    );
    let w = find(&warnings2, &WarningCode::ImageSourceUnsupported).unwrap();
    assert_eq!(w.severity, WarningSeverity::Semantic);
    assert_eq!(w.location, "/messages/0/content/1");
}

#[test]
fn tool_call_requires_id_and_object_arguments() {
    // Missing id.
    let r = req(vec![Message::assistant(vec![ContentBlock::tool_call(
        "f", "{}",
    )])]);
    assert!(matches!(
        AnthropicMessages.build_request(&r, &ctx()).unwrap_err(),
        Error::Conversion(ConversionError::MissingRequired { .. })
    ));
    // Invalid JSON, both modes.
    let r2 = req(vec![Message::assistant(vec![
        ContentBlock::tool_call_with_id("c1", "f", "{broken"),
    ])]);
    assert!(matches!(
        AnthropicMessages.build_request(&r2, &ctx()).unwrap_err(),
        Error::Conversion(ConversionError::InvalidToolArguments { .. })
    ));
    let mut strict = ctx();
    strict.convert.strict = true;
    assert!(AnthropicMessages.build_request(&r2, &strict).is_err());
    // Non-object JSON.
    let r3 = req(vec![Message::assistant(vec![
        ContentBlock::tool_call_with_id("c1", "f", "[1]"),
    ])]);
    assert!(matches!(
        AnthropicMessages.build_request(&r3, &ctx()).unwrap_err(),
        Error::Conversion(ConversionError::InvalidToolArguments { .. })
    ));
    // Valid arguments become the input object.
    let r4 = req(vec![Message::assistant(vec![
        ContentBlock::tool_call_with_id("c1", "f", r#"{"a": 1}"#),
    ])]);
    let (body, _) = build(&r4);
    assert_eq!(
        body["messages"][0]["content"][0],
        json!({"type": "tool_use", "id": "c1", "name": "f", "input": {"a": 1}})
    );
}

#[test]
fn tool_result_encodings() {
    // Empty content: `content` omitted.
    let empty = ContentBlock::tool_result(Some("t0".into()), vec![]);
    // Single plain text: string shorthand.
    let single = ContentBlock::tool_result_text(Some("t1".into()), "ok");
    // Multiple blocks keep boundaries; images map natively.
    let multi = ContentBlock::tool_result(
        Some("t2".into()),
        vec![
            ToolOutputBlock::text("a"),
            ToolOutputBlock::text("b"),
            ToolOutputBlock::image(llm_api::ImageSource::base64("image/png", "CCCC")),
        ],
    )
    .with_is_error(true)
    .with_cache(CacheHint::new());
    let r = req(vec![Message::tool(vec![empty, single, multi])]);
    let (body, warnings) = build(&r);
    assert_eq!(
        body["messages"][0]["content"],
        json!([
            {"type": "tool_result", "tool_use_id": "t0"},
            {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
            {"type": "tool_result", "tool_use_id": "t2", "content": [
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "CCCC"}},
            ], "is_error": true, "cache_control": {"type": "ephemeral"}},
        ])
    );
    assert!(warnings.is_empty());

    // Missing tool_use_id is structural.
    let r2 = req(vec![Message::tool(vec![ContentBlock::tool_result_text(
        None, "x",
    )])]);
    assert!(matches!(
        AnthropicMessages.build_request(&r2, &ctx()).unwrap_err(),
        Error::Conversion(ConversionError::MissingRequired { .. })
    ));
}

#[test]
fn tool_result_nested_cache_hints_drop_and_extra_forces_array() {
    let mut text = ToolOutputBlock::text("cached");
    if let ToolOutputBlock::Text { cache, .. } = &mut text {
        *cache = Some(CacheHint::new());
    }
    let r = req(vec![Message::tool(vec![ContentBlock::tool_result(
        Some("t1".into()),
        vec![text],
    )])]);
    let (body, warnings) = build(&r);
    // Hint dropped (v1 pin), string form still used.
    assert_eq!(
        body["messages"][0]["content"][0],
        json!({"type": "tool_result", "tool_use_id": "t1", "content": "cached"})
    );
    let w = find(&warnings, &WarningCode::CacheHintDropped).unwrap();
    assert_eq!(w.location, "/messages/0/content/0/content/0");

    // A nested block with format extra forces the array form.
    let mut text2 = ToolOutputBlock::text("x");
    if let ToolOutputBlock::Text { extra, .. } = &mut text2 {
        extra.set(FMT, "cache_control", json!({"type": "ephemeral"}));
    }
    let r2 = req(vec![Message::tool(vec![ContentBlock::tool_result(
        Some("t2".into()),
        vec![text2],
    )])]);
    let (body2, _) = build(&r2);
    assert_eq!(
        body2["messages"][0]["content"][0]["content"],
        json!([{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}])
    );
}

#[test]
fn thinking_provenance_rules() {
    // Native by signature with no namespace.
    let r = req(vec![Message::assistant(vec![
        ContentBlock::thinking_signed("chain", "sig-1"),
        ContentBlock::text("answer"),
    ])]);
    let (body, warnings) = build(&r);
    assert_eq!(
        body["messages"][0]["content"][0],
        json!({"type": "thinking", "thinking": "chain", "signature": "sig-1"})
    );
    assert!(warnings.is_empty());

    // Native by namespace, no signature: emitted without a signature key.
    let r2 = req(vec![Message::assistant(vec![
        ContentBlock::thinking("t").with_extra(FMT, "estimated_tokens", 3),
    ])]);
    let (body2, _) = build(&r2);
    assert_eq!(
        body2["messages"][0]["content"][0],
        json!({"type": "thinking", "thinking": "t", "estimated_tokens": 3})
    );

    // Redacted: reconstructed from the signature; the marker is consumed.
    let mut redacted = ContentBlock::thinking_signed("", "BLOB").with_extra(FMT, "redacted", true);
    if let ContentBlock::Thinking { text, .. } = &mut redacted {
        *text = None;
    }
    let r3 = req(vec![Message::assistant(vec![redacted])]);
    let (body3, _) = build(&r3);
    assert_eq!(
        body3["messages"][0]["content"][0],
        json!({"type": "redacted_thinking", "data": "BLOB"})
    );

    // Foreign: dropped with a semantic warning.
    let r4 = req(vec![Message::assistant(vec![
        ContentBlock::thinking_signed("t", "s").with_extra("openai_responses", "id", "rs_1"),
        ContentBlock::text("a"),
    ])]);
    let (body4, warnings4) = build(&r4);
    assert_eq!(
        body4["messages"][0]["content"],
        json!([{"type": "text", "text": "a"}])
    );
    let w = find(&warnings4, &WarningCode::ThinkingDropped).unwrap();
    assert_eq!(w.severity, WarningSeverity::Semantic);

    // Plaintext-only thinking is foreign here too.
    let r5 = req(vec![Message::assistant(vec![
        ContentBlock::thinking("t"),
        ContentBlock::text("a"),
    ])]);
    let (_, warnings5) = build(&r5);
    assert!(find(&warnings5, &WarningCode::ThinkingDropped).is_some());

    // thinking_as_text re-emits the text without a signature.
    let mut c = ctx();
    c.convert.thinking_as_text = true;
    let (body6, warnings6) = build_with(&r4, &c);
    assert_eq!(
        body6["messages"][0]["content"][0],
        json!({"type": "thinking", "thinking": "t"})
    );
    assert!(find(&warnings6, &WarningCode::ThinkingSignatureDropped).is_some());
}

#[test]
fn empty_namespaces_carry_no_thinking_provenance() {
    // An empty foreign namespace (created but never written) carries no
    // provenance: the signed block stays native — no ThinkingDropped.
    let mut block = ContentBlock::thinking_signed("t", "sig");
    if let ContentBlock::Thinking { extra, .. } = &mut block {
        extra.namespace_mut("openai_responses");
    }
    let r = req(vec![Message::assistant(vec![block])]);
    let (body, warnings) = build(&r);
    assert_eq!(
        body["messages"][0]["content"][0],
        json!({"type": "thinking", "thinking": "t", "signature": "sig"})
    );
    assert!(warnings.is_empty(), "{warnings:?}");

    // An empty own namespace is no different: the optimistic-replay arm
    // (signature, no non-empty namespace) keeps the block native.
    let mut block = ContentBlock::thinking_signed("t", "sig");
    if let ContentBlock::Thinking { extra, .. } = &mut block {
        extra.namespace_mut(FMT);
    }
    let r = req(vec![Message::assistant(vec![block])]);
    let (body, warnings) = build(&r);
    assert_eq!(
        body["messages"][0]["content"][0],
        json!({"type": "thinking", "thinking": "t", "signature": "sig"})
    );
    assert!(warnings.is_empty(), "{warnings:?}");

    // Signature-less with only an empty own namespace: no provenance at
    // all, so it is plaintext-only thinking — dropped with a warning.
    let mut block = ContentBlock::thinking("t");
    if let ContentBlock::Thinking { extra, .. } = &mut block {
        extra.namespace_mut(FMT);
    }
    let r = req(vec![Message::assistant(vec![
        block,
        ContentBlock::text("a"),
    ])]);
    let (body, warnings) = build(&r);
    assert_eq!(
        body["messages"][0]["content"],
        json!([{"type": "text", "text": "a"}])
    );
    assert!(find(&warnings, &WarningCode::ThinkingDropped).is_some());
}

#[test]
fn tools_map_with_synthesized_empty_schema() {
    let mut r = req(vec![Message::user_text("hi")]);
    r.tools = Some(vec![
        Tool::function(
            FunctionTool::new("no_params")
                .with_description("d")
                .with_strict(true),
        ),
        Tool::function(
            FunctionTool::new("with_schema")
                .with_parameters(json!({"type": "object", "properties": {"q": {"type": "string"}}}))
                .with_cache(CacheHint::with_ttl("5m")),
        ),
        Tool::opaque(
            FMT,
            json!({"type": "web_search_20260209", "name": "web_search"}),
        ),
        Tool::opaque("google_generate_content", json!({"foo": 1})),
    ]);
    let (body, warnings) = build(&r);
    assert_eq!(
        body["tools"],
        json!([
            {"name": "no_params", "description": "d",
             "input_schema": {"type": "object", "properties": {}, "additionalProperties": false},
             "strict": true},
            {"name": "with_schema",
             "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
             "cache_control": {"type": "ephemeral", "ttl": "5m"}},
            {"type": "web_search_20260209", "name": "web_search"},
        ])
    );
    let synth = find(&warnings, &WarningCode::EmptyParametersSynthesized).unwrap();
    assert_eq!(synth.location, "/tools/0/input_schema");
    let opaque = find(&warnings, &WarningCode::OpaqueDropped).unwrap();
    assert_eq!(opaque.location, "/tools/3");
    assert_eq!(opaque.severity, WarningSeverity::Semantic);
}

#[test]
fn tool_choice_cells() {
    for (choice, expected) in [
        (ToolChoice::Auto, json!({"type": "auto"})),
        (ToolChoice::None, json!({"type": "none"})),
        (ToolChoice::Required, json!({"type": "any"})),
        (ToolChoice::tool("f"), json!({"type": "tool", "name": "f"})),
    ] {
        let mut r = req(vec![Message::user_text("hi")]);
        r.tools = Some(vec![Tool::function(FunctionTool::new("f"))]);
        r.tool_choice = Some(choice);
        let (body, _) = build(&r);
        assert_eq!(body["tool_choice"], expected);
    }
}

#[test]
fn tool_choice_requires_wire_tools() {
    // `tool_choice` follows the emitted tools, not the IR list: without any
    // wire tool no `tool_choice` key is emitted (the synthesized
    // parallel-fold carrier included) and the settings warn.
    let expect_ignored = |r: &Request| {
        let (body, warnings) = build(r);
        assert!(body.get("tool_choice").is_none(), "{body}");
        let w = find(&warnings, &WarningCode::ToolChoiceIgnored)
            .unwrap_or_else(|| panic!("expected ToolChoiceIgnored: {warnings:?}"));
        assert_eq!(w.severity, WarningSeverity::Cosmetic);
        assert_eq!(w.location, "/tool_choice");
        (body, warnings)
    };

    // No IR tools at all.
    let mut r = req(vec![Message::user_text("hi")]);
    r.tool_choice = Some(ToolChoice::Required);
    let (body, _) = expect_ignored(&r);
    assert!(body.get("tools").is_none());

    // Every tool a dropped foreign opaque: no `tools` key either.
    r.tools = Some(vec![Tool::opaque(
        "openai_responses",
        json!({"type": "web_search"}),
    )]);
    let (body, warnings) = expect_ignored(&r);
    assert!(body.get("tools").is_none(), "{body}");
    assert!(find(&warnings, &WarningCode::OpaqueDropped).is_some());

    // An explicitly empty IR list replays `"tools": []` but still counts
    // as no wire tools.
    r.tools = Some(vec![]);
    let (body, _) = expect_ignored(&r);
    assert_eq!(body["tools"], json!([]));

    // A choice plus a parallel flag: both settings warn, nothing is
    // emitted.
    let mut r2 = req(vec![Message::user_text("hi")]);
    r2.tool_choice = Some(ToolChoice::Auto);
    r2.parallel_tool_calls = Some(false);
    let (_, warnings) = expect_ignored(&r2);
    assert!(find(&warnings, &WarningCode::ParallelToolCallsIgnored).is_some());

    // No choice + parallel Some(false) without wire tools: the carrier is
    // not synthesized; the flag warns (no ToolChoiceIgnored — there was no
    // tool_choice to ignore).
    let mut r3 = req(vec![Message::user_text("hi")]);
    r3.tools = Some(vec![Tool::opaque(
        "openai_responses",
        json!({"type": "web_search"}),
    )]);
    r3.parallel_tool_calls = Some(false);
    let (body3, w3) = build(&r3);
    assert!(body3.get("tool_choice").is_none(), "{body3}");
    assert!(find(&w3, &WarningCode::ParallelToolCallsIgnored).is_some());
    assert!(find(&w3, &WarningCode::ToolChoiceIgnored).is_none());
}

#[test]
fn parallel_tool_calls_combinations() {
    let base = || {
        let mut r = req(vec![Message::user_text("hi")]);
        r.tools = Some(vec![Tool::function(
            FunctionTool::new("f").with_parameters(json!({"type": "object"})),
        )]);
        r
    };
    // Explicit choice: the inverted flag is attached, for both values.
    let mut r = base();
    r.tool_choice = Some(ToolChoice::Auto);
    r.parallel_tool_calls = Some(false);
    let (body, w) = build(&r);
    assert_eq!(
        body["tool_choice"],
        json!({"type": "auto", "disable_parallel_tool_use": true})
    );
    assert!(w.is_empty());

    let mut r2 = base();
    r2.tool_choice = Some(ToolChoice::Required);
    r2.parallel_tool_calls = Some(true);
    let (body2, _) = build(&r2);
    assert_eq!(
        body2["tool_choice"],
        json!({"type": "any", "disable_parallel_tool_use": false})
    );

    // ToolChoice::None takes no flag.
    let mut r3 = base();
    r3.tool_choice = Some(ToolChoice::None);
    r3.parallel_tool_calls = Some(false);
    let (body3, w3) = build(&r3);
    assert_eq!(body3["tool_choice"], json!({"type": "none"}));
    assert!(find(&w3, &WarningCode::ParallelToolCallsIgnored).is_some());

    // No choice + Some(false): synthesized auto object.
    let mut r4 = base();
    r4.parallel_tool_calls = Some(false);
    let (body4, w4) = build(&r4);
    assert_eq!(
        body4["tool_choice"],
        json!({"type": "auto", "disable_parallel_tool_use": true})
    );
    assert!(w4.is_empty());

    // No choice + Some(true): canonicalized to nothing, silently.
    let mut r5 = base();
    r5.parallel_tool_calls = Some(true);
    let (body5, w5) = build(&r5);
    assert!(body5.get("tool_choice").is_none());
    assert!(w5.is_empty());

    // No tools: the setting is meaningless.
    let mut r6 = req(vec![Message::user_text("hi")]);
    r6.parallel_tool_calls = Some(false);
    let (body6, w6) = build(&r6);
    assert!(body6.get("tool_choice").is_none());
    assert!(find(&w6, &WarningCode::ParallelToolCallsIgnored).is_some());
}

#[test]
fn reasoning_table_cells() {
    // enabled: false → disabled; enabled: true → adaptive.
    let mut r = req(vec![Message::user_text("hi")]);
    r.reasoning = Some(Reasoning::enabled(false));
    let (body, _) = build(&r);
    assert_eq!(body["thinking"], json!({"type": "disabled"}));

    r.reasoning = Some(Reasoning::enabled(true));
    let (body2, _) = build(&r);
    assert_eq!(body2["thinking"], json!({"type": "adaptive"}));

    // Effort tiers.
    for (effort, expected) in [
        (Effort::Low, Some("low")),
        (Effort::Medium, Some("medium")),
        (Effort::High, Some("high")),
        (Effort::XHigh, Some("xhigh")),
        (Effort::Max, Some("max")),
        (Effort::Other("turbo".into()), Some("turbo")),
        (Effort::None, None),
        (Effort::Minimal, None),
    ] {
        let mut r = req(vec![Message::user_text("hi")]);
        r.reasoning = Some(Reasoning::effort(effort.clone()));
        let (body, warnings) = build(&r);
        match expected {
            Some(s) => {
                assert_eq!(body["output_config"]["effort"], json!(s));
                assert!(find(&warnings, &WarningCode::EffortUnsupported).is_none());
            }
            None => {
                assert!(body.get("output_config").is_none());
                let w = find(&warnings, &WarningCode::EffortUnsupported).unwrap();
                assert_eq!(w.severity, WarningSeverity::Semantic);
                assert_eq!(w.location, "/output_config/effort");
            }
        }
    }
}

#[test]
fn include_thoughts_display_rules() {
    // Rides the adaptive object produced by enabled: true.
    let mut r = req(vec![Message::user_text("hi")]);
    let mut reasoning = Reasoning::enabled(true);
    reasoning.include_thoughts = Some(true);
    r.reasoning = Some(reasoning.clone());
    let (body, _) = build(&r);
    assert_eq!(
        body["thinking"],
        json!({"type": "adaptive", "display": "summarized"})
    );

    reasoning.include_thoughts = Some(false);
    r.reasoning = Some(reasoning);
    let (body2, _) = build(&r);
    assert_eq!(
        body2["thinking"],
        json!({"type": "adaptive", "display": "omitted"})
    );

    // Effort enables thinking; adaptive carries the display.
    let mut r2 = req(vec![Message::user_text("hi")]);
    let mut reasoning2 = Reasoning::effort(Effort::High);
    reasoning2.include_thoughts = Some(true);
    r2.reasoning = Some(reasoning2);
    let (body3, w3) = build(&r2);
    assert_eq!(
        body3["thinking"],
        json!({"type": "adaptive", "display": "summarized"})
    );
    assert_eq!(body3["output_config"]["effort"], json!("high"));
    assert!(w3.is_empty());

    // No carrier on disabled or without any thinking basis.
    let mut r3 = req(vec![Message::user_text("hi")]);
    let mut reasoning3 = Reasoning::enabled(false);
    reasoning3.include_thoughts = Some(true);
    r3.reasoning = Some(reasoning3);
    let (body4, w4) = build(&r3);
    assert_eq!(body4["thinking"], json!({"type": "disabled"}));
    assert!(find(&w4, &WarningCode::IncludeThoughtsUnsupported).is_some());

    let mut r4 = req(vec![Message::user_text("hi")]);
    let mut reasoning4 = Reasoning::new();
    reasoning4.include_thoughts = Some(true);
    r4.reasoning = Some(reasoning4);
    let (body5, w5) = build(&r4);
    assert!(body5.get("thinking").is_none());
    assert!(find(&w5, &WarningCode::IncludeThoughtsUnsupported).is_some());
}

#[test]
fn reasoning_conflicts_effort_wins() {
    // enabled: true + effort: None.
    let mut r = req(vec![Message::user_text("hi")]);
    let mut reasoning = Reasoning::enabled(true);
    reasoning.effort = Some(Effort::None);
    r.reasoning = Some(reasoning);
    let (body, warnings) = build(&r);
    assert!(body.get("thinking").is_none());
    assert!(find(&warnings, &WarningCode::ReasoningConflict).is_some());
    assert!(find(&warnings, &WarningCode::EffortUnsupported).is_some());

    // enabled: false + effort: high.
    let mut r2 = req(vec![Message::user_text("hi")]);
    let mut reasoning2 = Reasoning::enabled(false);
    reasoning2.effort = Some(Effort::High);
    r2.reasoning = Some(reasoning2);
    let (body2, warnings2) = build(&r2);
    assert!(body2.get("thinking").is_none());
    assert_eq!(body2["output_config"]["effort"], json!("high"));
    assert!(find(&warnings2, &WarningCode::ReasoningConflict).is_some());
}

#[test]
fn extra_rewrites_generated_thinking_object() {
    // § 4.7: request-level extra fully rewrites the generated thinking
    // object (budget re-enable).
    let mut r = req(vec![Message::user_text("hi")]);
    r.reasoning = Some(Reasoning::enabled(true));
    r.extra.set(
        FMT,
        "thinking",
        json!({"type": "enabled", "budget_tokens": 2048}),
    );
    let (body, _) = build(&r);
    assert_eq!(
        body["thinking"],
        json!({"type": "enabled", "budget_tokens": 2048})
    );

    // Same through the reasoning-node extra (merged into /thinking).
    let mut r2 = req(vec![Message::user_text("hi")]);
    let mut reasoning = Reasoning::enabled(true);
    reasoning.include_thoughts = Some(false);
    reasoning.extra.set(FMT, "type", "enabled");
    reasoning.extra.set(FMT, "budget_tokens", 1024);
    r2.reasoning = Some(reasoning);
    let (body2, _) = build(&r2);
    assert_eq!(
        body2["thinking"],
        json!({"type": "enabled", "display": "omitted", "budget_tokens": 1024})
    );
}

#[test]
fn output_format_cells() {
    let mut r = req(vec![Message::user_text("hi")]);
    r.output_format = Some(OutputFormat::json_schema(json!({"type": "object"})));
    let (body, warnings) = build(&r);
    assert_eq!(
        body["output_config"]["format"],
        json!({"type": "json_schema", "schema": {"type": "object"}})
    );
    assert!(warnings.is_empty());

    // name/strict have no channel: cosmetic warning.
    let mut r2 = req(vec![Message::user_text("hi")]);
    r2.output_format = Some(
        OutputFormat::json_schema(json!({"type": "object"}))
            .with_name("answer")
            .with_strict(true),
    );
    let (body2, warnings2) = build(&r2);
    assert_eq!(
        body2["output_config"]["format"],
        json!({"type": "json_schema", "schema": {"type": "object"}})
    );
    let w = find(&warnings2, &WarningCode::OutputFormatDetailDropped).unwrap();
    assert_eq!(w.severity, WarningSeverity::Cosmetic);
    assert!(w.message.contains("name") && w.message.contains("strict"));

    // JsonObject: no schema-less mode — semantic.
    let mut r3 = req(vec![Message::user_text("hi")]);
    r3.output_format = Some(OutputFormat::json_object());
    let (body3, warnings3) = build(&r3);
    assert!(body3.get("output_config").is_none());
    let w3 = find(&warnings3, &WarningCode::JsonModeUnsupported).unwrap();
    assert_eq!(w3.severity, WarningSeverity::Semantic);
}

#[test]
fn strict_mode_escalates_and_extra_overrides() {
    let mut r = req(vec![Message::user_text("hi")]);
    r.output_format = Some(OutputFormat::json_object());
    let mut c = ctx();
    c.convert.strict = true;
    let err = AnthropicMessages.build_request(&r, &c).unwrap_err();
    assert!(matches!(
        err,
        Error::Conversion(ConversionError::Strict { .. })
    ));

    // Deleting the warned pointer through extra marks it overridden and
    // clears the strict gate.
    let mut r2 = r.clone();
    r2.extra.set(FMT, "output_config", json!({"format": null}));
    let built = AnthropicMessages.build_request(&r2, &c).unwrap();
    let w = find(&built.warnings, &WarningCode::JsonModeUnsupported).unwrap();
    assert!(w.overridden);

    // Cosmetic warnings never trip strict.
    let mut r3 = req(vec![Message::user_text("hi")]);
    r3.seed = Some(1);
    assert!(AnthropicMessages.build_request(&r3, &c).is_ok());
}

#[test]
fn hooks_visit_serialized_messages_and_can_fail() {
    let r = req(vec![
        Message::system_text("hoisted"),
        Message::user_text("hi"),
        Message::assistant_text("yo"),
        Message::developer_text("dev"),
    ]);
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let seen2 = seen.clone();
    let mut c = ctx();
    c.hooks = RequestHooks::new()
        .with_on_message(move |index, role, value| {
            seen2.lock().unwrap().push((index, *role));
            value["tagged"] = json!(index);
            Ok(())
        })
        .with_on_request(|value| {
            value["hooked"] = json!(true);
            Ok(())
        });
    let (body, _) = build_with(&r, &c);
    // The hoisted system message is not visited; the developer message is
    // visited with its serialized (downgraded) role.
    assert_eq!(
        *seen.lock().unwrap(),
        vec![(0, Role::User), (1, Role::Assistant), (2, Role::User)]
    );
    assert_eq!(body["messages"][0]["tagged"], json!(0));
    assert_eq!(body["messages"][2]["tagged"], json!(2));
    assert_eq!(body["hooked"], json!(true));

    let mut c2 = ctx();
    c2.hooks = RequestHooks::new().with_on_request(|_| Err(HookError::new("boom")));
    assert!(matches!(
        AnthropicMessages.build_request(&r, &c2).unwrap_err(),
        Error::Hook(_)
    ));
}

#[test]
fn merge_consecutive_roles_and_signature_barrier() {
    // Off by default: adjacent same-role messages stay separate.
    let r = req(vec![Message::user_text("a"), Message::user_text("b")]);
    let (body, _) = build(&r);
    assert_eq!(body["messages"].as_array().unwrap().len(), 2);

    // On: merged into one wire turn, blocks in order.
    let mut c = ctx();
    c.format_options.anthropic.merge_consecutive_roles = true;
    let (body2, w2) = build_with(&r, &c);
    assert_eq!(
        body2["messages"],
        json!([{"role": "user", "content": [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ]}])
    );
    assert!(w2.is_empty());

    // A signed thinking block is a merge barrier.
    let r2 = req(vec![
        Message::assistant(vec![ContentBlock::thinking_signed("t", "sig")]),
        Message::assistant_text("more"),
    ]);
    let (body3, w3) = build_with(&r2, &c);
    assert_eq!(body3["messages"].as_array().unwrap().len(), 2);
    let w = find(&w3, &WarningCode::MergeBlockedBySignature).unwrap();
    assert_eq!(w.severity, WarningSeverity::Semantic);

    // A Google thoughtSignature on a tool call is a barrier too.
    let signed_call = ContentBlock::tool_call_with_id("c1", "f", "{}").with_extra(
        "google_generate_content",
        "thoughtSignature",
        "gsig",
    );
    let r3 = req(vec![
        Message::assistant(vec![signed_call]),
        Message::assistant_text("more"),
        Message::tool(vec![ContentBlock::tool_result_text(
            Some("c1".into()),
            "ok",
        )]),
    ]);
    let (body4, w4) = build_with(&r3, &c);
    assert_eq!(body4["messages"].as_array().unwrap().len(), 3);
    assert!(find(&w4, &WarningCode::MergeBlockedBySignature).is_some());
}

#[test]
fn lone_opaque_wire_message_is_verbatim_and_never_merges() {
    // A lone own-format Opaque with a string `role` (the unknown-role
    // parse home) re-emits verbatim, untouched by merge_consecutive_roles.
    let wire_msg = json!({"role": "critic", "content": "meh", "weight": 2});
    let mut lone = Message::user(vec![ContentBlock::opaque(FMT, wire_msg.clone())]);
    lone.extra.set(FMT, "priority", 1); // message-level extra still merges
    let r = req(vec![Message::user_text("a"), lone, Message::user_text("b")]);
    let mut c = ctx();
    c.format_options.anthropic.merge_consecutive_roles = true;
    let (body, warnings) = build_with(&r, &c);
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3, "no merging through the verbatim message");
    assert_eq!(
        messages[0]["content"],
        json!([{"type": "text", "text": "a"}])
    );
    assert_eq!(
        messages[1],
        json!({"role": "critic", "content": "meh", "weight": 2, "priority": 1})
    );
    assert_eq!(
        messages[2]["content"],
        json!([{"type": "text", "text": "b"}])
    );
    // Silent barrier: not a same-role merge conflict.
    assert!(find(&warnings, &WarningCode::MergeBlockedBySignature).is_none());

    // A two-block message or a value without a role string is not verbatim:
    // the opaque serializes inside a regular content array as before.
    let r2 = req(vec![Message::user(vec![
        ContentBlock::opaque(FMT, wire_msg.clone()),
        ContentBlock::text("t"),
    ])]);
    let (body2, _) = build(&r2);
    assert_eq!(
        body2["messages"][0],
        json!({"role": "user", "content": [wire_msg, {"type": "text", "text": "t"}]})
    );
}

#[test]
fn lone_opaque_wire_message_blocks_turn_group_re_merge() {
    let wire_msg = json!({"role": "critic", "content": "meh"});
    let r = req(vec![
        Message::user_text("q"),
        Message::assistant(vec![ContentBlock::tool_call_with_id("t1", "f", "{}")]),
        Message::tool(vec![ContentBlock::tool_result_text(
            Some("t1".into()),
            "ok",
        )])
        .with_turn_group(7),
        Message::user(vec![ContentBlock::opaque(FMT, wire_msg.clone())]).with_turn_group(7),
        Message::user_text("continue").with_turn_group(7),
    ]);
    let (body, _) = build(&r);
    let messages = body["messages"].as_array().unwrap();
    // Without the verbatim message the grouped Tool + User pair would merge
    // into one wire turn; the verbatim message stays standalone and keeps
    // its neighbours apart.
    assert_eq!(messages.len(), 5);
    assert_eq!(
        messages[2]["content"],
        json!([{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}])
    );
    assert_eq!(messages[3], wire_msg);
    assert_eq!(
        messages[4]["content"],
        json!([{"type": "text", "text": "continue"}])
    );
}

#[test]
fn turn_group_meta_re_merges_tool_and_user_messages() {
    let call = ContentBlock::tool_call_with_id("t1", "f", "{}");
    let r = req(vec![
        Message::user_text("q"),
        Message::assistant(vec![call]),
        Message::tool(vec![ContentBlock::tool_result_text(
            Some("t1".into()),
            "ok",
        )])
        .with_turn_group(3),
        Message::user_text("continue").with_turn_group(3),
    ]);
    let (body, _) = build(&r);
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(
        messages[2]["content"],
        json!([
            {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
            {"type": "text", "text": "continue"},
        ])
    );

    // Without meta the messages stay separate wire messages.
    let r2 = req(vec![
        Message::user_text("q"),
        Message::assistant(vec![ContentBlock::tool_call_with_id("t1", "f", "{}")]),
        Message::tool(vec![ContentBlock::tool_result_text(
            Some("t1".into()),
            "ok",
        )]),
        Message::user_text("continue"),
    ]);
    let (body2, _) = build(&r2);
    assert_eq!(body2["messages"].as_array().unwrap().len(), 4);

    // Different group ids do not merge.
    let r3 = req(vec![
        Message::user_text("q"),
        Message::assistant(vec![ContentBlock::tool_call_with_id("t1", "f", "{}")]),
        Message::tool(vec![ContentBlock::tool_result_text(
            Some("t1".into()),
            "ok",
        )])
        .with_turn_group(1),
        Message::user_text("continue").with_turn_group(2),
    ]);
    let (body3, _) = build(&r3);
    assert_eq!(body3["messages"].as_array().unwrap().len(), 4);
}

#[test]
fn orphan_tool_call_policies() {
    // Mid-array orphans always warn, never repaired.
    let mid = req(vec![
        Message::assistant(vec![ContentBlock::tool_call_with_id("t1", "f", "{}")]),
        Message::user_text("unrelated"),
    ]);
    let (body, warnings) = build(&mid);
    assert_eq!(body["messages"].as_array().unwrap().len(), 2);
    let w = find(&warnings, &WarningCode::OrphanToolCalls).unwrap();
    assert_eq!(w.severity, WarningSeverity::Semantic);

    // Trailing orphans pass through silently by default.
    let trailing = req(vec![
        Message::user_text("q"),
        Message::assistant(vec![ContentBlock::tool_call_with_id("t1", "f", "{}")]),
    ]);
    let (_, w_pass) = build(&trailing);
    assert!(w_pass.is_empty());

    // DropTrailing removes the calls; an orphaned thinking block warns.
    let with_thinking = req(vec![
        Message::user_text("q"),
        Message::assistant(vec![
            ContentBlock::thinking_signed("t", "sig"),
            ContentBlock::tool_call_with_id("t1", "f", "{}"),
        ]),
    ]);
    let mut c = ctx();
    c.convert.orphan_tool_calls = OrphanToolCalls::DropTrailing;
    let (body2, w2) = build_with(&with_thinking, &c);
    assert_eq!(
        body2["messages"][1]["content"],
        json!([{"type": "thinking", "thinking": "t", "signature": "sig"}])
    );
    assert!(find(&w2, &WarningCode::OrphanToolCallsDropped).is_some());
    assert!(find(&w2, &WarningCode::ThinkingOrphaned).is_some());

    // The whole message goes when it becomes empty.
    let (body3, w3) = build_with(&trailing, &c);
    assert_eq!(body3["messages"].as_array().unwrap().len(), 1);
    assert!(find(&w3, &WarningCode::OrphanToolCallsDropped).is_some());

    // SynthesizeError appends error results.
    let mut c2 = ctx();
    c2.convert.orphan_tool_calls = OrphanToolCalls::SynthesizeError;
    let (body4, w4) = build_with(&trailing, &c2);
    let messages = body4["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(
        messages[2]["content"],
        json!([{"type": "tool_result", "tool_use_id": "t1", "content": "cancelled", "is_error": true}])
    );
    assert!(find(&w4, &WarningCode::OrphanToolCallsSynthesized).is_some());
}

#[test]
fn missing_thinking_with_tool_calls() {
    let mut r = req(vec![
        Message::user_text("q"),
        Message::assistant(vec![ContentBlock::tool_call_with_id("t1", "f", "{}")]),
        Message::tool(vec![ContentBlock::tool_result_text(
            Some("t1".into()),
            "ok",
        )]),
    ]);
    r.reasoning = Some(Reasoning::enabled(true));
    let (_, warnings) = build(&r);
    let w = find(&warnings, &WarningCode::MissingThinkingWithToolCalls).unwrap();
    assert_eq!(w.severity, WarningSeverity::Semantic);
    assert_eq!(w.location, "/messages/1");

    // fill_missing_thinking inserts a plaintext block; with thinking_as_text
    // it survives into the wire request.
    let mut c = ctx();
    c.convert.fill_missing_thinking = Some("tool call".into());
    c.convert.thinking_as_text = true;
    let (body, w2) = build_with(&r, &c);
    assert_eq!(
        body["messages"][1]["content"][0],
        json!({"type": "thinking", "thinking": "tool call"})
    );
    assert!(find(&w2, &WarningCode::MissingThinkingFilled).is_some());
    assert!(find(&w2, &WarningCode::MissingThinkingWithToolCalls).is_none());

    // An existing thinking block suppresses both.
    let mut r2 = r.clone();
    r2.messages[1]
        .content
        .insert(0, ContentBlock::thinking_signed("t", "s"));
    let (_, w3) = build(&r2);
    assert!(find(&w3, &WarningCode::MissingThinkingWithToolCalls).is_none());
}

#[test]
fn empty_serialized_messages_are_omitted_with_warning() {
    // A foreign thinking-only assistant turn (thinking_as_text off)
    // serializes to zero blocks: the wire message is omitted instead of
    // sending `content: []`, and the omission is disclosed.
    let foreign =
        ContentBlock::thinking_signed("t", "s").with_extra("openai_responses", "id", "rs_1");
    let r = req(vec![
        Message::user_text("q"),
        Message::assistant(vec![foreign]),
        Message::user_text("next"),
    ]);
    let (body, warnings) = build(&r);
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2, "{messages:?}");
    assert_eq!(
        messages[0]["content"],
        json!([{"type": "text", "text": "q"}])
    );
    assert_eq!(
        messages[1]["content"],
        json!([{"type": "text", "text": "next"}])
    );
    let dropped: Vec<&ConversionWarning> = warnings
        .iter()
        .filter(|w| w.code == WarningCode::EmptyMessageDropped)
        .collect();
    assert_eq!(dropped.len(), 1, "{warnings:?}");
    assert_eq!(dropped[0].location, "/messages");
    assert_eq!(dropped[0].severity, WarningSeverity::Semantic);
    assert!(
        dropped[0].message.contains("IR message(s) 1"),
        "{}",
        dropped[0].message
    );
    // The block-level drop is disclosed too, re-addressed to the container
    // (its positional wire location no longer exists).
    let td = find(&warnings, &WarningCode::ThinkingDropped).unwrap();
    assert_eq!(td.location, "/messages");

    // Native thinking is unaffected.
    let r2 = req(vec![
        Message::user_text("q"),
        Message::assistant(vec![ContentBlock::thinking_signed("t", "sig")]),
        Message::user_text("next"),
    ]);
    let (body2, w2) = build(&r2);
    assert_eq!(body2["messages"].as_array().unwrap().len(), 3);
    assert!(find(&w2, &WarningCode::EmptyMessageDropped).is_none());

    // A truly empty IR message still serializes as `content: []`, silently.
    let r3 = req(vec![Message::user_text("q"), Message::user(vec![])]);
    let (body3, w3) = build(&r3);
    assert_eq!(body3["messages"][1], json!({"role": "user", "content": []}));
    assert!(find(&w3, &WarningCode::EmptyMessageDropped).is_none());

    // All wire messages omitted: the empty array is sent as-is (nothing is
    // fabricated); the block- and message-level warnings disclose why.
    let r4 = req(vec![Message::user(vec![ContentBlock::opaque(
        "openai_responses",
        json!({"x": 1}),
    )])]);
    let (body4, w4) = build(&r4);
    assert_eq!(body4["messages"], json!([]));
    assert!(find(&w4, &WarningCode::OpaqueDropped).is_some());
    assert!(find(&w4, &WarningCode::EmptyMessageDropped).is_some());
}

#[test]
fn merged_wire_message_omitted_only_when_fully_empty() {
    // Same-role merge folds an unsigned foreign thinking-only message into
    // its neighbour: with surviving neighbour content the wire message is
    // kept, no omission.
    let foreign = || ContentBlock::thinking("t").with_extra("openai_responses", "id", "x");
    let mut c = ctx();
    c.format_options.anthropic.merge_consecutive_roles = true;
    let r = req(vec![
        Message::user_text("q"),
        Message::assistant(vec![foreign()]),
        Message::assistant_text("more"),
    ]);
    let (body, warnings) = build_with(&r, &c);
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(
        messages[1]["content"],
        json!([{"type": "text", "text": "more"}])
    );
    assert!(find(&warnings, &WarningCode::EmptyMessageDropped).is_none());

    // When the merged wire message ends up fully empty it is omitted; the
    // warning names the non-empty IR message that lost its content (the
    // truly empty neighbour is not blamed).
    let r2 = req(vec![
        Message::user_text("q"),
        Message::assistant(vec![foreign()]),
        Message::assistant(vec![]),
    ]);
    let (body2, w2) = build_with(&r2, &c);
    assert_eq!(body2["messages"].as_array().unwrap().len(), 1);
    let dropped = find(&w2, &WarningCode::EmptyMessageDropped).unwrap();
    assert!(
        dropped.message.contains("IR message(s) 1"),
        "{}",
        dropped.message
    );
}

#[test]
fn opaque_blocks_and_message_extra() {
    // Own-format opaque blocks serialize verbatim, in place.
    let doc = json!({"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "d"}});
    let mut msg = Message::user(vec![
        ContentBlock::opaque(FMT, doc.clone()),
        ContentBlock::text("after"),
    ]);
    msg.extra.set(FMT, "name", "bob");
    let r = req(vec![msg]);
    let (body, warnings) = build(&r);
    assert_eq!(
        body["messages"][0],
        json!({
            "role": "user",
            "content": [doc, {"type": "text", "text": "after"}],
            "name": "bob",
        })
    );
    assert!(warnings.is_empty());

    // Foreign opaque blocks drop with a semantic warning.
    let r2 = req(vec![Message::user(vec![
        ContentBlock::opaque("openai_responses", json!({"x": 1})),
        ContentBlock::text("t"),
    ])]);
    let (body2, warnings2) = build(&r2);
    assert_eq!(
        body2["messages"][0]["content"],
        json!([{"type": "text", "text": "t"}])
    );
    assert!(find(&warnings2, &WarningCode::OpaqueDropped).is_some());
}

#[test]
fn hoisted_system_message_carries_extra_and_own_opaque() {
    // A hoisted message's own extra merges into its first produced system
    // block; an own-format opaque block passes verbatim into the array.
    let mut sys_msg = Message::system_text("rules");
    sys_msg.extra.set(FMT, "meta_key", 1);
    let sys2 = Message::new(
        Role::System,
        vec![ContentBlock::opaque(
            FMT,
            json!({"type": "text", "text": "op", "citations": []}),
        )],
    );
    let r = req(vec![sys_msg, sys2, Message::user_text("hi")]);
    let (body, warnings) = build(&r);
    assert_eq!(
        body["system"],
        json!([
            {"type": "text", "text": "rules", "meta_key": 1},
            {"type": "text", "text": "op", "citations": []},
        ])
    );
    assert!(warnings.is_empty());
}

#[test]
fn request_extra_merges_last_and_can_delete() {
    let mut r = req(vec![Message::user_text("hi")]);
    r.temperature = Some(0.3);
    r.extra.set(FMT, "temperature", json!(0.9));
    r.extra.set(FMT, "top_p", Value::Null); // no-op delete
    r.extra.set(FMT, "service_tier", "auto");
    let (body, _) = build(&r);
    assert_eq!(body["temperature"], json!(0.9));
    assert_eq!(body["service_tier"], json!("auto"));
    assert!(body.get("top_p").is_none());
}
