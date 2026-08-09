//! Cross-format uniformity smoke tests: one IR, all four formats, through
//! the dynamic [`ApiFormat`] API. Depth per format lives in the per-format
//! test files; these assert the behaviors the § 6/§ 4.5 contracts require
//! to be uniform.

use llm_api::formats::anthropic_messages::AnthropicMessages;
use llm_api::formats::google_generate_content::GoogleGenerateContent;
use llm_api::formats::openai_chat_completions::OpenAiChatCompletions;
use llm_api::formats::openai_responses::OpenAiResponses;
use llm_api::{
    ApiFormat, BuildCtx, CallMode, ContentBlock, ConversionError, ConvertOptions, Effort,
    EndpointUrl, Error, Message, Reasoning, Request, Role, Tool, ToolChoice, WarningCode, ids,
};
use serde_json::Value;

fn formats() -> Vec<Box<dyn ApiFormat>> {
    vec![
        Box::new(OpenAiChatCompletions),
        Box::new(OpenAiResponses),
        Box::new(AnthropicMessages),
        Box::new(GoogleGenerateContent),
    ]
}

fn ctx() -> BuildCtx {
    BuildCtx::new(
        EndpointUrl::base("https://api.example.com/v1").unwrap(),
        "test-model",
        CallMode::Unary,
    )
}

fn strict_ctx() -> BuildCtx {
    let mut ctx = ctx();
    ctx.convert = ConvertOptions::default().strict(true);
    ctx
}

/// A small but complete tool-loop conversation valid on every format
/// (ids present, Anthropic `max_tokens` set, Google name resolvable).
fn tool_loop_request(is_error: Option<bool>) -> Request {
    let mut req = Request::with_messages(vec![
        Message::user_text("What's the weather?"),
        Message::assistant(vec![ContentBlock::tool_call_with_id(
            "c1",
            "get_weather",
            r#"{"city": "Paris"}"#,
        )]),
        Message::tool(vec![{
            let mut block = ContentBlock::tool_result_text(Some("c1".to_owned()), "rainy");
            if let Some(err) = is_error {
                block = block.with_is_error(err);
            }
            block
        }]),
    ]);
    req.max_output_tokens = Some(64);
    req.tools = Some(vec![Tool::function(
        llm_api::FunctionTool::new("get_weather")
            .with_parameters(serde_json::json!({"type": "object"})),
    )]);
    req
}

fn body_of(format: &dyn ApiFormat, req: &Request, ctx: &BuildCtx) -> (Value, Vec<WarningCode>) {
    let built = format.build_request(req, ctx).unwrap();
    let body: Value = serde_json::from_slice(&built.body).unwrap();
    let codes = built.warnings.iter().map(|w| w.code.clone()).collect();
    (body, codes)
}

#[test]
fn is_error_true_native_on_anthropic_and_google_only() {
    let req = tool_loop_request(Some(true));
    for format in formats() {
        let (body, codes) = body_of(format.as_ref(), &req, &ctx());
        let dropped = codes.contains(&WarningCode::IsErrorDropped);
        match format.id() {
            ids::OPENAI_CHAT_COMPLETIONS | ids::OPENAI_RESPONSES => {
                assert!(dropped, "{} must warn on is_error", format.id());
            }
            ids::ANTHROPIC_MESSAGES => {
                assert!(!dropped);
                let s = body.to_string();
                assert!(
                    s.contains("\"is_error\":true"),
                    "missing native is_error: {s}"
                );
            }
            ids::GOOGLE_GENERATE_CONTENT => {
                assert!(!dropped);
                let s = body.to_string();
                assert!(s.contains("\"error\""), "missing error key encoding: {s}");
            }
            other => panic!("unexpected format {other}"),
        }
    }
}

#[test]
fn is_error_false_is_silent_everywhere() {
    let req = tool_loop_request(Some(false));
    for format in formats() {
        let (_, codes) = body_of(format.as_ref(), &req, &ctx());
        assert!(
            !codes.contains(&WarningCode::IsErrorDropped),
            "{}: is_error Some(false) equals the default and must drop silently",
            format.id()
        );
    }
}

#[test]
fn foreign_opaque_drops_with_warning_native_survives() {
    let opaque_value = serde_json::json!({"type": "document", "source": {"data": "x"}});
    let mut req = Request::with_messages(vec![Message::user(vec![
        ContentBlock::text("hi"),
        ContentBlock::opaque(ids::ANTHROPIC_MESSAGES, opaque_value.clone()),
    ])]);
    req.max_output_tokens = Some(64);

    for format in formats() {
        let (body, codes) = body_of(format.as_ref(), &req, &ctx());
        if format.id() == ids::ANTHROPIC_MESSAGES {
            assert!(!codes.contains(&WarningCode::OpaqueDropped));
            assert!(
                body.to_string().contains("\"document\""),
                "native opaque must serialize verbatim"
            );
        } else {
            assert!(
                codes.contains(&WarningCode::OpaqueDropped),
                "{} must warn when dropping a foreign opaque block",
                format.id()
            );
        }
    }
}

#[test]
fn foreign_opaque_fails_under_strict_everywhere_but_home() {
    let mut req = Request::with_messages(vec![Message::user(vec![
        ContentBlock::text("hi"),
        ContentBlock::opaque(
            ids::ANTHROPIC_MESSAGES,
            serde_json::json!({"type": "document"}),
        ),
    ])]);
    req.max_output_tokens = Some(64);

    for format in formats() {
        let result = format.build_request(&req, &strict_ctx());
        if format.id() == ids::ANTHROPIC_MESSAGES {
            assert!(
                result.is_ok(),
                "native target must accept its own opaque node"
            );
        } else {
            match result {
                Err(Error::Conversion(ConversionError::Strict { .. })) => {}
                other => panic!(
                    "{}: strict must escalate the semantic OpaqueDropped warning, got {other:?}",
                    format.id()
                ),
            }
        }
    }
}

#[test]
fn effort_other_passes_verbatim_everywhere() {
    let mut req = Request::with_messages(vec![Message::user_text("hi")]);
    req.max_output_tokens = Some(64);
    req.reasoning = Some(Reasoning::effort(Effort::Other("weird-tier".to_owned())));

    let paths = [
        (ids::OPENAI_CHAT_COMPLETIONS, "/reasoning_effort"),
        (ids::OPENAI_RESPONSES, "/reasoning/effort"),
        (ids::ANTHROPIC_MESSAGES, "/output_config/effort"),
        (
            ids::GOOGLE_GENERATE_CONTENT,
            "/generationConfig/thinkingConfig/thinkingLevel",
        ),
    ];
    for format in formats() {
        let (body, codes) = body_of(format.as_ref(), &req, &ctx());
        let path = paths.iter().find(|(id, _)| *id == format.id()).unwrap().1;
        assert_eq!(
            body.pointer(path).and_then(Value::as_str),
            Some("weird-tier"),
            "{}: Effort::Other must pass through verbatim at {path}",
            format.id()
        );
        assert!(
            !codes.contains(&WarningCode::EffortUnsupported),
            "{}: Effort::Other must not warn",
            format.id()
        );
    }
}

#[test]
fn tool_choice_required_mapping_table() {
    let mut req = tool_loop_request(None);
    req.tool_choice = Some(ToolChoice::Required);

    for format in formats() {
        let (body, _) = body_of(format.as_ref(), &req, &ctx());
        match format.id() {
            ids::OPENAI_CHAT_COMPLETIONS | ids::OPENAI_RESPONSES => {
                assert_eq!(
                    body.pointer("/tool_choice").and_then(Value::as_str),
                    Some("required")
                );
            }
            ids::ANTHROPIC_MESSAGES => {
                assert_eq!(
                    body.pointer("/tool_choice/type").and_then(Value::as_str),
                    Some("any")
                );
            }
            ids::GOOGLE_GENERATE_CONTENT => {
                assert_eq!(
                    body.pointer("/toolConfig/functionCallingConfig/mode")
                        .and_then(Value::as_str),
                    Some("ANY")
                );
            }
            other => panic!("unexpected format {other}"),
        }
    }
}

#[test]
fn dynamic_round_trip_is_idempotent_on_every_format() {
    // A rich request touching system, sampling, tools and a tool loop.
    let mut req = tool_loop_request(None);
    req.system = Some(vec![ContentBlock::text("Be terse.")]);
    req.temperature = Some(0.5);
    req.tool_choice = Some(ToolChoice::Auto);

    for format in formats() {
        let first = format.build_request(&req, &ctx()).unwrap();
        let (reparsed, _) = format.parse_request(&first.body).unwrap();
        let second = format.build_request(&reparsed, &ctx()).unwrap();
        let a: Value = serde_json::from_slice(&first.body).unwrap();
        let b: Value = serde_json::from_slice(&second.body).unwrap();
        assert_eq!(
            a,
            b,
            "{}: build→parse→build must be a fixed point",
            format.id()
        );
    }
}

#[test]
fn on_message_hook_sees_serialized_sequence_on_every_format() {
    use std::sync::{Arc, Mutex};

    let req = tool_loop_request(None);
    for format in formats() {
        let seen: Arc<Mutex<Vec<(usize, Role)>>> = Arc::new(Mutex::new(Vec::new()));
        let seen_in_hook = Arc::clone(&seen);
        let mut ctx = ctx();
        ctx.hooks = llm_api::RequestHooks::new().with_on_message(move |index, role, _value| {
            seen_in_hook.lock().unwrap().push((index, *role));
            Ok(())
        });
        format.build_request(&req, &ctx).unwrap();
        let seen = seen.lock().unwrap();
        assert!(!seen.is_empty(), "{}: hook must run", format.id());
        // Indices are the serialized sequence: 0..n in order.
        for (expect, (index, _)) in seen.iter().enumerate() {
            assert_eq!(
                *index,
                expect,
                "{}: indices must be sequential",
                format.id()
            );
        }
        // The first serialized message is the user turn on every format.
        assert_eq!(seen[0].1, Role::User, "{}", format.id());
    }
}
