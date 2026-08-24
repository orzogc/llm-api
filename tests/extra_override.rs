//! End-to-end `extra` override registration (§ 6): object paths *created*
//! by the request-level `extra` merge count as exact-pointer overrides, so
//! a user can opt out of a semantic warning by supplying the
//! provider-native fields the converter did not generate — e.g. Google's
//! `thinkingBudget: 0` for `reasoning.enabled = false`.

use llm_api::formats::google_generate_content::{GoogleGenerateContent, request_from_ir};
use llm_api::{
    ApiFormat, BuildCtx, CallMode, ConvertOptions, EndpointUrl, Error,
    GoogleGenerateContentOptions, Message, Reasoning, Request, WarningCode, ids,
};
use serde_json::{Value, json};

/// `reasoning.enabled = false` (no Google channel → semantic warning at
/// `/generationConfig/thinkingConfig`) plus a request-level `extra` that
/// creates exactly that path.
fn request_with_disable_override() -> Request {
    let mut req = Request::with_messages(vec![Message::user_text("hi")]);
    req.reasoning = Some(Reasoning::enabled(false));
    req.extra.set(
        ids::GOOGLE_GENERATE_CONTENT,
        "generationConfig",
        json!({"thinkingConfig": {"thinkingBudget": 0}}),
    );
    req
}

fn strict_ctx() -> BuildCtx {
    let mut ctx = BuildCtx::new(
        EndpointUrl::base("https://generativelanguage.googleapis.com/v1beta").unwrap(),
        "gemini-2.5-pro",
        CallMode::Unary,
    );
    ctx.convert = ConvertOptions::default().strict(true);
    ctx
}

#[test]
fn extra_created_object_path_marks_warning_overridden() {
    // The converter emits no `generationConfig` for a disabled-reasoning
    // request, so the merge creates the whole path from vacant keys.
    let (body, warnings) = request_from_ir(
        &request_with_disable_override(),
        &ConvertOptions::default(),
        &GoogleGenerateContentOptions::default(),
    )
    .unwrap();
    assert_eq!(
        body["generationConfig"]["thinkingConfig"]["thinkingBudget"],
        json!(0)
    );
    let w = warnings
        .iter()
        .find(|w| w.code == WarningCode::ReasoningDisableUnsupported)
        .expect("the disable warning is still reported for debugging");
    assert!(
        w.overridden,
        "extra created {} — the warning must be marked overridden: {w:?}",
        w.location
    );
}

#[test]
fn strict_mode_accepts_extra_override_of_created_path() {
    // Without the override strict escalates the semantic warning…
    let mut bare = Request::with_messages(vec![Message::user_text("hi")]);
    bare.reasoning = Some(Reasoning::enabled(false));
    let err = GoogleGenerateContent
        .build_request(&bare, &strict_ctx())
        .unwrap_err();
    assert!(matches!(err, Error::Conversion(_)));

    // …and the extra-supplied `thinkingBudget` opts out of it.
    let built = GoogleGenerateContent
        .build_request(&request_with_disable_override(), &strict_ctx())
        .unwrap();
    let body: Value = serde_json::from_slice(&built.body).unwrap();
    assert_eq!(
        body["generationConfig"]["thinkingConfig"]["thinkingBudget"],
        json!(0)
    );
}
