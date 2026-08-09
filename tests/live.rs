//! Live tests against real provider endpoints (design § 15).
//!
//! All tests are `#[ignore]` and never run in CI by default. Run them
//! explicitly, e.g.:
//!
//! ```sh
//! OPENAI_API_KEY=… cargo test --test live -- --ignored openai
//! ```
//!
//! Keys and model ids are read from the process environment first, then
//! from a `.env` file at the crate root (`KEY=VALUE` lines, `#` comments,
//! optional quotes; the file is gitignored). A test silently passes without
//! its key. Model defaults can be overridden with
//! `LLM_API_LIVE_{OPENAI,ANTHROPIC,GOOGLE,DEEPSEEK}_MODEL`.
//!
//! The DeepSeek tests exercise the dialect paths of all three OpenAI-family
//! and Anthropic formats (per <https://api-docs.deepseek.com/guides/>):
//! Chat Completions, Anthropic Messages (`/anthropic` base) and the
//! Responses API — each in thinking and non-thinking mode.

#![cfg(feature = "reqwest")]

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use llm_api::formats::anthropic_messages::AnthropicMessages;
use llm_api::formats::google_generate_content::GoogleGenerateContent;
use llm_api::formats::openai_chat_completions::OpenAiChatCompletions;
use llm_api::formats::openai_responses::OpenAiResponses;
use llm_api::{
    ApiFormat, CallOptions, Client, ContentBlock, Message, ProviderConfig, Reasoning, Request,
    Response, StopReason, http::ApiKey, ids,
};
use serde_json::json;

/// Parses the crate-root `.env` once: `KEY=VALUE` lines, `#` comments and
/// blank lines skipped, an `export ` prefix tolerated, matching single or
/// double quotes stripped. Values never enter the process environment (no
/// unsafe `set_var`, no cross-thread races) — [`env`] consults this map as
/// the fallback.
fn dotenv() -> &'static HashMap<String, String> {
    static DOTENV: OnceLock<HashMap<String, String>> = OnceLock::new();
    DOTENV.get_or_init(|| {
        let mut map = HashMap::new();
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/.env");
        let Ok(content) = std::fs::read_to_string(path) else {
            return map;
        };
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let line = line.strip_prefix("export ").unwrap_or(line);
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            let mut value = value.trim();
            for quote in ['"', '\''] {
                if value.len() >= 2 && value.starts_with(quote) && value.ends_with(quote) {
                    value = &value[1..value.len() - 1];
                    break;
                }
            }
            map.insert(key.trim().to_owned(), value.to_owned());
        }
        map
    })
}

/// Reads a configuration value: process environment first, then `.env`.
fn env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .filter(|v| !v.is_empty())
        .or_else(|| dotenv().get(name).cloned().filter(|v| !v.is_empty()))
}

fn provider(
    format: impl ApiFormat + 'static,
    base: &str,
    key: &str,
    model_env: &str,
    default_model: &str,
) -> Option<ProviderConfig> {
    let key = env(key)?;
    let model = env(model_env).unwrap_or_else(|| default_model.to_owned());
    Some(
        ProviderConfig::new(Arc::new(format), base, &model)
            .expect("valid base URL")
            .with_auth(ApiKey::new(key)),
    )
}

fn openai(format: impl ApiFormat + 'static) -> Option<ProviderConfig> {
    provider(
        format,
        "https://api.openai.com/v1",
        "OPENAI_API_KEY",
        "LLM_API_LIVE_OPENAI_MODEL",
        "gpt-5.6-sol",
    )
}

fn anthropic() -> Option<ProviderConfig> {
    provider(
        AnthropicMessages,
        "https://api.anthropic.com/v1",
        "ANTHROPIC_API_KEY",
        "LLM_API_LIVE_ANTHROPIC_MODEL",
        "claude-opus-5",
    )
}

fn google() -> Option<ProviderConfig> {
    provider(
        GoogleGenerateContent,
        "https://generativelanguage.googleapis.com/v1beta",
        "GOOGLE_API_KEY",
        "LLM_API_LIVE_GOOGLE_MODEL",
        "gemini-3.6-flash",
    )
}

/// DeepSeek serves all its dialects off one key; `base` differs per format
/// (`/anthropic` for the Messages dialect). The Responses dialect currently
/// supports `deepseek-v4-flash` only, so that is the shared default.
fn deepseek(format: impl ApiFormat + 'static, base: &str) -> Option<ProviderConfig> {
    provider(
        format,
        base,
        "DEEPSEEK_API_KEY",
        "LLM_API_LIVE_DEEPSEEK_MODEL",
        "deepseek-v4-flash",
    )
}

fn hello_request() -> Request {
    let mut req = Request::with_messages(vec![Message::user_text(
        "Reply with the single word OK and nothing else.",
    )]);
    req.max_output_tokens = Some(128);
    req
}

/// A request roomy enough for a chain of thought plus the answer.
fn thinking_request() -> Request {
    let mut req = Request::with_messages(vec![Message::user_text(
        "Which is greater, 9.11 or 9.8? Answer in one short sentence.",
    )]);
    req.max_output_tokens = Some(2048);
    req
}

fn has_thinking(response: &Response) -> bool {
    response
        .message
        .content
        .iter()
        .any(|b| matches!(b, ContentBlock::Thinking { .. }))
}

async fn send(provider: &ProviderConfig, request: &Request) -> Response {
    let client = Client::new(reqwest::Client::new());
    client
        .send(provider, request, &CallOptions::default())
        .await
        .expect("live chat call failed")
}

async fn assert_chat_works(provider: ProviderConfig) {
    let response = send(&provider, &hello_request()).await;
    assert!(!response.text().is_empty(), "empty response text: {response:?}");
    assert!(matches!(
        response.stop_reason,
        Some(StopReason::EndTurn | StopReason::MaxTokens) | None
    ));
    assert!(response.usage.is_some());
}

async fn assert_stream_works(provider: ProviderConfig) {
    let client = Client::new(reqwest::Client::new());
    let handle = client
        .stream(&provider, &hello_request(), &CallOptions::default())
        .await
        .expect("live stream call failed");
    let response = handle.collect().await.expect("stream accumulation failed");
    assert!(!response.text().is_empty(), "empty streamed text: {response:?}");
}

// ---- OpenAI ----

#[tokio::test]
#[ignore = "live API call; needs OPENAI_API_KEY (env or .env)"]
async fn openai_chat_completions_live() {
    let Some(p) = openai(OpenAiChatCompletions) else { return };
    assert_chat_works(p).await;
}

#[tokio::test]
#[ignore = "live API call; needs OPENAI_API_KEY (env or .env)"]
async fn openai_chat_completions_stream_live() {
    let Some(p) = openai(OpenAiChatCompletions) else { return };
    assert_stream_works(p).await;
}

#[tokio::test]
#[ignore = "live API call; needs OPENAI_API_KEY (env or .env)"]
async fn openai_responses_live() {
    let Some(p) = openai(OpenAiResponses) else { return };
    assert_chat_works(p).await;
}

#[tokio::test]
#[ignore = "live API call; needs OPENAI_API_KEY (env or .env)"]
async fn openai_responses_stream_live() {
    let Some(p) = openai(OpenAiResponses) else { return };
    assert_stream_works(p).await;
}

#[tokio::test]
#[ignore = "live API call; needs OPENAI_API_KEY (env or .env)"]
async fn openai_responses_count_tokens_live() {
    let Some(p) = openai(OpenAiResponses) else { return };
    let client = Client::new(reqwest::Client::new());
    let count = client
        .count_tokens(&p, &hello_request(), &CallOptions::default())
        .await
        .expect("live count failed");
    assert!(count.input_tokens > 0);
}

// ---- Anthropic ----

#[tokio::test]
#[ignore = "live API call; needs ANTHROPIC_API_KEY (env or .env)"]
async fn anthropic_messages_live() {
    let Some(p) = anthropic() else { return };
    assert_chat_works(p).await;
}

#[tokio::test]
#[ignore = "live API call; needs ANTHROPIC_API_KEY (env or .env)"]
async fn anthropic_messages_stream_live() {
    let Some(p) = anthropic() else { return };
    assert_stream_works(p).await;
}

#[tokio::test]
#[ignore = "live API call; needs ANTHROPIC_API_KEY (env or .env)"]
async fn anthropic_models_and_count_live() {
    let Some(p) = anthropic() else { return };
    let client = Client::new(reqwest::Client::new());
    let models = client.list_models(&p).await.expect("live model list failed");
    assert!(!models.is_empty());
    let count = client
        .count_tokens(&p, &hello_request(), &CallOptions::default())
        .await
        .expect("live count failed");
    assert!(count.input_tokens > 0);
}

// ---- Google ----

#[tokio::test]
#[ignore = "live API call; needs GOOGLE_API_KEY (env or .env)"]
async fn google_generate_content_live() {
    let Some(p) = google() else { return };
    assert_chat_works(p).await;
}

#[tokio::test]
#[ignore = "live API call; needs GOOGLE_API_KEY (env or .env)"]
async fn google_generate_content_stream_live() {
    let Some(p) = google() else { return };
    assert_stream_works(p).await;
}

// ---- DeepSeek: Chat Completions dialect ----
//
// Thinking is toggled with the dialect's top-level `thinking` object
// (`{"type": "enabled"/"disabled"}`, sent via `Request.extra`); it is on by
// default upstream, so both modes set it explicitly.

#[tokio::test]
#[ignore = "live API call; needs DEEPSEEK_API_KEY (env or .env)"]
async fn deepseek_chat_completions_thinking_live() {
    let Some(p) = deepseek(OpenAiChatCompletions, "https://api.deepseek.com") else { return };
    let mut req = thinking_request();
    req.extra.set(ids::OPENAI_CHAT_COMPLETIONS, "thinking", json!({"type": "enabled"}));
    let response = send(&p, &req).await;
    assert!(has_thinking(&response), "expected reasoning_content: {response:?}");
    assert!(!response.text().is_empty());
}

#[tokio::test]
#[ignore = "live API call; needs DEEPSEEK_API_KEY (env or .env)"]
async fn deepseek_chat_completions_no_thinking_live() {
    let Some(p) = deepseek(OpenAiChatCompletions, "https://api.deepseek.com") else { return };
    let mut req = hello_request();
    req.extra.set(ids::OPENAI_CHAT_COMPLETIONS, "thinking", json!({"type": "disabled"}));
    let response = send(&p, &req).await;
    assert!(!has_thinking(&response), "unexpected thinking: {response:?}");
    assert!(!response.text().is_empty());
}

#[tokio::test]
#[ignore = "live API call; needs DEEPSEEK_API_KEY (env or .env)"]
async fn deepseek_chat_completions_thinking_stream_live() {
    let Some(p) = deepseek(OpenAiChatCompletions, "https://api.deepseek.com") else { return };
    let mut req = thinking_request();
    req.extra.set(ids::OPENAI_CHAT_COMPLETIONS, "thinking", json!({"type": "enabled"}));
    let client = Client::new(reqwest::Client::new());
    let handle = client
        .stream(&p, &req, &CallOptions::default())
        .await
        .expect("live stream call failed");
    let response = handle.collect().await.expect("stream accumulation failed");
    assert!(has_thinking(&response), "expected a streamed Thinking block: {response:?}");
    assert!(!response.text().is_empty());
}

// ---- DeepSeek: Anthropic Messages dialect ----
//
// Base `https://api.deepseek.com/anthropic`; `x-api-key` auth, the
// `anthropic-version` header is ignored upstream. Thinking is toggled with
// the native `thinking` object; DeepSeek accepts `{"type": "enabled"}`
// without `budget_tokens` (the field is ignored), so the enable side goes
// through `extra` and the disable side uses the IR-native
// `Reasoning::enabled(false)` → `{"type": "disabled"}`.

#[tokio::test]
#[ignore = "live API call; needs DEEPSEEK_API_KEY (env or .env)"]
async fn deepseek_messages_thinking_live() {
    let Some(p) = deepseek(AnthropicMessages, "https://api.deepseek.com/anthropic") else {
        return;
    };
    let mut req = thinking_request();
    req.extra.set(ids::ANTHROPIC_MESSAGES, "thinking", json!({"type": "enabled"}));
    let response = send(&p, &req).await;
    assert!(has_thinking(&response), "expected a thinking block: {response:?}");
    assert!(!response.text().is_empty());
}

#[tokio::test]
#[ignore = "live API call; needs DEEPSEEK_API_KEY (env or .env)"]
async fn deepseek_messages_no_thinking_live() {
    let Some(p) = deepseek(AnthropicMessages, "https://api.deepseek.com/anthropic") else {
        return;
    };
    let mut req = hello_request();
    req.reasoning = Some(Reasoning::enabled(false));
    let response = send(&p, &req).await;
    assert!(!has_thinking(&response), "unexpected thinking: {response:?}");
    assert!(!response.text().is_empty());
}

// ---- DeepSeek: Responses dialect ----
//
// Base `https://api.deepseek.com` (the SDK-documented base; the format
// appends `responses`). Thinking maps natively: `reasoning.effort` with
// `"none"` disabling — exactly the IR's § 4.7 encoding, so no `extra` is
// needed in either direction.

#[tokio::test]
#[ignore = "live API call; needs DEEPSEEK_API_KEY (env or .env)"]
async fn deepseek_responses_thinking_live() {
    let Some(p) = deepseek(OpenAiResponses, "https://api.deepseek.com") else { return };
    let mut req = thinking_request();
    req.reasoning = Some(Reasoning::effort(llm_api::Effort::High));
    let response = send(&p, &req).await;
    assert!(has_thinking(&response), "expected a reasoning item: {response:?}");
    assert!(!response.text().is_empty());
}

#[tokio::test]
#[ignore = "live API call; needs DEEPSEEK_API_KEY (env or .env)"]
async fn deepseek_responses_no_thinking_live() {
    let Some(p) = deepseek(OpenAiResponses, "https://api.deepseek.com") else { return };
    let mut req = hello_request();
    req.reasoning = Some(Reasoning::enabled(false));
    let response = send(&p, &req).await;
    assert!(!has_thinking(&response), "unexpected thinking: {response:?}");
    assert!(!response.text().is_empty());
}
