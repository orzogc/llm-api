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
//! Coverage per format: the full `streaming × thinking` chat matrix, model
//! listing, and token counting where the provider has an endpoint. The
//! DeepSeek tests exercise the dialect paths of three formats (per
//! <https://api-docs.deepseek.com/guides/>): Chat Completions, Anthropic
//! Messages (`/anthropic` base) and the Responses API.
//!
//! Thinking assertions accept either visible thinking content (a
//! `Thinking` block) or `usage.reasoning_tokens > 0` — OpenAI's own Chat
//! Completions never returns thinking content, and adaptive modes may
//! summarize invisibly; non-thinking assertions require the absence of
//! `Thinking` blocks.

#![cfg(feature = "reqwest")]

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use llm_api::formats::anthropic_messages::AnthropicMessages;
use llm_api::formats::google_generate_content::GoogleGenerateContent;
use llm_api::formats::openai_chat_completions::OpenAiChatCompletions;
use llm_api::formats::openai_responses::OpenAiResponses;
use llm_api::{
    ApiFormat, CallOptions, Client, ContentBlock, Effort, Message, ProviderConfig, Reasoning,
    Request, Response, http::ApiKey, ids,
};
use serde_json::json;

// ---- configuration ----

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

// ---- requests and assertions ----

fn hello_request() -> Request {
    let mut req = Request::with_messages(vec![Message::user_text(
        "Reply with the single word OK and nothing else.",
    )]);
    req.max_output_tokens = Some(128);
    req
}

/// A request roomy enough for a chain of thought plus the answer, with a
/// question that requires actual multi-step work — adaptive thinking modes
/// skip trivial prompts.
fn thinking_request() -> Request {
    let mut req = Request::with_messages(vec![Message::user_text(
        "How many prime numbers are there between 100 and 150? Reply with just the count.",
    )]);
    req.max_output_tokens = Some(4096);
    req
}

fn has_thinking(response: &Response) -> bool {
    response
        .message
        .content
        .iter()
        .any(|b| matches!(b, ContentBlock::Thinking { .. }))
}

fn reasoning_tokens(response: &Response) -> u64 {
    response.usage.as_ref().and_then(|u| u.reasoning_tokens).unwrap_or(0)
}

async fn chat(provider: &ProviderConfig, request: &Request, streaming: bool) -> Response {
    let client = Client::new(reqwest::Client::new());
    if streaming {
        client
            .stream(provider, request, &CallOptions::default())
            .await
            .expect("live stream call failed")
            .collect()
            .await
            .expect("stream accumulation failed")
    } else {
        client
            .send(provider, request, &CallOptions::default())
            .await
            .expect("live chat call failed")
    }
}

async fn assert_thinking_chat(provider: ProviderConfig, request: Request, streaming: bool) {
    let response = chat(&provider, &request, streaming).await;
    assert!(
        has_thinking(&response) || reasoning_tokens(&response) > 0,
        "expected thinking content or reasoning tokens: {response:?}"
    );
    assert!(!response.text().is_empty(), "empty response text: {response:?}");
}

async fn assert_no_thinking_chat(provider: ProviderConfig, request: Request, streaming: bool) {
    let response = chat(&provider, &request, streaming).await;
    assert!(!has_thinking(&response), "unexpected thinking content: {response:?}");
    assert!(!response.text().is_empty(), "empty response text: {response:?}");
    assert!(response.usage.is_some(), "missing usage: {response:?}");
}

async fn assert_models_work(provider: ProviderConfig) {
    let client = Client::new(reqwest::Client::new());
    let models = client.list_models(&provider).await.expect("live model list failed");
    assert!(!models.is_empty());
    assert!(models.iter().all(|m| !m.id.is_empty()));
}

async fn assert_count_works(provider: ProviderConfig) {
    let client = Client::new(reqwest::Client::new());
    let count = client
        .count_tokens(&provider, &hello_request(), &CallOptions::default())
        .await
        .expect("live count failed");
    assert!(count.input_tokens > 0);
}

/// OpenAI-family thinking switch: the IR `reasoning` field maps natively
/// (`reasoning_effort` on CC, `reasoning.effort` on Responses).
fn openai_thinking() -> Request {
    let mut req = thinking_request();
    req.reasoning = Some(Reasoning::effort(Effort::High));
    req
}

fn openai_no_thinking() -> Request {
    let mut req = hello_request();
    req.reasoning = Some(Reasoning::enabled(false));
    req
}

/// Anthropic thinking: `enabled(true)` → `{"type": "adaptive"}` plus
/// `output_config.effort: high` to bias the model toward thinking (the
/// current generation rejects the manual-budget `{"type": "enabled"}` —
/// the upstream error says to use adaptive + effort, confirming § 4.7).
fn anthropic_thinking() -> Request {
    let mut req = thinking_request();
    let mut reasoning = Reasoning::enabled(true);
    reasoning.effort = Some(Effort::High);
    req.reasoning = Some(reasoning);
    req
}

fn anthropic_no_thinking() -> Request {
    let mut req = hello_request();
    req.reasoning = Some(Reasoning::enabled(false));
    req
}

/// Google: effort → `thinkingLevel`, `includeThoughts` returns summaries
/// (the content channel for the assertion).
fn google_thinking() -> Request {
    let mut req = thinking_request();
    let mut reasoning = Reasoning::effort(Effort::High);
    reasoning.include_thoughts = Some(true);
    req.reasoning = Some(reasoning);
    req
}

/// Google has no off switch (§ 4.7); `MINIMAL` is the lowest level, and
/// without `includeThoughts` no thinking content is returned.
fn google_no_thinking() -> Request {
    let mut req = hello_request();
    req.reasoning = Some(Reasoning::effort(Effort::Minimal));
    req
}

/// DeepSeek CC dialect: the toggle is the dialect's top-level `thinking`
/// object, sent via `Request.extra`; it is on by default upstream, so both
/// modes set it explicitly.
fn deepseek_cc_thinking() -> Request {
    let mut req = thinking_request();
    req.extra.set(ids::OPENAI_CHAT_COMPLETIONS, "thinking", json!({"type": "enabled"}));
    req
}

fn deepseek_cc_no_thinking() -> Request {
    let mut req = hello_request();
    req.extra.set(ids::OPENAI_CHAT_COMPLETIONS, "thinking", json!({"type": "disabled"}));
    req
}

/// DeepSeek Messages dialect: DeepSeek accepts `{"type": "enabled"}`
/// without `budget_tokens` (the field is ignored), so the enable side goes
/// through `extra`; the disable side is the IR-native
/// `Reasoning::enabled(false)` → `{"type": "disabled"}`.
fn deepseek_messages_thinking() -> Request {
    let mut req = thinking_request();
    req.extra.set(ids::ANTHROPIC_MESSAGES, "thinking", json!({"type": "enabled"}));
    req
}

fn deepseek_messages_no_thinking() -> Request {
    let mut req = hello_request();
    req.reasoning = Some(Reasoning::enabled(false));
    req
}

/// DeepSeek Responses dialect: `reasoning.effort` with `"none"` disabling —
/// exactly the IR's § 4.7 encoding, no `extra` needed.
fn deepseek_responses_thinking() -> Request {
    let mut req = thinking_request();
    req.reasoning = Some(Reasoning::effort(Effort::High));
    req
}

fn deepseek_responses_no_thinking() -> Request {
    let mut req = hello_request();
    req.reasoning = Some(Reasoning::enabled(false));
    req
}

// ---- OpenAI: Chat Completions ----

#[tokio::test]
#[ignore = "live API call; needs OPENAI_API_KEY (env or .env)"]
async fn openai_chat_completions_thinking_live() {
    let Some(p) = openai(OpenAiChatCompletions) else { return };
    assert_thinking_chat(p, openai_thinking(), false).await;
}

#[tokio::test]
#[ignore = "live API call; needs OPENAI_API_KEY (env or .env)"]
async fn openai_chat_completions_thinking_stream_live() {
    let Some(p) = openai(OpenAiChatCompletions) else { return };
    assert_thinking_chat(p, openai_thinking(), true).await;
}

#[tokio::test]
#[ignore = "live API call; needs OPENAI_API_KEY (env or .env)"]
async fn openai_chat_completions_no_thinking_live() {
    let Some(p) = openai(OpenAiChatCompletions) else { return };
    assert_no_thinking_chat(p, openai_no_thinking(), false).await;
}

#[tokio::test]
#[ignore = "live API call; needs OPENAI_API_KEY (env or .env)"]
async fn openai_chat_completions_no_thinking_stream_live() {
    let Some(p) = openai(OpenAiChatCompletions) else { return };
    assert_no_thinking_chat(p, openai_no_thinking(), true).await;
}

#[tokio::test]
#[ignore = "live API call; needs OPENAI_API_KEY (env or .env)"]
async fn openai_chat_completions_models_live() {
    let Some(p) = openai(OpenAiChatCompletions) else { return };
    assert_models_work(p).await;
}

// ---- OpenAI: Responses ----

#[tokio::test]
#[ignore = "live API call; needs OPENAI_API_KEY (env or .env)"]
async fn openai_responses_thinking_live() {
    let Some(p) = openai(OpenAiResponses) else { return };
    assert_thinking_chat(p, openai_thinking(), false).await;
}

#[tokio::test]
#[ignore = "live API call; needs OPENAI_API_KEY (env or .env)"]
async fn openai_responses_thinking_stream_live() {
    let Some(p) = openai(OpenAiResponses) else { return };
    assert_thinking_chat(p, openai_thinking(), true).await;
}

#[tokio::test]
#[ignore = "live API call; needs OPENAI_API_KEY (env or .env)"]
async fn openai_responses_no_thinking_live() {
    let Some(p) = openai(OpenAiResponses) else { return };
    assert_no_thinking_chat(p, openai_no_thinking(), false).await;
}

#[tokio::test]
#[ignore = "live API call; needs OPENAI_API_KEY (env or .env)"]
async fn openai_responses_no_thinking_stream_live() {
    let Some(p) = openai(OpenAiResponses) else { return };
    assert_no_thinking_chat(p, openai_no_thinking(), true).await;
}

#[tokio::test]
#[ignore = "live API call; needs OPENAI_API_KEY (env or .env)"]
async fn openai_responses_models_live() {
    let Some(p) = openai(OpenAiResponses) else { return };
    assert_models_work(p).await;
}

#[tokio::test]
#[ignore = "live API call; needs OPENAI_API_KEY (env or .env)"]
async fn openai_responses_count_tokens_live() {
    let Some(p) = openai(OpenAiResponses) else { return };
    assert_count_works(p).await;
}

// ---- Anthropic ----

#[tokio::test]
#[ignore = "live API call; needs ANTHROPIC_API_KEY (env or .env)"]
async fn anthropic_messages_thinking_live() {
    let Some(p) = anthropic() else { return };
    assert_thinking_chat(p, anthropic_thinking(), false).await;
}

#[tokio::test]
#[ignore = "live API call; needs ANTHROPIC_API_KEY (env or .env)"]
async fn anthropic_messages_thinking_stream_live() {
    let Some(p) = anthropic() else { return };
    assert_thinking_chat(p, anthropic_thinking(), true).await;
}

#[tokio::test]
#[ignore = "live API call; needs ANTHROPIC_API_KEY (env or .env)"]
async fn anthropic_messages_no_thinking_live() {
    let Some(p) = anthropic() else { return };
    assert_no_thinking_chat(p, anthropic_no_thinking(), false).await;
}

#[tokio::test]
#[ignore = "live API call; needs ANTHROPIC_API_KEY (env or .env)"]
async fn anthropic_messages_no_thinking_stream_live() {
    let Some(p) = anthropic() else { return };
    assert_no_thinking_chat(p, anthropic_no_thinking(), true).await;
}

#[tokio::test]
#[ignore = "live API call; needs ANTHROPIC_API_KEY (env or .env)"]
async fn anthropic_models_and_count_live() {
    let Some(p) = anthropic() else { return };
    assert_models_work(p.clone()).await;
    assert_count_works(p).await;
}

// ---- Google ----

#[tokio::test]
#[ignore = "live API call; needs GOOGLE_API_KEY (env or .env)"]
async fn google_generate_content_thinking_live() {
    let Some(p) = google() else { return };
    assert_thinking_chat(p, google_thinking(), false).await;
}

#[tokio::test]
#[ignore = "live API call; needs GOOGLE_API_KEY (env or .env)"]
async fn google_generate_content_thinking_stream_live() {
    let Some(p) = google() else { return };
    assert_thinking_chat(p, google_thinking(), true).await;
}

#[tokio::test]
#[ignore = "live API call; needs GOOGLE_API_KEY (env or .env)"]
async fn google_generate_content_no_thinking_live() {
    let Some(p) = google() else { return };
    assert_no_thinking_chat(p, google_no_thinking(), false).await;
}

#[tokio::test]
#[ignore = "live API call; needs GOOGLE_API_KEY (env or .env)"]
async fn google_generate_content_no_thinking_stream_live() {
    let Some(p) = google() else { return };
    assert_no_thinking_chat(p, google_no_thinking(), true).await;
}

#[tokio::test]
#[ignore = "live API call; needs GOOGLE_API_KEY (env or .env)"]
async fn google_models_live() {
    let Some(p) = google() else { return };
    assert_models_work(p).await;
}

#[tokio::test]
#[ignore = "live API call; needs GOOGLE_API_KEY (env or .env)"]
async fn google_count_tokens_live() {
    let Some(p) = google() else { return };
    assert_count_works(p).await;
}

// ---- DeepSeek: Chat Completions dialect ----

#[tokio::test]
#[ignore = "live API call; needs DEEPSEEK_API_KEY (env or .env)"]
async fn deepseek_chat_completions_thinking_live() {
    let Some(p) = deepseek(OpenAiChatCompletions, "https://api.deepseek.com") else { return };
    assert_thinking_chat(p, deepseek_cc_thinking(), false).await;
}

#[tokio::test]
#[ignore = "live API call; needs DEEPSEEK_API_KEY (env or .env)"]
async fn deepseek_chat_completions_thinking_stream_live() {
    let Some(p) = deepseek(OpenAiChatCompletions, "https://api.deepseek.com") else { return };
    assert_thinking_chat(p, deepseek_cc_thinking(), true).await;
}

#[tokio::test]
#[ignore = "live API call; needs DEEPSEEK_API_KEY (env or .env)"]
async fn deepseek_chat_completions_no_thinking_live() {
    let Some(p) = deepseek(OpenAiChatCompletions, "https://api.deepseek.com") else { return };
    assert_no_thinking_chat(p, deepseek_cc_no_thinking(), false).await;
}

#[tokio::test]
#[ignore = "live API call; needs DEEPSEEK_API_KEY (env or .env)"]
async fn deepseek_chat_completions_no_thinking_stream_live() {
    let Some(p) = deepseek(OpenAiChatCompletions, "https://api.deepseek.com") else { return };
    assert_no_thinking_chat(p, deepseek_cc_no_thinking(), true).await;
}

// ---- DeepSeek: Anthropic Messages dialect ----

#[tokio::test]
#[ignore = "live API call; needs DEEPSEEK_API_KEY (env or .env)"]
async fn deepseek_messages_thinking_live() {
    let Some(p) = deepseek(AnthropicMessages, "https://api.deepseek.com/anthropic") else {
        return;
    };
    assert_thinking_chat(p, deepseek_messages_thinking(), false).await;
}

#[tokio::test]
#[ignore = "live API call; needs DEEPSEEK_API_KEY (env or .env)"]
async fn deepseek_messages_thinking_stream_live() {
    let Some(p) = deepseek(AnthropicMessages, "https://api.deepseek.com/anthropic") else {
        return;
    };
    assert_thinking_chat(p, deepseek_messages_thinking(), true).await;
}

#[tokio::test]
#[ignore = "live API call; needs DEEPSEEK_API_KEY (env or .env)"]
async fn deepseek_messages_no_thinking_live() {
    let Some(p) = deepseek(AnthropicMessages, "https://api.deepseek.com/anthropic") else {
        return;
    };
    assert_no_thinking_chat(p, deepseek_messages_no_thinking(), false).await;
}

#[tokio::test]
#[ignore = "live API call; needs DEEPSEEK_API_KEY (env or .env)"]
async fn deepseek_messages_no_thinking_stream_live() {
    let Some(p) = deepseek(AnthropicMessages, "https://api.deepseek.com/anthropic") else {
        return;
    };
    assert_no_thinking_chat(p, deepseek_messages_no_thinking(), true).await;
}

// ---- DeepSeek: Responses dialect ----

#[tokio::test]
#[ignore = "live API call; needs DEEPSEEK_API_KEY (env or .env)"]
async fn deepseek_responses_thinking_live() {
    let Some(p) = deepseek(OpenAiResponses, "https://api.deepseek.com") else { return };
    assert_thinking_chat(p, deepseek_responses_thinking(), false).await;
}

#[tokio::test]
#[ignore = "live API call; needs DEEPSEEK_API_KEY (env or .env)"]
async fn deepseek_responses_thinking_stream_live() {
    let Some(p) = deepseek(OpenAiResponses, "https://api.deepseek.com") else { return };
    assert_thinking_chat(p, deepseek_responses_thinking(), true).await;
}

#[tokio::test]
#[ignore = "live API call; needs DEEPSEEK_API_KEY (env or .env)"]
async fn deepseek_responses_no_thinking_live() {
    let Some(p) = deepseek(OpenAiResponses, "https://api.deepseek.com") else { return };
    assert_no_thinking_chat(p, deepseek_responses_no_thinking(), false).await;
}

#[tokio::test]
#[ignore = "live API call; needs DEEPSEEK_API_KEY (env or .env)"]
async fn deepseek_responses_no_thinking_stream_live() {
    let Some(p) = deepseek(OpenAiResponses, "https://api.deepseek.com") else { return };
    assert_no_thinking_chat(p, deepseek_responses_no_thinking(), true).await;
}
