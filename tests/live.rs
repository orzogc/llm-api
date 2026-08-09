//! Live tests against real provider endpoints (design § 15).
//!
//! All tests are `#[ignore]` and never run in CI by default. Run them
//! explicitly, e.g.:
//!
//! ```sh
//! OPENAI_API_KEY=… cargo test --test live -- --ignored openai
//! ```
//!
//! A test silently passes without its key. Model defaults can be overridden
//! with `LLM_API_LIVE_{OPENAI,ANTHROPIC,GOOGLE,DEEPSEEK}_MODEL`.

#![cfg(feature = "reqwest")]

use std::sync::Arc;

use llm_api::formats::anthropic_messages::AnthropicMessages;
use llm_api::formats::google_generate_content::GoogleGenerateContent;
use llm_api::formats::openai_chat_completions::OpenAiChatCompletions;
use llm_api::formats::openai_responses::OpenAiResponses;
use llm_api::{
    ApiFormat, CallOptions, Client, Message, ProviderConfig, Request, StopReason,
    http::ApiKey,
};

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
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

fn hello_request() -> Request {
    let mut req = Request::with_messages(vec![Message::user_text(
        "Reply with the single word OK and nothing else.",
    )]);
    req.max_output_tokens = Some(128);
    req
}

async fn assert_chat_works(provider: ProviderConfig) {
    let client = Client::new(reqwest::Client::new());
    let response = client
        .send(&provider, &hello_request(), &CallOptions::default())
        .await
        .expect("live chat call failed");
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

#[tokio::test]
#[ignore = "live API call; needs OPENAI_API_KEY"]
async fn openai_chat_completions_live() {
    let Some(p) = provider(
        OpenAiChatCompletions,
        "https://api.openai.com/v1",
        "OPENAI_API_KEY",
        "LLM_API_LIVE_OPENAI_MODEL",
        "gpt-5.6",
    ) else {
        return;
    };
    assert_chat_works(p).await;
}

#[tokio::test]
#[ignore = "live API call; needs OPENAI_API_KEY"]
async fn openai_chat_completions_stream_live() {
    let Some(p) = provider(
        OpenAiChatCompletions,
        "https://api.openai.com/v1",
        "OPENAI_API_KEY",
        "LLM_API_LIVE_OPENAI_MODEL",
        "gpt-5.6",
    ) else {
        return;
    };
    assert_stream_works(p).await;
}

#[tokio::test]
#[ignore = "live API call; needs OPENAI_API_KEY"]
async fn openai_responses_live() {
    let Some(p) = provider(
        OpenAiResponses,
        "https://api.openai.com/v1",
        "OPENAI_API_KEY",
        "LLM_API_LIVE_OPENAI_MODEL",
        "gpt-5.6",
    ) else {
        return;
    };
    assert_chat_works(p).await;
}

#[tokio::test]
#[ignore = "live API call; needs OPENAI_API_KEY"]
async fn openai_responses_stream_live() {
    let Some(p) = provider(
        OpenAiResponses,
        "https://api.openai.com/v1",
        "OPENAI_API_KEY",
        "LLM_API_LIVE_OPENAI_MODEL",
        "gpt-5.6",
    ) else {
        return;
    };
    assert_stream_works(p).await;
}

#[tokio::test]
#[ignore = "live API call; needs OPENAI_API_KEY"]
async fn openai_responses_count_tokens_live() {
    let Some(p) = provider(
        OpenAiResponses,
        "https://api.openai.com/v1",
        "OPENAI_API_KEY",
        "LLM_API_LIVE_OPENAI_MODEL",
        "gpt-5.6",
    ) else {
        return;
    };
    let client = Client::new(reqwest::Client::new());
    let count = client
        .count_tokens(&p, &hello_request(), &CallOptions::default())
        .await
        .expect("live count failed");
    assert!(count.input_tokens > 0);
}

#[tokio::test]
#[ignore = "live API call; needs ANTHROPIC_API_KEY"]
async fn anthropic_messages_live() {
    let Some(p) = provider(
        AnthropicMessages,
        "https://api.anthropic.com/v1",
        "ANTHROPIC_API_KEY",
        "LLM_API_LIVE_ANTHROPIC_MODEL",
        "claude-opus-4-6",
    ) else {
        return;
    };
    assert_chat_works(p).await;
}

#[tokio::test]
#[ignore = "live API call; needs ANTHROPIC_API_KEY"]
async fn anthropic_messages_stream_live() {
    let Some(p) = provider(
        AnthropicMessages,
        "https://api.anthropic.com/v1",
        "ANTHROPIC_API_KEY",
        "LLM_API_LIVE_ANTHROPIC_MODEL",
        "claude-opus-4-6",
    ) else {
        return;
    };
    assert_stream_works(p).await;
}

#[tokio::test]
#[ignore = "live API call; needs ANTHROPIC_API_KEY"]
async fn anthropic_models_and_count_live() {
    let Some(p) = provider(
        AnthropicMessages,
        "https://api.anthropic.com/v1",
        "ANTHROPIC_API_KEY",
        "LLM_API_LIVE_ANTHROPIC_MODEL",
        "claude-opus-4-6",
    ) else {
        return;
    };
    let client = Client::new(reqwest::Client::new());
    let models = client.list_models(&p).await.expect("live model list failed");
    assert!(!models.is_empty());
    let count = client
        .count_tokens(&p, &hello_request(), &CallOptions::default())
        .await
        .expect("live count failed");
    assert!(count.input_tokens > 0);
}

#[tokio::test]
#[ignore = "live API call; needs GOOGLE_API_KEY"]
async fn google_generate_content_live() {
    let Some(p) = provider(
        GoogleGenerateContent,
        "https://generativelanguage.googleapis.com/v1beta",
        "GOOGLE_API_KEY",
        "LLM_API_LIVE_GOOGLE_MODEL",
        "gemini-3.6-flash",
    ) else {
        return;
    };
    assert_chat_works(p).await;
}

#[tokio::test]
#[ignore = "live API call; needs GOOGLE_API_KEY"]
async fn google_generate_content_stream_live() {
    let Some(p) = provider(
        GoogleGenerateContent,
        "https://generativelanguage.googleapis.com/v1beta",
        "GOOGLE_API_KEY",
        "LLM_API_LIVE_GOOGLE_MODEL",
        "gemini-3.6-flash",
    ) else {
        return;
    };
    assert_stream_works(p).await;
}

#[tokio::test]
#[ignore = "live API call; needs DEEPSEEK_API_KEY (exercises the CC dialect path)"]
async fn deepseek_chat_completions_live() {
    let Some(p) = provider(
        OpenAiChatCompletions,
        "https://api.deepseek.com/v1",
        "DEEPSEEK_API_KEY",
        "LLM_API_LIVE_DEEPSEEK_MODEL",
        "deepseek-chat",
    ) else {
        return;
    };
    assert_chat_works(p).await;
}

#[tokio::test]
#[ignore = "live API call; needs DEEPSEEK_API_KEY (reasoning_content dialect)"]
async fn deepseek_reasoner_stream_live() {
    let Some(p) = provider(
        OpenAiChatCompletions,
        "https://api.deepseek.com/v1",
        "DEEPSEEK_API_KEY",
        "LLM_API_LIVE_DEEPSEEK_REASONER_MODEL",
        "deepseek-reasoner",
    ) else {
        return;
    };
    let client = Client::new(reqwest::Client::new());
    let handle = client
        .stream(&p, &hello_request(), &CallOptions::default())
        .await
        .expect("live stream call failed");
    let response = handle.collect().await.expect("stream accumulation failed");
    // The reasoner dialect emits reasoning_content → a Thinking block.
    assert!(!response.message.content.is_empty());
}
