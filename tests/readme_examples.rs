//! Compile guard for the code snippets embedded in `README.md` /
//! `README_zh-CN.md`: every snippet lives here verbatim (modulo `println!`
//! output lines) so documentation examples cannot silently rot. Nothing is
//! executed against the network.

#![cfg(feature = "reqwest")]

use std::sync::Arc;

use futures_util::StreamExt;
use llm_api::formats::anthropic_messages::AnthropicMessages;
use llm_api::formats::openai_chat_completions::OpenAiChatCompletions;
use llm_api::{
    ApiFormat, BlockDelta, BuildCtx, CallMode, CallOptions, Client, ConvertOptions, EndpointUrl,
    Message, ProviderConfig, Request, RequestHooks, StreamEvent,
};
use serde_json::json;

// README: Quick start.
async fn quick_start() -> Result<(), Box<dyn std::error::Error>> {
    let provider = ProviderConfig::new(
        Arc::new(AnthropicMessages),
        "https://api.anthropic.com/v1",
        "claude-opus-4-6",
    )?
    .with_auth(std::env::var("ANTHROPIC_API_KEY")?);

    let mut request = Request::with_messages(vec![Message::user_text(
        "Which mountain is the tallest on Earth?",
    )])
    .with_system_text("Answer in one sentence.");
    request.max_output_tokens = Some(1024);

    let client = Client::new(reqwest::Client::new());
    let response = client.send(&provider, &request, &CallOptions::default()).await?;

    let _text = response.text();
    for _warning in &response.warnings {
        // Nothing was silently dropped: inspect what the mapping lost.
    }
    Ok(())
}

// README: Streaming.
async fn streaming(
    client: &Client,
    provider: &ProviderConfig,
    request: &Request,
) -> llm_api::Result<()> {
    let mut stream = client.stream(provider, request, &CallOptions::default()).await?;
    while let Some(item) = stream.next().await {
        if let StreamEvent::BlockDelta { delta: BlockDelta::Text(_fragment), .. } = item?.event {
            // print!("{_fragment}");
        }
    }
    Ok(())
}

// README: Streaming, accumulated.
async fn streaming_accumulated(
    client: &Client,
    provider: &ProviderConfig,
    request: &Request,
) -> llm_api::Result<()> {
    let stream = client.stream(provider, request, &CallOptions::default()).await?;
    let _response = stream.collect().await?;
    Ok(())
}

// README: Free JSON manipulation via `extra`.
fn extra_manipulation(request: &mut Request) {
    // Re-enable a manual thinking budget on Anthropic (unmodeled field):
    request.extra.set(
        llm_api::ids::ANTHROPIC_MESSAGES,
        "thinking",
        json!({"type": "enabled", "budget_tokens": 2048}),
    );
    // Deep-merge into Google's generationConfig; `null` deletes a
    // generated key (RFC 7396):
    request.extra.set(
        llm_api::ids::GOOGLE_GENERATE_CONTENT,
        "generationConfig",
        json!({"thinkingConfig": {"thinkingLevel": null, "thinkingBudget": 512}}),
    );
}

// README: Hooks.
fn hooks() -> CallOptions {
    let hooks = RequestHooks::new()
        .with_on_message(|index, _role, message| {
            if index == 0 {
                message["cache_control"] = json!({"type": "ephemeral"});
            }
            Ok(())
        })
        .with_on_request(|body| {
            body["service_tier"] = json!("flex");
            Ok(())
        });
    CallOptions::default().with_hooks(hooks)
}

// README: Strict mode.
fn strict(provider: ProviderConfig) -> ProviderConfig {
    provider.with_convert(ConvertOptions::default().strict(true))
}

// README: The pure conversion layer (no client, no IO).
fn pure_conversion(request: &Request) -> llm_api::Result<()> {
    let ctx = BuildCtx::new(
        EndpointUrl::base("https://api.openai.com/v1")?,
        "gpt-5.6",
        CallMode::Unary,
    );
    let built = OpenAiChatCompletions.build_request(request, &ctx)?;
    let _url = &built.url;
    let _body_bytes = &built.body;
    let _warnings = &built.warnings;

    // And back: provider JSON -> IR.
    let (_ir, _parse_warnings) = OpenAiChatCompletions.parse_request(&built.body)?;
    Ok(())
}

// README: Model listing and token counting.
async fn models_and_tokens(
    client: &Client,
    provider: &ProviderConfig,
    request: &Request,
) -> llm_api::Result<()> {
    let _models = client.list_models(provider).await?;
    let _count = client.count_tokens(provider, request, &CallOptions::default()).await?;
    Ok(())
}

#[test]
fn readme_snippets_compile() {
    // Reference every snippet so nothing is dead code; none of them run.
    let _ = quick_start;
    let _ = streaming;
    let _ = streaming_accumulated;
    let _ = extra_manipulation;
    let _ = hooks;
    let _ = strict;
    let _ = pure_conversion;
    let _ = models_and_tokens;
}
