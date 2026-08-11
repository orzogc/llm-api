//! Compile guard for the README installation section, exercised as a real
//! downstream crate: `Cargo.toml` mirrors the documented dependency list,
//! and the snippets below are the README examples that use those
//! dependencies — kept verbatim (modulo `println!` output lines), like
//! `tests/readme_examples.rs`. Driven by `tests/downstream_readme.rs`.

use std::sync::Arc;

use llm_api::formats::anthropic_messages::AnthropicMessages;
use llm_api::{BlockDelta, CallOptions, Client, Message, ProviderConfig, Request, StreamEvent};

// README: Quick start (uses the downstream `reqwest` dependency).
pub async fn quick_start() -> Result<(), Box<dyn std::error::Error>> {
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
    let response = client
        .send(&provider, &request, &CallOptions::default())
        .await?;

    println!("{}", response.text());
    for warning in &response.warnings {
        eprintln!("[{:?}] {}", warning.severity, warning.message);
    }
    Ok(())
}

// README: Streaming (uses the downstream `futures-util` dependency).
pub async fn streaming(
    client: &Client,
    provider: &ProviderConfig,
    request: &Request,
) -> llm_api::Result<()> {
    use futures_util::StreamExt;

    let mut stream = client
        .stream(provider, request, &CallOptions::default())
        .await?;
    while let Some(item) = stream.next().await {
        if let StreamEvent::BlockDelta {
            delta: BlockDelta::Text(fragment),
            ..
        } = item?.event
        {
            print!("{fragment}");
        }
    }
    Ok(())
}

// README: Escape hatches (uses the downstream `serde_json` dependency).
pub fn extra_manipulation(request: &mut Request) {
    use serde_json::json;

    request.extra.set(
        llm_api::ids::ANTHROPIC_MESSAGES,
        "thinking",
        json!({"type": "enabled", "budget_tokens": 2048}),
    );
    request.extra.set(
        llm_api::ids::GOOGLE_GENERATE_CONTENT,
        "generationConfig",
        json!({"thinkingConfig": {"thinkingLevel": null, "thinkingBudget": 512}}),
    );
}
