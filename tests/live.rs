//! Live tests against real provider endpoints (design § 15).
//!
//! All tests are `#[ignore]` and never run in CI by default. Run them
//! explicitly, e.g.:
//!
//! ```sh
//! cargo test --test live -- --ignored deepseek
//! ```
//!
//! Keys and model ids are read from the process environment first, then
//! from a `.env` file at the crate root (`KEY=VALUE` lines, `#` comments,
//! optional quotes; the file is gitignored). A test silently passes without
//! its key. Model defaults can be overridden with
//! `LLM_API_LIVE_{OPENAI,ANTHROPIC,GOOGLE,DEEPSEEK}_MODEL`.
//!
//! **Anthropic key resolution differs**: development environments such as
//! Claude Code occupy `ANTHROPIC_API_KEY` in the process environment with
//! their own credential, so the tests read `LLM_API_ANTHROPIC_API_KEY`
//! (process env or `.env`) first and fall back to `ANTHROPIC_API_KEY`
//! **from `.env` only** — the process-environment value is never used.
//!
//! Coverage per format:
//! - multi-turn conversation matrix (`streaming × thinking`), asserting the
//!   model actually received the first turn's context — with history (incl.
//!   thinking signatures / item ids riding block `extra`) replayed verbatim;
//! - a multi-round tool-call loop (`streaming × thinking`, and the same in
//!   non-thinking mode to answer "can non-thinking models call tools?");
//! - structured output (JSON Schema), streaming and non-streaming;
//! - image input (OpenAI CC / Responses, Anthropic, Google — DeepSeek
//!   models are text-only);
//! - model listing, and token counting where the provider has an endpoint.
//!
//! The DeepSeek tests exercise the dialect paths of three formats (per
//! <https://api-docs.deepseek.com/guides/>): Chat Completions, Anthropic
//! Messages (`/anthropic` base) and the Responses API.
//!
//! Thinking assertions accept either visible thinking content (a
//! `Thinking` block) or `usage.reasoning_tokens > 0` — OpenAI's own Chat
//! Completions never returns thinking content; non-thinking assertions
//! require the absence of `Thinking` blocks.

#![cfg(feature = "reqwest")]

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use llm_api::formats::anthropic_messages::AnthropicMessages;
use llm_api::formats::google_generate_content::GoogleGenerateContent;
use llm_api::formats::openai_chat_completions::OpenAiChatCompletions;
use llm_api::formats::openai_responses::OpenAiResponses;
use llm_api::{
    ApiFormat, CallOptions, Client, ContentBlock, Effort, FunctionTool, Message, OutputFormat,
    ProviderConfig, Reasoning, Request, Response, Tool, http::ApiKey, ids,
};
use serde_json::{Value, json};

// ---- configuration ----

/// Parses the crate-root `.env` once: `KEY=VALUE` lines, `#` comments and
/// blank lines skipped, an `export ` prefix tolerated, matching single or
/// double quotes stripped. Values never enter the process environment (no
/// unsafe `set_var`, no cross-thread races).
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

fn dotenv_value(name: &str) -> Option<String> {
    dotenv().get(name).cloned().filter(|v| !v.is_empty())
}

/// Reads a configuration value: process environment first, then `.env`.
fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty()).or_else(|| dotenv_value(name))
}

/// The Anthropic key, dodging the `ANTHROPIC_API_KEY` the surrounding
/// development environment may own (see the module docs).
fn anthropic_key() -> Option<String> {
    env("LLM_API_ANTHROPIC_API_KEY").or_else(|| dotenv_value("ANTHROPIC_API_KEY"))
}

fn provider(
    format: impl ApiFormat + 'static,
    base: &str,
    key: String,
    model_env: &str,
    default_model: &str,
) -> ProviderConfig {
    let model = env(model_env).unwrap_or_else(|| default_model.to_owned());
    ProviderConfig::new(Arc::new(format), base, &model)
        .expect("valid base URL")
        .with_auth(ApiKey::new(key))
}

fn openai(format: impl ApiFormat + 'static) -> Option<ProviderConfig> {
    Some(provider(
        format,
        "https://api.openai.com/v1",
        env("OPENAI_API_KEY")?,
        "LLM_API_LIVE_OPENAI_MODEL",
        "gpt-5.6-sol",
    ))
}

fn anthropic() -> Option<ProviderConfig> {
    Some(provider(
        AnthropicMessages,
        "https://api.anthropic.com/v1",
        anthropic_key()?,
        "LLM_API_LIVE_ANTHROPIC_MODEL",
        "claude-opus-5",
    ))
}

fn google() -> Option<ProviderConfig> {
    Some(provider(
        GoogleGenerateContent,
        "https://generativelanguage.googleapis.com/v1beta",
        env("GOOGLE_API_KEY")?,
        "LLM_API_LIVE_GOOGLE_MODEL",
        "gemini-3.6-flash",
    ))
}

/// DeepSeek serves all its dialects off one key; `base` differs per format
/// (`/anthropic` for the Messages dialect). The Responses dialect currently
/// supports `deepseek-v4-flash` only, so that is the shared default.
fn deepseek(format: impl ApiFormat + 'static, base: &str) -> Option<ProviderConfig> {
    Some(provider(
        format,
        base,
        env("DEEPSEEK_API_KEY")?,
        "LLM_API_LIVE_DEEPSEEK_MODEL",
        "deepseek-v4-flash",
    ))
}

// ---- thinking configurations per provider/dialect ----

type Configure = fn(&mut Request);

fn cfg_openai_thinking(req: &mut Request) {
    req.reasoning = Some(Reasoning::effort(Effort::High));
}

fn cfg_openai_no_thinking(req: &mut Request) {
    req.reasoning = Some(Reasoning::enabled(false));
}

/// Anthropic: `enabled(true)` → `{"type": "adaptive"}` plus
/// `output_config.effort: high` to bias the model toward thinking (the
/// current generation rejects the manual-budget `{"type": "enabled"}`).
fn cfg_anthropic_thinking(req: &mut Request) {
    let mut reasoning = Reasoning::enabled(true);
    reasoning.effort = Some(Effort::High);
    req.reasoning = Some(reasoning);
}

fn cfg_anthropic_no_thinking(req: &mut Request) {
    req.reasoning = Some(Reasoning::enabled(false));
}

/// Google: effort → `thinkingLevel`, `includeThoughts` returns summaries
/// (the content channel for the assertion).
fn cfg_google_thinking(req: &mut Request) {
    let mut reasoning = Reasoning::effort(Effort::High);
    reasoning.include_thoughts = Some(true);
    req.reasoning = Some(reasoning);
}

/// Google has no off switch (§ 4.7); `MINIMAL` is the lowest level, and
/// without `includeThoughts` no thinking content is returned.
fn cfg_google_no_thinking(req: &mut Request) {
    req.reasoning = Some(Reasoning::effort(Effort::Minimal));
}

/// DeepSeek CC dialect: the toggle is the dialect's top-level `thinking`
/// object, sent via `Request.extra`; it is on by default upstream, so both
/// modes set it explicitly.
fn cfg_deepseek_cc_thinking(req: &mut Request) {
    req.extra.set(ids::OPENAI_CHAT_COMPLETIONS, "thinking", json!({"type": "enabled"}));
}

fn cfg_deepseek_cc_no_thinking(req: &mut Request) {
    req.extra.set(ids::OPENAI_CHAT_COMPLETIONS, "thinking", json!({"type": "disabled"}));
}

/// DeepSeek Messages dialect: DeepSeek accepts `{"type": "enabled"}`
/// without `budget_tokens` (the field is ignored), so the enable side goes
/// through `extra`; the disable side is the IR-native
/// `Reasoning::enabled(false)` → `{"type": "disabled"}`.
fn cfg_deepseek_messages_thinking(req: &mut Request) {
    req.extra.set(ids::ANTHROPIC_MESSAGES, "thinking", json!({"type": "enabled"}));
}

fn cfg_deepseek_messages_no_thinking(req: &mut Request) {
    req.reasoning = Some(Reasoning::enabled(false));
}

/// DeepSeek Responses dialect: `reasoning.effort` with `"none"` disabling —
/// exactly the IR's § 4.7 encoding, no `extra` needed.
fn cfg_deepseek_responses_thinking(req: &mut Request) {
    req.reasoning = Some(Reasoning::effort(Effort::High));
}

fn cfg_deepseek_responses_no_thinking(req: &mut Request) {
    req.reasoning = Some(Reasoning::enabled(false));
}

// ---- shared plumbing and assertions ----

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

fn assert_thinking_mode(response: &Response, thinking: bool) {
    if thinking {
        assert!(
            has_thinking(response) || reasoning_tokens(response) > 0,
            "expected thinking content or reasoning tokens: {response:?}"
        );
    } else {
        assert!(!has_thinking(response), "unexpected thinking content: {response:?}");
    }
}

/// The last run of ASCII digits in `text` (answers usually end with the
/// number).
fn last_number(text: &str) -> Option<String> {
    let mut runs: Vec<String> = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        if c.is_ascii_digit() {
            current.push(c);
        } else if !current.is_empty() {
            runs.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        runs.push(current);
    }
    runs.pop()
}

/// Two-turn conversation: turn 1 makes the model compute something, turn 2
/// asks for it back with the full history — including thinking blocks,
/// signatures and item ids riding block `extra` — replayed verbatim.
///
/// Turn 2 is compared against **turn 1's own answer**, not the true
/// product: context reception is what is under test, and a small model may
/// well miscalculate — faithfully repeating its earlier (wrong) answer
/// still proves the history arrived.
async fn assert_multi_turn(
    provider: ProviderConfig,
    configure: Configure,
    thinking: bool,
    streaming: bool,
) {
    let mut req = Request::with_messages(vec![Message::user_text(
        "How many prime numbers are there between 100 and 150? Reply with just the count.",
    )]);
    req.max_output_tokens = Some(4096);
    configure(&mut req);

    let first = chat(&provider, &req, streaming).await;
    assert_thinking_mode(&first, thinking);
    let answer = last_number(&first.text())
        .unwrap_or_else(|| panic!("first turn contains no number: {first:?}"));

    req.messages.push(first.message.clone());
    req.messages.push(Message::user_text(
        "What was the count you found earlier? Reply with just the number.",
    ));
    let second = chat(&provider, &req, streaming).await;
    assert!(
        second.text().contains(&answer),
        "the model did not receive the first turn's context (turn 1 answered {answer}): {second:?}"
    );
}

fn weather_tools() -> Vec<Tool> {
    vec![Tool::function(
        FunctionTool::new("get_weather")
            .with_description("Get the current weather of a city.")
            .with_parameters(json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name"}
                },
                "required": ["city"],
                "additionalProperties": false
            })),
    )]
}

/// Answers every tool call of `response` with a per-city weather report
/// (Tokyo is made distinctly warmer so the comparison is checkable).
fn tool_results_for(response: &Response) -> Message {
    let results: Vec<ContentBlock> = response
        .message
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::ToolCall { id, name, arguments, .. } => {
                let report = if arguments.to_lowercase().contains("tokyo") {
                    "Sunny, 25 degrees Celsius."
                } else {
                    "Cloudy, 12 degrees Celsius."
                };
                Some(
                    ContentBlock::tool_result_text(id.clone(), report)
                        .with_tool_name(name.clone()),
                )
            }
            _ => None,
        })
        .collect();
    Message::tool(results)
}

/// Thinking policy of a tool-loop test.
#[derive(Clone, Copy, PartialEq)]
enum Thinking {
    /// Thinking is on and the provider honors it deterministically:
    /// require thinking content or reasoning tokens somewhere in the loop.
    Required,
    /// Thinking is on but model-discretionary (Anthropic `adaptive`): run
    /// the enabled path — whenever the model does think, the replayed
    /// history must be accepted upstream — without requiring that it does.
    /// (Deterministic replay coverage lives in the multi-turn tests, whose
    /// arithmetic task reliably triggers thinking.)
    Enabled,
    /// Thinking is off: no `Thinking` block may appear in any round.
    Off,
}

/// Multi-round tool loop: call → feed results → repeat until the model
/// answers. History (thinking signatures, `thoughtSignature`s, item ids)
/// is replayed verbatim between rounds — the strictest end-to-end check of
/// the signature plumbing (Google and Responses require it with tools).
async fn assert_tool_loop(
    provider: ProviderConfig,
    configure: Configure,
    thinking: Thinking,
    streaming: bool,
) {
    let mut req = Request::with_messages(vec![Message::user_text(
        "Use the get_weather tool to check the weather in Paris and in Tokyo, \
         then tell me which city is warmer right now, in one short sentence.",
    )]);
    req.max_output_tokens = Some(4096);
    req.tools = Some(weather_tools());
    configure(&mut req);

    let mut called = false;
    let mut saw_thinking = false;
    let mut last = chat(&provider, &req, streaming).await;
    for _ in 0..5 {
        saw_thinking = saw_thinking || has_thinking(&last) || reasoning_tokens(&last) > 0;
        if thinking == Thinking::Off {
            assert!(!has_thinking(&last), "unexpected thinking content: {last:?}");
        }
        let has_calls =
            last.message.content.iter().any(|b| matches!(b, ContentBlock::ToolCall { .. }));
        if !has_calls {
            break;
        }
        called = true;
        req.messages.push(last.message.clone());
        req.messages.push(tool_results_for(&last));
        last = chat(&provider, &req, streaming).await;
    }
    assert!(called, "the model never called the tool: {last:?}");
    if thinking == Thinking::Required {
        assert!(
            saw_thinking,
            "no round of the tool loop showed thinking content or reasoning tokens: {last:?}"
        );
    }
    let text = last.text().to_lowercase();
    assert!(
        text.contains("tokyo") || text.contains("25"),
        "final answer does not reflect the tool results: {last:?}"
    );
}

/// JSON-Schema structured output: the reply must parse as JSON and match
/// the requested shape.
async fn assert_structured_output(provider: ProviderConfig, streaming: bool) {
    let mut req = Request::with_messages(vec![Message::user_text(
        "Return the capital of France as JSON with string fields \"city\" and \"country\".",
    )]);
    req.max_output_tokens = Some(4096);
    req.output_format = Some(
        OutputFormat::json_schema(json!({
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "country": {"type": "string"}
            },
            "required": ["city", "country"],
            "additionalProperties": false
        }))
        .with_name("capital_info"),
    );
    let response = chat(&provider, &req, streaming).await;
    let value: Value = response
        .parse_json()
        .unwrap_or_else(|e| panic!("reply is not valid JSON ({e}): {response:?}"));
    assert_eq!(
        value.get("city").and_then(Value::as_str).map(str::to_lowercase).as_deref(),
        Some("paris"),
        "unexpected structured reply: {value}"
    );
    assert!(value.get("country").and_then(Value::as_str).is_some(), "{value}");
}

/// A 64×64 solid-red PNG (base64), small enough to embed.
const RED_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAS0lEQVR42u3PQQkAAAgAsetfWiP4FgYrsKZeS0BAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEDgsqnc8OJg6Ln3AAAAAElFTkSuQmCC";

async fn assert_image_input(provider: ProviderConfig) {
    let mut req = Request::with_messages(vec![Message::user(vec![
        ContentBlock::text("What is the dominant color of this image? Answer with one word."),
        ContentBlock::image_base64("image/png", RED_PNG_BASE64),
    ])]);
    req.max_output_tokens = Some(1024);
    let response = chat(&provider, &req, false).await;
    assert!(
        response.text().to_lowercase().contains("red"),
        "expected the model to see a red image: {response:?}"
    );
}

async fn assert_models_work(provider: ProviderConfig) {
    let client = Client::new(reqwest::Client::new());
    let models = client.list_models(&provider).await.expect("live model list failed");
    assert!(!models.is_empty());
    assert!(models.iter().all(|m| !m.id.is_empty()));
}

async fn assert_count_works(provider: ProviderConfig) {
    let client = Client::new(reqwest::Client::new());
    let mut req = Request::with_messages(vec![Message::user_text("Hello!")]);
    req.max_output_tokens = Some(64);
    let count = client
        .count_tokens(&provider, &req, &CallOptions::default())
        .await
        .expect("live count failed");
    assert!(count.input_tokens > 0);
}

// ---- test matrix ----

/// Declares one `streaming × thinking` matrix cell.
macro_rules! chat_matrix {
    ($name:ident, $provider:expr, $cfg:expr, thinking = $t:literal, stream = $s:literal) => {
        #[tokio::test]
        #[ignore = "live API call; needs the provider key (env or .env)"]
        async fn $name() {
            let Some(p) = $provider else { return };
            assert_multi_turn(p, $cfg, $t, $s).await;
        }
    };
}

macro_rules! tool_matrix {
    ($name:ident, $provider:expr, $cfg:expr, thinking = $t:expr, stream = $s:literal) => {
        #[tokio::test]
        #[ignore = "live API call; needs the provider key (env or .env)"]
        async fn $name() {
            let Some(p) = $provider else { return };
            assert_tool_loop(p, $cfg, $t, $s).await;
        }
    };
}

macro_rules! structured_matrix {
    ($name:ident, $provider:expr, stream = $s:literal) => {
        #[tokio::test]
        #[ignore = "live API call; needs the provider key (env or .env)"]
        async fn $name() {
            let Some(p) = $provider else { return };
            assert_structured_output(p, $s).await;
        }
    };
}

// ---- OpenAI: Chat Completions ----

chat_matrix!(openai_chat_completions_multi_turn_thinking_live, openai(OpenAiChatCompletions), cfg_openai_thinking, thinking = true, stream = false);
chat_matrix!(openai_chat_completions_multi_turn_thinking_stream_live, openai(OpenAiChatCompletions), cfg_openai_thinking, thinking = true, stream = true);
chat_matrix!(openai_chat_completions_multi_turn_no_thinking_live, openai(OpenAiChatCompletions), cfg_openai_no_thinking, thinking = false, stream = false);
chat_matrix!(openai_chat_completions_multi_turn_no_thinking_stream_live, openai(OpenAiChatCompletions), cfg_openai_no_thinking, thinking = false, stream = true);

// OpenAI rejects function tools on `/v1/chat/completions` unless
// `reasoning_effort` is `"none"` for this model class ("use /v1/responses
// or set reasoning_effort to 'none'"), so the CC thinking×tools cell runs
// against the DeepSeek dialect below — where thinking tool loops are not
// only supported but REQUIRE the `reasoning_content` history passback our
// replay performs (a 400 otherwise, per the DeepSeek docs).
tool_matrix!(openai_chat_completions_tools_no_thinking_live, openai(OpenAiChatCompletions), cfg_openai_no_thinking, thinking = Thinking::Off, stream = false);
tool_matrix!(openai_chat_completions_tools_no_thinking_stream_live, openai(OpenAiChatCompletions), cfg_openai_no_thinking, thinking = Thinking::Off, stream = true);

structured_matrix!(openai_chat_completions_structured_output_live, openai(OpenAiChatCompletions), stream = false);
structured_matrix!(openai_chat_completions_structured_output_stream_live, openai(OpenAiChatCompletions), stream = true);

#[tokio::test]
#[ignore = "live API call; needs OPENAI_API_KEY (env or .env)"]
async fn openai_chat_completions_image_input_live() {
    let Some(p) = openai(OpenAiChatCompletions) else { return };
    assert_image_input(p).await;
}

#[tokio::test]
#[ignore = "live API call; needs OPENAI_API_KEY (env or .env)"]
async fn openai_chat_completions_models_live() {
    let Some(p) = openai(OpenAiChatCompletions) else { return };
    assert_models_work(p).await;
}

// ---- OpenAI: Responses ----

chat_matrix!(openai_responses_multi_turn_thinking_live, openai(OpenAiResponses), cfg_openai_thinking, thinking = true, stream = false);
chat_matrix!(openai_responses_multi_turn_thinking_stream_live, openai(OpenAiResponses), cfg_openai_thinking, thinking = true, stream = true);
chat_matrix!(openai_responses_multi_turn_no_thinking_live, openai(OpenAiResponses), cfg_openai_no_thinking, thinking = false, stream = false);
chat_matrix!(openai_responses_multi_turn_no_thinking_stream_live, openai(OpenAiResponses), cfg_openai_no_thinking, thinking = false, stream = true);

tool_matrix!(openai_responses_tools_thinking_live, openai(OpenAiResponses), cfg_openai_thinking, thinking = Thinking::Enabled, stream = false);
tool_matrix!(openai_responses_tools_thinking_stream_live, openai(OpenAiResponses), cfg_openai_thinking, thinking = Thinking::Enabled, stream = true);
tool_matrix!(openai_responses_tools_no_thinking_live, openai(OpenAiResponses), cfg_openai_no_thinking, thinking = Thinking::Off, stream = false);
tool_matrix!(openai_responses_tools_no_thinking_stream_live, openai(OpenAiResponses), cfg_openai_no_thinking, thinking = Thinking::Off, stream = true);

structured_matrix!(openai_responses_structured_output_live, openai(OpenAiResponses), stream = false);
structured_matrix!(openai_responses_structured_output_stream_live, openai(OpenAiResponses), stream = true);

#[tokio::test]
#[ignore = "live API call; needs OPENAI_API_KEY (env or .env)"]
async fn openai_responses_image_input_live() {
    let Some(p) = openai(OpenAiResponses) else { return };
    assert_image_input(p).await;
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

chat_matrix!(anthropic_messages_multi_turn_thinking_live, anthropic(), cfg_anthropic_thinking, thinking = true, stream = false);
chat_matrix!(anthropic_messages_multi_turn_thinking_stream_live, anthropic(), cfg_anthropic_thinking, thinking = true, stream = true);
chat_matrix!(anthropic_messages_multi_turn_no_thinking_live, anthropic(), cfg_anthropic_no_thinking, thinking = false, stream = false);
chat_matrix!(anthropic_messages_multi_turn_no_thinking_stream_live, anthropic(), cfg_anthropic_no_thinking, thinking = false, stream = true);

tool_matrix!(anthropic_messages_tools_thinking_live, anthropic(), cfg_anthropic_thinking, thinking = Thinking::Enabled, stream = false);
tool_matrix!(anthropic_messages_tools_thinking_stream_live, anthropic(), cfg_anthropic_thinking, thinking = Thinking::Enabled, stream = true);
tool_matrix!(anthropic_messages_tools_no_thinking_live, anthropic(), cfg_anthropic_no_thinking, thinking = Thinking::Off, stream = false);
tool_matrix!(anthropic_messages_tools_no_thinking_stream_live, anthropic(), cfg_anthropic_no_thinking, thinking = Thinking::Off, stream = true);

structured_matrix!(anthropic_messages_structured_output_live, anthropic(), stream = false);
structured_matrix!(anthropic_messages_structured_output_stream_live, anthropic(), stream = true);

#[tokio::test]
#[ignore = "live API call; needs LLM_API_ANTHROPIC_API_KEY (or ANTHROPIC_API_KEY in .env)"]
async fn anthropic_messages_image_input_live() {
    let Some(p) = anthropic() else { return };
    assert_image_input(p).await;
}

#[tokio::test]
#[ignore = "live API call; needs LLM_API_ANTHROPIC_API_KEY (or ANTHROPIC_API_KEY in .env)"]
async fn anthropic_models_and_count_live() {
    let Some(p) = anthropic() else { return };
    assert_models_work(p.clone()).await;
    assert_count_works(p).await;
}

// ---- Google ----

chat_matrix!(google_generate_content_multi_turn_thinking_live, google(), cfg_google_thinking, thinking = true, stream = false);
chat_matrix!(google_generate_content_multi_turn_thinking_stream_live, google(), cfg_google_thinking, thinking = true, stream = true);
chat_matrix!(google_generate_content_multi_turn_no_thinking_live, google(), cfg_google_no_thinking, thinking = false, stream = false);
chat_matrix!(google_generate_content_multi_turn_no_thinking_stream_live, google(), cfg_google_no_thinking, thinking = false, stream = true);

tool_matrix!(google_generate_content_tools_thinking_live, google(), cfg_google_thinking, thinking = Thinking::Required, stream = false);
tool_matrix!(google_generate_content_tools_thinking_stream_live, google(), cfg_google_thinking, thinking = Thinking::Required, stream = true);
tool_matrix!(google_generate_content_tools_no_thinking_live, google(), cfg_google_no_thinking, thinking = Thinking::Off, stream = false);
tool_matrix!(google_generate_content_tools_no_thinking_stream_live, google(), cfg_google_no_thinking, thinking = Thinking::Off, stream = true);

structured_matrix!(google_generate_content_structured_output_live, google(), stream = false);
structured_matrix!(google_generate_content_structured_output_stream_live, google(), stream = true);

#[tokio::test]
#[ignore = "live API call; needs GOOGLE_API_KEY (env or .env)"]
async fn google_image_input_live() {
    let Some(p) = google() else { return };
    assert_image_input(p).await;
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

// ---- DeepSeek dialects (text-only models: no image tests) ----

tool_matrix!(deepseek_chat_completions_tools_thinking_live, deepseek(OpenAiChatCompletions, "https://api.deepseek.com"), cfg_deepseek_cc_thinking, thinking = Thinking::Required, stream = false);
tool_matrix!(deepseek_chat_completions_tools_thinking_stream_live, deepseek(OpenAiChatCompletions, "https://api.deepseek.com"), cfg_deepseek_cc_thinking, thinking = Thinking::Required, stream = true);

chat_matrix!(deepseek_chat_completions_multi_turn_thinking_live, deepseek(OpenAiChatCompletions, "https://api.deepseek.com"), cfg_deepseek_cc_thinking, thinking = true, stream = false);
chat_matrix!(deepseek_chat_completions_multi_turn_thinking_stream_live, deepseek(OpenAiChatCompletions, "https://api.deepseek.com"), cfg_deepseek_cc_thinking, thinking = true, stream = true);
chat_matrix!(deepseek_chat_completions_multi_turn_no_thinking_live, deepseek(OpenAiChatCompletions, "https://api.deepseek.com"), cfg_deepseek_cc_no_thinking, thinking = false, stream = false);
chat_matrix!(deepseek_chat_completions_multi_turn_no_thinking_stream_live, deepseek(OpenAiChatCompletions, "https://api.deepseek.com"), cfg_deepseek_cc_no_thinking, thinking = false, stream = true);

chat_matrix!(deepseek_messages_multi_turn_thinking_live, deepseek(AnthropicMessages, "https://api.deepseek.com/anthropic"), cfg_deepseek_messages_thinking, thinking = true, stream = false);
chat_matrix!(deepseek_messages_multi_turn_thinking_stream_live, deepseek(AnthropicMessages, "https://api.deepseek.com/anthropic"), cfg_deepseek_messages_thinking, thinking = true, stream = true);
chat_matrix!(deepseek_messages_multi_turn_no_thinking_live, deepseek(AnthropicMessages, "https://api.deepseek.com/anthropic"), cfg_deepseek_messages_no_thinking, thinking = false, stream = false);
chat_matrix!(deepseek_messages_multi_turn_no_thinking_stream_live, deepseek(AnthropicMessages, "https://api.deepseek.com/anthropic"), cfg_deepseek_messages_no_thinking, thinking = false, stream = true);

chat_matrix!(deepseek_responses_multi_turn_thinking_live, deepseek(OpenAiResponses, "https://api.deepseek.com"), cfg_deepseek_responses_thinking, thinking = true, stream = false);
chat_matrix!(deepseek_responses_multi_turn_thinking_stream_live, deepseek(OpenAiResponses, "https://api.deepseek.com"), cfg_deepseek_responses_thinking, thinking = true, stream = true);
chat_matrix!(deepseek_responses_multi_turn_no_thinking_live, deepseek(OpenAiResponses, "https://api.deepseek.com"), cfg_deepseek_responses_no_thinking, thinking = false, stream = false);
chat_matrix!(deepseek_responses_multi_turn_no_thinking_stream_live, deepseek(OpenAiResponses, "https://api.deepseek.com"), cfg_deepseek_responses_no_thinking, thinking = false, stream = true);
