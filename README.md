# llm-api

[简体中文](README_zh-CN.md)

A Rust library providing a **unified intermediate representation (IR)** for
LLM chat APIs, **bidirectional conversion** between the IR and multiple
provider API formats, and a **pluggable HTTP transport**. You build requests
against the IR once and pick the upstream API format at call time.

Built for the calling side of agent applications: conversation history —
including thinking signatures, tool calls and provider-specific data —
survives round-trips faithfully, and everything the IR does not model stays
reachable.

## Why llm-api

- **Free JSON manipulation.** No IR can cover every provider feature, so
  every IR node carries a format-namespaced `extra` map merged into the
  serialized request with JSON Merge Patch semantics (RFC 7396): set,
  override or **delete** any generated field at any depth — whole request, a
  specific message, a specific content block. Request hooks additionally let
  you edit the final JSON in place.
- **Pluggable HTTP client.** The library performs no IO by itself. Any HTTP
  stack plugs in through a small `HttpClient` trait; a `reqwest`-based
  default is feature-gated (on by default). With
  `default-features = false` you get a pure data layer — no tokio, no TLS.
- **Nothing is silently dropped.** Every conversion returns warnings with
  stable codes, fixed severities (`Semantic` vs `Cosmetic`) and JSON-Pointer
  locations. Strict mode turns semantic losses into errors — unless your
  `extra` explicitly overrode the path in question.
- **Faithful round-trips.** Same-provider `format → IR → format` passes are
  canonicalizing then idempotent; unmodeled provider nodes (documents,
  built-in tool calls, executable code, …) ride along as `Opaque` values in
  their original positions. The one silent representational loss:
  explicitly-`null` unknown fields canonicalize to absent. Upstream-
  equivalent re-encodings (string shorthands, entry grouping) are silent;
  every non-equivalent loss carries a warning (`docs/design.md` § 1).
  Persisting agent history as IR JSON is a supported, semver-covered use
  case.

## Supported formats

| Capability | OpenAI Chat Completions¹ | OpenAI Responses | Anthropic Messages | Google `generateContent` |
|---|---|---|---|---|
| Chat (non-streaming + SSE streaming) | ✓ | ✓ | ✓ | ✓ |
| Tool calling | ✓ | ✓ | ✓ | ✓ |
| Image input (URL / base64 / file id) | URL, base64 | ✓ | ✓ | ✓ |
| Thinking incl. replay signatures | plaintext (`reasoning_content`) | ✓ | ✓ | ✓ |
| Structured output | ✓ | ✓ | JSON Schema only | ✓ |
| Model listing | ✓ | ✓ | ✓ (paginated) | ✓ (paginated) |
| Token counting | — (no endpoint) | ✓ | ✓ | ✓ |

¹ A single implementation covering CC dialects such as DeepSeek
(`reasoning_content`). Third-party formats can be added by implementing the
public `ApiFormat` trait.

## Installation

```toml
[dependencies]
llm-api = "0.1"            # includes the reqwest-based default transport
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }

# Pure data layer instead (no IO, no tokio, no TLS):
# llm-api = { version = "0.1", default-features = false }
```

MSRV: 1.88.

## Quick start

```rust
use std::sync::Arc;

use llm_api::formats::anthropic_messages::AnthropicMessages;
use llm_api::{CallOptions, Client, Message, ProviderConfig, Request};

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

    println!("{}", response.text());
    for warning in &response.warnings {
        // Nothing was silently dropped: inspect what the mapping lost.
        eprintln!("[{:?}] {}", warning.severity, warning.message);
    }
    Ok(())
}
```

Switching providers means switching the `ProviderConfig` — the `Request`
stays the same. `response.message` re-enters the next request as history;
thinking signatures, Responses item ids and Google `thoughtSignature`s flow
back through its namespaced `extra` automatically.

## Streaming

```rust
use futures_util::StreamExt;
use llm_api::{BlockDelta, StreamEvent};

let mut stream = client.stream(&provider, &request, &CallOptions::default()).await?;
while let Some(item) = stream.next().await {
    if let StreamEvent::BlockDelta { delta: BlockDelta::Text(fragment), .. } = item?.event {
        print!("{fragment}");
    }
}
```

Or keep the full message for history while rendering deltas — the
accumulator folds the unified block-level events (`MessageStart`,
`BlockStart`/`BlockDelta`/`BlockStop`, `MessageDelta`, `MessageStop`) back
into a `Response`:

```rust
let stream = client.stream(&provider, &request, &CallOptions::default()).await?;
let response = stream.collect().await?;
```

A stream that dies before its protocol terminator is reported as an error —
a silent EOF is never passed off as a complete response.

## Escape hatches

**`extra`** — format-namespaced free-form JSON on every IR node, merged with
RFC 7396 semantics (objects merge recursively, arrays/scalars replace,
`null` deletes):

```rust
use serde_json::json;

// Re-enable a manual thinking budget on Anthropic (unmodeled field):
request.extra.set(
    llm_api::ids::ANTHROPIC_MESSAGES,
    "thinking",
    json!({"type": "enabled", "budget_tokens": 2048}),
);
// Deep-merge into Google's generationConfig; `null` deletes a generated key:
request.extra.set(
    llm_api::ids::GOOGLE_GENERATE_CONTENT,
    "generationConfig",
    json!({"thinkingConfig": {"thinkingLevel": null, "thinkingBudget": 512}}),
);
```

Each namespace only applies when serializing to that format, so provider-
specific data never leaks across providers. Non-null unknown fields parsed
from a provider land in the same namespaces and round-trip verbatim.

**Hooks** — closures over the serialized JSON, run after conversion and the
strict gate, before sending:

```rust
use llm_api::RequestHooks;

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
let opts = CallOptions::default().with_hooks(hooks);
```

## Warnings and strict mode

Conversions never silently drop data: each loss produces a
`ConversionWarning` with a stable `WarningCode`, a fixed severity and a JSON
Pointer into the offending output. `Semantic` means model-visible behavior
could change (a thinking block dropped, an unsupported image source);
`Cosmetic` means tuning was lost (a cache hint, a sampling knob).

```rust
use llm_api::ConvertOptions;

let provider = provider.with_convert(ConvertOptions::default().strict(true));
```

Under strict, any non-overridden semantic warning on the build side fails
the call before IO. Parse-side warnings never fail a call — the response
already happened and was billed — they ride `Response::warnings` /
`StreamItem::warnings` instead.

## The pure conversion layer

Every format is usable without the client (and without IO): build provider
JSON from the IR, or parse provider JSON back into the IR.

```rust
use llm_api::formats::openai_chat_completions::OpenAiChatCompletions;
use llm_api::{ApiFormat, BuildCtx, CallMode, EndpointUrl};

let ctx = BuildCtx::new(
    EndpointUrl::base("https://api.openai.com/v1")?,
    "gpt-5.6",
    CallMode::Unary,
);
let built = OpenAiChatCompletions.build_request(&request, &ctx)?;
// built.url, built.body (bytes), built.headers, built.warnings

// And back: provider JSON -> IR.
let (ir, parse_warnings) = OpenAiChatCompletions.parse_request(&built.body)?;
```

Custom providers plug in the same way: implement the `ApiFormat` (and
`StreamParser`) traits and hand the client an `Arc` of your format.

## Model listing and token counting

```rust
let models = client.list_models(&provider).await?;   // auto-paginates
let count = client.count_tokens(&provider, &request, &CallOptions::default()).await?;
```

Token counts come from provider endpoints only (the library never estimates
locally); Chat Completions has no endpoint and returns `Error::NotSupported`.
Capabilities can be decoupled per endpoint — e.g. chat in Anthropic format
while listing models in OpenAI format — via
`ProviderConfig::with_models_endpoint` / `with_count_tokens_endpoint`.

## Custom HTTP transport

Implement one trait and the whole library runs on your stack:

```rust,ignore
pub trait HttpClient: Send + Sync {
    fn send(
        &self,
        request: http::Request<Bytes>,   // final URL, headers, body — sent as-is
        auth: Option<AuthHeader>,        // injected at send time, never logged
    ) -> Pin<Box<dyn Future<Output = Result<http::Response<BodyStream>, HttpError>> + Send + '_>>;
}
```

The API key is passed separately from the request so it never sits in an
`http::Request` your code might log; the bundled reqwest implementation
marks the injected header sensitive.

## Documentation

- [`docs/design.md`](docs/design.md) — the full v1 design: IR shape,
  per-format mapping rules, streaming model, error model.
- [`docs/impl_contract.md`](docs/impl_contract.md) — binding cross-format
  implementation decisions layered on top of the design.
- Every code snippet in this README is compile-checked by
  [`tests/readme_examples.rs`](tests/readme_examples.rs).

## License

MIT.
