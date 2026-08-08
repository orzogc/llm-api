# llm-api Design

Status: v1 design agreed on 2026-08-08. Implementation not started.

`llm-api` is a Rust library providing a unified intermediate representation (IR)
for LLM chat APIs, bidirectional conversion between the IR and multiple provider
API formats, and a pluggable HTTP transport. Users build requests against the IR
and pick the upstream API format at call time.

## 1. Goals and positioning

- Primary use case: the calling side of agent applications (SDK).
- Non-negotiable differentiators:
  - **Free JSON manipulation** — users can set/override/delete arbitrary fields
    of the serialized request, at any level (whole request, a specific message,
    a specific content block), because the IR can never cover every provider
    feature.
  - **Pluggable HTTP client** — the library performs no IO by itself; any HTTP
    stack can be plugged in via a small trait. A `reqwest`-based default is
    feature-gated.
- Format-to-format conversion (e.g. an OpenAI request converted to an Anthropic
  request) is a design constraint (the IR must not preclude it) but is **not
  implemented in v1**, neither streaming nor non-streaming.
- Same-provider round-trips must be lossless: `format -> IR -> format`
  preserves modeled fields, unknown fields, and message order. Nothing is
  silently dropped.

### Non-goals

- Agent loops / automatic tool execution, conversation management.
- Automatic retries or rate limiting (the unified error type exposes enough
  information — status, error kind, `retry_after` — for callers to implement
  their own).
- Local token estimation (token counts only come from provider endpoints).
- Embeddings, image generation, speech, and other non-chat APIs.

## 2. Scope

### v1

| Area | Contents |
|---|---|
| Formats | OpenAI Chat Completions ("CC" below; single implementation covering dialects such as DeepSeek `reasoning_content`), OpenAI Responses, Anthropic Messages, Google `generateContent` |
| Features | non-streaming + streaming (SSE), tool calling, image input (URL / base64 / provider file id), thinking (incl. signatures), structured output, usage, sampling parameters, reasoning effort |
| Endpoints | chat, model listing, token counting |

### Deferred

OpenAI Responses WebSocket transport (a transport for the same format, not a new
format), Google Interactions API (a fifth HTTP format; note its streaming is
SSE, not WebSocket), format-to-format conversion, audio/video/document input,
image generation. The IR reserves room via `#[non_exhaustive]` enums/structs; no
code is written for these in v1.

## 3. Crate layout

- Single crate `llm-api`, edition 2024. License: MIT.
- `default-features = false` yields a pure data layer: IR types, format types,
  conversions — no IO, no tokio, no TLS. Base dependencies: `serde`,
  `serde_json`, `http`, `bytes`, `futures-core`.
- Feature `reqwest`: default `HttpClient` implementation.
- MSRV: decided after implementation with `cargo-msrv`, then declared as
  `rust-version`; policy is to follow a recent stable.
- All public IR types are `#[non_exhaustive]`; enums additionally mark
  struct-like variants `#[non_exhaustive]` (enum-level `#[non_exhaustive]`
  alone does not cover variant fields, which would otherwise freeze a
  variant's shape). Construction goes through builders and constructor
  functions (`ContentBlock::text(...)`, …); matching these variants requires
  `..`. All IR types derive `Serialize`/`Deserialize`/`Clone`/`Debug` (and
  `PartialEq` where possible): persisting agent history as IR JSON is a
  supported use case, which is why closures are kept out of IR nodes (see
  § 5). One exception: `Response.headers` (`http::HeaderMap` has no serde
  support) serializes through a custom `Vec<(String, String)>` representation
  (lossy for non-UTF-8 header values).
- IR JSON representation: `ContentBlock` and `StreamEvent` use
  `#[serde(tag = "type", rename_all = "snake_case")]`; string-like enums
  (`Role`, `Effort`, `StopReason`) serialize as plain strings, with `Other`
  carrying its raw string (custom impls). Optional fields use
  `#[serde(default, skip_serializing_if)]`, and fields added later are always
  optional, so previously persisted IR JSON keeps deserializing. The IR JSON
  representation is covered by semver — an incompatible change is a
  major-version change.

Module sketch:

```
src/
  ir/          request, message, block, params, reasoning, response, usage, events
  convert/     warnings, options, per-format entry points
  formats/
    openai_chat_completions/  types, from_ir, to_ir, stream
    openai_responses/         ...
    anthropic_messages/       ...
    google_generate_content/  ...
  http/        HttpClient trait, AuthHeader, ApiKey, SSE parser
  client/      Client, ProviderConfig, EndpointConfig, hooks
  models.rs    model listing
  tokens.rs    token counting
  error.rs
```

Canonical format ids (used as `extra` namespaces, config keys, and
`ApiFormat::id()`): `openai_chat_completions`, `openai_responses`,
`anthropic_messages`, `google_generate_content`. Third-party formats register their own ids.

## 4. Intermediate representation

### 4.1 Request

```rust
#[non_exhaustive]
pub struct Request {
    pub system: Option<Vec<ContentBlock>>, // Text only; enforced at conversion time (ConversionError)
    pub messages: Vec<Message>,
    // sampling parameters, see § 4.6
    pub max_output_tokens: Option<u32>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub stop_sequences: Option<Vec<String>>,
    pub seed: Option<i64>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub metadata: Option<Map<String, Value>>,
    pub reasoning: Option<Reasoning>,        // § 4.7
    pub tools: Option<Vec<Tool>>,            // § 4.5
    pub tool_choice: Option<ToolChoice>,
    pub parallel_tool_calls: Option<bool>,
    pub output_format: Option<OutputFormat>, // § 4.9
    pub cache_key: Option<String>,           // § 4.8
    pub extra: Extra,                        // § 5
}
```

Whether a call is streaming is decided by the client method (`send` vs
`stream`), not by an IR field; the format layer sets `stream: true` or switches
the URL (`streamGenerateContent?alt=sse`) accordingly.

The model name is a plain `String` supplied via `ProviderConfig`/call options,
not an enum.

### 4.2 Messages and roles

```rust
#[non_exhaustive]
pub struct Message {
    pub role: Role, // System | Developer | User | Assistant | Tool
    pub content: Vec<ContentBlock>,
    pub extra: Extra,
}
```

`Role` is the union of all supported formats. Downgrade rules for formats that
lack a role are defined in § 7.1. Messages are never dropped or reordered by
conversion (outside the explicit opt-in policies of § 7.3); splits/merges
required by a target format are deterministic and
recorded in `extra["_ir"]` so the original shape is restored on the way back.

An assistant message at the end of `messages` (prefill) is passed through
as-is. Some newer models reject prefill (documented for recent Claude models;
newest Gemini models reportedly as well); any such upstream error is returned
to the caller unchanged.

### 4.3 Content blocks

```rust
#[non_exhaustive]
pub enum ContentBlock {
    Text { text: String, cache: Option<CacheHint>, extra: Extra },
    Image { source: ImageSource, cache: Option<CacheHint>, extra: Extra },
    ToolCall { id: String, name: String, arguments: String,
               cache: Option<CacheHint>, extra: Extra },
    ToolResult { tool_call_id: String, name: Option<String>,
                 content: Vec<ContentBlock>, is_error: Option<bool>,
                 cache: Option<CacheHint>, extra: Extra },
    Thinking { text: Option<String>, signature: Option<String>, extra: Extra },
}

#[non_exhaustive]
pub enum ImageSource {
    Url(String),
    Base64 { media_type: String, data: String },
    FileId(String), // provider-specific; cross-provider conversion warns
}
```

Conversion performs **zero IO** — the library never downloads a URL to inline
it. Source mapping:

| `ImageSource` | OpenAI CC | Responses | Anthropic | Google |
|---|---|---|---|---|
| `Url` | `image_url.url` | `input_image.image_url` | `source:{type:"url", url}` | `fileData.fileUri` (verbatim + cosmetic warning: documented URIs are Files API ones; arbitrary URLs may be rejected upstream) |
| `Base64` | `image_url.url` as `data:` URL | `input_image.image_url` as `data:` URL | `source:{type:"base64", media_type, data}` | `inlineData:{mimeType, data}` |
| `FileId` | semantic warning (no channel) | `input_image.file_id` | `source:{type:"file", file_id}` | `fileData.fileUri` |

A file id is only meaningful on the provider that issued it; the library maps
it syntactically and leaves validity to the upstream API. OpenAI's `detail`
and similar per-format image options live in `extra`.

### 4.4 Thinking

`Thinking { text, signature, extra }`:

- `text`: plaintext chain of thought (CC dialects) or joined summary text.
- `signature`: the opaque payload required for replay — Anthropic `signature`
  (also covers `redacted_thinking.data`), OpenAI Responses `encrypted_content`,
  Google `thoughtSignature` (attached to the signed part; carried in the
  block's `extra["google_generate_content"]` when it belongs to a tool call
  part). Original structures that do not fit `text`/`signature` (e.g. the
  Responses `summary[]` array, reasoning item `id`) are preserved in the
  block's format namespace of `extra` for lossless same-provider round-trips.
- Cross-provider: thinking blocks are **dropped with a warning** by default.
  `ConvertOptions.thinking_as_text: bool` (default `false`) instead converts
  plaintext thinking into the target's thinking-text channel — useful when
  switching between open-weight models whose chains of thought are plaintext.

### 4.5 Tools

```rust
#[non_exhaustive]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>, // JSON Schema, passed through verbatim
    pub strict: Option<bool>,
    pub extra: Extra,
}

#[non_exhaustive]
pub enum ToolChoice { Auto, None, Required, Tool { name: String } }
```

- `ToolCall.arguments` is a **`String`** (exact bytes as received; streaming
  delivers string fragments; invalid JSON from a model is preserved). Helper
  `arguments_json() -> Result<Value>` parses on demand. Serializing to formats
  whose native representation is an object (Anthropic `input`, Google `args`)
  parses the string; on invalid JSON: strict mode errors, lenient mode
  substitutes `{}` and attaches a semantic warning carrying the raw string.
- `ToolResult.content` is a block list (images are valid tool results on
  Responses, Anthropic and Google; CC tool messages are **text-only**, so
  image tool results converted to CC produce a semantic warning). `name` is
  required by Google's
  `functionResponse` and optional elsewhere. `is_error` is native to Anthropic
  only; other targets drop it with a warning.
- Google mapping uses `parametersJsonSchema` (standard JSON Schema
  passthrough), not the OpenAPI-style `parameters`.

`ToolChoice` mapping:

| IR | OpenAI CC | Responses | Anthropic | Google `functionCallingConfig` |
|---|---|---|---|---|
| `Auto` | `"auto"` | `"auto"` | `{type:"auto"}` | `mode: AUTO` |
| `None` | `"none"` | `"none"` | `{type:"none"}` | `mode: NONE` |
| `Required` | `"required"` | `"required"` | `{type:"any"}` | `mode: ANY` |
| `Tool{name}` | `{type:"function", function:{name}}` | `{type:"function", name}` | `{type:"tool", name}` | `mode: ANY` + `allowedFunctionNames: [name]` |

Allowed-tool lists (`allowed_tools`, `allowedFunctionNames` beyond one name)
and hosted/built-in tools are not modeled; use `extra`.

`parallel_tool_calls`: OpenAI CC/Responses `parallel_tool_calls`; Anthropic
`tool_choice.disable_parallel_tool_use` (inverted); Google has no equivalent
(warning).

### 4.6 Sampling parameters

Values are **passed through verbatim** — no scaling, no clamping, no
normalization. Ranges differ (e.g. temperature 0–2 on OpenAI/Google, 0–1 on
Anthropic); out-of-range values are the upstream API's error to report.
Parameters unsupported by the target format produce a warning:

| IR field | OpenAI CC | Responses | Anthropic | Google (`generationConfig`) |
|---|---|---|---|---|
| `max_output_tokens` | `max_completion_tokens` | `max_output_tokens` | `max_tokens` (required; unset ⇒ error¹) | `maxOutputTokens` |
| `temperature` | `temperature` | `temperature` | `temperature` | `temperature` |
| `top_p` | `top_p` | `top_p` | `top_p` | `topP` |
| `top_k` | warn | warn | `top_k` | `topK` |
| `stop_sequences` | `stop` | warn | `stop_sequences` | `stopSequences` |
| `seed` | `seed` | warn | warn | `seed` |
| `frequency_penalty` | `frequency_penalty` | warn | warn | `frequencyPenalty` |
| `presence_penalty` | `presence_penalty` | warn | warn | `presencePenalty` |
| `metadata` | `metadata` | `metadata` | `metadata.user_id` (only that key; others warn) | warn |

¹ Anthropic requires `max_tokens`; if the IR leaves it unset the format layer
must fail with a clear conversion error rather than invent a number
(`AnthropicOptions.default_max_tokens: Option<u32>` may configure a fallback).

`n`/`candidateCount` (multiple candidates) is deliberately not modeled; set it
via `extra` if needed — the response parser only reads the first
choice/candidate (the rest remain visible in `Response.raw`).

### 4.7 Reasoning configuration

Anthropic `budget_tokens` and Google `thinkingBudget` are superseded by
effort-style controls and are **not modeled** (project decision; note the
bundled reference docs do not mark them deprecated). Use `extra` where a
provider still needs them.

```rust
#[non_exhaustive]
pub struct Reasoning {
    pub enabled: Option<bool>,
    pub effort: Option<Effort>,
    pub include_thoughts: Option<bool>, // return thinking summaries?
    pub extra: Extra,
}

#[non_exhaustive]
pub enum Effort { None, Minimal, Low, Medium, High, XHigh, Max, Other(String) }
```

| IR | OpenAI CC | Responses | Anthropic | Google (`thinkingConfig`) |
|---|---|---|---|---|
| `enabled: false` | `reasoning_effort:"none"` | `reasoning.effort:"none"` | `thinking:{type:"disabled"}` | warn |
| `enabled: true` | no-op | no-op | `thinking:{type:"adaptive"}` | no-op |
| `effort` | `reasoning_effort` (accepts `none`/`minimal`/`low`/`medium`/`high`/`xhigh`/`max`) | `reasoning.effort` (same set as CC) | `output_config.effort` (accepts `low`/`medium`/`high`/`xhigh`/`max`; `none`/`minimal` warn) | `thinkingLevel` (`minimal`→`MINIMAL`, `low`→`LOW`, `medium`→`MEDIUM`, `high`→`HIGH`; `none`/`xhigh`/`max` warn) |
| `include_thoughts: true` | warn | `reasoning.summary:"auto"` | `thinking.display:"summarized"` | `includeThoughts: true` |
| `include_thoughts: false` | warn | omit `summary` | `thinking.display:"omitted"` | `includeThoughts: false` |

Effort values outside the target's set produce a warning; there is no automatic
effort↔budget or effort-tier translation. `Effort::Other(s)` passes `s`
through verbatim. If `enabled` and `effort` conflict (e.g. `enabled: true`
with `effort: None`), `effort` wins and a cosmetic warning is attached. Other
display options (`summary: "concise"/"detailed"` etc.) go through `extra`.

### 4.8 Caching

First-class block-level cache hints plus a top-level cache key:

- `CacheHint { ttl: Option<String> }` on a content block marks a cache
  breakpoint. Mapping: Anthropic `cache_control: {type:"ephemeral", ttl}` (ttl
  `"5m"`/`"1h"` passed through verbatim); OpenAI CC/Responses content-part
  `prompt_cache_breakpoint: {mode:"explicit"}` (per-block ttl has no OpenAI
  equivalent — warning); Google: warning.
- `Request.cache_key` → OpenAI `prompt_cache_key` (both APIs); Anthropic and
  Google: warning.
- OpenAI top-level `prompt_cache_options` and Google `cachedContent` (a
  resource reference, not a breakpoint) are not modeled — use `extra`.

### 4.9 Structured output

```rust
#[non_exhaustive]
pub enum OutputFormat {
    JsonSchema { name: Option<String>, description: Option<String>,
                 schema: Value, strict: Option<bool> },
    JsonObject, // schema-less JSON mode
}
```

| Target | `JsonSchema` | `JsonObject` |
|---|---|---|
| OpenAI CC | `response_format: {type:"json_schema", json_schema:{name, schema, strict}}` (`name` required upstream; `"response"` synthesized when unset) | `response_format: {type:"json_object"}` |
| Responses | `text.format: {type:"json_schema", name, schema, strict}` | `text.format: {type:"json_object"}` |
| Anthropic | `output_config.format: {type:"json_schema", schema}` (`name`/`strict` warn) | warn (no schema-less mode) |
| Google | `generationConfig.responseMimeType:"application/json"` + `responseJsonSchema: schema` (standard JSON Schema passthrough; the OpenAPI-style `responseSchema` is not used) | `responseMimeType:"application/json"` |

The schema itself is passed through verbatim. Providers accept different JSON
Schema subsets (OpenAI strict mode, Google's subset); the library performs no
subset validation or rewriting — upstream errors are authoritative. Responses
arrive as ordinary text content; `Response::parse_json()` is a convenience that
parses the first text block. No changes to the response or streaming models.

## 5. The `extra` mechanism and hooks

The escape hatch for everything the IR does not model. Two complementary parts:

### Data `extra` (on every IR node)

```rust
pub struct Extra(BTreeMap<String, Map<String, Value>>); // format id -> fields
```

- Namespaced by format id. Serializing to format F merges only `extra[F]` into
  that node's JSON output, with JSON Merge Patch semantics (RFC 7396): objects
  merge recursively, arrays and scalars replace, `Value::Null` **deletes** the
  key at any depth. Nested keys (e.g. Google's
  `generationConfig.thinkingConfig.…`) can thus be set without clobbering
  sibling fields the format layer generated.
- Parsing (format → IR) collects unknown fields into the source format's
  namespace — this is what makes same-provider round-trips lossless and keeps
  foreign dialect fields from leaking into other formats' JSON.
- The namespace key `"_ir"` is reserved for the library's round-trip markers
  (original placement of hoisted system content, tool-message grouping,
  developer-role origin, …). User code must not use it.
- Because `extra` is plain data, IR types stay serializable and comparable.

### Hooks (outside the IR)

Closures live in configuration, not in IR nodes:

```rust
#[non_exhaustive]
pub struct RequestHooks {
    /// Runs once per serialized message: (message index, role, &mut Value).
    pub on_message: Option<Arc<dyn Fn(usize, &Role, &mut Value) + Send + Sync>>,
    /// Runs on the final request JSON before sending.
    pub on_request: Option<Arc<dyn Fn(&mut Value) + Send + Sync>>,
}
```

Hooks run after IR→JSON serialization and before sending, and are also
reachable through the pure conversion API (no client needed). They can be set
on `ProviderConfig` and overridden per call. `on_message`'s index and role
refer to the **serialized target-format** message sequence (after the splits,
merges and downgrades of § 7); top-level system channels (Anthropic `system`,
Responses `instructions`, Google `systemInstruction`) are not visited — use
`on_request` for those. Message-level targeting ("modify the third message",
"set a breakpoint on the last user message") is served either by editing that
message's `extra` in the IR or by the indexed `on_message` hook.

## 6. Conversion model

- Every IR→format conversion returns the output **plus
  `Vec<ConversionWarning>`**; nothing is silently dropped.

  ```rust
  #[non_exhaustive]
  pub struct ConversionWarning {
      pub severity: WarningSeverity, // Semantic | Cosmetic
      pub location: String,          // e.g. "messages[3].content[0]"
      pub message: String,
  }
  ```

  `Semantic` = meaning lost (thinking dropped, unsupported image source);
  `Cosmetic` = tuning lost (cache hint dropped, unsupported sampling knob).
- Default mode is lenient. `ConvertOptions.strict: bool` turns any `Semantic`
  warning into an error.
- Warnings survive the client path: `Response.warnings` carries request-build
  and response-parse warnings; a streaming call exposes request-build warnings
  on the stream handle (§ 12).
- Conversions are pure functions and perform **zero IO**.
- v1 implements both directions for every format — IR→request and request→IR
  (the parse direction powers round-trip tests and future format-to-format
  conversion), plus response→IR and stream-event→IR.

## 7. Per-format mapping rules

### 7.1 system / developer

- IR has both a top-level `system` field and `System`/`Developer` roles in the
  message array; the rules below define where each lands.
- **Anthropic Messages**: `Request.system` plus the *leading run* of
  `role=system` messages are combined into the top-level `system` field —
  `Request.system` content first, then the leading messages in order. Exactly
  one text segment with no cache hint and no extra → string form; otherwise →
  text-block array (cache hints become `cache_control`; the string form cannot
  carry them). Mid-conversation system messages stay as in-array `role=system`
  (supported by current Anthropic); `AnthropicOptions.downgrade_mid_system:
  bool` (default `false`) converts them to `user` + warning for dialect
  providers without in-array system support. Original placement is recorded in
  `extra["_ir"]`; the parser maps a top-level `system` back to `Request.system`.
- **Google**: `Request.system` plus leading in-array system messages →
  `systemInstruction` (same ordering); mid-conversation system → `user` +
  warning.
- **OpenAI CC**: `Request.system` is inserted at the front of `messages` as a
  `system` message (marker in `extra["_ir"]` restores it on parse); in-array
  system/developer messages pass through natively.
- **Responses**: `Request.system` → top-level `instructions` (text blocks
  joined with `\n\n`;
  cache hints warn — put system in the message array if breakpoints are
  needed); in-array system/developer messages pass through natively.
- **Developer role**: native on OpenAI CC/Responses. Elsewhere it downgrades
  like system (Anthropic: merged/system rules do not apply — it becomes `user`
  + warning; Google: `user` + warning). `ConvertOptions.downgrade_developer:
  bool` (default `false`) additionally downgrades developer→user *within*
  OpenAI-family formats, for CC dialects that predate the developer role.

### 7.2 Tool messages

Canonical IR form: a `Tool`-role message containing one or more `ToolResult`
blocks.

- → OpenAI CC: one `role="tool"` message per `ToolResult` block
  (`tool_call_id`, content).
- → Responses: one top-level `function_call_output` item per block.
- → Anthropic: the blocks become `tool_result` content blocks of a `user`
  message; → Google: `functionResponse` parts of a `user` content. Adjacent
  IR `Tool`/`User` messages merge into that single user turn as required;
  grouping is recorded in `extra["_ir"]` and restored when parsing back.
- Text-only IR tool results map to Google's object-valued `response` as
  `{"output": "<text>"}` (unwrapped on parse). When `ToolResult.name` is
  unset, the Google converter resolves it from the `ToolCall` with the same id
  earlier in the request; if none exists, a semantic warning is attached and
  the `tool_call_id` is used as the name. Assistant `ToolCall` blocks map
  to `tool_calls[]` / `function_call` items / `tool_use` blocks /
  `functionCall` parts; provider-native ids (`call_id`, item `id`,
  `thoughtSignature`) round-trip via the block's `extra`.

### 7.3 Orphan tool calls and missing thinking

- `ConvertOptions.orphan_tool_calls` (applies to trailing assistant tool calls
  without matching results — e.g. an interrupted agent):
  - `Passthrough` (default): send as-is; the upstream error is returned.
  - `DropTrailing`: remove the unmatched trailing `ToolCall` blocks (and the
    whole message if it becomes empty), with a warning.
  - `SynthesizeError`: append a synthetic `is_error` tool result (content e.g.
    `"cancelled"`) for each orphan, keeping history well-formed.
  Orphans in the middle of the array are never repaired — warning only.
- An assistant message containing `ToolCall` but no `Thinking` block, while the
  request enables thinking, triggers a targeted warning — signatures cannot be
  fabricated, and the strictest providers reject or degrade such history
  (Google documents `thoughtSignature` as required with function calling on
  thinking models, with a `MISSING_THOUGHT_SIGNATURE` finish reason; Responses
  requires reasoning items alongside function calls under manual context
  management). `ConvertOptions.fill_missing_thinking: Option<String>`
  (default `None`) inserts a thinking block with the given text (e.g. `"tool
  call"`). This only genuinely helps signature-less channels (CC dialects'
  `reasoning_content`); for signature-validated providers the option cannot
  help and the warning + upstream error remain (if an upstream guide documents
  a placeholder workaround, it can be applied via `extra`).

## 8. Response model

Two layers with different lifecycles:

```rust
#[non_exhaustive]
pub struct Response {
    pub id: Option<String>,
    pub model: Option<String>,
    pub message: Message,          // assistant message, reuses ContentBlock
    pub stop_reason: StopReason,
    pub usage: Usage,
    pub status: u16,               // HTTP status
    pub headers: http::HeaderMap,  // rate-limit headers etc.
    pub raw: Option<Value>,        // full original response body
    pub warnings: Vec<ConversionWarning>, // request-build + response-parse
}
```

- **Envelope** (`id`/`model`/`stop_reason`/`usage`/`status`/`headers`): terminal
  data, never re-serialized. `raw` is the complete original body (`None` for
  responses accumulated from a stream — raw stream data is available via the
  `include_raw` stream option instead). Original values of mapped fields (e.g.
  the provider's own finish reason string) are always recoverable from `raw`.
- **`message` and its blocks**: these re-enter subsequent requests as history,
  so they carry the same namespaced `extra` as request-side blocks — thinking
  signatures, Responses item ids and `thoughtSignature`s flow back through this
  channel.

`StopReason`:

```rust
#[non_exhaustive]
pub enum StopReason { EndTurn, MaxTokens, StopSequence, ToolUse,
                      ContentFilter, Refusal, Other(String) }
```

| Source | Mapping |
|---|---|
| OpenAI CC `finish_reason` | `stop`→`EndTurn`, `length`→`MaxTokens`, `tool_calls`→`ToolUse`, `content_filter`→`ContentFilter`; anything else → `Other(original)` |
| Responses | no finish reason; derived from `status` + `incomplete_details` (`max_output_tokens`→`MaxTokens`, `content_filter`→`ContentFilter`) + presence of `function_call` output items→`ToolUse`; `status:"failed"` becomes an `Error::Api` |
| Anthropic `stop_reason` | `end_turn`/`max_tokens`/`stop_sequence`/`tool_use`/`refusal` map directly; `pause_turn`, `model_context_window_exceeded`, `compaction` → `Other(original)` |
| Google `finishReason` | `STOP`→`EndTurn`, `MAX_TOKENS`→`MaxTokens`, safety family (`SAFETY`, `PROHIBITED_CONTENT`, `BLOCKLIST`, `SPII`, `IMAGE_*`)→`ContentFilter`, everything else → `Other(original)` |

Normalization rule: a would-be `EndTurn` whose message contains `ToolCall`
blocks becomes `ToolUse` (Google reports `STOP` for function calls).

`Usage`:

```rust
#[non_exhaustive]
pub struct Usage {
    pub input_tokens: u64,             // ALL input tokens, incl. cache reads/writes
    pub output_tokens: u64,
    pub total_tokens: Option<u64>,
    pub cache_read_tokens: Option<u64>,
    pub cache_write_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
    pub raw: Option<Value>,            // original usage object
}
```

Semantic unification: OpenAI `prompt_tokens` and Google `promptTokenCount`
already include cached tokens; Anthropic's `input_tokens` does **not**, so the
Anthropic parser sums `input_tokens + cache_creation_input_tokens +
cache_read_input_tokens`. This is field-semantics alignment (addition), not
value conversion. `reasoning_tokens` ← OpenAI `reasoning_tokens` / Anthropic
`output_tokens_details.thinking_tokens` / Google `thoughtsTokenCount`.

## 9. Streaming

Unified block-level event model (closest to Anthropic's, mid-granularity —
every format maps into it):

```rust
#[non_exhaustive]
pub struct StreamItem {
    pub event: StreamEvent,
    pub raw: Option<String>, // original payload: always Some for Unknown,
                             // Some for every event when include_raw is on
}

#[non_exhaustive]
pub enum StreamEvent {
    MessageStart { id: Option<String>, model: Option<String>, usage: Option<Usage> },
    BlockStart { index: usize, block: ContentBlock },        // may be partial
    BlockDelta { index: usize, delta: BlockDelta },
    BlockStop  { index: usize },
    MessageDelta { stop_reason: Option<StopReason>, usage: Option<Usage> },
    MessageStop,
    Unknown,   // unrecognized event; payload in StreamItem.raw
}

#[non_exhaustive]
pub enum BlockDelta { Text(String), Thinking(String), Signature(String),
                      ToolArguments(String) }
```

- Per-format parsers: CC chunk deltas (tool calls grouped by `index`,
  `reasoning_content`↔`content` transitions open new blocks), Responses
  semantic events, Anthropic events (near-direct mapping), Google partial
  `GenerateContentResponse` objects (block boundaries inferred from parts).
  Boundary inference for CC/Google is the highest-risk parsing code and gets
  dedicated fixtures.
- **Accumulator** (`StreamEvent`s → `Response`): appends blocks strictly in
  arrival order and never merges same-typed blocks — interleaved
  thinking→text→thinking→text sequences survive verbatim. Usage merges
  field-wise (latest non-`None` value per field wins — e.g. Anthropic reports
  input counts in `message_start` and cumulative output in `message_delta`).
  Provided because agents typically render deltas while also keeping the full
  message for history.
- `include_raw: bool` (default `false`, in `CallOptions`) populates
  `StreamItem.raw` for every event; for `Unknown` it is populated regardless.
  A recognized event carrying an unrecognized delta type (e.g. a future
  `citations_delta`) is surfaced whole as `Unknown` rather than silently
  dropped. Refusal content (CC `refusal` field/delta, Responses refusal parts)
  parses into a `Text` block whose format namespace in `extra` records
  `{"refusal": true}` — it survives accumulation and round-trips even though
  `Response.raw` is `None` for streamed responses.
- SSE parsing lives in the library, on top of the transport's byte stream.
  Google streaming always uses `?alt=sse` (the JSON-array mode is not
  implemented). For CC, `stream_options: {include_usage: true}` is injected by
  default (`OpenAiChatCompletionsOptions.inject_include_usage: bool`, default `true`;
  disable for dialects that reject it). Anthropic usage in `message_delta` is
  cumulative; Google's final chunk carries the authoritative `usageMetadata`;
  Responses' terminal event carries the full response.
- Stream-event format-to-format conversion is out of scope for v1.

## 10. Transport layer

```rust
pub trait HttpClient: Send + Sync {
    fn send(
        &self,
        request: http::Request<Bytes>,       // complete, ready-to-send
        auth: Option<AuthHeader>,
    ) -> Pin<Box<dyn Future<Output = Result<http::Response<BodyStream>, HttpError>> + Send + '_>>;
}
pub type BodyStream = Pin<Box<dyn Stream<Item = Result<Bytes, HttpError>> + Send>>;

#[non_exhaustive]
pub struct HttpError {
    pub kind: HttpErrorKind, // Connect | Timeout | Body | Protocol | Other
    pub source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

pub struct AuthHeader {
    pub name: http::HeaderName,
    pub prefix: Option<String>, // e.g. "Bearer "; injected value = prefix + key
    pub value: ApiKey,
}
```

- Manual boxing keeps the trait dyn-compatible (`Arc<dyn HttpClient>`), which
  config-driven multi-provider setups need; one box per network call is noise.
- The request carries the **full URL** (path, query — e.g. `alt=sse` — all
  built by the format layer). The client does no URL joining.
- Contract: the client sends the request **as-is**; its sole permitted
  modification is injecting the provided `AuthHeader` at send time. Every other
  header (`content-type`, `anthropic-version`, beta flags) is the format
  layer's job. This split exists so the API key never sits in an
  `http::Request` that user code might log.
- `ApiKey` is a home-grown newtype: redacted `Debug`/`Display`, no `Serialize`,
  explicit `expose()` accessor. No `secrecy`/`zeroize` dependency; wrap your
  own if you need memory zeroing.
- Non-streaming calls collect the body internally; streaming feeds the SSE
  parser. Responses expose status + headers to the caller (§ 8).
- `reqwest` feature: `impl HttpClient for reqwest::Client` (auth injected with
  `HeaderValue::set_sensitive(true)`). The library does not wrap reqwest
  configuration — build your own client (proxy, timeouts, TLS) and pass it in.
- No WebSocket trait in v1 (added together with the Responses WebSocket
  transport later).

## 11. Format abstraction

Two layers:

- **Typed layer**: each format module owns complete serde types
  (`formats::openai_chat_completions::Request`, …) plus IR conversion functions returning
  `(output, Vec<ConversionWarning>)`. Usable standalone without the client.
- **Dynamic layer**: an object-safe, **public, third-party-implementable**
  trait — the client depends only on `dyn ApiFormat`:

```rust
pub trait ApiFormat: Send + Sync {
    fn id(&self) -> &str;
    fn build_request(&self, req: &Request, cfg: &BuildCtx)
        -> Result<BuiltRequest>; // JSON body + URL + headers + auth spec + warnings
    fn parse_response(&self, body: &[u8], meta: &ResponseMeta)
        -> Result<(Response, Vec<ConversionWarning>)>;
    /// One parser instance per stream: block-boundary inference is stateful
    /// (CC tool-call index grouping and reasoning_content transitions, Google
    /// part inference, Responses output/content index flattening).
    fn stream_parser(&self) -> Box<dyn StreamParser>;
    fn parse_request(&self, body: &[u8]) -> Result<(Request, Vec<ConversionWarning>)>;
    // Model listing / token counting; default impls return NotSupported.
    fn build_models_request(&self, ctx: &BuildCtx, cursor: Option<&str>) -> Result<BuiltRequest>;
    fn parse_models_response(&self, body: &[u8]) -> Result<(Vec<Model>, Option<String>)>; // page + next cursor
    fn build_count_tokens_request(&self, req: &Request, ctx: &BuildCtx) -> Result<BuiltRequest>;
    fn parse_count_tokens_response(&self, body: &[u8]) -> Result<TokenCount>;
}

pub trait StreamParser: Send {
    fn parse(&mut self, event: &SseEvent) -> Result<Vec<StreamEvent>>;
}
```

(Signatures illustrative; the real trait is designed once, carefully — it is a
semver commitment surface. `#[non_exhaustive]` context structs and default
methods keep it extensible.)

## 12. Client and configuration

```rust
#[non_exhaustive]
pub struct ProviderConfig {
    pub model: String,                        // default model; per-call override
    pub auth: Option<ApiKey>,                 // provider-level default
    pub extra_headers: http::HeaderMap,       // arbitrary additional headers
    pub extra_query: Vec<(String, String)>,
    pub convert: ConvertOptions,
    pub format_options: FormatOptions,        // per-format knobs
    pub hooks: RequestHooks,
    pub chat: EndpointConfig,                 // required
    pub models: Option<EndpointConfig>,       // capability-level decoupling
    pub count_tokens: Option<EndpointConfig>,
}

#[non_exhaustive]
pub struct EndpointConfig {
    pub format: Arc<dyn ApiFormat>,
    pub url: http::Uri,                       // base URL for this capability
    pub auth: Option<AuthHeader>,             // overrides provider default
    pub headers: Option<http::HeaderMap>,     // overrides provider default
}
```

- Capability decoupling supports real-world providers that, e.g., serve chat in
  Anthropic format but list models only in OpenAI format, or host capabilities
  on different URLs. Unset `models`/`count_tokens` derive from `chat` where the
  format supports it.
- Convenience constructors cover the common case (one format, one base URL, one
  key).
- `ConvertOptions { strict, downgrade_developer, orphan_tool_calls,
  thinking_as_text, fill_missing_thinking }`.
- `FormatOptions` (per-format):
  - `AnthropicOptions { auth_style: XApiKey | Bearer,
    version: Option<String> /* default Some("2023-06-01"); None omits the
    anthropic-version header, which some dialect providers do not want */,
    betas: Vec<String> /* joined into an anthropic-beta header */,
    downgrade_mid_system: bool, merge_consecutive_roles: bool,
    default_max_tokens: Option<u32> }`. `Bearer` only switches the auth header
    from `x-api-key` to `Authorization: Bearer` — needed by messages-format
    providers such as OpenRouter; if a particular setup additionally requires
    a beta flag, that is the user's responsibility (via `betas` or
    `extra_headers`), not the library's. `merge_consecutive_roles` (default `false`) merges adjacent
    same-role messages for dialect providers that do not auto-merge.
  - `OpenAiChatCompletionsOptions { inject_include_usage: bool /* default true */ }`.
  - Responses/Google: none yet.
- Per-call options: `CallOptions { model, convert, hooks, extra_headers,
  extra_query, include_raw }` — field-wise merge with the provider config,
  per-call wins. Format, URL and auth are deliberately not per-call
  (use another `ProviderConfig` for that).
- Default auth header per format: `Authorization: Bearer` (OpenAI family),
  `x-api-key` + `anthropic-version` (Anthropic), `x-goog-api-key` (Google —
  header, not query, to keep keys out of logs).

Client surface (sketch): `Client::new(http)`, then
`client.send(&provider, &request, opts) -> Result<Response>` (warnings ride in
`Response.warnings`), `client.stream(...) -> Result<StreamHandle>` where
`StreamHandle: Stream<Item = Result<StreamItem>>` and also exposes the
request-build warnings, `client.list_models(&provider)`,
`client.count_tokens(&provider, &request)`.

## 13. Model listing and token counting

- `Model { id, display_name: Option<String>, created: Option<SystemTime>,
  raw: Value }` — only the intersection is modeled; everything else stays in
  `raw`.
- `list_models` auto-paginates to exhaustion (Anthropic cursor via
  `after_id`/`has_more`; Google `pageToken` with `pageSize` up to 1000; OpenAI
  is a single page). Fine-grained pagination control = use the typed format
  layer directly.
- `count_tokens(&Request) -> TokenCount { input_tokens, raw }` — endpoints:
  OpenAI Responses `POST /v1/responses/input_tokens`, Anthropic
  `POST /v1/messages/count_tokens`, Google `:countTokens`. OpenAI CC has no
  endpoint → `Error::NotSupported`. The library never estimates tokens locally.

## 14. Errors

```rust
#[non_exhaustive]
pub enum Error {
    Transport(HttpError),
    Api { status: u16, kind: ApiErrorKind, message: String,
          raw: Value, retry_after: Option<Duration>, headers: http::HeaderMap },
    Conversion(ConversionError),   // strict-mode failures, invalid IR
    Parse { message: String, raw: Bytes },
    NotSupported(&'static str),    // e.g. count_tokens on openai_chat_completions
}

#[non_exhaustive]
pub enum ApiErrorKind { InvalidRequest, Auth, PermissionDenied, NotFound,
                        RateLimit, Overloaded, ServerError, Other(String) }
```

- `kind` is a coarse classification mapped from each provider's error shape;
  the **raw error body is always preserved**.
- `retry_after` is extracted from response headers where present — callers
  implement their own retry policies.
- `Error` implements `std::error::Error` and `Display`; `HttpError` nests as
  the `Transport` variant's source.

## 15. Testing

- Conversion is pure → exhaustive unit tests per field mapping, plus
  round-trip assertions (`format JSON -> IR -> format JSON` structural
  equality, including `extra` preservation).
- Fixtures under `tests/fixtures/<format_id>/` — real request/response JSON
  and complete SSE streams taken from `docs/official_api` and recorded
  sessions. Stream block-boundary inference (CC, Google) gets the densest
  coverage, including interleaved thinking/text and tool-argument fragments.
- HTTP layer tested with `wiremock`.
- Live tests: `#[ignore]` + env-gated API keys (`OPENAI_API_KEY`,
  `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `DEEPSEEK_API_KEY` — DeepSeek
  exercises the CC dialect path). Never run in CI by default.

## 16. Open items for future versions

- Responses WebSocket transport (reuses the Responses format types; introduces
  a WebSocket counterpart to `HttpClient`).
- Google Interactions API as a fifth format (server-side conversation state via
  `previous_interaction_id`; SSE streaming).
- Format-to-format conversion, composed through the IR (request parsing is
  already implemented in v1).
- Audio/video/document content blocks; image generation.
- Structured-output schema subset validation (deliberately absent today).
