# llm-api Design

Status: v1 design, agreed 2026-08-08 and revised through subsequent audit
rounds (see git history). Implemented 2026-08-09; sections updated where
implementation against the official docs refined a decision (see the git
history for the audit trail). `docs/impl_contract.md` records the binding
cross-format implementation decisions layered on top of this document.

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
- Same-provider round-trips are **canonicalizing, with explicitly documented
  representational losses**: the first `format -> IR -> format` pass may
  normalize equivalent encodings (string shorthands vs single-element arrays,
  explicit `null` vs absent optional fields), after which the mapping is
  idempotent — re-parsing and re-serializing the canonical form reproduces it
  exactly. Preserved verbatim: modeled fields, **non-null** unknown fields,
  unmodeled union members (`Opaque`, § 4.3) and message order. The one
  documented representational loss: null-valued unknown fields canonicalize
  to absent — for an unknown field the library cannot know whether `null`
  carries distinct semantics. Nothing else is silently dropped.

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
image generation, and Google tuned models — both `tunedModels/…` resource
names and Vertex AI tuned endpoints (a different resource and auth scheme);
first-party tuning availability is currently tied to the retiring Gemini 2.5
generation. The IR reserves room via `#[non_exhaustive]` enums/structs; no
code is written for these in v1.

## 3. Crate layout

- Single crate `llm-api`, edition 2024. License: MIT.
- `default-features = false` yields a pure data layer: IR types, format types,
  conversions — no IO, no tokio, no TLS. Base dependencies: `serde`,
  `serde_json`, `http`, `bytes`, `futures-core`, plus a small date-time crate
  for model timestamps (§ 13).
- Feature `reqwest`: default `HttpClient` implementation.
- MSRV: 1.88 (measured with `cargo-msrv`, declared as `rust-version`);
  policy is to follow a recent stable.
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
    pub round_trip: Option<RoundTripMeta>,   // § 4.2
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
    pub round_trip: Option<RoundTripMeta>,
    pub extra: Extra,
}
```

`Role` is the union of all supported formats. Downgrade rules for formats that
lack a role are defined in § 7.1. Messages are never dropped or reordered by
conversion (outside the explicit opt-in policies of § 7.3); splits/merges
required by a target format are deterministic and
recorded in the message's `round_trip` metadata so the original shape is
restored on the way back. Round-trip metadata flows in one direction:
format→IR parsers **attach** it (recording the original wire shape, e.g.
which IR messages were split out of one wire turn), and IR→format
serialization **consumes** it (wire JSON cannot carry markers). IR built by
hand has no metadata and serializes to the canonical shape.

`RoundTripMeta` is an opaque, versioned struct: serializable (so persisted IR
keeps it) but with no public fields. It is validated defensively on use — a
missing, malformed or future-version value degrades to canonical placement;
tampering can never corrupt a conversion, only forfeit exact-shape
restoration. `extra` holds provider data only and reserves nothing.

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
    ToolCall { id: Option<String>, name: String, arguments: String,
               cache: Option<CacheHint>, extra: Extra },
    ToolResult { tool_call_id: Option<String>, name: Option<String>,
                 content: Vec<ToolOutputBlock>, is_error: Option<bool>,
                 cache: Option<CacheHint>, extra: Extra },
    Thinking { text: Option<String>, signature: Option<String>, extra: Extra },
    /// An unmodeled provider node (Anthropic document/server-tool blocks,
    /// Responses built-in tool items, Google executable-code parts, …), kept
    /// in place so order and same-format round-trips survive.
    Opaque { format: String, value: Value },
}

#[non_exhaustive]
pub enum ImageSource {
    Url(String),
    Base64 { media_type: String, data: String },
    FileId(String), // provider-specific; cross-provider conversion warns
}

/// Tool-result content is a deliberately restricted union: no format can
/// express tool calls, tool results or thinking nested inside a tool result,
/// so the type rules those out by construction.
#[non_exhaustive]
pub enum ToolOutputBlock {
    Text { text: String, cache: Option<CacheHint>, extra: Extra },
    Image { source: ImageSource, cache: Option<CacheHint>, extra: Extra },
    /// Same semantics as ContentBlock::Opaque (unmodeled tool-result content,
    /// e.g. Anthropic document/search-result blocks).
    Opaque { format: String, value: Value },
}
```

Conversion performs **zero IO** — the library never downloads a URL to inline
it. Source mapping:

| `ImageSource` | OpenAI CC | Responses | Anthropic | Google |
|---|---|---|---|---|
| `Url` | `image_url.url` | `input_image.image_url` | `source:{type:"url", url}` | `fileData.fileUri` (verbatim + cosmetic warning: documented URIs are Files API ones; arbitrary URLs may be rejected upstream. The parse direction maps `fileData.fileUri` to `FileId`, its canonical IR home, so Google→Google round-trips are warning-free) |
| `Base64` | `image_url.url` as `data:` URL | `input_image.image_url` as `data:` URL | `source:{type:"base64", media_type, data}` | `inlineData:{mimeType, data}` |
| `FileId` | semantic warning (no **image** channel; CC's `file` content part is a document-input channel, reserved for a future Document block) | `input_image.file_id` | `source:{type:"file", file_id}` | `fileData.fileUri` |

A file id is only meaningful on the provider that issued it; the library maps
it syntactically and leaves validity to the upstream API. OpenAI's `detail`
and similar per-format image options live in `extra`.

`Opaque` blocks parse from any union member the IR does not model, at their
original position. They serialize back only to their own `format`, verbatim — `value` is its own
escape hatch, edit it directly; any other target raises a semantic warning. This — not
`#[non_exhaustive]` — is what lets v1 carry today's unmodeled provider nodes.
`ToolCall.id`/`ToolResult.tool_call_id` are optional because Google's
`functionCall.id` is optional (pairing there is by name/order); formats that
require ids (CC, Responses, Anthropic) raise a `ConversionError` when one is
absent.

### 4.4 Thinking

`Thinking { text, signature, extra }`:

- `text`: plaintext chain of thought (CC dialects' `reasoning_content`;
  Responses `content` arrays of `reasoning_text` parts — raw reasoning,
  preferred over the summary when both are exposed) or joined summary text.
  Note: replaying `reasoning_content` in **input** messages is rejected by
  some dialects (DeepSeek documents a 400); the library maps the channel
  faithfully and returns the upstream error unchanged.
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
- Provenance is namespace-based: a block is native to target `F` when its
  `extra` carries the `F` namespace, or when it has a signature and **no**
  format namespace at all (parsers leave no namespace when nothing beyond
  `text`/`signature` needs preserving, so provenance can be unknowable; such
  blocks are replayed optimistically — the upstream validates signatures
  authoritatively). Plaintext-only thinking is native to CC (its channel is
  plaintext); on signature-validated targets it follows the cross-provider
  rule above.

### 4.5 Tools

```rust
#[non_exhaustive]
pub enum Tool {
    Function(FunctionTool),
    /// Unmodeled tool kind (hosted/built-in tools, MCP toolsets, …);
    /// serializes back only to its own format, other targets warn.
    Opaque { format: String, value: Value },
}

#[non_exhaustive]
pub struct FunctionTool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>, // JSON Schema, passed through verbatim
    pub strict: Option<bool>,
    pub cache: Option<CacheHint>,  // Anthropic tool definitions accept cache_control
    pub extra: Extra,
}

#[non_exhaustive]
pub enum ToolChoice { Auto, None, Required, Tool { name: String } }
```

- `ToolCall.arguments` is a **`String`** (exact bytes as received; streaming
  delivers string fragments; invalid JSON from a model is preserved). Helper
  `arguments_json() -> Result<Value>` parses on demand. Serializing to formats
  whose native representation is an object (Anthropic `input`, Google `args`)
  parses the string; invalid JSON **or a non-object value** is a
  `ConversionError` in **both** modes (Anthropic `input` and Google `args`
  require JSON objects) — the library never fabricates arguments the model
  did not produce (the error carries the raw string for diagnosis).
- `ToolResult.content` is a `Vec<ToolOutputBlock>` (§ 4.3) — text, images and
  opaque nodes only. Tool-result image mapping per source:

  | `ImageSource` | CC | Responses | Anthropic | Google (`FunctionResponsePart`) |
  |---|---|---|---|---|
  | `Url` | dropped, semantic warning¹ | `input_image.image_url` | `source:{type:"url"}` | no channel — semantic warning |
  | `Base64` | dropped, semantic warning¹ | data URL | `source:{type:"base64"}` | `parts[].inlineData` |
  | `FileId` | dropped, semantic warning¹ | `input_image.file_id` | `source:{type:"file"}` | no channel — semantic warning |

  ¹ CC tool messages are text-only. Google's `FunctionResponsePart` union has
  a single member, `inlineData` (base64; `fileData` exists only on Vertex),
  so URL/FileId tool images have no zero-IO channel there. `name` is required
  by Google's `functionResponse` and optional elsewhere (on CC it maps to the
  tool message's `name` field — absent from the current official schema but
  accepted/required by common dialects). `is_error: true` is
  native to Anthropic (`is_error`) **and Google** (the documented
  `functionResponse.response` failure key: `{"error": …}` instead of
  `{"output": …}`); CC and Responses drop it with a **semantic** warning — an
  error result would otherwise read as success. `is_error: Some(false)`
  equals the default reading everywhere and canonicalizes to absent,
  silently.
- Google mapping uses `parametersJsonSchema` (standard JSON Schema
  passthrough), not the OpenAPI-style `parameters`.
- `FunctionTool.parameters: None` means "no parameters". CC: field omitted
  (officially "defines a function with an empty parameter list"); Google:
  `parametersJsonSchema` omitted; Responses: `parameters: null` — the field
  is required-but-nullable there, so omission would violate the schema;
  Anthropic: `input_schema` is required, so
  `{"type":"object","properties":{},"additionalProperties":false}` is emitted
  with a cosmetic warning — the faithful encoding of an empty parameter list
  (a bare `{"type":"object"}` would instead permit arbitrary arguments).
- `FunctionTool.strict` maps independently and is never rewritten by the
  `parameters` rules: CC/Responses/Anthropic send the user's value verbatim
  (Responses emits `strict: null` only when `strict` is also unset); Google
  has no per-tool strict — **semantic** warning (schema adherence is a
  tool-call contract, not tuning; mapping all-strict toolsets to
  `functionCallingConfig.mode: VALIDATED` is a possible future approximation,
  not v1). Whether `strict: true` next to an empty schema is meaningful is
  the upstream's call.
- `FunctionTool.cache` maps to Anthropic tool-level `cache_control`; other
  targets drop it with a cosmetic warning.

`ToolChoice` mapping:

| IR | OpenAI CC | Responses | Anthropic | Google `functionCallingConfig` |
|---|---|---|---|---|
| `Auto` | `"auto"` | `"auto"` | `{type:"auto"}` | `mode: AUTO` |
| `None` | `"none"` | `"none"` | `{type:"none"}` | `mode: NONE` |
| `Required` | `"required"` | `"required"` | `{type:"any"}` | `mode: ANY` |
| `Tool{name}` | `{type:"function", function:{name}}` | `{type:"function", name}` | `{type:"tool", name}` | `mode: ANY` + `allowedFunctionNames: [name]` |

Allowed-tool lists (`allowed_tools`, `allowedFunctionNames` beyond one name)
and hosted/built-in tools are not modeled; use `extra`.

`parallel_tool_calls`: OpenAI CC/Responses `parallel_tool_calls`. Anthropic
nests the inverted flag inside the tool-choice object, so the combinations
are pinned: `tool_choice: Some(x)` → attach `disable_parallel_tool_use` to
`x` (meaningless on `ToolChoice::None` — cosmetic warning, not emitted);
`tool_choice: None` + `Some(false)` → synthesize
`{"type":"auto","disable_parallel_tool_use":true}`; `tool_choice: None` +
`Some(true)` → emit nothing (parallel is Anthropic's default —
canonicalization); a request without `tools` makes the setting meaningless on
any format — cosmetic warning. Google has no equivalent — dropping
`Some(false)` (a serial-execution constraint) is a **semantic** warning,
dropping `Some(true)` is cosmetic.

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

Severity of the `warn` cells follows the § 6 test: losing `stop_sequences`
(an output-control contract) or an unmappable `Reasoning.enabled: false` is
**semantic**; losing pure sampling tuning (`seed`, `top_k`, penalties) or
`metadata` is cosmetic. Every `WarningCode` has a fixed severity so
classifications cannot drift between formats (asserted centrally in tests,
§ 15).

`n`/`candidateCount` (multiple candidates) is deliberately not modeled; set it
via `extra` if needed — the response parser only reads the first
choice/candidate (the rest remain visible in `Response.raw`).

### 4.7 Reasoning configuration

Anthropic `budget_tokens` and Google `thinkingBudget` are **not modeled**:
v1's model-support baseline excludes the retiring models that only accept
manual budgets. That is a project support policy, not a claim about the APIs
(the bundled reference docs do not mark those fields deprecated). Use `extra`
where a provider still needs them — the RFC 7396 merge makes the override complete,
e.g. `extra["anthropic_messages"] = {"thinking": {"type": "enabled",
"budget_tokens": 2048}}` rewrites the generated `thinking` object, and
`{"generationConfig": {"thinkingConfig": {"thinkingLevel": null,
"thinkingBudget": 512}}}` does the same for Google. Note that Anthropic's
`output_config.effort` governs overall output effort, not thinking alone: the
`effort` mapping to Anthropic is an approximation, not an exact equivalent.

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
On Anthropic, `include_thoughts` alone (no `enabled: true`, no mappable
effort) does not synthesize a `thinking` object — silently enabling thinking
would be a side effect — and warns instead; `thinking.display` is attached
only when thinking has a basis.

### 4.8 Caching

First-class block-level cache hints plus a top-level cache key:

- `CacheHint { ttl: Option<String> }` on a content block marks a cache
  breakpoint. Mapping: Anthropic `cache_control: {type:"ephemeral", ttl}` (ttl
  `"5m"`/`"1h"` passed through verbatim); OpenAI CC/Responses content-part
  `prompt_cache_breakpoint: {mode:"explicit"}` (per-block ttl has no OpenAI
  equivalent — warning); Google: warning. Anthropic accepts `cache_control`
  on every request block except thinking, so `ToolCall`/`ToolResult` hints map
  natively there; CC/Responses breakpoints exist only on content parts, so
  hints on `ToolCall` blocks warn (cosmetic). Cache hints on **nested**
  `ToolOutputBlock`s (inside a tool result) are dropped with a cosmetic
  warning on every target in v1 — the supported breakpoint channels are the
  `ToolResult` block itself (Anthropic) and regular content parts (OpenAI);
  if implementation verifies nested support somewhere, relaxing this is
  non-breaking (fewer warnings).
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
| Responses | `text.format: {type:"json_schema", name, schema, strict}` (`name` required upstream here too; same `"response"` synthesis) | `text.format: {type:"json_object"}` |
| Anthropic | `output_config.format: {type:"json_schema", schema}` (`name`/`strict` warn — cosmetic: Anthropic's json_schema output is natively enforced, the strict toggle has nothing to lose) | warn (no schema-less mode) |
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
  sibling fields the format layer generated. One consequence: `extra` cannot
  set a field to literal JSON `null` (null means delete); the rare case that
  needs one goes through the `on_request` hook, which edits the raw `Value`
  directly.
- Parsing (format → IR) collects unknown fields into the source format's
  namespace, which keeps foreign dialect fields from leaking into other
  formats' JSON. Non-null unknown fields round-trip verbatim; a null-valued
  unknown field canonicalizes to absent on re-serialization (null means
  delete in the merge) — the documented representational loss of § 1.
- Round-trip markers (original placement of hoisted system content,
  tool-message grouping, developer-role origin, …) do **not** live in `extra`;
  they ride the dedicated `round_trip: Option<RoundTripMeta>` field on
  `Request`/`Message` (§ 4.2). `extra` carries provider data only, with no
  reserved keys.
- Because `extra` is plain data, IR types stay serializable and comparable.

### Hooks (outside the IR)

Closures live in configuration, not in IR nodes:

```rust
#[non_exhaustive]
pub struct RequestHooks {
    /// Runs once per serialized message: (message index, role, &mut Value).
    pub on_message: Option<Arc<dyn Fn(usize, &Role, &mut Value) -> Result<(), HookError> + Send + Sync>>,
    /// Runs on the final request JSON before sending.
    pub on_request: Option<Arc<dyn Fn(&mut Value) -> Result<(), HookError> + Send + Sync>>,
}
```

Hooks run after IR→JSON serialization and before sending, and are also
reachable through the pure conversion API (no client needed); a hook returning
`Err` aborts the call with `Error::Hook`. They can be set
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
      pub code: WarningCode,         // #[non_exhaustive] enum — stable, matchable
      pub severity: WarningSeverity, // Semantic | Cosmetic
      pub location: String,          // JSON Pointer: build-side warnings point into
                                     // the final target JSON, parse-side warnings
                                     // into the JSON being consumed
      pub format: String,                 // the provider format involved
      pub direction: ConversionDirection, // ToFormat (build) | FromFormat (parse)
      pub overridden: bool,          // `extra` explicitly addressed this path (§ below)
      pub message: String,
  }
  ```

  `Semantic` = meaning lost (thinking dropped, unsupported image source);
  `Cosmetic` = tuning lost (cache hint dropped, unsupported sampling knob).
  The test is whether model-visible behavior or contract can change, not the
  field's category.
  Data the target format structurally **requires** (Anthropic `max_tokens`, a
  required tool-call id, an object parseable from `arguments`, a resolvable
  `functionResponse.name`) is never invented: its absence is a
  `ConversionError` in both modes — lenient mode drops extras, it does not
  fabricate.
- Default mode is lenient. `ConvertOptions.strict: bool` turns any
  non-overridden `Semantic` warning from the **IR→request conversion** into
  an error. Parse-side warnings (multi-candidate skipped, partial mappings,
  recoverable malformed data — non-streaming or streaming) always report and
  never fail the call: the response already happened and was billed, so the
  client never discards it — inspect `Response.warnings` /
  `StreamItem.warnings` and react in application code.
- Warnings survive the client path: `Response.warnings` carries request-build
  and response-parse warnings — parsers fill parse-side warnings in directly,
  the client prepends build-side ones. A streaming call exposes request-build
  warnings on the stream handle, parse-side warnings ride each
  `StreamItem.warnings` (§ 9), and the accumulator folds both into the
  accumulated `Response.warnings`.
- Build pipeline order: IR→JSON conversion including the `extra` merge
  (warnings collected; a warning is then marked `overridden` — kept for
  debugging, ignored by the strict gate — when `extra` set or deleted exactly
  its pointer, or set a non-object value at an ancestor of it: under RFC 7396
  arrays and scalars replace while objects merge, so only those constitute an
  ancestor override; the overriding value's validity is not re-assessed) →
  strict gate → hooks (fallible, § 5) → send. Post-hook JSON is **not**
  re-validated; warnings describe the conversion stage only — hook
  consequences are the user's.
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
  providers without in-array system support. The parser maps a top-level
  `system` back to `Request.system`, and tags in-array `system` messages with
  a `round_trip` marker so re-serialization keeps their placement (including
  leading ones — the combine rule applies to marker-less IR); a missing or
  invalid marker degrades to canonical hoisting.
- **Google**: `Request.system` plus leading in-array system messages →
  `systemInstruction` (same ordering); mid-conversation system → `user` +
  warning.
- **OpenAI CC**: `Request.system` is inserted at the front of `messages` as a
  `system` message; in-array system/developer messages pass through natively.
  The parser keeps leading system messages in-array (it never hoists them to
  `Request.system`) — so IR→CC→IR canonicalizes `Request.system` into a
  leading in-array message while the JSON round-trip stays the identity.
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
  grouping is recorded in `round_trip` metadata and restored when parsing back.
- Multiple `Text` blocks in a tool result keep their boundaries on CC,
  Responses and Anthropic (text-part/block arrays; a single block uses the
  string shorthand — both forms parse back, § 1 canonicalization). Only
  Google must flatten: text maps into the object-valued `response` as
  `{"output": "<text>"}` (unwrapped on parse), multiple blocks joining in
  order with `\n\n` **plus a cosmetic warning** — user-constructed
  multi-block results are a structural downgrade on Google, while anything
  parsed from Google is single-block and joins warning-free. Google also
  splits media from text — images go to `parts[]` (relative image order
  kept), text to `response` — so an IR sequence that interleaves text after
  an image cannot keep its order: serializing one adds a **semantic** warning
  (order lost). The parse-side canonical order is the `response` text first,
  then `parts[]` images, so anything parsed from Google round-trips
  warning-free.
- An empty `ToolResult.content` has a defined encoding per target — CC
  `content: ""`, Responses `output: ""`, Anthropic `content` omitted, Google
  `response: {}` — each parsing back to the empty list (Google distinguishes
  `{}` ↔ empty from `{"output": ""}` ↔ one empty `Text` block), keeping
  round-trips idempotent: canonical encoding of emptiness, not fabrication.
- When `ToolResult.name` is
  unset, the Google converter resolves it from the `ToolCall` with the same id
  earlier in the request; if none resolves, that is a `ConversionError` —
  an invented name would only produce an invalid call. Assistant `ToolCall` blocks map
  to `tool_calls[]` / `function_call` items / `tool_use` blocks /
  `functionCall` parts; provider-native ids (`call_id`, item `id`,
  `thoughtSignature`) round-trip via the block's `extra`.

### 7.3 Orphan tool calls and missing thinking

- `ConvertOptions.orphan_tool_calls` (applies to trailing assistant tool calls
  without matching results — e.g. an interrupted agent; *trailing* means in
  the final message of the array, and a call is *matched* by a later
  `ToolResult` with the same id — same name when the call has no id):
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

### 7.4 Role × content validity

| Block \ Role | System | Developer | User | Assistant | Tool |
|---|---|---|---|---|---|
| `Text` | ok | ok | ok | ok | error |
| `Image` | error | error | ok | parse-only¹ | error |
| `ToolCall` | error | error | error | ok | error |
| `ToolResult` | error | error | error | error | ok |
| `Thinking` | error | error | error | ok | error |
| `Opaque` | ok | ok | ok | ok | ok |

Invalid combinations are structural: `ConversionError` in both modes
(`Request.system` is Text-only under the same rule). ¹ assistant images occur
in provider responses (e.g. image output parts) and parse as-is;
user-constructed assistant images converting to a format without that channel
get the usual semantic warning. Inside CC, tool messages are text-only
(§ 4.5): image blocks in a `ToolResult` are dropped with a semantic warning in
lenient mode, error in strict; text is kept. Nesting inside a `ToolResult`
needs no validation row — the `ToolOutputBlock` type (§ 4.3) rules it out by
construction.

### 7.5 Signed-block invariants

Thinking signatures (Anthropic `signature`, Responses `encrypted_content`,
Google `thoughtSignature`) are position-sensitive: providers validate them
against the exact block/part they were issued for. The conversion rules above
never reorder blocks within a message, and same-format round-trips are
identity on block order, so plain replay is always safe. The opt-ins interact
as follows:

- `merge_consecutive_roles` treats any message containing a signed block
  (`Thinking.signature`, or a Google `thoughtSignature` riding a `ToolCall`'s
  `extra`) as a **merge barrier**: the adjacent pair stays unmerged, with a
  semantic warning explaining why. Where merges do happen, block order is
  preserved — but a dialect that requires merged turns cannot be assumed to
  still validate signatures across rebuilt message boundaries, so that
  trade-off is surfaced rather than taken silently.
- `orphan_tool_calls: DropTrailing` can leave a preceding `Thinking` block
  orphaned; this raises a dedicated semantic warning (the upstream may reject
  the turn).
- `thinking_as_text` destroys signatures by definition — it exists for
  cross-provider plaintext migration only.
- `fill_missing_thinking` only inserts blocks, never moves existing ones.
- Hooks can break signatures arbitrarily; per the § 6 pipeline note, post-hook
  consequences are the user's.

## 8. Response model

Two layers with different lifecycles:

```rust
#[non_exhaustive]
pub struct Response {
    pub id: Option<String>,
    pub model: Option<String>,
    pub message: Message,          // assistant message, reuses ContentBlock
    pub stop_reason: Option<StopReason>, // None when the source omits it
    pub usage: Option<Usage>,            // absent on some streams/dialects
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
                      ContentFilter, Refusal, PauseTurn, Other(String) }
```

| Source | Mapping |
|---|---|
| OpenAI CC `finish_reason` | `stop`→`EndTurn`, `length`→`MaxTokens`, `tool_calls`→`ToolUse`, `content_filter`→`ContentFilter`; anything else → `Other(original)` |
| Responses | no finish reason; derived from `status` + `incomplete_details` (`max_output_tokens`→`MaxTokens`, `content_filter`→`ContentFilter`) + presence of `function_call` output items→`ToolUse`; `status:"failed"` becomes an `Error::Api` |
| Anthropic `stop_reason` | `end_turn`/`max_tokens`/`stop_sequence`/`tool_use`/`refusal` map directly; `pause_turn`→`PauseTurn` (actionable: resend the turn as-is to continue); `model_context_window_exceeded`, `compaction` → `Other(original)` |
| Google `finishReason` | `STOP`→`EndTurn`, `MAX_TOKENS`→`MaxTokens`, safety family (`SAFETY`, `PROHIBITED_CONTENT`, `BLOCKLIST`, `SPII`, `IMAGE_*`)→`ContentFilter`, everything else → `Other(original)` |

Normalization rules, applied in order: a would-be `EndTurn` whose message
contains `ToolCall` blocks becomes `ToolUse` (Google reports `STOP` for
function calls); then a would-be `EndTurn` — or absent stop reason — whose
message contains a refusal-marked block (§ 9) becomes `Refusal`. CC and
Responses have no refusal finish reason or status of their own, so without
this rule `StopReason::Refusal` would never be produced for them.

A Google prompt blocked by safety returns **no candidates** (only
`promptFeedback`): this parses to a `Response` with an empty-content assistant
message, `stop_reason: Some(ContentFilter)` (from `blockReason`) and the full
body in `raw`. Multiple choices/candidates stay unsupported: when a response
carries more than one, the parser reads the first and attaches a semantic
warning to `Response.warnings`; the rest remain in `raw`.

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
value conversion. `output_tokens` is likewise pinned to "all generated tokens
including reasoning": CC `completion_tokens` and Anthropic `output_tokens`
already are (both documented as reasoning-inclusive), while Google's
`candidatesTokenCount` excludes thoughts — the Google parser sums
`candidatesTokenCount + thoughtsTokenCount`. Visible output =
`output_tokens - reasoning_tokens` (saturating — misbehaving provider data
must not underflow). `reasoning_tokens` ← OpenAI
`reasoning_tokens` / Anthropic `output_tokens_details.thinking_tokens` /
Google `thoughtsTokenCount`.

## 9. Streaming

Unified block-level event model (closest to Anthropic's, mid-granularity —
every format maps into it):

```rust
#[non_exhaustive]
pub struct StreamItem {
    pub event: StreamEvent,
    pub raw: Option<String>, // original payload: always Some for Unknown,
                             // Some for every event when include_raw is on
    pub warnings: Vec<ConversionWarning>, // parse-side warnings, usually empty
}

#[non_exhaustive]
pub enum StreamEvent {
    MessageStart { id: Option<String>, model: Option<String>, usage: Option<Usage> },
    BlockStart { index: usize, block: ContentBlock },        // may be partial
    BlockDelta { index: usize, delta: BlockDelta },
    /// `block`, when present, is the parser's finalized block — citations,
    /// annotations, final status and signatures already folded into `extra`.
    /// The accumulator replaces its incrementally built block with it.
    BlockStop  { index: usize, block: Option<ContentBlock> },
    MessageDelta { stop_reason: Option<StopReason>, usage: Option<Usage> },
    MessageStop,
    Unknown,   // unrecognized event; payload in StreamItem.raw
}

#[non_exhaustive]
pub enum BlockDelta {
    Text(String), Thinking(String), Signature(String), ToolArguments(String),
    /// Recognized but unmodeled delta payload (Anthropic citations_delta,
    /// Responses annotation.added, …) — surfaced for real-time consumers and
    /// folded into the finalized block at BlockStop.
    Other(Value),
}
```

- Per-format parsers: CC chunk deltas (tool calls grouped by `index`,
  `reasoning_content`↔`content` transitions open new blocks), Responses
  semantic events, Anthropic events (near-direct mapping), Google partial
  `GenerateContentResponse` objects (block boundaries inferred from parts).
  Boundary inference for CC/Google is the highest-risk parsing code and gets
  dedicated fixtures. Each parser emits `MessageStop` itself on its protocol
  terminator; a Google blocked-prompt chunk (`promptFeedback.blockReason`, no
  candidates) immediately yields `MessageStart` (if not yet emitted) +
  `MessageDelta { stop_reason: ContentFilter }` + `MessageStop`. Google
  terminal validation looks only at the first candidate (multi-candidate is
  unsupported, § 8).
- **Accumulator** (`StreamEvent`s → `Response`): appends blocks strictly in
  arrival order and never merges same-typed blocks — interleaved
  thinking→text→thinking→text sequences survive verbatim. Any `usage` a
  stream event carries is a **complete cumulative snapshot** — the stateful
  parser folds provider partials into a running total before emitting (e.g.
  Anthropic's `message_start` input counts are cached and merged into every
  `message_delta` usage); the accumulator simply keeps the latest snapshot.
  Provided because agents typically render deltas while also keeping the full
  message for history. A stream that errors before its terminal event — including a silent EOF
  that `StreamParser::finish` diagnoses as truncation — surfaces the error
  through the stream; accumulation then fails, and the events already
  delivered remain the partial record.
- `include_raw: bool` (default `false`, in `CallOptions`) populates
  `StreamItem.raw` for every event; for `Unknown` it is populated regardless.
  Known-but-unmodeled deltas that belong to a block (Anthropic
  `citations_delta` — part of the **current** protocol — Responses
  `output_text.annotation.added`, …) surface as `BlockDelta::Other` and are
  folded into the finalized block at `BlockStop`; events the parser cannot
  attribute to a block at all fall back to `Unknown`. Chunks for candidate
  indexes beyond the first surface as `Unknown` — multi-candidate is
  unsupported (§ 8). Refusal content (CC `refusal` field/delta, Responses refusal parts)
  parses into a `Text` block whose format namespace in `extra` records
  `{"refusal": true}` — it survives accumulation and round-trips even though
  `Response.raw` is `None` for streamed responses.
- Stream parse warnings ride `StreamItem.warnings`: the client attaches a
  parse call's warnings to the first item that call emits (held for the next
  item when a call emits none; `StreamParser::finish` flushes any still-held
  warnings at end of stream, § 11); the accumulator folds every item's
  warnings into `Response.warnings`.
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
- The request carries the **final URL** (path, query — e.g. `alt=sse` — all
  built by the format layer from the `EndpointUrl` rules in § 12). The client
  does no URL joining.
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
    /// BuildCtx carries the endpoint config, model and call mode
    /// (unary vs streaming — Google's URL differs by mode).
    fn build_request(&self, req: &Request, cfg: &BuildCtx)
        -> Result<BuiltRequest>; // JSON body + URL + headers + auth spec + warnings
    /// Parse-side warnings go directly into `Response.warnings`; the client
    /// prepends request-build warnings afterwards.
    fn parse_response(&self, body: &[u8], meta: &ResponseMeta) -> Result<Response>;
    /// Non-2xx responses: classify and preserve the provider error shape.
    /// Default impl builds a generic `Error::Api` from status + raw body.
    fn parse_error(&self, status: u16, headers: &http::HeaderMap, body: &[u8]) -> Error;
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
    /// Unified events plus parse-side warnings for this provider event
    /// (skipped extra candidates, lossy inference, …).
    fn parse(&mut self, event: &SseEvent)
        -> Result<(Vec<StreamEvent>, Vec<ConversionWarning>)>;
    /// Called exactly once when the byte stream ends: flushes held warnings
    /// and safely finalizable blocks, then validates terminal state — it
    /// never synthesizes MessageStop (parsers emit that themselves, § 9).
    /// Terminators: CC `[DONE]`; Anthropic `message_stop`; a terminal
    /// Responses event; Google either a `finishReason` on the first
    /// candidate or a blocked-prompt chunk (`promptFeedback.blockReason`
    /// with no candidates). A stream that showed none of these returns
    /// `Error::Parse` ("truncated stream") — a silent EOF must not pass a
    /// half response off as complete.
    fn finish(&mut self)
        -> Result<(Vec<StreamEvent>, Vec<ConversionWarning>)>;
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
    pub limits: Limits,                       // body/event size caps, § below
    pub hooks: RequestHooks,
    pub chat: EndpointConfig,                 // required
    pub models: Option<EndpointConfig>,       // capability-level decoupling
    pub count_tokens: Option<EndpointConfig>,
}

#[non_exhaustive]
pub struct EndpointConfig {
    pub format: Arc<dyn ApiFormat>,
    pub url: EndpointUrl,
    pub auth: Override<AuthHeader>,           // Inherit | Set(_) | Disable
    pub headers: Override<http::HeaderMap>,
}

#[non_exhaustive]
pub enum EndpointUrl {
    /// Base URL; the format appends its documented per-capability path
    /// (rules below).
    Base(http::Uri),
    /// Complete request URL, used as-is after substituting the placeholders
    /// the format documents ({model}; {method} on Google). Without
    /// placeholders the string is used verbatim for every call mode.
    Full(String),
}

#[non_exhaustive]
pub enum Override<T> { Inherit, Set(T), Disable }
```

- Capability decoupling supports real-world providers that, e.g., serve chat in
  Anthropic format but list models only in OpenAI format, or host capabilities
  on different URLs. Unset `models`/`count_tokens` derive from `chat` only
  when `chat.url` is `Base` — a `Full` chat URL cannot be reliably
  decomposed; with a `Full` chat URL and no explicit config, those
  capabilities return `Error::NotSupported` when called.
- URL construction. `Base` joining: trim the base's trailing `/`, append `/`
  plus the capability path; the base's own query string is preserved. Path
  templates — CC chat `chat/completions`; Responses chat `responses`, count
  `responses/input_tokens`; Anthropic chat `messages`, count
  `messages/count_tokens`; Google chat `models/{model}:generateContent`
  (`:streamGenerateContent` when streaming), count `models/{model}:countTokens`;
  models list `models` on all four. `{model}` is percent-encoded as a single
  path segment after stripping a leading `models/` prefix; Google
  `tunedModels/…` resource names are out of scope for v1 and return
  `Error::NotSupported` instead of producing a broken URL (tuned models are
  excluded entirely, § 2 — Vertex tuned endpoints are a different
  resource/auth scheme, not a `tunedModels/…` path). `Full` serves nonstandard
  paths (`/chat`, `/completions`, bare
  `/api`, no `/v1` prefix, …); for Google-format chat a `Full` URL should
  contain `{method}`, otherwise the same URL is used for both call modes.
  Query rules: format-required keys (`alt=sse`) are protected — a
  user-supplied query with the same key is a `ConversionError`; otherwise
  later layers replace same-name keys (provider `extra_query`, then
  per-call); a `Base` URL's own query is decomposed and rebuilt, never
  string-concatenated. Header precedence: format defaults (version,
  content-type, betas) < provider `extra_headers` < endpoint `Set` (same-name
  override, other names add) < per-call < auth injection (applied last, not
  overridable by any header layer). Endpoint `Disable` drops the provider
  `extra_headers` layer for that endpoint; format defaults are **not
  removable through the generic client** — removing them breaks the protocol,
  and the legitimate cases have dedicated knobs (e.g.
  `AnthropicOptions.version: None`). Full header control means a custom
  `ApiFormat`, or the typed layer with your own transport; an
  `http::Request`-level hook can be added non-breakingly later if a real
  case appears.
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
- `Limits` (`#[non_exhaustive]`): size caps bounding memory against
  misbehaving proxies —

  ```rust
  #[non_exhaustive]
  pub struct Limits {
      pub max_response_body: usize, // decompressed bytes, as delivered by the transport
      pub max_error_body: usize,
      pub max_sse_event: usize,     // one complete logical SSE event (joined data lines)
  }
  ```

  Plain byte counts, no magic values (`usize::MAX` ≈ unlimited). Defaults are
  generous, chosen at implementation, and may be tuned in minor versions —
  the semver guarantee is the mechanism, not the numbers. Exceeding a cap
  keeps what was already read: a 2xx body or an SSE event over its cap fails
  with `Error::BodyTooLarge` (status/headers where available + the read
  prefix); an oversized **error** body still produces a full `Error::Api`
  with `truncated: true` and the prefix as `raw` (§ 14) — status, headers
  and `retry_after` survive.
- Default auth header per format: `Authorization: Bearer` (OpenAI family),
  `x-api-key` + `anthropic-version` (Anthropic), `x-goog-api-key` (Google —
  header, not query, to keep keys out of logs).

Client surface (sketch): `Client::new(http)`, then
`client.send(&provider, &request, opts) -> Result<Response>` (warnings ride in
`Response.warnings`), `client.stream(...) -> Result<StreamHandle>` where
`StreamHandle: Stream<Item = Result<StreamItem>>` and also exposes the
request-build warnings, `client.list_models(&provider)`,
`client.count_tokens(&provider, &request, opts)`. Token counting accepts the
same `CallOptions`: every body-affecting option applies with the same merge
result as `send` — `model`, `convert`, `hooks`, `extra_headers`,
`extra_query` — and only `include_raw` is ignored. Pipeline: the prospective
**chat** JSON is built first (extra, convert options and hooks all act on it,
exactly as for `send`), then a per-format count adapter reshapes it for the
count endpoint — Google wraps it in `generateContentRequest`, Anthropic
filters it to the fields its count endpoint accepts, Responses maps it onto
the `input_tokens` body; what the adapter filters is graded in § 13.
Exactness is guaranteed for the **modeled and documented request surface**
when chat and count use the same format; dropped unknown/injected fields and
decoupled count formats make the result approximate and produce **semantic**
warnings — under strict that is a `ConversionError`, and a per-call
`convert` with `strict: false` opts into approximate counting.

## 13. Model listing and token counting

- `Model { id, display_name: Option<String>, created: Option<SystemTime>,
  raw: Value }` — only the intersection is modeled; everything else stays in
  `raw`. `id` is normalized to what `ProviderConfig.model` accepts: Google's
  `models/` prefix is stripped (the original resource name stays in `raw`,
  and the URL builder tolerates both forms). `created` parses OpenAI's Unix
  seconds directly and Anthropic's RFC 3339 via a small, well-tested
  date-time dependency (picked at implementation, e.g. `jiff` or `time`); a
  timestamp that fails to parse degrades to `None` and never fails the
  model-list call. Google has no creation time (`None`).
- `list_models` auto-paginates to exhaustion (Anthropic cursor via
  `after_id`/`has_more`; Google `pageToken` with `pageSize` up to 1000; OpenAI
  is a single page). A page token/cursor equal to one already seen aborts
  with `Error::Parse` (malformed pagination) instead of looping forever.
  Fine-grained pagination control = use the typed format layer directly.
- `count_tokens(&Request) -> TokenCount { input_tokens, raw, warnings }`
  (accepts `CallOptions`, § 12) — endpoints:
  OpenAI Responses `POST /v1/responses/input_tokens`, Anthropic
  `POST /v1/messages/count_tokens`, Google `:countTokens`. OpenAI CC has no
  endpoint → `Error::NotSupported`. The library never estimates tokens
  locally. Because the prospective **chat** body is built first (§ 12),
  chat-level structural requirements apply to counting too — e.g. Anthropic
  counting fails without `max_output_tokens`/`default_max_tokens` even
  though the count endpoint itself has no `max_tokens` field. Google's count
  endpoint accepts the entire `GenerateContentRequest` (nested under
  `generateContentRequest` with its own `model`), so its adapter drops
  nothing and the count is exact. `TokenCount.warnings` carries the chat-build warnings plus the
  count adapter's own: fields the converter itself generated that the
  endpoint ignores by design (sampling knobs) are filtered silently; fields
  the adapter drops that it did not generate (injected via `extra`, hooks or
  a dialect) get a **semantic** warning — the library cannot know whether
  they would have affected the count, so the result is no longer exact; a
  decoupled count format adds one **semantic** warning marking the result as
  an approximation. Under strict these fail the call; per-call `convert`
  with `strict: false` accepts an approximate count.

## 14. Errors

```rust
#[non_exhaustive]
pub enum Error {
    Transport(HttpError),
    Api { status: u16, kind: ApiErrorKind, message: String,
          raw: Bytes,                // body (may be non-JSON/non-UTF-8)
          truncated: bool,           // raw is only a prefix (max_error_body hit)
          parsed: Option<Value>,     // present when the body parsed as JSON
          retry_after: Option<Duration>, headers: http::HeaderMap },
    Conversion(ConversionError),   // structural/strict failures, invalid IR
    Hook(HookError),               // a request hook returned Err
    Parse { message: String, raw: Bytes },
    BodyTooLarge { kind: BodyKind, // SuccessBody | SseEvent
                   limit: usize, status: Option<u16>,
                   headers: Option<http::HeaderMap>, prefix: Bytes },
    NotSupported(&'static str),    // e.g. count_tokens on openai_chat_completions
}

#[non_exhaustive]
pub enum ApiErrorKind { InvalidRequest, Auth, PermissionDenied, NotFound,
                        RateLimit, Overloaded, ServerError, Other(String) }
```

- `kind` is a coarse classification mapped from each provider's error shape;
  error bodies **within `max_error_body` are preserved verbatim** (`Bytes` —
  proxies return HTML/plain-text errors too; `parsed` is set when it is
  JSON), and an oversized one keeps status, headers, `retry_after` and the
  read prefix with `truncated: true` — never downgraded to a bare parse
  error. Non-2xx handling belongs to `ApiFormat::parse_error` (§ 11);
  `parse_response` only sees 2xx.
- `retry_after` is extracted from response headers where present — callers
  implement their own retry policies.
- `Error` implements `std::error::Error` and `Display`; `HttpError` nests as
  the `Transport` variant's source.

## 15. Testing

- Conversion is pure → exhaustive unit tests per field mapping, plus
  round-trip assertions matching the § 1 guarantee: the first pass may
  canonicalize, so tests assert **idempotence** — the canonical JSON is a
  fixed point of parse→serialize — plus preservation of `extra`, `Opaque`
  nodes and `round_trip` metadata.
- Dedicated tests: the § 4.7 extra-override examples (budget re-enable on
  Anthropic/Google must fully rewrite the generated thinking fields), the
  § 7.5 signed-block merge barrier (merged vs skipped paths), the § 7.2
  empty/multi-text tool-result encodings (round-trip idempotence per target),
  and a single central assertion of the `WarningCode` → severity table
  (§ 4.6).
- Fixtures under `tests/fixtures/<format_id>/` — real request/response JSON
  and complete SSE streams taken from `docs/official_api` and recorded
  sessions. Stream block-boundary inference (CC, Google) gets the densest
  coverage, including interleaved thinking/text, tool-argument fragments,
  citation/annotation streams, blocked-prompt responses (non-streaming
  **and** SSE), multi-candidate warning paths, and truncated streams
  (missing protocol terminator ⇒ `finish()` error).
- HTTP layer tested with `wiremock`.
- Live tests: `#[ignore]` + env-gated API keys (`OPENAI_API_KEY`,
  `GOOGLE_API_KEY`, `DEEPSEEK_API_KEY`; Anthropic reads
  `LLM_API_ANTHROPIC_API_KEY` first and `ANTHROPIC_API_KEY` from `.env`
  only — development environments like Claude Code occupy that name in the
  process environment), also read from a gitignored crate-root `.env`
  (process environment wins otherwise). Coverage per format: a multi-turn
  conversation matrix (`streaming × thinking`) that checks context
  reception against the model's own first-turn answer, multi-round
  tool-call loops with verbatim history replay, JSON-Schema structured
  output (both call modes), image input (except text-only DeepSeek), model
  listing and token counting. DeepSeek exercises the dialect paths of three
  formats — Chat Completions, Anthropic Messages (`/anthropic` base) and
  Responses. Platform notes learned live: adaptive reasoning
  (Anthropic, OpenAI's adaptive models) is model-discretionary — those
  thinking assertions are cumulative or advisory; OpenAI rejects function
  tools on CC unless `reasoning_effort` is `"none"` for such models, so
  the CC thinking×tools cell runs against DeepSeek (whose docs require the
  `reasoning_content` passback the replay performs). Never run in CI by
  default.

## 16. Open items for future versions

- Responses WebSocket transport (reuses the Responses format types; introduces
  a WebSocket counterpart to `HttpClient`).
- Google Interactions API as a fifth format (server-side conversation state via
  `previous_interaction_id`; SSE streaming).
- Format-to-format conversion, composed through the IR (request parsing is
  already implemented in v1).
- Audio/video/document content blocks; image generation.
- Structured-output schema subset validation (deliberately absent today).
