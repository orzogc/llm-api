# Implementation contract (v1)

Binding decisions shared by all format implementations and the client, on
top of `docs/design.md`. Where this file pins a choice the design document
left open, follow this file; report (do not silently resolve) any conflict
with the official API docs.

## Module layout per format

```
src/formats/<id>/
  mod.rs      docs, re-exports, the ApiFormat struct + impl
  types.rs    complete serde wire types
  from_ir.rs  IR -> request JSON (build side)
  to_ir.rs    request -> IR, response -> IR (parse side)
  stream.rs   StreamParser impl
```

Integration tests in `tests/<id>.rs` (+ more files prefixed `<id>_`),
fixtures under `tests/fixtures/<id>/`.

## Wire types

- Every wire object type carries `#[serde(flatten)] pub extra:
  serde_json::Map<String, Value>` to capture unknown fields on parse.
- Optional fields: `#[serde(default, skip_serializing_if = "…")]`.
- Parse creates IR extras with `Extra::from_unknown(<id>, flattened_map)`
  (drops null-valued unknowns — the documented § 1 loss). Unknown keys
  nested in known sub-objects are stored at mirrored paths (e.g.
  `extra[id] = {"generationConfig": {"unknownKey": …}}`) so the RFC 7396
  merge puts them back.

## Build pipeline (IR → request), § 6 order

1. Convert each IR node to its wire shape, `serde_json::to_value` it, then
   `node.extra.merge_into(<id>, &mut value, <absolute JSON pointer of the
   node in the final body>, &mut merge_log)`. Assemble bottom-up. An IR
   `Tool`-role message that produces several wire messages merges its
   message-level extra into the **first** produced wire message.
2. Merge `request.extra[<id>]` into the whole body with base `""` last.
3. Collect `ConversionWarning`s using the exact `WarningCode`s from
   `src/convert/warnings.rs`; `location` is a JSON pointer into the final
   body (for a dropped field: the pointer where it would naturally live,
   e.g. `/top_k` on CC, `/generationConfig/thinkingConfig` for a Google
   thinking warning).
4. `finalize_request(&mut body, &mut warnings, &merge_log, ctx.convert
   .strict, &ctx.hooks, &message_pointers)` — `message_pointers` =
   `(pointer, Role)` per serialized message in order; wire roles map to IR
   `Role` (`model` → `Assistant`, `tool`/legacy `function` → `Tool`).
   Top-level system channels are not visited.
5. `build_url(&ctx.url, <path template>, &ctx.model, <method>, <protected
   query>, &ctx.extra_query)`.
6. `BuiltRequest { method: POST, url, headers: format defaults +
   content-type, body, auth: Some(<default scheme>), warnings }`.

Path templates and auth defaults:

| format | chat | count | models | auth |
|---|---|---|---|---|
| CC | `chat/completions` | — (`NotSupported`) | `models` | `AuthScheme::bearer()` |
| Responses | `responses` | `responses/input_tokens` | `models` | bearer |
| Anthropic | `messages` | `messages/count_tokens` | `models` | `x-api-key` or bearer per `AnthropicOptions.auth_style`; plus `anthropic-version` (unless `None`) and `anthropic-beta` (betas joined `,`) headers |
| Google | `models/{model}:{method}`, method `generateContent` / `streamGenerateContent` (+ protected query `alt=sse` when streaming) | `models/{model}:countTokens` | `models` | header `x-goog-api-key` |

Streaming: CC/Responses/Anthropic set `"stream": true` in the body; CC
additionally injects `stream_options: {"include_usage": true}` when
`OpenAiChatCompletionsOptions.inject_include_usage` (default true).

## Thinking provenance (pins § 4.4)

- A `Thinking` block is **native** to target format F iff its `extra` has
  namespace F, or it has a signature and *no* format namespace at all
  (optimistic replay: provenance unknowable, upstream validates
  signatures authoritatively). Parsers must store any structure needed for
  reconstruction in their namespace (Responses `id`/`summary`; Anthropic
  `redacted: true` for `redacted_thinking`, whose `data` goes to
  `signature`; Google tool-call-part `thoughtSignature` rides the
  `ToolCall` block's `extra["google_generate_content"]["thoughtSignature"]`).
- Native → reconstruct the provider structure. Foreign (has another
  format's namespace, or plaintext-only where F validates signatures) →
  drop + `ThinkingDropped` (semantic), unless `thinking_as_text` → emit
  `text` into F's thinking-text channel (CC `reasoning_content`; Anthropic
  `thinking` block without signature; Google `thought: true` part;
  Responses `content: [{type: "reasoning_text"}]` — the official raw-CoT
  channel), adding `ThinkingSignatureDropped` (semantic) when a signature
  existed.
- Plaintext-only thinking (no signature, no namespace) is native to CC
  (its channel is plaintext `reasoning_content`); on the other three it is
  foreign (see above).
- "Request enables thinking" (for § 7.3 warnings) := `reasoning` present
  and (`enabled == Some(true)`, or `enabled` unset and `effort` set to
  something other than `Effort::None`).
- `fill_missing_thinking`: insert `Thinking { text: Some(s) }` at the
  start of the offending assistant message + `MissingThinkingFilled`
  (cosmetic); otherwise warn `MissingThinkingWithToolCalls` (semantic).

## Orphan tool calls (pins § 7.3)

- A `ToolCall` is *matched* iff a later message contains a `ToolResult`
  with the same `tool_call_id` (when the call has no id: same `name`).
- *Trailing* orphans are orphans in the **last** message of the array;
  all others are mid-array → `OrphanToolCalls` (semantic), never repaired.
- `DropTrailing`: remove those blocks (and the message if it becomes
  empty) + `OrphanToolCallsDropped` (cosmetic); if the message keeps a
  `Thinking` block after all its tool calls were dropped →
  `ThinkingOrphaned` (semantic).
- `SynthesizeError`: append one `Tool` message after the last message
  containing, per orphan, `ToolResult { tool_call_id, name, is_error:
  Some(true), content: [Text "cancelled"] }` + `OrphanToolCallsSynthesized`
  (cosmetic).

## Turn grouping (pins § 7.2 / § 4.2 round-trip meta)

Round-trip metadata flows **parse-attach → serialize-consume**.

- Anthropic/Google parse: a wire user turn mixing `tool_result` /
  `functionResponse` parts with other content splits into runs
  (consecutive tool results → `Tool` message, consecutive others → `User`
  message), every produced message tagged `with_turn_group(n)` with the
  same fresh `n` per wire turn.
- Serialize: adjacent messages whose `turn_group_id()` are equal merge
  back into one wire user turn (blocks concatenated in message order).
  Google **additionally always** merges adjacent `Tool`/`User` messages
  (alternation is required there); Anthropic without meta keeps them as
  separate `user` messages (upstream combines consecutive same-role
  turns) unless `merge_consecutive_roles` is on.
- CC: each wire `tool` message parses to its own IR `Tool` message (one
  `ToolResult`); an IR `Tool` message with N results serializes to N wire
  messages. No meta.
- Responses parse: consecutive assistant-side items (`reasoning`,
  assistant `message`, `function_call`) group into **one** IR assistant
  message, blocks in item order; serialize explodes them again. Item ids
  ride each block's `extra["openai_responses"]["id"]`.
- CC parse keeps leading `system` messages in-array (no hoisting to
  `Request.system`); CC serialize inserts `Request.system` at the front
  as a `system` message. Anthropic/Google parse map the top-level
  system channel to `Request.system`.

## Response parsing

- Always run `normalize_stop_reason` (core) last — on non-streaming
  parses and in the accumulator.
- Refusal content (CC `refusal` field/delta, Responses refusal parts)
  parses into a `Text` block with `extra[<id>]["refusal"] = true`.
- Multi-choice/candidate: read the first, warn `MultipleCandidates`
  (semantic, parse side); in streams, chunks for candidate index > 0
  surface as `Unknown` with the warning emitted **once per stream**.
- Usage unification (§ 8): Anthropic `input_tokens` += cache read+write;
  Google `output_tokens` = `candidatesTokenCount + thoughtsTokenCount`.
  Keep the provider object in `Usage.raw`.
- Google blocked prompt (no candidates, `promptFeedback.blockReason`):
  empty assistant message, `stop_reason: Some(ContentFilter)`, body in
  `raw`; streaming: `MessageStart` (if needed) + `MessageDelta {
  ContentFilter }` + `MessageStop`.
- Unknown stream events → `StreamEvent::Unknown` + `UnknownStreamEvent`
  (cosmetic) warning; known-ignorable protocol noise (Anthropic `ping`,
  CC `[DONE]` handling, comment keep-alives) is silently consumed.
- `parse_error`: map provider error types to `ApiErrorKind` (OpenAI
  `error.type`; Anthropic `error.type` incl. `overloaded_error` →
  `Overloaded`; Google `error.status` gRPC-style codes, e.g.
  `RESOURCE_EXHAUSTED` → `RateLimit`, `UNAVAILABLE` → `Overloaded`);
  fall back to status classification; extract `retry_after` via
  `retry_after_from_headers`.

## Count tokens (pins § 13)

Build the full chat body first (extra + strict gate + hooks, exactly as
for `send` — reuse the chat build path with `CallMode::Unary`), snapshot
the set of top-level keys the **converter itself generated** (before extra
merge and hooks), then adapt:

- Drop keys outside the count endpoint's accepted set. A dropped key that
  was generated → silent; a dropped key that was *not* generated (came
  from `extra`, hooks or a dialect) → `CountTokensFieldDropped`
  (semantic).
- Google wraps the body in `{"generateContentRequest": {…, "model":
  "models/<model>"}}`; Anthropic filters to its accepted fields;
  Responses maps onto the `input_tokens` body.
- Warnings ride `BuiltRequest.warnings`; the client copies them into
  `TokenCount.warnings`.

## Models list (pins § 13)

`build_models_request` → GET; OpenAI-family: single page (`cursor`
ignored, next cursor `None`); Anthropic: `after_id=<cursor>` +
`has_more`/`last_id`; Google: `pageSize=1000` + `pageToken=<cursor>` /
`nextPageToken`, `models/` prefix stripped from ids. `created`: OpenAI
Unix seconds via `models::system_time_from_unix_seconds`, Anthropic RFC
3339 via `models::system_time_from_rfc3339`; parse failure → `None` +
`MalformedField` warning is *not* needed (silent degrade, § 13).

## Misc pins

- `Request.system` allows only `Text` blocks — anything else (incl.
  `Opaque`) is `ConversionError::InvalidBlockForRole` with role `System`.
  In-array messages follow the § 7.4 table exactly.
- Empty `ToolResult.content` encodings: CC `content: ""`, Responses
  `output: ""`, Anthropic `content` omitted, Google `response: {}`;
  each parses back to the empty list; Google `{"output": ""}` parses to
  one empty `Text` block.
- CC legacy `max_tokens` on parse maps to IR `max_output_tokens`
  (serializes back as `max_completion_tokens` — documented
  canonicalization).
- Anthropic `metadata`: only `user_id` maps; other keys →
  `MetadataDropped` (cosmetic) listing the keys. Google: whole `metadata`
  → `MetadataDropped`.
- OpenAI cache breakpoints: content-part `prompt_cache_breakpoint:
  {"mode": "explicit"}`; a hint TTL adds `CacheTtlDropped` (cosmetic).
  Hints on `ToolCall` blocks → `CacheHintDropped` (cosmetic). Nested
  `ToolOutputBlock` hints → `CacheHintDropped` on every target in v1.
- Effort mapping strictly per the § 4.7 table; out-of-set →
  `EffortUnsupported` (semantic); `Effort::Other(s)` passes verbatim
  (no warning) on CC/Responses/Anthropic/Google alike.
- `enabled`/`effort` conflicts (`enabled: true` + `effort: None`, or
  `enabled: false` + effort ≠ `None`): effort wins + `ReasoningConflict`
  (cosmetic).
- `include_thoughts` on CC (both values) → `IncludeThoughtsUnsupported`
  (cosmetic).
- Assistant-role `Image` blocks: native channel only on Google
  (`inlineData` part); CC/Responses/Anthropic drop them with
  `ImageSourceUnsupported` (semantic).
- Do not implement `include_raw` in parsers — raw attachment is the
  client's job; `StreamParser` just returns events + warnings.

## Testing bar

- Field-mapping unit tests per § 4.5–4.9 tables (every cell).
- Round-trip idempotence per § 1: `serialize(parse(J)) == J` for
  canonical fixtures; `parse(serialize(ir))` preserves modeled fields,
  `extra`, `Opaque` nodes, message order.
- Stream fixtures: complete SSE sessions (interleaved thinking/text,
  tool-argument fragments, blocked prompts, multi-candidate, truncated
  stream ⇒ `finish()` error). Put fixture files under
  `tests/fixtures/<id>/`.
- `cargo test --all-features` and `cargo clippy --all-features
  --all-targets` clean (missing docs are warnings — write the docs).
