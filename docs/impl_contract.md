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
`OpenAiChatCompletionsOptions.inject_include_usage` (default true). CC
parse consumes `model`/`stream`/`stream_options` (configuration, not IR
data); anything in `stream_options` other than a literal
`include_usage: true` — the only member configuration can rebuild — is
dropped with a cosmetic `StreamOptionsDropped` warning (not mirrored into
`extra` — a rebuilt unary body must not carry a bare `stream_options`).

## Thinking provenance (pins § 4.4)

- A `Thinking` block is **native** to target format F iff its `extra` has
  a **non-empty** namespace F, or it has a signature and no non-empty
  format namespace at all (optimistic replay: provenance unknowable,
  upstream validates signatures authoritatively). An empty namespace
  carries no provenance — `namespace_mut` creates on demand and
  `Extra::is_empty` already treats all-empty as empty, so all four
  formats test non-emptiness. Parsers must store any structure needed for
  reconstruction in their namespace (Responses `id`/`summary`; Anthropic
  `redacted: true` for `redacted_thinking`, whose `data` goes to
  `signature`; Google tool-call-part `thoughtSignature` rides the
  `ToolCall` block's `extra["google_generate_content"]["thoughtSignature"]`).
- Native → reconstruct the provider structure. Foreign (has another
  format's namespace, or plaintext-only where F validates signatures) →
  drop + `ThinkingDropped` (semantic), unless `thinking_as_text` → emit
  `text` into F's thinking-text channel (CC `reasoning_content` — the
  configured `reasoning_field`; Anthropic `thinking` block without
  signature; Google `thought: true` part; Responses `content: [{type:
  "reasoning_text"}]` — the official raw-CoT channel), adding
  `ThinkingSignatureDropped` (semantic) when a signature existed.
- Plaintext-only thinking (no signature, no namespace) is native to CC
  (its channel is a plaintext wire field, by default `reasoning_content`);
  on the other three it is foreign (see above).
- The CC thinking-channel field name comes from
  `OpenAiChatCompletionsOptions.reasoning_field` (default
  `reasoning_content`) and is the single authority on both sides: under a
  custom name the wire `reasoning_content` demotes to an ordinary unknown
  field (message extra on parse, stream-delta leftover fold), and the
  configured field is consumed from the unknown-field map instead of the
  typed one; warning locations and texts name the configured field.
  `validate_reasoning_field` rejects an empty name or a collision with
  `role`/`content`/`refusal`/`tool_calls`/`tool_call_id`/`name`/
  `function_call`/`audio` identically at build, parse and the stream
  parser's first `parse`. Options reach the parse/stream side through the
  defaulted `ApiFormat::{parse_response_with, parse_request_with,
  stream_parser_with}` — the client always calls the `_with` variants;
  count/models paths take no format options.
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
  message, blocks in item order; serialize explodes them again. Item-level
  reserved keys (`id`, `status`, `phase`, `item`) ride each Text block's
  `extra["openai_responses"]`. Serialize regroups adjacent Text blocks into
  one `message` item by identity key: a block with an `id` groups by that
  id alone (id is the item's identity; item-level fields follow the first
  block, and a later conflicting value drops with `ExtraDropped`,
  semantic, located at the item-level field's pointer — `/input/N/phase`,
  one warning per nested `item` field — so an explicit `extra` write to
  that pointer marks it overridden and passes the strict gate; the
  item-field restoration itself is parse-side backfill and does not enter
  the merge log); id-less blocks group only when `status`/`phase`/`item`
  are all equal — identical-metadata id-less items merge warning-free
  (§ 1 tier 2), differing metadata keeps the item boundary.
- CC parse keeps leading `system` messages in-array (no hoisting to
  `Request.system`); CC serialize inserts `Request.system` at the front
  as a `system` message. Anthropic/Google parse map the top-level
  system channel to `Request.system` — unless it contains non-text
  entries, in which case the whole channel parses into a marker-less
  leading `System` message (`Text` + own-format `Opaque`; the § 7.1
  combine rule hoists it back on serialization, `Request.system` stays
  Text-only).

## Response parsing

- Always run `normalize_stop_reason` (core) last — on non-streaming
  parses and in the accumulator.
- Refusal content (CC `refusal` field/delta, Responses refusal parts)
  parses into a `Text` block with
  `extra[<id>]["__llm_api_refusal"] = true` (the `ir::REFUSAL_MARKER`
  internal key — prefixed so a stray wire field named `refusal` stays
  plain unknown data and replays verbatim instead of rewriting the block).
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
  the first CC `[DONE]`, comment keep-alives) is silently consumed.
  Post-terminal input — any frame after the protocol terminator, including
  error frames and duplicate terminators — surfaces as `Unknown` +
  `UnknownStreamEvent` ("chunk received after the stream terminated") on
  all four formats: the response is already complete and must neither
  mutate nor fail. Independently, `Accumulator::push` ignores
  post-`MessageStop` events other than `Unknown` (first stop wins, no
  extra warning, item warnings still folded). The stream handle mirrors
  this past the parser's `MessageStop`: transport errors, SSE-level
  failures (size cap, decode) and stream-parser errors downgrade to one
  cosmetic `PostTerminalStreamFailure` warning delivered on the
  end-of-stream carrier instead of failing the stream. SSE events parsed
  before an in-chunk failure are delivered ahead of the failure (partial
  push), so a terminator fused with the failing bytes in one transport
  chunk still terminates the protocol. A cap error resets the SSE parser:
  buffered input and the half-built event are discarded (the error's
  `prefix` is the only retained copy), and subsequent pushes start at a
  fresh line boundary.
- Responses terminal events (`response.completed` / `response.incomplete`)
  carry the full final response; it reconciles the stream: items that
  never saw `output_item.done` finalize from the snapshot (matched by
  recorded id, falling back to `output_index`; an id mismatch rejects the
  positional match), items absent from the snapshot close with
  `block: None` (accumulated content stands), snapshot-only
  never-announced items synthesize start+stop — appended after all
  streamed blocks, so a synthesized item whose snapshot position precedes
  already-announced content warns `BlockOrderLost` (semantic, at most one
  per termination). Usage/stop-reason handling is unchanged; a compliant
  stream is unaffected.
- Responses streamed unknown message content parts wrap in a minimal
  assistant `message` shell (`{"type": "message", "role": "assistant",
  "content": [<part>]}`) as the block's Opaque value; the shell carries
  the original item identity (id/status/phase + unknown item fields under
  `item`) in the internal `__llm_api_item` marker key — the known
  `item_id` at BlockStart, replaced by the full identity when
  `output_item.done` / terminal reconciliation finalizes it. The marker
  never reaches the wire. On serialization a marked shell whose identity
  key (same rules as `item_group_key`) matches the current Text group —
  or opens an empty one — inlines its parts into that item, fully
  restoring the original item boundary with no warning; a mismatched
  identity serializes the shell as its own item (marker stripped,
  identity fields written back) + `ItemBoundaryLost` (semantic).
  Marker-less own-format Opaques stay verbatim top-level items (the user
  Opaque contract). Non-streaming keeps the whole item as one Opaque.
- CC streamed unknown delta fields (and legacy `function_call`) fold into
  the open block's `extra["openai_chat_completions"]` at `BlockStop` with
  delta merge conventions: strings concatenate, arrays append, objects
  merge recursively, anything else last-wins; `BlockDelta::Other` still
  surfaces each raw fragment in real time. Anthropic's unknown delta
  *types* on a known block emit `Other` + `MalformedField` and nothing
  folds at stop (nothing known to fold — disclosed, not silent).
- CC assistant `Text` blocks reserve two internal namespace keys
  (`text_block_reserved_key`, both `__llm_api_`-prefixed so wire fields
  named `refusal`/`message` cannot collide): `__llm_api_refusal` marks a
  refusal part (§ 9); `__llm_api_message` nests an object of
  message-level fields — the streaming parser folds unknown delta fields
  there, and serialization merges the object into the containing wire
  message in block order (later keys win) before `Message.extra`. All
  other Text-block namespace keys are part-level; a lone
  `__llm_api_message` key does not block the single-block `content`
  string shorthand.
- CC `Thinking` block extras have no dedicated wire object
  (`reasoning_content` is a plain string field) and merge wholesale into
  the containing assistant message on serialization; streamed unknown
  delta fields folded on the reasoning channel therefore stay top-level
  on the Thinking extra.
- Malformed tool payloads: a unified field (`id`, `name`, `arguments`,
  `tool_call_id`, `response`) missing or mangled — parsed with
  empty/`None` stand-ins, dropped, or kept verbatim in a state the
  source format cannot re-serialize (non-object wire arguments in IR
  `arguments`: § 4.5 rejects them on rebuild) — warns
  `MalformedToolCall` / `MalformedToolResult` (semantic).
  Verbatim-preserving recoveries that the source format **can**
  re-serialize (out-of-schema values mirrored into `extra`, whole nodes
  kept as `Opaque`) stay `MalformedField` (cosmetic).
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
- Count-response parsing must not silently zero the core field. Google:
  a negative `totalTokens` is `Error::Parse`; a **missing** one (or `{}`)
  is the legal proto3 zero-omission encoding and parses to 0 (same
  reading as § 8 usage). Anthropic: a missing `input_tokens` is
  `Error::Parse` (not proto3 — missing ≠ zero); negatives fail serde.
  Responses: `input_tokens` is required on the wire — missing/negative
  fail serde (`Error::Parse`).

## Models list (pins § 13)

`build_models_request` → GET; OpenAI-family: single page (`cursor`
ignored, next cursor `None`); Anthropic: `after_id=<cursor>` +
`has_more`/`last_id` (`has_more: true` with a missing/empty `last_id` is
`Error::Parse` — malformed pagination must not silently truncate the
list); Google: `pageSize=1000` + `pageToken=<cursor>` /
`nextPageToken`, `models/` prefix stripped from ids. `created`: OpenAI
Unix seconds via `models::system_time_from_unix_seconds`, Anthropic RFC
3339 via `models::system_time_from_rfc3339`; parse failure → `None` +
`MalformedField` warning is *not* needed (silent degrade, § 13).
Client-side, `list_models` aborts with `Error::Parse` (malformed
pagination) on a repeated cursor **and** past `Limits.max_model_pages`
(default 1000) — a fresh cursor on every page never repeats one.

## Misc pins

- `Request.system` allows only `Text` blocks — anything else (incl.
  `Opaque`) is `ConversionError::InvalidBlockForRole` with role `System`.
  In-array messages follow the § 7.4 table exactly.
- Unknown wire roles: CC, Anthropic and Responses keep the whole wire
  message/item as a lone own-format `Opaque` (in a `User`-role IR message
  for CC/Anthropic; in the assistant run for Responses) + `MalformedField`
  warning; serialize re-emits it verbatim (a merge barrier on Anthropic).
  Google's role union is closed (`user`/`model`, optional), so anything
  else is treated as `user` per upstream semantics — an out-of-set value
  still warns `MalformedField`.
- Usage arithmetic saturates (`saturating_add`) — misbehaving provider
  data must not overflow (§ 8's saturating precedent).
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
- URL query comparisons percent-decode once, byte-wise, before matching:
  protected-key checks and `extra_query` same-key replacement decode the
  URL's own raw keys (`%XX` only; malformed sequences stay verbatim; `+`
  is a literal — RFC 3986 query, not form encoding); the URL keeps its
  original spelling, only values are replaced. `extra_query` keys are
  logical values the library encodes itself (`%` → `%25`), so encoding
  cannot smuggle a protected key past the check. A key `extra_query` sets
  appears **exactly once** in the final URL: the first logical occurrence
  (percent-decode comparison) keeps its original spelling and position
  and receives the new value; all later logical duplicates of that key
  are dropped. Duplicate query keys the caller does **not** set are
  preserved verbatim (base-query passthrough). Within one merged
  `extra_query` list, later same-name entries still win.
- Anthropic `max_tokens`/`top_k` above `u32::MAX`: the IR field stays
  unset + `MalformedField` (cosmetic — the original value mirrors into
  `extra` and re-serializes verbatim); rebuilding such a request needs
  `default_max_tokens` (the extra merge then restores the original value)
  or fails `MissingRequired`. The other three formats' wire types are
  `u32`, so the same input fails their whole parse — a documented
  asymmetry.
- Google requests accept proto3 snake_case aliases on multi-word modeled
  fields, canonicalized to camelCase on re-serialization (§ 1 tier 2,
  warning-free); both spellings in one payload are a hard duplicate-field
  parse error (serde reports the primary camelCase name).
- Non-finite sampling values (§ 4.6): `temperature` / `top_p` /
  `frequency_penalty` / `presence_penalty` set to NaN or ±∞ fail every
  build path (chat, typed `request_from_ir`, count-tokens) with
  `ConversionError::NonFiniteNumber`, regardless of strict mode and even
  on formats that would drop the field with a warning — JSON cannot
  represent them, serde_json would silently write `null`, and there is no
  faithful degrade. `location` is the field's pointer in the would-be
  final body (`/temperature`, …; Google `/generationConfig/temperature`).
  Parse-side entry is impossible (JSON has no non-finite literals;
  `Value` is NaN-free), so the only entry is user-set IR fields; the IR's
  own serde round-trip is not guarded (documented on the fields).
- Do not implement `include_raw` in parsers — raw attachment is the
  client's job; `StreamParser` just returns events + warnings.

## Testing bar

- Field-mapping unit tests per § 4.5–4.9 tables (every cell).
- Round-trip idempotence per § 1: `serialize(parse(J)) == J` for
  canonical fixtures; `parse(serialize(ir))` preserves modeled fields,
  `extra`, `Opaque` nodes, message order.
- Stream fixtures: complete SSE sessions (interleaved thinking/text,
  tool-argument fragments, blocked prompts, multi-candidate, truncated
  stream ⇒ `finish()` returns `Error::TruncatedStream`). Put fixture
  files under `tests/fixtures/<id>/`.
- `cargo test --all-features` and `cargo clippy --all-features
  --all-targets` clean (missing docs are warnings — write the docs).
