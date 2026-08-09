# Anthropic Messages API Reference

Development reference for the Messages API (`POST /v1/messages`), extracted from the official
API docs: [Messages API](https://platform.claude.com/docs/en/api/messages.md),
[Create a Message](https://platform.claude.com/docs/en/api/beta/messages/create.md),
[Streaming Messages](https://platform.claude.com/docs/en/build-with-claude/streaming.md).
Covers request, non-streaming response, and SSE streaming. Count-tokens and Batches endpoints are
out of scope.

Note: the source docs describe the full (beta-inclusive) surface with `Beta`-prefixed schema
names (`BetaMessage`, `BetaContentBlock`, ...). The prefix is a schema-naming artifact only; wire
field names and `type` discriminators below are what appear in JSON.

## 1. Endpoint and headers

```
POST https://api.anthropic.com/v1/messages
```

| Header | Required | Value / Description |
|---|---|---|
| `content-type` | yes | `application/json` |
| `x-api-key` | yes | API key |
| `anthropic-version` | yes | API version date, e.g. `2023-06-01` |
| `anthropic-beta` | no | Comma-separated beta feature flags (see below) |
| `anthropic-user-profile-id` | no | User profile ID to attribute the request to (requires the `user-profiles` beta) |

Known `anthropic-beta` values (strings; open set): `message-batches-2024-09-24`,
`prompt-caching-2024-07-31`, `computer-use-2024-10-22`, `computer-use-2025-01-24`,
`pdfs-2024-09-25`, `token-counting-2024-11-01`, `token-efficient-tools-2025-02-19`,
`output-128k-2025-02-19`, `files-api-2025-04-14`, `mcp-client-2025-04-04`,
`mcp-client-2025-11-20`, `dev-full-thinking-2025-05-14`, `interleaved-thinking-2025-05-14`,
`code-execution-2025-05-22`, `extended-cache-ttl-2025-04-11`, `context-1m-2025-08-07`,
`context-management-2025-06-27`, `model-context-window-exceeded-2025-08-26`, `skills-2025-10-02`,
`fast-mode-2026-02-01`, `output-300k-2026-03-24`, `user-profiles-2026-03-24`,
`advisor-tool-2026-03-01`, `managed-agents-2026-04-01`, `cache-diagnosis-2026-04-07`,
`dreaming-2026-04-21`, `thinking-token-count-2026-05-13`, `server-side-fallback-2026-06-01`,
`server-side-fallback-2026-07-01`, `fallback-credit-2026-06-01`, `fallback-credit-2026-07-01`,
`agent-memory-2026-07-22`.

## 2. Request body

### 2.1 Top-level parameters

| Field | Type | Required | Description |
|---|---|---|---|
| `model` | string | yes | Model ID or alias (open string enum: `claude-opus-5`, `claude-sonnet-5`, `claude-fable-5`, `claude-opus-4-6`, `claude-haiku-4-5`, dated IDs, ...) |
| `messages` | array of MessageParam | yes | Input conversation (see 2.2). Max 100,000 messages per request |
| `max_tokens` | number | yes | Max tokens to generate (hard cap; model may stop earlier). `0` pre-warms the prompt cache without generating |
| `system` | string \| array of text blocks | no | System prompt. Array form allows `cache_control` and `citations` per block |
| `metadata` | object | no | `{ user_id?: string }` — opaque external user ID (no PII) |
| `stop_sequences` | array of string | no | Custom stop strings; on match, `stop_reason: "stop_sequence"` and `stop_sequence` is set |
| `stream` | boolean | no | Enable SSE streaming (see section 6) |
| `temperature` | number | no | Randomness, `0.0`–`1.0`, default `1.0`. Not fully deterministic even at `0.0` |
| `top_k` | number | no | Sample only from top K options (advanced) |
| `top_p` | number | no | Nucleus sampling cutoff (advanced) |
| `tools` | array of ToolUnion | no | Tool definitions (see 2.5) |
| `tool_choice` | ToolChoice | no | How the model may use tools (see 2.6) |
| `thinking` | ThinkingConfig | no | Extended thinking config (see 2.7) |
| `cache_control` | `{type:"ephemeral", ttl?}` | no | Top-level shortcut: marks the last cacheable block in the request |
| `container` | string \| `{id?, skills?}` | no | Code-execution container reuse; object form loads skills `{skill_id, type:"anthropic"\|"custom", version?}` |
| `context_management` | object | no | Context edit strategies (see 2.8) |
| `mcp_servers` | array | no | MCP servers: `{name, type:"url", url, authorization_token?, tool_configuration?:{allowed_tools?, enabled?}}` |
| `output_config` | object | no | `{effort?, format?, task_budget?}` — see 2.9 |
| `output_format` | JSONOutputFormat | no | Deprecated; use `output_config.format` |
| `service_tier` | `"auto"` \| `"standard_only"` | no | Priority vs standard capacity |
| `speed` | `"standard"` \| `"fast"` | no | Inference speed mode; `fast` is premium-priced and model-dependent (invalid combos rejected) |
| `inference_geo` | string | no | Geographic region for inference; defaults to workspace `default_inference_geo` |
| `diagnostics` | `{previous_message_id?: string\|null}` | no | Opt-in prompt-cache-miss diagnostics vs a prior response ID |
| `fallbacks` | `"default"` \| array | no | Server-side fallback models on policy declines. Entries: `{model, max_tokens?, output_config?, speed?, thinking?}`, tried in order |
| `fallback_credit_token` | string \| `{token, mode?:"strict"\|"best_effort"}` | no | Redeem a refusal's credit token on retry (object form needs `fallback-credit-2026-07-01`) |

### 2.2 `messages`

Each entry: `{ "role": "user" | "assistant" | "system", "content": string | ContentBlockParam[] }`.

- Models operate on alternating `user`/`assistant` turns; consecutive same-role messages are
  combined into one turn. The `"system"` role is for mid-conversation system content (top-level
  `system` remains the primary system prompt).
- Mid-conversation `system` messages are GA (no beta header) but **model-gated** (verified
  2026-08 against the mid-conversation-system-messages feature page): supported on Claude
  Opus 4.8, Opus 5, Fable 5 and Mythos 5; **not** supported on Sonnet 5 or older models, which
  reject them with a 400 (`role 'system' is not supported on this model`). Placement is also
  validated (not first; must follow a user turn — or an assistant turn ending in a server tool
  result; must precede an assistant turn or end the array; never between `tool_use` and its
  `tool_result`) — violations return a 400. Note: the official API reference's prose still
  contains an outdated "there is no `system` role for input messages" sentence contradicting
  its own schema; the schema and the feature page are authoritative.
- String `content` is shorthand for `[{"type": "text", "text": ...}]`.
- If the final message has role `assistant`, the response continues directly from its content
  (prefill; model-dependent availability).

```json
{"model": "claude-opus-4-6", "max_tokens": 1024,
 "messages": [
   {"role": "user", "content": "Hello there."},
   {"role": "assistant", "content": "Hi, I'm Claude. How can I help?"},
   {"role": "user", "content": [{"type": "text", "text": "Explain LLMs."}]}
 ]}
```

### 2.3 Request content block types

All request blocks accept an optional `cache_control: {type: "ephemeral", ttl?: "5m"|"1h"}`
(default TTL `5m`) unless noted. Fields marked `?` are optional.

| `type` | Fields | Notes |
|---|---|---|
| `text` | `text`, `citations?` | `citations` is an array of citation objects (see 2.4) |
| `image` | `source` | Source union: `{type:"base64", media_type:"image/jpeg"\|"image/png"\|"image/gif"\|"image/webp", data}` \| `{type:"url", url}` \| `{type:"file", file_id}` |
| `document` | `source`, `citations?:{enabled?}`, `context?`, `title?` | Source union: base64 PDF `{type:"base64", media_type:"application/pdf", data}`, plain text `{type:"text", media_type:"text/plain", data}`, content `{type:"content", content: string\|(text\|image blocks)[]}`, `{type:"url", url}`, `{type:"file", file_id}` |
| `search_result` | `content: text[]`, `source: string`, `title: string`, `citations?:{enabled?}` | RAG-style result block; citable |
| `thinking` | `thinking`, `signature` | Echo back from a prior response unchanged (no `cache_control`) |
| `redacted_thinking` | `data` | Opaque; echo back unchanged (no `cache_control`) |
| `tool_use` | `id`, `name`, `input: object`, `caller?` | Echoed assistant tool call. `caller`: `{type:"direct"}` \| `{type:"code_execution_20250825", tool_id}` \| `{type:"code_execution_20260120", tool_id}` |
| `tool_result` | `tool_use_id`, `content?`, `is_error?: bool` | `content`: string or array of `text` \| `image` \| `search_result` \| `document` \| `tool_reference` blocks |
| `tool_reference` | `tool_name` | Reference to a declared tool (used inside `tool_result` content) |
| `server_tool_use` | `id`, `name`, `input`, `caller?` | `name`: `advisor`, `web_search`, `web_fetch`, `code_execution`, `bash_code_execution`, `text_editor_code_execution`, `tool_search_tool_regex`, `tool_search_tool_bm25` |
| `web_search_tool_result` | `tool_use_id`, `content`, `caller?` | `content`: array of `{type:"web_search_result", title, url, encrypted_content, page_age?}` or error `{type:"web_search_tool_result_error", error_code}` |
| `web_fetch_tool_result` | `tool_use_id`, `content`, `caller?` | `content`: `{type:"web_fetch_result", url, content: document block, retrieved_at?}` or error `{type:"web_fetch_tool_result_error", error_code}` |
| `code_execution_tool_result` | `tool_use_id`, `content` | `content`: `{type:"code_execution_result", stdout, stderr, return_code, content:[{type:"code_execution_output", file_id}]}`, encrypted variant `{type:"encrypted_code_execution_result", encrypted_stdout, ...}`, or error `{type:"code_execution_tool_result_error", error_code}` |
| `bash_code_execution_tool_result` | `tool_use_id`, `content` | Same shape with `bash_code_execution_*` type names; extra error code `output_file_too_large` |
| `text_editor_code_execution_tool_result` | `tool_use_id`, `content` | `content`: `..._view_result` `{content, file_type:"text"\|"image"\|"pdf", num_lines?, start_line?, total_lines?}`, `..._create_result` `{is_file_update}`, `..._str_replace_result` `{lines?, new_lines?, new_start?, old_lines?, old_start?}`, or `..._tool_result_error` `{error_code, error_message?}` |
| `tool_search_tool_result` | `tool_use_id`, `content` | `content`: `{type:"tool_search_tool_search_result", tool_references:[tool_reference]}` or error |
| `advisor_tool_result` | `tool_use_id`, `content` | `content`: `{type:"advisor_result", text, stop_reason?}`, `{type:"advisor_redacted_result", encrypted_content, stop_reason?}` (round-trip verbatim), or `{type:"advisor_tool_result_error", error_code}` |
| `mcp_tool_use` | `id`, `name`, `server_name`, `input` | MCP tool invocation |
| `mcp_tool_result` | `tool_use_id`, `content?`, `is_error?` | `content`: string or array of `text` blocks |
| `container_upload` | `file_id` | File placed into the container's input directory |
| `compaction` | `content?`, `encrypted_content?` | Round-trip from responses to keep context across compaction; `content: null` = failed compaction (no-op) |
| `mid_conv_system` | `content` | Mid-conversation system instructions: array of `text`, `tool_addition`, `tool_removal` blocks |
| `tool_addition` / `tool_removal` | `tool` | Surfaces/withdraws a declared tool from this point on. `tool`: `{type:"tool_reference", name}` \| `{type:"mcp_tool_reference", name, server_name}` \| `{type:"mcp_toolset_reference", server_name}` |
| `fallback` | `from:{model}`, `to:{model}`, `trigger?` | Echoed from a prior fallback response; keep in original position (marks the model-boundary; misplacement around thinking runs is rejected) |

Tool-use error codes (request/response mirror): web_search — `invalid_tool_input`, `unavailable`,
`max_uses_exceeded`, `too_many_requests`, `query_too_long`, `request_too_large`; web_fetch —
`invalid_tool_input`, `url_too_long`, `url_not_allowed`, `url_not_in_prior_context`,
`url_not_accessible`, `unsupported_content_type`, `too_many_requests`, `max_uses_exceeded`,
`unavailable`; code_execution — `invalid_tool_input`, `unavailable`, `too_many_requests`,
`execution_time_exceeded` (+ `output_file_too_large` for bash, `file_not_found` for text editor);
advisor — `max_uses_exceeded`, `prompt_too_long`, `too_many_requests`, `overloaded`,
`unavailable`, `execution_time_exceeded`, `model_not_found`.

### 2.4 Citations (on request `text` blocks)

`citations` entries are a union on `type`:

| `type` | Fields |
|---|---|
| `char_location` | `cited_text`, `document_index`, `document_title`, `start_char_index`, `end_char_index` |
| `page_location` | `cited_text`, `document_index`, `document_title`, `start_page_number`, `end_page_number` |
| `content_block_location` | `cited_text`, `document_index`, `document_title`, `start_block_index`, `end_block_index` (exclusive) |
| `web_search_result_location` | `cited_text`, `encrypted_index`, `title`, `url` |
| `search_result_location` | `cited_text`, `source`, `title`, `search_result_index`, `start_block_index`, `end_block_index` |

Response citations use the same shapes plus a `file_id` field on the document-based variants
(see 3.3). Enable citations per `document`/`search_result` block via `citations: {enabled: true}`.

### 2.5 `tools`

Union of a custom tool and Anthropic-defined tool types. Common optional fields on every variant
except `mcp_toolset` (which only has `cache_control`): `cache_control`, `defer_loading` (exclude
from initial prompt until surfaced via tool search / `tool_addition`), `strict` (guarantee schema
validation of names/inputs), and `allowed_callers` (array of `"direct"`,
`"code_execution_20250825"`, `"code_execution_20260120"`, `"code_execution_20260521"` — who may
invoke the tool). `input_examples` (array of example input objects) exists only on custom, bash,
text-editor, computer, and memory tools.

Custom tool:

```json
{
  "name": "get_stock_price",
  "description": "Get the current stock price for a ticker symbol.",
  "input_schema": {
    "type": "object",
    "properties": {"ticker": {"type": "string"}},
    "required": ["ticker"]
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | yes | Name used by the model in `tool_use` blocks |
| `input_schema` | object | yes | JSON Schema (draft 2020-12): `{type:"object", properties?, required?}` |
| `description` | string | no | Strongly recommended; drives tool selection quality |
| `type` | `"custom"` | no | Optional discriminator for custom tools |
| `eager_input_streaming` | boolean | no | Stream tool input incrementally (fine-grained tool streaming); `null` = per beta-header default. Custom tools only |
| plus common fields | | | `strict`, `defer_loading`, `allowed_callers`, `input_examples`, `cache_control` |

Anthropic-defined tools (declare by `type` + fixed `name`; no `input_schema`):

| `type` | `name` | Extra parameters |
|---|---|---|
| `bash_20241022` / `bash_20250124` | `bash` | — |
| `text_editor_20241022` / `text_editor_20250124` | `str_replace_editor` | — |
| `text_editor_20250429` | `str_replace_based_edit_tool` | — |
| `text_editor_20250728` | `str_replace_based_edit_tool` | `max_characters?` |
| `computer_20241022` / `computer_20250124` | `computer` | `display_width_px`, `display_height_px` (required), `display_number?` |
| `computer_20251124` | `computer` | as above + `enable_zoom?` |
| `code_execution_20250522` / `20250825` / `20260120` / `20260521` | `code_execution` | — (`20260120`/`20260521` have REPL state persistence) |
| `memory_20250818` | `memory` | — |
| `web_search_20250305` / `web_search_20260209` | `web_search` | `max_uses?`, `allowed_domains?` xor `blocked_domains?`, `user_location?: {type:"approximate", city?, region?, country?, timezone?}` |
| `web_search_20260318` | `web_search` | as above + `response_inclusion?: "full"\|"excluded"` |
| `web_fetch_20250910` / `web_fetch_20260209` | `web_fetch` | `max_uses?`, `allowed_domains?`/`blocked_domains?`, `citations?:{enabled?}`, `max_content_tokens?` |
| `web_fetch_20260309` | `web_fetch` | as above + `use_cache?` |
| `web_fetch_20260318` | `web_fetch` | as above + `response_inclusion?` |
| `advisor_20260301` | `advisor` | `model` (required; advisor model), `max_tokens?`, `max_uses?`, `caching?` (cache_control for the advisor's own prompt) |
| `tool_search_tool_bm25_20251119` | `tool_search_tool_bm25` | `type` also accepts `"tool_search_tool_bm25"` |
| `tool_search_tool_regex_20251119` | `tool_search_tool_regex` | `type` also accepts `"tool_search_tool_regex"` |
| `mcp_toolset` | — | `mcp_server_name` (required), `default_config?:{enabled?, defer_loading?}`, `configs?: map<tool name, {enabled?, defer_loading?}>` |

### 2.6 `tool_choice`

Union on `type`:

| Variant | Fields | Behavior |
|---|---|---|
| `{"type": "auto"}` | `disable_parallel_tool_use?` | Model decides (default). If disabled, at most one tool use |
| `{"type": "any"}` | `disable_parallel_tool_use?` | Model must use at least one tool. If disabled, exactly one |
| `{"type": "tool", "name": "..."}` | `disable_parallel_tool_use?` | Model must use the named tool |
| `{"type": "none"}` | — | Tool use forbidden |

`disable_parallel_tool_use` defaults to `false`.

### 2.7 `thinking`

Union on `type`. Thinking output counts toward `max_tokens`.

| Variant | Fields | Notes |
|---|---|---|
| `{"type": "enabled", "budget_tokens": n, "display"?}` | `budget_tokens` ≥ 1024 and < `max_tokens` | Manual thinking budget |
| `{"type": "adaptive", "display"?}` | — | Model decides when/how much to think |
| `{"type": "disabled"}` | — | No thinking |

`display`: `"summarized"` (default; thinking text returned) or `"omitted"` (thinking text
redacted; only a signature is returned for multi-turn continuity).

### 2.8 `context_management`

`{ "edits": [ ... ] }` — list of strategies, union on `type`:

| `type` | Fields |
|---|---|
| `clear_tool_uses_20250919` | `trigger?: {type:"input_tokens"\|"tool_uses", value}`, `keep?: {type:"tool_uses", value}`, `clear_at_least?: {type:"input_tokens", value}`, `clear_tool_inputs?: bool\|string[]`, `exclude_tools?: string[]` |
| `clear_thinking_20251015` | `keep?: {type:"thinking_turns", value} \| {type:"all"} \| "all"` |
| `compact_20260112` | `instructions?: string`, `pause_after_compaction?: bool`, `trigger?: {type:"input_tokens", value}` (default 150000) |

### 2.9 `output_config`

| Field | Type | Description |
|---|---|---|
| `effort` | `"low"`\|`"medium"`\|`"high"`\|`"xhigh"`\|`"max"` | Output effort level |
| `format` | `{type:"json_schema", schema: object}` | Structured output (replaces deprecated `output_format`) |
| `task_budget` | `{type:"tokens", total, remaining?}` | Total token budget across contexts; `remaining` defaults to `total` |

## 3. Non-streaming response

### 3.1 Message object

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique message ID (e.g. `msg_...`; format may change) |
| `type` | `"message"` | Object type |
| `role` | `"assistant"` | Always `assistant` |
| `model` | string | Model that handled the request |
| `content` | array of ContentBlock | Generated content (see 3.3). Continues an assistant prefill if the input ended with one |
| `stop_reason` | StopReason \| null | Why generation stopped (see 3.2). Non-null except in streaming `message_start` |
| `stop_sequence` | string \| null | Matched custom stop sequence, if `stop_reason` is `stop_sequence` |
| `stop_details` | object \| null | Structured refusal info (see 3.4) |
| `usage` | Usage | Billing/rate-limit token accounting (see 3.5) |
| `container` | object \| null | Code-execution container info: `{id, expires_at, skills:[{skill_id, type:"anthropic"\|"custom", version}]}` |
| `context_management` | object \| null | `{applied_edits: [...]}`: `{type:"clear_tool_uses_20250919", cleared_input_tokens, cleared_tool_uses}` or `{type:"clear_thinking_20251015", cleared_input_tokens, cleared_thinking_turns}` |
| `diagnostics` | object \| null | Present when request set `diagnostics`. `cache_miss_reason` union: `model_changed` / `system_changed` / `tools_changed` / `messages_changed` (each with `cache_missed_input_tokens`), `previous_message_not_found`, `unavailable`; `null` = diagnosis pending |

### 3.2 `stop_reason` values

| Value | Meaning |
|---|---|
| `end_turn` | Model reached a natural stopping point |
| `max_tokens` | Exceeded requested `max_tokens` or the model maximum |
| `stop_sequence` | A custom `stop_sequences` entry was generated (see `stop_sequence`) |
| `tool_use` | Model invoked one or more tools; run them and send `tool_result`s back |
| `pause_turn` | Long-running turn was paused; resend the response as-is to continue |
| `refusal` | Streaming classifiers intervened for a potential policy violation (see `stop_details`) |
| `model_context_window_exceeded` | The model's context window was exceeded |
| `compaction` | In the enum (not described in prose); associated with context compaction, e.g. `compact_20260112` with `pause_after_compaction` |

### 3.3 Response content block types

Union on `type` (same set streamed via `content_block_start`):

| `type` | Fields |
|---|---|
| `text` | `text`, `citations` (array \| null; same shapes as 2.4, document-based variants also carry `file_id`) |
| `thinking` | `thinking`, `signature` |
| `redacted_thinking` | `data` |
| `tool_use` | `id`, `name`, `input: object`, `caller?` |
| `server_tool_use` | `id`, `name` (enum as in 2.3), `input`, `caller?` |
| `web_search_tool_result` | `tool_use_id`, `content` (result array or error object), `caller?` |
| `web_fetch_tool_result` | `tool_use_id`, `content` (`web_fetch_result` with a `document` payload `{type:"document", source: base64-pdf\|plaintext, title, citations:{enabled}}`, `url`, `retrieved_at`; or error), `caller?` |
| `advisor_tool_result` | `tool_use_id`, `content` (`advisor_result {text, stop_reason}` \| `advisor_redacted_result {encrypted_content, stop_reason}` \| error). `stop_reason` uses the same values as the top-level field |
| `code_execution_tool_result` | `tool_use_id`, `content` (`code_execution_result {stdout, stderr, return_code, content:[{type:"code_execution_output", file_id}]}` \| `encrypted_code_execution_result` \| error) |
| `bash_code_execution_tool_result` | Same shape, `bash_code_execution_*` type names |
| `text_editor_code_execution_tool_result` | `tool_use_id`, `content` (view/create/str_replace result or error; field lists in 2.3) |
| `tool_search_tool_result` | `tool_use_id`, `content` (`tool_search_tool_search_result {tool_references:[{type:"tool_reference", tool_name}]}` or error) |
| `mcp_tool_use` | `id`, `name`, `server_name`, `input` |
| `mcp_tool_result` | `tool_use_id`, `is_error`, `content` (string or `text` block array) |
| `container_upload` | `file_id` |
| `compaction` | `content` (string \| null; null = failed compaction), `encrypted_content` (round-trip verbatim) |
| `fallback` | `from: {model}`, `to: {model}`, `trigger: {type:"refusal", category}` — marks the boundary where one model's output gives way to a fallback model's. Served-by signal is a `fallback_message` entry in `usage.iterations`, not this block |

### 3.4 `stop_details`

Present for refusals; type `refusal`.

| Field | Type | Description |
|---|---|---|
| `type` | `"refusal"` | Discriminator |
| `category` | enum \| null | `cyber`, `bio`, `frontier_llm`, `reasoning_extraction`, `general_harms` |
| `explanation` | string \| null | Human-readable, unstable text |
| `fallback_credit_token` | string \| null | Opaque code refunding cache-miss cost on a retry (expires 5 min); pass as request `fallback_credit_token` |
| `fallback_has_prefill_claim` | boolean | Whether the token may be redeemed with the appended-assistant (continuation) retry form |
| `recommended_model` | string \| null | Suggested direct-retry model when a fallback attempt could not run |

### 3.5 `usage`

Total input tokens = `input_tokens` + `cache_creation_input_tokens` + `cache_read_input_tokens`.
Counts do not map 1:1 to visible content (`output_tokens` is non-zero even for empty output).

| Field | Type | Description |
|---|---|---|
| `input_tokens` | number | Uncached input tokens |
| `output_tokens` | number | Output tokens (billing-authoritative, includes thinking) |
| `cache_creation_input_tokens` | number \| null | Tokens written to cache |
| `cache_read_input_tokens` | number \| null | Tokens read from cache |
| `cache_creation` | object \| null | `{ephemeral_5m_input_tokens, ephemeral_1h_input_tokens}` breakdown by TTL |
| `output_tokens_details` | object \| null | `{thinking_tokens}` — raw reasoning tokens (≤ `output_tokens`; display decomposition, not separately billed) |
| `server_tool_use` | object \| null | `{web_search_requests, web_fetch_requests}` |
| `service_tier` | `"standard"`\|`"priority"`\|`"batch"` \| null | Tier actually used |
| `speed` | `"standard"`\|`"fast"` \| null | Speed mode actually used |
| `inference_geo` | string | Region where inference ran |
| `iterations` | array \| null | Per-iteration breakdown, union on `type`: `message` (`input_tokens`, `output_tokens`, cache fields, `model`), `compaction` (same, no `model`), `advisor_message`, `fallback_message` |
| `fallback_credit` | object \| null | `{status: {type:"redeemed"} \| {type:"not_applied", reason, remove_to_redeem?}}`; `reason` enum: `body_mismatch`, `continuation_excluded`, `continuation_only`, `expired`, `invalid_target_model`, `not_enabled`, `reprice_unavailable`, `temporarily_unavailable`, `variant_fields_present`, `wrong_organization`, `wrong_platform`, `wrong_workspace` |

### 3.6 Example (trimmed)

```json
{
  "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
  "type": "message",
  "role": "assistant",
  "model": "claude-opus-4-6",
  "content": [{"type": "text", "text": "Hi! My name is Claude.", "citations": null}],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 2095,
    "output_tokens": 503,
    "cache_creation_input_tokens": 2051,
    "cache_read_input_tokens": 0,
    "cache_creation": {"ephemeral_5m_input_tokens": 2051, "ephemeral_1h_input_tokens": 0},
    "output_tokens_details": {"thinking_tokens": 0},
    "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
    "service_tier": "standard",
    "speed": "standard",
    "inference_geo": "global"
  }
}
```

## 4. Error responses

Errors use one envelope (non-2xx HTTP status, or an `error` SSE event mid-stream):

```json
{
  "type": "error",
  "error": {"type": "overloaded_error", "message": "Overloaded"},
  "request_id": "req_..."
}
```

`error.type` values: `invalid_request_error` (400), `authentication_error` (401),
`billing_error`, `permission_error` (403), `not_found_error` (404), `rate_limit_error` (429),
`timeout_error`, `api_error` (500), `overloaded_error` (529).

## 5. Streaming (SSE)

Set `"stream": true`. The response is a server-sent event stream; each event has an SSE
`event:` name matching the `type` field inside its `data:` JSON payload.

### 5.1 Event flow

1. `message_start` — a `Message` object with empty `content` (`stop_reason: null`).
2. A series of content blocks. Each block is a `content_block_start`, zero or more
   `content_block_delta` events, then `content_block_stop`. Each block carries an `index` equal
   to its position in the final `content` array. Blocks stream one at a time, in order.
   Exception: during server-side fallback, a `fallback` block arrives as a
   `content_block_start`/`content_block_stop` pair with no deltas.
3. One or more `message_delta` events — top-level changes to the final `Message`
   (`stop_reason`, `stop_sequence`, cumulative `usage`).
4. A final `message_stop` event.

`ping` events may appear anywhere in the stream. `error` events may replace normal flow
(e.g. `overloaded_error` mid-stream instead of HTTP 529). New event types may be added over
time — handle unknown types gracefully.

### 5.2 Event reference

| SSE event | Payload fields | Notes |
|---|---|---|
| `message_start` | `message: Message` | `content: []`, `stop_reason: null`; `usage` has initial `input_tokens` and a small `output_tokens` |
| `content_block_start` | `index`, `content_block` | `content_block` is any response block from 3.3 with empty/initial content (`{"type":"text","text":""}`, `{"type":"tool_use","id","name","input":{}}`, `{"type":"thinking","thinking":"","signature":""}`, complete `web_search_tool_result`, ...) |
| `content_block_delta` | `index`, `delta` | Delta union — see 5.3 |
| `content_block_stop` | `index` | Block at `index` is complete |
| `message_delta` | `delta`, `usage`, `context_management?` | `delta`: `{stop_reason, stop_sequence, container?, stop_details?}`. `usage` is **cumulative** (see 5.4) |
| `message_stop` | — | End of stream |
| `ping` | — | Keep-alive; ignore |
| `error` | `error: {type, message}` | Error envelope (section 4) |

### 5.3 `content_block_delta` delta types

Union on `delta.type`; a delta updates the block at `index`.

| `delta.type` | Fields | Applies to block | Semantics |
|---|---|---|---|
| `text_delta` | `text` | `text` | Append to `text` |
| `input_json_delta` | `partial_json` | `tool_use`, `server_tool_use`, `mcp_tool_use` | Partial JSON string fragments of `input`. Accumulate all fragments and parse after `content_block_stop`; the final `input` is an object. Fragments may be empty. Models emit one complete key/value at a time, chunked into multiple deltas, so gaps between events are normal |
| `thinking_delta` | `thinking`, `estimated_tokens?` | `thinking` | Append to `thinking`. `estimated_tokens` only with the `thinking-token-count-2026-05-13` beta (lossy progress hint when display is `omitted`) |
| `signature_delta` | `signature` | `thinking` | Sent once, just before the block's `content_block_stop`; signature verifies thinking-block integrity. With `display: "omitted"`, no `thinking_delta` is sent — the block opens, receives a single `signature_delta`, and closes |
| `citations_delta` | `citation` | `text` | Adds one citation object (shapes as in 3.3) to the block's `citations` |
| `compaction_delta` | `content`, `encrypted_content` | `compaction` | Streams a compaction block; `encrypted_content` round-trips verbatim |

### 5.4 `message_delta` details

- `delta.stop_reason` / `delta.stop_sequence`: final values (see 3.2).
- `delta.container` and `delta.stop_details` may appear (same shapes as 3.1/3.4);
  `context_management` may accompany the event.
- `usage` is cumulative for the whole message, not incremental: `output_tokens`,
  `input_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`,
  `output_tokens_details`, `server_tool_use`, `iterations`, `fallback_credit`.

### 5.5 Example: basic text stream

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_1nZd...","type":"message","role":"assistant","content":[],"model":"claude-opus-5","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":25,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: ping
data: {"type":"ping"}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"!"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":15}}

event: message_stop
data: {"type":"message_stop"}
```

### 5.6 Example: tool use (partial-JSON input streaming)

A text block (index 0) may precede the tool call; the `tool_use` block starts with `input: {}`
and its input arrives as `input_json_delta` fragments:

```
event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01T1...","name":"get_weather","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"location\":"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":" \"San Francisco,"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":" CA\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":89}}

event: message_stop
data: {"type":"message_stop"}
```

Concatenating the fragments yields `{"location": "San Francisco, CA"}` — parse only after
`content_block_stop` (or use a partial-JSON parser for incremental display).

### 5.7 Example: thinking stream

```
event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"1071 = 2 × 462 + 147"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"\nGCD(1071, 462) = 21."}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"EqQBCgIYAhIM..."}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}
```

A `text` block (index 1) with the visible answer follows the thinking block.

### 5.8 Server-tool blocks in streams

Server-side tool use streams like client tool use: a `server_tool_use` block with
`input_json_delta` fragments, then the paired result block (e.g. `web_search_tool_result`)
arrives as a `content_block_start` carrying the complete result content and an immediate
`content_block_stop` (no deltas). The final `message_delta` usage then includes
`server_tool_use` counters, e.g.:

```
event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":10682,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":510,"server_tool_use":{"web_search_requests":1}}}
```

### 5.9 Interrupted-stream recovery

- Capture all content received before the failure.
- Claude 4.5 and earlier: resend with the partial response as the start of a new `assistant`
  message (prefill continuation).
- Claude 4.6 and later: instead add a `user` message containing the partial response and an
  instruction to continue ("Your previous response was interrupted and ended with
  [previous_response]. Continue from where you left off.").
- `tool_use` and `thinking` blocks cannot be partially recovered; resume from the most recent
  `text` block.
