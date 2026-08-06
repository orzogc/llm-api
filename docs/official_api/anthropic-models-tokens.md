# Anthropic Models & Token Counting API Reference

Covers the model-listing endpoints (`/v1/models`, `/v1/models/{model_id}`) and the Messages
token-counting endpoint (`POST /v1/messages/count_tokens`). Companion to
[`anthropic-messages.md`](./anthropic-messages.md); nested request types shared with Messages
create are cross-referenced there instead of re-documented.

Base URL: `https://api.anthropic.com`

Note on type names: the source schemas are the beta surface and prefix type names with `Beta`
(`BetaModelInfo`, `BetaMessageParam`, ...). The prefix is dropped here; GA shapes are identical
except where a field is marked "beta only".

## 1. Headers

| Header | Required | Value / Description |
|---|---|---|
| `x-api-key` | yes | API key |
| `anthropic-version` | yes | API version date, e.g. `2023-06-01` |
| `content-type` | POST only | `application/json` |
| `anthropic-beta` | no | Comma-separated beta flags (same open string set as Messages; see `anthropic-messages.md` §1) |
| `anthropic-user-profile-id` | no | `count_tokens` only. User profile ID to attribute the request to; requires the `user-profiles` beta |

## 2. List Models — `GET /v1/models`

Lists available models; more recently released models are listed first. Usable to determine
which models are available for use in the API.

### 2.1 Query parameters

| Param | Type | Required | Description |
|---|---|---|---|
| `before_id` | string | no | Pagination cursor: return the page immediately *before* this object ID |
| `after_id` | string | no | Pagination cursor: return the page immediately *after* this object ID |
| `limit` | number | no | Items per page. Default `20`, range `1`–`1000` |

### 2.2 Response

| Field | Type | Description |
|---|---|---|
| `data` | array of ModelInfo | The page of models (see 2.3) |
| `first_id` | string | First ID in `data`; usable as `before_id` for the previous page |
| `last_id` | string | Last ID in `data`; usable as `after_id` for the next page |
| `has_more` | boolean | Whether more results exist in the requested page direction |

Implementation note (not in source schema): model `first_id`/`last_id` as nullable — an empty
page has no IDs. This matches the `after_id`/`before_id`/`has_more` scheme used by Batches/Files.

### 2.3 ModelInfo

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique model identifier |
| `type` | `"model"` | Object type; always `"model"` |
| `display_name` | string | Human-readable name, e.g. `"Claude Opus 4.6"` |
| `created_at` | string | RFC 3339 release datetime; may be an epoch value if unknown |
| `max_input_tokens` | number | Maximum input context window size (tokens) |
| `max_tokens` | number | Maximum value of the `max_tokens` request parameter for this model |
| `capabilities` | ModelCapabilities | Capability information (see 2.4) |
| `allowed_fallback_models` | array of string | Beta only. Model IDs accepted as `fallbacks[i].model` on Messages; empty list = `fallbacks` unsupported with this model as primary |

### 2.4 ModelCapabilities

Every leaf is a CapabilitySupport object: `{"supported": boolean}`.

| Field | Type | Description |
|---|---|---|
| `batch` | CapabilitySupport | Batch API support |
| `citations` | CapabilitySupport | Citation generation |
| `code_execution` | CapabilitySupport | Code execution tools |
| `image_input` | CapabilitySupport | Accepts image content blocks |
| `pdf_input` | CapabilitySupport | Accepts PDF content blocks |
| `structured_outputs` | CapabilitySupport | Structured output / JSON mode / strict tool schemas |
| `context_management` | object | `{supported: bool, clear_thinking_20251015, clear_tool_uses_20250919, compact_20260112}` — overall flag + per-strategy CapabilitySupport |
| `effort` | object | `{supported: bool, low, medium, high, xhigh, max}` — overall flag + per-level CapabilitySupport |
| `thinking` | object | `{supported: bool, types: {adaptive, enabled}}` — per-thinking-type CapabilitySupport (`adaptive` = auto) |

### 2.5 Example

```http
curl https://api.anthropic.com/v1/models?limit=2 \
    -H 'anthropic-version: 2023-06-01' \
    -H "x-api-key: $ANTHROPIC_API_KEY"
```

```json
{
  "data": [
    {
      "id": "claude-opus-4-6",
      "type": "model",
      "display_name": "Claude Opus 4.6",
      "created_at": "2026-02-04T00:00:00Z",
      "max_input_tokens": 1000000,
      "max_tokens": 128000,
      "allowed_fallback_models": ["claude-opus-4-5"],
      "capabilities": {
        "batch": {"supported": true},
        "citations": {"supported": true},
        "code_execution": {"supported": true},
        "context_management": {
          "supported": true,
          "clear_thinking_20251015": {"supported": true},
          "clear_tool_uses_20250919": {"supported": true},
          "compact_20260112": {"supported": true}
        },
        "effort": {
          "supported": true,
          "low": {"supported": true}, "medium": {"supported": true},
          "high": {"supported": true}, "xhigh": {"supported": true},
          "max": {"supported": true}
        },
        "image_input": {"supported": true},
        "pdf_input": {"supported": true},
        "structured_outputs": {"supported": true},
        "thinking": {
          "supported": true,
          "types": {"adaptive": {"supported": true}, "enabled": {"supported": true}}
        }
      }
    }
  ],
  "first_id": "claude-opus-4-6",
  "last_id": "claude-sonnet-4-6",
  "has_more": true
}
```

## 3. Get a Model — `GET /v1/models/{model_id}`

Returns a single ModelInfo (same shape as `data[]` entries in 2.3). Also resolves model
aliases: `model_id` accepts a model identifier **or alias** (e.g. `claude-sonnet-4-5`), and the
response `id` is the resolved concrete model ID.

| Path param | Type | Required | Description |
|---|---|---|---|
| `model_id` | string | yes | Model identifier or alias |

```http
curl https://api.anthropic.com/v1/models/claude-opus-4-6 \
    -H 'anthropic-version: 2023-06-01' \
    -H "x-api-key: $ANTHROPIC_API_KEY"
```

Response: one ModelInfo object (no pagination envelope). Unknown IDs return `404 not_found_error`.

## 4. Count tokens — `POST /v1/messages/count_tokens`

Counts the tokens a Messages request would consume — including system prompt, tools, images,
and documents — without creating the message. Accepts the same input shapes as Messages create.

Remarks (from the token-counting guide):
- The count is an **estimate**; actual `input_tokens` when creating the message may differ slightly.
- Counts may include system-added tokens; those are **not billed** when the message is created.
- The endpoint is free to use but has its own requests-per-minute rate limits (per usage tier).
- Counts are model-specific (tokenizers differ across model generations) — count against the model you will run.

### 4.1 Request body

| Field | Type | Required | Description |
|---|---|---|---|
| `model` | string | yes | Model that will complete the prompt (ID or alias) |
| `messages` | array of MessageParam | yes | Input conversation. Same shape as Messages create: `{role: "user"\|"assistant"\|"system", content: string \| ContentBlockParam[]}`; string content = one `text` block; same request content-block union (text, image, document, search_result, thinking, tool_use, tool_result, server-tool results, tool_reference, ...) — see `anthropic-messages.md` §2.2–2.3 |
| `system` | string \| array of TextBlockParam | no | System prompt. Block form: `{type:"text", text, cache_control?, citations?}` |
| `tools` | array of ToolUnion | no | Tool definitions; counted into the total. Same union as Messages create (see 4.2 and `anthropic-messages.md` §2.5) |
| `tool_choice` | ToolChoice | no | `{type:"auto"\|"any"\|"none"}` or `{type:"tool", name}`; `auto`/`any`/`tool` take `disable_parallel_tool_use?: bool` — see `anthropic-messages.md` §2.6 |
| `thinking` | ThinkingConfig | no | `{type:"enabled", budget_tokens (≥1024), display?}` \| `{type:"disabled"}` \| `{type:"adaptive", display?}`; `display` ∈ `"summarized"` (default) \| `"omitted"` — see `anthropic-messages.md` §2.7 |
| `cache_control` | `{type:"ephemeral", ttl?}` | no | Top-level shortcut: marks the last cacheable block in the request |
| `context_management` | object | no | `{edits: [...]}` with `clear_tool_uses_20250919`, `clear_thinking_20251015`, `compact_20260112` strategies — see `anthropic-messages.md` §2.8 |
| `mcp_servers` | array | no | `{name, type:"url", url, authorization_token?, tool_configuration?: {allowed_tools?, enabled?}}` |
| `output_config` | object | no | `{effort?: "low"\|"medium"\|"high"\|"xhigh"\|"max", format?: {type:"json_schema", schema}, task_budget?: {type:"tokens", total, remaining?}}` — see `anthropic-messages.md` §2.9 |
| `output_format` | JSONOutputFormat | no | Deprecated; use `output_config.format` |
| `speed` | `"standard"` \| `"fast"` | no | Inference speed mode (`fast` is premium and model-dependent) |

Messages-create parameters **not accepted** here: `max_tokens`, `stream`, `stop_sequences`,
`temperature`, `top_k`, `top_p`, `metadata`, `service_tier`, `container`, `inference_geo`,
`diagnostics`, `fallbacks`, `fallback_credit_token`.

### 4.2 `tools` union members

`Tool` (custom: `{name, description?, input_schema, ...}`) plus the server/Anthropic-defined
tool types: `bash_20241022`, `bash_20250124`; `code_execution_20250522/20250825/20260120/20260521`;
`computer_20241022/20250124/20251124`; `memory_20250818`;
`text_editor_20241022/20250124/20250429/20250728`; `web_search_20250305/20260209/20260318`;
`web_fetch_20250910/20260209/20260309/20260318`; `advisor_20260301`;
`tool_search_tool_bm25_20251119`, `tool_search_tool_regex_20251119`; `mcp_toolset`.
Field details: `anthropic-messages.md` §2.5.

### 4.3 Response — MessageTokensCount

| Field | Type | Description |
|---|---|---|
| `input_tokens` | number | Total tokens across the provided messages, system prompt, and tools |
| `context_management` | object | Info about context management applied to the message: `{original_input_tokens: number}` — the token count *before* context-management edits. Listed as always-present in the schema; safe to model as optional |

### 4.4 Example

```http
curl https://api.anthropic.com/v1/messages/count_tokens \
    -H 'content-type: application/json' \
    -H 'anthropic-version: 2023-06-01' \
    -H "x-api-key: $ANTHROPIC_API_KEY" \
    -d '{
      "model": "claude-opus-4-6",
      "system": [{"type": "text", "text": "Today'"'"'s date is 2024-06-01."}],
      "messages": [{"role": "user", "content": "Hello, world"}],
      "thinking": {"type": "adaptive"},
      "tools": [{
        "name": "get_weather",
        "input_schema": {
          "type": "object",
          "properties": {"location": {"type": "string"}},
          "required": ["location"]
        }
      }]
    }'
```

```json
{
  "context_management": {"original_input_tokens": 0},
  "input_tokens": 2095
}
```

## Sources

- [List Models](https://platform.claude.com/docs/en/api/beta/models/list.md) — §2 (beta surface).
- [Retrieve a Model](https://platform.claude.com/docs/en/api/beta/models/retrieve.md) — §3.
- [Count tokens in a Message](https://platform.claude.com/docs/en/api/beta/messages/count_tokens.md) — §4;
  usage remarks from the official token-counting guide.
