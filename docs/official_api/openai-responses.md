# OpenAI Responses API (HTTP)

Reference for `POST /v1/responses` over HTTP: request body, the Response object,
and SSE streaming events. Compiled from the official API reference:
[Responses API](https://developers.openai.com/api/reference/resources/responses.md),
[Create a model response](https://developers.openai.com/api/reference/resources/responses/methods/create/index.md),
[streaming events](https://developers.openai.com/api/reference/resources/responses/streaming-events.md).

Out of scope: get/delete/cancel/compact response endpoints, list input items,
input token counts. For the WebSocket mode see
[openai-responses-websocket.md](./openai-responses-websocket.md).

## 1. Endpoint and authentication

```
POST https://api.openai.com/v1/responses
Content-Type: application/json
Authorization: Bearer $OPENAI_API_KEY
```

- Non-streaming: the call returns a single `Response` JSON object.
- Streaming: set `"stream": true` in the body; the server replies with
  `text/event-stream` (SSE, see section 4).

## 2. Request body

All top-level parameters (every one is optional at the JSON level; `model` and
`input` are the ones you need in practice):

| Field | Type | Description |
|---|---|---|
| `model` | string | Model ID, e.g. `gpt-5.1`, `o3`, `gpt-4o`. Free-form string; known enums include the gpt-5.x / o-series / gpt-4.x families and Responses-only models (`o1-pro`, `o3-pro`, `o3-deep-research`, `computer-use-preview`, `gpt-5-codex`, `gpt-5-pro`, ...). |
| `input` | string \| array of InputItem | Text, image, or file inputs. See 2.1. |
| `instructions` | string \| null | System/developer message inserted into context. Not carried over when using `previous_response_id`. (On the Response object this may also echo as an item array.) |
| `include` | array of string \| null | Extra output data to include. See 2.6. |
| `tools` | array of Tool | Tool definitions. See 2.2. |
| `tool_choice` | string \| object | Tool selection control. See 2.3. |
| `parallel_tool_calls` | boolean \| null | Allow parallel tool calls. |
| `max_tool_calls` | number \| null | Max total built-in tool calls in a response. |
| `text` | object | Output text config: `format` (structured outputs) + `verbosity`. See 2.4. |
| `reasoning` | object \| null | Reasoning config (gpt-5 and o-series only). See 2.5. |
| `conversation` | string \| `{id}` \| null | Conversation this response belongs to; items are prepended to input and results appended to the conversation. Mutually exclusive with `previous_response_id`. |
| `previous_response_id` | string \| null | ID of the previous response for multi-turn state. Mutually exclusive with `conversation`. |
| `store` | boolean \| null | Whether to store the response for later retrieval. |
| `background` | boolean \| null | Run the response in the background (poll or resume the stream later). |
| `stream` | boolean \| null | Enable SSE streaming. |
| `stream_options` | `{include_obfuscation?: boolean}` \| null | Only with `stream: true`. `include_obfuscation` (default true) adds a random-padding `obfuscation` field to delta events to normalize payload sizes; set false to save bandwidth. |
| `max_output_tokens` | number \| null | Upper bound on generated tokens (visible output + reasoning). |
| `temperature` | number \| null | Sampling temperature 0–2. |
| `top_p` | number \| null | Nucleus sampling. |
| `top_logprobs` | number \| null | 0–20; number of top logprobs per token position. |
| `truncation` | `"auto"` \| `"disabled"` \| null | `auto`: drop items from the start of the conversation when context overflows; `disabled` (default): 400 error instead. |
| `prompt` | object \| null | Prompt template reference: `{id, version?, variables?}`; variable values are strings or `input_text`/`input_image`/`input_file` objects. |
| `prompt_cache_key` | string \| null | Cache-bucketing key (replaces `user`). |
| `prompt_cache_options` | `{mode?, ttl?}` | Prompt caching (gpt-5.6+). `mode`: `"implicit"` (default) \| `"explicit"`; `ttl`: `"30m"` (only supported value). Explicit breakpoints are set per content part via `prompt_cache_breakpoint`. |
| `prompt_cache_retention` | `"in_memory"` \| `"24h"` \| null | Deprecated; use `prompt_cache_options.ttl`. |
| `metadata` | map<string,string> \| null | Up to 16 pairs; key ≤ 64 chars, value ≤ 512 chars. |
| `context_management` | array of `{type, compact_threshold?}` \| null | Context management entries; `type` currently only `"compaction"`, `compact_threshold` is the token threshold. |
| `moderation` | `{model, policy?}` \| null | Run moderation on input/output. `policy.input/.output`: `{mode: "score" \| "block"}`. |
| `service_tier` | string \| null | `"auto"` \| `"default"` \| `"flex"` \| `"scale"` \| `"priority"` \| `"fast"`. Response echoes the tier actually used. |
| `safety_identifier` | string \| null | Stable end-user ID for abuse detection (≤ 64 chars, hash recommended). |
| `user` | string | Deprecated; replaced by `safety_identifier` + `prompt_cache_key`. |

### 2.1 `input`

Polymorphic:

- **string** — shorthand for a single `user` text message.
- **array of input items** — each item is one of the types below (discriminated
  by `type`; for messages `type` may be omitted).

Item types (`type` value → meaning):

| `type` | Direction | Meaning |
|---|---|---|
| `message` | in/out | User/system/developer input message, or an assistant output message. |
| `function_call` | out (replayed as input) | Model's call to a developer function. |
| `function_call_output` | in | Developer-supplied result for a `function_call`. |
| `reasoning` | out (replayed as input) | Chain-of-thought item; resend on later turns when managing context manually. |
| `item_reference` | in | `{id, type?: "item_reference"}` — reference a stored item by ID. |
| `web_search_call`, `file_search_call`, `image_generation_call`, `code_interpreter_call`, `computer_call` | out | Built-in tool calls (see 3.2). |
| `computer_call_output` | in | Screenshot result for a `computer_call`. |
| `local_shell_call` / `local_shell_call_output`, `shell_call` / `shell_call_output`, `apply_patch_call` / `apply_patch_call_output` | out / in | Shell & apply-patch tool calls and their results. |
| `mcp_list_tools`, `mcp_approval_request`, `mcp_call` | out | MCP tool listing, approval request, invocation. |
| `mcp_approval_response` | in | `{approval_request_id, approve, reason?}` — answer an approval request. |
| `custom_tool_call` / `custom_tool_call_output` | out / in | Custom (free-form input) tool call and result. |
| `tool_search_call` / `tool_search_output`, `additional_tools` | out / in | Deferred-tool search call, its loaded tool definitions, and developer-injected extra tools (`{role: "developer", tools, type}`). |
| `program` / `program_output` | out | Programmatic tool calling: executed JS source (`{id, call_id, code, fingerprint}`) and its result. |
| `compaction` | in/out | `{encrypted_content, type, id?}` — compacted-context item. |
| `compaction_trigger` | in | `{type}` — compact the current context; must be the final input item. |

#### Input message (`EasyInputMessage` / structured input message)

| Field | Type | Required | Description |
|---|---|---|---|
| `content` | string \| array of content parts | yes | Message content. |
| `role` | `"user"` \| `"assistant"` \| `"system"` \| `"developer"` | yes | Structured input-message form allows only `user`/`system`/`developer`. |
| `type` | `"message"` | no | |
| `phase` | `"commentary"` \| `"final_answer"` \| null | no | Labels assistant messages; preserve and resend on follow-ups for `gpt-5.3-codex`+ models. |
| `status` | `"in_progress"` \| `"completed"` \| `"incomplete"` | no | Populated when items are returned via API. |

Input content parts (no audio part exists in the Responses API):

| Part `type` | Fields |
|---|---|
| `input_text` | `text` (string, required) |
| `input_image` | `detail` (`"low"`\|`"high"`\|`"auto"`\|`"original"`, default `auto`), `file_id?` (string\|null), `image_url?` (string\|null — URL or base64 data URL) |
| `input_file` | `detail?` (`"auto"`\|`"low"`\|`"high"`), `file_data?` (string), `file_id?` (string\|null), `file_url?` (string), `filename?` (string) |

All input parts additionally accept `prompt_cache_breakpoint?: {mode: "explicit"}`
(marks the end of a reusable prompt prefix; TTL from `prompt_cache_options.ttl`).

#### Assistant output message (replayable as input)

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | yes | Message ID. |
| `content` | array | yes | Parts: `output_text` \| `refusal`. |
| `role` | `"assistant"` | yes | |
| `status` | `"in_progress"` \| `"completed"` \| `"incomplete"` | yes | |
| `type` | `"message"` | yes | |
| `phase` | `"commentary"` \| `"final_answer"` \| null | no | |

Output content parts:

- `output_text`: `{text: string, type: "output_text", annotations: Annotation[], logprobs: Logprob[]}`
  - `Logprob`: `{token, bytes: number[], logprob, top_logprobs: {token, bytes, logprob}[]}`
- `refusal`: `{refusal: string, type: "refusal"}`

Annotations:

| `type` | Fields |
|---|---|
| `file_citation` | `file_id`, `filename`, `index` |
| `url_citation` | `url`, `title`, `start_index`, `end_index` |
| `container_file_citation` | `container_id`, `file_id`, `filename`, `start_index`, `end_index` |
| `file_path` | `file_id`, `index` |

#### `function_call` item

| Field | Type | Required | Description |
|---|---|---|---|
| `arguments` | string | yes | JSON string of arguments. |
| `call_id` | string | yes | Links call to its output. |
| `name` | string | yes | Function name. |
| `type` | `"function_call"` | yes | |
| `id` | string | no | Item ID (present when returned by the API). |
| `namespace` | string | no | Namespace of the function (namespace tools). |
| `caller` | `{type:"direct"}` \| `{type:"program", caller_id}` \| null | no | Execution context (programmatic tool calling). |
| `status` | `"in_progress"` \| `"completed"` \| `"incomplete"` | no | |

#### `function_call_output` item

| Field | Type | Required | Description |
|---|---|---|---|
| `call_id` | string | yes | Must match the `function_call.call_id`. |
| `output` | string \| array | yes | JSON string, or an array of `input_text` / `input_image` / `input_file` parts. |
| `type` | `"function_call_output"` | yes | |
| `id`, `name`, `namespace`, `caller`, `status` | — | no | Same semantics as on `function_call`. |

`custom_tool_call` is analogous but with free-form `input: string` (plus
`call_id`, `name`, `type: "custom_tool_call"`, optional `id`/`caller`/`namespace`);
`custom_tool_call_output` mirrors `function_call_output` with
`type: "custom_tool_call_output"`.

#### `reasoning` item

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | yes | |
| `summary` | array of `{text, type: "summary_text"}` | yes | Reasoning summary (may be empty). |
| `type` | `"reasoning"` | yes | |
| `content` | array of `{text, type: "reasoning_text"}` | no | Raw reasoning text (when exposed). |
| `encrypted_content` | string \| null | no | Encrypted reasoning; populated by default on `POST /v1/responses`. Needed for stateless multi-turn (`store: false` / ZDR) — see `include: ["reasoning.encrypted_content"]`. |
| `status` | `"in_progress"` \| `"completed"` \| `"incomplete"` | no | |

### 2.2 `tools`

Array of tool definitions, discriminated by `type`.

#### Function tool (full detail)

| Field | Type | Required | Description |
|---|---|---|---|
| `type` | `"function"` | yes | |
| `name` | string | yes | |
| `parameters` | JSON Schema object \| null | yes | Argument schema. |
| `strict` | boolean \| null | yes | Enforce strict schema validation. |
| `description` | string \| null | no | Helps the model decide when to call. |
| `output_schema` | JSON Schema object \| null | no | Schema of the JSON encoded in string outputs. |
| `allowed_callers` | array of `"direct"` \| `"programmatic"` \| null | no | Invocation contexts. |
| `defer_loading` | boolean | no | Defer; discovered via tool search. |

```json
{
  "type": "function",
  "name": "get_weather",
  "description": "Get current weather for a location.",
  "parameters": {
    "type": "object",
    "properties": { "location": { "type": "string" } },
    "required": ["location"],
    "additionalProperties": false
  },
  "strict": true
}
```

#### Custom tool (full detail)

| Field | Type | Required | Description |
|---|---|---|---|
| `type` | `"custom"` | yes | |
| `name` | string | yes | |
| `description` | string | no | |
| `format` | object | no | Input format. Default unconstrained text. `{type: "text"}` or `{type: "grammar", syntax: "lark" \| "regex", definition: string}`. |
| `allowed_callers`, `defer_loading` | — | no | As for function tools. |

#### Built-in and other tools (summary)

| `type` | Key parameters |
|---|---|
| `web_search` (or `web_search_2025_08_26`) | `filters?: {allowed_domains?: string[]}`, `search_context_size?: "low"\|"medium"\|"high"` (default medium), `user_location?: {type: "approximate", city?, country?, region?, timezone?}` |
| `web_search_preview` (or `web_search_preview_2025_03_11`) | Legacy variant; adds `search_content_types?: ("text"\|"image")[]` |
| `file_search` | `vector_store_ids: string[]` (required), `filters?` (ComparisonFilter `{key,type,value}` or CompoundFilter `{type:"and"\|"or", filters}`), `max_num_results?` (1–50), `ranking_options?: {ranker?, score_threshold?, hybrid_search?: {embedding_weight, text_weight}}` |
| `computer` | `{type}` only. |
| `computer_use_preview` | `display_width: number`, `display_height: number`, `environment: "windows"\|"mac"\|"linux"\|"ubuntu"\|"browser"` (all required) |
| `code_interpreter` | `container: string \| {type: "auto", file_ids?, memory_limit?: "1g"\|"4g"\|"16g"\|"64g", network_policy?}` (required), `allowed_callers?` |
| `image_generation` | `action?: "generate"\|"edit"\|"auto"`, `background?: "transparent"\|"opaque"\|"auto"`, `model?` (default `gpt-image-1`), `quality?: "low"\|"medium"\|"high"\|"auto"`, `size?` (`"1024x1024"` etc. or `auto`), `output_format?: "png"\|"webp"\|"jpeg"`, `output_compression?`, `moderation?: "auto"\|"low"`, `input_fidelity?: "high"\|"low"`, `input_image_mask?: {file_id?, image_url?}`, `partial_images?` (0–3, streaming) |
| `mcp` | `server_label` (required); one of `server_url` / `connector_id` / `tunnel_id`; `allowed_tools?: string[] \| {tool_names?, read_only?}`, `require_approval?: "always" \| "never" \| {always?: filter, never?: filter}`, `authorization?` (OAuth token), `headers?`, `server_description?`, `allowed_callers?`, `defer_loading?`. `connector_id` enum: `connector_dropbox`, `connector_gmail`, `connector_googlecalendar`, `connector_googledrive`, `connector_microsoftteams`, `connector_outlookcalendar`, `connector_outlookemail`, `connector_sharepoint`. |
| `local_shell` | `{type}` only. |
| `shell` | `allowed_callers?`, `environment?: ContainerAuto \| LocalEnvironment \| ContainerReference` |
| `apply_patch` | `allowed_callers?` |
| `namespace` | `name`, `description`, `tools: (function \| custom)[]` — groups function/custom tools; calls carry `namespace`. |
| `tool_search` | `execution?: "server"\|"client"`, `description?`, `parameters?` — searches deferred (`defer_loading`) tools. |
| `programmatic_tool_calling` | `{type}` only — lets the model drive tools from generated JS (`program` items). |

### 2.3 `tool_choice`

- String mode: `"none"` | `"auto"` | `"required"`.
- Object forms:

| `type` | Fields | Meaning |
|---|---|---|
| `allowed_tools` | `mode: "auto"\|"required"`, `tools: [{type, name?/server_label?...}]` | Restrict to a subset of the defined tools. |
| `function` | `name` | Force this function. |
| `custom` | `name` | Force this custom tool. |
| `mcp` | `server_label`, `name?` | Force an MCP server/tool. |
| built-in hosted tool | — | `{"type": "file_search" \| "web_search_preview" \| "computer" \| "computer_use_preview" \| "computer_use" \| "web_search_preview_2025_03_11" \| "image_generation" \| "code_interpreter"}` |
| `apply_patch` / `shell` / `programmatic_tool_calling` | — | Force that tool. |

### 2.4 `text`

```json
"text": { "format": { "type": "json_schema", "name": "result", "schema": { ... }, "strict": true }, "verbosity": "medium" }
```

- `format` (default `{"type": "text"}`):
  - `{type: "text"}` — plain text.
  - `{type: "json_schema", name, schema, description?, strict?}` — Structured
    Outputs. `name`: `[a-zA-Z0-9_-]{1,64}`; `schema`: JSON Schema object;
    `strict: true` restricts to the supported JSON Schema subset.
  - `{type: "json_object"}` — legacy JSON mode (prompt must ask for JSON).
- `verbosity`: `"low"` | `"medium"` (default) | `"high"` | null.

### 2.5 `reasoning`

| Field | Type | Description |
|---|---|---|
| `effort` | `"none"` \| `"minimal"` \| `"low"` \| `"medium"` \| `"high"` \| `"xhigh"` \| `"max"` \| null | Reasoning effort; model support varies. |
| `summary` | `"auto"` \| `"concise"` \| `"detailed"` \| null | Request reasoning summaries. |
| `context` | `"auto"` \| `"current_turn"` \| `"all_turns"` \| null | Which reasoning items are re-rendered on later turns (gpt-5.6 defaults to `all_turns`). |
| `mode` | string \| `"standard"` \| `"pro"` | Reasoning execution mode. |
| `generate_summary` | `"auto"` \| `"concise"` \| `"detailed"` \| null | Deprecated; use `summary`. |

### 2.6 `include` values

`"file_search_call.results"`, `"web_search_call.results"`,
`"web_search_call.action.sources"`, `"message.input_image.image_url"`,
`"computer_call_output.output.image_url"`, `"code_interpreter_call.outputs"`,
`"reasoning.encrypted_content"`, `"message.output_text.logprobs"`.

### 2.7 Request examples

Minimal:

```json
{ "model": "gpt-5.1", "input": "Tell me a three sentence bedtime story about a unicorn." }
```

Multi-turn with items, a function tool, and structured output:

```json
{
  "model": "gpt-5.1",
  "instructions": "You are a helpful assistant.",
  "input": [
    { "type": "message", "role": "user", "content": [
      { "type": "input_text", "text": "What is the weather here?" },
      { "type": "input_image", "image_url": "https://example.com/photo.jpg", "detail": "auto" }
    ]},
    { "type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{\"location\":\"Paris\"}" },
    { "type": "function_call_output", "call_id": "call_1", "output": "{\"temp_c\":21}" }
  ],
  "tools": [{ "type": "function", "name": "get_weather", "parameters": { "type": "object", "properties": { "location": { "type": "string" } } }, "strict": true }],
  "tool_choice": "auto",
  "text": { "format": { "type": "text" } },
  "reasoning": { "effort": "medium", "summary": "auto" },
  "store": false,
  "include": ["reasoning.encrypted_content"]
}
```

## 3. Response object (non-streaming)

Returned directly by the endpoint, and embedded in the
`response.created/in_progress/queued/completed/failed/incomplete` stream events.

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | yes | Response ID (`resp_...`). |
| `object` | `"response"` | yes | |
| `created_at` | number | yes | Unix seconds. |
| `status` | string | no | `"completed"` \| `"failed"` \| `"in_progress"` \| `"cancelled"` \| `"queued"` \| `"incomplete"`. |
| `completed_at` | number \| null | no | Unix seconds. |
| `error` | `{code, message}` \| null | yes | Set when generation failed. `code` enum: `server_error`, `rate_limit_exceeded`, `invalid_prompt`, `data_residency_mismatch`, `bio_policy`, `vector_store_timeout`, plus image errors (`invalid_image`, `invalid_image_format`, `invalid_base64_image`, `invalid_image_url`, `image_too_large`, `image_too_small`, `image_parse_error`, `image_content_policy_violation`, `invalid_image_mode`, `image_file_too_large`, `unsupported_image_media_type`, `empty_image_file`, `failed_to_download_image`, `image_file_not_found`). |
| `incomplete_details` | `{reason}` \| null | yes | `reason`: `"max_output_tokens"` \| `"content_filter"`. |
| `output` | array of OutputItem | yes | See 3.1. |
| `output_text` | string \| null | no | SDK-only convenience: concatenation of all `output_text` parts. Not sent on the wire by the API itself. |
| `usage` | object | no | See 3.2. |
| `model` | string | yes | Model actually used. |
| `instructions` | string \| item array \| null | yes | Echo of the request. |
| `metadata` | map \| null | yes | |
| `parallel_tool_calls` | boolean | yes | |
| `tool_choice` | string \| object | yes | Echo. |
| `tools` | array | yes | Echo. |
| `temperature`, `top_p` | number \| null | yes | Echo. |
| `top_logprobs` | number \| null | no | Echo. |
| `background` | boolean \| null | no | Echo. |
| `conversation` | `{id}` \| null | no | Conversation the response belongs to. |
| `previous_response_id` | string \| null | no | Echo. |
| `prompt` | object \| null | no | Echo of prompt template reference. |
| `prompt_cache_key` | string \| null | no | Echo. |
| `prompt_cache_options` | `{mode, ttl}` | no | Options actually applied. |
| `prompt_cache_retention` | `"in_memory"` \| `"24h"` \| null | no | Deprecated. |
| `reasoning` | object \| null | no | Effective reasoning config (`effort`, `summary`, `context`, `mode`, ...). |
| `max_output_tokens`, `max_tool_calls` | number \| null | no | Echo. |
| `moderation` | `{input, output}` \| null | no | Moderation results (`{flagged, categories, category_scores, category_applied_input_types, model, type: "moderation_result"}` per side). |
| `text` | object | no | Echo of text config. |
| `truncation` | `"auto"` \| `"disabled"` \| null | no | Echo. |
| `service_tier` | string \| null | no | Tier actually used. |
| `safety_identifier`, `user` | string | no | Echo (`user` deprecated). |

Note: wire examples also show a `store` field echoed on responses, but it is
not part of the current documented Response schema.

### 3.1 `output` items

Same shapes as the item types in 2.1 (with `id` always populated). The common
ones:

- **`message`** — always `role: "assistant"`; `content` holds `output_text`
  (with `annotations`, optional `logprobs`) and/or `refusal` parts; `status`,
  optional `phase`.
- **`reasoning`** — `summary` (`summary_text` parts), optional `content`
  (`reasoning_text` parts), optional `encrypted_content`.
- **`function_call`** — `arguments`, `call_id`, `name`, `status`, optional
  `namespace`/`caller`.
- Tool call items (brief; statuses in parentheses):
  - `web_search_call` — `action`: `{type:"search", query?, queries?, sources?}` \|
    `{type:"open_page", url?}` \| `{type:"find_in_page", url, pattern}`;
    `status` (`in_progress|searching|completed|failed`).
  - `file_search_call` — `queries: string[]`, `status`
    (`in_progress|searching|completed|incomplete|failed`), `results?`
    (`{file_id, filename, text, score, attributes}[]`, with `include`).
  - `image_generation_call` — `result: string|null` (base64 image), `status`
    (`in_progress|completed|generating|failed`).
  - `code_interpreter_call` — `code: string|null`, `container_id`, `outputs`
    (`{type:"logs", logs}` \| `{type:"image", url}`, with `include`), `status`
    (`in_progress|completed|incomplete|interpreting|failed`).
  - `computer_call` — `call_id`, `action` (click/double_click/drag/keypress/
    move/screenshot/scroll/type/wait variants), `pending_safety_checks`;
    answered with `computer_call_output` (`output: {type:"computer_screenshot",
    file_id?, image_url?}`, `acknowledged_safety_checks?`).
  - `local_shell_call` — `action: {type:"exec", command[], env, timeout_ms?,
    user?, working_directory?}`, `call_id`; answered with
    `local_shell_call_output` (`output: string`).
  - `shell_call` / `shell_call_output`, `apply_patch_call` (`operation`:
    create/delete/update file with `path`/`diff`) / `apply_patch_call_output`.
  - `mcp_call` — `arguments`, `name`, `server_label`, `output?`, `error?`,
    `approval_request_id?`, `status` (`in_progress|completed|incomplete|calling|failed`).
  - `mcp_list_tools` — `server_label`, `tools: {name, input_schema,
    description?, annotations?}[]`, `error?`.
  - `mcp_approval_request` — `arguments`, `name`, `server_label` (answer via
    input item `mcp_approval_response`).
  - `custom_tool_call` — `call_id`, `name`, `input`.
  - `tool_search_call` / `tool_search_output`, `program` / `program_output`,
    `compaction` — see 2.1.

### 3.2 `usage`

```json
"usage": {
  "input_tokens": 36,
  "input_tokens_details": { "cached_tokens": 0, "cache_write_tokens": 0 },
  "output_tokens": 87,
  "output_tokens_details": { "reasoning_tokens": 0 },
  "total_tokens": 123
}
```

| Field | Description |
|---|---|
| `input_tokens` | Input token count. |
| `input_tokens_details.cached_tokens` | Tokens read from the prompt cache. |
| `input_tokens_details.cache_write_tokens` | Tokens written to the prompt cache. |
| `output_tokens` | Output tokens (includes reasoning). |
| `output_tokens_details.reasoning_tokens` | Reasoning tokens. |
| `total_tokens` | Sum. |

### 3.3 Example (completed response)

```json
{
  "id": "resp_67ccd2bed1ec8190b14f964abc0542670bb6a6b452d3795b",
  "object": "response",
  "created_at": 1741476542,
  "status": "completed",
  "completed_at": 1741476543,
  "error": null,
  "incomplete_details": null,
  "instructions": null,
  "max_output_tokens": null,
  "model": "gpt-5.4",
  "output": [
    {
      "type": "message",
      "id": "msg_67ccd2bf17f0819081ff3bb2cf6508e60bb6a6b452d3795b",
      "status": "completed",
      "role": "assistant",
      "content": [
        { "type": "output_text", "text": "In a peaceful grove beneath a silver moon...", "annotations": [] }
      ]
    }
  ],
  "parallel_tool_calls": true,
  "previous_response_id": null,
  "reasoning": { "effort": null, "summary": null },
  "store": true,
  "temperature": 1.0,
  "text": { "format": { "type": "text" } },
  "tool_choice": "auto",
  "tools": [],
  "top_p": 1.0,
  "truncation": "disabled",
  "usage": {
    "input_tokens": 36,
    "input_tokens_details": { "cached_tokens": 0, "cache_write_tokens": 0 },
    "output_tokens": 87,
    "output_tokens_details": { "reasoning_tokens": 0 },
    "total_tokens": 123
  },
  "user": null,
  "metadata": {}
}
```

## 4. Streaming (SSE)

With `"stream": true`, the server sends standard SSE frames:

```
event: <event type>
data: <JSON payload>
```

- Every payload has `type` (same as the SSE `event` name) and
  `sequence_number` (monotonically increasing integer; use it for ordering /
  resumption).
- Delta events may carry an extra `obfuscation` field (random padding) unless
  `stream_options.include_obfuscation` is false. Ignore it when parsing.
- With `background: true` + `stream: true`, a `response.queued` event may
  precede `response.created`.

### 4.1 Event lifecycle

Typical order for a plain text answer:

```
response.created            (response snapshot, status "in_progress")
response.in_progress
  response.output_item.added        (item skeleton, output_index i)
    response.content_part.added     (part skeleton, content_index j)
      response.output_text.delta ×N
    response.output_text.done
    response.content_part.done
  response.output_item.done         (finalized item)
response.completed          (final Response incl. usage)
```

Per output item type, the middle section differs:

- reasoning item → `reasoning_summary_part.added` / `reasoning_summary_text.delta/done` /
  `reasoning_summary_part.done` (and `reasoning_text.delta/done` when raw
  reasoning content streams).
- function call item → `function_call_arguments.delta/done`.
- built-in tool items → their `*.in_progress` / `*.searching` /
  `*.generating` / `*.completed` status events.

Terminal events: exactly one of `response.completed`, `response.failed`,
`response.incomplete` (or an `error` event on fatal stream errors). Each of
these carries the full final `response` object.

### 4.2 Core events

Response lifecycle events — payload: `{type, response: Response, sequence_number}`:

| Event | When |
|---|---|
| `response.created` | First event; `response.status = "in_progress"`. |
| `response.queued` | Background mode: response is queued. |
| `response.in_progress` | Generation is running. |
| `response.completed` | Done; `response.usage` populated. |
| `response.failed` | `response.error` populated. |
| `response.incomplete` | `response.incomplete_details` populated. |

Output item events:

```json
{ "type": "response.output_item.added", "output_index": 0,
  "item": { "id": "msg_123", "status": "in_progress", "type": "message", "role": "assistant", "content": [] },
  "sequence_number": 3 }
```

| Event | Payload fields |
|---|---|
| `response.output_item.added` | `output_index`, `item` (skeleton), `sequence_number` |
| `response.output_item.done` | `output_index`, `item` (final), `sequence_number` |

Content part events (`part` is an `output_text` / `refusal` part):

| Event | Payload fields |
|---|---|
| `response.content_part.added` | `item_id`, `output_index`, `content_index`, `part`, `sequence_number` |
| `response.content_part.done` | same, with final `part` |

Text / refusal deltas:

| Event | Payload fields |
|---|---|
| `response.output_text.delta` | `item_id`, `output_index`, `content_index`, `delta`, `logprobs`, `sequence_number` |
| `response.output_text.done` | `item_id`, `output_index`, `content_index`, `text`, `logprobs`, `sequence_number` |
| `response.output_text.annotation.added` | `item_id`, `output_index`, `content_index`, `annotation_index`, `annotation`, `sequence_number` |
| `response.refusal.delta` | `item_id`, `output_index`, `content_index`, `delta`, `sequence_number` |
| `response.refusal.done` | `item_id`, `output_index`, `content_index`, `refusal`, `sequence_number` |

```json
{ "type": "response.output_text.delta", "item_id": "msg_123", "output_index": 0,
  "content_index": 0, "delta": "In", "sequence_number": 5 }
```

Function call arguments:

| Event | Payload fields |
|---|---|
| `response.function_call_arguments.delta` | `item_id`, `output_index`, `delta`, `sequence_number` |
| `response.function_call_arguments.done` | `item_id`, `output_index`, `name`, `arguments`, `sequence_number` |

```json
{ "type": "response.function_call_arguments.done", "item_id": "item-abc",
  "output_index": 1, "name": "get_weather", "arguments": "{ \"arg\": 123 }", "sequence_number": 9 }
```

Reasoning:

| Event | Payload fields |
|---|---|
| `response.reasoning_summary_part.added` | `item_id`, `output_index`, `summary_index`, `part` (`{type:"summary_text", text}`), `sequence_number` |
| `response.reasoning_summary_part.done` | same, with final `part` |
| `response.reasoning_summary_text.delta` | `item_id`, `output_index`, `summary_index`, `delta`, `sequence_number` |
| `response.reasoning_summary_text.done` | `item_id`, `output_index`, `summary_index`, `text`, `sequence_number` |
| `response.reasoning_text.delta` | `item_id`, `output_index`, `content_index`, `delta`, `sequence_number` |
| `response.reasoning_text.done` | `item_id`, `output_index`, `content_index`, `text`, `sequence_number` |

Stream error (does not necessarily end the stream by itself; also expect a
terminal response event):

```json
{ "type": "error", "code": "ERR_SOMETHING", "message": "Something went wrong", "param": null, "sequence_number": 1 }
```

### 4.3 Tool / niche events

All of these carry `item_id`, `output_index`, `sequence_number` (plus the field
listed). Status-only events have no extra fields.

| Event | Extra payload | Meaning |
|---|---|---|
| `response.file_search_call.in_progress` / `.searching` / `.completed` | — | File search progress. |
| `response.web_search_call.in_progress` / `.searching` / `.completed` | — | Web search progress. |
| `response.image_generation_call.in_progress` / `.generating` / `.completed` | — | Image generation progress. |
| `response.image_generation_call.partial_image` | `partial_image_index`, `partial_image_b64` | Partial image (needs `partial_images` > 0 on the tool). |
| `response.mcp_call_arguments.delta` / `.done` | `delta` / `arguments` | MCP call argument streaming. |
| `response.mcp_call.in_progress` / `.completed` / `.failed` | — | MCP invocation progress. |
| `response.mcp_list_tools.in_progress` / `.completed` / `.failed` | — | MCP tool-listing progress. |
| `response.code_interpreter_call.in_progress` / `.interpreting` / `.completed` | — | Code interpreter progress. |
| `response.code_interpreter_call_code.delta` / `.done` | `delta` / `code` | Streamed code. |
| `response.custom_tool_call_input.delta` / `.done` | `delta` / `input` | Custom tool input streaming. |
| `response.audio.delta` / `.done` | `delta` (base64 audio) / — | Audio output (legacy; payload has only `delta`/`sequence_number`/`type` per schema). |
| `response.audio.transcript.delta` / `.done` | `delta` / — | Audio transcript streaming (legacy). |

### 4.4 Full SSE example

Request: `{"model": "gpt-5.4", "instructions": "You are a helpful assistant.", "input": "Hello!", "stream": true}`

```
event: response.created
data: {"type":"response.created","response":{"id":"resp_67c9...","object":"response","created_at":1741290958,"status":"in_progress","error":null,"incomplete_details":null,"instructions":"You are a helpful assistant.","max_output_tokens":null,"model":"gpt-5.4","output":[],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{"effort":null,"summary":null},"store":true,"temperature":1.0,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1.0,"truncation":"disabled","usage":null,"user":null,"metadata":{}}}

event: response.in_progress
data: {"type":"response.in_progress","response":{"id":"resp_67c9...","status":"in_progress", ...}}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":0,"item":{"id":"msg_67c9...","type":"message","status":"in_progress","role":"assistant","content":[]}}

event: response.content_part.added
data: {"type":"response.content_part.added","item_id":"msg_67c9...","output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","item_id":"msg_67c9...","output_index":0,"content_index":0,"delta":"Hi"}

...

event: response.output_text.done
data: {"type":"response.output_text.done","item_id":"msg_67c9...","output_index":0,"content_index":0,"text":"Hi there! How can I assist you today?"}

event: response.content_part.done
data: {"type":"response.content_part.done","item_id":"msg_67c9...","output_index":0,"content_index":0,"part":{"type":"output_text","text":"Hi there! How can I assist you today?","annotations":[]}}

event: response.output_item.done
data: {"type":"response.output_item.done","output_index":0,"item":{"id":"msg_67c9...","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hi there! How can I assist you today?","annotations":[]}]}}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp_67c9...","status":"completed","output":[{"id":"msg_67c9...","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hi there! How can I assist you today?","annotations":[]}]}],"usage":{"input_tokens":37,"output_tokens":11,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":48}, ...}}
```

(Payloads abbreviated with `...`; the real events contain the full Response
object each time.)

## 5. Related

- WebSocket mode: [openai-responses-websocket.md](./openai-responses-websocket.md)
- Other Responses endpoints (not covered here): `GET/DELETE /v1/responses/{id}`,
  `POST /v1/responses/{id}/cancel`, `POST /v1/responses/compact`,
  `GET /v1/responses/{id}/input_items`, `POST /v1/responses/input_tokens`.
