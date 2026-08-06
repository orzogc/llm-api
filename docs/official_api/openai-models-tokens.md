# OpenAI Models API and Responses Input Token Counting

Reference for the model-listing endpoints and `POST /v1/responses/input_tokens`.
Compiled from the official API reference:
[List models](https://developers.openai.com/api/reference/resources/models/methods/list/index.md),
[Retrieve model](https://developers.openai.com/api/reference/resources/models/methods/retrieve/index.md),
[Delete model](https://developers.openai.com/api/reference/resources/models/methods/delete/index.md),
[Get input token counts](https://developers.openai.com/api/reference/resources/responses/subresources/input_tokens/methods/count/index.md).
Deep nested request types (input items, tools, ...) are shared with the
Responses create body — see [openai-responses.md](./openai-responses.md).

## 1. Endpoint and authentication

```
Base URL: https://api.openai.com/v1
Authorization: Bearer $OPENAI_API_KEY
Content-Type: application/json   (POST only)
```

## 2. Models

### 2.1 Model object

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | yes | Model identifier, referenced in API endpoints. |
| `created` | number | yes | Unix timestamp (seconds) when the model was created. |
| `object` | `"model"` | yes | Always `"model"`. |
| `owned_by` | string | yes | Organization that owns the model. |

### 2.2 List models — `GET /v1/models`

Lists the currently available models. No query or body parameters.

Response envelope:

| Field | Type | Description |
|---|---|---|
| `object` | `"list"` | Always `"list"`. |
| `data` | array of Model | The models. |

```json
{
  "object": "list",
  "data": [
    {
      "id": "model-id-0",
      "object": "model",
      "created": 1686935002,
      "owned_by": "organization-owner"
    },
    {
      "id": "model-id-1",
      "object": "model",
      "created": 1686935002,
      "owned_by": "openai"
    }
  ]
}
```

### 2.3 Retrieve model — `GET /v1/models/{model}`

Retrieves a single model instance; returns a Model object (2.1).
Path parameter: `model` (string, required) — the model ID.

*Source: [Retrieve model](https://developers.openai.com/api/reference/resources/models/methods/retrieve/index.md).*

### 2.4 Delete fine-tuned model — `DELETE /v1/models/{model}`

Deletes a fine-tuned model the caller owns. Path parameter: `model` (string,
required). Returns a ModelDeleted object:

| Field | Type | Description |
|---|---|---|
| `id` | string | The deleted model ID. |
| `object` | string | Object type (`"model"`). |
| `deleted` | boolean | Deletion status. |

*Source: [Delete model](https://developers.openai.com/api/reference/resources/models/methods/delete/index.md).*

## 3. Input token counting — `POST /v1/responses/input_tokens`

Returns the input token count for a prospective Responses request without
generating anything. The body is a subset of the `POST /v1/responses` create
body (only the parameters that affect the rendered input context).

### 3.1 Request body

Every parameter is optional (the SDK allows calling with no arguments at all).
All nested types are identical to the Responses create body; section refs
below point into [openai-responses.md](./openai-responses.md).

| Field | Type | Required | Description |
|---|---|---|---|
| `model` | string | no | Model ID used to render/count the input, e.g. `gpt-4o`, `o3`. |
| `input` | string \| array of InputItem | no | Text, image, or file inputs. Same polymorphic union of input item types as the create body (§2.1): messages, `function_call`/`function_call_output`, `reasoning`, `item_reference`, built-in tool calls and outputs, MCP items, shell/apply-patch items, etc. |
| `instructions` | string | no | System (or developer) message inserted into context. Not carried over from the previous response when combined with `previous_response_id`. |
| `conversation` | string \| `{id: string}` | no | Conversation the response belongs to; its items are prepended to `input`. Mutually exclusive with `previous_response_id`. |
| `previous_response_id` | string | no | ID of the previous response for multi-turn state. Mutually exclusive with `conversation`. |
| `tools` | array of Tool | no | Tool definitions (they consume input tokens). Same union as the create body (§2.2): `function`, `file_search`, `computer_use_preview`/`computer-use`, `web_search`/`web_search_preview`, `mcp`, `code_interpreter`, `programmatic_tool_calling`, `image_generation`, `local_shell`, `shell`, `custom`, `namespace`, `tool_search`, `apply_patch`. |
| `tool_choice` | string \| object | no | Same union as the create body (§2.3): `"none"` \| `"auto"` \| `"required"`, or an object — `allowed_tools`, hosted tool type, `function`, `mcp`, `custom`, `programmatic_tool_calling`, `apply_patch`, `shell`. |
| `parallel_tool_calls` | boolean | no | Whether the model may run tool calls in parallel. |
| `text` | object | no | Output text config, same as create body (§2.4): `format` (`text` \| `json_schema` \| `json_object` object) and `verbosity` (`"low"` \| `"medium"` \| `"high"`). |
| `reasoning` | object | no | Reasoning config (gpt-5 / o-series), same as create body (§2.5); fields below. |
| `personality` | string | no | Model-owned style preset; ≤ 64 chars. Known values: `"friendly"`, `"pragmatic"` (free-form string otherwise; set may expand). |
| `truncation` | `"auto"` \| `"disabled"` | no | `auto`: drop items from the start of the conversation when the context window overflows; `disabled` (default): request fails with a 400 instead. |

`reasoning` subfields:

| Field | Type | Description |
|---|---|---|
| `effort` | string | `"none"` \| `"minimal"` \| `"low"` \| `"medium"` \| `"high"` \| `"xhigh"` \| `"max"`. Model support varies. |
| `summary` | string | `"auto"` \| `"concise"` \| `"detailed"`. Reasoning summary verbosity. |
| `generate_summary` | string | Deprecated; use `summary`. Same values. |
| `context` | string | `"auto"` \| `"current_turn"` \| `"all_turns"` — which reasoning items are rendered back on later turns. |
| `mode` | string | Execution mode; known values `"standard"` \| `"pro"` (free-form string otherwise). |

Create-body parameters **not accepted** here (generation, sampling, storage,
and caching controls that do not change the counted input): `background`,
`context_management`, `include`, `max_output_tokens`, `max_tool_calls`,
`metadata`, `moderation`, `prompt`, `prompt_cache_key`, `prompt_cache_options`,
`prompt_cache_retention`, `safety_identifier`, `service_tier`, `store`,
`stream`, `stream_options`, `temperature`, `top_p`, `top_logprobs`, `user`.

### 3.2 Response object

| Field | Type | Description |
|---|---|---|
| `object` | `"response.input_tokens"` | Always `"response.input_tokens"`. |
| `input_tokens` | integer | Number of input tokens the request would consume. |

### 3.3 Example

```http
curl https://api.openai.com/v1/responses/input_tokens \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5",
    "input": "Tell me a joke."
  }'
```

Response:

```json
{
  "object": "response.input_tokens",
  "input_tokens": 11
}
```

## 4. Related

- [openai-responses.md](./openai-responses.md) — Responses create body:
  input item types (§2.1), tools (§2.2), `tool_choice` (§2.3), `text` (§2.4),
  `reasoning` (§2.5); Response object and `usage`.
- [openai-chat-completions.md](./openai-chat-completions.md) — Chat
  Completions API (no equivalent token-counting endpoint).
