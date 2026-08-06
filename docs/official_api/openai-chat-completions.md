# OpenAI Chat Completions API Reference

Condensed from the official API reference:
[Chat API](https://developers.openai.com/api/reference/resources/chat.md),
[Create chat completion](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create/index.md),
[streaming events](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events.md).
Covers only the "Create chat completion" endpoint, its request/response
bodies, and the streaming chunk format.

## Endpoint

```
POST https://api.openai.com/v1/chat/completions
Content-Type: application/json
Authorization: Bearer $OPENAI_API_KEY
```

Returns a `chat.completion` object, or a stream of `chat.completion.chunk` objects
(server-sent events) when `stream: true`.

## Request Body

Top-level parameters. Only `messages` and `model` are required; unnamed nested object
fields are required within their object unless marked optional below.

| Field | Type | Required | Description |
|---|---|---|---|
| `messages` | array of message objects | yes | Conversation so far. See [Messages](#messages). |
| `model` | string | yes | Model ID (e.g. `gpt-4o`, `o3`, `gpt-5.4`). Docs enumerate ~81 known IDs (`gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`, `gpt-5.4`, `gpt-5.2`, `gpt-5.1`, `gpt-5`, `gpt-4.1`, `o4-mini`, `o3`, `o1`, `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`, dated variants, ...) but the type is a plain string union, so any model ID string is accepted. |
| `audio` | object \| null | no | Audio output params; required when `modalities: ["audio"]`. See [audio](#audio-output-parameters). |
| `frequency_penalty` | number \| null | no | -2.0 to 2.0. Positive values penalize tokens by frequency so far. |
| `function_call` | `"none"` \| `"auto"` \| `{name: string}` | no | Deprecated; use `tool_choice`. |
| `functions` | array of `{name, description?, parameters?}` | no | Deprecated; use `tools`. `name` string (required), `description` string, `parameters` JSON Schema map. |
| `logit_bias` | map<string, number> \| null | no | Token ID -> bias (-100 to 100) added to logits. |
| `logprobs` | boolean \| null | no | Return log probabilities of output tokens. |
| `max_completion_tokens` | number \| null | no | Upper bound on generated tokens, including reasoning tokens. |
| `max_tokens` | number \| null | no | Deprecated in favor of `max_completion_tokens`; incompatible with o-series models. |
| `metadata` | map<string, string> \| null | no | Up to 16 pairs. Keys <= 64 chars, values <= 512 chars. |
| `modalities` | array of `"text"` \| `"audio"`, or null | no | Output types to generate. Default `["text"]`. |
| `moderation` | object \| null | no | Moderated-completions config: `model` string (required, e.g. `omni-moderation-latest`); `policy?` object \| null with `input?`/`output?` objects \| null, each `{mode: "score" \| "block"}`. |
| `n` | number \| null | no | Number of choices per input. Default 1. |
| `parallel_tool_calls` | boolean | no | Enable parallel function calling during tool use. |
| `prediction` | object \| null | no | Predicted output: `{type: "content", content: string \| ChatCompletionContentPartText[]}`. |
| `presence_penalty` | number \| null | no | -2.0 to 2.0. Positive values penalize tokens already present. |
| `prompt_cache_key` | string \| null | no | Cache-bucketing key; replaces `user`. |
| `prompt_cache_options` | object | no | `gpt-5.6`+ prompt caching. `mode?`: `"implicit"` (default; one implicit breakpoint + up to 3 explicit) or `"explicit"` (no implicit breakpoint, up to 4 explicit). `ttl?`: `"30m"` (only supported value; a minimum lifetime). |
| `prompt_cache_retention` | `"in_memory"` \| `"24h"` \| null | no | Deprecated; use `prompt_cache_options.ttl`. Max-retention policy; only `24h` for `gpt-5.5`+. |
| `reasoning_effort` | enum \| null | no | `"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`, `"max"`. Model support varies. |
| `response_format` | object | no | See [response_format](#response_format). |
| `safety_identifier` | string \| null | no | Stable end-user ID (<= 64 chars, hash recommended) for abuse detection. |
| `seed` | number \| null | no | Beta. Best-effort determinism; check `system_fingerprint`. |
| `service_tier` | enum \| null | no | `"auto"` (default), `"default"`, `"flex"`, `"scale"`, `"priority"`, `"fast"`. Response echoes the tier actually used (`fast` is reported as `priority`). |
| `stop` | string \| string[] \| null | no | Up to 4 stop sequences. Not supported by `o3`/`o4-mini`. |
| `store` | boolean \| null | no | Store output for distillation/evals. Image inputs > 8MB dropped. |
| `stream` | boolean \| null | no | Stream response as SSE. |
| `stream_options` | object \| null | no | Only with `stream: true`. See [stream_options](#stream_options). |
| `temperature` | number \| null | no | 0 to 2. Default 1. Alter this or `top_p`, not both. |
| `tool_choice` | enum \| object | no | See [tool_choice](#tool_choice). |
| `tools` | array of tool objects | no | See [tools](#tools). |
| `top_logprobs` | number \| null | no | 0-20; most likely tokens per position. Requires `logprobs: true`. |
| `top_p` | number \| null | no | Nucleus sampling probability mass (e.g. 0.1 = top 10%). |
| `user` | string | no | Deprecated; replaced by `safety_identifier` + `prompt_cache_key`. |
| `verbosity` | `"low"` \| `"medium"` \| `"high"` \| null | no | Response verbosity. Default `"medium"`. |
| `web_search_options` | object | no | See [web_search_options](#web_search_options). |

### Messages

`messages` items are discriminated by `role`: `developer`, `system`, `user`,
`assistant`, `tool`, `function` (deprecated).

| Role | Fields | Notes |
|---|---|---|
| `developer` | `content` (string \| text parts, required), `role`, `name?` | Instructions; replaces `system` for o1+ models. Text parts only. |
| `system` | `content` (string \| text parts, required), `role`, `name?` | Legacy instructions role. Text parts only. |
| `user` | `content` (string \| content parts, required), `role`, `name?` | Parts may be text, image, audio, file. |
| `assistant` | `role`, `content?` (string \| array of text/refusal parts \| null), `audio?` (`{id}` \| null), `function_call?` (deprecated, `{arguments, name}` \| null), `name?`, `refusal?` (string \| null), `tool_calls?` | `content` required unless `tool_calls` or `function_call` present. `audio.id` references a previous audio response. |
| `tool` | `content` (string \| text parts, required), `role`, `tool_call_id` (required) | Result for the tool call it responds to. Text parts only. |
| `function` | `content` (string \| null, required), `name` (required), `role` | Deprecated. |

#### Content parts

Union discriminated by `type`. Every input part also accepts an optional
`prompt_cache_breakpoint: {mode: "explicit"}` marking the end of a reusable prompt
prefix (used with `prompt_cache_options`).

| `type` | Other fields | Allowed in |
|---|---|---|
| `"text"` | `text: string` | developer/system/user/assistant/tool content; `prediction.content` |
| `"image_url"` | `image_url: {url: string (URL or base64 data), detail?: "auto" \| "low" \| "high"}` | user |
| `"input_audio"` | `input_audio: {data: string (base64), format: "wav" \| "mp3"}` | user |
| `"file"` | `file: {file_data?: string (base64), file_id?: string, filename?: string}` | user |
| `"refusal"` | `refusal: string` | assistant (at most one) |

#### Assistant `tool_calls` (in request messages)

Same shapes as in responses:

- Function call: `{id: string, type: "function", function: {name: string, arguments: string (JSON)}}`
- Custom call: `{id: string, type: "custom", custom: {name: string, input: string}}`

### Audio output parameters

`audio` (required when `modalities` includes `"audio"`):

| Field | Type | Required | Description |
|---|---|---|---|
| `format` | enum | yes | `"wav"`, `"aac"`, `"mp3"`, `"flac"`, `"opus"`, `"pcm16"`. |
| `voice` | string \| enum \| `{id: string}` | yes | Built-in voices: `alloy`, `ash`, `ballad`, `coral`, `echo`, `sage`, `shimmer`, `verse`, `marin`, `cedar` (prose also lists `fable`, `nova`, `onyx`). Or a custom voice object `{"id": "voice_1234"}`. |

### response_format

Union discriminated by `type`:

| Variant | Shape |
|---|---|
| Text (default) | `{type: "text"}` |
| JSON Schema (Structured Outputs) | `{type: "json_schema", json_schema: {name: string (required, <= 64 chars of a-zA-Z0-9_-), description?: string, schema?: map (JSON Schema), strict?: boolean \| null}}` |
| JSON object (legacy JSON mode) | `{type: "json_object"}` |

### stream_options

| Field | Type | Required | Description |
|---|---|---|---|
| `include_obfuscation` | boolean | no | Default true. Adds a random-character `obfuscation` field to streaming delta events to normalize payload sizes (side-channel mitigation). Set false to save bandwidth. |
| `include_usage` | boolean | no | Streams one extra chunk before `data: [DONE]` with `usage` populated and empty `choices`; all other chunks carry `usage: null`. |

### tool_choice

String mode or object union:

| Variant | Shape | Meaning |
|---|---|---|
| Mode | `"none"` \| `"auto"` \| `"required"` | Default: `none` without tools, `auto` with tools. |
| Allowed tools | `{type: "allowed_tools", allowed_tools: {mode: "auto" \| "required", tools: array of tool-reference maps}}` | Restrict to a subset, e.g. `{"type": "function", "function": {"name": "get_weather"}}` entries. |
| Named function | `{type: "function", function: {name: string}}` | Force a specific function. |
| Named custom | `{type: "custom", custom: {name: string}}` | Force a specific custom tool. |

### tools

Array of a union discriminated by `type`:

Function tool:

| Field | Type | Required | Description |
|---|---|---|---|
| `type` | `"function"` | yes | |
| `function.name` | string | yes | `a-zA-Z0-9_-`, <= 64 chars. |
| `function.description` | string | no | Helps the model choose when to call. |
| `function.parameters` | map (JSON Schema) | no | Omitting it means an empty parameter list. |
| `function.strict` | boolean \| null | no | Strict schema adherence (Structured Outputs subset). |

Custom tool (free-form or grammar-constrained input):

| Field | Type | Required | Description |
|---|---|---|---|
| `type` | `"custom"` | yes | |
| `custom.name` | string | yes | |
| `custom.description` | string | no | |
| `custom.format` | object | no | `{type: "text"}` (default, unconstrained) or `{type: "grammar", grammar: {definition: string, syntax: "lark" \| "regex"}}`. |

### web_search_options

| Field | Type | Required | Description |
|---|---|---|---|
| `search_context_size` | `"low"` \| `"medium"` \| `"high"` | no | Default `"medium"`. |
| `user_location` | object \| null | no | `{type: "approximate", approximate: {city?, country? (ISO 3166-1 two-letter), region?, timezone? (IANA)}}`; `approximate` is required inside, its fields optional strings. |

## Response (non-streaming): `chat.completion`

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | yes | Unique completion ID. |
| `choices` | array | yes | One per requested `n`. |
| `created` | number | yes | Unix timestamp (seconds). |
| `model` | string | yes | Model used. |
| `object` | `"chat.completion"` | yes | |
| `moderation` | object \| null | no | Present when moderated completions requested. `input`/`output` each: `{type: "moderation_results", model: string, results: [{categories: map<string,bool>, category_applied_input_types: map<string, ("text" \| "image")[]>, category_scores: map<string,number>, flagged: bool, model: string, type: "moderation_result"}]}` or error `{type: "error", code: string, message: string}`. |
| `service_tier` | enum \| null | no | `"auto"`, `"default"`, `"flex"`, `"scale"`, `"priority"`, `"fast"` — tier actually used. |
| `system_fingerprint` | string | no | Backend config fingerprint; compare with `seed` for determinism tracking. |
| `usage` | object | no | See [usage](#usage). |

### `choices[]`

| Field | Type | Required | Description |
|---|---|---|---|
| `finish_reason` | enum | yes | `"stop"`, `"length"`, `"tool_calls"`, `"content_filter"`, `"function_call"` (deprecated). |
| `index` | number | yes | Choice index. |
| `logprobs` | object \| null | yes (nullable) | `{content: TokenLogprob[] \| null, refusal: TokenLogprob[] \| null}`. |
| `message` | object | yes | See below. |

`TokenLogprob`: `{token: string, bytes: number[] \| null (UTF-8 bytes), logprob: number
(-9999.0 if outside top 20), top_logprobs: [{token, bytes, logprob}]}`.

### `choices[].message` (assistant message)

| Field | Type | Required | Description |
|---|---|---|---|
| `content` | string \| null | yes (nullable) | Message text. |
| `refusal` | string \| null | yes (nullable) | Refusal message. |
| `role` | `"assistant"` | yes | |
| `annotations` | array | no | Web-search citations: `{type: "url_citation", url_citation: {start_index: number, end_index: number, title: string, url: string}}`. |
| `audio` | object \| null | no | Audio output: `{id: string, data: string (base64), expires_at: number (Unix s), transcript: string}`. |
| `function_call` | object | no | Deprecated: `{arguments: string, name: string}`. |
| `tool_calls` | array | no | Function calls `{id, type: "function", function: {name, arguments (JSON string, may be invalid)}}` or custom calls `{id, type: "custom", custom: {name, input}}`. |

### usage

| Field | Type | Required | Description |
|---|---|---|---|
| `completion_tokens` | number | yes | Generated tokens. |
| `prompt_tokens` | number | yes | Prompt tokens. |
| `total_tokens` | number | yes | Prompt + completion. |
| `completion_tokens_details` | object | no | `accepted_prediction_tokens?`, `audio_tokens?`, `reasoning_tokens?`, `rejected_prediction_tokens?` (all numbers; rejected prediction tokens still billed as completion tokens). |
| `prompt_tokens_details` | object | no | `audio_tokens?`, `cache_write_tokens?` (unadjusted tokens written to cache), `cached_tokens?`. |

### Example

Request:

```json
{
  "model": "gpt-4o-mini",
  "messages": [
    {"role": "developer", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
}
```

Response:

```json
{
  "id": "chatcmpl-B9MBs8CjcvOU2jLn4n570S5qMJKcT",
  "object": "chat.completion",
  "created": 1741569952,
  "model": "gpt-4o-mini",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I assist you today?",
        "refusal": null,
        "annotations": []
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 19,
    "completion_tokens": 10,
    "total_tokens": 29,
    "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
    "completion_tokens_details": {
      "reasoning_tokens": 0,
      "audio_tokens": 0,
      "accepted_prediction_tokens": 0,
      "rejected_prediction_tokens": 0
    }
  },
  "service_tier": "default"
}
```

Tool-call response (`finish_reason: "tool_calls"`, `content: null`):

```json
{
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {"name": "get_current_weather", "arguments": "{\"location\": \"Boston, MA\"}"}
          }
        ]
      },
      "logprobs": null,
      "finish_reason": "tool_calls"
    }
  ]
}
```

## Streaming: `chat.completion.chunk`

With `stream: true`, the API sends SSE messages, each `data:` line holding one
`chat.completion.chunk` JSON object. The stream ends with a literal `data: [DONE]`
message. Unless `stream_options.include_obfuscation` is false, delta events carry an
extra `obfuscation` field of random characters (padding; ignore it when parsing).

### Chunk object

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | yes | Same ID in every chunk of a stream. |
| `choices` | array | yes | May hold multiple entries when `n > 1`; empty for the final usage chunk when `include_usage` is set. |
| `created` | number | yes | Same timestamp in every chunk. |
| `model` | string | yes | |
| `object` | `"chat.completion.chunk"` | yes | |
| `moderation` | object \| null | no | Same shape as non-streaming; present on the moderation chunk when requested. |
| `service_tier` | enum \| null | no | Same values as non-streaming. |
| `system_fingerprint` | string | no | |
| `usage` | object \| null | no | Only present when `stream_options: {"include_usage": true}`: `null` on every chunk except the last, which carries full request usage (same shape as non-streaming `usage`). If the stream is interrupted, the usage chunk may never arrive. |

### `choices[]` (chunk)

| Field | Type | Required | Description |
|---|---|---|---|
| `delta` | object | yes | Incremental message fragment. |
| `finish_reason` | enum \| null | yes (nullable) | `null` until the choice finishes; then `"stop"`, `"length"`, `"tool_calls"`, `"content_filter"`, or `"function_call"`. |
| `index` | number | yes | Choice index. |
| `logprobs` | object \| null | no | Same `{content, refusal}` token-logprob shape as non-streaming. |

### `choices[].delta`

All fields optional; concatenate fragments per choice `index`.

| Field | Type | Description |
|---|---|---|
| `content` | string \| null | Text fragment. |
| `refusal` | string \| null | Refusal text fragment. |
| `role` | enum | `"developer"`, `"system"`, `"user"`, `"assistant"`, `"tool"`; sent once in the first delta (in practice `"assistant"`). |
| `function_call` | object | Deprecated: `{arguments?: string, name?: string}` fragments. |
| `tool_calls` | array | Tool-call fragments; see below. |

`delta.tool_calls[]` items:

| Field | Type | Required | Description |
|---|---|---|---|
| `index` | number | yes | Position of the tool call in the message's `tool_calls` array; use it to merge fragments (ids/names arrive in the first fragment, argument pieces in later ones). |
| `id` | string | no | Tool call ID (first fragment only). |
| `type` | `"function"` | no | Only `function` is documented for deltas. |
| `function` | object | no | `{name?: string, arguments?: string}` — `arguments` arrives as incremental JSON string pieces to concatenate. |

### Example SSE sequence (text)

From the official docs (each line is one `data:` payload):

```json
{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4o-mini","system_fingerprint":"fp_44709d6fcb","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}

{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4o-mini","system_fingerprint":"fp_44709d6fcb","choices":[{"index":0,"delta":{"content":"Hello"},"logprobs":null,"finish_reason":null}]}

{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4o-mini","system_fingerprint":"fp_44709d6fcb","choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop"}]}
```

followed by:

```
data: [DONE]
```

### Example delta sequence (tool call, illustrative)

Deltas only (chunk envelope omitted), showing index-based argument streaming:

```json
{"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_abc123","type":"function","function":{"name":"get_current_weather","arguments":""}}]},"finish_reason":null}
{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"location\":"}}]},"finish_reason":null}
{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Boston, MA\"}"}}]},"finish_reason":null}
{"delta":{},"finish_reason":"tool_calls"}
```

With `include_usage: true`, one final chunk precedes `[DONE]` with `"choices": []` and a
populated `usage` object.
