# OpenAI Responses API — WebSocket Mode

WebSocket transport for the Responses API, aimed at long-running, tool-call-heavy
workflows (agentic loops). A persistent connection is kept to `/v1/responses`;
each turn sends only new input items plus `previous_response_id`, avoiding
per-request connection/validation overhead (up to ~40% faster end-to-end for
rollouts with 20+ tool calls).

This doc covers the WebSocket-specific surface only. Request body fields,
input/output item types, and the standard streaming events are shared with the
HTTP API — see [`./openai-responses.md`](./openai-responses.md).

Sources: the official
[WebSocket mode guide](https://developers.openai.com/api/docs/guides/websocket-mode.md) and
[WebSocket events reference](https://developers.openai.com/api/reference/resources/responses/websocket-events.md).
OpenAPI schema names in the source carry a `Beta` prefix (resource `beta.responses`).

## Connection

| Aspect | Value |
|---|---|
| URL | `wss://api.openai.com/v1/responses` |
| Auth | `Authorization: Bearer <OPENAI_API_KEY>` header on the upgrade request |
| Framing | JSON text messages in both directions, discriminated by top-level `type` |
| Duration limit | 60 minutes per connection; then reconnect (`websocket_connection_limit_reached`) |
| Concurrency | One in-flight response per connection; multiple `response.create` messages run sequentially |
| Multiplexing | Not supported today; use multiple connections for parallel runs (`stream_id` field is reserved for when multiplexing is enabled separately) |
| Data retention | Compatible with `store=false` and Zero Data Retention (ZDR) |

No WebSocket subprotocol or extra beta header is documented; the official
examples authenticate with the bearer header only.

### Differences from HTTP + SSE

- `stream` is implicit over WebSocket and should not be sent (`stream_options`
  is tied to `stream` and likewise does not apply).
- `background` is not supported over WebSocket.
- Server events and their ordering are identical to the HTTP streaming event
  model; events arrive as complete JSON messages instead of SSE frames.
- Two extra client events (`response.create`, `response.inject`) and two extra
  server events (`response.inject.created`, `response.inject.failed`) exist
  only on this transport.

### Continuation semantics

Chaining uses the same `previous_response_id` semantics as HTTP, plus a
connection-local fast path:

- The server keeps the **most recent** response state of the connection in an
  in-memory cache (response object, prior input/output items, tool definitions,
  rendered tokens). Continuing from that ID is fast and needs no disk storage,
  which is why `store=false`/ZDR works.
- Cache miss on `previous_response_id`:
  - `store=true`: the server may hydrate the ID from persisted state (works,
    but loses the in-memory latency benefit).
  - `store=false`/ZDR: no fallback; the request fails with
    `previous_response_not_found`.
- If a turn fails (4xx or 5xx), the referenced `previous_response_id` is
  evicted from the connection-local cache.
- Warmup: send `response.create` with `generate: false` to prepare request
  state (tools, instructions, messages) without generating output. It returns a
  response ID that can be chained via `previous_response_id`.

### Compaction

- Server-side compaction (`context_management` with `compact_threshold`)
  happens during normal generation; continue as usual with the latest
  `previous_response_id` and new input items.
- The standalone HTTP `POST /v1/responses/compact` endpoint returns a compacted
  input window (not a response ID). Start a **new** chain on the socket: omit
  `previous_response_id` (or set it to `null`) and pass the compacted output
  as-is (do not prune it) as the base of `input`, followed by new user/tool
  items.

### Reconnect and recover

When the connection closes (or hits the 60-minute limit), open a new connection
and continue with one of:

1. `store=true` and a valid response ID: continue with `previous_response_id`
   plus new input items.
2. Chain cannot continue (`store=false`/ZDR, or `previous_response_not_found`):
   start a new response with `previous_response_id` omitted/`null` and the full
   input context.
3. After `/responses/compact`: use the compacted window as the base `input` of
   the new response, then append the latest user/tool items.

## Client events

Union `BetaResponsesClientEvent` = `response.create` | `response.inject`.

### `response.create`

Creates a response on the connection. Schema
`BetaResponsesClientEventResponseCreate`: the payload uses the **same top-level
fields as the `POST /v1/responses` body** (see
[`./openai-responses.md`](./openai-responses.md) for full field semantics),
plus the `type` discriminator. All fields except `type` are optional;
nullability follows the HTTP create body.

| Field | Type | Notes |
|---|---|---|
| `type` | `"response.create"` | Required discriminator. |
| `background` | boolean | Not supported over WS; do not send. |
| `context_management` | array of objects | Server-side compaction config. |
| `conversation` | string \| object | |
| `generate` | boolean | Guide-documented: `false` = warmup only (no model output; returns a chainable response ID). Not listed in the events reference schema. |
| `include` | array of enum strings | |
| `input` | string \| array of input items | |
| `instructions` | string | |
| `max_output_tokens` | integer | |
| `max_tool_calls` | integer | |
| `metadata` | map<string, string> | |
| `model` | string | |
| `moderation` | object | |
| `multi_agent` | object | |
| `parallel_tool_calls` | boolean | |
| `previous_response_id` | string | Continuation; served from the connection-local cache when it is the most recent response. |
| `prompt` | object | Prompt template reference. |
| `prompt_cache_key` | string | |
| `prompt_cache_options` | object | |
| `prompt_cache_retention` | enum | Deprecated; use `prompt_cache_options.ttl`. |
| `reasoning` | object | |
| `safety_identifier` | string | |
| `service_tier` | enum | |
| `store` | boolean | |
| `stream` | boolean | Implicit over WS; do not send. |
| `stream_options` | object | Tied to `stream`; do not send. |
| `temperature` | number | |
| `text` | object | Output text/format config. |
| `tool_choice` | string \| object | |
| `tools` | array of tool objects | |
| `top_logprobs` | integer | |
| `top_p` | number | |
| `truncation` | enum | |
| `user` | string | Deprecated; use `safety_identifier` / `prompt_cache_key`. |

Note: the events reference schema still lists `background`, `stream`, and
`stream_options` (inherited from the HTTP body), but the transport notes say
they are not used over WebSocket.

Minimal example:

```json
{ "type": "response.create", "model": "gpt-5.5", "input": "Say hello." }
```

Continuation turn (tool output + next user message only):

```json
{
  "type": "response.create",
  "model": "gpt-5.6",
  "store": false,
  "previous_response_id": "resp_123",
  "input": [
    { "type": "function_call_output", "call_id": "call_123", "output": "tool result" },
    { "type": "message", "role": "user",
      "content": [{ "type": "input_text", "text": "Now optimize it." }] }
  ],
  "tools": []
}
```

### `response.inject`

Schema `BetaResponseInjectEvent`. Injects input items into an **active**
response. The items are validated and committed atomically. Currently the
server accepts client-owned tool outputs that resume a waiting agent. The
server answers with `response.inject.created` on success or
`response.inject.failed` on failure.

| Field | Type | Required | Description |
|---|---|---|---|
| `type` | `"response.inject"` | yes | Event discriminator. |
| `response_id` | string | yes | ID of the active response that should receive the input. |
| `input` | array of input items | yes | Items to inject. Same input-item union as the create request's `input` array (message, function_call_output, computer_call_output, custom_tool_call_output, MCP approval response, etc. — 35 variants; see [`./openai-responses.md`](./openai-responses.md)). |

```json
{
  "type": "response.inject",
  "response_id": "resp_123",
  "input": [
    { "type": "function_call_output", "call_id": "call_123", "output": "{\"temperature\":72}" }
  ]
}
```

## Server events (WebSocket only)

### `response.inject.created`

Schema `BetaResponseInjectCreatedEvent`. Emitted when all injected input items
were validated and committed to the active response.

| Field | Type | Required | Description |
|---|---|---|---|
| `type` | `"response.inject.created"` | yes | Event discriminator. |
| `response_id` | string | yes | ID of the response that accepted the input. |
| `sequence_number` | integer | yes | Sequence number for this event. |
| `stream_id` | string | no | Multiplexed WS stream that emitted the event; present only when WebSocket multiplexing is enabled separately. |

```json
{ "type": "response.inject.created", "response_id": "resp_123", "sequence_number": 8 }
```

### `response.inject.failed`

Schema `BetaResponseInjectFailedEvent`. Emitted when injected input could not
be committed. Returns the uncommitted raw input so the client can retry it in
another response when appropriate.

| Field | Type | Required | Description |
|---|---|---|---|
| `type` | `"response.inject.failed"` | yes | Event discriminator. |
| `response_id` | string | yes | ID of the response that rejected the input. |
| `input` | array of input items | yes | The raw input items that were not committed (same union as `response.inject.input`). |
| `error` | object | yes | Why the input was not committed. |
| `error.code` | enum | yes | `"response_already_completed"` \| `"response_not_found"`. |
| `error.message` | string | yes | Human-readable description. |
| `sequence_number` | integer | yes | Sequence number for this event. |
| `stream_id` | string | no | Present only when WebSocket multiplexing is enabled separately. |

```json
{
  "type": "response.inject.failed",
  "response_id": "resp_123",
  "input": [
    { "type": "function_call_output", "call_id": "call_123", "output": "{\"temperature\":72}" }
  ],
  "error": {
    "code": "response_already_completed",
    "message": "Response 'resp_123' has already completed."
  },
  "sequence_number": 9
}
```

## Standard server events

All other server events use **the same payloads over WebSocket and HTTP
streaming (SSE)** — shapes, `sequence_number`, and ordering are identical to
the streaming event model. See [`./openai-responses.md`](./openai-responses.md)
for per-event schemas. Since a connection runs one response at a time, all
events between one `response.create` and its terminal lifecycle event belong to
that response; lifecycle events embed the full `response` object (with its
`id`), and item-level events carry `item_id`/`output_index`.

Complete list of event types (grouped):

- Lifecycle: `response.created`, `response.queued`, `response.in_progress`,
  `response.completed`, `response.failed`, `response.incomplete`
- Output items: `response.output_item.added`, `response.output_item.done`
- Content parts: `response.content_part.added`, `response.content_part.done`
- Output text: `response.output_text.delta`, `response.output_text.done`,
  `response.output_text.annotation.added`
- Refusal: `response.refusal.delta`, `response.refusal.done`
- Function calls: `response.function_call_arguments.delta`,
  `response.function_call_arguments.done`
- Custom tool calls: `response.custom_tool_call_input.delta`,
  `response.custom_tool_call_input.done`
- File search: `response.file_search_call.in_progress`,
  `response.file_search_call.searching`, `response.file_search_call.completed`
- Web search: `response.web_search_call.in_progress`,
  `response.web_search_call.searching`, `response.web_search_call.completed`
- Reasoning: `response.reasoning_summary_part.added`,
  `response.reasoning_summary_part.done`,
  `response.reasoning_summary_text.delta`,
  `response.reasoning_summary_text.done`, `response.reasoning_text.delta`,
  `response.reasoning_text.done`
- Image generation: `response.image_generation_call.in_progress`,
  `response.image_generation_call.generating`,
  `response.image_generation_call.partial_image`,
  `response.image_generation_call.completed`
- MCP: `response.mcp_call_arguments.delta`, `response.mcp_call_arguments.done`,
  `response.mcp_call.in_progress`, `response.mcp_call.completed`,
  `response.mcp_call.failed`, `response.mcp_list_tools.in_progress`,
  `response.mcp_list_tools.completed`, `response.mcp_list_tools.failed`
- Code interpreter: `response.code_interpreter_call.in_progress`,
  `response.code_interpreter_call.interpreting`,
  `response.code_interpreter_call.completed`,
  `response.code_interpreter_call_code.delta`,
  `response.code_interpreter_call_code.done`
- Audio: `response.audio.delta`, `response.audio.done`,
  `response.audio.transcript.delta`, `response.audio.transcript.done`
- Error: `error`

## Errors

Two kinds of error messages appear on the socket.

In-stream `error` event — identical to the SSE `error` streaming event
(`{ "type": "error", "code": string|null, "message": string, "param":
string|null, "sequence_number": integer }`; see
[`./openai-responses.md`](./openai-responses.md)).

Request-level errors — WS-specific failures use an envelope with an HTTP-style
`status` and a nested `error` object, as shown in the guide:

`previous_response_not_found` (retry with full input context and
`previous_response_id` set to `null`):

```json
{
  "type": "error",
  "status": 400,
  "error": {
    "code": "previous_response_not_found",
    "message": "Previous response with id 'resp_abc' not found.",
    "param": "previous_response_id"
  }
}
```

`websocket_connection_limit_reached` (open a new connection and continue):

```json
{
  "type": "error",
  "status": 400,
  "error": {
    "type": "invalid_request_error",
    "code": "websocket_connection_limit_reached",
    "message": "Responses websocket connection limit reached (60 minutes). Create a new websocket connection to continue."
  }
}
```

## Example session

```text
client → { "type": "response.create", "model": "gpt-5.6", "store": false,
           "input": [ { "type": "message", "role": "user",
                        "content": [{ "type": "input_text", "text": "Find fizz_buzz()" }] } ],
           "tools": [ ...code_search tool... ] }

server → response.created                      (response.id = "resp_123")
server → response.in_progress
server → response.output_item.added            (item: function_call)
server → response.function_call_arguments.delta  (repeated)
server → response.function_call_arguments.done
server → response.output_item.done             (call_id = "call_123")
server → response.completed                    (output contains the function call)

# client runs the tool locally, then continues on the same socket:

client → { "type": "response.create", "model": "gpt-5.6", "store": false,
           "previous_response_id": "resp_123",
           "input": [ { "type": "function_call_output", "call_id": "call_123",
                        "output": "def fizz_buzz(): ..." } ],
           "tools": [ ...same tools... ] }

server → response.created                      (response.id = "resp_456")
server → response.in_progress
server → response.output_item.added            (item: message)
server → response.content_part.added
server → response.output_text.delta            (repeated)
server → response.output_text.done
server → response.content_part.done
server → response.output_item.done
server → response.completed
```
