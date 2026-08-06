# Google Gemini Interactions API Reference

Compiled from the official
[Interactions API reference](https://ai.google.dev/api/interactions-api.md.txt) and
[streaming guide](https://ai.google.dev/gemini-api/docs/streaming.md.txt)
(v1beta reference, 2026). Development reference for implementing types/clients.

Note: all JSON field names in this API are **snake_case** (e.g. `previous_interaction_id`,
`mime_type`, `event_type`), unlike the camelCase `generateContent` API.

## 1. Overview

The Interactions API is the unified endpoint for calling Gemini **models** and managed
**agents** (Deep Research, Antigravity, ...). Each call creates an `Interaction` resource:
one conversation turn/task stored server-side, containing a chronological `steps` timeline
(user input, thoughts, tool calls/results, model output).

- Base URL: `https://generativelanguage.googleapis.com/v1beta/interactions` (beta; a stable
  `v1` version also exists).
- Auth: API key via `x-goog-api-key: $GEMINI_API_KEY` header.
- State: interactions are stored by default (`store=true`; retention 55 days paid tier,
  1 day free tier). Chain turns via `previous_interaction_id`; only conversation history is
  carried over — `tools`, `system_instruction` and `generation_config` are per-request and
  must be re-sent each turn. `store=false` opts out but disables chaining and background runs.

### Endpoints

| Method | Path | Semantics |
|---|---|---|
| POST | `/v1beta/interactions` | Create an interaction (optionally streaming/background). |
| GET | `/v1beta/interactions/{id}` | Retrieve full interaction. Query: `stream` (bool, default false), `last_event_id` (string; resume SSE stream after that event, requires `stream=true`). |
| POST | `/v1beta/interactions/{id}/cancel` | Cancel a still-running **background** interaction. Returns the Interaction (`status: "cancelled"`). |
| DELETE | `/v1beta/interactions/{id}` | Delete a stored interaction. Empty response on success. |

The create response contains only model-generated steps; the stored resource (via GET) also
includes `user_input` steps.

## 2. Create interaction — request body

Exactly one of `model` / `agent` is required.

| Field | Type | Required | Description |
|---|---|---|---|
| `model` | string (ModelOption) | if no `agent` | Model name. |
| `agent` | string (AgentOption) | if no `model` | Agent name. |
| `input` | string \| Content \| Content[] \| Step[] \| Turn[] | yes | The inputs (see below). |
| `system_instruction` | string | no | System instruction. |
| `tools` | Tool[] | no | Tool declarations the model may call (section 7). |
| `response_format` | ResponseFormat \| ResponseFormat[] | no | Output format constraints, incl. JSON schema (section 8). |
| `stream` | boolean | no | Input only. Stream response via SSE (section 10). |
| `store` | boolean | no | Input only. Store request/response for later retrieval (default true). |
| `background` | boolean | no | Input only. Run in the background (poll via GET). |
| `generation_config` | GenerationConfig | no | Model configuration; only when `model` is set (alternative to `agent_config`). |
| `agent_config` | AgentConfig (union) | no | Agent configuration; only when `agent` is set. |
| `environment` | EnvironmentConfig \| string | no | Remote environment spec, or an existing environment ID string (section 9). |
| `labels` | object | no | User-defined metadata labels. |
| `previous_interaction_id` | string | no | ID of the previous interaction to continue from. |
| `response_modalities` | ResponseModality[] | no | Requested output modalities. |
| `safety_settings` | SafetySetting[] | no | Safety-blocking behavior. |
| `service_tier` | string enum | no | `flex` \| `standard` \| `priority`. |
| `webhook_config` | WebhookConfig | no | Webhook notification on completion. |

Response: an [Interaction](#3-interaction-resource-response) resource (non-streaming), or an
SSE event stream when `stream: true`.

### Model / agent options

`ModelOption` values documented (list evolves): `gemini-2.5-flash`, `gemini-2.5-pro`,
`gemini-2.5-flash-lite`, `gemini-2.5-flash-image`, `gemini-flash-latest`,
`gemini-flash-lite-latest`, `gemini-pro-latest`, `gemini-3-flash-preview`,
`gemini-3.1-pro-preview`, `gemini-3.1-pro-preview-customtools`, `gemini-3.1-flash-lite`,
`gemini-3-pro-image`, `nano-banana-pro-preview`, `gemini-3.1-flash-image`,
`gemini-3.5-flash`, `gemini-3.6-flash`, `gemma-4-26b-a4b-it`, `gemma-4-31b-it`,
`lyria-3-clip-preview`, `lyria-3-pro-preview`, `gemini-robotics-er-1.6-preview`,
`gemini-robotics-er-2-preview`. Model as an open string in client types.

`AgentOption` values: `deep-research-pro-preview-12-2025`, `deep-research-preview-04-2026`,
`deep-research-max-preview-04-2026`, `antigravity-preview-05-2026`.

### `input` polymorphism

| Form | Meaning |
|---|---|
| string | Plain text prompt. |
| Content object | Single content block (`{"type": "text", ...}` etc., section 6). |
| Content[] | Multimodal user input (e.g. image + text). |
| Step[] | Full/partial timeline items — used to send `function_result` steps back, or replay history statelessly (step types in section 5). |
| Turn[] | Role-based conversation history. |

`Turn` is referenced but not defined in the v1beta reference; the underlying schema
(deprecated there) defines it as `{ "role": "user"|"model", "content": string | Content[] }`.
Prefer `previous_interaction_id` or `Step[]` history.

### `generation_config` (GenerationConfig)

| Field | Type | Required | Description |
|---|---|---|---|
| `max_output_tokens` | integer | no | Max tokens in the response. |
| `seed` | integer | no | Decoding seed for reproducibility. |
| `speech_config` | SpeechConfig[] | no | Speech synthesis config: `{language?, speaker?, voice?}` (speaker matches name in prompt). |
| `stop_sequences` | string[] | no | Sequences that stop generation. |
| `thinking_level` | enum | no | `minimal` \| `low` \| `medium` \| `high`. |
| `thinking_summaries` | enum | no | `auto` \| `none` — include thought summaries. |
| `tool_choice` | ToolChoiceConfig \| enum | no | Enum: `auto` \| `any` \| `none` \| `validated`; or object `{"allowed_tools": {"mode": <enum>, "tools": ["name", ...]}}`. |
| `transcription_config` | TranscriptionConfig | no | Enables ASR: `{custom_vocabulary?: string[], diarization_mode?: string ("speaker"), language_codes?: string[] (BCP-47), timestamp_granularities?: string[] ("word")}`. |
| `video_config` | VideoConfig | no | Video generation: `{task?: "text_to_video"\|"image_to_video"\|"reference_to_video"\|"edit"}`. |

Note: no `temperature`/`top_p` are documented in this reference.

### `agent_config` (union, discriminator `type`)

| Variant (`type`) | Fields |
|---|---|
| `"antigravity"` | `max_total_tokens?: string`, `model?: string` (model for agent reasoning). |
| `"deep-research"` | `collaborative_planning?: bool` (human-in-the-loop plan confirmation), `enable_bigquery_tool?: bool`, `thinking_summaries?: "auto"\|"none"`, `visualization?: "off"\|"auto"`. |
| `"code-mender"` | `model?: string`, `find_request?: {description?, finding_id?, mode?: "scan"\|"verify", source_files?: [{content?, path?}]}`, `fix_request?: {description?, finding_id?, source_files?: [...]}`, `session_config?: {max_rounds?: int, session_id?: string}`. |
| `"dynamic"` | (no extra fields). |

### `safety_settings` (SafetySetting)

| Field | Type | Required | Description |
|---|---|---|---|
| `type` | HarmCategory | yes | `hate_speech`, `dangerous_content`, `harassment`, `sexually_explicit`, `civic_integrity` (deprecated), `image_hate`, `image_dangerous_content`, `image_harassment`, `image_sexually_explicit`, `jailbreak`. |
| `threshold` | enum | yes | `block_low_and_above`, `block_medium_and_above`, `block_only_high`, `block_none`, `off`. |
| `method` | enum | no | `severity` \| `probability` (default probability). |

### Other request sub-objects

- `response_modalities` item enum (ResponseModality): `text`, `image`, `audio`, `video`, `document`.
- `webhook_config`: `{uris?: string[], user_metadata?: object}` — `uris` override registered
  webhooks; `user_metadata` is echoed on each webhook event.

### Example request

```json
{
  "model": "gemini-3.6-flash",
  "input": "What is the weather in Paris right now?",
  "system_instruction": "Be terse.",
  "tools": [
    {
      "type": "function",
      "name": "get_weather",
      "description": "Get the current weather in a given location",
      "parameters": {
        "type": "object",
        "properties": { "location": { "type": "string" } },
        "required": ["location"]
      }
    }
  ],
  "generation_config": { "thinking_level": "low", "thinking_summaries": "auto" },
  "store": true
}
```

Follow-up turn returning a tool result:

```json
{
  "model": "gemini-3.6-flash",
  "previous_interaction_id": "v1_ChdGUVFJ...",
  "input": [
    {
      "type": "function_result",
      "name": "get_weather",
      "call_id": "un6k8t18",
      "result": [{ "type": "text", "text": "{\"weather\": \"Sunny and 22°C\"}" }]
    }
  ]
}
```

## 3. Interaction resource (response)

| Field | Type | Description |
|---|---|---|
| `id` | string | Output only. Unique interaction ID (e.g. `v1_Chd...`). |
| `object` | string | Output only. Resource type, `"interaction"`. |
| `status` | enum | Output only, required. See status values below. |
| `created` / `updated` | string | Output only. ISO 8601 timestamps (`YYYY-MM-DDThh:mm:ssZ`). |
| `model` / `agent` | string | Whichever was used for the interaction. |
| `agent_config` | AgentConfig | As requested. |
| `environment` | EnvironmentConfig \| string | As requested. |
| `environment_id` | string | Output only. Populated when an environment config was set. |
| `input` | same union as request | The request input (echoed on stored resource). |
| `labels` | object | User-defined metadata. |
| `previous_interaction_id` | string | As requested. |
| `response_format` | ResponseFormat \| ResponseFormat[] | As requested. |
| `response_modalities` | ResponseModality[] | As requested. |
| `safety_settings` | SafetySetting[] | As requested. |
| `service_tier` | enum | `flex` \| `standard` \| `priority`. |
| `steps` | Step[] | Output only. The interaction timeline (section 5). Create returns model-generated steps; GET also includes `user_input`. |
| `system_instruction` | string | As requested. |
| `tools` | Tool[] | As requested. |
| `usage` | Usage | Output only. Token usage statistics. |
| `webhook_config` | WebhookConfig | As requested. |
| `output_text` | string | SDK-added convenience: concatenated text of the last model output. |
| `output_image` / `output_audio` / `output_video` | ImageContent / AudioContent / VideoContent | SDK-added: last generated media. Not part of the wire response. |

### `status` values

| Value | Meaning |
|---|---|
| `queued` | Queued, waiting for processing. |
| `in_progress` | Running. |
| `requires_action` | Waiting for user action/input (e.g. pending `function_result`). |
| `completed` | Finished successfully. |
| `incomplete` | Finished but truncated (e.g. hit `max_output_tokens`). |
| `budget_exceeded` | Halted: token budget exceeded. |
| `failed` | Failed. |
| `cancelled` | Cancelled. |

### `usage` (Usage)

| Field | Type | Description |
|---|---|---|
| `total_tokens` | integer | Prompt + responses + internal tokens. |
| `total_input_tokens` | integer | Prompt (context) tokens. |
| `total_output_tokens` | integer | All generated response tokens. |
| `total_thought_tokens` | integer | Thinking tokens. |
| `total_cached_tokens` | integer | Cached prompt tokens. |
| `total_tool_use_tokens` | integer | Tool-use prompt tokens. |
| `input_tokens_by_modality` | ModalityTokens[] | Per-modality input breakdown. |
| `output_tokens_by_modality` | ModalityTokens[] | Per-modality output breakdown. |
| `cached_tokens_by_modality` | ModalityTokens[] | Per-modality cached breakdown. |
| `tool_use_tokens_by_modality` | ModalityTokens[] | Per-modality tool-use breakdown. |
| `grounding_tool_count` | GroundingToolCount[] | `{count?: int, type?: "google_search"\|"google_maps"\|"retrieval"}`. |

`ModalityTokens` = `{modality?: ResponseModality, tokens?: integer}`.

### Example response (function calling, non-streaming)

```json
{
  "id": "v1_ChdPU0F4...",
  "object": "interaction",
  "model": "gemini-3.6-flash",
  "status": "requires_action",
  "created": "2025-11-26T12:22:47Z",
  "updated": "2025-11-26T12:22:47Z",
  "steps": [
    {
      "type": "function_call",
      "id": "gth23981",
      "name": "get_weather",
      "arguments": { "location": "Boston, MA" }
    }
  ],
  "usage": {
    "input_tokens_by_modality": [{ "modality": "text", "tokens": 100 }],
    "total_cached_tokens": 0,
    "total_input_tokens": 100,
    "total_output_tokens": 25,
    "total_thought_tokens": 0,
    "total_tokens": 125,
    "total_tool_use_tokens": 50
  }
}
```

## 4. Errors

Failures inside a step surface as `error` on `model_output` steps using google.rpc `Status`:
`{code?: integer, message?: string, details?: object[]}`. Streaming errors use the `error`
SSE event with `{code?: string, message?: string}` where `code` is an identifier such as
`"not_found"` or `"gateway_timeout"` (section 10).

## 5. Step (union, discriminator `type`)

A step is one item in the interaction timeline. `id` identifies a tool call; the matching
result carries `call_id`. `signature` is "a signature hash for backend validation" — an
opaque string; preserve it when resending steps as input.

| `type` | Fields |
|---|---|
| `user_input` | `content?: Content[]`. |
| `model_output` | `content?: Content[]`, `error?: Status`. |
| `thought` | `signature?: string`, `summary?: (TextContent \| ImageContent)[]`. |
| `function_call` | `id: string`, `name: string`, `arguments: object`. |
| `function_result` | `call_id: string`, `result: (TextContent \| ImageContent)[] \| object \| string`, `name?: string`, `is_error?: bool`. |
| `code_execution_call` | `id: string`, `arguments: {code?: string, language?: "python"}`, `signature?`. |
| `code_execution_result` | `call_id: string`, `result: string`, `is_error?: bool`, `signature?`. |
| `google_search_call` | `id: string`, `arguments: {queries?: string[]}`, `search_type?: "web_search"\|"image_search"\|"enterprise_web_search"`, `signature?`. |
| `google_search_result` | `call_id: string`, `result: GoogleSearchResultItem[]`, `is_error?: bool`, `signature?`. |
| `google_maps_call` | `id: string`, `arguments?: {queries?: string[]}`, `signature?`. |
| `google_maps_result` | `call_id: string`, `result: GoogleMapsResultItem[]`, `signature?`. |
| `file_search_call` | `id: string`, `signature?`. |
| `file_search_result` | `call_id: string`, `signature?`. |
| `mcp_server_tool_call` | `id: string`, `name: string`, `server_name: string`, `arguments: object`. |
| `mcp_server_tool_result` | `call_id: string`, `result: (TextContent \| ImageContent)[] \| object \| string`, `name?`, `server_name?`. |
| `url_context_call` | `id: string`, `arguments: {urls?: string[]}`, `signature?`. |
| `url_context_result` | `call_id: string`, `result: UrlContextResult[]`, `is_error?: bool`, `signature?`. |

Result item shapes:

- `GoogleSearchResultItem`: documented field `search_suggestions?: string` (embeddable web
  snippet); examples also show `title`/`url`/`snippet` keys (not formally documented).
- `GoogleMapsResultItem`: `{places?: [{name?, place_id?, url?, review_snippets?: [{review_id?, title?, url?}]}], widget_context_token?: string}`.
- `UrlContextResult`: `{status?: "success"|"error"|"paywall"|"unsafe", url?: string}`.
- `FileSearchResult`: fields not documented in the reference.

Migration-guide examples additionally show a per-step `status` (e.g. `"done"`, `"waiting"`)
not present in the v1beta reference tables.

```json
{ "type": "thought", "signature": "thought_sig_abcd1234",
  "summary": [{ "type": "text", "text": "Searching for the capital of France." }] }
```

```json
{ "type": "model_output",
  "content": [{ "type": "text", "text": "The capital of France is Paris." }] }
```

## 6. Content (union, discriminator `type`)

Content blocks appear in `input`, step `content`, thought `summary`, tool `result`. Media is
either inline base64 `data` or a `uri`.

| `type` | Fields |
|---|---|
| `text` | `text: string` (required), `annotations?: Annotation[]`. |
| `image` | `data?: string (base64)`, `uri?: string`, `mime_type?` (`image/png`, `image/jpeg`, `image/webp`, `image/heic`, `image/heif`, `image/gif`, `image/bmp`, `image/tiff`), `resolution?: MediaResolution`. |
| `audio` | `data?`, `uri?`, `mime_type?` (`audio/wav`, `audio/mp3`, `audio/aiff`, `audio/aac`, `audio/ogg`, `audio/flac`, `audio/mpeg`, `audio/m4a`, `audio/l16`, `audio/opus`, `audio/alaw`, `audio/mulaw`), `channels?: int`, `sample_rate?: int`. |
| `video` | `data?`, `uri?`, `mime_type?` (`video/mp4`, `video/mpeg`, `video/mpg`, `video/mov`, `video/avi`, `video/x-flv`, `video/webm`, `video/wmv`, `video/3gpp`), `resolution?: MediaResolution`. |
| `document` | `data?`, `uri?`, `mime_type?` (`application/pdf`, `text/csv`). |

`MediaResolution`: `low` | `medium` | `high` | `ultra_high`.

### Annotation (union, discriminator `type`)

Citations attached to `TextContent.annotations`. All variants have `start_index?` /
`end_index?` (byte offsets into the text, end exclusive).

| `type` | Extra fields |
|---|---|
| `url_citation` | `url?`, `title?`. |
| `file_citation` | `file_name?`, `document_uri?`, `page_number?`, `source?`, `media_id?`, `custom_metadata?: object`. |
| `place_citation` | `name?`, `place_id?` (`places/{place_id}`), `url?`, `review_snippets?: [{review_id?, title?, url?}]`. |
| `word_info` | Word-level ASR: `text?`, `start_offset?`, `end_offset?` (durations, with `timestamp_granularities: ["word"]`), `speaker?` (with diarization). |

```json
{ "type": "text", "text": "Hello, how are you?" }
{ "type": "image", "data": "BASE64_ENCODED_IMAGE", "mime_type": "image/png" }
{ "type": "video", "uri": "https://www.youtube.com/watch?v=..." }
```

## 7. Tool (union, discriminator `type`)

| `type` | Fields |
|---|---|
| `function` | `name?: string`, `description?: string`, `parameters?: object (JSON Schema)`. |
| `google_search` | `search_types?: ("web_search"\|"image_search"\|"enterprise_web_search")[]`. |
| `google_maps` | `enable_widget?: bool`, `latitude?: number`, `longitude?: number`. |
| `code_execution` | (no fields). |
| `url_context` | (no fields). |
| `file_search` | `file_search_store_names?: string[]`, `metadata_filter?: string`, `top_k?: int`. |
| `computer_use` | `environment?: "browser"\|"mobile"\|"desktop"`, `enable_prompt_injection_detection?: bool`, `excluded_predefined_functions?: string[]`, `disabled_safety_policies?: enum[]` (`financial_transactions`, `sensitive_data_modification`, `communication_tool`, `account_creation`, `data_modification`, `user_consent_management`, `legal_terms_and_agreements`). |
| `mcp_server` | `name?: string`, `url?: string`, `headers?: object`, `allowed_tools?: [{mode?: "auto"\|"any"\|"none"\|"validated", tools?: string[]}]`. |
| `retrieval` | `retrieval_types?: ("rag_store"\|"exa_ai_search"\|"parallel_ai_search")[]`, `exa_ai_search_config?: {api_key: string, custom_config?: object}`, `parallel_ai_search_config?: {api_key?, custom_config?}`, `rag_store_config?: {rag_resources?: [{rag_corpus?, rag_file_ids?: string[]}], rag_retrieval_config?: {filter?: {metadata_filter?, vector_distance_threshold?, vector_similarity_threshold?}, hybrid_search?: {alpha?}, ranking?, top_k?}}`. |

## 8. ResponseFormat (union, discriminator `type`)

Single object or array (one per requested modality). `delivery`: `inline` | `uri`.

| `type` | Fields |
|---|---|
| `text` | `mime_type?: "application/json"\|"text/plain"`, `schema?: object` (JSON Schema; only with `application/json`). |
| `image` | `aspect_ratio?` (`1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9`, `1:8`, `8:1`, `1:4`, `4:1`), `image_size?: "512"\|"1K"\|"2K"\|"4K"`, `mime_type?: "image/jpeg"`, `delivery?`. |
| `audio` | `mime_type?` (`audio/mp3`, `audio/ogg_opus`, `audio/l16`, `audio/wav`, `audio/alaw`, `audio/mulaw`), `sample_rate?: int (Hz)`, `bit_rate?: int (bps, compressed formats)`, `delivery?`. |
| `video` | `aspect_ratio?: "16:9"\|"9:16"`, `duration?: string`, `gcs_uri?: string` (required on Vertex with `delivery: "uri"`), `delivery?`. |

```json
{ "type": "text", "mime_type": "application/json",
  "schema": { "type": "object", "properties": { "recipe_name": { "type": "string" } },
              "required": ["recipe_name"] } }
```

## 9. EnvironmentConfig (agent sandboxes)

`environment` accepts a string (existing environment ID) or an object:

| Field | Type | Description |
|---|---|---|
| `type` | `"remote"` | Discriminator. |
| `environment_id` | string | Update an existing environment instead of creating one. |
| `sources` | Source[] | Files/repos mounted into the environment. |
| `network` | object \| `"disabled"` | Egress allowlist, or disable all networking. Omit to allow all. |

`Source`: `{type?: "gcs"|"inline"|"repository"|"skill_registry", source?: string (path/URL),
content?: string (inline), encoding?: string (e.g. "base64"), target?: string (mount path)}`.

Network allowlist: `{"allowlist": [{domain?: string ("*.github.com", "*" allowed),
transform?: object|object[] (headers injected on matching requests)}]}`.

```json
{ "type": "remote",
  "sources": [{ "type": "repository", "source": "https://github.com/my-org/my-skills.git",
                "target": ".agents/skills" }],
  "network": { "allowlist": [{ "domain": "pypi.org" }] } }
```

## 10. Streaming (SSE)

Set `stream: true` on create (same endpoint; no separate `:stream` method). The response is
a server-sent event stream; each event has an SSE `event:` name and `data:` JSON whose
`event_type` field repeats the name. The stream terminates with `event: done` /
`data: [DONE]`. New event/delta types may be added — skip unknown ones gracefully.

Event flow: `interaction.created` → (`interaction.status_update`) → repeated step cycles
(`step.start` → `step.delta`* → `step.stop`) → `interaction.completed`. With `stream: false`
each cycle is instead returned fully assembled as one element of `steps`.

Every event may carry `event_id` (string): pass it as `last_event_id` to
`GET /v1beta/interactions/{id}?stream=true` to resume the stream after that event.

### Event catalog (union, discriminator `event_type`)

| `event_type` | Payload fields | Semantics |
|---|---|---|
| `interaction.created` | `interaction` (partial Interaction, required), `event_id?` | Stream opened; carries id, model/agent, initial `status`. |
| `interaction.status_update` | `interaction_id: string`, `status: enum` (required), `event_id?` | Interaction-level status transition; may appear between steps. |
| `step.start` | `index: int`, `step: Step` (required), `event_id?` | New step begins; `step.type` decides which deltas follow. For `function_call` includes `id`, `name` and empty `arguments: {}`. |
| `step.delta` | `index: int`, `delta: StepDeltaData` (required), `metadata?: {total_usage?: Usage}`, `event_id?` | Incremental data for the step at `index`. |
| `step.stop` | `index: int`, `usage?: Usage` (running total; Antigravity), `step_usage?: Usage` (this step; Antigravity), `event_id?` | Step complete. |
| `interaction.completed` | `interaction` (partial Interaction with final `usage`, no `steps`), `event_id?` | Terminal event before `done`. |
| `error` | `error: {code?: string, message?: string}`, `event_id?` | Error during the interaction (e.g. `"gateway_timeout"`). |

The partial interaction object in lifecycle events (`InteractionSseEventInteraction`) has:
`id`, `object`, `model`/`agent`, `status`, `service_tier`, `created`, `updated`, `usage?`,
`steps?` — streaming payloads may omit fields present on full responses.

### Step types → expected delta types

| Step type | Delta types |
|---|---|
| `model_output` | `text`, `image`, `audio` (also `video`, `document`). |
| `thought` | `thought_summary` (only when `thinking_summaries` enabled), then `thought_signature` as the final delta. |
| `function_call` | `arguments_delta` (accumulate JSON string fragments). |
| server-side tools | The matching `*_call` / `*_result` delta type. |

### StepDeltaData (union, discriminator `type`)

| `type` | Fields | Notes |
|---|---|---|
| `text` | `text: string` | Append to current text. |
| `image` | `data?`, `mime_type?`, `resolution?`, `uri?` | Same enums as ImageContent. |
| `audio` | `data?`, `mime_type?`, `channels?`, `sample_rate?`, `uri?` | Same enums as AudioContent. |
| `video` | `data?`, `mime_type?`, `resolution?`, `uri?` | Same enums as VideoContent. |
| `document` | `data?`, `mime_type?`, `uri?` | Same enums as DocumentContent. |
| `thought_summary` | `content?: Content` | A summary item (text or image) to append to the thought. |
| `thought_signature` | `signature?: string` | Encrypted reasoning signature; last delta of a `thought` step. |
| `arguments_delta` | `arguments?: string` | Partial JSON string of function-call arguments; concatenate across deltas. |
| `text_annotation_delta` | `annotations?: Annotation[]` | Citations for previously streamed text. |
| `function_result` | `call_id: string`, `result` (required), `name?`, `is_error?` | Mirrors FunctionResultStep. |
| `code_execution_call` | `arguments: {code?, language?}`, `signature?` | |
| `code_execution_result` | `result: string`, `is_error?`, `signature?` | |
| `google_search_call` | `arguments: {queries?: string[]}`, `signature?` | |
| `google_search_result` | `result: GoogleSearchResultItem[]`, `is_error?`, `signature?` | |
| `google_maps_call` | `arguments?: {queries?: string[]}`, `signature?` | |
| `google_maps_result` | `result?: GoogleMapsResultItem[]`, `signature?` | |
| `file_search_call` | `signature?` | |
| `file_search_result` | `result: FileSearchResult[]`, `signature?` | |
| `mcp_server_tool_call` | `arguments: object`, `name: string`, `server_name: string` | |
| `mcp_server_tool_result` | `result` (required), `name?`, `server_name?` | |
| `retrieval_call` | `arguments: {queries?: string[], retrieval_type?: "rag_store"\|"exa_ai_search"\|"parallel_ai_search"}`, `signature?` | Vertex retrieval tools; delta-only (no corresponding Step type documented). |
| `retrieval_result` | `is_error?`, `signature?` | Delta-only. |
| `url_context_call` | `arguments: {urls?: string[]}`, `signature?` | |
| `url_context_result` | `result: UrlContextResult[]`, `is_error?`, `signature?` | |

### Example stream (search + function call)

```
event: interaction.created
data: {"interaction":{"id":"v1_...","status":"in_progress","object":"interaction","model":"gemini-3.6-flash"},"event_type":"interaction.created"}

event: interaction.status_update
data: {"interaction_id":"v1_...","status":"in_progress","event_type":"interaction.status_update"}

event: step.start
data: {"index":0,"step":{"id":"mkutnkgn","signature":"","type":"google_search_call"},"event_type":"step.start"}

event: step.delta
data: {"index":0,"delta":{"signature":"...","type":"google_search_call","arguments":{"queries":["largest mountain in Europe"]}},"event_type":"step.delta"}

event: step.stop
data: {"index":0,"event_type":"step.stop"}

event: step.start
data: {"index":1,"step":{"call_id":"mkutnkgn","signature":"","type":"google_search_result"},"event_type":"step.start"}

event: step.delta
data: {"index":1,"delta":{"signature":"...","type":"google_search_result","is_error":false},"event_type":"step.delta"}

event: step.stop
data: {"index":1,"event_type":"step.stop"}

event: step.start
data: {"index":2,"step":{"type":"thought"},"event_type":"step.start"}

event: step.delta
data: {"index":2,"delta":{"signature":"...","type":"thought_signature"},"event_type":"step.delta"}

event: step.stop
data: {"index":2,"event_type":"step.stop"}

event: step.start
data: {"index":3,"step":{"id":"ktr5aysg","type":"function_call","name":"get_weather","arguments":{}},"event_type":"step.start"}

event: step.delta
data: {"index":3,"delta":{"arguments":"{\"location\":\"Mount Elbrus, Russia\"}","type":"arguments_delta"},"event_type":"step.delta"}

event: step.stop
data: {"index":3,"event_type":"step.stop"}

event: interaction.completed
data: {"interaction":{"id":"v1_...","status":"requires_action","usage":{"total_tokens":299,"total_input_tokens":138,"input_tokens_by_modality":[{"modality":"text","tokens":138}],"total_cached_tokens":0,"total_output_tokens":20,"total_tool_use_tokens":0,"total_thought_tokens":141},"created":"2026-05-12T17:24:26Z","updated":"2026-05-12T17:24:26Z","service_tier":"standard","object":"interaction","model":"gemini-3.6-flash"},"event_type":"interaction.completed"}

event: done
data: [DONE]
```

Streaming function-calling flow: turn 1 streams the `function_call` step (accumulate
`arguments_delta`) and completes with `status: "requires_action"`; turn 2 creates a new
interaction with `previous_interaction_id` and a `function_result` input item, which resumes
generation. Agents (e.g. Deep Research) combine `stream: true` with `background: true` to
stream progress of a background run.

## 11. Known gaps in the source docs

- `Turn` (input union member) is not defined in the v1beta reference; shape
  `{role, content}` comes from the deprecated internal schema.
- `FileSearchResult` item fields are not documented.
- `GoogleSearchResultItem` documents only `search_suggestions`, while examples show
  `title`/`url`/`snippet`.
- Per-step `status` (`"done"`, `"waiting"`) appears in migration-guide examples but not in
  the reference schema.
- The reference labels itself v1beta; a stable v1 exists and some official examples use
  `/v1beta2/`. Verify the version path against the live OpenAPI spec
  (`https://ai.google.dev/static/api/interactions.openapi.json`) when implementing.
