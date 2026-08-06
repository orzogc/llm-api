# Gemini API: Models & `countTokens`

Reference for the Google Gemini API (Generative Language API, `v1beta`) model
metadata endpoints and token counting. Compiled from the official
[models API reference](https://ai.google.dev/api/models.md.txt) and
[tokens API reference](https://ai.google.dev/api/tokens.md.txt).

Conventions follow [`google-generate-content.md`](./google-generate-content.md):
all JSON field names are camelCase (snake_case aliases accepted in requests).
proto3 JSON omits unset/default-valued fields, so a client should treat every
response field as optional when deserializing (even fields the reference marks
"Required"). "Req" column: R = required, O = optional, Out = output only.

## Endpoints

| Method | HTTP |
|---|---|
| `models.list` | `GET https://generativelanguage.googleapis.com/v1beta/models` |
| `models.get` | `GET https://generativelanguage.googleapis.com/v1beta/{name=models/*}` |
| `models.countTokens` | `POST https://generativelanguage.googleapis.com/v1beta/{model=models/*}:countTokens` |
| `models.predict` | `POST https://generativelanguage.googleapis.com/v1beta/{model=models/*}:predict` |
| `models.predictLongRunning` | `POST https://generativelanguage.googleapis.com/v1beta/{model=models/*}:predictLongRunning` |

Authentication: API key, either as the `x-goog-api-key: $GEMINI_API_KEY`
header or the `?key=$GEMINI_API_KEY` query parameter. The GET methods require
an empty request body; the POST methods take JSON
(`Content-Type: application/json`).

## Resource: `Model`

Metadata about a generative language model. Read-only (returned by
`models.get` / `models.list`).

| Field | Type | Req | Description |
|---|---|---|---|
| `name` | string | R | Resource name, format `models/{model}` where `{model}` is `{baseModelId}-{version}`, e.g. `models/gemini-1.5-flash-001`. |
| `baseModelId` | string | R | Base model name to pass in generation requests, e.g. `gemini-1.5-flash`. (Often omitted in practice.) |
| `version` | string | R | Version number of the model (major version, e.g. `1.0`, `1.5`). |
| `displayName` | string | O | Human-readable name, e.g. `"Gemini 1.5 Flash"`. Up to 128 UTF-8 characters. |
| `description` | string | O | Short description of the model. |
| `inputTokenLimit` | integer | O | Maximum number of input tokens allowed. |
| `outputTokenLimit` | integer | O | Maximum number of output tokens available. |
| `supportedGenerationMethods[]` | string | O | Supported API method names as Pascal-case-style strings, e.g. `generateContent`, `generateMessage`, `countTokens`, `embedContent`. |
| `thinking` | boolean | O | Whether the model supports thinking. |
| `temperature` | number | O | Default temperature used by the backend; range `[0.0, maxTemperature]`. |
| `maxTemperature` | number | O | Maximum temperature this model can use. |
| `topP` | number | O | Default nucleus-sampling value used by the backend. |
| `topK` | integer | O | Default top-k value. If absent, the model doesn't use top-k sampling and `topK` isn't allowed as a generation parameter. |

```json
{
  "name": "models/gemini-2.0-flash",
  "version": "2.0",
  "displayName": "Gemini 2.0 Flash",
  "description": "Fast multimodal model",
  "inputTokenLimit": 1048576,
  "outputTokenLimit": 8192,
  "supportedGenerationMethods": ["generateContent", "countTokens"],
  "temperature": 1,
  "maxTemperature": 2,
  "topP": 0.95,
  "topK": 40
}
```

## Method: `models.list`

Lists the models available through the Gemini API. Request body must be empty.

Query parameters:

| Param | Type | Req | Description |
|---|---|---|---|
| `pageSize` | integer | O | Max models per page. Default 50; capped at 1000 even if a larger value is passed. |
| `pageToken` | string | O | `nextPageToken` from a previous `models.list` call. All other parameters must match the call that produced the token. |

Response body (`ListModelsResponse`):

| Field | Type | Req | Description |
|---|---|---|---|
| `models[]` | `Model` | Out | The returned models. |
| `nextPageToken` | string | Out | Send as `pageToken` to get the next page; omitted when there are no more pages. |

```sh
curl "https://generativelanguage.googleapis.com/v1beta/models?key=$GEMINI_API_KEY"
```

```json
{"models": [{"name": "models/gemini-2.0-flash", "...": "..."}], "nextPageToken": "Chxtb2..."}
```

## Method: `models.get`

Gets metadata for one model. Path parameter: `name` (string, required) —
model resource name, format `models/{model}`; must match a name returned by
`models.list`. Request body must be empty. Response body: a `Model`.

```sh
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash?key=$GEMINI_API_KEY"
```

## Methods: `models.predict` / `models.predictLongRunning`

Generic prediction methods (path parameter `model`, format `models/{model}`).
Request body fields (both): `instances[]` (`google.protobuf.Value`, required)
— inputs of the prediction call; `parameters` (`Value`, optional) — parameters
governing the call. `predictLongRunning` additionally accepts
`webhookConfig.uris[]` (string[], optional) — webhook URIs used for webhook
events instead of the registered webhooks.

Response: `models.predict` returns `{"predictions": [value]}`;
`models.predictLongRunning` returns a long-running `Operation`.

## Method: `models.countTokens`

Runs the model's tokenizer on input content and returns the token count. No
generation happens. Path parameter: `model` (string, required), format
`models/{model}`.

### Request body

| Field | Type | Req | Description |
|---|---|---|---|
| `contents[]` | `Content` | O | The prompt input. Ignored when `generateContentRequest` is set. |
| `generateContentRequest` | `GenerateContentRequest` | O | The overall input to the model, including system instructions, function declarations, etc. |

The two variants are mutually exclusive: send URL model + `contents`, or a
`generateContentRequest`, never both.

- `Content` / `Part` and `GenerateContentRequest` (contents, tools,
  toolConfig, safetySettings, systemInstruction, generationConfig,
  cachedContent, ...) are documented in
  [`google-generate-content.md`](./google-generate-content.md).
- Unlike the top-level `generateContent` body (where the model is only in the
  URL path), the standalone `GenerateContentRequest` message nested here
  contains a required `model` field (format `models/{model}`); set it to the
  same model as the URL.
- Use the `generateContentRequest` variant to include `systemInstruction`,
  `tools`, or `cachedContent` in the count — plain `contents` cannot carry
  them.

`contents` variant:

```json
{
  "contents": [
    {"role": "user", "parts": [
      {"text": "Tell me about this instrument"},
      {"inlineData": {"mimeType": "image/jpeg", "data": "<base64>"}}
    ]}
  ]
}
```

`generateContentRequest` variant:

```json
{
  "generateContentRequest": {
    "model": "models/gemini-2.0-flash",
    "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
    "systemInstruction": {"parts": [{"text": "You are a cat named Neko."}]},
    "tools": [{"functionDeclarations": [{"name": "multiply", "description": "returns a * b."}]}]
  }
}
```

### Response body (`CountTokensResponse`)

| Field | Type | Req | Description |
|---|---|---|---|
| `totalTokens` | integer | Out | Number of tokens the model tokenizes the prompt into. Always non-negative. |
| `cachedContentTokenCount` | integer | Out | Tokens in the cached part of the prompt (the cached content). |
| `promptTokensDetails[]` | `ModalityTokenCount` | Out | Per-modality breakdown of the request input. |
| `cacheTokensDetails[]` | `ModalityTokenCount` | Out | Per-modality breakdown of the cached content. |

`ModalityTokenCount` = `{"modality": enum, "tokenCount": integer}`, modality
one of `TEXT`, `IMAGE`, `VIDEO`, `AUDIO`, `DOCUMENT` (or
`MODALITY_UNSPECIFIED`) — same type as in `usageMetadata` (see
[`google-generate-content.md`](./google-generate-content.md)).

```json
{
  "totalTokens": 268,
  "promptTokensDetails": [
    {"modality": "TEXT", "tokenCount": 10},
    {"modality": "IMAGE", "tokenCount": 258}
  ]
}
```

### Notes

- Multimodal input is supported: text, inline media (`inlineData`), and Files
  API references (`fileData`), including images, video, audio and PDFs. An
  image's display or file size does not affect its token count. See the
  official [token counting guide](https://ai.google.dev/gemini-api/docs/tokens)
  for per-media counting rules.
- The official examples show `generateContent`'s
  `usageMetadata.promptTokenCount` coming out one token higher than
  `countTokens.totalTokens` for the same input (e.g. 10 vs 11); don't expect
  exact equality.
- Cached content is not counted when you pass only the new prompt in
  `contents`; `cachedContentTokenCount` / `cacheTokensDetails` cover the
  cached part of the prompt when the counted request references it.

## Sources

- [models API reference](https://ai.google.dev/api/models.md.txt) —
  `models.get`, `models.list`, `Model` resource, `models.predict`,
  `models.predictLongRunning`.
- [tokens API reference](https://ai.google.dev/api/tokens.md.txt) —
  `models.countTokens` request/response and examples.
