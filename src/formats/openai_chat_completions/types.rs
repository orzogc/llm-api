//! Serde wire types for the OpenAI Chat Completions API.
//!
//! Every object type carries a `#[serde(flatten)]` map so unknown fields are
//! captured on parse and mirrored into the IR `extra` namespace (design § 5,
//! implementation contract "Wire types"). Only the modeled subset is
//! declared; polymorphic values (message content, content parts, tool
//! definitions, tool-call entries, `response_format`, `tool_choice`) are
//! kept as raw [`Value`]s and dispatched manually so unmodeled shapes
//! survive verbatim.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// The `POST /v1/chat/completions` request body (modeled subset).
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Request {
    /// Model id. Not part of the IR (supplied via configuration); consumed
    /// and dropped when parsing a request into the IR.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The conversation, kept raw for manual dispatch on `role`.
    #[serde(default)]
    pub messages: Vec<Value>,
    /// Upper bound on generated tokens, including reasoning tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    /// Deprecated alias of `max_completion_tokens`; parses into the same IR
    /// field and canonicalizes to `max_completion_tokens` on serialization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Sampling temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Nucleus sampling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Stop sequences: a single string or an array of strings (the string
    /// shorthand canonicalizes to a one-element array).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<Value>,
    /// Best-effort determinism seed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Frequency penalty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    /// Presence penalty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    /// Request metadata (up to 16 string pairs upstream).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Map<String, Value>>,
    /// Prompt-cache bucketing key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    /// Reasoning effort tier (`none` … `max`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// Structured-output configuration, kept raw for manual dispatch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    /// Tool definitions. `function` tools are interpreted; everything else
    /// (`custom` tools, dialect kinds) round-trips as `Tool::Opaque`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,
    /// Tool-choice constraint: `"none"` / `"auto"` / `"required"` or an
    /// object form. Only the string modes and `{type: "function",
    /// function: {name}}` are interpreted; other shapes round-trip through
    /// the request `extra`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    /// Whether the model may call tools in parallel.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// SSE streaming flag. Configuration, not IR data; consumed on parse.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Streaming options (`include_usage`, `include_obfuscation`).
    /// Streaming configuration tied to `stream`; consumed on parse — the
    /// build side re-injects `include_usage` per
    /// [`crate::OpenAiChatCompletionsOptions`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<Value>,
    /// Unknown fields (`n`, `logit_bias`, `service_tier`, dialect knobs,
    /// …), preserved verbatim.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// One wire message (modeled subset, lenient: every field except `role` is
/// an untyped [`Value`] so a malformed field degrades to the extra mirror
/// instead of failing the whole message).
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// `system` / `developer` / `user` / `assistant` / `tool`, the legacy
    /// `function`, or a dialect role. Missing or non-string values fail the
    /// typed parse and keep the whole message verbatim as `Opaque`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// String shorthand, a content-part array, or `null`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>,
    /// DeepSeek-dialect plaintext chain of thought (assistant messages).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<Value>,
    /// Assistant refusal text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal: Option<Value>,
    /// Assistant tool calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,
    /// Tool messages: id of the call this message answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<Value>,
    /// Participant name (system/developer/user/assistant) or the tool name
    /// on `tool` messages (dialect usage; maps to `ToolResult.name`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<Value>,
    /// Unknown fields (`audio`, `function_call`, …), preserved verbatim.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// A `text` content part.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TextPart {
    /// Part type, always `text`.
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub part_type: Option<String>,
    /// The text.
    pub text: String,
    /// Explicit cache breakpoint (`{mode: "explicit"}`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_breakpoint: Option<Value>,
    /// Unknown fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// An `image_url` content part.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImagePart {
    /// Part type, always `image_url`.
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub part_type: Option<String>,
    /// The image reference.
    pub image_url: ImageUrl,
    /// Explicit cache breakpoint (`{mode: "explicit"}`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_breakpoint: Option<Value>,
    /// Unknown fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The `image_url` object of an image part.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ImageUrl {
    /// Remote URL or base64 `data:` URL.
    pub url: String,
    /// Unknown fields (`detail`, …), mirrored into the block extra at
    /// `image_url.<key>`.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// A `refusal` content part of an assistant message.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RefusalPart {
    /// Part type, always `refusal`.
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub part_type: Option<String>,
    /// The refusal explanation.
    pub refusal: String,
    /// Unknown fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// One `tool_calls[]` entry (request or response side).
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolCallEntry {
    /// Call id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// `function` or `custom` (other kinds round-trip via the mirrored
    /// extra).
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    /// The `function` payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallPayload>,
    /// The `custom` payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom: Option<CustomCallPayload>,
    /// Unknown fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The `function` object of a tool call.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FunctionCallPayload {
    /// Function name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Raw JSON argument string (may be invalid JSON from a model).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
    /// Unknown fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The `custom` object of a custom tool call.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CustomCallPayload {
    /// Custom tool name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Raw input string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
    /// Unknown fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// A `function` tool definition (`tools[]` entry with `type: "function"`).
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FunctionToolDef {
    /// Tool type, always `function`.
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub tool_type: Option<String>,
    /// The nested function definition.
    pub function: FunctionDef,
    /// Unknown fields at the tool level.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The nested `function` object of a function tool definition.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FunctionDef {
    /// Function name.
    pub name: String,
    /// Human-readable description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema of the arguments; omitted means "no parameters".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
    /// Strict schema adherence flag.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    /// Unknown fields, mirrored into the tool extra at `function.<key>`.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The `chat.completion` response object (modeled subset).
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Response {
    /// Completion id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Model actually used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Choices; only the first is read (the rest remain in `raw`).
    #[serde(default)]
    pub choices: Vec<Value>,
    /// Token usage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Unknown fields (`created`, `object`, `system_fingerprint`,
    /// `service_tier`, …); terminal envelope data, available via `raw`.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// One `choices[]` entry.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Choice {
    /// Choice index.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<u64>,
    /// The assistant message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<Value>,
    /// `stop` / `length` / `tool_calls` / `content_filter` /
    /// `function_call` (deprecated).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// Unknown fields (`logprobs`, …).
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The response `usage` object. `prompt_tokens` already includes cached
/// tokens (design § 8).
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    /// All prompt tokens, including cache reads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    /// All generated tokens, including reasoning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
    /// Prompt + completion tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    /// Prompt token details.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    /// Completion token details.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    /// Unknown fields (DeepSeek `prompt_cache_hit_tokens`, …).
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The `usage.prompt_tokens_details` object.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    /// Tokens read from the prompt cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u64>,
    /// Tokens written to the prompt cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u64>,
    /// Unknown fields (`audio_tokens`, …).
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The `usage.completion_tokens_details` object.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CompletionTokensDetails {
    /// Reasoning tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
    /// Unknown fields (`audio_tokens`, prediction token counts, …).
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The `GET /v1/models` response envelope.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelList {
    /// Always `"list"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object: Option<String>,
    /// Model entries, kept raw (only `id` and `created` are read).
    #[serde(default)]
    pub data: Vec<Value>,
    /// Unknown fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The OpenAI error body shape (`{"error": {...}}`).
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ErrorBody {
    /// The error object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorDetail>,
    /// Unknown fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The OpenAI error object.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ErrorDetail {
    /// Human-readable message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Error type (`invalid_request_error`, `insufficient_quota`, …).
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub error_type: Option<String>,
    /// Machine-readable code (`invalid_api_key`, `model_not_found`, …).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<Value>,
    /// Offending parameter, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param: Option<Value>,
    /// Unknown fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
