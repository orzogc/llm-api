//! Serde wire types for the OpenAI Responses API.
//!
//! Every object type carries a `#[serde(flatten)]` map so unknown fields are
//! captured on parse and mirrored into the IR `extra` namespace (design § 5,
//! implementation contract "Wire types"). Only the modeled subset is
//! declared; everything else flows through the flatten maps or is kept as
//! raw [`Value`]s (input/output items are dispatched manually on their
//! `type` field so unmodeled items survive verbatim as `Opaque` nodes).

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// The `POST /v1/responses` request body (modeled subset).
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Request {
    /// Model id. Not part of the IR (supplied via configuration); consumed
    /// and dropped when parsing a request into the IR.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Text, image, or file inputs: a plain string (shorthand for a single
    /// `user` text message, canonicalized to an item array on parse) or an
    /// array of input items.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<RequestInput>,
    /// System/developer instructions. A string maps to `Request.system`;
    /// any other non-null shape is kept in the request `extra` namespace.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<Value>,
    /// Upper bound on generated tokens (visible output + reasoning).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// Sampling temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Nucleus sampling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Request metadata (up to 16 string pairs upstream).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Map<String, Value>>,
    /// Prompt-cache bucketing key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    /// Reasoning configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
    /// Output text configuration (structured outputs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<TextConfig>,
    /// Tool definitions. Kept as raw values; `function` tools are
    /// interpreted, everything else round-trips as `Tool::Opaque`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,
    /// Tool-choice constraint: `"none"` / `"auto"` / `"required"` or an
    /// object form. Only the string modes and `{type:"function", name}` are
    /// interpreted; other shapes round-trip through the request `extra`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    /// Whether the model may run tool calls in parallel.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// SSE streaming flag. Not part of the IR (the call mode decides it);
    /// consumed and dropped when parsing a request into the IR.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Unknown fields, preserved verbatim.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The polymorphic `input` field: string shorthand or an item array.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum RequestInput {
    /// Shorthand for a single `user` text message.
    Text(String),
    /// A list of input items, kept raw for manual dispatch on `type`.
    Items(Vec<Value>),
}

/// The `reasoning` request object.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ReasoningConfig {
    /// Reasoning effort tier (`none` … `max`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    /// Reasoning summary verbosity (`auto` / `concise` / `detailed`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    /// Unknown fields (`context`, `mode`, …), preserved verbatim.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The `text` request object.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TextConfig {
    /// Output format (`text` / `json_schema` / `json_object`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<Value>,
    /// Unknown fields (`verbosity`, …), preserved verbatim.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// A `function` tool definition. `parameters` and `strict` are
/// required-but-nullable upstream, so they always serialize (as `null` when
/// unset).
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FunctionToolDef {
    /// Tool type, always `function`.
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub tool_type: Option<String>,
    /// Function name.
    pub name: String,
    /// Human-readable description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema of the arguments; `null` means "no parameters".
    #[serde(default)]
    pub parameters: Option<Value>,
    /// Strict schema validation flag.
    #[serde(default)]
    pub strict: Option<bool>,
    /// Unknown fields (`output_schema`, `defer_loading`, …).
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// A `message` input item (`type` may be omitted on the wire when `role` is
/// present).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MessageItem {
    /// Item type, always `message` when present.
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub item_type: Option<String>,
    /// `user` / `system` / `developer` / `assistant`.
    pub role: String,
    /// String shorthand or an array of content parts (kept raw).
    pub content: MessageContent,
    /// Unknown fields (`id`, `status`, `phase`, …), preserved verbatim.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Message content: string shorthand or content parts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum MessageContent {
    /// A single text input.
    Text(String),
    /// Content parts, kept raw for manual dispatch on `type`.
    Parts(Vec<Value>),
}

/// A `function_call` item.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FunctionCallItem {
    /// Item type, always `function_call`.
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub item_type: Option<String>,
    /// Links the call to its `function_call_output`.
    pub call_id: String,
    /// Function name.
    pub name: String,
    /// Raw JSON argument string.
    pub arguments: String,
    /// Unknown fields (`id`, `status`, `namespace`, `caller`, …).
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// A `function_call_output` item.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FunctionCallOutputItem {
    /// Item type, always `function_call_output`.
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub item_type: Option<String>,
    /// Must match the `function_call.call_id`.
    pub call_id: String,
    /// A JSON string, or an array of `input_text` / `input_image` /
    /// `input_file` parts (kept raw).
    pub output: FunctionCallOutput,
    /// Name of the tool that produced the output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Unknown fields (`id`, `status`, `caller`, …).
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The polymorphic `function_call_output.output` field.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum FunctionCallOutput {
    /// String output (the empty string encodes an empty result).
    Text(String),
    /// An array of content parts, kept raw.
    Parts(Vec<Value>),
}

/// A `reasoning` item.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ReasoningItem {
    /// Item type, always `reasoning`.
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub item_type: Option<String>,
    /// Item id (`rs_…`). Required upstream; optional here so signature-only
    /// replay blocks without a stored id still serialize.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Reasoning summary: `{type:"summary_text", text}` parts (kept raw).
    #[serde(default)]
    pub summary: Vec<Value>,
    /// Encrypted reasoning payload for stateless replay.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encrypted_content: Option<String>,
    /// Unknown fields (`content`, `status`, …).
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// An `input_text` content part.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct InputTextPart {
    /// Part type, always `input_text`.
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub part_type: Option<String>,
    /// The text.
    pub text: String,
    /// Explicit cache breakpoint (`{mode:"explicit"}`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_breakpoint: Option<Value>,
    /// Unknown fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// An `input_image` content part.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct InputImagePart {
    /// Part type, always `input_image`.
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub part_type: Option<String>,
    /// Fully qualified URL or base64 `data:` URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
    /// Provider file id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    /// Explicit cache breakpoint (`{mode:"explicit"}`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_breakpoint: Option<Value>,
    /// Unknown fields (`detail`, …).
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// An `output_text` content part of an assistant message.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OutputTextPart {
    /// Part type, always `output_text`.
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub part_type: Option<String>,
    /// The text.
    pub text: String,
    /// Annotations (citations etc.), kept raw.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Vec<Value>>,
    /// Unknown fields (`logprobs`, …).
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

/// The Response object returned by `POST /v1/responses` (modeled subset).
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Response {
    /// Response id (`resp_…`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Model actually used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// `completed` / `failed` / `in_progress` / `cancelled` / `queued` /
    /// `incomplete`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    /// Populated when generation failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<ResponseError>,
    /// Populated when `status` is `incomplete`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<IncompleteDetails>,
    /// Output items, kept raw for manual dispatch on `type`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output: Vec<Value>,
    /// Token usage. The response parser splits this field off and parses
    /// it in a second, lenient stage so a malformed usage object degrades
    /// to `None` (with a warning) instead of failing the 2xx response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Unknown fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The `error` object of a failed Response.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ResponseError {
    /// Error code (`server_error`, `rate_limit_exceeded`, …).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// Human-readable message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Unknown fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The `incomplete_details` object.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct IncompleteDetails {
    /// `max_output_tokens` or `content_filter`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Unknown fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The Response `usage` object. `input_tokens` already includes cached
/// tokens (design § 8).
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    /// All input tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    /// All output tokens, including reasoning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    /// Sum of input and output tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    /// Input token details.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<InputTokensDetails>,
    /// Output token details.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<OutputTokensDetails>,
    /// Unknown fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The `usage.input_tokens_details` object.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct InputTokensDetails {
    /// Tokens read from the prompt cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u64>,
    /// Tokens written to the prompt cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u64>,
    /// Unknown fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The `usage.output_tokens_details` object.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OutputTokensDetails {
    /// Reasoning tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
    /// Unknown fields.
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

/// The `POST /v1/responses/input_tokens` response.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct InputTokenCount {
    /// Always `"response.input_tokens"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object: Option<String>,
    /// Input tokens the request would consume.
    pub input_tokens: u64,
    /// Unknown fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The OpenAI error body shape (`{"error": {...}}`), shared by non-2xx
/// responses.
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
    /// Error type (`invalid_request_error`, …).
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub error_type: Option<String>,
    /// Machine-readable code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// Offending parameter, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    /// Unknown fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
