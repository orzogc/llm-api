//! Complete serde wire types for the Anthropic Messages API.
//!
//! Every object type carries a flattened `extra` map that captures unknown
//! fields on parse, so non-null unknown fields round-trip verbatim (design
//! § 1). Content blocks are dispatched manually on their `type` field by the
//! conversion code — unmodeled block kinds are preserved as raw JSON
//! ([`crate::ir::ContentBlock::Opaque`]) instead of failing.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// The `POST /v1/messages` request body.
///
/// `model` and `max_tokens` are required upstream but optional here so that
/// foreign request bodies parse leniently.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct MessagesRequest {
    /// Model id or alias.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The conversation (entries kept raw: each one parses leniently, so a
    /// malformed message degrades to an opaque passthrough instead of
    /// failing the whole request).
    #[serde(default)]
    pub messages: Vec<Value>,
    /// Maximum tokens to generate (required upstream).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    /// System prompt: string shorthand or an array of text blocks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,
    /// Request metadata (`user_id` plus any dialect keys).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Map<String, Value>>,
    /// Custom stop strings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// SSE streaming toggle.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Sampling temperature (`0.0`–`1.0` upstream).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Top-k sampling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u64>,
    /// Nucleus sampling cutoff.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Tool definitions (custom tools and Anthropic-defined tool types; the
    /// union is kept as raw values and dispatched by the conversion code).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,
    /// Tool-choice constraint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoiceParam>,
    /// Extended-thinking configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
    /// Output configuration (`effort`, structured-output `format`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_config: Option<OutputConfig>,
    /// Unknown fields, preserved for round-trips.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// One `messages[]` entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct MessageParam {
    /// `user`, `assistant` or `system` (mid-conversation system content).
    pub role: String,
    /// String shorthand or an array of content blocks.
    pub content: MessageContent,
    /// Unknown fields, preserved for round-trips.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Message content: string shorthand for one text block, or a block array.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum MessageContent {
    /// Shorthand for `[{"type": "text", "text": …}]`.
    Text(String),
    /// Content block array (blocks dispatched manually on `type`).
    Blocks(Vec<Value>),
}

/// The top-level `system` field: string form or text-block array.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum SystemPrompt {
    /// Plain string form.
    Text(String),
    /// Text-block array form (allows `cache_control` per block).
    Blocks(Vec<Value>),
}

/// `cache_control: {type: "ephemeral", ttl?}` on request blocks and tools.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CacheControl {
    /// Cache type; `ephemeral` is the only documented value.
    #[serde(rename = "type")]
    pub kind: String,
    /// Optional TTL (`"5m"` / `"1h"`), passed through verbatim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
    /// Unknown fields, preserved for round-trips.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl CacheControl {
    /// An `ephemeral` cache control with an optional TTL.
    #[must_use]
    pub fn ephemeral(ttl: Option<String>) -> Self {
        Self {
            kind: "ephemeral".to_owned(),
            ttl,
            extra: Map::new(),
        }
    }
}

/// A `text` content block.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TextBlock {
    /// Discriminator, always `text`.
    #[serde(rename = "type")]
    pub kind: String,
    /// The text.
    pub text: String,
    /// Optional cache breakpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
    /// Unknown fields (including `citations`), preserved for round-trips.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// An `image` content block.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ImageBlock {
    /// Discriminator, always `image`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Image source union.
    pub source: ImageSourceParam,
    /// Optional cache breakpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
    /// Unknown fields, preserved for round-trips.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The `image.source` union.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum ImageSourceParam {
    /// Inline base64 data.
    #[serde(rename = "base64")]
    Base64 {
        /// MIME type (`image/png`, …).
        media_type: String,
        /// Base64-encoded bytes.
        data: String,
    },
    /// A remote URL.
    #[serde(rename = "url")]
    Url {
        /// The URL.
        url: String,
    },
    /// A Files API id.
    #[serde(rename = "file")]
    File {
        /// The file id.
        file_id: String,
    },
}

/// A `tool_use` content block (assistant tool call).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolUseBlock {
    /// Discriminator, always `tool_use`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Provider call id.
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Arguments as a JSON object.
    pub input: Value,
    /// Optional cache breakpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
    /// Unknown fields (e.g. `caller`), preserved for round-trips.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// A `tool_result` content block (user-side tool output).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolResultBlock {
    /// Discriminator, always `tool_result`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Id of the `tool_use` block this result answers.
    pub tool_use_id: String,
    /// Result content: string shorthand or a block array; omitted for an
    /// empty result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<ToolResultContent>,
    /// Marks the result as an error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
    /// Optional cache breakpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
    /// Unknown fields, preserved for round-trips.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// `tool_result.content`: string shorthand or nested block array.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum ToolResultContent {
    /// String shorthand for one text block.
    Text(String),
    /// Nested content blocks (text / image / unmodeled kinds).
    Blocks(Vec<Value>),
}

/// A `thinking` content block.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ThinkingBlock {
    /// Discriminator, always `thinking`.
    #[serde(rename = "type")]
    pub kind: String,
    /// The thinking text.
    pub thinking: String,
    /// Integrity signature required for replay. Optional here so that the
    /// `thinking_as_text` channel (a thinking block without a signature) can
    /// be represented.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    /// Unknown fields, preserved for round-trips.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// A `redacted_thinking` content block.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RedactedThinkingBlock {
    /// Discriminator, always `redacted_thinking`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Opaque payload; echo back unchanged.
    pub data: String,
    /// Unknown fields, preserved for round-trips.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// A custom (function) tool definition.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct FunctionToolParam {
    /// Tool name.
    pub name: String,
    /// Human-readable description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema for the arguments (required upstream).
    pub input_schema: Value,
    /// Strict schema-validation flag.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    /// Optional cache breakpoint on the tool definition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
    /// Unknown fields (e.g. `type: "custom"`, `defer_loading`), preserved
    /// for round-trips.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The `tool_choice` object.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolChoiceParam {
    /// `auto`, `any`, `tool` or `none`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Required tool name (`type: "tool"` only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Disables parallel tool use (not valid on `type: "none"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disable_parallel_tool_use: Option<bool>,
    /// Unknown fields, preserved for round-trips.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The `thinking` request configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ThinkingConfig {
    /// `enabled`, `adaptive` or `disabled`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Manual thinking budget (`type: "enabled"` only; not modeled by the
    /// IR — round-trips through `extra`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u64>,
    /// `summarized` (default) or `omitted`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display: Option<String>,
    /// Unknown fields, preserved for round-trips.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The `output_config` object.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct OutputConfig {
    /// Output effort level (`low` … `max`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    /// Structured-output format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<OutputFormatConfig>,
    /// Unknown fields (e.g. `task_budget`), preserved for round-trips.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// `output_config.format`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct OutputFormatConfig {
    /// Discriminator (`json_schema`).
    #[serde(rename = "type")]
    pub kind: String,
    /// The JSON Schema, verbatim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<Value>,
    /// Unknown fields, preserved for round-trips.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// A Messages API response (`type: "message"`), also nested in the
/// streaming `message_start` event.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct MessagesResponse {
    /// Message id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Object type (`message`).
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// Always `assistant`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Model that handled the request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Generated content blocks (dispatched manually on `type`).
    #[serde(default)]
    pub content: Vec<Value>,
    /// Why generation stopped; `null` in streaming `message_start`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    /// Matched custom stop sequence, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    /// Token accounting (kept raw: a usage object that fails the
    /// [`UsageWire`] shape degrades to a warning with no usage instead of
    /// failing an already-billed response or stream).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Value>,
    /// Unknown fields (e.g. `stop_details`, `container`), preserved.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The provider `usage` object.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct UsageWire {
    /// Uncached input tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    /// Output tokens (includes thinking).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    /// Tokens written to cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u64>,
    /// Tokens read from cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u64>,
    /// `{thinking_tokens}` breakdown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<OutputTokensDetails>,
    /// Unknown fields (e.g. `service_tier`, `server_tool_use`), preserved.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// `usage.output_tokens_details`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct OutputTokensDetails {
    /// Raw reasoning tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_tokens: Option<u64>,
    /// Unknown fields, preserved.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The `message_start` stream event payload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct MessageStartEvent {
    /// The initial message object (empty `content`, input-side usage).
    pub message: MessagesResponse,
}

/// The `content_block_start` stream event payload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ContentBlockStartEvent {
    /// Position of the block in the final `content` array.
    pub index: usize,
    /// The opening (possibly empty) content block.
    pub content_block: Value,
}

/// The `content_block_delta` stream event payload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ContentBlockDeltaEvent {
    /// Target block index.
    pub index: usize,
    /// The delta union (dispatched on `delta.type`).
    pub delta: Value,
}

/// The `content_block_stop` stream event payload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ContentBlockStopEvent {
    /// Completed block index.
    pub index: usize,
}

/// The `message_delta` stream event payload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct MessageDeltaEvent {
    /// Top-level changes to the final message.
    pub delta: MessageDeltaBody,
    /// Cumulative usage snapshot.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Map<String, Value>>,
    /// Unknown fields (e.g. `context_management`), preserved.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// `message_delta.delta`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct MessageDeltaBody {
    /// Final stop reason.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    /// Matched stop sequence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    /// Unknown fields (`container`, `stop_details`, …), preserved.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The provider error envelope (non-2xx bodies and `error` stream events).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ErrorEnvelope {
    /// The error detail.
    pub error: ErrorBody,
    /// Unknown fields (e.g. `request_id`), preserved.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// `error` detail object.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ErrorBody {
    /// Provider error type (`overloaded_error`, …).
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// Human-readable message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Unknown fields, preserved.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// One `GET /v1/models` page.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelsPage {
    /// The page of models (entries kept raw; `id`, `display_name` and
    /// `created_at` are extracted by the parser).
    #[serde(default)]
    pub data: Vec<Value>,
    /// First id in `data`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_id: Option<String>,
    /// Last id in `data`; the `after_id` cursor for the next page.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_id: Option<String>,
    /// Whether more results exist.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub has_more: Option<bool>,
    /// Unknown fields, preserved.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The `POST /v1/messages/count_tokens` response.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CountTokensResponse {
    /// Total input tokens for the prospective request.
    #[serde(default)]
    pub input_tokens: u64,
    /// Unknown fields (e.g. `context_management`), preserved.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
