//! The unified request.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::block::ContentBlock;
use super::extra::Extra;
use super::message::{Message, RoundTripMeta};
use super::output::OutputFormat;
use super::reasoning::Reasoning;
use super::tools::{Tool, ToolChoice};

/// The unified chat request (§ 4.1).
///
/// Whether a call is streaming is decided by the client method (`send` vs
/// `stream`), not by an IR field. The model name is supplied via
/// `ProviderConfig` / call options, not the IR.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Request {
    /// Top-level system content. Text blocks only; enforced at conversion
    /// time (`ConversionError`). Placement rules per format in § 7.1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<Vec<ContentBlock>>,
    /// The conversation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub messages: Vec<Message>,
    /// Maximum output tokens (required by Anthropic; unset there is a
    /// conversion error unless `AnthropicOptions.default_max_tokens` is
    /// configured).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// Sampling temperature, passed through verbatim (ranges differ per
    /// provider; out-of-range values are the upstream API's error).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Nucleus sampling, verbatim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Top-k sampling, verbatim (Anthropic/Google only; others warn).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Stop sequences.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// Sampling seed (OpenAI CC / Google; others warn).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Frequency penalty (OpenAI CC / Google; others warn).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    /// Presence penalty (OpenAI CC / Google; others warn).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    /// Request metadata (per-target mapping in § 4.6).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Map<String, Value>>,
    /// Reasoning configuration (§ 4.7).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
    /// Available tools (§ 4.5).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// Tool-choice constraint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// Whether the model may call tools in parallel (§ 4.5).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// Structured output (§ 4.9).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_format: Option<OutputFormat>,
    /// Prompt cache key → OpenAI `prompt_cache_key`; Anthropic and Google
    /// warn (§ 4.8).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_key: Option<String>,
    /// Round-trip metadata recorded by parsers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub round_trip: Option<RoundTripMeta>,
    /// Format-namespaced extra fields (§ 5).
    #[serde(default, skip_serializing_if = "Extra::is_empty")]
    pub extra: Extra,
}

impl Request {
    /// An empty request.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A request with the given messages.
    #[must_use]
    pub fn with_messages(messages: Vec<Message>) -> Self {
        Self { messages, ..Self::default() }
    }

    /// Sets `system` to a single text block.
    #[must_use]
    pub fn with_system_text(mut self, text: impl Into<String>) -> Self {
        self.system = Some(vec![ContentBlock::text(text)]);
        self
    }

    /// Appends a message.
    #[must_use]
    pub fn with_message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn request_serde_skips_unset() {
        let req = Request::with_messages(vec![Message::user_text("hi")]);
        let v = serde_json::to_value(&req).unwrap();
        assert_eq!(
            v,
            json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]})
        );
        assert_eq!(serde_json::from_value::<Request>(v).unwrap(), req);
    }

    #[test]
    fn persisted_request_with_future_field_deserializes() {
        let v = json!({"messages": [], "field_from_the_future": {"x": 1}});
        let req: Request = serde_json::from_value(v).unwrap();
        assert_eq!(req, Request::new());
    }
}
