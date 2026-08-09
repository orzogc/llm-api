//! Complete serde wire types for the Google `generateContent` API
//! (Generative Language API, `v1beta`).
//!
//! Every object type carries a flattened `extra` map that captures unknown
//! fields on parse, so non-null unknown fields round-trip verbatim (design
//! § 1). Only the fields the IR models are typed; everything else flows
//! through `extra`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// The `GenerateContentRequest` body shared by `generateContent` and
/// `streamGenerateContent`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentRequest {
    /// The conversation contents (required upstream).
    #[serde(default)]
    pub contents: Vec<Content>,
    /// Tools the model may use.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// Configuration for the tools in the request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<ToolConfig>,
    /// Developer system instruction (text only upstream).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<Content>,
    /// Generation and output options.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
    /// Unknown fields (`safetySettings`, `cachedContent`, `serviceTier`, …).
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The multi-part content of a single message.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Content {
    /// Producer of the content: `"user"` or `"model"`; may be unset for
    /// single-turn requests and `systemInstruction`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Ordered parts making up one message.
    #[serde(default)]
    pub parts: Vec<Part>,
    /// Unknown fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// One part of a [`Content`]. Holds exactly one member of the `data` union
/// plus optional metadata (`thought`, `thoughtSignature`).
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Part {
    /// Inline text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Inline media bytes (base64).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inline_data: Option<Blob>,
    /// A model-predicted function call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
    /// The client-provided result of a `functionCall`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function_response: Option<FunctionResponse>,
    /// URI-based data (typically a Files API URI).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_data: Option<FileData>,
    /// `true` if the part is a thought (reasoning summary) from the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thought: Option<bool>,
    /// Opaque signature for the thought; echoed back in subsequent requests
    /// to preserve reasoning context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
    /// Unknown fields (`executableCode`, `codeExecutionResult`,
    /// `videoMetadata`, `partMetadata`, …).
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Inline media bytes.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Blob {
    /// IANA MIME type of the data.
    #[serde(default)]
    pub mime_type: String,
    /// Base64-encoded raw bytes.
    #[serde(default)]
    pub data: String,
    /// Unknown fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// URI-based data.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileData {
    /// IANA MIME type of the source data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// URI of the file.
    #[serde(default)]
    pub file_uri: String,
    /// Unknown fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// A model-predicted function call.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Unique id of the call; when populated, the client returns a
    /// `functionResponse` with the matching id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Name of the function to call.
    #[serde(default)]
    pub name: String,
    /// Function arguments as a JSON object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub args: Option<Value>,
    /// Unknown fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The client-provided result of a `functionCall`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FunctionResponse {
    /// Id of the `functionCall` this responds to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Name of the called function (required upstream).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Function output as a JSON object. Documented keys: `"output"` for
    /// function output, `"error"` to report failure; with neither, the whole
    /// object is treated as function output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response: Option<Value>,
    /// Ordered media parts of the response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parts: Option<Vec<FunctionResponsePart>>,
    /// Unknown fields (`willContinue`, `scheduling`, …).
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// One media part of a `functionResponse`. The documented union has a
/// single member, `inlineData`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionResponsePart {
    /// Inline media bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inline_data: Option<Blob>,
    /// Unknown fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// One entry of the request `tools` array. An entry may combine
/// `functionDeclarations` with hosted tool members (`googleSearch`,
/// `codeExecution`, …); the unmodeled members are captured by `extra`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    /// Functions the model may call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function_declarations: Option<Vec<FunctionDeclaration>>,
    /// Unknown fields (hosted tool members).
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// A function the model may call.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionDeclaration {
    /// Function name.
    #[serde(default)]
    pub name: String,
    /// Brief description of the function.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Parameters as raw JSON Schema (the library's passthrough channel;
    /// mutually exclusive with the OpenAPI-style `parameters`, which is kept
    /// in `extra` when parsed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters_json_schema: Option<Value>,
    /// Unknown fields (`parameters`, `response`, `behavior`, …).
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Configuration for the tools in the request.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolConfig {
    /// Function calling behavior.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function_calling_config: Option<FunctionCallingConfig>,
    /// Unknown fields (`retrievalConfig`, …).
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Function calling behavior.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCallingConfig {
    /// Calling mode: `AUTO`, `ANY`, `NONE` or `VALIDATED`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    /// Restricts which functions may be called (`ANY`/`VALIDATED` only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_function_names: Option<Vec<String>>,
    /// Unknown fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Generation and output options.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    /// Stop sequences (up to 5 upstream).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// Output MIME type (`application/json` enables JSON mode).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<String>,
    /// Structured-output schema as raw JSON Schema.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_json_schema: Option<Value>,
    /// Maximum tokens per candidate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// Sampling temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Nucleus sampling cap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Top-k sampling cap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Decoding seed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Presence penalty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    /// Frequency penalty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    /// Thinking configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<ThinkingConfig>,
    /// Unknown fields (`candidateCount`, `responseModalities`,
    /// `responseSchema`, `speechConfig`, …).
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Thinking configuration.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingConfig {
    /// Include thought-summary parts (`thought: true`) in the response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_thoughts: Option<bool>,
    /// Depth of reasoning: `MINIMAL`, `LOW`, `MEDIUM` or `HIGH`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level: Option<String>,
    /// Unknown fields (`thinkingBudget`, …; deliberately not modeled,
    /// design § 4.7).
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The `GenerateContentResponse` body — also the per-chunk payload of
/// `streamGenerateContent`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentResponse {
    /// Candidate responses; empty only when the prompt itself was blocked.
    #[serde(default)]
    pub candidates: Vec<Candidate>,
    /// Prompt content-filter feedback (set when the prompt is blocked).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_feedback: Option<PromptFeedback>,
    /// Token usage for the request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage_metadata: Option<UsageMetadata>,
    /// Model version used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
    /// Identifier of this response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// Unknown fields (`modelStatus`, …).
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// One candidate response.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    /// Generated content (role `"model"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Content>,
    /// Why generation stopped; absent while still generating (streaming).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// Index of the candidate in `candidates`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<i64>,
    /// Unknown fields (`safetyRatings`, `citationMetadata`,
    /// `groundingMetadata`, `avgLogprobs`, …).
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Prompt content-filter feedback.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptFeedback {
    /// When set, the prompt was blocked and there are no candidates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_reason: Option<String>,
    /// Unknown fields (`safetyRatings`, …).
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Token usage. proto3 JSON omits zero-valued fields, so every count is
/// optional; negative values (never sent by well-behaved servers) are
/// clamped to zero when converted to IR usage.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    /// Tokens in the prompt (includes cached tokens when `cachedContent`
    /// is used).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_token_count: Option<i64>,
    /// Tokens in the cached part of the prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_content_token_count: Option<i64>,
    /// Total tokens across the generated candidates (excludes thoughts).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidates_token_count: Option<i64>,
    /// Thought tokens (thinking models).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thoughts_token_count: Option<i64>,
    /// Total token count (prompt + thoughts + candidates).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_token_count: Option<i64>,
    /// Unknown fields (per-modality breakdowns, `serviceTier`, …).
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The `models.countTokens` response body.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CountTokensResponse {
    /// Number of tokens the model tokenizes the prompt into.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<i64>,
    /// Unknown fields (`cachedContentTokenCount`, `promptTokensDetails`, …).
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The `models.list` response body. Model entries are kept as raw values —
/// the whole entry becomes [`crate::models::Model::raw`].
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ListModelsResponse {
    /// The returned models.
    #[serde(default)]
    pub models: Vec<Value>,
    /// Cursor for the next page; omitted on the last page.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_page_token: Option<String>,
    /// Unknown fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn unknown_fields_round_trip_through_extra() {
        let src = json!({
            "contents": [{
                "role": "user",
                "parts": [{"text": "hi", "partMetadata": {"k": 1}}],
                "customContentField": true,
            }],
            "safetySettings": [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}],
        });
        let req: GenerateContentRequest = serde_json::from_value(src.clone()).unwrap();
        assert_eq!(req.contents[0].parts[0].text.as_deref(), Some("hi"));
        assert_eq!(
            req.contents[0].parts[0].extra["partMetadata"],
            json!({"k": 1})
        );
        assert_eq!(req.contents[0].extra["customContentField"], json!(true));
        assert!(req.extra.contains_key("safetySettings"));
        assert_eq!(serde_json::to_value(&req).unwrap(), src);
    }

    #[test]
    fn part_union_members_parse() {
        let part: Part = serde_json::from_value(json!({
            "functionCall": {"id": "c1", "name": "f", "args": {"a": 1}},
            "thoughtSignature": "sig",
        }))
        .unwrap();
        let fc = part.function_call.unwrap();
        assert_eq!(fc.id.as_deref(), Some("c1"));
        assert_eq!(fc.args, Some(json!({"a": 1})));
        assert_eq!(part.thought_signature.as_deref(), Some("sig"));
    }

    #[test]
    fn response_defaults_are_lenient() {
        // A blocked-prompt response has no candidates at all.
        let resp: GenerateContentResponse = serde_json::from_value(json!({
            "promptFeedback": {"blockReason": "SAFETY"},
        }))
        .unwrap();
        assert!(resp.candidates.is_empty());
        assert_eq!(
            resp.prompt_feedback.unwrap().block_reason.as_deref(),
            Some("SAFETY")
        );

        // A final stream chunk may carry content without parts.
        let cand: Candidate = serde_json::from_value(json!({
            "content": {"role": "model"},
            "finishReason": "STOP",
        }))
        .unwrap();
        assert!(cand.content.unwrap().parts.is_empty());
    }
}
