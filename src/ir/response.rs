//! The unified response model.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use crate::convert::ConversionWarning;
use crate::error::Error;

use super::block::ContentBlock;
use super::message::Message;

/// Why generation stopped.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum StopReason {
    /// Natural end of turn.
    EndTurn,
    /// The output token limit was hit.
    MaxTokens,
    /// A stop sequence matched.
    StopSequence,
    /// The model called tools.
    ToolUse,
    /// Provider-side content filtering.
    ContentFilter,
    /// The model refused to answer.
    Refusal,
    /// Anthropic `pause_turn` — actionable: resend the turn as-is to
    /// continue.
    PauseTurn,
    /// Any unmapped provider value, verbatim.
    Other(String),
}

impl StopReason {
    /// The `snake_case` wire string of this reason.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::EndTurn => "end_turn",
            Self::MaxTokens => "max_tokens",
            Self::StopSequence => "stop_sequence",
            Self::ToolUse => "tool_use",
            Self::ContentFilter => "content_filter",
            Self::Refusal => "refusal",
            Self::PauseTurn => "pause_turn",
            Self::Other(s) => s,
        }
    }

    /// Parses a wire string (unknown strings become `Other`).
    #[must_use]
    pub fn from_str_lossy(s: &str) -> Self {
        match s {
            "end_turn" => Self::EndTurn,
            "max_tokens" => Self::MaxTokens,
            "stop_sequence" => Self::StopSequence,
            "tool_use" => Self::ToolUse,
            "content_filter" => Self::ContentFilter,
            "refusal" => Self::Refusal,
            "pause_turn" => Self::PauseTurn,
            other => Self::Other(other.to_owned()),
        }
    }
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for StopReason {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for StopReason {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(Self::from_str_lossy(&s))
    }
}

/// Unified token usage (§ 8).
///
/// `input_tokens` counts **all** input tokens including cache reads/writes;
/// `output_tokens` counts all generated tokens including reasoning. The
/// per-provider parsers perform the documented field-semantics alignment.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    /// All input tokens, including cache reads/writes.
    pub input_tokens: u64,
    /// All generated tokens, including reasoning.
    pub output_tokens: u64,
    /// Provider-reported total, where present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    /// Tokens read from cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u64>,
    /// Tokens written to cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u64>,
    /// Reasoning/thinking tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
    /// The original provider usage object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<Value>,
}

impl Usage {
    /// Visible (non-reasoning) output tokens, saturating so misbehaving
    /// provider data cannot underflow.
    #[must_use]
    pub fn visible_output_tokens(&self) -> u64 {
        self.output_tokens
            .saturating_sub(self.reasoning_tokens.unwrap_or(0))
    }
}

/// A parsed provider response (§ 8).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Response {
    /// Provider response id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Model that produced the response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The assistant message; re-enters subsequent requests as history and
    /// carries namespaced `extra` (signatures, item ids, …).
    pub message: Message,
    /// Normalized stop reason; `None` when the source omits it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,
    /// Token usage; absent on some streams/dialects.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// HTTP status.
    pub status: u16,
    /// Response headers (rate-limit headers etc.). Serialized through a
    /// `Vec<(String, String)>` representation, lossy for non-UTF-8 values.
    #[serde(with = "header_map_serde")]
    pub headers: http::HeaderMap,
    /// The complete original response body (`None` for responses
    /// accumulated from a stream).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<Value>,
    /// Request-build plus response-parse warnings.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<ConversionWarning>,
}

impl Response {
    /// A response wrapping the given assistant message; used by parsers and
    /// the stream accumulator.
    #[must_use]
    pub fn new(message: Message) -> Self {
        Self {
            id: None,
            model: None,
            message,
            stop_reason: None,
            usage: None,
            status: 200,
            headers: http::HeaderMap::new(),
            raw: None,
            warnings: Vec::new(),
        }
    }

    /// Concatenated text of all `Text` blocks of the message.
    #[must_use]
    pub fn text(&self) -> String {
        self.message.text()
    }

    /// Parses the first `Text` block as JSON — the § 4.9 structured-output
    /// convenience.
    pub fn parse_json(&self) -> Result<Value, Error> {
        let text = self.message.content.iter().find_map(|b| match b {
            ContentBlock::Text { text, .. } => Some(text.as_str()),
            _ => None,
        });
        let Some(text) = text else {
            return Err(Error::Parse {
                message: "response contains no text block".to_owned(),
                raw: bytes::Bytes::new(),
            });
        };
        serde_json::from_str(text).map_err(|e| Error::Parse {
            message: format!("first text block is not valid JSON: {e}"),
            raw: bytes::Bytes::copy_from_slice(text.as_bytes()),
        })
    }
}

/// `true` when the block is refusal-marked: a block whose `extra` records
/// `{"refusal": true}` in any format namespace (§ 9).
#[must_use]
pub fn is_refusal_block(block: &ContentBlock) -> bool {
    block.extra().is_some_and(|extra| {
        extra.formats().any(|fmt| {
            extra
                .get(fmt)
                .and_then(|ns| ns.get("refusal"))
                .and_then(Value::as_bool)
                .unwrap_or(false)
        })
    })
}

/// Applies the § 8 stop-reason normalization rules, in order:
/// 1. a would-be `EndTurn` whose message contains `ToolCall` blocks becomes
///    `ToolUse` (Google reports `STOP` for function calls);
/// 2. a would-be `EndTurn` — or absent stop reason — whose message contains
///    a refusal-marked block becomes `Refusal`.
#[must_use]
pub fn normalize_stop_reason(
    message: &Message,
    stop_reason: Option<StopReason>,
) -> Option<StopReason> {
    let mut sr = stop_reason;
    if sr == Some(StopReason::EndTurn)
        && message
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolCall { .. }))
    {
        sr = Some(StopReason::ToolUse);
    }
    if matches!(sr, Some(StopReason::EndTurn) | None)
        && message.content.iter().any(is_refusal_block)
    {
        sr = Some(StopReason::Refusal);
    }
    sr
}

mod header_map_serde {
    //! Serializes `http::HeaderMap` as `Vec<(String, String)>`, lossy for
    //! non-UTF-8 header values.

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(
        headers: &http::HeaderMap,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let pairs: Vec<(String, String)> = headers
            .iter()
            .map(|(name, value)| {
                (
                    name.as_str().to_owned(),
                    String::from_utf8_lossy(value.as_bytes()).into_owned(),
                )
            })
            .collect();
        pairs.serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<http::HeaderMap, D::Error> {
        let pairs = Vec::<(String, String)>::deserialize(deserializer)?;
        let mut map = http::HeaderMap::new();
        for (name, value) in pairs {
            // Entries that fail to parse are skipped (defensive: persisted
            // data may have been edited).
            if let (Ok(name), Ok(value)) = (
                name.parse::<http::HeaderName>(),
                http::HeaderValue::from_str(&value),
            ) {
                map.append(name, value);
            }
        }
        Ok(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Extra;
    use serde_json::json;

    #[test]
    fn stop_reason_serde() {
        assert_eq!(
            serde_json::to_string(&StopReason::EndTurn).unwrap(),
            "\"end_turn\""
        );
        let other: StopReason = serde_json::from_str("\"compaction\"").unwrap();
        assert_eq!(other, StopReason::Other("compaction".into()));
        assert_eq!(serde_json::to_string(&other).unwrap(), "\"compaction\"");
    }

    #[test]
    fn usage_visible_output_saturates() {
        let u = Usage {
            output_tokens: 5,
            reasoning_tokens: Some(9),
            ..Usage::default()
        };
        assert_eq!(u.visible_output_tokens(), 0);
        let u2 = Usage {
            output_tokens: 10,
            reasoning_tokens: Some(3),
            ..Usage::default()
        };
        assert_eq!(u2.visible_output_tokens(), 7);
    }

    #[test]
    fn normalization_rules() {
        // EndTurn + ToolCall → ToolUse.
        let msg = Message::assistant(vec![ContentBlock::tool_call("f", "{}")]);
        assert_eq!(
            normalize_stop_reason(&msg, Some(StopReason::EndTurn)),
            Some(StopReason::ToolUse)
        );
        // Only EndTurn is rewritten by the tool rule.
        assert_eq!(
            normalize_stop_reason(&msg, Some(StopReason::MaxTokens)),
            Some(StopReason::MaxTokens)
        );
        assert_eq!(normalize_stop_reason(&msg, None), None);

        // Refusal-marked text block.
        let mut extra = Extra::new();
        extra.set("openai_chat_completions", "refusal", true);
        let refusal_msg = Message::assistant(vec![ContentBlock::Text {
            text: "cannot help".into(),
            cache: None,
            extra,
        }]);
        assert_eq!(
            normalize_stop_reason(&refusal_msg, Some(StopReason::EndTurn)),
            Some(StopReason::Refusal)
        );
        assert_eq!(
            normalize_stop_reason(&refusal_msg, None),
            Some(StopReason::Refusal)
        );
        // A non-EndTurn reason is left alone.
        assert_eq!(
            normalize_stop_reason(&refusal_msg, Some(StopReason::MaxTokens)),
            Some(StopReason::MaxTokens)
        );
    }

    #[test]
    fn response_parse_json() {
        let resp = Response::new(Message::assistant_text(r#"{"a": 1}"#));
        assert_eq!(resp.parse_json().unwrap(), json!({"a": 1}));
        let bad = Response::new(Message::assistant_text("not json"));
        assert!(bad.parse_json().is_err());
        let empty = Response::new(Message::assistant(vec![]));
        assert!(empty.parse_json().is_err());
    }

    #[test]
    fn response_serde_round_trip_with_headers() {
        let mut resp = Response::new(Message::assistant_text("hi"));
        resp.headers
            .insert("x-request-id", http::HeaderValue::from_static("abc"));
        resp.status = 200;
        let s = serde_json::to_string(&resp).unwrap();
        let back: Response = serde_json::from_str(&s).unwrap();
        assert_eq!(back, resp);
    }
}
