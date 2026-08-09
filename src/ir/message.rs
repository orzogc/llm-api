//! Messages and round-trip metadata.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::block::ContentBlock;
use super::extra::Extra;
use super::role::Role;

/// Opaque, versioned round-trip metadata (§ 4.2).
///
/// Attached by format→IR parsers to record the original wire shape (e.g.
/// which IR messages were split out of one wire turn) and consumed by IR→
/// format serialization to restore it. Serializable so persisted IR keeps
/// it, but with no public fields. It is validated defensively on use: a
/// missing, malformed or future-version value degrades to canonical
/// placement — tampering can never corrupt a conversion, only forfeit
/// exact-shape restoration.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RoundTripMeta {
    // Raw payload; current version writes `{"v": 1, …}`. Unknown/future
    // payloads are preserved verbatim so a newer writer's metadata survives
    // being re-serialized by an older library.
    raw: Value,
}

impl RoundTripMeta {
    const VERSION: u64 = 1;

    /// Builds a v1 payload from format-specific fields (crate-internal:
    /// format parsers record their markers through this).
    pub(crate) fn v1(fields: Map<String, Value>) -> Self {
        let mut map = Map::new();
        map.insert("v".to_owned(), Value::from(Self::VERSION));
        map.extend(fields);
        Self { raw: Value::Object(map) }
    }

    /// Reads a field of a valid v1 payload; `None` for malformed or
    /// future-version payloads (defensive degradation, § 4.2).
    pub(crate) fn v1_field(&self, key: &str) -> Option<&Value> {
        let obj = self.raw.as_object()?;
        if obj.get("v").and_then(Value::as_u64) != Some(Self::VERSION) {
            return None;
        }
        obj.get(key)
    }

    /// Marks a message as split out of one wire turn; messages carrying the
    /// same group id, adjacent in the array, re-merge into a single turn on
    /// serialization.
    #[must_use]
    pub fn turn_group(group: u64) -> Self {
        let mut fields = Map::new();
        fields.insert("turn_group".to_owned(), Value::from(group));
        Self::v1(fields)
    }

    /// Reads the turn group id, if this metadata carries a valid one.
    #[must_use]
    pub fn turn_group_id(&self) -> Option<u64> {
        self.v1_field("turn_group")?.as_u64()
    }
}

/// One conversation message.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// The message role.
    pub role: Role,
    /// Ordered content blocks.
    pub content: Vec<ContentBlock>,
    /// Round-trip metadata recorded by parsers (§ 4.2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub round_trip: Option<RoundTripMeta>,
    /// Format-namespaced extra fields (provider data only).
    #[serde(default, skip_serializing_if = "Extra::is_empty")]
    pub extra: Extra,
}

impl Message {
    /// A message with the given role and content.
    #[must_use]
    pub fn new(role: Role, content: Vec<ContentBlock>) -> Self {
        Self { role, content, round_trip: None, extra: Extra::new() }
    }

    /// A user message.
    #[must_use]
    pub fn user(content: Vec<ContentBlock>) -> Self {
        Self::new(Role::User, content)
    }

    /// A user message holding one text block.
    #[must_use]
    pub fn user_text(text: impl Into<String>) -> Self {
        Self::user(vec![ContentBlock::text(text)])
    }

    /// An assistant message.
    #[must_use]
    pub fn assistant(content: Vec<ContentBlock>) -> Self {
        Self::new(Role::Assistant, content)
    }

    /// An assistant message holding one text block.
    #[must_use]
    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self::assistant(vec![ContentBlock::text(text)])
    }

    /// A system message holding one text block.
    #[must_use]
    pub fn system_text(text: impl Into<String>) -> Self {
        Self::new(Role::System, vec![ContentBlock::text(text)])
    }

    /// A developer message holding one text block.
    #[must_use]
    pub fn developer_text(text: impl Into<String>) -> Self {
        Self::new(Role::Developer, vec![ContentBlock::text(text)])
    }

    /// A tool message carrying tool results.
    #[must_use]
    pub fn tool(content: Vec<ContentBlock>) -> Self {
        Self::new(Role::Tool, content)
    }

    /// Concatenated text of all `Text` blocks.
    #[must_use]
    pub fn text(&self) -> String {
        let mut out = String::new();
        for block in &self.content {
            if let ContentBlock::Text { text, .. } = block {
                out.push_str(text);
            }
        }
        out
    }

    /// Attaches (replaces) the round-trip metadata.
    #[must_use]
    pub fn with_turn_group(mut self, group: u64) -> Self {
        self.round_trip = Some(RoundTripMeta::turn_group(group));
        self
    }

    /// Reads a valid turn-group id off this message, if any.
    #[must_use]
    pub fn turn_group_id(&self) -> Option<u64> {
        self.round_trip.as_ref().and_then(RoundTripMeta::turn_group_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn round_trip_meta_defensive() {
        // Valid v1.
        let m = RoundTripMeta::turn_group(3);
        assert_eq!(m.turn_group_id(), Some(3));
        assert_eq!(serde_json::to_value(&m).unwrap(), json!({"v": 1, "turn_group": 3}));

        // Malformed / future values degrade to None instead of failing.
        for raw in [json!(null), json!("junk"), json!({"v": 99, "turn_group": 3}), json!({})] {
            let meta: RoundTripMeta = serde_json::from_value(raw.clone()).unwrap();
            assert_eq!(meta.turn_group_id(), None);
            // Future payloads survive re-serialization verbatim.
            assert_eq!(serde_json::to_value(&meta).unwrap(), raw);
        }
    }

    #[test]
    fn message_serde_round_trip() {
        let msg = Message::user_text("hello").with_turn_group(1);
        let v = serde_json::to_value(&msg).unwrap();
        assert_eq!(
            v,
            json!({
                "role": "user",
                "content": [{"type": "text", "text": "hello"}],
                "round_trip": {"v": 1, "turn_group": 1},
            })
        );
        assert_eq!(serde_json::from_value::<Message>(v).unwrap(), msg);
    }

    #[test]
    fn message_text_concatenates() {
        let msg = Message::assistant(vec![
            ContentBlock::text("a"),
            ContentBlock::thinking("ignored"),
            ContentBlock::text("b"),
        ]);
        assert_eq!(msg.text(), "ab");
    }
}
