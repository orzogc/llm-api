//! Model listing types (§ 13).

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One model, as returned by a provider's model-listing endpoint. Only the
/// intersection is modeled; everything else stays in `raw`.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Model {
    /// Model id, normalized to what `ProviderConfig.model` accepts
    /// (Google's `models/` prefix is stripped; the original resource name
    /// stays in `raw`).
    pub id: String,
    /// Human-readable name, where the provider has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    /// Creation time, where the provider has one. A timestamp that fails
    /// to parse degrades to `None` and never fails the model-list call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created: Option<SystemTime>,
    /// The complete original entry.
    pub raw: Value,
}

impl Model {
    /// A model with just an id and its raw entry.
    #[must_use]
    pub fn new(id: impl Into<String>, raw: Value) -> Self {
        Self { id: id.into(), display_name: None, created: None, raw }
    }
}

/// Parses Unix seconds into a `SystemTime` (OpenAI `created`).
#[must_use]
pub fn system_time_from_unix_seconds(secs: i64) -> Option<SystemTime> {
    if secs >= 0 {
        UNIX_EPOCH.checked_add(Duration::from_secs(secs.unsigned_abs()))
    } else {
        UNIX_EPOCH.checked_sub(Duration::from_secs(secs.unsigned_abs()))
    }
}

/// Parses an RFC 3339 timestamp into a `SystemTime` (Anthropic
/// `created_at`). Returns `None` on any parse failure.
#[must_use]
pub fn system_time_from_rfc3339(s: &str) -> Option<SystemTime> {
    let parsed =
        time::OffsetDateTime::parse(s, &time::format_description::well_known::Rfc3339).ok()?;
    Some(parsed.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unix_seconds() {
        let t = system_time_from_unix_seconds(1_700_000_000).unwrap();
        assert_eq!(t.duration_since(UNIX_EPOCH).unwrap().as_secs(), 1_700_000_000);
        // Pre-epoch values still parse.
        assert!(system_time_from_unix_seconds(-1).is_some());
    }

    #[test]
    fn rfc3339() {
        let t = system_time_from_rfc3339("2024-02-01T00:00:00Z").unwrap();
        assert!(t > UNIX_EPOCH);
        assert!(system_time_from_rfc3339("not a date").is_none());
    }
}
