//! Reasoning (thinking) configuration.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::extra::Extra;

/// Reasoning effort tier. `Other` passes its raw string through verbatim.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Effort {
    /// Disable reasoning where the tier set supports it.
    None,
    /// Minimal effort.
    Minimal,
    /// Low effort.
    Low,
    /// Medium effort.
    Medium,
    /// High effort.
    High,
    /// Extra-high effort.
    XHigh,
    /// Maximum effort.
    Max,
    /// A tier this library does not model; passed through verbatim.
    Other(String),
}

impl Effort {
    /// The wire string for this tier.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
            Self::Other(s) => s,
        }
    }

    /// Parses a wire string into a tier (unknown strings become `Other`).
    #[must_use]
    pub fn from_str_lossy(s: &str) -> Self {
        match s {
            "none" => Self::None,
            "minimal" => Self::Minimal,
            "low" => Self::Low,
            "medium" => Self::Medium,
            "high" => Self::High,
            "xhigh" => Self::XHigh,
            "max" => Self::Max,
            other => Self::Other(other.to_owned()),
        }
    }
}

impl std::fmt::Display for Effort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for Effort {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for Effort {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(Self::from_str_lossy(&s))
    }
}

/// Reasoning configuration (§ 4.7).
///
/// Anthropic `budget_tokens` and Google `thinkingBudget` are not modeled;
/// use `extra` where a provider still needs them.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Reasoning {
    /// Turns reasoning on or off where the target distinguishes that from
    /// effort tiers. If `enabled` and `effort` conflict, `effort` wins and a
    /// cosmetic warning is attached.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    /// Effort tier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<Effort>,
    /// Whether to return thinking summaries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_thoughts: Option<bool>,
    /// Format-namespaced extra fields (e.g. Responses
    /// `reasoning.summary: "detailed"`).
    #[serde(default, skip_serializing_if = "Extra::is_empty")]
    pub extra: Extra,
}

impl Reasoning {
    /// An empty configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A configuration that just sets `enabled`.
    #[must_use]
    pub fn enabled(on: bool) -> Self {
        Self {
            enabled: Some(on),
            ..Self::default()
        }
    }

    /// A configuration that just sets an effort tier.
    #[must_use]
    pub fn effort(effort: Effort) -> Self {
        Self {
            effort: Some(effort),
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effort_serde() {
        assert_eq!(serde_json::to_string(&Effort::XHigh).unwrap(), "\"xhigh\"");
        assert_eq!(
            serde_json::from_str::<Effort>("\"max\"").unwrap(),
            Effort::Max
        );
        let other: Effort = serde_json::from_str("\"turbo\"").unwrap();
        assert_eq!(other, Effort::Other("turbo".into()));
        assert_eq!(serde_json::to_string(&other).unwrap(), "\"turbo\"");
    }

    #[test]
    fn reasoning_skips_unset_fields() {
        let r = Reasoning::effort(Effort::Low);
        assert_eq!(serde_json::to_string(&r).unwrap(), r#"{"effort":"low"}"#);
    }
}
