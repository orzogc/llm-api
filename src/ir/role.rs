//! Message roles.

use serde::{Deserialize, Serialize};

/// The union of the roles supported by all covered formats.
///
/// Downgrade rules for formats that lack a role are defined in § 7.1 of the
/// design document.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System instructions.
    System,
    /// OpenAI-family developer instructions.
    Developer,
    /// End-user input.
    User,
    /// Model output (also used for prefill).
    Assistant,
    /// Tool results; the canonical IR home for `ToolResult` blocks.
    Tool,
}

impl Role {
    /// The lowercase wire-style name of the role.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::Developer => "developer",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_round_trip() {
        for (role, s) in [
            (Role::System, "\"system\""),
            (Role::Developer, "\"developer\""),
            (Role::User, "\"user\""),
            (Role::Assistant, "\"assistant\""),
            (Role::Tool, "\"tool\""),
        ] {
            assert_eq!(serde_json::to_string(&role).unwrap(), s);
            assert_eq!(serde_json::from_str::<Role>(s).unwrap(), role);
        }
    }
}
