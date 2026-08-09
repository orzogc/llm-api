//! Structured-output configuration.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Structured output request (§ 4.9). The schema is passed through verbatim;
/// the library performs no subset validation or rewriting.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputFormat {
    /// JSON constrained by a schema.
    #[non_exhaustive]
    JsonSchema {
        /// Schema name (required by OpenAI CC upstream; `"response"` is
        /// synthesized when unset).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// Optional description.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        /// JSON Schema, verbatim.
        schema: Value,
        /// Strict schema adherence flag.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
    /// Schema-less JSON mode.
    JsonObject,
}

impl OutputFormat {
    /// Schema-constrained JSON output.
    #[must_use]
    pub fn json_schema(schema: Value) -> Self {
        Self::JsonSchema { name: None, description: None, schema, strict: None }
    }

    /// Schema-less JSON mode.
    #[must_use]
    pub fn json_object() -> Self {
        Self::JsonObject
    }

    /// Sets the schema name (`JsonSchema` only; ignored otherwise).
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        if let Self::JsonSchema { name: n, .. } = &mut self {
            *n = Some(name.into());
        }
        self
    }

    /// Sets the description (`JsonSchema` only; ignored otherwise).
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        if let Self::JsonSchema { description: d, .. } = &mut self {
            *d = Some(description.into());
        }
        self
    }

    /// Sets the strict flag (`JsonSchema` only; ignored otherwise).
    #[must_use]
    pub fn with_strict(mut self, strict: bool) -> Self {
        if let Self::JsonSchema { strict: s, .. } = &mut self {
            *s = Some(strict);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn serde_round_trip() {
        let f = OutputFormat::json_schema(json!({"type": "object"}))
            .with_name("answer")
            .with_strict(true);
        let v = serde_json::to_value(&f).unwrap();
        assert_eq!(
            v,
            json!({
                "type": "json_schema",
                "name": "answer",
                "schema": {"type": "object"},
                "strict": true,
            })
        );
        assert_eq!(serde_json::from_value::<OutputFormat>(v).unwrap(), f);
        assert_eq!(
            serde_json::to_value(OutputFormat::json_object()).unwrap(),
            json!({"type": "json_object"})
        );
    }
}
