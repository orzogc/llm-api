//! Tool definitions and tool choice.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::block::CacheHint;
use super::extra::Extra;

/// A tool the model may call.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Tool {
    /// A user-defined function tool.
    Function(FunctionTool),
    /// Unmodeled tool kind (hosted/built-in tools, MCP toolsets, …);
    /// serializes back only to its own format, other targets warn.
    #[non_exhaustive]
    Opaque {
        /// The canonical id of the format the tool belongs to.
        format: String,
        /// The original JSON value, verbatim.
        value: Value,
    },
}

impl Tool {
    /// A function tool.
    #[must_use]
    pub fn function(tool: FunctionTool) -> Self {
        Self::Function(tool)
    }

    /// An unmodeled provider tool.
    #[must_use]
    pub fn opaque(format: impl Into<String>, value: Value) -> Self {
        Self::Opaque {
            format: format.into(),
            value,
        }
    }
}

/// A user-defined function tool (§ 4.5).
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FunctionTool {
    /// Tool name.
    pub name: String,
    /// Human-readable description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema for the arguments, passed through verbatim.
    /// `None` means "no parameters" (per-target encodings in § 4.5).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
    /// Strict schema adherence flag (OpenAI-family / Anthropic).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    /// Anthropic tool definitions accept `cache_control`; other targets
    /// drop this with a cosmetic warning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache: Option<CacheHint>,
    /// Format-namespaced extra fields.
    #[serde(default, skip_serializing_if = "Extra::is_empty")]
    pub extra: Extra,
}

impl FunctionTool {
    /// A function tool with just a name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// Sets the description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Sets the JSON Schema parameters.
    #[must_use]
    pub fn with_parameters(mut self, schema: Value) -> Self {
        self.parameters = Some(schema);
        self
    }

    /// Sets the strict flag.
    #[must_use]
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = Some(strict);
        self
    }

    /// Sets a cache hint on the tool definition.
    #[must_use]
    pub fn with_cache(mut self, hint: CacheHint) -> Self {
        self.cache = Some(hint);
        self
    }
}

/// How the model is allowed to choose tools (§ 4.5 mapping table).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    /// The model decides.
    Auto,
    /// The model must not call tools.
    None,
    /// The model must call some tool.
    Required,
    /// The model must call the named tool.
    #[non_exhaustive]
    Tool {
        /// Name of the required tool.
        name: String,
    },
}

impl ToolChoice {
    /// Requires the named tool.
    #[must_use]
    pub fn tool(name: impl Into<String>) -> Self {
        Self::Tool { name: name.into() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tool_serde() {
        let t = Tool::function(
            FunctionTool::new("get_weather")
                .with_description("d")
                .with_parameters(json!({"type": "object"})),
        );
        let v = serde_json::to_value(&t).unwrap();
        assert_eq!(
            v,
            json!({
                "type": "function",
                "name": "get_weather",
                "description": "d",
                "parameters": {"type": "object"},
            })
        );
        assert_eq!(serde_json::from_value::<Tool>(v).unwrap(), t);
    }

    #[test]
    fn tool_choice_serde() {
        assert_eq!(
            serde_json::to_value(ToolChoice::Auto).unwrap(),
            json!({"type": "auto"})
        );
        let t = ToolChoice::tool("f");
        let v = serde_json::to_value(&t).unwrap();
        assert_eq!(v, json!({"type": "tool", "name": "f"}));
        assert_eq!(serde_json::from_value::<ToolChoice>(v).unwrap(), t);
    }
}
