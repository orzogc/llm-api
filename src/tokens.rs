//! Token counting types (§ 13). The library never estimates tokens locally.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::convert::ConversionWarning;

/// Result of a provider token-count endpoint.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TokenCount {
    /// Input tokens the provider reported for the prospective request.
    pub input_tokens: u64,
    /// The original response body.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<Value>,
    /// Chat-build warnings plus the count adapter's own (§ 13): dropped
    /// non-generated fields and decoupled count formats make the result
    /// approximate and produce semantic warnings.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<ConversionWarning>,
}

impl TokenCount {
    /// A count with the given token number.
    #[must_use]
    pub fn new(input_tokens: u64) -> Self {
        Self { input_tokens, raw: None, warnings: Vec::new() }
    }
}
