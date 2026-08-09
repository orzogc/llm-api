//! Content blocks — the unit of message content in the IR.

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

use super::extra::Extra;

/// A block-level cache breakpoint hint (§ 4.8).
///
/// Maps to Anthropic `cache_control: {type: "ephemeral", ttl}` and OpenAI
/// content-part `prompt_cache_breakpoint`; other targets warn.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheHint {
    /// Provider-specific TTL string (Anthropic accepts `"5m"` / `"1h"`),
    /// passed through verbatim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
}

impl CacheHint {
    /// A cache hint without a TTL.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A cache hint with the given TTL (passed through verbatim).
    #[must_use]
    pub fn with_ttl(ttl: impl Into<String>) -> Self {
        Self { ttl: Some(ttl.into()) }
    }
}

/// Where an image comes from. Conversion performs zero IO — the library
/// never downloads a URL to inline it (§ 4.3).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageSource {
    /// A remote URL.
    Url(String),
    /// Inline base64 data with its media type (e.g. `image/png`).
    #[non_exhaustive]
    Base64 {
        /// MIME type of the data.
        media_type: String,
        /// Base64-encoded bytes (no `data:` prefix).
        data: String,
    },
    /// A provider-issued file id; only meaningful on the provider that
    /// issued it — cross-provider conversion warns.
    FileId(String),
}

impl ImageSource {
    /// An image referenced by URL.
    #[must_use]
    pub fn url(url: impl Into<String>) -> Self {
        Self::Url(url.into())
    }

    /// An inline base64 image.
    #[must_use]
    pub fn base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self::Base64 { media_type: media_type.into(), data: data.into() }
    }

    /// A provider file id.
    #[must_use]
    pub fn file_id(id: impl Into<String>) -> Self {
        Self::FileId(id.into())
    }
}

// Manual serde: `{"type": "url", "url": …}` / `{"type": "base64",
// "media_type": …, "data": …}` / `{"type": "file_id", "file_id": …}`.
// Tuple variants cannot use serde's internal tagging.
impl Serialize for ImageSource {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(None)?;
        match self {
            Self::Url(url) => {
                map.serialize_entry("type", "url")?;
                map.serialize_entry("url", url)?;
            }
            Self::Base64 { media_type, data, .. } => {
                map.serialize_entry("type", "base64")?;
                map.serialize_entry("media_type", media_type)?;
                map.serialize_entry("data", data)?;
            }
            Self::FileId(id) => {
                map.serialize_entry("type", "file_id")?;
                map.serialize_entry("file_id", id)?;
            }
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for ImageSource {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut map = Map::deserialize(deserializer)?;
        let ty = map
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| D::Error::missing_field("type"))?
            .to_owned();
        let take_str = |map: &mut Map<String, Value>, key: &'static str| {
            match map.remove(key) {
                Some(Value::String(s)) => Ok(s),
                Some(_) => Err(D::Error::custom(format!("`{key}` must be a string"))),
                None => Err(D::Error::missing_field(key)),
            }
        };
        match ty.as_str() {
            "url" => Ok(Self::Url(take_str(&mut map, "url")?)),
            "base64" => Ok(Self::Base64 {
                media_type: take_str(&mut map, "media_type")?,
                data: take_str(&mut map, "data")?,
            }),
            "file_id" => Ok(Self::FileId(take_str(&mut map, "file_id")?)),
            other => Err(D::Error::custom(format!("unknown image source type `{other}`"))),
        }
    }
}

/// One unit of message content.
///
/// `Opaque` carries any provider node the IR does not model, in place, so
/// order and same-format round-trips survive; it serializes back only to its
/// own `format` — `value` is its own escape hatch, edit it directly.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Plain text.
    #[non_exhaustive]
    Text {
        /// The text.
        text: String,
        /// Optional cache breakpoint.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache: Option<CacheHint>,
        /// Format-namespaced extra fields.
        #[serde(default, skip_serializing_if = "Extra::is_empty")]
        extra: Extra,
    },
    /// Image input.
    #[non_exhaustive]
    Image {
        /// Where the image comes from.
        source: ImageSource,
        /// Optional cache breakpoint.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache: Option<CacheHint>,
        /// Format-namespaced extra fields.
        #[serde(default, skip_serializing_if = "Extra::is_empty")]
        extra: Extra,
    },
    /// An assistant-issued tool call.
    #[non_exhaustive]
    ToolCall {
        /// Provider call id. Optional because Google's `functionCall.id` is
        /// optional; formats that require ids raise a `ConversionError`
        /// when one is absent.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// Tool name.
        name: String,
        /// Raw argument string, exact bytes as received; invalid JSON from a
        /// model is preserved. Use [`ContentBlock::arguments_json`] to parse.
        arguments: String,
        /// Optional cache breakpoint.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache: Option<CacheHint>,
        /// Format-namespaced extra fields.
        #[serde(default, skip_serializing_if = "Extra::is_empty")]
        extra: Extra,
    },
    /// The result of a tool call, canonically carried by a `Tool`-role
    /// message.
    #[non_exhaustive]
    ToolResult {
        /// Id of the tool call this result answers.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_call_id: Option<String>,
        /// Tool name; required by Google's `functionResponse`, optional
        /// elsewhere.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// Result content (restricted union, § 4.3).
        content: Vec<ToolOutputBlock>,
        /// Marks the result as an error (native to Anthropic only).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        /// Optional cache breakpoint.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache: Option<CacheHint>,
        /// Format-namespaced extra fields.
        #[serde(default, skip_serializing_if = "Extra::is_empty")]
        extra: Extra,
    },
    /// Model thinking / chain of thought (§ 4.4).
    #[non_exhaustive]
    Thinking {
        /// Plaintext chain of thought or joined summary text.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text: Option<String>,
        /// Opaque payload required for replay (Anthropic `signature` /
        /// `redacted_thinking.data`, Responses `encrypted_content`, Google
        /// `thoughtSignature`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
        /// Format-namespaced extra fields (original structures that do not
        /// fit `text`/`signature`, e.g. the Responses `summary[]` array).
        #[serde(default, skip_serializing_if = "Extra::is_empty")]
        extra: Extra,
    },
    /// An unmodeled provider node kept in place (§ 4.3).
    #[non_exhaustive]
    Opaque {
        /// The canonical id of the format the node belongs to.
        format: String,
        /// The original JSON value, verbatim.
        value: Value,
    },
}

impl ContentBlock {
    /// A text block.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into(), cache: None, extra: Extra::new() }
    }

    /// An image block.
    #[must_use]
    pub fn image(source: ImageSource) -> Self {
        Self::Image { source, cache: None, extra: Extra::new() }
    }

    /// An image block referencing a URL.
    #[must_use]
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::image(ImageSource::url(url))
    }

    /// An inline base64 image block.
    #[must_use]
    pub fn image_base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self::image(ImageSource::base64(media_type, data))
    }

    /// An image block referencing a provider file id.
    #[must_use]
    pub fn image_file_id(id: impl Into<String>) -> Self {
        Self::image(ImageSource::file_id(id))
    }

    /// A tool call without an id.
    #[must_use]
    pub fn tool_call(name: impl Into<String>, arguments: impl Into<String>) -> Self {
        Self::ToolCall {
            id: None,
            name: name.into(),
            arguments: arguments.into(),
            cache: None,
            extra: Extra::new(),
        }
    }

    /// A tool call with a provider id.
    #[must_use]
    pub fn tool_call_with_id(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self::ToolCall {
            id: Some(id.into()),
            name: name.into(),
            arguments: arguments.into(),
            cache: None,
            extra: Extra::new(),
        }
    }

    /// A tool result.
    #[must_use]
    pub fn tool_result(tool_call_id: Option<String>, content: Vec<ToolOutputBlock>) -> Self {
        Self::ToolResult {
            tool_call_id,
            name: None,
            content,
            is_error: None,
            cache: None,
            extra: Extra::new(),
        }
    }

    /// A tool result holding a single text block.
    #[must_use]
    pub fn tool_result_text(tool_call_id: Option<String>, text: impl Into<String>) -> Self {
        Self::tool_result(tool_call_id, vec![ToolOutputBlock::text(text)])
    }

    /// A thinking block with plaintext only.
    #[must_use]
    pub fn thinking(text: impl Into<String>) -> Self {
        Self::Thinking { text: Some(text.into()), signature: None, extra: Extra::new() }
    }

    /// A thinking block with text and a replay signature.
    #[must_use]
    pub fn thinking_signed(text: impl Into<String>, signature: impl Into<String>) -> Self {
        Self::Thinking {
            text: Some(text.into()),
            signature: Some(signature.into()),
            extra: Extra::new(),
        }
    }

    /// An unmodeled provider node (§ 4.3). `format` is the canonical format
    /// id the node belongs to.
    #[must_use]
    pub fn opaque(format: impl Into<String>, value: Value) -> Self {
        Self::Opaque { format: format.into(), value }
    }

    /// Sets the cache hint. Ignored on `Thinking` and `Opaque`, which have
    /// no cache channel.
    #[must_use]
    pub fn with_cache(mut self, hint: CacheHint) -> Self {
        match &mut self {
            Self::Text { cache, .. }
            | Self::Image { cache, .. }
            | Self::ToolCall { cache, .. }
            | Self::ToolResult { cache, .. } => *cache = Some(hint),
            Self::Thinking { .. } | Self::Opaque { .. } => {}
        }
        self
    }

    /// Sets `is_error` on a `ToolResult`; ignored on other variants.
    #[must_use]
    pub fn with_is_error(mut self, error: bool) -> Self {
        if let Self::ToolResult { is_error, .. } = &mut self {
            *is_error = Some(error);
        }
        self
    }

    /// Sets `name` on a `ToolResult`; ignored on other variants.
    #[must_use]
    pub fn with_tool_name(mut self, name: impl Into<String>) -> Self {
        if let Self::ToolResult { name: n, .. } = &mut self {
            *n = Some(name.into());
        }
        self
    }

    /// Returns the block's `extra`, if the variant has one (`Opaque` does
    /// not — its `value` is its own escape hatch).
    #[must_use]
    pub fn extra(&self) -> Option<&Extra> {
        match self {
            Self::Text { extra, .. }
            | Self::Image { extra, .. }
            | Self::ToolCall { extra, .. }
            | Self::ToolResult { extra, .. }
            | Self::Thinking { extra, .. } => Some(extra),
            Self::Opaque { .. } => None,
        }
    }

    /// Mutable access to the block's `extra`, if the variant has one.
    pub fn extra_mut(&mut self) -> Option<&mut Extra> {
        match self {
            Self::Text { extra, .. }
            | Self::Image { extra, .. }
            | Self::ToolCall { extra, .. }
            | Self::ToolResult { extra, .. }
            | Self::Thinking { extra, .. } => Some(extra),
            Self::Opaque { .. } => None,
        }
    }

    /// Sets a top-level field in the block's `extra` namespace for `format`.
    /// Ignored on `Opaque`.
    #[must_use]
    pub fn with_extra(mut self, format: &str, key: impl Into<String>, value: impl Into<Value>) -> Self {
        if let Some(extra) = self.extra_mut() {
            extra.set(format, key, value);
        }
        self
    }

    /// Returns the block's cache hint, if any.
    #[must_use]
    pub fn cache(&self) -> Option<&CacheHint> {
        match self {
            Self::Text { cache, .. }
            | Self::Image { cache, .. }
            | Self::ToolCall { cache, .. }
            | Self::ToolResult { cache, .. } => cache.as_ref(),
            Self::Thinking { .. } | Self::Opaque { .. } => None,
        }
    }

    /// For a `ToolCall` block, parses `arguments` as JSON on demand.
    /// Returns `None` for other variants.
    pub fn arguments_json(&self) -> Option<serde_json::Result<Value>> {
        match self {
            Self::ToolCall { arguments, .. } => Some(serde_json::from_str(arguments)),
            _ => None,
        }
    }

    /// Human-readable kind name, used in errors.
    #[must_use]
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Text { .. } => "Text",
            Self::Image { .. } => "Image",
            Self::ToolCall { .. } => "ToolCall",
            Self::ToolResult { .. } => "ToolResult",
            Self::Thinking { .. } => "Thinking",
            Self::Opaque { .. } => "Opaque",
        }
    }
}

/// Tool-result content: a deliberately restricted union — no format can
/// express tool calls, tool results or thinking nested inside a tool result,
/// so the type rules those out by construction (§ 4.3).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolOutputBlock {
    /// Plain text.
    #[non_exhaustive]
    Text {
        /// The text.
        text: String,
        /// Optional cache breakpoint. Dropped with a cosmetic warning on
        /// every target in v1 (§ 4.8).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache: Option<CacheHint>,
        /// Format-namespaced extra fields.
        #[serde(default, skip_serializing_if = "Extra::is_empty")]
        extra: Extra,
    },
    /// An image.
    #[non_exhaustive]
    Image {
        /// Where the image comes from.
        source: ImageSource,
        /// Optional cache breakpoint (same v1 restriction as `Text`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache: Option<CacheHint>,
        /// Format-namespaced extra fields.
        #[serde(default, skip_serializing_if = "Extra::is_empty")]
        extra: Extra,
    },
    /// Unmodeled tool-result content, same semantics as
    /// [`ContentBlock::Opaque`].
    #[non_exhaustive]
    Opaque {
        /// The canonical id of the format the node belongs to.
        format: String,
        /// The original JSON value, verbatim.
        value: Value,
    },
}

impl ToolOutputBlock {
    /// A text block.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into(), cache: None, extra: Extra::new() }
    }

    /// An image block.
    #[must_use]
    pub fn image(source: ImageSource) -> Self {
        Self::Image { source, cache: None, extra: Extra::new() }
    }

    /// An unmodeled provider node.
    #[must_use]
    pub fn opaque(format: impl Into<String>, value: Value) -> Self {
        Self::Opaque { format: format.into(), value }
    }

    /// Sets the cache hint. Ignored on `Opaque`, which has no cache
    /// channel.
    #[must_use]
    pub fn with_cache(mut self, hint: CacheHint) -> Self {
        match &mut self {
            Self::Text { cache, .. } | Self::Image { cache, .. } => *cache = Some(hint),
            Self::Opaque { .. } => {}
        }
        self
    }

    /// Human-readable kind name, used in errors.
    #[must_use]
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Text { .. } => "Text",
            Self::Image { .. } => "Image",
            Self::Opaque { .. } => "Opaque",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn image_source_serde() {
        let url = ImageSource::url("https://x/y.png");
        assert_eq!(
            serde_json::to_value(&url).unwrap(),
            json!({"type": "url", "url": "https://x/y.png"})
        );
        let b64 = ImageSource::base64("image/png", "AAAA");
        assert_eq!(
            serde_json::to_value(&b64).unwrap(),
            json!({"type": "base64", "media_type": "image/png", "data": "AAAA"})
        );
        let file = ImageSource::file_id("file-1");
        assert_eq!(
            serde_json::to_value(&file).unwrap(),
            json!({"type": "file_id", "file_id": "file-1"})
        );
        for v in [url, b64, file] {
            let s = serde_json::to_string(&v).unwrap();
            assert_eq!(serde_json::from_str::<ImageSource>(&s).unwrap(), v);
        }
        assert!(serde_json::from_value::<ImageSource>(json!({"type": "nope"})).is_err());
    }

    #[test]
    fn content_block_tagged_serde() {
        let block = ContentBlock::text("hi");
        assert_eq!(serde_json::to_value(&block).unwrap(), json!({"type": "text", "text": "hi"}));

        let call = ContentBlock::tool_call_with_id("c1", "get", "{}");
        assert_eq!(
            serde_json::to_value(&call).unwrap(),
            json!({"type": "tool_call", "id": "c1", "name": "get", "arguments": "{}"})
        );

        let opaque = ContentBlock::opaque("anthropic_messages", json!({"type": "document"}));
        let v = serde_json::to_value(&opaque).unwrap();
        assert_eq!(
            v,
            json!({"type": "opaque", "format": "anthropic_messages", "value": {"type": "document"}})
        );
        assert_eq!(serde_json::from_value::<ContentBlock>(v).unwrap(), opaque);
    }

    #[test]
    fn arguments_json_parses_on_demand() {
        let ok = ContentBlock::tool_call("t", r#"{"a": 1}"#);
        assert_eq!(ok.arguments_json().unwrap().unwrap(), json!({"a": 1}));
        let bad = ContentBlock::tool_call("t", "{not json");
        assert!(bad.arguments_json().unwrap().is_err());
        assert!(ContentBlock::text("x").arguments_json().is_none());
    }

    #[test]
    fn with_cache_ignored_where_unsupported() {
        let t = ContentBlock::thinking("x").with_cache(CacheHint::new());
        assert!(t.cache().is_none());
        let txt = ContentBlock::text("x").with_cache(CacheHint::with_ttl("1h"));
        assert_eq!(txt.cache().unwrap().ttl.as_deref(), Some("1h"));
    }

    #[test]
    fn persisted_ir_with_future_fields_still_deserializes() {
        // Fields added later are always optional; unknown fields error is
        // NOT acceptable for enums with deny_unknown off (serde default
        // ignores unknowns) — assert that holds.
        let v = json!({"type": "text", "text": "hi", "some_future_field": 1});
        let block: ContentBlock = serde_json::from_value(v).unwrap();
        assert_eq!(block, ContentBlock::text("hi"));
    }
}
