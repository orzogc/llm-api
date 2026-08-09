//! Conversion options (§ 7.3, § 12).

/// Policy for trailing assistant tool calls without matching results —
/// e.g. an interrupted agent (§ 7.3). Orphans in the middle of the array
/// are never repaired — warning only.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OrphanToolCalls {
    /// Send as-is; the upstream error is returned.
    #[default]
    Passthrough,
    /// Remove the unmatched trailing `ToolCall` blocks (and the whole
    /// message if it becomes empty), with a warning.
    DropTrailing,
    /// Append a synthetic `is_error` tool result (content `"cancelled"`)
    /// for each orphan, keeping history well-formed.
    SynthesizeError,
}

/// Options controlling IR → format conversion (§ 6, § 7.3).
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ConvertOptions {
    /// Turns any non-overridden `Semantic` warning from the IR→request
    /// conversion into an error. Parse-side warnings always report and
    /// never fail the call.
    pub strict: bool,
    /// Additionally downgrades developer→user *within* OpenAI-family
    /// formats, for CC dialects that predate the developer role (§ 7.1).
    pub downgrade_developer: bool,
    /// Orphan tool-call policy (§ 7.3).
    pub orphan_tool_calls: OrphanToolCalls,
    /// Converts plaintext thinking into the target's thinking-text channel
    /// instead of dropping it — useful when switching between open-weight
    /// models whose chains of thought are plaintext. Destroys signatures by
    /// definition (§ 7.5).
    pub thinking_as_text: bool,
    /// Inserts a thinking block with the given text into assistant
    /// messages that carry tool calls but no thinking while the request
    /// enables thinking (§ 7.3). Only genuinely helps signature-less
    /// channels.
    pub fill_missing_thinking: Option<String>,
}

impl ConvertOptions {
    /// Default (lenient) options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets strict mode.
    #[must_use]
    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }
}
