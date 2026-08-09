//! Conversion warnings (§ 6).
//!
//! Every IR→format conversion returns its output plus warnings; nothing is
//! silently dropped. Each [`WarningCode`] has a **fixed severity** so
//! classifications cannot drift between formats; the mapping is asserted
//! centrally in tests.

use serde::{Deserialize, Serialize};

/// Whether a warning describes lost meaning or lost tuning.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarningSeverity {
    /// Meaning lost — model-visible behavior or contract can change
    /// (thinking dropped, unsupported image source, …). Escalated to an
    /// error by `ConvertOptions.strict` on the build side.
    Semantic,
    /// Tuning lost (cache hint dropped, unsupported sampling knob, …).
    Cosmetic,
}

/// Which conversion direction produced a warning.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConversionDirection {
    /// IR → provider request (build side). Locations point into the final
    /// target JSON.
    ToFormat,
    /// Provider JSON → IR (parse side, non-streaming or streaming).
    /// Locations point into the JSON being consumed. Parse-side warnings
    /// always report and never fail the call.
    FromFormat,
}

/// Stable, matchable warning identifiers.
///
/// The enum is `#[non_exhaustive]`: codes may be added in minor versions.
/// Severity is a fixed function of the code — see
/// [`WarningCode::severity`].
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarningCode {
    // ---- semantic, build side ----
    /// A thinking block was dropped for a cross-provider target (§ 4.4).
    ThinkingDropped,
    /// A thinking signature could not be carried (e.g. `thinking_as_text`).
    ThinkingSignatureDropped,
    /// An image source has no channel on the target (`FileId` → CC,
    /// assistant image on a format without that channel, …).
    ImageSourceUnsupported,
    /// A tool-result image was dropped (CC tool messages are text-only;
    /// Google `FunctionResponsePart` has no URL/file channel).
    ToolResultImageDropped,
    /// `ToolResult.is_error` was dropped — an error result would otherwise
    /// read as success.
    IsErrorDropped,
    /// `FunctionTool.strict` has no equivalent on the target (Google).
    StrictUnsupported,
    /// An `Opaque` block/tool was dropped because the target is not the
    /// format it belongs to.
    OpaqueDropped,
    /// `stop_sequences` (an output-control contract) was dropped.
    StopSequencesDropped,
    /// `parallel_tool_calls: false` (a serial-execution constraint) was
    /// dropped (Google).
    SerialToolCallsDropped,
    /// `Reasoning.enabled: false` cannot be expressed on the target
    /// (Google).
    ReasoningDisableUnsupported,
    /// The requested effort tier is outside the target's set.
    EffortUnsupported,
    /// Schema-less JSON mode is unsupported on the target (Anthropic).
    JsonModeUnsupported,
    /// A system/developer message was downgraded to `user`.
    RoleDowngraded,
    /// Google tool results split media from text; an interleaved sequence
    /// lost its order (§ 7.2).
    ToolResultOrderLost,
    /// Unmatched tool calls in the middle of the conversation (never
    /// repaired).
    OrphanToolCalls,
    /// `orphan_tool_calls: DropTrailing` left a preceding thinking block
    /// orphaned (§ 7.5).
    ThinkingOrphaned,
    /// An assistant message contains `ToolCall` but no `Thinking` block
    /// while the request enables thinking (§ 7.3).
    MissingThinkingWithToolCalls,
    /// `merge_consecutive_roles` skipped a merge because a signed block
    /// makes the pair a merge barrier (§ 7.5).
    MergeBlockedBySignature,
    /// A node's `extra` namespace **for the target format** had no landing
    /// spot in the output (e.g. block extras on `Request.system` when the
    /// target's system channel is a plain string) — user-addressed data was
    /// lost.
    ExtraDropped,
    /// Token counting: the count adapter dropped a field it did not
    /// generate (injected via `extra`, hooks or a dialect) — the result is
    /// no longer exact (§ 13).
    CountTokensFieldDropped,
    /// Token counting used a format decoupled from chat — the result is an
    /// approximation (§ 13).
    CountTokensApproximate,

    // ---- cosmetic, build side ----
    /// A pure sampling knob (`seed`, `top_k`, penalties) was dropped.
    SamplingParameterDropped,
    /// `metadata` (or its non-`user_id` keys on Anthropic) was dropped.
    MetadataDropped,
    /// A block/tool cache hint was dropped (no channel on the target,
    /// nested tool-output hints, …).
    CacheHintDropped,
    /// The cache breakpoint mapped, but its TTL has no equivalent (OpenAI).
    CacheTtlDropped,
    /// `Request.cache_key` was dropped (Anthropic/Google).
    CacheKeyDropped,
    /// `parallel_tool_calls` was meaningless or redundant and not emitted
    /// (no tools, `ToolChoice::None`, `Some(true)` on Google).
    ParallelToolCallsIgnored,
    /// Anthropic requires `input_schema`; an explicit empty-parameter
    /// schema was synthesized (§ 4.5).
    EmptyParametersSynthesized,
    /// Structured-output detail (`name`/`strict`) has no channel on the
    /// target (Anthropic).
    OutputFormatDetailDropped,
    /// `Reasoning.enabled` and `effort` conflicted; `effort` won (§ 4.7).
    ReasoningConflict,
    /// `include_thoughts` has no channel on the target (OpenAI CC).
    IncludeThoughtsUnsupported,
    /// A generic URL was mapped to Google `fileData.fileUri` verbatim;
    /// documented URIs are Files API ones and arbitrary URLs may be
    /// rejected upstream (§ 4.3).
    ImageUrlAsFileUri,
    /// Google tool results flatten multiple text blocks into one string
    /// (§ 7.2).
    ToolResultTextJoined,
    /// `orphan_tool_calls: DropTrailing` removed unmatched trailing tool
    /// calls (opt-in policy executed as requested).
    OrphanToolCallsDropped,
    /// `orphan_tool_calls: SynthesizeError` appended synthetic error
    /// results (opt-in policy executed as requested).
    OrphanToolCallsSynthesized,
    /// `fill_missing_thinking` inserted a placeholder thinking block.
    MissingThinkingFilled,

    // ---- parse side ----
    /// The response carried more than one choice/candidate; only the first
    /// was read (the rest remain in `raw`).
    MultipleCandidates,
    /// Recoverable malformed provider data was skipped or partially mapped.
    MalformedField,
    /// A stream event could not be attributed to a block and surfaced as
    /// `Unknown`.
    UnknownStreamEvent,
}

impl WarningCode {
    /// The fixed severity of this code (§ 4.6: asserted centrally so
    /// classifications cannot drift between formats).
    #[must_use]
    pub fn severity(&self) -> WarningSeverity {
        use WarningCode::*;
        match self {
            ThinkingDropped
            | ThinkingSignatureDropped
            | ImageSourceUnsupported
            | ToolResultImageDropped
            | IsErrorDropped
            | StrictUnsupported
            | OpaqueDropped
            | StopSequencesDropped
            | SerialToolCallsDropped
            | ReasoningDisableUnsupported
            | EffortUnsupported
            | JsonModeUnsupported
            | RoleDowngraded
            | ToolResultOrderLost
            | OrphanToolCalls
            | ThinkingOrphaned
            | MissingThinkingWithToolCalls
            | MergeBlockedBySignature
            | ExtraDropped
            | CountTokensFieldDropped
            | CountTokensApproximate
            | MultipleCandidates => WarningSeverity::Semantic,
            SamplingParameterDropped
            | MetadataDropped
            | CacheHintDropped
            | CacheTtlDropped
            | CacheKeyDropped
            | ParallelToolCallsIgnored
            | EmptyParametersSynthesized
            | OutputFormatDetailDropped
            | ReasoningConflict
            | IncludeThoughtsUnsupported
            | ImageUrlAsFileUri
            | ToolResultTextJoined
            | OrphanToolCallsDropped
            | OrphanToolCallsSynthesized
            | MissingThinkingFilled
            | MalformedField
            | UnknownStreamEvent => WarningSeverity::Cosmetic,
        }
    }
}

/// One conversion warning (§ 6).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConversionWarning {
    /// Stable, matchable code.
    pub code: WarningCode,
    /// Fixed severity of `code`.
    pub severity: WarningSeverity,
    /// JSON Pointer: build-side warnings point into the final target JSON,
    /// parse-side warnings into the JSON being consumed.
    pub location: String,
    /// The provider format involved (canonical format id).
    pub format: String,
    /// Build or parse side.
    pub direction: ConversionDirection,
    /// `true` when `extra` explicitly addressed this path (§ 6) — kept for
    /// debugging, ignored by the strict gate.
    #[serde(default)]
    pub overridden: bool,
    /// Human-readable description.
    pub message: String,
}

impl ConversionWarning {
    /// Creates a warning; severity is derived from the code.
    #[must_use]
    pub fn new(
        code: WarningCode,
        format: impl Into<String>,
        direction: ConversionDirection,
        location: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        let severity = code.severity();
        Self {
            code,
            severity,
            location: location.into(),
            format: format.into(),
            direction,
            overridden: false,
            message: message.into(),
        }
    }

    /// Build-side constructor shorthand.
    #[must_use]
    pub fn to_format(
        code: WarningCode,
        format: impl Into<String>,
        location: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self::new(code, format, ConversionDirection::ToFormat, location, message)
    }

    /// Parse-side constructor shorthand.
    #[must_use]
    pub fn from_format(
        code: WarningCode,
        format: impl Into<String>,
        location: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self::new(code, format, ConversionDirection::FromFormat, location, message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn severity_is_fixed_per_code() {
        let w = ConversionWarning::to_format(
            WarningCode::ThinkingDropped,
            "google_generate_content",
            "/contents/0",
            "thinking dropped",
        );
        assert_eq!(w.severity, WarningSeverity::Semantic);
        assert!(!w.overridden);

        let c = ConversionWarning::from_format(
            WarningCode::MalformedField,
            "openai_chat_completions",
            "/usage",
            "bad usage",
        );
        assert_eq!(c.severity, WarningSeverity::Cosmetic);
        assert_eq!(c.direction, ConversionDirection::FromFormat);
    }

    #[test]
    fn warning_serde_round_trip() {
        let w = ConversionWarning::to_format(
            WarningCode::CacheKeyDropped,
            "anthropic_messages",
            "/prompt_cache_key",
            "dropped",
        );
        let s = serde_json::to_string(&w).unwrap();
        let back: ConversionWarning = serde_json::from_str(&s).unwrap();
        assert_eq!(back, w);
    }
}
