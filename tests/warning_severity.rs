//! The single central assertion of the `WarningCode` → severity table
//! (design § 4.6 / § 15): severities are fixed per code so classifications
//! cannot drift between formats. The in-crate `severity()` match is
//! exhaustive by the compiler; this test pins each code's documented value.

use llm_api::{WarningCode, WarningSeverity};

#[test]
fn severity_table_is_pinned() {
    use WarningCode::*;
    use WarningSeverity::{Cosmetic, Semantic};

    let table = [
        // Build side, semantic: meaning lost.
        (ThinkingDropped, Semantic),
        (ThinkingSignatureDropped, Semantic),
        (ImageSourceUnsupported, Semantic),
        (ToolResultImageDropped, Semantic),
        (IsErrorDropped, Semantic),
        (StrictUnsupported, Semantic),
        (OpaqueDropped, Semantic),
        (StopSequencesDropped, Semantic),
        (SerialToolCallsDropped, Semantic),
        (ReasoningDisableUnsupported, Semantic),
        (EffortUnsupported, Semantic),
        (JsonModeUnsupported, Semantic),
        (RoleDowngraded, Semantic),
        (ToolResultOrderLost, Semantic),
        (BlockOrderLost, Semantic),
        (OrphanToolCalls, Semantic),
        (ThinkingOrphaned, Semantic),
        (MissingThinkingWithToolCalls, Semantic),
        (MergeBlockedBySignature, Semantic),
        (ExtraDropped, Semantic),
        (ItemBoundaryLost, Semantic),
        (EmptyMessageDropped, Semantic),
        (CountTokensFieldDropped, Semantic),
        (CountTokensApproximate, Semantic),
        // Build side, cosmetic: tuning lost, or an opt-in policy executed
        // as requested.
        (SamplingParameterDropped, Cosmetic),
        (MetadataDropped, Cosmetic),
        (CacheHintDropped, Cosmetic),
        (CacheTtlDropped, Cosmetic),
        (CacheKeyDropped, Cosmetic),
        (ParallelToolCallsIgnored, Cosmetic),
        (ToolChoiceIgnored, Cosmetic),
        (EmptyParametersSynthesized, Cosmetic),
        (OutputFormatDetailDropped, Cosmetic),
        (ReasoningConflict, Cosmetic),
        (IncludeThoughtsUnsupported, Cosmetic),
        (ImageUrlAsFileUri, Cosmetic),
        (ToolResultTextJoined, Cosmetic),
        (ThinkingBlocksJoined, Cosmetic),
        (OrphanToolCallsDropped, Cosmetic),
        (OrphanToolCallsSynthesized, Cosmetic),
        (MissingThinkingFilled, Cosmetic),
        // Parse side.
        (MultipleCandidates, Semantic),
        (MalformedField, Cosmetic),
        (MalformedToolCall, Semantic),
        (MalformedToolResult, Semantic),
        (UnknownStreamEvent, Cosmetic),
        (StreamOptionsDropped, Cosmetic),
        (PostTerminalStreamFailure, Cosmetic),
    ];
    for (code, expected) in table {
        assert_eq!(code.severity(), expected, "severity drifted for {code:?}");
    }
}
