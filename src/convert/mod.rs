//! Conversion infrastructure shared by all formats: warnings, options,
//! hooks and the strict gate.

mod hooks;
mod options;
mod warnings;

pub use hooks::{MessageHook, RequestHook, RequestHooks};
pub use options::{ConvertOptions, OrphanToolCalls};
pub use warnings::{ConversionDirection, ConversionWarning, WarningCode, WarningSeverity};

use crate::error::ConversionError;

/// The § 6 strict gate: when `strict` is on, any non-overridden `Semantic`
/// **build-side** warning fails the conversion. Parse-side warnings never
/// fail a call.
pub fn strict_gate(strict: bool, warnings: &[ConversionWarning]) -> Result<(), ConversionError> {
    if !strict {
        return Ok(());
    }
    let escalated: Vec<ConversionWarning> = warnings
        .iter()
        .filter(|w| {
            w.severity == WarningSeverity::Semantic
                && !w.overridden
                && w.direction == ConversionDirection::ToFormat
        })
        .cloned()
        .collect();
    if escalated.is_empty() {
        Ok(())
    } else {
        Err(ConversionError::Strict {
            warnings: escalated,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_gate_escalates_only_semantic_unoverridden_build_warnings() {
        let semantic = ConversionWarning::to_format(
            WarningCode::ThinkingDropped,
            "f",
            "/messages/0",
            "dropped",
        );
        let cosmetic =
            ConversionWarning::to_format(WarningCode::CacheKeyDropped, "f", "/x", "dropped");
        let mut overridden = semantic.clone();
        overridden.overridden = true;
        let parse_side = ConversionWarning::from_format(
            WarningCode::MultipleCandidates,
            "f",
            "/choices",
            "skipped",
        );

        assert!(strict_gate(false, std::slice::from_ref(&semantic)).is_ok());
        assert!(strict_gate(true, &[cosmetic, overridden, parse_side]).is_ok());
        let err = strict_gate(true, &[semantic]).unwrap_err();
        assert!(matches!(err, ConversionError::Strict { warnings, .. } if warnings.len() == 1));
    }
}
