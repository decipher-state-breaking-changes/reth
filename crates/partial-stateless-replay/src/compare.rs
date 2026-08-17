//! Checking a replay's own result against the producer's recorded expectation.
//!
//! **The oracle is an expectation, not an authority.** A mismatch means one of the two
//! implementations is wrong, and which one is an investigation. This module therefore reports
//! disagreements rather than judging them: every field that differs is named with both values, and
//! nothing here decides that the recording was right.
//!
//! The fields are not equally load-bearing and the report should not pretend they are. On a
//! builder-recorded stream the sidecar carries the producer's miss claim, and the replay checks
//! its own derivation against that claim inside the transition — so an `expected_miss` agreement
//! here is a consistency check rather than an independent one. The fields with teeth are the
//! fingerprints: `trie_cache_root` commits retained-path membership, appears in no sidecar, and is
//! computed by the replay entirely on its own.

use partial_stateless_stream::{BlockRef, CommitOracle, RecordedVerdict};
use partial_stateless_validator::{CoordinatedPair, SidecarValidationOutcome};

/// One field on which a replay and the recording disagreed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Disagreement {
    /// Field name, stable enough to grep a run log for.
    pub field: &'static str,
    /// What the recording said.
    pub recorded: String,
    /// What this replay computed.
    pub replayed: String,
}

impl std::fmt::Display for Disagreement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: recorded {}, replayed {}", self.field, self.recorded, self.replayed)
    }
}

/// Compares one replayed commit against its recorded oracle.
///
/// Returns every disagreement rather than the first, because a single differing field and a
/// wholesale divergence demand different investigations and stopping at the first would hide the
/// difference between them.
pub fn compare_accepted(
    oracle: &CommitOracle,
    outcome: &SidecarValidationOutcome,
    pair: &CoordinatedPair,
) -> Vec<Disagreement> {
    let mut out = Vec::new();

    if !oracle.verdict.is_accepted() {
        out.push(Disagreement {
            field: "verdict",
            recorded: oracle.verdict.label().to_string(),
            replayed: "accepted".to_string(),
        });
    }
    compare_option("state_root", oracle.state_root, Some(outcome.state_root), &mut out);
    compare_option(
        "next_cache_anchor",
        oracle.next_cache_anchor,
        Some(outcome.next_cache_anchor),
        &mut out,
    );
    if let Some(expected_miss) = &oracle.expected_miss &&
        expected_miss != &outcome.expected_miss
    {
        out.push(Disagreement {
            field: "expected_miss",
            recorded: format!(
                "{} accounts / {} storage / {} codes",
                expected_miss.accounts.len(),
                expected_miss.storage.len(),
                expected_miss.code_hashes.len()
            ),
            replayed: format!(
                "{} accounts / {} storage / {} codes",
                outcome.expected_miss.accounts.len(),
                outcome.expected_miss.storage.len(),
                outcome.expected_miss.code_hashes.len()
            ),
        });
    }

    let fingerprint = pair.fingerprint();
    if oracle.coordinated_fingerprint != fingerprint {
        out.push(Disagreement {
            field: "coordinated_fingerprint",
            recorded: format!("{:?}", oracle.coordinated_fingerprint),
            replayed: format!("{fingerprint:?}"),
        });
    }

    // The lifecycle fingerprint is compared on its retained generation only. A replayed pair's
    // accepted head is its own — it admitted the block itself — while the recording's came from a
    // node whose Engine had already accepted it, so requiring the whole value to match would fail
    // on a difference that is the point of the exercise rather than a defect.
    let lifecycle = pair.lifecycle_fingerprint();
    if oracle.lifecycle_fingerprint.retained_generation != lifecycle.retained_generation {
        out.push(Disagreement {
            field: "retained_generation",
            recorded: format!("{:?}", oracle.lifecycle_fingerprint.retained_generation),
            replayed: format!("{:?}", lifecycle.retained_generation),
        });
    }

    out
}

/// What comparing one commit's readiness watermarks established.
#[derive(Debug)]
pub enum WatermarkComparison {
    /// Both sides named a watermark and they agree.
    Agreed,
    /// The producer recorded no watermark, so there is nothing to disagree with.
    ///
    /// Not a mismatch, measured: a producer that warmed from live blocks has no contiguous
    /// acknowledgeable run when its stream opens, while a consumer restored from the checkpoint
    /// starts its watermark *at* the checkpoint by that restore's own contract. On the recorded
    /// reorg-recovery corpus this class covered the first 72 of 314 commits and every one of the
    /// remaining 242 agreed — the absence is a fact about the producer's history, not about the
    /// chain.
    Unrecorded,
    /// Both sides named a watermark and they differ — the class that would be a real finding.
    Mismatch(Disagreement),
}

/// Compares the recorded readiness watermark against this replay's own, counter-first.
///
/// Kept apart from [`compare_accepted`]'s disagreement set on purpose. The watermark is producer
/// state whose standalone counterpart had never been compared until this comparison was added —
/// both sides record it, nothing read it — so a mismatch is counted and sampled rather than
/// failing `agreed`, until
/// full-corpus runs show whether the both-recorded equality actually holds. Promotion into the
/// disagreement set is those runs' decision. Its sibling `durability_watermark` is *not*
/// comparable at all: the consumer persists nothing, the value regresses legitimately on a pure
/// revert, and a bare block number cannot name its branch — it stays telemetry.
pub fn compare_readiness_watermark(
    oracle: &CommitOracle,
    pair: &CoordinatedPair,
) -> WatermarkComparison {
    let Some(recorded) = oracle.readiness_watermark else { return WatermarkComparison::Unrecorded };
    let replayed =
        pair.readiness.acknowledgeable_height().map(|(number, hash)| BlockRef { number, hash });
    if replayed == Some(recorded) {
        return WatermarkComparison::Agreed
    }
    WatermarkComparison::Mismatch(Disagreement {
        field: "readiness_watermark",
        recorded: format!("{recorded:?}"),
        replayed: format!("{replayed:?}"),
    })
}

/// Compares a replay's rejection against a recorded one, by class and never by message.
///
/// The class is what two independent validators must agree on. The message is free to gain detail
/// without breaking a corpus, which is exactly why the comparison does not look at it.
pub fn compare_rejected(oracle: &CommitOracle, class: &str) -> Vec<Disagreement> {
    match &oracle.verdict {
        RecordedVerdict::Rejected { class: recorded } if recorded == class => Vec::new(),
        verdict => vec![Disagreement {
            field: "verdict",
            recorded: verdict.label().to_string(),
            replayed: class.to_string(),
        }],
    }
}

fn compare_option<T: PartialEq + std::fmt::Debug>(
    field: &'static str,
    recorded: Option<T>,
    replayed: Option<T>,
    out: &mut Vec<Disagreement>,
) {
    if recorded != replayed {
        out.push(Disagreement {
            field,
            recorded: format!("{recorded:?}"),
            replayed: format!("{replayed:?}"),
        });
    }
}

/// Renders a block reference the way the run log does.
pub fn block_label(block: BlockRef) -> String {
    format!("{} ({:?})", block.number, block.hash)
}
