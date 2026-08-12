//! Obtaining the Engine-API payload for a canonical block, and saying where it came from.
//!
//! The ExEx is handed a `RecoveredBlock` — a block its own Engine decoded, layout-checked, and
//! sender-recovered. The standalone validator's input is an `ExecutionData`, and the difference is
//! not a matter of packaging. Deriving a payload from a block hands the validator the very answers
//! its payload-layout, block-hash, and versioned-hash checks exist to question: the block hash it
//! would announce is the one this node computed, the versioned hashes are read back off the same
//! blob transactions they are meant to bind, and the requests travel as a hash rather than as
//! requests. Those checks then pass because they are comparing a derivation against itself.
//!
//! So a derived payload is not wrong, and it is not a substitute either. It is a different kind of
//! evidence, and the only sound thing to do with the difference is to record it. That is what
//! [`PayloadProvenance`] is: the same discipline schema V10 applies to admission timings, where a
//! phase nobody ran reads `null` rather than `0`, applied to inputs instead of costs. A run that
//! reports 100 accepted payloads is making a much weaker claim if 100 of them were reconstructions
//! of blocks this node had already accepted, and a corpus that cannot tell the two apart cannot be
//! read at all.

use alloy_rpc_types_engine::ExecutionData;
use reth_ethereum::EthPrimitives;
use reth_execution_access::{
    global_payload_handoff, payload_capture_enabled, HandoffEntry, HandoffStats, MissReason,
};
use reth_primitives_traits::{BlockTy, RecoveredBlock, SealedBlock};
use std::sync::atomic::{AtomicU64, Ordering};

/// One committed block's payload, with the provenance that says how much it proves.
#[derive(Debug)]
pub struct TappedPayload {
    /// The payload, absent only when this run obtained none at all.
    pub payload: Option<ExecutionData>,
    /// Where the payload above came from.
    pub provenance: PayloadProvenance,
    /// Why the Engine's own payload was unavailable, as far as the handoff can attest.
    ///
    /// `None` on a hit, and also on a run that never asked — the two are separated by
    /// [`PayloadProvenance`], not by this field.
    pub miss_reason: Option<MissReason>,
    /// Bytes the producer measured for the payload it published; `None` unless witnessed.
    ///
    /// Deliberately not filled in for a reconstruction. The producer's figure is a measurement of
    /// what it handed over; a number this side computed about a payload it derived itself would be
    /// a different quantity wearing the same name.
    pub approx_bytes: Option<usize>,
    /// How long the artifact waited in the handoff before this take; `None` unless witnessed.
    pub residence_us: Option<u64>,
    /// Handoff telemetry as of this take, absent when capture is off.
    pub stats: Option<HandoffStats>,
    /// Payloads witnessed so far in this process.
    pub witnessed_total: u64,
    /// Payloads derived so far in this process.
    pub reconstructed_total: u64,
}

/// Where a recorded payload came from, and therefore what a check against it is worth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PayloadProvenance {
    /// The payload a consensus client sent, taken from the Engine that validated it.
    ///
    /// The only provenance under which the payload-layout, block-hash, and versioned-hash checks
    /// are checks rather than tautologies.
    Witnessed,
    /// Derived from the block, because this run was capturing and the Engine's copy was not there.
    ///
    /// Structural for two cases rather than exceptional in either: blocks replayed from the ExEx
    /// WAL after a restart were validated by a process that no longer exists, and backfilled
    /// blocks were never handed to this node's Engine as payloads at all. Both leave the handoff
    /// with nothing to give, and `miss_reason` names which.
    Reconstructed,
    /// No payload was obtained, and none was derived.
    ///
    /// What a run that is not capturing reports. Kept distinct from `Reconstructed` for exactly
    /// the reason `None` is kept distinct from `Some(0)` in the admission timings: "nobody asked"
    /// and "asked and had to fall back" are different facts about the run.
    Absent,
}

impl PayloadProvenance {
    /// Stable name for telemetry.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Witnessed => "witnessed",
            Self::Reconstructed => "reconstructed",
            Self::Absent => "absent",
        }
    }

    /// Whether the standalone validator's admission checks would be checking anything.
    ///
    /// False on a reconstruction, which is not a defect of the reconstruction: a vacuous pass is
    /// reported as provenance rather than counted as coverage.
    pub const fn is_load_bearing(&self) -> bool {
        matches!(self, Self::Witnessed)
    }
}

/// Whether payload capture is armed for this process.
///
/// Re-exported through this module so the ExEx names one thing rather than reaching into the
/// handoff crate for the gate and this module for the meaning.
pub fn tap_enabled() -> bool {
    payload_capture_enabled()
}

/// Takes the Engine's payload for a committed block, deriving one from the block if it is absent.
///
/// Call this **before** anything else the block triggers. The handoff records how long each
/// artifact waited, and a take deferred behind sidecar construction would report a residence that
/// no real consumer would ever see.
pub fn tap_payload(block: &RecoveredBlock<BlockTy<EthPrimitives>>) -> TappedPayload {
    let Some(handoff) = global_payload_handoff() else {
        return TappedPayload {
            payload: None,
            provenance: PayloadProvenance::Absent,
            miss_reason: None,
            approx_bytes: None,
            residence_us: None,
            stats: None,
            witnessed_total: 0,
            reconstructed_total: 0,
        }
    };

    let outcome = handoff.take_outcome(&block.hash());
    let miss_reason = outcome.miss_reason();
    let stats = handoff.stats();
    let totals = TapTotals::get();

    // A downcast that fails means this process is configured for a payload type the Engine does
    // not produce, which is a miss with a cause the handoff cannot name — so it is reported as one
    // rather than as a hit that yields nothing.
    if let Some(artifact) = outcome.artifact() &&
        let Some(payload) = artifact.payload::<ExecutionData>()
    {
        let witnessed_total = totals.witnessed.fetch_add(1, Ordering::Relaxed) + 1;
        return TappedPayload {
            // Cloned out of the `Arc` because a frame owns its bytes. The handoff has already
            // released its reference, so this is the last copy rather than a second one.
            payload: Some(ExecutionData::clone(&payload)),
            provenance: PayloadProvenance::Witnessed,
            miss_reason: None,
            approx_bytes: Some(artifact.approx_bytes),
            residence_us: Some(artifact.residence().as_micros() as u64),
            stats: Some(stats),
            witnessed_total,
            reconstructed_total: totals.reconstructed.load(Ordering::Relaxed),
        }
    }

    TappedPayload {
        payload: Some(reconstruct(block)),
        provenance: PayloadProvenance::Reconstructed,
        miss_reason,
        approx_bytes: None,
        residence_us: None,
        stats: Some(stats),
        witnessed_total: totals.witnessed.load(Ordering::Relaxed),
        reconstructed_total: totals.reconstructed.fetch_add(1, Ordering::Relaxed) + 1,
    }
}

/// Derives an `ExecutionData` from a block this node already accepted.
///
/// `from_block_unchecked` rather than a hashing conversion, and the name is accurate about what is
/// unchecked: the announced block hash is supplied rather than computed, so it is this node's own
/// answer being handed back. Everything a derived payload cannot carry is derived the same way —
/// the versioned hashes come from the block's own blob transactions and the requests travel as the
/// header's `requests_hash`.
///
/// The clone is unavoidable and is charged to the right path: `SealedBlock` keeps its header and
/// body apart, so there is no `&Block` to borrow, and this runs only where the Engine's own copy
/// was missing — a WAL replay or a backfill, not a steady-state block.
fn reconstruct(block: &RecoveredBlock<BlockTy<EthPrimitives>>) -> ExecutionData {
    let sealed: SealedBlock<BlockTy<EthPrimitives>> = block.clone_sealed_block();
    let block_hash = sealed.hash();
    ExecutionData::from_block_unchecked(block_hash, &sealed.into_block())
}

/// Cumulative provenance counts, so a run summary can be read without grepping every block.
#[derive(Debug, Default)]
struct TapTotals {
    witnessed: AtomicU64,
    reconstructed: AtomicU64,
}

impl TapTotals {
    fn get() -> &'static Self {
        static TOTALS: std::sync::OnceLock<TapTotals> = std::sync::OnceLock::new();
        TOTALS.get_or_init(Self::default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The counts a frame writer reports have to survive the round trip through `as_str`, because
    /// they are compared across processes rather than within one.
    #[test]
    fn provenance_names_are_stable_and_distinct() {
        let names = [
            PayloadProvenance::Witnessed.as_str(),
            PayloadProvenance::Reconstructed.as_str(),
            PayloadProvenance::Absent.as_str(),
        ];
        assert_eq!(names, ["witnessed", "reconstructed", "absent"]);
    }

    /// Only a witnessed payload makes the admission checks mean anything. A reconstruction is
    /// recorded, not credited.
    #[test]
    fn only_a_witnessed_payload_is_load_bearing() {
        assert!(PayloadProvenance::Witnessed.is_load_bearing());
        assert!(!PayloadProvenance::Reconstructed.is_load_bearing());
        assert!(!PayloadProvenance::Absent.is_load_bearing());
    }

    /// A process that was never asked to capture reports absence rather than fabricating a
    /// payload, so a corpus recorded by accident cannot look like one recorded on purpose.
    #[test]
    fn a_run_that_is_not_capturing_obtains_nothing() {
        assert!(!tap_enabled());
    }
}
