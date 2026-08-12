//! The producer's own outcome for a block, recorded so a replay can be compared against it.
//!
//! **This is an expectation and not an authority.** A mismatch between a replay and one of these
//! means one of the two implementations is wrong; which one is an investigation, and reading the
//! oracle as ground truth would turn the one check capable of catching a uniformly wrong
//! extraction into a check that agrees with it. The producer that wrote these is the in-process
//! path that the paired benchmark screened, which makes it well-evidenced and not infallible.
//!
//! **It cannot be read while validating**, and that is enforced by the dependency graph rather
//! than by discipline. This type lives in `partial-stateless-stream`, which depends on
//! `partial-stateless-validator`; the arrow never runs the other way, so no code inside the
//! validator can name a `CommitOracle`, on any branch, even by accident. It is the same technique
//! the database-free claim rests on — make the wrong thing unnameable rather than unreached — and
//! it is worth far more here than a comment saying "do not read this".
//!
//! What is recorded is chosen so that two validators agreeing on every field have agreed on the
//! block in every sense that matters downstream: the verdict and, when it is a rejection, the
//! class; the state root; the next cache anchor; the miss set the next block is expected to want;
//! where the pair's lifecycle got to; and the two fingerprints that answer "same generation" and
//! "same way of getting there" separately.

use crate::event::BlockRef;
use alloy_primitives::B256;
use partial_stateless::{CacheAnchor, StateTargetSet};
use partial_stateless_validator::{CoordinatedFingerprint, LifecycleFingerprint};
use serde::{Deserialize, Serialize};

/// What the recording producer concluded about one block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommitOracle {
    /// Accepted, or rejected with the class it was rejected under.
    pub verdict: RecordedVerdict,
    /// Post-state root the producer computed. `None` on a rejection.
    pub state_root: Option<B256>,
    /// The cache anchor the transition produced. `None` on a rejection.
    pub next_cache_anchor: Option<CacheAnchor>,
    /// State the next block is expected to miss on, which the sidecar's miss manifest must match.
    ///
    /// `None` on a rejection. Compared as a set: two validators that agree on the block but
    /// disagree here have disagreed about what the *next* block will need, which surfaces one
    /// block later as a witness that does not cover its own execution.
    pub expected_miss: Option<StateTargetSet>,
    /// The pair's readiness classification after the event, by its stable label.
    pub readiness_state: String,
    /// Highest contiguously processed block after the event.
    ///
    /// Distinct from the block this commit is about, and the distinction is the point: a producer
    /// that skipped a block keeps applying later ones while this stays put.
    pub readiness_watermark: Option<BlockRef>,
    /// Highest block whose cache state the producer had written to durable storage.
    ///
    /// `None` for a producer that persists nothing, which is what the bounded in-memory benchmark
    /// runs as. Recorded because a restart resumes from this and not from the readiness watermark.
    pub durability_watermark: Option<u64>,
    /// The retained parent generation held after the event, if any.
    pub retained_generation: Option<BlockRef>,
    /// Are these the same cache generation.
    pub coordinated_fingerprint: CoordinatedFingerprint,
    /// Did they get there the same way.
    ///
    /// Separate from the fingerprint above on purpose: a snapshot restore reproduces the
    /// generation exactly and by construction did not reach it by applying the same blocks, so a
    /// single combined value could not express "restored correctly".
    pub lifecycle_fingerprint: LifecycleFingerprint,
}

/// The producer's verdict, with a rejection's stable class.
///
/// The class travels because two validators that reject the same block for different reasons have
/// not agreed on it. It is the coarse `class()` string the validator's own rejection types expose
/// rather than the full error, so a message change is not a corpus change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordedVerdict {
    /// The producer applied this block.
    Accepted,
    /// The producer refused it.
    Rejected {
        /// Stable coarse class: `payload`, `consensus`, `sender_recovery`, `no_accepted_parent`,
        /// or one of the transition's own.
        class: String,
    },
}

impl RecordedVerdict {
    /// Stable name for telemetry.
    pub fn label(&self) -> &str {
        match self {
            Self::Accepted => "accepted",
            Self::Rejected { class } => class,
        }
    }

    /// Whether the producer applied the block.
    pub const fn is_accepted(&self) -> bool {
        matches!(self, Self::Accepted)
    }
}
