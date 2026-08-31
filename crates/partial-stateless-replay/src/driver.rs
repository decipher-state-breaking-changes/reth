//! Restoring a pair from a recorded checkpoint and running the recorded commits through it.
//!
//! The sequence per commit is the standalone path in full, from bytes: decode the Engine payload,
//! admit it against the pair's *own* accepted parent, and only then run the transition. Nothing is
//! taken from the frame that a live validator would not have. In particular the parent header is
//! read from the pair rather than from the commit — the rule the standalone path is built on, and
//! the one a replay is most tempted to break, because the frame is right there and it is correct.

use crate::{
    compare::{
        block_label, compare_accepted, compare_readiness_watermark, compare_rejected, Disagreement,
        WatermarkComparison,
    },
    mutate::{Mutation, TransitionMutation},
    reorg::{apply_reorg, warn_inapplicable, ReorgOutcome, VerifiedHistory},
    spool::SpoolIter,
};
use alloy_rlp::Decodable;
use partial_stateless::{
    restore_snapshot, CacheConfig, PartialStatelessSidecar, TrustedCheckpoint,
};
use partial_stateless_stream::{
    BlockRef, Checkpoint, CommitInput, CommitOracle, FrameLimits, Manifest, SnapshotChunk,
    StreamEvent, DEFAULT_MAX_SNAPSHOT_BYTES,
};
use partial_stateless_validator::{
    admit_block, block_context,
    timings::{AdmissionTimings, ValidationPhaseTimings},
    verify_and_apply_sidecar, AdmissionError, BlockAdmission, CoordinatedPair, PayloadProvenance,
    SidecarReexecLimits, TrieCacheDisposition, UntrustedAdmission, ValidatorRules,
    POST_EXECUTION_REJECTION,
};
use reth_chainspec::{ChainSpec, MAINNET};
use reth_ethereum_consensus::EthBeaconConsensus;
use reth_evm_ethereum::EthEvmConfig;
use reth_primitives_traits::{Header, SealedHeader};
use std::{path::Path, sync::Arc, time::Instant};
use tracing::{error, info, warn};

/// How much of a corpus to replay, and what to do beyond checking it.
#[derive(Debug, Clone)]
pub struct ReplayOptions {
    /// Stop after this many commits. `None` replays the whole corpus.
    pub limit: Option<usize>,
    /// Derive negative frames from every witnessed commit and check the class each must produce.
    ///
    /// On by default, because a replay of a mainnet corpus without it proves the accept path only
    /// and reads as though it proved more.
    pub mutations: bool,
    /// Drive this many recorded commits through a transition-level negative frame as well.
    ///
    /// `None` — the default — is off, and deliberately so. A transition mutation executes a
    /// second, deliberately invalid block against the full EVM, so it costs about what the honest
    /// commit costs; leaving it on would put that cost inside the very measurement a cohort run
    /// exists to take. It is a coverage switch for an offline gate, not a profile.
    ///
    /// Bounded rather than boolean for the same reason: the rule it exercises is the same rule on
    /// every block, so the fifth block proves what the five-hundredth would.
    pub mutations_transition: Option<usize>,
    /// Bounds on frame decoding.
    pub frame_limits: FrameLimits,
    /// Bounds on sidecar witness decoding.
    pub reexec_limits: SidecarReexecLimits,
    /// Install the checkpoint at this sequence rather than comparing and skipping it.
    ///
    /// A recovery checkpoint is normally cross-checked against the generation the replay already
    /// recovered to and then skimmed, which proves the two agree but not that the snapshot behind
    /// it would restore anything. This forces the restore, so a run can show that the producer's
    /// re-checkpoint is what a consumer holding no retained generation would actually recover
    /// from — the claim skimming cannot make.
    pub force_restore_at: Option<u64>,
    /// Largest commit window a restore will replay from the spool rather than degrade to a reset.
    ///
    /// Defaults to `MAX_REWIND_FRAMES`, the follower's own bound. It is a field rather than only a
    /// constant so a test can reach the degradation: the branch that matters is the one where a
    /// window is *refused*, and building four thousand frames to reach it would test the
    /// arithmetic rather than the classification.
    pub max_rewind_frames: u64,
}

impl Default for ReplayOptions {
    fn default() -> Self {
        Self {
            limit: None,
            mutations: true,
            mutations_transition: None,
            frame_limits: FrameLimits::default(),
            reexec_limits: SidecarReexecLimits::default(),
            force_restore_at: None,
            max_rewind_frames: MAX_REWIND_FRAMES,
        }
    }
}

/// What one replay found.
#[derive(Debug, Default)]
pub struct ReplayReport {
    /// Commits replayed.
    pub commits: u64,
    /// Commits whose payload was the one a consensus client sent.
    pub witnessed: u64,
    /// Commits whose payload was derived from a block by the producer.
    pub reconstructed: u64,
    /// Commits carrying no payload, which admission cannot run on.
    pub absent: u64,
    /// Every field on which a replay and the recording disagreed, with the block it was on.
    pub disagreements: Vec<(BlockRef, Disagreement)>,
    /// Blocks the replay refused that the recording accepted, or that failed to decode.
    pub failures: Vec<String>,
    /// Negative frames derived and checked.
    pub mutations_checked: u64,
    /// Negative frames that produced the wrong class, or none at all.
    pub mutation_failures: Vec<String>,
    /// Transition-level negative frames driven through a full execution.
    pub transition_mutations_checked: u64,
    /// What those cost, in microseconds.
    ///
    /// Reported apart from `transition_us` and never added to it. The honest transition time is
    /// the number every latency population is built from, and a mutation is not a validation of
    /// anything the corpus contains.
    pub transition_mutation_us: u64,
    /// Total admission wall time across every commit, in microseconds.
    pub admission_us: u64,
    /// Total transition wall time across every commit, in microseconds.
    pub transition_us: u64,
    /// Total standalone-validation wall time across every attempt, in microseconds.
    ///
    /// The primary standalone boundary: frame bytes already in memory through the committed
    /// verdict.
    /// Accumulated for every attempt, including rejected and faulted ones, whose entries record
    /// the cost of the attempt up to where it stopped.
    pub standalone_validation_us: u64,
    /// Attempts whose disjoint phase sum exceeded their own wall boundary.
    ///
    /// Timer truncation makes the parts read slightly *smaller* than the whole, so the sum
    /// exceeding the wall is a measurement anomaly, counted rather than clamped away silently —
    /// `unattributed_validation_us` saturates to zero on such an entry.
    pub timing_anomalies: u64,
    /// Commits where both sides named a readiness watermark and they agreed.
    pub watermarks_agreed: u64,
    /// Commits whose producer recorded no readiness watermark, so nothing was comparable.
    ///
    /// Expected on a producer that warmed from live blocks: it has no contiguous acknowledgeable
    /// run when its stream opens, while a checkpoint-restored consumer starts its watermark at
    /// the checkpoint by that restore's own contract.
    pub watermarks_unrecorded: u64,
    /// Commits whose recorded readiness watermark differed from this replay's own.
    ///
    /// A counter and not a disagreement, deliberately: both sides record the watermark and nothing
    /// compared them until this counter existed, so whether the equality holds is what corpus runs
    /// establish first. Promotion into the disagreement set waits on those runs coming back
    /// all-zero.
    pub watermark_mismatches: u64,
    /// The first few mismatches verbatim, so a nonzero counter is investigable from the record.
    pub watermark_mismatch_samples: Vec<String>,
    /// Per-block timings, in the order they were replayed.
    ///
    /// Kept per block rather than only as totals because the A/B this corpus exists to enable is a
    /// *paired* comparison: the same block replayed by two builds is the one comparison with no
    /// workload variance in it at all, and a total would throw that pairing away.
    pub blocks: Vec<BlockTiming>,
    /// Whether the corpus ended with an `End` frame.
    pub closed: bool,
    /// The fault that stopped the replay, when one did. Everything after it was skipped.
    pub terminal: Option<String>,
    /// Commit frames not replayed because a fault preceded them.
    ///
    /// Counted rather than reported one by one: before this existed, every commit after a fault
    /// generated its own `BlockSkipped` failure string, and a single fault read as a cascade.
    pub skipped_after_fault: u64,
    /// Commits re-read from the spool after a restore landed on the block recovery asked for.
    ///
    /// These are verified against the restored pair, and they are verified a second time: this
    /// driver had already applied them on the generation the restore discarded. They are counted
    /// here rather than netted out of `commits`, so a reader can tell a corpus that carried a
    /// late recovery checkpoint from one that did not.
    pub rewind_replayed_commits: u64,
    /// Windows a restore refused to replay because they exceeded `MAX_REWIND_FRAMES`.
    ///
    /// Not a failure: the restore degrades to the explicit reset it would have been before
    /// windows existed. It is reported because that degradation changes what the run claims.
    pub rewind_windows_refused: u64,
    /// What kind of event stopped the replay, when one did.
    pub terminal_kind: Option<&'static str>,
    /// Recorded reorgs this replay undid against the retained generation.
    pub reorgs_applied: u64,
    /// Recorded reverts this replay undid. A subset of the same mechanism, counted apart because
    /// a revert has no branch to follow and so proves a different half of the lifecycle.
    pub reverts_applied: u64,
    /// Reorgs that named this consumer's own branch but were past what it could undo.
    ///
    /// Not a disagreement and not a failure: a depth-2 reorg is the chain behaving normally and
    /// the pair behaving correctly. It costs continuity until a checkpoint restores it, which is
    /// what [`continuous`](Self::continuous) reports.
    pub reorgs_inapplicable: u64,
    /// Commit frames not replayed because the driver was waiting to be re-bootstrapped.
    pub skipped_awaiting_resync: u64,
    /// Checkpoints the producer published after a reorg this driver had already applied itself,
    /// verified against the pair's own state and then skipped rather than installed.
    pub checkpoints_skimmed: u64,
    /// Late recovery checkpoints that disagreed with this consumer's own verified history.
    ///
    /// Recorded and *not* installed: the consumer verified every block past the ancestor itself,
    /// and its own chain outranks a late cross-check. The disagreement still fails the run's
    /// agreement claim; this counter is what makes the no-install path visible beside it.
    pub late_skim_mismatches: u64,
    /// Recovery expectations still open when the stream ended: a reorg/revert announced a
    /// checkpoint that never arrived. Legal under `PS_STREAM_REORG_CHECKPOINT=never`; under
    /// `always` a clean end with one of these is a gate failure.
    pub recovery_checkpoints_pending_at_end: u64,
    /// Winning branches the producer announced and did not deliver in full, with nothing valid
    /// taking their place. A branch replaced by a later reorg is counted as superseded instead.
    pub winning_branch_incomplete: u64,
    /// Producer restarts crossed: each is a second manifest, and everything above one
    /// rebootstrapped rather than continued.
    ///
    /// Boundaries, not epochs — a corpus holding two epochs crossed one. Named for what it counts
    /// because "epochs == 1" on a two-epoch corpus is the kind of off-by-one a reader inherits.
    pub epoch_transitions: u64,
    /// Winning branches a later announcement withdrew before the delivery reached their tip.
    ///
    /// Diagnostic only. The chain moving twice in quick succession is ordinary, and the interval
    /// between the two is not an unverified gap — those blocks never became canonical.
    pub winning_branches_superseded: u64,
    /// Every re-bootstrap this replay performed, and whether it left a hole.
    pub resyncs: Vec<ResyncRecord>,
    /// The highest block this replay verified for itself.
    pub last_verified: Option<u64>,
}

/// One re-bootstrap of the pair from a checkpoint that arrived mid-corpus.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ResyncRecord {
    /// Sequence of the checkpoint the pair was rebuilt from.
    pub at_sequence: u64,
    /// The block the checkpoint restored to.
    pub block: u64,
    /// Whether the checkpoint landed on the exact block recovery had asked for.
    ///
    /// False makes the recovery an explicit checkpoint reset: the pair is sound again, but the
    /// interval in [`unverified`](Self::unverified) was never validated by anything, and
    /// reporting that as continuous recovery would be the one claim this format exists to
    /// prevent.
    pub continuous: bool,
    /// The canonical interval this recovery skipped, when it was not continuous.
    pub unverified: Option<(u64, u64)>,
    /// Commit frames observed between the discontinuity and the checkpoint.
    pub commits_skipped: u64,
}

/// Why a replay can go no further on this pair. Every variant names the block it stopped on.
///
/// This is the transaction boundary the ExEx never needed: its process fail-stops on the first
/// error, so nothing ever observed the readiness tracker stranded in `Applying` after a
/// post-execution rejection. A standalone driver outlives the rejection, so the boundary is made
/// explicit here — the pair is moved to terminal `Blocked` before anything can observe it.
#[derive(Debug, Clone)]
pub enum ReplayFault {
    /// A failure after admission, inside the transition. The flat and trie caches are preserved
    /// at the parent generation (the transition's own rollback contract); readiness is moved to
    /// terminal `Blocked` explicitly rather than left in transient `Applying`.
    TransitionFailed {
        /// The block the transition failed on.
        block: BlockRef,
        /// What the transition objected to.
        detail: String,
    },
    /// The restored pair itself refused a block the recording accepted. The refusal latched the
    /// tracker `Blocked`; the pair was not touched past that.
    ReadinessRefused {
        /// The block the pair refused.
        block: BlockRef,
        /// What readiness objected to.
        reason: String,
    },
}

impl std::fmt::Display for ReplayFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TransitionFailed { block, detail } => {
                write!(f, "{}: the transition failed: {detail}", block_label(*block))
            }
            Self::ReadinessRefused { block, reason } => {
                write!(f, "{}: the restored pair refused the block: {reason}", block_label(*block))
            }
        }
    }
}

/// What one commit's replay did to the pair.
pub(crate) enum CommitOutcome {
    /// The commit ran to a verdict and was compared against the oracle.
    Compared,
    /// The commit could not run and touched no validator state; the next commit may proceed.
    Rejected,
    /// The pair can go no further; the driver must stop replaying commits.
    Fault(ReplayFault),
}

/// What one replayed attempt cost. One entry per commit frame the driver attempted, in order —
/// rejected and faulted attempts included, carrying the phases that ran and `null` for the rest.
///
/// The timing groups follow one discipline: `phases` holds only mutually exclusive leaves whose
/// sum is meaningful, `derived` holds aggregates reconstructed from them (never summed with
/// them), and `details` is the validator core's own instrumentation verbatim — a superset kept
/// for reference, never for addition.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BlockTiming {
    /// Height.
    pub number: u64,
    /// The commit frame's sequence, which is the key an aggregator must use: a reorg legitimately
    /// repeats a height, and those re-verdicts are abandoned-branch work, not double counting.
    pub sequence: u64,
    /// What the attempt produced: `accepted`, `disagreed`, `rejected`, or `fault`.
    pub verdict: &'static str,
    /// Decode, payload layout, block hash, sender recovery, and pre-execution consensus.
    /// `null` when admission never completed.
    pub admission_us: Option<u64>,
    /// The cache transition wall clock: witness materialization, execution, root, retention,
    /// anchor. `null` when the transition never ran.
    pub transition_us: Option<u64>,
    /// The primary standalone boundary: frame bytes already in memory through the committed
    /// verdict, as one wall measurement — never a sum of parts. Excludes the oracle comparison and
    /// the mutation checks, which are harness work. On a rejected or faulted attempt this is the
    /// cost up to where the attempt stopped.
    pub standalone_validation_us: u64,
    /// Statting and reading the frame file into memory. Transport cost, outside the primary.
    pub delivery_us: Option<u64>,
    /// Comparing the verdict against the producer's recorded oracle. Harness work, outside the
    /// primary: a standalone validator in production has no oracle to consult.
    pub oracle_compare_us: Option<u64>,
    /// Deriving and checking negative payloads. Harness work, outside the primary.
    pub mutation_check_us: Option<u64>,
    /// `standalone_validation_us` minus the sum of `phases`. Saturates at zero; an attempt whose
    /// sum exceeded its wall increments the report's `timing_anomalies` instead.
    pub unattributed_validation_us: u64,
    /// Mutually exclusive leaf phases. These are the only fields it is valid to add together.
    pub phases: PhaseLeaves,
    /// Aggregates derived from the leaves, kept so older metrics stay reconstructible.
    pub derived: DerivedTimings,
    /// The validator core's own `ValidationPhaseTimings`, verbatim, completed with the admission
    /// and sidecar-decode values the driver measured — the same completion the paired harness
    /// performs. `null` when the transition never ran. A superset of `phases`: reference only.
    pub details: Option<Box<ValidationPhaseTimings>>,
}

/// The disjoint leaf phases of one standalone validation, in execution order.
///
/// Every field is `null` when its phase did not run — never zero, which would claim the phase was
/// free. The sum of the present fields is comparable against `standalone_validation_us`; nothing
/// here overlaps anything else, which is the property the paired schema's flat field list did not
/// have and the reason this grouping exists.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct PhaseLeaves {
    /// Decoding the frame envelope and body once its bytes were in memory.
    pub frame_decode_us: Option<u64>,
    /// Decoding the Engine-API payload JSON out of the commit body.
    pub input_decode_us: Option<u64>,
    /// Decoding the sidecar with bincode. The paired schema calls this `deserialize_us`.
    pub sidecar_decode_us: Option<u64>,
    /// Payload layout and block-hash validation.
    pub payload_validation_us: Option<u64>,
    /// Recovering every transaction sender from its signature.
    pub sender_recovery_us: Option<u64>,
    /// Header, pre-execution, and against-parent consensus validation.
    pub pre_execution_consensus_us: Option<u64>,
    pub context_check_us: Option<u64>,
    pub witness_self_consistency_us: Option<u64>,
    pub materialize_us: Option<u64>,
    pub provider_setup_us: Option<u64>,
    /// The executor call, excluding benchmark-only access capture.
    pub evm_us: Option<u64>,
    /// Benchmark-only accessed-state capture. A leaf beside `evm_us`: the two sum to the
    /// executor-call wall clock.
    pub accessed_state_capture_us: Option<u64>,
    pub post_execution_consensus_us: Option<u64>,
    pub hash_post_state_us: Option<u64>,
    pub trie_clone_us: Option<u64>,
    pub state_root_us: Option<u64>,
    pub root_completeness_us: Option<u64>,
    pub miss_policy_check_us: Option<u64>,
    pub cache_update_us: Option<u64>,
    pub trie_retention_us: Option<u64>,
    pub next_cache_anchor_us: Option<u64>,
    pub trie_commit_us: Option<u64>,
    /// Committing the transition into the coordinated pair and recording the block in this
    /// consumer's own verified history — the close of the primary boundary.
    pub pair_commit_us: Option<u64>,
    /// Dropping the undo generations this consumer can no longer reach.
    ///
    /// A leaf of its own rather than part of `pair_commit_us`: reclaiming memory is not committing
    /// the pair, and folding it in would silently change what an already-published metric means.
    /// The cost is one `BlockCacheUndo` destructor per block, so it is expected to be small at the
    /// median and is measured because it need not be at the tail.
    pub undo_prune_us: Option<u64>,
}

impl PhaseLeaves {
    /// The sum of every phase that ran. Valid to compute here and nowhere else in this schema.
    pub fn sum_us(&self) -> u64 {
        [
            self.frame_decode_us,
            self.input_decode_us,
            self.sidecar_decode_us,
            self.payload_validation_us,
            self.sender_recovery_us,
            self.pre_execution_consensus_us,
            self.context_check_us,
            self.witness_self_consistency_us,
            self.materialize_us,
            self.provider_setup_us,
            self.evm_us,
            self.accessed_state_capture_us,
            self.post_execution_consensus_us,
            self.hash_post_state_us,
            self.trie_clone_us,
            self.state_root_us,
            self.root_completeness_us,
            self.miss_policy_check_us,
            self.cache_update_us,
            self.trie_retention_us,
            self.next_cache_anchor_us,
            self.trie_commit_us,
            self.pair_commit_us,
            self.undo_prune_us,
        ]
        .into_iter()
        .flatten()
        .fold(0u64, u64::saturating_add)
    }
}

/// Aggregates reconstructed from the leaves and the core's own totals. Never added to `phases`.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct DerivedTimings {
    /// The four admission phases together — the value v1 published as `admission_us`.
    pub admission_total_us: Option<u64>,
    /// The older in-process primary the paired ExEx benchmark published: deserialize + context
    /// check + self-consistency + materialize + provider setup + EVM. Kept so a v2 record still
    /// yields that metric.
    pub state_access_execution_us: Option<u64>,
    /// Deserialize + witness checks/materialization + EVM + hashing + sparse-trie root.
    pub execution_core_us: Option<u64>,
    /// The core's own DB-free validation total, diagnostics included.
    pub raw_total_us: Option<u64>,
}

/// What it cost to hand this commit's bytes to the validator, measured where the frame was read.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FrameCosts {
    /// The commit frame's sequence.
    pub(crate) sequence: u64,
    /// Statting and reading the frame file into memory.
    pub(crate) delivery_us: Option<u64>,
    /// Decoding the envelope and body. The first leaf inside the validation boundary.
    pub(crate) frame_decode_us: Option<u64>,
    /// The instant the frame decode began, which is where the primary boundary opens.
    ///
    /// Carried from the read so `standalone_validation_us` is one continuous wall reading from
    /// decode to pair commit — the untimed dispatch between the read path and `replay_commit`
    /// lands inside the boundary (and thus in `unattributed_validation_us`) instead of being
    /// silently dropped by summing separately-timed segments.
    pub(crate) validation_open: Option<Instant>,
}

impl FrameCosts {
    pub(crate) fn of(frame: &crate::spool::SpooledFrame) -> Self {
        Self {
            sequence: frame.header.sequence,
            delivery_us: Some(frame.delivery_us),
            frame_decode_us: Some(frame.frame_decode_us),
            validation_open: Some(frame.validation_open),
        }
    }
}

/// Accumulates one attempt's measurements and closes them into a [`BlockTiming`].
///
/// Every exit from [`replay_commit`] finishes exactly one of these, which is the contract the
/// follower's verdict line reads `report.blocks.last()` under: one entry per attempt, always.
struct AttemptTimer {
    started: Instant,
    number: u64,
    costs: FrameCosts,
    input_decode_us: Option<u64>,
    sidecar_decode_us: Option<u64>,
    mutation_check_us: Option<u64>,
    admission: Option<AdmissionTimings>,
    transition_us: Option<u64>,
    pair_commit_us: Option<u64>,
    undo_prune_us: Option<u64>,
    oracle_compare_us: Option<u64>,
    /// The primary wall, frozen before the oracle comparison so harness work stays outside it.
    validation_wall_us: Option<u64>,
    core: Option<Box<ValidationPhaseTimings>>,
}

impl AttemptTimer {
    fn open(number: u64, costs: FrameCosts) -> Self {
        Self {
            // The boundary opened where the frame decode began, when the read path said so; a
            // caller without a carried instant (tests, synthetic frames) opens it here instead.
            started: costs.validation_open.unwrap_or_else(Instant::now),
            number,
            costs,
            input_decode_us: None,
            sidecar_decode_us: None,
            mutation_check_us: None,
            admission: None,
            transition_us: None,
            pair_commit_us: None,
            undo_prune_us: None,
            oracle_compare_us: None,
            validation_wall_us: None,
            core: None,
        }
    }

    /// Freezes the primary boundary. Called after the pair commit and before the oracle compare;
    /// an early exit that never gets here is closed at [`finish`](Self::finish) time instead.
    fn close_validation(&mut self) {
        self.validation_wall_us = Some(self.started.elapsed().as_micros() as u64);
    }

    /// Closes the attempt into a [`BlockTiming`] and appends it to the report.
    fn finish(self, verdict: &'static str, report: &mut ReplayReport) {
        let wall =
            self.validation_wall_us.unwrap_or_else(|| self.started.elapsed().as_micros() as u64);
        // One wall reading, opened at the frame decode and closed at the pair commit. The
        // mutation checks run inside that window but are harness work, so they are the one
        // subtraction — labelled beside the primary rather than hidden in it.
        let standalone_validation_us =
            wall.saturating_sub(self.mutation_check_us.unwrap_or_default());
        let core = self.core.as_deref();
        let phases = PhaseLeaves {
            frame_decode_us: self.costs.frame_decode_us,
            input_decode_us: self.input_decode_us,
            sidecar_decode_us: self.sidecar_decode_us,
            payload_validation_us: self.admission.and_then(|a| a.payload_validation_us),
            sender_recovery_us: self.admission.and_then(|a| a.sender_recovery_us),
            pre_execution_consensus_us: self.admission.and_then(|a| a.pre_execution_consensus_us),
            context_check_us: core.map(|c| c.context_check_us),
            witness_self_consistency_us: core.map(|c| c.witness_self_consistency_us),
            materialize_us: core.map(|c| c.materialize_us),
            provider_setup_us: core.map(|c| c.provider_setup_us),
            evm_us: core.map(|c| c.evm_us),
            accessed_state_capture_us: core.map(|c| c.accessed_state_capture_us),
            post_execution_consensus_us: core.map(|c| c.post_execution_consensus_us),
            hash_post_state_us: core.map(|c| c.hash_post_state_us),
            trie_clone_us: core.map(|c| c.trie_clone_us),
            state_root_us: core.map(|c| c.state_root_us),
            root_completeness_us: core.map(|c| c.root_completeness_us),
            miss_policy_check_us: core.map(|c| c.miss_policy_check_us),
            cache_update_us: core.map(|c| c.cache_update_us),
            trie_retention_us: core.map(|c| c.trie_retention_us),
            next_cache_anchor_us: core.map(|c| c.next_cache_anchor_us),
            trie_commit_us: core.map(|c| c.trie_commit_us),
            pair_commit_us: self.pair_commit_us,
            undo_prune_us: self.undo_prune_us,
        };
        let leaf_sum = phases.sum_us();
        if leaf_sum > standalone_validation_us {
            report.timing_anomalies += 1;
        }
        let derived = DerivedTimings {
            admission_total_us: self.admission.map(|a| a.total_us()),
            state_access_execution_us: core.map(|c| c.state_access_execution_us),
            execution_core_us: core.map(|c| c.execution_core_us),
            raw_total_us: core.map(|c| c.raw_total_us),
        };
        report.standalone_validation_us =
            report.standalone_validation_us.saturating_add(standalone_validation_us);
        report.blocks.push(BlockTiming {
            number: self.number,
            sequence: self.costs.sequence,
            verdict,
            admission_us: self.admission.map(|a| a.total_us()),
            transition_us: self.transition_us,
            standalone_validation_us,
            delivery_us: self.costs.delivery_us,
            oracle_compare_us: self.oracle_compare_us,
            mutation_check_us: self.mutation_check_us,
            unattributed_validation_us: standalone_validation_us.saturating_sub(leaf_sum),
            phases,
            derived,
            details: self.core,
        });
    }
}

impl ReplayReport {
    /// Whether the replay agreed with the recording on every block it compared.
    ///
    /// This axis is about the two implementations, and deliberately not about the chain: a reorg
    /// the pair could not undo says nothing about whether the blocks it *did* replay matched. An
    /// earlier version folded the two together, so a corpus containing one deep reorg could never
    /// report agreement no matter how cleanly it recovered.
    pub fn agreed(&self) -> bool {
        self.disagreements.is_empty() &&
            self.failures.is_empty() &&
            self.mutation_failures.is_empty()
    }

    /// Whether every canonical block the corpus carried was actually verified.
    ///
    /// False when a recovery landed somewhere other than the block it asked for, when commits
    /// went by while the driver was waiting for one, or when an announced winning branch was
    /// never delivered in full. A run can agree everywhere it looked and still have holes.
    pub fn continuous(&self) -> bool {
        self.skipped_awaiting_resync == 0 &&
            self.winning_branch_incomplete == 0 &&
            self.resyncs.iter().all(|resync| resync.continuous)
    }

    /// Whether the replay reached the end of the corpus rather than stopping inside it.
    pub const fn complete(&self) -> bool {
        self.terminal.is_none()
    }

    /// Whether this run's admission checks were checking anything.
    ///
    /// A corpus of reconstructions exercises the code and proves nothing about the rules, so a
    /// report that did not say this would be read as stronger than it is.
    pub const fn admission_is_load_bearing(&self) -> bool {
        self.witnessed > 0
    }
}

/// Replays a recorded stream and checks it against its own oracle.
///
/// Frames are read one at a time rather than materialized: a long corpus is tens of gigabytes of
/// commits, and holding it whole would make the corpus-as-evidence design unusable at exactly the
/// lengths that matter.
pub fn replay(dir: &Path, options: &ReplayOptions) -> eyre::Result<ReplayReport> {
    let mut spool = SpoolIter::open(dir, &options.frame_limits)?;

    let mut report = ReplayReport::default();
    let mut phase = BatchPhase::AwaitingManifest;
    // The window opened by a restore that landed on its target, while it is being replayed.
    let mut rewind: Option<RewindWindow> = None;

    while let Some(frame) = spool.next_frame()? {
        // Whether the frame about to run is one the window is replaying. The count it feeds is
        // evidence that commits were *verified*, so it is incremented where the verdict is known
        // and not here: a fault on the first window commit skips the rest, and counting on entry
        // would report every one of them as replayed.
        let mut in_window = false;
        // The window closes where the checkpoint that opened it sits, and the stream resumes past
        // that checkpoint's chunks. Reaching `until` means every commit below it has been
        // verified against the restored pair, so the frame just read is the checkpoint itself and
        // is dropped rather than collected a second time.
        if let Some(window) = rewind {
            if frame.header.sequence >= window.until {
                rewind = None;
                spool.seek_to(window.resume_at);
                info!(
                    target: "ps_replay",
                    replayed = report.rewind_replayed_commits,
                    resume_at = window.resume_at,
                    "The rewind window closed; the replay resumes past the installed checkpoint"
                );
                continue
            }
            // The producer fences its export against anything that would change the branch, so a
            // published late checkpoint has only commits beneath it. Enforced, not assumed: a
            // window holding anything else is a spool this driver must not read as continuous.
            if !matches!(frame.event, StreamEvent::Commit(_)) {
                report.failures.push(format!(
                    "a {:?} frame sits at sequence {} inside a rewind window, which holds only \
                     commits",
                    frame.header.kind, frame.header.sequence
                ));
                rewind = None;
                spool.seek_to(window.resume_at);
                continue
            }
            in_window = true;
        }
        // A faulted pair replays nothing further. The remaining frames are counted rather than
        // run: replaying commits against a blocked pair would generate one `BlockSkipped`
        // failure per block, and a single fault would read as a cascade.
        if report.terminal.is_some() {
            if matches!(frame.event, StreamEvent::Commit(_)) {
                report.skipped_after_fault += 1;
            }
            continue
        }
        // Checked before the match so the phase is never moved out of without being put back.
        if matches!(frame.event, StreamEvent::Commit(_)) &&
            options.limit.is_some_and(|limit| report.commits as usize >= limit)
        {
            break
        }
        let sequence = frame.header.sequence;
        let costs = FrameCosts::of(&frame);
        let rewind_before = rewind;
        phase = match (phase, frame.event) {
            (phase, StreamEvent::Manifest(found)) => match phase {
                BatchPhase::AwaitingManifest => {
                    found.check_opens(sequence)?;
                    info!(
                        target: "ps_replay",
                        chain_id = found.chain_id,
                        epoch = found.epoch,
                        producer = %found.producer,
                        account_window = found.account_window,
                        storage_window = found.storage_window,
                        "Stream manifest"
                    );
                    BatchPhase::AwaitingCheckpoint { manifest: found }
                }
                // A second manifest is a new epoch: the producer restarted and its state broke,
                // so nothing below it can be continued into. Not a failure — a restart is
                // ordinary, and one that says so is behaving correctly — but the checkpoint that
                // follows rebootstraps this driver rather than continuing it, which is an
                // explicit reset and reported as one.
                BatchPhase::AwaitingCheckpoint { manifest } |
                BatchPhase::Collecting { manifest, .. } |
                BatchPhase::Live { manifest, .. } |
                BatchPhase::AwaitingResync { manifest, .. } => {
                    found.check_succeeds(&manifest, sequence)?;
                    report.epoch_transitions += 1;
                    warn!(
                        target: "ps_replay",
                        sequence,
                        epoch = found.epoch,
                        "The producer restarted its stream; the next checkpoint rebootstraps this \
                         replay and the interval across the boundary is not validated"
                    );
                    BatchPhase::AwaitingResync {
                        manifest: found,
                        target_ancestor: None,
                        announced_at: None,
                    }
                }
            },
            (phase, StreamEvent::Checkpoint(found)) => {
                // The declaration is checked before any chunk is buffered, so a corrupt or
                // hostile checkpoint cannot turn its own transport fields into an allocation.
                found
                    .validate_declared(DEFAULT_MAX_SNAPSHOT_BYTES)
                    .map_err(|err| eyre::eyre!("checkpoint declaration refused: {err}"))?;
                let (manifest, purpose) = match phase {
                    BatchPhase::AwaitingManifest => {
                        eyre::bail!("a checkpoint arrived before the manifest")
                    }
                    BatchPhase::AwaitingCheckpoint { manifest } => {
                        (manifest, CollectPurpose::Install)
                    }
                    // A discontinuity this driver could not undo: it skipped the frames between,
                    // so there is nothing behind the cursor it is entitled to replay. The window
                    // stays closed and those commits remain counted as skipped, exactly as before.
                    BatchPhase::AwaitingResync { manifest, target_ancestor, announced_at } => {
                        // The frames this driver skipped while it waited are the winning branch
                        // when the checkpoint lands on the block it is waiting for, and they are
                        // behind the cursor for the same reason the forced path's are: the
                        // producer published them before its export finished.
                        (
                            manifest,
                            CollectPurpose::Resync {
                                target_ancestor,
                                window_from: announced_at.map(|at| at + 1),
                            },
                        )
                    }
                    BatchPhase::Collecting { manifest, .. } => {
                        report.failures.push(format!(
                            "a checkpoint arrived at sequence {sequence} while the previous one's \
                             chunks were incomplete"
                        ));
                        (
                            manifest,
                            CollectPurpose::Resync { target_ancestor: None, window_from: None },
                        )
                    }
                    BatchPhase::Live { manifest, state, announced, announced_at, pending_tip } => {
                        match announced {
                            // The producer's recovery checkpoint for a reorg this driver already
                            // undid by itself. Two independent implementations reached the same
                            // generation, so comparing them is a live cross-check rather than a
                            // formality — and installing it would replace state this pair derived
                            // with state it has no reason to prefer.
                            Some(ancestor) if options.force_restore_at != Some(sequence) => (
                                manifest,
                                CollectPurpose::CrossCheck { state, ancestor, pending_tip },
                            ),
                            // Forced: install it instead, and judge the rest of the corpus against
                            // the pair the producer's own snapshot produces.
                            Some(ancestor) => {
                                // Everything this driver verified above the ancestor is about to
                                // be discarded with the pair it verified it on, and under
                                // write-through those commits are *behind* this frame. The window
                                // names them so the restored pair can climb back to the
                                // checkpoint's own position instead of meeting the next frame
                                // several blocks short of its parent.
                                let window_from = announced_at.map(|at| at + 1);
                                info!(
                                    target: "ps_replay",
                                    sequence,
                                    ancestor = ancestor.number,
                                    window_from,
                                    "Installing the recovery checkpoint rather than skimming it, \
                                     as a consumer with no retained generation would have to"
                                );
                                report.last_verified = state.history.tip().map(|tip| tip.number);
                                (
                                    manifest,
                                    CollectPurpose::Resync {
                                        target_ancestor: Some(ancestor),
                                        window_from,
                                    },
                                )
                            }
                            None => {
                                report.failures.push(format!(
                                    "an unannounced checkpoint arrived at sequence {sequence}; \
                                     the grammar has no mid-stream checkpoint without a reorg or \
                                     reset in front of it"
                                ));
                                report.last_verified = state.history.tip().map(|tip| tip.number);
                                (
                                    manifest,
                                    CollectPurpose::Resync {
                                        target_ancestor: None,
                                        window_from: None,
                                    },
                                )
                            }
                        }
                    }
                };
                let collecting = BatchPhase::Collecting {
                    manifest,
                    checkpoint_sequence: sequence,
                    checkpoint: found,
                    chunks: Vec::new(),
                    purpose,
                };
                finish_collection_if_complete(
                    collecting,
                    &mut report,
                    &mut rewind,
                    options.max_rewind_frames,
                )?
            }
            (
                BatchPhase::Collecting {
                    manifest,
                    checkpoint,
                    checkpoint_sequence,
                    mut chunks,
                    purpose,
                },
                StreamEvent::SnapshotChunk(chunk),
            ) => {
                chunks.push(chunk);
                finish_collection_if_complete(
                    BatchPhase::Collecting {
                        manifest,
                        checkpoint,
                        checkpoint_sequence,
                        chunks,
                        purpose,
                    },
                    &mut report,
                    &mut rewind,
                    options.max_rewind_frames,
                )?
            }
            (phase, StreamEvent::SnapshotChunk(_)) => {
                report.failures.push(format!(
                    "a snapshot chunk arrived at sequence {sequence} with no checkpoint expecting it"
                ));
                phase
            }
            (
                BatchPhase::Live { manifest, mut state, announced, announced_at, mut pending_tip },
                StreamEvent::Commit(commit),
            ) => {
                let (input, oracle) = commit.split();
                let block = input.block;
                match replay_commit(&mut state, input, &oracle, options, costs, &mut report) {
                    CommitOutcome::Fault(fault) => {
                        error!(
                            target: "ps_replay",
                            block = block.number,
                            %fault,
                            readiness = state.pair.readiness.state().label(),
                            "The pair can go no further; the remaining commits are skipped, not \
                             replayed"
                        );
                        report.failures.push(fault.to_string());
                        report.terminal = Some(fault.to_string());
                        report.terminal_kind = Some("replay_fault");
                    }
                    CommitOutcome::Compared => {
                        report.last_verified = Some(block.number);
                        if in_window {
                            report.rewind_replayed_commits += 1;
                        }
                        // The producer said which block completes the branch it moved to. Reaching
                        // that height with a different hash means the frame and the delivery
                        // disagree about what was replaced, which no later commit can settle.
                        if let Some(tip) = pending_tip &&
                            tip.number == block.number
                        {
                            if tip.hash == block.hash {
                                pending_tip = None;
                            } else {
                                report.failures.push(format!(
                                    "the winning branch reached {} as {:?} but the reorg announced \
                                     {:?}",
                                    block.number, block.hash, tip.hash
                                ));
                                report.terminal = Some(
                                    "the winning branch did not match its announced tip".into(),
                                );
                                report.terminal_kind = Some("winning_tip_mismatch");
                            }
                        }
                    }
                    CommitOutcome::Rejected => {}
                }
                BatchPhase::Live { manifest, state, announced, announced_at, pending_tip }
            }
            (
                BatchPhase::AwaitingResync { manifest, target_ancestor, announced_at },
                StreamEvent::Commit(_),
            ) => {
                report.skipped_awaiting_resync += 1;
                BatchPhase::AwaitingResync { manifest, target_ancestor, announced_at }
            }
            (_, StreamEvent::Commit(_)) => {
                eyre::bail!(
                    "a commit arrived at sequence {sequence} before a restorable checkpoint"
                )
            }
            (
                BatchPhase::Live { manifest, mut state, pending_tip, .. },
                StreamEvent::Reorg(found),
            ) => {
                // A branch that was still being delivered is not a hole when the producer
                // itself replaces it: the announcement below supersedes this one, and the blocks
                // between here and the old tip never became canonical. It *is* a hole when
                // nothing valid replaces it, which is decided per outcome below.
                let superseded = pending_tip;
                let outcome = apply_reorg(&mut state, &found);
                match (superseded, outcome.withdraws_an_announced_branch()) {
                    (Some(tip), true) => note_supersession(&mut report, tip, found.winning_tip),
                    (Some(_), false) => report.winning_branch_incomplete += 1,
                    (None, _) => {}
                }
                match outcome {
                    ReorgOutcome::Applied { ancestor, undone, revert, winning_tip } => {
                        if revert {
                            report.reverts_applied += 1;
                        } else {
                            report.reorgs_applied += 1;
                        }
                        report.last_verified = Some(ancestor.number);
                        info!(
                            target: "ps_replay",
                            ancestor = ancestor.number,
                            undone = undone.number,
                            revert,
                            "Applied a recorded reorg; the replay continues on the winning branch"
                        );
                        BatchPhase::Live {
                            manifest,
                            state,
                            announced: Some(ancestor),
                            announced_at: Some(sequence),
                            pending_tip: winning_tip,
                        }
                    }
                    // Bound to this driver's own branch, so the ancestor is a block it verified
                    // and a checkpoint there resumes it exactly. The superseded branch is
                    // withdrawn by a statement this driver could authenticate.
                    ReorgOutcome::Unrecoverable { ancestor, depth, detail } => {
                        report.reorgs_inapplicable += 1;
                        warn_inapplicable(ancestor, depth, &detail, true);
                        report.last_verified = state.history.tip().map(|tip| tip.number);
                        BatchPhase::AwaitingResync {
                            manifest,
                            target_ancestor: Some(ancestor),
                            announced_at: Some(sequence),
                        }
                    }
                    // Well-formed but about a branch this driver never held. It carries no
                    // authority: no target, so no recovery under it can be called continuous, and
                    // the branch it interrupted stays counted as unfinished.
                    ReorgOutcome::Unbound { ancestor, depth, detail } => {
                        report.reorgs_inapplicable += 1;
                        warn_inapplicable(ancestor, depth, &detail, false);
                        report.last_verified = state.history.tip().map(|tip| tip.number);
                        BatchPhase::AwaitingResync {
                            manifest,
                            target_ancestor: None,
                            announced_at: None,
                        }
                    }
                    ReorgOutcome::Malformed { detail } => {
                        report.failures.push(format!(
                            "the reorg at sequence {sequence} is not a reorg this driver can \
                             evaluate: {detail}"
                        ));
                        report.last_verified = state.history.tip().map(|tip| tip.number);
                        BatchPhase::AwaitingResync {
                            manifest,
                            target_ancestor: None,
                            announced_at: None,
                        }
                    }
                }
            }
            (phase, StreamEvent::Reorg(found)) => {
                report.failures.push(format!(
                    "a reorg at {} arrived with no pair to apply it to",
                    block_label(found.common_ancestor)
                ));
                phase
            }
            (phase, StreamEvent::Reset(reset)) => {
                // The producer's own statement that it moved somewhere no incremental event can
                // express. Not a failure of this replay, and not something it can recover from
                // without a checkpoint.
                warn!(
                    target: "ps_replay",
                    reason = ?reset.reason,
                    detail = %reset.detail,
                    "The producer recorded a reset; verification stops until a checkpoint arrives"
                );
                match phase {
                    BatchPhase::Live { manifest, state, pending_tip, .. } => {
                        // A reset withdraws the stream without naming a replacement branch, so an
                        // announced tip that never arrived stays a hole.
                        report.winning_branch_incomplete += u64::from(pending_tip.is_some());
                        report.last_verified = state.history.tip().map(|tip| tip.number);
                        BatchPhase::AwaitingResync {
                            manifest,
                            target_ancestor: None,
                            announced_at: None,
                        }
                    }
                    BatchPhase::AwaitingCheckpoint { manifest } |
                    BatchPhase::AwaitingResync { manifest, .. } |
                    BatchPhase::Collecting { manifest, .. } => BatchPhase::AwaitingResync {
                        manifest,
                        target_ancestor: None,
                        announced_at: None,
                    },
                    BatchPhase::AwaitingManifest => {
                        eyre::bail!("a reset arrived before the manifest")
                    }
                }
            }
            (phase, StreamEvent::End(end)) => {
                // Orderly termination, not success: the producer's close path ran, and the kind
                // says under what circumstances.
                info!(
                    target: "ps_replay",
                    kind = end.kind.as_str(),
                    reason = %end.reason,
                    last_sequence = end.last_sequence,
                    "Stream ended"
                );
                phase
            }
        };
        // A restore that opened a window did so while standing on the checkpoint frame; the
        // commits it has to replay are behind the cursor. Enter the window exactly once, here,
        // where the phase has settled into the restored pair.
        if let (None, Some(window)) = (rewind_before, rewind) {
            spool.seek_to(window.from);
        }
    }

    finish_phase(phase, &mut report);
    report.closed = spool.closed();
    info!(
        target: "ps_replay",
        dir = %dir.display(),
        commits = report.commits,
        reorgs_applied = report.reorgs_applied,
        reverts_applied = report.reverts_applied,
        disagreements = report.disagreements.len(),
        failures = report.failures.len(),
        // Bytes actually read, which a forced restore's window replay makes larger than the
        // corpus: those frames really were read twice.
        bytes_read = spool.bytes(),
        closed = report.closed,
        continuous = report.continuous(),
        "Read the recorded stream"
    );
    if !report.closed {
        warn!(
            target: "ps_replay",
            "The corpus has no End frame, so it was cut rather than finished. Everything above \
             describes the prefix that survived"
        );
    }

    Ok(report)
}

/// Where a batch replay is in the stream's grammar.
///
/// Replaced four loose locals whose only coupling was the order they happened to be assigned in.
/// That arrangement could not represent a second checkpoint at all: the restore was one-shot and
/// the chunk buffer was never cleared, so a corpus carrying a producer's recovery checkpoint was
/// read as one checkpoint with two checkpoints' worth of chunks appended to it.
enum BatchPhase {
    /// Nothing accepted yet; the first frame must be the manifest.
    AwaitingManifest,
    /// Identity known; waiting for a checkpoint to bootstrap from.
    AwaitingCheckpoint { manifest: Manifest },
    /// A checkpoint arrived and its declared chunks are being collected.
    Collecting {
        manifest: Manifest,
        checkpoint: Checkpoint,
        /// The checkpoint frame's own sequence, which is what a resume would have to name.
        checkpoint_sequence: u64,
        chunks: Vec<SnapshotChunk>,
        purpose: CollectPurpose,
    },
    /// A pair is restored and verifying commits.
    Live {
        manifest: Manifest,
        state: Box<ReplayState>,
        /// The ancestor an applied reorg just returned to, while its checkpoint may still arrive.
        announced: Option<BlockRef>,
        /// The sequence that announcement arrived at. Under write-through publication the
        /// winning branch is published before its recovery checkpoint, so this is where the
        /// window of commits below a late checkpoint begins.
        announced_at: Option<u64>,
        /// The tip an applied reorg said the winning branch would reach.
        pending_tip: Option<BlockRef>,
    },
    /// Verification stopped at a discontinuity; only a checkpoint can restart it.
    AwaitingResync {
        manifest: Manifest,
        /// The block a recovery checkpoint has to land on to be continuous.
        target_ancestor: Option<BlockRef>,
        /// The sequence the announcement that named that target arrived at, so the commits it
        /// published below its recovery checkpoint can be replayed rather than skipped.
        ///
        /// Bound to `target_ancestor`: every path that withdraws the target — a reset, an epoch,
        /// a reorg this driver cannot place — clears both, so a superseded announcement can never
        /// leave its own window behind for a later checkpoint to replay.
        announced_at: Option<u64>,
    },
}

/// Why a checkpoint's chunks are being collected.
enum CollectPurpose {
    /// The stream's first checkpoint. It becomes the pair.
    Install,
    /// A checkpoint announced by a reorg this driver already applied: compared against the pair
    /// it already holds, then skipped.
    CrossCheck { state: Box<ReplayState>, ancestor: BlockRef, pending_tip: Option<BlockRef> },
    /// Recovery from a discontinuity. It becomes the pair, and where it lands decides whether the
    /// recovery was continuous or an explicit reset.
    Resync {
        target_ancestor: Option<BlockRef>,
        /// First frame of the commit window below this checkpoint, when there is one to replay.
        window_from: Option<u64>,
    },
}

/// The commit frames a restored replay re-reads before resuming at the live edge: sequences
/// `[from, until)`, with `until` the installed checkpoint's own frame.
///
/// A recovery checkpoint published after the winning branch names an ancestor below frames the
/// reader has already passed. Installing it alone would leave the pair at that ancestor with the
/// next frame in the stream several blocks above it, and the replay would refuse that block for a
/// parent hash it never had a chance to build — which is what the batch driver did until this
/// existed, while the follower had replayed the same window all along.
#[derive(Debug, Clone, Copy)]
struct RewindWindow {
    from: u64,
    until: u64,
    /// First frame past the checkpoint's chunks: where the stream resumes once the window closes.
    resume_at: u64,
}

/// The largest commit window a recovery will replay from the spool rather than skip.
///
/// One definition for both consumers — the follower's live rewind and this driver's restore —
/// because a bound that differed between them would make the same corpus recoverable on one side
/// and an explicit reset on the other. Far above the real shape: a 195 s export at one block per
/// 12 s is about 17 commits, so it fires only on a pathological spool, and it degrades to the
/// explicit reset the same checkpoint produced before rewinds existed, never to an unbounded
/// re-read.
pub(crate) const MAX_REWIND_FRAMES: u64 = 4_096;

/// Undo generations a consumer keeps after a commit.
///
/// One. A consumer can only self-recover a reorg exactly one block deep: `reorg.rs` refuses
/// anything deeper with "a reorg {depth} blocks deep needs a snapshot at the common ancestor; the
/// retained generation reaches exactly one block", and the coordination layer says the same --
/// "Only depth 1. The flat undo log reaches further, but the trie does not, and the pair has to
/// move as one generation." Every read of the log is `back()`, so a record below the tip is
/// unreachable by any supported recovery path.
///
/// The producer sets its own retention by a different rule — finality, with a fixed-depth floor
/// when no finality is available (`partial-stateless-exex/src/lib.rs`). That is not a bound this
/// side inherits. Whatever the producer keeps its log for, retaining as much here would hold
/// records nothing on this side can read.
pub(crate) const CONSUMER_UNDO_RETAIN_BLOCKS: u64 = 1;

/// How often the memory probe reports, in blocks. `0` (the default) disables it.
///
/// Read once: this is diagnostic instrumentation on the hot path, and re-reading the environment
/// every block would itself be a cost the probe is supposed to be measuring around.
fn memory_probe_interval() -> u64 {
    static INTERVAL: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *INTERVAL.get_or_init(|| {
        std::env::var("PS_MEMORY_PROBE").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
    })
}

/// jemalloc's own accounting, refreshed. `None` unless built with `--features jemalloc-stats`.
///
/// The three numbers answer different questions and only together identify where resident memory
/// has gone: `allocated` is what the process owns right now, `active` is the pages jemalloc holds
/// because of those live allocations, and `resident` is what it has not returned to the kernel.
/// So `allocated` rising is owned structures or container capacity; `allocated` flat while
/// `active` rises is fragmentation inside pages; `active` flat while `resident` rises is dirty,
/// muzzy or arena retention. A decay setting moves only the third and therefore cannot tell the
/// three apart, which is why they are read rather than inferred.
#[cfg(feature = "jemalloc-stats")]
fn jemalloc_stats() -> Option<[u64; 5]> {
    use tikv_jemalloc_ctl::{epoch, stats};
    // The stats are cached behind an epoch; without advancing it every read returns the values
    // from process start.
    epoch::advance().ok()?;
    Some([
        stats::allocated::read().ok()? as u64,
        stats::active::read().ok()? as u64,
        stats::resident::read().ok()? as u64,
        stats::mapped::read().ok()? as u64,
        stats::retained::read().ok()? as u64,
    ])
}

#[cfg(not(feature = "jemalloc-stats"))]
fn jemalloc_stats() -> Option<[u64; 5]> {
    None
}

/// Emit one memory sample when `PS_MEMORY_PROBE` is set and this is a reporting block.
fn memory_probe(height: u64) {
    let interval = memory_probe_interval();
    if interval == 0 || height % interval != 0 {
        return
    }
    let mut rss_kib = 0u64;
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                rss_kib = rest.split_whitespace().next().and_then(|v| v.parse().ok()).unwrap_or(0);
            }
        }
    }
    match jemalloc_stats() {
        Some([allocated, active, resident, mapped, retained]) => eprintln!(
            "PS_MEMORY\tblock={height}\trss_kib={rss_kib}\tallocated={allocated}\t\
active={active}\tresident={resident}\tmapped={mapped}\tretained={retained}"
        ),
        None => eprintln!("PS_MEMORY\tblock={height}\trss_kib={rss_kib}\tjemalloc_stats=unavailable"),
    }
}


/// Restores or cross-checks a checkpoint once every chunk it declared has arrived.
fn finish_collection_if_complete(
    phase: BatchPhase,
    report: &mut ReplayReport,
    rewind: &mut Option<RewindWindow>,
    max_rewind_frames: u64,
) -> eyre::Result<BatchPhase> {
    let BatchPhase::Collecting { manifest, checkpoint, checkpoint_sequence, chunks, purpose } =
        phase
    else {
        return Ok(phase)
    };
    if chunks.len() < checkpoint.snapshot_chunks as usize {
        return Ok(BatchPhase::Collecting {
            manifest,
            checkpoint,
            checkpoint_sequence,
            chunks,
            purpose,
        })
    }
    match purpose {
        CollectPurpose::Install => {
            let state = restore(&manifest, &checkpoint, &chunks)?;
            Ok(BatchPhase::Live {
                manifest,
                state: Box::new(state),
                announced: None,
                announced_at: None,
                pending_tip: None,
            })
        }
        CollectPurpose::CrossCheck { state, ancestor, pending_tip } => {
            if let Err(disagreement) =
                cross_check_recovery_checkpoint(&state, &checkpoint, ancestor)
            {
                error!(
                    target: "ps_replay",
                    block = checkpoint.block.number,
                    field = disagreement.field,
                    recorded = %disagreement.recorded,
                    replayed = %disagreement.replayed,
                    "The producer's recovery checkpoint disagrees with the generation this replay \
                     recovered to. One of the two undid the reorg wrongly"
                );
                report.disagreements.push((checkpoint.block, disagreement));
                if consumer_is_at(&state, ancestor) {
                    // Recorded, and then the checkpoint is installed rather than skipped — the
                    // same answer the follower gives. Carrying on from a generation the
                    // operator-trusted checkpoint source contradicts would make every later block
                    // a comparison against disputed state, and a reader could not tell an
                    // independent second finding from the first one cascading. Installing
                    // isolates the finding to the block it is about; the interval it covers is an
                    // explicit reset, never a continuous recovery.
                    let state = restore(&manifest, &checkpoint, &chunks)?;
                    report.resyncs.push(ResyncRecord {
                        at_sequence: checkpoint_sequence,
                        block: checkpoint.block.number,
                        continuous: false,
                        unverified: None,
                        commits_skipped: 0,
                    });
                    return Ok(BatchPhase::Live {
                        manifest,
                        state: Box::new(state),
                        announced: None,
                        announced_at: None,
                        pending_tip,
                    })
                }
                // Late: this replay verified every block past the ancestor for itself, and its
                // own chain outranks the cross-check. Installing here would rewind the pair to a
                // height below commits it already stands behind and turn every remaining frame
                // into a refusal cascade. The disagreement above already fails the run's
                // agreement claim; validation continues — and the chunks are still checked
                // against their declaration, exactly as on the agreeing path: a transport fault
                // is a separate finding, and a mismatch is when hiding one would be easiest.
                report.late_skim_mismatches += 1;
                if let Err(err) = checkpoint.reassemble(&chunks) {
                    report.failures.push(format!(
                        "the recovery checkpoint at sequence {checkpoint_sequence} did not \
                         reassemble: {err}"
                    ));
                }
                return Ok(BatchPhase::Live {
                    manifest,
                    state,
                    announced: None,
                    announced_at: None,
                    pending_tip,
                })
            }
            report.checkpoints_skimmed += 1;
            // The bytes are checked even though the package is not installed: a chunk sequence
            // that does not hash to what the checkpoint declared is a transport fault, and the
            // pair this replay already holds is not evidence about the snapshot's own integrity.
            if let Err(err) = checkpoint.reassemble(&chunks) {
                report.failures.push(format!(
                    "the recovery checkpoint at sequence {checkpoint_sequence} did not reassemble: \
                     {err}"
                ));
            }
            Ok(BatchPhase::Live {
                manifest,
                state,
                announced: None,
                announced_at: None,
                pending_tip,
            })
        }
        CollectPurpose::Resync { target_ancestor, window_from } => {
            let state = restore(&manifest, &checkpoint, &chunks)?;
            // Only a checkpoint that landed on the block recovery asked for licenses a replay of
            // the commits below it: those are the winning branch by construction. A checkpoint
            // that landed anywhere else is an explicit reset, and re-reading frames under it
            // would be replaying a branch this restore has no claim about.
            let landed_on_target = target_ancestor.is_some_and(|target| target == checkpoint.block);
            let mut refused = false;
            *rewind = window_from.filter(|_| landed_on_target).and_then(|from| {
                let until = checkpoint_sequence;
                if until <= from {
                    return None
                }
                if until - from > max_rewind_frames {
                    refused = true;
                    report.rewind_windows_refused += 1;
                    warn!(
                        target: "ps_replay",
                        frames = until - from,
                        bound = max_rewind_frames,
                        "The rewind window exceeds the bound; the restore degrades to an explicit \
                         reset and the commits below the checkpoint go unverified"
                    );
                    return None
                }
                Some(RewindWindow {
                    from,
                    until,
                    resume_at: checkpoint_sequence + 1 + u64::from(checkpoint.snapshot_chunks),
                })
            });
            // A replayed window verifies the commits that would otherwise be skipped, so it is
            // what makes a recovery continuous rather than something continuity survives. A
            // *refused* one is the opposite: the restore stands below commits it will not
            // replay, which is the explicit reset the bound exists to degrade to, and saying so
            // here is the whole point of the bound. Without it a corpus that ends at its
            // checkpoint reports a clean continuous recovery having verified none of the branch.
            let replayed_window = rewind.is_some();
            let skipped = if replayed_window { 0 } else { report.skipped_awaiting_resync };
            let continuous = landed_on_target && skipped == 0 && !refused;
            // What the restore cannot account for. A checkpoint that landed short of the target
            // leaves the interval between the last verified block and itself; a refused window
            // leaves the branch above the checkpoint that this pair will no longer carry.
            let unverified = if refused {
                report
                    .last_verified
                    .filter(|last| *last > checkpoint.block.number)
                    .map(|last| (checkpoint.block.number + 1, last))
            } else {
                (!continuous)
                    .then(|| {
                        report
                            .last_verified
                            .filter(|last| *last < checkpoint.block.number)
                            .map(|last| (last + 1, checkpoint.block.number))
                    })
                    .flatten()
            };
            if continuous {
                info!(
                    target: "ps_replay",
                    block = checkpoint.block.number,
                    "Recovered at the exact block the reorg named; nothing went unverified"
                );
            } else {
                warn!(
                    target: "ps_replay",
                    block = checkpoint.block.number,
                    ?target_ancestor,
                    ?unverified,
                    "Recovered from a checkpoint that is not the block recovery asked for. This is \
                     an explicit checkpoint reset and makes no validation claim for the interval \
                     it skipped"
                );
            }
            if let Some(window) = *rewind {
                info!(
                    target: "ps_replay",
                    block = checkpoint.block.number,
                    replay_from = window.from,
                    replay_until = window.until,
                    resume_at = window.resume_at,
                    "Restored at the exact ancestor; the winning branch replays from the spool \
                     before the stream resumes"
                );
            }
            report.resyncs.push(ResyncRecord {
                at_sequence: checkpoint_sequence,
                block: checkpoint.block.number,
                continuous,
                unverified,
                commits_skipped: skipped,
            });
            report.skipped_awaiting_resync = 0;
            Ok(BatchPhase::Live {
                manifest,
                state: Box::new(state),
                announced: None,
                announced_at: None,
                pending_tip: None,
            })
        }
    }
}

/// Records what the corpus ending in this phase means.
fn finish_phase(phase: BatchPhase, report: &mut ReplayReport) {
    match phase {
        BatchPhase::AwaitingResync { target_ancestor, .. } => {
            let detail = match target_ancestor {
                Some(ancestor) => {
                    format!("the corpus ends waiting for a checkpoint at {}", block_label(ancestor))
                }
                None => "the corpus ends waiting for a checkpoint".to_string(),
            };
            report.terminal_kind = Some("awaiting_resync");
            report.terminal = Some(detail);
        }
        BatchPhase::Collecting { checkpoint, chunks, .. } => {
            report.terminal_kind = Some("incomplete_checkpoint");
            report.terminal = Some(format!(
                "the corpus ends with {} of the checkpoint's {} chunks",
                chunks.len(),
                checkpoint.snapshot_chunks
            ));
        }
        BatchPhase::Live { pending_tip, announced, .. } => {
            if let Some(tip) = pending_tip {
                report.winning_branch_incomplete += 1;
                warn!(
                    target: "ps_replay",
                    tip = tip.number,
                    "The corpus ends before the winning branch reached the tip the reorg announced"
                );
            }
            if announced.is_some() {
                report.recovery_checkpoints_pending_at_end += 1;
            }
        }
        BatchPhase::AwaitingManifest | BatchPhase::AwaitingCheckpoint { .. } => {}
    }
}

/// Records that a valid new announcement replaced a winning branch that was still being delivered.
///
/// Not a hole: the producer withdrew the old tip as a canonical goal, so the blocks between where
/// the delivery got to and where it had been heading never became canonical and were never owed a
/// verdict. Only a branch abandoned with nothing valid in its place is counted incomplete.
fn note_supersession(
    report: &mut ReplayReport,
    superseded: BlockRef,
    replacement: Option<BlockRef>,
) {
    report.winning_branches_superseded += 1;
    info!(
        target: "ps_replay",
        superseded = superseded.number,
        replacement = ?replacement.map(|tip| tip.number),
        "A second reorg replaced the winning branch before it finished; the old tip is withdrawn, \
         not missing"
    );
}

/// Compares a producer's recovery checkpoint against the generation this replay recovered to.
///
/// Every field is one both sides derived independently — the producer from its own database, this
/// consumer by undoing one block — so a mismatch is two implementations disagreeing about the same
/// chain, which is exactly the kind of defect a recorded corpus exists to surface.
pub(crate) fn cross_check_recovery_checkpoint(
    state: &ReplayState,
    checkpoint: &Checkpoint,
    ancestor: BlockRef,
) -> Result<(), Disagreement> {
    if consumer_is_at(state, ancestor) {
        return cross_check_at_ancestor(state, checkpoint, ancestor)
    }
    cross_check_against_history(state, checkpoint)
}

/// Whether the consumer's verified tip is exactly `ancestor` — the position in which a recovery
/// checkpoint's fields can be answered by the live pair. Once the pair has advanced past it (a
/// checkpoint published behind write-through commits, or an export retried at a fresher anchor),
/// only the per-height verified history can answer.
pub(crate) fn consumer_is_at(state: &ReplayState, ancestor: BlockRef) -> bool {
    state.history.tip() == Some(ancestor)
}

/// The original comparison: the pair still sits at the ancestor, so its own fingerprint answers.
fn cross_check_at_ancestor(
    state: &ReplayState,
    checkpoint: &Checkpoint,
    ancestor: BlockRef,
) -> Result<(), Disagreement> {
    if checkpoint.block != ancestor {
        return Err(Disagreement {
            field: "recovery_checkpoint_block",
            recorded: format!("{:?}", checkpoint.block),
            replayed: format!("{ancestor:?}"),
        })
    }
    if checkpoint.cache_policy_id != state.config.cache_policy_id() {
        return Err(Disagreement {
            field: "recovery_checkpoint_policy",
            recorded: format!("{:?}", checkpoint.cache_policy_id),
            replayed: format!("{:?}", state.config.cache_policy_id()),
        })
    }
    let fingerprint = state.pair.fingerprint();
    if fingerprint.trie_state_root != Some(checkpoint.state_root) {
        return Err(Disagreement {
            field: "recovery_checkpoint_state_root",
            recorded: format!("{:?}", checkpoint.state_root),
            replayed: format!("{:?}", fingerprint.trie_state_root),
        })
    }
    if fingerprint.cache_root != checkpoint.cache_root {
        return Err(Disagreement {
            field: "recovery_checkpoint_cache_root",
            recorded: format!("{:?}", checkpoint.cache_root),
            replayed: format!("{:?}", fingerprint.cache_root),
        })
    }
    let announced_head = decode_accepted_head(checkpoint).map(|header| header.hash());
    let held_head = state.pair.accepted_parent().map(|header| header.hash());
    if announced_head != held_head {
        return Err(Disagreement {
            field: "recovery_checkpoint_accepted_head",
            recorded: format!("{announced_head:?}"),
            replayed: format!("{held_head:?}"),
        })
    }
    Ok(())
}

/// The late comparison: the checkpoint binds by its own block, and every field is judged against
/// the record this consumer made when it verified that exact block. The disagreement field names
/// are the same strings the at-ancestor comparison uses, so a mismatch reads identically in
/// either position.
fn cross_check_against_history(
    state: &ReplayState,
    checkpoint: &Checkpoint,
) -> Result<(), Disagreement> {
    let Some(entry) = state.history.entry_at(checkpoint.block) else {
        return Err(Disagreement {
            field: "recovery_checkpoint_block",
            recorded: format!("{:?}", checkpoint.block),
            replayed: "not among the blocks this consumer verified within its retained history"
                .to_string(),
        })
    };
    if checkpoint.cache_policy_id != state.config.cache_policy_id() {
        return Err(Disagreement {
            field: "recovery_checkpoint_policy",
            recorded: format!("{:?}", checkpoint.cache_policy_id),
            replayed: format!("{:?}", state.config.cache_policy_id()),
        })
    }
    if entry.state_root != checkpoint.state_root {
        return Err(Disagreement {
            field: "recovery_checkpoint_state_root",
            recorded: format!("{:?}", checkpoint.state_root),
            replayed: format!("{:?}", Some(entry.state_root)),
        })
    }
    if entry.cache_root != checkpoint.cache_root {
        return Err(Disagreement {
            field: "recovery_checkpoint_cache_root",
            recorded: format!("{:?}", checkpoint.cache_root),
            replayed: format!("{:?}", entry.cache_root),
        })
    }
    // A well-formed checkpoint's accepted head is its own block's header, so at height H the
    // check degenerates to the block hash this consumer verified — and a headless checkpoint
    // still fails closed here.
    let announced_head = decode_accepted_head(checkpoint).map(|header| header.hash());
    if announced_head != Some(entry.hash) {
        return Err(Disagreement {
            field: "recovery_checkpoint_accepted_head",
            recorded: format!("{announced_head:?}"),
            replayed: format!("{:?}", Some(entry.hash)),
        })
    }
    Ok(())
}

/// Everything a replay carries between commits.
///
/// The rules are built once and held here, not per block. `EthBeaconConsensus` carries flags that
/// decide what a block is allowed to be, so a validator that rebuilt it per block would be one
/// configuration change away from disagreeing with itself mid-stream — and it would charge the
/// construction to every measured block.
pub(crate) struct ReplayState {
    /// What this consumer verified for itself, and the state roots it derived.
    ///
    /// Held beside the pair because recovery has to ask a question only the consumer can answer
    /// honestly: the canonical state root at a reorg target. A full node asks its provider; the
    /// recorded frames cannot be asked, because a producer attesting to its own reorg target
    /// turns the retained generation's authentication into a tautology.
    pub(crate) history: VerifiedHistory,
    pub(crate) pair: CoordinatedPair,
    pub(crate) config: CacheConfig,
    pub(crate) chain_spec: Arc<ChainSpec>,
    pub(crate) consensus: EthBeaconConsensus<ChainSpec>,
    pub(crate) evm_config: EthEvmConfig<ChainSpec>,
}

/// Restores the pair a replay validates against, from the checkpoint and its chunks.
pub(crate) fn restore(
    manifest: &Manifest,
    checkpoint: &Checkpoint,
    chunks: &[SnapshotChunk],
) -> eyre::Result<ReplayState> {
    let package_bytes = checkpoint
        .reassemble(chunks)
        .map_err(|err| eyre::eyre!("recorded snapshot did not reassemble: {err}"))?;
    let package = bincode::deserialize(&package_bytes)
        .map_err(|err| eyre::eyre!("recorded snapshot package did not decode: {err}"))?;

    let config = config_for(manifest)?;
    let trusted = TrustedCheckpoint {
        block_number: checkpoint.block.number,
        block_hash: checkpoint.block.hash,
        state_root: checkpoint.state_root,
        cache_root: checkpoint.cache_root,
        cache_policy_id: checkpoint.cache_policy_id,
    };
    let restored = restore_snapshot(package, &trusted, &config)?;

    // The header is installed only because every field a consumer checks it against is in the
    // checkpoint the operator vouched for. A header that fails any of them is dropped, and the
    // pair then waits a block rather than admitting its first child against an unverified parent.
    let accepted_head = decode_accepted_head(checkpoint);

    let chain_spec = chain_spec_for(manifest)?;
    info!(
        target: "ps_replay",
        block = checkpoint.block.number,
        block_hash = ?checkpoint.block.hash,
        accounts = restored.cache.accounts().len(),
        storage = restored.cache.storage().len(),
        codes = restored.cache.codes().len(),
        has_accepted_head = accepted_head.is_some(),
        "Restored a coordinated pair from the recorded checkpoint, with no database"
    );
    Ok(ReplayState {
        history: VerifiedHistory::restored_at(
            checkpoint.block,
            checkpoint.state_root,
            checkpoint.cache_root,
        ),
        pair: CoordinatedPair {
            cache: restored.cache,
            trie_cache: restored.trie_cache,
            previous_generation: None,
            accepted_head,
            readiness: restored.readiness,
        },
        config,
        consensus: EthBeaconConsensus::new(chain_spec.clone()),
        evm_config: EthEvmConfig::new(chain_spec.clone()),
        chain_spec,
    })
}

/// The cache configuration a manifest names, cross-checked against its own policy id.
pub(crate) fn config_for(manifest: &Manifest) -> eyre::Result<CacheConfig> {
    let config = CacheConfig {
        account_window: manifest.account_window,
        storage_window: manifest.storage_window,
    };
    if config.cache_policy_id() != manifest.cache_policy_id {
        eyre::bail!(
            "manifest names policy {:?} but its own windows derive {:?}",
            manifest.cache_policy_id,
            config.cache_policy_id()
        );
    }
    Ok(config)
}

/// Decodes the checkpoint's accepted head, refusing any header that disagrees with it.
pub(crate) fn decode_accepted_head(checkpoint: &Checkpoint) -> Option<SealedHeader> {
    if checkpoint.accepted_head_rlp.is_empty() {
        return None
    }
    let header = match Header::decode(&mut checkpoint.accepted_head_rlp.as_slice()) {
        Ok(header) => header,
        Err(err) => {
            warn!(target: "ps_replay", %err, "Checkpoint header did not decode; ignoring it");
            return None
        }
    };
    let sealed = SealedHeader::seal_slow(header);
    let agrees = sealed.hash() == checkpoint.block.hash &&
        sealed.number == checkpoint.block.number &&
        sealed.state_root == checkpoint.state_root;
    if !agrees {
        warn!(
            target: "ps_replay",
            ?sealed,
            expected = ?checkpoint.block,
            "Checkpoint header does not match the checkpoint; ignoring it"
        );
        return None
    }
    Some(sealed)
}

pub(crate) fn chain_spec_for(manifest: &Manifest) -> eyre::Result<Arc<ChainSpec>> {
    // One chain for now, and named rather than inferred: a validator that guessed a chain spec
    // from a chain id would be choosing fork activation times on the producer's behalf.
    if manifest.chain_id != MAINNET.chain.id() {
        eyre::bail!(
            "this driver is configured for mainnet ({}); the stream names chain {}",
            MAINNET.chain.id(),
            manifest.chain_id
        );
    }
    if manifest.genesis_hash != MAINNET.genesis_hash() {
        eyre::bail!(
            "the stream names genesis {:?}, which is not mainnet's {:?}",
            manifest.genesis_hash,
            MAINNET.genesis_hash()
        );
    }
    Ok(MAINNET.clone())
}

/// Runs one recorded commit through admission and the transition, then compares.
///
/// What each early return leaves behind is part of the contract, not an accident. A decode or
/// admission failure touches no validator state — `UntrustedAdmission` is stateless with respect
/// to the pair — so the next commit may still run ([`CommitOutcome::Rejected`]). Everything past
/// `admit_block` has moved readiness to `Applying`, so a failure there must not return with the
/// tracker still in that transient state: the ExEx's fail-stop masked exactly that leak, and a
/// standalone driver that outlives the rejection makes it observable.
pub(crate) fn replay_commit(
    state: &mut ReplayState,
    input: CommitInput,
    oracle: &CommitOracle,
    options: &ReplayOptions,
    costs: FrameCosts,
    report: &mut ReplayReport,
) -> CommitOutcome {
    match input.payload_provenance {
        PayloadProvenance::Witnessed => report.witnessed += 1,
        PayloadProvenance::Reconstructed => report.reconstructed += 1,
        PayloadProvenance::Absent => report.absent += 1,
    }
    let label = block_label(input.block);
    // Every exit below finishes this timer exactly once, so `report.blocks` gains one entry per
    // attempt — rejected and faulted ones included, which is what makes the follower's
    // `blocks.last()` this attempt's timing rather than a stale neighbour's.
    let mut timer = AttemptTimer::open(input.block.number, costs);

    let decode_started = Instant::now();
    let payload = match input.payload() {
        Ok(Some(payload)) => payload,
        Ok(None) => {
            timer.input_decode_us = Some(decode_started.elapsed().as_micros() as u64);
            report.failures.push(format!(
                "{label}: the commit carries no payload, so admission could not run on it"
            ));
            timer.finish("rejected", report);
            return CommitOutcome::Rejected
        }
        Err(err) => {
            timer.input_decode_us = Some(decode_started.elapsed().as_micros() as u64);
            report.failures.push(format!("{label}: recorded payload did not parse: {err}"));
            timer.finish("rejected", report);
            return CommitOutcome::Rejected
        }
    };
    timer.input_decode_us = Some(decode_started.elapsed().as_micros() as u64);

    let sidecar_started = Instant::now();
    let sidecar: PartialStatelessSidecar = match bincode::deserialize(&input.sidecar) {
        Ok(sidecar) => sidecar,
        Err(err) => {
            timer.sidecar_decode_us = Some(sidecar_started.elapsed().as_micros() as u64);
            report.failures.push(format!("{label}: recorded sidecar did not decode: {err}"));
            timer.finish("rejected", report);
            return CommitOutcome::Rejected
        }
    };
    timer.sidecar_decode_us = Some(sidecar_started.elapsed().as_micros() as u64);

    let admission = UntrustedAdmission::new(state.chain_spec.as_ref(), &state.consensus);

    if options.mutations && input.payload_provenance.is_load_bearing() {
        let mutations_started = Instant::now();
        check_mutations(&admission, &state.pair, &payload, report, &label);
        timer.mutation_check_us = Some(mutations_started.elapsed().as_micros() as u64);
    }

    // Deliberately *not* folded into `timer`: every field on it feeds a latency population, and
    // the cost of executing a block nobody sent belongs in none of them. The budget is spent on
    // the first commits the corpus offers rather than sampled across it — the rule is the same
    // rule on every block, and an offline gate that has proved it five times has proved it.
    if options.mutations_transition.is_some_and(|blocks| {
        // The budget counts blocks, and each one carries every transition mutation there is —
        // written as a product so adding a second kind widens the coverage rather than silently
        // halving the number of blocks that get any.
        report.transition_mutations_checked < (blocks * TransitionMutation::ALL.len()) as u64
    }) && input.payload_provenance.is_load_bearing()
    {
        let started = Instant::now();
        let compromised = check_transition_mutations(
            &admission,
            &mut state.pair,
            ValidatorRules::new(&state.evm_config, &state.consensus),
            state.config.cache_policy_id(),
            &options.reexec_limits,
            &payload,
            &sidecar,
            report,
            &label,
        );
        report.transition_mutation_us += started.elapsed().as_micros() as u64;
        if compromised {
            // Cannot happen, and therefore the one case where continuing is worse than stopping.
            // `TrieCacheDisposition::Discard` withholds the trie generation, but the flat cache is
            // advanced by the transition itself — so a mutation that was wrongly accepted has
            // already moved the pair onto a block that does not exist, and every verdict after it
            // would be measured against that state. The failure is already recorded; this makes it
            // terminal rather than contagious.
            timer.finish("fault", report);
            return CommitOutcome::Fault(fail_applied_block(
                &mut state.pair,
                input.block,
                "a transition mutation moved the pair; it may have advanced on an invalid block"
                    .to_string(),
            ))
        }
    }

    // The parent comes from the pair and never from the frame. A producer that supplied the parent
    // would be choosing the timestamp, gas limit, and base fee its own block is measured against.
    let admitted = match admission.admit(payload, state.pair.accepted_parent()) {
        Ok(mut admitted) => {
            // The reserved slot admission cannot fill itself: it is handed an already-parsed
            // payload, and this driver is "whatever read the payload off the wire".
            admitted.timings.input_decode_us = timer.input_decode_us;
            admitted
        }
        Err(err) => {
            let disagreements = compare_rejected(oracle, err.class());
            report.failures.push(format!(
                "{label}: the replay refused a block the recording accepted: {err} ({})",
                err.class()
            ));
            report.disagreements.extend(disagreements.into_iter().map(|d| (input.block, d)));
            timer.finish("rejected", report);
            return CommitOutcome::Rejected
        }
    };
    timer.admission = Some(admitted.timings);
    report.admission_us += admitted.timings.total_us();

    let block_ctx = block_context(&admitted.block);
    if let BlockAdmission::Rejected(reason) = admit_block(&mut state.pair.readiness, &block_ctx) {
        // The refusal itself latched the tracker `Blocked`; the caches were never touched.
        timer.finish("fault", report);
        return CommitOutcome::Fault(ReplayFault::ReadinessRefused {
            block: input.block,
            reason: format!("{reason:?}"),
        })
    }

    let started = Instant::now();
    let validated = verify_and_apply_sidecar(
        ValidatorRules::new(&state.evm_config, &state.consensus),
        &admitted.block,
        &mut state.pair.cache,
        &sidecar,
        state.config.cache_policy_id(),
        &options.reexec_limits,
        &mut state.pair.trie_cache,
        TrieCacheDisposition::Commit,
    );
    let transition_us = started.elapsed().as_micros() as u64;
    report.transition_us += transition_us;
    timer.transition_us = Some(transition_us);

    let validated = match validated {
        Ok(validated) => validated,
        Err(err) => {
            timer.finish("fault", report);
            return CommitOutcome::Fault(fail_applied_block(
                &mut state.pair,
                input.block,
                format!("{err:#}"),
            ))
        }
    };

    // The commit above pushed one undo record. Drop every older one: without this the log grows by
    // one record per block for the life of the run -- each holding the prior value of every account,
    // storage slot and code the block touched or evicted -- which is what took a 53-hour follower
    // from 0.8 GiB to 13.3 GiB of resident memory. The producer prunes to finality
    // (`partial-stateless-exex/src/lib.rs`); this side has no provider to ask, and needs far less.
    let prune_started = Instant::now();
    let height = state.pair.cache.current_block();
    state.pair.cache.prune_undo_below(height.saturating_sub(CONSUMER_UNDO_RETAIN_BLOCKS));
    timer.undo_prune_us = Some(prune_started.elapsed().as_micros() as u64);
    // The core's instrumentation, completed the way the paired harness completes it: admission
    // and the sidecar decode happened out here in the driver, so the core record carries them
    // only if the driver puts them in.
    let mut core = validated.timings;
    core.admission = admitted.timings;
    core.set_deserialize_us(timer.sidecar_decode_us.unwrap_or_default());
    timer.core = Some(Box::new(core));

    let mut outcome = validated.outcome;
    let displaced = outcome.displaced_trie_cache.take();
    let commit_started = Instant::now();
    state.pair.commit_transition(displaced, &block_ctx, admitted.block.clone_sealed_header(), true);
    // Recorded from this replay's own execution, before the oracle is consulted, so that a reorg
    // arriving later authenticates its target against what this process verified rather than
    // against what the producer said about it. The cache root rides along because a late
    // recovery checkpoint can only be judged against the record made at its height.
    let committed_cache_root = state.pair.fingerprint().cache_root;
    state.history.record(input.block, outcome.state_root, committed_cache_root);
    timer.pair_commit_us = Some(commit_started.elapsed().as_micros() as u64);
    // The verdict is committed; everything after this line is the harness checking itself.
    timer.close_validation();

    // Sampled here and not at the prune: until `commit_transition` above ran, three trie
    // generations were reachable at once — the new one, the parent the transition displaced and
    // handed back, and the one still sitting in `previous_generation`. A sample taken there reads
    // a transition, not a steady state, and would have counted a generation about to be dropped as
    // live. Being past `close_validation` also keeps the probe's own cost — a `/proc` read and six
    // mallctl calls — out of the primary boundary.
    memory_probe(height);

    let compare_started = Instant::now();
    let disagreements = compare_accepted(oracle, &outcome, &state.pair);
    match compare_readiness_watermark(oracle, &state.pair) {
        WatermarkComparison::Agreed => report.watermarks_agreed += 1,
        WatermarkComparison::Unrecorded => report.watermarks_unrecorded += 1,
        WatermarkComparison::Mismatch(mismatch) => {
            report.watermark_mismatches += 1;
            if report.watermark_mismatch_samples.len() < 8 {
                report.watermark_mismatch_samples.push(format!("{label}: {mismatch}"));
            }
        }
    }
    timer.oracle_compare_us = Some(compare_started.elapsed().as_micros() as u64);
    let verdict = if disagreements.is_empty() { "accepted" } else { "disagreed" };
    timer.finish(verdict, report);
    if disagreements.is_empty() {
        report.commits += 1;
        return CommitOutcome::Compared
    }
    for disagreement in disagreements {
        error!(
            target: "ps_replay",
            block = input.block.number,
            field = disagreement.field,
            recorded = %disagreement.recorded,
            replayed = %disagreement.replayed,
            "The replay disagreed with the recording. One of the two is wrong, and which one is \
             an investigation rather than an assumption"
        );
        report.disagreements.push((input.block, disagreement));
    }
    report.commits += 1;
    CommitOutcome::Compared
}

/// Closes the admit-verify-apply boundary after a post-admission failure.
///
/// `admit_block` moved readiness to `Applying`; the failed transition preserved both caches at
/// the parent generation but has no readiness handle to report through. Left there, the tracker
/// would refuse the next block as `BlockSkipped` while *looking* transient — under the ExEx the
/// process dies first, and only a standalone driver ever observes the difference. `abandon_block`
/// makes the stop explicit and terminal: the watermark freezes at the parent, and nothing short
/// of a reset or a reorg below the block releases it.
fn fail_applied_block(pair: &mut CoordinatedPair, block: BlockRef, detail: String) -> ReplayFault {
    pair.readiness.abandon_block(block.number);
    ReplayFault::TransitionFailed { block, detail }
}

/// Derives a transition-level negative frame and drives it through a full execution.
///
/// The opposite claim to [`check_mutations`], and it needs the opposite setup. An admission
/// mutation is evidence that a block was refused before the validator touched anything. This one
/// has to be *admitted* — well formed, consensus-legal against the parent, every signature
/// recoverable — and is evidence only if what refuses it is a rule that cannot be evaluated
/// without executing the block. Until this ran, no test in this repository had ever reached one:
/// the recorded corpus contains no such block, and every manufactured one was stopped earlier.
///
/// The probe is assembled, not cloned. `CoordinatedPair` is deliberately not `Clone` — its caches
/// are the generation this run is standing on, not a value to copy — so the fork is a cloned
/// readiness tracker over the pair's own caches, driven with `TrieCacheDisposition::Discard`. The
/// readiness tracker is the one part a failed transition really does move (`admit_block` puts it
/// in `Applying`, and `fail_applied_block` then latches it terminally blocked), which is exactly
/// why the probe needs its own. The caches need no copy because a failed transition leaves both
/// at the parent generation — the invariant `fail_applied_block` documents, and the one the
/// fingerprint comparison below exists to check rather than to assume.
///
/// Returns whether the pair moved. `Discard` covers the trie cache only — the flat cache is
/// advanced by the transition itself — so an accepted mutation is not merely a coverage failure
/// but a corrupted validator, and the caller stops rather than carrying it into the next commit.
#[expect(clippy::too_many_arguments)]
fn check_transition_mutations<C>(
    admission: &UntrustedAdmission<'_, ChainSpec, C>,
    pair: &mut CoordinatedPair,
    rules: ValidatorRules<'_, EthEvmConfig<ChainSpec>, C>,
    cache_policy_id: alloy_primitives::B256,
    reexec_limits: &SidecarReexecLimits,
    payload: &alloy_rpc_types_engine::ExecutionData,
    sidecar: &PartialStatelessSidecar,
    report: &mut ReplayReport,
    label: &str,
) -> bool
where
    C: reth_consensus::FullConsensus<reth_ethereum_primitives::EthPrimitives>
        + reth_consensus::Consensus<
            alloy_consensus::Block<reth_ethereum_primitives::TransactionSigned>,
        > + ?Sized,
{
    let mut compromised = false;
    for mutation in TransitionMutation::ALL {
        let mutated = match mutation.apply(payload) {
            Ok(mutated) => mutated,
            Err(err) => {
                report
                    .mutation_failures
                    .push(format!("{label}/{}: could not derive: {err}", mutation.as_str()));
                continue
            }
        };
        let mutated_sidecar = mutation.rebind_sidecar(sidecar, mutated.payload.block_hash());
        report.transition_mutations_checked += 1;

        // Admission is the *precondition* here, not the test. A mutation refused this early has
        // told us nothing about the rule it was built for, so it is reported as a coverage
        // failure rather than as a rejection.
        let admitted = match admission.admit(mutated, pair.accepted_parent()) {
            Ok(admitted) => admitted,
            Err(err) => {
                report.mutation_failures.push(format!(
                    "{label}/{}: refused as {} before the block could be executed: {err}",
                    mutation.as_str(),
                    err.class()
                ));
                continue
            }
        };

        let before = (pair.fingerprint(), pair.lifecycle_fingerprint());

        let mut probe_readiness = pair.readiness.clone();
        let block_ctx = block_context(&admitted.block);
        if let BlockAdmission::Rejected(reason) = admit_block(&mut probe_readiness, &block_ctx) {
            report.mutation_failures.push(format!(
                "{label}/{}: the readiness tracker refused it before execution: {reason:?}",
                mutation.as_str()
            ));
            continue
        }

        let outcome = verify_and_apply_sidecar(
            rules,
            &admitted.block,
            &mut pair.cache,
            &mutated_sidecar,
            cache_policy_id,
            reexec_limits,
            &mut pair.trie_cache,
            TrieCacheDisposition::Discard,
        );
        match outcome {
            Ok(_) => {
                report.mutation_failures.push(format!(
                    "{label}/{}: the transition accepted a block that must have been refused as {}",
                    mutation.as_str(),
                    mutation.expected_class()
                ));
                compromised = true;
            }
            Err(err) => {
                let detail = format!("{err:#}");
                if !detail.contains(POST_EXECUTION_REJECTION) {
                    report.mutation_failures.push(format!(
                        "{label}/{}: refused, but not by the post-execution rule it exists to \
                         reach: {detail}",
                        mutation.as_str()
                    ));
                }
            }
        }

        // The probe ends where a real failed commit ends: terminally blocked, watermark frozen at
        // the parent. Asserted rather than assumed, because a probe that quietly stayed usable
        // would mean the driver's own fault path leaves a tracker that lies about what it holds.
        probe_readiness.abandon_block(block_ctx.number);
        if !probe_readiness.is_blocked() {
            report.mutation_failures.push(format!(
                "{label}/{}: the probe's readiness survived a failed transition",
                mutation.as_str()
            ));
        }

        // And the pair the honest commit is about to use has to be untouched by all of it: the
        // same cache generation, reached by applying the same blocks. Both fingerprints, because
        // they answer different questions and a mutation could move either one.
        let after = (pair.fingerprint(), pair.lifecycle_fingerprint());
        if after != before {
            report.mutation_failures.push(format!(
                "{label}/{}: the mutation moved the real pair: {before:?} -> {after:?}",
                mutation.as_str()
            ));
            compromised = true;
        }
    }
    compromised
}

/// Derives negative frames from a recorded payload and checks the class each must produce.
///
/// Runs against the same pair the honest commit will use, and deliberately *before* it: admission
/// touches no validator state, so a rejected mutation leaves the pair exactly where it was, and
/// running them first means a mutation that wrongly succeeded would be caught rather than hidden
/// by the honest block that followed.
fn check_mutations<C>(
    admission: &UntrustedAdmission<'_, ChainSpec, C>,
    pair: &CoordinatedPair,
    payload: &alloy_rpc_types_engine::ExecutionData,
    report: &mut ReplayReport,
    label: &str,
) where
    C: reth_consensus::Consensus<
            alloy_consensus::Block<reth_ethereum_primitives::TransactionSigned>,
        > + ?Sized,
{
    for mutation in Mutation::ALL {
        let mutated = match mutation.apply(payload) {
            Ok(mutated) => mutated,
            Err(err) => {
                report
                    .mutation_failures
                    .push(format!("{label}/{}: could not derive: {err}", mutation.as_str()));
                continue
            }
        };
        report.mutations_checked += 1;
        match admission.admit(mutated, pair.accepted_parent()) {
            Ok(_) => report.mutation_failures.push(format!(
                "{label}/{}: admitted a block that must have been refused as {}",
                mutation.as_str(),
                mutation.expected_class()
            )),
            Err(err) if err.class() != mutation.expected_class() => {
                report.mutation_failures.push(format!(
                    "{label}/{}: refused as {} but must be {}: {err}",
                    mutation.as_str(),
                    err.class(),
                    mutation.expected_class()
                ));
            }
            Err(_) => {}
        }
    }

    // The one negative no frame can carry: a pair that cannot name a parent must refuse the block
    // rather than run the subset of rules that survive without one.
    match admission.admit(payload.clone(), None) {
        Ok(_) => report.mutation_failures.push(format!(
            "{label}/no_accepted_parent: admitted a block with no parent to check it against"
        )),
        Err(AdmissionError::NoAcceptedParent { .. }) => report.mutations_checked += 1,
        Err(err) => {
            report.mutations_checked += 1;
            report.mutation_failures.push(format!(
                "{label}/no_accepted_parent: refused as {} rather than no_accepted_parent: {err}",
                err.class()
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{fail_applied_block, AttemptTimer, FrameCosts, ReplayFault, ReplayReport};
    use alloy_primitives::B256;
    use partial_stateless::{readiness::BlockContext, CacheConfig, PartialTrieNodeCache};
    use partial_stateless_stream::BlockRef;
    use partial_stateless_validator::{
        admit_block, timings::ValidationPhaseTimings, BlockAdmission, CoordinatedPair,
    };

    fn pair() -> CoordinatedPair {
        let config = CacheConfig::default();
        CoordinatedPair {
            cache: config.new_cache(),
            trie_cache: PartialTrieNodeCache::new(),
            previous_generation: None,
            accepted_head: None,
            readiness: config.new_readiness_tracker(),
        }
    }

    fn ctx(number: u64) -> BlockContext {
        BlockContext {
            number,
            hash: B256::with_last_byte(number as u8),
            parent_hash: B256::with_last_byte(number as u8 - 1),
            state_root: B256::with_last_byte(0x55),
        }
    }

    fn costs(frame_decode_us: Option<u64>) -> FrameCosts {
        FrameCosts { sequence: 7, delivery_us: Some(12), frame_decode_us, validation_open: None }
    }

    /// Costs whose boundary opened `micros` ago, the way a real frame's read path opens it.
    fn costs_opened(frame_decode_us: Option<u64>, micros: u64) -> FrameCosts {
        FrameCosts {
            sequence: 7,
            delivery_us: Some(12),
            frame_decode_us,
            validation_open: std::time::Instant::now()
                .checked_sub(std::time::Duration::from_micros(micros)),
        }
    }

    /// The contract the follower's `blocks.last()` reads under: one entry per attempt, whatever
    /// the attempt produced, keyed by the frame's sequence rather than the block's height.
    #[test]
    fn every_attempt_leaves_exactly_one_timing_entry() {
        let mut report = ReplayReport::default();
        AttemptTimer::open(100, costs(Some(5))).finish("rejected", &mut report);
        AttemptTimer::open(101, costs(Some(5))).finish("accepted", &mut report);

        assert_eq!(report.blocks.len(), 2);
        assert_eq!(report.blocks[0].verdict, "rejected");
        assert_eq!(report.blocks[0].sequence, 7);
        assert_eq!(report.blocks[0].delivery_us, Some(12));
    }

    /// `phases` holds only mutually exclusive leaves, so their sum stays inside the wall and the
    /// residual is the boundary's honest remainder.
    #[test]
    fn the_disjoint_phases_never_exceed_the_wall_boundary() {
        let mut report = ReplayReport::default();
        // The boundary opened 10 ms ago, at the decode the 40 µs leaf times — a leaf inside the
        // wall, the way a carried `validation_open` puts it there.
        let timer = AttemptTimer::open(1, costs_opened(Some(40), 10_000));
        timer.finish("accepted", &mut report);

        let timing = &report.blocks[0];
        assert!(timing.standalone_validation_us >= timing.phases.sum_us());
        assert_eq!(
            timing.unattributed_validation_us,
            timing.standalone_validation_us - timing.phases.sum_us()
        );
        assert_eq!(report.timing_anomalies, 0);
    }

    /// The primary is one wall reading opened at the frame decode — not the decode cost added to
    /// a separately-started segment. A carried open instant is what the boundary reads from, and
    /// the frame-decode leaf changing must not move the primary.
    #[test]
    fn the_primary_is_one_wall_clock_from_the_carried_open_instant() {
        let mut report = ReplayReport::default();
        AttemptTimer::open(1, costs_opened(Some(40), 10_000)).finish("accepted", &mut report);
        AttemptTimer::open(2, costs_opened(Some(4_000), 10_000)).finish("accepted", &mut report);

        let (a, b) = (&report.blocks[0], &report.blocks[1]);
        assert!(a.standalone_validation_us >= 10_000, "the wall covers the carried open");
        assert!(b.standalone_validation_us >= 10_000);
        // Both walls are ~10 ms; the hundredfold difference in the decode leaf lives inside them.
        let spread = a.standalone_validation_us.abs_diff(b.standalone_validation_us);
        assert!(spread < 5_000, "the leaf is inside the wall, never added to it: {spread}");
    }

    /// A phase sum past its own wall is timer misbehaviour, and it is counted as such rather than
    /// clamped into a negative that saturation would silently hide.
    #[test]
    fn a_phase_sum_past_the_wall_is_an_anomaly_not_a_negative() {
        let mut report = ReplayReport::default();
        let mut timer = AttemptTimer::open(1, costs(Some(10)));
        timer.input_decode_us = Some(1_000_000);
        timer.finish("rejected", &mut report);

        assert_eq!(report.timing_anomalies, 1);
        assert_eq!(report.blocks[0].unattributed_validation_us, 0);
    }

    /// The primary freezes at the pair commit; the oracle comparison after it is harness work and
    /// must not widen the boundary however long it takes.
    #[test]
    fn the_oracle_compare_stays_outside_the_frozen_boundary() {
        let mut report = ReplayReport::default();
        let mut timer = AttemptTimer::open(1, costs(Some(3)));
        timer.close_validation();
        std::thread::sleep(std::time::Duration::from_millis(20));
        timer.oracle_compare_us = Some(20_000);
        timer.finish("accepted", &mut report);

        let timing = &report.blocks[0];
        assert!(
            timing.standalone_validation_us < 20_000,
            "the boundary was frozen before the sleep stood in for the compare"
        );
        assert_eq!(timing.oracle_compare_us, Some(20_000));
    }

    /// The older in-process primary must stay derivable from a v2 record: the six components it
    /// sums are all published leaves, with the sidecar decode standing where `deserialize_us`
    /// stood in the paired schema.
    #[test]
    fn the_old_primary_is_reconstructible_from_a_v2_record() {
        let mut core = ValidationPhaseTimings {
            context_check_us: 11,
            witness_self_consistency_us: 22,
            materialize_us: 33,
            provider_setup_us: 44,
            evm_us: 55,
            ..Default::default()
        };
        core.set_deserialize_us(7);

        let mut report = ReplayReport::default();
        let mut timer = AttemptTimer::open(1, costs(Some(1)));
        timer.sidecar_decode_us = Some(7);
        timer.core = Some(Box::new(core));
        timer.finish("accepted", &mut report);

        let timing = &report.blocks[0];
        let reconstructed = timing.phases.sidecar_decode_us.unwrap() +
            timing.phases.context_check_us.unwrap() +
            timing.phases.witness_self_consistency_us.unwrap() +
            timing.phases.materialize_us.unwrap() +
            timing.phases.provider_setup_us.unwrap() +
            timing.phases.evm_us.unwrap();
        assert_eq!(reconstructed, 7 + 11 + 22 + 33 + 44 + 55);
        assert_eq!(timing.derived.state_access_execution_us, Some(reconstructed));
    }

    /// A rejected attempt reports the phases that ran and `null` for the rest — never zero, which
    /// would claim the unrun work was free.
    #[test]
    fn a_rejected_attempt_reports_only_the_phases_that_ran() {
        let mut report = ReplayReport::default();
        let mut timer = AttemptTimer::open(9, costs(Some(4)));
        timer.input_decode_us = Some(6);
        timer.finish("rejected", &mut report);

        let timing = &report.blocks[0];
        assert_eq!(timing.phases.input_decode_us, Some(6));
        assert_eq!(timing.phases.sidecar_decode_us, None, "unrun is null, never zero");
        assert_eq!(timing.admission_us, None);
        assert_eq!(timing.transition_us, None);
        assert!(timing.details.is_none());
    }

    /// The leak this boundary closes, found while extracting the standalone path: a
    /// post-execution rejection preserved both
    /// caches but left readiness in transient `Applying`, which only the ExEx's fail-stop made
    /// safe. A standalone driver outlives the rejection, so the stop must be explicit, terminal,
    /// and observable — and it must not have touched what the rejection promised to preserve.
    #[test]
    fn a_post_admission_failure_moves_readiness_to_terminal_blocked() {
        let mut pair = pair();
        let block = ctx(100);
        assert!(matches!(admit_block(&mut pair.readiness, &block), BlockAdmission::Admitted(_)));
        assert_eq!(pair.readiness.state().label(), "applying", "the leak's starting point");
        let parent_fingerprint = pair.fingerprint();

        let fault = fail_applied_block(
            &mut pair,
            BlockRef { number: block.number, hash: block.hash },
            "witness self-consistency refused".to_string(),
        );

        assert!(matches!(fault, ReplayFault::TransitionFailed { .. }));
        assert_eq!(pair.readiness.state().label(), "blocked", "terminal, not transient");
        assert_eq!(
            pair.fingerprint(),
            parent_fingerprint,
            "the boundary reports the stop; it does not touch the caches"
        );
        assert_eq!(
            pair.readiness.first_gap(),
            Some(block.number),
            "the acknowledgement watermark froze at the failed block"
        );
        assert!(
            matches!(admit_block(&mut pair.readiness, &ctx(101)), BlockAdmission::Rejected(_)),
            "nothing runs against a pair that stopped"
        );
    }

    /// The fault on the very last commit is the case the old code got wrong silently: with no
    /// next block to trip over the stranded `Applying`, the report was the only witness.
    #[test]
    fn a_fault_on_the_final_commit_still_leaves_the_pair_blocked() {
        let mut pair = pair();
        let block = ctx(200);
        assert!(matches!(admit_block(&mut pair.readiness, &block), BlockAdmission::Admitted(_)));

        fail_applied_block(
            &mut pair,
            BlockRef { number: block.number, hash: block.hash },
            "anchor mismatch".to_string(),
        );

        assert_eq!(pair.readiness.state().label(), "blocked");
    }
}
