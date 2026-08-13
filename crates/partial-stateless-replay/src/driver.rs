//! Restoring a pair from a recorded checkpoint and running the recorded commits through it.
//!
//! The sequence per commit is the standalone path in full, from bytes: decode the Engine payload,
//! admit it against the pair's *own* accepted parent, and only then run the transition. Nothing is
//! taken from the frame that a live validator would not have. In particular the parent header is
//! read from the pair rather than from the commit, which is the rule S1b established and the one a
//! replay is most tempted to break, because the frame is right there and it is correct.

use crate::{
    compare::{block_label, compare_accepted, compare_rejected, Disagreement},
    mutate::Mutation,
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
    admit_block, block_context, verify_and_apply_sidecar, AdmissionError, BlockAdmission,
    CoordinatedPair, PayloadProvenance, SidecarReexecLimits, TrieCacheDisposition,
    UntrustedAdmission, ValidatorRules,
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
}

impl Default for ReplayOptions {
    fn default() -> Self {
        Self {
            limit: None,
            mutations: true,
            frame_limits: FrameLimits::default(),
            reexec_limits: SidecarReexecLimits::default(),
            force_restore_at: None,
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
    /// Total admission wall time across every commit, in microseconds.
    pub admission_us: u64,
    /// Total transition wall time across every commit, in microseconds.
    pub transition_us: u64,
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

/// What one replayed block cost.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct BlockTiming {
    /// Height.
    pub number: u64,
    /// Decode, payload layout, block hash, sender recovery, and pre-execution consensus.
    pub admission_us: u64,
    /// The cache transition: witness materialization, execution, root, retention, anchor.
    pub transition_us: u64,
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

    while let Some(frame) = spool.next_frame()? {
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
                    BatchPhase::AwaitingResync { manifest: found, target_ancestor: None }
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
                    BatchPhase::AwaitingResync { manifest, target_ancestor } => {
                        (manifest, CollectPurpose::Resync { target_ancestor })
                    }
                    BatchPhase::Collecting { manifest, .. } => {
                        report.failures.push(format!(
                            "a checkpoint arrived at sequence {sequence} while the previous one's \
                             chunks were incomplete"
                        ));
                        (manifest, CollectPurpose::Resync { target_ancestor: None })
                    }
                    BatchPhase::Live { manifest, state, announced, pending_tip } => {
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
                                info!(
                                    target: "ps_replay",
                                    sequence,
                                    ancestor = ancestor.number,
                                    "Installing the recovery checkpoint rather than skimming it, \
                                     as a consumer with no retained generation would have to"
                                );
                                report.last_verified = state.history.tip().map(|tip| tip.number);
                                (
                                    manifest,
                                    CollectPurpose::Resync { target_ancestor: Some(ancestor) },
                                )
                            }
                            None => {
                                report.failures.push(format!(
                                    "an unannounced checkpoint arrived at sequence {sequence}; \
                                     the grammar has no mid-stream checkpoint without a reorg or \
                                     reset in front of it"
                                ));
                                report.last_verified = state.history.tip().map(|tip| tip.number);
                                (manifest, CollectPurpose::Resync { target_ancestor: None })
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
                finish_collection_if_complete(collecting, &mut report)?
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
                )?
            }
            (phase, StreamEvent::SnapshotChunk(_)) => {
                report.failures.push(format!(
                    "a snapshot chunk arrived at sequence {sequence} with no checkpoint expecting it"
                ));
                phase
            }
            (
                BatchPhase::Live { manifest, mut state, announced, mut pending_tip },
                StreamEvent::Commit(commit),
            ) => {
                let (input, oracle) = commit.split();
                let block = input.block;
                match replay_commit(&mut state, input, &oracle, options, &mut report) {
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
                BatchPhase::Live { manifest, state, announced, pending_tip }
            }
            (BatchPhase::AwaitingResync { manifest, target_ancestor }, StreamEvent::Commit(_)) => {
                report.skipped_awaiting_resync += 1;
                BatchPhase::AwaitingResync { manifest, target_ancestor }
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
                        BatchPhase::AwaitingResync { manifest, target_ancestor: Some(ancestor) }
                    }
                    // Well-formed but about a branch this driver never held. It carries no
                    // authority: no target, so no recovery under it can be called continuous, and
                    // the branch it interrupted stays counted as unfinished.
                    ReorgOutcome::Unbound { ancestor, depth, detail } => {
                        report.reorgs_inapplicable += 1;
                        warn_inapplicable(ancestor, depth, &detail, false);
                        report.last_verified = state.history.tip().map(|tip| tip.number);
                        BatchPhase::AwaitingResync { manifest, target_ancestor: None }
                    }
                    ReorgOutcome::Malformed { detail } => {
                        report.failures.push(format!(
                            "the reorg at sequence {sequence} is not a reorg this driver can \
                             evaluate: {detail}"
                        ));
                        report.last_verified = state.history.tip().map(|tip| tip.number);
                        BatchPhase::AwaitingResync { manifest, target_ancestor: None }
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
                        BatchPhase::AwaitingResync { manifest, target_ancestor: None }
                    }
                    BatchPhase::AwaitingCheckpoint { manifest } |
                    BatchPhase::AwaitingResync { manifest, .. } |
                    BatchPhase::Collecting { manifest, .. } => {
                        BatchPhase::AwaitingResync { manifest, target_ancestor: None }
                    }
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
        bytes = spool.bytes(),
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
        /// The tip an applied reorg said the winning branch would reach.
        pending_tip: Option<BlockRef>,
    },
    /// Verification stopped at a discontinuity; only a checkpoint can restart it.
    AwaitingResync {
        manifest: Manifest,
        /// The block a recovery checkpoint has to land on to be continuous.
        target_ancestor: Option<BlockRef>,
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
    Resync { target_ancestor: Option<BlockRef> },
}

/// Restores or cross-checks a checkpoint once every chunk it declared has arrived.
fn finish_collection_if_complete(
    phase: BatchPhase,
    report: &mut ReplayReport,
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
                // Recorded, and then the checkpoint is installed rather than skipped — the same
                // answer the follower gives. Carrying on from a generation the operator-trusted
                // checkpoint source contradicts would make every later block a comparison against
                // disputed state, and a reader could not tell an independent second finding from
                // the first one cascading. Installing isolates the finding to the block it is
                // about; the interval it covers is an explicit reset, never a continuous recovery.
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
            Ok(BatchPhase::Live { manifest, state, announced: None, pending_tip })
        }
        CollectPurpose::Resync { target_ancestor } => {
            let state = restore(&manifest, &checkpoint, &chunks)?;
            let continuous = target_ancestor.is_some_and(|target| target == checkpoint.block) &&
                report.skipped_awaiting_resync == 0;
            let unverified = (!continuous)
                .then(|| {
                    report
                        .last_verified
                        .filter(|last| *last < checkpoint.block.number)
                        .map(|last| (last + 1, checkpoint.block.number))
                })
                .flatten();
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
            report.resyncs.push(ResyncRecord {
                at_sequence: checkpoint_sequence,
                block: checkpoint.block.number,
                continuous,
                unverified,
                commits_skipped: report.skipped_awaiting_resync,
            });
            report.skipped_awaiting_resync = 0;
            Ok(BatchPhase::Live {
                manifest,
                state: Box::new(state),
                announced: None,
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
        BatchPhase::Live { pending_tip: Some(tip), .. } => {
            report.winning_branch_incomplete += 1;
            warn!(
                target: "ps_replay",
                tip = tip.number,
                "The corpus ends before the winning branch reached the tip the reorg announced"
            );
        }
        BatchPhase::AwaitingManifest |
        BatchPhase::AwaitingCheckpoint { .. } |
        BatchPhase::Live { .. } => {}
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
        history: VerifiedHistory::restored_at(checkpoint.block, checkpoint.state_root),
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
    report: &mut ReplayReport,
) -> CommitOutcome {
    match input.payload_provenance {
        PayloadProvenance::Witnessed => report.witnessed += 1,
        PayloadProvenance::Reconstructed => report.reconstructed += 1,
        PayloadProvenance::Absent => report.absent += 1,
    }
    let label = block_label(input.block);

    let payload = match input.payload() {
        Ok(Some(payload)) => payload,
        Ok(None) => {
            report.failures.push(format!(
                "{label}: the commit carries no payload, so admission could not run on it"
            ));
            return CommitOutcome::Rejected
        }
        Err(err) => {
            report.failures.push(format!("{label}: recorded payload did not parse: {err}"));
            return CommitOutcome::Rejected
        }
    };
    let sidecar: PartialStatelessSidecar = match bincode::deserialize(&input.sidecar) {
        Ok(sidecar) => sidecar,
        Err(err) => {
            report.failures.push(format!("{label}: recorded sidecar did not decode: {err}"));
            return CommitOutcome::Rejected
        }
    };

    let admission = UntrustedAdmission::new(state.chain_spec.as_ref(), &state.consensus);

    if options.mutations && input.payload_provenance.is_load_bearing() {
        check_mutations(&admission, &state.pair, &payload, report, &label);
    }

    // The parent comes from the pair and never from the frame. A producer that supplied the parent
    // would be choosing the timestamp, gas limit, and base fee its own block is measured against.
    let admitted = match admission.admit(payload, state.pair.accepted_parent()) {
        Ok(admitted) => admitted,
        Err(err) => {
            let disagreements = compare_rejected(oracle, err.class());
            report.failures.push(format!(
                "{label}: the replay refused a block the recording accepted: {err} ({})",
                err.class()
            ));
            report.disagreements.extend(disagreements.into_iter().map(|d| (input.block, d)));
            return CommitOutcome::Rejected
        }
    };
    let admission_us = admitted.timings.total_us();
    report.admission_us += admission_us;

    let block_ctx = block_context(&admitted.block);
    if let BlockAdmission::Rejected(reason) = admit_block(&mut state.pair.readiness, &block_ctx) {
        // The refusal itself latched the tracker `Blocked`; the caches were never touched.
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
    report.blocks.push(BlockTiming { number: input.block.number, admission_us, transition_us });

    let validated = match validated {
        Ok(validated) => validated,
        Err(err) => {
            return CommitOutcome::Fault(fail_applied_block(
                &mut state.pair,
                input.block,
                format!("{err:#}"),
            ))
        }
    };
    let mut outcome = validated.outcome;
    let displaced = outcome.displaced_trie_cache.take();
    state.pair.commit_transition(displaced, &block_ctx, admitted.block.clone_sealed_header(), true);
    // Recorded from this replay's own execution, before the oracle is consulted, so that a reorg
    // arriving later authenticates its target against what this process verified rather than
    // against what the producer said about it.
    state.history.record(input.block, outcome.state_root);

    let disagreements = compare_accepted(oracle, &outcome, &state.pair);
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
    use super::{fail_applied_block, ReplayFault};
    use alloy_primitives::B256;
    use partial_stateless::{readiness::BlockContext, CacheConfig, PartialTrieNodeCache};
    use partial_stateless_stream::BlockRef;
    use partial_stateless_validator::{admit_block, BlockAdmission, CoordinatedPair};

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

    /// The leak S1 recorded and this boundary closes: a post-execution rejection preserved both
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
