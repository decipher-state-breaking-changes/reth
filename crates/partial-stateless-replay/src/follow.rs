//! The live consumer: verdicts on a stream a producer is still writing.
//!
//! This is the S3 exit-gate behaviour as a state machine. The follower verifies the manifest's
//! identity before it accepts anything, the checkpoint and its snapshot before it accepts a
//! commit, and exactly `H + 1` as the first commit — and it never hides a delivery violation as
//! a drop. A gap, a duplicate claim on a sequence, an undecodable frame, an epoch change, a
//! producer reset, or a reorg frame all take it to `NeedsSnapshot`: verdicts stop, and only a
//! fresh checkpoint that verifies end to end restarts them at its own `H′ + 1`.
//!
//! **`NeedsSnapshot` waits for the producer; it does not synthesize recovery.** Today's producer
//! writes one checkpoint per stream, so in a live run the state is effectively terminal — the
//! machinery below is proven against synthetic spools carrying a mid-stream second checkpoint,
//! and the producer-side re-checkpoint that would exercise it live is S4's recovery protocol.
//!
//! **A quiet spool is not a dead producer.** The follower cannot tell "no new block yet" from
//! "the producer was killed" by looking at files; an `End` frame is how a producer says it
//! stopped, and its absence is what "cut" means. So the default is to wait forever, and a harness
//! that killed the producer on purpose passes `idle_timeout` and judges the spool offline.

use crate::{
    driver::{
        chain_spec_for, config_for, cross_check_recovery_checkpoint, decode_accepted_head,
        replay_commit, restore, CommitOutcome, ReplayOptions, ReplayReport, ReplayState,
    },
    reorg::{apply_reorg, warn_inapplicable, ReorgOutcome},
    tail::{SpoolTail, TailEvent, TailFault},
};
use alloy_primitives::{Keccak256, B256};
use partial_stateless_stream::{
    BlockRef, Checkpoint, EndKind, FrameKind, FrameLimits, Manifest, ResetReason, SnapshotChunk,
    StreamEvent, DEFAULT_MAX_SNAPSHOT_BYTES,
};
use partial_stateless_validator::{PayloadProvenance, SidecarReexecLimits};
use std::{
    io::Write,
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime},
};
use tracing::{error, info, warn};

/// How a follower runs, and where its verdicts go.
#[derive(Debug, Clone)]
pub struct FollowOptions {
    /// Delay between polls when the spool has nothing new.
    pub poll: Duration,
    /// Stop after this many verified blocks. A harness bound; `None` follows indefinitely.
    pub max_blocks: Option<u64>,
    /// Stop after this long without a new frame. `None` — the default — waits forever, because
    /// a follower cannot distinguish a dead producer from a quiet chain and must not guess.
    pub idle_timeout: Option<Duration>,
    /// Derive negative frames from witnessed commits, as the batch replay does.
    ///
    /// Off by default in follow mode: a live follower's job is the verdict stream, and rejection
    /// coverage is the recorded corpus's job.
    pub mutations: bool,
    /// Bounds on frame decoding.
    pub frame_limits: FrameLimits,
    /// Bounds on sidecar witness decoding.
    pub reexec_limits: SidecarReexecLimits,
    /// JSONL verdict stream, one line per block and per state transition.
    pub verdicts: Option<PathBuf>,
    /// Atomically rewritten consumer watermark, outside the spool directory.
    pub ack: Option<PathBuf>,
    /// Label stamped into every record.
    pub label: String,
}

impl Default for FollowOptions {
    fn default() -> Self {
        Self {
            poll: Duration::from_millis(200),
            max_blocks: None,
            idle_timeout: None,
            mutations: false,
            frame_limits: FrameLimits::default(),
            reexec_limits: SidecarReexecLimits::default(),
            verdicts: None,
            ack: None,
            label: "unlabelled".to_string(),
        }
    }
}

/// Why the follower stopped publishing verdicts and started waiting for a checkpoint.
///
/// A superset of the producer's [`ResetReason`]: the first six arrive in `Reset` frames or are
/// detected as delivery faults, and the last three are violations only a consumer can see. This
/// lives in the follow driver rather than in the cache readiness state machine because every
/// trigger is a fact about *delivery* — the pair itself is still sound at its last verified
/// block. S4's deep-reorg recovery is what lifts it into the standalone state machine, once
/// `target_ancestor` means something.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeedsSnapshotReason {
    /// A frame is missing between two that exist.
    Gap,
    /// The producer restarted its stream; sequence numbers restarted with it.
    EpochChange,
    /// Two frames claimed the same sequence.
    DuplicateConflict,
    /// A frame arrived below the consumer's watermark.
    OutOfOrder,
    /// The producer's own queue overflowed.
    Overflow,
    /// The producer's state moved somewhere no incremental event can express.
    SnapshotRequired,
    /// A visible frame did not decode, or its name and header disagree.
    Undecodable,
    /// A frame arrived where the stream's grammar does not allow it.
    ProtocolViolation,
    /// The checkpoint carries no usable accepted head, so `H + 1` could never be admitted.
    HeadlessCheckpoint,
}

impl NeedsSnapshotReason {
    /// Stable name for records.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Gap => "gap",
            Self::EpochChange => "epoch_change",
            Self::DuplicateConflict => "duplicate_conflict",
            Self::OutOfOrder => "out_of_order",
            Self::Overflow => "overflow",
            Self::SnapshotRequired => "snapshot_required",
            Self::Undecodable => "undecodable",
            Self::ProtocolViolation => "protocol_violation",
            Self::HeadlessCheckpoint => "headless_checkpoint",
        }
    }
}

impl From<ResetReason> for NeedsSnapshotReason {
    fn from(reason: ResetReason) -> Self {
        match reason {
            ResetReason::Gap => Self::Gap,
            ResetReason::EpochChange => Self::EpochChange,
            ResetReason::DuplicateConflict => Self::DuplicateConflict,
            ResetReason::OutOfOrder => Self::OutOfOrder,
            ResetReason::Overflow => Self::Overflow,
            ResetReason::SnapshotRequired => Self::SnapshotRequired,
        }
    }
}

/// How a follow run ended.
#[derive(Debug, Clone)]
pub enum FollowOutcome {
    /// The producer closed the stream. Orderly termination, not success — the kind says why,
    /// and `before_checkpoint` marks a stream that closed before it ever opened.
    Ended {
        /// The producer's close kind.
        kind: EndKind,
        /// True when no checkpoint was ever published; the stream carried nothing verifiable.
        before_checkpoint: bool,
    },
    /// The pair can go no further; readiness is terminally `Blocked` and verdicts stopped.
    Faulted {
        /// What stopped it.
        detail: String,
    },
    /// The `max_blocks` harness bound was reached.
    MaxBlocks,
    /// The `idle_timeout` harness bound elapsed with no new frame.
    IdleTimeout {
        /// The state the follower was waiting in, `needs_snapshot` being the one a harness
        /// treats as "recovery never came".
        waiting_in: &'static str,
    },
}

/// What one follow run did.
#[derive(Debug)]
pub struct FollowReport {
    /// How the run ended.
    pub outcome: FollowOutcome,
    /// The per-commit counters, shared with the batch replay.
    pub replay: ReplayReport,
    /// Blocks verified and published with an `accepted` verdict.
    pub blocks_verified: u64,
    /// Pairs restored from a checkpoint, the first bootstrap included.
    pub restores: u64,
    /// Times the follower entered `NeedsSnapshot`.
    pub needs_snapshot_entries: u64,
    /// Commit frames skipped while waiting out a `NeedsSnapshot`, recorded rather than silently
    /// discarded.
    pub commits_skipped_in_recovery: u64,
    /// The most recent reason verdicts stopped, when they did.
    pub last_needs_snapshot: Option<NeedsSnapshotReason>,
    /// The last block that verified cleanly.
    pub last_verified: Option<BlockRef>,
    /// Reorgs this follower undid itself, without stopping verdicts.
    pub reorgs_applied: u64,
    /// Reverts it undid, counted apart because a revert has no branch to follow.
    pub reverts_applied: u64,
    /// Recovery checkpoints the producer published for a reorg already applied, checked against
    /// the follower's own generation and not installed.
    pub checkpoints_skimmed: u64,
    /// Restores that landed on the exact block recovery asked for.
    pub restores_continuous: u64,
    /// Restores that landed anywhere else — sound again, with an interval nothing validated.
    pub restores_reset: u64,
    /// Winning branches that reached the tip the reorg announced.
    pub winning_branches_completed: u64,
    /// Winning branches that did not, whether the stream ended or the chain moved again.
    pub winning_branches_incomplete: u64,
    /// Canonical intervals no verdict covers, in the order they opened.
    pub unverified_intervals: Vec<(u64, u64)>,
}

impl FollowReport {
    /// Whether every verdict published agreed with the recording and nothing faulted.
    pub fn agreed(&self) -> bool {
        self.replay.agreed() && !matches!(self.outcome, FollowOutcome::Faulted { .. })
    }

    /// Whether every canonical block this stream carried was actually verified.
    ///
    /// The axis `agreed` deliberately does not cover. A reorg the pair undid itself costs nothing
    /// here; one it could not, a recovery that landed somewhere other than the block it asked
    /// for, commits that went by during an outage, and a winning branch that never arrived all
    /// leave a hole, and a run can agree everywhere it looked and still have one.
    pub fn continuous(&self) -> bool {
        self.commits_skipped_in_recovery == 0 &&
            self.restores_reset == 0 &&
            self.winning_branches_incomplete == 0 &&
            self.unverified_intervals.is_empty()
    }
}

/// Follows a live spool and publishes a verdict per block.
pub fn follow(dir: &Path, options: &FollowOptions) -> eyre::Result<FollowReport> {
    Follower::new(dir, options)?.run()
}

/// Where the follower is in the stream's grammar.
enum Phase {
    /// Nothing accepted yet; the first frame must be the manifest.
    AwaitingManifest,
    /// Identity verified; waiting for a checkpoint to bootstrap from.
    AwaitingCheckpoint { manifest: Manifest },
    /// A checkpoint arrived; its declared chunks are being collected.
    CollectingChunks { manifest: Manifest, checkpoint: Checkpoint, chunks: Vec<SnapshotChunk> },
    /// A pair is restored and verifying commits.
    Streaming {
        manifest: Manifest,
        state: Box<ReplayState>,
        /// The exact first child this pair may verify; `None` once it has.
        expected_child: Option<(u64, B256)>,
        /// The block an applied reorg returned to, while the producer's checkpoint for it may
        /// still be coming. Consumed by whichever arrives first: a checkpoint is cross-checked
        /// against this pair, a commit means the producer publishes none.
        announced: Option<BlockRef>,
        /// The tip an applied reorg said the winning branch would reach.
        ///
        /// Tracked apart from `announced` because they answer different questions and are
        /// resolved by different frames. Without it the field the producer fills in to say when
        /// the branch is complete would never be checked against what arrived.
        pending_tip: Option<BlockRef>,
    },
    /// A checkpoint the producer published for a reorg this follower already applied.
    ///
    /// Its chunks are read for their digest and dropped. Installing it would replace a generation
    /// this pair derived by undoing the block itself with one it has no reason to prefer — and
    /// the comparison against what it already holds is worth more than the restore, because the
    /// two were reached independently.
    SkimmingChunks(Box<SkimPhase>),
    /// Verdicts stopped; scanning for a fresh checkpoint to rebootstrap from.
    NeedsSnapshot {
        manifest: Manifest,
        reason: NeedsSnapshotReason,
        scan_from: u64,
        /// The block a recovery checkpoint has to land on for the recovery to be continuous.
        ///
        /// `None` when the discontinuity named none — a reset, an epoch change, a delivery fault
        /// — in which case any checkpoint that verifies is an explicit reset by definition.
        target_ancestor: Option<BlockRef>,
        /// The epoch this request belongs to, so a later one supersedes it rather than racing it.
        epoch: u64,
    },
}

impl Phase {
    const fn label(&self) -> &'static str {
        match self {
            Self::AwaitingManifest => "awaiting_manifest",
            Self::AwaitingCheckpoint { .. } => "awaiting_checkpoint",
            Self::CollectingChunks { .. } => "collecting_chunks",
            Self::Streaming { .. } => "streaming",
            Self::SkimmingChunks(_) => "skimming_chunks",
            Self::NeedsSnapshot { .. } => "needs_snapshot",
        }
    }
}

/// What a recovery scan was looking for when it found a checkpoint.
struct PendingRecovery {
    /// The block a continuous recovery has to land on, after supersession.
    target: Option<BlockRef>,
    /// Commit frames that went by between the discontinuity and the checkpoint.
    skipped: u64,
}

/// What a recovery restore is entitled to claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecoveryClass {
    /// The checkpoint landed on the exact block the discontinuity asked for, with nothing
    /// verifiable in between. Verification resumes with no interval unaccounted for.
    Continuous,
    /// Anywhere else. The pair is sound again and the run says so, but the blocks between the
    /// last verified one and the checkpoint were validated by nothing — and reporting that as
    /// continuous recovery is the one claim the format exists to prevent.
    Reset {
        /// The interval nothing validated, when there was one.
        unverified: Option<(u64, u64)>,
    },
}

impl RecoveryClass {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Continuous => "continuous",
            Self::Reset { .. } => "checkpoint_reset",
        }
    }
}

/// [`Phase::Streaming`]'s fields, destructured out of the match so the arm stays readable.
struct StreamingPhase {
    manifest: Manifest,
    state: Box<ReplayState>,
    expected_child: Option<(u64, B256)>,
    announced: Option<BlockRef>,
    pending_tip: Option<BlockRef>,
}

/// [`Phase::SkimmingChunks`]'s fields, for the same reason.
struct SkimPhase {
    manifest: Manifest,
    state: Box<ReplayState>,
    checkpoint: Checkpoint,
    pending_tip: Option<BlockRef>,
    received: u32,
    accumulated: u64,
    digest: Keccak256,
}

/// One step's instruction to the loop.
enum Step {
    Continue,
    Done(FollowOutcome),
}

struct Follower<'a> {
    options: &'a FollowOptions,
    replay_options: ReplayOptions,
    tail: SpoolTail,
    sink: VerdictSink,
    phase: Phase,
    replay: ReplayReport,
    blocks_verified: u64,
    restores: u64,
    needs_snapshot_entries: u64,
    commits_skipped_in_recovery: u64,
    reorgs_applied: u64,
    reverts_applied: u64,
    checkpoints_skimmed: u64,
    winning_branches_completed: u64,
    winning_branches_incomplete: u64,
    restores_continuous: u64,
    restores_reset: u64,
    /// What the scan that found the next checkpoint was looking for, so the restore can say
    /// whether it got it.
    pending_recovery: Option<PendingRecovery>,
    unverified_intervals: Vec<(u64, u64)>,
    last_needs_snapshot: Option<NeedsSnapshotReason>,
    last_verified: Option<BlockRef>,
    last_frame_at: Instant,
}

impl<'a> Follower<'a> {
    fn new(dir: &Path, options: &'a FollowOptions) -> eyre::Result<Self> {
        Ok(Self {
            options,
            replay_options: ReplayOptions {
                limit: None,
                mutations: options.mutations,
                frame_limits: options.frame_limits,
                reexec_limits: options.reexec_limits.clone(),
                // An offline forensic switch; a live follower has a producer to ask instead.
                force_restore_at: None,
            },
            tail: SpoolTail::new(dir, options.frame_limits),
            sink: VerdictSink::open(options)?,
            phase: Phase::AwaitingManifest,
            replay: ReplayReport::default(),
            blocks_verified: 0,
            restores: 0,
            needs_snapshot_entries: 0,
            commits_skipped_in_recovery: 0,
            reorgs_applied: 0,
            reverts_applied: 0,
            checkpoints_skimmed: 0,
            winning_branches_completed: 0,
            winning_branches_incomplete: 0,
            restores_continuous: 0,
            restores_reset: 0,
            pending_recovery: None,
            unverified_intervals: Vec::new(),
            last_needs_snapshot: None,
            last_verified: None,
            last_frame_at: Instant::now(),
        })
    }

    fn run(mut self) -> eyre::Result<FollowReport> {
        loop {
            if self.options.max_blocks.is_some_and(|max| self.blocks_verified >= max) {
                return Ok(self.into_report(FollowOutcome::MaxBlocks))
            }

            let step = if matches!(self.phase, Phase::NeedsSnapshot { .. }) {
                self.scan_for_recovery()?
            } else {
                match self.tail.poll() {
                    Ok(TailEvent::Frame(frame)) => {
                        self.last_frame_at = Instant::now();
                        let frame = *frame;
                        self.handle(frame.header.sequence, frame.event)?
                    }
                    Ok(TailEvent::Idle) => self.idle(),
                    Err(fault) => self.tail_fault(fault)?,
                }
            };
            match step {
                Step::Continue => {}
                Step::Done(outcome) => return Ok(self.into_report(outcome)),
            }
        }
    }

    /// One frame, against the grammar the current phase allows.
    fn handle(&mut self, sequence: u64, event: StreamEvent) -> eyre::Result<Step> {
        let phase = std::mem::replace(&mut self.phase, Phase::AwaitingManifest);
        match (phase, event) {
            (Phase::AwaitingManifest, StreamEvent::Manifest(manifest)) => {
                // Identity before anything: the chain, the genesis, and the policy the windows
                // derive. A wrong manifest is an operator error, not a stream fault.
                chain_spec_for(&manifest)?;
                config_for(&manifest)?;
                if manifest.first_sequence != 1 {
                    return Err(eyre::eyre!(
                        "manifest names first_sequence {}, and this format writes 1",
                        manifest.first_sequence
                    ))
                }
                info!(
                    target: "ps_follow",
                    chain_id = manifest.chain_id,
                    epoch = manifest.epoch,
                    producer = %manifest.producer,
                    "Stream identity verified; awaiting the checkpoint"
                );
                self.phase = Phase::AwaitingCheckpoint { manifest };
                Ok(Step::Continue)
            }
            (Phase::AwaitingManifest, other) => Err(eyre::eyre!(
                "the stream opens with a {} frame; a manifest must come first",
                kind_of(&other).as_str()
            )),

            (Phase::AwaitingCheckpoint { manifest }, StreamEvent::Checkpoint(checkpoint)) => {
                self.accept_checkpoint(manifest, checkpoint, sequence)
            }
            (Phase::AwaitingCheckpoint { manifest }, StreamEvent::End(end)) => {
                self.phase = Phase::AwaitingCheckpoint { manifest };
                self.check_end_numbering(&end.reason, end.last_sequence, sequence)?;
                Ok(Step::Done(FollowOutcome::Ended { kind: end.kind, before_checkpoint: true }))
            }
            (Phase::AwaitingCheckpoint { manifest }, StreamEvent::Manifest(_)) => Ok(self
                .enter_needs_snapshot(
                    manifest,
                    NeedsSnapshotReason::EpochChange,
                    "a second manifest arrived; sequence spaces are per-epoch",
                    sequence + 1,
                    None,
                )),
            (Phase::AwaitingCheckpoint { manifest }, other) => Ok(self.enter_needs_snapshot(
                manifest,
                NeedsSnapshotReason::ProtocolViolation,
                &format!("a {} frame arrived before any checkpoint", kind_of(&other).as_str()),
                sequence + 1,
                None,
            )),

            (
                Phase::CollectingChunks { manifest, checkpoint, mut chunks },
                StreamEvent::SnapshotChunk(chunk),
            ) => {
                if chunk.index as usize != chunks.len() {
                    return Ok(self.enter_needs_snapshot(
                        manifest,
                        NeedsSnapshotReason::ProtocolViolation,
                        &format!(
                            "snapshot chunk {} arrived where {} was expected",
                            chunk.index,
                            chunks.len()
                        ),
                        sequence + 1,
                        None,
                    ))
                }
                chunks.push(chunk);
                if chunks.len() < checkpoint.snapshot_chunks as usize {
                    self.phase = Phase::CollectingChunks { manifest, checkpoint, chunks };
                    return Ok(Step::Continue)
                }
                self.restore_pair(manifest, checkpoint, chunks, sequence)
            }
            (Phase::CollectingChunks { manifest, .. }, other) => Ok(self.enter_needs_snapshot(
                manifest,
                NeedsSnapshotReason::ProtocolViolation,
                &format!(
                    "a {} frame arrived while the snapshot's chunks were incomplete",
                    kind_of(&other).as_str()
                ),
                sequence + 1,
                None,
            )),

            (
                Phase::Streaming { manifest, state, expected_child, announced, pending_tip },
                event,
            ) => self.stream_frame(
                StreamingPhase { manifest, state, expected_child, announced, pending_tip },
                sequence,
                event,
            ),

            (Phase::SkimmingChunks(phase), event) => self.skim_frame(*phase, sequence, event),

            // The scan delivers only checkpoint frames while in recovery; anything else here is
            // unreachable by construction, and refusing it loudly beats mis-stating the machine.
            (Phase::NeedsSnapshot { .. }, other) => Err(eyre::eyre!(
                "a {} frame reached the recovery scan, which only reads checkpoints",
                kind_of(&other).as_str()
            )),
        }
    }

    /// One frame while a restored pair is live.
    fn stream_frame(
        &mut self,
        phase: StreamingPhase,
        sequence: u64,
        event: StreamEvent,
    ) -> eyre::Result<Step> {
        let StreamingPhase { manifest, mut state, expected_child, announced, mut pending_tip } =
            phase;
        match event {
            StreamEvent::Commit(commit) => {
                let (input, oracle) = commit.split();
                let block = input.block;
                // Captured before `replay_commit` consumes the input: the S3 claim is that the
                // validator consumed the *recorded* Engine payload, and provenance is how each
                // verdict line proves which kind of payload it was measured against.
                let provenance = input.payload_provenance;

                // Exactly H + 1, checked on the frame itself before anything is decoded. The
                // admission path would also refuse a wrong child, but as a consensus failure —
                // this is a *delivery* failure and is typed as one.
                if let Some((number, parent_hash)) = expected_child &&
                    (block.number != number || input.parent_hash != parent_hash)
                {
                    return Ok(self.enter_needs_snapshot(
                        manifest,
                        NeedsSnapshotReason::Gap,
                        &format!(
                            "the first commit after the checkpoint is block {} onto parent \
                             {:?}; the checkpoint requires block {number} onto {parent_hash:?}",
                            block.number, input.parent_hash
                        ),
                        sequence + 1,
                        None,
                    ))
                }

                let before_disagreements = self.replay.disagreements.len();
                let outcome = replay_commit(
                    &mut state,
                    input,
                    &oracle,
                    &self.replay_options,
                    &mut self.replay,
                );
                let new_disagreements = self.replay.disagreements.len() - before_disagreements;
                match outcome {
                    CommitOutcome::Compared => {
                        let verdict = if new_disagreements == 0 { "accepted" } else { "disagreed" };
                        self.blocks_verified += 1;
                        self.last_verified = Some(block);
                        self.sink.verdict(
                            verdict,
                            block,
                            sequence,
                            provenance,
                            new_disagreements,
                            &self.replay,
                        )?;
                        // The producer named the block that completes the branch it moved to.
                        // Reaching that height with a different hash means the frame and the
                        // delivery disagree about what replaced what, and no later commit can
                        // settle it.
                        if let Some(tip) = pending_tip &&
                            tip.number == block.number
                        {
                            if tip.hash != block.hash {
                                return Ok(self.enter_needs_snapshot(
                                    manifest,
                                    NeedsSnapshotReason::ProtocolViolation,
                                    &format!(
                                        "the winning branch reached {} as {:?}, but the reorg \
                                         announced {:?}",
                                        block.number, block.hash, tip.hash
                                    ),
                                    sequence + 1,
                                    None,
                                ))
                            }
                            self.winning_branches_completed += 1;
                            pending_tip = None;
                        }
                        self.phase = Phase::Streaming {
                            manifest,
                            state,
                            expected_child: None,
                            // Consumed: the producer publishes a checkpoint before the winning
                            // branch's commits or not at all.
                            announced: None,
                            pending_tip,
                        };
                        Ok(Step::Continue)
                    }
                    CommitOutcome::Rejected => {
                        // Fail closed: the pair did not advance, so every later commit would be
                        // measured against a parent this follower never verified. The batch
                        // driver keeps scanning for forensics; a live verdict stream must not.
                        let detail = self
                            .replay
                            .failures
                            .last()
                            .cloned()
                            .unwrap_or_else(|| format!("block {} was rejected", block.number));
                        self.sink.verdict(
                            "rejected",
                            block,
                            sequence,
                            provenance,
                            new_disagreements,
                            &self.replay,
                        )?;
                        self.sink.state(
                            "faulted",
                            "rejected_commit",
                            &detail,
                            self.last_verified,
                        )?;
                        Ok(Step::Done(FollowOutcome::Faulted { detail }))
                    }
                    CommitOutcome::Fault(fault) => {
                        let detail = fault.to_string();
                        self.sink.verdict(
                            "fault",
                            block,
                            sequence,
                            provenance,
                            new_disagreements,
                            &self.replay,
                        )?;
                        self.sink.state("faulted", "fault", &detail, self.last_verified)?;
                        error!(
                            target: "ps_follow",
                            block = block.number,
                            %detail,
                            readiness = state.pair.readiness.state().label(),
                            "The pair can go no further; no further verdict will be published"
                        );
                        Ok(Step::Done(FollowOutcome::Faulted { detail }))
                    }
                }
            }
            StreamEvent::Manifest(_) => Ok(self.enter_needs_snapshot(
                manifest,
                NeedsSnapshotReason::EpochChange,
                "a mid-stream manifest means the producer restarted its sequence space",
                sequence + 1,
                None,
            )),
            StreamEvent::Checkpoint(checkpoint) => match announced {
                // The producer's recovery checkpoint for a reorg this follower already undid.
                // Two implementations reached the same generation independently, so comparing
                // them is a live disagreement detector; the bytes are read for their digest and
                // dropped, because installing would replace derived state with state this
                // consumer has no reason to prefer.
                Some(ancestor) => {
                    self.begin_skim(manifest, state, checkpoint, ancestor, pending_tip, sequence)
                }
                // Unannounced: the grammar has no mid-stream checkpoint without a reset or reorg
                // in front of it. Fail closed — and let the recovery scan re-examine this very
                // frame, so a producer's recovery grammar still converges. The violation stays
                // in the record either way.
                None => Ok(self.enter_needs_snapshot(
                    manifest,
                    NeedsSnapshotReason::ProtocolViolation,
                    "an unannounced checkpoint arrived mid-stream",
                    sequence,
                    None,
                )),
            },
            StreamEvent::Reorg(found) => {
                if pending_tip.is_some() {
                    // Legitimate: the chain moved again before the previous branch finished. The
                    // hole is real all the same, and counted rather than forgotten.
                    self.winning_branches_incomplete += 1;
                }
                match apply_reorg(&mut state, &found) {
                    ReorgOutcome::Applied { ancestor, undone, revert, winning_tip } => {
                        if revert {
                            self.reverts_applied += 1;
                        } else {
                            self.reorgs_applied += 1;
                        }
                        // The follower no longer stands behind the abandoned block, and an
                        // external consumer of these lines has to be told which block left the
                        // chain — its verdict is still on record as a valid block.
                        self.last_verified = Some(ancestor);
                        self.sink.lifecycle(
                            if revert { "revert_applied" } else { "reorg_applied" },
                            ancestor,
                            &found.abandoned,
                            winning_tip,
                            self.last_verified,
                        )?;
                        info!(
                            target: "ps_follow",
                            ancestor = ancestor.number,
                            undone = undone.number,
                            revert,
                            "Undid the reorg against the retained generation; verdicts continue"
                        );
                        self.phase = Phase::Streaming {
                            manifest,
                            state,
                            expected_child: Some((ancestor.number + 1, ancestor.hash)),
                            announced: Some(ancestor),
                            pending_tip: winning_tip,
                        };
                        Ok(Step::Continue)
                    }
                    ReorgOutcome::Inapplicable { ancestor, depth, detail } => {
                        warn_inapplicable(ancestor, depth, &detail);
                        Ok(self.enter_needs_snapshot(
                            manifest,
                            NeedsSnapshotReason::SnapshotRequired,
                            &detail,
                            sequence + 1,
                            Some(ancestor),
                        ))
                    }
                    ReorgOutcome::Malformed { detail } => Ok(self.enter_needs_snapshot(
                        manifest,
                        NeedsSnapshotReason::ProtocolViolation,
                        &detail,
                        sequence + 1,
                        None,
                    )),
                }
            }
            StreamEvent::Reset(reset) => Ok(self.enter_needs_snapshot(
                manifest,
                reset.reason.into(),
                &reset.detail,
                sequence + 1,
                None,
            )),
            StreamEvent::End(end) => {
                if pending_tip.is_some() {
                    self.winning_branches_incomplete += 1;
                    warn!(
                        target: "ps_follow",
                        "The stream ended before the winning branch reached the tip the reorg \
                         announced"
                    );
                }
                self.phase =
                    Phase::Streaming { manifest, state, expected_child, announced, pending_tip };
                self.check_end_numbering(&end.reason, end.last_sequence, sequence)?;
                info!(
                    target: "ps_follow",
                    kind = end.kind.as_str(),
                    reason = %end.reason,
                    "The producer closed the stream"
                );
                self.sink.state("ended", end.kind.as_str(), &end.reason, self.last_verified)?;
                Ok(Step::Done(FollowOutcome::Ended { kind: end.kind, before_checkpoint: false }))
            }
            StreamEvent::SnapshotChunk(_) => Ok(self.enter_needs_snapshot(
                manifest,
                NeedsSnapshotReason::ProtocolViolation,
                "a snapshot chunk arrived with no checkpoint announcing it",
                sequence + 1,
                None,
            )),
        }
    }

    /// Starts skimming a recovery checkpoint the follower already recovered past by itself.
    ///
    /// The comparison happens here, on the declaration, because that is what the two
    /// implementations each derived. The chunks that follow are hashed and dropped: their digest
    /// is the producer's own claim about the snapshot, and a chunk sequence that does not meet it
    /// is a transport fault the pair this follower holds says nothing about.
    fn begin_skim(
        &mut self,
        manifest: Manifest,
        state: Box<ReplayState>,
        checkpoint: Checkpoint,
        ancestor: BlockRef,
        pending_tip: Option<BlockRef>,
        sequence: u64,
    ) -> eyre::Result<Step> {
        if let Err(err) = checkpoint.validate_declared(DEFAULT_MAX_SNAPSHOT_BYTES) {
            return Ok(self.enter_needs_snapshot(
                manifest,
                NeedsSnapshotReason::ProtocolViolation,
                &format!("recovery checkpoint declaration refused: {err}"),
                sequence + 1,
                Some(ancestor),
            ))
        }
        if let Err(disagreement) = cross_check_recovery_checkpoint(&state, &checkpoint, ancestor) {
            error!(
                target: "ps_follow",
                block = checkpoint.block.number,
                field = disagreement.field,
                recorded = %disagreement.recorded,
                replayed = %disagreement.replayed,
                "The producer's recovery checkpoint disagrees with the generation this follower \
                 recovered to. One of the two undid the reorg wrongly"
            );
            self.replay.disagreements.push((checkpoint.block, disagreement));
            // The producer is the authority on what is canonical, so the run continues from its
            // checkpoint — as an explicit reset, and already recorded as non-agreeing.
            return Ok(self.enter_needs_snapshot(
                manifest,
                NeedsSnapshotReason::SnapshotRequired,
                "the recovery checkpoint disagrees with the recovered generation",
                sequence,
                None,
            ))
        }
        self.checkpoints_skimmed += 1;
        info!(
            target: "ps_follow",
            block = checkpoint.block.number,
            chunks = checkpoint.snapshot_chunks,
            "The producer's recovery checkpoint agrees with the generation this follower \
             recovered to; its snapshot is checked and not installed"
        );
        if checkpoint.snapshot_chunks == 0 {
            return self.finish_skim(
                SkimPhase {
                    manifest,
                    state,
                    checkpoint,
                    pending_tip,
                    received: 0,
                    accumulated: 0,
                    digest: Keccak256::new(),
                },
                sequence,
            )
        }
        self.phase = Phase::SkimmingChunks(Box::new(SkimPhase {
            manifest,
            state,
            checkpoint,
            pending_tip,
            received: 0,
            accumulated: 0,
            digest: Keccak256::new(),
        }));
        Ok(Step::Continue)
    }

    /// One frame while a recovery checkpoint's chunks are being read and discarded.
    fn skim_frame(
        &mut self,
        phase: SkimPhase,
        sequence: u64,
        event: StreamEvent,
    ) -> eyre::Result<Step> {
        let SkimPhase {
            manifest,
            state,
            checkpoint,
            pending_tip,
            received,
            accumulated,
            mut digest,
        } = phase;
        let StreamEvent::SnapshotChunk(chunk) = event else {
            return Ok(self.enter_needs_snapshot(
                manifest,
                NeedsSnapshotReason::ProtocolViolation,
                &format!(
                    "a {} frame arrived while the recovery snapshot's chunks were incomplete",
                    kind_of(&event).as_str()
                ),
                sequence + 1,
                None,
            ))
        };
        if chunk.index != received {
            return Ok(self.enter_needs_snapshot(
                manifest,
                NeedsSnapshotReason::ProtocolViolation,
                &format!("snapshot chunk {} arrived where {received} was expected", chunk.index),
                sequence + 1,
                None,
            ))
        }
        // Hashed as it goes rather than buffered: the point of skimming is that a recovery
        // checkpoint costs a follower no memory it did not already have.
        digest.update(&chunk.bytes);
        let accumulated = accumulated + chunk.bytes.len() as u64;
        let received = received + 1;
        if received < checkpoint.snapshot_chunks {
            self.phase = Phase::SkimmingChunks(Box::new(SkimPhase {
                manifest,
                state,
                checkpoint,
                pending_tip,
                received,
                accumulated,
                digest,
            }));
            return Ok(Step::Continue)
        }
        self.finish_skim(
            SkimPhase { manifest, state, checkpoint, pending_tip, received, accumulated, digest },
            sequence,
        )
    }

    /// Checks what the skim accumulated against what the checkpoint declared, and resumes.
    fn finish_skim(&mut self, phase: SkimPhase, sequence: u64) -> eyre::Result<Step> {
        let SkimPhase { manifest, state, checkpoint, pending_tip, accumulated, digest, .. } = phase;
        let digest = if checkpoint.snapshot_chunks == 0 { B256::ZERO } else { digest.finalize() };
        if checkpoint.snapshot_chunks > 0 &&
            (digest != checkpoint.snapshot_digest || accumulated != checkpoint.snapshot_bytes)
        {
            return Ok(self.enter_needs_snapshot(
                manifest,
                NeedsSnapshotReason::ProtocolViolation,
                &format!(
                    "the recovery snapshot hashed to {digest:?} over {accumulated} bytes; the \
                     checkpoint declared {:?} over {}",
                    checkpoint.snapshot_digest, checkpoint.snapshot_bytes
                ),
                sequence + 1,
                None,
            ))
        }
        self.phase = Phase::Streaming {
            manifest,
            state,
            expected_child: Some((checkpoint.block.number + 1, checkpoint.block.hash)),
            announced: None,
            pending_tip,
        };
        Ok(Step::Continue)
    }

    /// Verifies a checkpoint's declaration and head, and starts collecting its chunks.
    fn accept_checkpoint(
        &mut self,
        manifest: Manifest,
        checkpoint: Checkpoint,
        sequence: u64,
    ) -> eyre::Result<Step> {
        if let Err(err) = checkpoint.validate_declared(DEFAULT_MAX_SNAPSHOT_BYTES) {
            return Ok(self.enter_needs_snapshot(
                manifest,
                NeedsSnapshotReason::ProtocolViolation,
                &format!("checkpoint declaration refused: {err}"),
                sequence + 1,
                None,
            ))
        }
        // A live checkpoint must carry its own header: the restored pair's first act is to admit
        // H + 1 against it, and `NoAcceptedParent` is a rejection, not a wait. The offline
        // format keeps the field optional for forensic corpora; a live follower does not.
        if decode_accepted_head(&checkpoint).is_none() {
            return Ok(self.enter_needs_snapshot(
                manifest,
                NeedsSnapshotReason::HeadlessCheckpoint,
                "the checkpoint carries no verifiable accepted head, so its H + 1 could never \
                 be admitted",
                sequence + 1,
                None,
            ))
        }
        info!(
            target: "ps_follow",
            block = checkpoint.block.number,
            chunks = checkpoint.snapshot_chunks,
            "Checkpoint accepted; collecting its snapshot"
        );
        if checkpoint.snapshot_chunks == 0 {
            return self.restore_pair(manifest, checkpoint, Vec::new(), sequence)
        }
        self.phase = Phase::CollectingChunks { manifest, checkpoint, chunks: Vec::new() };
        Ok(Step::Continue)
    }

    /// Restores a fresh pair from a complete checkpoint+snapshot and opens the verdict stream.
    fn restore_pair(
        &mut self,
        manifest: Manifest,
        checkpoint: Checkpoint,
        chunks: Vec<SnapshotChunk>,
        sequence: u64,
    ) -> eyre::Result<Step> {
        match restore(&manifest, &checkpoint, &chunks) {
            Ok(state) => {
                self.restores += 1;
                let expected_child = Some((checkpoint.block.number + 1, checkpoint.block.hash));
                // Decided by the scan that found this checkpoint, because only the scan knows
                // what was asked for and what went by in between. The first restore of a stream
                // is neither: nothing preceded it to be continuous with.
                let class = self.pending_recovery.take().map(|pending| {
                    let continuous =
                        pending.target == Some(checkpoint.block) && pending.skipped == 0;
                    if continuous {
                        RecoveryClass::Continuous
                    } else {
                        // Everything between the last block this follower verified and the block
                        // the checkpoint restores to was validated by nothing.
                        let unverified = self
                            .last_verified
                            .map(|last| last.number)
                            .filter(|last| *last < checkpoint.block.number)
                            .map(|last| (last + 1, checkpoint.block.number));
                        RecoveryClass::Reset { unverified }
                    }
                });
                match class {
                    Some(RecoveryClass::Continuous) => self.restores_continuous += 1,
                    Some(RecoveryClass::Reset { unverified }) => {
                        self.restores_reset += 1;
                        if let Some(interval) = unverified {
                            self.unverified_intervals.push(interval);
                        }
                    }
                    None => {}
                }
                info!(
                    target: "ps_follow",
                    block = checkpoint.block.number,
                    restores = self.restores,
                    classification = class.map(RecoveryClass::as_str),
                    "Restored a pair with no database; verdicts start at the next block"
                );
                self.sink.restored(checkpoint.block, class, self.last_verified)?;
                self.phase = Phase::Streaming {
                    manifest,
                    state: Box::new(state),
                    expected_child,
                    announced: None,
                    pending_tip: None,
                };
                Ok(Step::Continue)
            }
            Err(err) => Ok(self.enter_needs_snapshot(
                manifest,
                NeedsSnapshotReason::ProtocolViolation,
                &format!("the checkpoint's snapshot did not verify: {err:#}"),
                sequence + 1,
                None,
            )),
        }
    }

    /// Stops verdicts and starts waiting for a checkpoint at or above `scan_from`.
    ///
    /// `target_ancestor` is what turns a refusal into a request: a recovery that lands on exactly
    /// that block resumes verification with no interval unaccounted for, and one that lands
    /// anywhere else is an explicit checkpoint reset. `None` means the discontinuity named no
    /// block, so any recovery from it is a reset by definition.
    fn enter_needs_snapshot(
        &mut self,
        manifest: Manifest,
        reason: NeedsSnapshotReason,
        detail: &str,
        scan_from: u64,
        target_ancestor: Option<BlockRef>,
    ) -> Step {
        self.needs_snapshot_entries += 1;
        self.last_needs_snapshot = Some(reason);
        let epoch = manifest.epoch;
        warn!(
            target: "ps_follow",
            reason = reason.as_str(),
            detail,
            scan_from,
            target_ancestor = target_ancestor.map(|block| block.number),
            epoch,
            "Entering NeedsSnapshot: no further verdict until a rebootstrap succeeds"
        );
        if let Err(err) = self.sink.needs_snapshot(
            reason.as_str(),
            detail,
            self.last_verified,
            target_ancestor,
            epoch,
        ) {
            warn!(target: "ps_follow", error = %err, "Could not record the state transition");
        }
        self.phase = Phase::NeedsSnapshot { manifest, reason, scan_from, target_ancestor, epoch };
        Step::Continue
    }

    /// The block a recovery still has to land on, after everything announced in the meantime.
    ///
    /// The producer keeps talking during an outage. A later reorg replaces the target with its own
    /// ancestor; a later reset or manifest withdraws it entirely, because neither names a block
    /// and any checkpoint following them is a reset by construction. Read through the same
    /// name-versus-header authority as ordinary delivery — recovery is exactly where a renamed
    /// frame would be aimed.
    fn supersede_target(
        &self,
        target: Option<BlockRef>,
        scan_from: u64,
        checkpoint_at: u64,
    ) -> eyre::Result<Option<BlockRef>> {
        let mut latest: Option<(u64, Option<BlockRef>)> = None;
        for kind in [FrameKind::Reorg, FrameKind::Reset, FrameKind::Manifest] {
            let found = self
                .tail
                .last_before(kind, scan_from, checkpoint_at)
                .map_err(|fault| eyre::eyre!("the recovery scan failed: {fault}"))?;
            let Some(sequence) = found else { continue };
            if latest.is_some_and(|(seen, _)| seen >= sequence) {
                continue
            }
            let announced = match kind {
                FrameKind::Reorg => {
                    let frame = self
                        .tail
                        .read_at(sequence, kind)
                        .map_err(|fault| eyre::eyre!("the recovery scan failed: {fault}"))?;
                    match frame.event {
                        StreamEvent::Reorg(reorg) => Some(reorg.common_ancestor),
                        _ => {
                            return Err(eyre::eyre!(
                                "the frame at sequence {sequence} is named Reorg but decoded as \
                                 something else"
                            ))
                        }
                    }
                }
                // Neither names a block, so nothing after them can be continuous with anything.
                _ => None,
            };
            latest = Some((sequence, announced));
        }
        match latest {
            Some((sequence, announced)) => {
                info!(
                    target: "ps_follow",
                    sequence,
                    superseded = ?target.map(|block| block.number),
                    now = ?announced.map(|block| block.number),
                    "A later announcement supersedes the recovery target"
                );
                Ok(announced)
            }
            None => Ok(target),
        }
    }

    /// One recovery pass: a fresh checkpoint restarts the grammar at its own sequence; an `End`
    /// proves no recovery is coming.
    ///
    /// Both are looked for on every pass and the lower sequence wins, because the sequence space
    /// is the stream's word order: a checkpoint written *after* an `End` is a trailing frame on a
    /// closed stream, not a recovery, and preferring it would let anything appended to a dead
    /// spool restart verdicts.
    fn scan_for_recovery(&mut self) -> eyre::Result<Step> {
        let Phase::NeedsSnapshot { manifest, reason, scan_from, target_ancestor, epoch } =
            std::mem::replace(&mut self.phase, Phase::AwaitingManifest)
        else {
            return Err(eyre::eyre!("the recovery scan ran outside NeedsSnapshot"))
        };

        let checkpoint_at = self
            .tail
            .scan_for(FrameKind::Checkpoint, scan_from)
            .map_err(|fault| eyre::eyre!("the recovery scan failed: {fault}"))?;
        let end_at = self
            .tail
            .scan_for(FrameKind::End, scan_from)
            .map_err(|fault| eyre::eyre!("the recovery scan failed: {fault}"))?;

        if let Some(found) = checkpoint_at.filter(|found| end_at.is_none_or(|end| *found < end)) {
            // Skipped commits are part of the record, and a count that failed to read is not a
            // count of zero — an I/O error here fails the recovery rather than shrinking it.
            let skipped = self
                .tail
                .count_commits_between(self.tail.next_sequence(), found)
                .map_err(|fault| eyre::eyre!("the recovery scan failed: {fault}"))?;
            self.commits_skipped_in_recovery += skipped;
            // The last word between the request and the checkpoint is the operative one: a second
            // reorg during the outage moves the block a snapshot has to be authenticated at, and
            // a reset withdraws the request. Without this, a stale target would let a checkpoint
            // that answers a superseded question be reported as continuous recovery.
            let target = self.supersede_target(target_ancestor, scan_from, found)?;
            // Classified where the checkpoint itself is in hand, which is `restore_pair`; what
            // the scan knows and it does not is what was asked for and what went by in between.
            self.pending_recovery = Some(PendingRecovery { target, skipped });
            info!(
                target: "ps_follow",
                checkpoint_sequence = found,
                skipped_commits = skipped,
                epoch,
                "A recovery checkpoint appeared; skipped commits are recorded, never verified"
            );
            self.tail.skip_to(found);
            self.phase = Phase::AwaitingCheckpoint { manifest };
            return Ok(Step::Continue)
        }

        // No usable checkpoint. An End frame above the watermark settles that no recovery is
        // coming — read back through the same authority checks as ordinary delivery, its
        // numbering promise included.
        if let Some(end_at) = end_at {
            let frame = self
                .tail
                .read_at(end_at, FrameKind::End)
                .map_err(|fault| eyre::eyre!("the End frame did not read back: {fault}"))?;
            let StreamEvent::End(end) = frame.event else {
                return Err(eyre::eyre!(
                    "the frame at sequence {end_at} is named End but decoded as something else"
                ))
            };
            self.check_end_numbering(&end.reason, end.last_sequence, end_at)?;
            if let Some(trailing) = checkpoint_at {
                warn!(
                    target: "ps_follow",
                    end_sequence = end_at,
                    checkpoint_sequence = trailing,
                    "A checkpoint exists past the End frame; a closed stream has no recovery, so \
                     it is a trailing frame, not a restart"
                );
            }
            info!(
                target: "ps_follow",
                kind = end.kind.as_str(),
                "The stream ended while waiting for a recovery checkpoint"
            );
            self.sink.state("ended", end.kind.as_str(), &end.reason, self.last_verified)?;
            return Ok(Step::Done(FollowOutcome::Ended { kind: end.kind, before_checkpoint: false }))
        }

        self.phase = Phase::NeedsSnapshot { manifest, reason, scan_from, target_ancestor, epoch };
        Ok(self.idle())
    }

    /// A delivery fault: fail closed everywhere identity is known, loudly where it is not.
    fn tail_fault(&mut self, fault: TailFault) -> eyre::Result<Step> {
        let reason = match &fault {
            TailFault::Gap { .. } => NeedsSnapshotReason::Gap,
            TailFault::DuplicateConflict { .. } => NeedsSnapshotReason::DuplicateConflict,
            TailFault::Undecodable { .. } => NeedsSnapshotReason::Undecodable,
        };
        let detail = fault.to_string();
        let phase = std::mem::replace(&mut self.phase, Phase::AwaitingManifest);
        let manifest = match phase {
            // Before the manifest there is no identity to recover under; this is not a stream
            // to wait on, it is a directory that cannot be followed.
            Phase::AwaitingManifest => return Err(eyre::eyre!("unfollowable spool: {detail}")),
            Phase::SkimmingChunks(phase) => phase.manifest,
            Phase::AwaitingCheckpoint { manifest } |
            Phase::CollectingChunks { manifest, .. } |
            Phase::Streaming { manifest, .. } |
            Phase::NeedsSnapshot { manifest, .. } => manifest,
        };
        let scan_from = self.tail.next_sequence();
        Ok(self.enter_needs_snapshot(manifest, reason, &detail, scan_from, None))
    }

    /// Nothing new: sleep one poll, or stop if the harness bounded the wait.
    fn idle(&mut self) -> Step {
        if self.options.idle_timeout.is_some_and(|bound| self.last_frame_at.elapsed() >= bound) {
            return Step::Done(FollowOutcome::IdleTimeout { waiting_in: self.phase.label() })
        }
        std::thread::sleep(self.options.poll);
        Step::Continue
    }

    /// The End frame's numbering promise, checked wherever one is consumed.
    fn check_end_numbering(
        &self,
        reason: &str,
        last_sequence: u64,
        sequence: u64,
    ) -> eyre::Result<()> {
        if last_sequence.checked_add(1) != Some(sequence) {
            return Err(eyre::eyre!(
                "the End frame ({reason}) at sequence {sequence} names {last_sequence} as the \
                 last frame; its predecessor was {}",
                sequence.saturating_sub(1)
            ))
        }
        Ok(())
    }

    fn into_report(mut self, outcome: FollowOutcome) -> FollowReport {
        if let Err(err) = self.sink.ack_state(&outcome, self.last_verified, &self.tail) {
            warn!(target: "ps_follow", error = %err, "Could not write the final ack");
        }
        FollowReport {
            outcome,
            replay: self.replay,
            blocks_verified: self.blocks_verified,
            restores: self.restores,
            needs_snapshot_entries: self.needs_snapshot_entries,
            commits_skipped_in_recovery: self.commits_skipped_in_recovery,
            last_needs_snapshot: self.last_needs_snapshot,
            last_verified: self.last_verified,
            reorgs_applied: self.reorgs_applied,
            reverts_applied: self.reverts_applied,
            checkpoints_skimmed: self.checkpoints_skimmed,
            restores_continuous: self.restores_continuous,
            restores_reset: self.restores_reset,
            winning_branches_completed: self.winning_branches_completed,
            winning_branches_incomplete: self.winning_branches_incomplete,
            unverified_intervals: self.unverified_intervals,
        }
    }
}

/// The follower's outputs: a JSONL verdict stream and an atomically rewritten ack file.
struct VerdictSink {
    verdicts: Option<std::fs::File>,
    ack: Option<PathBuf>,
    label: String,
    last_state: &'static str,
}

impl VerdictSink {
    fn open(options: &FollowOptions) -> eyre::Result<Self> {
        let verdicts = options
            .verdicts
            .as_ref()
            .map(|path| {
                if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::OpenOptions::new().create(true).append(true).open(path)
            })
            .transpose()?;
        Ok(Self {
            verdicts,
            ack: options.ack.clone(),
            label: options.label.clone(),
            last_state: "starting",
        })
    }

    /// One verdict line per block. The S5 timing boundaries are laid as `null` rather than
    /// guessed: `delivery_us` and `observed_verdict_latency_us` are measurement disciplines this
    /// phase does not claim.
    fn verdict(
        &mut self,
        verdict: &str,
        block: BlockRef,
        sequence: u64,
        provenance: PayloadProvenance,
        disagreements: usize,
        replay: &ReplayReport,
    ) -> eyre::Result<()> {
        self.last_state = "streaming";
        let timing = replay.blocks.last().filter(|timing| timing.number == block.number);
        self.write(serde_json::json!({
            "schema_version": 1,
            "benchmark": "standalone_follow_v1",
            "kind": "verdict",
            "label": self.label,
            "block": block.number,
            "block_hash": format!("{:?}", block.hash),
            "sequence": sequence,
            "verdict": verdict,
            "payload_provenance": provenance.as_str(),
            "disagreements": disagreements,
            "admission_us": timing.map(|timing| timing.admission_us),
            "transition_us": timing.map(|timing| timing.transition_us),
            "delivery_us": serde_json::Value::Null,
            "observed_verdict_latency_us": serde_json::Value::Null,
            "observed_at_ms": now_ms(),
        }))?;
        self.write_ack(sequence, Some(block), "streaming")
    }

    /// One line per state transition, so the record shows *when* verdicts stopped.
    /// The `needs_snapshot` transition, with what a recovery has to produce to end it.
    ///
    /// `target_ancestor` is the field an operator or a snapshot service reads: it names the exact
    /// block a bounded snapshot has to be authenticated at, which is what the standalone recovery
    /// protocol asks for and what a bare reason string could never say.
    fn needs_snapshot(
        &mut self,
        reason: &str,
        detail: &str,
        last_verified: Option<BlockRef>,
        target_ancestor: Option<BlockRef>,
        epoch: u64,
    ) -> eyre::Result<()> {
        self.last_state = "needs_snapshot";
        self.write(serde_json::json!({
            "schema_version": 1,
            "benchmark": "standalone_follow_v1",
            "kind": "state",
            "label": self.label,
            "state": "needs_snapshot",
            "reason": reason,
            "detail": detail,
            "last_verified": last_verified.map(|block| block.number),
            "target_ancestor": target_ancestor.map(|block| block.number),
            "target_ancestor_hash": target_ancestor.map(|block| format!("{:?}", block.hash)),
            "epoch": epoch,
            "observed_at_ms": now_ms(),
        }))
    }

    /// A restore, and whether it was continuous with what came before it.
    fn restored(
        &mut self,
        block: BlockRef,
        class: Option<RecoveryClass>,
        last_verified: Option<BlockRef>,
    ) -> eyre::Result<()> {
        self.last_state = "streaming";
        let unverified = match class {
            Some(RecoveryClass::Reset { unverified }) => unverified,
            _ => None,
        };
        self.write(serde_json::json!({
            "schema_version": 1,
            "benchmark": "standalone_follow_v1",
            "kind": "state",
            "label": self.label,
            "state": "streaming",
            "reason": "restored",
            "detail": format!("restored at block {}", block.number),
            "classification": class.map(RecoveryClass::as_str),
            "unverified_from": unverified.map(|(from, _)| from),
            "unverified_to": unverified.map(|(_, to)| to),
            "last_verified": last_verified.map(|block| block.number),
            "observed_at_ms": now_ms(),
        }))
    }

    /// A lifecycle event this follower applied rather than stopped at.
    ///
    /// Written apart from the verdicts because it says something no verdict can: the blocks it
    /// names were validated, and are no longer canonical. A consumer reading only verdict lines
    /// would carry an abandoned block as though it were still on the chain.
    fn lifecycle(
        &mut self,
        kind: &'static str,
        ancestor: BlockRef,
        abandoned: &[BlockRef],
        winning_tip: Option<BlockRef>,
        last_verified: Option<BlockRef>,
    ) -> eyre::Result<()> {
        self.write(serde_json::json!({
            "schema_version": 1,
            "benchmark": "standalone_follow_v1",
            "kind": "lifecycle",
            "label": self.label,
            "event": kind,
            "common_ancestor": ancestor.number,
            "common_ancestor_hash": format!("{:?}", ancestor.hash),
            "abandoned": abandoned.iter().map(|block| block.number).collect::<Vec<_>>(),
            "abandoned_hashes": abandoned
                .iter()
                .map(|block| format!("{:?}", block.hash))
                .collect::<Vec<_>>(),
            "winning_tip": winning_tip.map(|block| block.number),
            "last_verified": last_verified.map(|block| block.number),
            "observed_at_ms": now_ms(),
        }))
    }

    fn state(
        &mut self,
        state: &'static str,
        reason: &str,
        detail: &str,
        last_verified: Option<BlockRef>,
    ) -> eyre::Result<()> {
        self.last_state = state;
        self.write(serde_json::json!({
            "schema_version": 1,
            "benchmark": "standalone_follow_v1",
            "kind": "state",
            "label": self.label,
            "state": state,
            "reason": reason,
            "detail": detail,
            "last_verified": last_verified.map(|block| block.number),
            "observed_at_ms": now_ms(),
        }))
    }

    fn ack_state(
        &mut self,
        outcome: &FollowOutcome,
        last_verified: Option<BlockRef>,
        tail: &SpoolTail,
    ) -> eyre::Result<()> {
        let state = match outcome {
            FollowOutcome::Ended { .. } => "ended",
            FollowOutcome::Faulted { .. } => "faulted",
            FollowOutcome::MaxBlocks => "max_blocks",
            FollowOutcome::IdleTimeout { .. } => "idle_timeout",
        };
        self.write_ack(tail.next_sequence().saturating_sub(1), last_verified, state)
    }

    fn write(&mut self, record: serde_json::Value) -> eyre::Result<()> {
        if let Some(file) = self.verdicts.as_mut() {
            writeln!(file, "{record}")?;
        }
        Ok(())
    }

    /// The consumer watermark, rewritten atomically and kept outside the spool so the producer's
    /// own invariants (a fresh spool holds no foreign files) are untouched.
    fn write_ack(
        &self,
        last_sequence: u64,
        block: Option<BlockRef>,
        state: &str,
    ) -> eyre::Result<()> {
        let Some(path) = &self.ack else { return Ok(()) };
        let record = serde_json::json!({
            "last_sequence": last_sequence,
            "block": block.map(|block| block.number),
            "block_hash": block.map(|block| format!("{:?}", block.hash)),
            "state": state,
            "observed_at_ms": now_ms(),
        });
        let temporary = path.with_extension("tmp");
        std::fs::write(&temporary, record.to_string())?;
        std::fs::rename(&temporary, path)?;
        Ok(())
    }
}

fn now_ms() -> u128 {
    SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)
}

const fn kind_of(event: &StreamEvent) -> FrameKind {
    match event {
        StreamEvent::Manifest(_) => FrameKind::Manifest,
        StreamEvent::Checkpoint(_) => FrameKind::Checkpoint,
        StreamEvent::SnapshotChunk(_) => FrameKind::SnapshotChunk,
        StreamEvent::Commit(_) => FrameKind::Commit,
        StreamEvent::Reorg(_) => FrameKind::Reorg,
        StreamEvent::Reset(_) => FrameKind::Reset,
        StreamEvent::End(_) => FrameKind::End,
    }
}
