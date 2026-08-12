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
        chain_spec_for, config_for, decode_accepted_head, replay_commit, restore, CommitOutcome,
        ReplayOptions, ReplayReport, ReplayState,
    },
    tail::{SpoolTail, TailEvent, TailFault},
};
use alloy_primitives::B256;
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
}

impl FollowReport {
    /// Whether every verdict published agreed with the recording and nothing faulted.
    pub fn agreed(&self) -> bool {
        self.replay.agreed() && !matches!(self.outcome, FollowOutcome::Faulted { .. })
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
    },
    /// Verdicts stopped; scanning for a fresh checkpoint to rebootstrap from.
    NeedsSnapshot { manifest: Manifest, reason: NeedsSnapshotReason, scan_from: u64 },
}

impl Phase {
    const fn label(&self) -> &'static str {
        match self {
            Self::AwaitingManifest => "awaiting_manifest",
            Self::AwaitingCheckpoint { .. } => "awaiting_checkpoint",
            Self::CollectingChunks { .. } => "collecting_chunks",
            Self::Streaming { .. } => "streaming",
            Self::NeedsSnapshot { .. } => "needs_snapshot",
        }
    }
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
            },
            tail: SpoolTail::new(dir, options.frame_limits),
            sink: VerdictSink::open(options)?,
            phase: Phase::AwaitingManifest,
            replay: ReplayReport::default(),
            blocks_verified: 0,
            restores: 0,
            needs_snapshot_entries: 0,
            commits_skipped_in_recovery: 0,
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
                )),
            (Phase::AwaitingCheckpoint { manifest }, other) => Ok(self.enter_needs_snapshot(
                manifest,
                NeedsSnapshotReason::ProtocolViolation,
                &format!("a {} frame arrived before any checkpoint", kind_of(&other).as_str()),
                sequence + 1,
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
            )),

            (Phase::Streaming { manifest, state, expected_child }, event) => {
                self.stream_frame(manifest, state, expected_child, sequence, event)
            }

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
        manifest: Manifest,
        mut state: Box<ReplayState>,
        expected_child: Option<(u64, B256)>,
        sequence: u64,
        event: StreamEvent,
    ) -> eyre::Result<Step> {
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
                        self.phase = Phase::Streaming { manifest, state, expected_child: None };
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
            )),
            StreamEvent::Checkpoint(_) => {
                // Unannounced: today's grammar has no mid-stream checkpoint without a Reset or
                // Reorg in front of it. Fail closed — and let the recovery scan re-examine this
                // very frame, so a future producer's recovery grammar still converges. The
                // violation stays in the record either way.
                Ok(self.enter_needs_snapshot(
                    manifest,
                    NeedsSnapshotReason::ProtocolViolation,
                    "an unannounced checkpoint arrived mid-stream",
                    sequence,
                ))
            }
            StreamEvent::Reorg(reorg) => Ok(self.enter_needs_snapshot(
                manifest,
                NeedsSnapshotReason::SnapshotRequired,
                &format!(
                    "a reorg at ancestor {} arrived; reorg replay is S4, and verdicts past an \
                     unapplied reorg would describe a branch the producer left",
                    reorg.common_ancestor.number
                ),
                sequence + 1,
            )),
            StreamEvent::Reset(reset) => Ok(self.enter_needs_snapshot(
                manifest,
                reset.reason.into(),
                &reset.detail,
                sequence + 1,
            )),
            StreamEvent::End(end) => {
                self.phase = Phase::Streaming { manifest, state, expected_child };
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
            )),
        }
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
                info!(
                    target: "ps_follow",
                    block = checkpoint.block.number,
                    restores = self.restores,
                    "Restored a pair with no database; verdicts start at the next block"
                );
                self.sink.state(
                    "streaming",
                    "restored",
                    &format!("restored at block {}", checkpoint.block.number),
                    self.last_verified,
                )?;
                self.phase = Phase::Streaming { manifest, state: Box::new(state), expected_child };
                Ok(Step::Continue)
            }
            Err(err) => Ok(self.enter_needs_snapshot(
                manifest,
                NeedsSnapshotReason::ProtocolViolation,
                &format!("the checkpoint's snapshot did not verify: {err:#}"),
                sequence + 1,
            )),
        }
    }

    /// Stops verdicts and starts waiting for a checkpoint at or above `scan_from`.
    fn enter_needs_snapshot(
        &mut self,
        manifest: Manifest,
        reason: NeedsSnapshotReason,
        detail: &str,
        scan_from: u64,
    ) -> Step {
        self.needs_snapshot_entries += 1;
        self.last_needs_snapshot = Some(reason);
        warn!(
            target: "ps_follow",
            reason = reason.as_str(),
            detail,
            scan_from,
            "Entering NeedsSnapshot: no further verdict until an exact-anchor rebootstrap \
             succeeds"
        );
        if let Err(err) =
            self.sink.state("needs_snapshot", reason.as_str(), detail, self.last_verified)
        {
            warn!(target: "ps_follow", error = %err, "Could not record the state transition");
        }
        self.phase = Phase::NeedsSnapshot { manifest, reason, scan_from };
        Step::Continue
    }

    /// One recovery pass: a fresh checkpoint restarts the grammar at its own sequence; an `End`
    /// proves no recovery is coming.
    ///
    /// Both are looked for on every pass and the lower sequence wins, because the sequence space
    /// is the stream's word order: a checkpoint written *after* an `End` is a trailing frame on a
    /// closed stream, not a recovery, and preferring it would let anything appended to a dead
    /// spool restart verdicts.
    fn scan_for_recovery(&mut self) -> eyre::Result<Step> {
        let Phase::NeedsSnapshot { manifest, reason, scan_from } =
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
            info!(
                target: "ps_follow",
                checkpoint_sequence = found,
                skipped_commits = skipped,
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

        self.phase = Phase::NeedsSnapshot { manifest, reason, scan_from };
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
            Phase::AwaitingCheckpoint { manifest } |
            Phase::CollectingChunks { manifest, .. } |
            Phase::Streaming { manifest, .. } |
            Phase::NeedsSnapshot { manifest, .. } => manifest,
        };
        let scan_from = self.tail.next_sequence();
        Ok(self.enter_needs_snapshot(manifest, reason, &detail, scan_from))
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
