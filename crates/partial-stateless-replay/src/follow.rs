//! The live consumer: verdicts on a stream a producer is still writing.
//!
//! This is the live-equivalence gate's behaviour as a state machine. The follower verifies the
//! manifest's identity before it accepts anything, the checkpoint and its snapshot before it
//! accepts a commit, and exactly `H + 1` as the first commit — and it never hides a delivery
//! violation as a drop. A gap, a duplicate claim on a sequence, an undecodable frame, an epoch
//! change, a producer reset, or a reorg frame all take it to `NeedsSnapshot`: verdicts stop, and
//! only a fresh checkpoint that verifies end to end restarts them at its own `H′ + 1`.
//!
//! **`NeedsSnapshot` waits for the producer; it does not synthesize recovery.** Today's producer
//! writes one checkpoint per stream, so in a live run the state is effectively terminal — the
//! machinery below is proven against synthetic spools carrying a mid-stream second checkpoint, and
//! the producer-side re-checkpoint that would exercise it live is the deep-reorg recovery protocol
//! on the producer.
//!
//! **A quiet spool is not a dead producer.** The follower cannot tell "no new block yet" from "the
//! producer was killed" by looking at files; an `End` frame is how a producer says it stopped, and
//! its absence is what "cut" means. So the default is to wait forever, and a harness that killed
//! the producer on purpose passes `idle_timeout` and judges the spool offline.

use crate::{
    driver::{
        chain_spec_for, config_for, consumer_is_at, cross_check_recovery_checkpoint,
        decode_accepted_head, replay_commit, restore, BlockTiming, CommitOutcome, FrameCosts,
        ReplayOptions, ReplayReport, ReplayState, MAX_REWIND_FRAMES,
    },
    reorg::{apply_reorg, check_shape, warn_inapplicable, ReorgOutcome},
    spool::SpooledFrame,
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
    /// The power-loss profile for the ack: fsync it before its rename and its directory after.
    ///
    /// Off by default for the same reason the producer's `PS_STREAM_FSYNC` is: durability past a
    /// process restart is a semantics choice with a measured price, not a free upgrade.
    pub ack_fsync: bool,
    /// Start from the watermark the ack file records instead of from the stream's first frame.
    pub resume: bool,
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
            ack_fsync: false,
            resume: false,
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
/// block. Deep-reorg recovery is what lifts it into the standalone state machine, once
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
    /// Commits verified for the first time out of the spool inside completed rewind windows.
    /// A subset of `blocks_verified`, broken out so the recovery share is visible.
    pub rewind_replayed_commits: u64,
    /// Rewind windows refused for exceeding the bound; each degraded to an explicit reset.
    pub rewind_windows_refused: u64,
    /// Winning branches that reached the tip the reorg announced.
    pub winning_branches_completed: u64,
    /// Winning branches that did not, with nothing valid taking their place.
    pub winning_branches_incomplete: u64,
    /// Winning branches a later valid announcement withdrew before their tip arrived.
    ///
    /// Diagnostic only: the chain moving twice in a row is ordinary, and the blocks between the
    /// two announcements never became canonical, so no verdict was owed on them.
    pub winning_branches_superseded: u64,
    /// Frames a recovery scan read and would not act on, so no recovery under them is continuous.
    pub scan_refusals: u64,
    /// Blocks re-derived on the way back to a watermark a previous run left.
    ///
    /// Kept out of `blocks_verified`, which counts what *this* run verified for the first time.
    pub catch_up_blocks: u64,
    /// The checkpoint sequence a resumed run rebuilt its pair from.
    pub resumed_from: Option<u64>,
    /// Canonical intervals no verdict covers, in the order they opened.
    pub unverified_intervals: Vec<(u64, u64)>,
    /// Clock readings that went backwards while deriving mtime-based latency fields.
    ///
    /// Counted rather than clamped: a clamped zero would enter the latency distribution as an
    /// excellent measurement, and an anomaly is a fact about the measurement, not a sample.
    pub latency_anomalies: u64,
    /// Per-verdict cost of writing the verdict line — the publication half `decision_latency_us`
    /// cannot carry, reported as its own distribution in the summary.
    pub verdict_write_us: Vec<u64>,
    /// Per-verdict cost of writing the ack, sampled only when one was actually written.
    pub ack_write_us: Vec<u64>,
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
    CollectingChunks {
        manifest: Manifest,
        checkpoint: Checkpoint,
        /// The checkpoint frame's own sequence, not the last chunk's. It is what a restart has to
        /// name to come back to this exact pair, so it is carried rather than re-derived.
        checkpoint_sequence: u64,
        chunks: Vec<SnapshotChunk>,
    },
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
        /// The tip the reorg that opened this recovery said its branch would reach. Carried so
        /// the restore can keep checking arrival; without it a checkpoint published mid-branch
        /// would leave the branch's completion checked by nothing.
        announced_tip: Option<BlockRef>,
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
/// What the frames written during an outage did to the recovery this follower is waiting on.
struct Supersession {
    /// The block a checkpoint now has to land on to be a continuous recovery, if any still does.
    target: Option<BlockRef>,
    /// The tip the operative announcement said its branch would reach, carried with the target:
    /// a reset or epoch change withdraws it, a later reorg replaces it with its own.
    tip: Option<BlockRef>,
    /// The next epoch's manifest, once it has been checked to be this stream's successor.
    adopt: Option<Manifest>,
    /// Sequence of the last announcement in the window, when one superseded the target. The
    /// rewind window starts after it: commits below the last announcement belong to a branch
    /// that was withdrawn.
    at: Option<u64>,
}

/// The commit frames a rewound recovery replays: sequences `[from, until)`, with `until` the
/// installed checkpoint's announce frame.
#[derive(Debug, Clone, Copy)]
struct RewindWindow {
    from: u64,
    until: u64,
}

/// A rewind in progress: the pair is restored at the ancestor and the winning branch is being
/// replayed from the spool.
#[derive(Debug, Clone, Copy)]
struct ActiveRewind {
    /// The installed checkpoint's announce sequence; reaching it again completes the rewind.
    until: u64,
    /// First frame past the installed checkpoint's chunks — the live edge to hop to.
    resume_at: u64,
    /// Disagreement count at install; any growth during the replay downgrades the recovery from
    /// continuous to an explicit reset.
    disagreements_before: usize,
    /// Commits replayed inside the window so far.
    replayed: u64,
}

/// The rewound window a previous run's ack carried beside its restore point.
///
/// Not a dirty flag but the pair's reconstruction recipe: a rewound restore's state is the
/// checkpoint *plus* the window below it, so any resume of that restore point has to replay the
/// window again before the frames past the chunks make sense. All-or-nothing by design — no
/// mid-window progress is persisted — and the record stays on the ack for the life of the
/// restore point, superseded only by the next restore.
#[derive(Debug, Clone, Copy)]
struct AckRewind {
    checkpoint_sequence: u64,
    chunks_end: u64,
    replay_from: u64,
    replay_until: u64,
}

struct PendingRecovery {
    /// The block a continuous recovery has to land on, after supersession.
    target: Option<BlockRef>,
    /// The tip the operative announcement promised, so the restored pair still checks that the
    /// winning branch arrives instead of forgetting the promise across the outage.
    tip: Option<BlockRef>,
    /// Commit frames that went by between the discontinuity and the checkpoint.
    skipped: u64,
    /// The window those commits sit in, when they are replayable: a target was named and the
    /// window is within bounds. Whether the replay actually runs is judged at restore time,
    /// when the checkpoint's own block is in hand.
    rewind: Option<RewindWindow>,
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
    /// Carried through the skim untouched. At the ancestor it equals what the checkpoint would
    /// re-derive; behind a late checkpoint the pair has advanced, and resetting it to the
    /// checkpoint's child would gap the very next commit.
    expected_child: Option<(u64, B256)>,
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

/// What a previous run wrote down about where it got to.
///
/// Only the fields a restart acts on. Everything else in the file is for an operator reading it.
#[derive(Debug, Clone)]
struct Ack {
    /// Highest frame the previous run consumed.
    last_sequence: u64,
    /// The block it stood on there, checked when the catch-up reaches that sequence.
    block: Option<BlockRef>,
    /// What it was doing: `needs_snapshot` means it was already waiting for a checkpoint.
    state: String,
    /// Which epoch it was reading.
    epoch: u64,
    /// The checkpoint frame its pair came from. Absent in version 1 acks.
    restored_from_sequence: Option<u64>,
    /// The rewind transaction the previous run had open, when it died mid-window. Absent on
    /// version 1 and 2 acks and on any run that was not rewinding.
    recovery: Option<AckRewind>,
}

/// Reads an ack file, or reports that there is nothing to resume from.
///
/// A missing file is a fresh start — an operator asking to resume before anything ran should get
/// a run, not an error. A file that exists and does not parse is an error: it was written by
/// something, and guessing what it meant is exactly the guessing this format removed.
fn read_ack(path: &Path) -> eyre::Result<Option<Ack>> {
    let raw = match std::fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(eyre::eyre!("cannot read the ack at {}: {err}", path.display())),
    };
    let value: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|err| eyre::eyre!("the ack at {} is not readable: {err}", path.display()))?;
    let last_sequence = value
        .get("last_sequence")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| eyre::eyre!("the ack at {} names no sequence", path.display()))?;
    let block = match (
        value.get("block").and_then(serde_json::Value::as_u64),
        value.get("block_hash").and_then(serde_json::Value::as_str),
    ) {
        (Some(number), Some(hash)) => Some(BlockRef {
            number,
            hash: hash.parse().map_err(|_| {
                eyre::eyre!("the ack at {} names an unreadable block hash", path.display())
            })?,
        }),
        _ => None,
    };
    Ok(Some(Ack {
        last_sequence,
        block,
        state: value
            .get("state")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("streaming")
            .to_string(),
        // Version 1 predates epochs in the ack, and every version-1 ack was written by a producer
        // that could not resume a spool, so epoch 1 is not a guess.
        epoch: value.get("epoch").and_then(serde_json::Value::as_u64).unwrap_or(1).max(1),
        restored_from_sequence: value
            .get("restored_from_sequence")
            .and_then(serde_json::Value::as_u64),
        // A recovery block that exists but does not parse is refused, not treated as absent: it
        // was written by a run that needed its window replayed, and resuming without the window
        // would rebuild a different pair under the same watermark.
        recovery: match value.get("recovery").filter(|value| !value.is_null()) {
            None => None,
            Some(recovery) => {
                let field = |name: &str| {
                    recovery.get(name).and_then(serde_json::Value::as_u64).ok_or_else(|| {
                        eyre::eyre!(
                            "the ack at {} carries a recovery window with no readable {name}",
                            path.display()
                        )
                    })
                };
                Some(AckRewind {
                    checkpoint_sequence: field("checkpoint_sequence")?,
                    chunks_end: field("chunks_end")?,
                    replay_from: field("replay_from")?,
                    replay_until: field("replay_until")?,
                })
            }
        },
    }))
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
    winning_branches_superseded: u64,
    restores_continuous: u64,
    restores_reset: u64,
    /// Frames read during a recovery scan that did not verify, so nothing was taken from them.
    scan_refusals: u64,
    /// Highest sequence a previous run had already published a verdict for, while catching up.
    ///
    /// A resumed run re-derives everything between the checkpoint it restored from and this
    /// point. Those blocks were verified once already, so their verdicts are labelled and kept
    /// out of the live counters — reporting them again would double-count the same work.
    catch_up_until: Option<u64>,
    /// Blocks re-derived on the way back to the watermark.
    catch_up_blocks: u64,
    /// The sequence a resume started from, for the record.
    resumed_from: Option<u64>,
    /// What the ack said the previous run's head was, checked when the catch-up reaches it.
    resume_watermark: Option<(u64, B256)>,
    /// What the scan that found the next checkpoint was looking for, so the restore can say
    /// whether it got it.
    pending_recovery: Option<PendingRecovery>,
    /// The rewind replay in progress, when a recovery restored at the exact ancestor and its
    /// winning branch is being re-read from the spool.
    rewind_active: Option<ActiveRewind>,
    /// A rewind recorded by the previous run's ack, resumed all-or-nothing: the whole window
    /// replays again after the restore.
    resume_rewind: Option<AckRewind>,
    /// Commits verified inside completed rewind windows. These count in `blocks_verified` too;
    /// this counter is what makes the recovery share visible beside the total.
    rewind_replayed_commits: u64,
    /// Rewind windows refused for exceeding [`MAX_REWIND_FRAMES`]; each degraded to a reset.
    rewind_windows_refused: u64,
    unverified_intervals: Vec<(u64, u64)>,
    last_needs_snapshot: Option<NeedsSnapshotReason>,
    last_verified: Option<BlockRef>,
    last_frame_at: Instant,
    /// Read and decode costs of the frame currently being handled, for the verdict line.
    last_frame_costs: Option<FrameCosts>,
    /// The current frame's mtime — the producer's write instant on this host's clock, an
    /// availability *proxy* whose derived fields are labelled `available_at_source: "mtime"`.
    last_frame_available: Option<SystemTime>,
    /// Read-attempt start minus availability for the current frame, when it was consumed live
    /// and both clocks cooperated.
    last_queue_wait_us: Option<u64>,
    /// Whether the follower has observed the spool's live tail — an empty poll — at least once.
    ///
    /// Until it has, every frame it reads was sitting in the spool before this run got to it:
    /// backlog, not live delivery. An mtime-derived wait on such a frame measures the backlog's
    /// age, not the transport, so the latency fields stay null and the verdict line says
    /// `tail_live: false`. This is a fact about *this run's position*, not about the frames —
    /// a fresh (non-resume) follower re-reading an old spool gets `false` all the way through.
    reached_tail: bool,
    /// Whether the current frame was consumed live — read after the tail had been reached.
    last_frame_live: bool,
    /// Clock readings that went backwards. Counted rather than clamped: a clamped zero would
    /// read as an excellent latency, and an anomaly is a fact about the measurement.
    latency_anomalies: u64,
}

impl<'a> Follower<'a> {
    fn new(dir: &Path, options: &'a FollowOptions) -> eyre::Result<Self> {
        Ok(Self {
            options,
            replay_options: ReplayOptions {
                limit: None,
                mutations: options.mutations,
                // Never here. A transition mutation executes a whole extra block, and a follower
                // is measuring the wall-clock distance between a frame landing and its verdict —
                // the one place where the cost of proving something would be counted as the cost
                // of doing it. It belongs to the offline gate, which measures nothing.
                mutations_transition: None,
                frame_limits: options.frame_limits,
                reexec_limits: options.reexec_limits.clone(),
                // An offline forensic switch; a live follower has a producer to ask instead.
                force_restore_at: None,
                // Not reached from here — the follower drives its own window — but set to the
                // shared bound so the two can never be read as different policies.
                max_rewind_frames: MAX_REWIND_FRAMES,
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
            winning_branches_superseded: 0,
            restores_continuous: 0,
            restores_reset: 0,
            scan_refusals: 0,
            catch_up_until: None,
            catch_up_blocks: 0,
            resumed_from: None,
            resume_watermark: None,
            pending_recovery: None,
            rewind_active: None,
            resume_rewind: None,
            rewind_replayed_commits: 0,
            rewind_windows_refused: 0,
            unverified_intervals: Vec::new(),
            last_needs_snapshot: None,
            last_verified: None,
            last_frame_at: Instant::now(),
            last_frame_costs: None,
            last_frame_available: None,
            last_queue_wait_us: None,
            reached_tail: false,
            last_frame_live: false,
            latency_anomalies: 0,
        })
    }

    fn run(mut self) -> eyre::Result<FollowReport> {
        self.resume()?;
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
                        let sequence = frame.header.sequence;
                        if self.rewind_boundary(sequence) {
                            continue
                        }
                        self.note_frame_costs(&frame);
                        let step = self.handle(sequence, frame.event)?;
                        match self.catch_up_until {
                            // The frame the previous run stopped at has now been re-derived, and
                            // this is the one moment the two runs can be compared. Done or not:
                            // the frame that ends a catch-up is often the End the previous run
                            // consumed, and a divergence there is still a divergence.
                            Some(until) if sequence >= until => {
                                match (self.finish_catch_up()?, step) {
                                    (caught @ Step::Done(_), _) => caught,
                                    (Step::Continue, step) => step,
                                }
                            }
                            _ => step,
                        }
                    }
                    Ok(TailEvent::Idle) => {
                        // The one observation that separates backlog from live delivery: the
                        // spool had nothing new, so every frame from here on arrived while this
                        // follower was already waiting for it. Armed only once a pair has been
                        // restored — a follower started against a pre-checkpoint spool also
                        // observes an empty poll, and arming there is exactly the startup
                        // mislabeling both 1,001-block preflights had to hand-exclude.
                        if self.restores > 0 {
                            self.reached_tail = true;
                        }
                        self.idle()
                    }
                    Err(fault) => self.tail_fault(fault)?,
                }
            };
            match step {
                Step::Continue => {}
                Step::Done(outcome) => return Ok(self.into_report(outcome)),
            }
        }
    }

    /// Starts from where a previous run stopped, if the operator asked and an ack says where.
    ///
    /// Three things have to be true before a single frame is skipped, and none of them is
    /// inferred. The spool has to be the same stream, epoch by epoch, from its first manifest to
    /// the one the ack was written under — otherwise "sequence 900" names a frame in somebody
    /// else's numbering. The sequence the ack points at has to actually hold a checkpoint, in
    /// that epoch. And the state the previous run was in has to be honoured: one that stopped
    /// waiting for a snapshot resumes waiting for a snapshot, because the frames it refused are
    /// still there and replaying past them would accept what it refused.
    ///
    /// Everything from the checkpoint up to the ack's own sequence is then re-derived rather than
    /// trusted. That is the point: the pair is state, and the only honest way to get it back
    /// without a database is to build it again from the frames that built it the first time.
    fn resume(&mut self) -> eyre::Result<()> {
        if !self.options.resume {
            return Ok(())
        }
        let Some(path) = self.options.ack.clone() else {
            return Err(eyre::eyre!("--resume needs --ack: there is nowhere to resume from"))
        };
        let Some(ack) = read_ack(&path)? else {
            info!(
                target: "ps_follow",
                ack = %path.display(),
                "No ack to resume from; following from the start of the stream"
            );
            return Ok(())
        };
        let (manifest_at, manifest) = self.verify_epoch_chain(ack.epoch)?;
        if self.resume_past_a_closed_epoch(&ack, &manifest)? {
            return Ok(())
        }
        let Some(restored_from) = ack.restored_from_sequence else {
            // A version-1 ack. Its watermark is real, but it never recorded which checkpoint the
            // pair came from, and picking "the newest checkpoint at or below the watermark" would
            // let a restart adopt a checkpoint the previous run had examined and refused. Reading
            // the epoch from its start costs time and claims nothing false.
            warn!(
                target: "ps_follow",
                ack = %path.display(),
                "The ack does not say which checkpoint its pair came from, so the epoch is read \
                 from its start rather than guessed at"
            );
            self.tail.skip_to(manifest_at);
            return Ok(())
        };
        // Inside the epoch it claims, and at or below the watermark it claims. A checkpoint above
        // the watermark was never the pair that wrote it, and one below the manifest belongs to a
        // different epoch — both are acks that cannot be acted on rather than acks to guess at.
        if restored_from < manifest_at || restored_from > ack.last_sequence {
            return Err(eyre::eyre!(
                "the ack restored from sequence {restored_from}, which is outside epoch {}'s own \
                 frames [{manifest_at}, {}]",
                ack.epoch,
                ack.last_sequence
            ))
        }
        // Read by name and header, like every other recovery read: a restart is exactly when a
        // renamed frame would be aimed at a consumer.
        let frame = self
            .tail
            .read_at(restored_from, FrameKind::Checkpoint)
            .map_err(|fault| eyre::eyre!("the ack's checkpoint did not read back: {fault}"))?;
        let StreamEvent::Checkpoint(checkpoint) = frame.event else {
            return Err(eyre::eyre!(
                "the frame at sequence {restored_from} is named Checkpoint but decoded as \
                 something else"
            ))
        };
        self.resumed_from = Some(restored_from);
        self.tail.skip_to(restored_from);
        // Identity is already established — `verify_epoch_chain` read it from the spool's own
        // manifests and checked every link — so the checkpoint about to arrive has something to
        // be checked against, which is the whole reason the chain is walked before this point.
        self.phase = Phase::AwaitingCheckpoint { manifest };
        self.catch_up_until = Some(ack.last_sequence);
        // A rewound restore point carries its window on the ack, and a resume of it replays the
        // window all-or-nothing: restore at the checkpoint, then re-read the *whole* recorded
        // range. No mid-window progress was persisted, deliberately — re-deriving is this
        // format's restart philosophy, and the acks written below the checkpoint's sequence were
        // dropped by the monotonic guard anyway. The window is checked against the spool before
        // it is believed: an ack that names a different boundary than the checkpoint it points
        // at was not written against this spool.
        if let Some(recovery) = ack.recovery {
            if recovery.checkpoint_sequence != restored_from {
                return Err(eyre::eyre!(
                    "the ack's recovery window names checkpoint sequence {}, but the ack \
                     restored from {restored_from}; the file contradicts itself",
                    recovery.checkpoint_sequence
                ))
            }
            let chunks_end = restored_from + u64::from(checkpoint.snapshot_chunks);
            if recovery.replay_until != restored_from ||
                recovery.chunks_end != chunks_end ||
                recovery.replay_from >= recovery.replay_until ||
                recovery.replay_from < manifest_at
            {
                return Err(eyre::eyre!(
                    "the ack's recovery window [{}, {}) with chunks through {} does not describe \
                     the checkpoint at sequence {restored_from} with chunks through {chunks_end}; \
                     the ack was not written against this spool",
                    recovery.replay_from,
                    recovery.replay_until,
                    recovery.chunks_end
                ))
            }
            info!(
                target: "ps_follow",
                replay_from = recovery.replay_from,
                replay_until = recovery.replay_until,
                "The ack's restore point is a rewound recovery; the whole window replays again \
                 after the restore"
            );
            self.resume_rewind = Some(recovery);
        }
        // Checked only against a watermark a verdict wrote. A run that stopped waiting for a
        // snapshot will re-hit whatever stopped it before it gets this far, and that is not a
        // divergence — it is the same refusal, reached the same way, which is what makes a
        // restart safe rather than a way past it.
        self.resume_watermark = (ack.state == "streaming")
            .then_some(ack.block)
            .flatten()
            .map(|block| (block.number, block.hash));
        self.sink.suppress_ack = true;
        self.sink.ack_high_water = Some(ack.last_sequence);
        self.replay_options.mutations = false;
        info!(
            target: "ps_follow",
            restored_from,
            catch_up_until = ack.last_sequence,
            epoch = ack.epoch,
            "Resuming: the pair is rebuilt from the checkpoint the ack names and re-derived up to \
             its watermark before any new verdict is published"
        );
        Ok(())
    }

    /// Starts the *next* epoch when the one the ack was written under has already closed.
    ///
    /// A producer that was stopped and restarted into its own spool leaves an `End` and then a
    /// new epoch above it. A follower resuming an ack from below that `End` has nothing to catch
    /// up to: replaying the closed epoch would reach the same `End` and stop there, every time,
    /// and the frames it actually needs are the ones above it.
    ///
    /// So the boundary is crossed here, and only on evidence. The ack's own sequence has to hold
    /// the `End` it claims to have consumed, and the frame after it has to be a manifest that
    /// succeeds the one this ack was written under. Whatever that epoch restores from is an
    /// explicit checkpoint reset — the epoch says the producer's state broke — so the interval
    /// between the two runs is recorded as unverified rather than quietly closed.
    ///
    /// Returns whether the boundary was crossed. `false` means the ordinary catch-up applies.
    fn resume_past_a_closed_epoch(&mut self, ack: &Ack, current: &Manifest) -> eyre::Result<bool> {
        if ack.state != "ended" {
            return Ok(false)
        }
        let end = self
            .tail
            .read_at(ack.last_sequence, FrameKind::End)
            .map_err(|fault| eyre::eyre!("the ack's End frame did not read back: {fault}"))?;
        let StreamEvent::End(end) = end.event else {
            return Err(eyre::eyre!(
                "the ack says epoch {} ended at sequence {}, and that frame is not an End",
                ack.epoch,
                ack.last_sequence
            ))
        };
        self.check_end_numbering(&end.reason, end.last_sequence, ack.last_sequence)?;
        let next_at = ack.last_sequence + 1;
        let Ok(frame) = self.tail.read_at(next_at, FrameKind::Manifest) else {
            // Nothing above the End yet: the producer has not come back. Fall through, replay the
            // closed epoch, and stop at the same End the previous run did — which is the honest
            // answer, and the run that follows a restarted producer will cross here instead.
            info!(
                target: "ps_follow",
                epoch = ack.epoch,
                "The epoch this ack was written under is closed and nothing follows it yet"
            );
            return Ok(false)
        };
        let StreamEvent::Manifest(next) = frame.event else {
            return Err(eyre::eyre!(
                "the frame at sequence {next_at} is named Manifest but decoded as something else"
            ))
        };
        next.check_succeeds(current, next_at).map_err(|err| {
            eyre::eyre!("the frame after epoch {}'s End is not its successor: {err}", ack.epoch)
        })?;
        info!(
            target: "ps_follow",
            from_epoch = ack.epoch,
            to_epoch = next.epoch,
            manifest_at = next_at,
            "The producer restarted into this spool; resuming at the epoch above the End rather \
             than replaying the one below it"
        );
        // What the previous run stood on, so the interval this restart does not validate can be
        // named rather than merely counted.
        self.last_verified = ack.block;
        // No target: an epoch boundary names no block, so whatever it restores from is a reset.
        self.pending_recovery =
            Some(PendingRecovery { target: None, tip: None, skipped: 0, rewind: None });
        self.resumed_from = Some(next_at);
        self.tail.skip_to(next_at);
        self.phase = Phase::AwaitingManifest;
        Ok(true)
    }

    /// Ends a resume's catch-up, checking that this run reached where the last one said it was.
    ///
    /// The check is the whole reason the ack carries a block beside a sequence. Two runs reading
    /// the same frames from the same checkpoint must arrive at the same block; if they do not,
    /// something between them is not deterministic, and continuing would publish verdicts from a
    /// pair that is already provably not the one that wrote the watermark.
    fn finish_catch_up(&mut self) -> eyre::Result<Step> {
        self.catch_up_until = None;
        self.sink.suppress_ack = false;
        self.replay_options.mutations = self.options.mutations;
        let Some((number, hash)) = self.resume_watermark.take() else {
            info!(target: "ps_follow", catch_up_blocks = self.catch_up_blocks, "Caught up");
            return Ok(Step::Continue)
        };
        match self.last_verified {
            Some(head) if head.number == number && head.hash == hash => {
                info!(
                    target: "ps_follow",
                    block = number,
                    catch_up_blocks = self.catch_up_blocks,
                    "Caught up to the watermark the previous run left, on the same block it did"
                );
                Ok(Step::Continue)
            }
            other => {
                let detail = format!(
                    "resume divergence: the ack says block {number}/{hash:?}, and replaying the \
                     same frames from the same checkpoint reached {other:?}"
                );
                error!(target: "ps_follow", %detail, "Refusing to publish from a diverged pair");
                self.sink.state("faulted", "resume_divergence", &detail, self.last_verified)?;
                Ok(Step::Done(FollowOutcome::Faulted { detail }))
            }
        }
    }

    /// Walks the manifests from the spool's first to `epoch`, checking each follows the last.
    ///
    /// Returns the sequence of `epoch`'s own manifest, which is where that epoch's frames begin.
    /// A chain that does not hold is a hard error rather than a fresh start: the ack's sequence
    /// numbers were written in a numbering this spool does not have, so nothing in it can be
    /// acted on, and quietly starting over would hide that from whoever asked to resume.
    fn verify_epoch_chain(&self, epoch: u64) -> eyre::Result<(u64, Manifest)> {
        let mut at = 0;
        let mut previous: Option<Manifest> = None;
        loop {
            let frame = self
                .tail
                .read_at(at, FrameKind::Manifest)
                .map_err(|fault| eyre::eyre!("epoch {epoch} could not be traced: {fault}"))?;
            let StreamEvent::Manifest(found) = frame.event else {
                return Err(eyre::eyre!(
                    "the frame at sequence {at} is named Manifest but decoded as something else"
                ))
            };
            match &previous {
                None => {
                    chain_spec_for(&found)?;
                    config_for(&found)?;
                    found.check_opens(at)?;
                }
                Some(previous) => found.check_succeeds(previous, at)?,
            }
            if found.epoch == epoch {
                return Ok((at, found))
            }
            if found.epoch > epoch {
                return Err(eyre::eyre!(
                    "the ack names epoch {epoch}, and this spool's epochs go {} then {}",
                    found.epoch - 1,
                    found.epoch
                ))
            }
            let next = self
                .tail
                .scan_for(FrameKind::Manifest, at + 1)
                .map_err(|fault| eyre::eyre!("epoch {epoch} could not be traced: {fault}"))?
                .ok_or_else(|| {
                    eyre::eyre!(
                        "the ack names epoch {epoch}, and this spool ends in epoch {}",
                        found.epoch
                    )
                })?;
            previous = Some(found);
            at = next;
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
                manifest.check_opens(sequence)?;
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
            (Phase::AwaitingCheckpoint { manifest }, StreamEvent::Manifest(next)) => {
                match next.check_succeeds(&manifest, sequence) {
                    Ok(()) => Ok(self.enter_needs_snapshot(
                        next,
                        NeedsSnapshotReason::EpochChange,
                        "a second manifest arrived before any checkpoint of the first",
                        sequence + 1,
                        None,
                    )),
                    Err(err) => Ok(self.enter_needs_snapshot(
                        manifest,
                        NeedsSnapshotReason::ProtocolViolation,
                        &format!(
                            "a second manifest arrived that is not this stream's next epoch: {err}"
                        ),
                        sequence + 1,
                        None,
                    )),
                }
            }
            (Phase::AwaitingCheckpoint { manifest }, other) => Ok(self.enter_needs_snapshot(
                manifest,
                NeedsSnapshotReason::ProtocolViolation,
                &format!("a {} frame arrived before any checkpoint", kind_of(&other).as_str()),
                sequence + 1,
                None,
            )),

            (
                Phase::CollectingChunks { manifest, checkpoint, checkpoint_sequence, mut chunks },
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
                    self.phase = Phase::CollectingChunks {
                        manifest,
                        checkpoint,
                        checkpoint_sequence,
                        chunks,
                    };
                    return Ok(Step::Continue)
                }
                self.restore_pair(manifest, checkpoint, chunks, checkpoint_sequence)
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
        // The rewind window's contract, enforced rather than assumed: the producer's fencing
        // makes a published late checkpoint's window Commit-only, but an independent validator
        // refuses a spool that breaks the promise instead of acting on whatever sits there —
        // a reorg frame inside a window would otherwise be *applied*, mid-replay, as though the
        // chain had moved. The scan cannot deliver these shapes from a healthy spool; a resumed
        // ack against a doctored one can.
        if self.rewind_active.is_some() && !matches!(event, StreamEvent::Commit(_)) {
            return Ok(self.enter_needs_snapshot(
                manifest,
                NeedsSnapshotReason::ProtocolViolation,
                &format!(
                    "a {} frame sits inside a rewind window, which holds only commits",
                    kind_of(&event).as_str()
                ),
                sequence + 1,
                None,
            ))
        }
        match event {
            StreamEvent::Commit(commit) => {
                let (input, oracle) = commit.split();
                let block = input.block;
                // Captured before `replay_commit` consumes the input: the equivalence claim is
                // that the validator consumed the *recorded* Engine payload, and provenance is how
                // each verdict line proves which kind of payload it was measured against.
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
                let producer_durability_watermark = oracle.durability_watermark;
                let costs = self.last_frame_costs.unwrap_or(FrameCosts {
                    sequence,
                    delivery_us: None,
                    frame_decode_us: None,
                    validation_open: None,
                });
                let outcome = replay_commit(
                    &mut state,
                    input,
                    &oracle,
                    &self.replay_options,
                    costs,
                    &mut self.replay,
                );
                let new_disagreements = self.replay.disagreements.len() - before_disagreements;
                match outcome {
                    CommitOutcome::Compared => {
                        let verdict = if new_disagreements == 0 { "accepted" } else { "disagreed" };
                        // Re-derived on the way back to a watermark, not verified anew. Counted
                        // apart and labelled in the record, because a resumed run that added them
                        // to its own total would report the same work twice.
                        let catch_up = self.catch_up_until.is_some();
                        // First-time verification replayed out of the spool during a recovery
                        // rewind: counted in `blocks_verified` — excluding it would hide both
                        // the verified blocks and the recovery's cost — with the live-latency
                        // fields null, because a re-read frame's mtime distance measures the
                        // outage, not the transport.
                        let recovery_replay = self.rewind_active.is_some();
                        if catch_up {
                            self.catch_up_blocks += 1;
                        } else {
                            self.blocks_verified += 1;
                        }
                        if let Some(rewind) = self.rewind_active.as_mut() {
                            rewind.replayed += 1;
                        }
                        self.last_verified = Some(block);
                        let (queue_wait_us, decision_latency_us, available_at_source) =
                            self.latency_fields(catch_up || recovery_replay);
                        self.sink.verdict(Published {
                            verdict,
                            block,
                            sequence,
                            provenance,
                            disagreements: new_disagreements,
                            catch_up,
                            recovery_replay,
                            timing: attempt_timing(&self.replay, sequence),
                            tail_live: self.last_frame_live && !catch_up && !recovery_replay,
                            queue_wait_us,
                            decision_latency_us,
                            available_at_source,
                            producer_durability_watermark,
                        })?;
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
                            // Preserved across commits: under write-through publication the
                            // recovery checkpoint arrives *behind* the winning branch, and this
                            // expectation is what lets the follower skim it there instead of
                            // refusing it as unannounced. It resolves at the matching skim, is
                            // replaced by the next lifecycle event, and is counted — not failed —
                            // if the stream ends first, because `never` publishes no checkpoint
                            // at all.
                            announced,
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
                        let catch_up = self.catch_up_until.is_some();
                        let recovery_replay = self.rewind_active.is_some();
                        let (queue_wait_us, decision_latency_us, available_at_source) =
                            self.latency_fields(catch_up || recovery_replay);
                        self.sink.verdict(Published {
                            verdict: "rejected",
                            block,
                            sequence,
                            provenance,
                            disagreements: new_disagreements,
                            catch_up,
                            recovery_replay,
                            timing: attempt_timing(&self.replay, sequence),
                            tail_live: self.last_frame_live && !catch_up && !recovery_replay,
                            queue_wait_us,
                            decision_latency_us,
                            available_at_source,
                            producer_durability_watermark,
                        })?;
                        self.sink.state(
                            "faulted",
                            "rejected_commit",
                            &detail,
                            self.last_verified,
                        )?;
                        // Reinstalled so the final report reads the truth off the phase — a
                        // fault with a recovery expectation still pending is still a pending
                        // expectation.
                        self.phase = Phase::Streaming {
                            manifest,
                            state,
                            expected_child: None,
                            announced,
                            pending_tip,
                        };
                        Ok(Step::Done(FollowOutcome::Faulted { detail }))
                    }
                    CommitOutcome::Fault(fault) => {
                        let detail = fault.to_string();
                        let catch_up = self.catch_up_until.is_some();
                        let recovery_replay = self.rewind_active.is_some();
                        let (queue_wait_us, decision_latency_us, available_at_source) =
                            self.latency_fields(catch_up || recovery_replay);
                        self.sink.verdict(Published {
                            verdict: "fault",
                            block,
                            sequence,
                            provenance,
                            disagreements: new_disagreements,
                            catch_up,
                            recovery_replay,
                            timing: attempt_timing(&self.replay, sequence),
                            tail_live: self.last_frame_live && !catch_up && !recovery_replay,
                            queue_wait_us,
                            decision_latency_us,
                            available_at_source,
                            producer_durability_watermark,
                        })?;
                        self.sink.state("faulted", "fault", &detail, self.last_verified)?;
                        error!(
                            target: "ps_follow",
                            block = block.number,
                            %detail,
                            readiness = state.pair.readiness.state().label(),
                            "The pair can go no further; no further verdict will be published"
                        );
                        self.phase = Phase::Streaming {
                            manifest,
                            state,
                            expected_child: None,
                            announced,
                            pending_tip,
                        };
                        Ok(Step::Done(FollowOutcome::Faulted { detail }))
                    }
                }
            }
            // The producer restarted. Its state broke, so nothing below this can be continued
            // into — but the stream itself goes on, and the follower reads the next epoch under
            // the new identity once it has checked that it *is* the next epoch of this one.
            StreamEvent::Manifest(next) => match next.check_succeeds(&manifest, sequence) {
                Ok(()) => Ok(self.enter_needs_snapshot(
                    next,
                    NeedsSnapshotReason::EpochChange,
                    "the producer restarted its stream; the next checkpoint rebootstraps this \
                     follower",
                    sequence + 1,
                    None,
                )),
                Err(err) => Ok(self.enter_needs_snapshot(
                    manifest,
                    NeedsSnapshotReason::ProtocolViolation,
                    &format!(
                        "a second manifest arrived that is not this stream's next epoch: {err}"
                    ),
                    sequence + 1,
                    None,
                )),
            },
            StreamEvent::Checkpoint(checkpoint) => match announced {
                // The producer's recovery checkpoint for a reorg this follower already undid.
                // Two implementations reached the same generation independently, so comparing
                // them is a live disagreement detector; the bytes are read for their digest and
                // dropped, because installing would replace derived state with state this
                // consumer has no reason to prefer.
                Some(ancestor) => self.begin_skim(
                    StreamingPhase {
                        manifest,
                        state,
                        expected_child,
                        announced: None,
                        pending_tip,
                    },
                    checkpoint,
                    ancestor,
                    sequence,
                ),
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
                // A branch still being delivered is not a hole when the producer itself replaces
                // it: the blocks between here and the old tip never became canonical, so no
                // verdict was ever owed on them. It is a hole only when nothing valid takes its
                // place, which each outcome below decides for itself.
                let superseded = pending_tip;
                let outcome = apply_reorg(&mut state, &found);
                match (superseded, outcome.withdraws_an_announced_branch()) {
                    (Some(tip), true) => self.note_supersession(tip, found.winning_tip),
                    (Some(_), false) => self.winning_branches_incomplete += 1,
                    (None, _) => {}
                }
                match outcome {
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
                    // Bound to this follower's own branch: the ancestor is a block it verified,
                    // so a checkpoint at that exact block resumes it with no gap, and the branch
                    // this reorg interrupted was withdrawn by a statement it could authenticate.
                    // The winning tip travels into the recovery with the target: the branch's
                    // completion is still owed a check after the restore.
                    ReorgOutcome::Unrecoverable { ancestor, depth, detail } => {
                        warn_inapplicable(ancestor, depth, &detail, true);
                        Ok(self.enter_needs_snapshot_carrying(
                            manifest,
                            NeedsSnapshotReason::SnapshotRequired,
                            &detail,
                            sequence + 1,
                            Some(ancestor),
                            found.winning_tip,
                        ))
                    }
                    // Well-formed, but about a branch this follower never held. The ancestor is
                    // hearsay, so it is not offered as a recovery target: a checkpoint landing on
                    // it would otherwise be reported as a continuous recovery of a branch this
                    // follower cannot show it was ever on.
                    ReorgOutcome::Unbound { ancestor, depth, detail } => {
                        warn_inapplicable(ancestor, depth, &detail, false);
                        Ok(self.enter_needs_snapshot(
                            manifest,
                            NeedsSnapshotReason::SnapshotRequired,
                            &detail,
                            sequence + 1,
                            None,
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
            StreamEvent::Reset(reset) => {
                // A reset withdraws the stream without naming a replacement branch, so a tip that
                // never arrived stays a hole.
                self.winning_branches_incomplete += u64::from(pending_tip.is_some());
                Ok(self.enter_needs_snapshot(
                    manifest,
                    reset.reason.into(),
                    &reset.detail,
                    sequence + 1,
                    None,
                ))
            }
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
        phase: StreamingPhase,
        checkpoint: Checkpoint,
        ancestor: BlockRef,
        sequence: u64,
    ) -> eyre::Result<Step> {
        let StreamingPhase { manifest, state, expected_child, pending_tip, .. } = phase;
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
            if consumer_is_at(&state, ancestor) {
                // The operator-trusted checkpoint source is the authority on what is canonical
                // here — it is the same authority this follower bootstrapped under — so the run
                // continues from its checkpoint rather than from a generation it contradicts.
                // Recorded as non-agreeing, and re-entered through the scan so the restore is
                // classified: it will be an explicit reset, because a disputed ancestor cannot
                // anchor a continuous claim. The announced tip is let go with it; the reset
                // already says the run has a hole.
                return Ok(self.enter_needs_snapshot(
                    manifest,
                    NeedsSnapshotReason::SnapshotRequired,
                    "the recovery checkpoint disagrees with the recovered generation",
                    sequence,
                    None,
                ))
            }
            // Late: this follower verified every block past the ancestor for itself, against
            // canonical headers and its own roots, and that chain outranks a late cross-check.
            // Installing would rewind the pair below commits it already stands behind. The
            // disagreement above already fails the run's agreement claim; validation continues,
            // the expectation is dropped, and the chunks are still read off the stream for their
            // declared digest — a transport fault stays a separate finding.
            self.replay.late_skim_mismatches += 1;
        } else {
            self.checkpoints_skimmed += 1;
            info!(
                target: "ps_follow",
                block = checkpoint.block.number,
                chunks = checkpoint.snapshot_chunks,
                "The producer's recovery checkpoint agrees with the generation this follower \
                 recovered to; its snapshot is checked and not installed"
            );
            // Recorded with its sequence, because agreeing with a checkpoint is not the same as
            // having shown its snapshot would restore anything — and the offline proof that it
            // would needs to be told which frame to install.
            self.sink.skimmed(checkpoint.block, sequence, self.catch_up_until.is_some())?;
        }
        if checkpoint.snapshot_chunks == 0 {
            return self.finish_skim(
                SkimPhase {
                    manifest,
                    state,
                    checkpoint,
                    expected_child,
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
            expected_child,
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
            expected_child,
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
                expected_child,
                pending_tip,
                received,
                accumulated,
                digest,
            }));
            return Ok(Step::Continue)
        }
        self.finish_skim(
            SkimPhase {
                manifest,
                state,
                checkpoint,
                expected_child,
                pending_tip,
                received,
                accumulated,
                digest,
            },
            sequence,
        )
    }

    /// Checks what the skim accumulated against what the checkpoint declared, and resumes.
    fn finish_skim(&mut self, phase: SkimPhase, sequence: u64) -> eyre::Result<Step> {
        let SkimPhase {
            manifest,
            state,
            checkpoint,
            expected_child,
            pending_tip,
            accumulated,
            digest,
            ..
        } = phase;
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
        // The child constraint entered the skim untouched and leaves it untouched. At the
        // ancestor it is already `(checkpoint.block + 1, checkpoint.block.hash)` from the undo
        // that announced this checkpoint; behind a late one, the pair has advanced and resetting
        // it to the checkpoint's child would refuse the very next commit as a gap.
        self.phase =
            Phase::Streaming { manifest, state, expected_child, announced: None, pending_tip };
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
        self.phase = Phase::CollectingChunks {
            manifest,
            checkpoint,
            checkpoint_sequence: sequence,
            chunks: Vec::new(),
        };
        Ok(Step::Continue)
    }

    /// Restores a fresh pair from a complete checkpoint+snapshot and opens the verdict stream.
    fn restore_pair(
        &mut self,
        manifest: Manifest,
        checkpoint: Checkpoint,
        chunks: Vec<SnapshotChunk>,
        checkpoint_sequence: u64,
    ) -> eyre::Result<Step> {
        match restore(&manifest, &checkpoint, &chunks) {
            Ok(state) => {
                self.restores += 1;
                let expected_child = Some((checkpoint.block.number + 1, checkpoint.block.hash));
                let resume_rewind = self.resume_rewind.take();
                let pending = self.pending_recovery.take();
                // The tip the interrupted reorg promised, still owed a check after the restore.
                // A checkpoint at the tip itself completes the branch; one at or above its
                // height settles the question the reset classification already answers; below
                // it, the expectation rides into the streaming phase as `pending_tip`.
                let carried_tip =
                    pending.as_ref().and_then(|pending| pending.tip).and_then(|tip| {
                        if checkpoint.block == tip {
                            self.winning_branches_completed += 1;
                            None
                        } else if checkpoint.block.number >= tip.number {
                            None
                        } else {
                            Some(tip)
                        }
                    });
                // A rewind runs when the checkpoint landed exactly on the target with its
                // winning-branch commits below it in the spool — or when a previous run's ack
                // recorded one mid-flight, which replays whole. Classification then waits for
                // the replay: continuous is a claim about those commits, and they have not been
                // verified yet.
                let window = match (&resume_rewind, &pending) {
                    (Some(ack), _) => {
                        Some(RewindWindow { from: ack.replay_from, until: ack.replay_until })
                    }
                    (None, Some(recovery)) if recovery.target == Some(checkpoint.block) => {
                        recovery.rewind
                    }
                    _ => None,
                };
                if let Some(window) = window {
                    let resume_at = checkpoint_sequence + 1 + u64::from(checkpoint.snapshot_chunks);
                    self.rewind_active = Some(ActiveRewind {
                        until: window.until,
                        resume_at,
                        disagreements_before: self.replay.disagreements.len(),
                        replayed: 0,
                    });
                    self.sink.set_rewind(Some(AckRewind {
                        checkpoint_sequence,
                        chunks_end: resume_at.saturating_sub(1),
                        replay_from: window.from,
                        replay_until: window.until,
                    }));
                    info!(
                        target: "ps_follow",
                        block = checkpoint.block.number,
                        restores = self.restores,
                        replay_from = window.from,
                        replay_until = window.until,
                        "Restored at the exact ancestor; the winning branch replays from the \
                         spool before the live edge resumes"
                    );
                    self.sink.restored(
                        checkpoint.block,
                        None,
                        self.last_verified,
                        checkpoint_sequence,
                        manifest.epoch,
                        true,
                    )?;
                    self.tail.skip_to(window.from);
                    self.phase = Phase::Streaming {
                        manifest,
                        state: Box::new(state),
                        expected_child,
                        announced: None,
                        pending_tip: carried_tip,
                    };
                    return Ok(Step::Continue)
                }
                // The deferred skip count lands here for the candidate the scan could not judge:
                // the checkpoint anchored somewhere other than the target, so the window's
                // commits really were skipped, not replayed.
                if let Some(recovery) = &pending &&
                    recovery.rewind.is_some()
                {
                    self.commits_skipped_in_recovery += recovery.skipped;
                }
                // Decided by the scan that found this checkpoint, because only the scan knows
                // what was asked for and what went by in between. The first restore of a stream
                // is neither: nothing preceded it to be continuous with.
                let class = pending.map(|pending| {
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
                // This restore's pair is the checkpoint alone, so any window a previous restore
                // point carried on the ack is superseded here, before the ack that names the new
                // restore point is written.
                self.sink.set_rewind(None);
                info!(
                    target: "ps_follow",
                    block = checkpoint.block.number,
                    restores = self.restores,
                    classification = class.map(RecoveryClass::as_str),
                    "Restored a pair with no database; verdicts start at the next block"
                );
                self.sink.restored(
                    checkpoint.block,
                    class,
                    self.last_verified,
                    checkpoint_sequence,
                    manifest.epoch,
                    false,
                )?;
                self.phase = Phase::Streaming {
                    manifest,
                    state: Box::new(state),
                    expected_child,
                    announced: None,
                    pending_tip: carried_tip,
                };
                Ok(Step::Continue)
            }
            Err(err) => Ok(self.enter_needs_snapshot(
                manifest,
                NeedsSnapshotReason::ProtocolViolation,
                &format!("the checkpoint's snapshot did not verify: {err:#}"),
                checkpoint_sequence + 1,
                None,
            )),
        }
    }

    /// Hops over the installed checkpoint when a rewind replay reaches it again, and settles the
    /// recovery's classification: continuous when every rewound commit agreed, an explicit reset
    /// when one disagreed. Returns whether the frame at `sequence` was consumed by the hop.
    fn rewind_boundary(&mut self, sequence: u64) -> bool {
        let Some(rewind) = self.rewind_active else { return false };
        if sequence < rewind.until {
            return false
        }
        self.rewind_active = None;
        self.tail.skip_to(rewind.resume_at);
        let clean = self.replay.disagreements.len() == rewind.disagreements_before;
        if clean {
            self.restores_continuous += 1;
        } else {
            self.restores_reset += 1;
        }
        self.rewind_replayed_commits += rewind.replayed;
        // The ack's recovery window is deliberately *not* cleared here: it is the restore
        // point's reconstruction recipe, not a dirty flag. A resume of this restore point — a
        // kill one frame from now included — has to replay the window to rebuild the same pair,
        // and an ack that named the checkpoint without the window would gap at the first live
        // commit past the chunks. The next restore supersedes it.
        info!(
            target: "ps_follow",
            replayed = rewind.replayed,
            continuous = clean,
            resume_at = rewind.resume_at,
            "The rewound winning branch is fully replayed; resuming at the live edge"
        );
        true
    }

    /// Abandons a rewind that cannot finish — a fault, a fresh discontinuity — as the explicit
    /// reset it now is. The ack's recovery window stays: a resume replays it and re-hits the
    /// same discontinuity the same way, which is what makes a restart a repetition rather than
    /// a way past a refusal. The next restore supersedes the window.
    fn abort_rewind(&mut self, why: &str) {
        if let Some(rewind) = self.rewind_active.take() {
            self.restores_reset += 1;
            self.rewind_replayed_commits += rewind.replayed;
            warn!(
                target: "ps_follow",
                replayed = rewind.replayed,
                until = rewind.until,
                why,
                "A rewind replay stopped before its window closed; the recovery is an explicit \
                 reset"
            );
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
        self.enter_needs_snapshot_carrying(
            manifest,
            reason,
            detail,
            scan_from,
            target_ancestor,
            None,
        )
    }

    /// [`Self::enter_needs_snapshot`] with the winning tip an interrupting reorg announced, so
    /// the eventual restore keeps checking the branch's arrival. Only a recovery whose target
    /// came from an authenticated reorg carries one; every other discontinuity names no branch.
    fn enter_needs_snapshot_carrying(
        &mut self,
        manifest: Manifest,
        reason: NeedsSnapshotReason,
        detail: &str,
        scan_from: u64,
        target_ancestor: Option<BlockRef>,
        announced_tip: Option<BlockRef>,
    ) -> Step {
        self.needs_snapshot_entries += 1;
        self.last_needs_snapshot = Some(reason);
        self.abort_rewind("a discontinuity interrupted the rewind");
        if self.catch_up_until.take().is_some() {
            // The resumed run re-hit whatever stopped the previous one before reaching its
            // watermark. Expected, and not a divergence — but the catch-up is over, so this run
            // starts writing its own acks again rather than staying silent for its whole life.
            self.sink.suppress_ack = false;
            self.replay_options.mutations = self.options.mutations;
            self.resume_watermark = None;
        }
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
            self.tail.next_sequence().saturating_sub(1),
        ) {
            warn!(target: "ps_follow", error = %err, "Could not record the state transition");
        }
        self.phase = Phase::NeedsSnapshot {
            manifest,
            reason,
            scan_from,
            target_ancestor,
            announced_tip,
            epoch,
        };
        Step::Continue
    }

    /// Records that a valid new announcement replaced a branch that was still being delivered.
    ///
    /// Deliberately not counted as incomplete. The producer withdrew the old tip as a canonical
    /// goal, so the blocks between where delivery got to and where it had been heading never
    /// became canonical — counting them would fail a run for following the chain correctly.
    fn note_supersession(&mut self, superseded: BlockRef, replacement: Option<BlockRef>) {
        self.winning_branches_superseded += 1;
        info!(
            target: "ps_follow",
            superseded = superseded.number,
            replacement = ?replacement.map(|tip| tip.number),
            "A second reorg replaced the winning branch before it finished; the old tip is \
             withdrawn, not missing"
        );
    }

    /// The block a recovery still has to land on, after everything announced in the meantime.
    ///
    /// The producer keeps talking during an outage. A later reorg replaces the target with its own
    /// ancestor; a later reset or manifest withdraws it entirely, because neither names a block
    /// and any checkpoint following them is a reset by construction. Read through the same
    /// name-versus-header authority as ordinary delivery — recovery is exactly where a renamed
    /// frame would be aimed.
    fn supersede_target(
        &mut self,
        target: Option<BlockRef>,
        tip: Option<BlockRef>,
        current: &Manifest,
        scan_from: u64,
        checkpoint_at: u64,
    ) -> eyre::Result<Supersession> {
        let mut adopt = None;
        let mut latest: Option<(u64, Option<BlockRef>, Option<BlockRef>)> = None;
        for kind in [FrameKind::Reorg, FrameKind::Reset, FrameKind::Manifest] {
            let found = self
                .tail
                .last_before(kind, scan_from, checkpoint_at)
                .map_err(|fault| eyre::eyre!("the recovery scan failed: {fault}"))?;
            let Some(sequence) = found else { continue };
            if latest.is_some_and(|(seen, _, _)| seen >= sequence) {
                continue
            }
            let (announced, announced_tip) = match kind {
                FrameKind::Reorg => {
                    let frame = self
                        .tail
                        .read_at(sequence, kind)
                        .map_err(|fault| eyre::eyre!("the recovery scan failed: {fault}"))?;
                    let StreamEvent::Reorg(reorg) = frame.event else {
                        return Err(eyre::eyre!(
                            "the frame at sequence {sequence} is named Reorg but decoded as \
                             something else"
                        ))
                    };
                    // Judged by the same rules a reorg gets while streaming. An unchecked frame
                    // could name any block as the ancestor, and a checkpoint landing there would
                    // then be reported as a continuous recovery of a branch nobody described.
                    match check_shape(&reorg) {
                        Ok(()) => (Some(reorg.common_ancestor), reorg.winning_tip),
                        Err(detail) => {
                            self.refuse_in_scan(sequence, "reorg", &detail)?;
                            (None, None)
                        }
                    }
                }
                FrameKind::Manifest => {
                    let frame = self
                        .tail
                        .read_at(sequence, kind)
                        .map_err(|fault| eyre::eyre!("the recovery scan failed: {fault}"))?;
                    let StreamEvent::Manifest(next) = frame.event else {
                        return Err(eyre::eyre!(
                            "the frame at sequence {sequence} is named Manifest but decoded as \
                             something else"
                        ))
                    };
                    // A new epoch withdraws the target either way. It is still checked, because
                    // the checkpoint that follows will be restored under an identity, and one
                    // that fails here says the spool is not the stream this follower was reading.
                    match next.check_succeeds(current, sequence) {
                        Ok(()) => adopt = Some(next),
                        Err(err) => self.refuse_in_scan(sequence, "manifest", &err.to_string())?,
                    }
                    (None, None)
                }
                // A reset names no block, so nothing after it can be continuous with anything.
                _ => (None, None),
            };
            latest = Some((sequence, announced, announced_tip));
        }
        match latest {
            Some((sequence, announced, announced_tip)) => {
                info!(
                    target: "ps_follow",
                    sequence,
                    superseded = ?target.map(|block| block.number),
                    now = ?announced.map(|block| block.number),
                    epoch_changed = adopt.is_some(),
                    "A later announcement supersedes the recovery target"
                );
                Ok(Supersession {
                    target: announced,
                    tip: announced_tip,
                    adopt,
                    at: Some(sequence),
                })
            }
            None => Ok(Supersession { target, tip, adopt, at: None }),
        }
    }

    /// Records a frame the recovery scan read and would not act on.
    ///
    /// The recovery still proceeds — a later checkpoint can rebootstrap this follower from
    /// anywhere — but it proceeds with no target, so whatever it lands on is reported as an
    /// explicit reset rather than a continuous recovery.
    fn refuse_in_scan(
        &mut self,
        sequence: u64,
        frame: &'static str,
        detail: &str,
    ) -> eyre::Result<()> {
        self.scan_refusals += 1;
        warn!(
            target: "ps_follow",
            sequence,
            frame,
            detail,
            "A frame written during the outage did not verify; the recovery keeps no target and \
             cannot be continuous"
        );
        self.sink.scan_refused(sequence, frame, detail)
    }

    /// One recovery pass: a fresh checkpoint restarts the grammar at its own sequence; an `End`
    /// proves no recovery is coming.
    ///
    /// Both are looked for on every pass and the lower sequence wins, because the sequence space
    /// is the stream's word order: a checkpoint written *after* an `End` is a trailing frame on a
    /// closed stream, not a recovery, and preferring it would let anything appended to a dead
    /// spool restart verdicts.
    fn scan_for_recovery(&mut self) -> eyre::Result<Step> {
        let Phase::NeedsSnapshot {
            manifest,
            reason,
            scan_from,
            target_ancestor,
            announced_tip,
            epoch,
        } = std::mem::replace(&mut self.phase, Phase::AwaitingManifest)
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
            // The last word between the request and the checkpoint is the operative one: a second
            // reorg during the outage moves the block a snapshot has to be authenticated at, and
            // a reset withdraws the request. Without this, a stale target would let a checkpoint
            // that answers a superseded question be reported as continuous recovery.
            let superseded =
                self.supersede_target(target_ancestor, announced_tip, &manifest, scan_from, found)?;
            let target = superseded.target;
            let tip = superseded.tip;
            // A new epoch's frames are read under the new epoch's identity. It was checked to be
            // this stream's successor before it got here, so adopting it is not a change of
            // subject — it is the same producer saying where its restart begins.
            let manifest = superseded.adopt.unwrap_or(manifest);
            // The commits between the last announcement (or the scan's own position) and the
            // checkpoint are replayable from the spool when the checkpoint lands exactly on the
            // target — under write-through publication they are the winning branch itself. Their
            // counting is deferred to the restore for exactly that candidate shape: rewound
            // commits are verified, not skipped. Every other shape counts them here, as it
            // always did.
            let window_from = superseded.at.map_or(self.tail.next_sequence(), |at| at + 1);
            let candidate = target.is_some() && skipped > 0;
            let rewind = if candidate && found.saturating_sub(window_from) <= MAX_REWIND_FRAMES {
                Some(RewindWindow { from: window_from, until: found })
            } else {
                if candidate {
                    self.rewind_windows_refused += 1;
                    warn!(
                        target: "ps_follow",
                        frames = found.saturating_sub(window_from),
                        bound = MAX_REWIND_FRAMES,
                        "The rewind window exceeds the bound; the restore degrades to an \
                         explicit reset instead of replaying it"
                    );
                }
                None
            };
            if rewind.is_none() {
                self.commits_skipped_in_recovery += skipped;
            }
            // Classified where the checkpoint itself is in hand, which is `restore_pair`; what
            // the scan knows and it does not is what was asked for and what went by in between.
            self.pending_recovery = Some(PendingRecovery { target, tip, skipped, rewind });
            info!(
                target: "ps_follow",
                checkpoint_sequence = found,
                skipped_commits = skipped,
                rewindable = rewind.is_some(),
                epoch,
                "A recovery checkpoint appeared; skipped commits are recorded, never verified, \
                 unless the checkpoint lands on the target and the window replays"
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
            // The branch a reorg promised never arrived and nothing valid took its place: the
            // stream closed while the recovery that would have delivered it was still owed.
            if announced_tip.is_some() {
                self.winning_branches_incomplete += 1;
                warn!(
                    target: "ps_follow",
                    "The stream ended while a recovery still owed the winning branch's tip"
                );
            }
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

        self.phase = Phase::NeedsSnapshot {
            manifest,
            reason,
            scan_from,
            target_ancestor,
            announced_tip,
            epoch,
        };
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

    /// Stashes the frame's read costs and availability proxy for the verdict line it may become.
    ///
    /// Queue wait is the read attempt's *start* minus the file's mtime, both on this host's
    /// clock — the wait alone, with the read and decode costs (reported as their own fields)
    /// excluded rather than double-counted into it. The mtime lands on the producer's tmp write
    /// — a rename before visibility — so this is a labelled proxy; a reading that goes backwards
    /// is counted as an anomaly rather than clamped into a zero that would read as an excellent
    /// latency. Frames consumed before the tail was reached are backlog: their mtime distance is
    /// the backlog's age, not a wait, so the field stays null and nothing is counted.
    fn note_frame_costs(&mut self, frame: &SpooledFrame) {
        self.last_frame_costs = Some(FrameCosts::of(frame));
        self.last_frame_available = frame.modified;
        self.last_frame_live = self.reached_tail;
        self.last_queue_wait_us = match (self.reached_tail, frame.modified) {
            (true, Some(available)) => match frame.read_at.duration_since(available) {
                Ok(waited) => Some(waited.as_micros() as u64),
                Err(_) => {
                    self.latency_anomalies += 1;
                    None
                }
            },
            _ => None,
        };
    }

    /// The mtime-derived latency fields for one verdict, or nulls where they would mislead.
    ///
    /// A catch-up verdict re-derives a frame written long ago, and a backlog frame (read before
    /// this run ever saw the live tail) predates the run itself: for both, the mtime distance is
    /// history, not latency. Live verdicts get queue wait (captured at the read attempt) and
    /// available-to-decision latency, computed here — after the verdict is decided, before its
    /// record and ack are written. Publication costs are measured separately by the sink; a
    /// record cannot carry the cost of writing itself.
    fn latency_fields(
        &mut self,
        catch_up: bool,
    ) -> (Option<u64>, Option<u64>, Option<&'static str>) {
        if catch_up || !self.last_frame_live {
            return (None, None, None)
        }
        let Some(available) = self.last_frame_available else { return (None, None, None) };
        let latency = match SystemTime::now().duration_since(available) {
            Ok(elapsed) => Some(elapsed.as_micros() as u64),
            Err(_) => {
                self.latency_anomalies += 1;
                None
            }
        };
        (self.last_queue_wait_us, latency, Some("mtime"))
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
        // A run that stops mid-rewind counts the reset here; the ack's recovery transaction is
        // deliberately *not* cleared with it — the file already on disk is what lets the resumed
        // run replay the whole window again.
        if let Some(rewind) = self.rewind_active.take() {
            self.restores_reset += 1;
            self.rewind_replayed_commits += rewind.replayed;
        }
        let waiting = matches!(self.phase, Phase::NeedsSnapshot { .. });
        if let Err(err) = self.sink.ack_state(&outcome, waiting, self.last_verified, &self.tail) {
            warn!(target: "ps_follow", error = %err, "Could not write the final ack");
        }
        // Read off the final phase rather than instrumented at every exit, so End, fault, and
        // idle-timeout all count it the same way: a reorg announced a recovery checkpoint that
        // never arrived. Legal under `never`; a gate failure under `always`.
        if matches!(&self.phase, Phase::Streaming { announced: Some(_), .. }) {
            self.replay.recovery_checkpoints_pending_at_end += 1;
        }
        let verdict_write_us = std::mem::take(&mut self.sink.verdict_write_us);
        let ack_write_us = std::mem::take(&mut self.sink.ack_write_us);
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
            rewind_replayed_commits: self.rewind_replayed_commits,
            rewind_windows_refused: self.rewind_windows_refused,
            winning_branches_completed: self.winning_branches_completed,
            winning_branches_incomplete: self.winning_branches_incomplete,
            winning_branches_superseded: self.winning_branches_superseded,
            scan_refusals: self.scan_refusals,
            catch_up_blocks: self.catch_up_blocks,
            resumed_from: self.resumed_from,
            unverified_intervals: self.unverified_intervals,
            latency_anomalies: self.latency_anomalies,
            verdict_write_us,
            ack_write_us,
        }
    }
}

/// One published verdict, and what it was measured under.
struct Published<'a> {
    verdict: &'a str,
    block: BlockRef,
    sequence: u64,
    provenance: PayloadProvenance,
    disagreements: usize,
    /// Re-derived on the way back to a watermark rather than verified for the first time.
    catch_up: bool,
    /// Verified for the first time out of the spool during a recovery rewind. Counted in
    /// `blocks_verified` and the delivered population; live-latency fields are null.
    recovery_replay: bool,
    /// This attempt's timing entry, matched by sequence.
    timing: Option<&'a BlockTiming>,
    /// Whether the frame was consumed live — read after this run had observed the spool's tail.
    /// False marks backlog: the frame predates the run, so its mtime distance is not a wait.
    tail_live: bool,
    /// Read-attempt start minus the frame's mtime. `None` on catch-up, on backlog, and when the
    /// clock misbehaved.
    queue_wait_us: Option<u64>,
    /// Decision-ready minus the frame's mtime: delivery and validation included, the writing of
    /// this very record and its ack excluded — a record cannot carry the cost of writing itself.
    /// Publication costs are the sink's separately-reported distributions. Same nulls.
    decision_latency_us: Option<u64>,
    /// What "available" was measured from — `"mtime"` on the file spool. Named so a reader never
    /// pools distributions taken from different clocks.
    available_at_source: Option<&'static str>,
    /// The producer's recorded durability watermark. Telemetry only: the consumer persists
    /// nothing to compare it against, it regresses legitimately on a pure revert, and a bare
    /// number cannot name its branch — so it is surfaced, never judged.
    producer_durability_watermark: Option<u64>,
}

/// The timing entry `replay_commit` pushed for this attempt.
///
/// `replay_commit` finishes exactly one entry per call, so the last one is this attempt's; the
/// sequence filter is a guard against that contract drifting, not a search.
fn attempt_timing(replay: &ReplayReport, sequence: u64) -> Option<&BlockTiming> {
    replay.blocks.last().filter(|timing| timing.sequence == sequence)
}

/// The follower's outputs: a JSONL verdict stream and an atomically rewritten ack file.
struct VerdictSink {
    verdicts: Option<std::fs::File>,
    ack: Option<PathBuf>,
    ack_fsync: bool,
    label: String,
    last_state: &'static str,
    /// Highest sequence any ack has claimed, so a catch-up cannot walk the watermark backwards.
    ack_high_water: Option<u64>,
    /// Set while replaying frames a previous run already acknowledged.
    suppress_ack: bool,
    /// Fields the ack carries beside the watermark, kept here because they change on transitions
    /// rather than per verdict: which epoch, why it stopped, and where its pair came from.
    ack_epoch: u64,
    ack_reason: Option<&'static str>,
    ack_restored_from: Option<u64>,
    ack_target: Option<BlockRef>,
    /// The rewound window behind the current restore point, carried on every ack for that
    /// restore point's life. Its presence is what a resume acts on; no mid-window progress is
    /// ever written beside it.
    ack_rewind: Option<AckRewind>,
    /// Per-verdict cost of serializing and writing the verdict line, in file order.
    ///
    /// The publication half `decision_latency_us` cannot carry: the line is already written when
    /// the cost is known. Reported as a distribution in the run summary instead.
    verdict_write_us: Vec<u64>,
    /// Per-verdict cost of writing the ack, sampled only when one was actually written.
    ack_write_us: Vec<u64>,
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
            ack_fsync: options.ack_fsync,
            label: options.label.clone(),
            last_state: "starting",
            ack_high_water: None,
            suppress_ack: false,
            ack_epoch: 0,
            ack_reason: None,
            ack_restored_from: None,
            ack_target: None,
            ack_rewind: None,
            verdict_write_us: Vec::new(),
            ack_write_us: Vec::new(),
        })
    }

    /// Replaces the ack's recovery window. Set at a rewind install, cleared at the next
    /// windowless install — never in between, because the window is the restore point's
    /// reconstruction recipe and outlives the replay that first ran it.
    fn set_rewind(&mut self, rewind: Option<AckRewind>) {
        self.ack_rewind = rewind;
    }

    /// One verdict line per block, carrying the standalone timing boundaries: the validation
    /// primary and its disjoint phases from the attempt's timing entry, delivery as transport cost
    /// beside them, and the mtime-proxied latency fields with their source named.
    ///
    /// The write of the line itself and of the ack after it are timed into the sink's own
    /// publication-cost samples — they cannot land in this record, which is already written when
    /// they finish, and pretending `decision_latency_us` covered them would misname its endpoint.
    fn verdict(&mut self, published: Published<'_>) -> eyre::Result<()> {
        let Published {
            verdict,
            block,
            sequence,
            provenance,
            disagreements,
            catch_up,
            recovery_replay,
            timing,
            tail_live,
            queue_wait_us,
            decision_latency_us,
            available_at_source,
            producer_durability_watermark,
        } = published;
        self.last_state = "streaming";
        let write_started = Instant::now();
        self.write(serde_json::json!({
            "schema_version": 2,
            "benchmark": "standalone_follow_v1",
            "kind": "verdict",
            "label": self.label,
            "block": block.number,
            "block_hash": format!("{:?}", block.hash),
            "sequence": sequence,
            "verdict": verdict,
            "payload_provenance": provenance.as_str(),
            "disagreements": disagreements,
            "admission_us": timing.and_then(|timing| timing.admission_us),
            "transition_us": timing.and_then(|timing| timing.transition_us),
            "standalone_validation_us": timing.map(|timing| timing.standalone_validation_us),
            "delivery_us": timing.and_then(|timing| timing.delivery_us),
            "queue_wait_us": queue_wait_us,
            "decision_latency_us": decision_latency_us,
            "available_at_source": available_at_source,
            "producer_durability_watermark": producer_durability_watermark,
            "oracle_compare_us": timing.and_then(|timing| timing.oracle_compare_us),
            "mutation_check_us": timing.and_then(|timing| timing.mutation_check_us),
            "unattributed_validation_us": timing.map(|timing| timing.unattributed_validation_us),
            "phases": timing.map(|timing| timing.phases),
            "derived": timing.map(|timing| timing.derived),
            "details": timing.and_then(|timing| timing.details.as_deref()),
            // A verdict this run re-derived on its way back to a watermark a previous run left.
            // Labelled rather than withheld: the line is real evidence that the two runs agreed
            // on the block, and a reader aggregating live throughput has to be able to skip it.
            "catch_up": catch_up,
            // A first-time verification replayed out of the spool during a recovery rewind —
            // part of `blocks_verified` and the delivered population, unlike catch-up, with the
            // live-latency fields null like it.
            "recovery_replay": recovery_replay,
            // Read after this run had already observed the spool's tail. False is backlog: a
            // frame that predates the run, whose mtime-derived fields are null by construction.
            "tail_live": tail_live,
            "observed_at_ms": now_ms(),
        }))?;
        self.verdict_write_us.push(write_started.elapsed().as_micros() as u64);
        // Sampled only when an ack will actually be written; timing a no-op would fill the
        // distribution with zeros that read as free publication.
        if self.ack.is_some() && !self.suppress_ack {
            let ack_started = Instant::now();
            self.write_ack(sequence, Some(block), "streaming")?;
            self.ack_write_us.push(ack_started.elapsed().as_micros() as u64);
            Ok(())
        } else {
            self.write_ack(sequence, Some(block), "streaming")
        }
    }

    /// One line per state transition, so the record shows *when* verdicts stopped.
    /// The `needs_snapshot` transition, with what a recovery has to produce to end it.
    ///
    /// `target_ancestor` is the field an operator or a snapshot service reads: it names the exact
    /// block a bounded snapshot has to be authenticated at, which is what the standalone recovery
    /// protocol asks for and what a bare reason string could never say.
    fn needs_snapshot(
        &mut self,
        reason: &'static str,
        detail: &str,
        last_verified: Option<BlockRef>,
        target_ancestor: Option<BlockRef>,
        epoch: u64,
        last_sequence: u64,
    ) -> eyre::Result<()> {
        self.last_state = "needs_snapshot";
        self.ack_epoch = epoch;
        self.ack_reason = Some(reason);
        self.ack_target = target_ancestor;
        // Written here and not only per verdict: a follower killed while waiting for a checkpoint
        // would otherwise leave an ack saying "streaming", and a restart would replay towards a
        // block the previous run had already refused to stand on.
        self.write_ack(last_sequence, last_verified, "needs_snapshot")?;
        self.write(serde_json::json!({
            "schema_version": 2,
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
        checkpoint_sequence: u64,
        epoch: u64,
        rewind: bool,
    ) -> eyre::Result<()> {
        self.last_state = "streaming";
        // The one field version 1 could not carry, and the reason a restart had to guess. With it
        // a resume comes back to *this* checkpoint, not to whichever one looked close enough.
        self.ack_restored_from = Some(checkpoint_sequence);
        self.ack_epoch = epoch;
        self.ack_reason = None;
        self.ack_target = None;
        self.write_ack(checkpoint_sequence, last_verified, "restored")?;
        let unverified = match class {
            Some(RecoveryClass::Reset { unverified }) => unverified,
            _ => None,
        };
        self.write(serde_json::json!({
            "schema_version": 2,
            "benchmark": "standalone_follow_v1",
            "kind": "state",
            "label": self.label,
            "state": "streaming",
            "reason": "restored",
            "detail": format!("restored at block {}", block.number),
            // Null at a rewind install on purpose: continuous is a claim about the window's
            // commits, and they have not replayed yet. The completion log line carries the
            // eventual answer; the counters carry it into the report.
            "classification": class.map(RecoveryClass::as_str),
            "rewind": rewind,
            "unverified_from": unverified.map(|(from, _)| from),
            "unverified_to": unverified.map(|(_, to)| to),
            "last_verified": last_verified.map(|block| block.number),
            "observed_at_ms": now_ms(),
        }))
    }

    /// A recovery checkpoint compared against the pair this follower already holds, and skipped.
    fn skimmed(&mut self, block: BlockRef, sequence: u64, catch_up: bool) -> eyre::Result<()> {
        self.write(serde_json::json!({
            "schema_version": 2,
            "benchmark": "standalone_follow_v1",
            "kind": "skimmed",
            "label": self.label,
            "block": block.number,
            "block_hash": format!("{:?}", block.hash),
            "sequence": sequence,
            // A resumed run re-encounters and re-records the skims of its catch-up range; the
            // label is what keeps a combined-JSONL reader from counting them twice.
            "catch_up": catch_up,
            // The unified name; `timestamp_ms` stays one release so nothing parsing v1 breaks.
            "observed_at_ms": now_ms(),
            "timestamp_ms": now_ms(),
        }))
    }

    /// A frame the recovery scan read and refused to act on.
    ///
    /// Recorded rather than merely logged because it is the reason a recovery that lands on the
    /// block a reorg named is still not reported as continuous: the frame that named it did not
    /// verify, so its ancestor is hearsay.
    fn scan_refused(
        &mut self,
        sequence: u64,
        frame: &'static str,
        detail: &str,
    ) -> eyre::Result<()> {
        self.write(serde_json::json!({
            "schema_version": 2,
            "benchmark": "standalone_follow_v1",
            "kind": "scan_refused",
            "label": self.label,
            "sequence": sequence,
            "frame": frame,
            "detail": detail,
            "observed_at_ms": now_ms(),
            "timestamp_ms": now_ms(),
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
            "schema_version": 2,
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
            "schema_version": 2,
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
        waiting: bool,
        last_verified: Option<BlockRef>,
        tail: &SpoolTail,
    ) -> eyre::Result<()> {
        // What the follower was doing outranks how the process stopped. A run that ended while
        // waiting for a snapshot has to say so, or a restart reading "ended" would replay towards
        // a block the previous run had already refused to stand on.
        let state = if waiting {
            "needs_snapshot"
        } else {
            match outcome {
                FollowOutcome::Ended { .. } => "ended",
                FollowOutcome::Faulted { .. } => "faulted",
                FollowOutcome::MaxBlocks => "max_blocks",
                FollowOutcome::IdleTimeout { .. } => "idle_timeout",
            }
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
        &mut self,
        last_sequence: u64,
        block: Option<BlockRef>,
        state: &str,
    ) -> eyre::Result<()> {
        let Some(path) = &self.ack else { return Ok(()) };
        if self.suppress_ack {
            return Ok(())
        }
        // Monotonic in the sequence space, which is the stream's word order. A restart replays
        // frames it has already acknowledged, and letting those overwrite the watermark would
        // move it backwards — the next restart would then start further back again. The block
        // number is not the key: a reorg legitimately repeats a height on a different branch.
        if self.ack_high_water.is_some_and(|seen| last_sequence < seen) {
            return Ok(())
        }
        self.ack_high_water = Some(last_sequence);
        let record = serde_json::json!({
            // Version 1 had five fields and no way to say where the pair came from, so a restart
            // had to guess a checkpoint. Version 2 removed the guessing; version 3 adds the
            // all-or-nothing recovery window, present for as long as the current restore point
            // is a rewound one — it is the recipe that rebuilds that pair, not a dirty flag.
            "ack_version": 3,
            "label": self.label,
            "last_sequence": last_sequence,
            "block": block.map(|block| block.number),
            "block_hash": block.map(|block| format!("{:?}", block.hash)),
            "state": state,
            "epoch": self.ack_epoch,
            "reason": self.ack_reason,
            "restored_from_sequence": self.ack_restored_from,
            "target_ancestor": self.ack_target.map(|block| block.number),
            "target_ancestor_hash": self.ack_target.map(|block| format!("{:?}", block.hash)),
            "recovery": self.ack_rewind.map(|rewind| serde_json::json!({
                "checkpoint_sequence": rewind.checkpoint_sequence,
                "chunks_end": rewind.chunks_end,
                "replay_from": rewind.replay_from,
                "replay_until": rewind.replay_until,
            })),
            "observed_at_ms": now_ms(),
        });
        let temporary = path.with_extension("tmp");
        if self.ack_fsync {
            // The crash-durable write recipe, on the one file a restart trusts: write a tmp
            // file, fsync it, rename over the target, then fsync the parent directory. Without
            // the last two steps the ack survives a process restart but not a power loss, and
            // only the fsync profile claims the stronger property.
            use std::io::Write as _;
            let mut file = std::fs::File::create(&temporary)?;
            file.write_all(record.to_string().as_bytes())?;
            file.sync_all()?;
            std::fs::rename(&temporary, path)?;
            std::fs::File::open(durable_parent_of(path))?.sync_all()?;
        } else {
            std::fs::write(&temporary, record.to_string())?;
            std::fs::rename(&temporary, path)?;
        }
        Ok(())
    }
}

/// The directory whose fsync makes a rename of `path` durable.
///
/// A bare relative path like `ack.json` has an *empty* parent, and skipping the directory sync on
/// it would quietly drop the one step that makes the rename itself durable — so the empty parent
/// resolves to `"."`, the working directory the rename actually landed in.
fn durable_parent_of(path: &Path) -> &Path {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
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

#[cfg(test)]
mod tests {
    use super::durable_parent_of;
    use std::path::Path;

    /// The defect this pins down: a bare relative ack path has an empty parent, and an empty
    /// parent silently skipped the directory fsync that makes the rename durable.
    #[test]
    fn a_bare_relative_ack_path_still_names_a_directory_to_sync() {
        assert_eq!(durable_parent_of(Path::new("ack.json")), Path::new("."));
        assert_eq!(durable_parent_of(Path::new("out/ack.json")), Path::new("out"));
        assert_eq!(durable_parent_of(Path::new("/tmp/ack.json")), Path::new("/tmp"));
    }
}
