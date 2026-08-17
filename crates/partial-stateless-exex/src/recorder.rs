//! Writing the live producer's events to an inspectable file spool.
//!
//! The corpus this writes is two things at once, and both constrain it. It is the *equivalence*
//! evidence — every commit carries the in-process producer's own outcome, and a standalone replay
//! is checked against it field by field, which is the only check that would catch an extraction
//! that is internally deterministic and uniformly wrong. And it is the *experiment* corpus, so two
//! builds can consume byte-identical input and the 5.7% non-identical-workload floor comes out of
//! that comparison.
//!
//! Three rules follow from that, and each of them rules out something simpler.
//!
//! **Nothing lands on disk until the checkpoint does.** A replay driver has no state until it
//! restores one, so commits recorded while the pair was still warming could never be replayed.
//! While the snapshot export runs off-task, the frames for H + 1 onward wait in a bounded buffer
//! and are flushed behind the checkpoint, so the on-disk stream still starts at exactly the block
//! after the snapshot and stays contiguous. The buffer is dropped *whole* on overflow — a trimmed
//! buffer would be precisely the silent gap the sequence numbers exist to make impossible.
//!
//! **A write failure ends the stream rather than skipping a frame.** A corpus with a hole in it
//! reads as a corpus, and the replay driver would report agreement on a chain nobody ran. So the
//! first failure disables the recorder, and the missing `End` frame is what tells a reader the
//! stream was cut rather than finished. Reaching a configured spool bound is different: the
//! stream is complete up to the bound, so it closes with `End(SpoolLimit)` instead of a cut.
//!
//! **A non-empty directory is refused** unless the operator asks for a resume. Two runs sharing a
//! spool would otherwise interleave two epochs under one sequence space, and the resulting corpus
//! would be undetectably wrong rather than obviously wrong. `PS_STREAM_RESUME=1` is the explicit
//! ask, and it is not a shortcut: the whole spool is read and checked first, because appending to
//! a corpus this producer has not verified would put its own frames behind someone else's.
//!
//! What this module is not: a live delivery path. It writes files; `partial-stateless-replay`'s
//! follow mode is what reads them without sharing the datadir.

use alloy_primitives::B256;
use alloy_rlp::Encodable;
use partial_stateless::readiness::TrustedCheckpoint;
use partial_stateless_stream::{
    encode_event_body, encode_frame_bytes, BlockRef, Checkpoint, CommitFrame, CommitInput,
    CommitOracle, End, EndKind, FrameKind, FrameLimits, Manifest, Reorg, Reset, ResetReason,
    SnapshotChunk, StreamEvent, DEFAULT_SNAPSHOT_CHUNK_BYTES, FRAME_HEADER_BYTES,
};
use reth_primitives_traits::SealedHeader;
use std::{
    collections::VecDeque,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};
use tracing::{error, info, warn};

/// Directory the spool is written to. Unset means no recording.
const STREAM_DIR_VAR: &str = "PS_STREAM_DIR";

/// Set to `1` for the power-loss durability profile: frames are fsynced before their rename and
/// the spool directory after it. Unset or `0` is the default profile — tmp+rename, durable
/// across a process restart only.
const FSYNC_VAR: &str = "PS_STREAM_FSYNC";

/// Set to `1` to continue an existing spool as a new epoch instead of refusing a non-empty one.
const RESUME_VAR: &str = "PS_STREAM_RESUME";

/// Overrides [`DEFAULT_SNAPSHOT_CHUNK_BYTES`].
const CHUNK_BYTES_VAR: &str = "PS_STREAM_CHUNK_BYTES";

/// Stamps the producer identity into the manifest, for a harness that knows its own git sha.
const PRODUCER_VAR: &str = "PS_STREAM_PRODUCER";

/// Ceiling on bytes buffered while the snapshot export runs. Exceeding it drops the buffer whole.
const BUFFER_MAX_BYTES_VAR: &str = "PS_STREAM_BUFFER_MAX_BYTES";

/// Ceiling on frames buffered while the snapshot export runs.
const BUFFER_MAX_FRAMES_VAR: &str = "PS_STREAM_BUFFER_MAX_FRAMES";

/// Ceiling on total spool bytes. Reaching it closes the stream with `End(SpoolLimit)`.
const MAX_SPOOL_BYTES_VAR: &str = "PS_STREAM_MAX_SPOOL_BYTES";

/// Ceiling on total spool frames — the inode bound beside the byte bound.
const MAX_SPOOL_FRAMES_VAR: &str = "PS_STREAM_MAX_SPOOL_FRAMES";

/// Default export buffer byte bound.
///
/// Measured: the recorded mainnet capture's commits averaged 3.32 MiB and a 156 s export spans
/// ~13 blocks, so
/// the buffer's ordinary load is ~45 MiB. 256 MiB is enough headroom for the widest blocks without
/// being a number that competes with the snapshot copies for the process's memory.
const DEFAULT_BUFFER_MAX_BYTES: usize = 256 * 1024 * 1024;

/// Default export buffer frame bound.
const DEFAULT_BUFFER_MAX_FRAMES: usize = 128;

/// Default spool byte bound: a ~6,000-block adoption run at the measured 3.32 MiB per commit is
/// ~20 GiB plus one snapshot, so 64 GiB bounds the disk without touching any run this workstream
/// actually performs.
const DEFAULT_MAX_SPOOL_BYTES: u64 = 64 * 1024 * 1024 * 1024;

/// Default spool frame bound.
const DEFAULT_MAX_SPOOL_FRAMES: u64 = 100_000;

/// Where the spool's frames go while the snapshot export runs.
///
/// The stream must start at exactly H + 1 and stay contiguous, and the checkpoint for H is not
/// writable until the export finishes — so the frames between the two wait in memory. Bounded,
/// and dropped *whole* on overflow: a trimmed buffer would be precisely the silent gap the
/// sequence numbers exist to make impossible.
#[derive(Debug)]
enum SpoolPhase {
    /// No export has chosen an H. Frames are dropped, because nothing could ever replay them.
    Idle,
    /// An export is in flight; frames wait here as pre-encoded bodies, framed at flush time.
    Buffering {
        /// Bodies in arrival order, with the kind each will be framed under.
        frames: VecDeque<(FrameKind, Vec<u8>)>,
        /// Total buffered body bytes.
        bytes: usize,
    },
    /// The checkpoint and its chunks are on disk; frames write through.
    Streaming,
}

/// Writes v1 frames into a spool directory, one atomic file per frame.
///
/// One file per frame rather than one appended segment, because an append is not atomic and a
/// reader that caught a partial one would have to distinguish "still being written" from "written
/// wrong" — a distinction the frame format can make but the filesystem cannot help with. Files are
/// named by sequence so a reader's ordering is the producer's ordering and not the directory's.
#[derive(Debug)]
pub struct StreamRecorder {
    dir: PathBuf,
    chunk_bytes: usize,
    producer: String,
    sequence: u64,
    /// Where frames currently go: dropped, buffered behind an in-flight export, or written.
    phase: SpoolPhase,
    /// Set when the export buffer overflowed; read (and cleared) by the export owner, which is
    /// the one that decides whether the attempt is retried.
    buffer_overflowed: bool,
    /// Set by the first write failure, and never cleared.
    poisoned: bool,
    /// Set once an `End` frame is on disk. A closed stream takes no further frames, and a second
    /// `End` would make the first one a lie about where the stream stopped.
    ended: bool,
    /// Set once a checkpoint has landed, so a consumer could restore from this spool.
    ///
    /// What it decides is where an abandoned buffer returns to. Before the first checkpoint there
    /// is nothing on disk to continue, so dropping frames again is right; after it, dropping them
    /// would be the silent gap in an open stream that the sequence numbers exist to forbid.
    stream_opened: bool,
    /// Bound every written frame is checked against, the same one a reader applies.
    limits: FrameLimits,
    /// Export-buffer bounds: body bytes and frame count.
    buffer_max_bytes: usize,
    buffer_max_frames: usize,
    /// Whole-spool bounds: bytes and frames (the inode bound). Reaching either closes the stream.
    max_spool_bytes: u64,
    max_spool_frames: u64,
    frames: u64,
    bytes: u64,
    /// The newest manifest a resumed spool already held, kept until this producer writes its own.
    ///
    /// Identity cannot be checked at startup: `from_env` reads environment variables and knows
    /// nothing about which chain this node follows. So the survey's answer is carried to
    /// [`write_manifest`](Self::write_manifest), which is the first moment the two can be
    /// compared — and the comparison happens before anything is appended.
    resumed_from: Option<Manifest>,
    /// Epoch of the manifest this producer wrote; `0` until
    /// [`write_manifest`](Self::write_manifest) runs. The producer event log stamps it on every
    /// line, because frame sequences alone cannot name which manifest they continue.
    epoch: u64,
    /// Highest block whose cache state the producer has written to durable storage.
    ///
    /// Held here rather than passed in per commit because it is producer state that no coordinated
    /// pair carries: the pair knows what it has applied, not what has survived a restart. A
    /// consumer resuming a stream resumes from this and not from the readiness watermark.
    durable_block: Option<u64>,
    /// The power-loss profile: fsync each frame before its rename, and the directory after it.
    fsync: bool,
    /// Set across a checkpoint's frame batch so the directory is fsynced once behind it rather
    /// than per frame — the checkpoint writes 1 + N chunks + the buffered flush in one burst.
    defer_dir_sync: bool,
    /// Total wall time spent writing frames, every fsync included — the deferred per-batch
    /// directory sync too. Logged at `End`.
    frame_write_us: u64,
    /// The fsync share of the above, always a subset of it — the power-loss profile's price,
    /// measured not assumed.
    frame_fsync_us: u64,
    /// Directory fsyncs performed, so a test can pin the batching shape.
    dir_syncs: u64,
}

impl StreamRecorder {
    /// Builds a recorder if `PS_STREAM_DIR` names a directory that can hold a fresh corpus.
    ///
    /// Returns `Ok(None)` when recording is off. An unusable directory is an error rather than a
    /// silent skip: a run configured to record and recording nothing is the failure this returns
    /// loudly, because it would otherwise be discovered after the run.
    pub fn from_env() -> eyre::Result<Option<Self>> {
        let Some(dir) = std::env::var_os(STREAM_DIR_VAR).map(PathBuf::from) else {
            return Ok(None)
        };
        fs::create_dir_all(&dir)?;
        let limits = FrameLimits::default();
        let resume = std::env::var(RESUME_VAR).is_ok_and(|raw| raw.trim() == "1");
        let existing = survey_spool(&dir, &limits, resume)?;
        fn env_bound<T: std::str::FromStr + Copy>(name: &str, default: T) -> T {
            std::env::var(name).ok().and_then(|raw| raw.trim().parse().ok()).unwrap_or(default)
        }
        let chunk_bytes = env_bound(CHUNK_BYTES_VAR, DEFAULT_SNAPSHOT_CHUNK_BYTES).max(1);
        let buffer_max_bytes = env_bound(BUFFER_MAX_BYTES_VAR, DEFAULT_BUFFER_MAX_BYTES);
        let buffer_max_frames = env_bound(BUFFER_MAX_FRAMES_VAR, DEFAULT_BUFFER_MAX_FRAMES);
        let max_spool_bytes = env_bound(MAX_SPOOL_BYTES_VAR, DEFAULT_MAX_SPOOL_BYTES);
        let max_spool_frames = env_bound(MAX_SPOOL_FRAMES_VAR, DEFAULT_MAX_SPOOL_FRAMES);
        let producer = std::env::var(PRODUCER_VAR).unwrap_or_else(|_| {
            format!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"))
        });
        let fsync = stream_fsync_from_env()?;
        info!(
            target: "partial_stateless_stream",
            dir = %dir.display(),
            chunk_bytes,
            producer,
            buffer_max_bytes,
            buffer_max_frames,
            max_spool_bytes,
            max_spool_frames,
            fsync,
            "Event stream recording ENABLED (PS_STREAM_DIR) — commits are written from the block \
             after the snapshot checkpoint"
        );
        if let Some(survey) = &existing {
            info!(
                target: "partial_stateless_stream",
                dir = %dir.display(),
                frames = survey.frames,
                bytes = survey.bytes,
                next_sequence = survey.next_sequence,
                previous_epoch = survey.manifest.epoch,
                closed = survey.ended,
                "Resuming an existing spool as a new epoch; every frame in it was read and checked \
                 before this producer appended anything"
            );
        }
        write_run_manifest(&dir, &producer);
        Ok(Some(Self {
            dir,
            chunk_bytes,
            producer,
            sequence: existing.as_ref().map_or(0, |survey| survey.next_sequence),
            phase: SpoolPhase::Idle,
            buffer_overflowed: false,
            poisoned: false,
            ended: false,
            // Per epoch, not per directory: a resumed spool has a checkpoint in it, but not one
            // this producer's frames continue from, so its own first checkpoint still opens the
            // stream it is about to write.
            stream_opened: false,
            limits,
            buffer_max_bytes,
            buffer_max_frames,
            max_spool_bytes,
            max_spool_frames,
            frames: existing.as_ref().map_or(0, |survey| survey.frames),
            bytes: existing.as_ref().map_or(0, |survey| survey.bytes),
            resumed_from: existing.map(|survey| survey.manifest),
            epoch: 0,
            durable_block: None,
            fsync,
            defer_dir_sync: false,
            frame_write_us: 0,
            frame_fsync_us: 0,
            dir_syncs: 0,
        }))
    }

    /// The spool directory frames are written into.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Epoch of the manifest this producer wrote; `0` before
    /// [`write_manifest`](Self::write_manifest).
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// The sequence the next written frame will take. The frame written last, if any, holds
    /// `current_sequence() - 1`.
    pub const fn current_sequence(&self) -> u64 {
        self.sequence
    }

    /// Records that the producer's cache is durable through `block`.
    pub const fn note_durable(&mut self, block: u64) {
        self.durable_block = Some(block);
    }

    /// A recorder for in-crate tests, bypassing the environment gate.
    #[cfg(test)]
    pub(crate) fn for_tests(dir: &Path, buffer_max_frames: usize) -> Self {
        Self {
            dir: dir.to_path_buf(),
            chunk_bytes: 64,
            producer: "test".to_string(),
            sequence: 0,
            phase: SpoolPhase::Idle,
            buffer_overflowed: false,
            poisoned: false,
            ended: false,
            stream_opened: false,
            limits: FrameLimits::default(),
            buffer_max_bytes: DEFAULT_BUFFER_MAX_BYTES,
            buffer_max_frames,
            max_spool_bytes: DEFAULT_MAX_SPOOL_BYTES,
            max_spool_frames: DEFAULT_MAX_SPOOL_FRAMES,
            frames: 0,
            bytes: 0,
            resumed_from: None,
            epoch: 0,
            durable_block: None,
            fsync: false,
            defer_dir_sync: false,
            frame_write_us: 0,
            frame_fsync_us: 0,
            dir_syncs: 0,
        }
    }

    /// A power-loss-profile recorder for in-crate tests.
    #[cfg(test)]
    pub(crate) fn for_tests_with_fsync(dir: &Path, buffer_max_frames: usize) -> Self {
        let mut recorder = Self::for_tests(dir, buffer_max_frames);
        recorder.fsync = true;
        recorder
    }

    /// The write-cost totals, for tests that pin the batching shape.
    #[cfg(test)]
    pub(crate) const fn write_costs(&self) -> (u64, u64, u64) {
        (self.frame_write_us, self.frame_fsync_us, self.dir_syncs)
    }

    /// Whether a commit's frame material should be assembled at all.
    ///
    /// True while an export buffers H + 1 onward *and* after the checkpoint opens the stream —
    /// the two phases in which a frame emitted now lands in a replayable stream, immediately or
    /// at the flush behind the checkpoint. A caller that skipped assembly while the export ran
    /// would flush an empty buffer and record a stream with a hole at its head. False before an
    /// export chose H, after a write failure, and after the stream closed.
    pub const fn wants_commit_material(&self) -> bool {
        !self.poisoned && !self.ended && !matches!(self.phase, SpoolPhase::Idle)
    }

    /// Opens the buffering phase: an export has chosen H, and every frame from here belongs to
    /// the stream that will start behind its checkpoint.
    ///
    /// Re-entrant from `Streaming`, which is what a mid-stream re-checkpoint needs: after a reorg
    /// the producer exports at the common ancestor while the winning branch is already arriving,
    /// and those commits have to wait behind the checkpoint that makes them restorable. A second
    /// call while already buffering changes nothing — the export that opened it is still the one
    /// that will close it.
    pub fn begin_buffering(&mut self) {
        if self.poisoned || self.ended {
            return
        }
        if matches!(self.phase, SpoolPhase::Idle | SpoolPhase::Streaming) {
            self.phase = SpoolPhase::Buffering { frames: VecDeque::new(), bytes: 0 };
        }
    }

    /// Whether a checkpoint has landed, so this spool could be restored from.
    pub const fn stream_opened(&self) -> bool {
        self.stream_opened
    }

    /// Drops the buffer whole and returns frames to wherever they belonged before it; the attempt
    /// the buffer was held for failed.
    ///
    /// Where that is depends on whether a checkpoint has already landed. Before one, frames go
    /// back to being dropped: nothing on disk could replay them. After one, they go back to
    /// writing through — returning an open stream to `Idle` would silently swallow every frame
    /// from here on, which is exactly the hidden gap dropping the buffer whole exists to avoid.
    /// The dropped frames themselves are a hole either way, and a consumer reads them as the
    /// skipped commits a recovery scan counts.
    pub fn abandon_buffering(&mut self, why: &str) {
        if let SpoolPhase::Buffering { frames, bytes } = &mut self.phase {
            warn!(
                target: "partial_stateless_stream",
                dropped_frames = frames.len(),
                dropped_bytes = *bytes,
                stream_opened = self.stream_opened,
                why,
                "Export buffering abandoned; the buffered frames are dropped whole rather than \
                 trimmed"
            );
            self.phase = if self.stream_opened { SpoolPhase::Streaming } else { SpoolPhase::Idle };
        }
    }

    /// Whether the export buffer overflowed since the last check. Reading it clears it.
    ///
    /// Overflow already dropped the buffer and returned the recorder to dropping frames; what
    /// the owner of the export decides is only whether a fresh attempt is worth starting.
    pub const fn take_buffer_overflow(&mut self) -> bool {
        let overflowed = self.buffer_overflowed;
        self.buffer_overflowed = false;
        overflowed
    }

    /// Writes the stream identity. Must be the first frame.
    pub fn write_manifest(
        &mut self,
        chain_id: u64,
        genesis_hash: B256,
        cache_policy_id: B256,
        account_window: u64,
        storage_window: u64,
    ) -> eyre::Result<()> {
        let previous = self.resumed_from.take();
        let manifest = Manifest {
            chain_id,
            genesis_hash,
            cache_policy_id,
            account_window,
            storage_window,
            epoch: previous.as_ref().map_or(1, |manifest| manifest.epoch + 1),
            producer: self.producer.clone(),
            // Its own position plus one, which is 1 for a fresh spool and the continuation point
            // for a resumed one. A manifest that disagrees with where it sits is how a consumer
            // catches one lifted out of another spool.
            first_sequence: self.sequence + 1,
        };
        // The first moment the resumed spool's identity can be judged: this is where the chain
        // this node actually follows arrives. A mismatch stops the recorder before it appends,
        // because the alternative is a directory holding two chains under one sequence space —
        // exactly what refusing a non-empty directory exists to prevent.
        if let Some(previous) = previous &&
            let Err(err) = manifest.check_succeeds(&previous, self.sequence)
        {
            // Returned rather than merely poisoning the recorder. A run configured to record and
            // recording nothing is the failure this module exists to make loud, and a node that
            // kept producing blocks while its spool sat frozen would surface it hours later as a
            // corpus that stops mid-run for no stated reason.
            self.poisoned = true;
            return Err(eyre::eyre!(
                "the spool at {} is not this stream: {err}. Point PS_STREAM_DIR at an empty \
                 directory, or at the right spool",
                self.dir.display()
            ))
        }
        self.epoch = manifest.epoch;
        self.write_event(&StreamEvent::Manifest(manifest));
        Ok(())
    }

    /// Writes the checkpoint and the snapshot package it describes, and opens the commit stream.
    ///
    /// `accepted_head` is the header of the checkpoint's own block. Carrying it is what lets a
    /// restored pair admit its first child instead of waiting to apply one, and it is safe to
    /// install only because every field a consumer checks it against — hash, number, state root —
    /// is in the same frame and is what the operator's checkpoint already vouched for.
    pub fn write_checkpoint(
        &mut self,
        checkpoint: &TrustedCheckpoint,
        accepted_head: Option<&SealedHeader>,
        package: &[u8],
    ) -> Option<CheckpointPublication> {
        if self.poisoned || self.ended {
            // A poisoned or closed recorder no-ops silently here, so publication must be judged
            // from this return and never from the caller having made the call.
            return None
        }
        let mut accepted_head_rlp = Vec::new();
        if let Some(header) = accepted_head {
            if header.hash() == checkpoint.block_hash {
                header.header().encode(&mut accepted_head_rlp);
            } else {
                // Not fatal, and not silently substituted either: a restored pair simply waits a
                // block. Recording the wrong header would be worse than recording none.
                warn!(
                    target: "partial_stateless_stream",
                    checkpoint_block = checkpoint.block_number,
                    accepted = ?header.hash(),
                    expected = ?checkpoint.block_hash,
                    "The pair's accepted head is not the checkpoint's own block; the snapshot \
                     will carry no header and a restored pair will wait one block"
                );
            }
        }
        let mut frame = Checkpoint {
            block: BlockRef { number: checkpoint.block_number, hash: checkpoint.block_hash },
            state_root: checkpoint.state_root,
            cache_root: checkpoint.cache_root,
            cache_policy_id: checkpoint.cache_policy_id,
            accepted_head_rlp,
            snapshot_bytes: 0,
            snapshot_chunks: 0,
            snapshot_digest: B256::ZERO,
        };
        // Described rather than chunked: the chunks are framed one slice at a time below, so the
        // package is never copied whole a second time.
        frame.describe(package, self.chunk_bytes);
        let chunk_count = frame.snapshot_chunks as u64;
        let (kind, checkpoint_body) = match encode_event_body(&StreamEvent::Checkpoint(frame)) {
            Ok(encoded) => encoded,
            Err(err) => {
                self.poison("checkpoint", &eyre::eyre!("{err}"));
                return None
            }
        };

        // Preflight: everything the checkpoint opens must fit inside the spool bounds *before*
        // its first frame is written. A cap that fired between the checkpoint and its chunks, or
        // between the chunks and the buffered commits, would close a spool that no consumer can
        // restore from — "closed" and "unrestorable" at once.
        let header = FRAME_HEADER_BYTES as u64;
        // Exact body bytes plus a small per-chunk allowance for bincode's index and length fields.
        let chunk_frame_overhead = header + 16;
        let buffered_cost: u64 = match &self.phase {
            SpoolPhase::Buffering { frames, bytes } => *bytes as u64 + header * frames.len() as u64,
            _ => 0,
        };
        let buffered_count = match &self.phase {
            SpoolPhase::Buffering { frames, .. } => frames.len() as u64,
            _ => 0,
        };
        let projected_bytes = self
            .bytes
            .saturating_add(header + checkpoint_body.len() as u64)
            .saturating_add(package.len() as u64 + chunk_frame_overhead * chunk_count)
            .saturating_add(buffered_cost);
        let projected_frames = self.frames + 1 + chunk_count + buffered_count;
        if projected_bytes > self.max_spool_bytes || projected_frames > self.max_spool_frames {
            self.abandon_buffering(
                "the checkpoint, its chunks, and the buffered commits would \
                 not fit the spool bounds",
            );
            self.write_end(
                EndKind::SpoolLimit,
                format!(
                    "checkpoint at block {} needs ~{projected_bytes} bytes / {projected_frames} \
                     frames against bounds {} / {}",
                    checkpoint.block_number, self.max_spool_bytes, self.max_spool_frames
                ),
            );
            return None
        }

        // One directory fsync behind the whole burst — checkpoint, chunks, buffered flush —
        // rather than one per frame. The batch is durable when its last rename is.
        let burst_started = Instant::now();
        let announce_sequence = self.sequence;
        self.defer_dir_sync = self.fsync;
        self.write_body(kind, checkpoint_body);
        for (index, slice) in package.chunks(self.chunk_bytes.max(1)).enumerate() {
            if self.poisoned {
                self.finish_dir_sync_batch();
                return None
            }
            let chunk = SnapshotChunk { index: index as u32, bytes: slice.to_vec() };
            self.write_event(&StreamEvent::SnapshotChunk(chunk));
        }
        if self.poisoned {
            self.finish_dir_sync_batch();
            return None
        }
        // Flush what accumulated while the export ran — same order, fresh contiguous sequences —
        // and only then write through directly.
        let buffered = match &mut self.phase {
            SpoolPhase::Buffering { frames, .. } => std::mem::take(frames),
            _ => VecDeque::new(),
        };
        let flushed = buffered.len();
        for (kind, body) in buffered {
            self.write_body(kind, body);
        }
        self.finish_dir_sync_batch();
        if self.poisoned {
            return None
        }
        self.phase = SpoolPhase::Streaming;
        self.stream_opened = true;
        info!(
            target: "partial_stateless_stream",
            block = checkpoint.block_number,
            block_hash = ?checkpoint.block_hash,
            package_bytes = package.len(),
            chunks = chunk_count,
            flushed_commits = flushed,
            has_accepted_head = accepted_head.is_some(),
            reopened = flushed > 0,
            "Wrote the checkpoint and its snapshot; the commit stream continues behind it"
        );
        Some(CheckpointPublication {
            announce_sequence,
            chunks: chunk_count,
            flushed_commits: flushed as u64,
            announce_to_complete_us: burst_started.elapsed().as_micros() as u64,
        })
    }

    /// Records one canonical block: buffered while the export runs, written once the stream is
    /// open. Returns what actually happened to the frame, because the three outcomes are not
    /// interchangeable to a caller reporting publication — a buffered commit has no publication
    /// time yet, and a dropped one never will.
    ///
    /// The durability watermark is filled in here rather than by the caller, because it is the
    /// recorder's own bookkeeping and a caller that had to remember it would eventually forget.
    pub fn write_commit(
        &mut self,
        input: CommitInput,
        mut oracle: CommitOracle,
    ) -> CommitDisposition {
        if !self.wants_commit_material() {
            return CommitDisposition::Dropped
        }
        oracle.durability_watermark = self.durable_block;
        let block = input.block;
        let provenance = input.payload_provenance;
        let buffered = matches!(self.phase, SpoolPhase::Buffering { .. });
        let sequence_before = self.sequence;
        let buffered_before = match &self.phase {
            SpoolPhase::Buffering { frames, .. } => frames.len(),
            _ => 0,
        };
        self.emit(&StreamEvent::Commit(Box::new(CommitFrame::new(input, oracle))));
        info!(
            target: "partial_stateless_stream",
            block = block.number,
            block_hash = ?block.hash,
            sequence = self.sequence,
            provenance = provenance.as_str(),
            buffered,
            frames = self.frames,
            spool_bytes = self.bytes,
            "Recorded a commit frame"
        );
        // Judged from what moved, not from the phase the call entered under: a poisoned write
        // advances nothing, and a buffer overflow inside this very emit drops the frame whole.
        if self.sequence > sequence_before && !self.poisoned {
            CommitDisposition::Published { sequence: sequence_before }
        } else if matches!(&self.phase, SpoolPhase::Buffering { frames, .. } if frames.len() > buffered_before)
        {
            CommitDisposition::Buffered
        } else {
            CommitDisposition::Dropped
        }
    }

    /// Records an abandoned branch. The winning branch follows as ordinary commits.
    pub fn write_reorg(&mut self, reorg: Reorg) {
        if !self.wants_commit_material() {
            return
        }
        self.emit(&StreamEvent::Reorg(reorg));
    }

    /// Tells a consumer it cannot continue from what it holds.
    pub fn write_reset(&mut self, reason: ResetReason, detail: impl Into<String>) {
        if !self.wants_commit_material() {
            return
        }
        let reset = Reset { reason, detail: detail.into() };
        warn!(
            target: "partial_stateless_stream",
            reason = ?reset.reason,
            detail = %reset.detail,
            "Recorded a reset; a consumer of this stream must re-bootstrap here"
        );
        self.emit(&StreamEvent::Reset(reset));
    }

    /// Closes the stream. A corpus without this ended unexpectedly.
    ///
    /// An `End` frame means the writer ran its close path — orderly termination, never success on
    /// its own; the kind is what a reader judges. Idempotent, because both an explicit close and
    /// the drop below can reach it, and a second `End` would make the first one a lie. A recorder
    /// that wrote nothing writes no `End` either: an empty directory needs no closing, and
    /// `last_sequence` would have no frame to name.
    pub fn write_end(&mut self, kind: EndKind, reason: impl Into<String>) {
        if self.poisoned || self.ended || self.frames == 0 {
            return
        }
        let end = End { kind, reason: reason.into(), last_sequence: self.sequence - 1 };
        self.write_event(&StreamEvent::End(end));
        self.ended = true;
        info!(
            target: "partial_stateless_stream",
            kind = kind.as_str(),
            frames = self.frames,
            spool_bytes = self.bytes,
            dir = %self.dir.display(),
            fsync = self.fsync,
            frame_write_us = self.frame_write_us,
            frame_fsync_us = self.frame_fsync_us,
            dir_syncs = self.dir_syncs,
            "Closed the event stream"
        );
    }

    /// Closes a deferred directory-sync batch with the one sync the whole burst shares.
    fn finish_dir_sync_batch(&mut self) {
        if !std::mem::replace(&mut self.defer_dir_sync, false) {
            return
        }
        match fsync_dir(&self.dir) {
            Ok(cost_us) => {
                // Into both totals, the same way an inline directory sync lands in both through
                // `write_body`'s wall: `frame_fsync_us` stays a strict subset of `frame_write_us`
                // whichever path the sync took.
                self.frame_fsync_us = self.frame_fsync_us.saturating_add(cost_us);
                self.frame_write_us = self.frame_write_us.saturating_add(cost_us);
                self.dir_syncs += 1;
            }
            Err(err) => self.poison("dir_fsync", &err),
        }
    }

    /// Routes one event by phase: dropped before an export chose H, buffered while it runs,
    /// written through once the stream is open.
    fn emit(&mut self, event: &StreamEvent) {
        if self.poisoned || self.ended {
            return
        }
        match self.phase {
            SpoolPhase::Idle => {}
            SpoolPhase::Buffering { .. } => self.buffer(event),
            SpoolPhase::Streaming => self.write_event(event),
        }
    }

    /// Serializes one event into the buffer, dropping the buffer whole on overflow.
    fn buffer(&mut self, event: &StreamEvent) {
        let (kind, body) = match encode_event_body(event) {
            Ok(encoded) => encoded,
            Err(err) => {
                self.poison("encode", &eyre::eyre!("{err}"));
                return
            }
        };
        let (max_bytes, max_frames) = (self.buffer_max_bytes, self.buffer_max_frames);
        let SpoolPhase::Buffering { frames, bytes } = &mut self.phase else { return };
        let projected = bytes.saturating_add(body.len());
        if projected > max_bytes || frames.len() + 1 > max_frames {
            self.buffer_overflowed = true;
            self.abandon_buffering(
                "the export buffer bound was reached; the stream cannot start contiguously at \
                 H + 1 from this attempt",
            );
            return
        }
        *bytes = projected;
        frames.push_back((kind, body));
    }

    /// Encodes and writes one frame, poisoning the recorder if anything fails.
    fn write_event(&mut self, event: &StreamEvent) {
        if self.poisoned || self.ended {
            return
        }
        match encode_event_body(event) {
            Ok((kind, body)) => self.write_body(kind, body),
            Err(err) => self.poison("encode", &eyre::eyre!("{err}")),
        }
    }

    /// Frames one pre-encoded body with the next sequence and writes it atomically.
    fn write_body(&mut self, kind: FrameKind, body: Vec<u8>) {
        if self.poisoned || self.ended {
            return
        }
        // The spool bounds close the stream rather than poisoning it: a bounded stream is
        // complete up to its bound. The End frame itself is exempt, so reaching the bound is
        // still recorded as an orderly close rather than read back as a cut.
        if kind != FrameKind::End {
            let projected = self.bytes.saturating_add((FRAME_HEADER_BYTES + body.len()) as u64);
            if projected > self.max_spool_bytes || self.frames + 1 > self.max_spool_frames {
                self.write_end(
                    EndKind::SpoolLimit,
                    format!("spool bound reached before a {} frame", kind.as_str()),
                );
                return
            }
        }
        let sequence = self.sequence;
        let write_started = Instant::now();
        let result = encode_frame_bytes(kind, sequence, &body, &self.limits)
            .map_err(|err| eyre::eyre!("{err}"))
            .and_then(|bytes| {
                let path = self.dir.join(frame_file_name(sequence, kind));
                let mut fsync_us = write_atomically(&path, &bytes, self.fsync)?;
                // The rename alone is not durable across power loss; the directory entry is.
                // Deferred across a checkpoint's burst, where one sync behind the batch covers it.
                if self.fsync && !self.defer_dir_sync {
                    fsync_us = fsync_us.saturating_add(fsync_dir(&self.dir)?);
                }
                Ok((bytes.len(), fsync_us))
            });
        match result {
            Ok((len, fsync_us)) => {
                self.sequence += 1;
                self.frames += 1;
                self.bytes += len as u64;
                if self.fsync && !self.defer_dir_sync {
                    self.dir_syncs += 1;
                }
                self.frame_fsync_us = self.frame_fsync_us.saturating_add(fsync_us);
                self.frame_write_us =
                    self.frame_write_us.saturating_add(write_started.elapsed().as_micros() as u64);
            }
            Err(err) => self.poison(kind.as_str(), &err),
        }
    }

    /// Disables the recorder after a failure; the corpus ends without an `End` frame.
    fn poison(&mut self, kind: &'static str, err: &eyre::Report) {
        self.poisoned = true;
        error!(
            target: "partial_stateless_stream",
            sequence = self.sequence,
            kind,
            error = %err,
            frames = self.frames,
            "Failed to write a stream frame; recording stops here rather than leaving a \
             hole. The corpus ends without an End frame, which is how a reader will know"
        );
    }
}

/// What a published checkpoint burst looked like on disk.
///
/// Returned by [`StreamRecorder::write_checkpoint`] so the producer event log records
/// publication from an observable rather than from the call having been made — a poisoned or
/// closed recorder no-ops that call silently.
#[derive(Debug, Clone, Copy)]
pub struct CheckpointPublication {
    /// Sequence of the checkpoint announce frame.
    pub announce_sequence: u64,
    /// Snapshot chunk frames written behind the announce.
    pub chunks: u64,
    /// Buffered commits flushed behind the chunks; zero on a write-through re-checkpoint.
    pub flushed_commits: u64,
    /// Wall time from the announce write to the last frame of the burst.
    pub announce_to_complete_us: u64,
}

/// What actually became of one commit handed to [`StreamRecorder::write_commit`].
///
/// The distinction exists for publication reporting: "the recorder was called" is not "the frame
/// is on disk", and an event stamped from the former would date a publication that has not
/// happened — or never will.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitDisposition {
    /// Written through to the open stream, at this announce sequence.
    Published {
        /// The frame's own sequence.
        sequence: u64,
    },
    /// Held in the export buffer; it reaches disk when the checkpoint flushes, or never.
    Buffered,
    /// Dropped: no phase accepts frames here (pre-export idle, poisoned, ended, or the buffer
    /// overflowed under this very frame).
    Dropped,
}

impl Drop for StreamRecorder {
    /// Closes the stream when nothing else did.
    ///
    /// Reth's shutdown drops the ExEx future while the process is still alive — every spawned
    /// task is `select`ed against the shutdown signal — so this is what turns a SIGTERM into an
    /// `End` frame. Only an abrupt kill skips it, and an abruptly killed stream is exactly what a
    /// missing `End` frame is defined to mean. A panic unwinding through here is a producer
    /// fault, not a shutdown, and the frame says so; `write` converts I/O failure into poisoning
    /// rather than panicking, so this cannot double-panic.
    fn drop(&mut self) {
        let kind =
            if std::thread::panicking() { EndKind::ProducerFault } else { EndKind::Shutdown };
        self.write_end(kind, "recorder dropped before the stream was explicitly closed");
    }
}

/// Writes through a temporary file, so a reader never sees a partially written frame.
///
/// What an existing spool already holds, once every frame in it has been read and checked.
#[derive(Debug)]
struct SpoolSurvey {
    /// The first sequence this producer may write.
    next_sequence: u64,
    /// The newest manifest, which the resuming producer's own identity is compared against.
    manifest: Manifest,
    /// Frames already on disk, so the spool bounds count the whole directory.
    frames: u64,
    /// Bytes already on disk, for the same reason.
    bytes: u64,
    /// Whether the last epoch closed with an `End` frame rather than being cut.
    ended: bool,
}

/// Reads and checks an existing spool, or refuses it.
///
/// `Ok(None)` means the directory holds no frames, which is the ordinary fresh start. A directory
/// that does hold frames is an error unless `resume` was asked for — and when it was, every frame
/// is read: names against headers, digests against bodies, sequences against their positions, and
/// each manifest against the one before it. Nothing cheaper is honest. Reading only the highest
/// sequence would let a corrupt or foreign spool be extended, and the resulting corpus would carry
/// this producer's frames behind frames it never checked, which is precisely the shape of evidence
/// that cannot be withdrawn later.
///
/// The cost is a full read of the directory — tens of seconds for a spool of tens of gigabytes.
/// That is what the explicit opt-in buys.
fn survey_spool(
    dir: &Path,
    limits: &FrameLimits,
    resume: bool,
) -> eyre::Result<Option<SpoolSurvey>> {
    let mut paths: Vec<PathBuf> = Vec::new();
    let mut leftovers: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(dir)?.filter_map(Result::ok) {
        let path = entry.path();
        match path.extension().and_then(|extension| extension.to_str()) {
            Some("frame") => paths.push(path),
            Some("tmp") => leftovers.push(path),
            _ => {}
        }
    }
    if paths.is_empty() {
        // A directory holding only interrupted writes has no stream in it, so it is a fresh start
        // whether or not a resume was asked for. Clearing them keeps the next survey honest.
        for path in leftovers {
            let _ = fs::remove_file(path);
        }
        return Ok(None)
    }
    if !resume {
        eyre::bail!(
            "{} already holds {} frames; recording into it would interleave two epochs under one \
             sequence space. Point {STREAM_DIR_VAR} at an empty directory, or set {RESUME_VAR}=1 \
             to continue this spool as a new epoch.",
            dir.display(),
            paths.len()
        );
    }
    paths.sort();

    let mut manifest: Option<Manifest> = None;
    let mut ended = false;
    let mut bytes = 0u64;
    for (position, path) in paths.iter().enumerate() {
        let position = position as u64;
        let raw = fs::read(path)
            .map_err(|err| eyre::eyre!("cannot read frame {}: {err}", path.display()))?;
        let (header, event, rest) = partial_stateless_stream::decode_event(&raw, limits)
            .map_err(|err| eyre::eyre!("frame {} is unusable: {err}", path.display()))?;
        if !rest.is_empty() {
            eyre::bail!(
                "frame {} carries {} trailing bytes; one file holds exactly one frame",
                path.display(),
                rest.len()
            );
        }
        if header.sequence != position {
            eyre::bail!(
                "spool {} is not contiguous: sequence {} sits where {position} was expected. \
                 Appending to a spool with a hole would hide the hole behind new frames",
                dir.display(),
                header.sequence
            );
        }
        // The name is what orders the walk, so a renamed file could otherwise reorder the corpus
        // for anyone reading it in directory order rather than by header.
        let expected = dir.join(frame_file_name(header.sequence, header.kind));
        if path != &expected {
            eyre::bail!(
                "frame {} is named for something other than the frame it holds ({})",
                path.display(),
                expected.display()
            );
        }
        if ended && header.kind != FrameKind::Manifest {
            eyre::bail!(
                "spool {} continues past its End frame with a {} frame at sequence {}; only a \
                 next-epoch manifest may follow an End",
                dir.display(),
                header.kind.as_str(),
                header.sequence
            );
        }
        if position == 0 && !matches!(event, StreamEvent::Manifest(_)) {
            eyre::bail!("spool {} opens with something other than a manifest", dir.display());
        }
        match &event {
            StreamEvent::Manifest(found) => {
                match &manifest {
                    None => found.check_opens(position)?,
                    Some(previous) => found.check_succeeds(previous, position)?,
                }
                manifest = Some(found.clone());
                ended = false;
            }
            StreamEvent::End(end) => {
                if end.last_sequence.checked_add(1) != Some(header.sequence) {
                    eyre::bail!(
                        "the End frame at sequence {} names {} as the last frame",
                        header.sequence,
                        end.last_sequence
                    );
                }
                ended = true;
            }
            _ => {}
        }
        bytes = bytes.saturating_add(raw.len() as u64);
    }
    let frames = paths.len() as u64;
    let manifest = manifest.expect("a spool with frames opened with a manifest");
    // Two frames past the end: the manifest this producer is about to write, and the first frame
    // after it. A stream that cannot number its own continuation is not one to continue.
    if frames.checked_add(2).is_none() || manifest.epoch.checked_add(1).is_none() {
        eyre::bail!("spool {} has no room left in its sequence or epoch space", dir.display());
    }
    for path in leftovers {
        let _ = fs::remove_file(path);
    }
    Ok(Some(SpoolSurvey { next_sequence: frames, manifest, frames, bytes, ended }))
}

/// The one place a frame's file name is derived, so the writer and every reader agree on it.
fn frame_file_name(sequence: u64, kind: FrameKind) -> String {
    format!("{sequence:012}_{}.frame", kind.as_str())
}

/// The same tmp-and-rename the snapshot export uses. It is not crash-atomic across frames — a
/// crash can leave a complete prefix and nothing else — which is exactly the guarantee a sequence
/// numbered spool needs, because a prefix is a valid truncated stream.
///
/// With `fsync` — the power-loss profile — the file is synced *before* the rename, giving the full
/// crash-durable recipe: tmp, fsync, rename, parent-directory fsync. The directory half is the
/// caller's, because a checkpoint's burst wants one sync behind the batch, not one per frame.
/// Returns the microseconds the file sync cost; zero on the default profile.
fn write_atomically(path: &Path, bytes: &[u8], fsync: bool) -> eyre::Result<u64> {
    let temporary = path.with_extension("tmp");
    let fsync_us = if fsync {
        let mut file = fs::File::create(&temporary)?;
        std::io::Write::write_all(&mut file, bytes)?;
        let sync_started = Instant::now();
        file.sync_all()?;
        sync_started.elapsed().as_micros() as u64
    } else {
        fs::write(&temporary, bytes)?;
        0
    };
    fs::rename(&temporary, path)?;
    Ok(fsync_us)
}

/// Makes the renames durable: fsyncs the spool directory itself. Returns what it cost.
fn fsync_dir(dir: &Path) -> eyre::Result<u64> {
    let started = Instant::now();
    fs::File::open(dir)?.sync_all()?;
    Ok(started.elapsed().as_micros() as u64)
}

/// `PS_STREAM_FSYNC`, refused at startup on anything but `0` or `1`.
///
/// A durability profile this build cannot parse must not silently become the default one: the
/// run would then report a durability it is not providing.
pub(crate) fn stream_fsync_from_env() -> eyre::Result<bool> {
    parse_stream_fsync(std::env::var(FSYNC_VAR).ok().as_deref())
}

fn parse_stream_fsync(raw: Option<&str>) -> eyre::Result<bool> {
    match raw.map(str::trim) {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(eyre::eyre!("PS_STREAM_FSYNC must be `0` or `1`, not `{other}`")),
    }
}

/// Appends this producer's run provenance beside the spool it is about to write.
///
/// Beside, not inside: the spool holds nothing but frames, and `survey_spool` enforces that on
/// resume. One JSONL line per producer start, so a resumed spool accumulates the provenance of
/// every epoch that wrote to it. Best-effort by the same rule as the collector itself — a run
/// that cannot stamp its manifest is warned about, not stopped, because the stream is the
/// product and the manifest is its label.
fn write_run_manifest(dir: &Path, producer: &str) {
    use std::io::Write;
    let provenance = partial_stateless_stream::RunProvenance::collect(
        producer,
        partial_stateless_stream::BuildStamp {
            commit: option_env!("PS_BUILD_COMMIT"),
            dirty: option_env!("PS_BUILD_DIRTY"),
            cargo_lock_sha256: option_env!("PS_CARGO_LOCK_SHA256"),
        },
        Some(dir),
    );
    let record = serde_json::json!({
        "schema_version": 2,
        "benchmark": "partial_stateless_stream_producer",
        "kind": "run_manifest",
        "provenance": provenance,
    });
    // A sibling named after the spool directory, built by appending rather than by
    // `with_extension`, which would eat a dot in the directory's own name.
    let name = dir.file_name().map_or_else(
        || "spool.run-manifest.jsonl".into(),
        |name| format!("{}.run-manifest.jsonl", name.to_string_lossy()),
    );
    let path = dir.parent().unwrap_or_else(|| Path::new(".")).join(name);
    let appended = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut file| writeln!(file, "{record}"));
    match appended {
        Ok(()) => info!(
            target: "partial_stateless_stream",
            path = %path.display(),
            "Wrote the producer run manifest"
        ),
        Err(err) => warn!(
            target: "partial_stateless_stream",
            path = %path.display(),
            %err,
            "Could not write the producer run manifest; the run is unlabelled, not stopped"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use partial_stateless_stream::{decode_event, FrameLimits};

    fn recorder(dir: &Path) -> StreamRecorder {
        StreamRecorder::for_tests(dir, DEFAULT_BUFFER_MAX_FRAMES)
    }

    fn checkpoint() -> TrustedCheckpoint {
        TrustedCheckpoint {
            block_number: 25_737_234,
            block_hash: B256::with_last_byte(0x11),
            state_root: B256::with_last_byte(0x22),
            cache_root: B256::with_last_byte(0x33),
            cache_policy_id: B256::with_last_byte(0x44),
        }
    }

    fn spool_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ps-recorder-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    fn frames_in(dir: &Path) -> Vec<(u64, FrameKind)> {
        let mut paths: Vec<PathBuf> = fs::read_dir(dir)
            .expect("spool readable")
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "frame"))
            .collect();
        paths.sort();
        paths
            .iter()
            .map(|path| {
                let bytes = fs::read(path).expect("frame readable");
                let (header, _, rest) =
                    decode_event(&bytes, &FrameLimits::default()).expect("frame decodes");
                assert!(rest.is_empty(), "one frame per file");
                (header.sequence, header.kind)
            })
            .collect()
    }

    /// A commit before the checkpoint could never be replayed, because a driver has no state to
    /// replay it against. Recording it anyway would put unusable frames in a corpus whose whole
    /// purpose is to be replayable.

    /// The power-loss profile still writes frames every reader decodes — the profile changes
    /// durability, never bytes — and the directory sync count is the batching contract: one per
    /// single-frame write, one behind a checkpoint's whole burst.
    #[test]
    fn the_fsync_profile_shares_one_directory_sync_per_checkpoint_burst() {
        let dir = spool_dir("fsync-batch");
        let mut recorder = StreamRecorder::for_tests_with_fsync(&dir, DEFAULT_BUFFER_MAX_FRAMES);
        recorder
            .write_manifest(1, B256::ZERO, B256::with_last_byte(0x44), 60, 30)
            .expect("a fresh spool takes a manifest");
        let (_, _, after_manifest) = recorder.write_costs();
        assert_eq!(after_manifest, 1, "a single-frame write syncs the directory once");

        // 200 bytes at the 64-byte test chunk size: checkpoint + 4 chunks in one burst.
        recorder.write_checkpoint(&checkpoint(), None, &[7u8; 200]);
        let (_, _, after_checkpoint) = recorder.write_costs();
        assert_eq!(
            after_checkpoint,
            after_manifest + 1,
            "five frames in the burst, one directory sync behind them"
        );
        assert_eq!(
            frames_in(&dir).len(),
            6,
            "manifest, checkpoint, and four chunks are all on disk and decodable"
        );

        recorder.write_reset(ResetReason::Gap, "single frame after the burst");
        let (write_us, _, after_reset) = recorder.write_costs();
        assert_eq!(after_reset, after_checkpoint + 1, "back to one sync per frame");
        assert!(write_us > 0, "the write timer accumulated across the frames");
        let _ = fs::remove_dir_all(&dir);
    }

    /// The profile flag is refused at startup on anything it cannot parse: a run configured with
    /// a durability this build does not know must not silently run the default one.
    #[test]
    fn an_unknown_fsync_profile_is_a_startup_error() {
        assert!(!parse_stream_fsync(None).expect("unset is the default profile"));
        assert!(!parse_stream_fsync(Some("0")).expect("0 is the default profile"));
        assert!(parse_stream_fsync(Some("1")).expect("1 is the power-loss profile"));
        assert!(parse_stream_fsync(Some("yes")).is_err());
    }

    #[test]
    fn commits_are_not_recorded_until_the_checkpoint_is() {
        let dir = spool_dir("before-checkpoint");
        let mut recorder = recorder(&dir);
        recorder
            .write_manifest(1, B256::ZERO, B256::with_last_byte(0x44), 60, 30)
            .expect("a fresh spool takes a manifest");
        assert!(!recorder.wants_commit_material());

        recorder.write_reset(ResetReason::Gap, "ignored before the checkpoint");
        assert_eq!(frames_in(&dir), vec![(0, FrameKind::Manifest)]);

        recorder.write_checkpoint(&checkpoint(), None, &[7u8; 200]);
        assert!(recorder.wants_commit_material());

        // 200 bytes at a 64-byte chunk is four chunks, and they follow the checkpoint in order.
        assert_eq!(
            frames_in(&dir),
            vec![
                (0, FrameKind::Manifest),
                (1, FrameKind::Checkpoint),
                (2, FrameKind::SnapshotChunk),
                (3, FrameKind::SnapshotChunk),
                (4, FrameKind::SnapshotChunk),
                (5, FrameKind::SnapshotChunk),
            ]
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// A header that is not the checkpoint's own block is dropped rather than recorded, because a
    /// restored pair installs this header on the strength of the checkpoint vouching for it.
    #[test]
    fn a_header_that_is_not_the_checkpoints_own_block_is_not_carried() {
        let dir = spool_dir("wrong-header");
        let mut recorder = recorder(&dir);
        let mut header = alloy_consensus::Header { number: 25_737_234, ..Default::default() };
        header.parent_hash = B256::with_last_byte(0xaa);
        let sealed = SealedHeader::new_unhashed(header);

        recorder.write_checkpoint(&checkpoint(), Some(&sealed), &[1u8; 8]);

        let bytes = fs::read(dir.join("000000000000_checkpoint.frame")).expect("checkpoint");
        let (_, event, _) = decode_event(&bytes, &FrameLimits::default()).expect("decodes");
        let StreamEvent::Checkpoint(frame) = event else { panic!("checkpoint frame") };
        assert!(frame.accepted_head_rlp.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    /// The refusal exists because two runs sharing a spool interleave two sequence spaces, and the
    /// result reads as one stream rather than as an error.
    #[test]
    fn a_spool_that_already_holds_frames_is_refused() {
        let dir = spool_dir("non-empty");
        fs::write(dir.join("000000000000_manifest.frame"), b"anything").expect("write");
        // SAFETY: single-threaded test; the variable is read once here and removed immediately.
        unsafe { std::env::set_var(STREAM_DIR_VAR, &dir) };
        let result = StreamRecorder::from_env();
        unsafe { std::env::remove_var(STREAM_DIR_VAR) };

        let error = result.expect_err("a non-empty spool is refused").to_string();
        assert!(error.contains("already holds 1 frames"), "{error}");
        let _ = fs::remove_dir_all(&dir);
    }

    /// A stream that was cut has no `End` frame, which is exactly how a reader tells a truncated
    /// corpus from a short one.
    #[test]
    fn a_write_failure_ends_the_stream_rather_than_skipping_a_frame() {
        let dir = spool_dir("poisoned");
        let mut recorder = recorder(&dir);
        recorder
            .write_manifest(1, B256::ZERO, B256::with_last_byte(0x44), 60, 30)
            .expect("a fresh spool takes a manifest");
        recorder.write_checkpoint(&checkpoint(), None, &[1u8; 8]);

        // Removing the directory is the cheapest true I/O failure available here.
        fs::remove_dir_all(&dir).expect("remove spool");
        recorder.write_reset(ResetReason::Overflow, "producer queue overflowed");
        assert!(!recorder.wants_commit_material(), "the first failure stops recording");

        fs::create_dir_all(&dir).expect("recreate");
        recorder.write_end(EndKind::Shutdown, "shutdown");
        assert!(frames_in(&dir).is_empty(), "a poisoned recorder writes nothing further");
        drop(recorder);
        assert!(frames_in(&dir).is_empty(), "a poisoned recorder's drop writes nothing either");
        let _ = fs::remove_dir_all(&dir);
    }

    /// Decodes the last frame of the spool as an `End`, with the header sequence beside it.
    fn last_end(dir: &Path) -> (u64, End) {
        let (sequence, kind) = *frames_in(dir).last().expect("spool is not empty");
        assert_eq!(kind, FrameKind::End, "the stream's last frame is an End");
        let path = dir.join(format!("{sequence:012}_end.frame"));
        let bytes = fs::read(path).expect("end frame readable");
        let (_, event, _) = decode_event(&bytes, &FrameLimits::default()).expect("decodes");
        let StreamEvent::End(end) = event else { panic!("an end frame decodes as an end") };
        (sequence, end)
    }

    /// Reth's SIGTERM path drops the ExEx future rather than returning from it, so the drop is
    /// the only close path an operator stop ever takes. The frame it writes names its
    /// predecessor, which is the reader's check that it saw every frame.
    #[test]
    fn dropping_a_recorder_closes_the_stream_with_one_correctly_numbered_end() {
        let dir = spool_dir("dropped");
        let mut recorder = recorder(&dir);
        recorder
            .write_manifest(1, B256::ZERO, B256::with_last_byte(0x44), 60, 30)
            .expect("a fresh spool takes a manifest");
        recorder.write_checkpoint(&checkpoint(), None, &[7u8; 100]);
        drop(recorder);

        let (sequence, end) = last_end(&dir);
        assert_eq!(end.kind, EndKind::Shutdown);
        assert_eq!(end.last_sequence + 1, sequence, "the End frame names its predecessor");
        let ends = frames_in(&dir).iter().filter(|(_, kind)| *kind == FrameKind::End).count();
        assert_eq!(ends, 1, "exactly one End frame");
        let _ = fs::remove_dir_all(&dir);
    }

    /// An explicit close keeps its kind and its reason; the drop must not write a second `End`
    /// that contradicts it.
    #[test]
    fn an_explicit_end_survives_the_drop_unamended() {
        let dir = spool_dir("explicit-end");
        let mut recorder = recorder(&dir);
        recorder
            .write_manifest(1, B256::ZERO, B256::with_last_byte(0x44), 60, 30)
            .expect("a fresh spool takes a manifest");
        recorder.write_end(EndKind::SpoolLimit, "spool byte bound reached");
        drop(recorder);

        let (_, end) = last_end(&dir);
        assert_eq!(end.kind, EndKind::SpoolLimit);
        let ends = frames_in(&dir).iter().filter(|(_, kind)| *kind == FrameKind::End).count();
        assert_eq!(ends, 1, "the drop does not write a second End");
        let _ = fs::remove_dir_all(&dir);
    }

    /// A recorder that wrote nothing has nothing to close: an `End` in an otherwise empty spool
    /// would have no predecessor for `last_sequence` to name.
    #[test]
    fn a_recorder_that_wrote_nothing_writes_no_end_on_drop() {
        let dir = spool_dir("nothing-written");
        drop(recorder(&dir));
        assert!(frames_in(&dir).is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    /// A panic is a producer fault, not a shutdown, and the stream's close frame says which.
    #[test]
    fn a_panic_unwind_closes_the_stream_as_a_producer_fault() {
        let dir = spool_dir("panicked");
        let thread_dir = dir.clone();
        let result = std::thread::spawn(move || {
            let mut recorder = recorder(&thread_dir);
            recorder
                .write_manifest(1, B256::ZERO, B256::with_last_byte(0x44), 60, 30)
                .expect("a fresh spool takes a manifest");
            panic!("producer failed mid-run");
        })
        .join();
        assert!(result.is_err(), "the thread panicked");

        let (_, end) = last_end(&dir);
        assert_eq!(end.kind, EndKind::ProducerFault);
        let _ = fs::remove_dir_all(&dir);
    }

    /// The spec sentence, as a filesystem assertion: frames emitted while the export runs land
    /// *behind* the checkpoint and its chunks, in arrival order, with contiguous sequences.
    #[test]
    fn frames_buffered_during_the_export_flush_behind_the_checkpoint_in_order() {
        let dir = spool_dir("buffered-flush");
        let mut recorder = recorder(&dir);
        recorder
            .write_manifest(1, B256::ZERO, B256::with_last_byte(0x44), 60, 30)
            .expect("a fresh spool takes a manifest");
        recorder.begin_buffering();
        assert!(recorder.wants_commit_material(), "buffering wants commit material assembled");

        // Two frames arrive while the export is in flight; a reorg between them must keep its
        // position relative to the resets.
        recorder.write_reset(ResetReason::Gap, "first buffered");
        recorder.write_reorg(Reorg {
            common_ancestor: BlockRef { number: 1, hash: B256::with_last_byte(1) },
            abandoned: vec![],
            winning_tip: None,
        });
        recorder.write_reset(ResetReason::Gap, "second buffered");
        assert_eq!(frames_in(&dir), vec![(0, FrameKind::Manifest)], "nothing on disk yet");

        recorder.write_checkpoint(&checkpoint(), None, &[7u8; 100]);
        assert_eq!(
            frames_in(&dir),
            vec![
                (0, FrameKind::Manifest),
                (1, FrameKind::Checkpoint),
                (2, FrameKind::SnapshotChunk),
                (3, FrameKind::SnapshotChunk),
                (4, FrameKind::Reset),
                (5, FrameKind::Reorg),
                (6, FrameKind::Reset),
            ]
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// The grammar the reorg-recovery protocol depends on, as a filesystem assertion: a reorg is
    /// written through the open stream, the commits behind it wait for the checkpoint that makes
    /// them restorable, and the sequences stay contiguous across the whole shape.
    #[test]
    fn a_mid_stream_checkpoint_lands_behind_the_reorg_with_contiguous_sequences() {
        let dir = spool_dir("mid-stream-checkpoint");
        let mut recorder = recorder(&dir);
        recorder
            .write_manifest(1, B256::ZERO, B256::with_last_byte(0x44), 60, 30)
            .expect("a fresh spool takes a manifest");
        recorder.begin_buffering();
        recorder.write_checkpoint(&checkpoint(), None, &[7u8; 100]);
        assert!(recorder.stream_opened());
        recorder.write_reset(ResetReason::Gap, "a streamed frame before the reorg");

        // The reorg is written through: a consumer must learn the branch changed before it is
        // asked to skip anything.
        recorder.write_reorg(Reorg {
            common_ancestor: BlockRef { number: 25_737_234, hash: B256::with_last_byte(0x11) },
            abandoned: vec![BlockRef { number: 25_737_235, hash: B256::with_last_byte(0x12) }],
            winning_tip: None,
        });
        // Then the export re-arms, and the winning branch waits behind the new checkpoint.
        recorder.begin_buffering();
        recorder.write_reset(ResetReason::Gap, "buffered winning-branch material");
        assert_eq!(
            frames_in(&dir).len(),
            6,
            "manifest, checkpoint, two chunks, one streamed reset, the reorg"
        );

        recorder.write_checkpoint(&checkpoint(), None, &[9u8; 100]);

        assert_eq!(
            frames_in(&dir),
            vec![
                (0, FrameKind::Manifest),
                (1, FrameKind::Checkpoint),
                (2, FrameKind::SnapshotChunk),
                (3, FrameKind::SnapshotChunk),
                (4, FrameKind::Reset),
                (5, FrameKind::Reorg),
                (6, FrameKind::Checkpoint),
                (7, FrameKind::SnapshotChunk),
                (8, FrameKind::SnapshotChunk),
                (9, FrameKind::Reset),
            ]
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// Abandoning a buffer mid-stream returns to writing through, not to dropping.
    ///
    /// The distinction is the whole reason the recorder tracks whether a checkpoint has landed.
    /// Returning an open stream to `Idle` would swallow every frame from there on — a hole with
    /// no frame to mark it, which is worse than the buffered frames the abandon already drops.
    #[test]
    fn abandoning_a_mid_stream_buffer_returns_to_writing_through() {
        let dir = spool_dir("abandon-mid-stream");
        let mut recorder = recorder(&dir);
        recorder
            .write_manifest(1, B256::ZERO, B256::with_last_byte(0x44), 60, 30)
            .expect("a fresh spool takes a manifest");
        recorder.begin_buffering();
        recorder.write_checkpoint(&checkpoint(), None, &[7u8; 100]);
        recorder.begin_buffering();
        recorder.write_reset(ResetReason::Gap, "lost with the attempt");

        recorder.abandon_buffering("the export attempt was fenced");

        assert!(recorder.wants_commit_material(), "an open stream still takes frames");
        recorder.write_reset(ResetReason::Gap, "written through");
        let kinds: Vec<FrameKind> = frames_in(&dir).iter().map(|(_, kind)| *kind).collect();
        assert_eq!(
            kinds,
            vec![
                FrameKind::Manifest,
                FrameKind::Checkpoint,
                FrameKind::SnapshotChunk,
                FrameKind::SnapshotChunk,
                FrameKind::Reset,
            ],
            "the abandoned frame is gone and the next one is on disk"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// The same abandon before any checkpoint still drops back to `Idle`: nothing on disk could
    /// replay those frames, so writing them would put unusable bytes in the corpus.
    #[test]
    fn abandoning_a_buffer_before_the_stream_opens_still_drops_frames() {
        let dir = spool_dir("abandon-pre-open");
        let mut recorder = recorder(&dir);
        recorder
            .write_manifest(1, B256::ZERO, B256::with_last_byte(0x44), 60, 30)
            .expect("a fresh spool takes a manifest");
        recorder.begin_buffering();
        recorder.write_reset(ResetReason::Gap, "lost with the attempt");

        recorder.abandon_buffering("the first export attempt failed");

        assert!(!recorder.wants_commit_material());
        recorder.write_reset(ResetReason::Gap, "also dropped");
        assert_eq!(frames_in(&dir), vec![(0, FrameKind::Manifest)]);
        let _ = fs::remove_dir_all(&dir);
    }

    /// Overflow drops the buffer whole. A trimmed buffer would flush a stream whose head is
    /// missing — exactly the hidden drop the sequence numbers exist to make impossible.
    #[test]
    fn a_buffer_overflow_drops_the_buffer_whole_and_reports_once() {
        let dir = spool_dir("buffer-overflow");
        let mut recorder = recorder(&dir);
        recorder.buffer_max_frames = 1;
        recorder
            .write_manifest(1, B256::ZERO, B256::with_last_byte(0x44), 60, 30)
            .expect("a fresh spool takes a manifest");
        recorder.begin_buffering();

        recorder.write_reset(ResetReason::Gap, "fits");
        assert!(!recorder.take_buffer_overflow(), "one frame fits");
        recorder.begin_buffering(); // no-op: already buffering
        recorder.write_reset(ResetReason::Gap, "overflows");

        assert!(recorder.take_buffer_overflow(), "the second frame overflowed the bound");
        assert!(!recorder.take_buffer_overflow(), "reading the overflow clears it");
        assert!(!recorder.wants_commit_material(), "the failed attempt dropped back to Idle");
        assert_eq!(frames_in(&dir), vec![(0, FrameKind::Manifest)], "nothing was flushed");
        let _ = fs::remove_dir_all(&dir);
    }

    /// The preflight runs before the checkpoint's first frame: a spool that cannot hold the
    /// checkpoint, its chunks, and the buffered commits together gets an `End(SpoolLimit)` and
    /// no partial checkpoint — "closed but unrestorable" must not be constructible.
    #[test]
    fn a_checkpoint_that_would_not_fit_the_spool_bound_is_not_started() {
        let dir = spool_dir("preflight");
        let mut recorder = recorder(&dir);
        recorder
            .write_manifest(1, B256::ZERO, B256::with_last_byte(0x44), 60, 30)
            .expect("a fresh spool takes a manifest");
        recorder.max_spool_bytes = recorder.bytes + 64; // room for nothing more
        recorder.begin_buffering();
        recorder.write_checkpoint(&checkpoint(), None, &[7u8; 200]);

        let kinds: Vec<FrameKind> = frames_in(&dir).iter().map(|(_, kind)| *kind).collect();
        assert_eq!(kinds, vec![FrameKind::Manifest, FrameKind::End], "no partial checkpoint");
        let (_, end) = last_end(&dir);
        assert_eq!(end.kind, EndKind::SpoolLimit);
        let _ = fs::remove_dir_all(&dir);
    }

    /// The running spool bound closes the stream in order, with the End frame itself exempt so
    /// the close is recorded rather than read back as a cut.
    #[test]
    fn reaching_the_spool_bound_closes_the_stream_with_an_end_frame() {
        let dir = spool_dir("spool-bound");
        let mut recorder = recorder(&dir);
        recorder
            .write_manifest(1, B256::ZERO, B256::with_last_byte(0x44), 60, 30)
            .expect("a fresh spool takes a manifest");
        recorder.write_checkpoint(&checkpoint(), None, &[7u8; 64]);
        recorder.max_spool_frames = recorder.frames + 1; // one more payload frame fits
        recorder.write_reset(ResetReason::Gap, "fits");
        recorder.write_reset(ResetReason::Gap, "over the bound");

        let (_, end) = last_end(&dir);
        assert_eq!(end.kind, EndKind::SpoolLimit);
        assert!(!recorder.wants_commit_material(), "a closed stream takes no more frames");
        let _ = fs::remove_dir_all(&dir);
    }

    /// A recorder built the way `from_env` builds a resumed one, without the environment gate.
    fn resumed(dir: &Path, survey: SpoolSurvey) -> StreamRecorder {
        let mut recorder = StreamRecorder::for_tests(dir, DEFAULT_BUFFER_MAX_FRAMES);
        recorder.sequence = survey.next_sequence;
        recorder.frames = survey.frames;
        recorder.bytes = survey.bytes;
        recorder.resumed_from = Some(survey.manifest);
        recorder
    }

    /// Writes a complete one-epoch spool and closes it, which is what a producer leaves behind
    /// when it is stopped politely.
    fn closed_epoch(dir: &Path) {
        let mut recorder = recorder(dir);
        recorder
            .write_manifest(1, B256::ZERO, B256::with_last_byte(0x44), 60, 30)
            .expect("a fresh spool takes a manifest");
        recorder.write_checkpoint(&checkpoint(), None, &[7u8; 100]);
        recorder.write_end(EndKind::Shutdown, "stopped".to_string());
    }

    #[test]
    fn a_non_empty_spool_is_refused_unless_a_resume_was_asked_for() {
        let dir = spool_dir("resume-refused");
        closed_epoch(&dir);

        let refused = survey_spool(&dir, &FrameLimits::default(), false);

        assert!(refused.is_err(), "the default is still to refuse someone else's spool");
        assert!(
            survey_spool(&dir, &FrameLimits::default(), true).expect("checks out").is_some(),
            "and the same directory is readable once the operator asks for it"
        );
    }

    #[test]
    fn a_resumed_spool_continues_the_sequence_space_as_the_next_epoch() {
        // Sequence numbers do not restart with the epoch: one directory has one sequence space,
        // because frame files are named by it. What the epoch says is that the state broke, which
        // is the part a consumer must not read continuity into.
        let dir = spool_dir("resume-continues");
        closed_epoch(&dir);
        let survey = survey_spool(&dir, &FrameLimits::default(), true)
            .expect("checks out")
            .expect("frames exist");
        let before = frames_in(&dir);

        let mut recorder = resumed(&dir, survey);
        recorder
            .write_manifest(1, B256::ZERO, B256::with_last_byte(0x44), 60, 30)
            .expect("a fresh spool takes a manifest");

        let after = frames_in(&dir);
        assert_eq!(&after[..before.len()], &before[..], "nothing already written moved");
        assert_eq!(after[before.len()], (before.len() as u64, FrameKind::Manifest));
        let bytes = fs::read(dir.join(frame_file_name(before.len() as u64, FrameKind::Manifest)))
            .expect("readable");
        let (_, event, _) = decode_event(&bytes, &FrameLimits::default()).expect("decodes");
        let StreamEvent::Manifest(written) = event else { panic!("a manifest was written") };
        assert_eq!(written.epoch, 2);
        assert_eq!(written.first_sequence, before.len() as u64 + 1);
    }

    #[test]
    fn a_resumed_spool_is_not_appended_to_when_it_is_a_different_stream() {
        // The identity check cannot happen at startup — `from_env` reads environment variables and
        // has never heard of this node's chain. So it happens at the manifest, and it happens
        // *before* the manifest is written: a directory holding two chains under one sequence
        // space is exactly what refusing a non-empty directory exists to prevent.
        let dir = spool_dir("resume-foreign");
        closed_epoch(&dir);
        let survey = survey_spool(&dir, &FrameLimits::default(), true)
            .expect("checks out")
            .expect("frames exist");
        let before = frames_in(&dir);

        let mut recorder = resumed(&dir, survey);
        let refused = recorder.write_manifest(999, B256::ZERO, B256::with_last_byte(0x44), 60, 30);

        assert!(refused.is_err(), "and it is the caller's to fail the run on, not a silent stop");
        assert_eq!(frames_in(&dir), before, "nothing was appended");
        assert!(recorder.poisoned, "and the recorder will not write anything else either");
    }

    #[test]
    fn a_spool_with_a_hole_is_refused_rather_than_extended() {
        let dir = spool_dir("resume-hole");
        closed_epoch(&dir);
        let frames = frames_in(&dir);
        let (sequence, kind) = frames[1];
        fs::remove_file(dir.join(frame_file_name(sequence, kind))).expect("removable");

        assert!(
            survey_spool(&dir, &FrameLimits::default(), true).is_err(),
            "appending would hide the hole behind frames that are not missing"
        );
    }

    #[test]
    fn a_frame_named_for_something_else_is_refused() {
        // The names order the walk, so a rename is the one edit that can reorder a corpus for a
        // reader without breaking any single frame.
        let dir = spool_dir("resume-renamed");
        closed_epoch(&dir);
        let frames = frames_in(&dir);
        let (sequence, kind) = frames[1];
        fs::rename(
            dir.join(frame_file_name(sequence, kind)),
            dir.join(frame_file_name(sequence, FrameKind::Commit)),
        )
        .expect("renamable");

        assert!(survey_spool(&dir, &FrameLimits::default(), true).is_err());
    }

    #[test]
    fn a_corrupt_frame_is_refused_rather_than_extended() {
        let dir = spool_dir("resume-corrupt");
        closed_epoch(&dir);
        let frames = frames_in(&dir);
        let (sequence, kind) = frames[1];
        let path = dir.join(frame_file_name(sequence, kind));
        let mut bytes = fs::read(&path).expect("readable");
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        fs::write(&path, &bytes).expect("writable");

        assert!(
            survey_spool(&dir, &FrameLimits::default(), true).is_err(),
            "the digests are checked, so a body that changed under the header is caught here"
        );
    }

    #[test]
    fn only_a_next_epoch_manifest_may_follow_an_end() {
        let dir = spool_dir("resume-past-end");
        closed_epoch(&dir);
        let next = frames_in(&dir).len() as u64;
        // A commit appended to a closed stream: legal-looking bytes in an illegal place.
        let mut recorder = StreamRecorder::for_tests(&dir, DEFAULT_BUFFER_MAX_FRAMES);
        recorder.sequence = next;
        recorder.write_event(&StreamEvent::Reset(Reset {
            reason: ResetReason::Gap,
            detail: "appended to a closed stream".to_string(),
        }));

        assert!(survey_spool(&dir, &FrameLimits::default(), true).is_err());
    }

    #[test]
    fn interrupted_writes_are_cleared_rather_than_counted() {
        let dir = spool_dir("resume-tmp");
        closed_epoch(&dir);
        let leftover = dir.join("000000000009_commit.tmp");
        fs::write(&leftover, b"half a frame").expect("writable");

        let survey = survey_spool(&dir, &FrameLimits::default(), true)
            .expect("checks out")
            .expect("frames exist");

        assert!(!leftover.exists(), "a partial write is not part of the corpus");
        assert_eq!(survey.next_sequence, survey.frames, "and it is not counted as one either");
    }

    #[test]
    fn the_spool_bounds_count_what_a_resumed_spool_already_holds() {
        // Otherwise the bound is per-epoch rather than per-directory, and a producer restarted
        // often enough would pass every check while filling the disk.
        let dir = spool_dir("resume-bounds");
        closed_epoch(&dir);
        let survey = survey_spool(&dir, &FrameLimits::default(), true)
            .expect("checks out")
            .expect("frames exist");
        let on_disk: u64 = frames_in(&dir)
            .iter()
            .map(|(sequence, kind)| {
                fs::metadata(dir.join(frame_file_name(*sequence, *kind))).expect("stat").len()
            })
            .sum();

        assert_eq!(survey.bytes, on_disk);
        assert_eq!(survey.frames, frames_in(&dir).len() as u64);
    }
}
