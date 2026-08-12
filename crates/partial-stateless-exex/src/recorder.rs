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
//! **Nothing is written until the checkpoint is.** A replay driver has no state until it restores
//! one, so commits recorded while the pair was still warming could never be replayed. The stream
//! therefore starts at the block after the snapshot, which is also exactly what S3's live producer
//! does — choose H, export, then publish a contiguous stream from H + 1.
//!
//! **A write failure ends the stream rather than skipping a frame.** A corpus with a hole in it
//! reads as a corpus, and the replay driver would report agreement on a chain nobody ran. So the
//! first failure disables the recorder, and the missing `End` frame is what tells a reader the
//! stream was cut rather than finished.
//!
//! **A non-empty directory is refused.** Two runs sharing a spool would interleave two epochs
//! under one sequence space, and the resulting corpus would be undetectably wrong rather than
//! obviously wrong.
//!
//! What this module is not: a live delivery path. It writes files, and S3 is where a consumer
//! reads them without sharing the datadir.

use alloy_primitives::B256;
use alloy_rlp::Encodable;
use partial_stateless::readiness::TrustedCheckpoint;
use partial_stateless_stream::{
    encode_event, BlockRef, Checkpoint, CommitFrame, CommitInput, CommitOracle, End, FrameKind,
    Manifest, Reorg, Reset, ResetReason, StreamEvent, DEFAULT_SNAPSHOT_CHUNK_BYTES,
};
use reth_primitives_traits::SealedHeader;
use std::{
    fs,
    path::{Path, PathBuf},
};
use tracing::{error, info, warn};

/// Directory the spool is written to. Unset means no recording.
const STREAM_DIR_VAR: &str = "PS_STREAM_DIR";

/// Overrides [`DEFAULT_SNAPSHOT_CHUNK_BYTES`].
const CHUNK_BYTES_VAR: &str = "PS_STREAM_CHUNK_BYTES";

/// Stamps the producer identity into the manifest, for a harness that knows its own git sha.
const PRODUCER_VAR: &str = "PS_STREAM_PRODUCER";

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
    /// Set once the checkpoint and its chunks are on disk. Commits before this are not recorded.
    checkpointed: bool,
    /// Set by the first write failure, and never cleared.
    poisoned: bool,
    frames: u64,
    bytes: u64,
    /// Highest block whose cache state the producer has written to durable storage.
    ///
    /// Held here rather than passed in per commit because it is producer state that no coordinated
    /// pair carries: the pair knows what it has applied, not what has survived a restart. A
    /// consumer resuming a stream resumes from this and not from the readiness watermark.
    durable_block: Option<u64>,
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
        let existing = fs::read_dir(&dir)?
            .filter_map(Result::ok)
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "frame"))
            .count();
        if existing > 0 {
            eyre::bail!(
                "{} already holds {existing} frames; recording into it would interleave two \
                 epochs under one sequence space. Point {STREAM_DIR_VAR} at an empty directory.",
                dir.display()
            );
        }
        let chunk_bytes = std::env::var(CHUNK_BYTES_VAR)
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_SNAPSHOT_CHUNK_BYTES)
            .max(1);
        let producer = std::env::var(PRODUCER_VAR).unwrap_or_else(|_| {
            format!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"))
        });
        info!(
            target: "partial_stateless_stream",
            dir = %dir.display(),
            chunk_bytes,
            producer,
            "Event stream recording ENABLED (PS_STREAM_DIR) — commits are written from the block \
             after the snapshot checkpoint"
        );
        Ok(Some(Self {
            dir,
            chunk_bytes,
            producer,
            sequence: 0,
            checkpointed: false,
            poisoned: false,
            frames: 0,
            bytes: 0,
            durable_block: None,
        }))
    }

    /// Records that the producer's cache is durable through `block`.
    pub const fn note_durable(&mut self, block: u64) {
        self.durable_block = Some(block);
    }

    /// Whether a commit written now would land in a replayable stream.
    ///
    /// False before the checkpoint and after a write failure. Callers use it to skip the work of
    /// assembling a frame that would be discarded.
    pub const fn records_commits(&self) -> bool {
        self.checkpointed && !self.poisoned
    }

    /// Writes the stream identity. Must be the first frame.
    pub fn write_manifest(
        &mut self,
        chain_id: u64,
        genesis_hash: B256,
        cache_policy_id: B256,
        account_window: u64,
        storage_window: u64,
    ) {
        let manifest = Manifest {
            chain_id,
            genesis_hash,
            cache_policy_id,
            account_window,
            storage_window,
            // One directory holds one epoch, which `from_env` enforces by refusing a non-empty
            // one. A producer that resumed into an existing spool would need this to increment.
            epoch: 1,
            producer: self.producer.clone(),
            first_sequence: 1,
        };
        self.write(FrameKind::Manifest, &StreamEvent::Manifest(manifest));
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
    ) {
        if self.poisoned {
            return
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
        let chunks = frame.chunk(package, self.chunk_bytes);
        let chunk_count = chunks.len();
        self.write(FrameKind::Checkpoint, &StreamEvent::Checkpoint(frame));
        for chunk in chunks {
            self.write(FrameKind::SnapshotChunk, &StreamEvent::SnapshotChunk(chunk));
        }
        if self.poisoned {
            return
        }
        self.checkpointed = true;
        info!(
            target: "partial_stateless_stream",
            block = checkpoint.block_number,
            block_hash = ?checkpoint.block_hash,
            package_bytes = package.len(),
            chunks = chunk_count,
            has_accepted_head = accepted_head.is_some(),
            "Wrote the checkpoint and its snapshot; the commit stream starts at the next block"
        );
    }

    /// Writes one canonical block.
    ///
    /// The durability watermark is filled in here rather than by the caller, because it is the
    /// recorder's own bookkeeping and a caller that had to remember it would eventually forget.
    pub fn write_commit(&mut self, input: CommitInput, mut oracle: CommitOracle) {
        if !self.records_commits() {
            return
        }
        oracle.durability_watermark = self.durable_block;
        let block = input.block;
        let provenance = input.payload_provenance;
        self.write(
            FrameKind::Commit,
            &StreamEvent::Commit(Box::new(CommitFrame::new(input, oracle))),
        );
        info!(
            target: "partial_stateless_stream",
            block = block.number,
            block_hash = ?block.hash,
            sequence = self.sequence,
            provenance = provenance.as_str(),
            frames = self.frames,
            spool_bytes = self.bytes,
            "Recorded a commit frame"
        );
    }

    /// Writes an abandoned branch. The winning branch follows as ordinary commits.
    pub fn write_reorg(&mut self, reorg: Reorg) {
        if !self.records_commits() {
            return
        }
        self.write(FrameKind::Reorg, &StreamEvent::Reorg(reorg));
    }

    /// Tells a consumer it cannot continue from what it holds.
    pub fn write_reset(&mut self, reason: ResetReason, detail: impl Into<String>) {
        if !self.records_commits() {
            return
        }
        let reset = Reset { reason, detail: detail.into() };
        warn!(
            target: "partial_stateless_stream",
            reason = ?reset.reason,
            detail = %reset.detail,
            "Recorded a reset; a consumer of this stream must re-bootstrap here"
        );
        self.write(FrameKind::Reset, &StreamEvent::Reset(reset));
    }

    /// Closes the stream. A corpus without this ended unexpectedly.
    pub fn write_end(&mut self, reason: impl Into<String>) {
        if self.poisoned {
            return
        }
        let end = End { reason: reason.into(), last_sequence: self.sequence };
        self.write(FrameKind::End, &StreamEvent::End(end));
        info!(
            target: "partial_stateless_stream",
            frames = self.frames,
            spool_bytes = self.bytes,
            dir = %self.dir.display(),
            "Closed the event stream"
        );
    }

    /// Encodes and writes one frame, poisoning the recorder if anything fails.
    fn write(&mut self, kind: FrameKind, event: &StreamEvent) {
        if self.poisoned {
            return
        }
        let sequence = self.sequence;
        let result =
            encode_event(sequence, event).map_err(|err| eyre::eyre!("{err}")).and_then(|bytes| {
                let path = self.dir.join(format!("{sequence:012}_{}.frame", kind.as_str()));
                write_atomically(&path, &bytes)?;
                Ok(bytes.len())
            });
        match result {
            Ok(len) => {
                self.sequence += 1;
                self.frames += 1;
                self.bytes += len as u64;
            }
            Err(err) => {
                self.poisoned = true;
                error!(
                    target: "partial_stateless_stream",
                    sequence,
                    kind = kind.as_str(),
                    error = %err,
                    frames = self.frames,
                    "Failed to write a stream frame; recording stops here rather than leaving a \
                     hole. The corpus ends without an End frame, which is how a reader will know"
                );
            }
        }
    }
}

/// Writes through a temporary file, so a reader never sees a partially written frame.
///
/// The same tmp-and-rename the snapshot export uses. It is not crash-atomic across frames — a
/// crash can leave a complete prefix and nothing else — which is exactly the guarantee a sequence
/// numbered spool needs, because a prefix is a valid truncated stream.
fn write_atomically(path: &Path, bytes: &[u8]) -> eyre::Result<()> {
    let temporary = path.with_extension("tmp");
    fs::write(&temporary, bytes)?;
    fs::rename(&temporary, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use partial_stateless_stream::{decode_event, FrameLimits};

    fn recorder(dir: &Path) -> StreamRecorder {
        StreamRecorder {
            dir: dir.to_path_buf(),
            chunk_bytes: 64,
            producer: "test".to_string(),
            sequence: 0,
            checkpointed: false,
            poisoned: false,
            frames: 0,
            bytes: 0,
            durable_block: None,
        }
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
    #[test]
    fn commits_are_not_recorded_until_the_checkpoint_is() {
        let dir = spool_dir("before-checkpoint");
        let mut recorder = recorder(&dir);
        recorder.write_manifest(1, B256::ZERO, B256::with_last_byte(0x44), 60, 30);
        assert!(!recorder.records_commits());

        recorder.write_reset(ResetReason::Gap, "ignored before the checkpoint");
        assert_eq!(frames_in(&dir), vec![(0, FrameKind::Manifest)]);

        recorder.write_checkpoint(&checkpoint(), None, &[7u8; 200]);
        assert!(recorder.records_commits());

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
        recorder.write_manifest(1, B256::ZERO, B256::with_last_byte(0x44), 60, 30);
        recorder.write_checkpoint(&checkpoint(), None, &[1u8; 8]);

        // Removing the directory is the cheapest true I/O failure available here.
        fs::remove_dir_all(&dir).expect("remove spool");
        recorder.write_reset(ResetReason::Overflow, "producer queue overflowed");
        assert!(!recorder.records_commits(), "the first failure stops recording");

        fs::create_dir_all(&dir).expect("recreate");
        recorder.write_end("shutdown");
        assert!(frames_in(&dir).is_empty(), "a poisoned recorder writes nothing further");
        let _ = fs::remove_dir_all(&dir);
    }
}
