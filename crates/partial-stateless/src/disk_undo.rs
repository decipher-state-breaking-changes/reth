//! Disposable, session-local files for coordinated trie and value-cache undo.
//!
//! A one-slot queue bounds outstanding writes to one active and one queued bundle. A handle owns
//! its file even while the write is pending: pruning the handle cannot let a late completion
//! resurrect history.

use crate::{network_cache::BlockCacheUndo, TrieCacheUndoFrame};
use alloy_primitives::{keccak256, B256};
use bincode::Options;
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    sync::{mpsc, Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

const MAGIC: &[u8; 8] = b"PSUNDO01";
const MAX_BYTES: u64 = 512 * 1024 * 1024;
const SPILL_WARN_AFTER: Duration = Duration::from_millis(100);
const SPILL_TIMEOUT: Duration = Duration::from_secs(1);
const LOAD_TIMEOUT: Duration = Duration::from_secs(5);

/// Both halves of one block's undo, always written and loaded together.
#[derive(Debug, Serialize, Deserialize)]
pub struct DiskUndoBundle {
    /// Hash of the generation restored by this bundle.
    pub parent_hash: B256,
    /// Trie preimages and displaced storage tries.
    pub trie: TrieCacheUndoFrame,
    /// Values, access metadata and cached root before the block.
    pub flat: BlockCacheUndo,
}

#[derive(Debug, Clone, Copy)]
struct Written {
    bytes: u64,
    checksum: B256,
}

#[derive(Debug)]
struct FileState {
    session_lock: Arc<std::fs::File>,
    directory: Arc<tempfile::TempDir>,
    path: PathBuf,
    completion: Arc<Completion>,
    metrics: Arc<WriterMetrics>,
}

#[derive(Debug, Default)]
struct Completion {
    written: Mutex<Option<Result<Written, String>>>,
    ready: Condvar,
}

impl Completion {
    fn wait(&self, timeout: Duration) -> Result<Written, String> {
        let state = self.written.lock().map_err(|err| err.to_string())?;
        let (state, _) = self
            .ready
            .wait_timeout_while(state, timeout, |state| state.is_none())
            .map_err(|err| err.to_string())?;
        state.as_ref().cloned().unwrap_or_else(|| Err("undo writer completion timed out".into()))
    }
}

impl Drop for FileState {
    fn drop(&mut self) {
        let started = Instant::now();
        for path in [&self.path, &self.path.with_extension("tmp")] {
            if let Err(error) = reth_fs_util::remove_file_if_exists(path) {
                tracing::warn!(target: "partial_stateless", %error, ?path,
                    "Could not remove expired undo file");
            }
        }
        self.metrics.update(|metrics| metrics.expire_us += micros(started));
    }
}

/// A small reference to a pending or completed file, with no retained undo payload.
#[derive(Debug, Clone)]
pub struct DiskUndoHandle(Arc<FileState>);

impl DiskUndoHandle {
    /// Whether this bundle is known to be written; does not wait or read the file again.
    pub fn is_written(&self) -> bool {
        self.0.completion.written.lock().is_ok_and(|state| matches!(*state, Some(Ok(_))))
    }

    /// Load one bundle. A pending writer is joined before reading; errors refuse recovery.
    pub fn load(&self) -> Result<DiskUndoBundle, String> {
        self.load_timed().map(|(bundle, _)| bundle)
    }

    /// Read/decode costs, including waiting for a not-yet-completed writer.
    pub fn load_timed(&self) -> Result<(DiskUndoBundle, UndoLoadTimings), String> {
        let started = Instant::now();
        let written = self.0.completion.wait(LOAD_TIMEOUT)?;
        let wait_us = micros(started);
        let started = Instant::now();
        let file = std::fs::File::open(&self.0.path).map_err(|err| err.to_string())?;
        if file.metadata().map_err(|err| err.to_string())?.len() != written.bytes ||
            written.bytes > MAX_BYTES + 40
        {
            return Err("undo file length mismatch".into())
        }
        // Bound the read as well as decoding, including a file changed after metadata was read.
        use std::io::Read;
        let mut bytes = Vec::new();
        file.take(written.bytes + 1).read_to_end(&mut bytes).map_err(|err| err.to_string())?;
        if bytes.len() as u64 != written.bytes || bytes.len() < 40 || &bytes[..8] != MAGIC {
            return Err("undo file header mismatch".into())
        }
        let read_us = micros(started);
        let started = Instant::now();
        if bytes[8..40] != written.checksum[..] || keccak256(&bytes[40..]) != written.checksum {
            return Err("undo file checksum mismatch".into())
        }
        let checksum_us = micros(started);
        let started = Instant::now();
        let bundle = codec().deserialize(&bytes[40..]).map_err(|err| err.to_string())?;
        Ok((
            bundle,
            UndoLoadTimings {
                wait_us,
                read_us,
                checksum_us,
                decode_us: micros(started),
                bytes: written.bytes,
            },
        ))
    }

    /// File location, useful for diagnostics and fault-injection tests.
    pub fn path(&self) -> &Path {
        &self.0.path
    }
}

struct WriteJob {
    file: Arc<FileState>,
    bundle: DiskUndoBundle,
    enqueued: Instant,
}

/// One writer and a fresh directory per pair. Files are never reused across process restarts.
pub struct DiskUndoStore {
    session_lock: Arc<std::fs::File>,
    directory: Arc<tempfile::TempDir>,
    writer: mpsc::SyncSender<WriteJob>,
    sequence: u64,
    metrics: Arc<WriterMetrics>,
}

/// Cumulative writer counters. Pending payloads and encoding buffers are not resident history.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct DiskUndoMetrics {
    pub submitted: u64,
    pub completed: u64,
    pub failed: u64,
    pub cancelled: u64,
    pub enqueue_failures: u64,
    pub pending: u64,
    pub pending_peak: u64,
    pub bytes_written: u64,
    pub enqueue_us: u64,
    pub serialize_us: u64,
    pub checksum_us: u64,
    pub write_us: u64,
    pub payload_drop_us: u64,
    pub expire_us: u64,
    pub telemetry_failures: u64,
}

/// One bundle's writer work, reported independently of the commit that enqueued it.
#[derive(Debug, Default, Serialize)]
pub struct UndoWriteTimings {
    pub kind: &'static str,
    pub block: u64,
    pub parent_hash: B256,
    pub queue_wait_us: u64,
    pub serialize_us: u64,
    pub checksum_us: u64,
    pub write_us: u64,
    pub payload_drop_us: u64,
    pub completion_us: u64,
    pub bytes: u64,
    pub error: Option<String>,
}

/// Non-overlapping file-load phases; checksum validation is separate from decoding.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct UndoLoadTimings {
    pub wait_us: u64,
    pub read_us: u64,
    pub checksum_us: u64,
    pub decode_us: u64,
    pub bytes: u64,
}

#[derive(Debug, Default)]
struct WriterMetrics {
    state: Mutex<DiskUndoMetrics>,
    idle: Condvar,
    output: Mutex<Option<std::fs::File>>,
}

impl WriterMetrics {
    fn update(&self, update: impl FnOnce(&mut DiskUndoMetrics)) {
        update(&mut self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner));
    }

    fn record(&self, timing: &impl Serialize) {
        use std::io::Write;
        let mut output = self.output.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(output) = output.as_mut() {
            let result = (|| -> Result<(), String> {
                let mut row = serde_json::to_vec(timing).map_err(|error| error.to_string())?;
                row.push(b'\n');
                output.write_all(&row).map_err(|error| error.to_string())
            })();
            if let Err(error) = result {
                self.update(|metrics| metrics.telemetry_failures += 1);
                tracing::warn!(target: "partial_stateless", %error, "Undo telemetry write failed");
            }
        }
    }
}

fn micros(started: Instant) -> u64 {
    started.elapsed().as_micros().min(u64::MAX as u128) as u64
}

impl DiskUndoStore {
    /// Create a fresh session below `root`; never clear another pair's files.
    pub fn new(root: &Path) -> Result<Self, String> {
        reth_fs_util::create_dir_all(root).map_err(|err| err.to_string())?;
        // Serialize cleanup and session creation. Per-session locks distinguish crash leftovers
        // from another live pair (including another process using the same configured root).
        let root_lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(root.join(".undo-sessions.lock"))
            .map_err(|err| err.to_string())?;
        root_lock.lock().map_err(|err| err.to_string())?;
        for entry in reth_fs_util::read_dir(root).map_err(|err| err.to_string())? {
            let entry = entry.map_err(|err| err.to_string())?;
            if !entry.file_type().map_err(|err| err.to_string())?.is_dir() ||
                !entry.file_name().to_string_lossy().starts_with("undo-")
            {
                continue
            }
            if let Ok(lock) = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(entry.path().join("PSUNDO01.lock")) &&
                lock.try_lock().is_ok()
            {
                let _ = reth_fs_util::remove_dir_all(entry.path());
            }
        }
        let directory = Arc::new(
            tempfile::Builder::new()
                .prefix("undo-")
                .tempdir_in(root)
                .map_err(|err| err.to_string())?,
        );
        let session_lock = Arc::new(
            std::fs::File::create(directory.path().join("PSUNDO01.lock"))
                .map_err(|err| err.to_string())?,
        );
        session_lock.lock().map_err(|err| err.to_string())?;
        let (writer, reader) = mpsc::sync_channel::<WriteJob>(1);
        let metrics = Arc::new(WriterMetrics::default());
        let worker_metrics = Arc::clone(&metrics);
        std::thread::Builder::new()
            .name("ps-undo-writer".into())
            .spawn(move || {
                for job in reader {
                    // A pruned job with no remaining handle has nothing left to serve.
                    if Arc::strong_count(&job.file) == 1 {
                        drop(job);
                        worker_metrics.update(|metrics| {
                            metrics.cancelled += 1;
                            metrics.pending -= 1;
                        });
                        worker_metrics.idle.notify_all();
                        continue
                    }
                    let block = job.bundle.flat.block_number();
                    let mut timing = UndoWriteTimings {
                        kind: "write", block, parent_hash: job.bundle.parent_hash,
                        queue_wait_us: micros(job.enqueued), ..Default::default()
                    };
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        write_bundle(&job.file, &job.bundle, &mut timing)
                    }))
                    .unwrap_or_else(|_| Err("undo writer panicked".into()));
                    if result.is_err() &&
                        let Err(error) = reth_fs_util::remove_file_if_exists(job.file.path.with_extension("tmp"))
                    {
                        tracing::warn!(target: "partial_stateless", block, %error,
                            "Could not remove failed undo write's temporary file");
                    }
                    // Release the payload before reporting that the write has completed.
                    let started = Instant::now();
                    drop(job.bundle);
                    timing.payload_drop_us = micros(started);
                    timing.completion_us = micros(job.enqueued);
                    timing.error = result.as_ref().err().cloned();
                    let completion = Arc::clone(&job.file.completion);
                    drop(job.file);
                    if let Err(error) = &result {
                        tracing::warn!(target: "partial_stateless", cause = "disk_write_failed", block, %error,
                            "Undo coverage interrupted at this block; recovery crossing it requires fallback");
                    }
                    worker_metrics.record(&timing);
                    let succeeded = result.is_ok();
                    if let Ok(mut state) = completion.written.lock() {
                        *state = Some(result);
                        completion.ready.notify_all();
                    }
                    worker_metrics.update(|metrics| {
                        if succeeded { metrics.completed += 1; } else { metrics.failed += 1; }
                        metrics.bytes_written += timing.bytes;
                        metrics.serialize_us += timing.serialize_us;
                        metrics.checksum_us += timing.checksum_us;
                        metrics.write_us += timing.write_us;
                        metrics.payload_drop_us += timing.payload_drop_us;
                        metrics.pending -= 1;
                    });
                    worker_metrics.idle.notify_all();
                }
            })
            .map_err(|err| err.to_string())?;
        Ok(Self { session_lock, directory, writer, sequence: 0, metrics })
    }

    /// Optional per-bundle JSONL outside the disposable undo session. Call before the first spill.
    pub fn enable_metrics(&self, directory: &Path) -> Result<(), String> {
        if self.metrics().submitted != 0 {
            return Err("undo metrics must be enabled before the first spill".into())
        }
        reth_fs_util::create_dir_all(directory).map_err(|error| error.to_string())?;
        let name = self.directory.path().file_name().ok_or("undo session has no name")?;
        let path = directory.join(name).with_extension("jsonl");
        let output = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(path)
            .map_err(|error| error.to_string())?;
        *self.metrics.output.lock().map_err(|error| error.to_string())? = Some(output);
        Ok(())
    }

    /// A cheap cumulative snapshot; no file scanning, payload traversal or writer wait.
    pub fn metrics(&self) -> DiskUndoMetrics {
        *self.metrics.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Emits one recovery event even when a candidate is refused partway through.
    pub fn recovery_probe(&self, block: u64, depth: u64) -> UndoRecoveryProbe {
        UndoRecoveryProbe {
            metrics: Arc::clone(&self.metrics),
            started: Instant::now(),
            timings: UndoRecoveryTimings { block, depth, ..Default::default() },
        }
    }

    /// Drain background work at an experiment boundary, never at each commit.
    pub fn wait_for_idle(&self, timeout: Duration) -> Result<DiskUndoMetrics, String> {
        let state = self.metrics.state.lock().map_err(|error| error.to_string())?;
        let (state, _) = self
            .metrics
            .idle
            .wait_timeout_while(state, timeout, |state| state.pending != 0)
            .map_err(|error| error.to_string())?;
        if state.pending != 0 {
            return Err("undo writer drain timed out".into())
        }
        Ok(*state)
    }

    /// Transfer the payload with bounded backpressure. A stalled disk cannot block commits
    /// indefinitely: after one second, the caller must discard the unreachable undo suffix.
    pub fn spill(&mut self, bundle: DiskUndoBundle) -> Result<DiskUndoHandle, String> {
        self.sequence += 1;
        let file = Arc::new(FileState {
            session_lock: Arc::clone(&self.session_lock),
            directory: Arc::clone(&self.directory),
            path: self.directory.path().join(format!("{}.undo", self.sequence)),
            completion: Arc::default(),
            metrics: Arc::clone(&self.metrics),
        });
        let started = Instant::now();
        self.metrics.update(|metrics| {
            metrics.submitted += 1;
            metrics.pending += 1;
            metrics.pending_peak = metrics.pending_peak.max(metrics.pending);
        });
        let result = send_with_timeout(
            &self.writer,
            WriteJob { file: Arc::clone(&file), bundle, enqueued: started },
            SPILL_TIMEOUT,
        );
        self.metrics.update(|metrics| {
            metrics.enqueue_us += micros(started);
            if result.is_err() {
                metrics.enqueue_failures += 1;
                metrics.pending -= 1;
            }
        });
        if result.is_err() {
            self.metrics.idle.notify_all();
        }
        result?;
        Ok(DiskUndoHandle(file))
    }
}

/// Recovery phases. The residual includes checks and any failed load that could not return timings.
#[derive(Debug, Default, Serialize)]
pub struct UndoRecoveryTimings {
    pub block: u64,
    pub depth: u64,
    pub success: bool,
    pub candidate_copy_us: u64,
    pub pending_wait_us: u64,
    pub read_us: u64,
    pub checksum_us: u64,
    pub decode_us: u64,
    pub undo_us: u64,
    pub publish_us: u64,
    pub check_and_unattributed_us: u64,
    pub total_us: u64,
    pub bytes_read: u64,
}

pub struct UndoRecoveryProbe {
    metrics: Arc<WriterMetrics>,
    started: Instant,
    pub timings: UndoRecoveryTimings,
}

impl Drop for UndoRecoveryProbe {
    fn drop(&mut self) {
        self.timings.total_us = micros(self.started);
        let t = &self.timings;
        let measured = t.candidate_copy_us +
            t.pending_wait_us +
            t.read_us +
            t.checksum_us +
            t.decode_us +
            t.undo_us +
            t.publish_us;
        self.timings.check_and_unattributed_us = self.timings.total_us.saturating_sub(measured);
        self.metrics.record(&serde_json::json!({ "kind": "recovery", "timings": self.timings }));
    }
}

/// Keep the normal enqueue nonblocking, warn while stalled, and refuse after a bounded wait.
fn send_with_timeout<T>(
    writer: &mpsc::SyncSender<T>,
    mut job: T,
    timeout: Duration,
) -> Result<(), String> {
    let started = Instant::now();
    let mut warned = false;
    loop {
        match writer.try_send(job) {
            Ok(()) => return Ok(()),
            Err(mpsc::TrySendError::Disconnected(_)) => return Err("undo writer stopped".into()),
            Err(mpsc::TrySendError::Full(returned)) => job = returned,
        }
        let elapsed = started.elapsed();
        if !warned && elapsed >= SPILL_WARN_AFTER {
            tracing::warn!(target: "partial_stateless", blocked_ms = elapsed.as_millis() as u64,
                timeout_ms = timeout.as_millis() as u64, "Undo spill waiting for disk writer");
            warned = true;
        }
        if elapsed >= timeout {
            return Err(format!("undo spill timed out after {} ms", elapsed.as_millis()))
        }
        std::thread::sleep(Duration::from_millis(5).min(timeout.saturating_sub(elapsed)));
    }
}

fn codec() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(MAX_BYTES)
        .reject_trailing_bytes()
}

fn write_bundle(
    file: &FileState,
    bundle: &DiskUndoBundle,
    timing: &mut UndoWriteTimings,
) -> Result<Written, String> {
    use std::io::Write;
    let started = Instant::now();
    let payload = codec().serialize(bundle).map_err(|err| err.to_string())?;
    timing.serialize_us = micros(started);
    let started = Instant::now();
    let checksum = keccak256(&payload);
    timing.checksum_us = micros(started);
    let started = Instant::now();
    let temporary = file.path.with_extension("tmp");
    let mut output = std::fs::File::create(&temporary).map_err(|err| err.to_string())?;
    output.write_all(MAGIC).map_err(|err| err.to_string())?;
    output.write_all(checksum.as_slice()).map_err(|err| err.to_string())?;
    output.write_all(&payload).map_err(|err| err.to_string())?;
    drop(output);
    // The directory is owned until the last handle/pending job goes away. No fsync: this is
    // deliberately not a restart checkpoint, and a missing file always refuses the undo.
    let _keep_session_alive = (&file.directory, &file.session_lock);
    reth_fs_util::rename(&temporary, &file.path).map_err(|err| err.to_string())?;
    timing.write_us = micros(started);
    timing.bytes = payload.len() as u64 + 40;
    Ok(Written { bytes: payload.len() as u64 + 40, checksum })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disk_undo_writer_queue_has_one_slot_and_refuses_a_stall() {
        let (sender, receiver) = mpsc::sync_channel(1);
        send_with_timeout(&sender, 1, Duration::ZERO).unwrap();
        assert!(send_with_timeout(&sender, 2, Duration::ZERO).unwrap_err().contains("timed out"));
        assert_eq!(receiver.recv().unwrap(), 1);
        send_with_timeout(&sender, 3, Duration::ZERO).unwrap();
        assert_eq!(receiver.recv().unwrap(), 3);
        drop(receiver);
        assert!(send_with_timeout(&sender, 4, Duration::ZERO).unwrap_err().contains("stopped"));
    }

    #[test]
    fn disk_undo_completion_wait_is_bounded_and_can_be_retried() {
        let completion = Completion::default();
        assert!(completion.wait(Duration::ZERO).unwrap_err().contains("timed out"));
        *completion.written.lock().unwrap() = Some(Ok(Written { bytes: 40, checksum: B256::ZERO }));
        assert_eq!(completion.wait(Duration::ZERO).unwrap().bytes, 40);
    }

    #[test]
    fn disk_undo_startup_cleans_only_inactive_owned_sessions() {
        let root = tempfile::tempdir().unwrap();
        let active = DiskUndoStore::new(root.path()).unwrap();
        let active_path = active.directory.path().to_owned();
        let abandoned = root.path().join("undo-abandoned");
        std::fs::create_dir(&abandoned).unwrap();
        std::fs::write(abandoned.join("PSUNDO01.lock"), []).unwrap();
        std::fs::write(abandoned.join("1.undo"), b"crash leftover").unwrap();
        let unrelated = root.path().join("undo-unrelated");
        std::fs::create_dir(&unrelated).unwrap();
        let second = DiskUndoStore::new(root.path()).unwrap();
        assert!(active_path.exists(), "another pair's lock protects its files");
        assert!(!abandoned.exists());
        assert!(unrelated.exists(), "a directory without our marker is not ours");
        drop(second);
        drop(active);
        assert!(!active_path.exists());
    }
}
