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
        for path in [&self.path, &self.path.with_extension("tmp")] {
            if let Err(error) = reth_fs_util::remove_file_if_exists(path) {
                tracing::warn!(target: "partial_stateless", %error, ?path,
                    "Could not remove expired undo file");
            }
        }
    }
}

/// A small reference to a pending or completed file, with no retained undo payload.
#[derive(Debug, Clone)]
pub struct DiskUndoHandle(Arc<FileState>);

impl DiskUndoHandle {
    /// Load one bundle. A pending writer is joined before reading; errors refuse recovery.
    pub fn load(&self) -> Result<DiskUndoBundle, String> {
        let written = self.0.completion.wait(LOAD_TIMEOUT)?;
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
        if bytes[8..40] != written.checksum[..] || keccak256(&bytes[40..]) != written.checksum {
            return Err("undo file checksum mismatch".into())
        }
        codec().deserialize(&bytes[40..]).map_err(|err| err.to_string())
    }

    /// File location, useful for diagnostics and fault-injection tests.
    pub fn path(&self) -> &Path {
        &self.0.path
    }
}

struct WriteJob {
    file: Arc<FileState>,
    bundle: DiskUndoBundle,
}

/// One writer and a fresh directory per pair. Files are never reused across process restarts.
pub struct DiskUndoStore {
    session_lock: Arc<std::fs::File>,
    directory: Arc<tempfile::TempDir>,
    writer: mpsc::SyncSender<WriteJob>,
    sequence: u64,
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
        std::thread::Builder::new()
            .name("ps-undo-writer".into())
            .spawn(move || {
                for job in reader {
                    // A pruned job with no remaining handle has nothing left to serve.
                    if Arc::strong_count(&job.file) == 1 {
                        continue
                    }
                    let block = job.bundle.flat.block_number();
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        write_bundle(&job.file, &job.bundle)
                    }))
                    .unwrap_or_else(|_| Err("undo writer panicked".into()));
                    if result.is_err() &&
                        let Err(error) = reth_fs_util::remove_file_if_exists(job.file.path.with_extension("tmp"))
                    {
                        tracing::warn!(target: "partial_stateless", block, %error,
                            "Could not remove failed undo write's temporary file");
                    }
                    // Release the payload before reporting that the write has completed.
                    drop(job.bundle);
                    let completion = Arc::clone(&job.file.completion);
                    drop(job.file);
                    if let Err(error) = &result {
                        tracing::warn!(target: "partial_stateless", cause = "disk_write_failed", block, %error,
                            "Undo coverage interrupted at this block; recovery crossing it requires fallback");
                    }
                    if let Ok(mut state) = completion.written.lock() {
                        *state = Some(result);
                        completion.ready.notify_all();
                    }
                }
            })
            .map_err(|err| err.to_string())?;
        Ok(Self { session_lock, directory, writer, sequence: 0 })
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
        });
        send_with_timeout(
            &self.writer,
            WriteJob { file: Arc::clone(&file), bundle },
            SPILL_TIMEOUT,
        )?;
        Ok(DiskUndoHandle(file))
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

fn write_bundle(file: &FileState, bundle: &DiskUndoBundle) -> Result<Written, String> {
    use std::io::Write;
    let payload = codec().serialize(bundle).map_err(|err| err.to_string())?;
    let checksum = keccak256(&payload);
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
