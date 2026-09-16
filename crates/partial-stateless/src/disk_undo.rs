//! Disposable, session-local files for coordinated trie and value-cache undo.
//!
//! A rendezvous channel bounds outstanding writes to one worker. A handle owns its file even
//! while the write is pending: pruning the handle cannot let a late completion resurrect history.

use crate::{network_cache::BlockCacheUndo, TrieCacheUndoFrame};
use alloy_primitives::{keccak256, B256};
use bincode::Options;
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    sync::{mpsc, Arc, Condvar, Mutex},
};

const MAGIC: &[u8; 8] = b"PSUNDO01";
const MAX_BYTES: u64 = 512 * 1024 * 1024;

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

impl Drop for FileState {
    fn drop(&mut self) {
        let _ = reth_fs_util::remove_file(&self.path);
        let _ = reth_fs_util::remove_file(self.path.with_extension("tmp"));
    }
}

/// A small reference to a pending or completed file, with no retained undo payload.
#[derive(Debug, Clone)]
pub struct DiskUndoHandle(Arc<FileState>);

impl DiskUndoHandle {
    /// Load one bundle. A pending writer is joined before reading; errors refuse recovery.
    pub fn load(&self) -> Result<DiskUndoBundle, String> {
        let mut state = self.0.completion.written.lock().map_err(|err| err.to_string())?;
        while state.is_none() {
            state = self.0.completion.ready.wait(state).map_err(|err| err.to_string())?;
        }
        let written = state.as_ref().expect("writer completed").clone()?;
        drop(state);
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
        let (writer, reader) = mpsc::sync_channel::<WriteJob>(0);
        std::thread::Builder::new()
            .name("ps-undo-writer".into())
            .spawn(move || {
                for job in reader {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        write_bundle(&job.file, &job.bundle)
                    }))
                    .unwrap_or_else(|_| Err("undo writer panicked".into()));
                    // Release the payload before reporting that the write has completed.
                    drop(job.bundle);
                    let completion = Arc::clone(&job.file.completion);
                    drop(job.file);
                    if let Err(error) = &result {
                        tracing::warn!(target: "partial_stateless", %error, "Undo spill failed; this history will require recovery fallback");
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

    /// Transfer the payload to the writer. Backpressure bounds memory if the disk falls behind.
    pub fn spill(&mut self, bundle: DiskUndoBundle) -> DiskUndoHandle {
        self.sequence += 1;
        let file = Arc::new(FileState {
            session_lock: Arc::clone(&self.session_lock),
            directory: Arc::clone(&self.directory),
            path: self.directory.path().join(format!("{}.undo", self.sequence)),
            completion: Arc::default(),
        });
        if self.writer.send(WriteJob { file: Arc::clone(&file), bundle }).is_err() {
            *file.completion.written.lock().expect("new file lock") =
                Some(Err("undo writer stopped".into()));
            file.completion.ready.notify_all();
        }
        DiskUndoHandle(file)
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
