//! Operator-trusted snapshot export and import for a node that cannot replay.
//!
//! A full node reaches `Ready` through [`crate::rebuild`] and never needs a snapshot file. This
//! path exists so that a node *without* the database can reach the same place, which is the one
//! thing replay cannot do for it.
//!
//! **Trust model.** The snapshot is accepted on operator authority, not on peer consensus: the
//! importing side authenticates everything against a [`TrustedCheckpoint`] the operator supplied
//! out of band, and a package that disagrees with it is discarded. A node bootstrapped this way
//! trusts whoever configured the checkpoint, so this is not yet trustless new-node sync. For the
//! single-node proof of concept it is self-bootstrap — one run exports at `Ready(H)` and the
//! import side consumes it — which is the same trust model rather than a weaker version of it,
//! because the operator being trusted is the same operator.
//!
//! What the checkpoint still cannot check is the concrete policy *object* behind the policy ID it
//! names. Both sides here derive their policies and their ID from one [`CacheConfig`], which is
//! what keeps that gap closed on this path.

use alloy_primitives::B256;
// The restore half lives in `partial-stateless` and is re-exported here, because a node without a
// database restores exactly the same way a node with one does and a second copy would be a second
// thing to keep in step. Only the export needs a provider, so only the export stayed.
pub use partial_stateless::restore::{live_limits, restore_snapshot, RestoredSnapshot};
use partial_stateless::{
    build_snapshot_package_with_limits,
    network_cache::NetworkStateCache,
    readiness::{ReadyParent, TrustedCheckpoint},
    CacheSnapshotPackage,
};
use reth_provider::StateProvider;
use reth_trie_common::TrieInput;
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};
use tracing::{info, warn};

/// Filename of the snapshot package inside the configured bootstrap directory.
pub const PACKAGE_FILE: &str = "cache_snapshot.bin";
/// Filename of the operator checkpoint that authenticates the package.
pub const CHECKPOINT_FILE: &str = "cache_checkpoint.json";

/// Builds a snapshot package at the Ready parent and writes it with its checkpoint.
///
/// `state_provider` must be state *at* `ready.anchor.block_hash`, since the package's proof is
/// checked against that block's state root.
pub fn export_snapshot(
    dir: &Path,
    cache: &NetworkStateCache,
    ready: &ReadyParent,
    state_provider: &dyn StateProvider,
) -> eyre::Result<ExportedSnapshot> {
    let started = Instant::now();
    let mut proof_targets = 0usize;
    let package = build_snapshot_package_with_limits(
        cache,
        ready.anchor,
        ready.trie_state_root,
        |targets| {
            proof_targets = targets.chunking_length();
            state_provider.multiproof(TrieInput::default(), targets).map_err(|err| err.to_string())
        },
        &live_limits(),
    )
    .map_err(|err| {
        eyre::eyre!("failed to build snapshot at block {}: {err}", ready.anchor.block_number)
    })?;

    let checkpoint = TrustedCheckpoint {
        block_number: ready.anchor.block_number,
        block_hash: ready.anchor.block_hash,
        state_root: ready.trie_state_root,
        cache_root: ready.anchor.cache_root,
        cache_policy_id: ready.anchor.cache_policy_id,
    };

    let (package_path, checkpoint_path, package_bytes) =
        write_snapshot(dir, &package, &checkpoint)?;

    let exported = ExportedSnapshot {
        package,
        checkpoint,
        package_path,
        checkpoint_path,
        package_bytes,
        proof_targets,
        elapsed_us: started.elapsed().as_micros() as u64,
    };
    info!(
        target: "partial_stateless",
        block = checkpoint.block_number,
        block_hash = ?checkpoint.block_hash,
        path = %exported.package_path.display(),
        package_bytes = exported.package_bytes,
        proof_targets,
        elapsed_ms = exported.elapsed_us / 1_000,
        "Exported operator-trusted cache snapshot"
    );
    Ok(exported)
}

/// Writes a package and its checkpoint into `dir`, returning both paths and the package size.
pub fn write_snapshot(
    dir: &Path,
    package: &CacheSnapshotPackage,
    checkpoint: &TrustedCheckpoint,
) -> eyre::Result<(PathBuf, PathBuf, usize)> {
    fs::create_dir_all(dir)?;
    let package_path = dir.join(PACKAGE_FILE);
    let checkpoint_path = dir.join(CHECKPOINT_FILE);
    let package_bytes = bincode::serialize(package)
        .map_err(|err| eyre::eyre!("failed to serialize snapshot package: {err}"))?;
    write_atomically(&package_path, &package_bytes)?;
    let stored = serde_json::to_vec_pretty(&StoredCheckpoint::from(checkpoint))?;
    write_atomically(&checkpoint_path, &stored)?;
    Ok((package_path, checkpoint_path, package_bytes.len()))
}

/// A written snapshot, retained so the same process can restore from it without re-reading.
#[derive(Debug)]
pub struct ExportedSnapshot {
    /// The package as written.
    pub package: CacheSnapshotPackage,
    /// The checkpoint an importer must be given out of band.
    pub checkpoint: TrustedCheckpoint,
    /// Where the package was written.
    pub package_path: PathBuf,
    /// Where the checkpoint was written.
    pub checkpoint_path: PathBuf,
    /// Serialized package size.
    pub package_bytes: usize,
    /// Leaf targets in the final multiproof request.
    pub proof_targets: usize,
    /// Wall time of the export.
    pub elapsed_us: u64,
}

/// Reads a package and its checkpoint from `dir`.
///
/// Returns `Ok(None)` when the directory holds no package, which is the ordinary case for a node
/// that has never been given one.
pub fn load_snapshot(
    dir: &Path,
) -> eyre::Result<Option<(CacheSnapshotPackage, TrustedCheckpoint)>> {
    let package_path = dir.join(PACKAGE_FILE);
    let checkpoint_path = dir.join(CHECKPOINT_FILE);
    if !package_path.exists() && !checkpoint_path.exists() {
        return Ok(None)
    }
    // One of the two present without the other is an operator error, not an absence: restoring
    // needs both, and silently continuing cold would hide a misconfigured directory.
    if !package_path.exists() {
        eyre::bail!("bootstrap checkpoint {} has no package beside it", checkpoint_path.display());
    }
    if !checkpoint_path.exists() {
        eyre::bail!("bootstrap package {} has no checkpoint beside it", package_path.display());
    }

    let package: CacheSnapshotPackage = bincode::deserialize(&fs::read(&package_path)?)
        .map_err(|err| eyre::eyre!("failed to decode {}: {err}", package_path.display()))?;
    let stored: StoredCheckpoint = serde_json::from_slice(&fs::read(&checkpoint_path)?)
        .map_err(|err| eyre::eyre!("failed to decode {}: {err}", checkpoint_path.display()))?;
    Ok(Some((package, stored.into())))
}

/// Writes through a temporary file so a crash cannot leave a half-written artifact in place.
///
/// This is not the crash-atomic joint persistence the durable checkpoint design calls for — that
/// needs generation metadata across both caches — only enough that an interrupted export does not
/// present a truncated package as a complete one.
fn write_atomically(path: &Path, bytes: &[u8]) -> eyre::Result<()> {
    let temporary = path.with_extension("tmp");
    fs::write(&temporary, bytes)?;
    fs::rename(&temporary, path)?;
    Ok(())
}

/// On-disk form of [`TrustedCheckpoint`], which is deliberately not serializable itself.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct StoredCheckpoint {
    block_number: u64,
    block_hash: B256,
    state_root: B256,
    cache_root: B256,
    cache_policy_id: B256,
}

impl From<&TrustedCheckpoint> for StoredCheckpoint {
    fn from(checkpoint: &TrustedCheckpoint) -> Self {
        Self {
            block_number: checkpoint.block_number,
            block_hash: checkpoint.block_hash,
            state_root: checkpoint.state_root,
            cache_root: checkpoint.cache_root,
            cache_policy_id: checkpoint.cache_policy_id,
        }
    }
}

impl From<StoredCheckpoint> for TrustedCheckpoint {
    fn from(stored: StoredCheckpoint) -> Self {
        Self {
            block_number: stored.block_number,
            block_hash: stored.block_hash,
            state_root: stored.state_root,
            cache_root: stored.cache_root,
            cache_policy_id: stored.cache_policy_id,
        }
    }
}

/// Logs the drift a snapshot import always has: the chain moved on while the process started.
pub fn warn_on_head_drift(checkpoint: &TrustedCheckpoint, first_notified_block: u64) {
    if first_notified_block == checkpoint.block_number + 1 {
        return
    }
    warn!(
        target: "partial_stateless",
        snapshot_block = checkpoint.block_number,
        first_block = first_notified_block,
        "Imported snapshot is stale relative to the head; a node that can replay recovers by \
         canonical rebuild, a node that cannot stays Cold until a fresher snapshot is supplied"
    );
}
