//! Restoring a coordinated pair from an operator-trusted snapshot, without a database.
//!
//! The importing half of bootstrap, and deliberately separate from the exporting half. Exporting
//! needs a state provider to answer the package's proof; importing needs nothing but the package,
//! the checkpoint, and the policy configuration — which is what makes it the path a standalone
//! validator uses, and why it cannot live beside the ExEx.
//!
//! **Trust model.** The snapshot is accepted on operator authority, not on peer consensus: the
//! importing side authenticates everything against a [`TrustedCheckpoint`] the operator supplied
//! out of band, and a package that disagrees with it is discarded. A node bootstrapped this way
//! trusts whoever configured the checkpoint, so this is not trustless new-node sync.
//!
//! What the checkpoint still cannot check is the concrete policy *object* behind the policy ID it
//! names. Both sides deriving their policies from one [`CacheConfig`] is what keeps that gap
//! closed on this path.

use crate::{
    config::CacheConfig,
    network_cache::NetworkStateCache,
    readiness::{CacheObservation, CacheReadinessTracker, ReadyParent, TrustedCheckpoint},
    verify_and_restore_with_limits, BootstrapError, BootstrapLimits, CacheAnchor,
    CacheSnapshotPackage, PartialTrieNodeCache,
};
use alloy_primitives::B256;
use std::time::Instant;
use tracing::info;

/// Bounds for a snapshot this operator produced or was handed out of band.
///
/// [`BootstrapLimits::default`] is sized for an *untrusted peer* package and a live mainnet Last-N
/// window does not fit inside it: a 60-block account window holds on the order of a hundred
/// thousand accounts and several hundred thousand storage slots, and the multiproof covering all
/// of them is far larger than the default 64 MiB. These bounds still exist — a decode is not
/// unbounded — but they are set for the artifact this ExEx actually exchanges rather than for an
/// arbitrary peer, which is the same trust boundary the checkpoint already draws.
pub fn live_limits() -> BootstrapLimits {
    BootstrapLimits {
        max_accounts: 2_000_000,
        max_storage_slots: 8_000_000,
        max_codes: 500_000,
        max_state_proof_bytes: 2 * 1024 * 1024 * 1024,
        max_code_bytes: 1024 * 1024 * 1024,
    }
}

/// Verifies a package against its checkpoint and returns an installed, Ready coordinated pair.
///
/// Nothing is mutated in place: the caller receives the restored pair and the tracker only reaches
/// `Ready` if every check passed, so a rejected package leaves whatever the caller already had.
pub fn restore_snapshot(
    package: CacheSnapshotPackage,
    checkpoint: &TrustedCheckpoint,
    config: &CacheConfig,
) -> Result<RestoredSnapshot, RestoreError> {
    let started = Instant::now();
    if checkpoint.cache_policy_id != config.cache_policy_id() {
        return Err(RestoreError::PolicyMismatch {
            checkpoint: checkpoint.cache_policy_id,
            configured: config.cache_policy_id(),
        })
    }
    let expected_anchor = CacheAnchor {
        block_number: checkpoint.block_number,
        block_hash: checkpoint.block_hash,
        cache_policy_id: checkpoint.cache_policy_id,
        cache_root: checkpoint.cache_root,
    };
    let restored = verify_and_restore_with_limits(
        package,
        &expected_anchor,
        checkpoint.state_root,
        config.account_policy(),
        config.storage_policy(),
        &live_limits(),
    )
    .map_err(|err| RestoreError::Rejected { block: checkpoint.block_number, detail: err })?;

    let mut readiness = config.new_readiness_tracker();
    let observation = CacheObservation::capture(&restored.value_cache, &restored.trie_cache);
    let ready = readiness
        .restore_from_checkpoint(checkpoint, &observation)
        .cloned()
        .map_err(|err| RestoreError::NotReady { block: checkpoint.block_number, detail: err })?;

    info!(
        target: "partial_stateless",
        block = checkpoint.block_number,
        block_hash = ?checkpoint.block_hash,
        accounts = restored.value_cache.accounts().len(),
        storage = restored.value_cache.storage().len(),
        codes = restored.value_cache.codes().len(),
        elapsed_ms = started.elapsed().as_millis() as u64,
        "Restored coordinated cache pair from an operator-trusted snapshot; this node trusts \
         whoever supplied the checkpoint"
    );
    Ok(RestoredSnapshot {
        cache: restored.value_cache,
        trie_cache: restored.trie_cache,
        readiness,
        ready,
    })
}

/// A coordinated pair restored from a snapshot, with its own tracker already promoted.
pub struct RestoredSnapshot {
    /// Authenticated flat values.
    pub cache: NetworkStateCache,
    /// Authenticated trie paths for exactly those values.
    pub trie_cache: PartialTrieNodeCache,
    /// A tracker at `Ready(H)`, carried with the pair so a shadow pair has its own lifecycle.
    pub readiness: CacheReadinessTracker,
    /// The parent this pair may validate the child of.
    pub ready: ReadyParent,
}

/// Why a package was not installed.
///
/// Typed rather than a message, because a standalone validator has to *act* on the difference: a
/// policy mismatch is a configuration error the operator fixes, a rejected package means the
/// checkpoint and the bytes disagree and neither can be trusted, and a readiness refusal means the
/// package verified and still does not describe a state this tracker can validate against.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RestoreError {
    /// The checkpoint names a cache policy this node does not run.
    #[error("bootstrap checkpoint names policy {checkpoint} but this node runs {configured}")]
    PolicyMismatch {
        /// Policy the checkpoint names.
        checkpoint: B256,
        /// Policy this node derives from its own configuration.
        configured: B256,
    },
    /// The package did not verify against the checkpoint.
    #[error("bootstrap snapshot at block {block} rejected: {detail}")]
    Rejected {
        /// Block the checkpoint names.
        block: u64,
        /// What the verification objected to.
        detail: BootstrapError,
    },
    /// The package verified and the tracker still refused to become Ready on it.
    #[error(
        "restored snapshot at block {block} was rejected by the readiness checkpoint: {detail:?}"
    )]
    NotReady {
        /// Block the checkpoint names.
        block: u64,
        /// What readiness objected to.
        detail: crate::readiness::ReadinessError,
    },
}
