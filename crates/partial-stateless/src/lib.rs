//! Network-level state cache for Partial Statelessness PoC.
//!
//! This crate implements a **protocol-level** cache that represents the state subset
//! all network validators are assumed to hold. It is completely separate from reth's
//! internal `ExecutionCache` which optimizes local DB I/O.
//!
//! The cache supports separate eviction policies for accounts vs storage/codes,
//! and tracks which state keys would require a witness (Merkle proof) when a new
//! block arrives.

pub mod accessed_state;
pub mod bootstrap;
pub mod fixture;
pub mod network_cache;
pub mod participant;
pub mod persistence;
pub mod policy;
pub mod readiness;
pub mod trie_cache;
pub mod witness;

pub mod sender_proof;
pub mod sidecar;
pub mod witness_check;

pub use accessed_state::BlockAccessedState;
pub use bootstrap::{
    bootstrap_proof_targets, build_snapshot_package, verify_and_restore,
    verify_and_restore_with_limits, BootstrapError, BootstrapLimits, CacheSnapshotPackage,
    RestoredBootstrapState,
};
pub use fixture::{load_fixtures, save_fixture, AccessedStateFixture, LoadedFixtures};
pub use network_cache::{CachedEntry, NetworkStateCache};
pub use participant::ParticipantCache;
pub use persistence::CacheState;
pub use policy::{CachePolicy, LastNBlocksPolicy};
pub use readiness::{
    BlockContext, BlockedReason, CacheObservation, CacheReadiness, CacheReadinessTracker,
    ReadinessError, ReadyParent, TrustedCheckpoint,
};
pub use sender_proof::{
    SenderAccountProof, SenderAdmissionInput, SenderProofError, VerifiedSender,
};
pub use sidecar::{
    check_next_cache_anchor, check_sidecar_context, check_sidecar_miss_targets,
    check_sidecar_self_consistency, last_n_blocks_cache_policy_id, partial_witness_commitment,
    CacheAnchor, CacheFootprintStats, PartialExecutionWitness, PartialExecutionWitnessState,
    PartialStatelessSidecar, RootWitnessCompletenessReport, RootWitnessCompletenessSummary,
    SerializableMultiProof, SerializableStorageMultiProof, SidecarBenchmarkManifest,
    SidecarCheckError, StateTargetSet, StateTargetStats, WitnessReductionStats, WitnessTargets,
};
pub use trie_cache::{
    PartialTrieNodeCache, PrefixCoverage, StorageTrieMutation, TrieCacheValidationError,
    TrieChangeSet, TrieMutationMetrics, TrieShapeMetrics, TRIE_SHAPE_PREFIX_LEVELS,
};
pub use witness::{measure_multiproof_size, miss_to_proof_targets, WitnessResult};
pub use witness_check::{
    compute_trustless_state_root, root_witness_targets_from_bundle,
    try_compute_trustless_state_root, try_compute_trustless_state_root_v2,
    try_compute_trustless_state_root_v2_with_storage_targets, CacheAwareTransitionProgress,
    CacheAwareTrieTransition, MaterializedStateProof, TrieProofTarget, TrieProofTargetV2,
    TrieTransitionError,
};
