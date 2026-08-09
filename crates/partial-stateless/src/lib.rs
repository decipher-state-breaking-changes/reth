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
pub mod cold_admission;
pub mod fixture;
pub mod network_cache;
pub mod participant;
pub mod persistence;
pub mod policy;
pub mod readiness;
pub mod shared_trie;
pub mod trie_cache;
pub mod witness;

pub mod sender_proof;
pub mod sidecar;
pub mod witness_check;

pub use accessed_state::BlockAccessedState;
pub use bootstrap::{
    bootstrap_proof_targets, build_snapshot_package, build_snapshot_package_with_limits,
    rebuild_trie_cache, verify_and_restore, verify_and_restore_with_limits, BootstrapError,
    BootstrapLimits, CacheSnapshotPackage, RebuiltTrieCache, RestoredBootstrapState,
};
pub use cold_admission::{
    admit_cold_sender, ColdAdmissionError, ColdAdmissionRequest, ColdSenderAdmission,
};
pub use fixture::{load_fixtures, save_fixture, AccessedStateFixture, LoadedFixtures};
pub use network_cache::{CacheRootTimings, CachedEntry, MembershipDelta, NetworkStateCache};
pub use participant::ParticipantCache;
pub use persistence::CacheState;
pub use policy::{CachePolicy, EvictedStorage, LastNBlocksPolicy};
pub use readiness::{
    BlockContext, BlockedReason, CacheObservation, CacheReadiness, CacheReadinessTracker,
    ReadinessError, ReadyParent, TrustedCheckpoint,
};
pub use reth_trie_sparse::RetainWitnessPathsMetrics;
pub use sender_proof::{
    SenderAccountProof, SenderAdmissionInput, SenderProofError, VerifiedSender,
};
pub use shared_trie::{cow_copies_taken, SharedSparseTrie};
pub use sidecar::{
    check_next_cache_anchor, check_sidecar_context, check_sidecar_miss_targets,
    check_sidecar_self_consistency, last_n_blocks_cache_policy_id, partial_witness_commitment,
    CacheAnchor, CacheFootprintStats, PartialExecutionWitness, PartialExecutionWitnessState,
    PartialStatelessSidecar, RootWitnessCompletenessReport, RootWitnessCompletenessSummary,
    SerializableMultiProof, SerializableStorageMultiProof, SidecarBenchmarkManifest,
    SidecarCheckError, StateTargetSet, StateTargetStats, WitnessReductionStats, WitnessTargets,
};
pub use trie_cache::{
    PartialTrieNodeCache, PrefixCoverage, RetentionTimings, StorageTrieMutation,
    TrieCacheValidationError, TrieChangeSet, TrieCloneTimings, TrieMutationMetrics,
    TrieShapeMetrics, TRIE_SHAPE_PREFIX_LEVELS,
};
pub use witness::{measure_multiproof_size, miss_to_proof_targets, WitnessResult};
pub use witness_check::{
    compute_trustless_state_root, root_witness_targets_from_bundle,
    try_compute_trustless_state_root, try_compute_trustless_state_root_v2,
    try_compute_trustless_state_root_v2_with_storage_targets, CacheAwareTransitionProgress,
    CacheAwareTrieTransition, MaterializedStateProof, TrieProofTarget, TrieProofTargetV2,
    TrieTransitionError,
};
