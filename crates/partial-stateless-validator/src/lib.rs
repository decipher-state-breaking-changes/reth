//! Database-free partial-stateless block validator.
//!
//! This crate is the validation core by itself: given a parent value cache, a parent trie cache,
//! a block, and its sidecar, it re-executes the block against cache-plus-witness state, computes
//! the post-state root from a local sparse trie, checks the next coordinated cache anchor, and
//! commits the transition — or leaves both caches exactly at the parent generation.
//!
//! **It has no state database, and that is a property of the dependency graph rather than of the
//! code paths taken.** Nothing reachable from here can open a Reth provider or an MDBX
//! environment, so no error, restart, gap, or reorg branch can quietly fall back to one. Callers
//! that *do* have a database — the ExEx builder and live verifier — reach it through
//! [`PostStateRootOracle`], which lets them cross-check the transition against their own trie walk
//! without the core learning what is behind it.
//!
//! What this crate is not: it selects no canonical chain, downloads no blocks, and authenticates
//! no checkpoint. It validates the block it is handed against the caches it is handed.
//!
//! `scripts/check_validator_isolation.sh` in `partial-stateless-exex` enforces both invariants the
//! prose above claims — the forbidden dependency set, and the keccak build profile that makes any
//! timing taken from this package describe the binary production runs.

pub mod admission;
pub mod coordination;
pub mod reexec;
pub mod timings;

pub use admission::{AdmissionError, AdmittedBlock, UntrustedAdmission};
pub use coordination::{
    admit_block, block_context, inject_recovery, try_depth_one_recovery, BlockAdmission,
    CanonicalStateRoots, CoordinatedFingerprint, CoordinatedPair, LifecycleFingerprint,
    RetainedGeneration, RetainedGenerationBytes,
};
pub use reexec::{
    verify_and_apply_sidecar, verify_and_apply_sidecar_with_oracle, NoRootOracle,
    PostStateRootOracle, SidecarReexecLimits, SidecarValidationOutcome, TimedValidation,
    TrieCacheDisposition, ValidatorRules, POST_EXECUTION_REJECTION,
};
pub use timings::{AdmissionSource, AdmissionTimings, PayloadProvenance, ValidationPhaseTimings};
