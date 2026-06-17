//! Shared contracts for partial-stateless sidecar producers.
//!
//! This crate intentionally avoids owning a concrete cache policy. Producers and
//! consumers agree through [`CacheView`] and [`CacheDescriptor`], so an external
//! cache implementation can be plugged in without changing the sidecar pipeline.

mod cache_descriptor;
mod cache_view;
mod sidecar;
mod target_filter;
mod witness_targets;

pub use cache_descriptor::CacheDescriptor;
pub use cache_view::{AllHitDebugView, CacheView, NoCacheView};
pub use sidecar::{
    EncodedCodePreimage, EncodedProofNode, EncodedStorageMultiproof, PartialExecutionWitness,
    PartialExecutionWitnessSidecar, SidecarBlockContext, StateMultiproofPayload,
    StateMultiproofPayloadStats, WitnessPayload,
};
pub use target_filter::{filter_missing_targets, FilteredTargets, TargetStats};
pub use witness_targets::{StorageTarget, TargetCounts, TargetSourceKind, WitnessTargets};
