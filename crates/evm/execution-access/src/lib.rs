//! Neutral block-execution access artifacts and the bounded handoff that carries them.
//!
//! A block's *access set* -- every account, storage slot and bytecode the block read or wrote,
//! plus the ancestor range it asked for through `BLOCKHASH` -- is strictly larger than its
//! execution diff. It includes read-only accounts, code reads, and reads made by calls that
//! later reverted. Consumers that maintain access-keyed state therefore cannot reconstruct it
//! from a post-state diff and have historically re-executed the block to obtain it.
//!
//! This crate exists so that the node's own execution can hand that set over instead:
//!
//! - [`ExecutedBlockAccess::from_state`] is the *single* extraction function. Both the engine's
//!   live execution and any re-execution path call it, so the two cannot drift in what they mean by
//!   "accessed".
//! - [`BlockAccessHandoff`] is a bounded, block-hash-keyed store. Insertion never blocks the
//!   producer and consumption is a take by exact hash; anything a consumer misses is expected to
//!   fall back to re-execution.
//!
//! Sharing the extractor proves that the same [`State`](revm_database::State) yields the same
//! access set. It does not prove that two execution paths built the same `State` -- prewarming,
//! cross-block caches, and pre-execution changes all remain the consumer's to validate.

#![doc(
    html_logo_url = "https://raw.githubusercontent.com/paradigmxyz/reth/main/assets/reth-docs.png",
    html_favicon_url = "https://avatars0.githubusercontent.com/u/97369466?s=256",
    issue_tracker_base_url = "https://github.com/paradigmxyz/reth/issues/"
)]
#![cfg_attr(not(test), warn(unused_crate_dependencies))]
#![cfg_attr(docsrs, feature(doc_cfg))]

mod access;
pub use access::{AccountAccess, ExecutedBlockAccess};

mod handoff;
pub use handoff::{
    global_handoff, AccessCaptureMode, BlockAccessArtifact, BlockAccessHandoff, HandoffStats,
    MissReason, TakeOutcome, DEFAULT_HANDOFF_CAPACITY, DEFAULT_HANDOFF_MAX_BYTES,
};
