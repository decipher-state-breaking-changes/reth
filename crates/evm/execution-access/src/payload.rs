//! Bounded, block-hash-keyed handoff of the Engine-API payload the node was actually handed.
//!
//! A consumer that wants to *reproduce* a validator's verdict on untrusted input needs the input,
//! and a block is not it. Rebuilding an [`ExecutionData`](alloy_rpc_types_engine::ExecutionData)
//! from a block is not the inverse of receiving one: the conversion derives the block hash from
//! the header it is given, derives the blob versioned hashes from the transactions it is given,
//! and carries the requests as a hash rather than as the requests. A validator handed such a
//! reconstruction passes its payload-layout, block-hash, and versioned-hash checks *vacuously* --
//! it is checking the producer's arithmetic against itself. Only the payload the consensus layer
//! sent can make those checks mean anything, and this handoff is how it reaches a consumer that
//! is not the Engine.
//!
//! The policy is [`BlockAccessHandoff`](crate::BlockAccessHandoff)'s, deliberately and by reuse:
//! insertion never blocks the producer, eviction is by insertion order and never by canonical
//! height, and a consumer that misses is expected to say so rather than to stall anything. What
//! differs is only what a miss *means*. An access-set miss costs the consumer a re-execution; a
//! payload miss costs it the distinction between a witnessed input and a derived one, which is a
//! provenance fact it must record rather than a cost it can pay.

use crate::handoff::{env_override, BoundedHandoff, HandoffEntry};
use alloy_primitives::B256;
use std::{
    any::Any,
    sync::{Arc, OnceLock},
    time::Instant,
};
use tracing::debug;

/// Payloads retained by the global handoff before the oldest insert is evicted.
///
/// Matches the access handoff's depth for the same reason: a consumer that keeps up takes each
/// payload one notification after it is produced, so residence is normally a single block, and the
/// depth exists to absorb a brief lag and the occasional sibling.
pub const DEFAULT_PAYLOAD_HANDOFF_CAPACITY: usize = 4;

/// Payload bytes retained by the global handoff before the oldest insert is evicted.
///
/// Far smaller than the access handoff's budget, and it can be, because a payload's size is
/// already bounded by something this node enforces: the Engine accepted this payload, so its
/// transactions passed whatever block-size rule the active fork applies. The budget is a second
/// bound rather than the only one -- [`DEFAULT_PAYLOAD_HANDOFF_CAPACITY`] artifacts is what is
/// actually guaranteed, and an artifact larger than the budget is still inserted for the same
/// reason it is in the access store.
pub const DEFAULT_PAYLOAD_HANDOFF_MAX_BYTES: usize = 64 * 1024 * 1024;

/// Environment variable enabling payload capture.
const PAYLOAD_CAPTURE_VAR: &str = "PS_ENGINE_PAYLOAD";

/// Overrides [`DEFAULT_PAYLOAD_HANDOFF_CAPACITY`].
const CAPACITY_VAR: &str = "PS_PAYLOAD_HANDOFF_CAPACITY";

/// Overrides [`DEFAULT_PAYLOAD_HANDOFF_MAX_BYTES`], in bytes.
const MAX_BYTES_VAR: &str = "PS_PAYLOAD_HANDOFF_MAX_BYTES";

/// The Engine-payload store, bounded exactly as the access store is.
pub type EnginePayloadHandoff = BoundedHandoff<EnginePayloadArtifact>;

/// Whether the node should publish the payloads it validates.
///
/// Read once per process. Anything unrecognised is off, so a typo cannot silently turn a baseline
/// run into a capturing one -- the same rule
/// [`AccessCaptureMode::parse`](crate::AccessCaptureMode::parse) follows.
pub fn payload_capture_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let enabled = matches!(
            std::env::var(PAYLOAD_CAPTURE_VAR).ok().as_deref().map(str::trim),
            Some("on" | "ON" | "1" | "true" | "TRUE" | "yes")
        );
        if enabled {
            debug!(target: "execution_access", "Engine payload capture enabled");
        }
        enabled
    })
}

/// Returns the global payload handoff, or `None` when capture is off.
///
/// Allocated on first use, so a node running without capture pays one relaxed load. That matters
/// for the same reason it does for the access handoff: the un-captured path is the baseline every
/// captured measurement is compared against.
pub fn global_payload_handoff() -> Option<&'static EnginePayloadHandoff> {
    if !payload_capture_enabled() {
        return None
    }
    static HANDOFF: OnceLock<EnginePayloadHandoff> = OnceLock::new();
    Some(HANDOFF.get_or_init(|| {
        EnginePayloadHandoff::new(
            env_override(CAPACITY_VAR, DEFAULT_PAYLOAD_HANDOFF_CAPACITY),
            env_override(MAX_BYTES_VAR, DEFAULT_PAYLOAD_HANDOFF_MAX_BYTES),
        )
    }))
}

/// One Engine-API payload, as the node received it, keyed by the hash it announced.
///
/// Type-erased for the same reason the access artifact's execution output is: one global store
/// serves a producer generic over its payload type, and the consumer names the concrete type it
/// validates. A consumer whose downcast fails is configured for a different chain's payload type
/// and must treat that as a miss, not as an absence.
#[derive(Debug)]
pub struct EnginePayloadArtifact {
    /// Height of the captured block.
    pub block_number: u64,
    /// Hash the payload announced; the handoff key.
    ///
    /// This is the *announced* hash rather than a recomputed one, which is the point: a consumer
    /// checking the payload against it is repeating a check the Engine already made, which is
    /// exactly what a reconstruction cannot offer.
    pub block_hash: B256,
    /// Parent hash, so a consumer can bind the payload to the branch it expects.
    pub parent_hash: B256,
    /// Serialized size of the payload's transactions, used to bound the store.
    pub approx_bytes: usize,
    captured_at: Instant,
    payload: Arc<dyn Any + Send + Sync>,
}

impl EnginePayloadArtifact {
    /// Wraps one payload the node was handed.
    ///
    /// `approx_bytes` is the producer's own measure of what the payload weighs. It is passed in
    /// rather than derived here because the payload is opaque at this boundary, and a store that
    /// guessed would be reporting a number no producer stands behind.
    pub fn new<T: Any + Send + Sync>(
        block_number: u64,
        block_hash: B256,
        parent_hash: B256,
        payload: T,
        approx_bytes: usize,
    ) -> Self {
        Self {
            block_number,
            block_hash,
            parent_hash,
            approx_bytes,
            captured_at: Instant::now(),
            payload: Arc::new(payload),
        }
    }

    /// Returns the payload if it has the expected type.
    pub fn payload<T: Any + Send + Sync>(&self) -> Option<Arc<T>> {
        Arc::clone(&self.payload).downcast::<T>().ok()
    }
}

impl HandoffEntry for EnginePayloadArtifact {
    fn block_hash(&self) -> B256 {
        self.block_hash
    }

    fn approx_bytes(&self) -> usize {
        self.approx_bytes
    }

    fn residence(&self) -> std::time::Duration {
        self.captured_at.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MissReason;

    fn hash(tag: u8) -> B256 {
        B256::with_last_byte(tag)
    }

    fn artifact(number: u64, tag: u8, bytes: usize) -> EnginePayloadArtifact {
        EnginePayloadArtifact::new(
            number,
            hash(tag),
            hash(tag.wrapping_sub(1)),
            format!("payload-{number}"),
            bytes,
        )
    }

    #[test]
    fn a_payload_is_returned_to_the_hash_that_announced_it() {
        let handoff = EnginePayloadHandoff::new(4, usize::MAX);
        assert!(handoff.insert(artifact(10, 1, 0)));

        let taken = handoff.take(&hash(1)).expect("inserted payload is retrievable");
        assert_eq!(taken.block_number, 10);
        assert_eq!(*taken.payload::<String>().expect("payload keeps its type"), "payload-10");

        // Consumption removes it, so a second take is a miss rather than a stale hit.
        assert!(handoff.take(&hash(1)).is_none());
    }

    /// A consumer configured for another chain's payload type must see a miss, not a panic.
    #[test]
    fn a_payload_of_another_type_is_not_handed_over() {
        let handoff = EnginePayloadHandoff::new(4, usize::MAX);
        handoff.insert(artifact(10, 1, 0));
        let taken = handoff.take(&hash(1)).expect("inserted payload is retrievable");
        assert!(taken.payload::<u64>().is_none());
    }

    #[test]
    fn the_oldest_insert_is_the_one_evicted_and_it_says_so() {
        let handoff = EnginePayloadHandoff::new(2, usize::MAX);
        handoff.insert(artifact(10, 1, 0));
        handoff.insert(artifact(11, 2, 0));
        handoff.insert(artifact(12, 3, 0));

        assert_eq!(handoff.take_outcome(&hash(1)).miss_reason(), Some(MissReason::EvictedCapacity));
        assert!(handoff.take(&hash(2)).is_some());
        assert!(handoff.take(&hash(3)).is_some());
    }

    /// The byte budget is a second bound and never the only one, so the last payload standing is
    /// kept even when it exceeds the budget on its own.
    #[test]
    fn a_payload_larger_than_the_budget_is_still_delivered() {
        let handoff = EnginePayloadHandoff::new(4, 1_024);
        handoff.insert(artifact(10, 1, 4_096));
        assert!(handoff.take(&hash(1)).is_some());
    }

    #[test]
    fn the_byte_budget_evicts_before_the_count_does() {
        let handoff = EnginePayloadHandoff::new(4, 1_024);
        handoff.insert(artifact(10, 1, 768));
        handoff.insert(artifact(11, 2, 768));

        assert_eq!(handoff.take_outcome(&hash(1)).miss_reason(), Some(MissReason::EvictedBytes));
        assert!(handoff.take(&hash(2)).is_some());
        let stats = handoff.stats();
        assert_eq!(stats.inserted, 2);
        assert_eq!(stats.dropped_capacity, 1);
    }

    /// Capture is off unless it is switched on by an exact value, so a typo cannot arm a run that
    /// was meant to be a baseline.
    #[test]
    fn capture_is_off_unless_explicitly_enabled() {
        // Reads the process environment once, and this test suite never sets the variable.
        assert!(!payload_capture_enabled());
        assert!(global_payload_handoff().is_none());
    }
}
