//! Bounded, block-hash-keyed handoff from the executing node to an out-of-band consumer.

use crate::ExecutedBlockAccess;
use alloy_primitives::B256;
use std::{
    any::Any,
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex, OnceLock, TryLockError,
    },
    time::Instant,
};
use tracing::debug;

/// Blocks retained by the global handoff before the oldest insert is evicted.
///
/// A consumer that keeps up takes each artifact one notification after it is produced, so
/// residence is normally a single block. The depth exists to absorb a brief lag and to hold the
/// occasional sibling, not to buffer a consumer that has fallen behind: falling behind is
/// supposed to surface as a miss, not as unbounded memory.
pub const DEFAULT_HANDOFF_CAPACITY: usize = 4;

/// Access-set bytes retained by the global handoff before the oldest insert is evicted.
pub const DEFAULT_HANDOFF_MAX_BYTES: usize = 256 * 1024 * 1024;

/// Environment variable selecting the capture mode.
const CAPTURE_MODE_VAR: &str = "PS_ENGINE_ACCESS";

/// Returns the global handoff, or `None` when capture is off.
///
/// The store is allocated on first use, so a node running in [`AccessCaptureMode::Off`] pays one
/// relaxed load and nothing else. That matters because the un-captured path is the baseline every
/// captured measurement is compared against.
pub fn global_handoff() -> Option<&'static BlockAccessHandoff> {
    if !AccessCaptureMode::current().is_enabled() {
        return None
    }
    static HANDOFF: OnceLock<BlockAccessHandoff> = OnceLock::new();
    Some(HANDOFF.get_or_init(|| {
        BlockAccessHandoff::new(DEFAULT_HANDOFF_CAPACITY, DEFAULT_HANDOFF_MAX_BYTES)
    }))
}

/// How the node should treat execution-access capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessCaptureMode {
    /// No capture. The default, and the mode every baseline measurement must run in.
    Off,
    /// Capture and publish, but the consumer is expected to keep re-executing and compare.
    Shadow,
    /// Capture and publish, and the consumer may rely on the artifact.
    On,
}

impl AccessCaptureMode {
    /// Reads the mode from the environment once per process.
    pub fn current() -> Self {
        static MODE: OnceLock<AccessCaptureMode> = OnceLock::new();
        *MODE.get_or_init(|| {
            let mode = Self::parse(std::env::var(CAPTURE_MODE_VAR).ok().as_deref());
            if mode.is_enabled() {
                debug!(target: "execution_access", ?mode, "Execution access capture enabled");
            }
            mode
        })
    }

    /// Parses a mode from a raw environment value.
    ///
    /// Anything unrecognised is [`Self::Off`]: a typo must not silently enable capture on a run
    /// that was meant to be a baseline.
    pub fn parse(value: Option<&str>) -> Self {
        match value.map(str::trim) {
            Some("shadow" | "SHADOW") => Self::Shadow,
            Some("on" | "ON" | "1" | "true" | "TRUE" | "yes") => Self::On,
            _ => Self::Off,
        }
    }

    /// Whether artifacts should be captured and published at all.
    pub const fn is_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }

    /// Whether a consumer may skip its own re-execution when an artifact is present.
    pub const fn is_authoritative(self) -> bool {
        matches!(self, Self::On)
    }
}

/// One block's execution access set plus the execution output that produced it.
#[derive(Debug)]
pub struct BlockAccessArtifact {
    /// Height of the captured block.
    pub block_number: u64,
    /// Hash of the captured block; the handoff key.
    pub block_hash: B256,
    /// Parent hash, so a consumer can bind the artifact to the branch it expects.
    pub parent_hash: B256,
    /// Everything the block read or wrote.
    pub access: ExecutedBlockAccess,
    /// Wall time the capture itself cost the producer.
    ///
    /// Producers that time their execution must subtract this, or the capture lands inside a
    /// measurement boundary that was defined without it.
    pub capture_us: u64,
    /// Rough heap footprint of [`Self::access`], used to bound the store.
    pub approx_bytes: usize,
    captured_at: Instant,
    output: Arc<dyn Any + Send + Sync>,
}

impl BlockAccessArtifact {
    /// Builds an artifact around an already-shared execution output.
    ///
    /// The output is taken as an `Arc` because the producer already holds one: cloning it costs
    /// no additional resident bytes for as long as the node's own chain state keeps the block.
    pub fn new<T: Any + Send + Sync>(
        block_number: u64,
        block_hash: B256,
        parent_hash: B256,
        access: ExecutedBlockAccess,
        output: Arc<T>,
        capture_us: u64,
    ) -> Self {
        let approx_bytes = access.approx_heap_bytes();
        Self {
            block_number,
            block_hash,
            parent_hash,
            access,
            capture_us,
            approx_bytes,
            captured_at: Instant::now(),
            output,
        }
    }

    /// Returns the execution output if it has the expected type.
    ///
    /// The store is type-erased so that one global can serve a producer generic over its
    /// primitives; the consumer names the concrete type it executes with.
    pub fn output<T: Any + Send + Sync>(&self) -> Option<Arc<T>> {
        Arc::clone(&self.output).downcast::<T>().ok()
    }

    /// How long this artifact has been waiting to be consumed.
    pub fn residence(&self) -> std::time::Duration {
        self.captured_at.elapsed()
    }
}

/// A bounded map of artifacts keyed by block hash.
///
/// The policy is deliberately best-effort. Insertion never waits on a consumer and never waits
/// on the lock; a full store evicts its oldest insert. Both of those turn into a miss, and a miss
/// is expected to fall back to re-execution rather than to stall anything.
///
/// Eviction is by insertion order and never by canonical height. A sibling at the current tip may
/// still win a reorg, and discarding it because a competing block was committed would remove the
/// artifact precisely when it is about to be needed.
#[derive(Debug)]
pub struct BlockAccessHandoff {
    capacity: usize,
    max_bytes: usize,
    inner: Mutex<HandoffInner>,
    metrics: HandoffMetrics,
}

impl BlockAccessHandoff {
    /// Creates a store bounded by both an artifact count and an access-set byte budget.
    pub fn new(capacity: usize, max_bytes: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            max_bytes,
            inner: Mutex::new(HandoffInner::default()),
            metrics: HandoffMetrics::default(),
        }
    }

    /// Publishes an artifact, evicting older inserts if the store is at a bound.
    ///
    /// Returns `false` if the artifact was dropped. This never blocks: a contended lock drops
    /// rather than waits, because the producer here is a node's block-validation path and the
    /// consumer is not something it should ever be scheduled behind.
    pub fn insert(&self, artifact: BlockAccessArtifact) -> bool {
        let mut inner = match self.inner.try_lock() {
            Ok(inner) => inner,
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(TryLockError::WouldBlock) => {
                self.metrics.dropped_contended.fetch_add(1, Ordering::Relaxed);
                return false
            }
        };

        // Re-executing a block replaces its artifact rather than accumulating one per attempt.
        if inner.remove(&artifact.block_hash).is_some() {
            self.metrics.replaced.fetch_add(1, Ordering::Relaxed);
        }

        while inner.entries.len() >= self.capacity ||
            (!inner.entries.is_empty() && inner.bytes + artifact.approx_bytes > self.max_bytes)
        {
            let Some(oldest) = inner.order.front().copied() else { break };
            inner.remove(&oldest);
            self.metrics.dropped_capacity.fetch_add(1, Ordering::Relaxed);
        }

        inner.insert(artifact);
        self.metrics.inserted.fetch_add(1, Ordering::Relaxed);
        self.metrics.resident_bytes.store(inner.bytes, Ordering::Relaxed);
        self.metrics.queue_depth.store(inner.entries.len(), Ordering::Relaxed);
        true
    }

    /// Removes and returns the artifact for `block_hash`, if one is present.
    ///
    /// Lookup is by exact hash and never by height, so a consumer can never be handed a sibling's
    /// execution by accident.
    pub fn take(&self, block_hash: &B256) -> Option<BlockAccessArtifact> {
        let mut inner = match self.inner.lock() {
            Ok(inner) => inner,
            Err(poisoned) => poisoned.into_inner(),
        };
        let depth = inner.entries.len();
        let artifact = inner.remove(block_hash);

        match &artifact {
            Some(artifact) => {
                self.metrics.taken.fetch_add(1, Ordering::Relaxed);
                self.metrics
                    .residence_us_total
                    .fetch_add(artifact.residence().as_micros() as u64, Ordering::Relaxed);
                self.metrics.depth_at_take_total.fetch_add(depth as u64, Ordering::Relaxed);
            }
            None => {
                self.metrics.missed.fetch_add(1, Ordering::Relaxed);
            }
        }

        self.metrics.resident_bytes.store(inner.bytes, Ordering::Relaxed);
        self.metrics.queue_depth.store(inner.entries.len(), Ordering::Relaxed);
        artifact
    }

    /// Current telemetry snapshot.
    pub fn stats(&self) -> HandoffStats {
        let taken = self.metrics.taken.load(Ordering::Relaxed);
        HandoffStats {
            inserted: self.metrics.inserted.load(Ordering::Relaxed),
            taken,
            missed: self.metrics.missed.load(Ordering::Relaxed),
            dropped_capacity: self.metrics.dropped_capacity.load(Ordering::Relaxed),
            dropped_contended: self.metrics.dropped_contended.load(Ordering::Relaxed),
            replaced: self.metrics.replaced.load(Ordering::Relaxed),
            queue_depth: self.metrics.queue_depth.load(Ordering::Relaxed),
            resident_bytes: self.metrics.resident_bytes.load(Ordering::Relaxed),
            mean_residence_us: (taken > 0)
                .then(|| self.metrics.residence_us_total.load(Ordering::Relaxed) / taken),
            mean_depth_at_take: (taken > 0).then(|| {
                self.metrics.depth_at_take_total.load(Ordering::Relaxed) as f64 / taken as f64
            }),
        }
    }
}

/// A snapshot of handoff telemetry.
///
/// `missed` and `dropped_capacity` are gates rather than diagnostics: a store that keeps missing
/// in normal operation means the consumer is still re-executing every block, which is the state
/// this handoff exists to leave.
#[derive(Debug, Clone, PartialEq)]
pub struct HandoffStats {
    /// Artifacts published.
    pub inserted: u64,
    /// Artifacts consumed by hash.
    pub taken: u64,
    /// Lookups that found nothing.
    pub missed: u64,
    /// Artifacts evicted to stay within the count or byte bound.
    pub dropped_capacity: u64,
    /// Artifacts dropped because the store was locked.
    pub dropped_contended: u64,
    /// Artifacts overwritten by a re-execution of the same block.
    pub replaced: u64,
    /// Artifacts currently resident.
    pub queue_depth: usize,
    /// Access-set bytes currently resident.
    pub resident_bytes: usize,
    /// Mean wall time an artifact waited before being consumed.
    pub mean_residence_us: Option<u64>,
    /// Mean store depth observed at consumption.
    pub mean_depth_at_take: Option<f64>,
}

#[derive(Debug, Default)]
struct HandoffInner {
    order: VecDeque<B256>,
    entries: HashMap<B256, BlockAccessArtifact>,
    bytes: usize,
}

impl HandoffInner {
    fn insert(&mut self, artifact: BlockAccessArtifact) {
        self.bytes += artifact.approx_bytes;
        self.order.push_back(artifact.block_hash);
        self.entries.insert(artifact.block_hash, artifact);
    }

    fn remove(&mut self, block_hash: &B256) -> Option<BlockAccessArtifact> {
        let artifact = self.entries.remove(block_hash)?;
        self.bytes = self.bytes.saturating_sub(artifact.approx_bytes);
        if let Some(position) = self.order.iter().position(|hash| hash == block_hash) {
            self.order.remove(position);
        }
        Some(artifact)
    }
}

#[derive(Debug, Default)]
struct HandoffMetrics {
    inserted: AtomicU64,
    taken: AtomicU64,
    missed: AtomicU64,
    dropped_capacity: AtomicU64,
    dropped_contended: AtomicU64,
    replaced: AtomicU64,
    residence_us_total: AtomicU64,
    depth_at_take_total: AtomicU64,
    queue_depth: AtomicUsize,
    resident_bytes: AtomicUsize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AccountAccess;
    use alloy_primitives::{Address, U256};

    fn hash(tag: u8) -> B256 {
        B256::with_last_byte(tag)
    }

    fn artifact(number: u64, tag: u8) -> BlockAccessArtifact {
        artifact_with_accounts(number, tag, 0)
    }

    fn artifact_with_accounts(number: u64, tag: u8, accounts: u8) -> BlockAccessArtifact {
        let mut access = ExecutedBlockAccess::default();
        for index in 0..accounts {
            access.accounts.insert(
                Address::with_last_byte(index),
                AccountAccess { nonce: 0, balance: U256::ZERO, code_hash: None },
            );
        }
        BlockAccessArtifact::new(
            number,
            hash(tag),
            hash(tag.wrapping_sub(1)),
            access,
            Arc::new(number),
            0,
        )
    }

    #[test]
    fn an_artifact_is_returned_to_the_hash_that_produced_it() {
        let handoff = BlockAccessHandoff::new(4, usize::MAX);
        assert!(handoff.insert(artifact(10, 1)));

        let taken = handoff.take(&hash(1)).expect("inserted artifact is retrievable");
        assert_eq!(taken.block_number, 10);
        assert_eq!(*taken.output::<u64>().expect("output keeps its type"), 10);

        // Consumption removes it, so a second take is a miss rather than a stale hit.
        assert!(handoff.take(&hash(1)).is_none());
        let stats = handoff.stats();
        assert_eq!((stats.inserted, stats.taken, stats.missed), (1, 1, 1));
        assert_eq!(stats.queue_depth, 0);
    }

    #[test]
    fn a_sibling_hash_is_never_served_for_the_block_that_was_asked_for() {
        let handoff = BlockAccessHandoff::new(4, usize::MAX);
        handoff.insert(artifact(10, 1));

        // Same height, different branch.
        assert!(handoff.take(&hash(2)).is_none());
        assert!(handoff.take(&hash(1)).is_some());
    }

    #[test]
    fn the_oldest_insert_is_evicted_and_never_the_lowest_height() {
        let handoff = BlockAccessHandoff::new(2, usize::MAX);
        // A late-arriving low block, then two higher ones. Insertion order, not height, decides.
        handoff.insert(artifact(50, 1));
        handoff.insert(artifact(10, 2));
        handoff.insert(artifact(11, 3));

        assert!(handoff.take(&hash(1)).is_none(), "first insert is the one evicted");
        assert!(handoff.take(&hash(2)).is_some());
        assert!(handoff.take(&hash(3)).is_some());
        assert_eq!(handoff.stats().dropped_capacity, 1);
    }

    #[test]
    fn the_byte_bound_evicts_before_the_count_bound_is_reached() {
        let one = artifact_with_accounts(10, 1, 8).approx_bytes;
        let handoff = BlockAccessHandoff::new(16, one + one / 2);

        handoff.insert(artifact_with_accounts(10, 1, 8));
        handoff.insert(artifact_with_accounts(11, 2, 8));

        assert!(handoff.take(&hash(1)).is_none());
        assert!(handoff.take(&hash(2)).is_some());
        assert_eq!(handoff.stats().dropped_capacity, 1);
        assert_eq!(handoff.stats().resident_bytes, 0);
    }

    #[test]
    fn re_executing_a_block_replaces_its_artifact_rather_than_accumulating_one() {
        let handoff = BlockAccessHandoff::new(4, usize::MAX);
        handoff.insert(artifact_with_accounts(10, 1, 2));
        handoff.insert(artifact_with_accounts(10, 1, 5));

        assert_eq!(handoff.stats().queue_depth, 1);
        assert_eq!(handoff.stats().replaced, 1);
        let taken = handoff.take(&hash(1)).expect("the replacement is the one retained");
        assert_eq!(taken.access.accounts.len(), 5);
        assert_eq!(handoff.stats().resident_bytes, 0);
    }

    #[test]
    fn a_single_artifact_larger_than_the_byte_bound_is_still_served_once() {
        // Otherwise the store would evict the entry it just made room for and always miss.
        let handoff = BlockAccessHandoff::new(4, 1);
        assert!(handoff.insert(artifact_with_accounts(10, 1, 8)));
        assert!(handoff.take(&hash(1)).is_some());
    }

    #[test]
    fn an_unrecognised_mode_is_off() {
        assert_eq!(AccessCaptureMode::parse(None), AccessCaptureMode::Off);
        assert_eq!(AccessCaptureMode::parse(Some("")), AccessCaptureMode::Off);
        assert_eq!(AccessCaptureMode::parse(Some("shadwo")), AccessCaptureMode::Off);
        assert_eq!(AccessCaptureMode::parse(Some(" shadow ")), AccessCaptureMode::Shadow);
        assert_eq!(AccessCaptureMode::parse(Some("on")), AccessCaptureMode::On);

        assert!(!AccessCaptureMode::Off.is_enabled());
        assert!(AccessCaptureMode::Shadow.is_enabled());
        assert!(!AccessCaptureMode::Shadow.is_authoritative());
        assert!(AccessCaptureMode::On.is_authoritative());
    }
}
