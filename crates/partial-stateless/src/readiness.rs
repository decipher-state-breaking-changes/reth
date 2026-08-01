//! Readiness contract for the joint value + sparse-trie cache.
//!
//! [`NetworkStateCache`] and [`PartialTrieNodeCache`] are only a sound substitute for full state
//! when together they represent *exactly* the parent of the block being validated. Neither cache
//! records whether that holds: one block past a cold reset a cache is structurally
//! indistinguishable from a fully warmed one, because both simply contain whatever the last block
//! touched. This module carries the bookkeeping that tells them apart.
//!
//! Three independent conditions must hold before the caches may stand in for state at a parent
//! block, and each fails in a different way:
//!
//! 1. **Authenticated trie** — the sparse trie's root equals the parent's canonical state root.
//!    Without it the trie is a pile of nodes with no binding to any chain.
//! 2. **Exact parent fingerprint** — the cache sits at the parent's number *and* hash. Height alone
//!    cannot distinguish two competing chains at the same height.
//! 3. **Filled window** — enough blocks have been replayed to actually populate the LastN window
//!    the cache policy advertises. A cache one block past a reset satisfies conditions 1 and 2
//!    while holding a fraction of the entries its policy identifier claims.
//!
//! A cache holding no accounts and no storage is *not* excluded: restoring from a snapshot is the
//! normal way a cold node starts, and such a snapshot still carries an authenticated state root.
//! Emptiness is not the criterion; an unauthenticated root is.

use crate::{
    network_cache::NetworkStateCache, sidecar::CacheAnchor, trie_cache::PartialTrieNodeCache,
};
use alloy_primitives::B256;

/// Tracks whether the joint cache may be used to validate the next block.
///
/// The tracker observes block application; it never mutates the caches. Callers report what they
/// are about to do ([`begin_block`](Self::begin_block)) and what the caches then reported
/// ([`finish_block`](Self::finish_block)), and read the resulting classification back.
#[derive(Debug, Clone)]
pub struct CacheReadinessTracker {
    state: CacheReadiness,
    /// Blocks that must be replayed after a cold reset before the advertised LastN window is
    /// actually populated. Callers running separate account and storage windows pass the larger of
    /// the two: the window is only whole once both are.
    window_size: u64,
    /// Cleared by every cold reset. Set by replaying `window_size` blocks, or in one shot by
    /// restoring a snapshot that already carries a full window.
    window_filled: bool,
    replay_depth: u64,
    last_applied: Option<AppliedBlock>,
    cache_policy_id: B256,
}

impl CacheReadinessTracker {
    /// Creates a tracker for a cold cache.
    pub const fn new(window_size: u64, cache_policy_id: B256) -> Self {
        Self {
            state: CacheReadiness::Cold,
            window_size,
            window_filled: false,
            replay_depth: 0,
            last_applied: None,
            cache_policy_id,
        }
    }

    /// Current classification.
    pub const fn state(&self) -> &CacheReadiness {
        &self.state
    }

    /// The parent the caches may currently validate against, if any.
    pub fn ready_parent(&self) -> Option<&ReadyParent> {
        match &self.state {
            CacheReadiness::Ready(parent) => Some(parent),
            _ => None,
        }
    }

    /// Contiguous blocks applied since the last cold reset.
    ///
    /// Stays 0 across a snapshot restore: nothing was replayed, the window arrived whole.
    pub const fn replay_depth(&self) -> u64 {
        self.replay_depth
    }

    /// Whether the cache holds the full window its policy identifier advertises.
    pub const fn window_filled(&self) -> bool {
        self.window_filled
    }

    /// Whether progress has stopped pending operator action.
    pub const fn is_blocked(&self) -> bool {
        matches!(self.state, CacheReadiness::Blocked { .. })
    }

    /// Whether processed height may be acknowledged to the host node.
    ///
    /// Acknowledging a height is a durable promise that every block up to it was processed. Once
    /// blocked, the cache has *not* processed some block below the tip, so advancing the
    /// acknowledgement would make the gap permanent across restarts — the skipped block is never
    /// delivered again.
    pub const fn may_acknowledge_height(&self) -> bool {
        !self.is_blocked()
    }

    /// Drops all continuity claims, matching a cold reset of both caches.
    pub fn reset(&mut self) {
        self.state = CacheReadiness::Cold;
        self.window_filled = false;
        self.replay_depth = 0;
        self.last_applied = None;
    }

    /// Marks the caches as being unwound after a reorg or revert.
    ///
    /// Distinct from [`reset`](Self::reset) so that a reorg whose reset never completes cannot be
    /// mistaken for a clean cold start.
    pub fn begin_recovery(&mut self) {
        self.state = CacheReadiness::Recovering;
    }

    /// Promotes a snapshot-restored cache to [`CacheReadiness::Ready`] against an
    /// operator-supplied checkpoint.
    ///
    /// A snapshot bypasses the replay that would otherwise fill the window, so the height, hash and
    /// state root it claims are checked against a checkpoint the operator supplied out of band
    /// rather than against the snapshot's own assertions. A rejected snapshot leaves the tracker
    /// [`Cold`](CacheReadiness::Cold): warming from scratch is always sound.
    pub fn restore_from_checkpoint(
        &mut self,
        checkpoint: &TrustedCheckpoint,
        observed: &CacheObservation,
    ) -> Result<&ReadyParent, ReadinessError> {
        self.reset();

        if observed.cache_block != checkpoint.block_number {
            return Err(ReadinessError::CheckpointHeightMismatch {
                expected: checkpoint.block_number,
                actual: observed.cache_block,
            })
        }
        if observed.trie_state_root != Some(checkpoint.state_root) {
            return Err(ReadinessError::CheckpointRootMismatch {
                expected: checkpoint.state_root,
                actual: observed.trie_state_root,
            })
        }

        self.window_filled = true;
        self.last_applied =
            Some(AppliedBlock { number: checkpoint.block_number, hash: checkpoint.block_hash });
        self.state = CacheReadiness::Ready(ReadyParent {
            anchor: CacheAnchor {
                block_number: checkpoint.block_number,
                block_hash: checkpoint.block_hash,
                cache_policy_id: self.cache_policy_id,
                cache_root: observed.cache_root,
            },
            trie_state_root: checkpoint.state_root,
            replay_depth: 0,
        });

        Ok(self.ready_parent().expect("just set to Ready"))
    }

    /// Admits a block for application.
    ///
    /// Returns the reason on rejection, and leaves the tracker blocked on that reason. A rejected
    /// block must not be applied: every state the caches would reach from it is unsound.
    pub fn begin_block(&mut self, block: &BlockContext) -> Result<(), BlockedReason> {
        match &self.state {
            CacheReadiness::Blocked { reason } => return Err(self.block_on(reason.clone())),
            CacheReadiness::Recovering => {
                return Err(
                    self.block_on(BlockedReason::RecoveryIncomplete { block_number: block.number })
                )
            }
            // A block was admitted and never completed, so its state changes are unaccounted for.
            CacheReadiness::Applying { block_number } => {
                let block_number = *block_number;
                return Err(self.block_on(BlockedReason::BlockSkipped { block_number }))
            }
            CacheReadiness::Cold => {}
            CacheReadiness::Warming { .. } | CacheReadiness::Ready(_) => {
                let applied =
                    self.last_applied.as_ref().expect("warm state implies an applied block");
                if block.number != applied.number + 1 || block.parent_hash != applied.hash {
                    return Err(self.block_on(BlockedReason::BlockGap {
                        expected_number: applied.number + 1,
                        expected_parent: applied.hash,
                        actual_number: block.number,
                        actual_parent: block.parent_hash,
                    }))
                }
            }
        }

        self.state = CacheReadiness::Applying { block_number: block.number };
        Ok(())
    }

    /// Records the caches' condition after `block` was applied, and reclassifies.
    pub fn finish_block(
        &mut self,
        block: &BlockContext,
        observed: &CacheObservation,
    ) -> &CacheReadiness {
        if self.is_blocked() {
            return &self.state
        }

        if observed.cache_block != block.number {
            self.block_on(BlockedReason::CacheDrift {
                expected_block: block.number,
                cache_block: observed.cache_block,
            });
            return &self.state
        }

        self.replay_depth += 1;
        self.last_applied = Some(AppliedBlock { number: block.number, hash: block.hash });
        if self.replay_depth >= self.window_size {
            self.window_filled = true;
        }

        // The trie root proves the caches describe this block's post-state, which is precisely the
        // state the *next* block executes against.
        let authenticated = observed.trie_state_root == Some(block.state_root);
        self.state = if authenticated && self.window_filled {
            CacheReadiness::Ready(ReadyParent {
                anchor: CacheAnchor {
                    block_number: block.number,
                    block_hash: block.hash,
                    cache_policy_id: self.cache_policy_id,
                    cache_root: observed.cache_root,
                },
                trie_state_root: block.state_root,
                replay_depth: self.replay_depth,
            })
        } else {
            CacheReadiness::Warming { replay_depth: self.replay_depth }
        };

        &self.state
    }

    /// Records that a block could not be applied at all.
    ///
    /// Every later block builds on state this cache never saw, so this is terminal until a reset:
    /// skipping and carrying on is what silently produces a cache that lies about its parent.
    pub fn abandon_block(&mut self, block_number: u64) -> BlockedReason {
        self.block_on(BlockedReason::BlockSkipped { block_number })
    }

    /// Enters the blocked state, preserving the reason recorded first.
    fn block_on(&mut self, reason: BlockedReason) -> BlockedReason {
        if let CacheReadiness::Blocked { reason: existing } = &self.state {
            return existing.clone()
        }
        self.state = CacheReadiness::Blocked { reason: reason.clone() };
        reason
    }
}

/// Whether the joint cache may stand in for state, and if not, how far it is from doing so.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheReadiness {
    /// No authenticated trie root. Cannot validate any block.
    Cold,
    /// Contiguously replaying blocks; the advertised window is not yet populated.
    Warming {
        /// Contiguous blocks applied since the last cold reset.
        replay_depth: u64,
    },
    /// Safe to validate the direct child of the contained parent.
    Ready(ReadyParent),
    /// A block is mid-application; the caches are transiently inconsistent.
    Applying {
        /// The block being applied.
        block_number: u64,
    },
    /// Unwinding after a reorg or revert.
    Recovering,
    /// Cannot make progress without a reset.
    Blocked {
        /// What stopped progress.
        reason: BlockedReason,
    },
}

impl CacheReadiness {
    /// Stable name of the variant, for logs and metrics.
    ///
    /// Distinct from `Debug` because the payloads change every block — the anchor in `Ready` and
    /// the depth in `Warming` both advance — while the classification itself changes rarely. Only
    /// the latter is worth reporting.
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Cold => "cold",
            Self::Warming { .. } => "warming",
            Self::Ready(_) => "ready",
            Self::Applying { .. } => "applying",
            Self::Recovering => "recovering",
            Self::Blocked { .. } => "blocked",
        }
    }
}

/// The parent block the caches are authenticated against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadyParent {
    /// Binds the cache contents to this block, hash and policy.
    pub anchor: CacheAnchor,
    /// Sparse-trie root, equal to this block's canonical state root.
    pub trie_state_root: B256,
    /// Contiguous blocks replayed to reach this state; 0 for a snapshot restore.
    pub replay_depth: u64,
}

/// Why the cache stopped making progress.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockedReason {
    /// A block arrived that is not the direct child of the last block applied.
    BlockGap {
        /// Number the next block had to have.
        expected_number: u64,
        /// Hash the next block's parent had to be.
        expected_parent: B256,
        /// Number the block actually had.
        actual_number: u64,
        /// Parent the block actually named.
        actual_parent: B256,
    },
    /// A block was delivered but never applied.
    BlockSkipped {
        /// The block that was not applied.
        block_number: u64,
    },
    /// The value cache is not at the height of the block just applied.
    CacheDrift {
        /// Height the cache had to reach.
        expected_block: u64,
        /// Height it reported.
        cache_block: u64,
    },
    /// Blocks arrived before the post-reorg reset completed.
    RecoveryIncomplete {
        /// The block that arrived too early.
        block_number: u64,
    },
}

/// The block about to be applied, as the canonical chain describes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockContext {
    /// Block number.
    pub number: u64,
    /// Block hash.
    pub hash: B256,
    /// Parent hash.
    pub parent_hash: B256,
    /// This block's own post-state root.
    pub state_root: B256,
}

/// What the two caches report about themselves at a point in time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheObservation {
    /// Height the value cache believes it is at.
    pub cache_block: u64,
    /// Deterministic root over the value cache contents.
    pub cache_root: B256,
    /// Sparse-trie root, or `None` while the trie is unauthenticated.
    pub trie_state_root: Option<B256>,
}

impl CacheObservation {
    /// Reads the current condition of both caches.
    ///
    /// `cache_root` is memoized until the value cache is mutated, so calling this once per block
    /// costs nothing beyond what the sidecar anchor already computes.
    pub fn capture(values: &NetworkStateCache, trie: &PartialTrieNodeCache) -> Self {
        Self {
            cache_block: values.current_block(),
            cache_root: values.cache_root(),
            trie_state_root: trie.state_root(),
        }
    }
}

/// A block the operator vouches for out of band, used to accept a snapshot.
///
/// Snapshots are accepted on operator authority rather than on peer consensus: there is no scoring
/// or majority vote here, and a snapshot that disagrees with the checkpoint is simply discarded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrustedCheckpoint {
    /// Height the snapshot must be at.
    pub block_number: u64,
    /// Hash the snapshot must be at.
    pub block_hash: B256,
    /// Canonical state root the restored trie must reproduce.
    pub state_root: B256,
}

/// Why a snapshot could not be promoted to [`CacheReadiness::Ready`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadinessError {
    /// The restored cache sits at a different height than the checkpoint.
    CheckpointHeightMismatch {
        /// Checkpoint height.
        expected: u64,
        /// Restored height.
        actual: u64,
    },
    /// The restored trie does not reproduce the checkpoint's state root.
    CheckpointRootMismatch {
        /// Checkpoint state root.
        expected: B256,
        /// Restored trie root, or `None` when the trie was never authenticated.
        actual: Option<B256>,
    },
}

/// Last block applied since the most recent cold reset.
#[derive(Debug, Clone, Copy)]
struct AppliedBlock {
    number: u64,
    hash: B256,
}

#[cfg(test)]
mod tests {
    use super::*;

    const WINDOW: u64 = 4;

    fn policy_id() -> B256 {
        B256::repeat_byte(0xa0)
    }

    fn tracker() -> CacheReadinessTracker {
        CacheReadinessTracker::new(WINDOW, policy_id())
    }

    /// Synthesizes a chain where every block's hash and state root derive from its number, so a
    /// test can name any block without threading a fixture through.
    fn block(number: u64) -> BlockContext {
        BlockContext {
            number,
            hash: block_hash(number),
            parent_hash: block_hash(number - 1),
            state_root: state_root(number),
        }
    }

    fn block_hash(number: u64) -> B256 {
        let mut hash = B256::ZERO;
        hash[0..8].copy_from_slice(&number.to_be_bytes());
        hash[31] = 0xbb;
        hash
    }

    fn state_root(number: u64) -> B256 {
        let mut root = B256::ZERO;
        root[0..8].copy_from_slice(&number.to_be_bytes());
        root[31] = 0x55;
        root
    }

    /// The caches behaving correctly: at the block's height, with the block's authenticated root.
    fn observed(block: &BlockContext) -> CacheObservation {
        CacheObservation {
            cache_block: block.number,
            cache_root: B256::repeat_byte(0xcc),
            trie_state_root: Some(block.state_root),
        }
    }

    /// Applies `count` contiguous blocks starting at `from`, asserting each is admitted.
    fn apply_contiguous(tracker: &mut CacheReadinessTracker, from: u64, count: u64) {
        for number in from..from + count {
            let block = block(number);
            tracker.begin_block(&block).expect("contiguous block is admissible");
            tracker.finish_block(&block, &observed(&block));
        }
    }

    #[test]
    fn cold_reset_then_one_block_is_never_ready() {
        let mut tracker = tracker();
        apply_contiguous(&mut tracker, 100, 1);

        assert_eq!(*tracker.state(), CacheReadiness::Warming { replay_depth: 1 });
        assert!(tracker.ready_parent().is_none());
    }

    #[test]
    fn ready_only_after_the_window_is_replayed() {
        let mut tracker = tracker();
        apply_contiguous(&mut tracker, 100, WINDOW - 1);
        assert!(tracker.ready_parent().is_none(), "window is one block short");

        apply_contiguous(&mut tracker, 100 + WINDOW - 1, 1);

        let parent = tracker.ready_parent().expect("window filled");
        assert_eq!(parent.anchor.block_number, 100 + WINDOW - 1);
        assert_eq!(parent.anchor.block_hash, block_hash(100 + WINDOW - 1));
        assert_eq!(parent.anchor.cache_policy_id, policy_id());
        assert_eq!(parent.trie_state_root, state_root(100 + WINDOW - 1));
        assert_eq!(parent.replay_depth, WINDOW);
    }

    #[test]
    fn reset_drops_readiness_and_replay_depth() {
        let mut tracker = tracker();
        apply_contiguous(&mut tracker, 100, WINDOW);
        assert!(tracker.ready_parent().is_some());

        tracker.reset();

        assert_eq!(*tracker.state(), CacheReadiness::Cold);
        assert_eq!(tracker.replay_depth(), 0);
        assert!(!tracker.window_filled());

        apply_contiguous(&mut tracker, 200, 1);
        assert_eq!(*tracker.state(), CacheReadiness::Warming { replay_depth: 1 });
    }

    #[test]
    fn a_number_gap_blocks_instead_of_bumping_replay_depth() {
        let mut tracker = tracker();
        apply_contiguous(&mut tracker, 100, 2);

        let skipped = block(103);
        let reason = tracker.begin_block(&skipped).expect_err("103 is not the child of 101");

        assert_eq!(
            reason,
            BlockedReason::BlockGap {
                expected_number: 102,
                expected_parent: block_hash(101),
                actual_number: 103,
                actual_parent: block_hash(102),
            }
        );
        assert_eq!(tracker.replay_depth(), 2, "a rejected block does not count as replayed");
        assert!(tracker.is_blocked());
    }

    #[test]
    fn a_sibling_at_the_same_height_blocks() {
        let mut tracker = tracker();
        apply_contiguous(&mut tracker, 100, 2);

        // Correct height, but descends from a block on another branch.
        let sibling = BlockContext { parent_hash: B256::repeat_byte(0xfe), ..block(102) };
        let reason = tracker.begin_block(&sibling).expect_err("parent hash does not match");

        assert!(matches!(reason, BlockedReason::BlockGap { .. }));
        assert!(tracker.is_blocked());
    }

    #[test]
    fn blocked_stops_height_acknowledgement_until_reset() {
        let mut tracker = tracker();
        apply_contiguous(&mut tracker, 100, 1);
        assert!(tracker.may_acknowledge_height());

        tracker.abandon_block(101);

        assert!(!tracker.may_acknowledge_height());
        assert!(tracker.begin_block(&block(102)).is_err(), "blocked state is sticky");

        tracker.reset();
        assert!(tracker.may_acknowledge_height());
    }

    #[test]
    fn blocked_preserves_the_first_reason() {
        let mut tracker = tracker();
        apply_contiguous(&mut tracker, 100, 1);
        tracker.abandon_block(101);

        let reason = tracker.begin_block(&block(105)).expect_err("still blocked");

        assert_eq!(reason, BlockedReason::BlockSkipped { block_number: 101 });
    }

    #[test]
    fn an_unauthenticated_trie_never_reaches_ready() {
        let mut tracker = tracker();
        apply_contiguous(&mut tracker, 100, WINDOW);
        tracker.reset();

        for number in 100..100 + WINDOW + 2 {
            let block = block(number);
            tracker.begin_block(&block).expect("contiguous");
            tracker.finish_block(
                &block,
                &CacheObservation { trie_state_root: None, ..observed(&block) },
            );
        }

        assert!(tracker.ready_parent().is_none());
        assert!(tracker.window_filled(), "the window fills regardless of authentication");
        assert_eq!(*tracker.state(), CacheReadiness::Warming { replay_depth: WINDOW + 2 });
    }

    #[test]
    fn a_wrong_trie_root_never_reaches_ready() {
        let mut tracker = tracker();

        for number in 100..100 + WINDOW + 2 {
            let block = block(number);
            tracker.begin_block(&block).expect("contiguous");
            tracker.finish_block(
                &block,
                &CacheObservation {
                    trie_state_root: Some(B256::repeat_byte(0xad)),
                    ..observed(&block)
                },
            );
        }

        assert!(tracker.ready_parent().is_none());
    }

    #[test]
    fn a_drifted_value_cache_blocks() {
        let mut tracker = tracker();
        let block = block(100);
        tracker.begin_block(&block).expect("cold start admits any block");

        tracker.finish_block(&block, &CacheObservation { cache_block: 99, ..observed(&block) });

        assert_eq!(
            *tracker.state(),
            CacheReadiness::Blocked {
                reason: BlockedReason::CacheDrift { expected_block: 100, cache_block: 99 }
            }
        );
    }

    #[test]
    fn recovery_must_complete_before_blocks_resume() {
        let mut tracker = tracker();
        apply_contiguous(&mut tracker, 100, WINDOW);

        tracker.begin_recovery();
        assert_eq!(*tracker.state(), CacheReadiness::Recovering);
        assert!(tracker.ready_parent().is_none(), "a recovering cache validates nothing");

        let reason = tracker.begin_block(&block(200)).expect_err("reset has not run");
        assert_eq!(reason, BlockedReason::RecoveryIncomplete { block_number: 200 });

        tracker.reset();
        apply_contiguous(&mut tracker, 200, 1);
        assert_eq!(*tracker.state(), CacheReadiness::Warming { replay_depth: 1 });
    }

    #[test]
    fn an_abandoned_block_blocks_the_next_one() {
        let mut tracker = tracker();
        apply_contiguous(&mut tracker, 100, WINDOW);

        // Admitted, then never completed — the caches saw part of a block, or none of it.
        tracker.begin_block(&block(100 + WINDOW)).expect("contiguous");
        let reason =
            tracker.begin_block(&block(101 + WINDOW)).expect_err("previous never finished");

        assert_eq!(reason, BlockedReason::BlockSkipped { block_number: 100 + WINDOW });
    }

    #[test]
    fn a_checkpointed_snapshot_is_ready_without_replay() {
        let mut tracker = tracker();
        let checkpoint = TrustedCheckpoint {
            block_number: 5_000,
            block_hash: block_hash(5_000),
            state_root: state_root(5_000),
        };
        let observation = CacheObservation {
            cache_block: 5_000,
            cache_root: B256::repeat_byte(0xcc),
            trie_state_root: Some(state_root(5_000)),
        };

        let parent =
            tracker.restore_from_checkpoint(&checkpoint, &observation).expect("matches checkpoint");

        assert_eq!(parent.anchor.block_number, 5_000);
        assert_eq!(parent.trie_state_root, state_root(5_000));
        assert_eq!(parent.replay_depth, 0, "nothing was replayed");
        assert!(tracker.window_filled());

        // The restored parent is a real continuation point, not a one-off classification.
        apply_contiguous(&mut tracker, 5_001, 1);
        assert!(tracker.ready_parent().is_some());
    }

    #[test]
    fn an_empty_but_authenticated_snapshot_is_ready() {
        let mut tracker = tracker();
        let checkpoint = TrustedCheckpoint {
            block_number: 5_000,
            block_hash: block_hash(5_000),
            state_root: state_root(5_000),
        };
        // An empty cache still commits to a cache root and an authenticated trie root.
        let observation = CacheObservation {
            cache_block: 5_000,
            cache_root: B256::ZERO,
            trie_state_root: Some(state_root(5_000)),
        };

        assert!(tracker.restore_from_checkpoint(&checkpoint, &observation).is_ok());
    }

    #[test]
    fn a_snapshot_at_the_wrong_height_is_rejected() {
        let mut tracker = tracker();
        let checkpoint = TrustedCheckpoint {
            block_number: 5_000,
            block_hash: block_hash(5_000),
            state_root: state_root(5_000),
        };
        let observation = CacheObservation {
            cache_block: 4_999,
            cache_root: B256::repeat_byte(0xcc),
            trie_state_root: Some(state_root(5_000)),
        };

        let error = tracker.restore_from_checkpoint(&checkpoint, &observation).unwrap_err();

        assert_eq!(
            error,
            ReadinessError::CheckpointHeightMismatch { expected: 5_000, actual: 4_999 }
        );
        assert_eq!(
            *tracker.state(),
            CacheReadiness::Cold,
            "a rejected snapshot warms from scratch"
        );
    }

    #[test]
    fn a_snapshot_with_an_unauthenticated_trie_is_rejected() {
        let mut tracker = tracker();
        let checkpoint = TrustedCheckpoint {
            block_number: 5_000,
            block_hash: block_hash(5_000),
            state_root: state_root(5_000),
        };
        let observation = CacheObservation {
            cache_block: 5_000,
            cache_root: B256::repeat_byte(0xcc),
            trie_state_root: None,
        };

        let error = tracker.restore_from_checkpoint(&checkpoint, &observation).unwrap_err();

        assert_eq!(
            error,
            ReadinessError::CheckpointRootMismatch { expected: state_root(5_000), actual: None }
        );
        assert_eq!(*tracker.state(), CacheReadiness::Cold);
    }

    #[test]
    fn a_snapshot_claiming_a_forged_root_is_rejected() {
        let mut tracker = tracker();
        let checkpoint = TrustedCheckpoint {
            block_number: 5_000,
            block_hash: block_hash(5_000),
            state_root: state_root(5_000),
        };
        let observation = CacheObservation {
            cache_block: 5_000,
            cache_root: B256::repeat_byte(0xcc),
            trie_state_root: Some(B256::repeat_byte(0xad)),
        };

        assert!(tracker.restore_from_checkpoint(&checkpoint, &observation).is_err());
        assert!(tracker.ready_parent().is_none());
    }
}
