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
use tracing::debug;

/// Tracks whether the joint cache may be used to validate the next block.
///
/// The tracker observes block application; it never mutates the caches. Callers report what they
/// are about to do ([`begin_block`](Self::begin_block)) and what the caches then reported
/// ([`finish_block`](Self::finish_block)), and read the resulting classification back.
#[derive(Debug, Clone)]
pub struct CacheReadinessTracker {
    state: CacheReadiness,
    /// Eviction window the cache policy advertises. Callers running separate account and storage
    /// windows pass the larger of the two: the window is only whole once both are.
    window_size: u64,
    /// The [`replay_depth`](Self::replay_depth) at which the window became whole, if it has.
    ///
    /// A depth rather than a flag because undoing a block has to know how far back the window
    /// stays whole. Replaying from cold fills it at exactly `required_replay_depth`, so one block
    /// given back un-fills it; a snapshot arrives whole at depth 0, so every block replayed on top
    /// of it can be given back and the generation underneath is still the one the checkpoint
    /// vouched for. A single boolean cannot tell those two apart.
    ///
    /// Cleared by every cold reset, and never raised once set: the window only ever grows.
    window_filled_at: Option<u64>,
    replay_depth: u64,
    last_applied: Option<AppliedBlock>,
    /// Highest height for which this and every lower block was applied.
    ///
    /// Tracked apart from `last_applied` because a block that was never applied does not stop
    /// later blocks from being applied: after a gap, `last_applied` keeps advancing while this
    /// does not.
    acknowledgeable: Option<AppliedBlock>,
    /// Height of the first block that was delivered and never applied, if any.
    ///
    /// Latches `acknowledgeable` until a reorg or revert unwinds below it, at which point the
    /// missing block is no longer canonical and no longer needs to have been processed.
    first_gap: Option<u64>,
    cache_policy_id: B256,
}

impl CacheReadinessTracker {
    /// Creates a tracker for a cold cache.
    pub const fn new(window_size: u64, cache_policy_id: B256) -> Self {
        Self {
            state: CacheReadiness::Cold,
            window_size,
            window_filled_at: None,
            replay_depth: 0,
            last_applied: None,
            acknowledgeable: None,
            first_gap: None,
            cache_policy_id,
        }
    }

    /// Contiguous blocks that must be replayed after a cold reset before the advertised window is
    /// genuinely populated.
    ///
    /// One more than the window size. `LastNBlocksPolicy` retains entries whose last access is at
    /// or above `current_block - window_size`, so a cache at height `H` covers the closed range
    /// `[H - window_size, H]` — `window_size + 1` distinct heights, not `window_size`.
    pub const fn required_replay_depth(&self) -> u64 {
        self.window_size.saturating_add(1)
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
        self.window_filled_at.is_some()
    }

    /// Whether progress has stopped pending operator action.
    pub const fn is_blocked(&self) -> bool {
        matches!(self.state, CacheReadiness::Blocked { .. })
    }

    /// The highest block this ExEx may report as processed, if any.
    ///
    /// This is the *contiguous* watermark, not the newest block applied. Acknowledging a height is
    /// a durable promise to the host node, which prunes below it and never redelivers those blocks;
    /// a block that was skipped is therefore lost for good if the acknowledgement passes it. After
    /// a gap the caches can be reset and warmed again from the next block — that restores the
    /// caches, but it does not retroactively process the block that was missed, so this stops
    /// advancing.
    pub const fn acknowledgeable_height(&self) -> Option<(u64, B256)> {
        match self.acknowledgeable {
            Some(AppliedBlock { number, hash }) => Some((number, hash)),
            None => None,
        }
    }

    /// The first block that was delivered and never applied, if the watermark is stuck behind one.
    pub const fn first_gap(&self) -> Option<u64> {
        self.first_gap
    }

    /// Drops all continuity claims, matching a cold reset of both caches.
    ///
    /// Deliberately leaves the acknowledgement watermark alone. Resetting the caches makes them
    /// sound again, but it does not process a block that was skipped, and the watermark is a claim
    /// about processing rather than about cache contents.
    pub fn reset(&mut self) {
        self.state = CacheReadiness::Cold;
        self.window_filled_at = None;
        self.replay_depth = 0;
        self.last_applied = None;
    }

    /// Marks the caches as being unwound after a reorg or revert.
    ///
    /// Distinct from [`reset`](Self::reset) so that a reorg whose reset never completes cannot be
    /// mistaken for a clean cold start.
    ///
    /// `unwound_to` is the lowest height leaving the canonical chain. A gap at or above it is being
    /// unwound: the block that was never applied is no longer canonical, so nothing is owed for it
    /// and the watermark is released. The watermark itself drops to just below the unwind, since
    /// everything above it is about to be replaced.
    pub fn begin_recovery(&mut self, unwound_to: u64) {
        self.state = CacheReadiness::Recovering;
        if self.first_gap.is_some_and(|gap| gap >= unwound_to) {
            self.first_gap = None;
        }
        if self.acknowledgeable.is_some_and(|applied| applied.number >= unwound_to) {
            // Nothing at or above the unwind point is processed any more, and the pre-unwind block
            // hash is not known here, so the watermark restarts from the replayed chain.
            self.acknowledgeable = None;
        }
    }

    /// Promotes a snapshot-restored cache to [`CacheReadiness::Ready`] against an
    /// operator-supplied checkpoint.
    ///
    /// A snapshot bypasses the replay that would otherwise fill the window, so everything the
    /// resulting anchor will claim is checked against a checkpoint the operator supplied out of
    /// band rather than against the snapshot's own assertions. That includes the cache root and
    /// policy identifier: a snapshot can reproduce the canonical state root while holding a
    /// different set of cached values, and peers compare anchors, not state roots. A rejected
    /// snapshot leaves the tracker [`Cold`](CacheReadiness::Cold): warming from scratch is always
    /// sound.
    pub fn restore_from_checkpoint(
        &mut self,
        checkpoint: &TrustedCheckpoint,
        observed: &CacheObservation,
    ) -> Result<&ReadyParent, ReadinessError> {
        self.reset();
        self.check_checkpoint(checkpoint, observed)?;
        // A snapshot brought its own history and none of it was replayed here, so the window is
        // asserted rather than counted and arrives whole at depth zero.
        self.promote_to_checkpoint(checkpoint, 0, 0);
        Ok(self.ready_parent().expect("just set to Ready"))
    }

    /// Promotes a pair that undid exactly one applied block back to
    /// [`CacheReadiness::Ready`] at that block's parent.
    ///
    /// Distinct from [`restore_from_checkpoint`](Self::restore_from_checkpoint) because a depth-1
    /// undo is the one restore whose replay history this tracker already owns. A snapshot has to
    /// assert the window — nothing else vouches for it — but an undo must not: the pair ends up
    /// exactly as warm as it was before the reorg, minus the block given back. Asserting it here
    /// would promote a still-warming pair to `Ready` on the strength of a reorg, and `Ready` is
    /// what opens the sidecar publication gate.
    ///
    /// Refuses with [`ReadinessError::UndoneBlockStillWarming`] when the pair would not stay warm,
    /// leaving the tracker untouched so the caller can fall back to a rebuild — which is the only
    /// thing that genuinely fills the window. Every other rejection resets to
    /// [`Cold`](CacheReadiness::Cold), matching a checkpoint restore.
    pub fn restore_from_undone_block(
        &mut self,
        checkpoint: &TrustedCheckpoint,
        observed: &CacheObservation,
    ) -> Result<&ReadyParent, ReadinessError> {
        let Some((replay_depth, window_filled_at)) = self.depth_after_one_undo() else {
            return Err(ReadinessError::UndoneBlockStillWarming {
                replay_depth: self.replay_depth.saturating_sub(1),
                required: self.window_filled_at.unwrap_or_else(|| self.required_replay_depth()),
            })
        };

        self.reset();
        self.check_checkpoint(checkpoint, observed)?;
        self.replay_depth = replay_depth;
        self.promote_to_checkpoint(checkpoint, replay_depth, window_filled_at);
        Ok(self.ready_parent().expect("just set to Ready"))
    }

    /// Whether undoing the last applied block would leave the window still whole.
    ///
    /// Exposed so a caller can decline the undo before it mutates anything, rather than rolling
    /// the caches back and learning from [`restore_from_undone_block`](
    /// Self::restore_from_undone_block) that the result has no `Ready` to return to.
    pub const fn stays_warm_after_one_undo(&self) -> bool {
        self.depth_after_one_undo().is_some()
    }

    /// The replay depth an undo would land on, and the depth the window became whole at.
    ///
    /// `None` when the undo is not available at all: either nothing was replayed to give back —
    /// a checkpoint is a floor, not a block this run applied — or giving one back would drop the
    /// pair below where its window became whole.
    const fn depth_after_one_undo(&self) -> Option<(u64, u64)> {
        match (self.replay_depth.checked_sub(1), self.window_filled_at) {
            (Some(replay_depth), Some(filled_at)) if replay_depth >= filled_at => {
                Some((replay_depth, filled_at))
            }
            _ => None,
        }
    }

    /// Checks a checkpoint against what the caches actually hold, changing nothing.
    fn check_checkpoint(
        &self,
        checkpoint: &TrustedCheckpoint,
        observed: &CacheObservation,
    ) -> Result<(), ReadinessError> {
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
        if observed.cache_root != checkpoint.cache_root {
            return Err(ReadinessError::CheckpointCacheRootMismatch {
                expected: checkpoint.cache_root,
                actual: observed.cache_root,
            })
        }
        if checkpoint.cache_policy_id != self.cache_policy_id {
            return Err(ReadinessError::CheckpointPolicyMismatch {
                expected: self.cache_policy_id,
                actual: checkpoint.cache_policy_id,
            })
        }
        Ok(())
    }

    /// Moves an already-checked checkpoint into `Ready`, along with the watermarks it settles.
    fn promote_to_checkpoint(
        &mut self,
        checkpoint: &TrustedCheckpoint,
        replay_depth: u64,
        window_filled_at: u64,
    ) {
        self.window_filled_at = Some(window_filled_at);
        let applied = AppliedBlock { number: checkpoint.block_number, hash: checkpoint.block_hash };
        self.last_applied = Some(applied);
        // A checkpoint at H asserts that no block at or below H will be needed again, which is
        // exactly what the gap latch is withholding: the skipped block's effects are inside the
        // restored generation whether they arrived by replay or by snapshot. A gap *above* the
        // checkpoint is still owed and still latches.
        if self.first_gap.is_some_and(|gap| gap <= checkpoint.block_number) {
            self.first_gap = None;
        }
        // The watermark starts at the checkpoint rather than staying unset. It is not a claim that
        // this ExEx processed the blocks below it — it did not — but the watermark's contract is
        // that no earlier block will be needed again, and a snapshot at this height is exactly the
        // statement that none will be. Leaving it unset would be worse than useless: the first
        // block applied afterwards would find no watermark and no gap, and would set it to
        // `block_number + 1`, making the same claim with nothing vouching for it.
        //
        // This holds only while the restored generation is durable. A checkpoint that is never
        // persisted leaves a restarted node unable to reproduce state it has already acknowledged.
        self.acknowledgeable = Some(applied);
        self.state = CacheReadiness::Ready(ReadyParent {
            anchor: CacheAnchor {
                block_number: checkpoint.block_number,
                block_hash: checkpoint.block_hash,
                cache_policy_id: self.cache_policy_id,
                cache_root: checkpoint.cache_root,
            },
            trie_state_root: checkpoint.state_root,
            replay_depth,
        });
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
        let applied = AppliedBlock { number: block.number, hash: block.hash };
        self.last_applied = Some(applied);
        // Recorded at the depth it first happened and never raised afterwards, so that undoing a
        // block compares against where the window became whole rather than against where it is now.
        if self.window_filled_at.is_none() && self.replay_depth >= self.required_replay_depth() {
            self.window_filled_at = Some(self.replay_depth);
        }
        // The gap being filled by delivery rather than by a reorg. A cold reset re-executes the
        // block that was missed, and when the block now applied is exactly the one the watermark
        // is stuck below *and* chains onto the acknowledged block by hash, everything below it
        // has genuinely been processed — the latch has nothing left to protect. The hash is what
        // makes this safe: a sibling at the same height fills nothing, and would otherwise let a
        // watermark claim a block that was never applied on this branch.
        //
        // Recognised here rather than released by a caller, so that every consumer of this
        // tracker heals the same way and none of them has to know it needs to.
        if self.first_gap == Some(block.number) &&
            self.acknowledgeable.is_some_and(|previous| {
                block.number == previous.number + 1 && block.parent_hash == previous.hash
            })
        {
            debug!(
                target: "partial_stateless",
                block = block.number,
                "The block the acknowledgement watermark was stuck below was re-applied; the \
                 watermark advances again"
            );
            self.first_gap = None;
        }

        // Only contiguous from the start counts: once a block has been skipped, later blocks are
        // applied but the promise "everything below this is processed" is no longer true.
        let contiguous_from_start =
            self.acknowledgeable.is_none_or(|previous| block.number == previous.number + 1) &&
                self.first_gap.is_none();
        if contiguous_from_start {
            self.acknowledgeable = Some(applied);
        }

        // The trie root proves the caches describe this block's post-state, which is precisely the
        // state the *next* block executes against.
        let authenticated = observed.trie_state_root == Some(block.state_root);
        self.state = if authenticated && self.window_filled() {
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
    /// skipping and carrying on is what silently produces a cache that lies about its parent. The
    /// acknowledgement watermark stops here permanently, because resetting the caches does not
    /// process this block — only a reorg that drops it from the canonical chain releases it.
    pub fn abandon_block(&mut self, block_number: u64) -> BlockedReason {
        self.block_on(BlockedReason::BlockSkipped { block_number })
    }

    /// Enters the blocked state, preserving the reason recorded first.
    fn block_on(&mut self, reason: BlockedReason) -> BlockedReason {
        // Every blocking reason names a block that was delivered and not applied, so each one
        // freezes the watermark at the last block that genuinely was.
        let missing = match &reason {
            BlockedReason::BlockGap { expected_number, .. } => *expected_number,
            BlockedReason::BlockSkipped { block_number } |
            BlockedReason::RecoveryIncomplete { block_number } => *block_number,
            BlockedReason::CacheDrift { expected_block, .. } => *expected_block,
        };
        self.first_gap = Some(self.first_gap.map_or(missing, |first| first.min(missing)));

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

/// A cache generation the operator vouches for out of band, used to accept a snapshot.
///
/// Snapshots are accepted on operator authority rather than on peer consensus: there is no scoring
/// or majority vote here, and a snapshot that disagrees with the checkpoint is simply discarded.
///
/// It names every field of the [`CacheAnchor`] the restored cache will publish, not just the
/// canonical state root. Two caches can reproduce the same state root while holding different
/// values — different eviction windows retain different subsets — and peers compare anchors. An
/// operator who vouches only for the state root would be vouching for the chain, not for this
/// cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrustedCheckpoint {
    /// Height the snapshot must be at.
    pub block_number: u64,
    /// Hash the snapshot must be at.
    pub block_hash: B256,
    /// Canonical state root the restored trie must reproduce.
    pub state_root: B256,
    /// Deterministic root over the value-cache contents the snapshot must hold.
    pub cache_root: B256,
    /// Cache policy the snapshot was produced under.
    pub cache_policy_id: B256,
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
    /// The restored cache holds different values than the checkpoint vouches for.
    CheckpointCacheRootMismatch {
        /// Checkpoint cache root.
        expected: B256,
        /// Restored cache root.
        actual: B256,
    },
    /// The checkpoint was produced under a different cache policy than this node runs.
    CheckpointPolicyMismatch {
        /// Policy this node runs.
        expected: B256,
        /// Policy the checkpoint names.
        actual: B256,
    },
    /// Undoing one block would leave the pair short of a full policy window.
    ///
    /// Not a corruption: the pair is sound, it just has no `Ready` to return to. Only a rebuild
    /// can fill the window, so this is the signal to take that path rather than to reset.
    UndoneBlockStillWarming {
        /// Blocks that would remain replayed after the undo.
        replay_depth: u64,
        /// Blocks a full policy window needs.
        required: u64,
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
    /// `LastNBlocksPolicy` retains the closed range `[H - WINDOW, H]`, so filling the window takes
    /// one more block than the window size.
    const REPLAY: u64 = WINDOW + 1;

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
        numbered(number, 0xbb)
    }

    fn state_root(number: u64) -> B256 {
        numbered(number, 0x55)
    }

    fn cache_root(number: u64) -> B256 {
        numbered(number, 0xcc)
    }

    fn numbered(number: u64, tag: u8) -> B256 {
        let mut value = B256::ZERO;
        value[0..8].copy_from_slice(&number.to_be_bytes());
        value[31] = tag;
        value
    }

    fn checkpoint_at(number: u64) -> TrustedCheckpoint {
        TrustedCheckpoint {
            block_number: number,
            block_hash: block_hash(number),
            state_root: state_root(number),
            cache_root: cache_root(number),
            cache_policy_id: policy_id(),
        }
    }

    /// What a snapshot restored exactly at `checkpoint` reports about itself.
    fn restored(checkpoint: &TrustedCheckpoint) -> CacheObservation {
        CacheObservation {
            cache_block: checkpoint.block_number,
            cache_root: checkpoint.cache_root,
            trie_state_root: Some(checkpoint.state_root),
        }
    }

    /// The caches behaving correctly: at the block's height, with the block's authenticated root.
    fn observed(block: &BlockContext) -> CacheObservation {
        CacheObservation {
            cache_block: block.number,
            cache_root: cache_root(block.number),
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
    fn the_window_needs_one_more_block_than_its_size() {
        let mut tracker = tracker();
        assert_eq!(tracker.required_replay_depth(), WINDOW + 1);

        // A cache at height H retains entries last accessed at or above H - WINDOW, so replaying
        // exactly WINDOW blocks leaves the oldest height in that closed range unpopulated.
        apply_contiguous(&mut tracker, 100, WINDOW);
        assert!(tracker.ready_parent().is_none(), "{WINDOW} blocks cover only {WINDOW} heights");
        assert!(!tracker.window_filled());

        apply_contiguous(&mut tracker, 100 + WINDOW, 1);

        let parent = tracker.ready_parent().expect("window filled");
        assert_eq!(parent.replay_depth, REPLAY);
        assert_eq!(parent.anchor.block_number, 100 + WINDOW);
        assert_eq!(parent.anchor.block_hash, block_hash(100 + WINDOW));
        assert_eq!(parent.anchor.cache_policy_id, policy_id());
        assert_eq!(parent.anchor.cache_root, cache_root(100 + WINDOW));
        assert_eq!(parent.trie_state_root, state_root(100 + WINDOW));
    }

    #[test]
    fn reset_drops_readiness_and_replay_depth() {
        let mut tracker = tracker();
        apply_contiguous(&mut tracker, 100, REPLAY);
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
    fn the_watermark_follows_contiguous_blocks() {
        let mut tracker = tracker();
        assert_eq!(tracker.acknowledgeable_height(), None, "nothing processed yet");

        apply_contiguous(&mut tracker, 100, 3);

        assert_eq!(tracker.acknowledgeable_height(), Some((102, block_hash(102))));
        assert_eq!(tracker.first_gap(), None);
    }

    #[test]
    fn the_watermark_never_passes_a_skipped_block_even_after_recovery() {
        let mut tracker = tracker();
        apply_contiguous(&mut tracker, 100, 1);
        assert_eq!(tracker.acknowledgeable_height(), Some((100, block_hash(100))));

        // Block 101 was delivered and could not be applied.
        tracker.abandon_block(101);
        assert_eq!(tracker.first_gap(), Some(101));

        // Resetting the caches and warming again from 102 makes the caches sound, but 101 is still
        // unprocessed and the host node would prune it if the watermark moved.
        tracker.reset();
        apply_contiguous(&mut tracker, 102, REPLAY);

        assert!(tracker.ready_parent().is_some(), "the caches themselves recovered");
        assert_eq!(
            tracker.acknowledgeable_height(),
            Some((100, block_hash(100))),
            "the watermark is a claim about processing, which resetting the caches does not do"
        );
    }

    #[test]
    fn the_watermark_stays_below_a_gap_the_cache_recovered_from() {
        let mut tracker = tracker();
        apply_contiguous(&mut tracker, 100, 2);

        // A non-contiguous notification: 103 is admitted only after a reset.
        assert!(tracker.begin_block(&block(103)).is_err());
        tracker.reset();
        apply_contiguous(&mut tracker, 103, 3);

        assert_eq!(tracker.first_gap(), Some(102));
        assert_eq!(tracker.acknowledgeable_height(), Some((101, block_hash(101))));
    }

    #[test]
    fn readmitting_the_exact_gap_block_releases_the_watermark() {
        // The other way a gap stops being a gap. A reorg drops the missing block from the chain;
        // a cold reset re-executes it. Both leave nothing owed, and until now only the first was
        // recognised — so a producer that recovered by re-applying the block it missed reported
        // a durability watermark frozen for the rest of the run.
        let mut tracker = tracker();
        apply_contiguous(&mut tracker, 100, 1);
        tracker.abandon_block(101);
        assert_eq!(tracker.first_gap(), Some(101));

        // The cold reset, and then the very block that was missed.
        tracker.reset();
        apply_contiguous(&mut tracker, 101, 2);

        assert_eq!(tracker.first_gap(), None, "101 was processed after all");
        assert_eq!(tracker.acknowledgeable_height(), Some((102, block_hash(102))));
    }

    #[test]
    fn a_sibling_at_the_gap_height_does_not_release_the_watermark() {
        // Same height, different block. Nothing about executing it says the block that was
        // actually missed was ever processed, and a watermark is a claim about *blocks*.
        let mut tracker = tracker();
        apply_contiguous(&mut tracker, 100, 1);
        tracker.abandon_block(101);
        tracker.reset();

        let mut sibling = block(101);
        sibling.hash = B256::repeat_byte(0xee);
        sibling.parent_hash = B256::repeat_byte(0xdd);
        tracker.begin_block(&sibling).expect("a reset admits it");
        tracker.finish_block(&sibling, &observed(&sibling));

        assert_eq!(tracker.first_gap(), Some(101), "the missing block is still missing");
        assert_eq!(tracker.acknowledgeable_height(), Some((100, block_hash(100))));
    }

    #[test]
    fn unwinding_below_a_gap_releases_the_watermark() {
        let mut tracker = tracker();
        apply_contiguous(&mut tracker, 100, 1);
        tracker.abandon_block(101);
        assert_eq!(tracker.first_gap(), Some(101));

        // A reorg that drops 101 from the canonical chain: nothing is owed for it any more.
        tracker.begin_recovery(101);
        tracker.reset();

        assert_eq!(tracker.first_gap(), None);
        apply_contiguous(&mut tracker, 101, 2);
        assert_eq!(tracker.acknowledgeable_height(), Some((102, block_hash(102))));
    }

    #[test]
    fn unwinding_above_a_gap_leaves_the_watermark_stuck() {
        let mut tracker = tracker();
        apply_contiguous(&mut tracker, 100, 1);
        tracker.abandon_block(101);

        // The reorg starts above the missing block, so 101 is still canonical and still
        // unprocessed.
        tracker.begin_recovery(150);
        tracker.reset();

        assert_eq!(tracker.first_gap(), Some(101));
        apply_contiguous(&mut tracker, 150, 2);
        assert_eq!(tracker.acknowledgeable_height(), Some((100, block_hash(100))));
    }

    #[test]
    fn blocked_preserves_the_first_reason() {
        let mut tracker = tracker();
        apply_contiguous(&mut tracker, 100, 1);
        tracker.abandon_block(101);

        let reason = tracker.begin_block(&block(105)).expect_err("still blocked");

        assert_eq!(reason, BlockedReason::BlockSkipped { block_number: 101 });
        assert_eq!(tracker.first_gap(), Some(101), "the earliest gap is the binding one");
    }

    #[test]
    fn an_unauthenticated_trie_never_reaches_ready() {
        let mut tracker = tracker();

        for number in 100..100 + REPLAY + 2 {
            let block = block(number);
            tracker.begin_block(&block).expect("contiguous");
            tracker.finish_block(
                &block,
                &CacheObservation { trie_state_root: None, ..observed(&block) },
            );
        }

        assert!(tracker.ready_parent().is_none());
        assert!(tracker.window_filled(), "the window fills regardless of authentication");
        assert_eq!(*tracker.state(), CacheReadiness::Warming { replay_depth: REPLAY + 2 });
    }

    #[test]
    fn a_wrong_trie_root_never_reaches_ready() {
        let mut tracker = tracker();

        for number in 100..100 + REPLAY + 2 {
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
        assert_eq!(tracker.acknowledgeable_height(), None);
    }

    #[test]
    fn recovery_must_complete_before_blocks_resume() {
        let mut tracker = tracker();
        apply_contiguous(&mut tracker, 100, REPLAY);

        tracker.begin_recovery(100);
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
        apply_contiguous(&mut tracker, 100, REPLAY);

        // Admitted, then never completed — the caches saw part of a block, or none of it.
        tracker.begin_block(&block(100 + REPLAY)).expect("contiguous");
        let reason =
            tracker.begin_block(&block(101 + REPLAY)).expect_err("previous never finished");

        assert_eq!(reason, BlockedReason::BlockSkipped { block_number: 100 + REPLAY });
    }

    #[test]
    fn a_checkpointed_snapshot_is_ready_without_replay() {
        let mut tracker = tracker();
        let checkpoint = checkpoint_at(5_000);

        let parent = tracker
            .restore_from_checkpoint(&checkpoint, &restored(&checkpoint))
            .expect("matches checkpoint");

        assert_eq!(parent.anchor.block_number, 5_000);
        assert_eq!(parent.anchor.cache_root, checkpoint.cache_root);
        assert_eq!(parent.anchor.cache_policy_id, policy_id());
        assert_eq!(parent.trie_state_root, state_root(5_000));
        assert_eq!(parent.replay_depth, 0, "nothing was replayed");
        assert!(tracker.window_filled());

        // The checkpoint itself is what vouches for not needing earlier blocks, so the watermark
        // starts there instead of jumping to 5_001 on the first block applied afterwards.
        assert_eq!(tracker.acknowledgeable_height(), Some((5_000, block_hash(5_000))));

        // The restored parent is a real continuation point, not a one-off classification.
        apply_contiguous(&mut tracker, 5_001, 1);
        assert!(tracker.ready_parent().is_some());
        assert_eq!(tracker.acknowledgeable_height(), Some((5_001, block_hash(5_001))));
    }

    #[test]
    fn undoing_one_block_keeps_the_replay_history_it_did_not_undo() {
        let mut tracker = tracker();
        // One more than the window needs, so the pair survives giving one block back.
        apply_contiguous(&mut tracker, 100, REPLAY + 1);
        let undone_to = 100 + REPLAY - 1;
        tracker.begin_recovery(undone_to + 1);

        let checkpoint = checkpoint_at(undone_to);
        let parent = tracker
            .restore_from_undone_block(&checkpoint, &restored(&checkpoint))
            .expect("still warm after giving one block back");

        assert_eq!(parent.anchor.block_number, undone_to);
        assert_eq!(
            parent.replay_depth, REPLAY,
            "the undo gives back exactly one block, not the whole replay history"
        );
        assert_eq!(tracker.replay_depth(), REPLAY);
        assert!(tracker.window_filled());

        // The restored parent is a real continuation point, not a one-off classification.
        apply_contiguous(&mut tracker, undone_to + 1, 1);
        assert!(tracker.ready_parent().is_some());
    }

    #[test]
    fn undoing_one_block_from_a_warming_pair_is_refused() {
        let mut tracker = tracker();
        // Exactly at the boundary: the window is filled, but only by the block being undone.
        apply_contiguous(&mut tracker, 100, REPLAY);
        assert!(tracker.window_filled(), "the window did fill before the reorg");
        let undone_to = 100 + REPLAY - 2;
        tracker.begin_recovery(undone_to + 1);

        let checkpoint = checkpoint_at(undone_to);
        let error = tracker
            .restore_from_undone_block(&checkpoint, &restored(&checkpoint))
            .expect_err("one block short of a window");

        assert_eq!(
            error,
            ReadinessError::UndoneBlockStillWarming { replay_depth: REPLAY - 1, required: REPLAY }
        );
        assert_eq!(
            *tracker.state(),
            CacheReadiness::Recovering,
            "a refusal leaves the caller free to rebuild rather than forcing a cold start"
        );
    }

    #[test]
    fn a_block_applied_on_a_snapshot_can_be_undone_without_replaying_a_window() {
        // The snapshot's own generation is warm by assertion, so a pair one block past it can give
        // that block back and land on a generation the checkpoint already vouched for — even
        // though it has replayed nowhere near a window's worth of blocks.
        let mut tracker = tracker();
        let restore = checkpoint_at(5_000);
        tracker.restore_from_checkpoint(&restore, &restored(&restore)).expect("matches");
        apply_contiguous(&mut tracker, 5_001, 1);
        assert_eq!(tracker.replay_depth(), 1);
        tracker.begin_recovery(5_001);

        let parent = tracker
            .restore_from_undone_block(&restore, &restored(&restore))
            .expect("the generation underneath is the checkpointed one");

        assert_eq!(parent.anchor.block_number, 5_000);
        assert_eq!(parent.replay_depth, 0);

        // But only back to the checkpoint: there is nothing underneath it to undo into.
        tracker.begin_recovery(5_000);
        let error = tracker
            .restore_from_undone_block(&checkpoint_at(4_999), &restored(&checkpoint_at(4_999)))
            .expect_err("the checkpoint is the floor");
        assert!(matches!(error, ReadinessError::UndoneBlockStillWarming { .. }));
    }

    #[test]
    fn a_snapshot_promotes_a_warming_pair_but_an_undo_does_not() {
        // The asymmetry is the point: a checkpoint arrives with its own history and asserts the
        // window, while an undo can only give back what this run already replayed.
        let checkpoint = checkpoint_at(100);

        let mut snapshotted = tracker();
        apply_contiguous(&mut snapshotted, 96, 1);
        assert!(!snapshotted.window_filled());
        assert!(snapshotted.restore_from_checkpoint(&checkpoint, &restored(&checkpoint)).is_ok());

        let mut undone = tracker();
        apply_contiguous(&mut undone, 96, 1);
        assert!(undone.restore_from_undone_block(&checkpoint, &restored(&checkpoint)).is_err());
    }

    #[test]
    fn an_undo_still_checks_the_checkpoint_it_is_handed() {
        let mut tracker = tracker();
        apply_contiguous(&mut tracker, 100, REPLAY + 1);
        tracker.begin_recovery(100 + REPLAY);

        // Warm enough to undo, but the caches do not reproduce the canonical root.
        let checkpoint = checkpoint_at(100 + REPLAY - 1);
        let observation = CacheObservation {
            trie_state_root: Some(B256::repeat_byte(0xfe)),
            ..restored(&checkpoint)
        };
        let error = tracker.restore_from_undone_block(&checkpoint, &observation).unwrap_err();

        assert!(matches!(error, ReadinessError::CheckpointRootMismatch { .. }));
        assert_eq!(
            *tracker.state(),
            CacheReadiness::Cold,
            "a pair that failed authentication is reset, not left claiming a recovery"
        );
    }

    #[test]
    fn a_checkpoint_at_or_above_a_gap_releases_the_acknowledgement() {
        let mut tracker = tracker();
        apply_contiguous(&mut tracker, 100, 1);
        tracker.abandon_block(101);
        assert_eq!(tracker.first_gap(), Some(101));
        assert_eq!(tracker.acknowledgeable_height(), Some((100, block_hash(100))));

        // A rebuild or snapshot restore at 150 reconstructs a generation that already contains
        // whatever block 101 did, so nothing at or below 150 is owed any more.
        let checkpoint = checkpoint_at(150);
        tracker.restore_from_checkpoint(&checkpoint, &restored(&checkpoint)).expect("matches");

        assert_eq!(tracker.first_gap(), None);
        apply_contiguous(&mut tracker, 151, 1);
        assert_eq!(tracker.acknowledgeable_height(), Some((151, block_hash(151))));
    }

    #[test]
    fn a_checkpoint_below_a_gap_leaves_the_acknowledgement_pinned() {
        let mut tracker = tracker();
        apply_contiguous(&mut tracker, 100, 1);
        tracker.abandon_block(101);

        // Restoring *below* the skipped block says nothing about it, so the latch stays.
        let checkpoint = checkpoint_at(90);
        tracker.restore_from_checkpoint(&checkpoint, &restored(&checkpoint)).expect("matches");

        assert_eq!(tracker.first_gap(), Some(101));
        apply_contiguous(&mut tracker, 91, 1);
        assert_eq!(
            tracker.acknowledgeable_height(),
            Some((90, block_hash(90))),
            "the watermark cannot step over a block that is still unprocessed"
        );
    }

    #[test]
    fn an_empty_but_authenticated_snapshot_is_ready() {
        let mut tracker = tracker();
        // An empty cache still commits to a cache root and an authenticated trie root.
        let checkpoint = TrustedCheckpoint { cache_root: B256::ZERO, ..checkpoint_at(5_000) };

        assert!(tracker.restore_from_checkpoint(&checkpoint, &restored(&checkpoint)).is_ok());
    }

    #[test]
    fn a_snapshot_at_the_wrong_height_is_rejected() {
        let mut tracker = tracker();
        let checkpoint = checkpoint_at(5_000);
        let observation = CacheObservation { cache_block: 4_999, ..restored(&checkpoint) };

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
        let checkpoint = checkpoint_at(5_000);
        let observation = CacheObservation { trie_state_root: None, ..restored(&checkpoint) };

        let error = tracker.restore_from_checkpoint(&checkpoint, &observation).unwrap_err();

        assert_eq!(
            error,
            ReadinessError::CheckpointRootMismatch { expected: state_root(5_000), actual: None }
        );
        assert_eq!(*tracker.state(), CacheReadiness::Cold);
    }

    #[test]
    fn a_snapshot_claiming_a_forged_state_root_is_rejected() {
        let mut tracker = tracker();
        let checkpoint = checkpoint_at(5_000);
        let observation = CacheObservation {
            trie_state_root: Some(B256::repeat_byte(0xad)),
            ..restored(&checkpoint)
        };

        assert!(tracker.restore_from_checkpoint(&checkpoint, &observation).is_err());
        assert!(tracker.ready_parent().is_none());
    }

    #[test]
    fn a_snapshot_holding_different_values_is_rejected() {
        let mut tracker = tracker();
        let checkpoint = checkpoint_at(5_000);
        // Same canonical state root, different cached values: two nodes can agree on the chain and
        // still hold different subsets of it, and the anchor a peer compares carries the values.
        let observation =
            CacheObservation { cache_root: B256::repeat_byte(0xde), ..restored(&checkpoint) };

        let error = tracker.restore_from_checkpoint(&checkpoint, &observation).unwrap_err();

        assert_eq!(
            error,
            ReadinessError::CheckpointCacheRootMismatch {
                expected: cache_root(5_000),
                actual: B256::repeat_byte(0xde),
            }
        );
        assert!(tracker.ready_parent().is_none());
    }

    #[test]
    fn a_snapshot_from_a_different_policy_is_rejected() {
        let mut tracker = tracker();
        let foreign = B256::repeat_byte(0x0f);
        let checkpoint = TrustedCheckpoint { cache_policy_id: foreign, ..checkpoint_at(5_000) };

        let error =
            tracker.restore_from_checkpoint(&checkpoint, &restored(&checkpoint)).unwrap_err();

        assert_eq!(
            error,
            ReadinessError::CheckpointPolicyMismatch { expected: policy_id(), actual: foreign }
        );
        assert!(tracker.ready_parent().is_none());
    }
}
