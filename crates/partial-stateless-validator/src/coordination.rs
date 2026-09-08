//! The coordinated generation a partial-stateless validator maintains, and how it moves.
//!
//! One value cache and one trie cache advance together or not at all, authenticated by a readiness
//! tracker. This module owns that state and the operations that are protocol rather than policy:
//! taking its fingerprint, retaining and restoring the one previous trie generation a depth-1 reorg
//! needs, and reporting whether a block may be applied.
//!
//! What is deliberately *not* here is every fallback that needs a state database. A full node that
//! cannot undo one block rebuilds from canonical state or cold-resets and rewarms; a standalone
//! validator can do neither: it must instead request a snapshot at the exact common ancestor.
//! Keeping those fallbacks on the caller's side of the boundary is what makes the difference
//! explicit rather than a branch this module could accidentally grow.
//!
//! Logging state lives with the caller too. [`CoordinatedPair`] carries protocol state only, so the
//! ExEx wraps it to add the last readiness label its run log reports on.

use alloy_primitives::B256;
use partial_stateless::{
    network_cache::NetworkStateCache,
    readiness::{
        BlockContext, BlockedReason, CacheObservation, CacheReadinessTracker, ReadyParent,
        TrustedCheckpoint,
    },
    PartialTrieNodeCache, TrieCacheMemory,
};
use reth_ethereum_primitives::EthPrimitives;
use reth_primitives_traits::{AlloyBlockHeader, BlockTy, RecoveredBlock, SealedHeader};
/// Re-exported so [`CanonicalStateRoots`] can be implemented outside this crate without naming
/// the error crate: a standalone consumer answers the trait from its own verified history, and
/// making it add a dependency to spell the return type would be a boundary that means nothing.
pub use reth_storage_errors::provider::ProviderResult;
use serde::Serialize;
use std::{collections::VecDeque, time::Instant};
use tracing::{debug, info, warn};

/// The deepest reorg any pair may be configured to undo from its own retained generations.
///
/// A bound rather than a target. The cost of retention is memory that scales with K (§4.3 of the
/// deep-reorg plan), and the production default is 1; this exists so a misconfigured run is
/// refused at construction rather than discovered as an allocator report. The value is chosen to
/// sit well inside the replay history window, which is what lets a refusal at depth K still name
/// the right ancestor.
pub const MAX_RETENTION_DEPTH: u64 = 64;

/// How many trie generations a pair retains, and therefore the deepest reorg it can undo alone.
///
/// A newtype because three separate limits have to be the same number — the retained-generation
/// deque's cap, the flat undo log's prune depth, and the depth a reorg is refused above — and
/// nothing in the type system stopped them from being three constants that drifted apart. Every
/// one of them reads this off the pair, so there is one place to change and no way to disagree.
///
/// Zero is not a value. A pair that retains nothing is not "K = 0"; it is a pair with retention
/// switched off, which is a separate flag on the commit path, and conflating the two would make a
/// disabled run look like a legal configuration that always refuses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct RetentionDepth(u64);

impl RetentionDepth {
    /// The production default: today's behaviour exactly.
    pub const ONE: Self = Self(1);

    /// Validates a configured depth against `1 ..= MAX_RETENTION_DEPTH`.
    pub const fn new(depth: u64) -> Result<Self, RetentionDepthError> {
        if depth == 0 {
            return Err(RetentionDepthError::Zero)
        }
        if depth > MAX_RETENTION_DEPTH {
            return Err(RetentionDepthError::TooDeep { requested: depth, max: MAX_RETENTION_DEPTH })
        }
        Ok(Self(depth))
    }

    pub const fn get(self) -> u64 {
        self.0
    }

    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl Default for RetentionDepth {
    fn default() -> Self {
        Self::ONE
    }
}

impl std::fmt::Display for RetentionDepth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Why a configured retention depth was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetentionDepthError {
    /// Zero is retention switched off, which is a different setting.
    Zero,
    /// Deeper than [`MAX_RETENTION_DEPTH`].
    TooDeep { requested: u64, max: u64 },
}

impl std::fmt::Display for RetentionDepthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Zero => write!(
                f,
                "a retention depth of 0 is retention switched off, not a depth; use the retain \
                 flag on the commit path instead"
            ),
            Self::TooDeep { requested, max } => {
                write!(f, "a retention depth of {requested} is deeper than the maximum {max}")
            }
        }
    }
}

impl std::error::Error for RetentionDepthError {}

/// The run of blocks a caller believes it is giving back, ancestor first.
///
/// Consecutive `(number, hash)` does not prove parentage, and the pair cannot see the caller's own
/// verified history — so the caller states the lineage explicitly and the pair checks its retained
/// generations against it. Immutable once built, and built only through [`Self::new`], so a pair
/// that accepts one has already had the shape checked.
///
/// Preferred over walking `accepted_head.parent_hash` from the tip: that chain is `None` for a
/// generation retained one block out of a cold reset, which is a legal state a recovery must still
/// be able to land on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedLineage {
    /// The generations a depth-D undo consumes, ancestor first: `ancestor ..= ancestor + D - 1`.
    ///
    /// Generation tags, not abandoned blocks. A generation is tagged with the block it is the
    /// state *after*, and the tip of the abandoned run has no retained generation — its generation
    /// is the live cache, which the undo replaces rather than restores. Storing the tags makes
    /// that structural instead of an index adjustment at the point of comparison, and it is why a
    /// depth-1 lineage needs no hash for the block it gives back.
    tags: Vec<(u64, B256)>,
}

impl ExpectedLineage {
    /// Builds a lineage from the ancestor and the blocks above it, checking the shape.
    ///
    /// The shape check is arithmetic only: the run is non-empty, starts one above the ancestor,
    /// and does not skip a height. Whether those blocks are *this pair's* blocks is what
    /// [`CoordinatedPair::restore_retained_generations`] then decides against the deque; this
    /// constructor only rules out a run that could not describe any chain.
    pub fn new(ancestor: (u64, B256), abandoned: &[(u64, B256)]) -> Result<Self, LineageError> {
        let Some(first) = abandoned.first() else { return Err(LineageError::Empty) };
        if first.0 != ancestor.0 + 1 {
            return Err(LineageError::NotAboveAncestor { ancestor: ancestor.0, lowest: first.0 })
        }
        for window in abandoned.windows(2) {
            if window[1].0 != window[0].0 + 1 {
                return Err(LineageError::Gap { from: window[0].0, to: window[1].0 })
            }
        }
        // The tip is dropped: D abandoned blocks consume D generations, and they are the ancestor
        // plus the first D - 1 abandoned blocks.
        let mut tags = Vec::with_capacity(abandoned.len());
        tags.push(ancestor);
        tags.extend_from_slice(&abandoned[..abandoned.len() - 1]);
        Ok(Self { tags })
    }

    /// The one-block lineage, which needs nothing about the block it gives back.
    ///
    /// The depth-1 undo consumes exactly the ancestor's own generation, so there is no second tag
    /// to state and no hash for the abandoned block to check. Kept as its own constructor because
    /// the callers that have only a target hash — the ExEx's notification hook among them — would
    /// otherwise have to invent one.
    pub fn depth_one(ancestor: (u64, B256)) -> Self {
        Self { tags: vec![ancestor] }
    }

    /// The block the pair lands on.
    pub fn ancestor(&self) -> (u64, B256) {
        self.tags[0]
    }

    /// How many blocks are given back, which is also how many generations are consumed.
    pub fn depth(&self) -> u64 {
        self.tags.len() as u64
    }

    /// The `i`-th consumed generation's expected `(number, hash)`, ancestor first.
    fn generation_tag(&self, index: usize) -> Option<(u64, B256)> {
        self.tags.get(index).copied()
    }
}

/// Why a proposed lineage could not describe any chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineageError {
    /// A reorg that abandons no block is not a reorg.
    Empty,
    /// The lowest abandoned block is not the ancestor's child.
    NotAboveAncestor { ancestor: u64, lowest: u64 },
    /// The abandoned run skips a height.
    Gap { from: u64, to: u64 },
}

impl std::fmt::Display for LineageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "a lineage that abandons no block is not a reorg"),
            Self::NotAboveAncestor { ancestor, lowest } => write!(
                f,
                "the lowest abandoned block is {lowest} but the common ancestor is {ancestor}"
            ),
            Self::Gap { from, to } => {
                write!(f, "the abandoned blocks jump from {from} to {to}")
            }
        }
    }
}

impl std::error::Error for LineageError {}

/// The one coordinated generation a validator maintains, plus what it is authenticated against.
pub struct CoordinatedPair {
    pub cache: NetworkStateCache,
    pub trie_cache: PartialTrieNodeCache,
    pub readiness: CacheReadinessTracker,
    /// The previous trie generations, oldest at the front, capped at [`Self::retention_depth`].
    ///
    /// Retention is free at any depth, which is the correction this deque carries. The comment
    /// this replaces said "any K beyond 1 would need genuinely extra copies" — it does not. The
    /// transition already copies the parent trie every block and then overwrites the copy, and
    /// `make_mut` copies only when the handle is `Shared` *and* its strong count exceeds one,
    /// which one retained parent already makes true. Raising K adds no copies; it adds only the
    /// memory of not dropping them, which is why the depth is a runtime setting and not a
    /// compile-time constant.
    ///
    /// A generation is tagged with the block it is the state *after*, so the back of the deque is
    /// the pair's parent and the front is `retention_depth - 1` blocks below that.
    pub retained: VecDeque<RetainedGeneration>,
    /// How deep this pair retains, and therefore how deep a reorg it can undo alone.
    ///
    /// The single owner. The deque cap above, the flat undo log's prune depth, and the depth at
    /// which a reorg is refused all read this one field rather than keeping constants of their own.
    pub retention_depth: RetentionDepth,
    /// Header of the block this pair is the state *after*, kept so a child can be checked against
    /// it.
    ///
    /// Read through [`Self::accepted_parent`] rather than directly. Parent-dependent consensus —
    /// number and parent-hash linkage, timestamp monotonicity, the gas-limit ramp, the EIP-1559
    /// base fee, EIP-4844 blob gas — needs the whole parent header, and the readiness tracker
    /// keeps only a number and a hash. This is the validator's *own* record of what it accepted;
    /// a header offered alongside the block being validated is the producer describing the
    /// standard it wants to be held to.
    pub accepted_head: Option<SealedHeader>,
}

impl CoordinatedPair {
    pub fn fingerprint(&self) -> CoordinatedFingerprint {
        CoordinatedFingerprint {
            cache_block: self.cache.current_block(),
            cache_root: self.cache.cache_root(),
            trie_cache_root: self.trie_cache.cache_root(),
            trie_state_root: self.trie_cache.state_root(),
        }
    }

    /// The parent header a child block may be validated against, if this pair can vouch for one.
    ///
    /// Never trusted from the field. Any path that replaces the caches without replacing the
    /// header — a canonical rebuild, a snapshot restore, an undo that rolled back and then failed
    /// — would otherwise leave a header describing a generation this pair is no longer at, and a
    /// child checked against it would be checked against a parent that never was.
    ///
    /// **Height alone is not enough, and that is the whole reason this is a method.** A canonical
    /// rebuild installs the winning sibling at the *same* number the abandoned one had, so a
    /// height check would hand back the abandoned header while the caches hold the winner: parent
    /// consensus measured against one branch, execution against the other. So the header is
    /// checked against everything the readiness tracker independently authenticated — the anchor's
    /// hash and number, the cache root, the trie state root — and against the trie cache itself.
    /// Only a header that agrees with all of them can be the one these caches are the state after.
    ///
    /// Requiring `Ready` falls out of that and is correct on its own terms: a warming or
    /// recovering pair has no authenticated parent to offer, and admitting untrusted input
    /// against a guess is exactly what section 4.2 exists to forbid. Absence is a rejection.
    pub fn accepted_parent(&self) -> Option<&SealedHeader> {
        let header = self.accepted_head.as_ref()?;
        let ready = self.readiness.ready_parent()?;
        (header.number() == self.cache.current_block() &&
            header.number() == ready.anchor.block_number &&
            header.hash() == ready.anchor.block_hash &&
            header.state_root() == ready.trie_state_root &&
            self.cache.cache_root() == ready.anchor.cache_root &&
            self.trie_cache.state_root() == Some(header.state_root()))
        .then_some(header)
    }

    /// What two pairs must agree on to be at the same point in the chain's *history*.
    ///
    /// Separate from [`Self::fingerprint`], which answers whether two pairs hold the same cache
    /// generation. A snapshot reproduces the caches without reproducing how they were reached, so
    /// a restored pair legitimately has no accepted head and no retained generation while being
    /// cache-identical to the pair it came from. Folding these fields into the cache fingerprint
    /// would make the bootstrap gate fail on that difference, which is not the difference it
    /// exists to catch.
    pub fn lifecycle_fingerprint(&self) -> LifecycleFingerprint {
        LifecycleFingerprint {
            accepted_head: self
                .accepted_head
                .as_ref()
                .map(|header| (header.number(), header.hash())),
            // The newest generation only, which is exactly what the single-slot field meant. The
            // bootstrap gate compares two pairs' *positions*, and a pair's position is its parent
            // — how many further generations it happens to hold behind that is a memory setting,
            // not a difference in where it is.
            retained_generation: self
                .retained
                .back()
                .map(|retained| (retained.block_number, retained.block_hash)),
        }
    }

    /// Record the trie generation a committed block displaced, so the block can be undone.
    ///
    /// `None` means the transition did not commit, in which case the current trie cache is still
    /// the parent and there is nothing new to keep. The old retention is dropped either way: it
    /// described a generation two blocks back, which K = 1 does not promise to reach.
    pub fn retain_generation(
        &mut self,
        displaced: Option<PartialTrieNodeCache>,
        block_hash: B256,
        block_number: u64,
        accepted_head: SealedHeader,
        enabled: bool,
    ) {
        // Taken before the retention is rebuilt so the generation being kept carries the header it
        // was accepted under. An undo has to restore both together: rolling the caches back to the
        // parent while leaving the child's header in place would validate the replacement block
        // against the very block the reorg discarded.
        //
        // Unconditional, unlike the trie retention below. The accepted head is what parent-checks
        // run against on the *next* block, so it advances whether or not this run retains for
        // reorgs; the K = 1 memory control turns off retention, not admission.
        let displaced_accepted_head = self.accepted_head.replace(accepted_head);
        // Dropping `displaced` here rather than declining to produce it is deliberate: the
        // transition still copies the parent trie and still hands the copy back, so the control
        // arm pays exactly the work the production arm pays and differs only in what it keeps.
        // A control that also skipped the copy would be measuring two changes at once.
        let Some(trie_cache) = enabled.then_some(displaced).flatten() else {
            // Retention is off, or the transition did not commit. Either way this pair can no
            // longer vouch for an unbroken run of generations down from its parent, and a deque
            // with a hole at its newest end is worse than an empty one: a depth-D undo would walk
            // straight past the gap. The K = 1 form dropped its single slot here for the same
            // reason, stated as "a generation two blocks back, which K = 1 does not promise".
            self.retained.clear();
            return
        };
        self.retained.push_back(RetainedGeneration {
            trie_cache,
            block_hash,
            block_number,
            accepted_head: displaced_accepted_head,
        });
        while self.retained.len() > self.retention_depth.as_usize() {
            self.retained.pop_front();
        }
    }

    /// Installs the displaced generation and records the transition with readiness, as one step.
    ///
    /// Every consumer that applies a block ends here — the ExEx builder, the ExEx verifier, and the
    /// standalone replay driver — and it is one method rather than a sequence each of them repeats
    /// because the two halves are not independent. Retention has to see the caches before readiness
    /// is told the block finished, and a caller that got that order wrong would retain a generation
    /// readiness had already moved past. Returns the readiness label after the transition, which is
    /// the only part a run log wants and the only part a validator with no run log ignores.
    pub fn commit_transition(
        &mut self,
        displaced: Option<PartialTrieNodeCache>,
        block: &BlockContext,
        accepted_head: SealedHeader,
        retain: bool,
    ) -> &'static str {
        self.retain_generation(
            displaced,
            block.parent_hash,
            block.number.saturating_sub(1),
            accepted_head,
            retain,
        );
        let observation = CacheObservation::capture(&self.cache, &self.trie_cache);
        self.readiness.finish_block(block, &observation).label()
    }

    /// What the retained generation costs right now, for the K = 1 memory control.
    ///
    /// Read before a block is built, so it describes the generation the *previous* block
    /// displaced — the steady state a run spends every block in, rather than the instant after a
    /// transition when the live cache has not yet diverged from it.
    pub fn retained_generation_bytes(&self, enabled: bool) -> RetainedGenerationBytes {
        let Some(retained) = self.retained.back() else {
            return RetainedGenerationBytes { enabled, ..Default::default() }
        };
        let breakdown = retained.trie_cache.memory_breakdown();
        RetainedGenerationBytes {
            enabled,
            present: true,
            total_bytes: retained.trie_cache.estimated_memory_bytes(),
            exclusive_bytes: retained.trie_cache.exclusive_memory_bytes(),
            complete_total_bytes: breakdown.total_bytes(),
            complete_exclusive_bytes: breakdown.exclusive_bytes(),
            breakdown,
        }
    }

    /// Drop the retained generation because the pair no longer descends from it.
    ///
    /// Called wherever the pair is replaced wholesale — cold reset, snapshot restore, canonical
    /// rebuild. The arithmetic and hash checks in [`Self::restore_retained_generation`] would
    /// reject a stale retention anyway; clearing it is the cheaper, more obvious guard.
    pub fn forget_retained_generations(&mut self) {
        self.retained.clear();
    }

    /// The newest retained generation, which is the pair's parent.
    ///
    /// The read the single-slot field used to serve directly. Kept as a method so callers that
    /// only ever wanted "the parent" do not have to know the deque exists.
    pub fn retained_generation(&self) -> Option<&RetainedGeneration> {
        self.retained.back()
    }

    /// How many generations are held right now, which is at most [`Self::retention_depth`].
    pub fn retained_depth(&self) -> u64 {
        self.retained.len() as u64
    }

    /// Return both caches and the tracker to their empty state, keeping nothing.
    ///
    /// The mutation only. Whether cold-resetting is an acceptable answer to a gap is the caller's
    /// policy: a full node can warm again from live blocks, and a standalone validator cannot,
    /// which is why no decision is taken here.
    pub fn cold_reset(&mut self) {
        // Cold means empty, not reconfigured: the pair keeps running on whatever trie
        // representation it was constructed with, exactly as a fresh process would build it.
        self.trie_cache = PartialTrieNodeCache::new_with_repr(self.trie_cache.repr());
        self.cache.reset();
        self.readiness.reset();
        self.forget_retained_generations();
        // A pair that has accepted nothing has no parent to offer. `accepted_parent` would refuse
        // a stale header anyway once the cache height drops to zero; clearing it is the honest
        // representation rather than one the guard happens to catch.
        self.accepted_head = None;
    }

    /// Undo exactly one committed block, returning the pair to `target_hash`.
    ///
    /// This is the fast path for a depth-1 reorg, and it is a transaction: the pair ends at the
    /// parent generation or at the child, never between them. Every check — including the
    /// readiness tracker's, which runs against a copy of the tracker and a prediction of what the
    /// caches will report — happens before the first mutation, and the mutations that follow
    /// cannot be refused. That matters here and not for the full node that first needed it: a
    /// caller with a database can replace a half-restored pair wholesale, and a standalone
    /// validator has nothing to replace it with.
    ///
    /// What a rejection preserves, stated exactly: both caches, the readiness tracker, and the
    /// accepted head are untouched. The retained generation is *not* always kept — a retention
    /// tagged with a different block is dropped, because the caller has just named a canonical
    /// target it does not describe and nothing will ask for it again.
    ///
    /// `target_state_root` must come from the canonical header for `target_hash`. Comparing the
    /// retained trie's own root against it is what makes this an authentication rather than a
    /// tautology — the same reason installing a rebuilt pair leans on the header's state root
    /// rather than on the self-derived cache root.
    /// The depth-1 restore, kept at its original signature.
    ///
    /// Every caller that has a target hash and nothing else routes through here: the ExEx's
    /// notification hook, and the K = 1 suite that is this change's own regression test. It states
    /// the one-block lineage from the pair's own newest generation and hands it to the general
    /// form, so "K = 1 behaves exactly as it did" is a property of there being one implementation
    /// rather than of two implementations agreeing.
    pub fn restore_retained_generation(
        &mut self,
        target_hash: B256,
        target_state_root: B256,
        cache_policy_id: B256,
    ) -> Option<ReadyParent> {
        let ancestor_number = self.retained_generation()?.block_number;
        let lineage = ExpectedLineage::depth_one((ancestor_number, target_hash));
        self.restore_retained_generations(&lineage, target_state_root, cache_policy_id)
    }

    pub fn restore_retained_generations(
        &mut self,
        lineage: &ExpectedLineage,
        ancestor_state_root: B256,
        cache_policy_id: B256,
    ) -> Option<ReadyParent> {
        let depth = lineage.depth();
        let (ancestor_number, ancestor_hash) = lineage.ancestor();

        // ---- phase 1: read-only ------------------------------------------------------------
        // Every check below reads, and the tracker's verdict is taken on a copy. Nothing is
        // consumed until the commit marker, because a standalone validator has nothing to replace
        // a half-restored pair with.
        if depth > self.retention_depth.get() {
            debug!(
                target: "partial_stateless",
                depth,
                retention_depth = %self.retention_depth,
                "A reorg deeper than this pair retains cannot be undone from its own generations"
            );
            return None
        }
        let Some(base) = self.retained.len().checked_sub(depth as usize) else {
            debug!(
                target: "partial_stateless",
                depth,
                held = self.retained.len(),
                "The pair holds fewer generations than the reorg gives back; rebuilding"
            );
            return None
        };

        // Checked before the cheap hash checks are even worth running: a pair that is still
        // warming has no `Ready` to return to, so undoing into it would trade a rebuild that
        // genuinely fills the window for a claim nothing backs. Ordered ahead of the lineage check
        // on purpose — this refusal keeps the deque, and the lineage refusal below destroys it.
        if !self.readiness.stays_warm_after_undo(depth) {
            debug!(
                target: "partial_stateless",
                depth,
                replay_depth = self.readiness.replay_depth(),
                required = self.readiness.required_replay_depth(),
                "Pair is still warming, so undoing {depth} blocks cannot restore Ready; rebuilding"
            );
            return None
        }

        // The deque against the lineage, oldest of the run first. A generation is tagged with the
        // block it is the state after, so the run consumed by a depth-D undo is tagged
        // `ancestor ..= ancestor + D - 1`: the ancestor itself, plus every abandoned block except
        // the tip, whose generation is the live cache rather than a retained one.
        for offset in 0..depth as usize {
            let held = &self.retained[base + offset];
            let expected = lineage.generation_tag(offset).expect("offset is below the depth");
            if (held.block_number, held.block_hash) != expected {
                debug!(
                    target: "partial_stateless",
                    held_block = held.block_number,
                    held_hash = ?held.block_hash,
                    expected_block = expected.0,
                    expected_hash = ?expected.1,
                    "A retained generation does not match the lineage; falling back to a rebuild"
                );
                // The whole deque, not the mismatching part. Every generation above the mismatch
                // describes the branch the caller has just been told is not canonical, and the
                // ones below cannot be reached without walking through it — so partial truncation
                // would leave a run whose newest end nothing vouches for. The caches, the undo log
                // and the tracker are still exactly as they were.
                self.forget_retained_generations();
                return None
            }
        }

        // The flat side, proved to the same depth and mutating nothing. This is where a pruned
        // middle record or a missing memoized root is caught.
        let plan = match self.cache.can_rollback_to(ancestor_number) {
            Ok(plan) => plan,
            Err(err) => {
                debug!(
                    target: "partial_stateless",
                    ancestor = ancestor_number,
                    %err,
                    "The flat undo log cannot give back the reorg's depth; rebuilding"
                );
                return None
            }
        };
        if plan.depth() != depth {
            // The two halves disagree about how many blocks separate the pair from the ancestor,
            // which means one of them is describing a chain the other never applied.
            warn!(
                target: "partial_stateless",
                trie_depth = depth,
                flat_depth = plan.depth(),
                "The flat undo log and the reorg disagree about the depth; rebuilding"
            );
            return None
        }

        let landed_root = self.retained[base].trie_cache.state_root();
        if landed_root != Some(ancestor_state_root) {
            warn!(
                target: "partial_stateless",
                block = ancestor_number,
                retained_state_root = ?landed_root,
                canonical_state_root = ?ancestor_state_root,
                "Retained generation does not match the canonical state root at its own block; \
                 falling back to a rebuild"
            );
            return None
        }

        let previous_cache_root = plan.previous_cache_root();
        let checkpoint = TrustedCheckpoint {
            block_number: ancestor_number,
            block_hash: ancestor_hash,
            state_root: ancestor_state_root,
            cache_root: previous_cache_root,
            cache_policy_id,
        };
        // Exactly what `CacheObservation::capture` will report once the commit below runs: the
        // rollback restores `previous_block` and the memoized root verbatim, and the trie is
        // replaced by the landed generation whose root was just checked. So the tracker's answer
        // here is its answer there, taken while a refusal still costs nothing.
        let predicted = CacheObservation {
            cache_block: plan.previous_block(),
            cache_root: previous_cache_root,
            trie_state_root: landed_root,
        };
        let mut next_readiness = self.readiness.clone();
        let ready = match next_readiness.restore_from_undone_blocks(depth, &checkpoint, &predicted)
        {
            Ok(ready) => ready.clone(),
            Err(err) => {
                warn!(
                    target: "partial_stateless",
                    block = ancestor_number,
                    ?err,
                    "Readiness rejected the restored generation; falling back to a rebuild"
                );
                return None
            }
        };

        // ---- phase 2: commit, infallible ---------------------------------------------------
        // `rollback` cannot refuse: the plan proved the records present and contiguous, and the
        // borrow checker guarantees nothing touched the log between building it and consuming it.
        self.cache.rollback(plan);
        // Split rather than looped. The generations above the landing one are dropped and the ones
        // below are kept, so a second, shallower undo can still run against what this one left —
        // and the landing generation is identified by index rather than by counting pops, which is
        // where a D-times loop gets the off-by-one wrong.
        let mut given_back = self.retained.split_off(base);
        let landed = given_back.pop_front().expect("the lineage check proved this index is held");
        self.trie_cache = landed.trie_cache;
        // Restored together with the caches. Between here and the tracker swap the pair holds the
        // ancestor's header over the ancestor's caches, and both name the same generation.
        self.accepted_head = landed.accepted_head;
        self.readiness = next_readiness;
        Some(ready)
    }
}

/// What two coordinated pairs must agree on to be the same generation.
///
/// `trie_cache_root` commits the trie's state root together with its retained-path membership, so
/// comparing it covers "retained paths" without walking them. `cache_root` hashes every flat
/// value *and* its `last_accessed_block`, which is the only complete check on the replay metadata
/// a state proof cannot attest to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize)]
pub struct CoordinatedFingerprint {
    pub cache_block: u64,
    pub cache_root: B256,
    pub trie_cache_root: B256,
    pub trie_state_root: Option<B256>,
}

/// One previous trie generation, tagged with the block it is the state *after*.
///
/// The tag is a hash rather than a number on purpose: mid-reorg a height names whichever block the
/// database currently calls canonical, which is the failure the whole recovery path exists to
/// avoid. `NetworkStateCache` needs no counterpart here — its undo log already reaches finality.
pub struct RetainedGeneration {
    pub trie_cache: PartialTrieNodeCache,
    pub block_hash: B256,
    pub block_number: u64,
    /// Accepted head as of this generation, restored with it by a depth-1 undo.
    ///
    /// `None` when the generation predates any accepted header — a pair one block out of a cold
    /// reset retains a trie it has no header for. Undoing into that is sound: the pair is warming,
    /// and [`CoordinatedPair::accepted_parent`] then reports absence rather than a guess.
    pub accepted_head: Option<SealedHeader>,
}

/// What two coordinated pairs must agree on to have reached the same point the same way.
///
/// Deliberately not part of [`CoordinatedFingerprint`]. That one answers "are these the same cache
/// generation", which a snapshot restore reproduces exactly; this one answers "did they get here
/// by applying the same blocks", which a snapshot restore by construction does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize)]
pub struct LifecycleFingerprint {
    /// Number and hash of the block this pair is the state after.
    pub accepted_head: Option<(u64, B256)>,
    /// Number and hash of the block the retained generation is the state after.
    pub retained_generation: Option<(u64, B256)>,
}

/// What the K = 1 retained generation was holding when a block began.
///
/// `total_bytes` is what the retained trie cache measures on its own; `exclusive_bytes` is the
/// part of it that no other generation shares, which is what dropping it would give back. The two
/// are reported together because the gap between them is the point: a snapshot shares storage
/// tries with its parent, so the cost of keeping one is far below its apparent size, and only the
/// exclusive figure is comparable with a resident-memory difference.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct RetainedGenerationBytes {
    /// Whether this run retains at all. False is the memory control, not a failure.
    pub enabled: bool,
    /// Whether a generation was actually being held. False while cold, warming, or recovering.
    pub present: bool,
    /// The sparse trie alone, which is what every published cohort figure means. Kept at that
    /// definition so those rows stay comparable; `complete_total_bytes` is the honest number.
    pub total_bytes: usize,
    /// `total_bytes` less the storage tries another generation also holds.
    pub exclusive_bytes: usize,
    /// The sparse trie *plus* the warm sets and the retained-path indexes.
    ///
    /// Always at least `total_bytes`. The gap is what a retained generation costs beyond the trie
    /// and what no consumer measurement has previously included.
    pub complete_total_bytes: usize,
    /// `complete_total_bytes` less the storage tries another generation also holds.
    pub complete_exclusive_bytes: usize,
    /// Where the two complete figures came from.
    pub breakdown: TrieCacheMemory,
}

/// The one canonical-chain question depth-1 recovery has to ask.
///
/// Narrow on purpose. Recovery needs to know the state root of the block it is returning to, and
/// that is checked against the canonical header rather than against anything the pair derived
/// itself; a fake chain that answers this one question is therefore enough to exercise the whole
/// path, which is what [`inject_recovery`] and the equivalence gate rely on. It is also the whole
/// external surface of recovery, so a standalone validator can satisfy it from headers alone.
pub trait CanonicalStateRoots {
    /// `None` means there is no canonical header for `hash`, which is a rejection, not an error.
    fn state_root_of(&self, hash: B256) -> ProviderResult<Option<B256>>;
}

/// Drives the recovery half of a `ChainReorged` or `ChainReverted` notification.
///
/// This is the notification-injection hook. Both handlers do exactly this — mark the tracker
/// `Recovering` at the first unwound height, then attempt the depth-1 undo — and everything after
/// it differs only in whether a new branch follows. Mainnet produces the notification that reaches
/// this code roughly once a day and never at a depth the test chooses, so the gate on recovery
/// *equivalence* cannot be a live observation; injecting the notification against a chain the test
/// controls is what makes it a gate.
///
/// The fallback is deliberately on the caller's side. A full node rebuilds from its database; a
/// standalone validator has none, so the path this function covers is its only recovery.
pub fn inject_recovery(
    pair: &mut CoordinatedPair,
    chain: &impl CanonicalStateRoots,
    unwound_from: u64,
    target_hash: B256,
    cache_policy_id: B256,
) -> Option<ReadyParent> {
    pair.readiness.begin_recovery(unwound_from);
    try_depth_one_recovery(pair, chain, target_hash, cache_policy_id)
}

/// Undoes exactly one block to return the pair to `target_hash`, or `None` to fall back.
///
/// The depth-1 entry point, kept because the ExEx's notification hook has a target hash and no
/// list of abandoned blocks. It reads the ancestor's height off the pair's own newest generation —
/// the same thing the single-slot form did — and states the rest as a one-block lineage.
///
/// Split out so that everything between a notification and the restored pair can run against a
/// fake chain. Nothing here touches a database, and the fallback — which for a full node does — is
/// deliberately left on the caller's side of the seam.
pub fn try_depth_one_recovery(
    pair: &mut CoordinatedPair,
    chain: &impl CanonicalStateRoots,
    target_hash: B256,
    cache_policy_id: B256,
) -> Option<ReadyParent> {
    let ancestor_number = pair.retained_generation()?.block_number;
    let lineage = ExpectedLineage::depth_one((ancestor_number, target_hash));
    try_deep_recovery(pair, chain, &lineage, cache_policy_id)
}

/// Undoes the whole of `lineage` to return the pair to its ancestor, or `None` to fall back.
///
/// The general form. `lineage` states which blocks are being given back and which generations the
/// pair must be holding for them; the pair checks both halves against its own deque and undo log
/// before it moves anything, so a refusal here costs a rebuild and never a half-restored pair.
///
/// The canonical state root is looked up here rather than passed in, so that the one external fact
/// a recovery depends on — what the canonical header says the ancestor's state root is — enters
/// through a single seam that a fake chain can stand in for.
pub fn try_deep_recovery(
    pair: &mut CoordinatedPair,
    chain: &impl CanonicalStateRoots,
    lineage: &ExpectedLineage,
    cache_policy_id: B256,
) -> Option<ReadyParent> {
    let (ancestor_number, ancestor_hash) = lineage.ancestor();
    let state_root = match chain.state_root_of(ancestor_hash) {
        Ok(Some(state_root)) => state_root,
        Ok(None) => {
            debug!(
                target: "partial_stateless",
                target_hash = ?ancestor_hash,
                "No canonical header for the recovery target; rebuilding"
            );
            return None
        }
        Err(err) => {
            debug!(
                target: "partial_stateless",
                target_hash = ?ancestor_hash,
                %err,
                "Could not read the recovery target's header; rebuilding"
            );
            return None
        }
    };

    let depth = lineage.depth();
    let started = Instant::now();
    let ready = pair.restore_retained_generations(lineage, state_root, cache_policy_id)?;
    info!(
        target: "partial_stateless",
        block = ready.anchor.block_number,
        block_hash = ?ready.anchor.block_hash,
        ancestor = ancestor_number,
        depth,
        restore_us = started.elapsed().as_micros() as u64,
        "Recovered by undoing {depth} block(s) from the retained generations instead of rebuilding"
    );
    Some(ready)
}

/// Reports whether a block may be applied, without repairing anything.
pub fn admit_block(readiness: &mut CacheReadinessTracker, block: &BlockContext) -> BlockAdmission {
    // Captured before admission: applying a block moves the tracker to `Applying`, and the token
    // describes the parent this block builds on, not the block itself.
    let ready_parent = readiness.ready_parent().cloned();
    match readiness.begin_block(block) {
        Ok(()) => BlockAdmission::Admitted(ready_parent),
        Err(reason) => BlockAdmission::Rejected(reason),
    }
}

/// Whether a block may be applied, and what the caches were authenticated against beforehand.
#[derive(Debug)]
pub enum BlockAdmission {
    /// The block may be applied. `Some` carries the parent that partial output may be published
    /// against; `None` means the caches are not Ready and may only produce local measurements.
    Admitted(Option<ReadyParent>),
    /// The block must not be applied, and why.
    Rejected(BlockedReason),
}

/// Describes a canonical block for the readiness tracker.
pub fn block_context(block: &RecoveredBlock<BlockTy<EthPrimitives>>) -> BlockContext {
    BlockContext {
        number: block.number(),
        hash: block.hash(),
        parent_hash: block.parent_hash,
        state_root: block.state_root(),
    }
}
