//! The coordinated generation a partial-stateless validator maintains, and how it moves.
//!
//! One value cache and one trie cache advance together or not at all, authenticated by a readiness
//! tracker. This module owns that state and the operations that are protocol rather than policy:
//! taking its fingerprint, retaining and restoring the one previous trie generation a depth-1
//! reorg needs, and reporting whether a block may be applied.
//!
//! What is deliberately *not* here is every fallback that needs a state database. A full node that
//! cannot undo one block rebuilds from canonical state or cold-resets and rewarms; a standalone
//! validator can do neither, and per the Phase 4b plan must instead request a snapshot at the exact
//! common ancestor. Keeping those fallbacks on the caller's side of the boundary is what makes the
//! difference explicit rather than a branch this module could accidentally grow.
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
    PartialTrieNodeCache,
};
use reth_ethereum_primitives::EthPrimitives;
use reth_primitives_traits::{AlloyBlockHeader, BlockTy, RecoveredBlock, SealedHeader};
/// Re-exported so [`CanonicalStateRoots`] can be implemented outside this crate without naming
/// the error crate: a standalone consumer answers the trait from its own verified history, and
/// making it add a dependency to spell the return type would be a boundary that means nothing.
pub use reth_storage_errors::provider::ProviderResult;
use serde::Serialize;
use std::time::Instant;
use tracing::{debug, error, info, warn};

/// The one coordinated generation a validator maintains, plus what it is authenticated against.
pub struct CoordinatedPair {
    pub cache: NetworkStateCache,
    pub trie_cache: PartialTrieNodeCache,
    pub readiness: CacheReadinessTracker,
    /// The single previous trie generation, kept so a depth-1 reorg does not need a full rebuild.
    ///
    /// K is 1 because that is the depth at which retention is free: the transition already copies
    /// the parent trie and then overwrites it, so keeping the displaced copy costs no extra work.
    /// Any K beyond 1 would need genuinely extra copies, and a deeper reorg falls back to whatever
    /// the caller has — a rebuild for a full node, a snapshot request for a standalone validator.
    pub previous_generation: Option<RetainedGeneration>,
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
            retained_generation: self
                .previous_generation
                .as_ref()
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
        self.previous_generation =
            enabled.then_some(displaced).flatten().map(|trie_cache| RetainedGeneration {
                trie_cache,
                block_hash,
                block_number,
                accepted_head: displaced_accepted_head,
            });
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
        let Some(retained) = &self.previous_generation else {
            return RetainedGenerationBytes { enabled, ..Default::default() }
        };
        RetainedGenerationBytes {
            enabled,
            present: true,
            total_bytes: retained.trie_cache.estimated_memory_bytes(),
            exclusive_bytes: retained.trie_cache.exclusive_memory_bytes(),
        }
    }

    /// Drop the retained generation because the pair no longer descends from it.
    ///
    /// Called wherever the pair is replaced wholesale — cold reset, snapshot restore, canonical
    /// rebuild. The arithmetic and hash checks in [`Self::restore_retained_generation`] would
    /// reject a stale retention anyway; clearing it is the cheaper, more obvious guard.
    pub fn forget_retained_generation(&mut self) {
        self.previous_generation = None;
    }

    /// Return both caches and the tracker to their empty state, keeping nothing.
    ///
    /// The mutation only. Whether cold-resetting is an acceptable answer to a gap is the caller's
    /// policy: a full node can warm again from live blocks, and a standalone validator cannot,
    /// which is why no decision is taken here.
    pub fn cold_reset(&mut self) {
        self.trie_cache = PartialTrieNodeCache::new();
        self.cache.reset();
        self.readiness.reset();
        self.forget_retained_generation();
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
    pub fn restore_retained_generation(
        &mut self,
        target_hash: B256,
        target_state_root: B256,
        cache_policy_id: B256,
    ) -> Option<ReadyParent> {
        // Everything down to the commit marker below reads and never writes, and the tracker's own
        // verdict is taken on a copy — so the mutations, once they start, cannot be refused
        // half-way. The retention itself is not consumed until then.
        let (retained_hash, retained_number, retained_root) = {
            let retained = self.previous_generation.as_ref()?;
            (retained.block_hash, retained.block_number, retained.trie_cache.state_root())
        };
        // Checked before anything is mutated, and before the cheap hash checks are even worth
        // running: a pair that is still warming has no `Ready` to return to, so undoing into it
        // would trade a rebuild that genuinely fills the window for a claim nothing backs.
        if !self.readiness.stays_warm_after_one_undo() {
            debug!(
                target: "partial_stateless",
                replay_depth = self.readiness.replay_depth(),
                required = self.readiness.required_replay_depth(),
                "Pair is still warming, so undoing one block cannot restore Ready; rebuilding"
            );
            return None
        }
        if retained_hash != target_hash {
            debug!(
                target: "partial_stateless",
                retained_block = retained_number,
                retained_hash = ?retained_hash,
                ?target_hash,
                "Retained generation belongs to a different block; falling back to a rebuild"
            );
            // The one lifecycle change a refusal makes, and it is deliberate: this retention
            // describes a branch the caller has just been told is not canonical, so nothing will
            // ask for it again. The caches and the tracker are still exactly as they were.
            self.forget_retained_generation();
            return None
        }
        // Only depth 1. The flat undo log reaches further, but the trie does not, and the pair has
        // to move as one generation.
        let undone = self.cache.current_block();
        if undone != retained_number + 1 {
            return None
        }
        if retained_root != Some(target_state_root) {
            warn!(
                target: "partial_stateless",
                block = retained_number,
                retained_state_root = ?retained_root,
                canonical_state_root = ?target_state_root,
                "Retained generation does not match the canonical state root at its own block; \
                 falling back to a rebuild"
            );
            return None
        }

        // What the rollback will install, taken from the record it will consume. Comparing it here
        // is the same check `rollback_block` would make, moved ahead of the mutation, and it is
        // what lets the commit below treat the rollback as infallible.
        let Some(preview) = self.cache.undo_preview() else {
            debug!(
                target: "partial_stateless",
                block = undone,
                "No undo record to give back; falling back to a rebuild"
            );
            return None
        };
        if preview.block_number != undone || preview.previous_block != retained_number {
            debug!(
                target: "partial_stateless",
                block = undone,
                undo_block = preview.block_number,
                undo_previous = preview.previous_block,
                retained_block = retained_number,
                "The newest undo record does not describe the block the retention undoes"
            );
            return None
        }
        // The post-undo cache root has to be known *before* the undo for this to be a transaction,
        // and the record carries it only when the parent's root had already been computed. Every
        // path that reaches here in production computes it every block — the coordinated
        // fingerprint and the ready parent's anchor both do. A pair that applied a block without
        // ever rooting its parent falls back instead, which is the fail-closed direction.
        let Some(previous_cache_root) = preview.previous_cache_root else {
            debug!(
                target: "partial_stateless",
                block = undone,
                "The parent's cache root was never computed, so the undo cannot be made atomic; \
                 falling back to a rebuild"
            );
            return None
        };

        let checkpoint = TrustedCheckpoint {
            block_number: retained_number,
            block_hash: retained_hash,
            state_root: target_state_root,
            cache_root: previous_cache_root,
            cache_policy_id,
        };
        // Exactly what `CacheObservation::capture` will report once the commit below runs: the
        // rollback restores `previous_block` and the memoized root verbatim, and the trie is
        // replaced by the retained one whose root was just checked. So the tracker's answer here
        // is its answer there, taken while a refusal still costs nothing.
        let predicted = CacheObservation {
            cache_block: preview.previous_block,
            cache_root: previous_cache_root,
            trie_state_root: retained_root,
        };
        let mut next_readiness = self.readiness.clone();
        let ready = match next_readiness.restore_from_undone_block(&checkpoint, &predicted) {
            Ok(ready) => ready.clone(),
            Err(err) => {
                warn!(
                    target: "partial_stateless",
                    block = retained_number,
                    ?err,
                    "Readiness rejected the restored generation; falling back to a rebuild"
                );
                return None
            }
        };

        // ---- commit ----
        if let Err(err) = self.cache.rollback_block(undone) {
            // Unreachable: the preview above named this exact block, and nothing between then and
            // now touches the undo log. Reported rather than asserted because a validator that got
            // here has a broken invariant, not a block to reject — and the early return is still
            // sound, since `rollback_block` refuses before it pops.
            error!(
                target: "partial_stateless",
                block = undone,
                ?err,
                "Flat rollback refused a block its own undo record named; falling back to a rebuild"
            );
            return None
        }
        let retained = self.previous_generation.take().expect("checked present above");
        self.trie_cache = retained.trie_cache;
        // Restored together with the caches. Between here and the tracker swap the pair holds the
        // parent's header over the parent's caches, and both name the same generation.
        self.accepted_head = retained.accepted_head;
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
    pub total_bytes: usize,
    pub exclusive_bytes: usize,
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
/// Split out so that everything between a notification and the restored pair can run against a
/// fake chain. Nothing here touches a database, and the fallback — which for a full node does — is
/// deliberately left on the caller's side of the seam.
pub fn try_depth_one_recovery(
    pair: &mut CoordinatedPair,
    chain: &impl CanonicalStateRoots,
    target_hash: B256,
    cache_policy_id: B256,
) -> Option<ReadyParent> {
    let state_root = match chain.state_root_of(target_hash) {
        Ok(Some(state_root)) => state_root,
        Ok(None) => {
            debug!(
                target: "partial_stateless",
                ?target_hash,
                "No canonical header for the recovery target; rebuilding"
            );
            return None
        }
        Err(err) => {
            debug!(
                target: "partial_stateless",
                ?target_hash,
                %err,
                "Could not read the recovery target's header; rebuilding"
            );
            return None
        }
    };

    let started = Instant::now();
    let ready = pair.restore_retained_generation(target_hash, state_root, cache_policy_id)?;
    info!(
        target: "partial_stateless",
        block = ready.anchor.block_number,
        block_hash = ?ready.anchor.block_hash,
        restore_us = started.elapsed().as_micros() as u64,
        "Recovered by undoing one block from the retained generation instead of rebuilding"
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
