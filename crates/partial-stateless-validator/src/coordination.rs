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

use alloy_primitives::{
    map::{HashMap, HashSet},
    B256,
};
use partial_stateless::{
    disk_undo::{DiskUndoBundle, DiskUndoHandle, DiskUndoMetrics, DiskUndoStore},
    network_cache::NetworkStateCache,
    readiness::{
        BlockContext, BlockedReason, CacheObservation, CacheReadinessTracker, ReadyParent,
        TrustedCheckpoint,
    },
    PartialTrieNodeCache, TrieCacheMemory, TrieCacheUndoCounts, TrieCacheUndoFrame,
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
/// A bound rather than a target. Retention memory scales with K, and the production default
/// is 1; this exists so a misconfigured run is
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
    /// The minimal retention depth, also used by in-memory regression controls.
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

/// How retained trie generations are represented when undo recording is enabled.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub enum UndoLayout {
    /// Keep the newest generation whole and demote older generations to frames.
    #[default]
    #[serde(rename = "hybrid")]
    Hybrid,
    /// Turn each committed block's record into a frame immediately.
    #[serde(rename = "frames")]
    FramesOnly,
}

impl UndoLayout {
    /// Stable spelling used by command-line options and run manifests.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Hybrid => "hybrid",
            Self::FramesOnly => "frames",
        }
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
    /// which a reorg is refused all read this one field rather than keeping constants of their
    /// own.
    pub retention_depth: RetentionDepth,
    /// How generations are retained when the live cache records undo data.
    pub undo_layout: UndoLayout,
    /// Optional session-local disk retention. All completed undo payloads live on disk.
    pub undo_store: Option<DiskUndoStore>,
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
    /// Enable the disk-backed frames layout before processing any blocks.
    pub fn enable_disk_undo(&mut self, directory: &std::path::Path) -> Result<(), String> {
        if !self.retained.is_empty() {
            return Err("disk undo must be configured before retaining blocks".into())
        }
        self.undo_store = Some(DiskUndoStore::new(directory)?);
        if let Some(directory) = std::env::var_os("PS_UNDO_METRICS_DIR") {
            self.undo_store
                .as_ref()
                .expect("installed above")
                .enable_metrics(directory.as_ref())?;
        }
        self.undo_layout = UndoLayout::FramesOnly;
        self.trie_cache.set_undo_recording(true);
        Ok(())
    }

    /// Transfer the newest block's trie and flat records together; keep no resident history.
    fn spill_undo(&mut self) {
        let Some(store) = self.undo_store.as_mut() else { return };
        if let Some(held) = self.retained.back() {
            let block = held.block_number + 1;
            let cause = match &held.content {
                RetainedContent::Full(_) => Some("full_fallback"),
                RetainedContent::Frame(_) if self.cache.undo_record(block).is_none() => {
                    Some("missing_flat_record")
                }
                _ => None,
            };
            if let Some(cause) = cause {
                warn!(target: "partial_stateless", cause, block,
                    dropped_generations = self.retained.len(), retained_depth = 0,
                    configured_depth = self.retention_depth.get(),
                    "Undo coverage lost; discarding history behind an unspillable block");
                self.retained.clear();
            } else if matches!(held.content, RetainedContent::Frame(_)) {
                // Decide eligibility before taking either half of the bundle.
                let flat = self.cache.take_undo_record(block).expect("checked above");
                let held = self.retained.pop_back().expect("checked above");
                let RetainedContent::Frame(frame) = held.content else { unreachable!() };
                match store.spill(DiskUndoBundle {
                    parent_hash: held.block_hash,
                    trie: *frame,
                    flat,
                }) {
                    Ok(handle) => self.retained.push_back(RetainedGeneration {
                        content: RetainedContent::Disk(handle),
                        ..held
                    }),
                    Err(error) => {
                        warn!(target: "partial_stateless", cause = "writer_unavailable", %error,
                            block, dropped_generations = self.retained.len() + 1,
                            retained_depth = 0, configured_depth = self.retention_depth.get(),
                            "Undo coverage lost; writer could not accept this block");
                        self.retained.clear();
                    }
                }
            }
        }
        self.cache.prune_undo_below(self.cache.current_block());
    }
    /// Actual retained payloads in RAM, excluding disk handles and transient writer buffers.
    pub fn resident_undo_blocks(&self) -> usize {
        self.retained
            .iter()
            .filter(|held| {
                matches!(held.content, RetainedContent::Full(_) | RetainedContent::Frame(_))
            })
            .count()
    }

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
    /// against a guess would let unauthenticated parent state reach execution. Absence is a
    /// rejection.
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
        block: &BlockContext,
        accepted_head: SealedHeader,
        enabled: bool,
    ) -> CommitUndoReport {
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
        let Some(mut trie_cache) = enabled.then_some(displaced).flatten() else {
            // Retention is off, or the transition did not commit. Either way this pair can no
            // longer vouch for an unbroken run of generations down from its parent, and a deque
            // with a hole at its newest end is worse than an empty one: a depth-D undo would walk
            // straight past the gap. The K = 1 form dropped its single slot here for the same
            // reason, stated as "a generation two blocks back, which K = 1 does not promise".
            self.retained.clear();
            return CommitUndoReport::default()
        };
        // The first cold commit initializes the trie, but its uninitialized parent has no
        // authenticated state to recover. Do not manufacture a Full fallback and report lost
        // coverage when there was no history. Warming commits and authenticated checkpoints
        // still take the normal path, including warnings if a frame cannot be produced.
        if self.undo_store.is_some() &&
            self.undo_layout == UndoLayout::FramesOnly &&
            self.retained.is_empty() &&
            displaced_accepted_head.is_none() &&
            self.readiness.replay_depth() == 0 &&
            trie_cache.state_root().is_none() &&
            self.trie_cache.state_root().is_some()
        {
            self.trie_cache.clear_undo_record();
            self.cache.prune_undo_below(self.cache.current_block());
            info!(target: "partial_stateless", cause = "cold_start", block = block.number,
                retained_depth = 0, configured_depth = self.retention_depth.get(),
                "Initialized disk undo history; no recoverable parent");
            return CommitUndoReport::default()
        }
        let mut retained_head = Some(displaced_accepted_head);
        // A frame always names the generation it restores. The hybrid closes the displaced
        // generation's older record and demotes the deque entry below it. Frames-only closes the
        // live generation's current record against `trie_cache` and retains that parent directly
        // as a frame.
        //
        // Nothing is demoted when there is no record to demote it with — recording off, a
        // `Parallel` cache, a retention pass with no delta shape, or a generation this cache was
        // not cloned from, which is what the deque looks like immediately after an undo. The
        // record is ended either way: its preimages are the block's, and a generation nobody will
        // ask a frame from should not go on holding them.
        //
        // Hybrid does not attempt demotion at depth 1, where the entry it produced would be evicted
        // by the same push. Frames-only retains the frame created by the current commit even at
        // depth 1.
        //
        // Timed to exactly here and no further: ending both journals, dropping the preimages that
        // turned out to describe no change, and the pointer comparison over the storage-trie map.
        // Counting the frame, estimating its bytes and dropping the generation it replaces all
        // happen below, because none of them is assembly and the byte estimate in particular
        // walks every storage trie it holds — a cost the control arm does not pay, which inside
        // this bracket would show up as recording being slower than it is.
        let started = Instant::now();
        let frame = match self.undo_layout {
            UndoLayout::Hybrid => match self.retained.back_mut().map(|held| &mut held.content) {
                Some(RetainedContent::Full(parent)) if self.retention_depth.get() > 1 => {
                    trie_cache.take_undo_frame(parent)
                }
                _ => {
                    trie_cache.clear_undo_record();
                    None
                }
            },
            UndoLayout::FramesOnly => self.trie_cache.take_undo_frame(&mut trie_cache),
        };
        let mut report =
            CommitUndoReport { us: started.elapsed().as_micros() as u64, ..Default::default() };
        let accounting_started = Instant::now();
        if let Some(frame) = frame {
            let (block_number, block_hash) = match self.undo_layout {
                UndoLayout::Hybrid => (block.number.saturating_sub(1), block.parent_hash),
                UndoLayout::FramesOnly => (block.number, block.hash),
            };
            report.frame = Some(CommitUndoFrame {
                counts: frame.counts(),
                // The record's own weight, which is one pass over the entries it holds. Its
                // other half — the previous-version storage tries the frame keeps alive — is a
                // walk of every one of them, and belongs to the memory probe rather than to
                // every block.
                record_bytes: frame.record_bytes(),
                block_number,
                block_hash,
            });
            match self.undo_layout {
                UndoLayout::Hybrid => {
                    let held = self
                        .retained
                        .back_mut()
                        .expect("a hybrid frame is produced against a retained generation");
                    held.content = RetainedContent::Frame(Box::new(frame));
                }
                UndoLayout::FramesOnly => {
                    debug_assert_eq!(frame.target(), trie_cache.undo_id());
                    self.retained.push_back(RetainedGeneration {
                        undo_id: frame.target(),
                        state_root: trie_cache.state_root(),
                        content: RetainedContent::Frame(Box::new(frame)),
                        block_hash: block.parent_hash,
                        block_number: block.number.saturating_sub(1),
                        accepted_head: retained_head.take().expect("the generation head is unused"),
                    });
                }
            }
        }

        report.accounting_us = accounting_started.elapsed().as_micros() as u64;
        let parent_to_drop = if self.undo_layout == UndoLayout::Hybrid || report.frame.is_none() {
            self.retained.push_back(RetainedGeneration::full(
                trie_cache,
                block.parent_hash,
                block.number.saturating_sub(1),
                retained_head.take().expect("the generation head is unused"),
            ));
            None
        } else {
            Some(trie_cache)
        };
        let expiry_started = Instant::now();
        while self.retained.len() > self.retention_depth.as_usize() {
            self.retained.pop_front();
        }
        report.history_expire_us = expiry_started.elapsed().as_micros() as u64;
        let spill_started = Instant::now();
        self.spill_undo();
        report.spill_call_us = spill_started.elapsed().as_micros() as u64;
        // Keep the original lifetime: the displaced parent was dropped after enqueue, so the
        // writer may already be working while its remaining allocations are reclaimed.
        let drop_started = Instant::now();
        drop(parent_to_drop);
        report.parent_drop_us = drop_started.elapsed().as_micros() as u64;
        report
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
    ) -> CommitReport {
        let started = Instant::now();
        let undo = self.retain_generation(displaced, block, accepted_head, retain);
        debug!(target: "partial_stateless", block = block.number,
            retained_depth = self.retained_depth(), resident_blocks = self.resident_undo_blocks(),
            "Observed cache undo retention after commit");
        let observation = CacheObservation::capture(&self.cache, &self.trie_cache);
        let readiness = self.readiness.finish_block(block, &observation).label();
        // The newest handle was enqueued by this call, so it will almost always still be pending
        // at this synchronous observation. Report it separately, then count the contiguous older
        // history already on disk. Starting the suffix at the new handle made the old metric zero
        // even when the writer had completed every preceding bundle.
        let current_undo_written = self.retained.back().and_then(|held| match &held.content {
            RetainedContent::Disk(handle) => Some(handle.is_written()),
            _ => None,
        });
        let completed_prior_depth = self
            .retained
            .iter()
            .rev()
            .skip(1)
            .take_while(|held| {
                matches!(&held.content, RetainedContent::Disk(handle) if handle.is_written())
            })
            .count() as u64;
        let retained_depth = self.retained_depth();
        let resident_undo_blocks = self.resident_undo_blocks();
        let disk = self.undo_store.as_ref().map(DiskUndoStore::metrics);
        CommitReport {
            readiness,
            undo,
            total_us: started.elapsed().as_micros() as u64,
            retained_depth,
            resident_undo_blocks,
            current_undo_written,
            completed_prior_depth,
            disk,
        }
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
        match &retained.content {
            RetainedContent::Full(trie_cache) => {
                let breakdown = trie_cache.memory_breakdown();
                RetainedGenerationBytes {
                    enabled,
                    present: true,
                    total_bytes: trie_cache.estimated_memory_bytes(),
                    exclusive_bytes: trie_cache.exclusive_memory_bytes(),
                    complete_total_bytes: breakdown.total_bytes(),
                    complete_exclusive_bytes: breakdown.exclusive_bytes(),
                    frame_bytes: 0,
                    breakdown,
                }
            }
            // Normal on every committed block in the frames-only layout, and reachable after an
            // undo in the hybrid layout. Every other field stays zero: those fields are defined
            // against a whole cache, and a frame is not one.
            RetainedContent::Frame(frame) => RetainedGenerationBytes {
                enabled,
                present: true,
                frame_bytes: frame.allocated_bytes(),
                ..Default::default()
            },
            RetainedContent::Disk(_) => {
                RetainedGenerationBytes { enabled, present: true, ..Default::default() }
            }
        }
    }

    /// What the whole retained deque costs, which is not K times what one generation costs.
    ///
    /// `exclusive_memory_bytes` is defined against one other holder, and at K > 1 that definition
    /// stops meaning anything: a storage trie shared by three generations is non-exclusive in all
    /// three, so summing exclusives undercounts it — flattering the design — and summing totals
    /// counts it three times. The quantity an operator's budget cares about is what dropping the
    /// deque would return, and that is a union:
    ///
    /// ```text
    /// union(allocations reachable from every retained generation)
    ///   - (allocations reachable from the live cache)
    /// + per generation, the parts nothing else can be holding
    /// ```
    ///
    /// The subtraction is the half that is easy to forget. A trie the live cache still points at
    /// is not freed by dropping the deque, however many retained generations also point at it, so
    /// counting it as retention cost would charge the deque for memory the pair needs anyway.
    pub fn retained_deque_bytes(&self) -> RetainedDequeBytes {
        let live: HashSet<usize> =
            self.trie_cache.shared_allocations().into_iter().map(|(id, _)| id).collect();

        let mut pool: HashMap<usize, usize> = HashMap::default();
        let mut unshared = 0usize;
        let mut sum = 0usize;
        let mut full_generations = 0usize;
        let mut frames = 0usize;
        let mut frame_bytes = 0usize;
        for generation in &self.retained {
            // A frame partitions the same way a generation does — the storage tries it holds are
            // `Arc`s an older frame or the live cache may hold too, and everything else is its
            // own — so the two go through one union rather than being reported side by side.
            let (own, shared) = match &generation.content {
                RetainedContent::Full(trie_cache) => {
                    full_generations += 1;
                    sum = sum.saturating_add(trie_cache.memory_breakdown().total_bytes());
                    (trie_cache.unshared_bytes(), trie_cache.shared_allocations())
                }
                RetainedContent::Frame(frame) => {
                    frames += 1;
                    let held = frame.allocated_bytes();
                    frame_bytes = frame_bytes.saturating_add(held);
                    sum = sum.saturating_add(held);
                    (frame.unshared_bytes(), frame.shared_allocations())
                }
                RetainedContent::Disk(_) => {
                    frames += 1;
                    (0, Vec::new())
                }
            };
            unshared = unshared.saturating_add(own);
            for (id, bytes) in shared {
                // Inserted rather than added: the same allocation reached from two generations is
                // one allocation. `live` is excluded here rather than subtracted afterwards, so a
                // trie the pair still uses never enters the total in the first place.
                if !live.contains(&id) {
                    pool.insert(id, bytes);
                }
            }
        }
        let shared_pool_bytes: usize = pool.values().sum();

        RetainedDequeBytes {
            generations: self.retained.len(),
            full_generations,
            frames,
            frame_bytes,
            unshared_bytes: unshared,
            shared_pool_bytes,
            shared_allocations: pool.len(),
            total_bytes: unshared.saturating_add(shared_pool_bytes),
            generation_sum_bytes: sum,
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
        // representation, warm-set sizing policy and undo-recording setting it was constructed
        // with, exactly as a fresh process would build it. The sizing policy's interval state is
        // not carried — the tables it was counting against no longer exist, so the reset starts a
        // new interval — and neither is any record in progress, which described a cache that is
        // gone.
        let mut trie_cache = PartialTrieNodeCache::new_with_repr(self.trie_cache.repr());
        trie_cache.set_warm_shrink_policy(self.trie_cache.warm_shrink_policy());
        // Carried for the same reason as the sizing policy: a pair that rewarms after a gap has to
        // come back on the arm it was started on. Silently reverting to whole generations would
        // leave the manifest saying one thing and the deque doing another, which is exactly what a
        // measured arm cannot afford.
        trie_cache.set_undo_recording(self.trie_cache.records_undo());
        self.trie_cache = trie_cache;
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
            .map(|report| report.ready)
    }

    pub fn restore_retained_generations(
        &mut self,
        lineage: &ExpectedLineage,
        ancestor_state_root: B256,
        cache_policy_id: B256,
    ) -> Option<RecoveryReport> {
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
                warn!(target: "partial_stateless", cause = "lineage_mismatch",
                    block = held.block_number, held_hash = ?held.block_hash,
                    expected_block = expected.0, expected_hash = ?expected.1,
                    requested_depth = depth, dropped_generations = self.retained.len(),
                    retained_depth = 0, configured_depth = self.retention_depth.get(),
                    "Undo coverage lost; retained history does not match the requested lineage");
                // The whole deque, not the mismatching part. Every generation above the mismatch
                // describes the branch the caller has just been told is not canonical, and the
                // ones below cannot be reached without walking through it — so partial truncation
                // would leave a run whose newest end nothing vouches for. The caches, the undo log
                // and the tracker are still exactly as they were.
                self.forget_retained_generations();
                return None
            }
        }

        if self
            .retained
            .iter()
            .skip(base)
            .any(|held| matches!(held.content, RetainedContent::Disk(_)))
        {
            return self.restore_from_disk(base, lineage, ancestor_state_root, cache_policy_id)
        }

        // The frames chain: each one names the generation it must be applied to, and the chain
        // hangs off the deque's newest whole copy — or off the live cache, which is what the
        // deque looks like immediately after an undo, when its newest entry is the frame that
        // produced the generation now live. A frame applied to the wrong generation is a replay of
        // preimages onto content they do not describe: it corrupts silently and nothing downstream
        // can tell. So the links are proved here, where a refusal still costs nothing.
        //
        // `chain_base` is the cache each frame in the run below it will actually be applied to,
        // which is what decides whether it *can* be — the representation, and whether the account
        // trie is revealed. Neither moves when a frame is applied, so one check per run of frames
        // covers all of them, and phase 2 is then unable to refuse.
        let mut chain_base = &self.trie_cache;
        let mut expected = chain_base.undo_id();
        for index in (base..self.retained.len()).rev() {
            let held = &self.retained[index];
            match &held.content {
                // A whole copy re-bases the chain: everything below it hangs off this generation
                // and nothing above it matters to the frames beneath.
                RetainedContent::Full(trie_cache) => {
                    chain_base = trie_cache;
                    expected = trie_cache.undo_id();
                }
                RetainedContent::Frame(frame) => {
                    if frame.source() != expected || !chain_base.can_undo(frame) {
                        warn!(
                            target: "partial_stateless",
                            index,
                            frame_source = frame.source(),
                            expected,
                            "A retained frame does not describe the generation above it; rebuilding"
                        );
                        return None
                    }
                    expected = frame.target();
                }
                RetainedContent::Disk(_) => unreachable!("disk suffix handled above"),
            }
            if held.undo_id != expected {
                warn!(
                    target: "partial_stateless",
                    index,
                    held = held.undo_id,
                    expected,
                    "A retained generation disagrees with what its own content produces; rebuilding"
                );
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

        let landed_root = self.retained[base].state_root;
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

        // ---- phase 2: commit -----------------------------------------------------------------
        // The rollback goes first because it is the one step that can still refuse, and it refuses
        // before it mutates: `rollback` re-derives the plan and compares before touching a record.
        // So a refusal here is a clean fallback with nothing given back, exactly as the K = 1
        // form's was — and everything after this line cannot fail.
        if let Err(err) = self.cache.rollback(&plan) {
            // Unreachable on this path: nothing between the preflight and here touches the cache.
            // Reported rather than asserted because a validator that got here has a broken
            // invariant, not a block to reject.
            warn!(
                target: "partial_stateless",
                ancestor = ancestor_number,
                %err,
                "Flat rollback refused a plan it had just proved; falling back to a rebuild"
            );
            return None
        }
        // Split rather than looped. The generations above the landing one are consumed and the
        // ones below are kept, so a second, shallower undo can still run against what this one
        // left — and the landing generation is identified by index rather than by counting pops,
        // which is where a D-times loop gets the off-by-one wrong.
        let mut given_back = self.retained.split_off(base);
        // Restored together with the caches. Between here and the tracker swap the pair holds the
        // ancestor's header over the ancestor's caches, and both name the same generation.
        let landed_head = given_back
            .front_mut()
            .expect("the lineage check proved this index is held")
            .accepted_head
            .take();
        // Newest first: the whole copy at the top of the range, then every frame between it and
        // the landing generation, each applied to what the one above it produced. A frame at the
        // very top applies to the live cache in place, which is only reachable after a previous
        // undo and is exactly what phase 1's chain check started from. Every step here was proved
        // possible above and none of them can refuse.
        let mut landed: Option<PartialTrieNodeCache> = None;
        let mut frames_applied = 0;
        while let Some(held) = given_back.pop_back() {
            match held.content {
                RetainedContent::Full(trie_cache) => landed = Some(trie_cache),
                RetainedContent::Frame(frame) => {
                    let applied = match &mut landed {
                        Some(landed) => landed.undo(*frame),
                        None => self.trie_cache.undo(*frame),
                    };
                    debug_assert!(applied, "phase 1 proved every frame in the range applies");
                    frames_applied += u64::from(applied);
                }
                RetainedContent::Disk(_) => unreachable!("disk suffix handled above"),
            }
        }
        if let Some(landed) = landed {
            self.trie_cache = landed;
        }
        self.accepted_head = landed_head;
        self.readiness = next_readiness;
        Some(RecoveryReport { ready, frames_applied })
    }

    /// File I/O can fail at any depth. Work on private live-state copies and hold just one
    /// decoded bundle at a time; the original pair and its history stay intact on every error.
    /// Peak overhead includes another live flat/trie state plus file bytes and its decoded
    /// bundle, even at depth one. A verify/drop pass followed by in-place undo cannot preserve
    /// this guarantee if a second file read fails.
    fn restore_from_disk(
        &mut self,
        base: usize,
        lineage: &ExpectedLineage,
        ancestor_state_root: B256,
        cache_policy_id: B256,
    ) -> Option<RecoveryReport> {
        let Some(store) = self.undo_store.as_ref() else {
            warn!(target: "partial_stateless", cause = "missing_undo_store",
                block = lineage.ancestor().0, requested_depth = lineage.depth(),
                retained_depth = self.retained_depth(), dropped_generations = 0,
                "Disk undo refused; retained handles have no backing store");
            return None
        };
        let mut probe = store.recovery_probe(lineage.ancestor().0, lineage.depth());
        let reject = |cause: &'static str, block: u64, detail: String| {
            warn!(target: "partial_stateless", cause, block, %detail,
                requested_depth = lineage.depth(), retained_depth = self.retained_depth(),
                dropped_generations = 0,
                "Disk undo refused; live caches and retained history are unchanged");
            None
        };
        let copy_started = Instant::now();
        let mut cache = self.cache.fork_for_rollback();
        let mut trie = self.trie_cache.fork_for_rollback();
        probe.timings.candidate_copy_us = copy_started.elapsed().as_micros() as u64;
        let mut frames_applied = 0;
        for held in self.retained.iter().skip(base).rev() {
            let block = held.block_number + 1;
            let (frame, flat) = match &held.content {
                RetainedContent::Disk(handle) => {
                    let bundle = match handle.load_timed() {
                        Ok((bundle, timing)) => {
                            probe.timings.pending_wait_us += timing.wait_us;
                            probe.timings.read_us += timing.read_us;
                            probe.timings.checksum_us += timing.checksum_us;
                            probe.timings.decode_us += timing.decode_us;
                            probe.timings.bytes_read += timing.bytes;
                            bundle
                        }
                        Err(error) => return reject("file_unavailable", block, error),
                    };
                    if bundle.parent_hash != held.block_hash {
                        return reject(
                            "parent_hash_mismatch",
                            block,
                            format!("expected {}, found {}", held.block_hash, bundle.parent_hash),
                        )
                    }
                    (Some(bundle.trie), bundle.flat)
                }
                RetainedContent::Frame(frame) => {
                    let Some(flat) = self.cache.undo_record(block) else {
                        return reject(
                            "missing_flat_record",
                            block,
                            "resident frame has no flat undo".into(),
                        )
                    };
                    (Some((**frame).clone()), flat.clone())
                }
                RetainedContent::Full(parent) => {
                    trie = parent.fork_for_rollback();
                    let Some(flat) = self.cache.undo_record(block) else {
                        return reject(
                            "missing_flat_record",
                            block,
                            "resident generation has no flat undo".into(),
                        )
                    };
                    (None, flat.clone())
                }
            };
            if flat.previous_block() != held.block_number {
                return reject(
                    "previous_block_mismatch",
                    block,
                    format!("expected {}, found {}", held.block_number, flat.previous_block()),
                )
            }
            // Like can_rollback_to, only the landing record needs an authenticated cached root.
            // Intermediate records are immediately replaced by the next undo in this candidate.
            if held.block_number == lineage.ancestor().0 && flat.previous_cache_root().is_none() {
                return reject(
                    "missing_landing_cache_root",
                    block,
                    "landing flat undo has no cached root".into(),
                )
            }
            if let Some(frame) = frame {
                if frame.source() != trie.undo_id() || frame.target() != held.undo_id {
                    return reject(
                        "frame_generation_mismatch",
                        block,
                        format!(
                            "expected {} -> {}, found {} -> {}",
                            trie.undo_id(),
                            held.undo_id,
                            frame.source(),
                            frame.target()
                        ),
                    )
                }
                let undo_started = Instant::now();
                if !trie.undo(frame) {
                    return reject(
                        "trie_undo_refused",
                        block,
                        "trie representation cannot apply this frame".into(),
                    )
                }
                probe.timings.undo_us += undo_started.elapsed().as_micros() as u64;
                frames_applied += 1;
            }
            if trie.undo_id() != held.undo_id || trie.state_root() != held.state_root {
                return reject(
                    "restored_trie_mismatch",
                    block,
                    format!(
                        "expected generation {} root {:?}, found {} root {:?}",
                        held.undo_id,
                        held.state_root,
                        trie.undo_id(),
                        trie.state_root()
                    ),
                )
            }
            let undo_started = Instant::now();
            if let Err(error) = cache.rollback_record(flat) {
                return reject("flat_undo_refused", block, format!("{error:?}"))
            }
            probe.timings.undo_us += undo_started.elapsed().as_micros() as u64;
        }
        let (number, hash) = lineage.ancestor();
        if cache.current_block() != number || trie.state_root() != Some(ancestor_state_root) {
            return reject(
                "ancestor_state_mismatch",
                number,
                format!(
                    "expected height {number} root {ancestor_state_root}, found {} root {:?}",
                    cache.current_block(),
                    trie.state_root()
                ),
            )
        }
        let checkpoint = TrustedCheckpoint {
            block_number: number,
            block_hash: hash,
            state_root: ancestor_state_root,
            cache_root: cache.cache_root(),
            cache_policy_id,
        };
        let mut readiness = self.readiness.clone();
        let ready = match readiness.restore_from_undone_blocks(
            lineage.depth(),
            &checkpoint,
            &CacheObservation::capture(&cache, &trie),
        ) {
            Ok(ready) => ready.clone(),
            Err(error) => return reject("readiness_refused", number, format!("{error:?}")),
        };
        let head = self.retained[base].accepted_head.clone();
        // Publication has no remaining file reads or other fallible operations.
        let publish_started = Instant::now();
        self.cache.install_rollback(cache);
        self.trie_cache = trie;
        self.retained.truncate(base);
        self.accepted_head = head;
        self.readiness = readiness;
        probe.timings.publish_us = publish_started.elapsed().as_micros() as u64;
        probe.timings.success = true;
        Some(RecoveryReport { ready, frames_applied })
    }
}

/// The restored readiness and the number of undo frames actually applied during recovery.
#[derive(Debug, Clone)]
pub struct RecoveryReport {
    pub ready: ReadyParent,
    /// Whole retained generations do not count as frames.
    pub frames_applied: u64,
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
/// avoid. Flat undo is paired with trie undo in each disk bundle; without a disk store it
/// remains in `NetworkStateCache`'s undo log.
pub struct RetainedGeneration {
    /// The generation itself, or the frame that produces it from the one above.
    pub content: RetainedContent,
    /// Which generation this is, as [`PartialTrieNodeCache::undo_id`] names it.
    ///
    /// What links a frame to the generation it applies to. Kept beside the content rather than
    /// read out of it, because a frame's own `target` and this have to agree and a check needs two
    /// sources to compare.
    pub undo_id: u64,
    /// The state root this generation is the state at.
    ///
    /// Held as a scalar because the depth-D preflight compares it against the canonical header's
    /// before anything is mutated, and a generation held as a frame has no trie to ask.
    pub state_root: Option<B256>,
    pub block_hash: B256,
    pub block_number: u64,
    /// Accepted head as of this generation, restored with it by a depth-1 undo.
    ///
    /// `None` when the generation predates any accepted header — a pair one block out of a cold
    /// reset retains a trie it has no header for. Undoing into that is sound: the pair is warming,
    /// and [`CoordinatedPair::accepted_parent`] then reports absence rather than a guess.
    pub accepted_head: Option<SealedHeader>,
}

/// What one commit did beyond installing the block.
///
/// Returned rather than logged, because the two halves have different readers: the readiness label
/// is what a run log prints per block, and the undo report describes frame size and assembly
/// cost. A validator with no run log ignores both.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct CommitReport {
    /// Readiness after the transition, as a label.
    pub readiness: &'static str,
    /// What this commit recorded, and what ending the block's record cost.
    pub undo: CommitUndoReport,
    /// Complete synchronous coordination boundary, not durable-write latency.
    pub total_us: u64,
    pub retained_depth: u64,
    pub resident_undo_blocks: usize,
    /// Whether the handle enqueued by this commit had already completed at observation time.
    /// `None` means the newest retained generation is not disk-backed.
    pub current_undo_written: Option<bool>,
    /// Contiguous completed disk handles immediately behind the current commit's handle.
    ///
    /// This measures writer progress without letting the just-enqueued handle force the value to
    /// zero. [`Self::retained_depth`] remains the recoverable depth: recovery may wait for pending
    /// handles and verifies every file before using it.
    pub completed_prior_depth: u64,
    pub disk: Option<DiskUndoMetrics>,
}

/// The undo frame one commit produced, if it produced one.
///
/// `frame` is `None` on every commit that kept its predecessor whole — recording off, a
/// `Parallel` cache, or a record that could not describe its block — and also at depth 1 in the
/// hybrid layout, where a frame would be immediately evicted. `us` is measured either way.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct CommitUndoReport {
    /// The frame, and the block it undoes.
    pub frame: Option<CommitUndoFrame>,
    /// Ending the block's journals and, when one came out of them, assembling the frame.
    ///
    /// Not the cost of *recording*, which is a lookup per write spread across the whole block and
    /// is not bracketed anywhere. Measuring it requires comparison with recording disabled.
    pub us: u64,
    pub accounting_us: u64,
    pub parent_drop_us: u64,
    /// Retained-deque expiry wall time, including synchronous file-handle drops and unlinks.
    pub history_expire_us: u64,
    /// Bundle handoff call including enqueue backpressure and error cleanup.
    /// The store's cumulative `enqueue_us` is nested within this wall interval.
    pub spill_call_us: u64,
}

/// One frame, and which block it undoes.
///
/// The block is carried rather than inferred by the reader. A hybrid frame describes the previous
/// block whose generation is being demoted; a frames-only frame describes the block whose journal
/// is closed by this commit. In both cases the value is the block applying the frame would undo.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct CommitUndoFrame {
    /// What the frame holds, by kind.
    pub counts: TrieCacheUndoCounts,
    /// What the record itself weighs: the preimages and containers the block created.
    ///
    /// Not the frame's whole footprint. The previous-version storage tries it holds by `Arc` are
    /// bytes the block kept *alive* rather than created, they are shared with older frames and the
    /// live cache, and pricing them walks every one of them — so they are unioned at the deque
    /// level by `retained_deque_bytes` instead of added up here.
    pub record_bytes: usize,
    /// The block applying this frame would undo.
    pub block_number: u64,
    /// That block's hash. A height alone does not name a block across a reorg, which is the same
    /// reason a retained generation is tagged by hash.
    pub block_hash: B256,
}

/// How one retained generation is held: whole, or as the diff that produces it.
///
/// The hybrid layout keeps the newest generation `Full` and older generations as frames. The
/// frames-only layout closes the live cache's record at its own commit and can therefore hold the
/// newest generation as a frame too. A cache that cannot produce a frame is kept `Full` in either
/// layout.
// Keep the whole cache inline because the parent-read path accesses it every block. The map
// allocations dominate the small amount of enum padding that boxing would save.
#[expect(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum RetainedContent {
    /// The whole cache.
    Full(PartialTrieNodeCache),
    /// What one block changed, applied to the generation above this one to produce it.
    Frame(Box<TrieCacheUndoFrame>),
    /// Trie and flat preimages in a pending or completed session-local file.
    Disk(DiskUndoHandle),
}

impl RetainedContent {
    /// The whole cache, when this generation is held as one.
    pub const fn full(&self) -> Option<&PartialTrieNodeCache> {
        match self {
            Self::Full(cache) => Some(cache),
            Self::Frame(_) | Self::Disk(_) => None,
        }
    }

    /// The frame, when this generation is held as one.
    pub const fn frame(&self) -> Option<&TrieCacheUndoFrame> {
        match self {
            Self::Full(_) | Self::Disk(_) => None,
            Self::Frame(frame) => Some(frame),
        }
    }
}

impl RetainedGeneration {
    /// A generation held whole, which is how every one of them starts.
    pub fn full(
        trie_cache: PartialTrieNodeCache,
        block_hash: B256,
        block_number: u64,
        accepted_head: Option<SealedHeader>,
    ) -> Self {
        Self {
            undo_id: trie_cache.undo_id(),
            state_root: trie_cache.state_root(),
            content: RetainedContent::Full(trie_cache),
            block_hash,
            block_number,
            accepted_head,
        }
    }

    /// The whole cache, when this generation is held as one.
    pub const fn trie_cache(&self) -> Option<&PartialTrieNodeCache> {
        self.content.full()
    }
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
    /// What the newest generation's undo frame holds, when it is held as one rather than whole.
    ///
    /// Non-zero when the newest retained generation is a frame. Every field above is then zero —
    /// they are defined against a whole cache.
    pub frame_bytes: usize,
    /// Where the two complete figures came from.
    pub breakdown: TrieCacheMemory,
}

/// What the whole retained deque physically holds, over and above the live cache.
///
/// Never a sum of per-generation figures. See [`CoordinatedPair::retained_deque_bytes`] for why
/// the two obvious ways to build one from `exclusive_memory_bytes` are both wrong.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct RetainedDequeBytes {
    /// How many generations the figure covers. Zero means the deque is empty, not that it is free.
    pub generations: usize,
    /// Of those, how many are held as whole caches.
    ///
    /// One in the hybrid steady state and zero in a successful frames-only steady state. A frame
    /// fallback raises it; recording off makes it equal `generations`.
    pub full_generations: usize,
    /// Of those, how many are held as undo frames.
    pub frames: usize,
    /// What those frames hold, added with no deduplication — the frames' share of
    /// [`Self::generation_sum_bytes`].
    pub frame_bytes: usize,
    /// Summed over generations: the account trie, both warm sets, the retained account paths, and
    /// the retained-storage-paths map — and, for a generation held as a frame, everything it holds
    /// beside the storage tries. Nothing here can be shared, so summing is correct.
    pub unshared_bytes: usize,
    /// Storage tries and retained-path slices reachable from the deque but not from the live
    /// cache, each counted once however many generations hold it.
    pub shared_pool_bytes: usize,
    /// How many distinct allocations `shared_pool_bytes` covers, so a run can see whether sharing
    /// is doing any work at all: equal to the per-generation count times K means it is not.
    pub shared_allocations: usize,
    /// What dropping the whole deque would return.
    ///
    /// Not the same question as "what does the process hold". An allocation the live cache also
    /// points at is not freed by dropping the deque, so it is excluded here — and it is still
    /// resident. Use [`Self::generation_sum_bytes`] for a budget.
    pub total_bytes: usize,
    /// Each generation's own complete size — a frame's estimated bytes where it is one — added
    /// with no deduplication at all.
    ///
    /// The other end of the range `total_bytes` opens. It over-counts anything two generations
    /// genuinely share and under-counts nothing, so the true resident cost of the deque is between
    /// the two — and a run that reports both can say where, instead of leaving a reader to pick
    /// the flattering one.
    ///
    /// Reported because the first K=3 measurement found the gap is not small: three independent
    /// readings (RSS delta, jemalloc `allocated` delta, and a generation's own complete total) put
    /// the marginal cost of a generation at ~186 MiB while `total_bytes` charged ~128. Whatever
    /// explains that, an operator's budget cannot be set from the smaller number.
    pub generation_sum_bytes: usize,
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
    try_deep_recovery(pair, chain, &lineage, cache_policy_id).map(|report| report.ready)
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
) -> Option<RecoveryReport> {
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
    let report = pair.restore_retained_generations(lineage, state_root, cache_policy_id)?;
    info!(
        target: "partial_stateless",
        block = report.ready.anchor.block_number,
        block_hash = ?report.ready.anchor.block_hash,
        ancestor = ancestor_number,
        depth,
        frames_applied = report.frames_applied,
        restore_us = started.elapsed().as_micros() as u64,
        "Recovered by undoing {depth} block(s) from the retained generations instead of rebuilding"
    );
    Some(report)
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
