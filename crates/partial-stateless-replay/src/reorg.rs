//! Applying a recorded reorg to a standalone pair, with no database behind it.
//!
//! The producer writes a [`Reorg`] frame before the winning branch's commits, and before this
//! module existed both drivers refused it: the batch replay warned and kept going, the follower
//! stopped publishing.
//! Neither is what the corpus promises. A depth-1 reorg is exactly the one this pair can undo by
//! itself — the retained generation is the block it gave back — so this module binds the frame to
//! what the consumer verified for itself and, when the two agree, performs the undo.
//!
//! **The authority is the consumer's own history, never the frame.** [`try_depth_one_recovery`]
//! authenticates the retained trie against the canonical state root at the target, and a state
//! root taken from the reorg frame would make that check a tautology: the producer would be
//! attesting to its own claim. [`VerifiedHistory`] is the honest source — every root in it is one
//! this process computed while validating the block that produced it, and its seed is the
//! operator-trusted checkpoint. That is also why the frame needed no new field: what a
//! database-backed node asks its provider for, a standalone consumer already knows.

use alloy_primitives::B256;
use partial_stateless_stream::{BlockRef, Reorg};
use partial_stateless_validator::{
    coordination::ProviderResult, try_deep_recovery, CanonicalStateRoots, ExpectedLineage,
    MAX_RETENTION_DEPTH,
};
use std::collections::VecDeque;
use tracing::{info, warn};

use crate::driver::ReplayState;

/// How many verified blocks are kept for recovery questions.
///
/// A depth-D undo needs D + 1. The rest is there so that a reorg deeper than this consumer can
/// undo can still be *checked* against its own branch before it is refused — a refusal that names
/// the right ancestor is what lets recovery ask for a snapshot at that exact block.
///
/// Deliberately larger than `MAX_RETENTION_DEPTH + 1` rather than equal to it. Sizing it to the
/// retention depth would tie the window in which a deep reorg can be *recognised* to the window in
/// which it can be *undone*, and those are different jobs: the second is a memory setting, the
/// first is what keeps a refusal informative.
const HISTORY_DEPTH: usize = 128;

// The undo needs D + 1 entries to check a depth-D run against, so a retention depth the history
// cannot cover would be a configuration that passes validation and then always refuses.
const _: () = assert!(HISTORY_DEPTH > MAX_RETENTION_DEPTH as usize);

/// The blocks this consumer verified, and the state roots it computed for them.
///
/// Seeded with the checkpoint it restored from, which is the one entry it did not compute but was
/// authenticated against by the snapshot it installed.
#[derive(Debug, Clone)]
pub(crate) struct VerifiedHistory {
    entries: VecDeque<VerifiedBlock>,
}

impl VerifiedHistory {
    /// Starts a history at the checkpoint a pair was restored from.
    pub(crate) fn restored_at(block: BlockRef, state_root: B256, cache_root: B256) -> Self {
        let mut entries = VecDeque::with_capacity(HISTORY_DEPTH);
        entries.push_back(VerifiedBlock {
            number: block.number,
            hash: block.hash,
            state_root,
            cache_root,
        });
        Self { entries }
    }

    /// Records a block this consumer validated, with the state root it derived for it and the
    /// flat cache root its pair committed behind it.
    pub(crate) fn record(&mut self, block: BlockRef, state_root: B256, cache_root: B256) {
        if self.entries.len() == HISTORY_DEPTH {
            self.entries.pop_front();
        }
        self.entries.push_back(VerifiedBlock {
            number: block.number,
            hash: block.hash,
            state_root,
            cache_root,
        });
    }

    /// The newest block this consumer stands behind.
    pub(crate) fn tip(&self) -> Option<BlockRef> {
        self.entries.back().map(|entry| BlockRef { number: entry.number, hash: entry.hash })
    }

    /// The full entry this consumer verified at exactly `block`, if it is still within the
    /// retained depth. This is what a *late* recovery checkpoint is cross-checked against: by
    /// the time it arrives the pair has moved past its anchor, so the current fingerprint can no
    /// longer answer for height H — only the record made when H was verified can.
    pub(crate) fn entry_at(&self, block: BlockRef) -> Option<&VerifiedBlock> {
        self.entries
            .iter()
            .rev()
            .find(|entry| entry.number == block.number && entry.hash == block.hash)
    }

    /// Drops everything above `number`, which an applied undo has just left the chain.
    fn rewind_above(&mut self, number: u64) {
        while self.entries.back().is_some_and(|entry| entry.number > number) {
            self.entries.pop_back();
        }
    }

    /// Whether this exact block is one of the blocks this consumer verified.
    fn holds(&self, block: BlockRef) -> bool {
        self.entries.iter().any(|entry| entry.number == block.number && entry.hash == block.hash)
    }

    /// Whether `abandoned` is the newest run of blocks this consumer verified.
    ///
    /// Compared newest first and by hash, so a producer describing a branch this consumer never
    /// held is refused rather than undone. Blocks older than the retained window stop the walk:
    /// a reorg that deep is refused on depth anyway, and claiming to have checked what was already
    /// forgotten would be the more dangerous answer.
    fn is_canonical_suffix(&self, abandoned: &[BlockRef]) -> bool {
        let mut ours = self.entries.iter().rev();
        for theirs in abandoned.iter().rev() {
            match ours.next() {
                Some(our) if our.number == theirs.number && our.hash == theirs.hash => {}
                Some(_) => return false,
                None => break,
            }
        }
        true
    }
}

impl CanonicalStateRoots for VerifiedHistory {
    fn state_root_of(&self, hash: B256) -> ProviderResult<Option<B256>> {
        // Never an error: a consumer with no database cannot fail to read, only fail to know. An
        // unknown hash is a rejection, which is what the trait asks `Ok(None)` to mean.
        Ok(self.entries.iter().rev().find(|entry| entry.hash == hash).map(|entry| entry.state_root))
    }
}

/// What a recorded reorg did to the pair.
#[derive(Debug, Clone)]
pub(crate) enum ReorgOutcome {
    /// The branch was undone. The pair is `Ready` at `ancestor` and may verify its next child.
    Applied {
        /// The block both branches share, and the pair's new head.
        ancestor: BlockRef,
        /// The blocks that were given back, lowest first — `Reorg.abandoned`'s own order.
        ///
        /// A list rather than a block because a depth-D undo gives back D of them, and a caller
        /// that logged only one would under-report every reorg deeper than one. Lowest first so
        /// that `first()` is the block just above the ancestor and `last()` is the abandoned tip,
        /// which is the order every existing log line and JSONL field already prints.
        undone: Vec<BlockRef>,
        /// True when nothing replaces the abandoned blocks.
        revert: bool,
        /// The tip the producer is moving to, so the caller can tell when the branch is complete.
        winning_tip: Option<BlockRef>,
    },
    /// A real reorg of this consumer's own branch that it cannot undo by itself.
    ///
    /// The common ancestor is a block this consumer verified, so it knows exactly where the
    /// producer is asking it to stand; it just cannot get there. The pair is left `Recovering`,
    /// so it refuses every further commit — a consumer with no database has no rebuild — and
    /// because the ancestor is authenticated, a checkpoint at that exact block is a *continuous*
    /// recovery: everything below it was verified, and nothing above it is canonical any more.
    Unrecoverable {
        /// The block a recovery snapshot has to be authenticated at.
        ancestor: BlockRef,
        /// How many blocks left the chain.
        depth: u64,
        /// Why the undo was not available.
        detail: String,
    },
    /// A well-formed reorg naming a common ancestor this consumer never verified.
    ///
    /// Nothing was touched. The caller must still stop — the producer has moved somewhere this
    /// consumer cannot follow — but `ancestor` carries no authority here, so a checkpoint landing
    /// on it may not be reported as a continuous recovery: this consumer cannot show it ever
    /// stood on that block. Distinguishing this from [`Unrecoverable`](Self::Unrecoverable) is
    /// what keeps `continuous` honest.
    Unbound {
        /// The block the frame named, for the record only.
        ancestor: BlockRef,
        /// How many blocks the frame said left the chain.
        depth: u64,
        /// Why the frame could not be bound.
        detail: String,
    },
    /// The frame does not describe a reorg this consumer can evaluate. Nothing was touched.
    Malformed {
        /// What was wrong with it.
        detail: String,
    },
}

impl ReorgOutcome {
    /// Whether this frame has the standing to withdraw a winning branch still being delivered.
    ///
    /// A producer that announces a branch and then reorgs again has not left a hole: the blocks
    /// between where delivery got to and where it had been heading never became canonical, so no
    /// verdict was ever owed on them. But only a frame this consumer could authenticate against
    /// its own history says that. A malformed frame, or one about a branch this consumer never
    /// stood on, is not a retraction — under it the announced tip is simply unaccounted for, and
    /// that is a hole.
    pub(crate) const fn withdraws_an_announced_branch(&self) -> bool {
        matches!(self, Self::Applied { .. } | Self::Unrecoverable { .. })
    }
}

/// Applies a recorded reorg or revert to `state`, or explains why it could not be.
///
/// The order is deliberate, and it is the order of authority.
///
/// Shape is judged first and leaves the pair alone, because a frame that is not a reorg should not
/// stop a driver that could still read the rest of the corpus. The *common ancestor* is bound
/// next, against this consumer's own verified history, and that check alone decides whether the
/// block the frame names may anchor a recovery: a consumer that never stood on it cannot call
/// landing there continuous. Recovery begins the moment the ancestor binds, because from there the
/// frame is about this consumer's own chain and its blocks above that point are gone — true
/// whether or not the undo turns out to be available.
///
/// Everything after that is about performing the undo, not about locating it. The abandoned suffix
/// is checked because undoing on the strength of a branch this consumer never held would give back
/// the wrong block; failing it forfeits the undo, not the ancestor.
pub(crate) fn apply_reorg(state: &mut ReplayState, reorg: &Reorg) -> ReorgOutcome {
    if let Err(detail) = check_shape(reorg) {
        return ReorgOutcome::Malformed { detail }
    }
    let ancestor = reorg.common_ancestor;
    let depth = reorg.abandoned.len() as u64;
    let unwound_from = reorg.abandoned[0].number;

    if !state.history.holds(ancestor) {
        return ReorgOutcome::Unbound {
            ancestor,
            depth,
            detail: format!(
                "the common ancestor {}/{:?} is not a block this consumer verified",
                ancestor.number, ancestor.hash
            ),
        }
    }

    // The ancestor is this consumer's own block, and the producer says the chain left it behind.
    state.pair.readiness.begin_recovery(unwound_from);

    if !state.history.is_canonical_suffix(&reorg.abandoned) {
        return ReorgOutcome::Unrecoverable {
            ancestor,
            depth,
            detail: "the abandoned blocks are not the branch this consumer verified".to_string(),
        }
    }
    let retention_depth = state.pair.retention_depth;
    if depth > retention_depth.get() {
        return ReorgOutcome::Unrecoverable {
            ancestor,
            depth,
            detail: format!(
                "a reorg {depth} blocks deep needs a snapshot at the common ancestor; this \
                 consumer retains {retention_depth} generation(s)"
            ),
        }
    }

    // Built from this consumer's own frame *after* the suffix check has proved the run is its own
    // branch, so what the pair receives is a lineage the caller already stands behind. Its shape
    // was checked by `check_shape` above; `ExpectedLineage::new` re-derives that independently
    // rather than trusting it, because the pair's guarantee has to hold for every caller.
    let abandoned: Vec<(u64, B256)> =
        reorg.abandoned.iter().map(|block| (block.number, block.hash)).collect();
    let lineage = match ExpectedLineage::new((ancestor.number, ancestor.hash), &abandoned) {
        Ok(lineage) => lineage,
        Err(err) => return ReorgOutcome::Unrecoverable { ancestor, depth, detail: err.to_string() },
    };

    let ReplayState { pair, history, config, .. } = state;
    let policy_id = config.cache_policy_id();
    if try_deep_recovery(pair, &*history, &lineage, policy_id).is_none() {
        return ReorgOutcome::Unrecoverable {
            ancestor,
            depth,
            detail: format!(
                "the retained generations could not restore the common ancestor {} blocks back",
                depth
            ),
        }
    }
    let undone = reorg.abandoned.clone();
    history.rewind_above(ancestor.number);
    let revert = reorg.winning_tip.is_none();
    info!(
        target: "ps_replay",
        ancestor = ancestor.number,
        undone_from = undone.first().map(|block| block.number),
        undone_to = undone.last().map(|block| block.number),
        depth,
        revert,
        "Undid {depth} block(s) against the retained generations; the pair is back at the common \
         ancestor"
    );
    ReorgOutcome::Applied { ancestor, undone, revert, winning_tip: reorg.winning_tip }
}

/// Everything about a reorg frame that can be judged without consulting the pair.
///
/// Shared with the recovery scan, which reads reorg frames written while a consumer was not
/// following: a frame that is not a reorg must not be allowed to name the block a recovery is
/// measured against, or a checkpoint landing on an invented ancestor would be reported continuous.
pub(crate) fn check_shape(reorg: &Reorg) -> Result<(), String> {
    let Some(first) = reorg.abandoned.first() else {
        return Err("a reorg that abandons no block is not a reorg".to_string())
    };
    if first.number != reorg.common_ancestor.number + 1 {
        return Err(format!(
            "the lowest abandoned block is {} but the common ancestor is {}",
            first.number, reorg.common_ancestor.number
        ))
    }
    for pair in reorg.abandoned.windows(2) {
        if pair[1].number != pair[0].number + 1 {
            return Err(format!(
                "the abandoned blocks jump from {} to {}",
                pair[0].number, pair[1].number
            ))
        }
    }
    if let Some(tip) = reorg.winning_tip &&
        tip.number <= reorg.common_ancestor.number
    {
        return Err(format!(
            "the winning tip {} is not above the common ancestor {}",
            tip.number, reorg.common_ancestor.number
        ))
    }
    Ok(())
}

/// One block this consumer validated, and what it derived for it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct VerifiedBlock {
    pub(crate) number: u64,
    pub(crate) hash: B256,
    /// Computed here, not read from the frame. This is the whole reason the type exists.
    pub(crate) state_root: B256,
    /// The flat cache root the pair committed behind this block — the one field a late recovery
    /// checkpoint needs that nothing else retains per height. An undo restores exactly the value
    /// recorded here, so the entry stays valid across depth-1 cycles at its height.
    pub(crate) cache_root: B256,
}

/// Reports a reorg the driver could not apply, in the one place both drivers agree on the wording.
///
/// `bound` says whether the frame was about this consumer's own branch, because that is the
/// difference between "a snapshot at this block resumes me exactly" and "I no longer know where
/// I am".
pub(crate) fn warn_inapplicable(ancestor: BlockRef, depth: u64, detail: &str, bound: bool) {
    warn!(
        target: "ps_replay",
        ancestor = ancestor.number,
        ancestor_hash = ?ancestor.hash,
        depth,
        bound,
        detail,
        "A reorg arrived that this pair cannot undo; it stops here and needs a checkpoint, \
         authenticated at the common ancestor when the frame was bound to this branch"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::restore;
    use alloy_primitives::{keccak256, Address, U256};
    use alloy_rlp::Encodable;
    use partial_stateless::{
        bootstrap::CacheSnapshotPackage,
        network_cache::{CachedEntry, NetworkStateCache},
        policy::{AccountData, LastNBlocksPolicy},
        readiness::{BlockContext, CacheReadiness},
        sidecar::last_n_blocks_cache_policy_id,
        BlockAccessedState,
    };
    use partial_stateless_stream::{Checkpoint, Manifest};
    use partial_stateless_validator::{
        admit_block, BlockAdmission, CoordinatedPair, RetainedGeneration, RetentionDepth,
    };
    use reth_chainspec::{EthChainSpec, MAINNET};
    use reth_primitives_traits::{Account, SealedHeader};
    use reth_trie::HashBuilder;
    use reth_trie_common::{proof::ProofRetainer, MultiProof, Nibbles};
    use std::collections::HashMap;

    const ANCHOR_BLOCK: u64 = 100;
    const ACCOUNT_WINDOW: u64 = 64;
    const STORAGE_WINDOW: u64 = 32;

    fn manifest() -> Manifest {
        Manifest {
            chain_id: MAINNET.chain().id(),
            genesis_hash: MAINNET.genesis_hash(),
            cache_policy_id: last_n_blocks_cache_policy_id(ACCOUNT_WINDOW, STORAGE_WINDOW),
            account_window: ACCOUNT_WINDOW,
            storage_window: STORAGE_WINDOW,
            epoch: 1,
            producer: "reorg-test".to_string(),
            first_sequence: 1,
        }
    }

    /// A pair restored from a real one-account snapshot, the same fixture the follow tests use.
    ///
    /// Real rather than stubbed because the undo authenticates the retained trie against a state
    /// root, and a fixture whose trie cannot produce one would be testing the arithmetic around
    /// a check rather than the check.
    fn restored_state() -> (ReplayState, B256) {
        restored_state_at_depth(RetentionDepth::ONE)
    }

    /// The same fixture, configured to retain `depth` generations.
    fn restored_state_at_depth(depth: RetentionDepth) -> (ReplayState, B256) {
        let address = Address::repeat_byte(0x11);
        let account = Account { nonce: 7, balance: U256::from(1_000u64), bytecode_hash: None };
        let address_path = Nibbles::unpack(keccak256(address));
        let mut builder =
            HashBuilder::default().with_proof_retainer(ProofRetainer::from_iter([address_path]));
        builder.add_leaf(
            address_path,
            &alloy_rlp::encode(account.into_trie_account(reth_trie_common::EMPTY_ROOT_HASH)),
        );
        let state_root = builder.root();
        let proof = MultiProof {
            account_subtree: builder.take_proof_nodes(),
            branch_node_masks: Default::default(),
            storages: Default::default(),
        };

        let mut accounts = HashMap::new();
        accounts.insert(
            address,
            CachedEntry {
                value: AccountData { nonce: 7, balance: U256::from(1_000u64), code_hash: None },
                first_accessed_block: 90,
                last_accessed_block: 98,
                access_count: 3,
            },
        );
        let cache = NetworkStateCache::restore(
            accounts,
            HashMap::new(),
            HashMap::new(),
            ANCHOR_BLOCK,
            Box::new(LastNBlocksPolicy::new(ACCOUNT_WINDOW)),
            Box::new(LastNBlocksPolicy::new(STORAGE_WINDOW)),
        );

        let header =
            alloy_consensus::Header { number: ANCHOR_BLOCK, state_root, ..Default::default() };
        let sealed = SealedHeader::seal_slow(header.clone());
        let mut accepted_head_rlp = Vec::new();
        header.encode(&mut accepted_head_rlp);

        let policy_id = last_n_blocks_cache_policy_id(ACCOUNT_WINDOW, STORAGE_WINDOW);
        let anchor = cache.cache_anchor(ANCHOR_BLOCK, sealed.hash(), policy_id);
        let package = CacheSnapshotPackage::from_cache(&cache, anchor, &proof);
        let package_bytes = bincode::serialize(&package).expect("package serializes");

        let mut checkpoint = Checkpoint {
            block: BlockRef { number: ANCHOR_BLOCK, hash: sealed.hash() },
            state_root,
            cache_root: anchor.cache_root,
            cache_policy_id: policy_id,
            accepted_head_rlp,
            snapshot_bytes: 0,
            snapshot_chunks: 0,
            snapshot_digest: B256::ZERO,
        };
        let chunks = checkpoint.chunk(&package_bytes, 4096);
        let state =
            restore(&manifest(), &checkpoint, &chunks, depth).expect("the fixture restores");
        (state, state_root)
    }

    /// Advances the pair one block the way a commit would, retaining the displaced generation.
    ///
    /// The block is described as leaving the state root where it was, which is the only root this
    /// fixture's trie can authenticate; what is under test is the lifecycle, not the trie.
    fn advance(state: &mut ReplayState, number: u64, tag: u8, retain: bool) -> BlockRef {
        let parent = state.history.tip().expect("seeded at the checkpoint");
        let state_root =
            state.pair.trie_cache.state_root().expect("restored trie is authenticated");
        let block = BlockRef { number, hash: B256::with_last_byte(tag) };
        let ctx = BlockContext { number, hash: block.hash, parent_hash: parent.hash, state_root };
        assert!(
            matches!(admit_block(&mut state.pair.readiness, &ctx), BlockAdmission::Admitted(_)),
            "the fixture's block must be admissible"
        );
        let mut accessed = BlockAccessedState::default();
        accessed.accounts.insert(
            Address::repeat_byte(0x11),
            AccountData { nonce: number, balance: U256::from(number), code_hash: None },
        );
        state.pair.cache.on_block_executed(number, &accessed);
        let displaced = state.pair.trie_cache.clone();
        let header = alloy_consensus::Header {
            number,
            parent_hash: parent.hash,
            state_root,
            ..Default::default()
        };
        state.pair.commit_transition(
            Some(displaced),
            &ctx,
            SealedHeader::new(header, block.hash),
            retain,
        );
        let cache_root = state.pair.fingerprint().cache_root;
        state.history.record(block, state_root, cache_root);
        block
    }

    fn reorg_of(ancestor: BlockRef, abandoned: Vec<BlockRef>, tip: Option<BlockRef>) -> Reorg {
        Reorg { common_ancestor: ancestor, abandoned, winning_tip: tip }
    }

    /// Advances `count` blocks and returns them lowest first, which is `Reorg.abandoned`'s order.
    fn advance_run(state: &mut ReplayState, count: u64) -> Vec<BlockRef> {
        (0..count)
            .map(|offset| {
                let number = ANCHOR_BLOCK + 1 + offset;
                advance(state, number, 0xa0 + offset as u8, true)
            })
            .collect()
    }

    fn depth(n: u64) -> RetentionDepth {
        RetentionDepth::new(n).expect("a test depth is in range")
    }

    #[test]
    fn a_depth_two_reorg_is_undone_when_the_pair_retains_two() {
        let (mut state, _) = restored_state_at_depth(depth(2));
        let ancestor = state.history.tip().expect("seeded");
        let undone = advance_run(&mut state, 2);
        assert_eq!(state.pair.retained_depth(), 2, "both displaced generations are held");
        assert_eq!(state.pair.cache.current_block(), ANCHOR_BLOCK + 2);

        let winning = BlockRef { number: ANCHOR_BLOCK + 1, hash: B256::with_last_byte(0xbb) };
        let outcome = apply_reorg(&mut state, &reorg_of(ancestor, undone.clone(), Some(winning)));

        let ReorgOutcome::Applied { ancestor: at, undone: gave_back, .. } = outcome else {
            panic!("a depth-2 reorg is exactly what a pair retaining two can undo")
        };
        assert_eq!(at, ancestor);
        assert_eq!(gave_back, undone, "both abandoned blocks are reported, lowest first");
        assert_eq!(state.pair.cache.current_block(), ANCHOR_BLOCK, "the flat cache gave two back");
        assert!(matches!(state.pair.readiness.state(), CacheReadiness::Ready(_)));
        assert_eq!(state.history.tip(), Some(ancestor));
        assert_eq!(
            state.pair.retained_depth(),
            0,
            "the run consumed the landing generation and dropped the one above it"
        );
    }

    #[test]
    fn a_depth_three_reorg_is_undone_when_the_pair_retains_three() {
        let (mut state, _) = restored_state_at_depth(depth(3));
        let ancestor = state.history.tip().expect("seeded");
        let undone = advance_run(&mut state, 3);
        assert_eq!(state.pair.cache.current_block(), ANCHOR_BLOCK + 3);

        let outcome = apply_reorg(&mut state, &reorg_of(ancestor, undone.clone(), None));

        let ReorgOutcome::Applied { undone: gave_back, revert, .. } = outcome else {
            panic!("a depth-3 reorg is exactly what a pair retaining three can undo")
        };
        assert_eq!(gave_back, undone);
        assert!(revert, "no winning tip is a pure revert, at any depth");
        assert_eq!(state.pair.cache.current_block(), ANCHOR_BLOCK);
        assert!(matches!(state.pair.readiness.state(), CacheReadiness::Ready(_)));
    }

    #[test]
    fn a_reorg_one_deeper_than_the_pair_retains_is_refused_and_names_the_ancestor() {
        let (mut state, _) = restored_state_at_depth(depth(2));
        let ancestor = state.history.tip().expect("seeded");
        // Three blocks against a pair that keeps two: the deque has already dropped the ancestor's
        // own generation, so there is nothing to land on however good the rest of the frame is.
        let undone = advance_run(&mut state, 3);
        assert_eq!(state.pair.retained_depth(), 2, "the deque is capped at the configured depth");

        let outcome = apply_reorg(&mut state, &reorg_of(ancestor, undone, None));

        let ReorgOutcome::Unrecoverable { ancestor: at, depth: reported, detail } = outcome else {
            panic!("a reorg deeper than the pair retains cannot be undone")
        };
        assert_eq!(at, ancestor, "the refusal names the block a snapshot must be taken at");
        assert_eq!(reported, 3);
        assert!(
            detail.contains("retains 2"),
            "the refusal says what this consumer can do: {detail}"
        );
        assert_eq!(
            state.pair.cache.current_block(),
            ANCHOR_BLOCK + 3,
            "a refusal gives nothing back"
        );
    }

    #[test]
    fn a_second_undo_runs_against_the_generations_the_first_left() {
        let (mut state, _) = restored_state_at_depth(depth(3));
        let ancestor = state.history.tip().expect("seeded");
        let run = advance_run(&mut state, 3);

        // Give back the top two, landing on the ancestor's child.
        let mid = run[0];
        let outcome = apply_reorg(&mut state, &reorg_of(mid, run[1..].to_vec(), None));
        assert!(matches!(outcome, ReorgOutcome::Applied { .. }), "the first undo applies");
        assert_eq!(state.pair.cache.current_block(), mid.number);
        assert_eq!(
            state.pair.retained_depth(),
            1,
            "the generation below the landing one survived the split"
        );

        // Now give back the last one, against what the first undo left. This is where popping from
        // the wrong end, or truncating the whole deque, would show.
        let outcome = apply_reorg(&mut state, &reorg_of(ancestor, vec![mid], None));
        let ReorgOutcome::Applied { ancestor: at, .. } = outcome else {
            panic!("the surviving generation is exactly what a second undo needs")
        };
        assert_eq!(at, ancestor);
        assert_eq!(state.pair.cache.current_block(), ANCHOR_BLOCK);
        assert!(matches!(state.pair.readiness.state(), CacheReadiness::Ready(_)));
    }

    #[test]
    fn a_deep_frame_describing_a_branch_this_consumer_never_held_keeps_everything() {
        let (mut state, _) = restored_state_at_depth(depth(3));
        let ancestor = state.history.tip().expect("seeded");
        let mut run = advance_run(&mut state, 3);
        let before = state.pair.cache.current_block();

        // A substituted middle block. The suffix check catches this before any restore is
        // attempted, which is the earliest of the two layers that can — so the deque survives.
        run[1] = BlockRef { number: run[1].number, hash: B256::with_last_byte(0xf0) };
        let outcome = apply_reorg(&mut state, &reorg_of(ancestor, run, None));

        assert!(matches!(outcome, ReorgOutcome::Unrecoverable { .. }));
        assert_eq!(state.pair.cache.current_block(), before, "the caches are untouched");
        assert_eq!(
            state.pair.retained_depth(),
            3,
            "a frame refused before the restore runs costs the pair nothing"
        );
        assert!(
            matches!(state.pair.readiness.state(), CacheReadiness::Recovering { .. }),
            "the one lifecycle change a bound-but-refused frame makes"
        );
    }

    #[test]
    fn a_lineage_the_generations_do_not_match_clears_the_whole_deque() {
        let (mut state, _) = restored_state_at_depth(depth(3));
        let ancestor = state.history.tip().expect("seeded");
        let run = advance_run(&mut state, 3);
        let before = state.pair.cache.current_block();

        // Straight at the pair, past `apply_reorg`'s suffix check. That check is the replay
        // driver's own history speaking, and the ExEx's notification hook has no equivalent — so
        // the pair has to hold this line for itself, and this is the test that says it does.
        let mut abandoned: Vec<(u64, B256)> =
            run.iter().map(|block| (block.number, block.hash)).collect();
        abandoned[1].1 = B256::with_last_byte(0xf0);
        let lineage = ExpectedLineage::new((ancestor.number, ancestor.hash), &abandoned)
            .expect("the shape is still a chain");

        let policy_id = state.config.cache_policy_id();
        let ReplayState { pair, history, .. } = &mut state;
        assert!(
            try_deep_recovery(pair, &*history, &lineage, policy_id).is_none(),
            "a generation the lineage does not describe cannot be landed on"
        );

        assert_eq!(pair.cache.current_block(), before, "the caches are untouched");
        assert_eq!(
            pair.retained_depth(),
            0,
            "every generation is on the branch the caller withdrew, and partial truncation would \
             leave a run whose newest end nothing vouches for"
        );
    }

    #[test]
    fn a_missing_middle_undo_record_refuses_and_keeps_everything() {
        let (mut state, _) = restored_state_at_depth(depth(3));
        let ancestor = state.history.tip().expect("seeded");
        let run = advance_run(&mut state, 3);
        let before = state.pair.cache.current_block();

        // The trie half can still reach the ancestor; the flat half cannot. Both have to, and the
        // preflight is where that is found — not half way through the rollback.
        state.pair.cache.prune_undo_below(ANCHOR_BLOCK + 2);

        let outcome = apply_reorg(&mut state, &reorg_of(ancestor, run, None));
        assert!(matches!(outcome, ReorgOutcome::Unrecoverable { .. }));
        assert_eq!(state.pair.cache.current_block(), before, "the caches are untouched");
        assert_eq!(
            state.pair.retained_depth(),
            3,
            "the flat log's gap says nothing about the generations, so they are kept"
        );
    }

    #[test]
    fn a_generation_whose_root_disagrees_with_the_canonical_header_is_refused() {
        let (mut state, _) = restored_state_at_depth(depth(2));
        let ancestor = state.history.tip().expect("seeded");
        let run = advance_run(&mut state, 2);
        let before = state.pair.cache.current_block();

        // The authentication this whole path rests on: the landing generation's own state root has
        // to equal what the canonical header says the ancestor's root is. Moved on the chain's
        // side rather than the pair's, because that is the direction a real disagreement comes
        // from — the pair derived its root, the header is what it is checked against.
        let entry = state
            .history
            .entries
            .iter_mut()
            .find(|entry| entry.hash == ancestor.hash)
            .expect("the ancestor is in the history");
        entry.state_root = B256::with_last_byte(0x99);

        let outcome = apply_reorg(&mut state, &reorg_of(ancestor, run, None));
        assert!(matches!(outcome, ReorgOutcome::Unrecoverable { .. }));
        assert_eq!(state.pair.cache.current_block(), before, "the caches are untouched");
        assert_eq!(
            state.pair.retained_depth(),
            2,
            "a root mismatch is not evidence the branch was withdrawn, so the deque survives"
        );
    }

    #[test]
    fn the_deque_total_is_a_union_and_not_a_multiple_of_one_generation() {
        let (mut state, _) = restored_state_at_depth(depth(3));
        advance_run(&mut state, 3);
        assert_eq!(state.pair.retained_depth(), 3);

        let deque = state.pair.retained_deque_bytes();
        assert_eq!(deque.generations, 3);
        assert_eq!(deque.total_bytes, deque.unshared_bytes + deque.shared_pool_bytes);

        // A union can never exceed the sum it replaces, whatever the population.
        let one = state.pair.retained_generation_bytes(true);
        assert!(deque.total_bytes <= one.complete_exclusive_bytes * 3);

        // This fixture is a one-account snapshot whose blocks touch no storage, so it holds no
        // revealed storage tries and no retained-path slices — there is nothing to share, and the
        // union degenerates to the sum. Asserted rather than glossed over: a later fixture change
        // that introduces sharing should fail here and be re-read, not silently weaken the test.
        // The sharing case itself is covered where a cache can actually have it, in
        // `trie_cache`'s `a_clone_shares_its_retained_path_slices_by_identity`.
        assert_eq!(deque.shared_pool_bytes, 0, "this fixture has nothing shareable");
        assert_eq!(deque.total_bytes, one.complete_exclusive_bytes * 3);

        // Nothing the live cache still points at is charged to the deque — it would not be freed
        // by dropping it. Asserted by identity rather than by arithmetic on the totals.
        let live: std::collections::HashSet<usize> =
            state.pair.trie_cache.shared_allocations().into_iter().map(|(id, _)| id).collect();
        let charged: std::collections::HashSet<usize> = state
            .pair
            .retained
            .iter()
            .flat_map(|generation| generation.trie_cache.shared_allocations())
            .map(|(id, _)| id)
            .filter(|id| !live.contains(id))
            .collect();
        assert_eq!(
            charged.len(),
            deque.shared_allocations,
            "the pool is exactly the allocations the live cache does not hold"
        );
    }

    #[test]
    fn a_deque_of_generations_that_really_share_costs_less_than_their_sum() {
        let (mut state, _) = restored_state_at_depth(depth(3));

        // The checkpoint fixture's blocks touch no storage, so its caches have nothing shareable
        // and the union degenerates to a sum. Build a cache that does: a warm slot gives it a
        // `retained_storage_paths` entry, whose `Arc<[Nibbles]>` slice a clone shares rather than
        // copies. Installed directly, because what is under test is the accounting and not the
        // path that produces it.
        let address = Address::repeat_byte(0x11);
        let slot = B256::repeat_byte(0x22);
        let mut accessed = BlockAccessedState::default();
        accessed
            .accounts
            .insert(address, AccountData { nonce: 3, balance: U256::from(7), code_hash: None });
        accessed.storage.insert((address, slot), U256::from(1));
        let mut values = NetworkStateCache::new(
            Box::new(LastNBlocksPolicy::new(60)),
            Box::new(LastNBlocksPolicy::new(30)),
        );
        values.on_block_executed(1, &accessed);
        let mut warm = partial_stateless::PartialTrieNodeCache::new();
        warm.retain_from_value_cache(&values);

        state.pair.retained.clear();
        for offset in 0..3u64 {
            state.pair.retained.push_back(RetainedGeneration {
                trie_cache: warm.clone(),
                block_hash: B256::with_last_byte(offset as u8),
                block_number: offset,
                accepted_head: None,
            });
        }

        let deque = state.pair.retained_deque_bytes();
        assert_eq!(deque.generations, 3);
        assert!(deque.shared_pool_bytes > 0, "these generations really do share something");

        // The two figures bracket the resident cost from either side, and the ordering is the
        // whole point of reporting both: the union can only be smaller.
        assert!(deque.total_bytes <= deque.generation_sum_bytes);

        // The claim §4.3 rests on, now on a population that can show it: the union is strictly
        // less than the sum, and the gap is exactly the sharing counted once instead of K times.
        // Measured on a clone, not on `warm` itself. The deque holds clones, and a clone is
        // strictly smaller: `Vec::clone` allocates for the length rather than the capacity, so
        // `retained_account_paths` compacts across the copy. Using the original as the baseline
        // here overstates the sum by that difference and the identity below misses by exactly it —
        // which is the same reason §4.3 has to measure a union rather than multiply one reading.
        let one_total = warm.clone().memory_breakdown().total_bytes();
        assert!(
            deque.total_bytes < one_total * 3,
            "union {} should be under 3x one generation {}",
            deque.total_bytes,
            one_total * 3
        );
        let shared_once: usize =
            warm.clone().shared_allocations().iter().map(|(_, bytes)| bytes).sum::<usize>();
        assert_eq!(
            deque.total_bytes + shared_once * 2,
            one_total * 3,
            "the whole gap is the shared allocations charged twice too often by a sum"
        );

        // And the live cache's own share is excluded: install the same cache live and the pool
        // empties, because dropping the deque would not free what the pair still points at.
        state.pair.trie_cache = warm;
        let with_live = state.pair.retained_deque_bytes();
        assert_eq!(with_live.shared_pool_bytes, 0);
        assert!(with_live.total_bytes < deque.total_bytes);
    }

    #[test]
    fn an_empty_deque_costs_nothing_and_says_so() {
        let (state, _) = restored_state_at_depth(depth(3));
        let deque = state.pair.retained_deque_bytes();
        assert_eq!(deque.generations, 0, "a restored pair has not retained anything yet");
        assert_eq!(deque.total_bytes, 0);
        assert_eq!(deque.shared_allocations, 0);
        assert_eq!(deque.generation_sum_bytes, 0);
    }

    /// Everything §4.4's invariance table calls "untouched", in one comparable value.
    ///
    /// Wider than `structurally_eq` and `retention_fingerprint` on purpose. Those cover the trie's
    /// state root, warm sets, trie maps and retained paths, and a restore that half-applied could
    /// still leave all four intact while having moved the flat cache, popped an undo record,
    /// promoted the tracker or consumed a generation. Each line below is one of those.
    #[derive(Debug, PartialEq)]
    struct PairSnapshot {
        cache_block: u64,
        cache_root: B256,
        undo_log: B256,
        trie_state_root: Option<B256>,
        trie_cache_root: B256,
        retention: B256,
        readiness: String,
        accepted_head: Option<(u64, B256)>,
        /// Per retained generation: its tag, the header it carries, and its trie's own roots.
        generations: Vec<(u64, B256, Option<(u64, B256)>, Option<B256>, B256)>,
    }

    fn snapshot(pair: &mut CoordinatedPair) -> PairSnapshot {
        PairSnapshot {
            cache_block: pair.cache.current_block(),
            cache_root: pair.cache.cache_root(),
            undo_log: pair.cache.undo_log_fingerprint(),
            trie_state_root: pair.trie_cache.state_root(),
            trie_cache_root: pair.trie_cache.cache_root(),
            retention: pair.trie_cache.retention_fingerprint(),
            readiness: format!("{:?}", pair.readiness.state()),
            accepted_head: pair.accepted_head.as_ref().map(|h| (h.number, h.hash())),
            generations: pair
                .retained
                .iter_mut()
                .map(|generation| {
                    (
                        generation.block_number,
                        generation.block_hash,
                        generation.accepted_head.as_ref().map(|h| (h.number, h.hash())),
                        generation.trie_cache.state_root(),
                        generation.trie_cache.cache_root(),
                    )
                })
                .collect(),
        }
    }

    #[test]
    fn a_malformed_frame_changes_nothing_at_all() {
        let (mut state, _) = restored_state_at_depth(depth(3));
        advance_run(&mut state, 3);
        let before = snapshot(&mut state.pair);

        // Row 1: not even readiness moves, because the frame never binds an ancestor.
        let bogus = BlockRef { number: 500, hash: B256::with_last_byte(0xee) };
        let outcome = apply_reorg(&mut state, &reorg_of(bogus, Vec::new(), None));
        assert!(matches!(outcome, ReorgOutcome::Malformed { .. }));
        assert_eq!(snapshot(&mut state.pair), before);
    }

    #[test]
    fn a_bound_but_refused_frame_moves_readiness_and_nothing_else() {
        let (mut state, _) = restored_state_at_depth(depth(2));
        let ancestor = state.history.tip().expect("seeded");
        let run = advance_run(&mut state, 3);
        let before = snapshot(&mut state.pair);

        // Row 2: too deep for this pair. `apply_reorg` marks recovery as soon as the ancestor is
        // bound — before the depth check — so "byte-identical at every failure" is false by
        // design, and this is the one field that legitimately moves.
        let outcome = apply_reorg(&mut state, &reorg_of(ancestor, run, None));
        assert!(matches!(outcome, ReorgOutcome::Unrecoverable { .. }));

        let after = snapshot(&mut state.pair);
        assert_ne!(after.readiness, before.readiness, "the tracker went Recovering");
        assert_eq!(PairSnapshot { readiness: before.readiness.clone(), ..after }, before);
    }

    #[test]
    fn a_failed_depth_d_preflight_moves_nothing_but_readiness() {
        let (mut state, _) = restored_state_at_depth(depth(3));
        let ancestor = state.history.tip().expect("seeded");
        let run = advance_run(&mut state, 3);
        // The flat half cannot reach the ancestor, so the preflight fails after the lineage check
        // has already passed — the deepest a refusal gets before phase 2.
        state.pair.cache.prune_undo_below(ANCHOR_BLOCK + 2);
        let before = snapshot(&mut state.pair);

        // Row 3: the caches, the undo log, the tracker's history, the accepted head and the whole
        // deque are all still there. Readiness moves for the same reason as row 2.
        let outcome = apply_reorg(&mut state, &reorg_of(ancestor, run, None));
        assert!(matches!(outcome, ReorgOutcome::Unrecoverable { .. }));

        let after = snapshot(&mut state.pair);
        assert_eq!(after.generations.len(), 3, "no generation was consumed");
        assert_eq!(PairSnapshot { readiness: before.readiness.clone(), ..after }, before);
    }

    #[test]
    fn a_lineage_mismatch_clears_the_deque_and_leaves_everything_else() {
        let (mut state, _) = restored_state_at_depth(depth(3));
        let ancestor = state.history.tip().expect("seeded");
        let run = advance_run(&mut state, 3);
        let before = snapshot(&mut state.pair);

        let mut abandoned: Vec<(u64, B256)> =
            run.iter().map(|block| (block.number, block.hash)).collect();
        abandoned[1].1 = B256::with_last_byte(0xf0);
        let lineage = ExpectedLineage::new((ancestor.number, ancestor.hash), &abandoned)
            .expect("the shape is still a chain");

        let policy_id = state.config.cache_policy_id();
        let ReplayState { pair, history, .. } = &mut state;
        assert!(try_deep_recovery(pair, &*history, &lineage, policy_id).is_none());

        // Row 4: the deque goes wholesale — every generation is on the withdrawn branch — and the
        // caches, the undo log, the accepted head and readiness are all exactly as they were.
        // Readiness does *not* move here, because this path was entered past `apply_reorg`.
        let after = snapshot(pair);
        assert!(after.generations.is_empty(), "the whole deque is cleared, not truncated");
        assert_eq!(PairSnapshot { generations: before.generations.clone(), ..after }, before);
    }

    #[test]
    fn a_retention_depth_is_refused_outside_its_range() {
        assert!(RetentionDepth::new(0).is_err(), "zero is retention off, not a depth");
        assert!(RetentionDepth::new(1).is_ok());
        assert!(RetentionDepth::new(MAX_RETENTION_DEPTH).is_ok());
        assert!(RetentionDepth::new(MAX_RETENTION_DEPTH + 1).is_err());
        // The history window has to cover the deepest configurable undo, or a legal configuration
        // would pass validation and then always refuse. Asserted at compile time beside
        // `HISTORY_DEPTH`; restated here so the reason is discoverable from the test suite.
        assert!(HISTORY_DEPTH > MAX_RETENTION_DEPTH as usize);
    }

    #[test]
    fn a_depth_one_reorg_is_undone_against_the_retained_generation() {
        let (mut state, _) = restored_state();
        let ancestor = state.history.tip().expect("seeded");
        let undone = advance(&mut state, ANCHOR_BLOCK + 1, 0xaa, true);
        assert_eq!(state.pair.cache.current_block(), ANCHOR_BLOCK + 1);

        let winning = BlockRef { number: ANCHOR_BLOCK + 1, hash: B256::with_last_byte(0xbb) };
        let outcome = apply_reorg(&mut state, &reorg_of(ancestor, vec![undone], Some(winning)));

        let ReorgOutcome::Applied { ancestor: at, undone: gave_back, revert, winning_tip } =
            outcome
        else {
            panic!("a depth-1 reorg of this consumer's own branch is exactly what it can undo")
        };
        assert_eq!(at, ancestor);
        assert_eq!(gave_back, vec![undone], "a depth-1 undo gives back one block");
        assert!(!revert, "a reorg replaces the branch it abandons");
        assert_eq!(winning_tip, Some(winning));
        assert_eq!(state.pair.cache.current_block(), ANCHOR_BLOCK, "the flat cache gave one back");
        assert!(matches!(state.pair.readiness.state(), CacheReadiness::Ready(_)));
        assert_eq!(
            state.history.tip(),
            Some(ancestor),
            "and the consumer no longer stands behind the abandoned block"
        );
    }

    #[test]
    fn a_pure_revert_leaves_the_pair_ready_at_the_ancestor() {
        let (mut state, _) = restored_state();
        let ancestor = state.history.tip().expect("seeded");
        let undone = advance(&mut state, ANCHOR_BLOCK + 1, 0xaa, true);

        let outcome = apply_reorg(&mut state, &reorg_of(ancestor, vec![undone], None));

        let ReorgOutcome::Applied { revert, winning_tip, .. } = outcome else {
            panic!("a revert is the same undo with nothing replacing the branch")
        };
        assert!(revert);
        assert_eq!(winning_tip, None);
        assert!(matches!(state.pair.readiness.state(), CacheReadiness::Ready(_)));
    }

    #[test]
    fn a_depth_two_reorg_is_unrecoverable_and_leaves_the_pair_recovering() {
        let (mut state, _) = restored_state();
        let ancestor = state.history.tip().expect("seeded");
        let first = advance(&mut state, ANCHOR_BLOCK + 1, 0xaa, true);
        let second = advance(&mut state, ANCHOR_BLOCK + 2, 0xab, true);

        let outcome = apply_reorg(&mut state, &reorg_of(ancestor, vec![first, second], None));

        let ReorgOutcome::Unrecoverable { ancestor: named, depth, .. } = outcome else {
            panic!("K = 1 reaches one block; anything deeper needs a snapshot")
        };
        assert_eq!(named, ancestor, "naming the ancestor is what makes the refusal actionable");
        assert_eq!(depth, 2);
        assert!(
            matches!(state.pair.readiness.state(), CacheReadiness::Recovering),
            "the producer left this branch, so the pair must refuse every further commit"
        );
    }

    #[test]
    fn an_ancestor_this_consumer_never_verified_is_unbound_and_touches_nothing() {
        let (mut state, _) = restored_state();
        let undone = advance(&mut state, ANCHOR_BLOCK + 1, 0xaa, true);
        let foreign = BlockRef { number: ANCHOR_BLOCK, hash: B256::repeat_byte(0x99) };
        let before = state.pair.fingerprint();

        let outcome = apply_reorg(&mut state, &reorg_of(foreign, vec![undone], None));

        assert!(
            matches!(outcome, ReorgOutcome::Unbound { .. }),
            "a target outside this consumer's own history cannot be authenticated by it"
        );
        assert_eq!(state.pair.fingerprint(), before);
        assert!(
            matches!(state.pair.readiness.state(), CacheReadiness::Ready(_)),
            "an unbound frame is a claim about someone else's chain, and moves this pair's \
             lifecycle no more than it moves its caches"
        );
    }

    #[test]
    fn an_abandoned_branch_that_is_not_this_consumers_cannot_be_undone() {
        let (mut state, _) = restored_state();
        let ancestor = state.history.tip().expect("seeded");
        advance(&mut state, ANCHOR_BLOCK + 1, 0xaa, true);
        // Right height, wrong block: a producer describing someone else's branch, or delivery that
        // lost the block this consumer actually verified.
        let sibling = BlockRef { number: ANCHOR_BLOCK + 1, hash: B256::repeat_byte(0xcc) };

        let outcome = apply_reorg(&mut state, &reorg_of(ancestor, vec![sibling], None));

        assert!(matches!(outcome, ReorgOutcome::Unrecoverable { .. }));
        assert_eq!(
            state.pair.cache.current_block(),
            ANCHOR_BLOCK + 1,
            "and nothing was undone on the strength of it"
        );
        assert!(
            matches!(state.pair.readiness.state(), CacheReadiness::Recovering),
            "the ancestor is a block this consumer verified, so it still knows where a recovery \
             has to land; what it lost is the right to perform the undo itself"
        );
    }

    #[test]
    fn a_pair_that_retained_nothing_is_unrecoverable() {
        let (mut state, _) = restored_state();
        let ancestor = state.history.tip().expect("seeded");
        let undone = advance(&mut state, ANCHOR_BLOCK + 1, 0xaa, false);

        let outcome = apply_reorg(&mut state, &reorg_of(ancestor, vec![undone], None));

        assert!(
            matches!(outcome, ReorgOutcome::Unrecoverable { .. }),
            "without the displaced trie there is no generation to go back to"
        );
        assert_eq!(state.pair.cache.current_block(), ANCHOR_BLOCK + 1, "and none was faked");
        assert!(
            matches!(state.pair.readiness.state(), CacheReadiness::Recovering),
            "this frame *was* about this pair's own branch, so its blocks are gone whether or not \
             the undo was available"
        );
    }

    #[test]
    fn only_a_frame_this_consumer_could_authenticate_withdraws_an_announced_branch() {
        // The rule both drivers use to decide whether a winning branch that never arrived is a
        // hole in the record or a goal the producer itself retracted. It is here, and shared,
        // because getting it wrong in either direction is a wrong headline: counting a retraction
        // fails a run for following the chain, and not counting an unaccounted branch hides one.
        let block = BlockRef { number: 1, hash: B256::repeat_byte(0x01) };
        let detail = String::new();

        assert!(ReorgOutcome::Applied {
            ancestor: block,
            undone: vec![block],
            revert: false,
            winning_tip: None
        }
        .withdraws_an_announced_branch());
        assert!(ReorgOutcome::Unrecoverable { ancestor: block, depth: 2, detail: detail.clone() }
            .withdraws_an_announced_branch());
        assert!(
            !ReorgOutcome::Unbound { ancestor: block, depth: 1, detail: detail.clone() }
                .withdraws_an_announced_branch(),
            "a frame about a branch this consumer never stood on retracts nothing it owed"
        );
        assert!(!ReorgOutcome::Malformed { detail }.withdraws_an_announced_branch());
    }

    #[test]
    fn a_reorg_that_abandons_nothing_is_malformed_and_touches_nothing() {
        let (mut state, _) = restored_state();
        let ancestor = state.history.tip().expect("seeded");
        advance(&mut state, ANCHOR_BLOCK + 1, 0xaa, true);
        let before = state.pair.fingerprint();

        let outcome = apply_reorg(&mut state, &reorg_of(ancestor, vec![], None));

        assert!(matches!(outcome, ReorgOutcome::Malformed { .. }));
        assert_eq!(state.pair.fingerprint(), before);
        assert!(
            matches!(state.pair.readiness.state(), CacheReadiness::Ready(_)),
            "a frame that describes no unwind must not stop a pair that is still sound"
        );
    }

    #[test]
    fn a_winning_tip_at_or_below_the_ancestor_is_malformed() {
        let (mut state, _) = restored_state();
        let ancestor = state.history.tip().expect("seeded");
        let undone = advance(&mut state, ANCHOR_BLOCK + 1, 0xaa, true);
        let tip = BlockRef { number: ANCHOR_BLOCK, hash: B256::repeat_byte(0xdd) };

        let outcome = apply_reorg(&mut state, &reorg_of(ancestor, vec![undone], Some(tip)));

        assert!(matches!(outcome, ReorgOutcome::Malformed { .. }));
    }

    #[test]
    fn abandoned_blocks_that_skip_a_height_are_malformed() {
        let (mut state, _) = restored_state();
        let ancestor = state.history.tip().expect("seeded");
        let first = advance(&mut state, ANCHOR_BLOCK + 1, 0xaa, true);
        let skipped = BlockRef { number: ANCHOR_BLOCK + 3, hash: B256::repeat_byte(0xee) };

        let outcome = apply_reorg(&mut state, &reorg_of(ancestor, vec![first, skipped], None));

        assert!(matches!(outcome, ReorgOutcome::Malformed { .. }));
    }

    #[test]
    fn the_history_answers_only_for_blocks_it_verified() {
        let (mut state, state_root) = restored_state();
        let ancestor = state.history.tip().expect("seeded");
        let applied = advance(&mut state, ANCHOR_BLOCK + 1, 0xaa, true);

        assert_eq!(state.history.state_root_of(ancestor.hash).unwrap(), Some(state_root));
        assert_eq!(state.history.state_root_of(applied.hash).unwrap(), Some(state_root));
        assert_eq!(
            state.history.state_root_of(B256::repeat_byte(0x99)).unwrap(),
            None,
            "a consumer with no database cannot fail to read, only fail to know"
        );
    }
}
