//! Applying a recorded reorg to a standalone pair, with no database behind it.
//!
//! The producer writes a [`Reorg`] frame before the winning branch's commits, and until S4 both
//! drivers refused it: the batch replay warned and kept going, the follower stopped publishing.
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
    coordination::ProviderResult, try_depth_one_recovery, CanonicalStateRoots,
};
use std::collections::VecDeque;
use tracing::{info, warn};

use crate::driver::ReplayState;

/// How many verified blocks are kept for recovery questions.
///
/// A depth-1 undo needs two. The rest is there so that a deeper reorg can still be *checked*
/// against this consumer's own branch before it is refused — a refusal that names the right
/// ancestor is what lets recovery ask for a snapshot at that exact block.
const HISTORY_DEPTH: usize = 128;

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
    pub(crate) fn restored_at(block: BlockRef, state_root: B256) -> Self {
        let mut entries = VecDeque::with_capacity(HISTORY_DEPTH);
        entries.push_back(VerifiedBlock { number: block.number, hash: block.hash, state_root });
        Self { entries }
    }

    /// Records a block this consumer validated, with the state root it derived for it.
    pub(crate) fn record(&mut self, block: BlockRef, state_root: B256) {
        if self.entries.len() == HISTORY_DEPTH {
            self.entries.pop_front();
        }
        self.entries.push_back(VerifiedBlock {
            number: block.number,
            hash: block.hash,
            state_root,
        });
    }

    /// The newest block this consumer stands behind.
    pub(crate) fn tip(&self) -> Option<BlockRef> {
        self.entries.back().map(|entry| BlockRef { number: entry.number, hash: entry.hash })
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
        /// The block that was given back.
        undone: BlockRef,
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
    if depth != 1 {
        return ReorgOutcome::Unrecoverable {
            ancestor,
            depth,
            detail: format!(
                "a reorg {depth} blocks deep needs a snapshot at the common ancestor; the \
                 retained generation reaches exactly one block"
            ),
        }
    }

    let ReplayState { pair, history, config, .. } = state;
    let policy_id = config.cache_policy_id();
    if try_depth_one_recovery(pair, &*history, ancestor.hash, policy_id).is_none() {
        return ReorgOutcome::Unrecoverable {
            ancestor,
            depth,
            detail: "the retained generation could not restore the common ancestor".to_string(),
        }
    }
    let undone = reorg.abandoned[0];
    history.rewind_above(ancestor.number);
    let revert = reorg.winning_tip.is_none();
    info!(
        target: "ps_replay",
        ancestor = ancestor.number,
        undone = undone.number,
        revert,
        "Undid one block against the retained generation; the pair is back at the common ancestor"
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
struct VerifiedBlock {
    number: u64,
    hash: B256,
    /// Computed here, not read from the frame. This is the whole reason the type exists.
    state_root: B256,
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
    use partial_stateless_validator::{admit_block, BlockAdmission};
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
        let state = restore(&manifest(), &checkpoint, &chunks).expect("the fixture restores");
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
        state.history.record(block, state_root);
        block
    }

    fn reorg_of(ancestor: BlockRef, abandoned: Vec<BlockRef>, tip: Option<BlockRef>) -> Reorg {
        Reorg { common_ancestor: ancestor, abandoned, winning_tip: tip }
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
        assert_eq!(gave_back, undone);
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
            undone: block,
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
