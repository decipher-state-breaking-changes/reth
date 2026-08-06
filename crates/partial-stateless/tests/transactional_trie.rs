//! Gates for the copy-on-write transactional trie.
//!
//! The trie cache used to deep-copy every retained storage trie to make a block's transition
//! reversible. It now shares them with the parent generation and copies only what the block
//! writes to, which is only worth having if two properties hold exactly: a committed transition
//! produces the same cache the deep copy produced, and an abandoned one leaves the parent
//! byte-identical. Both are checked here against a forced deep copy of the same transition.

use alloy_primitives::{keccak256, map::B256Map, Address, B256, U256};
use partial_stateless::{
    policy::{AccountData, LastNBlocksPolicy},
    try_compute_trustless_state_root, BlockAccessedState, NetworkStateCache, PartialTrieNodeCache,
};
use reth_primitives_traits::Account;
use reth_trie::HashBuilder;
use reth_trie_common::{proof::ProofRetainer, MultiProof, Nibbles, StorageMultiProof};
use revm_database::{
    states::{StorageSlot, StorageWithOriginalValues},
    AccountStatus, BundleAccount, BundleState,
};
use revm_state::AccountInfo;

const ACCOUNTS: usize = 24;
/// Accounts below this index own a storage trie; the rest are plain.
const CONTRACTS: usize = 16;
const SLOTS_PER_CONTRACT: usize = 6;
/// The first contract owns a trie far wider than the window keeps warm, so retention blinds most
/// of it and a write to a cold slot is a genuine mid-transition failure rather than a rejected
/// input.
const WIDE_CONTRACT_SLOTS: usize = 64;

/// Slots the parent trie holds for a contract, of which only the first [`SLOTS_PER_CONTRACT`] are
/// ever warm.
const fn slots_in_trie(index: usize) -> usize {
    if index == 0 {
        WIDE_CONTRACT_SLOTS
    } else {
        SLOTS_PER_CONTRACT
    }
}

fn address(index: usize) -> Address {
    let mut bytes = [0u8; 20];
    bytes[..8].copy_from_slice(&keccak256(index.to_be_bytes())[..8]);
    Address::from(bytes)
}

fn slot(index: usize) -> B256 {
    keccak256(((index as u64) | 0xdead_0000).to_be_bytes())
}

fn account(index: usize) -> Account {
    Account {
        nonce: index as u64 + 1,
        balance: U256::from(1_000 + index),
        bytecode_hash: (index < CONTRACTS).then(|| keccak256(index.to_be_bytes())),
    }
}

fn slot_value(account_index: usize, slot_index: usize) -> U256 {
    U256::from((account_index as u64 + 1) * 1_000 + slot_index as u64 + 1)
}

fn account_info(account: Account) -> AccountInfo {
    AccountInfo {
        balance: account.balance,
        nonce: account.nonce,
        code_hash: account.bytecode_hash.unwrap_or_default(),
        code: None,
        ..Default::default()
    }
}

/// A parent state of [`ACCOUNTS`] accounts, [`CONTRACTS`] of which own a storage trie, together
/// with a proof of every leaf so the whole thing can be revealed into a cold cache.
fn parent_state() -> (MultiProof, B256) {
    let mut storages = B256Map::default();
    let mut account_leaves = Vec::new();

    for index in 0..ACCOUNTS {
        let storage_root = if index < CONTRACTS {
            let mut slots: Vec<_> = (0..slots_in_trie(index))
                .map(|slot_index| {
                    (
                        Nibbles::unpack(keccak256(slot(index * 100 + slot_index))),
                        slot_value(index, slot_index),
                    )
                })
                .collect();
            slots.sort_by_key(|slot| slot.0);

            let mut builder = HashBuilder::default()
                .with_proof_retainer(ProofRetainer::from_iter(slots.iter().map(|(path, _)| *path)));
            for (path, value) in &slots {
                builder.add_leaf(*path, &alloy_rlp::encode(value));
            }
            let root = builder.root();
            storages.insert(
                keccak256(address(index)),
                StorageMultiProof {
                    root,
                    subtree: builder.take_proof_nodes(),
                    branch_node_masks: Default::default(),
                },
            );
            root
        } else {
            reth_trie_common::EMPTY_ROOT_HASH
        };

        account_leaves.push((
            Nibbles::unpack(keccak256(address(index))),
            account(index).into_trie_account(storage_root),
        ));
    }

    account_leaves.sort_by_key(|leaf| leaf.0);
    let mut builder = HashBuilder::default()
        .with_proof_retainer(ProofRetainer::from_iter(account_leaves.iter().map(|(p, _)| *p)));
    for (path, trie_account) in &account_leaves {
        builder.add_leaf(*path, &alloy_rlp::encode(trie_account));
    }
    let state_root = builder.root();

    (
        MultiProof {
            account_subtree: builder.take_proof_nodes(),
            branch_node_masks: Default::default(),
            storages,
        },
        state_root,
    )
}

/// Everything the parent state touched, so retention keeps the whole fixture warm.
fn full_access() -> BlockAccessedState {
    let mut accessed = BlockAccessedState::default();
    for index in 0..ACCOUNTS {
        let account = account(index);
        accessed.accounts.insert(
            address(index),
            AccountData {
                nonce: account.nonce,
                balance: account.balance,
                code_hash: account.bytecode_hash,
            },
        );
        if index < CONTRACTS {
            for slot_index in 0..SLOTS_PER_CONTRACT {
                accessed.storage.insert(
                    (address(index), slot(index * 100 + slot_index)),
                    slot_value(index, slot_index),
                );
            }
        }
    }
    accessed
}

/// A value cache warmed by the parent state, then by each of `blocks` in turn.
///
/// Rebuilt rather than cloned because [`NetworkStateCache`] is deliberately not [`Clone`], and the
/// differential needs to drive the same block from the same parent twice.
fn values_after(blocks: &[(u64, BlockAccessedState)]) -> NetworkStateCache {
    let mut values = NetworkStateCache::new(
        Box::new(LastNBlocksPolicy::new(60)),
        Box::new(LastNBlocksPolicy::new(30)),
    );
    values.on_block_executed(1, &full_access());
    for (number, accessed) in blocks {
        values.on_block_executed(*number, accessed);
    }
    values
}

/// A warm cache at the parent state, paired with the value cache it is synchronized to.
fn warm_cache() -> (PartialTrieNodeCache, NetworkStateCache, B256) {
    let (proof, state_root) = parent_state();
    let mut trie = PartialTrieNodeCache::new();
    let revealed =
        try_compute_trustless_state_root(proof, &mut trie, &BundleState::default()).unwrap();
    assert_eq!(revealed, state_root);

    let values = values_after(&[]);
    trie.retain_from_value_cache(&values);
    trie.validate_against_value_cache(&values).unwrap();

    (trie, values, state_root)
}

/// A block that bumps two contracts' nonces and writes one slot in each of them.
fn transition() -> (BundleState, BlockAccessedState) {
    let changed = [0usize, 3];
    let mut bundle = BundleState::default();
    let mut accessed = BlockAccessedState::default();

    for index in changed {
        let previous = account(index);
        let updated = Account { nonce: previous.nonce + 1, ..previous };

        let mut storage = StorageWithOriginalValues::default();
        let written = slot(index * 100);
        storage.insert(
            U256::from_be_slice(written.as_slice()),
            StorageSlot {
                previous_or_original_value: slot_value(index, 0),
                present_value: slot_value(index, 0) + U256::from(7),
            },
        );

        bundle.state.insert(
            address(index),
            BundleAccount {
                info: Some(account_info(updated)),
                original_info: Some(account_info(previous)),
                storage,
                status: AccountStatus::Changed,
            },
        );
        accessed.accounts.insert(
            address(index),
            AccountData {
                nonce: updated.nonce,
                balance: updated.balance,
                code_hash: updated.bytecode_hash,
            },
        );
        accessed.storage.insert((address(index), written), slot_value(index, 0) + U256::from(7));
    }

    (bundle, accessed)
}

/// Applies one block to a snapshot of `parent`, optionally forcing the snapshot to deep-copy every
/// storage trie first, and returns the committed child.
fn apply_block(
    parent: &PartialTrieNodeCache,
    bundle: &BundleState,
    accessed: &BlockAccessedState,
    force_deep_copy: bool,
) -> (PartialTrieNodeCache, B256, NetworkStateCache) {
    let mut child = parent.clone();
    if force_deep_copy {
        child.materialize_shared_storage_tries();
    }

    let root = try_compute_trustless_state_root(MultiProof::default(), &mut child, bundle).unwrap();

    let next_values = values_after(&[(2, accessed.clone())]);
    child.retain_from_value_cache(&next_values);

    (child, root, next_values)
}

#[test]
fn a_shared_snapshot_produces_the_same_cache_as_a_deep_copy() {
    let (parent, _, _) = warm_cache();
    let (bundle, accessed) = transition();

    let (shared, shared_root, shared_values) = apply_block(&parent, &bundle, &accessed, false);
    let (copied, copied_root, _) = apply_block(&parent, &bundle, &accessed, true);

    assert_eq!(shared_root, copied_root);
    assert_eq!(shared.state_root(), copied.state_root());
    assert_eq!(shared.cache_root(), copied.cache_root());
    assert!(
        shared.structurally_eq(&copied),
        "copy-on-write and deep-copy transitions diverged structurally"
    );
    assert_eq!(shared.shape_metrics(), copied.shape_metrics());

    // The retained shape has to survive the differential too, not just the root: a snapshot that
    // silently kept more nodes would still hash correctly while leaking memory every block.
    let mut shared = shared;
    shared.validate_against_value_cache(&shared_values).unwrap();
}

#[test]
fn a_block_copies_only_the_storage_tries_it_writes_to() {
    let (parent, _, _) = warm_cache();
    let (bundle, accessed) = transition();

    let (child, _, _) = apply_block(&parent, &bundle, &accessed, false);

    assert_eq!(child.storage_trie_count(), CONTRACTS);
    // Two contracts were written; everything else must still be shared with the parent, both after
    // the transition and after retention pruned the child.
    assert_eq!(child.shared_storage_trie_count(), CONTRACTS - 2);
}

#[test]
fn an_abandoned_transition_leaves_the_parent_exactly_as_it_was() {
    let (mut parent, values, parent_root) = warm_cache();

    let mut reference = parent.clone();
    reference.materialize_shared_storage_tries();

    // Two writes, one of which lands on a storage path retention blinded. The transition applies
    // the reachable one first and only then discovers it needs a proof, so the snapshot is already
    // dirty when it is abandoned — the case a predicted write set would get wrong.
    let (mut bundle, _) = transition();
    let cold = slot(WIDE_CONTRACT_SLOTS - 1);
    let mut storage = StorageWithOriginalValues::default();
    storage.insert(
        U256::from_be_slice(cold.as_slice()),
        StorageSlot {
            previous_or_original_value: slot_value(0, WIDE_CONTRACT_SLOTS - 1),
            present_value: U256::from(1),
        },
    );
    bundle.state.get_mut(&address(0)).unwrap().storage.extend(storage);

    {
        let mut child = parent.clone();
        try_compute_trustless_state_root(MultiProof::default(), &mut child, &bundle)
            .expect_err("a write to a blinded storage path must ask for a proof");
        assert!(
            child.shared_storage_trie_count() < child.storage_trie_count(),
            "the failed transition never wrote anything, so it does not test rollback"
        );
        // `child` is dropped here, which is the entire rollback.
    }

    assert_eq!(parent.state_root(), Some(parent_root));
    assert!(
        parent.structurally_eq(&reference),
        "an abandoned transition mutated the parent generation"
    );
    parent.validate_against_value_cache(&values).unwrap();
}

#[test]
fn a_successful_transition_does_not_touch_the_parent_before_it_is_committed() {
    let (mut parent, values, parent_root) = warm_cache();
    let (bundle, accessed) = transition();

    let mut reference = parent.clone();
    reference.materialize_shared_storage_tries();

    let (_child, child_root, _) = apply_block(&parent, &bundle, &accessed, false);

    assert_ne!(child_root, parent_root);
    assert_eq!(parent.state_root(), Some(parent_root));
    assert!(
        parent.structurally_eq(&reference),
        "an uncommitted transition mutated the parent generation"
    );
    parent.validate_against_value_cache(&values).unwrap();
}

#[test]
fn repeated_retention_is_a_no_op_which_is_why_it_can_be_skipped() {
    // Retention is skipped for a storage trie the transition never wrote to and whose retained
    // slot set did not move. That is only sound because running it again would change nothing.
    let (mut trie, values, _) = warm_cache();

    let mut reference = trie.clone();
    reference.materialize_shared_storage_tries();

    trie.retain_from_value_cache(&values);
    trie.retain_from_value_cache(&values);

    assert!(trie.structurally_eq(&reference), "re-running retention changed the trie");
    trie.validate_against_value_cache(&values).unwrap();
}

#[test]
fn a_trie_that_loses_slots_is_still_pruned() {
    // The skip keys on the retained slot set as well as on writes: an eviction shrinks the set
    // without touching the trie, and skipping there would retain nodes forever.
    let (parent, _, _) = warm_cache();

    // Refresh everything except the first contract's slots so the window drops them.
    let mut accessed = full_access();
    accessed.storage.retain(|(address, _), _| *address != self::address(0));
    let refreshes: Vec<_> = (2..=32).map(|block| (block, accessed.clone())).collect();
    let evicted_values = values_after(&refreshes);

    let mut child = parent.clone();
    child.retain_from_value_cache(&evicted_values);

    assert_eq!(child.storage_trie_count(), CONTRACTS - 1);
    assert!(!child.tracks_storage(&address(0), &slot(0)));
    child.validate_against_value_cache(&evicted_values).unwrap();
}

#[test]
fn a_simultaneous_account_and_slot_insertion_matches_a_full_rebuild_on_real_tries() {
    // Reveal the complete authenticated fixture, then make contract 0 cold. Its account and
    // storage paths are genuinely blinded here; this is not a membership-only comparison over an
    // empty sparse trie.
    let (initial_proof, state_root) = parent_state();
    let mut parent = PartialTrieNodeCache::new();
    assert_eq!(
        try_compute_trustless_state_root(initial_proof, &mut parent, &BundleState::default())
            .unwrap(),
        state_root
    );

    let cold_contract = address(0);
    let mut initial_access = full_access();
    initial_access.accounts.remove(&cold_contract);
    initial_access.storage.retain(|(address, _), _| *address != cold_contract);
    let mut values = NetworkStateCache::new(
        Box::new(LastNBlocksPolicy::new(60)),
        Box::new(LastNBlocksPolicy::new(30)),
    );
    values.on_block_executed(1, &initial_access);
    parent.retain_from_value_cache(&values);
    parent.validate_against_value_cache(&values).unwrap();
    assert!(!parent.contains_account_path(&cold_contract));

    // Updating contract 0 makes its account and first slot enter their warm sets in the same
    // block. Supply the parent proof to both children, exactly as a live miss would, so the
    // transition re-reveals the cold authenticated paths before retention runs.
    let (bundle, accessed) = transition();
    let (transition_proof, _) = parent_state();
    let mut incremental = parent.clone();
    let mut reference = parent.clone();
    let incremental_root =
        try_compute_trustless_state_root(transition_proof.clone(), &mut incremental, &bundle)
            .unwrap();
    let reference_root =
        try_compute_trustless_state_root(transition_proof, &mut reference, &bundle).unwrap();
    assert_eq!(incremental_root, reference_root);

    values.on_block_executed(2, &accessed);
    let delta = values.last_block_membership_delta().unwrap();
    assert!(delta.accounts_added.contains(&cold_contract));
    assert!(
        delta.storage_added.iter().any(|(address, _)| *address == cold_contract),
        "the fixture must add account and storage membership together"
    );

    let timings = incremental.retain_from_value_cache(&values);
    assert!(!timings.full_rebuild, "the test must score the delta path");
    reference.retain_reference(&values);

    assert_eq!(incremental.retention_fingerprint(), reference.retention_fingerprint());
    assert_eq!(incremental.cache_root(), reference.cache_root());
    assert!(
        incremental.structurally_eq(&reference),
        "delta and full retention produced different sparse tries"
    );
    incremental.validate_against_value_cache(&values).unwrap();
    reference.validate_against_value_cache(&values).unwrap();
}
