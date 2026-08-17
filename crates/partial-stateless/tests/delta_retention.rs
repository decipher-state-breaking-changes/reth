//! The delta-driven retention path must equal the full rebuild, block after block.
//!
//! Retention is the largest validator phase, and a live build-parity run measured where it goes:
//! preparing the retained key sets is 23% of it, and almost all of that work re-derives sets that a
//! block barely moved. The optimization is therefore to patch those sets from the value cache's
//! undo record instead of rebuilding them — which is only sound if the patched result is
//! indistinguishable from the rebuilt one on every path, including the ones a reorg takes.
//!
//! These tests run both implementations over the same block sequence and compare them after every
//! block. The full rebuild is the specification; the incremental path has no independent definition
//! of correct.

use alloy_primitives::{keccak256, Address, B256, U256};
use partial_stateless::{
    accessed_state::BlockAccessedState,
    policy::{AccountData, LastNBlocksPolicy},
    MembershipDelta, NetworkStateCache, PartialTrieNodeCache,
};

/// Windows short enough that eviction fires inside a test-length run, which is the only way the
/// *removal* half of the delta is ever exercised.
const ACCOUNT_WINDOW: u64 = 6;
const STORAGE_WINDOW: u64 = 3;

fn cache() -> NetworkStateCache {
    NetworkStateCache::new(
        Box::new(LastNBlocksPolicy::new(ACCOUNT_WINDOW)),
        Box::new(LastNBlocksPolicy::new(STORAGE_WINDOW)),
    )
}

fn address(index: u64) -> Address {
    Address::from_slice(&keccak256(index.to_be_bytes())[..20])
}

fn slot(index: u64) -> B256 {
    keccak256((index | 0xfeed_0000).to_be_bytes())
}

/// A block whose access set is a deterministic function of its number, arranged so that keys
/// continuously enter and leave both windows.
fn block_access(number: u64) -> BlockAccessedState {
    let mut accessed = BlockAccessedState::default();
    // A rotating band of accounts: each block introduces some and stops touching others, so the
    // account window is always both admitting and evicting.
    for offset in 0..7 {
        let index = number * 3 + offset;
        accessed.accounts.insert(
            address(index),
            AccountData { nonce: number, balance: U256::from(index), code_hash: None },
        );
    }
    // Storage on a narrower rotation, including addresses that own slots without an account entry
    // of their own in the same block and addresses that lose their last slot.
    for offset in 0..5 {
        let index = number * 2 + offset;
        accessed.storage.insert((address(index), slot(offset)), U256::from(number + offset));
    }
    // One long-lived contract keeps a stable slot set, so the "unchanged, therefore skipped" case
    // is represented rather than only the churning one.
    accessed.storage.insert((address(9_999), slot(1)), U256::from(number));
    accessed
}

/// Applies `number` to the cache and returns the membership delta it produced.
fn apply(values: &mut NetworkStateCache, number: u64) -> MembershipDelta {
    apply_access(values, number, &block_access(number))
}

fn apply_access(
    values: &mut NetworkStateCache,
    number: u64,
    accessed: &BlockAccessedState,
) -> MembershipDelta {
    values.on_block_executed(number, accessed);
    values.last_block_membership_delta().expect("a just-applied block always has an undo record")
}

/// Asserts the two caches agree on everything retention derives and everything it commits.
fn assert_agrees(incremental: &PartialTrieNodeCache, reference: &PartialTrieNodeCache, at: &str) {
    assert_eq!(
        incremental.retention_fingerprint(),
        reference.retention_fingerprint(),
        "retained path sets diverged {at}"
    );
    assert_eq!(incremental.cache_root(), reference.cache_root(), "warm membership diverged {at}");
    assert!(incremental.structurally_eq(reference), "the pruned tries themselves diverged {at}");
}

#[test]
fn the_delta_path_equals_a_full_rebuild_on_every_block() {
    let mut values = cache();
    let mut incremental = PartialTrieNodeCache::new();
    let mut reference = PartialTrieNodeCache::new();

    // Both start from the same full rebuild; only `incremental` is allowed to patch afterwards.
    apply(&mut values, 1);
    incremental.retain_reference(&values);
    reference.retain_reference(&values);
    assert_agrees(&incremental, &reference, "at the first block");

    for number in 2..=40 {
        apply(&mut values, number);
        let timings = incremental.retain_from_value_cache(&values);
        assert!(
            !timings.full_rebuild,
            "block {number} fell back to the full rebuild, so this block compared the reference \
             against itself and proved nothing"
        );
        reference.retain_reference(&values);
        assert_agrees(&incremental, &reference, &format!("at block {number}"));
    }
}

#[test]
fn a_new_account_and_its_first_slot_add_the_account_path_once() {
    let mut values = cache();
    let mut incremental = PartialTrieNodeCache::new();
    let mut reference = PartialTrieNodeCache::new();

    let mut first = BlockAccessedState::default();
    first
        .accounts
        .insert(address(1), AccountData { nonce: 1, balance: U256::from(1), code_hash: None });
    apply_access(&mut values, 1, &first);
    incremental.retain_reference(&values);
    reference.retain_reference(&values);

    // This is the production failure shape: a previously unseen contract enters the account and
    // storage windows in the same block. The old implementation mutated `warm_accounts` before
    // asking whether the address used to be retained, then omitted this account path entirely.
    let contract = address(2);
    let mut second = BlockAccessedState::default();
    second.accounts.insert(
        contract,
        AccountData { nonce: 2, balance: U256::from(2), code_hash: Some(keccak256(b"code")) },
    );
    second.storage.insert((contract, slot(0)), U256::from(3));
    let delta = apply_access(&mut values, 2, &second);
    assert_eq!(delta.accounts_added, vec![contract]);
    assert_eq!(delta.storage_added, vec![(contract, slot(0))]);

    let timings = incremental.retain_from_value_cache(&values);
    assert!(!timings.full_rebuild, "the regression must exercise the delta path");
    reference.retain_reference(&values);
    assert_agrees(&incremental, &reference, "after a simultaneous account and slot insertion");
    assert!(incremental.tracks_account(&contract));
    assert!(incremental.tracks_storage(&contract, &slot(0)));
}

#[test]
fn eviction_is_actually_exercised_or_the_comparison_proves_nothing() {
    let mut values = cache();
    let mut removals = 0usize;
    for number in 1..=40 {
        let delta = apply(&mut values, number);
        removals += delta.accounts_removed.len() + delta.storage_removed.len();
    }
    assert!(
        removals > 0,
        "the fixture must evict, or the delta path's removal half is never run: {removals}"
    );
}

#[test]
fn a_rollback_falls_back_to_the_full_rebuild_rather_than_patching() {
    let mut values = cache();
    let mut incremental = PartialTrieNodeCache::new();

    for number in 1..=10 {
        apply(&mut values, number);
        incremental.retain_from_value_cache(&values);
    }

    // A depth-1 reorg rolls the value cache back one block. The undo record that would describe
    // the *next* step no longer refers to the height the trie cache's derived sets sit at, so the
    // delta must be refused and the sets rebuilt — patching here would carry block 10's keys into
    // a cache that no longer contains them.
    values.rollback_block(10).expect("the newest block is always rollbackable");
    let timings = incremental.retain_from_value_cache(&values);
    assert!(timings.full_rebuild, "a rollback must refuse the delta, not patch with it");

    let mut reference = PartialTrieNodeCache::new();
    reference.retain_reference(&values);
    assert_agrees(&incremental, &reference, "after a rollback");
}

#[test]
fn a_skipped_block_falls_back_rather_than_applying_the_wrong_delta() {
    let mut values = cache();
    let mut incremental = PartialTrieNodeCache::new();

    apply(&mut values, 1);
    incremental.retain_from_value_cache(&values);

    // Two blocks land on the value cache before retention runs again. The newest undo record
    // describes only the second of them, so patching with it would silently drop the first
    // block's keys. The height check is what turns that into a rebuild.
    apply(&mut values, 2);
    apply(&mut values, 3);
    let timings = incremental.retain_from_value_cache(&values);
    assert!(timings.full_rebuild, "a skipped block must refuse the delta, not patch with it");

    let mut reference = PartialTrieNodeCache::new();
    reference.retain_reference(&values);
    assert_agrees(&incremental, &reference, "after a skipped block");
}

#[test]
fn a_clone_keeps_patching_from_where_its_parent_left_off() {
    let mut values = cache();
    let mut parent = PartialTrieNodeCache::new();
    for number in 1..=8 {
        apply(&mut values, number);
        parent.retain_from_value_cache(&values);
    }

    // This is the transactional snapshot the validator applies a block to. Its derived sets come
    // across with it, so the child must be able to patch rather than rebuild — and must still
    // agree with a rebuild when it does.
    let mut child = parent.clone();
    apply(&mut values, 9);
    let timings = child.retain_from_value_cache(&values);
    assert!(
        !timings.full_rebuild,
        "a snapshot inherits its parent's derived sets and patches them"
    );

    let mut reference = PartialTrieNodeCache::new();
    reference.retain_reference(&values);
    assert_agrees(&child, &reference, "on a snapshot's first block");
}

#[test]
fn a_pair_recovered_by_undoing_one_block_keeps_patching_correctly() {
    // Where step 3 and step 4 meet, and the reason the recovery oracle had to land first. A
    // depth-1 reorg rolls the value cache back one block and swaps the trie cache for the
    // generation retained at the parent. Both halves of the pair are then one block behind, so
    // the *next* block's delta is the step from the parent — which is exactly the distance the
    // incremental path is allowed to patch across. If the retained generation's derived sets did
    // not describe the parent, this is where it would show.
    let mut values = cache();
    let mut live = PartialTrieNodeCache::new();
    for number in 1..=10 {
        apply(&mut values, number);
        live.retain_from_value_cache(&values);
    }

    // The generation the coordinated pair retains before applying block 11.
    let retained = live.clone();
    apply(&mut values, 11);
    live.retain_from_value_cache(&values);

    // The reorg: block 11 is abandoned on both halves at once.
    values.rollback_block(11).expect("the newest block is always rollbackable");
    let mut recovered = retained;

    // The winning block 11, applied to the recovered pair.
    apply(&mut values, 11);
    let timings = recovered.retain_from_value_cache(&values);
    assert!(
        !timings.full_rebuild,
        "a recovered pair is exactly one block behind, so it must patch rather than rebuild"
    );

    let mut reference = PartialTrieNodeCache::new();
    reference.retain_reference(&values);
    assert_agrees(&recovered, &reference, "on the first block after a depth-1 recovery");
}
