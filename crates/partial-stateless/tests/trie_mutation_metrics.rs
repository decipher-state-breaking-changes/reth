//! Verifies that the sparse-trie cache reports how much of itself a block dirties.
//!
//! These ratios decide whether the per-block deep clone of the trie is worth replacing with
//! copy-on-write or a journal, so a counter that flatters either side would send that design the
//! wrong way.

use alloy_primitives::{
    keccak256,
    map::{B256Map, HashSet},
    Address, B256, U256,
};
use partial_stateless::{
    policy::{AccountData, LastNBlocksPolicy},
    BlockAccessedState, NetworkStateCache, PartialTrieNodeCache, TrieChangeSet,
    TRIE_SHAPE_PREFIX_LEVELS,
};
use reth_trie_common::{HashedPostState, HashedStorage};

const RETAINED_ACCOUNTS: usize = 32;
const SLOTS_PER_ACCOUNT: usize = 8;

fn address(index: usize) -> Address {
    Address::repeat_byte(index as u8 + 1)
}

fn slot(index: usize) -> B256 {
    B256::repeat_byte(index as u8 + 0x80)
}

/// A cache retaining every account and, for the first two accounts, several storage slots.
fn warm_cache() -> PartialTrieNodeCache {
    let mut accessed = BlockAccessedState::default();
    for index in 0..RETAINED_ACCOUNTS {
        accessed.accounts.insert(
            address(index),
            AccountData { nonce: index as u64, balance: U256::from(index), code_hash: None },
        );
    }
    for account in 0..2 {
        for index in 0..SLOTS_PER_ACCOUNT {
            accessed.storage.insert((address(account), slot(index)), U256::from(index + 1));
        }
    }

    let mut values = NetworkStateCache::new(
        Box::new(LastNBlocksPolicy::new(64)),
        Box::new(LastNBlocksPolicy::new(64)),
    );
    values.on_block_executed(1, &accessed);

    let mut trie = PartialTrieNodeCache::new();
    trie.retain_from_value_cache(&values);
    trie
}

fn changed_accounts(indices: impl IntoIterator<Item = usize>) -> TrieChangeSet {
    TrieChangeSet {
        accounts: indices.into_iter().map(|index| keccak256(address(index))).collect(),
        ..Default::default()
    }
}

fn changed_slots(account: usize, slots: impl IntoIterator<Item = usize>) -> B256Map<HashSet<B256>> {
    let mut map = B256Map::default();
    map.insert(
        keccak256(address(account)),
        slots.into_iter().map(|index| keccak256(slot(index))).collect(),
    );
    map
}

#[test]
fn one_changed_account_dirties_a_small_share_of_the_retained_paths() {
    let trie = warm_cache();

    let metrics = trie.mutation_metrics(&changed_accounts([0]));

    assert_eq!(metrics.retained_account_paths, RETAINED_ACCOUNTS);
    assert_eq!(metrics.dirtied_account_paths, 1);
    assert_eq!(metrics.dirtied_storage_paths, 0);
    assert_eq!(metrics.dirtied_storage_tries, 0);

    let share = metrics.dirtied_path_share();
    assert!(share > 0.0 && share < 0.1, "one path out of many must be a small share: {share}");
}

#[test]
fn prefix_coverage_saturates_at_the_root_and_discriminates_with_depth() {
    let trie = warm_cache();

    let metrics = trie.mutation_metrics(&changed_accounts([0]));

    // Depth zero is the empty prefix: every path shares it, so one changed key covers all of it.
    assert_eq!(
        metrics.account_prefixes[0],
        partial_stateless::PrefixCoverage { retained: 1, dirtied: 1 }
    );
    assert_eq!(metrics.account_prefixes[0].dirtied_share(), 1.0);

    // Deeper levels separate the changed key from the rest, which is the whole point of the metric.
    let deepest = metrics.deepest_account_prefix();
    assert!(
        deepest.retained > 1,
        "the retained set must fan out by the deepest level: {deepest:?}"
    );
    assert_eq!(deepest.dirtied, 1);
    assert!(deepest.dirtied_share() < 1.0);

    // Coverage is monotone: a prefix set can only grow as depth increases.
    for depth in 1..TRIE_SHAPE_PREFIX_LEVELS {
        assert!(
            metrics.account_prefixes[depth].retained >=
                metrics.account_prefixes[depth - 1].retained,
            "retained prefixes shrank between depth {} and {}",
            depth - 1,
            depth
        );
    }
}

#[test]
fn changing_every_retained_account_dirties_everything() {
    let trie = warm_cache();

    let metrics = trie.mutation_metrics(&changed_accounts(0..RETAINED_ACCOUNTS));

    assert_eq!(metrics.dirtied_account_paths, RETAINED_ACCOUNTS);
    for depth in 0..TRIE_SHAPE_PREFIX_LEVELS {
        let coverage = metrics.account_prefixes[depth];
        assert_eq!(
            coverage.dirtied, coverage.retained,
            "every prefix at depth {depth} is dirtied when every leaf is"
        );
        assert_eq!(coverage.dirtied_share(), 1.0);
    }
}

#[test]
fn storage_changes_are_attributed_to_the_trie_that_owns_them() {
    let trie = warm_cache();

    let mut storage = changed_slots(0, [0, 1]);
    storage.extend(changed_slots(1, [0]));
    let metrics = trie.mutation_metrics(&TrieChangeSet { storage, ..Default::default() });

    assert_eq!(metrics.dirtied_storage_paths, 3);
    assert_eq!(metrics.dirtied_storage_tries, 2);
    assert_eq!(metrics.retained_storage_paths, 2 * SLOTS_PER_ACCOUNT);

    // Sorted most-dirtied first, so the head of the list is the trie a sharing scheme cares about.
    let worst = &metrics.per_storage_trie[0];
    assert_eq!(worst.hashed_address, keccak256(address(0)));
    assert_eq!(worst.dirtied_paths, 2);
    assert_eq!(worst.retained_paths, SLOTS_PER_ACCOUNT);
    assert_eq!(metrics.per_storage_trie[1].dirtied_paths, 1);
}

#[test]
fn a_storage_change_also_dirties_the_account_trie() {
    let trie = warm_cache();

    // The account map is empty: only slots changed. The account leaf still has to be rewritten,
    // because it carries the storage root.
    let metrics = trie
        .mutation_metrics(&TrieChangeSet { storage: changed_slots(0, [0]), ..Default::default() });

    assert!(
        metrics.account_prefixes[TRIE_SHAPE_PREFIX_LEVELS - 1].dirtied > 0,
        "changing a slot rewrites its account's leaf and every node above it: {metrics:?}"
    );
}

#[test]
fn a_wipe_dirties_every_retained_path_in_the_trie() {
    let trie = warm_cache();

    let wiped = TrieChangeSet {
        // A wipe lists no individual slots, which is exactly why it has to be handled separately.
        wiped_storage: [keccak256(address(0))].into_iter().collect(),
        ..Default::default()
    };
    let metrics = trie.mutation_metrics(&wiped);

    assert_eq!(metrics.dirtied_storage_paths, SLOTS_PER_ACCOUNT);
    assert_eq!(metrics.dirtied_storage_tries, 1);

    let wiped_trie = metrics
        .per_storage_trie
        .iter()
        .find(|trie| trie.hashed_address == keccak256(address(0)))
        .expect("the wiped trie is retained");
    assert!(wiped_trie.wiped);
    assert_eq!(wiped_trie.dirtied_paths, wiped_trie.retained_paths);
}

#[test]
fn a_key_the_cache_does_not_hold_still_dirties_the_nodes_above_it() {
    let trie = warm_cache();

    // Inserting a brand-new account re-hashes every branch node between it and the root, and those
    // nodes are in the clone. Ignoring changed keys that are not themselves retained would report
    // this block as dirtying nothing and overstate what copy-on-write could save.
    let stranger = keccak256(Address::repeat_byte(0xfe));
    let metrics = trie.mutation_metrics(&TrieChangeSet {
        accounts: [stranger].into_iter().collect(),
        ..Default::default()
    });

    assert_eq!(metrics.dirtied_account_paths, 0, "no retained leaf changed");
    assert_eq!(
        metrics.account_prefixes[0],
        partial_stateless::PrefixCoverage { retained: 1, dirtied: 1 },
        "the root is always shared, so it is always dirtied"
    );

    // Every ratio still stays within bounds even though the key is outside the retained set.
    for depth in 0..TRIE_SHAPE_PREFIX_LEVELS {
        let coverage = metrics.account_prefixes[depth];
        assert!(coverage.dirtied <= coverage.retained, "depth {depth}: {coverage:?}");
    }
}

#[test]
fn the_change_set_reads_wipes_and_storage_owners_out_of_the_post_state() {
    let mut post_state = HashedPostState::default();
    let owner = keccak256(address(0));
    let wiped_owner = keccak256(address(1));
    post_state.storages.insert(owner, {
        let mut storage = HashedStorage::new(false);
        storage.storage.insert(keccak256(slot(0)), U256::from(7));
        storage
    });
    post_state.storages.insert(wiped_owner, HashedStorage::new(true));

    let changed = TrieChangeSet::from_hashed_post_state(&post_state);

    assert!(changed.accounts.is_empty(), "no account entry appears in the post state");
    assert_eq!(changed.wiped_storage.len(), 1);
    assert!(changed.wiped_storage.contains(&wiped_owner));
    assert_eq!(changed.storage.get(&owner).map(HashSet::len), Some(1));

    // Both owners have their account leaf rewritten even though neither is in the account map.
    let dirtied = changed.dirtied_accounts();
    assert_eq!(dirtied.len(), 2);
    assert!(dirtied.contains(&owner) && dirtied.contains(&wiped_owner));
}

#[test]
fn a_cold_cache_reports_zero_rather_than_dividing_by_it() {
    let metrics = PartialTrieNodeCache::new().mutation_metrics(&changed_accounts([0]));

    assert_eq!(metrics.retained_paths(), 0);
    assert_eq!(metrics.dirtied_paths(), 0);
    assert_eq!(metrics.dirtied_path_share(), 0.0);
    assert_eq!(metrics.revealed_nodes(), 0);
    assert_eq!(metrics.deepest_account_prefix().dirtied_share(), 0.0);
}
