//! Verifies that the sparse-trie cache reports how much of itself a block dirties.
//!
//! These ratios decide whether the per-block deep clone of the trie is worth replacing with
//! copy-on-write or a journal, so a counter that flatters either side would send that design the
//! wrong way.

use alloy_primitives::{keccak256, Address, B256, U256};
use partial_stateless::{
    policy::{AccountData, LastNBlocksPolicy},
    BlockAccessedState, NetworkStateCache, PartialTrieNodeCache, TRIE_SHAPE_PREFIX_LEVELS,
};

const RETAINED_ACCOUNTS: usize = 32;
const SLOTS_PER_ACCOUNT: usize = 8;

fn address(index: usize) -> Address {
    Address::repeat_byte(index as u8 + 1)
}

fn slot(index: usize) -> B256 {
    B256::repeat_byte(index as u8 + 0x80)
}

/// A cache retaining every account and, for the first two accounts, several storage slots.
fn warm_cache() -> (NetworkStateCache, PartialTrieNodeCache) {
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
    (values, trie)
}

#[test]
fn one_changed_account_dirties_a_small_share_of_the_retained_paths() {
    let (_, trie) = warm_cache();

    let metrics = trie.mutation_metrics([keccak256(address(0))], []);

    assert_eq!(metrics.retained_account_paths, RETAINED_ACCOUNTS);
    assert_eq!(metrics.dirtied_account_paths, 1);
    assert_eq!(metrics.dirtied_storage_paths, 0);
    assert_eq!(metrics.dirtied_storage_tries, 0);

    let share = metrics.dirtied_path_share();
    assert!(share > 0.0 && share < 0.1, "one path out of many must be a small share: {share}");
}

#[test]
fn prefix_coverage_saturates_at_the_root_and_discriminates_with_depth() {
    let (_, trie) = warm_cache();

    let metrics = trie.mutation_metrics([keccak256(address(0))], []);

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
    let (_, trie) = warm_cache();

    let all = (0..RETAINED_ACCOUNTS).map(|index| keccak256(address(index)));
    let metrics = trie.mutation_metrics(all, []);

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
    let (_, trie) = warm_cache();

    let changed = [
        (keccak256(address(0)), keccak256(slot(0))),
        (keccak256(address(0)), keccak256(slot(1))),
        (keccak256(address(1)), keccak256(slot(0))),
    ];
    let metrics = trie.mutation_metrics([], changed);

    assert_eq!(metrics.dirtied_storage_paths, 3);
    assert_eq!(metrics.dirtied_storage_tries, 2);
    assert_eq!(metrics.retained_storage_paths, 2 * SLOTS_PER_ACCOUNT);
    assert_eq!(metrics.dirtied_account_paths, 0);

    // Sorted most-dirtied first, so the head of the list is the trie a sharing scheme cares about.
    let worst = &metrics.per_storage_trie[0];
    assert_eq!(worst.hashed_address, keccak256(address(0)));
    assert_eq!(worst.dirtied_paths, 2);
    assert_eq!(worst.retained_paths, SLOTS_PER_ACCOUNT);
    assert_eq!(metrics.per_storage_trie[1].dirtied_paths, 1);
}

#[test]
fn keys_outside_the_cache_cannot_dirty_it() {
    let (_, trie) = warm_cache();

    // A block routinely changes state this cache never retained; those changes dirty no node the
    // clone copied, so counting them would overstate the headroom for copy-on-write.
    let stranger = keccak256(Address::repeat_byte(0xfe));
    let metrics = trie.mutation_metrics(
        [stranger, keccak256(address(0))],
        [
            (stranger, keccak256(slot(0))),
            (keccak256(address(0)), keccak256(B256::repeat_byte(0xfd))),
        ],
    );

    assert_eq!(metrics.dirtied_account_paths, 1, "only the retained account counts");
    assert_eq!(metrics.dirtied_storage_paths, 0, "neither storage key is retained");
    assert_eq!(metrics.dirtied_storage_tries, 0);
}

#[test]
fn a_cold_cache_reports_zero_rather_than_dividing_by_it() {
    let metrics = PartialTrieNodeCache::new().mutation_metrics([keccak256(address(0))], []);

    assert_eq!(metrics.retained_paths(), 0);
    assert_eq!(metrics.dirtied_paths(), 0);
    assert_eq!(metrics.dirtied_path_share(), 0.0);
    assert_eq!(metrics.revealed_nodes(), 0);
    assert_eq!(metrics.deepest_account_prefix().dirtied_share(), 0.0);
}
