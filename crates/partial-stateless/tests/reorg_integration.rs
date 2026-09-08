//! Integration tests for reorg handling.
//!
//! These replay the exact cache-mutation sequence the ExEx handlers perform —
//! `ChainCommitted` (apply low→high), `ChainReorged` (roll back old newest→oldest,
//! then apply new low→high), `ChainReverted` (roll back old newest→oldest) — and
//! assert the cache reflects only the canonical chain afterwards.
//!
//! The handlers' EVM simulation is out of scope here; this exercises the cache
//! contract those handlers depend on via its public API.

use alloy_primitives::{Address, B256, U256};
use partial_stateless::{
    accessed_state::BlockAccessedState,
    network_cache::NetworkStateCache,
    policy::{AccountData, LastNBlocksPolicy},
};
use std::collections::BTreeMap;

fn make_cache(account_window: u64, storage_window: u64) -> NetworkStateCache {
    NetworkStateCache::new(
        Box::new(LastNBlocksPolicy::new(account_window)),
        Box::new(LastNBlocksPolicy::new(storage_window)),
    )
}

fn account(nonce: u64, balance: u64) -> AccountData {
    AccountData { nonce, balance: U256::from(balance), code_hash: None }
}

/// Build one block's accessed state from `(address, account)` and `(address, slot, value)` lists.
fn block_state(
    accounts: &[(Address, AccountData)],
    storage: &[(Address, B256, u64)],
) -> BlockAccessedState {
    let mut s = BlockAccessedState::default();
    for (addr, data) in accounts {
        s.accounts.insert(*addr, data.clone());
    }
    for (addr, slot, value) in storage {
        s.storage.insert((*addr, *slot), U256::from(*value));
    }
    s
}

/// Mirrors the `ChainCommitted` handler: apply blocks low→high.
fn commit(cache: &mut NetworkStateCache, blocks: &[(u64, BlockAccessedState)]) {
    for (number, accessed) in blocks {
        cache.on_block_executed(*number, accessed);
    }
}

/// Mirrors the `ChainReorged` handler: roll back `old` newest→oldest, cold-reset on
/// failure, then apply `new` oldest→newest. Returns whether rollback succeeded.
fn reorg(cache: &mut NetworkStateCache, old: &[u64], new: &[(u64, BlockAccessedState)]) -> bool {
    let mut rollback_ok = true;
    for number in old.iter().rev() {
        if cache.rollback_block(*number).is_err() {
            rollback_ok = false;
            break;
        }
    }
    if !rollback_ok {
        cache.reset();
    }
    for (number, accessed) in new {
        cache.on_block_executed(*number, accessed);
    }
    rollback_ok
}

/// Mirrors the `ChainReverted` handler: roll back `old` newest→oldest.
fn revert(cache: &mut NetworkStateCache, old: &[u64]) {
    for number in old.iter().rev() {
        if cache.rollback_block(*number).is_err() {
            cache.reset();
            break;
        }
    }
}

/// Full-content fingerprint (keys + values + freshness + height) for equivalence checks.
type Fingerprint =
    (BTreeMap<Address, (u64, U256, u64)>, BTreeMap<(Address, B256), (U256, u64)>, u64);

fn fingerprint(cache: &NetworkStateCache) -> Fingerprint {
    let accounts = cache
        .accounts()
        .iter()
        .map(|(k, e)| (*k, (e.value.nonce, e.value.balance, e.last_accessed_block)))
        .collect();
    let storage =
        cache.storage().iter().map(|(k, e)| (*k, (e.value, e.last_accessed_block))).collect();
    (accounts, storage, cache.current_block())
}

#[test]
fn test_reorg_drops_old_branch_state() {
    let mut cache = make_cache(20, 20);
    let shared = Address::repeat_byte(0x01);
    let only_old = Address::repeat_byte(0x0A);
    let only_new = Address::repeat_byte(0x0B);

    commit(
        &mut cache,
        &[
            (100, block_state(&[(shared, account(1, 100))], &[])),
            (101, block_state(&[(shared, account(2, 200))], &[])),
            (102, block_state(&[(only_old, account(1, 1))], &[])), // old-branch tip
        ],
    );
    assert!(cache.contains_account(&only_old));

    // Reorg: revert old block 102, apply new block 102' touching a different account.
    reorg(&mut cache, &[102], &[(102, block_state(&[(only_new, account(1, 1))], &[]))]);

    assert!(!cache.contains_account(&only_old), "old-branch state must be gone after reorg");
    assert!(cache.contains_account(&only_new), "new-branch state must be present");
    assert!(cache.contains_account(&shared), "shared state below the fork must remain");
    assert_eq!(cache.current_block(), 102);
}

#[test]
fn test_reorg_equals_fresh_replay_invariant() {
    // The core safety invariant: applying a reorg (roll back old + apply new) must
    // yield a cache identical to applying the new canonical chain fresh from the
    // fork point. If it doesn't, sidecar hit/miss accounting diverges between peers.
    let s100 = block_state(
        &[(Address::repeat_byte(1), account(1, 100))],
        &[(Address::repeat_byte(1), B256::repeat_byte(9), 1)],
    );
    let s101 = block_state(&[(Address::repeat_byte(2), account(1, 101))], &[]);
    let old102 = block_state(
        &[(Address::repeat_byte(0xA0), account(1, 1))],
        &[(Address::repeat_byte(0xA0), B256::repeat_byte(0xAA), 5)],
    );
    let old103 = block_state(&[(Address::repeat_byte(0xA1), account(1, 2))], &[]);
    let new102 = block_state(
        &[(Address::repeat_byte(0xB0), account(7, 7))],
        &[(Address::repeat_byte(0xB0), B256::repeat_byte(0xBB), 8)],
    );
    let new103 = block_state(&[(Address::repeat_byte(0xB1), account(9, 9))], &[]);

    // Cache A: commit shared blocks + old branch, then reorg onto the new branch.
    let mut a = make_cache(50, 50);
    commit(&mut a, &[(100, s100.clone()), (101, s101.clone())]);
    commit(&mut a, &[(102, old102), (103, old103)]);
    let ok = reorg(&mut a, &[102, 103], &[(102, new102.clone()), (103, new103.clone())]);
    assert!(ok, "rollback should succeed with full undo history");

    // Cache B: commit shared blocks, then the new branch directly.
    let mut b = make_cache(50, 50);
    commit(&mut b, &[(100, s100), (101, s101), (102, new102), (103, new103)]);

    assert_eq!(fingerprint(&a), fingerprint(&b), "reorg result must equal a fresh replay");
}

#[test]
fn test_pure_revert_returns_to_prior_state() {
    let s100 = block_state(&[(Address::repeat_byte(1), account(1, 100))], &[]);
    let s101 = block_state(&[(Address::repeat_byte(2), account(1, 101))], &[]);
    let s102 = block_state(&[(Address::repeat_byte(3), account(1, 102))], &[]);

    let mut cache = make_cache(50, 50);
    commit(&mut cache, &[(100, s100.clone()), (101, s101.clone()), (102, s102)]);

    // Revert blocks 101 and 102 with no replacement.
    revert(&mut cache, &[101, 102]);

    // Reference: only block 100 was ever applied.
    let mut reference = make_cache(50, 50);
    commit(&mut reference, &[(100, s100)]);

    assert_eq!(fingerprint(&cache), fingerprint(&reference));
}

#[test]
fn test_deep_reorg_beyond_history_cold_resets() {
    let mut cache = make_cache(50, 50);
    commit(
        &mut cache,
        &[
            (100, block_state(&[(Address::repeat_byte(1), account(1, 100))], &[])),
            (101, block_state(&[(Address::repeat_byte(2), account(1, 101))], &[])),
            (102, block_state(&[(Address::repeat_byte(3), account(1, 102))], &[])),
        ],
    );

    // Finalize through 102: undo history is pruned, so block 102 can no longer roll back.
    cache.prune_undo_below(102);

    let new_addr = Address::repeat_byte(0xC0);
    let ok = reorg(&mut cache, &[102], &[(103, block_state(&[(new_addr, account(1, 1))], &[]))]);

    assert!(!ok, "rollback must fail once undo history is pruned");
    // After cold reset + reapply, only the new block's state remains.
    assert!(cache.contains_account(&new_addr));
    assert!(!cache.contains_account(&Address::repeat_byte(1)), "cold reset must clear old state");
    assert_eq!(cache.current_block(), 103);
}

// ---- depth-D rollback planning ----------------------------------------------------------------

/// Applies `count` blocks from height 1, rooting the cache after each so every undo record
/// memoizes its parent's cache root — the condition `can_rollback_to` refuses without.
fn commit_rooted_chain(cache: &mut NetworkStateCache, count: u64) {
    let address = Address::repeat_byte(0x11);
    cache.cache_root();
    for number in 1..=count {
        let state = block_state(&[(address, account(number, number * 10))], &[]);
        cache.on_block_executed(number, &state);
        cache.cache_root();
    }
}

#[test]
fn a_plan_covers_the_whole_run_and_lands_where_it_was_asked_to() {
    let mut cache = make_cache(60, 30);
    commit_rooted_chain(&mut cache, 5);

    let plan = cache.can_rollback_to(2).expect("blocks 3, 4 and 5 are all in the log");
    assert_eq!(plan.depth(), 3);
    assert_eq!(plan.previous_block(), 2);

    // Nothing moved: a plan is a read.
    assert_eq!(cache.current_block(), 5);

    let landing_root = plan.previous_cache_root();
    cache.rollback(&plan).expect("the plan is fresh");
    assert_eq!(cache.current_block(), 2);
    assert_eq!(cache.cache_root(), landing_root, "the plan named the root the undo installs");
}

#[test]
fn a_plan_refuses_rather_than_landing_somewhere_it_was_not_asked_to() {
    let mut cache = make_cache(60, 30);
    commit_rooted_chain(&mut cache, 5);

    // Height 0 is reachable, not an error: the cache starts there, so block 1's record steps back
    // to it like any other. What is unreachable is a height whose records finality already pruned.
    assert!(cache.can_rollback_to(0).is_ok(), "the pre-chain height is a real landing");
    cache.prune_undo_below(3);

    // Now the log reaches back to 3 and no further. "Too short", which a longer log would fix —
    // and the error says how far back it does reach, so a caller can ask for something it can get.
    match cache.can_rollback_to(1) {
        Err(partial_stateless::network_cache::CacheError::RollbackExhausted {
            requested,
            oldest,
        }) => {
            assert_eq!(requested, 1);
            assert_eq!(oldest, Some(3));
        }
        other => panic!("expected RollbackExhausted, got {other:?}"),
    }

    assert!(cache.can_rollback_to(5).is_err(), "the cache is already at 5");
    assert!(cache.can_rollback_to(9).is_err(), "9 is above the head");

    // Every refusal above left the cache where it was.
    assert_eq!(cache.current_block(), 5);
}

#[test]
fn a_pruned_middle_record_is_caught_before_anything_moves() {
    let mut cache = make_cache(60, 30);
    commit_rooted_chain(&mut cache, 5);

    // Exactly what finality pruning does, and the case arithmetic on the endpoints would accept:
    // `5 - 2 == 3` still holds while the record that links 3 to 2 is gone.
    cache.prune_undo_below(3);

    assert!(cache.can_rollback_to(2).is_err(), "the run is no longer contiguous");
    assert_eq!(cache.current_block(), 5);

    // The part of the log that survived still plans.
    let plan = cache.can_rollback_to(4).expect("blocks 5 is still recorded");
    assert_eq!(plan.depth(), 1);
    assert_eq!(plan.previous_block(), 4);
}

#[test]
fn a_depth_one_plan_says_what_undo_preview_says() {
    let mut cache = make_cache(60, 30);
    commit_rooted_chain(&mut cache, 4);

    let preview = cache.undo_preview().expect("a record exists");
    let plan = cache.can_rollback_to(3).expect("the newest block is always rollbackable");

    // The generalisation has to agree with what it generalises, or a depth-1 recovery routed
    // through the new path would land differently than the one it replaced.
    assert_eq!(plan.depth(), 1);
    assert_eq!(plan.previous_block(), preview.previous_block);
    assert_eq!(Some(plan.previous_cache_root()), preview.previous_cache_root);
}

#[test]
fn a_stale_plan_is_refused_before_a_single_block_is_given_back() {
    let mut cache = make_cache(60, 30);
    commit_rooted_chain(&mut cache, 5);
    let plan = cache.can_rollback_to(2).expect("blocks 3, 4 and 5 are in the log");

    // A `RollbackPlan` is an owned value with no borrow on the cache, so nothing in the type
    // system stops this. What must not happen is a partial rollback: the plan still names three
    // reachable blocks, and a loop that checked each record as it consumed it would give back 6
    // and 5 before discovering that 4 is not what the plan said.
    let address = Address::repeat_byte(0x11);
    cache.on_block_executed(6, &block_state(&[(address, account(6, 60))], &[]));
    cache.cache_root();
    assert_eq!(cache.current_block(), 6);

    let err = cache.rollback(&plan).expect_err("the plan describes a log that no longer exists");
    assert!(
        matches!(err, partial_stateless::network_cache::CacheError::RollbackPlanStale { .. }),
        "got {err:?}"
    );
    assert_eq!(cache.current_block(), 6, "not one block was given back");

    // And a plan taken now still works, so the refusal is about staleness and not about the log.
    let fresh = cache.can_rollback_to(2).expect("the log still reaches back to 2");
    cache.rollback(&fresh).expect("a fresh plan applies");
    assert_eq!(cache.current_block(), 2);
}
