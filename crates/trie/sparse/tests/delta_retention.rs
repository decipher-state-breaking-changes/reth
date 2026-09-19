//! The narrowed retention walk: walking only toward the nodes a block wrote and the paths that
//! moved in or out of the retained set must leave the trie exactly as the full walk does.
//!
//! The full walk is the specification. Every test here runs a block on a recording copy of a
//! trie the previous block's retention left, then retains one copy with the full walk and one
//! with the narrowed walk, and requires the two tries to be equal — nodes, values, masks and all.

use alloy_primitives::{keccak256, map::B256Map, B256, U256};
use alloy_rlp::encode_fixed_size;
use rand::{rngs::StdRng, seq::IteratorRandom, Rng, SeedableRng};
use reth_trie::test_utils::TrieTestHarness;
use reth_trie_common::{Nibbles, ProofV2Target};
use reth_trie_sparse::{ExactSparseTrie, LeafUpdate, RetentionOptions, SparseTrie};
use std::collections::{BTreeMap, BTreeSet};

fn key(i: usize) -> B256 {
    keccak256(B256::from(U256::from(i)))
}

fn nibbles(key: B256) -> Nibbles {
    Nibbles::unpack(key)
}

/// A trie over `harness` with `revealed` of its keys revealed and the root computed.
fn trie_revealing(harness: &TrieTestHarness, revealed: &[B256]) -> ExactSparseTrie {
    let root = harness.root_node();
    let mut trie = ExactSparseTrie::default();
    trie.set_root(root.node, root.masks, false).unwrap();
    reveal(harness, &mut trie, revealed);
    trie.root();
    trie
}

/// Reveals the proofs of `keys`, present or absent, the way a block's witness does.
fn reveal(harness: &TrieTestHarness, trie: &mut ExactSparseTrie, keys: &[B256]) {
    if keys.is_empty() {
        return
    }
    let mut targets: Vec<_> = keys.iter().map(|k| ProofV2Target::new(*k)).collect();
    let (mut nodes, _) = harness.proof_v2(&mut targets);
    trie.reveal_nodes(&mut nodes).unwrap();
}

/// Applies `changes` (zero = delete) the way a block does, without computing the root.
fn update(harness: &TrieTestHarness, trie: &mut ExactSparseTrie, changes: &[(B256, U256)]) {
    let mut updates: B256Map<LeafUpdate> = changes
        .iter()
        .map(|(slot, value)| {
            let rlp = if value.is_zero() { Vec::new() } else { encode_fixed_size(value).to_vec() };
            (*slot, LeafUpdate::Changed(rlp))
        })
        .collect();
    loop {
        let mut targets = Vec::new();
        trie.update_leaves(&mut updates, |key, min_len| {
            targets.push(ProofV2Target::new(key).with_min_len(min_len));
        })
        .unwrap();
        if targets.is_empty() {
            break;
        }
        let (mut nodes, _) = harness.proof_v2(&mut targets);
        trie.reveal_nodes(&mut nodes).unwrap();
    }
}

fn sorted_paths(keys: &BTreeSet<B256>) -> Vec<Nibbles> {
    let mut paths: Vec<_> = keys.iter().copied().map(nibbles).collect();
    paths.sort_unstable();
    paths
}

fn moved_paths(before: &BTreeSet<B256>, after: &BTreeSet<B256>) -> Vec<Nibbles> {
    before.symmetric_difference(after).copied().map(nibbles).collect()
}

/// Retains `working` both ways and requires the same trie, returning the narrowed copy.
///
/// Also reports the two walks' visit counts, so a test can show the narrowed walk skipped work
/// rather than passing by visiting everything.
fn retain_both_ways(
    working: ExactSparseTrie,
    retained: &BTreeSet<B256>,
    moved: &[Nibbles],
    at: &str,
) -> (ExactSparseTrie, u64, u64) {
    let paths = sorted_paths(retained);
    let mut full = working.clone();
    let full_outcome =
        full.retain_witness_paths_with_options(&paths, RetentionOptions::sorted_input());
    let mut near = working;
    let near_outcome = near
        .retain_witness_paths_since_undo_began(&paths, moved, RetentionOptions::sorted_input())
        .unwrap_or_else(|| panic!("{at}: the record could not vouch for the block"));

    assert_eq!(near_outcome.metrics.delta_calls, 1, "{at}");
    assert_eq!(near_outcome.metrics.full_range_calls, 0, "{at}");
    assert_eq!(near_outcome.pruned, full_outcome.pruned, "{at}: different prune roots");
    assert_eq!(near.is_retention_clean(), full.is_retention_clean(), "{at}");
    assert!(near == full, "{at}: the narrowed walk left a different trie");
    assert_eq!(near.root(), full.root(), "{at}");
    (near, near_outcome.metrics.nodes_visited, full_outcome.metrics.nodes_visited)
}

/// Blocks of reveals, overwrites, deletes and inserts against a retained set that gains and loses
/// present and absent keys every block, checked against the full walk after each one.
#[test]
fn the_narrowed_walk_equals_the_full_walk_block_after_block() {
    let (visited_near, visited_full) = run_blocks(0x5eed_de17a, 3_000, 24, 12);
    assert!(
        visited_near * 2 < visited_full,
        "the narrowed walk visited {visited_near} nodes against the full walk's {visited_full}"
    );
}

/// Small tries under heavy deletion: branches collapse, lower subtries empty out and are cleared
/// into the record whole, and the record still names every path the walk has to reach.
#[test]
fn the_narrowed_walk_equals_the_full_walk_while_the_trie_collapses() {
    for seed in 1..=6 {
        run_blocks(seed, 400, 12, 25);
    }
}

/// Runs `blocks` blocks from a trie of `size` keys and returns the two walks' total visits.
fn run_blocks(seed: u64, size: usize, blocks: usize, deletes: usize) -> (u64, u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut entries: BTreeMap<B256, U256> =
        (0..size).map(|i| (key(i), U256::from(i + 1))).collect();
    let mut harness = TrieTestHarness::new(entries.clone());
    let mut next_key = size;

    // A witness-shaped start: half the keys revealed, a quarter of those retained, and a handful
    // of absent keys retained for their exclusion proofs.
    let revealed: Vec<B256> = entries.keys().copied().choose_multiple(&mut rng, size / 2);
    let mut retained: BTreeSet<B256> =
        revealed.iter().copied().choose_multiple(&mut rng, size / 7).into_iter().collect();
    retained.extend((10_000..10_020).map(key));
    let mut trie = trie_revealing(&harness, &revealed);
    trie.retain_witness_paths_with_options(
        &sorted_paths(&retained),
        RetentionOptions::sorted_input(),
    );
    assert!(trie.is_retention_clean(), "a retention after the root leaves nothing dirty");

    let (mut visited_near, mut visited_full) = (0u64, 0u64);
    for block in 0..blocks {
        let at = format!("seed {seed} block {block}");
        let mut working = trie.clone();
        working.begin_undo();

        // The witness: some retained keys, some unrelated ones, some absent.
        let mut accessed: Vec<B256> = entries.keys().copied().choose_multiple(&mut rng, 60);
        accessed.extend(retained.iter().copied().choose_multiple(&mut rng, 20));
        accessed.extend((0..5).map(|_| key(20_000 + rng.random_range(0..5_000))));
        reveal(&harness, &mut working, &accessed);

        // The writes: overwrites and deletes of present keys, inserts of new ones.
        let mut changes: BTreeMap<B256, U256> = BTreeMap::new();
        for slot in entries.keys().copied().choose_multiple(&mut rng, 40) {
            changes.insert(slot, U256::from(rng.random_range(1..u64::MAX)));
        }
        for slot in entries.keys().copied().choose_multiple(&mut rng, deletes) {
            changes.insert(slot, U256::ZERO);
        }
        for _ in 0..10 {
            changes.insert(key(next_key), U256::from(next_key));
            next_key += 1;
        }
        let changes_list: Vec<_> = changes.iter().map(|(k, v)| (*k, *v)).collect();
        update(&harness, &mut working, &changes_list);
        working.root();

        // The retained set moves: some leave, and written, accessed and absent keys enter.
        let before = retained.clone();
        for slot in before.iter().copied().choose_multiple(&mut rng, 30) {
            retained.remove(&slot);
        }
        retained.extend(changes.keys().copied().choose_multiple(&mut rng, 25));
        retained.extend(accessed.iter().copied().choose_multiple(&mut rng, 20));
        retained.extend((0..3).map(|_| key(30_000 + rng.random_range(0..5_000))));
        let moved = moved_paths(&before, &retained);

        let (near, near_visits, full_visits) = retain_both_ways(working, &retained, &moved, &at);
        visited_near += near_visits;
        visited_full += full_visits;

        // Commit: the narrowed copy becomes the next block's parent, record ended.
        trie = near;
        trie.take_undo();
        harness.apply_changeset(changes.clone());
        for (slot, value) in changes {
            if value.is_zero() {
                entries.remove(&slot);
            } else {
                entries.insert(slot, value);
            }
        }
        assert_eq!(trie.root(), harness.original_root(), "{at}");
    }
    (visited_near, visited_full)
}

/// Keys that share long prefixes produce extension nodes, and a retained path that diverges inside
/// an extension's key needs the extension and its child as the exclusion witness. The narrowed
/// walk has to reach that divergence through the candidate alone.
#[test]
fn a_retained_path_diverging_inside_an_extension_is_handled_like_the_full_walk() {
    let key = |prefix: u8, tail: u8| {
        let mut key = B256::repeat_byte(0x44);
        key.0[0] = prefix;
        key.0[31] = tail;
        key
    };
    // Several groups sharing 62-nibble prefixes: each group sits below an extension.
    let mut entries = BTreeMap::new();
    for prefix in [0x10u8, 0x11, 0x20, 0x21, 0x30] {
        for tail in [0x01u8, 0x02, 0x13] {
            entries.insert(key(prefix, tail), U256::MAX - U256::from(tail));
        }
    }
    let harness = TrieTestHarness::new(entries.clone());
    let all: Vec<B256> = entries.keys().copied().collect();
    let mut retained: BTreeSet<B256> = [key(0x10, 0x01), key(0x30, 0x13)].into_iter().collect();
    let mut trie = trie_revealing(&harness, &all);
    trie.retain_witness_paths_with_options(
        &sorted_paths(&retained),
        RetentionOptions::sorted_input(),
    );

    // Absent keys that diverge inside the shared run of each group, revealed as the witness would.
    let mut divergent = |prefix: u8| {
        let mut absent = key(prefix, 0x01);
        absent.0[10] = 0x99;
        absent
    };
    let absent: Vec<B256> = [0x11u8, 0x20, 0x21].into_iter().map(&mut divergent).collect();
    let mut working = trie.clone();
    working.begin_undo();
    reveal(&harness, &mut working, &absent);
    working.root();
    let before = retained.clone();
    retained.extend(absent.iter().copied());
    retained.remove(&key(0x10, 0x01));
    let moved = moved_paths(&before, &retained);
    retain_both_ways(working, &retained, &moved, "an extension divergence");
}

/// Deletes that collapse branches into extensions and leaves, including a collapse that reveals a
/// blinded sibling, with the retained set held still: only the written nodes are candidates.
#[test]
fn collapses_under_an_unchanged_retained_set_are_found_through_the_record() {
    let mut entries: BTreeMap<B256, U256> = (0..400).map(|i| (key(i), U256::from(i + 1))).collect();
    let harness = TrieTestHarness::new(entries.clone());
    let revealed: Vec<B256> = (0..400).map(key).collect();
    let retained: BTreeSet<B256> = (0..400).step_by(7).map(key).collect();
    let mut trie = trie_revealing(&harness, &revealed);
    trie.retain_witness_paths_with_options(
        &sorted_paths(&retained),
        RetentionOptions::sorted_input(),
    );

    let mut working = trie.clone();
    working.begin_undo();
    let deletes: Vec<(B256, U256)> =
        (0..400).filter(|i| i % 3 == 1).map(|i| (key(i), U256::ZERO)).collect();
    update(&harness, &mut working, &deletes);
    working.root();
    for (slot, _) in &deletes {
        entries.remove(slot);
    }
    let (mut near, _, _) = retain_both_ways(working, &retained, &[], "collapses");
    assert_eq!(near.root(), TrieTestHarness::new(entries).original_root());
}

/// A retention that runs before the root leaves dirty nodes it could not blind. Nothing about a
/// later block can vouch for what sits below them, so the trie refuses the narrowed walk.
#[test]
fn a_dirty_residual_refuses_the_narrowed_walk_until_a_full_walk_clears_it() {
    let harness = TrieTestHarness::new((0..600).map(|i| (key(i), U256::from(i + 1))).collect());
    let revealed: Vec<B256> = (0..600).map(key).collect();
    let retained: BTreeSet<B256> = (0..600).step_by(5).map(key).collect();
    let paths = sorted_paths(&retained);
    let mut trie = trie_revealing(&harness, &revealed);

    // Inserted, not hashed, then retained: the new leaves and the branches they split are dirty,
    // and none of them is retained.
    let changes: Vec<(B256, U256)> = (600..700).map(|i| (key(i), U256::from(9))).collect();
    update(&harness, &mut trie, &changes);
    let outcome = trie.retain_witness_paths_with_options(&paths, RetentionOptions::sorted_input());
    assert!(outcome.metrics.unprunable_dirty > 0, "the fixture must leave a dirty residual");
    assert!(!trie.is_retention_clean());

    trie.root();
    trie.begin_undo();
    assert!(
        trie.retain_witness_paths_since_undo_began(&paths, &[], RetentionOptions::sorted_input())
            .is_none(),
        "a dirty residual is no base for a narrowed walk"
    );
    trie.retain_witness_paths_with_options(&paths, RetentionOptions::sorted_input());
    assert!(trie.is_retention_clean(), "the full walk blinds what the root has now hashed");
    assert!(trie
        .retain_witness_paths_since_undo_began(&paths, &[], RetentionOptions::sorted_input())
        .is_some());
}

/// Every way the trie cannot vouch for its own past refuses the narrowed walk: no record to name
/// the writes, a wipe that replaced the content the last retention left, and a trie put back by an
/// undo, whose last retention the frame does not describe.
#[test]
fn the_narrowed_walk_is_refused_whenever_the_record_cannot_vouch() {
    let harness = TrieTestHarness::new((0..300).map(|i| (key(i), U256::from(i + 1))).collect());
    let revealed: Vec<B256> = (0..300).map(key).collect();
    let paths = sorted_paths(&(0..300).step_by(4).map(key).collect());
    let options = RetentionOptions::sorted_input();
    let mut trie = trie_revealing(&harness, &revealed);
    trie.retain_witness_paths_with_options(&paths, options);
    assert!(trie.is_retention_clean());

    let mut unrecorded = trie.clone();
    assert!(unrecorded.retain_witness_paths_since_undo_began(&paths, &[], options).is_none());

    let mut wiped = trie.clone();
    wiped.begin_undo();
    wiped.wipe();
    assert!(wiped.retain_witness_paths_since_undo_began(&paths, &[], options).is_none());

    let mut undone = trie.clone();
    undone.begin_undo();
    update(&harness, &mut undone, &[(key(1), U256::from(77))]);
    undone.root();
    let frame = undone.take_undo().unwrap();
    undone.undo(frame);
    assert!(undone == trie);
    assert!(!undone.is_retention_clean(), "an undo restores content, not retention history");
    undone.begin_undo();
    assert!(undone.retain_witness_paths_since_undo_began(&paths, &[], options).is_none());

    // And the flag rides along with a clone, so a working copy can use its parent's.
    let mut working = trie.clone();
    working.begin_undo();
    assert!(working.retain_witness_paths_since_undo_began(&paths, &[], options).is_some());
}

/// With no write and no moved path there is nothing to walk toward: the narrowed walk visits
/// nothing and changes nothing, which is what the full walk also concludes by visiting everything.
#[test]
fn an_empty_block_visits_nothing() {
    let harness = TrieTestHarness::new((0..500).map(|i| (key(i), U256::from(i + 1))).collect());
    let revealed: Vec<B256> = (0..500).map(key).collect();
    let retained: BTreeSet<B256> = (0..500).step_by(3).map(key).collect();
    let mut trie = trie_revealing(&harness, &revealed);
    trie.retain_witness_paths_with_options(
        &sorted_paths(&retained),
        RetentionOptions::sorted_input(),
    );
    let before = trie.clone();
    let mut working = trie.clone();
    working.begin_undo();
    working.root();
    let (near, near_visits, full_visits) = retain_both_ways(working, &retained, &[], "empty");
    assert!(near == before);
    assert_eq!(near_visits, 0);
    assert!(full_visits > 0);
}
