//! The undo record on `ExactSparseTrie`: a block of changes, recorded while it happens, is
//! reversed exactly, and the record is the size of the block rather than the trie.

use alloy_primitives::{keccak256, map::B256Map, B256, U256};
use alloy_rlp::encode_fixed_size;
use reth_trie::test_utils::TrieTestHarness;
use reth_trie_common::{Nibbles, ProofV2Target};
use reth_trie_sparse::{ExactSparseTrie, LeafUpdate, SparseTrie};
use std::collections::BTreeMap;

fn key(i: usize) -> B256 {
    keccak256(B256::from(U256::from(i)))
}

fn storage(n: usize) -> BTreeMap<B256, U256> {
    (0..n).map(|i| (key(i), U256::from(i + 1))).collect()
}

/// A trie over `harness`'s storage with `revealed` of its keys revealed and the root computed.
fn trie_revealing(harness: &TrieTestHarness, revealed: &[B256]) -> ExactSparseTrie {
    let root = harness.root_node();
    let mut trie = ExactSparseTrie::default();
    trie.set_root(root.node, root.masks, false).unwrap();
    if !revealed.is_empty() {
        let mut targets: Vec<_> = revealed.iter().map(|k| ProofV2Target::new(*k)).collect();
        let (mut nodes, _) = harness.proof_v2(&mut targets);
        trie.reveal_nodes(&mut nodes).unwrap();
    }
    trie.root();
    trie
}

/// Applies `changes` (zero = delete) the way a block does: update, reveal what the update asks
/// for, repeat until nothing more is asked, then compute the root.
fn apply(harness: &TrieTestHarness, trie: &mut ExactSparseTrie, changes: &[(B256, U256)]) -> B256 {
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
    trie.root()
}

fn nibbles(key: B256) -> Nibbles {
    Nibbles::unpack(key)
}

#[test]
fn a_block_of_writes_and_a_retention_pass_undo_to_the_trie_they_began_on() {
    let harness = TrieTestHarness::new(storage(2_000));
    let all: Vec<B256> = (0..2_000).map(key).collect();
    let mut trie = trie_revealing(&harness, &all);
    let before = trie.clone();
    let root_before = trie.root();

    trie.begin_undo();
    // Overwrite, delete, and insert keys the trie has never seen: 40 of 2,000 leaves.
    let mut changes: Vec<(B256, U256)> = (0..20).map(|i| (key(i), U256::from(1_000 + i))).collect();
    changes.extend((20..30).map(|i| (key(i), U256::ZERO)));
    changes.extend((2_000..2_010).map(|i| (key(i), U256::from(i))));
    let root_after = apply(&harness, &mut trie, &changes);
    assert_ne!(root_after, root_before);
    // Retention, as the cache runs it after a commit: drop one leaf in twenty.
    let retained: Vec<Nibbles> =
        (30..2_000).filter(|i| i % 20 != 0).map(|i| nibbles(key(i))).collect();
    let pruned = trie.prune(&retained);
    assert!(pruned > 0, "the retention pass converted nodes to hash stubs");
    assert_ne!(trie, before);

    let frame = trie.take_undo().expect("a record was being kept");
    assert!(!trie.is_recording_undo());
    let counts = frame.counts();
    assert!(counts.nodes > 0 && counts.values > 0, "{counts:?}");
    let bytes = frame.allocated_bytes();
    eprintln!(
        "record {counts:?}, {bytes} bytes estimated, trie {} estimated",
        before.memory_size()
    );
    assert!(
        bytes < before.memory_size() / 4,
        "the record ({bytes} bytes) is sized by the block, not the trie ({} bytes); both figures \
         are the same kind of estimate",
        before.memory_size()
    );

    trie.undo(frame);
    assert_eq!(trie, before);
    assert_eq!(trie.root(), root_before);
}

#[test]
fn revealing_lower_subtries_the_trie_did_not_have_undoes_to_blind() {
    let harness = TrieTestHarness::new(storage(300));
    // Three keys revealed: most of the 256 lower subtries are blind.
    let mut trie = trie_revealing(&harness, &[key(0), key(1), key(2)]);
    let before = trie.clone();

    trie.begin_undo();
    let changes: Vec<(B256, U256)> = (3..80).map(|i| (key(i), U256::from(7))).collect();
    apply(&harness, &mut trie, &changes);
    assert!(trie.memory_size() > before.memory_size(), "the block revealed subtries");

    let frame = trie.take_undo().unwrap();
    assert!(frame.counts().lower_transitions > 0, "{:?}", frame.counts());
    trie.undo(frame);
    assert_eq!(trie, before);
}

#[test]
fn deleting_every_leaf_of_a_subtrie_blinds_it_and_undoes_to_revealed() {
    let harness = TrieTestHarness::new(storage(40));
    let all: Vec<B256> = (0..40).map(key).collect();
    let mut trie = trie_revealing(&harness, &all);
    let before = trie.clone();

    trie.begin_undo();
    // Delete all but two leaves: subtries holding a single revealed leaf become empty.
    let changes: Vec<(B256, U256)> = (2..40).map(|i| (key(i), U256::ZERO)).collect();
    apply(&harness, &mut trie, &changes);

    let frame = trie.take_undo().unwrap();
    trie.undo(frame);
    assert_eq!(trie, before);
}

#[test]
fn a_wipe_undoes_without_having_copied_anything() {
    let harness = TrieTestHarness::new(storage(100));
    let all: Vec<B256> = (0..100).map(key).collect();
    let mut trie = trie_revealing(&harness, &all);
    let before = trie.clone();

    trie.begin_undo();
    trie.wipe();
    assert_ne!(trie, before);
    let frame = trie.take_undo().unwrap();
    let counts = frame.counts();
    assert!(counts.whole_maps > 0, "{counts:?}");
    assert_eq!(counts.nodes, 0, "a wipe moves the maps, it does not record them entry by entry");
    trie.undo(frame);
    assert_eq!(trie, before);
}

#[test]
fn a_clone_taken_while_recording_records_only_its_own_block() {
    let harness = TrieTestHarness::new(storage(120));
    let all: Vec<B256> = (0..120).map(key).collect();
    let mut live = trie_revealing(&harness, &all);
    let generation_0 = live.clone();

    // Block 1 on the live trie, recorded.
    live.begin_undo();
    apply(&harness, &mut live, &[(key(0), U256::from(11)), (key(1), U256::ZERO)]);
    let generation_1 = live.clone();

    // Block 2 on a copy — the working copy the consumer mutates — recorded from the copy.
    let mut working = live.clone();
    assert!(working.is_recording_undo());
    apply(&harness, &mut working, &[(key(2), U256::from(22)), (key(3), U256::ZERO)]);

    // The parent's record still describes block 1 only; the copy's describes block 2 only.
    let frame_1 = live.take_undo().unwrap();
    let frame_2 = working.take_undo().unwrap();
    working.undo(frame_2);
    assert_eq!(working, generation_1);
    live.undo(frame_1);
    assert_eq!(live, generation_0);
}

#[test]
fn a_trie_that_is_not_recording_has_no_record_to_take() {
    let harness = TrieTestHarness::new(storage(10));
    let mut trie = trie_revealing(&harness, &[key(0)]);
    apply(&harness, &mut trie, &[(key(0), U256::from(5))]);
    assert!(trie.take_undo().is_none());
}

#[test]
fn an_empty_block_leaves_an_empty_record() {
    let harness = TrieTestHarness::new(storage(10));
    let mut trie = trie_revealing(&harness, &[key(0)]);
    trie.begin_undo();
    trie.root();
    let frame = trie.take_undo().unwrap();
    assert!(frame.is_empty(), "{:?}", frame.counts());
}

#[test]
fn an_upper_prune_that_blinds_whole_subtries_undoes_to_revealed_and_the_trie_keeps_working() {
    let harness = TrieTestHarness::new(storage(2_000));
    let all: Vec<B256> = (0..2_000).map(key).collect();
    let mut trie = trie_revealing(&harness, &all);
    let before = trie.clone();

    trie.begin_undo();
    // Retaining nothing makes prune roots in the upper trie, above every lower subtrie, so
    // each lower subtrie is blinded whole rather than pruned entry by entry.
    let pruned = trie.prune(&[]);
    assert!(pruned > 0);
    assert!(trie.memory_size() < before.memory_size() / 2, "the subtries were cleared");

    let frame = trie.take_undo().unwrap();
    assert!(frame.counts().lower_transitions > 0, "{:?}", frame.counts());
    trie.undo(frame);
    assert_eq!(trie, before);

    // Usable afterwards: the leaves are readable and the next block lands the same as on the
    // clone that never saw the prune.
    assert!(trie.get_leaf_value(&nibbles(key(7))).is_some());
    let mut oracle = before;
    let changes = [(key(7), U256::from(77)), (key(8), U256::ZERO), (key(2_000), U256::from(1))];
    let root = apply(&harness, &mut trie, &changes);
    assert_eq!(root, apply(&harness, &mut oracle, &changes));
    assert_eq!(trie, oracle);
}

#[test]
fn a_block_that_changed_only_the_retained_updates_is_not_an_empty_record() {
    let harness = TrieTestHarness::new(storage(10));
    let mut trie = trie_revealing(&harness, &[key(0)]);
    let before = trie.clone();
    trie.begin_undo();
    trie.set_updates(true);
    let frame = trie.take_undo().unwrap();
    assert!(!frame.is_empty(), "the retained-updates setting changed with no map entry behind it");
    assert!(frame.counts().metadata_changed);
    trie.undo(frame);
    assert_eq!(trie, before);
}

/// A small deterministic generator, so a failure names its seed.
struct Rng(u64);

impl Rng {
    const fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    const fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    const fn chance(&mut self, percent: u64) -> bool {
        self.next() % 100 < percent
    }

    const fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())]
    }
}

/// Random blocks against the derived equality: every mix of overwrites, deletes, inserts,
/// reveals, prunes of any depth and wipes undoes to the clone taken before, and the trie then
/// takes the next block exactly as that clone does.
#[test]
fn random_blocks_undo_exactly_and_the_trie_takes_the_next_block_as_its_clone_does() {
    for seed in [0x9e37_79b9_7f4a_7c15u64, 0x2545_f491_4f6c_dd1d, 0xd1b5_4a32_d192_ed03] {
        let mut rng = Rng(seed);
        let n = 600;
        let harness = TrieTestHarness::new(storage(n));
        let revealed: Vec<B256> = (0..n).filter(|_| rng.chance(60)).map(key).collect();
        let mut trie = trie_revealing(&harness, &revealed);
        let mut present: Vec<B256> = (0..n).map(key).collect();
        let mut fresh = n;

        for round in 0..40 {
            let oracle = trie.clone();
            trie.begin_undo();

            // The recorded block: any of the operations, in any combination.
            if rng.chance(15) {
                trie.wipe();
            } else {
                if rng.chance(85) {
                    let mut changes = Vec::new();
                    for _ in 0..rng.below(12) {
                        changes.push((rng.pick(&present), U256::from(rng.next())));
                    }
                    for _ in 0..rng.below(6) {
                        changes.push((rng.pick(&present), U256::ZERO));
                    }
                    for _ in 0..rng.below(6) {
                        changes.push((key(fresh), U256::from(fresh)));
                        fresh += 1;
                    }
                    apply(&harness, &mut trie, &changes);
                }
                if rng.chance(50) {
                    trie.root();
                    let keep_percent = [0, 5, 50, 95][rng.below(4)];
                    let retained: Vec<Nibbles> = present
                        .iter()
                        .filter(|_| rng.chance(keep_percent))
                        .map(|k| nibbles(*k))
                        .collect();
                    trie.prune(&retained);
                }
            }

            let frame = trie.take_undo().unwrap();
            trie.undo(frame);
            assert_eq!(trie, oracle, "seed {seed:#x} round {round}: undo did not restore");

            // The next block lands on both, unrecorded, and they stay equal.
            let mut oracle = oracle;
            let mut changes = Vec::new();
            for _ in 0..1 + rng.below(4) {
                let slot = rng.pick(&present);
                if rng.chance(25) {
                    changes.push((slot, U256::ZERO));
                    present.retain(|k| *k != slot);
                } else {
                    changes.push((slot, U256::from(rng.next())));
                }
            }
            if rng.chance(50) {
                changes.push((key(fresh), U256::from(fresh)));
                present.push(key(fresh));
                fresh += 1;
            }
            let root = apply(&harness, &mut trie, &changes);
            assert_eq!(
                root,
                apply(&harness, &mut oracle, &changes),
                "seed {seed:#x} round {round}"
            );
            assert_eq!(trie, oracle, "seed {seed:#x} round {round}: the next block diverged");
        }
    }
}
