//! The process-wide parallelism floor on `ExactSparseTrie` changes how a trie is built, never
//! what it builds. Its own binary, because the floor is process state.

use alloy_primitives::{keccak256, map::B256Map, B256, U256};
use alloy_rlp::encode_fixed_size;
use reth_trie::test_utils::TrieTestHarness;
use reth_trie_common::ProofV2Target;
use reth_trie_sparse::{
    exact_parallelism_floor, set_exact_parallelism_floor, ExactSparseTrie, LeafUpdate,
    ParallelismThresholds, SparseTrie,
};
use std::collections::BTreeMap;

fn key(i: usize) -> B256 {
    keccak256(B256::from(U256::from(i)))
}

/// Reveals every key of `harness`, then applies `changes` the way a block does.
fn build(harness: &TrieTestHarness, keys: &[B256], changes: &[(B256, U256)]) -> ExactSparseTrie {
    let root = harness.root_node();
    let mut trie = ExactSparseTrie::default();
    trie.set_root(root.node, root.masks, false).unwrap();
    let mut targets: Vec<_> = keys.iter().map(|k| ProofV2Target::new(*k)).collect();
    let (mut nodes, _) = harness.proof_v2(&mut targets);
    trie.reveal_nodes(&mut nodes).unwrap();
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
            break
        }
        let (mut nodes, _) = harness.proof_v2(&mut targets);
        trie.reveal_nodes(&mut nodes).unwrap();
    }
    trie.root();
    trie
}

#[test]
fn a_serial_floor_builds_the_same_trie_as_the_parallel_default() {
    let storage: BTreeMap<B256, U256> = (0..3_000).map(|i| (key(i), U256::from(i + 1))).collect();
    let keys: Vec<B256> = storage.keys().copied().collect();
    let harness = TrieTestHarness::new(storage);
    let mut changes: Vec<(B256, U256)> = (0..200).map(|i| (key(i), U256::from(7 + i))).collect();
    changes.extend((200..260).map(|i| (key(i), U256::ZERO)));
    changes.extend((3_000..3_100).map(|i| (key(i), U256::from(i))));

    assert_eq!(exact_parallelism_floor(), ParallelismThresholds::default());
    let mut parallel = build(&harness, &keys, &changes);

    let serial_floor =
        ParallelismThresholds { min_revealed_nodes: usize::MAX, min_updated_nodes: usize::MAX };
    set_exact_parallelism_floor(serial_floor);
    assert_eq!(exact_parallelism_floor(), serial_floor);
    let mut serial = build(&harness, &keys, &changes);
    set_exact_parallelism_floor(ParallelismThresholds::default());

    assert_eq!(serial.root(), parallel.root());
    assert!(serial == parallel, "the floor is not part of any trie's content");
}
