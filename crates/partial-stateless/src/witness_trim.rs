//! Measures how much of a built transition witness the receiving validator already holds.
//!
//! Every proof in the flat witness is generated from the state root down, so the nodes near the
//! top of the account trie ride along with every miss path even though the receiver's trie cache
//! keeps them revealed and authenticated between blocks. This module sizes that redundancy: it
//! reconstructs the hypothetical witness a builder could have sent if it had trimmed every node
//! the receiver provably holds, and reports the difference in encoded bytes.
//!
//! The measurement is a set difference over whole flat node maps, not a per-target sum. Summing
//! per-target redundancy would count a shared upper node once per path through it, while the flat
//! witness — hash-deduplicated — only ever carries it once. Building both maps and subtracting is
//! what makes the reported bytes the bytes a trimmed wire format would actually save.
//!
//! A node is trimmable only when *every* occurrence of it is covered: the flat format stores one
//! copy of byte-identical nodes even when the account trie and a storage trie (or two storage
//! tries) each need one, and a copy the receiver holds under one trie proves nothing about a path
//! in another. Witness entries shorter than 32 bytes are counted separately: they are embedded in
//! their parents rather than hash-addressed, so they can neither anchor a fragment nor be trimmed
//! on their own.

use crate::trie_cache::PartialTrieNodeCache;
use alloy_primitives::{
    keccak256,
    map::{B256Map, HashSet},
    Bytes, B256,
};
use alloy_rlp::Encodable;
use reth_trie_common::{BranchNodeRef, DecodedMultiProofV2, ProofTrieNodeV2, TrieNodeV2};

/// Depth buckets the per-depth byte counts use: nibble path lengths zero through seven, with
/// everything deeper folded into the last bucket.
pub const WITNESS_TRIM_DEPTH_LEVELS: usize = 9;

/// How much of one block's flat witness the receiver's trie cache already reveals.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct WitnessTrimStats {
    /// Nodes in the flat witness.
    pub witness_nodes: usize,
    /// Bytes those nodes occupy.
    pub witness_node_bytes: usize,
    /// Witness nodes whose every occurrence the receiver's trie cache reveals.
    pub trimmable_nodes: usize,
    /// Bytes those nodes occupy — the wire saving a receiver-aware witness would realize.
    pub trimmable_bytes: usize,
    /// Of the trimmable nodes, those with an account-trie occurrence.
    pub trimmable_account_nodes: usize,
    /// Bytes of those account-side nodes.
    pub trimmable_account_bytes: usize,
    /// Of the trimmable nodes, those occurring only in storage tries.
    pub trimmable_storage_nodes: usize,
    /// Bytes of those storage-side nodes.
    pub trimmable_storage_bytes: usize,
    /// Witness entries shorter than 32 bytes: embedded in their parents, so neither trimmable nor
    /// usable as a fragment anchor.
    pub inline_nodes: usize,
    /// Witness entries the decoded proof walk did not attribute to any trie. Expected zero; a
    /// nonzero count means the witness and its decoded form disagree and the block's measurement
    /// should not be trusted.
    pub unattributed_nodes: usize,
    /// Trimmable account-side bytes by node depth, shallowest first.
    pub trimmable_account_bytes_by_depth: [u64; WITNESS_TRIM_DEPTH_LEVELS],
    /// Trimmable storage-side bytes by node depth within their storage tries, shallowest first.
    pub trimmable_storage_bytes_by_depth: [u64; WITNESS_TRIM_DEPTH_LEVELS],
}

/// Everywhere one witness node appears across the decoded proof, folded into what the trim needs.
#[derive(Debug, Clone, Copy)]
struct NodeOccurrence {
    /// Whether every occurrence so far is revealed by the receiver's trie cache.
    trimmable_everywhere: bool,
    /// Whether any occurrence is in the account trie.
    in_account_trie: bool,
    /// Shallowest depth this node appears at, in nibbles.
    min_depth: usize,
}

/// Measures, against the trie cache the receiver holds at this block's parent, how much of the
/// flat witness `nodes` is redundant.
///
/// `trie_cache` must be the parent generation the sidecar applies to — the builder's own
/// pre-transition trie cache, which deterministic retention keeps equal to the receiver's.
/// `decoded_proof` must be the decode of exactly `nodes`; it supplies the path and trie each node
/// belongs to, which the flat format itself does not carry.
pub fn measure_witness_trim(
    trie_cache: &PartialTrieNodeCache,
    nodes: &[Bytes],
    decoded_proof: &DecodedMultiProofV2,
) -> WitnessTrimStats {
    let account_revealed = trie_cache.revealed_account_node_hashes();
    let storage_revealed = trie_cache.revealed_storage_node_hashes();
    let empty = HashSet::default();

    let mut occurrences = B256Map::<NodeOccurrence>::default();
    record_occurrences(&mut occurrences, &decoded_proof.account_proofs, &account_revealed, true);
    for (hashed_address, proof_nodes) in &decoded_proof.storage_proofs {
        let revealed = storage_revealed.get(hashed_address).unwrap_or(&empty);
        record_occurrences(&mut occurrences, proof_nodes, revealed, false);
    }

    let mut stats = WitnessTrimStats::default();
    for node in nodes {
        stats.witness_nodes += 1;
        stats.witness_node_bytes += node.len();
        if node.len() < 32 {
            stats.inline_nodes += 1;
            continue;
        }
        let Some(occurrence) = occurrences.get(&keccak256(node)) else {
            stats.unattributed_nodes += 1;
            continue;
        };
        if !occurrence.trimmable_everywhere {
            continue;
        }
        stats.trimmable_nodes += 1;
        stats.trimmable_bytes += node.len();
        let bucket = occurrence.min_depth.min(WITNESS_TRIM_DEPTH_LEVELS - 1);
        if occurrence.in_account_trie {
            stats.trimmable_account_nodes += 1;
            stats.trimmable_account_bytes += node.len();
            stats.trimmable_account_bytes_by_depth[bucket] += node.len() as u64;
        } else {
            stats.trimmable_storage_nodes += 1;
            stats.trimmable_storage_bytes += node.len();
            stats.trimmable_storage_bytes_by_depth[bucket] += node.len() as u64;
        }
    }
    stats
}

/// Folds one trie's decoded proof nodes into the occurrence map.
///
/// Encoding mirrors the flat-witness writer exactly: each node's own RLP, plus — for a branch
/// carrying a short key — the bare branch encoding that the wire stores as a second entry. Any
/// divergence here surfaces as `unattributed_nodes` rather than a silent misattribution.
fn record_occurrences(
    occurrences: &mut B256Map<NodeOccurrence>,
    proof_nodes: &[ProofTrieNodeV2],
    revealed: &HashSet<B256>,
    in_account_trie: bool,
) {
    let mut encoded = Vec::new();
    let mut record = |hash: B256, depth: usize| {
        let trimmable = revealed.contains(&hash);
        occurrences
            .entry(hash)
            .and_modify(|occurrence| {
                occurrence.trimmable_everywhere &= trimmable;
                occurrence.in_account_trie |= in_account_trie;
                occurrence.min_depth = occurrence.min_depth.min(depth);
            })
            .or_insert(NodeOccurrence {
                trimmable_everywhere: trimmable,
                in_account_trie,
                min_depth: depth,
            });
    };

    for proof_node in proof_nodes {
        encoded.clear();
        proof_node.node.encode(&mut encoded);
        record(keccak256(&encoded), proof_node.path.len());

        if let TrieNodeV2::Branch(branch) = &proof_node.node &&
            !branch.key.is_empty()
        {
            encoded.clear();
            BranchNodeRef::new(&branch.stack, branch.state_mask).encode(&mut encoded);
            record(keccak256(&encoded), proof_node.path.len() + branch.key.len());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::CacheConfig, policy::AccountData, BlockAccessedState};
    use alloy_primitives::{Address, U256};
    use reth_trie_common::{proof::ProofRetainer, HashBuilder, Nibbles, TrieAccount};

    /// A committed-looking cache plus the witness it was restored from, over `addresses`.
    fn cache_and_witness(
        addresses: &[Address],
    ) -> (PartialTrieNodeCache, Vec<Bytes>, DecodedMultiProofV2) {
        cache_and_witness_with_warm(addresses, addresses)
    }

    /// Like [`cache_and_witness`], but the value cache retains only `warm`, so restoring blinds
    /// the proven-but-unretained remainder — the shape a committed cache actually has.
    fn cache_and_witness_with_warm(
        addresses: &[Address],
        warm: &[Address],
    ) -> (PartialTrieNodeCache, Vec<Bytes>, DecodedMultiProofV2) {
        let mut leaves: Vec<(B256, Vec<u8>)> = addresses
            .iter()
            .map(|address| {
                let account = TrieAccount {
                    nonce: 1,
                    balance: U256::from(1_000_000u64),
                    ..Default::default()
                };
                (keccak256(address), alloy_rlp::encode(&account))
            })
            .collect();
        leaves.sort_unstable_by_key(|(hashed, _)| *hashed);

        let targets = leaves.iter().map(|(hashed, _)| Nibbles::unpack(hashed));
        let mut hash_builder =
            HashBuilder::default().with_proof_retainer(ProofRetainer::from_iter(targets));
        for (hashed, value) in &leaves {
            hash_builder.add_leaf(Nibbles::unpack(hashed), value);
        }
        let root = hash_builder.root();

        let mut witness = B256Map::<Bytes>::default();
        for (_, encoded) in hash_builder.take_proof_nodes().into_inner() {
            witness.insert(keccak256(&encoded), encoded);
        }
        let proof = DecodedMultiProofV2::from_witness(root, &witness)
            .expect("hand-built witness must decode");

        let mut accessed = BlockAccessedState::default();
        for address in warm {
            accessed.accounts.insert(
                *address,
                AccountData { nonce: 1, balance: U256::from(1_000_000u64), code_hash: None },
            );
        }
        let config = CacheConfig::default();
        let mut value_cache = config.new_cache_at(0);
        value_cache.on_block_executed(1, &accessed);

        let cache = PartialTrieNodeCache::restore_from_decoded_multiproof(
            proof.clone(),
            root,
            &value_cache,
        )
        .expect("restore over its own witness must succeed");
        (cache, witness.into_values().collect(), proof)
    }

    fn addresses(count: u8) -> Vec<Address> {
        (1..=count).map(Address::with_last_byte).collect()
    }

    #[test]
    fn a_cold_cache_trims_nothing() {
        let (_, nodes, proof) = cache_and_witness(&addresses(8));
        let stats = measure_witness_trim(&PartialTrieNodeCache::new(), &nodes, &proof);
        assert_eq!(stats.witness_nodes, nodes.len());
        assert_eq!(stats.trimmable_nodes, 0);
        assert_eq!(stats.trimmable_bytes, 0);
        assert_eq!(stats.unattributed_nodes, 0);
    }

    #[test]
    fn a_cache_revealed_from_the_same_witness_trims_every_hash_addressed_node() {
        let (cache, nodes, proof) = cache_and_witness(&addresses(8));
        let stats = measure_witness_trim(&cache, &nodes, &proof);
        assert_eq!(stats.unattributed_nodes, 0);
        assert_eq!(
            stats.trimmable_nodes,
            stats.witness_nodes - stats.inline_nodes,
            "everything the receiver revealed and the wire can address must be trimmable"
        );
        assert_eq!(stats.trimmable_bytes + inline_bytes(&nodes), stats.witness_node_bytes);
        assert!(
            stats.trimmable_account_bytes_by_depth[0] > 0,
            "the root node is revealed and must be counted at depth zero"
        );
    }

    #[test]
    fn a_partially_overlapping_cache_trims_the_shared_prefix_only() {
        // The cache knows the first half of the addresses; the witness proves all of them. The
        // shared upper structure must trim, the cold leaves must not.
        let all = addresses(8);
        let (cache, _, _) = cache_and_witness(&all[..4]);
        let (_, nodes, proof) = cache_and_witness(&all);

        let stats = measure_witness_trim(&cache, &nodes, &proof);
        assert_eq!(stats.unattributed_nodes, 0);
        assert!(stats.trimmable_nodes > 0, "the shared root region must be trimmable");
        assert!(
            stats.trimmable_nodes < stats.witness_nodes - stats.inline_nodes,
            "paths the cache never revealed must stay in the witness"
        );
    }

    fn inline_bytes(nodes: &[Bytes]) -> usize {
        nodes.iter().filter(|node| node.len() < 32).map(|node| node.len()).sum()
    }

    #[test]
    fn the_branch_census_counts_blinding_that_retention_produced() {
        // The witness proves eight accounts, the value cache retains four: restoring prunes the
        // other four subtrees, and every pruned child must show up as a blinded slot.
        let all = addresses(8);
        let (cache, _, _) = cache_and_witness_with_warm(&all, &all[..4]);
        let census = cache.branch_slot_census();
        assert!(census.account.branches > 0, "eight distinct keys must branch somewhere");
        assert!(census.account.blinded_slots > 0, "pruned siblings must be blinded");
        assert!(census.account.blinded_slots <= census.account.present_slots);
        assert_eq!(
            census.account.branches,
            census.account.branches_by_depth.iter().sum::<u64>(),
            "the depth buckets must partition the branches"
        );
        assert_eq!(
            census.account.blinded_slots,
            census.account.blinded_by_depth.iter().sum::<u64>(),
        );

        // A fully revealed cache has no blinded child anywhere.
        let (full, _, _) = cache_and_witness(&all);
        assert_eq!(full.branch_slot_census().account.blinded_slots, 0);
    }
}
