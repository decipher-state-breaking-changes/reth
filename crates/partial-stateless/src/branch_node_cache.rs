//! Benchmark-only content-addressed cache for structural trie nodes.
//!
//! This does not change the protocol-level value cache. It models how much
//! witness data could be omitted if validators also retained the structural
//! proof nodes (branch and extension nodes) they have already seen.
//!
//! The cache is keyed solely by node content hash — the identity an MPT actually
//! uses, since a parent references each child by `keccak(child_rlp)`. A node is
//! avoidable when the same bytes were seen in a prior block, regardless of the
//! path or trie it now appears in. Branch and extension contributions are
//! reported separately, but there is a single cache.
//!
//! Eviction uses the same per-domain windows as the value cache: a node survives
//! while it was last seen within `account_window` in the account trie OR within
//! `storage_window` in a storage trie.

use crate::witness::{flatten_multiproof_nodes, ProofNodeKind, ProofNodeRecord};
use alloy_primitives::{map::HashMap, Bytes, B256};
use reth_trie_common::MultiProof;

const CONTENT_HASH_BYTES: usize = 32;
const NODE_KIND_BYTES: usize = 1;
const CACHE_METADATA_BYTES: usize = 24;
const HASHMAP_ENTRY_OVERHEAD_BYTES: usize = 32;

/// Content-addressed cache entry: one entry per node content hash, regardless of
/// where the node appears in the trie.
///
/// The same content can be reached in both the account and a storage trie, so
/// the last sighting is tracked per domain. This lets eviction use the same
/// per-domain windows as the value cache: the entry survives while it was last
/// seen within `account_window` (account trie) or `storage_window` (storage trie).
#[derive(Debug, Clone)]
struct CachedNode {
    rlp: Bytes,
    kind: ProofNodeKind,
    last_seen_account: Option<u64>,
    last_seen_storage: Option<u64>,
}

impl CachedNode {
    fn new(node: &ProofNodeRecord, block: u64) -> Self {
        let mut entry = Self {
            rlp: node.rlp.clone(),
            kind: node.kind,
            last_seen_account: None,
            last_seen_storage: None,
        };
        entry.touch(block, node.is_storage());
        entry
    }

    fn touch(&mut self, block: u64, is_storage: bool) {
        if is_storage {
            self.last_seen_storage = Some(block);
        } else {
            self.last_seen_account = Some(block);
        }
    }
}

/// Current structural-node cache footprint.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct BranchNodeCacheFootprint {
    pub total_nodes: usize,
    pub branch_nodes: usize,
    pub extension_nodes: usize,
    /// Modeled cache bytes, not allocator RSS.
    pub estimated_memory_bytes: usize,
}

/// Avoidable bytes for a proof against the current structural-node cache.
///
/// "Avoidable" means the exact node bytes are already in the cache (seen in a
/// prior block), so a node that also retained structural nodes could omit them
/// from the witness. Branch and extension are reported separately; the
/// account/storage split attributes each avoidable node to the trie it appeared
/// in for this block.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct BranchNodeAvoidanceStats {
    pub partial_mpt_bytes_before: usize,
    /// Branch + extension nodes already present in the cache.
    pub avoidable_structural_bytes: usize,
    pub avoidable_structural_nodes: usize,
    pub avoidable_branch_bytes: usize,
    pub avoidable_branch_nodes: usize,
    pub avoidable_extension_bytes: usize,
    pub avoidable_extension_nodes: usize,
    pub avoidable_account_branch_bytes: usize,
    pub avoidable_account_branch_nodes: usize,
    pub avoidable_storage_branch_bytes: usize,
    pub avoidable_storage_branch_nodes: usize,
    pub adjusted_partial_mpt_bytes: usize,
    pub branch_redundancy_ratio: Option<f64>,
    pub extension_redundancy_ratio: Option<f64>,
    pub structural_redundancy_ratio: Option<f64>,
    pub account_branch_redundancy_ratio: Option<f64>,
    pub storage_branch_redundancy_ratio: Option<f64>,
}

/// Result of measuring a proof against the cache and then observing it.
#[derive(Debug, Clone, Default)]
pub struct BranchNodeCacheUpdate {
    pub cache_after: BranchNodeCacheFootprint,
    pub avoidance: BranchNodeAvoidanceStats,
}

/// Benchmark-only content-addressed structural-node cache.
#[derive(Debug, Default)]
pub struct ObservedBranchNodeCache {
    nodes: HashMap<B256, CachedNode>,
    account_window: u64,
    storage_window: u64,
}

impl ObservedBranchNodeCache {
    pub fn new(account_window: u64, storage_window: u64) -> Self {
        Self { nodes: HashMap::default(), account_window, storage_window }
    }

    pub fn footprint(&self) -> BranchNodeCacheFootprint {
        let mut footprint = BranchNodeCacheFootprint::default();
        for cached in self.nodes.values() {
            footprint.total_nodes += 1;
            match cached.kind {
                ProofNodeKind::Branch => footprint.branch_nodes += 1,
                ProofNodeKind::Extension => footprint.extension_nodes += 1,
                ProofNodeKind::Other => {}
            }
            // Content-addressed entries carry no domain or path: just the hash key,
            // the retained bytes, and lightweight metadata.
            footprint.estimated_memory_bytes += CONTENT_HASH_BYTES +
                cached.rlp.len() +
                NODE_KIND_BYTES +
                CACHE_METADATA_BYTES +
                HASHMAP_ENTRY_OVERHEAD_BYTES;
        }
        footprint
    }

    pub fn measure_and_update(
        &mut self,
        block_number: u64,
        proof: &MultiProof,
    ) -> BranchNodeCacheUpdate {
        let proof_nodes = flatten_multiproof_nodes(proof);
        let mut avoidance = Self::avoidable_stats_for(&self.nodes, &proof_nodes);

        for node in &proof_nodes {
            if !node.is_structural() {
                continue;
            }
            Self::observe_node(&mut self.nodes, block_number, node);
        }

        self.evict(block_number);
        avoidance.finalize();

        BranchNodeCacheUpdate { cache_after: self.footprint(), avoidance }
    }

    fn observe_node(
        nodes: &mut HashMap<B256, CachedNode>,
        block_number: u64,
        node: &ProofNodeRecord,
    ) {
        match nodes.get_mut(&node.rlp_hash) {
            Some(cached) => cached.touch(block_number, node.is_storage()),
            None => {
                nodes.insert(node.rlp_hash, CachedNode::new(node, block_number));
            }
        }
    }

    fn avoidable_stats_for(
        nodes: &HashMap<B256, CachedNode>,
        proof_nodes: &[ProofNodeRecord],
    ) -> BranchNodeAvoidanceStats {
        let partial_mpt_bytes_before: usize = proof_nodes.iter().map(|node| node.rlp.len()).sum();
        let mut stats = BranchNodeAvoidanceStats {
            partial_mpt_bytes_before,
            adjusted_partial_mpt_bytes: partial_mpt_bytes_before,
            ..Default::default()
        };

        for node in proof_nodes {
            if !node.is_structural() {
                continue;
            }
            let Some(cached) = nodes.get(&node.rlp_hash) else { continue };
            if cached.rlp != node.rlp {
                continue;
            }
            let bytes = node.rlp.len();
            stats.avoidable_structural_bytes += bytes;
            stats.avoidable_structural_nodes += 1;
            let is_storage = node.is_storage();
            match node.kind {
                ProofNodeKind::Branch => {
                    stats.avoidable_branch_bytes += bytes;
                    stats.avoidable_branch_nodes += 1;
                    if is_storage {
                        stats.avoidable_storage_branch_bytes += bytes;
                        stats.avoidable_storage_branch_nodes += 1;
                    } else {
                        stats.avoidable_account_branch_bytes += bytes;
                        stats.avoidable_account_branch_nodes += 1;
                    }
                }
                ProofNodeKind::Extension => {
                    stats.avoidable_extension_bytes += bytes;
                    stats.avoidable_extension_nodes += 1;
                }
                ProofNodeKind::Other => {}
            }
        }

        stats
    }

    fn evict(&mut self, current_block: u64) {
        // A content-addressed entry survives while any of its sightings is still
        // within that domain's window — matching the positional/value-cache
        // retention so the only modeled difference is the content-hash key.
        let account_cutoff = current_block.saturating_sub(self.account_window);
        let storage_cutoff = current_block.saturating_sub(self.storage_window);
        self.nodes.retain(|_, cached| {
            let account_ok = cached.last_seen_account.is_some_and(|block| block >= account_cutoff);
            let storage_ok = cached.last_seen_storage.is_some_and(|block| block >= storage_cutoff);
            account_ok || storage_ok
        });
    }
}

impl BranchNodeAvoidanceStats {
    fn finalize(&mut self) {
        self.adjusted_partial_mpt_bytes =
            self.partial_mpt_bytes_before.saturating_sub(self.avoidable_structural_bytes);
        if self.partial_mpt_bytes_before > 0 {
            let before = self.partial_mpt_bytes_before as f64;
            self.branch_redundancy_ratio = Some(self.avoidable_branch_bytes as f64 / before);
            self.extension_redundancy_ratio = Some(self.avoidable_extension_bytes as f64 / before);
            self.structural_redundancy_ratio =
                Some(self.avoidable_structural_bytes as f64 / before);
            self.account_branch_redundancy_ratio =
                Some(self.avoidable_account_branch_bytes as f64 / before);
            self.storage_branch_redundancy_ratio =
                Some(self.avoidable_storage_branch_bytes as f64 / before);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::map::{B256Map, HashMap};
    use alloy_rlp::Encodable;
    use reth_trie_common::{
        proof::ProofNodes, BranchNodeMasksMap, ExtensionNode, MultiProof, Nibbles, RlpNode,
        StorageMultiProof, TrieNode,
    };

    fn branch_rlp_vec() -> Vec<u8> {
        let mut bytes = vec![0xd1];
        bytes.extend([0x80; 17]);
        bytes
    }

    /// A second, distinct, *valid* branch: one 32-byte child at slot 0, value slot empty.
    fn other_branch_rlp_vec() -> Vec<u8> {
        let mut payload = vec![0xa0];
        payload.extend([0x22; 32]); // slot 0 = 32-byte child reference
        payload.extend([0x80; 16]); // slots 1..=15 empty + value slot empty
        let mut bytes = vec![0xc0 + payload.len() as u8];
        bytes.extend(payload);
        bytes
    }

    fn extension_rlp_vec() -> Vec<u8> {
        let child = RlpNode::word_rlp(&B256::repeat_byte(0x11));
        let node =
            TrieNode::Extension(ExtensionNode::new(Nibbles::from_nibbles([0x1, 0x2]), child));
        let mut buf = Vec::new();
        node.encode(&mut buf);
        buf
    }

    fn proof(
        account_nodes: Vec<(&[u8], Vec<u8>)>,
        storage_nodes: Vec<(B256, &[u8], Vec<u8>)>,
    ) -> MultiProof {
        let mut account_map: HashMap<Nibbles, Bytes> = HashMap::default();
        for (path, rlp) in account_nodes {
            account_map.insert(Nibbles::from_nibbles(path), Bytes::from(rlp));
        }

        let mut storages = B256Map::default();
        for (hashed_address, path, rlp) in storage_nodes {
            let storage = storages.entry(hashed_address).or_insert_with(|| StorageMultiProof {
                root: B256::ZERO,
                subtree: ProofNodes::default(),
                branch_node_masks: BranchNodeMasksMap::default(),
            });
            storage.subtree.insert(Nibbles::from_nibbles(path), Bytes::from(rlp));
        }

        MultiProof {
            account_subtree: ProofNodes::from_iter(account_map),
            branch_node_masks: BranchNodeMasksMap::default(),
            storages,
        }
    }

    #[test]
    fn exact_content_match_counts_as_avoidable() {
        let mut cache = ObservedBranchNodeCache::new(10, 10);
        let rlp = branch_rlp_vec();
        let first_update = cache.measure_and_update(1, &proof(vec![(&[1], rlp.clone())], vec![]));
        assert_eq!(first_update.avoidance.avoidable_branch_bytes, 0);

        let second_update = cache.measure_and_update(2, &proof(vec![(&[1], rlp.clone())], vec![]));
        assert_eq!(second_update.avoidance.avoidable_branch_bytes, rlp.len());
        assert_eq!(second_update.avoidance.avoidable_structural_bytes, rlp.len());
    }

    #[test]
    fn requires_same_rlp_bytes() {
        let mut cache = ObservedBranchNodeCache::new(10, 10);
        cache.measure_and_update(1, &proof(vec![(&[1], branch_rlp_vec())], vec![]));

        let update =
            cache.measure_and_update(2, &proof(vec![(&[1], other_branch_rlp_vec())], vec![]));
        assert_eq!(update.avoidance.avoidable_branch_bytes, 0);
        assert_eq!(update.avoidance.avoidable_branch_nodes, 0);
    }

    #[test]
    fn same_bytes_at_different_path_are_avoidable() {
        let mut cache = ObservedBranchNodeCache::new(10, 10);
        let rlp = branch_rlp_vec();
        cache.measure_and_update(1, &proof(vec![(&[1], rlp.clone())], vec![]));

        // Content addressing: same bytes, different path -> still a hit.
        let update = cache.measure_and_update(2, &proof(vec![(&[2], rlp.clone())], vec![]));
        assert_eq!(update.avoidance.avoidable_branch_bytes, rlp.len());
    }

    #[test]
    fn same_content_across_tries_is_avoidable() {
        let mut cache = ObservedBranchNodeCache::new(10, 10);
        let rlp = branch_rlp_vec();
        // Seen in the account trie...
        cache.measure_and_update(1, &proof(vec![(&[1], rlp.clone())], vec![]));
        // ...then the identical node appears in a storage trie: content-addressed hit.
        let update = cache.measure_and_update(
            2,
            &proof(vec![], vec![(B256::repeat_byte(0xaa), &[1], rlp.clone())]),
        );
        assert_eq!(update.avoidance.avoidable_branch_bytes, rlp.len());
        assert_eq!(update.avoidance.avoidable_storage_branch_bytes, rlp.len());
        assert_eq!(update.avoidance.avoidable_account_branch_bytes, 0);
    }

    #[test]
    fn extension_nodes_are_cached_and_reported_separately() {
        let mut cache = ObservedBranchNodeCache::new(10, 10);
        let branch = branch_rlp_vec();
        let ext = extension_rlp_vec();
        let observed = proof(vec![(&[1], branch.clone()), (&[2], ext.clone())], vec![]);
        cache.measure_and_update(1, &observed);

        let update = cache.measure_and_update(2, &observed);
        assert_eq!(update.avoidance.avoidable_branch_bytes, branch.len());
        assert_eq!(update.avoidance.avoidable_branch_nodes, 1);
        assert_eq!(update.avoidance.avoidable_extension_bytes, ext.len());
        assert_eq!(update.avoidance.avoidable_extension_nodes, 1);
        assert_eq!(update.avoidance.avoidable_structural_bytes, branch.len() + ext.len());
        assert_eq!(update.cache_after.branch_nodes, 1);
        assert_eq!(update.cache_after.extension_nodes, 1);
    }

    #[test]
    fn account_storage_split_attributes_by_trie() {
        let mut cache = ObservedBranchNodeCache::new(10, 10);
        let account_rlp = branch_rlp_vec();
        let storage_rlp = other_branch_rlp_vec();
        let observed = proof(
            vec![(&[1], account_rlp.clone())],
            vec![(B256::repeat_byte(0xaa), &[2], storage_rlp.clone())],
        );
        cache.measure_and_update(1, &observed);

        let update = cache.measure_and_update(2, &observed);
        assert_eq!(update.avoidance.avoidable_account_branch_bytes, account_rlp.len());
        assert_eq!(update.avoidance.avoidable_storage_branch_bytes, storage_rlp.len());
        assert_eq!(update.avoidance.avoidable_branch_bytes, account_rlp.len() + storage_rlp.len());
    }

    #[test]
    fn evicts_storage_content_on_storage_window() {
        let mut cache = ObservedBranchNodeCache::new(10, 3);
        let rlp = branch_rlp_vec();
        let storage_account = B256::repeat_byte(0xaa);
        cache.measure_and_update(10, &proof(vec![], vec![(storage_account, &[2], rlp.clone())]));
        // Advance past the storage window (10 + 3 < 14) to trigger eviction.
        cache.measure_and_update(14, &proof(vec![], vec![]));

        let update =
            cache.measure_and_update(15, &proof(vec![], vec![(storage_account, &[2], rlp)]));
        assert_eq!(update.avoidance.avoidable_structural_bytes, 0);
    }

    #[test]
    fn keeps_account_content_for_account_window() {
        let mut cache = ObservedBranchNodeCache::new(10, 3);
        let rlp = branch_rlp_vec();
        cache.measure_and_update(10, &proof(vec![(&[1], rlp.clone())], vec![]));
        cache.measure_and_update(14, &proof(vec![], vec![]));

        // Account window is 10, so content seen at block 10 is still cached at 15.
        let update = cache.measure_and_update(15, &proof(vec![(&[1], rlp.clone())], vec![]));
        assert_eq!(update.avoidance.avoidable_structural_bytes, rlp.len());
    }

    #[test]
    fn measure_uses_cache_state_before_current_block() {
        let mut cache = ObservedBranchNodeCache::new(10, 10);
        let rlp = branch_rlp_vec();
        let update = cache
            .measure_and_update(1, &proof(vec![(&[1], rlp.clone()), (&[2], rlp.clone())], vec![]));
        // Both share one content hash -> a single cache entry, nothing avoidable yet.
        assert_eq!(update.avoidance.avoidable_branch_bytes, 0);
        assert_eq!(update.cache_after.total_nodes, 1);

        let update = cache.measure_and_update(2, &proof(vec![(&[1], rlp.clone())], vec![]));
        assert_eq!(update.avoidance.avoidable_branch_bytes, rlp.len());
    }

    #[test]
    fn footprint_memory_formula() {
        let mut cache = ObservedBranchNodeCache::new(10, 10);
        let rlp = branch_rlp_vec();
        cache.measure_and_update(1, &proof(vec![(&[1, 2, 3], rlp.clone())], vec![]));

        let footprint = cache.footprint();
        assert_eq!(footprint.total_nodes, 1);
        assert_eq!(footprint.branch_nodes, 1);
        assert_eq!(footprint.extension_nodes, 0);
        assert_eq!(
            footprint.estimated_memory_bytes,
            CONTENT_HASH_BYTES +
                rlp.len() +
                NODE_KIND_BYTES +
                CACHE_METADATA_BYTES +
                HASHMAP_ENTRY_OVERHEAD_BYTES
        );
    }

    #[test]
    fn empty_cache_has_no_avoidable_bytes() {
        let mut cache = ObservedBranchNodeCache::new(10, 10);
        let update = cache.measure_and_update(1, &proof(vec![(&[1], branch_rlp_vec())], vec![]));
        assert_eq!(update.avoidance.avoidable_structural_bytes, 0);
        assert_eq!(
            update.avoidance.adjusted_partial_mpt_bytes,
            update.avoidance.partial_mpt_bytes_before
        );
    }
}
