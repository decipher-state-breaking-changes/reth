//! Benchmark-only cache for trie nodes observed in prior proofs.
//!
//! This does not change the protocol-level value cache. It models how much
//! witness data could be omitted if validators also retained structural proof
//! nodes (branch and extension nodes) they have already seen.
//!
//! Three cache models are measured side by side from the same proof stream:
//! - the positional model keyed by `(domain, path, content hash)` (the default),
//! - an account-only positional model (storage structural nodes excluded), kept for its memory
//!   footprint, and
//! - a content-addressed (hash-only) model keyed solely by content hash, which estimates the extra
//!   redundancy a protocol-accurate hash-keyed node cache would capture (same bytes reachable at a
//!   different path or trie).

use crate::witness::{flatten_multiproof_nodes, ProofNodeDomain, ProofNodeKind, ProofNodeRecord};
use alloy_primitives::{map::HashMap, Bytes, B256};
use reth_trie_common::MultiProof;

const DOMAIN_ACCOUNT_BYTES: usize = 1;
const DOMAIN_STORAGE_BYTES: usize = 1 + 32;
const CONTENT_HASH_BYTES: usize = 32;
const NODE_KIND_BYTES: usize = 1;
const CACHE_METADATA_BYTES: usize = 24;
const HASHMAP_ENTRY_OVERHEAD_BYTES: usize = 32;

/// Compact identity for an exactly cached proof node.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ProofNodeCacheKey {
    domain: ProofNodeDomain,
    path: Vec<u8>,
    rlp_hash: B256,
}

impl ProofNodeCacheKey {
    fn from_node(node: &ProofNodeRecord) -> Self {
        Self { domain: node.domain.clone(), path: node.path.clone(), rlp_hash: node.rlp_hash }
    }

    fn is_storage(&self) -> bool {
        matches!(self.domain, ProofNodeDomain::Storage(_))
    }

    fn domain_memory_bytes(&self) -> usize {
        if self.is_storage() {
            DOMAIN_STORAGE_BYTES
        } else {
            DOMAIN_ACCOUNT_BYTES
        }
    }
}

/// Trie structural-node payload retained from prior sidecars/proofs.
#[derive(Debug, Clone)]
pub struct CachedProofNode {
    pub rlp: Bytes,
    pub kind: ProofNodeKind,
    pub first_seen_block: u64,
    pub last_seen_block: u64,
    pub seen_count: u32,
}

impl CachedProofNode {
    fn new(node: &ProofNodeRecord, block: u64) -> Self {
        Self {
            rlp: node.rlp.clone(),
            kind: node.kind,
            first_seen_block: block,
            last_seen_block: block,
            seen_count: 1,
        }
    }

    fn touch(&mut self, block: u64) {
        self.last_seen_block = block;
        self.seen_count = self.seen_count.saturating_add(1);
    }
}

/// Current structural-node cache footprint.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct BranchNodeCacheFootprint {
    pub total_nodes: usize,
    pub account_nodes: usize,
    pub storage_nodes: usize,
    pub branch_nodes: usize,
    pub extension_nodes: usize,
    /// Modeled cache bytes, not allocator RSS.
    pub estimated_memory_bytes: usize,
}

/// Avoidable bytes for a proof against the current structural-node cache.
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
    pub avoidable_account_extension_bytes: usize,
    pub avoidable_account_extension_nodes: usize,
    pub avoidable_storage_extension_bytes: usize,
    pub avoidable_storage_extension_nodes: usize,
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
    pub cache_before: BranchNodeCacheFootprint,
    pub cache_after: BranchNodeCacheFootprint,
    pub avoidance: BranchNodeAvoidanceStats,
    /// Account-only positional model: kept for its memory footprint. Its avoidable
    /// bytes equal the account-domain subset already reported in `avoidance`.
    pub account_only_cache_after: BranchNodeCacheFootprint,
    pub account_only_avoidable_structural_bytes: usize,
    pub account_only_avoidable_structural_nodes: usize,
    /// Content-addressed (hash-only) model: estimates redundancy a hash-keyed node
    /// cache would capture beyond the positional model.
    pub hash_only_cache_after: BranchNodeCacheFootprint,
    pub hash_only_avoidable_structural_bytes: usize,
    pub hash_only_avoidable_structural_nodes: usize,
}

/// Benchmark-only observed proof-node cache.
#[derive(Debug, Default)]
pub struct ObservedBranchNodeCache {
    nodes: HashMap<ProofNodeCacheKey, CachedProofNode>,
    account_only_nodes: HashMap<ProofNodeCacheKey, CachedProofNode>,
    hash_only_nodes: HashMap<B256, CachedProofNode>,
    account_window: u64,
    storage_window: u64,
}

impl ObservedBranchNodeCache {
    pub fn new(account_window: u64, storage_window: u64) -> Self {
        Self {
            nodes: HashMap::default(),
            account_only_nodes: HashMap::default(),
            hash_only_nodes: HashMap::default(),
            account_window,
            storage_window,
        }
    }

    pub fn footprint(&self) -> BranchNodeCacheFootprint {
        Self::footprint_for(&self.nodes)
    }

    pub fn account_only_footprint(&self) -> BranchNodeCacheFootprint {
        Self::footprint_for(&self.account_only_nodes)
    }

    pub fn hash_only_footprint(&self) -> BranchNodeCacheFootprint {
        let mut footprint = BranchNodeCacheFootprint::default();
        for cached in self.hash_only_nodes.values() {
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
        let cache_before = self.footprint();
        let mut avoidance = Self::avoidable_stats_for(&self.nodes, &proof_nodes);
        let (hash_only_avoidable_structural_bytes, hash_only_avoidable_structural_nodes) =
            Self::hash_only_avoidable_stats_for(&self.hash_only_nodes, &proof_nodes);

        for node in &proof_nodes {
            if !node.is_structural() {
                continue;
            }
            Self::observe_node(&mut self.nodes, block_number, node);
            if !node.is_storage() {
                Self::observe_node(&mut self.account_only_nodes, block_number, node);
            }
            Self::observe_hash_only_node(&mut self.hash_only_nodes, block_number, node);
        }

        self.evict(block_number);
        self.evict_account_only(block_number);
        self.evict_hash_only(block_number);

        avoidance.finalize();

        // The account-only positional cache retains the same account-domain
        // structural nodes as the main cache, so its avoidable bytes are exactly
        // the account-domain subset already computed above.
        let account_only_avoidable_structural_bytes =
            avoidance.avoidable_account_branch_bytes + avoidance.avoidable_account_extension_bytes;
        let account_only_avoidable_structural_nodes =
            avoidance.avoidable_account_branch_nodes + avoidance.avoidable_account_extension_nodes;

        BranchNodeCacheUpdate {
            cache_before,
            cache_after: self.footprint(),
            avoidance,
            account_only_cache_after: self.account_only_footprint(),
            account_only_avoidable_structural_bytes,
            account_only_avoidable_structural_nodes,
            hash_only_cache_after: self.hash_only_footprint(),
            hash_only_avoidable_structural_bytes,
            hash_only_avoidable_structural_nodes,
        }
    }

    fn observe_node(
        nodes: &mut HashMap<ProofNodeCacheKey, CachedProofNode>,
        block_number: u64,
        node: &ProofNodeRecord,
    ) {
        let key = ProofNodeCacheKey::from_node(node);
        match nodes.get_mut(&key) {
            Some(cached) => cached.touch(block_number),
            None => {
                nodes.insert(key, CachedProofNode::new(node, block_number));
            }
        }
    }

    fn observe_hash_only_node(
        nodes: &mut HashMap<B256, CachedProofNode>,
        block_number: u64,
        node: &ProofNodeRecord,
    ) {
        match nodes.get_mut(&node.rlp_hash) {
            Some(cached) => cached.touch(block_number),
            None => {
                nodes.insert(node.rlp_hash, CachedProofNode::new(node, block_number));
            }
        }
    }

    fn footprint_for(
        nodes: &HashMap<ProofNodeCacheKey, CachedProofNode>,
    ) -> BranchNodeCacheFootprint {
        let mut footprint = BranchNodeCacheFootprint::default();
        for (key, cached) in nodes {
            footprint.total_nodes += 1;
            if key.is_storage() {
                footprint.storage_nodes += 1;
            } else {
                footprint.account_nodes += 1;
            }
            match cached.kind {
                ProofNodeKind::Branch => footprint.branch_nodes += 1,
                ProofNodeKind::Extension => footprint.extension_nodes += 1,
                ProofNodeKind::Other => {}
            }
            footprint.estimated_memory_bytes += Self::estimated_entry_memory_bytes(key, cached);
        }
        footprint
    }

    fn estimated_entry_memory_bytes(key: &ProofNodeCacheKey, cached: &CachedProofNode) -> usize {
        // Modeled cache footprint: compact key + retained node bytes + lightweight metadata.
        key.domain_memory_bytes() +
            key.path.len() +
            CONTENT_HASH_BYTES +
            cached.rlp.len() +
            NODE_KIND_BYTES +
            CACHE_METADATA_BYTES +
            HASHMAP_ENTRY_OVERHEAD_BYTES
    }

    fn avoidable_stats_for(
        nodes: &HashMap<ProofNodeCacheKey, CachedProofNode>,
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
            let key = ProofNodeCacheKey::from_node(node);
            let Some(cached) = nodes.get(&key) else { continue };
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
                    if is_storage {
                        stats.avoidable_storage_extension_bytes += bytes;
                        stats.avoidable_storage_extension_nodes += 1;
                    } else {
                        stats.avoidable_account_extension_bytes += bytes;
                        stats.avoidable_account_extension_nodes += 1;
                    }
                }
                ProofNodeKind::Other => {}
            }
        }

        stats
    }

    /// Content-addressed avoidance: a structural node is avoidable if a node with
    /// the same content hash was seen in a prior block, regardless of path or trie.
    fn hash_only_avoidable_stats_for(
        nodes: &HashMap<B256, CachedProofNode>,
        proof_nodes: &[ProofNodeRecord],
    ) -> (usize, usize) {
        let mut bytes = 0usize;
        let mut count = 0usize;
        for node in proof_nodes {
            if !node.is_structural() {
                continue;
            }
            let Some(cached) = nodes.get(&node.rlp_hash) else { continue };
            if cached.rlp != node.rlp {
                continue;
            }
            bytes += node.rlp.len();
            count += 1;
        }
        (bytes, count)
    }

    fn evict(&mut self, current_block: u64) {
        let account_cutoff = current_block.saturating_sub(self.account_window);
        let storage_cutoff = current_block.saturating_sub(self.storage_window);
        self.nodes.retain(|node, cached| {
            let cutoff = if node.is_storage() { storage_cutoff } else { account_cutoff };
            cached.last_seen_block >= cutoff
        });
    }

    fn evict_account_only(&mut self, current_block: u64) {
        let account_cutoff = current_block.saturating_sub(self.account_window);
        self.account_only_nodes.retain(|_, cached| cached.last_seen_block >= account_cutoff);
    }

    fn evict_hash_only(&mut self, current_block: u64) {
        // Content-addressed entries are domain-agnostic, so retain them for the
        // wider of the two windows.
        let window = self.account_window.max(self.storage_window);
        let cutoff = current_block.saturating_sub(window);
        self.hash_only_nodes.retain(|_, cached| cached.last_seen_block >= cutoff);
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

    fn alternate_branch_rlp_vec() -> Vec<u8> {
        let mut bytes = branch_rlp_vec();
        *bytes.last_mut().unwrap() = 0x00;
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
    fn compact_key_exact_match_counts_as_avoidable() {
        let mut cache = ObservedBranchNodeCache::new(10, 10);
        let rlp = branch_rlp_vec();
        let first = proof(vec![(&[1], rlp.clone())], vec![]);
        let second = proof(vec![(&[1], rlp.clone())], vec![]);

        let first_update = cache.measure_and_update(1, &first);
        assert_eq!(first_update.avoidance.avoidable_branch_bytes, 0);

        let second_update = cache.measure_and_update(2, &second);
        assert_eq!(second_update.avoidance.avoidable_branch_bytes, rlp.len());
        assert_eq!(second_update.avoidance.avoidable_account_branch_bytes, rlp.len());
        assert_eq!(second_update.avoidance.avoidable_storage_branch_bytes, 0);
    }

    #[test]
    fn compact_key_requires_same_rlp_bytes() {
        let mut cache = ObservedBranchNodeCache::new(10, 10);
        let cached_rlp = branch_rlp_vec();
        let changed_rlp = alternate_branch_rlp_vec();
        cache.measure_and_update(1, &proof(vec![(&[1], cached_rlp)], vec![]));

        let update = cache.measure_and_update(2, &proof(vec![(&[1], changed_rlp)], vec![]));
        assert_eq!(update.avoidance.avoidable_branch_bytes, 0);
        assert_eq!(update.avoidance.avoidable_branch_nodes, 0);
    }

    #[test]
    fn compact_key_separates_account_and_storage_domains() {
        let mut cache = ObservedBranchNodeCache::new(10, 10);
        let rlp = branch_rlp_vec();
        let storage_account = B256::repeat_byte(0xaa);
        cache.measure_and_update(1, &proof(vec![], vec![(storage_account, &[1], rlp.clone())]));

        let update = cache.measure_and_update(2, &proof(vec![(&[1], rlp)], vec![]));
        assert_eq!(update.avoidance.avoidable_branch_bytes, 0);
    }

    #[test]
    fn compact_key_separates_storage_accounts() {
        let mut cache = ObservedBranchNodeCache::new(10, 10);
        let rlp = branch_rlp_vec();
        cache.measure_and_update(
            1,
            &proof(vec![], vec![(B256::repeat_byte(0xaa), &[1], rlp.clone())]),
        );

        let update =
            cache.measure_and_update(2, &proof(vec![], vec![(B256::repeat_byte(0xbb), &[1], rlp)]));
        assert_eq!(update.avoidance.avoidable_branch_bytes, 0);
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
    fn hash_only_model_matches_same_bytes_at_different_path() {
        let mut cache = ObservedBranchNodeCache::new(10, 10);
        let rlp = branch_rlp_vec();
        cache.measure_and_update(1, &proof(vec![(&[1], rlp.clone())], vec![]));

        // Same node bytes, different path: positional model misses, hash-only hits.
        let update = cache.measure_and_update(2, &proof(vec![(&[2], rlp.clone())], vec![]));
        assert_eq!(update.avoidance.avoidable_branch_bytes, 0);
        assert_eq!(update.hash_only_avoidable_structural_bytes, rlp.len());
        assert_eq!(update.hash_only_avoidable_structural_nodes, 1);
    }

    #[test]
    fn footprint_counts_path_and_rlp_once() {
        let mut cache = ObservedBranchNodeCache::new(10, 10);
        let rlp = branch_rlp_vec();
        let path = [1, 2, 3];
        cache.measure_and_update(1, &proof(vec![(&path, rlp.clone())], vec![]));

        let footprint = cache.footprint();
        assert_eq!(footprint.total_nodes, 1);
        assert_eq!(footprint.account_nodes, 1);
        assert_eq!(footprint.storage_nodes, 0);
        assert_eq!(footprint.branch_nodes, 1);
        assert_eq!(footprint.extension_nodes, 0);
        assert_eq!(
            footprint.estimated_memory_bytes,
            DOMAIN_ACCOUNT_BYTES +
                path.len() +
                CONTENT_HASH_BYTES +
                rlp.len() +
                NODE_KIND_BYTES +
                CACHE_METADATA_BYTES +
                HASHMAP_ENTRY_OVERHEAD_BYTES
        );

        let representative_path_len = 32usize;
        let representative_rlp_len = 256usize;
        let compact_estimate = DOMAIN_ACCOUNT_BYTES +
            representative_path_len +
            CONTENT_HASH_BYTES +
            representative_rlp_len +
            NODE_KIND_BYTES +
            CACHE_METADATA_BYTES +
            HASHMAP_ENTRY_OVERHEAD_BYTES;
        let duplicate_record_estimate = 2 *
            (DOMAIN_ACCOUNT_BYTES + representative_path_len + representative_rlp_len) +
            CACHE_METADATA_BYTES +
            HASHMAP_ENTRY_OVERHEAD_BYTES;
        assert!(compact_estimate < duplicate_record_estimate);
    }

    #[test]
    fn measure_and_update_uses_cache_state_before_current_block() {
        let mut cache = ObservedBranchNodeCache::new(10, 10);
        let rlp = branch_rlp_vec();
        let update = cache
            .measure_and_update(1, &proof(vec![(&[1], rlp.clone()), (&[2], rlp.clone())], vec![]));
        assert_eq!(update.avoidance.avoidable_branch_bytes, 0);
        assert_eq!(update.cache_after.branch_nodes, 2);

        let update = cache.measure_and_update(2, &proof(vec![(&[1], rlp.clone())], vec![]));
        assert_eq!(update.avoidance.avoidable_branch_bytes, rlp.len());
    }

    #[test]
    fn eviction_follows_account_and_storage_windows() {
        let mut cache = ObservedBranchNodeCache::new(10, 3);
        let rlp = branch_rlp_vec();
        cache.measure_and_update(
            10,
            &proof(vec![(&[1], rlp.clone())], vec![(B256::repeat_byte(0xaa), &[2], rlp.clone())]),
        );

        let update = cache.measure_and_update(14, &proof(vec![], vec![]));
        assert_eq!(update.cache_after.account_nodes, 1);
        assert_eq!(update.cache_after.storage_nodes, 0);
    }

    #[test]
    fn account_only_model_excludes_storage_nodes_and_bytes() {
        let mut cache = ObservedBranchNodeCache::new(10, 10);
        let account_rlp = branch_rlp_vec();
        let storage_rlp = branch_rlp_vec();
        let storage_account = B256::repeat_byte(0xaa);
        let observed = proof(
            vec![(&[1], account_rlp.clone())],
            vec![(storage_account, &[2], storage_rlp.clone())],
        );
        cache.measure_and_update(1, &observed);

        let update = cache.measure_and_update(2, &observed);
        assert_eq!(update.avoidance.avoidable_branch_bytes, account_rlp.len() + storage_rlp.len());
        assert_eq!(
            update.avoidance.avoidable_branch_bytes,
            update.avoidance.avoidable_account_branch_bytes +
                update.avoidance.avoidable_storage_branch_bytes
        );
        assert_eq!(update.account_only_avoidable_structural_bytes, account_rlp.len());
        assert_eq!(update.account_only_cache_after.account_nodes, 1);
        assert_eq!(update.account_only_cache_after.storage_nodes, 0);
    }

    #[test]
    fn empty_cache_has_no_avoidable_bytes() {
        let mut cache = ObservedBranchNodeCache::new(10, 10);
        let update = cache.measure_and_update(1, &proof(vec![(&[1], branch_rlp_vec())], vec![]));
        assert_eq!(update.avoidance.avoidable_branch_bytes, 0);
        assert_eq!(update.avoidance.avoidable_extension_bytes, 0);
        assert_eq!(update.avoidance.avoidable_structural_bytes, 0);
        assert_eq!(update.hash_only_avoidable_structural_bytes, 0);
        assert_eq!(
            update.avoidance.adjusted_partial_mpt_bytes,
            update.avoidance.partial_mpt_bytes_before
        );
    }
}
