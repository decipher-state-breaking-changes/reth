//! Witness computation for cache-missed state.
//!
//! Converts `MissResult` into `MultiProofTargets` and provides helpers
//! for measuring witness (Merkle proof) size.

use crate::{network_cache::MissResult, BlockAccessedState, StateTargetSet, WitnessTargets};
use alloy_primitives::{keccak256, Address, Bytes, B256};
use alloy_rlp::Decodable;
use reth_trie_common::{MultiProof, MultiProofTargets, TrieNode};
use std::collections::HashSet;

/// Trie domain for a flattened proof node.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProofNodeDomain {
    /// Node from the account/state trie.
    Account,
    /// Node from a storage trie, namespaced by hashed account address.
    Storage(B256),
}

/// Structural classification of a flattened proof node.
///
/// Only `Branch` and `Extension` are reusable structural nodes that a node-level
/// cache can retain across blocks. Leaves (and the empty root) are tied to a
/// specific value and are classified as `Other`; they are never cached.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProofNodeKind {
    /// MPT branch node (17 items).
    Branch,
    /// MPT extension node (2 items, non-leaf).
    Extension,
    /// Leaf node or empty root; not a structural node.
    Other,
}

/// A flattened proof node with precomputed content hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofNodeRecord {
    pub domain: ProofNodeDomain,
    pub path: Vec<u8>,
    pub rlp: Bytes,
    pub rlp_hash: B256,
    pub kind: ProofNodeKind,
}

impl ProofNodeRecord {
    pub fn is_storage(&self) -> bool {
        matches!(self.domain, ProofNodeDomain::Storage(_))
    }

    pub fn is_branch(&self) -> bool {
        matches!(self.kind, ProofNodeKind::Branch)
    }

    pub fn is_extension(&self) -> bool {
        matches!(self.kind, ProofNodeKind::Extension)
    }

    /// Branch and extension nodes are the structural nodes a cache can retain.
    pub fn is_structural(&self) -> bool {
        matches!(self.kind, ProofNodeKind::Branch | ProofNodeKind::Extension)
    }
}

fn classify_node(node_bytes: &[u8]) -> ProofNodeKind {
    match TrieNode::decode(&mut &node_bytes[..]) {
        Ok(TrieNode::Branch(_)) => ProofNodeKind::Branch,
        Ok(TrieNode::Extension(_)) => ProofNodeKind::Extension,
        _ => ProofNodeKind::Other,
    }
}

/// Flatten a reth [`MultiProof`] into exact proof-node records.
pub fn flatten_multiproof_nodes(proof: &MultiProof) -> Vec<ProofNodeRecord> {
    let mut nodes = Vec::new();

    for (path, node_bytes) in proof.account_subtree.iter() {
        nodes.push(ProofNodeRecord {
            domain: ProofNodeDomain::Account,
            path: path.to_vec(),
            rlp: node_bytes.clone(),
            rlp_hash: keccak256(node_bytes),
            kind: classify_node(node_bytes),
        });
    }

    for (hashed_address, storage_proof) in &proof.storages {
        for (path, node_bytes) in storage_proof.subtree.iter() {
            nodes.push(ProofNodeRecord {
                domain: ProofNodeDomain::Storage(*hashed_address),
                path: path.to_vec(),
                rlp: node_bytes.clone(),
                rlp_hash: keccak256(node_bytes),
                kind: classify_node(node_bytes),
            });
        }
    }

    nodes
}

/// Result of witness computation for a single block.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WitnessResult {
    /// Total size of witness in bytes (account trie nodes + storage trie nodes + bytecode bytes).
    pub total_size_bytes: usize,
    /// Size of account proof nodes in bytes.
    pub account_proof_bytes: usize,
    /// Size of storage proof nodes in bytes.
    pub storage_proof_bytes: usize,
    /// Size of missed contract bytecodes in bytes.
    pub bytecode_bytes: usize,
    /// Number of account proof trie nodes.
    pub account_proof_nodes: usize,
    /// Number of storage proof trie nodes (across all accounts).
    pub storage_proof_nodes: usize,
    /// Number of unique accounts in the proof targets.
    pub target_accounts: usize,
    /// Number of unique storage slots in the proof targets.
    pub target_storage_slots: usize,
    /// Time taken to compute the multiproof (if measured).
    pub computation_time_ms: Option<u64>,
}

/// Convert a `MissResult` into `MultiProofTargets` suitable for `StateProofProvider::multiproof()`.
///
/// This hashes addresses and storage slots with keccak256, which is what reth's
/// trie infrastructure expects.
pub fn miss_to_proof_targets(miss: &MissResult) -> MultiProofTargets {
    let mut targets = MultiProofTargets::with_capacity(miss.missed_accounts.len());

    // Add all missed accounts (even those without storage misses)
    for address in &miss.missed_accounts {
        let hashed_address = keccak256(address);
        targets.entry(hashed_address).or_default();
    }

    // Add missed storage slots grouped by account
    for (address, slot) in &miss.missed_storage {
        let hashed_address = keccak256(address);
        let hashed_slot = keccak256(slot);
        targets.entry(hashed_address).or_default().insert(hashed_slot);
    }

    targets
}

/// Convert the complete block accessed universe into benchmark target metadata.
pub fn accessed_to_state_targets(accessed: &BlockAccessedState) -> StateTargetSet {
    let mut targets = StateTargetSet {
        accounts: accessed.accounts.keys().copied().collect(),
        storage: accessed.storage.keys().copied().collect(),
        code_hashes: accessed.codes.keys().copied().collect(),
    };
    targets.sort_dedup();
    targets
}

/// Compute the cache-hit side of `accessed == cache_hit ∪ sidecar_miss`.
pub fn cache_hit_targets(accessed: &BlockAccessedState, miss: &MissResult) -> StateTargetSet {
    let missed_accounts: HashSet<Address> = miss.missed_accounts.iter().copied().collect();
    let missed_storage: HashSet<(Address, B256)> = miss.missed_storage.iter().copied().collect();
    let missed_codes: HashSet<B256> = miss.missed_codes.iter().copied().collect();

    let mut targets = StateTargetSet {
        accounts: accessed
            .accounts
            .keys()
            .filter(|address| !missed_accounts.contains(*address))
            .copied()
            .collect(),
        storage: accessed
            .storage
            .keys()
            .filter(|key| !missed_storage.contains(*key))
            .copied()
            .collect(),
        code_hashes: accessed
            .codes
            .keys()
            .filter(|code_hash| !missed_codes.contains(*code_hash))
            .copied()
            .collect(),
    };
    targets.sort_dedup();
    targets
}

/// Convert raw state targets into hashed `MultiProofTargets`.
pub fn state_targets_to_proof_targets(targets: &StateTargetSet) -> MultiProofTargets {
    let mut multiproof_targets = MultiProofTargets::with_capacity(targets.accounts.len());

    for address in &targets.accounts {
        let hashed_address = keccak256(address);
        multiproof_targets.entry(hashed_address).or_default();
    }

    for (address, slot) in &targets.storage {
        let hashed_address = keccak256(address);
        let hashed_slot = keccak256(slot);
        multiproof_targets.entry(hashed_address).or_default().insert(hashed_slot);
    }

    multiproof_targets
}

/// Measure the total byte size of a `MultiProof`, adding the size of any missed bytecodes.
///
/// This counts the raw bytes of all trie nodes in the proof (account + storage subtrees)
/// and sums them with the provided bytecode bytes.
pub fn measure_multiproof_size(proof: &MultiProof, missed_bytecode_bytes: usize) -> WitnessResult {
    // Account proof size
    let mut account_proof_bytes = 0usize;
    let mut account_proof_nodes = 0usize;
    for node_bytes in proof.account_subtree.values() {
        account_proof_bytes += node_bytes.len();
        account_proof_nodes += 1;
    }

    // Storage proof sizes
    let mut storage_proof_bytes = 0usize;
    let mut storage_proof_nodes = 0usize;
    for storage_mp in proof.storages.values() {
        for node_bytes in storage_mp.subtree.values() {
            storage_proof_bytes += node_bytes.len();
            storage_proof_nodes += 1;
        }
    }

    let total_size_bytes = account_proof_bytes + storage_proof_bytes + missed_bytecode_bytes;

    // Count targets
    let target_accounts = proof.storages.len().max(
        // account_subtree doesn't directly tell us account count,
        // but storages map has one entry per targeted account
        proof.storages.len(),
    );
    let target_storage_slots: usize = proof
        .storages
        .values()
        .map(|s| {
            // Estimate slots from number of leaf nodes in storage proof
            // (this is approximate, actual slot count comes from targets)
            s.subtree.len()
        })
        .sum();

    WitnessResult {
        total_size_bytes,
        account_proof_bytes,
        storage_proof_bytes,
        bytecode_bytes: missed_bytecode_bytes,
        account_proof_nodes,
        storage_proof_nodes,
        target_accounts,
        target_storage_slots,
        computation_time_ms: None,
    }
}

/// Measure witness result from a `MissResult` and targets (before proof computation).
/// This provides target counts without actual proof size.
pub fn witness_targets_summary(miss: &MissResult) -> WitnessTargetsSummary {
    // Group storage misses by account
    let mut storage_by_account: std::collections::HashMap<Address, usize> =
        std::collections::HashMap::new();
    for (address, _slot) in &miss.missed_storage {
        *storage_by_account.entry(*address).or_default() += 1;
    }

    WitnessTargetsSummary {
        missed_accounts: miss.missed_accounts.len(),
        missed_storage_slots: miss.missed_storage.len(),
        missed_codes: miss.missed_codes.len(),
        accounts_with_storage: storage_by_account.len(),
        max_slots_per_account: storage_by_account.values().copied().max().unwrap_or(0),
    }
}

/// Summary of witness targets (before proof computation).
#[derive(Debug, Clone)]
pub struct WitnessTargetsSummary {
    /// Number of accounts that need witness.
    pub missed_accounts: usize,
    /// Number of storage slots that need witness.
    pub missed_storage_slots: usize,
    /// Number of code entries that need witness (codes are not part of trie proof).
    pub missed_codes: usize,
    /// Number of unique accounts that have storage misses.
    pub accounts_with_storage: usize,
    /// Maximum number of missed slots for a single account.
    pub max_slots_per_account: usize,
}

/// Builds raw `WitnessTargets` (for Sidecar data payload) and hashed `MultiProofTargets` (for Trie
/// Provider) in a single pass from `MissResult`.
pub fn build_sidecar_targets(miss: &MissResult) -> (WitnessTargets, MultiProofTargets) {
    let mut multiproof_targets = MultiProofTargets::with_capacity(miss.missed_accounts.len());

    // 1. Convert missed accounts to WitnessTargets & hashed multiproof targets
    let missed_accounts = miss.missed_accounts.clone();
    for address in &missed_accounts {
        let hashed_address = keccak256(address);
        multiproof_targets.entry(hashed_address).or_default();
    }

    // 2. Convert missed storage to WitnessTargets & hashed multiproof targets
    let missed_storage = miss.missed_storage.clone();
    for (address, slot) in &missed_storage {
        let hashed_address = keccak256(address);
        let hashed_slot = keccak256(slot);
        multiproof_targets.entry(hashed_address).or_default().insert(hashed_slot);
    }

    // 3. Convert missed codes to WitnessTargets
    let missed_code_hashes = miss.missed_codes.clone();

    let raw_targets = WitnessTargets { missed_accounts, missed_storage, missed_code_hashes };

    (raw_targets, multiproof_targets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::map::{B256Map, HashMap};
    use alloy_rlp::Encodable;
    use reth_trie_common::{
        proof::ProofNodes, BranchNodeMasksMap, ExtensionNode, Nibbles, RlpNode, StorageMultiProof,
    };

    fn branch_rlp() -> Bytes {
        let mut bytes = vec![0xd1];
        bytes.extend([0x80; 17]);
        Bytes::from(bytes)
    }

    fn extension_rlp() -> Bytes {
        let child = RlpNode::word_rlp(&B256::repeat_byte(0x11));
        let node =
            TrieNode::Extension(ExtensionNode::new(Nibbles::from_nibbles([0x1, 0x2]), child));
        let mut buf = Vec::new();
        node.encode(&mut buf);
        Bytes::from(buf)
    }

    fn proof_with_nodes() -> MultiProof {
        let branch = branch_rlp();
        let leaf_like = Bytes::from(vec![0xc0]);
        let account_path = Nibbles::from_nibbles(vec![0x01]);
        let storage_path = Nibbles::from_nibbles(vec![0x02]);
        let hashed_account_a = B256::repeat_byte(0xaa);
        let hashed_account_b = B256::repeat_byte(0xbb);

        let mut account_subtree_map: HashMap<Nibbles, Bytes> = HashMap::default();
        account_subtree_map.insert(account_path, branch.clone());
        let account_subtree = ProofNodes::from_iter(account_subtree_map);

        let mut storage_subtree_a: HashMap<Nibbles, Bytes> = HashMap::default();
        storage_subtree_a.insert(storage_path, branch.clone());
        let mut storage_subtree_b: HashMap<Nibbles, Bytes> = HashMap::default();
        storage_subtree_b.insert(storage_path, leaf_like);

        let mut storages = B256Map::default();
        storages.insert(
            hashed_account_a,
            StorageMultiProof {
                root: B256::ZERO,
                subtree: ProofNodes::from_iter(storage_subtree_a),
                branch_node_masks: BranchNodeMasksMap::default(),
            },
        );
        storages.insert(
            hashed_account_b,
            StorageMultiProof {
                root: B256::ZERO,
                subtree: ProofNodes::from_iter(storage_subtree_b),
                branch_node_masks: BranchNodeMasksMap::default(),
            },
        );

        MultiProof { account_subtree, branch_node_masks: BranchNodeMasksMap::default(), storages }
    }

    #[test]
    fn flatten_multiproof_nodes_keeps_domains_distinct() {
        let nodes = flatten_multiproof_nodes(&proof_with_nodes());
        assert_eq!(nodes.len(), 3);
        assert!(nodes.iter().any(|node| matches!(node.domain, ProofNodeDomain::Account)));
        assert!(nodes.iter().any(|node| matches!(node.domain, ProofNodeDomain::Storage(addr) if addr == B256::repeat_byte(0xaa))));
        assert!(nodes.iter().any(|node| matches!(node.domain, ProofNodeDomain::Storage(addr) if addr == B256::repeat_byte(0xbb))));
    }

    #[test]
    fn flatten_multiproof_nodes_classifies_branch_nodes() {
        let nodes = flatten_multiproof_nodes(&proof_with_nodes());
        assert_eq!(nodes.iter().filter(|node| node.is_branch()).count(), 2);
        assert_eq!(nodes.iter().filter(|node| !node.is_branch()).count(), 1);
    }

    #[test]
    fn flatten_multiproof_nodes_classifies_extension_nodes() {
        let mut account_map: HashMap<Nibbles, Bytes> = HashMap::default();
        account_map.insert(Nibbles::from_nibbles(vec![0x03]), extension_rlp());
        account_map.insert(Nibbles::from_nibbles(vec![0x04]), branch_rlp());
        let proof = MultiProof {
            account_subtree: ProofNodes::from_iter(account_map),
            branch_node_masks: BranchNodeMasksMap::default(),
            storages: B256Map::default(),
        };

        let nodes = flatten_multiproof_nodes(&proof);
        assert_eq!(nodes.iter().filter(|node| node.is_extension()).count(), 1);
        assert_eq!(nodes.iter().filter(|node| node.is_branch()).count(), 1);
        assert!(nodes.iter().all(|node| node.is_structural()));
    }

    #[test]
    fn proof_node_record_identity_includes_domain_path_and_bytes() {
        let nodes = flatten_multiproof_nodes(&proof_with_nodes());
        let account =
            nodes.iter().find(|node| matches!(node.domain, ProofNodeDomain::Account)).unwrap();
        let storage = nodes.iter().find(|node| matches!(node.domain, ProofNodeDomain::Storage(addr) if addr == B256::repeat_byte(0xaa))).unwrap();
        assert_ne!(account, storage, "same RLP bytes in different domains must not collide");
    }
}
