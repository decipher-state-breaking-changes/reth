//! The trimmed (v3) sidecar witness: receiver-aware fragments instead of root-connected proofs.
//!
//! A v2 flat witness is decodable from the parent state root alone, which forces every miss and
//! mutation proof to re-ship the upper trie nodes the receiving validator already holds revealed
//! and authenticated. The v3 wire drops exactly those nodes: the builder walks each proof target
//! through its own committed parent trie first and ships only what lies **below the blinded
//! frontier** — the first branch child on the key's path whose content the trie does not hold.
//! The receiving validator, holding a byte-identical trie by the deterministic-retention
//! contract, runs the same walk and re-attaches ("grafts") each fragment at the same frontier.
//!
//! Both sides run the *same* composite walk in this module — local revealed nodes take priority,
//! the witness map is consulted only from a blinded frontier hash downward — so the set of nodes
//! the builder ships is by construction the set the validator consumes. That symmetry is what
//! makes the v3 canonicality rules checkable: a duplicate node, a node missing while a chain
//! needs it, a sub-32-byte entry (nothing on the wire can address one: sub-32 nodes are embedded
//! in their parents, and no real account or storage root encodes below 32 bytes), and any node
//! left unconsumed after the transition all reject the sidecar.
//!
//! Content authentication is by construction rather than by a separate check: the node map is
//! keyed by `keccak256` of each wire entry, so looking an entry up by the blinded frontier hash
//! *is* the `keccak(RLP) == anchor` verification — a wrong-content node can only surface as a
//! missing hash, never as accepted content.
//!
//! Everything here is read-only against the parent trie cache. The walks mutate only the
//! consumption ledger inside [`WitnessNodeMap`]; grafting into a trie happens through the
//! ordinary reveal path with the fragments this module collects.

use crate::{transition_build::V2TargetSet, trie_cache::PartialTrieNodeCache};
use alloy_primitives::{
    keccak256,
    map::{B256Map, HashSet},
    Bytes, B256,
};
use alloy_rlp::Decodable;

use reth_trie_common::{
    DecodedMultiProofV2, Nibbles, ProofTrieNodeV2, TrieAccount, TrieNode, EMPTY_ROOT_HASH,
};
use reth_trie_sparse::{LeafLookup, LeafLookupError, SparseTrie};

/// The frontier protocol version a trimmed witness assumes on the receiver.
///
/// This is protocol surface, and it is wider than the retention algorithm alone: the blinded
/// frontier the fragments anchor on — and the graft that re-attaches them — depends on the
/// deterministic retention rules (`PartialTrieNodeCacheRetention/v1`), the sparse-trie
/// representation's observable shape, extension normalization (the merged extension+branch form
/// and the divergent-extension one-level resolution), and the reveal/fold/composite-walk rules
/// in this module. Neither `cache_root` nor the retention fingerprint commits any of that, so a
/// builder and validator disagreeing on this number must not attempt a graft.
///
/// Bump it for **any** change that can alter the revealed frontier shape or graft semantics.
/// A change that intends to preserve them — a trie-representation swap, say — keeps the number
/// only when a cross-representation oracle has shown the fragment sets and anchors identical;
/// absent that evidence, bump.
pub const WITNESS_V3_FRONTIER_VERSION: u16 = 1;

/// A trimmed-witness node map with a consumption ledger.
///
/// Entries are keyed by `keccak256` of the wire bytes, so a lookup by a blinded frontier hash is
/// itself the content check. Every successful lookup is recorded; after the transition, entries
/// never consumed by any walk reject the sidecar (see [`WitnessNodeMap::ensure_fully_consumed`]).
#[derive(Debug, Clone)]
pub struct WitnessNodeMap {
    nodes: B256Map<Bytes>,
    consumed: HashSet<B256>,
}

impl WitnessNodeMap {
    /// Decodes a v3 wire node list, enforcing the canonical wire form: entries in strictly
    /// ascending byte order (which makes duplicates impossible), none shorter than 32 bytes.
    pub fn decode(nodes: &[Bytes]) -> Result<Self, WitnessV3Error> {
        let mut map = B256Map::with_capacity_and_hasher(nodes.len(), Default::default());
        for (index, node) in nodes.iter().enumerate() {
            if node.len() < 32 {
                return Err(WitnessV3Error::StandaloneInline { len: node.len() });
            }
            if let Some(previous) = index.checked_sub(1).map(|prev| &nodes[prev]) {
                // Strictly ascending: equality is a duplicate, descent is a re-ordering. Both
                // reject — there is exactly one canonical encoding of a fragment set, so a
                // builder never produces either and a sidecar carrying one is non-canonical
                // even when its commitment was recomputed to match.
                if previous >= node {
                    if previous == node {
                        return Err(WitnessV3Error::DuplicateNode(keccak256(node)));
                    }
                    return Err(WitnessV3Error::UnsortedNodes { index });
                }
            }
            let hash = keccak256(node);
            map.insert(hash, node.clone());
        }
        Ok(Self { nodes: map, consumed: HashSet::default() })
    }

    /// Wraps a builder-side full flat map (already hash-keyed and duplicate-free) so the wire
    /// walk can record which entries a validator would consume.
    pub fn from_flat(nodes: B256Map<Bytes>) -> Self {
        Self { nodes, consumed: HashSet::default() }
    }

    /// Looks an entry up by hash, recording the consumption. `None` means the wire cannot
    /// satisfy the chain — a rejection on the validator, a construction bug on the builder.
    fn get_consuming(&mut self, hash: &B256) -> Option<Bytes> {
        let bytes = self.nodes.get(hash)?.clone();
        self.consumed.insert(*hash);
        Some(bytes)
    }

    /// The consumed entries, byte-sorted — the builder's v3 wire node list.
    pub fn consumed_nodes_sorted(&self) -> Vec<Bytes> {
        let mut nodes: Vec<Bytes> =
            self.consumed.iter().filter_map(|hash| self.nodes.get(hash).cloned()).collect();
        nodes.sort_unstable();
        nodes
    }

    /// Total entries in the map.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// True when the map holds no entries.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Entries never consumed by any walk.
    pub fn unconsumed(&self) -> usize {
        self.nodes.len() - self.consumed.len()
    }

    /// Rejects when any entry was never consumed: an unconsumed node is unaddressable content a
    /// canonical builder would never have shipped.
    pub fn ensure_fully_consumed(&self) -> Result<(), WitnessV3Error> {
        let count = self.unconsumed();
        if count == 0 {
            return Ok(());
        }
        let first = self
            .nodes
            .keys()
            .find(|hash| !self.consumed.contains(*hash))
            .copied()
            .unwrap_or_default();
        Err(WitnessV3Error::UnconsumedNodes { count, first })
    }
}

/// Outcome of walking one key to its terminal through the composite (local trie, witness map)
/// view of the parent state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainValue {
    /// The leaf exists; its RLP-encoded value.
    Present(Vec<u8>),
    /// The key provably does not exist (exclusion shown locally or by a divergent witness node).
    Absent,
}

/// Rejection and construction errors for the trimmed witness.
#[derive(Debug, thiserror::Error)]
pub enum WitnessV3Error {
    /// Two wire entries hashed identically.
    #[error("trimmed witness contains duplicate node {0}")]
    DuplicateNode(B256),
    /// A wire entry shorter than 32 bytes: nothing can address it by hash.
    #[error("trimmed witness contains a standalone {len}-byte entry; sub-32-byte nodes are embedded in their parents and must not appear on the wire")]
    StandaloneInline {
        /// Encoded length of the offending entry.
        len: usize,
    },
    /// The local trie cache has no revealed parent state to graft against.
    #[error("trimmed witness requires a revealed local trie cache anchored to the parent state")]
    CacheNotReady,
    /// A chain needed a node the wire does not carry (or carries with different content, which
    /// is the same thing: the map is keyed by content hash).
    #[error("witness node {hash} needed at path {path:?} is missing from the trimmed witness")]
    MissingNode {
        /// The blinded hash the chain had to resolve.
        hash: B256,
        /// The path the node was needed at.
        path: Nibbles,
    },
    /// A wire entry did not decode as a trie node.
    #[error("failed to decode witness node {hash}: {source}")]
    NodeDecode {
        /// Hash of the undecodable entry.
        hash: B256,
        /// The RLP error.
        source: alloy_rlp::Error,
    },
    /// A chain terminal value did not decode.
    #[error("failed to decode a value on a witness chain: {0}")]
    ValueDecode(String),
    /// Entries the graft and value walks never touched.
    #[error("{count} trimmed-witness nodes were never consumed by any walk (first: {first})")]
    UnconsumedNodes {
        /// How many entries were left over.
        count: usize,
        /// One of them, for the log line.
        first: B256,
    },
    /// A wire entry out of the canonical strictly-ascending byte order.
    #[error("trimmed witness node at index {index} is not in strictly ascending byte order")]
    UnsortedNodes {
        /// Index of the first out-of-order entry.
        index: usize,
    },
    /// The sidecar names a frontier protocol version this build does not implement.
    #[error("trimmed witness names frontier version {got}, local implementation is {expected}")]
    FrontierVersionMismatch {
        /// Version named by the sidecar.
        got: u16,
        /// Version this build implements.
        expected: u16,
    },
    /// The sidecar was trimmed against a different retained-path derivation than the local one.
    #[error("trimmed witness was cut against retention fingerprint {got}, local fingerprint is {expected}")]
    RetentionFingerprintMismatch {
        /// Fingerprint named by the sidecar.
        got: B256,
        /// The local parent generation's fingerprint.
        expected: B256,
    },
    /// The local sparse trie behaved in a way the composite walk cannot interpret.
    #[error("local trie walk failed: {0}")]
    LocalTrie(String),
}

/// Node sinks for one composite walk: `None` consumes only, `Some` also collects the
/// map-resolved nodes (path-attributed) for a later reveal.
type NodeSink<'s> = Option<&'s mut Vec<(Nibbles, TrieNode)>>;

/// Walks `full_key` from a blinded frontier `(seed_path, seed_hash)` down through the witness
/// map, consuming every node on the chain.
///
/// The chain follows the key: at a branch the next nibble selects one child, at an extension the
/// key must match the prefix or the walk ends as a proven exclusion, at a leaf the remaining key
/// either matches (inclusion) or not (exclusion). Inline (sub-32-byte) children are decoded from
/// the bytes embedded in their parent, exactly as the v2 root walk does.
fn walk_map_chain(
    map: &mut WitnessNodeMap,
    seed_path: Nibbles,
    seed_hash: B256,
    full_key: &Nibbles,
    mut out: NodeSink<'_>,
) -> Result<ChainValue, WitnessV3Error> {
    enum Pending {
        Hashed(B256),
        Inline(Vec<u8>),
    }

    let mut path = seed_path;
    let mut pending = Pending::Hashed(seed_hash);
    loop {
        let node = match pending {
            Pending::Hashed(hash) => {
                let Some(bytes) = map.get_consuming(&hash) else {
                    return Err(WitnessV3Error::MissingNode { hash, path });
                };
                TrieNode::decode(&mut bytes.as_ref())
                    .map_err(|source| WitnessV3Error::NodeDecode { hash, source })?
            }
            Pending::Inline(bytes) => TrieNode::decode(&mut bytes.as_slice())
                .map_err(|source| WitnessV3Error::NodeDecode { hash: keccak256(&bytes), source })?,
        };
        if let Some(sink) = out.as_deref_mut() {
            sink.push((path, node.clone()));
        }

        match node {
            TrieNode::Branch(branch) => {
                debug_assert!(path.len() < full_key.len(), "branch at or below full key depth");
                let nibble = full_key.get_unchecked(path.len());
                let branch_ref = branch.as_ref();
                let mut selected = None;
                for (idx, maybe_child) in branch_ref.children() {
                    if idx == nibble {
                        selected = maybe_child;
                    }
                }
                let Some(child) = selected else { return Ok(ChainValue::Absent) };
                path.push_unchecked(nibble);
                pending = match child.as_hash() {
                    Some(hash) => Pending::Hashed(hash),
                    None => Pending::Inline(child.as_slice().to_vec()),
                };
            }
            TrieNode::Extension(ext) => {
                let mut child_path = path;
                child_path.extend(&ext.key);
                let diverged = !full_key.starts_with(&child_path);
                path = child_path;
                pending = match ext.child.as_hash() {
                    Some(hash) => Pending::Hashed(hash),
                    None => Pending::Inline(ext.child.as_slice().to_vec()),
                };
                if diverged {
                    // Exclusion at the extension. The chain still resolves the extension's own
                    // child: a bare extension cannot be revealed into a sparse trie (only
                    // branch children carry blinded hashes), which is exactly why the v2 wire
                    // ships an extension and its child branch as one merged node. Consuming
                    // one level here keeps the trimmed wire graftable and byte-consistent with
                    // what the builder's own transition revealed.
                    let child = match pending {
                        Pending::Hashed(hash) => {
                            let Some(bytes) = map.get_consuming(&hash) else {
                                return Err(WitnessV3Error::MissingNode { hash, path });
                            };
                            TrieNode::decode(&mut bytes.as_ref())
                                .map_err(|source| WitnessV3Error::NodeDecode { hash, source })?
                        }
                        Pending::Inline(bytes) => {
                            TrieNode::decode(&mut bytes.as_slice()).map_err(|source| {
                                WitnessV3Error::NodeDecode { hash: keccak256(&bytes), source }
                            })?
                        }
                    };
                    if let Some(sink) = out.as_deref_mut() {
                        sink.push((path, child));
                    }
                    return Ok(ChainValue::Absent);
                }
            }
            TrieNode::Leaf(leaf) => {
                let mut leaf_full = path;
                leaf_full.extend(&leaf.key);
                return Ok(if &leaf_full == full_key {
                    ChainValue::Present(leaf.value)
                } else {
                    ChainValue::Absent
                });
            }
            TrieNode::EmptyRoot => return Ok(ChainValue::Absent),
        }
    }
}

/// Walks one hashed account key through the composite parent view: the local account trie
/// first, the witness map only from the blinded frontier down.
pub(crate) fn composite_account_chain(
    parent: &PartialTrieNodeCache,
    map: &mut WitnessNodeMap,
    hashed_address: B256,
    out: NodeSink<'_>,
) -> Result<ChainValue, WitnessV3Error> {
    let sparse = parent.sparse_ref();
    let Some(state) = sparse.state_trie_ref() else { return Err(WitnessV3Error::CacheNotReady) };
    let key = Nibbles::unpack(hashed_address);
    match state.find_leaf(&key, None) {
        Ok(LeafLookup::Exists) => {
            let value = sparse.get_account_value(&hashed_address).cloned().ok_or_else(|| {
                WitnessV3Error::LocalTrie(format!(
                    "account {hashed_address} found but its leaf value is unavailable"
                ))
            })?;
            Ok(ChainValue::Present(value))
        }
        Ok(LeafLookup::NonExistent) => Ok(ChainValue::Absent),
        Err(LeafLookupError::BlindedNode { path, hash }) => {
            walk_map_chain(map, path, hash, &key, out)
        }
        Err(err @ LeafLookupError::ValueMismatch { .. }) => {
            Err(WitnessV3Error::LocalTrie(format!("unexpected local lookup failure: {err:?}")))
        }
    }
}

/// Walks one hashed storage key through the composite parent view.
///
/// When the local cache has no revealed storage trie for the account, the owner's account chain
/// is walked (consume-only) to learn the authenticated storage root, and the storage chain then
/// starts from that root hash. Those owner-chain nodes are never part of a storage reveal —
/// mirroring the builder, whose structural reveals carry storage nodes only.
pub(crate) fn composite_storage_chain(
    parent: &PartialTrieNodeCache,
    map: &mut WitnessNodeMap,
    hashed_address: B256,
    hashed_slot: B256,
    out: NodeSink<'_>,
) -> Result<ChainValue, WitnessV3Error> {
    let sparse = parent.sparse_ref();
    let key = Nibbles::unpack(hashed_slot);
    if let Some(storage) = sparse.storage_trie_ref(&hashed_address) {
        return match storage.find_leaf(&key, None) {
            Ok(LeafLookup::Exists) => {
                let value = sparse
                    .get_storage_slot_value(&hashed_address, &hashed_slot)
                    .cloned()
                    .ok_or_else(|| {
                    WitnessV3Error::LocalTrie(format!(
                        "slot {hashed_slot} of {hashed_address} found but its value is unavailable"
                    ))
                })?;
                Ok(ChainValue::Present(value))
            }
            Ok(LeafLookup::NonExistent) => Ok(ChainValue::Absent),
            Err(LeafLookupError::BlindedNode { path, hash }) => {
                walk_map_chain(map, path, hash, &key, out)
            }
            Err(err @ LeafLookupError::ValueMismatch { .. }) => {
                Err(WitnessV3Error::LocalTrie(format!("unexpected local lookup failure: {err:?}")))
            }
        };
    }

    // No local storage trie: learn the storage root from the owner's account chain.
    let storage_root = match composite_account_chain(parent, map, hashed_address, None)? {
        ChainValue::Present(account_rlp) => {
            TrieAccount::decode(&mut account_rlp.as_slice())
                .map_err(|err| {
                    WitnessV3Error::ValueDecode(format!(
                        "account {hashed_address} on a storage chain: {err}"
                    ))
                })?
                .storage_root
        }
        // A nonexistent account has no storage.
        ChainValue::Absent => return Ok(ChainValue::Absent),
    };
    if storage_root == EMPTY_ROOT_HASH {
        return Ok(ChainValue::Absent);
    }
    walk_map_chain(map, Nibbles::default(), storage_root, &key, out)
}

/// Consumes the chains of every target in `targets` without collecting nodes.
///
/// The builder runs this over everything it ever requested to derive the wire node set; the
/// validator runs it over the flat-context targets whose chains are needed for reachability but
/// are never revealed. Both directions must call the same code: the consumed set defines the
/// wire, and the wire defines what the unconsumed-node rule accepts.
pub(crate) fn consume_target_chains(
    parent: &PartialTrieNodeCache,
    map: &mut WitnessNodeMap,
    targets: &V2TargetSet,
) -> Result<(), WitnessV3Error> {
    for hashed_address in targets.account_keys() {
        composite_account_chain(parent, map, hashed_address, None)?;
    }
    for (hashed_address, hashed_slot) in targets.storage_keys() {
        composite_storage_chain(parent, map, hashed_address, hashed_slot, None)?;
    }
    Ok(())
}

/// Collects the reveal fragments for `targets`: every map-resolved node on their chains,
/// path-attributed and folded into a [`DecodedMultiProofV2`] ready for the ordinary reveal path.
///
/// Locally resolved targets contribute nothing — their nodes are already revealed in the trie
/// the fragments graft into.
pub(crate) fn collect_reveal_fragments(
    parent: &PartialTrieNodeCache,
    map: &mut WitnessNodeMap,
    targets: &V2TargetSet,
) -> Result<DecodedMultiProofV2, WitnessV3Error> {
    let mut account_nodes: Vec<(Nibbles, TrieNode)> = Vec::new();
    for hashed_address in targets.account_keys() {
        composite_account_chain(parent, map, hashed_address, Some(&mut account_nodes))?;
    }
    let mut storage_nodes: B256Map<Vec<(Nibbles, TrieNode)>> = B256Map::default();
    for (hashed_address, hashed_slot) in targets.storage_keys() {
        composite_storage_chain(
            parent,
            map,
            hashed_address,
            hashed_slot,
            Some(storage_nodes.entry(hashed_address).or_default()),
        )?;
    }

    let account_proofs = fold_fragment_nodes(account_nodes);
    let mut storage_proofs = B256Map::default();
    for (hashed_address, nodes) in storage_nodes {
        let folded = fold_fragment_nodes(nodes);
        if !folded.is_empty() {
            storage_proofs.insert(hashed_address, folded);
        }
    }
    Ok(DecodedMultiProofV2 { account_proofs, storage_proofs })
}

/// Sorts, deduplicates, and folds chain nodes into V2 proof nodes.
///
/// Chains of neighbouring targets overlap below a shared frontier, so the same `(path, node)`
/// pair can be collected more than once; content addressing makes equal paths carry equal nodes,
/// and the fold input must be duplicate-free and sorted children-first.
fn fold_fragment_nodes(mut nodes: Vec<(Nibbles, TrieNode)>) -> Vec<ProofTrieNodeV2> {
    nodes.sort_by(|(a, _), (b, _)| b.cmp(a));
    nodes.dedup_by(|(a, _), (b, _)| a == b);
    ProofTrieNodeV2::from_sorted_trie_nodes(
        nodes.into_iter().map(|(path, node)| (path, node, None)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::U256;
    use reth_trie_common::{proof::ProofRetainer, HashBuilder};

    /// Builds a storage-shaped trie whose root is an extension: two 64-nibble keys sharing a
    /// three-nibble prefix, so the root is `ext(0x111) -> branch{1, 2}`.
    fn extension_rooted_trie() -> (B256, B256Map<Bytes>, Nibbles, Nibbles) {
        let mut key_a = Nibbles::default();
        let mut key_b = Nibbles::default();
        for i in 0..64u8 {
            match i {
                0..=2 => {
                    key_a.push_unchecked(1);
                    key_b.push_unchecked(1);
                }
                3 => {
                    key_a.push_unchecked(1);
                    key_b.push_unchecked(2);
                }
                _ => {
                    key_a.push_unchecked(0xa);
                    key_b.push_unchecked(0xb);
                }
            }
        }
        let value = alloy_rlp::encode(U256::from(7u64));
        let mut leaves = vec![(key_a, value.clone()), (key_b, value)];
        leaves.sort_unstable_by_key(|(key, _)| *key);
        let mut builder = HashBuilder::default()
            .with_proof_retainer(ProofRetainer::from_iter(leaves.iter().map(|(key, _)| *key)));
        for (key, value) in &leaves {
            builder.add_leaf(*key, value);
        }
        let root = builder.root();
        let mut witness = B256Map::default();
        for (_, node) in builder.take_proof_nodes().into_inner() {
            witness.insert(keccak256(&node), Bytes::from(node.to_vec()));
        }
        (root, witness, key_a, key_b)
    }

    /// The regression the first real corpus block caught: a key that diverges *inside* an
    /// extension's key must still resolve the extension's child, because a bare extension
    /// cannot be revealed into a sparse trie — only branch children carry blinded hashes. The
    /// v2 wire ships the extension and its child branch as one merged node for the same reason.
    #[test]
    fn a_divergent_extension_still_resolves_its_child_branch() {
        let (root, witness, key_a, _) = extension_rooted_trie();

        // A key sharing only the first nibble: diverges at the extension's second key nibble.
        let mut absent_key = Nibbles::default();
        absent_key.push_unchecked(1);
        absent_key.push_unchecked(9);
        for _ in 2..64 {
            absent_key.push_unchecked(0);
        }

        let mut map = WitnessNodeMap::from_flat(witness.clone());
        let mut nodes = Vec::new();
        let outcome =
            walk_map_chain(&mut map, Nibbles::default(), root, &absent_key, Some(&mut nodes))
                .expect("the exclusion chain resolves");
        assert_eq!(outcome, ChainValue::Absent);

        // Both the extension and its child branch were collected and consumed, and the fold
        // produces a single revealable merged node rather than dropping a dangling extension.
        assert_eq!(nodes.len(), 2, "extension and child branch: {nodes:?}");
        assert!(matches!(nodes[0].1, TrieNode::Extension(_)));
        assert!(matches!(nodes[1].1, TrieNode::Branch(_)));
        let folded = fold_fragment_nodes(nodes);
        assert_eq!(folded.len(), 1, "the fold merges the pair into one branch-with-key node");

        // An included key still walks to its leaf.
        let mut map = WitnessNodeMap::from_flat(witness);
        let outcome = walk_map_chain(&mut map, Nibbles::default(), root, &key_a, None)
            .expect("the inclusion chain resolves");
        assert!(matches!(outcome, ChainValue::Present(_)));
    }
    /// A chain through inline (sub-32-byte) children: two keys sharing 63 nibbles produce a
    /// deep branch whose leaves — and possibly the branch itself — encode below 32 bytes and are
    /// embedded in their parents rather than hash-addressed. The walk must decode them from the
    /// embedded bytes, never look them up in the map, and still land on the value; the wire
    /// carries no standalone entry for any of them.
    #[test]
    fn an_inline_child_chain_resolves_without_standalone_wire_entries() {
        let mut key_a = Nibbles::default();
        let mut key_b = Nibbles::default();
        for i in 0..64u8 {
            if i < 63 {
                key_a.push_unchecked((i % 15) + 1);
                key_b.push_unchecked((i % 15) + 1);
            } else {
                key_a.push_unchecked(0x4);
                key_b.push_unchecked(0x9);
            }
        }
        let value = alloy_rlp::encode(U256::from(3u64));
        let mut leaves = vec![(key_a, value.clone()), (key_b, value)];
        leaves.sort_unstable_by_key(|(key, _)| *key);
        let mut builder = HashBuilder::default()
            .with_proof_retainer(ProofRetainer::from_iter(leaves.iter().map(|(key, _)| *key)));
        for (key, value) in &leaves {
            builder.add_leaf(*key, value);
        }
        let root = builder.root();
        let mut witness = B256Map::default();
        let mut sub32 = 0usize;
        for (_, node) in builder.take_proof_nodes().into_inner() {
            // The retainer only yields hash-addressed nodes; anything embedded stays inside its
            // parent's bytes. Count what would have been a standalone inline entry to prove the
            // fixture actually exercises embedding.
            if node.len() < 32 {
                sub32 += 1;
                continue;
            }
            witness.insert(keccak256(&node), Bytes::from(node.to_vec()));
        }
        assert!(
            witness.values().any(|node| {
                let decoded = TrieNode::decode(&mut node.as_ref()).expect("wire nodes decode");
                match decoded {
                    TrieNode::Extension(ext) => ext.child.as_hash().is_none(),
                    TrieNode::Branch(branch) => branch
                        .as_ref()
                        .children()
                        .any(|(_, child)| child.is_some_and(|child| child.as_hash().is_none())),
                    _ => false,
                }
            }) || sub32 > 0,
            "the fixture produced no embedded child at all"
        );

        let mut map = WitnessNodeMap::from_flat(witness);
        let mut nodes = Vec::new();
        let outcome = walk_map_chain(&mut map, Nibbles::default(), root, &key_a, Some(&mut nodes))
            .expect("the inline chain resolves");
        assert!(matches!(outcome, ChainValue::Present(_)));
        // The inline nodes were collected for the reveal even though nothing on the wire
        // addresses them.
        assert!(nodes.len() >= 2, "the chain has at least the root and the deep structure");
        // Absent key through the same deep structure: divergence at the final nibble, which no
        // leaf uses.
        let mut absent = Nibbles::default();
        for i in 0..63 {
            absent.push_unchecked(key_a.get_unchecked(i));
        }
        absent.push_unchecked(0xf);
        let outcome = walk_map_chain(&mut map, Nibbles::default(), root, &absent, None)
            .expect("the inline exclusion resolves");
        assert_eq!(outcome, ChainValue::Absent);
    }
}
