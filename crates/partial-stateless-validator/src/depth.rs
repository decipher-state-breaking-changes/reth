//! Trie-depth reconstruction over a witness node set.
//!
//! The v2 byte cache treats the **top of the trie (`depth <= N`) as free**, so we
//! need each witness node's depth. We BFS from the pre-state root (depth 0),
//! following branch/extension hash children and **account-leaf -> storage-root**
//! edges, so a hot contract's storage-trie nodes continue the depth numbering
//! past the account trie (this is where real cross-block reuse lives). This
//! mirrors the team's `scripts/compute_depth.py`.

use crate::witness::IndexedWitness;
use alloy_primitives::B256;
use alloy_rlp::Decodable;
use alloy_trie::nodes::TrieNode;
use reth_trie_common::TrieAccount;
use std::collections::{HashMap, VecDeque};

/// Depth assigned to any node not reached from the root — treated as "deep" so
/// it is never mistaken for a free top node.
pub const DEEP: u8 = 255;

/// Compute the trie depth of every witness node reachable from the pre-state
/// root (root = 0). BFS over the unweighted node graph, so the first visit is
/// the minimum depth. Nodes not reached are absent from the map (callers should
/// substitute [`DEEP`]).
pub fn compute_depths(witness: &IndexedWitness) -> HashMap<B256, u8> {
    let mut depth: HashMap<B256, u8> = HashMap::new();
    let mut queue: VecDeque<(B256, u8)> = VecDeque::new();

    let root = witness.pre_state_root;
    if witness.nodes.contains_key(&root) {
        depth.insert(root, 0);
        queue.push_back((root, 0));
    }

    while let Some((hash, d)) = queue.pop_front() {
        let Some(bytes) = witness.nodes.get(&hash) else { continue };
        let child_depth = d.saturating_add(1);
        for child in child_hashes(bytes.as_ref()) {
            if witness.nodes.contains_key(&child) && !depth.contains_key(&child) {
                depth.insert(child, child_depth);
                queue.push_back((child, child_depth));
            }
        }
    }
    depth
}

/// Hash-referenced children of a trie node: branch slot hashes, an extension's
/// child hash, or (for an account leaf) the account's storage root. Inline
/// (`< 32` byte) children are embedded in the parent and carry no separate hash,
/// so they are not returned. Malformed nodes yield no children.
fn child_hashes(rlp: &[u8]) -> Vec<B256> {
    let mut out = Vec::new();
    let mut buf = rlp;
    let Ok(node) = TrieNode::decode(&mut buf) else { return out };
    match node {
        TrieNode::Branch(branch) => {
            for child in &branch.stack {
                if let Some(h) = child.as_hash() {
                    out.push(h);
                }
            }
        }
        TrieNode::Extension(ext) => {
            if let Some(h) = ext.child.as_hash() {
                out.push(h);
            }
        }
        TrieNode::Leaf(leaf) => {
            // An account leaf's value is a TrieAccount; its storage root roots a
            // storage trie whose nodes continue the depth numbering. Storage-slot
            // leaves (value = RLP(U256)) fail this decode and contribute nothing.
            if let Ok(acct) = TrieAccount::decode(&mut leaf.value.as_slice()) {
                if !acct.storage_root.is_zero() {
                    out.push(acct.storage_root);
                }
            }
        }
        TrieNode::EmptyRoot => {}
    }
    out
}
