//! Merkle-Patricia-Trie walker over a witness/cache node set.
//!
//! This is the heart of the partial-stateless validator: instead of a state
//! database, state reads are served by walking the trie whose nodes come from a
//! [`NodeSource`] — conceptually `{hot cache ∪ block witness}`. Every node is
//! addressed by its keccak hash.
//!
//! Crucially, there is **no database fallback**. If a node needed to resolve a
//! key is in neither the cache nor the witness, [`resolve`] returns
//! [`ResolveError::MissingNode`]. That is the whole point: a real stateless
//! validator only holds the cache + received witness, so a missing node means
//! the witness was insufficient — a validation failure, not a silent DB read.

use alloy_primitives::{B256, U256};
use alloy_rlp::Decodable;
use alloy_trie::{
    nodes::{RlpNode, TrieNode},
    EMPTY_ROOT_HASH,
};
use reth_trie_common::TrieAccount;

/// Supplies trie node preimages by hash. Returns `None` when the node is held by
/// neither the cache nor the witness.
pub trait NodeSource {
    /// Fetch the RLP bytes of the trie node with the given keccak hash.
    fn get(&self, hash: &B256) -> Option<&[u8]>;
}

/// Error from a trie traversal.
#[derive(Debug)]
pub enum ResolveError {
    /// A node required to follow the path was not in the cache or witness.
    /// Holds the hash of the missing node. This is a validation failure.
    MissingNode(B256),
    /// A node failed to RLP-decode (malformed witness).
    Rlp(alloy_rlp::Error),
}

/// Resolve a 32-byte hashed key against the trie rooted at `root`.
///
/// Returns:
/// - `Ok(Some(value))` — the leaf RLP value bytes for the key (account or slot).
/// - `Ok(None)` — the key is *provably absent* (the path ends without a match).
/// - `Err(MissingNode)` — a node on the path is missing (witness insufficient).
pub fn resolve(
    src: &impl NodeSource,
    root: B256,
    hashed_key: B256,
) -> Result<Option<Vec<u8>>, ResolveError> {
    if root == EMPTY_ROOT_HASH {
        return Ok(None);
    }

    let nibbles = to_nibbles(hashed_key.as_slice());
    let mut node = decode_at_hash(src, root)?;
    let mut offset = 0usize;

    loop {
        match node {
            TrieNode::Branch(branch) => {
                let nib = nibbles[offset];
                offset += 1;
                match branch_child(&branch, nib) {
                    None => return Ok(None), // empty slot => key absent
                    Some(child) => node = follow(src, child)?,
                }
            }
            TrieNode::Extension(ext) => {
                let key = ext.key.to_vec();
                if nibbles.len() < offset + key.len()
                    || nibbles[offset..offset + key.len()] != key[..]
                {
                    return Ok(None); // diverges => key absent
                }
                offset += key.len();
                node = follow(src, &ext.child)?;
            }
            TrieNode::Leaf(leaf) => {
                let key = leaf.key.to_vec();
                return Ok(if nibbles[offset..] == key[..] { Some(leaf.value) } else { None });
            }
            TrieNode::EmptyRoot => return Ok(None),
        }
    }
}

/// Resolve an account, decoding the leaf into a [`TrieAccount`].
pub fn resolve_account(
    src: &impl NodeSource,
    state_root: B256,
    hashed_address: B256,
) -> Result<Option<TrieAccount>, ResolveError> {
    match resolve(src, state_root, hashed_address)? {
        Some(value) => {
            let account = TrieAccount::decode(&mut value.as_slice()).map_err(ResolveError::Rlp)?;
            Ok(Some(account))
        }
        None => Ok(None),
    }
}

/// Resolve a storage slot, decoding the leaf into a [`U256`].
///
/// A zero / absent slot returns `U256::ZERO`.
pub fn resolve_storage(
    src: &impl NodeSource,
    storage_root: B256,
    hashed_slot: B256,
) -> Result<U256, ResolveError> {
    if storage_root == EMPTY_ROOT_HASH {
        return Ok(U256::ZERO);
    }
    match resolve(src, storage_root, hashed_slot)? {
        Some(value) => {
            let slot = U256::decode(&mut value.as_slice()).map_err(ResolveError::Rlp)?;
            Ok(slot)
        }
        None => Ok(U256::ZERO),
    }
}

/// Decode the trie node stored at `hash` (must be present in the source).
fn decode_at_hash(src: &impl NodeSource, hash: B256) -> Result<TrieNode, ResolveError> {
    let mut rlp = src.get(&hash).ok_or(ResolveError::MissingNode(hash))?;
    TrieNode::decode(&mut rlp).map_err(ResolveError::Rlp)
}

/// Follow a branch/extension child reference to the next node.
///
/// A child is either a 32-byte hash (look it up) or an inline (<32 byte) node
/// embedded directly in the parent (decode in place).
fn follow(src: &impl NodeSource, child: &RlpNode) -> Result<TrieNode, ResolveError> {
    if let Some(hash) = child.as_hash() {
        decode_at_hash(src, hash)
    } else {
        let mut bytes: &[u8] = child.as_ref();
        TrieNode::decode(&mut bytes).map_err(ResolveError::Rlp)
    }
}

/// Pick the child at branch index `nib` (0..=15), if present.
///
/// A branch stores only its present children in `stack`, in nibble order, so the
/// stack index is the number of set mask bits below `nib`.
fn branch_child(branch: &alloy_trie::nodes::BranchNode, nib: u8) -> Option<&RlpNode> {
    if !branch.state_mask.is_bit_set(nib) {
        return None;
    }
    let index = (branch.state_mask.get() & ((1u16 << nib) - 1)).count_ones() as usize;
    branch.stack.get(index)
}

/// Expand bytes into their nibble sequence (high nibble first).
fn to_nibbles(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(b >> 4);
        out.push(b & 0x0f);
    }
    out
}

impl core::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingNode(h) => write!(f, "trie node missing from cache+witness: {h}"),
            Self::Rlp(e) => write!(f, "trie node rlp decode error: {e}"),
        }
    }
}

impl std::error::Error for ResolveError {}
