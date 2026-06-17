//! Loading and indexing an execution witness produced by `debug_executionWitness`
//! (as dumped to disk by the `ps-replay` tool).
//!
//! The witness is a *flat* set of state-trie node preimages plus contract
//! bytecodes and ancestor headers. We index the trie nodes by their keccak hash
//! so the trie walker ([`crate::trie`]) can resolve any account/slot the block
//! touched — without a state database.

use alloy_consensus::Header;
use alloy_primitives::{keccak256, map::B256Map, Address, Bytes, B256};
use alloy_rlp::Decodable;
use std::{collections::HashMap, path::Path};

/// Raw `debug_executionWitness` response, as stored by `ps-replay`.
///
/// Field semantics (legacy/canonical witness format):
/// - `state`   — RLP-encoded state-trie nodes (`keccak(node) => node`)
/// - `codes`   — contract bytecodes (`keccak(code) => code`)
/// - `keys`    — preimages of hashed trie keys (20-byte addresses, 32-byte slots)
/// - `headers` — RLP-encoded ancestor block headers (most recent = parent)
#[derive(Debug, Clone, serde::Deserialize)]
pub struct WitnessJson {
    pub state: Vec<Bytes>,
    #[serde(default)]
    pub codes: Vec<Bytes>,
    #[serde(default)]
    pub keys: Vec<Bytes>,
    #[serde(default)]
    pub headers: Vec<Bytes>,
}

/// An execution witness indexed for state reconstruction.
pub struct IndexedWitness {
    /// State-trie nodes: `keccak(rlp_node) => rlp_node`.
    pub nodes: B256Map<Bytes>,
    /// Contract bytecodes: `keccak(code) => code`.
    pub codes: B256Map<Bytes>,
    /// Key preimages recovered from the witness (addresses and storage slots).
    pub keys: Vec<Bytes>,
    /// Pre-state root the witness is proven against (= parent block's state root).
    pub pre_state_root: B256,
    /// Ancestor block hashes by number (for the `BLOCKHASH` opcode).
    pub block_hashes: HashMap<u64, B256>,
}

impl IndexedWitness {
    /// Load and index a witness JSON file written by `ps-replay`.
    pub fn load(path: &Path) -> eyre::Result<Self> {
        let raw = std::fs::read(path)?;
        let witness: WitnessJson = serde_json::from_slice(&raw)?;
        Self::from_witness(witness)
    }

    /// Index an already-deserialized witness.
    pub fn from_witness(w: WitnessJson) -> eyre::Result<Self> {
        let mut nodes = B256Map::default();
        for node in &w.state {
            nodes.insert(keccak256(node), node.clone());
        }
        let mut codes = B256Map::default();
        for code in &w.codes {
            codes.insert(keccak256(code), code.clone());
        }
        let (pre_state_root, block_hashes) = index_headers(&w.headers)?;
        Ok(Self { nodes, codes, keys: w.keys, pre_state_root, block_hashes })
    }

    /// Addresses recovered from the key preimages (20-byte entries).
    pub fn addresses(&self) -> Vec<Address> {
        self.keys
            .iter()
            .filter(|k| k.len() == 20)
            .map(|k| Address::from_slice(k))
            .collect()
    }
}

/// Decode the witness ancestor headers, returning:
/// - the state root of the most recent ancestor (the parent block, whose
///   post-state is this block's pre-state), and
/// - a `number => hash` map for the `BLOCKHASH` opcode.
fn index_headers(headers: &[Bytes]) -> eyre::Result<(B256, HashMap<u64, B256>)> {
    let mut block_hashes = HashMap::with_capacity(headers.len());
    let mut parent: Option<Header> = None;
    for raw in headers {
        // The canonical block hash is keccak of the RLP-encoded header.
        let hash = keccak256(raw);
        let header = Header::decode(&mut raw.as_ref())?;
        block_hashes.insert(header.number, hash);
        if parent.as_ref().is_none_or(|p| header.number > p.number) {
            parent = Some(header);
        }
    }
    let pre_state_root = parent.map(|h| h.state_root).ok_or_else(|| {
        eyre::eyre!("witness has no ancestor headers; cannot determine pre-state root")
    })?;
    Ok((pre_state_root, block_hashes))
}
