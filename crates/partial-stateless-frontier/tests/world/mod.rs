//! Shared synthetic-chain harness for the frontier's builder tests: a small real trie world,
//! a three-block chain with deletion-driven structure, and proof sources that stand in for a
//! state database and a recorded witness.
#![allow(dead_code)]

use alloy_primitives::{keccak256, map::B256Map, Address, Bytes, B256, U256};
use alloy_trie::{proof::ProofRetainer, HashBuilder};
use partial_stateless::{
    accessed_state::BlockAccessedState, policy::AccountData, CacheConfig, NetworkStateCache,
    ParallelProof, PartialTrieNodeCache, TransitionProofSource,
};
use reth_primitives_traits::Account;
use reth_trie_common::{
    DecodedMultiProofV2, HashedPostState, HashedStorage, MultiProofTargetsV2, Nibbles,
};
use std::collections::BTreeMap;

/// A whole parent state, small enough to build a real trie from and rich enough to have branches.
#[derive(Debug, Clone, Default)]
pub struct World {
    pub accounts: BTreeMap<Address, (Account, BTreeMap<B256, U256>)>,
}

impl World {
    pub fn set_account(&mut self, address: Address, nonce: u64, balance: u64) {
        self.accounts
            .entry(address)
            .or_insert_with(|| {
                (Account { nonce: 0, balance: U256::ZERO, bytecode_hash: None }, BTreeMap::new())
            })
            .0 = Account { nonce, balance: U256::from(balance), bytecode_hash: None };
    }

    pub fn set_storage(&mut self, address: Address, slot: B256, value: u64) {
        let entry = self.accounts.entry(address).or_insert_with(|| {
            (Account { nonce: 0, balance: U256::ZERO, bytecode_hash: None }, BTreeMap::new())
        });
        if value == 0 {
            entry.1.remove(&slot);
        } else {
            entry.1.insert(slot, U256::from(value));
        }
    }

    pub fn state_root(&self) -> B256 {
        reth_trie::test_utils::state_root(self.accounts.iter().map(
            |(address, (account, storage))| {
                (*address, (*account, storage.iter().map(|(slot, value)| (*slot, *value))))
            },
        ))
    }

    /// Every node of the account trie and of every storage trie, as a flat content-addressed map.
    ///
    /// This is what stands in for the node's database: a source that can answer any target,
    /// because it holds the whole trie.
    pub fn complete_witness(&self) -> B256Map<Bytes> {
        let mut witness = B256Map::default();
        let mut account_leaves = Vec::new();
        for (address, (account, storage)) in &self.accounts {
            let storage_root = storage_trie(storage, &mut witness);
            let trie_account = account.into_trie_account(storage_root);
            account_leaves.push((keccak256(address), alloy_rlp::encode(trie_account)));
        }
        account_leaves.sort_by_key(|(hashed, _)| *hashed);
        build_trie(&account_leaves, &mut witness);
        witness
    }
}

/// Builds one trie from sorted `(hashed_key, rlp_value)` leaves, adding every node to `witness`.
pub fn build_trie(leaves: &[(B256, Vec<u8>)], witness: &mut B256Map<Bytes>) -> B256 {
    if leaves.is_empty() {
        return alloy_trie::EMPTY_ROOT_HASH
    }
    let targets = leaves.iter().map(|(key, _)| Nibbles::unpack(key)).collect::<Vec<_>>();
    let mut builder = HashBuilder::default().with_proof_retainer(ProofRetainer::new(targets));
    for (key, value) in leaves {
        builder.add_leaf(Nibbles::unpack(key), value);
    }
    let root = builder.root();
    for (_, node) in builder.take_proof_nodes().into_inner() {
        witness.insert(keccak256(&node), node);
    }
    root
}

pub fn storage_trie(storage: &BTreeMap<B256, U256>, witness: &mut B256Map<Bytes>) -> B256 {
    let mut leaves = storage
        .iter()
        .filter(|(_, value)| !value.is_zero())
        .map(|(slot, value)| (keccak256(slot), alloy_rlp::encode_fixed_size(value).to_vec()))
        .collect::<Vec<_>>();
    leaves.sort_by_key(|(hashed, _)| *hashed);
    build_trie(&leaves, witness)
}

/// A source that answers any target from a complete trie — what a state database is, for this test.
#[derive(Debug)]
pub struct WholeTrieSource {
    pub proof: DecodedMultiProofV2,
}

impl WholeTrieSource {
    pub fn new(state_root: B256, witness: &B256Map<Bytes>) -> Self {
        Self {
            proof: DecodedMultiProofV2::from_witness(state_root, witness)
                .expect("the complete trie decodes"),
        }
    }
}

impl TransitionProofSource for WholeTrieSource {
    fn multiproof_v2(&self, targets: MultiProofTargetsV2) -> eyre::Result<DecodedMultiProofV2> {
        let mut proof = self.proof.clone();
        proof.retain_targets(&targets);
        Ok(proof)
    }

    fn parallel_initial_proof(
        &self,
    ) -> Option<&dyn Fn(MultiProofTargetsV2) -> eyre::Result<ParallelProof>> {
        None
    }
}

pub fn address(tag: u8) -> Address {
    Address::repeat_byte(tag)
}

pub fn slot(tag: u8) -> B256 {
    B256::repeat_byte(tag)
}

/// The parent state every block in this test builds on.
pub fn genesis() -> World {
    let mut world = World::default();
    for tag in 1..=64u8 {
        world.set_account(address(tag), u64::from(tag), u64::from(tag) * 1_000);
    }
    for tag in 1..=32u8 {
        world.set_storage(address(1), slot(0x10 + tag), u64::from(tag) * 7);
        world.set_storage(address(2), slot(0x20 + tag), u64::from(tag) * 11);
    }
    world
}

/// What one block did: which state it touched, and what it changed.
pub struct BlockSpec {
    pub number: u64,
    pub touched_accounts: Vec<Address>,
    pub touched_storage: Vec<(Address, B256)>,
    pub account_writes: Vec<(Address, u64, u64)>,
    pub storage_writes: Vec<(Address, B256, u64)>,
    /// Accounts whose entire storage this block destroys before its writes apply.
    pub wiped_storage: Vec<Address>,
}

/// The three blocks: two to warm the caches, one to compare on.
pub fn chain() -> Vec<BlockSpec> {
    vec![
        BlockSpec {
            number: 101,
            touched_accounts: vec![address(1), address(2), address(3)],
            touched_storage: vec![(address(1), slot(0x11)), (address(1), slot(0x12))],
            account_writes: vec![(address(1), 10, 5_000), (address(3), 30, 7_000)],
            storage_writes: vec![(address(1), slot(0x11), 101)],
            wiped_storage: vec![],
        },
        BlockSpec {
            number: 102,
            touched_accounts: vec![address(1), address(4)],
            touched_storage: vec![(address(2), slot(0x21)), (address(2), slot(0x22))],
            account_writes: vec![(address(4), 40, 9_000)],
            storage_writes: vec![(address(2), slot(0x21), 202), (address(2), slot(0x23), 0)],
            wiped_storage: vec![],
        },
        BlockSpec {
            number: 103,
            touched_accounts: vec![address(1), address(5), address(6)],
            touched_storage: vec![
                (address(1), slot(0x11)),
                (address(1), slot(0x13)),
                (address(2), slot(0x21)),
            ],
            account_writes: vec![(address(5), 50, 11_000), (address(1), 11, 5_500)],
            // Deletions, deliberately: a collapsing branch is what makes a transition discover it
            // needs a sibling nobody proved for it, which is the structural-round path.
            storage_writes: vec![
                (address(1), slot(0x13), 303),
                (address(1), slot(0x14), 0),
                (address(1), slot(0x15), 0),
                (address(1), slot(0x16), 0),
                (address(1), slot(0x17), 0),
                (address(1), slot(0x18), 0),
                (address(1), slot(0x19), 0),
                (address(1), slot(0x1a), 0),
                (address(1), slot(0x1b), 0),
                (address(1), slot(0x1c), 0),
                (address(1), slot(0x1d), 0),
                (address(1), slot(0x1e), 0),
                (address(1), slot(0x1f), 0),
                (address(1), slot(0x20), 0),
            ],
            wiped_storage: vec![],
        },
    ]
}

/// The access set a block observed, drawn from the parent world so the values are the parent's.
pub fn accessed_state(parent: &World, spec: &BlockSpec) -> BlockAccessedState {
    let mut accessed = BlockAccessedState::default();
    for address in &spec.touched_accounts {
        let (account, _) = &parent.accounts[address];
        accessed.accounts.insert(
            *address,
            AccountData { nonce: account.nonce, balance: account.balance, code_hash: None },
        );
    }
    for (address, slot) in &spec.touched_storage {
        let value = parent.accounts[address].1.get(slot).copied().unwrap_or_default();
        accessed.storage.insert((*address, *slot), value);
    }
    // Every write is also an access — a block cannot change what it did not touch.
    for (address, nonce, balance) in &spec.account_writes {
        accessed.accounts.entry(*address).or_insert(AccountData {
            nonce: *nonce,
            balance: U256::from(*balance),
            code_hash: None,
        });
    }
    for (address, slot, _) in &spec.storage_writes {
        let value = parent.accounts[address].1.get(slot).copied().unwrap_or_default();
        accessed.storage.entry((*address, *slot)).or_insert(value);
    }
    accessed
}

/// Applies a block's writes, returning the child world and the hashed post state that describes it.
pub fn apply(parent: &World, spec: &BlockSpec) -> (World, HashedPostState) {
    let mut child = parent.clone();
    let mut post = HashedPostState::default();
    // Wipes destroy the whole storage first; the block's own writes then apply on top, which is
    // the self-destruct-and-recreate shape a wiped `HashedStorage` describes.
    for address in &spec.wiped_storage {
        if let Some((_, storage)) = child.accounts.get_mut(address) {
            storage.clear();
        }
    }
    for (address, nonce, balance) in &spec.account_writes {
        child.set_account(*address, *nonce, *balance);
        let (account, _) = &child.accounts[address];
        post.accounts.insert(keccak256(address), Some(*account));
    }
    let mut by_account: BTreeMap<Address, Vec<(B256, U256)>> = BTreeMap::new();
    for (address, slot, value) in &spec.storage_writes {
        child.set_storage(*address, *slot, *value);
        by_account.entry(*address).or_default().push((keccak256(slot), U256::from(*value)));
    }
    for address in &spec.wiped_storage {
        by_account.entry(*address).or_default();
    }
    for (address, slots) in by_account {
        let wiped = spec.wiped_storage.contains(&address);
        post.storages.insert(keccak256(address), HashedStorage::from_iter(wiped, slots));
        // A storage write touches the account's own leaf too: its storage root moves.
        let (account, _) = &child.accounts[&address];
        post.accounts.entry(keccak256(address)).or_insert(Some(*account));
    }
    (child, post)
}

/// One side of the comparison: a builder pair advanced block by block.
pub struct Side {
    pub cache: NetworkStateCache,
    pub trie: PartialTrieNodeCache,
}

impl Side {
    pub fn cold_at(config: &CacheConfig, parent_block: u64) -> Self {
        Self { cache: config.new_cache_at(parent_block), trie: PartialTrieNodeCache::new() }
    }
}
