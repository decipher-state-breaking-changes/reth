//! `EvmStateProvider` backed by `{cache ∪ witness}`, with no database fallback.
//!
//! This adapts the witness trie walker ([`crate::trie`]) to reth's
//! [`EvmStateProvider`] interface, so it can be wrapped in
//! `StateProviderDatabase` and handed to a block executor. Every state read is
//! served from the reconstructed trie — cache first, then witness. A node that
//! is in neither surfaces as [`ProviderError::TrieWitnessError`] — a validation
//! failure, never a silent full-state read.

use crate::{
    cache::NodeCache,
    trie::{resolve_account, resolve_storage, NodeSource, ResolveError},
    witness::IndexedWitness,
};
use alloy_consensus::constants::KECCAK_EMPTY;
use alloy_primitives::{keccak256, Address, B256, U256};
use reth_primitives_traits::{Account, Bytecode};
use reth_revm::database::EvmStateProvider;
use reth_storage_errors::provider::{ProviderError, ProviderResult};
use reth_trie_common::TrieAccount;
use std::{cell::RefCell, collections::HashMap};

/// Serves EVM state reads from `{cache ∪ witness}`, without any backing database.
///
/// Borrows the witness (and optional hot cache) so the caller can keep mutating
/// the cache between blocks (warming) while reusing the same witness data.
pub struct WitnessStateProvider<'a> {
    witness: &'a IndexedWitness,
    /// Optional hot trie-node cache, consulted before the witness.
    cache: Option<&'a dyn NodeCache>,
    /// Memoizes resolved accounts so repeated storage reads for one account
    /// don't re-walk the account trie.
    accounts: RefCell<HashMap<Address, Option<TrieAccount>>>,
}

impl<'a> WitnessStateProvider<'a> {
    /// Serve reads from the witness only (empty cache).
    pub fn new(witness: &'a IndexedWitness) -> Self {
        Self { witness, cache: None, accounts: RefCell::new(HashMap::new()) }
    }

    /// Serve reads from `{cache ∪ witness}` — cache first, then witness.
    pub fn with_cache(witness: &'a IndexedWitness, cache: &'a dyn NodeCache) -> Self {
        Self { witness, cache: Some(cache), accounts: RefCell::new(HashMap::new()) }
    }

    /// Resolve (and memoize) the trie account for an address.
    fn account(&self, address: &Address) -> ProviderResult<Option<TrieAccount>> {
        if let Some(cached) = self.accounts.borrow().get(address) {
            return Ok(cached.clone());
        }
        let hashed = keccak256(address);
        let account =
            resolve_account(self, self.witness.pre_state_root, hashed).map_err(witness_err)?;
        self.accounts.borrow_mut().insert(*address, account.clone());
        Ok(account)
    }
}

impl NodeSource for WitnessStateProvider<'_> {
    fn get(&self, hash: &B256) -> Option<&[u8]> {
        if let Some(cache) = self.cache {
            if let Some(bytes) = cache.get(hash) {
                return Some(bytes.as_ref());
            }
        }
        self.witness.nodes.get(hash).map(|bytes| bytes.as_ref())
    }
}

impl EvmStateProvider for WitnessStateProvider<'_> {
    fn basic_account(&self, address: &Address) -> ProviderResult<Option<Account>> {
        Ok(self.account(address)?.map(|a| Account {
            nonce: a.nonce,
            balance: a.balance,
            bytecode_hash: (a.code_hash != KECCAK_EMPTY).then_some(a.code_hash),
        }))
    }

    fn block_hash(&self, number: u64) -> ProviderResult<Option<B256>> {
        Ok(self.witness.block_hashes.get(&number).copied())
    }

    fn bytecode_by_hash(&self, code_hash: &B256) -> ProviderResult<Option<Bytecode>> {
        Ok(self.witness.codes.get(code_hash).map(|code| Bytecode::new_raw(code.clone())))
    }

    fn storage(&self, account: Address, storage_key: B256) -> ProviderResult<Option<U256>> {
        let Some(trie_account) = self.account(&account)? else { return Ok(None) };
        let hashed_slot = keccak256(storage_key);
        let value =
            resolve_storage(self, trie_account.storage_root, hashed_slot).map_err(witness_err)?;
        Ok(Some(value))
    }
}

/// Map a trie-resolution failure to a provider error. A `MissingNode` means the
/// witness (and cache) did not contain a node required to serve this read.
fn witness_err(err: ResolveError) -> ProviderError {
    ProviderError::TrieWitnessError(err.to_string())
}
