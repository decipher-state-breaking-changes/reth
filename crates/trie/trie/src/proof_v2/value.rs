//! Generic value encoder types for proof calculation with lazy evaluation.

use crate::{
    hashed_cursor::HashedCursorFactory, prefix_set::PrefixSet, proof_v2::ProofCalculator,
    trie_cursor::TrieCursorFactory,
};
use alloy_primitives::{map::B256Map, B256, U256};
use alloy_rlp::Encodable;
use reth_execution_errors::trie::StateProofError;
use reth_primitives_traits::Account;
use std::rc::Rc;

/// A trait for deferred RLP-encoding of leaf values.
pub trait DeferredValueEncoder {
    /// RLP encodes the value into the provided buffer.
    ///
    /// # Arguments
    ///
    /// * `buf` - A mutable buffer to encode the data into
    fn encode(self, buf: &mut Vec<u8>) -> Result<(), StateProofError>;
}

/// A trait for RLP-encoding values for proof calculation. This trait is designed to allow the lazy
/// computation of leaf values in a generic way.
///
/// When calculating a leaf value in a storage trie the [`DeferredValueEncoder`] simply holds onto
/// the slot value, and the `encode` method synchronously RLP-encodes it.
///
/// When calculating a leaf value in the accounts trie we create a [`DeferredValueEncoder`] to
/// initiate any asynchronous computation of the account's storage root we want to do. Later we call
/// [`DeferredValueEncoder::encode`] to obtain the result of that computation and RLP-encode it.
pub trait LeafValueEncoder {
    /// The type of value being encoded (e.g., U256 for storage, Account for accounts).
    type Value;

    /// The type that will compute and encode the value when needed.
    type DeferredEncoder: DeferredValueEncoder;

    /// Returns an encoder that will RLP-encode the value when its `encode` method is called.
    ///
    /// # Arguments
    ///
    /// * `key` - The key the value was stored at in the DB
    /// * `value` - The value to encode
    ///
    /// The returned deferred encoder will be called as late as possible in the algorithm to
    /// maximize the time available for parallel computation (e.g., storage root calculation).
    fn deferred_encoder(&mut self, key: B256, value: Self::Value) -> Self::DeferredEncoder;
}

/// An encoder for storage slot values.
///
/// This encoder simply RLP-encodes U256 storage values directly.
#[derive(Debug, Clone, Copy, Default)]
pub struct StorageValueEncoder;

/// The deferred encoder for a storage slot value.
#[derive(Debug, Clone, Copy)]
pub struct StorageDeferredValueEncoder(U256);

impl DeferredValueEncoder for StorageDeferredValueEncoder {
    fn encode(self, buf: &mut Vec<u8>) -> Result<(), StateProofError> {
        self.0.encode(buf);
        Ok(())
    }
}

impl LeafValueEncoder for StorageValueEncoder {
    type Value = U256;
    type DeferredEncoder = StorageDeferredValueEncoder;

    fn deferred_encoder(&mut self, _key: B256, value: Self::Value) -> Self::DeferredEncoder {
        StorageDeferredValueEncoder(value)
    }
}

/// An account value encoder that synchronously computes storage roots.
///
/// This encoder contains factories for creating trie and hashed cursors. Storage roots are
/// computed synchronously within the deferred encoder using a `StorageProofCalculator`.
#[derive(Debug, Clone)]
pub struct SyncAccountValueEncoder<T, H> {
    /// Factory for creating trie cursors.
    trie_cursor_factory: Rc<T>,
    /// Factory for creating hashed cursors.
    hashed_cursor_factory: Rc<H>,
    /// Storage prefix sets keyed by hashed address.
    storage_prefix_sets: Rc<B256Map<PrefixSet>>,
    /// Storage roots already computed while generating storage proofs.
    precomputed_storage_roots: Rc<B256Map<B256>>,
}

impl<T, H> SyncAccountValueEncoder<T, H> {
    /// Create a new account value encoder with the given factories.
    pub fn new(trie_cursor_factory: T, hashed_cursor_factory: H) -> Self {
        Self {
            trie_cursor_factory: Rc::new(trie_cursor_factory),
            hashed_cursor_factory: Rc::new(hashed_cursor_factory),
            storage_prefix_sets: Rc::new(B256Map::default()),
            precomputed_storage_roots: Rc::new(B256Map::default()),
        }
    }

    /// Sets the storage prefix sets. When given, all cached storage trie hashes matching the
    /// prefix sets will be invalidated during storage root calculation for the corresponding
    /// accounts.
    pub fn with_storage_prefix_sets(mut self, storage_prefix_sets: B256Map<PrefixSet>) -> Self {
        self.storage_prefix_sets = Rc::new(storage_prefix_sets);
        self
    }

    /// Sets storage roots computed by preceding storage proof calculations.
    pub fn with_precomputed_storage_roots(
        mut self,
        precomputed_storage_roots: B256Map<B256>,
    ) -> Self {
        self.precomputed_storage_roots = Rc::new(precomputed_storage_roots);
        self
    }
}

/// The deferred encoder for an account value with synchronous storage root calculation.
#[derive(Debug, Clone)]
pub struct SyncAccountDeferredValueEncoder<T, H> {
    trie_cursor_factory: Rc<T>,
    hashed_cursor_factory: Rc<H>,
    storage_prefix_sets: Rc<B256Map<PrefixSet>>,
    hashed_address: B256,
    account: Account,
    precomputed_storage_root: Option<B256>,
}

impl<T, H> DeferredValueEncoder for SyncAccountDeferredValueEncoder<T, H>
where
    T: TrieCursorFactory,
    H: HashedCursorFactory,
{
    fn encode(self, buf: &mut Vec<u8>) -> Result<(), StateProofError> {
        let storage_root = if let Some(storage_root) = self.precomputed_storage_root {
            storage_root
        } else {
            let trie_cursor = self.trie_cursor_factory.storage_trie_cursor(self.hashed_address)?;
            let hashed_cursor =
                self.hashed_cursor_factory.hashed_storage_cursor(self.hashed_address)?;
            let mut calculator = ProofCalculator::new_storage(trie_cursor, hashed_cursor);
            if let Some(prefix_set) = self.storage_prefix_sets.get(&self.hashed_address) {
                calculator = calculator.with_prefix_set(prefix_set.clone());
            }
            let root_node = calculator.storage_root_node(self.hashed_address)?;
            calculator
                .compute_root_hash(&[root_node])?
                .expect("storage_root_node returns a node at empty path")
        };

        let trie_account = self.account.into_trie_account(storage_root);
        trie_account.encode(buf);

        Ok(())
    }
}

impl<T, H> LeafValueEncoder for SyncAccountValueEncoder<T, H>
where
    T: TrieCursorFactory,
    H: HashedCursorFactory,
{
    type Value = Account;
    type DeferredEncoder = SyncAccountDeferredValueEncoder<T, H>;

    fn deferred_encoder(
        &mut self,
        hashed_address: B256,
        account: Self::Value,
    ) -> Self::DeferredEncoder {
        SyncAccountDeferredValueEncoder {
            trie_cursor_factory: Rc::clone(&self.trie_cursor_factory),
            hashed_cursor_factory: Rc::clone(&self.hashed_cursor_factory),
            storage_prefix_sets: Rc::clone(&self.storage_prefix_sets),
            hashed_address,
            account,
            precomputed_storage_root: self.precomputed_storage_roots.get(&hashed_address).copied(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        hashed_cursor::noop::NoopHashedCursorFactory, trie_cursor::noop::NoopTrieCursorFactory,
    };
    use reth_trie_common::EMPTY_ROOT_HASH;

    #[test]
    fn account_encoder_reuses_precomputed_storage_root() {
        let hashed_address = B256::repeat_byte(0x11);
        let storage_root = B256::repeat_byte(0x22);
        let account = Account { nonce: 7, ..Default::default() };
        let mut encoder =
            SyncAccountValueEncoder::new(NoopTrieCursorFactory, NoopHashedCursorFactory)
                .with_precomputed_storage_roots(B256Map::from_iter([(
                    hashed_address,
                    storage_root,
                )]));

        let deferred = encoder.deferred_encoder(hashed_address, account);
        let mut encoded = Vec::new();
        deferred.encode(&mut encoded).unwrap();

        let mut expected = Vec::new();
        account.into_trie_account(storage_root).encode(&mut expected);
        assert_eq!(encoded, expected);
    }

    #[test]
    fn account_encoder_falls_back_when_precomputed_root_is_missing() {
        let hashed_address = B256::repeat_byte(0x33);
        let account = Account { nonce: 9, ..Default::default() };
        let mut encoder =
            SyncAccountValueEncoder::new(NoopTrieCursorFactory, NoopHashedCursorFactory);

        let deferred = encoder.deferred_encoder(hashed_address, account);
        let mut encoded = Vec::new();
        deferred.encode(&mut encoded).unwrap();

        let mut expected = Vec::new();
        account.into_trie_account(EMPTY_ROOT_HASH).encode(&mut expected);
        assert_eq!(encoded, expected);
    }
}
