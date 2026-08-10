//! The neutral access set and the single function that extracts it.

use alloy_primitives::{Address, Bytes, B256, U256};
use revm_database::State;
use revm_primitives::KECCAK_EMPTY;
use std::collections::HashMap;

/// Every state key a single block's execution touched, together with the value it saw.
///
/// This is deliberately a *read* set and not a diff: `accounts` holds entries that were only
/// loaded, `storage` holds slots whose writes were later reverted, and `codes` holds bytecode
/// that was merely executed. A consumer keyed on access cannot recover any of that from a
/// post-state diff.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExecutedBlockAccess {
    /// Accounts accessed (address -> account data).
    pub accounts: HashMap<Address, AccountAccess>,
    /// Storage slots accessed ((address, slot) -> value).
    pub storage: HashMap<(Address, B256), U256>,
    /// Bytecodes accessed (code hash -> bytecode bytes).
    pub codes: HashMap<B256, Bytes>,
    /// Lowest ancestor height the block asked for via `BLOCKHASH`, if any.
    ///
    /// Consumers use this to bound the range of ancestor headers a block depends on. It is part
    /// of the access set rather than a companion artifact because it is read off the same
    /// `State` at the same moment as everything else here.
    pub lowest_block_hash_number: Option<u64>,
}

impl ExecutedBlockAccess {
    /// Extracts the access set from revm's `State` after block execution.
    ///
    /// # Capture point
    ///
    /// This must be called after `merge_transitions` and before `take_bundle`. Earlier and
    /// contracts created by this block are not yet in `bundle_state`; later and the bundle is
    /// gone. `Executor::execute_with_state_closure` invokes its closure at exactly that point,
    /// so a re-execution path gets the same capture for free.
    pub fn from_state<DB>(statedb: &State<DB>) -> Self {
        let mut accounts = HashMap::new();
        let mut storage = HashMap::new();
        let mut codes = HashMap::new();

        // Cached accounts cover both read-only and modified state.
        for (address, cache_account) in &statedb.cache.accounts {
            if let Some(account) = &cache_account.account {
                let code_hash = if account.info.code_hash == KECCAK_EMPTY {
                    None
                } else {
                    Some(account.info.code_hash)
                };
                accounts.insert(
                    *address,
                    AccountAccess {
                        nonce: account.info.nonce,
                        balance: account.info.balance,
                        code_hash,
                    },
                );

                for (slot, value) in &account.storage {
                    storage.insert((*address, B256::from(*slot)), *value);
                }
            } else {
                // Empty, destroyed, or absent -- but accessed, which is what the set records.
                accounts.insert(
                    *address,
                    AccountAccess { nonce: 0, balance: U256::ZERO, code_hash: None },
                );
            }
        }

        for (code_hash, code) in &statedb.cache.contracts {
            let bytes = code.original_bytes();
            if !bytes.is_empty() {
                codes.insert(*code_hash, bytes);
            }
        }

        // Contracts created during this block only appear in the bundle.
        for (code_hash, code) in &statedb.bundle_state.contracts {
            let bytes = code.original_bytes();
            if !bytes.is_empty() {
                codes.insert(*code_hash, bytes);
            }
        }

        Self {
            accounts,
            storage,
            codes,
            lowest_block_hash_number: statedb.block_hashes.lowest().map(|(number, _)| number),
        }
    }

    /// Total number of unique state keys accessed.
    pub fn total_keys(&self) -> usize {
        self.accounts.len() + self.storage.len() + self.codes.len()
    }

    /// Rough heap footprint of the access set, for bounding a handoff store.
    ///
    /// This counts the access maps only. An artifact's execution output is shared with the
    /// node's in-memory chain state, so charging it here would double-count memory the store
    /// does not actually own.
    pub fn approx_heap_bytes(&self) -> usize {
        const ACCOUNT_ENTRY: usize = size_of::<Address>() + size_of::<AccountAccess>() + 16;
        const STORAGE_ENTRY: usize = size_of::<(Address, B256)>() + size_of::<U256>() + 16;
        const CODE_ENTRY: usize = size_of::<B256>() + size_of::<Bytes>() + 16;

        self.accounts.len() * ACCOUNT_ENTRY +
            self.storage.len() * STORAGE_ENTRY +
            self.codes.len() * CODE_ENTRY +
            self.codes.values().map(|code| code.len()).sum::<usize>()
    }
}

/// The account fields an access set records.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AccountAccess {
    /// Account nonce as seen during execution.
    pub nonce: u64,
    /// Account balance as seen during execution.
    pub balance: U256,
    /// Code hash, or `None` for an account with no code.
    ///
    /// `KECCAK_EMPTY` is normalized to `None` so that "no code" has one representation.
    pub code_hash: Option<B256>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use revm_database::states::{cache_account::CacheAccount, plain_account::PlainStorage};
    use revm_database_interface::EmptyDB;
    use revm_state::{AccountInfo, Bytecode};

    fn state() -> State<EmptyDB> {
        State::builder().with_database(EmptyDB::default()).with_bundle_update().build()
    }

    fn address(tag: u8) -> Address {
        Address::with_last_byte(tag)
    }

    fn code(byte: u8) -> Bytecode {
        Bytecode::new_raw(Bytes::from(vec![byte]))
    }

    #[test]
    fn an_account_that_was_only_loaded_is_still_in_the_set() {
        // The whole reason this is a read set: nothing about this account appears in a diff.
        let mut state = state();
        state.cache.accounts.insert(
            address(1),
            CacheAccount::new_loaded(
                AccountInfo { nonce: 7, balance: U256::from(9), ..Default::default() },
                PlainStorage::default(),
            ),
        );

        let access = ExecutedBlockAccess::from_state(&state);
        assert_eq!(
            access.accounts.get(&address(1)),
            Some(&AccountAccess { nonce: 7, balance: U256::from(9), code_hash: None })
        );
    }

    #[test]
    fn an_empty_code_hash_is_normalized_to_no_code() {
        let mut state = state();
        let contract = code(0x60);
        let contract_hash = contract.hash_slow();
        state.cache.accounts.insert(
            address(1),
            CacheAccount::new_loaded(
                AccountInfo { code_hash: KECCAK_EMPTY, ..Default::default() },
                PlainStorage::default(),
            ),
        );
        state.cache.accounts.insert(
            address(2),
            CacheAccount::new_loaded(
                AccountInfo { code_hash: contract_hash, ..Default::default() },
                PlainStorage::default(),
            ),
        );

        let access = ExecutedBlockAccess::from_state(&state);
        assert_eq!(access.accounts[&address(1)].code_hash, None);
        assert_eq!(access.accounts[&address(2)].code_hash, Some(contract_hash));
    }

    #[test]
    fn an_account_that_does_not_exist_is_recorded_as_accessed() {
        // Non-existence is a fact the block observed, and a consumer needs it to prove absence.
        let mut state = state();
        state.cache.accounts.insert(address(1), CacheAccount::new_loaded_not_existing());

        let access = ExecutedBlockAccess::from_state(&state);
        assert_eq!(
            access.accounts.get(&address(1)),
            Some(&AccountAccess { nonce: 0, balance: U256::ZERO, code_hash: None })
        );
    }

    #[test]
    fn storage_slots_are_keyed_by_account_and_slot() {
        let mut state = state();
        let mut storage = PlainStorage::default();
        storage.insert(U256::from(1), U256::from(11));
        storage.insert(U256::from(2), U256::from(22));
        state
            .cache
            .accounts
            .insert(address(1), CacheAccount::new_loaded(AccountInfo::default(), storage));

        let access = ExecutedBlockAccess::from_state(&state);
        assert_eq!(access.storage[&(address(1), B256::from(U256::from(1)))], U256::from(11));
        assert_eq!(access.storage[&(address(1), B256::from(U256::from(2)))], U256::from(22));
        assert_eq!(access.storage.len(), 2);
    }

    #[test]
    fn code_comes_from_both_the_cache_and_the_bundle() {
        // A contract created by this block is only ever in the bundle, which is why the capture
        // has to sit between `merge_transitions` and `take_bundle`.
        let mut state = state();
        let executed = code(0x01);
        let created = code(0x02);
        state.cache.contracts.insert(executed.hash_slow(), executed.clone());
        state.bundle_state.contracts.insert(created.hash_slow(), created.clone());
        state.cache.contracts.insert(B256::with_last_byte(9), Bytecode::default());

        let access = ExecutedBlockAccess::from_state(&state);
        assert_eq!(access.codes.len(), 2, "empty bytecode is not a code access");
        assert_eq!(access.codes[&executed.hash_slow()], executed.original_bytes());
        assert_eq!(access.codes[&created.hash_slow()], created.original_bytes());
    }

    #[test]
    fn the_blockhash_range_is_the_lowest_ancestor_the_block_asked_for() {
        let mut state = state();
        assert_eq!(ExecutedBlockAccess::from_state(&state).lowest_block_hash_number, None);

        state.block_hashes.insert(40, B256::with_last_byte(40));
        state.block_hashes.insert(12, B256::with_last_byte(12));
        state.block_hashes.insert(25, B256::with_last_byte(25));

        assert_eq!(ExecutedBlockAccess::from_state(&state).lowest_block_hash_number, Some(12));
    }
}
