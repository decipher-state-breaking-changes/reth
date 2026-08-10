//! Representation of state accessed during a single block's execution.
//!
//! This is the input to `NetworkStateCache::on_block_executed()`.
//! It can be constructed from revm's `State` database after block execution.

use crate::policy::AccountData;
use alloy_primitives::{Address, Bytes, B256, U256};
use reth_execution_access::ExecutedBlockAccess;
use revm_database::State;
use std::collections::{HashMap, HashSet};

/// All state keys accessed during a single block's execution.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct BlockAccessedState {
    /// Accounts accessed (address → account data).
    pub accounts: HashMap<Address, AccountData>,
    /// Storage slots accessed (address, slot) → value.
    pub storage: HashMap<(Address, B256), U256>,
    /// Bytecodes accessed: code_hash → bytecode bytes.
    pub codes: HashMap<B256, Bytes>,
}

impl BlockAccessedState {
    /// Construct from revm's `State` database after block execution.
    ///
    /// This captures the complete read-set (including read-only accounts and contracts)
    /// from the database cache and bundle state.
    ///
    /// The extraction itself lives in `reth-execution-access` so that a set captured here by
    /// re-execution and one captured by the node's own execution cannot disagree about what
    /// "accessed" means. The BLOCKHASH range that extractor also returns is dropped here;
    /// callers that need it read it off [`ExecutedBlockAccess`] before converting.
    pub fn from_simulated_state<DB>(statedb: &State<DB>) -> Self {
        ExecutedBlockAccess::from_state(statedb).into()
    }

    /// Total number of unique state keys accessed.
    pub fn total_keys(&self) -> usize {
        self.accounts.len() + self.storage.len() + self.codes.len()
    }

    /// Set of all account addresses accessed.
    pub fn account_addresses(&self) -> HashSet<Address> {
        self.accounts.keys().cloned().collect()
    }

    /// Set of all storage keys accessed.
    pub fn storage_keys(&self) -> HashSet<(Address, B256)> {
        self.storage.keys().cloned().collect()
    }
}

impl From<ExecutedBlockAccess> for BlockAccessedState {
    /// Moves the three access maps across without rebuilding them.
    ///
    /// The map value types are shared with `reth-execution-access` precisely so this is a move.
    /// Rebuilding would cost a per-entry copy on every block, on the path this handoff exists to
    /// make cheaper.
    fn from(access: ExecutedBlockAccess) -> Self {
        let ExecutedBlockAccess { accounts, storage, codes, lowest_block_hash_number: _ } = access;
        Self { accounts, storage, codes }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::AccountData;

    #[test]
    fn conversion_carries_every_access_map_across() {
        // Pins the shape the handoff depends on: a captured set becomes cache input without a
        // per-entry rebuild, and only the BLOCKHASH range is dropped.
        let address = Address::with_last_byte(1);
        let code_hash = B256::with_last_byte(2);
        let mut access =
            ExecutedBlockAccess { lowest_block_hash_number: Some(42), ..Default::default() };
        access
            .accounts
            .insert(address, AccountData { nonce: 3, balance: U256::from(4), code_hash: None });
        access.storage.insert((address, B256::with_last_byte(5)), U256::from(6));
        access.codes.insert(code_hash, Bytes::from_static(&[7]));

        let accessed = BlockAccessedState::from(access);
        assert_eq!(accessed.accounts[&address].nonce, 3);
        assert_eq!(accessed.storage[&(address, B256::with_last_byte(5))], U256::from(6));
        assert_eq!(accessed.codes[&code_hash], Bytes::from_static(&[7]));
        assert_eq!(accessed.total_keys(), 3);
    }
}
