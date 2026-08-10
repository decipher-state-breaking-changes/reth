//! Cache eviction policy trait and implementations.
//!
//! Accounts and Storage/Codes can have different policies applied independently.

use crate::CachedEntry;
use alloy_primitives::{Address, Bytes, B256, U256};
use std::collections::HashMap;

/// A cache eviction policy that can be applied independently to accounts and storage/codes.
pub trait CachePolicy: Send + Sync {
    /// Evict accounts that no longer satisfy the policy, returning every entry removed.
    ///
    /// The return value is the only surviving record of the eviction: the caller writes it into
    /// the block's undo record, and the values cannot be recovered from the map afterwards. A
    /// policy that drops an entry without returning it makes that block unrollbackable.
    fn evict_accounts(
        &self,
        accounts: &mut HashMap<Address, CachedEntry<AccountData>>,
        current_block: u64,
    ) -> Vec<(Address, CachedEntry<AccountData>)>;

    /// Evict storage slots and bytecodes that no longer satisfy the policy, returning every entry
    /// removed from each map. The record contract of [`Self::evict_accounts`] applies here too.
    fn evict_storage(
        &self,
        storage: &mut HashMap<(Address, B256), CachedEntry<U256>>,
        codes: &mut HashMap<B256, CachedEntry<Bytes>>,
        current_block: u64,
    ) -> EvictedStorage;

    /// Human-readable name of the policy (for logging/stats).
    fn name(&self) -> &str;
}

/// What [`CachePolicy::evict_storage`] removed.
///
/// The two maps stay separate because the caller records them under different undo keys and
/// bytecode eviction additionally has to drop the evicted code's leaf-hash memo.
#[derive(Debug, Default)]
pub struct EvictedStorage {
    /// Slots removed from the storage map.
    pub storage: Vec<((Address, B256), CachedEntry<U256>)>,
    /// Bytecodes removed from the code map.
    pub codes: Vec<(B256, CachedEntry<Bytes>)>,
}

/// Account data stored in the cache.
///
/// This is the neutral account-access record, re-exported under the name the cache has always
/// used for it. Sharing the type with `reth-execution-access` is what lets an access set
/// captured by the node's own execution move into the cache without being rebuilt entry by
/// entry.
pub use reth_execution_access::AccountAccess as AccountData;

/// Simplest policy: retain only state accessed within the last N blocks.
///
/// Can use different `window_size` for accounts vs storage by creating
/// two separate instances.
#[derive(Debug, Clone)]
pub struct LastNBlocksPolicy {
    /// Number of blocks to retain. State not accessed within this window is evicted.
    pub window_size: u64,
}

impl LastNBlocksPolicy {
    pub fn new(window_size: u64) -> Self {
        Self { window_size }
    }

    /// The oldest `last_accessed_block` this policy still keeps at `current_block`.
    ///
    /// Retention is inclusive of the cutoff, so eviction is strictly below it. Every predicate
    /// below is written as `< cutoff` to keep that boundary in one place: an entry last accessed
    /// exactly `window_size` blocks ago survives.
    fn cutoff(&self, current_block: u64) -> u64 {
        current_block.saturating_sub(self.window_size)
    }
}

impl CachePolicy for LastNBlocksPolicy {
    fn evict_accounts(
        &self,
        accounts: &mut HashMap<Address, CachedEntry<AccountData>>,
        current_block: u64,
    ) -> Vec<(Address, CachedEntry<AccountData>)> {
        let cutoff = self.cutoff(current_block);
        // `extract_if` is lazy and only removes what the iterator yields, so this must be
        // consumed to completion — `collect` is what makes the eviction happen at all.
        accounts.extract_if(|_, entry| entry.last_accessed_block < cutoff).collect()
    }

    fn evict_storage(
        &self,
        storage: &mut HashMap<(Address, B256), CachedEntry<U256>>,
        codes: &mut HashMap<B256, CachedEntry<Bytes>>,
        current_block: u64,
    ) -> EvictedStorage {
        let cutoff = self.cutoff(current_block);
        EvictedStorage {
            storage: storage.extract_if(|_, entry| entry.last_accessed_block < cutoff).collect(),
            codes: codes.extract_if(|_, entry| entry.last_accessed_block < cutoff).collect(),
        }
    }

    fn name(&self) -> &str {
        "LastNBlocks"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_last_n_blocks_eviction() {
        let policy = LastNBlocksPolicy::new(10);
        let mut accounts = HashMap::new();

        // Insert an account accessed at block 5
        accounts.insert(
            Address::ZERO,
            CachedEntry {
                value: AccountData { nonce: 1, balance: U256::from(100), code_hash: None },
                first_accessed_block: 5,
                last_accessed_block: 5,
                access_count: 1,
            },
        );

        // At block 14, it should still be retained (cutoff = 4)
        let evicted = policy.evict_accounts(&mut accounts, 14);
        assert_eq!(accounts.len(), 1);
        assert!(evicted.is_empty(), "nothing expired, so nothing may be reported as evicted");

        // At block 15, cutoff = 5, and retention is inclusive of the cutoff
        let evicted = policy.evict_accounts(&mut accounts, 15);
        assert_eq!(accounts.len(), 1, "an entry last accessed exactly window blocks ago survives");
        assert!(evicted.is_empty());

        // At block 16, cutoff = 6, so block 5 entry is evicted
        let evicted = policy.evict_accounts(&mut accounts, 16);
        assert_eq!(accounts.len(), 0);
        assert_eq!(evicted.len(), 1, "the removed entry must come back out");
        assert_eq!(evicted[0].0, Address::ZERO);
        assert_eq!(evicted[0].1.last_accessed_block, 5, "the entry is returned as it was stored");
    }

    /// The caller rebuilds its undo record from the return value alone, so anything the policy
    /// removes without returning is silently unrollbackable. Cross-check the two directly.
    #[test]
    fn every_removed_entry_is_reported() {
        let policy = LastNBlocksPolicy::new(4);
        let mut accounts = HashMap::new();
        let mut storage = HashMap::new();
        let mut codes = HashMap::new();

        for i in 0..20u64 {
            let key = Address::with_last_byte(i as u8);
            accounts.insert(
                key,
                CachedEntry {
                    value: AccountData { nonce: i, balance: U256::from(i), code_hash: None },
                    first_accessed_block: i,
                    last_accessed_block: i,
                    access_count: 1,
                },
            );
            storage.insert(
                (key, B256::with_last_byte(i as u8)),
                CachedEntry {
                    value: U256::from(i),
                    first_accessed_block: i,
                    last_accessed_block: i,
                    access_count: 1,
                },
            );
            codes.insert(
                B256::with_last_byte(i as u8),
                CachedEntry {
                    value: Bytes::from(vec![i as u8]),
                    first_accessed_block: i,
                    last_accessed_block: i,
                    access_count: 1,
                },
            );
        }

        let before: Vec<Address> = accounts.keys().copied().collect();
        let storage_before: Vec<(Address, B256)> = storage.keys().copied().collect();
        let codes_before: Vec<B256> = codes.keys().copied().collect();

        let evicted_accounts = policy.evict_accounts(&mut accounts, 15);
        let evicted = policy.evict_storage(&mut storage, &mut codes, 15);

        // cutoff = 11, so blocks 0..=10 expire and 11..=19 stay.
        assert_eq!(evicted_accounts.len(), 11);
        assert_eq!(evicted.storage.len(), 11);
        assert_eq!(evicted.codes.len(), 11);

        for key in before {
            assert_eq!(
                accounts.contains_key(&key),
                !evicted_accounts.iter().any(|(evicted, _)| *evicted == key),
                "an account is either still cached or reported evicted, never neither or both"
            );
        }
        for key in storage_before {
            assert_eq!(
                storage.contains_key(&key),
                !evicted.storage.iter().any(|(slot, _)| *slot == key),
                "a slot is either still cached or reported evicted, never neither or both"
            );
        }
        for key in codes_before {
            assert_eq!(
                codes.contains_key(&key),
                !evicted.codes.iter().any(|(code_hash, _)| *code_hash == key),
                "a code is either still cached or reported evicted, never neither or both"
            );
        }
    }

    #[test]
    fn test_separate_policies_different_windows() {
        let account_policy = LastNBlocksPolicy::new(20); // keep accounts longer
        let storage_policy = LastNBlocksPolicy::new(5); // evict storage faster

        let mut accounts = HashMap::new();
        let mut storage = HashMap::new();
        let mut codes = HashMap::new();

        let addr = Address::ZERO;
        let slot = B256::ZERO;

        accounts.insert(
            addr,
            CachedEntry {
                value: AccountData { nonce: 0, balance: U256::ZERO, code_hash: None },
                first_accessed_block: 10,
                last_accessed_block: 10,
                access_count: 1,
            },
        );

        storage.insert(
            (addr, slot),
            CachedEntry {
                value: U256::from(42),
                first_accessed_block: 10,
                last_accessed_block: 10,
                access_count: 1,
            },
        );

        // At block 20: account cutoff=0 (retained), storage cutoff=15 (evicted!)
        let evicted_accounts = account_policy.evict_accounts(&mut accounts, 20);
        let evicted = storage_policy.evict_storage(&mut storage, &mut codes, 20);

        assert_eq!(accounts.len(), 1, "account should be retained with window=20");
        assert_eq!(storage.len(), 0, "storage should be evicted with window=5");
        assert!(evicted_accounts.is_empty());
        assert_eq!(evicted.storage.len(), 1, "the evicted slot is reported by the storage policy");
        assert_eq!(evicted.storage[0].0, (addr, slot));
    }
}
