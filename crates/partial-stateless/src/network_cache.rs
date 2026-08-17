//! NetworkStateCache: the protocol-level cache representing state that all
//! validators are assumed to hold.
//!
//! Completely separate from reth's internal `ExecutionCache`.

use crate::{
    accessed_state::BlockAccessedState,
    policy::{AccountData, CachePolicy},
    sidecar::{CacheAnchor, StateTargetSet},
};
use alloy_primitives::{keccak256, Address, Bytes, Keccak256, B256, U256};
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    sync::OnceLock,
    time::Instant,
};
use tracing::{debug, info};

fn initialized_cache_root(root: Option<B256>) -> OnceLock<B256> {
    let memoized = OnceLock::new();
    if let Some(root) = root {
        memoized.set(root).expect("new OnceLock is empty");
    }
    memoized
}

fn push_u256(out: &mut Vec<u8>, value: U256) {
    out.extend_from_slice(&value.to_be_bytes::<32>());
}

fn namespace_root(label: &[u8], leaves: &[B256]) -> B256 {
    let mut preimage = Vec::new();
    preimage.extend_from_slice(b"NetworkStateCacheNamespaceRoot/v1");
    preimage.extend_from_slice(label);
    preimage.extend_from_slice(&(leaves.len() as u64).to_be_bytes());
    for leaf in leaves {
        preimage.extend_from_slice(leaf.as_slice());
    }
    keccak256(preimage)
}

/// [`namespace_root`] over a borrowed stream of leaves instead of a slice.
///
/// The index holds its digests in a `BTreeMap`, whose values are ordered but not contiguous, and
/// collecting them into a slice first would reintroduce the ~2 MiB per-block allocation the index
/// exists to remove. Absorbing them one at a time is bit-identical — keccak is a sponge, and
/// `absorb(a) ++ absorb(b)` is `absorb(a ++ b)` — and measures within 5% of the contiguous form,
/// which is why the streaming form is worth having (`tests/digest_index_profile.rs`).
fn namespace_root_streamed<'a>(
    label: &[u8],
    leaves: impl ExactSizeIterator<Item = &'a B256>,
) -> B256 {
    let mut hasher = Keccak256::new();
    hasher.update(b"NetworkStateCacheNamespaceRoot/v1");
    hasher.update(label);
    hasher.update((leaves.len() as u64).to_be_bytes());
    for leaf in leaves {
        hasher.update(leaf.as_slice());
    }
    hasher.finalize()
}

/// The v2 cache root over three namespace roots and their populations.
///
/// The two leaf-producing paths — the index-backed one and the slow reference — both end here, so
/// the commitment's byte format has exactly one definition. What they deliberately do not share is
/// how the leaves are produced and ordered: that is what the index changed, and what a differential
/// test has to be free to disagree about.
fn cache_root_v2(
    account_root: B256,
    account_count: usize,
    storage_root: B256,
    storage_count: usize,
    code_root: B256,
    code_count: usize,
) -> B256 {
    let mut preimage = Vec::new();
    preimage.extend_from_slice(b"NetworkStateCacheRoot/v2");

    preimage.extend_from_slice(b"account_root");
    preimage.extend_from_slice(account_root.as_slice());
    preimage.extend_from_slice(b"account_count");
    preimage.extend_from_slice(&(account_count as u64).to_be_bytes());

    preimage.extend_from_slice(b"storage_root");
    preimage.extend_from_slice(storage_root.as_slice());
    preimage.extend_from_slice(b"storage_count");
    preimage.extend_from_slice(&(storage_count as u64).to_be_bytes());

    preimage.extend_from_slice(b"code_root");
    preimage.extend_from_slice(code_root.as_slice());
    preimage.extend_from_slice(b"code_count");
    preimage.extend_from_slice(&(code_count as u64).to_be_bytes());

    keccak256(preimage)
}

fn hash_account(address: Address, entry: &CachedEntry<AccountData>) -> B256 {
    let mut preimage = Vec::new();
    preimage.extend_from_slice(b"NetworkStateCacheLeaf/v1/account");
    preimage.extend_from_slice(address.as_slice());
    preimage.extend_from_slice(&entry.value.nonce.to_be_bytes());
    push_u256(&mut preimage, entry.value.balance);
    match entry.value.code_hash {
        Some(code_hash) => {
            preimage.extend_from_slice(b"code_hash");
            preimage.extend_from_slice(code_hash.as_slice());
        }
        None => preimage.extend_from_slice(b"no_code_hash"),
    }
    preimage.extend_from_slice(&entry.last_accessed_block.to_be_bytes());
    keccak256(preimage)
}

fn hash_storage(address: Address, slot: B256, entry: &CachedEntry<U256>) -> B256 {
    let mut preimage = Vec::new();
    preimage.extend_from_slice(b"NetworkStateCacheLeaf/v1/storage");
    preimage.extend_from_slice(address.as_slice());
    preimage.extend_from_slice(slot.as_slice());
    push_u256(&mut preimage, entry.value);
    preimage.extend_from_slice(&entry.last_accessed_block.to_be_bytes());
    keccak256(preimage)
}

/// Absorbs every part of a code leaf preimage that `last_accessed_block` does not follow.
///
/// The bytecode is the largest input in the whole cache root — about 23 MiB per block on the
/// 2026-08-05 mainnet run — and it is a function of `code_hash` alone, so absorbing it once and
/// keeping the sponge state turns each later block's code leaf into a single 8-byte update.
/// Keccak is a sponge, so `absorb(prefix) → clone → absorb(suffix) → finalize` is bit-identical to
/// hashing the concatenation; [`hash_code_reference`] is the differential check on that.
fn code_leaf_prefix_hasher(code_hash: B256, code: &Bytes) -> Keccak256 {
    let mut hasher = Keccak256::new();
    hasher.update(b"NetworkStateCacheLeaf/v1/code");
    hasher.update(code_hash.as_slice());
    hasher.update((code.len() as u64).to_be_bytes());
    hasher.update(code);
    hasher
}

fn finish_code_leaf(mut hasher: Keccak256, last_accessed_block: u64) -> B256 {
    hasher.update(last_accessed_block.to_be_bytes());
    hasher.finalize()
}

/// A code leaf digest, finishing the memoized sponge when one is present.
///
/// A memo miss is only reachable through a path that inserted a code without going through
/// `on_block_executed`, `restore`, or `rollback_block`. Recompute rather than assume, so the
/// digest never depends on the memo being populated.
fn code_leaf_digest(
    hashers: &HashMap<B256, Keccak256>,
    code_hash: B256,
    entry: &CachedEntry<Bytes>,
) -> B256 {
    match hashers.get(&code_hash) {
        Some(hasher) => finish_code_leaf(hasher.clone(), entry.last_accessed_block),
        None => hash_code_reference(code_hash, entry),
    }
}

/// Hashes a code leaf without consulting any memo. The reference the memoized path must equal.
fn hash_code_reference(code_hash: B256, entry: &CachedEntry<Bytes>) -> B256 {
    let mut preimage = Vec::new();
    preimage.extend_from_slice(b"NetworkStateCacheLeaf/v1/code");
    preimage.extend_from_slice(code_hash.as_slice());
    preimage.extend_from_slice(&(entry.value.len() as u64).to_be_bytes());
    preimage.extend_from_slice(&entry.value);
    preimage.extend_from_slice(&entry.last_accessed_block.to_be_bytes());
    keccak256(preimage)
}

/// Where one `cache_root` recomputation spent its time, and over how many entries.
///
/// Computing this root is one of the two largest costs a validator pays outside execution, and it
/// used to be a single opaque timer — so the three ways to make it cheaper (reusing the leaf
/// preimage buffer, memoizing leaf digests, and keeping the keys ordered) could not be sized
/// against each other. Each namespace is reported separately because their leaf shapes, key
/// widths, and entry counts all differ. `accounts`/`storage`/`codes` are the populations these
/// times are per-entry costs over; the cost scales with them, so two runs over different blocks
/// are comparable only after dividing by them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheRootTimings {
    /// Draining each namespace's `HashMap` into a `Vec` and sorting it by key.
    pub account_collect_sort_us: u64,
    pub storage_collect_sort_us: u64,
    pub code_collect_sort_us: u64,
    /// Building each leaf preimage and hashing it. Code leaves finish a memoized sponge.
    pub account_leaf_hash_us: u64,
    pub storage_leaf_hash_us: u64,
    pub code_leaf_hash_us: u64,
    /// Concatenating a namespace's leaf digests into one buffer and hashing it.
    pub account_namespace_us: u64,
    pub storage_namespace_us: u64,
    pub code_namespace_us: u64,
    /// Hashing the final preimage over the three namespace roots and counts.
    pub root_us: u64,
    /// Entries in each namespace at the moment the root was computed.
    pub accounts: u64,
    pub storage: u64,
    pub codes: u64,
    /// True when the memo answered and no work was done, so the zero times are not a measurement.
    pub memo_hit: bool,
}

impl CacheRootTimings {
    /// What a memoized answer costs: nothing, over the population it would have hashed.
    fn memo_hit(cache: &NetworkStateCache) -> Self {
        Self {
            accounts: cache.accounts.len() as u64,
            storage: cache.storage.len() as u64,
            codes: cache.codes.len() as u64,
            memo_hit: true,
            ..Default::default()
        }
    }

    /// Collecting entries out of the hash maps and sorting them, across all three namespaces.
    pub const fn collect_sort_us(&self) -> u64 {
        self.account_collect_sort_us
            .saturating_add(self.storage_collect_sort_us)
            .saturating_add(self.code_collect_sort_us)
    }

    /// Per-leaf preimage construction and hashing, across all three namespaces.
    pub const fn leaf_hash_us(&self) -> u64 {
        self.account_leaf_hash_us
            .saturating_add(self.storage_leaf_hash_us)
            .saturating_add(self.code_leaf_hash_us)
    }

    /// Namespace digest-stream hashing, across all three namespaces.
    pub const fn namespace_hash_us(&self) -> u64 {
        self.account_namespace_us
            .saturating_add(self.storage_namespace_us)
            .saturating_add(self.code_namespace_us)
    }

    /// Sum of the measured phases. Slightly below the caller's outer timer by the timer calls.
    pub const fn total_us(&self) -> u64 {
        self.collect_sort_us()
            .saturating_add(self.leaf_hash_us())
            .saturating_add(self.namespace_hash_us())
            .saturating_add(self.root_us)
    }

    /// Leaves hashed, which is what every per-entry coefficient here divides by.
    pub const fn leaves(&self) -> u64 {
        self.accounts.saturating_add(self.storage).saturating_add(self.codes)
    }
}

/// An entry in the network state cache, tracking access metadata.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CachedEntry<T> {
    pub value: T,
    pub first_accessed_block: u64,
    pub last_accessed_block: u64,
    pub access_count: u32,
}

impl<T> CachedEntry<T> {
    fn new(value: T, block: u64) -> Self {
        Self { value, first_accessed_block: block, last_accessed_block: block, access_count: 1 }
    }

    fn touch(&mut self, block: u64) {
        self.last_accessed_block = block;
        self.access_count += 1;
    }
}

/// Statistics for a single cache update operation.
#[derive(Debug, Clone, Default)]
pub struct UpdateStats {
    /// Number of new account entries added.
    pub accounts_added: usize,
    /// Number of existing account entries refreshed (access time updated).
    pub accounts_refreshed: usize,
    /// Number of account entries evicted by policy.
    pub accounts_evicted: usize,
    /// Number of new storage entries added.
    pub storage_added: usize,
    /// Number of existing storage entries refreshed.
    pub storage_refreshed: usize,
    /// Number of storage entries evicted by policy.
    pub storage_evicted: usize,
    /// Number of new code entries added.
    pub codes_added: usize,
    /// Number of existing code entries refreshed (access time updated).
    ///
    /// Counted for the same reason as the account and storage refreshes: a code leaf preimage
    /// ends in `last_accessed_block`, so touching a code changes its leaf digest even though the
    /// bytecode and the map are unchanged.
    pub codes_refreshed: usize,
    /// Number of code entries evicted.
    pub codes_evicted: usize,
    /// Time spent bringing the leaf digest index back into agreement with the value maps.
    ///
    /// This is the cost the cache root no longer pays: rehashing the leaves a block moved, charged
    /// to the block that moved them instead of to every later root. It is measured inside the
    /// cache update, so a caller reporting both must report this one as a component and never
    /// add it to a total.
    pub index_maintenance_us: u64,
}

impl UpdateStats {
    /// Entries whose leaf digest this block invalidated, per namespace.
    ///
    /// Added and refreshed entries both need a new digest — the preimage commits
    /// `last_accessed_block`, so a read that changes no value still changes the leaf. Evictions
    /// are excluded: they remove a leaf rather than recompute one.
    pub const fn leaf_digests_invalidated(&self) -> (usize, usize, usize) {
        (
            self.accounts_added + self.accounts_refreshed,
            self.storage_added + self.storage_refreshed,
            self.codes_added + self.codes_refreshed,
        )
    }
}

/// Which keys entered or left the cache when the newest undo record's block was applied.
///
/// Derived from the undo record rather than recorded alongside it: the record already names every
/// key the block touched or evicted, so this is a read over roughly the 5% of the cache a block
/// moves rather than a second pass over the whole map. Refreshes are deliberately absent — they
/// change `last_accessed_block`, which the cache root commits, but not membership, which is the
/// only thing trie retention is a function of.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MembershipDelta {
    /// The block whose application produced this delta.
    pub block_number: u64,
    /// Accounts absent before the block and present after it.
    pub accounts_added: Vec<Address>,
    /// Accounts present before the block and absent after it, whether evicted or rolled back.
    pub accounts_removed: Vec<Address>,
    /// Storage slots absent before the block and present after it.
    pub storage_added: Vec<(Address, B256)>,
    /// Storage slots present before the block and absent after it.
    pub storage_removed: Vec<(Address, B256)>,
}

impl MembershipDelta {
    /// True when nothing entered or left, so a consumer's derived state is already correct.
    pub fn is_empty(&self) -> bool {
        self.accounts_added.is_empty() &&
            self.accounts_removed.is_empty() &&
            self.storage_added.is_empty() &&
            self.storage_removed.is_empty()
    }
}

/// Snapshot of the cache state at a point in time.
#[derive(Debug, Clone, Default)]
pub struct CacheSnapshot {
    pub total_accounts: usize,
    pub total_storage_slots: usize,
    pub total_codes: usize,
    pub current_block: u64,
}

/// Network-level state cache.
///
/// Represents the state that all validators in the network are assumed to hold.
/// When a new block arrives, state that is NOT in this cache requires a witness
/// (Merkle proof) to be transmitted as a sidecar.
pub struct NetworkStateCache {
    /// Cached accounts: address → (AccountData, metadata)
    accounts: HashMap<Address, CachedEntry<AccountData>>,
    /// Cached storage: (address, slot) → (value, metadata)
    storage: HashMap<(Address, B256), CachedEntry<U256>>,
    /// Cached bytecodes: code_hash → (bytes, metadata)
    codes: HashMap<B256, CachedEntry<Bytes>>,
    /// Eviction policy for accounts (can differ from storage policy).
    account_policy: Box<dyn CachePolicy>,
    /// Eviction policy for storage & codes (can differ from account policy).
    storage_policy: Box<dyn CachePolicy>,
    /// Current block number.
    current_block: u64,
    /// Locally derived root for the current cache contents.
    ///
    /// Block, hash, and policy context remain outside this memo and are rebound by
    /// [`cache_anchor`](Self::cache_anchor) on every call.
    memoized_cache_root: OnceLock<B256>,
    /// Per code hash, a Keccak sponge that has already absorbed that code leaf's preimage up to
    /// but not including `last_accessed_block`.
    ///
    /// Purely derived: the absorbed bytes are a function of the map key, so a missing entry is a
    /// slower recomputation rather than a wrong answer, and no rollback, reorg, or restart path
    /// has to invalidate it. That is what keeps it out of the correctness surface — contrast
    /// [`Self::memoized_cache_root`], whose value depends on cache contents and therefore does
    /// need explicit invalidation.
    code_leaf_hashers: HashMap<B256, Keccak256>,
    /// Every cache entry's leaf digest, in the key order the cache root hashes them in.
    ///
    /// Unlike [`Self::code_leaf_hashers`], losing an entry here is not merely slow: the root is
    /// read straight out of this index, so a stale digest is a wrong answer. See
    /// [`CacheDigestIndex`] for what keeps it in agreement with the value maps.
    digest_index: CacheDigestIndex,
    /// Per-block undo records (oldest→newest) enabling rollback on reorg.
    /// Retained only for the unfinalized window; pruned below the finalized block.
    undo_log: VecDeque<BlockCacheUndo>,
}

impl NetworkStateCache {
    /// Create a new cache with separate policies for accounts and storage/codes.
    pub fn new(account_policy: Box<dyn CachePolicy>, storage_policy: Box<dyn CachePolicy>) -> Self {
        Self {
            accounts: HashMap::new(),
            storage: HashMap::new(),
            codes: HashMap::new(),
            account_policy,
            storage_policy,
            current_block: 0,
            memoized_cache_root: OnceLock::new(),
            code_leaf_hashers: HashMap::new(),
            digest_index: CacheDigestIndex::default(),
            undo_log: VecDeque::new(),
        }
    }

    /// Create a new cache with the same policy applied to both accounts and storage/codes.
    pub fn with_uniform_policy(policy_fn: impl Fn() -> Box<dyn CachePolicy>) -> Self {
        Self::new(policy_fn(), policy_fn())
    }

    /// Restore a cache from previously persisted state.
    pub fn restore(
        accounts: HashMap<Address, CachedEntry<AccountData>>,
        storage: HashMap<(Address, B256), CachedEntry<U256>>,
        codes: HashMap<B256, CachedEntry<Bytes>>,
        current_block: u64,
        account_policy: Box<dyn CachePolicy>,
        storage_policy: Box<dyn CachePolicy>,
    ) -> Self {
        // Undo history is not persisted: a freshly restored cache has no rollback
        // history, so a reorg deeper than what arrives after restart triggers a
        // cold reset (see `reset`). This is safe — only accuracy of the affected
        // blocks degrades until the cache warms again.
        let code_leaf_hashers = codes
            .iter()
            .map(|(code_hash, entry)| {
                (*code_hash, code_leaf_prefix_hasher(*code_hash, &entry.value))
            })
            .collect();
        let mut cache = Self {
            accounts,
            storage,
            codes,
            account_policy,
            storage_policy,
            current_block,
            memoized_cache_root: OnceLock::new(),
            code_leaf_hashers,
            digest_index: CacheDigestIndex::default(),
            undo_log: VecDeque::new(),
        };
        // The index is not persisted either: it is a pure function of the value maps, so rebuilding
        // it is a one-off full scan at restore rather than a format the snapshot has to carry and
        // then be trusted about.
        cache.rebuild_digest_index();
        cache
    }

    /// Fork the current cache values with fresh policies and no undo history.
    ///
    /// If the source already has a locally computed cache root, the fork inherits
    /// it because the value maps and access metadata are cloned exactly. The root
    /// cannot be supplied by the caller, which prevents an unverified root from
    /// being seeded into a validation snapshot.
    pub fn fork_for_reexecution(
        &self,
        account_policy: Box<dyn CachePolicy>,
        storage_policy: Box<dyn CachePolicy>,
    ) -> Self {
        Self {
            accounts: self.accounts.clone(),
            storage: self.storage.clone(),
            codes: self.codes.clone(),
            account_policy,
            storage_policy,
            current_block: self.current_block,
            memoized_cache_root: initialized_cache_root(self.memoized_cache_root.get().copied()),
            code_leaf_hashers: self.code_leaf_hashers.clone(),
            // Cloned rather than rebuilt: the fork inherits the value maps exactly, so the digests
            // over them are already the right answer, and rebuilding would cost a full rehash on
            // the re-execution path this call exists to keep cheap.
            digest_index: self.digest_index.clone(),
            undo_log: VecDeque::new(),
        }
    }

    /// Process a new block's execution results.
    ///
    /// 1. Inserts/refreshes accessed state entries.
    /// 2. Applies eviction policies.
    /// 3. Returns update statistics.
    pub fn on_block_executed(
        &mut self,
        block_number: u64,
        accessed: &BlockAccessedState,
    ) -> UpdateStats {
        // Transfer the parent root into the undo record before changing any
        // root-bound value or access metadata. The next root is computed lazily.
        let previous_cache_root = self.memoized_cache_root.take();
        // Capture the cache's pre-block state so this block can be rolled back on
        // reorg. For every key we touch or evict we record its prior value:
        // `Some(entry)` = existed before, `None` = absent before.
        let mut undo = BlockCacheUndo::new(block_number, self.current_block, previous_cache_root);
        self.current_block = block_number;
        let mut stats = UpdateStats::default();

        // --- Insert/refresh accounts ---
        for (address, account_data) in &accessed.accounts {
            undo.accounts_before
                .entry(*address)
                .or_insert_with(|| self.accounts.get(address).cloned());
            match self.accounts.get_mut(address) {
                Some(entry) => {
                    entry.value = account_data.clone();
                    entry.touch(block_number);
                    stats.accounts_refreshed += 1;
                }
                None => {
                    self.accounts
                        .insert(*address, CachedEntry::new(account_data.clone(), block_number));
                    stats.accounts_added += 1;
                }
            }
        }

        // --- Insert/refresh storage ---
        for ((address, slot), value) in &accessed.storage {
            undo.storage_before
                .entry((*address, *slot))
                .or_insert_with(|| self.storage.get(&(*address, *slot)).cloned());
            match self.storage.get_mut(&(*address, *slot)) {
                Some(entry) => {
                    entry.value = *value;
                    entry.touch(block_number);
                    stats.storage_refreshed += 1;
                }
                None => {
                    self.storage.insert((*address, *slot), CachedEntry::new(*value, block_number));
                    stats.storage_added += 1;
                }
            }
        }

        // --- Insert/refresh codes ---
        for (code_hash, bytecode) in &accessed.codes {
            undo.codes_before
                .entry(*code_hash)
                .or_insert_with(|| self.codes.get(code_hash).cloned());
            match self.codes.get_mut(code_hash) {
                Some(entry) => {
                    entry.touch(block_number);
                    stats.codes_refreshed += 1;
                }
                None => {
                    self.codes.insert(*code_hash, CachedEntry::new(bytecode.clone(), block_number));
                    self.code_leaf_hashers
                        .entry(*code_hash)
                        .or_insert_with(|| code_leaf_prefix_hasher(*code_hash, bytecode));
                    stats.codes_added += 1;
                }
            }
        }

        // --- Apply eviction policies ---
        // The policy hands back what it removed, which is the whole record of the eviction: the
        // values are gone from the maps by the time it returns. Recovering the same set by
        // cloning all three maps first and diffing them afterwards costs three full-map copies,
        // three full scans, and three drops per block, to learn a set proportional to the ~1% of
        // the cache that actually expired.
        let evicted_accounts = self.account_policy.evict_accounts(&mut self.accounts, block_number);
        let evicted_storage =
            self.storage_policy.evict_storage(&mut self.storage, &mut self.codes, block_number);

        stats.accounts_evicted = evicted_accounts.len();
        stats.storage_evicted = evicted_storage.storage.len();
        stats.codes_evicted = evicted_storage.codes.len();

        // A key this block also touched is already recorded above with its pre-touch value, which
        // is the state a rollback has to restore. `or_insert` keeps that record rather than
        // overwriting it with the post-touch entry the policy just evicted.
        for (address, entry) in evicted_accounts {
            undo.accounts_before.entry(address).or_insert(Some(entry));
        }
        for (key, entry) in evicted_storage.storage {
            undo.storage_before.entry(key).or_insert(Some(entry));
        }
        for (code_hash, entry) in evicted_storage.codes {
            // Bound the memo to live codes. Eviction is the only path that drops a code without
            // going through rollback, which maintains the memo itself, so removing exactly the
            // evicted keys here replaces a scan over the whole memo.
            self.code_leaf_hashers.remove(&code_hash);
            undo.codes_before.entry(code_hash).or_insert(Some(entry));
        }

        // The undo record names every key this block touched or evicted, and nothing else in the
        // cache moved — an untouched entry keeps both its value and its `last_accessed_block`, so
        // its digest is still correct. Patching from the record is therefore a pass over the ~5%
        // of the cache the block moved rather than over all of it.
        let maintenance = Instant::now();
        self.resync_digest_index(
            undo.accounts_before.keys().copied(),
            undo.storage_before.keys().copied(),
            undo.codes_before.keys().copied(),
        );
        stats.index_maintenance_us = maintenance.elapsed().as_micros() as u64;

        self.undo_log.push_back(undo);

        debug!(
            target: "partial_stateless::cache",
            block = block_number,
            accounts_total = self.accounts.len(),
            storage_total = self.storage.len(),
            codes_total = self.codes.len(),
            ?stats,
            "Cache updated"
        );

        stats
    }

    /// Check if an account is in the cache.
    pub fn contains_account(&self, address: &Address) -> bool {
        self.accounts.contains_key(address)
    }

    /// Check if a storage slot is in the cache.
    pub fn contains_storage(&self, address: &Address, slot: &B256) -> bool {
        self.storage.contains_key(&(*address, *slot))
    }

    /// Check if a bytecode is in the cache.
    pub fn contains_code(&self, code_hash: &B256) -> bool {
        self.codes.contains_key(code_hash)
    }

    /// Get current cache snapshot (sizes).
    pub fn snapshot(&self) -> CacheSnapshot {
        CacheSnapshot {
            total_accounts: self.accounts.len(),
            total_storage_slots: self.storage.len(),
            total_codes: self.codes.len(),
            current_block: self.current_block,
        }
    }

    /// Compute which state from `accessed` is NOT in the cache (= needs witness).
    ///
    /// This represents what a builder would need to include in the witness sidecar.
    pub fn compute_miss(&self, accessed: &BlockAccessedState) -> MissResult {
        let mut missed_accounts: Vec<Address> = Vec::new();
        let mut missed_storage: Vec<(Address, B256)> = Vec::new();
        let mut missed_codes: Vec<B256> = Vec::new();

        for address in accessed.accounts.keys() {
            if !self.accounts.contains_key(address) {
                missed_accounts.push(*address);
            }
        }

        for (address, slot) in accessed.storage.keys() {
            if !self.storage.contains_key(&(*address, *slot)) {
                missed_storage.push((*address, *slot));
            }
        }

        for code_hash in accessed.codes.keys() {
            if !self.codes.contains_key(code_hash) {
                missed_codes.push(*code_hash);
            }
        }

        let total_accessed = accessed.total_keys();
        let total_missed = missed_accounts.len() + missed_storage.len() + missed_codes.len();
        let miss_ratio =
            if total_accessed > 0 { total_missed as f64 / total_accessed as f64 } else { 0.0 };

        MissResult {
            missed_accounts,
            missed_storage,
            missed_codes,
            total_accessed,
            total_missed,
            miss_ratio,
        }
    }

    /// Compute the canonical miss target set for `accessed` against this cache.
    pub fn expected_miss_targets(&self, accessed: &BlockAccessedState) -> StateTargetSet {
        let miss = self.compute_miss(accessed);
        let mut targets = StateTargetSet {
            accounts: miss.missed_accounts,
            storage: miss.missed_storage,
            code_hashes: miss.missed_codes,
        };
        targets.sort_dedup();
        targets
    }

    /// Get a reference to the accounts map (for persistence/inspection).
    pub fn accounts(&self) -> &HashMap<Address, CachedEntry<AccountData>> {
        &self.accounts
    }

    /// Get a reference to the storage map (for persistence/inspection).
    pub fn storage(&self) -> &HashMap<(Address, B256), CachedEntry<U256>> {
        &self.storage
    }

    /// Get a reference to the codes map (for persistence/inspection).
    pub fn codes(&self) -> &HashMap<B256, CachedEntry<Bytes>> {
        &self.codes
    }

    /// Current block number.
    pub fn current_block(&self) -> u64 {
        self.current_block
    }

    /// Compute a deterministic key+value root over the current cache contents.
    ///
    /// The protocol root includes values that affect execution or eviction:
    /// account values, storage values, bytecode values, and `last_accessed_block`.
    /// Local-only metadata such as `first_accessed_block` and `access_count` is
    /// excluded.
    pub fn cache_root(&self) -> B256 {
        self.cache_root_timed().0
    }

    /// [`Self::cache_root`], reporting where the computation spent its time.
    ///
    /// The timings describe one full recomputation. A memoized answer reports
    /// [`CacheRootTimings::memo_hit`] with zero times and the current entry counts, so an average
    /// over blocks stays honest instead of silently mixing hits into the phase cost.
    pub fn cache_root_timed(&self) -> (B256, CacheRootTimings) {
        if let Some(root) = self.memoized_cache_root.get() {
            return (*root, CacheRootTimings::memo_hit(self));
        }
        let (root, timings) = self.compute_cache_root_uncached_timed();
        (*self.memoized_cache_root.get_or_init(|| root), timings)
    }

    /// Compute the canonical root from the leaf digest index.
    ///
    /// The differential tests compare this against [`Self::compute_cache_root_reference`]; the
    /// production path reaches the same computation through [`Self::cache_root_timed`].
    #[cfg(test)]
    fn compute_cache_root_uncached(&self) -> B256 {
        self.compute_cache_root_uncached_timed().0
    }

    /// [`Self::compute_cache_root_uncached`], reporting the internal split.
    ///
    /// The collect-sort and leaf-hash timers stay zero here, and that is a measurement rather than
    /// a gap: the index is already in key order and its digests were computed by the block that
    /// last moved each entry. That work is now reported as
    /// [`UpdateStats::index_maintenance_us`] inside the cache update, so a phase table that adds
    /// the two is comparing like with like against a run that predates the index.
    fn compute_cache_root_uncached_timed(&self) -> (B256, CacheRootTimings) {
        let index = &self.digest_index;
        debug_assert_eq!(
            (index.accounts.len(), index.storage.len(), index.codes.len()),
            (self.accounts.len(), self.storage.len(), self.codes.len()),
            "digest index population disagrees with the value maps; the root would be wrong"
        );
        let mut timings = CacheRootTimings {
            accounts: index.accounts.len() as u64,
            storage: index.storage.len() as u64,
            codes: index.codes.len() as u64,
            ..Default::default()
        };

        let start = Instant::now();
        let account_root = namespace_root_streamed(b"accounts", index.accounts.values());
        timings.account_namespace_us = start.elapsed().as_micros() as u64;

        let start = Instant::now();
        let storage_root = namespace_root_streamed(b"storage", index.storage.values());
        timings.storage_namespace_us = start.elapsed().as_micros() as u64;

        let start = Instant::now();
        let code_root = namespace_root_streamed(b"codes", index.codes.values());
        timings.code_namespace_us = start.elapsed().as_micros() as u64;

        let start = Instant::now();
        let root = cache_root_v2(
            account_root,
            index.accounts.len(),
            storage_root,
            index.storage.len(),
            code_root,
            index.codes.len(),
        );
        timings.root_us = start.elapsed().as_micros() as u64;
        (root, timings)
    }

    /// Compute the canonical root from the value maps alone, consulting neither memo nor index.
    ///
    /// Deliberately a second implementation of the leaf ordering rather than a shared one. The
    /// index is derived state whose failure mode is a digest that no longer matches its entry, and
    /// an oracle that reads the index cannot see that — it would agree with the fast path by
    /// construction. Only the byte format of the commitment is shared ([`cache_root_v2`] and the
    /// `hash_*` leaf functions), because that is what must not drift between the two.
    ///
    /// Slow: it sorts the whole cache and rehashes every leaf, which is what the pre-index root
    /// cost. Used by the differential tests and available as the periodic diagnostic a run can use
    /// to prove the index has not drifted.
    pub fn compute_cache_root_reference(&self) -> B256 {
        let mut account_entries: Vec<_> = self.accounts.iter().collect();
        account_entries.sort_by_key(|(address, _)| **address);
        let account_leaves: Vec<_> =
            account_entries.iter().map(|(address, entry)| hash_account(**address, entry)).collect();

        let mut storage_entries: Vec<_> = self.storage.iter().collect();
        storage_entries.sort_by_key(|((address, slot), _)| (*address, *slot));
        let storage_leaves: Vec<_> = storage_entries
            .iter()
            .map(|((address, slot), entry)| hash_storage(*address, *slot, entry))
            .collect();

        let mut code_entries: Vec<_> = self.codes.iter().collect();
        code_entries.sort_by_key(|(code_hash, _)| **code_hash);
        let code_leaves: Vec<_> = code_entries
            .iter()
            .map(|(code_hash, entry)| hash_code_reference(**code_hash, entry))
            .collect();

        cache_root_v2(
            namespace_root(b"accounts", &account_leaves),
            account_leaves.len(),
            namespace_root(b"storage", &storage_leaves),
            storage_leaves.len(),
            namespace_root(b"codes", &code_leaves),
            code_leaves.len(),
        )
    }

    /// Bring the leaf digest index back into agreement with the value maps, for exactly the keys
    /// named.
    ///
    /// Symmetric by construction: it reads the current entry and writes what the entry says, so it
    /// does not need to know whether the caller just applied a block or just undid one. A named key
    /// that is present gets a fresh digest, a named key that is absent loses its digest, and a key
    /// that is not named is not looked at.
    ///
    /// The caller owes the one thing this cannot check: that the names cover every key whose entry
    /// changed. Both callers take them from the same undo record, which is built for exactly that
    /// purpose.
    fn resync_digest_index(
        &mut self,
        accounts: impl IntoIterator<Item = Address>,
        storage: impl IntoIterator<Item = (Address, B256)>,
        codes: impl IntoIterator<Item = B256>,
    ) {
        for address in accounts {
            match self.accounts.get(&address) {
                Some(entry) => {
                    self.digest_index.accounts.insert(address, hash_account(address, entry))
                }
                None => self.digest_index.accounts.remove(&address),
            };
        }
        for key in storage {
            match self.storage.get(&key) {
                Some(entry) => {
                    self.digest_index.storage.insert(key, hash_storage(key.0, key.1, entry))
                }
                None => self.digest_index.storage.remove(&key),
            };
        }
        for code_hash in codes {
            match self.codes.get(&code_hash) {
                Some(entry) => self
                    .digest_index
                    .codes
                    .insert(code_hash, code_leaf_digest(&self.code_leaf_hashers, code_hash, entry)),
                None => self.digest_index.codes.remove(&code_hash),
            };
        }
    }

    /// Discard the leaf digest index and recompute it over the whole cache.
    ///
    /// The cost of the pre-index root, paid once. Only for entry points that did not arrive through
    /// a block: construction from persisted maps, and the tests that check the incremental path
    /// against a full rebuild.
    fn rebuild_digest_index(&mut self) {
        self.digest_index.accounts = self
            .accounts
            .iter()
            .map(|(address, entry)| (*address, hash_account(*address, entry)))
            .collect();
        self.digest_index.storage = self
            .storage
            .iter()
            .map(|(key, entry)| (*key, hash_storage(key.0, key.1, entry)))
            .collect();
        self.digest_index.codes = self
            .codes
            .iter()
            .map(|(code_hash, entry)| {
                (*code_hash, code_leaf_digest(&self.code_leaf_hashers, *code_hash, entry))
            })
            .collect();
    }

    /// Bind the current cache root to a specific canonical block and cache policy.
    pub fn cache_anchor(
        &self,
        block_number: u64,
        block_hash: B256,
        cache_policy_id: B256,
    ) -> CacheAnchor {
        self.cache_anchor_timed(block_number, block_hash, cache_policy_id).0
    }

    /// [`Self::cache_anchor`], reporting the root computation's internal split.
    pub fn cache_anchor_timed(
        &self,
        block_number: u64,
        block_hash: B256,
        cache_policy_id: B256,
    ) -> (CacheAnchor, CacheRootTimings) {
        let (cache_root, timings) = self.cache_root_timed();
        (CacheAnchor { block_number, block_hash, cache_policy_id, cache_root }, timings)
    }

    /// Estimated memory usage in bytes.
    pub fn estimated_memory_bytes(&self) -> usize {
        // Rough estimates:
        // Account entry: 20 (address) + 8 (nonce) + 32 (balance) + 32 (code_hash) + 20 (metadata) ≈
        // 112 Storage entry: 20 (address) + 32 (slot) + 32 (value) + 20 (metadata) ≈ 104
        // Code entry: 32 (hash) + avg ~8KB (bytecode) + 20 (metadata)
        let accounts_size = self.accounts.len() * 112;
        let storage_size = self.storage.len() * 104;
        let codes_size: usize = self.codes.values().map(|e| 52 + e.value.len()).sum();
        accounts_size + storage_size + codes_size
    }

    /// Roll back the most recently applied block, restoring the cache to its exact
    /// state before that block (including values overwritten by refresh and entries
    /// removed by eviction).
    ///
    /// `block_number` must equal the newest undo record (the most recently applied
    /// block); otherwise [`CacheError::RollbackMismatch`] is returned and the caller
    /// should cold-reset via [`reset`](Self::reset). Reorgs revert blocks newest→oldest,
    /// matching the undo stack order.
    /// The membership change the most recently applied block produced, if it can still be named.
    ///
    /// `None` once the undo record is pruned below finality, or on a cache that has applied
    /// nothing — both of which mean a consumer must rebuild its derived state from scratch rather
    /// than patch it. That is the fail-safe direction: a missing delta costs a full recomputation,
    /// never a wrong one.
    pub fn last_block_membership_delta(&self) -> Option<MembershipDelta> {
        let undo = self.undo_log.back()?;
        let mut delta = MembershipDelta { block_number: undo.block_number, ..Default::default() };
        for (address, before) in &undo.accounts_before {
            match (before.is_some(), self.accounts.contains_key(address)) {
                (false, true) => delta.accounts_added.push(*address),
                (true, false) => delta.accounts_removed.push(*address),
                // Present both sides is a refresh; absent both sides is a key the block touched
                // and the same block evicted. Neither moves membership.
                _ => {}
            }
        }
        for (key, before) in &undo.storage_before {
            match (before.is_some(), self.storage.contains_key(key)) {
                (false, true) => delta.storage_added.push(*key),
                (true, false) => delta.storage_removed.push(*key),
                _ => {}
            }
        }
        Some(delta)
    }

    /// What rolling back the newest applied block would restore, without rolling anything back.
    ///
    /// Exists so a caller can make the undo a transaction. [`rollback_block`](Self::rollback_block)
    /// is the only fallible step of a depth-1 recovery, and the values it will install are already
    /// decided by the record it consumes — so a caller that checks them here can commit knowing
    /// the rollback cannot refuse, rather than discovering a mismatch with the cache half moved.
    pub fn undo_preview(&self) -> Option<UndoPreview> {
        self.undo_log.back().map(|undo| UndoPreview {
            block_number: undo.block_number,
            previous_block: undo.previous_block,
            previous_cache_root: undo.previous_cache_root,
        })
    }

    pub fn rollback_block(&mut self, block_number: u64) -> Result<(), CacheError> {
        match self.undo_log.back() {
            Some(undo) if undo.block_number == block_number => {}
            other => {
                return Err(CacheError::RollbackMismatch {
                    requested: block_number,
                    found: other.map(|u| u.block_number),
                })
            }
        }

        let undo = self.undo_log.pop_back().expect("checked non-empty above");
        let previous_cache_root = undo.previous_cache_root;
        // The same key set the forward direction patched the index with, kept as the undo record is
        // consumed. Undoing a block changes exactly the entries applying it changed, so the index
        // is resynced from the same names in both directions.
        let mut touched_accounts = Vec::with_capacity(undo.accounts_before.len());
        let mut touched_storage = Vec::with_capacity(undo.storage_before.len());
        let mut touched_codes = Vec::with_capacity(undo.codes_before.len());
        for (address, before) in undo.accounts_before {
            touched_accounts.push(address);
            match before {
                Some(entry) => {
                    self.accounts.insert(address, entry);
                }
                None => {
                    self.accounts.remove(&address);
                }
            }
        }
        for (key, before) in undo.storage_before {
            touched_storage.push(key);
            match before {
                Some(entry) => {
                    self.storage.insert(key, entry);
                }
                None => {
                    self.storage.remove(&key);
                }
            }
        }
        for (code_hash, before) in undo.codes_before {
            touched_codes.push(code_hash);
            match before {
                Some(entry) => {
                    // Restoring a code the block evicted also restores its memo, which eviction
                    // pruned. Correctness does not depend on this — a miss recomputes — but a
                    // reorg would otherwise rehash that bytecode on the next root.
                    self.code_leaf_hashers
                        .entry(code_hash)
                        .or_insert_with(|| code_leaf_prefix_hasher(code_hash, &entry.value));
                    self.codes.insert(code_hash, entry);
                }
                None => {
                    self.codes.remove(&code_hash);
                    self.code_leaf_hashers.remove(&code_hash);
                }
            }
        }
        self.resync_digest_index(touched_accounts, touched_storage, touched_codes);
        self.current_block = undo.previous_block;
        self.memoized_cache_root = initialized_cache_root(previous_cache_root);
        Ok(())
    }

    /// Drop undo records at or below `finalized_block`. Reorgs never cross a finalized
    /// block, so these records can never be needed again — this bounds undo-log memory
    /// to the unfinalized window.
    pub fn prune_undo_below(&mut self, finalized_block: u64) {
        while let Some(front) = self.undo_log.front() {
            if front.block_number <= finalized_block {
                self.undo_log.pop_front();
            } else {
                break;
            }
        }
    }

    /// Clear the entire cache and undo history (cold reset). Used to recover when a
    /// reorg is deeper than the retained undo history and cannot be rolled back; the
    /// cache is then rebuilt from the new canonical chain.
    pub fn reset(&mut self) {
        self.accounts.clear();
        self.storage.clear();
        self.codes.clear();
        self.code_leaf_hashers.clear();
        self.digest_index = CacheDigestIndex::default();
        self.undo_log.clear();
        self.current_block = 0;
        self.memoized_cache_root = OnceLock::new();
    }
}

/// What a rollback of the newest applied block would restore.
///
/// Returned by [`NetworkStateCache::undo_preview`]; every field is read straight out of the undo
/// record, so comparing them against what a recovery expects is the same check the rollback would
/// have made, moved ahead of the mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UndoPreview {
    /// The block the newest undo record can undo. A rollback of any other block is refused.
    pub block_number: u64,
    /// The height the cache returns to.
    pub previous_block: u64,
    /// The cache root the rollback reinstalls, when the parent's root had already been computed.
    ///
    /// `None` means the memo was empty when the block applied, so the root after the undo is
    /// whatever recomputing the rolled-back contents yields. It is not wrong, only unpredictable
    /// before the fact, which is the one thing a transaction needs it to be.
    pub previous_cache_root: Option<B256>,
}

/// Result of computing cache misses for a block.
#[derive(Debug, Clone)]
pub struct MissResult {
    /// Account addresses not in cache.
    pub missed_accounts: Vec<Address>,
    /// Storage slots not in cache.
    pub missed_storage: Vec<(Address, B256)>,
    /// Bytecodes not in cache.
    pub missed_codes: Vec<B256>,
    /// Total state keys accessed in the block.
    pub total_accessed: usize,
    /// Total state keys missed (not in cache).
    pub total_missed: usize,
    /// Miss ratio (0.0 = all cached, 1.0 = nothing cached).
    pub miss_ratio: f64,
}

impl MissResult {
    /// Log a summary of the miss result.
    pub fn log_summary(&self, block_number: u64) {
        info!(
            target: "partial_stateless::miss",
            block = block_number,
            total_accessed = self.total_accessed,
            total_missed = self.total_missed,
            miss_ratio = format!("{:.2}%", self.miss_ratio * 100.0),
            missed_accounts = self.missed_accounts.len(),
            missed_storage = self.missed_storage.len(),
            missed_codes = self.missed_codes.len(),
            "Witness requirement computed"
        );
    }
}

/// Undo record for a single applied block, enabling rollback on reorg.
///
/// Each `*_before` map stores the value of a touched or evicted key *before* the
/// block was applied: `Some(entry)` = the key existed (restore it on rollback),
/// `None` = the key was absent (remove it on rollback).
#[derive(Debug, Clone)]
struct BlockCacheUndo {
    /// The block this record can undo.
    block_number: u64,
    /// Cache `current_block` before this block was applied (restored on rollback).
    previous_block: u64,
    /// Locally computed root before this block, if it had already been derived.
    previous_cache_root: Option<B256>,
    accounts_before: HashMap<Address, Option<CachedEntry<AccountData>>>,
    storage_before: HashMap<(Address, B256), Option<CachedEntry<U256>>>,
    codes_before: HashMap<B256, Option<CachedEntry<Bytes>>>,
}

impl BlockCacheUndo {
    fn new(block_number: u64, previous_block: u64, previous_cache_root: Option<B256>) -> Self {
        Self {
            block_number,
            previous_block,
            previous_cache_root,
            accounts_before: HashMap::new(),
            storage_before: HashMap::new(),
            codes_before: HashMap::new(),
        }
    }
}

/// Every cache entry's leaf digest, held in the key order the cache root hashes them in.
///
/// The root used to sort the whole cache and rehash every leaf on every block, but a block moves
/// about 4.6% of the entries as measured over 300 live samples — so all but that fraction of both
/// terms was recomputing an answer that had not changed. Holding the digests in order turns the
/// root into one pass over already-ordered bytes and charges the leaf hashing to the block that
/// moved the entry.
///
/// `BTreeMap` rather than a sorted `Vec`: at this composition the three containers land within 2%
/// of each other and all of them clear the budget, so the choice was made on correctness surface
/// instead. Point mutation is the same eight lines forwards and backwards, which is what the
/// rollback invariant needs, where a merge into a sorted `Vec` is a two-pointer walk with three
/// tail cases that has to be exactly right in both directions. The measured price is a 39% memory
/// overhead on the node pointers, about 6.7 MiB — 0.16% of the process.
///
/// This is not [`NetworkStateCache::code_leaf_hashers`]. That memo is a pure function of its key,
/// so losing an entry costs a rehash; a stale entry here is a wrong root.
#[derive(Debug, Clone, Default)]
struct CacheDigestIndex {
    accounts: BTreeMap<Address, B256>,
    storage: BTreeMap<(Address, B256), B256>,
    codes: BTreeMap<B256, B256>,
}

/// Errors returned by reorg-related cache operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheError {
    /// Rollback was requested for a block that is not the newest undo record.
    /// `found` is the newest retained undo block (or `None` if no history remains).
    RollbackMismatch { requested: u64, found: Option<u64> },
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CacheError::RollbackMismatch { requested, found } => write!(
                f,
                "cache rollback mismatch: requested block {requested}, newest undo record is {found:?}"
            ),
        }
    }
}

impl std::error::Error for CacheError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        policy::{EvictedStorage, LastNBlocksPolicy},
        sidecar::last_n_blocks_cache_policy_id,
    };
    use alloy_primitives::b256;

    fn make_cache(account_window: u64, storage_window: u64) -> NetworkStateCache {
        NetworkStateCache::new(
            Box::new(LastNBlocksPolicy::new(account_window)),
            Box::new(LastNBlocksPolicy::new(storage_window)),
        )
    }

    /// An unchanged re-read still moves the cache root, and `UpdateStats` still counts it.
    ///
    /// Nothing about the entry changes except `last_accessed_block`, which every leaf preimage
    /// commits, so the populations are identical block to block while every touched leaf has a
    /// new digest. Anything deriving per-entry state from this cache therefore has to invalidate
    /// on the touch rather than on the value, and `MembershipDelta` is the wrong source for it —
    /// it tracks membership, which a refresh does not move.
    #[test]
    fn an_unchanged_reread_still_invalidates_every_touched_leaf() {
        let mut cache = make_cache(60, 60);
        let address = Address::repeat_byte(0x11);
        let slot = B256::repeat_byte(0x22);
        let code = Bytes::from_static(b"runtime bytecode");
        let code_hash = keccak256(&code);

        let mut accessed = BlockAccessedState::default();
        accessed.accounts.insert(
            address,
            AccountData { nonce: 7, balance: U256::from(1234), code_hash: Some(code_hash) },
        );
        accessed.storage.insert((address, slot), U256::from(99));
        accessed.codes.insert(code_hash, code);

        cache.on_block_executed(100, &accessed);
        let root_after_insert = cache.cache_root();
        let population = (cache.accounts.len(), cache.storage.len(), cache.codes.len());

        // Same block content, replayed at the next height: identical values, nothing added or
        // evicted, so every population count is unmoved.
        let stats = cache.on_block_executed(101, &accessed);
        assert_eq!(
            (cache.accounts.len(), cache.storage.len(), cache.codes.len()),
            population,
            "an unchanged re-read must not move any population count"
        );
        assert_eq!(
            (stats.accounts_added, stats.storage_added, stats.codes_added),
            (0, 0, 0),
            "nothing was added"
        );
        assert_eq!(
            (stats.accounts_refreshed, stats.storage_refreshed, stats.codes_refreshed),
            (1, 1, 1),
            "every namespace, code included, must count the refresh"
        );
        assert_eq!(stats.leaf_digests_invalidated(), (1, 1, 1));

        assert_ne!(
            cache.cache_root(),
            root_after_insert,
            "last_accessed_block is in every leaf preimage, so a pure re-read moves the root"
        );

        // And the undo record — the memo's invalidation source — names all three keys.
        let undo = cache.undo_log.back().expect("the block left an undo record");
        assert!(undo.accounts_before.contains_key(&address));
        assert!(undo.storage_before.contains_key(&(address, slot)));
        assert!(undo.codes_before.contains_key(&code_hash));
    }

    #[test]
    fn test_basic_insert_and_lookup() {
        let mut cache = make_cache(10, 10);
        let addr = Address::repeat_byte(0x01);
        let slot = B256::repeat_byte(0x02);

        let mut accessed = BlockAccessedState::default();
        accessed
            .accounts
            .insert(addr, AccountData { nonce: 1, balance: U256::from(1000), code_hash: None });
        accessed.storage.insert((addr, slot), U256::from(42));

        cache.on_block_executed(100, &accessed);

        assert!(cache.contains_account(&addr));
        assert!(cache.contains_storage(&addr, &slot));
        assert!(!cache.contains_account(&Address::repeat_byte(0xFF)));
    }

    #[test]
    fn test_eviction_after_window() {
        let mut cache = make_cache(5, 3);
        let addr = Address::repeat_byte(0x01);
        let slot = B256::repeat_byte(0x02);

        // Block 10: insert both
        let mut accessed = BlockAccessedState::default();
        accessed
            .accounts
            .insert(addr, AccountData { nonce: 1, balance: U256::from(100), code_hash: None });
        accessed.storage.insert((addr, slot), U256::from(1));
        cache.on_block_executed(10, &accessed);

        // Block 13: storage window=3, cutoff=10, so block 10 entry is still at boundary
        cache.on_block_executed(13, &BlockAccessedState::default());
        assert!(cache.contains_account(&addr));
        assert!(cache.contains_storage(&addr, &slot));

        // Block 14: storage cutoff=11, evicts the slot (last accessed at 10)
        cache.on_block_executed(14, &BlockAccessedState::default());
        assert!(cache.contains_account(&addr)); // account window=5, cutoff=9
        assert!(!cache.contains_storage(&addr, &slot)); // evicted!

        // Block 16: account cutoff=11, evicts the account
        cache.on_block_executed(16, &BlockAccessedState::default());
        assert!(!cache.contains_account(&addr));
    }

    #[test]
    fn test_miss_computation() {
        let mut cache = make_cache(10, 10);
        let addr_cached = Address::repeat_byte(0x01);
        let addr_missed = Address::repeat_byte(0x02);
        let slot_cached = B256::repeat_byte(0xAA);
        let slot_missed = B256::repeat_byte(0xBB);

        // Pre-populate cache
        let mut pre = BlockAccessedState::default();
        pre.accounts.insert(
            addr_cached,
            AccountData { nonce: 1, balance: U256::from(100), code_hash: None },
        );
        pre.storage.insert((addr_cached, slot_cached), U256::from(1));
        cache.on_block_executed(100, &pre);

        // New block accesses both cached and uncached state
        let mut new_block = BlockAccessedState::default();
        new_block.accounts.insert(
            addr_cached,
            AccountData { nonce: 1, balance: U256::from(100), code_hash: None },
        );
        new_block
            .accounts
            .insert(addr_missed, AccountData { nonce: 0, balance: U256::ZERO, code_hash: None });
        new_block.storage.insert((addr_cached, slot_cached), U256::from(1));
        new_block.storage.insert((addr_cached, slot_missed), U256::from(2));

        let miss = cache.compute_miss(&new_block);

        // addr_cached is in cache, addr_missed is not
        assert_eq!(miss.missed_accounts.len(), 1);
        assert_eq!(miss.missed_accounts[0], addr_missed);

        // slot_cached is in cache, slot_missed is not
        assert_eq!(miss.missed_storage.len(), 1);
        assert_eq!(miss.missed_storage[0], (addr_cached, slot_missed));

        // 4 total keys, 2 missed
        assert_eq!(miss.total_accessed, 4);
        assert_eq!(miss.total_missed, 2);
        assert!((miss.miss_ratio - 0.5).abs() < 0.001);
    }

    #[test]
    fn cache_root_is_deterministic_and_key_value_bound() {
        let addr = Address::repeat_byte(0x01);
        let slot = B256::repeat_byte(0x02);
        let code_hash = B256::repeat_byte(0x03);

        let mut accessed = BlockAccessedState::default();
        accessed.accounts.insert(
            addr,
            AccountData { nonce: 1, balance: U256::from(100), code_hash: Some(code_hash) },
        );
        accessed.storage.insert((addr, slot), U256::from(42));
        accessed.codes.insert(code_hash, Bytes::from(vec![1, 2, 3]));

        let mut cache_a = make_cache(10, 10);
        cache_a.on_block_executed(100, &accessed);

        let mut cache_b = make_cache(10, 10);
        cache_b.on_block_executed(100, &accessed);

        assert_eq!(cache_a.cache_root(), cache_b.cache_root());

        let mut changed = accessed.clone();
        changed.accounts.get_mut(&addr).unwrap().balance = U256::from(101);
        let mut cache_c = make_cache(10, 10);
        cache_c.on_block_executed(100, &changed);

        assert_ne!(cache_a.cache_root(), cache_c.cache_root());
    }

    #[test]
    fn cache_root_is_memoized_and_invalidated_by_cache_updates() {
        let mut cache = make_cache(2, 2);
        assert!(cache.memoized_cache_root.get().is_none());

        let empty_root = cache.cache_root();
        assert_eq!(cache.memoized_cache_root.get().copied(), Some(empty_root));
        assert_eq!(cache.cache_root(), empty_root);

        let address = Address::repeat_byte(0x01);
        let slot = B256::repeat_byte(0x02);
        let code_hash = B256::repeat_byte(0x03);
        let mut inserted = BlockAccessedState::default();
        inserted.accounts.insert(
            address,
            AccountData { nonce: 1, balance: U256::from(10), code_hash: Some(code_hash) },
        );
        inserted.storage.insert((address, slot), U256::from(20));
        inserted.codes.insert(code_hash, Bytes::from_static(&[1, 2, 3]));

        cache.on_block_executed(1, &inserted);
        assert!(cache.memoized_cache_root.get().is_none());
        let inserted_root = cache.cache_root();
        assert_eq!(inserted_root, cache.compute_cache_root_uncached());
        assert_ne!(inserted_root, empty_root);

        let mut refreshed = inserted;
        refreshed.accounts.get_mut(&address).unwrap().balance = U256::from(11);
        refreshed.storage.insert((address, slot), U256::from(21));
        cache.on_block_executed(2, &refreshed);
        assert!(cache.memoized_cache_root.get().is_none());
        let refreshed_root = cache.cache_root();
        assert_eq!(refreshed_root, cache.compute_cache_root_uncached());
        assert_ne!(refreshed_root, inserted_root);

        cache.on_block_executed(5, &BlockAccessedState::default());
        assert!(cache.memoized_cache_root.get().is_none());
        let evicted_root = cache.cache_root();
        assert_eq!(evicted_root, cache.compute_cache_root_uncached());
        assert_eq!(evicted_root, empty_root);
    }

    #[test]
    fn an_undo_preview_names_what_the_rollback_would_restore() {
        let mut cache = make_cache(10, 10);
        assert!(cache.undo_preview().is_none(), "nothing applied, nothing to give back");

        let mut parent_accessed = BlockAccessedState::default();
        parent_accessed.accounts.insert(
            Address::repeat_byte(0x01),
            AccountData { nonce: 1, balance: U256::from(10), code_hash: None },
        );
        cache.on_block_executed(10, &parent_accessed);
        let parent_root = cache.cache_root();

        let mut child_accessed = BlockAccessedState::default();
        child_accessed.accounts.insert(
            Address::repeat_byte(0x02),
            AccountData { nonce: 2, balance: U256::from(20), code_hash: None },
        );
        cache.on_block_executed(11, &child_accessed);

        let preview = cache.undo_preview().expect("a block was applied");
        assert_eq!(preview.block_number, 11);
        assert_eq!(preview.previous_block, 10);
        assert_eq!(preview.previous_cache_root, Some(parent_root));

        cache.rollback_block(11).unwrap();
        assert_eq!(cache.current_block(), preview.previous_block, "the preview was exact");
        assert_eq!(cache.cache_root(), parent_root);
    }

    #[test]
    fn an_undo_preview_reports_an_unrooted_parent_as_unpredictable() {
        let mut cache = make_cache(10, 10);
        let mut accessed = BlockAccessedState::default();
        accessed.accounts.insert(
            Address::repeat_byte(0x01),
            AccountData { nonce: 1, balance: U256::from(10), code_hash: None },
        );
        // No `cache_root()` between the two blocks, so the memo is empty when the second applies.
        cache.on_block_executed(10, &accessed);
        cache.on_block_executed(11, &accessed);

        let preview = cache.undo_preview().expect("a block was applied");
        assert_eq!(preview.previous_block, 10);
        assert_eq!(
            preview.previous_cache_root, None,
            "the root after the undo is honest but not knowable before it"
        );
    }

    #[test]
    fn cache_root_memo_is_restored_by_rollback() {
        let mut cache = make_cache(10, 10);
        let mut parent_accessed = BlockAccessedState::default();
        parent_accessed.accounts.insert(
            Address::repeat_byte(0x01),
            AccountData { nonce: 1, balance: U256::from(10), code_hash: None },
        );
        cache.on_block_executed(10, &parent_accessed);
        let parent_root = cache.cache_root();

        let mut child_accessed = BlockAccessedState::default();
        child_accessed.accounts.insert(
            Address::repeat_byte(0x02),
            AccountData { nonce: 2, balance: U256::from(20), code_hash: None },
        );
        cache.on_block_executed(11, &child_accessed);
        assert!(cache.memoized_cache_root.get().is_none());
        let child_root = cache.cache_root();
        assert_ne!(child_root, parent_root);

        cache.rollback_block(11).unwrap();
        assert_eq!(cache.memoized_cache_root.get().copied(), Some(parent_root));
        assert_eq!(cache.cache_root(), parent_root);
        assert_eq!(cache.cache_root(), cache.compute_cache_root_uncached());
    }

    /// Bytecode of a distinctive length, keyed by its real hash as the cache's invariants require.
    fn code(byte: u8, len: usize) -> (B256, Bytes) {
        let bytes = Bytes::from(vec![byte; len]);
        (keccak256(&bytes), bytes)
    }

    fn accessed_with_code(code_hash: B256, bytes: Bytes) -> BlockAccessedState {
        let mut accessed = BlockAccessedState::default();
        accessed.codes.insert(code_hash, bytes);
        accessed
    }

    /// The memoized code leaf must be bit-identical to hashing the whole preimage, at every point
    /// in a code entry's life: insertion, refresh at a new height, eviction, and rollback.
    #[test]
    fn code_leaf_memo_equals_slow_reference_across_the_entry_lifecycle() {
        let mut cache = make_cache(10, 3);
        let (first_hash, first_bytes) = code(0xab, 4096);
        let (second_hash, second_bytes) = code(0xcd, 1);

        // Empty cache: the code namespace is hashed over zero leaves either way.
        assert_eq!(cache.compute_cache_root_uncached(), cache.compute_cache_root_reference());

        cache.on_block_executed(10, &accessed_with_code(first_hash, first_bytes.clone()));
        assert_eq!(cache.compute_cache_root_uncached(), cache.compute_cache_root_reference());

        // Refresh: only `last_accessed_block` moves, which is exactly the suffix the memo does not
        // absorb. A memo that had baked it in would fail here.
        let root_at_10 = cache.cache_root();
        cache.on_block_executed(11, &accessed_with_code(first_hash, first_bytes.clone()));
        assert_ne!(cache.cache_root(), root_at_10);
        assert_eq!(cache.compute_cache_root_uncached(), cache.compute_cache_root_reference());

        // A second code with a very different length exercises a different sponge block boundary.
        cache.on_block_executed(12, &accessed_with_code(second_hash, second_bytes));
        assert_eq!(cache.compute_cache_root_uncached(), cache.compute_cache_root_reference());

        // Storage window 3 evicts the first code at block 15 (cutoff 12, last accessed 11).
        cache.on_block_executed(15, &BlockAccessedState::default());
        assert!(!cache.contains_code(&first_hash));
        assert!(cache.contains_code(&second_hash));
        assert_eq!(cache.compute_cache_root_uncached(), cache.compute_cache_root_reference());

        // Rolling that eviction back restores the entry, and the root must return to what the
        // reference says about the restored contents.
        cache.rollback_block(15).unwrap();
        assert!(cache.contains_code(&first_hash));
        assert_eq!(cache.compute_cache_root_uncached(), cache.compute_cache_root_reference());
    }

    /// Populate a cache with `accounts` accounts, `slots` slots each, and one shared code.
    fn populated_cache(accounts: u8, slots: u8) -> NetworkStateCache {
        let mut cache = make_cache(1000, 1000);
        let (code_hash, bytes) = code(0x5a, 512);
        let mut accessed = accessed_with_code(code_hash, bytes);
        for account in 0..accounts {
            let address = Address::repeat_byte(account);
            accessed.accounts.insert(
                address,
                AccountData {
                    nonce: u64::from(account),
                    balance: U256::from(account),
                    code_hash: Some(code_hash),
                },
            );
            for slot in 0..slots {
                accessed
                    .storage
                    .insert((address, B256::repeat_byte(slot)), U256::from(u16::from(slot) + 1));
            }
        }
        cache.on_block_executed(10, &accessed);
        cache
    }

    /// Instrumenting the root must not change it, and the counts must describe what it hashed.
    ///
    /// The split exists to size the candidate optimizations against each other, so a count that
    /// disagrees with the maps would silently corrupt every per-entry coefficient derived from it.
    #[test]
    fn timed_cache_root_equals_the_untimed_root_and_counts_what_it_hashed() {
        let cache = populated_cache(24, 5);
        let (timed_root, timings) = cache.cache_root_timed();

        assert_eq!(timed_root, cache.compute_cache_root_reference());
        assert!(!timings.memo_hit);
        assert_eq!(timings.accounts, cache.accounts.len() as u64);
        assert_eq!(timings.storage, cache.storage.len() as u64);
        assert_eq!(timings.codes, cache.codes.len() as u64);
        assert_eq!(timings.leaves(), timings.accounts + timings.storage + timings.codes);
        assert_eq!(
            timings.total_us(),
            timings.collect_sort_us() +
                timings.leaf_hash_us() +
                timings.namespace_hash_us() +
                timings.root_us
        );
    }

    /// A memo hit is free, and must say so rather than report a zero measurement.
    ///
    /// The validator path never takes this branch — the cache update immediately before the anchor
    /// invalidates the memo — but averaging a hit into the phase mean as a real zero would
    /// understate it, so the flag is what the analyzer excludes on.
    #[test]
    fn a_memoized_cache_root_reports_a_memo_hit_with_the_current_composition() {
        let cache = populated_cache(8, 2);
        let (first_root, first) = cache.cache_root_timed();
        let (second_root, second) = cache.cache_root_timed();

        assert_eq!(first_root, second_root);
        assert!(!first.memo_hit);
        assert!(second.memo_hit);
        assert_eq!(second.total_us(), 0);
        assert_eq!(
            (second.accounts, second.storage, second.codes),
            (first.accounts, first.storage, first.codes)
        );
    }

    /// The memo is derived state, so discarding it may cost time but must not change the root.
    ///
    /// The index has to be rebuilt alongside it for this to test anything: the root reads the code
    /// digest out of the index, so an index left in place would answer without ever consulting the
    /// memo and the assertion would hold no matter what the memo did.
    #[test]
    fn cache_root_is_unchanged_when_the_code_leaf_memo_is_absent() {
        let mut cache = make_cache(10, 10);
        let (code_hash, bytes) = code(0x7f, 2048);
        cache.on_block_executed(10, &accessed_with_code(code_hash, bytes));
        let memoized_root = cache.cache_root();

        cache.code_leaf_hashers.clear();
        cache.rebuild_digest_index();
        assert_eq!(cache.compute_cache_root_uncached(), memoized_root);
        assert_eq!(cache.compute_cache_root_reference(), memoized_root);
    }

    /// The memo must track the codes map rather than accumulate for the life of the process.
    #[test]
    fn code_leaf_memo_follows_code_membership() {
        let mut cache = make_cache(10, 2);
        let (code_hash, bytes) = code(0x11, 64);

        cache.on_block_executed(10, &accessed_with_code(code_hash, bytes.clone()));
        assert_eq!(cache.code_leaf_hashers.len(), 1);

        // Cutoff 11 at block 13 evicts the entry, and the memo is pruned with it.
        cache.on_block_executed(13, &BlockAccessedState::default());
        assert!(!cache.contains_code(&code_hash));
        assert!(cache.code_leaf_hashers.is_empty());

        // Rollback restores both, so a reorg does not leave the next root rehashing the bytecode.
        cache.rollback_block(13).unwrap();
        assert!(cache.contains_code(&code_hash));
        assert_eq!(cache.code_leaf_hashers.len(), 1);
        assert_eq!(cache.compute_cache_root_uncached(), cache.compute_cache_root_reference());
    }

    /// The index must equal a rebuild from the value maps, and the root it produces must equal the
    /// reference. Checked at a named point so a failure says which operation broke it.
    fn assert_digest_index_is_exact(cache: &NetworkStateCache, at: &str) {
        let mut rebuilt = cache.fork_for_reexecution(
            Box::new(LastNBlocksPolicy::new(1)),
            Box::new(LastNBlocksPolicy::new(1)),
        );
        rebuilt.rebuild_digest_index();
        assert_eq!(cache.digest_index.accounts, rebuilt.digest_index.accounts, "accounts, {at}");
        assert_eq!(cache.digest_index.storage, rebuilt.digest_index.storage, "storage, {at}");
        assert_eq!(cache.digest_index.codes, rebuilt.digest_index.codes, "codes, {at}");
        assert_eq!(
            cache.compute_cache_root_uncached(),
            cache.compute_cache_root_reference(),
            "root, {at}"
        );
    }

    /// The index is patched from the undo record at every point that moves an entry, so this walks
    /// an entry through all of them: insertion, refresh at a new height, eviction, rollback of that
    /// eviction, and the touch-and-evict-in-the-same-block case where a key appears in the undo
    /// record but not in the cache afterwards.
    #[test]
    fn digest_index_equals_a_full_rebuild_across_the_entry_lifecycle() {
        let mut cache = make_cache(10, 3);
        assert_digest_index_is_exact(&cache, "empty");

        let address = Address::repeat_byte(0x41);
        let slot = B256::repeat_byte(0x42);
        let (code_hash, bytes) = code(0x43, 1024);
        let mut accessed = BlockAccessedState::default();
        accessed.accounts.insert(address, account(1, 100));
        accessed.storage.insert((address, slot), U256::from(7));
        accessed.codes.insert(code_hash, bytes);

        cache.on_block_executed(10, &accessed);
        assert_digest_index_is_exact(&cache, "after insertion");

        // A refresh moves only `last_accessed_block`, which every leaf preimage commits, so the
        // digests must all change while the populations do not.
        let before_refresh = cache.digest_index.clone();
        cache.on_block_executed(11, &accessed);
        assert_digest_index_is_exact(&cache, "after refresh");
        assert_ne!(cache.digest_index.accounts, before_refresh.accounts, "a touch is a new digest");
        assert_ne!(cache.digest_index.storage, before_refresh.storage);
        assert_ne!(cache.digest_index.codes, before_refresh.codes);

        // Storage window 3 evicts the slot and the code at block 15 (cutoff 12, last accessed 11);
        // the account's window of 10 keeps it.
        cache.on_block_executed(15, &BlockAccessedState::default());
        assert!(!cache.contains_storage(&address, &slot));
        assert!(!cache.contains_code(&code_hash));
        assert!(cache.contains_account(&address));
        assert_digest_index_is_exact(&cache, "after eviction");

        cache.rollback_block(15).unwrap();
        assert!(cache.contains_storage(&address, &slot));
        assert_digest_index_is_exact(&cache, "after rolling the eviction back");

        cache.reset();
        assert!(cache.digest_index.accounts.is_empty());
        assert!(cache.digest_index.storage.is_empty());
        assert!(cache.digest_index.codes.is_empty());
        assert_digest_index_is_exact(&cache, "after reset");
    }

    /// Evicts the whole cache on every block, whatever it was just accessed at.
    ///
    /// [`LastNBlocksPolicy`] keeps anything the current block touched, so under it a key can never
    /// be named by the undo record and be absent from the cache at the same time. That is a
    /// property of the policy rather than of the cache, and a size-capped policy would not have it,
    /// so this exists to reach the branch anyway.
    struct EvictEverything;

    impl CachePolicy for EvictEverything {
        fn evict_accounts(
            &self,
            accounts: &mut HashMap<Address, CachedEntry<AccountData>>,
            _current_block: u64,
        ) -> Vec<(Address, CachedEntry<AccountData>)> {
            accounts.drain().collect()
        }

        fn evict_storage(
            &self,
            storage: &mut HashMap<(Address, B256), CachedEntry<U256>>,
            codes: &mut HashMap<B256, CachedEntry<Bytes>>,
            _current_block: u64,
        ) -> EvictedStorage {
            EvictedStorage { storage: storage.drain().collect(), codes: codes.drain().collect() }
        }

        fn name(&self) -> &str {
            "evict-everything"
        }
    }

    /// A key the block touched and the same block evicted is named by the undo record but absent
    /// from the cache when the index is patched, so the patch has to remove it rather than rehash
    /// it. Getting this backwards would leave a digest behind for an entry that no longer exists.
    #[test]
    fn a_key_touched_and_evicted_by_the_same_block_leaves_no_digest() {
        let mut cache =
            NetworkStateCache::new(Box::new(EvictEverything), Box::new(EvictEverything));
        let address = Address::repeat_byte(0x51);
        let (code_hash, bytes) = code(0x52, 128);
        let mut accessed = BlockAccessedState::default();
        accessed.accounts.insert(address, account(1, 100));
        accessed.storage.insert((address, B256::repeat_byte(0x53)), U256::from(5));
        accessed.codes.insert(code_hash, bytes);

        let stats = cache.on_block_executed(10, &accessed);
        assert_eq!((stats.accounts_added, stats.accounts_evicted), (1, 1), "added then evicted");
        assert!(cache.accounts.is_empty() && cache.storage.is_empty() && cache.codes.is_empty());
        assert!(cache.digest_index.accounts.is_empty(), "the digest must go with the entry");
        assert!(cache.digest_index.storage.is_empty());
        assert!(cache.digest_index.codes.is_empty());
        assert_digest_index_is_exact(&cache, "after a touch the same block evicted");

        // Rolling it back restores nothing, because nothing survived the block.
        cache.rollback_block(10).unwrap();
        assert_digest_index_is_exact(&cache, "after rolling that block back");
    }

    /// A restored cache rebuilds the index from the persisted maps; a fork inherits it.
    #[test]
    fn digest_index_survives_restore_and_fork() {
        let cache = populated_cache(12, 3);
        let root = cache.compute_cache_root_reference();

        let fork = cache.fork_for_reexecution(
            Box::new(LastNBlocksPolicy::new(1000)),
            Box::new(LastNBlocksPolicy::new(1000)),
        );
        assert_eq!(fork.digest_index.accounts, cache.digest_index.accounts);
        assert_eq!(fork.compute_cache_root_uncached(), root);
        assert_digest_index_is_exact(&fork, "fork");

        let restored = NetworkStateCache::restore(
            cache.accounts.clone(),
            cache.storage.clone(),
            cache.codes.clone(),
            cache.current_block,
            Box::new(LastNBlocksPolicy::new(1000)),
            Box::new(LastNBlocksPolicy::new(1000)),
        );
        assert_eq!(restored.digest_index.accounts, cache.digest_index.accounts);
        assert_eq!(restored.compute_cache_root_uncached(), root);
        assert_digest_index_is_exact(&restored, "restore");
    }

    /// Pruning drops the ability to roll a block back. It must not drop the digests of the entries
    /// that block left behind, which are still part of the current cache.
    #[test]
    fn prune_undo_below_does_not_disturb_the_digest_index() {
        let mut cache = populated_cache(8, 2);
        cache.on_block_executed(11, &BlockAccessedState::default());
        let before = cache.digest_index.clone();

        cache.prune_undo_below(11);
        assert!(cache.undo_log.is_empty());
        assert_eq!(cache.digest_index.accounts, before.accounts);
        assert_eq!(cache.digest_index.storage, before.storage);
        assert_eq!(cache.digest_index.codes, before.codes);
        assert_digest_index_is_exact(&cache, "after pruning the undo log");
    }

    /// Streaming a namespace's digests into the sponge must be bit-identical to hashing the
    /// concatenation, which is what lets the index skip materializing the leaves into a slice.
    #[test]
    fn the_streamed_namespace_root_matches_the_buffered_form() {
        for count in [0usize, 1, 5, 300] {
            let leaves: Vec<B256> =
                (0..count).map(|i| keccak256((i as u64).to_be_bytes())).collect();
            assert_eq!(
                namespace_root(b"accounts", &leaves),
                namespace_root_streamed(b"accounts", leaves.iter()),
                "{count} leaves"
            );
        }
        // The label is part of the preimage in both forms, so two namespaces of the same leaves
        // must not collide.
        assert_ne!(
            namespace_root_streamed(b"accounts", [].iter()),
            namespace_root(b"storage", &[])
        );
    }

    /// The slow reference has to be able to disagree with the index, or it is not an oracle.
    ///
    /// This is the reason [`NetworkStateCache::compute_cache_root_reference`] duplicates the leaf
    /// ordering instead of sharing it: the index's failure mode is a digest that no longer matches
    /// its entry, and a reference that read the index would agree by construction and see nothing.
    #[test]
    fn the_slow_reference_disagrees_when_the_index_is_wrong() {
        let mut cache = populated_cache(6, 2);
        assert_eq!(cache.compute_cache_root_uncached(), cache.compute_cache_root_reference());

        let address = *cache.digest_index.accounts.keys().next().expect("populated");
        cache.digest_index.accounts.insert(address, B256::repeat_byte(0xff));
        assert_ne!(
            cache.compute_cache_root_uncached(),
            cache.compute_cache_root_reference(),
            "a corrupted digest must be visible to the reference"
        );

        cache.rebuild_digest_index();
        assert_eq!(cache.compute_cache_root_uncached(), cache.compute_cache_root_reference());
    }

    /// A deterministic 32-bit xorshift. The sequence only has to shuffle which keys a block
    /// touches and be identical on every machine; it is not used for anything statistical.
    struct Xorshift(u32);

    impl Xorshift {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            self.0
        }

        fn below(&mut self, bound: u32) -> u32 {
            self.next() % bound
        }
    }

    /// The eviction policy is the sole record of what it removed, and the undo record is the sole
    /// record of the values, so a policy that drops an entry without reporting it makes the block
    /// unrollbackable. Apply a block, roll it straight back, and require the whole cache to return.
    #[test]
    fn rollback_restores_every_entry_eviction_removed() {
        // Windows this narrow expire entries within a handful of rounds, which is the point:
        // eviction is the only path that removes an entry the block never touched.
        let mut cache = make_cache(3, 2);
        let mut rng = Xorshift(0x5eed_1234);
        let mut evictions_seen = 0usize;

        for round in 0..40u64 {
            let block = 100 + round;
            let mut accessed = BlockAccessedState::default();
            for _ in 0..(1 + rng.below(8)) {
                let address = Address::repeat_byte(rng.below(20) as u8);
                accessed
                    .accounts
                    .insert(address, account(u64::from(rng.below(50)), u64::from(rng.below(1000))));
                for _ in 0..rng.below(4) {
                    accessed.storage.insert(
                        (address, B256::repeat_byte(rng.below(8) as u8)),
                        U256::from(rng.below(10_000)),
                    );
                }
            }
            for _ in 0..(1 + rng.below(3)) {
                let (code_hash, bytes) = code(rng.below(7) as u8, 32 + rng.below(200) as usize);
                accessed.codes.insert(code_hash, bytes);
            }

            let accounts_before = cache.accounts.clone();
            let storage_before = cache.storage.clone();
            let codes_before = cache.codes.clone();
            let root_before = cache.compute_cache_root_reference();

            let stats = cache.on_block_executed(block, &accessed);
            evictions_seen += stats.accounts_evicted + stats.storage_evicted + stats.codes_evicted;

            cache.rollback_block(block).expect("newest undo record");

            // Whole-map equality rather than a digest-index check: an eviction the undo record
            // failed to name leaves the cache self-consistent but short one entry, and the index
            // assertions cannot see that because the map and the index would both lack it.
            assert_eq!(cache.accounts, accounts_before, "round {round}: accounts after rollback");
            assert_eq!(cache.storage, storage_before, "round {round}: storage after rollback");
            assert_eq!(cache.codes, codes_before, "round {round}: codes after rollback");
            assert_eq!(
                cache.compute_cache_root_reference(),
                root_before,
                "round {round}: root recomputed from the restored maps"
            );

            // Re-apply so the next round starts from a cache that has actually moved.
            cache.on_block_executed(block, &accessed);
            assert_digest_index_is_exact(&cache, &format!("round {round}, reapplied {block}"));
        }

        assert!(
            evictions_seen > 0,
            "the windows must be narrow enough to exercise eviction at all"
        );
    }

    /// The lifecycle test walks the transitions one at a time; this one interleaves them.
    ///
    /// Eviction windows are small enough that entries leave on their own, blocks overlap in the
    /// keys they touch, and every few blocks a reorg rolls back a run of them and applies different
    /// ones at the same heights. The index is checked against a full rebuild after every single
    /// step, so the first block that breaks it is the one that fails.
    #[test]
    fn a_randomized_block_and_reorg_sequence_keeps_the_index_exact() {
        let mut cache = make_cache(6, 4);
        let mut rng = Xorshift(0x9e37_79b9);
        let mut applied: Vec<u64> = Vec::new();
        let mut block = 100u64;

        for round in 0..60u32 {
            let mut accessed = BlockAccessedState::default();
            for _ in 0..rng.below(10) {
                let address = Address::repeat_byte(rng.below(14) as u8);
                accessed
                    .accounts
                    .insert(address, account(u64::from(rng.below(50)), u64::from(rng.below(1000))));
                for _ in 0..rng.below(4) {
                    accessed.storage.insert(
                        (address, B256::repeat_byte(rng.below(6) as u8)),
                        U256::from(rng.below(10_000)),
                    );
                }
            }
            for _ in 0..rng.below(3) {
                let (code_hash, bytes) = code(rng.below(5) as u8, 32 + rng.below(200) as usize);
                accessed.codes.insert(code_hash, bytes);
            }

            cache.on_block_executed(block, &accessed);
            applied.push(block);
            assert_digest_index_is_exact(&cache, &format!("round {round}, block {block}"));
            block += 1;

            // Reorg: unwind newest-first, then let the loop rebuild a different chain from here.
            if round % 7 == 6 {
                let depth = 1 + rng.below(3) as usize;
                for _ in 0..depth.min(applied.len()) {
                    let target = applied.pop().expect("non-empty");
                    cache.rollback_block(target).expect("newest undo record");
                    assert_digest_index_is_exact(
                        &cache,
                        &format!("round {round}, rolled back {target}"),
                    );
                    block = target;
                }
            }

            // Finality moves behind the reorg window, which drops undo records without touching
            // the entries they describe.
            if round % 11 == 10 {
                let finalized = block.saturating_sub(5);
                cache.prune_undo_below(finalized);
                applied.retain(|applied_block| *applied_block > finalized);
                assert_digest_index_is_exact(&cache, &format!("round {round}, pruned {finalized}"));
            }
        }

        assert!(
            !cache.accounts.is_empty() && !cache.storage.is_empty(),
            "the sequence must leave a populated cache, or it proved nothing"
        );
    }

    /// Pins the v2 cache root against a value captured from the implementation that predates the
    /// code leaf memo (commit `432efb77d2`).
    ///
    /// The other tests here compare the memoized path with the reference path in the same build,
    /// which cannot catch a change that moves both. This one can: the expected root was produced by
    /// running this exact fixture before the memo existed. Peers compare anchors, so a change to
    /// this value is a protocol change and needs a commitment version, not a new constant.
    #[test]
    fn cache_root_matches_the_pre_memo_test_vector() {
        let mut cache = make_cache(50, 50);
        let mut accessed = BlockAccessedState::default();
        for i in 0u8..8 {
            accessed.accounts.insert(
                Address::repeat_byte(i),
                AccountData {
                    nonce: i as u64,
                    balance: U256::from(1_000u64 * i as u64 + 7),
                    code_hash: if i % 2 == 0 { None } else { Some(B256::repeat_byte(0xf0 | i)) },
                },
            );
            for j in 0u8..4 {
                accessed.storage.insert(
                    (Address::repeat_byte(i), B256::repeat_byte(j)),
                    U256::from(i * 10 + j),
                );
            }
            // Lengths from 32 to 256 bytes straddle the keccak rate boundary, so the memo's
            // absorbed prefix ends mid-block for some codes and on a block boundary for others.
            let bytes = Bytes::from(vec![i; 32 * (i as usize + 1)]);
            accessed.codes.insert(keccak256(&bytes), bytes);
        }
        cache.on_block_executed(1_000, &accessed);

        // A second block refreshes one account and adds one code, so the vector covers a cache
        // whose entries do not all share a `last_accessed_block`.
        let mut second = BlockAccessedState::default();
        second.accounts.insert(
            Address::repeat_byte(3),
            AccountData { nonce: 99, balance: U256::from(12345u64), code_hash: None },
        );
        let bytes = Bytes::from(vec![0xee; 11]);
        second.codes.insert(keccak256(&bytes), bytes);
        cache.on_block_executed(1_001, &second);

        let expected = b256!("0x7ab01b6994f2e7d54db8cb7118e1a2a7051c7fc2b3e4ad6392330da15d9bcf83");
        assert_eq!(cache.cache_root(), expected);
        assert_eq!(cache.compute_cache_root_reference(), expected);
    }

    /// A restored or forked cache must arrive with a populated memo and the same root.
    #[test]
    fn code_leaf_memo_survives_restore_and_fork() {
        let mut cache = make_cache(10, 10);
        let (code_hash, bytes) = code(0x22, 512);
        cache.on_block_executed(10, &accessed_with_code(code_hash, bytes));
        let root = cache.cache_root();

        let fork = cache.fork_for_reexecution(
            Box::new(LastNBlocksPolicy::new(10)),
            Box::new(LastNBlocksPolicy::new(10)),
        );
        assert_eq!(fork.code_leaf_hashers.len(), 1);
        assert_eq!(fork.compute_cache_root_uncached(), root);
        assert_eq!(fork.compute_cache_root_reference(), root);

        let restored = NetworkStateCache::restore(
            cache.accounts.clone(),
            cache.storage.clone(),
            cache.codes.clone(),
            cache.current_block,
            Box::new(LastNBlocksPolicy::new(10)),
            Box::new(LastNBlocksPolicy::new(10)),
        );
        assert_eq!(restored.code_leaf_hashers.len(), 1);
        assert_eq!(restored.compute_cache_root_uncached(), root);
        assert_eq!(restored.compute_cache_root_reference(), root);
    }

    #[test]
    fn cache_fork_preserves_only_a_locally_computed_root() {
        let mut parent = make_cache(10, 10);
        let address = Address::repeat_byte(0x01);
        let mut accessed = BlockAccessedState::default();
        accessed
            .accounts
            .insert(address, AccountData { nonce: 1, balance: U256::from(10), code_hash: None });
        parent.on_block_executed(10, &accessed);

        let fork_without_root = parent.fork_for_reexecution(
            Box::new(LastNBlocksPolicy::new(10)),
            Box::new(LastNBlocksPolicy::new(10)),
        );
        assert!(fork_without_root.memoized_cache_root.get().is_none());

        let parent_root = parent.cache_root();
        let mut fork = parent.fork_for_reexecution(
            Box::new(LastNBlocksPolicy::new(10)),
            Box::new(LastNBlocksPolicy::new(10)),
        );
        assert_eq!(fork.memoized_cache_root.get().copied(), Some(parent_root));
        assert_eq!(fork.cache_root(), parent_root);
        assert!(fork.undo_log.is_empty());
        assert!(fork.rollback_block(10).is_err(), "fork must not inherit undo history");

        fork.on_block_executed(11, &BlockAccessedState::default());
        assert!(fork.memoized_cache_root.get().is_none());
        assert_eq!(parent.memoized_cache_root.get().copied(), Some(parent_root));
        assert_eq!(parent.current_block(), 10);
    }

    #[test]
    fn reset_and_restore_start_without_a_cached_root() {
        let mut cache = make_cache(10, 10);
        let mut accessed = BlockAccessedState::default();
        accessed.accounts.insert(
            Address::repeat_byte(0x01),
            AccountData { nonce: 1, balance: U256::from(10), code_hash: None },
        );
        cache.on_block_executed(10, &accessed);
        let populated_root = cache.cache_root();

        let restored = NetworkStateCache::restore(
            cache.accounts.clone(),
            cache.storage.clone(),
            cache.codes.clone(),
            cache.current_block,
            Box::new(LastNBlocksPolicy::new(10)),
            Box::new(LastNBlocksPolicy::new(10)),
        );
        assert!(restored.memoized_cache_root.get().is_none());
        assert_eq!(restored.cache_root(), populated_root);

        cache.reset();
        assert!(cache.memoized_cache_root.get().is_none());
        assert_eq!(cache.cache_root(), make_cache(10, 10).cache_root());
    }

    #[test]
    fn cache_root_binds_code_bytecode_value() {
        let code_hash = B256::repeat_byte(0x03);

        // Deliberately use different bytes under the same key to pin key+value
        // root semantics for code entries.
        let mut codes_a = HashMap::new();
        codes_a.insert(
            code_hash,
            CachedEntry {
                value: Bytes::from(vec![1, 2, 3]),
                first_accessed_block: 1,
                last_accessed_block: 10,
                access_count: 1,
            },
        );
        let mut codes_b = HashMap::new();
        codes_b.insert(
            code_hash,
            CachedEntry {
                value: Bytes::from(vec![9, 9, 9]),
                first_accessed_block: 1,
                last_accessed_block: 10,
                access_count: 1,
            },
        );

        let cache_a = NetworkStateCache::restore(
            HashMap::new(),
            HashMap::new(),
            codes_a,
            10,
            Box::new(LastNBlocksPolicy::new(10)),
            Box::new(LastNBlocksPolicy::new(10)),
        );
        let cache_b = NetworkStateCache::restore(
            HashMap::new(),
            HashMap::new(),
            codes_b,
            10,
            Box::new(LastNBlocksPolicy::new(10)),
            Box::new(LastNBlocksPolicy::new(10)),
        );

        assert_ne!(cache_a.cache_root(), cache_b.cache_root());

        let mut codes_c = HashMap::new();
        codes_c.insert(
            B256::repeat_byte(0x04),
            CachedEntry {
                value: Bytes::from(vec![1, 2, 3]),
                first_accessed_block: 1,
                last_accessed_block: 10,
                access_count: 1,
            },
        );
        let cache_c = NetworkStateCache::restore(
            HashMap::new(),
            HashMap::new(),
            codes_c,
            10,
            Box::new(LastNBlocksPolicy::new(10)),
            Box::new(LastNBlocksPolicy::new(10)),
        );

        assert_ne!(cache_a.cache_root(), cache_c.cache_root());
    }

    #[test]
    fn cache_root_excludes_local_access_metadata() {
        let addr = Address::repeat_byte(0x01);
        let value = AccountData { nonce: 1, balance: U256::from(100), code_hash: None };

        let mut accounts_a = HashMap::new();
        accounts_a.insert(
            addr,
            CachedEntry {
                value: value.clone(),
                first_accessed_block: 1,
                last_accessed_block: 10,
                access_count: 1,
            },
        );
        let mut accounts_b = HashMap::new();
        accounts_b.insert(
            addr,
            CachedEntry {
                value: value.clone(),
                first_accessed_block: 9,
                last_accessed_block: 10,
                access_count: 99,
            },
        );

        let cache_a = NetworkStateCache::restore(
            accounts_a,
            HashMap::new(),
            HashMap::new(),
            10,
            Box::new(LastNBlocksPolicy::new(10)),
            Box::new(LastNBlocksPolicy::new(10)),
        );
        let cache_b = NetworkStateCache::restore(
            accounts_b,
            HashMap::new(),
            HashMap::new(),
            10,
            Box::new(LastNBlocksPolicy::new(10)),
            Box::new(LastNBlocksPolicy::new(10)),
        );

        assert_eq!(cache_a.cache_root(), cache_b.cache_root());

        let mut accounts_c = HashMap::new();
        accounts_c.insert(
            addr,
            CachedEntry {
                value,
                first_accessed_block: 1,
                last_accessed_block: 11,
                access_count: 1,
            },
        );
        let cache_c = NetworkStateCache::restore(
            accounts_c,
            HashMap::new(),
            HashMap::new(),
            11,
            Box::new(LastNBlocksPolicy::new(10)),
            Box::new(LastNBlocksPolicy::new(10)),
        );

        assert_ne!(cache_a.cache_root(), cache_c.cache_root());
    }

    #[test]
    fn cache_anchor_binds_policy_and_fork_context() {
        let mut cache = make_cache(10, 10);
        cache.on_block_executed(100, &BlockAccessedState::default());

        let policy_id = last_n_blocks_cache_policy_id(10, 10);
        let anchor = cache.cache_anchor(100, B256::repeat_byte(0xaa), policy_id);
        let different_context =
            cache.cache_anchor(101, B256::repeat_byte(0xbb), B256::repeat_byte(0xcc));

        assert_eq!(anchor.block_number, 100);
        assert_eq!(anchor.block_hash, B256::repeat_byte(0xaa));
        assert_eq!(anchor.cache_policy_id, policy_id);
        assert_eq!(anchor.cache_root, cache.cache_root());
        assert_ne!(anchor, different_context);
        assert_eq!(anchor.cache_root, different_context.cache_root);
    }

    #[test]
    fn test_refresh_extends_lifetime() {
        let mut cache = make_cache(10, 5);
        let addr = Address::repeat_byte(0x01);
        let slot = B256::repeat_byte(0x02);

        // Block 10: insert storage
        let mut accessed = BlockAccessedState::default();
        accessed.storage.insert((addr, slot), U256::from(1));
        cache.on_block_executed(10, &accessed);

        // Block 14: re-access the same slot (refreshes last_accessed_block to 14)
        let mut accessed2 = BlockAccessedState::default();
        accessed2.storage.insert((addr, slot), U256::from(2));
        cache.on_block_executed(14, &accessed2);

        // Block 18: storage cutoff=13. Since last_accessed=14, it should be retained.
        cache.on_block_executed(18, &BlockAccessedState::default());
        assert!(cache.contains_storage(&addr, &slot));

        // Block 20: storage cutoff=15. Since last_accessed=14, now evicted.
        cache.on_block_executed(20, &BlockAccessedState::default());
        assert!(!cache.contains_storage(&addr, &slot));
    }

    fn account(nonce: u64, balance: u64) -> AccountData {
        AccountData { nonce, balance: U256::from(balance), code_hash: None }
    }

    #[test]
    fn test_rollback_removes_newly_inserted() {
        let mut cache = make_cache(10, 10);
        let addr = Address::repeat_byte(0x01);
        let mut accessed = BlockAccessedState::default();
        accessed.accounts.insert(addr, account(1, 1));
        cache.on_block_executed(100, &accessed);
        assert!(cache.contains_account(&addr));

        cache.rollback_block(100).unwrap();
        assert!(!cache.contains_account(&addr));
        assert_eq!(cache.current_block(), 0);
    }

    #[test]
    fn test_rollback_restores_refreshed_value() {
        let mut cache = make_cache(10, 10);
        let addr = Address::repeat_byte(0x01);

        let mut b1 = BlockAccessedState::default();
        b1.accounts.insert(addr, account(1, 100));
        cache.on_block_executed(10, &b1);

        let mut b2 = BlockAccessedState::default();
        b2.accounts.insert(addr, account(2, 200));
        cache.on_block_executed(11, &b2);

        cache.rollback_block(11).unwrap();
        let entry = cache.accounts().get(&addr).expect("account still cached");
        assert_eq!(entry.value.nonce, 1);
        assert_eq!(entry.value.balance, U256::from(100));
        assert_eq!(entry.last_accessed_block, 10);
        assert_eq!(cache.current_block(), 10);
    }

    #[test]
    fn test_rollback_restores_evicted_entry() {
        // account window = 3: entry inserted at block 10 is evicted by block 14 (cutoff 11).
        let mut cache = make_cache(3, 3);
        let addr = Address::repeat_byte(0x01);
        let mut b = BlockAccessedState::default();
        b.accounts.insert(addr, account(1, 1));
        cache.on_block_executed(10, &b);

        cache.on_block_executed(14, &BlockAccessedState::default());
        assert!(!cache.contains_account(&addr), "should be evicted at block 14");

        cache.rollback_block(14).unwrap();
        assert!(cache.contains_account(&addr), "rollback must restore the evicted entry");
    }

    #[test]
    fn test_rollback_matches_prior_state_invariant() {
        let mut cache = make_cache(10, 10);
        let a1 = Address::repeat_byte(0x01);
        let a2 = Address::repeat_byte(0x02);

        let mut b1 = BlockAccessedState::default();
        b1.accounts.insert(a1, account(1, 1));
        cache.on_block_executed(10, &b1);
        let snap_after_10 = cache.snapshot();

        let mut b2 = BlockAccessedState::default();
        b2.accounts.insert(a2, account(1, 2));
        cache.on_block_executed(11, &b2);

        cache.rollback_block(11).unwrap();
        let rolled_back = cache.snapshot();
        assert_eq!(rolled_back.total_accounts, snap_after_10.total_accounts);
        assert_eq!(rolled_back.current_block, snap_after_10.current_block);
        assert!(cache.contains_account(&a1));
        assert!(!cache.contains_account(&a2));
    }

    #[test]
    fn test_rollback_mismatch_is_rejected() {
        let mut cache = make_cache(10, 10);
        let mut b = BlockAccessedState::default();
        b.accounts.insert(Address::repeat_byte(0x01), account(1, 1));
        cache.on_block_executed(10, &b);
        // Newest undo is block 10; requesting any other block must fail.
        assert!(cache.rollback_block(9).is_err());
        assert!(cache.rollback_block(11).is_err());
    }

    #[test]
    fn test_prune_below_finalized_then_rollback_rejected() {
        let mut cache = make_cache(10, 10);
        let mut b = BlockAccessedState::default();
        b.accounts.insert(Address::repeat_byte(0x01), account(1, 1));
        cache.on_block_executed(10, &b);

        cache.prune_undo_below(10);
        // Undo for block 10 was pruned (finalized) → it can no longer be rolled back.
        assert!(cache.rollback_block(10).is_err());
    }

    #[test]
    fn test_reset_clears_everything() {
        let mut cache = make_cache(10, 10);
        let mut b = BlockAccessedState::default();
        b.accounts.insert(Address::repeat_byte(0x01), account(1, 1));
        b.storage.insert((Address::repeat_byte(0x01), B256::repeat_byte(0x02)), U256::from(5));
        cache.on_block_executed(10, &b);

        cache.reset();
        assert_eq!(cache.snapshot().total_accounts, 0);
        assert_eq!(cache.snapshot().total_storage_slots, 0);
        assert_eq!(cache.current_block(), 0);
    }
}
