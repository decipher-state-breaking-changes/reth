//! Persistent sparse-trie cache for partial-stateless validation.
//!
//! The value cache answers reads. This cache keeps the corresponding account and storage proof
//! paths in a locally updated [`SparseStateTrie`]. A sidecar supplies parent-state miss paths and
//! execution updates this trie in place. The flat value cache is authoritative for hits and its
//! account/storage windows determine which inclusion or exclusion witness paths remain decoded.

use crate::{
    accessed_state::BlockAccessedState,
    network_cache::{MissResult, NetworkStateCache},
    participant::ParticipantCache,
};
use alloy_primitives::{
    keccak256,
    map::{B256Map, HashSet},
    Address, B256,
};
use reth_trie_common::{DecodedMultiProofV2, HashedPostState, Nibbles};
use reth_trie_sparse::{RevealableSparseTrie, SparseStateTrie, SparseTrie};
use std::fmt;

/// Sparse trie plus the value-cache membership whose paths it is required to retain.
///
/// Cloning this type creates a transactional snapshot. Producers and validators apply a block to a
/// clone, check the post-state root and next anchor, and only then replace the previous cache.
#[derive(Debug)]
pub struct PartialTrieNodeCache {
    sparse: SparseStateTrie,
    warm_accounts: HashSet<Address>,
    warm_storage: HashSet<(Address, B256)>,
    state_root: Option<B256>,
}

impl Clone for PartialTrieNodeCache {
    fn clone(&self) -> Self {
        let accounts = self
            .sparse
            .state_trie_ref()
            .map(|trie| RevealableSparseTrie::Revealed(Box::new(trie.clone())))
            .unwrap_or_else(RevealableSparseTrie::blind);
        let mut sparse = SparseStateTrie::new().with_accounts_trie(accounts);

        // `retain_from_value_cache` removes every storage trie without a warm slot, so this is
        // the complete semantic storage snapshot. Allocation-reuse buffers and process-local LFU
        // history are deliberately not copied; value-cache membership is authoritative.
        let mut storage_addresses: Vec<_> =
            self.warm_storage.iter().map(|(address, _)| keccak256(address)).collect();
        storage_addresses.sort_unstable();
        storage_addresses.dedup();
        for hashed_address in storage_addresses {
            if let Some(trie) = self.sparse.storage_trie_ref(&hashed_address) {
                sparse.insert_storage_trie(
                    hashed_address,
                    RevealableSparseTrie::Revealed(Box::new(trie.clone())),
                );
            }
        }

        Self {
            sparse,
            warm_accounts: self.warm_accounts.clone(),
            warm_storage: self.warm_storage.clone(),
            state_root: self.state_root,
        }
    }
}

impl Default for PartialTrieNodeCache {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialTrieNodeCache {
    /// Creates a cold local sparse trie.
    pub fn new() -> Self {
        Self {
            sparse: SparseStateTrie::new(),
            warm_accounts: HashSet::default(),
            warm_storage: HashSet::default(),
            state_root: None,
        }
    }

    pub(crate) fn restore_from_decoded_multiproof(
        multiproof: DecodedMultiProofV2,
        expected_state_root: B256,
        value_cache: &NetworkStateCache,
    ) -> Result<Self, TrieCacheValidationError> {
        let mut cache = Self::new();
        cache
            .sparse
            .reveal_decoded_multiproof_v2(multiproof)
            .map_err(|err| TrieCacheValidationError::ProofReveal(err.to_string()))?;
        let actual_root = cache
            .sparse
            .root()
            .map_err(|err| TrieCacheValidationError::RootComputation(err.to_string()))?;
        drop(cache.sparse.take_deferred_drops());
        if actual_root != expected_state_root {
            return Err(TrieCacheValidationError::StateRootMismatch {
                expected: expected_state_root,
                actual: actual_root,
            })
        }
        cache.state_root = Some(actual_root);
        cache.retain_from_value_cache(value_cache);
        cache.validate_against_value_cache(value_cache)?;
        Ok(cache)
    }

    /// Returns the post-state root represented by the local sparse trie, when initialized.
    pub const fn state_root(&self) -> Option<B256> {
        self.state_root
    }

    pub(crate) fn sparse_mut(&mut self) -> &mut SparseStateTrie {
        &mut self.sparse
    }

    pub(crate) fn set_state_root(&mut self, state_root: B256) {
        self.state_root = Some(state_root);
    }

    /// Retains only the updated sparse-trie paths required by the value cache after each block.
    ///
    /// The value cache remains authoritative for hits. Unlike leaf-only pruning,
    /// [`SparseTrie::retain_witness_paths`] keeps terminal extension/leaf mismatches that prove a
    /// cached zero or nonexistent account while blinding unrelated decoded subtrees.
    pub fn retain_from_value_cache(&mut self, value_cache: &NetworkStateCache) {
        self.warm_accounts = value_cache.accounts().keys().copied().collect();
        self.warm_storage = value_cache.storage().keys().copied().collect();

        let mut retained_accounts = self.warm_accounts.clone();
        let mut retained_storage = B256Map::<Vec<Nibbles>>::default();
        for (address, slot) in &self.warm_storage {
            retained_accounts.insert(*address);
            retained_storage
                .entry(keccak256(address))
                .or_default()
                .push(Nibbles::unpack(keccak256(slot)));
        }

        let mut retained_account_paths: Vec<_> = retained_accounts
            .into_iter()
            .map(|address| Nibbles::unpack(keccak256(address)))
            .collect();
        retained_account_paths.sort_unstable();
        retained_account_paths.dedup();
        if let Some(trie) = self.sparse.trie_mut().as_revealed_mut() {
            trie.retain_witness_paths(&retained_account_paths);
        }

        for slots in retained_storage.values_mut() {
            slots.sort_unstable();
            slots.dedup();
        }
        self.sparse.storage_tries_mut().retain(|hashed_address, trie| {
            let Some(slots) = retained_storage.get(hashed_address) else { return false };
            if let Some(trie) = trie.as_revealed_mut() {
                trie.retain_witness_paths(slots);
            }
            true
        });
    }

    /// Whether the current sparse shape can prove this account value or absence.
    pub fn contains_account_path(&self, address: &Address) -> bool {
        self.contains_hashed_account_path(keccak256(address))
    }

    /// Whether the current sparse shape can prove this hashed account path.
    pub fn contains_hashed_account_path(&self, hashed_address: B256) -> bool {
        self.sparse.is_account_revealed(hashed_address)
    }

    /// Returns whether an account exists when its authenticated path is revealed.
    ///
    /// `None` means the sparse trie cannot currently prove the path. `Some(false)` is an
    /// authenticated exclusion and must not be confused with an existing empty account.
    pub fn account_exists(&self, address: &Address) -> Option<bool> {
        let hashed_address = keccak256(address);
        self.sparse
            .is_account_revealed(hashed_address)
            .then(|| self.sparse.get_account_value(&hashed_address).is_some())
    }

    /// Whether the current sparse shape can prove this storage value or absence.
    pub fn contains_storage_path(&self, address: &Address, slot: &B256) -> bool {
        self.contains_hashed_storage_path(keccak256(address), keccak256(slot))
    }

    /// Whether the current sparse shape can prove this hashed storage path.
    pub fn contains_hashed_storage_path(&self, hashed_address: B256, hashed_slot: B256) -> bool {
        self.sparse.check_valid_storage_witness(hashed_address, hashed_slot)
    }

    #[cfg(test)]
    pub(crate) fn has_storage_trie(&self, address: &Address) -> bool {
        self.sparse.storage_trie_ref(&keccak256(address)).is_some()
    }

    /// Whether the authoritative value cache currently tracks this account.
    pub fn tracks_account(&self, address: &Address) -> bool {
        self.warm_accounts.contains(address)
    }

    /// Whether the authoritative value cache currently tracks this storage slot.
    pub fn tracks_storage(&self, address: &Address, slot: &B256) -> bool {
        self.warm_storage.contains(&(*address, *slot))
    }

    /// Number of warm value paths represented by the sparse trie.
    pub fn warm_node_count(&self) -> usize {
        self.warm_accounts.len() + self.warm_storage.len()
    }

    pub fn tracked_account_count(&self) -> usize {
        self.warm_accounts.len()
    }

    pub fn tracked_storage_slot_count(&self) -> usize {
        self.warm_storage.len()
    }

    /// Heuristic memory size of the retained sparse trie.
    pub fn estimated_memory_bytes(&self) -> usize {
        self.sparse.memory_size()
    }

    /// Returns diagnostics for comparing deterministic path retention with a fixed-depth pinned
    /// account-trie cache.
    ///
    /// `account_key_prefixes[depth]` counts distinct hashed-account prefixes represented by the
    /// retained account paths. It is a coverage proxy, not a literal MPT node count: Patricia
    /// extension nodes can compress several nibble levels. `account_revealed_nodes` and
    /// `storage_revealed_nodes` are the actual decoded non-hash sparse-trie node counts.
    pub fn shape_metrics(&self) -> TrieShapeMetrics {
        let retained_accounts = self.retained_account_addresses();
        let mut prefixes: [HashSet<Vec<u8>>; TRIE_SHAPE_PREFIX_LEVELS] =
            std::array::from_fn(|_| HashSet::default());
        for address in &retained_accounts {
            let path = Nibbles::unpack(keccak256(address));
            for (depth, level) in prefixes.iter_mut().enumerate() {
                level.insert(path.slice(0..depth).to_vec());
            }
        }

        let mut storage_addresses = HashSet::<B256>::default();
        let mut storage_revealed_nodes = 0;
        for (address, _) in &self.warm_storage {
            let hashed_address = keccak256(address);
            if storage_addresses.insert(hashed_address) {
                storage_revealed_nodes +=
                    self.sparse.storage_trie_ref(&hashed_address).map_or(0, SparseTrie::size_hint);
            }
        }

        let account_key_prefixes = std::array::from_fn(|depth| prefixes[depth].len());
        let account_prefix_coverage = std::array::from_fn(|depth| {
            let capacity = 16usize.pow(depth as u32);
            account_key_prefixes[depth] as f64 / capacity as f64
        });

        TrieShapeMetrics {
            retained_account_paths: retained_accounts.len(),
            retained_storage_tries: storage_addresses.len(),
            retained_storage_paths: self.warm_storage.len(),
            account_revealed_nodes: self.sparse.state_trie_ref().map_or(0, SparseTrie::size_hint),
            storage_revealed_nodes,
            estimated_memory_bytes: self.estimated_memory_bytes(),
            account_key_prefixes,
            account_prefix_coverage,
        }
    }

    /// Reports how much of the retained trie a block's changed keys dirty.
    ///
    /// Changing a leaf re-hashes every node between it and the root, so the share of the trie a
    /// block invalidates is the share of *path prefixes* its changed keys cover. At each depth this
    /// intersects the prefixes of the changed keys with the prefixes of everything retained: that
    /// ratio is the fraction of the per-block clone a copy-on-write or journalling snapshot would
    /// avoid copying at that level.
    ///
    /// The intersection matters. A key the cache does not retain still dirties the clone whenever
    /// it shares a prefix with something the cache does hold — inserting a new account
    /// re-hashes the branch nodes above it, and those nodes were copied. Counting only changed
    /// keys that are themselves retained would understate the dirty set and overstate the
    /// headroom for copy-on-write.
    ///
    /// Prefix counts are a structural proxy rather than literal node counts, because Patricia
    /// extension nodes compress several nibble levels into one. The approximation applies equally
    /// to both sides of each ratio, which is why the ratios are more trustworthy than either
    /// count on its own.
    pub fn mutation_metrics(&self, changed: &TrieChangeSet) -> TrieMutationMetrics {
        let retained_accounts =
            self.retained_account_addresses().into_iter().map(keccak256).collect::<HashSet<_>>();
        let account_prefixes = prefix_coverage(&retained_accounts, &changed.dirtied_accounts());
        let dirtied_account_paths =
            retained_accounts.iter().filter(|key| changed.accounts.contains(*key)).count();

        let mut retained_by_trie = B256Map::<HashSet<B256>>::default();
        for (address, slot) in &self.warm_storage {
            retained_by_trie.entry(keccak256(address)).or_default().insert(keccak256(slot));
        }

        let mut per_storage_trie = Vec::with_capacity(retained_by_trie.len());
        let mut retained_storage_paths = 0;
        let mut dirtied_storage_paths = 0;
        let mut dirtied_storage_tries = 0;
        for (hashed_address, retained) in &retained_by_trie {
            // A wipe removes every leaf, so it dirties every path this cache retained for the
            // address even though the post-state lists no individual slot for most of them.
            let wiped = changed.wiped_storage.contains(hashed_address);
            let changed_slots = if wiped {
                retained.clone()
            } else {
                changed.storage.get(hashed_address).cloned().unwrap_or_default()
            };
            let dirtied = retained.iter().filter(|slot| changed_slots.contains(*slot)).count();

            retained_storage_paths += retained.len();
            dirtied_storage_paths += dirtied;
            if !changed_slots.is_empty() {
                dirtied_storage_tries += 1;
            }
            per_storage_trie.push(StorageTrieMutation {
                hashed_address: *hashed_address,
                revealed_nodes: self
                    .sparse
                    .storage_trie_ref(hashed_address)
                    .map_or(0, SparseTrie::size_hint),
                retained_paths: retained.len(),
                dirtied_paths: dirtied,
                wiped,
                prefixes: prefix_coverage(retained, &changed_slots),
            });
        }
        // Largest first: the tail of this distribution is what a per-trie sharing scheme has to
        // handle well, and a log line only ever shows the head.
        per_storage_trie.sort_unstable_by(|a, b| {
            b.dirtied_paths.cmp(&a.dirtied_paths).then(a.hashed_address.cmp(&b.hashed_address))
        });

        TrieMutationMetrics {
            retained_account_paths: retained_accounts.len(),
            dirtied_account_paths,
            account_prefixes,
            account_revealed_nodes: self.sparse.state_trie_ref().map_or(0, SparseTrie::size_hint),
            retained_storage_paths,
            dirtied_storage_paths,
            dirtied_storage_tries,
            storage_revealed_nodes: per_storage_trie.iter().map(|trie| trie.revealed_nodes).sum(),
            per_storage_trie,
        }
    }

    /// Validates that flat-cache membership, authenticated paths, and the stored state root agree.
    ///
    /// This scans every retained account and storage path and is intended for tests and opt-in ExEx
    /// diagnostics, not the normal per-block hot path.
    pub fn validate_against_value_cache(
        &mut self,
        value_cache: &NetworkStateCache,
    ) -> Result<TrieShapeMetrics, TrieCacheValidationError> {
        let expected_accounts: HashSet<_> = value_cache.accounts().keys().copied().collect();
        if expected_accounts != self.warm_accounts {
            return Err(TrieCacheValidationError::AccountMembership {
                missing: expected_accounts.difference(&self.warm_accounts).count(),
                extra: self.warm_accounts.difference(&expected_accounts).count(),
            })
        }

        let expected_storage: HashSet<_> = value_cache.storage().keys().copied().collect();
        if expected_storage != self.warm_storage {
            return Err(TrieCacheValidationError::StorageMembership {
                missing: expected_storage.difference(&self.warm_storage).count(),
                extra: self.warm_storage.difference(&expected_storage).count(),
            })
        }

        for address in self.retained_account_addresses() {
            if !self.sparse.is_account_revealed(keccak256(address)) {
                return Err(TrieCacheValidationError::MissingAccountPath(address))
            }
        }
        for (address, slot) in &self.warm_storage {
            if !self.sparse.check_valid_storage_witness(keccak256(address), keccak256(slot)) {
                return Err(TrieCacheValidationError::MissingStoragePath {
                    address: *address,
                    slot: *slot,
                })
            }
        }

        let expected_root = self.state_root.ok_or(TrieCacheValidationError::MissingStateRoot)?;
        let actual_root = self
            .sparse
            .root()
            .map_err(|err| TrieCacheValidationError::RootComputation(err.to_string()))?;
        if actual_root != expected_root {
            return Err(TrieCacheValidationError::StateRootMismatch {
                expected: expected_root,
                actual: actual_root,
            })
        }

        Ok(self.shape_metrics())
    }

    fn retained_account_addresses(&self) -> HashSet<Address> {
        let mut accounts = self.warm_accounts.clone();
        accounts.extend(self.warm_storage.iter().map(|(address, _)| *address));
        accounts
    }

    /// Deterministic commitment to the local sparse-trie state and retained path set.
    ///
    /// The canonical state root authenticates the node contents; the sorted membership determines
    /// which authenticated paths the deterministic pruning algorithm retains.
    pub fn cache_root(&self) -> B256 {
        let mut preimage = Vec::new();
        preimage.extend_from_slice(b"PartialTrieNodeCacheRoot/v3");
        match self.state_root {
            Some(root) => {
                preimage.push(1);
                preimage.extend_from_slice(root.as_slice());
            }
            None => preimage.push(0),
        }

        let mut accounts: Vec<_> = self.warm_accounts.iter().copied().collect();
        accounts.sort_unstable();
        preimage.extend_from_slice(&(accounts.len() as u64).to_be_bytes());
        for address in accounts {
            preimage.extend_from_slice(address.as_slice());
        }

        let mut storage: Vec<_> = self.warm_storage.iter().copied().collect();
        storage.sort_unstable();
        preimage.extend_from_slice(&(storage.len() as u64).to_be_bytes());
        for (address, slot) in storage {
            preimage.extend_from_slice(address.as_slice());
            preimage.extend_from_slice(slot.as_slice());
        }

        keccak256(preimage)
    }
}

/// Number of account-key prefix levels reported for comparison with the old depth-five pinned
/// cache. The array covers depths zero through five, inclusive.
pub const TRIE_SHAPE_PREFIX_LEVELS: usize = 6;

/// Snapshot of the retained sparse-trie shape for live benchmarking.
#[derive(Debug, Clone, PartialEq)]
pub struct TrieShapeMetrics {
    pub retained_account_paths: usize,
    pub retained_storage_tries: usize,
    pub retained_storage_paths: usize,
    pub account_revealed_nodes: usize,
    pub storage_revealed_nodes: usize,
    pub estimated_memory_bytes: usize,
    pub account_key_prefixes: [usize; TRIE_SHAPE_PREFIX_LEVELS],
    pub account_prefix_coverage: [f64; TRIE_SHAPE_PREFIX_LEVELS],
}

/// How much of the retained sparse trie a single block dirties.
///
/// The clone-per-block snapshot copies the whole retained trie; these counts say how much of it the
/// block then invalidates. The gap between the two is the headroom a copy-on-write or journalling
/// snapshot could recover, and `*_share` near 1.0 means the clone is already close to optimal.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrieMutationMetrics {
    /// Account paths the cache retains.
    pub retained_account_paths: usize,
    /// Retained account paths the block changed.
    pub dirtied_account_paths: usize,
    /// Account-trie prefix coverage by depth, retained against dirtied.
    pub account_prefixes: [PrefixCoverage; TRIE_SHAPE_PREFIX_LEVELS],
    /// Nodes of every kind revealed in the account trie.
    pub account_revealed_nodes: usize,
    /// Storage paths the cache retains, across all tries.
    pub retained_storage_paths: usize,
    /// Retained storage paths the block changed.
    pub dirtied_storage_paths: usize,
    /// Storage tries the block dirtied at all.
    pub dirtied_storage_tries: usize,
    /// Nodes of every kind revealed across every retained storage trie.
    pub storage_revealed_nodes: usize,
    /// Per-trie breakdown, most dirtied first.
    pub per_storage_trie: Vec<StorageTrieMutation>,
}

impl TrieMutationMetrics {
    /// Retained paths the block changed, across both tries.
    pub const fn dirtied_paths(&self) -> usize {
        self.dirtied_account_paths + self.dirtied_storage_paths
    }

    /// Retained paths in total.
    pub const fn retained_paths(&self) -> usize {
        self.retained_account_paths + self.retained_storage_paths
    }

    /// Nodes the per-block clone copies.
    pub const fn revealed_nodes(&self) -> usize {
        self.account_revealed_nodes + self.storage_revealed_nodes
    }

    /// Share of retained leaf paths the block changed, or 0.0 when nothing is retained.
    ///
    /// This is the leaf-level share. Nodes higher up are shared between paths, so the share of
    /// *nodes* dirtied is larger; [`account_prefixes`](Self::account_prefixes) shows how it grows
    /// with depth.
    pub fn dirtied_path_share(&self) -> f64 {
        fraction(self.dirtied_paths(), self.retained_paths())
    }

    /// The deepest prefix level where the account trie retains anything, and its coverage.
    ///
    /// The shallowest levels saturate — every block touches the root — so the useful signal is at
    /// the deepest level that still discriminates.
    pub fn deepest_account_prefix(&self) -> PrefixCoverage {
        self.account_prefixes
            .iter()
            .rev()
            .find(|coverage| coverage.retained > 0)
            .copied()
            .unwrap_or_default()
    }
}

/// Retained against dirtied distinct key prefixes at one trie depth.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefixCoverage {
    /// Distinct prefixes across all retained paths.
    pub retained: usize,
    /// Distinct prefixes across the paths the block changed.
    pub dirtied: usize,
}

impl PrefixCoverage {
    /// Share of this depth's retained prefixes that the block dirtied.
    pub fn dirtied_share(&self) -> f64 {
        fraction(self.dirtied, self.retained)
    }
}

/// One storage trie's contribution to [`TrieMutationMetrics`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageTrieMutation {
    /// Hashed address owning the trie.
    pub hashed_address: B256,
    /// Nodes of every kind revealed in this trie.
    pub revealed_nodes: usize,
    /// Storage paths the cache retains for this address.
    pub retained_paths: usize,
    /// Retained paths the block changed.
    pub dirtied_paths: usize,
    /// Whether the block wiped this storage trie, which dirties every retained path in it.
    pub wiped: bool,
    /// Prefix coverage by depth within this trie.
    pub prefixes: [PrefixCoverage; TRIE_SHAPE_PREFIX_LEVELS],
}

/// Invariant failure reported by [`PartialTrieNodeCache::validate_against_value_cache`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrieCacheValidationError {
    ProofReveal(String),
    AccountMembership { missing: usize, extra: usize },
    StorageMembership { missing: usize, extra: usize },
    MissingAccountPath(Address),
    MissingStoragePath { address: Address, slot: B256 },
    MissingStateRoot,
    RootComputation(String),
    StateRootMismatch { expected: B256, actual: B256 },
}

impl fmt::Display for TrieCacheValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProofReveal(error) => write!(f, "failed to reveal bootstrap proof: {error}"),
            Self::AccountMembership { missing, extra } => write!(
                f,
                "account membership differs from value cache: missing={missing}, extra={extra}"
            ),
            Self::StorageMembership { missing, extra } => write!(
                f,
                "storage membership differs from value cache: missing={missing}, extra={extra}"
            ),
            Self::MissingAccountPath(address) => {
                write!(f, "retained account path is blind: {address}")
            }
            Self::MissingStoragePath { address, slot } => {
                write!(f, "retained storage path is blind: address={address}, slot={slot}")
            }
            Self::MissingStateRoot => f.write_str("local sparse trie has no recorded state root"),
            Self::RootComputation(error) => {
                write!(f, "failed to recompute local sparse-trie root: {error}")
            }
            Self::StateRootMismatch { expected, actual } => {
                write!(f, "local sparse-trie root mismatch: expected={expected}, actual={actual}")
            }
        }
    }
}

impl std::error::Error for TrieCacheValidationError {}

impl ParticipantCache for PartialTrieNodeCache {
    fn contains_account(&self, address: &Address) -> bool {
        self.contains_account_path(address)
    }

    fn contains_storage(&self, address: &Address, slot: &B256) -> bool {
        self.contains_storage_path(address, slot)
    }

    fn contains_code(&self, _code_hash: &B256) -> bool {
        false
    }

    fn compute_miss(&self, accessed: &BlockAccessedState) -> MissResult {
        let missed_accounts = accessed
            .accounts
            .keys()
            .filter(|address| !self.contains_account_path(address))
            .copied()
            .collect::<Vec<_>>();
        let missed_storage = accessed
            .storage
            .keys()
            .filter(|(address, slot)| !self.contains_storage_path(address, slot))
            .copied()
            .collect::<Vec<_>>();
        let total_accessed = accessed.accounts.len() + accessed.storage.len();
        let total_missed = missed_accounts.len() + missed_storage.len();
        let miss_ratio =
            if total_accessed == 0 { 0.0 } else { total_missed as f64 / total_accessed as f64 };

        MissResult {
            missed_accounts,
            missed_storage,
            missed_codes: Vec::new(),
            total_accessed,
            total_missed,
            miss_ratio,
        }
    }

    fn cache_root(&self) -> B256 {
        PartialTrieNodeCache::cache_root(self)
    }
}

/// The keys a block changed, in the trie's own hashed key space.
///
/// Built from a block's post-state rather than from the cache, so it deliberately includes keys the
/// cache does not retain: those still dirty the retained nodes above them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TrieChangeSet {
    /// Hashed addresses whose account leaf changed.
    pub accounts: HashSet<B256>,
    /// Hashed storage slots that changed, by hashed address.
    pub storage: B256Map<HashSet<B256>>,
    /// Hashed addresses whose storage trie was wiped.
    pub wiped_storage: HashSet<B256>,
}

impl TrieChangeSet {
    /// Reads the change set out of a block's hashed post state.
    pub fn from_hashed_post_state(post_state: &HashedPostState) -> Self {
        let mut changed =
            Self { accounts: post_state.accounts.keys().copied().collect(), ..Default::default() };
        for (hashed_address, storage) in &post_state.storages {
            if storage.wiped {
                changed.wiped_storage.insert(*hashed_address);
            }
            if !storage.storage.is_empty() {
                changed.storage.insert(*hashed_address, storage.storage.keys().copied().collect());
            }
        }
        changed
    }

    /// Hashed addresses whose account-trie leaf this block re-hashes.
    ///
    /// Wider than [`accounts`](Self::accounts): an account's leaf holds its storage root, so
    /// touching any of its slots rewrites the account leaf even when the account itself is
    /// unchanged and absent from the post state's account map.
    pub fn dirtied_accounts(&self) -> HashSet<B256> {
        let mut dirtied = self.accounts.clone();
        dirtied.extend(self.storage.keys().copied());
        dirtied.extend(self.wiped_storage.iter().copied());
        dirtied
    }
}

/// Ratio guarded against an empty denominator, which a cold trie always has.
fn fraction(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        return 0.0
    }
    numerator as f64 / denominator as f64
}

/// Counts, per depth, how many of `retained`'s distinct nibble prefixes `changed` also covers.
///
/// `changed` need not be a subset of `retained`. Intersecting at each depth is the point: a changed
/// key the cache does not hold still re-hashes every retained node it shares a prefix with, while a
/// changed key that diverges at the first nibble from everything retained dirties nothing. The
/// intersection also keeps every ratio at or below 1.
fn prefix_coverage(
    retained: &HashSet<B256>,
    changed: &HashSet<B256>,
) -> [PrefixCoverage; TRIE_SHAPE_PREFIX_LEVELS] {
    let mut retained_prefixes: [HashSet<Vec<u8>>; TRIE_SHAPE_PREFIX_LEVELS] =
        std::array::from_fn(|_| HashSet::default());
    for key in retained {
        let path = Nibbles::unpack(key);
        for (depth, level) in retained_prefixes.iter_mut().enumerate() {
            level.insert(path.slice(0..depth).to_vec());
        }
    }

    let mut dirtied_prefixes: [HashSet<Vec<u8>>; TRIE_SHAPE_PREFIX_LEVELS] =
        std::array::from_fn(|_| HashSet::default());
    for key in changed {
        let path = Nibbles::unpack(key);
        for depth in 0..TRIE_SHAPE_PREFIX_LEVELS {
            let prefix = path.slice(0..depth).to_vec();
            if retained_prefixes[depth].contains(&prefix) {
                dirtied_prefixes[depth].insert(prefix);
            }
        }
    }

    std::array::from_fn(|depth| PrefixCoverage {
        retained: retained_prefixes[depth].len(),
        dirtied: dirtied_prefixes[depth].len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        policy::{AccountData, LastNBlocksPolicy},
        NetworkStateCache,
    };
    use alloy_primitives::U256;

    fn value_cache() -> NetworkStateCache {
        NetworkStateCache::new(
            Box::new(LastNBlocksPolicy::new(60)),
            Box::new(LastNBlocksPolicy::new(30)),
        )
    }

    #[test]
    fn cold_cache_misses_every_path() {
        let address = Address::repeat_byte(0x11);
        let slot = B256::repeat_byte(0x22);
        let mut accessed = BlockAccessedState::default();
        accessed
            .accounts
            .insert(address, AccountData { nonce: 0, balance: U256::ZERO, code_hash: None });
        accessed.storage.insert((address, slot), U256::from(1));

        let miss = PartialTrieNodeCache::new().compute_miss(&accessed);
        assert_eq!(miss.missed_accounts, vec![address]);
        assert_eq!(miss.missed_storage, vec![(address, slot)]);
    }

    #[test]
    fn validation_rejects_value_cache_membership_drift() {
        let address = Address::repeat_byte(0x33);
        let mut accessed = BlockAccessedState::default();
        accessed
            .accounts
            .insert(address, AccountData { nonce: 1, balance: U256::from(2), code_hash: None });

        let mut values = value_cache();
        values.on_block_executed(1, &accessed);
        let mut trie = PartialTrieNodeCache::new();

        assert_eq!(
            trie.validate_against_value_cache(&values),
            Err(TrieCacheValidationError::AccountMembership { missing: 1, extra: 0 })
        );

        trie.retain_from_value_cache(&values);
        assert!(matches!(
            trie.validate_against_value_cache(&values),
            Err(TrieCacheValidationError::MissingAccountPath(missing)) if missing == address
        ));
    }

    #[test]
    fn membership_tracking_does_not_fabricate_witness_paths() {
        let address = Address::repeat_byte(0x11);
        let slot = B256::repeat_byte(0x22);
        let mut accessed = BlockAccessedState::default();
        accessed
            .accounts
            .insert(address, AccountData { nonce: 0, balance: U256::ZERO, code_hash: None });
        accessed.storage.insert((address, slot), U256::from(1));

        let mut values = value_cache();
        values.on_block_executed(1, &accessed);
        let mut trie = PartialTrieNodeCache::new();
        trie.retain_from_value_cache(&values);

        assert!(trie.tracks_account(&address));
        assert!(trie.tracks_storage(&address, &slot));
        assert!(!trie.contains_account_path(&address));
        assert_eq!(trie.account_exists(&address), None);
        assert!(!trie.contains_storage_path(&address, &slot));
        assert_eq!(trie.tracked_account_count(), 1);
        assert_eq!(trie.tracked_storage_slot_count(), 1);
    }

    #[test]
    fn cache_root_commits_state_root_and_membership() {
        let mut a = PartialTrieNodeCache::new();
        let mut b = a.clone();
        assert_eq!(a.cache_root(), b.cache_root());

        b.set_state_root(B256::repeat_byte(0x44));
        assert_ne!(a.cache_root(), b.cache_root());

        a.set_state_root(B256::repeat_byte(0x44));
        assert_eq!(a.cache_root(), b.cache_root());
    }
}
