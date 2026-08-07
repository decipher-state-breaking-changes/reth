//! Persistent sparse-trie cache for partial-stateless validation.
//!
//! The value cache answers reads. This cache keeps the corresponding account and storage proof
//! paths in a locally updated [`SparseStateTrie`]. A sidecar supplies parent-state miss paths and
//! execution updates this trie in place. The flat value cache is authoritative for hits and its
//! account/storage windows determine which inclusion or exclusion witness paths remain decoded.

use crate::{
    accessed_state::BlockAccessedState,
    network_cache::{MembershipDelta, MissResult, NetworkStateCache},
    participant::ParticipantCache,
    shared_trie::SharedSparseTrie,
};
use alloy_primitives::{
    keccak256,
    map::{B256Map, HashSet},
    Address, B256,
};
use reth_trie_common::{DecodedMultiProofV2, HashedPostState, Nibbles};
use reth_trie_sparse::{
    ParallelSparseTrie, RetainWitnessPathsMetrics, RetentionOptions, RevealableSparseTrie,
    SparseStateTrie, SparseTrie,
};
use std::{fmt, sync::Arc, time::Instant};

/// The sparse state trie this cache runs on.
///
/// The account trie is owned outright — every block rewrites the path from the root to each
/// changed account, so nothing about it is worth sharing — while storage tries are shared
/// copy-on-write with the generation the snapshot was taken from. See [`SharedSparseTrie`].
type CacheSparseStateTrie = SparseStateTrie<ParallelSparseTrie, SharedSparseTrie>;

/// Sparse trie plus the value-cache membership whose paths it is required to retain.
///
/// Cloning this type creates a transactional snapshot. Producers and validators apply a block to a
/// clone, check the post-state root and next anchor, and only then replace the previous cache.
///
/// The snapshot deep-copies the account trie and shares every storage trie with its parent until
/// something writes to it, so a block pays for the storage tries it dirties rather than for every
/// trie the cache retains. Both outcomes stay exact: a committed snapshot ends up owning what it
/// wrote and sharing the rest with a parent that is about to be dropped, and an abandoned one
/// drops its private copies and leaves the parent untouched.
#[derive(Debug)]
pub struct PartialTrieNodeCache {
    sparse: CacheSparseStateTrie,
    warm_accounts: HashSet<Address>,
    warm_storage: HashSet<(Address, B256)>,
    state_root: Option<B256>,
    /// The retained slot paths each storage trie was last pruned to, sorted and deduplicated.
    ///
    /// Retention is idempotent on a trie that has not been written to since it was pruned to the
    /// same paths, so this is what makes skipping it safe rather than merely cheap.
    retained_storage_paths: B256Map<Arc<[Nibbles]>>,
    /// The sorted, deduplicated account-trie paths the last prune retained.
    ///
    /// Kept so the next block can patch it with the ~5% of keys that moved instead of rehashing
    /// and re-sorting every warm account. Equal by construction to what a full rebuild produces —
    /// [`Self::retain_from_value_cache`] falls back to that rebuild whenever it cannot prove the
    /// value cache is exactly one block ahead of this state, and the differential test in
    /// `tests/delta_retention.rs` is what holds the two implementations to the same output.
    retained_account_paths: Vec<Nibbles>,
    /// The value-cache height the three derived sets above describe.
    ///
    /// `None` on a cache that has never retained. The incremental path is taken only when the
    /// value cache is exactly one block ahead of this, because that is the only distance the undo
    /// log can describe; every other distance — a gap, a rollback, a restore — falls back to the
    /// full rebuild rather than patching state whose base is unproven.
    synced_to_block: Option<u64>,
}

impl Clone for PartialTrieNodeCache {
    fn clone(&self) -> Self {
        let accounts = self
            .sparse
            .state_trie_ref()
            .map(|trie| RevealableSparseTrie::Revealed(Box::new(trie.clone())))
            .unwrap_or_else(RevealableSparseTrie::blind);
        let mut sparse = CacheSparseStateTrie::default().with_accounts_trie(accounts);

        // Copying the map wholesale is both cheaper and more faithful than rebuilding it from
        // warm membership: each value is a refcount bump, and `retain_from_value_cache` has
        // already reduced the map to exactly the tries warm membership requires. Allocation-reuse
        // buffers and process-local LFU history are deliberately not copied.
        let storage = sparse.storage_tries_mut();
        storage.reserve(self.sparse.storage_tries_ref().len());
        for (hashed_address, trie) in self.sparse.storage_tries_ref() {
            storage.insert(*hashed_address, trie.clone());
        }

        Self {
            sparse,
            warm_accounts: self.warm_accounts.clone(),
            warm_storage: self.warm_storage.clone(),
            state_root: self.state_root,
            retained_storage_paths: self.retained_storage_paths.clone(),
            retained_account_paths: self.retained_account_paths.clone(),
            synced_to_block: self.synced_to_block,
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
            sparse: CacheSparseStateTrie::default(),
            warm_accounts: HashSet::default(),
            warm_storage: HashSet::default(),
            state_root: None,
            retained_storage_paths: B256Map::default(),
            retained_account_paths: Vec::new(),
            synced_to_block: None,
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

    pub(crate) fn sparse_mut(&mut self) -> &mut CacheSparseStateTrie {
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
    ///
    /// A storage trie is pruned only when the transition wrote to it or when its retained slot set
    /// moved. Pruning anything else would reproduce the shape it already has, and under
    /// copy-on-write storage tries that no-op would cost a full copy of a trie the block never
    /// touched — the copy the snapshot exists to avoid.
    pub fn retain_from_value_cache(&mut self, value_cache: &NetworkStateCache) -> RetentionTimings {
        // The undo record names the keys the newest block moved, which is the ~5% this cache's
        // derived sets have to change. It is only usable when it describes the step from the
        // height these sets already reflect; anything else — a gap, a rollback, a restore, a
        // pruned undo log — falls back to the rebuild, which is always correct.
        let delta = match (self.synced_to_block, value_cache.last_block_membership_delta()) {
            (Some(synced), Some(delta)) if delta.block_number == synced + 1 => Some(delta),
            _ => None,
        };
        let timings = match delta {
            Some(delta) => self.retain_incrementally(&delta),
            None => {
                let mut timings = self.retain_fully(value_cache);
                timings.full_rebuild = true;
                timings
            }
        };
        self.synced_to_block = Some(value_cache.current_block());
        timings
    }

    /// Retains from scratch, discarding any incremental state.
    ///
    /// The reference [`Self::retain_from_value_cache`]'s delta path is held to: the two must leave
    /// the cache in the same state, and `tests/delta_retention.rs` is where that is enforced.
    pub fn retain_reference(&mut self, value_cache: &NetworkStateCache) -> RetentionTimings {
        let timings = self.retain_fully(value_cache);
        self.synced_to_block = Some(value_cache.current_block());
        timings
    }

    /// Commitment to everything retention derives, so two implementations can be compared as one
    /// value rather than field by field.
    ///
    /// Deliberately separate from [`Self::cache_root`], which commits warm membership and the
    /// state root — the protocol's surface. The retained *paths* are a local derivation of that
    /// membership, and this is what proves the derivation itself agrees.
    pub fn retention_fingerprint(&self) -> B256 {
        let mut preimage = Vec::new();
        preimage.extend_from_slice(b"PartialTrieNodeCacheRetention/v1");
        preimage.extend_from_slice(&(self.retained_account_paths.len() as u64).to_be_bytes());
        for path in &self.retained_account_paths {
            preimage.extend_from_slice(&path.to_vec());
        }
        let mut storage: Vec<_> = self.retained_storage_paths.iter().collect();
        storage.sort_unstable_by_key(|(hashed_address, _)| **hashed_address);
        preimage.extend_from_slice(&(storage.len() as u64).to_be_bytes());
        for (hashed_address, slots) in storage {
            preimage.extend_from_slice(hashed_address.as_slice());
            preimage.extend_from_slice(&(slots.len() as u64).to_be_bytes());
            for path in slots.iter() {
                preimage.extend_from_slice(&path.to_vec());
            }
        }
        keccak256(preimage)
    }

    /// Recomputes every retained path from the value cache, ignoring whatever was derived before.
    ///
    /// The reference implementation: correct from any starting state, and the oracle
    /// [`Self::retain_incrementally`] is held to.
    fn retain_fully(&mut self, value_cache: &NetworkStateCache) -> RetentionTimings {
        let mut timings = RetentionTimings::default();

        let start = Instant::now();
        self.warm_accounts = value_cache.accounts().keys().copied().collect();
        self.warm_storage = value_cache.storage().keys().copied().collect();
        timings.warm_membership_us = start.elapsed().as_micros() as u64;

        let start = Instant::now();
        let mut retained_accounts = self.warm_accounts.clone();
        let mut retained_storage = B256Map::<Vec<Nibbles>>::default();
        for (address, slot) in &self.warm_storage {
            retained_accounts.insert(*address);
            retained_storage
                .entry(keccak256(address))
                .or_default()
                .push(Nibbles::unpack(keccak256(slot)));
        }
        timings.storage_paths_us = start.elapsed().as_micros() as u64;

        let start = Instant::now();
        self.retained_account_paths = retained_accounts
            .into_iter()
            .map(|address| Nibbles::unpack(keccak256(address)))
            .collect();
        self.retained_account_paths.sort_unstable();
        self.retained_account_paths.dedup();
        timings.account_paths = self.retained_account_paths.len() as u64;
        timings.account_paths_us = start.elapsed().as_micros() as u64;

        let start = Instant::now();
        for slots in retained_storage.values_mut() {
            slots.sort_unstable();
            slots.dedup();
        }
        self.retained_storage_paths = retained_storage
            .into_iter()
            .map(|(hashed_address, slots)| (hashed_address, Arc::from(slots)))
            .collect();
        timings.storage_paths_us += start.elapsed().as_micros() as u64;

        (timings.account_trie_us, timings.account_trie) = self.prune_account_trie();
        // Nothing is known to be unmoved after a full rebuild, so every trie is pruned. This is
        // the cost the incremental path exists to avoid, not a case it has to reproduce.
        let (storage_us, pruned, skipped, storage_metrics) =
            self.prune_storage_tries(&B256Map::default(), true);
        timings.storage_tries_us = storage_us;
        timings.storage_tries = storage_metrics;
        timings.storage_tries_pruned = pruned;
        timings.storage_tries_skipped = skipped;
        timings
    }

    /// Patches the retained sets with the keys one block moved.
    ///
    /// Every step here is the delta-shaped equivalent of a line in [`Self::retain_fully`], and the
    /// two must produce byte-identical sets — that equality is a differential test, not a comment.
    fn retain_incrementally(&mut self, delta: &MembershipDelta) -> RetentionTimings {
        let mut timings = RetentionTimings::default();

        let start = Instant::now();
        // Capture the old retention state before changing either membership set. Account and
        // storage membership can both cross zero for the same address in one block. Reading the
        // old state after inserting the new warm account would make a genuinely new
        // account+storage pair look as though it had already been retained, leaving its account
        // path out of `retained_account_paths`.
        let mut affected = B256Map::<(Address, bool)>::default();
        for address in delta.accounts_removed.iter().chain(&delta.accounts_added) {
            let hashed = keccak256(address);
            affected
                .entry(hashed)
                .or_insert_with(|| (*address, self.is_retained_address(address, hashed)));
        }
        for (address, _) in delta.storage_removed.iter().chain(&delta.storage_added) {
            let hashed = keccak256(address);
            affected
                .entry(hashed)
                .or_insert_with(|| (*address, self.is_retained_address(address, hashed)));
        }
        for address in &delta.accounts_removed {
            self.warm_accounts.remove(address);
        }
        for address in &delta.accounts_added {
            self.warm_accounts.insert(*address);
        }
        for key in &delta.storage_removed {
            self.warm_storage.remove(key);
        }
        for key in &delta.storage_added {
            self.warm_storage.insert(*key);
        }
        timings.warm_membership_us = start.elapsed().as_micros() as u64;

        // Which addresses own a storage slot that moved, and how. Grouping first means an address
        // whose slots both entered and left rebuilds its sorted vector once.
        let start = Instant::now();
        let mut moved = B256Map::<StorageSlotDelta>::default();
        for (address, slot) in &delta.storage_removed {
            moved
                .entry(keccak256(address))
                .or_insert_with(StorageSlotDelta::new)
                .removed
                .push(Nibbles::unpack(keccak256(slot)));
        }
        for (address, slot) in &delta.storage_added {
            moved
                .entry(keccak256(address))
                .or_insert_with(StorageSlotDelta::new)
                .added
                .push(Nibbles::unpack(keccak256(slot)));
        }

        for (hashed_address, slots) in &moved {
            let updated = self.apply_slot_delta(*hashed_address, slots);
            if updated.is_empty() {
                self.retained_storage_paths.remove(hashed_address);
            } else {
                self.retained_storage_paths.insert(*hashed_address, Arc::from(updated));
            }
        }
        timings.storage_paths_us = start.elapsed().as_micros() as u64;

        // An address is retained when it owns a warm account entry or at least one warm slot. Now
        // that both membership dimensions have reached their new state, compare them with the
        // snapshot above exactly once per address. This covers account-only, storage-only, and
        // simultaneous account+storage transitions without order-dependent special cases.
        let start = Instant::now();
        let mut paths_added = Vec::new();
        let mut paths_removed = Vec::new();
        for (hashed_address, (address, was_retained)) in affected {
            let is_retained = self.is_retained_address(&address, hashed_address);
            match (was_retained, is_retained) {
                (false, true) => paths_added.push(Nibbles::unpack(hashed_address)),
                (true, false) => paths_removed.push(Nibbles::unpack(hashed_address)),
                _ => {}
            }
        }
        splice_sorted(&mut self.retained_account_paths, &mut paths_added, &paths_removed);
        timings.account_paths = self.retained_account_paths.len() as u64;
        timings.account_paths_us = start.elapsed().as_micros() as u64;

        (timings.account_trie_us, timings.account_trie) = self.prune_account_trie();
        let (storage_us, pruned, skipped, storage_metrics) =
            self.prune_storage_tries(&moved, false);
        timings.storage_tries_us = storage_us;
        timings.storage_tries = storage_metrics;
        timings.storage_tries_pruned = pruned;
        timings.storage_tries_skipped = skipped;
        timings
    }

    /// True when `address` is in the retained account-path set as the cache currently stands.
    fn is_retained_address(&self, address: &Address, hashed_address: B256) -> bool {
        self.warm_accounts.contains(address) ||
            self.retained_storage_paths
                .get(&hashed_address)
                .is_some_and(|slots| !slots.is_empty())
    }

    /// The address's new sorted slot-path set after `delta` is applied to the previous one.
    fn apply_slot_delta(&self, hashed_address: B256, delta: &StorageSlotDelta) -> Vec<Nibbles> {
        let mut slots: Vec<Nibbles> = self
            .retained_storage_paths
            .get(&hashed_address)
            .map(|previous| previous.to_vec())
            .unwrap_or_default();
        let mut added = delta.added.clone();
        splice_sorted(&mut slots, &mut added, &delta.removed);
        slots
    }

    /// Prunes the account trie to the retained paths, returning what it cost.
    fn prune_account_trie(&mut self) -> (u64, RetainWitnessPathsMetrics) {
        let start = Instant::now();
        let metrics = self
            .sparse
            .trie_mut()
            .as_revealed_mut()
            .map(|trie| {
                trie.retain_witness_paths_with_options(
                    &self.retained_account_paths,
                    RetentionOptions::sorted_input(),
                )
                .metrics
            })
            .unwrap_or_default();
        (start.elapsed().as_micros() as u64, metrics)
    }

    /// Prunes every storage trie the block could have moved, and drops the ones no longer retained.
    ///
    /// `moved` names the tries whose retained slot set changed. A trie outside it that the
    /// transition also never wrote to is already pruned to exactly these paths, and pruning it
    /// again would reproduce the shape it has — which under copy-on-write costs a full copy of a
    /// trie the block never touched, the copy the snapshot exists to avoid.
    fn prune_storage_tries(
        &mut self,
        moved: &B256Map<StorageSlotDelta>,
        prune_everything: bool,
    ) -> (u64, u64, u64, RetainWitnessPathsMetrics) {
        let start = Instant::now();
        let retained = std::mem::take(&mut self.retained_storage_paths);
        let mut pruned = 0;
        let mut skipped = 0;
        let mut metrics = RetainWitnessPathsMetrics::default();
        self.sparse.storage_tries_mut().retain(|hashed_address, trie| {
            let Some(slots) = retained.get(hashed_address) else { return false };
            // `as_revealed_ref` first: `as_revealed_mut` hands out a `&mut SharedSparseTrie`, which
            // is harmless on its own, but reaching for it before knowing whether the prune is
            // needed makes the skip easy to lose in a later edit.
            let untouched = trie.as_revealed_ref().is_some_and(SharedSparseTrie::is_untouched);
            let unchanged = !prune_everything && !moved.contains_key(hashed_address);
            if untouched && unchanged {
                skipped += 1;
            } else if let Some(trie) = trie.as_revealed_mut() {
                let outcome = trie
                    .make_mut()
                    .retain_witness_paths_with_options(slots, RetentionOptions::sorted_input());
                metrics.accumulate(&outcome.metrics);
                pruned += 1;
            }
            true
        });
        self.retained_storage_paths = retained;
        (start.elapsed().as_micros() as u64, pruned, skipped, metrics)
    }

    /// Storage tries this snapshot still shares with the generation it was cloned from.
    ///
    /// Reported as a delta against [`Self::storage_trie_count`] so a run can show how much of the
    /// old per-block deep copy the transition actually needed.
    pub fn shared_storage_trie_count(&self) -> usize {
        self.sparse
            .storage_tries_ref()
            .values()
            .filter(|trie| trie.as_revealed_ref().is_some_and(SharedSparseTrie::is_untouched))
            .count()
    }

    /// Storage tries the cache holds.
    pub fn storage_trie_count(&self) -> usize {
        self.sparse.storage_tries_ref().len()
    }

    /// Takes a private copy of every storage trie still shared with the parent generation.
    ///
    /// This reproduces the eager deep clone that the copy-on-write snapshot replaced, so a
    /// differential test can run one transition both ways and compare the results.
    pub fn materialize_shared_storage_tries(&mut self) -> usize {
        let mut copied = 0;
        for trie in self.sparse.storage_tries_mut().values_mut() {
            if let Some(trie) = trie.as_revealed_mut() &&
                trie.is_untouched()
            {
                trie.make_mut();
                copied += 1;
            }
        }
        copied
    }

    /// Structural equality of the account trie, every storage trie, membership, and the root.
    ///
    /// Deliberately not [`PartialEq`]: it walks every revealed node in every trie, which is a
    /// differential-test and diagnostic operation rather than something the hot path should reach
    /// for by accident.
    pub fn structurally_eq(&self, other: &Self) -> bool {
        self.state_root == other.state_root &&
            self.warm_accounts == other.warm_accounts &&
            self.warm_storage == other.warm_storage &&
            self.sparse.state_trie_ref() == other.sparse.state_trie_ref() &&
            self.sparse.storage_tries_ref() == other.sparse.storage_tries_ref()
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

    /// Of [`Self::estimated_memory_bytes`], the part that would actually be freed by dropping
    /// this cache.
    ///
    /// A snapshot shares storage tries with the generation it was cloned from, so the total
    /// counts the same allocation once per generation holding it. That is the right number for
    /// "how large is this cache" and the wrong one for "what does keeping it cost": the answer to
    /// the second is the account trie, which is never shared, plus only those storage tries this
    /// cache is the last owner of.
    ///
    /// Storage tries retained for allocation reuse after clearing are counted as exclusive
    /// because the sparse trie does not expose them, which makes this an upper bound on the
    /// marginal cost rather than an underestimate of it.
    pub fn exclusive_memory_bytes(&self) -> usize {
        let shared: usize = self
            .sparse
            .storage_tries_ref()
            .values()
            .filter_map(|trie| trie.as_revealed_ref())
            .filter(|trie| !trie.is_sole_owner())
            .map(SparseTrie::memory_size)
            .sum();
        self.estimated_memory_bytes().saturating_sub(shared)
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
        // Both the leaf count and the prefix coverage read from `dirtied_accounts`, which folds in
        // storage owners: a slot change rewrites its account's leaf even when the post state lists
        // no account entry for it. Counting only `changed.accounts` here would report a
        // storage-only block as dirtying no account leaf while the prefixes above it say otherwise.
        let dirtied_accounts = changed.dirtied_accounts();
        let account_prefixes = prefix_coverage(&retained_accounts, &dirtied_accounts);
        let dirtied_account_paths =
            retained_accounts.iter().filter(|key| dirtied_accounts.contains(*key)).count();

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

/// The slot paths one address gained and lost in a block.
///
/// Keyed by hashed address so it can be looked up against the storage-trie map directly.
#[derive(Debug, Clone)]
struct StorageSlotDelta {
    added: Vec<Nibbles>,
    removed: Vec<Nibbles>,
}

impl StorageSlotDelta {
    const fn new() -> Self {
        Self { added: Vec::new(), removed: Vec::new() }
    }
}

/// Removes `removed` from the sorted `target`, then merges `added` into it, keeping it sorted and
/// deduplicated.
///
/// One pass over `target` rather than a binary-search insert per key: a block moves thousands of
/// keys in a set of tens of thousands, so shifting the tail once beats shifting it once per key.
/// `added` is sorted in place because the caller has no use for its original order.
fn splice_sorted(target: &mut Vec<Nibbles>, added: &mut Vec<Nibbles>, removed: &[Nibbles]) {
    if !removed.is_empty() {
        let drop: HashSet<Nibbles> = removed.iter().copied().collect();
        target.retain(|path| !drop.contains(path));
    }
    if added.is_empty() {
        return
    }
    added.sort_unstable();
    added.dedup();

    let mut merged = Vec::with_capacity(target.len() + added.len());
    let mut left = target.iter().copied().peekable();
    let mut right = added.iter().copied().peekable();
    loop {
        match (left.peek(), right.peek()) {
            (Some(a), Some(b)) if a < b => merged.push(left.next().expect("peeked")),
            (Some(a), Some(b)) => {
                // Equal keys collapse: the full rebuild deduplicates through a set, so an address
                // already retained must not appear twice here either.
                if a == b {
                    left.next();
                }
                merged.push(right.next().expect("peeked"));
            }
            (Some(_), None) => merged.push(left.next().expect("peeked")),
            (None, Some(_)) => merged.push(right.next().expect("peeked")),
            (None, None) => break,
        }
    }
    *target = merged;
}

/// Where [`PartialTrieNodeCache::retain_from_value_cache`] spent a block's retention budget.
///
/// Retention is the largest validator phase, and its published cost has only ever been one
/// number. That number scales with the value cache at roughly the same rate per account as per
/// storage slot, which points at the per-key preparation rather than at the account-trie walk —
/// but pointing is not measuring. These fields separate the two so the next optimization is
/// aimed rather than guessed, and so `storage_tries_skipped` keeps reporting what the
/// untouched-trie skip is actually worth.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RetentionTimings {
    /// Rebuilding warm account and storage membership from the value cache's key sets.
    pub warm_membership_us: u64,
    /// Hashing every warm storage key into per-trie retained slot paths.
    pub storage_paths_us: u64,
    /// Hashing, sorting, and deduplicating the retained account paths.
    pub account_paths_us: u64,
    /// Pruning the account trie to those paths.
    pub account_trie_us: u64,
    /// Internal account-trie retention phases and work counters.
    pub account_trie: RetainWitnessPathsMetrics,
    /// Sorting each storage trie's slot set and pruning the tries that moved.
    pub storage_tries_us: u64,
    /// Aggregate internal retention phases and work counters for all pruned storage tries.
    pub storage_tries: RetainWitnessPathsMetrics,
    /// Retained account paths the account-trie prune was given.
    pub account_paths: u64,
    /// Storage tries whose prune ran.
    pub storage_tries_pruned: u64,
    /// Storage tries skipped because they were untouched and their slot set had not moved.
    pub storage_tries_skipped: u64,
    /// True when the retained sets were rebuilt from the whole value cache rather than patched.
    ///
    /// The delta path needs the value cache to be exactly one block ahead of the state the trie
    /// cache already reflects, which every ordinary block satisfies. Anything else — a rollback, a
    /// gap, a restore, an undo log pruned below finality — falls back here. Reported because the
    /// fallback is correct but expensive, so its *rate* is the thing worth watching in production:
    /// a run where it fires often has lost the optimization without losing correctness.
    pub full_rebuild: bool,
}

impl RetentionTimings {
    /// Time attributable to preparing key sets, as opposed to walking either trie.
    pub const fn preparation_us(&self) -> u64 {
        self.warm_membership_us
            .saturating_add(self.storage_paths_us)
            .saturating_add(self.account_paths_us)
    }

    /// Sum of the measured phases. Slightly below the caller's outer timer by the timer calls.
    pub const fn total_us(&self) -> u64 {
        self.preparation_us()
            .saturating_add(self.account_trie_us)
            .saturating_add(self.storage_tries_us)
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
