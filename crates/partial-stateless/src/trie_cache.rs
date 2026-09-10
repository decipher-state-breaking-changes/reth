//! Persistent sparse-trie cache for partial-stateless validation.
//!
//! The value cache answers reads. This cache keeps the corresponding account and storage proof
//! paths in a locally updated [`SparseStateTrie`]. A sidecar supplies parent-state miss paths and
//! execution updates this trie in place. The flat value cache is authoritative for hits and its
//! account/storage windows determine which inclusion or exclusion witness paths remain decoded.

use crate::{
    accessed_state::BlockAccessedState,
    cache_trie::{CacheTrie, CacheTrieRepr},
    network_cache::{MembershipDelta, MissResult, NetworkStateCache},
    participant::ParticipantCache,
    shared_trie::{self, SharedSparseTrie},
    trie_cache_undo::{
        next_undo_id, CacheStorageTrie, CacheUndoRecord, MembershipUndo, MembershipWhole,
        StorageTrieBefore, TrieCacheUndoFrame,
    },
};
use alloy_primitives::{
    keccak256,
    map::{B256Map, HashSet},
    Address, B256,
};
use reth_trie_common::{DecodedMultiProofV2, HashedPostState, Nibbles};
use reth_trie_sparse::{
    BranchSlotCensus, CloneBreakdown, CloneMeasureOptions, RetainWitnessPathsMetrics,
    RetentionOptions, RevealableSparseTrie, SparseStateTrie, SparseTrie,
};
use serde::Serialize;
use std::{
    fmt,
    num::NonZeroU64,
    sync::{Arc, OnceLock},
    time::Instant,
};

/// How much the account trie measures about its own shape, from `PS_TRIE_SHAPE_DIAGNOSTICS`.
///
/// Phase timers are always on: they ride along with work the block already does, and the splits
/// they give are the reason those phases are legible at all. Everything selected here is extra
/// work, so it is off unless a run asks:
///
/// - `1`, `on`: the copy's byte, allocation, and structural census, which walks every node and
///   value entry — something the copy itself never does, since a hash-map copy is a bulk operation
///   — plus the retention walk's obligatory-visit share and orphaned-mask count.
/// - `probe`: the above, plus the price of the unconditional branch-hash box, which means
///   allocating, copying, and freeing one per branch node.
///
/// All of them answer structural questions that move with cache size rather than with the block. A
/// 300-sample run measured the census at 8.94 ms, the probe at 9.11 ms, and the walk's descents at
/// 0.49 ms — together 4.7% of raw validation, enough to make a default-on run incomparable to one
/// without them and to distort the phase this workstream is trying to reduce.
fn shape_diagnostics() -> ShapeDiagnostics {
    static LEVEL: OnceLock<ShapeDiagnostics> = OnceLock::new();
    *LEVEL.get_or_init(|| match std::env::var("PS_TRIE_SHAPE_DIAGNOSTICS").as_deref() {
        Ok("probe") => ShapeDiagnostics::Probe,
        Ok("1" | "on" | "true" | "TRUE" | "yes") => ShapeDiagnostics::Census,
        _ => ShapeDiagnostics::Off,
    })
}

/// What the trie cache collects beyond the phase timers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShapeDiagnostics {
    /// Phase timers only, which is what a production block and a timing benchmark want.
    Off,
    /// Byte, allocation, and structural counts for the copy and the walks.
    Census,
    /// The census plus the measured price of the branch-hash box.
    Probe,
}

impl ShapeDiagnostics {
    const fn is_on(self) -> bool {
        !matches!(self, Self::Off)
    }

    const fn clone_options(self) -> CloneMeasureOptions {
        match self {
            Self::Off => CloneMeasureOptions::timers_only(),
            Self::Census => CloneMeasureOptions::accounting(),
            Self::Probe => CloneMeasureOptions::accounting().with_branch_hash_probe(),
        }
    }

    fn retention_options(self) -> RetentionOptions {
        let options = RetentionOptions::sorted_input();
        if self.is_on() {
            options.with_diagnostics()
        } else {
            options
        }
    }
}

/// The sparse state trie this cache runs on.
///
/// The account trie is owned outright — every block rewrites the path from the root to each
/// changed account, so nothing about it is worth sharing — while storage tries are shared
/// copy-on-write with the generation the snapshot was taken from. See [`SharedSparseTrie`].
type CacheSparseStateTrie = SparseStateTrie<CacheTrie, SharedSparseTrie<CacheTrie>>;

/// A blind account-trie slot pre-seeded with the right representation, so the first reveal
/// builds on it instead of falling back to the wrapper's default trie.
fn blind_account_trie(repr: CacheTrieRepr) -> RevealableSparseTrie<CacheTrie> {
    RevealableSparseTrie::Blind(Some(Box::new(CacheTrie::new(repr))))
}

/// A state trie whose account slot and storage-trie template both carry the representation.
fn sparse_state_trie_for(repr: CacheTrieRepr) -> CacheSparseStateTrie {
    CacheSparseStateTrie::default()
        .with_accounts_trie(blind_account_trie(repr))
        .with_default_storage_trie(RevealableSparseTrie::Blind(Some(Box::new(
            SharedSparseTrie::new(CacheTrie::new(repr)),
        ))))
}

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
    /// The trie representation every trie in this cache (account and storage) is built on.
    repr: CacheTrieRepr,
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
    /// Warm-set sizing policy and the interval state it needs.
    ///
    /// Carried through a clone so a snapshot continues its parent's interval instead of restarting
    /// it, and restored with a generation on a reorg — which is the consistent choice, since the
    /// bucket counts the interval is tracking are restored with it too.
    ///
    /// Living inside the cache rather than beside it is also what makes a rejected block free:
    /// retention runs on the candidate snapshot, so a block that is refused discards the interval
    /// advance along with the trie it was counted against, and the parent's count is untouched.
    warm_shrink: WarmShrink,
    /// Which generation this cache is, so a frame can name the two ends it joins.
    ///
    /// Fresh on every construction and on every clone. A retained generation is identified by
    /// this and not by its block number, for the reason [`crate::trie_cache_undo`] gives:
    /// mid-reorg a height names whichever block the database currently calls canonical.
    undo_id: u64,
    /// Whether a clone of this cache records what the block does to it.
    ///
    /// Carried through a clone the way the shrink policy is, so it is set once on the pair's live
    /// cache and every working copy inherits it. Off by default: recording costs a lookup per
    /// write on the hot path, and its holder has a correct fallback — keeping whole generations —
    /// which is what a cache on the `Parallel` representation gets whatever this says.
    record_undo: bool,
    /// The record of what this block has done to this cache so far, when one is being kept.
    ///
    /// Only ever started by [`Self::clone_timed`]: a record describes the step from one generation
    /// to the next, and a cache that was not cloned from anything has no such step to describe.
    undo: Option<CacheUndoRecord>,
}

impl Clone for PartialTrieNodeCache {
    fn clone(&self) -> Self {
        self.clone_timed().0
    }
}

impl PartialTrieNodeCache {
    /// [`Clone::clone`], reporting which of the snapshot's four copies the time went to.
    ///
    /// The account trie is the copy storage-trie sharing deliberately left in place, because
    /// sharing it needs node-granular structural sharing inside the trie itself. But it is not the
    /// only copy: warm membership is two hash sets holding every cached account and slot, and the
    /// retained-path indexes hold one `Nibbles` per retained account plus an `Arc` per storage
    /// trie. Those three are proportional to cache *size* rather than to the block's changes, and
    /// an `Arc` would share them, so they are a much smaller change than sharing trie nodes.
    /// Splitting the timer is what decides whether that change is worth making.
    pub fn clone_timed(&self) -> (Self, TrieCloneTimings) {
        let mut timings = TrieCloneTimings::default();

        // Timed from inside the copy rather than around it, so the instrumentation the breakdown
        // costs stays out of the phase number this run reports for the phase.
        let accounts = self
            .sparse
            .state_trie_ref()
            .map(|trie| {
                let (copy, breakdown) = trie.clone_measured(shape_diagnostics().clone_options());
                timings.account_trie_breakdown = breakdown;
                RevealableSparseTrie::Revealed(Box::new(copy))
            })
            .unwrap_or_else(|| blind_account_trie(self.repr));
        let mut sparse = sparse_state_trie_for(self.repr).with_accounts_trie(accounts);
        timings.account_trie_us = timings.account_trie_breakdown.total_us;

        // Copying the map wholesale is both cheaper and more faithful than rebuilding it from
        // warm membership: each value is a refcount bump, and `retain_from_value_cache` has
        // already reduced the map to exactly the tries warm membership requires. Allocation-reuse
        // buffers and process-local LFU history are deliberately not copied.
        let start = Instant::now();
        let storage = sparse.storage_tries_mut();
        storage.reserve(self.sparse.storage_tries_ref().len());
        for (hashed_address, trie) in self.sparse.storage_tries_ref() {
            storage.insert(*hashed_address, trie.clone());
        }
        timings.storage_tries_us = start.elapsed().as_micros() as u64;
        timings.storage_tries = self.sparse.storage_tries_ref().len() as u64;

        let start = Instant::now();
        let warm_accounts = self.warm_accounts.clone();
        let warm_storage = self.warm_storage.clone();
        timings.warm_membership_us = start.elapsed().as_micros() as u64;
        timings.warm_accounts = warm_accounts.len() as u64;
        timings.warm_storage = warm_storage.len() as u64;

        let start = Instant::now();
        let retained_storage_paths = self.retained_storage_paths.clone();
        let retained_account_paths = self.retained_account_paths.clone();
        timings.retained_paths_us = start.elapsed().as_micros() as u64;
        timings.retained_account_paths = retained_account_paths.len() as u64;

        let mut copy = Self {
            sparse,
            repr: self.repr,
            warm_accounts,
            warm_storage,
            state_root: self.state_root,
            retained_storage_paths,
            retained_account_paths,
            synced_to_block: self.synced_to_block,
            warm_shrink: self.warm_shrink,
            undo_id: next_undo_id(),
            record_undo: self.record_undo,
            undo: None,
        };
        // The working copy starts recording here and not a line later, because everything the
        // block does to it — the transition, the retention pass, the prune — has to be inside the
        // record for the frame to describe the whole step from this parent. The account trie's own
        // clone already carries the record forward when the parent was recording; this begins one
        // when it was not, and is a no-op when it was.
        if copy.record_undo {
            copy.begin_undo(self.undo_id);
        }
        (copy, timings)
    }
}

impl Default for PartialTrieNodeCache {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialTrieNodeCache {
    /// Creates a cold local sparse trie on the default representation.
    pub fn new() -> Self {
        Self::new_with_repr(CacheTrieRepr::default())
    }

    /// Creates a cold local sparse trie on the given representation.
    ///
    /// Every trie this cache ever creates — the account trie on first reveal and each storage
    /// trie template — inherits it, so a cache never mixes representations.
    pub fn new_with_repr(repr: CacheTrieRepr) -> Self {
        Self {
            sparse: sparse_state_trie_for(repr),
            repr,
            warm_accounts: HashSet::default(),
            warm_storage: HashSet::default(),
            state_root: None,
            retained_storage_paths: B256Map::default(),
            retained_account_paths: Vec::new(),
            synced_to_block: None,
            warm_shrink: WarmShrink::default(),
            undo_id: next_undo_id(),
            record_undo: false,
            undo: None,
        }
    }

    /// Configures how often the warm sets are returned to a fitted size.
    ///
    /// Resets the interval state, so a policy changed mid-run measures its own interval rather
    /// than inheriting a high-water mark taken under the previous one.
    pub fn set_warm_shrink_policy(&mut self, policy: WarmSetShrinkPolicy) {
        self.warm_shrink = WarmShrink { policy, ..WarmShrink::default() };
    }

    /// The warm-set sizing policy this cache and its snapshots run under.
    pub const fn warm_shrink_policy(&self) -> WarmSetShrinkPolicy {
        self.warm_shrink.policy
    }

    /// The trie representation this cache runs on.
    pub const fn repr(&self) -> CacheTrieRepr {
        self.repr
    }

    /// Which generation this cache is.
    ///
    /// The identity a [`TrieCacheUndoFrame`] names at both ends. Stable for the life of the
    /// object, changed only by [`Self::undo`], which makes the cache a different generation.
    pub const fn undo_id(&self) -> u64 {
        self.undo_id
    }

    /// Sets whether clones of this cache record what a block does to them.
    ///
    /// Set on the pair's live cache; every working copy inherits it through the clone, and so
    /// does every generation the pair retains. Turning it off does not end a record already in
    /// progress on *this* cache — that record describes a step this switch has no opinion about —
    /// it decides what the next clone does.
    pub const fn set_undo_recording(&mut self, record: bool) {
        self.record_undo = record;
    }

    /// Whether clones of this cache record.
    pub const fn records_undo(&self) -> bool {
        self.record_undo
    }

    /// Whether this cache is keeping a record right now.
    pub const fn is_recording_undo(&self) -> bool {
        self.undo.is_some()
    }

    /// Starts recording the step from the generation `parent` names.
    ///
    /// Private because a record has exactly one legitimate starting point: the clone that begins
    /// a block. Starting one anywhere else would produce a frame that claims to describe a step
    /// nothing took, and the whole recovery path downstream trusts that claim.
    fn begin_undo(&mut self, parent: u64) {
        let mut record =
            CacheUndoRecord::new(parent, self.state_root, self.synced_to_block, self.warm_shrink);
        match self.sparse.trie_mut().as_revealed_mut() {
            Some(trie) => {
                trie.begin_undo();
                // `Parallel` carries no record, so a cache on it poisons here and its holder keeps
                // whole generations. That is the two-representation split of section 5.1 landing
                // at runtime rather than a second journal in `parallel.rs`.
                record.poisoned = !trie.is_recording_undo();
            }
            // A blind slot holds nothing to preimage. Recorded rather than poisoned, because a
            // trie still blind at the commit changed nothing; the reveal that ends that state is
            // caught where the record is taken.
            None => record.account_blind = true,
        }
        self.undo = Some(record);
    }

    /// Ends any record in progress, keeping nothing.
    ///
    /// The preimages a record holds are the block's, not the cache's, so a generation nobody will
    /// ask for a frame from should not go on holding them.
    pub fn clear_undo_record(&mut self) {
        self.undo = None;
        if let Some(trie) = self.sparse.trie_mut().as_revealed_mut() {
            drop(trie.take_undo());
        }
    }

    /// Ends the record and returns the frame that turns this cache back into `parent`.
    ///
    /// `parent` is the generation this cache was cloned from and is consumed in the process: the
    /// storage tries it holds and this cache no longer does are *moved* into the frame, because
    /// the caller is about to drop it and a refcount bump would be a copy of a decision already
    /// made. It is left holding blind placeholders for those addresses.
    ///
    /// `None` — with the record ended either way — when this cache was not recording, when the
    /// record was poisoned, or when `parent` is not the generation the record describes. Every one
    /// of those means the caller keeps `parent` whole instead, which is correct at any depth and
    /// only costs memory.
    pub fn take_undo_frame(&mut self, parent: &mut Self) -> Option<TrieCacheUndoFrame> {
        let record = self.undo.take();
        let account = self.sparse.trie_mut().as_revealed_mut().and_then(CacheTrie::take_undo);
        let record = record?;
        if record.poisoned || record.parent != parent.undo_id {
            return None
        }
        // The slot's reveal state, *read* rather than inferred from whether a record came back. A
        // trie that was blind when recording began and is revealed now has moved a whole trie into
        // existence, which no preimage here describes — and its `take_undo` returns `None` exactly
        // as a still-blind slot's does, because nothing ever told it to record. Inferring from
        // `account` alone therefore mistakes a reveal for "blind throughout" and produces a frame
        // that leaves the revealed content in place while claiming to be the generation below it.
        let blind_now = self.sparse.state_trie_ref().is_none();
        if blind_now != record.account_blind {
            return None
        }
        // Revealed at both ends, but no record: recording never began on this trie.
        if blind_now != account.is_none() {
            return None
        }
        Some(TrieCacheUndoFrame {
            source: self.undo_id,
            target: parent.undo_id,
            account,
            storage: self.storage_undo_against(parent),
            membership: record.membership,
            state_root: record.state_root,
            synced_to_block: record.synced_to_block,
            warm_shrink: record.warm_shrink,
        })
    }

    /// Whether a frame could be applied to this cache at all.
    ///
    /// The representation and the account trie's reveal state, checked before [`Self::undo`]
    /// touches anything, so that call either does the whole undo or none of it.
    pub fn can_undo(&self, frame: &TrieCacheUndoFrame) -> bool {
        self.repr == CacheTrieRepr::Exact &&
            (!frame.records_account_trie() || self.sparse.state_trie_ref().is_some())
    }

    /// Reverses `frame`, making this cache the generation the frame describes.
    ///
    /// The frame must have been taken from a cache whose content equalled this one's — which
    /// [`TrieCacheUndoFrame::source`] against [`Self::undo_id`] is how a caller checks — with no
    /// other change since. It is a replay of preimages and not a merge, so applying one to the
    /// wrong generation corrupts silently; nothing below can detect it.
    ///
    /// Returns `false`, having changed nothing, when this cache cannot hold a frame at all. What
    /// is deliberately *not* restored is the sparse state trie's scratch: the cleared-trie pool,
    /// the rlp buffer, the deferred drops and the two LFUs. A working copy resets those every
    /// block by building a fresh `SparseStateTrie`, an undone cache keeps whatever it had, and
    /// none of them is read by `cache_root`, `retention_fingerprint` or `structurally_eq`. Warm-set
    /// *capacity* is the same kind of difference and is the known price of a frame over a whole
    /// generation: a generation carries the bucket count of its own creation, a frame does not
    /// take one back.
    #[must_use]
    pub fn undo(&mut self, frame: TrieCacheUndoFrame) -> bool {
        if !self.can_undo(&frame) {
            return false
        }
        debug_assert_eq!(
            self.undo_id, frame.source,
            "a frame is being applied to a generation it was not recorded against"
        );
        // The record in progress described the content this call is about to replace.
        self.clear_undo_record();

        if let Some(account) = frame.account {
            let applied = self
                .sparse
                .trie_mut()
                .as_revealed_mut()
                .expect("can_undo checked the account trie is revealed")
                .undo(account);
            debug_assert!(applied, "can_undo checked the representation carries a record");
        }

        let storage_tries = self.sparse.storage_tries_mut();
        for (hashed_address, before) in frame.storage {
            match before {
                StorageTrieBefore::Held(trie) => {
                    storage_tries.insert(hashed_address, *trie);
                }
                StorageTrieBefore::Absent => {
                    storage_tries.remove(&hashed_address);
                }
            }
        }

        // The whole preimage first, then the delta recorded before the rebuild that produced it.
        // With no rebuild there is no whole; with a rebuild before anything else there is no
        // delta. The order is what makes the two-pass case exact rather than nearly right.
        let MembershipUndo { whole, delta } = frame.membership;
        if let Some(whole) = whole {
            let MembershipWhole {
                warm_accounts,
                warm_storage,
                retained_account_paths,
                retained_storage_paths,
            } = *whole;
            self.warm_accounts = warm_accounts;
            self.warm_storage = warm_storage;
            self.retained_account_paths = retained_account_paths;
            self.retained_storage_paths = retained_storage_paths;
        }
        for (address, was_present) in delta.warm_accounts {
            if was_present {
                self.warm_accounts.insert(address);
            } else {
                self.warm_accounts.remove(&address);
            }
        }
        for (key, was_present) in delta.warm_storage {
            if was_present {
                self.warm_storage.insert(key);
            } else {
                self.warm_storage.remove(&key);
            }
        }
        let mut restored = Vec::new();
        let mut dropped = Vec::new();
        for (path, was_present) in delta.account_paths {
            if was_present {
                restored.push(path);
            } else {
                dropped.push(path);
            }
        }
        splice_sorted(&mut self.retained_account_paths, &mut restored, &dropped);
        for (hashed_address, before) in delta.storage_paths {
            match before {
                Some(paths) => {
                    self.retained_storage_paths.insert(hashed_address, paths);
                }
                None => {
                    self.retained_storage_paths.remove(&hashed_address);
                }
            }
        }

        self.state_root = frame.state_root;
        self.synced_to_block = frame.synced_to_block;
        self.warm_shrink = frame.warm_shrink;
        self.undo_id = frame.target;
        true
    }

    /// The storage-trie half of the frame that turns this cache back into `parent`.
    ///
    /// A pointer comparison per retained trie and a move for the few that moved — 100 of ~3,630
    /// on the measured corpus, p95 148 — rather than a hook inside `make_mut`, which would need a
    /// per-block generation stamp on every handle to tell a first touch from a tenth. The two maps
    /// exist side by side exactly once, at the commit that displaces `parent`, and that is the one
    /// moment the comparison is available for free.
    ///
    /// An entry with no revealed trie behind it has no `Arc` to compare, so it is recorded rather
    /// than assumed unchanged. Recording it costs nothing here: `parent` is being dropped, so the
    /// handle is moved out of it.
    fn storage_undo_against(&self, parent: &mut Self) -> Vec<(B256, StorageTrieBefore)> {
        fn identity(trie: &CacheStorageTrie) -> Option<usize> {
            trie.as_revealed_ref().map(SharedSparseTrie::allocation_id)
        }

        let mut undo = Vec::new();
        let mine = self.sparse.storage_tries_ref();
        for (hashed_address, held) in parent.sparse.storage_tries_mut() {
            let unchanged = match (identity(held), mine.get(hashed_address).and_then(identity)) {
                (Some(before), Some(now)) => before == now,
                _ => false,
            };
            if !unchanged {
                undo.push((
                    *hashed_address,
                    StorageTrieBefore::Held(Box::new(std::mem::take(held))),
                ));
            }
        }
        let held_before = parent.sparse.storage_tries_ref();
        for hashed_address in mine.keys() {
            if !held_before.contains_key(hashed_address) {
                undo.push((*hashed_address, StorageTrieBefore::Absent));
            }
        }
        undo
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

    pub(crate) fn sparse_ref(&self) -> &CacheSparseStateTrie {
        &self.sparse
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
        let mut timings = match delta {
            Some(delta) => self.retain_incrementally(&delta),
            None => {
                let mut timings = self.retain_fully(value_cache);
                timings.full_rebuild = true;
                timings
            }
        };
        self.synced_to_block = Some(value_cache.current_block());
        (timings.warm_shrink_us, timings.warm_shrink) = self.maintain_warm_capacity();
        timings
    }

    /// Records this block against the interval's high-water mark, and returns the warm sets to a
    /// fitted size when the interval closes.
    ///
    /// Runs after retention rather than before it, so the mark describes membership as the block
    /// left it and a shrink is not immediately undone by the removals of the block that triggered
    /// it. Deliberately absent from [`Self::retain_reference`]: that path is the differential
    /// oracle for what the sets *contain*, and capacity is in neither fingerprint it is compared
    /// on, so running a sizing policy there would add cost to the reference without adding
    /// coverage.
    fn maintain_warm_capacity(&mut self) -> (u64, bool) {
        let Some(interval) = self.warm_shrink.policy.interval() else { return (0, false) };

        self.warm_shrink.accounts_high_water =
            self.warm_shrink.accounts_high_water.max(self.warm_accounts.len());
        self.warm_shrink.storage_high_water =
            self.warm_shrink.storage_high_water.max(self.warm_storage.len());
        self.warm_shrink.blocks_since_shrink += 1;
        if self.warm_shrink.blocks_since_shrink < interval.get() {
            return (0, false)
        }

        let start = Instant::now();
        self.warm_accounts.shrink_to(self.warm_shrink.accounts_high_water);
        self.warm_storage.shrink_to(self.warm_shrink.storage_high_water);
        let elapsed = start.elapsed().as_micros() as u64;

        // The next interval starts its mark from what is held now rather than from what the last
        // one peaked at, so membership that has genuinely fallen is allowed to keep the ground.
        self.warm_shrink.blocks_since_shrink = 0;
        self.warm_shrink.accounts_high_water = self.warm_accounts.len();
        self.warm_shrink.storage_high_water = self.warm_storage.len();
        (elapsed, true)
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

        // A rebuild replaces all four derived structures at once, so there is no delta to reverse
        // and the record takes the whole previous state as one preimage — the same bulk rule
        // `JournaledMap` applies to a map a `clear` empties. It is the one retention path that
        // costs the record a copy proportional to the cache, and it is the path that runs on 0 of
        // 10,005 blocks in the steady state: a rebuild is what a gap, a restore or a rollback
        // forces, and every one of those clears the deque anyway.
        let whole = self.undo.is_some().then(|| MembershipWhole {
            warm_accounts: self.warm_accounts.clone(),
            warm_storage: self.warm_storage.clone(),
            retained_account_paths: self.retained_account_paths.clone(),
            retained_storage_paths: self.retained_storage_paths.clone(),
        });
        if let Some((record, whole)) = self.undo.as_mut().zip(whole) {
            record.record_whole(whole);
        }

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
        timings.record_storage_prune(self.prune_storage_tries(&B256Map::default(), true));
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
        self.record_warm_membership(delta);
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
            self.record_storage_paths(*hashed_address);
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
        self.record_account_paths(paths_added.iter().chain(&paths_removed));
        splice_sorted(&mut self.retained_account_paths, &mut paths_added, &paths_removed);
        timings.account_paths = self.retained_account_paths.len() as u64;
        timings.account_paths_us = start.elapsed().as_micros() as u64;

        (timings.account_trie_us, timings.account_trie) = self.prune_account_trie();
        timings.record_storage_prune(self.prune_storage_tries(&moved, false));
        timings
    }

    /// Records what warm membership held for every key `delta` is about to move.
    ///
    /// First write wins: a key this block has already touched keeps the preimage from that first
    /// touch, which is the only one that describes the generation the frame restores. Read from
    /// the sets themselves rather than trusted from the delta's own added/removed split, so a
    /// delta that disagreed with this cache's membership would produce a record that still puts
    /// the cache back where it was.
    fn record_warm_membership(&mut self, delta: &MembershipDelta) {
        let Self { undo, warm_accounts, warm_storage, .. } = self;
        let Some(record) = undo.as_mut().and_then(CacheUndoRecord::delta_mut) else { return };
        for address in delta.accounts_removed.iter().chain(&delta.accounts_added) {
            record.warm_accounts.entry(*address).or_insert_with(|| warm_accounts.contains(address));
        }
        for key in delta.storage_removed.iter().chain(&delta.storage_added) {
            record.warm_storage.entry(*key).or_insert_with(|| warm_storage.contains(key));
        }
    }

    /// Records what `retained_storage_paths` holds for `hashed_address` before it is replaced.
    fn record_storage_paths(&mut self, hashed_address: B256) {
        let Self { undo, retained_storage_paths, .. } = self;
        let Some(record) = undo.as_mut().and_then(CacheUndoRecord::delta_mut) else { return };
        record
            .storage_paths
            .entry(hashed_address)
            .or_insert_with(|| retained_storage_paths.get(&hashed_address).cloned());
    }

    /// Records whether each of `paths` was in the retained account-path set before the splice.
    ///
    /// The caller has just computed these as the paths that enter and leave, so their preimages
    /// are known — but they are looked up anyway, for the reason `record_warm_membership` gives
    /// and because the set is sorted, which makes the lookup a binary search over a few hundred
    /// keys rather than a scan.
    fn record_account_paths<'a>(&mut self, paths: impl Iterator<Item = &'a Nibbles>) {
        let Self { undo, retained_account_paths, .. } = self;
        let Some(record) = undo.as_mut().and_then(CacheUndoRecord::delta_mut) else { return };
        for path in paths {
            record
                .account_paths
                .entry(*path)
                .or_insert_with(|| retained_account_paths.binary_search(path).is_ok());
        }
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
                    shape_diagnostics().retention_options(),
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
    ) -> StoragePruneOutcome {
        let start = Instant::now();
        let copies_before = shared_trie::cow_copies_taken();
        let retained = std::mem::take(&mut self.retained_storage_paths);
        let mut outcome = StoragePruneOutcome::default();
        // Tries whose address left the retained set are moved out here and freed together below,
        // so the cost of releasing a whole storage trie is measured rather than folded into the
        // map scan that discovered it.
        let mut evicted_tries = Vec::new();
        self.sparse.storage_tries_mut().retain(|hashed_address, trie| {
            let Some(slots) = retained.get(hashed_address) else {
                evicted_tries.push(std::mem::take(trie));
                return false;
            };
            // `as_revealed_ref` first: `as_revealed_mut` hands out a `&mut SharedSparseTrie`, which
            // is harmless on its own, but reaching for it before knowing whether the prune is
            // needed makes the skip easy to lose in a later edit.
            let untouched = trie.as_revealed_ref().is_some_and(SharedSparseTrie::is_untouched);
            let unchanged = !prune_everything && !moved.contains_key(hashed_address);
            if untouched && unchanged {
                outcome.skipped += 1;
            } else if let Some(trie) = trie.as_revealed_mut() {
                // `make_mut` is timed apart from the walk it precedes. A trie still shared with
                // the retained generation is copied whole here, before the walk reads a single
                // node — transactional-snapshot cost that lands inside retention's timer rather
                // than the clone phase's, and that the walk's own phases cannot see.
                let copy = Instant::now();
                trie.make_mut();
                outcome.cow_us += copy.elapsed().as_micros() as u64;

                let walk = trie.make_mut().retain_witness_paths_with_options(
                    slots,
                    shape_diagnostics().retention_options(),
                );
                outcome.metrics.accumulate(&walk.metrics);
                outcome.pruned += 1;
            }
            true
        });
        self.retained_storage_paths = retained;

        outcome.dropped = evicted_tries.len() as u64;
        let release = Instant::now();
        drop(evicted_tries);
        outcome.drop_us = release.elapsed().as_micros() as u64;

        outcome.cow_copies = shared_trie::cow_copies_taken().saturating_sub(copies_before);
        outcome.total_us = start.elapsed().as_micros() as u64;
        outcome
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

    /// Where this cache's memory is, component by component.
    ///
    /// [`Self::estimated_memory_bytes`] reports the sparse trie and nothing else, which is the
    /// figure every published cohort quotes and is deliberately left alone. It is also incomplete:
    /// the two warm sets and the two retained-path indexes are copied per generation and cost
    /// real bytes that no consumer measurement has ever included. This reports all of them, so a
    /// retained generation's true cost is known rather than bounded below.
    ///
    /// Heuristic in the same sense as `memory_size`, and charged against what each container holds
    /// from the allocator rather than what it currently stores. For a `Vec` that is `capacity`.
    /// For a hash table it is neither `len` nor `capacity` — see [`hashbrown_table_bytes`], which
    /// is where that distinction turned out to matter.
    pub fn memory_breakdown(&self) -> TrieCacheMemory {
        let nibble = std::mem::size_of::<Nibbles>();
        let table = |capacity: usize, entry: usize| hashbrown_table_bytes(capacity, entry);

        let shared_storage_trie_bytes = self
            .sparse
            .storage_tries_ref()
            .values()
            .filter_map(|trie| trie.as_revealed_ref())
            .filter(|trie| !trie.is_sole_owner())
            .map(SparseTrie::memory_size)
            .sum();

        TrieCacheMemory {
            sparse_bytes: self.sparse.memory_size(),
            shared_storage_trie_bytes,
            warm_accounts_bytes: table(
                self.warm_accounts.capacity(),
                std::mem::size_of::<Address>(),
            ),
            warm_storage_bytes: table(
                self.warm_storage.capacity(),
                std::mem::size_of::<(Address, B256)>(),
            ),
            // `Vec::capacity` *is* the allocation, so this one needs no reconstruction.
            retained_account_paths_bytes: self.retained_account_paths.capacity() * nibble,
            retained_storage_paths_map_bytes: table(
                self.retained_storage_paths.capacity(),
                std::mem::size_of::<B256>() + std::mem::size_of::<Arc<[Nibbles]>>(),
            ),
            retained_storage_paths_slice_bytes: self
                .retained_storage_paths
                .values()
                .map(|paths| paths.len() * nibble)
                .sum(),
        }
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

    /// Every allocation this cache may be sharing with another generation, as `(identity, bytes)`.
    ///
    /// Two structures qualify. The revealed storage tries are shared by `SharedSparseTrie`'s
    /// `Arc`, which is the whole point of the copy-on-write layer; and the `Arc<[Nibbles]>` slices
    /// behind `retained_storage_paths` are shared by a clone of the map, which copies the handles
    /// and not the slices.
    ///
    /// Identity rather than a boolean, because `exclusive_memory_bytes` is defined against exactly
    /// one other holder and stops being meaningful past it. What a caller holding K generations
    /// needs is the union — one entry per allocation however many generations point at it — and
    /// only an identity supports that.
    ///
    /// Addresses are valid only while every handle involved is alive, which a measurement taken
    /// against a live pair satisfies. They are not identities to store.
    pub fn shared_allocations(&self) -> Vec<(usize, usize)> {
        let nibble = std::mem::size_of::<Nibbles>();
        let tries = self
            .sparse
            .storage_tries_ref()
            .values()
            .filter_map(|trie| trie.as_revealed_ref())
            .map(|trie| (trie.allocation_id(), trie.memory_size()));
        let slices = self
            .retained_storage_paths
            .values()
            .map(|paths| (Arc::as_ptr(paths) as *const () as usize, paths.len() * nibble));
        tries.chain(slices).collect()
    }

    /// Of [`Self::memory_breakdown`], the part no other generation can be holding.
    ///
    /// The complement of [`Self::shared_allocations`] within the same total, so the two partition
    /// the cache: a caller unions the shared halves across generations and adds these unchanged.
    pub fn unshared_bytes(&self) -> usize {
        let total = self.memory_breakdown().total_bytes();
        let shared: usize = self.shared_allocations().iter().map(|(_, bytes)| bytes).sum();
        total.saturating_sub(shared)
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

    /// Path-and-hash of every hash-addressed node the account trie currently reveals.
    ///
    /// The path is part of the identity, not a convenience: byte-identical nodes can appear at
    /// several paths of one trie, and a node revealed at one path proves nothing about the same
    /// bytes needed at another. Dirty nodes carry no cached hash and are skipped, so this is
    /// complete only on a cache whose root has been computed — which every committed generation
    /// has.
    pub fn revealed_account_nodes(&self) -> HashSet<(Nibbles, B256)> {
        let mut nodes = HashSet::default();
        if let Some(trie) = self.sparse.state_trie_ref() {
            trie.for_each_cached_node_hash(|path, hash| {
                nodes.insert((*path, hash));
            });
        }
        nodes
    }

    /// Per-storage-trie revealed node paths and hashes, keyed by hashed account address.
    ///
    /// Kept per trie rather than pooled, and per path within a trie, for the same reason as
    /// [`Self::revealed_account_nodes`]: identical bytes under another trie or at another path
    /// are a different node.
    pub fn revealed_storage_nodes(&self) -> B256Map<HashSet<(Nibbles, B256)>> {
        let mut tries = B256Map::default();
        for (hashed_address, trie) in self.sparse.storage_tries_ref() {
            let Some(shared) = trie.as_revealed_ref() else { continue };
            let mut nodes = HashSet::default();
            shared.shared_ref().for_each_cached_node_hash(|path, hash| {
                nodes.insert((*path, hash));
            });
            tries.insert(*hashed_address, nodes);
        }
        tries
    }

    /// Branch child-slot occupancy across the account trie and every storage trie.
    ///
    /// Diagnostic: walks every revealed branch node, so it belongs beside the other opt-in
    /// censuses rather than on the per-block hot path.
    pub fn branch_slot_census(&self) -> TrieBranchCensus {
        let account =
            self.sparse.state_trie_ref().map(CacheTrie::branch_slot_census).unwrap_or_default();
        let mut storage = BranchSlotCensus::default();
        let mut storage_tries = 0u64;
        for trie in self.sparse.storage_tries_ref().values() {
            if let Some(shared) = trie.as_revealed_ref() {
                storage_tries += 1;
                storage.accumulate(&shared.shared_ref().branch_slot_census());
            }
        }
        TrieBranchCensus { account, storage, storage_tries }
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

/// How often the warm sets are returned to a size fitted to their contents.
///
/// hashbrown sizes a table for the churn it has seen rather than for what it holds. When growth
/// pressure arrives it rehashes in place if the live items still fit in half the buckets, and
/// otherwise resizes to the next power of two — "conservatively resize to at least the next size
/// up to avoid churning deletes into frequent rehashes", in `reserve_rehash_inner`'s own words.
/// Steady-state warm membership sits above that half line, so each set settles at the smallest
/// power of two above `max_items * 16/7`, exactly one doubling past the `max_items * 8/7` its
/// contents need, and stays there. Measured on the 10,000-block corpus: 2^17 and 2^18 buckets
/// against 41,000 and 71,000 entries — 15.88 MiB where 8.32 MiB fits, an 18-27% load factor. Since
/// a clone copies `buckets()` and never reads `items`, the whole doubling is paid again on every
/// block's snapshot, which is what makes the sizing a latency question and not only a memory one.
///
/// Shrinking takes the doubling back and buys into the regime that rule exists to avoid: at the
/// fitted size the live items no longer fit in half the buckets, so the next growth pressure
/// resizes instead of rehashing in place, and the shrink has to be repeated. Whether the per-block
/// copy saved outweighs those resizes is a property of the workload's removal rate and is not
/// derivable from the sizes alone — so this is a measured knob, and [`Self::Never`] is the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WarmSetShrinkPolicy {
    /// Leave hashbrown's own sizing alone. Today's behaviour, and the production default.
    #[default]
    Never,
    /// Shrink both sets every `n` blocks, to the high-water mark of the interval just ended.
    ///
    /// The high-water mark rather than the current length, because membership oscillates within a
    /// run — 23,000-41,000 warm accounts and 51,000-71,000 warm slots on the measured corpus — and
    /// fitting a trough only guarantees a resize on the way back up.
    EveryBlocks(NonZeroU64),
}

impl WarmSetShrinkPolicy {
    /// The interval in blocks, or `None` when shrinking is off.
    pub const fn interval(self) -> Option<NonZeroU64> {
        match self {
            Self::Never => None,
            Self::EveryBlocks(n) => Some(n),
        }
    }
}

impl fmt::Display for WarmSetShrinkPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Never => f.write_str("never"),
            Self::EveryBlocks(n) => write!(f, "{n}"),
        }
    }
}

/// [`WarmSetShrinkPolicy`] together with the interval state it needs.
///
/// `Copy`, so carrying it through a clone is a field assignment rather than a decision.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct WarmShrink {
    policy: WarmSetShrinkPolicy,
    blocks_since_shrink: u64,
    accounts_high_water: usize,
    storage_high_water: usize,
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
    /// Copy-on-write copies taken before the storage walks, and what they cost.
    ///
    /// A trie still shared with the retained generation is copied whole by `make_mut` before the
    /// walk reads a node. That is transactional-snapshot cost charged to retention's timer rather
    /// than the clone phase's, and the walk's own input/traversal/mutation/finalization phases
    /// cannot see it — which is why `storage_tries_us` exceeds their sum.
    pub storage_trie_cow_us: u64,
    /// Storage tries the prune actually copied, as opposed to already owning outright.
    pub storage_trie_cow_copies: u64,
    /// Releasing storage tries whose address left the retained set.
    pub storage_trie_drop_us: u64,
    /// Storage tries dropped because their address is no longer retained.
    pub storage_tries_dropped: u64,
    /// True when the retained sets were rebuilt from the whole value cache rather than patched.
    ///
    /// The delta path needs the value cache to be exactly one block ahead of the state the trie
    /// cache already reflects, which every ordinary block satisfies. Anything else — a rollback, a
    /// gap, a restore, an undo log pruned below finality — falls back here. Reported because the
    /// fallback is correct but expensive, so its *rate* is the thing worth watching in production:
    /// a run where it fires often has lost the optimization without losing correctness.
    pub full_rebuild: bool,
    /// Returning the warm sets to a fitted size, when [`WarmSetShrinkPolicy`] asked for one.
    pub warm_shrink_us: u64,
    /// True on the block that closed a shrink interval.
    ///
    /// Reported so a run can divide the cost by the blocks it was amortised over, and so an arm
    /// configured to shrink but never reaching an interval boundary reads as a misconfiguration
    /// rather than as a null result.
    pub warm_shrink: bool,
}

/// One storage-prune pass, split into the parts that scale differently.
///
/// The walk metrics scale with the tries actually pruned, the copies with how many of those were
/// still shared with the retained generation, and the scan with the size of the storage-trie map.
/// Kept as one value so the two callers cannot record a partial set of them.
#[derive(Debug, Default)]
struct StoragePruneOutcome {
    total_us: u64,
    cow_us: u64,
    cow_copies: u64,
    drop_us: u64,
    pruned: u64,
    skipped: u64,
    dropped: u64,
    metrics: RetainWitnessPathsMetrics,
}

impl RetentionTimings {
    /// Folds one storage-prune pass into this block's retention timings.
    fn record_storage_prune(&mut self, outcome: StoragePruneOutcome) {
        self.storage_tries_us = outcome.total_us;
        self.storage_tries = outcome.metrics;
        self.storage_tries_pruned = outcome.pruned;
        self.storage_tries_skipped = outcome.skipped;
        self.storage_trie_cow_us = outcome.cow_us;
        self.storage_trie_cow_copies = outcome.cow_copies;
        self.storage_trie_drop_us = outcome.drop_us;
        self.storage_tries_dropped = outcome.dropped;
    }

    /// Storage-prune time the measured walk phases and the copies do not account for.
    ///
    /// What is left is the pass over the storage-trie map itself and the skip decisions it makes.
    /// Reported as a residual rather than timed directly because bracketing the closure body would
    /// cost a timer call per trie against a per-trie body of a hash lookup and two flag reads.
    pub const fn storage_trie_scan_us(&self) -> u64 {
        let walk = self
            .storage_tries
            .input_us
            .saturating_add(self.storage_tries.traversal_us)
            .saturating_add(self.storage_tries.mutation_us)
            .saturating_add(self.storage_tries.finalization_us);
        self.storage_tries_us
            .saturating_sub(walk)
            .saturating_sub(self.storage_trie_cow_us)
            .saturating_sub(self.storage_trie_drop_us)
    }

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
            .saturating_add(self.warm_shrink_us)
    }
}

/// What opening a transactional snapshot copied, split by component.
///
/// `trie_clone_us` was a single timer over [`PartialTrieNodeCache::clone`], which made it read as
/// "the account-trie deep copy" even though three size-proportional copies ride along with it.
/// The counters beside each time are the populations those copies scale with.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TrieCloneTimings {
    /// Deep-copying the revealed account trie. The copy storage-trie sharing did not remove.
    pub account_trie_us: u64,
    /// Where that copy's time, bytes, and allocations went. Included in `account_trie_us`.
    ///
    /// The phase has been the largest single one since V2 landed, and one timer over it cannot say
    /// whether a narrower node representation would help or whether the cost is spread evenly
    /// across everything the trie holds. Its own instrumentation terms — `accounting_us` and
    /// `branch_hash_probe_us` — are outside `account_trie_us` and reported so a run carrying them
    /// stays comparable to one that does not.
    pub account_trie_breakdown: CloneBreakdown,
    /// Copying the storage-trie map. One refcount bump per entry, not a trie copy.
    pub storage_tries_us: u64,
    /// Copying the warm account and storage key sets.
    pub warm_membership_us: u64,
    /// Copying the retained account-path vector and the per-trie retained-slot map.
    pub retained_paths_us: u64,
    /// Storage tries whose `Arc` was bumped.
    pub storage_tries: u64,
    /// Warm accounts and slots copied, which is the whole value-cache key population.
    pub warm_accounts: u64,
    pub warm_storage: u64,
    /// Retained account paths copied.
    pub retained_account_paths: u64,
}

impl TrieCloneTimings {
    /// The copies that scale with cache size rather than with the account trie's node count.
    ///
    /// Reported together because they share one possible fix — sharing these three behind an
    /// `Arc` and copying on write — which is independent of, and far cheaper than, node-granular
    /// sharing inside the account trie.
    pub const fn membership_and_paths_us(&self) -> u64 {
        self.storage_tries_us
            .saturating_add(self.warm_membership_us)
            .saturating_add(self.retained_paths_us)
    }

    /// Sum of the measured phases. Slightly below the caller's outer timer by the timer calls.
    pub const fn total_us(&self) -> u64 {
        self.account_trie_us.saturating_add(self.membership_and_paths_us())
    }
}

/// Branch child-slot occupancy for both cache tries, from
/// [`PartialTrieNodeCache::branch_slot_census`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TrieBranchCensus {
    /// The account trie's census.
    pub account: BranchSlotCensus,
    /// Every revealed storage trie's census, accumulated.
    pub storage: BranchSlotCensus,
    /// Revealed storage tries the storage census covers.
    pub storage_tries: u64,
}

/// What a hashbrown table of this reported capacity actually holds from the allocator.
///
/// `HashMap::capacity` is *not* the allocation. hashbrown reports `items + growth_left`, and
/// erasing an entry leaves a tombstone that consumes a bucket without returning growth — so a
/// table that has had entries removed reports less capacity than it has buckets, while the bucket
/// array is unchanged. Retention removes warm accounts every block, so this cache is exactly the
/// shape that drifts: a measured run saw the warm-account table report anywhere from 54,079 to
/// 57,344 while holding 65,536 buckets throughout.
///
/// Reconstructed from the reported capacity rather than from `len`, because both errors exist and
/// they point opposite ways: `len` misses the buckets a table keeps after shrinking, and
/// `capacity` misses the ones tombstones hide. Rounding up to hashbrown's own power-of-two bucket
/// count absorbs the second — the tombstone deficit has to exceed half the table before it changes
/// which power of two you land on, and a table that far gone has bigger problems.
///
/// Still a heuristic: it is hashbrown's sizing rule restated, not a reading of the allocation, and
/// it omits the trailing control group hashbrown replicates for its SIMD probe. Both are small
/// against a table of this size, and neither is a reason to report a number that is knowably low.
pub(crate) fn hashbrown_table_bytes(capacity: usize, entry_bytes: usize) -> usize {
    // hashbrown's `capacity_to_buckets`, which is the only place the mapping is defined.
    let buckets = match capacity {
        0 => return 0,
        1..=3 => 4,
        4..=7 => 8,
        capacity => (capacity * 8 / 7).next_power_of_two(),
    };
    // One control byte per bucket, beside the bucket itself.
    buckets * (entry_bytes + 1)
}

/// Where one trie cache's memory is, component by component.
///
/// Exists because `estimated_memory_bytes` is the sparse trie alone: the four structures beside it
/// are copied per generation and were outside every memory figure a cohort has published, so the
/// cost of retaining a generation was known only as a lower bound.
///
/// The `Arc<[Nibbles]>` slices are split out from the map that holds them because they are shared
/// with any generation cloned from this one — the map is per-generation, the slices are not, and
/// a K-generation total has to union them rather than add them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct TrieCacheMemory {
    /// `SparseStateTrie::memory_size()`: the account trie, every revealed storage trie, and the
    /// cleared tries held for allocation reuse. Unchanged from `estimated_memory_bytes`.
    pub sparse_bytes: usize,
    /// Of `sparse_bytes`, the storage tries this cache is not the sole owner of.
    pub shared_storage_trie_bytes: usize,
    pub warm_accounts_bytes: usize,
    pub warm_storage_bytes: usize,
    pub retained_account_paths_bytes: usize,
    /// The `B256Map` itself — keys, `Arc` handles, control bytes — not what the handles point at.
    pub retained_storage_paths_map_bytes: usize,
    /// The `Arc<[Nibbles]>` slices behind that map.
    pub retained_storage_paths_slice_bytes: usize,
}

impl TrieCacheMemory {
    /// Everything this cache holds, sharing counted once per holder.
    pub const fn total_bytes(&self) -> usize {
        self.sparse_bytes +
            self.warm_accounts_bytes +
            self.warm_storage_bytes +
            self.retained_account_paths_bytes +
            self.retained_storage_paths_map_bytes +
            self.retained_storage_paths_slice_bytes
    }

    /// What dropping this cache would actually free, against one other holder.
    ///
    /// The same definition `exclusive_memory_bytes` uses, extended to the four structures it
    /// omitted. All four are unshared per generation, so only the storage tries subtract.
    pub const fn exclusive_bytes(&self) -> usize {
        self.total_bytes().saturating_sub(self.shared_storage_trie_bytes)
    }

    /// Of `total_bytes`, the part `estimated_memory_bytes` never counted.
    pub const fn beyond_sparse_bytes(&self) -> usize {
        self.total_bytes() - self.sparse_bytes
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
        trie_cache_undo::TrieCacheUndoFrame,
        NetworkStateCache,
    };
    use alloy_primitives::{map::B256Map, U256};
    use alloy_rlp::encode_fixed_size;
    use reth_trie::test_utils::TrieTestHarness;
    use reth_trie_common::ProofV2Target;
    use reth_trie_sparse::{ExactSparseTrie, LeafUpdate};
    use std::collections::BTreeMap;

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

    /// Splitting the clone's timer must not change what the snapshot contains, and each counter
    /// must describe the copy it sits beside — those counters are what a per-entry cost for the
    /// size-proportional copies would be divided by.
    #[test]
    fn timed_clone_matches_the_plain_clone_and_counts_each_copy() {
        let address = Address::repeat_byte(0x11);
        let slot = B256::repeat_byte(0x22);
        let mut accessed = BlockAccessedState::default();
        accessed
            .accounts
            .insert(address, AccountData { nonce: 3, balance: U256::from(7), code_hash: None });
        accessed.storage.insert((address, slot), U256::from(1));

        let mut values = value_cache();
        values.on_block_executed(1, &accessed);
        let mut trie = PartialTrieNodeCache::new();
        trie.retain_from_value_cache(&values);
        trie.set_state_root(B256::repeat_byte(0x44));

        let (timed, timings) = trie.clone_timed();
        assert_eq!(timed.cache_root(), trie.clone().cache_root());
        assert_eq!(timings.warm_accounts, trie.tracked_account_count() as u64);
        assert_eq!(timings.warm_storage, trie.tracked_storage_slot_count() as u64);
        assert_eq!(timings.retained_account_paths, trie.retained_account_paths.len() as u64);
        assert_eq!(timings.storage_tries, trie.sparse.storage_tries_ref().len() as u64);
        assert_eq!(timings.total_us(), timings.account_trie_us + timings.membership_and_paths_us());
    }

    #[test]
    fn memory_breakdown_counts_what_estimated_memory_bytes_leaves_out() {
        let address = Address::repeat_byte(0x11);
        let slot = B256::repeat_byte(0x22);
        let mut accessed = BlockAccessedState::default();
        accessed
            .accounts
            .insert(address, AccountData { nonce: 3, balance: U256::from(7), code_hash: None });
        accessed.storage.insert((address, slot), U256::from(1));

        let mut values = value_cache();
        values.on_block_executed(1, &accessed);
        let mut trie = PartialTrieNodeCache::new();
        trie.retain_from_value_cache(&values);

        let memory = trie.memory_breakdown();

        // The published definition is unchanged: the sparse component *is* the old figure.
        assert_eq!(memory.sparse_bytes, trie.estimated_memory_bytes());

        // The warm sets are populated, so the part the old figure omitted is real rather than a
        // structure that happens to be empty on this fixture — which is the whole claim.
        assert!(trie.tracked_account_count() > 0, "fixture must warm an account");
        assert!(trie.tracked_storage_slot_count() > 0, "fixture must warm a slot");
        assert!(memory.warm_accounts_bytes > 0);
        assert!(memory.warm_storage_bytes > 0);
        assert_eq!(memory.beyond_sparse_bytes(), memory.total_bytes() - memory.sparse_bytes);
        assert!(memory.beyond_sparse_bytes() > 0);
        assert!(memory.total_bytes() > trie.estimated_memory_bytes());

        // Exclusive is the same relation the old pair had, extended to the new components: with no
        // second holder nothing is shared, so exclusive and total agree.
        assert_eq!(memory.shared_storage_trie_bytes, 0);
        assert_eq!(memory.exclusive_bytes(), memory.total_bytes());
        assert_eq!(trie.exclusive_memory_bytes(), trie.estimated_memory_bytes());
    }

    #[test]
    fn cloning_a_generation_compacts_capacity_so_the_two_are_not_byte_equal() {
        let address = Address::repeat_byte(0x11);
        let slot = B256::repeat_byte(0x22);
        let mut accessed = BlockAccessedState::default();
        accessed
            .accounts
            .insert(address, AccountData { nonce: 3, balance: U256::from(7), code_hash: None });
        accessed.storage.insert((address, slot), U256::from(1));

        let mut values = value_cache();
        values.on_block_executed(1, &accessed);
        let mut trie = PartialTrieNodeCache::new();
        trie.retain_from_value_cache(&values);

        let live = trie.memory_breakdown();
        let retained = trie.clone().memory_breakdown();

        // The trie itself copies exactly. What does not is `retained_account_paths`: `Vec::clone`
        // allocates for the length, not the capacity, so a retained generation can be strictly
        // smaller than the live cache it was copied from. Retention grows the live vector again
        // the next block, which is the same ratchet §5.5 names from the other direction — and the
        // reason a K-generation total has to be measured per generation rather than as K times one.
        assert_eq!(retained.sparse_bytes, live.sparse_bytes);
        assert!(
            retained.retained_account_paths_bytes <= live.retained_account_paths_bytes,
            "a clone cannot hold more path capacity than its source"
        );
        assert!(retained.total_bytes() <= live.total_bytes());
    }

    #[test]
    fn a_hash_table_is_charged_its_buckets_and_not_its_reported_capacity() {
        // hashbrown's own sizing rule at the boundaries it defines.
        assert_eq!(hashbrown_table_bytes(0, 21), 0, "an unallocated table costs nothing");
        assert_eq!(hashbrown_table_bytes(3, 21), 4 * 22);
        assert_eq!(hashbrown_table_bytes(4, 21), 8 * 22);
        assert_eq!(hashbrown_table_bytes(7, 21), 8 * 22);

        // The case this function exists for, from a measured run: retention erases warm accounts
        // every block, and each erasure leaves a tombstone that consumes a bucket without
        // returning growth. The reported capacity drifted across 194 distinct values between
        // 54,079 and 57,344 while the bucket array never moved off 65,536.
        let full = hashbrown_table_bytes(57_344, 21);
        assert_eq!(full, 65_536 * 22, "57,344 is exactly 65,536 buckets' worth");
        assert_eq!(
            hashbrown_table_bytes(54_079, 21),
            full,
            "a table riddled with tombstones holds the same buckets it always did"
        );

        // And the rounding does not swallow a genuine growth step: one item past what 65,536
        // buckets can hold is a table twice the size, and it is charged as one.
        assert_eq!(hashbrown_table_bytes(57_345, 21), 131_072 * 22);
    }

    #[test]
    fn mutation_metrics_measure_the_block_against_the_trie_it_is_applied_to() {
        // Eight accounts, so the two-nibble prefix level has something to discriminate on. The
        // lower-subtrie proposal turns on exactly this level: how many of the 256 two-nibble
        // subtries a block leaves alone.
        let addresses: Vec<Address> = (1u8..=8).map(Address::repeat_byte).collect();
        let mut accessed = BlockAccessedState::default();
        for (nonce, address) in addresses.iter().enumerate() {
            accessed.accounts.insert(
                *address,
                AccountData { nonce: nonce as u64, balance: U256::from(1), code_hash: None },
            );
        }
        let mut values = value_cache();
        values.on_block_executed(1, &accessed);
        let mut trie = PartialTrieNodeCache::new();
        trie.retain_from_value_cache(&values);

        // A block that changed nothing dirties nothing, at every depth. This is the assertion that
        // catches the measurement being taken after the commit instead of before it: against the
        // child generation every retained path reads as dirtied, and depth 0 would be 1 of 1.
        let untouched = trie.mutation_metrics(&TrieChangeSet::default());
        assert!(untouched.retained_account_paths > 0, "the fixture retains something to dirty");
        assert_eq!(untouched.dirtied_account_paths, 0);
        for depth in 0..TRIE_SHAPE_PREFIX_LEVELS {
            assert_eq!(untouched.account_prefixes[depth].dirtied, 0);
        }

        // One account changed. Prefix coverage is monotone in depth on the retained side — deeper
        // levels split what shallower ones grouped — and one leaf can dirty at most one prefix per
        // depth.
        let one = TrieChangeSet {
            accounts: [keccak256(addresses[0])].into_iter().collect(),
            ..Default::default()
        };
        let one = trie.mutation_metrics(&one);
        assert_eq!(one.dirtied_account_paths, 1);
        for depth in 0..TRIE_SHAPE_PREFIX_LEVELS {
            assert_eq!(
                one.account_prefixes[depth].dirtied, 1,
                "one leaf, one prefix at depth {depth}"
            );
            assert!(one.account_prefixes[depth].retained >= one.account_prefixes[depth].dirtied);
        }
        assert_eq!(one.account_prefixes[0].retained, 1, "depth zero is the root, always one");

        // Everything changed. Dirtied meets retained at every depth, which is the saturation the
        // proposal predicts for a block that writes enough accounts.
        let all = TrieChangeSet {
            accounts: addresses.iter().copied().map(keccak256).collect(),
            ..Default::default()
        };
        let all = trie.mutation_metrics(&all);
        assert_eq!(all.dirtied_account_paths, all.retained_account_paths);
        for depth in 0..TRIE_SHAPE_PREFIX_LEVELS {
            assert_eq!(all.account_prefixes[depth].dirtied, all.account_prefixes[depth].retained);
        }
        assert!(
            all.account_prefixes[2].retained > 1,
            "eight distinct accounts should spread over more than one two-nibble prefix"
        );
    }

    #[test]
    fn a_clone_shares_its_retained_path_slices_by_identity() {
        let address = Address::repeat_byte(0x11);
        let slot = B256::repeat_byte(0x22);
        let mut accessed = BlockAccessedState::default();
        accessed
            .accounts
            .insert(address, AccountData { nonce: 3, balance: U256::from(7), code_hash: None });
        accessed.storage.insert((address, slot), U256::from(1));

        let mut values = value_cache();
        values.on_block_executed(1, &accessed);
        let mut trie = PartialTrieNodeCache::new();
        trie.retain_from_value_cache(&values);

        let shared = trie.shared_allocations();
        assert!(!shared.is_empty(), "a cache with a warm slot has a retained-path slice to share");

        // The property the union rests on: a clone points at the *same* allocations, so a caller
        // holding both must count each once. Identity, not equal byte counts — two separately
        // allocated slices of the same length would compare equal on size and be double-counted.
        let clone = trie.clone();
        let mine: HashSet<usize> = shared.iter().map(|(id, _)| *id).collect();
        let theirs: HashSet<usize> = clone.shared_allocations().iter().map(|(id, _)| *id).collect();
        assert_eq!(mine, theirs, "a clone shares every allocation rather than copying it");

        // And the two halves partition the cache, which is what lets a caller union the shared
        // side and add the unshared side unchanged.
        let shared_bytes: usize = shared.iter().map(|(_, bytes)| bytes).sum();
        assert_eq!(trie.unshared_bytes() + shared_bytes, trie.memory_breakdown().total_bytes());
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

    /// A value cache that forgets an untouched account after two blocks.
    ///
    /// Short on purpose: the production ratchet takes ~1,600 blocks to saturate, and no unit test
    /// can wait for that. What a fast window buys is the shape the policy has to handle — a set
    /// whose length collapses while the table it lives in does not follow.
    fn fast_forgetting_value_cache() -> NetworkStateCache {
        NetworkStateCache::new(
            Box::new(LastNBlocksPolicy::new(2)),
            Box::new(LastNBlocksPolicy::new(2)),
        )
    }

    fn account_at(index: usize) -> Address {
        let mut bytes = [0u8; 20];
        bytes[..8].copy_from_slice(&(index as u64).to_be_bytes());
        Address::from(bytes)
    }

    fn block_touching(indices: std::ops::Range<usize>) -> BlockAccessedState {
        let mut accessed = BlockAccessedState::default();
        for index in indices {
            accessed.accounts.insert(
                account_at(index),
                AccountData { nonce: 0, balance: U256::ZERO, code_hash: None },
            );
        }
        accessed
    }

    /// Drives one block through both caches and hands back what retention reported.
    fn advance(
        trie: &mut PartialTrieNodeCache,
        values: &mut NetworkStateCache,
        block: u64,
        touched: std::ops::Range<usize>,
    ) -> RetentionTimings {
        values.on_block_executed(block, &block_touching(touched));
        trie.retain_from_value_cache(values)
    }

    #[test]
    fn the_default_policy_leaves_a_table_exactly_where_it_found_it() {
        let mut values = fast_forgetting_value_cache();
        let mut trie = PartialTrieNodeCache::new();
        assert_eq!(trie.warm_shrink_policy(), WarmSetShrinkPolicy::Never);

        let timings = advance(&mut trie, &mut values, 1, 0..5_000);
        assert!(!timings.warm_shrink, "the default policy has no interval to close");
        assert_eq!(timings.warm_shrink_us, 0);
        let peak_bytes = trie.memory_breakdown().warm_accounts_bytes;
        assert_eq!(trie.tracked_account_count(), 5_000);

        for block in 2..=8 {
            assert!(!advance(&mut trie, &mut values, block, 5_000..5_001).warm_shrink);
        }

        // The membership collapsed by three orders of magnitude and the allocation did not move.
        // This is the ratchet in miniature, and it is what `Never` is a decision to keep.
        assert_eq!(trie.tracked_account_count(), 1);
        assert_eq!(trie.memory_breakdown().warm_accounts_bytes, peak_bytes);
    }

    #[test]
    fn an_interval_protects_the_peak_it_covered_and_takes_the_ground_in_the_next_one() {
        let mut values = fast_forgetting_value_cache();
        let mut trie = PartialTrieNodeCache::new();
        trie.set_warm_shrink_policy(WarmSetShrinkPolicy::EveryBlocks(
            NonZeroU64::new(5).expect("5 is not zero"),
        ));

        advance(&mut trie, &mut values, 1, 0..5_000);
        let peak_bytes = trie.memory_breakdown().warm_accounts_bytes;
        for block in 2..=4 {
            assert!(!advance(&mut trie, &mut values, block, 5_000..5_001).warm_shrink);
        }

        // Block 5 closes an interval whose high-water mark is the 5,000 of block 1, even though
        // the set holds one account by the time it lands. Fitting what is held here would size the
        // table for a trough the workload has already left.
        let closing = advance(&mut trie, &mut values, 5, 5_000..5_001);
        assert!(closing.warm_shrink);
        assert_eq!(trie.tracked_account_count(), 1);
        assert_eq!(
            trie.memory_breakdown().warm_accounts_bytes,
            peak_bytes,
            "the interval that saw the peak must not shrink below it"
        );

        // The next interval saw only the trough, so it is the one entitled to the ground.
        for block in 6..=9 {
            assert!(!advance(&mut trie, &mut values, block, 5_000..5_001).warm_shrink);
        }
        assert!(advance(&mut trie, &mut values, 10, 5_000..5_001).warm_shrink);
        assert!(
            trie.memory_breakdown().warm_accounts_bytes < peak_bytes / 100,
            "an interval that only ever saw one account should fit one account"
        );
    }

    #[test]
    fn shrinking_the_warm_sets_changes_nothing_the_cache_commits_to() {
        // `tests/delta_retention.rs` compares the full and incremental retention paths through
        // `cache_root` and `retention_fingerprint`, and capacity appears in neither — so a sizing
        // policy is invisible to the differential oracle that covers everything else about these
        // sets, and needs its own statement of what it must not disturb.
        let mut values = fast_forgetting_value_cache();
        let mut shrinking = PartialTrieNodeCache::new();
        shrinking.set_warm_shrink_policy(WarmSetShrinkPolicy::EveryBlocks(
            NonZeroU64::new(2).expect("2 is not zero"),
        ));
        let mut left_alone = PartialTrieNodeCache::new();
        // A second value cache driven identically, because `NetworkStateCache` is not `Clone` and
        // the two tries have to be fed from equal-but-separate state to be compared at all.
        let mut control_values = fast_forgetting_value_cache();

        let mut shrank = false;
        for block in 1..=9usize {
            let touched = if block == 1 { 0..5_000 } else { 5_000 + block..5_001 + block };
            shrank |=
                advance(&mut shrinking, &mut values, block as u64, touched.clone()).warm_shrink;
            advance(&mut left_alone, &mut control_values, block as u64, touched);

            assert_eq!(shrinking.cache_root(), left_alone.cache_root());
            assert_eq!(shrinking.retention_fingerprint(), left_alone.retention_fingerprint());
            assert_eq!(shrinking.tracked_account_count(), left_alone.tracked_account_count());
            assert_eq!(shrinking.synced_to_block, left_alone.synced_to_block);
        }

        assert!(shrank, "the fixture has to actually shrink or it proves nothing");
        assert!(
            shrinking.memory_breakdown().warm_accounts_bytes <
                left_alone.memory_breakdown().warm_accounts_bytes,
            "the two agree on everything committed and differ only in what they hold from the \
             allocator, which is the whole point of the change"
        );
    }

    fn block_touching_slots(indices: std::ops::Range<usize>) -> BlockAccessedState {
        let owner = Address::repeat_byte(0x77);
        let mut accessed = BlockAccessedState::default();
        accessed
            .accounts
            .insert(owner, AccountData { nonce: 0, balance: U256::ZERO, code_hash: None });
        for index in indices {
            let mut slot = [0u8; 32];
            slot[..8].copy_from_slice(&(index as u64).to_be_bytes());
            accessed.storage.insert((owner, B256::from(slot)), U256::from(1));
        }
        accessed
    }

    #[test]
    fn shrinking_the_storage_set_changes_nothing_the_cache_commits_to() {
        // The storage set is 83% of the bytes and the one whose allocation crosses the size that
        // matters, so the equivalence has to be shown on it and not only on the account set.
        let mut values = fast_forgetting_value_cache();
        let mut control_values = fast_forgetting_value_cache();
        let mut shrinking = PartialTrieNodeCache::new();
        shrinking.set_warm_shrink_policy(WarmSetShrinkPolicy::EveryBlocks(
            NonZeroU64::new(2).expect("2 is not zero"),
        ));
        let mut left_alone = PartialTrieNodeCache::new();

        let mut shrank = false;
        for block in 1..=9usize {
            let touched = if block == 1 { 0..5_000 } else { 5_000 + block..5_001 + block };
            values.on_block_executed(block as u64, &block_touching_slots(touched.clone()));
            control_values.on_block_executed(block as u64, &block_touching_slots(touched));
            shrank |= shrinking.retain_from_value_cache(&values).warm_shrink;
            left_alone.retain_from_value_cache(&control_values);

            assert_eq!(shrinking.cache_root(), left_alone.cache_root());
            assert_eq!(shrinking.retention_fingerprint(), left_alone.retention_fingerprint());
            assert_eq!(
                shrinking.tracked_storage_slot_count(),
                left_alone.tracked_storage_slot_count()
            );
            assert_eq!(shrinking.synced_to_block, left_alone.synced_to_block);
        }

        assert!(shrank, "the fixture has to actually shrink or it proves nothing");
        assert!(
            shrinking.memory_breakdown().warm_storage_bytes <
                left_alone.memory_breakdown().warm_storage_bytes / 100,
            "five thousand slots aged out; the shrunk set should fit the handful that remain"
        );
    }

    #[test]
    fn a_candidate_refused_on_an_interval_boundary_leaves_its_parent_untouched() {
        // Retention runs on the candidate snapshot, so a refused block's interval advance and any
        // shrink it took are discarded with the candidate. The parent must neither lose its
        // allocation nor have its count moved — the second would close its own interval early.
        let mut values = fast_forgetting_value_cache();
        let mut parent = PartialTrieNodeCache::new();
        parent.set_warm_shrink_policy(WarmSetShrinkPolicy::EveryBlocks(
            NonZeroU64::new(4).expect("4 is not zero"),
        ));
        // Interval one (blocks 1-4) sees the peak and protects it. Its boundary lands at block 4,
        // after the value cache has aged the peak out, so the mark it seeds the next interval
        // with is one account — a boundary any earlier would seed the peak and make interval two
        // protect it as well. Interval two (blocks 5-8) only ever sees one account.
        advance(&mut parent, &mut values, 1, 0..5_000);
        for block in 2..=7 {
            advance(&mut parent, &mut values, block, 5_000..5_001);
        }
        let held = parent.memory_breakdown().warm_accounts_bytes;

        // Block 8 is validated on a snapshot: it closes interval two there and shrinks to one
        // account. Then it is refused, and the snapshot is dropped.
        let mut candidate = parent.clone();
        values.on_block_executed(8, &block_touching(5_000..5_001));
        assert!(candidate.retain_from_value_cache(&values).warm_shrink);
        assert!(candidate.memory_breakdown().warm_accounts_bytes < held / 100);
        drop(candidate);

        assert_eq!(
            parent.memory_breakdown().warm_accounts_bytes,
            held,
            "the parent kept its table"
        );
        // The parent's own block 8 is the one that closes its interval. Had the candidate's tick
        // moved the parent's count, this would already be a fresh interval and would not close.
        let timings = parent.retain_from_value_cache(&values);
        assert!(
            timings.warm_shrink,
            "the parent's count was not advanced by the refused candidate"
        );
        assert!(parent.memory_breakdown().warm_accounts_bytes < held / 100);
    }

    #[test]
    fn a_snapshot_continues_its_parent_interval_rather_than_restarting_it() {
        let mut values = fast_forgetting_value_cache();
        let mut trie = PartialTrieNodeCache::new();
        trie.set_warm_shrink_policy(WarmSetShrinkPolicy::EveryBlocks(
            NonZeroU64::new(3).expect("3 is not zero"),
        ));

        advance(&mut trie, &mut values, 1, 0..64);
        advance(&mut trie, &mut values, 2, 64..128);

        // A commit replaces the live cache with a snapshot of it every block. If the interval
        // restarted there, a policy with an interval longer than one block would never close one.
        let mut snapshot = trie.clone();
        assert_eq!(snapshot.warm_shrink_policy(), trie.warm_shrink_policy());
        assert!(advance(&mut snapshot, &mut values, 3, 128..192).warm_shrink);
    }

    #[test]
    fn setting_a_policy_starts_its_own_interval() {
        let mut values = fast_forgetting_value_cache();
        let mut trie = PartialTrieNodeCache::new();
        trie.set_warm_shrink_policy(WarmSetShrinkPolicy::EveryBlocks(
            NonZeroU64::new(2).expect("2 is not zero"),
        ));
        advance(&mut trie, &mut values, 1, 0..64);

        // Re-setting mid-run drops the block already counted, so the new interval is measured
        // whole. The alternative — inheriting a partial count and a mark taken under the old
        // setting — would make the first interval after a change describe neither policy.
        trie.set_warm_shrink_policy(WarmSetShrinkPolicy::EveryBlocks(
            NonZeroU64::new(2).expect("2 is not zero"),
        ));
        assert!(!advance(&mut trie, &mut values, 2, 64..128).warm_shrink);
        assert!(advance(&mut trie, &mut values, 3, 128..192).warm_shrink);
    }

    /// Drives one block the way a commit does — onto a working copy, which then replaces the
    /// live cache — and hands back the frame that turns the new live cache into the generation it
    /// displaced, together with that generation.
    ///
    /// The frame is taken from the *committed* copy against the cache it was cloned from, which is
    /// the same call the pair makes one commit later against the generation behind it.
    fn commit_recorded(
        live: &mut PartialTrieNodeCache,
        values: &mut NetworkStateCache,
        block: u64,
        accessed: &BlockAccessedState,
    ) -> (Option<TrieCacheUndoFrame>, PartialTrieNodeCache) {
        let (mut next, _) = live.clone_timed();
        values.on_block_executed(block, accessed);
        next.retain_from_value_cache(values);
        let mut displaced = std::mem::replace(live, next);
        let frame = live.take_undo_frame(&mut displaced);
        (frame, displaced)
    }

    /// What a frame has to put back, as one value.
    fn committed_state(cache: &PartialTrieNodeCache) -> (B256, B256, Option<B256>, Option<u64>) {
        (
            cache.cache_root(),
            cache.retention_fingerprint(),
            cache.state_root(),
            cache.synced_to_block,
        )
    }

    #[test]
    fn a_frame_puts_the_cache_back_where_the_block_found_it() {
        let mut values = fast_forgetting_value_cache();
        let mut live = PartialTrieNodeCache::new();
        live.set_undo_recording(true);
        // Warmed first, so the measured block moves membership that already exists rather than
        // creating all of it: the delta path is the one the record has a shape for.
        values.on_block_executed(1, &block_touching(0..200));
        live.retain_from_value_cache(&values);
        values.on_block_executed(2, &block_touching_slots(0..40));
        live.retain_from_value_cache(&values);

        let control = live.clone();
        let expected = committed_state(&control);

        // A block that adds accounts, drops others by aging them out of the two-block window, and
        // moves one address's retained slot set.
        let mut accessed = block_touching(180..320);
        for (key, value) in block_touching_slots(20..60).storage {
            accessed.storage.insert(key, value);
        }
        let (frame, displaced) = commit_recorded(&mut live, &mut values, 3, &accessed);
        let frame = frame.expect("the working copy recorded the block");
        assert_eq!(frame.source(), live.undo_id());
        assert_eq!(frame.target(), displaced.undo_id());
        assert_ne!(committed_state(&live), expected, "the block has to change something");

        let counts = frame.counts();
        assert!(!counts.membership_whole, "the incremental path leaves a delta, not a preimage");
        assert!(counts.warm_accounts > 0, "the block moved warm accounts");
        assert!(counts.storage_paths > 0, "the block moved one address's retained slot set");

        assert!(live.undo(frame));
        assert_eq!(committed_state(&live), expected);
        assert_eq!(live.undo_id(), displaced.undo_id(), "the cache is that generation now");
        assert!(live.structurally_eq(&control));
    }

    #[test]
    fn a_retention_rebuild_is_recorded_as_one_whole_preimage() {
        // The delta path is only taken when the value cache is exactly one block ahead of what the
        // derived sets describe. Everything else rebuilds, and a rebuild replaces all four
        // structures at once with no delta to reverse.
        let mut values = fast_forgetting_value_cache();
        let mut live = PartialTrieNodeCache::new();
        live.set_undo_recording(true);
        values.on_block_executed(1, &block_touching(0..200));
        live.retain_from_value_cache(&values);

        let control = live.clone();
        let expected = committed_state(&control);

        let (mut next, _) = live.clone_timed();
        // Two blocks against one retention pass, which is the gap that forces the rebuild.
        values.on_block_executed(2, &block_touching(200..260));
        values.on_block_executed(3, &block_touching(260..320));
        assert!(next.retain_from_value_cache(&values).full_rebuild);
        let mut displaced = std::mem::replace(&mut live, next);
        let frame = live.take_undo_frame(&mut displaced).expect("a rebuild is still recorded");

        assert!(frame.counts().membership_whole);
        assert!(live.undo(frame));
        assert_eq!(committed_state(&live), expected);
        assert!(live.structurally_eq(&control));
    }

    #[test]
    fn a_rebuild_after_a_delta_pass_is_recorded_as_both() {
        // Two retention passes in one block, the second a rebuild. The whole preimage restores
        // what that rebuild found — the state the *first* pass left — and only the delta recorded
        // before it walks the rest of the way back. A record that kept just one of the two would
        // land a block short or a block long, and both look like a working undo until compared.
        let mut values = fast_forgetting_value_cache();
        let mut live = PartialTrieNodeCache::new();
        live.set_undo_recording(true);
        values.on_block_executed(1, &block_touching(0..200));
        live.retain_from_value_cache(&values);

        let control = live.clone();
        let expected = committed_state(&control);

        let (mut next, _) = live.clone_timed();
        values.on_block_executed(2, &block_touching(200..260));
        assert!(!next.retain_from_value_cache(&values).full_rebuild, "one block ahead: the delta");
        // A gap now, which is what sends the second pass down the rebuild path.
        values.on_block_executed(3, &block_touching(260..320));
        values.on_block_executed(4, &block_touching(320..380));
        assert!(next.retain_from_value_cache(&values).full_rebuild, "two ahead: the rebuild");

        let mut displaced = std::mem::replace(&mut live, next);
        let frame = live.take_undo_frame(&mut displaced).expect("both passes are recorded");
        let counts = frame.counts();
        assert!(counts.membership_whole, "the rebuild left a whole preimage");
        assert!(counts.warm_accounts > 0, "and the pass before it left a delta");

        assert!(live.undo(frame));
        assert_eq!(committed_state(&live), expected);
        assert!(live.structurally_eq(&control));
    }

    #[test]
    fn a_frame_is_not_produced_for_a_generation_the_record_does_not_describe() {
        let mut values = fast_forgetting_value_cache();
        let mut live = PartialTrieNodeCache::new();
        live.set_undo_recording(true);
        values.on_block_executed(1, &block_touching(0..50));
        live.retain_from_value_cache(&values);

        let (mut next, _) = live.clone_timed();
        values.on_block_executed(2, &block_touching(50..100));
        next.retain_from_value_cache(&values);

        // A sibling of the same parent, which is a different generation and not the one the
        // record describes. Nothing about the two is distinguishable by height or content.
        let mut stranger = live.clone();
        assert!(
            next.take_undo_frame(&mut stranger).is_none(),
            "a record names the generation it restores, and this is not it"
        );
        assert!(!next.is_recording_undo(), "the record is ended either way");
    }

    /// A cache whose account trie is revealed over `harness` and holds real leaves.
    ///
    /// The fixtures above run on a blind account trie, which is what a cold cache has and what the
    /// replay fixture's one-account snapshot amounts to. Nothing there moves account-trie
    /// *content* across an undo, and the plumbing that carries the trie's own record through the
    /// frame is exactly where a defect hides — so this builds a trie a block can actually change.
    fn revealed_cache(harness: &TrieTestHarness, revealed: &[B256]) -> PartialTrieNodeCache {
        let mut trie = ExactSparseTrie::default();
        let root = harness.root_node();
        trie.set_root(root.node, root.masks, false).expect("the harness root reveals");
        let mut targets: Vec<_> = revealed.iter().map(|key| ProofV2Target::new(*key)).collect();
        let (mut nodes, _) = harness.proof_v2(&mut targets);
        trie.reveal_nodes(&mut nodes).expect("the harness proof reveals");
        let state_root = trie.root();

        let mut cache = PartialTrieNodeCache::new();
        *cache.sparse_mut().trie_mut() =
            RevealableSparseTrie::Revealed(Box::new(CacheTrie::Exact(trie)));
        cache.set_state_root(state_root);
        cache
    }

    /// Applies leaf changes the way a block does, revealing whatever the update asks for.
    fn apply_leaves(
        harness: &TrieTestHarness,
        cache: &mut PartialTrieNodeCache,
        changes: &[(B256, U256)],
    ) -> B256 {
        let mut updates: B256Map<LeafUpdate> = changes
            .iter()
            .map(|(key, value)| {
                let rlp =
                    if value.is_zero() { Vec::new() } else { encode_fixed_size(value).to_vec() };
                (*key, LeafUpdate::Changed(rlp))
            })
            .collect();
        let trie = cache
            .sparse_mut()
            .trie_mut()
            .as_revealed_mut()
            .expect("the fixture's account trie is revealed");
        loop {
            let mut targets = Vec::new();
            trie.update_leaves(&mut updates, |key, min_len| {
                targets.push(ProofV2Target::new(key).with_min_len(min_len));
            })
            .expect("the update applies");
            if targets.is_empty() {
                break
            }
            let (mut nodes, _) = harness.proof_v2(&mut targets);
            trie.reveal_nodes(&mut nodes).expect("the harness answers what the update asked for");
        }
        trie.root()
    }

    #[test]
    fn a_frame_carries_the_account_tries_own_changes_back() {
        let entries: BTreeMap<B256, U256> = (0..64usize)
            .map(|i| (keccak256(B256::from(U256::from(i))), U256::from(i + 1)))
            .collect();
        let harness = TrieTestHarness::new(entries.clone());
        let keys: Vec<B256> = entries.keys().copied().collect();
        let mut live = revealed_cache(&harness, &keys);
        live.set_undo_recording(true);

        let control = live.clone();
        let before_root = live.state_root().expect("the fixture is authenticated");

        // Overwrites, a delete, and an insert of a key the trie has never held — the three shapes
        // that move a Patricia trie's structure rather than only its values.
        let (mut next, _) = live.clone_timed();
        let changes = vec![
            (keys[3], U256::from(999u64)),
            (keys[17], U256::ZERO),
            (keccak256(B256::from(U256::from(4_242usize))), U256::from(7u64)),
        ];
        let after_root = apply_leaves(&harness, &mut next, &changes);
        next.set_state_root(after_root);
        assert_ne!(
            after_root, before_root,
            "the block has to move the root or this proves nothing"
        );

        let mut displaced = std::mem::replace(&mut live, next);
        let frame = live.take_undo_frame(&mut displaced).expect("the working copy recorded it");
        let counts = frame.counts();
        assert!(counts.account_nodes > 0, "the frame carries node preimages");
        assert!(counts.account_values > 0, "and value preimages");

        assert!(live.undo(frame));
        assert_eq!(live.state_root(), Some(before_root));
        assert!(
            live.structurally_eq(&control),
            "every revealed node and value is back where the block found it"
        );
        assert_eq!(
            live.sparse_mut().trie_mut().as_revealed_mut().expect("revealed").root(),
            before_root,
            "and the trie recomputes the root it had, rather than only remembering it"
        );
    }

    #[test]
    fn an_account_trie_revealed_mid_block_produces_no_frame() {
        // A blind slot has nothing to preimage and a trie still blind at the commit changed
        // nothing, so both are recorded rather than poisoned. A slot that was blind and is
        // revealed now is neither: a whole trie came into existence and no preimage describes it.
        // Its `take_undo` returns `None` exactly as a still-blind slot's does — nothing ever told
        // it to record — so the two are only distinguishable by asking the slot itself, and a
        // frame produced here would restore the parent's identity over the child's content.
        let mut values = fast_forgetting_value_cache();
        let mut live = PartialTrieNodeCache::new();
        live.set_undo_recording(true);
        values.on_block_executed(1, &block_touching(0..50));
        live.retain_from_value_cache(&values);
        assert!(live.sparse_ref().state_trie_ref().is_none(), "the fixture starts blind");

        let (mut next, _) = live.clone_timed();
        values.on_block_executed(2, &block_touching(50..100));
        next.retain_from_value_cache(&values);
        *next.sparse_mut().trie_mut() = RevealableSparseTrie::revealed_empty();

        let mut displaced = std::mem::replace(&mut live, next);
        assert!(
            live.take_undo_frame(&mut displaced).is_none(),
            "a reveal is not a change any preimage in the record describes"
        );
        assert!(!live.is_recording_undo(), "the record is ended either way");
    }

    #[test]
    fn a_cache_that_does_not_record_produces_no_frame() {
        let mut values = fast_forgetting_value_cache();
        let mut live = PartialTrieNodeCache::new();
        assert!(!live.records_undo(), "recording is off until a run asks for it");
        values.on_block_executed(1, &block_touching(0..50));
        live.retain_from_value_cache(&values);

        let (frame, _) = commit_recorded(&mut live, &mut values, 2, &block_touching(50..100));
        assert!(frame.is_none(), "no record, no frame — the holder keeps the generation whole");
    }

    #[test]
    fn recording_is_inherited_by_a_working_copy_and_is_not_the_parents_record() {
        let mut live = PartialTrieNodeCache::new();
        live.set_undo_recording(true);
        assert!(!live.is_recording_undo(), "a cache nothing was cloned from records nothing");

        let (child, _) = live.clone_timed();
        assert!(child.records_undo());
        assert!(child.is_recording_undo());
        assert_ne!(child.undo_id(), live.undo_id(), "a clone is a different generation");

        let (grandchild, _) = child.clone_timed();
        assert_ne!(grandchild.undo_id(), child.undo_id());
    }
}
