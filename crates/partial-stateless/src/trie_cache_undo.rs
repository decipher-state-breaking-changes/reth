//! What one block did to a [`PartialTrieNodeCache`](crate::trie_cache::PartialTrieNodeCache),
//! kept so it can be reversed.
//!
//! A retained generation is a whole copy of the cache — 186 MiB of it on the measured corpus —
//! and the depth a reorg can be undone from is how many of those the process is willing to hold.
//! This is the diff that replaces the copy: the account trie's own first-write record, the
//! storage tries the block replaced, the keys warm membership and the retained-path indexes
//! moved, and the three scalars, sized by what the block touched rather than by what the cache
//! holds.
//!
//! **Direction.** A frame turns the generation it was recorded *on* back into the generation that
//! generation was cloned *from* — newer into older, never the other way. [`TrieCacheUndoFrame`]
//! carries both ends' identities so a holder can check the chain links before it applies
//! anything; nothing here can tell that a caller has applied a frame to the wrong cache, and the
//! sparse trie's `undo` is a replay of preimages rather than a merge, so a mismatch is silent
//! corruption rather than an error.
//!
//! **First write wins.** Every recorder here keeps the preimage of a key's *first* change in the
//! block and ignores the ones after it, which is what makes the record exact when a block touches
//! the same key twice — the discipline `JournaledMap` applies inside the trie, applied again to
//! the four structures beside it. A bulk rebuild that has no delta shape is recorded as one whole
//! preimage that supersedes everything recorded before it.

use crate::{
    cache_trie::CacheTrie,
    shared_trie::SharedSparseTrie,
    trie_cache::{hashbrown_table_bytes, WarmShrink},
};
use alloy_primitives::{
    map::{B256Map, HashMap, HashSet},
    Address, B256,
};
use reth_trie_common::Nibbles;
use reth_trie_sparse::{RevealableSparseTrie, SparseTrie, UndoFrame};
use serde::Serialize;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

/// One storage trie as the cache's map holds it.
pub(crate) type CacheStorageTrie = RevealableSparseTrie<SharedSparseTrie<CacheTrie>>;

/// Hands out the identity a trie cache generation is known by.
///
/// Process-wide and monotonic, so an identity is never reused and a frame that names a generation
/// names one generation. Not an address: an address is only unique while the object is alive, and
/// a frame outlives the generation it restores.
static NEXT_UNDO_ID: AtomicU64 = AtomicU64::new(1);

/// The next unused generation identity.
pub(crate) fn next_undo_id() -> u64 {
    NEXT_UNDO_ID.fetch_add(1, Ordering::Relaxed)
}

/// Everything needed to turn one trie cache generation back into the one it was cloned from.
///
/// Sized by the block, not by the cache. The storage tries it holds are `Arc` handles moved out
/// of the generation it replaces rather than copies of anything, so a frame's real weight is the
/// account-trie preimages plus whatever share of the old storage tries nothing else still points
/// at — which is why [`Self::shared_allocations`] exists beside [`Self::allocated_bytes`].
#[derive(Debug)]
pub struct TrieCacheUndoFrame {
    /// The generation this frame applies *to*, and the one it produces.
    ///
    /// Identities rather than block numbers: a holder chains frames together and has to be able
    /// to prove the chain links before it mutates anything, and mid-reorg a height names
    /// whichever block the database currently calls canonical.
    pub(crate) source: u64,
    pub(crate) target: u64,
    /// The account trie's own record.
    ///
    /// `None` only for an account trie that was blind at both ends of the block — a cache that has
    /// never been revealed, which is the fixture case and not the consumer's. A slot that was
    /// blind and became revealed is a change no preimage here describes, and poisons the record
    /// instead of landing here.
    pub(crate) account: Option<UndoFrame>,
    /// Storage-trie map entries the block replaced, dropped or added.
    pub(crate) storage: Vec<(B256, StorageTrieBefore)>,
    /// Warm membership and the retained-path indexes, reversed.
    pub(crate) membership: MembershipUndo,
    /// The three scalars as they stood before the block.
    pub(crate) state_root: Option<B256>,
    pub(crate) synced_to_block: Option<u64>,
    pub(crate) warm_shrink: WarmShrink,
}

impl TrieCacheUndoFrame {
    /// The generation this frame must be applied to.
    pub const fn source(&self) -> u64 {
        self.source
    }

    /// The generation applying this frame produces.
    pub const fn target(&self) -> u64 {
        self.target
    }

    /// Whether the frame carries an account-trie record, which only a cache whose account trie
    /// was blind for the whole block does not.
    pub const fn records_account_trie(&self) -> bool {
        self.account.is_some()
    }

    /// What the frame holds, by kind. Logical counts; bytes are [`Self::allocated_bytes`].
    pub fn counts(&self) -> TrieCacheUndoCounts {
        let account = self.account.as_ref().map(UndoFrame::counts).unwrap_or_default();
        let mut counts = TrieCacheUndoCounts {
            account_nodes: account.nodes,
            account_values: account.values,
            account_masks: account.masks,
            account_lower_transitions: account.lower_transitions,
            account_whole_maps: account.whole_maps,
            account_metadata_changed: account.metadata_changed,
            ..Default::default()
        };
        for (_, before) in &self.storage {
            match before {
                StorageTrieBefore::Held(trie) => {
                    counts.storage_tries_held += 1;
                    match &**trie {
                        CacheStorageTrie::Revealed(_) => counts.storage_tries_revealed += 1,
                        CacheStorageTrie::Blind(Some(_)) => {
                            counts.storage_tries_blind_retained += 1
                        }
                        CacheStorageTrie::Blind(None) => counts.storage_tries_blind_empty += 1,
                    }
                }
                StorageTrieBefore::Absent => counts.storage_tries_absent += 1,
            }
        }
        let delta = &self.membership.delta;
        counts.warm_accounts = delta.warm_accounts.len();
        counts.warm_storage = delta.warm_storage.len();
        counts.account_paths = delta.account_paths.len();
        counts.storage_paths = delta.storage_paths.len();
        if let Some(whole) = &self.membership.whole {
            counts.membership_whole = true;
            counts.warm_accounts += whole.warm_accounts.len();
            counts.warm_storage += whole.warm_storage.len();
            counts.account_paths += whole.retained_account_paths.len();
            counts.storage_paths += whole.retained_storage_paths.len();
        }
        counts
    }

    /// Heap bytes the frame holds, as an estimate, storage tries included.
    ///
    /// [`Self::record_bytes`] plus [`Self::storage_bytes`]. Charged the way
    /// [`crate::trie_cache::TrieCacheMemory`] charges the cache: every hash table its buckets by
    /// hashbrown's sizing rule, every `Vec` its capacity. Not a reading of the allocator, and it
    /// counts an `Arc`'d storage trie in full — several frames holding the same old trie each
    /// count it, which is why a K-frame total unions [`Self::shared_allocations`] instead of
    /// adding these.
    pub fn allocated_bytes(&self) -> usize {
        self.record_bytes() + self.storage_bytes()
    }

    /// What this block's record itself weighs: everything the block *created*.
    ///
    /// The account-trie preimages, the membership preimages, and the frame's own containers —
    /// bounded by what the block touched, and the half of the estimate that belongs on a per-block
    /// record. Costs one pass over the recorded entries and nothing else, which is why it is
    /// computed on every commit while [`Self::storage_bytes`] is not.
    pub fn record_bytes(&self) -> usize {
        let nibble = std::mem::size_of::<Nibbles>();
        let mut bytes = std::mem::size_of::<Self>();
        bytes += self.account.as_ref().map_or(0, UndoFrame::allocated_bytes);
        bytes += self.storage.capacity() * std::mem::size_of::<(B256, StorageTrieBefore)>();
        let delta = &self.membership.delta;
        bytes += hashbrown_table_bytes(
            delta.warm_accounts.capacity(),
            std::mem::size_of::<Address>() + 1,
        ) + hashbrown_table_bytes(
            delta.warm_storage.capacity(),
            std::mem::size_of::<(Address, B256)>() + 1,
        ) + hashbrown_table_bytes(delta.account_paths.capacity(), nibble + 1) +
            hashbrown_table_bytes(
                delta.storage_paths.capacity(),
                std::mem::size_of::<B256>() + std::mem::size_of::<Option<Arc<[Nibbles]>>>(),
            ) +
            delta
                .storage_paths
                .values()
                .flatten()
                .map(|paths| paths.len() * nibble)
                .sum::<usize>();
        if let Some(whole) = &self.membership.whole {
            bytes += hashbrown_table_bytes(
                whole.warm_accounts.capacity(),
                std::mem::size_of::<Address>(),
            ) + hashbrown_table_bytes(
                whole.warm_storage.capacity(),
                std::mem::size_of::<(Address, B256)>(),
            ) + whole.retained_account_paths.capacity() * nibble +
                hashbrown_table_bytes(
                    whole.retained_storage_paths.capacity(),
                    std::mem::size_of::<B256>() + std::mem::size_of::<Arc<[Nibbles]>>(),
                ) +
                whole
                    .retained_storage_paths
                    .values()
                    .map(|paths| paths.len() * nibble)
                    .sum::<usize>();
        }
        bytes
    }

    /// What the frame keeps *alive*: the previous-version storage tries it holds by `Arc`.
    ///
    /// Not bytes the block created — those allocations already existed and the frame only stops
    /// them being freed — which is why they are reported at the deque level rather than per block.
    /// Costs a walk of every held trie, so it is memory-probe work: `SparseTrie::memory_size` is
    /// linear in a trie's nodes and a block holds ~100 of them.
    pub fn storage_bytes(&self) -> usize {
        self.storage
            .iter()
            .filter_map(|(_, before)| before.retained())
            .map(SparseTrie::memory_size)
            .sum()
    }

    /// Of [`Self::allocated_bytes`], the part no generation or other frame can be holding.
    ///
    /// The complement of [`Self::shared_allocations`] within the same total, so the two partition
    /// the frame exactly as `PartialTrieNodeCache`'s pair of the same names partition a whole
    /// generation — which is what lets a holder union frames and generations in one pass.
    pub fn unshared_bytes(&self) -> usize {
        let shared: usize = self.shared_allocations().iter().map(|(_, bytes)| bytes).sum();
        self.allocated_bytes().saturating_sub(shared)
    }

    /// Every allocation this frame may be sharing with a generation or another frame.
    ///
    /// The storage tries it holds, by `Arc` identity and size — the same `(identity, bytes)`
    /// shape [`crate::trie_cache::PartialTrieNodeCache::shared_allocations`] reports, so a holder
    /// unions frames and generations in one pass. An old storage trie is very often held by the
    /// frame *and* by an older frame that displaced it, and counting it twice would charge the
    /// deque for memory dropping it would not return.
    pub fn shared_allocations(&self) -> Vec<(usize, usize)> {
        self.storage
            .iter()
            .filter_map(|(_, before)| before.shared())
            .map(|trie| (trie.allocation_id(), trie.memory_size()))
            .collect()
    }
}

/// What a [`TrieCacheUndoFrame`] holds, by kind.
///
/// Logical counts. The `account_*` fields are the sparse trie's own report flattened — a run log
/// wants one flat record — and everything beside them is one entry per key the block moved in the
/// four structures the trie does not cover.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct TrieCacheUndoCounts {
    /// Account-trie node preimages recorded.
    pub account_nodes: usize,
    /// Account-trie leaf value preimages recorded.
    pub account_values: usize,
    /// Account-trie branch mask preimages recorded.
    pub account_masks: usize,
    /// Lower subtries of the account trie whose reveal state changed.
    pub account_lower_transitions: usize,
    /// Account-trie maps captured whole by a bulk clear.
    pub account_whole_maps: usize,
    /// Whether the account trie's prefix set or retained updates changed with no map entry
    /// behind it.
    pub account_metadata_changed: bool,
    /// Storage tries held for an address the block wrote to.
    ///
    /// The sum of the three fields below, kept as one number because it is what a reorg-cost
    /// argument counts: one entry the frame puts back into the map. The split beside it is what a
    /// *memory* argument counts, and the two are not the same population.
    pub storage_tries_held: usize,
    /// Of those, the revealed ones: real trie nodes, and the only ones a frame reports as
    /// shareable allocations.
    ///
    /// [`TrieCacheUndoFrame::shared_allocations`] lists exactly these, so a deque union charges
    /// each once however many frames hold it. Everything below lands in the frame's unshared half
    /// instead.
    pub storage_tries_revealed: usize,
    /// Of those, blind slots that kept their allocation.
    ///
    /// Real bytes with no identity to union on: `storage_bytes` charges them,
    /// `shared_allocations` cannot list them, so every frame holding one is charged separately.
    /// The distance between this and zero is the size of that upper bound.
    pub storage_tries_blind_retained: usize,
    /// Of those, blind slots holding nothing: an entry to restore, and no bytes behind it.
    pub storage_tries_blind_empty: usize,
    /// Addresses the block added to the storage-trie map, removed again on undo.
    pub storage_tries_absent: usize,
    /// Warm account keys whose membership the block moved.
    pub warm_accounts: usize,
    /// Warm storage keys whose membership the block moved.
    pub warm_storage: usize,
    /// Retained account paths the block spliced in or out.
    pub account_paths: usize,
    /// Addresses whose retained slot-path set the block replaced.
    pub storage_paths: usize,
    /// Whether membership was recorded as a whole preimage rather than as a delta, which a
    /// retention pass that rebuilt from the value cache rather than patching forces.
    pub membership_whole: bool,
}

/// One storage-trie map entry as the older generation held it.
#[derive(Debug)]
pub(crate) enum StorageTrieBefore {
    /// The map held this trie for the address; put it back.
    ///
    /// Moved out of the generation being replaced rather than cloned: that generation is dropped
    /// in the same breath, so the handle it held is the frame's to take.
    Held(Box<CacheStorageTrie>),
    /// The map had no entry for the address; remove it.
    Absent,
}

impl StorageTrieBefore {
    /// The trie behind a held handle, revealed or blind-with-allocation.
    ///
    /// A blind slot that kept its allocation holds real bytes: `SparseStateTrie::memory_size`
    /// counts it, so anything charging a frame for what it retains has to as well. Reading only
    /// the revealed ones would let a frame report a storage trie held and no bytes to go with it.
    fn retained(&self) -> Option<&SharedSparseTrie<CacheTrie>> {
        match self {
            Self::Held(trie) => match &**trie {
                CacheStorageTrie::Revealed(inner) | CacheStorageTrie::Blind(Some(inner)) => {
                    Some(inner)
                }
                CacheStorageTrie::Blind(None) => None,
            },
            Self::Absent => None,
        }
    }

    /// The revealed trie behind a held handle, for identity.
    ///
    /// Revealed only, so a frame's shared half is the same population
    /// `PartialTrieNodeCache::shared_allocations` reports and the two union without an asymmetry —
    /// a blind allocation the live cache also holds would otherwise be charged to the deque
    /// because the live side never listed it. It lands in the frame's unshared half instead, which
    /// counts it once per frame holding it: an upper bound, in the same direction and for the same
    /// reason as `exclusive_memory_bytes`'s.
    fn shared(&self) -> Option<&SharedSparseTrie<CacheTrie>> {
        match self {
            Self::Held(trie) => trie.as_revealed_ref(),
            Self::Absent => None,
        }
    }
}

/// Warm membership and the retained-path indexes, reversed.
///
/// Two layers, applied in this order, because a block can run more than one retention pass and
/// the second can be a rebuild: the whole preimage puts the four structures back to what the
/// *first* rebuild found, and the delta recorded before that rebuild walks them the rest of the
/// way back to where the block started. With no rebuild there is no whole and the delta is the
/// whole record; with a rebuild first there is no delta to apply.
#[derive(Debug, Default)]
pub(crate) struct MembershipUndo {
    /// The four structures as the block's first retention rebuild found them.
    pub(crate) whole: Option<Box<MembershipWhole>>,
    /// The keys moved before that rebuild — everything, when there was none — each with what it
    /// held the first time this block touched it.
    pub(crate) delta: MembershipUndoDelta,
}

/// The preimage of every membership key one block moved.
///
/// Keyed maps rather than the added/removed vectors the forward delta is shaped as, because a
/// block can move the same key twice — an account that leaves warm membership and comes back
/// within one retention pass — and only the first preimage describes the generation being
/// restored.
#[derive(Debug, Default)]
pub(crate) struct MembershipUndoDelta {
    /// Whether each touched warm account was present before the block.
    pub(crate) warm_accounts: HashMap<Address, bool>,
    /// Whether each touched warm storage key was present before the block.
    pub(crate) warm_storage: HashMap<(Address, B256), bool>,
    /// Whether each touched retained account path was present before the block.
    pub(crate) account_paths: HashMap<Nibbles, bool>,
    /// What `retained_storage_paths` held for each address the block moved.
    pub(crate) storage_paths: B256Map<Option<Arc<[Nibbles]>>>,
}

/// Warm membership and the retained-path indexes copied whole.
#[derive(Debug)]
pub(crate) struct MembershipWhole {
    pub(crate) warm_accounts: HashSet<Address>,
    pub(crate) warm_storage: HashSet<(Address, B256)>,
    pub(crate) retained_account_paths: Vec<Nibbles>,
    pub(crate) retained_storage_paths: B256Map<Arc<[Nibbles]>>,
}

/// The record a cache keeps while a block is applied to it.
///
/// Created at the clone that starts the block, filled in as retention runs, and turned into a
/// [`TrieCacheUndoFrame`] at the commit that displaces the generation it describes. Poisoned
/// rather than made wrong by anything it cannot describe — a trie representation with no record,
/// a blind account trie — and a poisoned record produces no frame, which leaves its holder
/// keeping the whole generation instead.
#[derive(Debug)]
pub(crate) struct CacheUndoRecord {
    /// The identity of the generation this record restores: the cache it was cloned from.
    pub(crate) parent: u64,
    pub(crate) state_root: Option<B256>,
    pub(crate) synced_to_block: Option<u64>,
    pub(crate) warm_shrink: WarmShrink,
    pub(crate) membership: MembershipUndo,
    /// Whether the account trie was blind when recording began.
    ///
    /// A blind slot holds no content to preimage, so a trie blind at both ends of the block
    /// records nothing and is not a hole in the frame. One that was revealed in between is, and
    /// poisons the record.
    pub(crate) account_blind: bool,
    /// Set when something happened the record cannot describe. Never unset.
    pub(crate) poisoned: bool,
}

impl CacheUndoRecord {
    /// Starts a record of the step from `parent`, with the scalars as they stand.
    pub(crate) fn new(
        parent: u64,
        state_root: Option<B256>,
        synced_to_block: Option<u64>,
        warm_shrink: WarmShrink,
    ) -> Self {
        Self {
            parent,
            state_root,
            synced_to_block,
            warm_shrink,
            membership: MembershipUndo::default(),
            account_blind: false,
            poisoned: false,
        }
    }

    /// The delta being filled in, or `None` once a whole preimage has been taken.
    ///
    /// A whole preimage restores everything a later key-level record could describe, so recording
    /// after one is work with no reader — the same rule `MapJournal` applies to a map a bulk clear
    /// has captured. What was recorded *before* it is not superseded: the whole restores the state
    /// that rebuild found, and the earlier delta walks it back to where the block started.
    pub(crate) const fn delta_mut(&mut self) -> Option<&mut MembershipUndoDelta> {
        match self.membership.whole {
            None => Some(&mut self.membership.delta),
            Some(_) => None,
        }
    }

    /// Takes the whole membership state as one preimage. Only the first in a block is kept.
    pub(crate) fn record_whole(&mut self, whole: MembershipWhole) {
        if self.membership.whole.is_none() {
            self.membership.whole = Some(Box::new(whole));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reth_trie_sparse::SparseTrie;

    /// A frame holding one storage trie in the given slot state, and nothing else.
    fn frame_holding(held: CacheStorageTrie) -> TrieCacheUndoFrame {
        TrieCacheUndoFrame {
            source: 2,
            target: 1,
            account: None,
            storage: vec![(B256::ZERO, StorageTrieBefore::Held(Box::new(held)))],
            membership: MembershipUndo::default(),
            state_root: None,
            synced_to_block: None,
            warm_shrink: WarmShrink::default(),
        }
    }

    #[test]
    fn a_blind_slot_that_kept_its_allocation_is_charged_for_it() {
        // `SparseStateTrie::memory_size` counts a blind slot that kept its trie, so a frame that
        // reports the trie held and no bytes to go with it is under-reporting real retention —
        // and `storage_tries_held` would say one while `allocated_bytes` said none.
        let trie = SharedSparseTrie::new(CacheTrie::default());
        let bytes = trie.memory_size();
        assert!(bytes > 0, "even an empty trie holds something");

        let blind = frame_holding(CacheStorageTrie::Blind(Some(Box::new(trie))));
        assert_eq!(blind.counts().storage_tries_held, 1);
        assert_eq!(blind.storage_bytes(), bytes, "the kept allocation is in the estimate");
        assert_eq!(blind.allocated_bytes(), blind.record_bytes() + bytes);
        assert!(
            blind.record_bytes() > 0 && blind.record_bytes() < blind.allocated_bytes(),
            "a trie the frame keeps alive is not part of what the block recorded"
        );

        // Identity is revealed-only, so the frame's shared half stays the same population
        // `PartialTrieNodeCache::shared_allocations` reports and the two union without an
        // asymmetry. The blind allocation lands in the unshared half instead.
        assert!(blind.shared_allocations().is_empty());
        assert!(blind.unshared_bytes() >= bytes);

        // A slot with nothing behind it is charged nothing, and still counts as held: the frame
        // has to put the absence back.
        let empty = frame_holding(CacheStorageTrie::Blind(None));
        assert_eq!(empty.counts().storage_tries_held, 1);
        assert_eq!(empty.storage_bytes(), 0);
        assert!(empty.allocated_bytes() < blind.allocated_bytes());
    }

    #[test]
    fn a_revealed_slot_is_shared_by_identity_and_charged_once() {
        let trie = SharedSparseTrie::new(CacheTrie::default());
        let id = trie.allocation_id();
        let revealed = frame_holding(CacheStorageTrie::Revealed(Box::new(trie)));

        let shared = revealed.shared_allocations();
        assert_eq!(shared.len(), 1);
        assert_eq!(shared[0].0, id);
        assert_eq!(revealed.allocated_bytes(), revealed.unshared_bytes() + shared[0].1);
    }
}
