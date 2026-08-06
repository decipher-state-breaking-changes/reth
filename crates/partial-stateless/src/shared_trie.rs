//! Copy-on-write storage tries for the transactional trie cache.
//!
//! The trie cache is transactional: a block is applied to a snapshot of the parent cache and the
//! snapshot replaces the parent only after the post-state root and the next cache anchor both
//! check out. A rejected block must leave the parent generation byte-identical.
//!
//! Deep-copying every retained storage trie to get that property costs about as much as the rest
//! of validation put together, and almost all of it is wasted: a block dirties a few hundred
//! storage tries out of the thousands the cache retains. [`SharedSparseTrie`] makes the snapshot
//! share each storage trie with its parent and take a private copy only when something actually
//! writes to it, so an untouched trie is never copied by either side.
//!
//! Sharing is enforced by the type rather than by a predicted write set: the inner trie is
//! reachable only through [`SharedSparseTrie::make_mut`], so every `&mut` path in the
//! [`SparseTrie`] surface copies before it writes. A missed write would corrupt the parent
//! generation, which is exactly the failure a predicted write set invites.

use alloy_primitives::{
    map::{B256Map, HashMap, HashSet},
    B256,
};
use reth_trie_common::{BranchNodeCompact, BranchNodeMasks, Nibbles, ProofTrieNodeV2, TrieNodeV2};
use reth_trie_sparse::{
    errors::SparseTrieResult, LeafLookup, LeafLookupError, LeafUpdate, ParallelSparseTrie,
    SparseTrie, SparseTrieUpdates,
};
use std::{
    borrow::Cow,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

/// Process-wide count of copy-on-write copies actually taken.
///
/// A diagnostic counter, not a correctness input. It counts copies rather than survivors, which is
/// the distinction that makes it worth keeping over inspecting the committed cache: a trie copied
/// and then dropped by retention still cost a copy, and a trie the transition created rather than
/// copied did not.
static COW_COPIES: AtomicU64 = AtomicU64::new(0);

/// Total copy-on-write copies taken since process start.
///
/// Read as a delta bracketing one transaction. Process-wide rather than per-cache because the
/// copies happen deep inside the trie transition, including on rayon workers; that is exact only
/// while transitions do not overlap, which is what the ExEx's one-notification-at-a-time handler
/// guarantees.
pub fn cow_copies_taken() -> u64 {
    COW_COPIES.load(Ordering::Relaxed)
}

/// A storage trie shared with the generation it was snapshotted from until it is written to.
///
/// Cloning is a refcount bump. The first `&mut` access takes a private copy if the trie is still
/// shared, which is what keeps a committed child and an aborted child equally correct: the parent
/// only ever sees writes through its own handle.
#[derive(Debug)]
pub struct SharedSparseTrie<T = ParallelSparseTrie> {
    inner: Arc<T>,
    ownership: Ownership,
}

/// How this handle came to own, or not own, its trie.
///
/// `Copied` is kept apart from `Private` because the two cost very different things and the
/// benchmark counter has to tell them apart: a trie the transition created was never copied out of
/// anything, and counting it would overstate what the block paid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ownership {
    /// Still shared with the generation this handle was cloned from.
    Shared,
    /// Privately owned, having copied the trie out from under a sharing parent.
    Copied,
    /// Privately owned without a copy: created here, or already unique when first written.
    Private,
}

impl<T> SharedSparseTrie<T> {
    /// Wraps an owned trie.
    pub fn new(trie: T) -> Self {
        Self { inner: Arc::new(trie), ownership: Ownership::Private }
    }

    /// Whether this handle has not been written to since it was cloned from its parent.
    ///
    /// An untouched handle is byte-identical to the parent's, which is what lets retention skip
    /// re-running a prune that would produce the same shape.
    pub const fn is_untouched(&self) -> bool {
        matches!(self.ownership, Ownership::Shared)
    }

    /// Whether this handle paid for a copy, as opposed to being created or already unique.
    pub const fn took_copy(&self) -> bool {
        matches!(self.ownership, Ownership::Copied)
    }

    /// Read-only access to the shared trie.
    pub fn shared_ref(&self) -> &T {
        &self.inner
    }

    /// Whether dropping this handle would free the trie behind it.
    ///
    /// Distinct from [`Self::is_untouched`], which describes how this handle came to exist. This
    /// describes who holds the allocation *now*: a handle can be untouched and still be the only
    /// one left if every other generation sharing it has been dropped. The K = 1 memory
    /// measurement needs exactly this question — bytes a retained generation would give back —
    /// and neither ownership flag answers it.
    pub fn is_sole_owner(&self) -> bool {
        Arc::strong_count(&self.inner) == 1
    }
}

impl<T: Clone> SharedSparseTrie<T> {
    /// Mutable access, taking a private copy first if the trie is still shared.
    pub fn make_mut(&mut self) -> &mut T {
        if self.ownership == Ownership::Shared {
            self.ownership = if Arc::strong_count(&self.inner) > 1 {
                COW_COPIES.fetch_add(1, Ordering::Relaxed);
                Ownership::Copied
            } else {
                Ownership::Private
            };
        }
        Arc::make_mut(&mut self.inner)
    }
}

impl<T> Clone for SharedSparseTrie<T> {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner), ownership: Ownership::Shared }
    }
}

impl<T: Default> Default for SharedSparseTrie<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: PartialEq> PartialEq for SharedSparseTrie<T> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner) || *self.inner == *other.inner
    }
}

impl<T: Eq> Eq for SharedSparseTrie<T> {}

impl<T: SparseTrie + Clone + Default> SparseTrie for SharedSparseTrie<T> {
    fn set_root(
        &mut self,
        root: TrieNodeV2,
        masks: Option<BranchNodeMasks>,
        retain_updates: bool,
    ) -> SparseTrieResult<()> {
        self.make_mut().set_root(root, masks, retain_updates)
    }

    fn set_updates(&mut self, retain_updates: bool) {
        self.make_mut().set_updates(retain_updates)
    }

    fn reserve_nodes(&mut self, additional: usize) {
        self.make_mut().reserve_nodes(additional)
    }

    fn reveal_nodes(&mut self, nodes: &mut [ProofTrieNodeV2]) -> SparseTrieResult<()> {
        self.make_mut().reveal_nodes(nodes)
    }

    /// Reads the cached root rather than copying to recompute one that would not change.
    ///
    /// Account-trie preparation asks every changed account for its storage root, so a `root` that
    /// always reached for `&mut` would copy exactly the tries a block reads but never writes.
    fn root(&mut self) -> B256 {
        self.inner.cached_root().unwrap_or_else(|| self.make_mut().root())
    }

    fn is_root_cached(&self) -> bool {
        self.inner.is_root_cached()
    }

    fn cached_root(&self) -> Option<B256> {
        self.inner.cached_root()
    }

    fn update_subtrie_hashes(&mut self) {
        self.make_mut().update_subtrie_hashes()
    }

    fn get_leaf_value(&self, full_path: &Nibbles) -> Option<&Vec<u8>> {
        self.inner.get_leaf_value(full_path)
    }

    fn find_leaf(
        &self,
        full_path: &Nibbles,
        expected_value: Option<&Vec<u8>>,
    ) -> Result<LeafLookup, LeafLookupError> {
        self.inner.find_leaf(full_path, expected_value)
    }

    fn updates_ref(&self) -> Cow<'_, SparseTrieUpdates> {
        self.inner.updates_ref()
    }

    fn take_updates(&mut self) -> SparseTrieUpdates {
        self.make_mut().take_updates()
    }

    fn wipe(&mut self) {
        self.make_mut().wipe()
    }

    fn clear(&mut self) {
        self.make_mut().clear()
    }

    fn shrink_nodes_to(&mut self, size: usize) {
        self.make_mut().shrink_nodes_to(size)
    }

    fn shrink_values_to(&mut self, size: usize) {
        self.make_mut().shrink_values_to(size)
    }

    fn size_hint(&self) -> usize {
        self.inner.size_hint()
    }

    fn memory_size(&self) -> usize {
        self.inner.memory_size()
    }

    fn prune(&mut self, retained_leaves: &[Nibbles]) -> usize {
        self.make_mut().prune(retained_leaves)
    }

    fn retain_witness_paths(&mut self, retained_paths: &[Nibbles]) -> usize {
        self.make_mut().retain_witness_paths(retained_paths)
    }

    fn update_leaves(
        &mut self,
        updates: &mut B256Map<LeafUpdate>,
        proof_required_fn: impl FnMut(B256, u8),
    ) -> SparseTrieResult<()> {
        self.make_mut().update_leaves(updates, proof_required_fn)
    }

    fn commit_updates(
        &mut self,
        updated: &HashMap<Nibbles, BranchNodeCompact>,
        removed: &HashSet<Nibbles>,
    ) {
        self.make_mut().commit_updates(updated, removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(prefix: &[u8]) -> B256 {
        let mut bytes = [0u8; 32];
        bytes[..prefix.len()].copy_from_slice(prefix);
        B256::from(bytes)
    }

    fn trie_with_leaves(keys: &[B256]) -> SharedSparseTrie {
        let mut trie = SharedSparseTrie::default();
        let mut updates = B256Map::default();
        for (index, key) in keys.iter().enumerate() {
            updates.insert(*key, LeafUpdate::Changed(vec![index as u8 + 1; 64]));
        }
        trie.update_leaves(&mut updates, |target, min_len| {
            panic!("fresh trie unexpectedly requested proof target {target} at {min_len}")
        })
        .unwrap();
        trie.root();
        trie
    }

    #[test]
    fn a_clone_is_untouched_until_it_is_written_to() {
        let parent = trie_with_leaves(&[key(&[0x10]), key(&[0x90])]);
        let mut child = parent.clone();
        assert!(child.is_untouched());

        // Reads must not take the copy, or nothing is shared in practice.
        assert!(child.get_leaf_value(&Nibbles::unpack(key(&[0x10]))).is_some());
        assert_eq!(child.size_hint(), parent.size_hint());
        assert!(child.is_untouched());

        let mut updates = B256Map::default();
        updates.insert(key(&[0x20]), LeafUpdate::Changed(vec![7; 64]));
        child.update_leaves(&mut updates, |_, _| {}).unwrap();
        assert!(!child.is_untouched());
    }

    #[test]
    fn only_a_write_that_displaces_a_sharing_parent_counts_as_a_copy() {
        // What the benchmark counter increments on. A handle ending up private is not the same
        // thing: the transition creates storage tries it never copied out of anything, and
        // counting those would overstate what the block paid.
        let parent = trie_with_leaves(&[key(&[0x10]), key(&[0x90])]);
        let mut child = parent.clone();
        let mut updates = B256Map::default();
        updates.insert(key(&[0x20]), LeafUpdate::Changed(vec![7; 64]));
        child.update_leaves(&mut updates, |_, _| {}).unwrap();
        assert!(child.took_copy());

        let mut created = trie_with_leaves(&[key(&[0x30])]);
        assert!(!created.is_untouched());
        assert!(!created.took_copy(), "a trie created here was never copied out of a parent");
        created.update_leaves(&mut B256Map::default(), |_, _| {}).unwrap();
        assert!(!created.took_copy());

        // A clone whose parent has already been dropped is unique, so writing it copies nothing.
        let mut sole = trie_with_leaves(&[key(&[0x40])]).clone();
        assert!(sole.is_untouched());
        sole.update_leaves(&mut B256Map::default(), |_, _| {}).unwrap();
        assert!(!sole.took_copy());
    }

    #[test]
    fn writing_a_clone_leaves_the_parent_exactly_as_it_was() {
        let mut parent = trie_with_leaves(&[key(&[0x10]), key(&[0x90])]);
        let parent_root = parent.root();
        let mut child = parent.clone();

        let mut updates = B256Map::default();
        updates.insert(key(&[0x20]), LeafUpdate::Changed(vec![7; 64]));
        child.update_leaves(&mut updates, |_, _| {}).unwrap();
        let child_root = child.root();

        assert_ne!(child_root, parent_root);
        assert_eq!(parent.root(), parent_root);
        assert!(parent.get_leaf_value(&Nibbles::unpack(key(&[0x20]))).is_none());
        assert!(child.get_leaf_value(&Nibbles::unpack(key(&[0x20]))).is_some());
    }

    #[test]
    fn a_cached_root_is_read_without_taking_a_copy() {
        // The account-trie preparation phase asks every changed account for its storage root, so
        // a `root()` that copied would undo the sharing for exactly the tries a block reads but
        // does not write.
        let parent = trie_with_leaves(&[key(&[0x10]), key(&[0x90])]);
        let mut child = parent.clone();

        let root = child.root();
        assert!(child.is_untouched(), "reading a cached root must not take the copy");
        assert_eq!(root, parent.clone().root());
    }

    #[test]
    fn a_dirty_clone_reports_no_cached_root_until_it_is_recomputed() {
        let parent = trie_with_leaves(&[key(&[0x10]), key(&[0x90])]);
        let mut child = parent.clone();
        assert!(child.cached_root().is_some());

        let mut updates = B256Map::default();
        updates.insert(key(&[0x20]), LeafUpdate::Changed(vec![7; 64]));
        child.update_leaves(&mut updates, |_, _| {}).unwrap();

        assert!(child.cached_root().is_none(), "a pending write must not report a cached root");
        let root = child.root();
        assert_eq!(child.cached_root(), Some(root));
    }

    #[test]
    fn a_dirty_clone_recomputes_its_own_root() {
        let parent = trie_with_leaves(&[key(&[0x10]), key(&[0x90])]);
        let mut child = parent.clone();

        let mut updates = B256Map::default();
        updates.insert(key(&[0x20]), LeafUpdate::Changed(vec![7; 64]));
        child.update_leaves(&mut updates, |_, _| {}).unwrap();

        let mut reference = parent.shared_ref().clone();
        let mut reference_updates = B256Map::default();
        reference_updates.insert(key(&[0x20]), LeafUpdate::Changed(vec![7; 64]));
        reference.update_leaves(&mut reference_updates, |_, _| {}).unwrap();

        assert_eq!(child.root(), reference.root());
    }

    #[test]
    fn wiping_a_clone_invalidates_only_its_own_root() {
        let mut parent = trie_with_leaves(&[key(&[0x10]), key(&[0x90])]);
        let parent_root = parent.root();
        let mut child = parent.clone();

        child.wipe();

        assert_ne!(child.root(), parent_root);
        assert_eq!(parent.root(), parent_root);
    }

    #[test]
    fn retention_leaves_the_root_readable_so_the_next_block_reads_it_free() {
        // Pruning does not dirty the trie, so a snapshot taken after retention must still be able
        // to read its storage roots without copying.
        let mut parent = trie_with_leaves(&[key(&[0x10]), key(&[0x90])]);
        let parent_root = parent.root();
        parent.retain_witness_paths(&[Nibbles::unpack(key(&[0x10]))]);
        assert_eq!(parent.root(), parent_root);

        let mut child = parent.clone();
        assert_eq!(child.root(), parent_root);
        assert!(child.is_untouched(), "retention left the root unreadable without a copy");
    }
}
