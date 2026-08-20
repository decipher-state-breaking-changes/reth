//! The trie representation the cache runs on, selected at construction time.
//!
//! [`CacheTrie`] is a two-variant runtime wrapper over the node-map parallel sparse trie and its
//! exact-size blinded-hash sibling, in the same shape the engine uses for its own configurable
//! trie. A runtime enum rather than a generic parameter, because the cache type is named across
//! four crates and an A/B measurement needs both representations inside one binary and one
//! corpus replay — the differential is selected by a flag, not by a rebuild.
//!
//! The two variants never mix: a cache is constructed on one representation and every trie it
//! creates (account and storage alike) inherits it. Cross-variant equality is `false` by
//! definition — comparing representations is the cross-representation oracle's job, and it
//! compares observables (roots, verdicts, proof targets, anchors, fragments), never structure.

use alloy_primitives::{
    map::{B256Map, HashMap, HashSet},
    B256,
};
use reth_trie_common::{BranchNodeCompact, BranchNodeMasks, Nibbles, ProofTrieNodeV2, TrieNodeV2};
use reth_trie_sparse::{
    errors::SparseTrieResult, BranchSlotCensus, CloneBreakdown, CloneMeasureOptions,
    ExactSparseTrie, LeafLookup, LeafLookupError, LeafUpdate, ParallelSparseTrie, RetainOutcome,
    RetentionOptions, SparseTrie, SparseTrieUpdates,
};
use std::borrow::Cow;

/// Which sparse-trie representation a cache runs on.
///
/// `Exact` is the default since the exact-size representation cleared its adoption gates: the
/// cross-representation oracle showed identical observables on the whole accepted corpus, the
/// counting-allocator differential confirmed the net memory saving, and the live screen showed
/// the clone and retention phases getting faster, not slower. `Parallel` remains selectable for
/// differentials against the engine's own representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CacheTrieRepr {
    /// The node-map parallel sparse trie with fixed 16-slot blinded-hash boxes.
    Parallel,
    /// The exact-size blinded-hash sibling: 32 bytes per actually blinded child.
    #[default]
    Exact,
}

impl CacheTrieRepr {
    /// Stable label for manifests and run summaries.
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Parallel => "parallel",
            Self::Exact => "exact",
        }
    }
}

impl std::str::FromStr for CacheTrieRepr {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "parallel" => Ok(Self::Parallel),
            "exact" => Ok(Self::Exact),
            other => Err(format!("unknown trie representation {other:?}; use parallel or exact")),
        }
    }
}

/// The cache's sparse trie: one of two representations, chosen when the cache is built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheTrie {
    /// Node-map parallel sparse trie.
    Parallel(ParallelSparseTrie),
    /// Exact-size blinded-hash sparse trie.
    Exact(ExactSparseTrie),
}

impl Default for CacheTrie {
    fn default() -> Self {
        Self::new(CacheTrieRepr::default())
    }
}

/// Forwards a method call to whichever representation this trie holds.
macro_rules! delegate {
    ($self:ident => $trie:ident.$($rest:tt)*) => {
        match $self {
            CacheTrie::Parallel($trie) => $trie.$($rest)*,
            CacheTrie::Exact($trie) => $trie.$($rest)*,
        }
    };
}

impl CacheTrie {
    /// A blank trie on the given representation.
    pub fn new(repr: CacheTrieRepr) -> Self {
        match repr {
            CacheTrieRepr::Parallel => Self::Parallel(ParallelSparseTrie::default()),
            CacheTrieRepr::Exact => Self::Exact(ExactSparseTrie::default()),
        }
    }

    /// Which representation this trie holds.
    pub const fn repr(&self) -> CacheTrieRepr {
        match self {
            Self::Parallel(_) => CacheTrieRepr::Parallel,
            Self::Exact(_) => CacheTrieRepr::Exact,
        }
    }

    /// Prunes to the retained witness paths with explicit options; the fork-added retention
    /// entry point both representations implement.
    pub fn retain_witness_paths_with_options(
        &mut self,
        retained_paths: &[Nibbles],
        options: RetentionOptions,
    ) -> RetainOutcome {
        delegate!(self => trie.retain_witness_paths_with_options(retained_paths, options))
    }

    /// Clones the trie while timing and accounting the copy.
    pub fn clone_measured(&self, options: CloneMeasureOptions) -> (Self, CloneBreakdown) {
        match self {
            Self::Parallel(trie) => {
                let (copy, breakdown) = trie.clone_measured(options);
                (Self::Parallel(copy), breakdown)
            }
            Self::Exact(trie) => {
                let (copy, breakdown) = trie.clone_measured(options);
                (Self::Exact(copy), breakdown)
            }
        }
    }

    /// Calls `f` with the path and cached hash of every node whose hash is current.
    pub fn for_each_cached_node_hash(&self, f: impl FnMut(&Nibbles, B256)) {
        delegate!(self => trie.for_each_cached_node_hash(f))
    }

    /// Counts branch child slots, blinded slots, and their depth distribution.
    pub fn branch_slot_census(&self) -> BranchSlotCensus {
        delegate!(self => trie.branch_slot_census())
    }
}

impl SparseTrie for CacheTrie {
    fn set_root(
        &mut self,
        root: TrieNodeV2,
        masks: Option<BranchNodeMasks>,
        retain_updates: bool,
    ) -> SparseTrieResult<()> {
        delegate!(self => trie.set_root(root, masks, retain_updates))
    }

    fn set_updates(&mut self, retain_updates: bool) {
        delegate!(self => trie.set_updates(retain_updates))
    }

    fn reserve_nodes(&mut self, additional: usize) {
        delegate!(self => trie.reserve_nodes(additional))
    }

    fn reveal_node(
        &mut self,
        path: Nibbles,
        node: TrieNodeV2,
        masks: Option<BranchNodeMasks>,
    ) -> SparseTrieResult<()> {
        delegate!(self => trie.reveal_node(path, node, masks))
    }

    fn reveal_nodes(&mut self, nodes: &mut [ProofTrieNodeV2]) -> SparseTrieResult<()> {
        delegate!(self => trie.reveal_nodes(nodes))
    }

    fn root(&mut self) -> B256 {
        delegate!(self => trie.root())
    }

    fn is_root_cached(&self) -> bool {
        delegate!(self => trie.is_root_cached())
    }

    // Must be forwarded: the copy-on-write storage wrapper reads the root through this, and the
    // default `None` would turn every shared-root read into a private copy.
    fn cached_root(&self) -> Option<B256> {
        delegate!(self => trie.cached_root())
    }

    fn update_subtrie_hashes(&mut self) {
        delegate!(self => trie.update_subtrie_hashes())
    }

    fn get_leaf_value(&self, full_path: &Nibbles) -> Option<&Vec<u8>> {
        delegate!(self => trie.get_leaf_value(full_path))
    }

    fn find_leaf(
        &self,
        full_path: &Nibbles,
        expected_value: Option<&Vec<u8>>,
    ) -> Result<LeafLookup, LeafLookupError> {
        delegate!(self => trie.find_leaf(full_path, expected_value))
    }

    fn updates_ref(&self) -> Cow<'_, SparseTrieUpdates> {
        delegate!(self => trie.updates_ref())
    }

    fn take_updates(&mut self) -> SparseTrieUpdates {
        delegate!(self => trie.take_updates())
    }

    fn wipe(&mut self) {
        delegate!(self => trie.wipe())
    }

    fn clear(&mut self) {
        delegate!(self => trie.clear())
    }

    fn shrink_nodes_to(&mut self, size: usize) {
        delegate!(self => trie.shrink_nodes_to(size))
    }

    fn shrink_values_to(&mut self, size: usize) {
        delegate!(self => trie.shrink_values_to(size))
    }

    fn size_hint(&self) -> usize {
        delegate!(self => trie.size_hint())
    }

    fn memory_size(&self) -> usize {
        delegate!(self => trie.memory_size())
    }

    fn prune(&mut self, retained_leaves: &[Nibbles]) -> usize {
        delegate!(self => trie.prune(retained_leaves))
    }

    fn retain_witness_paths(&mut self, retained_paths: &[Nibbles]) -> usize {
        delegate!(self => trie.retain_witness_paths(retained_paths))
    }

    fn update_leaves(
        &mut self,
        updates: &mut B256Map<LeafUpdate>,
        proof_required_fn: impl FnMut(B256, u8),
    ) -> SparseTrieResult<()> {
        delegate!(self => trie.update_leaves(updates, proof_required_fn))
    }

    fn commit_updates(
        &mut self,
        updated: &HashMap<Nibbles, BranchNodeCompact>,
        removed: &HashSet<Nibbles>,
    ) {
        delegate!(self => trie.commit_updates(updated, removed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_adopted_default_is_exact_at_both_wrapper_layers() {
        assert_eq!(CacheTrieRepr::default(), CacheTrieRepr::Exact);
        assert_eq!(CacheTrie::default().repr(), CacheTrieRepr::Exact);
    }
}
