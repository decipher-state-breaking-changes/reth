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
    RetentionOptions, SparseTrie, SparseTrieUpdates, UndoFrame,
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

// Disk undo supports the recording representation. Parallel caches retain a Full fallback;
// callers must shorten the disk-retained suffix rather than silently encoding another layout.
impl serde::Serialize for CacheTrie {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Exact(trie) => serde::Serialize::serialize(trie, serializer),
            Self::Parallel(_) => Err(serde::ser::Error::custom("parallel trie has no disk undo")),
        }
    }
}

impl<'de> serde::Deserialize<'de> for CacheTrie {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self::Exact(serde::Deserialize::deserialize(deserializer)?))
    }
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

    /// Retains exactly what [`Self::retain_witness_paths_with_options`] would, walking only toward
    /// what the undo record says changed and `moved_paths`. `None`, having changed nothing, when
    /// the trie cannot vouch for that; `Parallel` never can, having no record.
    ///
    /// See [`ExactSparseTrie::retain_witness_paths_since_undo_began`] for what the caller vouches.
    pub fn retain_witness_paths_since_undo_began(
        &mut self,
        retained_paths: &[Nibbles],
        moved_paths: &[Nibbles],
        options: RetentionOptions,
    ) -> Option<RetainOutcome> {
        match self {
            Self::Parallel(_) => None,
            Self::Exact(trie) => {
                trie.retain_witness_paths_since_undo_began(retained_paths, moved_paths, options)
            }
        }
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

    /// Starts keeping an undo record, if this representation has one.
    ///
    /// Only `Exact` carries the record. A cache built on `Parallel` reports `false` from
    /// [`Self::is_recording_undo`] for the rest of the block and its holder keeps whole retained
    /// generations instead of frames — the fallback the two-representation split has always had,
    /// rather than a second journal to maintain in `parallel.rs`.
    pub fn begin_undo(&mut self) {
        match self {
            Self::Parallel(_) => {}
            Self::Exact(trie) => trie.begin_undo(),
        }
    }

    /// Whether an undo record is being kept.
    pub const fn is_recording_undo(&self) -> bool {
        match self {
            Self::Parallel(_) => false,
            Self::Exact(trie) => trie.is_recording_undo(),
        }
    }

    /// Stops recording and returns the record, or `None` if none was being kept.
    pub fn take_undo(&mut self) -> Option<UndoFrame> {
        match self {
            Self::Parallel(_) => None,
            Self::Exact(trie) => trie.take_undo(),
        }
    }

    /// Reverses every change `frame` recorded, returning whether it could be applied.
    ///
    /// `false` means the frame reached a representation that cannot hold one, which only a caller
    /// that mixed a frame from one cache with a trie from another can produce: `take_undo` hands
    /// out a frame on `Exact` alone. Reported rather than asserted, because the recovery path this
    /// serves has a fallback — rebuild — and no block to reject.
    #[must_use]
    pub fn undo(&mut self, frame: UndoFrame) -> bool {
        match self {
            Self::Parallel(_) => false,
            Self::Exact(trie) => {
                trie.undo(frame);
                true
            }
        }
    }

    /// Counts branch child slots, blinded slots, and their depth distribution.
    pub fn branch_slot_census(&self) -> BranchSlotCensus {
        delegate!(self => trie.branch_slot_census())
    }
}

/// A storage trie's record starts at its first write, from the handle that shares it, rather than
/// at a snapshot the way the account trie's does.
impl crate::shared_trie::UndoRecording for CacheTrie {
    type Record = UndoFrame;

    fn restart_undo(&mut self) {
        if let Self::Exact(trie) = self {
            drop(trie.take_undo());
            trie.begin_undo();
        }
    }

    fn take_undo(&mut self) -> Option<UndoFrame> {
        Self::take_undo(self)
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

/// The variable that sets the parallelism floor of this process's `Exact` tries.
///
/// `<reveal>,<update>`: the fewest nodes a reveal, and the fewest changed keys a hash update, must
/// have before either runs on the thread pool. Unset means `0,0`, under which every reveal and
/// update may go parallel, as before the variable existed. Process-wide because the floor is: it
/// applies to every `Exact` trie in the process, the producer's included when they share one.
pub const TRIE_PARALLEL_MIN_VAR: &str = "PS_TRIE_PARALLEL_MIN";

/// Applies [`TRIE_PARALLEL_MIN_VAR`] to this process and returns the floor now in force as
/// `[reveal, update]`. An unparseable value is an error rather than the default, so a sweep arm
/// cannot silently run the baseline.
pub fn apply_trie_parallel_min_from_env() -> Result<[usize; 2], String> {
    let floor = match std::env::var(TRIE_PARALLEL_MIN_VAR) {
        Ok(raw) => parse_trie_parallel_min(&raw)
            .map_err(|err| format!("{TRIE_PARALLEL_MIN_VAR}={raw:?}: {err}"))?,
        Err(std::env::VarError::NotPresent) => [0, 0],
        Err(err) => return Err(format!("{TRIE_PARALLEL_MIN_VAR}: {err}")),
    };
    reth_trie_sparse::set_exact_parallelism_floor(reth_trie_sparse::ParallelismThresholds {
        min_revealed_nodes: floor[0],
        min_updated_nodes: floor[1],
    });
    Ok(floor)
}

/// The floor now in force for this process's `Exact` tries, as `[reveal, update]`.
pub fn trie_parallel_min() -> [usize; 2] {
    let floor = reth_trie_sparse::exact_parallelism_floor();
    [floor.min_revealed_nodes, floor.min_updated_nodes]
}

fn parse_trie_parallel_min(raw: &str) -> Result<[usize; 2], String> {
    let (reveal, update) =
        raw.split_once(',').ok_or("want `<reveal>,<update>`, two node counts")?;
    let count = |part: &str| part.trim().parse::<usize>().map_err(|err| err.to_string());
    Ok([count(reveal)?, count(update)?])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two counts or nothing: a sweep arm that mistypes its value must fail, not run the default.
    #[test]
    fn the_parallel_floor_is_two_counts() {
        assert_eq!(parse_trie_parallel_min("0,0"), Ok([0, 0]));
        assert_eq!(parse_trie_parallel_min(" 64, 256 "), Ok([64, 256]));
        for raw in ["", "64", "64,", ",64", "a,b", "-1,0", "1,2,3"] {
            assert!(parse_trie_parallel_min(raw).is_err(), "{raw:?}");
        }
    }

    #[test]
    fn the_adopted_default_is_exact_at_both_wrapper_layers() {
        assert_eq!(CacheTrieRepr::default(), CacheTrieRepr::Exact);
        assert_eq!(CacheTrie::default().repr(), CacheTrieRepr::Exact);
    }
}
