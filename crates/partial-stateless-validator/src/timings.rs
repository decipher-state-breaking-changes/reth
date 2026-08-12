//! Phase instrumentation for one database-free block validation.
//!
//! These types are the validator's own measurement surface, separate from the benchmark *record*
//! types that consume them: the phases are internal to the transition, so only the core can time
//! them, while how a harness serializes a run is the harness's business. Keeping the split here is
//! what lets the standalone validator publish timings without depending on the ExEx benchmark
//! harness, and what keeps [`SidecarValidationOutcome`] a protocol result rather than a report.
//!
//! [`SidecarValidationOutcome`]: crate::SidecarValidationOutcome

use partial_stateless::{
    network_cache::UpdateStats, CacheRootTimings, CloneBreakdown, RetainWitnessPathsMetrics,
    TrieCloneTimings,
};
use serde::Serialize;

/// Serializable trie-walk detail kept inside the enclosing retention total.
#[derive(Debug, Clone, Default, Serialize)]
pub struct RetentionWalkMetrics {
    pub calls: u64,
    pub full_range_calls: u64,
    pub presorted_inputs: u64,
    pub sorted_input_fallbacks: u64,
    pub input_us: u64,
    pub traversal_us: u64,
    pub mutation_us: u64,
    pub finalization_us: u64,
    pub nodes_visited: u64,
    pub edges_visited: u64,
    pub global_prefix_lookups: u64,
    pub retained_path_comparisons: u64,
    pub branch_clones: u64,
    pub branch_clone_bytes: u64,
    pub prune_roots: u64,
    pub nodes_converted: u64,
    pub finalization_upper_nodes_scanned: u64,
    pub finalization_upper_values_scanned: u64,
    pub finalization_branch_masks_scanned: u64,
    pub finalization_lower_subtries_scanned: u64,
    pub unprunable_dirty: u64,
    pub unprunable_inline: u64,
    /// Visited nodes on the descent to a node the walk actually blinded.
    ///
    /// `nodes_visited` minus this is the walk's unproductive share for this block, which is the
    /// ceiling on what any narrower traversal frontier could remove — a ceiling, not a target,
    /// since the same visits also prove exclusion.
    pub visits_on_productive_path: u64,
    /// What computing the field above cost. Outside every walk phase; subtract to compare runs.
    pub productive_path_us: u64,
    /// Prune roots in the upper subtrie, which force finalization to touch every lower slot.
    pub finalization_upper_roots: u64,
    /// Distinct lower subtries holding a prune root, against 256 slots.
    pub finalization_lower_subtries_with_roots: u64,
    /// Node and value entries finalization removed: the size of the subtrees the walk stops at.
    pub finalization_nodes_removed: u64,
    pub finalization_values_removed: u64,
    pub finalization_masks_removed: u64,
    /// Removed masks whose node was already gone.
    ///
    /// Nonzero means a finalization driven by enumerating descendants through the node maps would
    /// diverge from the full scan rather than merely be faster than it.
    pub finalization_masks_removed_without_node: u64,
    /// `finalization_us` split by which map the scan was over.
    pub finalization_masks_us: u64,
    pub finalization_maps_us: u64,
    pub finalization_subtries_us: u64,
}

impl From<&RetainWitnessPathsMetrics> for RetentionWalkMetrics {
    fn from(metrics: &RetainWitnessPathsMetrics) -> Self {
        Self {
            calls: metrics.calls,
            full_range_calls: metrics.full_range_calls,
            presorted_inputs: metrics.presorted_inputs,
            sorted_input_fallbacks: metrics.sorted_input_fallbacks,
            input_us: metrics.input_us,
            traversal_us: metrics.traversal_us,
            mutation_us: metrics.mutation_us,
            finalization_us: metrics.finalization_us,
            nodes_visited: metrics.nodes_visited,
            edges_visited: metrics.edges_visited,
            global_prefix_lookups: metrics.global_prefix_lookups,
            retained_path_comparisons: metrics.retained_path_comparisons,
            branch_clones: metrics.branch_clones,
            branch_clone_bytes: metrics.branch_clone_bytes,
            prune_roots: metrics.prune_roots,
            nodes_converted: metrics.nodes_converted,
            finalization_upper_nodes_scanned: metrics.finalization_upper_nodes_scanned,
            finalization_upper_values_scanned: metrics.finalization_upper_values_scanned,
            finalization_branch_masks_scanned: metrics.finalization_branch_masks_scanned,
            finalization_lower_subtries_scanned: metrics.finalization_lower_subtries_scanned,
            unprunable_dirty: metrics.unprunable_dirty,
            unprunable_inline: metrics.unprunable_inline,
            visits_on_productive_path: metrics.visits_on_productive_path,
            productive_path_us: metrics.productive_path_us,
            finalization_upper_roots: metrics.finalization_upper_roots,
            finalization_lower_subtries_with_roots: metrics.finalization_lower_subtries_with_roots,
            finalization_nodes_removed: metrics.finalization_nodes_removed,
            finalization_values_removed: metrics.finalization_values_removed,
            finalization_masks_removed: metrics.finalization_masks_removed,
            finalization_masks_removed_without_node: metrics
                .finalization_masks_removed_without_node,
            finalization_masks_us: metrics.finalization_masks_us,
            finalization_maps_us: metrics.finalization_maps_us,
            finalization_subtries_us: metrics.finalization_subtries_us,
        }
    }
}

/// Serializable next-anchor detail kept inside `next_cache_anchor_us`.
///
/// The counts are the value-cache composition the root was computed over. This phase scales with
/// them, so two runs covering different blocks are comparable only per entry, not per block.
#[derive(Debug, Clone, Default, Serialize)]
pub struct CacheRootMetrics {
    pub account_collect_sort_us: u64,
    pub storage_collect_sort_us: u64,
    pub code_collect_sort_us: u64,
    pub account_leaf_hash_us: u64,
    pub storage_leaf_hash_us: u64,
    pub code_leaf_hash_us: u64,
    pub account_namespace_us: u64,
    pub storage_namespace_us: u64,
    pub code_namespace_us: u64,
    pub root_us: u64,
    /// Entries each namespace hashed. Mirrored to the record's top-level `cache_*` fields, which
    /// are assigned from these and so cannot disagree with them.
    pub accounts: u64,
    pub storage: u64,
    pub codes: u64,
    /// Nonzero only if the memo answered, which the validator path never lets happen: the cache
    /// update immediately before invalidates it. A nonzero sum here means the phase mean is
    /// diluted by free samples and the record should be excluded, not averaged.
    pub memo_hits: u64,
}

impl From<&CacheRootTimings> for CacheRootMetrics {
    fn from(timings: &CacheRootTimings) -> Self {
        Self {
            account_collect_sort_us: timings.account_collect_sort_us,
            storage_collect_sort_us: timings.storage_collect_sort_us,
            code_collect_sort_us: timings.code_collect_sort_us,
            account_leaf_hash_us: timings.account_leaf_hash_us,
            storage_leaf_hash_us: timings.storage_leaf_hash_us,
            code_leaf_hash_us: timings.code_leaf_hash_us,
            account_namespace_us: timings.account_namespace_us,
            storage_namespace_us: timings.storage_namespace_us,
            code_namespace_us: timings.code_namespace_us,
            root_us: timings.root_us,
            accounts: timings.accounts,
            storage: timings.storage,
            codes: timings.codes,
            memo_hits: u64::from(timings.memo_hit),
        }
    }
}

/// What this block's transition moved in the value cache, per namespace.
///
/// Published because the *gross* movement is not recoverable from anything else a record carries.
/// The `cache_*` populations only show net change, and a refresh moves no population at all while
/// still changing that entry's leaf: `last_accessed_block` is part of every leaf preimage. So
/// `added + refreshed`, the leaves whose digest this block invalidated, can be read here and
/// nowhere else, while `evicted` counts the leaves it dropped instead.
#[derive(Debug, Clone, Default, Serialize)]
pub struct CacheDeltaMetrics {
    pub accounts_added: u64,
    pub accounts_refreshed: u64,
    pub accounts_evicted: u64,
    pub storage_added: u64,
    pub storage_refreshed: u64,
    pub storage_evicted: u64,
    pub codes_added: u64,
    pub codes_refreshed: u64,
    pub codes_evicted: u64,
}

impl From<&UpdateStats> for CacheDeltaMetrics {
    fn from(stats: &UpdateStats) -> Self {
        Self {
            accounts_added: stats.accounts_added as u64,
            accounts_refreshed: stats.accounts_refreshed as u64,
            accounts_evicted: stats.accounts_evicted as u64,
            storage_added: stats.storage_added as u64,
            storage_refreshed: stats.storage_refreshed as u64,
            storage_evicted: stats.storage_evicted as u64,
            codes_added: stats.codes_added as u64,
            codes_refreshed: stats.codes_refreshed as u64,
            codes_evicted: stats.codes_evicted as u64,
        }
    }
}

/// Serializable transactional-snapshot detail kept inside `trie_clone_us`.
#[derive(Debug, Clone, Default, Serialize)]
pub struct TrieCloneMetrics {
    pub account_trie_us: u64,
    /// The account copy's own split; included in `account_trie_us`, never summed into a total.
    pub account_trie_detail: CloneBreakdownMetrics,
    pub storage_tries_us: u64,
    pub warm_membership_us: u64,
    pub retained_paths_us: u64,
    pub storage_tries: u64,
    pub warm_accounts: u64,
    pub warm_storage: u64,
    pub retained_account_paths: u64,
}

impl From<&TrieCloneTimings> for TrieCloneMetrics {
    fn from(timings: &TrieCloneTimings) -> Self {
        Self {
            account_trie_us: timings.account_trie_us,
            account_trie_detail: CloneBreakdownMetrics::from(&timings.account_trie_breakdown),
            storage_tries_us: timings.storage_tries_us,
            warm_membership_us: timings.warm_membership_us,
            retained_paths_us: timings.retained_paths_us,
            storage_tries: timings.storage_tries,
            warm_accounts: timings.warm_accounts,
            warm_storage: timings.warm_storage,
            retained_account_paths: timings.retained_account_paths,
        }
    }
}

/// Where the account-trie copy's time, bytes, and allocations went.
///
/// The five `_us` components sum to `account_trie_us`; the two beside them do not, because they are
/// what the measurement itself costs. `branch_hash_*` are a subset of `nodes_*`, being the
/// unconditional 512-byte box every branch node carries whether or not a child is blinded — the one
/// narrow representation candidate this phase has, and the reason the split is worth taking.
#[derive(Debug, Clone, Default, Serialize)]
pub struct CloneBreakdownMetrics {
    pub nodes_us: u64,
    pub values_us: u64,
    pub masks_us: u64,
    pub buffers_us: u64,
    pub rest_us: u64,
    /// The accounting walk's own cost. Outside `account_trie_us`; subtract to compare runs.
    pub accounting_us: u64,
    /// Allocating, copying, and freeing one branch-hash box per branch node.
    ///
    /// Outside `account_trie_us` for the same reason, and zero when the probe is off. This is the
    /// box priced rather than inferred from its byte share, which is what the candidate must beat.
    pub branch_hash_probe_us: u64,
    pub total_bytes: u64,
    pub nodes_bytes: u64,
    pub values_bytes: u64,
    pub masks_bytes: u64,
    pub buffers_bytes: u64,
    pub rest_bytes: u64,
    /// Bytes in branch-hash boxes. A subset of `nodes_bytes`.
    pub branch_hash_bytes: u64,
    pub total_allocs: u64,
    pub nodes_allocs: u64,
    pub values_allocs: u64,
    /// Separate allocations for branch-hash boxes. A subset of `nodes_allocs`.
    pub branch_hash_allocs: u64,
    pub subtries: u64,
    pub node_entries: u64,
    pub branch_nodes: u64,
    pub extension_nodes: u64,
    pub leaf_nodes: u64,
    pub value_entries: u64,
    pub mask_entries: u64,
}

impl From<&CloneBreakdown> for CloneBreakdownMetrics {
    fn from(breakdown: &CloneBreakdown) -> Self {
        Self {
            nodes_us: breakdown.nodes_us,
            values_us: breakdown.values_us,
            masks_us: breakdown.masks_us,
            buffers_us: breakdown.buffers_us,
            rest_us: breakdown.rest_us,
            accounting_us: breakdown.accounting_us,
            branch_hash_probe_us: breakdown.branch_hash_probe_us,
            total_bytes: breakdown.total_bytes,
            nodes_bytes: breakdown.nodes_bytes,
            values_bytes: breakdown.values_bytes,
            masks_bytes: breakdown.masks_bytes,
            buffers_bytes: breakdown.buffers_bytes,
            rest_bytes: breakdown.rest_bytes,
            branch_hash_bytes: breakdown.branch_hash_bytes,
            total_allocs: breakdown.total_allocs,
            nodes_allocs: breakdown.nodes_allocs,
            values_allocs: breakdown.values_allocs,
            branch_hash_allocs: breakdown.branch_hash_allocs,
            subtries: breakdown.subtries,
            node_entries: breakdown.node_entries,
            branch_nodes: breakdown.branch_nodes,
            extension_nodes: breakdown.extension_nodes,
            leaf_nodes: breakdown.leaf_nodes,
            value_entries: breakdown.value_entries,
            mask_entries: breakdown.mask_entries,
        }
    }
}

/// What it cost to turn untrusted input into a block this validator agreed to execute.
///
/// Every field is `None` on a path that was handed an already-admitted block. See
/// [`ValidationPhaseTimings::admission`] for why that is not the same as zero.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct AdmissionTimings {
    /// Source the block arrived from, naming which of the fields below can be `Some`.
    pub source: AdmissionSource,
    /// Decoding the transport frame into an execution payload.
    pub input_decode_us: Option<u64>,
    /// Engine-API payload layout and block-hash validation.
    pub payload_validation_us: Option<u64>,
    /// Recovering every transaction sender from its signature.
    pub sender_recovery_us: Option<u64>,
    /// Canonical header, pre-execution, and against-parent consensus validation.
    pub pre_execution_consensus_us: Option<u64>,
}

impl AdmissionTimings {
    /// Admission work actually performed, treating an absent phase as no cost rather than as zero.
    pub fn total_us(&self) -> u64 {
        self.input_decode_us
            .unwrap_or_default()
            .saturating_add(self.payload_validation_us.unwrap_or_default())
            .saturating_add(self.sender_recovery_us.unwrap_or_default())
            .saturating_add(self.pre_execution_consensus_us.unwrap_or_default())
    }
}

/// Which entry point handed this validator its block.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionSource {
    /// A caller that had already decoded, layout-checked, and sender-recovered the block.
    ///
    /// The default because it is the ExEx's path, and because a record written before this field
    /// existed describes exactly that.
    #[default]
    Recovered,
    /// Untrusted Engine-API payload bytes this validator admitted itself.
    ExecutionData,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ValidationPhaseTimings {
    pub deserialize_us: u64,
    /// Admission phases, `None` where the caller's input arrived already past that stage.
    ///
    /// `None` and `Some(0)` are different facts and must stay distinguishable. A full node's ExEx
    /// is handed a block its Engine has already decoded, layout-checked, and sender-recovered, so
    /// these read `None` — the work was done, elsewhere, and is nobody's cost here. A standalone
    /// validator admits untrusted input and does all of it itself. Collapsing that to `0` would
    /// report a validator that skips admission as one that admits for free, which is exactly the
    /// coverage gap B3's telemetry had to be corrected for.
    pub admission: AdmissionTimings,
    pub context_check_us: u64,
    pub witness_self_consistency_us: u64,
    pub materialize_us: u64,
    pub provider_setup_us: u64,
    /// Executor call wall time, including benchmark-only accessed-state capture.
    pub evm_call_us: u64,
    /// Benchmark-only time spent extracting `BlockAccessedState` from the EVM state.
    pub accessed_state_capture_us: u64,
    /// Executor call excluding benchmark-only access capture. Includes all state-provider lookups
    /// performed by the EVM: Full DB/cache reads for Vanilla, in-memory witness/cache reads here.
    pub evm_us: u64,
    /// Canonical post-execution consensus validation: gas used, receipts root, logs bloom, and the
    /// fork-gated requests hash.
    ///
    /// Runs before the post state is hashed, so a block that fails it pays none of the trie work
    /// below. Unlike the admission phases this is not optional — every path through the validator
    /// executes it — so it is a plain `u64`.
    pub post_execution_consensus_us: u64,
    pub hash_post_state_us: u64,
    /// Cost of opening the transactional trie snapshot.
    ///
    /// Since V3 this is the account-trie copy plus a refcount bump per storage trie. The storage
    /// copies the block turns out to need are paid inside `state_root_us` and `trie_retention_us`
    /// instead, and counted by `trie_storage_tries_copied`.
    pub trie_clone_us: u64,
    /// The clone's internal split; included in `trie_clone_us`, never summed into a total.
    pub trie_clone_detail: TrieCloneMetrics,
    /// Copy-on-write copies of storage tries taken across the whole transaction.
    ///
    /// Counts copies, not survivors: a trie copied and then dropped by retention still counts, and
    /// a trie the transition created rather than copied does not.
    pub trie_storage_tries_copied: u64,
    /// Storage tries the snapshot held after retention, copied or shared.
    pub trie_storage_tries_total: u64,
    pub state_root_us: u64,
    pub root_completeness_us: u64,
    pub miss_policy_check_us: u64,
    pub cache_update_us: u64,
    /// Rehashing the leaf digests of the entries this block moved; measured inside
    /// `cache_update_us`, never summed into a total.
    ///
    /// The cost the cache root stopped paying, charged to the block that caused it. It belongs
    /// beside `next_cache_anchor_us` and not inside it, so comparing the anchor across the change
    /// means comparing the sum of the two against the anchor a pre-index run reported — the anchor
    /// alone would credit the change with work that merely moved.
    pub cache_root_index_maintenance_us: u64,
    /// What the transition inside `cache_update_us` moved, per namespace. Counts, not times:
    /// reported beside the phase rather than as a component of it.
    pub cache_delta: CacheDeltaMetrics,
    pub trie_retention_us: u64,
    /// Retention's internal split. Reported beside `trie_retention_us`, never summed into a total:
    /// the phases below are measured inside it, so adding them would double-count.
    pub retention_warm_membership_us: u64,
    pub retention_storage_paths_us: u64,
    pub retention_account_paths_us: u64,
    pub retention_account_trie_us: u64,
    /// Account-trie subphases and range-walk work; included in `retention_account_trie_us`.
    pub retention_account_trie_detail: RetentionWalkMetrics,
    pub retention_storage_tries_us: u64,
    /// Aggregated storage-trie subphases and work; included in `retention_storage_tries_us`.
    pub retention_storage_trie_detail: RetentionWalkMetrics,
    /// Retained account paths handed to the account-trie prune.
    pub retention_account_paths: u64,
    pub retention_storage_tries_pruned: u64,
    pub retention_storage_tries_skipped: u64,
    /// Copy-on-write copies taken before the storage walks, and what they cost.
    ///
    /// Included in `retention_storage_tries_us` but outside `retention_storage_trie_detail`, whose
    /// timers start after the copy. This is the term that makes the storage total exceed the sum
    /// of its walk phases, and it is transactional-snapshot cost rather than retention work.
    pub retention_storage_trie_cow_us: u64,
    pub retention_storage_trie_cow_copies: u64,
    /// Releasing storage tries whose address left the retained set, and how many were released.
    pub retention_storage_trie_drop_us: u64,
    pub retention_storage_tries_dropped: u64,
    /// 1 when the retained sets were rebuilt from the whole value cache instead of patched.
    ///
    /// Summed over a run this is the fallback count: the delta path is correct either way, so
    /// this measures how much of the optimization a run actually got.
    pub retention_full_rebuild: u64,
    pub next_cache_anchor_us: u64,
    /// The anchor's internal split; included in `next_cache_anchor_us`, never summed into a total.
    pub next_cache_anchor_detail: CacheRootMetrics,
    pub trie_commit_us: u64,
    /// Measured validator work outside the named phase boundaries (executor setup/fingerprints).
    pub unattributed_us: u64,
    pub provider_root_us: u64,
    /// Actual DB-free validation time, including diagnostics and persistent-cache maintenance.
    pub raw_total_us: u64,
    /// `raw_total_us` without the warn-only root-completeness report.
    pub protocol_total_us: u64,
    /// `protocol_total_us` without exact miss-only canonicality policy checking.
    pub root_validation_total_us: u64,
    /// Deserialize + witness checks/materialization + EVM + hashing + sparse-trie root.
    pub execution_core_us: u64,
    /// Serialized sidecar validation/materialization + witness-backed provider + EVM.
    pub state_access_execution_us: u64,
}

impl ValidationPhaseTimings {
    pub fn recompute_totals(&mut self) {
        self.raw_total_us = self
            .deserialize_us
            .saturating_add(self.context_check_us)
            .saturating_add(self.witness_self_consistency_us)
            .saturating_add(self.materialize_us)
            .saturating_add(self.provider_setup_us)
            .saturating_add(self.admission.total_us())
            .saturating_add(self.evm_call_us)
            .saturating_add(self.post_execution_consensus_us)
            .saturating_add(self.hash_post_state_us)
            .saturating_add(self.trie_clone_us)
            .saturating_add(self.state_root_us)
            .saturating_add(self.root_completeness_us)
            .saturating_add(self.miss_policy_check_us)
            .saturating_add(self.cache_update_us)
            .saturating_add(self.trie_retention_us)
            .saturating_add(self.next_cache_anchor_us)
            .saturating_add(self.trie_commit_us)
            .saturating_add(self.unattributed_us);
        self.protocol_total_us = self.raw_total_us.saturating_sub(self.root_completeness_us);
        self.root_validation_total_us =
            self.protocol_total_us.saturating_sub(self.miss_policy_check_us);
        self.execution_core_us = self
            .deserialize_us
            .saturating_add(self.witness_self_consistency_us)
            .saturating_add(self.materialize_us)
            .saturating_add(self.provider_setup_us)
            .saturating_add(self.evm_us)
            .saturating_add(self.hash_post_state_us)
            .saturating_add(self.state_root_us);
        self.state_access_execution_us = self
            .deserialize_us
            .saturating_add(self.context_check_us)
            .saturating_add(self.witness_self_consistency_us)
            .saturating_add(self.materialize_us)
            .saturating_add(self.provider_setup_us)
            .saturating_add(self.evm_us);
    }

    pub fn set_deserialize_us(&mut self, deserialize_us: u64) {
        self.deserialize_us = deserialize_us;
        self.recompute_totals();
    }
}
