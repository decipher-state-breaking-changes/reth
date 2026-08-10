use alloy_primitives::B256;
use partial_stateless::{
    network_cache::UpdateStats, CacheRootTimings, CloneBreakdown, PartialExecutionWitness,
    PartialStatelessSidecar, RetainWitnessPathsMetrics, TrieCloneTimings, TrieMutationMetrics,
};
use serde::Serialize;
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
};

/// Current on-disk schema for paired validation benchmark records.
///
/// V5 adds the next-anchor and trie-clone splits plus the value-cache composition the anchor was
/// computed over. V6 adds `cache_delta`, the per-namespace added/refreshed/evicted counts the
/// transition moved. V7 adds `cache_root_index_maintenance_us`, without which the anchor cannot be
/// compared across the leaf digest index — the index moves work out of the anchor and into the
/// cache update, so a V6 record and a V7 record report the same phase over different scopes.
/// Analyzer reports emit those sections only when the fields are present, so a V4, V5, or V6 run
/// still regenerates.
///
/// V8 adds the storage-prune copy-on-write and drop split. Storage retention's total has always
/// exceeded the sum of its walk phases; these fields name the difference rather than leaving it as
/// an unattributed residual, which is what makes the storage half of retention a targetable number.
///
/// V9 opens the two halves of the account trie, which are the two largest phases left.
///
/// `trie_clone_detail.account_trie_detail` splits the copy by component in time, bytes, and
/// allocations. One timer over it cannot distinguish a cost spread evenly across everything the
/// trie holds from one concentrated in a single field, and the two imply different fixes.
///
/// The retention walk gains `visits_on_productive_path`, which bounds what a narrower traversal
/// frontier could remove, and a finalization split by map plus the counts that decide whether
/// enumerating descendants can replace the branch-mask scan at all.
///
/// Both splits' phase timers are free and always present, because they ride along with work the
/// block already does. The counts that describe shape rather than cost are not free — the copy's
/// census walks every node and value entry, which a hash-map copy never does; the box probe
/// allocates one box per branch node; the walk re-descends to every prune root — so they are
/// collected only when `PS_TRIE_SHAPE_DIAGNOSTICS` asks: `1` for the census and the walk counts,
/// `probe` to also price the branch-hash box. A.16 measured them at 8.94, 0.49, and 9.11 ms per
/// block, so a default run leaves them off and reports zeroes rather than paying 4.7% of raw
/// validation for numbers that move with cache size rather than with the block.
///
/// Three fields therefore measure the instrumentation rather than the work and sit outside their
/// phase: `account_trie_detail.accounting_us`, `account_trie_detail.branch_hash_probe_us`, and
/// each walk's `productive_path_us`. All three are zero unless requested. Subtract whichever are
/// nonzero before comparing a V9 run's totals or per-entry coefficients against a V8 one.
pub const VALIDATION_BENCHMARK_SCHEMA_VERSION: u64 = 9;

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

#[derive(Debug, Clone, Default, Serialize)]
pub struct ValidationPhaseTimings {
    pub deserialize_us: u64,
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
            .saturating_add(self.evm_call_us)
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

#[derive(Debug, Clone, Default, Serialize)]
pub struct WitnessSizeBreakdown {
    pub serialized_witness_bytes: usize,
    pub state_node_bytes: usize,
    pub state_nodes: usize,
    pub bytecode_bytes: usize,
    pub key_preimage_bytes: usize,
    pub ancestor_header_bytes: usize,
}

impl WitnessSizeBreakdown {
    pub fn from_witness(witness: &PartialExecutionWitness) -> eyre::Result<Self> {
        let (state_node_bytes, state_nodes) = match &witness.state {
            partial_stateless::PartialExecutionWitnessState::MptMultiProof(bytes) => {
                (bytes.len(), 0)
            }
            partial_stateless::PartialExecutionWitnessState::MptTransitionNodes(nodes) => {
                (nodes.iter().map(|node| node.len()).sum(), nodes.len())
            }
        };
        Ok(Self {
            serialized_witness_bytes: bincode::serialize(witness)?.len(),
            state_node_bytes,
            state_nodes,
            bytecode_bytes: witness.codes.iter().map(|code| code.len()).sum(),
            key_preimage_bytes: witness.keys.iter().map(|key| key.len()).sum(),
            ancestor_header_bytes: witness.headers.iter().map(|header| header.len()).sum(),
        })
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ValidationBenchmarkRecord {
    pub schema_version: u64,
    pub block_number: u64,
    pub block_hash: B256,
    pub gas_used: u64,
    pub historical_gas_used: u64,
    pub tx_count: usize,
    pub verifier_order: &'static str,
    /// Which witness was *built* first. Alternates independently of `verifier_order` so that
    /// neither side permanently pays the cold page-cache read; see `weak_builds_first`.
    pub builder_order: &'static str,
    pub historical_full_db_evm_us: u64,
    pub partial_witness_build_us: u64,
    pub weak_witness_build_us: u64,
    pub partial_serialize_us: u64,
    pub weak_serialize_us: u64,
    pub partial: ValidationPhaseTimings,
    pub weak: ValidationPhaseTimings,
    pub partial_witness: WitnessSizeBreakdown,
    pub weak_witness: WitnessSizeBreakdown,
    pub partial_sidecar_bytes: usize,
    pub weak_sidecar_bytes: usize,
    pub value_cache_bytes: usize,
    /// Value-cache composition after this block's transition, read off the next-anchor
    /// computation so it is exactly the population that phase hashed rather than a second count
    /// taken elsewhere. The next anchor and trie retention are both functions of these three
    /// numbers, so comparing two runs over different blocks requires normalizing by them.
    pub cache_accounts: u64,
    pub cache_storage: u64,
    pub cache_codes: u64,
    pub trie_cache_bytes: usize,
    pub retained_generation: RetainedGenerationBytes,
    pub expected_state_root: B256,
    pub partial_state_root: B256,
    pub weak_state_root: B256,
    pub expected_receipts_root: B256,
    pub historical_receipts_root: B256,
    pub partial_receipts_root: B256,
    pub weak_receipts_root: B256,
    pub expected_requests_hash: Option<B256>,
    pub historical_requests_hash: B256,
    pub partial_requests_hash: B256,
    pub weak_requests_hash: B256,
    pub valid: bool,
}

/// What the K = 1 retained generation was holding when a block began.
///
/// `total_bytes` is what the retained trie cache measures on its own; `exclusive_bytes` is the
/// part of it that no other generation shares, which is what dropping it would give back. The two
/// are reported together because the gap between them is the point: a snapshot shares storage
/// tries with its parent, so the cost of keeping one is far below its apparent size, and only the
/// exclusive figure is comparable with a resident-memory difference.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct RetainedGenerationBytes {
    /// Whether this run retains at all. False is the memory control, not a failure.
    pub enabled: bool,
    /// Whether a generation was actually being held. False while cold, warming, or recovering.
    pub present: bool,
    pub total_bytes: usize,
    pub exclusive_bytes: usize,
}

/// Per-block builder telemetry used to isolate cache snapshot and initial proof costs.
#[derive(Debug, Clone, Default, Serialize)]
pub struct BuilderBenchmarkRecord {
    pub schema_version: u64,
    pub block_number: u64,
    pub block_hash: B256,
    pub historical_full_db_evm_us: u64,
    pub builder_total_us: u64,
    pub transition_witness_build_us: u64,
    pub snapshot_created: bool,
    pub snapshot_us: u64,
    pub snapshot_estimated_bytes: usize,
    pub cache_parent_synced: bool,
    pub initial_proof_source: &'static str,
    pub initial_provider_us: u64,
    pub initial_targets: usize,
    pub distinct_storage_tries: usize,
    pub parallel_storage_workers: usize,
    pub parallel_account_workers: usize,
    pub initial_proof_nodes: usize,
    pub initial_proof_bytes: usize,
    pub witness_commitment: Option<B256>,
    pub sidecar_constructed: bool,
    pub sidecar_published: bool,
    pub value_cache_bytes: usize,
    pub trie_cache_bytes: usize,
    pub retained_generation: RetainedGenerationBytes,
    /// Logical size of the trie the per-block snapshot covers.
    ///
    /// Since V3 the snapshot shares storage tries with its parent, so this is the size it stands
    /// for rather than the size it copied; `trie_storage_tries_copied` is what it really copied.
    pub trie_clone_bytes: usize,
    /// Process RSS moved across the clone. Process-wide, so meaningful only in aggregate.
    pub trie_clone_rss_delta_bytes: i64,
    /// Copy-on-write copies of storage tries the builder's transaction took.
    ///
    /// Counts copies, not survivors: a trie copied and then dropped by retention still counts, and
    /// a trie the transition created rather than copied does not.
    pub trie_storage_tries_copied: u64,
    /// Storage tries the snapshot held after retention, copied or shared.
    pub trie_storage_tries_total: u64,
    /// Mutation footprint of this block against the cloned parent trie. Only present under
    /// `PS_TRIE_CACHE_DIAGNOSTICS`, since measuring it walks every retained path.
    pub trie_mutation: Option<TrieMutationSummary>,
}

/// Scalar summary of how much of the cloned trie a block dirtied.
///
/// The per-storage-trie breakdown behind these totals goes to the log rather than here: it is
/// unbounded in length, and a benchmark record is appended once per block.
#[derive(Debug, Clone, Default, Serialize)]
pub struct TrieMutationSummary {
    pub retained_account_paths: usize,
    pub dirtied_account_paths: usize,
    pub retained_storage_paths: usize,
    pub dirtied_storage_paths: usize,
    pub dirtied_storage_tries: usize,
    pub revealed_nodes: usize,
    /// Dirtied share of retained leaf paths.
    pub dirtied_path_share: f64,
    /// Dirtied share of distinct account-key prefixes at the deepest level that still fans out.
    /// Closer than the leaf share to the share of *nodes* a block invalidates.
    pub deepest_account_prefix_share: f64,
}

impl From<&TrieMutationMetrics> for TrieMutationSummary {
    fn from(metrics: &TrieMutationMetrics) -> Self {
        Self {
            retained_account_paths: metrics.retained_account_paths,
            dirtied_account_paths: metrics.dirtied_account_paths,
            retained_storage_paths: metrics.retained_storage_paths,
            dirtied_storage_paths: metrics.dirtied_storage_paths,
            dirtied_storage_tries: metrics.dirtied_storage_tries,
            revealed_nodes: metrics.revealed_nodes(),
            dirtied_path_share: metrics.dirtied_path_share(),
            deepest_account_prefix_share: metrics.deepest_account_prefix().dirtied_share(),
        }
    }
}

pub fn serialize_sidecar_for_benchmark(sidecar: &PartialStatelessSidecar) -> eyre::Result<Vec<u8>> {
    Ok(bincode::serialize(sidecar)?)
}

pub fn deserialize_sidecar_for_benchmark(
    bytes: &[u8],
) -> eyre::Result<(PartialStatelessSidecar, u64)> {
    let start = std::time::Instant::now();
    let decoded = bincode::deserialize(bytes)?;
    Ok((decoded, start.elapsed().as_micros() as u64))
}

pub fn append_record(path: &Path, record: &ValidationBenchmarkRecord) -> eyre::Result<()> {
    append_json_record(path, record)
}

pub fn append_builder_record(path: &Path, record: &BuilderBenchmarkRecord) -> eyre::Result<()> {
    append_json_record(path, record)
}

fn append_json_record(path: &Path, record: &impl Serialize) -> eyre::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut file, record)?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::Bytes;
    use partial_stateless::{PartialExecutionWitnessState, PrefixCoverage};

    #[test]
    fn headline_totals_include_deserialization_and_cache_maintenance() {
        let mut timings = ValidationPhaseTimings {
            evm_call_us: 83,
            evm_us: 83,
            trie_clone_us: 3,
            cache_update_us: 5,
            trie_retention_us: 4,
            next_cache_anchor_us: 5,
            ..Default::default()
        };
        timings.recompute_totals();
        timings.set_deserialize_us(7);

        assert_eq!(timings.deserialize_us, 7);
        assert_eq!(timings.raw_total_us, 107);
    }

    #[test]
    fn adjusted_totals_subtract_only_declared_diagnostic_phases() {
        let mut timings = ValidationPhaseTimings {
            deserialize_us: 1,
            context_check_us: 2,
            witness_self_consistency_us: 3,
            materialize_us: 4,
            evm_call_us: 5,
            evm_us: 5,
            hash_post_state_us: 6,
            trie_clone_us: 7,
            state_root_us: 8,
            root_completeness_us: 9,
            miss_policy_check_us: 10,
            cache_update_us: 11,
            trie_retention_us: 12,
            next_cache_anchor_us: 13,
            trie_commit_us: 14,
            unattributed_us: 15,
            // Detail is nested inside `trie_retention_us` and must not be re-added.
            retention_account_trie_detail: RetentionWalkMetrics {
                traversal_us: 10_000,
                mutation_us: 20_000,
                finalization_us: 30_000,
                ..Default::default()
            },
            ..Default::default()
        };
        timings.recompute_totals();

        assert_eq!(timings.raw_total_us, 120);
        assert_eq!(timings.protocol_total_us, 111);
        assert_eq!(timings.root_validation_total_us, 101);
        assert_eq!(timings.execution_core_us, 27);
        assert_eq!(timings.state_access_execution_us, 15);
    }

    #[test]
    fn primary_excludes_access_capture_and_post_execution_work() {
        let mut timings = ValidationPhaseTimings {
            deserialize_us: 2,
            context_check_us: 3,
            witness_self_consistency_us: 5,
            materialize_us: 7,
            provider_setup_us: 11,
            evm_call_us: 113,
            accessed_state_capture_us: 13,
            evm_us: 100,
            hash_post_state_us: 17,
            state_root_us: 19,
            cache_update_us: 23,
            ..Default::default()
        };
        timings.recompute_totals();

        assert_eq!(timings.state_access_execution_us, 128);
        assert_eq!(timings.execution_core_us, 161);
        assert!(timings.raw_total_us > timings.state_access_execution_us);
    }

    #[test]
    fn json_schema_contains_join_keys_phases_fingerprints_and_cache_cost() {
        let value = serde_json::to_value(ValidationBenchmarkRecord::default()).unwrap();

        assert_eq!(VALIDATION_BENCHMARK_SCHEMA_VERSION, 9);
        // The storage walk's own timers start after the copy, so without this field the gap
        // between storage retention's total and its phases has no name.
        assert!(value["partial"].get("retention_storage_trie_cow_us").is_some());
        // Nested inside `cache_update_us`. Without it the anchor is not comparable across the
        // index, so a record that omits it is a record the combined gate cannot be read off.
        assert!(value["partial"].get("cache_root_index_maintenance_us").is_some());
        assert_eq!(value["schema_version"], 0);
        assert!(value.get("block_hash").is_some());
        assert!(value["partial"].get("state_access_execution_us").is_some());
        assert!(value["partial"].get("retention_account_trie_detail").is_some());
        assert!(value["partial"]["retention_account_trie_detail"]
            .get("global_prefix_lookups")
            .is_some());
        assert!(value["partial"].get("retention_storage_trie_detail").is_some());
        assert!(value["weak"].get("raw_total_us").is_some());
        assert!(value.get("expected_receipts_root").is_some());
        assert!(value.get("expected_requests_hash").is_some());
        assert!(value.get("value_cache_bytes").is_some());
        assert!(value.get("trie_cache_bytes").is_some());
        for field in ["cache_accounts", "cache_storage", "cache_codes"] {
            assert!(value.get(field).is_some(), "missing composition field {field}");
        }
        for field in [
            "accounts_added",
            "accounts_refreshed",
            "accounts_evicted",
            "storage_added",
            "storage_refreshed",
            "storage_evicted",
            "codes_added",
            "codes_refreshed",
            "codes_evicted",
        ] {
            assert!(
                value["partial"]["cache_delta"].get(field).is_some(),
                "missing cache delta field {field}"
            );
        }
        for field in [
            "account_collect_sort_us",
            "storage_collect_sort_us",
            "account_leaf_hash_us",
            "storage_leaf_hash_us",
            "account_namespace_us",
            "storage_namespace_us",
            "root_us",
            "accounts",
            "storage",
            "memo_hits",
        ] {
            assert!(
                value["partial"]["next_cache_anchor_detail"].get(field).is_some(),
                "missing anchor split field {field}"
            );
        }
        for field in [
            "account_trie_us",
            "storage_tries_us",
            "warm_membership_us",
            "retained_paths_us",
            "warm_accounts",
            "retained_account_paths",
        ] {
            assert!(
                value["partial"]["trie_clone_detail"].get(field).is_some(),
                "missing clone split field {field}"
            );
        }
        for field in [
            "nodes_us",
            "values_us",
            "masks_us",
            "buffers_us",
            "rest_us",
            "accounting_us",
            "branch_hash_probe_us",
            "total_bytes",
            "nodes_bytes",
            "branch_hash_bytes",
            "total_allocs",
            "nodes_allocs",
            "branch_hash_allocs",
            "branch_nodes",
            "node_entries",
            "value_entries",
        ] {
            assert!(
                value["partial"]["trie_clone_detail"]["account_trie_detail"].get(field).is_some(),
                "missing account copy decomposition field {field}"
            );
        }
        for field in [
            "visits_on_productive_path",
            "productive_path_us",
            "finalization_upper_roots",
            "finalization_lower_subtries_with_roots",
            "finalization_nodes_removed",
            "finalization_masks_removed",
            "finalization_masks_removed_without_node",
            "finalization_masks_us",
            "finalization_maps_us",
            "finalization_subtries_us",
        ] {
            assert!(
                value["partial"]["retention_account_trie_detail"].get(field).is_some(),
                "missing walk frontier field {field}"
            );
        }
        assert!(value["retained_generation"].get("exclusive_bytes").is_some());
        assert!(value.get("partial_witness_build_us").is_some());
        assert!(value.get("partial_serialize_us").is_some());
    }

    #[test]
    fn builder_json_schema_exposes_p0_proof_and_snapshot_costs() {
        let value = serde_json::to_value(BuilderBenchmarkRecord::default()).unwrap();

        assert!(value.get("block_hash").is_some());
        assert!(value.get("builder_total_us").is_some());
        assert!(value.get("snapshot_created").is_some());
        assert!(value.get("snapshot_us").is_some());
        assert!(value.get("initial_proof_source").is_some());
        assert!(value.get("initial_provider_us").is_some());
        assert!(value.get("distinct_storage_tries").is_some());
        assert!(value.get("parallel_storage_workers").is_some());
        assert!(value.get("parallel_account_workers").is_some());
        assert!(value.get("witness_commitment").is_some());
        assert!(value["retained_generation"].get("exclusive_bytes").is_some());
    }

    #[test]
    fn builder_json_schema_exposes_trie_clone_cost_and_mutation_footprint() {
        let value = serde_json::to_value(BuilderBenchmarkRecord::default()).unwrap();

        assert!(value.get("trie_clone_bytes").is_some());
        assert!(value.get("trie_clone_rss_delta_bytes").is_some());
        assert!(
            value.get("trie_mutation").is_some_and(serde_json::Value::is_null),
            "the footprint is absent rather than zero when diagnostics are off, so a run without \
             them cannot be read as a block that dirtied nothing"
        );
    }

    #[test]
    fn trie_mutation_summary_carries_both_dirtied_shares() {
        let metrics = TrieMutationMetrics {
            retained_account_paths: 40,
            dirtied_account_paths: 4,
            retained_storage_paths: 60,
            dirtied_storage_paths: 6,
            dirtied_storage_tries: 2,
            account_revealed_nodes: 300,
            storage_revealed_nodes: 200,
            account_prefixes: std::array::from_fn(|depth| PrefixCoverage {
                retained: depth + 1,
                dirtied: 1,
            }),
            per_storage_trie: Vec::new(),
        };

        let summary = TrieMutationSummary::from(&metrics);
        let value = serde_json::to_value(&summary).unwrap();

        assert_eq!(summary.revealed_nodes, 500);
        // 10 dirtied leaf paths out of 100 retained.
        assert!((summary.dirtied_path_share - 0.1).abs() < f64::EPSILON);
        // Deepest level retains 6 prefixes, of which 1 is dirtied.
        assert!((summary.deepest_account_prefix_share - 1.0 / 6.0).abs() < f64::EPSILON);
        assert!(value.get("dirtied_storage_tries").is_some());
    }

    #[test]
    fn witness_size_breakdown_uses_exact_serialized_payload_and_components() {
        let witness = PartialExecutionWitness {
            state: PartialExecutionWitnessState::MptTransitionNodes(vec![
                Bytes::from_static(&[1, 2]),
                Bytes::from_static(&[3, 4, 5]),
            ]),
            codes: vec![Bytes::from_static(&[6, 7, 8])],
            keys: vec![Bytes::from_static(&[9, 10, 11, 12])],
            headers: vec![Bytes::from_static(&[13, 14, 15, 16, 17])],
        };
        let sizes = WitnessSizeBreakdown::from_witness(&witness).unwrap();

        assert_eq!(sizes.serialized_witness_bytes, bincode::serialize(&witness).unwrap().len());
        assert_eq!(sizes.state_node_bytes, 5);
        assert_eq!(sizes.state_nodes, 2);
        assert_eq!(sizes.bytecode_bytes, 3);
        assert_eq!(sizes.key_preimage_bytes, 4);
        assert_eq!(sizes.ancestor_header_bytes, 5);
    }
}
