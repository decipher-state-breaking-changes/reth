use alloy_primitives::B256;
use partial_stateless::{PartialExecutionWitness, PartialStatelessSidecar, TrieMutationMetrics};
use serde::Serialize;
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
};

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
    pub trie_clone_us: u64,
    pub state_root_us: u64,
    pub root_completeness_us: u64,
    pub miss_policy_check_us: u64,
    pub cache_update_us: u64,
    pub trie_retention_us: u64,
    pub next_cache_anchor_us: u64,
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
    pub trie_cache_bytes: usize,
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
    /// Logical size of the trie the per-block snapshot copied.
    pub trie_clone_bytes: usize,
    /// Process RSS moved across the clone. Process-wide, so meaningful only in aggregate.
    pub trie_clone_rss_delta_bytes: i64,
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

        assert_eq!(value["schema_version"], 0);
        assert!(value.get("block_hash").is_some());
        assert!(value["partial"].get("state_access_execution_us").is_some());
        assert!(value["weak"].get("raw_total_us").is_some());
        assert!(value.get("expected_receipts_root").is_some());
        assert!(value.get("expected_requests_hash").is_some());
        assert!(value.get("value_cache_bytes").is_some());
        assert!(value.get("trie_cache_bytes").is_some());
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
