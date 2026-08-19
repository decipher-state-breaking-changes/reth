use alloy_primitives::B256;
use partial_stateless::{PartialExecutionWitness, PartialStatelessSidecar, TrieMutationMetrics};
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
/// `probe` to also price the branch-hash box. A 300-sample run measured them at 8.94, 0.49, and
/// 9.11 ms per block, so a default run leaves them off and reports zeroes rather than paying 4.7%
/// of raw validation for numbers that move with cache size rather than with the block.
///
/// Three fields therefore measure the instrumentation rather than the work and sit outside their
/// phase: `account_trie_detail.accounting_us`, `account_trie_detail.branch_hash_probe_us`, and each
/// walk's `productive_path_us`. All three are zero unless requested. Subtract whichever are nonzero
/// before comparing a V9 run's totals or per-entry coefficients against a V8 one.
///
/// **V10 adds admission.** Every `partial`/`weak` timing block now carries an `admission` object —
/// `source`, `input_decode_us`, `payload_validation_us`, `sender_recovery_us`,
/// `pre_execution_consensus_us` — plus a sibling `post_execution_consensus_us`. The four admission
/// timings are nullable and a null is not a zero: it means the path that produced this record was
/// handed a block that had already cleared that stage, which is what `source: "recovered"` names.
/// Every record this harness writes is `recovered`, because a run driven by a reth node is handed
/// blocks its Engine already admitted; a standalone validator reading Engine payloads reports
/// `execution_data` and fills the phases in. Reading a null as zero would report a validator that
/// skips admission as one that admits for free.
///
/// V10 changes no V9 field and moves no work between existing phases, so a V9 run and a V10 run
/// from an ExEx are directly comparable: the new phases are null or, for
/// `post_execution_consensus_us`, the small cost the delegated post-execution check adds.
pub const VALIDATION_BENCHMARK_SCHEMA_VERSION: u64 = 10;

/// Phase instrumentation, produced by the validator core rather than by this harness.
///
/// The core has to time its own phases, but the record schema below is the harness's format
/// rather than the validator's, which is why this is the only type that crosses the boundary.
/// Its nested metric structs come with it, so the published schema is byte-identical across
/// the extraction: an analyzer that parsed a V9 record before still parses one now.
pub use partial_stateless_validator::{
    coordination::RetainedGenerationBytes, timings::ValidationPhaseTimings,
};

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
            partial_stateless::PartialExecutionWitnessState::MptTrimmedTransitionNodes {
                nodes,
                ..
            } => (nodes.iter().map(|node| node.len()).sum(), nodes.len()),
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

/// Schema of [`BuilderBenchmarkRecord`].
///
/// 4 declares the fields that make an engine-access run readable: `artifact_available`,
/// `shadow_sampled`, and `fallback_reason`. Schema 3 as specified carried only `artifact_reused`,
/// which cannot answer what these files are read for -- a reused block looks the same either way,
/// but a delivered-and-not-reused block is indistinguishable from a miss, and a delivery rate
/// computed over 3 is not a low delivery rate, it is an absent field printed as zero.
///
/// The three fields were added without raising this constant, so files exist that declare 3 and
/// carry all of 4 — everything written during the engine-access artifact runs of 2026-08-10/11.
/// They are correct; only the label is stale. `analyze_builder_bench.py` therefore keys on field
/// presence and reports the declared version rather than trusting it, and this bump only fixes
/// files written from here on.
pub const BUILDER_BENCHMARK_SCHEMA_VERSION: u64 = 4;

/// Per-block builder telemetry used to isolate cache snapshot and initial proof costs.
#[derive(Debug, Clone, Default, Serialize)]
pub struct BuilderBenchmarkRecord {
    pub schema_version: u64,
    pub block_number: u64,
    pub block_hash: B256,
    pub historical_full_db_evm_us: u64,
    /// Whether the handoff delivered an artifact for this block. **This is the delivery rate.**
    ///
    /// Distinct from `artifact_reused`, and the distinction is not pedantic: a sampled block is
    /// delivered and deliberately not reused, so at the default 1-in-50 sampling a perfectly
    /// healthy handoff still reports only ~98% reuse. Read delivery here and the win there.
    ///
    /// Delivery, not usability: a `type_mismatch` block sets this and still re-executes. Since
    /// schema 4 (never in 3, which has no such field).
    pub artifact_available: bool,
    /// Whether the artifact actually replaced this block's re-execution.
    ///
    /// When true `historical_full_db_evm_us` is zero because no EVM ran here, so the two fields
    /// must be read together: a median over blocks that mixes both paths measures neither.
    pub artifact_reused: bool,
    /// Whether this block re-executed on purpose to feed the differential comparison.
    pub shadow_sampled: bool,
    /// Why the artifact was not reused, or `None` when it was.
    ///
    /// `capture_off`, `shadow_mode`, `shadow_sampled`, `type_mismatch`, `not_published`,
    /// `evicted_capacity`, `evicted_bytes`, `dropped_contended`.
    ///
    /// `not_published` is the residual, not a claim about the producer: the handoff's tombstone
    /// rings are bounded, so a cause that aged out arrives here. Since schema 4.
    pub fallback_reason: Option<&'static str>,
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
    use partial_stateless_validator::timings::RetentionWalkMetrics;

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

        assert_eq!(VALIDATION_BENCHMARK_SCHEMA_VERSION, 10);
        // Admission is nullable, and the distinction is the point: `null` means the stage was
        // already cleared before this validator saw the block, `0` means it ran and cost nothing.
        // A record that could not express the difference would make an ExEx run look like a
        // standalone one that admits for free.
        assert_eq!(value["partial"]["admission"]["source"], "recovered");
        assert!(value["partial"]["admission"]["sender_recovery_us"].is_null());
        assert!(value["partial"]["admission"].get("payload_validation_us").is_some());
        assert!(value["partial"].get("post_execution_consensus_us").is_some());
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
    fn the_builder_schema_version_covers_every_artifact_field_an_analyzer_needs() {
        // These four fields arrived together and are read together. A file that carries some of
        // them under a version that promises none is the failure this guards: the analyzer would
        // find `artifact_available` absent and report 0% delivery rather than "not recorded".
        let value = serde_json::to_value(BuilderBenchmarkRecord::default()).unwrap();

        assert_eq!(BUILDER_BENCHMARK_SCHEMA_VERSION, 4);
        for field in ["artifact_available", "artifact_reused", "shadow_sampled", "fallback_reason"]
        {
            assert!(value.get(field).is_some(), "schema 4 must carry {field}");
        }
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
