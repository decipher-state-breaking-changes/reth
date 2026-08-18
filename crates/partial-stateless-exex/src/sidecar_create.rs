use crate::{
    access_shadow::{
        record_shadow_comparison, shadow_sample_selects, simulation_from_artifact,
        take_engine_access, ArtifactDisposition, EngineAccessTake,
    },
    benchmark::{
        append_builder_record, append_record, deserialize_sidecar_for_benchmark,
        serialize_sidecar_for_benchmark, BuilderBenchmarkRecord, RetainedGenerationBytes,
        TrieMutationSummary, ValidationBenchmarkRecord, WitnessSizeBreakdown,
        BUILDER_BENCHMARK_SCHEMA_VERSION, VALIDATION_BENCHMARK_SCHEMA_VERSION,
    },
    format_bytes,
    policy_dataset_capture::PolicyDatasetMaterial,
    process_rss_bytes, process_rusage,
    rebuild::{simulate_block, HistoricalSimulation},
    sidecar_io::sidecar_path,
    sidecar_reexec::{verify_and_apply_provider_assisted_sidecar, SidecarReexecLimits},
    CacheConfig,
};
use alloy_consensus::{proofs::calculate_receipt_root, TxReceipt};
use alloy_primitives::{keccak256, map::B256Map, Bloom, Bytes, B256};
use partial_stateless::{
    accessed_state::BlockAccessedState,
    assemble_sidecar, build_cache_aware_flat_transition, build_full_witness_sidecar,
    cow_copies_taken,
    fixture::{save_fixture, AccessedStateFixture},
    generate_cache_aware_base_proof, last_n_blocks_cache_policy_id,
    measure_transition_witness_size,
    network_cache::{NetworkStateCache, UpdateStats},
    policy::LastNBlocksPolicy,
    policy_dataset::RecordedAccessProvenance,
    readiness::ReadyParent,
    witness::{
        accessed_to_state_targets, build_sidecar_targets, cache_hit_targets,
        measure_multiproof_size, state_targets_to_proof_targets, WitnessResult,
    },
    CacheAnchor, CacheAwareFlatBuild, CacheFootprintStats, FullWitnessBuild, ParallelProof,
    PartialExecutionWitnessState, PartialStatelessSidecar, PartialTrieNodeCache,
    RootWitnessCompletenessSummary, SidecarAssembly, SidecarBenchmarkManifest, StateTargetStats,
    TransitionBuildContext, TransitionProofSource, TrieChangeSet, TrieProofTargetV2, V2TargetSet,
    WitnessReductionStats,
};
use partial_stateless_validator::{
    verify_and_apply_sidecar, TimedValidation, TrieCacheDisposition, ValidatorRules,
};
use reth_consensus::FullConsensus;
use reth_ethereum::EthPrimitives;
use reth_evm::ConfigureEvm;
use reth_execution_access::{AccessCaptureMode, MissReason};
use reth_primitives_traits::{AlloyBlockHeader, BlockTy, RecoveredBlock};
use reth_provider::{ProviderResult, StateProvider};
use reth_trie_common::{DecodedMultiProofV2, HashedPostState, MultiProofTargetsV2, TrieInput};
use std::{
    fs, mem,
    path::{Path, PathBuf},
    time::Instant,
};
use tracing::{debug, error, info, warn};

pub(crate) struct BuilderOptions<'a> {
    pub(crate) capture_dir: Option<&'a Path>,
    /// Whether this block also builds, validates, and hands back a policy-neutral full witness.
    ///
    /// Off in every ordinary run. On, it costs a second parent-state multiproof over *every*
    /// accessed key and mutation path, a database-free re-execution of the block against it, and
    /// the memory to hold the result — which is why the capture contract refuses to share a
    /// process with anything that measures.
    pub(crate) capture_policy_dataset: bool,
    pub(crate) sidecar_dir: &'a Path,
    pub(crate) compute_baseline: bool,
    pub(crate) resource_metrics: bool,
    pub(crate) trie_cache_diagnostics: bool,
    pub(crate) run_sidecar_preflight: bool,
    pub(crate) validation_bench_output: Option<&'a Path>,
    pub(crate) builder_bench_output: Option<&'a Path>,
    pub(crate) force_previous_cache_snapshot: bool,
    /// What the K = 1 retained generation held when this block began.
    pub(crate) retained_generation: RetainedGenerationBytes,
    pub(crate) reexec_limits: &'a SidecarReexecLimits,
    pub(crate) parallel_initial_proof: Option<&'a ParallelInitialProofFn<'a>>,
    /// The parent the caches are authenticated against, when they are Ready.
    ///
    /// Publication requires it. A Warming cache produces an arithmetically correct sidecar while
    /// holding only part of the LastN window its policy identifier advertises, so a peer that
    /// accepted it would be trusting a window that was never replayed. Building and measuring the
    /// sidecar still happens while Warming — that is what the benchmark needs — only the write to
    /// the shared sidecar directory is withheld.
    pub(crate) ready_parent: Option<&'a ReadyParent>,
    /// Whether the constructed sidecar is returned to the caller.
    ///
    /// Only the in-process bootstrap gate needs it, and it pays a clone of the whole witness, so
    /// the ordinary builder leaves it off.
    pub(crate) retain_sidecar: bool,
}

#[derive(Debug)]
pub(crate) struct ParallelInitialProofOutput {
    pub(crate) proof: DecodedMultiProofV2,
    pub(crate) storage_workers: usize,
    pub(crate) account_workers: usize,
}

pub(crate) type ParallelInitialProofFn<'a> =
    dyn Fn(MultiProofTargetsV2) -> ProviderResult<ParallelInitialProofOutput> + 'a;

#[derive(Debug)]
pub(crate) struct BuilderBlockReport {
    pub(crate) cache_update: UpdateStats,
    pub(crate) witness: Option<WitnessResult>,
    pub(crate) sidecar_path: Option<PathBuf>,
    /// The sidecar as built, when `BuilderOptions::retain_sidecar` asked for it.
    pub(crate) sidecar: Option<PartialStatelessSidecar>,
    /// The parent trie generation this block's transition displaced, when the transition
    /// committed.
    ///
    /// Handing it back rather than dropping it is what makes a one-deep retained generation
    /// free: the transition already copies the parent into `next_trie_cache` and then
    /// overwrites the original, so the object exists either way and the only question is
    /// whether anything still holds it. `None` means the transition did not commit and the
    /// caller's trie cache is still the parent.
    pub(crate) displaced_trie_cache: Option<PartialTrieNodeCache>,
    /// The policy-neutral capture material, when [`BuilderOptions::capture_policy_dataset`] asked
    /// for it and the full witness proved itself.
    pub(crate) policy_dataset_material: Option<PolicyDatasetMaterial>,
}

const fn needs_previous_cache_snapshot(
    run_sidecar_preflight: bool,
    validation_bench_enabled: bool,
    force_snapshot: bool,
) -> bool {
    run_sidecar_preflight || validation_bench_enabled || force_snapshot
}

fn previous_cache_snapshot(
    cache: &NetworkStateCache,
    config: &CacheConfig,
    cache_parent_synced: bool,
    run_sidecar_preflight: bool,
    validation_bench_enabled: bool,
    force_snapshot: bool,
) -> Option<NetworkStateCache> {
    (cache_parent_synced &&
        needs_previous_cache_snapshot(
            run_sidecar_preflight,
            validation_bench_enabled,
            force_snapshot,
        ))
    .then(|| {
        cache.fork_for_reexecution(
            Box::new(LastNBlocksPolicy::new(config.account_window)),
            Box::new(LastNBlocksPolicy::new(config.storage_window)),
        )
    })
}

/// Decides whether a built sidecar may be written for peers to accept.
///
/// Both anchors must exist, and the readiness tracker must independently agree that the cache was
/// authenticated against exactly the parent the sidecar claims. The two are derived separately —
/// one from the cache at build time, one from the readiness state machine — so requiring them to
/// agree catches a cache that drifted from what the tracker believes about it.
fn sidecar_publication_anchors(
    prev_cache_anchor: Option<CacheAnchor>,
    next_cache_anchor: Option<CacheAnchor>,
    ready_parent: Option<&ReadyParent>,
) -> Option<(CacheAnchor, CacheAnchor)> {
    let (prev, next) = prev_cache_anchor.zip(next_cache_anchor)?;
    (ready_parent?.anchor == prev).then_some((prev, next))
}

fn require_previous_cache_snapshot(
    snapshot: Option<&mut NetworkStateCache>,
) -> eyre::Result<&mut NetworkStateCache> {
    snapshot
        .ok_or_else(|| eyre::eyre!("previous cache snapshot missing for builder-side preflight"))
}

fn rollback_sidecar_transition(
    cache: &mut NetworkStateCache,
    block_number: u64,
    cause: eyre::Report,
) -> eyre::Report {
    match cache.rollback_block(block_number) {
        Ok(()) => eyre::eyre!(
            "sidecar generation failed; cache transition for block {block_number} was rolled back: {cause:#}"
        ),
        Err(rollback_err) => eyre::eyre!(
            "sidecar generation failed for block {block_number} ({cause:#}); cache rollback also failed: {rollback_err}"
        ),
    }
}

fn persist_preflight_failure_artifacts(
    sidecar_dir: &Path,
    state_provider: &dyn StateProvider,
    sidecar: &PartialStatelessSidecar,
    expected_state_root: B256,
    parent_cache: &NetworkStateCache,
    parent_trie_cache: &PartialTrieNodeCache,
    error: &eyre::Report,
) -> eyre::Result<PathBuf> {
    let failure_dir = sidecar_dir
        .join("preflight-failures")
        .join(format!("{}_{:?}", sidecar.block_number, sidecar.block_hash));
    fs::create_dir_all(&failure_dir)?;

    let sidecar_bytes = bincode::serialize(sidecar)?;
    fs::write(failure_dir.join("sidecar.bin"), &sidecar_bytes)?;
    let value_cache_bytes = bincode::serialize(&(
        1u64,
        parent_cache.current_block(),
        parent_cache.accounts(),
        parent_cache.storage(),
        parent_cache.codes(),
    ))?;
    fs::write(failure_dir.join("parent-value-cache.bin"), &value_cache_bytes)?;

    let mut parent_targets = V2TargetSet::default();
    for address in parent_cache.accounts().keys() {
        parent_targets.insert(TrieProofTargetV2::Account { key: keccak256(address), min_len: 0 });
    }
    for (address, slot) in parent_cache.storage().keys() {
        let hashed_address = keccak256(address);
        parent_targets.insert(TrieProofTargetV2::Account { key: hashed_address, min_len: 0 });
        parent_targets.insert(TrieProofTargetV2::Storage {
            hashed_address,
            key: keccak256(slot),
            min_len: 0,
        });
    }
    if parent_targets.is_empty() {
        parent_targets.insert(TrieProofTargetV2::Account { key: B256::ZERO, min_len: 0 });
    }

    let mut parent_witness_nodes = 0usize;
    let mut parent_witness_bytes = 0usize;
    let mut parent_witness_error = None;
    match state_provider.multiproof_v2(TrieInput::default(), parent_targets.to_provider_targets()) {
        Ok(proof) => {
            let mut flat_nodes = B256Map::<Bytes>::default();
            proof.extend_flat_witness(&mut flat_nodes);
            let mut nodes = flat_nodes.into_values().collect::<Vec<_>>();
            nodes.sort_unstable();
            nodes.dedup();
            parent_witness_nodes = nodes.len();
            parent_witness_bytes = nodes.iter().map(|node| node.len()).sum();
            fs::write(failure_dir.join("parent-trie-witness.bin"), bincode::serialize(&nodes)?)?;
        }
        Err(proof_error) => parent_witness_error = Some(proof_error.to_string()),
    }

    let shape = parent_trie_cache.shape_metrics();
    let metadata = serde_json::json!({
        "schema_version": 1,
        "block_number": sidecar.block_number,
        "block_hash": format!("{:?}", sidecar.block_hash),
        "parent_hash": format!("{:?}", sidecar.parent_hash),
        "parent_state_root": format!("{:?}", sidecar.parent_state_root),
        "expected_state_root": format!("{:?}", expected_state_root),
        "preflight_error": format!("{error:#}"),
        "sidecar_bytes": sidecar_bytes.len(),
        "parent_value_cache_bytes": value_cache_bytes.len(),
        "parent_cache_accounts": parent_cache.accounts().len(),
        "parent_cache_storage": parent_cache.storage().len(),
        "parent_cache_codes": parent_cache.codes().len(),
        "parent_trie_state_root": parent_trie_cache.state_root().map(|root| format!("{root:?}")),
        "parent_trie_warm_accounts": shape.retained_account_paths,
        "parent_trie_warm_storage": shape.retained_storage_paths,
        "parent_trie_account_nodes": shape.account_revealed_nodes,
        "parent_trie_storage_nodes": shape.storage_revealed_nodes,
        "parent_trie_witness_targets": parent_targets.len(),
        "parent_trie_witness_nodes": parent_witness_nodes,
        "parent_trie_witness_bytes": parent_witness_bytes,
        "parent_trie_witness_error": parent_witness_error,
        "files": {
            "sidecar": "sidecar.bin",
            "parent_value_cache": "parent-value-cache.bin",
            "parent_trie_witness": (parent_witness_nodes > 0).then_some("parent-trie-witness.bin"),
        },
    });
    fs::write(failure_dir.join("metadata.json"), serde_json::to_vec_pretty(&metadata)?)?;
    Ok(failure_dir)
}

/// The node's state database, as a [`TransitionProofSource`].
///
/// Bound to one parent state: `state_provider` was opened at this block's parent hash, which is
/// what lets the trait's method carry no root of its own. The wide path is threaded through
/// unchanged — the eligibility rule, the fallback, and the worker counts all belong to the shared
/// core, so the only thing adapted here is the error type.
pub(crate) struct RethStateProviderSource<'a> {
    state_provider: &'a dyn StateProvider,
    parallel: Option<Box<dyn Fn(MultiProofTargetsV2) -> eyre::Result<ParallelProof> + 'a>>,
}

impl<'a> RethStateProviderSource<'a> {
    pub(crate) fn new(
        state_provider: &'a dyn StateProvider,
        parallel_initial_proof: Option<&'a ParallelInitialProofFn<'a>>,
    ) -> Self {
        let parallel = parallel_initial_proof.map(|proof_fn| {
            Box::new(move |targets: MultiProofTargetsV2| {
                proof_fn(targets)
                    .map(|output| ParallelProof {
                        proof: output.proof,
                        storage_workers: output.storage_workers,
                        account_workers: output.account_workers,
                    })
                    .map_err(|err| eyre::eyre!("{err}"))
            }) as Box<dyn Fn(MultiProofTargetsV2) -> eyre::Result<ParallelProof> + 'a>
        });
        Self { state_provider, parallel }
    }

    /// The build context a live builder uses: this source, sampling the process it runs in.
    pub(crate) fn context(&self) -> TransitionBuildContext<'_> {
        TransitionBuildContext { proofs: self, rss_sampler: Some(process_rss_bytes) }
    }
}

impl TransitionProofSource for RethStateProviderSource<'_> {
    fn multiproof_v2(&self, targets: MultiProofTargetsV2) -> eyre::Result<DecodedMultiProofV2> {
        self.state_provider
            .multiproof_v2(TrieInput::default(), targets)
            .map_err(|err| eyre::eyre!("{err}"))
    }

    fn parallel_initial_proof(
        &self,
    ) -> Option<&dyn Fn(MultiProofTargetsV2) -> eyre::Result<ParallelProof>> {
        self.parallel.as_deref()
    }
}

#[derive(Debug)]
pub struct WeakStatelessBuild {
    pub sidecar: PartialStatelessSidecar,
    pub build_us: u64,
}

/// Builds the Weak-stateless sidecar for one block against the node's database.
///
/// A thin binding of the shared policy-neutral full-witness build to a provider-backed proof
/// source. There is no second implementation of what a full witness contains.
#[expect(clippy::too_many_arguments)]
pub fn build_weak_stateless_sidecar(
    state_provider: &dyn StateProvider,
    parent_state_root: B256,
    expected_state_root: B256,
    parent_hash: B256,
    block_hash: B256,
    block_number: u64,
    hashed_post_state: &HashedPostState,
    accessed: &BlockAccessedState,
    ancestor_headers: &[Bytes],
    config: &CacheConfig,
) -> eyre::Result<WeakStatelessBuild> {
    let source = RethStateProviderSource::new(state_provider, None);
    let FullWitnessBuild { sidecar, build_us } = build_full_witness_sidecar(
        &source.context(),
        parent_state_root,
        expected_state_root,
        parent_hash,
        block_hash,
        block_number,
        hashed_post_state,
        accessed,
        ancestor_headers,
        config,
    )?;
    Ok(WeakStatelessBuild { sidecar, build_us })
}
fn benchmark_one_sidecar_validation<Evm, Consensus>(
    rules: ValidatorRules<'_, Evm, Consensus>,
    block: &RecoveredBlock<BlockTy<EthPrimitives>>,
    cache: &mut NetworkStateCache,
    trie_cache: &mut PartialTrieNodeCache,
    sidecar_bytes: &[u8],
    expected_cache_policy_id: B256,
    limits: &SidecarReexecLimits,
) -> eyre::Result<TimedValidation>
where
    Evm: ConfigureEvm<Primitives = EthPrimitives>,
    Consensus: FullConsensus<EthPrimitives> + ?Sized,
{
    let (sidecar, deserialize_us) = deserialize_sidecar_for_benchmark(sidecar_bytes)?;
    let mut report = verify_and_apply_sidecar(
        rules,
        block,
        cache,
        &sidecar,
        expected_cache_policy_id,
        limits,
        trie_cache,
        TrieCacheDisposition::Discard,
    )?;
    report.timings.set_deserialize_us(deserialize_us);
    Ok(report)
}

const fn partial_runs_first(block_number: u64) -> bool {
    block_number % 2 == 0
}

/// Whether the Weak witness is *built* before the Partial one for this block.
///
/// The two sidecars are proved against the same state, so whichever is built first pays the cold
/// page-cache read and the second reuses the pages the first warmed. Left fixed, that is a
/// systematic bias worth hundreds of milliseconds: Weak proves a strictly larger target set than
/// Partial — every accessed key against a cold cache, rather than only the miss set — yet measured
/// *faster* purely because it always ran second.
///
/// This alternates on bit 1 while [`partial_runs_first`] alternates on bit 0, so build order and
/// validation order are independent and all four combinations occur equally often. Using the same
/// bit for both would leave them perfectly correlated and simply trade one fixed bias for another.
const fn weak_builds_first(block_number: u64) -> bool {
    (block_number >> 1) % 2 == 1
}

/// The consensus-visible outputs of the full-DB execution of one block.
///
/// Bundled so the differential has one value to compare rather than four loose arguments, and so
/// the canonical verdict can travel with the numbers it was computed from. Partial and Weak cannot
/// reach the differential disagreeing with the header — the validator core rejects them before it
/// returns — which makes this the only side of the three-way comparison still able to fail there.
#[derive(Clone, Copy)]
struct HistoricalOutputs {
    gas_used: u64,
    receipts_root: B256,
    requests_hash: B256,
    requests_empty: bool,
    /// Whether the full-DB output passed the same canonical post-execution validation the two
    /// stateless paths passed, logs bloom and fork gating included.
    ///
    /// Recorded rather than raised: the differential record exists to carry exactly this
    /// observation, and aborting on it would discard the record instead of writing it.
    consensus_ok: bool,
}

#[expect(clippy::too_many_arguments)]
fn benchmark_sidecar_validation<Evm, Consensus>(
    rules: ValidatorRules<'_, Evm, Consensus>,
    state_provider: &dyn StateProvider,
    block: &RecoveredBlock<BlockTy<EthPrimitives>>,
    prev_cache: &mut NetworkStateCache,
    trie_cache: &mut PartialTrieNodeCache,
    partial_sidecar: &PartialStatelessSidecar,
    prebuilt_weak: Option<WeakStatelessBuild>,
    hashed_post_state: &HashedPostState,
    accessed: &BlockAccessedState,
    ancestor_headers: &[Bytes],
    config: &CacheConfig,
    limits: &SidecarReexecLimits,
    output_path: &Path,
    historical_full_db_evm_us: u64,
    partial_witness_build_us: u64,
    historical: HistoricalOutputs,
    value_cache_bytes: usize,
    retained_generation: RetainedGenerationBytes,
) -> eyre::Result<TimedValidation>
where
    Evm: ConfigureEvm<Primitives = EthPrimitives>,
    Consensus: FullConsensus<EthPrimitives> + ?Sized,
{
    // Already built if this block's turn was Weak-first; otherwise build it now, after Partial.
    let WeakStatelessBuild { sidecar: weak_sidecar, build_us: weak_build_us } = match prebuilt_weak
    {
        Some(prebuilt) => prebuilt,
        None => build_weak_stateless_sidecar(
            state_provider,
            partial_sidecar.parent_state_root,
            block.state_root(),
            block.parent_hash,
            block.hash(),
            block.number(),
            hashed_post_state,
            accessed,
            ancestor_headers,
            config,
        )?,
    };

    // Serialization is builder-side preparation and is outside both execution timers. Each
    // sidecar is deserialized immediately before its execution so decode order follows EVM order.
    let partial_serialize_start = Instant::now();
    let partial_bytes = serialize_sidecar_for_benchmark(partial_sidecar)?;
    let partial_serialize_us = partial_serialize_start.elapsed().as_micros() as u64;
    let weak_serialize_start = Instant::now();
    let weak_bytes = serialize_sidecar_for_benchmark(&weak_sidecar)?;
    let weak_serialize_us = weak_serialize_start.elapsed().as_micros() as u64;
    let partial_witness = WitnessSizeBreakdown::from_witness(&partial_sidecar.witness)?;
    let weak_witness = WitnessSizeBreakdown::from_witness(&weak_sidecar.witness)?;

    let mut weak_cache = config.new_cache_at(block.number().saturating_sub(1));
    let mut weak_trie = PartialTrieNodeCache::new();
    let expected_cache_policy_id =
        last_n_blocks_cache_policy_id(config.account_window, config.storage_window);
    let partial_first = partial_runs_first(block.number());
    let weak_first_build = weak_builds_first(block.number());
    let (partial_report, weak_report) = if partial_first {
        let partial_report = benchmark_one_sidecar_validation(
            rules,
            block,
            prev_cache,
            trie_cache,
            &partial_bytes,
            expected_cache_policy_id,
            limits,
        )?;
        let weak_report = benchmark_one_sidecar_validation(
            rules,
            block,
            &mut weak_cache,
            &mut weak_trie,
            &weak_bytes,
            expected_cache_policy_id,
            limits,
        )?;
        (partial_report, weak_report)
    } else {
        let weak_report = benchmark_one_sidecar_validation(
            rules,
            block,
            &mut weak_cache,
            &mut weak_trie,
            &weak_bytes,
            expected_cache_policy_id,
            limits,
        )?;
        let partial_report = benchmark_one_sidecar_validation(
            rules,
            block,
            prev_cache,
            trie_cache,
            &partial_bytes,
            expected_cache_policy_id,
            limits,
        )?;
        (partial_report, weak_report)
    };
    info!(
        target: "partial_stateless_bench",
        block = block.number(),
        block_hash = ?block.hash(),
        verifier_order = if partial_first { "partial-then-weak" } else { "weak-then-partial" },
        "Paired Partial/Weak timed validation complete"
    );
    let expected_root = block.state_root();
    let partial_root = partial_report.outcome.state_root;
    let weak_root = weak_report.outcome.state_root;
    let expected_gas_used = block.header().gas_used();
    let expected_receipts_root = block.header().receipts_root();
    let expected_requests_hash = block.header().requests_hash();
    let requests_valid = expected_requests_hash.map_or_else(
        || {
            historical.requests_empty &&
                partial_report.outcome.execution_requests_empty &&
                weak_report.outcome.execution_requests_empty
        },
        |expected| {
            historical.requests_hash == expected &&
                partial_report.outcome.execution_requests_hash == expected &&
                weak_report.outcome.execution_requests_hash == expected
        },
    );
    // The Partial and Weak clauses below are retained even though the validator core now rejects a
    // disagreeing block before returning one: they cost nothing, and reading the record should not
    // require knowing which comparisons are load-bearing this month. `consensus_ok` is the clause
    // that carries logs bloom and fork gating, which no equality here covers.
    let valid = historical.consensus_ok &&
        partial_root == expected_root &&
        weak_root == expected_root &&
        historical.gas_used == expected_gas_used &&
        partial_report.outcome.execution_gas_used == expected_gas_used &&
        weak_report.outcome.execution_gas_used == expected_gas_used &&
        historical.receipts_root == expected_receipts_root &&
        partial_report.outcome.execution_receipts_root == expected_receipts_root &&
        weak_report.outcome.execution_receipts_root == expected_receipts_root &&
        requests_valid;
    let record = ValidationBenchmarkRecord {
        schema_version: VALIDATION_BENCHMARK_SCHEMA_VERSION,
        block_number: block.number(),
        block_hash: block.hash(),
        gas_used: expected_gas_used,
        historical_gas_used: historical.gas_used,
        tx_count: block.transaction_count(),
        verifier_order: if partial_first { "partial-then-weak" } else { "weak-then-partial" },
        builder_order: if weak_first_build { "weak-then-partial" } else { "partial-then-weak" },
        historical_full_db_evm_us,
        partial_witness_build_us,
        weak_witness_build_us: weak_build_us,
        partial_serialize_us,
        weak_serialize_us,
        partial: partial_report.timings.clone(),
        weak: weak_report.timings.clone(),
        partial_witness,
        weak_witness,
        partial_sidecar_bytes: partial_bytes.len(),
        weak_sidecar_bytes: weak_bytes.len(),
        value_cache_bytes,
        cache_accounts: partial_report.timings.next_cache_anchor_detail.accounts,
        cache_storage: partial_report.timings.next_cache_anchor_detail.storage,
        cache_codes: partial_report.timings.next_cache_anchor_detail.codes,
        trie_cache_bytes: trie_cache.estimated_memory_bytes(),
        retained_generation,
        expected_state_root: expected_root,
        partial_state_root: partial_root,
        weak_state_root: weak_root,
        expected_receipts_root,
        historical_receipts_root: historical.receipts_root,
        partial_receipts_root: partial_report.outcome.execution_receipts_root,
        weak_receipts_root: weak_report.outcome.execution_receipts_root,
        expected_requests_hash,
        historical_requests_hash: historical.requests_hash,
        partial_requests_hash: partial_report.outcome.execution_requests_hash,
        weak_requests_hash: weak_report.outcome.execution_requests_hash,
        valid,
    };
    append_record(output_path, &record)?;
    info!(
        target: "partial_stateless_bench",
        benchmark = "paired_validation",
        block = block.number(),
        block_hash = ?block.hash(),
        verifier_order = record.verifier_order,
        builder_order = record.builder_order,
        historical_full_db_evm_us,
        partial_state_access_execution_us = partial_report.timings.state_access_execution_us,
        weak_state_access_execution_us = weak_report.timings.state_access_execution_us,
        partial_evm_us = partial_report.timings.evm_us,
        weak_evm_us = weak_report.timings.evm_us,
        partial_witness_bytes = record.partial_witness.serialized_witness_bytes,
        weak_witness_bytes = record.weak_witness.serialized_witness_bytes,
        weak_witness_build_us = weak_build_us,
        valid,
        "Recorded paired Partial/Weak validation benchmark"
    );
    if !valid {
        return Err(eyre::eyre!(
            "paired validation mismatch: expected_root={expected_root:?}, partial_root={partial_root:?}, weak_root={weak_root:?}, expected_gas={expected_gas_used}, partial_gas={}, weak_gas={}",
            partial_report.outcome.execution_gas_used,
            weak_report.outcome.execution_gas_used,
        ));
    }

    Ok(partial_report)
}
pub fn create_sidecar_for_block<Evm, Consensus, ParentStateRootFn, AncestorHeadersFn>(
    rules: ValidatorRules<'_, Evm, Consensus>,
    state_provider: &dyn StateProvider,
    block: &RecoveredBlock<BlockTy<EthPrimitives>>,
    cache: &mut NetworkStateCache,
    trie_cache: &mut PartialTrieNodeCache,
    config: &CacheConfig,
    options: BuilderOptions<'_>,
    parent_state_root_by_hash: ParentStateRootFn,
    ancestor_headers_for_range: AncestorHeadersFn,
) -> eyre::Result<BuilderBlockReport>
where
    Evm: ConfigureEvm<Primitives = EthPrimitives>,
    Consensus: FullConsensus<EthPrimitives> + ?Sized,
    ParentStateRootFn: FnOnce(B256) -> eyre::Result<B256>,
    AncestorHeadersFn: FnOnce(Option<u64>, u64) -> eyre::Result<Vec<Bytes>>,
{
    let block_number = block.number();
    let builder_total_start = Instant::now();
    let parent_block_number = block_number.saturating_sub(1);

    // Taken before the re-execution below, not after, so the recorded residence is the wait a
    // consumer that skipped re-execution would actually see. `None` unless capture is enabled.
    let engine_access = take_engine_access(block.hash());

    // Stage 4: in `on` mode the Engine's artifact replaces the re-execution entirely, except on
    // sampled blocks, which re-execute deliberately so the differential oracle outlives stage 3.
    // Anything else -- shadow mode, a miss, an artifact whose output will not downcast -- falls
    // back to executing the block here, which is always correct and merely slower.
    let sampled = shadow_sample_selects(block_number);
    let authoritative = AccessCaptureMode::current().is_authoritative();
    let reuse_artifact = authoritative && !sampled;

    let (reused, engine_access, disposition) = match engine_access {
        None => (None, None, ArtifactDisposition::CaptureOff),
        Some(EngineAccessTake { artifact, miss_reason, stats }) if reuse_artifact => {
            match artifact {
                None => (
                    None,
                    // Keep the take so the miss still reaches the record: stage 4's gate is stated
                    // per miss, and one that wrote nothing would be indistinguishable from a block
                    // that never ran.
                    Some(EngineAccessTake { artifact: None, miss_reason, stats }),
                    ArtifactDisposition::Missed(miss_reason.unwrap_or(MissReason::NotPublished)),
                ),
                Some(artifact) => match simulation_from_artifact(artifact) {
                    Some(simulation) => (Some(simulation), None, ArtifactDisposition::Reused),
                    None => (None, None, ArtifactDisposition::TypeMismatch),
                },
            }
        }
        // Shadow mode, or a sampled block: keep the take for the comparison below.
        Some(take) => {
            let disposition = match (&take.artifact, take.miss_reason) {
                (Some(_), _) if sampled && authoritative => ArtifactDisposition::Sampled,
                (Some(_), _) => ArtifactDisposition::Shadowed,
                (None, reason) => {
                    ArtifactDisposition::Missed(reason.unwrap_or(MissReason::NotPublished))
                }
            };
            (None, Some(take), disposition)
        }
    };

    // Taken while the disposition is still a local: the dataset record has to say whether its
    // access set came from the Engine's artifact or from this builder re-executing, and the two
    // are not interchangeable evidence about what production runs.
    let artifact_reused = disposition.artifact_reused();

    // Shared with the canonical rebuild, which has to replay exactly what this applies: the cache
    // is a function of *accessed* state, so an execution diff would miss read-only accounts, code
    // reads, and reads made by calls that later reverted.
    let HistoricalSimulation {
        accessed,
        lowest_block_number,
        output: execution_output,
        elapsed_us: historical_full_db_evm_us,
    } = match reused {
        Some(simulation) => simulation,
        None => simulate_block(rules.evm_config(), state_provider, block)?,
    };

    // Only reached when this block re-executed: in shadow mode always, in `on` mode on the
    // sampled fraction. The comparison changes nothing about what this block produces.
    if let Some(take) = engine_access {
        record_shadow_comparison(
            block_number,
            block.hash(),
            block.parent_hash,
            take,
            &accessed,
            lowest_block_number,
            historical_full_db_evm_us,
        );
    }
    // The same canonical post-execution validation the two stateless paths run, applied to the
    // full-DB output so the differential compares three consensus verdicts rather than three
    // subsets of one. Only this side can still fail it: the validator core rejects a disagreeing
    // Partial or Weak block before returning. On the artifact-reuse path the output is the
    // Engine's own,
    // which already passed this — redundant there, and the only check of the historical execution
    // on every other block.
    let historical_receipts_with_bloom =
        execution_output.result.receipts.iter().map(TxReceipt::with_bloom_ref).collect::<Vec<_>>();
    let historical_receipts_root = calculate_receipt_root(&historical_receipts_with_bloom);
    let historical_logs_bloom = historical_receipts_with_bloom
        .iter()
        .fold(Bloom::ZERO, |bloom, receipt| bloom | receipt.bloom_ref());
    drop(historical_receipts_with_bloom);
    let historical_consensus_ok = match rules.consensus().validate_block_post_execution(
        block,
        &execution_output.result,
        Some((historical_receipts_root, historical_logs_bloom)),
    ) {
        Ok(()) => true,
        Err(err) => {
            error!(
                target: "partial_stateless",
                block = block_number,
                %err,
                "Full-DB execution disagrees with the block's own header"
            );
            false
        }
    };
    let historical = HistoricalOutputs {
        gas_used: execution_output.result.gas_used,
        receipts_root: historical_receipts_root,
        requests_hash: execution_output.result.requests.requests_hash(),
        requests_empty: execution_output.result.requests.is_empty(),
        consensus_ok: historical_consensus_ok,
    };

    let hashed_post_state = state_provider.hashed_post_state(&execution_output.state);

    let block_hash = block.hash();
    let parent_hash = block.parent_hash;
    let parent_state_root_result = parent_state_root_by_hash(parent_hash);

    if let Some(dir) = options.capture_dir {
        let fixture = AccessedStateFixture {
            block_number,
            block_hash,
            parent_state_root: parent_state_root_result.as_ref().copied().unwrap_or_default(),
            accessed: accessed.clone(),
        };
        match save_fixture(dir, &fixture) {
            Ok(path) => info!(
                target: "partial_stateless",
                block = block_number,
                path = %path.display(),
                accounts = accessed.accounts.len(),
                storage = accessed.storage.len(),
                codes = accessed.codes.len(),
                "Captured accessed-state fixture"
            ),
            Err(e) => warn!(
                target: "partial_stateless",
                block = block_number,
                error = %e,
                "Failed to capture accessed-state fixture"
            ),
        }
    }

    let cache_policy_id =
        last_n_blocks_cache_policy_id(config.account_window, config.storage_window);
    let cache_policy_metadata = format!(
        "LastNBlocks(account: {}, storage/code: {})",
        config.account_window, config.storage_window
    );
    let cache_block_before = cache.current_block();
    let cache_parent_synced = cache_block_before == parent_block_number;
    if !cache_parent_synced {
        warn!(
            target: "partial_stateless",
            block = block_number,
            cache_block = cache_block_before,
            expected_parent_block = parent_block_number,
            "Cache is not synced to the parent block. Metrics will update the local cache, but cache-coherent sidecar generation is disabled for this block."
        );
    }
    let prev_cache_anchor = cache_parent_synced
        .then(|| cache.cache_anchor(parent_block_number, parent_hash, cache_policy_id));
    let snapshot_start = Instant::now();
    let mut prev_cache_for_reexec = previous_cache_snapshot(
        cache,
        config,
        cache_parent_synced,
        options.run_sidecar_preflight,
        options.validation_bench_output.is_some(),
        options.force_previous_cache_snapshot,
    );
    let snapshot_us = snapshot_start.elapsed().as_micros() as u64;
    let snapshot_created = prev_cache_for_reexec.is_some();
    let snapshot_estimated_bytes = prev_cache_for_reexec
        .as_ref()
        .map(NetworkStateCache::estimated_memory_bytes)
        .unwrap_or_default();
    let cache_snapshot_before = cache.snapshot();
    let cache_memory_before = cache.estimated_memory_bytes();
    let miss = cache.compute_miss(&accessed);
    let accessed_targets = accessed_to_state_targets(&accessed);
    let cache_hit_targets = cache_hit_targets(&accessed, &miss);

    let stats = cache.on_block_executed(block_number, &accessed);
    let snapshot = cache.snapshot();
    let cache_memory_after = cache.estimated_memory_bytes();

    info!(
        target: "partial_stateless",
        block = block_number,
        "═══════════════════════════════════════════════════"
    );
    info!(
        target: "partial_stateless",
        block = block_number,
        accessed_accounts = accessed.accounts.len(),
        accessed_storage = accessed.storage.len(),
        accessed_codes = accessed.codes.len(),
        total_accessed = accessed.total_keys(),
        "Block state access"
    );
    info!(
        target: "partial_stateless",
        block = block_number,
        miss_ratio = format!("{:.1}%", miss.miss_ratio * 100.0),
        missed_accounts = miss.missed_accounts.len(),
        missed_storage = miss.missed_storage.len(),
        missed_codes = miss.missed_codes.len(),
        total_missed = miss.total_missed,
        "Witness requirement (cache miss)"
    );

    let (raw_targets, _) = build_sidecar_targets(&miss);

    let missed_bytecode_bytes: usize = miss
        .missed_codes
        .iter()
        .filter_map(|code_hash| accessed.codes.get(code_hash))
        .map(|bytes| bytes.len())
        .sum();
    let missed_bytecodes: Vec<Bytes> = miss
        .missed_codes
        .iter()
        .filter_map(|code_hash| accessed.codes.get(code_hash).cloned())
        .collect();

    let full_sidecar_baseline_stats: Option<WitnessResult> = if options.compute_baseline {
        let full_targets = state_targets_to_proof_targets(&accessed_targets);
        let full_target_accounts = full_targets.len();
        let full_target_slots: usize = full_targets.values().map(|slots| slots.len()).sum();
        let full_bytecode_bytes: usize = accessed.codes.values().map(|bytes| bytes.len()).sum();
        let full_start = Instant::now();
        match state_provider.multiproof(TrieInput::default(), full_targets) {
            Ok(full_proof) => {
                let elapsed_ms = full_start.elapsed().as_millis() as u64;
                let mut full_result = measure_multiproof_size(&full_proof, full_bytecode_bytes);
                full_result.computation_time_ms = Some(elapsed_ms);
                full_result.target_accounts = full_target_accounts;
                full_result.target_storage_slots = full_target_slots;
                Some(full_result)
            }
            Err(e) => {
                warn!(
                    target: "partial_stateless",
                    block = block_number,
                    error = %e,
                    "Failed to compute full sidecar baseline multiproof (comparison dropped for this block)"
                );
                None
            }
        }
    } else {
        None
    };

    let parent_state_root = parent_state_root_result.map_err(|err| {
        rollback_sidecar_transition(
            cache,
            block_number,
            eyre::eyre!("failed to resolve parent state root: {err}"),
        )
    })?;

    // Hoisted out of the sidecar block below because a Weak-first block needs it before the
    // Partial witness is built. `ancestor_headers_for_range` is `FnOnce`, so it is resolved once
    // here and reused; the cost is one extra header range read on blocks that end up publishing
    // nothing.
    let ancestor_headers = ancestor_headers_for_range(lowest_block_number, block_number)
        .map_err(|err| rollback_sidecar_transition(cache, block_number, err))?;

    // Build order alternates so that neither witness permanently pays the cold page-cache read.
    // Measured before this: Weak proved a strictly larger target set than Partial yet came out
    // 215.8 ms against 552.2 ms, because it always ran second on pages Partial had just warmed.
    let prebuilt_weak =
        if options.validation_bench_output.is_some() && weak_builds_first(block_number) {
            Some(
                build_weak_stateless_sidecar(
                    state_provider,
                    parent_state_root,
                    block.state_root(),
                    parent_hash,
                    block_hash,
                    block_number,
                    &hashed_post_state,
                    &accessed,
                    &ancestor_headers,
                    config,
                )
                .map_err(|err| rollback_sidecar_transition(cache, block_number, err))?,
            )
        } else {
            None
        };

    let rusage_before = options.resource_metrics.then(process_rusage);
    let start = Instant::now();
    let saved_sidecar_path;
    let builder_initial_proof_source;
    let builder_initial_provider_us;
    let builder_initial_targets;
    let builder_distinct_storage_tries;
    let builder_parallel_storage_workers;
    let builder_parallel_account_workers;
    let builder_initial_proof_nodes;
    let builder_initial_proof_bytes;
    let builder_transition_witness_build_us;
    let builder_trie_clone_bytes;
    let builder_trie_clone_rss_delta_bytes;
    let builder_trie_storage_tries_copied;
    let builder_trie_storage_tries_total;
    // The snapshot's copies are spread across the transition and retention rather than paid at the
    // clone, so counting them means bracketing the whole transaction. The counter is process-wide,
    // which is exact here because the ExEx applies one transition at a time and the paired
    // verification runs after this delta is read.
    let cow_copies_before = cow_copies_taken();
    let mut builder_trie_mutation = None;
    let mut builder_witness_commitment = None;
    let mut sidecar_constructed = false;
    let mut retained_sidecar = None;
    // Declared uninitialized like the timing bindings above: every path that reaches the report
    // either assigns it or returns, so an initial `None` would be dead.
    let displaced_trie_cache;
    let proof_source = RethStateProviderSource::new(state_provider, options.parallel_initial_proof);
    let build_ctx = proof_source.context();
    let witness = {
        let base =
            generate_cache_aware_base_proof(&build_ctx, &hashed_post_state, &miss, trie_cache)
                .map_err(|err| rollback_sidecar_transition(cache, block_number, err))?;
        let CacheAwareFlatBuild {
            nodes,
            decoded_proof: proof,
            mut next_trie_cache,
            state_root: local_state_root,
            provider_calls,
            structural_rounds,
            initial_targets,
            initial_proof_nodes,
            initial_proof_bytes,
            structural_targets,
            structural_account_targets,
            structural_storage_targets,
            cache_covered_mutation_targets,
            trie_clone_us,
            trie_clone_bytes,
            trie_clone_rss_delta_bytes,
            transition_us: local_root_us,
            structural_provider_us,
        } = build_cache_aware_flat_transition(
            &build_ctx,
            parent_state_root,
            block.state_root(),
            hashed_post_state.clone(),
            &miss,
            trie_cache,
            &base,
        )
        .map_err(|err| rollback_sidecar_transition(cache, block_number, err))?;
        let initial_provider_us = base.provider_us;
        let initial_proof_source = base.proof_source;
        let transition_witness_build_us = start.elapsed().as_micros() as u64;
        builder_initial_proof_source = initial_proof_source;
        builder_initial_provider_us = initial_provider_us;
        builder_initial_targets = initial_targets;
        builder_distinct_storage_tries = base.targets.distinct_storage_tries();
        builder_parallel_storage_workers = base.parallel_storage_workers;
        builder_parallel_account_workers = base.parallel_account_workers;
        builder_initial_proof_nodes = initial_proof_nodes;
        builder_initial_proof_bytes = initial_proof_bytes;
        builder_transition_witness_build_us = transition_witness_build_us;
        builder_trie_clone_bytes = trie_clone_bytes;
        builder_trie_clone_rss_delta_bytes = trie_clone_rss_delta_bytes;

        // Measured against the *parent* trie, which is what the clone copied. Walking every
        // retained path is linear in cache size, so it stays behind the diagnostics flag.
        if options.trie_cache_diagnostics {
            let changed = TrieChangeSet::from_hashed_post_state(&hashed_post_state);
            let mutation = trie_cache.mutation_metrics(&changed);
            info!(
                target: "partial_stateless",
                block = block_number,
                dirtied_account_paths = mutation.dirtied_account_paths,
                retained_account_paths = mutation.retained_account_paths,
                dirtied_storage_paths = mutation.dirtied_storage_paths,
                retained_storage_paths = mutation.retained_storage_paths,
                dirtied_storage_tries = mutation.dirtied_storage_tries,
                dirtied_path_share = mutation.dirtied_path_share(),
                deepest_prefix_share = mutation.deepest_account_prefix().dirtied_share(),
                revealed_nodes = mutation.revealed_nodes(),
                trie_clone_us,
                trie_clone_bytes,
                trie_clone_rss_delta_bytes,
                "Measured trie mutation footprint against the cloned parent trie"
            );
            for trie in
                mutation.per_storage_trie.iter().filter(|trie| trie.dirtied_paths > 0).take(8)
            {
                debug!(
                    target: "partial_stateless",
                    block = block_number,
                    hashed_address = ?trie.hashed_address,
                    dirtied_paths = trie.dirtied_paths,
                    retained_paths = trie.retained_paths,
                    revealed_nodes = trie.revealed_nodes,
                    wiped = trie.wiped,
                    "Storage trie mutation footprint"
                );
            }
            builder_trie_mutation = Some(mutation);
        }
        let elapsed_ms = transition_witness_build_us / 1_000;
        let mut result = measure_transition_witness_size(&nodes, &proof, missed_bytecode_bytes);
        let witness_state = PartialExecutionWitnessState::MptTransitionNodes(nodes);
        result.computation_time_ms = Some(elapsed_ms);
        if let Some((cpu_us_before, majflt_before, minflt_before)) = rusage_before {
            let (cpu_us_after, majflt_after, minflt_after) = process_rusage();
            result.cpu_time_ms = Some(cpu_us_after.saturating_sub(cpu_us_before) / 1000);
            result.major_page_faults = Some(majflt_after.saturating_sub(majflt_before));
            result.minor_page_faults = Some(minflt_after.saturating_sub(minflt_before));
        }
        result.target_accounts = base.targets.account_count() + structural_account_targets;
        result.target_storage_slots = base.targets.storage_count() + structural_storage_targets;
        info!(
            target: "partial_stateless",
            block = block_number,
            block_hash = ?block_hash,
            transition_witness_nodes = result.account_proof_nodes + result.storage_proof_nodes,
            initial_targets,
            structural_targets,
            structural_rounds,
            provider_calls,
            structural_provider_us,
            initial_provider_us,
            initial_proof_source,
            initial_proof_nodes,
            initial_proof_bytes,
            cache_covered_mutation_targets,
            computed_state_root = ?local_state_root,
            "Generated cache-aware canonical transition witness"
        );

        let trie_retention_start = Instant::now();
        next_trie_cache.retain_from_value_cache(cache);
        let trie_retention_us = trie_retention_start.elapsed().as_micros() as u64;
        builder_trie_storage_tries_total = next_trie_cache.storage_trie_count() as u64;
        builder_trie_storage_tries_copied = cow_copies_taken().saturating_sub(cow_copies_before);

        let validation_start = Instant::now();
        let trie_shape_metrics = if options.trie_cache_diagnostics {
            match next_trie_cache.validate_against_value_cache(cache) {
                Ok(metrics) => Some(metrics),
                Err(err) => {
                    return Err(rollback_sidecar_transition(
                        cache,
                        block_number,
                        eyre::eyre!("trie-cache invariant validation failed: {err}"),
                    ))
                }
            }
        } else {
            None
        };
        let trie_validation_us = validation_start.elapsed().as_micros() as u64;
        let next_cache_anchor = cache_parent_synced
            .then(|| cache.cache_anchor(block_number, block_hash, cache_policy_id));

        let sidecar_generation_result: eyre::Result<Option<PathBuf>> = 'sidecar: {
            let Some((prev_cache_anchor, next_cache_anchor)) = sidecar_publication_anchors(
                prev_cache_anchor,
                next_cache_anchor,
                options.ready_parent,
            ) else {
                debug!(
                    target: "partial_stateless",
                    block = block_number,
                    ready = options.ready_parent.is_some(),
                    "Sidecar built but not published: the cache is not Ready for this parent"
                );
                break 'sidecar Ok(None);
            };
            if cache.current_block() != block_number {
                warn!(
                    target: "partial_stateless",
                    block = block_number,
                    cache_block = cache.current_block(),
                    expected_block = block_number,
                    "Cache state mismatch: cache block is not synced to block number. Skipping sidecar generation."
                );
                break 'sidecar Ok(None);
            }

            // The flat canonical witness contains every parent-state node required for
            // both cache misses and the post-state trie transition.
            //
            // Assembled by the shared constructor rather than here, so a sidecar this builder
            // publishes and one an offline generator produces for the same block under the same
            // policy are the same object built by the same code — which is the only thing that
            // makes a size or miss comparison between two policies a fact about the policies.
            let sidecar = assemble_sidecar(SidecarAssembly {
                parent_hash,
                parent_state_root,
                block_hash,
                block_number,
                cache_block: parent_block_number,
                cache_policy_id,
                cache_policy_metadata: cache_policy_metadata.clone(),
                prev_cache_anchor,
                next_cache_anchor,
                miss_manifest: raw_targets.clone(),
                witness_state: witness_state.clone(),
                codes: missed_bytecodes.clone(),
                headers: ancestor_headers.clone(),
                stats: result.clone(),
            });
            builder_witness_commitment = Some(sidecar.witness_commitment);
            sidecar_constructed = true;
            if options.retain_sidecar {
                retained_sidecar = Some(sidecar.clone());
            }

            let root_witness_completeness = if options.run_sidecar_preflight {
                let prev_cache_for_reexec =
                    match require_previous_cache_snapshot(prev_cache_for_reexec.as_mut()) {
                        Ok(snapshot) => snapshot,
                        Err(err) => break 'sidecar Err(err),
                    };
                let reexec_report = if let Some(output_path) = options.validation_bench_output {
                    benchmark_sidecar_validation(
                        rules,
                        state_provider,
                        block,
                        prev_cache_for_reexec,
                        trie_cache,
                        &sidecar,
                        prebuilt_weak,
                        &hashed_post_state,
                        &accessed,
                        &ancestor_headers,
                        config,
                        options.reexec_limits,
                        output_path,
                        historical_full_db_evm_us,
                        transition_witness_build_us,
                        historical,
                        cache_memory_before,
                        options.retained_generation,
                    )
                } else {
                    verify_and_apply_provider_assisted_sidecar(
                        rules,
                        state_provider,
                        block,
                        prev_cache_for_reexec,
                        &sidecar,
                        cache_policy_id,
                        options.reexec_limits,
                        trie_cache,
                        TrieCacheDisposition::Discard,
                    )
                };
                let reexec_report = match reexec_report {
                    Ok(report) => report,
                    Err(err) => {
                        match persist_preflight_failure_artifacts(
                            options.sidecar_dir,
                            state_provider,
                            &sidecar,
                            block.state_root(),
                            prev_cache_for_reexec,
                            trie_cache,
                            &err,
                        ) {
                            Ok(path) => warn!(
                                target: "partial_stateless",
                                block = block_number,
                                path = %path.display(),
                                error = %err,
                                "Saved sidecar preflight failure artifacts"
                            ),
                            Err(artifact_error) => warn!(
                                target: "partial_stateless",
                                block = block_number,
                                error = %err,
                                artifact_error = %artifact_error,
                                "Failed to save sidecar preflight failure artifacts"
                            ),
                        }
                        break 'sidecar Err(eyre::eyre!("sidecar preflight failed: {err}"))
                    }
                };

                if options.validation_bench_output.is_none() &&
                    !reexec_report.outcome.root_witness_completeness.trustless_root_ready
                {
                    warn!(
                        target: "partial_stateless",
                        block = block_number,
                        partial_state_trustless_verification_ready = false,
                        missing_account_paths = reexec_report
                            .outcome.root_witness_completeness
                            .missing_account_paths
                            .len(),
                        missing_storage_paths = reexec_report
                            .outcome.root_witness_completeness
                            .missing_storage_paths
                            .len(),
                        "Partial-state node trustless verification is not ready; current state_root check is provider-assisted"
                    );
                }
                info!(
                    target: "partial_stateless",
                    block = block_number,
                    partial_state_trustless_verification_ready = reexec_report
                        .outcome.root_witness_completeness
                        .trustless_root_ready,
                    computed_state_root = ?reexec_report.outcome.state_root,
                    reexec_accounts = reexec_report.outcome.actual_accessed.accounts.len(),
                    reexec_storage = reexec_report.outcome.actual_accessed.storage.len(),
                    reexec_codes = reexec_report.outcome.actual_accessed.codes.len(),
                    expected_miss_accounts = reexec_report.outcome.expected_miss.accounts.len(),
                    expected_miss_storage = reexec_report.outcome.expected_miss.storage.len(),
                    expected_miss_codes = reexec_report.outcome.expected_miss.code_hashes.len(),
                    next_cache_root = ?reexec_report.outcome.next_cache_anchor.cache_root,
                    "Sidecar preflight succeeded"
                );
                // Unconditional for the same reason as in the live verifier: the core compares
                // this root against the header and bails before returning, so the mismatch and
                // blind-path arms this used to carry were both unreachable.
                info!(
                    target: "partial_stateless",
                    block = block_number,
                    trustless_state_root = ?reexec_report.outcome.state_root,
                    "Trustless state root VERIFIED (trie node cache + witness only)"
                );
                RootWitnessCompletenessSummary::from_report(
                    &reexec_report.outcome.root_witness_completeness,
                )
            } else {
                RootWitnessCompletenessSummary::default()
            };

            if options.validation_bench_output.is_some() {
                info!(
                    target: "partial_stateless_bench",
                    block = block_number,
                    "Paired benchmark serialized sidecars in memory for this block; file and manifest writes skipped"
                );
                break 'sidecar Ok(None);
            }

            fs::create_dir_all(options.sidecar_dir)
                .map_err(|err| eyre::eyre!("failed to create sidecar directory: {err}"))?;
            let sidecar_path = sidecar_path(options.sidecar_dir, block_number, block_hash);
            let serialize_start = Instant::now();
            let sidecar_bytes = bincode::serialize(&sidecar)
                .map_err(|err| eyre::eyre!("failed to serialize sidecar: {err}"))?;
            let sidecar_serialize_us = serialize_start.elapsed().as_micros() as u64;
            let write_start = Instant::now();
            fs::write(&sidecar_path, &sidecar_bytes).map_err(|err| {
                eyre::eyre!("failed to write sidecar file {:?}: {err}", sidecar_path)
            })?;
            let sidecar_write_us = write_start.elapsed().as_micros() as u64;
            let sidecar_bytes_len =
                fs::metadata(&sidecar_path).map(|m| m.len() as usize).unwrap_or(0);
            let partial_state_trustless_verification_ready =
                root_witness_completeness.trustless_root_ready;
            let manifest = SidecarBenchmarkManifest {
                schema_version: 1,
                block_number,
                block_hash,
                parent_hash,
                parent_state_root,
                cache_block: parent_block_number,
                cache_policy_id,
                prev_cache_anchor,
                next_cache_anchor,
                cache_policy_metadata: cache_policy_metadata.clone(),
                sidecar_file: sidecar_path
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_else(|| sidecar_path.display().to_string()),
                sidecar_bytes: sidecar_bytes_len,
                cache_before: CacheFootprintStats::new(
                    cache_snapshot_before.total_accounts,
                    cache_snapshot_before.total_storage_slots,
                    cache_snapshot_before.total_codes,
                    cache_memory_before,
                ),
                cache_after: CacheFootprintStats::new(
                    snapshot.total_accounts,
                    snapshot.total_storage_slots,
                    snapshot.total_codes,
                    cache_memory_after,
                ),
                accessed: StateTargetStats::from_targets(&accessed_targets),
                cache_hit: StateTargetStats::from_targets(&cache_hit_targets),
                sidecar_miss: StateTargetStats::from_targets(&sidecar.cache_miss_targets),
                provider_assisted_preflight: options.run_sidecar_preflight,
                partial_state_trustless_verification_ready,
                root_witness_completeness,
                full_sidecar_baseline_stats: full_sidecar_baseline_stats.clone(),
                partial_sidecar_stats: result.clone(),
                reduction: full_sidecar_baseline_stats
                    .as_ref()
                    .map(|full| WitnessReductionStats::new(&result, full)),
            };
            let manifest_path = sidecar_path.with_extension("manifest.json");
            let manifest_saved = match serde_json::to_vec_pretty(&manifest) {
                Ok(manifest_bytes) => match fs::write(&manifest_path, manifest_bytes) {
                    Ok(()) => true,
                    Err(err) => {
                        warn!(
                            target: "partial_stateless",
                            block = block_number,
                            path = %manifest_path.display(),
                            error = %err,
                            "Failed to write diagnostic sidecar manifest"
                        );
                        false
                    }
                },
                Err(err) => {
                    warn!(
                        target: "partial_stateless",
                        block = block_number,
                        error = %err,
                        "Failed to serialize diagnostic sidecar manifest"
                    );
                    false
                }
            };
            info!(
                target: "partial_stateless",
                block = block_number,
                path = %sidecar_path.display(),
                manifest = %manifest_path.display(),
                manifest_saved,
                size = format_bytes(sidecar_bytes_len),
                sidecar_serialize_us,
                sidecar_write_us,
                "Saved witness sidecar successfully"
            );
            Ok(Some(sidecar_path))
        };

        match sidecar_generation_result {
            Ok(path) => saved_sidecar_path = path,
            Err(e) => return Err(rollback_sidecar_transition(cache, block_number, e)),
        }

        // Advance only after sidecar generation/preflight has either succeeded or been
        // intentionally skipped. Coherent failures return above after rolling back the
        // value cache, so the two caches remain aligned.
        //
        // The parent is handed back instead of being dropped here. That is the whole cost of a
        // one-deep retained generation: the copy was already made above, and the caller decides
        // whether to keep it or let it fall out of scope.
        displaced_trie_cache = Some(mem::replace(trie_cache, next_trie_cache));
        if let Some(metrics) = trie_shape_metrics {
            info!(
                target: "partial_stateless",
                block = block_number,
                validation = "passed",
                trie_clone_us,
                local_root_update_us = local_root_us,
                trie_retention_us,
                trie_validation_us,
                retained_account_paths = metrics.retained_account_paths,
                retained_storage_tries = metrics.retained_storage_tries,
                retained_storage_paths = metrics.retained_storage_paths,
                account_revealed_nodes = metrics.account_revealed_nodes,
                storage_revealed_nodes = metrics.storage_revealed_nodes,
                trie_cache_bytes = metrics.estimated_memory_bytes,
                account_key_prefixes_d0 = metrics.account_key_prefixes[0],
                account_key_prefixes_d1 = metrics.account_key_prefixes[1],
                account_key_prefixes_d2 = metrics.account_key_prefixes[2],
                account_key_prefixes_d3 = metrics.account_key_prefixes[3],
                account_key_prefixes_d4 = metrics.account_key_prefixes[4],
                account_key_prefixes_d5 = metrics.account_key_prefixes[5],
                account_prefix_coverage_d0_pct = metrics.account_prefix_coverage[0] * 100.0,
                account_prefix_coverage_d1_pct = metrics.account_prefix_coverage[1] * 100.0,
                account_prefix_coverage_d2_pct = metrics.account_prefix_coverage[2] * 100.0,
                account_prefix_coverage_d3_pct = metrics.account_prefix_coverage[3] * 100.0,
                account_prefix_coverage_d4_pct = metrics.account_prefix_coverage[4] * 100.0,
                account_prefix_coverage_d5_pct = metrics.account_prefix_coverage[5] * 100.0,
                "Trie-shape cache benchmark"
            );
        }
        Some(result)
    };

    if let Some(witness) = &witness {
        info!(
            target: "partial_stateless",
            block = block_number,
            witness_total_bytes = witness.total_size_bytes,
            witness_total = format_bytes(witness.total_size_bytes),
            account_proof_bytes = witness.account_proof_bytes,
            account_proof = format_bytes(witness.account_proof_bytes),
            storage_proof_bytes = witness.storage_proof_bytes,
            storage_proof = format_bytes(witness.storage_proof_bytes),
            bytecode_bytes = witness.bytecode_bytes,
            bytecode_size = format_bytes(witness.bytecode_bytes),
            account_proof_nodes = witness.account_proof_nodes,
            storage_proof_nodes = witness.storage_proof_nodes,
            target_accounts = witness.target_accounts,
            target_storage_slots = witness.target_storage_slots,
            computation_time_ms = witness.computation_time_ms.unwrap_or(0),
            cpu_time_ms = witness.cpu_time_ms.unwrap_or(0),
            major_page_faults = witness.major_page_faults.unwrap_or(0),
            minor_page_faults = witness.minor_page_faults.unwrap_or(0),
            "Witness size (Merkle proof)"
        );
    }

    let builder_total_us = builder_total_start.elapsed().as_micros() as u64;
    if let Some(path) = options.builder_bench_output {
        let record = BuilderBenchmarkRecord {
            schema_version: BUILDER_BENCHMARK_SCHEMA_VERSION,
            block_number,
            block_hash,
            historical_full_db_evm_us,
            artifact_available: disposition.artifact_available(),
            artifact_reused: disposition.artifact_reused(),
            shadow_sampled: disposition.shadow_sampled(),
            fallback_reason: disposition.fallback_reason(),
            builder_total_us,
            transition_witness_build_us: builder_transition_witness_build_us,
            snapshot_created,
            snapshot_us,
            snapshot_estimated_bytes,
            cache_parent_synced,
            initial_proof_source: builder_initial_proof_source,
            initial_provider_us: builder_initial_provider_us,
            initial_targets: builder_initial_targets,
            distinct_storage_tries: builder_distinct_storage_tries,
            parallel_storage_workers: builder_parallel_storage_workers,
            parallel_account_workers: builder_parallel_account_workers,
            initial_proof_nodes: builder_initial_proof_nodes,
            initial_proof_bytes: builder_initial_proof_bytes,
            witness_commitment: builder_witness_commitment,
            sidecar_constructed,
            sidecar_published: saved_sidecar_path.is_some(),
            value_cache_bytes: cache.estimated_memory_bytes(),
            trie_cache_bytes: trie_cache.estimated_memory_bytes(),
            retained_generation: options.retained_generation,
            trie_clone_bytes: builder_trie_clone_bytes,
            trie_clone_rss_delta_bytes: builder_trie_clone_rss_delta_bytes,
            trie_storage_tries_copied: builder_trie_storage_tries_copied,
            trie_storage_tries_total: builder_trie_storage_tries_total,
            trie_mutation: builder_trie_mutation.as_ref().map(TrieMutationSummary::from),
        };
        if let Err(err) = append_builder_record(path, &record) {
            warn!(
                target: "partial_stateless_bench",
                block = block_number,
                error = %err,
                "Failed to append builder benchmark record"
            );
        }
    }

    info!(
        target: "partial_stateless",
        block = block_number,
        cache_accounts = snapshot.total_accounts,
        cache_storage = snapshot.total_storage_slots,
        cache_codes = snapshot.total_codes,
        estimated_memory = format_bytes(cache.estimated_memory_bytes()),
        accounts_added = stats.accounts_added,
        accounts_refreshed = stats.accounts_refreshed,
        accounts_evicted = stats.accounts_evicted,
        storage_added = stats.storage_added,
        storage_refreshed = stats.storage_refreshed,
        storage_evicted = stats.storage_evicted,
        "Cache state after update"
    );

    info!(
        target: "partial_stateless_bench",
        benchmark = "builder_end_to_end",
        block = block_number,
        block_hash = ?block_hash,
        historical_full_db_evm_us,
        artifact_reused = disposition.artifact_reused(),
        fallback_reason = disposition.fallback_reason(),
        builder_total_us,
        snapshot_created,
        snapshot_us,
        initial_proof_source = builder_initial_proof_source,
        initial_provider_us = builder_initial_provider_us,
        sidecar_published = saved_sidecar_path.is_some(),
        "Builder sidecar end-to-end benchmark"
    );

    // Last, deliberately. The Partial witness above is the one an ordinary run builds, and running
    // the full witness ahead of it would hand it a warmed page cache no production block gets.
    // Nothing here feeds the sidecar, the caches, or the record — a capture that changed what the
    // builder produced would be capturing a different system than the one it is describing.
    let policy_dataset_material = if options.capture_policy_dataset {
        Some(
            capture_policy_dataset_material(
                rules,
                state_provider,
                block,
                parent_state_root,
                &hashed_post_state,
                &accessed,
                &ancestor_headers,
                config,
                options.reexec_limits,
                &historical,
                artifact_reused,
            )
            .map_err(|err| {
                eyre::eyre!(
                    "policy replay dataset capture failed for block {block_number}: {err:#}"
                )
            })?,
        )
    } else {
        None
    };

    Ok(BuilderBlockReport {
        cache_update: stats,
        witness,
        sidecar_path: saved_sidecar_path,
        sidecar: retained_sidecar,
        displaced_trie_cache,
        policy_dataset_material,
    })
}

/// Builds the policy-neutral full witness for one block and proves it before it is recorded.
///
/// The proof is not a formality. A recorded witness is the *only* parent-state input the offline
/// generator will ever have for this block, so an incomplete one does not fail at capture time — it
/// fails a thousand blocks later, inside a generator that cannot tell an incomplete recording from
/// a policy that genuinely needed a node nobody proved. Checking here converts that into a capture
/// error on the block that caused it.
///
/// What the check establishes, and it is the full list the offline stage depends on: the witness
/// alone re-executes the block with no database (which covers the ancestor headers, since a
/// missing one fails BLOCKHASH), the state root it reconstructs matches the block's own header,
/// and the gas, receipts root, and requests hash of that database-free execution equal the ones
/// this node's full-database execution produced. The access set is compared last and exactly:
/// every policy's miss set is computed from it, so a set that is wrong anywhere produces a witness
/// that is wrong somewhere.
#[expect(clippy::too_many_arguments)]
fn capture_policy_dataset_material<Evm, Consensus>(
    rules: ValidatorRules<'_, Evm, Consensus>,
    state_provider: &dyn StateProvider,
    block: &RecoveredBlock<BlockTy<EthPrimitives>>,
    parent_state_root: B256,
    hashed_post_state: &HashedPostState,
    accessed: &BlockAccessedState,
    ancestor_headers: &[Bytes],
    config: &CacheConfig,
    limits: &SidecarReexecLimits,
    historical: &HistoricalOutputs,
    artifact_reused: bool,
) -> eyre::Result<PolicyDatasetMaterial>
where
    Evm: ConfigureEvm<Primitives = EthPrimitives>,
    Consensus: FullConsensus<EthPrimitives> + ?Sized,
{
    if !historical.consensus_ok {
        eyre::bail!(
            "this node's own full-database execution disagrees with the block header; a corpus \
             recorded from it would carry a block nobody validated"
        )
    }

    let source = RethStateProviderSource::new(state_provider, None);
    let full = build_full_witness_sidecar(
        &source.context(),
        parent_state_root,
        block.state_root(),
        block.parent_hash,
        block.hash(),
        block.number(),
        hashed_post_state,
        accessed,
        ancestor_headers,
        config,
    )?;

    // A cold pair: no cached value, no revealed trie node. Everything the re-execution reads has
    // to come out of the witness, which is exactly the condition the offline generator runs under.
    let mut cold_cache = config.new_cache_at(block.number().saturating_sub(1));
    let mut cold_trie = PartialTrieNodeCache::new();
    let verified = verify_and_apply_sidecar(
        rules,
        block,
        &mut cold_cache,
        &full.sidecar,
        config.cache_policy_id(),
        limits,
        &mut cold_trie,
        TrieCacheDisposition::Discard,
    )
    .map_err(|err| eyre::eyre!("the full witness does not re-execute this block: {err:#}"))?;
    let outcome = verified.outcome;

    if outcome.state_root != block.state_root() {
        eyre::bail!(
            "database-free re-execution computed state root {:?}, but the header claims {:?}",
            outcome.state_root,
            block.state_root()
        )
    }
    if outcome.execution_gas_used != historical.gas_used ||
        outcome.execution_receipts_root != historical.receipts_root ||
        outcome.execution_requests_hash != historical.requests_hash ||
        outcome.execution_requests_empty != historical.requests_empty
    {
        eyre::bail!(
            "database-free re-execution disagrees with this node's own execution: gas {} vs {}, \
             receipts {:?} vs {:?}, requests {:?} vs {:?} (empty {} vs {})",
            outcome.execution_gas_used,
            historical.gas_used,
            outcome.execution_receipts_root,
            historical.receipts_root,
            outcome.execution_requests_hash,
            historical.requests_hash,
            outcome.execution_requests_empty,
            historical.requests_empty,
        )
    }
    if let Some(divergence) = first_access_divergence(accessed, &outcome.actual_accessed) {
        eyre::bail!(
            "the access set the database-free re-execution observed differs from the recorded \
             one: {divergence}"
        )
    }

    // Taken only now: the sidecar was needed intact for the validation above, and taking the node
    // set by move afterwards costs nothing.
    let build_us = full.build_us;
    let nodes = full.into_nodes()?;
    info!(
        target: "partial_stateless",
        block = block.number(),
        witness_nodes = nodes.len(),
        witness_bytes = nodes.iter().map(|node| node.len()).sum::<usize>(),
        codes = accessed.codes.len(),
        ancestor_headers = ancestor_headers.len(),
        build_us,
        "Proved the policy-neutral full witness for the policy replay dataset"
    );

    Ok(PolicyDatasetMaterial {
        parent_state_root,
        expected_state_root: block.state_root(),
        accessed: accessed.clone(),
        access_provenance: if artifact_reused {
            RecordedAccessProvenance::EngineArtifact
        } else {
            RecordedAccessProvenance::Reexecution
        },
        full_transition_nodes: nodes,
        ancestor_headers: ancestor_headers.to_vec(),
    })
}

/// The first way two access sets differ, described well enough to act on.
///
/// Returns a sentence rather than a bool: a capture that failed on divergence and could not say
/// which key diverged would send whoever reads the log back to re-running the capture.
fn first_access_divergence(
    recorded: &BlockAccessedState,
    observed: &BlockAccessedState,
) -> Option<String> {
    for (address, data) in &recorded.accounts {
        match observed.accounts.get(address) {
            None => return Some(format!("account {address:?} is absent from the re-execution")),
            Some(seen) if seen != data => {
                return Some(format!("account {address:?} differs: {data:?} vs {seen:?}"))
            }
            Some(_) => {}
        }
    }
    for key in observed.accounts.keys() {
        if !recorded.accounts.contains_key(key) {
            return Some(format!("the re-execution saw account {key:?}, which was not recorded"))
        }
    }
    for (key, value) in &recorded.storage {
        match observed.storage.get(key) {
            None => return Some(format!("storage {key:?} is absent from the re-execution")),
            Some(seen) if seen != value => {
                return Some(format!("storage {key:?} differs: {value:?} vs {seen:?}"))
            }
            Some(_) => {}
        }
    }
    for key in observed.storage.keys() {
        if !recorded.storage.contains_key(key) {
            return Some(format!("the re-execution saw storage {key:?}, which was not recorded"))
        }
    }
    for (code_hash, code) in &recorded.codes {
        match observed.codes.get(code_hash) {
            None => return Some(format!("code {code_hash:?} is absent from the re-execution")),
            Some(seen) if seen != code => {
                return Some(format!(
                    "code {code_hash:?} differs in length: {} vs {}",
                    code.len(),
                    seen.len()
                ))
            }
            Some(_) => {}
        }
    }
    for code_hash in observed.codes.keys() {
        if !recorded.codes.contains_key(code_hash) {
            return Some(format!("the re-execution saw code {code_hash:?}, which was not recorded"))
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::{Address, U256};
    use partial_stateless::policy::AccountData;

    #[test]
    fn builder_and_verifier_order_alternate_independently() {
        // Bit 0 drives validation order, bit 1 drives build order, so all four combinations occur
        // equally often. Sharing a bit would leave the two perfectly correlated and would trade
        // the fixed build-order bias for a fixed pairing instead of removing it.
        let combinations: std::collections::HashSet<_> =
            (100..104).map(|n| (partial_runs_first(n), weak_builds_first(n))).collect();
        assert_eq!(combinations.len(), 4, "every order pairing must occur: {combinations:?}");
    }

    #[test]
    fn weak_does_not_always_build_second() {
        assert!(!weak_builds_first(100));
        assert!(!weak_builds_first(101));
        assert!(weak_builds_first(102));
        assert!(weak_builds_first(103));
    }

    #[test]
    fn paired_validation_order_alternates_by_block() {
        assert!(partial_runs_first(100));
        assert!(!partial_runs_first(101));
        assert!(partial_runs_first(102));
    }

    #[test]
    fn previous_cache_snapshot_is_only_needed_for_reexecution_paths() {
        assert!(!needs_previous_cache_snapshot(false, false, false));
        assert!(needs_previous_cache_snapshot(true, false, false));
        assert!(needs_previous_cache_snapshot(false, true, false));
        assert!(needs_previous_cache_snapshot(true, true, false));
        assert!(needs_previous_cache_snapshot(false, false, true));
    }

    #[test]
    fn normal_builder_skips_snapshot_while_preflight_forks_the_parent_cache() {
        let config = CacheConfig::default();
        let mut cache = config.new_cache();
        let address = Address::repeat_byte(0x11);
        let mut accessed = BlockAccessedState::default();
        accessed
            .accounts
            .insert(address, AccountData { nonce: 1, balance: U256::from(10), code_hash: None });
        cache.on_block_executed(99, &accessed);
        let parent_root = cache.cache_root();

        assert!(previous_cache_snapshot(&cache, &config, true, false, false, false).is_none());
        assert!(previous_cache_snapshot(&cache, &config, false, true, false, false).is_none());

        let preflight = previous_cache_snapshot(&cache, &config, true, true, false, false)
            .expect("preflight fork");
        let paired = previous_cache_snapshot(&cache, &config, true, false, true, false)
            .expect("paired fork");
        let forced = previous_cache_snapshot(&cache, &config, true, false, false, true)
            .expect("forced diagnostic fork");
        assert_eq!(preflight.current_block(), 99);
        assert_eq!(preflight.cache_root(), parent_root);
        assert_eq!(paired.cache_root(), parent_root);
        assert_eq!(forced.cache_root(), parent_root);

        cache.on_block_executed(100, &accessed);
        assert_ne!(cache.cache_root(), parent_root);
        assert_eq!(preflight.current_block(), 99);
        assert_eq!(preflight.cache_root(), parent_root);
    }

    #[test]
    fn publication_requires_anchors_and_a_matching_ready_parent() {
        let prev = CacheAnchor {
            block_number: 99,
            block_hash: B256::repeat_byte(0x11),
            cache_policy_id: B256::repeat_byte(0x22),
            cache_root: B256::repeat_byte(0x33),
        };
        let next = CacheAnchor {
            block_number: 100,
            block_hash: B256::repeat_byte(0x44),
            cache_policy_id: prev.cache_policy_id,
            cache_root: B256::repeat_byte(0x55),
        };
        let ready = ReadyParent {
            anchor: prev,
            trie_state_root: B256::repeat_byte(0x66),
            replay_depth: 61,
        };

        assert_eq!(
            sidecar_publication_anchors(Some(prev), Some(next), Some(&ready)),
            Some((prev, next))
        );
        assert_eq!(sidecar_publication_anchors(None, Some(next), Some(&ready)), None);
        assert_eq!(sidecar_publication_anchors(Some(prev), None, Some(&ready)), None);

        // A Warming cache builds the same sidecar and must not publish it: its policy identifier
        // advertises a window it has not replayed.
        assert_eq!(sidecar_publication_anchors(Some(prev), Some(next), None), None);

        // Ready for a different parent is not Ready for this one.
        let stale = ReadyParent { anchor: next, ..ready };
        assert_eq!(sidecar_publication_anchors(Some(prev), Some(next), Some(&stale)), None);
    }

    #[test]
    fn requested_preflight_fails_closed_without_previous_cache_snapshot() {
        assert!(require_previous_cache_snapshot(None).is_err());
    }

    #[test]
    fn cold_weak_cache_marks_every_accessed_value_as_a_witness_miss() {
        let address = Address::repeat_byte(0x11);
        let slot = B256::repeat_byte(0x22);
        let code_hash = B256::repeat_byte(0x33);
        let mut accessed = BlockAccessedState::default();
        accessed.accounts.insert(
            address,
            AccountData { nonce: 7, balance: U256::from(8), code_hash: Some(code_hash) },
        );
        accessed.storage.insert((address, slot), U256::from(9));
        accessed.codes.insert(code_hash, Bytes::from_static(&[1, 2, 3]));

        let miss = CacheConfig::default().new_cache().compute_miss(&accessed);

        assert_eq!(miss.missed_accounts, vec![address]);
        assert_eq!(miss.missed_storage, vec![(address, slot)]);
        assert_eq!(miss.missed_codes, vec![code_hash]);
        assert_eq!(miss.total_missed, accessed.total_keys());
        assert_eq!(miss.miss_ratio, 1.0);
    }
}
