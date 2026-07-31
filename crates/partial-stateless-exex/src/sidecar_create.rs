use crate::{
    benchmark::{
        append_builder_record, append_record, deserialize_sidecar_for_benchmark,
        serialize_sidecar_for_benchmark, BuilderBenchmarkRecord, ValidationBenchmarkRecord,
        WitnessSizeBreakdown,
    },
    format_bytes, process_rusage,
    sidecar_io::sidecar_path,
    sidecar_reexec::{
        verify_and_apply_provider_assisted_sidecar,
        verify_and_apply_trustless_sidecar_for_benchmark, SidecarReexecLimits,
        TrieCacheDisposition,
    },
    CacheConfig,
};
use alloy_primitives::{keccak256, map::B256Map, Bytes, B256};
use alloy_rlp::{Encodable, EMPTY_STRING_CODE};
use partial_stateless::{
    accessed_state::BlockAccessedState,
    fixture::{save_fixture, AccessedStateFixture},
    last_n_blocks_cache_policy_id,
    network_cache::{MissResult, NetworkStateCache, UpdateStats},
    partial_witness_commitment,
    policy::LastNBlocksPolicy,
    witness::{
        accessed_to_state_targets, build_sidecar_targets, cache_hit_targets,
        measure_multiproof_size, state_targets_to_proof_targets, WitnessResult,
    },
    CacheAnchor, CacheAwareTransitionProgress, CacheAwareTrieTransition, CacheFootprintStats,
    PartialExecutionWitness, PartialExecutionWitnessState, PartialStatelessSidecar,
    PartialTrieNodeCache, RootWitnessCompletenessSummary, SidecarBenchmarkManifest, StateTargetSet,
    StateTargetStats, TrieProofTargetV2, WitnessReductionStats,
};
use reth_ethereum::{calculate_receipt_root_no_memo, EthPrimitives};
use reth_evm::{execute::Executor, ConfigureEvm};
use reth_primitives_traits::{AlloyBlockHeader, BlockTy, RecoveredBlock};
use reth_provider::{ProviderResult, StateProvider};
use reth_revm::database::StateProviderDatabase;
use reth_trie_common::{
    DecodedMultiProofV2, HashedPostState, MultiProofTargetsV2, ProofV2Target, TrieInput,
    EMPTY_ROOT_HASH,
};
use revm::database::State;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};
use tracing::{info, warn};

pub(crate) struct BuilderOptions<'a> {
    pub(crate) capture_dir: Option<&'a Path>,
    pub(crate) sidecar_dir: &'a Path,
    pub(crate) compute_baseline: bool,
    pub(crate) resource_metrics: bool,
    pub(crate) trie_cache_diagnostics: bool,
    pub(crate) run_sidecar_preflight: bool,
    pub(crate) validation_bench_output: Option<&'a Path>,
    pub(crate) builder_bench_output: Option<&'a Path>,
    pub(crate) force_previous_cache_snapshot: bool,
    pub(crate) reexec_limits: &'a SidecarReexecLimits,
    pub(crate) parallel_initial_proof: Option<&'a ParallelInitialProofFn<'a>>,
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
}

const PARALLEL_INITIAL_PROOF_MIN_STORAGE_TRIES: usize = 2;
const PARALLEL_INITIAL_PROOF_MIN_TOTAL_TARGETS: usize = 64;

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

fn sidecar_publication_anchors(
    prev_cache_anchor: Option<CacheAnchor>,
    next_cache_anchor: Option<CacheAnchor>,
) -> Option<(CacheAnchor, CacheAnchor)> {
    prev_cache_anchor.zip(next_cache_anchor)
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

#[derive(Clone, Debug, Default)]
struct V2TargetSet {
    accounts: BTreeMap<B256, u8>,
    storage: BTreeMap<(B256, B256), u8>,
}

impl V2TargetSet {
    fn insert(&mut self, target: TrieProofTargetV2) {
        match target {
            TrieProofTargetV2::Account { key, min_len } => {
                self.accounts
                    .entry(key)
                    .and_modify(|current| *current = (*current).min(min_len))
                    .or_insert(min_len);
            }
            TrieProofTargetV2::Storage { hashed_address, key, min_len } => {
                self.storage
                    .entry((hashed_address, key))
                    .and_modify(|current| *current = (*current).min(min_len))
                    .or_insert(min_len);
            }
        }
    }

    fn extend(&mut self, targets: impl IntoIterator<Item = TrieProofTargetV2>) {
        for target in targets {
            self.insert(target);
        }
    }

    fn with_zero_min_len(&self) -> Self {
        Self {
            accounts: self.accounts.keys().copied().map(|key| (key, 0)).collect(),
            storage: self.storage.keys().copied().map(|key| (key, 0)).collect(),
        }
    }

    /// Expands targets for the context-free flat wire format.
    ///
    /// A flattened storage node does not carry its account address. Including the parent account
    /// proof makes the storage root reachable from the state root, allowing a standalone decoder
    /// to recover the account/storage association. Native structured V2 proofs retain that
    /// association directly and do not need these additional account targets.
    fn with_flat_storage_context(&self) -> Self {
        let mut targets = self.with_zero_min_len();
        for &(hashed_address, _) in self.storage.keys() {
            targets.accounts.entry(hashed_address).or_insert(0);
        }
        targets
    }

    fn difference_and_record(&self, requested: &mut Self) -> Self {
        let mut delta = Self::default();
        for (&key, &min_len) in &self.accounts {
            if requested.accounts.get(&key).is_some_and(|current| *current <= min_len) {
                continue;
            }
            requested.accounts.insert(key, min_len);
            delta.accounts.insert(key, min_len);
        }
        for (&key, &min_len) in &self.storage {
            if requested.storage.get(&key).is_some_and(|current| *current <= min_len) {
                continue;
            }
            requested.storage.insert(key, min_len);
            delta.storage.insert(key, min_len);
        }
        delta
    }

    fn is_empty(&self) -> bool {
        self.accounts.is_empty() && self.storage.is_empty()
    }

    fn len(&self) -> usize {
        self.accounts.len() + self.storage.len()
    }

    fn distinct_storage_tries(&self) -> usize {
        let mut previous = None;
        let mut count = 0;
        for &(hashed_address, _) in self.storage.keys() {
            if previous != Some(hashed_address) {
                previous = Some(hashed_address);
                count += 1;
            }
        }
        count
    }

    fn should_use_parallel_initial_proof(&self) -> bool {
        self.distinct_storage_tries() >= PARALLEL_INITIAL_PROOF_MIN_STORAGE_TRIES &&
            self.len() >= PARALLEL_INITIAL_PROOF_MIN_TOTAL_TARGETS
    }

    fn to_provider_targets(&self) -> MultiProofTargetsV2 {
        let account_targets = self
            .accounts
            .iter()
            .map(|(&key, &min_len)| ProofV2Target::new(key).with_min_len(min_len))
            .collect();
        let mut storage_targets = B256Map::default();
        for (&(hashed_address, key), &min_len) in &self.storage {
            storage_targets
                .entry(hashed_address)
                .or_insert_with(Vec::new)
                .push(ProofV2Target::new(key).with_min_len(min_len));
        }
        MultiProofTargetsV2 { account_targets, storage_targets }
    }
}

#[derive(Debug)]
struct CacheAwareFlatBuild {
    nodes: Vec<Bytes>,
    decoded_proof: DecodedMultiProofV2,
    next_trie_cache: PartialTrieNodeCache,
    state_root: B256,
    provider_calls: usize,
    structural_rounds: usize,
    initial_targets: usize,
    initial_proof_nodes: usize,
    initial_proof_bytes: usize,
    structural_targets: usize,
    structural_account_targets: usize,
    structural_storage_targets: usize,
    cache_covered_mutation_targets: usize,
    trie_clone_us: u64,
    transition_us: u64,
    structural_provider_us: u64,
}

#[derive(Debug)]
struct CacheAwareBaseProof {
    targets: V2TargetSet,
    proof: DecodedMultiProofV2,
    cache_covered_mutation_targets: usize,
    provider_us: u64,
    proof_source: &'static str,
    parallel_storage_workers: usize,
    parallel_account_workers: usize,
}

fn initial_cache_aware_targets(
    post_state: &HashedPostState,
    miss: &MissResult,
    trie_cache: &PartialTrieNodeCache,
) -> (V2TargetSet, usize) {
    let mut targets = V2TargetSet::default();

    // Cache misses are value proofs, even if the mutation path happens to be cached locally.
    for address in &miss.missed_accounts {
        targets.insert(TrieProofTargetV2::Account { key: keccak256(address), min_len: 0 });
    }
    for (address, slot) in &miss.missed_storage {
        let hashed_address = keccak256(address);
        targets.insert(TrieProofTargetV2::Account { key: hashed_address, min_len: 0 });
        targets.insert(TrieProofTargetV2::Storage {
            hashed_address,
            key: keccak256(slot),
            min_len: 0,
        });
    }

    let mut mutation_target_count = 0usize;
    let mut uncovered_mutation_count = 0usize;
    let mut account_paths =
        post_state.accounts.keys().chain(post_state.storages.keys()).copied().collect::<Vec<_>>();
    account_paths.sort_unstable();
    account_paths.dedup();
    for hashed_address in account_paths {
        mutation_target_count += 1;
        if !trie_cache.contains_hashed_account_path(hashed_address) {
            uncovered_mutation_count += 1;
            targets.insert(TrieProofTargetV2::Account { key: hashed_address, min_len: 0 });
        }
    }
    for (&hashed_address, storage) in &post_state.storages {
        if storage.wiped {
            continue;
        }
        for &hashed_slot in storage.storage.keys() {
            mutation_target_count += 1;
            if !trie_cache.contains_hashed_storage_path(hashed_address, hashed_slot) {
                uncovered_mutation_count += 1;
                targets.insert(TrieProofTargetV2::Storage {
                    hashed_address,
                    key: hashed_slot,
                    min_len: 0,
                });
            }
        }
    }

    (targets, mutation_target_count.saturating_sub(uncovered_mutation_count))
}

fn generate_cache_aware_base_proof(
    state_provider: &dyn StateProvider,
    parallel_initial_proof: Option<&ParallelInitialProofFn<'_>>,
    post_state: &HashedPostState,
    miss: &MissResult,
    trie_cache: &PartialTrieNodeCache,
) -> eyre::Result<CacheAwareBaseProof> {
    let (mut targets, cache_covered_mutation_targets) =
        initial_cache_aware_targets(post_state, miss, trie_cache);
    if targets.is_empty() && trie_cache.state_root().is_none() {
        targets.insert(TrieProofTargetV2::Account { key: B256::ZERO, min_len: 0 });
    }

    let provider_start = Instant::now();
    let (proof, proof_source, parallel_storage_workers, parallel_account_workers) = if targets
        .is_empty()
    {
        (DecodedMultiProofV2::default(), "empty", 0, 0)
    } else if let Some(parallel_initial_proof) = parallel_initial_proof &&
        targets.should_use_parallel_initial_proof()
    {
        match parallel_initial_proof(targets.to_provider_targets()) {
            Ok(ParallelInitialProofOutput { proof, storage_workers, account_workers }) => {
                (proof, "parallel", storage_workers, account_workers)
            }
            Err(err) => {
                warn!(
                    target: "partial_stateless",
                    error = %err,
                    initial_targets = targets.len(),
                    distinct_storage_tries = targets.distinct_storage_tries(),
                    "Parallel initial V2 multiproof failed; retrying with the serial parent provider"
                );
                let proof = state_provider
                    .multiproof_v2(TrieInput::default(), targets.to_provider_targets())
                    .map_err(|serial_err| {
                        eyre::eyre!(
                            "parallel initial V2 multiproof failed ({err}); serial fallback also failed: {serial_err}"
                        )
                    })?;
                (proof, "serial-after-parallel-error", 0, 0)
            }
        }
    } else {
        let proof = state_provider
            .multiproof_v2(TrieInput::default(), targets.to_provider_targets())
            .map_err(|err| eyre::eyre!("failed to generate initial V2 multiproof: {err}"))?;
        let source = if parallel_initial_proof.is_some() { "serial-low-width" } else { "serial" };
        (proof, source, 0, 0)
    };
    let provider_us = provider_start.elapsed().as_micros() as u64;

    Ok(CacheAwareBaseProof {
        targets,
        proof,
        cache_covered_mutation_targets,
        provider_us,
        proof_source,
        parallel_storage_workers,
        parallel_account_workers,
    })
}

fn build_cache_aware_flat_transition(
    state_provider: &dyn StateProvider,
    parent_state_root: B256,
    expected_state_root: B256,
    post_state: HashedPostState,
    miss: &MissResult,
    trie_cache: &PartialTrieNodeCache,
    base: &CacheAwareBaseProof,
) -> eyre::Result<CacheAwareFlatBuild> {
    let clone_start = Instant::now();
    let mut next_trie_cache = trie_cache.clone();
    let trie_clone_us = clone_start.elapsed().as_micros() as u64;
    let mut requested_flat = base.targets.clone();
    let mut revealed_exact = base.targets.clone();
    let mut flat_nodes = B256Map::<Bytes>::default();
    base.proof.extend_flat_witness(&mut flat_nodes);
    let initial_proof_nodes = flat_nodes.len();
    let initial_proof_bytes = flat_nodes.values().map(|node| node.len()).sum();
    let mut accumulated_parent_proof = base.proof.clone();
    let mut provider_calls = (!base.proof.is_empty()) as usize;
    let initial_target_count = base.targets.len();
    let read_only_storage_targets =
        miss.missed_storage.iter().map(|(address, _)| keccak256(address)).collect::<Vec<_>>();
    let transition_start = Instant::now();
    let mut structural_rounds = 0usize;
    let mut structural_target_count = 0usize;
    let mut structural_account_target_count = 0usize;
    let mut structural_storage_target_count = 0usize;
    let mut structural_provider_us = 0u64;

    // Flat storage nodes have no address/path context. Fetch any missing parent account proofs
    // once so every storage root in the flat witness is independently reachable from the state
    // root. This flat-only overhead is reported with the structural proof metrics.
    let flat_base_targets = base.targets.with_flat_storage_context();
    let context_delta = flat_base_targets.difference_and_record(&mut requested_flat);
    if !context_delta.is_empty() {
        structural_target_count += context_delta.len();
        structural_account_target_count += context_delta.accounts.len();
        structural_storage_target_count += context_delta.storage.len();
        let provider_start = Instant::now();
        let proof = state_provider
            .multiproof_v2(TrieInput::default(), context_delta.to_provider_targets())
            .map_err(|err| {
                eyre::eyre!("failed to generate flat storage-context V2 proof delta: {err}")
            })?;
        structural_provider_us += provider_start.elapsed().as_micros() as u64;
        provider_calls += 1;
        proof.extend_flat_witness(&mut flat_nodes);
        accumulated_parent_proof.extend(proof);
    }

    let state_root = {
        let mut session = CacheAwareTrieTransition::new(
            &mut next_trie_cache,
            post_state,
            read_only_storage_targets,
        );
        if !base.proof.is_empty() {
            session
                .reveal(base.proof.clone())
                .map_err(|err| eyre::eyre!("failed to reveal initial V2 multiproof: {err}"))?;
        }
        loop {
            match session
                .advance()
                .map_err(|err| eyre::eyre!("cache-aware transition failed: {err}"))?
            {
                CacheAwareTransitionProgress::Complete(root) => break root,
                CacheAwareTransitionProgress::ProofRequired(targets) => {
                    if structural_rounds >= 128 {
                        return Err(eyre::eyre!(
                            "cache-aware transition exceeded 128 structural proof rounds"
                        ));
                    }
                    structural_rounds += 1;
                    let mut exact = V2TargetSet::default();
                    exact.extend(targets);
                    let flat = exact.with_flat_storage_context();
                    let delta = flat.difference_and_record(&mut requested_flat);
                    if !delta.is_empty() {
                        structural_target_count += delta.len();
                        structural_account_target_count += delta.accounts.len();
                        structural_storage_target_count += delta.storage.len();
                        let provider_start = Instant::now();
                        let proof = state_provider
                            .multiproof_v2(TrieInput::default(), delta.to_provider_targets())
                            .map_err(|err| {
                                eyre::eyre!(
                                    "failed to generate structural V2 multiproof delta: {err}"
                                )
                            })?;
                        structural_provider_us += provider_start.elapsed().as_micros() as u64;
                        provider_calls += 1;
                        proof.extend_flat_witness(&mut flat_nodes);
                        accumulated_parent_proof.extend(proof);
                    }

                    let reveal_delta = exact.difference_and_record(&mut revealed_exact);
                    if reveal_delta.is_empty() {
                        return Err(eyre::eyre!(
                            "cache-aware transition made no progress: all {} structural targets were already requested",
                            exact.len()
                        ));
                    }
                    let mut reveal_proof = accumulated_parent_proof.clone();
                    reveal_proof.retain_targets(&reveal_delta.to_provider_targets());
                    if reveal_proof.is_empty() {
                        return Err(eyre::eyre!(
                            "cache-aware transition structural proof delta was empty for {} targets",
                            reveal_delta.len()
                        ));
                    }
                    session.reveal(reveal_proof).map_err(|err| {
                        eyre::eyre!("failed to reveal structural V2 proof delta: {err}")
                    })?;
                }
            }
        }
    };
    let transition_us = transition_start.elapsed().as_micros() as u64;

    if state_root != expected_state_root {
        return Err(eyre::eyre!(
            "cache-aware sparse-trie root mismatch: expected {expected_state_root:?}, got {state_root:?}"
        ));
    }

    let mut nodes = flat_nodes.into_values().collect::<Vec<_>>();
    nodes.sort_unstable();
    nodes.dedup();
    let decoded_proof = decode_transition_witness(parent_state_root, &nodes)?;

    Ok(CacheAwareFlatBuild {
        nodes,
        decoded_proof,
        next_trie_cache,
        state_root,
        provider_calls,
        structural_rounds,
        initial_targets: initial_target_count,
        initial_proof_nodes,
        initial_proof_bytes,
        structural_targets: structural_target_count,
        structural_account_targets: structural_account_target_count,
        structural_storage_targets: structural_storage_target_count,
        cache_covered_mutation_targets: base.cache_covered_mutation_targets,
        trie_clone_us,
        transition_us,
        structural_provider_us,
    })
}

fn decode_transition_witness(
    parent_state_root: B256,
    nodes: &[Bytes],
) -> eyre::Result<DecodedMultiProofV2> {
    let mut witness = B256Map::with_capacity_and_hasher(nodes.len(), Default::default());
    for node in nodes {
        witness.insert(keccak256(node), node.clone());
    }
    if parent_state_root == EMPTY_ROOT_HASH {
        witness.entry(parent_state_root).or_insert_with(|| Bytes::from([EMPTY_STRING_CODE]));
    }
    DecodedMultiProofV2::from_witness(parent_state_root, &witness)
        .map_err(|err| eyre::eyre!("failed to decode canonical transition witness: {err}"))
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

fn measure_transition_witness_size(
    nodes: &[Bytes],
    proof: &DecodedMultiProofV2,
    bytecode_bytes: usize,
) -> WitnessResult {
    let mut storage_hashes = std::collections::HashSet::new();
    for storage_proof in proof.storage_proofs.values() {
        for proof_node in storage_proof {
            let mut encoded = Vec::new();
            proof_node.node.encode(&mut encoded);
            storage_hashes.insert(keccak256(&encoded));
        }
    }

    let (mut account_proof_bytes, mut storage_proof_bytes) = (0usize, 0usize);
    let (mut account_proof_nodes, mut storage_proof_nodes) = (0usize, 0usize);
    for node in nodes {
        if storage_hashes.contains(&keccak256(node)) {
            storage_proof_bytes += node.len();
            storage_proof_nodes += 1;
        } else {
            account_proof_bytes += node.len();
            account_proof_nodes += 1;
        }
    }

    WitnessResult {
        total_size_bytes: account_proof_bytes + storage_proof_bytes + bytecode_bytes,
        account_proof_bytes,
        storage_proof_bytes,
        bytecode_bytes,
        account_proof_nodes,
        storage_proof_nodes,
        target_accounts: 0,
        target_storage_slots: 0,
        computation_time_ms: None,
        cpu_time_ms: None,
        major_page_faults: None,
        minor_page_faults: None,
    }
}

#[derive(Debug)]
pub struct WeakStatelessBuild {
    pub sidecar: PartialStatelessSidecar,
    pub build_us: u64,
}

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
    let build_start = Instant::now();
    let parent_block_number = block_number.saturating_sub(1);
    let cold_cache = config.new_cache_at(parent_block_number);
    let full_miss = cold_cache.compute_miss(accessed);
    let cold_trie = PartialTrieNodeCache::new();
    let base = generate_cache_aware_base_proof(
        state_provider,
        None,
        hashed_post_state,
        &full_miss,
        &cold_trie,
    )?;
    let build = build_cache_aware_flat_transition(
        state_provider,
        parent_state_root,
        expected_state_root,
        hashed_post_state.clone(),
        &full_miss,
        &cold_trie,
        &base,
    )?;

    let (raw_targets, _) = build_sidecar_targets(&full_miss);
    let full_targets = StateTargetSet::from(&raw_targets);
    let mut codes = accessed.codes.iter().collect::<Vec<_>>();
    codes.sort_unstable_by_key(|(code_hash, _)| **code_hash);
    let codes = codes.into_iter().map(|(_, code)| code.clone()).collect::<Vec<_>>();
    let bytecode_bytes = codes.iter().map(|code| code.len()).sum();
    let mut stats =
        measure_transition_witness_size(&build.nodes, &build.decoded_proof, bytecode_bytes);
    stats.target_accounts = base.targets.accounts.len() + build.structural_account_targets;
    stats.target_storage_slots = base.targets.storage.len() + build.structural_storage_targets;
    stats.computation_time_ms = Some(build_start.elapsed().as_millis() as u64);

    let witness = PartialExecutionWitness {
        state: PartialExecutionWitnessState::MptTransitionNodes(build.nodes),
        codes,
        keys: raw_targets.key_preimages(),
        headers: ancestor_headers.to_vec(),
    };
    let cache_policy_id =
        last_n_blocks_cache_policy_id(config.account_window, config.storage_window);
    let prev_cache_anchor =
        cold_cache.cache_anchor(parent_block_number, parent_hash, cache_policy_id);
    let mut next_cache = config.new_cache_at(parent_block_number);
    next_cache.on_block_executed(block_number, accessed);
    let next_cache_anchor = next_cache.cache_anchor(block_number, block_hash, cache_policy_id);
    let witness_commitment = partial_witness_commitment(parent_state_root, &full_targets, &witness);

    Ok(WeakStatelessBuild {
        sidecar: PartialStatelessSidecar {
            parent_hash,
            parent_state_root,
            block_hash,
            block_number,
            cache_block: parent_block_number,
            cache_policy_id,
            prev_cache_anchor,
            next_cache_anchor,
            cache_policy_metadata: "WeakStateless(no persistent cache)".to_string(),
            cache_miss_targets: full_targets,
            witness_commitment,
            miss_manifest: raw_targets,
            witness,
            stats,
        },
        build_us: build_start.elapsed().as_micros() as u64,
    })
}

fn benchmark_one_sidecar_validation<Evm>(
    evm_config: &Evm,
    block: &RecoveredBlock<BlockTy<EthPrimitives>>,
    cache: &mut NetworkStateCache,
    trie_cache: &mut PartialTrieNodeCache,
    sidecar_bytes: &[u8],
    expected_cache_policy_id: B256,
    limits: &SidecarReexecLimits,
) -> eyre::Result<crate::sidecar_reexec::SidecarReexecReport>
where
    Evm: ConfigureEvm<Primitives = EthPrimitives>,
{
    let (sidecar, deserialize_us) = deserialize_sidecar_for_benchmark(sidecar_bytes)?;
    let mut report = verify_and_apply_trustless_sidecar_for_benchmark(
        evm_config,
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

#[expect(clippy::too_many_arguments)]
fn benchmark_sidecar_validation<Evm>(
    evm_config: &Evm,
    state_provider: &dyn StateProvider,
    block: &RecoveredBlock<BlockTy<EthPrimitives>>,
    prev_cache: &mut NetworkStateCache,
    trie_cache: &mut PartialTrieNodeCache,
    partial_sidecar: &PartialStatelessSidecar,
    hashed_post_state: &HashedPostState,
    accessed: &BlockAccessedState,
    ancestor_headers: &[Bytes],
    config: &CacheConfig,
    limits: &SidecarReexecLimits,
    output_path: &Path,
    historical_full_db_evm_us: u64,
    partial_witness_build_us: u64,
    historical_gas_used: u64,
    historical_receipts_root: B256,
    historical_requests_hash: B256,
    historical_requests_empty: bool,
    value_cache_bytes: usize,
) -> eyre::Result<crate::sidecar_reexec::SidecarReexecReport>
where
    Evm: ConfigureEvm<Primitives = EthPrimitives>,
{
    let WeakStatelessBuild { sidecar: weak_sidecar, build_us: weak_build_us } =
        build_weak_stateless_sidecar(
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
        )?;

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
    let (partial_report, weak_report) = if partial_first {
        let partial_report = benchmark_one_sidecar_validation(
            evm_config,
            block,
            prev_cache,
            trie_cache,
            &partial_bytes,
            expected_cache_policy_id,
            limits,
        )?;
        let weak_report = benchmark_one_sidecar_validation(
            evm_config,
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
            evm_config,
            block,
            &mut weak_cache,
            &mut weak_trie,
            &weak_bytes,
            expected_cache_policy_id,
            limits,
        )?;
        let partial_report = benchmark_one_sidecar_validation(
            evm_config,
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
    let partial_root = partial_report.trustless_state_root.unwrap_or_default();
    let weak_root = weak_report.trustless_state_root.unwrap_or_default();
    let expected_gas_used = block.header().gas_used();
    let expected_receipts_root = block.header().receipts_root();
    let expected_requests_hash = block.header().requests_hash();
    let requests_valid = expected_requests_hash.map_or_else(
        || {
            historical_requests_empty &&
                partial_report.execution_requests_empty &&
                weak_report.execution_requests_empty
        },
        |expected| {
            historical_requests_hash == expected &&
                partial_report.execution_requests_hash == expected &&
                weak_report.execution_requests_hash == expected
        },
    );
    let valid = partial_root == expected_root &&
        weak_root == expected_root &&
        historical_gas_used == expected_gas_used &&
        partial_report.execution_gas_used == expected_gas_used &&
        weak_report.execution_gas_used == expected_gas_used &&
        historical_receipts_root == expected_receipts_root &&
        partial_report.execution_receipts_root == expected_receipts_root &&
        weak_report.execution_receipts_root == expected_receipts_root &&
        requests_valid;
    let record = ValidationBenchmarkRecord {
        schema_version: 2,
        block_number: block.number(),
        block_hash: block.hash(),
        gas_used: expected_gas_used,
        historical_gas_used,
        tx_count: block.transaction_count(),
        verifier_order: if partial_first { "partial-then-weak" } else { "weak-then-partial" },
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
        trie_cache_bytes: trie_cache.estimated_memory_bytes(),
        expected_state_root: expected_root,
        partial_state_root: partial_root,
        weak_state_root: weak_root,
        expected_receipts_root,
        historical_receipts_root,
        partial_receipts_root: partial_report.execution_receipts_root,
        weak_receipts_root: weak_report.execution_receipts_root,
        expected_requests_hash,
        historical_requests_hash,
        partial_requests_hash: partial_report.execution_requests_hash,
        weak_requests_hash: weak_report.execution_requests_hash,
        valid,
    };
    append_record(output_path, &record)?;
    info!(
        target: "partial_stateless_bench",
        benchmark = "paired_validation",
        block = block.number(),
        block_hash = ?block.hash(),
        verifier_order = record.verifier_order,
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
            partial_report.execution_gas_used,
            weak_report.execution_gas_used,
        ));
    }

    Ok(partial_report)
}
pub fn create_sidecar_for_block<Evm, ParentStateRootFn, AncestorHeadersFn>(
    evm_config: &Evm,
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
    ParentStateRootFn: FnOnce(B256) -> eyre::Result<B256>,
    AncestorHeadersFn: FnOnce(Option<u64>, u64) -> eyre::Result<Vec<Bytes>>,
{
    let block_number = block.number();
    let builder_total_start = Instant::now();
    let parent_block_number = block_number.saturating_sub(1);

    let historical_execution_start = Instant::now();
    let state_provider_db = StateProviderDatabase::new(state_provider);
    let mut db = State::builder().with_bundle_update().with_database(state_provider_db).build();
    let block_executor = evm_config.executor(&mut db);

    let mut accessed = BlockAccessedState::default();
    let mut lowest_block_number = None;
    let execution_output = block_executor
        .execute_with_state_closure(block, |statedb: &State<_>| {
            accessed = BlockAccessedState::from_simulated_state(statedb);
            lowest_block_number = statedb.block_hashes.lowest().map(|(num, _)| num);
        })
        .map_err(|err| eyre::eyre!("simulation failed for block: {err}"))?;
    let historical_full_db_evm_us = historical_execution_start.elapsed().as_micros() as u64;
    let historical_gas_used = execution_output.result.gas_used;
    let historical_receipts_root =
        calculate_receipt_root_no_memo(&execution_output.result.receipts);
    let historical_requests_hash = execution_output.result.requests.requests_hash();
    let historical_requests_empty = execution_output.result.requests.is_empty();

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
    let mut builder_witness_commitment = None;
    let mut sidecar_constructed = false;
    let witness = {
        let base = generate_cache_aware_base_proof(
            state_provider,
            options.parallel_initial_proof,
            &hashed_post_state,
            &miss,
            trie_cache,
        )
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
            transition_us: local_root_us,
            structural_provider_us,
        } = build_cache_aware_flat_transition(
            state_provider,
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
        result.target_accounts = base.targets.accounts.len() + structural_account_targets;
        result.target_storage_slots = base.targets.storage.len() + structural_storage_targets;
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
            let Some((prev_cache_anchor, next_cache_anchor)) =
                sidecar_publication_anchors(prev_cache_anchor, next_cache_anchor)
            else {
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

            let ancestor_headers = ancestor_headers_for_range(lowest_block_number, block_number)?;
            // The flat canonical witness contains every parent-state node required for
            // both cache misses and the post-state trie transition.
            let sidecar_miss = StateTargetSet::from(&raw_targets);
            let witness_payload = PartialExecutionWitness {
                state: witness_state.clone(),
                codes: missed_bytecodes.clone(),
                keys: raw_targets.key_preimages(),
                headers: ancestor_headers.clone(),
            };
            let witness_commitment =
                partial_witness_commitment(parent_state_root, &sidecar_miss, &witness_payload);
            builder_witness_commitment = Some(witness_commitment);
            sidecar_constructed = true;
            let sidecar = PartialStatelessSidecar {
                parent_hash,
                parent_state_root,
                block_hash,
                block_number,
                cache_block: parent_block_number,
                cache_policy_id,
                prev_cache_anchor,
                next_cache_anchor,
                cache_policy_metadata: cache_policy_metadata.clone(),
                cache_miss_targets: sidecar_miss.clone(),
                witness_commitment,
                miss_manifest: raw_targets.clone(),
                witness: witness_payload,
                stats: result.clone(),
            };

            let root_witness_completeness = if options.run_sidecar_preflight {
                let prev_cache_for_reexec =
                    match require_previous_cache_snapshot(prev_cache_for_reexec.as_mut()) {
                        Ok(snapshot) => snapshot,
                        Err(err) => break 'sidecar Err(err),
                    };
                let reexec_report = if let Some(output_path) = options.validation_bench_output {
                    benchmark_sidecar_validation(
                        evm_config,
                        state_provider,
                        block,
                        prev_cache_for_reexec,
                        trie_cache,
                        &sidecar,
                        &hashed_post_state,
                        &accessed,
                        &ancestor_headers,
                        config,
                        options.reexec_limits,
                        output_path,
                        historical_full_db_evm_us,
                        transition_witness_build_us,
                        historical_gas_used,
                        historical_receipts_root,
                        historical_requests_hash,
                        historical_requests_empty,
                        cache_memory_before,
                    )
                } else {
                    verify_and_apply_provider_assisted_sidecar(
                        evm_config,
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
                    !reexec_report.root_witness_completeness.trustless_root_ready
                {
                    warn!(
                        target: "partial_stateless",
                        block = block_number,
                        partial_state_trustless_verification_ready = false,
                        missing_account_paths = reexec_report
                            .root_witness_completeness
                            .missing_account_paths
                            .len(),
                        missing_storage_paths = reexec_report
                            .root_witness_completeness
                            .missing_storage_paths
                            .len(),
                        "Partial-state node trustless verification is not ready; current state_root check is provider-assisted"
                    );
                }
                info!(
                    target: "partial_stateless",
                    block = block_number,
                    partial_state_trustless_verification_ready = reexec_report
                        .root_witness_completeness
                        .trustless_root_ready,
                    computed_state_root = ?reexec_report.computed_state_root,
                    reexec_accounts = reexec_report.actual_accessed.accounts.len(),
                    reexec_storage = reexec_report.actual_accessed.storage.len(),
                    reexec_codes = reexec_report.actual_accessed.codes.len(),
                    expected_miss_accounts = reexec_report.expected_miss.accounts.len(),
                    expected_miss_storage = reexec_report.expected_miss.storage.len(),
                    expected_miss_codes = reexec_report.expected_miss.code_hashes.len(),
                    next_cache_root = ?reexec_report.next_cache_anchor.cache_root,
                    "Sidecar preflight succeeded"
                );
                match reexec_report.trustless_state_root {
                    Some(root) if root == block.state_root() => info!(
                        target: "partial_stateless",
                        block = block_number,
                        trustless_state_root = ?root,
                        "Trustless state root VERIFIED (trie node cache + witness only)"
                    ),
                    Some(root) => warn!(
                        target: "partial_stateless",
                        block = block_number,
                        trustless_state_root = ?root,
                        expected = ?block.state_root(),
                        "Trustless state root MISMATCH — trie cache/witness stale or insufficient"
                    ),
                    None => info!(
                        target: "partial_stateless",
                        block = block_number,
                        trie_warm_nodes = trie_cache.warm_node_count(),
                        tracked_accounts = trie_cache.tracked_account_count(),
                        "Trustless state root unavailable — trie node cache not warm enough this block (blind path)"
                    ),
                }
                RootWitnessCompletenessSummary::from_report(
                    &reexec_report.root_witness_completeness,
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
                sidecar_miss: StateTargetStats::from_targets(&sidecar_miss),
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
        *trie_cache = next_trie_cache;
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
            schema_version: 1,
            block_number,
            block_hash,
            historical_full_db_evm_us,
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
        builder_total_us,
        snapshot_created,
        snapshot_us,
        initial_proof_source = builder_initial_proof_source,
        initial_provider_us = builder_initial_provider_us,
        sidecar_published = saved_sidecar_path.is_some(),
        "Builder sidecar end-to-end benchmark"
    );

    Ok(BuilderBlockReport { cache_update: stats, witness, sidecar_path: saved_sidecar_path })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::{Address, U256};
    use partial_stateless::policy::AccountData;

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
    fn sidecar_publication_depends_only_on_cache_anchors() {
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

        assert_eq!(sidecar_publication_anchors(Some(prev), Some(next)), Some((prev, next)));
        assert_eq!(sidecar_publication_anchors(None, Some(next)), None);
        assert_eq!(sidecar_publication_anchors(Some(prev), None), None);
    }

    #[test]
    fn requested_preflight_fails_closed_without_previous_cache_snapshot() {
        assert!(require_previous_cache_snapshot(None).is_err());
    }

    #[test]
    fn parallel_initial_proof_requires_both_storage_width_and_total_work() {
        let storage_a = B256::repeat_byte(0xaa);
        let storage_b = B256::repeat_byte(0xbb);
        let mut too_little_work = V2TargetSet::default();
        too_little_work.insert(TrieProofTargetV2::Storage {
            hashed_address: storage_a,
            key: B256::repeat_byte(0x01),
            min_len: 0,
        });
        too_little_work.insert(TrieProofTargetV2::Storage {
            hashed_address: storage_b,
            key: B256::repeat_byte(0x02),
            min_len: 0,
        });
        assert_eq!(too_little_work.distinct_storage_tries(), 2);
        assert!(!too_little_work.should_use_parallel_initial_proof());

        let mut wide = V2TargetSet::default();
        for index in 0..62u8 {
            let mut key = [0u8; 32];
            key[31] = index;
            wide.insert(TrieProofTargetV2::Account { key: B256::from(key), min_len: 0 });
        }
        wide.insert(TrieProofTargetV2::Storage {
            hashed_address: storage_a,
            key: B256::repeat_byte(0x01),
            min_len: 0,
        });
        wide.insert(TrieProofTargetV2::Storage {
            hashed_address: storage_b,
            key: B256::repeat_byte(0x02),
            min_len: 0,
        });
        assert_eq!(wide.len(), PARALLEL_INITIAL_PROOF_MIN_TOTAL_TARGETS);
        assert!(wide.should_use_parallel_initial_proof());

        wide.storage.remove(&(storage_b, B256::repeat_byte(0x02)));
        wide.insert(TrieProofTargetV2::Storage {
            hashed_address: storage_a,
            key: B256::repeat_byte(0x03),
            min_len: 0,
        });
        assert_eq!(wide.distinct_storage_tries(), 1);
        assert!(!wide.should_use_parallel_initial_proof());
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

    #[test]
    fn v2_target_difference_handles_accounts_storage_duplicates_and_min_len() {
        let account = B256::repeat_byte(0x11);
        let storage_account = B256::repeat_byte(0x22);
        let slot = B256::repeat_byte(0x33);
        let mut requested = V2TargetSet::default();
        requested.insert(TrieProofTargetV2::Account { key: account, min_len: 8 });
        requested.insert(TrieProofTargetV2::Storage {
            hashed_address: storage_account,
            key: slot,
            min_len: 12,
        });

        let mut desired = V2TargetSet::default();
        desired.extend([
            TrieProofTargetV2::Account { key: account, min_len: 10 },
            TrieProofTargetV2::Account { key: account, min_len: 10 },
            TrieProofTargetV2::Storage { hashed_address: storage_account, key: slot, min_len: 5 },
        ]);
        let delta = desired.difference_and_record(&mut requested);

        assert!(delta.accounts.is_empty(), "shallower account proof already covers target");
        assert_eq!(delta.storage.get(&(storage_account, slot)), Some(&5));
        assert_eq!(requested.storage.get(&(storage_account, slot)), Some(&5));
        assert!(desired.difference_and_record(&mut requested).is_empty());
    }

    #[test]
    fn flat_target_normalization_is_deterministic() {
        let account = B256::repeat_byte(0x44);
        let storage_account = B256::repeat_byte(0x55);
        let slot = B256::repeat_byte(0x66);
        let mut targets = V2TargetSet::default();
        targets.insert(TrieProofTargetV2::Storage {
            hashed_address: storage_account,
            key: slot,
            min_len: 19,
        });
        targets.insert(TrieProofTargetV2::Account { key: account, min_len: 7 });

        let flat = targets.with_flat_storage_context();
        assert_eq!(flat.accounts.get(&account), Some(&0));
        assert_eq!(flat.accounts.get(&storage_account), Some(&0));
        assert_eq!(flat.storage.get(&(storage_account, slot)), Some(&0));
    }
}
