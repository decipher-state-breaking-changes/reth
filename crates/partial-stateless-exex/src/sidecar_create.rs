use crate::{
    format_bytes,
    sidecar_io::sidecar_path,
    sidecar_reexec::{
        verify_and_apply_provider_assisted_sidecar, SidecarReexecLimits, TrieCacheDisposition,
    },
    thread_rusage, CacheConfig,
};
use alloy_primitives::{keccak256, map::B256Map, Bytes, B256};
use alloy_rlp::{Decodable, Encodable, EMPTY_STRING_CODE};
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
    CacheAwareTransitionProgress, CacheAwareTrieTransition, CacheFootprintStats,
    PartialExecutionWitness, PartialExecutionWitnessState, PartialStatelessSidecar,
    PartialTrieNodeCache, RootWitnessCompletenessSummary, SidecarBenchmarkManifest, StateTargetSet,
    StateTargetStats, TrieProofTargetV2, WitnessReductionStats,
};
use reth_ethereum::EthPrimitives;
use reth_evm::{execute::Executor, ConfigureEvm};
use reth_primitives_traits::{AlloyBlockHeader, BlockTy, RecoveredBlock};
use reth_provider::StateProvider;
use reth_revm::database::StateProviderDatabase;
use reth_trie_common::{
    DecodedMultiProofV2, HashedPostState, MultiProofTargetsV2, ProofV2Target, TrieInput,
    TrieNodeV2, EMPTY_ROOT_HASH,
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
    pub(crate) reexec_limits: &'a SidecarReexecLimits,
}

#[derive(Debug)]
pub(crate) struct BuilderBlockReport {
    pub(crate) cache_update: UpdateStats,
    pub(crate) witness: Option<WitnessResult>,
    pub(crate) sidecar_path: Option<PathBuf>,
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
                continue
            }
            requested.accounts.insert(key, min_len);
            delta.accounts.insert(key, min_len);
        }
        for (&key, &min_len) in &self.storage {
            if requested.storage.get(&key).is_some_and(|current| *current <= min_len) {
                continue
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
            continue
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
    let proof = if targets.is_empty() {
        DecodedMultiProofV2::default()
    } else {
        state_provider
            .multiproof_v2(TrieInput::default(), targets.to_provider_targets())
            .map_err(|err| eyre::eyre!("failed to generate initial V2 multiproof: {err}"))?
    };
    let provider_us = provider_start.elapsed().as_micros() as u64;

    Ok(CacheAwareBaseProof { targets, proof, cache_covered_mutation_targets, provider_us })
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
                        ))
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
                        ))
                    }
                    let mut reveal_proof = accumulated_parent_proof.clone();
                    reveal_proof.retain_targets(&reveal_delta.to_provider_targets());
                    if reveal_proof.is_empty() {
                        return Err(eyre::eyre!(
                            "cache-aware transition structural proof delta was empty for {} targets",
                            reveal_delta.len()
                        ))
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
        ))
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

#[derive(Debug, Default)]
struct TransitionNodeKindMetrics {
    physical_nodes: usize,
    physical_bytes: usize,
    empty_roots: usize,
    leaves: usize,
    extensions: usize,
    branches: usize,
    fused_extension_branches: usize,
    undecodable_nodes: usize,
}

impl TransitionNodeKindMetrics {
    fn record(&mut self, encoded: &[u8]) {
        self.physical_nodes += 1;
        self.physical_bytes += encoded.len();
        let mut input = encoded;
        match TrieNodeV2::decode(&mut input) {
            Ok(TrieNodeV2::EmptyRoot) => self.empty_roots += 1,
            Ok(TrieNodeV2::Leaf(_)) => self.leaves += 1,
            Ok(TrieNodeV2::Extension(_)) => self.extensions += 1,
            Ok(TrieNodeV2::Branch(branch)) => {
                self.branches += 1;
                if !branch.key.is_empty() {
                    self.extensions += 1;
                    self.fused_extension_branches += 1;
                }
            }
            Err(_) => self.undecodable_nodes += 1,
        }
    }
}

fn measure_transition_node_kinds(
    nodes: &[Bytes],
    proof: &DecodedMultiProofV2,
) -> (TransitionNodeKindMetrics, TransitionNodeKindMetrics) {
    let mut storage_hashes = std::collections::HashSet::new();
    for storage_proof in proof.storage_proofs.values() {
        for proof_node in storage_proof {
            let mut encoded = Vec::new();
            proof_node.node.encode(&mut encoded);
            storage_hashes.insert(keccak256(&encoded));
        }
    }

    let mut account = TransitionNodeKindMetrics::default();
    let mut storage = TransitionNodeKindMetrics::default();
    for node in nodes {
        if storage_hashes.contains(&keccak256(node)) {
            storage.record(node);
        } else {
            account.record(node);
        }
    }
    (account, storage)
}

pub(crate) fn create_sidecar_for_block<Evm, ParentStateRootFn, AncestorHeadersFn>(
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
    let parent_block_number = block_number.saturating_sub(1);

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
    let mut prev_cache_for_reexec = cache_parent_synced.then(|| {
        NetworkStateCache::restore(
            cache.accounts().clone(),
            cache.storage().clone(),
            cache.codes().clone(),
            cache.current_block(),
            Box::new(LastNBlocksPolicy::new(config.account_window)),
            Box::new(LastNBlocksPolicy::new(config.storage_window)),
        )
    });
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

    let rusage_before = options.resource_metrics.then(thread_rusage);
    let start = Instant::now();
    let saved_sidecar_path;
    let witness = {
        let base =
            generate_cache_aware_base_proof(state_provider, &hashed_post_state, &miss, trie_cache)
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
        let elapsed_ms = start.elapsed().as_millis() as u64;
        let mut result = measure_transition_witness_size(&nodes, &proof, missed_bytecode_bytes);
        let (account_node_kinds, storage_node_kinds) =
            measure_transition_node_kinds(&nodes, &proof);
        let witness_state = PartialExecutionWitnessState::MptTransitionNodes(nodes);
        result.computation_time_ms = Some(elapsed_ms);
        if let Some((cpu_us_before, majflt_before, minflt_before)) = rusage_before {
            let (cpu_us_after, majflt_after, minflt_after) = thread_rusage();
            result.cpu_time_ms = Some(cpu_us_after.saturating_sub(cpu_us_before) / 1000);
            result.major_page_faults = Some(majflt_after.saturating_sub(majflt_before));
            result.minor_page_faults = Some(minflt_after.saturating_sub(minflt_before));
        }
        result.target_accounts = base.targets.accounts.len() + structural_account_targets;
        result.target_storage_slots = base.targets.storage.len() + structural_storage_targets;
        info!(
            target: "partial_stateless",
            block = block_number,
            transition_witness_nodes = result.account_proof_nodes + result.storage_proof_nodes,
            initial_targets,
            structural_targets,
            structural_rounds,
            provider_calls,
            structural_provider_us,
            initial_provider_us,
            initial_proof_nodes,
            initial_proof_bytes,
            cache_covered_mutation_targets,
            account_branch_nodes = account_node_kinds.branches,
            account_extension_nodes = account_node_kinds.extensions,
            account_leaf_nodes = account_node_kinds.leaves,
            account_fused_extension_branches = account_node_kinds.fused_extension_branches,
            account_node_bytes = account_node_kinds.physical_bytes,
            storage_branch_nodes = storage_node_kinds.branches,
            storage_extension_nodes = storage_node_kinds.extensions,
            storage_leaf_nodes = storage_node_kinds.leaves,
            storage_fused_extension_branches = storage_node_kinds.fused_extension_branches,
            storage_node_bytes = storage_node_kinds.physical_bytes,
            undecodable_transition_nodes = account_node_kinds.undecodable_nodes + storage_node_kinds.undecodable_nodes,
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
            let (Some(prev_cache_anchor), Some(next_cache_anchor), Some(prev_cache_for_reexec)) =
                (prev_cache_anchor, next_cache_anchor, prev_cache_for_reexec.as_mut())
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
                let reexec_report = verify_and_apply_provider_assisted_sidecar(
                    evm_config,
                    state_provider,
                    block,
                    prev_cache_for_reexec,
                    &sidecar,
                    options.reexec_limits,
                    trie_cache,
                    TrieCacheDisposition::Discard,
                )
                .map_err(|err| eyre::eyre!("provider-assisted sidecar preflight failed: {err}"))?;

                if !reexec_report.root_witness_completeness.trustless_root_ready {
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
                    "Provider-assisted sidecar preflight succeeded"
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

            fs::create_dir_all(options.sidecar_dir)
                .map_err(|err| eyre::eyre!("failed to create sidecar directory: {err}"))?;
            let sidecar_path = sidecar_path(options.sidecar_dir, block_number, block_hash);
            let sidecar_bytes = bincode::serialize(&sidecar)
                .map_err(|err| eyre::eyre!("failed to serialize sidecar: {err}"))?;
            fs::write(&sidecar_path, sidecar_bytes).map_err(|err| {
                eyre::eyre!("failed to write sidecar file {:?}: {err}", sidecar_path)
            })?;
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
                account_cached_branch_nodes = metrics.account_node_kinds.branches,
                account_cached_extension_nodes = metrics.account_node_kinds.extensions,
                account_cached_leaf_nodes = metrics.account_node_kinds.leaves,
                account_cached_blinded_children = metrics.account_node_kinds.blinded_children,
                storage_cached_branch_nodes = metrics.storage_node_kinds.branches,
                storage_cached_extension_nodes = metrics.storage_node_kinds.extensions,
                storage_cached_leaf_nodes = metrics.storage_node_kinds.leaves,
                storage_cached_blinded_children = metrics.storage_node_kinds.blinded_children,
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

    Ok(BuilderBlockReport { cache_update: stats, witness, sidecar_path: saved_sidecar_path })
}

#[cfg(test)]
mod tests {
    use super::*;

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
