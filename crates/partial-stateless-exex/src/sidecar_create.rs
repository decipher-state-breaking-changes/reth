use crate::{
    format_bytes,
    sidecar_io::sidecar_path,
    sidecar_reexec::{
        build_transition_structural_witness, complete_transition_multiproof,
        verify_and_apply_trustless_sidecar, SidecarReexecLimits,
    },
    thread_rusage, CacheConfig,
};
use alloy_primitives::{Bytes, B256};
use partial_stateless::{
    accessed_state::BlockAccessedState,
    fixture::{save_fixture, AccessedStateFixture},
    last_n_blocks_cache_policy_id,
    network_cache::{NetworkStateCache, UpdateStats},
    partial_witness_commitment,
    policy::LastNBlocksPolicy,
    witness::{
        accessed_to_state_targets, build_sidecar_targets, cache_hit_targets,
        measure_multiproof_size, state_targets_to_proof_targets, WitnessResult,
    },
    witness_check::write_state_targets_from_bundle,
    CacheFootprintStats, PartialExecutionWitness, PartialExecutionWitnessState,
    PartialStatelessSidecar, SerializableMultiProof, SidecarBenchmarkManifest, StateTargetSet,
    StateTargetStats, WitnessReductionStats,
};
use reth_ethereum::EthPrimitives;
use reth_evm::{execute::Executor, ConfigureEvm};
use reth_primitives_traits::{AlloyBlockHeader, BlockTy, RecoveredBlock};
use reth_provider::StateProvider;
use reth_revm::database::StateProviderDatabase;
use reth_trie_common::TrieInput;
use revm::database::State;
use std::{
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

fn align_cache_to_parent(cache: &mut NetworkStateCache, parent_block_number: u64) -> bool {
    let parent_synced = cache.current_block() == parent_block_number;
    if !parent_synced {
        cache.reset();
    }
    parent_synced
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn create_sidecar_for_block<Evm, ParentStateRootFn, AncestorHeadersFn>(
    evm_config: &Evm,
    state_provider: &dyn StateProvider,
    block: &RecoveredBlock<BlockTy<EthPrimitives>>,
    cache: &mut NetworkStateCache,
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
    let parent_state_root = parent_state_root_result?;

    let cache_policy_id =
        last_n_blocks_cache_policy_id(config.account_window, config.storage_window);
    let cache_policy_metadata = format!(
        "LastNBlocks(account: {}, storage/code: {})",
        config.account_window, config.storage_window
    );
    let cache_block_before = cache.current_block();
    let cache_parent_synced = align_cache_to_parent(cache, parent_block_number);
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
    let next_cache_anchor =
        cache_parent_synced.then(|| cache.cache_anchor(block_number, block_hash, cache_policy_id));
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
    let sidecar_miss = StateTargetSet::from(&raw_targets);
    let write_targets = write_state_targets_from_bundle(&execution_output.state);
    let mut proof_state_targets = sidecar_miss.clone();
    proof_state_targets.extend(&write_targets);
    // Cold paths authenticate values absent from the cache. Warm paths are only needed when the
    // replay changes them and the sparse trie must recompute their hashes up to the state root.
    let proof_targets = state_targets_to_proof_targets(&proof_state_targets);
    let target_accounts = proof_targets.len();
    let target_slots: usize = proof_targets.values().map(|slots| slots.len()).sum();
    info!(
        target: "partial_stateless",
        block = block_number,
        write_accounts = write_targets.accounts.len(),
        write_storage = write_targets.storage.len(),
        proof_accounts = target_accounts,
        proof_storage = target_slots,
        "Trustless proof targets (cold misses plus writes)"
    );
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

    if !cache_parent_synced {
        info!(
            target: "partial_stateless",
            block = block_number,
            cache_accounts = snapshot.total_accounts,
            cache_storage = snapshot.total_storage_slots,
            cache_codes = snapshot.total_codes,
            "Cache primed from an unsynchronized block; proof generation starts with the next contiguous block"
        );
        return Ok(BuilderBlockReport { cache_update: stats, witness: None, sidecar_path: None })
    }

    let rusage_before = options.resource_metrics.then(thread_rusage);
    let start = Instant::now();
    let proof_result = state_provider.multiproof(TrieInput::default(), proof_targets);
    let structural_witness = match &proof_result {
        Ok(_) => Some(
            build_transition_structural_witness(
                state_provider,
                parent_state_root,
                &execution_output.state,
            )
            .map_err(|err| rollback_sidecar_transition(cache, block_number, err))?,
        ),
        Err(_) => None,
    };
    let generation_elapsed_ms = start.elapsed().as_millis() as u64;
    let rusage_after_generation = options.resource_metrics.then(thread_rusage);

    let full_sidecar_baseline_stats: Option<WitnessResult> = if options.compute_baseline {
        let Some(structural_witness) = structural_witness.as_ref() else {
            return Err(rollback_sidecar_transition(
                cache,
                block_number,
                eyre::eyre!("cannot compute full baseline without the transition proof"),
            ));
        };
        let mut full_state_targets = accessed_targets.clone();
        full_state_targets.extend(&write_targets);
        let full_proof_targets = state_targets_to_proof_targets(&full_state_targets);
        let full_target_accounts = full_proof_targets.len();
        let full_target_slots: usize = full_proof_targets.values().map(|slots| slots.len()).sum();
        let full_bytecode_bytes: usize = accessed.codes.values().map(|bytes| bytes.len()).sum();
        let full_start = Instant::now();
        match state_provider.multiproof(TrieInput::default(), full_proof_targets) {
            Ok(full_proof) => match complete_transition_multiproof(
                parent_state_root,
                &execution_output.state,
                full_proof,
                structural_witness,
            ) {
                Ok(completed) if completed.state_root == block.header().state_root => {
                    let elapsed_ms = full_start.elapsed().as_millis() as u64;
                    let mut full_result =
                        measure_multiproof_size(&completed.multiproof, full_bytecode_bytes);
                    full_result.computation_time_ms = Some(elapsed_ms);
                    full_result.target_accounts = full_target_accounts;
                    full_result.target_storage_slots = full_target_slots;
                    Some(full_result)
                }
                Ok(completed) => {
                    warn!(
                        target: "partial_stateless",
                        block = block_number,
                        expected = ?block.header().state_root,
                        actual = ?completed.state_root,
                        "Full sidecar baseline produced wrong post-state root"
                    );
                    None
                }
                Err(err) => {
                    warn!(
                        target: "partial_stateless",
                        block = block_number,
                        error = %err,
                        "Failed to structurally complete full sidecar baseline multiproof"
                    );
                    None
                }
            },
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

    let mut saved_sidecar_path = None;
    let witness = match proof_result {
        Ok(proof) => {
            let completed = complete_transition_multiproof(
                parent_state_root,
                &execution_output.state,
                proof,
                structural_witness
                    .as_ref()
                    .expect("structural witness exists when the base proof succeeded"),
            )
            .map_err(|err| rollback_sidecar_transition(cache, block_number, err))?;
            if completed.state_root != block.header().state_root {
                return Err(eyre::eyre!(
                    "completed sidecar proof produced wrong post-state root: expected {:?}, got {:?}",
                    block.header().state_root,
                    completed.state_root
                ));
            }
            let proof = completed.multiproof;
            let mut result = measure_multiproof_size(&proof, missed_bytecode_bytes);
            result.computation_time_ms = Some(generation_elapsed_ms);
            if let (
                Some((cpu_us_before, majflt_before, minflt_before)),
                Some((cpu_us_after, majflt_after, minflt_after)),
            ) = (rusage_before, rusage_after_generation)
            {
                result.cpu_time_ms = Some(cpu_us_after.saturating_sub(cpu_us_before) / 1000);
                result.major_page_faults = Some(majflt_after.saturating_sub(majflt_before));
                result.minor_page_faults = Some(minflt_after.saturating_sub(minflt_before));
            }
            result.target_accounts = target_accounts;
            result.target_storage_slots = target_slots;

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

                let ancestor_headers =
                    ancestor_headers_for_range(lowest_block_number, block_number)?;
                let serializable_proof = SerializableMultiProof::from_multiproof(&proof);
                let serialized_multiproof = bincode::serialize(&serializable_proof)
                    .map_err(|err| eyre::eyre!("failed to serialize multiproof: {err}"))?;
                let witness_payload = PartialExecutionWitness {
                    state: PartialExecutionWitnessState::MptMultiProof(serialized_multiproof),
                    codes: missed_bytecodes.clone(),
                    keys: raw_targets.key_preimages(),
                    headers: ancestor_headers,
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
                let sidecar_bytes = bincode::serialize(&sidecar)
                    .map_err(|err| eyre::eyre!("failed to serialize sidecar: {err}"))?;

                let partial_state_trustless_verification_ready = if options.run_sidecar_preflight {
                    let wire_sidecar: PartialStatelessSidecar =
                        bincode::deserialize(&sidecar_bytes).map_err(|err| {
                            eyre::eyre!("failed to deserialize serialized sidecar preflight: {err}")
                        })?;
                    let reexec_report = verify_and_apply_trustless_sidecar(
                        evm_config,
                        block,
                        parent_state_root,
                        prev_cache_for_reexec,
                        &wire_sidecar,
                        options.reexec_limits,
                    )
                    .map_err(|err| eyre::eyre!("trustless sidecar preflight failed: {err}"))?;

                    info!(
                        target: "partial_stateless",
                        block = block_number,
                        partial_state_trustless_verification_ready = true,
                        computed_state_root = ?reexec_report.computed_state_root,
                        reexec_accounts = reexec_report.actual_accessed.accounts.len(),
                        reexec_storage = reexec_report.actual_accessed.storage.len(),
                        reexec_codes = reexec_report.actual_accessed.codes.len(),
                        expected_miss_accounts = reexec_report.expected_miss.accounts.len(),
                        expected_miss_storage = reexec_report.expected_miss.storage.len(),
                        expected_miss_codes = reexec_report.expected_miss.code_hashes.len(),
                        write_accounts = reexec_report.write_targets.accounts.len(),
                        write_storage = reexec_report.write_targets.storage.len(),
                        next_cache_root = ?reexec_report.next_cache_anchor.cache_root,
                        "Trustless sidecar preflight succeeded"
                    );
                    true
                } else {
                    false
                };

                fs::create_dir_all(options.sidecar_dir)
                    .map_err(|err| eyre::eyre!("failed to create sidecar directory: {err}"))?;
                let sidecar_path = sidecar_path(options.sidecar_dir, block_number, block_hash);
                fs::write(&sidecar_path, &sidecar_bytes).map_err(|err| {
                    eyre::eyre!("failed to write sidecar file {:?}: {err}", sidecar_path)
                })?;
                let sidecar_bytes_len =
                    fs::metadata(&sidecar_path).map(|m| m.len() as usize).unwrap_or(0);
                let manifest = SidecarBenchmarkManifest {
                    schema_version: 2,
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
                    write_targets: StateTargetStats::from_targets(&write_targets),
                    proof_targets: StateTargetStats::from_targets(&proof_state_targets),
                    trustless_preflight: options.run_sidecar_preflight,
                    partial_state_trustless_verification_ready,
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
                Err(e) if cache_parent_synced => {
                    return Err(rollback_sidecar_transition(cache, block_number, e));
                }
                Err(e) => warn!(
                    target: "partial_stateless",
                    block = block_number,
                    error = %e,
                    "Sidecar generation failed while cache was not coherent"
                ),
            }
            Some(result)
        }
        Err(e) => {
            if cache_parent_synced {
                return Err(rollback_sidecar_transition(
                    cache,
                    block_number,
                    eyre::eyre!("failed to compute sidecar multiproof: {e}"),
                ));
            }
            warn!(
                target: "partial_stateless",
                block = block_number,
                error = %e,
                "Failed to compute multiproof while cache was not coherent"
            );
            None
        }
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
    use alloy_primitives::{Address, U256};
    use partial_stateless::policy::AccountData;

    #[test]
    fn discontinuous_parent_cold_resets_before_cache_prime() {
        let mut cache = NetworkStateCache::new(
            Box::new(LastNBlocksPolicy::new(60)),
            Box::new(LastNBlocksPolicy::new(30)),
        );
        let address = Address::repeat_byte(0x11);
        let mut accessed = BlockAccessedState::default();
        accessed.accounts.insert(
            address,
            AccountData { exists: true, nonce: 1, balance: U256::from(10), code_hash: None },
        );
        cache.on_block_executed(10, &accessed);

        assert!(!align_cache_to_parent(&mut cache, 12));
        assert_eq!(cache.current_block(), 0);
        assert!(cache.accounts().is_empty());

        cache.on_block_executed(12, &accessed);
        assert!(align_cache_to_parent(&mut cache, 12));
        assert!(cache.accounts().contains_key(&address));
    }
}
