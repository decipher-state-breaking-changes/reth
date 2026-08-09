use crate::benchmark::{
    CacheDeltaMetrics, CacheRootMetrics, RetentionWalkMetrics, TrieCloneMetrics,
    ValidationPhaseTimings,
};
use alloy_primitives::{Address, Bytes, B256, U256};
use eyre::{bail, eyre, Result};
use partial_stateless::{
    accessed_state::BlockAccessedState,
    check_sidecar_context, check_sidecar_miss_targets, cow_copies_taken,
    network_cache::{NetworkStateCache, UpdateStats},
    try_compute_trustless_state_root, try_compute_trustless_state_root_v2_with_storage_targets,
    witness_check::{
        check_sidecar_witness_prefilter, materialize_sidecar_witness_after_prefilter,
        root_witness_completeness_from_bundle_with_cache, SidecarWitnessCheckLimits,
    },
    CacheAnchor, CacheRootTimings, MaterializedStateProof, PartialStatelessSidecar,
    PartialTrieNodeCache, RetentionTimings, RootWitnessCompletenessReport, StateTargetSet,
};
use reth_ethereum::{calculate_receipt_root_no_memo, EthPrimitives};
use reth_evm::{execute::Executor, ConfigureEvm};
use reth_primitives_traits::{Account, AlloyBlockHeader, BlockTy, Bytecode, RecoveredBlock};
use reth_provider::{ProviderError, ProviderResult, StateProvider};
use reth_revm::database::{EvmStateProvider, StateProviderDatabase};
use reth_trie_common::{HashedPostState, KeccakKeyHasher};
use revm::database::State;
use std::{collections::HashMap, mem, time::Instant};

pub(crate) type SidecarReexecLimits = SidecarWitnessCheckLimits;

/// Controls whether a successfully verified transition replaces the caller's trie cache.
///
/// Builder-side preflights only need the verification result. Discarding their transactional
/// result avoids cloning the parent trie cache once in the caller and then again here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TrieCacheDisposition {
    Commit,
    Discard,
}

/// Deliberately not `Clone`: it carries a whole trie generation, and copying a report to read a
/// number off it would silently pay for a deep trie copy.
#[derive(Debug)]
pub(crate) struct SidecarReexecReport {
    pub computed_state_root: B256,
    pub actual_accessed: BlockAccessedState,
    pub expected_miss: StateTargetSet,
    pub next_cache_anchor: CacheAnchor,
    pub cache_update: UpdateStats,
    pub root_witness_completeness: RootWitnessCompletenessReport,
    /// State root computed from the local sparse trie + parent-state miss proof.
    pub trustless_state_root: Option<B256>,
    pub execution_gas_used: u64,
    pub execution_receipts_root: B256,
    pub execution_requests_hash: B256,
    pub execution_requests_empty: bool,
    pub timings: ValidationPhaseTimings,
    /// The parent trie generation this transition displaced, when it committed.
    ///
    /// `None` under [`TrieCacheDisposition::Discard`], where nothing was displaced and the
    /// caller's trie cache is still the parent.
    pub displaced_trie_cache: Option<PartialTrieNodeCache>,
}

pub(crate) fn verify_and_apply_provider_assisted_sidecar<Evm>(
    evm_config: &Evm,
    full_provider: &dyn StateProvider,
    block: &RecoveredBlock<BlockTy<EthPrimitives>>,
    prev_cache: &mut NetworkStateCache,
    sidecar: &PartialStatelessSidecar,
    expected_cache_policy_id: B256,
    limits: &SidecarReexecLimits,
    trie_cache: &mut PartialTrieNodeCache,
    trie_cache_disposition: TrieCacheDisposition,
) -> Result<SidecarReexecReport>
where
    Evm: ConfigureEvm<Primitives = EthPrimitives>,
{
    verify_and_apply_sidecar_inner(
        evm_config,
        Some(full_provider),
        block,
        prev_cache,
        sidecar,
        expected_cache_policy_id,
        limits,
        trie_cache,
        trie_cache_disposition,
        true,
    )
}

#[expect(clippy::too_many_arguments)]
pub(crate) fn verify_and_apply_trustless_sidecar_for_benchmark<Evm>(
    evm_config: &Evm,
    block: &RecoveredBlock<BlockTy<EthPrimitives>>,
    prev_cache: &mut NetworkStateCache,
    sidecar: &PartialStatelessSidecar,
    expected_cache_policy_id: B256,
    limits: &SidecarReexecLimits,
    trie_cache: &mut PartialTrieNodeCache,
    trie_cache_disposition: TrieCacheDisposition,
) -> Result<SidecarReexecReport>
where
    Evm: ConfigureEvm<Primitives = EthPrimitives>,
{
    verify_and_apply_sidecar_inner(
        evm_config,
        None,
        block,
        prev_cache,
        sidecar,
        expected_cache_policy_id,
        limits,
        trie_cache,
        trie_cache_disposition,
        false,
    )
}

fn verify_and_apply_sidecar_inner<Evm>(
    evm_config: &Evm,
    full_provider: Option<&dyn StateProvider>,
    block: &RecoveredBlock<BlockTy<EthPrimitives>>,
    prev_cache: &mut NetworkStateCache,
    sidecar: &PartialStatelessSidecar,
    expected_cache_policy_id: B256,
    limits: &SidecarReexecLimits,
    trie_cache: &mut PartialTrieNodeCache,
    trie_cache_disposition: TrieCacheDisposition,
    compute_root_completeness: bool,
) -> Result<SidecarReexecReport>
where
    Evm: ConfigureEvm<Primitives = EthPrimitives>,
{
    let assisted_start = Instant::now();
    let context_start = Instant::now();
    prefilter(block, prev_cache, sidecar, expected_cache_policy_id)?;
    let context_check_us = context_start.elapsed().as_micros() as u64;

    let witness_check_start = Instant::now();
    check_sidecar_witness_prefilter(sidecar, limits)
        .map_err(|err| eyre!("sidecar witness check failed: {err}"))?;
    let witness_self_consistency_us = witness_check_start.elapsed().as_micros() as u64;
    let materialize_start = Instant::now();
    let materialized = materialize_sidecar_witness_after_prefilter(sidecar)
        .map_err(|err| eyre!("sidecar witness materialization failed: {err}"))?;
    let materialize_us = materialize_start.elapsed().as_micros() as u64;
    // Retained for trustless root computation before the other fields are moved into the provider.
    let state_proof = materialized.state_proof;
    if let Some(cached_parent_root) = trie_cache.state_root() {
        if cached_parent_root != sidecar.parent_state_root {
            bail!(
                "trie cache is anchored to the wrong parent root: sidecar={:?}, cache={:?}, cache_block={}, sidecar_parent_block={}",
                sidecar.parent_state_root,
                cached_parent_root,
                prev_cache.current_block(),
                sidecar.cache_block,
            );
        }
    }
    let (proof_kind, proof_account_nodes, proof_storage_tries, proof_storage_nodes) =
        match &state_proof {
            MaterializedStateProof::Legacy(proof) => (
                "legacy",
                proof.account_subtree.len(),
                proof.storages.len(),
                proof.storages.values().map(|storage| storage.subtree.len()).sum::<usize>(),
            ),
            MaterializedStateProof::Transition(proof) => (
                "transition-v2",
                proof.account_proofs.len(),
                proof.storage_proofs.len(),
                proof.storage_proofs.values().map(Vec::len).sum::<usize>(),
            ),
        };
    let witness_provider = WitnessBackedStateProvider {
        cache: prev_cache,
        trie_cache,
        witness_accounts: materialized.accounts,
        witness_storage: materialized.storage,
        witness_codes: materialized.codes,
        witness_headers: materialized.headers,
        block_number: sidecar.block_number,
    };

    let provider_setup_start = Instant::now();
    let state_provider_db = StateProviderDatabase::new(witness_provider);
    let mut db = State::builder().with_bundle_update().with_database(state_provider_db).build();
    let block_executor = evm_config.executor(&mut db);
    let provider_setup_us = provider_setup_start.elapsed().as_micros() as u64;

    let evm_start = Instant::now();
    let mut actual_accessed = BlockAccessedState::default();
    let mut accessed_state_capture_us = 0;
    let execution_output = block_executor
        .execute_with_state_closure(block, |statedb: &State<_>| {
            let capture_start = Instant::now();
            actual_accessed = BlockAccessedState::from_simulated_state(statedb);
            accessed_state_capture_us = capture_start.elapsed().as_micros() as u64;
        })
        .map_err(|err| eyre!("partial sidecar re-execution failed: {err:?}"))?;
    let evm_call_us = evm_start.elapsed().as_micros() as u64;
    let evm_us = evm_call_us.saturating_sub(accessed_state_capture_us);
    drop(db);
    let execution_receipts_root = calculate_receipt_root_no_memo(&execution_output.result.receipts);
    let execution_requests_hash = execution_output.result.requests.requests_hash();
    let execution_requests_empty = execution_output.result.requests.is_empty();

    let hash_start = Instant::now();
    let hashed_post_state =
        HashedPostState::from_bundle_state::<KeccakKeyHasher>(execution_output.state.state());
    let hash_post_state_us = hash_start.elapsed().as_micros() as u64;

    // Apply the block to a transactional sparse-trie snapshot. The original cache is left
    // untouched until the consensus root and next cache anchor have both been checked.
    //
    // The snapshot shares its storage tries with the parent, so the copies it goes on to take are
    // spread across the transition and retention rather than paid here. Bracketing the whole
    // transaction is the only way to count them; the counter is process-wide, which is exact
    // because the ExEx applies one transition at a time.
    let cow_copies_before = cow_copies_taken();
    let clone_start = Instant::now();
    let (mut next_trie_cache, trie_clone_detail) = trie_cache.clone_timed();
    let trie_clone_us = clone_start.elapsed().as_micros() as u64;
    let transition_storage_targets = sidecar
        .cache_miss_targets
        .storage
        .iter()
        .map(|(address, _)| alloy_primitives::keccak256(address))
        .collect::<Vec<_>>();
    let root_start = Instant::now();
    let trustless_state_root = match state_proof {
        MaterializedStateProof::Legacy(proof) => {
            try_compute_trustless_state_root(proof, &mut next_trie_cache, &execution_output.state)
        }
        MaterializedStateProof::Transition(proof) => {
            try_compute_trustless_state_root_v2_with_storage_targets(
                proof,
                &mut next_trie_cache,
                &execution_output.state,
                transition_storage_targets.iter().copied(),
            )
        }
    }
    .map_err(|err| eyre!("local sparse-trie transition failed: {err}"))?;
    let state_root_us = root_start.elapsed().as_micros() as u64;
    if trustless_state_root != block.state_root() {
        let shape = trie_cache.shape_metrics();
        let storage_wipes =
            hashed_post_state.storages.values().filter(|storage| storage.wiped).count();
        let storage_slot_mutations =
            hashed_post_state.storages.values().map(|storage| storage.storage.len()).sum::<usize>();
        let storage_removals = hashed_post_state
            .storages
            .values()
            .flat_map(|storage| storage.storage.values())
            .filter(|value| value.is_zero())
            .count();
        let (sidecar_state_nodes, sidecar_state_bytes) = match &sidecar.witness.state {
            partial_stateless::PartialExecutionWitnessState::MptMultiProof(bytes) => {
                (0, bytes.len())
            }
            partial_stateless::PartialExecutionWitnessState::MptTransitionNodes(nodes) => {
                (nodes.len(), nodes.iter().map(|node| node.len()).sum())
            }
        };
        bail!(
            "local sparse-trie state root mismatch: expected {:?}, got {:?}; parent_root={:?}, cache_root={:?}, proof_kind={}, proof_account_nodes={}, proof_storage_tries={}, proof_storage_nodes={}, sidecar_state_nodes={}, sidecar_state_bytes={}, post_accounts={}, post_storage_tries={}, post_storage_slots={}, storage_wipes={}, storage_removals={}, transition_storage_targets={}, cache_warm_accounts={}, cache_warm_storage={}, cache_account_nodes={}, cache_storage_nodes={}",
            block.state_root(),
            trustless_state_root,
            sidecar.parent_state_root,
            trie_cache.state_root(),
            proof_kind,
            proof_account_nodes,
            proof_storage_tries,
            proof_storage_nodes,
            sidecar_state_nodes,
            sidecar_state_bytes,
            hashed_post_state.accounts.len(),
            hashed_post_state.storages.len(),
            storage_slot_mutations,
            storage_wipes,
            storage_removals,
            transition_storage_targets.len(),
            shape.retained_account_paths,
            shape.retained_storage_paths,
            shape.account_revealed_nodes,
            shape.storage_revealed_nodes,
        );
    }

    // Optional builder/live-verifier correctness cross-check. It is deliberately excluded from
    // `raw_total_us`; the validator's cache+witness path does not need this DB walk.
    let provider_start = Instant::now();
    let computed_state_root = if let Some(full_provider) = full_provider {
        let (computed_state_root, _) = full_provider
            .state_root_with_updates(hashed_post_state)
            .map_err(|err| eyre!("provider-assisted state root failed: {err}"))?;
        if computed_state_root != block.state_root() {
            bail!(
                "provider-assisted state root mismatch: expected {:?}, got {:?}",
                block.state_root(),
                computed_state_root
            );
        }
        computed_state_root
    } else {
        trustless_state_root
    };
    let provider_root_us = full_provider
        .is_some()
        .then(|| provider_start.elapsed().as_micros() as u64)
        .unwrap_or_default();

    let completeness_start = Instant::now();
    let root_witness_completeness = if compute_root_completeness {
        root_witness_completeness_from_bundle_with_cache(
            &execution_output.state,
            &sidecar.cache_miss_targets,
            trie_cache,
        )
    } else {
        RootWitnessCompletenessReport::default()
    };
    let root_completeness_us = compute_root_completeness
        .then(|| completeness_start.elapsed().as_micros() as u64)
        .unwrap_or_default();

    let miss_policy_start = Instant::now();
    let expected_miss = prev_cache.expected_miss_targets(&actual_accessed);
    check_sidecar_miss_targets(sidecar, &expected_miss)
        .map_err(|err| eyre!("cache-miss-only check failed: {err:?}"))?;
    let miss_policy_check_us = miss_policy_start.elapsed().as_micros() as u64;

    let (cache_update, next_cache_anchor, cache_timings, displaced_trie_cache) =
        apply_cache_transition_and_check(
            prev_cache,
            &actual_accessed,
            sidecar.block_number,
            sidecar.block_hash,
            expected_cache_policy_id,
            sidecar.next_cache_anchor,
            trie_cache,
            next_trie_cache,
            trie_cache_disposition,
        )?;
    let trie_storage_tries_copied = cow_copies_taken().saturating_sub(cow_copies_before);

    let mut timings = ValidationPhaseTimings {
        context_check_us,
        witness_self_consistency_us,
        materialize_us,
        provider_setup_us,
        evm_call_us,
        accessed_state_capture_us,
        evm_us,
        hash_post_state_us,
        trie_clone_us,
        trie_clone_detail: TrieCloneMetrics::from(&trie_clone_detail),
        trie_storage_tries_copied,
        trie_storage_tries_total: cache_timings.storage_tries_total,
        state_root_us,
        root_completeness_us,
        miss_policy_check_us,
        cache_update_us: cache_timings.update_us,
        cache_root_index_maintenance_us: cache_update.index_maintenance_us,
        cache_delta: CacheDeltaMetrics::from(&cache_update),
        trie_retention_us: cache_timings.retention_us,
        retention_warm_membership_us: cache_timings.retention.warm_membership_us,
        retention_storage_paths_us: cache_timings.retention.storage_paths_us,
        retention_account_paths_us: cache_timings.retention.account_paths_us,
        retention_account_trie_us: cache_timings.retention.account_trie_us,
        retention_account_trie_detail: RetentionWalkMetrics::from(
            &cache_timings.retention.account_trie,
        ),
        retention_storage_tries_us: cache_timings.retention.storage_tries_us,
        retention_storage_trie_detail: RetentionWalkMetrics::from(
            &cache_timings.retention.storage_tries,
        ),
        retention_account_paths: cache_timings.retention.account_paths,
        retention_storage_tries_pruned: cache_timings.retention.storage_tries_pruned,
        retention_storage_tries_skipped: cache_timings.retention.storage_tries_skipped,
        retention_storage_trie_cow_us: cache_timings.retention.storage_trie_cow_us,
        retention_storage_trie_cow_copies: cache_timings.retention.storage_trie_cow_copies,
        retention_storage_trie_drop_us: cache_timings.retention.storage_trie_drop_us,
        retention_storage_tries_dropped: cache_timings.retention.storage_tries_dropped,
        retention_full_rebuild: u64::from(cache_timings.retention.full_rebuild),
        next_cache_anchor_us: cache_timings.anchor_us,
        next_cache_anchor_detail: CacheRootMetrics::from(&cache_timings.anchor),
        trie_commit_us: cache_timings.commit_us,
        provider_root_us,
        ..Default::default()
    };
    timings.recompute_totals();
    let assisted_wall_us = assisted_start.elapsed().as_micros() as u64;
    let db_free_wall_us = assisted_wall_us.saturating_sub(provider_root_us);
    timings.unattributed_us = db_free_wall_us.saturating_sub(timings.raw_total_us);
    timings.recompute_totals();

    Ok(SidecarReexecReport {
        computed_state_root,
        actual_accessed,
        expected_miss,
        next_cache_anchor,
        cache_update,
        root_witness_completeness,
        trustless_state_root: Some(trustless_state_root),
        execution_gas_used: execution_output.result.gas_used,
        execution_receipts_root,
        execution_requests_hash,
        execution_requests_empty,
        timings,
        displaced_trie_cache,
    })
}

#[derive(Debug, Default)]
struct CacheTransitionTimings {
    update_us: u64,
    retention_us: u64,
    /// The retention phase's internal split. See [`RetentionTimings`].
    retention: RetentionTimings,
    anchor_us: u64,
    /// The anchor phase's internal split. See [`CacheRootTimings`].
    anchor: CacheRootTimings,
    commit_us: u64,
    /// Storage tries the snapshot held after retention.
    ///
    /// Paired with the copy count taken across the whole transaction, this is what the
    /// copy-on-write snapshot saved over the deep clone, which is no longer visible in
    /// `trie_clone_us` once that stops doing the copying.
    storage_tries_total: u64,
}

fn apply_cache_transition_and_check(
    cache: &mut NetworkStateCache,
    accessed: &BlockAccessedState,
    block_number: u64,
    block_hash: B256,
    cache_policy_id: B256,
    expected_next_anchor: CacheAnchor,
    trie_cache: &mut PartialTrieNodeCache,
    mut next_trie_cache: PartialTrieNodeCache,
    trie_cache_disposition: TrieCacheDisposition,
) -> Result<(UpdateStats, CacheAnchor, CacheTransitionTimings, Option<PartialTrieNodeCache>)> {
    let mut timings = CacheTransitionTimings::default();
    let start = Instant::now();
    let cache_update = cache.on_block_executed(block_number, accessed);
    timings.update_us = start.elapsed().as_micros() as u64;
    let start = Instant::now();
    timings.retention = next_trie_cache.retain_from_value_cache(cache);
    timings.retention_us = start.elapsed().as_micros() as u64;
    timings.storage_tries_total = next_trie_cache.storage_trie_count() as u64;
    let start = Instant::now();
    let (next_cache_anchor, anchor_detail) =
        cache.cache_anchor_timed(block_number, block_hash, cache_policy_id);
    timings.anchor_us = start.elapsed().as_micros() as u64;
    timings.anchor = anchor_detail;
    if next_cache_anchor != expected_next_anchor {
        cache.rollback_block(block_number).map_err(|rollback_err| {
            eyre!("next cache anchor mismatch; cache rollback also failed: {rollback_err}")
        })?;
        bail!(
            "next cache anchor mismatch: expected {expected_next_anchor:?}, got {next_cache_anchor:?}"
        );
    }
    // Committing hands the displaced parent back rather than dropping it, which is what makes a
    // one-deep retained generation free: `next_trie_cache` is already a copy of the parent, so the
    // object exists either way and the only question is whether anything still holds it. A
    // discarded transition never displaced anything — the caller's trie cache is still the parent.
    let displaced_trie_cache = if trie_cache_disposition == TrieCacheDisposition::Commit {
        let start = Instant::now();
        let displaced = mem::replace(trie_cache, next_trie_cache);
        timings.commit_us = start.elapsed().as_micros() as u64;
        Some(displaced)
    } else {
        None
    };
    Ok((cache_update, next_cache_anchor, timings, displaced_trie_cache))
}

fn prefilter(
    block: &RecoveredBlock<BlockTy<EthPrimitives>>,
    prev_cache: &NetworkStateCache,
    sidecar: &PartialStatelessSidecar,
    expected_cache_policy_id: B256,
) -> Result<()> {
    check_expected_cache_policy_id(sidecar.cache_policy_id, expected_cache_policy_id)?;
    if sidecar.block_hash != block.hash() {
        bail!("sidecar block_hash mismatch");
    }
    if sidecar.parent_hash != block.parent_hash() {
        bail!("sidecar parent_hash mismatch");
    }
    if sidecar.block_number != block.number() {
        bail!("sidecar block_number mismatch");
    }
    check_parent_cache_height(block.number(), prev_cache.current_block(), sidecar.cache_block)?;

    let local_prev_anchor =
        prev_cache.cache_anchor(sidecar.cache_block, sidecar.parent_hash, expected_cache_policy_id);
    check_sidecar_context(sidecar, &local_prev_anchor)
        .map_err(|err| eyre!("sidecar cache context mismatch: {err:?}"))?;

    Ok(())
}

fn check_expected_cache_policy_id(
    sidecar_cache_policy_id: B256,
    expected_cache_policy_id: B256,
) -> Result<()> {
    if sidecar_cache_policy_id != expected_cache_policy_id {
        bail!(
            "sidecar cache_policy_id mismatch: expected {:?}, got {:?}",
            expected_cache_policy_id,
            sidecar_cache_policy_id
        );
    }
    Ok(())
}

fn check_parent_cache_height(
    block_number: u64,
    local_cache_block: u64,
    sidecar_cache_block: u64,
) -> Result<()> {
    let expected_parent = block_number.saturating_sub(1);
    if local_cache_block != expected_parent {
        bail!(
            "local cache block mismatch: expected parent {expected_parent}, got {local_cache_block}"
        );
    }
    if sidecar_cache_block != expected_parent {
        bail!(
            "sidecar cache_block mismatch: expected parent {expected_parent}, got {sidecar_cache_block}"
        );
    }
    Ok(())
}

struct WitnessBackedStateProvider<'a> {
    cache: &'a NetworkStateCache,
    trie_cache: &'a PartialTrieNodeCache,
    witness_accounts: HashMap<Address, Option<Account>>,
    witness_storage: HashMap<(Address, B256), U256>,
    witness_codes: HashMap<B256, Bytes>,
    witness_headers: HashMap<u64, B256>,
    block_number: u64,
}

impl WitnessBackedStateProvider<'_> {
    fn missing(label: &str) -> ProviderError {
        ProviderError::TrieWitnessError(label.to_string())
    }
}

impl EvmStateProvider for WitnessBackedStateProvider<'_> {
    fn basic_account(&self, address: &Address) -> ProviderResult<Option<Account>> {
        if let Some(entry) = self.cache.accounts().get(address) {
            match self.trie_cache.account_exists(address) {
                Some(false) => return Ok(None),
                Some(true) => {}
                None => {
                    return Err(Self::missing(&format!(
                        "cached account is missing an authenticated trie path for {address:?}"
                    )))
                }
            }
            return Ok(Some(Account {
                nonce: entry.value.nonce,
                balance: entry.value.balance,
                bytecode_hash: entry.value.code_hash,
            }));
        }

        if let Some(account) = self.witness_accounts.get(address) {
            return Ok(*account);
        }

        Err(Self::missing(&format!("missing account witness for {address:?}")))
    }

    fn block_hash(&self, number: u64) -> ProviderResult<Option<B256>> {
        if number >= self.block_number || number.saturating_add(256) < self.block_number {
            return Ok(None);
        }

        self.witness_headers
            .get(&number)
            .copied()
            .map(Some)
            .ok_or_else(|| Self::missing(&format!("missing ancestor header witness for {number}")))
    }

    fn bytecode_by_hash(&self, code_hash: &B256) -> ProviderResult<Option<Bytecode>> {
        if let Some(entry) = self.cache.codes().get(code_hash) {
            return Ok(Some(Bytecode::new_raw(entry.value.clone())));
        }

        if let Some(code) = self.witness_codes.get(code_hash) {
            return Ok(Some(Bytecode::new_raw(code.clone())));
        }

        Err(Self::missing(&format!("missing bytecode witness for {code_hash:?}")))
    }

    fn storage(&self, account: Address, storage_key: B256) -> ProviderResult<Option<U256>> {
        if let Some(entry) = self.cache.storage().get(&(account, storage_key)) {
            return Ok(Some(entry.value));
        }

        if let Some(value) = self.witness_storage.get(&(account, storage_key)) {
            return Ok(Some(*value));
        }

        Err(Self::missing(&format!(
            "missing storage witness for account={account:?}, slot={storage_key:?}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use partial_stateless::{
        check_sidecar_context,
        policy::{AccountData, LastNBlocksPolicy},
        PartialExecutionWitness, PartialExecutionWitnessState, WitnessResult, WitnessTargets,
    };

    #[test]
    fn parent_cache_height_is_bound_to_the_block_parent() {
        check_parent_cache_height(100, 99, 99).expect("matching parent context");
        assert!(check_parent_cache_height(100, 98, 99).is_err());
        assert!(check_parent_cache_height(100, 99, 98).is_err());
    }

    #[test]
    fn consistently_forged_policy_id_is_rejected() {
        let expected_policy_id = B256::repeat_byte(0x11);
        let forged_policy_id = B256::repeat_byte(0x22);
        let parent_hash = B256::repeat_byte(0x33);
        let block_hash = B256::repeat_byte(0x44);
        let mut cache = NetworkStateCache::new(
            Box::new(LastNBlocksPolicy::new(60)),
            Box::new(LastNBlocksPolicy::new(30)),
        );
        cache.on_block_executed(99, &BlockAccessedState::default());
        let forged_prev_anchor = cache.cache_anchor(99, parent_hash, forged_policy_id);
        let sidecar = PartialStatelessSidecar {
            parent_hash,
            parent_state_root: B256::repeat_byte(0x55),
            block_hash,
            block_number: 100,
            cache_block: 99,
            cache_policy_id: forged_policy_id,
            prev_cache_anchor: forged_prev_anchor,
            next_cache_anchor: CacheAnchor {
                block_number: 100,
                block_hash,
                cache_policy_id: forged_policy_id,
                cache_root: B256::repeat_byte(0x66),
            },
            cache_policy_metadata: "forged".to_string(),
            cache_miss_targets: StateTargetSet::default(),
            witness_commitment: B256::ZERO,
            miss_manifest: WitnessTargets {
                missed_accounts: vec![],
                missed_storage: vec![],
                missed_code_hashes: vec![],
            },
            witness: PartialExecutionWitness {
                state: PartialExecutionWitnessState::MptMultiProof(vec![]),
                codes: vec![],
                keys: vec![],
                headers: vec![],
            },
            stats: WitnessResult {
                total_size_bytes: 0,
                account_proof_bytes: 0,
                storage_proof_bytes: 0,
                bytecode_bytes: 0,
                account_proof_nodes: 0,
                storage_proof_nodes: 0,
                target_accounts: 0,
                target_storage_slots: 0,
                computation_time_ms: None,
                cpu_time_ms: None,
                major_page_faults: None,
                minor_page_faults: None,
            },
        };

        // The old core path derived its "local" anchor from the untrusted policy ID, so a
        // consistently forged sidecar passed the generic context check.
        check_sidecar_context(&sidecar, &forged_prev_anchor)
            .expect("internally consistent forged policy context");

        let err = check_expected_cache_policy_id(sidecar.cache_policy_id, expected_policy_id)
            .expect_err("locally configured policy must bind core verification");
        assert!(err.to_string().contains("sidecar cache_policy_id mismatch"));
    }

    #[test]
    fn next_anchor_mismatch_rolls_back_cache_transition() {
        let mut cache = NetworkStateCache::new(
            Box::new(LastNBlocksPolicy::new(60)),
            Box::new(LastNBlocksPolicy::new(30)),
        );
        cache.on_block_executed(99, &BlockAccessedState::default());
        let root_before = cache.cache_root();
        let mut trie_cache = PartialTrieNodeCache::new();
        let trie_root_before = trie_cache.cache_root();
        let address = Address::repeat_byte(0x11);
        let mut accessed = BlockAccessedState::default();
        accessed
            .accounts
            .insert(address, AccountData { nonce: 1, balance: U256::from(10), code_hash: None });

        let _error = apply_cache_transition_and_check(
            &mut cache,
            &accessed,
            100,
            B256::repeat_byte(0x22),
            B256::repeat_byte(0x33),
            CacheAnchor {
                block_number: 100,
                block_hash: B256::repeat_byte(0x22),
                cache_policy_id: B256::repeat_byte(0x33),
                cache_root: B256::ZERO,
            },
            &mut trie_cache,
            PartialTrieNodeCache::new(),
            TrieCacheDisposition::Commit,
        )
        .expect_err("wrong next cache root must fail");

        assert_eq!(cache.current_block(), 99);
        assert_eq!(cache.cache_root(), root_before);
        assert_eq!(trie_cache.cache_root(), trie_root_before);
        assert!(!cache.contains_account(&address));
    }

    /// Everything a transition at block 100 needs: a parent cache, the block's access set, and the
    /// next anchor the sidecar would have to name for the transition to be accepted.
    fn transition_fixture() -> (NetworkStateCache, BlockAccessedState, CacheAnchor, Address) {
        fn cache_at_block_99() -> NetworkStateCache {
            let mut cache = NetworkStateCache::new(
                Box::new(LastNBlocksPolicy::new(60)),
                Box::new(LastNBlocksPolicy::new(30)),
            );
            cache.on_block_executed(99, &BlockAccessedState::default());
            cache
        }

        let address = Address::repeat_byte(0x11);
        let mut accessed = BlockAccessedState::default();
        accessed
            .accounts
            .insert(address, AccountData { nonce: 1, balance: U256::from(10), code_hash: None });

        let mut expected_cache = cache_at_block_99();
        expected_cache.on_block_executed(100, &accessed);
        let expected_anchor =
            expected_cache.cache_anchor(100, B256::repeat_byte(0x22), B256::repeat_byte(0x33));

        (cache_at_block_99(), accessed, expected_anchor, address)
    }

    #[test]
    fn successful_preflight_can_discard_transactional_trie_cache() {
        let (mut cache, accessed, expected_anchor, address) = transition_fixture();
        let mut trie_cache = PartialTrieNodeCache::new();
        let (_, _, _, displaced) = apply_cache_transition_and_check(
            &mut cache,
            &accessed,
            100,
            B256::repeat_byte(0x22),
            B256::repeat_byte(0x33),
            expected_anchor,
            &mut trie_cache,
            PartialTrieNodeCache::new(),
            TrieCacheDisposition::Discard,
        )
        .expect("valid preflight transition should succeed");

        assert_eq!(cache.current_block(), 100);
        assert!(!trie_cache.tracks_account(&address));
        assert!(
            displaced.is_none(),
            "a discarded transition displaced nothing: the caller's trie cache is still the parent"
        );
    }

    #[test]
    fn a_committed_transition_hands_back_the_parent_generation() {
        let (mut cache, accessed, expected_anchor, address) = transition_fixture();
        let mut trie_cache = PartialTrieNodeCache::new();
        let (_, _, _, displaced) = apply_cache_transition_and_check(
            &mut cache,
            &accessed,
            100,
            B256::repeat_byte(0x22),
            B256::repeat_byte(0x33),
            expected_anchor,
            &mut trie_cache,
            PartialTrieNodeCache::new(),
            TrieCacheDisposition::Commit,
        )
        .expect("valid transition should succeed");

        assert_eq!(cache.current_block(), 100);
        assert!(
            trie_cache.tracks_account(&address),
            "the caller's trie cache advanced to the child generation"
        );
        let displaced = displaced.expect("a committed transition displaces the parent");
        assert!(
            !displaced.tracks_account(&address),
            "and what came back is the generation before this block, which is what a depth-1 \
             reorg undoes into"
        );
    }
}
