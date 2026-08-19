use crate::timings::{
    AdmissionTimings, CacheDeltaMetrics, CacheRootMetrics, RetentionWalkMetrics, TrieCloneMetrics,
    ValidationPhaseTimings,
};
use alloy_consensus::{proofs::calculate_receipt_root, TxReceipt};
use alloy_primitives::{Address, Bloom, Bytes, B256, U256};
use eyre::{bail, eyre, Result};
use partial_stateless::{
    accessed_state::BlockAccessedState,
    check_sidecar_context, check_sidecar_miss_targets, cow_copies_taken,
    network_cache::{NetworkStateCache, UpdateStats},
    try_compute_trustless_state_root, try_compute_trustless_state_root_v2_with_storage_targets,
    try_compute_trustless_state_root_v3,
    witness_check::{
        check_sidecar_witness_prefilter, materialize_sidecar_witness_after_prefilter_with_cache,
        root_witness_completeness_from_bundle_with_cache, SidecarWitnessCheckLimits,
    },
    CacheAnchor, CacheRootTimings, MaterializedStateProof, PartialStatelessSidecar,
    PartialTrieNodeCache, RetentionTimings, RootWitnessCompletenessReport, StateTargetSet,
};
use reth_consensus::FullConsensus;
use reth_ethereum_primitives::EthPrimitives;
use reth_evm::{execute::Executor, ConfigureEvm};
use reth_primitives_traits::{Account, AlloyBlockHeader, BlockTy, Bytecode, RecoveredBlock};
use reth_revm::database::{EvmStateProvider, StateProviderDatabase};
use reth_storage_errors::provider::{ProviderError, ProviderResult};
use reth_trie_common::{HashedPostState, KeccakKeyHasher};
use revm::database::State;
use std::{collections::HashMap, fmt, mem, time::Instant};

pub type SidecarReexecLimits = SidecarWitnessCheckLimits;

/// The message every post-execution consensus rejection carries.
///
/// A block can be refused a dozen ways before the EVM runs and exactly one way after it — the
/// receipts and logs it actually produced against the ones its header committed to. Nothing in an
/// `eyre` chain distinguishes the two but the text, so a test that means to prove the *late*
/// refusal has to match on it. Exported so the match is against a symbol: renaming this breaks
/// the test rather than quietly widening what it accepts.
pub const POST_EXECUTION_REJECTION: &str = "post-execution consensus validation failed";

/// Controls whether a successfully verified transition replaces the caller's trie cache.
///
/// Builder-side preflights only need the verification result. Discarding their transactional
/// result avoids cloning the parent trie cache once in the caller and then again here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrieCacheDisposition {
    Commit,
    Discard,
}

/// The EVM configuration and consensus rules one validator runs a block under.
///
/// Bundled rather than passed as two arguments because they have to describe the same chain, and
/// because the consensus object carries more than a chain spec: `EthBeaconConsensus` holds flags
/// such as `skip_requests_hash_check` that change what a block is allowed to be. A full node must
/// therefore hand its ExEx the node's own consensus rather than a fresh one, or the ExEx could
/// reject a block its Engine accepted. A standalone validator builds both from the single chain
/// spec it was configured with.
pub struct ValidatorRules<'a, Evm, Consensus: ?Sized> {
    evm_config: &'a Evm,
    consensus: &'a Consensus,
}

// Hand-written because deriving would bound `Evm: Copy` and `Consensus: Copy`, which no EVM
// config or consensus object satisfies. This struct is two shared references and is always freely
// copyable regardless of what they point at.
impl<Evm, Consensus: ?Sized> Clone for ValidatorRules<'_, Evm, Consensus> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Evm, Consensus: ?Sized> Copy for ValidatorRules<'_, Evm, Consensus> {}

impl<Evm, Consensus: ?Sized> fmt::Debug for ValidatorRules<'_, Evm, Consensus> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ValidatorRules").finish_non_exhaustive()
    }
}

impl<'a, Evm, Consensus: ?Sized> ValidatorRules<'a, Evm, Consensus> {
    pub const fn new(evm_config: &'a Evm, consensus: &'a Consensus) -> Self {
        Self { evm_config, consensus }
    }

    pub const fn evm_config(&self) -> &'a Evm {
        self.evm_config
    }

    pub const fn consensus(&self) -> &'a Consensus {
        self.consensus
    }
}

/// A second opinion on the post-state root, computed by something other than this validator.
///
/// The point of the trait is which side of the crate boundary the state database sits on. A full
/// node can walk its own trie and cross-check the transition; a standalone validator has no such
/// thing, and must not be able to acquire one by taking a different branch. So the core accepts an
/// oracle it knows nothing about, the ExEx supplies one backed by its provider, and the standalone
/// validator supplies [`NoRootOracle`] — which is not a disabled fast path but the only
/// implementation its dependency graph can name.
pub trait PostStateRootOracle {
    /// Returns an independently computed post-state root, or `None` when there is no oracle.
    ///
    /// Takes the hashed post state by value because the only implementation that does real work
    /// consumes it, and because nothing in the transition reads it afterwards.
    fn post_state_root(&self, post_state: HashedPostState) -> Result<Option<B256>>;
}

/// The absence of a second opinion: this validator's own root is the only one available.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoRootOracle;

impl PostStateRootOracle for NoRootOracle {
    fn post_state_root(&self, _post_state: HashedPostState) -> Result<Option<B256>> {
        Ok(None)
    }
}

/// What the protocol says about one block, with no timing and no harness fields.
///
/// Deliberately not `Clone`: it carries a whole trie generation, and copying a result to read a
/// number off it would silently pay for a deep trie copy.
#[derive(Debug)]
pub struct SidecarValidationOutcome {
    /// The state root computed from the local sparse trie and the parent-state miss proof.
    ///
    /// Already checked against the block header — reaching this struct at all means it matched.
    /// There is one root field rather than the former pair because the provider-assisted
    /// cross-check compares its own answer against the same header value, so a returned outcome
    /// always had both equal to `block.state_root()`.
    pub state_root: B256,
    pub actual_accessed: BlockAccessedState,
    pub expected_miss: StateTargetSet,
    pub next_cache_anchor: CacheAnchor,
    pub cache_update: UpdateStats,
    pub root_witness_completeness: RootWitnessCompletenessReport,
    pub execution_gas_used: u64,
    pub execution_receipts_root: B256,
    pub execution_requests_hash: B256,
    pub execution_requests_empty: bool,
    /// The parent trie generation this transition displaced, when it committed.
    ///
    /// `None` under [`TrieCacheDisposition::Discard`], where nothing was displaced and the
    /// caller's trie cache is still the parent.
    pub displaced_trie_cache: Option<PartialTrieNodeCache>,
}

/// A protocol outcome together with the phase timings that produced it.
#[derive(Debug)]
pub struct TimedValidation {
    pub outcome: SidecarValidationOutcome,
    pub timings: ValidationPhaseTimings,
}

/// Validates and applies one sidecar using nothing but the caller's caches and the witness.
///
/// This is the entry point a standalone validator uses: no oracle, and therefore no way for the
/// transition to consult a state database even on a failure branch.
#[expect(clippy::too_many_arguments)]
pub fn verify_and_apply_sidecar<Evm, Consensus>(
    rules: ValidatorRules<'_, Evm, Consensus>,
    block: &RecoveredBlock<BlockTy<EthPrimitives>>,
    prev_cache: &mut NetworkStateCache,
    sidecar: &PartialStatelessSidecar,
    expected_cache_policy_id: B256,
    limits: &SidecarReexecLimits,
    trie_cache: &mut PartialTrieNodeCache,
    trie_cache_disposition: TrieCacheDisposition,
) -> Result<TimedValidation>
where
    Evm: ConfigureEvm<Primitives = EthPrimitives>,
    Consensus: FullConsensus<EthPrimitives> + ?Sized,
{
    verify_and_apply_sidecar_with_oracle(
        rules,
        block,
        prev_cache,
        sidecar,
        expected_cache_policy_id,
        limits,
        trie_cache,
        trie_cache_disposition,
        &NoRootOracle,
        false,
    )
}

/// Validates and applies one sidecar, cross-checking the post state against an external oracle.
///
/// The oracle runs at the same point the provider-assisted walk always did: after the local root
/// has already been checked against the header, and before any cache is committed. That ordering
/// is load-bearing — a disagreeing oracle must leave both caches at the parent generation.
#[expect(clippy::too_many_arguments)]
pub fn verify_and_apply_sidecar_with_oracle<Evm, Consensus, Oracle>(
    rules: ValidatorRules<'_, Evm, Consensus>,
    block: &RecoveredBlock<BlockTy<EthPrimitives>>,
    prev_cache: &mut NetworkStateCache,
    sidecar: &PartialStatelessSidecar,
    expected_cache_policy_id: B256,
    limits: &SidecarReexecLimits,
    trie_cache: &mut PartialTrieNodeCache,
    trie_cache_disposition: TrieCacheDisposition,
    root_oracle: &Oracle,
    compute_root_completeness: bool,
) -> Result<TimedValidation>
where
    Evm: ConfigureEvm<Primitives = EthPrimitives>,
    Consensus: FullConsensus<EthPrimitives> + ?Sized,
    Oracle: PostStateRootOracle + ?Sized,
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
    // The trimmed (v3) arm reads miss values through a read-only composite walk over the parent
    // trie cache and the witness map; the self-contained arms ignore the cache. Nothing before
    // the post-consensus commit writes to `trie_cache` either way.
    let materialized =
        materialize_sidecar_witness_after_prefilter_with_cache(sidecar, Some(trie_cache))
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
            MaterializedStateProof::TrimmedTransition(session) => {
                ("transition-v3-trimmed", session.map().len(), 0, 0)
            }
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
    let block_executor = rules.evm_config().executor(&mut db);
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

    // Canonical post-execution consensus admission, before anything downstream is computed and
    // long before anything is mutated.
    //
    // Position is deliberate. Everything below this point — hashing the post state, cloning the
    // trie, walking the sparse trie for the root, the external oracle, the miss-policy check — is
    // work a block that already disagrees with its own header does not deserve, and the state root
    // check further down cannot stand in for this one: a block can carry a correct state root and
    // still lie about gas used, receipts, logs bloom, or its requests hash.
    //
    // Reth's own implementation rather than a comparison written here. The three ad-hoc equality
    // checks this replaces omitted the logs bloom entirely and had no fork gating, so they passed
    // blocks that Reth's Engine would reject. Delegating also means `skip_requests_hash_check` and
    // every future fork rule arrive with the consensus object instead of being reimplemented.
    //
    // The receipts root is handed over rather than recomputed inside: materializing the blooms
    // once serves both the root and the block bloom, which is what `verify_receipts` would do
    // anyway, and it leaves `execution_receipts_root` available for the caller's differential.
    let post_execution_start = Instant::now();
    let receipts_with_bloom =
        execution_output.result.receipts.iter().map(TxReceipt::with_bloom_ref).collect::<Vec<_>>();
    let execution_receipts_root = calculate_receipt_root(&receipts_with_bloom);
    let execution_logs_bloom =
        receipts_with_bloom.iter().fold(Bloom::ZERO, |bloom, receipt| bloom | receipt.bloom_ref());
    drop(receipts_with_bloom);
    rules
        .consensus()
        .validate_block_post_execution(
            block,
            &execution_output.result,
            Some((execution_receipts_root, execution_logs_bloom)),
        )
        .map_err(|err| eyre!("{POST_EXECUTION_REJECTION}: {err}"))?;
    let post_execution_consensus_us = post_execution_start.elapsed().as_micros() as u64;

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
        MaterializedStateProof::TrimmedTransition(session) => try_compute_trustless_state_root_v3(
            session,
            trie_cache,
            &mut next_trie_cache,
            &execution_output.state,
            &sidecar.miss_manifest,
        ),
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
            partial_stateless::PartialExecutionWitnessState::MptTrimmedTransitionNodes {
                nodes,
                ..
            } => (nodes.len(), nodes.iter().map(|node| node.len()).sum()),
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
    // `raw_total_us`; the validator's cache+witness path does not need this walk, and under
    // `NoRootOracle` there is nothing here to exclude.
    let provider_start = Instant::now();
    let oracle_state_root = root_oracle
        .post_state_root(hashed_post_state)
        .map_err(|err| eyre!("external post-state root failed: {err}"))?;
    let provider_root_us = oracle_state_root
        .is_some()
        .then(|| provider_start.elapsed().as_micros() as u64)
        .unwrap_or_default();
    if let Some(oracle_state_root) = oracle_state_root &&
        oracle_state_root != block.state_root()
    {
        bail!(
            "provider-assisted state root mismatch: expected {:?}, got {:?}",
            block.state_root(),
            oracle_state_root
        );
    }

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
        // Left at its default: this entry point is handed an already-recovered block, so every
        // admission phase reads `None` rather than zero. `admission::admit_execution_data` fills
        // it in for the path that does the work.
        admission: AdmissionTimings::default(),
        context_check_us,
        witness_self_consistency_us,
        materialize_us,
        provider_setup_us,
        evm_call_us,
        accessed_state_capture_us,
        evm_us,
        post_execution_consensus_us,
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

    Ok(TimedValidation {
        outcome: SidecarValidationOutcome {
            state_root: trustless_state_root,
            actual_accessed,
            expected_miss,
            next_cache_anchor,
            cache_update,
            root_witness_completeness,
            execution_gas_used: execution_output.result.gas_used,
            execution_receipts_root,
            execution_requests_hash,
            execution_requests_empty,
            displaced_trie_cache,
        },
        timings,
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

    /// The delegated post-execution check catches what the comparisons it replaced could not.
    ///
    /// The three ad-hoc equality checks that used to run in the benchmark caller compared gas used,
    /// the receipts root, and the requests hash — and nothing else. A header carrying a correct
    /// receipts root beside a wrong logs bloom passed all three. It is not a theoretical gap: the
    /// bloom is what light clients filter on, and it is derived from the same receipts the root
    /// commits to, so a block can only disagree on one of them by lying.
    ///
    /// Driving the whole `verify_and_apply_sidecar` path would need an EVM, a witness, and a real
    /// block; what matters here is which rule set the core now defers to, so this exercises that
    /// rule set directly against the same consensus object the core is handed.
    #[test]
    fn a_correct_receipts_root_beside_a_wrong_logs_bloom_is_rejected() {
        use alloy_consensus::{
            constants::EMPTY_OMMER_ROOT_HASH, Block as AlloyBlock, BlockBody, Header,
            EMPTY_ROOT_HASH,
        };
        use alloy_evm::block::BlockExecutionResult;
        use alloy_primitives::{Address, Log, LogData, U256};
        use reth_chainspec::ChainSpecBuilder;
        use reth_consensus::FullConsensus;
        use reth_ethereum_primitives::{EthPrimitives, Receipt as EthReceipt, TxType};
        use std::sync::Arc;

        let chain_spec = Arc::new(ChainSpecBuilder::mainnet().paris_activated().build());
        let consensus = reth_ethereum_consensus::EthBeaconConsensus::new(chain_spec);

        let receipt = EthReceipt {
            tx_type: TxType::Legacy,
            success: true,
            cumulative_gas_used: 21_000,
            logs: vec![Log {
                address: Address::repeat_byte(0x11),
                data: LogData::new_unchecked(vec![B256::repeat_byte(0x22)], Bytes::new()),
            }],
        };
        let receipts = [receipt.clone()];
        let receipts_with_bloom =
            receipts.iter().map(TxReceipt::with_bloom_ref).collect::<Vec<_>>();
        let honest_root = calculate_receipt_root(&receipts_with_bloom);
        let honest_bloom = receipts_with_bloom
            .iter()
            .fold(Bloom::ZERO, |bloom, receipt| bloom | receipt.bloom_ref());
        assert_ne!(honest_bloom, Bloom::ZERO, "the fixture must produce a non-empty bloom");

        let result = BlockExecutionResult {
            receipts: vec![receipt],
            requests: Default::default(),
            gas_used: 21_000,
            blob_gas_used: 0,
        };
        let header = Header {
            number: 2,
            gas_used: 21_000,
            gas_limit: 30_000_000,
            receipts_root: honest_root,
            logs_bloom: honest_bloom,
            ommers_hash: EMPTY_OMMER_ROOT_HASH,
            transactions_root: EMPTY_ROOT_HASH,
            difficulty: U256::ZERO,
            ..Default::default()
        };
        let honest = RecoveredBlock::new_unhashed(
            AlloyBlock { header: header.clone(), body: BlockBody::default() },
            Vec::new(),
        );
        FullConsensus::<EthPrimitives>::validate_block_post_execution(
            &consensus, &honest, &result, None,
        )
        .expect("the fixture itself must be a block the rules accept");

        // One bit of the bloom flipped: the receipts root still matches, and every comparison the
        // caller used to make still passes.
        let mut tampered_bloom = honest_bloom;
        tampered_bloom.0[0] ^= 0x01;
        let tampered = RecoveredBlock::new_unhashed(
            AlloyBlock {
                header: Header { logs_bloom: tampered_bloom, ..header },
                body: BlockBody::default(),
            },
            Vec::new(),
        );
        assert_eq!(
            tampered.header().receipts_root,
            honest_root,
            "the receipts root must still agree, or this proves nothing about the bloom"
        );

        FullConsensus::<EthPrimitives>::validate_block_post_execution(
            &consensus, &tampered, &result, None,
        )
        .expect_err("a wrong logs bloom must be rejected");
    }

    /// An admission phase that did not run is not an admission phase that was free.
    #[test]
    fn absent_admission_phases_cost_nothing_and_zero_ones_are_still_reported() {
        use crate::timings::AdmissionTimings;

        assert_eq!(AdmissionTimings::default().total_us(), 0);
        let performed = AdmissionTimings {
            sender_recovery_us: Some(0),
            pre_execution_consensus_us: Some(7),
            ..Default::default()
        };
        assert_eq!(performed.total_us(), 7);
        assert_eq!(
            performed.sender_recovery_us,
            Some(0),
            "a phase that ran in under a microsecond must stay distinguishable from one that did \
             not run"
        );
    }

    /// The standalone validator's cross-check is absent, not merely disabled.
    ///
    /// Worth pinning because the whole database-free claim rests on it: `provider_root_us` is only
    /// nonzero, and the header comparison at the oracle site only runs, when an oracle answers. A
    /// `NoRootOracle` that ever returned `Some` would put a second root in a path that has no way
    /// to compute one.
    #[test]
    fn the_absent_oracle_answers_nothing_for_any_post_state() {
        assert_eq!(
            NoRootOracle.post_state_root(HashedPostState::default()).expect("infallible"),
            None
        );
    }

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
