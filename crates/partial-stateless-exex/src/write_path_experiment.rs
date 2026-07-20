use crate::sidecar_reexec::{verify_and_apply_provider_assisted_sidecar, SidecarReexecLimits};
use alloy_primitives::B256;
use eyre::{bail, eyre, Result};
use partial_stateless::{
    partial_witness_commitment,
    policy::LastNBlocksPolicy,
    root_witness_targets_from_bundle, try_compute_trustless_state_root,
    witness::{measure_multiproof_size, state_targets_to_proof_targets},
    NetworkStateCache, PartialExecutionWitnessState, PartialStatelessSidecar, PartialTrieNodeCache,
    SerializableMultiProof, TrieProofTarget, TrieTransitionError,
};
use reth_ethereum::EthPrimitives;
use reth_evm::ConfigureEvm;
use reth_primitives_traits::{AlloyBlockHeader, BlockTy, RecoveredBlock};
use reth_provider::StateProvider;
use reth_trie_common::TrieInput;
use revm::database::BundleState;
use std::{fs, path::Path};

const MAX_SUPPLEMENTAL_PROOF_ROUNDS: usize = 128;

#[derive(Debug, Clone)]
pub(crate) struct WritePathExperimentReport {
    pub(crate) block_number: u64,
    pub(crate) computed_state_root: B256,
    pub(crate) wire_sidecar_bytes: usize,
    pub(crate) proof_bytes: usize,
    pub(crate) base_target_accounts: usize,
    pub(crate) base_target_storage: usize,
    pub(crate) final_target_accounts: usize,
    pub(crate) final_target_storage: usize,
    pub(crate) supplemental_rounds: usize,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_write_path_experiment<Evm>(
    evm_config: &Evm,
    state_provider: &dyn StateProvider,
    block: &RecoveredBlock<BlockTy<EthPrimitives>>,
    prev_cache: &NetworkStateCache,
    account_window: u64,
    storage_window: u64,
    base_sidecar: &PartialStatelessSidecar,
    bundle_state: &BundleState,
    limits: &SidecarReexecLimits,
    output_path: &Path,
) -> Result<WritePathExperimentReport>
where
    Evm: ConfigureEvm<Primitives = EthPrimitives>,
{
    let mut proof_state_targets = base_sidecar.cache_miss_targets.clone();
    let write_targets = root_witness_targets_from_bundle(bundle_state);
    proof_state_targets.accounts.extend(write_targets.accounts);
    proof_state_targets.storage.extend(write_targets.storage);
    proof_state_targets.sort_dedup();

    let mut targets = state_targets_to_proof_targets(&proof_state_targets);
    let base_target_accounts = targets.len();
    let base_target_storage = targets.values().map(|slots| slots.len()).sum();
    let mut supplemental_rounds = 0usize;

    let (proof, computed_state_root) = loop {
        let proof = state_provider
            .multiproof(TrieInput::default(), targets.clone())
            .map_err(|err| eyre!("write-path multiproof generation failed: {err}"))?;
        let mut empty_trie_cache = PartialTrieNodeCache::new();

        match try_compute_trustless_state_root(proof.clone(), &mut empty_trie_cache, bundle_state) {
            Ok(root) => break (proof, root),
            Err(TrieTransitionError::ProofRequired(required)) => {
                let accounts_before = targets.len();
                let storage_before: usize = targets.values().map(|slots| slots.len()).sum();
                add_supplemental_targets(&mut targets, required);
                let storage_after: usize = targets.values().map(|slots| slots.len()).sum();
                supplemental_rounds += 1;

                if supplemental_rounds > MAX_SUPPLEMENTAL_PROOF_ROUNDS ||
                    (accounts_before == targets.len() && storage_before == storage_after)
                {
                    bail!(
                        "write-path proof completion made no progress after {supplemental_rounds} rounds"
                    );
                }
            }
            Err(err) => bail!("write-path root replay failed: {err}"),
        }
    };

    if computed_state_root != block.state_root() {
        bail!(
            "write-path generation root mismatch: expected {:?}, got {:?}",
            block.state_root(),
            computed_state_root
        );
    }

    let proof_stats = measure_multiproof_size(&proof, 0);
    let serializable_proof = SerializableMultiProof::from_multiproof(&proof);
    let serialized_proof = bincode::serialize(&serializable_proof)
        .map_err(|err| eyre!("failed to serialize write-path multiproof: {err}"))?;

    let mut experiment_sidecar = base_sidecar.clone();
    experiment_sidecar.witness.state =
        PartialExecutionWitnessState::MptMultiProof(serialized_proof);
    experiment_sidecar.witness_commitment = partial_witness_commitment(
        experiment_sidecar.parent_state_root,
        &experiment_sidecar.cache_miss_targets,
        &experiment_sidecar.witness,
    );
    experiment_sidecar.stats = measure_multiproof_size(
        &proof,
        experiment_sidecar.witness.codes.iter().map(|code| code.len()).sum(),
    );

    let wire_bytes = bincode::serialize(&experiment_sidecar)
        .map_err(|err| eyre!("failed to serialize write-path sidecar: {err}"))?;
    let wire_sidecar: PartialStatelessSidecar = bincode::deserialize(&wire_bytes)
        .map_err(|err| eyre!("failed to deserialize write-path sidecar: {err}"))?;

    let mut experiment_cache = NetworkStateCache::restore(
        prev_cache.accounts().clone(),
        prev_cache.storage().clone(),
        prev_cache.codes().clone(),
        prev_cache.current_block(),
        Box::new(LastNBlocksPolicy::new(account_window)),
        Box::new(LastNBlocksPolicy::new(storage_window)),
    );
    let mut empty_trie_cache = PartialTrieNodeCache::new();
    let verification = verify_and_apply_provider_assisted_sidecar(
        evm_config,
        state_provider,
        block,
        &mut experiment_cache,
        &wire_sidecar,
        limits,
        &mut empty_trie_cache,
    )
    .map_err(|err| eyre!("write-path wire replay failed: {err}"))?;
    if verification.trustless_state_root != Some(block.state_root()) {
        bail!(
            "write-path wire replay did not reproduce the block state root: {:?}",
            verification.trustless_state_root
        );
    }

    fs::write(output_path, &wire_bytes)
        .map_err(|err| eyre!("failed to write write-path sidecar {output_path:?}: {err}"))?;

    Ok(WritePathExperimentReport {
        block_number: block.number(),
        computed_state_root,
        wire_sidecar_bytes: wire_bytes.len(),
        proof_bytes: proof_stats.total_size_bytes,
        base_target_accounts,
        base_target_storage,
        final_target_accounts: targets.len(),
        final_target_storage: targets.values().map(|slots| slots.len()).sum(),
        supplemental_rounds,
    })
}

fn add_supplemental_targets(
    targets: &mut reth_trie_common::MultiProofTargets,
    required: Vec<TrieProofTarget>,
) {
    for target in required {
        match target {
            TrieProofTarget::Account(address) => {
                targets.entry(address).or_default();
            }
            TrieProofTarget::Storage { hashed_address, hashed_slot } => {
                targets.entry(hashed_address).or_default().insert(hashed_slot);
            }
        }
    }
}

pub(crate) fn save_write_path_report(
    output_path: &Path,
    report: &WritePathExperimentReport,
) -> Result<()> {
    let report_path = output_path.with_extension("write-path.json");
    let value = serde_json::json!({
        "block_number": report.block_number,
        "computed_state_root": report.computed_state_root,
        "wire_sidecar_bytes": report.wire_sidecar_bytes,
        "proof_bytes": report.proof_bytes,
        "base_target_accounts": report.base_target_accounts,
        "base_target_storage": report.base_target_storage,
        "final_target_accounts": report.final_target_accounts,
        "final_target_storage": report.final_target_storage,
        "supplemental_rounds": report.supplemental_rounds,
    });
    let bytes = serde_json::to_vec_pretty(&value)
        .map_err(|err| eyre!("failed to serialize write-path report: {err}"))?;
    fs::write(&report_path, bytes)
        .map_err(|err| eyre!("failed to write write-path report {report_path:?}: {err}"))
}
