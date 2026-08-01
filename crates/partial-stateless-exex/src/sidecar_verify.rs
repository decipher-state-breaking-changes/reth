use crate::{
    sidecar_io::{read_sidecar, sidecar_path},
    sidecar_reexec::{
        verify_and_apply_provider_assisted_sidecar, SidecarReexecLimits, TrieCacheDisposition,
    },
    CacheConfig,
};
use partial_stateless::{
    last_n_blocks_cache_policy_id, network_cache::NetworkStateCache, PartialTrieNodeCache,
    ReadyParent,
};
use reth_ethereum::EthPrimitives;
use reth_evm::ConfigureEvm;
use reth_primitives_traits::{AlloyBlockHeader, BlockTy, RecoveredBlock};
use reth_provider::StateProvider;
use std::{path::Path, time::Duration};
use tracing::{info, warn};

pub(crate) fn verify_live_sidecar<Evm>(
    evm_config: &Evm,
    state_provider: &dyn StateProvider,
    block: &RecoveredBlock<BlockTy<EthPrimitives>>,
    cache: &mut NetworkStateCache,
    trie_cache: &mut PartialTrieNodeCache,
    config: &CacheConfig,
    sidecar_dir: &Path,
    limits: &SidecarReexecLimits,
    wait: Duration,
    ready_parent: Option<&ReadyParent>,
) -> eyre::Result<()>
where
    Evm: ConfigureEvm<Primitives = EthPrimitives>,
{
    let block_number = block.number();
    let block_hash = block.hash();
    let parent_block_number = block_number.saturating_sub(1);
    if cache.current_block() != parent_block_number {
        return Err(eyre::eyre!(
            "verifier cache is not synced to parent: cache_block={}, expected_parent={}",
            cache.current_block(),
            parent_block_number
        ));
    }

    let path = sidecar_path(sidecar_dir, block_number, block_hash);
    let (bytes, sidecar) = read_sidecar(&path, wait)?;

    let expected_policy_id =
        last_n_blocks_cache_policy_id(config.account_window, config.storage_window);
    if sidecar.cache_policy_id != expected_policy_id {
        return Err(eyre::eyre!(
            "sidecar cache_policy_id mismatch: expected {:?}, got {:?}",
            expected_policy_id,
            sidecar.cache_policy_id
        ));
    }

    // This must run before `TrieCacheDisposition::Commit` verification: that call advances the
    // flat cache and installs the child trie on success, and a rejection must leave both caches
    // exactly at the parent generation.
    check_ready_anchor(ready_parent, &sidecar)?;

    let report = verify_and_apply_provider_assisted_sidecar(
        evm_config,
        state_provider,
        block,
        cache,
        &sidecar,
        expected_policy_id,
        limits,
        trie_cache,
        TrieCacheDisposition::Commit,
    )?;

    if !report.root_witness_completeness.trustless_root_ready {
        warn!(
            target: "partial_stateless",
            block = block_number,
            partial_state_trustless_verification_ready = false,
            missing_account_paths = report.root_witness_completeness.missing_account_paths.len(),
            missing_storage_paths = report.root_witness_completeness.missing_storage_paths.len(),
            "Partial-state node trustless verification is not ready; current state_root check is provider-assisted"
        );
    }

    match report.trustless_state_root {
        Some(root) => info!(
            target: "partial_stateless",
            block = block_number,
            accepted_as_partial = ready_parent.is_some(),
            trustless_state_root = ?root,
            "Trustless state root VERIFIED (trie node cache + witness only)"
        ),
        None => info!(
            target: "partial_stateless",
            block = block_number,
            trie_warm_nodes = trie_cache.warm_node_count(),
            tracked_accounts = trie_cache.tracked_account_count(),
            "Trustless state root unavailable — trie node cache not warm enough this block (blind path)"
        ),
    }

    let stats = &report.cache_update;
    info!(
        target: "partial_stateless",
        block = block_number,
        path = %path.display(),
        sidecar_bytes = bytes.len(),
        partial_state_trustless_verification_ready = report
            .root_witness_completeness
            .trustless_root_ready,
        computed_state_root = ?report.computed_state_root,
        reexec_accounts = report.actual_accessed.accounts.len(),
        reexec_storage = report.actual_accessed.storage.len(),
        reexec_codes = report.actual_accessed.codes.len(),
        expected_miss_accounts = report.expected_miss.accounts.len(),
        expected_miss_storage = report.expected_miss.storage.len(),
        expected_miss_codes = report.expected_miss.code_hashes.len(),
        next_cache_root = ?report.next_cache_anchor.cache_root,
        accounts_added = stats.accounts_added,
        accounts_refreshed = stats.accounts_refreshed,
        accounts_evicted = stats.accounts_evicted,
        storage_added = stats.storage_added,
        storage_refreshed = stats.storage_refreshed,
        storage_evicted = stats.storage_evicted,
        "Live sidecar verification succeeded"
    );

    Ok(())
}

/// Rejects a sidecar whose previous anchor disagrees with the tracker-authenticated Ready parent.
///
/// Witness-only execution and the state-root check are fail-closed in every readiness state; what
/// readiness adds is an independent view of which parent the caches describe. The verification
/// step compares the sidecar's previous anchor against the cache itself, so a cache that drifted
/// from what the tracker authenticated would still vouch for the wrong branch. Taking only the
/// sidecar and the tracker's parent — never the caches — keeps this check incapable of the
/// mutation its caller must not have performed yet.
fn check_ready_anchor(
    ready_parent: Option<&ReadyParent>,
    sidecar: &partial_stateless::PartialStatelessSidecar,
) -> eyre::Result<()> {
    if let Some(parent) = ready_parent {
        if parent.anchor != sidecar.prev_cache_anchor {
            return Err(eyre::eyre!(
                "sidecar at block {} names previous anchor {:?}, but the cache is authenticated against {:?}",
                sidecar.block_number,
                sidecar.prev_cache_anchor,
                parent.anchor
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::B256;
    use partial_stateless::{
        CacheAnchor, PartialExecutionWitness, PartialExecutionWitnessState,
        PartialStatelessSidecar, StateTargetSet, WitnessResult, WitnessTargets,
    };

    fn anchor(tag: u8) -> CacheAnchor {
        CacheAnchor {
            block_number: 99,
            block_hash: B256::repeat_byte(tag),
            cache_policy_id: B256::repeat_byte(0x22),
            cache_root: B256::repeat_byte(0x33),
        }
    }

    fn sidecar_with_prev_anchor(prev: CacheAnchor) -> PartialStatelessSidecar {
        PartialStatelessSidecar {
            parent_hash: prev.block_hash,
            parent_state_root: B256::repeat_byte(0x55),
            block_hash: B256::repeat_byte(0x44),
            block_number: 100,
            cache_block: 99,
            cache_policy_id: prev.cache_policy_id,
            prev_cache_anchor: prev,
            next_cache_anchor: CacheAnchor { block_number: 100, ..prev },
            cache_policy_metadata: String::new(),
            cache_miss_targets: StateTargetSet::default(),
            witness_commitment: B256::ZERO,
            miss_manifest: WitnessTargets {
                missed_accounts: vec![],
                missed_storage: vec![],
                missed_code_hashes: vec![],
            },
            witness: PartialExecutionWitness {
                state: PartialExecutionWitnessState::MptTransitionNodes(Vec::new()),
                codes: Vec::new(),
                keys: Vec::new(),
                headers: Vec::new(),
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
        }
    }

    #[test]
    fn a_ready_parent_must_match_the_sidecar_previous_anchor() {
        let prev = anchor(0x11);
        let sidecar = sidecar_with_prev_anchor(prev);
        let matching = ReadyParent {
            anchor: prev,
            trie_state_root: B256::repeat_byte(0x66),
            replay_depth: 61,
        };
        assert!(check_ready_anchor(Some(&matching), &sidecar).is_ok());

        // A drifted cache: the tracker vouches for a different parent than the sidecar names.
        let drifted = ReadyParent { anchor: anchor(0xfe), ..matching };
        assert!(check_ready_anchor(Some(&drifted), &sidecar).is_err());

        // Warming: no Ready parent exists and the provider-assisted path stays open.
        assert!(check_ready_anchor(None, &sidecar).is_ok());
    }
}
