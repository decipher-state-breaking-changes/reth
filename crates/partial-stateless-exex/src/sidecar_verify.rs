use crate::{
    sidecar_io::{read_sidecar, sidecar_path},
    sidecar_reexec::{verify_and_apply_trustless_sidecar, SidecarReexecLimits},
    CacheConfig,
};
use partial_stateless::{last_n_blocks_cache_policy_id, network_cache::NetworkStateCache};
use reth_ethereum::EthPrimitives;
use reth_evm::ConfigureEvm;
use reth_primitives_traits::{AlloyBlockHeader, BlockTy, RecoveredBlock};
use std::{path::Path, time::Duration};
use tracing::info;

#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_live_sidecar<Evm>(
    evm_config: &Evm,
    block: &RecoveredBlock<BlockTy<EthPrimitives>>,
    parent_state_root: alloy_primitives::B256,
    cache: &mut NetworkStateCache,
    config: &CacheConfig,
    sidecar_dir: &Path,
    limits: &SidecarReexecLimits,
    wait: Duration,
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

    let report = verify_and_apply_trustless_sidecar(
        evm_config,
        block,
        parent_state_root,
        cache,
        &sidecar,
        limits,
    )?;

    let stats = &report.cache_update;
    info!(
        target: "partial_stateless",
        block = block_number,
        path = %path.display(),
        sidecar_bytes = bytes.len(),
        partial_state_trustless_verification_ready = true,
        computed_state_root = ?report.computed_state_root,
        reexec_accounts = report.actual_accessed.accounts.len(),
        reexec_storage = report.actual_accessed.storage.len(),
        reexec_codes = report.actual_accessed.codes.len(),
        expected_miss_accounts = report.expected_miss.accounts.len(),
        expected_miss_storage = report.expected_miss.storage.len(),
        expected_miss_codes = report.expected_miss.code_hashes.len(),
        write_accounts = report.write_targets.accounts.len(),
        write_storage = report.write_targets.storage.len(),
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
