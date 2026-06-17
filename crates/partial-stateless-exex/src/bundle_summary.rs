use reth_execution_types::ExecutionOutcome;

/// Manifest-only summary of Reth's post-execution bundle.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BundleSummary {
    pub(crate) first_block: u64,
    pub(crate) last_block: u64,
    pub(crate) block_count: usize,
    pub(crate) changed_accounts: usize,
    pub(crate) changed_account_infos: usize,
    pub(crate) changed_storage_slots: usize,
    pub(crate) code_hash_changed_accounts: usize,
    pub(crate) contract_bytecodes: usize,
    pub(crate) state_size: usize,
    pub(crate) reverts_size: usize,
}

impl BundleSummary {
    pub(crate) fn from_execution_outcome<T>(outcome: &ExecutionOutcome<T>) -> Self {
        let bundle = outcome.state();

        Self {
            first_block: outcome.first_block(),
            last_block: outcome.last_block(),
            block_count: outcome.len(),
            changed_accounts: bundle.state.len(),
            changed_account_infos: bundle
                .state
                .values()
                .filter(|account| account.is_info_changed())
                .count(),
            changed_storage_slots: bundle.state.values().map(|account| account.storage.len()).sum(),
            code_hash_changed_accounts: bundle
                .state
                .values()
                .filter(|account| account.is_contract_changed())
                .count(),
            contract_bytecodes: bundle.contracts.len(),
            state_size: bundle.state_size,
            reverts_size: bundle.reverts_size,
        }
    }
}
