use alloy_primitives::B256;
use partial_stateless::{StorageTarget, TargetSourceKind, WitnessTargets};
use reth_execution_types::ExecutionOutcome;
use std::env;

const TARGET_SOURCE_ENV: &str = "CACHE_VOPS_TARGET_SOURCE";

#[derive(Debug, Clone, Copy)]
pub(crate) enum TargetSourceMode {
    BundleChangedState,
    FullExecutionWitness,
}

pub(crate) trait TargetSource<T> {
    fn kind(&self) -> TargetSourceKind;

    fn collect_targets(&self, outcome: &ExecutionOutcome<T>) -> eyre::Result<WitnessTargets>;
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct BundleChangedStateTargetSource;

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct FullExecutionWitnessTargetSource;

impl TargetSourceMode {
    pub(crate) fn from_env() -> eyre::Result<Self> {
        let raw = env::var(TARGET_SOURCE_ENV)
            .unwrap_or_else(|_| TargetSourceKind::BundleChangedState.as_str().to_string());

        Self::parse_raw(&raw)
    }

    pub(crate) fn parse_raw(raw: &str) -> eyre::Result<Self> {
        match raw.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "bundle_changed_state" => Ok(Self::BundleChangedState),
            "full_execution_witness" | "execution_witness_record" => Ok(Self::FullExecutionWitness),
            other => eyre::bail!(
                "unsupported {TARGET_SOURCE_ENV}={other}; expected bundle_changed_state or full_execution_witness"
            ),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::BundleChangedState => TargetSourceKind::BundleChangedState.as_str(),
            Self::FullExecutionWitness => TargetSourceKind::FullExecutionWitness.as_str(),
        }
    }
}

pub(crate) fn collect_targets<T>(
    mode: TargetSourceMode,
    outcome: &ExecutionOutcome<T>,
) -> eyre::Result<WitnessTargets> {
    match mode {
        TargetSourceMode::BundleChangedState => {
            BundleChangedStateTargetSource.collect_targets(outcome)
        }
        TargetSourceMode::FullExecutionWitness => {
            FullExecutionWitnessTargetSource.collect_targets(outcome)
        }
    }
}

impl<T> TargetSource<T> for BundleChangedStateTargetSource {
    fn kind(&self) -> TargetSourceKind {
        TargetSourceKind::BundleChangedState
    }

    fn collect_targets(&self, outcome: &ExecutionOutcome<T>) -> eyre::Result<WitnessTargets> {
        let bundle = outcome.state();

        let mut accounts = Vec::with_capacity(bundle.state.len());
        let mut storage_slots = Vec::new();
        for (address, account) in &bundle.state {
            let address = format!("{address:?}");
            accounts.push(address.clone());

            storage_slots.extend(account.storage.keys().map(|slot| StorageTarget {
                address: address.clone(),
                slot: format!("{:?}", B256::from(*slot)),
            }));
        }

        let mut code_hashes =
            bundle.contracts.keys().map(|code_hash| format!("{code_hash:?}")).collect::<Vec<_>>();

        accounts.sort();
        storage_slots.sort_by(|a, b| a.address.cmp(&b.address).then_with(|| a.slot.cmp(&b.slot)));
        code_hashes.sort();

        Ok(WitnessTargets {
            source: <Self as TargetSource<T>>::kind(self),
            accounts,
            storage_slots,
            code_hashes,
            header_numbers: Vec::new(),
        })
    }
}

impl<T> TargetSource<T> for FullExecutionWitnessTargetSource {
    fn kind(&self) -> TargetSourceKind {
        TargetSourceKind::FullExecutionWitness
    }

    fn collect_targets(&self, _outcome: &ExecutionOutcome<T>) -> eyre::Result<WitnessTargets> {
        eyre::bail!(
            "target source {} is unsupported in this minimal ExEx; wire full execution witness target source before enabling it",
            <Self as TargetSource<T>>::kind(self).as_str()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_execution_witness_source_uses_concept_name() {
        let mode = TargetSourceMode::parse_raw("full_execution_witness").unwrap();

        assert_eq!(mode.as_str(), "full_execution_witness");
    }

    #[test]
    fn execution_witness_record_alias_maps_to_full_execution_witness() {
        let mode = TargetSourceMode::parse_raw("execution_witness_record").unwrap();

        assert_eq!(mode.as_str(), "full_execution_witness");
    }

    #[test]
    fn unsupported_source_names_full_execution_witness() {
        let err = TargetSourceMode::parse_raw("bad_source").unwrap_err().to_string();

        assert!(err.contains("full_execution_witness"));
        assert!(!err.contains("execution_witness_record"));
    }
}
