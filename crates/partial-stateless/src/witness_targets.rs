use serde_json::json;

/// Source used to construct sidecar witness targets.
#[derive(Debug, Clone, Copy)]
pub enum TargetSourceKind {
    /// Uses post-execution bundle state; useful for plumbing, but not a complete execution witness.
    BundleChangedState,
    /// Intended complete execution-witness target source.
    FullExecutionWitness,
}

impl TargetSourceKind {
    /// Stable string representation.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BundleChangedState => "bundle_changed_state",
            Self::FullExecutionWitness => "full_execution_witness",
        }
    }

    /// Whether this source is complete enough to claim execution-witness coverage.
    pub const fn completeness(self) -> &'static str {
        match self {
            Self::BundleChangedState => {
                "plumbing_only_changed_state_not_complete_execution_witness"
            }
            Self::FullExecutionWitness => "intended_full_execution_witness_targets_placeholder",
        }
    }
}

/// Counts by witness target class.
#[derive(Debug, Clone, Copy, Default)]
pub struct TargetCounts {
    /// Account targets.
    pub accounts: usize,
    /// Storage slot targets.
    pub storage_slots: usize,
    /// Bytecode targets.
    pub code_hashes: usize,
    /// Ancestor header targets.
    pub headers: usize,
}

/// Storage target represented with raw address and slot strings.
#[derive(Debug, Clone)]
pub struct StorageTarget {
    /// Account address.
    pub address: String,
    /// Storage slot.
    pub slot: String,
}

/// Witness target set.
#[derive(Debug, Clone)]
pub struct WitnessTargets {
    /// Target source kind.
    pub source: TargetSourceKind,
    /// Account targets.
    pub accounts: Vec<String>,
    /// Storage targets.
    pub storage_slots: Vec<StorageTarget>,
    /// Code targets.
    pub code_hashes: Vec<String>,
    /// Header targets.
    pub header_numbers: Vec<u64>,
}

impl WitnessTargets {
    /// Create an empty target set for a source kind.
    pub const fn empty(source: TargetSourceKind) -> Self {
        Self {
            source,
            accounts: Vec::new(),
            storage_slots: Vec::new(),
            code_hashes: Vec::new(),
            header_numbers: Vec::new(),
        }
    }

    /// Return target counts.
    pub fn counts(&self) -> TargetCounts {
        TargetCounts {
            accounts: self.accounts.len(),
            storage_slots: self.storage_slots.len(),
            code_hashes: self.code_hashes.len(),
            headers: self.header_numbers.len(),
        }
    }

    /// Convert targets to the current JSON sidecar representation.
    pub fn to_json_value(&self) -> serde_json::Value {
        json!({
            "source": self.source.as_str(),
            "accounts": self.accounts,
            "storage_slots": self.storage_slots.iter().map(|target| {
                json!({
                    "address": target.address,
                    "slot": target.slot,
                })
            }).collect::<Vec<_>>(),
            "code_hashes": self.code_hashes,
            "header_numbers": self.header_numbers,
        })
    }
}

impl TargetCounts {
    /// Convert counts to JSON.
    pub fn to_json_value(self) -> serde_json::Value {
        json!({
            "accounts": self.accounts,
            "storage_slots": self.storage_slots,
            "code_hashes": self.code_hashes,
            "headers": self.headers,
        })
    }
}
