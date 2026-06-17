use serde_json::json;

/// Deterministic cache profile assumed by a partial execution witness sidecar.
#[derive(Debug, Clone)]
pub struct CacheDescriptor {
    /// Stable cache policy identifier.
    pub policy_id: &'static str,
    /// Cache policy schema version.
    pub policy_version: u32,
    /// Block number the cache view is anchored to.
    pub cache_block: u64,
    /// Optional account retention window.
    pub account_window: Option<u64>,
    /// Optional storage retention window.
    pub storage_window: Option<u64>,
    /// Code retention policy descriptor.
    pub code_policy: &'static str,
    /// Header retention policy descriptor.
    pub header_policy: &'static str,
}

impl CacheDescriptor {
    /// Descriptor for a cold cache where every target is a miss.
    pub const fn no_cache(cache_block: u64) -> Self {
        Self {
            policy_id: "no_cache",
            policy_version: 1,
            cache_block,
            account_window: None,
            storage_window: None,
            code_policy: "all_miss",
            header_policy: "all_miss",
        }
    }

    /// Descriptor for a debug cache where every target is a hit.
    pub const fn all_hit_debug(cache_block: u64) -> Self {
        Self {
            policy_id: "all_hit_debug",
            policy_version: 1,
            cache_block,
            account_window: None,
            storage_window: None,
            code_policy: "all_hit_debug",
            header_policy: "all_hit_debug",
        }
    }

    /// Convert the descriptor to the current JSON sidecar representation.
    pub fn to_json_value(&self) -> serde_json::Value {
        json!({
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "cache_block": self.cache_block,
            "account_window": self.account_window,
            "storage_window": self.storage_window,
            "code_policy": self.code_policy,
            "header_policy": self.header_policy,
        })
    }
}
