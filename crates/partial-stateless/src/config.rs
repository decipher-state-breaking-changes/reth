//! The cache windows a node runs, and everything derived from them.

use crate::{
    network_cache::NetworkStateCache,
    policy::{CachePolicy, LastNBlocksPolicy},
    readiness::CacheReadinessTracker,
    sidecar::last_n_blocks_cache_policy_id,
};
use alloy_primitives::B256;

/// Configuration for the partial statelessness cache.
///
/// Lives here rather than beside the ExEx because everything it decides — the two eviction
/// policies and the identifier peers compare anchors under — is protocol configuration that a
/// database-free consumer needs as much as a full node does. A replay driver restoring a snapshot
/// derives its policies from exactly this type, so the two cannot disagree about what a policy
/// identifier means.
#[derive(Debug, Clone, Copy)]
pub struct CacheConfig {
    /// Window size for account eviction policy (in blocks).
    pub account_window: u64,
    /// Window size for storage/code eviction policy (in blocks).
    pub storage_window: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self { account_window: 60, storage_window: 30 }
    }
}

impl CacheConfig {
    /// A cold cache at height zero.
    pub fn new_cache(&self) -> NetworkStateCache {
        self.new_cache_at(0)
    }

    /// An empty cache that claims to sit at `current_block`.
    pub fn new_cache_at(&self, current_block: u64) -> NetworkStateCache {
        NetworkStateCache::restore(
            Default::default(),
            Default::default(),
            Default::default(),
            current_block,
            self.account_policy(),
            self.storage_policy(),
        )
    }

    /// The account eviction policy this configuration runs.
    ///
    /// Bootstrap binds a snapshot to a policy *identifier* but cannot check the policy object
    /// behind it (failure mode 11), so every caller that needs one must derive it here rather
    /// than construct its own.
    pub fn account_policy(&self) -> Box<dyn CachePolicy> {
        Box::new(LastNBlocksPolicy::new(self.account_window))
    }

    /// The storage/code eviction policy this configuration runs.
    pub fn storage_policy(&self) -> Box<dyn CachePolicy> {
        Box::new(LastNBlocksPolicy::new(self.storage_window))
    }

    /// Identifier peers compare cache anchors under.
    pub fn cache_policy_id(&self) -> B256 {
        last_n_blocks_cache_policy_id(self.account_window, self.storage_window)
    }

    /// Blocks that must be replayed before the advertised window is genuinely populated.
    ///
    /// The larger of the two windows: the cache only holds everything its policy identifier
    /// advertises once the longer of the two has been replayed.
    pub const fn max_window(&self) -> u64 {
        if self.account_window > self.storage_window {
            self.account_window
        } else {
            self.storage_window
        }
    }

    /// A readiness tracker for a cold cache under this configuration.
    pub fn new_readiness_tracker(&self) -> CacheReadinessTracker {
        CacheReadinessTracker::new(self.max_window(), self.cache_policy_id())
    }
}
