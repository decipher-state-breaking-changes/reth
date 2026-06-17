use crate::{CacheView, TargetCounts, WitnessTargets};
use serde_json::json;

/// Result of applying a cache view to full witness targets.
#[derive(Debug, Clone)]
pub struct FilteredTargets {
    /// Targets missing from the cache.
    pub missing_targets: WitnessTargets,
    /// Target count statistics.
    pub stats: TargetStats,
}

/// Target counts before and after cache filtering.
#[derive(Debug, Clone, Copy)]
pub struct TargetStats {
    /// Full target counts before cache filtering.
    pub full: TargetCounts,
    /// Cache hit counts.
    pub cache_hits: TargetCounts,
    /// Cache miss counts.
    pub cache_misses: TargetCounts,
}

/// Filter a full target set through a deterministic cache view.
pub fn filter_missing_targets(
    full_targets: &WitnessTargets,
    cache_view: &impl CacheView,
) -> FilteredTargets {
    let full = full_targets.counts();
    let mut cache_hits = TargetCounts::default();
    let mut missing_targets = WitnessTargets::empty(full_targets.source);

    for account in &full_targets.accounts {
        if cache_view.contains_account(account) {
            cache_hits.accounts += 1;
        } else {
            missing_targets.accounts.push(account.clone());
        }
    }

    for target in &full_targets.storage_slots {
        if cache_view.contains_storage(&target.address, &target.slot) {
            cache_hits.storage_slots += 1;
        } else {
            missing_targets.storage_slots.push(target.clone());
        }
    }

    for code_hash in &full_targets.code_hashes {
        if cache_view.contains_code(code_hash) {
            cache_hits.code_hashes += 1;
        } else {
            missing_targets.code_hashes.push(code_hash.clone());
        }
    }

    for number in &full_targets.header_numbers {
        if cache_view.contains_header(*number) {
            cache_hits.headers += 1;
        } else {
            missing_targets.header_numbers.push(*number);
        }
    }

    let cache_misses = missing_targets.counts();

    FilteredTargets { missing_targets, stats: TargetStats { full, cache_hits, cache_misses } }
}

impl TargetStats {
    /// Convert stats to JSON.
    pub fn to_json_value(self) -> serde_json::Value {
        json!({
            "full": self.full.to_json_value(),
            "cache_hits": self.cache_hits.to_json_value(),
            "cache_misses": self.cache_misses.to_json_value(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AllHitDebugView, NoCacheView, StorageTarget, TargetSourceKind};

    fn targets() -> WitnessTargets {
        WitnessTargets {
            source: TargetSourceKind::BundleChangedState,
            accounts: vec!["0x01".to_string()],
            storage_slots: vec![StorageTarget {
                address: "0x01".to_string(),
                slot: "0x02".to_string(),
            }],
            code_hashes: vec!["0x03".to_string()],
            header_numbers: vec![1],
        }
    }

    #[test]
    fn no_cache_marks_everything_missing() {
        let filtered = filter_missing_targets(&targets(), &NoCacheView::new(10));

        assert_eq!(filtered.stats.full.accounts, 1);
        assert_eq!(filtered.stats.cache_hits.accounts, 0);
        assert_eq!(filtered.stats.cache_misses.accounts, 1);
        assert_eq!(filtered.missing_targets.counts().storage_slots, 1);
        assert_eq!(filtered.missing_targets.counts().code_hashes, 1);
        assert_eq!(filtered.missing_targets.counts().headers, 1);
    }

    #[test]
    fn all_hit_debug_marks_everything_cached() {
        let filtered = filter_missing_targets(&targets(), &AllHitDebugView::new(10));

        assert_eq!(filtered.stats.full.accounts, 1);
        assert_eq!(filtered.stats.cache_hits.accounts, 1);
        assert_eq!(filtered.stats.cache_misses.accounts, 0);
        assert_eq!(filtered.missing_targets.counts().storage_slots, 0);
        assert_eq!(filtered.missing_targets.counts().code_hashes, 0);
        assert_eq!(filtered.missing_targets.counts().headers, 0);
    }
}
