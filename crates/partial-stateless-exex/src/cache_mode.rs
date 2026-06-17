use partial_stateless::{AllHitDebugView, CacheDescriptor, CacheView, NoCacheView};
use std::env;

const CACHE_MODE_ENV: &str = "CACHE_VOPS_CACHE_MODE";

#[derive(Debug, Clone, Copy)]
pub(crate) enum CacheMode {
    NoCache,
    AllHitDebug,
}

#[derive(Debug, Clone)]
pub(crate) enum StaticCacheView {
    NoCache(NoCacheView),
    AllHitDebug(AllHitDebugView),
}

impl CacheMode {
    pub(crate) fn from_env() -> eyre::Result<Self> {
        let raw = env::var(CACHE_MODE_ENV).unwrap_or_else(|_| "no_cache".to_string());
        match raw.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "no_cache" | "dev_all_miss" => Ok(Self::NoCache),
            "all_hit_debug" => Ok(Self::AllHitDebug),
            other => eyre::bail!(
                "unsupported {CACHE_MODE_ENV}={other}; expected no_cache or all_hit_debug"
            ),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::NoCache => "no_cache",
            Self::AllHitDebug => "all_hit_debug",
        }
    }

    pub(crate) fn view_for_block(self, cache_block: u64) -> StaticCacheView {
        match self {
            Self::NoCache => StaticCacheView::NoCache(NoCacheView::new(cache_block)),
            Self::AllHitDebug => StaticCacheView::AllHitDebug(AllHitDebugView::new(cache_block)),
        }
    }
}

impl CacheView for StaticCacheView {
    fn descriptor(&self) -> &CacheDescriptor {
        match self {
            Self::NoCache(view) => view.descriptor(),
            Self::AllHitDebug(view) => view.descriptor(),
        }
    }

    fn contains_account(&self, address: &str) -> bool {
        match self {
            Self::NoCache(view) => view.contains_account(address),
            Self::AllHitDebug(view) => view.contains_account(address),
        }
    }

    fn contains_storage(&self, address: &str, slot: &str) -> bool {
        match self {
            Self::NoCache(view) => view.contains_storage(address, slot),
            Self::AllHitDebug(view) => view.contains_storage(address, slot),
        }
    }

    fn contains_code(&self, code_hash: &str) -> bool {
        match self {
            Self::NoCache(view) => view.contains_code(code_hash),
            Self::AllHitDebug(view) => view.contains_code(code_hash),
        }
    }

    fn contains_header(&self, number: u64) -> bool {
        match self {
            Self::NoCache(view) => view.contains_header(number),
            Self::AllHitDebug(view) => view.contains_header(number),
        }
    }
}
