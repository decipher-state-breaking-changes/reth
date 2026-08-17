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
    /// The variables that override the compiled defaults, and the defaults themselves.
    pub const ACCOUNT_WINDOW_VAR: &'static str = "PS_ACCOUNT_WINDOW";
    pub const STORAGE_WINDOW_VAR: &'static str = "PS_STORAGE_WINDOW";

    /// Reads the windows from the environment, or fails.
    ///
    /// Strict on purpose, and fail-fast rather than fall-back. The windows are not a tuning knob:
    /// they decide the cache policy identifier, which is the thing every peer compares anchors
    /// under and every sidecar is bound to. A run that silently fell back to 60/30 after a
    /// mistyped variable would produce a whole arm of measurements labelled with a window it never
    /// ran, and nothing downstream could tell — the manifest would record the variable that was
    /// set, the policy identifier would record the window that was used, and only someone
    /// comparing the two by hand would ever notice.
    ///
    /// Unset means the compiled default, which is the one case that is not ambiguous.
    pub fn from_env() -> Result<Self, CacheConfigEnvError> {
        let default = Self::default();
        Ok(Self {
            account_window: read_window(Self::ACCOUNT_WINDOW_VAR, default.account_window)?,
            storage_window: read_window(Self::STORAGE_WINDOW_VAR, default.storage_window)?,
        })
    }

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

/// Why a window could not be read from the environment.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CacheConfigEnvError {
    /// The variable was set to something that is not a block count a cache can hold.
    #[error("{var} is `{value}`, which is not a positive block count")]
    NotAPositiveWindow {
        /// The variable that was set.
        var: &'static str,
        /// What it was set to, quoted back so a stray space is visible.
        value: String,
    },
}

/// Reads one window, treating everything that is not a plain positive integer as an error.
///
/// Zero is refused rather than treated as "no window": a `LastNBlocksPolicy(0)` evicts what it was
/// just given, so every block would miss on every access and the run would be measuring a
/// stateless validator wearing a partial-stateless label. Leading spaces, `+60`, `60blocks` and
/// `1.5` are refused for the duller reason that a value nobody typed on purpose is a typo.
fn read_window(var: &'static str, default: u64) -> Result<u64, CacheConfigEnvError> {
    let raw = match std::env::var(var) {
        Ok(raw) => raw,
        Err(std::env::VarError::NotPresent) => return Ok(default),
        Err(std::env::VarError::NotUnicode(raw)) => {
            return Err(CacheConfigEnvError::NotAPositiveWindow {
                var,
                value: raw.to_string_lossy().into_owned(),
            })
        }
    };
    // Digits only, checked before the parse rather than by it: `u64::from_str` accepts `+60`,
    // and a value nobody would type on purpose is a typo whichever way it happens to parse.
    let digits_only = !raw.is_empty() && raw.bytes().all(|byte| byte.is_ascii_digit());
    match raw.parse::<u64>() {
        Ok(window) if digits_only && window > 0 => Ok(window),
        _ => Err(CacheConfigEnvError::NotAPositiveWindow { var, value: raw }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `from_env` reads process-wide state, so the cases share one lock rather than racing.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn with_windows<T>(
        account: Option<&str>,
        storage: Option<&str>,
        body: impl FnOnce() -> T,
    ) -> T {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        // SAFETY: the lock above serialises every test in this module, and nothing else in this
        // crate reads these variables.
        unsafe {
            match account {
                Some(value) => std::env::set_var(CacheConfig::ACCOUNT_WINDOW_VAR, value),
                None => std::env::remove_var(CacheConfig::ACCOUNT_WINDOW_VAR),
            }
            match storage {
                Some(value) => std::env::set_var(CacheConfig::STORAGE_WINDOW_VAR, value),
                None => std::env::remove_var(CacheConfig::STORAGE_WINDOW_VAR),
            }
        }
        let out = body();
        unsafe {
            std::env::remove_var(CacheConfig::ACCOUNT_WINDOW_VAR);
            std::env::remove_var(CacheConfig::STORAGE_WINDOW_VAR);
        }
        out
    }

    /// The default is the shipped operating point, and an absent variable must not move it.
    #[test]
    fn an_unset_environment_is_the_compiled_default() {
        let config =
            with_windows(None, None, || CacheConfig::from_env().expect("no variables set"));
        assert_eq!((config.account_window, config.storage_window), (60, 30));
    }

    #[test]
    fn set_windows_are_read_independently() {
        let config =
            with_windows(Some("120"), None, || CacheConfig::from_env().expect("one variable set"));
        assert_eq!((config.account_window, config.storage_window), (120, 30));

        let config = with_windows(Some("15"), Some("240"), || {
            CacheConfig::from_env().expect("both variables set")
        });
        assert_eq!((config.account_window, config.storage_window), (15, 240));
    }

    /// Every one of these has been typed by someone, and none of them means 60.
    #[test]
    fn anything_that_is_not_a_positive_integer_is_refused() {
        for value in ["0", "", " 60", "60 ", "+60", "-1", "1.5", "60blocks", "sixty"] {
            let err = with_windows(Some(value), None, || {
                CacheConfig::from_env().expect_err("a window that is not a positive integer")
            });
            assert_eq!(
                err,
                CacheConfigEnvError::NotAPositiveWindow {
                    var: CacheConfig::ACCOUNT_WINDOW_VAR,
                    value: value.to_string(),
                },
                "{value:?} must be refused rather than rounded to something"
            );
        }
    }

    /// The storage window has its own variable and its own message, or a failing run would send
    /// the operator to the wrong one.
    #[test]
    fn the_storage_window_names_itself_when_it_is_the_broken_one() {
        let err = with_windows(None, Some("0"), || {
            CacheConfig::from_env().expect_err("zero is not a window")
        });
        assert_eq!(
            err,
            CacheConfigEnvError::NotAPositiveWindow {
                var: CacheConfig::STORAGE_WINDOW_VAR,
                value: "0".to_string(),
            }
        );
    }

    /// What the gap tolerance and the readiness tracker are both derived from: a cache is only
    /// fully stale once the *longer* window has gone by, whichever of the two that is.
    #[test]
    fn the_maximum_window_follows_whichever_window_is_larger() {
        assert_eq!(CacheConfig { account_window: 60, storage_window: 30 }.max_window(), 60);
        assert_eq!(CacheConfig { account_window: 15, storage_window: 240 }.max_window(), 240);
        assert_eq!(CacheConfig { account_window: 30, storage_window: 30 }.max_window(), 30);
    }
}
