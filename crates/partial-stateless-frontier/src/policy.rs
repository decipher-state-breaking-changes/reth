//! The policies a run compares, and the two cache pairs each one carries.
//!
//! Every policy gets a **builder** pair and a **validator** pair, advanced in lockstep and never
//! shared. That is the same arrangement a live builder-verifier node runs, and it is what makes the
//! per-block validation a check rather than a restatement: the builder pair produces the sidecar,
//! and a pair that has only ever seen sidecars consumes it. A single shared pair would accept every
//! sidecar it produced by construction.

use eyre::Context as _;
use partial_stateless::{CacheConfig, CacheTrieRepr, NetworkStateCache, PartialTrieNodeCache};
use std::str::FromStr;

/// One arm of a comparison: a cache policy, or the no-cache baseline.
///
/// Weak is an arm rather than a flag because it has to be measured the same way the policies are —
/// in the same rotation, over the same blocks, through the same validator. A baseline computed off
/// to the side is a baseline nobody can check against the thing it is a baseline for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ArmKind {
    /// No persistent cache at all: every accessed key is a miss, every block starts cold.
    ///
    /// Ordered first so it appears first in a sorted arm list, which is where a reader looks for
    /// the baseline.
    Weak,
    /// A `LastNBlocks` window pair.
    Policy(PolicySpec),
}

impl ArmKind {
    /// The label this arm appears under in reports.
    pub fn label(&self) -> String {
        match self {
            Self::Weak => "weak".to_string(),
            Self::Policy(spec) => spec.label(),
        }
    }

    /// Blocks that must be replayed before this arm is at steady state.
    ///
    /// Zero for Weak: it holds nothing between blocks, so there is nothing to warm. It still
    /// replays the warm-up blocks — it is in the rotation, and an arm that skipped them would be
    /// measured over a different block set than the rest.
    pub const fn warmup_floor(&self) -> u64 {
        match self {
            Self::Weak => 0,
            Self::Policy(spec) => spec.warmup_floor(),
        }
    }
}

impl FromStr for ArmKind {
    type Err = eyre::Report;

    /// Parses `weak`, or `account/storage`.
    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        if raw.trim().eq_ignore_ascii_case("weak") {
            return Ok(Self::Weak)
        }
        raw.parse::<PolicySpec>().map(Self::Policy)
    }
}

/// One `LastNBlocks` window pair, as a run names it on the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PolicySpec {
    /// Blocks an account stays cached for.
    pub account_window: u64,
    /// Blocks a storage slot or bytecode stays cached for.
    pub storage_window: u64,
}

impl PolicySpec {
    /// The cache configuration this policy builds under.
    pub const fn config(&self) -> CacheConfig {
        CacheConfig { account_window: self.account_window, storage_window: self.storage_window }
    }

    /// Blocks that must be replayed before the advertised window is genuinely populated.
    pub const fn warmup_floor(&self) -> u64 {
        if self.account_window > self.storage_window {
            self.account_window
        } else {
            self.storage_window
        }
    }

    /// The label this policy appears under in reports: `account/storage`.
    pub fn label(&self) -> String {
        format!("{}/{}", self.account_window, self.storage_window)
    }
}

impl FromStr for PolicySpec {
    type Err = eyre::Report;

    /// Parses `account/storage`, e.g. `60/30`.
    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        let (account, storage) = raw.split_once('/').ok_or_else(|| {
            eyre::eyre!("policy `{raw}` must be written as <account_window>/<storage_window>")
        })?;
        let account_window = account
            .trim()
            .parse::<u64>()
            .wrap_err_with(|| format!("policy `{raw}` has a non-numeric account window"))?;
        let storage_window = storage
            .trim()
            .parse::<u64>()
            .wrap_err_with(|| format!("policy `{raw}` has a non-numeric storage window"))?;
        if account_window == 0 || storage_window == 0 {
            eyre::bail!("policy `{raw}` must have both windows >= 1")
        }
        Ok(Self { account_window, storage_window })
    }
}

/// One arm's live state through a run.
pub struct PolicyState {
    /// Which arm this is.
    pub kind: ArmKind,
    /// The configuration derived from it, held so it is built once rather than per block.
    ///
    /// For [`ArmKind::Weak`] this is only the source of a policy identifier — the caches below are
    /// rebuilt cold every block, so no window ever applies.
    pub config: CacheConfig,
    /// The pair the sidecar is built from.
    pub builder_cache: NetworkStateCache,
    /// The trie generation the builder pair is anchored to.
    pub builder_trie: PartialTrieNodeCache,
    /// The pair the sidecar is validated against — it has never seen anything but sidecars.
    pub validator_cache: NetworkStateCache,
    /// The trie generation the validator pair is anchored to.
    pub validator_trie: PartialTrieNodeCache,
}

impl std::fmt::Debug for PolicyState {
    /// Names the arm and its sizes. The caches themselves are megabytes of map and there is no
    /// context in which printing one is the useful thing to do.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolicyState")
            .field("kind", &self.kind)
            .field("builder_cache_bytes", &self.builder_cache.estimated_memory_bytes())
            .field("builder_trie_bytes", &self.builder_trie.estimated_memory_bytes())
            .field("validator_cache_bytes", &self.validator_cache.estimated_memory_bytes())
            .field("validator_trie_bytes", &self.validator_trie.estimated_memory_bytes())
            .finish()
    }
}

impl PolicyState {
    /// A cold pair for `kind`, claiming to sit at `parent_block`, on the default representation.
    pub fn cold_at(kind: ArmKind, parent_block: u64) -> Self {
        Self::cold_at_with_repr(kind, parent_block, CacheTrieRepr::default())
    }

    /// A cold pair for `kind` whose trie generations run on `repr`.
    pub fn cold_at_with_repr(kind: ArmKind, parent_block: u64, repr: CacheTrieRepr) -> Self {
        // Weak's windows are never applied — its caches are discarded every block — but it still
        // needs *some* configuration to name a policy identifier with, and the default is the one
        // value that is not a claim about any policy under comparison.
        let config = match kind {
            ArmKind::Weak => CacheConfig::default(),
            ArmKind::Policy(spec) => spec.config(),
        };
        Self {
            kind,
            builder_cache: config.new_cache_at(parent_block),
            builder_trie: PartialTrieNodeCache::new_with_repr(repr),
            validator_cache: config.new_cache_at(parent_block),
            validator_trie: PartialTrieNodeCache::new_with_repr(repr),
            config,
        }
    }

    /// Returns both pairs to a cold generation at `parent_block`.
    ///
    /// Weak's whole definition: a validator that held nothing when the block arrived. Rebuilding
    /// rather than clearing, because "cold" has to mean the same object a first-ever run would
    /// have, not one carrying whatever a previous block left behind.
    pub fn reset_cold_at(&mut self, parent_block: u64) {
        let repr = self.builder_trie.repr();
        self.builder_cache = self.config.new_cache_at(parent_block);
        self.builder_trie = PartialTrieNodeCache::new_with_repr(repr);
        self.validator_cache = self.config.new_cache_at(parent_block);
        self.validator_trie = PartialTrieNodeCache::new_with_repr(repr);
    }
}

/// The order arms are visited in for `block_index`, rotated so no arm keeps a fixed slot.
///
/// Whichever policy runs first on a block pays the cold read of whatever the others then find
/// warm — in the allocator, in the page cache, in the branch predictor. Left fixed, that is a
/// systematic advantage handed to a particular policy, and it would land in exactly the column a
/// policy comparison reports.
pub fn rotated_order(policies: usize, block_index: usize) -> impl Iterator<Item = usize> {
    let offset = if policies == 0 { 0 } else { block_index % policies };
    (0..policies).map(move |slot| (slot + offset) % policies)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_arm_is_weak_or_two_windows() {
        assert_eq!("weak".parse::<ArmKind>().unwrap(), ArmKind::Weak);
        assert_eq!("WEAK".parse::<ArmKind>().unwrap(), ArmKind::Weak);
        assert_eq!(
            "60/30".parse::<ArmKind>().unwrap(),
            ArmKind::Policy(PolicySpec { account_window: 60, storage_window: 30 })
        );
        assert!("nonsense".parse::<ArmKind>().is_err());
    }

    /// Weak sorts first, which is where a reader looks for a baseline.
    #[test]
    fn the_baseline_sorts_ahead_of_the_policies() {
        let mut arms = [
            "120/45".parse::<ArmKind>().unwrap(),
            "weak".parse::<ArmKind>().unwrap(),
            "60/30".parse::<ArmKind>().unwrap(),
        ];
        arms.sort_unstable();
        assert_eq!(arms[0], ArmKind::Weak);
    }

    #[test]
    fn a_policy_is_written_as_two_windows() {
        assert_eq!(
            "60/30".parse::<PolicySpec>().unwrap(),
            PolicySpec { account_window: 60, storage_window: 30 }
        );
        assert!("60".parse::<PolicySpec>().is_err());
        assert!("60/0".parse::<PolicySpec>().is_err());
        assert!("a/30".parse::<PolicySpec>().is_err());
    }

    /// Over any window of `n` blocks every policy occupies every slot exactly once, so the
    /// first-position cost is shared rather than assigned.
    #[test]
    fn rotation_gives_every_policy_every_slot() {
        let policies = 3;
        for slot in 0..policies {
            let occupants =
                (0..policies).map(|block| rotated_order(policies, block).nth(slot).unwrap());
            let mut seen = occupants.collect::<Vec<_>>();
            seen.sort_unstable();
            assert_eq!(seen, vec![0, 1, 2], "slot {slot} is not shared evenly");
        }
    }

    #[test]
    fn rotation_of_nothing_yields_nothing() {
        assert_eq!(rotated_order(0, 7).count(), 0);
    }
}
