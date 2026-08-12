//! ExEx-side adapter over the database-free validator core.
//!
//! The whole transition lives in `partial-stateless-validator`, which cannot reach a state
//! database. What is left here is the one thing a full node can do and a standalone validator
//! cannot: walk its own trie and offer a second opinion on the post-state root. That cross-check
//! is why this module exists, and keeping it on this side of the crate boundary is what lets the
//! core's dependency graph carry the database-free claim.

use alloy_primitives::B256;
use eyre::{eyre, Result};
use partial_stateless::{
    network_cache::NetworkStateCache, PartialStatelessSidecar, PartialTrieNodeCache,
};
use partial_stateless_validator::{
    verify_and_apply_sidecar_with_oracle, PostStateRootOracle, TimedValidation,
    TrieCacheDisposition, ValidatorRules,
};
use reth_consensus::FullConsensus;
use reth_ethereum::EthPrimitives;
use reth_evm::ConfigureEvm;
use reth_primitives_traits::{BlockTy, RecoveredBlock};
use reth_provider::StateProvider;
use reth_trie_common::HashedPostState;

pub(crate) use partial_stateless_validator::SidecarReexecLimits;

/// Cross-checks the transition against a full node's own state-root walk.
///
/// Always answers, so the caller always pays the walk; there is no mode where this is attached and
/// silently skipped. A standalone validator has no counterpart and passes `NoRootOracle` instead.
struct ProviderRootOracle<'a>(&'a dyn StateProvider);

impl PostStateRootOracle for ProviderRootOracle<'_> {
    fn post_state_root(&self, post_state: HashedPostState) -> Result<Option<B256>> {
        let (root, _) = self
            .0
            .state_root_with_updates(post_state)
            .map_err(|err| eyre!("provider-assisted state root failed: {err}"))?;
        Ok(Some(root))
    }
}

/// Verifies and applies a sidecar, cross-checking the post state against `full_provider`.
///
/// Identical to the core path in everything but that cross-check, which runs after the local root
/// has been checked against the header and before either cache is committed.
#[expect(clippy::too_many_arguments)]
pub(crate) fn verify_and_apply_provider_assisted_sidecar<Evm, Consensus>(
    rules: ValidatorRules<'_, Evm, Consensus>,
    full_provider: &dyn StateProvider,
    block: &RecoveredBlock<BlockTy<EthPrimitives>>,
    prev_cache: &mut NetworkStateCache,
    sidecar: &PartialStatelessSidecar,
    expected_cache_policy_id: B256,
    limits: &SidecarReexecLimits,
    trie_cache: &mut PartialTrieNodeCache,
    trie_cache_disposition: TrieCacheDisposition,
) -> Result<TimedValidation>
where
    Evm: ConfigureEvm<Primitives = EthPrimitives>,
    Consensus: FullConsensus<EthPrimitives> + ?Sized,
{
    verify_and_apply_sidecar_with_oracle(
        rules,
        block,
        prev_cache,
        sidecar,
        expected_cache_policy_id,
        limits,
        trie_cache,
        trie_cache_disposition,
        &ProviderRootOracle(full_provider),
        true,
    )
}
