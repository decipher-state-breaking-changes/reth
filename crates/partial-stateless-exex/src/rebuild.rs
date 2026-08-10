//! Provider-backed canonical rebuild of the coordinated cache pair.
//!
//! This is a primitive, not a scenario: given a canonical block hash it produces an exact
//! coordinated pair at that block, from nothing, without consulting whatever the caches held
//! before. Nothing here is an undo, and that is precisely what lets one function serve a cold
//! start, a reorg, and a revert alike — for a reorg, replaying forward from the cached generation
//! would be *incorrect* rather than merely slow, because that generation sits on the abandoned
//! branch.
//!
//! It requires canonical state, so it does not replace the snapshot bootstrap path. A full node
//! cold-starts by replay and needs no snapshot file; a stateless verifier cannot replay at all and
//! needs the snapshot. Neither subsumes the other. The replay also assumes the last
//! `max_window + 1` heights are still readable through `history_by_block_hash`, which holds for a
//! full node but not under aggressive pruning.
//!
//! **This must never become reachable from the measured verification path.** The validator numbers
//! only mean anything because `verify_and_apply_trustless_sidecar_for_benchmark` validates from
//! serialized sidecar bytes against the cache, trie cache, and witness alone. Everything here
//! belongs to cache *maintenance* — warming and recovery, driven by the side that holds the
//! database — and runs between measured samples, never inside one.

use crate::CacheConfig;
use alloy_primitives::B256;
use partial_stateless::{
    accessed_state::BlockAccessedState,
    network_cache::NetworkStateCache,
    readiness::{CacheObservation, CacheReadinessTracker, ReadyParent, TrustedCheckpoint},
    rebuild_trie_cache, PartialTrieNodeCache,
};
use reth_ethereum::EthPrimitives;
use reth_evm::{
    execute::{BlockExecutionOutput, Executor},
    ConfigureEvm,
};
use reth_execution_access::ExecutedBlockAccess;
use reth_primitives_traits::{AlloyBlockHeader, BlockTy, NodePrimitives, RecoveredBlock};
use reth_provider::{BlockReader, StateProvider, StateProviderFactory, TransactionVariant};
use reth_revm::database::StateProviderDatabase;
use reth_trie_common::TrieInput;
use revm::database::State;
use std::time::Instant;
use tracing::{info, warn};

/// Rebuilds an exact coordinated pair at the canonical block `tip_hash`.
///
/// The bound is `max_window + 1` heights ending at the tip, and it is the exact bound rather than
/// a shortcut. `LastNBlocksPolicy` retains exactly `last_accessed >= current_block - window_size`
/// and applies that eviction unconditionally on every block, so the cache at `H` *is by
/// definition* the set of keys touched in `[H - window, H]` — it does not accumulate. Replaying
/// that range reproduces a continuously running node's map entry for entry, including the
/// boundary: a key touched at exactly `H - window` has `last_accessed == cutoff` and is retained
/// on both sides, while a key touched at `H - window - 1` was already evicted on one side and
/// never inserted on the other.
///
/// The flat cache is replayed per block but the trie is rebuilt in one authenticated shot at the
/// tip. That trades away per-block root validation, which would localize a divergence to the block
/// that caused it; in exchange the trie's correctness stops depending on the replay at all, and
/// every retained flat value is compared against its canonical leaf at the tip, which is a
/// stricter value check than a root comparison. What a proof cannot attest to is `last_accessed`
/// metadata — that comes from the per-block replay, and it is covered instead by cache-root
/// equality against a continuously running pair, since `last_accessed_block` is hashed into every
/// leaf preimage of the cache root.
pub fn rebuild_coordinated_pair<P, Evm>(
    provider: &P,
    evm_config: &Evm,
    tip_hash: B256,
    config: &CacheConfig,
) -> eyre::Result<RebuiltPair>
where
    P: StateProviderFactory + BlockReader<Block = BlockTy<EthPrimitives>>,
    Evm: ConfigureEvm<Primitives = EthPrimitives>,
{
    let started = Instant::now();
    let chain = canonical_window(provider, tip_hash, config.max_window())?;
    let tip = chain.last().expect("canonical_window returns at least the tip");
    let tip_number = tip.number();
    let tip_state_root = tip.state_root();
    let first_number = chain[0].number();

    let replay_start = Instant::now();
    let mut cache = config.new_cache();
    for block in &chain {
        // Bound to the replayed block's own parent hash: resolving by number would admit an
        // abandoned branch mid-reorg, which is the exact failure this rebuild exists to avoid.
        let state_provider = provider.history_by_block_hash(block.parent_hash())?;
        let simulation = simulate_block(evm_config, state_provider.as_ref(), block)?;
        cache.on_block_executed(block.number(), &simulation.accessed);
    }
    // Every undo record below the tip describes a generation this pair will never return to: a
    // reorg rebuilds rather than unwinds. Keeping 61 of them would just be replay scratch space
    // held for the life of the cache.
    cache.prune_undo_below(tip_number);
    let replay_us = replay_start.elapsed().as_micros() as u64;

    let proof_start = Instant::now();
    let tip_state = provider.history_by_block_hash(tip_hash)?;
    let mut proof_targets = 0usize;
    let rebuilt = rebuild_trie_cache(&cache, tip_state_root, |targets| {
        proof_targets = targets.chunking_length();
        tip_state.multiproof(TrieInput::default(), targets).map_err(|err| err.to_string())
    })
    .map_err(|err| eyre::eyre!("canonical trie rebuild at block {tip_number} failed: {err}"))?;
    let proof_us = proof_start.elapsed().as_micros() as u64;

    let checkpoint = TrustedCheckpoint {
        block_number: tip_number,
        block_hash: tip_hash,
        state_root: tip_state_root,
        cache_root: cache.cache_root(),
        cache_policy_id: config.cache_policy_id(),
    };

    let report = RebuiltPair {
        cache,
        trie_cache: rebuilt.trie_cache,
        checkpoint,
        replayed_from: first_number,
        replayed_blocks: chain.len() as u64,
        proof_rounds: rebuilt.proof_rounds,
        proof_targets,
        replay_us,
        proof_us,
        total_us: started.elapsed().as_micros() as u64,
    };
    info!(
        target: "partial_stateless",
        block = tip_number,
        block_hash = ?tip_hash,
        replayed_from = report.replayed_from,
        replayed_blocks = report.replayed_blocks,
        proof_rounds = report.proof_rounds,
        proof_targets = report.proof_targets,
        replay_ms = report.replay_us / 1_000,
        proof_ms = report.proof_us / 1_000,
        total_ms = report.total_us / 1_000,
        "Rebuilt coordinated cache pair from canonical state"
    );
    Ok(report)
}

/// A coordinated pair rebuilt at one canonical block, not yet installed anywhere.
pub struct RebuiltPair {
    /// Flat values replayed over the policy window.
    pub cache: NetworkStateCache,
    /// Trie paths authenticated against the tip's canonical state root.
    pub trie_cache: PartialTrieNodeCache,
    /// Everything the anchor this pair will publish claims.
    pub checkpoint: TrustedCheckpoint,
    /// Lowest height replayed.
    pub replayed_from: u64,
    /// Contiguous heights replayed, ending at the tip.
    pub replayed_blocks: u64,
    /// Provider round trips the one-shot multiproof took.
    pub proof_rounds: usize,
    /// Leaf targets in the final multiproof request.
    pub proof_targets: usize,
    /// Time spent re-executing the window.
    pub replay_us: u64,
    /// Time spent on the one-shot multiproof and trie authentication.
    pub proof_us: u64,
    /// Wall time for the whole rebuild.
    pub total_us: u64,
}

/// Installs a rebuilt pair over the live caches and promotes the tracker to `Ready`.
///
/// The promotion runs before either cache is moved, so a rejected rebuild leaves the caller's
/// caches exactly as they were. The tracker is left `Cold` in that case, which is the fail-closed
/// outcome: nothing can publish, and warming from scratch is always sound.
///
/// One narrowing to record rather than hide: promoting a self-derived checkpoint makes
/// `restore_from_checkpoint`'s cache-root comparison tautological, because the checkpoint was
/// built from this very cache. The real authentication is the trie root against the canonical
/// header's state root, which is not tautological. The single promotion path is reused anyway —
/// more ways to reach `Ready` is the worse failure.
pub fn install_rebuilt_pair(
    rebuilt: RebuiltPair,
    cache: &mut NetworkStateCache,
    trie_cache: &mut PartialTrieNodeCache,
    readiness: &mut CacheReadinessTracker,
) -> eyre::Result<ReadyParent> {
    let observation = CacheObservation::capture(&rebuilt.cache, &rebuilt.trie_cache);
    let ready = readiness
        .restore_from_checkpoint(&rebuilt.checkpoint, &observation)
        .cloned()
        .map_err(|err| {
            eyre::eyre!(
                "rebuilt pair at block {} was rejected by the readiness checkpoint: {err:?}",
                rebuilt.checkpoint.block_number
            )
        })?;
    *cache = rebuilt.cache;
    *trie_cache = rebuilt.trie_cache;
    Ok(ready)
}

/// Re-executes one canonical block against its parent state and captures everything it touched.
///
/// This is the historical simulation the builder performs per block, shared so that the canonical
/// rebuild replays exactly what the live path applies. An execution diff would not do: the cache
/// is a function of *accessed* state, which includes read-only accounts, code reads, and reads
/// made by calls that later reverted.
pub fn simulate_block<Evm>(
    evm_config: &Evm,
    state_provider: &dyn StateProvider,
    block: &RecoveredBlock<BlockTy<EthPrimitives>>,
) -> eyre::Result<HistoricalSimulation>
where
    Evm: ConfigureEvm<Primitives = EthPrimitives>,
{
    let started = Instant::now();
    let state_provider_db = StateProviderDatabase::new(state_provider);
    let mut db = State::builder().with_bundle_update().with_database(state_provider_db).build();
    let block_executor = evm_config.executor(&mut db);

    let mut accessed = BlockAccessedState::default();
    let mut lowest_block_number = None;
    let output = block_executor
        .execute_with_state_closure(block, |statedb: &State<_>| {
            // One extraction produces both the access set and the BLOCKHASH range, and it is the
            // same extraction the node's own execution uses. Reading the two separately is what
            // would let a captured artifact and this simulation drift apart.
            let access = ExecutedBlockAccess::from_state(statedb);
            lowest_block_number = access.lowest_block_hash_number;
            accessed = access.into();
        })
        .map_err(|err| eyre::eyre!("simulation failed for block {}: {err}", block.number()))?;

    Ok(HistoricalSimulation {
        accessed,
        lowest_block_number,
        output,
        elapsed_us: started.elapsed().as_micros() as u64,
    })
}

/// What one block's historical re-execution produced.
#[derive(Debug)]
pub struct HistoricalSimulation {
    /// Every account, storage slot, and bytecode the block read or wrote.
    pub accessed: BlockAccessedState,
    /// Lowest ancestor height the block asked for via `BLOCKHASH`, if any.
    pub lowest_block_number: Option<u64>,
    /// Receipts, requests, gas, and the resulting bundle state.
    pub output: BlockExecutionOutput<<EthPrimitives as NodePrimitives>::Receipt>,
    /// Wall time of the re-execution.
    pub elapsed_us: u64,
}

/// Collects `[tip - window, tip]` in ascending order, fixing the branch by parent hash.
///
/// Every step resolves the parent *hash* rather than a height. Resolving by number would admit an
/// abandoned branch mid-reorg, which is exactly the state this function is most often called from.
fn canonical_window<P>(
    provider: &P,
    tip_hash: B256,
    window: u64,
) -> eyre::Result<Vec<RecoveredBlock<BlockTy<EthPrimitives>>>>
where
    P: BlockReader<Block = BlockTy<EthPrimitives>>,
{
    let tip = load_block(provider, tip_hash)?;
    let tip_number = tip.number();
    // Genesis is not executable, so the replay can never start below block 1. On a chain shorter
    // than the window this yields fewer than `window + 1` heights, which is still the complete
    // history the policy could possibly retain.
    let first_number = tip_number.saturating_sub(window).max(1);
    if tip_number < first_number {
        eyre::bail!("cannot rebuild at block {tip_number}: it is below the first executable block");
    }
    if tip_number.saturating_sub(first_number) < window {
        warn!(
            target: "partial_stateless",
            tip = tip_number,
            first = first_number,
            window,
            "Canonical rebuild covers fewer heights than the policy window; the chain is shorter \
             than the window"
        );
    }

    let mut chain = Vec::with_capacity((tip_number - first_number + 1) as usize);
    let mut next_hash = tip.parent_hash();
    let mut next_number = tip_number;
    chain.push(tip);
    while next_number > first_number {
        let block = load_block(provider, next_hash)?;
        if block.number() + 1 != next_number {
            eyre::bail!(
                "canonical window is not contiguous: block {:?} reports height {} but is the \
                 parent of height {next_number}",
                next_hash,
                block.number()
            );
        }
        next_number = block.number();
        next_hash = block.parent_hash();
        chain.push(block);
    }
    chain.reverse();
    Ok(chain)
}

fn load_block<P>(provider: &P, hash: B256) -> eyre::Result<RecoveredBlock<BlockTy<EthPrimitives>>>
where
    P: BlockReader<Block = BlockTy<EthPrimitives>>,
{
    provider
        .recovered_block(hash.into(), TransactionVariant::WithHash)?
        .ok_or_else(|| eyre::eyre!("canonical block {hash:?} is not available for replay"))
}
