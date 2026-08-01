//! Partial Statelessness ExEx — reth Execution Extension that maintains a
//! network-level state cache and reports witness requirements per block.
//!
//! Run with:
//!   cargo run -p partial-stateless-exex -- node --chain mainnet --datadir /path/to/data
//!
//! This ExEx subscribes to canonical chain commits and:
//! 1. Extracts `BlockAccessedState` via EVM simulation of each block
//! 2. Updates the `NetworkStateCache` with the accessed state
//! 3. Computes and logs cache miss ratio (= witness requirement)
//! 4. Builds and measures the canonical parent-state transition witness

#[global_allocator]
static ALLOC: reth_cli_util::allocator::Allocator = reth_cli_util::allocator::new_allocator();

// Required for jemalloc to override the allocator on supported Unix platforms.
#[cfg(unix)]
use reth_cli_util::allocator::tikv_jemalloc_sys as _;

mod benchmark;
mod sidecar_create;
mod sidecar_io;
mod sidecar_reexec;
mod sidecar_verify;

use alloy_primitives::B256;
use alloy_rlp::Encodable;
use futures::TryStreamExt;
use partial_stateless::{
    network_cache::NetworkStateCache,
    persistence::{load_from_file, save_to_file},
    policy::LastNBlocksPolicy,
    readiness::{BlockContext, CacheObservation, CacheReadinessTracker},
    sidecar::last_n_blocks_cache_policy_id,
    PartialTrieNodeCache,
};
use reth_ethereum::{
    chainspec::EthChainSpec,
    exex::{ExExContext, ExExEvent, ExExNotification},
    node::{
        api::{FullNodeComponents, NodeTypes},
        builder::NodeHandleFor,
        EthereumNode,
    },
    provider::StateProviderFactory,
    EthPrimitives,
};
use reth_primitives_traits::{AlloyBlockHeader, BlockTy, RecoveredBlock};
use reth_provider::{BlockIdReader, CanonicalOverlayFactory, HeaderProvider, ProviderResult};
use reth_trie_common::MultiProofTargetsV2;
use reth_trie_parallel::proof_task::parallel_multiproof_v2_with_stats;
use sidecar_create::{
    create_sidecar_for_block, BuilderBlockReport, BuilderOptions, ParallelInitialProofFn,
    ParallelInitialProofOutput,
};
use sidecar_reexec::SidecarReexecLimits;
use sidecar_verify::verify_live_sidecar;
use std::{path::PathBuf, time::Duration};
use tracing::{error, info, warn};

/// Configuration for the partial statelessness cache.
#[derive(Debug, Clone, Copy)]
struct CacheConfig {
    /// Window size for account eviction policy (in blocks).
    account_window: u64,
    /// Window size for storage/code eviction policy (in blocks).
    storage_window: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self { account_window: 60, storage_window: 30 }
    }
}

impl CacheConfig {
    fn new_cache(&self) -> NetworkStateCache {
        self.new_cache_at(0)
    }

    fn new_cache_at(&self, current_block: u64) -> NetworkStateCache {
        NetworkStateCache::restore(
            Default::default(),
            Default::default(),
            Default::default(),
            current_block,
            Box::new(LastNBlocksPolicy::new(self.account_window)),
            Box::new(LastNBlocksPolicy::new(self.storage_window)),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SidecarRole {
    Builder,
    BuilderVerifier,
    Verifier,
}

impl SidecarRole {
    const fn runs_preflight(self) -> bool {
        matches!(self, Self::BuilderVerifier)
    }

    fn from_env() -> Self {
        let Some(value) = std::env::var("PS_SIDECAR_ROLE").ok() else {
            return Self::Builder;
        };
        let normalized = value.to_ascii_lowercase().replace('_', "-");

        match normalized.as_str() {
            "builder" | "build" => Self::Builder,
            "builder-verifier" | "both" | "test" => Self::BuilderVerifier,
            "verifier" | "verify" | "client" => Self::Verifier,
            other => {
                warn!(
                    target: "partial_stateless",
                    value = other,
                    "Unknown PS_SIDECAR_ROLE; falling back to builder"
                );
                Self::Builder
            }
        }
    }
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}

fn env_millis(name: &str, default: u64) -> Duration {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .map(Duration::from_millis)
        .unwrap_or_else(|| Duration::from_millis(default))
}

fn configured_sidecar_dir() -> PathBuf {
    std::env::var_os("PS_SIDECAR_DIR").map(PathBuf::from).unwrap_or_else(|| {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")).join("sidecar")
    })
}

/// The ExEx function that processes chain notifications and maintains the cache.
async fn partial_stateless_exex<
    Node: FullNodeComponents<Types: NodeTypes<Primitives = EthPrimitives>>,
>(
    mut ctx: ExExContext<Node>,
    config: CacheConfig,
) -> eyre::Result<()>
where
    Node::Provider: CanonicalOverlayFactory,
{
    // Resolve the cache file path: datadir/partial_stateless_cache.bin
    let cache_dir = ctx.config.datadir.clone().resolve_datadir(ctx.config.chain.chain());
    let cache_path = cache_dir.as_ref().join("partial_stateless_cache.bin");

    // Cache coherence guarantee: the LastNBlocksPolicy is fully deterministic —
    // given the same canonical chain and window parameters, all peers converge
    // to the same cache state. Loading from disk and replaying missed blocks
    // during sync produces an identical result to continuous operation.
    let mut cache = if cache_path.exists() {
        match load_from_file(
            &cache_path,
            Box::new(LastNBlocksPolicy::new(config.account_window)),
            Box::new(LastNBlocksPolicy::new(config.storage_window)),
        ) {
            Ok(loaded_cache) => {
                let cache_block = loaded_cache.current_block();
                let head_block = ctx.head.number;

                // Validation: Gap Tolerance based on config.account_window
                let max_allowed_gap = config.account_window;
                if cache_block <= head_block && head_block - cache_block <= max_allowed_gap {
                    info!(
                        target: "partial_stateless",
                        cache_block = cache_block,
                        head_block = head_block,
                        gap = head_block - cache_block,
                        "Warm state cache loaded successfully from disk. Continuing sync..."
                    );
                    loaded_cache
                } else {
                    warn!(
                            target: "partial_stateless",
                            cache_block = cache_block,
                            head_block = head_block,
                            max_allowed_gap = max_allowed_gap,
                            "Cache file block state is too far from head block or in the future. Starting with cold cache."
                    );
                    config.new_cache()
                }
            }
            Err(e) => {
                warn!(
                    target: "partial_stateless",
                    error = %e,
                    "Failed to load cache file from disk. Starting with cold cache."
                );
                config.new_cache()
            }
        }
    } else {
        info!(
            target: "partial_stateless",
            "No existing cache file found at {}. Starting with cold cache.",
            cache_path.display()
        );
        config.new_cache()
    };

    let sidecar_role = SidecarRole::from_env();
    let sidecar_dir = configured_sidecar_dir();
    let verifier_wait = env_millis("PS_SIDECAR_VERIFIER_WAIT_MS", 2_000);
    let validation_bench = env_flag("PS_VALIDATION_BENCH");
    let parallel_initial_proof_enabled = env_flag("PS_PARALLEL_INITIAL_PROOF");
    if validation_bench && sidecar_role != SidecarRole::BuilderVerifier {
        return Err(eyre::eyre!("PS_VALIDATION_BENCH requires PS_SIDECAR_ROLE=builder-verifier"));
    }
    let bench_output = validation_bench.then(|| {
        std::env::var_os("PS_BENCH_OUTPUT")
            .map(PathBuf::from)
            .unwrap_or_else(|| sidecar_dir.join("validation_bench.jsonl"))
    });
    let builder_bench_output = std::env::var_os("PS_BUILDER_BENCH_OUTPUT").map(PathBuf::from);
    let force_previous_cache_snapshot = env_flag("PS_FORCE_PREVIOUS_CACHE_SNAPSHOT");

    info!(
        target: "partial_stateless",
        account_window = config.account_window,
        storage_window = config.storage_window,
        cache_path = %cache_path.display(),
        sidecar_role = ?sidecar_role,
        sidecar_dir = %sidecar_dir.display(),
        verifier_wait_ms = verifier_wait.as_millis(),
        "Partial Stateless ExEx started — monitoring cache state per block"
    );

    // Optional reproducible-dataset capture: when `PS_CAPTURE_DIR` is set, dump the
    // per-block `BlockAccessedState` (the only cache input) so the cache-window
    // benchmark can replay a fixed range offline, with no node or EVM. This reuses
    // the exact execution path the live system uses, so the dataset is faithful.
    let capture_dir = if validation_bench {
        None
    } else {
        std::env::var("PS_CAPTURE_DIR").ok().map(std::path::PathBuf::from)
    };
    if let Some(dir) = &capture_dir {
        info!(
            target: "partial_stateless",
            dir = %dir.display(),
            "Accessed-state fixture capture ENABLED (PS_CAPTURE_DIR) — run until ~300 blocks captured"
        );
    }

    // Optional benchmark-only comparison against the FULL witness (every accessed
    // key, ignoring the cache). It computes a second, larger multiproof per block,
    // so it is off by default and gated behind `PS_WITNESS_BASELINE`. Core sidecar
    // generation never depends on it.
    let compute_baseline = !validation_bench && env_flag("PS_WITNESS_BASELINE");
    if compute_baseline {
        info!(
            target: "partial_stateless",
            "Full-witness baseline comparison ENABLED (PS_WITNESS_BASELINE) — extra multiproof per block"
        );
    }

    // Optional sparse-trie shape benchmark and full cache-invariant scan. This walks every
    // retained path, so keep it off during normal operation and enable it for bounded runs.
    let trie_cache_diagnostics = !validation_bench && env_flag("PS_TRIE_CACHE_DIAGNOSTICS");
    if trie_cache_diagnostics {
        info!(
            target: "partial_stateless",
            "Trie-shape cache diagnostics ENABLED (PS_TRIE_CACHE_DIAGNOSTICS) — per-block timings, depth-0..5 prefix coverage, and full retained-path validation"
        );
    }

    // Optional process-wide resource metrics (CPU time + page faults) captured
    // around the canonical transition witness, to attribute its cost between
    // compute and disk I/O. Off by default and gated behind `PS_RESOURCE_METRICS`; when
    // disabled the getrusage syscalls are skipped and the metric fields stay None.
    let resource_metrics = !validation_bench && env_flag("PS_RESOURCE_METRICS");
    if resource_metrics {
        info!(
            target: "partial_stateless",
            "Process-wide resource metrics ENABLED (PS_RESOURCE_METRICS) — includes parallel proof-worker CPU and page faults"
        );
        #[cfg(not(target_os = "linux"))]
        warn!(
            target: "partial_stateless",
            "Process CPU/page-fault metrics require getrusage; this platform will log default zeros for cpu_time_ms, major_page_faults, and minor_page_faults"
        );
        if compute_baseline {
            warn!(
                target: "partial_stateless",
                "PS_WITNESS_BASELINE runs before the transition witness; resource/page-fault metrics may be lower because the full baseline can warm the OS page cache"
            );
        }
    }
    if parallel_initial_proof_enabled {
        info!(
            target: "partial_stateless",
            "Parallel initial V2 multiproof ENABLED (PS_PARALLEL_INITIAL_PROOF); low-width target sets remain serial"
        );
    }

    // Optional provider-assisted validator preflight. This re-executes each
    // generated sidecar with a cache+witness-backed provider and checks the
    // cache-state transition. It is useful for PoC acceptance checks, but it
    // adds another block execution on the sidecar generation path.
    let run_sidecar_preflight = sidecar_role.runs_preflight();
    if run_sidecar_preflight {
        info!(
            target: "partial_stateless",
            sidecar_role = ?sidecar_role,
            "Provider-assisted sidecar preflight ENABLED — extra re-execution per sidecar"
        );
    }
    if let Some(path) = &bench_output {
        info!(
            target: "partial_stateless",
            path = %path.display(),
            "Paired Partial/Weak/Vanilla validation benchmark ENABLED (PS_VALIDATION_BENCH)"
        );
    }
    if let Some(path) = &builder_bench_output {
        info!(
            target: "partial_stateless",
            path = %path.display(),
            "Structured builder benchmark output ENABLED (PS_BUILDER_BENCH_OUTPUT)"
        );
    }
    if force_previous_cache_snapshot {
        warn!(
            target: "partial_stateless",
            "Forced previous-cache snapshot ENABLED for B2 benchmark control"
        );
    }
    if sidecar_role == SidecarRole::Verifier {
        info!(
            target: "partial_stateless",
            sidecar_dir = %sidecar_dir.display(),
            verifier_wait_ms = verifier_wait.as_millis(),
            "Live sidecar verifier ENABLED — consuming sidecar files and advancing cache only after successful verification"
        );
    }
    let reexec_limits = SidecarReexecLimits::default();

    // This persistent-in-memory sparse trie mirrors value-cache account and storage paths. It has
    // no persisted snapshot or branch-aware undo log yet, so it must reset together with values.
    let mut trie_cache = PartialTrieNodeCache::new();
    if cache.current_block() != 0 {
        warn!(
            target: "partial_stateless",
            cache_block = cache.current_block(),
            "Persisted values have no matching sparse-trie snapshot; cold-resetting both caches"
        );
        cache.reset();
    }

    // Tracks whether the two caches together still describe the parent of the block being
    // processed. Both start cold here, and the tracker starts cold with them. The window is the
    // larger of the two eviction windows: the cache only holds everything its policy identifier
    // advertises once the longer of the two has been replayed.
    let mut readiness = CacheReadinessTracker::new(
        config.account_window.max(config.storage_window),
        last_n_blocks_cache_policy_id(config.account_window, config.storage_window),
    );

    while let Some(notification) = ctx.notifications.try_next().await? {
        match &notification {
            ExExNotification::ChainCommitted { new } => {
                let range = new.range();
                let tip_block = *range.end();

                // Process blocks in chronological order
                for (block_number, block) in new.blocks() {
                    let block_ctx = block_context(block);
                    if !admit_block(&mut readiness, &mut cache, &mut trie_cache, &block_ctx) {
                        continue;
                    }

                    // The parent is addressed by hash rather than by number: a height can name a
                    // block on an abandoned branch while a reorg is in flight, and building the
                    // cache on that state would silently mix two chains.
                    let state_provider =
                        match ctx.provider().history_by_block_hash(block.parent_hash) {
                            Ok(provider) => provider,
                            Err(e) => {
                                // The block is not applied, so every later block would build on
                                // state these caches never saw. Recording that keeps the gap from
                                // being papered over on the next block.
                                let reason = readiness.abandon_block(*block_number);
                                warn!(
                                    target: "partial_stateless",
                                    block = *block_number,
                                    error = %e,
                                    ?reason,
                                    "Failed to get state provider for parent block. Skipping block."
                                );
                                continue;
                            }
                        };

                    if sidecar_role == SidecarRole::Verifier {
                        if let Err(e) = verify_live_sidecar(
                            ctx.evm_config(),
                            state_provider.as_ref(),
                            block,
                            &mut cache,
                            &mut trie_cache,
                            &config,
                            &sidecar_dir,
                            &reexec_limits,
                            verifier_wait,
                        ) {
                            return Err(eyre::eyre!(
                                "live sidecar verification failed at block {}: {e}",
                                block_number
                            ));
                        }
                        observe_readiness(&mut readiness, &block_ctx, &cache, &trie_cache);
                        continue;
                    }

                    let parent_state_root_by_hash = |parent_hash: B256| {
                        ctx.provider()
                            .sealed_header_by_hash(parent_hash)
                            .map_err(|err| eyre::eyre!("failed to fetch parent header: {err}"))?
                            .map(|header| header.state_root)
                            .ok_or_else(|| {
                                eyre::eyre!("parent header not found for hash {:?}", parent_hash)
                            })
                    };
                    let ancestor_headers_for_range =
                        |lowest_block_number: Option<u64>, block_number: u64| {
                            let Some(smallest) = lowest_block_number else {
                                return Ok(Vec::new());
                            };
                            let headers = ctx
                                .provider()
                                .headers_range(smallest..block_number)
                                .map_err(|err| {
                                    eyre::eyre!("failed to fetch ancestor headers for range: {err}")
                                })?;
                            Ok(headers
                                .into_iter()
                                .map(|header| {
                                    let mut buf = Vec::new();
                                    let _ = header.encode(&mut buf);
                                    buf.into()
                                })
                                .collect())
                        };
                    let parallel_initial_proof =
                        |targets: MultiProofTargetsV2| -> ProviderResult<ParallelInitialProofOutput> {
                            let parallel_factory =
                                ctx.provider().overlay_factory_at_block(block.parent_hash);
                            parallel_multiproof_v2_with_stats(
                                ctx.task_executor(),
                                parallel_factory,
                                targets,
                                true,
                            )
                            .map(|(proof, stats)| ParallelInitialProofOutput {
                                proof,
                                storage_workers: stats.storage_workers,
                                account_workers: stats.account_workers,
                            })
                            .map_err(Into::into)
                        };

                    match create_sidecar_for_block(
                        ctx.evm_config(),
                        state_provider.as_ref(),
                        block,
                        &mut cache,
                        &mut trie_cache,
                        &config,
                        BuilderOptions {
                            capture_dir: capture_dir.as_deref(),
                            sidecar_dir: &sidecar_dir,
                            compute_baseline,
                            resource_metrics,
                            trie_cache_diagnostics,
                            run_sidecar_preflight,
                            validation_bench_output: bench_output.as_deref(),
                            builder_bench_output: builder_bench_output.as_deref(),
                            force_previous_cache_snapshot,
                            reexec_limits: &reexec_limits,
                            parallel_initial_proof: parallel_initial_proof_enabled
                                .then_some(&parallel_initial_proof as &ParallelInitialProofFn<'_>),
                        },
                        parent_state_root_by_hash,
                        ancestor_headers_for_range,
                    ) {
                        Ok(BuilderBlockReport {
                            cache_update: _cache_update,
                            witness: _witness,
                            sidecar_path: _sidecar_path,
                        }) => {
                            observe_readiness(&mut readiness, &block_ctx, &cache, &trie_cache);
                        }
                        Err(e) => {
                            return Err(eyre::eyre!(
                                "sidecar builder failed at block {}: {e}",
                                block_number
                            ));
                        }
                    }
                }

                // Cache persistence is unrelated to validation and can perturb later
                // Engine samples, so the bounded paired benchmark keeps it in memory only.
                if !validation_bench {
                    if let Err(e) = save_to_file(&cache, &cache_path) {
                        warn!(
                            target: "partial_stateless",
                            block = tip_block,
                            error = %e,
                            "Failed to save cache state to disk"
                        );
                    }
                }
            }
            ExExNotification::ChainReorged { old, new } => {
                let tip_block = *new.range().end();
                warn!(
                    target: "partial_stateless",
                    from_chain = ?old.range(),
                    to_chain = ?new.range(),
                    "Chain reorg detected — cold-resetting value and sparse-trie caches, then applying new chain"
                );

                // Sparse-trie snapshots currently have no branch-aware undo log. Reset both
                // caches together so value hits can never outlive their authenticated paths.
                readiness.begin_recovery();
                trie_cache = PartialTrieNodeCache::new();
                cache.reset();
                readiness.reset();

                // Apply the new canonical chain block-by-block (records value-cache undo).
                for (block_number, block) in new.blocks() {
                    let block_ctx = block_context(block);
                    if !admit_block(&mut readiness, &mut cache, &mut trie_cache, &block_ctx) {
                        continue;
                    }

                    let state_provider = match ctx
                        .provider()
                        .history_by_block_hash(block.parent_hash)
                    {
                        Ok(provider) => provider,
                        Err(e) => {
                            let reason = readiness.abandon_block(*block_number);
                            warn!(
                                target: "partial_stateless",
                                block = *block_number,
                                error = %e,
                                ?reason,
                                "Failed to get state provider for block parent on reorg. Skipping."
                            );
                            continue;
                        }
                    };

                    if sidecar_role == SidecarRole::Verifier {
                        if let Err(e) = verify_live_sidecar(
                            ctx.evm_config(),
                            state_provider.as_ref(),
                            block,
                            &mut cache,
                            &mut trie_cache,
                            &config,
                            &sidecar_dir,
                            &reexec_limits,
                            verifier_wait,
                        ) {
                            return Err(eyre::eyre!(
                                "live sidecar verification failed while applying reorg block {}: {e}",
                                block_number
                            ));
                        }
                        observe_readiness(&mut readiness, &block_ctx, &cache, &trie_cache);
                        continue;
                    }

                    let parent_state_root_by_hash = |parent_hash: B256| {
                        ctx.provider()
                            .sealed_header_by_hash(parent_hash)
                            .map_err(|err| eyre::eyre!("failed to fetch parent header: {err}"))?
                            .map(|header| header.state_root)
                            .ok_or_else(|| {
                                eyre::eyre!("parent header not found for hash {:?}", parent_hash)
                            })
                    };
                    let ancestor_headers_for_range =
                        |lowest_block_number: Option<u64>, block_number: u64| {
                            let Some(smallest) = lowest_block_number else {
                                return Ok(Vec::new());
                            };
                            let headers = ctx
                                .provider()
                                .headers_range(smallest..block_number)
                                .map_err(|err| {
                                    eyre::eyre!("failed to fetch ancestor headers for range: {err}")
                                })?;
                            Ok(headers
                                .into_iter()
                                .map(|header| {
                                    let mut buf = Vec::new();
                                    let _ = header.encode(&mut buf);
                                    buf.into()
                                })
                                .collect())
                        };
                    let parallel_initial_proof =
                        |targets: MultiProofTargetsV2| -> ProviderResult<ParallelInitialProofOutput> {
                            let parallel_factory =
                                ctx.provider().overlay_factory_at_block(block.parent_hash);
                            parallel_multiproof_v2_with_stats(
                                ctx.task_executor(),
                                parallel_factory,
                                targets,
                                true,
                            )
                            .map(|(proof, stats)| ParallelInitialProofOutput {
                                proof,
                                storage_workers: stats.storage_workers,
                                account_workers: stats.account_workers,
                            })
                            .map_err(Into::into)
                        };

                    match create_sidecar_for_block(
                        ctx.evm_config(),
                        state_provider.as_ref(),
                        block,
                        &mut cache,
                        &mut trie_cache,
                        &config,
                        BuilderOptions {
                            capture_dir: capture_dir.as_deref(),
                            sidecar_dir: &sidecar_dir,
                            compute_baseline,
                            resource_metrics,
                            trie_cache_diagnostics,
                            run_sidecar_preflight,
                            validation_bench_output: bench_output.as_deref(),
                            builder_bench_output: builder_bench_output.as_deref(),
                            force_previous_cache_snapshot,
                            reexec_limits: &reexec_limits,
                            parallel_initial_proof: parallel_initial_proof_enabled
                                .then_some(&parallel_initial_proof as &ParallelInitialProofFn<'_>),
                        },
                        parent_state_root_by_hash,
                        ancestor_headers_for_range,
                    ) {
                        Ok(BuilderBlockReport {
                            cache_update: _cache_update,
                            witness: _witness,
                            sidecar_path: _sidecar_path,
                        }) => {
                            observe_readiness(&mut readiness, &block_ctx, &cache, &trie_cache);
                        }
                        Err(e) => {
                            return Err(eyre::eyre!(
                                "sidecar builder failed while applying reorg block {}: {e}",
                                block_number
                            ));
                        }
                    }
                }

                // Production persists the rebuilt cache so a restart cannot reload the old
                // branch. Benchmark mode deliberately keeps all cache state in memory.
                if !validation_bench {
                    if let Err(e) = save_to_file(&cache, &cache_path) {
                        warn!(
                            target: "partial_stateless",
                            block = tip_block,
                            error = %e,
                            "Failed to persist cache after reorg"
                        );
                    }
                }
            }
            ExExNotification::ChainReverted { old } => {
                warn!(
                    target: "partial_stateless",
                    reverted_chain = ?old.range(),
                    "Chain reverted — cold-resetting value and sparse-trie caches"
                );

                readiness.begin_recovery();
                trie_cache = PartialTrieNodeCache::new();
                cache.reset();
                readiness.reset();

                // Production persists the cold value cache; benchmark mode has no
                // restart contract and avoids this unrelated disk write.
                if !validation_bench {
                    if let Err(e) = save_to_file(&cache, &cache_path) {
                        warn!(
                            target: "partial_stateless",
                            error = %e,
                            "Failed to persist cache after revert"
                        );
                    }
                }
            }
        }

        // Prune undo history below the finalized block: reorgs never cross
        // finality, so once a block is finalized its undo record is unreachable.
        // Keeping records down to finality means any legal reorg can be rolled
        // back precisely (this cache has no re-execution fallback — a missing
        // undo record forces a cold reset). When finality is unavailable (early
        // sync / no-finality chains) fall back to a fixed depth floor so the log
        // stays bounded. Mirrors reth's CHANGESET_CACHE_RETENTION_BLOCKS.
        const UNDO_LOG_FALLBACK_DEPTH: u64 = 64;
        let threshold = ctx
            .provider()
            .finalized_block_number()
            .ok()
            .flatten()
            .unwrap_or_else(|| cache.current_block().saturating_sub(UNDO_LOG_FALLBACK_DEPTH));
        cache.prune_undo_below(threshold);

        // Acknowledge processed height. This is a durable promise that everything up to the tip was
        // processed, and reth prunes below it — so while a block is known to be missing from the
        // caches, withholding the acknowledgement is what keeps the gap recoverable across a
        // restart. A cold reset clears the block, since a reset cache no longer claims to describe
        // any state the gap could corrupt.
        if let Some(committed_chain) = notification.committed_chain() {
            if readiness.may_acknowledge_height() {
                ctx.events.send(ExExEvent::FinishedHeight(committed_chain.tip().num_hash()))?;
            } else {
                error!(
                    target: "partial_stateless",
                    tip = committed_chain.tip().number,
                    state = readiness.state().label(),
                    "Withholding processed-height acknowledgement while the cache is blocked"
                );
            }
        }
    }

    Ok(())
}

/// Admits a block for application, cold-resetting both caches first if they no longer describe its
/// parent.
///
/// A block that is not the direct child of the last one applied means the caches describe a state
/// this chain never passed through. No undo log here is deep enough to unwind an arbitrary gap, so
/// the only recovery is to drop everything and warm again from this block onwards. Returns whether
/// the block may be applied at all.
fn admit_block(
    readiness: &mut CacheReadinessTracker,
    cache: &mut NetworkStateCache,
    trie_cache: &mut PartialTrieNodeCache,
    block: &BlockContext,
) -> bool {
    match readiness.begin_block(block) {
        Ok(()) => true,
        Err(reason) => {
            error!(
                target: "partial_stateless",
                block = block.number,
                ?reason,
                "Cache continuity broken — cold-resetting both caches and warming again from this block"
            );
            *trie_cache = PartialTrieNodeCache::new();
            cache.reset();
            readiness.reset();
            readiness.begin_block(block).is_ok()
        }
    }
}

/// Reclassifies the caches after `block` was applied, logging only genuine changes.
///
/// Readiness is advisory here: nothing yet refuses to build or verify a sidecar because the caches
/// are still warming. Gating on it would change what the benchmark measures.
fn observe_readiness(
    readiness: &mut CacheReadinessTracker,
    block: &BlockContext,
    cache: &NetworkStateCache,
    trie_cache: &PartialTrieNodeCache,
) {
    let before = readiness.state().label();
    let after = readiness.finish_block(block, &CacheObservation::capture(cache, trie_cache));
    if before != after.label() {
        info!(
            target: "partial_stateless",
            block = block.number,
            from = before,
            to = after.label(),
            replay_depth = readiness.replay_depth(),
            "Cache readiness changed"
        );
    }
}

/// Describes a canonical block for the readiness tracker.
fn block_context(block: &RecoveredBlock<BlockTy<EthPrimitives>>) -> BlockContext {
    BlockContext {
        number: block.number(),
        hash: block.hash(),
        parent_hash: block.parent_hash,
        state_root: block.state_root(),
    }
}

/// Format bytes into human-readable string.
fn format_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

/// Process-wide resource snapshot used to attribute multiproof cost.
///
/// Returns `(cpu_micros, major_faults, minor_faults)` for the whole process. `RUSAGE_SELF`
/// intentionally includes the Rayon proof workers used by parallel multiproof. Other node work
/// running during the interval is also included, so this remains diagnostic rather than protocol
/// telemetry.
///
/// Diagnostic use: comparing the CPU delta against the wall-clock elapsed
/// separates compute-bound blocks (cpu ≈ wall) from I/O/wait-bound blocks
/// (cpu ≪ wall); a nonzero major-fault delta proves the cold trie read hit
/// disk/swap rather than the page cache — the signature of the environmental
/// tail. Linux-only; returns zeros elsewhere or if the syscall fails.
#[cfg(target_os = "linux")]
fn process_rusage() -> (u64, u64, u64) {
    // SAFETY: `getrusage` only writes into the `rusage` we hand it; the struct
    // is fully zero-initialized before the call.
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut ru) != 0 {
            return (0, 0, 0);
        }
        let cpu_us = (ru.ru_utime.tv_sec as u64) * 1_000_000 +
            (ru.ru_utime.tv_usec as u64) +
            (ru.ru_stime.tv_sec as u64) * 1_000_000 +
            (ru.ru_stime.tv_usec as u64);
        (cpu_us, ru.ru_majflt as u64, ru.ru_minflt as u64)
    }
}

#[cfg(not(target_os = "linux"))]
fn process_rusage() -> (u64, u64, u64) {
    (0, 0, 0)
}

/// Resident set size of the whole process in bytes.
///
/// Sampled around the per-block trie clone to size what that clone costs in real memory, which
/// `estimated_memory_bytes` cannot show: the estimate counts logical node contents, while RSS also
/// captures allocator overhead and fragmentation. Process-wide, so concurrent node work lands in
/// the same number — a single sample is noise, and only the distribution over many blocks is
/// meaningful. Linux-only; returns 0 elsewhere or if the read fails.
#[cfg(target_os = "linux")]
fn process_rss_bytes() -> u64 {
    // Field 2 of /proc/self/statm is the resident set, counted in pages.
    let Ok(statm) = std::fs::read_to_string("/proc/self/statm") else { return 0 };
    let Some(resident_pages) = statm.split_whitespace().nth(1) else { return 0 };
    let Ok(resident_pages) = resident_pages.parse::<u64>() else { return 0 };
    // SAFETY: `sysconf` reads a process-global constant and writes nothing.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return 0
    }
    resident_pages.saturating_mul(page_size as u64)
}

#[cfg(not(target_os = "linux"))]
fn process_rss_bytes() -> u64 {
    0
}

#[cfg(test)]
mod tests {
    use super::{
        admit_block, block_context, last_n_blocks_cache_policy_id, observe_readiness, BlockContext,
        CacheConfig, CacheReadinessTracker, NetworkStateCache, PartialTrieNodeCache, SidecarRole,
    };
    use alloy_primitives::{Address, B256, U256};
    use partial_stateless::{
        accessed_state::BlockAccessedState, policy::AccountData, readiness::CacheReadiness,
    };
    use reth_primitives_traits::Block as _;

    const TOUCHED: Address = Address::repeat_byte(0x11);

    fn tracker() -> CacheReadinessTracker {
        let config = CacheConfig::default();
        CacheReadinessTracker::new(
            config.account_window.max(config.storage_window),
            last_n_blocks_cache_policy_id(config.account_window, config.storage_window),
        )
    }

    /// Synthesizes a chain whose hashes and state roots derive from block numbers, so a test can
    /// name any block without a fixture.
    fn ctx(number: u64) -> BlockContext {
        BlockContext {
            number,
            hash: numbered(number, 0xbb),
            parent_hash: numbered(number - 1, 0xbb),
            state_root: numbered(number, 0x55),
        }
    }

    fn numbered(number: u64, tag: u8) -> B256 {
        let mut value = B256::ZERO;
        value[0..8].copy_from_slice(&number.to_be_bytes());
        value[31] = tag;
        value
    }

    /// Stands in for a block application: advances the cache height and leaves one entry behind.
    fn apply(cache: &mut NetworkStateCache, number: u64) {
        let mut accessed = BlockAccessedState::default();
        accessed.accounts.insert(
            TOUCHED,
            AccountData { nonce: number, balance: U256::from(number), code_hash: None },
        );
        cache.on_block_executed(number, &accessed);
    }

    #[test]
    fn only_builder_verifier_runs_builder_side_preflight() {
        assert!(!SidecarRole::Builder.runs_preflight());
        assert!(SidecarRole::BuilderVerifier.runs_preflight());
        assert!(!SidecarRole::Verifier.runs_preflight());
    }

    #[test]
    fn empty_cache_can_be_bound_to_a_parent_height() {
        let cache = CacheConfig::default().new_cache_at(99);
        assert_eq!(cache.current_block(), 99);
        assert!(cache.accounts().is_empty());
        assert!(cache.storage().is_empty());
        assert!(cache.codes().is_empty());
    }

    #[test]
    fn contiguous_blocks_are_admitted_without_disturbing_the_caches() {
        let mut readiness = tracker();
        let mut cache = CacheConfig::default().new_cache();
        let mut trie_cache = PartialTrieNodeCache::new();

        for number in 100..103 {
            let block = ctx(number);
            assert!(admit_block(&mut readiness, &mut cache, &mut trie_cache, &block));
            apply(&mut cache, number);
            observe_readiness(&mut readiness, &block, &cache, &trie_cache);
        }

        assert_eq!(readiness.replay_depth(), 3);
        assert_eq!(cache.current_block(), 102);
        assert!(cache.contains_account(&TOUCHED), "contiguous blocks never trigger a reset");
    }

    #[test]
    fn a_gap_cold_resets_the_caches_and_restarts_the_replay_count() {
        let mut readiness = tracker();
        let mut cache = CacheConfig::default().new_cache();
        let mut trie_cache = PartialTrieNodeCache::new();

        for number in 100..102 {
            let block = ctx(number);
            assert!(admit_block(&mut readiness, &mut cache, &mut trie_cache, &block));
            apply(&mut cache, number);
            observe_readiness(&mut readiness, &block, &cache, &trie_cache);
        }

        // 105 is not the child of 101, so nothing the caches hold describes its parent.
        let gapped = ctx(105);
        assert!(
            admit_block(&mut readiness, &mut cache, &mut trie_cache, &gapped),
            "the gapped block is still applied, against caches that were reset first"
        );

        assert_eq!(cache.current_block(), 0);
        assert!(!cache.contains_account(&TOUCHED), "stale entries cannot survive the gap");
        assert_eq!(readiness.replay_depth(), 0);
        assert_eq!(*readiness.state(), CacheReadiness::Applying { block_number: 105 });
    }

    #[test]
    fn a_sibling_at_the_expected_height_is_treated_as_a_gap() {
        let mut readiness = tracker();
        let mut cache = CacheConfig::default().new_cache();
        let mut trie_cache = PartialTrieNodeCache::new();

        let first = ctx(100);
        assert!(admit_block(&mut readiness, &mut cache, &mut trie_cache, &first));
        apply(&mut cache, 100);
        observe_readiness(&mut readiness, &first, &cache, &trie_cache);

        // Right height, wrong branch: only the parent hash distinguishes the two.
        let sibling = BlockContext { parent_hash: B256::repeat_byte(0xfe), ..ctx(101) };
        assert!(admit_block(&mut readiness, &mut cache, &mut trie_cache, &sibling));

        assert_eq!(cache.current_block(), 0, "the competing branch's parent forced a reset");
    }

    #[test]
    fn an_unapplied_block_withholds_the_height_acknowledgement_until_recovery() {
        let mut readiness = tracker();
        let mut cache = CacheConfig::default().new_cache();
        let mut trie_cache = PartialTrieNodeCache::new();

        let first = ctx(100);
        assert!(admit_block(&mut readiness, &mut cache, &mut trie_cache, &first));
        apply(&mut cache, 100);
        observe_readiness(&mut readiness, &first, &cache, &trie_cache);
        assert!(readiness.may_acknowledge_height());

        // What the handler does when the parent state provider is unavailable.
        readiness.abandon_block(101);
        assert!(
            !readiness.may_acknowledge_height(),
            "acknowledging here would let reth prune the block the caches never saw"
        );

        // The next block's admission performs the recovery reset, which restores the promise.
        let next = ctx(102);
        assert!(admit_block(&mut readiness, &mut cache, &mut trie_cache, &next));
        apply(&mut cache, 102);
        observe_readiness(&mut readiness, &next, &cache, &trie_cache);

        assert!(readiness.may_acknowledge_height());
        assert_eq!(readiness.replay_depth(), 1, "recovery warms from scratch");
    }

    #[test]
    fn block_context_reads_the_header_fields_it_names() {
        let parent_hash = B256::repeat_byte(0x77);
        let state_root = B256::repeat_byte(0x88);
        let mut block = reth_ethereum::Block::default();
        block.header.number = 4_242;
        block.header.parent_hash = parent_hash;
        block.header.state_root = state_root;
        let block = block.seal_slow().try_recover().expect("empty body needs no sender recovery");

        let context = block_context(&block);

        assert_eq!(context.number, 4_242);
        assert_eq!(context.parent_hash, parent_hash);
        assert_eq!(context.state_root, state_root);
        assert_eq!(context.hash, block.hash());
        assert_ne!(context.hash, context.parent_hash);
    }
}

fn main() -> eyre::Result<()> {
    reth_ethereum::cli::Cli::parse_args().run(async move |builder, _| {
        let config = CacheConfig::default();

        let handle: NodeHandleFor<EthereumNode> = builder
            .node(EthereumNode::default())
            .install_exex("partial-stateless", move |ctx| async move {
                Ok(partial_stateless_exex(ctx, config))
            })
            .launch()
            .await?;

        handle.wait_for_node_exit().await
    })
}
