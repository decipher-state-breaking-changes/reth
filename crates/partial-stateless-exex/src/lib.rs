//! Partial Statelessness ExEx — reth Execution Extension that maintains a
//! network-level state cache and reports witness requirements per block.
//!
//! Run the binary with:
//!   cargo run -p partial-stateless-exex -- node --chain mainnet --datadir /path/to/data
//!
//! This ExEx subscribes to canonical chain commits and:
//! 1. Extracts `BlockAccessedState` via EVM simulation of each block
//! 2. Updates the `NetworkStateCache` with the accessed state
//! 3. Computes and logs cache miss ratio (= witness requirement)
//! 4. Builds and measures the canonical parent-state transition witness
//!
//! The crate is split into a library and a thin binary so that the recovery, bootstrap, and
//! admission paths can be tested from `tests/`: every one of those tests needs a state provider
//! and an EVM, which the `partial-stateless` library deliberately does not depend on.

pub mod access_shadow;
pub mod bootstrap_io;
pub mod cold_eoa;
pub mod rebuild;

mod benchmark;
mod sidecar_create;
mod sidecar_io;
mod sidecar_reexec;
mod sidecar_verify;

use alloy_eips::BlockNumHash;
use alloy_primitives::B256;
use alloy_rlp::Encodable;
use futures::TryStreamExt;
use partial_stateless::{
    network_cache::NetworkStateCache,
    persistence::{load_from_file, save_to_file},
    policy::{CachePolicy, LastNBlocksPolicy},
    readiness::{
        BlockContext, BlockedReason, CacheObservation, CacheReadiness, CacheReadinessTracker,
        ReadyParent,
    },
    sidecar::last_n_blocks_cache_policy_id,
    PartialStatelessSidecar, PartialTrieNodeCache,
};
use partial_stateless_validator::{
    admit_block, block_context, inject_recovery, BlockAdmission, CanonicalStateRoots,
    CoordinatedPair, RetainedGenerationBytes, ValidatorRules,
};
use reth_ethereum::{
    chainspec::EthChainSpec,
    exex::{ExExContext, ExExEvent, ExExNotification},
    node::api::{FullNodeComponents, NodeTypes},
    provider::StateProviderFactory,
    EthPrimitives,
};
use reth_primitives_traits::{AlloyBlockHeader, BlockTy, RecoveredBlock, SealedHeader};
use reth_provider::{
    BlockIdReader, BlockReader, CanonicalOverlayFactory, HeaderProvider, ProviderResult,
};
use reth_trie_common::MultiProofTargetsV2;
use reth_trie_parallel::proof_task::parallel_multiproof_v2_with_stats;
use sidecar_create::{
    create_sidecar_for_block, BuilderBlockReport, BuilderOptions, ParallelInitialProofFn,
    ParallelInitialProofOutput,
};
use sidecar_reexec::SidecarReexecLimits;
use sidecar_verify::verify_live_sidecar;
use std::{
    ops::{Deref, DerefMut},
    path::{Path, PathBuf},
    time::Duration,
};
use tracing::{error, info, warn};

/// Configuration for the partial statelessness cache.
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

/// What this process does with sidecars.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SidecarRole {
    /// Build and publish sidecars.
    Builder,
    /// Build, preflight, and publish sidecars in one process.
    BuilderVerifier,
    /// Consume sidecars produced elsewhere.
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

/// Reads a flag whose absence means on rather than off.
///
/// Separate from [`env_flag`] because the two answer different questions. `env_flag` asks whether
/// a run opted into something extra; this asks whether a run opted *out* of production behaviour,
/// which a benchmark control does and nothing else should.
fn env_flag_enabled_by_default(name: &str) -> bool {
    std::env::var(name)
        .map(|v| !matches!(v.as_str(), "0" | "false" | "FALSE" | "no" | "off"))
        .unwrap_or(true)
}

fn env_u32(name: &str, default: u32) -> u32 {
    let Ok(value) = std::env::var(name) else { return default };
    // A bare `PS_..=1`-style flag should still mean "on"; a count is the more useful spelling.
    match value.as_str() {
        "true" | "TRUE" | "yes" | "on" => 1,
        other => other.parse().unwrap_or(default),
    }
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

/// Everything the environment decided for one run, resolved once at startup.
#[derive(Debug, Clone)]
pub struct RunOptions {
    /// Cache windows and the policy identifier derived from them.
    pub config: CacheConfig,
    /// Whether this process builds, builds and preflights, or consumes sidecars.
    pub sidecar_role: SidecarRole,
    /// Where sidecars are read from or written to.
    pub sidecar_dir: PathBuf,
    /// How long the live verifier waits for a block's sidecar file to appear.
    pub verifier_wait: Duration,
    /// Where per-block accessed-state fixtures are dumped, when capture is on.
    pub capture_dir: Option<PathBuf>,
    /// Whether to compute the full-witness baseline for the reduction ratio.
    pub compute_baseline: bool,
    /// Whether to sample process CPU time and page faults around witness construction.
    pub resource_metrics: bool,
    /// Whether to validate retained trie paths and log trie shape every block.
    pub trie_cache_diagnostics: bool,
    /// Whether the builder preflights each sidecar before publishing it.
    pub run_sidecar_preflight: bool,
    /// Where paired Partial/Weak validation records go.
    pub validation_bench_output: Option<PathBuf>,
    /// Where per-block builder records go.
    pub builder_bench_output: Option<PathBuf>,
    /// B2 benchmark control that recreates the old unconditional parent-cache clone.
    pub force_previous_cache_snapshot: bool,
    /// Whether a committed block's displaced trie generation is kept for depth-1 recovery.
    ///
    /// On in production: K = 1 is what makes a depth-1 reorg an undo rather than a rebuild. The
    /// only reason to turn it off is the memory control — an otherwise identical run that pays no
    /// retention, so the difference in resident memory is attributable to retention alone.
    pub retain_generation: bool,
    /// Whether eligible initial V2 multiproofs use reth's proof workers.
    pub parallel_initial_proof: bool,
    /// Whether the paired in-memory validation benchmark is running.
    pub validation_bench: bool,
    /// Bounds on sidecar witness decoding.
    pub reexec_limits: SidecarReexecLimits,
    /// Whether a cold or recovering pair is rebuilt from canonical state instead of warmed.
    ///
    /// Opt-in, because the choice is a trade between two costs rather than a free win: the
    /// rebuild pays a whole-cache multiproof up front — tens of seconds, in one long-lived read
    /// transaction — while warming pays a full policy window of live blocks instead, spread out
    /// and invisible. A run that cannot afford a stall at its start wants it off; a run that
    /// cannot afford to lose a window of blocks per cache epoch wants it on.
    pub canonical_rebuild: bool,
    /// Where snapshot packages and their checkpoints live.
    pub bootstrap_dir: PathBuf,
    /// Whether to export a snapshot the first time the tracker reaches Ready.
    pub bootstrap_export: bool,
    /// Whether to restore from a snapshot at startup, ahead of the persisted flat cache.
    pub bootstrap_import: bool,
    /// How many blocks a snapshot-restored shadow pair is carried and compared against the live
    /// pair. Zero disables the in-process bootstrap gate.
    pub bootstrap_self_test_blocks: u32,
}

impl RunOptions {
    /// Reads the run configuration from the environment.
    pub fn from_env(config: CacheConfig) -> eyre::Result<Self> {
        let sidecar_role = SidecarRole::from_env();
        let sidecar_dir = configured_sidecar_dir();
        let validation_bench = env_flag("PS_VALIDATION_BENCH");
        if validation_bench && sidecar_role != SidecarRole::BuilderVerifier {
            return Err(eyre::eyre!("PS_VALIDATION_BENCH requires PS_SIDECAR_ROLE=builder-verifier"))
        }
        let bootstrap_self_test_blocks = env_u32("PS_BOOTSTRAP_SELF_TEST", 0);
        Ok(Self {
            config,
            sidecar_role,
            verifier_wait: env_millis("PS_SIDECAR_VERIFIER_WAIT_MS", 2_000),
            // The fixture dump and the paired benchmark both want the whole block budget; running
            // them together would measure the capture.
            capture_dir: (!validation_bench)
                .then(|| std::env::var_os("PS_CAPTURE_DIR").map(PathBuf::from))
                .flatten(),
            compute_baseline: !validation_bench && env_flag("PS_WITNESS_BASELINE"),
            resource_metrics: !validation_bench && env_flag("PS_RESOURCE_METRICS"),
            trie_cache_diagnostics: !validation_bench && env_flag("PS_TRIE_CACHE_DIAGNOSTICS"),
            run_sidecar_preflight: sidecar_role.runs_preflight(),
            validation_bench_output: validation_bench.then(|| {
                std::env::var_os("PS_BENCH_OUTPUT")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| sidecar_dir.join("validation_bench.jsonl"))
            }),
            builder_bench_output: std::env::var_os("PS_BUILDER_BENCH_OUTPUT").map(PathBuf::from),
            force_previous_cache_snapshot: env_flag("PS_FORCE_PREVIOUS_CACHE_SNAPSHOT"),
            retain_generation: env_flag_enabled_by_default("PS_RETAIN_GENERATION"),
            parallel_initial_proof: env_flag("PS_PARALLEL_INITIAL_PROOF"),
            validation_bench,
            reexec_limits: SidecarReexecLimits::default(),
            canonical_rebuild: env_flag("PS_CANONICAL_REBUILD"),
            bootstrap_dir: std::env::var_os("PS_BOOTSTRAP_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| sidecar_dir.join("bootstrap")),
            // The self-test needs a package to restore from, so it implies the export.
            bootstrap_export: env_flag("PS_BOOTSTRAP_EXPORT") || bootstrap_self_test_blocks > 0,
            bootstrap_import: env_flag("PS_BOOTSTRAP_IMPORT"),
            bootstrap_self_test_blocks,
            sidecar_dir,
        })
    }

    fn log_summary(&self, cache_path: &Path) {
        info!(
            target: "partial_stateless",
            account_window = self.config.account_window,
            storage_window = self.config.storage_window,
            cache_path = %cache_path.display(),
            sidecar_role = ?self.sidecar_role,
            sidecar_dir = %self.sidecar_dir.display(),
            verifier_wait_ms = self.verifier_wait.as_millis(),
            canonical_rebuild = self.canonical_rebuild,
            "Partial Stateless ExEx started — monitoring cache state per block"
        );
        if let Some(dir) = &self.capture_dir {
            info!(
                target: "partial_stateless",
                dir = %dir.display(),
                "Accessed-state fixture capture ENABLED (PS_CAPTURE_DIR) — run until ~300 blocks captured"
            );
        }
        if self.compute_baseline {
            info!(
                target: "partial_stateless",
                "Full-witness baseline comparison ENABLED (PS_WITNESS_BASELINE) — extra multiproof per block"
            );
        }
        if self.trie_cache_diagnostics {
            info!(
                target: "partial_stateless",
                "Trie-shape cache diagnostics ENABLED (PS_TRIE_CACHE_DIAGNOSTICS) — per-block timings, depth-0..5 prefix coverage, and full retained-path validation"
            );
        }
        if self.resource_metrics {
            info!(
                target: "partial_stateless",
                "Process-wide resource metrics ENABLED (PS_RESOURCE_METRICS) — includes parallel proof-worker CPU and page faults"
            );
            #[cfg(not(target_os = "linux"))]
            warn!(
                target: "partial_stateless",
                "Process CPU/page-fault metrics require getrusage; this platform will log default zeros for cpu_time_ms, major_page_faults, and minor_page_faults"
            );
            if self.compute_baseline {
                warn!(
                    target: "partial_stateless",
                    "PS_WITNESS_BASELINE runs before the transition witness; resource/page-fault metrics may be lower because the full baseline can warm the OS page cache"
                );
            }
        }
        if self.parallel_initial_proof {
            info!(
                target: "partial_stateless",
                "Parallel initial V2 multiproof ENABLED (PS_PARALLEL_INITIAL_PROOF); low-width target sets remain serial"
            );
        }
        if !self.retain_generation {
            warn!(
                target: "partial_stateless",
                "Depth-1 retained generation DISABLED (PS_RETAIN_GENERATION=0) — benchmark memory \
                 control only. Every reorg and revert now costs a full rebuild; do not read this \
                 run's recovery timings as production behaviour"
            );
        }
        if self.run_sidecar_preflight {
            info!(
                target: "partial_stateless",
                sidecar_role = ?self.sidecar_role,
                "Provider-assisted sidecar preflight ENABLED — extra re-execution per sidecar"
            );
        }
        if let Some(path) = &self.validation_bench_output {
            info!(
                target: "partial_stateless",
                path = %path.display(),
                "Paired Partial/Weak/Vanilla validation benchmark ENABLED (PS_VALIDATION_BENCH)"
            );
        }
        if let Some(path) = &self.builder_bench_output {
            info!(
                target: "partial_stateless",
                path = %path.display(),
                "Structured builder benchmark output ENABLED (PS_BUILDER_BENCH_OUTPUT)"
            );
        }
        if self.force_previous_cache_snapshot {
            warn!(
                target: "partial_stateless",
                "Forced previous-cache snapshot ENABLED for B2 benchmark control"
            );
        }
        if self.sidecar_role == SidecarRole::Verifier {
            info!(
                target: "partial_stateless",
                sidecar_dir = %self.sidecar_dir.display(),
                verifier_wait_ms = self.verifier_wait.as_millis(),
                "Live sidecar verifier ENABLED — consuming sidecar files and advancing cache only after successful verification"
            );
        }
        if self.bootstrap_export || self.bootstrap_import || self.bootstrap_self_test_blocks > 0 {
            info!(
                target: "partial_stateless",
                dir = %self.bootstrap_dir.display(),
                export = self.bootstrap_export,
                import = self.bootstrap_import,
                self_test_blocks = self.bootstrap_self_test_blocks,
                "Operator-trusted snapshot bootstrap ENABLED — a node restoring from this package \
                 trusts whoever supplied its checkpoint"
            );
        }
    }

    fn builder_options<'a>(
        &'a self,
        parallel_initial_proof: Option<&'a ParallelInitialProofFn<'a>>,
        ready_parent: Option<&'a ReadyParent>,
        retain_sidecar: bool,
        retained_generation: RetainedGenerationBytes,
    ) -> BuilderOptions<'a> {
        BuilderOptions {
            capture_dir: self.capture_dir.as_deref(),
            sidecar_dir: &self.sidecar_dir,
            compute_baseline: self.compute_baseline,
            resource_metrics: self.resource_metrics,
            trie_cache_diagnostics: self.trie_cache_diagnostics,
            run_sidecar_preflight: self.run_sidecar_preflight,
            validation_bench_output: self.validation_bench_output.as_deref(),
            builder_bench_output: self.builder_bench_output.as_deref(),
            force_previous_cache_snapshot: self.force_previous_cache_snapshot,
            retained_generation,
            reexec_limits: &self.reexec_limits,
            parallel_initial_proof,
            ready_parent,
            retain_sidecar,
        }
    }
}

/// The coordinated pair as this ExEx carries it: protocol state plus what the run log needs.
///
/// `CoordinatedPair` lives in `partial-stateless-validator` and holds protocol state only, so the
/// label below — which exists purely so the run checklist can read "exactly one transition to
/// ready" — stays on this side rather than travelling into a validator that has no run log.
///
/// `Deref` is here so the hundred-odd `pair.cache` / `pair.readiness` sites read the same after
/// the extraction as before it. This is a private newtype over one field, not a base class.
struct LivePair {
    coordinated: CoordinatedPair,
    /// Last classification that was *reported*, which is never the transient `Applying`.
    ///
    /// Reading the tracker's own label instead would report a change on every block: admitting a
    /// block moves it to `Applying` before anything can observe the previous classification, so
    /// `Applying -> Ready` would be logged once per block and the run checklist's "exactly one
    /// transition to ready" would be unreadable.
    last_readiness_label: &'static str,
}

impl LivePair {
    /// Wraps a freshly built or restored pair, reporting whatever it is already classified as.
    fn new(coordinated: CoordinatedPair) -> Self {
        let last_readiness_label = coordinated.readiness.state().label();
        Self { coordinated, last_readiness_label }
    }
}

impl Deref for LivePair {
    type Target = CoordinatedPair;

    fn deref(&self) -> &Self::Target {
        &self.coordinated
    }
}

impl DerefMut for LivePair {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.coordinated
    }
}

/// State of the in-process bootstrap gate: export once, then carry a second pair for a few blocks.
struct BootstrapGate {
    export_pending: bool,
    self_test_blocks: u32,
    shadow: Option<ShadowPair>,
}

impl BootstrapGate {
    const fn new(options: &RunOptions) -> Self {
        Self {
            export_pending: options.bootstrap_export,
            self_test_blocks: options.bootstrap_self_test_blocks,
            shadow: None,
        }
    }

    const fn wants_sidecar(&self) -> bool {
        self.shadow.is_some()
    }
}

/// A second coordinated pair restored from an exported snapshot, carried alongside the live one.
///
/// This is how the sync/bootstrap gate closes inside one process. Two live runs cannot overlap on
/// one datadir, and sequencing them lets the chain advance across the restart, so the importing
/// run would restore at H and receive H + k — bridging that drift with a canonical rebuild would
/// mean the snapshot did no work and the gate never closed.
struct ShadowPair {
    pair: LivePair,
    remaining_blocks: u32,
    restored_at: u64,
}

/// The ExEx function that processes chain notifications and maintains the cache.
pub async fn partial_stateless_exex<
    Node: FullNodeComponents<Types: NodeTypes<Primitives = EthPrimitives>>,
>(
    mut ctx: ExExContext<Node>,
    config: CacheConfig,
) -> eyre::Result<()>
where
    Node::Provider: CanonicalOverlayFactory + BlockReader<Block = BlockTy<EthPrimitives>>,
{
    // Resolve the cache file path: datadir/partial_stateless_cache.bin
    let cache_dir = ctx.config.datadir.clone().resolve_datadir(ctx.config.chain.chain());
    let cache_path = cache_dir.as_ref().join("partial_stateless_cache.bin");
    let options = RunOptions::from_env(config)?;
    options.log_summary(&cache_path);

    let mut pair = load_initial_pair(&options, &cache_path, ctx.head.number);
    let mut gate = BootstrapGate::new(&options);
    // A rebuild that keeps failing is almost always a persistent condition — pruned history, or a
    // provider that cannot reach far enough back — and retrying it every block would spend the
    // whole run re-executing windows that never install.
    let mut rebuild_failures = 0u32;

    while let Some(notification) = ctx.notifications.try_next().await? {
        match &notification {
            ExExNotification::ChainCommitted { new } => {
                let tip_block = *new.range().end();

                // A cold pair has nothing to warm from but live blocks, which costs a full policy
                // window. A full node can instead rebuild the exact pair at the parent this
                // notification starts from, and be Ready for its first block.
                if let Some(first) = new.blocks().values().next() {
                    maybe_rebuild_before_applying(
                        &ctx,
                        &options,
                        &mut pair,
                        first.parent_hash,
                        &mut rebuild_failures,
                    );
                }

                for (block_number, block) in new.blocks() {
                    if let Err(err) = process_canonical_block(
                        &ctx,
                        &options,
                        &mut pair,
                        &mut gate,
                        &mut rebuild_failures,
                        block,
                    ) {
                        return Err(eyre::eyre!("block {block_number} failed: {err:#}"))
                    }
                }

                // Cache persistence is unrelated to validation and can perturb later
                // Engine samples, so the bounded paired benchmark keeps it in memory only.
                persist_cache(&options, &pair, &cache_path, tip_block, true);
            }
            ExExNotification::ChainReorged { old, new } => {
                let tip_block = *new.range().end();
                // The common ancestor, addressed the only way that is unambiguous mid-reorg: the
                // parent hash of the new chain's first block. A height would name whichever block
                // the database currently calls canonical at that number.
                let ancestor_hash = new.blocks().values().next().map(|block| block.parent_hash);
                warn!(
                    target: "partial_stateless",
                    from_chain = ?old.range(),
                    to_chain = ?new.range(),
                    ?ancestor_hash,
                    "Chain reorg detected — recovering the coordinated pair at the common ancestor"
                );

                // The pair still sits on the abandoned branch until a rebuild replaces it. It is
                // left in place rather than cleared: `Recovering` already refuses every block, and
                // clearing it here would make a failed recovery look like a clean cold start,
                // which is exactly the distinction the state exists to preserve.
                recover_at(
                    &ctx,
                    &options,
                    &mut pair,
                    *old.range().start(),
                    ancestor_hash,
                    &mut rebuild_failures,
                );

                // Apply the new canonical chain block-by-block, so the builder still produces a
                // sidecar for each block on the new branch. Rebuilding directly at the new tip
                // would be fewer steps and would skip all of them.
                for (block_number, block) in new.blocks() {
                    if let Err(err) = process_canonical_block(
                        &ctx,
                        &options,
                        &mut pair,
                        &mut gate,
                        &mut rebuild_failures,
                        block,
                    ) {
                        return Err(eyre::eyre!("reorg block {block_number} failed: {err:#}"))
                    }
                }

                // Production persists the rebuilt cache so a restart cannot reload the old
                // branch. Benchmark mode deliberately keeps all cache state in memory.
                persist_cache(&options, &pair, &cache_path, tip_block, true);
            }
            ExExNotification::ChainReverted { old } => {
                // The new tip is the parent of the reverted chain's first block. Addressing it by
                // hash rather than by `range().start() - 1` matters for the same reason it does on
                // the reorg path, and the hash is already in the notification.
                let new_tip_hash = old.blocks().values().next().map(|block| block.parent_hash);
                warn!(
                    target: "partial_stateless",
                    reverted_chain = ?old.range(),
                    ?new_tip_hash,
                    "Chain reverted — recovering the coordinated pair at the new tip"
                );

                let recovered = recover_at(
                    &ctx,
                    &options,
                    &mut pair,
                    *old.range().start(),
                    new_tip_hash,
                    &mut rebuild_failures,
                );

                // A pair that could not be rebuilt still describes the reverted branch, and
                // persisting it would let a restart reload exactly that.
                persist_cache(
                    &options,
                    &pair,
                    &cache_path,
                    old.range().start().saturating_sub(1),
                    recovered,
                );
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
        let threshold =
            ctx.provider().finalized_block_number().ok().flatten().unwrap_or_else(|| {
                pair.cache.current_block().saturating_sub(UNDO_LOG_FALLBACK_DEPTH)
            });
        pair.cache.prune_undo_below(threshold);

        // Acknowledge the highest *contiguously* processed block rather than the notification tip.
        // reth prunes below this height and never redelivers those blocks, so acknowledging a tip
        // above a block this ExEx skipped would lose that block permanently. Resetting the caches
        // after a gap restores the caches but does not process the missing block, which is why the
        // watermark can sit far below the tip while the caches are Ready again.
        if let Some(committed_chain) = notification.committed_chain() {
            let tip = committed_chain.tip().number;
            match pair.readiness.acknowledgeable_height() {
                Some((number, hash)) => {
                    ctx.events.send(ExExEvent::FinishedHeight(BlockNumHash::new(number, hash)))?;
                    if let Some(gap) = pair.readiness.first_gap() {
                        error!(
                            target: "partial_stateless",
                            tip,
                            acknowledged = number,
                            missing_block = gap,
                            "Processed-height acknowledgement is pinned below an unprocessed block; \
                             reth cannot prune above it until a reorg drops that block"
                        );
                    }
                }
                None => {
                    error!(
                        target: "partial_stateless",
                        tip,
                        state = pair.readiness.state().label(),
                        "No block has been processed contiguously; withholding acknowledgement"
                    );
                }
            }
        }
    }

    Ok(())
}

/// Writes the flat cache to disk, unless the run is a bounded in-memory benchmark.
///
/// `canonical` is false when the pair may still describe an abandoned branch, in which case
/// persisting would let a restart reload exactly that.
fn persist_cache(
    options: &RunOptions,
    pair: &LivePair,
    cache_path: &Path,
    block: u64,
    canonical: bool,
) {
    if options.validation_bench || !canonical {
        return
    }
    if let Err(e) = save_to_file(&pair.cache, cache_path) {
        warn!(
            target: "partial_stateless",
            block,
            error = %e,
            "Failed to save cache state to disk"
        );
    }
}

/// Builds the coordinated pair this run starts from.
///
/// A snapshot import takes precedence over the persisted flat cache, which loads values and then
/// discards them for want of a matching trie snapshot — restoring both halves together is the
/// whole point of the package.
fn load_initial_pair(options: &RunOptions, cache_path: &Path, head_block: u64) -> LivePair {
    let config = &options.config;
    if options.bootstrap_import {
        match bootstrap_io::load_snapshot(&options.bootstrap_dir) {
            Ok(Some((package, checkpoint))) => {
                match bootstrap_io::restore_snapshot(package, &checkpoint, config) {
                    Ok(restored) => {
                        bootstrap_io::warn_on_head_drift(&checkpoint, head_block + 1);
                        return LivePair::new(CoordinatedPair {
                            cache: restored.cache,
                            trie_cache: restored.trie_cache,
                            previous_generation: None,
                            accepted_head: None,
                            readiness: restored.readiness,
                        })
                    }
                    Err(err) => error!(
                        target: "partial_stateless",
                        error = %err,
                        "Snapshot bootstrap rejected; starting cold"
                    ),
                }
            }
            Ok(None) => warn!(
                target: "partial_stateless",
                dir = %options.bootstrap_dir.display(),
                "PS_BOOTSTRAP_IMPORT is set but no snapshot package is present; starting cold"
            ),
            Err(err) => error!(
                target: "partial_stateless",
                error = %err,
                "Failed to read the bootstrap snapshot; starting cold"
            ),
        }
    }

    // Cache coherence guarantee: the LastNBlocksPolicy is fully deterministic —
    // given the same canonical chain and window parameters, all peers converge
    // to the same cache state. Loading from disk and replaying missed blocks
    // during sync produces an identical result to continuous operation.
    let mut cache = if cache_path.exists() {
        match load_from_file(cache_path, config.account_policy(), config.storage_policy()) {
            Ok(loaded_cache) => {
                let cache_block = loaded_cache.current_block();

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

    // This persistent-in-memory sparse trie mirrors value-cache account and storage paths. It has
    // no persisted snapshot or branch-aware undo log yet, so it must reset together with values.
    if cache.current_block() != 0 {
        warn!(
            target: "partial_stateless",
            cache_block = cache.current_block(),
            "Persisted values have no matching sparse-trie snapshot; cold-resetting both caches"
        );
        cache.reset();
    }

    // Tracks whether the two caches together still describe the parent of the block being
    // processed. Both start cold here, and the tracker starts cold with them.
    LivePair::new(CoordinatedPair {
        cache,
        trie_cache: PartialTrieNodeCache::new(),
        readiness: config.new_readiness_tracker(),
        previous_generation: None,
        accepted_head: None,
    })
}

/// Rebuilds the coordinated pair at `parent_hash` when it is cold and a rebuild is available.
///
/// Trades a bounded stall for a full policy window of live warming — roughly `window + 1`
/// historical executions and one multiproof, against about twelve minutes of live blocks at
/// mainnet block time — and it is the only way a pair that was reset mid-run becomes useful again
/// inside a bounded run. Available only under `PS_CANONICAL_REBUILD`, so a run that would rather
/// start immediately and warm quietly gets that by default.
fn maybe_rebuild_before_applying<Node>(
    ctx: &ExExContext<Node>,
    options: &RunOptions,
    pair: &mut LivePair,
    parent_hash: B256,
    failures: &mut u32,
) where
    Node: FullNodeComponents<Types: NodeTypes<Primitives = EthPrimitives>>,
    Node::Provider: BlockReader<Block = BlockTy<EthPrimitives>>,
{
    if !matches!(pair.readiness.state(), CacheReadiness::Cold) {
        return
    }
    rebuild_pair_at(ctx, options, pair, parent_hash, failures);
}

/// Recovers the pair at an exact canonical block after a reorg or revert.
///
/// Returns whether the pair now describes the canonical chain. On failure the tracker is left
/// `Recovering`, which refuses every block: the next notification's `admit_block` reports
/// `RecoveryIncomplete` and the fallback to a cold warm-up is taken explicitly and logged, rather
/// than silently sharing a code path with a clean cold start.
fn recover_at<Node>(
    ctx: &ExExContext<Node>,
    options: &RunOptions,
    pair: &mut LivePair,
    unwound_from: u64,
    target_hash: Option<B256>,
    failures: &mut u32,
) -> bool
where
    Node: FullNodeComponents<Types: NodeTypes<Primitives = EthPrimitives>>,
    Node::Provider: BlockReader<Block = BlockTy<EthPrimitives>>,
{
    let Some(target_hash) = target_hash else {
        pair.readiness.begin_recovery(unwound_from);
        error!(
            target: "partial_stateless",
            "Recovery target is unknown: the notification carried no blocks to take a parent hash \
             from"
        );
        return false
    };

    // Depth-1 fast path. Every rejection falls through to the rebuild below, including a failed
    // header lookup: the rebuild is the correct answer whenever the cheap answer cannot be proven.
    if inject_recovery(
        pair,
        &CanonicalChain(ctx.provider()),
        unwound_from,
        target_hash,
        options.config.cache_policy_id(),
    )
    .is_some()
    {
        *failures = 0;
        return true
    }

    rebuild_pair_at(ctx, options, pair, target_hash, failures)
}

/// The rules this ExEx validates under: the node's own EVM config and the node's own consensus.
///
/// Taking `ctx.components.consensus()` rather than constructing an `EthBeaconConsensus` here is
/// load-bearing. The consensus object carries flags — `skip_requests_hash_check` among them — that
/// decide what a block is allowed to be, so a freshly built instance could reject a block this
/// very node's Engine already accepted, and the ExEx would report a consensus failure that says
/// more about its own configuration than about the block.
fn node_rules<Node>(ctx: &ExExContext<Node>) -> ValidatorRules<'_, Node::Evm, Node::Consensus>
where
    Node: FullNodeComponents,
{
    ValidatorRules::new(ctx.evm_config(), ctx.components.consensus())
}

/// Adapts a node provider to [`CanonicalStateRoots`].
struct CanonicalChain<P>(P);

impl<P: HeaderProvider<Header: AlloyBlockHeader>> CanonicalStateRoots for CanonicalChain<P> {
    fn state_root_of(&self, hash: B256) -> ProviderResult<Option<B256>> {
        Ok(self.0.sealed_header_by_hash(hash)?.map(|header| header.state_root()))
    }
}

fn rebuild_pair_at<Node>(
    ctx: &ExExContext<Node>,
    options: &RunOptions,
    pair: &mut LivePair,
    target_hash: B256,
    failures: &mut u32,
) -> bool
where
    Node: FullNodeComponents<Types: NodeTypes<Primitives = EthPrimitives>>,
    Node::Provider: BlockReader<Block = BlockTy<EthPrimitives>>,
{
    const MAX_CONSECUTIVE_REBUILD_FAILURES: u32 = 3;
    // `false` here means the operator did not ask for a rebuild, not that one failed. Callers read
    // it the same way either way: warm or cold-reset instead, which is the ordinary path now.
    if !options.canonical_rebuild {
        return false
    }
    if *failures >= MAX_CONSECUTIVE_REBUILD_FAILURES {
        return false
    }

    let rebuilt = rebuild::rebuild_coordinated_pair(
        ctx.provider(),
        ctx.evm_config(),
        target_hash,
        &options.config,
    )
    .and_then(|rebuilt| {
        rebuild::install_rebuilt_pair(
            rebuilt,
            &mut pair.coordinated.cache,
            &mut pair.coordinated.trie_cache,
            &mut pair.coordinated.readiness,
        )
    });

    match rebuilt {
        Ok(ready) => {
            *failures = 0;
            // The rebuild replaced the pair from canonical state; whatever was retained described
            // a generation this one does not descend from. The accepted head goes with it, and for
            // a sharper reason: a reorg rebuild installs the winning sibling at the *same* number
            // the abandoned block had, so a header left behind here would be one `accepted_parent`
            // has to reject on hash rather than on height. Clearing it makes the state honest
            // instead of leaving the guard to catch it, and the pair readmits a head by applying
            // its next block.
            pair.forget_retained_generation();
            pair.accepted_head = None;
            pair.last_readiness_label = pair.readiness.state().label();
            info!(
                target: "partial_stateless",
                block = ready.anchor.block_number,
                block_hash = ?ready.anchor.block_hash,
                cache_root = ?ready.anchor.cache_root,
                trie_state_root = ?ready.trie_state_root,
                "Coordinated pair is Ready from a canonical rebuild rather than from live warming"
            );
            true
        }
        Err(err) => {
            *failures += 1;
            error!(
                target: "partial_stateless",
                ?target_hash,
                consecutive_failures = *failures,
                max_failures = MAX_CONSECUTIVE_REBUILD_FAILURES,
                error = %err,
                "Canonical rebuild failed"
            );
            false
        }
    }
}

/// Applies one canonical block to the live pair, and to the bootstrap gate's shadow pair.
fn process_canonical_block<Node>(
    ctx: &ExExContext<Node>,
    options: &RunOptions,
    pair: &mut LivePair,
    gate: &mut BootstrapGate,
    rebuild_failures: &mut u32,
    block: &RecoveredBlock<BlockTy<EthPrimitives>>,
) -> eyre::Result<()>
where
    Node: FullNodeComponents<Types: NodeTypes<Primitives = EthPrimitives>>,
    Node::Provider: CanonicalOverlayFactory + BlockReader<Block = BlockTy<EthPrimitives>>,
{
    let block_number = block.number();
    let block_ctx = block_context(block);
    let ready_parent = match admit_block(&mut pair.readiness, &block_ctx) {
        BlockAdmission::Admitted(ready_parent) => ready_parent,
        BlockAdmission::Rejected(reason) => {
            error!(
                target: "partial_stateless",
                block = block_number,
                ?reason,
                recovery_failed = matches!(reason, BlockedReason::RecoveryIncomplete { .. }),
                "Cache continuity broken"
            );
            // An exact rebuild at this block's own parent is worth trying before falling back to a
            // policy window of live warming, and it is the same primitive a reorg recovers with.
            if rebuild_pair_at(ctx, options, pair, block.parent_hash, rebuild_failures) {
                match admit_block(&mut pair.readiness, &block_ctx) {
                    BlockAdmission::Admitted(ready_parent) => ready_parent,
                    BlockAdmission::Rejected(reason) => eyre::bail!(
                        "block {block_number} was refused by a pair rebuilt at its own parent: \
                         {reason:?}"
                    ),
                }
            } else {
                admit_after_cold_reset(pair, &block_ctx)?
            }
        }
    };

    // The parent is addressed by hash rather than by number: a height can name a
    // block on an abandoned branch while a reorg is in flight, and building the
    // cache on that state would silently mix two chains.
    let state_provider = match ctx.provider().history_by_block_hash(block.parent_hash) {
        Ok(provider) => provider,
        Err(e) => {
            // The block is not applied, so every later block would build on
            // state these caches never saw. Recording that keeps the gap from
            // being papered over on the next block.
            let reason = pair.readiness.abandon_block(block_number);
            warn!(
                target: "partial_stateless",
                block = block_number,
                error = %e,
                ?reason,
                "Failed to get state provider for parent block. Skipping block."
            );
            return Ok(())
        }
    };

    if options.sidecar_role == SidecarRole::Verifier {
        let displaced_trie_cache = verify_live_sidecar(
            node_rules(ctx),
            state_provider.as_ref(),
            block,
            &mut pair.coordinated.cache,
            &mut pair.coordinated.trie_cache,
            &options.config,
            &options.sidecar_dir,
            &options.reexec_limits,
            options.verifier_wait,
            ready_parent.as_ref(),
        )
        .map_err(|err| eyre::eyre!("live sidecar verification failed: {err}"))?;
        // Retained on the verifier for the same reason and at the same cost as on the builder: a
        // depth-1 reorg is the common case, and undoing one block beats replaying a whole window.
        finish_committed_transition(
            pair,
            displaced_trie_cache,
            &block_ctx,
            block.clone_sealed_header(),
            options.retain_generation,
        );
        return Ok(())
    }

    let parent_state_root_by_hash = |parent_hash: B256| {
        ctx.provider()
            .sealed_header_by_hash(parent_hash)
            .map_err(|err| eyre::eyre!("failed to fetch parent header: {err}"))?
            .map(|header| header.state_root)
            .ok_or_else(|| eyre::eyre!("parent header not found for hash {:?}", parent_hash))
    };
    let ancestor_headers_for_range = |lowest_block_number: Option<u64>, block_number: u64| {
        let Some(smallest) = lowest_block_number else {
            return Ok(Vec::new());
        };
        let headers = ctx
            .provider()
            .headers_range(smallest..block_number)
            .map_err(|err| eyre::eyre!("failed to fetch ancestor headers for range: {err}"))?;
        Ok(headers
            .into_iter()
            .map(|header| {
                let mut buf = Vec::new();
                header.encode(&mut buf);
                buf.into()
            })
            .collect())
    };
    // Measured before the transition, so it describes the generation the previous block displaced
    // rather than the one this block is about to. Taking it after would report a retained trie that
    // still shares nearly every storage trie with the live cache and understate the steady cost.
    let retained_generation = pair.retained_generation_bytes(options.retain_generation);
    let parallel_initial_proof =
        |targets: MultiProofTargetsV2| -> ProviderResult<ParallelInitialProofOutput> {
            let parallel_factory = ctx.provider().overlay_factory_at_block(block.parent_hash);
            parallel_multiproof_v2_with_stats(ctx.task_executor(), parallel_factory, targets, true)
                .map(|(proof, stats)| ParallelInitialProofOutput {
                    proof,
                    storage_workers: stats.storage_workers,
                    account_workers: stats.account_workers,
                })
                .map_err(Into::into)
        };

    let report = create_sidecar_for_block(
        node_rules(ctx),
        state_provider.as_ref(),
        block,
        &mut pair.coordinated.cache,
        &mut pair.coordinated.trie_cache,
        &options.config,
        options.builder_options(
            options
                .parallel_initial_proof
                .then_some(&parallel_initial_proof as &ParallelInitialProofFn<'_>),
            ready_parent.as_ref(),
            gate.wants_sidecar(),
            retained_generation,
        ),
        parent_state_root_by_hash,
        ancestor_headers_for_range,
    )
    .map_err(|err| eyre::eyre!("sidecar builder failed: {err}"))?;
    let BuilderBlockReport {
        cache_update: _cache_update,
        witness: _witness,
        sidecar_path: _sidecar_path,
        sidecar,
        displaced_trie_cache,
    } = report;
    finish_committed_transition(
        pair,
        displaced_trie_cache,
        &block_ctx,
        block.clone_sealed_header(),
        options.retain_generation,
    );

    advance_bootstrap_gate(ctx, options, pair, gate, block, sidecar)
}

/// Installs the parent generation displaced by a successful transition and makes that transition
/// visible to readiness as one operation.
///
/// Both production roles call this exact tail: the builder gets `displaced_trie_cache` from
/// sidecar creation, while the verifier gets it from committed reexecution. Keeping the sequence
/// shared prevents either arm from becoming unable to service the same depth-1 recovery hook.
fn finish_committed_transition(
    pair: &mut LivePair,
    displaced_trie_cache: Option<PartialTrieNodeCache>,
    block: &BlockContext,
    accepted_head: SealedHeader,
    retain_generation: bool,
) {
    pair.retain_generation(
        displaced_trie_cache,
        block.parent_hash,
        block.number.saturating_sub(1),
        accepted_head,
        retain_generation,
    );
    observe_readiness(pair, block);
}

/// Runs the in-process sync/bootstrap gate: export at the first Ready, then compare.
///
/// Step by step this is the section 4.7 bootstrap gate almost verbatim: the run warms normally and
/// exports at `Ready(H)`; a second pair is restored from that package in the same process; and
/// when H + 1 arrives it is validated against both pairs through the same provider-free path. That
/// path already checks that the restored cache's own expected miss set equals the miss manifest
/// the live pair built, so miss-set agreement is structural rather than a separate assertion here.
fn advance_bootstrap_gate<Node>(
    ctx: &ExExContext<Node>,
    options: &RunOptions,
    pair: &mut LivePair,
    gate: &mut BootstrapGate,
    block: &RecoveredBlock<BlockTy<EthPrimitives>>,
    sidecar: Option<PartialStatelessSidecar>,
) -> eyre::Result<()>
where
    Node: FullNodeComponents<Types: NodeTypes<Primitives = EthPrimitives>>,
{
    let block_number = block.number();

    if let Some(shadow) = gate.shadow.as_mut() {
        let Some(sidecar) = sidecar else {
            eyre::bail!(
                "bootstrap gate is carrying a shadow pair at block {block_number} but the builder \
                 produced no sidecar to validate it against"
            )
        };
        let block_ctx = block_context(block);
        match admit_block(&mut shadow.pair.readiness, &block_ctx) {
            BlockAdmission::Admitted(_) => {}
            BlockAdmission::Rejected(reason) => eyre::bail!(
                "the bootstrapped pair refused block {block_number}, which the live pair \
                 accepted: {reason:?}"
            ),
        }
        // Deliberately the provider-free path: a bootstrapped verifier is the case that has no
        // database to fall back on, so validating it with provider help would prove nothing.
        let report = partial_stateless_validator::verify_and_apply_sidecar(
            node_rules(ctx),
            block,
            &mut shadow.pair.coordinated.cache,
            &sidecar,
            options.config.cache_policy_id(),
            &options.reexec_limits,
            &mut shadow.pair.coordinated.trie_cache,
            partial_stateless_validator::TrieCacheDisposition::Commit,
        )
        .map_err(|err| {
            eyre::eyre!("the bootstrapped pair failed to verify block {block_number}: {err:#}")
        })?;
        if report.outcome.state_root != block.state_root() {
            eyre::bail!(
                "the bootstrapped pair computed state root {:?} for block {block_number}, \
                 expected {:?}",
                report.outcome.state_root,
                block.state_root()
            );
        }
        observe_readiness(&mut shadow.pair, &block_ctx);

        let live = pair.fingerprint();
        let bootstrapped = shadow.pair.fingerprint();
        if live != bootstrapped {
            eyre::bail!(
                "the bootstrapped and continuously warmed pairs diverged at block \
                 {block_number}: live={live:?}, bootstrapped={bootstrapped:?}"
            );
        }
        info!(
            target: "partial_stateless",
            block = block_number,
            restored_at = shadow.restored_at,
            cache_root = ?live.cache_root,
            trie_cache_root = ?live.trie_cache_root,
            trie_state_root = ?live.trie_state_root,
            "Bootstrap gate: the snapshot-restored pair verified this block provider-free and \
             agrees with the continuously warmed pair"
        );

        shadow.remaining_blocks -= 1;
        if shadow.remaining_blocks == 0 {
            info!(
                target: "partial_stateless",
                blocks = options.bootstrap_self_test_blocks,
                "Bootstrap gate closed; dropping the shadow pair"
            );
            gate.shadow = None;
        }
    }

    if !gate.export_pending {
        return Ok(())
    }
    let Some(ready) = pair.readiness.ready_parent().cloned() else { return Ok(()) };
    // The export is attempted once. A failure here is operational — the proof provider, a resource
    // bound, the filesystem — not a correctness result about the caches, so it must not take the
    // node down, and retrying it every block would just repeat an expensive whole-cache multiproof.
    // Everything past this point is the gate itself, and those failures stay fatal.
    gate.export_pending = false;
    // The snapshot's proof must be answered against the state the Ready parent names, which is
    // this block's own post-state.
    let exported = match ctx
        .provider()
        .history_by_block_hash(ready.anchor.block_hash)
        .map_err(eyre::Report::from)
        .and_then(|state_provider| {
            bootstrap_io::export_snapshot(
                &options.bootstrap_dir,
                &pair.cache,
                &ready,
                state_provider.as_ref(),
            )
        }) {
        Ok(exported) => exported,
        Err(err) => {
            error!(
                target: "partial_stateless",
                block = ready.anchor.block_number,
                error = %err,
                self_test_blocks = gate.self_test_blocks,
                "Snapshot export failed; the bootstrap gate cannot run this session"
            );
            return Ok(())
        }
    };

    if gate.self_test_blocks > 0 {
        let restored = bootstrap_io::restore_snapshot(
            exported.package,
            &exported.checkpoint,
            &options.config,
        )?;
        let shadow = ShadowPair {
            pair: LivePair::new(CoordinatedPair {
                cache: restored.cache,
                trie_cache: restored.trie_cache,
                previous_generation: None,
                accepted_head: None,
                readiness: restored.readiness,
            }),
            remaining_blocks: gate.self_test_blocks,
            restored_at: ready.anchor.block_number,
        };
        let live = pair.fingerprint();
        let bootstrapped = shadow.pair.fingerprint();
        if live != bootstrapped {
            eyre::bail!(
                "the snapshot restored a different generation than it was exported from at block \
                 {}: live={live:?}, bootstrapped={bootstrapped:?}",
                ready.anchor.block_number
            );
        }
        info!(
            target: "partial_stateless",
            block = ready.anchor.block_number,
            self_test_blocks = gate.self_test_blocks,
            "Bootstrap gate: restored a second coordinated pair in-process; it matches the live \
             pair and will now validate the next blocks provider-free"
        );
        gate.shadow = Some(shadow);
    }
    Ok(())
}

/// Drops both caches and readmits the block, warming again from it.
///
/// The last resort when no rebuild is available: a block that is not the direct child of the last
/// one applied means the caches describe a state this chain never passed through, and no undo log
/// here is deep enough to unwind an arbitrary gap.
///
/// This is *not* inside [`admit_block`], which only reports. A reorg whose rebuild never completed
/// leaves the tracker `Recovering`, and repairing that with the same silent reset-and-retry as a
/// plain gap would make a failed recovery indistinguishable from a clean cold start.
fn admit_after_cold_reset(
    pair: &mut LivePair,
    block: &BlockContext,
) -> eyre::Result<Option<ReadyParent>> {
    warn!(
        target: "partial_stateless",
        block = block.number,
        "Cold-resetting both caches and warming again from this block"
    );
    // `cold_reset` rather than the same four field assignments open-coded here. They were
    // identical when this was written and then were not: the accepted head was added to the pair
    // and only the shared version learned to clear it.
    pair.cold_reset();
    match admit_block(&mut pair.readiness, block) {
        // The reset discarded whatever the token described, so this block publishes nothing.
        BlockAdmission::Admitted(_) => Ok(None),
        BlockAdmission::Rejected(reason) => Err(eyre::eyre!(
            "block {} could not be admitted even after a cold reset: {reason:?}",
            block.number
        )),
    }
}

/// Reclassifies the caches after `block` was applied, logging only genuine changes.
///
/// Readiness is advisory here: nothing yet refuses to build or verify a sidecar because the caches
/// are still warming. Gating on it would change what the benchmark measures.
fn observe_readiness(pair: &mut LivePair, block: &BlockContext) {
    let before = pair.last_readiness_label;
    let observation = CacheObservation::capture(&pair.cache, &pair.trie_cache);
    let after = pair.readiness.finish_block(block, &observation).label();
    pair.last_readiness_label = after;
    if before != after {
        info!(
            target: "partial_stateless",
            block = block.number,
            from = before,
            to = after,
            replay_depth = pair.readiness.replay_depth(),
            "Cache readiness changed"
        );
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
        admit_after_cold_reset, admit_block, block_context, finish_committed_transition,
        inject_recovery, observe_readiness, BlockAdmission, BlockContext, CacheConfig,
        CanonicalStateRoots, CoordinatedPair, LivePair, PartialTrieNodeCache, ProviderResult,
        SealedHeader, SidecarRole,
    };
    use alloy_primitives::{keccak256, Address, B256, U256};
    use partial_stateless::{
        accessed_state::BlockAccessedState,
        policy::AccountData,
        readiness::{BlockedReason, CacheReadiness, TrustedCheckpoint},
        CacheAnchor, CacheSnapshotPackage,
    };
    use reth_primitives_traits::{Account, Block as _};
    use reth_trie_common::{
        proof::ProofRetainer, HashBuilder, MultiProof, Nibbles, EMPTY_ROOT_HASH,
    };

    const TOUCHED: Address = Address::repeat_byte(0x11);

    fn cold_pair() -> LivePair {
        let config = CacheConfig::default();
        LivePair::new(CoordinatedPair {
            cache: config.new_cache(),
            trie_cache: PartialTrieNodeCache::new(),
            readiness: config.new_readiness_tracker(),
            previous_generation: None,
            accepted_head: None,
        })
    }

    /// Synthesizes a chain whose hashes and state roots derive from block numbers, so a test can
    /// name any block without a fixture.
    /// A header carrying the identity `ctx` describes, so a synthesized transition advances the
    /// accepted head the same way a real one does.
    fn sealed(block: &BlockContext) -> SealedHeader {
        SealedHeader::new(
            alloy_consensus::Header {
                number: block.number,
                parent_hash: block.parent_hash,
                state_root: block.state_root,
                ..Default::default()
            },
            block.hash,
        )
    }

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
    fn apply(pair: &mut LivePair, number: u64) {
        let mut accessed = BlockAccessedState::default();
        accessed.accounts.insert(
            TOUCHED,
            AccountData { nonce: number, balance: U256::from(number), code_hash: None },
        );
        pair.cache.on_block_executed(number, &accessed);
    }

    /// Admits a block the way the handler does when no canonical rebuild is available.
    fn admit(pair: &mut LivePair, block: &BlockContext) -> Option<super::ReadyParent> {
        match admit_block(&mut pair.readiness, block) {
            BlockAdmission::Admitted(ready) => ready,
            BlockAdmission::Rejected(_) => {
                admit_after_cold_reset(pair, block).expect("a cold reset always readmits")
            }
        }
    }

    /// Runs one block end to end the way the notification handler does.
    fn process(pair: &mut LivePair, number: u64) -> Option<super::ReadyParent> {
        let block = ctx(number);
        let ready = admit(pair, &block);
        apply(pair, number);
        observe_readiness(pair, &block);
        ready
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
        let mut pair = cold_pair();

        for number in 100..103 {
            process(&mut pair, number);
        }

        assert_eq!(pair.readiness.replay_depth(), 3);
        assert_eq!(pair.cache.current_block(), 102);
        assert!(pair.cache.contains_account(&TOUCHED), "contiguous blocks never trigger a reset");
    }

    #[test]
    fn warming_blocks_are_admitted_without_a_publication_token() {
        let mut pair = cold_pair();

        // The trie root is never authenticated here, so the cache stays Warming for good. It still
        // processes every block — that is what the benchmark measures — but hands the builder no
        // token, and the builder publishes nothing without one.
        for number in 100..110 {
            assert!(
                process(&mut pair, number).is_none(),
                "a warming cache must not authorize publication at block {number}"
            );
        }

        assert_eq!(*pair.readiness.state(), CacheReadiness::Warming { replay_depth: 10 });
    }

    #[test]
    fn a_gap_cold_resets_the_caches_and_restarts_the_replay_count() {
        let mut pair = cold_pair();

        for number in 100..102 {
            process(&mut pair, number);
        }

        // 105 is not the child of 101, so nothing the caches hold describes its parent.
        let gapped = ctx(105);
        assert!(
            admit(&mut pair, &gapped).is_none(),
            "the gapped block is applied against caches that were reset first, and publishes nothing"
        );

        assert_eq!(pair.cache.current_block(), 0);
        assert!(!pair.cache.contains_account(&TOUCHED), "stale entries cannot survive the gap");
        assert_eq!(pair.readiness.replay_depth(), 0);
        assert_eq!(*pair.readiness.state(), CacheReadiness::Applying { block_number: 105 });
    }

    #[test]
    fn an_incomplete_recovery_is_reported_as_such_rather_than_as_a_gap() {
        let mut pair = cold_pair();
        process(&mut pair, 100);

        // What the reorg handler leaves behind when the canonical rebuild fails: the pair still
        // describes the abandoned branch, and nothing may be applied against it.
        pair.readiness.begin_recovery(101);
        let block = ctx(101);
        assert!(matches!(
            admit_block(&mut pair.readiness, &block),
            BlockAdmission::Rejected(BlockedReason::RecoveryIncomplete { block_number: 101 })
        ));

        // The caller — not the admission check — decides to fall back to a cold warm-up, so the
        // failed recovery is visible before the reset happens.
        assert!(admit_after_cold_reset(&mut pair, &block)
            .expect("the fallback readmits")
            .is_none());
        assert_eq!(pair.cache.current_block(), 0, "the abandoned branch's values were dropped");
    }

    #[test]
    fn a_sibling_at_the_expected_height_is_treated_as_a_gap() {
        let mut pair = cold_pair();

        process(&mut pair, 100);

        // Right height, wrong branch: only the parent hash distinguishes the two.
        let sibling = BlockContext { parent_hash: B256::repeat_byte(0xfe), ..ctx(101) };
        admit(&mut pair, &sibling);

        assert_eq!(pair.cache.current_block(), 0, "the competing branch's parent forced a reset");
    }

    #[test]
    fn an_unapplied_block_pins_the_acknowledgement_below_it_forever() {
        let mut pair = cold_pair();

        process(&mut pair, 100);
        assert_eq!(pair.readiness.acknowledgeable_height(), Some((100, numbered(100, 0xbb))));

        // What the handler does when the parent state provider is unavailable for block 101.
        pair.readiness.abandon_block(101);

        // Later blocks still process, and the caches recover, but 101 was never applied. reth
        // prunes below the acknowledged height and never redelivers, so acknowledging 102
        // would lose it.
        for number in 102..106 {
            process(&mut pair, number);
        }

        assert_eq!(pair.cache.current_block(), 105, "processing continued");
        assert_eq!(
            pair.readiness.acknowledgeable_height(),
            Some((100, numbered(100, 0xbb))),
            "the acknowledgement must not step over the block that was skipped"
        );
        assert_eq!(pair.readiness.first_gap(), Some(101));
    }

    #[test]
    fn a_reorg_below_the_gap_releases_the_acknowledgement() {
        let mut pair = cold_pair();

        process(&mut pair, 100);
        pair.readiness.abandon_block(101);
        process(&mut pair, 102);
        assert_eq!(pair.readiness.acknowledgeable_height(), Some((100, numbered(100, 0xbb))));

        // The reorg drops everything from 101 up, including the block that was never applied.
        pair.readiness.begin_recovery(101);
        pair.trie_cache = PartialTrieNodeCache::new();
        pair.cache.reset();
        pair.readiness.reset();

        assert_eq!(pair.readiness.first_gap(), None);
        for number in 101..104 {
            process(&mut pair, number);
        }
        assert_eq!(pair.readiness.acknowledgeable_height(), Some((103, numbered(103, 0xbb))));
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

    // --- One-deep retained generation (K = 1) ---------------------------------------------------
    //
    // The value of this feature is the accept path, so it is tested against a real authenticated
    // trie rather than a stub: a snapshot restore is the only way to build a `PartialTrieNodeCache`
    // carrying a state root from outside the crate that owns it. Every other test here is a
    // rejection, because a false accept installs a generation that is not the canonical one.

    const SNAP_BLOCK: u64 = 21_000_000;
    const SNAP_HASH: B256 = B256::repeat_byte(0xb1);
    const SNAP_ADDRESS: Address = Address::repeat_byte(0x11);

    /// A cache holding one account, plus the multiproof and state root that authenticate it.
    fn warm_snapshot(config: &CacheConfig) -> (CacheSnapshotPackage, TrustedCheckpoint, B256) {
        let account = Account { nonce: 7, balance: U256::from(1_000u64), bytecode_hash: None };
        let hashed_address = keccak256(SNAP_ADDRESS);
        let mut builder = HashBuilder::default()
            .with_proof_retainer(ProofRetainer::from_iter([Nibbles::unpack(hashed_address)]));
        builder.add_leaf(
            Nibbles::unpack(hashed_address),
            &alloy_rlp::encode(account.into_trie_account(EMPTY_ROOT_HASH)),
        );
        let state_root = builder.root();
        let proof = MultiProof {
            account_subtree: builder.take_proof_nodes(),
            branch_node_masks: Default::default(),
            storages: Default::default(),
        };

        let mut cache = config.new_cache_at(SNAP_BLOCK);
        let mut accessed = BlockAccessedState::default();
        accessed.accounts.insert(
            SNAP_ADDRESS,
            AccountData { nonce: account.nonce, balance: account.balance, code_hash: None },
        );
        cache.on_block_executed(SNAP_BLOCK, &accessed);

        let checkpoint = TrustedCheckpoint {
            block_number: SNAP_BLOCK,
            block_hash: SNAP_HASH,
            state_root,
            cache_root: cache.cache_root(),
            cache_policy_id: config.cache_policy_id(),
        };
        let anchor = CacheAnchor {
            block_number: SNAP_BLOCK,
            block_hash: SNAP_HASH,
            cache_policy_id: config.cache_policy_id(),
            cache_root: cache.cache_root(),
        };
        (CacheSnapshotPackage::from_cache(&cache, anchor, &proof), checkpoint, state_root)
    }

    /// A pair sitting at `SNAP_BLOCK + 1` with the generation at `SNAP_BLOCK` retained, which is
    /// exactly the state a depth-1 reorg has to undo.
    fn pair_one_block_past_a_snapshot() -> (LivePair, CacheConfig, B256) {
        pair_one_block_past_a_snapshot_with(true)
    }

    /// The same pair, built with retention either on or off.
    ///
    /// `retain` false is the K = 1 memory control, which reaches the identical block by the
    /// identical route and differs only in whether the displaced generation is kept.
    fn pair_one_block_past_a_snapshot_with(retain: bool) -> (LivePair, CacheConfig, B256) {
        let config = CacheConfig::default();
        let (package, checkpoint, state_root) = warm_snapshot(&config);
        let retained = crate::bootstrap_io::restore_snapshot(package.clone(), &checkpoint, &config)
            .expect("an honest snapshot restores");
        let current = crate::bootstrap_io::restore_snapshot(package, &checkpoint, &config)
            .expect("an honest snapshot restores");

        let mut pair = LivePair::new(CoordinatedPair {
            cache: current.cache,
            trie_cache: current.trie_cache,
            readiness: current.readiness,
            previous_generation: None,
            accepted_head: None,
        });
        // Advance the flat cache one block, which is what leaves the undo record the rollback
        // consumes, and retain the generation that block displaced.
        apply(&mut pair, SNAP_BLOCK + 1);
        // The tracker has to see the block too, or the pair would claim a depth-1 undo is
        // available with nothing replayed to give back. The block is described as leaving the
        // state root where it was, which is the only root this fixture's trie can authenticate
        // against — the tracker's arithmetic is what is under test here, not the trie's.
        let applied = BlockContext {
            number: SNAP_BLOCK + 1,
            hash: numbered(SNAP_BLOCK + 1, 0xbb),
            parent_hash: SNAP_HASH,
            state_root,
        };
        admit(&mut pair, &applied);
        finish_committed_transition(
            &mut pair,
            Some(retained.trie_cache),
            &applied,
            sealed(&applied),
            retain,
        );
        (pair, config, state_root)
    }

    #[test]
    fn a_retained_generation_undoes_exactly_one_block() {
        let (mut pair, config, state_root) = pair_one_block_past_a_snapshot();
        let expected_root = {
            let (package, checkpoint, _) = warm_snapshot(&config);
            crate::bootstrap_io::restore_snapshot(package, &checkpoint, &config)
                .expect("an honest snapshot restores")
                .cache
                .cache_root()
        };
        assert_ne!(pair.cache.cache_root(), expected_root, "the applied block must have moved it");

        let ready = pair
            .restore_retained_generation(SNAP_HASH, state_root, config.cache_policy_id())
            .expect("the retained generation is the block being asked for");

        assert_eq!(ready.anchor.block_number, SNAP_BLOCK);
        assert_eq!(ready.anchor.block_hash, SNAP_HASH);
        assert_eq!(pair.cache.current_block(), SNAP_BLOCK, "the flat cache rolled back one block");
        assert_eq!(pair.cache.cache_root(), expected_root, "and to the exact prior generation");
        assert_eq!(pair.trie_cache.state_root(), Some(state_root));
        assert!(matches!(pair.readiness.state(), CacheReadiness::Ready(_)));
        assert!(
            pair.previous_generation.is_none(),
            "the retention is consumed: what it described is now the live generation"
        );
    }

    #[test]
    fn the_memory_control_keeps_nothing_and_says_so() {
        let (control, _, _) = pair_one_block_past_a_snapshot_with(false);
        let (retaining, _, _) = pair_one_block_past_a_snapshot_with(true);

        // Both pairs took the same route to the same block, so the caches themselves must agree.
        // A control that also changed the state would not isolate the memory it saves.
        assert_eq!(control.fingerprint(), retaining.fingerprint());
        assert!(control.previous_generation.is_none());
        assert!(retaining.previous_generation.is_some());

        let control_bytes = control.retained_generation_bytes(false);
        assert!(!control_bytes.enabled);
        assert!(!control_bytes.present);
        assert_eq!(control_bytes.exclusive_bytes, 0);

        let retained_bytes = retaining.retained_generation_bytes(true);
        assert!(retained_bytes.enabled && retained_bytes.present);
        assert!(retained_bytes.exclusive_bytes > 0, "a held generation costs something");
        assert!(retained_bytes.exclusive_bytes <= retained_bytes.total_bytes);
    }

    #[test]
    fn the_memory_control_gives_up_the_depth_one_undo_it_is_not_paying_for() {
        let (mut pair, config, state_root) = pair_one_block_past_a_snapshot_with(false);

        // The recovery the control declines is the one the production arm takes for free; a run
        // with this flag set must not be read for reorg timings, and this is why.
        assert!(
            pair.restore_retained_generation(SNAP_HASH, state_root, config.cache_policy_id())
                .is_none(),
            "with nothing retained there is nothing to undo, so recovery must fall through"
        );
        assert_eq!(pair.cache.current_block(), SNAP_BLOCK + 1, "and nothing was mutated");
    }

    #[test]
    fn a_retained_generation_for_another_branch_is_refused() {
        let (mut pair, config, state_root) = pair_one_block_past_a_snapshot();
        let other_branch = B256::repeat_byte(0xee);

        assert!(
            pair.restore_retained_generation(other_branch, state_root, config.cache_policy_id())
                .is_none(),
            "a retention tagged with a different hash must never be installed"
        );
        assert_eq!(pair.cache.current_block(), SNAP_BLOCK + 1, "and nothing may be mutated");
        assert!(pair.previous_generation.is_none(), "the rejected retention is dropped");
    }

    #[test]
    fn a_retained_generation_is_refused_against_a_forged_state_root() {
        let (mut pair, config, _) = pair_one_block_past_a_snapshot();

        assert!(
            pair.restore_retained_generation(
                SNAP_HASH,
                B256::repeat_byte(0x77),
                config.cache_policy_id()
            )
            .is_none(),
            "the retained trie must match the canonical header's state root"
        );
        assert_eq!(pair.cache.current_block(), SNAP_BLOCK + 1);
    }

    #[test]
    fn a_reorg_deeper_than_one_block_is_refused() {
        let (mut pair, config, state_root) = pair_one_block_past_a_snapshot();
        // A second block puts the pair two ahead of the retention, which K = 1 does not reach.
        apply(&mut pair, SNAP_BLOCK + 2);

        assert!(
            pair.restore_retained_generation(SNAP_HASH, state_root, config.cache_policy_id())
                .is_none(),
            "K = 1 covers depth 1 only; anything deeper must fall back to a rebuild"
        );
        assert_eq!(pair.cache.current_block(), SNAP_BLOCK + 2);
    }

    #[test]
    fn a_transition_that_did_not_commit_leaves_nothing_retained() {
        let (mut pair, _, _) = pair_one_block_past_a_snapshot();
        assert!(pair.previous_generation.is_some());

        // `None` is what the builder reports when the transition rolled back, and the old
        // retention describes a generation two blocks back that K = 1 does not promise.
        pair.retain_generation(None, SNAP_HASH, SNAP_BLOCK + 1, sealed(&ctx(SNAP_BLOCK + 2)), true);

        assert!(pair.previous_generation.is_none());
    }

    #[test]
    fn a_still_warming_pair_refuses_the_undo_and_leaves_the_caches_alone() {
        // What a reorg finds on a node that never got a canonical rebuild: the pair is sound and
        // advancing, but it has replayed nowhere near a window and has no `Ready` to return to.
        // Promoting it anyway would open the sidecar publication gate on an under-warmed cache.
        let config = CacheConfig::default();
        let (package, checkpoint, state_root) = warm_snapshot(&config);
        let retained = crate::bootstrap_io::restore_snapshot(package, &checkpoint, &config)
            .expect("an honest snapshot restores");
        let mut pair = cold_pair();
        // Warm from cold rather than from the snapshot, so the window was never asserted.
        for number in (SNAP_BLOCK - 2)..=SNAP_BLOCK {
            process(&mut pair, number);
        }
        assert!(!pair.readiness.window_filled(), "three blocks is not a window");
        apply(&mut pair, SNAP_BLOCK + 1);
        pair.retain_generation(
            Some(retained.trie_cache),
            SNAP_HASH,
            SNAP_BLOCK,
            sealed(&ctx(SNAP_BLOCK + 1)),
            true,
        );

        assert!(
            pair.restore_retained_generation(SNAP_HASH, state_root, config.cache_policy_id())
                .is_none(),
            "a warming pair has no Ready to be restored to"
        );
        assert_eq!(
            pair.cache.current_block(),
            SNAP_BLOCK + 1,
            "the refusal is taken before anything is mutated, so a rebuild starts from a clean pair"
        );
        assert!(!matches!(pair.readiness.state(), CacheReadiness::Ready(_)));
    }

    /// A canonical chain the test controls, answering the one question recovery asks of a node.
    enum FakeChain {
        /// The recovery target is canonical and has this state root.
        Canonical(B256, B256),
        /// No canonical header for the target — a rejection, not an error.
        Unknown,
        /// The lookup itself failed, which must be a fallback rather than a false recovery.
        Unavailable,
    }

    impl CanonicalStateRoots for FakeChain {
        fn state_root_of(&self, hash: B256) -> ProviderResult<Option<B256>> {
            match self {
                Self::Canonical(known, state_root) if *known == hash => Ok(Some(*state_root)),
                Self::Canonical(..) | Self::Unknown => Ok(None),
                Self::Unavailable => {
                    Err(reth_provider::ProviderError::TrieWitnessError("injected".to_string()))
                }
            }
        }
    }

    /// Advances the flat cache by a block that touches `address`, so two branches can be made to
    /// leave different residue behind.
    fn apply_touching(pair: &mut LivePair, number: u64, address: Address) {
        let mut accessed = BlockAccessedState::default();
        accessed.accounts.insert(
            address,
            AccountData { nonce: number, balance: U256::from(number), code_hash: None },
        );
        pair.cache.on_block_executed(number, &accessed);
    }

    /// The memo, the leaf digest index, and the value maps must all be telling the same story.
    ///
    /// [`CoordinatedPair::fingerprint`] reads `cache_root()`, which answers from the memo — and a
    /// rollback restores that memo from the undo record rather than recomputing it. Two pairs can
    /// therefore be fingerprint-equal while the digest index under one of them has drifted, and the
    /// drift would first surface a block later, on the next root the memo does not answer. The slow
    /// reference reads neither memo nor index, so it is what closes that gap.
    fn assert_cache_root_is_independently_reproducible(pair: &LivePair, at: &str) {
        assert_eq!(
            pair.cache.cache_root(),
            pair.cache.compute_cache_root_reference(),
            "cache root disagrees with a from-scratch recomputation, {at}"
        );
    }

    /// The pair a depth-1 reorg finds, plus a reference pair that never saw the abandoned block.
    ///
    /// The abandoned block touches `ABANDONED` and nothing else does, so any residue the rollback
    /// fails to remove shows up as a `cache_root` divergence rather than having to be looked for.
    fn pair_and_reference_before_a_reorg() -> (LivePair, LivePair, CacheConfig, B256) {
        const ABANDONED: Address = Address::repeat_byte(0xa1);
        let config = CacheConfig::default();
        let (package, checkpoint, state_root) = warm_snapshot(&config);
        let retained = crate::bootstrap_io::restore_snapshot(package.clone(), &checkpoint, &config)
            .expect("an honest snapshot restores");
        let current = crate::bootstrap_io::restore_snapshot(package.clone(), &checkpoint, &config)
            .expect("an honest snapshot restores");
        let reference = crate::bootstrap_io::restore_snapshot(package, &checkpoint, &config)
            .expect("an honest snapshot restores");

        let mut pair = LivePair::new(CoordinatedPair {
            cache: current.cache,
            trie_cache: current.trie_cache,
            readiness: current.readiness,
            previous_generation: None,
            accepted_head: None,
        });
        let reference = LivePair::new(CoordinatedPair {
            cache: reference.cache,
            trie_cache: reference.trie_cache,
            readiness: reference.readiness,
            previous_generation: None,
            accepted_head: None,
        });

        apply_touching(&mut pair, SNAP_BLOCK + 1, ABANDONED);
        let applied = BlockContext {
            number: SNAP_BLOCK + 1,
            hash: numbered(SNAP_BLOCK + 1, 0xaa),
            parent_hash: SNAP_HASH,
            state_root,
        };
        admit(&mut pair, &applied);
        finish_committed_transition(
            &mut pair,
            Some(retained.trie_cache),
            &applied,
            sealed(&applied),
            true,
        );

        (pair, reference, config, state_root)
    }

    #[test]
    fn a_depth_one_undo_returns_the_accepted_head_to_the_parent() {
        let (mut pair, _reference, config, state_root) = pair_and_reference_before_a_reorg();
        let chain = FakeChain::Canonical(SNAP_HASH, state_root);
        assert_eq!(
            pair.accepted_parent().map(|header| header.number),
            Some(SNAP_BLOCK + 1),
            "the applied block must have advanced the head, or the undo proves nothing"
        );

        inject_recovery(&mut pair, &chain, SNAP_BLOCK + 1, SNAP_HASH, config.cache_policy_id())
            .expect("the retained generation is exactly the reorg target");

        // Restored from the retained generation, which predates any accepted header here because
        // the pair came from a snapshot. Absence is the correct answer: the replacement block must
        // not be checked against the block the reorg just discarded, and this pair cannot name a
        // parent it never applied.
        assert_eq!(
            pair.accepted_head, None,
            "the undo must not leave the abandoned block standing as the parent"
        );
        assert_eq!(pair.accepted_parent(), None);
    }

    #[test]
    fn an_accepted_head_that_outlived_its_caches_is_not_offered_as_a_parent() {
        // A rebuild or snapshot restore replaces the caches wholesale. Whatever header the pair
        // was carrying then describes a generation it is no longer at, and admitting a child
        // against it would check that child against a parent this pair never had.
        let (mut pair, ..) = pair_and_reference_before_a_reorg();
        assert!(pair.accepted_parent().is_some(), "precondition: the pair has a usable parent");

        pair.cache.reset();

        assert!(
            pair.accepted_head.is_some(),
            "the field is deliberately left alone; the guard is what refuses it"
        );
        assert_eq!(
            pair.accepted_parent(),
            None,
            "a header whose number no longer matches the cache must read as absent"
        );
    }

    #[test]
    fn a_sibling_header_at_the_same_height_is_not_offered_as_a_parent() {
        // The case a height check cannot see, and the one a reorg actually produces: the winning
        // sibling has the number the abandoned block had. A pair holding the abandoned header over
        // the winner's caches would measure parent consensus against one branch while executing
        // against the other, and every number would line up.
        let (mut pair, ..) = pair_and_reference_before_a_reorg();
        let accepted = pair.accepted_parent().expect("precondition: usable parent").clone();

        // Same number, same everything the height guard looks at, different block.
        let sibling = SealedHeader::new(accepted.clone_header(), B256::repeat_byte(0xfe));
        assert_eq!(sibling.number, accepted.number, "the fixture must keep the height identical");
        pair.accepted_head = Some(sibling);

        assert_eq!(
            pair.accepted_parent(),
            None,
            "a header at the right height but the wrong hash must read as absent"
        );
    }

    #[test]
    fn a_canonical_rebuild_clears_the_accepted_head_it_no_longer_descends_from() {
        // Not the guard this time — the state. `rebuild_pair_at` installs a different branch's
        // caches, so leaving the header behind would be a field that lies until something reads
        // it. There is no provider in a unit test, so this drives the same clearing the rebuild
        // performs and pins that the pair reports absence afterwards.
        let (mut pair, ..) = pair_and_reference_before_a_reorg();
        assert!(pair.accepted_parent().is_some(), "precondition: the pair has a usable parent");

        pair.forget_retained_generation();
        pair.accepted_head = None;

        assert_eq!(pair.accepted_parent(), None);
        assert_eq!(pair.lifecycle_fingerprint().accepted_head, None);
    }

    #[test]
    fn a_cold_reset_forgets_the_accepted_head() {
        let (mut pair, ..) = pair_and_reference_before_a_reorg();
        assert!(pair.accepted_parent().is_some(), "precondition: the pair has a usable parent");

        pair.cold_reset();

        assert_eq!(pair.accepted_head, None);
        assert_eq!(pair.lifecycle_fingerprint().accepted_head, None);
        assert_eq!(pair.lifecycle_fingerprint().retained_generation, None);
    }

    #[test]
    fn the_accepted_head_advances_even_when_retention_is_off() {
        // The K = 1 memory control turns off reorg retention, not admission: the next block still
        // has to be checked against this one. A control arm that also stopped tracking the parent
        // would be measuring two changes at once, and the second one is a correctness change.
        let (mut pair, _reference, ..) = pair_and_reference_before_a_reorg();
        let next = BlockContext {
            number: SNAP_BLOCK + 2,
            hash: numbered(SNAP_BLOCK + 2, 0xdd),
            parent_hash: numbered(SNAP_BLOCK + 1, 0xaa),
            state_root: numbered(SNAP_BLOCK + 2, 0x55),
        };
        apply_touching(&mut pair, next.number, Address::repeat_byte(0xa2));
        admit(&mut pair, &next);

        finish_committed_transition(&mut pair, None, &next, sealed(&next), false);

        assert!(pair.previous_generation.is_none(), "retention is off in this arm");
        // The field, not `accepted_parent()`. This fixture advances the pair with a synthesized
        // state root the trie cache cannot reproduce, so the pair is no longer `Ready` and the
        // guard correctly declines to vouch for the header. What is under test is that the head
        // advanced at all: the K = 1 control turns off reorg retention, not admission, and a
        // control arm that also stopped tracking the parent would be a correctness change wearing
        // a memory measurement's clothes.
        assert_eq!(
            pair.accepted_head.as_ref().map(SealedHeader::hash),
            Some(next.hash),
            "the accepted head must advance regardless of retention"
        );
    }

    #[test]
    fn a_recovered_pair_is_indistinguishable_from_one_that_never_saw_the_block() {
        let (mut pair, mut reference, config, state_root) = pair_and_reference_before_a_reorg();
        let chain = FakeChain::Canonical(SNAP_HASH, state_root);

        assert_ne!(
            pair.fingerprint(),
            reference.fingerprint(),
            "the abandoned block must have moved the pair, or this proves nothing"
        );

        inject_recovery(&mut pair, &chain, SNAP_BLOCK + 1, SNAP_HASH, config.cache_policy_id())
            .expect("the retained generation is exactly the reorg target");

        assert_eq!(
            pair.fingerprint(),
            reference.fingerprint(),
            "a recovered pair must equal one that never saw the abandoned block, on every field"
        );
        assert_cache_root_is_independently_reproducible(&pair, "after recovery");

        // Equality at the anchor is necessary but not sufficient: `last_accessed_block` decides
        // what the *next* eviction does, and no state proof attests to it. Driving both pairs
        // through the winning block is what would expose residue the rollback left behind.
        const WINNER: Address = Address::repeat_byte(0xb2);
        apply_touching(&mut pair, SNAP_BLOCK + 1, WINNER);
        apply_touching(&mut reference, SNAP_BLOCK + 1, WINNER);
        assert_eq!(
            pair.fingerprint(),
            reference.fingerprint(),
            "and it must still agree after the winning branch is applied to both"
        );
        // The winning block invalidated the memo, so this root was recomputed from the index
        // rather than restored — which is the point at which a rollback that left the index
        // stale would produce a wrong anchor and be rejected by peers.
        assert_cache_root_is_independently_reproducible(&pair, "after the winning branch");
    }

    #[test]
    fn a_pure_revert_returns_the_pair_to_the_new_tip_with_no_branch_to_follow() {
        let (mut pair, reference, config, state_root) = pair_and_reference_before_a_reorg();
        let chain = FakeChain::Canonical(SNAP_HASH, state_root);

        // `ChainReverted` carries only the dropped branch: the pair recovers at the new tip and
        // then simply stops, with no replacement blocks to apply.
        let ready =
            inject_recovery(&mut pair, &chain, SNAP_BLOCK + 1, SNAP_HASH, config.cache_policy_id())
                .expect(
                    "a revert to the retained generation takes the same fast path a reorg does",
                );

        assert_eq!(ready.anchor.block_number, SNAP_BLOCK);
        assert_eq!(ready.anchor.block_hash, SNAP_HASH);
        assert_eq!(pair.fingerprint(), reference.fingerprint());
        assert_cache_root_is_independently_reproducible(&pair, "after a pure revert");
        assert!(
            matches!(pair.readiness.state(), CacheReadiness::Ready(_)),
            "a reverted pair is Ready at the new tip, not left Recovering"
        );
    }

    #[test]
    fn an_unknown_recovery_target_leaves_a_verifier_with_nothing_to_fall_back_to() {
        let (mut pair, _, config, _) = pair_and_reference_before_a_reorg();

        assert!(
            inject_recovery(
                &mut pair,
                &FakeChain::Unknown,
                SNAP_BLOCK + 1,
                SNAP_HASH,
                config.cache_policy_id()
            )
            .is_none(),
            "an unproven target must never install the retained generation"
        );
        // A full node rebuilds from here. A stateless verifier has no database to rebuild from, so
        // this is where its recovery ends — recorded as the shape of the gap, not as a pass.
        assert!(matches!(pair.readiness.state(), CacheReadiness::Recovering { .. }));
        assert_eq!(pair.cache.current_block(), SNAP_BLOCK + 1, "and nothing was mutated");
        let next = BlockContext {
            number: SNAP_BLOCK + 2,
            hash: numbered(SNAP_BLOCK + 2, 0xcc),
            parent_hash: numbered(SNAP_BLOCK + 1, 0xaa),
            state_root: B256::repeat_byte(0xdd),
        };
        assert!(matches!(
            admit_block(&mut pair.readiness, &next),
            BlockAdmission::Rejected(BlockedReason::RecoveryIncomplete { block_number })
                if block_number == SNAP_BLOCK + 2
        ));
    }

    #[test]
    fn a_failed_header_lookup_is_a_fallback_rather_than_a_false_recovery() {
        let (mut pair, _, config, _) = pair_and_reference_before_a_reorg();

        assert!(
            inject_recovery(
                &mut pair,
                &FakeChain::Unavailable,
                SNAP_BLOCK + 1,
                SNAP_HASH,
                config.cache_policy_id()
            )
            .is_none(),
            "a lookup that errored proves nothing, so it must be treated as a rejection"
        );
        assert!(matches!(pair.readiness.state(), CacheReadiness::Recovering { .. }));
    }

    #[test]
    fn a_cold_reset_forgets_the_retained_generation() {
        let (mut pair, _, _) = pair_one_block_past_a_snapshot();
        assert!(pair.previous_generation.is_some());

        admit_after_cold_reset(&mut pair, &ctx(SNAP_BLOCK + 2))
            .expect("a cold reset always readmits");

        assert!(
            pair.previous_generation.is_none(),
            "the reset pair does not descend from the retained generation"
        );
    }
}
