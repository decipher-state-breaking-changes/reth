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
pub mod payload_tap;
pub mod policy_dataset_capture;
pub mod rebuild;
pub mod recorder;

mod benchmark;
mod producer_events;
mod sidecar_create;
mod sidecar_io;
mod sidecar_reexec;
mod sidecar_verify;

use alloy_eips::BlockNumHash;
use alloy_primitives::B256;
use alloy_rlp::Encodable;
use futures::TryStreamExt;
pub use partial_stateless::CacheConfig;
use partial_stateless::{
    persistence::{load_from_file, save_to_file, CacheState},
    readiness::{BlockContext, BlockedReason, CacheObservation, CacheReadiness, ReadyParent},
    CacheAnchor, CacheTrieRepr, PartialStatelessSidecar, PartialTrieNodeCache,
};
use partial_stateless_stream::{
    BlockRef as StreamBlockRef, CommitInput, CommitOracle, EndKind, RecordedVerdict, Reorg,
    ResetReason,
};
use partial_stateless_validator::{
    admit_block, block_context, inject_recovery, BlockAdmission, CanonicalStateRoots,
    CoordinatedPair, RetainedGenerationBytes, ValidatorRules,
};
use reth_ethereum::{
    chainspec::EthChainSpec,
    exex::{ExExContext, ExExEvent, ExExNotification},
    node::api::{FullNodeComponents, NodeTypes},
    provider::{Chain, StateProviderFactory},
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
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc,
    },
    time::{Duration, Instant},
};
use tracing::{error, info, warn};

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

/// `PS_TRIE_REPR`, refused at startup on anything but a known representation name.
///
/// Strict like the windows rather than defaulting like a flag: the value labels every trie the
/// run constructs, and an A/B arm whose mistyped variable silently fell back to the default
/// would be a measurement filed under a representation it never ran.
fn trie_repr_from_env() -> eyre::Result<CacheTrieRepr> {
    match std::env::var("PS_TRIE_REPR") {
        Err(_) => Ok(CacheTrieRepr::default()),
        Ok(raw) => raw.parse::<CacheTrieRepr>().map_err(|err| eyre::eyre!("PS_TRIE_REPR: {err}")),
    }
}

fn env_u32(name: &str, default: u32) -> u32 {
    let Ok(value) = std::env::var(name) else { return default };
    // A bare `PS_..=1`-style flag should still mean "on"; a count is the more useful spelling.
    match value.as_str() {
        "true" | "TRUE" | "yes" | "on" => 1,
        other => other.parse().unwrap_or(default),
    }
}

/// `PS_STREAM_EXPORT_MAX_WORKERS`, refused at startup on anything but an integer >= 1.
///
/// Hard-erroring where the older `env_u32` silently defaults is deliberate: a run configured
/// with a bound this build cannot parse would otherwise report a cap it is not enforcing.
fn env_export_max_workers() -> eyre::Result<usize> {
    match std::env::var("PS_STREAM_EXPORT_MAX_WORKERS") {
        Err(_) => Ok(4),
        Ok(raw) => match raw.trim().parse::<usize>() {
            Ok(value) if value >= 1 => Ok(value),
            _ => Err(eyre::eyre!(
                "PS_STREAM_EXPORT_MAX_WORKERS must be an integer >= 1, not `{raw}`"
            )),
        },
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
    /// Where the policy replay dataset is captured, when that opt-in capture is on.
    ///
    /// `None` in every ordinary run, and that `None` is the whole contract: no full witness is
    /// built, no payload is cloned for a dataset, and no writer exists. Resolved once here rather
    /// than re-read per block, so a variable changed under a running node cannot turn a measured
    /// run into a capturing one halfway through.
    pub policy_dataset: Option<policy_dataset_capture::PolicyDatasetCaptureConfig>,
    /// Whether to compute the full-witness baseline for the reduction ratio.
    pub compute_baseline: bool,
    /// Whether to sample process CPU time and page faults around witness construction.
    pub resource_metrics: bool,
    /// Whether to validate retained trie paths and log trie shape every block.
    pub trie_cache_diagnostics: bool,
    /// Whether Ready-cache sidecars carry the receiver-aware trimmed (v3) witness.
    pub witness_v3: bool,
    /// The sparse-trie representation every trie cache this run constructs is built on.
    ///
    /// Not protocol: the cross-representation oracle showed the two representations produce
    /// identical observables (roots, anchors, witness bytes, fragments), so the choice never
    /// reaches the wire or the policy identifier. It is a measurement label all the same — a run
    /// is meaningless as an A/B arm unless the label names what actually ran — so it is resolved
    /// once here, logged at startup, and preserved across cold resets.
    pub trie_repr: CacheTrieRepr,
    /// Whether the builder preflights each sidecar before publishing it.
    pub run_sidecar_preflight: bool,
    /// Where paired Partial/Weak validation records go.
    pub validation_bench_output: Option<PathBuf>,
    /// Where per-block builder records go.
    pub builder_bench_output: Option<PathBuf>,
    /// Benchmark control that recreates the old unconditional parent-cache clone, so its cost can
    /// be priced against the current conditional one.
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
    /// Fresh export attempts allowed after the first one fails or its buffer overflows.
    ///
    /// Each retry chooses a new H at the next `Ready`; when they run out the stream closes with
    /// `End(ExportFailure)`. A branch change does not consume one: fencing an attempt because the
    /// chain moved is not the attempt failing.
    pub stream_export_retries: u32,
    /// Live export workers allowed at once, abandoned ones included.
    ///
    /// An abandoned worker cannot be cancelled and holds an MDBX read transaction for its whole
    /// multiproof — minutes — so a reorg storm that fences attempt after attempt would otherwise
    /// accumulate readers without bound. At the cap a fresh attempt is not spawned;
    /// `export_pending` stays armed and the next Ready retries once a worker has drained.
    pub stream_export_max_workers: usize,
    /// Whether a branch change re-checkpoints the open stream at the block it recovered to.
    pub reorg_checkpoint: ReorgCheckpointPolicy,
    /// The `PS_STREAM_FSYNC` power-loss durability profile, applied to frames, the ack's producer
    /// counterparts, and snapshot packages. Off by default: the default/benchmark profile keeps
    /// process-restart durability only, and the price of the power-loss profile is measured
    /// before it is ever made anyone's default.
    pub stream_fsync: bool,
    /// Whether the Engine publishes the payloads it validated and this ExEx takes them.
    ///
    /// Not read from a variable of its own: the producer's gate is `PS_ENGINE_PAYLOAD`, and a
    /// consumer with a second switch could be armed against a node that publishes nothing, which
    /// would report every block as a reconstruction and look like a delivery failure.
    pub payload_tap: bool,
}

/// Whether a branch change re-checkpoints the stream at the block the pair recovered to.
///
/// The default is `Always`, and the reason is that the producer cannot know what its consumer can
/// do for itself. A follower holding a retained generation undoes a depth-1 reorg on its own; one
/// that just restored, just restarted, or met a deeper reorg cannot, and there is no back-channel
/// to ask which it is. A checkpoint at the common ancestor is what makes both cases recoverable,
/// and it is the only route for anything past depth 1. The price is a snapshot's worth of spool
/// per branch change, which the spool bound already governs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReorgCheckpointPolicy {
    /// Re-checkpoint after every reorg, revert, or recovery discontinuity.
    Always,
    /// Never re-checkpoint an open stream. The experiment control: it leaves a consumer that
    /// cannot self-recover waiting, which is what the default exists to avoid.
    Never,
}

impl ReorgCheckpointPolicy {
    fn from_env() -> eyre::Result<Self> {
        Self::parse(std::env::var("PS_STREAM_REORG_CHECKPOINT").ok().as_deref())
    }

    fn parse(raw: Option<&str>) -> eyre::Result<Self> {
        match raw {
            None | Some("always") => Ok(Self::Always),
            Some("never") => Ok(Self::Never),
            // Refused at startup rather than defaulted: a run configured with a value this build
            // does not know would report a policy it is not running.
            Some(other) => Err(eyre::eyre!(
                "PS_STREAM_REORG_CHECKPOINT must be `always` or `never`, not `{other}`"
            )),
        }
    }
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
        if bootstrap_self_test_blocks > 0 && std::env::var_os("PS_STREAM_DIR").is_some() {
            // The self-test compares fingerprints at the export block, which requires the export
            // to run synchronously with the pair frozen at H. A recording run exports off-task
            // precisely so the pair keeps moving — the two are structurally incompatible, and the
            // live follower is the recording run's bootstrap evidence.
            return Err(eyre::eyre!(
                "PS_BOOTSTRAP_SELF_TEST cannot be combined with PS_STREAM_DIR: a recording run \
                 exports off-task while the pair keeps advancing, and the in-process self-test \
                 needs the pair frozen at the export block. Validate the stream with ps-replay \
                 instead."
            ))
        }
        let trie_repr = trie_repr_from_env()?;
        if trie_repr != CacheTrieRepr::default() &&
            (env_flag("PS_BOOTSTRAP_IMPORT") ||
                env_flag("PS_CANONICAL_REBUILD") ||
                bootstrap_self_test_blocks > 0)
        {
            // Snapshot restore and canonical rebuild construct their tries inside shared
            // bootstrap code that builds the default representation; letting them run under a
            // non-default PS_TRIE_REPR would produce a pair whose actual representation
            // contradicts the run's label. Refused rather than converted: there is no cheap
            // in-place conversion, and a mislabelled measurement is worse than no run.
            return Err(eyre::eyre!(
                "PS_TRIE_REPR={} applies to cold-started pairs only; snapshot restore, canonical \
                 rebuild, and the bootstrap self-test build the default representation. Unset \
                 PS_BOOTSTRAP_IMPORT / PS_CANONICAL_REBUILD / PS_BOOTSTRAP_SELF_TEST, or run on \
                 the default representation.",
                trie_repr.label()
            ))
        }
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
            witness_v3: env_flag("PS_WITNESS_V3"),
            trie_repr,
            run_sidecar_preflight: sidecar_role.runs_preflight(),
            validation_bench_output: validation_bench.then(|| {
                std::env::var_os("PS_BENCH_OUTPUT")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| sidecar_dir.join("validation_bench.jsonl"))
            }),
            builder_bench_output: std::env::var_os("PS_BUILDER_BENCH_OUTPUT").map(PathBuf::from),
            policy_dataset: policy_dataset_capture::PolicyDatasetCaptureConfig::from_env()?,
            force_previous_cache_snapshot: env_flag("PS_FORCE_PREVIOUS_CACHE_SNAPSHOT"),
            retain_generation: env_flag_enabled_by_default("PS_RETAIN_GENERATION"),
            parallel_initial_proof: env_flag("PS_PARALLEL_INITIAL_PROOF"),
            validation_bench,
            reexec_limits: SidecarReexecLimits::default(),
            canonical_rebuild: env_flag("PS_CANONICAL_REBUILD"),
            bootstrap_dir: std::env::var_os("PS_BOOTSTRAP_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| sidecar_dir.join("bootstrap")),
            // Both the self-test and the recorder need a package to restore from, so both imply
            // the export. A recording run without one would write a manifest and then nothing,
            // because a commit before the checkpoint is a commit no consumer could replay.
            bootstrap_export: env_flag("PS_BOOTSTRAP_EXPORT") ||
                bootstrap_self_test_blocks > 0 ||
                std::env::var_os("PS_STREAM_DIR").is_some(),
            bootstrap_import: env_flag("PS_BOOTSTRAP_IMPORT"),
            bootstrap_self_test_blocks,
            stream_export_retries: env_u32("PS_STREAM_EXPORT_RETRIES", 1),
            stream_export_max_workers: env_export_max_workers()?,
            stream_fsync: recorder::stream_fsync_from_env()?,
            reorg_checkpoint: ReorgCheckpointPolicy::from_env()?,
            payload_tap: payload_tap::tap_enabled(),
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
            reorg_checkpoint = ?self.reorg_checkpoint,
            stream_fsync = self.stream_fsync,
            trie_repr = self.trie_repr.label(),
            "Partial Stateless ExEx started — monitoring cache state per block"
        );
        if self.trie_repr != CacheTrieRepr::default() {
            info!(
                target: "partial_stateless",
                trie_repr = self.trie_repr.label(),
                "Non-default cache trie representation ENABLED (PS_TRIE_REPR) — every trie cache \
                 this run constructs uses it; observables are representation-independent"
            );
        }
        if let Some(dir) = &self.capture_dir {
            info!(
                target: "partial_stateless",
                dir = %dir.display(),
                "Accessed-state fixture capture ENABLED (PS_CAPTURE_DIR) — run until ~300 blocks captured"
            );
        }
        if let Some(dataset) = &self.policy_dataset {
            warn!(
                target: "partial_stateless",
                dir = %dataset.dir.display(),
                max_blocks = dataset.max_blocks,
                "Policy replay dataset capture ENABLED (PS_POLICY_DATASET_CAPTURE_DIR) — a full \
                 witness is built, validated, and written per block. This run is NOT measurement \
                 eligible and its manifest says so"
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
        if self.witness_v3 {
            info!(
                target: "partial_stateless",
                "Trimmed witness ENABLED (PS_WITNESS_V3) — Ready-cache sidecars carry receiver-aware fragments; warming and full-witness sidecars stay self-contained"
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
                "Forced previous-cache snapshot ENABLED as a benchmark control"
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
        if self.payload_tap {
            info!(
                target: "partial_stateless",
                "Engine payload tap ENABLED (PS_ENGINE_PAYLOAD) — canonical blocks carry the \
                 payload the consensus layer sent, or a derived one labelled as such"
            );
        }
    }

    fn builder_options<'a>(
        &'a self,
        parallel_initial_proof: Option<&'a ParallelInitialProofFn<'a>>,
        ready_parent: Option<&'a ReadyParent>,
        retain_sidecar: bool,
        retained_generation: RetainedGenerationBytes,
        capture_policy_dataset: bool,
    ) -> BuilderOptions<'a> {
        BuilderOptions {
            capture_dir: self.capture_dir.as_deref(),
            capture_policy_dataset,
            sidecar_dir: &self.sidecar_dir,
            compute_baseline: self.compute_baseline,
            resource_metrics: self.resource_metrics,
            trie_cache_diagnostics: self.trie_cache_diagnostics,
            witness_v3: self.witness_v3,
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
    /// The off-task export, when one is running or has finished.
    job: ExportJob,
    /// Fresh attempts allowed after a failure. Zero means the next failure closes the stream.
    retries_left: u32,
    /// Monotonic attempt counter, naming each attempt's own export directory.
    attempt: u32,
    /// Cause of the export `export_pending` is armed for. A branch change writes commits through
    /// and publishes its checkpoint at the tail; every other cause buffers behind the checkpoint,
    /// because no consumer can cross a discontinuity from commits alone.
    pending_cause: RecheckpointCause,
    /// Cause of the most recently spawned attempt, read when that attempt fails.
    attempt_cause: RecheckpointCause,
    /// Monotonic id allocator for causes. Zero is the initial export; every lifecycle event that
    /// arms or re-arms an export allocates the next id, and the id rides every producer event
    /// the cause goes on to generate — detection, attempts, fencing, publication, first commit —
    /// so overlapping retries and back-to-back reorgs stay attributable offline.
    cause_counter: u64,
    /// The id of the cause `pending_cause` belongs to.
    pending_cause_id: u64,
    /// The id of the cause the in-flight (or last spawned) attempt belongs to.
    attempt_cause_id: u64,
    /// Set by a reorg/revert so the next recorded commit stamps a
    /// `first_winning_commit_published` event — the freshness endpoint of a recovery.
    first_winning_commit_pending: bool,
    /// The cause id that armed `first_winning_commit_pending`, stamped into that event.
    first_winning_commit_cause: u64,
    /// The out-of-band lifecycle event log, present exactly when a stream is recorded.
    events: Option<producer_events::ProducerEvents>,
    /// Whether the accepted-head wait has been logged, so a rare condition does not log per block.
    waiting_for_accepted_head_logged: bool,
    /// Workers alive right now, the abandoned ones included.
    ///
    /// The fence drops a worker's receiver, not the worker: it runs its multiproof to completion
    /// holding an MDBX read transaction the whole way. This gauge is the only thing that knows
    /// how many are still doing that, so it is what the spawn cap reads — and it is decremented
    /// by the worker's own drop guard, which is the one signal a fenced worker still sends.
    live_workers: Arc<AtomicUsize>,
    /// Ceiling on `live_workers` before a fresh spawn waits instead.
    max_workers: usize,
    /// Whether the cap wait has been logged, so a long-running worker does not log per block.
    worker_cap_logged: bool,
}

impl BootstrapGate {
    fn new(options: &RunOptions) -> Self {
        Self {
            export_pending: options.bootstrap_export,
            self_test_blocks: options.bootstrap_self_test_blocks,
            shadow: None,
            job: ExportJob::Idle,
            retries_left: options.stream_export_retries,
            attempt: 0,
            pending_cause: RecheckpointCause::Initial,
            attempt_cause: RecheckpointCause::Initial,
            cause_counter: 0,
            pending_cause_id: 0,
            attempt_cause_id: 0,
            first_winning_commit_pending: false,
            first_winning_commit_cause: 0,
            events: None,
            waiting_for_accepted_head_logged: false,
            live_workers: Arc::new(AtomicUsize::new(0)),
            max_workers: options.stream_export_max_workers,
            worker_cap_logged: false,
        }
    }

    const fn wants_sidecar(&self) -> bool {
        self.shadow.is_some()
    }

    /// Allocates the next cause id, the correlation key that travels from a detection event
    /// through the export attempts it arms to the checkpoint and first commit it produces.
    fn begin_cause(&mut self) -> u64 {
        self.cause_counter += 1;
        self.cause_counter
    }

    /// Closes an armed first-winning-commit measurement without a publication.
    ///
    /// Every armed measurement terminates in exactly one event — published or unmeasured —
    /// because a pending flag that silently outlives its cause gets resolved by whatever
    /// unrelated commit publishes next, stamping the old cause id onto a block that has
    /// nothing to do with its branch. No-op when nothing is pending.
    fn abandon_first_winning(&mut self, reason: &str) {
        if std::mem::replace(&mut self.first_winning_commit_pending, false) {
            let cause_id = self.first_winning_commit_cause;
            self.emit_event(
                "first_winning_commit_unmeasured",
                serde_json::json!({ "cause_id": cause_id, "reason": reason }),
            );
        }
    }

    /// Appends one producer lifecycle event, when a stream is being recorded.
    fn emit_event(&mut self, kind: &str, fields: serde_json::Value) {
        let attempt = self.attempt;
        if let Some(events) = self.events.as_mut() {
            events.emit(kind, attempt, fields);
        }
    }

    /// Whether a fresh export may spawn, counting what the fence cannot stop: abandoned workers
    /// still holding their MDBX read transactions. Waiting costs one Ready per check; unbounded
    /// readers cost the database. The caller must not consume `export_pending` on a refusal, so
    /// the next Ready retries once a worker has drained.
    fn worker_slot_available(&mut self) -> bool {
        let live = self.live_workers.load(Ordering::SeqCst);
        if live >= self.max_workers {
            if !self.worker_cap_logged {
                warn!(
                    target: "partial_stateless",
                    live,
                    cap = self.max_workers,
                    "Export worker cap reached; the fresh attempt waits for a worker to drain \
                     rather than stacking another read transaction"
                );
                self.worker_cap_logged = true;
            }
            return false
        }
        self.worker_cap_logged = false;
        true
    }
}

/// How often the notification loop polls an in-flight export for completion, so a finished
/// checkpoint publishes on its own clock instead of the next block's.
const EXPORT_COMPLETION_TICK_MS: u64 = 500;

/// Why a snapshot export is (or will be) armed, which decides the publication ordering.
///
/// After a reorg or revert the pair recovered in place, so every winning-branch commit is
/// independently verifiable by a consumer holding the retained generation — those commits write
/// through and the recovery checkpoint publishes behind them. Every other cause keeps the
/// checkpoint-first ordering: the initial export opens the stream, and a discontinuity (rebuild,
/// cold reset, or a mid-branch `Reset` frame) can only be crossed by restoring a checkpoint
/// anchored above it, never by replaying commits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecheckpointCause {
    /// The stream-opening bootstrap export.
    Initial,
    /// A reorg or revert recovered in place; commits stay independently applicable.
    BranchChange,
    /// A rebuild, cold reset, or mid-branch reset broke stream continuity.
    Discontinuity,
}

impl RecheckpointCause {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Initial => "initial",
            Self::BranchChange => "branch_change",
            Self::Discontinuity => "discontinuity",
        }
    }
}

/// One occupied slot in the export-worker gauge.
///
/// Held by the worker closure and released by `Drop`, so every exit — completion, error, panic
/// unwind — gives the slot back. Nothing else may decrement the gauge: the notification task
/// cannot know when a fenced worker finishes, which is the entire reason the gauge exists.
struct WorkerSlot(Arc<AtomicUsize>);

impl WorkerSlot {
    fn occupy(gauge: Arc<AtomicUsize>) -> Self {
        gauge.fetch_add(1, Ordering::SeqCst);
        Self(gauge)
    }
}

impl Drop for WorkerSlot {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// The snapshot export as the notification loop sees it.
///
/// The blocking task itself cannot be cancelled — once spawned it runs to completion — so
/// "abandoning" an attempt means dropping the receiver: the worker's send fails silently and its
/// files sit in an attempt directory nothing will ever promote. That is also why each attempt
/// exports into its own directory rather than the fixed operator paths: a stale worker finishing
/// minutes late must not overwrite a newer attempt's package.
enum ExportJob {
    /// No export running.
    Idle,
    /// An export is running on a blocking thread; the pair keeps advancing meanwhile.
    InFlight {
        /// Where the worker's result arrives.
        rx: mpsc::Receiver<eyre::Result<bootstrap_io::FinishedExport>>,
        /// The accepted head captured at H — the pair's own has advanced past it by the time the
        /// export completes, and the checkpoint frame must carry H's header, not the tip's.
        accepted_head: SealedHeader,
        /// The attempt's private export directory.
        attempt_dir: PathBuf,
        /// When the export was spawned.
        started: Instant,
    },
    /// An export completed and its checkpoint (if recording) is on disk.
    Finished,
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
    // The windows are part of the name because the file's contents mean nothing without them:
    // the format carries no policy of its own, and the loader attaches whatever policy the running
    // process holds. A cache filled under a 240-block storage window and reloaded under a 30-block
    // one would hold entries a genuine 30-block peer evicted long ago, and nothing downstream
    // would object — the anchors it produces are internally consistent, so the arm would report a
    // hit ratio belonging to neither window. A name cannot go stale against what it describes, and
    // a run whose windows changed simply finds no file and starts cold.
    let cache_path = cache_dir.as_ref().join(format!(
        "partial_stateless_cache-a{}-s{}.bin",
        config.account_window, config.storage_window
    ));
    let options = RunOptions::from_env(config)?;
    options.log_summary(&cache_path);

    let mut pair = load_initial_pair(&options, &cache_path, ctx.head.number);
    let mut gate = BootstrapGate::new(&options);
    // Built before the first notification so a misconfigured spool fails the run at startup rather
    // than after the snapshot export has already been paid for.
    let mut recorder = recorder::StreamRecorder::from_env()?;
    // Built here for the same reason as the spool: a capture directory that already holds records,
    // or a conflicting benchmark variable, fails the run at startup rather than after hours of
    // blocks. `None` in every ordinary run, and nothing in the capture path exists behind it.
    let mut dataset = policy_dataset_capture::PolicyDatasetRecorder::open(
        options.policy_dataset.clone(),
        format!("{}/{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")),
        option_env!("PS_BUILD_COMMIT").map(str::to_string),
        ctx.config.chain.chain().to_string(),
    )?;
    if let Some(dataset) = dataset.as_mut() {
        dataset.note_started(ctx.head.number.saturating_add(1));
    }
    // Attempt directories are export machinery, and exports run only when a stream is being
    // recorded — a run without one has no business deleting anything under the bootstrap dir.
    if recorder.is_some() {
        remove_stale_attempt_dirs(&options.bootstrap_dir);
    }
    if let Some(recorder) = recorder.as_mut() {
        recorder.write_manifest(
            ctx.config.chain.chain().id(),
            ctx.config.chain.genesis_hash(),
            options.config.cache_policy_id(),
            options.config.account_window,
            options.config.storage_window,
        )?;
        // After the manifest, so the log carries the epoch frames are actually written under.
        gate.events =
            Some(producer_events::ProducerEvents::beside_spool(recorder.dir(), recorder.epoch()));
    }
    // A rebuild that keeps failing is almost always a persistent condition — pruned history, or a
    // provider that cannot reach far enough back — and retrying it every block would spend the
    // whole run re-executing windows that never install.
    let mut rebuild_failures = 0u32;

    loop {
        // Every `?`-shaped exit from this loop must classify itself first: the recorder's drop
        // cannot tell an error return from reth dropping the future at shutdown — both run the
        // destructor — so an unclassified propagation would close a crashed producer's stream as
        // an orderly `Shutdown`.
        // While an export is in flight, completion must publish on its own clock: waiting for the
        // next notification would quantize checkpoint publication to block cadence, and a quiet
        // chain (or a pure revert with nothing behind it) would hold a finished checkpoint
        // unpublished indefinitely. `try_next` on the notification channel is cancel-safe, so the
        // timeout drops nothing.
        let notification = if matches!(gate.job, ExportJob::InFlight { .. }) {
            match tokio::time::timeout(
                std::time::Duration::from_millis(EXPORT_COMPLETION_TICK_MS),
                ctx.notifications.try_next(),
            )
            .await
            {
                Err(_tick) => {
                    poll_export_job(&options, &mut gate, &mut recorder.as_mut());
                    continue
                }
                Ok(Ok(Some(notification))) => notification,
                Ok(Ok(None)) => break,
                Ok(Err(err)) => {
                    return Err(fail_producer(
                        recorder.as_mut(),
                        err,
                        "the notification stream failed",
                    ))
                }
            }
        } else {
            match ctx.notifications.try_next().await {
                Ok(Some(notification)) => notification,
                Ok(None) => break,
                Err(err) => {
                    return Err(fail_producer(
                        recorder.as_mut(),
                        err,
                        "the notification stream failed",
                    ))
                }
            }
        };
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
                        recorder.as_mut(),
                        dataset.as_mut(),
                        block,
                    ) {
                        // Written explicitly rather than left to the recorder's drop: the drop
                        // cannot tell an error return from a clean shutdown, and the reason is
                        // only known here.
                        if let Some(recorder) = recorder.as_mut() {
                            recorder.write_end(
                                EndKind::ProducerFault,
                                format!("block {block_number} failed: {err:#}"),
                            );
                        }
                        return Err(eyre::eyre!("block {block_number} failed: {err:#}"))
                    }
                }

                // Cache persistence is unrelated to validation and can perturb later
                // Engine samples, so the bounded paired benchmark keeps it in memory only.
                if persist_cache(&options, &pair, &cache_path, tip_block, true) &&
                    let Some(recorder) = recorder.as_mut()
                {
                    recorder.note_durable(tip_block);
                }
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

                let cause_id = gate.begin_cause();
                gate.emit_event(
                    "reorg_detected",
                    serde_json::json!({
                        "abandoned_from": *old.range().start(),
                        "abandoned_to": *old.range().end(),
                        "winning_tip": tip_block,
                        "ancestor_hash": ancestor_hash.map(|hash| format!("{hash:?}")),
                        "cause_id": cause_id,
                    }),
                );
                // Fenced before the recovery runs. Whatever H an in-flight export chose, it
                // chose it on the branch this notification is abandoning.
                interrupt_export(
                    &options,
                    &mut gate,
                    &mut recorder.as_mut(),
                    RecheckpointCause::BranchChange,
                    cause_id,
                    "the chain reorged under the export",
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
                // Written before the winning branch's commits, so a consumer learns which blocks
                // left the chain before it is asked to apply the ones that replaced them.
                if let Some(recorder) = recorder.as_mut() {
                    recorder.write_reorg(branch_change(old, Some(new)));
                }
                note_dataset_branch_change(dataset.as_mut(), old);
                // A branch change that lands while an earlier first-winning measurement is
                // still armed supersedes it: that branch's first commit never published.
                gate.abandon_first_winning("superseded_by_branch_change");
                gate.first_winning_commit_pending = true;
                gate.first_winning_commit_cause = cause_id;
                // Started here, before a single winning block applies, because the checkpoint a
                // consumer needs is the one authenticated at the *common ancestor*, which is the
                // only block a recovery snapshot may be named at. One block later the pair
                // has already moved past it. The winning branch buffers behind the checkpoint
                // from this point on.
                maybe_start_export(&ctx, &options, &pair, &mut gate, recorder.as_mut());

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
                        recorder.as_mut(),
                        dataset.as_mut(),
                        block,
                    ) {
                        if let Some(recorder) = recorder.as_mut() {
                            recorder.write_end(
                                EndKind::ProducerFault,
                                format!("reorg block {block_number} failed: {err:#}"),
                            );
                        }
                        return Err(eyre::eyre!("reorg block {block_number} failed: {err:#}"))
                    }
                }

                // Production persists the rebuilt cache so a restart cannot reload the old
                // branch. Benchmark mode deliberately keeps all cache state in memory.
                if persist_cache(&options, &pair, &cache_path, tip_block, true) &&
                    let Some(recorder) = recorder.as_mut()
                {
                    recorder.note_durable(tip_block);
                }
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

                let cause_id = gate.begin_cause();
                gate.emit_event(
                    "revert_detected",
                    serde_json::json!({
                        "reverted_from": *old.range().start(),
                        "reverted_to": *old.range().end(),
                        "new_tip_hash": new_tip_hash.map(|hash| format!("{hash:?}")),
                        "cause_id": cause_id,
                    }),
                );
                interrupt_export(
                    &options,
                    &mut gate,
                    &mut recorder.as_mut(),
                    RecheckpointCause::BranchChange,
                    cause_id,
                    "the chain reverted under the export",
                );

                let recovered = recover_at(
                    &ctx,
                    &options,
                    &mut pair,
                    *old.range().start(),
                    new_tip_hash,
                    &mut rebuild_failures,
                );
                // A pure revert: the same event with no winning tip, because nothing replaces the
                // abandoned blocks. Collapsing it into the reorg variant keeps a consumer from
                // needing two ways to unwind.
                if let Some(recorder) = recorder.as_mut() {
                    recorder.write_reorg(branch_change(old, None));
                }
                note_dataset_branch_change(dataset.as_mut(), old);
                // Nothing replaces the reverted blocks, so the "first winning commit" after a
                // pure revert is simply the next committed block — still the freshness endpoint.
                gate.abandon_first_winning("superseded_by_branch_change");
                gate.first_winning_commit_pending = true;
                gate.first_winning_commit_cause = cause_id;
                maybe_start_export(&ctx, &options, &pair, &mut gate, recorder.as_mut());

                // A pair that could not be rebuilt still describes the reverted branch, and
                // persisting it would let a restart reload exactly that.
                let durable = old.range().start().saturating_sub(1);
                if persist_cache(&options, &pair, &cache_path, durable, recovered) &&
                    let Some(recorder) = recorder.as_mut()
                {
                    recorder.note_durable(durable);
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
                    if let Err(err) =
                        ctx.events.send(ExExEvent::FinishedHeight(BlockNumHash::new(number, hash)))
                    {
                        return Err(fail_producer(
                            recorder.as_mut(),
                            eyre::Report::new(err),
                            "the acknowledgement channel closed",
                        ))
                    }
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

    // The notification stream ended, which is a clean shutdown. Reth's SIGTERM path never returns
    // from the loop above — it drops this future — so the recorder's own drop is what closes the
    // stream there; this explicit call only covers the stream draining on its own.
    if let Some(recorder) = recorder.as_mut() {
        recorder.write_end(EndKind::Shutdown, "exex notification stream ended");
    }
    // A capture that stopped short of its budget still gets a terminator, and the terminator says
    // it stopped short. The alternative is a dataset directory that reads as incomplete forever,
    // which is indistinguishable from one whose producer crashed.
    if let Some(dataset) = dataset.as_mut() {
        dataset.close(
            partial_stateless::DatasetEndKind::ProducerShutdown,
            "exex notification stream ended".to_string(),
        );
    }

    Ok(())
}

/// Joins one block's capture material to its payload and parent header, and records it.
///
/// Split out so the caller has one fallible expression to mark the dataset failed on. Inline, each
/// `?` would be a separate exit that left the dataset without its terminator — readable as a
/// crashed capture rather than as the refused one it is.
fn record_dataset_block<Node>(
    ctx: &ExExContext<Node>,
    recorder: &mut policy_dataset_capture::PolicyDatasetRecorder,
    material: policy_dataset_capture::PolicyDatasetMaterial,
    payload: Option<(Option<Vec<u8>>, partial_stateless::RecordedPayloadProvenance)>,
    block: &RecoveredBlock<BlockTy<EthPrimitives>>,
) -> eyre::Result<()>
where
    Node: FullNodeComponents<Types: NodeTypes<Primitives = EthPrimitives>>,
    Node::Provider: CanonicalOverlayFactory + BlockReader<Block = BlockTy<EthPrimitives>>,
{
    let (payload_json, payload_provenance) =
        payload.unwrap_or((None, partial_stateless::RecordedPayloadProvenance::Absent));
    // Fetched here rather than inside the builder: the parent header is the one thing a record
    // needs that the builder never handles, and threading a second header lookup through it would
    // put a capture-only read on the production path.
    let parent_header = ctx
        .provider()
        .sealed_header_by_hash(block.parent_hash)
        .map_err(|err| eyre::eyre!("failed to fetch parent header for the dataset: {err}"))?
        .ok_or_else(|| {
            eyre::eyre!("parent header {:?} not found for the dataset", block.parent_hash)
        })?;
    let mut parent_header_rlp = Vec::new();
    parent_header.header().encode(&mut parent_header_rlp);
    let body = material.into_record_body(
        block.number(),
        block.hash(),
        block.parent_hash,
        parent_header_rlp.into(),
        payload_json,
        payload_provenance,
    )?;
    recorder.record(body)
}

/// Files a branch change in the dataset's lifecycle log.
///
/// The abandoned records stay on disk. Deciding which of two records at one height is canonical is
/// the offline stage's job, and it needs both the records and this event to do it — a capture that
/// deleted the loser would leave the exclusion unauditable.
fn note_dataset_branch_change(
    dataset: Option<&mut policy_dataset_capture::PolicyDatasetRecorder>,
    old: &Chain,
) {
    let Some(dataset) = dataset else { return };
    let abandoned =
        old.blocks().iter().map(|(number, block)| (*number, block.hash())).collect::<Vec<_>>();
    let common_ancestor = old.range().start().saturating_sub(1);
    dataset.note_reorg(common_ancestor, abandoned);
}

/// Classifies an error exit from the notification loop before it propagates.
///
/// Idempotent through `write_end` itself: a path that already wrote a more specific `End` keeps
/// it, because the first close wins.
fn fail_producer(
    recorder: Option<&mut recorder::StreamRecorder>,
    err: eyre::Report,
    what: &str,
) -> eyre::Report {
    if let Some(recorder) = recorder {
        recorder.write_end(EndKind::ProducerFault, format!("{what}: {err:#}"));
    }
    err.wrap_err(what.to_string())
}

/// Describes a branch change as the stream's one unwind event.
///
/// A pure revert is `new = None`: the same abandoned list, no winning tip. Two event kinds would
/// mean two ways for a consumer to unwind, and the second one would be the one nothing exercises.
fn branch_change(old: &Chain, new: Option<&Chain>) -> Reorg {
    let block_ref = |block: &RecoveredBlock<BlockTy<EthPrimitives>>| StreamBlockRef {
        number: block.number(),
        hash: block.hash(),
    };
    // The common ancestor by hash rather than by height, for the reason the recovery path already
    // addresses it that way: mid-reorg a height names whichever block the database calls canonical.
    let first_abandoned = old.blocks().values().next();
    Reorg {
        common_ancestor: StreamBlockRef {
            number: old.range().start().saturating_sub(1),
            hash: first_abandoned.map(|block| block.parent_hash).unwrap_or_default(),
        },
        abandoned: old.blocks().values().map(block_ref).collect(),
        winning_tip: new.and_then(|new| new.blocks().values().next_back()).map(block_ref),
    }
}

/// Writes the flat cache to disk, unless the run is a bounded in-memory benchmark.
///
/// `canonical` is false when the pair may still describe an abandoned branch, in which case
/// persisting would let a restart reload exactly that. Returns whether the cache is now durable
/// through `block`, which is the fact a recorded commit reports and a restart resumes from.
fn persist_cache(
    options: &RunOptions,
    pair: &LivePair,
    cache_path: &Path,
    block: u64,
    canonical: bool,
) -> bool {
    if options.validation_bench || !canonical {
        return false
    }
    if let Err(e) = save_to_file(&pair.cache, cache_path) {
        warn!(
            target: "partial_stateless",
            block,
            error = %e,
            "Failed to save cache state to disk"
        );
        return false
    }
    true
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

                // Gap tolerance is the *larger* window, not the account one: the cache is only
                // fully stale once the longer of the two has gone by, and under a sweep arm the
                // storage window can be the longer one.
                let max_allowed_gap = config.max_window();
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
        trie_cache: PartialTrieNodeCache::new_with_repr(options.trie_repr),
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
            // a generation this one does not descend from, so it goes.
            pair.forget_retained_generation();
            // The header that *was* left behind is the sharper problem: a reorg rebuild installs
            // the winning sibling at the same number the abandoned block had, so keeping the old
            // one leaves a header `accepted_parent` has to reject on hash rather than on height.
            // The answer is not to have none — a headless pair cannot admit its next child and
            // cannot export a checkpoint anything could restore from — but to install the header
            // for the hash the rebuild was performed at, fetched from the provider whose state
            // root the rebuild's own multiproof was just authenticated against. It is by
            // construction the header this generation is the state after, and `accepted_parent`
            // re-checks number, hash, state root, cache root and trie root before ever offering
            // it, so a wrong install is still refused at use. A lookup that fails degrades to
            // `None` and the pair waits a block, which is what it used to do every time.
            pair.accepted_head =
                ctx.provider().sealed_header_by_hash(target_hash).ok().flatten().filter(|header| {
                    header.hash() == target_hash && header.number() == ready.anchor.block_number
                });
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
#[expect(clippy::too_many_arguments)]
fn process_canonical_block<Node>(
    ctx: &ExExContext<Node>,
    options: &RunOptions,
    pair: &mut LivePair,
    gate: &mut BootstrapGate,
    rebuild_failures: &mut u32,
    mut recorder: Option<&mut recorder::StreamRecorder>,
    mut dataset: Option<&mut policy_dataset_capture::PolicyDatasetRecorder>,
    block: &RecoveredBlock<BlockTy<EthPrimitives>>,
) -> eyre::Result<()>
where
    Node: FullNodeComponents<Types: NodeTypes<Primitives = EthPrimitives>>,
    Node::Provider: CanonicalOverlayFactory + BlockReader<Block = BlockTy<EthPrimitives>>,
{
    let block_number = block.number();
    // Taken before anything else this block triggers. The handoff records how long each artifact
    // waited, and a take deferred behind sidecar construction would report a residence no real
    // consumer would ever see.
    let tapped = options.payload_tap.then(|| {
        let tapped = payload_tap::tap_payload(block);
        report_payload_tap(block_number, block.hash(), &tapped);
        tapped
    });
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
            // Whatever the export was going to checkpoint, it is not the generation this pair is
            // about to become. A discontinuity cause keeps the checkpoint-first ordering: the
            // consumer this Reset strands can only continue from a checkpoint above it.
            gate.abandon_first_winning("discontinuity");
            let cause_id = gate.begin_cause();
            interrupt_export(
                options,
                gate,
                &mut recorder,
                RecheckpointCause::Discontinuity,
                cause_id,
                "the pair was rebuilt or reset under the export",
            );
            // The consumer has to be told, or it would keep applying commits from a pair whose
            // generation silently changed underneath the stream. Before this, a producer that
            // rebuilt or cold-reset wrote no frame at all and the commits simply kept coming —
            // which reads to a follower as the validator disagreeing with itself, block after
            // block, with nothing in the stream to explain it.
            let reset_reason = if matches!(reason, BlockedReason::RecoveryIncomplete { .. }) {
                ResetReason::SnapshotRequired
            } else {
                ResetReason::Gap
            };
            if let Some(recorder) = recorder.as_mut() {
                recorder.write_reset(
                    reset_reason,
                    format!("block {block_number} broke cache continuity: {reason:?}"),
                );
            }
            // The dataset carries no cache state, so a pair reset does not invalidate a record.
            // It is filed anyway: an offline run that finds an unexplained discontinuity in the
            // producer's own history should be able to see that the producer knew about it.
            if let Some(dataset) = dataset.as_deref_mut() {
                dataset.note_reset(
                    block_number,
                    format!("the producer's cache pair was reset here: {reason:?}"),
                );
            }
            // An exact rebuild at this block's own parent is worth trying before falling back to a
            // policy window of live warming, and it is the same primitive a reorg recovers with.
            let admitted =
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
                };
            // A rebuild lands `Ready` at this block's parent, so the checkpoint can be taken
            // there — the exact block the consumer's stream continues from. A cold reset has no
            // such block: it warms for a policy window first, and the checkpoint it eventually
            // publishes is an explicit reset rather than continuous recovery. The gate below
            // starts whichever of the two is available.
            maybe_start_export(ctx, options, pair, gate, recorder.as_deref_mut());
            admitted
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
        // Deliberately no frame. A verifier's outcome is a second opinion about a sidecar someone
        // else built, and recording it as the producer's own would put a consumer's verdict where
        // a producer's belongs.
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

    // Every canonical block advances the capture's confirmation clock, including the ones after
    // its budget is met — those cost it nothing and are the whole of what separates a corpus whose
    // tail is reorg-exposed from one that is not. This is also where the dataset closes itself.
    if let Some(dataset) = dataset.as_deref_mut() {
        dataset.observe_head(block_number);
    }

    // Serialized here rather than after the build, because the stream recorder consumes `tapped`
    // and the dataset needs the same bytes. Serialized at all only when a dataset is being
    // captured: an ordinary run never pays for this.
    let dataset_payload = dataset.as_ref().filter(|d| d.wants_block()).and_then(|_| {
        tapped.as_ref().map(|tapped| {
            let json = tapped.payload.as_ref().and_then(|payload| {
                serde_json::to_vec(payload)
                    .inspect_err(|err| {
                        warn!(
                            target: "partial_stateless",
                            block = block_number,
                            error = %err,
                            "Engine payload could not be serialized for the policy replay dataset"
                        );
                    })
                    .ok()
            });
            // Provenance describes what the record carries, never what the producer held: a
            // payload that failed to serialize is an absent one, and the dataset refuses it.
            let provenance = match (json.is_some(), tapped.provenance) {
                (true, payload_tap::PayloadProvenance::Witnessed) => {
                    partial_stateless::RecordedPayloadProvenance::Witnessed
                }
                (true, payload_tap::PayloadProvenance::Reconstructed) => {
                    partial_stateless::RecordedPayloadProvenance::Reconstructed
                }
                _ => partial_stateless::RecordedPayloadProvenance::Absent,
            };
            (json, provenance)
        })
    });

    let records_commits =
        recorder.as_ref().is_some_and(|recorder| recorder.wants_commit_material());
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
            // The recorder needs the sidecar as a value, and the builder otherwise hands back
            // only the path it wrote it to.
            gate.wants_sidecar() || records_commits,
            retained_generation,
            // Asked for per block rather than once, so a capture stops paying for full witnesses
            // the moment its block budget is met rather than at the end of the run.
            dataset.as_ref().is_some_and(|recorder| recorder.wants_block()),
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
        policy_dataset_material,
    } = report;
    finish_committed_transition(
        pair,
        displaced_trie_cache,
        &block_ctx,
        block.clone_sealed_header(),
        options.retain_generation,
    );

    // Before the stream frame, so a capture that cannot record a block fails the run on that block
    // rather than after a commit has already claimed it. Fail-closed twice over: the dataset gets a
    // `Failed` terminator so a later reader refuses it outright, *and* the run stops — an
    // incomplete corpus is worse than no corpus, because it looks like a complete one, and a
    // capture run that kept going after its dataset died would waste the operator's night proving
    // nothing.
    //
    // Both of those are statements about a corpus that has records in it. Before the first one the
    // capture is still arriving: the node spends most of a minute getting to where the ExEx takes
    // its first notification, and the Engine taps hand off through ring buffers that keep moving
    // meanwhile, so the block the ExEx lands on can be one whose payload was already evicted.
    // Nothing is wrong with the corpus then — it has not started — and killing the node over it
    // costs the operator the run for a reason that will have passed by the next block.
    if let Some(material) = policy_dataset_material {
        let recorder = dataset
            .ok_or_else(|| eyre::eyre!("the builder captured dataset material with no recorder"))?;
        match record_dataset_block(ctx, recorder, material, dataset_payload, block) {
            Ok(()) => {}
            Err(err) => {
                let skipped = err
                    .downcast_ref::<policy_dataset_capture::PolicyDatasetMaterialError>()
                    .is_some_and(|miss| recorder.skip_startup_handoff_miss(block_number, miss));
                if !skipped {
                    recorder.fail(format!("block {block_number} could not be recorded: {err:#}"));
                    return Err(err.wrap_err(format!(
                        "policy replay dataset capture failed at block {block_number}"
                    )))
                }
            }
        }
    }

    // After the transition, so the fingerprints describe the generation this block produced rather
    // than the one it displaced, and before the bootstrap gate, which touches only its own shadow.
    let commit_recorded = recorder
        .as_deref_mut()
        .map(|recorder| record_commit(recorder, options, pair, block, sidecar.as_ref(), tapped));
    match commit_recorded {
        Some(CommitRecordingOutcome::Reset) => {
            // The commit became a Reset, which discontinues the stream. A recovery checkpoint
            // anchored below a Reset can never carry a consumer across it, so an in-flight export
            // is fenced here and re-armed at the current tip (policy permitting) — the same
            // treatment every other discontinuity gets, and the guarantee behind the invariant
            // that a published late checkpoint's replay window holds Commit frames only.
            gate.abandon_first_winning("stream_reset");
            let cause_id = gate.begin_cause();
            interrupt_export(
                options,
                gate,
                &mut recorder,
                RecheckpointCause::Discontinuity,
                cause_id,
                "a reset discontinued the stream under the export",
            );
        }
        // Only an actually-published frame is a publication: a commit that landed in a
        // discontinuity export's buffer has no publication time yet, and stamping one from the
        // recorder's stale sequence is exactly what the typed disposition exists to prevent.
        // The pending flag survives a buffered or dropped commit and resolves at the first
        // frame that really reaches the open stream.
        Some(CommitRecordingOutcome::Recorded(recorder::CommitDisposition::Published {
            sequence,
        })) if gate.first_winning_commit_pending => {
            gate.first_winning_commit_pending = false;
            gate.emit_event(
                "first_winning_commit_published",
                serde_json::json!({
                    "block": block.number(),
                    "sequence": sequence,
                    "cause_id": gate.first_winning_commit_cause,
                }),
            );
        }
        // A commit that did not reach the open stream terminates the measurement rather than
        // deferring it: the frame is either behind a checkpoint-first export (its availability
        // endpoint is that checkpoint's publication, not any later commit) or gone. Leaving the
        // flag armed would stamp the old cause id onto the next unrelated published commit.
        Some(CommitRecordingOutcome::Recorded(recorder::CommitDisposition::Buffered))
            if gate.first_winning_commit_pending =>
        {
            gate.abandon_first_winning("buffered_behind_checkpoint");
        }
        Some(CommitRecordingOutcome::Recorded(recorder::CommitDisposition::Dropped))
            if gate.first_winning_commit_pending =>
        {
            gate.abandon_first_winning("not_published");
        }
        _ => {}
    }

    advance_bootstrap_gate(ctx, options, pair, gate, recorder, block, sidecar.as_ref())
}

/// Writes one commit frame: the block's input, and what this producer concluded about it.
///
/// **What the oracle's fields are worth is not uniform, and pretending otherwise would overstate
/// the gate.** `next_cache_anchor` is read off the pair's own post-transition state rather than
/// copied out of the sidecar, so it is the producer's result; `expected_miss` is the sidecar's
/// claim, which on a builder-recorded stream is the same value the producer computed, so a replay
/// comparing against it is checking its own derivation against the producer's rather than against
/// an independent third value. The fields with real teeth are the two fingerprints: the trie cache
/// root commits retained-path membership, which appears in no sidecar and which a replay computes
/// entirely on its own.
fn record_commit(
    recorder: &mut recorder::StreamRecorder,
    options: &RunOptions,
    pair: &LivePair,
    block: &RecoveredBlock<BlockTy<EthPrimitives>>,
    sidecar: Option<&PartialStatelessSidecar>,
    tapped: Option<payload_tap::TappedPayload>,
) -> CommitRecordingOutcome {
    let Some(sidecar) = sidecar else {
        // A block that published no sidecar cannot be replayed by anything, so recording it would
        // add a frame no consumer could use and cost the stream its contiguity claim.
        recorder.write_reset(
            ResetReason::Gap,
            format!("block {} produced no sidecar to record", block.number()),
        );
        return CommitRecordingOutcome::Reset
    };
    let sidecar_bytes = match bincode::serialize(sidecar) {
        Ok(bytes) => bytes,
        Err(err) => {
            recorder.write_reset(
                ResetReason::Gap,
                format!("block {} sidecar failed to serialize: {err}", block.number()),
            );
            return CommitRecordingOutcome::Reset
        }
    };
    let (payload_json, payload_provenance) = match tapped {
        Some(tapped) => {
            let json = tapped.payload.as_ref().and_then(|payload| {
                serde_json::to_vec(payload)
                    .inspect_err(|err| {
                        warn!(
                            target: "partial_stateless_stream",
                            block = block.number(),
                            error = %err,
                            "Engine payload could not be serialized; recording the commit without \
                             one rather than with a payload no consumer could parse"
                        );
                    })
                    .ok()
            });
            // A payload that failed to serialize is an absent payload and not a witnessed one: the
            // provenance describes what the frame carries, never what the producer held.
            let provenance = if json.is_some() {
                tapped.provenance
            } else {
                payload_tap::PayloadProvenance::Absent
            };
            (json, provenance)
        }
        None => (None, payload_tap::PayloadProvenance::Absent),
    };

    let fingerprint = pair.fingerprint();
    let lifecycle = pair.lifecycle_fingerprint();
    let input = CommitInput {
        block: StreamBlockRef { number: block.number(), hash: block.hash() },
        parent_hash: block.parent_hash,
        payload_provenance,
        payload_json,
        sidecar: sidecar_bytes,
    };
    let oracle = CommitOracle {
        verdict: RecordedVerdict::Accepted,
        state_root: Some(block.state_root()),
        next_cache_anchor: Some(CacheAnchor {
            block_number: block.number(),
            block_hash: block.hash(),
            cache_policy_id: options.config.cache_policy_id(),
            cache_root: fingerprint.cache_root,
        }),
        expected_miss: Some(sidecar.cache_miss_targets.clone()),
        readiness_state: pair.readiness.state().label().to_string(),
        readiness_watermark: pair
            .readiness
            .acknowledgeable_height()
            .map(|(number, hash)| StreamBlockRef { number, hash }),
        // Filled in by the recorder, which is the only thing that knows it.
        durability_watermark: None,
        retained_generation: lifecycle
            .retained_generation
            .map(|(number, hash)| StreamBlockRef { number, hash }),
        coordinated_fingerprint: fingerprint,
        lifecycle_fingerprint: lifecycle,
    };
    CommitRecordingOutcome::Recorded(recorder.write_commit(input, oracle))
}

/// What one canonical block became on the stream: a commit frame with its typed disposition, or
/// the Reset that discontinued the stream instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommitRecordingOutcome {
    /// A commit frame was handed to the recorder; the disposition says where it landed.
    Recorded(recorder::CommitDisposition),
    /// The block could not be recorded and a Reset frame took its place.
    Reset,
}

/// Logs one block's payload provenance, and drops the payload.
///
/// This is the log line for the tap itself: the payload is measured here, and [`record_commit`] is
/// what writes it into a commit frame. The level is the claim: a witnessed payload is the ordinary
/// case, and a derived one is the case where a later corpus will contain a block
/// whose admission checks pass without checking anything.
fn report_payload_tap(block_number: u64, block_hash: B256, tapped: &payload_tap::TappedPayload) {
    let stats = tapped.stats.as_ref();
    let queue_depth = stats.map(|stats| stats.queue_depth);
    let missed = stats.map(|stats| stats.missed);
    match tapped.provenance {
        payload_tap::PayloadProvenance::Witnessed => info!(
            target: "partial_stateless_payload",
            block = block_number,
            ?block_hash,
            provenance = tapped.provenance.as_str(),
            approx_bytes = tapped.approx_bytes,
            residence_us = tapped.residence_us,
            witnessed = tapped.witnessed_total,
            reconstructed = tapped.reconstructed_total,
            queue_depth,
            "Took the Engine's own payload for this block"
        ),
        payload_tap::PayloadProvenance::Reconstructed => warn!(
            target: "partial_stateless_payload",
            block = block_number,
            ?block_hash,
            provenance = tapped.provenance.as_str(),
            reason = tapped.miss_reason.as_ref().map(reth_execution_access::MissReason::as_str),
            witnessed = tapped.witnessed_total,
            reconstructed = tapped.reconstructed_total,
            queue_depth,
            missed,
            "No Engine payload for this block; derived one from the block, whose layout, \
             block-hash and versioned-hash checks a validator would pass vacuously"
        ),
        // Unreachable while `payload_tap` gates the call, and logged rather than asserted because
        // the gate is read once at startup and the handoff is allocated lazily.
        payload_tap::PayloadProvenance::Absent => warn!(
            target: "partial_stateless_payload",
            block = block_number,
            ?block_hash,
            provenance = tapped.provenance.as_str(),
            "Payload tap is armed but this process is publishing no payloads"
        ),
    }
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
    let before = pair.last_readiness_label;
    let after =
        pair.commit_transition(displaced_trie_cache, block, accepted_head, retain_generation);
    pair.last_readiness_label = after;
    log_readiness_change(pair, block, before, after);
}

/// Runs the bootstrap gate: export at the first usable Ready, then compare or stream.
///
/// Two export paths live here, and which one runs is decided at startup. The self-test keeps the
/// original synchronous export — it compares fingerprints at H, which requires the pair frozen
/// there, and `RunOptions::from_env` refuses to combine it with a recorder. Every other run
/// exports **off-task**: the state is captured at `Ready(H)` in milliseconds, the whole-cache
/// multiproof runs on a blocking thread, and the pair keeps advancing — which is what keeps the
/// Engine payload handoff drained and the head of a recorded stream `witnessed` instead of
/// `reconstructed`. Frames for H + 1 onward buffer in the recorder and flush behind the
/// checkpoint when the export completes.
fn advance_bootstrap_gate<Node>(
    ctx: &ExExContext<Node>,
    options: &RunOptions,
    pair: &mut LivePair,
    gate: &mut BootstrapGate,
    mut recorder: Option<&mut recorder::StreamRecorder>,
    block: &RecoveredBlock<BlockTy<EthPrimitives>>,
    sidecar: Option<&PartialStatelessSidecar>,
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
            sidecar,
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

    // The self-test keeps the synchronous export; everything else exports off-task. The poll
    // runs before a new attempt can start, so a checkpoint that completed during this block
    // flushes before the next block buffers on top of it.
    if options.bootstrap_self_test_blocks > 0 {
        return advance_self_test_export(ctx, options, pair, gate)
    }
    poll_export_job(options, gate, &mut recorder);
    maybe_start_export(ctx, options, pair, gate, recorder);
    Ok(())
}

/// The self-test's synchronous export: the pair is frozen at H for the export's whole duration,
/// which is exactly what lets the restored shadow be compared against it fingerprint-for-
/// fingerprint on the same block. Recording runs never come here (`RunOptions::from_env`).
fn advance_self_test_export<Node>(
    ctx: &ExExContext<Node>,
    options: &RunOptions,
    pair: &mut LivePair,
    gate: &mut BootstrapGate,
) -> eyre::Result<()>
where
    Node: FullNodeComponents<Types: NodeTypes<Primitives = EthPrimitives>>,
{
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
                options.stream_fsync,
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

    {
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

/// Removes export attempt directories a previous process left behind.
///
/// Safe exactly at startup: no worker of this process exists yet, and an attempt directory is
/// only ever promoted by the run that created it, so anything present now is an abandoned
/// attempt's leavings. `complete_export` removes the promoted attempt's own directory; fenced
/// attempts have nothing else that ever would.
fn remove_stale_attempt_dirs(bootstrap_dir: &Path) {
    let Ok(entries) = std::fs::read_dir(bootstrap_dir) else { return };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        // Exactly the names this process's own export machinery mints: `export-attempt-<digits>`.
        // A prefix match would also claim an operator's `export-attempt-notes/`, and deleting
        // anything this process did not create is not cleanup.
        let stale = path.is_dir() &&
            path.file_name()
                .and_then(|name| name.to_str())
                .and_then(|name| name.strip_prefix("export-attempt-"))
                .is_some_and(|digits| {
                    !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
                });
        if !stale {
            continue
        }
        match std::fs::remove_dir_all(&path) {
            Ok(()) => info!(
                target: "partial_stateless",
                dir = %path.display(),
                "Removed a stale export attempt directory from a previous run"
            ),
            Err(err) => warn!(
                target: "partial_stateless",
                dir = %path.display(),
                %err,
                "Could not remove a stale export attempt directory"
            ),
        }
    }
}

/// Checks the running export: a completed one opens the stream, a failed one is retried or ends
/// it, and a buffer overflow fails the attempt even while the worker is still running.
fn poll_export_job(
    options: &RunOptions,
    gate: &mut BootstrapGate,
    recorder: &mut Option<&mut recorder::StreamRecorder>,
) {
    // Overflow first: once the buffer is gone the stream cannot start contiguously at this
    // attempt's H + 1, and waiting minutes for the multiproof to finish buys nothing.
    if recorder.as_mut().is_some_and(|recorder| recorder.take_buffer_overflow()) &&
        matches!(gate.job, ExportJob::InFlight { .. })
    {
        // Dropping the receiver abandons the worker: its send fails silently and its files sit
        // in an attempt directory nothing promotes.
        gate.job = ExportJob::Idle;
        fail_export_attempt(gate, recorder, "the export buffer overflowed");
        return
    }

    let received = match &gate.job {
        ExportJob::InFlight { rx, .. } => match rx.try_recv() {
            Err(mpsc::TryRecvError::Empty) => return,
            received => received,
        },
        _ => return,
    };
    let ExportJob::InFlight { accepted_head, attempt_dir, started, .. } =
        std::mem::replace(&mut gate.job, ExportJob::Idle)
    else {
        return
    };
    match received {
        Ok(Ok(finished)) => {
            complete_export(options, gate, recorder, finished, accepted_head, attempt_dir, started)
        }
        Ok(Err(err)) => {
            fail_export_attempt(gate, recorder, &format!("snapshot export failed: {err:#}"))
        }
        Err(_) => fail_export_attempt(
            gate,
            recorder,
            "the export worker exited without a result (panic or early drop)",
        ),
    }
}

/// Retires a failed attempt: a fresh H at the next Ready while retries remain, `End` after.
fn fail_export_attempt(
    gate: &mut BootstrapGate,
    recorder: &mut Option<&mut recorder::StreamRecorder>,
    why: &str,
) {
    if let Some(recorder) = recorder.as_mut() {
        recorder.abandon_buffering(why);
    }
    if gate.retries_left > 0 {
        gate.retries_left -= 1;
        gate.export_pending = true;
        // `pending_cause` is left as it was: a retry re-runs the same kind of attempt.
        gate.emit_event(
            "export_failed",
            serde_json::json!({
                "why": why,
                "retries_left": gate.retries_left,
                "stream_ended": false,
                "cause_id": gate.attempt_cause_id,
            }),
        );
        warn!(
            target: "partial_stateless",
            why,
            retries_left = gate.retries_left,
            "Snapshot export attempt failed; a fresh H will be chosen at the next Ready"
        );
    } else {
        // A write-through re-checkpoint that ran out of retries leaves a stream that is complete
        // and contiguous on disk — only the recovery checkpoint is missing. Closing it with
        // `End(ExportFailure)` would discard a healthy record to report a telemetry-grade
        // failure, so the failure is reported out of band and the stream stays open. The
        // checkpoint-first causes keep the `End`: their buffered branch is gone with the attempt,
        // and a spool that cannot be restored from should say so terminally.
        let opened = recorder.as_ref().is_some_and(|recorder| recorder.stream_opened());
        let write_through_attempt = opened && gate.attempt_cause == RecheckpointCause::BranchChange;
        if !write_through_attempt && let Some(recorder) = recorder.as_mut() {
            recorder.write_end(
                EndKind::ExportFailure,
                format!("snapshot export failed with no retries left: {why}"),
            );
        }
        gate.emit_event(
            "export_failed",
            serde_json::json!({
                "why": why,
                "retries_left": 0,
                "stream_ended": !write_through_attempt,
                "cause_id": gate.attempt_cause_id,
            }),
        );
        if write_through_attempt {
            error!(
                target: "partial_stateless",
                why,
                "Snapshot re-checkpoint failed with no retries left; the live stream stays open \
                 without a fresh recovery checkpoint"
            );
        } else {
            error!(
                target: "partial_stateless",
                why,
                "Snapshot export failed with no retries left; this stream records no commits"
            );
        }
    }
}

/// Whether an export runs behind a live stream, commits writing through ahead of its checkpoint.
///
/// True only for a branch change on an already-open stream: the pair recovered in place, so a
/// consumer holding the retained generation can verify every winning-branch commit without the
/// checkpoint, which then publishes at the stream tail. The stream-opening export and every
/// discontinuity keep the checkpoint-first ordering — commits alone cannot carry a consumer
/// across either.
fn export_writes_through(
    cause: RecheckpointCause,
    recorder: Option<&recorder::StreamRecorder>,
) -> bool {
    cause == RecheckpointCause::BranchChange &&
        recorder.is_some_and(recorder::StreamRecorder::stream_opened)
}

/// Abandons whatever export is armed or running, and decides whether to arm a fresh one.
///
/// Called when the chain moves under an export: a reorg, a revert, or a discontinuity the pair
/// recovered from by rebuilding or resetting. The attempt in flight chose its H before any of
/// that happened, so its checkpoint may describe a block that is no longer canonical — and a
/// checkpoint on an abandoned branch is worse than no checkpoint, because a consumer would
/// restore from it and believe it was on the winning chain.
///
/// Dropping the receiver is the whole fence. The blocking task cannot be cancelled, but
/// `complete_export` is only reachable through the current job's receiver, and each worker writes
/// into its own attempt directory — so a stale worker finishing minutes later lands somewhere
/// nothing reads. A completed-but-unpolled result is dropped for the same reason rather than
/// taken: it is exactly the race this closes.
fn interrupt_export(
    options: &RunOptions,
    gate: &mut BootstrapGate,
    recorder: &mut Option<&mut recorder::StreamRecorder>,
    cause: RecheckpointCause,
    cause_id: u64,
    why: &str,
) {
    match std::mem::replace(&mut gate.job, ExportJob::Idle) {
        ExportJob::InFlight { attempt_dir, started, .. } => {
            warn!(
                target: "partial_stateless",
                why,
                attempt = gate.attempt,
                attempt_dir = %attempt_dir.display(),
                elapsed_ms = started.elapsed().as_millis() as u64,
                "Abandoning a snapshot export the chain moved under; its files are left where \
                 nothing will promote them"
            );
            let elapsed_ms = started.elapsed().as_millis() as u64;
            // The fenced attempt belongs to the cause that armed it, not to the incoming one.
            gate.emit_event(
                "export_fenced",
                serde_json::json!({
                    "why": why,
                    "elapsed_ms": elapsed_ms,
                    "cause_id": gate.attempt_cause_id,
                    "fenced_by_cause_id": cause_id,
                }),
            );
        }
        // Terminal until now: one stream, one checkpoint. Reopening it is what lets the producer
        // publish a second one at the block a reorg recovered to.
        ExportJob::Finished | ExportJob::Idle => {}
    }
    if let Some(recorder) = recorder.as_mut() {
        // Consumed here or the next poll would fail a fresh attempt for the abandoned one's
        // overflow.
        let _ = recorder.take_buffer_overflow();
        recorder.abandon_buffering(why);
    }
    // `retries_left` is untouched: this attempt did not fail, it was overtaken. Retries meter
    // genuine export failures, and spending one here would close a stream over a healthy chain.
    let opened = recorder.as_ref().is_some_and(|recorder| recorder.stream_opened());
    gate.export_pending = options.bootstrap_export &&
        (!opened || options.reorg_checkpoint == ReorgCheckpointPolicy::Always);
    // The cause travels with the armed export: it is what decides whether the next attempt's
    // commits buffer behind its checkpoint or write through ahead of it. Its id travels too,
    // so the attempt this arms is attributable to this exact lifecycle event.
    gate.pending_cause = cause;
    gate.pending_cause_id = cause_id;
    // Every cause announces its own arming, discontinuities included — they have no detection
    // event of their own, and without this line an armed-but-never-attempted cause would be
    // invisible to the event-log reducer that judges quiescence.
    gate.emit_event(
        "recheckpoint_armed",
        serde_json::json!({
            "cause": cause.as_str(),
            "cause_id": cause_id,
            "armed": gate.export_pending,
        }),
    );
    gate.waiting_for_accepted_head_logged = false;
    if !gate.export_pending {
        info!(
            target: "partial_stateless",
            why,
            "The stream is not re-checkpointed: PS_STREAM_REORG_CHECKPOINT=never"
        );
    }
}

/// Opens the stream behind a completed export: promote, read back, checkpoint, flush.
fn complete_export(
    options: &RunOptions,
    gate: &mut BootstrapGate,
    recorder: &mut Option<&mut recorder::StreamRecorder>,
    finished: bootstrap_io::FinishedExport,
    accepted_head: SealedHeader,
    attempt_dir: PathBuf,
    started: Instant,
) {
    // Promote the attempt's files onto the fixed operator paths. Only the accepted attempt is
    // ever promoted, which is what makes a stale worker finishing minutes late harmless: it
    // writes into a directory nothing reads.
    let package_path = options.bootstrap_dir.join(bootstrap_io::PACKAGE_FILE);
    let checkpoint_path = options.bootstrap_dir.join(bootstrap_io::CHECKPOINT_FILE);
    if let Err(err) = std::fs::rename(&finished.package_path, &package_path)
        .and_then(|()| std::fs::rename(&finished.checkpoint_path, &checkpoint_path))
    {
        fail_export_attempt(
            gate,
            recorder,
            &format!("could not promote the exported snapshot onto the operator paths: {err}"),
        );
        return
    }
    let _ = std::fs::remove_dir(&attempt_dir);
    info!(
        target: "partial_stateless",
        block = finished.checkpoint.block_number,
        block_hash = ?finished.checkpoint.block_hash,
        package_bytes = finished.package_bytes,
        proof_targets = finished.proof_targets,
        export_us = finished.elapsed_us,
        wall_ms = started.elapsed().as_millis() as u64,
        "Off-task snapshot export completed; the pair kept advancing throughout"
    );
    gate.emit_event(
        "export_completed",
        serde_json::json!({
            "block": finished.checkpoint.block_number,
            "block_hash": format!("{:?}", finished.checkpoint.block_hash),
            "package_bytes": finished.package_bytes,
            "proof_targets": finished.proof_targets,
            "export_us": finished.elapsed_us,
            "wall_ms": started.elapsed().as_millis() as u64,
            "cause_id": gate.attempt_cause_id,
        }),
    );

    // Written from the file rather than from the in-memory package, so the stream carries exactly
    // the bytes an operator would ship and a mismatch between the two cannot hide here. The
    // accepted head is the one captured at H — the pair's own has advanced past it.
    if recorder.is_some() {
        let package = match std::fs::read(&package_path) {
            Ok(package) => package,
            Err(err) => {
                fail_export_attempt(
                    gate,
                    recorder,
                    &format!("could not read back the exported snapshot: {err}"),
                );
                return
            }
        };
        let publication = recorder.as_mut().and_then(|recorder| {
            recorder.write_checkpoint(&finished.checkpoint, Some(&accepted_head), &package)
        });
        // Judged from the recorder's answer, never from the call having been made: a poisoned or
        // closed recorder no-ops the write silently.
        match publication {
            Some(publication) => gate.emit_event(
                "checkpoint_published",
                serde_json::json!({
                    "block": finished.checkpoint.block_number,
                    "announce_sequence": publication.announce_sequence,
                    "chunks": publication.chunks,
                    "flushed_commits": publication.flushed_commits,
                    "announce_to_complete_us": publication.announce_to_complete_us,
                    "cause_id": gate.attempt_cause_id,
                }),
            ),
            None => gate.emit_event(
                "checkpoint_publication_skipped",
                serde_json::json!({
                    "block": finished.checkpoint.block_number,
                    "cause_id": gate.attempt_cause_id,
                }),
            ),
        }
    }
    gate.job = ExportJob::Finished;
}

/// Starts an off-task export at `Ready(H)`, capturing everything the worker needs on this task.
///
/// The capture is the cheap half — a plain-data copy of the flat cache, the Ready parent, and
/// the accepted head — and it happens here so the worker's view is exactly H no matter how far
/// the pair advances while the multiproof runs. The accepted head is *required*: a live stream's
/// checkpoint without its own header would leave a standalone consumer unable to admit H + 1,
/// and `NoAcceptedParent` is a rejection, not a wait.
fn maybe_start_export<Node>(
    ctx: &ExExContext<Node>,
    options: &RunOptions,
    pair: &LivePair,
    gate: &mut BootstrapGate,
    mut recorder: Option<&mut recorder::StreamRecorder>,
) where
    Node: FullNodeComponents<Types: NodeTypes<Primitives = EthPrimitives>>,
{
    if !gate.export_pending || !matches!(gate.job, ExportJob::Idle) {
        return
    }
    if !gate.worker_slot_available() {
        return
    }
    let Some(ready) = pair.readiness.ready_parent().cloned() else { return };
    let Some(accepted_head) = pair.accepted_parent().cloned() else {
        // Ready with no accepted head happens one block out of a snapshot restore or a cold
        // reset; the next applied block supplies one, so this is a wait rather than a failure.
        if !gate.waiting_for_accepted_head_logged {
            info!(
                target: "partial_stateless",
                block = ready.anchor.block_number,
                "Ready without an accepted head; the snapshot export waits for the next block"
            );
            gate.waiting_for_accepted_head_logged = true;
        }
        return
    };
    gate.export_pending = false;
    gate.waiting_for_accepted_head_logged = false;
    gate.attempt += 1;
    let cause = gate.pending_cause;
    gate.attempt_cause = cause;
    gate.attempt_cause_id = gate.pending_cause_id;
    let write_through = export_writes_through(cause, recorder.as_deref());

    let capture_started = Instant::now();
    let state = CacheState::from_cache(&pair.cache);
    let cache_capture_us = capture_started.elapsed().as_micros() as u64;
    let provider_started = Instant::now();
    // The snapshot's proof must be answered against the state the Ready parent names. The
    // provider is opened here and moved into the worker, so its read transaction spans the
    // multiproof — off this task, but still one long transaction; shortening it is deferred
    // hardening rather than a requirement of the live follower.
    let provider = match ctx.provider().history_by_block_hash(ready.anchor.block_hash) {
        Ok(provider) => provider,
        Err(err) => {
            fail_export_attempt(
                gate,
                &mut recorder,
                &format!("no state provider at the export anchor: {err}"),
            );
            return
        }
    };
    let provider_open_us = provider_started.elapsed().as_micros() as u64;

    let attempt_dir = options.bootstrap_dir.join(format!("export-attempt-{}", gate.attempt));
    let worker_dir = attempt_dir.clone();
    let config = options.config;
    let fsync = options.stream_fsync;
    let worker_ready = ready.clone();
    let (tx, rx) = mpsc::sync_channel(1);
    let slot = WorkerSlot::occupy(gate.live_workers.clone());
    tokio::task::spawn_blocking(move || {
        // Held for the closure's whole life: the drop guard is what tells the gauge this worker
        // — fenced or not — has actually released its read transaction.
        let _slot = slot;
        let result = bootstrap_io::export_snapshot_from_state(
            &worker_dir,
            state,
            &worker_ready,
            &config,
            provider.as_ref(),
            fsync,
        )
        .map(bootstrap_io::FinishedExport::from);
        // A dropped receiver means the attempt was abandoned; the files stay in the attempt
        // directory, unpromoted and inert.
        let _ = tx.send(result);
    });
    if write_through {
        info!(
            target: "partial_stateless",
            block = ready.anchor.block_number,
            block_hash = ?ready.anchor.block_hash,
            attempt = gate.attempt,
            cause = cause.as_str(),
            cache_capture_us,
            provider_open_us,
            "Snapshot export started off-task; commits write through and its checkpoint will \
             publish at the stream tail"
        );
    } else {
        info!(
            target: "partial_stateless",
            block = ready.anchor.block_number,
            block_hash = ?ready.anchor.block_hash,
            attempt = gate.attempt,
            cause = cause.as_str(),
            cache_capture_us,
            provider_open_us,
            "Snapshot export started off-task; frames buffer until its checkpoint lands"
        );
    }
    gate.emit_event(
        "export_started",
        serde_json::json!({
            "block": ready.anchor.block_number,
            "block_hash": format!("{:?}", ready.anchor.block_hash),
            "cause": cause.as_str(),
            "cause_id": gate.attempt_cause_id,
            "write_through": write_through,
            "cache_capture_us": cache_capture_us,
            "provider_open_us": provider_open_us,
        }),
    );
    if !write_through && let Some(recorder) = recorder {
        recorder.begin_buffering();
    }
    gate.job = ExportJob::InFlight { rx, accepted_head, attempt_dir, started: Instant::now() };
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
    log_readiness_change(pair, block, before, after);
}

/// Logs the one readiness transition the run checklist reads: exactly one move to ready.
fn log_readiness_change(
    pair: &LivePair,
    block: &BlockContext,
    before: &'static str,
    after: &'static str,
) {
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
        CacheTrieRepr, CanonicalStateRoots, CoordinatedPair, LivePair, PartialTrieNodeCache,
        ProviderResult, SealedHeader, SidecarRole,
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
    fn a_policy_mismatch_is_refused_before_any_mutation() {
        // The rejection that used to arrive too late. The tracker checks the policy id only after
        // it has reset itself, and it was reached only after the flat cache had been rolled back
        // and the trie replaced — so a pair configured against one policy and handed a checkpoint
        // naming another ended up at neither generation. A caller with a database can throw such a
        // pair away; the standalone validator this path exists for cannot.
        let (mut pair, _config, state_root) = pair_one_block_past_a_snapshot();
        let before = pair.fingerprint();
        let lifecycle_before = pair.lifecycle_fingerprint();
        let state_before = pair.readiness.state().label();

        assert!(
            pair.restore_retained_generation(SNAP_HASH, state_root, B256::repeat_byte(0x77))
                .is_none(),
            "a checkpoint naming a policy this pair is not running is not restorable"
        );

        assert_eq!(pair.fingerprint(), before, "both caches are where the refusal found them");
        assert_eq!(pair.lifecycle_fingerprint(), lifecycle_before, "and so is the retention");
        assert_eq!(pair.readiness.state().label(), state_before, "and the tracker was not reset");
        assert!(pair.previous_generation.is_some(), "the retention is still the caller's to use");
    }

    #[test]
    fn a_pruned_undo_log_is_refused_without_mutation() {
        // The retained trie reaches one block back; the flat undo log is pruned at finality. When
        // the two disagree the undo is not available, and the pair must learn that before it moves
        // rather than from a rollback that refuses half-way through the generation swap.
        let (mut pair, config, state_root) = pair_one_block_past_a_snapshot();
        pair.cache.prune_undo_below(SNAP_BLOCK + 1);
        let before = pair.fingerprint();
        let state_before = pair.readiness.state().label();

        assert!(
            pair.restore_retained_generation(SNAP_HASH, state_root, config.cache_policy_id())
                .is_none(),
            "there is no undo record to give the block back with"
        );

        assert_eq!(pair.fingerprint(), before, "nothing moved");
        assert_eq!(pair.readiness.state().label(), state_before);
        assert!(pair.previous_generation.is_some(), "and the retention was not consumed");
    }

    #[test]
    fn a_cold_reset_readmission_heals_the_acknowledgement_watermark() {
        // The whole path, through the code the notification handler actually runs. A block that
        // could not be admitted freezes the watermark below it, and the recovery available to
        // this producer is a cold reset that re-executes that very block. Until the tracker
        // recognised that, every run that hit one gap reported a durability watermark stuck at
        // the block before it for as long as the node stayed up — and a consumer resuming from
        // that watermark would have replayed from far behind where the producer actually was.
        let mut pair = cold_pair();
        for number in 100..=102 {
            process(&mut pair, number);
        }
        assert_eq!(pair.readiness.acknowledgeable_height().map(|(number, _)| number), Some(102));

        // 103 is delivered and never applied, which is what `abandon_block` records.
        pair.readiness.abandon_block(103);
        assert_eq!(pair.readiness.first_gap(), Some(103));

        // The handler's own recovery: `admit` cold-resets and readmits, and the block that was
        // missed is the one that arrives next.
        process(&mut pair, 103);
        process(&mut pair, 104);

        assert_eq!(pair.readiness.first_gap(), None, "103 was executed after all");
        assert_eq!(
            pair.readiness.acknowledgeable_height().map(|(number, _)| number),
            Some(104),
            "and the watermark moves with the chain again"
        );
    }

    #[test]
    fn an_unrooted_parent_is_refused_without_mutation() {
        // The one refusal the transaction cannot make good on after the fact: the post-undo cache
        // root has to be known *before* the rollback for the tracker to be run on a clone, and the
        // undo record carries it only if the parent's root was ever computed. Production computes
        // it every block, so this is a fixture built to reach the branch — and what it pins is
        // that the branch is a refusal, not a half-applied generation swap.
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
        // Restoring computed the anchor's cache root, which is what the *next* block would carry
        // into its undo record. Spending it on a block that is not the one being undone leaves the
        // record below with nothing to restore the root from.
        apply(&mut pair, SNAP_BLOCK);
        apply(&mut pair, SNAP_BLOCK + 1);
        assert_eq!(
            pair.cache.undo_preview().expect("a record exists").previous_cache_root,
            None,
            "the fixture only means anything if it reached the branch"
        );
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
            true,
        );
        let before = pair.fingerprint();
        let lifecycle_before = pair.lifecycle_fingerprint();
        let state_before = pair.readiness.state().label();

        assert!(
            pair.restore_retained_generation(SNAP_HASH, state_root, config.cache_policy_id())
                .is_none(),
            "an undo whose result cannot be predicted is not one this pair may start"
        );

        assert_eq!(pair.fingerprint(), before, "both caches are where the refusal found them");
        assert_eq!(pair.lifecycle_fingerprint(), lifecycle_before, "and so is the retention");
        assert_eq!(pair.readiness.state().label(), state_before, "and the tracker was not reset");
        assert!(pair.previous_generation.is_some(), "the retention is still the caller's to use");
    }

    #[test]
    fn a_refused_warming_undo_keeps_the_retained_generation() {
        // A warming pair is refused because it has no `Ready` to return to — but the retention it
        // holds still describes the branch the caller named, so a later attempt against a warmer
        // pair is entitled to it. Only a retention tagged with a *different* block is dropped.
        let config = CacheConfig::default();
        let (package, checkpoint, state_root) = warm_snapshot(&config);
        let retained = crate::bootstrap_io::restore_snapshot(package, &checkpoint, &config)
            .expect("an honest snapshot restores");
        let mut pair = cold_pair();
        for number in (SNAP_BLOCK - 2)..=SNAP_BLOCK {
            process(&mut pair, number);
        }
        apply(&mut pair, SNAP_BLOCK + 1);
        pair.retain_generation(
            Some(retained.trie_cache),
            SNAP_HASH,
            SNAP_BLOCK,
            sealed(&ctx(SNAP_BLOCK + 1)),
            true,
        );

        assert!(pair
            .restore_retained_generation(SNAP_HASH, state_root, config.cache_policy_id())
            .is_none());

        assert_eq!(
            pair.lifecycle_fingerprint().retained_generation,
            Some((SNAP_BLOCK, SNAP_HASH)),
            "the refusal was about this pair's warmth, not about what it retained"
        );
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
    fn a_canonical_header_installed_after_a_rebuild_is_offered_as_a_parent() {
        // What makes the rebuild install a header rather than clear one. A rebuild used to leave
        // the pair headless, which cost it a block before it could admit anything and made the
        // checkpoint it exported unrestorable — a headless checkpoint can never admit H + 1. It
        // now installs the canonical header for the exact hash it rebuilt at. This is the
        // soundness that rests on: `accepted_parent` re-derives every field against the readiness
        // anchor, so an externally supplied header is offered only when it *is* the block this
        // generation is the state after. There is no provider in a unit test, so the fetch is
        // stood in for by taking the same header the provider would have returned.
        let (mut pair, ..) = pair_and_reference_before_a_reorg();
        let canonical = pair.accepted_head.clone().expect("the fixture advanced a block");
        pair.forget_retained_generation();
        pair.accepted_head = None;
        assert_eq!(pair.accepted_parent(), None, "a headless pair offers nothing");

        pair.accepted_head = Some(canonical.clone());

        assert_eq!(
            pair.accepted_parent().map(|header| header.hash()),
            Some(canonical.hash()),
            "the header for the block the rebuild targeted is exactly the parent it may admit on"
        );
        assert_eq!(
            pair.lifecycle_fingerprint().accepted_head,
            Some((canonical.number, canonical.hash()))
        );
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
    fn a_cold_reset_keeps_the_trie_representation() {
        // Cold means empty, not reconfigured: a mid-run gap on a non-default representation must
        // come back on that representation, or an A/B arm would silently finish on the other one.
        let mut pair = cold_pair();
        pair.trie_cache = PartialTrieNodeCache::new_with_repr(CacheTrieRepr::Exact);

        pair.cold_reset();

        assert_eq!(pair.trie_cache.repr(), CacheTrieRepr::Exact);
        assert!(pair.trie_cache.state_root().is_none(), "the reset must still empty the trie");
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

    /// The off-task export's lifecycle, driven without a worker: the gate and the recorder are
    /// what own an attempt's outcome, and the blocking task itself is just a sender.
    mod export_job {
        use super::super::{
            bootstrap_io, export_writes_through, fail_export_attempt, poll_export_job,
            producer_events, recorder::StreamRecorder, remove_stale_attempt_dirs, BootstrapGate,
            EndKind, ExportJob, RecheckpointCause, RunOptions, WorkerSlot,
        };
        use crate::CacheConfig;
        use alloy_primitives::B256;
        use partial_stateless::readiness::TrustedCheckpoint;
        use partial_stateless_stream::{decode_event, FrameKind, FrameLimits, StreamEvent};
        use reth_primitives_traits::SealedHeader;
        use std::{
            fs,
            path::{Path, PathBuf},
            sync::{
                atomic::{AtomicUsize, Ordering},
                mpsc, Arc,
            },
            time::Instant,
        };

        fn temp_dir(name: &str) -> PathBuf {
            let dir =
                std::env::temp_dir().join(format!("ps-export-job-{name}-{}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).expect("temp dir");
            dir
        }

        fn gate_with(job: ExportJob, retries_left: u32) -> BootstrapGate {
            BootstrapGate {
                export_pending: false,
                self_test_blocks: 0,
                shadow: None,
                job,
                retries_left,
                attempt: 1,
                pending_cause: RecheckpointCause::Initial,
                attempt_cause: RecheckpointCause::Initial,
                cause_counter: 0,
                pending_cause_id: 0,
                attempt_cause_id: 0,
                first_winning_commit_pending: false,
                first_winning_commit_cause: 0,
                events: None,
                waiting_for_accepted_head_logged: false,
                live_workers: Arc::new(AtomicUsize::new(0)),
                max_workers: 4,
                worker_cap_logged: false,
            }
        }

        fn in_flight(
            rx: mpsc::Receiver<eyre::Result<bootstrap_io::FinishedExport>>,
            attempt_dir: PathBuf,
        ) -> ExportJob {
            ExportJob::InFlight {
                rx,
                accepted_head: SealedHeader::new_unhashed(alloy_consensus::Header::default()),
                attempt_dir,
                started: Instant::now(),
            }
        }

        /// The cap refuses a spawn while the gauge is full and yields as soon as a slot drains.
        /// The refusal must not consume `export_pending` — that contract lives in
        /// `maybe_start_export`, which returns before touching it.
        #[test]
        fn a_spawn_is_refused_at_the_worker_cap() {
            let mut gate = gate_with(ExportJob::Idle, 1);
            gate.max_workers = 1;
            let slot = WorkerSlot::occupy(gate.live_workers.clone());

            assert!(!gate.worker_slot_available(), "one live worker fills a cap of one");
            drop(slot);
            assert!(gate.worker_slot_available(), "the drained slot reopens the cap");
        }

        /// The fence drops the receiver, not the worker: the gauge keeps counting an abandoned
        /// worker until its own guard drops, because the read transaction it holds is just as
        /// real after the fence as before it.
        #[test]
        fn the_gauge_survives_a_fence_and_decrements_on_completion() {
            let mut gate = gate_with(ExportJob::Idle, 1);
            let slot = WorkerSlot::occupy(gate.live_workers.clone());
            let (_tx, rx) = mpsc::sync_channel::<eyre::Result<bootstrap_io::FinishedExport>>(1);
            gate.job = in_flight(rx, PathBuf::from("unused"));

            // The fence: replace the job, dropping the receiver. The worker (its slot) lives on.
            gate.job = ExportJob::Idle;
            assert_eq!(
                gate.live_workers.load(Ordering::SeqCst),
                1,
                "an abandoned worker still holds its slot"
            );
            drop(slot);
            assert_eq!(gate.live_workers.load(Ordering::SeqCst), 0);
        }

        /// A worker that panics still gives its slot back: the guard drops on unwind, which is
        /// the one exit path a counter incremented after the work would miss.
        #[test]
        fn a_panicking_worker_still_decrements() {
            let gauge = Arc::new(AtomicUsize::new(0));
            let slot = WorkerSlot::occupy(gauge.clone());
            let worker = std::thread::spawn(move || {
                let _slot = slot;
                panic!("simulated worker panic");
            });
            assert!(worker.join().is_err(), "the worker panicked as arranged");
            assert_eq!(gauge.load(Ordering::SeqCst), 0, "the unwind released the slot");
        }

        /// Startup removes what fenced attempts leave behind — and nothing else. Only this moment
        /// may clean: while the process runs, an attempt directory may belong to a live worker.
        #[test]
        fn startup_removes_stale_attempt_dirs() {
            let dir = temp_dir("stale-attempts");
            fs::create_dir_all(dir.join("export-attempt-3")).expect("stale attempt");
            fs::create_dir_all(dir.join("export-attempt-7")).expect("stale attempt");
            fs::create_dir_all(dir.join("keep-me")).expect("unrelated dir");
            fs::write(dir.join("checkpoint.json"), b"{}").expect("operator file");

            remove_stale_attempt_dirs(&dir);

            assert!(!dir.join("export-attempt-3").exists());
            assert!(!dir.join("export-attempt-7").exists());
            assert!(dir.join("keep-me").exists(), "only attempt directories are touched");
            assert!(dir.join("checkpoint.json").exists());
            let _ = fs::remove_dir_all(&dir);
        }

        /// The name filter is exact, not a prefix: this process mints `export-attempt-<digits>`
        /// and nothing else, so an operator's `export-attempt-notes/` is not cleanup material.
        #[test]
        fn cleanup_claims_only_the_names_this_process_mints() {
            let dir = temp_dir("stale-attempts-exact");
            fs::create_dir_all(dir.join("export-attempt-12")).expect("stale attempt");
            fs::create_dir_all(dir.join("export-attempt-notes")).expect("operator dir");
            fs::create_dir_all(dir.join("export-attempt-")).expect("no digits");
            fs::create_dir_all(dir.join("export-attempt-1a")).expect("mixed suffix");

            remove_stale_attempt_dirs(&dir);

            assert!(!dir.join("export-attempt-12").exists(), "digits-only names are claimed");
            assert!(dir.join("export-attempt-notes").exists(), "a word suffix is not ours");
            assert!(dir.join("export-attempt-").exists(), "an empty suffix is not ours");
            assert!(dir.join("export-attempt-1a").exists(), "a mixed suffix is not ours");
            let _ = fs::remove_dir_all(&dir);
        }

        /// Every interrupt announces the cause it arms (`recheckpoint_armed`), and an armed
        /// first-winning measurement terminates in exactly one event — here, `unmeasured`,
        /// because the branch change supersedes it. Without the announcement, an
        /// armed-but-never-attempted discontinuity cause is invisible to the quiesce reducer;
        /// without the termination, the stale pending flag stamps the old cause id onto the
        /// next unrelated published commit.
        #[test]
        fn an_interrupt_announces_its_cause_and_terminates_a_pending_measurement() {
            let dir = temp_dir("interrupt-events");
            let spool = dir.join("spool");
            fs::create_dir_all(&spool).expect("spool dir");
            let options = options_with_bootstrap_dir(&dir);
            let mut gate = gate_with(ExportJob::Idle, 1);
            gate.events = Some(producer_events::ProducerEvents::beside_spool(&spool, 1));
            gate.first_winning_commit_pending = true;
            gate.first_winning_commit_cause = 7;

            gate.abandon_first_winning("superseded_by_branch_change");
            assert!(!gate.first_winning_commit_pending);
            // Idempotent: a second call must not invent a second termination.
            gate.abandon_first_winning("superseded_by_branch_change");

            let cause_id = gate.begin_cause();
            let mut holder: Option<&mut StreamRecorder> = None;
            super::super::interrupt_export(
                &options,
                &mut gate,
                &mut holder,
                RecheckpointCause::Discontinuity,
                cause_id,
                "a discontinuity with no export in flight",
            );

            let raw = fs::read_to_string(dir.join("spool.producer-events.jsonl"))
                .expect("the events file exists");
            let events: Vec<serde_json::Value> =
                raw.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
            assert_eq!(events.len(), 2, "one termination, one arming — nothing else");
            assert_eq!(events[0]["kind"], "first_winning_commit_unmeasured");
            assert_eq!(events[0]["cause_id"], 7);
            assert_eq!(events[0]["reason"], "superseded_by_branch_change");
            assert_eq!(events[1]["kind"], "recheckpoint_armed");
            assert_eq!(events[1]["cause_id"], cause_id);
            assert_eq!(events[1]["cause"], "discontinuity");
            assert_eq!(events[1]["armed"], true);
            // The recovery-lifecycle ingest (`scripts/analyze_follow_bench.py`) keys every cause,
            // attempt and interval on these envelope fields. Renaming one here would not fail
            // anything on this side of the language boundary — it would silently empty a table in
            // the report — so the contract is asserted where the names are written.
            for event in &events {
                for key in
                    ["kind", "epoch", "cause_id", "attempt", "mono_elapsed_us", "observed_at_ms"]
                {
                    assert!(event.get(key).is_some(), "{key} is missing from {event}");
                }
            }
            let _ = fs::remove_dir_all(&dir);
        }

        fn options_with_bootstrap_dir(dir: &Path) -> RunOptions {
            let mut options =
                RunOptions::from_env(CacheConfig::default()).expect("default options");
            options.bootstrap_dir = dir.to_path_buf();
            // Every test in this module is about a run that exports; a run that does not would
            // have no gate to drive. Set here rather than through the environment so the tests
            // do not depend on what the process was started with.
            options.bootstrap_export = true;
            options
        }

        fn spool_kinds(dir: &Path) -> Vec<FrameKind> {
            let mut paths: Vec<PathBuf> = fs::read_dir(dir)
                .expect("spool readable")
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| path.extension().is_some_and(|ext| ext == "frame"))
                .collect();
            paths.sort();
            paths
                .iter()
                .map(|path| {
                    let bytes = fs::read(path).expect("frame readable");
                    let (header, _, _) =
                        decode_event(&bytes, &FrameLimits::default()).expect("frame decodes");
                    header.kind
                })
                .collect()
        }

        fn last_end_kind(dir: &Path) -> EndKind {
            let mut paths: Vec<PathBuf> = fs::read_dir(dir)
                .expect("spool readable")
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| path.extension().is_some_and(|ext| ext == "frame"))
                .collect();
            paths.sort();
            let bytes = fs::read(paths.last().expect("spool not empty")).expect("readable");
            let (_, event, _) =
                decode_event(&bytes, &FrameLimits::default()).expect("frame decodes");
            let StreamEvent::End(end) = event else { panic!("last frame is an End") };
            end.kind
        }

        /// A branch change fences the attempt without spending a retry.
        ///
        /// The two are different events and were never distinguished before, because a stream had
        /// exactly one export and nothing could overtake it. `PS_STREAM_EXPORT_RETRIES` meters
        /// exports that *failed*; spending one because the chain moved would eventually close a
        /// stream over a perfectly healthy producer.
        #[test]
        fn a_branch_change_fences_an_attempt_without_spending_a_retry() {
            let spool = temp_dir("fence-spool");
            let bootstrap = temp_dir("fence-bootstrap");
            let options = options_with_bootstrap_dir(&bootstrap);
            let mut recorder = StreamRecorder::for_tests(&spool, 8);
            recorder
                .write_manifest(1, B256::ZERO, B256::with_last_byte(0x44), 60, 30)
                .expect("a fresh spool takes a manifest");
            recorder.begin_buffering();

            let (tx, rx) = mpsc::sync_channel(1);
            let mut gate = gate_with(in_flight(rx, bootstrap.join("export-attempt-1")), 1);
            let mut holder = Some(&mut recorder);
            let cause_id = gate.begin_cause();
            super::super::interrupt_export(
                &options,
                &mut gate,
                &mut holder,
                RecheckpointCause::BranchChange,
                cause_id,
                "the chain reorged",
            );

            assert!(
                matches!(gate.job, ExportJob::Idle),
                "the receiver is dropped, so the fence holds"
            );
            assert_eq!(gate.retries_left, 1, "a fenced attempt is not a failed one");
            assert!(gate.export_pending, "and a fresh one is armed at the block it recovered to");
            assert_eq!(spool_kinds(&spool), vec![FrameKind::Manifest], "no End: nothing failed");
            drop(tx);
            let _ = fs::remove_dir_all(&spool);
            let _ = fs::remove_dir_all(&bootstrap);
        }

        /// A result already sitting in the channel is dropped rather than taken.
        ///
        /// This is the race the fence exists for: the worker finished, the notification loop had
        /// not polled yet, and the chain moved in between. Promoting that checkpoint would
        /// publish a snapshot of a block that is no longer canonical — and a consumer restoring
        /// from it would believe it was on the winning chain.
        #[test]
        fn a_finished_but_unpolled_attempt_is_never_promoted() {
            let spool = temp_dir("unpolled-spool");
            let bootstrap = temp_dir("unpolled-bootstrap");
            let options = options_with_bootstrap_dir(&bootstrap);
            let attempt_dir = bootstrap.join("export-attempt-1");
            fs::create_dir_all(&attempt_dir).expect("attempt dir");
            let package_path = attempt_dir.join("package.bin");
            let checkpoint_path = attempt_dir.join("checkpoint.json");
            fs::write(&package_path, [1u8; 8]).expect("package");
            fs::write(&checkpoint_path, "{}").expect("checkpoint");

            let mut recorder = StreamRecorder::for_tests(&spool, 8);
            recorder
                .write_manifest(1, B256::ZERO, B256::with_last_byte(0x44), 60, 30)
                .expect("a fresh spool takes a manifest");
            recorder.begin_buffering();

            let (tx, rx) = mpsc::sync_channel(1);
            tx.send(Ok(bootstrap_io::FinishedExport {
                checkpoint: TrustedCheckpoint {
                    block_number: 1,
                    block_hash: B256::ZERO,
                    state_root: B256::ZERO,
                    cache_root: B256::ZERO,
                    cache_policy_id: B256::with_last_byte(0x44),
                },
                package_path: package_path.clone(),
                checkpoint_path: checkpoint_path.clone(),
                package_bytes: 8,
                proof_targets: 1,
                elapsed_us: 1,
            }))
            .expect("the channel takes one");

            let mut gate = gate_with(in_flight(rx, attempt_dir), 1);
            let mut holder = Some(&mut recorder);
            let cause_id = gate.begin_cause();
            super::super::interrupt_export(
                &options,
                &mut gate,
                &mut holder,
                RecheckpointCause::BranchChange,
                cause_id,
                "the chain reorged",
            );

            assert!(
                !options.bootstrap_dir.join(bootstrap_io::PACKAGE_FILE).exists(),
                "the fenced attempt's package was never promoted onto the operator path"
            );
            assert!(package_path.exists(), "it is still in the attempt directory nothing reads");
            assert_eq!(spool_kinds(&spool), vec![FrameKind::Manifest], "and no checkpoint landed");
            drop(tx);
            let _ = fs::remove_dir_all(&spool);
            let _ = fs::remove_dir_all(&bootstrap);
        }

        /// A finished export is terminal no longer: reopening it is what lets the producer publish
        /// a second checkpoint at the block a reorg recovered to.
        #[test]
        fn a_finished_export_is_reopened_by_a_branch_change() {
            let spool = temp_dir("reopen-spool");
            let bootstrap = temp_dir("reopen-bootstrap");
            let options = options_with_bootstrap_dir(&bootstrap);
            let mut gate = gate_with(ExportJob::Finished, 0);
            let mut holder = None;

            let cause_id = gate.begin_cause();

            super::super::interrupt_export(
                &options,
                &mut gate,
                &mut holder,
                RecheckpointCause::BranchChange,
                cause_id,
                "the chain reorged",
            );

            assert!(matches!(gate.job, ExportJob::Idle));
            assert!(gate.export_pending);
            let _ = fs::remove_dir_all(&spool);
            let _ = fs::remove_dir_all(&bootstrap);
        }

        /// The fence consumes the overflow flag it inherits.
        ///
        /// Latent until re-arming existed: the flag is read once, by whoever polls next. A fresh
        /// attempt started after a fence would otherwise be failed at its first poll for an
        /// overflow that belonged to the attempt the fence already threw away.
        #[test]
        fn the_fence_consumes_a_stale_overflow_flag() {
            let spool = temp_dir("stale-overflow-spool");
            let bootstrap = temp_dir("stale-overflow-bootstrap");
            let options = options_with_bootstrap_dir(&bootstrap);
            let mut recorder = StreamRecorder::for_tests(&spool, 1);
            recorder
                .write_manifest(1, B256::ZERO, B256::with_last_byte(0x44), 60, 30)
                .expect("a fresh spool takes a manifest");
            recorder.begin_buffering();
            recorder.write_reset(partial_stateless_stream::ResetReason::Gap, "fits");
            recorder.write_reset(partial_stateless_stream::ResetReason::Gap, "overflows");

            let mut gate = gate_with(ExportJob::Idle, 1);
            let mut holder = Some(&mut recorder);
            let cause_id = gate.begin_cause();
            super::super::interrupt_export(
                &options,
                &mut gate,
                &mut holder,
                RecheckpointCause::BranchChange,
                cause_id,
                "the chain reorged",
            );

            assert!(
                !recorder.take_buffer_overflow(),
                "the abandoned attempt's overflow does not fail the next one"
            );
            let _ = fs::remove_dir_all(&spool);
            let _ = fs::remove_dir_all(&bootstrap);
        }

        /// `never` still arms the *first* checkpoint: a stream that never opened has nothing to
        /// re-checkpoint, and leaving it closed would strand a spool with no restorable point.
        #[test]
        fn the_never_policy_still_opens_a_stream_that_never_opened() {
            let spool = temp_dir("never-first-spool");
            let bootstrap = temp_dir("never-first-bootstrap");
            let mut options = options_with_bootstrap_dir(&bootstrap);
            options.reorg_checkpoint = super::super::ReorgCheckpointPolicy::Never;
            let mut recorder = StreamRecorder::for_tests(&spool, 8);
            recorder
                .write_manifest(1, B256::ZERO, B256::with_last_byte(0x44), 60, 30)
                .expect("a fresh spool takes a manifest");
            recorder.begin_buffering();

            let mut gate = gate_with(ExportJob::Idle, 1);
            let mut holder = Some(&mut recorder);
            let cause_id = gate.begin_cause();
            super::super::interrupt_export(
                &options,
                &mut gate,
                &mut holder,
                RecheckpointCause::BranchChange,
                cause_id,
                "the chain reorged",
            );

            assert!(gate.export_pending, "the stream has no checkpoint yet, so one is still owed");
            let _ = fs::remove_dir_all(&spool);
            let _ = fs::remove_dir_all(&bootstrap);
        }

        /// And `never` declines the second one, which is the control the default is measured
        /// against: the consumer is left to recover on its own or not at all.
        #[test]
        fn the_never_policy_declines_to_recheckpoint_an_open_stream() {
            let spool = temp_dir("never-second-spool");
            let bootstrap = temp_dir("never-second-bootstrap");
            let mut options = options_with_bootstrap_dir(&bootstrap);
            options.reorg_checkpoint = super::super::ReorgCheckpointPolicy::Never;
            let mut recorder = StreamRecorder::for_tests(&spool, 8);
            recorder
                .write_manifest(1, B256::ZERO, B256::with_last_byte(0x44), 60, 30)
                .expect("a fresh spool takes a manifest");
            recorder.begin_buffering();
            let checkpoint = TrustedCheckpoint {
                block_number: 25_737_234,
                block_hash: B256::with_last_byte(0x11),
                state_root: B256::with_last_byte(0x22),
                cache_root: B256::with_last_byte(0x33),
                cache_policy_id: B256::with_last_byte(0x44),
            };
            recorder.write_checkpoint(&checkpoint, None, &[7u8; 32]);
            assert!(recorder.stream_opened());

            let mut gate = gate_with(ExportJob::Finished, 1);
            let mut holder = Some(&mut recorder);
            let cause_id = gate.begin_cause();
            super::super::interrupt_export(
                &options,
                &mut gate,
                &mut holder,
                RecheckpointCause::BranchChange,
                cause_id,
                "the chain reorged",
            );

            assert!(!gate.export_pending);
            assert!(recorder.wants_commit_material(), "the open stream keeps taking frames");
            let _ = fs::remove_dir_all(&spool);
            let _ = fs::remove_dir_all(&bootstrap);
        }

        /// An unrecognised policy is refused at startup rather than defaulted, so a run cannot
        /// report a policy it is not running.
        #[test]
        fn an_unknown_reorg_checkpoint_policy_is_a_startup_error() {
            assert!(matches!(
                super::super::ReorgCheckpointPolicy::parse(Some("always")),
                Ok(super::super::ReorgCheckpointPolicy::Always)
            ));
            assert!(matches!(
                super::super::ReorgCheckpointPolicy::parse(None),
                Ok(super::super::ReorgCheckpointPolicy::Always)
            ));
            assert!(matches!(
                super::super::ReorgCheckpointPolicy::parse(Some("never")),
                Ok(super::super::ReorgCheckpointPolicy::Never)
            ));
            let err = super::super::ReorgCheckpointPolicy::parse(Some("deep-only"))
                .expect_err("an unknown policy is refused");
            assert!(err.to_string().contains("deep-only"), "{err}");
        }

        /// A worker that died (panicked or dropped its sender) is a failed attempt; while a
        /// retry remains, the gate re-arms so the next Ready chooses a fresh H.
        #[test]
        fn a_dead_worker_rearms_the_export_while_retries_remain() {
            let spool = temp_dir("dead-worker-spool");
            let bootstrap = temp_dir("dead-worker-bootstrap");
            let options = options_with_bootstrap_dir(&bootstrap);
            let mut recorder = StreamRecorder::for_tests(&spool, 8);
            recorder
                .write_manifest(1, B256::ZERO, B256::with_last_byte(0x44), 60, 30)
                .expect("a fresh spool takes a manifest");
            recorder.begin_buffering();

            let (tx, rx) = mpsc::sync_channel(1);
            drop(tx);
            let mut gate = gate_with(in_flight(rx, bootstrap.join("export-attempt-1")), 1);
            let mut holder = Some(&mut recorder);
            poll_export_job(&options, &mut gate, &mut holder);

            assert!(gate.export_pending, "a fresh H will be chosen at the next Ready");
            assert_eq!(gate.retries_left, 0);
            assert!(matches!(gate.job, ExportJob::Idle));
            assert!(!recorder.wants_commit_material(), "the buffer was abandoned whole");
            assert_eq!(spool_kinds(&spool), vec![FrameKind::Manifest], "no End yet");
            let _ = fs::remove_dir_all(&spool);
            let _ = fs::remove_dir_all(&bootstrap);
        }

        /// With no retries left, a failed attempt closes the stream as `ExportFailure`: an
        /// intentionally closed empty stream, not a cut one.
        #[test]
        fn an_exhausted_export_closes_the_stream_as_an_export_failure() {
            let spool = temp_dir("exhausted-spool");
            let mut recorder = StreamRecorder::for_tests(&spool, 8);
            recorder
                .write_manifest(1, B256::ZERO, B256::with_last_byte(0x44), 60, 30)
                .expect("a fresh spool takes a manifest");
            recorder.begin_buffering();

            let mut gate = gate_with(ExportJob::Idle, 0);
            let mut holder = Some(&mut recorder);
            fail_export_attempt(&mut gate, &mut holder, "test failure");

            assert!(!gate.export_pending, "no further attempt is armed");
            assert_eq!(spool_kinds(&spool), vec![FrameKind::Manifest, FrameKind::End]);
            assert_eq!(last_end_kind(&spool), EndKind::ExportFailure);
            let _ = fs::remove_dir_all(&spool);
        }

        fn opened_recorder(spool: &Path) -> StreamRecorder {
            let mut recorder = StreamRecorder::for_tests(spool, 8);
            recorder
                .write_manifest(1, B256::ZERO, B256::with_last_byte(0x44), 60, 30)
                .expect("a fresh spool takes a manifest");
            recorder.begin_buffering();
            let checkpoint = TrustedCheckpoint {
                block_number: 25_737_234,
                block_hash: B256::with_last_byte(0x11),
                state_root: B256::with_last_byte(0x22),
                cache_root: B256::with_last_byte(0x33),
                cache_policy_id: B256::with_last_byte(0x44),
            };
            recorder
                .write_checkpoint(&checkpoint, None, &[7u8; 32])
                .expect("the opening checkpoint publishes");
            assert!(recorder.stream_opened());
            recorder
        }

        /// The cause gate, three ways: only a branch change on an already-open stream writes
        /// commits through ahead of its checkpoint. The initial export and every discontinuity
        /// keep the checkpoint-first ordering, because commits alone cannot carry a consumer
        /// across either.
        #[test]
        fn only_a_branch_change_on_an_open_stream_writes_through() {
            let spool = temp_dir("write-through-spool");
            let mut recorder = StreamRecorder::for_tests(&spool, 8);
            recorder
                .write_manifest(1, B256::ZERO, B256::with_last_byte(0x44), 60, 30)
                .expect("a fresh spool takes a manifest");
            assert!(
                !export_writes_through(RecheckpointCause::BranchChange, Some(&recorder)),
                "a reorg before the first checkpoint still buffers: nothing is restorable yet"
            );

            let recorder = opened_recorder(&temp_dir("write-through-open-spool"));
            assert!(export_writes_through(RecheckpointCause::BranchChange, Some(&recorder)));
            assert!(
                !export_writes_through(RecheckpointCause::Discontinuity, Some(&recorder)),
                "a rebuild/reset export keeps checkpoint-first ordering on an open stream"
            );
            assert!(!export_writes_through(RecheckpointCause::Initial, Some(&recorder)));
            assert!(
                !export_writes_through(RecheckpointCause::BranchChange, None),
                "no recorder, no stream to write through"
            );
            let _ = fs::remove_dir_all(&spool);
        }

        /// A write-through re-checkpoint that exhausts its retries leaves a stream that is
        /// complete and contiguous on disk; closing it with `End(ExportFailure)` would discard a
        /// healthy record over a telemetry-grade failure. The checkpoint-first causes keep the
        /// terminal `End`.
        #[test]
        fn a_failed_write_through_recheckpoint_leaves_the_stream_open() {
            let spool = temp_dir("wt-fail-spool");
            let mut recorder = opened_recorder(&spool);
            let mut gate = gate_with(ExportJob::Idle, 0);
            gate.attempt_cause = RecheckpointCause::BranchChange;
            let mut holder = Some(&mut recorder);
            fail_export_attempt(&mut gate, &mut holder, "test failure");

            assert!(
                recorder.wants_commit_material(),
                "the live stream stays open without its fresh recovery checkpoint"
            );
            assert!(
                !spool_kinds(&spool).contains(&FrameKind::End),
                "no End frame closes a healthy write-through stream"
            );

            let discontinuity_spool = temp_dir("wt-fail-discontinuity-spool");
            let mut recorder = opened_recorder(&discontinuity_spool);
            let mut gate = gate_with(ExportJob::Idle, 0);
            gate.attempt_cause = RecheckpointCause::Discontinuity;
            let mut holder = Some(&mut recorder);
            fail_export_attempt(&mut gate, &mut holder, "test failure");

            assert_eq!(
                last_end_kind(&discontinuity_spool),
                EndKind::ExportFailure,
                "a failed checkpoint-first attempt still closes terminally: the spool cannot be \
                 restored across its discontinuity"
            );
            let _ = fs::remove_dir_all(&spool);
            let _ = fs::remove_dir_all(&discontinuity_spool);
        }

        /// The fence carries its cause to the attempt it re-arms: a reorg's replacement export
        /// writes through, a discontinuity's buffers.
        #[test]
        fn interrupt_export_arms_the_cause_it_was_given() {
            let bootstrap = temp_dir("cause-bootstrap");
            let options = options_with_bootstrap_dir(&bootstrap);
            let mut gate = gate_with(ExportJob::Finished, 1);
            let mut holder = None;

            let cause_id = gate.begin_cause();

            super::super::interrupt_export(
                &options,
                &mut gate,
                &mut holder,
                RecheckpointCause::Discontinuity,
                cause_id,
                "a reset discontinued the stream under the export",
            );

            assert_eq!(gate.pending_cause, RecheckpointCause::Discontinuity);
            let _ = fs::remove_dir_all(&bootstrap);
        }

        /// Publication is the recorder's answer, not the call: a closed recorder no-ops
        /// `write_checkpoint` silently, and the event log must not report a checkpoint that was
        /// never written.
        #[test]
        fn write_checkpoint_reports_publication_only_when_it_wrote() {
            let spool = temp_dir("publication-spool");
            let mut recorder = StreamRecorder::for_tests(&spool, 8);
            recorder
                .write_manifest(1, B256::ZERO, B256::with_last_byte(0x44), 60, 30)
                .expect("a fresh spool takes a manifest");
            recorder.begin_buffering();
            recorder.write_reset(partial_stateless_stream::ResetReason::Gap, "buffered frame");
            let checkpoint = TrustedCheckpoint {
                block_number: 25_737_234,
                block_hash: B256::with_last_byte(0x11),
                state_root: B256::with_last_byte(0x22),
                cache_root: B256::with_last_byte(0x33),
                cache_policy_id: B256::with_last_byte(0x44),
            };

            let publication = recorder
                .write_checkpoint(&checkpoint, None, &[7u8; 100])
                .expect("an open recorder publishes");
            assert_eq!(publication.announce_sequence, 1, "manifest holds sequence 0");
            assert_eq!(publication.chunks, 2, "100 bytes over 64-byte chunks");
            assert_eq!(publication.flushed_commits, 1, "the buffered reset flushed behind it");

            recorder.write_end(EndKind::Shutdown, "closing");
            assert!(
                recorder.write_checkpoint(&checkpoint, None, &[7u8; 100]).is_none(),
                "a closed recorder reports no publication"
            );
            let _ = fs::remove_dir_all(&spool);
        }

        /// An error propagating out of the notification loop is a producer fault, and it must be
        /// classified *before* the propagation: the recorder's drop runs on both an error return
        /// and reth's shutdown, and would close a crashed producer as an orderly `Shutdown`.
        #[test]
        fn an_error_exit_from_the_loop_is_a_producer_fault_not_a_shutdown() {
            let spool = temp_dir("loop-error-spool");
            let mut recorder = StreamRecorder::for_tests(&spool, 8);
            recorder
                .write_manifest(1, B256::ZERO, B256::with_last_byte(0x44), 60, 30)
                .expect("a fresh spool takes a manifest");

            let err = super::super::fail_producer(
                Some(&mut recorder),
                eyre::eyre!("channel closed"),
                "the notification stream failed",
            );

            assert!(err.to_string().contains("the notification stream failed"), "{err:#}");
            assert_eq!(last_end_kind(&spool), EndKind::ProducerFault);
            drop(recorder);
            assert_eq!(
                last_end_kind(&spool),
                EndKind::ProducerFault,
                "the drop must not re-close a classified stream"
            );
            let _ = fs::remove_dir_all(&spool);
        }

        /// A buffer overflow fails the attempt even though the worker is still running: the
        /// stream can no longer start contiguously at this attempt's H + 1, and the abandoned
        /// worker's files land in an attempt directory nothing promotes.
        #[test]
        fn a_buffer_overflow_fails_the_attempt_while_the_worker_still_runs() {
            let spool = temp_dir("overflow-spool");
            let bootstrap = temp_dir("overflow-bootstrap");
            let options = options_with_bootstrap_dir(&bootstrap);
            let mut recorder = StreamRecorder::for_tests(&spool, 1);
            recorder
                .write_manifest(1, B256::ZERO, B256::with_last_byte(0x44), 60, 30)
                .expect("a fresh spool takes a manifest");
            recorder.begin_buffering();
            recorder.write_reset(partial_stateless_stream::ResetReason::Gap, "fits");
            recorder.write_reset(partial_stateless_stream::ResetReason::Gap, "overflows");

            // The sender stays alive: the worker is still grinding through the multiproof.
            let (_tx, rx) = mpsc::sync_channel::<eyre::Result<bootstrap_io::FinishedExport>>(1);
            let mut gate = gate_with(in_flight(rx, bootstrap.join("export-attempt-1")), 1);
            let mut holder = Some(&mut recorder);
            poll_export_job(&options, &mut gate, &mut holder);

            assert!(matches!(gate.job, ExportJob::Idle), "the receiver was dropped");
            assert!(gate.export_pending, "the attempt is retried with a fresh H");
            assert_eq!(gate.retries_left, 0);
            let _ = fs::remove_dir_all(&spool);
            let _ = fs::remove_dir_all(&bootstrap);
        }

        /// The completion path end to end, with a fabricated worker result: promote onto the
        /// operator paths, read back, checkpoint, and flush the buffered frames behind it in
        /// arrival order.
        #[test]
        fn a_completed_export_promotes_checkpoints_and_flushes_the_buffer() {
            let spool = temp_dir("complete-spool");
            let bootstrap = temp_dir("complete-bootstrap");
            let attempt_dir = bootstrap.join("export-attempt-1");
            fs::create_dir_all(&attempt_dir).expect("attempt dir");
            let package_path = attempt_dir.join(bootstrap_io::PACKAGE_FILE);
            let checkpoint_path = attempt_dir.join(bootstrap_io::CHECKPOINT_FILE);
            fs::write(&package_path, [7u8; 100]).expect("package");
            fs::write(&checkpoint_path, b"{}").expect("checkpoint");

            let options = options_with_bootstrap_dir(&bootstrap);
            let mut recorder = StreamRecorder::for_tests(&spool, 8);
            recorder
                .write_manifest(1, B256::ZERO, B256::with_last_byte(0x44), 60, 30)
                .expect("a fresh spool takes a manifest");
            recorder.begin_buffering();
            recorder.write_reset(partial_stateless_stream::ResetReason::Gap, "buffered");

            let finished = bootstrap_io::FinishedExport {
                checkpoint: TrustedCheckpoint {
                    block_number: 25_737_234,
                    block_hash: B256::with_last_byte(0x11),
                    state_root: B256::with_last_byte(0x22),
                    cache_root: B256::with_last_byte(0x33),
                    cache_policy_id: B256::with_last_byte(0x44),
                },
                package_path,
                checkpoint_path,
                package_bytes: 100,
                proof_targets: 1,
                elapsed_us: 1,
            };
            let (tx, rx) = mpsc::sync_channel(1);
            tx.send(Ok(finished)).expect("send");
            let mut gate = gate_with(in_flight(rx, attempt_dir.clone()), 1);
            let mut holder = Some(&mut recorder);
            poll_export_job(&options, &mut gate, &mut holder);

            assert!(matches!(gate.job, ExportJob::Finished));
            assert!(bootstrap.join(bootstrap_io::PACKAGE_FILE).exists(), "promoted");
            assert!(!attempt_dir.exists(), "the empty attempt directory was removed");
            // 100 bytes at the test chunk size of 64 is two chunks; the buffered reset follows.
            assert_eq!(
                spool_kinds(&spool),
                vec![
                    FrameKind::Manifest,
                    FrameKind::Checkpoint,
                    FrameKind::SnapshotChunk,
                    FrameKind::SnapshotChunk,
                    FrameKind::Reset,
                ]
            );
            assert!(recorder.wants_commit_material(), "the stream is open");
            let _ = fs::remove_dir_all(&spool);
            let _ = fs::remove_dir_all(&bootstrap);
        }
    }
}
