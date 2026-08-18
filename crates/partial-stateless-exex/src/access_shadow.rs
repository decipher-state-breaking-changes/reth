//! Consumption of the Engine's captured access set, and its comparison against re-execution.
//!
//! In `shadow` mode the builder re-executes every block and additionally
//! consumes the artifact the Engine published for it, comparing the two access sets key by key;
//! nothing it produces changes, because the comparison exists only to decide whether the artifact
//! may be *relied* on. In `on` mode it is relied on: [`simulation_from_artifact`] replaces the
//! re-execution outright, and only a sampled fraction of blocks still executes both ways.
//!
//! That sampling is not leftover scaffolding. The comparison needs a second opinion, and `on` mode
//! deletes the very re-execution that provides it, so an unsampled `on` run would run with no
//! oracle at all. Sampling keeps one alive permanently for a proportional share of the win.
//!
//! Both sides run the same extraction function, so this is not testing the extractor. What it
//! tests is whether the two execution paths built the same `State` to extract from — prewarming
//! and the cross-block cache, read-only and reverted access survival, pre-execution changes, and
//! BLOCKHASH observation are all outside what shared code can prove.
//!
//! A miss is not a divergence. Artifacts cannot exist for backfilled blocks, for notifications
//! replayed from the WAL after a restart, or for a sibling already evicted, and the builder falls
//! back to its own execution in every one of those cases. The gate is that among the blocks that
//! *did* hit, divergence is zero.

use crate::rebuild::HistoricalSimulation;
use alloy_primitives::B256;
use partial_stateless::accessed_state::BlockAccessedState;
use reth_ethereum::EthPrimitives;
use reth_evm::execute::BlockExecutionOutput;
use reth_execution_access::{global_handoff, BlockAccessArtifact, HandoffStats, MissReason};
use reth_primitives_traits::NodePrimitives;
use std::{
    fmt::Write as _,
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex, OnceLock,
    },
};
use tracing::{error, info, warn};

/// Divergence samples carried on a single block's report.
const MAX_SAMPLES: usize = 8;

/// Selects the stage-4 shadow sampling rate.
const SHADOW_SAMPLE_VAR: &str = "PS_SHADOW_SAMPLE";

/// One block in fifty, costing about 2% of the item's win to keep the oracle permanently live.
const DEFAULT_SHADOW_SAMPLE: u64 = 50;

/// Removes the Engine's artifact for `block_hash`, if capture is enabled at all.
///
/// Returns `None` when capture is off, which is what separates "not running this experiment" from
/// "running it and missing". Callers must take *before* re-executing, so that the recorded
/// residence is the wait a stage-4 consumer would actually see rather than one inflated by the
/// simulation this stage still performs.
pub fn take_engine_access(block_hash: B256) -> Option<EngineAccessTake> {
    let handoff = global_handoff()?;
    let outcome = handoff.take_outcome(&block_hash);
    let miss_reason = outcome.miss_reason();
    Some(EngineAccessTake { artifact: outcome.artifact(), miss_reason, stats: handoff.stats() })
}

/// Compares a captured artifact against this block's re-execution and records the outcome.
///
/// Logs at `error` on any divergence, because a single mismatched key is enough to disqualify the
/// artifact path: the cache is a function of the access set, so a set that is wrong anywhere
/// produces a witness that is wrong somewhere.
pub fn record_shadow_comparison(
    block_number: u64,
    block_hash: B256,
    parent_hash: B256,
    take: EngineAccessTake,
    simulated: &BlockAccessedState,
    simulated_lowest_block: Option<u64>,
    simulation_us: u64,
) -> ShadowOutcome {
    let totals = ShadowTotals::get();
    let EngineAccessTake { artifact, miss_reason, stats } = take;

    let Some(artifact) = artifact else {
        let blocks = totals.blocks.fetch_add(1, Ordering::Relaxed) + 1;
        let missed = totals.missed.fetch_add(1, Ordering::Relaxed) + 1;
        warn!(
            target: "partial_stateless_shadow",
            block = block_number,
            block_hash = ?block_hash,
            blocks,
            missed,
            queue_depth = stats.queue_depth,
            dropped_capacity = stats.dropped_capacity,
            reason = miss_reason.as_ref().map(MissReason::as_str),
            "No Engine access artifact for this block; the builder re-executed it"
        );
        let outcome = ShadowOutcome {
            hit: false,
            divergence: None,
            coverage: None,
            capture_us: 0,
            miss_reason,
        };
        append_shadow_record(block_number, block_hash, &outcome, simulation_us, &stats);
        return outcome
    };

    let (divergence, coverage) = compare(parent_hash, simulated, simulated_lowest_block, &artifact);
    let blocks = totals.blocks.fetch_add(1, Ordering::Relaxed) + 1;
    let hits = totals.hits.fetch_add(1, Ordering::Relaxed) + 1;

    if divergence.is_empty() {
        info!(
            target: "partial_stateless_shadow",
            block = block_number,
            blocks,
            hits,
            diverged = totals.diverged.load(Ordering::Relaxed),
            accounts = simulated.accounts.len(),
            storage = simulated.storage.len(),
            codes = simulated.codes.len(),
            capture_us = artifact.capture_us,
            simulation_us,
            residence_us = stats.mean_residence_us,
            queue_depth = stats.queue_depth,
            "Engine access artifact matches the builder re-execution exactly"
        );
    } else {
        let diverged = totals.diverged.fetch_add(1, Ordering::Relaxed) + 1;
        error!(
            target: "partial_stateless_shadow",
            block = block_number,
            block_hash = ?block_hash,
            blocks,
            hits,
            diverged,
            %divergence,
            "Engine access artifact diverges from the builder re-execution"
        );
    }

    let outcome = ShadowOutcome {
        hit: true,
        divergence: Some(divergence),
        coverage: Some(coverage),
        capture_us: artifact.capture_us,
        miss_reason: None,
    };
    append_shadow_record(block_number, block_hash, &outcome, simulation_us, &stats);
    outcome
}

/// Rebuilds a block's simulation result from a captured artifact, without re-executing it.
///
/// This is the whole point of engine-access reuse: the Engine already produced every field below,
/// and the only
/// work left is moving them. Returns `None` if the artifact's execution output is missing or of
/// an unexpected type, which the caller must treat as a miss and re-execute -- never as an empty
/// result, since a silently empty output would produce a wrong sidecar rather than a slow one.
///
/// `elapsed_us` is reported as zero because no EVM ran; the artifact's own `capture_us` is what
/// this path cost, and it was paid on the Engine's thread.
pub fn simulation_from_artifact(artifact: BlockAccessArtifact) -> Option<HistoricalSimulation> {
    let output =
        artifact.output::<BlockExecutionOutput<<EthPrimitives as NodePrimitives>::Receipt>>()?;
    let lowest_block_number = artifact.access.lowest_block_hash_number;
    Some(HistoricalSimulation {
        accessed: artifact.access.into(),
        lowest_block_number,
        // Shared, not cloned. The Engine keeps its own reference for the canonical commit, so an
        // owned value here would mean copying the whole `BundleState` on every reused block --
        // the largest remaining cost on a path whose entire purpose is to not pay for the block
        // twice. Every downstream consumer reads through the reference.
        output,
        elapsed_us: 0,
    })
}

/// What became of the Engine's artifact for one block.
///
/// One total value rather than a set of booleans, because the interesting question -- "did the
/// handoff deliver?" -- is not the same as "did the builder skip re-execution?", and independent
/// flags let those two drift into contradiction. A sampled block is a *delivery success* and a
/// *reuse refusal* at the same time, which is exactly the case that makes a single
/// `artifact_reused` field misleading: at the default 1-in-50 sampling, perfect delivery still
/// reads as 98% reuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactDisposition {
    /// Capture is off; no artifact was expected.
    CaptureOff,
    /// The artifact replaced this block's re-execution.
    Reused,
    /// Shadow mode: the artifact arrived and was compared, and the block re-executed anyway.
    Shadowed,
    /// `on` mode, sampled block: the artifact arrived and was compared, deliberately not reused.
    Sampled,
    /// No artifact, for the stated reason.
    Missed(MissReason),
    /// An artifact arrived but its output was not the expected type, so the block re-executed.
    TypeMismatch,
}

impl ArtifactDisposition {
    /// Whether the handoff delivered an artifact. This, not reuse, is the delivery rate.
    ///
    /// [`Self::TypeMismatch`] counts as delivered. The artifact arrived; what failed was the
    /// consumer's downcast, and scoring that as a delivery failure would charge the handoff for
    /// a defect on the other side of it. Delivery and usability are separate questions, and the
    /// second one is answered by [`Self::fallback_reason`], which names `type_mismatch`
    /// explicitly rather than leaving it among the misses.
    pub const fn artifact_available(&self) -> bool {
        matches!(self, Self::Reused | Self::Shadowed | Self::Sampled | Self::TypeMismatch)
    }

    /// Whether the artifact replaced re-execution. This is the share of blocks that got the win.
    pub const fn artifact_reused(&self) -> bool {
        matches!(self, Self::Reused)
    }

    /// Whether this block re-executed on purpose to feed the comparison.
    pub const fn shadow_sampled(&self) -> bool {
        matches!(self, Self::Shadowed | Self::Sampled)
    }

    /// Why the artifact was not reused, or `None` when it was.
    pub const fn fallback_reason(&self) -> Option<&'static str> {
        match self {
            Self::Reused => None,
            Self::CaptureOff => Some("capture_off"),
            Self::Shadowed => Some("shadow_mode"),
            Self::Sampled => Some("shadow_sampled"),
            Self::TypeMismatch => Some("type_mismatch"),
            Self::Missed(reason) => Some(reason.as_str()),
        }
    }
}

/// Whether stage 4 re-executes this block anyway, purely to keep the differential oracle alive.
///
/// In `on` mode the artifact replaces the re-execution that shadow comparison needs as its second
/// opinion, so an unsampled `on` run has nothing to compare and accrues no evidence. Sampling a
/// fraction of blocks buys a permanent oracle for a proportional share of the win, and it is the
/// only arrangement under which a reorg sibling is ever actually compared in `on` mode. It buys
/// nothing for a post-restart WAL replay: the handoff starts empty, so a replayed notification
/// misses and there is no artifact for any sampling rate to compare against.
pub fn shadow_sample_selects(block_number: u64) -> bool {
    sample_selects(shadow_sample_interval(), block_number)
}

const fn sample_selects(interval: u64, block_number: u64) -> bool {
    match interval {
        0 => false,
        interval => block_number % interval == 0,
    }
}

/// `PS_SHADOW_SAMPLE`: one block in this many is re-executed for comparison. 0 disables sampling.
///
/// Visible outside this module because the policy-dataset capture has to *refuse* a nonzero
/// interval: a sampled block records its own re-executed access set rather than the Engine's, and
/// a corpus that claims otherwise about 2% of its records is a corpus nobody can check.
pub fn shadow_sample_interval() -> u64 {
    static INTERVAL: OnceLock<u64> = OnceLock::new();
    *INTERVAL.get_or_init(|| {
        let Some(raw) = std::env::var_os(SHADOW_SAMPLE_VAR) else {
            return DEFAULT_SHADOW_SAMPLE;
        };
        raw.to_str().and_then(|value| value.trim().parse().ok()).unwrap_or_else(|| {
            warn!(
                target: "partial_stateless_shadow",
                var = SHADOW_SAMPLE_VAR,
                "unparsable shadow sample interval; using the default"
            );
            DEFAULT_SHADOW_SAMPLE
        })
    })
}

/// What a take from the handoff produced, plus the store telemetry at that moment.
#[derive(Debug)]
pub struct EngineAccessTake {
    /// The artifact, or `None` on a miss.
    pub artifact: Option<BlockAccessArtifact>,
    /// Why the artifact was absent, or `None` on a hit.
    pub miss_reason: Option<MissReason>,
    /// Handoff counters read immediately after the take.
    pub stats: HandoffStats,
}

/// One block's shadow result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShadowOutcome {
    /// Whether an artifact was present for this block.
    pub hit: bool,
    /// The comparison, present only on a hit.
    pub divergence: Option<AccessDivergence>,
    /// How much that comparison examined, present only on a hit.
    pub coverage: Option<AccessCoverage>,
    /// What the Engine-side capture cost for this block.
    pub capture_us: u64,
    /// Why the artifact was absent, or `None` on a hit.
    pub miss_reason: Option<MissReason>,
}

impl ShadowOutcome {
    /// Whether this block cleared the stage-3 gate: either it missed, or it matched exactly.
    pub fn is_clean(&self) -> bool {
        self.divergence.as_ref().is_none_or(AccessDivergence::is_empty)
    }
}

/// Key-level differences between a captured access set and a re-executed one.
///
/// Counted per category and per direction, because the three failure modes have different
/// meanings: a key only the simulation saw means the Engine's capture is incomplete, a key only
/// the artifact saw means the Engine touched state the builder did not, and a value mismatch
/// means the two executed against different parent state.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AccessDivergence {
    /// Accounts present only in the builder's re-execution.
    pub accounts_only_simulated: usize,
    /// Accounts present only in the Engine's capture.
    pub accounts_only_captured: usize,
    /// Accounts present in both with different values.
    pub accounts_mismatched: usize,
    /// Storage slots present only in the builder's re-execution.
    pub storage_only_simulated: usize,
    /// Storage slots present only in the Engine's capture.
    pub storage_only_captured: usize,
    /// Storage slots present in both with different values.
    pub storage_mismatched: usize,
    /// Bytecodes present only in the builder's re-execution.
    pub codes_only_simulated: usize,
    /// Bytecodes present only in the Engine's capture.
    pub codes_only_captured: usize,
    /// Bytecodes present in both with different bytes.
    pub codes_mismatched: usize,
    /// The BLOCKHASH range disagreed.
    pub lowest_block_mismatched: bool,
    /// The artifact was filed under a different parent than this block's.
    pub parent_mismatched: bool,
    /// A bounded sample of concrete differences, for diagnosis.
    pub samples: Vec<String>,
}

impl AccessDivergence {
    /// Whether the two access sets are identical in every respect this compares.
    pub fn is_empty(&self) -> bool {
        self.total() == 0 && !self.lowest_block_mismatched && !self.parent_mismatched
    }

    /// Total number of diverging keys across all categories.
    pub fn total(&self) -> usize {
        self.accounts_only_simulated +
            self.accounts_only_captured +
            self.accounts_mismatched +
            self.storage_only_simulated +
            self.storage_only_captured +
            self.storage_mismatched +
            self.codes_only_simulated +
            self.codes_only_captured +
            self.codes_mismatched
    }
}

/// How much the comparison actually examined, for one block.
///
/// A divergence count is only as strong as its denominator. Zero mismatches across a million
/// key comparisons and zero across none are the same number in the record but not the same
/// evidence, and `lowest_block_mismatched` in particular is `false` when *neither* side read
/// BLOCKHASH -- a vacuous pass that no amount of extra blocks converts into coverage. These
/// fields make the gate report what was tested rather than only what failed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AccessCoverage {
    /// Distinct accounts examined, counting each side's keys once.
    pub accounts: usize,
    /// Distinct storage slots examined.
    pub storage: usize,
    /// Distinct bytecodes examined.
    pub codes: usize,
    /// Whether either side observed a BLOCKHASH read, making the range comparison non-vacuous.
    pub blockhash_observed: bool,
}

impl AccessCoverage {
    /// Total key comparisons this block contributed to the gate.
    pub fn total(&self) -> usize {
        self.accounts + self.storage + self.codes
    }
}

impl std::fmt::Display for AccessDivergence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "accounts(-{} +{} ~{}) storage(-{} +{} ~{}) codes(-{} +{} ~{}) blockhash={} parent={} samples={:?}",
            self.accounts_only_simulated,
            self.accounts_only_captured,
            self.accounts_mismatched,
            self.storage_only_simulated,
            self.storage_only_captured,
            self.storage_mismatched,
            self.codes_only_simulated,
            self.codes_only_captured,
            self.codes_mismatched,
            self.lowest_block_mismatched,
            self.parent_mismatched,
            self.samples,
        )
    }
}

fn compare(
    parent_hash: B256,
    simulated: &BlockAccessedState,
    simulated_lowest_block: Option<u64>,
    artifact: &BlockAccessArtifact,
) -> (AccessDivergence, AccessCoverage) {
    let captured = &artifact.access;
    let mut divergence = AccessDivergence {
        lowest_block_mismatched: simulated_lowest_block != captured.lowest_block_hash_number,
        parent_mismatched: parent_hash != artifact.parent_hash,
        ..Default::default()
    };
    let mut samples = Samples::default();

    if divergence.lowest_block_mismatched {
        samples.push(format_args!(
            "blockhash sim={simulated_lowest_block:?} cap={:?}",
            captured.lowest_block_hash_number
        ));
    }
    if divergence.parent_mismatched {
        samples.push(format_args!("parent sim={parent_hash:?} cap={:?}", artifact.parent_hash));
    }

    for (address, simulated_account) in &simulated.accounts {
        match captured.accounts.get(address) {
            None => {
                divergence.accounts_only_simulated += 1;
                samples.push(format_args!("account-missing {address:?}"));
            }
            Some(captured_account) if captured_account != simulated_account => {
                divergence.accounts_mismatched += 1;
                samples.push(format_args!(
                    "account-value {address:?} sim={simulated_account:?} cap={captured_account:?}"
                ));
            }
            Some(_) => {}
        }
    }
    for address in captured.accounts.keys() {
        if !simulated.accounts.contains_key(address) {
            divergence.accounts_only_captured += 1;
            samples.push(format_args!("account-extra {address:?}"));
        }
    }

    for (key, simulated_value) in &simulated.storage {
        match captured.storage.get(key) {
            None => {
                divergence.storage_only_simulated += 1;
                samples.push(format_args!("storage-missing {key:?}"));
            }
            Some(captured_value) if captured_value != simulated_value => {
                divergence.storage_mismatched += 1;
                samples.push(format_args!(
                    "storage-value {key:?} sim={simulated_value} cap={captured_value}"
                ));
            }
            Some(_) => {}
        }
    }
    for key in captured.storage.keys() {
        if !simulated.storage.contains_key(key) {
            divergence.storage_only_captured += 1;
            samples.push(format_args!("storage-extra {key:?}"));
        }
    }

    for (code_hash, simulated_code) in &simulated.codes {
        match captured.codes.get(code_hash) {
            None => {
                divergence.codes_only_simulated += 1;
                samples.push(format_args!("code-missing {code_hash:?}"));
            }
            Some(captured_code) if captured_code != simulated_code => {
                divergence.codes_mismatched += 1;
                samples.push(format_args!(
                    "code-value {code_hash:?} sim_len={} cap_len={}",
                    simulated_code.len(),
                    captured_code.len()
                ));
            }
            Some(_) => {}
        }
    }
    for code_hash in captured.codes.keys() {
        if !simulated.codes.contains_key(code_hash) {
            divergence.codes_only_captured += 1;
            samples.push(format_args!("code-extra {code_hash:?}"));
        }
    }

    // Each side's keys counted once: everything the simulation saw, plus what only the capture did.
    let coverage = AccessCoverage {
        accounts: simulated.accounts.len() + divergence.accounts_only_captured,
        storage: simulated.storage.len() + divergence.storage_only_captured,
        codes: simulated.codes.len() + divergence.codes_only_captured,
        blockhash_observed: simulated_lowest_block.is_some() ||
            captured.lowest_block_hash_number.is_some(),
    };

    divergence.samples = samples.into_inner();
    (divergence, coverage)
}

fn append_shadow_record(
    block_number: u64,
    block_hash: B256,
    outcome: &ShadowOutcome,
    simulation_us: u64,
    stats: &HandoffStats,
) {
    let path = std::env::var_os("PS_SHADOW_OUTPUT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("access_shadow.jsonl"));
    static WRITE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let Ok(_guard) = WRITE_LOCK.get_or_init(|| Mutex::new(())).lock() else {
        warn!(target: "partial_stateless_shadow", "shadow output lock poisoned");
        return
    };
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) &&
        let Err(error) = std::fs::create_dir_all(parent)
    {
        warn!(target: "partial_stateless_shadow", %error, path = %path.display(), "failed to create shadow output directory");
        return
    }

    let divergence = outcome.divergence.as_ref();
    let coverage = outcome.coverage.as_ref();
    let record = serde_json::json!({
        "schema_version": 2,
        "benchmark": "engine_access_shadow",
        "block_number": block_number,
        "block_hash": block_hash,
        "artifact_hit": outcome.hit,
        "clean": outcome.is_clean(),
        "diverging_keys": divergence.map(AccessDivergence::total),
        "accounts_only_simulated": divergence.map(|d| d.accounts_only_simulated),
        "accounts_only_captured": divergence.map(|d| d.accounts_only_captured),
        "accounts_mismatched": divergence.map(|d| d.accounts_mismatched),
        "storage_only_simulated": divergence.map(|d| d.storage_only_simulated),
        "storage_only_captured": divergence.map(|d| d.storage_only_captured),
        "storage_mismatched": divergence.map(|d| d.storage_mismatched),
        "codes_only_simulated": divergence.map(|d| d.codes_only_simulated),
        "codes_only_captured": divergence.map(|d| d.codes_only_captured),
        "codes_mismatched": divergence.map(|d| d.codes_mismatched),
        "lowest_block_mismatched": divergence.map(|d| d.lowest_block_mismatched),
        "parent_mismatched": divergence.map(|d| d.parent_mismatched),
        "compared_keys": coverage.map(AccessCoverage::total),
        "accounts_compared": coverage.map(|c| c.accounts),
        "storage_compared": coverage.map(|c| c.storage),
        "codes_compared": coverage.map(|c| c.codes),
        "blockhash_observed": coverage.map(|c| c.blockhash_observed),
        "samples": divergence.map(|d| d.samples.clone()),
        "miss_reason": outcome.miss_reason.as_ref().map(MissReason::as_str),
        "engine_capture_us": outcome.capture_us,
        "builder_simulation_us": simulation_us,
        "handoff_queue_depth": stats.queue_depth,
        "handoff_resident_bytes": stats.resident_bytes,
        "handoff_inserted": stats.inserted,
        "handoff_taken": stats.taken,
        "handoff_missed": stats.missed,
        "handoff_dropped_capacity": stats.dropped_capacity,
        "handoff_dropped_contended": stats.dropped_contended,
        "handoff_replaced": stats.replaced,
        "handoff_mean_residence_us": stats.mean_residence_us,
        "handoff_mean_depth_at_take": stats.mean_depth_at_take,
    });

    let result = (|| -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new().create(true).append(true).open(&path)?;
        serde_json::to_writer(&mut file, &record).map_err(std::io::Error::other)?;
        file.write_all(b"\n")?;
        file.flush()
    })();
    if let Err(error) = result {
        warn!(target: "partial_stateless_shadow", %error, path = %path.display(), "failed to append shadow record");
    }
}

/// A bounded collector, so a block that diverges in ten thousand keys still logs one line.
#[derive(Debug, Default)]
struct Samples(Vec<String>);

impl Samples {
    fn push(&mut self, args: std::fmt::Arguments<'_>) {
        if self.0.len() >= MAX_SAMPLES {
            return
        }
        let mut sample = String::new();
        let _ = sample.write_fmt(args);
        self.0.push(sample);
    }

    fn into_inner(self) -> Vec<String> {
        self.0
    }
}

#[derive(Debug, Default)]
struct ShadowTotals {
    blocks: AtomicU64,
    hits: AtomicU64,
    missed: AtomicU64,
    diverged: AtomicU64,
}

impl ShadowTotals {
    fn get() -> &'static Self {
        static TOTALS: OnceLock<ShadowTotals> = OnceLock::new();
        TOTALS.get_or_init(Self::default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::{Address, Bytes, U256};
    use partial_stateless::policy::AccountData;
    use reth_execution_access::ExecutedBlockAccess;
    use std::sync::Arc;

    fn account(nonce: u64) -> AccountData {
        AccountData { nonce, balance: U256::from(nonce), code_hash: None }
    }

    fn populated() -> ExecutedBlockAccess {
        let mut access =
            ExecutedBlockAccess { lowest_block_hash_number: Some(7), ..Default::default() };
        access.accounts.insert(Address::with_last_byte(1), account(1));
        access.storage.insert((Address::with_last_byte(1), B256::with_last_byte(2)), U256::from(3));
        access.codes.insert(B256::with_last_byte(4), Bytes::from_static(&[5, 6]));
        access
    }

    fn artifact_of(access: ExecutedBlockAccess, parent: B256) -> BlockAccessArtifact {
        BlockAccessArtifact::new(10, B256::with_last_byte(10), parent, access, Arc::new(0u8), 0)
    }

    fn parent() -> B256 {
        B256::with_last_byte(9)
    }

    #[test]
    fn sampling_reserves_a_fraction_of_blocks_for_comparison() {
        assert!(sample_selects(50, 100));
        assert!(!sample_selects(50, 101));
        let selected = (0..500u64).filter(|block| sample_selects(50, *block)).count();
        assert_eq!(selected, 10, "one block in fifty");

        // Zero is the escape hatch for a pure timing run, where any re-execution would pollute
        // the builder median that stage 5 reads.
        assert!(!sample_selects(0, 0), "interval 0 must disable sampling, not select every block");
        assert!(!sample_selects(0, 100));
    }

    #[test]
    fn an_artifact_whose_output_is_not_an_execution_output_is_a_miss_not_an_empty_result() {
        // `artifact_of` files an `Arc<u8>` as the output. Downcasting it to a BlockExecutionOutput
        // must fail into `None` so the caller re-executes. Returning a default here would publish
        // a sidecar built from an empty bundle: a wrong result rather than a slow one.
        let artifact = artifact_of(populated(), parent());
        assert!(simulation_from_artifact(artifact).is_none());
    }

    #[test]
    fn a_reused_artifact_carries_the_access_set_and_blockhash_range_across() {
        let access = populated();
        let expected_lowest = access.lowest_block_hash_number;
        let artifact = BlockAccessArtifact::new(
            10,
            B256::with_last_byte(10),
            parent(),
            access.clone(),
            Arc::new(BlockExecutionOutput::<<EthPrimitives as NodePrimitives>::Receipt>::default()),
            0,
        );

        let simulation = simulation_from_artifact(artifact).expect("output downcasts");
        assert_eq!(simulation.lowest_block_number, expected_lowest);
        let expected = BlockAccessedState::from(access);
        assert_eq!(simulation.accessed.accounts, expected.accounts);
        assert_eq!(simulation.accessed.storage, expected.storage);
        assert_eq!(simulation.accessed.codes, expected.codes);
        assert_eq!(simulation.elapsed_us, 0, "no EVM ran on this path");
    }

    #[test]
    fn identical_sets_do_not_diverge() {
        let simulated = BlockAccessedState::from(populated());
        let (divergence, coverage) =
            compare(parent(), &simulated, Some(7), &artifact_of(populated(), parent()));

        assert!(divergence.is_empty(), "{divergence}");
        assert_eq!(divergence.total(), 0);
        assert!(divergence.samples.is_empty());
        assert_eq!(coverage.total(), 3, "a clean block must still report what it compared");
        assert!(coverage.blockhash_observed);
    }

    #[test]
    fn a_block_that_never_read_blockhash_reports_the_range_check_as_uncovered() {
        // `lowest_block_mismatched` is false when neither side read BLOCKHASH, so a run made
        // entirely of such blocks would report a clean range check it never actually exercised.
        // Coverage is what separates that from a real pass; more blocks would not.
        let mut without_blockhash = populated();
        without_blockhash.lowest_block_hash_number = None;
        let simulated = BlockAccessedState::from(without_blockhash.clone());

        let (divergence, coverage) =
            compare(parent(), &simulated, None, &artifact_of(without_blockhash, parent()));
        assert!(divergence.is_empty());
        assert!(!divergence.lowest_block_mismatched);
        assert!(!coverage.blockhash_observed, "a vacuous pass must be visible in the record");
    }

    #[test]
    fn a_key_the_engine_did_not_capture_is_counted_as_missing() {
        // The failure this whole stage exists to catch: prewarming or a cross-block cache leaving
        // something out of the Engine's `State` that re-execution still sees.
        let mut captured = populated();
        captured.accounts.clear();
        let simulated = BlockAccessedState::from(populated());

        let (divergence, _) =
            compare(parent(), &simulated, Some(7), &artifact_of(captured, parent()));
        assert_eq!(divergence.accounts_only_simulated, 1);
        assert_eq!(divergence.accounts_only_captured, 0);
        assert!(!divergence.is_empty());
        assert_eq!(divergence.samples.len(), 1);
    }

    #[test]
    fn a_key_only_the_engine_saw_is_counted_separately_from_a_missing_one() {
        let mut captured = populated();
        captured.accounts.insert(Address::with_last_byte(2), account(2));
        let simulated = BlockAccessedState::from(populated());

        let (divergence, _) =
            compare(parent(), &simulated, Some(7), &artifact_of(captured, parent()));
        assert_eq!(divergence.accounts_only_captured, 1);
        assert_eq!(divergence.accounts_only_simulated, 0);
        assert_eq!(divergence.accounts_mismatched, 0);
    }

    #[test]
    fn the_same_key_with_a_different_value_is_a_mismatch_in_every_category() {
        let mut captured = populated();
        captured.accounts.insert(Address::with_last_byte(1), account(99));
        captured.storage.insert((Address::with_last_byte(1), B256::with_last_byte(2)), U256::MAX);
        captured.codes.insert(B256::with_last_byte(4), Bytes::from_static(&[0]));
        let simulated = BlockAccessedState::from(populated());

        let (divergence, _) =
            compare(parent(), &simulated, Some(7), &artifact_of(captured, parent()));
        assert_eq!(divergence.accounts_mismatched, 1);
        assert_eq!(divergence.storage_mismatched, 1);
        assert_eq!(divergence.codes_mismatched, 1);
        assert_eq!(divergence.total(), 3);
    }

    #[test]
    fn the_blockhash_range_and_the_parent_are_part_of_the_comparison() {
        let simulated = BlockAccessedState::from(populated());

        let (wrong_range, _) =
            compare(parent(), &simulated, Some(6), &artifact_of(populated(), parent()));
        assert!(wrong_range.lowest_block_mismatched);
        assert!(!wrong_range.is_empty(), "a BLOCKHASH disagreement is a divergence on its own");
        assert_eq!(wrong_range.total(), 0, "but it is not a key count");

        let (wrong_parent, _) = compare(
            parent(),
            &simulated,
            Some(7),
            &artifact_of(populated(), B256::with_last_byte(200)),
        );
        assert!(wrong_parent.parent_mismatched);
        assert!(!wrong_parent.is_empty());
    }

    #[test]
    fn samples_are_bounded_so_one_bad_block_cannot_produce_an_unbounded_log_line() {
        let mut captured = populated();
        captured.accounts.clear();
        let mut simulated = populated();
        for index in 0..100u8 {
            simulated.accounts.insert(Address::with_last_byte(index), account(index.into()));
        }
        let simulated = BlockAccessedState::from(simulated);

        let (divergence, _) =
            compare(parent(), &simulated, Some(7), &artifact_of(captured, parent()));
        assert_eq!(divergence.accounts_only_simulated, 100);
        assert_eq!(divergence.samples.len(), MAX_SAMPLES);
    }
}
