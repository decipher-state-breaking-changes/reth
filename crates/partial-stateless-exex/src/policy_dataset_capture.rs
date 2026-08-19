//! Opt-in capture of the corpus every cache policy is later generated from, offline.
//!
//! The whole of this module is behind one environment variable, and the reason is arithmetic
//! rather than taste. A capturing block builds a **second** parent-state witness — the
//! policy-neutral full one, proving every accessed key and every mutation path from the root —
//! then validates it database-free and writes it to disk. That is a multiproof, a re-execution,
//! and megabytes of I/O per block that production never pays. A build that did any of it by
//! default would have no measured numbers left.
//!
//! So the contract is stated as a property of the process rather than as a per-block branch:
//! [`PolicyDatasetCaptureConfig::from_env`] runs once at startup, and when it returns `None`
//! nothing in this module is ever constructed. No full witness is built, no payload is cloned for
//! a dataset, no writer exists, and no file is created.
//!
//! Capture is also refused alongside anything that measures. The paired validation benchmark, the
//! accessed-state fixture dump, and the full-witness baseline each want the whole block budget,
//! and a run that combined one of them with this would be reporting the capture. Those are startup
//! errors, not warnings: a run configured for two incompatible jobs has already decided to produce
//! a number nobody can use, and the useful moment to say so is before it starts.
//!
//! In the other direction, capture *requires* the Engine handoffs. The recorded payload has to be
//! the one a consensus client actually sent (`PS_ENGINE_PAYLOAD=on`), because a payload derived
//! from a block this node already accepted hands a later validator the answers its own admission
//! checks exist to question. And the recorded access set has to be the Engine's own
//! (`PS_ENGINE_ACCESS=on`), because that is the set production runs on, and a corpus recorded off
//! a different execution path would be a corpus of a system nobody runs.

use alloy_primitives::{Bytes, B256};
use partial_stateless::{
    policy_dataset::{
        DatasetEndKind, LifecycleEvent, PolicyDatasetManifest, PolicyDatasetRecord,
        PolicyDatasetRecordBody, PolicyDatasetWriter, RecordedAccessProvenance,
        RecordedPayloadProvenance,
    },
    BlockAccessedState,
};
use reth_execution_access::{payload_capture_enabled, AccessCaptureMode};
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};
use tracing::{error, info, warn};

/// Where a policy replay dataset is written. Absent means capture is off.
const CAPTURE_DIR_VAR: &str = "PS_POLICY_DATASET_CAPTURE_DIR";
/// How many usable blocks a capture records before it starts confirming them.
const MAX_BLOCKS_VAR: &str = "PS_POLICY_DATASET_MAX_BLOCKS";
/// Canonical blocks the chain must advance past the recorded range before it is vouched for.
const CONFIRMATIONS_VAR: &str = "PS_POLICY_DATASET_CONFIRMATIONS";
/// Set to capture from a build that cannot name the commit it was built from.
const ALLOW_UNSTAMPED_VAR: &str = "PS_POLICY_DATASET_ALLOW_UNSTAMPED";

/// Blocks a capture may fail to record before its first record, before it gives up.
///
/// A capture starts behind: the node spends most of a minute reaching the point where the ExEx
/// takes its first notification, and the Engine taps hand off through small ring buffers meanwhile.
/// The block the ExEx lands on can therefore be one whose payload and access artifact were already
/// evicted, and there is nothing wrong with the corpus in that case — it has not started. Waiting
/// is right, and waiting forever is not: a run whose configuration means *no* block will ever
/// qualify would otherwise build a full witness per block for a corpus that never begins. This
/// bound is generous against the first reading and short against the second.
const MAX_SKIPS_BEFORE_FIRST_RECORD: u64 = 64;

/// Default confirmation depth: three epochs of slots.
///
/// Mainnet finalizes two epochs back, so two epochs of *blocks* is the floor and this is a margin
/// on top of it. It is a default rather than a requirement because a smoke test over five blocks
/// should not have to wait an hour, and it is logged at every startup because a capture that
/// silently used the wrong depth would produce a corpus whose tail is reorg-exposed with nothing
/// on disk to say so.
const DEFAULT_CONFIRMATIONS: u64 = 96;

/// The variables a capture run refuses to share a process with.
///
/// Every one of them either wants the whole block budget or reports a per-block cost, and a
/// capturing run supplies neither: it spends part of every block on a second witness nobody
/// measured, and it perturbs the page cache, the allocator, and the process's own resource
/// counters while doing it.
const CONFLICTING_VARS: [(&str, &str); 6] = [
    ("PS_VALIDATION_BENCH", "the paired Partial/Weak validation benchmark"),
    ("PS_BENCH_OUTPUT", "the paired benchmark's output"),
    ("PS_BUILDER_BENCH_OUTPUT", "the per-block builder cost record"),
    ("PS_CAPTURE_DIR", "the accessed-state fixture dump"),
    ("PS_WITNESS_BASELINE", "the full-witness baseline"),
    ("PS_RESOURCE_METRICS", "process CPU and page-fault sampling"),
];

/// One process's capture configuration, resolved once at startup.
#[derive(Debug, Clone)]
pub struct PolicyDatasetCaptureConfig {
    /// Dataset root. Absolute, so a dataset cannot land relative to whatever directory the node
    /// happened to be launched from.
    pub dir: PathBuf,
    /// Usable blocks to record. There is no default: a capture that quietly ran forever would
    /// fill a disk, and one that quietly stopped early would produce a shorter corpus than the run
    /// that reads it expects.
    ///
    /// Counted in blocks the capture still believes are canonical. A block a later reorg abandons
    /// stops counting, and the capture records a replacement — so a corpus of `max_blocks` is
    /// `max_blocks` usable blocks, not `max_blocks` files.
    pub max_blocks: u64,
    /// Canonical blocks the chain must advance past the recorded range before the terminator is
    /// written.
    ///
    /// Writing `END.json` the instant the last file lands would vouch for a tip that is still
    /// reorg-exposed. Zero disables the wait, which is a deliberate choice for a smoke test and a
    /// mistake for anything else; the terminator records which was made.
    pub confirmations: u64,
}

impl PolicyDatasetCaptureConfig {
    /// Reads the capture contract, or `None` when this process is not capturing.
    ///
    /// Every failure here is a startup error rather than a fallback. A capture is a job someone
    /// asked for explicitly; defaulting any part of it would produce a dataset whose description
    /// and contents disagree.
    pub fn from_env() -> eyre::Result<Option<Self>> {
        Self::resolve(
            std::env::var_os(CAPTURE_DIR_VAR).map(PathBuf::from),
            std::env::var(MAX_BLOCKS_VAR).ok(),
            std::env::var(CONFIRMATIONS_VAR).ok(),
            &|name| std::env::var_os(name).is_some(),
            AccessCaptureMode::current(),
            payload_capture_enabled(),
            crate::access_shadow::shadow_sample_interval(),
        )
    }

    /// [`Self::from_env`] with the environment passed in.
    ///
    /// Split out so the contract can be tested. Every rule below is a refusal, and a refusal only
    /// protects a run if it is exercised — which a test that had to mutate the process environment
    /// could not reliably do beside other tests in the same process.
    fn resolve(
        dir: Option<PathBuf>,
        max_blocks: Option<String>,
        confirmations: Option<String>,
        conflict_present: &dyn Fn(&str) -> bool,
        access_mode: AccessCaptureMode,
        payload_capture: bool,
        shadow_sample_interval: u64,
    ) -> eyre::Result<Option<Self>> {
        let Some(dir) = dir else {
            return Ok(None);
        };
        if !dir.is_absolute() {
            eyre::bail!(
                "{CAPTURE_DIR_VAR} must be an absolute path, not {:?}: a dataset resolved against \
                 the launch directory cannot be found again by the offline generator",
                dir
            );
        }

        let max_blocks = match max_blocks {
            None => eyre::bail!(
                "{CAPTURE_DIR_VAR} is set, so {MAX_BLOCKS_VAR} is required: a capture with no \
                 block budget has no defined end and no defined size"
            ),
            Some(raw) => match raw.trim().parse::<u64>() {
                Ok(value) if value >= 1 => value,
                _ => eyre::bail!("{MAX_BLOCKS_VAR} must be an integer >= 1, not `{raw}`"),
            },
        };

        for (var, what) in CONFLICTING_VARS {
            if conflict_present(var) {
                eyre::bail!(
                    "{CAPTURE_DIR_VAR} cannot be combined with {var} ({what}): a capturing run \
                     builds and writes a full witness per block, so anything it measured would be \
                     measuring the capture"
                );
            }
        }

        if !matches!(access_mode, AccessCaptureMode::On) {
            eyre::bail!(
                "{CAPTURE_DIR_VAR} requires PS_ENGINE_ACCESS=on: the recorded access set must be \
                 the one the node's own Engine produced, or the corpus describes an execution \
                 path production does not run"
            );
        }
        if !payload_capture {
            eyre::bail!(
                "{CAPTURE_DIR_VAR} requires PS_ENGINE_PAYLOAD=on: a payload derived from a block \
                 this node already accepted hands a later validator the answers its own admission \
                 checks exist to question"
            );
        }

        // Sampling is what makes `PS_ENGINE_ACCESS=on` a *mostly*-Engine access set: one block in
        // `PS_SHADOW_SAMPLE` is re-executed and its own set used. That is the right trade for a
        // measured run, which wants the differential oracle alive. It is the wrong trade for a
        // corpus, which claims every record came from the Engine — and 2% of records that quietly
        // did not is exactly the kind of thing nobody finds later.
        //
        // Nothing is lost by turning it off here. The capture re-executes every block
        // database-free against its own witness and compares the access sets before writing it,
        // and the offline generator does the same again on another host. Both are stronger
        // oracles than the sampled comparison this replaces.
        if shadow_sample_interval != 0 {
            eyre::bail!(
                "{CAPTURE_DIR_VAR} requires PS_SHADOW_SAMPLE=0, not {shadow_sample_interval}: \
                 sampling re-executes one block in {shadow_sample_interval} and records that \
                 block's own access set, so a corpus captured with it on would claim every record \
                 came from the Engine while some did not"
            );
        }

        let confirmations = match confirmations {
            None => DEFAULT_CONFIRMATIONS,
            Some(raw) => raw.trim().parse::<u64>().map_err(|_| {
                eyre::eyre!("{CONFIRMATIONS_VAR} must be a non-negative integer, not `{raw}`")
            })?,
        };

        Ok(Some(Self { dir, max_blocks, confirmations }))
    }
}

/// Refuses a capture from a build that cannot say which commit it is.
///
/// `PS_BUILD_COMMIT` is read by `option_env!` at **compile** time, so a binary built without it
/// carries no commit and no amount of exporting the variable afterwards puts one there. The
/// failure is silent in the direction that matters: the capture runs perfectly, for hours, and
/// writes a manifest whose `build_commit` is `null` — a corpus that cannot be tied to the code
/// that produced it, discovered when someone reads the manifest rather than when the mistake is
/// cheap to fix. Checking here costs one comparison at startup and turns that into an immediate
/// error naming what to do.
///
/// [`ALLOW_UNSTAMPED_VAR`] exists for smoke tests, where a throwaway corpus from a working tree is
/// the point. It hides nothing: the manifest still records `build_commit: null`, so a dataset
/// captured through the escape hatch says so on its face.
fn require_stamped_build(build_commit: Option<&str>, allow_unstamped: bool) -> eyre::Result<()> {
    if build_commit.is_some_and(|commit| !commit.trim().is_empty()) || allow_unstamped {
        return Ok(())
    }
    eyre::bail!(
        "this build carries no PS_BUILD_COMMIT, so its dataset could not name the code that \
         produced it. PS_BUILD_COMMIT is read at compile time: export it (with PS_BUILD_DIRTY and \
         PS_CARGO_LOCK_SHA256) and rebuild before capturing, or set {ALLOW_UNSTAMPED_VAR}=1 to \
         capture a corpus that records no commit"
    )
}

/// The live writer, held for the length of a capturing run.
///
/// It has three states, and the middle one is the reason it is a state machine at all.
///
/// **Capturing** — under budget, paying for a full witness per block.
/// **Confirming** — budget met, paying for nothing, watching the chain advance past the range it
/// recorded. A reorg here can send it back to capturing.
/// **Closed** — the range settled, the terminator written, everything after it a no-op.
#[derive(Debug)]
pub struct PolicyDatasetRecorder {
    writer: Option<PolicyDatasetWriter>,
    dir: PathBuf,
    max_blocks: u64,
    confirmations: u64,
    /// Every `(height, hash)` this run has a file for, reorg-abandoned ones included.
    ///
    /// A set rather than a counter because a chain that reorganises away from a branch and back
    /// onto it re-records the same block, which rewrites the same file rather than adding one. A
    /// counter would then claim more records than the directory holds and the loader would refuse
    /// the dataset for disagreeing with itself.
    written: BTreeSet<(u64, B256)>,
    /// Every record this run wrote and still believes canonical, lowest first.
    ///
    /// Kept as a list rather than a count because a reorg has to remove specific heights: the
    /// budget is a budget of *usable* blocks, and a run that let abandoned ones spend it would
    /// hand the offline stage a corpus shorter than the one it asked for.
    usable: Vec<(u64, B256)>,
    /// The canonical head the producer has observed.
    head: u64,
    /// Set once the terminator is written, so it is written exactly once.
    complete: bool,
    /// Blocks refused before the first record landed, bounded by
    /// [`MAX_SKIPS_BEFORE_FIRST_RECORD`].
    skipped_before_first: u64,
}

impl PolicyDatasetRecorder {
    /// Opens a dataset for a capturing run, or returns `None` when capture is off.
    pub fn open(
        config: Option<PolicyDatasetCaptureConfig>,
        producer: String,
        build_commit: Option<String>,
        chain: String,
    ) -> eyre::Result<Option<Self>> {
        let Some(config) = config else { return Ok(None) };
        require_stamped_build(
            build_commit.as_deref(),
            std::env::var_os(ALLOW_UNSTAMPED_VAR).is_some(),
        )?;
        let manifest = PolicyDatasetManifest::new(producer, build_commit, chain, config.max_blocks);
        let writer = PolicyDatasetWriter::create(&config.dir, &manifest)?;
        info!(
            target: "partial_stateless",
            dir = %config.dir.display(),
            max_blocks = config.max_blocks,
            "Policy replay dataset capture ENABLED — this run builds a full witness per block and \
             is NOT a measurement run"
        );
        Ok(Some(Self {
            writer: Some(writer),
            dir: config.dir,
            max_blocks: config.max_blocks,
            confirmations: config.confirmations,
            written: BTreeSet::new(),
            usable: Vec::new(),
            head: 0,
            complete: false,
            skipped_before_first: 0,
        }))
    }

    /// Whether the builder should still pay for a full witness on this block.
    ///
    /// Read before the witness is built rather than before the record is written: the budget's
    /// purpose is to bound the work, and a run that built witnesses it then discarded would cost
    /// the same as one with no budget at all.
    ///
    /// Counts *usable* records, so a reorg that abandons one reopens the budget by one and the
    /// capture records a replacement.
    pub fn wants_block(&self) -> bool {
        !self.complete && (self.usable.len() as u64) < self.max_blocks
    }

    /// The canonical range the producer would vouch for right now, with the hash at its top.
    ///
    /// The tip hash travels with the range because it is what a reader walks parents down from to
    /// recover the canonical set — a job the lifecycle log cannot do, since a chain that
    /// reorganises away from a branch and back onto it leaves both branches listed as abandoned.
    fn usable_range(&self) -> Option<(u64, u64, B256)> {
        let (high, tip) = *self.usable.last()?;
        Some((self.usable.first()?.0, high, tip))
    }

    /// Whether the chain has advanced far enough past the recorded range to close the dataset.
    fn range_is_settled(&self) -> bool {
        let Some((_, high, _)) = self.usable_range() else { return false };
        (self.usable.len() as u64) >= self.max_blocks &&
            self.head >= high.saturating_add(self.confirmations)
    }

    /// Notes the canonical head, and closes the dataset once its range has settled.
    ///
    /// Called on every committed block, including the ones after the budget is met. Those blocks
    /// cost the capture nothing — no witness is built for them — and they are the whole of what
    /// separates a corpus whose tail is reorg-exposed from one that is not.
    pub fn observe_head(&mut self, block_number: u64) {
        if self.complete {
            return;
        }
        self.head = self.head.max(block_number);
        if self.range_is_settled() {
            let range = self.usable_range();
            let head = self.head;
            self.close_with(
                DatasetEndKind::BlockBudgetReached,
                range,
                Some(head),
                format!(
                    "{} usable blocks, confirmed by {} canonical blocks on top",
                    self.usable.len(),
                    self.confirmations
                ),
            );
        }
    }

    /// Dataset root, for logging.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Decides what a startup Engine-handoff miss means for the run.
    ///
    /// Returns `true` when the run should carry on. That is the case only while the corpus is
    /// empty: a block refused then leaves nothing behind it, so the corpus starts later and is
    /// whole. Once a record exists the same refusal would put a hole in the middle of it, and a
    /// corpus with a hole is worse than none — it looks complete.
    ///
    /// No other recording error may call this path. Provider, encoding, and filesystem failures
    /// are fatal even before the first record: unlike a moving handoff ring, none is expected to
    /// heal merely because the next block arrived.
    ///
    /// The skips are counted and bounded, because "not yet" and "never" look identical from here
    /// for as long as one is willing to wait.
    pub(crate) fn skip_startup_handoff_miss(
        &mut self,
        block_number: u64,
        miss: &PolicyDatasetMaterialError,
    ) -> bool {
        if !miss.is_startup_handoff_miss() {
            return false
        }
        if !self.written.is_empty() || self.complete {
            return false
        }
        self.skipped_before_first += 1;
        if self.skipped_before_first > MAX_SKIPS_BEFORE_FIRST_RECORD {
            return false
        }
        warn!(
            target: "partial_stateless",
            block = block_number,
            skipped = self.skipped_before_first,
            limit = MAX_SKIPS_BEFORE_FIRST_RECORD,
            reason = %miss,
            "Policy replay dataset has not started yet; this block cannot be recorded and the \
             corpus will begin at a later one"
        );
        self.note(LifecycleEvent::Skipped { block_number, reason: miss.to_string() });
        true
    }

    /// Records one block, and closes the dataset when the budget is met.
    pub fn record(&mut self, body: PolicyDatasetRecordBody) -> eyre::Result<()> {
        if !self.wants_block() {
            return Ok(());
        }
        let block_number = body.block_number;
        let block_hash = body.block_hash;
        let record: PolicyDatasetRecord = body.seal()?;
        let Some(writer) = self.writer.as_mut() else { return Ok(()) };
        let path = writer.write_record(&record)?;
        self.written.insert((block_number, block_hash));
        self.usable.push((block_number, block_hash));
        info!(
            target: "partial_stateless",
            block = block_number,
            usable = self.usable.len(),
            max_blocks = self.max_blocks,
            files = self.written.len(),
            path = %path.display(),
            "Recorded policy replay dataset block"
        );
        if (self.usable.len() as u64) >= self.max_blocks {
            info!(
                target: "partial_stateless",
                usable = self.usable.len(),
                confirmations = self.confirmations,
                until_head = block_number.saturating_add(self.confirmations),
                "Policy replay dataset reached its block budget; confirming before closing"
            );
        }
        // The budget being met is not the end. The terminator waits for the chain to advance past
        // this range, which `observe_head` decides.
        self.observe_head(block_number);
        Ok(())
    }

    /// Records a branch change, and stops counting whatever it abandoned.
    ///
    /// The abandoned records stay on disk — the offline stage needs them to audit the exclusion —
    /// but they no longer spend the budget, and a reorg that lands during the confirmation wait
    /// puts the capture back to work rather than letting it close over a range that just changed
    /// underneath it.
    pub fn note_reorg(&mut self, common_ancestor: u64, abandoned: Vec<(u64, B256)>) {
        if self.complete {
            // The dataset is closed and its range was confirmed to a depth this reorg did not
            // reach — or the operator asked for no depth at all, and the terminator says so.
            return;
        }
        let before = self.usable.len();
        self.usable.retain(|recorded| !abandoned.contains(recorded));
        let dropped = before - self.usable.len();
        if dropped > 0 {
            warn!(
                target: "partial_stateless",
                common_ancestor,
                dropped,
                usable = self.usable.len(),
                max_blocks = self.max_blocks,
                "A reorg abandoned recorded blocks; they stay on disk but no longer count, and \
                 the capture will record replacements"
            );
        }
        self.note(LifecycleEvent::Reorg { common_ancestor, abandoned });
    }

    /// Records a continuity break, which the offline stage must not join across.
    pub fn note_reset(&mut self, block_number: u64, detail: String) {
        self.note(LifecycleEvent::Reset { block_number, detail });
    }

    /// Records where the capture began.
    pub fn note_started(&mut self, block_number: u64) {
        self.note(LifecycleEvent::Started { block_number });
    }

    /// Appends one lifecycle event, failing the whole dataset if it cannot be written.
    ///
    /// A warning would not do. The lifecycle log is the only record of what the chain did under the
    /// capture, and a reorg that happened but was never logged leaves a corpus whose exclusions
    /// cannot be audited — indistinguishable, on disk, from one where nothing happened. Failing
    /// here converts a silently unauditable dataset into a refused one.
    fn note(&mut self, event: LifecycleEvent) {
        let Some(writer) = self.writer.as_ref() else { return };
        let Err(err) = writer.write_lifecycle(&event) else { return };
        error!(
            target: "partial_stateless",
            error = %err,
            ?event,
            "Failed to append a policy dataset lifecycle event; failing the dataset rather than \
             leaving a corpus whose chain history cannot be audited"
        );
        self.fail(format!("a lifecycle event could not be recorded: {err}"));
    }

    /// Closes the dataset, vouching for whatever range has actually settled.
    ///
    /// Reached by the run's own shutdown, which is the case where the range usually has *not*
    /// settled — so the terminator names the settled part and nothing more, and the loader drops
    /// the rest. A shutdown that vouched for its whole output would be vouching for a tip it never
    /// saw confirmed.
    pub fn close(&mut self, kind: DatasetEndKind, detail: String) {
        let settled = self.settled_range();
        let head = (self.head > 0).then_some(self.head);
        self.close_with(kind, settled, head, detail);
    }

    /// The prefix of the recorded range that the observed head confirms.
    ///
    /// A shutdown mid-capture still leaves a usable corpus: everything at least `confirmations`
    /// below the head has had that many canonical blocks stacked on it, whether or not the budget
    /// was ever met.
    fn settled_range(&self) -> Option<(u64, u64, B256)> {
        let ceiling = self.head.checked_sub(self.confirmations)?;
        let low = self.usable.first()?.0;
        let (high, tip) = self.usable.iter().copied().rfind(|(n, _)| *n <= ceiling)?;
        Some((low, high, tip))
    }

    fn close_with(
        &mut self,
        kind: DatasetEndKind,
        usable: Option<(u64, u64, B256)>,
        confirmed_at_head: Option<u64>,
        detail: String,
    ) {
        let Some(writer) = self.writer.take() else { return };
        self.complete = true;
        let written = self.written.len();
        match writer.finish(kind, usable, self.confirmations, confirmed_at_head, detail) {
            Ok(()) => info!(
                target: "partial_stateless",
                dir = %self.dir.display(),
                records = written,
                usable_range = ?usable.map(|(low, high, _)| (low, high)),
                confirmations = self.confirmations,
                ?kind,
                "Policy replay dataset closed"
            ),
            Err(err) => warn!(
                target: "partial_stateless",
                dir = %self.dir.display(),
                error = %err,
                "Failed to write the policy replay dataset terminator; the dataset is incomplete \
                 and the offline generator will refuse it"
            ),
        }
    }

    /// Marks the dataset unusable, for a capture that could not produce what it promised.
    ///
    /// Vouches for no range at all, whatever settled. A capture that failed on one block cannot
    /// say the blocks before it are sound, because the reason it failed is by definition one it
    /// did not anticipate.
    pub fn fail(&mut self, detail: String) {
        let head = (self.head > 0).then_some(self.head);
        self.close_with(DatasetEndKind::Failed, None, head, detail);
    }
}

/// Everything a capturing builder produced for one block, minus the payload.
///
/// The payload is joined on at the ExEx loop rather than carried through the builder: it is taken
/// from the Engine handoff before the block is processed at all, and threading it down only to
/// hand it back would put the same value in two places.
#[derive(Debug)]
pub struct PolicyDatasetMaterial {
    /// Parent state root the full witness is proved against.
    pub parent_state_root: B256,
    /// The state root the block's own header claims.
    pub expected_state_root: B256,
    /// State the block accessed.
    pub accessed: BlockAccessedState,
    /// Where that access set came from.
    pub access_provenance: RecordedAccessProvenance,
    /// The policy-neutral full transition witness.
    pub full_transition_nodes: Vec<Bytes>,
    /// Ancestor headers this block's BLOCKHASH range needs.
    pub ancestor_headers: Vec<Bytes>,
}

/// A material refusal whose class determines whether an empty capture may start one block later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PolicyDatasetMaterialError {
    /// The Engine payload ring no longer held this block.
    PayloadHandoffMiss { block_number: u64, provenance: RecordedPayloadProvenance },
    /// The Engine access-artifact ring no longer held this block.
    AccessHandoffMiss { block_number: u64, provenance: RecordedAccessProvenance },
    /// The handoff claimed success but supplied no payload bytes.
    MissingWitnessedPayload { block_number: u64 },
}

impl PolicyDatasetMaterialError {
    const fn is_startup_handoff_miss(&self) -> bool {
        matches!(self, Self::PayloadHandoffMiss { .. } | Self::AccessHandoffMiss { .. })
    }
}

impl std::fmt::Display for PolicyDatasetMaterialError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PayloadHandoffMiss { block_number, provenance } => write!(
                f,
                "block {block_number} has {provenance:?} payload provenance; a policy replay \
                 dataset records only payloads the Engine witnessed, because an admission check \
                 run against a derived payload is checking a derivation against itself"
            ),
            Self::AccessHandoffMiss { block_number, provenance } => write!(
                f,
                "block {block_number} has {provenance:?} access provenance; a policy replay \
                 dataset records only the access set the node's own Engine produced, because \
                 that is the set production runs on"
            ),
            Self::MissingWitnessedPayload { block_number } => {
                write!(f, "block {block_number} was reported as witnessed but carries no payload")
            }
        }
    }
}

impl std::error::Error for PolicyDatasetMaterialError {}

impl PolicyDatasetMaterial {
    /// Joins the builder's material to the Engine payload the ExEx loop took.
    ///
    /// Refuses a block with no payload rather than recording one. `Absent` is a fact about a
    /// producer that obtained nothing — a WAL replay, a backfill — and a dataset carrying such a
    /// record would hand the offline generator a block it cannot admit, discovered a thousand
    /// blocks into a run rather than here.
    pub(crate) fn into_record_body(
        self,
        block_number: u64,
        block_hash: B256,
        parent_hash: B256,
        parent_header: Bytes,
        payload_json: Option<Vec<u8>>,
        payload_provenance: RecordedPayloadProvenance,
    ) -> Result<PolicyDatasetRecordBody, PolicyDatasetMaterialError> {
        if payload_provenance != RecordedPayloadProvenance::Witnessed {
            return Err(PolicyDatasetMaterialError::PayloadHandoffMiss {
                block_number,
                provenance: payload_provenance,
            })
        }
        // The same rule on the other input. Startup already refuses a nonzero `PS_SHADOW_SAMPLE`,
        // so deliberate sampling cannot reach here — what can is a handoff miss, which is
        // structural rather than exceptional on a WAL replay or a backfill, and an artifact whose
        // output would not downcast. Either way the recorded set came from this builder
        // re-executing rather than from the Engine, and a corpus that mixed the two would be
        // describing two execution paths under one name.
        if self.access_provenance != RecordedAccessProvenance::EngineArtifact {
            return Err(PolicyDatasetMaterialError::AccessHandoffMiss {
                block_number,
                provenance: self.access_provenance,
            })
        }
        let Some(payload_json) = payload_json else {
            return Err(PolicyDatasetMaterialError::MissingWitnessedPayload { block_number })
        };
        Ok(PolicyDatasetRecordBody {
            schema_version: partial_stateless::POLICY_DATASET_SCHEMA_VERSION,
            block_number,
            block_hash,
            parent_hash,
            parent_state_root: self.parent_state_root,
            expected_state_root: self.expected_state_root,
            payload_json: Some(payload_json),
            payload_provenance,
            accessed: self.accessed,
            access_provenance: self.access_provenance,
            full_transition_nodes: self.full_transition_nodes,
            ancestor_headers: self.ancestor_headers,
            parent_header,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_conflicts(_: &str) -> bool {
        false
    }

    fn resolve(
        dir: Option<&str>,
        max_blocks: Option<&str>,
    ) -> eyre::Result<Option<PolicyDatasetCaptureConfig>> {
        PolicyDatasetCaptureConfig::resolve(
            dir.map(PathBuf::from),
            max_blocks.map(str::to_string),
            None,
            &no_conflicts,
            AccessCaptureMode::On,
            true,
            0,
        )
    }

    #[test]
    fn a_configured_capture_resolves_to_its_directory_and_budget() {
        let config = resolve(Some("/data/capture"), Some("1200")).unwrap().unwrap();
        assert_eq!(config.dir, PathBuf::from("/data/capture"));
        assert_eq!(config.max_blocks, 1_200);
    }

    /// A dataset resolved against whatever directory the node was launched from cannot be found
    /// again by the offline generator, so the relative path is refused rather than canonicalized.
    #[test]
    fn a_relative_capture_directory_is_refused() {
        assert!(resolve(Some("capture"), Some("1200")).is_err());
    }

    #[test]
    fn a_capture_without_a_block_budget_is_refused() {
        assert!(resolve(Some("/data/capture"), None).is_err());
        assert!(resolve(Some("/data/capture"), Some("0")).is_err());
        assert!(resolve(Some("/data/capture"), Some("lots")).is_err());
    }

    /// Each of these is a job that either wants the whole block budget or reports a per-block
    /// cost, and a capturing run supplies neither.
    #[test]
    fn every_measuring_variable_refuses_to_share_the_process() {
        for (var, _) in CONFLICTING_VARS {
            let refused = PolicyDatasetCaptureConfig::resolve(
                Some(PathBuf::from("/data/capture")),
                Some("1200".to_string()),
                None,
                &|name| name == var,
                AccessCaptureMode::On,
                true,
                0,
            );
            assert!(refused.is_err(), "{var} was allowed alongside a capture");
        }
    }

    /// A corpus recorded off a different execution path, or off a payload this node derived for
    /// itself, describes a system nobody runs and an admission check that checks nothing.
    #[test]
    fn a_capture_requires_both_engine_handoffs() {
        for mode in [AccessCaptureMode::Off, AccessCaptureMode::Shadow] {
            assert!(PolicyDatasetCaptureConfig::resolve(
                Some(PathBuf::from("/data/capture")),
                Some("1200".to_string()),
                None,
                &no_conflicts,
                mode,
                true,
                0,
            )
            .is_err());
        }
        assert!(PolicyDatasetCaptureConfig::resolve(
            Some(PathBuf::from("/data/capture")),
            Some("1200".to_string()),
            None,
            &no_conflicts,
            AccessCaptureMode::On,
            false,
            0,
        )
        .is_err());
    }

    /// A recorder over a fresh directory, standing in for a stamped build.
    ///
    /// The commit is not decoration here: a capture from a build that carries none is refused, so
    /// a recorder built with `None` would exercise that refusal instead of whatever the test is
    /// about.
    fn recorder_at(dir: &Path, max_blocks: u64, confirmations: u64) -> PolicyDatasetRecorder {
        PolicyDatasetRecorder::open(
            Some(PolicyDatasetCaptureConfig { dir: dir.to_path_buf(), max_blocks, confirmations }),
            "test".into(),
            Some("0000000000000000000000000000000000000000".into()),
            "mainnet".into(),
        )
        .expect("a fresh directory opens")
        .expect("a configured capture yields a recorder")
    }

    fn temp_dir(name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("ps-capture-{name}-{nanos}"));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn record(recorder: &mut PolicyDatasetRecorder, number: u64, tag: u8) {
        record_on(recorder, number, tag.wrapping_sub(1), tag);
    }

    fn record_on(recorder: &mut PolicyDatasetRecorder, number: u64, parent_tag: u8, tag: u8) {
        recorder
            .record(partial_stateless::PolicyDatasetRecordBody {
                schema_version: partial_stateless::POLICY_DATASET_SCHEMA_VERSION,
                block_number: number,
                block_hash: B256::repeat_byte(tag),
                parent_hash: B256::repeat_byte(parent_tag),
                parent_state_root: B256::ZERO,
                expected_state_root: B256::ZERO,
                payload_json: Some(b"{}".to_vec()),
                payload_provenance: RecordedPayloadProvenance::Witnessed,
                accessed: BlockAccessedState::default(),
                access_provenance: RecordedAccessProvenance::EngineArtifact,
                full_transition_nodes: Vec::new(),
                ancestor_headers: Vec::new(),
                parent_header: Bytes::new(),
            })
            .expect("the record writes");
    }

    /// Meeting the budget is not finishing. The terminator waits for the chain to advance past the
    /// range, because writing it at the last file would vouch for a tip that is still exposed.
    #[test]
    fn a_dataset_is_not_closed_until_its_range_has_been_confirmed() {
        let dir = temp_dir("confirm");
        let mut recorder = recorder_at(&dir, 2, 3);
        record(&mut recorder, 10, 0x0a);
        record(&mut recorder, 11, 0x0b);
        assert!(!recorder.wants_block(), "the budget is met");
        assert!(!dir.join("END.json").exists(), "closed at the budget rather than at confirmation");

        recorder.observe_head(13);
        assert!(!dir.join("END.json").exists(), "closed one block early");
        recorder.observe_head(14);
        assert!(dir.join("END.json").exists(), "never closed");
    }

    /// A reorg during the confirmation wait reopens the budget: the abandoned record stays on disk
    /// but stops counting, and the capture records a replacement before closing.
    #[test]
    fn a_reorg_under_the_confirmation_wait_reopens_the_budget() {
        let dir = temp_dir("reorg-budget");
        let mut recorder = recorder_at(&dir, 2, 1);
        record(&mut recorder, 10, 0x0a);
        record(&mut recorder, 11, 0x0b);
        assert!(!recorder.wants_block());

        recorder.note_reorg(10, vec![(11, B256::repeat_byte(0x0b))]);
        assert!(recorder.wants_block(), "an abandoned record still spends the budget");
        assert!(!dir.join("END.json").exists());

        record(&mut recorder, 11, 0xb2);
        recorder.observe_head(12);
        assert!(dir.join("END.json").exists(), "the replacement never closed the dataset");
    }

    /// Re-observing a branch after leaving it rewrites no physical record and therefore cannot
    /// inflate the terminator count the loader cross-checks against the directory.
    #[test]
    fn a_branch_that_wins_again_closes_and_loads_through_the_recorder() {
        let dir = temp_dir("reorg-back-recorder");
        let mut recorder = recorder_at(&dir, 3, 1);
        record_on(&mut recorder, 10, 0x09, 0x0a);
        record_on(&mut recorder, 11, 0x0a, 0xa1);

        recorder.note_reorg(10, vec![(11, B256::repeat_byte(0xa1))]);
        record_on(&mut recorder, 11, 0x0a, 0xb1);
        recorder.note_reorg(10, vec![(11, B256::repeat_byte(0xb1))]);
        record_on(&mut recorder, 11, 0x0a, 0xa1);
        record_on(&mut recorder, 12, 0xa1, 0x0c);
        recorder.observe_head(13);

        let loaded = partial_stateless::load_dataset(&dir).expect("the closed dataset loads");
        assert_eq!(loaded.end.records, 4);
        assert_eq!(
            loaded.records.iter().map(|record| record.body.block_hash).collect::<Vec<_>>(),
            vec![B256::repeat_byte(0x0a), B256::repeat_byte(0xa1), B256::repeat_byte(0x0c)]
        );
        assert_eq!(loaded.abandoned.len(), 1);
        assert_eq!(loaded.abandoned[0].body.block_hash, B256::repeat_byte(0xb1));
    }

    /// A run that stops early still leaves a usable corpus: the part the head confirms, and no
    /// more. Vouching for the whole output would vouch for a tip it never saw settle.
    #[test]
    fn an_early_shutdown_vouches_only_for_what_settled() {
        let dir = temp_dir("early-stop");
        let mut recorder = recorder_at(&dir, 100, 2);
        for (number, tag) in [(10, 0x0a), (11, 0x0b), (12, 0x0c)] {
            record(&mut recorder, number, tag);
        }
        recorder.close(DatasetEndKind::ProducerShutdown, "stopped".into());

        let end: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("END.json")).unwrap()).unwrap();
        assert_eq!(end["records"], 3);
        // Head reached 12, confirmations 2, so only block 10 is settled.
        assert_eq!(end["usable_range"][0], 10);
        assert_eq!(end["usable_range"][1], 10);
    }

    /// The property the whole opt-in contract rests on: a process nobody configured for capture
    /// resolves to `None`, and every downstream cost is behind that `None`.
    #[test]
    fn an_unconfigured_process_captures_nothing() {
        assert!(std::env::var_os(CAPTURE_DIR_VAR).is_none());
        assert!(PolicyDatasetCaptureConfig::from_env().expect("no capture configured").is_none());
        assert!(PolicyDatasetRecorder::open(None, "test".into(), None, "mainnet".into())
            .expect("no capture configured")
            .is_none());
    }

    #[test]
    fn a_derived_payload_is_refused_rather_than_recorded() {
        let material = PolicyDatasetMaterial {
            parent_state_root: B256::ZERO,
            expected_state_root: B256::ZERO,
            accessed: BlockAccessedState::default(),
            access_provenance: RecordedAccessProvenance::EngineArtifact,
            full_transition_nodes: Vec::new(),
            ancestor_headers: Vec::new(),
        };
        let refused = material.into_record_body(
            10,
            B256::ZERO,
            B256::ZERO,
            Bytes::new(),
            Some(b"{}".to_vec()),
            RecordedPayloadProvenance::Reconstructed,
        );
        assert_eq!(
            refused.unwrap_err(),
            PolicyDatasetMaterialError::PayloadHandoffMiss {
                block_number: 10,
                provenance: RecordedPayloadProvenance::Reconstructed,
            }
        );
    }

    /// A handoff miss falls back to this builder re-executing, which is a different execution path
    /// than the one the corpus claims every record came from.
    #[test]
    fn a_re_executed_access_set_is_refused_rather_than_recorded() {
        let material = PolicyDatasetMaterial {
            parent_state_root: B256::ZERO,
            expected_state_root: B256::ZERO,
            accessed: BlockAccessedState::default(),
            access_provenance: RecordedAccessProvenance::Reexecution,
            full_transition_nodes: Vec::new(),
            ancestor_headers: Vec::new(),
        };
        let refused = material.into_record_body(
            10,
            B256::ZERO,
            B256::ZERO,
            Bytes::new(),
            Some(b"{}".to_vec()),
            RecordedPayloadProvenance::Witnessed,
        );
        assert_eq!(
            refused.unwrap_err(),
            PolicyDatasetMaterialError::AccessHandoffMiss {
                block_number: 10,
                provenance: RecordedAccessProvenance::Reexecution,
            }
        );
    }

    /// The caller recovers the class of a refusal by downcasting the `eyre::Report` that
    /// `record_dataset_block` returns. If that ever stopped working the skip would silently become
    /// unreachable and a cold start would kill the node again, so it is asserted rather than
    /// assumed.
    #[test]
    fn a_material_refusal_survives_the_trip_through_eyre() {
        fn as_report() -> eyre::Result<()> {
            let material = PolicyDatasetMaterial {
                parent_state_root: B256::ZERO,
                expected_state_root: B256::ZERO,
                accessed: BlockAccessedState::default(),
                access_provenance: RecordedAccessProvenance::EngineArtifact,
                full_transition_nodes: Vec::new(),
                ancestor_headers: Vec::new(),
            };
            material.into_record_body(
                10,
                B256::ZERO,
                B256::ZERO,
                Bytes::new(),
                None,
                RecordedPayloadProvenance::Reconstructed,
            )?;
            Ok(())
        }
        let report = as_report().unwrap_err();
        assert_eq!(
            report.downcast_ref::<PolicyDatasetMaterialError>().copied(),
            Some(PolicyDatasetMaterialError::PayloadHandoffMiss {
                block_number: 10,
                provenance: RecordedPayloadProvenance::Reconstructed,
            }),
            "the caller's downcast cannot see the refusal class"
        );

        // And an unrelated failure must not look like one.
        let unrelated: eyre::Report = eyre::eyre!("provider read failed");
        assert!(unrelated.downcast_ref::<PolicyDatasetMaterialError>().is_none());
    }

    /// A claimed handoff with no bytes is an internal invariant failure, not an evicted ring item.
    #[test]
    fn missing_witnessed_payload_is_not_a_startup_handoff_miss() {
        let material = PolicyDatasetMaterial {
            parent_state_root: B256::ZERO,
            expected_state_root: B256::ZERO,
            accessed: BlockAccessedState::default(),
            access_provenance: RecordedAccessProvenance::EngineArtifact,
            full_transition_nodes: Vec::new(),
            ancestor_headers: Vec::new(),
        };
        let refused = material
            .into_record_body(
                10,
                B256::ZERO,
                B256::ZERO,
                Bytes::new(),
                None,
                RecordedPayloadProvenance::Witnessed,
            )
            .unwrap_err();
        assert_eq!(
            refused,
            PolicyDatasetMaterialError::MissingWitnessedPayload { block_number: 10 }
        );

        let dir = temp_dir("missing-witnessed-payload");
        let mut recorder = recorder_at(&dir, 4, 0);
        assert!(!recorder.skip_startup_handoff_miss(10, &refused));
    }

    /// Sampling records the sampled block's own re-executed set, so a capture must refuse it up
    /// front rather than discover it two percent of the way through a corpus.
    #[test]
    fn a_capture_refuses_a_nonzero_shadow_sample() {
        assert!(PolicyDatasetCaptureConfig::resolve(
            Some(PathBuf::from("/data/capture")),
            Some("1200".to_string()),
            None,
            &no_conflicts,
            AccessCaptureMode::On,
            true,
            50,
        )
        .is_err());
    }

    /// The capture arrives after the node does, and the Engine taps keep moving while it does.
    /// Losing the run over the first block it lands on would cost hours for a condition that has
    /// passed by the next block.
    #[test]
    fn a_block_refused_before_the_corpus_starts_lets_the_run_continue() {
        let dir = temp_dir("skip-before-start");
        let mut recorder = recorder_at(&dir, 4, 0);
        for block_number in [10, 11] {
            let miss = PolicyDatasetMaterialError::PayloadHandoffMiss {
                block_number,
                provenance: RecordedPayloadProvenance::Absent,
            };
            assert!(recorder.skip_startup_handoff_miss(block_number, &miss));
        }

        // And the corpus that follows is whole, starting where the capture actually began.
        for number in 12..=15u64 {
            record(&mut recorder, number, number as u8);
        }
        assert_eq!(recorder.settled_range().map(|(low, high, _)| (low, high)), Some((12, 15)));

        let log = std::fs::read_to_string(dir.join("lifecycle.jsonl")).unwrap();
        assert_eq!(
            log.lines().filter(|line| line.contains("\"skipped\"")).count(),
            2,
            "the skipped blocks were not filed for the reader to see: {log}"
        );
    }

    /// The same refusal after a record exists would put a hole in the middle of the corpus, and a
    /// corpus with a hole is worse than none: it looks complete.
    #[test]
    fn a_block_refused_after_the_corpus_starts_is_fatal() {
        let dir = temp_dir("skip-after-start");
        let mut recorder = recorder_at(&dir, 4, 0);
        record(&mut recorder, 10, 0x0a);
        let miss = PolicyDatasetMaterialError::PayloadHandoffMiss {
            block_number: 11,
            provenance: RecordedPayloadProvenance::Absent,
        };
        assert!(!recorder.skip_startup_handoff_miss(11, &miss));
    }

    /// "Not yet" and "never" look identical from inside the wait, so the wait is bounded.
    #[test]
    fn waiting_for_a_corpus_to_start_gives_up_eventually() {
        let dir = temp_dir("skip-budget");
        let mut recorder = recorder_at(&dir, 4, 0);
        for block in 0..MAX_SKIPS_BEFORE_FIRST_RECORD {
            let miss = PolicyDatasetMaterialError::PayloadHandoffMiss {
                block_number: block,
                provenance: RecordedPayloadProvenance::Absent,
            };
            assert!(recorder.skip_startup_handoff_miss(block, &miss), "gave up at {block}");
        }
        let miss = PolicyDatasetMaterialError::PayloadHandoffMiss {
            block_number: MAX_SKIPS_BEFORE_FIRST_RECORD,
            provenance: RecordedPayloadProvenance::Absent,
        };
        assert!(
            !recorder.skip_startup_handoff_miss(MAX_SKIPS_BEFORE_FIRST_RECORD, &miss),
            "the wait was unbounded"
        );
    }

    /// A corpus that cannot name the code that produced it is not evidence, and the moment to
    /// say so is before the hours are spent rather than after.
    #[test]
    fn a_capture_from_an_unstamped_build_is_refused() {
        assert!(require_stamped_build(None, false).is_err());
        assert!(require_stamped_build(Some("  "), false).is_err(), "a blank stamp is no stamp");
        require_stamped_build(Some("dc225dbf50"), false).unwrap();
        require_stamped_build(None, true).expect("the smoke-test escape hatch is closed");
    }

    #[test]
    fn a_confirmation_depth_defaults_and_parses() {
        assert_eq!(
            resolve(Some("/data/capture"), Some("10")).unwrap().unwrap().confirmations,
            DEFAULT_CONFIRMATIONS
        );
        let explicit = PolicyDatasetCaptureConfig::resolve(
            Some(PathBuf::from("/data/capture")),
            Some("10".to_string()),
            Some("0".to_string()),
            &no_conflicts,
            AccessCaptureMode::On,
            true,
            0,
        )
        .unwrap()
        .unwrap();
        assert_eq!(explicit.confirmations, 0);
        assert!(PolicyDatasetCaptureConfig::resolve(
            Some(PathBuf::from("/data/capture")),
            Some("10".to_string()),
            Some("soon".to_string()),
            &no_conflicts,
            AccessCaptureMode::On,
            true,
            0,
        )
        .is_err());
    }
}
