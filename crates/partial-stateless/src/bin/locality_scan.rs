//! Reuse distance and coverage-versus-window over a recorded policy replay dataset.
//!
//! This is the offline measurement behind the paper's motivation section, and it exists so that
//! the motivation and the result are the same number seen twice rather than two numbers that
//! happen to point the same way. The frontier generator answers "how many bytes does policy
//! `a/s` ship"; this answers "how much of what a block touches did the last `N` blocks already
//! touch". They are complements, and this tool is written so that the complement is checkable:
//! its per-block hit counts are emitted at the windows the frontier's arms used, so a reader can
//! diff the two files rather than take the identity on faith.
//!
//! **The window is `N + 1` heights, not `N`.** [`partial_stateless::LastNBlocksPolicy`] retains an
//! entry whose `last_accessed_block` is at or above `current_block - window_size`, and eviction
//! runs with the block that was just applied. A cache about to look up block `B` has applied
//! through `B - 1`, so it holds the *closed* range `[B - 1 - N, B - 1]`. Coverage at window
//! parameter `N` is therefore over `N + 1` prior heights, and a key's reuse distance `d` is
//! covered when `d <= N + 1`. Every curve here uses that convention, which is what makes it the
//! exact complement of the generator's miss counts.
//!
//! **The unit is a `(block, key)` access-set incidence.** The input is the per-block
//! `BlockAccessedState`, whose fields are `HashMap`s, so a key touched forty times inside one block
//! counts once. This measures inter-block locality — which is the only thing `LastNBlocksPolicy`
//! can act on, since it evicts on `last_accessed_block` — and says nothing about per-transaction
//! locality or intra-block repeat frequency.
//!
//! **Bytes are a logical key-plus-value payload size, not proof paths and not a footprint.** The
//! weighting includes the key (address, slot key, code hash) and excludes trie nodes, proof paths
//! and sidecar framing. It is deliberately *not* the cache's own `estimated_memory_bytes`
//! accounting and not a serialized size; it exists to separate bytecode from fixed-width entries.
//! That makes the slot byte curve identical to its key-count curve by construction and leaves
//! bytecode as the category where the two diverge, which is the effect worth isolating. A
//! proof-path-weighted curve is a different measurement and needs the witness, not the access set.
//!
//! Fully offline: the only input is a recorded policy replay dataset. Records are processed one at
//! a time rather than loaded as a set, because the corpus is larger than this host's free memory.
//! Each record is fully deserialized and its seal digest verified — which hashes the witness bytes
//! — and is then reduced to its access set and dropped; the witness is never *used*, but it is
//! read.
//!
//! Usage:
//!   locality_scan --dataset <dir> --warmup <n> --samples <n> [--max-window <n>] --out <dir>

use std::{
    collections::HashMap,
    error::Error,
    fs,
    path::{Path, PathBuf},
    process,
    time::Instant,
};

use alloy_primitives::{keccak256, Address, B256};
use partial_stateless::{
    DatasetEnd, DatasetEndKind, PolicyDatasetManifest, PolicyDatasetRecord,
    POLICY_DATASET_SCHEMA_VERSION,
};
use serde::Serialize;

/// Windows the per-block stream reports hits at.
///
/// Chosen to be dense where the curve bends and to *include* every window the frontier's arms
/// used (30, 45, 60, 90, 120), because those five are what make the cross-check against
/// `frontier.jsonl` a per-block equality rather than an aggregate coincidence.
const LADDER: &[u64] =
    &[0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 32, 45, 60, 75, 90, 105, 120];

/// Bytes charged to one account entry: the address, plus the account record the cache holds.
///
/// `AccountData` is `nonce: u64`, `balance: U256`, `code_hash: Option<B256>`, and the option is
/// charged only when it is populated — `KECCAK_EMPTY` is normalized to `None` upstream, so a
/// codeless account genuinely carries no code hash.
fn account_entry_bytes(has_code_hash: bool) -> u64 {
    20 + 8 + 32 + if has_code_hash { 32 } else { 0 }
}

/// Bytes charged to one storage entry: the `(address, slot)` key and the 32-byte word.
const STORAGE_ENTRY_BYTES: u64 = 20 + 32 + 32;

/// Bytes charged to one bytecode entry: the code hash and the bytecode itself.
fn code_entry_bytes(code_len: usize) -> u64 {
    32 + code_len as u64
}

/// One block reduced to what a locality scan needs, with the witness dropped.
struct BlockSummary {
    block_number: u64,
    block_hash: B256,
    parent_hash: B256,
    /// Accessed accounts and the bytes each is charged.
    accounts: Vec<(Address, u64)>,
    /// Accessed slots and the bytes each is charged.
    storage: Vec<((Address, B256), u64)>,
    /// Accessed bytecodes and the bytes each is charged.
    codes: Vec<(B256, u64)>,
}

/// A reuse-distance histogram for one state category, counted and byte-weighted together.
///
/// Indexed by distance `d` in `1..=max_distance`; `d` is the number of blocks since the key was
/// last accessed, so `d = 1` means "the immediately preceding block also touched it".
///
/// The overflow bucket is a *mixture* and its name says so: `no_reuse_within` pools keys whose
/// observed reuse distance exceeds the histogram with keys that have no earlier access anywhere in
/// the observed prefix. Only the second is unresolvable — for the first the distance is known and
/// merely out of range. Reporting it as one right-censored mass would overstate what is unknown,
/// and reading it as a first-touch rate would understate it.
#[derive(Clone)]
struct Hist {
    count: Vec<u64>,
    bytes: Vec<u64>,
    no_reuse_within_count: u64,
    no_reuse_within_bytes: u64,
    total_count: u64,
    total_bytes: u64,
}

impl Hist {
    fn new(max_distance: u64) -> Self {
        let len = max_distance as usize + 1;
        Self {
            count: vec![0; len],
            bytes: vec![0; len],
            no_reuse_within_count: 0,
            no_reuse_within_bytes: 0,
            total_count: 0,
            total_bytes: 0,
        }
    }

    fn observe(&mut self, distance: Option<u64>, entry_bytes: u64) {
        self.total_count += 1;
        self.total_bytes += entry_bytes;
        match distance {
            Some(d) if (d as usize) < self.count.len() => {
                self.count[d as usize] += 1;
                self.bytes[d as usize] += entry_bytes;
            }
            _ => {
                self.no_reuse_within_count += 1;
                self.no_reuse_within_bytes += entry_bytes;
            }
        }
    }

    fn merge(&mut self, other: &Self) {
        for (slot, add) in self.count.iter_mut().zip(&other.count) {
            *slot += add;
        }
        for (slot, add) in self.bytes.iter_mut().zip(&other.bytes) {
            *slot += add;
        }
        self.no_reuse_within_count += other.no_reuse_within_count;
        self.no_reuse_within_bytes += other.no_reuse_within_bytes;
        self.total_count += other.total_count;
        self.total_bytes += other.total_bytes;
    }

    fn clear(&mut self) {
        self.count.iter_mut().for_each(|slot| *slot = 0);
        self.bytes.iter_mut().for_each(|slot| *slot = 0);
        self.no_reuse_within_count = 0;
        self.no_reuse_within_bytes = 0;
        self.total_count = 0;
        self.total_bytes = 0;
    }

    /// Keys covered by window parameter `window`, i.e. reuse distance `<= window + 1`.
    fn hits(&self, window: u64) -> (u64, u64) {
        let ceiling = (window + 1).min(self.count.len() as u64 - 1) as usize;
        let count = self.count[..=ceiling].iter().sum();
        let bytes = self.bytes[..=ceiling].iter().sum();
        (count, bytes)
    }
}

/// The curve one category contributes to the summary.
#[derive(Serialize)]
struct CategoryCurve {
    /// Accessed keys over the measured blocks.
    total_keys: u64,
    /// Accessed value bytes over the measured blocks.
    total_bytes: u64,
    /// Keys with no reuse inside the histogram: observed distance out of range, or no earlier
    /// access in the observed prefix. A mixture, deliberately not called "censored".
    no_reuse_within_keys: u64,
    no_reuse_within_bytes: u64,
    /// `hist[d]` for `d` in `0..=max_distance`; index 0 is always zero and exists so the array
    /// index *is* the distance.
    distance_count: Vec<u64>,
    distance_bytes: Vec<u64>,
    /// Coverage at window parameter `N`, for `N` in `0..=max_window`, pooled over every measured
    /// block (a ratio of sums, which is how the generator's miss rates are formed too).
    coverage_by_count: Vec<f64>,
    coverage_by_bytes: Vec<f64>,
}

impl CategoryCurve {
    fn from_hist(hist: &Hist, max_window: u64) -> Self {
        let mut coverage_by_count = Vec::with_capacity(max_window as usize + 1);
        let mut coverage_by_bytes = Vec::with_capacity(max_window as usize + 1);
        for window in 0..=max_window {
            let (count, bytes) = hist.hits(window);
            coverage_by_count.push(if hist.total_count > 0 {
                count as f64 / hist.total_count as f64
            } else {
                0.0
            });
            coverage_by_bytes.push(if hist.total_bytes > 0 {
                bytes as f64 / hist.total_bytes as f64
            } else {
                0.0
            });
        }
        Self {
            total_keys: hist.total_count,
            total_bytes: hist.total_bytes,
            no_reuse_within_keys: hist.no_reuse_within_count,
            no_reuse_within_bytes: hist.no_reuse_within_bytes,
            distance_count: hist.count.clone(),
            distance_bytes: hist.bytes.clone(),
            coverage_by_count,
            coverage_by_bytes,
        }
    }
}

/// The whole run, as the summary file records it.
#[derive(Serialize)]
struct Summary {
    tool: &'static str,
    dataset: String,
    producer: String,
    producer_build_commit: Option<String>,
    dataset_schema_version: u32,
    canonical_blocks: usize,
    abandoned_or_unconfirmed_blocks: usize,
    warmup_blocks: u64,
    measured_blocks: u64,
    measured_range: (u64, u64),
    /// keccak256 over the measured block hashes in order — the same construction the frontier
    /// generator uses, so two files that agree here measured the same block set.
    measured_block_set_digest: B256,
    max_window: u64,
    /// The `N + 1` convention, stated in the artifact rather than only in the prose that reads it.
    window_covers_heights: &'static str,
    byte_weighting: &'static str,
    accounts: CategoryCurve,
    storage: CategoryCurve,
    codes: CategoryCurve,
    /// The three categories pooled, which is the curve that predicts a value-bytes saving.
    all: CategoryCurve,
    scan_seconds: f64,
}

/// Per-block record, emitted so the identity against `frontier.jsonl` is checkable block by block.
#[derive(Serialize)]
struct PerBlock {
    block_number: u64,
    block_hash: B256,
    accessed_accounts: u64,
    accessed_storage: u64,
    accessed_codes: u64,
    accessed_account_bytes: u64,
    accessed_storage_bytes: u64,
    accessed_code_bytes: u64,
    /// The windows `hits_*` is indexed by.
    windows: &'static [u64],
    hit_accounts: Vec<u64>,
    hit_storage: Vec<u64>,
    hit_codes: Vec<u64>,
    hit_account_bytes: Vec<u64>,
    hit_storage_bytes: Vec<u64>,
    hit_code_bytes: Vec<u64>,
}

fn usage() -> &'static str {
    "usage: locality_scan --dataset <dir> --warmup <n> --samples <n> [--max-window <n>] \
     --out <dir>"
}

struct Options {
    dataset: PathBuf,
    warmup: u64,
    samples: u64,
    max_window: u64,
    out: PathBuf,
}

fn parse_options() -> Result<Options, Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let mut dataset = None;
    let mut warmup = None;
    let mut samples = None;
    let mut max_window = 120u64;
    let mut out = None;

    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value\n{}", usage()));
        match flag.as_str() {
            "--dataset" => dataset = Some(PathBuf::from(value()?)),
            "--warmup" => warmup = Some(value()?.parse::<u64>()?),
            "--samples" => samples = Some(value()?.parse::<u64>()?),
            "--max-window" => max_window = value()?.parse::<u64>()?,
            "--out" => out = Some(PathBuf::from(value()?)),
            "-h" | "--help" => {
                println!("{}", usage());
                process::exit(0);
            }
            other => return Err(format!("unknown flag `{other}`\n{}", usage()).into()),
        }
    }

    Ok(Options {
        dataset: dataset.ok_or_else(|| format!("--dataset is required\n{}", usage()))?,
        warmup: warmup.ok_or_else(|| format!("--warmup is required\n{}", usage()))?,
        samples: samples.ok_or_else(|| format!("--samples is required\n{}", usage()))?,
        max_window,
        out: out.ok_or_else(|| format!("--out is required\n{}", usage()))?,
    })
}

/// Streams every record in the dataset directory, reducing each to its access set.
///
/// This is deliberately not [`partial_stateless::load_dataset`]: that holds whole records, and the
/// whole corpus does not fit in this host's free memory. Each record is still fully deserialized
/// and authenticated — the seal digest covers every byte, witness included — before being reduced
/// to the access set and dropped.
///
/// It verifies a strict *subset* of what `load_dataset` does: the per-record seal digest and
/// schema version here, plus the record count and the canonical `parent_hash` walk below. Not
/// re-derived: the manifest schema version, the lifecycle log, the terminator's
/// `block_range`-versus-records cross-check, the inverted-range check, and the confirmation-depth
/// claim. Those gate whether a corpus may be measured at all, which is a question already answered
/// for any corpus a frontier run has accepted — and the reported block-set digest is what proves
/// this scan landed on the same blocks. Point this tool at an unvetted corpus and it will be less
/// careful than `load_dataset` is.
fn stream_summaries(root: &Path) -> Result<Vec<BlockSummary>, Box<dyn Error>> {
    let blocks_dir = root.join("blocks");
    let mut paths = Vec::new();
    for entry in fs::read_dir(&blocks_dir)? {
        let path = entry?.path();
        let is_record = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("block_") && name.ends_with(".bin"));
        if is_record {
            paths.push(path);
        }
    }
    paths.sort();

    let mut summaries = Vec::with_capacity(paths.len());
    for path in &paths {
        let bytes = fs::read(path)?;
        let record: PolicyDatasetRecord = bincode::deserialize(&bytes)?;
        record.verify_digest()?;
        if record.body.schema_version != POLICY_DATASET_SCHEMA_VERSION {
            return Err(format!(
                "block {} was written by schema {} but this build reads {}",
                record.body.block_number, record.body.schema_version, POLICY_DATASET_SCHEMA_VERSION
            )
            .into())
        }
        let body = record.body;
        let accounts = body
            .accessed
            .accounts
            .iter()
            .map(|(address, data)| (*address, account_entry_bytes(data.code_hash.is_some())))
            .collect();
        let storage = body.accessed.storage.keys().map(|key| (*key, STORAGE_ENTRY_BYTES)).collect();
        let codes = body
            .accessed
            .codes
            .iter()
            .map(|(hash, code)| (*hash, code_entry_bytes(code.len())))
            .collect();
        summaries.push(BlockSummary {
            block_number: body.block_number,
            block_hash: body.block_hash,
            parent_hash: body.parent_hash,
            accounts,
            storage,
            codes,
        });
    }
    Ok(summaries)
}

/// Reduces the streamed summaries to the canonical chain the terminator vouches for.
///
/// Mirrors `policy_dataset::canonical_chain`: the winning branch is decided by walking
/// `parent_hash` down from the terminator's usable tip, not by trusting the lifecycle log, so a
/// chain that reorganised away and back is still read correctly.
fn canonical_chain(
    end: &DatasetEnd,
    all: Vec<BlockSummary>,
) -> Result<(Vec<BlockSummary>, usize), Box<dyn Error>> {
    let total = all.len();
    let Some((low, high)) = end.usable_range else {
        return Err("the terminator vouches for no usable range".into())
    };
    let tip_hash = end
        .usable_tip_hash
        .ok_or("the terminator names a usable range but no tip hash to walk down from")?;

    if end.records != total as u64 {
        return Err(format!(
            "the terminator claims {} records, but {total} are present",
            end.records
        )
        .into())
    }

    let mut by_hash: HashMap<B256, BlockSummary> =
        all.into_iter().map(|summary| (summary.block_hash, summary)).collect();

    let mut chain = Vec::new();
    let mut wanted = tip_hash;
    let mut height = high;
    loop {
        let summary = by_hash
            .remove(&wanted)
            .ok_or_else(|| format!("the dataset has a gap at height {height}"))?;
        if summary.block_number != height {
            return Err(format!(
                "broken chain: hash {wanted:?} should be height {height} but is {}",
                summary.block_number
            )
            .into())
        }
        wanted = summary.parent_hash;
        chain.push(summary);
        if height == low {
            break
        }
        height -= 1;
    }
    chain.reverse();
    let excluded = total - chain.len();
    Ok((chain, excluded))
}

fn main() {
    if let Err(err) = run() {
        eprintln!("locality_scan: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let options = parse_options()?;
    let started = Instant::now();

    let manifest: PolicyDatasetManifest =
        serde_json::from_slice(&fs::read(options.dataset.join("manifest.json"))?)?;
    let end: DatasetEnd = serde_json::from_slice(&fs::read(options.dataset.join("END.json"))?)?;
    if end.kind == DatasetEndKind::Failed {
        return Err("the capture recorded its own failure; it must not be measured".into())
    }

    eprintln!("locality_scan: streaming records from {}", options.dataset.display());
    let all = stream_summaries(&options.dataset)?;
    let (canonical, excluded) = canonical_chain(&end, all)?;
    eprintln!(
        "locality_scan: {} canonical blocks ({} excluded), streamed in {:.1}s",
        canonical.len(),
        excluded,
        started.elapsed().as_secs_f64()
    );

    let needed = options.warmup + options.samples;
    if (canonical.len() as u64) < needed {
        return Err(format!(
            "dataset holds {} canonical blocks but {needed} are needed ({} warm-up + {} measured)",
            canonical.len(),
            options.warmup,
            options.samples
        )
        .into())
    }

    // A window of `N` reaches `N + 1` heights back, so the histogram must reach one further than
    // the widest window any curve reports.
    let max_distance = options.max_window + 1;

    // The widest window can reach further back than the warm-up goes, and when it does the first
    // measured blocks are scored against a history that is not there yet. Those keys are counted
    // as un-reused when the truth is unobservable, which biases the tail of the curve downward.
    //
    // Warned rather than refused, because the boundary case is legitimate and is the one that gets
    // run: holding a measured block set fixed pins the warm-up, and asking for `--max-window` equal
    // to it costs exactly one height at exactly the first block. What must not happen is that cost
    // going unrecorded, so the shortfall is named here and belongs in whatever reports the curve.
    if max_distance > options.warmup {
        let short_blocks = max_distance - options.warmup;
        eprintln!(
            "locality_scan: WARNING --max-window {} reaches {max_distance} heights back but the \
             warm-up supplies only {}; the first {short_blocks} measured block(s) are scored \
             against an incomplete history and the widest windows are biased low there. Reduce \
             --max-window to {} to remove the effect, or report the shortfall.",
            options.max_window,
            options.warmup,
            options.warmup.saturating_sub(1),
        );
    }

    let mut last_account: HashMap<Address, u64> = HashMap::new();
    let mut last_storage: HashMap<(Address, B256), u64> = HashMap::new();
    let mut last_code: HashMap<B256, u64> = HashMap::new();

    let mut acc_accounts = Hist::new(max_distance);
    let mut acc_storage = Hist::new(max_distance);
    let mut acc_codes = Hist::new(max_distance);
    let mut acc_all = Hist::new(max_distance);

    let mut blk_accounts = Hist::new(max_distance);
    let mut blk_storage = Hist::new(max_distance);
    let mut blk_codes = Hist::new(max_distance);

    fs::create_dir_all(&options.out)?;
    let mut per_block_out = String::new();
    let mut measured_hashes = Vec::new();
    let mut measured_range = None;

    for (index, block) in canonical.iter().enumerate() {
        let index = index as u64;
        let measured = index >= options.warmup && index < options.warmup + options.samples;

        if measured {
            blk_accounts.clear();
            blk_storage.clear();
            blk_codes.clear();

            for (address, entry_bytes) in &block.accounts {
                let distance = last_account.get(address).map(|seen| block.block_number - seen);
                blk_accounts.observe(distance, *entry_bytes);
            }
            for (key, entry_bytes) in &block.storage {
                let distance = last_storage.get(key).map(|seen| block.block_number - seen);
                blk_storage.observe(distance, *entry_bytes);
            }
            for (hash, entry_bytes) in &block.codes {
                let distance = last_code.get(hash).map(|seen| block.block_number - seen);
                blk_codes.observe(distance, *entry_bytes);
            }

            acc_accounts.merge(&blk_accounts);
            acc_storage.merge(&blk_storage);
            acc_codes.merge(&blk_codes);
            acc_all.merge(&blk_accounts);
            acc_all.merge(&blk_storage);
            acc_all.merge(&blk_codes);

            let mut hit_accounts = Vec::with_capacity(LADDER.len());
            let mut hit_storage = Vec::with_capacity(LADDER.len());
            let mut hit_codes = Vec::with_capacity(LADDER.len());
            let mut hit_account_bytes = Vec::with_capacity(LADDER.len());
            let mut hit_storage_bytes = Vec::with_capacity(LADDER.len());
            let mut hit_code_bytes = Vec::with_capacity(LADDER.len());
            for window in LADDER {
                let (count, bytes) = blk_accounts.hits(*window);
                hit_accounts.push(count);
                hit_account_bytes.push(bytes);
                let (count, bytes) = blk_storage.hits(*window);
                hit_storage.push(count);
                hit_storage_bytes.push(bytes);
                let (count, bytes) = blk_codes.hits(*window);
                hit_codes.push(count);
                hit_code_bytes.push(bytes);
            }

            let record = PerBlock {
                block_number: block.block_number,
                block_hash: block.block_hash,
                accessed_accounts: blk_accounts.total_count,
                accessed_storage: blk_storage.total_count,
                accessed_codes: blk_codes.total_count,
                accessed_account_bytes: blk_accounts.total_bytes,
                accessed_storage_bytes: blk_storage.total_bytes,
                accessed_code_bytes: blk_codes.total_bytes,
                windows: LADDER,
                hit_accounts,
                hit_storage,
                hit_codes,
                hit_account_bytes,
                hit_storage_bytes,
                hit_code_bytes,
            };
            per_block_out.push_str(&serde_json::to_string(&record)?);
            per_block_out.push('\n');

            measured_hashes.extend_from_slice(block.block_hash.as_slice());
            let range = measured_range.get_or_insert((block.block_number, block.block_number));
            range.1 = block.block_number;
        }

        // Applied after the lookup, exactly as the cache applies a block after building against
        // the generation the block arrived at.
        for (address, _) in &block.accounts {
            last_account.insert(*address, block.block_number);
        }
        for (key, _) in &block.storage {
            last_storage.insert(*key, block.block_number);
        }
        for (hash, _) in &block.codes {
            last_code.insert(*hash, block.block_number);
        }
    }

    let summary = Summary {
        tool: "locality_scan",
        dataset: options.dataset.display().to_string(),
        producer: manifest.producer.clone(),
        producer_build_commit: manifest.build_commit.clone(),
        dataset_schema_version: manifest.schema_version,
        canonical_blocks: canonical.len(),
        abandoned_or_unconfirmed_blocks: excluded,
        warmup_blocks: options.warmup,
        measured_blocks: options.samples,
        measured_range: measured_range.ok_or("no measured blocks")?,
        measured_block_set_digest: keccak256(&measured_hashes),
        max_window: options.max_window,
        window_covers_heights: "window parameter N covers the closed range [B-1-N, B-1], i.e. N+1 \
                                heights; a key is covered when its reuse distance d satisfies \
                                d <= N+1",
        byte_weighting: "value bytes only: account = 20+8+32(+32 if code_hash), slot = 20+32+32, \
                         code = 32+len(bytecode). No trie nodes, no proof paths, no sidecar \
                         framing.",
        accounts: CategoryCurve::from_hist(&acc_accounts, options.max_window),
        storage: CategoryCurve::from_hist(&acc_storage, options.max_window),
        codes: CategoryCurve::from_hist(&acc_codes, options.max_window),
        all: CategoryCurve::from_hist(&acc_all, options.max_window),
        scan_seconds: started.elapsed().as_secs_f64(),
    };

    fs::write(options.out.join("locality-per-block.jsonl"), per_block_out)?;
    fs::write(options.out.join("locality-summary.json"), serde_json::to_vec_pretty(&summary)?)?;

    // A flat curve file, because every consumer of this run is a plot or a table and neither
    // wants to walk a nested document to find one column.
    let mut csv = String::from(
        "window_n,acct_cov_count,acct_cov_bytes,slot_cov_count,slot_cov_bytes,code_cov_count,\
         code_cov_bytes,all_cov_count,all_cov_bytes\n",
    );
    for window in 0..=options.max_window as usize {
        csv.push_str(&format!(
            "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            window,
            summary.accounts.coverage_by_count[window],
            summary.accounts.coverage_by_bytes[window],
            summary.storage.coverage_by_count[window],
            summary.storage.coverage_by_bytes[window],
            summary.codes.coverage_by_count[window],
            summary.codes.coverage_by_bytes[window],
            summary.all.coverage_by_count[window],
            summary.all.coverage_by_bytes[window],
        ));
    }
    fs::write(options.out.join("locality-curve.csv"), csv)?;

    eprintln!(
        "locality_scan: {} measured blocks {}..={}, digest {:?}, {:.1}s",
        summary.measured_blocks,
        summary.measured_range.0,
        summary.measured_range.1,
        summary.measured_block_set_digest,
        summary.scan_seconds
    );
    Ok(())
}
