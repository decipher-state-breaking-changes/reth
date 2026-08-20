//! Replays a recorded corpus and prices each block's cache-complement witness under the MPT's
//! successors.
//!
//! Reads the corpus and, when one is named, the recorded frontier run it must agree with. No
//! provider, no state database, no network. The run is fully determined by the corpus, that
//! comparison input, and the parameters echoed into its report.

use eyre::{bail, Context, Result};
use partial_stateless_treestudy::{
    corpus::Corpus,
    keys::HeaderLayout,
    report::RunReport,
    study::{price_block, Arm, ArmSpec, BlockResult, CensusBuilder, Populations, StudyParams},
    witness::StemOccupancy,
};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let options = Options::parse()?;
    let corpus = Corpus::open(&options.dataset)?;
    let (first, last) = corpus.range();
    tracing::info!(blocks = corpus.len(), first, last, "corpus opened");

    let recorded = match &options.frontier {
        Some(path) => {
            let recorded = RecordedRun::load(path)?;
            tracing::info!(
                pairs = recorded.len(),
                digest = %recorded.digest(),
                "recorded run loaded; miss sets will be checked pair by pair"
            );
            Some(recorded)
        }
        None => {
            tracing::warn!(
                "no --frontier given: the miss set will not be checked against a recorded run"
            );
            None
        }
    };

    let specs = vec![
        ArmSpec::weak(),
        ArmSpec::windows(60, 30),
        ArmSpec::windows(90, 60),
        ArmSpec::windows(120, 45),
    ];

    let params = options.params.clone();
    let populations = Populations::new(&params);
    let mut arms: Vec<Arm> =
        specs.iter().cloned().map(|spec| Arm::new(spec, params.header_layout)).collect();
    let mut results: BTreeMap<String, Vec<BlockResult>> = BTreeMap::new();
    let mut census = CensusBuilder::new(params.header_layout);
    let mut checked = 0usize;

    let started = Instant::now();
    let mut seen = 0usize;
    let total = options.limit.unwrap_or(corpus.len()).min(corpus.len());
    corpus.for_each(options.limit, |block| {
        census.observe(&block.accessed);
        let measuring = seen >= options.warmup;
        for arm in &mut arms {
            if measuring {
                let row = price_block(arm, &block.accessed, &params, &populations, block.number);
                if let Some(recorded) = &recorded {
                    checked += usize::from(recorded.check(&row)?);
                }
                results.entry(arm.name().to_string()).or_default().push(row);
            }
            arm.advance(block.number, &block.accessed);
        }
        seen += 1;
        if seen.is_multiple_of(100) {
            tracing::info!(
                done = seen,
                total,
                measured = seen.saturating_sub(options.warmup),
                elapsed_s = started.elapsed().as_secs(),
                "progress"
            );
        }
        Ok(())
    })?;

    if seen <= options.warmup {
        bail!("corpus ran out during warm-up: {seen} blocks read, {} needed", options.warmup);
    }
    if recorded.is_some() {
        if checked == 0 {
            bail!(
                "the recorded run at {:?} shares no (block, arm) pair with this one, so nothing \
                 was checked",
                options.frontier.as_ref().expect("frontier present")
            );
        }
        tracing::info!(checked, "miss set agrees with the recorded run on every shared pair");
    }

    let census = census.finish();
    tracing::info!(
        distinct_slots = census.distinct_slots,
        distinct_storage_stems = census.distinct_storage_stems,
        slots_per_stem = census.slots_per_stem(),
        deployments_per_code = census.deployments_per_code(),
        extrapolated_stems =
            census.extrapolate(params.account_population, params.storage_population),
        "stem census over the corpus's own accesses"
    );

    fs::create_dir_all(&options.out)?;
    fs::write(
        options.out.join("census.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "census": census,
            "slots_per_stem": census.slots_per_stem(),
            "deployments_per_code": census.deployments_per_code(),
            "extrapolated_stem_population":
                census.extrapolate(params.account_population, params.storage_population),
            "note": "Accessed slots cluster more than the state as a whole, so the extrapolation \
                     is a lower bound on the stem count. The sweep over --stems is what the result \
                     is quoted against.",
        }))?,
    )?;

    let jsonl = options.out.join("blocks.jsonl");
    let mut sink = fs::File::create(&jsonl)?;
    for rows in results.values() {
        for row in rows {
            writeln!(sink, "{}", serde_json::to_string(row)?)?;
        }
    }

    let report = RunReport::build(
        options.dataset.display().to_string(),
        &params,
        &options.header_layout_name,
        &specs,
        &results,
        options.warmup,
        recorded.as_ref().map(|recorded| (recorded.digest(), checked)),
    );
    let summary = options.out.join("summary.json");
    fs::write(&summary, serde_json::to_string_pretty(&report)?)?;

    print_summary(&report);
    tracing::info!(
        elapsed_s = started.elapsed().as_secs(),
        blocks = report.measured_blocks,
        ?jsonl,
        ?summary,
        "run complete"
    );
    Ok(())
}

/// The recorded run this one must agree with about which keys each block missed.
///
/// The miss set is the study's invariant: the cache policy is defined over addresses and slots, so
/// it is the same number under every tree, and the whole comparison rests on it being the same
/// number the measured MPT arms were built from. Checking it inside the run rather than in a script
/// afterwards is what makes a mismatch stop the run instead of reaching a report.
struct RecordedRun {
    misses: BTreeMap<(u64, String), (usize, usize, usize)>,
    digest: String,
}

impl RecordedRun {
    fn load(path: &Path) -> Result<Self> {
        let raw = fs::read(path).with_context(|| format!("reading {path:?}"))?;
        let digest = format!("{:x}", alloy_primitives::keccak256(&raw));
        let mut misses = BTreeMap::new();
        for line in raw.split(|byte| *byte == b'\n') {
            if line.is_empty() {
                continue
            }
            let record: serde_json::Value = serde_json::from_slice(line)
                .with_context(|| format!("decoding a record of {path:?}"))?;
            if record.get("measured").and_then(serde_json::Value::as_bool) != Some(true) {
                continue
            }
            let Some(block) = record.get("block_number").and_then(serde_json::Value::as_u64) else {
                bail!("a record of {path:?} names no block number");
            };
            let Some(policies) = record.get("policies").and_then(serde_json::Value::as_array)
            else {
                continue
            };
            for policy in policies {
                let field = |name: &str| -> Result<usize> {
                    policy
                        .get(name)
                        .and_then(serde_json::Value::as_u64)
                        .map(|value| value as usize)
                        .ok_or_else(|| eyre::eyre!("block {block} record has no `{name}`"))
                };
                let Some(name) = policy.get("policy").and_then(serde_json::Value::as_str) else {
                    bail!("a policy of block {block} is unnamed");
                };
                misses.insert(
                    (block, name.to_string()),
                    (field("missed_accounts")?, field("missed_storage")?, field("missed_codes")?),
                );
            }
        }
        if misses.is_empty() {
            bail!("{path:?} holds no measured records");
        }
        Ok(Self { misses, digest })
    }

    fn len(&self) -> usize {
        self.misses.len()
    }

    fn digest(&self) -> String {
        format!("0x{}", self.digest)
    }

    /// Fails the run when this block's miss set differs from the recorded one.
    fn check(&self, row: &BlockResult) -> Result<bool> {
        let Some(recorded) = self.misses.get(&(row.block_number, row.arm.clone())) else {
            return Ok(false)
        };
        let recorded = *recorded;
        let ours = (row.missed_accounts, row.missed_storage, row.missed_codes);
        if ours != recorded {
            bail!(
                "block {} arm {}: miss set {:?} does not match the recorded {:?}; the arms are no \
                 longer pricing one workload",
                row.block_number,
                row.arm,
                ours,
                recorded
            );
        }
        Ok(true)
    }
}

fn print_summary(report: &RunReport) {
    println!();
    println!(
        "measured {} blocks {}..={} | stems {} | layout {} | code coverage {:.2} | occupancy {}/{}",
        report.measured_blocks,
        report.measured_range.0,
        report.measured_range.1,
        report.params.total_stem_population,
        report.params.header_layout,
        report.params.code_coverage,
        report.params.stem_occupancy_in_header,
        report.params.stem_occupancy_outside_header,
    );
    match &report.miss_set_check {
        Some(check) => println!(
            "miss set: checked {} pairs against {} — all agree",
            check.pairs_checked, check.recorded_digest
        ),
        None => println!("miss set: NOT checked against a recorded run"),
    }
    println!();
    println!(
        "{:<8} {:>9} {:>9} {:>12} {:>12} {:>12} {:>11}",
        "arm", "miss_acct", "miss_slot", "binary_MB", "verkle_MB", "mpt_nodes", "stems"
    );
    for arm in &report.arms {
        println!(
            "{:<8} {:>9.0} {:>9.0} {:>12.3} {:>12.3} {:>12.0} {:>11.0}",
            arm.arm,
            arm.mean_missed_accounts,
            arm.mean_missed_storage,
            arm.mean_binary_total_bytes / 1e6,
            arm.mean_verkle_total_bytes / 1e6,
            arm.mean_mpt_model_nodes,
            arm.mean_stems_opened,
        );
    }
    println!();
    println!("{:<8} {:>16} {:>16}", "arm", "binary_vs_weak", "verkle_vs_weak");
    for ratio in &report.ratios {
        println!(
            "{:<8} {:>15.2}x {:>15.2}x",
            ratio.arm, ratio.binary_vs_weak_median, ratio.verkle_vs_weak_median
        );
    }
    println!();
}

struct Options {
    dataset: PathBuf,
    frontier: Option<PathBuf>,
    out: PathBuf,
    limit: Option<usize>,
    warmup: usize,
    params: StudyParams,
    header_layout_name: String,
}

impl Options {
    fn parse() -> Result<Self> {
        let mut dataset: Option<PathBuf> = None;
        let mut frontier: Option<PathBuf> = None;
        let mut out: Option<PathBuf> = None;
        let mut limit = None;
        let mut warmup = 120usize;
        let mut params = StudyParams::default();
        let mut header_layout_name = "table".to_string();

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            let mut value = || -> Result<String> {
                args.next().ok_or_else(|| eyre::eyre!("{arg} needs a value"))
            };
            match arg.as_str() {
                "--dataset" => dataset = Some(PathBuf::from(value()?)),
                "--frontier" => frontier = Some(PathBuf::from(value()?)),
                "--out" => out = Some(PathBuf::from(value()?)),
                "--limit" => limit = Some(value()?.parse()?),
                "--warmup" => warmup = value()?.parse()?,
                "--stems" => params.total_stem_population = value()?.parse()?,
                "--code-stems" => params.code_stem_population = value()?.parse()?,
                "--accounts" => params.account_population = value()?.parse()?,
                "--slots" => params.storage_population = value()?.parse()?,
                "--code-coverage" => params.code_coverage = value()?.parse()?,
                "--stem-occupancy" => {
                    params.stem_occupancy.get_or_insert_with(StemOccupancy::zero).outside_header =
                        value()?.parse()?
                }
                "--header-stem-occupancy" => {
                    params.stem_occupancy.get_or_insert_with(StemOccupancy::zero).in_header =
                        value()?.parse()?
                }
                "--mpt-storage-trie" => params.mpt_storage_trie_population = value()?.parse()?,
                "--header-layout" => {
                    header_layout_name = value()?;
                    params.header_layout = HeaderLayout::by_name(&header_layout_name)?;
                }
                "--help" | "-h" => {
                    eprintln!("{USAGE}");
                    std::process::exit(0);
                }
                other => return Err(eyre::eyre!("unknown argument `{other}`\n{USAGE}")),
            }
        }

        let dataset = dataset.ok_or_else(|| eyre::eyre!("--dataset is required\n{USAGE}"))?;
        if !Path::new(&dataset).is_dir() {
            bail!("--dataset {dataset:?} is not a directory");
        }
        let out = out.ok_or_else(|| eyre::eyre!("--out is required\n{USAGE}"))?;
        Ok(Self { dataset, frontier, out, limit, warmup, params, header_layout_name })
    }
}

const USAGE: &str = "\
ps-tree-study --dataset <corpus dir> --out <results dir> [options]

  --frontier PATH          recorded frontier.jsonl; miss sets are checked against it pair by pair
  --limit N                stop after N corpus records
  --warmup N               blocks used only to fill the caches (default 120)
  --stems N                stems the whole tree is modelled to hold (default 2000000000)
  --code-stems N           stems the out-of-header code region holds (default 3000000)
  --accounts N             accounts in the modelled state (default 412044818, measured)
  --slots N                storage slots in the modelled state (default 1636513307, measured)
  --code-coverage F        fraction of a contract's chunks a call runs (default 1.0)
  --stem-occupancy N       extra occupied suffixes in a non-header stem; overrides the value
                           derived from --slots and --stems
  --header-stem-occupancy N  extra occupied suffixes assumed in a header stem (default 0)
  --mpt-storage-trie N     modelled per-account storage trie size for the MPT calibration
  --header-layout NAME     `table` or `prose` (EIP-7864's two disagreeing header layouts)
";
