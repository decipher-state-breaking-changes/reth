//! `ps-policy-frontier` — generate every cache policy's sidecars from one recorded corpus.
//!
//! ```text
//! ps-policy-frontier \
//!   --dataset /path/to/dataset \
//!   --arm weak --arm 60/30 --arm 90/60 --arm 120/45 \
//!   --warmup 120 --samples 1000 \
//!   --out /path/to/frontier-out
//! ```
//!
//! `--arm weak` is the no-cache baseline: a validator holding nothing when each block arrives. It
//! runs in the same rotation over the same blocks as the policies, because a baseline measured any
//! other way is a baseline nobody can check against the thing it is a baseline for.
//!
//! Warm-up blocks are replayed in full — a cache that skipped them would not be warm — and simply
//! do not count toward the reported population. The tool refuses a warm-up shorter than the widest
//! policy window, because a policy measured before its window is populated is not the policy the
//! report names.

// Global allocator, selected at compile time by the `jemalloc`/`snmalloc` features and left as the
// platform allocator when neither is named.
//
// This binary rotates every arm over the same block in one process, so the allocator is shared by
// arms that stress it very differently: a no-cache arm does no cache maintenance while a windowed
// arm does all of it. An allocator that degrades under churn therefore lands unevenly across the
// arms of the very comparison this tool exists to make. It belongs in `main.rs` because a
// `#[global_allocator]` in a library is only honoured when that library happens to be the root
// crate.
//
// jemalloc takes precedence when both features are on, matching `reth_cli_util::allocator`.
#[cfg(all(feature = "jemalloc", unix))]
#[global_allocator]
static GLOBAL_ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

// Required for jemalloc to override the allocator on supported Unix platforms: the symbols only
// reach the linker if something `use`s the sys crate.
#[cfg(all(feature = "jemalloc", unix))]
use tikv_jemalloc_sys as _;

#[cfg(all(feature = "snmalloc", not(feature = "jemalloc"), unix))]
#[global_allocator]
static GLOBAL_ALLOCATOR: snmalloc_rs::SnMalloc = snmalloc_rs::SnMalloc;

// Keeps `--all-features` from warning about the crate the precedence rule above shuts out.
#[cfg(all(feature = "snmalloc", feature = "jemalloc", unix))]
use snmalloc_rs as _;

/// Name of the allocator this binary was built with, for the run summary.
///
/// Compile-time rather than probed: the point is to record what the build selected, and a probe
/// would report what the process happens to have loaded.
pub const ALLOCATOR_NAME: &str = if cfg!(all(feature = "jemalloc", unix)) {
    "jemalloc"
} else if cfg!(all(feature = "snmalloc", unix)) {
    "snmalloc"
} else {
    "system"
};

use partial_stateless::{load_dataset, CacheTrieRepr};
use partial_stateless_frontier::{
    generate::{generate_block, ChainCursor, GeneratorRules},
    policy::{ArmKind, PolicyState},
    report::{RunReport, RunSummary},
};
use partial_stateless_validator::{SidecarReexecLimits, UntrustedAdmission, ValidatorRules};
use reth_chainspec::{ChainSpec, MAINNET};
use reth_ethereum_consensus::EthBeaconConsensus;
use reth_evm_ethereum::EthEvmConfig;
use std::{path::PathBuf, sync::Arc};
use tracing::info;

/// Everything the command line decided.
#[derive(Debug)]
struct Options {
    dataset: PathBuf,
    arms: Vec<ArmKind>,
    warmup: u64,
    samples: u64,
    out: PathBuf,
    trie_diagnostics: bool,
    witness_v3: bool,
    trie_repr: CacheTrieRepr,
    compress_sidecars: bool,
}

fn usage() -> String {
    "usage: ps-policy-frontier --dataset <dir> --arm <weak|a/s> [--arm <weak|a/s> ...] \
     --warmup <n> --samples <n> --out <dir> [--trie-diagnostics] [--witness-v3] \
     [--trie-repr <parallel|exact>] [--compress-sidecars]"
        .to_string()
}

fn parse_args() -> eyre::Result<Options> {
    let mut dataset = None;
    let mut arms = Vec::new();
    let mut warmup = None;
    let mut samples = None;
    let mut out = None;
    let mut trie_diagnostics = false;
    let mut witness_v3 = false;
    let mut trie_repr = CacheTrieRepr::default();
    let mut compress_sidecars = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value =
            || args.next().ok_or_else(|| eyre::eyre!("{arg} needs a value\n{}", usage()));
        match arg.as_str() {
            "--dataset" => dataset = Some(PathBuf::from(value()?)),
            // `--policy` kept as an alias: it is what the first runs were launched with, and a
            // rename that silently rejected an operator's saved command line would cost more than
            // it saves.
            "--arm" | "--policy" => arms.push(value()?.parse::<ArmKind>()?),
            "--warmup" => warmup = Some(value()?.parse::<u64>()?),
            "--samples" => samples = Some(value()?.parse::<u64>()?),
            "--out" => out = Some(PathBuf::from(value()?)),
            "--trie-diagnostics" => trie_diagnostics = true,
            "--witness-v3" => witness_v3 = true,
            "--compress-sidecars" => compress_sidecars = true,
            "--trie-repr" => {
                trie_repr = value()?.parse().map_err(|err| eyre::eyre!("{err}"))?;
            }
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            other => eyre::bail!("unknown argument `{other}`\n{}", usage()),
        }
    }

    let dataset = dataset.ok_or_else(|| eyre::eyre!("--dataset is required\n{}", usage()))?;
    let out = out.ok_or_else(|| eyre::eyre!("--out is required\n{}", usage()))?;
    let warmup = warmup.ok_or_else(|| eyre::eyre!("--warmup is required\n{}", usage()))?;
    let samples = samples.ok_or_else(|| eyre::eyre!("--samples is required\n{}", usage()))?;
    if arms.is_empty() {
        eyre::bail!("at least one --arm is required\n{}", usage())
    }
    arms.sort_unstable();
    let deduped = {
        let mut seen = arms.clone();
        seen.dedup();
        seen
    };
    if deduped.len() != arms.len() {
        eyre::bail!("the same arm was given twice; each --arm must be distinct")
    }
    if samples == 0 {
        eyre::bail!("--samples must be at least 1")
    }

    Ok(Options {
        dataset,
        arms,
        warmup,
        samples,
        out,
        trie_diagnostics,
        witness_v3,
        trie_repr,
        compress_sidecars,
    })
}

fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let options = parse_args()?;

    // A warm-up shorter than the widest window leaves that policy holding part of the window its
    // identifier advertises, and every number it then produces belongs to a policy nobody named.
    let widest = options.arms.iter().map(ArmKind::warmup_floor).max().unwrap_or(0);
    if options.warmup < widest {
        eyre::bail!(
            "--warmup {} is shorter than the widest policy window ({widest}); a policy measured \
             before its window is populated is not the policy the report names",
            options.warmup
        )
    }

    let dataset = load_dataset(&options.dataset)?;
    let needed = options.warmup + options.samples;
    if (dataset.records.len() as u64) < needed {
        eyre::bail!(
            "dataset holds {} canonical blocks but {needed} are needed ({} warm-up + {} measured)",
            dataset.records.len(),
            options.warmup,
            options.samples
        )
    }
    info!(
        dataset = %options.dataset.display(),
        producer = %dataset.manifest.producer,
        canonical_blocks = dataset.records.len(),
        abandoned_blocks = dataset.abandoned.len(),
        unconfirmed_blocks = dataset.unconfirmed.len(),
        arms = options.arms.len(),
        weak_baseline = options.arms.contains(&ArmKind::Weak),
        warmup = options.warmup,
        samples = options.samples,
        witness_v3 = options.witness_v3,
        trie_repr = options.trie_repr.label(),
        allocator = ALLOCATOR_NAME,
        compress_sidecars = options.compress_sidecars,
        "Loaded policy replay dataset"
    );
    if !dataset.abandoned.is_empty() {
        info!(
            abandoned = dataset.abandoned.len(),
            "Records on branches the chain left are present in the dataset and excluded from the \
             replay"
        );
    }
    // Different exclusion, different meaning: nothing is wrong with these blocks, the capture just
    // stopped before it could say they had settled. Worth saying out loud, because a corpus that is
    // shorter than the operator expected usually is this and not a missing file.
    if !dataset.unconfirmed.is_empty() {
        info!(
            unconfirmed = dataset.unconfirmed.len(),
            usable_range = ?dataset.end.usable_range,
            confirmations = dataset.end.confirmations,
            "The capture wrote blocks past the range it vouched for; they are excluded"
        );
    }

    let chain_spec: Arc<ChainSpec> = chain_spec_for(&dataset.manifest.chain)?;
    let consensus = EthBeaconConsensus::new(chain_spec.clone());
    let evm_config = EthEvmConfig::new(chain_spec.clone());
    let admission = UntrustedAdmission::new(chain_spec.as_ref(), &consensus);
    let limits = SidecarReexecLimits::default();
    let rules = GeneratorRules {
        validator: ValidatorRules::new(&evm_config, &consensus),
        admission: &admission,
        limits: &limits,
        trie_diagnostics: options.trie_diagnostics,
        witness_v3: options.witness_v3,
        compress_sidecars: options.compress_sidecars,
    };

    let replayed = &dataset.records[..needed as usize];
    let first = &replayed[0];
    let mut cursor = ChainCursor::enter(first)?;
    let mut policies = options
        .arms
        .iter()
        .map(|arm| {
            PolicyState::cold_at_with_repr(
                *arm,
                first.body.block_number.saturating_sub(1),
                options.trie_repr,
            )
        })
        .collect::<Vec<_>>();

    std::fs::create_dir_all(&options.out)?;
    let mut report = RunReport::create(&options.out.join("frontier.jsonl"))?;
    let mut results = Vec::with_capacity(replayed.len());

    for (index, record) in replayed.iter().enumerate() {
        let measured = index as u64 >= options.warmup;
        let result = generate_block(&rules, record, &mut policies, &mut cursor, index, measured)?;
        report.append(&result)?;
        if measured && result.block_number % 100 == 0 {
            info!(
                block = result.block_number,
                measured_so_far = index as u64 - options.warmup + 1,
                "Generated every policy's sidecar"
            );
        }
        results.push(result);
    }

    let stream = report.finish()?;
    let summary = RunSummary::accumulate(
        &options.dataset,
        dataset.manifest.producer.clone(),
        dataset.manifest.build_commit.clone(),
        &options.arms,
        &results,
        options.witness_v3,
        options.trie_repr,
        ALLOCATOR_NAME,
        options.compress_sidecars,
    );
    let summary_path = options.out.join("frontier-summary.json");
    summary.write(&summary_path)?;

    for (arm, totals) in &summary.policies {
        // The self-description gate: a v3 run whose measured policy blocks were not all v3 is
        // reporting mixed wire formats, and its byte totals must not be quoted as v3 figures.
        if options.witness_v3 && arm != "weak" && totals.witness_v3_blocks != totals.blocks {
            tracing::warn!(
                arm = %arm,
                blocks = totals.blocks,
                witness_v3_blocks = totals.witness_v3_blocks,
                "witness-v3 run produced non-v3 measured sidecars; byte totals are mixed-format"
            );
        }
        info!(
            arm = %arm,
            blocks = totals.blocks,
            witness_v3_blocks = totals.witness_v3_blocks,
            mean_sidecar_bytes = totals.mean_sidecar_bytes(),
            total_missed_accounts = totals.total_missed_accounts,
            total_missed_storage = totals.total_missed_storage,
            final_cache_bytes = totals.final_cache_bytes,
            final_trie_cache_bytes = totals.final_trie_cache_bytes,
            "Arm totals"
        );
        if totals.trim_measured_blocks > 0 {
            let ratio =
                totals.trim_trimmable_bytes as f64 / totals.trim_witness_node_bytes.max(1) as f64;
            info!(
                arm = %arm,
                trim_measured_blocks = totals.trim_measured_blocks,
                trimmable_share = format!("{:.4}", ratio),
                trimmable_bytes = totals.trim_trimmable_bytes,
                witness_node_bytes = totals.trim_witness_node_bytes,
                account_bytes = totals.trim_trimmable_account_bytes,
                storage_bytes = totals.trim_trimmable_storage_bytes,
                unattributed_nodes = totals.trim_unattributed_nodes,
                "Witness-trim potential"
            );
        }
    }
    info!(
        stream = %stream.display(),
        summary = %summary_path.display(),
        measured_blocks = summary.measured_blocks,
        block_set_digest = ?summary.measured_block_set_digest,
        "Frontier run complete"
    );

    Ok(())
}

/// The chain spec a dataset's blocks belong to.
///
/// Refused rather than defaulted: a corpus replayed under the wrong fork schedule would produce
/// consensus rejections that look like validator defects.
fn chain_spec_for(chain: &str) -> eyre::Result<Arc<ChainSpec>> {
    if chain.eq_ignore_ascii_case("mainnet") || chain == MAINNET.chain.to_string() {
        return Ok(MAINNET.clone())
    }
    eyre::bail!("this build replays mainnet corpora only, and the dataset names `{chain}`")
}
