//! `ps-replay` — replays a recorded partial-stateless stream with no state database.
//!
//! ```text
//! ps-replay <spool-dir> [--limit N] [--no-mutations] [--json <path>] [--label <name>]
//! ```
//!
//! Exits non-zero when the replay disagreed with the recording anywhere, because the whole point
//! of running it is that a disagreement is a result rather than a diagnostic.

use partial_stateless_replay::{replay, ReplayOptions, ReplayReport};
use std::path::PathBuf;
use tracing::{error, info, warn};

fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let Args { dir, options, json, label } = parse_args()?;
    let started = std::time::Instant::now();
    let report = replay(&dir, &options)?;
    let elapsed_ms = started.elapsed().as_millis() as u64;

    info!(
        target: "ps_replay",
        commits = report.commits,
        witnessed = report.witnessed,
        reconstructed = report.reconstructed,
        absent = report.absent,
        mutations_checked = report.mutations_checked,
        admission_us = report.admission_us,
        transition_us = report.transition_us,
        elapsed_ms,
        closed = report.closed,
        skipped_after_fault = report.skipped_after_fault,
        "Replay finished"
    );

    if let Some(terminal) = &report.terminal {
        error!(
            target: "ps_replay",
            %terminal,
            skipped = report.skipped_after_fault,
            "The replay stopped on a fault; every commit after it was skipped, not replayed"
        );
    }

    if !report.admission_is_load_bearing() {
        warn!(
            target: "ps_replay",
            "No commit in this corpus carried a witnessed payload, so every admission check passed \
             against a payload derived from a block this node had already accepted. The code ran; \
             the rules were not tested"
        );
    }

    for (block, disagreement) in &report.disagreements {
        error!(target: "ps_replay", block = block.number, %disagreement, "Disagreement");
    }
    for failure in &report.failures {
        error!(target: "ps_replay", %failure, "Replay failure");
    }
    for failure in &report.mutation_failures {
        error!(target: "ps_replay", %failure, "Mutation coverage failure");
    }

    if let Some(path) = json {
        write_record(&path, &label, &report, elapsed_ms)?;
    }

    if report.agreed() {
        info!(
            target: "ps_replay",
            "The standalone validator agreed with the recording on every field it compared"
        );
        return Ok(())
    }
    Err(eyre::eyre!(
        "{} disagreements, {} failures, {} mutation failures",
        report.disagreements.len(),
        report.failures.len(),
        report.mutation_failures.len()
    ))
}

/// One JSON line per run, so alternating A/B rounds append to one file.
///
/// Per block rather than only in total: the same block replayed by two builds is the one
/// comparison with no workload variance in it, and a summary would throw that pairing away.
fn write_record(
    path: &std::path::Path,
    label: &str,
    report: &ReplayReport,
    elapsed_ms: u64,
) -> eyre::Result<()> {
    use std::io::Write;
    let record = serde_json::json!({
        "schema_version": 1,
        "benchmark": "standalone_replay_v1",
        "label": label,
        "commits": report.commits,
        "witnessed": report.witnessed,
        "reconstructed": report.reconstructed,
        "absent": report.absent,
        "agreed": report.agreed(),
        "admission_is_load_bearing": report.admission_is_load_bearing(),
        "disagreements": report.disagreements.len(),
        "failures": report.failures.len(),
        "mutations_checked": report.mutations_checked,
        "mutation_failures": report.mutation_failures.len(),
        "admission_us": report.admission_us,
        "transition_us": report.transition_us,
        "elapsed_ms": elapsed_ms,
        "terminal": report.terminal,
        "skipped_after_fault": report.skipped_after_fault,
        "blocks": report.blocks,
    });
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{record}")?;
    Ok(())
}

struct Args {
    dir: PathBuf,
    options: ReplayOptions,
    json: Option<PathBuf>,
    label: String,
}

fn parse_args() -> eyre::Result<Args> {
    let mut args = std::env::args().skip(1);
    let mut dir = None;
    let mut options = ReplayOptions::default();
    let mut json = None;
    let mut label = "unlabelled".to_string();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--limit" => {
                let raw = args.next().ok_or_else(|| eyre::eyre!("--limit needs a block count"))?;
                options.limit = Some(raw.parse()?);
            }
            "--no-mutations" => options.mutations = false,
            "--json" => {
                json = Some(PathBuf::from(
                    args.next().ok_or_else(|| eyre::eyre!("--json needs a path"))?,
                ));
            }
            "--label" => {
                label = args.next().ok_or_else(|| eyre::eyre!("--label needs a name"))?;
            }
            "-h" | "--help" => {
                println!(
                    "ps-replay <spool-dir> [--limit N] [--no-mutations] [--json <path>] \
                     [--label <name>]"
                );
                std::process::exit(0);
            }
            other if dir.is_none() => dir = Some(PathBuf::from(other)),
            other => return Err(eyre::eyre!("unexpected argument {other}")),
        }
    }
    let dir = dir.ok_or_else(|| eyre::eyre!("usage: ps-replay <spool-dir> [--limit N]"))?;
    Ok(Args { dir, options, json, label })
}
