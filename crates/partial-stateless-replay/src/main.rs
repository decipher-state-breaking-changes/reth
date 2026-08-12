//! `ps-replay` — replays a recorded partial-stateless stream with no state database.
//!
//! ```text
//! ps-replay <spool-dir> [--limit N] [--no-mutations]
//! ```
//!
//! Exits non-zero when the replay disagreed with the recording anywhere, because the whole point
//! of running it is that a disagreement is a result rather than a diagnostic.

use partial_stateless_replay::{replay, ReplayOptions};
use std::path::PathBuf;
use tracing::{error, info, warn};

fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let (dir, options) = parse_args()?;
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
        "Replay finished"
    );

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

fn parse_args() -> eyre::Result<(PathBuf, ReplayOptions)> {
    let mut args = std::env::args().skip(1);
    let mut dir = None;
    let mut options = ReplayOptions::default();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--limit" => {
                let raw = args.next().ok_or_else(|| eyre::eyre!("--limit needs a block count"))?;
                options.limit = Some(raw.parse()?);
            }
            "--no-mutations" => options.mutations = false,
            "-h" | "--help" => {
                println!("ps-replay <spool-dir> [--limit N] [--no-mutations]");
                std::process::exit(0);
            }
            other if dir.is_none() => dir = Some(PathBuf::from(other)),
            other => return Err(eyre::eyre!("unexpected argument {other}")),
        }
    }
    let dir = dir.ok_or_else(|| eyre::eyre!("usage: ps-replay <spool-dir> [--limit N]"))?;
    Ok((dir, options))
}
