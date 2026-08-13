//! `ps-replay` — replays a recorded partial-stateless stream with no state database.
//!
//! ```text
//! ps-replay <spool-dir> [--limit N] [--no-mutations] [--json <path>] [--label <name>]
//! ps-replay --follow <spool-dir> [--poll-ms N] [--max-blocks N] [--idle-timeout-secs N]
//!           [--ack <path>] [--resume] [--mutations] [--json <path>] [--label <name>]
//! ```
//!
//! Batch mode exits non-zero when the replay disagreed with the recording anywhere, because the
//! whole point of running it is that a disagreement is a result rather than a diagnostic.
//!
//! Follow mode consumes the spool while a producer is still writing it — the S3 live consumer —
//! and its exit codes are states, not just errors: `0` a cleanly ended, fully agreeing stream;
//! `1` a disagreement, fault, or fault-kind end; `2` the run timed out waiting in
//! `NeedsSnapshot` (recovery never came); `3` the stream ended before any checkpoint; `4` the
//! run timed out waiting for frames. Without `--idle-timeout-secs` the follower waits forever,
//! because a quiet spool and a killed producer are indistinguishable from files alone.

use partial_stateless_replay::{
    follow, replay, FollowOptions, FollowOutcome, FollowReport, ReplayOptions, ReplayReport,
};
use partial_stateless_stream::EndKind;
use std::path::PathBuf;
use tracing::{error, info, warn};

fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let mode = parse_args()?;
    let Mode::Batch(Args { dir, options, json, label }) = mode else {
        let Mode::Follow { dir, options } = mode else { unreachable!() };
        return run_follow(&dir, &options)
    };
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

    if report.agreed() && report.continuous() && report.complete() {
        info!(
            target: "ps_replay",
            "The standalone validator agreed with the recording on every field it compared, and \
             verified every canonical block the corpus carried"
        );
        return Ok(())
    }
    // Reported apart, because they fail for different reasons and a run that agreed everywhere it
    // looked but has a hole in what it looked at is a different result from a run that disagreed.
    Err(eyre::eyre!(
        "{} disagreements, {} failures, {} mutation failures; continuous={}, complete={}{}",
        report.disagreements.len(),
        report.failures.len(),
        report.mutation_failures.len(),
        report.continuous(),
        report.complete(),
        report.terminal.as_ref().map(|why| format!(" ({why})")).unwrap_or_default()
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
        "closed": report.closed,
        "terminal": report.terminal,
        "terminal_kind": report.terminal_kind,
        "skipped_after_fault": report.skipped_after_fault,
        // Three axes rather than one. `agreed` is about the two implementations, `continuous`
        // about whether any canonical block went unverified, and `complete` about whether the
        // replay reached the end of the corpus. A reorg the pair could not undo moves the last
        // two and must not move the first.
        "continuous": report.continuous(),
        "complete": report.complete(),
        "reorgs_applied": report.reorgs_applied,
        "reverts_applied": report.reverts_applied,
        "reorgs_inapplicable": report.reorgs_inapplicable,
        "epochs": report.epochs,
        "checkpoints_skimmed": report.checkpoints_skimmed,
        "skipped_awaiting_resync": report.skipped_awaiting_resync,
        "winning_branch_incomplete": report.winning_branch_incomplete,
        "winning_branches_superseded": report.winning_branches_superseded,
        "resyncs": report.resyncs,
        "last_verified": report.last_verified,
        "blocks": report.blocks,
    });
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{record}")?;
    Ok(())
}

/// Runs the live follower and maps its outcome onto the documented exit codes.
fn run_follow(dir: &std::path::Path, options: &FollowOptions) -> eyre::Result<()> {
    let started = std::time::Instant::now();
    let report = follow(dir, options)?;
    let elapsed_ms = started.elapsed().as_millis() as u64;
    report_follow(&report);
    if let Some(path) = &options.verdicts {
        write_follow_summary(path, &options.label, &report, elapsed_ms)?;
    }
    let code = follow_exit_code(&report);
    if code == 0 {
        return Ok(())
    }
    std::process::exit(code);
}

fn report_follow(report: &FollowReport) {
    info!(
        target: "ps_follow",
        outcome = ?report.outcome,
        blocks_verified = report.blocks_verified,
        catch_up_blocks = report.catch_up_blocks,
        resumed_from = ?report.resumed_from,
        restores = report.restores,
        needs_snapshot_entries = report.needs_snapshot_entries,
        commits_skipped_in_recovery = report.commits_skipped_in_recovery,
        witnessed = report.replay.witnessed,
        reconstructed = report.replay.reconstructed,
        absent = report.replay.absent,
        disagreements = report.replay.disagreements.len(),
        failures = report.replay.failures.len(),
        agreed = report.agreed(),
        continuous = report.continuous(),
        "Follow finished"
    );
    for (block, disagreement) in &report.replay.disagreements {
        error!(target: "ps_follow", block = block.number, %disagreement, "Disagreement");
    }
    for failure in &report.replay.failures {
        error!(target: "ps_follow", %failure, "Failure");
    }
}

/// One `kind: "summary"` line closing the verdict stream, so a harness judges the run on the
/// follower's own aggregate counters rather than inferring them from an exit code — an exit code
/// alone cannot distinguish "verified 100 witnessed blocks" from "verified nothing at all".
fn write_follow_summary(
    path: &std::path::Path,
    label: &str,
    report: &FollowReport,
    elapsed_ms: u64,
) -> eyre::Result<()> {
    use std::io::Write;
    let (outcome, end_kind, before_checkpoint, waiting_in) = match &report.outcome {
        FollowOutcome::Ended { kind, before_checkpoint } => {
            ("ended", Some(kind.as_str()), Some(*before_checkpoint), None)
        }
        FollowOutcome::Faulted { .. } => ("faulted", None, None, None),
        FollowOutcome::MaxBlocks => ("max_blocks", None, None, None),
        FollowOutcome::IdleTimeout { waiting_in } => {
            ("idle_timeout", None, None, Some(*waiting_in))
        }
    };
    let record = serde_json::json!({
        "schema_version": 1,
        "benchmark": "standalone_follow_v1",
        "kind": "summary",
        "label": label,
        "outcome": outcome,
        "end_kind": end_kind,
        "before_checkpoint": before_checkpoint,
        "waiting_in": waiting_in,
        "blocks_verified": report.blocks_verified,
        "commits": report.replay.commits,
        "witnessed": report.replay.witnessed,
        "reconstructed": report.replay.reconstructed,
        "absent": report.replay.absent,
        "disagreements": report.replay.disagreements.len(),
        "failures": report.replay.failures.len(),
        "mutations_checked": report.replay.mutations_checked,
        "mutation_failures": report.replay.mutation_failures.len(),
        "restores": report.restores,
        "needs_snapshot_entries": report.needs_snapshot_entries,
        "commits_skipped_in_recovery": report.commits_skipped_in_recovery,
        "last_needs_snapshot": report.last_needs_snapshot.map(|reason| reason.as_str()),
        "last_verified": report.last_verified.map(|block| block.number),
        "agreed": report.agreed(),
        // The second axis: whether any canonical block went unverified. A follower that undid a
        // reorg itself stays continuous; one that recovered from a checkpoint elsewhere does not,
        // however cleanly it agreed on the blocks it did see.
        "continuous": report.continuous(),
        "reorgs_applied": report.reorgs_applied,
        "reverts_applied": report.reverts_applied,
        "checkpoints_skimmed": report.checkpoints_skimmed,
        "restores_continuous": report.restores_continuous,
        "restores_reset": report.restores_reset,
        "winning_branches_completed": report.winning_branches_completed,
        "winning_branches_incomplete": report.winning_branches_incomplete,
        "winning_branches_superseded": report.winning_branches_superseded,
        "scan_refusals": report.scan_refusals,
        "catch_up_blocks": report.catch_up_blocks,
        "resumed_from": report.resumed_from,
        "unverified_intervals": report.unverified_intervals,
        "admission_is_load_bearing": report.replay.admission_is_load_bearing(),
        "elapsed_ms": elapsed_ms,
    });
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{record}")?;
    Ok(())
}

/// Both axes, because a follower that agreed everywhere it looked can still have looked at less
/// than the chain: a recovery that landed somewhere other than the block it asked for leaves an
/// interval nothing validated, and exiting 0 on that would report a hole as a clean run.
fn follow_succeeded(report: &FollowReport) -> bool {
    report.agreed() && report.continuous()
}

fn follow_exit_code(report: &FollowReport) -> i32 {
    match &report.outcome {
        FollowOutcome::Ended { before_checkpoint: true, .. } => 3,
        FollowOutcome::Ended { kind: EndKind::Shutdown | EndKind::SpoolLimit, .. } => {
            i32::from(!follow_succeeded(report))
        }
        FollowOutcome::Ended { .. } | FollowOutcome::Faulted { .. } => 1,
        FollowOutcome::MaxBlocks => i32::from(!follow_succeeded(report)),
        FollowOutcome::IdleTimeout { waiting_in } if *waiting_in == "needs_snapshot" => 2,
        FollowOutcome::IdleTimeout { .. } => 4,
    }
}

struct Args {
    dir: PathBuf,
    options: ReplayOptions,
    json: Option<PathBuf>,
    label: String,
}

enum Mode {
    Batch(Args),
    Follow { dir: PathBuf, options: FollowOptions },
}

fn parse_args() -> eyre::Result<Mode> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.iter().any(|arg| arg == "--follow") {
        return parse_follow_args(raw)
    }
    let mut args = raw.into_iter();
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
            "--force-restore-at" => {
                let raw = args
                    .next()
                    .ok_or_else(|| eyre::eyre!("--force-restore-at needs a frame sequence"))?;
                options.force_restore_at = Some(raw.parse()?);
            }
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
                    "ps-replay <spool-dir> [--limit N] [--no-mutations] \
                     [--force-restore-at <sequence>] [--json <path>] \
                     [--label <name>]\nps-replay --follow <spool-dir> [--poll-ms N] \
                     [--max-blocks N] [--idle-timeout-secs N] [--ack <path>] [--resume] \
                     [--mutations] [--json <path>] [--label <name>]"
                );
                std::process::exit(0);
            }
            other if dir.is_none() => dir = Some(PathBuf::from(other)),
            other => return Err(eyre::eyre!("unexpected argument {other}")),
        }
    }
    let dir = dir.ok_or_else(|| eyre::eyre!("usage: ps-replay <spool-dir> [--limit N]"))?;
    Ok(Mode::Batch(Args { dir, options, json, label }))
}

fn parse_follow_args(raw: Vec<String>) -> eyre::Result<Mode> {
    let mut args = raw.into_iter();
    let mut dir = None;
    let mut options = FollowOptions::default();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--follow" => {}
            "--poll-ms" => {
                let raw = args.next().ok_or_else(|| eyre::eyre!("--poll-ms needs a value"))?;
                options.poll = std::time::Duration::from_millis(raw.parse()?);
            }
            "--max-blocks" => {
                let raw = args.next().ok_or_else(|| eyre::eyre!("--max-blocks needs a count"))?;
                options.max_blocks = Some(raw.parse()?);
            }
            "--idle-timeout-secs" => {
                let raw =
                    args.next().ok_or_else(|| eyre::eyre!("--idle-timeout-secs needs a value"))?;
                options.idle_timeout = Some(std::time::Duration::from_secs(raw.parse()?));
            }
            "--ack" => {
                options.ack = Some(PathBuf::from(
                    args.next().ok_or_else(|| eyre::eyre!("--ack needs a path"))?,
                ));
            }
            "--resume" => options.resume = true,
            "--mutations" => options.mutations = true,
            "--json" => {
                options.verdicts = Some(PathBuf::from(
                    args.next().ok_or_else(|| eyre::eyre!("--json needs a path"))?,
                ));
            }
            "--label" => {
                options.label = args.next().ok_or_else(|| eyre::eyre!("--label needs a name"))?;
            }
            other if dir.is_none() => dir = Some(PathBuf::from(other)),
            other => return Err(eyre::eyre!("unexpected argument {other}")),
        }
    }
    let dir = dir.ok_or_else(|| eyre::eyre!("usage: ps-replay --follow <spool-dir>"))?;
    Ok(Mode::Follow { dir, options })
}
