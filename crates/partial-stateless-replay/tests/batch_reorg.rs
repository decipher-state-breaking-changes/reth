//! What a batch replay does with the lifecycle events the corpus carries.
//!
//! Before reorg recovery landed, the driver replayed past a reorg it had not applied and reported
//! every block after it as a refusal — one real mainnet reorg turned 78 winning-branch commits into
//! 78 failures and 78 disagreements. The events are now part of the grammar: a reorg the pair can
//! undo is undone, one it cannot is a single typed stop, and only a checkpoint restarts
//! verification.
//!
//! Empty post-merge blocks exercise successful recovery without a database; the recorded
//! mainnet corpus remains the gate on real workloads.

mod common;

use alloy_primitives::B256;
use common::{
    commit_frame, end_frame, fixture, fixture_at, manifest, spool_dir, write_checkpoint,
    write_frame, ANCHOR_BLOCK,
};
use partial_stateless_replay::{replay, ForcedReorg, ReplayOptions};
use partial_stateless_stream::{
    BlockRef, EndKind, FrameKind, Reorg, Reset, ResetReason, StreamEvent,
};
use partial_stateless_validator::RetentionDepth;
use std::path::Path;

/// Mutations off: these tests are about the frames between commits, and a synthetic payload has
/// nothing for the mutation layer to derive from.
fn options() -> ReplayOptions {
    ReplayOptions { mutations: false, ..Default::default() }
}

/// The block the fixture's checkpoint restores to, as the frames name it.
fn anchor(fixture: &common::Fixture) -> BlockRef {
    fixture.checkpoint.block
}

/// A reorg abandoning one block above `ancestor` that this consumer never verified.
fn reorg_above(ancestor: BlockRef) -> StreamEvent {
    StreamEvent::Reorg(Reorg {
        common_ancestor: ancestor,
        abandoned: vec![BlockRef { number: ancestor.number + 1, hash: B256::repeat_byte(0xa1) }],
        winning_tip: Some(BlockRef { number: ancestor.number + 1, hash: B256::repeat_byte(0xb2) }),
    })
}

fn write_manifest(dir: &Path) {
    write_frame(dir, 0, FrameKind::Manifest, &StreamEvent::Manifest(manifest()));
}

const EVIDENCE_COMMIT: &str = "1111111111111111111111111111111111111111";

fn write_jsonl(path: &Path, rows: &[serde_json::Value]) {
    std::fs::write(path, rows.iter().map(|row| format!("{row}\n")).collect::<String>()).unwrap();
}

fn evidence_with_test_provenance(path: &Path) -> Vec<serde_json::Value> {
    let mut rows: Vec<serde_json::Value> = std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    // Execution evidence comes from the real CLI. Build provenance is a separate test fixture
    // so this test also works when the test binary was compiled from a developer's dirty tree.
    let manifest = rows.iter_mut().find(|row| row["kind"] == "run_manifest").unwrap();
    manifest["provenance"]["build_commit"] = EVIDENCE_COMMIT.into();
    manifest["provenance"]["build_dirty"] = false.into();
    rows
}

fn check_evidence(
    path: &Path,
    rows: &[serde_json::Value],
    args: &[String],
) -> std::process::Output {
    write_jsonl(path, rows);
    let manifest = rows.iter().find(|row| row["kind"] == "run_manifest").unwrap();
    std::process::Command::new("python3")
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/check_forced_reorg_smoke.py"))
        .arg(path)
        .args([
            "--expected-commit",
            EVIDENCE_COMMIT,
            "--allocator",
            manifest["allocator"].as_str().unwrap(),
            "--layout",
            manifest["undo_layout"].as_str().unwrap(),
        ])
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn an_inapplicable_reorg_is_a_lifecycle_event_not_a_cascade() {
    let dir = spool_dir("batch-inapplicable");
    let fixture = fixture();
    write_manifest(&dir);
    let mut next = write_checkpoint(&dir, 1, &fixture);
    write_frame(&dir, next, FrameKind::Reorg, &reorg_above(anchor(&fixture)));
    next += 1;
    for number in 0..3 {
        write_frame(
            &dir,
            next,
            FrameKind::Commit,
            &commit_frame(ANCHOR_BLOCK + 1 + number, B256::repeat_byte(0xb2)),
        );
        next += 1;
    }
    write_frame(&dir, next, FrameKind::End, &end_frame(next, EndKind::Shutdown));

    let report = replay(&dir, &options()).expect("the corpus reads");

    assert_eq!(report.reorgs_inapplicable, 1);
    assert!(
        report.failures.is_empty(),
        "a reorg past what K = 1 reaches is the chain behaving normally, not a replay failure: {:?}",
        report.failures
    );
    assert!(report.disagreements.is_empty());
    assert!(report.agreed(), "nothing this replay compared disagreed with the recording");
    assert_eq!(
        report.skipped_awaiting_resync, 3,
        "the winning branch is counted, not replayed against a pair that never unwound"
    );
    assert_eq!(report.terminal_kind, Some("awaiting_resync"));
    assert!(!report.continuous(), "three canonical blocks went unverified and the run says so");
    assert!(!report.complete());
    assert!(report.closed, "and the corpus itself was whole");
}

#[test]
fn a_forced_refusal_replays_from_the_first_abandoned_commit() {
    const DEPTH: u64 = 4;
    const SKIPPED: u64 = 2;
    const AFTER: u64 = 3;
    let dir = spool_dir("batch-forced-refusal-window");
    let (fixture, commits) = common::empty_chain::chain(DEPTH + SKIPPED + AFTER);
    write_manifest(&dir);
    let mut next = write_checkpoint(&dir, 1, &fixture);
    // Complete a recorded reorg's winning tip before the forced refusal.
    let (_, abandoned) = common::empty_chain::chain_with_tag(1, 1);
    write_frame(&dir, next, FrameKind::Commit, &abandoned[0]);
    next += 1;
    let block_of = |event: &StreamEvent| match event {
        StreamEvent::Commit(commit) => commit.input().block,
        _ => unreachable!(),
    };
    write_frame(
        &dir,
        next,
        FrameKind::Reorg,
        &StreamEvent::Reorg(Reorg {
            common_ancestor: fixture.checkpoint.block,
            abandoned: vec![block_of(&abandoned[0])],
            winning_tip: Some(block_of(&commits[0])),
        }),
    );
    next += 1;
    let first_sequence = next;
    for commit in commits.iter().take((DEPTH + SKIPPED) as usize) {
        write_frame(&dir, next, FrameKind::Commit, commit);
        next += 1;
    }
    let recovery_sequence = next;
    next = write_checkpoint(&dir, next, &fixture);
    let resume_at = next;
    for commit in commits.iter().skip((DEPTH + SKIPPED) as usize) {
        write_frame(&dir, next, FrameKind::Commit, commit);
        next += 1;
    }
    write_frame(&dir, next, FrameKind::End, &end_frame(next, EndKind::Shutdown));
    let report = replay(
        &dir,
        &ReplayOptions {
            retain_depth: RetentionDepth::new(DEPTH - 1).unwrap(),
            undo_record: true,
            forced_reorgs: vec![ForcedReorg {
                depth: DEPTH,
                at: fixture.checkpoint.block.number + DEPTH,
            }],
            ..options()
        },
    )
    .expect("the synthetic corpus reads");
    assert_eq!(report.forced_reorgs.len(), 1);
    let forced = &report.forced_reorgs[0];
    assert_eq!(forced.outcome, "refused");
    assert_eq!(forced.ancestor, Some(fixture.checkpoint.block));
    assert_eq!(forced.frames_applied, None);
    assert_eq!(forced.resumed_after_commits, Some(SKIPPED));
    assert_eq!(report.resyncs.len(), 1);
    let resync = &report.resyncs[0];
    assert_eq!(resync.at_sequence, recovery_sequence);
    assert_eq!(resync.block, fixture.checkpoint.block.number);
    assert!(resync.continuous);
    assert_eq!(resync.commits_skipped, 0);
    assert_eq!(resync.commits_skipped_before_restore, SKIPPED);
    assert_eq!(report.rewind_replayed_commits, DEPTH + SKIPPED, "{:?}", report.failures);
    let replayed = &report.blocks[(DEPTH + 1) as usize..(2 * DEPTH + SKIPPED + 1) as usize];
    assert_eq!(replayed[0].number, fixture.checkpoint.block.number + 1);
    assert_eq!(replayed[0].sequence, first_sequence);
    assert!(replayed.iter().all(|block| block.verdict == "accepted"));
    let after: Vec<_> = report.blocks.iter().filter(|block| block.sequence >= resume_at).collect();
    assert_eq!(after.len(), AFTER as usize);
    assert!(after.iter().all(|block| block.verdict == "accepted"));
    assert_eq!(report.commits, commits.len() as u64 + 1);
    assert_eq!(report.reorgs_applied, 1);
    assert_eq!(report.winning_branch_incomplete, 0);
    assert!(report.agreed() && report.continuous() && report.complete(), "{:?}", report.failures);

    // Exercise the recovery evidence gate with both an applied injection and a later refusal.
    let at = fixture.checkpoint.block.number + 3;
    let refusal_at = fixture.checkpoint.block.number + DEPTH;
    let json = dir.join("recovery.jsonl");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_ps-replay"))
        .arg(&dir)
        .args([
            "--no-mutations",
            "--retain-depth",
            "3",
            "--undo-record",
            "--warm-shrink",
            "never",
            "--forced-reorg",
            &format!("2@{at}"),
            "--forced-reorg",
            &format!("{DEPTH}@{refusal_at}"),
        ])
        .arg("--json")
        .arg(&json)
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_ps-replay"))
        .arg("--list-frames")
        .arg(&dir)
        .output()
        .unwrap();
    assert!(output.status.success());
    let inventory = dir.join("frames.jsonl");
    std::fs::write(&inventory, output.stdout).unwrap();
    let args = vec![
        "--mode".into(),
        "recovery".into(),
        "--at".into(),
        at.to_string(),
        "--commits".into(),
        (commits.len() + 1).to_string(),
        "--refusal-at".into(),
        refusal_at.to_string(),
        "--refusal-depth".into(),
        DEPTH.to_string(),
        "--checkpoint-sequence".into(),
        recovery_sequence.to_string(),
        "--checkpoints-skimmed".into(),
        "0".into(),
        "--frames".into(),
        inventory.to_str().unwrap().into(),
    ];
    let rows = evidence_with_test_provenance(&json);
    let checked = dir.join("checked-recovery.jsonl");
    let output = check_evidence(&checked, &rows, &args);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    for (pointer, value, message) in [
        ("/1/rewind_replayed_commits", serde_json::json!(0), "rewind_replayed_commits"),
        ("/1/resyncs/0/commits_skipped", serde_json::json!(SKIPPED), "commits_skipped"),
        ("/1/resyncs/0/at_sequence", serde_json::json!(recovery_sequence + 1), "at_sequence"),
        ("/1/checkpoints_skimmed", serde_json::json!(1), "checkpoints_skimmed"),
    ] {
        let mut bad = serde_json::to_value(&rows).unwrap();
        *bad.pointer_mut(pointer).unwrap() = value;
        let bad: Vec<serde_json::Value> = serde_json::from_value(bad).unwrap();
        let output = check_evidence(&checked, &bad, &args);
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains(message));
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn an_applied_forced_reorg_serializes_the_smoke_evidence() {
    let dir = spool_dir("batch-forced-smoke-json");
    let (fixture, commits) = common::empty_chain::chain(8);
    write_manifest(&dir);
    let mut next = write_checkpoint(&dir, 1, &fixture);
    for commit in &commits {
        write_frame(&dir, next, FrameKind::Commit, commit);
        next += 1;
    }
    write_frame(&dir, next, FrameKind::End, &end_frame(next, EndKind::Shutdown));
    let at = fixture.checkpoint.block.number + 3;
    let json = dir.join("smoke.jsonl");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_ps-replay"))
        .arg(&dir)
        .args([
            "--no-mutations",
            "--retain-depth",
            "3",
            "--undo-record",
            "--warm-shrink",
            "never",
            "--forced-reorg",
            &format!("2@{at}"),
        ])
        .arg("--json")
        .arg(&json)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let rows = evidence_with_test_provenance(&json);
    let checked = dir.join("checked-smoke.jsonl");
    let args = vec!["--at".into(), at.to_string(), "--commits".into(), "8".into()];
    let output = check_evidence(&checked, &rows, &args);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));

    for (pointer, value, message) in [
        ("/1/forced_reorgs/0/frames_applied", serde_json::json!(0), "frames_applied"),
        ("/1/forced_reorgs/0/ancestor/number", serde_json::json!(at - 1), "number"),
        ("/0/provenance/build_dirty", serde_json::json!(true), "build_dirty"),
        (
            "/0/provenance/build_commit",
            serde_json::json!("2222222222222222222222222222222222222222"),
            "build_commit",
        ),
    ] {
        let mut bad = serde_json::to_value(&rows).unwrap();
        *bad.pointer_mut(pointer).unwrap() = value;
        let bad: Vec<serde_json::Value> = serde_json::from_value(bad).unwrap();
        let output = check_evidence(&checked, &bad, &args);
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains(message));
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_frames_only_forced_reorg_serializes_layout_aware_evidence() {
    let dir = spool_dir("batch-forced-frames-json");
    let (fixture, commits) = common::empty_chain::chain(8);
    write_manifest(&dir);
    let mut next = write_checkpoint(&dir, 1, &fixture);
    for commit in &commits {
        write_frame(&dir, next, FrameKind::Commit, commit);
        next += 1;
    }
    write_frame(&dir, next, FrameKind::End, &end_frame(next, EndKind::Shutdown));
    let at = fixture.checkpoint.block.number + 3;
    let json = dir.join("frames.jsonl");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_ps-replay"))
        .arg(&dir)
        .args([
            "--no-mutations",
            "--retain-depth",
            "3",
            "--undo-record",
            "--undo-layout",
            "frames",
            "--forced-reorg",
            &format!("2@{at}"),
        ])
        .arg("--json")
        .arg(&json)
        .arg("--undo-dir")
        .arg(dir.join("undo"))
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));

    let rows = evidence_with_test_provenance(&json);
    assert_eq!(rows[0]["undo_layout"], "frames");
    assert_eq!(rows[0]["undo_resident_blocks"], 1);
    assert_eq!(rows[0]["undo_dir"], dir.join("undo").to_str().unwrap());
    assert_eq!(rows[1]["forced_reorgs"][0]["frames_applied"], 2);
    let checked = dir.join("checked-frames.jsonl");
    let args = vec!["--at".into(), at.to_string(), "--commits".into(), "8".into()];
    let output = check_evidence(&checked, &rows, &args);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));

    let mut bad = rows.clone();
    bad[1]["forced_reorgs"][0]["frames_applied"] = serde_json::json!(1);
    let output = check_evidence(&checked, &bad, &args);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("frames_applied"));

    // `check_evidence` passes the manifest's own layout; a later `--layout` wins, which is how a
    // runner pins the layout it launched instead of taking the manifest's word for it.
    let mut pinned = args.clone();
    pinned.extend(["--layout".to_string(), "hybrid".to_string()]);
    let output = check_evidence(&checked, &rows, &pinned);
    assert!(!output.status.success(), "a frames-only run passed a check pinned to hybrid");
    assert!(String::from_utf8_lossy(&output.stderr).contains("undo_layout"));
    std::fs::remove_dir_all(dir).unwrap();
}

/// A re-applied block is timed from the boundary its re-application opened, so nothing measured on
/// the first pass may sit among its leaves: the carried decode leaf put the phase sum past the wall
/// on every re-applied block of the 10k spool's forced runs.
#[test]
fn a_reapplied_block_carries_no_first_pass_transport_leaves() {
    let dir = spool_dir("batch-forced-reapply-timing");
    let (fixture, commits) = common::empty_chain::chain(8);
    write_manifest(&dir);
    let mut next = write_checkpoint(&dir, 1, &fixture);
    for commit in &commits {
        write_frame(&dir, next, FrameKind::Commit, commit);
        next += 1;
    }
    write_frame(&dir, next, FrameKind::End, &end_frame(next, EndKind::Shutdown));
    let report = replay(
        &dir,
        &ReplayOptions {
            retain_depth: RetentionDepth::new(3).unwrap(),
            undo_record: true,
            forced_reorgs: vec![ForcedReorg { depth: 2, at: fixture.checkpoint.block.number + 3 }],
            ..options()
        },
    )
    .expect("the synthetic corpus reads");

    let forced = &report.forced_reorgs[0];
    assert_eq!(forced.outcome, "applied");
    assert_eq!(forced.reapplied_sequences.len(), 2);
    assert_eq!(report.blocks.len(), commits.len() + 2);
    for sequence in &forced.reapplied_sequences {
        let attempts: Vec<_> =
            report.blocks.iter().filter(|block| block.sequence == *sequence).collect();
        assert_eq!(attempts.len(), 2, "one first pass and one re-application of {sequence}");
        assert!(attempts[0].phases.frame_decode_us.is_some() && attempts[0].delivery_us.is_some());
        assert_eq!(attempts[1].phases.frame_decode_us, None, "re-applied {sequence}");
        assert_eq!(attempts[1].delivery_us, None, "re-applied {sequence}");
    }
    assert_eq!(report.timing_anomalies, 0);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_recovery_checkpoint_at_the_exact_ancestor_is_continuous() {
    let dir = spool_dir("batch-continuous");
    let fixture = fixture();
    write_manifest(&dir);
    let mut next = write_checkpoint(&dir, 1, &fixture);
    write_frame(&dir, next, FrameKind::Reorg, &reorg_above(anchor(&fixture)));
    next += 1;
    // The producer's recovery checkpoint, authenticated at the block the reorg named.
    next = write_checkpoint(&dir, next, &fixture);
    write_frame(&dir, next, FrameKind::End, &end_frame(next, EndKind::Shutdown));

    let report = replay(&dir, &options()).expect("the corpus reads");

    assert_eq!(report.resyncs.len(), 1);
    let resync = &report.resyncs[0];
    assert!(resync.continuous, "the checkpoint landed on the exact block recovery asked for");
    assert_eq!(resync.block, ANCHOR_BLOCK);
    assert_eq!(resync.unverified, None);
    assert_eq!(resync.commits_skipped, 0);
    assert!(report.agreed() && report.continuous() && report.complete());
}

#[test]
fn a_recovery_checkpoint_elsewhere_is_an_explicit_reset() {
    let dir = spool_dir("batch-reset");
    let fixture = fixture();
    let elsewhere = fixture_at(ANCHOR_BLOCK + 40);
    write_manifest(&dir);
    let mut next = write_checkpoint(&dir, 1, &fixture);
    write_frame(&dir, next, FrameKind::Reorg, &reorg_above(anchor(&fixture)));
    next += 1;
    next = write_checkpoint(&dir, next, &elsewhere);
    write_frame(&dir, next, FrameKind::End, &end_frame(next, EndKind::Shutdown));

    let report = replay(&dir, &options()).expect("the corpus reads");

    assert_eq!(report.resyncs.len(), 1);
    let resync = &report.resyncs[0];
    assert!(
        !resync.continuous,
        "a checkpoint at the new tip is a reset, and reporting it as continuous recovery is the \
         one claim this format exists to prevent"
    );
    assert_eq!(resync.block, ANCHOR_BLOCK + 40);
    assert_eq!(
        resync.unverified,
        Some((ANCHOR_BLOCK + 1, ANCHOR_BLOCK + 40)),
        "and the interval nothing validated is named rather than implied"
    );
    assert!(report.agreed(), "the blocks it did compare still agreed");
    assert!(!report.continuous());
    assert!(report.complete(), "the pair is sound again, so the run reached the end");
}

#[test]
fn a_mid_stream_checkpoint_no_longer_corrupts_the_restore() {
    // The regression. The restore was one-shot and the chunk buffer was never cleared, so a second
    // checkpoint's chunks were appended to the first checkpoint's list and the second checkpoint
    // was silently never installed.
    let dir = spool_dir("batch-midstream");
    let first = fixture();
    let second = fixture_at(ANCHOR_BLOCK + 7);
    write_manifest(&dir);
    let mut next = write_checkpoint(&dir, 1, &first);
    next = write_checkpoint(&dir, next, &second);
    write_frame(&dir, next, FrameKind::End, &end_frame(next, EndKind::Shutdown));

    let report = replay(&dir, &options()).expect("the corpus reads");

    assert_eq!(
        report.failures.len(),
        1,
        "the unannounced checkpoint is reported once: {:?}",
        report.failures
    );
    assert!(report.failures[0].contains("unannounced checkpoint"));
    assert_eq!(report.resyncs.len(), 1, "and it re-bootstrapped rather than being swallowed");
    assert_eq!(report.resyncs[0].block, ANCHOR_BLOCK + 7);
    assert!(!report.resyncs[0].continuous);
}

#[test]
fn a_reset_frame_stops_verification_once() {
    let dir = spool_dir("batch-reset-frame");
    let fixture = fixture();
    write_manifest(&dir);
    let mut next = write_checkpoint(&dir, 1, &fixture);
    write_frame(
        &dir,
        next,
        FrameKind::Reset,
        &StreamEvent::Reset(Reset {
            reason: ResetReason::SnapshotRequired,
            detail: "cold reset".into(),
        }),
    );
    next += 1;
    for number in 0..2 {
        write_frame(
            &dir,
            next,
            FrameKind::Commit,
            &commit_frame(ANCHOR_BLOCK + 1 + number, B256::ZERO),
        );
        next += 1;
    }
    write_frame(&dir, next, FrameKind::End, &end_frame(next, EndKind::Shutdown));

    let report = replay(&dir, &options()).expect("the corpus reads");

    assert!(
        report.failures.is_empty(),
        "the producer saying it reset is a lifecycle fact, not a failure of this replay: {:?}",
        report.failures
    );
    assert_eq!(report.skipped_awaiting_resync, 2);
    assert_eq!(report.terminal_kind, Some("awaiting_resync"));
    assert!(report.agreed() && !report.continuous() && !report.complete());
}

/// The window a restore would have to replay is bounded, and a corpus past the bound degrades to
/// an explicit reset rather than to a clean continuous recovery it did not earn.
///
/// This is also what proves the window is computed for the *waiting* path at all: only a
/// `window_from` derived from the announcement that set the target can be refused for its size.
/// Before that, `AwaitingResync` forced `window_from: None`, so a consumer that could not undo
/// skipped the winning branch, installed the ancestor checkpoint, and met the next commit with a
/// parent it never built.
///
/// The bound is set low here on purpose. The branch under test is the classification, and four
/// thousand synthetic frames would test the arithmetic instead.
#[test]
fn a_window_past_the_bound_degrades_to_an_explicit_reset() {
    let dir = spool_dir("batch-window-refused");
    let fixture = fixture();
    write_manifest(&dir);
    let mut next = write_checkpoint(&dir, 1, &fixture);
    write_frame(&dir, next, FrameKind::Reorg, &reorg_above(anchor(&fixture)));
    next += 1;
    for number in 0..3 {
        write_frame(
            &dir,
            next,
            FrameKind::Commit,
            &commit_frame(ANCHOR_BLOCK + 1 + number, B256::repeat_byte(0xb2)),
        );
        next += 1;
    }
    // The producer's recovery checkpoint, at the exact block the reorg named.
    next = write_checkpoint(&dir, next, &fixture);
    write_frame(&dir, next, FrameKind::End, &end_frame(next, EndKind::Shutdown));

    let options = ReplayOptions { max_rewind_frames: 2, ..options() };
    let report = replay(&dir, &options).expect("the corpus reads");

    assert_eq!(
        report.rewind_windows_refused, 1,
        "a three-frame window under a two-frame bound is refused — and it could only be refused \
         if the waiting path offered one at all"
    );
    assert_eq!(report.rewind_replayed_commits, 0, "a refused window replays nothing");
    assert_eq!(report.resyncs.len(), 1);
    let resync = &report.resyncs[0];
    assert_eq!(resync.block, ANCHOR_BLOCK, "it still landed on the block recovery asked for");
    assert!(
        !resync.continuous,
        "but it stands below three commits it will not replay, which is the explicit reset the \
         bound exists to degrade to. Reporting it as continuous is how a corpus that ends at its \
         checkpoint could claim a clean recovery having verified none of the branch"
    );
    assert_eq!(resync.commits_skipped, 3, "and the commits it did not replay are counted");
    assert!(!report.continuous(), "the run as a whole says so too");
}

#[test]
fn a_malformed_reorg_is_a_failure() {
    let dir = spool_dir("batch-malformed");
    let fixture = fixture();
    write_manifest(&dir);
    let mut next = write_checkpoint(&dir, 1, &fixture);
    write_frame(
        &dir,
        next,
        FrameKind::Reorg,
        &StreamEvent::Reorg(Reorg {
            common_ancestor: anchor(&fixture),
            abandoned: Vec::new(),
            winning_tip: None,
        }),
    );
    next += 1;
    write_frame(&dir, next, FrameKind::End, &end_frame(next, EndKind::Shutdown));

    let report = replay(&dir, &options()).expect("the corpus reads");

    assert_eq!(report.reorgs_inapplicable, 0, "a frame that is not a reorg is not a deep reorg");
    assert_eq!(report.failures.len(), 1, "{:?}", report.failures);
    assert!(!report.agreed(), "a producer emitting a frame this shape is a defect somewhere");
}

#[test]
fn forcing_a_restore_installs_the_recovery_checkpoint() {
    // Skimming shows the producer's recovery checkpoint agrees with the generation this replay
    // recovered to. It does not show that the snapshot behind it restores anything, which is the
    // claim a consumer holding no retained generation depends on — so the flag forces the install.
    //
    // What this test can reach: the flag parses, plumbs through, and a forced install produces a
    // continuous run. What it cannot: the skim-versus-install *difference*, which only appears
    // when the reorg was applied, and a synthetic spool holds no commit that passes mainnet
    // admission. That half is the live gate's, against a producer that publishes one.
    let dir = spool_dir("batch-forced-restore");
    let fixture = fixture();
    write_manifest(&dir);
    let mut next = write_checkpoint(&dir, 1, &fixture);
    write_frame(&dir, next, FrameKind::Reorg, &reorg_above(anchor(&fixture)));
    next += 1;
    let recovery_at = next;
    next = write_checkpoint(&dir, next, &fixture);
    write_frame(&dir, next, FrameKind::End, &end_frame(next, EndKind::Shutdown));

    let forced = replay(&dir, &ReplayOptions { force_restore_at: Some(recovery_at), ..options() })
        .expect("the corpus reads");

    assert_eq!(forced.resyncs.len(), 1, "the checkpoint was installed rather than compared");
    assert_eq!(forced.resyncs[0].at_sequence, recovery_at);
    assert!(forced.resyncs[0].continuous);
    assert_eq!(forced.checkpoints_skimmed, 0);
    assert!(forced.agreed() && forced.continuous() && forced.complete());
}
