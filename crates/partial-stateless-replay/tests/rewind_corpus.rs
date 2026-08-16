//! Deterministic successful-rewind coverage, from a *recorded* corpus.
//!
//! The synthetic-spool suites pin every refusal path of the rewind state machine, but no
//! synthetic spool can supply a commit that passes mainnet admission — so the success paths
//! (a window that replays clean, a kill inside it, a resume after it completed) would otherwise
//! wait for a live reorg to happen to occur. Instead, these tests *splice* a recorded spool into
//! the write-through ordering: real checkpoint, a reorg back to its own block, the recorded
//! winning-branch commits, the same checkpoint again at the stream tail, and the recorded live
//! edge behind it. Every frame that reaches admission is a producer-recorded one; only the
//! ordering is constructed.
//!
//! Corpus-gated and `#[ignore]`d: the corpus is measured in hundreds of megabytes and lives
//! beside the bench runs, not in the repository, so a default `cargo test` reports these as
//! ignored — never as passed-without-running. To run them:
//!
//! ```text
//! PS_REWIND_FIXTURE_SPOOL=/data/bench-runs/s2-capture/stream \
//!     cargo test -p partial-stateless-replay --release --test rewind_corpus \
//!     -- --ignored --test-threads=1
//! ```
//!
//! With the env var missing at actual run time, the tests panic rather than skip.

mod common;

use alloy_primitives::B256;
use partial_stateless_replay::{follow, FollowOptions, FollowOutcome, FollowReport};
use partial_stateless_stream::{
    decode_event, BlockRef, CommitFrame, FrameKind, FrameLimits, Reorg, StreamEvent,
};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::OnceLock,
};

/// How many recorded commits ride in the rewind window, and behind it on the live edge.
const WINDOW_COMMITS: u64 = 2;
const LIVE_COMMITS: u64 = 8;

struct Spliced {
    dir: PathBuf,
    /// The recovery checkpoint's announce sequence — the window's boundary and the ack's
    /// restore point.
    recovery_checkpoint: u64,
    /// The End frame's sequence.
    end: u64,
}

fn read_event(dir: &Path, sequence: u64, kind: FrameKind) -> StreamEvent {
    let path = dir.join(format!("{sequence:012}_{}.frame", kind.as_str()));
    let bytes = fs::read(&path).expect("the source frame reads");
    let (header, event, _) = decode_event(&bytes, &FrameLimits::default()).expect("decodes");
    assert_eq!(header.sequence, sequence, "the source spool is self-consistent");
    event
}

fn write_event(dir: &Path, sequence: u64, event: &StreamEvent) {
    let kind = match event {
        StreamEvent::Manifest(_) => FrameKind::Manifest,
        StreamEvent::Checkpoint(_) => FrameKind::Checkpoint,
        StreamEvent::SnapshotChunk(_) => FrameKind::SnapshotChunk,
        StreamEvent::Commit(_) => FrameKind::Commit,
        StreamEvent::Reorg(_) => FrameKind::Reorg,
        StreamEvent::Reset(_) => FrameKind::Reset,
        StreamEvent::End(_) => FrameKind::End,
    };
    common::write_frame(dir, sequence, kind, event);
}

fn commit_block(event: &StreamEvent) -> (BlockRef, B256) {
    let StreamEvent::Commit(frame) = event else { panic!("a commit frame was expected") };
    // Rebuilt below from the same parts, so peeking by split is non-destructive in effect.
    let (input, oracle) = frame.as_ref().clone().split();
    let block = input.block;
    let parent = input.parent_hash;
    let _ = CommitFrame::new(input, oracle);
    (block, parent)
}

/// Builds the spliced spool once per process and shares it read-only across the tests; each
/// test brings its own ack path, and the follower never writes into the spool.
fn spliced() -> &'static Spliced {
    static BUILT: OnceLock<Spliced> = OnceLock::new();
    BUILT.get_or_init(|| {
        let source = match std::env::var("PS_REWIND_FIXTURE_SPOOL") {
            Ok(path) if !path.is_empty() => PathBuf::from(path),
            _ => panic!(
                "PS_REWIND_FIXTURE_SPOOL is not set; these #[ignore]d tests only run \
                     against a recorded linear spool such as s2-capture/stream"
            ),
        };
        let dir = std::env::temp_dir().join(format!("ps-rewind-corpus-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("temp dir");

        let manifest = read_event(&source, 0, FrameKind::Manifest);
        let checkpoint_event = read_event(&source, 1, FrameKind::Checkpoint);
        let StreamEvent::Checkpoint(checkpoint) = &checkpoint_event else {
            panic!("the source spool's sequence 1 is not a checkpoint")
        };
        let anchor = checkpoint.block;
        let chunks = u64::from(checkpoint.snapshot_chunks);
        let commits_from = 2 + chunks;
        let needed = WINDOW_COMMITS + LIVE_COMMITS;
        let commits: Vec<StreamEvent> = (0..needed)
            .map(|index| read_event(&source, commits_from + index, FrameKind::Commit))
            .collect();
        let (first, first_parent) = commit_block(&commits[0]);
        assert_eq!(first_parent, anchor.hash, "the first commit is the checkpoint's child");
        let (tip, _) = commit_block(&commits[WINDOW_COMMITS as usize - 1]);

        // The write-through ordering: bootstrap pair, a reorg back to the anchor announcing
        // the recorded branch as the winner, the branch itself, and the recovery checkpoint
        // published at the tail — the same checkpoint, because the pair recovered to the
        // same block.
        write_event(&dir, 0, &manifest);
        write_event(&dir, 1, &checkpoint_event);
        for index in 0..chunks {
            write_event(&dir, 2 + index, &read_event(&source, 2 + index, FrameKind::SnapshotChunk));
        }
        let mut next = 2 + chunks;
        write_event(
            &dir,
            next,
            &StreamEvent::Reorg(Reorg {
                common_ancestor: anchor,
                abandoned: vec![BlockRef { number: first.number, hash: B256::repeat_byte(0xa1) }],
                winning_tip: Some(tip),
            }),
        );
        next += 1;
        for commit in commits.iter().take(WINDOW_COMMITS as usize) {
            write_event(&dir, next, commit);
            next += 1;
        }
        let recovery_checkpoint = next;
        write_event(&dir, next, &checkpoint_event);
        next += 1;
        for index in 0..chunks {
            write_event(&dir, next, &read_event(&source, 2 + index, FrameKind::SnapshotChunk));
            next += 1;
        }
        for commit in commits.iter().skip(WINDOW_COMMITS as usize) {
            write_event(&dir, next, commit);
            next += 1;
        }
        let end = next;
        write_event(
            &dir,
            end,
            &common::end_frame(end, partial_stateless_stream::EndKind::Shutdown),
        );
        Spliced { dir, recovery_checkpoint, end }
    })
}

fn run(spool: &Path, ack: &Path, resume: bool, max_blocks: Option<u64>) -> FollowReport {
    let options =
        FollowOptions { ack: Some(ack.to_path_buf()), resume, max_blocks, ..common::options() };
    follow(spool, &options).expect("follows")
}

fn ack_json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&fs::read_to_string(path).expect("ack exists")).expect("ack is json")
}

/// The success path continuity is earned by: the window replays clean, the recovery classifies
/// continuous, the rewound commits count as first verifications, and the branch tip announced
/// before the outage completes *inside* the window.
#[test]
#[ignore = "needs PS_REWIND_FIXTURE_SPOOL; see the module header"]
fn a_clean_rewind_replay_is_a_continuous_recovery() {
    let spliced = spliced();
    let ack = spliced.dir.join("ack-clean.json");
    let _ = fs::remove_file(&ack);

    let report = run(&spliced.dir, &ack, false, None);

    assert!(matches!(report.outcome, FollowOutcome::Ended { .. }), "{:?}", report.outcome);
    assert!(report.replay.agreed(), "every recorded commit verified: {:?}", report.replay.failures);
    assert_eq!(report.restores, 2, "the bootstrap install and the rewind install");
    assert_eq!(report.restores_continuous, 1, "the window replayed clean");
    assert_eq!(report.restores_reset, 0);
    assert_eq!(report.rewind_replayed_commits, WINDOW_COMMITS);
    assert_eq!(report.commits_skipped_in_recovery, 0, "rewound commits are verified, not skipped");
    assert_eq!(report.blocks_verified, WINDOW_COMMITS + LIVE_COMMITS);
    assert_eq!(report.winning_branches_completed, 1, "the announced tip arrived in the window");
    assert_eq!(report.winning_branches_incomplete, 0);
    assert!(report.continuous(), "the whole point: no interval left unaccounted for");

    let ack = ack_json(&ack);
    assert_eq!(ack["restored_from_sequence"], spliced.recovery_checkpoint);
    assert_eq!(
        ack["recovery"]["checkpoint_sequence"], spliced.recovery_checkpoint,
        "the window is the restore point's reconstruction recipe and outlives its replay"
    );
}

/// A kill inside the window resumes all-or-nothing: the ack still carries the window, the
/// restore replays it whole, and the commits the killed run never reached are first-verified
/// by the resumed one.
#[test]
#[ignore = "needs PS_REWIND_FIXTURE_SPOOL; see the module header"]
fn a_kill_inside_the_window_resumes_and_replays_it_whole() {
    let spliced = spliced();
    let ack = spliced.dir.join("ack-kill.json");
    let _ = fs::remove_file(&ack);

    // Stops after the first window verdict — mid-window, rewind still active.
    let killed = run(&spliced.dir, &ack, false, Some(1));
    assert!(matches!(killed.outcome, FollowOutcome::MaxBlocks));
    assert_eq!(killed.blocks_verified, 1, "one window commit verified before the stop");
    assert_eq!(killed.restores_reset, 1, "a run that stops mid-rewind reports the reset");
    let open = ack_json(&ack);
    assert_eq!(open["state"], "restored", "the install ack survived the mid-window acks");
    assert_eq!(open["recovery"]["replay_until"], spliced.recovery_checkpoint);

    let resumed = run(&spliced.dir, &ack, true, None);

    assert!(matches!(resumed.outcome, FollowOutcome::Ended { .. }), "{:?}", resumed.outcome);
    assert!(resumed.replay.agreed());
    assert_eq!(resumed.needs_snapshot_entries, 0, "no gap, no violation — the window carried it");
    assert_eq!(resumed.restores_continuous, 1, "the resumed replay finished the recovery");
    assert_eq!(resumed.rewind_replayed_commits, WINDOW_COMMITS);
    assert_eq!(
        resumed.blocks_verified,
        WINDOW_COMMITS + LIVE_COMMITS,
        "the whole window re-verifies — all-or-nothing, no mid-window progress to trust — and \
         the analyzer dedupes re-published frames by their sequence"
    );
    assert_eq!(resumed.catch_up_blocks, 0, "the install ack's watermark is the checkpoint itself");
}

/// A resume *after* the rewind completed: the ack's restore point is still the rewound
/// checkpoint, so the window must replay again — as catch-up this time, since every block in it
/// was already verified — before the live edge past the chunks makes sense. This is the exact
/// shape that used to gap: a recovery record cleared at completion left the ack pointing at a
/// checkpoint whose first live commit is not its child.
#[test]
#[ignore = "needs PS_REWIND_FIXTURE_SPOOL; see the module header"]
fn a_resume_after_the_rewind_completed_replays_the_window_as_catch_up() {
    let spliced = spliced();
    let ack = spliced.dir.join("ack-after.json");
    let _ = fs::remove_file(&ack);

    let first = run(&spliced.dir, &ack, false, None);
    assert!(matches!(first.outcome, FollowOutcome::Ended { .. }));
    assert!(first.continuous());

    let resumed = run(&spliced.dir, &ack, true, None);

    assert!(matches!(resumed.outcome, FollowOutcome::Ended { .. }), "{:?}", resumed.outcome);
    assert!(resumed.replay.agreed());
    assert_eq!(
        resumed.needs_snapshot_entries, 0,
        "the recipe on the ack is what keeps this from reading as a gap"
    );
    assert_eq!(resumed.blocks_verified, 0, "everything was verified once already");
    assert_eq!(
        resumed.catch_up_blocks,
        WINDOW_COMMITS + LIVE_COMMITS,
        "window and live edge alike re-derive on the way back to the watermark"
    );
    assert_eq!(resumed.restores_continuous, 1, "the re-run rewind still classifies itself");
    assert_eq!(ack_json(&ack)["last_sequence"], spliced.end, "the watermark returned to the End");
}
