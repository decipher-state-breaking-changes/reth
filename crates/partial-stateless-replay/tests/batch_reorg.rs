//! What a batch replay does with the lifecycle events the corpus carries.
//!
//! Before reorg recovery landed, the driver replayed past a reorg it had not applied and reported
//! every block after it as a refusal — one real mainnet reorg turned 78 winning-branch commits into
//! 78 failures and 78 disagreements. The events are now part of the grammar: a reorg the pair can
//! undo is undone, one it cannot is a single typed stop, and only a checkpoint restarts
//! verification.
//!
//! These spools cannot carry a commit that passes mainnet admission, so what they pin is the
//! lifecycle around the commits rather than the commits. The recorded mainnet corpus is where the
//! undo itself is judged.

mod common;

use alloy_primitives::B256;
use common::{
    commit_frame, end_frame, fixture, fixture_at, manifest, spool_dir, write_checkpoint,
    write_frame, ANCHOR_BLOCK,
};
use partial_stateless_replay::{replay, ReplayOptions};
use partial_stateless_stream::{
    BlockRef, EndKind, FrameKind, Reorg, Reset, ResetReason, StreamEvent,
};
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
