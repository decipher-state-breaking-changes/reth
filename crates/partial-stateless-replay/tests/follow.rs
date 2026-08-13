//! Fail-closed behaviour of the live follower, against synthetic spools.
//!
//! Every test writes a spool the way the producer does — one atomically named frame file per
//! sequence — and asserts the follower's *state*, not just its logs: verdicts stop on every
//! delivery violation, and only a checkpoint that verifies end to end restarts them.
//!
//! The restorable checkpoint is a real one: a one-account state proved with the same trie
//! machinery the producer's export uses, so `restore` runs the full verification path rather
//! than a stub. What no synthetic spool can supply is a commit that passes mainnet admission —
//! that is the live gate's job — so the streaming-phase tests here exercise the checks that run
//! *before* admission, which is exactly where the delivery grammar lives.

mod common;

use alloy_primitives::B256;
use common::{
    commit_frame, end_frame, fixture, fixture_at, manifest, options, spool_dir, write_checkpoint,
    write_frame, ANCHOR_BLOCK,
};
use partial_stateless_replay::{follow, FollowOutcome, NeedsSnapshotReason};
use partial_stateless_stream::{
    BlockRef, End, EndKind, FrameKind, Reset, ResetReason, StreamEvent,
};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// Runs the follower with a JSONL path so the state lines can be asserted on, not just counters.
fn dir_json(dir: &Path) -> PathBuf {
    dir.join("follow.jsonl")
}

/// The last `kind=state` record the follower wrote.
fn last_state_line(path: &Path) -> serde_json::Value {
    let text = fs::read_to_string(path).expect("the follower wrote its records");
    text.lines()
        .rev()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find(|value| value["kind"] == "state")
        .expect("at least one state record")
}

#[test]
fn a_foreign_chain_is_refused_before_anything_else() {
    let dir = spool_dir("foreign-chain");
    let mut foreign = manifest();
    foreign.chain_id = 10;
    write_frame(&dir, 0, FrameKind::Manifest, &StreamEvent::Manifest(foreign));

    let error = follow(&dir, &options()).expect_err("identity is checked first");
    assert!(error.to_string().contains("configured for mainnet"), "{error}");
    let _ = fs::remove_dir_all(&dir);
}

/// `Manifest` + `End` is an intentionally closed empty stream — closed, and distinctly not a
/// stream anyone verified anything against.
#[test]
fn a_stream_that_ends_before_any_checkpoint_is_a_distinct_outcome() {
    let dir = spool_dir("ended-early");
    write_frame(&dir, 0, FrameKind::Manifest, &StreamEvent::Manifest(manifest()));
    write_frame(&dir, 1, FrameKind::End, &end_frame(1, EndKind::ExportFailure));

    let report = follow(&dir, &options()).expect("follows");
    assert!(matches!(
        report.outcome,
        FollowOutcome::Ended { kind: EndKind::ExportFailure, before_checkpoint: true }
    ));
    assert_eq!(report.blocks_verified, 0);
    let _ = fs::remove_dir_all(&dir);
}

/// A live checkpoint without its own header could never admit H + 1 — `NoAcceptedParent` is a
/// rejection, not a wait — so the follower refuses to open the stream on it.
#[test]
fn a_headless_checkpoint_cannot_open_the_stream() {
    let dir = spool_dir("headless");
    write_frame(&dir, 0, FrameKind::Manifest, &StreamEvent::Manifest(manifest()));
    let fixture = fixture();
    let mut headless = fixture.checkpoint.clone();
    headless.accepted_head_rlp = Vec::new();
    let chunks = headless.chunk(&fixture.package_bytes, 256);
    let _ = chunks;
    write_frame(&dir, 1, FrameKind::Checkpoint, &StreamEvent::Checkpoint(headless));

    let report = follow(&dir, &options()).expect("follows");
    assert!(matches!(report.outcome, FollowOutcome::IdleTimeout { waiting_in: "needs_snapshot" }));
    assert_eq!(report.last_needs_snapshot, Some(NeedsSnapshotReason::HeadlessCheckpoint));
    assert_eq!(report.blocks_verified, 0, "no verdict was ever published");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_second_manifest_is_an_epoch_change() {
    let dir = spool_dir("epoch");
    write_frame(&dir, 0, FrameKind::Manifest, &StreamEvent::Manifest(manifest()));
    write_frame(&dir, 1, FrameKind::Manifest, &StreamEvent::Manifest(manifest()));

    let report = follow(&dir, &options()).expect("follows");
    assert_eq!(report.last_needs_snapshot, Some(NeedsSnapshotReason::EpochChange));
    assert!(matches!(report.outcome, FollowOutcome::IdleTimeout { waiting_in: "needs_snapshot" }));
    let _ = fs::remove_dir_all(&dir);
}

/// The gap arrives as a delivery fault — sequence 1 never exists while 2 does — and verdicts
/// stop rather than the hole being skipped.
#[test]
fn a_sequence_gap_is_never_skipped() {
    let dir = spool_dir("gap");
    write_frame(&dir, 0, FrameKind::Manifest, &StreamEvent::Manifest(manifest()));
    write_frame(
        &dir,
        2,
        FrameKind::Reset,
        &StreamEvent::Reset(Reset { reason: ResetReason::Gap, detail: "orphan".into() }),
    );

    let report = follow(&dir, &options()).expect("follows");
    assert_eq!(report.last_needs_snapshot, Some(NeedsSnapshotReason::Gap));
    assert_eq!(report.blocks_verified, 0);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn two_claims_on_one_sequence_are_a_duplicate_conflict() {
    let dir = spool_dir("dup");
    write_frame(&dir, 0, FrameKind::Manifest, &StreamEvent::Manifest(manifest()));
    write_frame(
        &dir,
        1,
        FrameKind::Reset,
        &StreamEvent::Reset(Reset { reason: ResetReason::Gap, detail: "one".into() }),
    );
    write_frame(&dir, 1, FrameKind::End, &end_frame(1, EndKind::Shutdown));

    let report = follow(&dir, &options()).expect("follows");
    assert_eq!(report.last_needs_snapshot, Some(NeedsSnapshotReason::DuplicateConflict));
    let _ = fs::remove_dir_all(&dir);
}

/// A frame that is not a chunk, arriving while the snapshot is incomplete, is a grammar
/// violation — a chunk that merely has not arrived yet is patience, not a fault.
#[test]
fn a_wrong_frame_during_chunk_collection_is_a_protocol_violation() {
    let dir = spool_dir("chunk-grammar");
    write_frame(&dir, 0, FrameKind::Manifest, &StreamEvent::Manifest(manifest()));
    let fixture = fixture();
    let mut checkpoint = fixture.checkpoint.clone();
    // Declare the real chunks, then deliver something else where chunk 0 belongs.
    checkpoint.describe(&fixture.package_bytes, 256);
    write_frame(&dir, 1, FrameKind::Checkpoint, &StreamEvent::Checkpoint(checkpoint));
    write_frame(
        &dir,
        2,
        FrameKind::Reset,
        &StreamEvent::Reset(Reset { reason: ResetReason::Gap, detail: "not a chunk".into() }),
    );

    let report = follow(&dir, &options()).expect("follows");
    assert_eq!(report.last_needs_snapshot, Some(NeedsSnapshotReason::ProtocolViolation));
    let _ = fs::remove_dir_all(&dir);
}

/// The full happy prefix: identity, checkpoint, snapshot, restore — and a clean `End`.
#[test]
fn a_restorable_checkpoint_opens_the_stream_and_an_end_closes_it() {
    let dir = spool_dir("restore-end");
    write_frame(&dir, 0, FrameKind::Manifest, &StreamEvent::Manifest(manifest()));
    let fixture = fixture();
    let next = write_checkpoint(&dir, 1, &fixture);
    write_frame(&dir, next, FrameKind::End, &end_frame(next, EndKind::Shutdown));

    let report = follow(&dir, &options()).expect("follows");
    assert!(matches!(
        report.outcome,
        FollowOutcome::Ended { kind: EndKind::Shutdown, before_checkpoint: false }
    ));
    assert_eq!(report.restores, 1, "the pair restored from the recorded snapshot, no database");
    assert_eq!(report.needs_snapshot_entries, 0);
    let _ = fs::remove_dir_all(&dir);
}

/// The first commit must be exactly H + 1 onto H — checked on the frame itself, before anything
/// is decoded, and typed as the delivery failure it is.
#[test]
fn a_first_commit_that_is_not_h_plus_one_is_a_gap() {
    let dir = spool_dir("wrong-child");
    write_frame(&dir, 0, FrameKind::Manifest, &StreamEvent::Manifest(manifest()));
    let fixture = fixture();
    let next = write_checkpoint(&dir, 1, &fixture);
    // Right parent hash, wrong height: still not the checkpoint's child.
    write_frame(
        &dir,
        next,
        FrameKind::Commit,
        &commit_frame(ANCHOR_BLOCK + 5, fixture.checkpoint.block.hash),
    );

    let report = follow(&dir, &options()).expect("follows");
    assert_eq!(report.last_needs_snapshot, Some(NeedsSnapshotReason::Gap));
    assert_eq!(report.blocks_verified, 0, "no verdict on a stream that skipped blocks");
    let _ = fs::remove_dir_all(&dir);
}

/// A reorg frame is fail-closed in follow mode: applying it is S4, and verdicts past an
/// unapplied reorg would describe a branch the producer left.
#[test]
fn a_malformed_reorg_frame_is_a_protocol_violation_with_no_target() {
    // A frame that abandons no block is not a reorg, so it names no ancestor a snapshot could be
    // authenticated at. Kept apart from the reorgs this follower can act on, because a request
    // that names nothing is a different thing to hand an operator than one that names a block.
    let dir = spool_dir("reorg-malformed");
    write_frame(&dir, 0, FrameKind::Manifest, &StreamEvent::Manifest(manifest()));
    let fixture = fixture();
    let next = write_checkpoint(&dir, 1, &fixture);
    write_frame(
        &dir,
        next,
        FrameKind::Reorg,
        &StreamEvent::Reorg(partial_stateless_stream::Reorg {
            common_ancestor: BlockRef { number: ANCHOR_BLOCK, hash: fixture.checkpoint.block.hash },
            abandoned: vec![],
            winning_tip: None,
        }),
    );

    let report = follow(&dir, &options()).expect("follows");
    assert_eq!(report.last_needs_snapshot, Some(NeedsSnapshotReason::ProtocolViolation));
    assert_eq!(report.blocks_verified, 0);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn an_inapplicable_reorg_names_the_block_a_snapshot_must_be_authenticated_at() {
    // The pair restored a moment ago and has applied nothing, so it has no generation to give
    // back — the permanent condition of a follower that just started, and the reason the
    // producer's re-checkpoint exists. What makes the refusal actionable is the ancestor.
    let dir = spool_dir("reorg-inapplicable");
    write_frame(&dir, 0, FrameKind::Manifest, &StreamEvent::Manifest(manifest()));
    let fixture = fixture();
    let next = write_checkpoint(&dir, 1, &fixture);
    write_frame(
        &dir,
        next,
        FrameKind::Reorg,
        &StreamEvent::Reorg(partial_stateless_stream::Reorg {
            common_ancestor: fixture.checkpoint.block,
            abandoned: vec![BlockRef { number: ANCHOR_BLOCK + 1, hash: B256::repeat_byte(0xa1) }],
            winning_tip: Some(BlockRef { number: ANCHOR_BLOCK + 1, hash: B256::repeat_byte(0xb2) }),
        }),
    );

    let mut with_json = options();
    with_json.verdicts = Some(dir_json(&dir));
    let report = follow(&dir, &with_json).expect("follows");

    assert_eq!(report.last_needs_snapshot, Some(NeedsSnapshotReason::SnapshotRequired));
    assert_eq!(report.reorgs_applied, 0);
    assert_eq!(report.blocks_verified, 0);
    let line = last_state_line(&dir_json(&dir));
    assert_eq!(line["target_ancestor"], serde_json::json!(ANCHOR_BLOCK));
    assert_eq!(line["epoch"], serde_json::json!(1));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_recovery_checkpoint_at_the_exact_ancestor_is_continuous() {
    let dir = spool_dir("reorg-continuous");
    write_frame(&dir, 0, FrameKind::Manifest, &StreamEvent::Manifest(manifest()));
    let fixture = fixture();
    let mut next = write_checkpoint(&dir, 1, &fixture);
    write_frame(
        &dir,
        next,
        FrameKind::Reorg,
        &StreamEvent::Reorg(partial_stateless_stream::Reorg {
            common_ancestor: fixture.checkpoint.block,
            abandoned: vec![BlockRef { number: ANCHOR_BLOCK + 1, hash: B256::repeat_byte(0xa1) }],
            winning_tip: None,
        }),
    );
    next += 1;
    // The producer's recovery checkpoint, authenticated at the block the reorg named.
    next = write_checkpoint(&dir, next, &fixture);
    write_frame(&dir, next, FrameKind::End, &end_frame(next, EndKind::Shutdown));

    let report = follow(&dir, &options()).expect("follows");

    assert_eq!(report.restores, 2);
    assert_eq!(report.restores_continuous, 1, "it landed on exactly the block asked for");
    assert_eq!(report.restores_reset, 0);
    assert!(report.continuous(), "nothing canonical went unverified");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_recovery_checkpoint_elsewhere_is_an_explicit_reset() {
    let dir = spool_dir("reorg-reset");
    write_frame(&dir, 0, FrameKind::Manifest, &StreamEvent::Manifest(manifest()));
    let fixture = fixture();
    let elsewhere = fixture_at(ANCHOR_BLOCK + 40);
    let mut next = write_checkpoint(&dir, 1, &fixture);
    write_frame(
        &dir,
        next,
        FrameKind::Reorg,
        &StreamEvent::Reorg(partial_stateless_stream::Reorg {
            common_ancestor: fixture.checkpoint.block,
            abandoned: vec![BlockRef { number: ANCHOR_BLOCK + 1, hash: B256::repeat_byte(0xa1) }],
            winning_tip: None,
        }),
    );
    next += 1;
    next = write_checkpoint(&dir, next, &elsewhere);
    write_frame(&dir, next, FrameKind::End, &end_frame(next, EndKind::Shutdown));

    let report = follow(&dir, &options()).expect("follows");

    assert_eq!(report.restores_reset, 1);
    assert_eq!(report.restores_continuous, 0);
    assert!(
        !report.continuous(),
        "a checkpoint at the new tip makes no validation claim for the interval it skipped"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_later_reorg_supersedes_the_stale_recovery_target() {
    // The chain kept moving while the follower waited. A checkpoint answering the *first* reorg's
    // ancestor would be answering a superseded question, and calling that continuous recovery
    // would claim an interval nothing validated.
    let dir = spool_dir("reorg-supersede");
    write_frame(&dir, 0, FrameKind::Manifest, &StreamEvent::Manifest(manifest()));
    let fixture = fixture();
    let later = fixture_at(ANCHOR_BLOCK + 40);
    let mut next = write_checkpoint(&dir, 1, &fixture);
    write_frame(
        &dir,
        next,
        FrameKind::Reorg,
        &StreamEvent::Reorg(partial_stateless_stream::Reorg {
            common_ancestor: fixture.checkpoint.block,
            abandoned: vec![BlockRef { number: ANCHOR_BLOCK + 1, hash: B256::repeat_byte(0xa1) }],
            winning_tip: None,
        }),
    );
    next += 1;
    // A second reorg during the outage, naming a different ancestor.
    write_frame(
        &dir,
        next,
        FrameKind::Reorg,
        &StreamEvent::Reorg(partial_stateless_stream::Reorg {
            common_ancestor: later.checkpoint.block,
            abandoned: vec![BlockRef { number: ANCHOR_BLOCK + 41, hash: B256::repeat_byte(0xc3) }],
            winning_tip: None,
        }),
    );
    next += 1;
    next = write_checkpoint(&dir, next, &fixture);
    write_frame(&dir, next, FrameKind::End, &end_frame(next, EndKind::Shutdown));

    let report = follow(&dir, &options()).expect("follows");

    assert_eq!(
        report.restores_continuous, 0,
        "the checkpoint answers the first reorg, which the second one replaced"
    );
    assert_eq!(report.restores_reset, 1);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_reset_during_the_scan_withdraws_the_recovery_target() {
    let dir = spool_dir("reorg-reset-withdraws");
    write_frame(&dir, 0, FrameKind::Manifest, &StreamEvent::Manifest(manifest()));
    let fixture = fixture();
    let mut next = write_checkpoint(&dir, 1, &fixture);
    write_frame(
        &dir,
        next,
        FrameKind::Reorg,
        &StreamEvent::Reorg(partial_stateless_stream::Reorg {
            common_ancestor: fixture.checkpoint.block,
            abandoned: vec![BlockRef { number: ANCHOR_BLOCK + 1, hash: B256::repeat_byte(0xa1) }],
            winning_tip: None,
        }),
    );
    next += 1;
    write_frame(
        &dir,
        next,
        FrameKind::Reset,
        &StreamEvent::Reset(Reset {
            reason: ResetReason::SnapshotRequired,
            detail: "the producer cold reset".into(),
        }),
    );
    next += 1;
    next = write_checkpoint(&dir, next, &fixture);
    write_frame(&dir, next, FrameKind::End, &end_frame(next, EndKind::Shutdown));

    let report = follow(&dir, &options()).expect("follows");

    assert_eq!(
        report.restores_continuous, 0,
        "a reset names no block, so nothing after it can be continuous with anything"
    );
    assert_eq!(report.restores_reset, 1);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn commits_in_the_recovery_gap_downgrade_a_continuous_recovery_to_a_reset() {
    let dir = spool_dir("reorg-gap-commits");
    write_frame(&dir, 0, FrameKind::Manifest, &StreamEvent::Manifest(manifest()));
    let fixture = fixture();
    let mut next = write_checkpoint(&dir, 1, &fixture);
    write_frame(
        &dir,
        next,
        FrameKind::Reorg,
        &StreamEvent::Reorg(partial_stateless_stream::Reorg {
            common_ancestor: fixture.checkpoint.block,
            abandoned: vec![BlockRef { number: ANCHOR_BLOCK + 1, hash: B256::repeat_byte(0xa1) }],
            winning_tip: None,
        }),
    );
    next += 1;
    // Winning-branch commits that went by while the follower was waiting: verified by nothing.
    for number in 0..2 {
        write_frame(
            &dir,
            next,
            FrameKind::Commit,
            &commit_frame(ANCHOR_BLOCK + 1 + number, B256::repeat_byte(0xb2)),
        );
        next += 1;
    }
    next = write_checkpoint(&dir, next, &fixture);
    write_frame(&dir, next, FrameKind::End, &end_frame(next, EndKind::Shutdown));

    let report = follow(&dir, &options()).expect("follows");

    assert_eq!(report.commits_skipped_in_recovery, 2);
    assert_eq!(
        report.restores_continuous, 0,
        "blocks went by that this follower never verified, so the recovery has a hole in it"
    );
    assert!(!report.continuous());
    let _ = fs::remove_dir_all(&dir);
}

/// Recovery reads the sequence space in order: an `End` below a later checkpoint means the
/// stream closed there, and whatever was appended past it must not restart verdicts.
#[test]
fn recovery_takes_an_end_below_a_later_checkpoint() {
    let dir = spool_dir("end-before-checkpoint");
    write_frame(&dir, 0, FrameKind::Manifest, &StreamEvent::Manifest(manifest()));
    let fixture = fixture();
    let next = write_checkpoint(&dir, 1, &fixture);
    // The gap: `next` never exists, so the follower enters NeedsSnapshot. The stream then ends —
    // and a checkpoint sits *past* the End, which on a closed stream is trailing garbage.
    write_frame(&dir, next + 1, FrameKind::End, &end_frame(next + 1, EndKind::Shutdown));
    let _ = write_checkpoint(&dir, next + 2, &fixture);

    let report = follow(&dir, &options()).expect("follows");
    assert!(matches!(
        report.outcome,
        FollowOutcome::Ended { kind: EndKind::Shutdown, before_checkpoint: false }
    ));
    assert_eq!(report.restores, 1, "the trailing checkpoint never restored a second pair");
    let _ = fs::remove_dir_all(&dir);
}

/// An End reached through the recovery scan is held to the same numbering promise as one
/// delivered in order; recovery is not a laxer grammar.
#[test]
fn a_mis_numbered_end_in_recovery_is_refused() {
    let dir = spool_dir("recovery-end-numbering");
    write_frame(&dir, 0, FrameKind::Manifest, &StreamEvent::Manifest(manifest()));
    write_frame(
        &dir,
        2,
        FrameKind::End,
        &StreamEvent::End(End { kind: EndKind::Shutdown, reason: "test".into(), last_sequence: 5 }),
    );

    let error = follow(&dir, &options()).expect_err("a numbering lie is refused");
    assert!(error.to_string().contains("names 5"), "{error}");
    let _ = fs::remove_dir_all(&dir);
}

/// Every verdict line names its payload's provenance: the standalone claim is that recorded
/// Engine payloads were consumed directly, and the per-block record is where that is proven
/// rather than asserted.
#[test]
fn a_verdict_line_carries_the_payload_provenance() {
    let dir = spool_dir("provenance");
    write_frame(&dir, 0, FrameKind::Manifest, &StreamEvent::Manifest(manifest()));
    let fixture = fixture();
    let next = write_checkpoint(&dir, 1, &fixture);
    write_frame(
        &dir,
        next,
        FrameKind::Commit,
        &commit_frame(ANCHOR_BLOCK + 1, fixture.checkpoint.block.hash),
    );

    let verdicts = dir.join("out").join("follow.jsonl");
    let mut with_verdicts = options();
    with_verdicts.verdicts = Some(verdicts.clone());
    let report = follow(&dir, &with_verdicts).expect("follows");

    // The synthetic commit carries no payload, so it is rejected — and the rejection line still
    // names what the payload was.
    assert!(matches!(report.outcome, FollowOutcome::Faulted { .. }));
    let lines = fs::read_to_string(&verdicts).expect("verdict stream");
    assert!(lines.contains("\"payload_provenance\":\"absent\""), "{lines}");
    let _ = fs::remove_dir_all(&dir);
}

/// The rebootstrap gate: a gap stops verdicts, a fresh checkpoint restarts them at its own
/// H′ + 1, and the commits that fell in between are counted rather than silently discarded.
///
/// The wrong-height commit after the second restore is what proves the resync armed a fresh
/// `H′ + 1` expectation rather than resuming where the old stream left off.
#[test]
fn a_fresh_checkpoint_rebootstraps_after_a_gap() {
    let dir = spool_dir("resync");
    write_frame(&dir, 0, FrameKind::Manifest, &StreamEvent::Manifest(manifest()));
    let fixture = fixture();
    let next = write_checkpoint(&dir, 1, &fixture);

    // The gap: `next` never exists. Two commits land beyond it and must be skipped, counted.
    write_frame(
        &dir,
        next + 1,
        FrameKind::Commit,
        &commit_frame(ANCHOR_BLOCK + 1, fixture.checkpoint.block.hash),
    );
    write_frame(
        &dir,
        next + 2,
        FrameKind::Commit,
        &commit_frame(ANCHOR_BLOCK + 2, B256::with_last_byte(1)),
    );
    // The recovery checkpoint, then a commit that is not its H′ + 1.
    let after_recovery = write_checkpoint(&dir, next + 3, &fixture);
    write_frame(
        &dir,
        after_recovery,
        FrameKind::Commit,
        &commit_frame(ANCHOR_BLOCK + 9, fixture.checkpoint.block.hash),
    );

    let report = follow(&dir, &options()).expect("follows");
    assert_eq!(report.restores, 2, "the rebootstrap restored a second pair");
    assert_eq!(report.needs_snapshot_entries, 2, "the gap, then the wrong child");
    assert_eq!(report.commits_skipped_in_recovery, 2, "skipped, recorded, never verified");
    assert_eq!(
        report.last_needs_snapshot,
        Some(NeedsSnapshotReason::Gap),
        "the wrong child after the rebootstrap proves H' + 1 was re-armed"
    );
    assert_eq!(report.blocks_verified, 0);
    let _ = fs::remove_dir_all(&dir);
}
