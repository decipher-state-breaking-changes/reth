//! Stopping a follower and starting it again where it left off.
//!
//! A follower holds a coordinated pair, and a pair is state with no database under it. So a
//! restart cannot pick up where the last one stopped by reading a number — it has to rebuild the
//! pair from the checkpoint that built it and re-derive every block between there and the
//! watermark. What the ack contributes is *which* checkpoint, and that is the field version 2
//! added: a restart that had to guess could adopt a checkpoint the previous run examined and
//! refused, which is the one thing a restart must never be a way to do.
//!
//! These spools cannot carry a commit that passes mainnet admission, so what is pinned here is
//! what a restart reads, checks, and refuses — not the re-derivation itself, which the live gate
//! and the recorded corpus cover.

mod common;

use common::{
    end_frame, fixture, manifest, options, spool_dir, write_checkpoint, write_frame, ANCHOR_BLOCK,
};
use partial_stateless_replay::{follow, FollowOptions, FollowReport};
use partial_stateless_stream::{EndKind, FrameKind, Manifest, StreamEvent};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// A one-epoch spool whose checkpoint sits at sequence 1, closed with an `End`.
fn one_epoch(dir: &Path) -> u64 {
    write_frame(dir, 0, FrameKind::Manifest, &StreamEvent::Manifest(manifest()));
    let next = write_checkpoint(dir, 1, &fixture());
    write_frame(dir, next, FrameKind::End, &end_frame(next, EndKind::Shutdown));
    next
}

fn ack_path(dir: &Path) -> PathBuf {
    dir.join("ack.json")
}

fn read_ack(dir: &Path) -> serde_json::Value {
    let raw = fs::read_to_string(ack_path(dir)).expect("the follower wrote an ack");
    serde_json::from_str(&raw).expect("the ack is json")
}

fn run(dir: &Path, resume: bool) -> FollowReport {
    let options = FollowOptions { ack: Some(ack_path(dir)), resume, ..options() };
    follow(dir, &options).expect("follows")
}

#[test]
fn a_restore_records_the_checkpoint_it_came_from() {
    // Without this field a restart has to pick a checkpoint by proximity, and proximity cannot
    // tell "the one I restored from" apart from "one I looked at and would not install".
    let dir = spool_dir("ack-restored-from");
    one_epoch(&dir);

    run(&dir, false);

    let ack = read_ack(&dir);
    assert_eq!(ack["ack_version"], 2);
    assert_eq!(ack["restored_from_sequence"], 1);
    assert_eq!(ack["epoch"], 1);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn an_ack_written_while_waiting_says_so_and_names_the_block_it_waits_for() {
    // A follower stopped in recovery must not leave an ack claiming it was streaming: a restart
    // reading that would replay towards a block the previous run had already refused to stand on.
    // No `End` here, so the run stops the way a real one does — still waiting.
    let dir = spool_dir("ack-needs-snapshot");
    write_frame(&dir, 0, FrameKind::Manifest, &StreamEvent::Manifest(manifest()));
    let fixture = fixture();
    let next = write_checkpoint(&dir, 1, &fixture);
    write_frame(
        &dir,
        next,
        FrameKind::Reorg,
        &StreamEvent::Reorg(partial_stateless_stream::Reorg {
            common_ancestor: fixture.checkpoint.block,
            abandoned: vec![partial_stateless_stream::BlockRef {
                number: ANCHOR_BLOCK + 1,
                hash: alloy_primitives::B256::repeat_byte(0xa1),
            }],
            winning_tip: None,
        }),
    );

    run(&dir, false);

    let ack = read_ack(&dir);
    assert_eq!(ack["state"], "needs_snapshot");
    assert_eq!(ack["target_ancestor"], ANCHOR_BLOCK, "and what would end the wait");
    assert_eq!(ack["reason"], "snapshot_required");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_resume_comes_back_to_the_checkpoint_the_ack_names() {
    let dir = spool_dir("resume-same-checkpoint");
    one_epoch(&dir);
    run(&dir, false);

    let resumed = run(&dir, true);

    assert_eq!(resumed.resumed_from, Some(1), "it rebuilt from the recorded checkpoint");
    assert_eq!(resumed.restores, 1);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_resume_with_no_ack_yet_is_a_fresh_run() {
    // Asking to resume before anything has run should produce a run, not an error.
    let dir = spool_dir("resume-no-ack");
    one_epoch(&dir);

    let report = run(&dir, true);

    assert_eq!(report.resumed_from, None);
    assert_eq!(report.restores, 1);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn an_unreadable_ack_is_an_error_rather_than_a_silent_fresh_start() {
    // It was written by something. Starting over would hide from whoever asked to resume that
    // their watermark is gone.
    let dir = spool_dir("resume-corrupt-ack");
    one_epoch(&dir);
    fs::write(ack_path(&dir), "{ not json").expect("writable");

    let options = FollowOptions { ack: Some(ack_path(&dir)), resume: true, ..options() };
    assert!(follow(&dir, &options).is_err());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_v1_ack_replays_the_epoch_rather_than_guessing_a_checkpoint() {
    // The fallback that is deliberately absent. Choosing "the newest checkpoint at or below the
    // watermark" would let a restart install one the previous run had refused, which is a way
    // past a refusal rather than a way back to a pair.
    let dir = spool_dir("resume-v1-ack");
    one_epoch(&dir);
    run(&dir, false);
    let mut ack = read_ack(&dir);
    let object = ack.as_object_mut().expect("an object");
    object.remove("restored_from_sequence");
    object.remove("ack_version");
    fs::write(ack_path(&dir), ack.to_string()).expect("writable");

    let resumed = run(&dir, true);

    assert_eq!(resumed.resumed_from, None, "nothing was adopted on proximity");
    assert_eq!(resumed.restores, 1, "the epoch was read from its start instead");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_resume_will_not_restore_from_a_sequence_that_is_not_a_checkpoint() {
    let dir = spool_dir("resume-not-a-checkpoint");
    one_epoch(&dir);
    run(&dir, false);
    let mut ack = read_ack(&dir);
    // Sequence 0 is the manifest.
    ack["restored_from_sequence"] = serde_json::json!(0);
    fs::write(ack_path(&dir), ack.to_string()).expect("writable");

    let options = FollowOptions { ack: Some(ack_path(&dir)), resume: true, ..options() };
    assert!(follow(&dir, &options).is_err());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_resume_into_an_epoch_this_spool_does_not_have_is_refused() {
    // The ack's numbers were written in a numbering this spool does not have, so nothing in it
    // can be acted on — and starting over quietly would hide that.
    let dir = spool_dir("resume-missing-epoch");
    one_epoch(&dir);
    run(&dir, false);
    let mut ack = read_ack(&dir);
    ack["epoch"] = serde_json::json!(3);
    fs::write(ack_path(&dir), ack.to_string()).expect("writable");

    let options = FollowOptions { ack: Some(ack_path(&dir)), resume: true, ..options() };
    assert!(follow(&dir, &options).is_err());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_resume_across_an_epoch_boundary_verifies_the_chain_to_get_there() {
    let dir = spool_dir("resume-epoch-two");
    write_frame(&dir, 0, FrameKind::Manifest, &StreamEvent::Manifest(manifest()));
    let mut next = write_checkpoint(&dir, 1, &fixture());
    write_frame(&dir, next, FrameKind::End, &end_frame(next, EndKind::Shutdown));
    next += 1;
    let boundary = next;
    write_frame(
        &dir,
        next,
        FrameKind::Manifest,
        &StreamEvent::Manifest(Manifest { epoch: 2, first_sequence: next + 1, ..manifest() }),
    );
    next += 1;
    let second_checkpoint = next;
    next = write_checkpoint(&dir, next, &fixture());
    write_frame(&dir, next, FrameKind::End, &end_frame(next, EndKind::Shutdown));

    // A first run ends at the first epoch's End, so its ack is written in epoch 1.
    run(&dir, false);
    let mut ack = read_ack(&dir);
    // What the second epoch's own run would have left behind.
    ack["epoch"] = serde_json::json!(2);
    ack["restored_from_sequence"] = serde_json::json!(second_checkpoint);
    ack["last_sequence"] = serde_json::json!(second_checkpoint);
    fs::write(ack_path(&dir), ack.to_string()).expect("writable");

    let resumed = run(&dir, true);

    assert_eq!(resumed.resumed_from, Some(second_checkpoint));
    assert!(boundary > 1, "the spool really did hold two epochs");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_broken_epoch_chain_is_refused_rather_than_walked_past() {
    let dir = spool_dir("resume-broken-chain");
    write_frame(&dir, 0, FrameKind::Manifest, &StreamEvent::Manifest(manifest()));
    let mut next = write_checkpoint(&dir, 1, &fixture());
    // A second manifest that is not this stream's successor: right position, wrong chain.
    let mut foreign = Manifest { epoch: 2, first_sequence: next + 1, ..manifest() };
    foreign.chain_id += 1;
    write_frame(&dir, next, FrameKind::Manifest, &StreamEvent::Manifest(foreign));
    next += 1;
    let checkpoint = next;
    next = write_checkpoint(&dir, next, &fixture());
    write_frame(&dir, next, FrameKind::End, &end_frame(next, EndKind::Shutdown));

    fs::write(
        ack_path(&dir),
        serde_json::json!({
            "ack_version": 2,
            "last_sequence": checkpoint,
            "block": ANCHOR_BLOCK,
            "block_hash": format!("{:?}", fixture().checkpoint.block.hash),
            "state": "restored",
            "epoch": 2,
            "restored_from_sequence": checkpoint,
        })
        .to_string(),
    )
    .expect("writable");

    let options = FollowOptions { ack: Some(ack_path(&dir)), resume: true, ..options() };
    assert!(follow(&dir, &options).is_err());
    let _ = fs::remove_dir_all(&dir);
}
