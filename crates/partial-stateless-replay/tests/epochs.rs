//! What a replay does with a spool a producer restarted into.
//!
//! One directory is one sequence space — frame files are named by it — so a producer that resumes
//! does not restart its numbering. What it does restart is its *state*, and the manifest is where
//! it says so. The whole point of the epoch field is that continuity of numbering must not be read
//! as continuity of state: everything below a new epoch is a stream this driver was reading, and
//! everything above it rebootstraps.

mod common;

use common::{end_frame, fixture, fixture_at, manifest, spool_dir, write_checkpoint, write_frame};
use partial_stateless_replay::{replay, ReplayOptions};
use partial_stateless_stream::{EndKind, FrameKind, Manifest, StreamEvent};
use std::path::Path;

fn options() -> ReplayOptions {
    ReplayOptions { mutations: false, ..Default::default() }
}

/// The manifest a resumed producer writes at `sequence`.
fn next_epoch(sequence: u64) -> Manifest {
    Manifest { epoch: 2, first_sequence: sequence + 1, ..manifest() }
}

fn write_manifest(dir: &Path, sequence: u64, manifest: Manifest) {
    write_frame(dir, sequence, FrameKind::Manifest, &StreamEvent::Manifest(manifest));
}

#[test]
fn a_closed_epoch_and_the_one_that_follows_are_both_replayed() {
    let dir = spool_dir("epoch-closed");
    let first = fixture();
    let second = fixture_at(common::ANCHOR_BLOCK + 40);
    write_manifest(&dir, 0, manifest());
    let mut next = write_checkpoint(&dir, 1, &first);
    write_frame(&dir, next, FrameKind::End, &end_frame(next, EndKind::Shutdown));
    next += 1;
    write_manifest(&dir, next, next_epoch(next));
    next += 1;
    next = write_checkpoint(&dir, next, &second);
    write_frame(&dir, next, FrameKind::End, &end_frame(next, EndKind::Shutdown));

    let report = replay(&dir, &options()).expect("both epochs read");

    assert_eq!(report.epoch_transitions, 1, "one boundary was crossed between two epochs");
    assert_eq!(report.resyncs.len(), 1, "and the second epoch's checkpoint rebootstrapped");
    assert!(!report.resyncs[0].continuous, "a restart is never a continuous recovery");
    assert!(report.agreed(), "a producer restart is not a disagreement");
    assert!(!report.continuous());
    assert!(report.complete(), "and the corpus was read to its end");
    assert!(report.closed);
}

/// A producer killed rather than stopped leaves no `End`. The next epoch's manifest is still the
/// boundary, and the frames below it are still the ones that were verified.
#[test]
fn an_epoch_cut_without_an_end_is_still_a_boundary() {
    let dir = spool_dir("epoch-cut");
    let first = fixture();
    let second = fixture_at(common::ANCHOR_BLOCK + 40);
    write_manifest(&dir, 0, manifest());
    let mut next = write_checkpoint(&dir, 1, &first);
    write_manifest(&dir, next, next_epoch(next));
    next += 1;
    next = write_checkpoint(&dir, next, &second);
    write_frame(&dir, next, FrameKind::End, &end_frame(next, EndKind::Shutdown));

    let report = replay(&dir, &options()).expect("both epochs read");

    assert_eq!(report.epoch_transitions, 1);
    assert_eq!(report.resyncs.len(), 1);
    assert!(report.agreed() && !report.continuous() && report.complete());
}

/// The check that makes the boundary mean anything: a manifest that is not this stream's
/// successor is a different stream sharing a directory, and nothing under it may be restored as
/// though it continued this one.
#[test]
fn a_manifest_that_is_not_the_next_epoch_stops_the_replay() {
    let dir = spool_dir("epoch-foreign");
    let first = fixture();
    write_manifest(&dir, 0, manifest());
    let next = write_checkpoint(&dir, 1, &first);
    let mut foreign = next_epoch(next);
    foreign.chain_id += 1;
    write_manifest(&dir, next, foreign);

    assert!(
        replay(&dir, &options()).is_err(),
        "a corpus that holds two chains is not one to report agreement on"
    );
}

/// Numbering is checked against position, so a manifest lifted out of another spool is caught even
/// when it names the right chain and the right epoch.
#[test]
fn an_epoch_manifest_that_disagrees_with_its_own_position_is_refused() {
    let dir = spool_dir("epoch-misplaced");
    let first = fixture();
    write_manifest(&dir, 0, manifest());
    let next = write_checkpoint(&dir, 1, &first);
    let mut misplaced = next_epoch(next);
    misplaced.first_sequence += 3;
    write_manifest(&dir, next, misplaced);

    assert!(replay(&dir, &options()).is_err());
}

/// Everything else past an `End` is still a trailing frame on a closed stream.
#[test]
fn a_spool_that_continues_past_its_end_with_anything_else_is_refused() {
    let dir = spool_dir("epoch-trailing");
    let first = fixture();
    write_manifest(&dir, 0, manifest());
    let mut next = write_checkpoint(&dir, 1, &first);
    write_frame(&dir, next, FrameKind::End, &end_frame(next, EndKind::Shutdown));
    next += 1;
    write_checkpoint(&dir, next, &first);

    assert!(replay(&dir, &options()).is_err());
}

/// Writes a two-epoch spool to `$PS_GATE_SPOOL` and leaves it there.
///
/// Ignored by default: it exists so the live gate script's `GATE_MODE=epoch` arm can be exercised
/// against a corpus without a node, which is the only way to find out that a *script* is wrong
/// before a run that takes hours does it for you.
#[test]
#[ignore = "harness fixture: writes a spool for the gate script"]
fn emit_a_two_epoch_spool_for_the_gate_script() {
    let dir = std::path::PathBuf::from(
        std::env::var("PS_GATE_SPOOL").expect("set PS_GATE_SPOOL to the directory to write"),
    );
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("creatable");
    let first = fixture();
    let second = fixture_at(common::ANCHOR_BLOCK + 40);
    write_manifest(&dir, 0, manifest());
    let mut next = write_checkpoint(&dir, 1, &first);
    write_frame(&dir, next, FrameKind::End, &end_frame(next, EndKind::Shutdown));
    next += 1;
    write_manifest(&dir, next, next_epoch(next));
    next += 1;
    next = write_checkpoint(&dir, next, &second);
    write_frame(&dir, next, FrameKind::End, &end_frame(next, EndKind::Shutdown));
    println!("wrote a two-epoch spool to {}", dir.display());
}
