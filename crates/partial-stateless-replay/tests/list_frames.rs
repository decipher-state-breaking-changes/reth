mod common;

use alloy_primitives::B256;
use common::{
    commit_frame, end_frame, fixture, manifest, spool_dir, write_checkpoint, write_frame,
};
use partial_stateless_stream::{BlockRef, EndKind, FrameKind, Reorg, StreamEvent};
use std::{fs, process::Command};

#[test]
fn listing_reads_content_across_epochs_without_executing_payloads() {
    let dir = spool_dir("list-frames-epochs");
    let fixture = fixture();
    write_frame(&dir, 0, FrameKind::Manifest, &StreamEvent::Manifest(manifest()));
    let mut next = write_checkpoint(&dir, 1, &fixture);
    let commit_sequence = next;
    write_frame(
        &dir,
        next,
        FrameKind::Commit,
        &commit_frame(fixture.checkpoint.block.number + 1, fixture.checkpoint.block.hash),
    );
    next += 1;
    let abandoned = vec![BlockRef {
        number: fixture.checkpoint.block.number + 1,
        hash: B256::repeat_byte(0xaa),
    }];
    let reorg_sequence = next;
    write_frame(
        &dir,
        next,
        FrameKind::Reorg,
        &StreamEvent::Reorg(Reorg {
            common_ancestor: fixture.checkpoint.block,
            abandoned: abandoned.clone(),
            winning_tip: None,
        }),
    );
    next += 1;
    write_frame(&dir, next, FrameKind::End, &end_frame(next, EndKind::Shutdown));
    next += 1;
    let mut second = manifest();
    second.epoch = 2;
    second.first_sequence = next + 1;
    write_frame(&dir, next, FrameKind::Manifest, &StreamEvent::Manifest(second));
    next += 1;
    let second_checkpoint = next;
    next = write_checkpoint(&dir, next, &fixture);
    write_frame(&dir, next, FrameKind::End, &end_frame(next, EndKind::Shutdown));

    let output = Command::new(env!("CARGO_BIN_EXE_ps-replay"))
        .arg("--list-frames")
        .arg(&dir)
        // Inspection is independent of replay configuration, including invalid environment values.
        .env("PS_RETAIN_DEPTH", "invalid")
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let rows: Vec<serde_json::Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).expect("one JSON object per frame"))
        .collect();
    assert_eq!(rows.len(), next as usize + 1);
    for (sequence, row) in rows.iter().enumerate() {
        assert_eq!(row["sequence"], sequence);
    }
    assert_eq!(rows[1]["kind"], "checkpoint");
    assert_eq!(rows[0]["epoch"], 1);
    assert_eq!(rows[(second_checkpoint - 1) as usize]["epoch"], 2);
    assert_eq!(rows[1]["snapshot_chunks"], commit_sequence - 2);
    assert_eq!(rows[second_checkpoint as usize]["snapshot_chunks"], rows[1]["snapshot_chunks"]);
    assert_eq!(rows[1]["block"], serde_json::to_value(fixture.checkpoint.block).unwrap());
    assert_eq!(rows[second_checkpoint as usize]["block"], rows[1]["block"]);
    assert_eq!(rows[commit_sequence as usize]["kind"], "commit");
    assert_eq!(
        rows[commit_sequence as usize]["block"]["number"],
        fixture.checkpoint.block.number + 1
    );
    assert_eq!(
        rows[commit_sequence as usize]["parent_hash"],
        serde_json::to_value(fixture.checkpoint.block.hash).unwrap()
    );
    assert_eq!(rows[reorg_sequence as usize]["ancestor"], rows[1]["block"]);
    assert_eq!(
        rows[reorg_sequence as usize]["abandoned"],
        serde_json::to_value(abandoned).unwrap()
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn listing_refuses_a_gap_instead_of_omitting_it() {
    let dir = spool_dir("list-frames-gap");
    write_frame(&dir, 0, FrameKind::Manifest, &StreamEvent::Manifest(manifest()));
    write_frame(&dir, 2, FrameKind::End, &end_frame(2, EndKind::Shutdown));
    let output = Command::new(env!("CARGO_BIN_EXE_ps-replay"))
        .arg("--list-frames")
        .arg(&dir)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("not contiguous"));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn listing_refuses_conflicting_replay_arguments() {
    let output = Command::new(env!("CARGO_BIN_EXE_ps-replay"))
        .args(["--list-frames", "/unused", "--follow"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("usage: ps-replay --list-frames"));
}
