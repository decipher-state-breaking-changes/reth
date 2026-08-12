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

use alloy_primitives::{keccak256, Address, B256, U256};
use alloy_rlp::Encodable;
use partial_stateless::{
    bootstrap::CacheSnapshotPackage,
    network_cache::{CachedEntry, NetworkStateCache},
    policy::{AccountData, LastNBlocksPolicy},
    sidecar::last_n_blocks_cache_policy_id,
};
use partial_stateless_replay::{follow, FollowOptions, FollowOutcome, NeedsSnapshotReason};
use partial_stateless_stream::{
    encode_event, BlockRef, Checkpoint, CommitFrame, CommitInput, CommitOracle, End, EndKind,
    FrameKind, FrameLimits, Manifest, RecordedVerdict, Reset, ResetReason, StreamEvent,
};
use partial_stateless_validator::{
    CoordinatedFingerprint, LifecycleFingerprint, PayloadProvenance,
};
use reth_chainspec::{EthChainSpec, MAINNET};
use reth_primitives_traits::{Account, SealedHeader};
use reth_trie::HashBuilder;
use reth_trie_common::{proof::ProofRetainer, MultiProof, Nibbles};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

const ANCHOR_BLOCK: u64 = 100;
const ACCOUNT_WINDOW: u64 = 64;
const STORAGE_WINDOW: u64 = 32;

fn spool_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ps-follow-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("temp dir");
    dir
}

fn write_frame(dir: &Path, sequence: u64, kind: FrameKind, event: &StreamEvent) {
    let bytes = encode_event(sequence, event, &FrameLimits::default()).expect("encodes");
    fs::write(dir.join(format!("{sequence:012}_{}.frame", kind.as_str())), bytes).expect("write");
}

fn manifest() -> Manifest {
    Manifest {
        chain_id: MAINNET.chain().id(),
        genesis_hash: MAINNET.genesis_hash(),
        cache_policy_id: last_n_blocks_cache_policy_id(ACCOUNT_WINDOW, STORAGE_WINDOW),
        account_window: ACCOUNT_WINDOW,
        storage_window: STORAGE_WINDOW,
        epoch: 1,
        producer: "follow-test".to_string(),
        first_sequence: 1,
    }
}

/// Options that make a test terminate: fast polls and a short idle bound.
fn options() -> FollowOptions {
    FollowOptions {
        poll: Duration::from_millis(5),
        idle_timeout: Some(Duration::from_millis(200)),
        ..Default::default()
    }
}

/// A restorable checkpoint: one account with storage and code, proved against a real state root
/// built by the same `HashBuilder` the export's multiproof provider uses.
struct Fixture {
    checkpoint: Checkpoint,
    package_bytes: Vec<u8>,
}

fn fixture() -> Fixture {
    let address = Address::repeat_byte(0x11);
    let account = Account { nonce: 7, balance: U256::from(1_000u64), bytecode_hash: None };

    // One account, no storage targets: the account subtree alone anchors the proof.
    let address_path = Nibbles::unpack(keccak256(address));
    let mut builder =
        HashBuilder::default().with_proof_retainer(ProofRetainer::from_iter([address_path]));
    builder.add_leaf(
        address_path,
        &alloy_rlp::encode(account.into_trie_account(reth_trie_common::EMPTY_ROOT_HASH)),
    );
    let state_root = builder.root();
    let proof = MultiProof {
        account_subtree: builder.take_proof_nodes(),
        branch_node_masks: Default::default(),
        storages: Default::default(),
    };

    let mut accounts = HashMap::new();
    accounts.insert(
        address,
        CachedEntry {
            value: AccountData { nonce: 7, balance: U256::from(1_000u64), code_hash: None },
            first_accessed_block: 90,
            last_accessed_block: 98,
            access_count: 3,
        },
    );
    let cache = NetworkStateCache::restore(
        accounts,
        HashMap::new(),
        HashMap::new(),
        ANCHOR_BLOCK,
        Box::new(LastNBlocksPolicy::new(ACCOUNT_WINDOW)),
        Box::new(LastNBlocksPolicy::new(STORAGE_WINDOW)),
    );

    // The accepted head is the checkpoint's own block: its hash *is* the checkpoint's hash, and
    // its state root is the proved root, which is exactly what the follower verifies.
    let header = alloy_consensus::Header { number: ANCHOR_BLOCK, state_root, ..Default::default() };
    let sealed = SealedHeader::seal_slow(header.clone());
    let mut accepted_head_rlp = Vec::new();
    header.encode(&mut accepted_head_rlp);

    let policy_id = last_n_blocks_cache_policy_id(ACCOUNT_WINDOW, STORAGE_WINDOW);
    let anchor = cache.cache_anchor(ANCHOR_BLOCK, sealed.hash(), policy_id);
    let package = CacheSnapshotPackage::from_cache(&cache, anchor, &proof);
    let package_bytes = bincode::serialize(&package).expect("package serializes");

    let checkpoint = Checkpoint {
        block: BlockRef { number: ANCHOR_BLOCK, hash: sealed.hash() },
        state_root,
        cache_root: anchor.cache_root,
        cache_policy_id: policy_id,
        accepted_head_rlp,
        snapshot_bytes: 0,
        snapshot_chunks: 0,
        snapshot_digest: B256::ZERO,
    };
    Fixture { checkpoint, package_bytes }
}

/// Writes a checkpoint and its chunks starting at `sequence`; returns the next free sequence.
fn write_checkpoint(dir: &Path, sequence: u64, fixture: &Fixture) -> u64 {
    let mut checkpoint = fixture.checkpoint.clone();
    let chunks = checkpoint.chunk(&fixture.package_bytes, 256);
    write_frame(dir, sequence, FrameKind::Checkpoint, &StreamEvent::Checkpoint(checkpoint));
    let mut next = sequence + 1;
    for chunk in chunks {
        write_frame(dir, next, FrameKind::SnapshotChunk, &StreamEvent::SnapshotChunk(chunk));
        next += 1;
    }
    next
}

/// A commit frame whose *frame-level* fields are real and whose payload is deliberately not:
/// every check these tests exercise runs before the payload is decoded.
fn commit_frame(number: u64, parent_hash: B256) -> StreamEvent {
    let block = BlockRef { number, hash: B256::with_last_byte(number as u8) };
    StreamEvent::Commit(Box::new(CommitFrame::new(
        CommitInput {
            block,
            parent_hash,
            payload_provenance: PayloadProvenance::Absent,
            payload_json: None,
            sidecar: Vec::new(),
        },
        CommitOracle {
            verdict: RecordedVerdict::Accepted,
            state_root: None,
            next_cache_anchor: None,
            expected_miss: None,
            readiness_state: "ready".to_string(),
            readiness_watermark: None,
            durability_watermark: None,
            retained_generation: None,
            coordinated_fingerprint: CoordinatedFingerprint {
                cache_block: number,
                cache_root: B256::ZERO,
                trie_cache_root: B256::ZERO,
                trie_state_root: None,
            },
            lifecycle_fingerprint: LifecycleFingerprint {
                accepted_head: None,
                retained_generation: None,
            },
        },
    )))
}

fn end_frame(sequence: u64, kind: EndKind) -> StreamEvent {
    StreamEvent::End(End { kind, reason: "test".into(), last_sequence: sequence - 1 })
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
fn a_reorg_frame_stops_verdicts() {
    let dir = spool_dir("reorg");
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
    assert_eq!(report.last_needs_snapshot, Some(NeedsSnapshotReason::SnapshotRequired));
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
