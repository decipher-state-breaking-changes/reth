//! What the v1 format promises, checked against its own bytes.
//!
//! These live outside the crate on purpose: they use only the public API, so a test that keeps
//! passing is evidence that a *consumer* can still read what a *producer* wrote. A test with
//! access to private fields could keep passing across a change that breaks every real reader.

use alloy_primitives::{b256, keccak256, Address, B256, U256};
use alloy_rpc_types_engine::{ExecutionData, ExecutionPayload};
use partial_stateless::{CacheAnchor, StateTargetSet};
use partial_stateless_stream::{
    decode_event, encode_event, event::SnapshotError, BlockRef, Checkpoint, CommitFrame,
    CommitInput, CommitOracle, End, EndKind, FrameKind, FrameLimits, Manifest, RecordedVerdict,
    Reorg, Reset, ResetReason, SnapshotChunk, StreamEvent, DEFAULT_MAX_SNAPSHOT_BYTES,
    FRAME_HEADER_BYTES, MAX_SNAPSHOT_CHUNKS,
};
use partial_stateless_validator::{
    CoordinatedFingerprint, LifecycleFingerprint, PayloadProvenance,
};

fn block(number: u64, tag: u8) -> BlockRef {
    BlockRef { number, hash: B256::with_last_byte(tag) }
}

fn anchor() -> CacheAnchor {
    CacheAnchor {
        block_number: 25_737_234,
        block_hash: B256::with_last_byte(0x11),
        cache_policy_id: B256::with_last_byte(0x22),
        cache_root: B256::with_last_byte(0x33),
    }
}

fn oracle() -> CommitOracle {
    CommitOracle {
        verdict: RecordedVerdict::Accepted,
        state_root: Some(B256::with_last_byte(0x44)),
        next_cache_anchor: Some(anchor()),
        expected_miss: Some(StateTargetSet {
            accounts: vec![Address::with_last_byte(1)],
            storage: vec![(Address::with_last_byte(2), B256::with_last_byte(3))],
            code_hashes: vec![B256::with_last_byte(4)],
        }),
        readiness_state: "ready".to_string(),
        readiness_watermark: Some(block(25_737_234, 0x11)),
        durability_watermark: Some(25_737_230),
        retained_generation: Some(block(25_737_233, 0x10)),
        coordinated_fingerprint: CoordinatedFingerprint {
            cache_block: 25_737_234,
            cache_root: B256::with_last_byte(0x33),
            trie_cache_root: B256::with_last_byte(0x55),
            trie_state_root: Some(B256::with_last_byte(0x44)),
        },
        lifecycle_fingerprint: LifecycleFingerprint {
            accepted_head: Some((25_737_234, B256::with_last_byte(0x11))),
            retained_generation: Some((25_737_233, B256::with_last_byte(0x10))),
        },
    }
}

/// A minimal but real Engine payload, so the JSON path is exercised by the type a node receives
/// rather than by a stand-in that happens to serialize.
fn execution_data() -> ExecutionData {
    let block = alloy_consensus::Block::<alloy_consensus::TxEnvelope> {
        header: alloy_consensus::Header {
            number: 25_737_235,
            gas_limit: 60_000_000,
            base_fee_per_gas: Some(1_000_000_000),
            difficulty: U256::ZERO,
            ..Default::default()
        },
        body: alloy_consensus::BlockBody::default(),
    };
    let (payload, sidecar) = ExecutionPayload::from_block_slow(&block);
    ExecutionData::new(payload, sidecar)
}

fn commit(provenance: PayloadProvenance, payload_json: Option<Vec<u8>>) -> CommitFrame {
    CommitFrame::new(
        CommitInput {
            block: block(25_737_235, 0x12),
            parent_hash: B256::with_last_byte(0x11),
            payload_provenance: provenance,
            payload_json,
            sidecar: vec![7; 128],
        },
        oracle(),
    )
}

fn round_trip(sequence: u64, event: StreamEvent) -> StreamEvent {
    let encoded = encode_event(sequence, &event, &FrameLimits::default()).expect("encodes");
    let (header, decoded, rest) = decode_event(&encoded, &FrameLimits::default()).expect("decodes");
    assert_eq!(header.sequence, sequence);
    assert!(rest.is_empty());
    assert_eq!(header.payload_digest, keccak256(&encoded[FRAME_HEADER_BYTES..]));
    decoded
}

/// All seven kinds are in v1, including the four the first executable replay will ignore. Adding
/// them later would force the live follower and its reorg recovery to migrate a spool format
/// exactly when the lifecycle work is hardest to reason about.
#[test]
fn every_v1_event_survives_its_own_encoding() {
    let manifest = Manifest {
        chain_id: 1,
        genesis_hash: b256!("d4e56740f876aef8c010b86a40d5f56745a118d0906a34e69aec8c0db1cb8fa3"),
        cache_policy_id: B256::with_last_byte(0x22),
        account_window: 60,
        storage_window: 30,
        epoch: 1,
        producer: "reth-partial-stateless 2.2.0 (3346b01)".to_string(),
        first_sequence: 1,
    };
    assert_eq!(
        round_trip(0, StreamEvent::Manifest(manifest.clone())),
        StreamEvent::Manifest(manifest)
    );

    let mut checkpoint = Checkpoint {
        block: block(25_737_234, 0x11),
        state_root: B256::with_last_byte(0x44),
        cache_root: B256::with_last_byte(0x33),
        cache_policy_id: B256::with_last_byte(0x22),
        accepted_head_rlp: vec![0xf9, 0x02, 0x1a],
        snapshot_bytes: 0,
        snapshot_chunks: 0,
        snapshot_digest: B256::ZERO,
    };
    let chunks = checkpoint.chunk(&[9u8; 300], 128);
    assert_eq!(
        round_trip(1, StreamEvent::Checkpoint(checkpoint.clone())),
        StreamEvent::Checkpoint(checkpoint)
    );
    assert_eq!(
        round_trip(2, StreamEvent::SnapshotChunk(chunks[0].clone())),
        StreamEvent::SnapshotChunk(chunks[0].clone())
    );

    let commit = commit(PayloadProvenance::Witnessed, Some(b"{}".to_vec()));
    assert_eq!(
        round_trip(3, StreamEvent::Commit(Box::new(commit.clone()))),
        StreamEvent::Commit(Box::new(commit))
    );

    let reorg = Reorg {
        common_ancestor: block(25_737_233, 0x10),
        abandoned: vec![block(25_737_234, 0x11)],
        winning_tip: Some(block(25_737_235, 0x99)),
    };
    assert_eq!(round_trip(4, StreamEvent::Reorg(reorg.clone())), StreamEvent::Reorg(reorg));

    let reset = Reset { reason: ResetReason::Gap, detail: "block 25737236 never arrived".into() };
    assert_eq!(round_trip(5, StreamEvent::Reset(reset.clone())), StreamEvent::Reset(reset));

    let end = End { kind: EndKind::Shutdown, reason: "producer shutdown".into(), last_sequence: 5 };
    assert_eq!(round_trip(6, StreamEvent::End(end.clone())), StreamEvent::End(end));
}

/// The recorded payload is Engine-API JSON, so a replay driver's decode is the decode a live node
/// performs. A codec only this crate could read would make `input_decode_us` meaningless.
#[test]
fn a_recorded_payload_parses_with_the_deserializer_a_node_uses() {
    let expected = execution_data();
    let json = serde_json::to_vec(&expected).expect("payload serializes");

    let commit = commit(PayloadProvenance::Witnessed, Some(json));
    let StreamEvent::Commit(decoded) = round_trip(1, StreamEvent::Commit(Box::new(commit))) else {
        panic!("a commit frame decodes as a commit")
    };

    let payload = decoded.input().payload().expect("payload parses").expect("payload is present");
    assert_eq!(payload.payload.block_hash(), expected.payload.block_hash());
    assert_eq!(payload.payload.block_number(), expected.payload.block_number());
}

/// A commit with no payload is a legitimate record and not a broken one, so parsing it must
/// distinguish "absent" from "malformed".
#[test]
fn a_commit_without_a_payload_reads_as_absent_rather_than_as_a_parse_failure() {
    let commit = commit(PayloadProvenance::Absent, None);
    let StreamEvent::Commit(decoded) = round_trip(1, StreamEvent::Commit(Box::new(commit))) else {
        panic!("a commit frame decodes as a commit")
    };
    assert_eq!(decoded.input().payload_provenance, PayloadProvenance::Absent);
    assert!(decoded.input().payload().expect("no parse error").is_none());
    assert!(!decoded.input().payload_provenance.is_load_bearing());
}

/// Reaching the oracle costs the caller an explicit `split`, and what comes back is two values
/// rather than one. The compile-time half of this guarantee is the dependency arrow — the
/// validator crate cannot name `CommitOracle` — which no test can observe.
#[test]
fn the_expectation_is_only_reachable_by_separating_it_from_the_input() {
    let (input, oracle) = commit(PayloadProvenance::Witnessed, None).split();
    assert_eq!(input.block.number, 25_737_235);
    assert!(oracle.verdict.is_accepted());
    assert_eq!(oracle.verdict.label(), "accepted");
    assert_eq!(RecordedVerdict::Rejected { class: "consensus".into() }.label(), "consensus");
}

/// Reassembly is the exact inverse of chunking, and every way a delivery can go wrong is named
/// rather than collapsed into one "corrupt snapshot".
#[test]
fn a_chunked_snapshot_reassembles_and_says_how_a_broken_one_differs() {
    let package: Vec<u8> = (0..1_000u32).map(|byte| byte as u8).collect();
    let mut checkpoint = Checkpoint {
        block: block(25_737_234, 0x11),
        state_root: B256::with_last_byte(0x44),
        cache_root: B256::with_last_byte(0x33),
        cache_policy_id: B256::with_last_byte(0x22),
        accepted_head_rlp: Vec::new(),
        snapshot_bytes: 0,
        snapshot_chunks: 0,
        snapshot_digest: B256::ZERO,
    };
    let chunks = checkpoint.chunk(&package, 256);

    assert_eq!(checkpoint.snapshot_chunks, 4);
    assert_eq!(checkpoint.snapshot_bytes, 1_000);
    assert_eq!(checkpoint.reassemble(&chunks).expect("reassembles"), package);

    assert_eq!(
        checkpoint.reassemble(&chunks[..3]),
        Err(SnapshotError::ChunkCount { expected: 4, actual: 3 })
    );

    let mut reordered = chunks.clone();
    reordered.swap(1, 2);
    assert_eq!(
        checkpoint.reassemble(&reordered),
        Err(SnapshotError::OutOfOrder { expected: 1, actual: 2 })
    );

    let mut short = chunks.clone();
    short[3] = SnapshotChunk { index: 3, bytes: Vec::new() };
    assert_eq!(
        checkpoint.reassemble(&short),
        Err(SnapshotError::Length { expected: 1_000, actual: 768 })
    );

    let mut corrupt = chunks;
    corrupt[2].bytes[0] ^= 0xff;
    assert!(matches!(checkpoint.reassemble(&corrupt), Err(SnapshotError::Digest { .. })));
}

/// The frame kind a body was written under travels with it, so a consumer never has to guess what
/// it is holding from the shape of the bytes.
#[test]
fn a_decoded_frame_names_its_own_kind() {
    let end = StreamEvent::End(End {
        kind: EndKind::SpoolLimit,
        reason: "done".into(),
        last_sequence: 9,
    });
    let encoded = encode_event(10, &end, &FrameLimits::default()).expect("encodes");
    let (header, decoded, _) = decode_event(&encoded, &FrameLimits::default()).expect("decodes");
    assert_eq!(header.kind, FrameKind::End);
    assert_eq!(header.kind.as_str(), "end");
    let StreamEvent::End(decoded) = decoded else { panic!("an end frame decodes as an end") };
    assert_eq!(decoded.kind, EndKind::SpoolLimit);
    assert_eq!(decoded.kind.as_str(), "spool_limit");
}

/// The declaration is checked before it can size anything. A checkpoint is operator-trusted for
/// what it *attests*, not for how much memory its transport fields may claim.
#[test]
fn a_snapshot_declaration_past_the_bound_is_refused_before_reassembly() {
    let oversized = Checkpoint {
        block: block(25_737_234, 0x11),
        state_root: B256::with_last_byte(0x44),
        cache_root: B256::with_last_byte(0x33),
        cache_policy_id: B256::with_last_byte(0x22),
        accepted_head_rlp: Vec::new(),
        snapshot_bytes: DEFAULT_MAX_SNAPSHOT_BYTES + 1,
        snapshot_chunks: 1,
        snapshot_digest: B256::ZERO,
    };
    assert_eq!(
        oversized.reassemble(&[]),
        Err(SnapshotError::DeclaredTooLarge {
            declared: DEFAULT_MAX_SNAPSHOT_BYTES + 1,
            limit: DEFAULT_MAX_SNAPSHOT_BYTES,
        })
    );

    let too_many =
        Checkpoint { snapshot_bytes: 1, snapshot_chunks: MAX_SNAPSHOT_CHUNKS + 1, ..oversized };
    assert_eq!(
        too_many.validate_declared(DEFAULT_MAX_SNAPSHOT_BYTES),
        Err(SnapshotError::DeclaredTooManyChunks {
            declared: MAX_SNAPSHOT_CHUNKS + 1,
            limit: MAX_SNAPSHOT_CHUNKS,
        })
    );
}
