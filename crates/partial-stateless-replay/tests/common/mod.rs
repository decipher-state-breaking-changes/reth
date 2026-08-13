//! The synthetic-spool fixture both integration suites are written against.
//!
//! The restorable checkpoint is a real one: a one-account state proved with the same trie
//! machinery the producer's export uses, so `restore` runs the full verification path rather than
//! a stub. What no synthetic spool can supply is a commit that passes mainnet admission — that is
//! the live gate's job — so the tests here exercise the checks that run *before* admission, which
//! is exactly where the delivery grammar and the reorg lifecycle live.

#![allow(dead_code)]

use alloy_primitives::{keccak256, Address, B256, U256};
use alloy_rlp::Encodable;
use partial_stateless::{
    bootstrap::CacheSnapshotPackage,
    network_cache::{CachedEntry, NetworkStateCache},
    policy::{AccountData, LastNBlocksPolicy},
    sidecar::last_n_blocks_cache_policy_id,
};
use partial_stateless_replay::FollowOptions;
use partial_stateless_stream::{
    encode_event, BlockRef, Checkpoint, CommitFrame, CommitInput, CommitOracle, End, EndKind,
    FrameKind, FrameLimits, Manifest, RecordedVerdict, StreamEvent,
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

pub const ANCHOR_BLOCK: u64 = 100;
pub const ACCOUNT_WINDOW: u64 = 64;
pub const STORAGE_WINDOW: u64 = 32;

pub fn spool_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ps-follow-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("temp dir");
    dir
}

pub fn write_frame(dir: &Path, sequence: u64, kind: FrameKind, event: &StreamEvent) {
    let bytes = encode_event(sequence, event, &FrameLimits::default()).expect("encodes");
    fs::write(dir.join(format!("{sequence:012}_{}.frame", kind.as_str())), bytes).expect("write");
}

pub fn manifest() -> Manifest {
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
pub fn options() -> FollowOptions {
    FollowOptions {
        poll: Duration::from_millis(5),
        idle_timeout: Some(Duration::from_millis(200)),
        ..Default::default()
    }
}

/// A restorable checkpoint: one account with storage and code, proved against a real state root
/// built by the same `HashBuilder` the export's multiproof provider uses.
pub struct Fixture {
    pub checkpoint: Checkpoint,
    pub package_bytes: Vec<u8>,
}

pub fn fixture() -> Fixture {
    fixture_at(ANCHOR_BLOCK)
}

/// The same fixture at any height, so a test can hold two checkpoints that are not the same block.
pub fn fixture_at(anchor_block: u64) -> Fixture {
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
        anchor_block,
        Box::new(LastNBlocksPolicy::new(ACCOUNT_WINDOW)),
        Box::new(LastNBlocksPolicy::new(STORAGE_WINDOW)),
    );

    // The accepted head is the checkpoint's own block: its hash *is* the checkpoint's hash, and
    // its state root is the proved root, which is exactly what the follower verifies.
    let header = alloy_consensus::Header { number: anchor_block, state_root, ..Default::default() };
    let sealed = SealedHeader::seal_slow(header.clone());
    let mut accepted_head_rlp = Vec::new();
    header.encode(&mut accepted_head_rlp);

    let policy_id = last_n_blocks_cache_policy_id(ACCOUNT_WINDOW, STORAGE_WINDOW);
    let cache_anchor = cache.cache_anchor(anchor_block, sealed.hash(), policy_id);
    let package = CacheSnapshotPackage::from_cache(&cache, cache_anchor, &proof);
    let package_bytes = bincode::serialize(&package).expect("package serializes");

    let checkpoint = Checkpoint {
        block: BlockRef { number: anchor_block, hash: sealed.hash() },
        state_root,
        cache_root: cache_anchor.cache_root,
        cache_policy_id: policy_id,
        accepted_head_rlp,
        snapshot_bytes: 0,
        snapshot_chunks: 0,
        snapshot_digest: B256::ZERO,
    };
    Fixture { checkpoint, package_bytes }
}

/// Writes a checkpoint and its chunks starting at `sequence`; returns the next free sequence.
pub fn write_checkpoint(dir: &Path, sequence: u64, fixture: &Fixture) -> u64 {
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
pub fn commit_frame(number: u64, parent_hash: B256) -> StreamEvent {
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

pub fn end_frame(sequence: u64, kind: EndKind) -> StreamEvent {
    StreamEvent::End(End { kind, reason: "test".into(), last_sequence: sequence - 1 })
}
