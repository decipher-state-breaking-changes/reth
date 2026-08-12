//! The on-disk half of the operator-trusted bootstrap: what is written must restore to the same
//! coordinated generation, and a checkpoint that disagrees with it must not.
//!
//! The provider-backed half — building the package's multiproof and rebuilding a pair from
//! canonical state — is exercised by the live run, because both need a real node. What is testable
//! without one is the artifact format, the trust boundary the checkpoint draws, and the readiness
//! promotion, which is where a silent mismatch would otherwise become a Ready pair holding the
//! wrong values.

use alloy_primitives::{keccak256, Address, Bytes, B256, U256};
use partial_stateless::{
    network_cache::{CachedEntry, NetworkStateCache},
    policy::AccountData,
    readiness::{CacheReadiness, TrustedCheckpoint},
    CacheAnchor, CacheSnapshotPackage,
};
use partial_stateless_exex::{bootstrap_io, CacheConfig};
use reth_primitives_traits::Account;
use reth_trie_common::{proof::ProofRetainer, HashBuilder, MultiProof, Nibbles, StorageMultiProof};
use std::collections::HashMap;

const BLOCK: u64 = 21_000_000;
const BLOCK_HASH: B256 = B256::repeat_byte(0xb1);
const ADDRESS: Address = Address::repeat_byte(0x11);
const SLOT: B256 = B256::repeat_byte(0x01);

#[test]
fn a_written_snapshot_restores_the_same_ready_generation() {
    let dir = scratch_dir("roundtrip");
    let config = CacheConfig::default();
    let (cache, proof, state_root) = warm_fixture(&config);
    let checkpoint = checkpoint_for(&cache, &config, state_root);
    let package = CacheSnapshotPackage::from_cache(&cache, anchor_for(&checkpoint), &proof);

    bootstrap_io::write_snapshot(&dir, &package, &checkpoint).expect("snapshot must be writable");
    let (loaded_package, loaded_checkpoint) = bootstrap_io::load_snapshot(&dir)
        .expect("reading back a snapshot this process just wrote must succeed")
        .expect("the directory holds a package");
    assert_eq!(loaded_checkpoint, checkpoint, "the checkpoint must survive the JSON round trip");

    let restored = bootstrap_io::restore_snapshot(loaded_package, &loaded_checkpoint, &config)
        .expect("an honest snapshot restores");

    assert_eq!(restored.cache.cache_root(), cache.cache_root());
    assert_eq!(restored.cache.current_block(), BLOCK);
    assert_eq!(restored.trie_cache.state_root(), Some(state_root));
    assert!(restored.trie_cache.contains_account_path(&ADDRESS));
    assert!(restored.trie_cache.contains_storage_path(&ADDRESS, &SLOT));
    assert!(matches!(restored.readiness.state(), CacheReadiness::Ready(_)));
    // A snapshot arrives with the window whole rather than replayed, which is exactly why the
    // checkpoint has to vouch for every anchor field instead of only the state root.
    assert_eq!(restored.readiness.replay_depth(), 0);
    assert!(restored.readiness.window_filled());
    assert_eq!(restored.readiness.acknowledgeable_height(), Some((BLOCK, BLOCK_HASH)));
    assert_eq!(restored.ready.anchor, anchor_for(&checkpoint));
}

#[test]
fn a_checkpoint_naming_another_policy_is_refused_before_the_package_is_examined() {
    let dir = scratch_dir("wrong-policy");
    let config = CacheConfig::default();
    let (cache, proof, state_root) = warm_fixture(&config);
    let mut checkpoint = checkpoint_for(&cache, &config, state_root);
    let package = CacheSnapshotPackage::from_cache(&cache, anchor_for(&checkpoint), &proof);
    checkpoint.cache_policy_id = B256::repeat_byte(0x99);

    bootstrap_io::write_snapshot(&dir, &package, &checkpoint).expect("snapshot must be writable");
    let (loaded_package, loaded_checkpoint) =
        bootstrap_io::load_snapshot(&dir).unwrap().expect("the directory holds a package");

    let error = expect_rejection(
        bootstrap_io::restore_snapshot(loaded_package, &loaded_checkpoint, &config),
        "a checkpoint produced under another policy must not promote anything",
    );
    assert!(format!("{error}").contains("names policy"), "unexpected: {error}");
}

#[test]
fn a_checkpoint_vouching_for_a_different_cache_root_is_refused() {
    let config = CacheConfig::default();
    let (cache, proof, state_root) = warm_fixture(&config);
    let mut checkpoint = checkpoint_for(&cache, &config, state_root);
    let package = CacheSnapshotPackage::from_cache(&cache, anchor_for(&checkpoint), &proof);
    // Two caches can reproduce the same canonical state root while holding different values, and
    // peers compare anchors rather than state roots. This is the case the checkpoint exists for.
    checkpoint.cache_root = B256::repeat_byte(0x42);

    let error = expect_rejection(
        bootstrap_io::restore_snapshot(package, &checkpoint, &config),
        "a package that disagrees with the checkpoint must be discarded",
    );
    assert!(format!("{error}").contains("anchor mismatch"), "unexpected: {error}");
}

#[test]
fn a_package_without_its_checkpoint_is_an_error_rather_than_an_absence() {
    let dir = scratch_dir("orphan-package");
    let config = CacheConfig::default();
    let (cache, proof, state_root) = warm_fixture(&config);
    let checkpoint = checkpoint_for(&cache, &config, state_root);
    let package = CacheSnapshotPackage::from_cache(&cache, anchor_for(&checkpoint), &proof);
    bootstrap_io::write_snapshot(&dir, &package, &checkpoint).unwrap();
    std::fs::remove_file(dir.join(bootstrap_io::CHECKPOINT_FILE)).unwrap();

    let error = bootstrap_io::load_snapshot(&dir)
        .expect_err("a package with no checkpoint beside it is a misconfigured directory");
    assert!(format!("{error}").contains("has no checkpoint"), "unexpected: {error}");
}

#[test]
fn an_empty_bootstrap_directory_is_not_an_error() {
    let dir = scratch_dir("empty");
    assert!(bootstrap_io::load_snapshot(&dir).unwrap().is_none());
}

fn cached<T>(value: T) -> CachedEntry<T> {
    CachedEntry { value, first_accessed_block: BLOCK, last_accessed_block: BLOCK, access_count: 1 }
}

/// `RestoredSnapshot` holds a `NetworkStateCache`, which has no `Debug`, so `expect_err` is out.
fn expect_rejection(
    result: Result<bootstrap_io::RestoredSnapshot, partial_stateless::RestoreError>,
    context: &str,
) -> partial_stateless::RestoreError {
    match result {
        Ok(_) => panic!("{context}"),
        Err(error) => error,
    }
}

fn scratch_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir()
        .join("partial-stateless-bootstrap-tests")
        .join(format!("{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch directory must be creatable");
    dir
}

fn anchor_for(checkpoint: &TrustedCheckpoint) -> CacheAnchor {
    CacheAnchor {
        block_number: checkpoint.block_number,
        block_hash: checkpoint.block_hash,
        cache_policy_id: checkpoint.cache_policy_id,
        cache_root: checkpoint.cache_root,
    }
}

fn checkpoint_for(
    cache: &NetworkStateCache,
    config: &CacheConfig,
    state_root: B256,
) -> TrustedCheckpoint {
    TrustedCheckpoint {
        block_number: BLOCK,
        block_hash: BLOCK_HASH,
        state_root,
        cache_root: cache.cache_root(),
        cache_policy_id: config.cache_policy_id(),
    }
}

/// One warm account with a storage slot and its bytecode, plus the multiproof covering both.
fn warm_fixture(config: &CacheConfig) -> (NetworkStateCache, MultiProof, B256) {
    let code = Bytes::from_static(&[0x60, 0x00, 0x60, 0x00, 0xf3]);
    let code_hash = keccak256(&code);
    let account =
        Account { nonce: 7, balance: U256::from(1_000u64), bytecode_hash: Some(code_hash) };
    let value = U256::from(42u64);

    let hashed_slot = keccak256(SLOT);
    let mut storage_builder = HashBuilder::default()
        .with_proof_retainer(ProofRetainer::from_iter([Nibbles::unpack(hashed_slot)]));
    storage_builder.add_leaf(Nibbles::unpack(hashed_slot), &alloy_rlp::encode(value));
    let storage_root = storage_builder.root();
    let storage_subtree = storage_builder.take_proof_nodes();

    let hashed_address = keccak256(ADDRESS);
    let mut account_builder = HashBuilder::default()
        .with_proof_retainer(ProofRetainer::from_iter([Nibbles::unpack(hashed_address)]));
    account_builder.add_leaf(
        Nibbles::unpack(hashed_address),
        &alloy_rlp::encode(account.into_trie_account(storage_root)),
    );
    let state_root = account_builder.root();

    let mut storages = alloy_primitives::map::B256Map::default();
    storages.insert(
        hashed_address,
        StorageMultiProof {
            root: storage_root,
            subtree: storage_subtree,
            branch_node_masks: Default::default(),
        },
    );
    let proof = MultiProof {
        account_subtree: account_builder.take_proof_nodes(),
        branch_node_masks: Default::default(),
        storages,
    };

    let mut accounts = HashMap::new();
    accounts.insert(
        ADDRESS,
        cached(AccountData {
            nonce: account.nonce,
            balance: account.balance,
            code_hash: account.bytecode_hash,
        }),
    );
    let mut storage = HashMap::new();
    storage.insert((ADDRESS, SLOT), cached(value));
    let mut codes = HashMap::new();
    codes.insert(code_hash, cached(code));

    let cache = NetworkStateCache::restore(
        accounts,
        storage,
        codes,
        BLOCK,
        config.account_policy(),
        config.storage_policy(),
    );
    (cache, proof, state_root)
}
