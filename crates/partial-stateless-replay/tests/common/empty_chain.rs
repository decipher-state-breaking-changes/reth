//! Executable empty blocks under mainnet's Paris rules, before Shanghai system calls.

use super::*;
use alloy_consensus::{
    constants::EMPTY_OMMER_ROOT_HASH, Block, BlockBody, Header, EMPTY_ROOT_HASH,
};
use alloy_rpc_types_engine::{ExecutionData, ExecutionPayload, ExecutionPayloadSidecar};
use partial_stateless::{
    restore_snapshot, sidecar::partial_witness_commitment, BlockAccessedState, CacheConfig,
    PartialExecutionWitness, PartialExecutionWitnessState, PartialStatelessSidecar, StateTargetSet,
    TrustedCheckpoint, WitnessResult, WitnessTargets,
};
use partial_stateless_validator::{
    admit_block, block_context, verify_and_apply_sidecar, BlockAdmission, CoordinatedPair,
    RetentionDepth, TrieCacheDisposition, UntrustedAdmission, ValidatorRules,
};
use reth_ethereum_consensus::EthBeaconConsensus;
use reth_ethereum_primitives::TransactionSigned;
use reth_evm_ethereum::EthEvmConfig;

pub fn chain(count: u64) -> (Fixture, Vec<StreamEvent>) {
    chain_with_tag(count, 0)
}

pub fn chain_with_tag(count: u64, tag: u8) -> (Fixture, Vec<StreamEvent>) {
    let mut parent = Header {
        number: 15_600_000,
        timestamp: 1_665_000_000,
        gas_limit: 30_000_000,
        base_fee_per_gas: Some(1_000_000_000),
        ommers_hash: EMPTY_OMMER_ROOT_HASH,
        transactions_root: EMPTY_ROOT_HASH,
        receipts_root: EMPTY_ROOT_HASH,
        ..Default::default()
    };
    let fixture = super::fixture_with_header(parent.clone());
    parent.state_root = fixture.checkpoint.state_root;
    let package: CacheSnapshotPackage = bincode::deserialize(&fixture.package_bytes).unwrap();
    let proof_bytes = bincode::serialize(&package.proof).unwrap();
    let config = CacheConfig { account_window: ACCOUNT_WINDOW, storage_window: STORAGE_WINDOW };
    let policy = config.cache_policy_id();
    let trusted = TrustedCheckpoint {
        block_number: parent.number,
        block_hash: fixture.checkpoint.block.hash,
        state_root: parent.state_root,
        cache_root: fixture.checkpoint.cache_root,
        cache_policy_id: policy,
    };
    let restored = restore_snapshot(package, &trusted, &config).unwrap();
    let mut pair = CoordinatedPair {
        undo_store: None,
        cache: restored.cache,
        trie_cache: restored.trie_cache,
        readiness: restored.readiness,
        accepted_head: Some(SealedHeader::seal_slow(parent.clone())),
        retained: Default::default(),
        retention_depth: RetentionDepth::new(3).unwrap(),
        undo_layout: Default::default(),
    };
    let consensus = EthBeaconConsensus::new(MAINNET.clone());
    let evm = EthEvmConfig::new(MAINNET.clone());
    let mut commits = Vec::new();
    for _ in 0..count {
        let parent_hash = parent.hash_slow();
        let timestamp = parent.timestamp + 12;
        let header = Header {
            number: parent.number + 1,
            parent_hash,
            timestamp,
            extra_data: vec![tag].into(),
            base_fee_per_gas: parent
                .next_block_base_fee(MAINNET.base_fee_params_at_timestamp(timestamp)),
            ..parent.clone()
        };
        let block: Block<TransactionSigned> =
            Block { header: header.clone(), body: BlockBody::default() };
        let block_ref = BlockRef { number: header.number, hash: header.hash_slow() };
        let payload = ExecutionData {
            payload: ExecutionPayload::from_block_slow(&block).0,
            sidecar: ExecutionPayloadSidecar::none(),
        };
        // Empty Paris blocks have no rewards, withdrawals, transactions, or system calls.
        let prev_cache_anchor = pair.cache.cache_anchor(parent.number, parent_hash, policy);
        let mut next_cache = NetworkStateCache::restore(
            pair.cache.accounts().clone(),
            pair.cache.storage().clone(),
            pair.cache.codes().clone(),
            parent.number,
            Box::new(LastNBlocksPolicy::new(ACCOUNT_WINDOW)),
            Box::new(LastNBlocksPolicy::new(STORAGE_WINDOW)),
        );
        next_cache.on_block_executed(header.number, &BlockAccessedState::default());
        let next_cache_anchor = next_cache.cache_anchor(header.number, block_ref.hash, policy);
        let targets = StateTargetSet::default();
        let witness = PartialExecutionWitness {
            state: PartialExecutionWitnessState::MptMultiProof(proof_bytes.clone()),
            codes: vec![],
            keys: vec![],
            headers: vec![],
        };
        let sidecar = PartialStatelessSidecar {
            parent_hash,
            parent_state_root: parent.state_root,
            block_hash: block_ref.hash,
            block_number: header.number,
            cache_block: parent.number,
            cache_policy_id: policy,
            prev_cache_anchor,
            next_cache_anchor,
            cache_policy_metadata: "empty-chain fixture".into(),
            witness_commitment: partial_witness_commitment(parent.state_root, &targets, &witness),
            cache_miss_targets: targets,
            miss_manifest: WitnessTargets {
                missed_accounts: vec![],
                missed_storage: vec![],
                missed_code_hashes: vec![],
            },
            witness,
            stats: WitnessResult {
                total_size_bytes: 0,
                account_proof_bytes: 0,
                storage_proof_bytes: 0,
                bytecode_bytes: 0,
                account_proof_nodes: 0,
                storage_proof_nodes: 0,
                target_accounts: 0,
                target_storage_slots: 0,
                computation_time_ms: None,
                cpu_time_ms: None,
                major_page_faults: None,
                minor_page_faults: None,
            },
        };
        let admitted = UntrustedAdmission::new(MAINNET.as_ref(), &consensus)
            .admit(payload.clone(), pair.accepted_parent())
            .expect("fixture passes admission");
        let context = block_context(&admitted.block);
        assert!(matches!(admit_block(&mut pair.readiness, &context), BlockAdmission::Admitted(_)));
        let mut validated = verify_and_apply_sidecar(
            ValidatorRules::new(&evm, &consensus),
            &admitted.block,
            &mut pair.cache,
            &sidecar,
            policy,
            &Default::default(),
            &mut pair.trie_cache,
            TrieCacheDisposition::Commit,
        )
        .expect("fixture executes and verifies its state root");
        pair.commit_transition(
            validated.outcome.displaced_trie_cache.take(),
            &context,
            SealedHeader::seal_slow(header.clone()),
            true,
        );
        let oracle = CommitOracle {
            verdict: RecordedVerdict::Accepted,
            state_root: Some(validated.outcome.state_root),
            next_cache_anchor: Some(validated.outcome.next_cache_anchor),
            expected_miss: Some(validated.outcome.expected_miss),
            readiness_state: "ready".into(),
            readiness_watermark: None,
            durability_watermark: None,
            retained_generation: None,
            coordinated_fingerprint: pair.fingerprint(),
            lifecycle_fingerprint: pair.lifecycle_fingerprint(),
        };
        commits.push(StreamEvent::Commit(Box::new(CommitFrame::new(
            CommitInput {
                block: block_ref,
                parent_hash,
                payload_provenance: PayloadProvenance::Reconstructed,
                payload_json: Some(serde_json::to_vec(&payload).unwrap()),
                sidecar: bincode::serialize(&sidecar).unwrap(),
            },
            oracle,
        ))));
        parent = header;
    }
    (fixture, commits)
}
