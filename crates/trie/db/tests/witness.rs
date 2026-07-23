#![allow(missing_docs)]

use alloy_consensus::EMPTY_ROOT_HASH;
use alloy_primitives::{
    keccak256,
    map::{HashMap, HashSet},
    Address, Bytes, B256, U256,
};
use alloy_rlp::EMPTY_STRING_CODE;
use reth_db::{cursor::DbCursorRW, tables};
use reth_db_api::transaction::DbTxMut;
use reth_primitives_traits::{Account, StorageEntry};
use reth_provider::{test_utils::create_test_provider_factory, HashingWriter};
use reth_storage_api::StorageSettingsCache;
use reth_trie::{
    proof::Proof, witness::TrieWitness, DecodedMultiProof, DecodedMultiProofV2,
    ExecutionWitnessMode, HashedPostState, HashedStorage, LeafNode, MultiProofTargets,
    MultiProofTargetsV2, Nibbles, ProofV2Target, StateRoot, StorageRoot, TrieInput, TrieNodeV2,
};
use reth_trie_db::{
    DatabaseHashedCursorFactory, DatabaseProof, DatabaseStateRoot, DatabaseStorageRoot,
    DatabaseTrieCursorFactory,
};

type DbStateRoot<'a, TX, A> =
    StateRoot<DatabaseTrieCursorFactory<&'a TX, A>, DatabaseHashedCursorFactory<&'a TX>>;
type DbStorageRoot<'a, TX, A> =
    StorageRoot<DatabaseTrieCursorFactory<&'a TX, A>, DatabaseHashedCursorFactory<&'a TX>>;
type DbProof<'a, TX, A> =
    Proof<DatabaseTrieCursorFactory<&'a TX, A>, DatabaseHashedCursorFactory<&'a TX>>;

#[test]
fn native_v2_proof_preserves_overlay_root_context_and_min_len() {
    let factory = create_test_provider_factory();
    let provider = factory.provider_rw().unwrap();
    let address = Address::repeat_byte(0x11);
    let other_address = Address::repeat_byte(0x22);
    let overlay_address = Address::repeat_byte(0x33);
    let slot = B256::repeat_byte(0x44);
    let hashed_address = keccak256(address);
    let hashed_slot = keccak256(slot);
    let hashed_overlay_address = keccak256(overlay_address);

    provider
        .insert_account_for_hashing([
            (address, Some(Account::default())),
            (other_address, Some(Account { nonce: 1, ..Default::default() })),
        ])
        .unwrap();
    provider
        .insert_storage_for_hashing([(address, [StorageEntry { key: slot, value: U256::from(7) }])])
        .unwrap();

    reth_trie_db::with_adapter!(provider, |A| {
        let expected_root = DbStateRoot::<_, A>::from_tx(provider.tx_ref()).root().unwrap();
        let proof = <DbProof<'_, _, A> as DatabaseProof>::from_tx(provider.tx_ref());
        let full = proof
            .overlay_multiproof_v2(
                TrieInput::default(),
                MultiProofTargetsV2 {
                    account_targets: vec![ProofV2Target::new(hashed_address)],
                    storage_targets: HashMap::from_iter([(
                        hashed_address,
                        vec![ProofV2Target::new(hashed_slot)],
                    )]),
                },
            )
            .unwrap();
        let root = full.account_proofs.iter().find(|node| node.path.is_empty()).unwrap();
        assert_eq!(keccak256(alloy_rlp::encode(&root.node)), expected_root);
        assert!(full.storage_proofs.contains_key(&hashed_address));

        let legacy = proof
            .overlay_multiproof(
                TrieInput::default(),
                MultiProofTargets::account_with_slots(hashed_address, [hashed_slot]),
            )
            .unwrap();
        let legacy: DecodedMultiProofV2 = DecodedMultiProof::try_from(legacy).unwrap().into();
        let legacy_root = legacy.account_proofs.iter().find(|node| node.path.is_empty()).unwrap();
        assert_eq!(keccak256(alloy_rlp::encode(&legacy_root.node)), expected_root);
        let native_storage_root =
            full.storage_proofs[&hashed_address].iter().find(|node| node.path.is_empty()).unwrap();
        let legacy_storage_root = legacy.storage_proofs[&hashed_address]
            .iter()
            .find(|node| node.path.is_empty())
            .unwrap();
        assert_eq!(
            keccak256(alloy_rlp::encode(&native_storage_root.node)),
            keccak256(alloy_rlp::encode(&legacy_storage_root.node))
        );

        let suffix_only = proof
            .overlay_multiproof_v2(
                TrieInput::default(),
                MultiProofTargetsV2 {
                    account_targets: vec![ProofV2Target::new(hashed_address).with_min_len(1)],
                    storage_targets: HashMap::from_iter([(
                        hashed_address,
                        vec![ProofV2Target::new(hashed_slot).with_min_len(1)],
                    )]),
                },
            )
            .unwrap();
        assert!(suffix_only.account_proofs.iter().all(|node| node.path.len() >= 1));
        assert!(suffix_only.storage_proofs[&hashed_address]
            .iter()
            .all(|node| node.path.len() >= 1));

        let overlay = HashedPostState {
            accounts: HashMap::from_iter([(
                hashed_overlay_address,
                Some(Account { nonce: 2, ..Default::default() }),
            )]),
            storages: HashMap::default(),
        };
        let overlay_root =
            DbStateRoot::<_, A>::overlay_root(provider.tx_ref(), &overlay.clone().into_sorted())
                .unwrap();
        let overlay_proof = proof
            .overlay_multiproof_v2(
                TrieInput::from_state(overlay),
                MultiProofTargetsV2 {
                    account_targets: vec![ProofV2Target::new(hashed_overlay_address)],
                    storage_targets: HashMap::default(),
                },
            )
            .unwrap();
        let overlay_root_node =
            overlay_proof.account_proofs.iter().find(|node| node.path.is_empty()).unwrap();
        assert_eq!(keccak256(alloy_rlp::encode(&overlay_root_node.node)), overlay_root);
    });
}

#[test]
fn includes_empty_node_preimage() {
    let factory = create_test_provider_factory();
    let provider = factory.provider_rw().unwrap();

    let address = Address::random();
    let hashed_address = keccak256(address);
    let hashed_slot = B256::random();

    reth_trie_db::with_adapter!(provider, |A| {
        let legacy_empty_witness = TrieWitness::new(
            DatabaseTrieCursorFactory::<_, A>::new(provider.tx_ref()),
            DatabaseHashedCursorFactory::new(provider.tx_ref()),
        )
        .with_execution_witness_mode(ExecutionWitnessMode::Legacy)
        .compute(HashedPostState {
            accounts: HashMap::from_iter([(hashed_address, Some(Account::default()))]),
            storages: HashMap::default(),
        })
        .unwrap();
        assert_eq!(
            legacy_empty_witness,
            HashMap::from_iter([(EMPTY_ROOT_HASH, Bytes::from([EMPTY_STRING_CODE]))])
        );

        let canonical_empty_witness = TrieWitness::new(
            DatabaseTrieCursorFactory::<_, A>::new(provider.tx_ref()),
            DatabaseHashedCursorFactory::new(provider.tx_ref()),
        )
        .with_execution_witness_mode(ExecutionWitnessMode::Canonical)
        .compute(HashedPostState {
            accounts: HashMap::from_iter([(hashed_address, Some(Account::default()))]),
            storages: HashMap::default(),
        })
        .unwrap();
        assert!(canonical_empty_witness.is_empty());

        // Insert account into database
        provider.insert_account_for_hashing([(address, Some(Account::default()))]).unwrap();

        let state_root = DbStateRoot::<_, A>::from_tx(provider.tx_ref()).root().unwrap();
        let proof = <DbProof<'_, _, A> as DatabaseProof>::from_tx(provider.tx_ref());
        let multiproof = proof
            .multiproof(MultiProofTargets::from_iter([(
                hashed_address,
                HashSet::from_iter([hashed_slot]),
            )]))
            .unwrap();

        let legacy_witness = TrieWitness::new(
            DatabaseTrieCursorFactory::<_, A>::new(provider.tx_ref()),
            DatabaseHashedCursorFactory::new(provider.tx_ref()),
        )
        .with_execution_witness_mode(ExecutionWitnessMode::Legacy)
        .compute(HashedPostState {
            accounts: HashMap::from_iter([(hashed_address, Some(Account::default()))]),
            storages: HashMap::from_iter([(
                hashed_address,
                HashedStorage::from_iter(false, [(hashed_slot, U256::from(1))]),
            )]),
        })
        .unwrap();
        assert!(legacy_witness.contains_key(&state_root));
        for node in multiproof.account_subtree.values() {
            assert_eq!(legacy_witness.get(&keccak256(node)), Some(node));
        }
        assert_eq!(legacy_witness.get(&EMPTY_ROOT_HASH), Some(&Bytes::from([EMPTY_STRING_CODE])));

        let canonical_witness = TrieWitness::new(
            DatabaseTrieCursorFactory::<_, A>::new(provider.tx_ref()),
            DatabaseHashedCursorFactory::new(provider.tx_ref()),
        )
        .with_execution_witness_mode(ExecutionWitnessMode::Canonical)
        .compute(HashedPostState {
            accounts: HashMap::from_iter([(hashed_address, Some(Account::default()))]),
            storages: HashMap::from_iter([(
                hashed_address,
                HashedStorage::from_iter(false, [(hashed_slot, U256::from(1))]),
            )]),
        })
        .unwrap();
        assert!(canonical_witness.contains_key(&state_root));
        for node in multiproof.account_subtree.values() {
            assert_eq!(canonical_witness.get(&keccak256(node)), Some(node));
        }
        assert!(!canonical_witness.contains_key(&EMPTY_ROOT_HASH));
    });
}

#[test]
fn includes_nodes_for_destroyed_storage_nodes() {
    let factory = create_test_provider_factory();
    let provider = factory.provider_rw().unwrap();

    let address = Address::random();
    let hashed_address = keccak256(address);
    let slot = B256::random();
    let hashed_slot = keccak256(slot);

    // Insert account and slot into database
    provider.insert_account_for_hashing([(address, Some(Account::default()))]).unwrap();
    provider
        .insert_storage_for_hashing([(address, [StorageEntry { key: slot, value: U256::from(1) }])])
        .unwrap();

    reth_trie_db::with_adapter!(provider, |A| {
        let state_root = DbStateRoot::<_, A>::from_tx(provider.tx_ref()).root().unwrap();
        let proof = <DbProof<'_, _, A> as DatabaseProof>::from_tx(provider.tx_ref());
        let multiproof = proof
            .multiproof(MultiProofTargets::from_iter([(
                hashed_address,
                HashSet::from_iter([hashed_slot]),
            )]))
            .unwrap();

        let witness =
            TrieWitness::new(
                DatabaseTrieCursorFactory::<_, A>::new(provider.tx_ref()),
                DatabaseHashedCursorFactory::new(provider.tx_ref()),
            )
            .compute(HashedPostState {
                accounts: HashMap::from_iter([(hashed_address, Some(Account::default()))]),
                storages: HashMap::from_iter([(
                    hashed_address,
                    HashedStorage::from_iter(true, []),
                )]), // destroyed
            })
            .unwrap();
        assert!(witness.contains_key(&state_root));
        for node in multiproof.account_subtree.values() {
            assert_eq!(witness.get(&keccak256(node)), Some(node));
        }
        for node in multiproof.storages.values().flat_map(|storage| storage.subtree.values()) {
            assert_eq!(witness.get(&keccak256(node)), Some(node));
        }
    });
}

#[test]
fn correctly_decodes_branch_node_values() {
    let factory = create_test_provider_factory();
    let provider = factory.provider_rw().unwrap();

    let address = Address::random();
    let hashed_address = keccak256(address);
    let hashed_slot1 = B256::repeat_byte(1);
    let hashed_slot2 = B256::repeat_byte(2);

    // Insert account and slots into database
    provider.insert_account_for_hashing([(address, Some(Account::default()))]).unwrap();
    let mut hashed_storage_cursor =
        provider.tx_ref().cursor_dup_write::<tables::HashedStorages>().unwrap();
    hashed_storage_cursor
        .upsert(hashed_address, &StorageEntry { key: hashed_slot1, value: U256::from(1) })
        .unwrap();
    hashed_storage_cursor
        .upsert(hashed_address, &StorageEntry { key: hashed_slot2, value: U256::from(1) })
        .unwrap();

    reth_trie_db::with_adapter!(provider, |A| {
        let state_root = DbStateRoot::<_, A>::from_tx(provider.tx_ref()).root().unwrap();
        let proof = <DbProof<'_, _, A> as DatabaseProof>::from_tx(provider.tx_ref());
        let multiproof = proof
            .multiproof(MultiProofTargets::from_iter([(
                hashed_address,
                HashSet::from_iter([hashed_slot1, hashed_slot2]),
            )]))
            .unwrap();

        let witness = TrieWitness::new(
            DatabaseTrieCursorFactory::<_, A>::new(provider.tx_ref()),
            DatabaseHashedCursorFactory::new(provider.tx_ref()),
        )
        .compute(HashedPostState {
            accounts: HashMap::from_iter([(hashed_address, Some(Account::default()))]),
            storages: HashMap::from_iter([(
                hashed_address,
                HashedStorage::from_iter(
                    false,
                    [hashed_slot1, hashed_slot2].map(|hashed_slot| (hashed_slot, U256::from(2))),
                ),
            )]),
        })
        .unwrap();
        assert!(witness.contains_key(&state_root));
        for node in multiproof.account_subtree.values() {
            assert_eq!(witness.get(&keccak256(node)), Some(node));
        }
        for node in multiproof.storages.values().flat_map(|storage| storage.subtree.values()) {
            assert_eq!(witness.get(&keccak256(node)), Some(node));
        }
    });
}

#[test]
fn skips_storage_root_node_for_account_only_changes_in_canonical_mode() {
    let factory = create_test_provider_factory();
    let provider = factory.provider_rw().unwrap();

    let address = Address::random();
    let hashed_address = keccak256(address);
    let slot = B256::random();

    provider
        .insert_account_for_hashing([(
            address,
            Some(Account { balance: U256::from(1), ..Default::default() }),
        )])
        .unwrap();
    provider
        .insert_storage_for_hashing([(address, [StorageEntry { key: slot, value: U256::from(7) }])])
        .unwrap();

    reth_trie_db::with_adapter!(provider, |A| {
        let state_root = DbStateRoot::<_, A>::from_tx(provider.tx_ref()).root().unwrap();
        let storage_root =
            DbStorageRoot::<_, A>::from_tx(provider.tx_ref(), address).root().unwrap();
        let proof = <DbProof<'_, _, A> as DatabaseProof>::from_tx(provider.tx_ref());
        let multiproof = proof
            .multiproof(MultiProofTargets::from_iter([(hashed_address, HashSet::default())]))
            .unwrap();

        let target_state = HashedPostState {
            accounts: HashMap::from_iter([(
                hashed_address,
                Some(Account { balance: U256::from(2), ..Default::default() }),
            )]),
            storages: HashMap::default(),
        };

        let legacy_witness = TrieWitness::new(
            DatabaseTrieCursorFactory::<_, A>::new(provider.tx_ref()),
            DatabaseHashedCursorFactory::new(provider.tx_ref()),
        )
        .with_execution_witness_mode(ExecutionWitnessMode::Legacy)
        .compute(target_state.clone())
        .unwrap();
        assert!(legacy_witness.contains_key(&state_root));
        for node in multiproof.account_subtree.values() {
            assert_eq!(legacy_witness.get(&keccak256(node)), Some(node));
        }
        assert!(legacy_witness.contains_key(&storage_root));

        let canonical_witness = TrieWitness::new(
            DatabaseTrieCursorFactory::<_, A>::new(provider.tx_ref()),
            DatabaseHashedCursorFactory::new(provider.tx_ref()),
        )
        .with_execution_witness_mode(ExecutionWitnessMode::Canonical)
        .compute(target_state)
        .unwrap();
        assert!(canonical_witness.contains_key(&state_root));
        for node in multiproof.account_subtree.values() {
            assert_eq!(canonical_witness.get(&keccak256(node)), Some(node));
        }
        assert!(!canonical_witness.contains_key(&storage_root));
    });
}

#[test]
fn canonical_mode_handles_mixed_storage_inserts_and_removals() {
    let factory = create_test_provider_factory();
    let provider = factory.provider_rw().unwrap();

    let address = Address::random();
    let hashed_address = keccak256(address);

    // Pre-state storage root is a branch with two children at nibbles 1 and 2.
    let removed_slot = {
        let mut bytes = [0u8; 32];
        bytes[0] = 0x10;
        B256::from(bytes)
    };
    let retained_slot = {
        let mut bytes = [0u8; 32];
        bytes[0] = 0x20;
        B256::from(bytes)
    };
    let inserted_slot = {
        let mut bytes = [0u8; 32];
        bytes[0] = 0x30;
        B256::from(bytes)
    };
    let retained_leaf = Bytes::from(alloy_rlp::encode(TrieNodeV2::Leaf(LeafNode::new(
        Nibbles::unpack(retained_slot).slice(1..),
        alloy_rlp::encode_fixed_size(&U256::from(2)).to_vec(),
    ))));
    let retained_leaf_hash = keccak256(&retained_leaf);

    // The initial proof targets only the removed slot and the missing inserted slot, so the
    // surviving sibling starts out blinded.
    provider.insert_account_for_hashing([(address, Some(Account::default()))]).unwrap();
    let mut hashed_storage_cursor =
        provider.tx_ref().cursor_dup_write::<tables::HashedStorages>().unwrap();
    hashed_storage_cursor
        .upsert(hashed_address, &StorageEntry { key: removed_slot, value: U256::from(1) })
        .unwrap();
    hashed_storage_cursor
        .upsert(hashed_address, &StorageEntry { key: retained_slot, value: U256::from(2) })
        .unwrap();

    reth_trie_db::with_adapter!(provider, |A| {
        let state_root = DbStateRoot::<_, A>::from_tx(provider.tx_ref()).root().unwrap();
        let proof = <DbProof<'_, _, A> as DatabaseProof>::from_tx(provider.tx_ref());
        let initial_multiproof = proof
            .multiproof(MultiProofTargets::from_iter([(
                hashed_address,
                HashSet::from_iter([removed_slot, inserted_slot]),
            )]))
            .unwrap();
        assert!(initial_multiproof
            .storages
            .values()
            .flat_map(|storage| storage.subtree.values())
            .all(|node| node != &retained_leaf));

        // Apply one removal and one insertion in the same storage trie subtree.
        let state_diff = HashedPostState {
            accounts: HashMap::from_iter([(hashed_address, Some(Account::default()))]),
            storages: HashMap::from_iter([(
                hashed_address,
                HashedStorage::from_iter(
                    false,
                    [(removed_slot, U256::ZERO), (inserted_slot, U256::from(3))],
                ),
            )]),
        };

        // Legacy removes first, so the two-child branch collapses onto the surviving sibling
        // before the new child exists. That forces an extra proof fetch for the sibling leaf.
        let legacy_witness = TrieWitness::new(
            DatabaseTrieCursorFactory::<_, A>::new(provider.tx_ref()),
            DatabaseHashedCursorFactory::new(provider.tx_ref()),
        )
        .with_execution_witness_mode(ExecutionWitnessMode::Legacy)
        .compute(state_diff.clone())
        .unwrap();
        assert!(legacy_witness.contains_key(&state_root));
        assert_eq!(legacy_witness.get(&retained_leaf_hash), Some(&retained_leaf));

        // Canonical inserts first, so the branch never compresses to a single blinded child and
        // the surviving sibling leaf is not needed in the witness.
        let canonical_witness = TrieWitness::new(
            DatabaseTrieCursorFactory::<_, A>::new(provider.tx_ref()),
            DatabaseHashedCursorFactory::new(provider.tx_ref()),
        )
        .with_execution_witness_mode(ExecutionWitnessMode::Canonical)
        .compute(state_diff)
        .unwrap();
        assert!(canonical_witness.contains_key(&state_root));
        assert!(!canonical_witness.contains_key(&retained_leaf_hash));
        assert!(canonical_witness.iter().all(|(_, node)| node.as_ref() != [EMPTY_STRING_CODE]));
    });
}
