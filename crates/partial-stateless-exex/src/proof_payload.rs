use alloy_primitives::{hex, keccak256, map::B256Set, Address, B256};
use partial_stateless::{
    EncodedCodePreimage, EncodedProofNode, EncodedStorageMultiproof, StateMultiproofPayload,
    StateMultiproofPayloadStats, WitnessPayload, WitnessTargets,
};
use reth_storage_api::StateProvider;
use reth_trie::{MultiProof, MultiProofTargets, Nibbles, TrieInput};
use std::{str::FromStr, time::Instant};

pub(crate) fn build_state_multiproof_payload(
    state_provider: &dyn StateProvider,
    missing_targets: &WitnessTargets,
) -> eyre::Result<WitnessPayload> {
    let started = Instant::now();
    let proof_targets = build_multiproof_targets(missing_targets)?;
    let multiproof = if proof_targets.is_empty() {
        MultiProof::default()
    } else {
        state_provider.multiproof(TrieInput::default(), proof_targets)?
    };

    let mut payload = encode_multiproof(multiproof);
    add_code_preimages(state_provider, missing_targets, &mut payload)?;
    payload.stats.generation_latency_ms = started.elapsed().as_millis();

    Ok(WitnessPayload::StateMultiproofV1(payload))
}

fn build_multiproof_targets(missing_targets: &WitnessTargets) -> eyre::Result<MultiProofTargets> {
    let mut targets = MultiProofTargets::with_capacity(
        missing_targets.accounts.len() + missing_targets.storage_slots.len(),
    );

    for address in &missing_targets.accounts {
        let address = parse_address(address)?;
        targets.entry(keccak256(address)).or_default();
    }

    for target in &missing_targets.storage_slots {
        let address = parse_address(&target.address)?;
        let raw_slot = parse_b256(&target.slot, "storage slot")?;
        targets
            .entry(keccak256(address))
            .or_insert_with(B256Set::default)
            .insert(keccak256(raw_slot));
    }

    Ok(targets)
}

fn encode_multiproof(multiproof: MultiProof) -> StateMultiproofPayload {
    let account_nodes = multiproof
        .account_subtree
        .nodes_sorted()
        .into_iter()
        .map(|(path, node)| encode_node(path, node.as_ref()))
        .collect::<Vec<_>>();

    let mut storages = multiproof.storages.into_iter().collect::<Vec<_>>();
    storages.sort_unstable_by_key(|(hashed_address, _)| *hashed_address);

    let storage_multiproofs = storages
        .into_iter()
        .map(|(hashed_address, storage)| EncodedStorageMultiproof {
            hashed_address: format!("{hashed_address:?}"),
            storage_root: format!("{:?}", storage.root),
            nodes: storage
                .subtree
                .nodes_sorted()
                .into_iter()
                .map(|(path, node)| encode_node(path, node.as_ref()))
                .collect(),
        })
        .collect::<Vec<_>>();

    let mut stats = StateMultiproofPayloadStats {
        account_node_count: account_nodes.len(),
        account_node_bytes: account_nodes.iter().map(|node| node.byte_len).sum(),
        storage_multiproof_count: storage_multiproofs.len(),
        storage_node_count: storage_multiproofs.iter().map(|proof| proof.nodes.len()).sum(),
        storage_node_bytes: storage_multiproofs
            .iter()
            .flat_map(|proof| proof.nodes.iter())
            .map(|node| node.byte_len)
            .sum(),
        ..Default::default()
    };
    stats.unavailable_code_preimage_count = 0;

    StateMultiproofPayload {
        account_nodes,
        storage_multiproofs,
        code_preimages: Vec::new(),
        unavailable_code_hashes: Vec::new(),
        stats,
    }
}

fn add_code_preimages(
    state_provider: &dyn StateProvider,
    missing_targets: &WitnessTargets,
    payload: &mut StateMultiproofPayload,
) -> eyre::Result<()> {
    for code_hash in &missing_targets.code_hashes {
        let code_hash_value = parse_b256(code_hash, "code hash")?;
        match state_provider.bytecode_by_hash(&code_hash_value)? {
            Some(bytecode) => {
                let raw = bytecode.original_byte_slice();
                if keccak256(raw) != code_hash_value {
                    eyre::bail!("bytecode preimage hash mismatch for {code_hash}");
                }
                payload.code_preimages.push(EncodedCodePreimage {
                    code_hash: code_hash.clone(),
                    bytecode: hex::encode_prefixed(raw),
                    byte_len: raw.len(),
                });
            }
            None => payload.unavailable_code_hashes.push(code_hash.clone()),
        }
    }

    payload.stats.code_preimage_count = payload.code_preimages.len();
    payload.stats.code_preimage_bytes =
        payload.code_preimages.iter().map(|code| code.byte_len).sum();
    payload.stats.unavailable_code_preimage_count = payload.unavailable_code_hashes.len();
    Ok(())
}

fn encode_node(path: Nibbles, rlp: &[u8]) -> EncodedProofNode {
    EncodedProofNode {
        path: nibbles_hex(&path),
        rlp: hex::encode_prefixed(rlp),
        byte_len: rlp.len(),
    }
}

fn parse_address(raw: &str) -> eyre::Result<Address> {
    Address::from_str(raw).map_err(|err| eyre::eyre!("invalid address target {raw}: {err}"))
}

fn parse_b256(raw: &str, label: &str) -> eyre::Result<B256> {
    B256::from_str(raw).map_err(|err| eyre::eyre!("invalid {label} target {raw}: {err}"))
}

fn nibbles_hex(path: &Nibbles) -> String {
    let mut encoded = String::with_capacity(2 + path.len());
    encoded.push_str("0x");
    for nibble in path.to_vec() {
        encoded.push(char::from_digit(nibble as u32, 16).expect("nibble is hex"));
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::U256;
    use partial_stateless::{StorageTarget, TargetSourceKind};

    #[test]
    fn multiproof_targets_hash_account_and_plain_slot() {
        let address = Address::with_last_byte(0x11);
        let raw_slot = B256::from(U256::from(7));
        let targets = WitnessTargets {
            source: TargetSourceKind::BundleChangedState,
            accounts: vec![format!("{address:?}")],
            storage_slots: vec![StorageTarget {
                address: format!("{address:?}"),
                slot: format!("{raw_slot:?}"),
            }],
            code_hashes: Vec::new(),
            header_numbers: Vec::new(),
        };

        let proof_targets = build_multiproof_targets(&targets).unwrap();
        let hashed_address = keccak256(address);
        let hashed_slot = keccak256(raw_slot);

        assert!(proof_targets.contains_key(&hashed_address));
        assert!(proof_targets.get(&hashed_address).unwrap().contains(&hashed_slot));
    }

    #[test]
    fn nibble_paths_are_hex_encoded() {
        let path = Nibbles::unpack(B256::with_last_byte(0xab));

        assert!(nibbles_hex(&path).starts_with("0x"));
        assert!(nibbles_hex(&path).ends_with("ab"));
    }
}
