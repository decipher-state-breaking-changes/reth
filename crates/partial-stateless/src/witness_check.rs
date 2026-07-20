use crate::sidecar::{
    check_sidecar_self_consistency, PartialStatelessSidecar, SerializableMultiProof,
    SidecarCheckError, StateTargetSet,
};
use alloy_consensus::Header;
use alloy_primitives::{keccak256, Address, Bytes, B256, U256};
use alloy_rlp::Decodable;
use reth_primitives_traits::Account;
use reth_trie_common::{
    proof::ProofNodes, MultiProof, Nibbles, RlpNode, TrieNode, EMPTY_ROOT_HASH,
};
use revm_database::BundleState;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    error::Error,
    fmt,
};

#[derive(Debug, Clone)]
pub struct SidecarWitnessCheckLimits {
    pub max_accounts: usize,
    pub max_storage_slots: usize,
    pub max_code_hashes: usize,
    pub max_headers: usize,
    pub max_state_proof_bytes: usize,
    pub max_code_bytes: usize,
    pub max_header_bytes: usize,
    pub max_key_bytes: usize,
}

impl Default for SidecarWitnessCheckLimits {
    fn default() -> Self {
        Self {
            max_accounts: 100_000,
            max_storage_slots: 300_000,
            max_code_hashes: 20_000,
            max_headers: 256,
            max_state_proof_bytes: 64 * 1024 * 1024,
            max_code_bytes: 64 * 1024 * 1024,
            max_header_bytes: 2 * 1024 * 1024,
            max_key_bytes: 16 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SidecarWitnessCheckError {
    Sidecar(SidecarCheckError),
    LimitExceeded { label: &'static str, actual: usize, cap: usize },
    Decode(String),
    Proof(String),
    Bytecode(String),
    Header(String),
}

impl fmt::Display for SidecarWitnessCheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sidecar(err) => write!(f, "sidecar self-consistency failed: {err:?}"),
            Self::LimitExceeded { label, actual, cap } => {
                write!(f, "{label} exceeds witness check cap: actual={actual}, cap={cap}")
            }
            Self::Decode(err) => write!(f, "decode failed: {err}"),
            Self::Proof(err) => write!(f, "proof check failed: {err}"),
            Self::Bytecode(err) => write!(f, "bytecode check failed: {err}"),
            Self::Header(err) => write!(f, "header witness check failed: {err}"),
        }
    }
}

impl Error for SidecarWitnessCheckError {}

type Result<T> = std::result::Result<T, SidecarWitnessCheckError>;
type MaterializedAccounts = HashMap<Address, Option<Account>>;
type MaterializedStorage = HashMap<(Address, B256), U256>;
type MaterializedState = (MaterializedAccounts, MaterializedStorage);

#[derive(Debug)]
pub struct MaterializedSidecarWitness {
    pub accounts: HashMap<Address, Option<Account>>,
    pub storage: HashMap<(Address, B256), U256>,
    pub codes: HashMap<B256, Bytes>,
    pub headers: HashMap<u64, B256>,
    /// Parent-state trie proof retained for post-execution write-path verification and root
    /// calculation. Account/storage values above remain restricted to cache misses.
    pub state_proof: MultiProof,
}

pub fn check_sidecar_witness_prefilter(
    sidecar: &PartialStatelessSidecar,
    limits: &SidecarWitnessCheckLimits,
) -> Result<()> {
    check_sidecar_self_consistency(sidecar).map_err(SidecarWitnessCheckError::Sidecar)?;

    let targets = &sidecar.cache_miss_targets;
    ensure_cap("account targets", targets.accounts.len(), limits.max_accounts)?;
    ensure_cap("storage targets", targets.storage.len(), limits.max_storage_slots)?;
    ensure_cap("code targets", targets.code_hashes.len(), limits.max_code_hashes)?;
    ensure_cap("header witnesses", sidecar.witness.headers.len(), limits.max_headers)?;
    ensure_cap(
        "state proof bytes",
        sidecar.witness.state.mpt_multiproof_bytes().len(),
        limits.max_state_proof_bytes,
    )?;
    ensure_cap(
        "bytecode witness bytes",
        sidecar.witness.codes.iter().map(|bytes| bytes.len()).sum(),
        limits.max_code_bytes,
    )?;
    ensure_cap(
        "header witness bytes",
        sidecar.witness.headers.iter().map(|bytes| bytes.len()).sum(),
        limits.max_header_bytes,
    )?;
    ensure_cap(
        "key witness bytes",
        sidecar.witness.keys.iter().map(|bytes| bytes.len()).sum(),
        limits.max_key_bytes,
    )?;

    Ok(())
}

pub fn materialize_sidecar_witness(
    sidecar: &PartialStatelessSidecar,
) -> Result<MaterializedSidecarWitness> {
    materialize_sidecar_witness_with_limits(sidecar, &SidecarWitnessCheckLimits::default())
}

pub fn materialize_sidecar_witness_with_limits(
    sidecar: &PartialStatelessSidecar,
    limits: &SidecarWitnessCheckLimits,
) -> Result<MaterializedSidecarWitness> {
    check_sidecar_witness_prefilter(sidecar, limits)?;

    let crate::sidecar::PartialExecutionWitnessState::MptMultiProof(bytes) = &sidecar.witness.state;
    let serializable: SerializableMultiProof = bincode::deserialize(bytes).map_err(|err| {
        SidecarWitnessCheckError::Decode(format!("failed to decode sidecar multiproof: {err}"))
    })?;
    let multiproof = serializable.to_multiproof();

    let (accounts, storage) = materialize_state_targets(
        &multiproof,
        sidecar.parent_state_root,
        &sidecar.cache_miss_targets,
    )?;

    let codes = materialize_codes(sidecar)?;
    let headers = materialize_headers(sidecar)?;

    Ok(MaterializedSidecarWitness { accounts, storage, codes, headers, state_proof: multiproof })
}

/// Verify that `multiproof` authenticates every account/storage path in `targets` against the
/// parent state root. This is intentionally separate from cache-miss materialization: execution
/// values may come from the cache, while every path needed to recompute the post-state root must
/// still be authenticated by the sidecar proof.
pub fn verify_state_proof_targets(
    multiproof: &MultiProof,
    parent_state_root: B256,
    targets: &StateTargetSet,
) -> Result<()> {
    materialize_state_targets(multiproof, parent_state_root, targets).map(|_| ())
}

fn materialize_state_targets(
    multiproof: &MultiProof,
    parent_state_root: B256,
    targets: &StateTargetSet,
) -> Result<MaterializedState> {
    let mut grouped_targets: BTreeMap<Address, BTreeSet<B256>> = BTreeMap::new();
    for address in &targets.accounts {
        grouped_targets.entry(*address).or_default();
    }
    for (address, slot) in &targets.storage {
        grouped_targets.entry(*address).or_default().insert(*slot);
    }

    let mut accounts = HashMap::new();
    let mut storage = HashMap::new();
    for (address, slots) in grouped_targets {
        let slots = slots.into_iter().collect::<Vec<_>>();
        let hashed_address = keccak256(address);
        if !proof_path_is_complete(&multiproof.account_subtree, hashed_address, parent_state_root)?
        {
            return Err(SidecarWitnessCheckError::Proof(format!(
                "account path is not fully revealed: {address:?}"
            )))
        }
        let account_proof = multiproof.account_proof(address, &slots).map_err(|err| {
            SidecarWitnessCheckError::Proof(format!(
                "failed to materialize account proof for {address:?}: {err}"
            ))
        })?;

        account_proof.verify(parent_state_root).map_err(|err| {
            SidecarWitnessCheckError::Proof(format!(
                "invalid account/storage proof for {address:?}: {err}"
            ))
        })?;

        if account_proof.info.is_some() && account_proof.storage_root != EMPTY_ROOT_HASH {
            let storage_proof = multiproof.storages.get(&hashed_address).ok_or_else(|| {
                SidecarWitnessCheckError::Proof(format!(
                    "missing storage multiproof for account {address:?}"
                ))
            })?;
            for slot in &slots {
                if !proof_path_is_complete(
                    &storage_proof.subtree,
                    keccak256(slot),
                    storage_proof.root,
                )? {
                    return Err(SidecarWitnessCheckError::Proof(format!(
                        "storage path is not fully revealed: {address:?}/{slot:?}"
                    )))
                }
            }
        }

        accounts.insert(address, account_proof.info);
        for proof in account_proof.storage_proofs {
            storage.insert((address, proof.key), proof.value);
        }
    }

    Ok((accounts, storage))
}

/// Check path completeness directly in the legacy path-addressed proof.
///
/// Converting a proof to V2 is not suitable for this check: a lone extension whose prefix differs
/// from the target already proves non-existence, but V2 intentionally drops extensions whose child
/// branch is not revealed. Conversely, merely verifying a proof root is insufficient because a
/// hashed child on the target path can remain unresolved. This traversal accepts terminal
/// non-existence while rejecting unresolved hashed children.
fn proof_path_is_complete(nodes: &ProofNodes, target: B256, expected_root: B256) -> Result<bool> {
    if expected_root == EMPTY_ROOT_HASH && nodes.is_empty() {
        return Ok(true)
    }

    let target = Nibbles::unpack(target);
    let mut path = Nibbles::default();
    let Some(root) = nodes.get(&path) else { return Ok(false) };
    let mut node = decode_proof_node(root)?;

    loop {
        match node {
            TrieNode::EmptyRoot | TrieNode::Leaf(_) => return Ok(true),
            TrieNode::Extension(extension) => {
                let remaining = target.slice(path.len()..);
                if remaining.len() < extension.key.len() ||
                    remaining.slice(..extension.key.len()) != extension.key
                {
                    return Ok(true)
                }
                path.extend(&extension.key);
                let Some(next) = resolve_proof_child(&extension.child, nodes, &path)? else {
                    return Ok(false)
                };
                node = next;
            }
            TrieNode::Branch(branch) => {
                if path.len() >= target.len() {
                    return Ok(false)
                }
                let nibble = target.get(path.len()).expect("target length checked above");
                let branch_ref = branch.as_ref();
                let child = branch_ref
                    .children()
                    .find_map(|(index, child)| (index == nibble).then_some(child).flatten());
                let Some(child) = child else { return Ok(true) };
                path.push(nibble);
                let Some(next) = resolve_proof_child(child, nodes, &path)? else {
                    return Ok(false)
                };
                node = next;
            }
        }
    }
}

fn resolve_proof_child(
    child: &RlpNode,
    nodes: &ProofNodes,
    child_path: &Nibbles,
) -> Result<Option<TrieNode>> {
    if child.is_hash() {
        nodes.get(child_path).map(|node| decode_proof_node(node)).transpose()
    } else {
        decode_proof_node(child.as_slice()).map(Some)
    }
}

fn decode_proof_node(bytes: &[u8]) -> Result<TrieNode> {
    let mut input = bytes;
    let node = TrieNode::decode(&mut input).map_err(|err| {
        SidecarWitnessCheckError::Decode(format!("invalid trie proof node: {err}"))
    })?;
    if !input.is_empty() {
        return Err(SidecarWitnessCheckError::Decode(
            "trie proof node has trailing RLP bytes".to_string(),
        ))
    }
    Ok(node)
}

fn ensure_cap(label: &'static str, actual: usize, cap: usize) -> Result<()> {
    if actual > cap {
        return Err(SidecarWitnessCheckError::LimitExceeded { label, actual, cap });
    }
    Ok(())
}

fn materialize_codes(sidecar: &PartialStatelessSidecar) -> Result<HashMap<B256, Bytes>> {
    let declared: HashSet<B256> = sidecar.cache_miss_targets.code_hashes.iter().copied().collect();
    let mut codes = HashMap::new();
    for code in &sidecar.witness.codes {
        let code_hash = keccak256(code.as_ref());
        if !declared.contains(&code_hash) {
            return Err(SidecarWitnessCheckError::Bytecode(format!(
                "sidecar carries undeclared bytecode preimage: {code_hash:?}"
            )));
        }
        if codes.insert(code_hash, code.clone()).is_some() {
            return Err(SidecarWitnessCheckError::Bytecode(format!(
                "sidecar carries duplicate bytecode preimage: {code_hash:?}"
            )));
        }
    }
    if codes.len() != declared.len() {
        let missing = declared
            .into_iter()
            .filter(|code_hash| !codes.contains_key(code_hash))
            .collect::<Vec<_>>();
        return Err(SidecarWitnessCheckError::Bytecode(format!(
            "sidecar missing bytecode preimages: {missing:?}"
        )));
    }
    Ok(codes)
}

fn materialize_headers(sidecar: &PartialStatelessSidecar) -> Result<HashMap<u64, B256>> {
    let mut decoded = Vec::with_capacity(sidecar.witness.headers.len());
    let mut headers = HashMap::new();
    for raw in &sidecar.witness.headers {
        let mut raw = raw.as_ref();
        let header = Header::decode(&mut raw).map_err(|err| {
            SidecarWitnessCheckError::Header(format!(
                "failed to decode ancestor header witness: {err}"
            ))
        })?;
        if header.number >= sidecar.block_number {
            return Err(SidecarWitnessCheckError::Header(format!(
                "ancestor header witness is not an ancestor: number={}",
                header.number
            )));
        }
        let hash = header.hash_slow();
        if headers.insert(header.number, hash).is_some() {
            return Err(SidecarWitnessCheckError::Header(format!(
                "duplicate ancestor header witness: number={}",
                header.number
            )));
        }
        decoded.push((header.number, hash, header.parent_hash));
    }

    decoded.sort_by_key(|(number, _, _)| *number);
    if sidecar.block_number > 0 && !decoded.is_empty() {
        let Some(parent) = decoded.last() else { unreachable!("checked non-empty") };
        if parent.0 != sidecar.cache_block {
            return Err(SidecarWitnessCheckError::Header(format!(
                "ancestor header witness range must end at parent block: expected={}, got={}",
                sidecar.cache_block, parent.0
            )));
        }
        if parent.1 != sidecar.parent_hash {
            return Err(SidecarWitnessCheckError::Header(format!(
                "parent header witness hash mismatch: expected {:?}, got {:?}",
                sidecar.parent_hash, parent.1
            )));
        }
    }

    for pair in decoded.windows(2) {
        let [left, right] = pair else { continue };
        if left.0 + 1 != right.0 {
            return Err(SidecarWitnessCheckError::Header(format!(
                "ancestor header witness range has a gap: left={}, right={}",
                left.0, right.0
            )));
        }
        if right.2 != left.1 {
            return Err(SidecarWitnessCheckError::Header(format!(
                "ancestor header witness chain mismatch: child={}, parent_hash={:?}, expected={:?}",
                right.0, right.2, left.1
            )));
        }
    }

    Ok(headers)
}

/// Return the account and storage paths whose trie values change in this bundle.
///
/// Storage changes also target the containing account because its storage root is embedded in the
/// account leaf. Code hashes are represented by the changed account leaf and therefore do not need
/// a separate trie proof target.
pub fn write_state_targets_from_bundle(bundle_state: &BundleState) -> StateTargetSet {
    let mut targets = StateTargetSet::default();

    for (address, account) in &bundle_state.state {
        if account.is_info_changed() || account.was_destroyed() {
            targets.accounts.push(*address);
        }
        for (slot, value) in &account.storage {
            if value.is_changed() {
                targets.accounts.push(*address);
                targets.storage.push((*address, B256::new(slot.to_be_bytes())));
            }
        }
    }

    targets.sort_dedup();
    targets
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        sidecar::{
            CacheAnchor, PartialExecutionWitness, PartialExecutionWitnessState, WitnessTargets,
        },
        witness::WitnessResult,
    };
    use alloy_primitives::map::HashMap as AlloyHashMap;
    use alloy_rlp::Encodable;
    use reth_trie_common::{
        hash_builder::HashBuilder, proof::ProofRetainer, BranchNodeMasksMap, Nibbles,
        StorageMultiProof,
    };

    fn empty_stats() -> WitnessResult {
        WitnessResult {
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
        }
    }

    fn test_anchor(block_number: u64, block_hash: B256, cache_policy_id: B256) -> CacheAnchor {
        CacheAnchor { block_number, block_hash, cache_policy_id, cache_root: keccak256(block_hash) }
    }

    fn header(number: u64, parent_hash: B256) -> Header {
        Header { number, parent_hash, gas_limit: 1, ..Default::default() }
    }

    fn encode_header(header: &Header) -> Bytes {
        let mut out = Vec::new();
        header.encode(&mut out);
        out.into()
    }

    fn sidecar_with_headers(
        block_number: u64,
        parent_hash: B256,
        headers: Vec<Bytes>,
    ) -> PartialStatelessSidecar {
        let cache_block = block_number.saturating_sub(1);
        let block_hash = B256::repeat_byte(0xbb);
        let cache_policy_id = B256::repeat_byte(0xcc);
        PartialStatelessSidecar {
            parent_hash,
            parent_state_root: B256::ZERO,
            block_hash,
            block_number,
            cache_block,
            cache_policy_id,
            prev_cache_anchor: test_anchor(cache_block, parent_hash, cache_policy_id),
            next_cache_anchor: test_anchor(block_number, block_hash, cache_policy_id),
            cache_policy_metadata: String::new(),
            cache_miss_targets: StateTargetSet::default(),
            witness_commitment: B256::ZERO,
            miss_manifest: WitnessTargets {
                missed_accounts: vec![],
                missed_storage: vec![],
                missed_code_hashes: vec![],
            },
            witness: PartialExecutionWitness {
                state: PartialExecutionWitnessState::MptMultiProof(vec![]),
                codes: vec![],
                keys: vec![],
                headers,
            },
            stats: empty_stats(),
        }
    }

    #[test]
    fn extension_prefix_mismatch_authenticates_absent_storage_slot() {
        let address = Address::repeat_byte(0x45);
        let account = Account { nonce: 1, balance: U256::ZERO, bytecode_hash: None };

        let mut existing = Vec::new();
        let mut target = None;
        for candidate in 0u64.. {
            let raw = B256::from(U256::from(candidate));
            match keccak256(raw)[0] >> 4 {
                0x3 if existing.len() < 2 => {
                    existing.push((keccak256(raw), U256::from(candidate + 1)));
                }
                0x7 if target.is_none() => target = Some(raw),
                _ => {}
            }
            if existing.len() == 2 && target.is_some() {
                break
            }
        }
        existing.sort_by_key(|(key, _)| *key);
        let target = target.unwrap();
        let target_path = Nibbles::unpack(keccak256(target));

        let mut storage_builder =
            HashBuilder::default().with_proof_retainer(ProofRetainer::from_iter([target_path]));
        for (key, value) in existing {
            storage_builder.add_leaf(Nibbles::unpack(key), &alloy_rlp::encode_fixed_size(&value));
        }
        let storage_root = storage_builder.root();
        let storage_subtree = storage_builder.take_proof_nodes();

        let hashed_address = keccak256(address);
        let account_path = Nibbles::unpack(hashed_address);
        let mut account_builder =
            HashBuilder::default().with_proof_retainer(ProofRetainer::from_iter([account_path]));
        account_builder
            .add_leaf(account_path, &alloy_rlp::encode(account.into_trie_account(storage_root)));
        let state_root = account_builder.root();
        let mut multiproof = MultiProof {
            account_subtree: account_builder.take_proof_nodes(),
            ..Default::default()
        };
        multiproof.storages.insert(
            hashed_address,
            StorageMultiProof {
                root: storage_root,
                subtree: storage_subtree,
                branch_node_masks: BranchNodeMasksMap::default(),
            },
        );
        let targets = StateTargetSet { storage: vec![(address, target)], ..Default::default() };

        let (_, storage) = materialize_state_targets(&multiproof, state_root, &targets).unwrap();

        assert_eq!(storage[&(address, target)], U256::ZERO);
    }

    #[test]
    fn write_targets_only_include_changed_bundle_paths() {
        let address = Address::repeat_byte(0x11);
        let account = Account { nonce: 1, balance: U256::from(10), bytecode_hash: None };
        let changed_slot = U256::from(7);
        let unchanged_slot = U256::from(8);
        let bundle = BundleState::builder(1..=1)
            .state_original_account_info(address, account.into())
            .state_present_account_info(address, account.into())
            .state_storage(
                address,
                AlloyHashMap::from_iter([
                    (changed_slot, (U256::from(1), U256::from(2))),
                    (unchanged_slot, (U256::from(3), U256::from(3))),
                ]),
            )
            .build();

        let targets = write_state_targets_from_bundle(&bundle);

        assert_eq!(targets.accounts, vec![address]);
        assert_eq!(targets.storage, vec![(address, B256::from(changed_slot))]);
        assert!(targets.code_hashes.is_empty());
    }

    #[test]
    fn write_targets_cover_every_account_leaf_change_kind() {
        let nonce_account = Address::repeat_byte(0x10);
        let balance_account = Address::repeat_byte(0x20);
        let code_account = Address::repeat_byte(0x30);
        let created_account = Address::repeat_byte(0x40);
        let deleted_account = Address::repeat_byte(0x50);
        let storage_account = Address::repeat_byte(0x60);
        let unchanged_account = Address::repeat_byte(0x70);
        let base = Account { nonce: 1, balance: U256::from(10), bytecode_hash: None };
        let mut nonce_changed = base;
        nonce_changed.nonce += 1;
        let mut balance_changed = base;
        balance_changed.balance += U256::from(1);
        let code_changed = Account { bytecode_hash: Some(B256::repeat_byte(0xaa)), ..base };
        let slot = U256::from(7);

        let bundle = BundleState::builder(1..=1)
            .state_original_account_info(nonce_account, base.into())
            .state_present_account_info(nonce_account, nonce_changed.into())
            .state_original_account_info(balance_account, base.into())
            .state_present_account_info(balance_account, balance_changed.into())
            .state_original_account_info(code_account, base.into())
            .state_present_account_info(code_account, code_changed.into())
            .state_present_account_info(created_account, base.into())
            .state_original_account_info(deleted_account, base.into())
            .state_original_account_info(storage_account, base.into())
            .state_present_account_info(storage_account, base.into())
            .state_storage(
                storage_account,
                AlloyHashMap::from_iter([(slot, (U256::from(1), U256::from(2)))]),
            )
            .state_original_account_info(unchanged_account, base.into())
            .state_present_account_info(unchanged_account, base.into())
            .build();

        let targets = write_state_targets_from_bundle(&bundle);

        assert_eq!(
            targets.accounts,
            vec![
                nonce_account,
                balance_account,
                code_account,
                created_account,
                deleted_account,
                storage_account,
            ]
        );
        assert_eq!(targets.storage, vec![(storage_account, B256::from(slot))]);
        assert!(!targets.accounts.contains(&unchanged_account));
    }

    #[test]
    fn any_balance_delta_covers_transfer_withdrawal_coinbase_and_fee_effects() {
        let address = Address::repeat_byte(0x11);
        let before = Account { nonce: 1, balance: U256::from(10), bytecode_hash: None };
        let after = Account { balance: U256::from(11), ..before };
        let bundle = BundleState::builder(1..=1)
            .state_original_account_info(address, before.into())
            .state_present_account_info(address, after.into())
            .build();

        let targets = write_state_targets_from_bundle(&bundle);

        // BundleState records the resulting balance delta, not whether it came from a transfer,
        // withdrawal, block reward/coinbase fee, or another protocol-level balance change.
        assert_eq!(targets.accounts, vec![address]);
    }

    #[test]
    fn destroyed_status_is_a_write_even_when_account_info_matches() {
        let address = Address::repeat_byte(0x11);
        let account = Account { nonce: 1, balance: U256::from(10), bytecode_hash: None };
        let mut bundle = BundleState::builder(1..=1)
            .state_original_account_info(address, account.into())
            .state_present_account_info(address, account.into())
            .build();
        bundle.state.get_mut(&address).unwrap().status =
            revm_database::states::AccountStatus::DestroyedChanged;

        let targets = write_state_targets_from_bundle(&bundle);

        assert_eq!(targets.accounts, vec![address]);
    }

    #[test]
    fn header_witness_allows_empty_when_blockhash_is_not_used() {
        let sidecar = sidecar_with_headers(11, B256::repeat_byte(0xaa), vec![]);
        let headers = materialize_headers(&sidecar).unwrap();

        assert!(headers.is_empty());
    }

    #[test]
    fn header_witness_rejects_unanchored_range() {
        let header_9 = header(9, B256::repeat_byte(0x09));
        let sidecar =
            sidecar_with_headers(11, B256::repeat_byte(0xaa), vec![encode_header(&header_9)]);

        assert!(matches!(
            materialize_headers(&sidecar),
            Err(SidecarWitnessCheckError::Header(err))
                if err.contains("must end at parent block")
        ));
    }

    #[test]
    fn header_witness_accepts_contiguous_chain_to_parent() {
        let header_9 = header(9, B256::repeat_byte(0x08));
        let header_10 = header(10, header_9.hash_slow());
        let parent_hash = header_10.hash_slow();
        let sidecar = sidecar_with_headers(
            11,
            parent_hash,
            vec![encode_header(&header_9), encode_header(&header_10)],
        );
        let headers = materialize_headers(&sidecar).unwrap();

        assert_eq!(headers.get(&9), Some(&header_9.hash_slow()));
        assert_eq!(headers.get(&10), Some(&parent_hash));
    }

    #[test]
    fn header_witness_rejects_chain_gaps() {
        let header_8 = header(8, B256::repeat_byte(0x07));
        let header_9 = header(9, header_8.hash_slow());
        let header_10 = header(10, header_9.hash_slow());
        let sidecar = sidecar_with_headers(
            11,
            header_10.hash_slow(),
            vec![encode_header(&header_8), encode_header(&header_10)],
        );

        assert!(matches!(
            materialize_headers(&sidecar),
            Err(SidecarWitnessCheckError::Header(err))
                if err.contains("range has a gap")
        ));
    }
}
