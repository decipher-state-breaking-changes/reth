use alloy_primitives::{
    keccak256,
    map::{B256Map, B256Set},
    Address, Bytes, B256, U256,
};
use alloy_rlp::{Decodable, Encodable};
use eyre::{bail, eyre, Result};
use partial_stateless::{
    accessed_state::BlockAccessedState,
    check_sidecar_context, check_sidecar_miss_targets,
    network_cache::{NetworkStateCache, UpdateStats},
    witness_check::{
        materialize_sidecar_witness_with_limits, write_state_targets_from_bundle,
        SidecarWitnessCheckLimits,
    },
    CacheAnchor, PartialStatelessSidecar, StateTargetSet,
};
use reth_ethereum::EthPrimitives;
use reth_evm::{execute::Executor, ConfigureEvm};
use reth_primitives_traits::{Account, AlloyBlockHeader, BlockTy, Bytecode, RecoveredBlock};
use reth_provider::{ProviderError, ProviderResult, StateProvider};
use reth_revm::database::{EvmStateProvider, StateProviderDatabase};
use reth_trie_common::{
    proof::ProofNodes, BranchNodeMasksMap, BranchNodeV2, DecodedMultiProofV2, ExecutionWitnessMode,
    HashedPostState, HashedStorage, MultiProof, Nibbles, ProofTrieNodeV2, StorageMultiProof,
    TrieInput, TrieNodeV2, EMPTY_ROOT_HASH,
};
use reth_trie_sparse::{LeafUpdate, SparseStateTrie, SparseTrie as _};
use revm::database::{BundleState, State};
use std::collections::HashMap;

pub(crate) type SidecarReexecLimits = SidecarWitnessCheckLimits;

#[derive(Debug, Clone)]
pub(crate) struct SidecarReexecReport {
    pub computed_state_root: B256,
    pub actual_accessed: BlockAccessedState,
    pub expected_miss: StateTargetSet,
    pub write_targets: StateTargetSet,
    pub next_cache_anchor: CacheAnchor,
    pub cache_update: UpdateStats,
}

pub(crate) fn verify_and_apply_trustless_sidecar<Evm>(
    evm_config: &Evm,
    block: &RecoveredBlock<BlockTy<EthPrimitives>>,
    expected_parent_state_root: B256,
    prev_cache: &mut NetworkStateCache,
    sidecar: &PartialStatelessSidecar,
    limits: &SidecarReexecLimits,
) -> Result<SidecarReexecReport>
where
    Evm: ConfigureEvm<Primitives = EthPrimitives>,
{
    prefilter(block, expected_parent_state_root, prev_cache, sidecar)?;

    let materialized = materialize_sidecar_witness_with_limits(sidecar, limits)
        .map_err(|err| eyre!("sidecar witness check failed: {err}"))?;
    let state_proof = materialized.state_proof;
    let witness_provider = WitnessBackedStateProvider {
        cache: prev_cache,
        witness_accounts: materialized.accounts,
        witness_storage: materialized.storage,
        witness_codes: materialized.codes,
        witness_headers: materialized.headers,
        block_number: sidecar.block_number,
    };

    let state_provider_db = StateProviderDatabase::new(witness_provider);
    let mut db = State::builder().with_bundle_update().with_database(state_provider_db).build();
    let block_executor = evm_config.executor(&mut db);

    let mut actual_accessed = BlockAccessedState::default();
    let execution_output = block_executor
        .execute_with_state_closure(block, |statedb: &State<_>| {
            actual_accessed = BlockAccessedState::from_simulated_state(statedb);
        })
        .map_err(|err| eyre!("partial sidecar re-execution failed: {err:?}"))?;
    drop(db);

    let expected_miss = prev_cache.expected_miss_targets(&actual_accessed);
    check_sidecar_miss_targets(sidecar, &expected_miss)
        .map_err(|err| eyre!("cache-miss-only check failed: {err:?}"))?;

    let write_targets = write_state_targets_from_bundle(&execution_output.state);
    let computed_state_root = compute_trustless_state_root(
        sidecar.parent_state_root,
        &execution_output.state,
        state_proof,
    )?;
    if computed_state_root != block.state_root() {
        bail!(
            "trustless state root mismatch: expected {:?}, got {:?}",
            block.state_root(),
            computed_state_root
        );
    }

    let (cache_update, next_cache_anchor) = apply_cache_transition_and_check(
        prev_cache,
        &actual_accessed,
        sidecar.block_number,
        sidecar.block_hash,
        sidecar.cache_policy_id,
        sidecar.next_cache_anchor,
    )?;

    Ok(SidecarReexecReport {
        computed_state_root,
        actual_accessed,
        expected_miss,
        write_targets,
        next_cache_anchor,
        cache_update,
    })
}

fn compute_trustless_state_root(
    parent_state_root: B256,
    bundle_state: &BundleState,
    multiproof: MultiProof,
) -> Result<B256> {
    let hashed_state = hashed_write_state_from_bundle(bundle_state);
    apply_hashed_state_transition(parent_state_root, &hashed_state, multiproof)
        .map(|result| result.state_root)
}

#[derive(Debug)]
pub(crate) struct CompletedTransitionProof {
    pub(crate) multiproof: MultiProof,
    pub(crate) state_root: B256,
}

#[derive(Debug, Clone)]
pub(crate) struct TransitionStructuralWitness {
    decoded: DecodedMultiProofV2,
}

pub(crate) fn build_transition_structural_witness(
    state_provider: &dyn StateProvider,
    parent_state_root: B256,
    bundle_state: &BundleState,
) -> Result<TransitionStructuralWitness> {
    let hashed_state = hashed_write_state_from_bundle(bundle_state);
    if hashed_state.is_empty() {
        return Ok(TransitionStructuralWitness { decoded: DecodedMultiProofV2::default() })
    }

    let nodes = state_provider
        .witness(TrieInput::default(), hashed_state, ExecutionWitnessMode::Canonical)
        .map_err(|err| eyre!("failed to generate canonical transition witness: {err}"))?;
    let witness = B256Map::from_iter(nodes.into_iter().map(|node| (keccak256(&node), node)));
    let decoded = DecodedMultiProofV2::from_witness(parent_state_root, &witness)
        .map_err(|err| eyre!("failed to decode canonical transition witness: {err}"))?;
    if decoded.is_empty() {
        bail!("canonical transition witness returned no trie nodes");
    }

    Ok(TransitionStructuralWitness { decoded })
}

/// Complete a `cold ∪ warm_writes` proof for trie topology changes.
///
/// Updating an existing leaf only needs its ordinary proof path. Insertion and deletion can split
/// or collapse a branch and therefore require the structure hidden behind a blinded sibling. Reth's
/// canonical transition witness supplies those additional nodes. We merge them into the ordinary
/// proof and independently replay the transition to prove that the resulting wire proof is
/// sufficient.
pub(crate) fn complete_transition_multiproof(
    parent_state_root: B256,
    bundle_state: &BundleState,
    multiproof: MultiProof,
    structural_witness: &TransitionStructuralWitness,
) -> Result<CompletedTransitionProof> {
    let hashed_state = hashed_write_state_from_bundle(bundle_state);
    if hashed_state.is_empty() {
        return Ok(CompletedTransitionProof { multiproof, state_root: parent_state_root })
    }

    let structural_proof =
        decoded_v2_to_multiproof(structural_witness.decoded.clone(), &multiproof)?;

    let mut completed_proof = multiproof;
    extend_multiproof_checked(&mut completed_proof, structural_proof)?;

    apply_hashed_state_transition(parent_state_root, &hashed_state, completed_proof)
}

/// Hash only state that changes the persisted trie.
///
/// `BundleState` also retains storage loaded during execution. Hashing all bundle entries would
/// turn warm read-only slots into update targets even though the sidecar deliberately proves only
/// cold reads and actual writes. Destroyed storage is handled like REVM's plain-state changeset:
/// wipe the old trie, then insert any non-zero values present after recreation.
fn hashed_write_state_from_bundle(bundle_state: &BundleState) -> HashedPostState {
    let mut hashed_state = HashedPostState::with_capacity(bundle_state.state().len());

    for (address, account) in bundle_state.state() {
        let hashed_address = keccak256(address);
        let was_destroyed = account.was_destroyed();
        let changed_storage = account.storage.iter().filter_map(|(slot, value)| {
            let persists_after_wipe = was_destroyed && !value.present_value.is_zero();
            let changed_without_wipe = !was_destroyed && value.is_changed();
            (persists_after_wipe || changed_without_wipe)
                .then(|| (keccak256(B256::from(*slot)), value.present_value))
        });
        let storage = HashedStorage::from_iter(was_destroyed, changed_storage);
        let storage_changed = !storage.is_empty();

        if account.is_info_changed() || storage_changed {
            hashed_state.accounts.insert(hashed_address, account.info.as_ref().map(Into::into));
        }
        if storage_changed {
            hashed_state.storages.insert(hashed_address, storage);
        }
    }

    hashed_state
}

fn decoded_v2_to_multiproof(decoded: DecodedMultiProofV2, base: &MultiProof) -> Result<MultiProof> {
    let (account_subtree, branch_node_masks) = encode_v2_proof_nodes(decoded.account_proofs)?;
    let mut storages =
        B256Map::with_capacity_and_hasher(decoded.storage_proofs.len(), Default::default());
    for (address, nodes) in decoded.storage_proofs {
        let (subtree, branch_node_masks) = encode_v2_proof_nodes(nodes)?;
        let root = if let Some(proof) = base.storages.get(&address) {
            proof.root
        } else {
            proof_nodes_root(&subtree)?
        };
        storages.insert(address, StorageMultiProof { root, subtree, branch_node_masks });
    }
    Ok(MultiProof { account_subtree, branch_node_masks, storages })
}

fn encode_v2_proof_nodes(nodes: Vec<ProofTrieNodeV2>) -> Result<(ProofNodes, BranchNodeMasksMap)> {
    let mut encoded_nodes = ProofNodes::default();
    let mut branch_node_masks = BranchNodeMasksMap::default();

    for ProofTrieNodeV2 { path, node, masks } in nodes {
        match node {
            TrieNodeV2::Branch(branch) if !branch.key.is_empty() => {
                let mut extension_rlp = Vec::new();
                TrieNodeV2::Branch(branch.clone()).encode(&mut extension_rlp);
                insert_proof_node(&mut encoded_nodes, path, Bytes::from(extension_rlp))?;

                let mut branch_path = path;
                branch_path.extend(&branch.key);
                let branch_only = TrieNodeV2::Branch(BranchNodeV2::new(
                    Nibbles::default(),
                    branch.stack,
                    branch.state_mask,
                    None,
                ));
                let mut branch_rlp = Vec::new();
                branch_only.encode(&mut branch_rlp);
                insert_proof_node(&mut encoded_nodes, branch_path, Bytes::from(branch_rlp))?;
                if let Some(masks) = masks {
                    branch_node_masks.insert(branch_path, masks);
                }
            }
            node => {
                let mut encoded = Vec::new();
                node.encode(&mut encoded);
                insert_proof_node(&mut encoded_nodes, path, Bytes::from(encoded))?;
                if let Some(masks) = masks {
                    branch_node_masks.insert(path, masks);
                }
            }
        }
    }

    Ok((encoded_nodes, branch_node_masks))
}

fn insert_proof_node(nodes: &mut ProofNodes, path: Nibbles, node: Bytes) -> Result<()> {
    if let Some(existing) = nodes.get(&path) {
        if existing != &node {
            bail!("structural witness contains conflicting trie nodes at path {path:?}");
        }
        return Ok(())
    }
    nodes.insert(path, node);
    Ok(())
}

fn proof_nodes_root(nodes: &ProofNodes) -> Result<B256> {
    let root = nodes
        .get(&Nibbles::default())
        .ok_or_else(|| eyre!("structural storage witness is missing its root node"))?;
    if root.as_ref() == [alloy_rlp::EMPTY_STRING_CODE] {
        Ok(EMPTY_ROOT_HASH)
    } else {
        Ok(keccak256(root))
    }
}

fn extend_multiproof_checked(base: &mut MultiProof, additional: MultiProof) -> Result<()> {
    for (path, node) in additional.account_subtree.iter() {
        if let Some(existing) = base.account_subtree.get(path) &&
            existing != node
        {
            bail!("ordinary and structural account proofs conflict at path {path:?}");
        }
    }
    for (address, storage) in &additional.storages {
        if let Some(existing_storage) = base.storages.get(address) {
            if existing_storage.root != storage.root {
                bail!("ordinary and structural storage roots conflict for account {address:?}");
            }
            for (path, node) in storage.subtree.iter() {
                if let Some(existing) = existing_storage.subtree.get(path) &&
                    existing != node
                {
                    bail!(
                        "ordinary and structural storage proofs conflict for account {address:?} at path {path:?}"
                    );
                }
            }
        }
    }
    base.extend(additional);
    Ok(())
}

/// Build the subset of a state multiproof needed to replay persisted storage writes.
///
/// The ordinary multiproof generator creates one storage entry for every account target. For a
/// non-empty storage trie with no requested slots, that entry can contain only a root extension
/// whose child remains hashed. Legacy-to-V2 conversion intentionally drops such an isolated
/// extension, leaving `SparseStateTrie` with a zero-node attempt to reveal a blind storage trie.
/// Read-only storage proofs remain in the wire multiproof for value authentication, but root replay
/// only reveals storage tries that are actually updated.
fn transition_reveal_multiproof(
    multiproof: &MultiProof,
    hashed_state: &HashedPostState,
) -> MultiProof {
    let mut reveal = multiproof.clone();
    reveal.storages.retain(|address, _| {
        hashed_state
            .storages
            .get(address)
            .is_some_and(|storage| !storage.wiped && !storage.storage.is_empty())
    });

    // A destroyed-and-recreated account starts its new storage trie from the canonical empty root;
    // the old storage shape is irrelevant after the wipe.
    for (address, storage) in &hashed_state.storages {
        if storage.wiped && !storage.storage.is_empty() {
            reveal.storages.insert(*address, StorageMultiProof::empty());
        }
    }
    reveal
}

fn apply_hashed_state_transition(
    parent_state_root: B256,
    hashed_state: &HashedPostState,
    multiproof: MultiProof,
) -> Result<CompletedTransitionProof> {
    if hashed_state.is_empty() {
        return Ok(CompletedTransitionProof { multiproof, state_root: parent_state_root })
    }

    let reveal_proof = transition_reveal_multiproof(&multiproof, hashed_state);
    let mut trie = SparseStateTrie::new();
    trie.reveal_multiproof(reveal_proof)
        .map_err(|err| eyre!("failed to reveal sidecar multiproof: {err}"))?;
    let revealed_parent_root =
        trie.root().map_err(|err| eyre!("failed to reconstruct parent state root: {err}"))?;
    if revealed_parent_root != parent_state_root {
        bail!(
            "sidecar multiproof root mismatch: expected {parent_state_root:?}, got {revealed_parent_root:?}"
        );
    }

    for (address, storage) in &hashed_state.storages {
        if storage.wiped {
            trie.wipe_storage(*address)
                .map_err(|err| eyre!("failed to wipe storage trie {address:?}: {err}"))?;
        }
    }

    let mut storage_removals: B256Map<B256Map<LeafUpdate>> = B256Map::default();
    let mut storage_upserts: B256Map<B256Map<LeafUpdate>> = B256Map::default();
    for (address, storage) in &hashed_state.storages {
        for (slot, value) in &storage.storage {
            let update = if value.is_zero() {
                LeafUpdate::Changed(Vec::new())
            } else {
                LeafUpdate::Changed(alloy_rlp::encode_fixed_size(value).to_vec())
            };
            if value.is_zero() {
                storage_removals.entry(*address).or_default().insert(*slot, update);
            } else {
                storage_upserts.entry(*address).or_default().insert(*slot, update);
            }
        }
    }

    for storage_updates in [&mut storage_upserts, &mut storage_removals] {
        let mut blinded = false;
        for (address, updates) in storage_updates.iter_mut() {
            if updates.is_empty() {
                continue
            }
            let storage_trie = trie.storage_trie_mut(address).ok_or_else(|| {
                eyre!("missing revealed storage trie for changed account {address:?}")
            })?;
            storage_trie
                .update_leaves(updates, |_key, _min_len| blinded = true)
                .map_err(|err| eyre!("failed to batch-update storage trie {address:?}: {err}"))?;
        }
        if blinded {
            bail!("sidecar multiproof is structurally incomplete for a storage transition");
        }
    }

    let mut account_removals: B256Map<LeafUpdate> = B256Map::default();
    let mut account_upserts: B256Map<LeafUpdate> = B256Map::default();
    let mut seen_accounts = B256Set::default();
    for address in hashed_state.accounts.keys().chain(hashed_state.storages.keys()) {
        if !seen_accounts.insert(*address) {
            continue
        }
        let account = hashed_state.accounts.get(address).ok_or_else(|| {
            eyre!("missing account transition for changed storage trie {address:?}")
        })?;
        let storage_root = if let Some(storage_trie) = trie.storage_trie_mut(address) {
            storage_trie.root()
        } else if hashed_state.storages.get(address).is_some_and(|storage| storage.wiped) {
            EMPTY_ROOT_HASH
        } else if let Some(value) = trie.get_account_value(address) {
            reth_trie_common::TrieAccount::decode(&mut &value[..])?.storage_root
        } else {
            EMPTY_ROOT_HASH
        };

        let leaf_update = match account {
            Some(account) if !account.is_empty() || storage_root != EMPTY_ROOT_HASH => {
                let mut encoded = Vec::new();
                account.into_trie_account(storage_root).encode(&mut encoded);
                LeafUpdate::Changed(encoded)
            }
            _ => LeafUpdate::Changed(Vec::new()),
        };
        if matches!(leaf_update, LeafUpdate::Changed(ref value) if value.is_empty()) {
            account_removals.insert(*address, leaf_update);
        } else {
            account_upserts.insert(*address, leaf_update);
        }
    }

    for account_updates in [&mut account_upserts, &mut account_removals] {
        let mut blinded = false;
        trie.trie_mut()
            .update_leaves(account_updates, |_key, _min_len| blinded = true)
            .map_err(|err| eyre!("failed to batch-update account trie: {err}"))?;
        if blinded {
            bail!("sidecar multiproof is structurally incomplete for an account transition");
        }
    }

    let state_root =
        trie.root().map_err(|err| eyre!("failed to compute trustless post-state root: {err}"))?;
    Ok(CompletedTransitionProof { multiproof, state_root })
}

fn apply_cache_transition_and_check(
    cache: &mut NetworkStateCache,
    accessed: &BlockAccessedState,
    block_number: u64,
    block_hash: B256,
    cache_policy_id: B256,
    expected_next_anchor: CacheAnchor,
) -> Result<(UpdateStats, CacheAnchor)> {
    let cache_update = cache.on_block_executed(block_number, accessed);
    let next_cache_anchor = cache.cache_anchor(block_number, block_hash, cache_policy_id);
    if next_cache_anchor != expected_next_anchor {
        cache.rollback_block(block_number).map_err(|rollback_err| {
            eyre!("next cache anchor mismatch; cache rollback also failed: {rollback_err}")
        })?;
        bail!(
            "next cache anchor mismatch: expected {expected_next_anchor:?}, got {next_cache_anchor:?}"
        );
    }
    Ok((cache_update, next_cache_anchor))
}

fn prefilter(
    block: &RecoveredBlock<BlockTy<EthPrimitives>>,
    expected_parent_state_root: B256,
    prev_cache: &NetworkStateCache,
    sidecar: &PartialStatelessSidecar,
) -> Result<()> {
    if sidecar.block_hash != block.hash() {
        bail!("sidecar block_hash mismatch");
    }
    if sidecar.parent_hash != block.parent_hash() {
        bail!("sidecar parent_hash mismatch");
    }
    if sidecar.block_number != block.number() {
        bail!("sidecar block_number mismatch");
    }
    if sidecar.parent_state_root != expected_parent_state_root {
        bail!(
            "sidecar parent_state_root mismatch: expected {expected_parent_state_root:?}, got {:?}",
            sidecar.parent_state_root
        );
    }

    let local_prev_anchor =
        prev_cache.cache_anchor(sidecar.cache_block, sidecar.parent_hash, sidecar.cache_policy_id);
    check_sidecar_context(sidecar, &local_prev_anchor)
        .map_err(|err| eyre!("sidecar cache context mismatch: {err:?}"))?;

    Ok(())
}

struct WitnessBackedStateProvider<'a> {
    cache: &'a NetworkStateCache,
    witness_accounts: HashMap<Address, Option<Account>>,
    witness_storage: HashMap<(Address, B256), U256>,
    witness_codes: HashMap<B256, Bytes>,
    witness_headers: HashMap<u64, B256>,
    block_number: u64,
}

impl WitnessBackedStateProvider<'_> {
    fn missing(label: &str) -> ProviderError {
        ProviderError::TrieWitnessError(label.to_string())
    }
}

impl EvmStateProvider for WitnessBackedStateProvider<'_> {
    fn basic_account(&self, address: &Address) -> ProviderResult<Option<Account>> {
        if let Some(entry) = self.cache.accounts().get(address) {
            if !entry.value.exists {
                return Ok(None)
            }
            return Ok(Some(Account {
                nonce: entry.value.nonce,
                balance: entry.value.balance,
                bytecode_hash: entry.value.code_hash,
            }));
        }

        if let Some(account) = self.witness_accounts.get(address) {
            return Ok(*account);
        }

        Err(Self::missing(&format!("missing account witness for {address:?}")))
    }

    fn block_hash(&self, number: u64) -> ProviderResult<Option<B256>> {
        if number >= self.block_number || number.saturating_add(256) < self.block_number {
            return Ok(None);
        }

        self.witness_headers
            .get(&number)
            .copied()
            .map(Some)
            .ok_or_else(|| Self::missing(&format!("missing ancestor header witness for {number}")))
    }

    fn bytecode_by_hash(&self, code_hash: &B256) -> ProviderResult<Option<Bytecode>> {
        if let Some(entry) = self.cache.codes().get(code_hash) {
            return Ok(Some(Bytecode::new_raw(entry.value.clone())));
        }

        if let Some(code) = self.witness_codes.get(code_hash) {
            return Ok(Some(Bytecode::new_raw(code.clone())));
        }

        Err(Self::missing(&format!("missing bytecode witness for {code_hash:?}")))
    }

    fn storage(&self, account: Address, storage_key: B256) -> ProviderResult<Option<U256>> {
        if let Some(entry) = self.cache.storage().get(&(account, storage_key)) {
            return Ok(Some(entry.value));
        }

        if let Some(value) = self.witness_storage.get(&(account, storage_key)) {
            return Ok(Some(*value));
        }

        Err(Self::missing(&format!(
            "missing storage witness for account={account:?}, slot={storage_key:?}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::{keccak256, map::HashMap as AlloyHashMap};
    use partial_stateless::{
        policy::{AccountData, LastNBlocksPolicy},
        witness_check::verify_state_proof_targets,
    };
    use reth_trie_common::{
        hash_builder::HashBuilder, proof::ProofRetainer, BranchNodeMasksMap, MultiProofTargets,
        Nibbles, StorageMultiProof,
    };

    fn account_trie_root(address: Address, account: Account, storage_root: B256) -> B256 {
        let path = Nibbles::unpack(keccak256(address));
        let mut builder = HashBuilder::default();
        builder.add_leaf(path, &alloy_rlp::encode(account.into_trie_account(storage_root)));
        builder.root()
    }

    fn single_account_proof(
        address: Address,
        account: Account,
        storage_root: B256,
    ) -> (B256, MultiProof) {
        let path = Nibbles::unpack(keccak256(address));
        let mut builder =
            HashBuilder::default().with_proof_retainer(ProofRetainer::from_iter([path]));
        builder.add_leaf(path, &alloy_rlp::encode(account.into_trie_account(storage_root)));
        let root = builder.root();
        let account_subtree = builder.take_proof_nodes();
        (root, MultiProof { account_subtree, ..Default::default() })
    }

    fn account_multiproof(
        entries: &[(B256, Account)],
        targets: impl IntoIterator<Item = B256>,
    ) -> (B256, MultiProof) {
        let target_paths = targets.into_iter().map(Nibbles::unpack);
        let mut builder =
            HashBuilder::default().with_proof_retainer(ProofRetainer::from_iter(target_paths));
        for (path, account) in entries.iter().map(|(key, account)| {
            (Nibbles::unpack(*key), account.into_trie_account(EMPTY_ROOT_HASH))
        }) {
            builder.add_leaf(path, &alloy_rlp::encode(account));
        }
        let root = builder.root();
        (root, MultiProof { account_subtree: builder.take_proof_nodes(), ..Default::default() })
    }

    fn addresses_with_hashed_first_nibble(nibble: u8, count: usize) -> Vec<(Address, B256)> {
        let mut result = Vec::with_capacity(count);
        for candidate in 0u64.. {
            let mut bytes = [0u8; 20];
            bytes[12..].copy_from_slice(&candidate.to_be_bytes());
            let address = Address::from(bytes);
            let hashed = keccak256(address);
            if hashed[0] >> 4 == nibble {
                result.push((address, hashed));
                if result.len() == count {
                    break;
                }
            }
        }
        result
    }

    fn slots_with_hashed_first_nibble(nibble: u8, count: usize) -> Vec<(U256, B256)> {
        let mut result = Vec::with_capacity(count);
        for candidate in 0u64.. {
            let slot = U256::from(candidate);
            let hashed = keccak256(B256::from(slot));
            if hashed[0] >> 4 == nibble {
                result.push((slot, hashed));
                if result.len() == count {
                    break;
                }
            }
        }
        result
    }

    fn storage_multiproof(
        entries: &[(B256, U256)],
        targets: impl IntoIterator<Item = B256>,
    ) -> (B256, StorageMultiProof) {
        let target_paths = targets.into_iter().map(Nibbles::unpack);
        let mut builder =
            HashBuilder::default().with_proof_retainer(ProofRetainer::from_iter(target_paths));
        for (key, value) in entries {
            builder.add_leaf(Nibbles::unpack(*key), &alloy_rlp::encode_fixed_size(value));
        }
        let root = builder.root();
        (
            root,
            StorageMultiProof {
                root,
                subtree: builder.take_proof_nodes(),
                branch_node_masks: BranchNodeMasksMap::default(),
            },
        )
    }

    fn proof_for_storage_targets(
        address: Address,
        account: Account,
        entries: &[(B256, U256)],
        targets: &MultiProofTargets,
    ) -> (B256, MultiProof) {
        let hashed_address = keccak256(address);
        let target_slots =
            targets.get(&hashed_address).into_iter().flat_map(|slots| slots.iter().copied());
        let (storage_root, storage_proof) = storage_multiproof(entries, target_slots);
        let (state_root, mut proof) = single_account_proof(address, account, storage_root);
        proof.storages.insert(hashed_address, storage_proof);
        (state_root, proof)
    }

    #[test]
    fn execution_witness_v2_converts_back_to_path_addressed_multiproof() {
        let address = Address::repeat_byte(0x44);
        let hashed_address = keccak256(address);
        let account = Account { nonce: 7, balance: U256::from(99), bytecode_hash: None };
        let slots = slots_with_hashed_first_nibble(3, 16);
        let mut entries = slots
            .iter()
            .enumerate()
            .map(|(index, (_, key))| (*key, U256::from(index + 1)))
            .collect::<Vec<_>>();
        entries.sort_by_key(|(key, _)| *key);
        let targets = MultiProofTargets::account_with_slots(
            hashed_address,
            slots.iter().map(|(_, key)| *key),
        );
        let (parent_root, proof) = proof_for_storage_targets(address, account, &entries, &targets);
        let witness = proof
            .account_subtree
            .values()
            .chain(proof.storages.values().flat_map(|storage| storage.subtree.values()))
            .cloned()
            .map(|node| (keccak256(&node), node))
            .collect::<B256Map<_>>();

        let decoded = DecodedMultiProofV2::from_witness(parent_root, &witness).unwrap();
        let converted = decoded_v2_to_multiproof(decoded, &proof).unwrap();
        assert!(converted.account_subtree.contains_key(&Nibbles::default()));
        assert!(converted.storages[&hashed_address].subtree.len() > 1);

        let mut trie = SparseStateTrie::new();
        trie.reveal_multiproof(converted).unwrap();
        assert_eq!(trie.root().unwrap(), parent_root);
    }

    #[test]
    fn supplemental_storage_fragment_reuses_authenticated_base_root() {
        let hashed_address = B256::repeat_byte(0x45);
        let storage_root = B256::repeat_byte(0x67);
        let fragment_path = Nibbles::from_nibbles([0x1]);
        let decoded = DecodedMultiProofV2 {
            storage_proofs: B256Map::from_iter([(
                hashed_address,
                vec![ProofTrieNodeV2 {
                    path: fragment_path,
                    node: TrieNodeV2::Leaf(reth_trie_common::LeafNode::new(
                        Nibbles::from_nibbles([0x2]),
                        vec![0x01],
                    )),
                    masks: None,
                }],
            )]),
            ..Default::default()
        };
        let mut base = MultiProof::default();
        base.storages.insert(
            hashed_address,
            StorageMultiProof {
                root: storage_root,
                subtree: ProofNodes::default(),
                branch_node_masks: BranchNodeMasksMap::default(),
            },
        );

        let converted = decoded_v2_to_multiproof(decoded, &base).unwrap();

        assert_eq!(converted.storages[&hashed_address].root, storage_root);
        assert!(converted.storages[&hashed_address].subtree.contains_key(&fragment_path));
    }

    #[test]
    fn trustless_root_recomputes_warm_account_write() {
        let address = Address::repeat_byte(0x11);
        let original = Account { nonce: 1, balance: U256::from(10), bytecode_hash: None };
        let updated = Account {
            nonce: 2,
            balance: U256::from(25),
            bytecode_hash: Some(B256::repeat_byte(0xaa)),
        };
        let (parent_root, proof) = single_account_proof(address, original, EMPTY_ROOT_HASH);
        let bundle = BundleState::builder(1..=1)
            .state_original_account_info(address, original.into())
            .state_present_account_info(address, updated.into())
            .build();

        let actual = compute_trustless_state_root(parent_root, &bundle, proof).unwrap();
        let expected = account_trie_root(address, updated, EMPTY_ROOT_HASH);

        assert_eq!(actual, expected);
    }

    #[test]
    fn warm_read_only_storage_is_not_replayed_as_a_trie_write() {
        let address = Address::repeat_byte(0x12);
        let account = Account { nonce: 1, balance: U256::from(10), bytecode_hash: None };
        let slot = U256::from(7);
        let value = U256::from(3);
        let storage_entries = [(keccak256(B256::from(slot)), value)];
        let storage_root = storage_multiproof(&storage_entries, core::iter::empty()).0;
        let (parent_root, proof) = single_account_proof(address, account, storage_root);
        let bundle = BundleState::builder(1..=1)
            .state_original_account_info(address, account.into())
            .state_present_account_info(address, account.into())
            .state_storage(address, AlloyHashMap::from_iter([(slot, (value, value))]))
            .build();

        assert!(hashed_write_state_from_bundle(&bundle).is_empty());
        assert_eq!(compute_trustless_state_root(parent_root, &bundle, proof).unwrap(), parent_root);
    }

    #[test]
    fn account_write_preserves_root_without_proving_unchanged_warm_storage() {
        let address = Address::repeat_byte(0x13);
        let original = Account { nonce: 1, balance: U256::from(10), bytecode_hash: None };
        let updated = Account { nonce: 2, ..original };
        let slot = U256::from(7);
        let value = U256::from(3);
        let storage_entries = [(keccak256(B256::from(slot)), value)];
        let storage_root = storage_multiproof(&storage_entries, core::iter::empty()).0;
        let (parent_root, proof) = single_account_proof(address, original, storage_root);
        assert!(proof.storages.is_empty());
        let bundle = BundleState::builder(1..=1)
            .state_original_account_info(address, original.into())
            .state_present_account_info(address, updated.into())
            .state_storage(address, AlloyHashMap::from_iter([(slot, (value, value))]))
            .build();

        let hashed_state = hashed_write_state_from_bundle(&bundle);
        assert_eq!(hashed_state.accounts.len(), 1);
        assert!(hashed_state.storages.is_empty());
        assert_eq!(
            compute_trustless_state_root(parent_root, &bundle, proof).unwrap(),
            account_trie_root(address, updated, storage_root)
        );
    }

    #[test]
    fn account_write_ignores_empty_storage_proof_placeholder() {
        let address = Address::repeat_byte(0x14);
        let hashed_address = keccak256(address);
        let original = Account { nonce: 1, balance: U256::from(10), bytecode_hash: None };
        let updated = Account { nonce: 2, ..original };
        let storage_entries = slots_with_hashed_first_nibble(3, 2)
            .into_iter()
            .enumerate()
            .map(|(index, (_, key))| (key, U256::from(index + 1)))
            .collect::<Vec<_>>();
        let (storage_root, storage_placeholder) =
            storage_multiproof(&storage_entries, core::iter::empty());

        let (parent_root, mut proof) = single_account_proof(address, original, storage_root);
        proof.storages.insert(hashed_address, storage_placeholder);
        let mut unfiltered_trie = SparseStateTrie::new();
        assert!(
            unfiltered_trie.reveal_multiproof(proof.clone()).is_err(),
            "an account-only storage extension cannot reveal its hashed child"
        );
        let bundle = BundleState::builder(1..=1)
            .state_original_account_info(address, original.into())
            .state_present_account_info(address, updated.into())
            .build();
        let hashed_state = hashed_write_state_from_bundle(&bundle);

        let completed = apply_hashed_state_transition(parent_root, &hashed_state, proof).unwrap();

        assert!(completed.multiproof.storages.contains_key(&hashed_address));
        assert_eq!(completed.state_root, account_trie_root(address, updated, storage_root));
    }

    #[test]
    fn account_creation_uses_authenticated_nonexistence_path() {
        let existing_address = Address::repeat_byte(0x11);
        let created_address = Address::repeat_byte(0x22);
        let existing = Account { nonce: 1, balance: U256::from(10), bytecode_hash: None };
        let created = Account {
            nonce: 1,
            balance: U256::from(20),
            bytecode_hash: Some(B256::repeat_byte(0xbb)),
        };
        let existing_key = keccak256(existing_address);
        let created_key = keccak256(created_address);
        let entries = vec![(existing_key, existing)];
        let (parent_root, proof) = account_multiproof(&entries, core::iter::once(created_key));
        let bundle = BundleState::builder(1..=1)
            .state_present_account_info(created_address, created.into())
            .build();

        let actual = compute_trustless_state_root(parent_root, &bundle, proof).unwrap();
        let mut expected_entries = vec![(existing_key, existing), (created_key, created)];
        expected_entries.sort_by_key(|(key, _)| *key);
        let expected = account_multiproof(&expected_entries, core::iter::empty()).0;

        assert_eq!(actual, expected);
    }

    #[test]
    fn account_deletion_requires_completed_sibling_structure() {
        let (deleted_address, deleted_key) = addresses_with_hashed_first_nibble(1, 1)[0];
        let sibling_accounts = addresses_with_hashed_first_nibble(2, 16);
        let account = Account { nonce: 1, balance: U256::from(10), bytecode_hash: None };
        let mut entries = Vec::with_capacity(1 + sibling_accounts.len());
        entries.push((deleted_key, account));
        entries.extend(sibling_accounts.iter().map(|(_, key)| (*key, account)));
        entries.sort_by_key(|(key, _)| *key);

        let (parent_root, initial_proof) =
            account_multiproof(&entries, core::iter::once(deleted_key));
        let bundle = BundleState::builder(1..=1)
            .state_original_account_info(deleted_address, account.into())
            .build();
        let hashed_state = hashed_write_state_from_bundle(&bundle);

        let incomplete =
            apply_hashed_state_transition(parent_root, &hashed_state, initial_proof.clone());
        assert!(incomplete.is_err(), "ordinary target proof must expose the blind sibling gap");

        let complete_proof = account_multiproof(&entries, entries.iter().map(|(key, _)| *key)).1;
        let completed =
            apply_hashed_state_transition(parent_root, &hashed_state, complete_proof).unwrap();

        let remaining =
            entries.iter().copied().filter(|(key, _)| *key != deleted_key).collect::<Vec<_>>();
        let expected_root = account_multiproof(&remaining, core::iter::empty()).0;
        assert_eq!(completed.state_root, expected_root);
        assert_eq!(
            compute_trustless_state_root(parent_root, &bundle, completed.multiproof).unwrap(),
            expected_root
        );
    }

    #[test]
    fn storage_deletion_requires_completed_sibling_structure() {
        let address = Address::repeat_byte(0x22);
        let hashed_address = keccak256(address);
        let account = Account { nonce: 1, balance: U256::from(10), bytecode_hash: None };
        let (deleted_slot, deleted_key) = slots_with_hashed_first_nibble(1, 1)[0];
        let sibling_slots = slots_with_hashed_first_nibble(2, 16);
        let mut entries = Vec::with_capacity(1 + sibling_slots.len());
        entries.push((deleted_key, U256::from(1)));
        entries.extend(
            sibling_slots.iter().enumerate().map(|(index, (_, key))| (*key, U256::from(index + 2))),
        );
        entries.sort_by_key(|(key, _)| *key);

        let initial_targets =
            MultiProofTargets::account_with_slots(hashed_address, core::iter::once(deleted_key));
        let (parent_root, initial_proof) =
            proof_for_storage_targets(address, account, &entries, &initial_targets);
        let bundle = BundleState::builder(1..=1)
            .state_original_account_info(address, account.into())
            .state_present_account_info(address, account.into())
            .state_storage(
                address,
                AlloyHashMap::from_iter([(deleted_slot, (U256::from(1), U256::ZERO))]),
            )
            .build();
        let hashed_state = hashed_write_state_from_bundle(&bundle);

        let incomplete =
            apply_hashed_state_transition(parent_root, &hashed_state, initial_proof.clone());
        assert!(incomplete.is_err(), "ordinary storage proof must expose the blind sibling gap");

        let complete_targets = MultiProofTargets::account_with_slots(
            hashed_address,
            entries.iter().map(|(key, _)| *key),
        );
        let complete_proof =
            proof_for_storage_targets(address, account, &entries, &complete_targets).1;
        let completed =
            apply_hashed_state_transition(parent_root, &hashed_state, complete_proof).unwrap();

        let remaining =
            entries.iter().copied().filter(|(key, _)| *key != deleted_key).collect::<Vec<_>>();
        let expected_storage_root = storage_multiproof(&remaining, core::iter::empty()).0;
        let expected_root = account_trie_root(address, account, expected_storage_root);
        assert_eq!(completed.state_root, expected_root);
        assert_eq!(
            compute_trustless_state_root(parent_root, &bundle, completed.multiproof).unwrap(),
            expected_root
        );
    }

    #[test]
    fn state_proof_rejects_unproved_warm_write_path() {
        let proved_address = Address::repeat_byte(0x11);
        let proved_first_nibble = keccak256(proved_address)[0] >> 4;
        let unproved_address = (0u8..=u8::MAX)
            .map(Address::with_last_byte)
            .find(|address| keccak256(address)[0] >> 4 != proved_first_nibble)
            .expect("an address in another root branch");
        let account = Account { nonce: 1, balance: U256::from(10), bytecode_hash: None };
        let proved_path = Nibbles::unpack(keccak256(proved_address));
        let unproved_path = Nibbles::unpack(keccak256(unproved_address));
        let mut entries = [(proved_path, account), (unproved_path, account)];
        entries.sort_by_key(|entry| entry.0);
        let mut builder =
            HashBuilder::default().with_proof_retainer(ProofRetainer::from_iter([proved_path]));
        for (path, account) in entries {
            builder.add_leaf(path, &alloy_rlp::encode(account.into_trie_account(EMPTY_ROOT_HASH)));
        }
        let parent_root = builder.root();
        let proof =
            MultiProof { account_subtree: builder.take_proof_nodes(), ..Default::default() };
        let targets = StateTargetSet {
            accounts: vec![unproved_address],
            storage: vec![],
            code_hashes: vec![],
        };

        assert!(verify_state_proof_targets(&proof, parent_root, &targets).is_err());
    }

    #[test]
    fn trustless_root_recomputes_warm_storage_write() {
        let address = Address::repeat_byte(0x22);
        let account = Account { nonce: 1, balance: U256::from(10), bytecode_hash: None };
        let slot = U256::from(7);
        let raw_slot = B256::from(slot);
        let hashed_slot = keccak256(raw_slot);
        let slot_path = Nibbles::unpack(hashed_slot);
        let original_value = U256::from(3);
        let updated_value = U256::from(9);

        let mut storage_builder =
            HashBuilder::default().with_proof_retainer(ProofRetainer::from_iter([slot_path]));
        storage_builder.add_leaf(slot_path, &alloy_rlp::encode_fixed_size(&original_value));
        let storage_root = storage_builder.root();
        let storage_subtree = storage_builder.take_proof_nodes();

        let (parent_root, mut proof) = single_account_proof(address, account, storage_root);
        proof.storages.insert(
            keccak256(address),
            StorageMultiProof {
                root: storage_root,
                subtree: storage_subtree,
                branch_node_masks: BranchNodeMasksMap::default(),
            },
        );

        let bundle = BundleState::builder(1..=1)
            .state_original_account_info(address, account.into())
            .state_present_account_info(address, account.into())
            .state_storage(
                address,
                AlloyHashMap::from_iter([(slot, (original_value, updated_value))]),
            )
            .build();

        let actual = compute_trustless_state_root(parent_root, &bundle, proof).unwrap();

        let mut expected_storage_builder = HashBuilder::default();
        expected_storage_builder.add_leaf(slot_path, &alloy_rlp::encode_fixed_size(&updated_value));
        let expected_storage_root = expected_storage_builder.root();
        let expected = account_trie_root(address, account, expected_storage_root);

        assert_eq!(actual, expected);
    }

    #[test]
    fn storage_creation_uses_authenticated_nonexistence_path() {
        let address = Address::repeat_byte(0x33);
        let hashed_address = keccak256(address);
        let account = Account { nonce: 1, balance: U256::from(10), bytecode_hash: None };
        let existing_slot = U256::from(7);
        let created_slot = U256::from(8);
        let existing_key = keccak256(B256::from(existing_slot));
        let created_key = keccak256(B256::from(created_slot));
        let entries = vec![(existing_key, U256::from(3))];
        let targets =
            MultiProofTargets::account_with_slots(hashed_address, core::iter::once(created_key));
        let (parent_root, proof) = proof_for_storage_targets(address, account, &entries, &targets);
        let bundle = BundleState::builder(1..=1)
            .state_original_account_info(address, account.into())
            .state_present_account_info(address, account.into())
            .state_storage(
                address,
                AlloyHashMap::from_iter([(created_slot, (U256::ZERO, U256::from(9)))]),
            )
            .build();

        let actual = compute_trustless_state_root(parent_root, &bundle, proof).unwrap();
        let mut expected_entries =
            vec![(existing_key, U256::from(3)), (created_key, U256::from(9))];
        expected_entries.sort_by_key(|(key, _)| *key);
        let expected_storage_root = storage_multiproof(&expected_entries, core::iter::empty()).0;
        let expected = account_trie_root(address, account, expected_storage_root);

        assert_eq!(actual, expected);
    }

    #[test]
    fn first_storage_write_retains_canonical_empty_trie_proof() {
        let address = Address::repeat_byte(0x34);
        let hashed_address = keccak256(address);
        let account = Account { nonce: 1, balance: U256::from(10), bytecode_hash: None };
        let slot = U256::from(7);
        let value = U256::from(9);
        let hashed_slot = keccak256(B256::from(slot));
        let (parent_root, mut proof) = single_account_proof(address, account, EMPTY_ROOT_HASH);
        proof.storages.insert(hashed_address, StorageMultiProof::empty());
        let bundle = BundleState::builder(1..=1)
            .state_original_account_info(address, account.into())
            .state_present_account_info(address, account.into())
            .state_storage(address, AlloyHashMap::from_iter([(slot, (U256::ZERO, value))]))
            .build();

        let completed = apply_hashed_state_transition(
            parent_root,
            &hashed_write_state_from_bundle(&bundle),
            proof,
        )
        .unwrap();

        assert!(completed.multiproof.storages.contains_key(&hashed_address));
        let expected_storage_root = storage_multiproof(&[(hashed_slot, value)], []).0;
        assert_eq!(
            completed.state_root,
            account_trie_root(address, account, expected_storage_root)
        );
    }

    #[test]
    fn recreated_account_storage_starts_from_empty_trie() {
        let address = Address::repeat_byte(0x35);
        let hashed_address = keccak256(address);
        let original = Account { nonce: 7, balance: U256::from(10), bytecode_hash: None };
        let recreated = Account { nonce: 1, balance: U256::from(4), bytecode_hash: None };
        let old_slot = U256::from(7);
        let new_slot = U256::from(8);
        let old_entries = [(keccak256(B256::from(old_slot)), U256::from(3))];
        let targets = MultiProofTargets::account_with_slots(
            hashed_address,
            core::iter::once(keccak256(B256::from(new_slot))),
        );
        let (parent_root, proof) =
            proof_for_storage_targets(address, original, &old_entries, &targets);
        let mut bundle = BundleState::builder(1..=1)
            .state_original_account_info(address, original.into())
            .state_present_account_info(address, recreated.into())
            .state_storage(
                address,
                AlloyHashMap::from_iter([(new_slot, (U256::ZERO, U256::from(9)))]),
            )
            .build();
        bundle.state.get_mut(&address).unwrap().status =
            revm::database::states::AccountStatus::DestroyedChanged;

        let completed = apply_hashed_state_transition(
            parent_root,
            &hashed_write_state_from_bundle(&bundle),
            proof,
        )
        .unwrap();

        let new_storage_root =
            storage_multiproof(&[(keccak256(B256::from(new_slot)), U256::from(9))], []).0;
        assert_eq!(completed.state_root, account_trie_root(address, recreated, new_storage_root));
    }

    #[test]
    fn next_anchor_mismatch_rolls_back_cache_transition() {
        let mut cache = NetworkStateCache::new(
            Box::new(LastNBlocksPolicy::new(60)),
            Box::new(LastNBlocksPolicy::new(30)),
        );
        cache.on_block_executed(99, &BlockAccessedState::default());
        let root_before = cache.cache_root();
        let address = Address::repeat_byte(0x11);
        let mut accessed = BlockAccessedState::default();
        accessed.accounts.insert(
            address,
            AccountData { exists: true, nonce: 1, balance: U256::from(10), code_hash: None },
        );

        let _error = apply_cache_transition_and_check(
            &mut cache,
            &accessed,
            100,
            B256::repeat_byte(0x22),
            B256::repeat_byte(0x33),
            CacheAnchor {
                block_number: 100,
                block_hash: B256::repeat_byte(0x22),
                cache_policy_id: B256::repeat_byte(0x33),
                cache_root: B256::ZERO,
            },
        )
        .expect_err("wrong next cache root must fail");

        assert_eq!(cache.current_block(), 99);
        assert_eq!(cache.cache_root(), root_before);
        assert!(!cache.contains_account(&address));
    }

    #[test]
    fn witness_provider_preserves_cached_account_absence() {
        let address = Address::repeat_byte(0x68);
        let mut cache = NetworkStateCache::new(
            Box::new(LastNBlocksPolicy::new(60)),
            Box::new(LastNBlocksPolicy::new(30)),
        );
        let mut accessed = BlockAccessedState::default();
        accessed.accounts.insert(
            address,
            AccountData { exists: false, nonce: 0, balance: U256::ZERO, code_hash: None },
        );
        cache.on_block_executed(1, &accessed);

        let provider = WitnessBackedStateProvider {
            cache: &cache,
            witness_accounts: HashMap::new(),
            witness_storage: HashMap::new(),
            witness_codes: HashMap::new(),
            witness_headers: HashMap::new(),
            block_number: 2,
        };

        assert_eq!(provider.basic_account(&address).unwrap(), None);
    }
}
