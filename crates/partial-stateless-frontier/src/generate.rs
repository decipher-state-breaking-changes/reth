//! Generating every arm's real sidecar for one recorded block, and validating each one.
//!
//! An arm is a cache policy, or the no-cache Weak baseline. The order below is the whole method,
//! and each step exists because the one before it cannot be trusted to stand alone:
//!
//! 1. **Admit the payload.** The recorded `ExecutionData` goes through the same untrusted-input
//!    boundary a standalone validator uses, so the block this run works on is one it derived from
//!    the payload rather than one the record asserted. The record's own block hash is then checked
//!    against it, which is how a corrupted or mislabelled record is caught here rather than three
//!    steps later.
//! 2. **Execute once, database-free, against the recorded full witness.** This produces the post
//!    state and the access set *this* run observed. Both are then used by every arm, and neither is
//!    taken from the record. The sidecar that execution rides is the Weak one by construction, so
//!    the baseline arm has nothing separate to build.
//! 3. **Compare that access set against the recorded one.** The producer captured the set from a
//!    live Engine; this one came from a witness-backed re-execution on another machine. Agreement
//!    is the evidence that the corpus describes the block it claims to.
//! 4. **Build each policy arm's sidecar** through the same construction core a live builder uses,
//!    with the recorded witness standing in for the state database.
//! 5. **Validate each sidecar** against that arm's own validator pair — a pair that has never seen
//!    anything but sidecars — so the result is a check and not a restatement.
//!
//! Two things here must not be read as measurements of a production system.
//!
//! A **production builder's proof latency** is not measured: selecting nodes out of a decoded
//! witness in memory is not generating a multiproof from a state database, and the two differ by
//! orders of magnitude. An **absolute standalone validation latency** is not measured either: the
//! per-arm timer opens at the sidecar decode, which covers every cost that varies with the cache
//! policy but none of the delivery path a live consumer pays. Sizes, node sets, miss sets, cache
//! footprints, and the policy-dependent part of validation cost are properties of the sidecar, and
//! are exactly what this is for.

use crate::{
    policy::{rotated_order, ArmKind, PolicyState},
    source::RecordedFullWitnessSource,
};
use alloy_consensus::Header;
use alloy_primitives::B256;
use alloy_rpc_types_engine::ExecutionData;
use partial_stateless::{
    full_witness_sidecar_from_nodes, measure_witness_trim, policy_dataset::PolicyDatasetRecord,
    BlockAccessedState, BlockTransitionRef, CacheAwareFlatBuild, PartialStatelessSidecar,
    PolicySidecarBuild, TransitionBuildContext, TrieBranchCensus, WitnessTrimStats,
};
use partial_stateless_validator::{
    verify_and_apply_sidecar, verify_and_apply_sidecar_with_oracle, PostStateRootOracle,
    SidecarReexecLimits, TrieCacheDisposition, UntrustedAdmission, ValidatorRules,
};
use reth_consensus::{Consensus, FullConsensus};
use reth_ethereum_primitives::EthPrimitives;
use reth_evm::ConfigureEvm;
use reth_primitives_traits::{AlloyBlockHeader, BlockTy, RecoveredBlock, SealedHeader};
use reth_trie_common::HashedPostState;
use std::{cell::RefCell, time::Instant};

/// Takes the post state out of a validation without ever answering it.
///
/// The validator core hands its hashed post state to whatever root oracle it was given, on the way
/// to an optional second opinion it does not need. An offline generator needs that value — every
/// policy's transition is computed from it — and this is the seam that already exists for handing
/// it over. Returning `None` keeps the core on its own root, which is the only one this side has.
#[derive(Debug, Default)]
struct PostStateTap(RefCell<Option<HashedPostState>>);

impl PostStateTap {
    fn take(&self) -> Option<HashedPostState> {
        self.0.borrow_mut().take()
    }
}

impl PostStateRootOracle for PostStateTap {
    fn post_state_root(&self, post_state: HashedPostState) -> eyre::Result<Option<B256>> {
        *self.0.borrow_mut() = Some(post_state);
        // Deliberately no root. This is a tap, not an oracle: a second opinion computed on this
        // side would be computed from the same witness the first one came from.
        Ok(None)
    }
}

/// One arm's result for one block.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PolicyBlockResult {
    /// `weak` or `account/storage`, as the run named it.
    pub policy: String,
    /// Position this policy occupied in this block's rotation, from zero.
    pub rotation_slot: usize,
    /// Serialized sidecar size, which is the figure a bandwidth claim is about.
    pub sidecar_bytes: usize,
    /// keccak256 of the serialized sidecar, so two hosts can be compared without shipping either.
    pub sidecar_digest: B256,
    /// The sidecar's own witness commitment.
    pub witness_commitment: B256,
    /// Parent-state nodes in the witness.
    pub witness_nodes: usize,
    /// Bytes those nodes occupy.
    pub witness_node_bytes: usize,
    /// Bytecode bytes the sidecar carries.
    pub witness_code_bytes: usize,
    /// Accounts in the miss manifest.
    pub missed_accounts: usize,
    /// Storage slots in the miss manifest.
    pub missed_storage: usize,
    /// Bytecodes in the miss manifest.
    pub missed_codes: usize,
    /// The cache state this sidecar applies to.
    pub prev_cache_root: B256,
    /// The cache state applying it produces.
    pub next_cache_root: B256,
    /// Estimated resident size of the policy's flat cache after this block.
    pub cache_bytes: usize,
    /// Estimated resident size of the policy's trie cache after this block.
    pub trie_cache_bytes: usize,
    /// Proof-source calls the transition made. Zero-cost here; recorded to compare shapes.
    pub provider_calls: usize,
    /// Structural discovery rounds the transition needed.
    pub structural_rounds: usize,
    /// Wall time from the serialized sidecar to a committed cache transition.
    ///
    /// **Not** a whole standalone validation, and the name says which part it is. It opens at the
    /// sidecar decode and closes at the cache commit, covering everything that varies with the
    /// cache policy: decoding a witness whose size the policy decided, materializing it,
    /// re-executing against it, computing the root, and committing. It excludes what does *not*
    /// vary — payload decode, sender recovery, and pre-execution consensus are the same work on
    /// the same block for every arm, and are reported once per block as
    /// [`BlockResult::block_admission_us`].
    ///
    /// A whole-block standalone latency is the sum, and an *absolute* standalone latency is
    /// neither: it is measured by a live `ps-replay` run, whose boundary opens at the frame read.
    pub sidecar_decode_and_commit_us: u64,
    /// The decode half of the figure above, which is the part that scales with witness size.
    pub sidecar_decode_us: u64,
    /// Wall time the offline build took, or `None` for an arm whose sidecar was built once for the
    /// whole block rather than inside the rotation.
    ///
    /// `None` on the Weak arm, and the distinction is not cosmetic: Weak's witness is the
    /// policy-neutral full one this block was executed against, built before the rotation started.
    /// Reporting that build under a rotation slot would attribute a cost to a position it never
    /// occupied. **Not** a production builder latency either way — see the module docs.
    pub offline_build_us: Option<u64>,
    /// How much of this sidecar's witness the validator's own trie cache already reveals,
    /// measured against the parent generation the sidecar applies to. Present only when the run
    /// asked for trie diagnostics; always absent on the Weak arm, which retains no trie.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub witness_trim: Option<WitnessTrimStats>,
    /// Branch child-slot occupancy of this arm's trie cache after the block committed. Present
    /// only when the run asked for trie diagnostics; absent on the Weak arm.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch_census: Option<BranchCensusReport>,
}

/// Serializable form of [`TrieBranchCensus`], flattened for the JSONL stream.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct BranchCensusReport {
    /// Account-trie branch nodes.
    pub account_branches: u64,
    /// Account-trie child slots present in state masks.
    pub account_present_slots: u64,
    /// Account-trie child slots holding a blinded hash.
    pub account_blinded_slots: u64,
    /// Account-trie branches per depth bucket, shallowest first.
    pub account_branches_by_depth: [u64; 9],
    /// Account-trie blinded slots per depth bucket.
    pub account_blinded_by_depth: [u64; 9],
    /// Storage-trie branch nodes, over every revealed storage trie.
    pub storage_branches: u64,
    /// Storage-trie child slots present in state masks.
    pub storage_present_slots: u64,
    /// Storage-trie child slots holding a blinded hash.
    pub storage_blinded_slots: u64,
    /// Storage-trie branches per depth bucket.
    pub storage_branches_by_depth: [u64; 9],
    /// Storage-trie blinded slots per depth bucket.
    pub storage_blinded_by_depth: [u64; 9],
    /// Revealed storage tries the storage census covers.
    pub storage_tries: u64,
}

impl From<TrieBranchCensus> for BranchCensusReport {
    fn from(census: TrieBranchCensus) -> Self {
        Self {
            account_branches: census.account.branches,
            account_present_slots: census.account.present_slots,
            account_blinded_slots: census.account.blinded_slots,
            account_branches_by_depth: census.account.branches_by_depth,
            account_blinded_by_depth: census.account.blinded_by_depth,
            storage_branches: census.storage.branches,
            storage_present_slots: census.storage.present_slots,
            storage_blinded_slots: census.storage.blinded_slots,
            storage_branches_by_depth: census.storage.branches_by_depth,
            storage_blinded_by_depth: census.storage.blinded_by_depth,
            storage_tries: census.storage_tries,
        }
    }
}

/// Everything one recorded block produced.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BlockResult {
    /// Canonical block number.
    pub block_number: u64,
    /// Canonical block hash, as this run derived it from the payload.
    pub block_hash: B256,
    /// Whether this block counts toward the reported population, or is warm-up.
    pub measured: bool,
    /// Accounts the block accessed.
    pub accessed_accounts: usize,
    /// Storage slots the block accessed.
    pub accessed_storage: usize,
    /// Bytecodes the block accessed.
    pub accessed_codes: usize,
    /// Nodes in the recorded policy-neutral full witness.
    pub full_witness_nodes: usize,
    /// Bytes those nodes occupy.
    pub full_witness_node_bytes: usize,
    /// Wall time to decode this block's payload and admit it: layout checks, pre-execution
    /// consensus, and sender recovery.
    ///
    /// Paid once per block and identical for every arm, because it is the same block. Reported so
    /// a whole-block standalone figure can be assembled by addition rather than by an arm's timer
    /// silently including or excluding it.
    pub block_admission_us: u64,
    /// One entry per arm, in the order the rotation visited them.
    pub policies: Vec<PolicyBlockResult>,
}

/// The rules and limits one generator run works under.
pub struct GeneratorRules<'a, Evm, C: ?Sized, ChainSpec> {
    /// Consensus and EVM configuration, built once for the run.
    pub validator: ValidatorRules<'a, Evm, C>,
    /// The untrusted-input boundary recorded payloads enter through.
    pub admission: &'a UntrustedAdmission<'a, ChainSpec, C>,
    /// Bounds on sidecar witness decoding.
    pub limits: &'a SidecarReexecLimits,
    /// Whether to measure witness-trim potential and the branch-slot census per arm and block.
    ///
    /// Both walk every revealed node of the arm's trie cache, so they are opt-in: a run carrying
    /// them reports the same bytes and miss sets but its timing figures are not comparable to a
    /// run without them.
    pub trie_diagnostics: bool,
}

/// What the previous block left behind for the next one.
#[derive(Debug)]
pub struct ChainCursor {
    /// The header of the last block this run admitted, which is the next block's parent.
    pub accepted_parent: SealedHeader,
}

impl ChainCursor {
    /// Enters a corpus at its first record, using the parent header the record carries.
    ///
    /// The header is checked against the parent hash before it is trusted for anything. It is the
    /// one input a run cannot derive for itself, so it is the one input worth hashing on arrival.
    pub fn enter(record: &PolicyDatasetRecord) -> eyre::Result<Self> {
        let mut raw = record.body.parent_header.as_ref();
        let header = <Header as alloy_rlp::Decodable>::decode(&mut raw)
            .map_err(|err| eyre::eyre!("the first record's parent header did not decode: {err}"))?;
        let sealed = SealedHeader::seal_slow(header);
        if sealed.hash() != record.body.parent_hash {
            eyre::bail!(
                "the first record's parent header hashes to {:?}, but the record names parent {:?}",
                sealed.hash(),
                record.body.parent_hash
            )
        }
        Ok(Self { accepted_parent: sealed })
    }
}

/// Generates and validates every policy's sidecar for one recorded block.
///
/// `measured` decides only whether the result counts toward the reported population; a warm-up
/// block does exactly the same work, because a cache that skipped it would not be warm.
pub fn generate_block<Evm, C, ChainSpec>(
    rules: &GeneratorRules<'_, Evm, C, ChainSpec>,
    record: &PolicyDatasetRecord,
    policies: &mut [PolicyState],
    cursor: &mut ChainCursor,
    block_index: usize,
    measured: bool,
) -> eyre::Result<BlockResult>
where
    Evm: ConfigureEvm<Primitives = EthPrimitives>,
    C: FullConsensus<EthPrimitives>
        + Consensus<alloy_consensus::Block<reth_ethereum_primitives::TransactionSigned>>
        + ?Sized,
    ChainSpec: reth_chainspec::EthereumHardforks,
{
    let body = &record.body;

    // 1. Admission. The block this run works on is the one the payload produced.
    let payload_json = body
        .payload_json
        .as_deref()
        .ok_or_else(|| eyre::eyre!("record for block {} carries no payload", body.block_number))?;
    // Opened at the payload decode rather than at the admission call, so the figure covers
    // everything a standalone validator pays before it ever looks at a sidecar.
    let admission_start = Instant::now();
    let payload: ExecutionData = serde_json::from_slice(payload_json).map_err(|err| {
        eyre::eyre!("record for block {} has an unparseable payload: {err}", body.block_number)
    })?;
    let admitted = rules
        .admission
        .admit(payload, Some(&cursor.accepted_parent))
        .map_err(|err| eyre::eyre!("block {} was refused: {err:?}", body.block_number))?;
    let block_admission_us = admission_start.elapsed().as_micros() as u64;
    let block = admitted.block;
    if block.hash() != body.block_hash {
        eyre::bail!(
            "record claims block {:?} at height {}, but its payload admits as {:?}",
            body.block_hash,
            body.block_number,
            block.hash()
        )
    }
    if block.number() != body.block_number {
        eyre::bail!(
            "record claims height {} but its payload admits at height {}",
            body.block_number,
            block.number()
        )
    }

    // 2. One database-free execution against the recorded full witness. The sidecar it rides is
    // the Weak one by construction — a validator holding nothing — so the Weak arm reuses it
    // rather than building a second, identical witness.
    let FullWitnessExecution { accessed, hashed_post_state, weak_sidecar } =
        execute_against_full_witness(rules, record, &block)?;

    // 3. The recorded set and the derived set must be the same set.
    if let Some(divergence) = first_access_divergence(&body.accessed, &accessed) {
        eyre::bail!(
            "block {}: the recorded access set and this run's re-execution disagree: {divergence}",
            body.block_number
        )
    }

    // 4 and 5, per arm, in an order that rotates so no arm keeps the first slot.
    let source =
        RecordedFullWitnessSource::new(body.parent_state_root, &body.full_transition_nodes)?;
    let ctx = TransitionBuildContext::uninstrumented(&source);
    let block_ref = BlockTransitionRef {
        block_number: body.block_number,
        block_hash: body.block_hash,
        parent_hash: body.parent_hash,
        parent_state_root: body.parent_state_root,
        expected_state_root: body.expected_state_root,
        ancestor_headers: &body.ancestor_headers,
    };

    let mut results = vec![None; policies.len()];
    for (rotation_slot, index) in rotated_order(policies.len(), block_index).enumerate() {
        let state = &mut policies[index];
        let result = match state.kind {
            // Weak holds nothing between blocks, so both its pairs go back to cold before it
            // validates. Its sidecar is the full-witness one step 2 already built and proved —
            // building a second identical witness would double the run's most expensive step to
            // produce the same bytes.
            ArmKind::Weak => {
                state.reset_cold_at(body.block_number.saturating_sub(1));
                validate_and_commit(rules, &block, state, weak_sidecar.clone(), None, rotation_slot)
            }
            ArmKind::Policy(_) => {
                let built = partial_stateless::build_policy_sidecar(
                    &ctx,
                    block_ref,
                    &hashed_post_state,
                    &accessed,
                    &mut state.builder_cache,
                    &state.builder_trie,
                    &state.config,
                )
                .map_err(|err| eyre::eyre!("sidecar build failed: {err:#}"))?;
                let PolicySidecarBuild { sidecar, build, base: _, build_us } = built;
                // Against the parent generation, which is what the sidecar applies to — the
                // commit below replaces it, so this is the last moment the receiver's view of
                // this witness exists to measure against.
                let witness_trim = rules.trie_diagnostics.then(|| {
                    measure_witness_trim(&state.builder_trie, &build.nodes, &build.decoded_proof)
                });
                validate_and_commit(
                    rules,
                    &block,
                    state,
                    sidecar,
                    Some((build, build_us)),
                    rotation_slot,
                )
                .map(|mut result| {
                    result.witness_trim = witness_trim;
                    result.branch_census = rules
                        .trie_diagnostics
                        .then(|| state.builder_trie.branch_slot_census().into());
                    result
                })
            }
        }
        .map_err(|err| {
            eyre::eyre!("block {} arm {}: {err:#}", body.block_number, state.kind.label())
        })?;
        results[index] = Some(result);
    }

    cursor.accepted_parent = block.clone_sealed_header();

    let mut policies_in_rotation = Vec::with_capacity(policies.len());
    for index in rotated_order(policies.len(), block_index) {
        policies_in_rotation.push(results[index].take().expect("every policy produced a result"));
    }

    Ok(BlockResult {
        block_number: body.block_number,
        block_hash: block.hash(),
        measured,
        accessed_accounts: accessed.accounts.len(),
        accessed_storage: accessed.storage.len(),
        accessed_codes: accessed.codes.len(),
        full_witness_nodes: body.full_transition_nodes.len(),
        full_witness_node_bytes: body.full_transition_nodes.iter().map(|node| node.len()).sum(),
        block_admission_us,
        policies: policies_in_rotation,
    })
}

/// What one database-free execution against the recorded witness produced.
#[derive(Debug)]
struct FullWitnessExecution {
    /// The access set *this* run observed, derived rather than taken from the record.
    accessed: BlockAccessedState,
    /// The post state every arm's transition is computed from.
    hashed_post_state: HashedPostState,
    /// The sidecar that execution rode, which is the Weak arm's sidecar by construction.
    ///
    /// A validator holding nothing is exactly what the full witness describes, so the Weak arm has
    /// no separate thing to build. Handing it back rather than rebuilding it saves the run's most
    /// expensive step per block and, more importantly, guarantees the Weak arm is measured on the
    /// same bytes the corpus was proved with.
    weak_sidecar: PartialStatelessSidecar,
}

/// Executes the block with nothing but the recorded witness, and hands back what it observed.
fn execute_against_full_witness<Evm, C, ChainSpec>(
    rules: &GeneratorRules<'_, Evm, C, ChainSpec>,
    record: &PolicyDatasetRecord,
    block: &RecoveredBlock<BlockTy<EthPrimitives>>,
) -> eyre::Result<FullWitnessExecution>
where
    Evm: ConfigureEvm<Primitives = EthPrimitives>,
    C: FullConsensus<EthPrimitives> + ?Sized,
{
    let body = &record.body;
    // The miss manifest here names every accessed key, so the materializer reads every value this
    // block needs out of the witness. The *values* still come from the proof, not from the record:
    // a cold cache has nothing to fall back on, so a key the witness cannot serve is an error
    // rather than a silent read of the recorded number.
    let full_sidecar = full_witness_sidecar_from_nodes(
        body.parent_state_root,
        BlockTransitionRef {
            block_number: body.block_number,
            block_hash: body.block_hash,
            parent_hash: body.parent_hash,
            parent_state_root: body.parent_state_root,
            expected_state_root: body.expected_state_root,
            ancestor_headers: &body.ancestor_headers,
        },
        &body.accessed,
        body.full_transition_nodes.clone(),
        // Windows are irrelevant to a cold pair that is discarded straight after: the policy
        // identifier just has to match the sidecar's, and both come from this one config.
        &partial_stateless::CacheConfig::default(),
    )?;

    let config = partial_stateless::CacheConfig::default();
    let mut cold_cache = config.new_cache_at(body.block_number.saturating_sub(1));
    let mut cold_trie = partial_stateless::PartialTrieNodeCache::new();
    let tap = PostStateTap::default();
    let validated = verify_and_apply_sidecar_with_oracle(
        rules.validator,
        block,
        &mut cold_cache,
        &full_sidecar,
        config.cache_policy_id(),
        rules.limits,
        &mut cold_trie,
        TrieCacheDisposition::Discard,
        &tap,
        false,
    )
    .map_err(|err| {
        eyre::eyre!(
            "block {}: the recorded full witness does not re-execute this block: {err:#}",
            body.block_number
        )
    })?;

    let hashed_post_state =
        tap.take().ok_or_else(|| eyre::eyre!("the validator did not hand over a post state"))?;
    Ok(FullWitnessExecution {
        accessed: validated.outcome.actual_accessed,
        hashed_post_state,
        weak_sidecar: full_sidecar,
    })
}

/// Validates one arm's sidecar against that arm's validator pair, then commits both sides.
///
/// `built` carries the transition an arm built for itself; the Weak arm passes `None`, because its
/// sidecar was built once for the whole block before the rotation opened.
fn validate_and_commit<Evm, C, ChainSpec>(
    rules: &GeneratorRules<'_, Evm, C, ChainSpec>,
    block: &RecoveredBlock<BlockTy<EthPrimitives>>,
    state: &mut PolicyState,
    sidecar: PartialStatelessSidecar,
    built: Option<(CacheAwareFlatBuild, u64)>,
    rotation_slot: usize,
) -> eyre::Result<PolicyBlockResult>
where
    Evm: ConfigureEvm<Primitives = EthPrimitives>,
    C: FullConsensus<EthPrimitives> + ?Sized,
{
    // Serialization is builder-side preparation and stays outside the timer below. The *decode* is
    // inside it, because that is what a validator pays and it scales with a witness size the cache
    // policy decided — the one policy-dependent cost the previous boundary left out.
    let sidecar_bytes = bincode::serialize(&sidecar)
        .map_err(|err| eyre::eyre!("sidecar failed to serialize: {err}"))?;

    let validation_start = Instant::now();
    let decoded: PartialStatelessSidecar = bincode::deserialize(&sidecar_bytes)
        .map_err(|err| eyre::eyre!("sidecar failed to deserialize: {err}"))?;
    let sidecar_decode_us = validation_start.elapsed().as_micros() as u64;
    let validated = verify_and_apply_sidecar(
        rules.validator,
        block,
        &mut state.validator_cache,
        &decoded,
        state.config.cache_policy_id(),
        rules.limits,
        &mut state.validator_trie,
        TrieCacheDisposition::Commit,
    )
    .map_err(|err| eyre::eyre!("standalone validation refused the generated sidecar: {err:#}"))?;
    let sidecar_decode_and_commit_us = validation_start.elapsed().as_micros() as u64;

    if validated.outcome.next_cache_anchor != sidecar.next_cache_anchor {
        eyre::bail!(
            "the validator's own next cache anchor {:?} differs from the sidecar's {:?}",
            validated.outcome.next_cache_anchor,
            sidecar.next_cache_anchor
        )
    }

    // Only now. A refused sidecar leaves the builder's trie at the parent generation, which is
    // what stops the next block being built on a generation nothing agreed to — the flat cache
    // above has already advanced, and a run that reached this branch is about to stop anyway.
    // The Weak arm has no trie to carry: its next block starts cold either way.
    let (provider_calls, structural_rounds, offline_build_us) = match built {
        Some((build, build_us)) => {
            let shape = (build.provider_calls, build.structural_rounds, Some(build_us));
            state.builder_trie = build.next_trie_cache;
            shape
        }
        None => (0, 0, None),
    };

    Ok(PolicyBlockResult {
        policy: state.kind.label(),
        rotation_slot,
        sidecar_bytes: sidecar_bytes.len(),
        // Over the sidecar's semantic content rather than its bytes: the bytes carry this host's
        // own build time, so hashing them would report two hosts as disagreeing about a sidecar
        // they both produced identically.
        sidecar_digest: partial_stateless::sidecar_semantic_digest(&sidecar)
            .map_err(|err| eyre::eyre!("sidecar digest failed: {err}"))?,
        witness_commitment: sidecar.witness_commitment,
        witness_nodes: witness_node_count(&sidecar),
        witness_node_bytes: sidecar.stats.account_proof_bytes + sidecar.stats.storage_proof_bytes,
        witness_code_bytes: sidecar.stats.bytecode_bytes,
        missed_accounts: sidecar.miss_manifest.missed_accounts.len(),
        missed_storage: sidecar.miss_manifest.missed_storage.len(),
        missed_codes: sidecar.miss_manifest.missed_code_hashes.len(),
        prev_cache_root: sidecar.prev_cache_anchor.cache_root,
        next_cache_root: sidecar.next_cache_anchor.cache_root,
        cache_bytes: state.builder_cache.estimated_memory_bytes(),
        trie_cache_bytes: state.builder_trie.estimated_memory_bytes(),
        provider_calls,
        structural_rounds,
        sidecar_decode_and_commit_us,
        sidecar_decode_us,
        offline_build_us,
        witness_trim: None,
        branch_census: None,
    })
}

fn witness_node_count(sidecar: &PartialStatelessSidecar) -> usize {
    sidecar.stats.account_proof_nodes + sidecar.stats.storage_proof_nodes
}

/// The first way two access sets differ, described well enough to act on.
pub fn first_access_divergence(
    recorded: &BlockAccessedState,
    observed: &BlockAccessedState,
) -> Option<String> {
    for (address, data) in &recorded.accounts {
        match observed.accounts.get(address) {
            None => return Some(format!("account {address:?} is absent from the re-execution")),
            Some(seen) if seen != data => {
                return Some(format!("account {address:?} differs: {data:?} vs {seen:?}"))
            }
            Some(_) => {}
        }
    }
    for key in observed.accounts.keys() {
        if !recorded.accounts.contains_key(key) {
            return Some(format!("the re-execution saw account {key:?}, which was not recorded"))
        }
    }
    for (key, value) in &recorded.storage {
        match observed.storage.get(key) {
            None => return Some(format!("storage {key:?} is absent from the re-execution")),
            Some(seen) if seen != value => {
                return Some(format!("storage {key:?} differs: {value:?} vs {seen:?}"))
            }
            Some(_) => {}
        }
    }
    for key in observed.storage.keys() {
        if !recorded.storage.contains_key(key) {
            return Some(format!("the re-execution saw storage {key:?}, which was not recorded"))
        }
    }
    for (code_hash, code) in &recorded.codes {
        match observed.codes.get(code_hash) {
            None => return Some(format!("code {code_hash:?} is absent from the re-execution")),
            Some(seen) if seen != code => {
                return Some(format!(
                    "code {code_hash:?} differs in length: {} vs {}",
                    code.len(),
                    seen.len()
                ))
            }
            Some(_) => {}
        }
    }
    for code_hash in observed.codes.keys() {
        if !recorded.codes.contains_key(code_hash) {
            return Some(format!("the re-execution saw code {code_hash:?}, which was not recorded"))
        }
    }
    None
}
