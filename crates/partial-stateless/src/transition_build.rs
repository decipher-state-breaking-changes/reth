//! The one implementation of parent-state proof selection and sidecar assembly.
//!
//! Everything a partial-stateless sidecar is made of lives here: which parent-state targets a
//! block needs proved, how the structural rounds discover the rest, which nodes end up in the
//! witness, how the trie cache is carried forward, and how the finished sidecar is committed to.
//! A node builder running against a live database and an offline generator running against a
//! recorded witness both call these functions, so a policy comparison cannot be an artifact of two
//! builders disagreeing about what a sidecar is.
//!
//! The one thing that differs between those two callers is where a parent-state proof comes from,
//! and that is the whole of [`TransitionProofSource`]. A live builder answers it out of the node's
//! state database; an offline generator answers it out of a recorded policy-neutral full witness.
//! Nothing else in this module knows which it is talking to.
//!
//! This module deliberately depends on no state-database crate. That is what lets the offline
//! generator link it at all: its package's dependency graph is checked for provider and database
//! crates by `crates/partial-stateless-exex/scripts/check_validator_isolation.sh`, and a leaf
//! module that reached one would fail that check for every package downstream of it.

use crate::{
    accessed_state::BlockAccessedState,
    network_cache::{MissResult, NetworkStateCache},
    sidecar::{
        last_n_blocks_cache_policy_id, partial_witness_commitment, CacheAnchor,
        PartialExecutionWitness, PartialExecutionWitnessState, PartialStatelessSidecar,
        StateTargetSet, WitnessTargets,
    },
    trie_cache::PartialTrieNodeCache,
    witness::{build_sidecar_targets, WitnessResult},
    witness_check::{CacheAwareTransitionProgress, CacheAwareTrieTransition, TrieProofTargetV2},
    CacheConfig,
};
use alloy_primitives::{keccak256, map::B256Map, Bytes, B256};
use alloy_rlp::{Encodable, EMPTY_STRING_CODE};
use reth_trie_common::{
    DecodedMultiProofV2, HashedPostState, MultiProofTargetsV2, ProofV2Target, EMPTY_ROOT_HASH,
};
use std::{collections::BTreeMap, time::Instant};
use tracing::warn;

/// Distinct storage tries below which the wide-proof path is not worth its fan-out.
const PARALLEL_INITIAL_PROOF_MIN_STORAGE_TRIES: usize = 2;
/// Total targets below which the wide-proof path is not worth its fan-out.
const PARALLEL_INITIAL_PROOF_MIN_TOTAL_TARGETS: usize = 64;

/// Structural proof rounds one transition may take before it is called a loop.
pub(crate) const MAX_STRUCTURAL_ROUNDS: usize = 128;

/// A wide initial proof and what it cost in workers.
#[derive(Debug)]
pub struct ParallelProof {
    /// The proof itself.
    pub proof: DecodedMultiProofV2,
    /// Storage-trie workers the source used.
    pub storage_workers: usize,
    /// Account-trie workers the source used.
    pub account_workers: usize,
}

/// Where parent-state proofs come from.
///
/// The single seam between a live builder and an offline generator. A source proves targets
/// against one fixed parent state — the one the transition is anchored to — so it carries no
/// block or root argument: binding a source to the wrong parent is a construction-site mistake,
/// not something each call re-decides.
pub trait TransitionProofSource {
    /// Proves `targets` against this source's parent state.
    fn multiproof_v2(&self, targets: MultiProofTargetsV2) -> eyre::Result<DecodedMultiProofV2>;

    /// A wider path for the initial proof, when the source has one worth using.
    ///
    /// `None` — the default, and the only answer a recorded-witness source gives — means the
    /// source has [`Self::multiproof_v2`] and nothing else. The distinction reaches the benchmark
    /// record: a source with no wide path reports its initial proof as `serial`, while one that
    /// has a wide path and declined it on width reports `serial-low-width`, and the two are not
    /// the same observation.
    fn parallel_initial_proof(
        &self,
    ) -> Option<&dyn Fn(MultiProofTargetsV2) -> eyre::Result<ParallelProof>> {
        None
    }
}

/// The proof source a transition builds against, plus optional process instrumentation.
pub struct TransitionBuildContext<'a> {
    /// Where parent-state proofs come from.
    pub proofs: &'a dyn TransitionProofSource,
    /// Samples process resident memory around the per-block trie clone.
    ///
    /// `None` leaves the reported delta at zero, which is what an offline generator wants: its
    /// process memory is not the production builder's and reporting it under the same field name
    /// would put two different quantities in one column.
    pub rss_sampler: Option<fn() -> u64>,
    /// Also produce the receiver-aware trimmed (v3) witness alongside the full flat one.
    ///
    /// Requires the parent trie cache to be revealed and anchored — a cold or warming cache has
    /// no frontier to trim against, and callers degrade to the self-contained v2 wire instead.
    pub trim_witness: bool,
}

impl std::fmt::Debug for TransitionBuildContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransitionBuildContext")
            .field("rss_sampler", &self.rss_sampler.is_some())
            .finish_non_exhaustive()
    }
}

impl<'a> TransitionBuildContext<'a> {
    /// A context that measures nothing about the host process.
    pub const fn uninstrumented(proofs: &'a dyn TransitionProofSource) -> Self {
        Self { proofs, rss_sampler: None, trim_witness: false }
    }

    /// The same context, additionally producing the trimmed (v3) witness.
    pub const fn with_trimmed_witness(mut self) -> Self {
        self.trim_witness = true;
        self
    }

    fn rss_bytes(&self) -> i64 {
        self.rss_sampler.map_or(0, |sample| sample() as i64)
    }
}

/// A deduplicated set of V2 proof targets, keyed so two equal sets iterate in one order.
///
/// Ordering is load-bearing rather than tidy: the target order decides the order proof nodes are
/// requested in, and a sidecar's node vector is what a size comparison between two cache policies
/// is measuring. A hash-ordered set would make the same block produce different bytes per run.
#[derive(Clone, Debug, Default)]
pub struct V2TargetSet {
    accounts: BTreeMap<B256, u8>,
    storage: BTreeMap<(B256, B256), u8>,
}

impl V2TargetSet {
    /// Adds one target, keeping the shallowest `min_len` when the key is already present.
    pub fn insert(&mut self, target: TrieProofTargetV2) {
        match target {
            TrieProofTargetV2::Account { key, min_len } => {
                self.accounts
                    .entry(key)
                    .and_modify(|current| *current = (*current).min(min_len))
                    .or_insert(min_len);
            }
            TrieProofTargetV2::Storage { hashed_address, key, min_len } => {
                self.storage
                    .entry((hashed_address, key))
                    .and_modify(|current| *current = (*current).min(min_len))
                    .or_insert(min_len);
            }
        }
    }

    /// Adds every target in `targets`.
    pub fn extend(&mut self, targets: impl IntoIterator<Item = TrieProofTargetV2>) {
        for target in targets {
            self.insert(target);
        }
    }

    /// Hashed account keys, in the deterministic set order.
    pub(crate) fn account_keys(&self) -> impl Iterator<Item = B256> + '_ {
        self.accounts.keys().copied()
    }

    /// Hashed `(account, slot)` keys, in the deterministic set order.
    pub(crate) fn storage_keys(&self) -> impl Iterator<Item = (B256, B256)> + '_ {
        self.storage.keys().copied()
    }

    fn with_zero_min_len(&self) -> Self {
        Self {
            accounts: self.accounts.keys().copied().map(|key| (key, 0)).collect(),
            storage: self.storage.keys().copied().map(|key| (key, 0)).collect(),
        }
    }

    /// Expands targets for the context-free flat wire format.
    ///
    /// A flattened storage node does not carry its account address. Including the parent account
    /// proof makes the storage root reachable from the state root, allowing a standalone decoder
    /// to recover the account/storage association. Native structured V2 proofs retain that
    /// association directly and do not need these additional account targets.
    pub(crate) fn with_flat_storage_context(&self) -> Self {
        let mut targets = self.with_zero_min_len();
        for &(hashed_address, _) in self.storage.keys() {
            targets.accounts.entry(hashed_address).or_insert(0);
        }
        targets
    }

    pub(crate) fn difference_and_record(&self, requested: &mut Self) -> Self {
        let mut delta = Self::default();
        for (&key, &min_len) in &self.accounts {
            if requested.accounts.get(&key).is_some_and(|current| *current <= min_len) {
                continue;
            }
            requested.accounts.insert(key, min_len);
            delta.accounts.insert(key, min_len);
        }
        for (&key, &min_len) in &self.storage {
            if requested.storage.get(&key).is_some_and(|current| *current <= min_len) {
                continue;
            }
            requested.storage.insert(key, min_len);
            delta.storage.insert(key, min_len);
        }
        delta
    }

    /// True when no target is present.
    pub fn is_empty(&self) -> bool {
        self.accounts.is_empty() && self.storage.is_empty()
    }

    /// Account plus storage targets.
    pub fn len(&self) -> usize {
        self.accounts.len() + self.storage.len()
    }

    /// Account targets only.
    pub fn account_count(&self) -> usize {
        self.accounts.len()
    }

    /// Storage targets only.
    pub fn storage_count(&self) -> usize {
        self.storage.len()
    }

    /// Storage tries the targets reach into.
    pub fn distinct_storage_tries(&self) -> usize {
        let mut previous = None;
        let mut count = 0;
        for &(hashed_address, _) in self.storage.keys() {
            if previous != Some(hashed_address) {
                previous = Some(hashed_address);
                count += 1;
            }
        }
        count
    }

    fn should_use_parallel_initial_proof(&self) -> bool {
        self.distinct_storage_tries() >= PARALLEL_INITIAL_PROOF_MIN_STORAGE_TRIES &&
            self.len() >= PARALLEL_INITIAL_PROOF_MIN_TOTAL_TARGETS
    }

    /// The wire form a [`TransitionProofSource`] is asked in.
    pub fn to_provider_targets(&self) -> MultiProofTargetsV2 {
        let account_targets = self
            .accounts
            .iter()
            .map(|(&key, &min_len)| ProofV2Target::new(key).with_min_len(min_len))
            .collect();
        let mut storage_targets = B256Map::default();
        for (&(hashed_address, key), &min_len) in &self.storage {
            storage_targets
                .entry(hashed_address)
                .or_insert_with(Vec::new)
                .push(ProofV2Target::new(key).with_min_len(min_len));
        }
        MultiProofTargetsV2 { account_targets, storage_targets }
    }
}

/// A completed cache-aware transition: the witness it produced and what producing it cost.
#[derive(Debug)]
pub struct CacheAwareFlatBuild {
    /// The flat, hash-deduplicated parent-state node witness, sorted.
    pub nodes: Vec<Bytes>,
    /// The same witness, decoded against the parent state root.
    pub decoded_proof: DecodedMultiProofV2,
    /// The receiver-aware trimmed (v3) wire, when the context asked for one and the parent trie
    /// had a frontier to trim against.
    pub trimmed: Option<TrimmedWitnessBuild>,
    /// The trie cache this block's transition produced, before retention.
    pub next_trie_cache: PartialTrieNodeCache,
    /// The post-state root the local sparse trie computed.
    pub state_root: B256,
    /// Calls made to the proof source.
    pub provider_calls: usize,
    /// Structural discovery rounds the transition needed.
    pub structural_rounds: usize,
    /// Targets in the initial proof.
    pub initial_targets: usize,
    /// Nodes the initial proof contributed.
    pub initial_proof_nodes: usize,
    /// Bytes the initial proof contributed.
    pub initial_proof_bytes: usize,
    /// Targets discovered by the structural rounds.
    pub structural_targets: usize,
    /// Account targets among them.
    pub structural_account_targets: usize,
    /// Storage targets among them.
    pub structural_storage_targets: usize,
    /// Mutation paths the trie cache already authenticated, so no proof was asked for.
    pub cache_covered_mutation_targets: usize,
    /// Wall time of the per-block trie clone.
    pub trie_clone_us: u64,
    /// Logical size of the trie the per-block snapshot copied.
    pub trie_clone_bytes: usize,
    /// Process RSS moved across the clone. Process-wide, so meaningful only in aggregate, and
    /// zero when the caller supplied no sampler.
    pub trie_clone_rss_delta_bytes: i64,
    /// Wall time of the transition itself.
    pub transition_us: u64,
    /// Wall time the proof source spent inside the structural rounds.
    pub structural_provider_us: u64,
}

/// The trimmed (v3) wire a transition build produced alongside the full flat witness.
#[derive(Debug)]
pub struct TrimmedWitnessBuild {
    /// The fragment node list: hash-deduplicated, byte-sorted, a strict subset of the full flat
    /// witness whenever the parent trie reveals anything on a target chain.
    pub nodes: Vec<Bytes>,
    /// The same fragments, path-attributed for size accounting and diagnostics.
    pub fragments: DecodedMultiProofV2,
    /// The parent generation's retention fingerprint the fragments were cut against.
    pub retention_fingerprint: B256,
}

/// The initial proof a transition starts from.
#[derive(Debug)]
pub struct CacheAwareBaseProof {
    /// Targets it was asked for.
    pub targets: V2TargetSet,
    /// The proof itself.
    pub proof: DecodedMultiProofV2,
    /// Mutation paths the trie cache already authenticated.
    pub cache_covered_mutation_targets: usize,
    /// Wall time the proof source spent.
    pub provider_us: u64,
    /// Which path answered: `empty`, `parallel`, `serial`, `serial-low-width`, or
    /// `serial-after-parallel-error`.
    pub proof_source: &'static str,
    /// Storage workers the wide path used, zero otherwise.
    pub parallel_storage_workers: usize,
    /// Account workers the wide path used, zero otherwise.
    pub parallel_account_workers: usize,
}

/// The parent-state targets a block needs proved before its transition can start.
///
/// Two disjoint reasons a target appears. A **cache miss** is a value the validator does not hold
/// and must be sent, so it is proved whether or not its path happens to be locally authenticated.
/// A **mutation path** is a path the post-state changes, which the sparse trie must have revealed
/// to recompute a root — and that one is skipped when the trie cache already covers it, which is
/// the whole of what the trie cache buys.
pub fn initial_cache_aware_targets(
    post_state: &HashedPostState,
    miss: &MissResult,
    trie_cache: &PartialTrieNodeCache,
) -> (V2TargetSet, usize) {
    let mut targets = V2TargetSet::default();
    // Cache misses are value proofs, even if the mutation path happens to be cached locally.
    for address in &miss.missed_accounts {
        targets.insert(TrieProofTargetV2::Account { key: keccak256(address), min_len: 0 });
    }
    for (address, slot) in &miss.missed_storage {
        let hashed_address = keccak256(address);
        targets.insert(TrieProofTargetV2::Account { key: hashed_address, min_len: 0 });
        targets.insert(TrieProofTargetV2::Storage {
            hashed_address,
            key: keccak256(slot),
            min_len: 0,
        });
    }

    let mut mutation_target_count = 0usize;
    let mut uncovered_mutation_count = 0usize;
    let mut account_paths =
        post_state.accounts.keys().chain(post_state.storages.keys()).copied().collect::<Vec<_>>();
    account_paths.sort_unstable();
    account_paths.dedup();
    for hashed_address in account_paths {
        mutation_target_count += 1;
        if !trie_cache.contains_hashed_account_path(hashed_address) {
            uncovered_mutation_count += 1;
            targets.insert(TrieProofTargetV2::Account { key: hashed_address, min_len: 0 });
        }
    }
    for (&hashed_address, storage) in &post_state.storages {
        if storage.wiped {
            continue;
        }
        for &hashed_slot in storage.storage.keys() {
            mutation_target_count += 1;
            if !trie_cache.contains_hashed_storage_path(hashed_address, hashed_slot) {
                uncovered_mutation_count += 1;
                targets.insert(TrieProofTargetV2::Storage {
                    hashed_address,
                    key: hashed_slot,
                    min_len: 0,
                });
            }
        }
    }

    (targets, mutation_target_count.saturating_sub(uncovered_mutation_count))
}

/// Proves everything [`initial_cache_aware_targets`] asked for, in one call where it can be.
pub fn generate_cache_aware_base_proof(
    ctx: &TransitionBuildContext<'_>,
    post_state: &HashedPostState,
    miss: &MissResult,
    trie_cache: &PartialTrieNodeCache,
) -> eyre::Result<CacheAwareBaseProof> {
    let (mut targets, cache_covered_mutation_targets) =
        initial_cache_aware_targets(post_state, miss, trie_cache);
    if targets.is_empty() && trie_cache.state_root().is_none() {
        targets.insert(TrieProofTargetV2::Account { key: B256::ZERO, min_len: 0 });
    }

    let parallel_initial_proof = ctx.proofs.parallel_initial_proof();
    let provider_start = Instant::now();
    let (proof, proof_source, parallel_storage_workers, parallel_account_workers) = if targets
        .is_empty()
    {
        (DecodedMultiProofV2::default(), "empty", 0, 0)
    } else if let Some(parallel_initial_proof) = parallel_initial_proof &&
        targets.should_use_parallel_initial_proof()
    {
        match parallel_initial_proof(targets.to_provider_targets()) {
            Ok(ParallelProof { proof, storage_workers, account_workers }) => {
                (proof, "parallel", storage_workers, account_workers)
            }
            Err(err) => {
                warn!(
                    target: "partial_stateless",
                    error = %err,
                    initial_targets = targets.len(),
                    distinct_storage_tries = targets.distinct_storage_tries(),
                    "Parallel initial V2 multiproof failed; retrying with the serial parent provider"
                );
                let proof =
                    ctx.proofs.multiproof_v2(targets.to_provider_targets()).map_err(
                        |serial_err| {
                            eyre::eyre!(
                            "parallel initial V2 multiproof failed ({err}); serial fallback also failed: {serial_err}"
                        )
                        },
                    )?;
                (proof, "serial-after-parallel-error", 0, 0)
            }
        }
    } else {
        let proof = ctx
            .proofs
            .multiproof_v2(targets.to_provider_targets())
            .map_err(|err| eyre::eyre!("failed to generate initial V2 multiproof: {err}"))?;
        let source = if parallel_initial_proof.is_some() { "serial-low-width" } else { "serial" };
        (proof, source, 0, 0)
    };
    let provider_us = provider_start.elapsed().as_micros() as u64;

    Ok(CacheAwareBaseProof {
        targets,
        proof,
        cache_covered_mutation_targets,
        provider_us,
        proof_source,
        parallel_storage_workers,
        parallel_account_workers,
    })
}

/// Runs one block's cache-aware transition to a checked post-state root.
///
/// The initial proof is revealed into a cloned sparse trie; whatever the trie still needs is
/// discovered a round at a time and proved as a delta, and every node either side contributed
/// lands in one flat, hash-deduplicated witness. The root is checked against the block header
/// here rather than by the caller, because a build that returns is a build whose witness is known
/// to reconstruct the block's own state root.
pub fn build_cache_aware_flat_transition(
    ctx: &TransitionBuildContext<'_>,
    parent_state_root: B256,
    expected_state_root: B256,
    post_state: HashedPostState,
    miss: &MissResult,
    trie_cache: &PartialTrieNodeCache,
    base: &CacheAwareBaseProof,
) -> eyre::Result<CacheAwareFlatBuild> {
    // The clone is the cost the transactional-trie design exists to remove, so it is measured three
    // ways: wall time, the logical size of what was copied, and the process RSS it moved. RSS is
    // process-wide and noisy per block; it is only interpretable across many blocks.
    let rss_before = ctx.rss_bytes();
    let clone_start = Instant::now();
    let mut next_trie_cache = trie_cache.clone();
    let trie_clone_us = clone_start.elapsed().as_micros() as u64;
    let trie_clone_bytes = next_trie_cache.estimated_memory_bytes();
    // Signed: the clone can overlap with memory being released elsewhere in the process, and
    // clamping those samples to zero would bias the distribution upwards.
    let trie_clone_rss_delta_bytes = ctx.rss_bytes() - rss_before;
    let mut requested_flat = base.targets.clone();
    let mut revealed_exact = base.targets.clone();
    let mut flat_nodes = B256Map::<Bytes>::default();
    base.proof.extend_flat_witness(&mut flat_nodes);
    let initial_proof_nodes = flat_nodes.len();
    let initial_proof_bytes = flat_nodes.values().map(|node| node.len()).sum();
    let mut accumulated_parent_proof = base.proof.clone();
    let mut provider_calls = (!base.proof.is_empty()) as usize;
    let initial_target_count = base.targets.len();
    let read_only_storage_targets =
        miss.missed_storage.iter().map(|(address, _)| keccak256(address)).collect::<Vec<_>>();
    let transition_start = Instant::now();
    let mut structural_rounds = 0usize;
    let mut structural_target_count = 0usize;
    let mut structural_account_target_count = 0usize;
    let mut structural_storage_target_count = 0usize;
    let mut structural_provider_us = 0u64;

    // Flat storage nodes have no address/path context. Fetch any missing parent account proofs
    // once so every storage root in the flat witness is independently reachable from the state
    // root. This flat-only overhead is reported with the structural proof metrics.
    let flat_base_targets = base.targets.with_flat_storage_context();
    let context_delta = flat_base_targets.difference_and_record(&mut requested_flat);
    if !context_delta.is_empty() {
        structural_target_count += context_delta.len();
        structural_account_target_count += context_delta.account_count();
        structural_storage_target_count += context_delta.storage_count();
        let provider_start = Instant::now();
        let proof =
            ctx.proofs.multiproof_v2(context_delta.to_provider_targets()).map_err(|err| {
                eyre::eyre!("failed to generate flat storage-context V2 proof delta: {err}")
            })?;
        structural_provider_us += provider_start.elapsed().as_micros() as u64;
        provider_calls += 1;
        proof.extend_flat_witness(&mut flat_nodes);
        accumulated_parent_proof.extend(proof);
    }

    let state_root = {
        let mut session = CacheAwareTrieTransition::new(
            &mut next_trie_cache,
            post_state,
            read_only_storage_targets,
        );
        if !base.proof.is_empty() {
            session
                .reveal(base.proof.clone())
                .map_err(|err| eyre::eyre!("failed to reveal initial V2 multiproof: {err}"))?;
        }
        loop {
            match session
                .advance()
                .map_err(|err| eyre::eyre!("cache-aware transition failed: {err}"))?
            {
                CacheAwareTransitionProgress::Complete(root) => break root,
                CacheAwareTransitionProgress::ProofRequired(targets) => {
                    if structural_rounds >= MAX_STRUCTURAL_ROUNDS {
                        return Err(eyre::eyre!(
                            "cache-aware transition exceeded {MAX_STRUCTURAL_ROUNDS} structural proof rounds"
                        ));
                    }
                    structural_rounds += 1;
                    let mut exact = V2TargetSet::default();
                    exact.extend(targets);
                    let flat = exact.with_flat_storage_context();
                    let delta = flat.difference_and_record(&mut requested_flat);
                    if !delta.is_empty() {
                        structural_target_count += delta.len();
                        structural_account_target_count += delta.account_count();
                        structural_storage_target_count += delta.storage_count();
                        let provider_start = Instant::now();
                        let proof = ctx.proofs.multiproof_v2(delta.to_provider_targets()).map_err(
                            |err| {
                                eyre::eyre!(
                                    "failed to generate structural V2 multiproof delta: {err}"
                                )
                            },
                        )?;
                        structural_provider_us += provider_start.elapsed().as_micros() as u64;
                        provider_calls += 1;
                        proof.extend_flat_witness(&mut flat_nodes);
                        accumulated_parent_proof.extend(proof);
                    }

                    let reveal_delta = exact.difference_and_record(&mut revealed_exact);
                    if reveal_delta.is_empty() {
                        return Err(eyre::eyre!(
                            "cache-aware transition made no progress: all {} structural targets were already requested",
                            exact.len()
                        ));
                    }
                    let mut reveal_proof = accumulated_parent_proof.clone();
                    reveal_proof.retain_targets(&reveal_delta.to_provider_targets());
                    if reveal_proof.is_empty() {
                        return Err(eyre::eyre!(
                            "cache-aware transition structural proof delta was empty for {} targets",
                            reveal_delta.len()
                        ));
                    }
                    session.reveal(reveal_proof).map_err(|err| {
                        eyre::eyre!("failed to reveal structural V2 proof delta: {err}")
                    })?;
                }
            }
        }
    };
    let transition_us = transition_start.elapsed().as_micros() as u64;

    if state_root != expected_state_root {
        return Err(eyre::eyre!(
            "cache-aware sparse-trie root mismatch: expected {expected_state_root:?}, got {state_root:?}"
        ));
    }

    // The trimmed (v3) wire is derived by running the receiver's own composite walk against the
    // full flat map: every target ever requested is walked through the parent trie first, and
    // the map entries the walk consumes below the blinded frontier are exactly the nodes a
    // grafting validator will consume. Producing the wire with the consumer's algorithm is what
    // makes the unconsumed-node rejection rule exact rather than approximate.
    let trimmed = if ctx.trim_witness && trie_cache.state_root().is_some() {
        let mut wire_map = crate::witness_v3::WitnessNodeMap::from_flat(flat_nodes.clone());
        crate::witness_v3::consume_target_chains(trie_cache, &mut wire_map, &requested_flat)
            .map_err(|err| {
                eyre::eyre!("trimmed-witness wire walk failed against the builder's own map: {err}")
            })?;
        let fragments =
            crate::witness_v3::collect_reveal_fragments(trie_cache, &mut wire_map, &requested_flat)
                .map_err(|err| eyre::eyre!("trimmed-witness fragment collection failed: {err}"))?;
        Some(TrimmedWitnessBuild {
            nodes: wire_map.consumed_nodes_sorted(),
            fragments,
            retention_fingerprint: trie_cache.retention_fingerprint(),
        })
    } else {
        None
    };

    let mut nodes = flat_nodes.into_values().collect::<Vec<_>>();
    nodes.sort_unstable();
    nodes.dedup();
    let decoded_proof = decode_transition_witness(parent_state_root, &nodes)?;

    Ok(CacheAwareFlatBuild {
        nodes,
        decoded_proof,
        trimmed,
        next_trie_cache,
        state_root,
        provider_calls,
        structural_rounds,
        initial_targets: initial_target_count,
        initial_proof_nodes,
        initial_proof_bytes,
        structural_targets: structural_target_count,
        structural_account_targets: structural_account_target_count,
        structural_storage_targets: structural_storage_target_count,
        cache_covered_mutation_targets: base.cache_covered_mutation_targets,
        trie_clone_us,
        trie_clone_bytes,
        trie_clone_rss_delta_bytes,
        transition_us,
        structural_provider_us,
    })
}

/// Decodes a flat node witness back into the structured proof its account/storage split needs.
pub fn decode_transition_witness(
    parent_state_root: B256,
    nodes: &[Bytes],
) -> eyre::Result<DecodedMultiProofV2> {
    let mut witness = B256Map::with_capacity_and_hasher(nodes.len(), Default::default());
    for node in nodes {
        witness.insert(keccak256(node), node.clone());
    }
    if parent_state_root == EMPTY_ROOT_HASH {
        witness.entry(parent_state_root).or_insert_with(|| Bytes::from([EMPTY_STRING_CODE]));
    }
    DecodedMultiProofV2::from_witness(parent_state_root, &witness)
        .map_err(|err| eyre::eyre!("failed to decode canonical transition witness: {err}"))
}

/// Splits a flat witness into its account and storage halves for the size report.
pub fn measure_transition_witness_size(
    nodes: &[Bytes],
    proof: &DecodedMultiProofV2,
    bytecode_bytes: usize,
) -> WitnessResult {
    let mut storage_hashes = std::collections::HashSet::new();
    for storage_proof in proof.storage_proofs.values() {
        for proof_node in storage_proof {
            let mut encoded = Vec::new();
            proof_node.node.encode(&mut encoded);
            storage_hashes.insert(keccak256(&encoded));
        }
    }

    let (mut account_proof_bytes, mut storage_proof_bytes) = (0usize, 0usize);
    let (mut account_proof_nodes, mut storage_proof_nodes) = (0usize, 0usize);
    for node in nodes {
        if storage_hashes.contains(&keccak256(node)) {
            storage_proof_bytes += node.len();
            storage_proof_nodes += 1;
        } else {
            account_proof_bytes += node.len();
            account_proof_nodes += 1;
        }
    }

    WitnessResult {
        total_size_bytes: account_proof_bytes + storage_proof_bytes + bytecode_bytes,
        account_proof_bytes,
        storage_proof_bytes,
        bytecode_bytes,
        account_proof_nodes,
        storage_proof_nodes,
        target_accounts: 0,
        target_storage_slots: 0,
        computation_time_ms: None,
        cpu_time_ms: None,
        major_page_faults: None,
        minor_page_faults: None,
    }
}

/// Everything a finished sidecar is committed to, before it is committed to.
///
/// One struct rather than fourteen arguments, and one construction site rather than two: a live
/// builder and an offline generator that assembled sidecars separately could differ in field
/// order, in which preimages travel, or in what the commitment covers, and every one of those
/// differences would read as a cache-policy effect.
#[derive(Debug)]
pub struct SidecarAssembly {
    /// Parent block hash.
    pub parent_hash: B256,
    /// Parent state root the witness is proved against.
    pub parent_state_root: B256,
    /// This block's hash.
    pub block_hash: B256,
    /// This block's number.
    pub block_number: u64,
    /// The block the consuming cache must be at.
    pub cache_block: u64,
    /// Policy identifier both anchors carry.
    pub cache_policy_id: B256,
    /// Human-readable policy description.
    pub cache_policy_metadata: String,
    /// The cache state this sidecar applies to.
    pub prev_cache_anchor: CacheAnchor,
    /// The cache state applying it produces.
    pub next_cache_anchor: CacheAnchor,
    /// Keys the validator does not hold and must be sent.
    pub miss_manifest: WitnessTargets,
    /// The parent-state witness body.
    pub witness_state: PartialExecutionWitnessState,
    /// Bytecodes for missed code hashes.
    pub codes: Vec<Bytes>,
    /// Ancestor headers this block's BLOCKHASH range needs.
    pub headers: Vec<Bytes>,
    /// Measured witness size.
    pub stats: WitnessResult,
}

/// Builds the sidecar and its commitment from an assembled description.
pub fn assemble_sidecar(assembly: SidecarAssembly) -> PartialStatelessSidecar {
    let SidecarAssembly {
        parent_hash,
        parent_state_root,
        block_hash,
        block_number,
        cache_block,
        cache_policy_id,
        cache_policy_metadata,
        prev_cache_anchor,
        next_cache_anchor,
        miss_manifest,
        witness_state,
        codes,
        headers,
        stats,
    } = assembly;

    let cache_miss_targets = StateTargetSet::from(&miss_manifest);
    let witness = PartialExecutionWitness {
        state: witness_state,
        codes,
        keys: miss_manifest.key_preimages(),
        headers,
    };
    let witness_commitment =
        partial_witness_commitment(parent_state_root, &cache_miss_targets, &witness);

    PartialStatelessSidecar {
        parent_hash,
        parent_state_root,
        block_hash,
        block_number,
        cache_block,
        cache_policy_id,
        prev_cache_anchor,
        next_cache_anchor,
        cache_policy_metadata,
        cache_miss_targets,
        witness_commitment,
        miss_manifest,
        witness,
        stats,
    }
}

/// One block's identity and post-state, as every sidecar build needs it.
#[derive(Debug, Clone, Copy)]
pub struct BlockTransitionRef<'a> {
    /// This block's number.
    pub block_number: u64,
    /// This block's hash.
    pub block_hash: B256,
    /// Parent block hash.
    pub parent_hash: B256,
    /// Parent state root the witness is proved against.
    pub parent_state_root: B256,
    /// The state root the block's own header claims.
    pub expected_state_root: B256,
    /// Ancestor headers this block's BLOCKHASH range needs.
    pub ancestor_headers: &'a [Bytes],
}

/// A policy sidecar and the transition that produced it.
#[derive(Debug)]
pub struct PolicySidecarBuild {
    /// The sidecar.
    pub sidecar: PartialStatelessSidecar,
    /// The transition, for callers that report its internals.
    pub build: CacheAwareFlatBuild,
    /// The initial proof, for the same reason.
    pub base: CacheAwareBaseProof,
    /// Wall time of the whole build.
    pub build_us: u64,
}

/// Builds one block's sidecar under one cache policy, against one proof source.
///
/// `cache` must already be at the block's parent and must **not** yet have this block applied:
/// the miss set is computed against it, the block is then applied, and both anchors are read off
/// the same object either side of that application. `trie_cache` is likewise the parent
/// generation and is left untouched — the transition works on a clone, which the returned build
/// carries.
///
/// This is the composition an offline generator runs. A live node builder runs the same steps in
/// the same order with its own instrumentation and publication checks interleaved, calling every
/// function this one calls.
pub fn build_policy_sidecar(
    ctx: &TransitionBuildContext<'_>,
    block: BlockTransitionRef<'_>,
    hashed_post_state: &HashedPostState,
    accessed: &BlockAccessedState,
    cache: &mut NetworkStateCache,
    trie_cache: &PartialTrieNodeCache,
    config: &CacheConfig,
) -> eyre::Result<PolicySidecarBuild> {
    let build_start = Instant::now();
    let parent_block_number = block.block_number.saturating_sub(1);
    if cache.current_block() != parent_block_number {
        return Err(eyre::eyre!(
            "cache is at block {} but block {} needs it at {parent_block_number}",
            cache.current_block(),
            block.block_number,
        ));
    }

    let cache_policy_id =
        last_n_blocks_cache_policy_id(config.account_window, config.storage_window);
    let prev_cache_anchor =
        cache.cache_anchor(parent_block_number, block.parent_hash, cache_policy_id);

    let miss = cache.compute_miss(accessed);
    let (miss_manifest, _) = build_sidecar_targets(&miss);
    let missed_bytecode_bytes: usize = miss
        .missed_codes
        .iter()
        .filter_map(|code_hash| accessed.codes.get(code_hash))
        .map(|bytes| bytes.len())
        .sum();
    let missed_bytecodes: Vec<Bytes> = miss
        .missed_codes
        .iter()
        .filter_map(|code_hash| accessed.codes.get(code_hash).cloned())
        .collect();

    let base = generate_cache_aware_base_proof(ctx, hashed_post_state, &miss, trie_cache)?;
    let mut build = build_cache_aware_flat_transition(
        ctx,
        block.parent_state_root,
        block.expected_state_root,
        hashed_post_state.clone(),
        &miss,
        trie_cache,
        &base,
    )?;

    // Applied only after the transition has agreed with the block's own state root, and in the
    // same order the live builder applies it: the next anchor and the retained trie shape are both
    // functions of the post-application cache.
    cache.on_block_executed(block.block_number, accessed);
    build.next_trie_cache.retain_from_value_cache(cache);
    let next_cache_anchor =
        cache.cache_anchor(block.block_number, block.block_hash, cache_policy_id);

    // The wire the sidecar carries is what its stats must describe: for a trimmed build that is
    // the fragment list, attributed through the fragment proof rather than the full decode.
    let (witness_state, measured_nodes, measured_proof) = match &build.trimmed {
        Some(trimmed) => (
            PartialExecutionWitnessState::MptTrimmedTransitionNodes {
                frontier_version: crate::witness_v3::WITNESS_V3_FRONTIER_VERSION,
                retention_fingerprint: trimmed.retention_fingerprint,
                nodes: trimmed.nodes.clone(),
            },
            &trimmed.nodes,
            &trimmed.fragments,
        ),
        None => (
            PartialExecutionWitnessState::MptTransitionNodes(build.nodes.clone()),
            &build.nodes,
            &build.decoded_proof,
        ),
    };
    let mut stats =
        measure_transition_witness_size(measured_nodes, measured_proof, missed_bytecode_bytes);
    stats.target_accounts = base.targets.account_count() + build.structural_account_targets;
    stats.target_storage_slots = base.targets.storage_count() + build.structural_storage_targets;
    stats.computation_time_ms = Some(build_start.elapsed().as_millis() as u64);

    let sidecar = assemble_sidecar(SidecarAssembly {
        parent_hash: block.parent_hash,
        parent_state_root: block.parent_state_root,
        block_hash: block.block_hash,
        block_number: block.block_number,
        cache_block: parent_block_number,
        cache_policy_id,
        cache_policy_metadata: format!(
            "LastNBlocks(account: {}, storage/code: {})",
            config.account_window, config.storage_window
        ),
        prev_cache_anchor,
        next_cache_anchor,
        miss_manifest,
        witness_state,
        codes: missed_bytecodes,
        headers: block.ancestor_headers.to_vec(),
        stats,
    });

    Ok(PolicySidecarBuild {
        sidecar,
        build,
        base,
        build_us: build_start.elapsed().as_micros() as u64,
    })
}

/// A policy-neutral full witness: the parent-state proof a validator holding *nothing* would need.
///
/// The cache is cold and the trie cache is empty, so every accessed key is a miss and every
/// mutation path is proved from the root. Two consequences make this the right artifact to record
/// once and generate every cache policy from. It mentions no policy — no window, no anchor, no
/// miss set that a policy chose — and its node set is a superset of what any policy's transition
/// over the same block can ask for, because every target a warm cache would skip is one this build
/// already proved.
#[expect(clippy::too_many_arguments)]
pub fn build_full_witness_sidecar(
    ctx: &TransitionBuildContext<'_>,
    parent_state_root: B256,
    expected_state_root: B256,
    parent_hash: B256,
    block_hash: B256,
    block_number: u64,
    hashed_post_state: &HashedPostState,
    accessed: &BlockAccessedState,
    ancestor_headers: &[Bytes],
    config: &CacheConfig,
) -> eyre::Result<FullWitnessBuild> {
    let build_start = Instant::now();
    let parent_block_number = block_number.saturating_sub(1);
    let cold_cache = config.new_cache_at(parent_block_number);
    let full_miss = cold_cache.compute_miss(accessed);
    let cold_trie = PartialTrieNodeCache::new();
    let base = generate_cache_aware_base_proof(ctx, hashed_post_state, &full_miss, &cold_trie)?;
    let build = build_cache_aware_flat_transition(
        ctx,
        parent_state_root,
        expected_state_root,
        hashed_post_state.clone(),
        &full_miss,
        &cold_trie,
        &base,
    )?;

    let (miss_manifest, _) = build_sidecar_targets(&full_miss);
    let mut codes = accessed.codes.iter().collect::<Vec<_>>();
    codes.sort_unstable_by_key(|(code_hash, _)| **code_hash);
    let codes = codes.into_iter().map(|(_, code)| code.clone()).collect::<Vec<_>>();
    let bytecode_bytes = codes.iter().map(|code| code.len()).sum();
    let mut stats =
        measure_transition_witness_size(&build.nodes, &build.decoded_proof, bytecode_bytes);
    stats.target_accounts = base.targets.account_count() + build.structural_account_targets;
    stats.target_storage_slots = base.targets.storage_count() + build.structural_storage_targets;
    stats.computation_time_ms = Some(build_start.elapsed().as_millis() as u64);

    let cache_policy_id =
        last_n_blocks_cache_policy_id(config.account_window, config.storage_window);
    let prev_cache_anchor =
        cold_cache.cache_anchor(parent_block_number, parent_hash, cache_policy_id);
    let mut next_cache = config.new_cache_at(parent_block_number);
    next_cache.on_block_executed(block_number, accessed);
    let next_cache_anchor = next_cache.cache_anchor(block_number, block_hash, cache_policy_id);

    let sidecar = assemble_sidecar(SidecarAssembly {
        parent_hash,
        parent_state_root,
        block_hash,
        block_number,
        cache_block: parent_block_number,
        cache_policy_id,
        cache_policy_metadata: "WeakStateless(no persistent cache)".to_string(),
        prev_cache_anchor,
        next_cache_anchor,
        miss_manifest,
        witness_state: PartialExecutionWitnessState::MptTransitionNodes(build.nodes.clone()),
        codes,
        headers: ancestor_headers.to_vec(),
        stats,
    });

    Ok(FullWitnessBuild { sidecar, build_us: build_start.elapsed().as_micros() as u64 })
}

/// Rebuilds the full-witness sidecar from a witness that was already proved and recorded.
///
/// The offline counterpart to [`build_full_witness_sidecar`], for a caller that holds the node set
/// and needs no proof source at all. It exists so a recorded corpus can be *executed* — the
/// resulting sidecar is the vehicle a database-free re-execution rides, which is how an offline
/// generator derives the block's post state and access set without trusting the record's own claim
/// about either.
///
/// `stats` describes the witness only. The target counts a live build reports come from the proof
/// source's own accounting and have no offline counterpart, so they are left at zero rather than
/// filled in with a number that would look like the same measurement.
pub fn full_witness_sidecar_from_nodes(
    parent_state_root: B256,
    expected_block: BlockTransitionRef<'_>,
    accessed: &BlockAccessedState,
    nodes: Vec<Bytes>,
    config: &CacheConfig,
) -> eyre::Result<PartialStatelessSidecar> {
    let block_number = expected_block.block_number;
    let parent_block_number = block_number.saturating_sub(1);
    let cold_cache = config.new_cache_at(parent_block_number);
    let full_miss = cold_cache.compute_miss(accessed);
    let (miss_manifest, _) = build_sidecar_targets(&full_miss);

    let mut codes = accessed.codes.iter().collect::<Vec<_>>();
    codes.sort_unstable_by_key(|(code_hash, _)| **code_hash);
    let codes = codes.into_iter().map(|(_, code)| code.clone()).collect::<Vec<_>>();
    let bytecode_bytes = codes.iter().map(|code| code.len()).sum();
    let decoded = decode_transition_witness(parent_state_root, &nodes)?;
    let stats = measure_transition_witness_size(&nodes, &decoded, bytecode_bytes);

    let cache_policy_id =
        last_n_blocks_cache_policy_id(config.account_window, config.storage_window);
    let prev_cache_anchor =
        cold_cache.cache_anchor(parent_block_number, expected_block.parent_hash, cache_policy_id);
    let mut next_cache = config.new_cache_at(parent_block_number);
    next_cache.on_block_executed(block_number, accessed);
    let next_cache_anchor =
        next_cache.cache_anchor(block_number, expected_block.block_hash, cache_policy_id);

    Ok(assemble_sidecar(SidecarAssembly {
        parent_hash: expected_block.parent_hash,
        parent_state_root,
        block_hash: expected_block.block_hash,
        block_number,
        cache_block: parent_block_number,
        cache_policy_id,
        cache_policy_metadata: "WeakStateless(no persistent cache)".to_string(),
        prev_cache_anchor,
        next_cache_anchor,
        miss_manifest,
        witness_state: PartialExecutionWitnessState::MptTransitionNodes(nodes),
        codes,
        headers: expected_block.ancestor_headers.to_vec(),
        stats,
    }))
}

/// The result of [`build_full_witness_sidecar`].
#[derive(Debug)]
pub struct FullWitnessBuild {
    /// The sidecar a validator holding nothing would be sent.
    pub sidecar: PartialStatelessSidecar,
    /// Wall time of the whole build.
    pub build_us: u64,
}

impl FullWitnessBuild {
    /// Takes the flat parent-state node set out, consuming the build.
    ///
    /// By move rather than by clone, and handed back separately rather than alongside the sidecar,
    /// because this function is on a measured path: the paired validation benchmark builds a Weak
    /// sidecar per block and reports what it cost, and a copy of the node vector taken for a
    /// caller that may not want it would land in that number. A recorder that wants the witness
    /// body without the policy-bound sidecar around it calls this after it is done with the
    /// sidecar; everyone else never pays.
    pub fn into_nodes(self) -> eyre::Result<Vec<Bytes>> {
        match self.sidecar.witness.state {
            PartialExecutionWitnessState::MptTransitionNodes(nodes) => Ok(nodes),
            PartialExecutionWitnessState::MptMultiProof(_) => Err(eyre::eyre!(
                "the full witness build produced a legacy multiproof rather than transition nodes"
            )),
            PartialExecutionWitnessState::MptTrimmedTransitionNodes { .. } => {
                Err(eyre::eyre!("the full witness build must be self-contained, never trimmed"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_initial_proof_requires_both_storage_width_and_total_work() {
        let storage_a = B256::repeat_byte(0xaa);
        let storage_b = B256::repeat_byte(0xbb);
        let mut too_little_work = V2TargetSet::default();
        too_little_work.insert(TrieProofTargetV2::Storage {
            hashed_address: storage_a,
            key: B256::repeat_byte(0x01),
            min_len: 0,
        });
        too_little_work.insert(TrieProofTargetV2::Storage {
            hashed_address: storage_b,
            key: B256::repeat_byte(0x02),
            min_len: 0,
        });
        assert_eq!(too_little_work.distinct_storage_tries(), 2);
        assert!(!too_little_work.should_use_parallel_initial_proof());

        let mut wide = V2TargetSet::default();
        for index in 0..62u8 {
            let mut key = [0u8; 32];
            key[31] = index;
            wide.insert(TrieProofTargetV2::Account { key: B256::from(key), min_len: 0 });
        }
        wide.insert(TrieProofTargetV2::Storage {
            hashed_address: storage_a,
            key: B256::repeat_byte(0x01),
            min_len: 0,
        });
        wide.insert(TrieProofTargetV2::Storage {
            hashed_address: storage_b,
            key: B256::repeat_byte(0x02),
            min_len: 0,
        });
        assert_eq!(wide.len(), PARALLEL_INITIAL_PROOF_MIN_TOTAL_TARGETS);
        assert!(wide.should_use_parallel_initial_proof());

        wide.storage.remove(&(storage_b, B256::repeat_byte(0x02)));
        wide.insert(TrieProofTargetV2::Storage {
            hashed_address: storage_a,
            key: B256::repeat_byte(0x03),
            min_len: 0,
        });
        assert_eq!(wide.distinct_storage_tries(), 1);
        assert!(!wide.should_use_parallel_initial_proof());
    }

    #[test]
    fn v2_target_difference_handles_accounts_storage_duplicates_and_min_len() {
        let account = B256::repeat_byte(0x11);
        let storage_account = B256::repeat_byte(0x22);
        let slot = B256::repeat_byte(0x33);
        let mut requested = V2TargetSet::default();
        requested.insert(TrieProofTargetV2::Account { key: account, min_len: 8 });
        requested.insert(TrieProofTargetV2::Storage {
            hashed_address: storage_account,
            key: slot,
            min_len: 12,
        });

        let mut desired = V2TargetSet::default();
        desired.extend([
            TrieProofTargetV2::Account { key: account, min_len: 10 },
            TrieProofTargetV2::Account { key: account, min_len: 10 },
            TrieProofTargetV2::Storage { hashed_address: storage_account, key: slot, min_len: 5 },
        ]);
        let delta = desired.difference_and_record(&mut requested);

        assert!(delta.accounts.is_empty(), "shallower account proof already covers target");
        assert_eq!(delta.storage.get(&(storage_account, slot)), Some(&5));
        assert_eq!(requested.storage.get(&(storage_account, slot)), Some(&5));
        assert!(desired.difference_and_record(&mut requested).is_empty());
    }

    #[test]
    fn flat_target_normalization_is_deterministic() {
        let account = B256::repeat_byte(0x44);
        let storage_account = B256::repeat_byte(0x55);
        let slot = B256::repeat_byte(0x66);
        let mut targets = V2TargetSet::default();
        targets.insert(TrieProofTargetV2::Storage {
            hashed_address: storage_account,
            key: slot,
            min_len: 19,
        });
        targets.insert(TrieProofTargetV2::Account { key: account, min_len: 7 });

        let flat = targets.with_flat_storage_context();
        assert_eq!(flat.accounts.get(&account), Some(&0));
        assert_eq!(flat.accounts.get(&storage_account), Some(&0));
        assert_eq!(flat.storage.get(&(storage_account, slot)), Some(&0));
    }
}
