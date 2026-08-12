//! Restoring a pair from a recorded checkpoint and running the recorded commits through it.
//!
//! The sequence per commit is the standalone path in full, from bytes: decode the Engine payload,
//! admit it against the pair's *own* accepted parent, and only then run the transition. Nothing is
//! taken from the frame that a live validator would not have. In particular the parent header is
//! read from the pair rather than from the commit, which is the rule S1b established and the one a
//! replay is most tempted to break, because the frame is right there and it is correct.

use crate::{
    compare::{block_label, compare_accepted, compare_rejected, Disagreement},
    mutate::Mutation,
    spool::SpoolIter,
};
use alloy_rlp::Decodable;
use partial_stateless::{
    restore_snapshot, CacheConfig, PartialStatelessSidecar, TrustedCheckpoint,
};
use partial_stateless_stream::{
    BlockRef, Checkpoint, CommitInput, CommitOracle, FrameLimits, Manifest, SnapshotChunk,
    StreamEvent, DEFAULT_MAX_SNAPSHOT_BYTES,
};
use partial_stateless_validator::{
    admit_block, block_context, verify_and_apply_sidecar, AdmissionError, BlockAdmission,
    CoordinatedPair, PayloadProvenance, SidecarReexecLimits, TrieCacheDisposition,
    UntrustedAdmission, ValidatorRules,
};
use reth_chainspec::{ChainSpec, MAINNET};
use reth_ethereum_consensus::EthBeaconConsensus;
use reth_evm_ethereum::EthEvmConfig;
use reth_primitives_traits::{Header, SealedHeader};
use std::{path::Path, sync::Arc, time::Instant};
use tracing::{error, info, warn};

/// How much of a corpus to replay, and what to do beyond checking it.
#[derive(Debug, Clone)]
pub struct ReplayOptions {
    /// Stop after this many commits. `None` replays the whole corpus.
    pub limit: Option<usize>,
    /// Derive negative frames from every witnessed commit and check the class each must produce.
    ///
    /// On by default, because a replay of a mainnet corpus without it proves the accept path only
    /// and reads as though it proved more.
    pub mutations: bool,
    /// Bounds on frame decoding.
    pub frame_limits: FrameLimits,
    /// Bounds on sidecar witness decoding.
    pub reexec_limits: SidecarReexecLimits,
}

impl Default for ReplayOptions {
    fn default() -> Self {
        Self {
            limit: None,
            mutations: true,
            frame_limits: FrameLimits::default(),
            reexec_limits: SidecarReexecLimits::default(),
        }
    }
}

/// What one replay found.
#[derive(Debug, Default)]
pub struct ReplayReport {
    /// Commits replayed.
    pub commits: u64,
    /// Commits whose payload was the one a consensus client sent.
    pub witnessed: u64,
    /// Commits whose payload was derived from a block by the producer.
    pub reconstructed: u64,
    /// Commits carrying no payload, which admission cannot run on.
    pub absent: u64,
    /// Every field on which a replay and the recording disagreed, with the block it was on.
    pub disagreements: Vec<(BlockRef, Disagreement)>,
    /// Blocks the replay refused that the recording accepted, or that failed to decode.
    pub failures: Vec<String>,
    /// Negative frames derived and checked.
    pub mutations_checked: u64,
    /// Negative frames that produced the wrong class, or none at all.
    pub mutation_failures: Vec<String>,
    /// Total admission wall time across every commit, in microseconds.
    pub admission_us: u64,
    /// Total transition wall time across every commit, in microseconds.
    pub transition_us: u64,
    /// Per-block timings, in the order they were replayed.
    ///
    /// Kept per block rather than only as totals because the A/B this corpus exists to enable is a
    /// *paired* comparison: the same block replayed by two builds is the one comparison with no
    /// workload variance in it at all, and a total would throw that pairing away.
    pub blocks: Vec<BlockTiming>,
    /// Whether the corpus ended with an `End` frame.
    pub closed: bool,
    /// The fault that stopped the replay, when one did. Everything after it was skipped.
    pub terminal: Option<String>,
    /// Commit frames not replayed because a fault preceded them.
    ///
    /// Counted rather than reported one by one: before this existed, every commit after a fault
    /// generated its own `BlockSkipped` failure string, and a single fault read as a cascade.
    pub skipped_after_fault: u64,
}

/// Why a replay can go no further on this pair. Every variant names the block it stopped on.
///
/// This is the transaction boundary the ExEx never needed: its process fail-stops on the first
/// error, so nothing ever observed the readiness tracker stranded in `Applying` after a
/// post-execution rejection. A standalone driver outlives the rejection, so the boundary is made
/// explicit here — the pair is moved to terminal `Blocked` before anything can observe it.
#[derive(Debug, Clone)]
pub enum ReplayFault {
    /// A failure after admission, inside the transition. The flat and trie caches are preserved
    /// at the parent generation (the transition's own rollback contract); readiness is moved to
    /// terminal `Blocked` explicitly rather than left in transient `Applying`.
    TransitionFailed {
        /// The block the transition failed on.
        block: BlockRef,
        /// What the transition objected to.
        detail: String,
    },
    /// The restored pair itself refused a block the recording accepted. The refusal latched the
    /// tracker `Blocked`; the pair was not touched past that.
    ReadinessRefused {
        /// The block the pair refused.
        block: BlockRef,
        /// What readiness objected to.
        reason: String,
    },
}

impl std::fmt::Display for ReplayFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TransitionFailed { block, detail } => {
                write!(f, "{}: the transition failed: {detail}", block_label(*block))
            }
            Self::ReadinessRefused { block, reason } => {
                write!(f, "{}: the restored pair refused the block: {reason}", block_label(*block))
            }
        }
    }
}

/// What one commit's replay did to the pair.
pub(crate) enum CommitOutcome {
    /// The commit ran to a verdict and was compared against the oracle.
    Compared,
    /// The commit could not run and touched no validator state; the next commit may proceed.
    Rejected,
    /// The pair can go no further; the driver must stop replaying commits.
    Fault(ReplayFault),
}

/// What one replayed block cost.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct BlockTiming {
    /// Height.
    pub number: u64,
    /// Decode, payload layout, block hash, sender recovery, and pre-execution consensus.
    pub admission_us: u64,
    /// The cache transition: witness materialization, execution, root, retention, anchor.
    pub transition_us: u64,
}

impl ReplayReport {
    /// Whether the replay agreed with the recording everywhere it could.
    pub fn agreed(&self) -> bool {
        self.disagreements.is_empty() &&
            self.failures.is_empty() &&
            self.mutation_failures.is_empty()
    }

    /// Whether this run's admission checks were checking anything.
    ///
    /// A corpus of reconstructions exercises the code and proves nothing about the rules, so a
    /// report that did not say this would be read as stronger than it is.
    pub const fn admission_is_load_bearing(&self) -> bool {
        self.witnessed > 0
    }
}

/// Replays a recorded stream and checks it against its own oracle.
///
/// Frames are read one at a time rather than materialized: a long corpus is tens of gigabytes of
/// commits, and holding it whole would make the corpus-as-evidence design unusable at exactly the
/// lengths that matter.
pub fn replay(dir: &Path, options: &ReplayOptions) -> eyre::Result<ReplayReport> {
    let mut spool = SpoolIter::open(dir, &options.frame_limits)?;

    let mut report = ReplayReport::default();
    let mut manifest: Option<Manifest> = None;
    let mut checkpoint: Option<Checkpoint> = None;
    let mut chunks: Vec<SnapshotChunk> = Vec::new();
    let mut restored: Option<ReplayState> = None;

    while let Some(frame) = spool.next_frame()? {
        // A faulted pair replays nothing further. The remaining frames are counted rather than
        // run: replaying commits against a blocked pair would generate one `BlockSkipped`
        // failure per block, and a single fault would read as a cascade.
        if report.terminal.is_some() {
            if matches!(frame.event, StreamEvent::Commit(_)) {
                report.skipped_after_fault += 1;
            }
            continue
        }
        match frame.event {
            StreamEvent::Manifest(found) => {
                info!(
                    target: "ps_replay",
                    chain_id = found.chain_id,
                    epoch = found.epoch,
                    producer = %found.producer,
                    account_window = found.account_window,
                    storage_window = found.storage_window,
                    "Stream manifest"
                );
                manifest = Some(found);
            }
            StreamEvent::Checkpoint(found) => {
                // The declaration is checked before any chunk is buffered, so a corrupt or
                // hostile checkpoint cannot turn its own transport fields into an allocation.
                found
                    .validate_declared(DEFAULT_MAX_SNAPSHOT_BYTES)
                    .map_err(|err| eyre::eyre!("checkpoint declaration refused: {err}"))?;
                checkpoint = Some(found);
            }
            StreamEvent::SnapshotChunk(chunk) => chunks.push(chunk),
            StreamEvent::Commit(commit) => {
                let state = match restored.as_mut() {
                    Some(state) => state,
                    None => {
                        let manifest = manifest
                            .as_ref()
                            .ok_or_else(|| eyre::eyre!("a commit arrived before the manifest"))?;
                        let checkpoint = checkpoint
                            .as_ref()
                            .ok_or_else(|| eyre::eyre!("a commit arrived before the checkpoint"))?;
                        restored = Some(restore(manifest, checkpoint, &chunks)?);
                        restored.as_mut().expect("just restored")
                    }
                };
                if options.limit.is_some_and(|limit| report.commits as usize >= limit) {
                    break
                }
                let (input, oracle) = commit.split();
                let block = input.block;
                if let CommitOutcome::Fault(fault) =
                    replay_commit(state, input, &oracle, options, &mut report)
                {
                    error!(
                        target: "ps_replay",
                        block = block.number,
                        %fault,
                        readiness = state.pair.readiness.state().label(),
                        "The pair can go no further; the remaining commits are skipped, not \
                         replayed"
                    );
                    report.failures.push(fault.to_string());
                    report.terminal = Some(fault.to_string());
                }
            }
            StreamEvent::Reorg(reorg) => {
                // Recorded in v1 and not yet replayed: depth-1 rollback against the retained
                // generation is S4's gate, and claiming it here by skipping the event would be
                // exactly the silent drop the format exists to prevent.
                warn!(
                    target: "ps_replay",
                    common_ancestor = reorg.common_ancestor.number,
                    abandoned = reorg.abandoned.len(),
                    "Corpus contains a reorg, which this driver does not yet apply; the commits \
                     after it are replayed against a pair that never unwound and their results are \
                     not evidence"
                );
                report.failures.push(format!(
                    "reorg at {} is not replayed by this driver",
                    block_label(reorg.common_ancestor)
                ));
            }
            StreamEvent::Reset(reset) => {
                report.failures.push(format!(
                    "producer recorded a reset ({:?}): {}",
                    reset.reason, reset.detail
                ));
            }
            StreamEvent::End(end) => {
                // Orderly termination, not success: the producer's close path ran, and the kind
                // says under what circumstances.
                info!(
                    target: "ps_replay",
                    kind = end.kind.as_str(),
                    reason = %end.reason,
                    last_sequence = end.last_sequence,
                    "Stream ended"
                );
            }
        }
    }

    report.closed = spool.closed();
    info!(
        target: "ps_replay",
        dir = %dir.display(),
        bytes = spool.bytes(),
        closed = report.closed,
        "Read the recorded stream"
    );
    if !report.closed {
        warn!(
            target: "ps_replay",
            "The corpus has no End frame, so it was cut rather than finished. Everything above \
             describes the prefix that survived"
        );
    }

    Ok(report)
}

/// Everything a replay carries between commits.
///
/// The rules are built once and held here, not per block. `EthBeaconConsensus` carries flags that
/// decide what a block is allowed to be, so a validator that rebuilt it per block would be one
/// configuration change away from disagreeing with itself mid-stream — and it would charge the
/// construction to every measured block.
pub(crate) struct ReplayState {
    pub(crate) pair: CoordinatedPair,
    pub(crate) config: CacheConfig,
    pub(crate) chain_spec: Arc<ChainSpec>,
    pub(crate) consensus: EthBeaconConsensus<ChainSpec>,
    pub(crate) evm_config: EthEvmConfig<ChainSpec>,
}

/// Restores the pair a replay validates against, from the checkpoint and its chunks.
pub(crate) fn restore(
    manifest: &Manifest,
    checkpoint: &Checkpoint,
    chunks: &[SnapshotChunk],
) -> eyre::Result<ReplayState> {
    let package_bytes = checkpoint
        .reassemble(chunks)
        .map_err(|err| eyre::eyre!("recorded snapshot did not reassemble: {err}"))?;
    let package = bincode::deserialize(&package_bytes)
        .map_err(|err| eyre::eyre!("recorded snapshot package did not decode: {err}"))?;

    let config = config_for(manifest)?;
    let trusted = TrustedCheckpoint {
        block_number: checkpoint.block.number,
        block_hash: checkpoint.block.hash,
        state_root: checkpoint.state_root,
        cache_root: checkpoint.cache_root,
        cache_policy_id: checkpoint.cache_policy_id,
    };
    let restored = restore_snapshot(package, &trusted, &config)?;

    // The header is installed only because every field a consumer checks it against is in the
    // checkpoint the operator vouched for. A header that fails any of them is dropped, and the
    // pair then waits a block rather than admitting its first child against an unverified parent.
    let accepted_head = decode_accepted_head(checkpoint);

    let chain_spec = chain_spec_for(manifest)?;
    info!(
        target: "ps_replay",
        block = checkpoint.block.number,
        block_hash = ?checkpoint.block.hash,
        accounts = restored.cache.accounts().len(),
        storage = restored.cache.storage().len(),
        codes = restored.cache.codes().len(),
        has_accepted_head = accepted_head.is_some(),
        "Restored a coordinated pair from the recorded checkpoint, with no database"
    );
    Ok(ReplayState {
        pair: CoordinatedPair {
            cache: restored.cache,
            trie_cache: restored.trie_cache,
            previous_generation: None,
            accepted_head,
            readiness: restored.readiness,
        },
        config,
        consensus: EthBeaconConsensus::new(chain_spec.clone()),
        evm_config: EthEvmConfig::new(chain_spec.clone()),
        chain_spec,
    })
}

/// The cache configuration a manifest names, cross-checked against its own policy id.
pub(crate) fn config_for(manifest: &Manifest) -> eyre::Result<CacheConfig> {
    let config = CacheConfig {
        account_window: manifest.account_window,
        storage_window: manifest.storage_window,
    };
    if config.cache_policy_id() != manifest.cache_policy_id {
        eyre::bail!(
            "manifest names policy {:?} but its own windows derive {:?}",
            manifest.cache_policy_id,
            config.cache_policy_id()
        );
    }
    Ok(config)
}

/// Decodes the checkpoint's accepted head, refusing any header that disagrees with it.
pub(crate) fn decode_accepted_head(checkpoint: &Checkpoint) -> Option<SealedHeader> {
    if checkpoint.accepted_head_rlp.is_empty() {
        return None
    }
    let header = match Header::decode(&mut checkpoint.accepted_head_rlp.as_slice()) {
        Ok(header) => header,
        Err(err) => {
            warn!(target: "ps_replay", %err, "Checkpoint header did not decode; ignoring it");
            return None
        }
    };
    let sealed = SealedHeader::seal_slow(header);
    let agrees = sealed.hash() == checkpoint.block.hash &&
        sealed.number == checkpoint.block.number &&
        sealed.state_root == checkpoint.state_root;
    if !agrees {
        warn!(
            target: "ps_replay",
            ?sealed,
            expected = ?checkpoint.block,
            "Checkpoint header does not match the checkpoint; ignoring it"
        );
        return None
    }
    Some(sealed)
}

pub(crate) fn chain_spec_for(manifest: &Manifest) -> eyre::Result<Arc<ChainSpec>> {
    // One chain for now, and named rather than inferred: a validator that guessed a chain spec
    // from a chain id would be choosing fork activation times on the producer's behalf.
    if manifest.chain_id != MAINNET.chain.id() {
        eyre::bail!(
            "this driver is configured for mainnet ({}); the stream names chain {}",
            MAINNET.chain.id(),
            manifest.chain_id
        );
    }
    if manifest.genesis_hash != MAINNET.genesis_hash() {
        eyre::bail!(
            "the stream names genesis {:?}, which is not mainnet's {:?}",
            manifest.genesis_hash,
            MAINNET.genesis_hash()
        );
    }
    Ok(MAINNET.clone())
}

/// Runs one recorded commit through admission and the transition, then compares.
///
/// What each early return leaves behind is part of the contract, not an accident. A decode or
/// admission failure touches no validator state — `UntrustedAdmission` is stateless with respect
/// to the pair — so the next commit may still run ([`CommitOutcome::Rejected`]). Everything past
/// `admit_block` has moved readiness to `Applying`, so a failure there must not return with the
/// tracker still in that transient state: the ExEx's fail-stop masked exactly that leak, and a
/// standalone driver that outlives the rejection makes it observable.
pub(crate) fn replay_commit(
    state: &mut ReplayState,
    input: CommitInput,
    oracle: &CommitOracle,
    options: &ReplayOptions,
    report: &mut ReplayReport,
) -> CommitOutcome {
    match input.payload_provenance {
        PayloadProvenance::Witnessed => report.witnessed += 1,
        PayloadProvenance::Reconstructed => report.reconstructed += 1,
        PayloadProvenance::Absent => report.absent += 1,
    }
    let label = block_label(input.block);

    let payload = match input.payload() {
        Ok(Some(payload)) => payload,
        Ok(None) => {
            report.failures.push(format!(
                "{label}: the commit carries no payload, so admission could not run on it"
            ));
            return CommitOutcome::Rejected
        }
        Err(err) => {
            report.failures.push(format!("{label}: recorded payload did not parse: {err}"));
            return CommitOutcome::Rejected
        }
    };
    let sidecar: PartialStatelessSidecar = match bincode::deserialize(&input.sidecar) {
        Ok(sidecar) => sidecar,
        Err(err) => {
            report.failures.push(format!("{label}: recorded sidecar did not decode: {err}"));
            return CommitOutcome::Rejected
        }
    };

    let admission = UntrustedAdmission::new(state.chain_spec.as_ref(), &state.consensus);

    if options.mutations && input.payload_provenance.is_load_bearing() {
        check_mutations(&admission, &state.pair, &payload, report, &label);
    }

    // The parent comes from the pair and never from the frame. A producer that supplied the parent
    // would be choosing the timestamp, gas limit, and base fee its own block is measured against.
    let admitted = match admission.admit(payload, state.pair.accepted_parent()) {
        Ok(admitted) => admitted,
        Err(err) => {
            let disagreements = compare_rejected(oracle, err.class());
            report.failures.push(format!(
                "{label}: the replay refused a block the recording accepted: {err} ({})",
                err.class()
            ));
            report.disagreements.extend(disagreements.into_iter().map(|d| (input.block, d)));
            return CommitOutcome::Rejected
        }
    };
    let admission_us = admitted.timings.total_us();
    report.admission_us += admission_us;

    let block_ctx = block_context(&admitted.block);
    if let BlockAdmission::Rejected(reason) = admit_block(&mut state.pair.readiness, &block_ctx) {
        // The refusal itself latched the tracker `Blocked`; the caches were never touched.
        return CommitOutcome::Fault(ReplayFault::ReadinessRefused {
            block: input.block,
            reason: format!("{reason:?}"),
        })
    }

    let started = Instant::now();
    let validated = verify_and_apply_sidecar(
        ValidatorRules::new(&state.evm_config, &state.consensus),
        &admitted.block,
        &mut state.pair.cache,
        &sidecar,
        state.config.cache_policy_id(),
        &options.reexec_limits,
        &mut state.pair.trie_cache,
        TrieCacheDisposition::Commit,
    );
    let transition_us = started.elapsed().as_micros() as u64;
    report.transition_us += transition_us;
    report.blocks.push(BlockTiming { number: input.block.number, admission_us, transition_us });

    let validated = match validated {
        Ok(validated) => validated,
        Err(err) => {
            return CommitOutcome::Fault(fail_applied_block(
                &mut state.pair,
                input.block,
                format!("{err:#}"),
            ))
        }
    };
    let mut outcome = validated.outcome;
    let displaced = outcome.displaced_trie_cache.take();
    state.pair.commit_transition(displaced, &block_ctx, admitted.block.clone_sealed_header(), true);

    let disagreements = compare_accepted(oracle, &outcome, &state.pair);
    if disagreements.is_empty() {
        report.commits += 1;
        return CommitOutcome::Compared
    }
    for disagreement in disagreements {
        error!(
            target: "ps_replay",
            block = input.block.number,
            field = disagreement.field,
            recorded = %disagreement.recorded,
            replayed = %disagreement.replayed,
            "The replay disagreed with the recording. One of the two is wrong, and which one is \
             an investigation rather than an assumption"
        );
        report.disagreements.push((input.block, disagreement));
    }
    report.commits += 1;
    CommitOutcome::Compared
}

/// Closes the admit-verify-apply boundary after a post-admission failure.
///
/// `admit_block` moved readiness to `Applying`; the failed transition preserved both caches at
/// the parent generation but has no readiness handle to report through. Left there, the tracker
/// would refuse the next block as `BlockSkipped` while *looking* transient — under the ExEx the
/// process dies first, and only a standalone driver ever observes the difference. `abandon_block`
/// makes the stop explicit and terminal: the watermark freezes at the parent, and nothing short
/// of a reset or a reorg below the block releases it.
fn fail_applied_block(pair: &mut CoordinatedPair, block: BlockRef, detail: String) -> ReplayFault {
    pair.readiness.abandon_block(block.number);
    ReplayFault::TransitionFailed { block, detail }
}

/// Derives negative frames from a recorded payload and checks the class each must produce.
///
/// Runs against the same pair the honest commit will use, and deliberately *before* it: admission
/// touches no validator state, so a rejected mutation leaves the pair exactly where it was, and
/// running them first means a mutation that wrongly succeeded would be caught rather than hidden
/// by the honest block that followed.
fn check_mutations<C>(
    admission: &UntrustedAdmission<'_, ChainSpec, C>,
    pair: &CoordinatedPair,
    payload: &alloy_rpc_types_engine::ExecutionData,
    report: &mut ReplayReport,
    label: &str,
) where
    C: reth_consensus::Consensus<
            alloy_consensus::Block<reth_ethereum_primitives::TransactionSigned>,
        > + ?Sized,
{
    for mutation in Mutation::ALL {
        let mutated = match mutation.apply(payload) {
            Ok(mutated) => mutated,
            Err(err) => {
                report
                    .mutation_failures
                    .push(format!("{label}/{}: could not derive: {err}", mutation.as_str()));
                continue
            }
        };
        report.mutations_checked += 1;
        match admission.admit(mutated, pair.accepted_parent()) {
            Ok(_) => report.mutation_failures.push(format!(
                "{label}/{}: admitted a block that must have been refused as {}",
                mutation.as_str(),
                mutation.expected_class()
            )),
            Err(err) if err.class() != mutation.expected_class() => {
                report.mutation_failures.push(format!(
                    "{label}/{}: refused as {} but must be {}: {err}",
                    mutation.as_str(),
                    err.class(),
                    mutation.expected_class()
                ));
            }
            Err(_) => {}
        }
    }

    // The one negative no frame can carry: a pair that cannot name a parent must refuse the block
    // rather than run the subset of rules that survive without one.
    match admission.admit(payload.clone(), None) {
        Ok(_) => report.mutation_failures.push(format!(
            "{label}/no_accepted_parent: admitted a block with no parent to check it against"
        )),
        Err(AdmissionError::NoAcceptedParent { .. }) => report.mutations_checked += 1,
        Err(err) => {
            report.mutations_checked += 1;
            report.mutation_failures.push(format!(
                "{label}/no_accepted_parent: refused as {} rather than no_accepted_parent: {err}",
                err.class()
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{fail_applied_block, ReplayFault};
    use alloy_primitives::B256;
    use partial_stateless::{readiness::BlockContext, CacheConfig, PartialTrieNodeCache};
    use partial_stateless_stream::BlockRef;
    use partial_stateless_validator::{admit_block, BlockAdmission, CoordinatedPair};

    fn pair() -> CoordinatedPair {
        let config = CacheConfig::default();
        CoordinatedPair {
            cache: config.new_cache(),
            trie_cache: PartialTrieNodeCache::new(),
            previous_generation: None,
            accepted_head: None,
            readiness: config.new_readiness_tracker(),
        }
    }

    fn ctx(number: u64) -> BlockContext {
        BlockContext {
            number,
            hash: B256::with_last_byte(number as u8),
            parent_hash: B256::with_last_byte(number as u8 - 1),
            state_root: B256::with_last_byte(0x55),
        }
    }

    /// The leak S1 recorded and this boundary closes: a post-execution rejection preserved both
    /// caches but left readiness in transient `Applying`, which only the ExEx's fail-stop made
    /// safe. A standalone driver outlives the rejection, so the stop must be explicit, terminal,
    /// and observable — and it must not have touched what the rejection promised to preserve.
    #[test]
    fn a_post_admission_failure_moves_readiness_to_terminal_blocked() {
        let mut pair = pair();
        let block = ctx(100);
        assert!(matches!(admit_block(&mut pair.readiness, &block), BlockAdmission::Admitted(_)));
        assert_eq!(pair.readiness.state().label(), "applying", "the leak's starting point");
        let parent_fingerprint = pair.fingerprint();

        let fault = fail_applied_block(
            &mut pair,
            BlockRef { number: block.number, hash: block.hash },
            "witness self-consistency refused".to_string(),
        );

        assert!(matches!(fault, ReplayFault::TransitionFailed { .. }));
        assert_eq!(pair.readiness.state().label(), "blocked", "terminal, not transient");
        assert_eq!(
            pair.fingerprint(),
            parent_fingerprint,
            "the boundary reports the stop; it does not touch the caches"
        );
        assert_eq!(
            pair.readiness.first_gap(),
            Some(block.number),
            "the acknowledgement watermark froze at the failed block"
        );
        assert!(
            matches!(admit_block(&mut pair.readiness, &ctx(101)), BlockAdmission::Rejected(_)),
            "nothing runs against a pair that stopped"
        );
    }

    /// The fault on the very last commit is the case the old code got wrong silently: with no
    /// next block to trip over the stranded `Applying`, the report was the only witness.
    #[test]
    fn a_fault_on_the_final_commit_still_leaves_the_pair_blocked() {
        let mut pair = pair();
        let block = ctx(200);
        assert!(matches!(admit_block(&mut pair.readiness, &block), BlockAdmission::Admitted(_)));

        fail_applied_block(
            &mut pair,
            BlockRef { number: block.number, hash: block.hash },
            "anchor mismatch".to_string(),
        );

        assert_eq!(pair.readiness.state().label(), "blocked");
    }
}
