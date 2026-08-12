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
    spool::read_spool,
};
use alloy_rlp::Decodable;
use partial_stateless::{
    restore_snapshot, CacheConfig, PartialStatelessSidecar, TrustedCheckpoint,
};
use partial_stateless_stream::{
    BlockRef, Checkpoint, CommitInput, CommitOracle, FrameLimits, Manifest, SnapshotChunk,
    StreamEvent,
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
    /// Whether the corpus ended with an `End` frame.
    pub closed: bool,
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
pub fn replay(dir: &Path, options: &ReplayOptions) -> eyre::Result<ReplayReport> {
    let spool = read_spool(dir, &options.frame_limits)?;
    info!(
        target: "ps_replay",
        dir = %dir.display(),
        frames = spool.frames.len(),
        bytes = spool.bytes,
        closed = spool.closed,
        "Read the recorded stream"
    );
    if !spool.closed {
        warn!(
            target: "ps_replay",
            "The corpus has no End frame, so it was cut rather than finished. Everything below \
             describes the prefix that survived"
        );
    }

    let mut report = ReplayReport { closed: spool.closed, ..Default::default() };
    let mut manifest: Option<Manifest> = None;
    let mut checkpoint: Option<Checkpoint> = None;
    let mut chunks: Vec<SnapshotChunk> = Vec::new();
    let mut restored: Option<ReplayState> = None;

    for frame in spool.frames {
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
            StreamEvent::Checkpoint(found) => checkpoint = Some(found),
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
                replay_commit(state, input, &oracle, options, &mut report);
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
                info!(target: "ps_replay", reason = %end.reason, "Stream ended");
            }
        }
    }

    Ok(report)
}

/// Everything a replay carries between commits.
///
/// The rules are built once and held here, not per block. `EthBeaconConsensus` carries flags that
/// decide what a block is allowed to be, so a validator that rebuilt it per block would be one
/// configuration change away from disagreeing with itself mid-stream — and it would charge the
/// construction to every measured block.
struct ReplayState {
    pair: CoordinatedPair,
    config: CacheConfig,
    chain_spec: Arc<ChainSpec>,
    consensus: EthBeaconConsensus<ChainSpec>,
    evm_config: EthEvmConfig<ChainSpec>,
}

/// Restores the pair a replay validates against, from the checkpoint and its chunks.
fn restore(
    manifest: &Manifest,
    checkpoint: &Checkpoint,
    chunks: &[SnapshotChunk],
) -> eyre::Result<ReplayState> {
    let package_bytes = checkpoint
        .reassemble(chunks)
        .map_err(|err| eyre::eyre!("recorded snapshot did not reassemble: {err}"))?;
    let package = bincode::deserialize(&package_bytes)
        .map_err(|err| eyre::eyre!("recorded snapshot package did not decode: {err}"))?;

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

/// Decodes the checkpoint's accepted head, refusing any header that disagrees with it.
fn decode_accepted_head(checkpoint: &Checkpoint) -> Option<SealedHeader> {
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

fn chain_spec_for(manifest: &Manifest) -> eyre::Result<Arc<ChainSpec>> {
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
fn replay_commit(
    state: &mut ReplayState,
    input: CommitInput,
    oracle: &CommitOracle,
    options: &ReplayOptions,
    report: &mut ReplayReport,
) {
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
            return
        }
        Err(err) => {
            report.failures.push(format!("{label}: recorded payload did not parse: {err}"));
            return
        }
    };
    let sidecar: PartialStatelessSidecar = match bincode::deserialize(&input.sidecar) {
        Ok(sidecar) => sidecar,
        Err(err) => {
            report.failures.push(format!("{label}: recorded sidecar did not decode: {err}"));
            return
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
            return
        }
    };
    report.admission_us += admitted.timings.total_us();

    let block_ctx = block_context(&admitted.block);
    if let BlockAdmission::Rejected(reason) = admit_block(&mut state.pair.readiness, &block_ctx) {
        report.failures.push(format!("{label}: the restored pair refused the block: {reason:?}"));
        return
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
    report.transition_us += started.elapsed().as_micros() as u64;

    let validated = match validated {
        Ok(validated) => validated,
        Err(err) => {
            report.failures.push(format!("{label}: the transition failed: {err:#}"));
            return
        }
    };
    let mut outcome = validated.outcome;
    let displaced = outcome.displaced_trie_cache.take();
    state.pair.commit_transition(displaced, &block_ctx, admitted.block.clone_sealed_header(), true);

    let disagreements = compare_accepted(oracle, &outcome, &state.pair);
    if disagreements.is_empty() {
        report.commits += 1;
        return
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
