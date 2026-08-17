//! Turning untrusted consensus-layer input into a block this validator agreed to execute.
//!
//! This is the standalone validator's outermost boundary, and the one place where nothing may be
//! taken on trust. A full node's ExEx never comes here: its Engine already decoded the payload,
//! checked its layout, and recovered every sender before the block reached the exex, so it enters
//! the core through the recovered path and pays none of this. A standalone validator has no Engine
//! in front of it and must do all of it itself.
//!
//! **The input is an Engine-API payload, not a wire block.** A consensus client receives a signed
//! beacon block, extracts its `ExecutionPayload`, and hands it to the execution layer alongside the
//! expected blob versioned hashes, the parent beacon block root, and the execution requests. That
//! bundle is [`ExecutionData`], and taking anything narrower — a bare sealed block, say — would
//! quietly drop the fields that only exist at that boundary: the block hash the consensus layer
//! expects, and the versioned hashes the payload's blob transactions must match. Those are checks
//! nothing downstream repeats.
//!
//! What is deliberately not reimplemented here: the layout rules themselves. `reth-payload-
//! validator` holds them and depends on nothing but primitives and alloy engine types, so they
//! arrive with their fork gating intact. Reth's own `ensure_well_formed_payload` is assembled from
//! the same pieces, but it lives in `reth-ethereum-payload-builder`, which reaches the transaction
//! pool and a state provider — this crate cannot name it. The order below is the order
//! `reth-engine-tree` validates in, for the same reason this crate delegates rather than
//! hand-writes: a subset that agrees with canonical rules today is a subset that disagrees after
//! the next fork.

use crate::timings::{AdmissionSource, AdmissionTimings};
use alloy_consensus::{Block as AlloyBlock, BlockHeader};
use alloy_rpc_types_engine::{ExecutionData, PayloadError};
use reth_chainspec::EthereumHardforks;
use reth_consensus::{Consensus, ConsensusError};
use reth_ethereum_primitives::TransactionSigned;
use reth_payload_validator::{cancun, prague, shanghai};
use reth_primitives_traits::{Block as _, RecoveredBlock, SealedBlock, SealedHeader};
use std::time::Instant;

/// The consensus rules a standalone validator admits untrusted payloads under.
///
/// Counterpart to [`ValidatorRules`](crate::ValidatorRules), which covers execution. Both are
/// built from the one chain spec the validator was configured with, and neither may be assembled
/// per block: the chain spec decides fork activation and the consensus object carries flags that
/// decide what a block is allowed to be.
#[derive(Debug, Clone, Copy)]
pub struct UntrustedAdmission<'a, ChainSpec, Consensus: ?Sized> {
    chain_spec: &'a ChainSpec,
    consensus: &'a Consensus,
}

impl<'a, ChainSpec, C> UntrustedAdmission<'a, ChainSpec, C>
where
    ChainSpec: EthereumHardforks,
    C: Consensus<AlloyBlock<TransactionSigned>> + ?Sized,
{
    pub const fn new(chain_spec: &'a ChainSpec, consensus: &'a C) -> Self {
        Self { chain_spec, consensus }
    }

    /// Admits one Engine-API payload, or rejects it without touching any validator state.
    ///
    /// `accepted_parent` is the validator's **own** record of the block it last accepted, from
    /// [`CoordinatedPair::accepted_parent`](crate::CoordinatedPair::accepted_parent). It is not an
    /// input to this boundary and must never be read out of the payload: parent-dependent rules
    /// are the ones a dishonest producer has the most to gain from, and a producer that supplies
    /// the parent chooses the timestamp, gas limit, and base fee its block is measured against.
    ///
    /// `None` is a rejection rather than a skip. A pair that cannot name its parent — cold,
    /// warming from nothing, or freshly restored from a snapshot that does not yet carry a header
    /// — cannot check the rules that need one, and admitting the block anyway would silently
    /// weaken the check to whatever survives without a parent.
    pub fn admit(
        &self,
        payload: ExecutionData,
        accepted_parent: Option<&SealedHeader>,
    ) -> Result<AdmittedBlock, AdmissionError> {
        let payload_start = Instant::now();
        let sealed = self.ensure_well_formed_payload(payload)?;
        let payload_validation_us = payload_start.elapsed().as_micros() as u64;

        let consensus_start = Instant::now();
        self.consensus.validate_header(sealed.sealed_header())?;
        self.consensus.validate_block_pre_execution(&sealed)?;
        // Not `validate_body_against_header`: `validate_block_pre_execution` already covers the
        // ommers hash, the transactions root, Shanghai withdrawals, Cancun blob gas, and the Osaka
        // block size limit. Calling both would recompute the transaction trie for nothing.
        let Some(parent) = accepted_parent else {
            return Err(AdmissionError::NoAcceptedParent { block_number: sealed.number() })
        };
        self.consensus.validate_header_against_parent(sealed.sealed_header(), parent)?;
        let pre_execution_consensus_us = consensus_start.elapsed().as_micros() as u64;

        // Checked recovery, so a signature outside the EIP-2 low-`s` range is refused rather than
        // resolved to whatever address it happens to produce. An unrecoverable sender is an
        // input-validation failure and not an execution failure: the block never becomes something
        // this validator would run, so nothing downstream ever sees it.
        let recovery_start = Instant::now();
        let block = sealed.try_recover().map_err(|_| AdmissionError::SenderRecovery)?;
        let sender_recovery_us = recovery_start.elapsed().as_micros() as u64;

        Ok(AdmittedBlock {
            block,
            timings: AdmissionTimings {
                source: AdmissionSource::ExecutionData,
                // Filled in by whatever read the payload off the wire; this function is handed one
                // already parsed. The spool/socket frame reader is what sets it.
                input_decode_us: None,
                payload_validation_us: Some(payload_validation_us),
                sender_recovery_us: Some(sender_recovery_us),
                pre_execution_consensus_us: Some(pre_execution_consensus_us),
            },
        })
    }

    /// Rebuilds the block from the payload and checks everything that is true of its *layout*.
    ///
    /// Mirrors `reth_ethereum_payload_builder::ensure_well_formed_payload`, which this crate's
    /// dependency graph cannot reach. The block hash comparison is the load-bearing line: every
    /// later check reads fields out of the reconstructed block, and without it a producer could
    /// hand over a block whose contents and whose announced identity disagree.
    fn ensure_well_formed_payload(
        &self,
        payload: ExecutionData,
    ) -> Result<SealedBlock<AlloyBlock<TransactionSigned>>, PayloadError> {
        let ExecutionData { payload, sidecar } = payload;
        let expected_hash = payload.block_hash();

        let sealed: SealedBlock<AlloyBlock<TransactionSigned>> =
            payload.try_into_block_with_sidecar(&sidecar)?.seal_slow();
        if expected_hash != sealed.hash() {
            return Err(PayloadError::BlockHash {
                execution: sealed.hash(),
                consensus: expected_hash,
            })
        }

        shanghai::ensure_well_formed_fields(
            sealed.body(),
            self.chain_spec.is_shanghai_active_at_timestamp(sealed.timestamp()),
        )?;
        cancun::ensure_well_formed_fields(
            &sealed,
            sidecar.cancun(),
            self.chain_spec.is_cancun_active_at_timestamp(sealed.timestamp()),
        )?;
        prague::ensure_well_formed_fields(
            sealed.body(),
            sidecar.prague(),
            self.chain_spec.is_prague_active_at_timestamp(sealed.timestamp()),
        )?;
        Ok(sealed)
    }
}

/// A payload that passed every pre-execution check, and what admitting it cost.
#[derive(Debug)]
pub struct AdmittedBlock {
    pub block: RecoveredBlock<AlloyBlock<TransactionSigned>>,
    pub timings: AdmissionTimings,
}

/// Why a payload was refused before it was ever executed.
///
/// Typed rather than flattened to a string because the rejection *class* is part of what a replay
/// has to reproduce: two validators that both reject a block for different reasons have not agreed
/// on it. [`Self::class`] is the stable name a recorded oracle compares.
#[derive(Debug)]
pub enum AdmissionError {
    /// The payload's layout, fork fields, or announced block hash were wrong.
    Payload(PayloadError),
    /// The block broke a consensus rule that does not require executing it.
    Consensus(ConsensusError),
    /// A transaction signature did not yield a sender.
    SenderRecovery,
    /// This validator holds no header for the block's parent, so parent-dependent rules could not
    /// run. Fail-closed: an unadmitted block is never an accepted one.
    NoAcceptedParent { block_number: u64 },
}

impl AdmissionError {
    /// Stable, comparable name for the rejection.
    ///
    /// Coarse on purpose. It names the stage that refused the block, which is what two independent
    /// validators must agree on; the specific rule is in the [`Display`](core::fmt::Display) text
    /// and is free to gain detail without breaking a recorded corpus.
    pub const fn class(&self) -> &'static str {
        match self {
            Self::Payload(_) => "payload",
            Self::Consensus(_) => "consensus",
            Self::SenderRecovery => "sender_recovery",
            Self::NoAcceptedParent { .. } => "no_accepted_parent",
        }
    }
}

impl core::fmt::Display for AdmissionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Payload(err) => write!(f, "payload is not well formed: {err}"),
            Self::Consensus(err) => write!(f, "pre-execution consensus validation failed: {err}"),
            Self::SenderRecovery => {
                write!(f, "a transaction signature did not recover to a sender")
            }
            Self::NoAcceptedParent { block_number } => write!(
                f,
                "no accepted parent header to validate block {block_number} against; the pair \
                 must reach a known head before it can admit a child"
            ),
        }
    }
}

impl core::error::Error for AdmissionError {}

impl From<PayloadError> for AdmissionError {
    fn from(err: PayloadError) -> Self {
        Self::Payload(err)
    }
}

impl From<ConsensusError> for AdmissionError {
    fn from(err: ConsensusError) -> Self {
        Self::Consensus(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_consensus::{constants::EMPTY_OMMER_ROOT_HASH, BlockBody, Header, EMPTY_ROOT_HASH};
    use alloy_primitives::{Address, B256, U256};
    use alloy_rpc_types_engine::{ExecutionPayload, ExecutionPayloadSidecar};
    use reth_chainspec::{ChainSpec, ChainSpecBuilder};
    use reth_ethereum_consensus::EthBeaconConsensus;
    use std::sync::Arc;

    /// Paris at genesis and nothing after it.
    ///
    /// The narrowest spec these tests can run under: post-merge header rules apply from block 0,
    /// while Shanghai and later stay inactive so the payload is a V1 with no withdrawals, blob, or
    /// requests fields to construct. The fork field rules themselves belong to
    /// `reth-payload-validator` and are tested there; what is under test here is the *sequence*.
    fn spec() -> Arc<ChainSpec> {
        Arc::new(ChainSpecBuilder::mainnet().paris_activated().build())
    }

    /// A header whose every field already matches what the payload round-trip reconstructs.
    ///
    /// An Engine payload does not carry the ommers hash, the difficulty, or the nonce, so
    /// rebuilding a block from one fills those in with post-merge constants. A fixture that
    /// left them at anything else would fail the block-hash check for a reason that has nothing
    /// to do with the rule under test.
    fn block_at(number: u64, parent_hash: B256, base_fee: u64) -> AlloyBlock<TransactionSigned> {
        AlloyBlock {
            header: Header {
                number,
                parent_hash,
                gas_limit: 30_000_000,
                base_fee_per_gas: Some(base_fee),
                ommers_hash: EMPTY_OMMER_ROOT_HASH,
                transactions_root: EMPTY_ROOT_HASH,
                difficulty: U256::ZERO,
                beneficiary: Address::ZERO,
                ..Default::default()
            },
            body: BlockBody::default(),
        }
    }

    fn payload_of(block: &AlloyBlock<TransactionSigned>) -> ExecutionData {
        ExecutionData {
            payload: ExecutionPayload::from_block_slow(block).0,
            sidecar: ExecutionPayloadSidecar::none(),
        }
    }

    /// An accepted parent and a child that legitimately follows it.
    ///
    /// The child's base fee is derived from the parent with the chain's own EIP-1559 rule rather
    /// than chosen, so the pair is one `validate_header_against_parent` accepts for the same reason
    /// mainnet would.
    fn parent_and_child(chain_spec: &ChainSpec) -> (SealedHeader, AlloyBlock<TransactionSigned>) {
        let parent = block_at(1, B256::repeat_byte(0x11), 1_000_000_000);
        let timestamp = parent.header.timestamp + 12;
        let base_fee = parent
            .header
            .next_block_base_fee(chain_spec.base_fee_params_at_timestamp(timestamp))
            .expect("London is active");
        let parent = SealedHeader::new_unhashed(parent.header);

        let mut child = block_at(2, parent.hash(), base_fee);
        child.header.timestamp = timestamp;
        child.header.gas_limit = parent.gas_limit;
        (parent, child)
    }

    #[test]
    fn an_honest_payload_is_admitted_and_reports_what_admitting_it_cost() {
        let chain_spec = spec();
        let consensus = EthBeaconConsensus::new(chain_spec.clone());
        let (parent, child) = parent_and_child(&chain_spec);

        let admitted = UntrustedAdmission::new(chain_spec.as_ref(), &consensus)
            .admit(payload_of(&child), Some(&parent))
            .expect("an honest payload is admitted");

        assert_eq!(admitted.block.header().number, 2);
        assert_eq!(admitted.timings.source, AdmissionSource::ExecutionData);
        // Every phase this entry point performs must report a number rather than absence, or a
        // record cannot tell "admitted for free" from "did not admit".
        assert!(admitted.timings.payload_validation_us.is_some());
        assert!(admitted.timings.pre_execution_consensus_us.is_some());
        assert!(admitted.timings.sender_recovery_us.is_some());
        // Not this one: the payload arrived already parsed. The frame reader sets it.
        assert_eq!(admitted.timings.input_decode_us, None);
    }

    #[test]
    fn a_payload_whose_announced_hash_disagrees_with_its_contents_is_refused() {
        // The check every later one depends on. Without it a producer could announce one identity
        // and supply another block's fields, and each subsequent rule would validate the fields
        // while the caller went on believing the announced hash.
        let chain_spec = spec();
        let consensus = EthBeaconConsensus::new(chain_spec.clone());
        let (parent, child) = parent_and_child(&chain_spec);
        let mut payload = payload_of(&child);
        payload.payload.as_v1_mut().block_hash = B256::repeat_byte(0xde);

        let err = UntrustedAdmission::new(chain_spec.as_ref(), &consensus)
            .admit(payload, Some(&parent))
            .expect_err("a mismatched block hash is not admissible");

        assert_eq!(err.class(), "payload");
        assert!(matches!(err, AdmissionError::Payload(PayloadError::BlockHash { .. })));
    }

    #[test]
    fn a_block_that_does_not_follow_the_accepted_parent_is_refused() {
        let chain_spec = spec();
        let consensus = EthBeaconConsensus::new(chain_spec.clone());
        let (parent, mut child) = parent_and_child(&chain_spec);
        // Correct in isolation, wrong against this parent: nothing about the block itself is
        // malformed, and only the parent-dependent rules can see the break.
        child.header.parent_hash = B256::repeat_byte(0xaa);

        let err = UntrustedAdmission::new(chain_spec.as_ref(), &consensus)
            .admit(payload_of(&child), Some(&parent))
            .expect_err("a block that does not descend from the accepted head is not admissible");

        assert_eq!(err.class(), "consensus");
    }

    #[test]
    fn a_parent_the_validator_cannot_name_refuses_the_block_rather_than_skipping_the_check() {
        // The failure mode this guards is silent: with no parent, every parent-dependent rule
        // simply would not run, and the block would be admitted on a strictly weaker check than
        // the one the caller believes was applied.
        let chain_spec = spec();
        let consensus = EthBeaconConsensus::new(chain_spec.clone());
        let (_parent, child) = parent_and_child(&chain_spec);

        let err = UntrustedAdmission::new(chain_spec.as_ref(), &consensus)
            .admit(payload_of(&child), None)
            .expect_err("a pair that cannot name its parent must not admit a child");

        assert_eq!(err.class(), "no_accepted_parent");
        assert!(matches!(err, AdmissionError::NoAcceptedParent { block_number: 2 }));
    }

    #[test]
    fn the_parent_is_taken_from_the_validator_and_never_from_the_payload() {
        // A producer that could name its own parent would choose the timestamp, gas limit, and
        // base fee its block is measured against. The payload carries only a parent *hash*, and
        // this shows the header behind that hash is the validator's own: the same payload against
        // a parent that permits it and one that does not gives opposite answers.
        let chain_spec = spec();
        let consensus = EthBeaconConsensus::new(chain_spec.clone());
        let (honest, child) = parent_and_child(&chain_spec);
        let admission = UntrustedAdmission::new(chain_spec.as_ref(), &consensus);

        assert!(admission.admit(payload_of(&child), Some(&honest)).is_ok());

        // Same payload, same parent hash, but a parent dated after the block that claims to follow
        // it. Only the validator's own record can tell the two parents apart.
        let mut future = honest.clone_header();
        future.timestamp = child.header.timestamp + 1;
        let future = SealedHeader::new(future, honest.hash());

        let err = admission
            .admit(payload_of(&child), Some(&future))
            .expect_err("the validator's own parent decides, so a later parent must refuse");
        assert_eq!(err.class(), "consensus");
    }

    /// A legacy transaction carrying a signature that recovers to nothing.
    ///
    /// `s = 0` is outside the valid range, so checked recovery refuses it. Built by hand rather
    /// than signed-then-corrupted because what is under test is that admission *refuses* rather
    /// than resolving the signature to whatever address it happens to produce.
    fn unrecoverable_transaction() -> TransactionSigned {
        use alloy_consensus::{SignableTransaction, TxLegacy};
        use alloy_primitives::{Signature, TxKind};

        let tx = TxLegacy {
            chain_id: Some(1),
            nonce: 0,
            gas_price: 1_000_000_000,
            gas_limit: 21_000,
            to: TxKind::Call(Address::repeat_byte(0x99)),
            value: U256::ZERO,
            input: Default::default(),
        };
        tx.into_signed(Signature::new(U256::from(1), U256::ZERO, false)).into()
    }

    #[test]
    fn a_transaction_whose_signature_recovers_to_nothing_is_refused_as_input() {
        // An unrecoverable sender is an input-validation failure, not an execution failure: the
        // block never becomes something this validator would run, so nothing downstream sees it.
        // The ExEx never exercises this — its Engine recovered every sender before the block
        // arrived — which is exactly why the standalone path needs its own test.
        let chain_spec = spec();
        let consensus = EthBeaconConsensus::new(chain_spec.clone());
        let (parent, mut child) = parent_and_child(&chain_spec);
        child.body.transactions.push(unrecoverable_transaction());
        child.header.transactions_root =
            alloy_consensus::proofs::calculate_transaction_root(&child.body.transactions);
        child.header.gas_used = 21_000;

        let err = UntrustedAdmission::new(chain_spec.as_ref(), &consensus)
            .admit(payload_of(&child), Some(&parent))
            .expect_err("a signature that recovers to no sender is not admissible");

        assert_eq!(err.class(), "sender_recovery");
    }

    #[test]
    fn a_transactions_root_that_does_not_commit_to_the_body_is_refused_by_the_block_hash() {
        // Refused as `payload`, not as `consensus`, and the reason is a property of this boundary
        // worth writing down: an Engine payload does not carry a transactions root. It carries the
        // transactions, and rebuilding the block derives the root from them, so a producer cannot
        // put a lying root in the payload at all — it can only announce a block hash taken over
        // one, and the hash check is what catches that.
        //
        // `validate_block_pre_execution` still checks the root a few lines later. That check is
        // redundant here and is not redundant everywhere: the same consensus object validates
        // blocks that arrived over the wire with their header intact. Delegating rather than
        // hand-picking rules is what keeps both callers correct.
        let chain_spec = spec();
        let consensus = EthBeaconConsensus::new(chain_spec.clone());
        let (parent, mut child) = parent_and_child(&chain_spec);
        child.header.transactions_root = B256::repeat_byte(0x77);

        let err = UntrustedAdmission::new(chain_spec.as_ref(), &consensus)
            .admit(payload_of(&child), Some(&parent))
            .expect_err("a transactions root that commits to nothing is not admissible");

        assert_eq!(err.class(), "payload");
        assert!(matches!(err, AdmissionError::Payload(PayloadError::BlockHash { .. })));
    }

    #[test]
    fn a_base_fee_the_parent_does_not_imply_is_refused() {
        // The EIP-1559 base fee is a pure function of the parent, so this is a rule only the
        // validator's own accepted head can enforce, and one a producer would gain from bending.
        let chain_spec = spec();
        let consensus = EthBeaconConsensus::new(chain_spec.clone());
        let (parent, mut child) = parent_and_child(&chain_spec);
        child.header.base_fee_per_gas =
            Some(child.header.base_fee_per_gas.expect("London is active") + 1);

        let err = UntrustedAdmission::new(chain_spec.as_ref(), &consensus)
            .admit(payload_of(&child), Some(&parent))
            .expect_err("a base fee the parent does not imply is not admissible");

        assert_eq!(err.class(), "consensus");
    }

    #[test]
    fn a_gas_limit_outside_the_ramp_the_parent_allows_is_refused() {
        let chain_spec = spec();
        let consensus = EthBeaconConsensus::new(chain_spec.clone());
        let (parent, mut child) = parent_and_child(&chain_spec);
        child.header.gas_limit = parent.gas_limit * 2;

        let err = UntrustedAdmission::new(chain_spec.as_ref(), &consensus)
            .admit(payload_of(&child), Some(&parent))
            .expect_err("a gas limit outside the ramp bound is not admissible");

        assert_eq!(err.class(), "consensus");
    }

    #[test]
    fn withdrawals_are_gated_on_the_chain_spec_and_not_on_the_payload() {
        // What this pins is the wiring, not the rule: `ensure_well_formed_payload` must pass the
        // *chain spec's* fork activation to `reth-payload-validator` rather than infer activation
        // from whether the payload happens to carry the field. Getting that backwards would make
        // every fork gate self-fulfilling — a producer that omits withdrawals would be treated as
        // pre-Shanghai. Shanghai is inactive under `spec()`, so a payload that carries
        // withdrawals anyway must be refused.
        let chain_spec = spec();
        let consensus = EthBeaconConsensus::new(chain_spec.clone());
        let (parent, mut child) = parent_and_child(&chain_spec);
        child.body.withdrawals = Some(Default::default());
        child.header.withdrawals_root = Some(EMPTY_ROOT_HASH);

        let err = UntrustedAdmission::new(chain_spec.as_ref(), &consensus)
            .admit(payload_of(&child), Some(&parent))
            .expect_err("withdrawals before Shanghai are not admissible");

        assert_eq!(err.class(), "payload");
        assert!(matches!(
            err,
            AdmissionError::Payload(PayloadError::PreShanghaiBlockWithWithdrawals)
        ));
    }

    #[test]
    fn a_shanghai_chain_refuses_a_payload_that_omits_withdrawals() {
        // The same wiring from the other side, so neither direction can be satisfied by ignoring
        // the chain spec.
        let chain_spec = Arc::new(ChainSpecBuilder::mainnet().shanghai_activated().build());
        let consensus = EthBeaconConsensus::new(chain_spec.clone());
        let (parent, child) = parent_and_child(&chain_spec);

        let err = UntrustedAdmission::new(chain_spec.as_ref(), &consensus)
            .admit(payload_of(&child), Some(&parent))
            .expect_err("a post-Shanghai payload without withdrawals is not admissible");

        assert_eq!(err.class(), "payload");
        assert!(matches!(
            err,
            AdmissionError::Payload(PayloadError::PostShanghaiBlockWithoutWithdrawals)
        ));
    }

    #[test]
    fn an_admitted_block_is_the_type_the_execution_core_takes() {
        // Compile-time: `admit` must hand the core exactly what the recovered entry point wants,
        // or admission and execution do not join, and the standalone path needs a conversion the
        // ExEx path does not have.
        fn takes_core_block(
            _: &RecoveredBlock<
                reth_primitives_traits::BlockTy<reth_ethereum_primitives::EthPrimitives>,
            >,
        ) {
        }
        let chain_spec = spec();
        let consensus = EthBeaconConsensus::new(chain_spec.clone());
        let (parent, child) = parent_and_child(&chain_spec);

        let admitted = UntrustedAdmission::new(chain_spec.as_ref(), &consensus)
            .admit(payload_of(&child), Some(&parent))
            .expect("an honest payload is admitted");

        takes_core_block(&admitted.block);
    }
}
