//! The negatives a mainnet corpus cannot supply, checked against the real admission boundary.
//!
//! `mutate`'s own unit tests check that each mutation produces the *shape* it claims — a
//! well-formed payload with one field changed. They cannot check the thing that matters, which is
//! that `UntrustedAdmission` refuses each one under the class the driver expects. That needs the
//! admission boundary itself, and it is what makes the driver's mutation pass evidence rather than
//! an assertion about its own bookkeeping.

use alloy_consensus::{
    constants::{EMPTY_OMMER_ROOT_HASH, EMPTY_ROOT_HASH},
    BlockBody, Header,
};
use alloy_primitives::{Address, B256, U256};
use alloy_rpc_types_engine::ExecutionData;
use partial_stateless_replay::Mutation;
use partial_stateless_validator::{AdmissionError, UntrustedAdmission};
use reth_chainspec::{ChainSpec, ChainSpecBuilder};
use reth_ethereum_consensus::EthBeaconConsensus;
use reth_ethereum_primitives::Block;
use reth_primitives_traits::{SealedBlock, SealedHeader};
use std::sync::Arc;

fn spec() -> Arc<ChainSpec> {
    Arc::new(ChainSpecBuilder::mainnet().paris_activated().build())
}

/// A header whose every field already matches what the payload round trip reconstructs, so a
/// failure here is the rule under test and never the fixture.
fn block_at(number: u64, parent_hash: B256, base_fee: u64) -> Block {
    Block {
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

/// A parent and a child that legitimately follows it, with the base fee derived from the chain's
/// own EIP-1559 rule rather than chosen.
fn parent_and_child(chain_spec: &ChainSpec) -> (SealedHeader, Block) {
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

fn payload_of(block: &Block) -> ExecutionData {
    let sealed = SealedBlock::seal_slow(block.clone());
    let hash = sealed.hash();
    ExecutionData::from_block_unchecked(hash, &sealed.into_block())
}

/// The honest payload must be admitted, or every rejection below would be evidence about the
/// fixture rather than about the mutation.
#[test]
fn the_unmutated_payload_is_admitted() {
    let chain_spec = spec();
    let consensus = EthBeaconConsensus::new(chain_spec.clone());
    let (parent, child) = parent_and_child(&chain_spec);

    UntrustedAdmission::new(chain_spec.as_ref(), &consensus)
        .admit(payload_of(&child), Some(&parent))
        .expect("the fixture is a block this validator accepts");
}

/// Each derived negative is refused, and refused under the class the driver compares. A mutation
/// that produced the *right verdict for the wrong reason* would pass a coverage count and prove
/// nothing, which is why the class is asserted rather than the failure.
#[test]
fn every_mutation_is_refused_under_the_class_it_claims() {
    let chain_spec = spec();
    let consensus = EthBeaconConsensus::new(chain_spec.clone());
    let (parent, child) = parent_and_child(&chain_spec);
    let honest = payload_of(&child);
    let admission = UntrustedAdmission::new(chain_spec.as_ref(), &consensus);

    for mutation in Mutation::ALL {
        let mutated = mutation.apply(&honest).expect("mutation derives from a recorded payload");
        let error = admission
            .admit(mutated, Some(&parent))
            .expect_err(&format!("{} must be refused", mutation.as_str()));
        assert_eq!(
            error.class(),
            mutation.expected_class(),
            "{} was refused as {} ({error})",
            mutation.as_str(),
            error.class()
        );
    }
}

/// The negative no frame can carry: a pair that cannot name a parent refuses the block rather than
/// running the subset of rules that survive without one.
#[test]
fn a_pair_with_no_accepted_parent_refuses_an_otherwise_valid_block() {
    let chain_spec = spec();
    let consensus = EthBeaconConsensus::new(chain_spec.clone());
    let (_, child) = parent_and_child(&chain_spec);

    let error = UntrustedAdmission::new(chain_spec.as_ref(), &consensus)
        .admit(payload_of(&child), None)
        .expect_err("a validator with no parent has nothing to check the block against");

    assert!(matches!(error, AdmissionError::NoAcceptedParent { block_number: 2 }), "{error:?}");
    assert_eq!(error.class(), "no_accepted_parent");
}
