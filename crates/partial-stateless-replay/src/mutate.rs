//! Deriving invalid payloads from valid recorded ones.
//!
//! A mainnet recording contains no invalid block, so a replay over one proves the accept path and
//! nothing else. That is a real gap: a validator that accepted everything would pass such a replay
//! perfectly. The negatives have to be manufactured, and manufacturing them from *recorded* frames
//! rather than from synthetic fixtures is what makes them evidence about the corpus — the same
//! bytes, the same decode, the same admission entry point, one field different.
//!
//! Every mutation here re-announces the block hash it produces. That is not a shortcut around the
//! block-hash check; it is what makes the other checks reachable at all. A payload whose contents
//! changed and whose announced hash did not is refused as `payload` no matter what else is wrong
//! with it, so a mutation meant to exercise a consensus rule must first be *well formed*. Only
//! [`Mutation::AnnouncedBlockHash`] leaves the two disagreeing, and that is the point of it.

use alloy_consensus::{proofs, SignableTransaction, TxLegacy};
use alloy_primitives::{Address, Signature, TxKind, B256, U256};
use alloy_rpc_types_engine::ExecutionData;
use reth_ethereum_primitives::{Block, TransactionSigned};
use reth_primitives_traits::SealedBlock;

/// One way to make a recorded payload invalid, and the class it must be refused under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mutation {
    /// The announced block hash no longer commits to the contents.
    ///
    /// The one mutation that leaves the payload ill-formed on purpose. Refused before any
    /// consensus rule runs, which is why it is also the mutation that proves the *order* of the
    /// checks rather than any single one.
    AnnouncedBlockHash,
    /// A gas limit outside the ramp the parent allows.
    ///
    /// Well formed and consensus-invalid: the payload carries `gas_limit`, so the mutation
    /// survives the round trip through the payload and the re-announced hash commits to it.
    GasLimit,
    /// A transaction whose signature recovers to no sender.
    ///
    /// Appended rather than corrupted in place, because appending is deterministic — a flipped bit
    /// inside an existing signature usually recovers to a *different* address rather than to none,
    /// which would exercise nothing. The transactions root is recomputed so the block is otherwise
    /// well formed, and recovery is then the only thing left to fail.
    UnrecoverableSender,
}

impl Mutation {
    /// Every mutation a recorded payload can carry.
    pub const ALL: [Self; 3] =
        [Self::AnnouncedBlockHash, Self::GasLimit, Self::UnrecoverableSender];

    /// The rejection class this mutation must produce.
    pub const fn expected_class(&self) -> &'static str {
        match self {
            Self::AnnouncedBlockHash => "payload",
            Self::GasLimit => "consensus",
            Self::UnrecoverableSender => "sender_recovery",
        }
    }

    /// Stable name for the run log.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::AnnouncedBlockHash => "announced_block_hash",
            Self::GasLimit => "gas_limit",
            Self::UnrecoverableSender => "unrecoverable_sender",
        }
    }

    /// Derives the invalid payload, or explains why this recording cannot carry it.
    pub fn apply(&self, payload: &ExecutionData) -> eyre::Result<ExecutionData> {
        let mut block: Block = payload
            .clone()
            .try_into_block::<TransactionSigned>()
            .map_err(|err| eyre::eyre!("recorded payload does not decode into a block: {err}"))?;

        match self {
            Self::AnnouncedBlockHash => {
                // The contents are untouched and the announcement is not, which is the only shape
                // of this failure a producer can actually cause.
                let honest = SealedBlock::seal_slow(block);
                let mut wrong = honest.hash().0;
                wrong[0] ^= 0xff;
                return Ok(ExecutionData::from_block_unchecked(
                    B256::from(wrong),
                    &honest.into_block(),
                ))
            }
            Self::GasLimit => {
                // Far outside any ramp the parent could permit, in the direction that also leaves
                // `gas_used` above the limit — so this is refused whether the header rule or the
                // against-parent rule reaches it first, and both are `consensus`.
                block.header.gas_limit = 1;
            }
            Self::UnrecoverableSender => {
                block.body.transactions.push(unrecoverable_transaction());
                block.header.transactions_root =
                    proofs::calculate_transaction_root(&block.body.transactions);
            }
        }

        let sealed = SealedBlock::seal_slow(block);
        let hash = sealed.hash();
        Ok(ExecutionData::from_block_unchecked(hash, &sealed.into_block()))
    }
}

/// A signature with `s = 0`, which is syntactically decodable and recovers to nothing.
fn unrecoverable_transaction() -> TransactionSigned {
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

/// Convenience: whether a block can carry every mutation.
///
/// All three apply to any block, including an empty one — the sender mutation appends rather than
/// corrupting, so it needs no existing transaction.
pub const fn applies_to_every_block() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_consensus::{BlockBody, BlockHeader, Header};
    use alloy_primitives::B256;

    fn payload_of(block: &Block) -> ExecutionData {
        let sealed = SealedBlock::seal_slow(block.clone());
        let hash = sealed.hash();
        ExecutionData::from_block_unchecked(hash, &sealed.into_block())
    }

    fn block() -> Block {
        Block {
            header: Header {
                number: 25_737_235,
                gas_limit: 30_000_000,
                gas_used: 21_000,
                base_fee_per_gas: Some(1_000_000_000),
                difficulty: U256::ZERO,
                ommers_hash: alloy_consensus::constants::EMPTY_OMMER_ROOT_HASH,
                transactions_root: alloy_consensus::constants::EMPTY_ROOT_HASH,
                parent_hash: B256::repeat_byte(0x11),
                ..Default::default()
            },
            body: BlockBody::default(),
        }
    }

    /// The whole point of re-announcing: a mutation that left the hash stale would be refused as
    /// `payload` before the rule it means to exercise ever ran.
    #[test]
    fn every_mutation_but_the_hash_one_produces_a_well_formed_payload() {
        let payload = payload_of(&block());

        for mutation in [Mutation::GasLimit, Mutation::UnrecoverableSender] {
            let mutated = mutation.apply(&payload).expect("mutation applies");
            let block = mutated
                .clone()
                .try_into_block::<TransactionSigned>()
                .expect("mutated payload decodes");
            assert_eq!(
                SealedBlock::seal_slow(block).hash(),
                mutated.payload.block_hash(),
                "{} must announce the hash of what it produced",
                mutation.as_str()
            );
        }
    }

    #[test]
    fn the_hash_mutation_leaves_the_announcement_disagreeing_with_the_contents() {
        let payload = payload_of(&block());
        let mutated = Mutation::AnnouncedBlockHash.apply(&payload).expect("mutation applies");
        let block = mutated.clone().try_into_block::<TransactionSigned>().expect("still decodes");
        assert_ne!(SealedBlock::seal_slow(block).hash(), mutated.payload.block_hash());
    }

    #[test]
    fn the_gas_limit_mutation_changes_the_field_it_names_and_nothing_else() {
        let payload = payload_of(&block());
        let mutated = Mutation::GasLimit.apply(&payload).expect("mutation applies");
        let mutated_block =
            mutated.try_into_block::<TransactionSigned>().expect("mutated payload decodes");
        assert_eq!(mutated_block.header.gas_limit(), 1);
        assert_eq!(mutated_block.header.number(), block().header.number);
        assert_eq!(mutated_block.body.transactions.len(), 0);
    }

    /// Appending rather than corrupting is what makes this deterministic, and it is also what lets
    /// it apply to an empty block.
    #[test]
    fn the_sender_mutation_applies_to_a_block_with_no_transactions() {
        let payload = payload_of(&block());
        let mutated = Mutation::UnrecoverableSender.apply(&payload).expect("mutation applies");
        let mutated_block =
            mutated.try_into_block::<TransactionSigned>().expect("mutated payload decodes");
        assert_eq!(mutated_block.body.transactions.len(), 1);
        assert_eq!(
            mutated_block.header.transactions_root,
            proofs::calculate_transaction_root(&mutated_block.body.transactions)
        );
        assert!(applies_to_every_block());
    }

    /// The classes are what a recorded oracle compares, so they are pinned rather than derived.
    #[test]
    fn each_mutation_names_the_class_it_must_produce() {
        assert_eq!(Mutation::AnnouncedBlockHash.expected_class(), "payload");
        assert_eq!(Mutation::GasLimit.expected_class(), "consensus");
        assert_eq!(Mutation::UnrecoverableSender.expected_class(), "sender_recovery");
        assert_eq!(Mutation::ALL.len(), 3);
    }
}
