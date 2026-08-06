//! Admission of a transaction whose sender is a cache miss.
//!
//! [`SenderAccountProof::verify`] is the cryptographic core: it binds the proof to the recovered
//! sender, checks a canonical anchor inside the account window, verifies the MPT proof, and
//! returns the proven nonce and balance. Three things sit between that and an admission decision,
//! and this module is all three:
//!
//! 1. The **cold precondition** that [`SenderAccountProof::verify`] assigns to its caller and
//!    nobody performed. A warm sender must be validated from the cache, never from an attached
//!    proof: the "cold implies unchanged within the window" invariant that makes an in-window proof
//!    exact only holds for a *confirmed-cold* sender.
//! 2. The **admission rules** — balance against maximum cost, nonce, EIP-3607 eligibility — which
//!    nothing consumed [`VerifiedSender`] to apply.
//! 3. The **canonicality lookup**, which this module takes as a closure because it needs a header
//!    source. That is deliberate: this crate does not depend on `reth-provider`, so nothing here
//!    *can* read state, which is what makes the no-state-access property structural rather than a
//!    claim about intent.
//!
//! What is not here is the transaction pool. Reth's pooled transaction type carries no proof
//! field, so a real pool integration forces a custom transaction type; that is deferred wire
//! distribution, along with proof relay, DoS accounting, and pending-head/reorg behaviour.

use crate::{
    network_cache::NetworkStateCache,
    sender_proof::{SenderAccountProof, SenderAdmissionInput, SenderProofError, VerifiedSender},
};
use alloy_primitives::{Address, U256};

/// Admits a transaction from a sender the cache does not hold, using an attached account proof.
///
/// The caller supplies `is_canonical(number, hash) -> Option<state_root>`, which must perform a
/// header lookup and a canonicality check and nothing else. Header-only access is what makes the
/// no-state-access property true; a lookup that reaches for account state would defeat the point
/// of admitting the sender from a proof at all.
pub fn admit_cold_sender(
    cache: &NetworkStateCache,
    proof: &SenderAccountProof,
    request: &ColdAdmissionRequest,
    is_canonical: impl Fn(u64, alloy_primitives::B256) -> Option<alloy_primitives::B256>,
) -> Result<ColdSenderAdmission, ColdAdmissionError> {
    // The cache must be the one at the head the request names. A cache at some other height
    // cannot answer "is this sender cold at the head", and an absent entry there would be
    // meaningless rather than reassuring.
    if cache.current_block() != request.input.head_block_number {
        return Err(ColdAdmissionError::CacheHeadMismatch {
            cache_block: cache.current_block(),
            head_block: request.input.head_block_number,
        })
    }
    if cache.contains_account(&request.input.tx_sender) {
        return Err(ColdAdmissionError::SenderIsWarm { address: request.input.tx_sender })
    }

    let sender = proof.verify(&request.input, is_canonical).map_err(ColdAdmissionError::Proof)?;

    // EIP-3607 eligibility needs no separate check: `verify` rejects a sender holding code that is
    // not a revealed EIP-7702 delegation designator, so reaching here means the sender is either a
    // plain EOA or a delegated one, and both may send.
    if sender.balance < request.max_cost {
        return Err(ColdAdmissionError::InsufficientFunds {
            address: sender.address,
            balance: sender.balance,
            required: request.max_cost,
        })
    }

    // `verify` already rejected a nonce below the proven one. A nonce above it is a legitimate
    // future transaction rather than a rejection, so it is reported instead of refused: the proven
    // nonce is a lower bound, and only the pool knows whether the gap is filled.
    let executable = request.input.tx_nonce == sender.nonce;
    Ok(ColdSenderAdmission { sender, executable })
}

/// Everything the admission decision needs beyond the proof itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColdAdmissionRequest {
    /// Sender, nonce, head height, and the account window the proof's freshness is bounded by.
    pub input: SenderAdmissionInput,
    /// Maximum the transaction can cost its sender: `gas_limit * max_fee_per_gas + value`, plus
    /// any blob fee. Supplied by the caller because computing it needs the transaction type.
    pub max_cost: U256,
}

/// A sender admitted from a proof rather than from state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColdSenderAdmission {
    /// Address, proven nonce and balance, and whether the sender is a delegated EOA.
    pub sender: VerifiedSender,
    /// Whether the transaction can execute now (`tx_nonce == proven nonce`) or must wait behind a
    /// nonce gap.
    pub executable: bool,
}

/// Why a cold sender was not admitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColdAdmissionError {
    /// The cache is not at the head the request names, so it cannot say whether the sender is
    /// cold.
    CacheHeadMismatch {
        /// Height the cache is at.
        cache_block: u64,
        /// Height the request named.
        head_block: u64,
    },
    /// The sender is in the account cache and must be admitted from it, not from a proof.
    SenderIsWarm {
        /// The warm sender.
        address: Address,
    },
    /// The attached proof did not verify.
    Proof(SenderProofError),
    /// The proven balance cannot cover the transaction's maximum cost.
    InsufficientFunds {
        /// The sender.
        address: Address,
        /// Balance proven at the anchor.
        balance: U256,
        /// Maximum the transaction can cost.
        required: U256,
    },
}

impl std::fmt::Display for ColdAdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CacheHeadMismatch { cache_block, head_block } => write!(
                f,
                "cold admission needs the cache at the head: cache is at {cache_block}, head is \
                 {head_block}"
            ),
            Self::SenderIsWarm { address } => {
                write!(f, "sender {address} is warm; admit it from the cache, not from a proof")
            }
            Self::Proof(err) => write!(f, "sender proof rejected: {err}"),
            Self::InsufficientFunds { address, balance, required } => write!(
                f,
                "sender {address} cannot cover the transaction: balance={balance}, \
                 required={required}"
            ),
        }
    }
}

impl std::error::Error for ColdAdmissionError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{AccountData, LastNBlocksPolicy};
    use alloy_primitives::{keccak256, B256};
    use reth_primitives_traits::Account;
    use reth_trie_common::{
        proof::ProofRetainer, AccountProof, HashBuilder, Nibbles, EMPTY_ROOT_HASH,
    };

    const WINDOW: u64 = 60;
    const HEAD: u64 = 1_000;
    const SENDER: Address = Address::repeat_byte(0x42);

    fn cache_at(block: u64) -> NetworkStateCache {
        NetworkStateCache::restore(
            Default::default(),
            Default::default(),
            Default::default(),
            block,
            Box::new(LastNBlocksPolicy::new(WINDOW)),
            Box::new(LastNBlocksPolicy::new(WINDOW)),
        )
    }

    /// Builds a single-account trie and the inclusion proof for `SENDER` inside it.
    fn funded_sender_proof(nonce: u64, balance: U256) -> (SenderAccountProof, B256) {
        let account = Account { nonce, balance, bytecode_hash: None };
        let nibbles = Nibbles::unpack(keccak256(SENDER));
        let value = alloy_rlp::encode(account.into_trie_account(EMPTY_ROOT_HASH));

        let mut builder =
            HashBuilder::default().with_proof_retainer(ProofRetainer::new(vec![nibbles]));
        builder.add_leaf(nibbles, &value);
        let state_root = builder.root();
        let proof_nodes =
            builder.take_proof_nodes().into_nodes_sorted().into_iter().map(|(_, b)| b).collect();

        let proof = AccountProof {
            address: SENDER,
            info: Some(account),
            proof: proof_nodes,
            storage_root: EMPTY_ROOT_HASH,
            storage_proofs: Vec::new(),
        };
        (
            SenderAccountProof::new(proof, HEAD - 1, B256::repeat_byte(0xaa), state_root, None),
            state_root,
        )
    }

    fn request(nonce: u64, max_cost: U256) -> ColdAdmissionRequest {
        ColdAdmissionRequest {
            input: SenderAdmissionInput {
                tx_sender: SENDER,
                tx_nonce: nonce,
                head_block_number: HEAD,
                account_window: WINDOW,
            },
            max_cost,
        }
    }

    #[test]
    fn a_funded_cold_sender_is_admitted_from_its_proof() {
        let cache = cache_at(HEAD);
        let (proof, state_root) = funded_sender_proof(7, U256::from(1_000_000u64));

        let admitted =
            admit_cold_sender(&cache, &proof, &request(7, U256::from(999u64)), |_, _| {
                Some(state_root)
            })
            .expect("a funded plain EOA with a canonical in-window proof is admissible");

        assert_eq!(admitted.sender.address, SENDER);
        assert_eq!(admitted.sender.nonce, 7);
        assert_eq!(admitted.sender.balance, U256::from(1_000_000u64));
        assert!(!admitted.sender.is_delegated);
        assert!(admitted.executable, "the tx nonce equals the proven nonce");
    }

    #[test]
    fn a_non_canonical_anchor_is_rejected_before_any_admission_decision() {
        let cache = cache_at(HEAD);
        let (proof, _) = funded_sender_proof(7, U256::from(1_000_000u64));

        let error = admit_cold_sender(&cache, &proof, &request(7, U256::ZERO), |_, _| None)
            .expect_err("an anchor that is not on the canonical chain cannot admit anything");

        assert!(matches!(
            error,
            ColdAdmissionError::Proof(SenderProofError::AnchorNotCanonical { .. })
        ));
    }

    #[test]
    fn a_forged_state_root_is_rejected() {
        let cache = cache_at(HEAD);
        let (proof, _) = funded_sender_proof(7, U256::from(1_000_000u64));

        // The anchor is canonical, but its canonical state root is not the one the proof claims.
        let error = admit_cold_sender(&cache, &proof, &request(7, U256::ZERO), |_, _| {
            Some(B256::repeat_byte(0xee))
        })
        .expect_err("a proof anchored to a forged root must not be admitted");

        assert!(matches!(
            error,
            ColdAdmissionError::Proof(SenderProofError::AnchorRootMismatch { .. })
        ));
    }

    #[test]
    fn a_warm_sender_must_come_from_the_cache_not_from_a_proof() {
        let mut cache = cache_at(HEAD);
        let mut accessed = crate::accessed_state::BlockAccessedState::default();
        accessed
            .accounts
            .insert(SENDER, AccountData { nonce: 7, balance: U256::ZERO, code_hash: None });
        cache.on_block_executed(HEAD, &accessed);
        let (proof, state_root) = funded_sender_proof(7, U256::from(1_000_000u64));

        let error =
            admit_cold_sender(&cache, &proof, &request(7, U256::ZERO), |_, _| Some(state_root))
                .expect_err("the cold precondition must reject a sender the cache already holds");

        assert_eq!(error, ColdAdmissionError::SenderIsWarm { address: SENDER });
    }

    #[test]
    fn a_cache_behind_the_head_cannot_answer_the_cold_question() {
        let cache = cache_at(HEAD - 1);
        let (proof, state_root) = funded_sender_proof(7, U256::from(1_000_000u64));

        let error =
            admit_cold_sender(&cache, &proof, &request(7, U256::ZERO), |_, _| Some(state_root))
                .expect_err("absence from a stale cache says nothing about the head");

        assert_eq!(
            error,
            ColdAdmissionError::CacheHeadMismatch { cache_block: HEAD - 1, head_block: HEAD }
        );
    }

    #[test]
    fn a_sender_that_cannot_cover_its_maximum_cost_is_rejected() {
        let cache = cache_at(HEAD);
        let (proof, state_root) = funded_sender_proof(7, U256::from(100u64));

        let error = admit_cold_sender(&cache, &proof, &request(7, U256::from(101u64)), |_, _| {
            Some(state_root)
        })
        .expect_err("balance below the maximum cost is not admissible");

        assert!(matches!(error, ColdAdmissionError::InsufficientFunds { .. }));
    }

    #[test]
    fn a_future_nonce_is_admitted_but_not_executable() {
        let cache = cache_at(HEAD);
        let (proof, state_root) = funded_sender_proof(7, U256::from(1_000_000u64));

        let admitted =
            admit_cold_sender(&cache, &proof, &request(9, U256::ZERO), |_, _| Some(state_root))
                .expect("a nonce above the proven one is a queued transaction, not a rejection");

        assert!(!admitted.executable);
    }

    #[test]
    fn a_nonce_below_the_proven_one_is_rejected() {
        let cache = cache_at(HEAD);
        let (proof, state_root) = funded_sender_proof(7, U256::from(1_000_000u64));

        let error =
            admit_cold_sender(&cache, &proof, &request(6, U256::ZERO), |_, _| Some(state_root))
                .expect_err("the proven nonce is a lower bound on the true one");

        assert!(matches!(
            error,
            ColdAdmissionError::Proof(SenderProofError::NonceTooLow { tx: 6, state: 7 })
        ));
    }
}
