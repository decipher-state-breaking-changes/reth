//! The Cold EOA core-scenario gate: a funded plain-EOA cache miss is admitted through the proof
//! adapter without state-provider account access, and a bad canonical/root anchor is rejected
//! before anything downstream sees a decision.
//!
//! The no-state-access property is proved structurally rather than by assertion. The chain source
//! below implements `BlockHashReader` and `HeaderProvider` and *nothing else* — no `AccountReader`,
//! no `StateProvider`, no `StateProofProvider` — so if the admission path ever needed to read an
//! account, a storage slot, or a bytecode, this file would not compile. A provider that panics on
//! state reads would be the weaker version of the same claim: it would only catch reads the test
//! happened to trigger.

use alloy_consensus::Header;
use alloy_primitives::{keccak256, Address, BlockNumber, B256, U256};
use partial_stateless::{
    accessed_state::BlockAccessedState,
    admit_cold_sender,
    network_cache::NetworkStateCache,
    policy::{AccountData, LastNBlocksPolicy},
    sender_proof::{SenderAccountProof, SenderAdmissionInput, SenderProofError},
    ColdAdmissionError, ColdAdmissionRequest,
};
use partial_stateless_exex::cold_eoa::canonical_state_root_lookup;
use reth_primitives_traits::{Account, SealedHeader};
use reth_provider::{BlockHashReader, HeaderProvider, ProviderResult};
use reth_trie_common::{proof::ProofRetainer, AccountProof, HashBuilder, Nibbles, EMPTY_ROOT_HASH};
use std::ops::RangeBounds;

const ACCOUNT_WINDOW: u64 = 60;
const STORAGE_WINDOW: u64 = 30;
const HEAD: u64 = 21_000_000;
const SENDER: Address = Address::repeat_byte(0x7e);

#[test]
fn a_funded_cold_eoa_is_admitted_without_reading_state() {
    let (proof, state_root) = funded_sender_proof(12, U256::from(5_000_000_000_000_000_000u128));
    let chain = HeaderOnlyChain::canonical_at(HEAD - 5, state_root);
    let cache = cold_cache_at(HEAD);

    let admitted = admit_cold_sender(
        &cache,
        &proof,
        &request(12, U256::from(1_000_000_000_000_000_000u128)),
        canonical_state_root_lookup(&chain),
    )
    .expect("a funded plain EOA with a canonical, in-window proof is admissible");

    assert_eq!(admitted.sender.address, SENDER);
    assert_eq!(admitted.sender.nonce, 12);
    assert!(admitted.executable);
    assert!(!admitted.sender.is_delegated);
    assert_eq!(chain.header_reads(), 1, "one header read, and no state read at all");
}

#[test]
fn an_anchor_on_an_abandoned_branch_is_rejected() {
    let (proof, state_root) = funded_sender_proof(12, U256::from(5_000_000_000_000_000_000u128));
    // The anchor's header still exists on the abandoned branch; only the canonical-hash check at
    // that height distinguishes it from the canonical one.
    let mut chain = HeaderOnlyChain::canonical_at(HEAD - 5, state_root);
    chain.canonical_hash = B256::repeat_byte(0x01);
    let cache = cold_cache_at(HEAD);

    let error = admit_cold_sender(
        &cache,
        &proof,
        &request(12, U256::ZERO),
        canonical_state_root_lookup(&chain),
    )
    .expect_err("an anchor that is not canonical must not admit anything");

    assert!(matches!(
        error,
        ColdAdmissionError::Proof(SenderProofError::AnchorNotCanonical { .. })
    ));
}

#[test]
fn a_proof_anchored_to_a_forged_state_root_is_rejected() {
    let (proof, _) = funded_sender_proof(12, U256::from(5_000_000_000_000_000_000u128));
    // Canonical anchor, but the canonical header's state root is not the one the proof claims.
    let chain = HeaderOnlyChain::canonical_at(HEAD - 5, B256::repeat_byte(0xfe));
    let cache = cold_cache_at(HEAD);

    let error = admit_cold_sender(
        &cache,
        &proof,
        &request(12, U256::ZERO),
        canonical_state_root_lookup(&chain),
    )
    .expect_err("a forged anchor root must not admit anything");

    assert!(matches!(
        error,
        ColdAdmissionError::Proof(SenderProofError::AnchorRootMismatch { .. })
    ));
}

#[test]
fn a_warm_sender_is_refused_before_the_proof_is_examined() {
    let (proof, state_root) = funded_sender_proof(12, U256::from(5_000_000_000_000_000_000u128));
    let chain = HeaderOnlyChain::canonical_at(HEAD - 5, state_root);
    let mut cache = cold_cache_at(HEAD - 1);
    let mut accessed = BlockAccessedState::default();
    accessed
        .accounts
        .insert(SENDER, AccountData { nonce: 12, balance: U256::ZERO, code_hash: None });
    cache.on_block_executed(HEAD, &accessed);

    let error = admit_cold_sender(
        &cache,
        &proof,
        &request(12, U256::ZERO),
        canonical_state_root_lookup(&chain),
    )
    .expect_err("a warm sender must be admitted from the cache, not from an attached proof");

    assert_eq!(error, ColdAdmissionError::SenderIsWarm { address: SENDER });
    assert_eq!(chain.header_reads(), 0, "the cold precondition runs before any lookup");
}

#[test]
fn an_anchor_older_than_the_account_window_is_rejected() {
    let (mut proof, state_root) = funded_sender_proof(12, U256::from(1_000u64));
    proof.anchor_block_number = HEAD - ACCOUNT_WINDOW - 1;
    let chain = HeaderOnlyChain::canonical_at(proof.anchor_block_number, state_root);
    let cache = cold_cache_at(HEAD);

    let error = admit_cold_sender(
        &cache,
        &proof,
        &request(12, U256::ZERO),
        canonical_state_root_lookup(&chain),
    )
    .expect_err("outside the account window the cold-implies-unchanged invariant does not hold");

    assert!(matches!(error, ColdAdmissionError::Proof(SenderProofError::StaleAnchor { .. })));
}

fn cold_cache_at(block: u64) -> NetworkStateCache {
    NetworkStateCache::restore(
        Default::default(),
        Default::default(),
        Default::default(),
        block,
        Box::new(LastNBlocksPolicy::new(ACCOUNT_WINDOW)),
        Box::new(LastNBlocksPolicy::new(STORAGE_WINDOW)),
    )
}

fn request(nonce: u64, max_cost: U256) -> ColdAdmissionRequest {
    ColdAdmissionRequest {
        input: SenderAdmissionInput {
            tx_sender: SENDER,
            tx_nonce: nonce,
            head_block_number: HEAD,
            account_window: ACCOUNT_WINDOW,
        },
        max_cost,
    }
}

/// A real single-account trie and the inclusion proof for `SENDER` inside it.
fn funded_sender_proof(nonce: u64, balance: U256) -> (SenderAccountProof, B256) {
    let account = Account { nonce, balance, bytecode_hash: None };
    let nibbles = Nibbles::unpack(keccak256(SENDER));
    let value = alloy_rlp::encode(account.into_trie_account(EMPTY_ROOT_HASH));

    let mut builder = HashBuilder::default().with_proof_retainer(ProofRetainer::new(vec![nibbles]));
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
    (SenderAccountProof::new(proof, HEAD - 5, ANCHOR_HASH, state_root, None), state_root)
}

const ANCHOR_HASH: B256 = B256::repeat_byte(0xa9);

/// A chain that can answer canonicality and header questions and nothing else.
///
/// The whole point of this type is what it does *not* implement. It is not a `StateProvider`, so
/// the admission path it is handed to cannot read an account even if it wanted to.
struct HeaderOnlyChain {
    anchor_number: BlockNumber,
    canonical_hash: B256,
    header: Header,
    header_reads: std::cell::Cell<usize>,
}

impl HeaderOnlyChain {
    fn canonical_at(anchor_number: BlockNumber, state_root: B256) -> Self {
        Self {
            anchor_number,
            canonical_hash: ANCHOR_HASH,
            header: Header { number: anchor_number, state_root, ..Default::default() },
            header_reads: std::cell::Cell::new(0),
        }
    }

    fn header_reads(&self) -> usize {
        self.header_reads.get()
    }
}

impl BlockHashReader for HeaderOnlyChain {
    fn block_hash(&self, number: BlockNumber) -> ProviderResult<Option<B256>> {
        Ok((number == self.anchor_number).then_some(self.canonical_hash))
    }

    fn canonical_hashes_range(
        &self,
        _start: BlockNumber,
        _end: BlockNumber,
    ) -> ProviderResult<Vec<B256>> {
        Ok(Vec::new())
    }
}

impl HeaderProvider for HeaderOnlyChain {
    type Header = Header;

    fn header(&self, block_hash: B256) -> ProviderResult<Option<Header>> {
        self.header_reads.set(self.header_reads.get() + 1);
        Ok((block_hash == ANCHOR_HASH).then(|| self.header.clone()))
    }

    fn header_by_number(&self, num: u64) -> ProviderResult<Option<Header>> {
        Ok((num == self.anchor_number).then(|| self.header.clone()))
    }

    fn headers_range(&self, _range: impl RangeBounds<BlockNumber>) -> ProviderResult<Vec<Header>> {
        Ok(Vec::new())
    }

    fn sealed_header(&self, number: BlockNumber) -> ProviderResult<Option<SealedHeader<Header>>> {
        Ok(self.header_by_number(number)?.map(|header| SealedHeader::new(header, ANCHOR_HASH)))
    }

    fn sealed_headers_while(
        &self,
        _range: impl RangeBounds<BlockNumber>,
        _predicate: impl FnMut(&SealedHeader<Header>) -> bool,
    ) -> ProviderResult<Vec<SealedHeader<Header>>> {
        Ok(Vec::new())
    }
}
