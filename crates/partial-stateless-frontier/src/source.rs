//! Answering parent-state proof requests out of a recorded witness instead of a database.
//!
//! This is the offline half of the seam the shared construction core is built around. The core
//! asks for targets; a live builder answers from the node's state database, and this answers from
//! the policy-neutral full witness the capture recorded for the same block. Nothing else about the
//! build differs, which is the point: a sidecar produced here and one produced live are the same
//! object built by the same code from the same parent state.
//!
//! **Why a subset of the recorded witness is always the right answer.** The recording proved every
//! accessed key and every post-state mutation path from the root, against a cold cache and an empty
//! trie. A policy with a warm cache asks for a strict subset of that: its miss set is a subset of
//! the accessed set, its uncovered mutation paths are a subset of all mutation paths, and the
//! structural nodes its transition discovers are nodes the cold transition already had to obtain
//! for itself. So the recorded node set is a superset, and selecting from it by target is exactly
//! what a provider would have returned.
//!
//! Where that argument could fail, it fails loudly rather than quietly. A target this source
//! cannot serve produces a proof that reveals nothing, and the transition then either makes no
//! progress or computes a root that does not match the block header — both hard errors, on the
//! block that caused them.

use alloy_primitives::{Bytes, B256};
use partial_stateless::{decode_transition_witness, ParallelProof, TransitionProofSource};
use reth_trie_common::{DecodedMultiProofV2, MultiProofTargetsV2};

/// A recorded full witness, serving proofs by selection.
#[derive(Debug)]
pub struct RecordedFullWitnessSource {
    /// The parent state root every selection is anchored to.
    parent_state_root: B256,
    /// The whole recorded witness, decoded once per block rather than per proof request.
    full: DecodedMultiProofV2,
}

impl RecordedFullWitnessSource {
    /// Decodes a recorded node set against the parent state root it was proved under.
    pub fn new(parent_state_root: B256, nodes: &[Bytes]) -> eyre::Result<Self> {
        let full = decode_transition_witness(parent_state_root, nodes)?;
        Ok(Self { parent_state_root, full })
    }

    /// The parent state root this source is anchored to.
    pub const fn parent_state_root(&self) -> B256 {
        self.parent_state_root
    }
}

impl TransitionProofSource for RecordedFullWitnessSource {
    fn multiproof_v2(&self, targets: MultiProofTargetsV2) -> eyre::Result<DecodedMultiProofV2> {
        let requested_accounts = targets.account_targets.len();
        let requested_storage: usize = targets.storage_targets.values().map(Vec::len).sum();
        let mut proof = self.full.clone();
        proof.retain_targets(&targets);
        // The one incompleteness this side can name on its own. Anything subtler — a path that is
        // present but truncated — is caught by the transition that consumes it, and caught with
        // the block number attached either way.
        if proof.is_empty() && requested_accounts + requested_storage > 0 {
            eyre::bail!(
                "the recorded full witness has no node for any of {requested_accounts} account \
                 and {requested_storage} storage targets under parent state root {:?}; the \
                 recording is incomplete for this block",
                self.parent_state_root
            )
        }
        Ok(proof)
    }

    /// None, always.
    ///
    /// A recorded witness has one access path and no worker pool, and saying so is not a detail:
    /// the shared core labels an initial proof `serial` here and `serial-low-width` where a wide
    /// path existed and was declined on width, so a report cannot confuse "there was no fast path"
    /// with "the fast path was not worth taking".
    fn parallel_initial_proof(
        &self,
    ) -> Option<&dyn Fn(MultiProofTargetsV2) -> eyre::Result<ParallelProof>> {
        None
    }
}
