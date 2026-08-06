//! The caller half of cold-EOA admission: the canonicality lookup a header source can answer.
//!
//! [`partial_stateless::admit_cold_sender`] applies the cold precondition, the proof verification,
//! and the admission rules, and it takes canonicality as a closure precisely so that it cannot
//! reach for state. This module supplies that closure from a node provider, using only header
//! reads. Keeping the two apart is what makes the no-state-access property structural: the crate
//! holding the admission logic does not depend on `reth-provider` at all, so the only thing left
//! to check is that the caller does not read state either.

use alloy_primitives::B256;
use reth_primitives_traits::AlloyBlockHeader;
use reth_provider::{BlockHashReader, HeaderProvider};

/// A canonicality lookup for [`partial_stateless::admit_cold_sender`], backed by headers alone.
///
/// Returns the canonical state root at `(number, hash)`, or `None` when that block is not on the
/// canonical chain. Two reads, both header-only: the canonical hash at the height, then the header
/// the caller named. Comparing them is what makes an anchor on an abandoned branch fail — the
/// header still exists there, so looking it up without the canonical-hash check would accept it.
pub fn canonical_state_root_lookup<P>(provider: &P) -> impl Fn(u64, B256) -> Option<B256> + '_
where
    P: BlockHashReader + HeaderProvider,
{
    move |number, hash| {
        let canonical = provider.block_hash(number).ok().flatten()?;
        if canonical != hash {
            return None
        }
        provider.header(hash).ok().flatten().map(|header| header.state_root())
    }
}
