//! A structural model of the MPT witness, used to check the modelling rather than to report it.
//!
//! The MPT arm of this study does not need a model: the corpus was built by a live node and its
//! witnesses were measured. What the model is for is the opposite direction — the binary and Verkle
//! arms *are* modelled, and nothing about their numbers is checkable on its own. Running the same
//! machinery over the one tree whose real answer is known turns "the model is plausible" into a
//! number: how far the predicted node count sits from the measured one, on the same blocks, under
//! the same cache policy.
//!
//! The model deliberately predicts node *counts* and not bytes. Node counts are what the path model
//! produces; bytes are what RLP, inline nodes under 32 subject bytes, and branch masks produce, and
//! folding those in would let an encoding error cancel a structural one.
//!
//! Two populations drive it. The account trie's size is measured from the node the corpus came
//! from. Per-account storage trie sizes are not knowable from the corpus, so a single global size
//! stands in for all of them and is swept; one fitted parameter against four measured arms is a
//! weak enough hand to make agreement mean something.

use crate::{
    keys::{Prefix, TreeRegion},
    population::BackgroundPopulation,
    witness::{prefix_range, RetainedStems},
};
use std::collections::BTreeSet;

/// Nibbles per hexary level.
const LEVEL_BITS: u32 = 4;
/// A keccak key is 64 nibbles.
const MAX_LEVELS: u32 = 64;

/// What the MPT path model predicts for one block.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct MptStructure {
    /// Distinct trie nodes the witness must carry.
    pub nodes: u64,
    /// Nodes the retained frontier already covered.
    pub nodes_held_by_cache: u64,
    /// Account-trie share of the carried nodes.
    pub account_nodes: u64,
    /// Storage-trie share of the carried nodes.
    pub storage_nodes: u64,
}

/// Counts the distinct trie nodes a hexary multiproof over `targets` carries.
///
/// A hexary path is a quarter as long as the binary path over the same population, and each of its
/// nodes carries up to fifteen sibling references rather than one. That trade is the reason a
/// binary tree is proposed at all, and stating it in the same accounting as the binary arm is what
/// lets the two be compared rather than merely both reported.
pub fn hexary_path_nodes(
    targets: &BTreeSet<[u8; 32]>,
    retained: &RetainedStems,
    population: &BackgroundPopulation,
) -> u64 {
    let mut carried: BTreeSet<Prefix> = BTreeSet::new();
    for key in targets {
        let descent = population.descend(key, MAX_LEVELS * LEVEL_BITS);
        let levels = descent.alone_at.div_ceil(LEVEL_BITS).clamp(1, MAX_LEVELS);
        let levels = deepen_for_real_keys(key, levels, targets, retained);
        // Inclusive of `levels`: the branches sit at depths 0..levels-1 and the leaf the path ends
        // at is itself a node the witness carries.
        for level in 0..=levels {
            let node = Prefix::of(key, level * LEVEL_BITS);
            // A node whose subtree the receiver already retains is in its trie already.
            if retained.any_under(TreeRegion::Unified, node) {
                continue
            }
            carried.insert(node);
        }
    }
    carried.len() as u64
}

/// Pushes a branch point deeper when another real key shares the path.
fn deepen_for_real_keys(
    key: &[u8; 32],
    from: u32,
    targets: &BTreeSet<[u8; 32]>,
    retained: &RetainedStems,
) -> u32 {
    let mut levels = from;
    while levels < MAX_LEVELS {
        let prefix = Prefix::of(key, levels * LEVEL_BITS);
        let (lo, hi) = prefix_range(prefix);
        let shares = targets.range(lo..=hi).any(|other| other != key) ||
            retained.any_under(TreeRegion::Unified, prefix);
        if !shares {
            break
        }
        levels += 1;
    }
    levels
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(byte: u8) -> [u8; 32] {
        let mut out = [0u8; 32];
        out[0] = byte;
        out
    }

    #[test]
    fn a_hexary_path_is_shorter_than_the_binary_one_it_models() {
        let population = BackgroundPopulation::new(412_044_818, "accounts");
        let mut targets = BTreeSet::new();
        targets.insert(key(0x33));
        let nodes = hexary_path_nodes(&targets, &RetainedStems::new(), &population);
        // log16(4.12e8) is about 7.2, so one key alone should sit around eight nodes deep.
        assert!((5..=11).contains(&nodes), "one account path produced {nodes} nodes");
    }

    #[test]
    fn shared_prefixes_are_counted_once() {
        let population = BackgroundPopulation::new(412_044_818, "accounts");
        let mut one = BTreeSet::new();
        one.insert(key(0x33));
        let single = hexary_path_nodes(&one, &RetainedStems::new(), &population);

        let mut two = one.clone();
        two.insert(key(0x34));
        let pair = hexary_path_nodes(&two, &RetainedStems::new(), &population);
        assert!(pair < single * 2, "shared root levels must not be counted twice");
    }
}
