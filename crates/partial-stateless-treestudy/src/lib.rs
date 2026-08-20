//! What a cache-complement witness would cost if Ethereum's state were committed in a different
//! tree.
//!
//! The system this crate belongs to reduces witness transfer by keeping a bounded authenticated
//! cache and shipping only the complement. Every byte of that saving is currently measured against
//! the hexary Merkle-Patricia Trie, because that is what mainnet headers commit to. Two proposals
//! would change the commitment: EIP-7864's unified binary tree, and EIP-6800's Verkle tree. Both
//! change the witness by construction — the binary tree by trading branch width for depth, Verkle
//! by replacing sibling hashes with a vector commitment — and neither preserves the two-layer
//! account/storage split the current witness is shaped around.
//!
//! The question this crate answers is therefore not "does the design still work" but "how much of
//! the measured saving survives the tree". It answers it offline, from a recorded corpus, with the
//! cache policy and the access sequence held exactly fixed, so that the only thing varying between
//! arms is the commitment scheme.
//!
//! It cannot answer it on a live chain, and does not try to. No chain commits a binary or Verkle
//! root today, so a validator cannot check its recomputed root against an untrusted header — the
//! property the live experiment rests on. What survives the move offline is the part that does not
//! need an anchor: how many nodes a miss set opens, how many of them a retained cache already
//! holds, and how many bytes the difference is.

pub mod corpus;
pub mod keys;
pub mod mpt;
pub mod population;
pub mod report;
pub mod study;
pub mod witness;

pub use corpus::{Corpus, CorpusBlock};
pub use keys::{Eip6800Keys, Eip7864Keys, HeaderLayout, StemId, TreeEmbedding, TreeKey};
pub use population::BackgroundPopulation;
pub use report::{ArmRatios, ArmSummary, RunReport};
pub use study::{Arm, ArmSpec, BlockResult, Populations, StudyParams};
pub use witness::{RetainedStems, StemOccupancy, StemTargets, WitnessCost};
