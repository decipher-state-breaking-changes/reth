//! Partial-stateless validator library.
//!
//! Reconstructs Ethereum state from a `{hot cache ∪ block witness}` node set
//! (no state database), so a block can be validated statelessly. Shared by the
//! `ps-validator` and `cache-experiment` binaries.

pub mod cache;
pub mod db;
pub mod trie;
pub mod witness;
