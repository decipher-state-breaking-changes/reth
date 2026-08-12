//! Database-free replay of a recorded partial-stateless validation stream.
//!
//! This is the first thing that runs the validator core **as a process of its own**. S0 made the
//! core database-free as a property of its dependency graph, but the only thing that ran it was
//! the ExEx inside a reth node; the standalone entry point `UntrustedAdmission::admit` had unit
//! tests and no driver. Here it gets one, over bytes a live node produced.
//!
//! Two claims come out of a run, and they are different in kind.
//!
//! **Equivalence.** Every commit carries the recording producer's own outcome, and the replay is
//! checked against it field by field. This is what makes S1a's reordering and S1b's
//! decode-and-recover path demonstrably equivalent to the measured path rather than merely
//! self-consistent, and it is the only check that would catch an extraction that is internally
//! deterministic and uniformly wrong. The oracle is an expectation and not an authority: a
//! mismatch means one of the two is wrong, and which one is an investigation.
//!
//! **Rejection coverage, which the corpus cannot supply.** A mainnet recording contains no invalid
//! block, so a replay over one exercises the accept path only. The negatives are derived from
//! recorded frames by [`mutate`], each carrying the class it must produce.
//!
//! **No database, enforced the way S0 enforces it.** This package's normal dependency graph
//! reaches none of `reth-provider`, `reth-db`, `reth-libmdbx`, `reth-mdbx-sys`, `reth-exex`, or
//! `reth-node-builder`, and `check_validator_isolation.sh` runs against it. That is a statement
//! about a Reth state-database access path, not about filesystem I/O — this binary reads a spool
//! directory, which is the whole point of it.

pub mod compare;
pub mod driver;
pub mod mutate;
pub mod spool;

pub use compare::Disagreement;
pub use driver::{replay, ReplayOptions, ReplayReport};
pub use mutate::Mutation;
pub use spool::{read_spool, Spool, SpooledFrame};
