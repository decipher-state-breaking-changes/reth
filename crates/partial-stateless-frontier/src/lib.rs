//! Generating every cache policy's real sidecar for one recorded block set, without a database.
//!
//! A cache-policy comparison measured across separate live runs cannot separate the policy from
//! the run: different blocks, different mempool, different machine state. This crate removes that
//! by replaying **one** recorded corpus through every policy — the block set is identical by
//! construction, so the only thing that varies is the policy.
//!
//! **It has no state database, and that is a property of the dependency graph rather than of the
//! code paths taken.** Nothing reachable from here can open a Reth provider or an MDBX
//! environment, so no error, gap, or fallback branch can quietly answer a proof request from a
//! database instead of from the recorded witness. `scripts/check_validator_isolation.sh` in
//! `partial-stateless-exex` enforces that, and the same script enforces the keccak and
//! signature-recovery build profile that makes any timing taken from this binary describe the
//! implementation production runs.
//!
//! Every sidecar it produces is built by the same construction core a live node builder uses, with
//! [`RecordedFullWitnessSource`](source::RecordedFullWitnessSource) standing where the state
//! database stands. Two builders would make a policy comparison a comparison of builders.
//!
//! What this measures: sidecar sizes, node and miss sets, cache footprints, and the
//! **policy-dependent** part of validation cost — the sidecar decode plus the transition and cache
//! commit, which is everything that changes when the cache policy changes.
//!
//! What it does not measure, and must not be read as measuring:
//!
//! - **Production builder latency.** Selecting nodes from a decoded witness in memory is not
//!   generating a multiproof from a state database. That number belongs to a live run.
//! - **Absolute standalone validation latency.** The per-arm timer opens at the sidecar decode, so
//!   it excludes the payload decode and admission every arm pays identically (reported once per
//!   block instead) and the whole delivery path a live consumer pays. A figure that could be called
//!   a standalone latency comes from replaying a recorded stream through `ps-replay`, whose
//!   boundary opens at the frame read.
//!
//! Both exclusions are stated in the output as well as here: the summary carries
//! `builder_latency_eligible: false` and `standalone_latency_eligible: false`.

pub mod generate;
pub mod policy;
pub mod report;
pub mod source;

pub use generate::{generate_block, BlockResult, ChainCursor, GeneratorRules, PolicyBlockResult};
pub use policy::{rotated_order, ArmKind, PolicySpec, PolicyState};
pub use report::{RunReport, RunSummary};
pub use source::RecordedFullWitnessSource;
