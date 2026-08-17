//! The one rejection that cannot be reached without executing the block.
//!
//! `mutations.rs` proves that every *admission* mutation is refused, and refused early. That is
//! the accept path's negative for everything a validator can check by looking: the announced hash,
//! the gas-limit ramp, sender recovery. None of them reaches a rule that needs the block's own
//! execution, so until this suite existed the post-execution boundary — the receipts root and
//! logs bloom the EVM actually produces, compared with the ones the header committed to — had no
//! end-to-end evidence at all.
//!
//! Corpus-gated and `#[ignore]`d, for the same reason the rewind suite is: a real transition needs
//! a real witness, and witnesses live beside the bench runs rather than in the repository. To run:
//!
//! ```text
//! PS_MUTATION_FIXTURE_SPOOL=<recorded-spool-dir> \
//!     cargo test -p partial-stateless-replay --release --test transition_mutations \
//!     -- --ignored --test-threads=1
//! ```
//!
//! With the variable missing at run time these panic rather than skip: a coverage test that
//! reports success without having run is worse than one that is absent.

use partial_stateless::PartialStatelessSidecar;
use partial_stateless_replay::{
    driver::replay, spool::SpoolIter, ReplayOptions, TransitionMutation,
};
use partial_stateless_stream::{FrameLimits, StreamEvent};
use std::path::PathBuf;

/// How many recorded commits carry a mutation. The rule under test is the same rule on every
/// block, so this is a coverage budget rather than a sample: five blocks prove what five hundred
/// would, and each one costs a full extra execution.
const MUTATED_BLOCKS: usize = 5;

/// How much of the corpus each replay covers. Kept just above the mutation budget so the run also
/// shows that the honest commits *after* the mutated ones are unaffected by them.
const REPLAYED_BLOCKS: usize = 10;

fn fixture_spool() -> PathBuf {
    match std::env::var("PS_MUTATION_FIXTURE_SPOOL") {
        Ok(path) if !path.is_empty() => PathBuf::from(path),
        _ => panic!(
            "PS_MUTATION_FIXTURE_SPOOL is not set; these #[ignore]d tests only run against a \
             recorded spool such as a bench run's spool/ directory"
        ),
    }
}

fn options(mutations_transition: Option<usize>) -> ReplayOptions {
    ReplayOptions {
        limit: Some(REPLAYED_BLOCKS),
        // The admission mutations are a separate claim with a separate cost; leaving them on
        // would put their failures in the same list and make this test's subject ambiguous.
        mutations: false,
        mutations_transition,
        ..ReplayOptions::default()
    }
}

/// The end-to-end claim: admitted, executed in full, and refused by the post-execution rule —
/// with the pair that has to go on validating left exactly where it was.
///
/// Every one of those is asserted through `mutation_failures`, which the driver appends to when
/// the mutation is refused too early, accepted, refused for the wrong reason, leaves its probe
/// usable, or moves the real pair. An empty list is the whole chain holding.
#[test]
#[ignore = "needs a recorded spool in PS_MUTATION_FIXTURE_SPOOL"]
fn a_wrong_receipts_root_is_refused_after_the_block_has_executed() {
    let dir = fixture_spool();
    let report = replay(&dir, &options(Some(MUTATED_BLOCKS))).expect("the corpus replays");

    assert_eq!(
        report.transition_mutations_checked,
        (MUTATED_BLOCKS * TransitionMutation::ALL.len()) as u64,
        "the budget must be spent, or the rest of this test is asserting about nothing"
    );
    assert!(
        report.mutation_failures.is_empty(),
        "transition mutation coverage failed: {:?}",
        report.mutation_failures
    );
    assert!(
        report.transition_mutation_us > 0,
        "a full execution cannot cost nothing; the mutations did not run"
    );
    assert!(
        report.disagreements.is_empty(),
        "the honest commits disagreed: {:?}",
        report.disagreements
    );
    assert!(report.failures.is_empty(), "the honest commits failed: {:?}", report.failures);
}

/// The mutation must be invisible to everything the run measures.
///
/// Not a nicety: the driver's per-block timings are what every reported latency population is
/// built from, and a coverage switch that inflated them would make the evidence for one claim the
/// noise in another. Compared against a control replay of the same commits with the switch off —
/// the verdicts, the sequence, and the honest counters all have to be identical.
#[test]
#[ignore = "needs a recorded spool in PS_MUTATION_FIXTURE_SPOOL"]
fn the_mutation_does_not_touch_what_the_run_measures() {
    let dir = fixture_spool();
    let control = replay(&dir, &options(None)).expect("the corpus replays");
    let armed = replay(&dir, &options(Some(MUTATED_BLOCKS))).expect("the corpus replays");

    assert_eq!(control.transition_mutations_checked, 0, "the control must be a control");
    assert_eq!(armed.commits, control.commits);
    assert_eq!(armed.witnessed, control.witnessed);
    assert_eq!(armed.reconstructed, control.reconstructed);
    assert_eq!(armed.absent, control.absent);
    assert_eq!(armed.disagreements.len(), control.disagreements.len());
    assert_eq!(armed.failures.len(), control.failures.len());
    assert_eq!(armed.continuous(), control.continuous());
    assert_eq!(armed.complete(), control.complete());

    let verdicts =
        |report: &partial_stateless_replay::ReplayReport| -> Vec<(u64, u64, &'static str)> {
            report
                .blocks
                .iter()
                .map(|block| (block.number, block.sequence, block.verdict))
                .collect()
        };
    assert_eq!(
        verdicts(&armed),
        verdicts(&control),
        "the same corpus must produce the same verdicts on the same frames"
    );
}

/// The rebinding Codex's review caught, checked against a sidecar a producer really wrote.
///
/// Two fields commit to the block hash, and the validator checks them in that order: the prefilter
/// compares `sidecar.block_hash` to the block's own hash, and the cache-context check then compares
/// `next_cache_anchor.block_hash` to `sidecar.block_hash`. Rebinding one and not the other trades
/// one pre-execution refusal for another — and a mutation refused before the EVM runs still *looks*
/// refused, which is exactly why this is asserted directly rather than left to the run above.
#[test]
#[ignore = "needs a recorded spool in PS_MUTATION_FIXTURE_SPOOL"]
fn the_rebinding_moves_both_hash_bindings_and_nothing_else() {
    let dir = fixture_spool();
    let limits = FrameLimits::default();
    let mut iter = SpoolIter::open(&dir, &limits).expect("the spool opens");

    let (payload, sidecar) = loop {
        let frame = iter.next_frame().expect("the spool reads").expect("a commit before the end");
        if let StreamEvent::Commit(commit) = frame.event {
            let (input, _) = commit.as_ref().clone().split();
            let Ok(Some(payload)) = input.payload() else { continue };
            let sidecar: PartialStatelessSidecar =
                bincode::deserialize(&input.sidecar).expect("the recorded sidecar decodes");
            break (payload, sidecar)
        }
    };

    let announced = payload.payload.block_hash();
    let mutated = TransitionMutation::ReceiptsRoot.apply(&payload).expect("the mutation applies");
    let rebound_hash = mutated.payload.block_hash();
    assert_ne!(rebound_hash, announced, "a re-sealed block must announce a new hash");
    assert_eq!(sidecar.block_hash, announced, "the recorded sidecar binds to the recorded block");

    let rebound = TransitionMutation::ReceiptsRoot.rebind_sidecar(&sidecar, rebound_hash);
    assert_eq!(rebound.block_hash, rebound_hash);
    assert_eq!(rebound.next_cache_anchor.block_hash, rebound_hash);

    // Everything else is a claim about the *parent* state or about cache contents, and re-sealing
    // a header falsifies none of it. A rebinding that moved any of these would be manufacturing a
    // second defect and the rejection could no longer be attributed to the receipts root.
    assert_eq!(rebound.parent_hash, sidecar.parent_hash);
    assert_eq!(rebound.parent_state_root, sidecar.parent_state_root);
    assert_eq!(rebound.block_number, sidecar.block_number);
    assert_eq!(rebound.cache_block, sidecar.cache_block);
    assert_eq!(rebound.cache_policy_id, sidecar.cache_policy_id);
    assert_eq!(rebound.witness_commitment, sidecar.witness_commitment);
    assert_eq!(rebound.cache_miss_targets, sidecar.cache_miss_targets);
    assert_eq!(rebound.miss_manifest, sidecar.miss_manifest);
    assert_eq!(rebound.prev_cache_anchor, sidecar.prev_cache_anchor);
    assert_eq!(rebound.next_cache_anchor.cache_root, sidecar.next_cache_anchor.cache_root);
    assert_eq!(rebound.next_cache_anchor.block_number, sidecar.next_cache_anchor.block_number);
    assert_eq!(
        rebound.next_cache_anchor.cache_policy_id,
        sidecar.next_cache_anchor.cache_policy_id
    );
}
