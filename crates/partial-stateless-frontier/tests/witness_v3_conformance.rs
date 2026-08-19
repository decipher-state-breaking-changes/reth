//! Conformance for the trimmed (v3) sidecar witness: a v3 build must agree with the v2 build of
//! the same block on everything but the wire bytes, its node list must be a strict subset of the
//! v2 list, a validator holding the same parent generation must graft it onto the identical
//! post-transition frontier the builder reached, and every canonicality violation must reject
//! while leaving the validator's parent generation untouched.
//!
//! The chain deliberately ends on the deletion-heavy block: branch collapses force structural
//! discovery rounds, which is where the graft loop has to resolve demands the initial target set
//! never named — the hardest part of the v2/v3 symmetry claim.

mod world;
use world::*;

use alloy_primitives::{keccak256, Bytes, B256};
use partial_stateless::{
    build_policy_sidecar, sidecar_semantic_digest, try_compute_trustless_state_root_v3_from_hashed,
    witness_check::{
        materialize_sidecar_witness_after_prefilter,
        materialize_sidecar_witness_after_prefilter_with_cache,
    },
    BlockTransitionRef, CacheConfig, MaterializedStateProof, PartialExecutionWitnessState,
    PartialStatelessSidecar, PolicySidecarBuild, TransitionBuildContext,
    WITNESS_V3_FRONTIER_VERSION,
};
use reth_trie_common::HashedPostState;
use std::collections::BTreeSet;

/// Everything the last block's comparison needs, produced by warming three identical sides
/// through the first two blocks and stopping at the third.
struct WarmedComparison {
    config: CacheConfig,
    /// Warmed to the last block's parent; used for the v2 build.
    v2_side: Side,
    /// Warmed identically; used for the v3 build.
    v3_side: Side,
    /// Warmed identically; plays the receiving validator.
    validator: Side,
    parent: World,
    parent_hash: B256,
    last: BlockSpec,
}

fn warm_three_sides() -> WarmedComparison {
    let mut blocks = chain();
    let last = blocks.pop().expect("the chain has blocks");
    warm_sides_through(&blocks, last)
}

fn warm_sides_through(blocks: &[BlockSpec], last: BlockSpec) -> WarmedComparison {
    let config = CacheConfig { account_window: 60, storage_window: 30 };
    let first = blocks.first().expect("the chain has warmup blocks").number;

    let mut sides = [
        Side::cold_at(&config, first - 1),
        Side::cold_at(&config, first - 1),
        Side::cold_at(&config, first - 1),
    ];
    let mut parent = genesis();
    let mut parent_hash = B256::repeat_byte(0xf0);

    for spec in blocks {
        let parent_state_root = parent.state_root();
        let complete = parent.complete_witness();
        let provider = WholeTrieSource::new(parent_state_root, &complete);
        let ctx = TransitionBuildContext::uninstrumented(&provider);
        let accessed = accessed_state(&parent, spec);
        let (child, post) = apply(&parent, spec);
        let block_hash = keccak256(spec.number.to_be_bytes());
        let block_ref = BlockTransitionRef {
            block_number: spec.number,
            block_hash,
            parent_hash,
            parent_state_root,
            expected_state_root: child.state_root(),
            ancestor_headers: &[],
        };
        for side in &mut sides {
            let build = build_policy_sidecar(
                &ctx,
                block_ref,
                &post,
                &accessed,
                &mut side.cache,
                &side.trie,
                &config,
            )
            .expect("the warmup sidecar builds");
            side.trie = build.build.next_trie_cache;
        }
        parent = child;
        parent_hash = block_hash;
    }

    let [v2_side, v3_side, validator] = sides;
    WarmedComparison { config, v2_side, v3_side, validator, parent, parent_hash, last }
}

/// The last block's inputs, shared by every test below.
struct LastBlock {
    block_ref_hash: B256,
    parent_state_root: B256,
    expected_state_root: B256,
    post: HashedPostState,
    accessed: partial_stateless::BlockAccessedState,
    complete: alloy_primitives::map::B256Map<Bytes>,
}

fn last_block_inputs(warmed: &WarmedComparison) -> LastBlock {
    let parent_state_root = warmed.parent.state_root();
    let complete = warmed.parent.complete_witness();
    let accessed = accessed_state(&warmed.parent, &warmed.last);
    let (child, post) = apply(&warmed.parent, &warmed.last);
    LastBlock {
        block_ref_hash: keccak256(warmed.last.number.to_be_bytes()),
        parent_state_root,
        expected_state_root: child.state_root(),
        post,
        accessed,
        complete,
    }
}

fn build_last(
    warmed: &mut WarmedComparison,
    inputs: &LastBlock,
    trimmed: bool,
) -> PolicySidecarBuild {
    let provider = WholeTrieSource::new(inputs.parent_state_root, &inputs.complete);
    let ctx = if trimmed {
        TransitionBuildContext::uninstrumented(&provider).with_trimmed_witness()
    } else {
        TransitionBuildContext::uninstrumented(&provider)
    };
    let block_ref = BlockTransitionRef {
        block_number: warmed.last.number,
        block_hash: inputs.block_ref_hash,
        parent_hash: warmed.parent_hash,
        parent_state_root: inputs.parent_state_root,
        expected_state_root: inputs.expected_state_root,
        ancestor_headers: &[],
    };
    let side = if trimmed { &mut warmed.v3_side } else { &mut warmed.v2_side };
    build_policy_sidecar(
        &ctx,
        block_ref,
        &inputs.post,
        &inputs.accessed,
        &mut side.cache,
        &side.trie,
        &warmed.config,
    )
    .expect("the last block's sidecar builds")
}

fn trimmed_nodes(sidecar: &PartialStatelessSidecar) -> &Vec<Bytes> {
    match &sidecar.witness.state {
        PartialExecutionWitnessState::MptTrimmedTransitionNodes { nodes, .. } => nodes,
        other => panic!("expected a trimmed witness, got {other:?}"),
    }
}

#[test]
fn a_trimmed_build_agrees_with_the_full_build_and_ships_a_strict_subset() {
    let mut warmed = warm_three_sides();
    let inputs = last_block_inputs(&warmed);
    let parent_fingerprint = warmed.v3_side.trie.retention_fingerprint();

    let v2 = build_last(&mut warmed, &inputs, false);
    let v3 = build_last(&mut warmed, &inputs, true);

    // The transition itself is untouched by trimming: same root, same structural rounds, and the
    // same post-retention trie generation on both sides.
    assert_eq!(v2.build.state_root, v3.build.state_root);
    assert_eq!(v2.build.structural_rounds, v3.build.structural_rounds);
    assert!(
        v2.build.structural_rounds > 0,
        "the fixture stopped exercising structural rounds, so the graft loop is untested"
    );
    assert!(v2.build.next_trie_cache.structurally_eq(&v3.build.next_trie_cache));
    assert_eq!(v2.sidecar.miss_manifest, v3.sidecar.miss_manifest);
    assert_eq!(v2.sidecar.prev_cache_anchor, v3.sidecar.prev_cache_anchor);
    assert_eq!(v2.sidecar.next_cache_anchor, v3.sidecar.next_cache_anchor);

    // The wire names the retention contract it was cut against.
    match &v3.sidecar.witness.state {
        PartialExecutionWitnessState::MptTrimmedTransitionNodes {
            frontier_version,
            retention_fingerprint,
            nodes,
        } => {
            assert_eq!(*frontier_version, WITNESS_V3_FRONTIER_VERSION);
            assert_eq!(*retention_fingerprint, parent_fingerprint);
            assert!(!nodes.is_empty(), "a warm block with misses still ships fragments");
        }
        other => panic!("expected a trimmed witness, got {other:?}"),
    }

    // Strict subset: every fragment byte-string is a v2 node, and the trim removed something.
    let v2_nodes: BTreeSet<&Bytes> = v2.build.nodes.iter().collect();
    let v3_nodes = trimmed_nodes(&v3.sidecar);
    for node in v3_nodes {
        assert!(v2_nodes.contains(node), "a trimmed node is not part of the full witness");
    }
    assert!(
        v3_nodes.len() < v2.build.nodes.len(),
        "the trim removed nothing on a warm trie ({} vs {})",
        v3_nodes.len(),
        v2.build.nodes.len()
    );

    // The two wires are different protocol content, and the digest knows it.
    assert_ne!(
        sidecar_semantic_digest(&v2.sidecar).unwrap(),
        sidecar_semantic_digest(&v3.sidecar).unwrap()
    );
}

#[test]
fn a_validator_grafts_the_trimmed_witness_onto_the_builder_frontier() {
    let mut warmed = warm_three_sides();
    let inputs = last_block_inputs(&warmed);

    let v2 = build_last(&mut warmed, &inputs, false);
    let v3 = build_last(&mut warmed, &inputs, true);

    // Pre-EVM value materialization: the composite walk over (validator trie, fragment map) must
    // read exactly the values the self-contained v2 witness proves.
    let v2_materialized = materialize_sidecar_witness_after_prefilter(&v2.sidecar)
        .expect("the self-contained witness materializes");
    let v3_materialized = materialize_sidecar_witness_after_prefilter_with_cache(
        &v3.sidecar,
        Some(&warmed.validator.trie),
    )
    .expect("the trimmed witness materializes against the identical parent trie");
    assert_eq!(v2_materialized.accounts, v3_materialized.accounts);
    assert_eq!(v2_materialized.storage, v3_materialized.storage);

    let MaterializedStateProof::TrimmedTransition(session) = v3_materialized.state_proof else {
        panic!("a trimmed witness materializes into a trimmed session");
    };

    // The graft: transactional clone, same demand-driven loop the builder ran, full consumption.
    let mut next_trie = warmed.validator.trie.clone();
    let root = try_compute_trustless_state_root_v3_from_hashed(
        session,
        &warmed.validator.trie,
        &mut next_trie,
        inputs.post.clone(),
        &v3.sidecar.miss_manifest,
    )
    .expect("the graft completes and consumes the whole wire");
    assert_eq!(root, inputs.expected_state_root);

    // Frontier symmetry — the compatibility contract v3 rests on: after the same retention, the
    // validator's grafted generation is structurally identical to the builder's.
    warmed.validator.cache.on_block_executed(warmed.last.number, &inputs.accessed);
    next_trie.retain_from_value_cache(&warmed.validator.cache);
    assert!(
        next_trie.structurally_eq(&v3.build.next_trie_cache),
        "the validator landed on a different frontier than the builder"
    );
}

/// Runs the whole validator pipeline over a (possibly tampered) trimmed sidecar and returns the
/// result, asserting afterwards that the validator's parent generation did not move.
fn graft_outcome(
    validator: &Side,
    sidecar: &PartialStatelessSidecar,
    post: &HashedPostState,
) -> Result<B256, String> {
    let pristine = validator.trie.clone();
    let result = (|| {
        let materialized =
            materialize_sidecar_witness_after_prefilter_with_cache(sidecar, Some(&validator.trie))
                .map_err(|err| err.to_string())?;
        let MaterializedStateProof::TrimmedTransition(session) = materialized.state_proof else {
            return Err("not a trimmed session".to_string());
        };
        let mut next_trie = validator.trie.clone();
        try_compute_trustless_state_root_v3_from_hashed(
            session,
            &validator.trie,
            &mut next_trie,
            post.clone(),
            &sidecar.miss_manifest,
        )
        .map_err(|err| err.to_string())
    })();
    assert!(
        validator.trie.structurally_eq(&pristine),
        "a rejected sidecar moved the validator's parent generation"
    );
    assert_eq!(validator.trie.state_root(), pristine.state_root());
    result
}

fn with_nodes(
    sidecar: &PartialStatelessSidecar,
    mutate: impl FnOnce(&mut Vec<Bytes>),
) -> PartialStatelessSidecar {
    let mut tampered = sidecar.clone();
    let PartialExecutionWitnessState::MptTrimmedTransitionNodes { nodes, .. } =
        &mut tampered.witness.state
    else {
        panic!("expected a trimmed witness");
    };
    mutate(nodes);
    tampered
}

#[test]
fn every_canonicality_violation_rejects_and_leaves_the_parent_untouched() {
    let mut warmed = warm_three_sides();
    let inputs = last_block_inputs(&warmed);

    let v2 = build_last(&mut warmed, &inputs, false);
    let v3 = build_last(&mut warmed, &inputs, true);
    let sidecar = &v3.sidecar;

    // The untampered sidecar passes — otherwise the rejections below prove nothing.
    graft_outcome(&warmed.validator, sidecar, &inputs.post).expect("the canonical sidecar grafts");

    // Missing node: drop each fragment in turn; every single one must be load-bearing.
    let node_count = trimmed_nodes(sidecar).len();
    for index in 0..node_count {
        let tampered = with_nodes(sidecar, |nodes| {
            nodes.remove(index);
        });
        let err = graft_outcome(&warmed.validator, &tampered, &inputs.post)
            .expect_err("a witness with a missing node was accepted");
        assert!(
            err.contains("missing from the trimmed witness"),
            "unexpected rejection for a dropped node: {err}"
        );
    }

    // Re-ordered nodes: the wire has exactly one canonical encoding, so an adjacent swap
    // rejects even though the content is identical.
    let err = graft_outcome(
        &warmed.validator,
        &with_nodes(sidecar, |nodes| nodes.swap(0, 1)),
        &inputs.post,
    )
    .expect_err("a witness with re-ordered nodes was accepted");
    assert!(err.contains("ascending"), "unexpected rejection for a swap: {err}");

    // Duplicate node, adjacent so it is a duplicate rather than a re-ordering; a duplicate
    // anywhere else breaks the ascending order first and rejects as that.
    let err = graft_outcome(
        &warmed.validator,
        &with_nodes(sidecar, |nodes| {
            let first = nodes[0].clone();
            nodes.insert(1, first);
        }),
        &inputs.post,
    )
    .expect_err("a witness with a duplicate node was accepted");
    assert!(err.contains("duplicate node"), "unexpected rejection for a duplicate: {err}");

    // Unconsumed node: append a genuine parent-state node the fragments never reach. The v2
    // witness is a strict superset, so it has one to offer.
    let v3_set: BTreeSet<&Bytes> = trimmed_nodes(sidecar).iter().collect();
    let extra = v2
        .build
        .nodes
        .iter()
        .find(|node| !v3_set.contains(node))
        .expect("the trim removed at least one node")
        .clone();
    let err = graft_outcome(
        &warmed.validator,
        &with_nodes(sidecar, |nodes| {
            // At its sorted position, so the wire stays canonical and the rejection is
            // genuinely the unconsumed-node rule.
            let position = nodes.binary_search(&extra).expect_err("not already present");
            nodes.insert(position, extra);
        }),
        &inputs.post,
    )
    .expect_err("a witness with an unconsumed node was accepted");
    assert!(err.contains("never consumed"), "unexpected rejection for an extra node: {err}");

    // Standalone inline entry: sub-32-byte content is unaddressable on the wire.
    let err = graft_outcome(
        &warmed.validator,
        &with_nodes(sidecar, |nodes| nodes.push(Bytes::from_static(&[0x80]))),
        &inputs.post,
    )
    .expect_err("a witness with a standalone inline entry was accepted");
    assert!(err.contains("standalone"), "unexpected rejection for an inline entry: {err}");

    // Retention fingerprint mismatch: fail-closed before a single node is read.
    let mut wrong_fingerprint = sidecar.clone();
    if let PartialExecutionWitnessState::MptTrimmedTransitionNodes {
        retention_fingerprint, ..
    } = &mut wrong_fingerprint.witness.state
    {
        *retention_fingerprint = B256::repeat_byte(0xde);
    }
    let err = graft_outcome(&warmed.validator, &wrong_fingerprint, &inputs.post)
        .expect_err("a witness cut against a different retention fingerprint was accepted");
    assert!(err.contains("retention fingerprint"), "unexpected rejection: {err}");

    // Retention version mismatch.
    let mut wrong_version = sidecar.clone();
    if let PartialExecutionWitnessState::MptTrimmedTransitionNodes { frontier_version, .. } =
        &mut wrong_version.witness.state
    {
        *frontier_version = WITNESS_V3_FRONTIER_VERSION + 1;
    }
    let err = graft_outcome(&warmed.validator, &wrong_version, &inputs.post)
        .expect_err("a witness naming an unknown frontier version was accepted");
    assert!(err.contains("frontier version"), "unexpected rejection: {err}");

    // A cache-less verifier cannot verify a trimmed sidecar at all.
    let err = materialize_sidecar_witness_after_prefilter(sidecar)
        .expect_err("a trimmed witness materialized without the parent trie");
    assert!(err.to_string().contains("not self-contained"), "unexpected rejection: {err}");

    // And tampering with the node list is not free even before the graft: the witness
    // commitment covers every fragment byte.
    let tampered = with_nodes(sidecar, |nodes| {
        nodes.pop();
    });
    partial_stateless::check_sidecar_self_consistency(&tampered)
        .expect_err("the witness commitment did not cover the fragment bytes");
}

#[test]
fn a_cold_trie_degrades_the_trimmed_request_to_the_self_contained_wire() {
    let config = CacheConfig { account_window: 60, storage_window: 30 };
    let blocks = chain();
    let spec = &blocks[0];
    let mut side = Side::cold_at(&config, spec.number - 1);

    let parent = genesis();
    let parent_state_root = parent.state_root();
    let complete = parent.complete_witness();
    let provider = WholeTrieSource::new(parent_state_root, &complete);
    let ctx = TransitionBuildContext::uninstrumented(&provider).with_trimmed_witness();

    let accessed = accessed_state(&parent, spec);
    let (child, post) = apply(&parent, spec);
    let build = build_policy_sidecar(
        &ctx,
        BlockTransitionRef {
            block_number: spec.number,
            block_hash: keccak256(spec.number.to_be_bytes()),
            parent_hash: B256::repeat_byte(0xf0),
            parent_state_root,
            expected_state_root: child.state_root(),
            ancestor_headers: &[],
        },
        &post,
        &accessed,
        &mut side.cache,
        &side.trie,
        &config,
    )
    .expect("the cold-trie sidecar builds");

    assert!(build.build.trimmed.is_none(), "a cold trie has no frontier to trim against");
    assert!(
        matches!(build.sidecar.witness.state, PartialExecutionWitnessState::MptTransitionNodes(_)),
        "a cold-trie build under a trim request must stay byte-identical to v2"
    );
}

/// A fourth block, applied after the whole warmup chain: a storage wipe with a
/// rewrite-after-wipe, plus a first-ever write into an account whose storage trie neither cache
/// has — the fresh-trie path that forces the graft to learn a storage root through the owner's
/// account chain.
fn wipe_and_fresh_trie_block() -> BlockSpec {
    BlockSpec {
        number: 104,
        touched_accounts: vec![address(2), address(6)],
        touched_storage: vec![(address(2), slot(0x21)), (address(6), slot(0x61))],
        account_writes: vec![(address(2), 21, 2_100)],
        storage_writes: vec![(address(2), slot(0x21), 999), (address(6), slot(0x61), 606)],
        wiped_storage: vec![address(2)],
    }
}

#[test]
fn a_storage_wipe_and_a_fresh_storage_trie_graft_identically() {
    let mut warmed = warm_sides_through(&chain(), wipe_and_fresh_trie_block());
    let inputs = last_block_inputs(&warmed);

    let v2 = build_last(&mut warmed, &inputs, false);
    let v3 = build_last(&mut warmed, &inputs, true);

    assert_eq!(v2.build.state_root, v3.build.state_root);
    assert!(v2.build.next_trie_cache.structurally_eq(&v3.build.next_trie_cache));
    assert_eq!(v2.sidecar.miss_manifest, v3.sidecar.miss_manifest);
    assert_eq!(v2.sidecar.next_cache_anchor, v3.sidecar.next_cache_anchor);
    let v2_nodes: BTreeSet<&Bytes> = v2.build.nodes.iter().collect();
    let v3_nodes = trimmed_nodes(&v3.sidecar);
    for node in v3_nodes {
        assert!(v2_nodes.contains(node), "a trimmed node is not part of the full witness");
    }
    assert!(v3_nodes.len() < v2.build.nodes.len(), "the wipe block trimmed nothing");

    let v3_materialized = materialize_sidecar_witness_after_prefilter_with_cache(
        &v3.sidecar,
        Some(&warmed.validator.trie),
    )
    .expect("the wipe-block trimmed witness materializes");
    let MaterializedStateProof::TrimmedTransition(session) = v3_materialized.state_proof else {
        panic!("a trimmed witness materializes into a trimmed session");
    };
    let mut next_trie = warmed.validator.trie.clone();
    let root = try_compute_trustless_state_root_v3_from_hashed(
        session,
        &warmed.validator.trie,
        &mut next_trie,
        inputs.post.clone(),
        &v3.sidecar.miss_manifest,
    )
    .expect("the graft handles the wipe and the fresh storage trie");
    assert_eq!(root, inputs.expected_state_root);

    warmed.validator.cache.on_block_executed(104, &inputs.accessed);
    next_trie.retain_from_value_cache(&warmed.validator.cache);
    assert!(
        next_trie.structurally_eq(&v3.build.next_trie_cache),
        "the validator landed on a different frontier than the builder after the wipe"
    );
}
