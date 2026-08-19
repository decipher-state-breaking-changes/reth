//! The gate the whole offline method rests on: a sidecar generated from a recorded witness is the
//! **same sidecar**, byte for byte, as the one a database-backed builder produces for the same
//! block under the same policy.
//!
//! Two claims are being checked, and they are different claims.
//!
//! The first is **sufficiency**: a policy-neutral full witness, recorded once against a cold cache
//! and an empty trie, contains every parent-state node any warm-cache policy's transition later
//! asks for. The argument for that is a superset argument — every target a warm cache lets a policy
//! skip is a target the cold build already proved — and an argument is not a check. Here it is
//! checked against a real trie, with a *warm* cache and a *warm* trie generation, which is the only
//! configuration where the two builds request different things and the superset claim can fail.
//!
//! The second is **identity**: the two builds do not merely agree on the state root, they produce
//! the same serialized bytes. That is stronger than it needs to be on purpose. A comparison between
//! two cache policies reports sizes, node sets, miss manifests, and commitments, and every one of
//! those would be silently wrong if the offline builder differed from the live one in ordering,
//! deduplication, or which nodes it kept.
//!
//! The stand-in for the node's state database is a source that answers any target from the
//! *complete* parent trie, which is what a provider does. The offline side is the real
//! `RecordedFullWitnessSource` over the recorded node set alone.

mod world;
use world::*;

use alloy_primitives::{keccak256, B256};
use partial_stateless::{
    build_full_witness_sidecar, build_policy_sidecar, sidecar_semantic_digest, BlockTransitionRef,
    CacheConfig, PartialTrieNodeCache, TransitionBuildContext,
};
use partial_stateless_frontier::RecordedFullWitnessSource;

#[test]
fn an_offline_sidecar_is_byte_identical_to_the_one_a_database_backed_builder_produces() {
    let config = CacheConfig { account_window: 60, storage_window: 30 };
    let blocks = chain();
    let first = blocks[0].number;

    let mut live = Side::cold_at(&config, first - 1);
    let mut offline = Side::cold_at(&config, first - 1);

    let mut parent = genesis();
    let mut parent_hash = B256::repeat_byte(0xf0);
    let mut exercised = Exercised::default();

    for spec in &blocks {
        let parent_state_root = parent.state_root();
        let complete = parent.complete_witness();
        let provider = WholeTrieSource::new(parent_state_root, &complete);
        let provider_ctx = TransitionBuildContext::uninstrumented(&provider);

        let accessed = accessed_state(&parent, spec);
        let (child, post) = apply(&parent, spec);
        let expected_state_root = child.state_root();
        let block_hash = keccak256(spec.number.to_be_bytes());
        let block_ref = BlockTransitionRef {
            block_number: spec.number,
            block_hash,
            parent_hash,
            parent_state_root,
            expected_state_root,
            ancestor_headers: &[],
        };

        // The capture: one policy-neutral full witness, proved against the parent state, exactly
        // as a capturing node would record it.
        let recorded = build_full_witness_sidecar(
            &provider_ctx,
            parent_state_root,
            expected_state_root,
            parent_hash,
            block_hash,
            spec.number,
            &post,
            &accessed,
            &[],
            &config,
        )
        .expect("the full witness builds against the whole trie")
        .into_nodes()
        .expect("the full witness is a transition-node witness");

        // The live build: the node's database answers every proof request.
        let live_build = build_policy_sidecar(
            &provider_ctx,
            block_ref,
            &post,
            &accessed,
            &mut live.cache,
            &live.trie,
            &config,
        )
        .expect("the live policy sidecar builds");

        // The offline build: nothing but the recorded node set answers every proof request.
        let source = RecordedFullWitnessSource::new(parent_state_root, &recorded)
            .expect("the recorded witness decodes");
        let offline_ctx = TransitionBuildContext::uninstrumented(&source);
        let offline_build = build_policy_sidecar(
            &offline_ctx,
            block_ref,
            &post,
            &accessed,
            &mut offline.cache,
            &offline.trie,
            &config,
        )
        .unwrap_or_else(|err| {
            panic!(
                "block {} could not be built from the recorded witness alone: {err:#}",
                spec.number
            )
        });

        // Compared through the semantic digest, which is what a cross-host comparison uses and
        // which normalizes exactly one class of field: the wall-clock and resource measurements a
        // sidecar carries about the machine that built it. Everything a policy comparison reports
        // — identity, both anchors, the miss manifest, the witness body, the witness commitment,
        // and every size — is inside it. Comparing raw serialized bytes here would be comparing
        // the two hosts' clocks along with their sidecars.
        let live_sidecar = live_build.sidecar;
        let offline_sidecar = offline_build.sidecar;
        assert_eq!(
            sidecar_semantic_digest(&live_sidecar).unwrap(),
            sidecar_semantic_digest(&offline_sidecar).unwrap(),
            "block {} produced a different sidecar offline than live",
            spec.number
        );

        // And the digest is not merely equal on both sides — it is the *only* thing that differs
        // between the raw encodings, so the normalization is not hiding a real difference.
        let mut live_bytes = live_sidecar.clone();
        let mut offline_bytes = offline_sidecar.clone();
        live_bytes.stats.computation_time_ms = None;
        offline_bytes.stats.computation_time_ms = None;
        assert_eq!(
            bincode::serialize(&live_bytes).unwrap(),
            bincode::serialize(&offline_bytes).unwrap(),
            "block {} produced different bytes offline than live",
            spec.number
        );
        assert_eq!(live_sidecar.witness_commitment, offline_sidecar.witness_commitment);
        assert_eq!(
            live_sidecar.next_cache_anchor.cache_root,
            offline_sidecar.next_cache_anchor.cache_root
        );

        // What this block did or did not exercise, so the test cannot quietly stop testing the
        // case it exists for. A fixture edit that made every block build from an empty trie, or
        // that made the policy witness equal to the full one, would pass every assertion above
        // while proving nothing about subset selection.
        let policy_nodes =
            live_sidecar.stats.account_proof_nodes + live_sidecar.stats.storage_proof_nodes;
        if live.trie.shape_metrics().account_revealed_nodes > 0 {
            exercised.warm_trie = true;
            if policy_nodes < recorded.len() {
                exercised.strict_subset = true;
            }
        }
        if live_build.build.structural_rounds > 0 {
            exercised.structural_round = true;
        }

        live.trie = live_build.build.next_trie_cache;
        offline.trie = offline_build.build.next_trie_cache;
        parent = child;
        parent_hash = block_hash;
    }

    assert!(exercised.warm_trie, "every block built from an empty trie");
    assert!(
        exercised.strict_subset,
        "no block's policy witness was smaller than the recorded full one, so subset selection \
         was never exercised"
    );
    assert!(
        exercised.structural_round,
        "no block's transition took a structural round, so the recorded witness was never asked \
         for a node the initial target set did not name — which is the case the superset argument \
         is really about"
    );
}

/// Which parts of the claim this run actually reached.
#[derive(Debug, Default)]
struct Exercised {
    /// A block built against a trie generation that already held revealed nodes.
    warm_trie: bool,
    /// A policy's witness was strictly smaller than the recorded full one.
    strict_subset: bool,
    /// A transition discovered it needed a node its initial targets did not name, and the
    /// recorded witness served it.
    structural_round: bool,
}

/// A witness with a node removed is refused, rather than silently producing a different sidecar.
///
/// The failure mode this rules out is the one that would be hardest to notice: an offline generator
/// that quietly built a smaller witness because the recording was short, and reported the smaller
/// size as the policy's.
#[test]
fn a_truncated_recording_fails_the_build_rather_than_shrinking_the_sidecar() {
    let config = CacheConfig { account_window: 60, storage_window: 30 };
    let spec = &chain()[0];
    let parent = genesis();
    let parent_state_root = parent.state_root();
    let complete = parent.complete_witness();
    let provider = WholeTrieSource::new(parent_state_root, &complete);
    let provider_ctx = TransitionBuildContext::uninstrumented(&provider);

    let accessed = accessed_state(&parent, spec);
    let (child, post) = apply(&parent, spec);
    let expected_state_root = child.state_root();
    let block_hash = keccak256(spec.number.to_be_bytes());
    let parent_hash = B256::repeat_byte(0xf0);

    let recorded = build_full_witness_sidecar(
        &provider_ctx,
        parent_state_root,
        expected_state_root,
        parent_hash,
        block_hash,
        spec.number,
        &post,
        &accessed,
        &[],
        &config,
    )
    .expect("the full witness builds")
    .into_nodes()
    .expect("the full witness is a transition-node witness");

    let mut truncated = recorded;
    assert!(truncated.len() > 2, "the fixture must produce a multi-node witness");
    truncated.pop();

    let source = RecordedFullWitnessSource::new(parent_state_root, &truncated);
    let mut cache = config.new_cache_at(spec.number - 1);
    let trie = PartialTrieNodeCache::new();
    let refused = source.and_then(|source| {
        let ctx = TransitionBuildContext::uninstrumented(&source);
        build_policy_sidecar(
            &ctx,
            BlockTransitionRef {
                block_number: spec.number,
                block_hash,
                parent_hash,
                parent_state_root,
                expected_state_root,
                ancestor_headers: &[],
            },
            &post,
            &accessed,
            &mut cache,
            &trie,
            &config,
        )
    });

    assert!(
        refused.is_err(),
        "a truncated recording produced a sidecar instead of an error; an incomplete corpus would \
         be reported as a smaller policy footprint"
    );
}

/// The digest two hosts compare must not depend on how fast either of them was.
///
/// This is the property the byte comparison above cannot state for itself: it nulls the timing
/// field by hand, which proves the field differs but not that the digest ignores it.
#[test]
fn the_semantic_digest_ignores_how_long_the_build_took() {
    let config = CacheConfig { account_window: 60, storage_window: 30 };
    let spec = &chain()[0];
    let parent = genesis();
    let parent_state_root = parent.state_root();
    let complete = parent.complete_witness();
    let provider = WholeTrieSource::new(parent_state_root, &complete);
    let ctx = TransitionBuildContext::uninstrumented(&provider);

    let accessed = accessed_state(&parent, spec);
    let (child, post) = apply(&parent, spec);
    let block_hash = keccak256(spec.number.to_be_bytes());
    let parent_hash = B256::repeat_byte(0xf0);
    let mut cache = config.new_cache_at(spec.number - 1);
    let build = build_policy_sidecar(
        &ctx,
        BlockTransitionRef {
            block_number: spec.number,
            block_hash,
            parent_hash,
            parent_state_root,
            expected_state_root: child.state_root(),
            ancestor_headers: &[],
        },
        &post,
        &accessed,
        &mut cache,
        &PartialTrieNodeCache::new(),
        &config,
    )
    .expect("the sidecar builds");

    let mut slower = build.sidecar.clone();
    slower.stats.computation_time_ms =
        Some(build.sidecar.stats.computation_time_ms.unwrap_or(0) + 9_999);
    slower.stats.cpu_time_ms = Some(1_234);
    slower.stats.major_page_faults = Some(56);
    slower.stats.minor_page_faults = Some(78);

    assert_ne!(
        bincode::serialize(&build.sidecar).unwrap(),
        bincode::serialize(&slower).unwrap(),
        "the fixture must actually change the encoding, or this proves nothing"
    );
    assert_eq!(
        sidecar_semantic_digest(&build.sidecar).unwrap(),
        sidecar_semantic_digest(&slower).unwrap(),
        "the digest moved when only the host's own measurements did"
    );

    // A real difference still moves it.
    let mut different = build.sidecar.clone();
    different.stats.account_proof_bytes += 1;
    assert_ne!(
        sidecar_semantic_digest(&build.sidecar).unwrap(),
        sidecar_semantic_digest(&different).unwrap(),
        "the digest ignored a witness that is a different size"
    );
}
