//! Offline profile of where the next-cache-anchor phase spends its time.
//!
//! The anchor is the largest validator cost outside execution and was a single opaque timer, so
//! the three ways to make it cheaper — reusing the leaf preimage buffer, memoizing leaf digests,
//! and keeping the keys ordered — could not be sized against each other without a live run. This
//! builds a cache at the composition a production run actually carries and reports the split,
//! which answers "is the per-leaf cost hashing or allocation?" in seconds instead of an hour of
//! mainnet blocks.
//!
//! Ignored by default because it allocates a production-sized cache, and it refuses to run
//! without the keccak features the production binary enables — building this package alone
//! selects only its own dependency graph, which is how every benchmark before 2026-08-06 came to
//! measure a keccak the node never runs. Run it with:
//!
//! ```text
//! cargo test -p partial-stateless --release --features asm-keccak,keccak-cache-global \
//!   --test anchor_cost_profile -- --ignored --nocapture
//! ```
//!
//! **What this cannot tell you.** The maps here are built in one pass, so their memory layout is
//! more compact than a cache that has lived through sixty blocks of insertion and eviction. That
//! biases the collect-and-sort term *down* — it is the term most sensitive to layout — so treat
//! its share as a lower bound and the leaf-hash share as an upper bound. The live schema-V5 split
//! is the measurement; this is the thing that says which change is worth writing first.

use alloy_primitives::{keccak256, Address, Bytes, Keccak256, B256, U256};
use partial_stateless::{
    accessed_state::BlockAccessedState,
    policy::{AccountData, LastNBlocksPolicy},
    CacheRootTimings, NetworkStateCache,
};
use std::time::Instant;

/// Composition of the 2026-08-07 production-profile run, averaged over its last 250 blocks.
const ACCOUNTS: u32 = 30_100;
const STORAGE_SLOTS: u32 = 32_900;
const CODES: u32 = 2_300;
/// Mean cached bytecode size, from that run's ~23 MiB of code over ~2.3k entries.
const CODE_BYTES: usize = 10_000;

/// Refuse to print a number that describes a keccak the production node does not use.
fn require_production_keccak() {
    #[cfg(not(all(feature = "asm-keccak", feature = "keccak-cache-global")))]
    panic!(
        "build with --features asm-keccak,keccak-cache-global; without them this profile \
         measures a keccak implementation `bin/reth` and `partial-stateless-exex` do not run, \
         and its split between hashing and everything else is wrong"
    );
}

fn address_at(index: u32) -> Address {
    Address::from_slice(&keccak256(index.to_be_bytes())[..20])
}

fn production_sized_cache() -> NetworkStateCache {
    let mut cache = NetworkStateCache::new(
        Box::new(LastNBlocksPolicy::new(60)),
        Box::new(LastNBlocksPolicy::new(30)),
    );

    let mut accessed = BlockAccessedState::default();
    for index in 0..CODES {
        // Vary the leading bytes rather than fill with `index as u8`, which would collapse to 256
        // distinct bytecodes and understate the code namespace.
        let mut body = vec![0xefu8; CODE_BYTES];
        body[..4].copy_from_slice(&index.to_be_bytes());
        let bytes = Bytes::from(body);
        accessed.codes.insert(keccak256(&bytes), bytes);
    }
    let code_hashes: Vec<B256> = accessed.codes.keys().copied().collect();
    for index in 0..ACCOUNTS {
        accessed.accounts.insert(
            address_at(index),
            AccountData {
                nonce: u64::from(index),
                balance: U256::from(index),
                // Roughly the observed share of cached accounts that are contracts.
                code_hash: (index % 13 == 0)
                    .then(|| code_hashes[index as usize % code_hashes.len()]),
            },
        );
    }
    // Slots are spread over the accounts rather than concentrated, which is what makes the
    // storage sort key `(Address, B256)` comparison do real work.
    for index in 0..STORAGE_SLOTS {
        accessed.storage.insert(
            (address_at(index % ACCOUNTS), keccak256(index.to_be_bytes())),
            U256::from(index),
        );
    }

    cache.on_block_executed(1, &accessed);
    cache
}

fn row(label: &str, us: u64, total: u64, entries: u64) -> String {
    let share = if total == 0 { 0.0 } else { 100.0 * us as f64 / total as f64 };
    let per_entry = if entries == 0 { 0.0 } else { us as f64 / entries as f64 };
    format!(
        "| {label:38} | {:9.2} ms | {share:5.1}% | {per_entry:8.3} µs/entry |",
        us as f64 / 1000.0
    )
}

#[test]
#[ignore = "production-sized profile; run explicitly in release mode"]
fn profile_next_cache_anchor_cost() {
    require_production_keccak();
    let cache = production_sized_cache();

    // The memo answers a second call for free, so each round advances the cache by an empty
    // block first. That is exactly how the validator reaches this phase — the cache update
    // immediately before the anchor invalidates the memo — and it keeps the composition fixed,
    // because nothing is added and the windows are far from evicting anything yet.
    let mut cache = cache;
    let mut rounds: Vec<CacheRootTimings> = Vec::new();
    for round in 0..5u64 {
        cache.on_block_executed(2 + round, &BlockAccessedState::default());
        let (_, timings) = cache.cache_root_timed();
        assert!(!timings.memo_hit, "profiling a memo hit measures nothing");
        rounds.push(timings);
    }

    let mean = |pick: fn(&CacheRootTimings) -> u64| -> u64 {
        rounds.iter().map(pick).sum::<u64>() / rounds.len() as u64
    };
    let accounts = rounds[0].accounts;
    let storage = rounds[0].storage;
    let codes = rounds[0].codes;
    let leaves = rounds[0].leaves();
    let total = mean(CacheRootTimings::total_us);

    println!("\n## Next cache anchor cost profile (offline, {} rounds)\n", rounds.len());
    println!(
        "Composition: {accounts} accounts, {storage} storage, {codes} codes ({leaves} leaves)\n"
    );
    println!(
        "| Component                               |      Mean | Share |          Per entry |"
    );
    println!("| --- | ---: | ---: | ---: |");
    println!(
        "{}",
        row("Account collect + sort", mean(|t| t.account_collect_sort_us), total, accounts)
    );
    println!(
        "{}",
        row("Storage collect + sort", mean(|t| t.storage_collect_sort_us), total, storage)
    );
    println!("{}", row("Code collect + sort", mean(|t| t.code_collect_sort_us), total, codes));
    println!(
        "{}",
        row("Account leaf preimage + hash", mean(|t| t.account_leaf_hash_us), total, accounts)
    );
    println!(
        "{}",
        row("Storage leaf preimage + hash", mean(|t| t.storage_leaf_hash_us), total, storage)
    );
    println!(
        "{}",
        row("Code leaf finish (memoized sponge)", mean(|t| t.code_leaf_hash_us), total, codes)
    );
    println!(
        "{}",
        row("Account namespace hash", mean(|t| t.account_namespace_us), total, accounts)
    );
    println!("{}", row("Storage namespace hash", mean(|t| t.storage_namespace_us), total, storage));
    println!("{}", row("Code namespace hash", mean(|t| t.code_namespace_us), total, codes));
    println!("{}", row("Final root hash", mean(|t| t.root_us), total, 1));
    println!("{}", row("**Total**", total, total, leaves));

    println!("\nGrouped by the change that would remove it:\n");
    println!(
        "{}",
        row(
            "collect + sort  (ordered index)",
            mean(CacheRootTimings::collect_sort_us),
            total,
            leaves
        )
    );
    println!(
        "{}",
        row("leaf hash       (digest memo)", mean(CacheRootTimings::leaf_hash_us), total, leaves)
    );
    println!(
        "{}",
        row(
            "namespace hash  (irreducible)",
            mean(CacheRootTimings::namespace_hash_us),
            total,
            leaves
        )
    );
    println!();

    assert!(total > 0, "profile measured nothing");
}

/// Split the leaf-hash term into the allocator's share and keccak's share.
///
/// The leaf term is the largest of the three above, but it covers two very different changes:
/// dropping the per-leaf `Vec` costs nothing in correctness, while memoizing the digest has to be
/// right across rollback and reorg. This times the three preimage strategies over the exact byte
/// shapes of an account and a storage leaf, so the split decides whether the cheap change is worth
/// landing on its own. All three produce the same digest, which the test asserts.
#[test]
#[ignore = "production-sized profile; run explicitly in release mode"]
fn profile_leaf_preimage_strategies() {
    require_production_keccak();
    let leaves = (ACCOUNTS + STORAGE_SLOTS) as usize;
    let address = address_at(7);
    let slot = keccak256(11u32.to_be_bytes());
    let balance = U256::from(1_234_567_890u64);
    let code_hash = keccak256(b"code");

    // Exactly what `hash_account` builds today: a fresh `Vec` grown field by field.
    let fresh_vec = || {
        let mut digest = B256::ZERO;
        for index in 0..leaves {
            let mut preimage = Vec::new();
            preimage.extend_from_slice(b"NetworkStateCacheLeaf/v1/account");
            preimage.extend_from_slice(address.as_slice());
            preimage.extend_from_slice(&(index as u64).to_be_bytes());
            preimage.extend_from_slice(&balance.to_be_bytes::<32>());
            preimage.extend_from_slice(b"code_hash");
            preimage.extend_from_slice(code_hash.as_slice());
            preimage.extend_from_slice(&slot.as_slice()[..8]);
            digest = keccak256(&preimage);
        }
        digest
    };

    // One buffer for the whole namespace, cleared per leaf. Same bytes, one allocation.
    let reused_vec = || {
        let mut preimage = Vec::with_capacity(160);
        let mut digest = B256::ZERO;
        for index in 0..leaves {
            preimage.clear();
            preimage.extend_from_slice(b"NetworkStateCacheLeaf/v1/account");
            preimage.extend_from_slice(address.as_slice());
            preimage.extend_from_slice(&(index as u64).to_be_bytes());
            preimage.extend_from_slice(&balance.to_be_bytes::<32>());
            preimage.extend_from_slice(b"code_hash");
            preimage.extend_from_slice(code_hash.as_slice());
            preimage.extend_from_slice(&slot.as_slice()[..8]);
            digest = keccak256(&preimage);
        }
        digest
    };

    // No buffer at all: the same bytes absorbed straight into the sponge.
    let streamed = || {
        let mut digest = B256::ZERO;
        for index in 0..leaves {
            let mut hasher = Keccak256::new();
            hasher.update(b"NetworkStateCacheLeaf/v1/account");
            hasher.update(address.as_slice());
            hasher.update((index as u64).to_be_bytes());
            hasher.update(balance.to_be_bytes::<32>());
            hasher.update(b"code_hash");
            hasher.update(code_hash.as_slice());
            hasher.update(&slot.as_slice()[..8]);
            digest = hasher.finalize();
        }
        digest
    };

    assert_eq!(fresh_vec(), reused_vec(), "reusing the buffer changed the digest");
    assert_eq!(fresh_vec(), streamed(), "streaming the preimage changed the digest");

    let time = |run: &dyn Fn() -> B256| -> f64 {
        let start = Instant::now();
        std::hint::black_box(run());
        start.elapsed().as_micros() as f64 / 1000.0
    };
    let fresh = time(&fresh_vec);
    let reused = time(&reused_vec);
    let stream = time(&streamed);

    println!("\n## Leaf preimage strategy, {leaves} leaves\n");
    println!("| Strategy | Mean | Against today |");
    println!("| --- | ---: | ---: |");
    println!("| `Vec::new()` per leaf (today) | {fresh:.2} ms | — |");
    println!("| One reused buffer | {reused:.2} ms | {:+.1}% |", 100.0 * (reused - fresh) / fresh);
    println!(
        "| Streamed into the sponge | {stream:.2} ms | {:+.1}% |",
        100.0 * (stream - fresh) / fresh
    );
    println!(
        "\nAllocator share of the leaf term: **{:.1}%**; the rest is keccak and field copies.\n",
        100.0 * (fresh - stream) / fresh
    );
}
