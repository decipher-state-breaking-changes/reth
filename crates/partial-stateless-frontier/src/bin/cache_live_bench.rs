//! Measures a real committed trie cache's live heap bytes under each sparse-trie
//! representation, through one counting allocator.
//!
//! The per-witness probe (`trie_repr_probe`) reveals one block's full witness — a mostly
//! unblinded trie — so it cannot price a representation whose whole point is the *blinded*
//! slots a committed, retention-pruned cache accumulates. This binary replays real policy
//! generation for a window of blocks (the same `generate_block` path the frontier runs),
//! then measures the committed builder-trie generation the only way two representations can
//! be compared: as the live-byte delta the counting allocator observes when the trie is
//! dropped. `memory_size` rides along for reference; it is representation-specific and never
//! the comparison.
//!
//! Both representations run in one process, sequentially, each fully dropped before the next
//! starts, so both see the same allocator, the same corpus, and the same code.
//!
//! Usage:
//!   cache_live_bench --dataset <dir> --arm <a/s> [--blocks <n>]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    error::Error,
    path::PathBuf,
    process,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use partial_stateless::{load_dataset, CacheTrieRepr, PartialTrieNodeCache};
use partial_stateless_frontier::{
    generate_block, ArmKind, ChainCursor, GeneratorRules, PolicyState,
};
use partial_stateless_validator::{SidecarReexecLimits, UntrustedAdmission, ValidatorRules};
use reth_chainspec::{ChainSpec, MAINNET};
use reth_ethereum_consensus::EthBeaconConsensus;
use reth_evm_ethereum::EthEvmConfig;

static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);

/// System allocator wrapped with a live-byte counter — the same meter for both representations.
struct CountingAllocator;

// SAFETY: delegates every operation to `System` unchanged; only the accounting is added.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            LIVE_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            LIVE_BYTES.fetch_add(new_size, Ordering::Relaxed);
            LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        }
        new_ptr
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn live_bytes() -> usize {
    LIVE_BYTES.load(Ordering::Relaxed)
}

type Res<T> = Result<T, Box<dyn Error>>;

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Res<()> {
    let (dataset_dir, arm, blocks) = parse_args()?;
    let dataset = load_dataset(&dataset_dir)?;
    let count = blocks.min(dataset.records.len());
    println!("dataset={} arm={} blocks={}", dataset_dir.display(), arm.label(), count);

    let chain_spec: Arc<ChainSpec> = match dataset.manifest.chain.as_str() {
        "mainnet" => MAINNET.clone(),
        other => return Err(format!("unsupported chain {other:?}").into()),
    };
    let consensus = EthBeaconConsensus::new(chain_spec.clone());
    let evm_config = EthEvmConfig::new(chain_spec.clone());
    let admission = UntrustedAdmission::new(chain_spec.as_ref(), &consensus);
    let limits = SidecarReexecLimits::default();
    let rules = GeneratorRules {
        validator: ValidatorRules::new(&evm_config, &consensus),
        admission: &admission,
        limits: &limits,
        trie_diagnostics: false,
        witness_v3: false,
    };

    println!(
        "repr,committed_blocks,builder_trie_live_bytes,builder_trie_memory_size,\
         validator_trie_live_bytes,account_branches,account_blinded_slots,\
         storage_branches,storage_blinded_slots"
    );

    for repr in [CacheTrieRepr::Parallel, CacheTrieRepr::Exact] {
        let replayed = &dataset.records[..count];
        let first = replayed.first().ok_or("empty dataset")?;
        let mut cursor = ChainCursor::enter(first)?;
        let mut policies = vec![PolicyState::cold_at_with_repr(
            arm,
            first.body.block_number.saturating_sub(1),
            repr,
        )];

        for (index, record) in replayed.iter().enumerate() {
            generate_block(&rules, record, &mut policies, &mut cursor, index, false)
                .map_err(|err| format!("{} block {index}: {err:#}", repr.label()))?;
        }

        let state = &mut policies[0];
        let census = state.builder_trie.branch_slot_census();
        let self_reported = state.builder_trie.estimated_memory_bytes();

        // The committed generation's live bytes are what the allocator loses when it drops —
        // fat pointers, size-class rounding, and every internal buffer included, which is
        // exactly what the gross projection left out.
        let before = live_bytes();
        let builder = std::mem::replace(&mut state.builder_trie, PartialTrieNodeCache::new());
        drop(builder);
        let builder_live = before.saturating_sub(live_bytes());

        let before = live_bytes();
        let validator = std::mem::replace(&mut state.validator_trie, PartialTrieNodeCache::new());
        drop(validator);
        let validator_live = before.saturating_sub(live_bytes());

        println!(
            "{},{},{},{},{},{},{},{},{}",
            repr.label(),
            count,
            builder_live,
            self_reported,
            validator_live,
            census.account.branches,
            census.account.blinded_slots,
            census.storage.branches,
            census.storage.blinded_slots,
        );

        drop(policies);
        drop(cursor);
    }

    Ok(())
}

fn parse_args() -> Res<(PathBuf, ArmKind, usize)> {
    let mut dataset = None;
    let mut arm = None;
    let mut blocks = 121usize;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--dataset" => dataset = Some(PathBuf::from(value()?)),
            "--arm" => {
                arm = Some(
                    value()?
                        .parse::<partial_stateless_frontier::PolicySpec>()
                        .map(ArmKind::Policy)
                        .map_err(|err| format!("{err}"))?,
                )
            }
            "--blocks" => blocks = value()?.parse()?,
            other => return Err(format!("unknown flag {other}").into()),
        }
    }
    Ok((
        dataset.ok_or("--dataset is required")?,
        arm.ok_or("--arm is required (e.g. --arm 90/60)")?,
        blocks,
    ))
}
