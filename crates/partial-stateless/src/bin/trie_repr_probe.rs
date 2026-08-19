//! Compares sparse-trie representations on real recorded witnesses, with an external memory
//! measure.
//!
//! Each representation reports its own `memory_size`, but those definitions move with the
//! representation, so a self-reported comparison between two of them is unfalsifiable. This probe
//! measures both through the same counting allocator instead: it reveals the same recorded
//! full-block witness into each trie type, computes the root, and reads live heap bytes while the
//! trie is the only thing the iteration holds. Clone cost rides along because the per-block
//! transactional snapshot is the largest consumer of whatever the representation costs.
//!
//! Fully offline: the only input is a recorded policy replay dataset.
//!
//! Usage:
//!   trie_repr_probe --dataset <dir> [--blocks <n>]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    error::Error,
    path::PathBuf,
    process,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

use alloy_primitives::{keccak256, map::B256Map, Bytes, B256};
use partial_stateless::load_dataset;
use reth_trie_common::DecodedMultiProofV2;
use reth_trie_sparse::{ExactSparseTrie, ParallelSparseTrie, SparseStateTrie, SparseTrie};

/// Live heap bytes allocated through the global allocator.
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);

/// System allocator wrapped with a live-byte counter, so the two representations are measured by
/// the same external meter rather than by their own `memory_size` definitions.
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

/// One representation's measurements over one revealed witness.
#[derive(Debug, Clone, Copy, Default)]
struct ReprSample {
    /// Live heap bytes the revealed trie held, by the counting allocator.
    live_bytes: usize,
    /// The representation's own `memory_size`, for reference only.
    self_reported_bytes: usize,
    /// Revealing the witness and computing the root.
    build_us: u64,
    /// Cloning the revealed account trie once.
    clone_us: u64,
}

/// Reveals `proof` into a fresh `T`-backed state trie and measures it.
///
/// The live-byte figure is `after - before`, taken while the built trie is the only allocation
/// the call holds, and the trie is dropped before the function returns so consecutive samples
/// start from the same baseline.
fn sample_repr<T>(proof: &DecodedMultiProofV2, expected_root: B256) -> Result<ReprSample, String>
where
    T: SparseTrie + Clone + Default,
{
    let before = live_bytes();
    let build_start = Instant::now();
    let mut trie = SparseStateTrie::<T, T>::default();
    trie.reveal_decoded_multiproof_v2(proof.clone()).map_err(|err| err.to_string())?;
    let root = trie.root().map_err(|err| err.to_string())?;
    drop(trie.take_deferred_drops());
    let build_us = build_start.elapsed().as_micros() as u64;
    if root != expected_root {
        return Err(format!("revealed root {root:?} != recorded parent root {expected_root:?}"));
    }

    // The proof clone made for the reveal has been consumed and freed by here, so the delta is
    // the trie plus its internal buffers and nothing else this function created.
    let live = live_bytes().saturating_sub(before);
    let self_reported = trie.memory_size();

    let clone_start = Instant::now();
    let account_clone = trie.state_trie_ref().cloned();
    let clone_us = clone_start.elapsed().as_micros() as u64;
    drop(account_clone);

    Ok(ReprSample { live_bytes: live, self_reported_bytes: self_reported, build_us, clone_us })
}

type Res<T> = Result<T, Box<dyn Error>>;

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Res<()> {
    let (dataset_dir, blocks) = parse_args()?;
    let dataset = load_dataset(&dataset_dir)?;
    let count = blocks.min(dataset.records.len());
    println!(
        "dataset={} records={} probing={}",
        dataset_dir.display(),
        dataset.records.len(),
        count
    );
    println!(
        "block,nodes,node_bytes,parallel_live,exact_live,exact_live_ratio,\
         parallel_self,exact_self,parallel_build_us,exact_build_us,\
         parallel_clone_us,exact_clone_us"
    );

    let mut totals = [ReprSample::default(); 2];
    let mut total_nodes = 0u64;
    for record in &dataset.records[..count] {
        let body = &record.body;
        let mut witness = B256Map::<Bytes>::default();
        for node in &body.full_transition_nodes {
            witness.insert(keccak256(node), node.clone());
        }
        let proof = DecodedMultiProofV2::from_witness(body.parent_state_root, &witness)
            .map_err(|err| format!("block {}: witness did not decode: {err}", body.block_number))?;
        drop(witness);

        // Parallel first, exact second, each dropped before the other is built, so neither
        // measurement contains the other's allocations.
        let parallel = sample_repr::<ParallelSparseTrie>(&proof, body.parent_state_root)
            .map_err(|err| format!("block {} parallel: {err}", body.block_number))?;
        let exact = sample_repr::<ExactSparseTrie>(&proof, body.parent_state_root)
            .map_err(|err| format!("block {} exact: {err}", body.block_number))?;

        let node_bytes: usize = body.full_transition_nodes.iter().map(|node| node.len()).sum();
        total_nodes += body.full_transition_nodes.len() as u64;
        println!(
            "{},{},{},{},{},{:.4},{},{},{},{},{},{}",
            body.block_number,
            body.full_transition_nodes.len(),
            node_bytes,
            parallel.live_bytes,
            exact.live_bytes,
            exact.live_bytes as f64 / parallel.live_bytes.max(1) as f64,
            parallel.self_reported_bytes,
            exact.self_reported_bytes,
            parallel.build_us,
            exact.build_us,
            parallel.clone_us,
            exact.clone_us,
        );
        for (total, sample) in totals.iter_mut().zip([parallel, exact]) {
            total.live_bytes += sample.live_bytes;
            total.self_reported_bytes += sample.self_reported_bytes;
            total.build_us += sample.build_us;
            total.clone_us += sample.clone_us;
        }
    }

    let [parallel, exact] = totals;
    println!(
        "TOTALS blocks={count} nodes={total_nodes} \
         parallel_live={} arena_live={} arena_live_ratio={:.4} \
         parallel_self={} arena_self={} \
         parallel_build_us={} arena_build_us={} parallel_clone_us={} arena_clone_us={}",
        parallel.live_bytes,
        exact.live_bytes,
        exact.live_bytes as f64 / parallel.live_bytes.max(1) as f64,
        parallel.self_reported_bytes,
        exact.self_reported_bytes,
        parallel.build_us,
        exact.build_us,
        parallel.clone_us,
        exact.clone_us,
    );
    Ok(())
}

fn parse_args() -> Res<(PathBuf, usize)> {
    let mut dataset = None;
    let mut blocks = 25usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--dataset" => dataset = Some(PathBuf::from(value()?)),
            "--blocks" => blocks = value()?.parse()?,
            "-h" | "--help" => {
                println!("usage: trie_repr_probe --dataset <dir> [--blocks <n>]");
                process::exit(0);
            }
            other => return Err(format!("unknown argument `{other}`").into()),
        }
    }
    let dataset = dataset.ok_or("--dataset is required")?;
    Ok((dataset, blocks))
}
