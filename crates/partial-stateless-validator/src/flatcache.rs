//! `flatcache-experiment` — does a **flat leaf cache** (state values keyed by
//! trie path) cut the witness on top of v2?
//!
//! v2 caches trie *nodes* by hash, so a repeated read still needs every node on
//! the root→leaf path. A flat leaf cache stores the *value* keyed by the leaf's
//! position, so a repeated read of an unchanged leaf needs **no** path nodes —
//! but state-root recomputation still needs every *written* path.
//!
//! For each block we walk the witness trie, enumerate every accessed leaf
//! (account + storage slot) with its root→leaf node-hash path and value, and
//! classify it read/written by comparing its value to the next block (these are
//! consecutive blocks). Then we replay two node sets through the **same** v2
//! cache:
//!   - `v2`      — every accessed trie node (read + write paths)
//!   - `v2+flat` — only nodes on written paths or flat-cache-missed read paths
//! Code accesses are identical in both. Lower `v2+flat` witness = flat helps.
//!
//! The flat leaf cache here is **unbounded and free** — this measures the
//! *ceiling* benefit; its own memory cost (reported) would be charged in a real
//! design.
//!
//! Usage: flatcache-experiment --dir replay_data --budgets-mb 10,20,50,100 --top-n 3

use alloy_primitives::{keccak256, B256};
use alloy_rlp::Decodable;
use alloy_trie::nodes::{RlpNode, TrieNode};
use partial_stateless_validator::{
    byte_cache::{self, Item, ItemKind, V2Cache, V2Config},
    depth::{compute_depths, DEEP},
    witness::{IndexedWitness, WitnessJson},
};
use reth_trie_common::TrieAccount;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

/// One accessed leaf: its stable cross-block identity, current value, and the
/// hashes of the trie nodes on its root→leaf path.
struct LeafRec {
    key: Vec<u8>,
    value: Vec<u8>,
    path: Vec<B256>,
}

fn main() -> eyre::Result<()> {
    let args = Args::parse()?;

    // Load consecutive blocks; keep raw witness (ordered state/codes) + walk data.
    let mut files: Vec<(u64, PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(&args.dir)? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else { continue };
        let Some(num) = name.strip_prefix("witness_").and_then(|s| s.strip_suffix(".json")) else {
            continue;
        };
        files.push((num.parse()?, path));
    }
    files.sort_by_key(|(n, _)| *n);
    if files.len() < 2 {
        eyre::bail!("need ≥2 consecutive witness_*.json blocks in {}", args.dir.display());
    }

    // Per-block: all state nodes (id,size,depth), code items, and enumerated leaves.
    struct Block {
        number: u64,
        nodes: Vec<Item>,          // every state node (the v2 node set)
        codes: Vec<Item>,
        leaves: Vec<LeafRec>,
        values: HashMap<Vec<u8>, Vec<u8>>, // key -> value, for write detection
    }
    let mut blocks: Vec<Block> = Vec::with_capacity(files.len());
    for (number, path) in &files {
        let raw = std::fs::read(path)?;
        let wj: WitnessJson = serde_json::from_slice(&raw)?;
        let indexed = IndexedWitness::from_witness(wj.clone())?;
        let depths = compute_depths(&indexed);

        let mut nodes = Vec::with_capacity(wj.state.len());
        for n in &wj.state {
            let id = keccak256(n);
            nodes.push(Item {
                kind: ItemKind::Node,
                id,
                size: n.len() as u64,
                depth: depths.get(&id).copied().unwrap_or(DEEP),
            });
        }
        let codes = wj
            .codes
            .iter()
            .map(|c| Item { kind: ItemKind::Code, id: keccak256(c), size: c.len() as u64, depth: 0 })
            .collect();

        let leaves = enumerate_leaves(&indexed);
        let mut values = HashMap::with_capacity(leaves.len());
        for l in &leaves {
            values.insert(l.key.clone(), l.value.clone());
        }
        blocks.push(Block { number: *number, nodes, codes, leaves, values });
    }

    // node size+depth lookup (for rebuilding items from a hash subset).
    let mut meta: HashMap<B256, (u64, u8)> = HashMap::new();
    for b in &blocks {
        for it in &b.nodes {
            meta.insert(it.id, (it.size, it.depth));
        }
    }

    // Build the two traces. v2+flat maintains a flat leaf cache (post-state values).
    let mut trace_v2: Vec<Vec<Item>> = Vec::with_capacity(blocks.len());
    let mut trace_flat: Vec<Vec<Item>> = Vec::with_capacity(blocks.len());
    let mut leaf_cache: HashMap<Vec<u8>, Vec<u8>> = HashMap::new();
    let mut droppable_bytes: u64 = 0;
    let mut peak_cache_entries = 0usize;
    let mut peak_cache_bytes: u64 = 0;

    for i in 0..blocks.len() {
        // v2 trace = all nodes + code.
        let mut v2_items = blocks[i].nodes.clone();
        v2_items.extend(blocks[i].codes.iter().cloned());
        trace_v2.push(v2_items);

        // Next block (for write detection) only if contiguous.
        let next_values = blocks
            .get(i + 1)
            .filter(|nb| nb.number == blocks[i].number + 1)
            .map(|nb| &nb.values);

        let all_hashes: HashSet<B256> = blocks[i].nodes.iter().map(|it| it.id).collect();
        let mut kept: HashSet<B256> = HashSet::new();
        let mut dropped: HashSet<B256> = HashSet::new();

        for l in &blocks[i].leaves {
            let written = match next_values {
                Some(nv) => nv.get(&l.key).is_some_and(|v| v != &l.value),
                None => false, // last/non-contiguous block: assume read
            };
            // read hit only when not written and the cached (post-)value matches.
            let read_hit = !written && leaf_cache.get(&l.key).is_some_and(|v| v == &l.value);
            let keep = !read_hit; // written or cold/stale read -> path needed
            for h in &l.path {
                if keep {
                    kept.insert(*h);
                } else {
                    dropped.insert(*h);
                }
            }
        }

        // A node is droppable iff it is ONLY on cached-read paths.
        let mut needed: HashSet<B256> = all_hashes.clone();
        for h in &dropped {
            if !kept.contains(h) {
                needed.remove(h);
                if let Some((size, _)) = meta.get(h) {
                    droppable_bytes += size;
                }
            }
        }

        let mut flat_items: Vec<Item> = blocks[i]
            .nodes
            .iter()
            .filter(|it| needed.contains(&it.id))
            .cloned()
            .collect();
        flat_items.extend(blocks[i].codes.iter().cloned());
        trace_flat.push(flat_items);

        // Update the flat cache with end-of-block (post-state) values.
        for l in &blocks[i].leaves {
            let post = next_values.and_then(|nv| nv.get(&l.key)).unwrap_or(&l.value);
            let entry = leaf_cache.entry(l.key.clone()).or_default();
            *entry = post.clone();
        }
        peak_cache_entries = peak_cache_entries.max(leaf_cache.len());
        let cb: u64 = leaf_cache.iter().map(|(k, v)| (k.len() + v.len()) as u64).sum();
        peak_cache_bytes = peak_cache_bytes.max(cb);
    }

    let total: u64 = trace_v2.iter().flatten().map(|it| it.size).sum();
    println!(
        "# flat-leaf experiment: {} blocks ({}..{})  uncached witness {:.3} GB",
        blocks.len(),
        blocks.first().unwrap().number,
        blocks.last().unwrap().number,
        gb(total),
    );
    println!(
        "# max droppable (nodes only on cached-read paths, pre-trie-cache) = {:.3} GB | flat-cache peak = {} entries / {:.1} MB",
        gb(droppable_bytes),
        peak_cache_entries,
        peak_cache_bytes as f64 / 1e6,
    );
    println!("{:<10} {:>9} {:>11} {:>11} {:>9}", "topN", "budget", "v2", "v2+flat", "delta");

    for &budget in &args.budgets {
        let mut v2 = V2Cache::new(V2Config::new(budget, args.top_n));
        let w_v2 = byte_cache::run(&mut v2, &trace_v2).bytes_served;
        let mut vf = V2Cache::new(V2Config::new(budget, args.top_n));
        let w_flat = byte_cache::run(&mut vf, &trace_flat).bytes_served;
        let delta = if w_v2 > 0 {
            format!("{:+.1}%", 100.0 * (w_flat as f64 - w_v2 as f64) / w_v2 as f64)
        } else {
            String::new()
        };
        println!(
            "{:<10} {:>7}MB {:>9.3}GB {:>9.3}GB {:>9}",
            args.top_n,
            budget / 1_000_000,
            gb(w_v2),
            gb(w_flat),
            delta
        );
    }
    Ok(())
}

/// Walk the witness trie from the pre-state root, returning every accessed leaf
/// (account + storage) with its stable key, value, and root→leaf node-hash path.
fn enumerate_leaves(w: &IndexedWitness) -> Vec<LeafRec> {
    let mut out = Vec::new();
    let root = w.pre_state_root;
    if w.nodes.contains_key(&root) {
        let mut path = vec![root];
        let mut nib = Vec::new();
        descend_hash(w, root, &mut path, &mut nib, false, &[], &mut out);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn descend_hash(
    w: &IndexedWitness,
    hash: B256,
    path: &mut Vec<B256>,
    nib: &mut Vec<u8>,
    in_storage: bool,
    acct: &[u8],
    out: &mut Vec<LeafRec>,
) {
    let Some(bytes) = w.nodes.get(&hash).cloned() else { return };
    let Ok(node) = TrieNode::decode(&mut bytes.as_ref()) else { return };
    descend_node(w, node, path, nib, in_storage, acct, out);
}

fn descend_node(
    w: &IndexedWitness,
    node: TrieNode,
    path: &mut Vec<B256>,
    nib: &mut Vec<u8>,
    in_storage: bool,
    acct: &[u8],
    out: &mut Vec<LeafRec>,
) {
    match node {
        TrieNode::Leaf(leaf) => {
            let save = nib.len();
            nib.extend(leaf.key.to_vec());
            let key = compose_key(in_storage, acct, nib);
            out.push(LeafRec { key, value: leaf.value.clone(), path: path.clone() });
            // Account leaf -> recurse into its storage trie (a distinct set of leaves).
            if !in_storage {
                if let Ok(acct_node) = TrieAccount::decode(&mut leaf.value.as_slice()) {
                    let sroot = acct_node.storage_root;
                    if !sroot.is_zero() && w.nodes.contains_key(&sroot) {
                        let acct_key = nib.clone();
                        path.push(sroot);
                        let mut snib = Vec::new();
                        descend_hash(w, sroot, path, &mut snib, true, &acct_key, out);
                        path.pop();
                    }
                }
            }
            nib.truncate(save);
        }
        TrieNode::Extension(ext) => {
            let save = nib.len();
            nib.extend(ext.key.to_vec());
            follow(w, &ext.child, path, nib, in_storage, acct, out);
            nib.truncate(save);
        }
        TrieNode::Branch(branch) => {
            for i in 0u8..16 {
                if branch.state_mask.is_bit_set(i) {
                    let idx = (branch.state_mask.get() & ((1u16 << i) - 1)).count_ones() as usize;
                    if let Some(child) = branch.stack.get(idx) {
                        nib.push(i);
                        follow(w, child, path, nib, in_storage, acct, out);
                        nib.pop();
                    }
                }
            }
        }
        TrieNode::EmptyRoot => {}
    }
}

fn follow(
    w: &IndexedWitness,
    child: &RlpNode,
    path: &mut Vec<B256>,
    nib: &mut Vec<u8>,
    in_storage: bool,
    acct: &[u8],
    out: &mut Vec<LeafRec>,
) {
    if let Some(h) = child.as_hash() {
        path.push(h);
        descend_hash(w, h, path, nib, in_storage, acct, out);
        path.pop();
    } else {
        // Inline (<32 byte) node: embedded in parent, no separate hash on the path.
        let mut b: &[u8] = child.as_ref();
        if let Ok(node) = TrieNode::decode(&mut b) {
            descend_node(w, node, path, nib, in_storage, acct, out);
        }
    }
}

/// Stable cross-block leaf identity from accumulated nibbles. Storage leaves are
/// namespaced under their account (nibbles are 0..15, so 16 is a safe separator).
fn compose_key(in_storage: bool, acct: &[u8], nib: &[u8]) -> Vec<u8> {
    if in_storage {
        let mut k = Vec::with_capacity(acct.len() + 1 + nib.len());
        k.extend_from_slice(acct);
        k.push(16);
        k.extend_from_slice(nib);
        k
    } else {
        nib.to_vec()
    }
}

fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}

struct Args {
    dir: PathBuf,
    budgets: Vec<u64>,
    top_n: u8,
}

impl Args {
    fn parse() -> eyre::Result<Self> {
        let mut dir = PathBuf::from("replay_data");
        let mut budgets = vec![10, 20, 50, 100];
        let mut top_n = 3u8;
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--dir" => dir = PathBuf::from(it.next().ok_or_else(|| eyre::eyre!("--dir value"))?),
                "--budgets-mb" => {
                    budgets = it
                        .next()
                        .ok_or_else(|| eyre::eyre!("--budgets-mb value"))?
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                }
                "--top-n" => {
                    top_n = it.next().ok_or_else(|| eyre::eyre!("--top-n value"))?.parse()?;
                }
                other => eyre::bail!("unknown argument: {other}"),
            }
        }
        let budgets = budgets.into_iter().map(|mb: u64| mb * 1_000_000).collect();
        Ok(Self { dir, budgets, top_n })
    }
}
