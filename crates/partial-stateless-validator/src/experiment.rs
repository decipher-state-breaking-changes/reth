//! `cache-experiment` — sweep trie-node cache policies over a range of real
//! blocks and report hit/miss rates, the Rust counterpart to the team's Go
//! `psl-cache-bench` (so numbers are directly comparable).
//!
//! The access set per block is the witness `state` node set (every node the
//! builder would ship), keyed by **keccak** (the trie address used for
//! reconstruction). A node already resident from an earlier block is a hit — i.e.
//! a node the builder would NOT need to put in this block's witness.
//!
//! Usage:
//!   cache-experiment --dir replay_data --policy lru,fifo --caps 100000,500000
//!   cache-experiment --dir replay_data --policy all --caps 1000000
//!
//! Get the data first with: `ps-replay --blocks 100 --out replay_data`.

use alloy_primitives::B256;
use partial_stateless_validator::{
    cache::{self, AVAILABLE_POLICIES},
    witness::IndexedWitness,
};
use std::{collections::HashSet, path::PathBuf};

fn main() -> eyre::Result<()> {
    let args = Args::parse()?;

    // Load each block's witness once; keep node maps in memory so the sweep can
    // replay every (policy, capacity) combo without re-reading from disk.
    let mut blocks = load_blocks(&args.dir)?;
    blocks.sort_by_key(|(number, _)| *number);
    if blocks.is_empty() {
        eyre::bail!("no witness_*.json files found in {}", args.dir.display());
    }

    // Per-block access sets (keccak node hashes) and the compulsory-miss floor.
    let access_sets: Vec<Vec<B256>> =
        blocks.iter().map(|(_, w)| w.nodes.keys().copied().collect()).collect();
    let accesses: u64 = access_sets.iter().map(|s| s.len() as u64).sum();
    let distinct = access_sets.iter().flatten().copied().collect::<HashSet<_>>().len() as u64;
    let max_hit_rate = if accesses > 0 { (accesses - distinct) as f64 / accesses as f64 } else { 0.0 };

    println!(
        "# blocks={} ({}..{}) accesses={accesses} distinct_nodes={distinct} (compulsory floor) max_hit_rate={max_hit_rate:.4}",
        blocks.len(),
        blocks.first().unwrap().0,
        blocks.last().unwrap().0,
    );
    println!("{:<8} {:>10} {:>9} {:>9} {:>12} {:>12} {:>9}", "policy", "capacity", "hit_rate", "miss_rate", "hits", "misses", "%_of_max");

    for &capacity in &args.caps {
        for policy in &args.policies {
            let mut cache = cache::build(policy, capacity)?;
            let (mut hits, mut misses) = (0u64, 0u64);

            for (block_idx, ids) in access_sets.iter().enumerate() {
                // Count residency against the cache as of the start of this block.
                for id in ids {
                    if cache.contains(id) {
                        hits += 1;
                    } else {
                        misses += 1;
                    }
                }
                // Warm the cache with this block's nodes (admit + policy eviction).
                let node_map = &blocks[block_idx].1.nodes;
                for id in ids {
                    if let Some(bytes) = node_map.get(id) {
                        cache.insert(*id, bytes.clone());
                    }
                }
            }

            let hit_rate = if accesses > 0 { hits as f64 / accesses as f64 } else { 0.0 };
            let pct_of_max = if max_hit_rate > 0.0 { hit_rate / max_hit_rate * 100.0 } else { 0.0 };
            println!(
                "{:<8} {:>10} {:>9.4} {:>9.4} {:>12} {:>12} {:>8.1}%",
                policy, capacity, hit_rate, 1.0 - hit_rate, hits, misses, pct_of_max
            );
        }
    }

    Ok(())
}

/// Command-line arguments for the sweep.
struct Args {
    dir: PathBuf,
    policies: Vec<String>,
    caps: Vec<usize>,
}

impl Args {
    fn parse() -> eyre::Result<Self> {
        let mut dir = PathBuf::from("replay_data");
        let mut policies = vec!["lru".to_string()];
        let mut caps = vec![1_000_000usize];

        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--dir" => dir = PathBuf::from(next(&mut it, "--dir")?),
                "--policy" => policies = parse_policies(&next(&mut it, "--policy")?)?,
                "--caps" => caps = parse_caps(&next(&mut it, "--caps")?)?,
                other => eyre::bail!("unknown argument: {other}"),
            }
        }
        Ok(Self { dir, policies, caps })
    }
}

fn next(it: &mut impl Iterator<Item = String>, flag: &str) -> eyre::Result<String> {
    it.next().ok_or_else(|| eyre::eyre!("{flag} requires a value"))
}

fn parse_policies(spec: &str) -> eyre::Result<Vec<String>> {
    if spec.trim() == "all" {
        return Ok(AVAILABLE_POLICIES.iter().map(|s| s.to_string()).collect());
    }
    let names: Vec<String> = spec.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    for name in &names {
        if !AVAILABLE_POLICIES.contains(&name.as_str()) {
            eyre::bail!("unknown policy {name:?}; available: {AVAILABLE_POLICIES:?}");
        }
    }
    Ok(names)
}

fn parse_caps(spec: &str) -> eyre::Result<Vec<usize>> {
    let mut caps: Vec<usize> = spec
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().map_err(|e| eyre::eyre!("bad capacity {s:?}: {e}")))
        .collect::<eyre::Result<_>>()?;
    caps.sort_unstable();
    Ok(caps)
}

/// Load every `witness_<N>.json` in `dir`, returning `(block_number, witness)`.
fn load_blocks(dir: &std::path::Path) -> eyre::Result<Vec<(u64, IndexedWitness)>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else { continue };
        let Some(num) = name.strip_prefix("witness_").and_then(|s| s.strip_suffix(".json")) else {
            continue;
        };
        let number: u64 = num.parse().map_err(|e| eyre::eyre!("bad block number in {name:?}: {e}"))?;
        out.push((number, IndexedWitness::load(&path)?));
    }
    Ok(out)
}
