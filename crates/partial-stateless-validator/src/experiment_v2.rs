//! `cache-experiment-v2` — **byte-aware** cache experiment over real blocks.
//!
//! An abstract harness ([`partial_stateless_validator::byte_cache::run`]) drives
//! the validator loop — simulate each block against the current cache, then
//! update the cache — and measures **hit rate** and **sidecar (witness) bytes**.
//! Any [`Cache`] plugs in; this binary compares a client-style LRU baseline
//! against the configurable v2 design ([`V2Cache`]).
//!
//! Each node's trie depth is computed natively by walking the witness trie (see
//! [`partial_stateless_validator::depth`]), so the same `witness_<N>.json` files
//! [`ps-replay`] produces are enough; pre-lowered `<N>.acc` byte traces also work.
//!
//! Usage:
//!   cache-experiment-v2 --dir replay_data --budgets-mb 50,200,500,1000,2000 --top-n 3
//!   cache-experiment-v2 --dir replay_data --policies lru,lru+top,v2 --top-n auto
//!   cache-experiment-v2 --dir replay_data --budgets-mb 1000 --sweep-ntop 0,1,2,3,4,5,6
//!
//! Get the data first with: `ps-replay --blocks 300 --mode canonical --out replay_data`.

use alloy_primitives::{keccak256, B256};
use partial_stateless_validator::{
    byte_cache::{
        self, auto_top_n, trace_floor, Item, ItemKind, V2Cache, V2Config, DEFAULT_TOP_NODE_SIZE,
    },
    depth::{compute_depths, DEEP},
    witness::{IndexedWitness, WitnessJson},
};
use std::path::{Path, PathBuf};

fn main() -> eyre::Result<()> {
    let args = Args::parse()?;

    let mut blocks = load_blocks(&args.dir)?;
    blocks.sort_by_key(|(number, _)| *number);
    if blocks.is_empty() {
        eyre::bail!("no witness_*.json or *.acc files found in {}", args.dir.display());
    }
    let first = blocks.first().unwrap().0;
    let last = blocks.last().unwrap().0;
    let trace: Vec<Vec<Item>> = blocks.into_iter().map(|(_, items)| items).collect();

    // Cache-independent context: total accessed bytes, code share, compulsory floor.
    let (mut total, mut code_total) = (0u64, 0u64);
    for block in &trace {
        for it in block {
            total += it.size;
            if it.kind == ItemKind::Code {
                code_total += it.size;
            }
        }
    }
    let floor = trace_floor(&trace);
    println!("# v2 byte-aware experiment: {} blocks ({first}..{last})", trace.len());
    println!(
        "#   uncached witness = {:.2}GB (code {:.2}GB) | accesses={} distinct={} | max_hit_rate={:.4}",
        gb(total),
        gb(code_total),
        floor.accesses,
        floor.distinct,
        floor.max_hit_rate(),
    );
    println!("#   metrics: hit_rate (served from cache) + sidecar witness BYTES served (lower=better)");

    if let Some(spec) = &args.sweep_ntop {
        return ntop_sweep(&trace, args.budgets[0], spec, args.top_node_size);
    }

    println!(
        "{:<9} {:>8} {:>5} {:>7} {:>9} {:>8} {:>8} {:>9}",
        "policy", "budget", "topN", "hit%", "witness", "node", "code", "vs_lru"
    );
    for &budget in &args.budgets {
        // Run every policy; remember the raw-LRU baseline for the vs column.
        let mut rows = Vec::new();
        let mut base = None;
        for spec in &args.policies {
            let tn = resolve_top_n(spec, &args.top_n, budget, args.top_node_size);
            let mut cache = byte_cache::build(spec, budget, tn, args.top_node_size, args.code_mb * 1_000_000)?;
            let m = byte_cache::run(cache.as_mut(), &trace);
            if spec == "lru" {
                base = Some(m.bytes_served);
            }
            rows.push((spec.clone(), tn, wants_top(spec), m));
        }
        let base = base.or_else(|| rows.first().map(|r| r.3.bytes_served)).unwrap_or(0);
        for (spec, tn, top, m) in rows {
            let vs = if base > 0 {
                format!("{:+.1}%", 100.0 * (m.bytes_served as f64 - base as f64) / base as f64)
            } else {
                String::new()
            };
            let tn_disp = if top { tn.to_string() } else { "-".to_string() };
            println!(
                "{:<9} {:>6}MB {:>5} {:>6.1}% {:>7.2}GB {:>6.2}GB {:>6.2}GB {:>9}",
                spec,
                budget / 1_000_000,
                tn_disp,
                m.hit_rate() * 100.0,
                gb(m.bytes_served),
                gb(m.node_served),
                gb(m.code_served),
                vs
            );
        }
    }
    Ok(())
}

/// Sweep the top-cache depth `N` at a fixed budget using the configurable v2
/// cache (which charges the top-residency cost against the budget), and pick the
/// N with the lowest sidecar bytes.
fn ntop_sweep(trace: &[Vec<Item>], budget: u64, spec: &str, node_size: u64) -> eyre::Result<()> {
    println!(
        "# ntop sweep at budget {}MB: v2 pins top depth<=N (reads free) but reserves the full ~Σ(16^d) top partial-trie; deep+code use the rest",
        budget / 1_000_000
    );
    println!(
        "{:<3} {:>14} {:>14} {:>7} {:>9} {:>9} {:>9}",
        "N", "resident_top", "cache_budget", "hit%", "witness", "node", "code"
    );
    let (mut best, mut best_n) = (u64::MAX, -1i32);
    for tok in spec.split(',') {
        let tok = tok.trim();
        if tok.is_empty() {
            continue;
        }
        let n: u8 = tok.parse().map_err(|e| eyre::eyre!("bad N {tok:?}: {e}"))?;
        let cfg = V2Config { total_budget: budget, top_n: n, top_node_size: node_size };
        let r = cfg.top_resident();
        if r >= budget {
            println!("{n:<3} {:>11.1}MB {:>14} {:>7} {:>9} {:>9} {:>9}", mb(r), "(top>budget)", "-", "-", "-", "-");
            continue;
        }
        let mut cache = V2Cache::new(cfg);
        let m = byte_cache::run(&mut cache, trace);
        if m.bytes_served < best {
            best = m.bytes_served;
            best_n = n as i32;
        }
        println!(
            "{n:<3} {:>11.1}MB {:>11.0}MB {:>6.1}% {:>7.3}GB {:>7.3}GB {:>7.3}GB",
            mb(r),
            mb(cfg.cache_budget()),
            m.hit_rate() * 100.0,
            gb(m.bytes_served),
            gb(m.node_served),
            gb(m.code_served),
        );
    }
    println!("# OPTIMAL N = {best_n}  (lowest witness {:.3}GB at budget {}MB)", gb(best), budget / 1_000_000);
    Ok(())
}

fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}
fn mb(bytes: u64) -> f64 {
    bytes as f64 / 1e6
}

/// Whether a policy spec uses top-pinning (and so has a meaningful `N`).
fn wants_top(spec: &str) -> bool {
    spec.starts_with("v2") || spec.contains("top")
}

/// Top-N depth for a policy at a given budget: fixed, or auto-fit to the budget.
fn resolve_top_n(spec: &str, top_n: &TopN, budget: u64, node_size: u64) -> u8 {
    if !wants_top(spec) {
        return 0;
    }
    match top_n {
        TopN::Fixed(n) => *n,
        TopN::Auto => auto_top_n(budget, node_size),
    }
}

/// Top-N configuration: a fixed depth or auto-fit to the budget.
enum TopN {
    Fixed(u8),
    Auto,
}

/// Command-line arguments.
struct Args {
    dir: PathBuf,
    budgets: Vec<u64>,
    policies: Vec<String>,
    top_n: TopN,
    top_node_size: u64,
    code_mb: u64,
    sweep_ntop: Option<String>,
}

impl Args {
    fn parse() -> eyre::Result<Self> {
        let mut dir = PathBuf::from("replay_data");
        let mut budgets = vec![50, 200, 500, 1000, 2000];
        let mut policies = vec!["lru".to_string(), "v2".to_string()];
        let mut top_n = TopN::Fixed(3);
        let mut top_node_size = DEFAULT_TOP_NODE_SIZE;
        let mut code_mb = 100u64;
        let mut sweep_ntop = None;

        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--dir" => dir = PathBuf::from(next(&mut it, "--dir")?),
                "--budgets-mb" => budgets = parse_u64s(&next(&mut it, "--budgets-mb")?)?,
                "--policies" => {
                    policies = next(&mut it, "--policies")?
                        .split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(String::from)
                        .collect();
                    for p in &policies {
                        if !byte_cache::POLICIES.contains(&p.as_str()) {
                            eyre::bail!("unknown policy {p:?}; available: {:?}", byte_cache::POLICIES);
                        }
                    }
                }
                "--top-n" => {
                    let v = next(&mut it, "--top-n")?;
                    top_n = if v == "auto" {
                        TopN::Auto
                    } else {
                        TopN::Fixed(v.parse().map_err(|e| eyre::eyre!("bad --top-n: {e}"))?)
                    };
                }
                "--top-node-size" => {
                    top_node_size = next(&mut it, "--top-node-size")?
                        .parse()
                        .map_err(|e| eyre::eyre!("bad --top-node-size: {e}"))?;
                }
                "--code-mb" => {
                    code_mb = next(&mut it, "--code-mb")?
                        .parse()
                        .map_err(|e| eyre::eyre!("bad --code-mb: {e}"))?;
                }
                "--sweep-ntop" => sweep_ntop = Some(next(&mut it, "--sweep-ntop")?),
                other => eyre::bail!("unknown argument: {other}"),
            }
        }
        if budgets.is_empty() {
            eyre::bail!("--budgets-mb must list at least one budget");
        }
        if policies.is_empty() {
            eyre::bail!("--policies must list at least one policy");
        }
        let mut budgets: Vec<u64> = budgets.into_iter().map(|mb| mb * 1_000_000).collect();
        budgets.sort_unstable();
        Ok(Self { dir, budgets, policies, top_n, top_node_size, code_mb, sweep_ntop })
    }
}

fn next(it: &mut impl Iterator<Item = String>, flag: &str) -> eyre::Result<String> {
    it.next().ok_or_else(|| eyre::eyre!("{flag} requires a value"))
}

fn parse_u64s(spec: &str) -> eyre::Result<Vec<u64>> {
    spec.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<u64>().map_err(|e| eyre::eyre!("bad number {s:?}: {e}")))
        .collect()
}

/// Load every block in `dir` as a byte-aware, block-ordered access list.
///
/// Auto-detected: `witness_<N>.json` (raw witnesses; depth/size/code computed
/// natively here) or `<N>.acc` (pre-lowered byte trace; replays the exact
/// dataset the Go `bytebench` used).
fn load_blocks(dir: &Path) -> eyre::Result<Vec<(u64, Vec<Item>)>> {
    let has_acc = std::fs::read_dir(dir)?
        .filter_map(Result::ok)
        .any(|e| e.path().extension().and_then(|x| x.to_str()) == Some("acc"));

    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else { continue };
        if has_acc {
            let Some(num) = name.strip_suffix(".acc") else { continue };
            let number: u64 = num.parse().map_err(|e| eyre::eyre!("bad block number in {name:?}: {e}"))?;
            out.push((number, lower_acc(&path)?));
        } else {
            let Some(num) = name.strip_prefix("witness_").and_then(|s| s.strip_suffix(".json")) else {
                continue;
            };
            let number: u64 = num.parse().map_err(|e| eyre::eyre!("bad block number in {name:?}: {e}"))?;
            out.push((number, lower_block(&path)?));
        }
    }
    Ok(out)
}

/// Parse one `<N>.acc` file: 14-byte records `[kind:u8][depth:u8][size:u32 LE][id:u64 LE]`.
fn lower_acc(path: &Path) -> eyre::Result<Vec<Item>> {
    let data = std::fs::read(path)?;
    let mut items = Vec::with_capacity(data.len() / 14);
    let mut i = 0;
    while i + 14 <= data.len() {
        let kind = if data[i] == 1 { ItemKind::Code } else { ItemKind::Node };
        let depth = data[i + 1];
        let size = u32::from_le_bytes(data[i + 2..i + 6].try_into().unwrap()) as u64;
        let id_u64 = u64::from_le_bytes(data[i + 6..i + 14].try_into().unwrap());
        let mut id = [0u8; 32];
        id[24..32].copy_from_slice(&id_u64.to_be_bytes());
        items.push(Item { kind, id: B256::from(id), size, depth });
        i += 14;
    }
    Ok(items)
}

/// Parse one `witness_<N>.json`: trie nodes (in witness order) then code, each
/// tagged with its size and (for nodes) trie depth computed natively in reth.
fn lower_block(path: &Path) -> eyre::Result<Vec<Item>> {
    let raw = std::fs::read(path)?;
    let witness: WitnessJson = serde_json::from_slice(&raw)?;
    let indexed = IndexedWitness::from_witness(witness.clone())?;
    let depths = compute_depths(&indexed);

    let mut items = Vec::with_capacity(witness.state.len() + witness.codes.len());
    for node in &witness.state {
        let id = keccak256(node);
        items.push(Item {
            kind: ItemKind::Node,
            id,
            size: node.len() as u64,
            depth: depths.get(&id).copied().unwrap_or(DEEP),
        });
    }
    for code in &witness.codes {
        let id = keccak256(code);
        items.push(Item { kind: ItemKind::Code, id, size: code.len() as u64, depth: 0 });
    }
    Ok(items)
}
