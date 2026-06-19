//! `witness-sim` — offline measurement of the **real** witness a validator
//! receives, using the network-level state-cache model (`partial-stateless`),
//! equivalent to what `partial-stateless-exex` computes live but without a node.
//!
//! For each block we:
//!   1. Re-execute it statelessly from its witness (no DB) to get a `BundleState`,
//!      then `BlockAccessedState::from_bundle` — the exact accessed account/storage/
//!      code set the ExEx would see.
//!   2. `NetworkStateCache::compute_miss` (BEFORE warming) — the keys not already
//!      held by the validator, i.e. what the builder must prove this block.
//!   3. Measure the witness the builder ships: the Merkle multiproof for the
//!      missed accounts/storage (reconstructed by walking the block's witness
//!      nodes) plus the missed contract-code bytes — all raw bytes, matching
//!      `measure_multiproof_size`.
//!   4. `on_block_executed` to warm/evict the cache for the next block.
//!
//! Block bodies come from `debug_getRawBlock` (witness has no txs); set
//! `RETH_RPC_URL` (default talks to the Docker host gateway). Witnesses are read
//! from the `ps-replay` dump dir.
//!
//! Usage:
//!   RETH_RPC_URL=http://172.17.0.1:18545 \
//!     witness-sim --dir replay_data --windows 8,32,128 [--blocks N]
//!
//! The `NOCACHE` row treats every accessed key as missed; its account+storage
//! proof bytes should match the witness `state` size and its code bytes the
//! witness `codes` size — a built-in fidelity check on the walk reconstruction.

use alloy_primitives::{keccak256, map::B256Map, Address, B256};
use alloy_rlp::Decodable;
use alloy_trie::EMPTY_ROOT_HASH;
use eyre::WrapErr;
use partial_stateless::{
    accessed_state::BlockAccessedState,
    network_cache::NetworkStateCache,
    policy::LastNBlocksPolicy,
};
use partial_stateless_validator::{
    db::WitnessStateProvider,
    trie::{resolve, resolve_account, NodeSource},
    witness::IndexedWitness,
};
use alloy_evm::block::BlockExecutor;
use reth_ethereum_primitives::Block;
use reth_evm::ConfigureEvm;
use reth_evm_ethereum::EthEvmConfig;
use reth_primitives_traits::RecoveredBlock;
use reth_revm::{
    database::StateProviderDatabase,
    db::{states::bundle_state::BundleRetention, State},
};
use std::{
    cell::RefCell,
    path::{Path, PathBuf},
};

fn main() -> eyre::Result<()> {
    let args = Args::parse()?;
    let rpc_url = std::env::var("RETH_RPC_URL")
        .unwrap_or_else(|_| "http://172.17.0.1:18545".to_string());

    // --- Phase 1: load each witness and re-execute to get its accessed state. ---
    let mut files = witness_files(&args.dir)?;
    files.sort_by_key(|(n, _)| *n);
    if files.is_empty() {
        eyre::bail!("no witness_*.json files in {}", args.dir.display());
    }
    if let Some(limit) = args.blocks {
        files.truncate(limit);
    }

    let mut blocks: Vec<BlockData> = Vec::with_capacity(files.len());
    let mut skipped = 0u64;
    for (number, path) in &files {
        let witness = IndexedWitness::load(path)?;
        let block = match fetch_block(&rpc_url, *number) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("  skip block {number}: fetch failed ({e}) — is RETH_RPC_URL reachable?");
                skipped += 1;
                continue;
            }
        };
        match reexecute(&witness, &block) {
            Ok(accessed) => {
                eprintln!(
                    "  loaded block {number}: accessed accounts={} storage={} codes={}",
                    accessed.accounts.len(),
                    accessed.storage.len(),
                    accessed.codes.len()
                );
                blocks.push(BlockData { number: *number, witness, accessed });
            }
            Err(e) => {
                eprintln!("  skip block {number}: re-execution failed ({e:#})");
                skipped += 1;
            }
        }
    }
    if blocks.is_empty() {
        eyre::bail!("no blocks could be re-executed ({skipped} skipped)");
    }
    if skipped > 0 {
        eprintln!("  ({skipped} block(s) skipped)");
    }

    let n = blocks.len() as u64;
    println!(
        "# blocks={} ({}..{}) — witness bytes per block (raw), network_cache model",
        n,
        blocks.first().unwrap().number,
        blocks.last().unwrap().number,
    );
    println!(
        "{:<10} {:>12} {:>12} {:>12} {:>14} {:>10}",
        "policy", "acct_proof", "stor_proof", "codes", "witness/blk", "missing"
    );

    // NOCACHE baseline: every accessed key missed (≈ the full builder witness,
    // and a fidelity check vs the witness `state`/`codes` sizes).
    {
        let (mut a, mut s, mut c, mut miss) = (0u64, 0u64, 0u64, 0u64);
        for bd in &blocks {
            let accounts: Vec<Address> = bd.accessed.accounts.keys().copied().collect();
            let storage: Vec<(Address, B256)> = bd.accessed.storage.keys().copied().collect();
            let codes: Vec<B256> = bd.accessed.codes.keys().copied().collect();
            let m = measure(&bd.witness, &accounts, &storage, &codes, &bd.accessed);
            a += m.account_bytes;
            s += m.storage_bytes;
            c += m.code_bytes;
            miss += m.missing_nodes;
        }
        print_row("NOCACHE", a, s, c, miss, n);
    }

    // --- Phase 2: sweep cache windows (account & storage use the same window). ---
    for &w in &args.windows {
        let mut cache = NetworkStateCache::new(
            Box::new(LastNBlocksPolicy::new(w)),
            Box::new(LastNBlocksPolicy::new(w)),
        );
        let (mut a, mut s, mut c, mut miss) = (0u64, 0u64, 0u64, 0u64);
        for bd in &blocks {
            // Miss against the cache as the validator holds it entering the block.
            let m = cache.compute_miss(&bd.accessed);
            let r = measure(&bd.witness, &m.missed_accounts, &m.missed_storage, &m.missed_codes, &bd.accessed);
            a += r.account_bytes;
            s += r.storage_bytes;
            c += r.code_bytes;
            miss += r.missing_nodes;
            // Warm the cache for subsequent blocks.
            cache.on_block_executed(bd.number, &bd.accessed);
        }
        print_row(&format!("win={w}"), a, s, c, miss, n);
    }

    Ok(())
}

/// A loaded block: its witness and the state it accessed (from re-execution).
struct BlockData {
    number: u64,
    witness: IndexedWitness,
    accessed: BlockAccessedState,
}

/// Witness bytes the builder must ship for one block's missed keys.
struct Measured {
    account_bytes: u64,
    storage_bytes: u64,
    code_bytes: u64,
    /// Proof nodes that were not in the witness (should be 0 — counts gaps).
    missing_nodes: u64,
}

/// Re-execute a block's transactions statelessly from its witness and return the
/// accessed state.
///
/// We drive the [`BlockExecutor`] manually and **skip
/// `apply_pre_execution_changes`** (the EIP-2935 blockhash and EIP-4788
/// beacon-root system calls). Those calls write to fixed system contracts whose
/// write-path trie nodes the saved witness omits (it was produced by a different
/// reth version), so the stock `execute_one` aborts at block start. Skipping them
/// only drops the accessed state of those two system contracts — a fixed,
/// per-block constant that is irrelevant to the cache-policy comparison.
/// Post-execution changes (withdrawals) are applied best-effort.
fn reexecute(witness: &IndexedWitness, block: &RecoveredBlock<Block>) -> eyre::Result<BlockAccessedState> {
    let provider = WitnessStateProvider::new(witness);
    let evm_config = EthEvmConfig::mainnet();
    let mut state =
        State::builder().with_database(StateProviderDatabase::new(provider)).with_bundle_update().build();
    {
        let mut executor = evm_config
            .executor_for_block(&mut state, block)
            .map_err(|e| eyre::eyre!("executor_for_block: {e:?}"))?;
        // (skip executor.apply_pre_execution_changes())
        for tx in block.transactions_recovered() {
            executor.execute_transaction(tx).wrap_err("execute transaction")?;
        }
        // Withdrawals etc. — best effort; a missing system node here is non-fatal.
        if let Err(e) = executor.apply_post_execution_changes() {
            eprintln!("    note: post-execution changes skipped ({e})");
        }
    }
    state.merge_transitions(BundleRetention::Reverts);
    let bundle = state.take_bundle();
    Ok(BlockAccessedState::from_bundle(&bundle))
}

/// Measure the multiproof bytes for the given missed keys against the block's
/// pre-state, reconstructed by walking the witness node set, plus missed code
/// bytes. Mirrors `partial_stateless::witness::measure_multiproof_size`.
fn measure(
    witness: &IndexedWitness,
    missed_accounts: &[Address],
    missed_storage: &[(Address, B256)],
    missed_codes: &[B256],
    accessed: &BlockAccessedState,
) -> Measured {
    let mut missing = 0u64;

    // Account-proof targets = missed accounts ∪ accounts that have missed storage
    // (the account proof anchors the storage root). Walk each, recording the
    // path nodes; capture storage roots for the storage phase.
    let mut account_targets: Vec<Address> = missed_accounts.to_vec();
    for (addr, _) in missed_storage {
        if !account_targets.contains(addr) {
            account_targets.push(*addr);
        }
    }

    let acct_rec = RecordingSource::new(witness);
    let mut storage_roots: B256Map<B256> = B256Map::default(); // keccak(addr) -> storage_root
    for addr in &account_targets {
        let ha = keccak256(addr);
        match resolve_account(&acct_rec, witness.pre_state_root, ha) {
            Ok(Some(account)) => {
                storage_roots.insert(ha, account.storage_root);
            }
            Ok(None) => {} // provably absent — path nodes still recorded
            Err(_) => missing += 1,
        }
    }
    let account_bytes = acct_rec.bytes();

    // Storage-proof: group missed slots by account, walk each account's storage
    // trie (recording its nodes) from the captured storage root.
    let storage_rec = RecordingSource::new(witness);
    let mut by_account: std::collections::HashMap<Address, Vec<B256>> = std::collections::HashMap::new();
    for (addr, slot) in missed_storage {
        by_account.entry(*addr).or_default().push(*slot);
    }
    for (addr, slots) in &by_account {
        let ha = keccak256(addr);
        let Some(&root) = storage_roots.get(&ha) else { continue };
        if root == EMPTY_ROOT_HASH {
            continue;
        }
        for slot in slots {
            let hs = keccak256(slot);
            if let Err(_) = resolve(&storage_rec, root, hs) {
                missing += 1;
            }
        }
    }
    let storage_bytes = storage_rec.bytes();

    // Missed contract code bytes (codes are not part of the trie proof).
    let code_bytes: u64 = missed_codes
        .iter()
        .filter_map(|h| accessed.codes.get(h))
        .map(|b| b.len() as u64)
        .sum();

    Measured { account_bytes, storage_bytes, code_bytes, missing_nodes: missing }
}

/// A [`NodeSource`] over the witness that records every distinct node it serves,
/// so the set of nodes touched while resolving the proof targets = the proof.
struct RecordingSource<'a> {
    witness: &'a IndexedWitness,
    /// `node hash => byte length` (dedup: each node counted once).
    seen: RefCell<B256Map<u64>>,
}

impl<'a> RecordingSource<'a> {
    fn new(witness: &'a IndexedWitness) -> Self {
        Self { witness, seen: RefCell::new(B256Map::default()) }
    }

    /// Total bytes of all distinct nodes served (the proof size).
    fn bytes(&self) -> u64 {
        self.seen.borrow().values().sum()
    }
}

impl NodeSource for RecordingSource<'_> {
    fn get(&self, hash: &B256) -> Option<&[u8]> {
        let node = self.witness.nodes.get(hash);
        if let Some(bytes) = node {
            self.seen.borrow_mut().entry(*hash).or_insert(bytes.len() as u64);
        }
        node.map(|b| b.as_ref())
    }
}

fn print_row(name: &str, account: u64, storage: u64, code: u64, missing: u64, blocks: u64) {
    let total = account + storage + code;
    let per = |b: u64| b as f64 / blocks as f64 / 1024.0; // KiB/block
    println!(
        "{:<10} {:>10.1}K {:>10.1}K {:>10.1}K {:>11.3}M {:>10}",
        name,
        per(account),
        per(storage),
        per(code),
        total as f64 / blocks as f64 / (1024.0 * 1024.0),
        missing,
    );
}

/// Command-line arguments.
struct Args {
    dir: PathBuf,
    windows: Vec<u64>,
    blocks: Option<usize>,
}

impl Args {
    fn parse() -> eyre::Result<Self> {
        let mut dir = None;
        let mut windows = vec![8u64, 32, 128];
        let mut blocks = None;
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--dir" => dir = Some(PathBuf::from(req(&mut it, "--dir")?)),
                "--windows" => {
                    windows = req(&mut it, "--windows")?
                        .split(',')
                        .map(|s| s.trim())
                        .filter(|s| !s.is_empty())
                        .map(|s| s.parse::<u64>().map_err(|e| eyre::eyre!("bad window {s:?}: {e}")))
                        .collect::<eyre::Result<_>>()?;
                }
                "--blocks" => {
                    blocks = Some(req(&mut it, "--blocks")?.parse().wrap_err("--blocks must be an integer")?)
                }
                other => eyre::bail!("unknown argument: {other}"),
            }
        }
        Ok(Self { dir: dir.ok_or_else(|| eyre::eyre!("--dir is required"))?, windows, blocks })
    }
}

fn req(it: &mut impl Iterator<Item = String>, flag: &str) -> eyre::Result<String> {
    it.next().ok_or_else(|| eyre::eyre!("{flag} requires a value"))
}

/// List `witness_<N>.json` files in `dir` as `(block_number, path)`.
fn witness_files(dir: &Path) -> eyre::Result<Vec<(u64, PathBuf)>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else { continue };
        let Some(num) = name.strip_prefix("witness_").and_then(|s| s.strip_suffix(".json")) else {
            continue;
        };
        let number = num.parse().map_err(|e| eyre::eyre!("bad block number in {name:?}: {e}"))?;
        out.push((number, path));
    }
    Ok(out)
}

/// Fetch a block as `RecoveredBlock` via `debug_getRawBlock` (raw RLP).
fn fetch_block(rpc_url: &str, number: u64) -> eyre::Result<RecoveredBlock<Block>> {
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "debug_getRawBlock",
        "params": [format!("0x{number:x}")],
        "id": 1,
    });
    let resp: serde_json::Value = reqwest::blocking::Client::new()
        .post(rpc_url)
        .json(&body)
        .send()
        .wrap_err("debug_getRawBlock request failed")?
        .json()
        .wrap_err("decoding debug_getRawBlock response failed")?;
    if let Some(err) = resp.get("error").filter(|e| !e.is_null()) {
        eyre::bail!("debug_getRawBlock RPC error: {err}");
    }
    let hex = resp
        .get("result")
        .and_then(|v| v.as_str())
        .ok_or_else(|| eyre::eyre!("debug_getRawBlock returned no result"))?;
    let raw = alloy_primitives::hex::decode(hex.trim_start_matches("0x")).wrap_err("bad raw block hex")?;
    let block = Block::decode(&mut raw.as_slice()).wrap_err("decode block RLP")?;
    RecoveredBlock::try_recover(block).wrap_err("recover transaction senders")
}
