//! Partial Statelessness — RPC replay fetcher.
//!
//! Pulls `(block, execution witness)` pairs for a recent range of blocks from a
//! running reth node via JSON-RPC and dumps them to disk. This lets the
//! stateless-execution experiment (cache + witness re-execution) run offline,
//! against the same data a validator would receive, without having to embed an
//! ExEx into the running node.
//!
//! For each block `N` we fetch:
//!   - `eth_getBlockByNumber(N, true)`   → full block (header + transactions)
//!   - `debug_executionWitness(N)`       → execution witness (trie node preimages,
//!                                          bytecodes, key preimages, ancestor headers)
//!
//! Run with (defaults talk to the local yebon-reth node):
//!   RETH_RPC_URL=http://localhost:18545 ps-replay --blocks 100 --out replay_data
//!
//! Output layout:
//!   <out>/block_<N>.json
//!   <out>/witness_<N>.json
//!   <out>/manifest.json   (range + per-block summary)

use eyre::{eyre, Result, WrapErr};
use serde_json::{json, Value};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use tracing::{info, warn};

/// Default RPC endpoint (the local yebon-reth node).
const DEFAULT_RPC_URL: &str = "http://localhost:18545";
/// Default number of recent blocks to fetch.
const DEFAULT_BLOCKS: u64 = 100;
/// Default output directory.
const DEFAULT_OUT: &str = "replay_data";

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let cfg = Config::from_env_and_args()?;
    fs::create_dir_all(&cfg.out_dir)
        .wrap_err_with(|| format!("failed to create output dir {}", cfg.out_dir.display()))?;

    let client = RpcClient::new(cfg.rpc_url.clone());

    // Resolve the block range: [head - blocks + 1, head].
    let head = client.block_number().await.wrap_err("eth_blockNumber failed")?;
    let start = head.saturating_sub(cfg.blocks.saturating_sub(1));
    info!(head, start, end = head, blocks = cfg.blocks, mode = %cfg.mode, out = %cfg.out_dir.display(), "Fetching block + witness range");

    let mut summaries = Vec::new();
    for bn in start..=head {
        match fetch_one(&client, bn, &cfg.out_dir, &cfg.mode).await {
            Ok(summary) => {
                info!(
                    block = bn,
                    txs = summary.txs,
                    witness_state_nodes = summary.state_nodes,
                    witness_codes = summary.codes,
                    witness_bytes = summary.witness_bytes,
                    "saved"
                );
                summaries.push(summary);
            }
            Err(e) => {
                // One bad block (e.g. pruned witness) shouldn't abort the whole run.
                warn!(block = bn, error = %e, "skipping block");
            }
        }
    }

    // Write a manifest so downstream phases know exactly what was captured.
    let manifest = json!({
        "rpc_url": cfg.rpc_url,
        "head": head,
        "range": { "start": start, "end": head },
        "requested_blocks": cfg.blocks,
        "captured_blocks": summaries.len(),
        "blocks": summaries.iter().map(Summary::to_json).collect::<Vec<_>>(),
    });
    let manifest_path = cfg.out_dir.join("manifest.json");
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)
        .wrap_err_with(|| format!("failed to write {}", manifest_path.display()))?;

    let total_witness: u64 = summaries.iter().map(|s| s.witness_bytes).sum();
    info!(
        captured = summaries.len(),
        total_witness_bytes = total_witness,
        manifest = %manifest_path.display(),
        "Done"
    );
    Ok(())
}

/// Fetch and persist one block's data, returning a summary.
async fn fetch_one(
    client: &RpcClient,
    bn: u64,
    out_dir: &std::path::Path,
    mode: &str,
) -> Result<Summary> {
    let tag = format!("0x{bn:x}");

    let block = client
        .call("eth_getBlockByNumber", json!([tag, true]))
        .await
        .wrap_err("eth_getBlockByNumber failed")?;
    if block.is_null() {
        return Err(eyre!("block {bn} not found"));
    }

    // Witness format: "legacy" (server default) or "canonical" (the modern,
    // leaner spec — match this with the cache-benchmark tool's `-mode`).
    let witness = client
        .call("debug_executionWitness", json!([tag, mode]))
        .await
        .wrap_err("debug_executionWitness failed")?;

    // Persist raw JSON; downstream phases decode into typed alloy/reth structures.
    let block_bytes = serde_json::to_vec(&block)?;
    let witness_bytes = serde_json::to_vec(&witness)?;
    fs::write(out_dir.join(format!("block_{bn}.json")), &block_bytes)?;
    fs::write(out_dir.join(format!("witness_{bn}.json")), &witness_bytes)?;

    Ok(Summary {
        block: bn,
        txs: block.get("transactions").and_then(Value::as_array).map_or(0, Vec::len),
        state_nodes: witness.get("state").and_then(Value::as_array).map_or(0, Vec::len),
        codes: witness.get("codes").and_then(Value::as_array).map_or(0, Vec::len),
        keys: witness.get("keys").and_then(Value::as_array).map_or(0, Vec::len),
        witness_bytes: witness_bytes.len() as u64,
    })
}

/// Runtime configuration resolved from environment variables and CLI flags.
struct Config {
    rpc_url: String,
    blocks: u64,
    out_dir: PathBuf,
    mode: String,
}

impl Config {
    /// `RETH_RPC_URL` env + `--blocks <N>` / `--out <dir>` flags (flags win over env/defaults).
    fn from_env_and_args() -> Result<Self> {
        let mut rpc_url =
            std::env::var("RETH_RPC_URL").unwrap_or_else(|_| DEFAULT_RPC_URL.to_string());
        let mut blocks = DEFAULT_BLOCKS;
        let mut out_dir = PathBuf::from(DEFAULT_OUT);
        let mut mode = "legacy".to_string();

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--mode" => {
                    mode = args.next().ok_or_else(|| eyre!("--mode requires a value"))?;
                    if mode != "legacy" && mode != "canonical" {
                        return Err(eyre!("--mode must be 'legacy' or 'canonical'"));
                    }
                }
                "--blocks" => {
                    blocks = args
                        .next()
                        .ok_or_else(|| eyre!("--blocks requires a value"))?
                        .parse()
                        .wrap_err("--blocks must be a positive integer")?;
                }
                "--out" => {
                    out_dir =
                        PathBuf::from(args.next().ok_or_else(|| eyre!("--out requires a value"))?);
                }
                "--rpc-url" => {
                    rpc_url = args.next().ok_or_else(|| eyre!("--rpc-url requires a value"))?;
                }
                other => return Err(eyre!("unknown argument: {other}")),
            }
        }

        if blocks == 0 {
            return Err(eyre!("--blocks must be > 0"));
        }
        Ok(Self { rpc_url, blocks, out_dir, mode })
    }
}

/// Per-block capture summary, recorded in the manifest.
struct Summary {
    block: u64,
    txs: usize,
    state_nodes: usize,
    codes: usize,
    keys: usize,
    witness_bytes: u64,
}

impl Summary {
    fn to_json(&self) -> Value {
        json!({
            "block": self.block,
            "txs": self.txs,
            "witness_state_nodes": self.state_nodes,
            "witness_codes": self.codes,
            "witness_keys": self.keys,
            "witness_bytes": self.witness_bytes,
        })
    }
}

/// Minimal JSON-RPC 2.0 client over HTTP.
struct RpcClient {
    url: String,
    http: reqwest::Client,
    next_id: AtomicU64,
}

impl RpcClient {
    fn new(url: String) -> Self {
        // A per-request timeout so one stuck block (e.g. a witness outside the
        // node's pruned state window, which can hang indefinitely) is skipped
        // rather than stalling the whole run.
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .expect("reqwest client");
        Self { url, http, next_id: AtomicU64::new(1) }
    }

    /// Convenience wrapper for `eth_blockNumber`, returning the height as `u64`.
    async fn block_number(&self) -> Result<u64> {
        let result = self.call("eth_blockNumber", json!([])).await?;
        let hex = result.as_str().ok_or_else(|| eyre!("eth_blockNumber: non-string result"))?;
        u64::from_str_radix(hex.trim_start_matches("0x"), 16)
            .wrap_err("eth_blockNumber: invalid hex")
    }

    /// Issue a single JSON-RPC call and return the `result` field.
    async fn call(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let body = json!({ "jsonrpc": "2.0", "method": method, "params": params, "id": id });

        let resp: Value = self
            .http
            .post(&self.url)
            .json(&body)
            .send()
            .await
            .wrap_err_with(|| format!("HTTP request for {method} failed"))?
            .json()
            .await
            .wrap_err_with(|| format!("decoding JSON response for {method} failed"))?;

        if let Some(err) = resp.get("error").filter(|e| !e.is_null()) {
            return Err(eyre!("RPC error for {method}: {err}"));
        }
        resp.get("result")
            .cloned()
            .ok_or_else(|| eyre!("RPC response for {method} had no result"))
    }
}

fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt().with_env_filter(filter).init();
}
