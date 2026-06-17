//! `validate` — re-execute blocks **statelessly** from their witness and check them.
//!
//! Given only a block's execution witness (no state database), reconstruct the
//! state it reads, run reth's real block executor against it, and confirm the
//! executed `gas_used` matches the header. If execution completes, the witness
//! (plus cache) served every read — [`WitnessStateProvider`] never falls back to
//! a DB, so a missing node aborts execution.
//!
//! Two modes:
//!   validate <witness_N.json> [rpc-url]            # single block, witness only
//!   validate --dir <d> [--policy lru] [--cap N]    # window: warm a cache across
//!                                                    # blocks, run each from
//!                                                    # {cache ∪ witness}, and report
//!                                                    # how much of the witness the
//!                                                    # cache already covered.
//!
//! The block body (txs) is fetched as raw RLP via `debug_getRawBlock`; the
//! witness supplies the pre-state.

use alloy_consensus::BlockHeader;
use alloy_primitives::B256;
use alloy_rlp::Decodable;
use eyre::WrapErr;
use partial_stateless_validator::{cache, db::WitnessStateProvider, witness::IndexedWitness};
use reth_ethereum_primitives::Block;
use reth_evm::{execute::Executor, ConfigureEvm};
use reth_evm_ethereum::EthEvmConfig;
use reth_primitives_traits::RecoveredBlock;
use reth_revm::database::StateProviderDatabase;
use reth_trie_common::{DecodedMultiProofV2, HashedPostState, KeccakKeyHasher, Nibbles};
use reth_trie_sparse::{RevealableSparseTrie, SparseStateTrie};
use std::path::{Path, PathBuf};
use tracing::{info, warn};

const DEFAULT_RPC_URL: &str = "http://localhost:18545";

fn main() -> eyre::Result<()> {
    init_tracing();
    let rpc_url =
        std::env::var("RETH_RPC_URL").unwrap_or_else(|_| DEFAULT_RPC_URL.to_string());

    // `--dir` selects the windowed cache sweep; a positional path selects a
    // single block.
    let mut args = std::env::args().skip(1).peekable();
    if args.peek().map(|a| a == "--dir").unwrap_or(false) {
        run_window(WindowArgs::parse(std::env::args().skip(1))?, &rpc_url)
    } else {
        let path = args
            .next()
            .map(PathBuf::from)
            .ok_or_else(|| eyre::eyre!("usage: validate <witness_N.json> | validate --dir <d>"))?;
        let rpc = args.next().unwrap_or(rpc_url);
        run_single(&path, &rpc)
    }
}

/// Validate one block from its witness alone: re-execute, check gas, and
/// recompute the post-state root (gold-standard, consensus-grade check).
fn run_single(path: &Path, rpc_url: &str) -> eyre::Result<()> {
    let block_number = parse_block_number(path)?;
    let witness = IndexedWitness::load(path)?;
    let block = fetch_block(rpc_url, block_number)?;

    info!(
        block = block_number,
        txs = block.body().transactions.len(),
        witness_nodes = witness.nodes.len(),
        "re-executing block statelessly from witness (no state DB)"
    );

    // Execute, then take the resulting state changes for the root recompute.
    let provider = WitnessStateProvider::new(&witness);
    let evm_config = EthEvmConfig::mainnet();
    let mut executor = evm_config.batch_executor(StateProviderDatabase::new(provider));
    let result = executor
        .execute_one(&block)
        .wrap_err("stateless execution failed — witness insufficient or block invalid")?;

    let header_gas = block.header().gas_used();
    let gas_match = result.gas_used == header_gas;
    info!(block = block_number, executed_gas = result.gas_used, header_gas, gas_match, "execution complete");
    if gas_match {
        info!("stateless re-execution succeeded and gas_used matches the header");
    } else {
        warn!("gas_used mismatch — execution diverged from canonical");
    }

    // Gold-standard check: recompute the post-state root from {witness ∪ changes}.
    let bundle = executor.into_state().take_bundle();
    let hashed = HashedPostState::from_bundle_state::<KeccakKeyHasher>(&bundle.state);
    match recompute_state_root(&witness, &hashed) {
        Ok(computed_root) => {
            let header_root = block.header().state_root();
            let root_match = computed_root == header_root;
            info!(%computed_root, %header_root, root_match, "post-state root check");
            if root_match {
                info!("post-state root matches — consensus-grade stateless validation complete");
            } else {
                warn!("post-state root mismatch");
            }
        }
        Err(e) => warn!(error = %e, "post-state root recompute failed (gas check still holds)"),
    }
    Ok(())
}

/// Recompute the post-execution state root from the witness pre-state plus the
/// block's state changes, using a sparse trie — no database.
///
/// Reveal the witness into a [`SparseStateTrie`], apply the hashed changes
/// (storage first so account storage roots are recomputed, then accounts), and
/// hash up to the new root.
fn recompute_state_root(
    witness: &IndexedWitness,
    hashed: &HashedPostState,
) -> eyre::Result<B256> {
    let proof = DecodedMultiProofV2::from_witness(witness.pre_state_root, &witness.nodes)
        .map_err(|e| eyre::eyre!("rebuild proof from witness: {e}"))?;
    let mut sparse = SparseStateTrie::new();
    sparse
        .reveal_decoded_multiproof_v2(proof)
        .map_err(|e| eyre::eyre!("reveal witness into sparse trie: {e:?}"))?;

    for (address, storage) in &hashed.storages {
        // Accounts whose pre-state storage was empty have no revealed storage
        // trie; start them from an empty (revealed) one so leaf updates apply.
        if sparse.storage_trie_ref(address).is_none() {
            sparse.insert_storage_trie(*address, RevealableSparseTrie::revealed_empty());
        }
        if storage.wiped {
            sparse
                .wipe_storage(*address)
                .map_err(|e| eyre::eyre!("wipe storage {address}: {e:?}"))?;
        }
        for (slot, value) in &storage.storage {
            let path = Nibbles::unpack(*slot);
            if value.is_zero() {
                sparse
                    .remove_storage_leaf(*address, &path)
                    .map_err(|e| eyre::eyre!("remove storage leaf: {e:?}"))?;
            } else {
                sparse
                    .update_storage_leaf(*address, path, alloy_rlp::encode(*value))
                    .map_err(|e| eyre::eyre!("update storage leaf: {e:?}"))?;
            }
        }
    }

    for (address, account) in &hashed.accounts {
        sparse
            .update_account_stateless(*address, *account)
            .map_err(|e| eyre::eyre!("update account {address}: {e:?}"))?;
    }

    sparse.root().map_err(|e| eyre::eyre!("compute sparse root: {e:?}"))
}

/// Warm a cache across a window of blocks, running each from `{cache ∪ witness}`
/// and reporting how many of its witness nodes the cache already held (= nodes
/// the builder would NOT need to ship).
fn run_window(args: WindowArgs, rpc_url: &str) -> eyre::Result<()> {
    let mut blocks = load_blocks(&args.dir)?;
    blocks.sort_by_key(|(number, _)| *number);
    if blocks.is_empty() {
        eyre::bail!("no witness_*.json files in {}", args.dir.display());
    }

    let mut cache = cache::build(&args.policy, args.cap)?;
    info!(policy = %args.policy, capacity = args.cap, blocks = blocks.len(), "windowed stateless validation");
    println!(
        "{:<10} {:>5} {:>11} {:>11} {:>14} {:>9}",
        "block", "txs", "wit_nodes", "cached", "shipped(witness)", "gas_ok"
    );

    let (mut ok_count, mut total) = (0u64, 0u64);
    for (number, witness) in &blocks {
        // How much of this block's witness is already cached (would be omitted).
        let node_ids: Vec<B256> = witness.nodes.keys().copied().collect();
        let cached = node_ids.iter().filter(|h| cache.contains(h)).count();
        let shipped = node_ids.len() - cached;

        // Execute from {cache ∪ witness}.
        let block = fetch_block(rpc_url, *number)?;
        let gas_ok = match execute(witness, Some(cache.as_ref()), &block) {
            Ok(gas) => gas == block.header().gas_used(),
            Err(e) => {
                warn!(block = number, error = %e, "execution failed");
                false
            }
        };

        println!(
            "{:<10} {:>5} {:>11} {:>11} {:>14} {:>9}",
            number,
            block.body().transactions.len(),
            node_ids.len(),
            cached,
            shipped,
            gas_ok
        );
        total += 1;
        if gas_ok {
            ok_count += 1;
        }

        // Warm the cache with this block's nodes for the next blocks.
        for id in &node_ids {
            if let Some(bytes) = witness.nodes.get(id) {
                cache.insert(*id, bytes.clone());
            }
        }
    }

    info!(validated_ok = ok_count, total, cache_nodes = cache.len(), "window complete");
    Ok(())
}

/// Build the executor over `{cache ∪ witness}`, run the block, and return the
/// executed `gas_used`. The provider (and its cache borrow) is dropped before
/// returning, so the caller may mutate the cache afterwards.
fn execute(
    witness: &IndexedWitness,
    cache: Option<&dyn cache::NodeCache>,
    block: &RecoveredBlock<Block>,
) -> eyre::Result<u64> {
    let provider = match cache {
        Some(c) => WitnessStateProvider::with_cache(witness, c),
        None => WitnessStateProvider::new(witness),
    };
    let evm_config = EthEvmConfig::mainnet();
    let mut executor = evm_config.batch_executor(StateProviderDatabase::new(provider));
    let result = executor
        .execute_one(block)
        .wrap_err("stateless execution failed — witness/cache insufficient or block invalid")?;
    Ok(result.gas_used)
}

/// Fetch a block via `debug_getRawBlock` and recover its transaction senders.
fn fetch_block(rpc_url: &str, number: u64) -> eyre::Result<RecoveredBlock<Block>> {
    let raw = fetch_raw_block(rpc_url, number)?;
    let block = Block::decode(&mut raw.as_slice()).wrap_err("decode block RLP")?;
    RecoveredBlock::try_recover(block).wrap_err("recover transaction senders")
}

/// Command-line arguments for the windowed sweep.
struct WindowArgs {
    dir: PathBuf,
    policy: String,
    cap: usize,
}

impl WindowArgs {
    fn parse(args: impl Iterator<Item = String>) -> eyre::Result<Self> {
        let (mut dir, mut policy, mut cap) = (None, "lru".to_string(), 1_000_000usize);
        let mut it = args;
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--dir" => dir = Some(PathBuf::from(req(&mut it, "--dir")?)),
                "--policy" => policy = req(&mut it, "--policy")?,
                "--cap" => cap = req(&mut it, "--cap")?.parse().wrap_err("--cap must be an integer")?,
                other => eyre::bail!("unknown argument: {other}"),
            }
        }
        Ok(Self { dir: dir.ok_or_else(|| eyre::eyre!("--dir is required"))?, policy, cap })
    }
}

fn req(it: &mut impl Iterator<Item = String>, flag: &str) -> eyre::Result<String> {
    it.next().ok_or_else(|| eyre::eyre!("{flag} requires a value"))
}

/// Load every `witness_<N>.json` in `dir`, returning `(block_number, witness)`.
fn load_blocks(dir: &Path) -> eyre::Result<Vec<(u64, IndexedWitness)>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else { continue };
        let Some(num) = name.strip_prefix("witness_").and_then(|s| s.strip_suffix(".json")) else {
            continue;
        };
        let number = num.parse().map_err(|e| eyre::eyre!("bad block number in {name:?}: {e}"))?;
        out.push((number, IndexedWitness::load(&path)?));
    }
    Ok(out)
}

/// Extract `N` from a `witness_<N>.json` path.
fn parse_block_number(path: &Path) -> eyre::Result<u64> {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| eyre::eyre!("invalid witness path"))?;
    name.strip_prefix("witness_")
        .and_then(|s| s.strip_suffix(".json"))
        .ok_or_else(|| eyre::eyre!("witness file must be named witness_<N>.json"))?
        .parse()
        .map_err(|e| eyre::eyre!("bad block number in {name:?}: {e}"))
}

/// Fetch a block as raw RLP bytes via `debug_getRawBlock`.
fn fetch_raw_block(rpc_url: &str, number: u64) -> eyre::Result<Vec<u8>> {
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
    alloy_primitives::hex::decode(hex.trim_start_matches("0x")).wrap_err("bad raw block hex")
}

fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt().with_env_filter(filter).init();
}
