//! `ps-validator` — reconstruct account state from a single block's witness and
//! report how much resolves, purely from `{cache ∪ witness}` (here: witness
//! only, empty cache), against real `debug_executionWitness` data.
//!
//! Usage:
//!   ps-validator <witness_N.json>
//!
//! For the cache-vs-witness sweep across many blocks, see the `cache-experiment`
//! binary. For block re-execution + state-root verification, see `validate`.

use partial_stateless_validator::{db::WitnessStateProvider, witness::IndexedWitness};
use reth_revm::database::EvmStateProvider;
use std::path::PathBuf;

fn main() -> eyre::Result<()> {
    init_tracing();

    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| eyre::eyre!("usage: ps-validator <witness_N.json>"))?;

    let witness = IndexedWitness::load(&path)?;
    tracing::info!(
        nodes = witness.nodes.len(),
        codes = witness.codes.len(),
        keys = witness.keys.len(),
        headers = witness.block_hashes.len(),
        pre_state_root = %witness.pre_state_root,
        "loaded witness"
    );

    let addresses = witness.addresses();
    let provider = WitnessStateProvider::new(&witness);

    let (mut resolved, mut absent, mut errored) = (0u64, 0u64, 0u64);
    let mut samples = 0u64;
    for address in &addresses {
        match provider.basic_account(address) {
            Ok(Some(account)) => {
                resolved += 1;
                if samples < 5 {
                    tracing::info!(
                        %address,
                        nonce = account.nonce,
                        balance = %account.balance,
                        bytecode_hash = ?account.bytecode_hash,
                        "resolved account via EvmStateProvider"
                    );
                    samples += 1;
                }
            }
            Ok(None) => absent += 1,
            Err(e) => {
                errored += 1;
                if errored <= 5 {
                    tracing::warn!(%address, error = %e, "read failed (node not in cache+witness)");
                }
            }
        }
    }

    tracing::info!(
        total_addresses = addresses.len(),
        resolved,
        absent,
        errored,
        "state reconstruction summary (via EvmStateProvider)"
    );

    Ok(())
}

fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt().with_env_filter(filter).init();
}
