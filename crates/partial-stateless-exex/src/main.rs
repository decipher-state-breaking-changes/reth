//! `reth-partial-stateless` — a full Ethereum node with the partial-statelessness ExEx installed.
//!
//! Run with:
//!   cargo run -p partial-stateless-exex -- node --chain mainnet --datadir /path/to/data
//!
//! Everything the ExEx does lives in the library half of this crate; this file exists only to pick
//! the allocator and install the extension, both of which are binary concerns.

#[global_allocator]
static ALLOC: reth_cli_util::allocator::Allocator = reth_cli_util::allocator::new_allocator();

// Required for jemalloc to override the allocator on supported Unix platforms.
#[cfg(unix)]
use reth_cli_util::allocator::tikv_jemalloc_sys as _;

use partial_stateless_exex::{partial_stateless_exex, CacheConfig};
use reth_ethereum::node::{builder::NodeHandleFor, EthereumNode};

fn main() -> eyre::Result<()> {
    reth_ethereum::cli::Cli::parse_args().run(async move |builder, _| {
        let config = CacheConfig::default();

        let handle: NodeHandleFor<EthereumNode> = builder
            .node(EthereumNode::default())
            .install_exex("partial-stateless", move |ctx| async move {
                Ok(partial_stateless_exex(ctx, config))
            })
            .launch()
            .await?;

        handle.wait_for_node_exit().await
    })
}
