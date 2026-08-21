//! Measures which contract code chunks a recorded corpus actually executes.
//!
//! The tree study's largest assumption is how much of a contract a call runs, because EIP-7864 and
//! EIP-6800 carry only the chunks that ran while the MPT carries whole bytecode. This binary
//! replaces the assumption with a measurement over the same corpus, using no state database, no
//! provider, and no network: a block's recorded access set is its complete read set, so the block
//! re-executes against the access set itself with an inspector watching the program counter.
//!
//! Its output feeds `ps-tree-study --coverage`.

use alloy_consensus::Header;
use alloy_primitives::B256;
use alloy_rlp::Decodable;
use eyre::{bail, Context, Result};
use partial_stateless::{
    full_witness_sidecar_from_nodes,
    witness_check::materialize_sidecar_witness_after_prefilter_with_cache, BlockTransitionRef,
    CacheConfig,
};
use partial_stateless_treestudy::{
    corpus::Corpus,
    coverage::{ChunkInspector, CodeCoverage, WitnessDatabase},
};
use partial_stateless_validator::admission::UntrustedAdmission;
use reth_chainspec::{ChainSpec, MAINNET};
use reth_ethereum_consensus::EthBeaconConsensus;
use reth_evm::{execute::BlockExecutor, ConfigureEvm};
use reth_evm_ethereum::EthEvmConfig;
use reth_primitives_traits::SealedHeader;
use revm::database::State;
use std::{collections::BTreeMap, fs, path::PathBuf, sync::Arc, time::Instant};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let options = Options::parse()?;
    let corpus = Corpus::open(&options.dataset)?;
    let (first, last) = corpus.range();
    tracing::info!(blocks = corpus.len(), first, last, "corpus opened");

    let chain_spec: Arc<ChainSpec> = MAINNET.clone();
    let consensus = EthBeaconConsensus::new(chain_spec.clone());
    let admission = UntrustedAdmission::new(chain_spec.as_ref(), &consensus);
    let evm_config = EthEvmConfig::new(chain_spec.clone());

    let mut coverage = CodeCoverage::default();
    let started = Instant::now();
    let (mut executed, mut skipped) = (0usize, 0usize);
    let mut parent: Option<SealedHeader> = None;

    let cache_config = CacheConfig::default();
    corpus.for_each_with_witness(options.limit, |block| {
        let Some(payload_json) = block.payload_json.as_ref() else {
            // A record without a payload cannot be re-executed. Counted rather than ignored: the
            // coverage figure is a mean over blocks, and a silently smaller denominator would move
            // it.
            skipped += 1;
            return Ok(())
        };
        let payload = serde_json::from_slice(payload_json)
            .with_context(|| format!("decoding the payload of block {}", block.number))?;
        if parent.is_none() {
            // Only the first record needs this. Every block after it is admitted against the header
            // this run derived from the block below, which is stronger evidence than a recorded
            // one.
            let header =
                Header::decode(&mut block.parent_header.as_slice()).with_context(|| {
                    format!("decoding the recorded parent of block {}", block.number)
                })?;
            parent = Some(SealedHeader::seal_slow(header));
        }
        let admitted = match admission.admit(payload, parent.as_ref()) {
            Ok(admitted) => admitted,
            Err(err) => {
                return Err(eyre::eyre!(
                    "block {} was refused before execution: {err}",
                    block.number
                ))
            }
        };
        let recovered = admitted.block;
        parent = Some(recovered.clone_sealed_header());

        // Parent state, from the recorded transition witness. Built through the production
        // construction path so the values this executes against are the values a validator would
        // have been handed.
        let sidecar = full_witness_sidecar_from_nodes(
            block.parent_state_root,
            BlockTransitionRef {
                block_number: recovered.header().number,
                block_hash: block.hash,
                parent_hash: block.parent_hash,
                parent_state_root: block.parent_state_root,
                expected_state_root: recovered.header().state_root,
                ancestor_headers: &[],
            },
            &block.accessed,
            block.transition_nodes.clone(),
            &cache_config,
        )
        .with_context(|| format!("building the parent-state witness of block {}", block.number))?;
        let materialized = materialize_sidecar_witness_after_prefilter_with_cache(&sidecar, None)
            .map_err(|err| eyre::eyre!("block {}: {err}", block.number))?;

        let block_hashes: BTreeMap<u64, B256> = block.ancestor_hashes.iter().copied().collect();
        let db = WitnessDatabase::new(
            materialized.accounts,
            materialized.storage,
            materialized.codes,
            block_hashes,
        );
        let mut state = State::builder().with_database(db).with_bundle_update().build();
        let mut inspector = ChunkInspector::new();
        {
            let evm_env = evm_config.evm_env(recovered.header())?;
            let evm = evm_config.evm_with_env_and_inspector(&mut state, evm_env, &mut inspector);
            let ctx = evm_config.context_for_block(recovered.sealed_block())?;
            let executor = evm_config.create_executor(evm, ctx);
            let result = executor
                .execute_block(recovered.transactions_recovered())
                .map_err(|err| eyre::eyre!("block {} failed to execute: {err}", block.number))?;
            // Gas is the cheapest whole-block check that this replay went where the real block
            // went. Coverage read off a divergent execution would be a measurement of
            // something else, and would look exactly as plausible as a correct one.
            if result.gas_used != recovered.header().gas_used {
                bail!(
                    "block {} replayed to {} gas against the header's {}; the coverage this would \
                     record is not this block's",
                    block.number,
                    result.gas_used,
                    recovered.header().gas_used
                );
            }
        }
        coverage.merge(&inspector.finish(&block.accessed));
        executed += 1;
        if executed.is_multiple_of(100) {
            tracing::info!(
                executed,
                skipped,
                bytecodes = coverage.total_chunks.len(),
                fraction = coverage.overall_fraction(),
                elapsed_s = started.elapsed().as_secs(),
                "progress"
            );
        }
        Ok(())
    })?;

    if executed == 0 {
        bail!("no block in the corpus could be executed");
    }
    let (entered, read_only) = coverage.entered_and_read_only();
    tracing::info!(
        executed,
        skipped,
        entered,
        read_only,
        overall_fraction = coverage.overall_fraction(),
        elapsed_s = started.elapsed().as_secs(),
        "coverage measured"
    );

    if let Some(parent) = options.out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&options.out, serde_json::to_vec(&coverage)?)?;
    println!();
    println!("blocks executed        {executed}");
    println!("blocks skipped         {skipped}");
    println!("bytecodes seen         {}", coverage.total_chunks.len());
    println!("  entered              {entered}");
    println!("  read but never run   {read_only}");
    println!("chunks run / chunks in entered bytecodes  {:.4}", coverage.overall_fraction());
    println!();
    tracing::info!(out = ?options.out, "written");
    Ok(())
}

struct Options {
    dataset: PathBuf,
    out: PathBuf,
    limit: Option<usize>,
}

impl Options {
    fn parse() -> Result<Self> {
        let (mut dataset, mut out, mut limit) = (None, None, None);
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            let mut value = || -> Result<String> {
                args.next().ok_or_else(|| eyre::eyre!("{arg} needs a value"))
            };
            match arg.as_str() {
                "--dataset" => dataset = Some(PathBuf::from(value()?)),
                "--out" => out = Some(PathBuf::from(value()?)),
                "--limit" => limit = Some(value()?.parse()?),
                "--help" | "-h" => {
                    eprintln!("{USAGE}");
                    std::process::exit(0);
                }
                other => return Err(eyre::eyre!("unknown argument `{other}`\n{USAGE}")),
            }
        }
        Ok(Self {
            dataset: dataset.ok_or_else(|| eyre::eyre!("--dataset is required\n{USAGE}"))?,
            out: out.ok_or_else(|| eyre::eyre!("--out is required\n{USAGE}"))?,
            limit,
        })
    }
}

const USAGE: &str = "\
ps-code-coverage --dataset <corpus dir> --out <coverage.json> [--limit N]

Re-executes every recorded block against its own access set and records which 31-byte code chunks
the EVM entered. The result feeds `ps-tree-study --coverage`.
";
