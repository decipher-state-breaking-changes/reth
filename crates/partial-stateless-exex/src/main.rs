//! Minimal partial-stateless ExEx sidecar producer.
//!
//! The binary installs one ExEx, observes canonical chain notifications, filters
//! target sets through a deterministic cache view, and writes minimal sidecar
//! artifacts. It does not implement validator consumption or wire transport.

mod bundle_summary;
mod cache_mode;
mod manifest;
mod proof_payload;
mod sidecar_writer;
mod target_source;

use crate::{
    bundle_summary::BundleSummary,
    cache_mode::CacheMode,
    manifest::{BlockManifestEntry, ManifestWriter, SidecarManifestEntry},
    proof_payload::build_state_multiproof_payload,
    sidecar_writer::SidecarWriter,
    target_source::{collect_targets, TargetSourceMode},
};
use futures::TryStreamExt;
use partial_stateless::{
    filter_missing_targets, CacheView, PartialExecutionWitness, PartialExecutionWitnessSidecar,
    SidecarBlockContext, WitnessPayload, WitnessTargets,
};
use reth_ethereum::{
    exex::{ExExContext, ExExEvent, ExExNotification},
    node::{api::FullNodeComponents, builder::NodeHandleFor, EthereumNode},
    primitives::AlloyBlockHeader,
};
use reth_provider::HeaderProvider;
use reth_storage_api::StateProviderFactory;
use tracing::{info, warn};

async fn partial_stateless_exex<Node: FullNodeComponents>(
    mut ctx: ExExContext<Node>,
) -> eyre::Result<()> {
    let manifest_writer = ManifestWriter::from_env();
    let sidecar_writer = SidecarWriter::from_env();
    let cache_mode = CacheMode::from_env()?;
    let target_source_mode = TargetSourceMode::from_env()?;
    info!(
        manifest_path = %manifest_writer.path().display(),
        sidecar_dir = %sidecar_writer.dir().display(),
        cache_mode = cache_mode.as_str(),
        target_source = target_source_mode.as_str(),
        "started partial-stateless ExEx producer"
    );

    while let Some(notification) = ctx.notifications.try_next().await? {
        match &notification {
            ExExNotification::ChainCommitted { new } => {
                info!(
                    committed_chain = ?new.range(),
                    tip = ?new.tip().num_hash(),
                    "partial-stateless producer received commit"
                );

                let bundle_summary = BundleSummary::from_execution_outcome(new.execution_outcome());
                let targets = collect_targets(target_source_mode, new.execution_outcome())?;

                for block in new.blocks_iter() {
                    write_block_artifacts(
                        &ctx,
                        &manifest_writer,
                        &sidecar_writer,
                        "commit",
                        block.number(),
                        format!("{:?}", block.hash()),
                        block.parent_hash(),
                        format!("{:?}", block.parent_hash()),
                        format!("{:?}", block.state_root()),
                        block.transaction_count(),
                        bundle_summary,
                        cache_mode,
                        &targets,
                    )?;
                }
            }
            ExExNotification::ChainReorged { old, new } => {
                warn!(
                    from_chain = ?old.range(),
                    to_chain = ?new.range(),
                    "partial-stateless producer received reorg"
                );

                let old_bundle_summary =
                    BundleSummary::from_execution_outcome(old.execution_outcome());
                let old_targets = collect_targets(target_source_mode, old.execution_outcome())?;

                for block in old.blocks_iter() {
                    write_block_artifacts(
                        &ctx,
                        &manifest_writer,
                        &sidecar_writer,
                        "reorg_old",
                        block.number(),
                        format!("{:?}", block.hash()),
                        block.parent_hash(),
                        format!("{:?}", block.parent_hash()),
                        format!("{:?}", block.state_root()),
                        block.transaction_count(),
                        old_bundle_summary,
                        cache_mode,
                        &old_targets,
                    )?;
                }

                let new_bundle_summary =
                    BundleSummary::from_execution_outcome(new.execution_outcome());
                let new_targets = collect_targets(target_source_mode, new.execution_outcome())?;

                for block in new.blocks_iter() {
                    write_block_artifacts(
                        &ctx,
                        &manifest_writer,
                        &sidecar_writer,
                        "reorg_new",
                        block.number(),
                        format!("{:?}", block.hash()),
                        block.parent_hash(),
                        format!("{:?}", block.parent_hash()),
                        format!("{:?}", block.state_root()),
                        block.transaction_count(),
                        new_bundle_summary,
                        cache_mode,
                        &new_targets,
                    )?;
                }
            }
            ExExNotification::ChainReverted { old } => {
                warn!(
                    reverted_chain = ?old.range(),
                    "partial-stateless producer received revert"
                );

                let bundle_summary = BundleSummary::from_execution_outcome(old.execution_outcome());
                let targets = collect_targets(target_source_mode, old.execution_outcome())?;

                for block in old.blocks_iter() {
                    write_block_artifacts(
                        &ctx,
                        &manifest_writer,
                        &sidecar_writer,
                        "revert",
                        block.number(),
                        format!("{:?}", block.hash()),
                        block.parent_hash(),
                        format!("{:?}", block.parent_hash()),
                        format!("{:?}", block.state_root()),
                        block.transaction_count(),
                        bundle_summary,
                        cache_mode,
                        &targets,
                    )?;
                }
            }
        }

        if let Some(committed_chain) = notification.committed_chain() {
            ctx.events.send(ExExEvent::FinishedHeight(committed_chain.tip().num_hash()))?;
        }
    }

    Ok(())
}

#[expect(clippy::too_many_arguments)]
fn write_block_artifacts<Node: FullNodeComponents>(
    ctx: &ExExContext<Node>,
    manifest_writer: &ManifestWriter,
    sidecar_writer: &SidecarWriter,
    event: &'static str,
    block_number: u64,
    block_hash: String,
    parent_hash_value: alloy_primitives::B256,
    parent_hash: String,
    state_root: String,
    tx_count: usize,
    bundle_summary: BundleSummary,
    cache_mode: CacheMode,
    targets: &WitnessTargets,
) -> eyre::Result<()> {
    if bundle_summary.block_count != 1 {
        warn!(
            event,
            block_number,
            bundle_block_count = bundle_summary.block_count,
            bundle_first_block = bundle_summary.first_block,
            bundle_last_block = bundle_summary.last_block,
            "skipping sidecar generation for multi-block execution outcome"
        );

        return manifest_writer.append_block_entry(BlockManifestEntry {
            event,
            block_number,
            block_hash,
            parent_hash,
            state_root,
            tx_count,
            bundle_summary,
            sidecar: None,
        });
    }

    let parent_state_root = ctx
        .provider()
        .header(parent_hash_value)?
        .map(|header| format!("{:?}", header.state_root()));
    let cache_view = cache_mode.view_for_block(block_number.saturating_sub(1));
    let filtered_targets = filter_missing_targets(targets, &cache_view);
    let payload = if filtered_targets.missing_targets.counts().accounts == 0
        && filtered_targets.missing_targets.counts().storage_slots == 0
        && filtered_targets.missing_targets.counts().code_hashes == 0
    {
        WitnessPayload::NoneSkeletonOnly
    } else {
        let state_provider = ctx.provider().state_by_block_hash(parent_hash_value)?;
        build_state_multiproof_payload(state_provider.as_ref(), &filtered_targets.missing_targets)?
    };

    let sidecar_result = sidecar_writer.write(&PartialExecutionWitnessSidecar {
        envelope: SidecarBlockContext {
            event,
            block_number,
            block_hash: block_hash.clone(),
            parent_hash: parent_hash.clone(),
            parent_state_root,
            post_state_root: state_root.clone(),
            tx_count,
        },
        cache_descriptor: cache_view.descriptor().clone(),
        partial_execution_witness: PartialExecutionWitness {
            missing_targets: filtered_targets.missing_targets,
            stats: filtered_targets.stats,
            payload,
        },
    })?;

    manifest_writer.append_block_entry(BlockManifestEntry {
        event,
        block_number,
        block_hash,
        parent_hash,
        state_root,
        tx_count,
        bundle_summary,
        sidecar: Some(SidecarManifestEntry {
            path: sidecar_result.path.display().to_string(),
            target_source: sidecar_result.target_source,
            payload_kind: sidecar_result.payload_kind,
            payload_total_bytes: sidecar_result.payload_total_bytes,
            target_counts: sidecar_result.target_counts,
            target_stats: sidecar_result.target_stats,
        }),
    })
}

fn main() -> eyre::Result<()> {
    reth_ethereum::cli::Cli::parse_args().run(async move |builder, _| {
        let handle: NodeHandleFor<EthereumNode> = builder
            .node(EthereumNode::default())
            .install_exex("partial-stateless", async move |ctx| Ok(partial_stateless_exex(ctx)))
            .launch()
            .await?;

        handle.wait_for_node_exit().await
    })
}
