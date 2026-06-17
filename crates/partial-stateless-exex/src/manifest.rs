use crate::bundle_summary::BundleSummary;
use partial_stateless::{TargetCounts, TargetStats};
use serde_json::json;
use std::{
    env,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

const DEFAULT_MANIFEST_PATH: &str = "/var/lib/reth/cache-vops/manifests/events.jsonl";

#[derive(Debug)]
pub(crate) struct ManifestWriter {
    path: PathBuf,
}

#[derive(Debug)]
pub(crate) struct BlockManifestEntry {
    pub(crate) event: &'static str,
    pub(crate) block_number: u64,
    pub(crate) block_hash: String,
    pub(crate) parent_hash: String,
    pub(crate) state_root: String,
    pub(crate) tx_count: usize,
    pub(crate) bundle_summary: BundleSummary,
    pub(crate) sidecar: Option<SidecarManifestEntry>,
}

#[derive(Debug)]
pub(crate) struct SidecarManifestEntry {
    pub(crate) path: String,
    pub(crate) target_source: &'static str,
    pub(crate) payload_kind: &'static str,
    pub(crate) payload_total_bytes: Option<usize>,
    pub(crate) target_counts: TargetCounts,
    pub(crate) target_stats: TargetStats,
}

impl ManifestWriter {
    pub(crate) fn from_env() -> Self {
        let path = env::var_os("CACHE_VOPS_MANIFEST_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_MANIFEST_PATH));
        Self { path }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn append_block_entry(&self, entry: BlockManifestEntry) -> eyre::Result<()> {
        self.append_line(json!({
            "schema_version": 1,
            "event": entry.event,
            "observed_at_unix_secs": unix_timestamp_secs(),
            "block_number": entry.block_number,
            "block_hash": entry.block_hash,
            "parent_hash": entry.parent_hash,
            "state_root": entry.state_root,
            "tx_count": entry.tx_count,
            "bundle_first_block": entry.bundle_summary.first_block,
            "bundle_last_block": entry.bundle_summary.last_block,
            "bundle_block_count": entry.bundle_summary.block_count,
            "bundle_changed_accounts": entry.bundle_summary.changed_accounts,
            "bundle_changed_account_infos": entry.bundle_summary.changed_account_infos,
            "bundle_changed_storage_slots": entry.bundle_summary.changed_storage_slots,
            "bundle_code_hash_changed_accounts": entry.bundle_summary.code_hash_changed_accounts,
            "bundle_contract_bytecodes": entry.bundle_summary.contract_bytecodes,
            "bundle_state_size": entry.bundle_summary.state_size,
            "bundle_reverts_size": entry.bundle_summary.reverts_size,
            "sidecar_path": entry.sidecar.as_ref().map(|sidecar| sidecar.path.as_str()),
            "sidecar_target_source": entry.sidecar.as_ref().map(|sidecar| sidecar.target_source),
            "sidecar_payload_kind": entry.sidecar.as_ref().map(|sidecar| sidecar.payload_kind),
            "sidecar_payload_total_bytes": entry.sidecar.as_ref().and_then(|sidecar| sidecar.payload_total_bytes),
            "sidecar_target_counts": entry.sidecar.as_ref().map(|sidecar| sidecar.target_counts.to_json_value()),
            "sidecar_target_stats": entry.sidecar.as_ref().map(|sidecar| sidecar.target_stats.to_json_value()),
        }))
    }

    fn append_line(&self, value: serde_json::Value) -> eyre::Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut file = OpenOptions::new().create(true).append(true).open(&self.path)?;
        serde_json::to_writer(&mut file, &value)?;
        file.write_all(b"\n")?;
        Ok(())
    }
}

fn unix_timestamp_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}
