use partial_stateless::{
    PartialExecutionWitnessSidecar, TargetCounts, TargetStats, WitnessPayload,
};
use std::{
    env,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

const DEFAULT_SIDECAR_DIR: &str = "/var/lib/reth/cache-vops/sidecars";

#[derive(Debug)]
pub(crate) struct SidecarWriter {
    dir: PathBuf,
}

#[derive(Debug)]
pub(crate) struct SidecarWriteResult {
    pub(crate) path: PathBuf,
    pub(crate) target_source: &'static str,
    pub(crate) payload_kind: &'static str,
    pub(crate) payload_total_bytes: Option<usize>,
    pub(crate) target_counts: TargetCounts,
    pub(crate) target_stats: TargetStats,
}

impl SidecarWriter {
    pub(crate) fn from_env() -> Self {
        let dir = env::var_os("CACHE_VOPS_SIDECAR_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_SIDECAR_DIR));
        Self { dir }
    }

    pub(crate) fn dir(&self) -> &Path {
        &self.dir
    }

    pub(crate) fn write(
        &self,
        sidecar: &PartialExecutionWitnessSidecar,
    ) -> eyre::Result<SidecarWriteResult> {
        fs::create_dir_all(&self.dir)?;

        let path = self.dir.join(sidecar_file_name(
            sidecar.envelope.event,
            sidecar.envelope.block_number,
            &sidecar.envelope.block_hash,
        ));

        let mut file = File::create(&path)?;
        serde_json::to_writer_pretty(&mut file, &sidecar.to_json_value())?;
        file.write_all(b"\n")?;

        Ok(SidecarWriteResult {
            path,
            target_source: sidecar.partial_execution_witness.missing_targets.source.as_str(),
            payload_kind: sidecar.partial_execution_witness.payload.kind(),
            payload_total_bytes: payload_total_bytes(&sidecar.partial_execution_witness.payload),
            target_counts: sidecar.partial_execution_witness.missing_targets.counts(),
            target_stats: sidecar.partial_execution_witness.stats,
        })
    }
}

fn payload_total_bytes(payload: &WitnessPayload) -> Option<usize> {
    match payload {
        WitnessPayload::NoneSkeletonOnly => None,
        WitnessPayload::StateMultiproofV1(payload) => Some(payload.stats.total_bytes()),
    }
}

fn sidecar_file_name(event: &str, block_number: u64, block_hash: &str) -> String {
    let hash = block_hash.strip_prefix("0x").unwrap_or(block_hash);
    let short_hash = hash.get(..16).unwrap_or(hash);
    format!("block_{block_number}_{short_hash}_{event}.json")
}
