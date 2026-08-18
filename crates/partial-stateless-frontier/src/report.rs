//! What a run writes down, and what it refuses to claim.
//!
//! Two files, because they answer two different questions. The JSONL stream is one line per block
//! per policy — the population an analysis works from, and the only thing a distribution can be
//! computed out of. The summary is the run's own account of itself: what it was given, what it
//! produced, and which comparisons the numbers support.

use crate::{generate::BlockResult, policy::ArmKind};
use alloy_primitives::B256;
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

/// The JSONL stream, one block per line.
#[derive(Debug)]
pub struct RunReport {
    writer: BufWriter<File>,
    path: PathBuf,
}

impl RunReport {
    /// Creates the stream, replacing any file already at `path`.
    pub fn create(path: &Path) -> eyre::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(Self { writer: BufWriter::new(File::create(path)?), path: path.to_path_buf() })
    }

    /// Appends one block's result.
    pub fn append(&mut self, result: &BlockResult) -> eyre::Result<()> {
        serde_json::to_writer(&mut self.writer, result)?;
        self.writer.write_all(b"\n")?;
        Ok(())
    }

    /// Flushes and returns where the stream landed.
    pub fn finish(mut self) -> eyre::Result<PathBuf> {
        self.writer.flush()?;
        Ok(self.path)
    }
}

/// One arm's totals over the measured blocks.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct PolicySummary {
    /// Measured blocks this policy produced a sidecar for.
    pub blocks: u64,
    /// Total serialized sidecar bytes.
    pub total_sidecar_bytes: u64,
    /// Total parent-state witness bytes.
    pub total_witness_node_bytes: u64,
    /// Total bytecode bytes.
    pub total_witness_code_bytes: u64,
    /// Total missed accounts.
    pub total_missed_accounts: u64,
    /// Total missed storage slots.
    pub total_missed_storage: u64,
    /// Total missed bytecodes.
    pub total_missed_codes: u64,
    /// Total time from serialized sidecar to committed transition, summed over measured blocks.
    ///
    /// See [`PolicyBlockResult::sidecar_decode_and_commit_us`](crate::PolicyBlockResult) for what
    /// this boundary does and does not contain. It is not a standalone validation latency.
    pub total_sidecar_decode_and_commit_us: u64,
    /// The decode half of the figure above, which is the part that scales with witness size.
    pub total_sidecar_decode_us: u64,
    /// Flat cache size after the last measured block.
    pub final_cache_bytes: usize,
    /// Trie cache size after the last measured block.
    pub final_trie_cache_bytes: usize,
}

impl PolicySummary {
    /// Mean serialized sidecar size over the measured blocks.
    pub fn mean_sidecar_bytes(&self) -> f64 {
        if self.blocks == 0 {
            return 0.0
        }
        self.total_sidecar_bytes as f64 / self.blocks as f64
    }
}

/// The run's own account of itself.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RunSummary {
    /// Dataset this run read.
    pub dataset: String,
    /// The dataset's producer, carried forward so a report names the corpus's origin.
    pub dataset_producer: String,
    /// The commit the capturing node was built from, carried forward from the dataset's manifest.
    ///
    /// Copied here so this file alone answers "which code recorded the blocks these numbers came
    /// from". `null` means the capture ran from a build that carried no stamp, which is a fact
    /// about the corpus rather than about this run.
    pub dataset_build_commit: Option<String>,
    /// The commit *this* binary was built from, read at compile time.
    ///
    /// The corpus and the generator are two different pieces of code and either can move under a
    /// result. `null` means this build carried no stamp.
    pub generator_build_commit: Option<String>,
    /// Blocks replayed but not counted, so the caches reached their advertised windows.
    pub warmup_blocks: u64,
    /// Blocks counted.
    pub measured_blocks: u64,
    /// The block range the measured population covers.
    pub measured_range: Option<(u64, u64)>,
    /// keccak256 over the measured block hashes, in order.
    ///
    /// One value that answers "did every policy see the same blocks, and did the other host see
    /// them too". Two runs that agree here are comparing the same corpus; two that do not are not
    /// comparable at all, whatever their distributions look like.
    pub measured_block_set_digest: B256,
    /// Per-arm totals, keyed by `weak` or `account/storage`.
    pub policies: BTreeMap<String, PolicySummary>,
    /// Whether a no-cache baseline arm ran, so a Partial-versus-Weak comparison is available.
    ///
    /// Recorded rather than inferred from the key set, because "there is no `weak` row" and "the
    /// run was not asked for one" are the same absence and only one of them is a mistake.
    pub weak_baseline_present: bool,
    /// Total payload decode and admission time over the measured blocks, paid once per block and
    /// shared by every arm.
    pub total_block_admission_us: u64,
    /// Always false: an offline build time is not a production builder latency.
    pub builder_latency_eligible: bool,
    /// Always false: no figure here opens where a standalone validator's boundary opens.
    ///
    /// The closest thing this run produces is `total_sidecar_decode_and_commit_us` plus
    /// `total_block_admission_us`, which is a whole-block cost with no delivery path in it. An
    /// absolute standalone latency comes from a live `ps-replay` run.
    pub standalone_latency_eligible: bool,
    /// What the numbers above do support.
    pub supported_claims: Vec<String>,
}

impl RunSummary {
    /// Accumulates block results into a summary.
    pub fn accumulate(
        dataset: &Path,
        dataset_producer: String,
        dataset_build_commit: Option<String>,
        specs: &[ArmKind],
        results: &[BlockResult],
    ) -> Self {
        let mut policies = BTreeMap::new();
        for spec in specs {
            policies.insert(spec.label(), PolicySummary::default());
        }

        let mut measured_hashes = Vec::new();
        let mut warmup_blocks = 0u64;
        let mut measured_blocks = 0u64;
        let mut total_block_admission_us = 0u64;
        let mut first = None;
        let mut last = None;
        let weak_baseline_present = specs.contains(&ArmKind::Weak);

        for result in results {
            if !result.measured {
                warmup_blocks += 1;
                continue;
            }
            measured_blocks += 1;
            total_block_admission_us += result.block_admission_us;
            first.get_or_insert(result.block_number);
            last = Some(result.block_number);
            measured_hashes.extend_from_slice(result.block_hash.as_slice());
            for policy in &result.policies {
                let entry = policies.entry(policy.policy.clone()).or_default();
                entry.blocks += 1;
                entry.total_sidecar_bytes += policy.sidecar_bytes as u64;
                entry.total_witness_node_bytes += policy.witness_node_bytes as u64;
                entry.total_witness_code_bytes += policy.witness_code_bytes as u64;
                entry.total_missed_accounts += policy.missed_accounts as u64;
                entry.total_missed_storage += policy.missed_storage as u64;
                entry.total_missed_codes += policy.missed_codes as u64;
                entry.total_sidecar_decode_and_commit_us += policy.sidecar_decode_and_commit_us;
                entry.total_sidecar_decode_us += policy.sidecar_decode_us;
                entry.final_cache_bytes = policy.cache_bytes;
                entry.final_trie_cache_bytes = policy.trie_cache_bytes;
            }
        }

        Self {
            dataset: dataset.display().to_string(),
            dataset_producer,
            dataset_build_commit,
            generator_build_commit: option_env!("PS_BUILD_COMMIT").map(str::to_string),
            warmup_blocks,
            measured_blocks,
            measured_range: first.zip(last),
            measured_block_set_digest: alloy_primitives::keccak256(&measured_hashes),
            policies,
            weak_baseline_present,
            total_block_admission_us,
            builder_latency_eligible: false,
            standalone_latency_eligible: false,
            supported_claims: claims(weak_baseline_present),
        }
    }

    /// Writes the summary beside the JSONL stream.
    pub fn write(&self, path: &Path) -> eyre::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_vec_pretty(self)?)?;
        Ok(())
    }
}

/// What a run's numbers support, which depends on whether a baseline arm ran.
///
/// Written as a function rather than a literal because the Partial-versus-Weak claim is the one
/// that was previously recorded unconditionally while nothing ever produced a Weak arm — a report
/// asserting a comparison it did not make.
fn claims(weak_baseline_present: bool) -> Vec<String> {
    let mut claims = vec![
        "sidecar size for the same block under each arm".to_string(),
        "cache and trie-cache footprint under each arm".to_string(),
        "cache-miss counts under each arm".to_string(),
        "sidecar decode-and-commit cost per arm, which is the policy-dependent part of validation"
            .to_string(),
        "arm versus arm on one identical block set".to_string(),
    ];
    if weak_baseline_present {
        claims.push("Partial versus Weak on that same block set".to_string());
    }
    claims
}
