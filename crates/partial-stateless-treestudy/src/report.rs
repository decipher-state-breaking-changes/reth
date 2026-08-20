//! What a run produces, and what it is careful not to claim.

use crate::study::{ArmSpec, BlockResult, StudyParams};
use serde::Serialize;
use std::collections::BTreeMap;

/// Per-arm totals over the measured block set.
#[derive(Debug, Clone, Serialize)]
pub struct ArmSummary {
    /// Arm name.
    pub arm: String,
    /// Blocks measured.
    pub blocks: usize,
    /// Mean accounts missed per block.
    pub mean_missed_accounts: f64,
    /// Mean storage slots missed per block.
    pub mean_missed_storage: f64,
    /// Mean bytecodes missed per block.
    pub mean_missed_codes: f64,
    /// Mean raw bytecode bytes missed per block, as the MPT ships them.
    pub mean_missed_code_bytes: f64,
    /// Mean raw bytecode bytes the study could not place in the tree for want of an owner.
    pub mean_unowned_code_bytes: f64,
    /// Mean binary-tree witness bytes per block.
    pub mean_binary_total_bytes: f64,
    /// Mean Verkle witness bytes per block.
    pub mean_verkle_total_bytes: f64,
    /// Mean accounts whose code leaves the tree witness opened.
    pub mean_code_bearing_accounts: f64,
    /// Mean hexary trie nodes the MPT path model predicts.
    pub mean_mpt_model_nodes: f64,
    /// Mean stems the miss set opened.
    pub mean_stems_opened: f64,
    /// Mean stems the receiver retained.
    pub mean_retained_stems: f64,
    /// Mean binary path nodes the retained frontier removed.
    pub mean_binary_nodes_held: f64,
    /// Median binary-tree witness bytes, code included.
    pub median_binary_total_bytes: f64,
    /// Median Verkle witness bytes, code included.
    pub median_verkle_total_bytes: f64,
}

/// Per-arm ratios against the no-cache baseline, paired block by block.
#[derive(Debug, Clone, Serialize)]
pub struct ArmRatios {
    /// Arm name.
    pub arm: String,
    /// Paired median of `weak / arm` binary-tree bytes.
    pub binary_vs_weak_median: f64,
    /// Paired median of `weak / arm` Verkle bytes.
    pub verkle_vs_weak_median: f64,
    /// Paired median of `weak / arm` predicted MPT nodes.
    pub mpt_model_nodes_vs_weak_median: f64,
}

/// The whole run.
#[derive(Debug, Clone, Serialize)]
pub struct RunReport {
    /// Corpus the run read.
    pub corpus: String,
    /// Block range measured, after warm-up.
    pub measured_range: (u64, u64),
    /// Blocks measured.
    pub measured_blocks: usize,
    /// Blocks used only to fill the caches.
    pub warmup_blocks: usize,
    /// Modelled state sizes and layout.
    pub params: ReportedParams,
    /// Per-arm totals.
    pub arms: Vec<ArmSummary>,
    /// Per-arm ratios against the no-cache arm.
    pub ratios: Vec<ArmRatios>,
    /// The recorded run this one was checked against, when it was.
    pub miss_set_check: Option<MissSetCheck>,
    /// What the numbers above do and do not support.
    pub disclaimers: Vec<String>,
}

/// Evidence that this run priced the same workload the recorded run measured.
#[derive(Debug, Clone, Serialize)]
pub struct MissSetCheck {
    /// Keccak of the recorded run's own bytes.
    pub recorded_digest: String,
    /// How many (block, arm) pairs were compared and agreed.
    pub pairs_checked: usize,
}

/// The run's parameters, restated in the report so a result is never separated from them.
#[derive(Debug, Clone, Serialize)]
pub struct ReportedParams {
    /// Accounts in the modelled state.
    pub account_population: u64,
    /// Storage slots in the modelled state.
    pub storage_population: u64,
    /// Stems the whole tree was modelled to hold.
    pub total_stem_population: u64,
    /// Extra occupied suffixes assumed in an opened stem outside an account header.
    pub stem_occupancy_outside_header: u32,
    /// Extra occupied suffixes assumed in an account's header stem.
    pub stem_occupancy_in_header: u32,
    /// Which EIP-7864 header layout was used.
    pub header_layout: String,
    /// Fraction of a contract's code chunks a call was assumed to run.
    pub code_coverage: f64,
    /// Storage-trie size the MPT calibration used.
    pub mpt_storage_trie_population: u64,
}

impl RunReport {
    /// Folds per-block results into a report.
    pub fn build(
        corpus: String,
        params: &StudyParams,
        header_layout: &str,
        specs: &[ArmSpec],
        results: &BTreeMap<String, Vec<BlockResult>>,
        warmup_blocks: usize,
        miss_set_check: Option<(String, usize)>,
    ) -> Self {
        let mut arms = Vec::new();
        for spec in specs {
            let Some(rows) = results.get(&spec.name) else { continue };
            if rows.is_empty() {
                continue
            }
            arms.push(summarise(&spec.name, rows));
        }

        let baseline = results.get("weak");
        let mut ratios = Vec::new();
        if let Some(baseline) = baseline {
            for spec in specs {
                if spec.name == "weak" {
                    continue
                }
                let Some(rows) = results.get(&spec.name) else { continue };
                ratios.push(ArmRatios {
                    arm: spec.name.clone(),
                    binary_vs_weak_median: paired_median(baseline, rows, |r| {
                        r.binary_total_bytes() as f64
                    }),
                    verkle_vs_weak_median: paired_median(baseline, rows, |r| {
                        r.verkle_total_bytes() as f64
                    }),
                    mpt_model_nodes_vs_weak_median: paired_median(baseline, rows, |r| {
                        r.mpt_model_nodes as f64
                    }),
                });
            }
        }

        let measured = results.values().next().map(Vec::as_slice).unwrap_or_default();
        let measured_range = measured
            .first()
            .zip(measured.last())
            .map(|(first, last)| (first.block_number, last.block_number))
            .unwrap_or((0, 0));

        Self {
            corpus,
            measured_range,
            measured_blocks: measured.len(),
            warmup_blocks,
            params: ReportedParams {
                account_population: params.account_population,
                storage_population: params.storage_population,
                total_stem_population: params.total_stem_population,
                stem_occupancy_outside_header: params.effective_occupancy().outside_header,
                stem_occupancy_in_header: params.effective_occupancy().in_header,
                header_layout: header_layout.to_string(),
                code_coverage: params.code_coverage,
                mpt_storage_trie_population: params.mpt_storage_trie_population,
            },
            arms,
            ratios,
            miss_set_check: miss_set_check.map(|(recorded_digest, pairs_checked)| MissSetCheck {
                recorded_digest,
                pairs_checked,
            }),
            disclaimers: disclaimers(),
        }
    }
}

fn summarise(arm: &str, rows: &[BlockResult]) -> ArmSummary {
    let n = rows.len() as f64;
    let mean = |f: fn(&BlockResult) -> f64| rows.iter().map(f).sum::<f64>() / n;
    ArmSummary {
        arm: arm.to_string(),
        blocks: rows.len(),
        mean_missed_accounts: mean(|r| r.missed_accounts as f64),
        mean_missed_storage: mean(|r| r.missed_storage as f64),
        mean_missed_codes: mean(|r| r.missed_codes as f64),
        mean_missed_code_bytes: mean(|r| r.missed_code_bytes as f64),
        mean_unowned_code_bytes: mean(|r| r.unowned_code_bytes as f64),
        mean_binary_total_bytes: mean(|r| r.binary_total_bytes() as f64),
        mean_verkle_total_bytes: mean(|r| r.verkle_total_bytes() as f64),
        mean_code_bearing_accounts: mean(|r| r.code_bearing_accounts as f64),
        mean_mpt_model_nodes: mean(|r| r.mpt_model_nodes as f64),
        mean_stems_opened: mean(|r| r.binary_stems_opened as f64),
        mean_retained_stems: mean(|r| r.binary_retained_stems as f64),
        mean_binary_nodes_held: mean(|r| r.binary.nodes_held_by_cache as f64),
        median_binary_total_bytes: median(rows.iter().map(|r| r.binary_total_bytes() as f64)),
        median_verkle_total_bytes: median(rows.iter().map(|r| r.verkle_total_bytes() as f64)),
    }
}

/// Median of `weak / arm`, taken per block rather than as a ratio of two means.
///
/// Same-block pairing is the only comparison the corpus supports: two arms see the same block, so
/// the block's own size drops out. A ratio of separately averaged totals would be dominated by
/// whichever blocks happened to be large.
fn paired_median(
    baseline: &[BlockResult],
    arm: &[BlockResult],
    value: impl Fn(&BlockResult) -> f64,
) -> f64 {
    let mut ratios: Vec<f64> = baseline
        .iter()
        .zip(arm.iter())
        .filter(|(base, other)| base.block_number == other.block_number)
        .filter_map(|(base, other)| {
            let denominator = value(other);
            (denominator > 0.0).then(|| value(base) / denominator)
        })
        .collect();
    median_of(&mut ratios)
}

fn median(values: impl Iterator<Item = f64>) -> f64 {
    let mut collected: Vec<f64> = values.collect();
    median_of(&mut collected)
}

fn median_of(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN
    }
    values.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

fn disclaimers() -> Vec<String> {
    vec![
        "Witness bytes for the binary and Verkle arms are structural: they count the nodes, stems, \
         and values a multiproof carries under each scheme's own rules. They are not produced by \
         running either scheme's cryptography."
            .into(),
        "Neither tree arm reports verification time. A hash-based binary proof and an elliptic-curve \
         Verkle proof do not have comparable verification costs, and this study measures neither."
            .into(),
        "The MPT column is a path model kept for calibration against the measured corpus. The \
         measured MPT witness bytes are the ones the recorded run produced; nothing here replaces \
         them."
            .into(),
        "State sizes above the corpus's own keys are modelled as a uniform population of the \
         measured size. Path depth grows as log of that size, so the sweep over it is the \
         sensitivity this substitution has to answer for."
            .into(),
        "Every arm sees the identical block sequence and the identical miss set. Differences \
         between arms are therefore attributable to the commitment scheme and the cache policy, and \
         to nothing about block composition."
            .into(),
    ]
}
