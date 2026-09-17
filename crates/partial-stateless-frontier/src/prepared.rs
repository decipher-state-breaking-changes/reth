//! Policy-neutral payloads and generated sidecars, replayed one policy per process.
//!
//! These files are benchmark inputs, not network checkpoints. The first parent header is an
//! explicitly recorded trust anchor; every subsequent header, state root and cache anchor is
//! verified from the payload and sidecar. Warm-up transitions use the same commit path as samples.

use crate::policy::ArmKind;
use alloy_consensus::Header;
use alloy_primitives::{keccak256, Bytes, Keccak256, B256};
use alloy_rpc_types_engine::ExecutionData;
use bincode::Options as _;
use partial_stateless::{
    BlockContext, CacheReadinessTracker, CacheTrieRepr, PartialStatelessSidecar,
    PartialTrieNodeCache, WarmSetShrinkPolicy,
};
use partial_stateless_validator::{
    verify_and_apply_sidecar, CoordinatedPair, RetentionDepth, SidecarReexecLimits,
    TrieCacheDisposition, UndoLayout, UntrustedAdmission, ValidatorRules,
};
use reth_chainspec::MAINNET;
use reth_ethereum_consensus::EthBeaconConsensus;
use reth_evm_ethereum::EthEvmConfig;
use reth_primitives_traits::{AlloyBlockHeader, SealedHeader};
use serde::{Deserialize, Serialize};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

const MAX_INPUT_BYTES: u64 = 512 * 1024 * 1024;

/// Writes a complete manifest only after every warm-up and measured input has been saved.
pub struct PreparedWriter {
    directory: PathBuf,
    manifest: PreparedManifest,
}

impl PreparedWriter {
    pub fn new(
        directory: &Path,
        arm: ArmKind,
        chain: String,
        warmup: u64,
        samples: u64,
        parent_header: Bytes,
        warm_shrink_blocks: Option<u64>,
    ) -> eyre::Result<Self> {
        std::fs::create_dir(directory)?;
        Ok(Self {
            directory: directory.into(),
            manifest: PreparedManifest {
                schema_version: 1,
                arm: arm.label(),
                chain,
                warmup,
                samples,
                parent_header,
                warm_shrink_blocks,
                build_commit: option_env!("PS_BUILD_COMMIT").map(str::to_owned),
                inputs: Vec::new(),
            },
        })
    }

    pub fn append(
        &mut self,
        number: u64,
        hash: B256,
        payload_json: &[u8],
        sidecar: &[u8],
    ) -> eyre::Result<()> {
        let index = self.manifest.inputs.len();
        let input =
            PreparedInput { payload_json: payload_json.to_vec(), sidecar: sidecar.to_vec() };
        let encoded = bincode::serialize(&input)?;
        if encoded.len() as u64 > MAX_INPUT_BYTES {
            eyre::bail!("prepared input exceeds size limit")
        }
        let path = self.directory.join(format!("{index:08}.bin"));
        OpenOptions::new().create_new(true).write(true).open(path)?.write_all(&encoded)?;
        self.manifest.inputs.push(InputIdentity { number, hash, digest: keccak256(&encoded) });
        Ok(())
    }

    pub fn finish(self) -> eyre::Result<()> {
        self.manifest.validate()?;
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(self.directory.join("manifest.json"))?;
        serde_json::to_writer_pretty(file, &self.manifest)?;
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PreparedManifest {
    pub schema_version: u64,
    pub arm: String,
    pub chain: String,
    pub warmup: u64,
    pub samples: u64,
    pub parent_header: Bytes,
    pub warm_shrink_blocks: Option<u64>,
    pub build_commit: Option<String>,
    pub inputs: Vec<InputIdentity>,
}

impl PreparedManifest {
    fn validate(&self) -> eyre::Result<()> {
        let arm: ArmKind = self.arm.parse()?;
        if self.schema_version != 1 || self.samples == 0 || self.warmup < arm.warmup_floor() {
            eyre::bail!("invalid prepared schema, sample count or inclusive-window warm-up")
        }
        let count = self
            .warmup
            .checked_add(self.samples)
            .ok_or_else(|| eyre::eyre!("input count overflow"))?;
        if self.inputs.len() as u64 != count {
            eyre::bail!("prepared input count mismatch")
        }
        if self.inputs.windows(2).any(|w| w[0].number.checked_add(1) != Some(w[1].number)) {
            eyre::bail!("prepared heights are not consecutive")
        }
        parse_warm_shrink(
            self.warm_shrink_blocks.map_or_else(|| "never".into(), |n| n.to_string()).as_str(),
        )?;
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InputIdentity {
    pub number: u64,
    pub hash: B256,
    pub digest: B256,
}

#[derive(Debug, Serialize, Deserialize)]
struct PreparedInput {
    payload_json: Vec<u8>,
    sidecar: Vec<u8>,
}

/// Shared strict parsing for the generation and validation passes.
pub fn parse_warm_shrink(raw: &str) -> eyre::Result<WarmSetShrinkPolicy> {
    raw.parse().map_err(eyre::Report::msg)
}

fn read_input(
    directory: &Path,
    index: usize,
    identity: &InputIdentity,
) -> eyre::Result<PreparedInput> {
    let file = File::open(directory.join(format!("{index:08}.bin")))?;
    let mut bytes = Vec::new();
    file.take(MAX_INPUT_BYTES + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_INPUT_BYTES || keccak256(&bytes) != identity.digest {
        eyre::bail!("prepared input {index} has an invalid size or digest")
    }
    Ok(bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(MAX_INPUT_BYTES)
        .reject_trailing_bytes()
        .deserialize(&bytes)?)
}

/// Execute one arm without proof generation, compression, or another arm's writer.
pub fn run_cli(args: &[String], allocator: &str) -> eyre::Result<()> {
    for (name, allowed) in [
        ("PS_TRIE_REPR", &["exact"][..]),
        ("PS_UNDO_LAYOUT", &["frames"][..]),
        ("PS_UNDO_RECORD", &["1", "on", "true", "yes"][..]),
        ("PS_RETAIN_GENERATION", &["1", "on", "true", "yes"][..]),
    ] {
        if let Ok(value) = std::env::var(name) &&
            !allowed.contains(&value.as_str())
        {
            eyre::bail!("{name} conflicts with prepared disk-undo validation")
        }
    }
    let mut input = None;
    let mut output = None;
    let mut undo_dir = None;
    let mut depth = RetentionDepth::new(
        std::env::var("PS_RETAIN_DEPTH").unwrap_or_else(|_| "32".into()).parse()?,
    )?;
    let mut interval_ms = 0u64;
    let mut memory_probe_every = 0usize;
    let mut shrink_override = std::env::var("PS_WARM_SHRINK").ok();
    let storage_undo: partial_stateless::StorageUndo = match std::env::var("PS_STORAGE_UNDO") {
        Ok(raw) => raw.parse().map_err(|err| eyre::eyre!("PS_STORAGE_UNDO: {err}"))?,
        Err(_) => Default::default(),
    };
    // Where a Partial arm keeps its frames: session-local disk files (the production profile), or
    // the retained deque itself, which prices keeping the same K frames resident instead.
    let frames_in_memory = match std::env::var("PS_UNDO_STORE").as_deref() {
        Ok("memory") => true,
        Ok("disk") | Err(_) => false,
        Ok(other) => eyre::bail!("PS_UNDO_STORE={other:?}; use disk or memory"),
    };
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        let value = args.next().ok_or_else(|| eyre::eyre!("{arg} needs a value"))?;
        match arg.as_str() {
            "--validate-inputs" => input = Some(PathBuf::from(value)),
            "--out" => output = Some(PathBuf::from(value)),
            "--undo-dir" => undo_dir = Some(PathBuf::from(value)),
            "--retain-depth" => depth = RetentionDepth::new(value.parse()?)?,
            "--interval-ms" => interval_ms = value.parse()?,
            "--memory-probe-every" => memory_probe_every = value.parse()?,
            "--warm-shrink" => shrink_override = Some(value.clone()),
            _ => eyre::bail!("unknown validation option {arg}"),
        }
    }
    let input = input.ok_or_else(|| eyre::eyre!("--validate-inputs is required"))?;
    let output = output.ok_or_else(|| eyre::eyre!("--out is required"))?;
    let manifest_bytes = std::fs::read(input.join("manifest.json"))?;
    let manifest: PreparedManifest = serde_json::from_slice(&manifest_bytes)?;
    manifest.validate()?;
    if !manifest.chain.eq_ignore_ascii_case("mainnet") &&
        manifest.chain != MAINNET.chain.to_string()
    {
        eyre::bail!("prepared validator only supports mainnet")
    }
    let arm: ArmKind = manifest.arm.parse()?;
    let config = match arm {
        ArmKind::Weak => partial_stateless::CacheConfig::default(),
        ArmKind::Policy(p) => p.config(),
    };
    let shrink = parse_warm_shrink(&shrink_override.unwrap_or_else(|| {
        manifest.warm_shrink_blocks.map_or_else(|| "never".into(), |n| n.to_string())
    }))?;
    let mut header_bytes = manifest.parent_header.as_ref();
    let header = <Header as alloy_rlp::Decodable>::decode(&mut header_bytes)?;
    if !header_bytes.is_empty() || header.number.checked_add(1) != Some(manifest.inputs[0].number) {
        eyre::bail!("prepared parent header does not precede the input range")
    }
    let mut parent = SealedHeader::seal_slow(header);
    std::fs::create_dir_all(&output)?;
    let mut rows =
        OpenOptions::new().create_new(true).write(true).open(output.join("validation.jsonl"))?;
    let make_pair = |height| {
        let mut pair = CoordinatedPair {
            cache: config.new_cache_at(height),
            trie_cache: PartialTrieNodeCache::new_with_repr(CacheTrieRepr::Exact),
            readiness: CacheReadinessTracker::new(
                config.account_window.max(config.storage_window),
                config.cache_policy_id(),
            ),
            retained: Default::default(),
            retention_depth: depth,
            undo_layout: UndoLayout::FramesOnly,
            undo_store: None,
            accepted_head: None,
        };
        pair.trie_cache.set_warm_shrink_policy(shrink);
        pair.trie_cache.set_storage_undo(storage_undo);
        pair
    };
    let mut pair = make_pair(parent.number());
    if arm != ArmKind::Weak && frames_in_memory {
        if undo_dir.is_some() {
            eyre::bail!("--undo-dir conflicts with PS_UNDO_STORE=memory")
        }
        pair.trie_cache.set_undo_recording(true);
    } else if arm != ArmKind::Weak {
        pair.enable_disk_undo(&undo_dir.unwrap_or_else(|| output.join("undo")))
            .map_err(eyre::Report::msg)?;
        // Each standalone pass owns its telemetry; shared global output paths are unnecessary.
        if std::env::var_os("PS_UNDO_METRICS_DIR").is_none() {
            pair.undo_store
                .as_ref()
                .expect("enabled")
                .enable_metrics(&output.join("writer"))
                .map_err(eyre::Report::msg)?;
        }
    }
    let consensus = EthBeaconConsensus::new(MAINNET.clone());
    let evm = EthEvmConfig::new(MAINNET.clone());
    let admission = UntrustedAdmission::new(MAINNET.as_ref(), &consensus);
    let limits = SidecarReexecLimits::default();
    let mut executable = File::open(std::env::current_exe()?)?;
    let mut hasher = Keccak256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let count = executable.read(&mut buffer)?;
        if count == 0 {
            break
        }
        hasher.update(&buffer[..count]);
    }
    let run_identity = serde_json::json!({
        "schema_version": 1, "arm": manifest.arm, "input_manifest_digest": keccak256(&manifest_bytes),
        "build_commit": option_env!("PS_BUILD_COMMIT"),
        "build_dirty": match option_env!("PS_BUILD_DIRTY") { Some("0") => Some(false), Some("1") => Some(true), _ => None },
        "cargo_lock_sha256": option_env!("PS_CARGO_LOCK_SHA256"),
        "allocator": allocator, "trie_repr": "exact", "asm_keccak": cfg!(feature = "asm-keccak"),
        "binary_keccak256": hasher.finalize(), "keccak_cache_global": cfg!(feature = "keccak-cache-global"),
        "rayon_num_threads": std::env::var("RAYON_NUM_THREADS").ok(), "malloc_conf": std::env::var("MALLOC_CONF").ok(),
        "undo_recording": arm != ArmKind::Weak,
        "retention_depth": if arm == ArmKind::Weak { 0 } else { depth.get() },
        "undo_layout": match (arm, frames_in_memory) {
            (ArmKind::Weak, _) => "none",
            (_, false) => "disk-frames",
            (_, true) => "memory-frames",
        },
        "storage_undo": (arm != ArmKind::Weak).then(|| pair.trie_cache.storage_undo().label()),
        "warm_shrink_blocks": shrink.interval().map(std::num::NonZeroU64::get), "warmup": manifest.warmup, "samples": manifest.samples,
        "interval_ms": interval_ms, "timing_boundary": "payload_decode_through_coordinated_commit",
        "file_read_included": false, "writer_completion_included": false,
        "memory_probe_every": memory_probe_every,
        "timing_eligible": memory_probe_every == 0 && std::env::var_os("PS_TRIE_SHAPE_DIAGNOSTICS").is_none(),
    });
    serde_json::to_writer_pretty(
        OpenOptions::new().create_new(true).write(true).open(output.join("run.json"))?,
        &run_identity,
    )?;
    let mut measured_hashes = Vec::new();
    for (index, identity) in manifest.inputs.iter().enumerate() {
        let input = read_input(&input, index, identity)?;
        let measured = index as u64 >= manifest.warmup;
        if measured && arm != ArmKind::Weak && !pair.readiness.window_filled() {
            eyre::bail!("measured input reached an incomplete cache window")
        }
        let started = Instant::now();
        let payload: ExecutionData = serde_json::from_slice(&input.payload_json)?;
        let admitted = admission
            .admit(payload, Some(&parent))
            .map_err(|e| eyre::eyre!("admission refused: {e:?}"))?;
        let admission_us = started.elapsed().as_micros() as u64;
        let block = &admitted.block;
        let ctx = BlockContext {
            number: block.number(),
            hash: block.hash(),
            parent_hash: block.parent_hash(),
            state_root: block.state_root(),
        };
        pair.readiness.begin_block(&ctx).map_err(|e| eyre::eyre!("readiness refused: {e:?}"))?;
        let decoded_at = Instant::now();
        let sidecar: PartialStatelessSidecar = bincode::deserialize(&input.sidecar)?;
        let sidecar_decode_us = decoded_at.elapsed().as_micros() as u64;
        let core_at = Instant::now();
        let mut validated = verify_and_apply_sidecar(
            ValidatorRules::new(&evm, &consensus),
            block,
            &mut pair.cache,
            &sidecar,
            config.cache_policy_id(),
            &limits,
            &mut pair.trie_cache,
            TrieCacheDisposition::Commit,
        )?;
        let core_us = core_at.elapsed().as_micros() as u64;
        // This is when all checks actually returned, including the cache-anchor check. It is
        // not a subtraction of nested phase medians or a hypothetical earlier attestation.
        let verified_us = started.elapsed().as_micros() as u64;
        let commit = pair.commit_transition(
            validated.outcome.displaced_trie_cache.take(),
            &ctx,
            block.clone_sealed_header(),
            arm != ArmKind::Weak,
        );
        let prune_at = Instant::now();
        // Disk frames carry their block's flat record with them; resident frames need theirs kept.
        let prune_below = if frames_in_memory && arm != ArmKind::Weak {
            ctx.number.saturating_sub(depth.get())
        } else {
            ctx.number
        };
        pair.cache.prune_undo_below(prune_below);
        let undo_prune_us = prune_at.elapsed().as_micros() as u64;
        // Cacheless cleanup belongs to its step too. No state from the previous block survives.
        if arm == ArmKind::Weak {
            pair = make_pair(ctx.number);
        }
        let block_step_us = started.elapsed().as_micros() as u64;
        if ctx.number != identity.number || ctx.hash != identity.hash {
            eyre::bail!("admitted block does not match prepared identity")
        }
        if let Some(metrics) = commit.disk &&
            (metrics.failed != 0 ||
                metrics.enqueue_failures != 0 ||
                metrics.telemetry_failures != 0)
        {
            eyre::bail!("undo writer failed; pass is not performance evidence")
        }
        // A disk store keeps nothing for the first cold commit, whose empty parent has no state to
        // spill. Resident frames keep that generation, so their deque fills one block sooner.
        let expected_depth = (index as u64 + u64::from(frames_in_memory)).min(depth.get());
        if arm != ArmKind::Weak && index > 0 && pair.retained_depth() != expected_depth {
            eyre::bail!("undo coverage was lost during prepared validation")
        }
        let memory = (memory_probe_every != 0 && index % memory_probe_every == 0).then(|| {
            (pair.cache.estimated_memory_bytes(), pair.trie_cache.memory_breakdown().total_bytes())
        });
        let row = serde_json::json!({
            "schema_version": 1, "block_number": ctx.number, "block_hash": ctx.hash,
            "arm": manifest.arm, "measured": measured, "valid": true,
            "admission_us": admission_us, "sidecar_decode_us": sidecar_decode_us,
            "validation_core_us": core_us, "verified_us": verified_us, "block_step_us": block_step_us,
            "undo_prune_us": undo_prune_us, "commit": commit, "phases": validated.timings,
            "value_cache_bytes": memory.map(|m| m.0), "trie_cache_total_bytes": memory.map(|m| m.1),
            "resident_flat_undo_records": pair.cache.undo_records_len(), "sidecar_bytes": input.sidecar.len(),
            "active_warm_shrink_blocks": pair.trie_cache.warm_shrink_policy().interval().map(std::num::NonZeroU64::get),
        });
        serde_json::to_writer(&mut rows, &row)?;
        rows.write_all(b"\n")?;
        parent = block.clone_sealed_header();
        if measured {
            measured_hashes.extend_from_slice(ctx.hash.as_slice());
        }
        if interval_ms != 0 {
            std::thread::sleep(
                Duration::from_millis(interval_ms).saturating_sub(started.elapsed()),
            );
        }
    }
    let writer = pair
        .undo_store
        .as_ref()
        .map(|store| store.wait_for_idle(Duration::from_secs(30)))
        .transpose()
        .map_err(eyre::Report::msg)?;
    if writer.is_some_and(|m| m.failed != 0 || m.enqueue_failures != 0 || m.telemetry_failures != 0)
    {
        eyre::bail!("undo writer failed during final drain")
    }
    rows.flush()?;
    let result = serde_json::json!({"complete": true, "samples": manifest.samples,
        "measured_block_set_digest": keccak256(&measured_hashes), "writer": writer});
    serde_json::to_writer_pretty(
        OpenOptions::new().create_new(true).write(true).open(output.join("result.json"))?,
        &result,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_manifest_refuses_old_warmup_and_incomplete_inputs() {
        let mut manifest = PreparedManifest {
            schema_version: 1,
            arm: "120/45".into(),
            chain: "mainnet".into(),
            warmup: 120,
            samples: 1,
            parent_header: Bytes::new(),
            warm_shrink_blocks: None,
            build_commit: None,
            inputs: (0..121)
                .map(|number| InputIdentity { number, hash: B256::ZERO, digest: B256::ZERO })
                .collect(),
        };
        assert!(manifest.validate().is_err());
        manifest.warmup = 121;
        assert!(manifest.validate().is_err());
        manifest.inputs.push(InputIdentity { number: 121, hash: B256::ZERO, digest: B256::ZERO });
        assert!(manifest.validate().is_ok());
        manifest.inputs[1].number += 1;
        assert!(manifest.validate().is_err());
    }
}
