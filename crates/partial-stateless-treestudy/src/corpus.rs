//! Reading the recorded corpus one block at a time.
//!
//! The dataset loader in `partial-stateless` reads a whole capture into memory, which is the right
//! shape for a generator that rebuilds sidecars: it needs every record's full transition witness.
//! This study needs the opposite — the access set of every block and none of the MPT witnesses that
//! make the capture 8.8 GB — so it reads records itself and drops the witness on the way past.
//!
//! The checks the loader performs are not dropped with it. Each record's digest is recomputed, the
//! schema version is matched exactly, the parent hash is chained to the record below, and heights
//! with two surviving records are refused rather than silently resolved. A study that quietly read
//! a reorged sibling would be replaying a cache policy over a block sequence its report described
//! wrongly, and the cache is path-dependent, so the error would not stay local to the bad block.

use alloy_consensus::Header;
use alloy_primitives::{keccak256, B256};
use alloy_rlp::Decodable;
use eyre::{bail, Context, Result};
use partial_stateless::{
    accessed_state::BlockAccessedState,
    policy_dataset::{PolicyDatasetRecord, POLICY_DATASET_SCHEMA_VERSION},
};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

/// One block, reduced to what a tree comparison needs.
#[derive(Debug, Clone)]
pub struct CorpusBlock {
    /// Canonical block number.
    pub number: u64,
    /// Canonical block hash.
    pub hash: B256,
    /// Parent block hash, kept so the sequence can be chained.
    pub parent_hash: B256,
    /// State the block touched.
    pub accessed: BlockAccessedState,
    /// Engine-API JSON, when the capture recorded one.
    ///
    /// Carried because re-executing a block needs the block, and the corpus's own payload is the
    /// only version of it that a validator would have seen. Absent under a capture that recorded
    /// no payload; a consumer that needs one must say what it does about the gap rather than
    /// skip silently.
    pub payload_json: Option<Vec<u8>>,
    /// Hashes of the ancestors the block's `BLOCKHASH` range can reach, lowest first.
    pub ancestor_hashes: Vec<(u64, B256)>,
    /// Parent state root the transition witness is proved against.
    pub parent_state_root: B256,
    /// The policy-neutral full transition witness: parent-state trie nodes, flat and sorted.
    ///
    /// Empty unless the reader was asked for it. This is the bulk of the corpus — 8.8 GB against
    /// roughly 300 MB of access sets — and only the coverage pass needs it, because only
    /// re-execution needs *parent* state. The access set records what execution left behind,
    /// not what it started from, so it cannot stand in.
    pub transition_nodes: Vec<alloy_primitives::Bytes>,
    /// The parent block's RLP header.
    ///
    /// Every block but the first can take its parent from the record below, which a consumer
    /// admitted itself and is therefore better evidence. The first has no record below it, and
    /// without this the corpus could not be entered at all.
    pub parent_header: Vec<u8>,
}

/// A corpus, opened but not yet read.
#[derive(Debug)]
pub struct Corpus {
    ordered: Vec<(u64, PathBuf)>,
    range: (u64, u64),
}

impl Corpus {
    /// Opens the capture at `root`, refusing anything the ordinary loader would refuse.
    pub fn open(root: &Path) -> Result<Self> {
        let manifest_path = root.join("manifest.json");
        let manifest: serde_json::Value = serde_json::from_slice(
            &fs::read(&manifest_path).with_context(|| format!("reading {manifest_path:?}"))?,
        )?;
        let schema = manifest.get("schema_version").and_then(serde_json::Value::as_u64);
        if schema != Some(u64::from(POLICY_DATASET_SCHEMA_VERSION)) {
            bail!(
                "corpus schema {schema:?} is not the {POLICY_DATASET_SCHEMA_VERSION} this build reads"
            );
        }

        let end_path = root.join("END.json");
        if !end_path.exists() {
            bail!("corpus at {root:?} has no terminator, so its tail was never vouched for");
        }
        let end: serde_json::Value = serde_json::from_slice(&fs::read(&end_path)?)?;
        if end.get("kind").and_then(serde_json::Value::as_str) == Some("failed") {
            bail!("corpus recorded its own failure: {}", end);
        }
        let usable = end
            .get("usable_range")
            .and_then(serde_json::Value::as_array)
            .and_then(|values| Some((values.first()?.as_u64()?, values.get(1)?.as_u64()?)))
            .ok_or_else(|| eyre::eyre!("corpus terminator carries no usable range"))?;

        let tip_hash: B256 = end
            .get("usable_tip_hash")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| eyre::eyre!("corpus terminator names no tip hash to walk down from"))?
            .parse()?;

        let blocks_dir = root.join("blocks");
        let mut by_height: BTreeMap<u64, Vec<(B256, PathBuf)>> = BTreeMap::new();
        for entry in fs::read_dir(&blocks_dir).with_context(|| format!("reading {blocks_dir:?}"))? {
            let path = entry?.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else { continue };
            if !name.starts_with("block_") || !name.ends_with(".bin") {
                continue
            }
            let mut parts = name.trim_end_matches(".bin").splitn(3, '_');
            let (_, height, hash) = (parts.next(), parts.next(), parts.next());
            let (Some(height), Some(hash)) = (height, hash) else {
                bail!("record {name} does not name a height and a hash");
            };
            let height: u64 = height.parse().with_context(|| format!("height in {name}"))?;
            let hash: B256 =
                format!("0x{hash}").parse().with_context(|| format!("hash in {name}"))?;
            if height < usable.0 || height > usable.1 {
                continue
            }
            by_height.entry(height).or_default().push((hash, path));
        }
        if by_height.is_empty() {
            bail!("corpus at {root:?} holds no usable records");
        }
        for height in usable.0..=usable.1 {
            if !by_height.contains_key(&height) {
                bail!("corpus has no record at height {height}");
            }
        }

        // A reorg leaves two records at one height, and which of them is canonical is decided by
        // the chain rather than by the lifecycle log: walk `parent_hash` down from the terminator's
        // tip. Only a record whose parent's height is contested has to be opened to do it, so the
        // walk costs two reads on this corpus rather than a pass over all 8.8 GB of it.
        let mut ordered: Vec<(u64, PathBuf)> = Vec::with_capacity(by_height.len());
        let mut wanted = tip_hash;
        for height in (usable.0..=usable.1).rev() {
            let candidates = by_height.get(&height).expect("checked above");
            let chosen = if candidates.len() == 1 {
                &candidates[0]
            } else {
                candidates.iter().find(|(hash, _)| *hash == wanted).ok_or_else(|| {
                    eyre::eyre!(
                        "height {height} has {} records and none of them is the {wanted} the \
                             chain above asked for",
                        candidates.len()
                    )
                })?
            };
            ordered.push((height, chosen.1.clone()));
            let parent_contested =
                height > usable.0 && by_height.get(&(height - 1)).is_some_and(|c| c.len() > 1);
            if parent_contested {
                let record: PolicyDatasetRecord = bincode::deserialize(&fs::read(&chosen.1)?)
                    .with_context(|| format!("decoding {:?} to resolve its parent", chosen.1))?;
                wanted = record.body.parent_hash;
            }
        }
        ordered.reverse();

        let range = (ordered[0].0, ordered[ordered.len() - 1].0);
        Ok(Self { ordered, range })
    }

    /// Block range the corpus covers.
    pub const fn range(&self) -> (u64, u64) {
        self.range
    }

    /// How many records the corpus holds.
    pub const fn len(&self) -> usize {
        self.ordered.len()
    }

    /// Whether the corpus is empty.
    pub const fn is_empty(&self) -> bool {
        self.ordered.is_empty()
    }

    /// Reads every record in order, handing each to `visit` and dropping it before the next.
    ///
    /// Streaming rather than collecting because the witness fields dominate the record and are not
    /// wanted: a whole-corpus read would cost about 8.8 GB to deliver roughly 300 MB of access
    /// sets. Each block arrives without its transition witness for the same reason.
    pub fn for_each(
        &self,
        limit: Option<usize>,
        visit: impl FnMut(CorpusBlock) -> Result<()>,
    ) -> Result<()> {
        self.walk(limit, false, visit)
    }

    /// As [`Self::for_each`], but each block also carries its transition witness.
    ///
    /// Only re-execution needs it, and only because the access set records post-execution values:
    /// the witness is the corpus's one record of the state a block *started* from.
    pub fn for_each_with_witness(
        &self,
        limit: Option<usize>,
        visit: impl FnMut(CorpusBlock) -> Result<()>,
    ) -> Result<()> {
        self.walk(limit, true, visit)
    }

    fn walk(
        &self,
        limit: Option<usize>,
        with_witness: bool,
        mut visit: impl FnMut(CorpusBlock) -> Result<()>,
    ) -> Result<()> {
        let take = limit.unwrap_or(self.ordered.len()).min(self.ordered.len());
        let mut previous: Option<B256> = None;
        for (height, path) in self.ordered.iter().take(take) {
            let bytes = fs::read(path).with_context(|| format!("reading {path:?}"))?;
            let record: PolicyDatasetRecord =
                bincode::deserialize(&bytes).with_context(|| format!("decoding {path:?}"))?;
            record.verify_digest().map_err(|err| eyre::eyre!("{err}"))?;
            if record.body.schema_version != POLICY_DATASET_SCHEMA_VERSION {
                bail!(
                    "record {height} is schema {} not {POLICY_DATASET_SCHEMA_VERSION}",
                    record.body.schema_version
                );
            }
            if record.body.block_number != *height {
                bail!("record at {path:?} claims height {}", record.body.block_number);
            }
            if previous.is_some_and(|parent| record.body.parent_hash != parent) {
                bail!("record {height} does not chain to the record below it");
            }
            previous = Some(record.body.block_hash);

            let PolicyDatasetRecord { body, .. } = record;
            let mut ancestor_hashes = Vec::with_capacity(body.ancestor_headers.len());
            for raw in &body.ancestor_headers {
                let header = Header::decode(&mut raw.as_ref())
                    .with_context(|| format!("decoding an ancestor header of block {height}"))?;
                ancestor_hashes.push((header.number, keccak256(raw)));
            }
            visit(CorpusBlock {
                number: body.block_number,
                hash: body.block_hash,
                parent_hash: body.parent_hash,
                accessed: body.accessed,
                payload_json: body.payload_json,
                ancestor_hashes,
                parent_state_root: body.parent_state_root,
                transition_nodes: if with_witness {
                    body.full_transition_nodes
                } else {
                    Vec::new()
                },
                parent_header: body.parent_header.to_vec(),
            })?;
        }
        Ok(())
    }
}
