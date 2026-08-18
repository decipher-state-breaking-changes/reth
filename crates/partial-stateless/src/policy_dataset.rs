//! A recorded corpus that every cache policy can be generated from, once, offline.
//!
//! Comparing cache policies has an experimental problem before it has an engineering one: two
//! policies measured on two live runs are measured on two different block sets, two different
//! machine states, and two different mempools, and nothing in the result separates the policy from
//! the run. Replaying one recorded block set through every policy removes all of that — the block
//! set is identical by construction, and the only thing that varies is the policy.
//!
//! What makes that possible is recording something no policy chose. A dataset record carries the
//! block's Engine payload, the state it accessed, and a **policy-neutral full transition witness**
//! — the parent-state proof a validator holding nothing at all would need. Every policy's own
//! witness is a subset of that one, because every target a warm cache lets a policy skip is a
//! target the full witness already proved. So the recording mentions no window, no anchor, and no
//! miss set, and a generator reading it can produce any policy's real sidecar without a database.
//!
//! **This is a data-production format, not a measurement one.** A run that writes it is paying for
//! a full witness per block that production never builds, and the manifest says so in a field
//! rather than in a comment: see [`PolicyDatasetManifest::measurement_eligible`].
//!
//! The layout, under one dataset directory:
//!
//! ```text
//! <dataset>/
//! ├── manifest.json      identity, capture configuration, and the measurement disclaimer
//! ├── blocks/            one record per captured block
//! │   └── block_<number>_<hash>.bin
//! ├── lifecycle.jsonl    reorgs and resets, in the order the producer saw them
//! └── END.json           written last; its absence means the dataset is incomplete
//! ```
//!
//! Every file is written to a temporary name and renamed into place, so a killed capture leaves
//! whole files or no file. `END.json` is what separates "the producer stopped here" from "the
//! producer died here", and a reader refuses a dataset without one rather than silently reporting
//! a shorter corpus than was asked for.

use crate::accessed_state::BlockAccessedState;
use alloy_primitives::{keccak256, Bytes, B256};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::{self, File},
    io::{self, BufRead, BufReader, Write},
    path::{Path, PathBuf},
};

/// The only record layout this module reads or writes.
///
/// An exact match rather than a floor, for the reason the event stream's frame version is: a newer
/// producer's record read by an older generator would be parsed for the fields it knows and
/// silently missing whatever changed.
///
/// **Version 2 exists because version 1's digest was not a function of the record.** It hashed
/// `bincode::serialize(body)`, and a body holds the access set in `HashMap`s — whose iteration
/// order is seeded per process and rebuilt on deserialization. A record therefore hashed to one
/// value when written and another when read back, so every schema-1 dataset fails its own
/// integrity check on load and none can be accepted as a measurement input. Version 2 hashes an
/// explicit, sorted, length-prefixed encoding instead: see [`PolicyDatasetRecordBody::digest`].
pub const POLICY_DATASET_SCHEMA_VERSION: u32 = 2;

/// Subdirectory holding one file per captured block.
pub const BLOCKS_DIR: &str = "blocks";
/// Dataset identity and capture configuration.
pub const MANIFEST_FILE: &str = "manifest.json";
/// Reorgs and resets, one JSON object per line.
pub const LIFECYCLE_FILE: &str = "lifecycle.jsonl";
/// Written last. Its absence means the capture did not finish.
pub const END_FILE: &str = "END.json";

/// How the producer obtained the payload it recorded.
///
/// Mirrors the validator's own provenance rather than importing it: this crate sits below the
/// validator, and a dataset that could only be described in the validator's vocabulary could not
/// be written by the cache library that owns the witness.
/// Discriminants are pinned because the record digest hashes them. Reordering the variants would
/// silently change every digest this build computes, which is the kind of change that shows up as
/// a corrupt dataset rather than as a version bump.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum RecordedPayloadProvenance {
    /// The payload a consensus client sent, taken from the Engine that validated it.
    Witnessed = 0,
    /// Derived from a block the producer had already accepted.
    Reconstructed = 1,
    /// No payload was obtained and none was derived.
    Absent = 2,
}

/// How the producer obtained the access set it recorded.
/// Pinned for the same reason as [`RecordedPayloadProvenance`]: the digest hashes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum RecordedAccessProvenance {
    /// The artifact the node's own Engine published when it validated the block.
    EngineArtifact = 0,
    /// The producer re-executed the block against its parent state.
    Reexecution = 1,
}

/// One captured block, minus the digest that seals it.
///
/// Separated from [`PolicyDatasetRecord`] so the digest covers an unambiguous byte string: it is
/// the serialization of *this* struct, and nothing has to agree about which fields are excluded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyDatasetRecordBody {
    /// Record layout version.
    pub schema_version: u32,
    /// Canonical block number.
    pub block_number: u64,
    /// Canonical block hash.
    pub block_hash: B256,
    /// Parent block hash.
    pub parent_hash: B256,
    /// Parent state root the witness is proved against.
    pub parent_state_root: B256,
    /// The state root this block's header claims.
    pub expected_state_root: B256,
    /// Engine-API JSON, exactly as a consensus client sends it.
    ///
    /// `None` only under [`RecordedPayloadProvenance::Absent`]. A generator cannot replay such a
    /// record through admission, and a dataset capture refuses to write one rather than leaving a
    /// hole a later run would have to remember.
    pub payload_json: Option<Vec<u8>>,
    /// How the payload above was obtained.
    pub payload_provenance: RecordedPayloadProvenance,
    /// State the block accessed while executing.
    pub accessed: BlockAccessedState,
    /// How the access set above was obtained.
    pub access_provenance: RecordedAccessProvenance,
    /// The policy-neutral full transition witness: flat, hash-deduplicated, sorted.
    pub full_transition_nodes: Vec<Bytes>,
    /// RLP ancestor headers this block's BLOCKHASH range needs, lowest first.
    pub ancestor_headers: Vec<Bytes>,
    /// The parent block's RLP header.
    ///
    /// Recorded because admission needs it and nothing else in the record carries it. Every block
    /// after the first can take its parent header from the record below — that one this run
    /// admitted itself, which is strictly better evidence — but the first has no record below it,
    /// and without this the corpus could not be entered at all.
    pub parent_header: Bytes,
}

impl PolicyDatasetRecordBody {
    /// The digest that seals this record.
    ///
    /// Hashes an explicit encoding rather than the record's serialized form, and every map is
    /// emitted in sorted key order. Both properties are load-bearing and neither is decoration:
    ///
    /// - **Explicit** rather than `bincode::serialize(self)`, because that ties the digest to field
    ///   declaration order and to a serializer's behaviour, neither of which is part of what a
    ///   record *is*.
    /// - **Sorted**, because the access set lives in `HashMap`s. Their iteration order is seeded
    ///   per process and rebuilt on deserialization, so a digest taken over it is not a function of
    ///   the record at all: it changes between writing a record and reading it back, which is
    ///   exactly how every schema-1 dataset came to fail its own integrity check.
    ///
    /// Infallible, unlike the encoder it replaces — there is nothing here that can fail to encode.
    pub fn digest(&self) -> B256 {
        keccak256(self.digest_preimage())
    }

    /// The exact bytes [`Self::digest`] hashes.
    ///
    /// Separate so a test can assert the ordering property directly rather than inferring it from
    /// two hashes agreeing, which they can do by luck.
    fn digest_preimage(&self) -> Vec<u8> {
        fn bytes(out: &mut Vec<u8>, value: &[u8]) {
            out.extend_from_slice(&(value.len() as u64).to_be_bytes());
            out.extend_from_slice(value);
        }
        fn count(out: &mut Vec<u8>, label: &[u8], n: usize) {
            out.extend_from_slice(label);
            out.extend_from_slice(&(n as u64).to_be_bytes());
        }

        let mut preimage = Vec::new();
        preimage.extend_from_slice(b"PolicyDatasetRecord/v2");
        preimage.extend_from_slice(&self.schema_version.to_be_bytes());
        preimage.extend_from_slice(&self.block_number.to_be_bytes());
        preimage.extend_from_slice(self.block_hash.as_slice());
        preimage.extend_from_slice(self.parent_hash.as_slice());
        preimage.extend_from_slice(self.parent_state_root.as_slice());
        preimage.extend_from_slice(self.expected_state_root.as_slice());

        preimage.extend_from_slice(b"payload");
        match &self.payload_json {
            None => preimage.push(0),
            Some(json) => {
                preimage.push(1);
                bytes(&mut preimage, json);
            }
        }
        preimage.extend_from_slice(b"payload_provenance");
        preimage.push(self.payload_provenance as u8);
        preimage.extend_from_slice(b"access_provenance");
        preimage.push(self.access_provenance as u8);

        let mut accounts = self.accessed.accounts.iter().collect::<Vec<_>>();
        accounts.sort_unstable_by_key(|(address, _)| *address);
        count(&mut preimage, b"accessed_accounts", accounts.len());
        for (address, data) in accounts {
            preimage.extend_from_slice(address.as_slice());
            preimage.extend_from_slice(&data.nonce.to_be_bytes());
            preimage.extend_from_slice(&data.balance.to_be_bytes::<32>());
            match data.code_hash {
                None => preimage.push(0),
                Some(code_hash) => {
                    preimage.push(1);
                    preimage.extend_from_slice(code_hash.as_slice());
                }
            }
        }

        let mut storage = self.accessed.storage.iter().collect::<Vec<_>>();
        storage.sort_unstable_by_key(|(key, _)| *key);
        count(&mut preimage, b"accessed_storage", storage.len());
        for ((address, slot), value) in storage {
            preimage.extend_from_slice(address.as_slice());
            preimage.extend_from_slice(slot.as_slice());
            preimage.extend_from_slice(&value.to_be_bytes::<32>());
        }

        let mut codes = self.accessed.codes.iter().collect::<Vec<_>>();
        codes.sort_unstable_by_key(|(code_hash, _)| *code_hash);
        count(&mut preimage, b"accessed_codes", codes.len());
        for (code_hash, code) in codes {
            preimage.extend_from_slice(code_hash.as_slice());
            bytes(&mut preimage, code);
        }

        count(&mut preimage, b"full_transition_nodes", self.full_transition_nodes.len());
        for node in &self.full_transition_nodes {
            bytes(&mut preimage, node);
        }
        count(&mut preimage, b"ancestor_headers", self.ancestor_headers.len());
        for header in &self.ancestor_headers {
            bytes(&mut preimage, header);
        }
        preimage.extend_from_slice(b"parent_header");
        bytes(&mut preimage, &self.parent_header);

        preimage
    }

    /// Seals this body into a record.
    pub fn seal(self) -> Result<PolicyDatasetRecord, DatasetError> {
        let digest = self.digest();
        Ok(PolicyDatasetRecord { body: self, digest })
    }
}

/// One captured block, sealed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyDatasetRecord {
    /// What was captured.
    pub body: PolicyDatasetRecordBody,
    /// The body's canonical digest, as [`PolicyDatasetRecordBody::digest`] computes it.
    pub digest: B256,
}

impl PolicyDatasetRecord {
    /// File name this record occupies inside [`BLOCKS_DIR`].
    pub fn file_name(&self) -> String {
        record_file_name(self.body.block_number, self.body.block_hash)
    }

    /// Recomputes the digest and refuses the record if it does not match.
    pub fn verify_digest(&self) -> Result<(), DatasetError> {
        let recomputed = self.body.digest();
        if recomputed == self.digest {
            return Ok(())
        }
        Err(DatasetError::DigestMismatch {
            block_number: self.body.block_number,
            recorded: self.digest,
            recomputed,
        })
    }
}

/// The name a record occupies inside [`BLOCKS_DIR`].
///
/// The hash is in the name so two records at one height — which a reorg produces — are two files
/// rather than one overwriting the other. Deciding which of them is canonical is the reader's job,
/// and it needs both present to do it.
pub fn record_file_name(block_number: u64, block_hash: B256) -> String {
    format!("block_{block_number:012}_{block_hash:x}.bin")
}

/// What the producer was, what it was configured with, and what its output is not for.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyDatasetManifest {
    /// Record layout version every record in this dataset carries.
    pub schema_version: u32,
    /// Producer identity, e.g. crate name and version.
    pub producer: String,
    /// Git commit the producer was built from, when the build stamped one.
    pub build_commit: Option<String>,
    /// Chain identifier the blocks belong to.
    pub chain: String,
    /// The block the capture intends to start at, when the producer knew it in advance.
    pub first_block: Option<u64>,
    /// Blocks the capture will stop after.
    pub max_blocks: u64,
    /// Always false, and not a formality.
    ///
    /// A capturing run builds a full witness per block that production never builds, serializes
    /// it, and writes it to disk. Every latency it reports is a latency of that run, not of a
    /// builder, and a corpus that did not say so could be read back into a performance claim
    /// by someone who was not there when it was recorded.
    pub measurement_eligible: bool,
    /// Always true: the capture's own cost is not accounted for in anything this dataset carries.
    pub capture_overhead_excluded: bool,
}

impl PolicyDatasetManifest {
    /// A manifest for a fresh capture.
    pub fn new(
        producer: String,
        build_commit: Option<String>,
        chain: String,
        max_blocks: u64,
    ) -> Self {
        Self {
            schema_version: POLICY_DATASET_SCHEMA_VERSION,
            producer,
            build_commit,
            chain,
            first_block: None,
            max_blocks,
            measurement_eligible: false,
            capture_overhead_excluded: true,
        }
    }
}

/// Something that happened to the chain under the capture.
///
/// Recorded rather than hidden. A capture that silently dropped abandoned blocks would hand the
/// offline stage a block set it could not audit, and one that silently kept them would hand it a
/// set containing two blocks at one height. Writing the event and deciding later is the only
/// option that leaves the decision inspectable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum LifecycleEvent {
    /// The capture began.
    Started {
        /// First block the capture recorded.
        block_number: u64,
    },
    /// A branch was abandoned.
    Reorg {
        /// The last block both branches share.
        common_ancestor: u64,
        /// Blocks leaving the canonical chain, lowest first, as `(number, hash)`.
        abandoned: Vec<(u64, B256)>,
    },
    /// The producer lost continuity and the records either side cannot be joined.
    Reset {
        /// Where it happened.
        block_number: u64,
        /// Free text for the run log. Never parsed.
        detail: String,
    },
}

/// Why a capture stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetEndKind {
    /// The configured block budget was reached.
    BlockBudgetReached,
    /// The producer shut down cleanly before the budget was reached.
    ProducerShutdown,
    /// The capture failed and the dataset must not be used.
    Failed,
}

/// The terminator that makes a dataset readable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetEnd {
    /// Why the capture stopped.
    pub kind: DatasetEndKind,
    /// Records written, reorg-abandoned ones included.
    pub records: u64,
    /// Lowest and highest block numbers written, when anything was.
    pub block_range: Option<(u64, u64)>,
    /// The canonical range a consumer may use, and the only one it should.
    ///
    /// Narrower than `block_range` by design: it excludes the tail the capture did not carry far
    /// enough past to call settled. `None` means no range qualified, which is a usable dataset of
    /// zero blocks rather than a usable dataset of all of them.
    #[serde(default)]
    pub usable_range: Option<(u64, u64)>,
    /// The block hash at the top of [`Self::usable_range`].
    ///
    /// This is what makes the canonical set derivable rather than inferred. A reader walks
    /// `parent_hash` down from here to the bottom of the range and gets exactly one record per
    /// height — which is the right answer even when the chain reorganised away from a branch and
    /// back onto it, a case no accumulated list of abandoned hashes can represent.
    #[serde(default)]
    pub usable_tip_hash: Option<B256>,
    /// Blocks the capture required on top of the usable range before writing this file.
    ///
    /// Zero means the operator explicitly asked for no confirmation depth, which leaves the tail
    /// of `usable_range` reorg-exposed. Recorded rather than assumed, so a reader sees the choice
    /// that was made rather than having to infer it.
    #[serde(default)]
    pub confirmations: u64,
    /// The canonical head the producer had reached when it wrote this file.
    #[serde(default)]
    pub confirmed_at_head: Option<u64>,
    /// Free text for the run log. Never parsed.
    pub detail: String,
}

/// Everything that can go wrong reading or writing a dataset.
#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    /// Filesystem failure, with the path it happened on.
    #[error("dataset I/O failed at {path}: {source}")]
    Io {
        /// The path.
        path: PathBuf,
        /// The underlying failure.
        #[source]
        source: io::Error,
    },
    /// A record or manifest could not be serialized.
    #[error("dataset encode failed: {0}")]
    Encode(String),
    /// A record or manifest could not be deserialized.
    #[error("dataset decode failed at {path}: {detail}")]
    Decode {
        /// The file.
        path: PathBuf,
        /// What the decoder said.
        detail: String,
    },
    /// The producer tried to write different contents under an existing record identity.
    #[error(
        "dataset record for block {block_number} ({block_hash:?}) was produced twice with \
         different contents"
    )]
    ConflictingDuplicate {
        /// The block.
        block_number: u64,
        /// The hash that makes the record filename unique at that height.
        block_hash: B256,
    },
    /// A record's digest does not cover its body.
    #[error("record for block {block_number} has digest {recorded:?} but its body hashes to {recomputed:?}")]
    DigestMismatch {
        /// The block.
        block_number: u64,
        /// What the file claims.
        recorded: B256,
        /// What the body hashes to.
        recomputed: B256,
    },
    /// A record was written by a layout this build does not read.
    #[error("record for block {block_number} is schema version {found}, but this build reads only {expected}")]
    SchemaMismatch {
        /// The block.
        block_number: u64,
        /// What the file claims.
        found: u32,
        /// What this build reads.
        expected: u32,
    },
    /// The dataset has no `END.json`, so the capture did not finish.
    #[error(
        "dataset at {0} has no {END_FILE}: the capture did not finish and the corpus is incomplete"
    )]
    Incomplete(PathBuf),
    /// The dataset has no lifecycle log, so its reorg history cannot be inspected.
    #[error(
        "dataset has no lifecycle log at {0}: every capture writes one before its first block, so \
         its absence means the log was lost and the dataset's exclusions cannot be audited"
    )]
    MissingLifecycle(PathBuf),
    /// The capture was pointed at a directory that already held something.
    #[error(
        "dataset directory {0} is not empty; capture into a fresh directory. A leftover \
         terminator would close this run's blocks with the previous run's verdict, and nothing \
         afterwards could tell the two apart"
    )]
    NotFresh(PathBuf),
    /// The manifest was written by a layout this build does not read.
    #[error("dataset manifest is schema version {found}, but this build reads only {expected}")]
    ManifestSchemaMismatch {
        /// What the manifest claims.
        found: u32,
        /// What this build reads.
        expected: u32,
    },
    /// The terminator's own account of the dataset does not match what is on disk.
    ///
    /// One variant for all of the cross-checks, because they answer one question: does the file
    /// that says the capture finished describe the capture that actually happened?
    #[error("dataset terminator disagrees with the files on disk: {detail}")]
    TerminatorMismatch {
        /// Which count or range disagreed, and how.
        detail: String,
    },
    /// The capture recorded its own failure.
    #[error("dataset at {path} ended as {kind:?}: {detail}")]
    EndedBadly {
        /// The dataset.
        path: PathBuf,
        /// How it ended.
        kind: DatasetEndKind,
        /// What the producer said.
        detail: String,
    },
    /// The canonical walk ran out of records before it reached the bottom of the usable range.
    #[error(
        "dataset is missing block {missing}: walking parents down from the usable tip stopped at \
         {stopped_at}, which names a parent no record holds"
    )]
    Gap {
        /// The height the walk needed next.
        missing: u64,
        /// The lowest height the walk did reach.
        stopped_at: u64,
    },
    /// A record's number does not sit one below the child that names it as parent.
    #[error(
        "dataset record {hash:?} is at height {found}, but the record below block {block_number} \
         names it as its parent"
    )]
    BrokenChain {
        /// The child.
        block_number: u64,
        /// The parent's hash.
        hash: B256,
        /// The height that parent actually claims.
        found: u64,
    },
}

fn io_err(path: &Path, source: io::Error) -> DatasetError {
    DatasetError::Io { path: path.to_path_buf(), source }
}

/// Writes `bytes` to `path` through a temporary file, so no reader ever sees a partial file.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), DatasetError> {
    let tmp = path.with_extension("tmp");
    let mut file = File::create(&tmp).map_err(|err| io_err(&tmp, err))?;
    file.write_all(bytes).map_err(|err| io_err(&tmp, err))?;
    file.sync_all().map_err(|err| io_err(&tmp, err))?;
    drop(file);
    fs::rename(&tmp, path).map_err(|err| io_err(path, err))
}

/// Writes one capture's records, lifecycle, and terminator.
#[derive(Debug)]
pub struct PolicyDatasetWriter {
    root: PathBuf,
    blocks: PathBuf,
    records: u64,
    lowest: Option<u64>,
    highest: Option<u64>,
}

impl PolicyDatasetWriter {
    /// Creates the dataset directory and writes its manifest.
    ///
    /// Refuses a root that holds anything at all, not merely one that holds records. A directory
    /// with no `blocks/` but a leftover `END.json` is the dangerous case: a capture started there
    /// and then killed would leave a corpus that reads as complete, terminated by the *previous*
    /// run's verdict over the *current* run's blocks. Nothing after the fact can tell that apart,
    /// so it is refused before the fact.
    pub fn create(root: &Path, manifest: &PolicyDatasetManifest) -> Result<Self, DatasetError> {
        if root.exists() {
            let occupied = fs::read_dir(root)
                .map_err(|err| io_err(root, err))?
                .filter_map(Result::ok)
                .next()
                .is_some();
            if occupied {
                return Err(DatasetError::NotFresh(root.to_path_buf()))
            }
        }
        let blocks = root.join(BLOCKS_DIR);
        fs::create_dir_all(&blocks).map_err(|err| io_err(&blocks, err))?;
        let manifest_bytes = serde_json::to_vec_pretty(manifest)
            .map_err(|err| DatasetError::Encode(err.to_string()))?;
        write_atomic(&root.join(MANIFEST_FILE), &manifest_bytes)?;
        Ok(Self { root: root.to_path_buf(), blocks, records: 0, lowest: None, highest: None })
    }

    /// Records written so far.
    pub const fn records(&self) -> u64 {
        self.records
    }

    /// Writes one block record.
    pub fn write_record(&mut self, record: &PolicyDatasetRecord) -> Result<PathBuf, DatasetError> {
        let bytes =
            bincode::serialize(record).map_err(|err| DatasetError::Encode(err.to_string()))?;
        let path = self.blocks.join(record.file_name());

        // A branch can be abandoned and later become canonical again. In that case the producer
        // sees the exact same `(height, hash)` twice, which names the same file and must count as
        // one physical record in `END.json`. The second observation must not silently overwrite
        // different capture material under the same block identity either, so the two are compared.
        //
        // The comparison is by digest and not by serialized bytes. The body holds its access set in
        // `HashMap`s, and rebuilding the record for the returning branch builds fresh maps, which
        // iterate in a different order even inside one process: `RandomState` re-seeds per map.
        // Equal bytes therefore prove sameness but unequal bytes prove nothing, and a byte
        // comparison would abort a capture over a re-canonicalized branch that changed nothing.
        // The digest is a function of the record's contents, so it decides the question the byte
        // comparison was only approximating.
        if path.exists() {
            let existing_bytes = fs::read(&path).map_err(|err| io_err(&path, err))?;
            if existing_bytes != bytes {
                let existing: PolicyDatasetRecord =
                    bincode::deserialize(&existing_bytes).map_err(|err| DatasetError::Decode {
                        path: path.clone(),
                        detail: err.to_string(),
                    })?;
                // Checked rather than read off the file, so a record that was damaged after it was
                // written fails here, at the block that touched it, instead of at load time.
                existing.verify_digest()?;
                if existing.body.digest() != record.body.digest() {
                    return Err(DatasetError::ConflictingDuplicate {
                        block_number: record.body.block_number,
                        block_hash: record.body.block_hash,
                    })
                }
            }
            return Ok(path)
        }

        write_atomic(&path, &bytes)?;
        self.records += 1;
        let number = record.body.block_number;
        self.lowest = Some(self.lowest.map_or(number, |low| low.min(number)));
        self.highest = Some(self.highest.map_or(number, |high| high.max(number)));
        Ok(path)
    }

    /// Appends one lifecycle event.
    pub fn write_lifecycle(&self, event: &LifecycleEvent) -> Result<(), DatasetError> {
        let path = self.root.join(LIFECYCLE_FILE);
        let mut line =
            serde_json::to_vec(event).map_err(|err| DatasetError::Encode(err.to_string()))?;
        line.push(b'\n');
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|err| io_err(&path, err))?;
        file.write_all(&line).map_err(|err| io_err(&path, err))?;
        file.sync_all().map_err(|err| io_err(&path, err))
    }

    /// Writes the terminator that makes the dataset readable.
    ///
    /// `usable` is the canonical range the producer is willing to stand behind — the part it
    /// carried far enough past to call settled — as `(low, high, hash at high)`, and
    /// `confirmations` says how far past it the chain went. The tip hash is what a reader walks
    /// parents down from to recover the canonical set.
    pub fn finish(
        self,
        kind: DatasetEndKind,
        usable: Option<(u64, u64, B256)>,
        confirmations: u64,
        confirmed_at_head: Option<u64>,
        detail: String,
    ) -> Result<(), DatasetError> {
        let end = DatasetEnd {
            kind,
            records: self.records,
            block_range: self.lowest.zip(self.highest),
            usable_range: usable.map(|(low, high, _)| (low, high)),
            usable_tip_hash: usable.map(|(_, _, tip)| tip),
            confirmations,
            confirmed_at_head,
            detail,
        };
        let bytes =
            serde_json::to_vec_pretty(&end).map_err(|err| DatasetError::Encode(err.to_string()))?;
        write_atomic(&self.root.join(END_FILE), &bytes)
    }
}

/// A complete dataset, as a generator sees it.
#[derive(Debug)]
pub struct LoadedDataset {
    /// The capture's manifest.
    pub manifest: PolicyDatasetManifest,
    /// Canonical records, ascending by block number, gap-free and chained.
    pub records: Vec<PolicyDatasetRecord>,
    /// Records on a branch the chain left, kept so a reader can audit the exclusion.
    pub abandoned: Vec<PolicyDatasetRecord>,
    /// Records above the usable range: written, but never carried far enough past to be vouched
    /// for. Excluded from [`Self::records`] for that reason alone, not because anything is wrong
    /// with them.
    pub unconfirmed: Vec<PolicyDatasetRecord>,
    /// Lifecycle events in the order the producer wrote them.
    pub lifecycle: Vec<LifecycleEvent>,
    /// How the capture ended.
    pub end: DatasetEnd,
}

/// Reads a dataset, refusing anything a generator must not silently work around.
///
/// Fail-closed on every one of: a missing terminator, a capture that recorded its own failure, a
/// record whose digest does not cover it, a record written by another layout, a height with two
/// surviving canonical records, a gap, and a parent hash that does not match the record below.
/// Every one of those has a plausible-looking partial reading, and every partial reading produces
/// a policy comparison run over a block set the report would describe wrongly.
pub fn load_dataset(root: &Path) -> Result<LoadedDataset, DatasetError> {
    let manifest_path = root.join(MANIFEST_FILE);
    let manifest_bytes = fs::read(&manifest_path).map_err(|err| io_err(&manifest_path, err))?;
    let manifest: PolicyDatasetManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|err| DatasetError::Decode { path: manifest_path, detail: err.to_string() })?;
    if manifest.schema_version != POLICY_DATASET_SCHEMA_VERSION {
        return Err(DatasetError::ManifestSchemaMismatch {
            found: manifest.schema_version,
            expected: POLICY_DATASET_SCHEMA_VERSION,
        })
    }

    let end_path = root.join(END_FILE);
    if !end_path.exists() {
        return Err(DatasetError::Incomplete(root.to_path_buf()))
    }
    let end_bytes = fs::read(&end_path).map_err(|err| io_err(&end_path, err))?;
    let end: DatasetEnd = serde_json::from_slice(&end_bytes)
        .map_err(|err| DatasetError::Decode { path: end_path, detail: err.to_string() })?;
    if end.kind == DatasetEndKind::Failed {
        return Err(DatasetError::EndedBadly {
            path: root.to_path_buf(),
            kind: end.kind,
            detail: end.detail.clone(),
        })
    }

    let lifecycle = read_lifecycle(&root.join(LIFECYCLE_FILE))?;

    let blocks_dir = root.join(BLOCKS_DIR);
    let mut all = Vec::new();
    for entry in fs::read_dir(&blocks_dir).map_err(|err| io_err(&blocks_dir, err))? {
        let path = entry.map_err(|err| io_err(&blocks_dir, err))?.path();
        let is_record = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("block_") && name.ends_with(".bin"));
        if !is_record {
            continue;
        }
        let bytes = fs::read(&path).map_err(|err| io_err(&path, err))?;
        let record: PolicyDatasetRecord = bincode::deserialize(&bytes)
            .map_err(|err| DatasetError::Decode { path: path.clone(), detail: err.to_string() })?;
        record.verify_digest()?;
        if record.body.schema_version != POLICY_DATASET_SCHEMA_VERSION {
            return Err(DatasetError::SchemaMismatch {
                block_number: record.body.block_number,
                found: record.body.schema_version,
                expected: POLICY_DATASET_SCHEMA_VERSION,
            })
        }
        all.push(record);
    }

    all.sort_by_key(|record| record.body.block_number);
    check_terminator_against_files(&end, &all)?;

    // The canonical set is *derived* from the records, not inferred from the lifecycle log. Every
    // record outside it is on disk because the capture wrote it, and neither kind is something a
    // consumer may replay — but they are excluded for different reasons, and a reader auditing a
    // corpus needs to see which is which.
    let (records, rest) = canonical_chain(&end, all)?;
    let ceiling = end.usable_range.map(|(_, high)| high);
    let (unconfirmed, abandoned) = rest
        .into_iter()
        .partition(|record| ceiling.is_none_or(|high| record.body.block_number > high));

    Ok(LoadedDataset { manifest, records, abandoned, unconfirmed, lifecycle, end })
}

/// Walks `parent_hash` down from the terminator's usable tip to the bottom of its usable range.
///
/// This is the whole of how a canonical set is decided, and it is decided by the chain rather than
/// by the lifecycle log for one specific reason. A log of abandoned hashes can only ever
/// accumulate: once a branch is listed it stays listed, so a chain that reorganised away from a
/// branch and then back onto it would leave *both* branches marked abandoned and the corpus
/// unreadable. Walking parents asks the records themselves which chain won, and gets the right
/// answer however many times the chain changed its mind.
///
/// Returns the canonical records lowest-first, and everything else — losing branches and the
/// unconfirmed tail alike.
fn canonical_chain(
    end: &DatasetEnd,
    all: Vec<PolicyDatasetRecord>,
) -> Result<(Vec<PolicyDatasetRecord>, Vec<PolicyDatasetRecord>), DatasetError> {
    let Some((low, high)) = end.usable_range else {
        // A capture that vouched for no range yields no blocks, whatever it managed to write.
        return Ok((Vec::new(), all))
    };
    let tip_hash = end.usable_tip_hash.ok_or_else(|| DatasetError::TerminatorMismatch {
        detail: format!("it names usable range {low}..={high} but no tip hash to walk down from"),
    })?;

    let mut by_hash =
        all.into_iter().map(|record| (record.body.block_hash, record)).collect::<HashMap<_, _>>();

    let mut chain = Vec::new();
    let mut wanted = tip_hash;
    let mut height = high;
    loop {
        let Some(record) = by_hash.remove(&wanted) else {
            return Err(DatasetError::Gap { missing: height, stopped_at: height.saturating_add(1) })
        };
        if record.body.block_number != height {
            return Err(DatasetError::BrokenChain {
                block_number: height.saturating_add(1),
                hash: wanted,
                found: record.body.block_number,
            })
        }
        wanted = record.body.parent_hash;
        chain.push(record);
        if height == low {
            break
        }
        height -= 1;
    }

    chain.reverse();
    let mut rest = by_hash.into_values().collect::<Vec<_>>();
    rest.sort_by_key(|record| record.body.block_number);
    Ok((chain, rest))
}

/// Cross-checks the terminator's own account of the capture against the files it left behind.
///
/// A terminator is the one file that says a dataset is complete, so a terminator that does not
/// describe the directory it sits in is the single most dangerous artifact here: everything
/// downstream trusts it, and nothing downstream re-derives it. Every number it carries is checked
/// against something computed from the records, including the confirmation claim — a dataset whose
/// terminator says a range settled without the head to back it up is a dataset vouching for a tip
/// nobody saw confirmed.
fn check_terminator_against_files(
    end: &DatasetEnd,
    all: &[PolicyDatasetRecord],
) -> Result<(), DatasetError> {
    let on_disk = all.len() as u64;
    if end.records != on_disk {
        return Err(DatasetError::TerminatorMismatch {
            detail: format!("it claims {} records, but {on_disk} are present", end.records),
        })
    }

    let actual_range =
        all.iter().map(|record| record.body.block_number).fold(None::<(u64, u64)>, |range, n| {
            Some(range.map_or((n, n), |(low, high)| (low.min(n), high.max(n))))
        });
    if end.block_range != actual_range {
        return Err(DatasetError::TerminatorMismatch {
            detail: format!(
                "it claims block range {:?}, but the records span {actual_range:?}",
                end.block_range
            ),
        })
    }

    let Some((low, high)) = end.usable_range else {
        // Nothing vouched for, so there is no confirmation claim to check. The head, if it recorded
        // one, is informational.
        return Ok(())
    };
    if low > high {
        return Err(DatasetError::TerminatorMismatch {
            detail: format!("its usable range {low}..={high} is inverted"),
        })
    }

    // The confirmation claim, checked rather than taken. Without this the depth is a number in a
    // file: a terminator could name any range and any depth, and a reader replaying its tail would
    // be replaying blocks that were never carried past.
    let Some(head) = end.confirmed_at_head else {
        return Err(DatasetError::TerminatorMismatch {
            detail: format!(
                "it vouches for {low}..={high} but records no canonical head, so its \
                 {} -block confirmation claim rests on nothing",
                end.confirmations
            ),
        })
    };
    let required =
        high.checked_add(end.confirmations).ok_or_else(|| DatasetError::TerminatorMismatch {
            detail: format!(
                "its usable tip {high} plus {} confirmations overflows a block number",
                end.confirmations
            ),
        })?;
    if head < required {
        return Err(DatasetError::TerminatorMismatch {
            detail: format!(
                "it vouches for a tip at {high} with {} confirmations, which needs a head of at \
                 least {required}, but it only reached {head}",
                end.confirmations
            ),
        })
    }
    Ok(())
}

/// Reads the lifecycle log, refusing a dataset that has none.
///
/// Absence is a refusal rather than an empty list. Every capture writes a `Started` event before
/// its first block, so a dataset without this file is one whose log was lost or removed — and a
/// corpus whose reorg history cannot be inspected is one whose exclusions cannot be audited, even
/// though the canonical set itself is derived from the records rather than from here.
fn read_lifecycle(path: &Path) -> Result<Vec<LifecycleEvent>, DatasetError> {
    if !path.exists() {
        return Err(DatasetError::MissingLifecycle(path.to_path_buf()))
    }
    let file = File::open(path).map_err(|err| io_err(path, err))?;
    let mut events = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line.map_err(|err| io_err(path, err))?;
        if line.trim().is_empty() {
            continue;
        }
        events.push(serde_json::from_str(&line).map_err(|err| DatasetError::Decode {
            path: path.to_path_buf(),
            detail: err.to_string(),
        })?);
    }
    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::AccountData;
    use alloy_primitives::{Address, U256};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("ps-policy-dataset-{name}-{nanos}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn body(number: u64, parent: B256, hash: B256) -> PolicyDatasetRecordBody {
        PolicyDatasetRecordBody {
            schema_version: POLICY_DATASET_SCHEMA_VERSION,
            block_number: number,
            block_hash: hash,
            parent_hash: parent,
            parent_state_root: B256::repeat_byte(0x01),
            expected_state_root: B256::repeat_byte(0x02),
            payload_json: Some(b"{}".to_vec()),
            payload_provenance: RecordedPayloadProvenance::Witnessed,
            accessed: BlockAccessedState::default(),
            access_provenance: RecordedAccessProvenance::Reexecution,
            full_transition_nodes: vec![Bytes::from_static(&[0x80])],
            ancestor_headers: Vec::new(),
            parent_header: Bytes::from_static(&[0x80]),
        }
    }

    /// Closes a test dataset vouching for the range it names, with a head that confirms it.
    fn finish_all(
        writer: PolicyDatasetWriter,
        usable: Option<(u64, u64, B256)>,
        detail: &str,
    ) -> Result<(), DatasetError> {
        let head = usable.map(|(_, high, _)| high + CONFIRMATIONS);
        writer.finish(
            DatasetEndKind::BlockBudgetReached,
            usable,
            CONFIRMATIONS,
            head,
            detail.to_string(),
        )
    }

    /// Replaces the terminator with one written by hand.
    ///
    /// The writer cannot emit a terminator that disagrees with its own directory, which is the
    /// whole reason the loader has to check one it did not write.
    fn write_end(dir: &Path, end: DatasetEnd) {
        fs::write(dir.join(END_FILE), serde_json::to_vec(&end).unwrap()).unwrap();
    }

    /// A writer over a fresh directory with its opening lifecycle event already written.
    ///
    /// Every real capture writes one before its first block, and the loader now requires the log,
    /// so a test that skipped it would be testing a dataset shape the producer cannot emit.
    fn started_writer(dir: &Path) -> PolicyDatasetWriter {
        let writer = PolicyDatasetWriter::create(dir, &manifest()).unwrap();
        writer.write_lifecycle(&LifecycleEvent::Started { block_number: 10 }).unwrap();
        writer
    }

    const CONFIRMATIONS: u64 = 8;

    fn manifest() -> PolicyDatasetManifest {
        PolicyDatasetManifest::new("test".into(), None, "mainnet".into(), 8)
    }

    fn numbered(tag: u8) -> B256 {
        B256::repeat_byte(tag)
    }

    /// A populated access set, built twice by different routes.
    ///
    /// The maps are `HashMap`s, so the two are equal as sets and need not iterate alike.
    fn populated_access(reverse: bool, reserve: usize) -> BlockAccessedState {
        let mut accessed = BlockAccessedState {
            accounts: HashMap::with_capacity(reserve),
            storage: HashMap::with_capacity(reserve),
            codes: HashMap::with_capacity(reserve),
        };
        let mut tags = (1..=48u8).collect::<Vec<_>>();
        if reverse {
            tags.reverse();
        }
        for tag in tags {
            let address = Address::repeat_byte(tag);
            accessed.accounts.insert(
                address,
                AccountData {
                    nonce: u64::from(tag),
                    balance: U256::from(tag) * U256::from(1_000u64),
                    code_hash: (tag % 3 == 0).then(|| B256::repeat_byte(tag)),
                },
            );
            accessed.storage.insert((address, B256::repeat_byte(tag ^ 0x5a)), U256::from(tag));
            accessed.codes.insert(B256::repeat_byte(tag), Bytes::from(vec![tag; tag as usize]));
        }
        accessed
    }

    fn populated_body(reverse: bool, reserve: usize) -> PolicyDatasetRecordBody {
        let mut record = body(10, numbered(0x09), numbered(0x0a));
        record.accessed = populated_access(reverse, reserve);
        record
    }

    /// The defect that made every schema-1 dataset unusable, as a test.
    ///
    /// A record hashed one way when written and another when read back, because the digest was
    /// taken over `bincode::serialize(body)` and the access set lives in `HashMap`s whose order is
    /// rebuilt on deserialization. Nothing caught it, because the fixtures every other test uses
    /// carry an empty access set and an empty map has only one order.
    #[test]
    fn a_digest_survives_the_round_trip_that_broke_schema_one() {
        let record = populated_body(false, 0).seal().unwrap();
        let encoded = bincode::serialize(&record).unwrap();
        let decoded: PolicyDatasetRecord = bincode::deserialize(&encoded).unwrap();

        assert_eq!(
            decoded.digest, record.digest,
            "the stored digest did not survive the round trip"
        );
        decoded
            .verify_digest()
            .expect("a record must hash to the same value after it is read back");
    }

    /// The property behind that fix, asserted on the bytes rather than inferred from two hashes
    /// agreeing — which they can do by luck on any single run.
    #[test]
    fn the_digest_preimage_lists_every_map_in_key_order() {
        let preimage = populated_body(true, 512).digest_preimage();
        let section = |from: &[u8], to: &[u8]| {
            let find = |needle: &[u8]| {
                preimage.windows(needle.len()).position(|window| window == needle).unwrap_or_else(
                    || panic!("no {} label in the preimage", String::from_utf8_lossy(needle)),
                )
            };
            // Scoped to one section, because a 20-byte run can occur anywhere — the record's own
            // parent hash is twenty repeated bytes too, and an unscoped search finds that first.
            preimage[find(from)..find(to)].to_vec()
        };

        let accounts = section(b"accessed_accounts", b"accessed_storage");
        let mut previous = 0;
        for tag in 1..=48u8 {
            let needle = Address::repeat_byte(tag);
            let at = accounts
                .windows(20)
                .position(|window| window == needle.as_slice())
                .unwrap_or_else(|| panic!("account {tag} is missing from the preimage"));
            assert!(at > previous || tag == 1, "account {tag} is encoded out of order");
            previous = at;
        }

        let codes = section(b"accessed_codes", b"full_transition_nodes");
        let mut previous = 0;
        for tag in 1..=48u8 {
            let needle = B256::repeat_byte(tag);
            let at = codes
                .windows(32)
                .position(|window| window == needle.as_slice())
                .unwrap_or_else(|| panic!("code {tag} is missing from the preimage"));
            assert!(at > previous || tag == 1, "code {tag} is encoded out of order");
            previous = at;
        }
    }

    /// Two access sets that are equal as sets hash alike however they were built.
    #[test]
    fn a_digest_is_a_function_of_the_record_and_not_of_its_maps() {
        assert_eq!(populated_body(false, 0).digest(), populated_body(true, 512).digest());
        // And still moves when the record actually differs.
        let mut changed = populated_body(false, 0);
        changed
            .accessed
            .storage
            .insert((Address::repeat_byte(1), B256::repeat_byte(0xee)), U256::from(7u64));
        assert_ne!(populated_body(false, 0).digest(), changed.digest());
    }

    /// A schema-1 dataset is refused rather than half-read: its digests cannot be reproduced, so
    /// there is nothing to check its records against.
    #[test]
    fn a_record_from_the_superseded_schema_is_refused() {
        let dir = temp_dir("old-schema");
        let mut writer = started_writer(&dir);
        let mut stale = body(10, numbered(0x09), numbered(0x0a));
        stale.schema_version = 1;
        writer.write_record(&stale.seal().unwrap()).unwrap();
        finish_all(writer, Some((10, 10, numbered(0x0a))), "").unwrap();
        assert!(matches!(
            load_dataset(&dir),
            Err(DatasetError::SchemaMismatch { found: 1, expected: 2, .. })
        ));
    }

    #[test]
    fn a_written_dataset_reads_back_as_the_chain_it_recorded() {
        let dir = temp_dir("roundtrip");
        let mut writer = started_writer(&dir);
        for (number, parent, hash) in
            [(10, numbered(0x09), numbered(0x0a)), (11, numbered(0x0a), numbered(0x0b))]
        {
            writer.write_record(&body(number, parent, hash).seal().unwrap()).unwrap();
        }
        finish_all(writer, Some((10, 11, numbered(0x0b))), "done").unwrap();

        let loaded = load_dataset(&dir).unwrap();
        assert_eq!(loaded.records.len(), 2);
        assert_eq!(loaded.records[0].body.block_number, 10);
        assert_eq!(loaded.end.block_range, Some((10, 11)));
        assert!(!loaded.manifest.measurement_eligible);
        assert_eq!(loaded.lifecycle.len(), 1);
    }

    /// A capture that was killed leaves records but no terminator, and a generator that read it
    /// would silently run over a shorter corpus than the one it was asked for.
    #[test]
    fn a_dataset_without_its_terminator_is_refused() {
        let dir = temp_dir("incomplete");
        let mut writer = started_writer(&dir);
        writer.write_record(&body(10, numbered(0x09), numbered(0x0a)).seal().unwrap()).unwrap();
        assert!(matches!(load_dataset(&dir), Err(DatasetError::Incomplete(_))));
    }

    #[test]
    fn a_tampered_record_is_refused_rather_than_replayed() {
        let dir = temp_dir("digest");
        let mut writer = started_writer(&dir);
        let mut record = body(10, numbered(0x09), numbered(0x0a)).seal().unwrap();
        record.body.expected_state_root = B256::repeat_byte(0xff);
        writer.write_record(&record).unwrap();
        finish_all(writer, Some((10, 10, numbered(0x0a))), "").unwrap();
        assert!(matches!(load_dataset(&dir), Err(DatasetError::DigestMismatch { .. })));
    }

    #[test]
    fn a_gap_is_refused_rather_than_closed_over() {
        let dir = temp_dir("gap");
        let mut writer = started_writer(&dir);
        writer.write_record(&body(10, numbered(0x09), numbered(0x0a)).seal().unwrap()).unwrap();
        writer.write_record(&body(12, numbered(0x0b), numbered(0x0c)).seal().unwrap()).unwrap();
        finish_all(writer, Some((10, 12, numbered(0x0c))), "").unwrap();
        assert!(matches!(load_dataset(&dir), Err(DatasetError::Gap { missing: 11, .. })));
    }

    /// The reorg case the hash-in-filename layout exists for: both records survive on disk, the
    /// lifecycle event says which one lost, and the canonical set is the winner alone.
    #[test]
    fn a_reorged_height_keeps_one_canonical_record_and_files_the_other() {
        let dir = temp_dir("reorg");
        let mut writer = started_writer(&dir);
        writer.write_record(&body(10, numbered(0x09), numbered(0x0a)).seal().unwrap()).unwrap();
        writer.write_record(&body(11, numbered(0x0a), numbered(0xb1)).seal().unwrap()).unwrap();
        writer
            .write_lifecycle(&LifecycleEvent::Reorg {
                common_ancestor: 10,
                abandoned: vec![(11, numbered(0xb1))],
            })
            .unwrap();
        writer.write_record(&body(11, numbered(0x0a), numbered(0xb2)).seal().unwrap()).unwrap();
        finish_all(writer, Some((10, 11, numbered(0xb2))), "").unwrap();

        let loaded = load_dataset(&dir).unwrap();
        assert_eq!(loaded.records.len(), 2);
        assert_eq!(loaded.records[1].body.block_hash, numbered(0xb2));
        assert_eq!(loaded.abandoned.len(), 1);
        assert_eq!(loaded.abandoned[0].body.block_hash, numbered(0xb1));
    }

    /// The walk needs a record at every height down the chain. A record whose parent nothing holds
    /// stops it, which is the same failure as a hole and is reported as one.
    #[test]
    fn a_chain_that_does_not_join_is_refused() {
        let dir = temp_dir("chain");
        let mut writer = started_writer(&dir);
        writer.write_record(&body(10, numbered(0x09), numbered(0x0a)).seal().unwrap()).unwrap();
        writer.write_record(&body(11, numbered(0xee), numbered(0x0b)).seal().unwrap()).unwrap();
        finish_all(writer, Some((10, 11, numbered(0x0b))), "").unwrap();
        assert!(matches!(load_dataset(&dir), Err(DatasetError::Gap { missing: 10, .. })));
    }

    #[test]
    fn a_capture_that_recorded_its_own_failure_is_refused() {
        let dir = temp_dir("failed");
        let writer = started_writer(&dir);
        writer
            .finish(DatasetEndKind::Failed, None, CONFIRMATIONS, None, "witness incomplete".into())
            .unwrap();
        assert!(matches!(load_dataset(&dir), Err(DatasetError::EndedBadly { .. })));
    }

    /// The dangerous reuse: no records left, but the previous run's terminator still there. A
    /// capture started here and killed would read as complete.
    #[test]
    fn a_directory_holding_only_a_stale_terminator_is_refused() {
        let dir = temp_dir("stale-end");
        fs::write(dir.join(END_FILE), b"{}").unwrap();
        assert!(matches!(
            PolicyDatasetWriter::create(&dir, &manifest()),
            Err(DatasetError::NotFresh(_))
        ));
    }

    #[test]
    fn a_terminator_that_miscounts_its_records_is_refused() {
        let dir = temp_dir("miscount");
        let mut writer = started_writer(&dir);
        writer.write_record(&body(10, numbered(0x09), numbered(0x0a)).seal().unwrap()).unwrap();
        // Written by hand: the writer itself cannot produce a disagreeing terminator, which is
        // exactly why the loader has to check one it did not write.
        let forged = DatasetEnd {
            kind: DatasetEndKind::BlockBudgetReached,
            records: 7,
            block_range: Some((10, 10)),
            usable_range: Some((10, 10)),
            usable_tip_hash: Some(numbered(0x0a)),
            confirmations: CONFIRMATIONS,
            confirmed_at_head: Some(18),
            detail: String::new(),
        };
        fs::write(dir.join(END_FILE), serde_json::to_vec(&forged).unwrap()).unwrap();
        assert!(matches!(load_dataset(&dir), Err(DatasetError::TerminatorMismatch { .. })));
    }

    /// The tail a capture wrote but never carried far enough past is dropped, not reported. A
    /// consumer that replayed it would be replaying blocks the producer declined to vouch for.
    #[test]
    fn records_past_the_usable_range_are_dropped_rather_than_replayed() {
        let dir = temp_dir("unconfirmed-tail");
        let mut writer = started_writer(&dir);
        for (number, parent, hash) in [
            (10, numbered(0x09), numbered(0x0a)),
            (11, numbered(0x0a), numbered(0x0b)),
            (12, numbered(0x0b), numbered(0x0c)),
        ] {
            writer.write_record(&body(number, parent, hash).seal().unwrap()).unwrap();
        }
        finish_all(writer, Some((10, 11, numbered(0x0b))), "block 12 was never confirmed").unwrap();

        let loaded = load_dataset(&dir).unwrap();
        assert_eq!(loaded.records.len(), 2);
        assert_eq!(loaded.records.last().unwrap().body.block_number, 11);
        assert_eq!(loaded.unconfirmed.len(), 1);
        assert!(loaded.abandoned.is_empty());
    }

    /// A capture that vouched for nothing yields nothing, whatever it managed to write.
    #[test]
    fn a_capture_that_confirmed_no_range_yields_no_blocks() {
        let dir = temp_dir("nothing-confirmed");
        let mut writer = started_writer(&dir);
        writer.write_record(&body(10, numbered(0x09), numbered(0x0a)).seal().unwrap()).unwrap();
        writer
            .finish(
                DatasetEndKind::ProducerShutdown,
                None,
                CONFIRMATIONS,
                Some(10),
                "stopped before anything settled".into(),
            )
            .unwrap();

        let loaded = load_dataset(&dir).unwrap();
        assert!(loaded.records.is_empty());
        assert_eq!(loaded.unconfirmed.len(), 1);
        assert!(loaded.abandoned.is_empty());
    }

    /// A record that lies about its own height cannot be walked through: the chain would silently
    /// skip or repeat a block.
    #[test]
    fn a_record_at_the_wrong_height_is_refused() {
        let dir = temp_dir("wrong-height");
        let mut writer = started_writer(&dir);
        writer.write_record(&body(10, numbered(0x09), numbered(0x0a)).seal().unwrap()).unwrap();
        // Claims height 9 while sitting under a block at 11.
        writer.write_record(&body(9, numbered(0x09), numbered(0xaa)).seal().unwrap()).unwrap();
        writer.write_record(&body(11, numbered(0xaa), numbered(0x0b)).seal().unwrap()).unwrap();
        write_end(
            &dir,
            DatasetEnd {
                kind: DatasetEndKind::BlockBudgetReached,
                records: 3,
                block_range: Some((9, 11)),
                usable_range: Some((9, 11)),
                usable_tip_hash: Some(numbered(0x0b)),
                confirmations: CONFIRMATIONS,
                confirmed_at_head: Some(11 + CONFIRMATIONS),
                detail: String::new(),
            },
        );
        assert!(matches!(load_dataset(&dir), Err(DatasetError::BrokenChain { .. })));
    }

    /// The case an accumulated list of abandoned hashes cannot represent: the chain leaves a branch
    /// and comes back to it. Walking parents from the recorded tip gets it right; a
    /// lifecycle-derived exclusion set would mark both branches abandoned and leave the contested
    /// height empty, refusing a corpus that is in fact sound.
    #[test]
    fn a_branch_that_is_abandoned_and_then_wins_again_is_canonical() {
        let dir = temp_dir("reorg-back");
        let mut writer = started_writer(&dir);
        writer.write_record(&body(10, numbered(0x09), numbered(0x0a)).seal().unwrap()).unwrap();
        let branch_a = body(11, numbered(0x0a), numbered(0xa1)).seal().unwrap();
        writer.write_record(&branch_a).unwrap();

        writer
            .write_lifecycle(&LifecycleEvent::Reorg {
                common_ancestor: 10,
                abandoned: vec![(11, numbered(0xa1))],
            })
            .unwrap();
        writer.write_record(&body(11, numbered(0x0a), numbered(0xb1)).seal().unwrap()).unwrap();

        // And back again. The same branch-A block is re-recorded without adding another physical
        // file or inflating the terminator's record count.
        writer
            .write_lifecycle(&LifecycleEvent::Reorg {
                common_ancestor: 10,
                abandoned: vec![(11, numbered(0xb1))],
            })
            .unwrap();
        writer.write_record(&branch_a).unwrap();
        writer.write_record(&body(12, numbered(0xa1), numbered(0x0c)).seal().unwrap()).unwrap();

        finish_all(writer, Some((10, 12, numbered(0x0c))), "done").unwrap();

        let loaded = load_dataset(&dir).unwrap();
        assert_eq!(
            loaded.records.iter().map(|record| record.body.block_hash).collect::<Vec<_>>(),
            vec![numbered(0x0a), numbered(0xa1), numbered(0x0c)],
            "the branch that won twice was not treated as canonical"
        );
        assert_eq!(loaded.abandoned.len(), 1);
        assert_eq!(loaded.abandoned[0].body.block_hash, numbered(0xb1));
        assert_eq!(loaded.end.records, 4, "the re-recorded branch-A block counted twice");
    }

    /// Rebuilds `record` until it serializes to different bytes than it does now.
    ///
    /// A round trip is how the producer's second observation of a returning branch arrives: fresh
    /// `HashMap`s, same contents, and `RandomState` re-seeds per map, so the iteration order — and
    /// nothing else — moves. It loops because one round trip can land back on the original order by
    /// chance; over an access set this wide, doing so 64 times running is not a case worth
    /// designing for.
    fn rebuilt_with_different_bytes(record: &PolicyDatasetRecord) -> PolicyDatasetRecord {
        let original = bincode::serialize(record).unwrap();
        for _ in 0..64 {
            let candidate: PolicyDatasetRecord = bincode::deserialize(&original).unwrap();
            if bincode::serialize(&candidate).unwrap() != original {
                return candidate
            }
        }
        panic!(
            "64 rebuilds all reproduced one byte order, which is not how HashMap iteration works"
        )
    }

    /// The returning branch of a reorg is the same record, and the writer has to see that.
    ///
    /// Its access set is rebuilt into fresh maps, so it serializes to different bytes while
    /// carrying identical contents. A writer that compared bytes would call that a conflicting
    /// duplicate and abort the capture over a block nothing was wrong with. Unlike
    /// [`a_branch_that_is_abandoned_and_then_wins_again_is_canonical`], which re-writes one
    /// in-memory record whose empty maps have only one order, this rebuilds the record the way the
    /// producer does.
    #[test]
    fn a_returning_branch_rebuilt_into_fresh_maps_is_the_same_record() {
        let dir = temp_dir("reorg-back-rebuilt");
        let mut writer = started_writer(&dir);
        let record = populated_body(false, 64).seal().unwrap();
        let path = writer.write_record(&record).unwrap();

        let rebuilt = rebuilt_with_different_bytes(&record);
        assert_eq!(
            rebuilt.body.digest(),
            record.body.digest(),
            "the rebuild changed the record's contents, so this tests the wrong thing"
        );

        let same_path = writer
            .write_record(&rebuilt)
            .expect("a rebuilt record with identical contents was refused as a conflict");
        assert_eq!(same_path, path);
        assert_eq!(writer.records(), 1, "the same block counted as two physical records");
    }

    /// A record damaged on disk is caught by the block that re-observes it, not left for the
    /// reader.
    #[test]
    fn a_duplicate_over_a_damaged_record_is_refused() {
        let dir = temp_dir("duplicate-over-damage");
        let mut writer = started_writer(&dir);
        let record = populated_body(false, 64).seal().unwrap();
        let path = writer.write_record(&record).unwrap();

        // Its stored digest no longer covers its body: the file is damaged, whatever damaged it.
        let mut damaged: PolicyDatasetRecord =
            bincode::deserialize(&fs::read(&path).unwrap()).unwrap();
        damaged.body.expected_state_root = numbered(0xee);
        fs::write(&path, bincode::serialize(&damaged).unwrap()).unwrap();

        let err = writer.write_record(&record).unwrap_err();
        assert!(
            matches!(err, DatasetError::DigestMismatch { block_number: 10, .. }),
            "expected the damage to be named, got {err}"
        );
    }

    /// One block identity cannot conceal two different captures. Treating the later write as an
    /// ordinary duplicate would make the result depend on which observation happened to win the
    /// filesystem overwrite.
    #[test]
    fn a_duplicate_identity_with_different_contents_is_refused() {
        let dir = temp_dir("conflicting-duplicate");
        let mut writer = started_writer(&dir);
        let first = body(10, numbered(0x09), numbered(0x0a)).seal().unwrap();
        writer.write_record(&first).unwrap();

        let mut conflicting = body(10, numbered(0xff), numbered(0x0a));
        conflicting.expected_state_root = numbered(0xee);
        let err = writer.write_record(&conflicting.seal().unwrap()).unwrap_err();
        assert!(matches!(
            err,
            DatasetError::ConflictingDuplicate {
                block_number: 10,
                block_hash
            } if block_hash == numbered(0x0a)
        ));
        assert_eq!(writer.records(), 1, "the refused duplicate changed the physical count");
    }

    /// A terminator can name any depth it likes; the head it recorded has to back it up.
    #[test]
    fn a_confirmation_claim_with_no_head_behind_it_is_refused() {
        let dir = temp_dir("unbacked-confirmation");
        let mut writer = started_writer(&dir);
        writer.write_record(&body(10, numbered(0x09), numbered(0x0a)).seal().unwrap()).unwrap();

        let mut end = DatasetEnd {
            kind: DatasetEndKind::BlockBudgetReached,
            records: 1,
            block_range: Some((10, 10)),
            usable_range: Some((10, 10)),
            usable_tip_hash: Some(numbered(0x0a)),
            confirmations: CONFIRMATIONS,
            confirmed_at_head: None,
            detail: String::new(),
        };
        write_end(&dir, end.clone());
        assert!(
            matches!(load_dataset(&dir), Err(DatasetError::TerminatorMismatch { .. })),
            "a range vouched for with no head at all was accepted"
        );

        end.confirmed_at_head = Some(10 + CONFIRMATIONS - 1);
        write_end(&dir, end.clone());
        assert!(
            matches!(load_dataset(&dir), Err(DatasetError::TerminatorMismatch { .. })),
            "a head one block short of the claimed depth was accepted"
        );

        // Overflow is refused rather than wrapping into a claim that looks satisfied.
        end.confirmations = u64::MAX;
        end.confirmed_at_head = Some(u64::MAX);
        write_end(&dir, end.clone());
        assert!(matches!(load_dataset(&dir), Err(DatasetError::TerminatorMismatch { .. })));

        end.confirmations = CONFIRMATIONS;
        end.confirmed_at_head = Some(10 + CONFIRMATIONS);
        write_end(&dir, end);
        assert!(load_dataset(&dir).is_ok(), "a properly backed claim was refused");
    }

    /// A usable range with no tip hash cannot be walked, so it cannot be trusted either.
    #[test]
    fn a_usable_range_with_no_tip_to_walk_from_is_refused() {
        let dir = temp_dir("no-tip");
        let mut writer = started_writer(&dir);
        writer.write_record(&body(10, numbered(0x09), numbered(0x0a)).seal().unwrap()).unwrap();
        write_end(
            &dir,
            DatasetEnd {
                kind: DatasetEndKind::BlockBudgetReached,
                records: 1,
                block_range: Some((10, 10)),
                usable_range: Some((10, 10)),
                usable_tip_hash: None,
                confirmations: CONFIRMATIONS,
                confirmed_at_head: Some(10 + CONFIRMATIONS),
                detail: String::new(),
            },
        );
        assert!(matches!(load_dataset(&dir), Err(DatasetError::TerminatorMismatch { .. })));
    }

    /// The log is the only record of what the chain did under the capture. Without it the
    /// exclusions cannot be audited, so the dataset is refused rather than read.
    #[test]
    fn a_dataset_without_its_lifecycle_log_is_refused() {
        let dir = temp_dir("no-lifecycle");
        let mut writer = started_writer(&dir);
        writer.write_record(&body(10, numbered(0x09), numbered(0x0a)).seal().unwrap()).unwrap();
        finish_all(writer, Some((10, 10, numbered(0x0a))), "").unwrap();
        assert!(load_dataset(&dir).is_ok());

        fs::remove_file(dir.join(LIFECYCLE_FILE)).unwrap();
        assert!(matches!(load_dataset(&dir), Err(DatasetError::MissingLifecycle(_))));
    }

    #[test]
    fn a_second_capture_refuses_to_append_to_an_existing_one() {
        let dir = temp_dir("append");
        let mut writer = started_writer(&dir);
        writer.write_record(&body(10, numbered(0x09), numbered(0x0a)).seal().unwrap()).unwrap();
        assert!(PolicyDatasetWriter::create(&dir, &manifest()).is_err());
    }
}
