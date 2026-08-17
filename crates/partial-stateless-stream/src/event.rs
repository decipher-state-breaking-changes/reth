//! What each frame kind carries.
//!
//! The bodies are plain data and deliberately hold no behaviour. Two encoding choices in here are
//! decisions rather than conveniences, and both are about keeping a replay honest.
//!
//! **The payload travels as Engine-API JSON.** Not because JSON is compact — it is not, and it
//! roughly doubles the transaction bytes — but because that is the form a consensus client
//! actually sends. A replay driver that decodes it is running the same deserialization a live node
//! runs, so `input_decode_us` measures a real cost and a payload that only *this* codec could
//! parse can never enter the corpus. `ExecutionPayload` is a `serde(untagged)` enum whose
//! deserializer picks a version from the fields present, which is exactly the discrimination a
//! live node performs and exactly what a self-describing binary codec would have replaced with its
//! own.
//!
//! **The sidecar travels as the bytes the producer already wrote.** It is embedded verbatim rather
//! than re-encoded, so a frame's sidecar and the spool file's sidecar cannot drift, and a
//! consumer's decode is the same decode the live verifier performs.

use crate::oracle::CommitOracle;
use alloy_primitives::B256;
use alloy_rpc_types_engine::ExecutionData;
use partial_stateless_validator::PayloadProvenance;
use serde::{Deserialize, Serialize};

/// A block, addressed the only way that stays unambiguous mid-reorg.
///
/// The hash is not decoration beside the number. A height names whichever block the producer
/// currently calls canonical, and every event in this format that refers to a *past* block —
/// a common ancestor, an abandoned branch, a retained generation — refers to one that a reorg may
/// be in the middle of replacing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockRef {
    /// Height.
    pub number: u64,
    /// Hash.
    pub hash: B256,
}

/// Stream identity and the policy every later frame is relative to.
///
/// Exactly one per stream, first. A consumer that accepts frames before reading this has accepted
/// them without knowing which chain or which cache policy they describe, and both are things a
/// restored pair is checked against rather than told.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    /// Chain the producer was following.
    pub chain_id: u64,
    /// Genesis hash, so two streams on the same chain id but different networks cannot be mixed.
    pub genesis_hash: B256,
    /// The cache policy identifier every anchor in this stream is scoped to.
    ///
    /// A label rather than the policy itself: it identifies the *object* only as far as both sides
    /// derive it from the same configuration. That gap is failure mode 11 and this format does not
    /// close it.
    pub cache_policy_id: B256,
    /// Blocks an account entry survives without access.
    pub account_window: u64,
    /// Blocks a storage entry survives without access.
    pub storage_window: u64,
    /// Increments whenever the producer restarts its stream from a new snapshot.
    ///
    /// A consumer that sees a new epoch must re-bootstrap rather than continue. The sequence
    /// space does *not* restart with it in a file spool — one directory has one sequence space,
    /// and frame files are named by it — so continuity of numbering is exactly what may not be
    /// read as continuity of state. The epoch is what says the state broke; the numbering only
    /// says nothing was lost in transit.
    pub epoch: u64,
    /// Producer build identity, recorded so a divergence has somewhere to start.
    pub producer: String,
    /// Sequence number of the frame that follows this one.
    ///
    /// Always this manifest's own sequence plus one. It is written down anyway because it is the
    /// one field that ties the manifest to *where it sits*: a manifest lifted out of another
    /// spool, or written at the wrong place in this one, disagrees with its own position.
    pub first_sequence: u64,
}

/// Why a manifest is not usable where it was found.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ManifestError {
    /// The manifest describes a chain or a cache policy the stream was not already about.
    #[error("the manifest describes a different {field}")]
    Identity {
        /// Which field disagreed.
        field: &'static str,
    },
    /// The epoch did not follow the one before it.
    #[error("the manifest names epoch {found} where epoch {expected} follows the stream")]
    Epoch {
        /// What it said.
        found: u64,
        /// What the position allowed.
        expected: u64,
    },
    /// The manifest disagrees with the sequence it was written at.
    #[error("the manifest at sequence {sequence} names first_sequence {found}, and its own \
             successor is {}", .sequence + 1)]
    Position {
        /// Where it is.
        sequence: u64,
        /// What it claimed.
        found: u64,
    },
}

impl Manifest {
    /// Whether this manifest can open a stream at `sequence`.
    ///
    /// Nothing precedes it, so only its own position is checkable. Identity is the operator's to
    /// judge, and a consumer does that against its own configuration rather than against the
    /// frame.
    pub const fn check_opens(&self, sequence: u64) -> Result<(), ManifestError> {
        if self.first_sequence != sequence + 1 {
            return Err(ManifestError::Position { sequence, found: self.first_sequence })
        }
        Ok(())
    }

    /// Whether this manifest is the next epoch of the stream `previous` opened.
    ///
    /// A second manifest in one spool is how a restarted producer says its state broke and a
    /// consumer has to re-bootstrap. It is only that if it is about the same chain under the same
    /// policy and numbers itself as the successor. Anything else is a different stream sharing a
    /// directory, and nothing under it may be restored as though it continued this one.
    pub fn check_succeeds(&self, previous: &Self, sequence: u64) -> Result<(), ManifestError> {
        let field = if self.chain_id != previous.chain_id {
            Some("chain")
        } else if self.genesis_hash != previous.genesis_hash {
            Some("genesis")
        } else if self.cache_policy_id != previous.cache_policy_id {
            Some("cache policy")
        } else if self.account_window != previous.account_window ||
            self.storage_window != previous.storage_window
        {
            Some("cache window")
        } else {
            None
        };
        if let Some(field) = field {
            return Err(ManifestError::Identity { field })
        }
        if self.epoch != previous.epoch + 1 {
            return Err(ManifestError::Epoch { found: self.epoch, expected: previous.epoch + 1 })
        }
        self.check_opens(sequence)
    }
}

/// The operator-trusted checkpoint, and the header of the snapshot that follows it.
///
/// Carries the accepted-head header, which the standalone bring-up found missing: a pair restored
/// from a snapshot has no header for its own anchor, so it cannot check the parent-dependent
/// consensus rules on its first child and must wait a block to become useful. The header is
/// installable only because everything about it is checked against the rest of this struct — its
/// hash, its number, and its state root — which is the same rule the live path applies and not a
/// weaker one for restored pairs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Checkpoint {
    /// The block the caches are the state after.
    pub block: BlockRef,
    /// Canonical state root at that block.
    pub state_root: B256,
    /// Cache root at that block, under this stream's policy.
    pub cache_root: B256,
    /// Policy the cache root is scoped to; must equal the manifest's.
    pub cache_policy_id: B256,
    /// RLP of the accepted head header, so a restored pair can admit its first child.
    ///
    /// Empty when the producer had none, which is legitimate: a pair one block out of a cold reset
    /// holds a generation it has no header for.
    pub accepted_head_rlp: Vec<u8>,
    /// Total bytes of the snapshot package split across the chunks that follow.
    pub snapshot_bytes: u64,
    /// Number of [`SnapshotChunk`] frames that follow.
    pub snapshot_chunks: u32,
    /// keccak256 of the whole package, so reassembly is checked against one value rather than
    /// against the per-frame digests alone.
    pub snapshot_digest: B256,
}

/// One slice of the package the preceding [`Checkpoint`] described.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotChunk {
    /// Zero-based position in the package.
    pub index: u32,
    /// The slice.
    pub bytes: Vec<u8>,
}

/// Default ceiling on a declared snapshot package.
///
/// The first recorded mainnet capture's package was 121.8 MiB, so 2 GiB — the same bound the
/// restore path applies to its own proof — leaves an order of magnitude of headroom while keeping a
/// corrupt or hostile declaration from becoming a 16 EiB allocation.
pub const DEFAULT_MAX_SNAPSHOT_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Ceiling on the declared chunk count.
///
/// At the default 8 MiB chunk, 8192 chunks is a 64 GiB package — far past the byte bound above,
/// so this only ever fires on a declaration that is wrong on its own terms.
pub const MAX_SNAPSHOT_CHUNKS: u32 = 8192;

impl Checkpoint {
    /// Fills in the snapshot transport fields from the package, without materializing chunks.
    ///
    /// A producer writing chunks one at a time uses this and then frames each slice itself, so
    /// the whole package is never copied a second time; [`Self::chunk`] is the convenience that
    /// does both at once. Chunking lives with the format rather than with the producer because
    /// reassembly has to be its exact inverse, and the two ends are written months and one
    /// process apart.
    pub fn describe(&mut self, package: &[u8], chunk_bytes: usize) {
        self.snapshot_bytes = package.len() as u64;
        self.snapshot_chunks = package.chunks(chunk_bytes.max(1)).count() as u32;
        self.snapshot_digest = alloy_primitives::keccak256(package);
    }

    /// Splits a package into the chunks that follow this checkpoint, and fills in its header.
    pub fn chunk(&mut self, package: &[u8], chunk_bytes: usize) -> Vec<SnapshotChunk> {
        self.describe(package, chunk_bytes);
        package
            .chunks(chunk_bytes.max(1))
            .enumerate()
            .map(|(index, bytes)| SnapshotChunk { index: index as u32, bytes: bytes.to_vec() })
            .collect()
    }

    /// Checks that the declared snapshot sizes are ones a consumer can honour.
    ///
    /// Runs on the declaration alone, before any chunk is buffered, because the declaration is
    /// what a consumer would otherwise size its buffers from — and a corrupt or hostile
    /// checkpoint must not be able to turn its own claim into an allocation.
    pub const fn validate_declared(&self, max_snapshot_bytes: u64) -> Result<(), SnapshotError> {
        if self.snapshot_bytes > max_snapshot_bytes {
            return Err(SnapshotError::DeclaredTooLarge {
                declared: self.snapshot_bytes,
                limit: max_snapshot_bytes,
            })
        }
        if self.snapshot_chunks > MAX_SNAPSHOT_CHUNKS {
            return Err(SnapshotError::DeclaredTooManyChunks {
                declared: self.snapshot_chunks,
                limit: MAX_SNAPSHOT_CHUNKS,
            })
        }
        Ok(())
    }

    /// Reassembles the package this checkpoint described, or says how it disagrees.
    ///
    /// Checks the declaration, count, order, total length, and digest, because each catches a
    /// delivery failure the others do not. A chunk delivered twice passes the digest only if the
    /// length also matches; a reordered pair passes both length and count. The length is also
    /// enforced while accumulating, so chunks that oversize the declared package stop being
    /// copied at the first byte past it rather than after all of them.
    pub fn reassemble(&self, chunks: &[SnapshotChunk]) -> Result<Vec<u8>, SnapshotError> {
        self.validate_declared(DEFAULT_MAX_SNAPSHOT_BYTES)?;
        if chunks.len() != self.snapshot_chunks as usize {
            return Err(SnapshotError::ChunkCount {
                expected: self.snapshot_chunks,
                actual: chunks.len() as u32,
            })
        }
        let mut package = Vec::with_capacity(self.snapshot_bytes as usize);
        for (position, chunk) in chunks.iter().enumerate() {
            if chunk.index as usize != position {
                return Err(SnapshotError::OutOfOrder {
                    expected: position as u32,
                    actual: chunk.index,
                })
            }
            let projected = package.len().saturating_add(chunk.bytes.len());
            if projected as u64 > self.snapshot_bytes {
                return Err(SnapshotError::Length {
                    expected: self.snapshot_bytes,
                    actual: projected as u64,
                })
            }
            package.extend_from_slice(&chunk.bytes);
        }
        if package.len() as u64 != self.snapshot_bytes {
            return Err(SnapshotError::Length {
                expected: self.snapshot_bytes,
                actual: package.len() as u64,
            })
        }
        let digest = alloy_primitives::keccak256(&package);
        if digest != self.snapshot_digest {
            return Err(SnapshotError::Digest { expected: self.snapshot_digest, actual: digest })
        }
        Ok(package)
    }
}

/// How a delivered snapshot disagreed with the checkpoint that described it.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SnapshotError {
    /// The checkpoint declared a package larger than this consumer will hold.
    #[error("checkpoint declares a {declared}-byte snapshot; this consumer's bound is {limit}")]
    DeclaredTooLarge {
        /// Bytes the checkpoint declared.
        declared: u64,
        /// The configured bound.
        limit: u64,
    },
    /// The checkpoint declared more chunks than any package inside the byte bound needs.
    #[error("checkpoint declares {declared} snapshot chunks; the bound is {limit}")]
    DeclaredTooManyChunks {
        /// Chunks the checkpoint declared.
        declared: u32,
        /// The configured bound.
        limit: u32,
    },
    /// A different number of chunks arrived than were announced.
    #[error("snapshot has {actual} chunks; the checkpoint announced {expected}")]
    ChunkCount {
        /// Chunks the checkpoint announced.
        expected: u32,
        /// Chunks that arrived.
        actual: u32,
    },
    /// A chunk arrived out of position.
    #[error("snapshot chunk {actual} arrived where {expected} was expected")]
    OutOfOrder {
        /// Index the position calls for.
        expected: u32,
        /// Index that arrived.
        actual: u32,
    },
    /// The reassembled package is the wrong size.
    #[error("reassembled snapshot is {actual} bytes; the checkpoint announced {expected}")]
    Length {
        /// Bytes the checkpoint announced.
        expected: u64,
        /// Bytes that arrived.
        actual: u64,
    },
    /// The reassembled package is the right size and the wrong bytes.
    #[error("reassembled snapshot digest {actual} does not match the announced {expected}")]
    Digest {
        /// Digest the checkpoint announced.
        expected: B256,
        /// Digest of what arrived.
        actual: B256,
    },
}

/// One canonical block, split into what a validator may read and what it must not.
///
/// The split is the whole design. An expectation that a validator can read while validating is not
/// an expectation, and the ways that goes wrong are quiet: a state root used to shortcut a root
/// walk, an expected miss set used to skip a policy check, a fingerprint used to decide the answer
/// was already known. `partial-stateless-validator` cannot name [`CommitOracle`] — the dependency
/// arrow runs from this crate to it — so the isolation is a property of the dependency graph and
/// not a convention anyone has to remember.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommitFrame {
    input: CommitInput,
    oracle: CommitOracle,
}

impl CommitFrame {
    /// Builds a commit from its two halves.
    pub const fn new(input: CommitInput, oracle: CommitOracle) -> Self {
        Self { input, oracle }
    }

    /// Separates the input from the expectation.
    ///
    /// The only way to reach either. A caller that wanted the oracle has to take the input with
    /// it, and a caller that wanted the input has to name the oracle it is putting aside.
    pub fn split(self) -> (CommitInput, CommitOracle) {
        (self.input, self.oracle)
    }

    /// The input, for a caller that is going to validate it.
    pub const fn input(&self) -> &CommitInput {
        &self.input
    }

    /// The recorded expectation, for a caller that is going to compare against it.
    pub const fn oracle(&self) -> &CommitOracle {
        &self.oracle
    }
}

/// Everything a validator is allowed to see about a commit.
///
/// Nothing derived from the producer's own execution is in here. The sidecar is, because the
/// sidecar is the block's *transport* and not the producer's conclusion: a validator re-derives
/// every claim in it and rejects the block if any disagrees.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommitInput {
    /// The block this commit is about.
    pub block: BlockRef,
    /// Parent hash, so a consumer can bind the commit to the branch it expects.
    pub parent_hash: B256,
    /// How much a validator's admission checks on `payload` would be proving.
    pub payload_provenance: PayloadProvenance,
    /// Engine-API JSON, exactly as a consensus client sends it.
    ///
    /// `None` only when [`PayloadProvenance::Absent`] — a producer that obtained no payload and
    /// derived none. Such a commit can still be replayed through the recovered entry point; it
    /// cannot be replayed through admission, and a driver that tried would be reporting coverage
    /// it does not have.
    pub payload_json: Option<Vec<u8>>,
    /// The sidecar as the producer serialized it.
    pub sidecar: Vec<u8>,
}

impl CommitInput {
    /// Parses the recorded payload with the deserializer a live node uses.
    ///
    /// Returns `Ok(None)` when the commit carries no payload, which is a fact about the recording
    /// rather than a failure of it.
    pub fn payload(&self) -> Result<Option<ExecutionData>, serde_json::Error> {
        self.payload_json.as_deref().map(serde_json::from_slice).transpose()
    }
}

/// A branch was abandoned.
///
/// The winning branch is not in here. It arrives as ordinary [`CommitFrame`]s after this one, so a
/// consumer applies exactly the same code to a post-reorg block as to any other, and a producer
/// cannot accidentally define a second way for a block to be applied.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reorg {
    /// The last block both branches share.
    pub common_ancestor: BlockRef,
    /// Blocks leaving the canonical chain, lowest first.
    pub abandoned: Vec<BlockRef>,
    /// The tip the producer is moving to, so a consumer knows when the branch is complete.
    ///
    /// `None` on a pure revert, where the chain is unwinding to `common_ancestor` and nothing
    /// replaces the abandoned blocks.
    pub winning_tip: Option<BlockRef>,
}

/// The consumer cannot continue from what it holds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reset {
    /// Why.
    pub reason: ResetReason,
    /// Free text for the run log. Never parsed.
    pub detail: String,
}

/// Why a stream told its consumer to stop and re-bootstrap.
///
/// Every one of these is a case that must not be delivered as a silent drop. A consumer that
/// skipped a block and kept publishing verdicts would be publishing them about a state the chain
/// never passed through.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResetReason {
    /// A block is missing between two the consumer received.
    Gap,
    /// The producer restarted its stream; sequence numbers restarted with it.
    EpochChange,
    /// Two different frames claimed the same sequence.
    DuplicateConflict,
    /// A frame arrived below the consumer's watermark.
    OutOfOrder,
    /// The producer's own queue overflowed.
    Overflow,
    /// The producer's state moved somewhere no incremental event can express.
    SnapshotRequired,
}

/// Why the producer stopped, as a value a reader may act on.
///
/// [`End::reason`] is free text for the run log; this is the parsed field. An `End` frame of any
/// kind means the writer ran its close path — orderly termination, never success on its own. The
/// producer writes one on a clean shutdown, on an error return, and on a panic unwind alike, so a
/// reader judging an outcome reads the kind, not the presence of the frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndKind {
    /// The producer stopped on request: the notification stream ended, or the process was told
    /// to shut down and its teardown ran.
    Shutdown,
    /// The producer hit an internal failure and closed the stream deliberately rather than
    /// leaving it to be read as cut.
    ProducerFault,
    /// A configured spool bound was reached. The stream is complete up to the bound.
    SpoolLimit,
    /// The snapshot export failed, so the stream never opened or could not continue.
    ExportFailure,
}

impl EndKind {
    /// Stable name for logs and records.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Shutdown => "shutdown",
            Self::ProducerFault => "producer_fault",
            Self::SpoolLimit => "spool_limit",
            Self::ExportFailure => "export_failure",
        }
    }
}

/// The producer has stopped.
///
/// A stream that ends without one of these was cut — the writer never ran its close path — and
/// the distinction matters to a replay driver deciding whether a short corpus is a short corpus
/// or a truncated one. What this frame does *not* mean is that the run succeeded: the writer's
/// destructor emits one on failures too, and [`EndKind`] is what carries the difference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct End {
    /// Why, as a value. See [`EndKind`].
    pub kind: EndKind,
    /// Free text for the run log. Never parsed.
    pub reason: String,
    /// Sequence of the last frame before this one, so a consumer can check it saw them all.
    ///
    /// The End frame's own header sequence is therefore always `last_sequence + 1`, and a reader
    /// checks exactly that equality.
    pub last_sequence: u64,
}

/// One decoded frame body.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// Stream identity and policy.
    Manifest(Manifest),
    /// Trusted checkpoint and snapshot header.
    Checkpoint(Checkpoint),
    /// One slice of the snapshot package.
    SnapshotChunk(SnapshotChunk),
    /// One canonical block.
    Commit(Box<CommitFrame>),
    /// An abandoned branch.
    Reorg(Reorg),
    /// A demand to re-bootstrap.
    Reset(Reset),
    /// End of stream.
    End(End),
}
