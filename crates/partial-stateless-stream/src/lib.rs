//! One event format for recording, deterministic replay, and live delivery.
//!
//! A file spool and a socket carry the **same frame bytes**. That is not tidiness: a recorded
//! corpus in a format the live path never uses would be a corpus that proves nothing about the
//! live path, and a live failure would be unreproducible by construction. So this crate defines
//! bytes and nothing else — there is no producer and no consumer here, and no filesystem or
//! socket code. What it exports is an envelope, seven frame kinds covering the six events S3 and
//! S4 need, and one structural guarantee.
//!
//! **The structural guarantee is the oracle split.** A commit frame carries the recording
//! producer's own outcome for its block, and that outcome must never become an input to the
//! validator being checked against it. Comments cannot enforce this and reviewers forget. So the
//! dependency arrow does it: [`CommitOracle`] lives here, this crate depends on
//! `partial-stateless-validator`, and the reverse dependency cannot be added — it would be a cycle
//! cargo refuses to build. No code inside the validator can name a `CommitOracle` on any branch,
//! and no future change can quietly make one nameable. A decoded commit hands back a
//! [`CommitInput`] and a [`CommitOracle`] as two values, and the validator's entry points take
//! only the first. It is the same technique the database-free claim rests on: make the wrong thing
//! unnameable rather than merely unreached.
//!
//! What this crate does not decide: whether the checkpoint it carries is trustworthy, which chain
//! is canonical, or whether a producer is honest. It carries what it was given and refuses what it
//! cannot parse.

#![doc(
    html_logo_url = "https://raw.githubusercontent.com/paradigmxyz/reth/main/assets/reth-docs.png",
    html_favicon_url = "https://avatars0.githubusercontent.com/u/97369466?s=256",
    issue_tracker_base_url = "https://github.com/paradigmxyz/reth/issues/"
)]
#![cfg_attr(not(test), warn(unused_crate_dependencies))]

pub mod event;
pub mod frame;
pub mod oracle;

pub use event::{
    BlockRef, Checkpoint, CommitFrame, CommitInput, End, EndKind, Manifest, Reorg, Reset,
    ResetReason, SnapshotChunk, SnapshotError, StreamEvent, DEFAULT_MAX_SNAPSHOT_BYTES,
    MAX_SNAPSHOT_CHUNKS,
};
pub use frame::{
    decode_body, decode_frame, decode_header, encode_frame, encode_frame_bytes, FrameError,
    FrameHeader, FrameKind, FrameLimits, DEFAULT_MAX_FRAME_BYTES, DEFAULT_SNAPSHOT_CHUNK_BYTES,
    FORMAT_VERSION, FRAME_HEADER_BYTES, FRAME_MAGIC,
};
pub use oracle::{CommitOracle, RecordedVerdict};

/// Decodes one frame into the event its kind names.
///
/// Returns the header beside the event, because a consumer needs the sequence number to notice a
/// gap and the event body deliberately does not carry it — a body that repeated the sequence would
/// be a second place for it to be wrong.
pub fn decode_event<'a>(
    bytes: &'a [u8],
    limits: &FrameLimits,
) -> Result<(FrameHeader, StreamEvent, &'a [u8]), FrameError> {
    let (header, payload, rest) = decode_frame(bytes, limits)?;
    let event = match header.kind {
        FrameKind::Manifest => StreamEvent::Manifest(decode_body(header.kind, payload)?),
        FrameKind::Checkpoint => StreamEvent::Checkpoint(decode_body(header.kind, payload)?),
        FrameKind::SnapshotChunk => StreamEvent::SnapshotChunk(decode_body(header.kind, payload)?),
        FrameKind::Commit => StreamEvent::Commit(decode_body(header.kind, payload)?),
        FrameKind::Reorg => StreamEvent::Reorg(decode_body(header.kind, payload)?),
        FrameKind::Reset => StreamEvent::Reset(decode_body(header.kind, payload)?),
        FrameKind::End => StreamEvent::End(decode_body(header.kind, payload)?),
    };
    Ok((header, event, rest))
}

/// Serializes one event's body without framing it, naming the kind the frame must carry.
///
/// For a producer that buffers events and assigns sequence numbers later than it serializes:
/// the digest covers the body alone, so a body serialized now and framed with
/// [`encode_frame_bytes`] at flush time is byte-identical to one encoded in a single step.
pub fn encode_event_body(event: &StreamEvent) -> Result<(FrameKind, Vec<u8>), FrameError> {
    let (kind, result) = match event {
        StreamEvent::Manifest(body) => (FrameKind::Manifest, bincode::serialize(body)),
        StreamEvent::Checkpoint(body) => (FrameKind::Checkpoint, bincode::serialize(body)),
        StreamEvent::SnapshotChunk(body) => (FrameKind::SnapshotChunk, bincode::serialize(body)),
        StreamEvent::Commit(body) => (FrameKind::Commit, bincode::serialize(body)),
        StreamEvent::Reorg(body) => (FrameKind::Reorg, bincode::serialize(body)),
        StreamEvent::Reset(body) => (FrameKind::Reset, bincode::serialize(body)),
        StreamEvent::End(body) => (FrameKind::End, bincode::serialize(body)),
    };
    let body =
        result.map_err(|err| FrameError::Body { kind: kind.as_str(), detail: err.to_string() })?;
    Ok((kind, body))
}

/// Encodes one event as a complete frame.
///
/// Takes the same limits a decoder applies, and refuses at encode what no decoder could read —
/// the producer is the one party that can still say *why* a frame was too large.
pub fn encode_event(
    sequence: u64,
    event: &StreamEvent,
    limits: &FrameLimits,
) -> Result<Vec<u8>, FrameError> {
    match event {
        StreamEvent::Manifest(body) => encode_frame(FrameKind::Manifest, sequence, body, limits),
        StreamEvent::Checkpoint(body) => {
            encode_frame(FrameKind::Checkpoint, sequence, body, limits)
        }
        StreamEvent::SnapshotChunk(body) => {
            encode_frame(FrameKind::SnapshotChunk, sequence, body, limits)
        }
        StreamEvent::Commit(body) => encode_frame(FrameKind::Commit, sequence, body, limits),
        StreamEvent::Reorg(body) => encode_frame(FrameKind::Reorg, sequence, body, limits),
        StreamEvent::Reset(body) => encode_frame(FrameKind::Reset, sequence, body, limits),
        StreamEvent::End(body) => encode_frame(FrameKind::End, sequence, body, limits),
    }
}
