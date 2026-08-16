//! Structural readiness inspection of a spool, for the gate's follower-start precondition.
//!
//! Both 1,001-block preflights started their follower against an empty pre-checkpoint spool and
//! burned their raw latency tails on the mislabeled startup backlog. The runbook rule — "the
//! follower starts only after the checkpoint exists" — is enforced here as a *structural* check
//! rather than a file-existence probe, because a resumed spool holds previous epochs' checkpoint
//! files and a name glob would wave the gate through on one of those.
//!
//! Ready means: the manifest chain verifies link by link, the **current** epoch holds a
//! checkpoint, every declared snapshot chunk is present with the declared digest and byte count,
//! the checkpoint carries a decodable accepted head, and the epoch has not already closed below
//! it. Anything less is a named refusal, never a guess — and the refusal says whether waiting
//! can fix it. `pending` is absence: the producer has not written the thing yet, and a polling
//! gate keeps waiting. `invalid` is contradiction: a frame that is present and wrong, a broken
//! chain, an epoch that closed checkpointless — waiting cannot repair any of these, and a gate
//! that polled through its whole deadline on one would report a corruption as a timeout.

use crate::tail::SpoolTail;
use alloy_primitives::Keccak256;
use partial_stateless_stream::{FrameKind, FrameLimits, StreamEvent};
use std::path::Path;

/// What the inspection concluded, with the evidence a gate log wants.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SpoolReadiness {
    /// Whether a follower may start against this spool now.
    pub ready: bool,
    /// `"ready"`, `"pending"` (not written yet; polling can succeed), or `"invalid"`
    /// (structurally wrong; polling cannot).
    pub state: &'static str,
    /// The current (last) epoch's number, once at least one manifest verified.
    pub epoch: Option<u64>,
    /// The current epoch's checkpoint announce sequence, when one exists.
    pub checkpoint_sequence: Option<u64>,
    /// Declared snapshot chunks behind it.
    pub chunks: Option<u32>,
    /// Why the spool is not ready, when it is not.
    pub reason: Option<String>,
}

impl SpoolReadiness {
    fn pending(epoch: Option<u64>, reason: impl Into<String>) -> Self {
        Self {
            ready: false,
            state: "pending",
            epoch,
            checkpoint_sequence: None,
            chunks: None,
            reason: Some(reason.into()),
        }
    }

    fn invalid(epoch: Option<u64>, reason: impl Into<String>) -> Self {
        Self {
            ready: false,
            state: "invalid",
            epoch,
            checkpoint_sequence: None,
            chunks: None,
            reason: Some(reason.into()),
        }
    }
}

/// Inspects a spool without following it. Errors are I/O-shaped; a spool that is merely not
/// ready yet is a normal `ready: false` answer, because the gate polls this in a loop.
///
/// A visible `.frame` file is complete — the producer writes to a `.tmp` name and renames — so
/// every fault on a file that *exists* is classified `invalid`, never waited out.
pub fn inspect_ready(dir: &Path) -> eyre::Result<SpoolReadiness> {
    let tail = SpoolTail::new(dir, FrameLimits::default());

    // Walk the manifest chain from the spool's origin: each link is read back and verified the
    // way the follower would, so a manifest lifted out of another spool cannot vouch for an
    // epoch here.
    let mut previous: Option<(u64, partial_stateless_stream::Manifest)> = None;
    let mut from = 0u64;
    loop {
        let found = match tail.scan_for(FrameKind::Manifest, from) {
            Ok(found) => found,
            Err(fault) => return Ok(SpoolReadiness::invalid(None, fault.to_string())),
        };
        let Some(sequence) = found else { break };
        let frame = match tail.read_at(sequence, FrameKind::Manifest) {
            Ok(frame) => frame,
            Err(fault) => return Ok(SpoolReadiness::invalid(None, fault.to_string())),
        };
        let StreamEvent::Manifest(manifest) = frame.event else {
            return Ok(SpoolReadiness::invalid(
                None,
                format!(
                    "the frame at sequence {sequence} is named Manifest but decoded as \
                         something else"
                ),
            ))
        };
        let checked = match &previous {
            None => manifest.check_opens(sequence).map_err(|err| err.to_string()),
            Some((_, prior)) => {
                manifest.check_succeeds(prior, sequence).map_err(|err| err.to_string())
            }
        };
        if let Err(reason) = checked {
            return Ok(SpoolReadiness::invalid(
                previous.map(|(_, manifest)| manifest.epoch),
                format!("manifest chain broken at sequence {sequence}: {reason}"),
            ))
        }
        previous = Some((sequence, manifest));
        from = sequence + 1;
    }
    let Some((manifest_at, manifest)) = previous else {
        return Ok(SpoolReadiness::pending(None, "no manifest yet"))
    };
    let epoch = manifest.epoch;

    let checkpoint_at = match tail.scan_for(FrameKind::Checkpoint, manifest_at + 1) {
        Ok(found) => found,
        Err(fault) => return Ok(SpoolReadiness::invalid(Some(epoch), fault.to_string())),
    };
    let Some(checkpoint_at) = checkpoint_at else {
        return Ok(SpoolReadiness::pending(Some(epoch), "the current epoch has no checkpoint yet"))
    };
    // An End below the checkpoint means this epoch closed without one; whatever sits above the
    // End belongs to no epoch a follower may start into. The producer that would write a fresh
    // epoch already gave up, so waiting cannot fix it.
    match tail.scan_for(FrameKind::End, manifest_at + 1) {
        Ok(Some(end_at)) if end_at < checkpoint_at => {
            return Ok(SpoolReadiness::invalid(
                Some(epoch),
                format!("the epoch closed at sequence {end_at} before any checkpoint"),
            ))
        }
        Ok(_) => {}
        Err(fault) => return Ok(SpoolReadiness::invalid(Some(epoch), fault.to_string())),
    }

    let frame = match tail.read_at(checkpoint_at, FrameKind::Checkpoint) {
        Ok(frame) => frame,
        Err(fault) => return Ok(SpoolReadiness::invalid(Some(epoch), fault.to_string())),
    };
    let StreamEvent::Checkpoint(checkpoint) = frame.event else {
        return Ok(SpoolReadiness::invalid(
            Some(epoch),
            format!(
                "the frame at sequence {checkpoint_at} is named Checkpoint but decoded as \
                     something else"
            ),
        ))
    };
    if let Err(err) =
        checkpoint.validate_declared(partial_stateless_stream::DEFAULT_MAX_SNAPSHOT_BYTES)
    {
        return Ok(SpoolReadiness::invalid(
            Some(epoch),
            format!("checkpoint declaration refused: {err}"),
        ))
    }
    if crate::driver::decode_accepted_head(&checkpoint).is_none() {
        return Ok(SpoolReadiness::invalid(
            Some(epoch),
            "the checkpoint carries no decodable accepted head",
        ))
    }

    // Every declared chunk, present and hashing to the declaration. This is the expensive read,
    // and it is paid exactly once: the polling iterations before the checkpoint exists return
    // above on cheap name scans. A chunk that does not exist yet is the one in-progress shape
    // this loop can see — the producer publishes the announce first and the chunks behind it —
    // so absence is pending and every present-but-wrong chunk is invalid.
    let mut digest = Keccak256::new();
    let mut accumulated = 0u64;
    for index in 0..checkpoint.snapshot_chunks {
        let sequence = checkpoint_at + 1 + u64::from(index);
        let name = format!("{sequence:012}_{}.frame", FrameKind::SnapshotChunk.as_str());
        if !dir.join(&name).exists() {
            return Ok(SpoolReadiness::pending(
                Some(epoch),
                format!("snapshot chunk {index} is not written yet"),
            ))
        }
        let frame = match tail.read_at(sequence, FrameKind::SnapshotChunk) {
            Ok(frame) => frame,
            Err(fault) => {
                return Ok(SpoolReadiness::invalid(
                    Some(epoch),
                    format!("snapshot chunk {index} did not read back: {fault}"),
                ))
            }
        };
        let StreamEvent::SnapshotChunk(chunk) = frame.event else {
            return Ok(SpoolReadiness::invalid(
                Some(epoch),
                format!(
                    "the frame at sequence {sequence} is named SnapshotChunk but decoded as \
                         something else"
                ),
            ))
        };
        if chunk.index != index {
            return Ok(SpoolReadiness::invalid(
                Some(epoch),
                format!("snapshot chunk {} sits where {index} belongs", chunk.index),
            ))
        }
        digest.update(&chunk.bytes);
        accumulated += chunk.bytes.len() as u64;
    }
    let digest = digest.finalize();
    if checkpoint.snapshot_chunks > 0 &&
        (digest != checkpoint.snapshot_digest || accumulated != checkpoint.snapshot_bytes)
    {
        return Ok(SpoolReadiness::invalid(
            Some(epoch),
            format!(
                "the snapshot hashed to {digest:?} over {accumulated} bytes; the checkpoint \
                 declared {:?} over {}",
                checkpoint.snapshot_digest, checkpoint.snapshot_bytes
            ),
        ))
    }

    Ok(SpoolReadiness {
        ready: true,
        state: "ready",
        epoch: Some(epoch),
        checkpoint_sequence: Some(checkpoint_at),
        chunks: Some(checkpoint.snapshot_chunks),
        reason: None,
    })
}
