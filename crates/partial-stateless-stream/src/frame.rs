//! The envelope every recorded, replayed, and delivered event travels in.
//!
//! One format for all three, which is the point rather than an economy. A file spool that a replay
//! driver reads and a socket that a live consumer reads carry the *same bytes*; if they did not,
//! the deterministic replay would be validating a format the live path never uses, and the live
//! path's failures would be unreproducible by construction.
//!
//! The header is fixed width and self-describing, so a reader can reject a stream before it
//! allocates anything for it. Every field in it exists to make one specific silent failure loud:
//!
//! - **Magic** separates "this is not our file" from "this is our file and it is broken".
//! - **Format version** is checked against an exact value rather than a floor. A newer producer's
//!   frames are refused rather than parsed by an older reader that would read the fields it knows
//!   and silently ignore what it does not.
//! - **Flags** are reserved and must be zero in v1, for the same reason: a v1 reader that ignored
//!   an unknown flag would be honouring a semantic it has never heard of.
//! - **Sequence** is the producer's own monotonic counter. A gap is a fact about delivery that a
//!   consumer must act on, and a stream without it can only notice loss when the content happens to
//!   disagree.
//! - **Length** is bounded before the body is read, so a corrupt length cannot become an
//!   allocation.
//! - **Digest** is keccak256 of the body. Truncation is already caught by the length; the digest
//!   catches the case where the bytes arrived and are wrong.

use alloy_primitives::{keccak256, B256};
use serde::{de::DeserializeOwned, Serialize};

/// Leading bytes of every frame.
pub const FRAME_MAGIC: [u8; 8] = *b"PSSTREAM";

/// The only format version this crate reads or writes.
///
/// Deliberately a constant rather than a range. Every event kind S3 and S4 need is already in v1,
/// so the first reason to bump this would be a change of meaning, and a reader that accepted a
/// range would be guessing at what that change was.
pub const FORMAT_VERSION: u16 = 1;

/// Fixed header width: magic, version, kind, flags, sequence, length, digest.
pub const FRAME_HEADER_BYTES: usize = 8 + 2 + 1 + 1 + 8 + 4 + 32;

/// Default ceiling on one frame's body.
///
/// Measured rather than chosen. The S2-0 run's sidecars ran to 5.57 MiB at their widest with a
/// 2.86 MiB median, and a commit frame carries one of those beside an Engine payload, so 64 MiB
/// leaves better than an order of magnitude of headroom over the widest block mainnet produced.
/// The snapshot package does *not* fit and is not meant to: at 121.8 MiB it is a chunk sequence,
/// and [`DEFAULT_SNAPSHOT_CHUNK_BYTES`] is what each of its frames is bounded by.
pub const DEFAULT_MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

/// Default size of one snapshot chunk body.
///
/// Puts the S2-0 package at sixteen frames. Small enough that a consumer's buffer is bounded well
/// below the package, large enough that the per-frame header is noise.
pub const DEFAULT_SNAPSHOT_CHUNK_BYTES: usize = 8 * 1024 * 1024;

/// What a frame carries, as a stable one-byte tag.
///
/// All of v1's kinds are here even though the first executable replay accepts only
/// [`Commit`](Self::Commit). Adding [`Reorg`](Self::Reorg) or [`Reset`](Self::Reset) later would
/// mean a second spool format for S3 and S4 to migrate across, and the migration would land
/// exactly when the lifecycle work is hardest to reason about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum FrameKind {
    /// Stream identity and policy descriptor. Exactly one, first.
    Manifest = 1,
    /// The operator-trusted checkpoint, and the header of the snapshot that follows it.
    Checkpoint = 2,
    /// One slice of the snapshot package the preceding checkpoint described.
    ///
    /// Not a seventh event: a checkpoint and its chunks are one "checkpoint+snapshot" event that
    /// does not fit in one frame. Splitting the *bytes* rather than the *event* is what keeps the
    /// per-frame bound meaningful.
    SnapshotChunk = 3,
    /// One canonical block: the payload, its sidecar, and the producer's own outcome.
    Commit = 4,
    /// A branch was abandoned. The winning branch arrives as [`Commit`](Self::Commit) frames.
    Reorg = 5,
    /// The consumer cannot continue from what it has and must be re-bootstrapped.
    Reset = 6,
    /// The producer has stopped. A stream without one of these ended unexpectedly.
    End = 7,
}

impl FrameKind {
    /// Stable name for logs and records.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Manifest => "manifest",
            Self::Checkpoint => "checkpoint",
            Self::SnapshotChunk => "snapshot_chunk",
            Self::Commit => "commit",
            Self::Reorg => "reorg",
            Self::Reset => "reset",
            Self::End => "end",
        }
    }

    const fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            1 => Some(Self::Manifest),
            2 => Some(Self::Checkpoint),
            3 => Some(Self::SnapshotChunk),
            4 => Some(Self::Commit),
            5 => Some(Self::Reorg),
            6 => Some(Self::Reset),
            7 => Some(Self::End),
            _ => None,
        }
    }
}

/// A decoded frame header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    /// Format version the producer wrote.
    pub format_version: u16,
    /// What the body is.
    pub kind: FrameKind,
    /// The producer's monotonic counter, so a consumer can see a gap rather than infer one.
    pub sequence: u64,
    /// Body length in bytes.
    pub payload_len: u32,
    /// keccak256 of the body.
    pub payload_digest: B256,
}

/// Bounds a decoder applies before it allocates.
#[derive(Debug, Clone, Copy)]
pub struct FrameLimits {
    /// Largest body this consumer will read.
    pub max_frame_bytes: usize,
}

impl Default for FrameLimits {
    fn default() -> Self {
        Self { max_frame_bytes: DEFAULT_MAX_FRAME_BYTES }
    }
}

/// Everything that can be wrong with a frame, named rather than collapsed.
///
/// The distinctions are not cosmetic. A truncated frame means the producer is still writing or
/// died; a digest mismatch means the bytes arrived and are wrong; an unknown kind means the
/// producer is newer. Those demand different responses from a consumer, and a single "malformed"
/// would leave it guessing.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FrameError {
    /// Fewer bytes than the frame claims. The ordinary state of a spool file being written.
    #[error("frame is truncated: expected {expected} bytes, found {actual}")]
    Truncated {
        /// Bytes the header says are there.
        expected: usize,
        /// Bytes actually available.
        actual: usize,
    },
    /// Not a frame at all.
    #[error("bad magic {found:?}; this is not a partial-stateless stream")]
    BadMagic {
        /// The leading bytes that were there instead.
        found: [u8; 8],
    },
    /// A version this reader has no definition for.
    #[error("unsupported format version {found}; this reader implements {FORMAT_VERSION}")]
    UnsupportedVersion {
        /// Version the producer wrote.
        found: u16,
    },
    /// A kind tag this reader has no definition for.
    #[error("unknown frame kind {tag}")]
    UnknownKind {
        /// The tag byte.
        tag: u8,
    },
    /// Reserved header bits were set.
    #[error("reserved frame flags {flags:#04x} are set; a v1 reader cannot honour them")]
    ReservedFlags {
        /// The flag byte.
        flags: u8,
    },
    /// The declared body exceeds this consumer's bound.
    #[error("frame body of {len} bytes exceeds the {limit}-byte bound")]
    TooLarge {
        /// Declared body length.
        len: usize,
        /// The configured bound.
        limit: usize,
    },
    /// The body arrived intact in length and wrong in content.
    #[error("frame body digest {actual} does not match the declared {expected}")]
    ChecksumMismatch {
        /// Digest the header declared.
        expected: B256,
        /// Digest of the bytes that arrived.
        actual: B256,
    },
    /// The body did not decode into the event its kind names.
    #[error("frame body of kind {kind} did not decode: {detail}")]
    Body {
        /// The kind that was declared.
        kind: &'static str,
        /// The codec's own message.
        detail: String,
    },
    /// A caller asked for one kind and the frame is another.
    #[error("expected a {expected} frame, found {actual}")]
    KindMismatch {
        /// What the caller asked for.
        expected: &'static str,
        /// What the frame is.
        actual: &'static str,
    },
}

/// Encodes one body into a complete frame.
///
/// The digest is over the encoded body and nothing else, so a frame's bytes can be verified
/// without knowing what kind it is.
pub fn encode_frame<T: Serialize>(
    kind: FrameKind,
    sequence: u64,
    body: &T,
) -> Result<Vec<u8>, FrameError> {
    let payload = bincode::serialize(body)
        .map_err(|err| FrameError::Body { kind: kind.as_str(), detail: err.to_string() })?;
    Ok(encode_frame_bytes(kind, sequence, &payload))
}

/// Encodes an already-serialized body into a complete frame.
pub fn encode_frame_bytes(kind: FrameKind, sequence: u64, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(FRAME_HEADER_BYTES + payload.len());
    out.extend_from_slice(&FRAME_MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.push(kind as u8);
    out.push(0); // reserved flags
    out.extend_from_slice(&sequence.to_le_bytes());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(keccak256(payload).as_slice());
    out.extend_from_slice(payload);
    out
}

/// Reads a header without touching the body.
///
/// Separate from [`decode_frame`] because a consumer that is streaming needs to know how many
/// bytes to wait for before it has them, and because bounding the length is only useful if it
/// happens before the body is buffered.
pub fn decode_header(bytes: &[u8], limits: &FrameLimits) -> Result<FrameHeader, FrameError> {
    if bytes.len() < FRAME_HEADER_BYTES {
        return Err(FrameError::Truncated { expected: FRAME_HEADER_BYTES, actual: bytes.len() })
    }
    let magic: [u8; 8] = bytes[0..8].try_into().expect("checked length");
    if magic != FRAME_MAGIC {
        return Err(FrameError::BadMagic { found: magic })
    }
    let format_version = u16::from_le_bytes(bytes[8..10].try_into().expect("checked length"));
    if format_version != FORMAT_VERSION {
        return Err(FrameError::UnsupportedVersion { found: format_version })
    }
    let kind = FrameKind::from_tag(bytes[10]).ok_or(FrameError::UnknownKind { tag: bytes[10] })?;
    if bytes[11] != 0 {
        return Err(FrameError::ReservedFlags { flags: bytes[11] })
    }
    let sequence = u64::from_le_bytes(bytes[12..20].try_into().expect("checked length"));
    let payload_len = u32::from_le_bytes(bytes[20..24].try_into().expect("checked length"));
    if payload_len as usize > limits.max_frame_bytes {
        return Err(FrameError::TooLarge {
            len: payload_len as usize,
            limit: limits.max_frame_bytes,
        })
    }
    let payload_digest = B256::from_slice(&bytes[24..56]);
    Ok(FrameHeader { format_version, kind, sequence, payload_len, payload_digest })
}

/// Splits one frame off the front of `bytes`, returning its header, body, and the rest.
///
/// The digest is checked here, so every caller downstream is working with bytes that arrived
/// intact and no caller has to remember to check.
pub fn decode_frame<'a>(
    bytes: &'a [u8],
    limits: &FrameLimits,
) -> Result<(FrameHeader, &'a [u8], &'a [u8]), FrameError> {
    let header = decode_header(bytes, limits)?;
    let end = FRAME_HEADER_BYTES + header.payload_len as usize;
    if bytes.len() < end {
        return Err(FrameError::Truncated { expected: end, actual: bytes.len() })
    }
    let payload = &bytes[FRAME_HEADER_BYTES..end];
    let actual = keccak256(payload);
    if actual != header.payload_digest {
        return Err(FrameError::ChecksumMismatch { expected: header.payload_digest, actual })
    }
    Ok((header, payload, &bytes[end..]))
}

/// Decodes a body that was declared to be of `kind`.
pub fn decode_body<T: DeserializeOwned>(kind: FrameKind, payload: &[u8]) -> Result<T, FrameError> {
    bincode::deserialize(payload)
        .map_err(|err| FrameError::Body { kind: kind.as_str(), detail: err.to_string() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    struct Body {
        number: u64,
        note: String,
    }

    fn body() -> Body {
        Body { number: 25_737_234, note: "checkpoint".to_string() }
    }

    #[test]
    fn a_frame_round_trips_through_its_own_bytes() {
        let encoded = encode_frame(FrameKind::Commit, 7, &body()).expect("encodes");
        let (header, payload, rest) =
            decode_frame(&encoded, &FrameLimits::default()).expect("decodes");

        assert_eq!(header.format_version, FORMAT_VERSION);
        assert_eq!(header.kind, FrameKind::Commit);
        assert_eq!(header.sequence, 7);
        assert_eq!(header.payload_len as usize, payload.len());
        assert!(rest.is_empty());
        assert_eq!(decode_body::<Body>(header.kind, payload).expect("body decodes"), body());
    }

    /// Frames concatenate, because a spool file and a socket are the same bytes back to back.
    #[test]
    fn frames_decode_back_to_back_from_one_buffer() {
        let mut buffer = encode_frame(FrameKind::Manifest, 1, &body()).expect("encodes");
        buffer.extend(encode_frame(FrameKind::Commit, 2, &body()).expect("encodes"));

        let (first, _, rest) = decode_frame(&buffer, &FrameLimits::default()).expect("decodes");
        let (second, _, tail) = decode_frame(rest, &FrameLimits::default()).expect("decodes");

        assert_eq!((first.kind, first.sequence), (FrameKind::Manifest, 1));
        assert_eq!((second.kind, second.sequence), (FrameKind::Commit, 2));
        assert!(tail.is_empty());
    }

    /// The ordinary state of a spool file the producer is still writing. It must read as "not yet",
    /// with the number of bytes still wanted, and never as a decode failure.
    #[test]
    fn a_truncated_frame_says_how_much_is_missing() {
        let encoded = encode_frame(FrameKind::Commit, 1, &body()).expect("encodes");
        let cut = encoded.len() - 4;
        assert_eq!(
            decode_frame(&encoded[..cut], &FrameLimits::default()),
            Err(FrameError::Truncated { expected: encoded.len(), actual: cut })
        );
    }

    /// A header that is itself incomplete is still a truncation and not a bad-magic error, because
    /// a reader that has one byte cannot yet know whether the magic is wrong.
    #[test]
    fn a_partial_header_is_a_truncation_rather_than_a_bad_stream() {
        let encoded = encode_frame(FrameKind::End, 1, &body()).expect("encodes");
        assert_eq!(
            decode_frame(&encoded[..3], &FrameLimits::default()),
            Err(FrameError::Truncated { expected: FRAME_HEADER_BYTES, actual: 3 })
        );
    }

    #[test]
    fn a_stream_that_is_not_ours_is_rejected_on_its_first_bytes() {
        let mut encoded = encode_frame(FrameKind::Commit, 1, &body()).expect("encodes");
        encoded[..8].copy_from_slice(b"NOTASTRM");
        assert_eq!(
            decode_frame(&encoded, &FrameLimits::default()),
            Err(FrameError::BadMagic { found: *b"NOTASTRM" })
        );
    }

    /// A body that arrived complete and wrong. Length alone cannot see this.
    #[test]
    fn a_corrupt_body_fails_its_digest_rather_than_decoding() {
        let mut encoded = encode_frame(FrameKind::Commit, 1, &body()).expect("encodes");
        let last = encoded.len() - 1;
        encoded[last] ^= 0xff;
        let error = decode_frame(&encoded, &FrameLimits::default()).expect_err("digest fails");
        assert!(matches!(error, FrameError::ChecksumMismatch { .. }), "{error:?}");
    }

    /// A newer producer is refused rather than partly understood.
    #[test]
    fn a_future_format_version_is_refused_and_not_parsed() {
        let mut encoded = encode_frame(FrameKind::Commit, 1, &body()).expect("encodes");
        encoded[8..10].copy_from_slice(&(FORMAT_VERSION + 1).to_le_bytes());
        assert_eq!(
            decode_frame(&encoded, &FrameLimits::default()),
            Err(FrameError::UnsupportedVersion { found: FORMAT_VERSION + 1 })
        );
    }

    #[test]
    fn an_unknown_kind_is_named_rather_than_skipped() {
        let mut encoded = encode_frame(FrameKind::Commit, 1, &body()).expect("encodes");
        encoded[10] = 200;
        assert_eq!(
            decode_frame(&encoded, &FrameLimits::default()),
            Err(FrameError::UnknownKind { tag: 200 })
        );
    }

    /// Reserved bits mean a semantic this reader has never heard of. Ignoring them would be
    /// honouring it by omission.
    #[test]
    fn reserved_flags_are_refused() {
        let mut encoded = encode_frame(FrameKind::Commit, 1, &body()).expect("encodes");
        encoded[11] = 0b0000_0001;
        assert_eq!(
            decode_frame(&encoded, &FrameLimits::default()),
            Err(FrameError::ReservedFlags { flags: 1 })
        );
    }

    /// The bound is checked against the declared length, before the body is looked at, so a
    /// corrupt length cannot become an allocation.
    #[test]
    fn an_oversized_frame_is_refused_before_its_body_is_read() {
        let encoded = encode_frame(FrameKind::Commit, 1, &body()).expect("encodes");
        let limits = FrameLimits { max_frame_bytes: 4 };
        let error = decode_frame(&encoded, &limits).expect_err("bound applies");
        assert!(matches!(error, FrameError::TooLarge { limit: 4, .. }), "{error:?}");

        // And the check is on the header alone: a body that never arrived is still refused.
        assert_eq!(decode_header(&encoded[..FRAME_HEADER_BYTES], &limits), Err(error));
    }

    #[test]
    fn a_body_that_is_not_what_its_kind_claims_fails_to_decode() {
        let encoded = encode_frame(FrameKind::Commit, 1, &body()).expect("encodes");
        let (header, payload, _) =
            decode_frame(&encoded, &FrameLimits::default()).expect("decodes");
        let error = decode_body::<[u8; 32]>(header.kind, payload).expect_err("wrong shape");
        assert!(matches!(error, FrameError::Body { kind: "commit", .. }), "{error:?}");
    }

    /// The header is fixed width; a change to it is a format change, and this pins the constant
    /// against the layout the encoder actually writes.
    #[test]
    fn the_header_is_the_width_the_constant_claims() {
        let encoded = encode_frame_bytes(FrameKind::End, 0, &[]);
        assert_eq!(encoded.len(), FRAME_HEADER_BYTES);
    }
}
