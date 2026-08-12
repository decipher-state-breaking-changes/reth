//! Following a spool a live producer is still writing, without sharing anything but the
//! directory.
//!
//! The producer's atomic rename is what makes this exact rather than heuristic: a visible
//! `.frame` file is a complete frame, a `.tmp` file is invisible by the extension filter, and a
//! decode failure on a visible file is therefore corruption, never "try again later".
//!
//! The ordinary poll does not list the directory. The producer names files
//! `{sequence:012}_{kind}.frame` and the follower knows exactly which sequence it wants next, so
//! it probes the seven candidate kind names with `stat` — constant work per poll no matter how
//! large the spool has grown. A full listing runs only where it is the point: deciding whether an
//! absent expected frame is a gap, and scanning for a recovery checkpoint.
//!
//! What a poll can not tell apart is a dead producer and a quiet chain: both look like an absent
//! next frame. The follower therefore waits by default — an `End` frame is how a producer says it
//! stopped, and its absence after an abrupt kill is exactly what "cut" means — and any impatience
//! (a harness that killed the producer on purpose) belongs to the caller as a timeout.

use crate::spool::{read_frame_file, SpooledFrame};
use partial_stateless_stream::{FrameKind, FrameLimits};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// Every frame kind a sequence's file could be named under.
const ALL_KINDS: [FrameKind; 7] = [
    FrameKind::Manifest,
    FrameKind::Checkpoint,
    FrameKind::SnapshotChunk,
    FrameKind::Commit,
    FrameKind::Reorg,
    FrameKind::Reset,
    FrameKind::End,
];

/// An incremental reader over a spool another process is writing.
#[derive(Debug)]
pub struct SpoolTail {
    dir: PathBuf,
    limits: FrameLimits,
    next_sequence: u64,
}

/// What one poll produced.
#[derive(Debug)]
pub enum TailEvent {
    /// The next frame, in sequence. Boxed: a frame is two orders of magnitude larger than the
    /// `Idle` it alternates with.
    Frame(Box<SpooledFrame>),
    /// Nothing new. The caller owns the pacing; this call never sleeps.
    Idle,
}

/// A delivery integrity violation. None of these may be handled by skipping.
#[derive(Debug, thiserror::Error)]
pub enum TailFault {
    /// A later sequence exists and the expected one does not, on two consecutive listings.
    #[error("sequence {expected} is missing while {found} exists; the spool has a gap")]
    Gap {
        /// The sequence the follower needed next.
        expected: u64,
        /// The lowest sequence found above it.
        found: u64,
    },
    /// Two files claim the same sequence.
    #[error("sequence {sequence} is claimed by more than one frame file: {names:?}")]
    DuplicateConflict {
        /// The contested sequence.
        sequence: u64,
        /// The file names claiming it.
        names: Vec<String>,
    },
    /// A visible frame file did not decode, or its name disagrees with its own header.
    ///
    /// Visible means complete — the producer renames finished frames into place — so this is
    /// corruption or tampering, not a frame still being written.
    #[error("frame {path} is unusable: {detail}")]
    Undecodable {
        /// The offending file.
        path: PathBuf,
        /// What was wrong with it.
        detail: String,
    },
}

impl SpoolTail {
    /// Follows `dir` from sequence 0. The directory may be empty or not yet interesting; the
    /// first poll simply reports [`TailEvent::Idle`] until the manifest appears.
    pub fn new(dir: &Path, limits: FrameLimits) -> Self {
        Self { dir: dir.to_path_buf(), limits, next_sequence: 0 }
    }

    /// The sequence the next [`TailEvent::Frame`] will carry.
    pub const fn next_sequence(&self) -> u64 {
        self.next_sequence
    }

    /// One poll pass: the next frame, nothing yet, or a fault. Never blocks.
    pub fn poll(&mut self) -> Result<TailEvent, TailFault> {
        let expected = self.next_sequence;
        if let Some(path) = self.probe(expected)? {
            return self.deliver(&path, expected).map(|frame| TailEvent::Frame(Box::new(frame)))
        }
        // The expected frame is absent. If nothing later exists either, the producer just has
        // not written it yet. If something later does exist, that is *almost always* a gap —
        // the producer writes sequences in order on one thread — except for one benign race:
        // this probe's `stat` and the producer's `rename` can interleave within one pass. A
        // second look settles it deterministically, with no wall-clock in the decision.
        let Some(found) = self.lowest_sequence_above(expected)? else { return Ok(TailEvent::Idle) };
        if let Some(path) = self.probe(expected)? {
            return self.deliver(&path, expected).map(|frame| TailEvent::Frame(Box::new(frame)))
        }
        Err(TailFault::Gap { expected, found })
    }

    /// Scans the whole directory for the lowest frame of `kind` at or above `from`, for
    /// `NeedsSnapshot` recovery.
    ///
    /// This is the one place contiguity is deliberately not required: the follower has already
    /// declared it cannot continue, and what it is looking for is a fresh place to start (a
    /// checkpoint) or proof that none is coming (an `End`).
    pub fn scan_for(&self, wanted: FrameKind, from: u64) -> Result<Option<u64>, TailFault> {
        let mut lowest: Option<u64> = None;
        for entry in self.entries()? {
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some((sequence, kind)) = parse_frame_name(name) else { continue };
            if kind == wanted && sequence >= from && lowest.is_none_or(|current| sequence < current)
            {
                lowest = Some(sequence);
            }
        }
        Ok(lowest)
    }

    /// Reads the frame at a scanned-for sequence, by its name.
    pub fn read_at(&self, sequence: u64, kind: FrameKind) -> Result<SpooledFrame, TailFault> {
        let path = self.dir.join(frame_name(sequence, kind));
        read_frame_file(&path, &self.limits)
            .map_err(|err| TailFault::Undecodable { path, detail: format!("{err:#}") })
    }

    /// Counts the commit frames in `[from, to)`, so a recovery can say what it skipped.
    pub fn count_commits_between(&self, from: u64, to: u64) -> Result<u64, TailFault> {
        let mut count = 0;
        for entry in self.entries()? {
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            if let Some((sequence, kind)) = parse_frame_name(name) &&
                kind == FrameKind::Commit &&
                sequence >= from &&
                sequence < to
            {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Abandons contiguity and resumes at `sequence` — the recovery checkpoint's own.
    pub const fn skip_to(&mut self, sequence: u64) {
        self.next_sequence = sequence;
    }

    /// Finds the file for `sequence` by probing the seven kind names, without listing.
    fn probe(&self, sequence: u64) -> Result<Option<PathBuf>, TailFault> {
        let mut hits: Vec<PathBuf> = Vec::new();
        for kind in ALL_KINDS {
            let path = self.dir.join(frame_name(sequence, kind));
            if fs::metadata(&path).is_ok() {
                hits.push(path);
            }
        }
        match hits.len() {
            0 => Ok(None),
            1 => Ok(Some(hits.remove(0))),
            _ => Err(TailFault::DuplicateConflict {
                sequence,
                names: hits
                    .iter()
                    .filter_map(|path| path.file_name())
                    .map(|name| name.to_string_lossy().into_owned())
                    .collect(),
            }),
        }
    }

    /// Reads one complete frame file and holds it to what its name promised.
    fn deliver(&mut self, path: &Path, expected: u64) -> Result<SpooledFrame, TailFault> {
        let frame = read_frame_file(path, &self.limits).map_err(|err| TailFault::Undecodable {
            path: path.to_path_buf(),
            detail: format!("{err:#}"),
        })?;
        // The name is how the frame was found; the header is the authority. A file whose two
        // identities disagree was renamed or rewritten, and either way it cannot be trusted.
        let named =
            path.file_name().and_then(|name| name.to_str()).and_then(parse_frame_name).ok_or_else(
                || TailFault::Undecodable {
                    path: path.to_path_buf(),
                    detail: "the file name does not parse as a frame name".to_string(),
                },
            )?;
        if frame.header.sequence != expected || named != (expected, frame.header.kind) {
            return Err(TailFault::Undecodable {
                path: path.to_path_buf(),
                detail: format!(
                    "the file name claims sequence {} kind {}, the header says sequence {} kind \
                     {}",
                    named.0,
                    named.1.as_str(),
                    frame.header.sequence,
                    frame.header.kind.as_str()
                ),
            })
        }
        self.next_sequence = expected + 1;
        Ok(frame)
    }

    /// The lowest frame sequence strictly above `from`, from a full listing.
    fn lowest_sequence_above(&self, from: u64) -> Result<Option<u64>, TailFault> {
        let mut lowest: Option<u64> = None;
        for entry in self.entries()? {
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some((sequence, _)) = parse_frame_name(name) else { continue };
            if sequence > from && lowest.is_none_or(|current| sequence < current) {
                lowest = Some(sequence);
            }
        }
        Ok(lowest)
    }

    fn entries(&self) -> Result<Vec<fs::DirEntry>, TailFault> {
        fs::read_dir(&self.dir)
            .map_err(|err| TailFault::Undecodable {
                path: self.dir.clone(),
                detail: format!("cannot list the spool: {err}"),
            })
            .map(|entries| entries.filter_map(Result::ok).collect())
    }
}

/// The producer's file name for a frame.
fn frame_name(sequence: u64, kind: FrameKind) -> String {
    format!("{sequence:012}_{}.frame", kind.as_str())
}

/// Parses `{sequence:012}_{kind}.frame` back apart. `None` for anything else in the directory.
fn parse_frame_name(name: &str) -> Option<(u64, FrameKind)> {
    let stem = name.strip_suffix(".frame")?;
    let (sequence, kind) = stem.split_once('_')?;
    if sequence.len() != 12 {
        return None
    }
    let sequence = sequence.parse().ok()?;
    let kind = ALL_KINDS.into_iter().find(|candidate| candidate.as_str() == kind)?;
    Some((sequence, kind))
}

#[cfg(test)]
mod tests {
    use super::*;
    use partial_stateless_stream::{encode_event, End, EndKind, Reset, ResetReason, StreamEvent};

    fn spool_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ps-tail-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    fn filler() -> StreamEvent {
        StreamEvent::Reset(Reset { reason: ResetReason::Gap, detail: "test".into() })
    }

    fn write(dir: &Path, sequence: u64, event: &StreamEvent, kind: FrameKind) {
        let bytes = encode_event(sequence, event, &FrameLimits::default()).expect("encodes");
        fs::write(dir.join(frame_name(sequence, kind)), bytes).expect("write");
    }

    #[test]
    fn frames_appearing_between_polls_are_consumed_in_order() {
        let dir = spool_dir("ordered");
        let mut tail = SpoolTail::new(&dir, FrameLimits::default());
        assert!(matches!(tail.poll(), Ok(TailEvent::Idle)), "an empty spool is not an error");

        write(&dir, 0, &filler(), FrameKind::Reset);
        let Ok(TailEvent::Frame(frame)) = tail.poll() else { panic!("frame 0 delivers") };
        assert_eq!(frame.header.sequence, 0);
        assert!(matches!(tail.poll(), Ok(TailEvent::Idle)), "nothing more yet");

        write(&dir, 1, &filler(), FrameKind::Reset);
        let Ok(TailEvent::Frame(frame)) = tail.poll() else { panic!("frame 1 delivers") };
        assert_eq!(frame.header.sequence, 1);
        let _ = fs::remove_dir_all(&dir);
    }

    /// A `.tmp` file is a frame still being written; the extension filter is the atomicity
    /// contract with the producer.
    #[test]
    fn a_tmp_file_is_invisible() {
        let dir = spool_dir("tmp");
        fs::write(dir.join("000000000000_reset.tmp"), b"partial").expect("write");
        let mut tail = SpoolTail::new(&dir, FrameLimits::default());
        assert!(matches!(tail.poll(), Ok(TailEvent::Idle)));
        let _ = fs::remove_dir_all(&dir);
    }

    /// The failure the double-scan exists to make deterministic: a later frame visible while an
    /// earlier one is genuinely absent is a gap, not a race, on the second look.
    #[test]
    fn a_missing_sequence_below_an_existing_one_is_a_gap() {
        let dir = spool_dir("gap");
        write(&dir, 0, &filler(), FrameKind::Reset);
        write(&dir, 2, &filler(), FrameKind::Reset);
        let mut tail = SpoolTail::new(&dir, FrameLimits::default());
        assert!(matches!(tail.poll(), Ok(TailEvent::Frame(_))));

        let fault = tail.poll().expect_err("the hole is a fault");
        assert!(matches!(fault, TailFault::Gap { expected: 1, found: 2 }), "{fault}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn two_files_claiming_one_sequence_are_a_duplicate_conflict() {
        let dir = spool_dir("dup");
        write(&dir, 0, &filler(), FrameKind::Reset);
        write(
            &dir,
            0,
            &StreamEvent::End(End {
                kind: EndKind::Shutdown,
                reason: "imposter".into(),
                last_sequence: 0,
            }),
            FrameKind::End,
        );
        let mut tail = SpoolTail::new(&dir, FrameLimits::default());
        let fault = tail.poll().expect_err("two claims on one sequence");
        assert!(matches!(fault, TailFault::DuplicateConflict { sequence: 0, .. }), "{fault}");
        let _ = fs::remove_dir_all(&dir);
    }

    /// A visible frame is complete, so failing to decode it is corruption rather than patience.
    #[test]
    fn a_corrupt_visible_frame_is_undecodable_not_retried() {
        let dir = spool_dir("corrupt");
        fs::write(dir.join(frame_name(0, FrameKind::Reset)), b"not a frame").expect("write");
        let mut tail = SpoolTail::new(&dir, FrameLimits::default());
        let fault = tail.poll().expect_err("corruption is a fault");
        assert!(matches!(fault, TailFault::Undecodable { .. }), "{fault}");
        let _ = fs::remove_dir_all(&dir);
    }

    /// The name found the frame; the header is the authority. Disagreement is tampering.
    #[test]
    fn a_renamed_frame_is_refused() {
        let dir = spool_dir("renamed");
        let bytes = encode_event(5, &filler(), &FrameLimits::default()).expect("encodes");
        fs::write(dir.join(frame_name(0, FrameKind::Reset)), bytes).expect("write");
        let mut tail = SpoolTail::new(&dir, FrameLimits::default());
        let fault = tail.poll().expect_err("a renamed frame is refused");
        assert!(matches!(fault, TailFault::Undecodable { .. }), "{fault}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn recovery_scans_find_the_lowest_checkpoint_at_or_above_the_watermark() {
        let dir = spool_dir("scan");
        write(&dir, 0, &filler(), FrameKind::Reset);
        // Fabricated checkpoints for the name scan only; the follower verifies content later.
        write(&dir, 3, &filler(), FrameKind::Reset);
        let bytes = encode_event(7, &filler(), &FrameLimits::default()).expect("encodes");
        fs::write(dir.join(frame_name(7, FrameKind::Checkpoint)), &bytes).expect("write");
        let tail = SpoolTail::new(&dir, FrameLimits::default());

        assert_eq!(tail.scan_for(FrameKind::Checkpoint, 0).expect("scans"), Some(7));
        assert_eq!(tail.scan_for(FrameKind::Checkpoint, 8).expect("scans"), None);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn skipped_commits_are_countable_for_the_recovery_record() {
        let dir = spool_dir("count");
        let bytes = encode_event(2, &filler(), &FrameLimits::default()).expect("encodes");
        fs::write(dir.join(frame_name(2, FrameKind::Commit)), &bytes).expect("write");
        fs::write(dir.join(frame_name(4, FrameKind::Commit)), &bytes).expect("write");
        let tail = SpoolTail::new(&dir, FrameLimits::default());

        assert_eq!(tail.count_commits_between(0, 5).expect("counts"), 2);
        assert_eq!(tail.count_commits_between(3, 5).expect("counts"), 1);
        let _ = fs::remove_dir_all(&dir);
    }
}
