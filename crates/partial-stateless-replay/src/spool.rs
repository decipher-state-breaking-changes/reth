//! Reading a recorded stream back out of a directory, in the order it was written.
//!
//! Ordering comes from the sequence number the producer stamped into each frame, not from the
//! directory listing and not from the filename. The filename carries the sequence so an operator
//! can read the spool, but a reader that trusted it would be trusting a rename.
//!
//! Contiguity is checked rather than assumed. A recorded corpus with a hole in it looks exactly
//! like a complete one to anything that just decodes what it finds, and a replay over such a
//! corpus would report agreement about a chain nobody ran.

use partial_stateless_stream::{
    decode_event, FrameHeader, FrameLimits, StreamEvent, FRAME_HEADER_BYTES,
};
use std::{fs, path::Path};

/// One frame as read back.
#[derive(Debug)]
pub struct SpooledFrame {
    /// The frame's own header, which is where the sequence lives.
    pub header: FrameHeader,
    /// The decoded body.
    pub event: StreamEvent,
    /// Bytes the frame file held, for the reader's own accounting.
    pub bytes: u64,
}

/// Reads and decodes exactly one frame file.
///
/// The size check runs on the file's metadata before the read, because the decoder's own bound
/// only applies after `fs::read` has already paid for the allocation — a spool entry past every
/// frame bound is refused without buffering it.
pub fn read_frame_file(path: &Path, limits: &FrameLimits) -> eyre::Result<SpooledFrame> {
    let metadata = fs::metadata(path)
        .map_err(|err| eyre::eyre!("cannot stat frame {}: {err}", path.display()))?;
    let bound = (FRAME_HEADER_BYTES + limits.max_frame_bytes) as u64;
    if metadata.len() > bound {
        eyre::bail!(
            "frame {} is {} bytes, past the {bound}-byte bound; refused before reading it",
            path.display(),
            metadata.len()
        );
    }
    let raw =
        fs::read(path).map_err(|err| eyre::eyre!("cannot read frame {}: {err}", path.display()))?;
    let (header, event, rest) = decode_event(&raw, limits)
        .map_err(|err| eyre::eyre!("frame {} is unusable: {err}", path.display()))?;
    if !rest.is_empty() {
        eyre::bail!(
            "frame {} carries {} trailing bytes; one file holds exactly one frame",
            path.display(),
            rest.len()
        );
    }
    Ok(SpooledFrame { header, event, bytes: raw.len() as u64 })
}

/// Iterates a complete spool in sequence order, holding one frame in memory at a time.
///
/// This is what lets a deterministic replay of a long corpus run in constant frame memory: a
/// 6,000-block corpus is roughly 20 GiB of commits, and the old whole-directory read held all of
/// it. The directory listing orders the walk, but the *authority* stays with each frame's own
/// header: a file whose header sequence is not the position the walk expects is refused, so a
/// rename cannot reorder the corpus — it can only break it loudly.
#[derive(Debug)]
pub struct SpoolIter {
    paths: Vec<std::path::PathBuf>,
    position: usize,
    limits: FrameLimits,
    /// Set when an `End` frame was yielded; any frame after it is a refusal.
    ended: bool,
    closed: bool,
    bytes: u64,
}

impl SpoolIter {
    /// Opens a spool for iteration. Enumerates the directory; reads nothing yet.
    pub fn open(dir: &Path, limits: &FrameLimits) -> eyre::Result<Self> {
        let mut paths: Vec<_> = fs::read_dir(dir)
            .map_err(|err| eyre::eyre!("cannot read spool {}: {err}", dir.display()))?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|extension| extension == "frame"))
            .collect();
        if paths.is_empty() {
            eyre::bail!("spool {} holds no frames", dir.display());
        }
        paths.sort();
        Ok(Self { paths, position: 0, limits: *limits, ended: false, closed: false, bytes: 0 })
    }

    /// Reads the next frame, enforcing contiguity and the End-frame placement rules.
    pub fn next_frame(&mut self) -> eyre::Result<Option<SpooledFrame>> {
        let Some(path) = self.paths.get(self.position) else { return Ok(None) };
        if self.ended {
            eyre::bail!("spool continues past its End frame: {}", path.display());
        }
        let frame = read_frame_file(path, &self.limits)?;
        if frame.header.sequence != self.position as u64 {
            eyre::bail!(
                "spool is not contiguous: sequence {} arrived where {} was expected. A corpus \
                 with a hole reads as a corpus, so this is refused rather than replayed",
                frame.header.sequence,
                self.position
            );
        }
        if let StreamEvent::End(end) = &frame.event {
            if end.last_sequence.checked_add(1) != Some(frame.header.sequence) {
                eyre::bail!(
                    "the End frame at sequence {} names {} as the last frame; its predecessor \
                     was {}",
                    frame.header.sequence,
                    end.last_sequence,
                    frame.header.sequence.saturating_sub(1)
                );
            }
            self.ended = true;
            self.closed = true;
        }
        self.position += 1;
        self.bytes = self.bytes.saturating_add(frame.bytes);
        Ok(Some(frame))
    }

    /// Whether the stream ended with an `End` frame rather than being cut.
    ///
    /// Meaningful once iteration is done; before that it reports what has been seen so far.
    pub const fn closed(&self) -> bool {
        self.closed
    }

    /// Total bytes read so far.
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }
}

/// A whole spool, read and checked for contiguity.
#[derive(Debug)]
pub struct Spool {
    /// Frames in sequence order.
    pub frames: Vec<SpooledFrame>,
    /// Whether the stream ended with an `End` frame rather than being cut.
    pub closed: bool,
    /// Total bytes read.
    pub bytes: u64,
}

/// Reads every `.frame` file in `dir` and returns them in sequence order.
///
/// Fails rather than skips on anything malformed. A driver whose corpus is its evidence cannot
/// treat an unreadable frame as an absent one. This materializes the whole spool; a caller with a
/// large corpus should walk [`SpoolIter`] instead.
pub fn read_spool(dir: &Path, limits: &FrameLimits) -> eyre::Result<Spool> {
    let mut iter = SpoolIter::open(dir, limits)?;
    let mut frames = Vec::new();
    while let Some(frame) = iter.next_frame()? {
        frames.push(frame);
    }
    Ok(Spool { frames, closed: iter.closed(), bytes: iter.bytes() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use partial_stateless_stream::{encode_event, End, EndKind, FrameKind, Reset, ResetReason};

    fn spool_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("ps-spool-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    fn write(dir: &Path, sequence: u64, event: &StreamEvent, kind: FrameKind) {
        let bytes = encode_event(sequence, event, &FrameLimits::default()).expect("encodes");
        fs::write(dir.join(format!("{sequence:012}_{}.frame", kind.as_str())), bytes)
            .expect("write");
    }

    fn filler() -> StreamEvent {
        StreamEvent::Reset(Reset { reason: ResetReason::Overflow, detail: "test".into() })
    }

    fn end(last: u64) -> StreamEvent {
        StreamEvent::End(End {
            kind: EndKind::Shutdown,
            reason: "test".into(),
            last_sequence: last,
        })
    }

    #[test]
    fn a_spool_reads_back_in_sequence_order_and_reports_that_it_closed() {
        let dir = spool_dir("ordered");
        write(&dir, 0, &filler(), FrameKind::Reset);
        write(&dir, 1, &end(0), FrameKind::End);

        let spool = read_spool(&dir, &FrameLimits::default()).expect("reads");
        assert_eq!(
            spool.frames.iter().map(|frame| frame.header.sequence).collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert!(spool.closed);
        let _ = fs::remove_dir_all(&dir);
    }

    /// The failure this exists to prevent: a gap that a decoder cannot see, replayed as if it were
    /// a contiguous chain.
    #[test]
    fn a_gap_in_the_sequence_is_refused_rather_than_replayed() {
        let dir = spool_dir("gap");
        write(&dir, 0, &filler(), FrameKind::Reset);
        write(&dir, 2, &filler(), FrameKind::Reset);

        let error = read_spool(&dir, &FrameLimits::default()).expect_err("gap is refused");
        assert!(error.to_string().contains("not contiguous"), "{error}");
        let _ = fs::remove_dir_all(&dir);
    }

    /// A corpus cut mid-run is usable — it is just shorter — but a driver has to know that it is
    /// reading a prefix rather than a whole stream.
    #[test]
    fn a_stream_without_an_end_frame_reports_that_it_was_cut() {
        let dir = spool_dir("cut");
        write(&dir, 0, &filler(), FrameKind::Reset);

        let spool = read_spool(&dir, &FrameLimits::default()).expect("reads");
        assert!(!spool.closed);
        let _ = fs::remove_dir_all(&dir);
    }

    /// An End frame closes the stream; frames after it mean the closer lied about where the
    /// stream stopped, and the corpus cannot be trusted about anything else.
    #[test]
    fn a_spool_that_continues_past_its_end_frame_is_refused() {
        let dir = spool_dir("past-end");
        write(&dir, 0, &filler(), FrameKind::Reset);
        write(&dir, 1, &end(0), FrameKind::End);
        write(&dir, 2, &filler(), FrameKind::Reset);

        let error = read_spool(&dir, &FrameLimits::default()).expect_err("trailing refused");
        assert!(error.to_string().contains("continues past its End frame"), "{error}");
        let _ = fs::remove_dir_all(&dir);
    }

    /// `last_sequence` names the frame before the End, and the reader holds it to that. The
    /// off-by-one this catches — an End carrying its *own* sequence — was the recorder's actual
    /// defect before S3.
    #[test]
    fn an_end_frame_that_misnames_its_predecessor_is_refused() {
        let dir = spool_dir("end-off-by-one");
        write(&dir, 0, &filler(), FrameKind::Reset);
        write(&dir, 1, &end(1), FrameKind::End);

        let error = read_spool(&dir, &FrameLimits::default()).expect_err("misnumbered End");
        assert!(error.to_string().contains("names 1 as the last frame"), "{error}");
        let _ = fs::remove_dir_all(&dir);
    }

    /// The size check runs on metadata, so a file past every frame bound is refused without the
    /// read that would have buffered it.
    #[test]
    fn an_oversized_frame_file_is_refused_before_it_is_read() {
        let dir = spool_dir("oversized");
        write(&dir, 0, &filler(), FrameKind::Reset);
        let limits = FrameLimits { max_frame_bytes: 8 };

        let path = dir.join("000000000000_reset.frame");
        let error = read_frame_file(&path, &limits).expect_err("bound applies");
        assert!(error.to_string().contains("refused before reading"), "{error}");
        let _ = fs::remove_dir_all(&dir);
    }
}
