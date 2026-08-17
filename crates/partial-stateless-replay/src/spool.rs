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
    decode_event, FrameHeader, FrameKind, FrameLimits, StreamEvent, FRAME_HEADER_BYTES,
};
use std::{fs, path::Path, time::Instant};

/// One frame as read back.
#[derive(Debug)]
pub struct SpooledFrame {
    /// The frame's own header, which is where the sequence lives.
    pub header: FrameHeader,
    /// The decoded body.
    pub event: StreamEvent,
    /// Bytes the frame file held, for the reader's own accounting.
    pub bytes: u64,
    /// Statting and reading the file into memory. Transport cost, outside the validation
    /// boundary: `standalone_validation_us` opens where these bytes already exist.
    pub delivery_us: u64,
    /// Decoding the envelope and body once the bytes were in memory. The first phase inside the
    /// validation boundary.
    pub frame_decode_us: u64,
    /// The file's modification time — when the producer wrote it, on the same host's clock.
    ///
    /// A proxy for when the frame became available, not a measurement of it: the stamp lands on
    /// the producer's tmp write, a rename before the file is visible, so anything derived from it
    /// must say so (`available_at_source: "mtime"`).
    pub modified: Option<std::time::SystemTime>,
    /// Wall-clock time at the start of the read attempt, before the stat.
    ///
    /// Queue wait is this minus the mtime: how long the frame sat visible before a reader came
    /// for it. Measured at the read's *start* so the read and decode costs — already reported as
    /// `delivery_us` and `frame_decode_us` — are not double-counted into the wait.
    pub read_at: std::time::SystemTime,
    /// Monotonic instant captured immediately before the decode.
    ///
    /// This opens the `standalone_validation_us` boundary: the driver closes it after the pair
    /// commit, so the primary is one continuous wall-clock reading rather than a sum of
    /// separately-timed segments with untimed gaps between them.
    pub validation_open: Instant,
}

/// Reads and decodes exactly one frame file.
///
/// The size check runs on the file's metadata before the read, because the decoder's own bound
/// only applies after `fs::read` has already paid for the allocation — a spool entry past every
/// frame bound is refused without buffering it.
pub fn read_frame_file(path: &Path, limits: &FrameLimits) -> eyre::Result<SpooledFrame> {
    let read_at = std::time::SystemTime::now();
    let read_started = Instant::now();
    let metadata = fs::metadata(path)
        .map_err(|err| eyre::eyre!("cannot stat frame {}: {err}", path.display()))?;
    let modified = metadata.modified().ok();
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
    let delivery_us = read_started.elapsed().as_micros() as u64;
    // One instant serves both readings: it times the decode and opens the validation boundary
    // that the driver will close after the pair commit.
    let decode_started = Instant::now();
    let (header, event, rest) = decode_event(&raw, limits)
        .map_err(|err| eyre::eyre!("frame {} is unusable: {err}", path.display()))?;
    if !rest.is_empty() {
        eyre::bail!(
            "frame {} carries {} trailing bytes; one file holds exactly one frame",
            path.display(),
            rest.len()
        );
    }
    let frame_decode_us = decode_started.elapsed().as_micros() as u64;
    Ok(SpooledFrame {
        header,
        event,
        bytes: raw.len() as u64,
        delivery_us,
        frame_decode_us,
        modified,
        read_at,
        validation_open: decode_started,
    })
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
        let frame = read_frame_file(path, &self.limits)?;
        // One thing may follow an End: the manifest of the next epoch. A producer that restarted
        // into the same directory closed one stream and opened another, and refusing that would
        // make its own recovery unreadable. Everything else past an End is a trailing frame on a
        // closed stream, and replaying it would be replaying something nobody ran.
        if self.ended {
            if frame.header.kind != FrameKind::Manifest {
                eyre::bail!("spool continues past its End frame: {}", path.display());
            }
            self.ended = false;
            self.closed = false;
        }
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
    /// off-by-one this catches — an End carrying its *own* sequence — was a real recorder defect,
    /// found before the live follower shipped.
    #[test]
    fn an_end_frame_that_misnames_its_predecessor_is_refused() {
        let dir = spool_dir("end-off-by-one");
        write(&dir, 0, &filler(), FrameKind::Reset);
        write(&dir, 1, &end(1), FrameKind::End);

        let error = read_spool(&dir, &FrameLimits::default()).expect_err("misnumbered End");
        assert!(error.to_string().contains("names 1 as the last frame"), "{error}");
        let _ = fs::remove_dir_all(&dir);
    }

    /// Delivery is the stat and the read; decode is the boundary's first phase; the mtime rides
    /// along as the availability proxy. All three come from the one read the frame already pays
    /// for.
    #[test]
    fn a_frame_read_reports_its_costs_and_its_mtime() {
        let dir = spool_dir("costs");
        write(&dir, 0, &filler(), FrameKind::Reset);

        let frame = read_frame_file(&dir.join("000000000000_reset.frame"), &FrameLimits::default())
            .expect("reads");
        assert!(frame.modified.is_some(), "the availability proxy travels with the frame");
        assert!(
            frame.modified.unwrap() <= std::time::SystemTime::now(),
            "a fresh file's mtime is in the past"
        );
        assert!(
            frame.read_at >= frame.modified.unwrap(),
            "the read attempt started after the file was written, so queue wait is non-negative"
        );
        assert!(
            frame.validation_open.elapsed() >=
                std::time::Duration::from_micros(frame.frame_decode_us),
            "the validation boundary opened before the decode it times"
        );
        // Zero is a legal reading on a fast filesystem; the fields existing is the contract.
        let _ = (frame.delivery_us, frame.frame_decode_us);
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
