//! Reading a recorded stream back out of a directory, in the order it was written.
//!
//! Ordering comes from the sequence number the producer stamped into each frame, not from the
//! directory listing and not from the filename. The filename carries the sequence so an operator
//! can read the spool, but a reader that trusted it would be trusting a rename.
//!
//! Contiguity is checked rather than assumed. A recorded corpus with a hole in it looks exactly
//! like a complete one to anything that just decodes what it finds, and a replay over such a
//! corpus would report agreement about a chain nobody ran.

use partial_stateless_stream::{decode_event, FrameHeader, FrameLimits, StreamEvent};
use std::{fs, path::Path};

/// One frame as read back.
#[derive(Debug)]
pub struct SpooledFrame {
    /// The frame's own header, which is where the sequence lives.
    pub header: FrameHeader,
    /// The decoded body.
    pub event: StreamEvent,
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
/// treat an unreadable frame as an absent one.
pub fn read_spool(dir: &Path, limits: &FrameLimits) -> eyre::Result<Spool> {
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

    let mut frames = Vec::with_capacity(paths.len());
    let mut bytes = 0u64;
    for path in &paths {
        let raw = fs::read(path)
            .map_err(|err| eyre::eyre!("cannot read frame {}: {err}", path.display()))?;
        bytes += raw.len() as u64;
        let (header, event, rest) = decode_event(&raw, limits)
            .map_err(|err| eyre::eyre!("frame {} is unusable: {err}", path.display()))?;
        if !rest.is_empty() {
            eyre::bail!(
                "frame {} carries {} trailing bytes; one file holds exactly one frame",
                path.display(),
                rest.len()
            );
        }
        frames.push(SpooledFrame { header, event });
    }

    frames.sort_by_key(|frame| frame.header.sequence);
    for (position, frame) in frames.iter().enumerate() {
        if frame.header.sequence != position as u64 {
            eyre::bail!(
                "spool {} is not contiguous: sequence {} arrived where {position} was expected. \
                 A corpus with a hole reads as a corpus, so this is refused rather than replayed",
                dir.display(),
                frame.header.sequence
            );
        }
    }

    let closed = matches!(frames.last().map(|frame| &frame.event), Some(StreamEvent::End(_)));
    Ok(Spool { frames, closed, bytes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use partial_stateless_stream::{encode_event, End, FrameKind};

    fn spool_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("ps-spool-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    fn write(dir: &Path, sequence: u64, event: &StreamEvent, kind: FrameKind) {
        let bytes = encode_event(sequence, event).expect("encodes");
        fs::write(dir.join(format!("{sequence:012}_{}.frame", kind.as_str())), bytes)
            .expect("write");
    }

    fn end(last: u64) -> StreamEvent {
        StreamEvent::End(End { reason: "test".into(), last_sequence: last })
    }

    #[test]
    fn a_spool_reads_back_in_sequence_order_and_reports_that_it_closed() {
        let dir = spool_dir("ordered");
        write(&dir, 0, &end(0), FrameKind::End);
        write(&dir, 1, &end(1), FrameKind::End);

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
        write(&dir, 0, &end(0), FrameKind::End);
        write(&dir, 2, &end(2), FrameKind::End);

        let error = read_spool(&dir, &FrameLimits::default()).expect_err("gap is refused");
        assert!(error.to_string().contains("not contiguous"), "{error}");
        let _ = fs::remove_dir_all(&dir);
    }

    /// A corpus cut mid-run is usable — it is just shorter — but a driver has to know that it is
    /// reading a prefix rather than a whole stream.
    #[test]
    fn a_stream_without_an_end_frame_reports_that_it_was_cut() {
        let dir = spool_dir("cut");
        write(
            &dir,
            0,
            &StreamEvent::Reset(partial_stateless_stream::Reset {
                reason: partial_stateless_stream::ResetReason::Overflow,
                detail: "test".into(),
            }),
            FrameKind::Reset,
        );

        let spool = read_spool(&dir, &FrameLimits::default()).expect("reads");
        assert!(!spool.closed);
        let _ = fs::remove_dir_all(&dir);
    }
}
