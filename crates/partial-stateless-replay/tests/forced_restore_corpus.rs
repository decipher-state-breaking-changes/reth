//! The offline rebootstrap, judged against a *recorded* corpus that carries a real reorg.
//!
//! `--force-restore-at` answers one question: could a consumer holding nothing install the
//! producer's recovery checkpoint and carry on? Under write-through publication the winning
//! branch is published *before* that checkpoint, so the commits between the ancestor it names
//! and its own frame sit behind the reader. A driver that installs and then reads forward skips
//! them and refuses the next block for a parent hash it never had a chance to build — which is
//! what both 2026-08-21 rehearsals did, on two hosts, against the same mainnet reorg.
//!
//! The synthetic spools in `batch_reorg` cannot pin this: no synthetic commit passes mainnet
//! admission, so nothing there can prove the window replayed. Nor can a corpus whose recovery
//! checkpoint is the last thing in the stream — the earlier run that passed had seven
//! winning-branch commits below its checkpoint and nothing but chunks and an `End` above it, so
//! the driver installed at the ancestor, met the end of the corpus, and was never asked to chain
//! a block onto the restored pair. It reported that the winning branch agreed throughout without
//! having replayed any of it. That is how this defect survived a green gate.
//!
//! So the assertion that matters is not "did it finish" but "did it replay": this test requires
//! `rewind_replayed_commits` to be non-zero before it looks at anything else. A driver that
//! installs and reads on scores zero there on every corpus, including the one that used to pass.
//!
//! Corpus-gated and `#[ignore]`d — the corpus is gigabytes and lives beside the bench runs:
//!
//! ```text
//! PS_FORCED_RESTORE_FIXTURE_SPOOL=<recorded-spool-with-a-reorg> \
//!     cargo test -p partial-stateless-replay --release --test forced_restore_corpus \
//!     -- --ignored --test-threads=1
//! ```
//!
//! With the variable missing at run time these panic rather than skip: a coverage test that
//! reports success without having run anything is worse than no test.

use partial_stateless_replay::{replay, ReplayOptions};
use partial_stateless_stream::{decode_event, FrameKind, FrameLimits, StreamEvent};
use std::{fs, path::PathBuf};

/// The recorded spool, and the sequence of the recovery checkpoint inside it.
///
/// The stream's *first* checkpoint bootstraps the replay; a second one is a recovery checkpoint,
/// which is the frame this test forces an install at.
fn fixture() -> (PathBuf, u64) {
    let dir = match std::env::var("PS_FORCED_RESTORE_FIXTURE_SPOOL") {
        Ok(path) if !path.is_empty() => PathBuf::from(path),
        _ => panic!(
            "PS_FORCED_RESTORE_FIXTURE_SPOOL is not set; this #[ignore]d test only runs against \
             a recorded spool whose producer met a reorg and published a recovery checkpoint"
        ),
    };
    let mut checkpoints = Vec::new();
    for entry in fs::read_dir(&dir).expect("the fixture spool reads") {
        let path = entry.expect("a spool entry").path();
        if path.extension().is_none_or(|extension| extension != "frame") {
            continue
        }
        let name = path.file_name().and_then(|name| name.to_str()).unwrap_or_default();
        if !name.contains(FrameKind::Checkpoint.as_str()) {
            continue
        }
        let bytes = fs::read(&path).expect("a checkpoint frame reads");
        let (header, event, _) = decode_event(&bytes, &FrameLimits::default()).expect("decodes");
        if matches!(event, StreamEvent::Checkpoint(_)) {
            checkpoints.push(header.sequence);
        }
    }
    checkpoints.sort_unstable();
    let recovery = *checkpoints.get(1).unwrap_or_else(|| {
        panic!(
            "the fixture spool at {} carries {} checkpoint(s); this test needs the recovery \
             checkpoint a reorg publishes, which is the second one",
            dir.display(),
            checkpoints.len()
        )
    });
    (dir, recovery)
}

/// Installing the recovery checkpoint rewinds the pair below commits the corpus already
/// delivered, and those commits are replayed from the spool rather than skipped.
#[test]
#[ignore = "needs PS_FORCED_RESTORE_FIXTURE_SPOOL; see the module header"]
fn a_forced_restore_replays_the_window_below_its_checkpoint() {
    let (dir, recovery) = fixture();
    let options = ReplayOptions { force_restore_at: Some(recovery), ..ReplayOptions::default() };
    let report = replay(&dir, &options).expect("the forced replay runs to the end of the corpus");

    assert!(
        report.rewind_replayed_commits > 0,
        "the corpus published no commits between the ancestor and its recovery checkpoint, so it \
         cannot judge this path: a checkpoint at the stream tail passes whether or not the window \
         can replay. Use a corpus whose producer kept recording after the reorg"
    );
    assert_eq!(
        report.disagreements.len(),
        0,
        "the winning branch replayed against the installed checkpoint disagreed: {:?}",
        report.disagreements
    );
    assert_eq!(
        report.failures.len(),
        0,
        "the winning branch replayed against the installed checkpoint failed: {:?}",
        report.failures
    );
    assert!(report.agreed(), "the forced replay must agree with the recording");
    assert!(report.complete(), "the forced replay must reach the end of the corpus");
    assert_eq!(report.rewind_windows_refused, 0, "the window is far below the frame bound");

    let resync = report
        .resyncs
        .iter()
        .find(|record| record.at_sequence == recovery)
        .expect("the forced install is recorded as a resync at the checkpoint's own sequence");
    assert!(resync.continuous, "a checkpoint that lands on the reorg's ancestor is continuous");
    assert_eq!(resync.unverified, None, "nothing between the ancestor and the tip went unverified");
}

/// The same corpus, replayed without forcing: the driver undoes the reorg with its own retained
/// generation and skims the checkpoint. No window is opened, and nothing is replayed twice.
#[test]
#[ignore = "needs PS_FORCED_RESTORE_FIXTURE_SPOOL; see the module header"]
fn an_ordinary_replay_of_the_same_corpus_opens_no_window() {
    let (dir, _) = fixture();
    let report = replay(&dir, &ReplayOptions::default()).expect("the ordinary replay runs");

    assert_eq!(
        report.rewind_replayed_commits, 0,
        "an unforced replay skims the recovery checkpoint against the generation it already \
         holds; it must not replay anything a second time"
    );
    assert!(report.agreed(), "the ordinary replay must agree with the recording");
    assert!(report.complete(), "the ordinary replay must reach the end of the corpus");
}
