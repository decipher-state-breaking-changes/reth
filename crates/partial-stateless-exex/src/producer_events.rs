//! Producer lifecycle event log: the out-of-band JSONL sibling of the spool.
//!
//! The measurement contract needs recovery availability observable independently of frame
//! `mtime` — event time, export start/end, checkpoint publication, first winning commit — and
//! the v1 frame grammar is frozen, so none of that may ride in a frame body. It lands here
//! instead: one appended JSON line per lifecycle event, beside the run manifest, correlated to
//! frames by sequence number and to attempts by the gate's attempt counter.
//!
//! Best-effort by design: a failed append warns and drops the line. The events are measurement
//! provenance, not consensus state, and a producer that stopped streaming because its telemetry
//! disk filled would invert that priority.

use std::{
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use tracing::warn;

/// Schema version of every line this module writes.
const SCHEMA_VERSION: u32 = 1;

/// Appends producer lifecycle events beside a spool directory.
#[derive(Debug)]
pub(crate) struct ProducerEvents {
    path: PathBuf,
    /// Monotonic origin, so intervals survive a wall-clock step between two stamps.
    origin: Instant,
    epoch: u64,
    /// Set once an append failed, so a dead disk warns once instead of per event.
    failed: bool,
}

impl ProducerEvents {
    /// Sits beside the spool as `<dir>.producer-events.jsonl`, the run-manifest convention.
    pub(crate) fn beside_spool(spool_dir: &Path, epoch: u64) -> Self {
        let name = spool_dir
            .file_name()
            .map_or_else(|| "stream".to_string(), |name| name.to_string_lossy().into_owned());
        let path = spool_dir
            .parent()
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
            .join(format!("{name}.producer-events.jsonl"));
        Self { path, origin: Instant::now(), epoch, failed: false }
    }

    /// Appends one event. `fields` supplies the event-specific keys; the envelope adds schema,
    /// epoch, and both clocks.
    pub(crate) fn emit(&mut self, kind: &str, attempt: u32, fields: serde_json::Value) {
        let mut record = serde_json::json!({
            "schema_version": SCHEMA_VERSION,
            "benchmark": "partial_stateless_producer_events",
            "kind": kind,
            "epoch": self.epoch,
            "attempt": attempt,
            "observed_at_ms": SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |elapsed| elapsed.as_millis() as u64),
            "mono_elapsed_us": self.origin.elapsed().as_micros() as u64,
        });
        if let (Some(record), Some(extra)) = (record.as_object_mut(), fields.as_object()) {
            for (key, value) in extra {
                record.insert(key.clone(), value.clone());
            }
        }
        let mut line = record.to_string();
        line.push('\n');
        let written = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .and_then(|mut file| file.write_all(line.as_bytes()).and_then(|()| file.flush()));
        if let Err(err) = written &&
            !self.failed
        {
            self.failed = true;
            warn!(
                target: "partial_stateless_stream",
                path = %self.path.display(),
                error = %err,
                "Producer event log append failed; further failures are silent"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_append_as_jsonl_with_the_envelope() {
        let dir = std::env::temp_dir().join(format!("ps-producer-events-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let spool = dir.join("spool");
        std::fs::create_dir_all(&spool).unwrap();
        let mut events = ProducerEvents::beside_spool(&spool, 3);
        events.emit("reorg_detected", 0, serde_json::json!({ "ancestor": 7 }));
        events.emit(
            "export_started",
            1,
            serde_json::json!({ "block": 8, "cause": "branch_change" }),
        );

        let raw = std::fs::read_to_string(dir.join("spool.producer-events.jsonl")).unwrap();
        let lines: Vec<serde_json::Value> =
            raw.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["kind"], "reorg_detected");
        assert_eq!(lines[0]["epoch"], 3);
        assert_eq!(lines[0]["ancestor"], 7);
        assert_eq!(lines[1]["attempt"], 1);
        assert_eq!(lines[1]["cause"], "branch_change");
        assert!(lines[1]["observed_at_ms"].as_u64().unwrap() > 0);
        // Monotonic stamps never run backwards between two appends.
        assert!(
            lines[1]["mono_elapsed_us"].as_u64().unwrap() >=
                lines[0]["mono_elapsed_us"].as_u64().unwrap()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
