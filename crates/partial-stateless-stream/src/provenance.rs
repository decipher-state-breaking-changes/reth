//! Best-effort run provenance, stamped by both ends of the stream.
//!
//! Not part of the wire format: nothing here is framed, sequenced, or digest-checked, and no
//! frame ever carries it. It lives in this crate only because this is the one crate the producer
//! and every consumer already share, and a provenance record whose fields differ per process is
//! a record that cannot be joined.
//!
//! Everything is best-effort by design. A missing `/proc` entry, an unreadable binary, or an
//! unset compile-time commit yields `null` plus a note saying why — never a guess and never a
//! panic — because the record's job is to make a run attributable, and an honest gap is
//! attributable while a fabricated value is not. The precedent is this project's own: a missing
//! cargo-feature list hid a build-profile defect across every early benchmark, because provenance
//! that is not collected automatically ends up not collected.

use alloy_primitives::keccak256;
use serde::Serialize;
use std::{collections::BTreeMap, fs, path::Path};

/// Binaries past this size are fingerprinted by size alone rather than hashed.
///
/// A debug-profile reth binary runs to gigabytes, and hashing it would turn a startup stamp into
/// a startup stall. The note names the skip, so a reader knows the fingerprint is weaker.
const MAX_HASHED_BINARY_BYTES: u64 = 512 * 1024 * 1024;

/// Compile-time build identity, captured by the crate that defines the process.
///
/// Every field is an `option_env!` reading — `PS_BUILD_COMMIT`, `PS_BUILD_DIRTY` (`"0"`/`"1"`),
/// and `PS_CARGO_LOCK_SHA256` — taken at *compile* time by the calling crate, because only the
/// defining crate can capture its own build. Cargo tracks `option_env!` inputs, so exporting a
/// different value and rebuilding re-stamps the binary; reading them at *run* time would let a
/// stale binary claim whatever commit the shell happened to be on.
#[derive(Debug, Clone, Copy, Default)]
pub struct BuildStamp<'a> {
    /// `git rev-parse HEAD` at build time.
    pub commit: Option<&'a str>,
    /// Whether the working tree was dirty at build time: `"0"` clean, `"1"` dirty.
    pub dirty: Option<&'a str>,
    /// SHA-256 of the workspace `Cargo.lock` at build time.
    pub cargo_lock_sha256: Option<&'a str>,
}

/// One process's account of what ran, where, and under what host conditions.
#[derive(Debug, Clone, Serialize)]
pub struct RunProvenance {
    /// Who collected this: the crate name and version of the process stamping the run.
    pub collector: String,
    /// The build commit the binary was compiled from, when the build embedded one.
    pub build_commit: Option<String>,
    /// Whether the working tree was dirty when the binary was built. A dirty build cannot serve
    /// as code-freeze evidence: the commit it names is not the code that ran.
    pub build_dirty: Option<bool>,
    /// SHA-256 of the workspace `Cargo.lock` the binary was built under.
    pub cargo_lock_sha256: Option<String>,
    /// Path of the running executable.
    pub binary: Option<String>,
    /// Size of the executable in bytes.
    pub binary_bytes: Option<u64>,
    /// Keccak-256 of the executable — a fingerprint, not a distribution checksum.
    pub binary_keccak256: Option<String>,
    /// The process's command line, verbatim.
    pub args: Vec<String>,
    /// Environment variables that steer these components: the `PS_*` family plus `RUST_LOG`.
    pub env: BTreeMap<String, String>,
    /// Host name.
    pub hostname: Option<String>,
    /// Kernel type and release.
    pub kernel: Option<String>,
    /// CPU model string, from the first `/proc/cpuinfo` entry.
    pub cpu_model: Option<String>,
    /// Logical CPUs available to this process.
    pub cpu_count: Option<u64>,
    /// The cpufreq governor of cpu0. Recorded because a measured run is only comparable against
    /// another run at the same governor, so the manifest has to name it rather than assume it.
    pub governor: Option<String>,
    /// Installed memory in kilobytes.
    pub total_memory_kb: Option<u64>,
    /// The directory the run works against (spool or output), as given.
    pub target_dir: Option<String>,
    /// Device, filesystem type, and mount point holding `target_dir`.
    pub filesystem: Option<String>,
    /// Wall-clock collection time, milliseconds since the epoch.
    pub collected_at_ms: u128,
    /// Why any field above is `null`. Empty when everything collected.
    pub notes: Vec<String>,
}

impl RunProvenance {
    /// Collects what this host and process will say about themselves, never failing.
    ///
    /// `collector` names the stamping process (`"ps-replay 0.1.0"`); `build` is the caller's
    /// `option_env!`-captured [`BuildStamp`], because only the defining crate can capture its
    /// own; `target_dir` is the directory whose filesystem matters to the run.
    pub fn collect(collector: &str, build: BuildStamp<'_>, target_dir: Option<&Path>) -> Self {
        let mut notes = Vec::new();
        if build.commit.is_none() {
            notes.push("build commit not embedded at compile time".to_string());
        }
        let build_dirty = match build.dirty {
            Some("0") => Some(false),
            Some("1") => Some(true),
            Some(other) => {
                notes.push(format!("PS_BUILD_DIRTY was `{other}`, not `0`/`1`; recorded as null"));
                None
            }
            None => {
                notes.push("build dirty flag not embedded at compile time".to_string());
                None
            }
        };
        if build.cargo_lock_sha256.is_none() {
            notes.push("Cargo.lock hash not embedded at compile time".to_string());
        }

        let binary = std::env::current_exe().ok();
        let (binary_bytes, binary_keccak256) = match &binary {
            Some(path) => fingerprint_binary(path, &mut notes),
            None => {
                notes.push("current_exe unavailable".to_string());
                (None, None)
            }
        };

        let env = std::env::vars()
            .filter(|(key, _)| key.starts_with("PS_") || key == "RUST_LOG")
            .collect();

        let filesystem = target_dir.and_then(|dir| filesystem_of(dir, &mut notes));

        Self {
            collector: collector.to_string(),
            build_commit: build.commit.map(str::to_string),
            build_dirty,
            cargo_lock_sha256: build.cargo_lock_sha256.map(str::to_string),
            binary: binary.as_ref().map(|path| path.display().to_string()),
            binary_bytes,
            binary_keccak256,
            args: std::env::args().collect(),
            env,
            hostname: read_note("/proc/sys/kernel/hostname", &mut notes),
            kernel: kernel_string(&mut notes),
            cpu_model: cpu_model(&mut notes),
            cpu_count: std::thread::available_parallelism().ok().map(|n| n.get() as u64),
            governor: read_note(
                "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
                &mut notes,
            ),
            total_memory_kb: total_memory_kb(&mut notes),
            target_dir: target_dir.map(|dir| dir.display().to_string()),
            filesystem,
            collected_at_ms: std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .map(|elapsed| elapsed.as_millis())
                .unwrap_or(0),
            notes,
        }
    }
}

/// Reads a one-line pseudo-file, noting the miss instead of failing.
fn read_note(path: &str, notes: &mut Vec<String>) -> Option<String> {
    match fs::read_to_string(path) {
        Ok(raw) => Some(raw.trim().to_string()),
        Err(err) => {
            notes.push(format!("{path}: {err}"));
            None
        }
    }
}

fn kernel_string(notes: &mut Vec<String>) -> Option<String> {
    let ostype = read_note("/proc/sys/kernel/ostype", notes)?;
    let release = read_note("/proc/sys/kernel/osrelease", notes)?;
    Some(format!("{ostype} {release}"))
}

fn cpu_model(notes: &mut Vec<String>) -> Option<String> {
    let cpuinfo = read_note("/proc/cpuinfo", notes)?;
    cpuinfo
        .lines()
        .find(|line| line.starts_with("model name"))
        .and_then(|line| line.split_once(':'))
        .map(|(_, model)| model.trim().to_string())
}

fn total_memory_kb(notes: &mut Vec<String>) -> Option<u64> {
    let meminfo = read_note("/proc/meminfo", notes)?;
    meminfo
        .lines()
        .find(|line| line.starts_with("MemTotal:"))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|kb| kb.parse().ok())
}

fn fingerprint_binary(path: &Path, notes: &mut Vec<String>) -> (Option<u64>, Option<String>) {
    let size = match fs::metadata(path) {
        Ok(metadata) => metadata.len(),
        Err(err) => {
            notes.push(format!("binary metadata: {err}"));
            return (None, None)
        }
    };
    if size > MAX_HASHED_BINARY_BYTES {
        notes.push(format!(
            "binary is {size} bytes, past the {MAX_HASHED_BINARY_BYTES}-byte hashing bound; \
             fingerprinted by size only"
        ));
        return (Some(size), None)
    }
    match fs::read(path) {
        Ok(bytes) => (Some(size), Some(format!("{:?}", keccak256(&bytes)))),
        Err(err) => {
            notes.push(format!("binary read: {err}"));
            (Some(size), None)
        }
    }
}

/// The mount holding `dir`, from the longest matching mount point in `/proc/mounts`.
fn filesystem_of(dir: &Path, notes: &mut Vec<String>) -> Option<String> {
    let resolved = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    let mounts = read_note("/proc/mounts", notes)?;
    let mut best: Option<(usize, String)> = None;
    for line in mounts.lines() {
        let mut fields = line.split_whitespace();
        let (Some(device), Some(mount_point), Some(fs_type)) =
            (fields.next(), fields.next(), fields.next())
        else {
            continue
        };
        if resolved.starts_with(mount_point) &&
            best.as_ref().is_none_or(|(depth, _)| mount_point.len() > *depth)
        {
            best = Some((mount_point.len(), format!("{device} {fs_type} {mount_point}")));
        }
    }
    if best.is_none() {
        notes.push(format!("no mount matches {}", resolved.display()));
    }
    best.map(|(_, description)| description)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The collector never fails, and what it could not read it explains. On any host at all,
    /// args and the collected-at stamp exist, and every absent field has a note or a reason.
    #[test]
    fn collection_is_best_effort_and_self_describing() {
        let provenance = RunProvenance::collect(
            "test 0.0.0",
            BuildStamp::default(),
            Some(Path::new("/definitely/missing")),
        );
        assert!(!provenance.args.is_empty());
        assert!(provenance.collected_at_ms > 0);
        assert!(
            provenance.notes.iter().any(|note| note.contains("build commit")),
            "an unset build commit is explained, not silent"
        );
        assert!(
            provenance.notes.iter().any(|note| note.contains("dirty")),
            "an unset dirty flag is explained, not silent"
        );
        assert!(provenance.filesystem.is_none() || provenance.target_dir.is_some());
    }

    /// The env capture is a filter, not a dump: only the `PS_*` family and `RUST_LOG` steer
    /// these components, and a full environment dump would leak whatever else the shell holds.
    #[test]
    fn only_steering_environment_variables_are_captured() {
        let provenance = RunProvenance::collect("test 0.0.0", BuildStamp::default(), None);
        assert!(provenance.env.keys().all(|key| key.starts_with("PS_") || key == "RUST_LOG"));
    }

    /// The dirty flag is a claim about code-freeze evidence, so it is parsed strictly: `0`/`1`
    /// mean what they say and anything else is a noted null, never a guess.
    #[test]
    fn the_dirty_flag_is_parsed_strictly_and_misreadings_are_noted() {
        let stamp = |dirty| BuildStamp { commit: Some("abc"), dirty, cargo_lock_sha256: None };
        assert_eq!(RunProvenance::collect("t", stamp(Some("0")), None).build_dirty, Some(false));
        assert_eq!(RunProvenance::collect("t", stamp(Some("1")), None).build_dirty, Some(true));
        let odd = RunProvenance::collect("t", stamp(Some("maybe")), None);
        assert_eq!(odd.build_dirty, None);
        assert!(odd.notes.iter().any(|note| note.contains("maybe")));
    }
}
