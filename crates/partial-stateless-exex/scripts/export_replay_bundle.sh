#!/usr/bin/env bash
# Packages a recorded spool for replay on another machine, and writes the exact commands the
# operator runs there. The F2 two-machine replay: `standalone_validation_us` is the CPU/memory
# side of the cost and `delivery_us` the storage side, and a second host separates them only if
# it demonstrably replayed the same bytes with the same build — which is what the hash manifest
# and the pinned commit in the command sheet are for.
#
# Usage:
#     export_replay_bundle.sh <spool-dir> <bundle-dir> [label]
#
# Produces, in <bundle-dir>:
#     corpus-manifest.json         every frame's size and SHA-256, corpus totals, and the source
#     producer-run-manifest.jsonl  the producer's own provenance, copied from beside the spool
#     verify_corpus.sh             re-checks the manifest against a copied spool, byte for byte
#     RUN-ON-SECOND-HOST.md        paste-ready commands: provenance-stamped build at the pinned
#                                  commit, isolation check, hash verification, batch replay, and
#                                  the files to send back
#
# The spool itself is NOT copied: 6,000 blocks are ~19 GiB and the transfer tool is the
# operator's choice (rsync -a preserves mtimes, which the availability proxy reads).
set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'
    exit 64
fi

SPOOL_DIR=$(cd "$1" && pwd)
BUNDLE_DIR=$2
LABEL=${3:-f2-second-host}
mkdir -p "$BUNDLE_DIR"

REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
DIRTY=$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null | head -1)
if [ -n "$DIRTY" ] || [ "$COMMIT" = "unknown" ]; then
    echo "error: the working tree is dirty or not a git checkout; a bundle must pin a commit" >&2
    echo "       (the archive's build-provenance rule: commit first, then package)" >&2
    exit 1
fi

echo "==> hashing $SPOOL_DIR"
MANIFEST="$BUNDLE_DIR/corpus-manifest.json"
python3 - "$SPOOL_DIR" "$MANIFEST" "$COMMIT" <<'PYEOF'
import hashlib, json, os, sys
spool, out, commit = sys.argv[1], sys.argv[2], sys.argv[3]
frames = sorted(name for name in os.listdir(spool) if name.endswith(".frame"))
entries, total = [], 0
for name in frames:
    path = os.path.join(spool, name)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    size = os.path.getsize(path)
    total += size
    entries.append({"name": name, "bytes": size, "sha256": digest.hexdigest()})
json.dump(
    {
        "corpus_manifest_version": 1,
        "source_spool": spool,
        "pinned_commit": commit,
        "frames": len(entries),
        "total_bytes": total,
        "entries": entries,
    },
    open(out, "w"),
    indent=1,
)
print(f"    {len(entries)} frames, {total} bytes")
PYEOF

# The producer stamped its own provenance beside the spool; a bundle that leaves it behind
# ships a corpus that cannot say which build recorded it.
PRODUCER_MANIFEST="$(dirname "$SPOOL_DIR")/$(basename "$SPOOL_DIR").run-manifest.jsonl"
if [ -f "$PRODUCER_MANIFEST" ]; then
    cp "$PRODUCER_MANIFEST" "$BUNDLE_DIR/producer-run-manifest.jsonl"
    echo "==> included the producer run manifest"
else
    echo "warning: no producer run manifest at $PRODUCER_MANIFEST — the corpus's own build" >&2
    echo "         provenance is missing from this bundle; record where the spool came from" >&2
fi

cat > "$BUNDLE_DIR/verify_corpus.sh" <<'VERIFY'
#!/usr/bin/env bash
# Re-checks a copied spool against corpus-manifest.json. Run before replaying: a replay of a
# corrupted copy reports on a corpus nobody recorded.
#     verify_corpus.sh <spool-dir> [manifest]
set -euo pipefail
SPOOL_DIR=$1
MANIFEST=${2:-$(dirname "$0")/corpus-manifest.json}
python3 - "$SPOOL_DIR" "$MANIFEST" <<'PYEOF'
import hashlib, json, os, sys
spool, manifest_path = sys.argv[1], sys.argv[2]
manifest = json.load(open(manifest_path))
bad = 0
for entry in manifest["entries"]:
    path = os.path.join(spool, entry["name"])
    if not os.path.exists(path):
        print(f"MISSING {entry['name']}"); bad += 1; continue
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != entry["sha256"]:
        print(f"MISMATCH {entry['name']}"); bad += 1
extra = sorted(set(n for n in os.listdir(spool) if n.endswith(".frame")) -
               set(e["name"] for e in manifest["entries"]))
for name in extra:
    print(f"EXTRA {name}"); bad += 1
if bad:
    print(f"FAILED: {bad} problems"); sys.exit(1)
print(f"OK: {manifest['frames']} frames, {manifest['total_bytes']} bytes, all hashes match")
PYEOF
VERIFY
chmod +x "$BUNDLE_DIR/verify_corpus.sh"

cat > "$BUNDLE_DIR/RUN-ON-SECOND-HOST.md" <<RUNMD
# F2 corpus replay — second host

Everything below is paste-ready. The corpus is the evidence, so the order is: verify the copy,
pin the build, prove the isolation, then replay. Send back the files in the last section.

## 0. What to copy to this host

- the spool directory (use \`rsync -a\` — it preserves mtimes, which the availability proxy reads)
- this bundle directory (manifest + verifier + this sheet)

## 1. Verify the copied corpus

\`\`\`bash
./verify_corpus.sh /path/to/copied/spool
\`\`\`

## 2. Build at the pinned commit, provenance-stamped

\`\`\`bash
cd /path/to/reth
git fetch && git checkout $COMMIT
export PS_BUILD_COMMIT=\$(git rev-parse HEAD)
export PS_BUILD_DIRTY=\$([ -z "\$(git status --porcelain)" ] && echo 0 || echo 1)
export PS_CARGO_LOCK_SHA256=\$(sha256sum Cargo.lock | cut -d' ' -f1)
cargo build --release -p partial-stateless-replay
crates/partial-stateless-exex/scripts/check_validator_isolation.sh
\`\`\`

The three exports are read at *compile* time and baked into the binary, so the run manifest the
replay stamps can name its own build; a dirty tree stamps \`build_dirty: true\` and the record
says so. The isolation script must pass — it is what proves the binary you are about to run has
no provider or database path and carries the keccak and secp256k1 build profile.

## 3. Replay

\`\`\`bash
OUT=\$HOME/f2-replay
mkdir -p "\$OUT"
target/release/ps-replay /path/to/copied/spool \\
    --no-mutations \\
    --json "\$OUT/batch-$LABEL.jsonl" \\
    --label "$LABEL" |& tee "\$OUT/batch-$LABEL.log"
echo "exit: \$?"
\`\`\`

\`--no-mutations\` because F2 is a timing replay: the mutation sweep quadruples admission calls
and is coverage work, not measurement. The run must end \`agreed/continuous/complete\` (exit 0).

## 4. Send back

- \`\$OUT/batch-$LABEL.jsonl\`  (the record — includes the run manifest line and per-block timings)
- \`\$OUT/batch-$LABEL.log\`    (the log)
- the output of: \`uname -a; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor\`
  (the run manifest captures these too; the copy is a cross-check)

Pinned commit: \`$COMMIT\`
Corpus: $(jq -r '.frames' "$MANIFEST") frames, $(jq -r '.total_bytes' "$MANIFEST") bytes
RUNMD

echo "==> bundle ready in $BUNDLE_DIR"
echo "    corpus-manifest.json, verify_corpus.sh, RUN-ON-SECOND-HOST.md (commit $COMMIT)"
