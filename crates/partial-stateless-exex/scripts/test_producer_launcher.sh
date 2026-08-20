#!/usr/bin/env bash
# Does the cohort launcher hand the producer the profile it claims to?
#
#     bash crates/partial-stateless-exex/scripts/test_producer_launcher.sh
#
# `bash -n` cannot answer this. A comment placed inside the launcher's line continuation once
# ended the `env` command early: what ran was `env` with assignments and no command — which prints
# the environment and exits — and then a second statement, assembled from the remaining lines,
# that started the producer carrying none of the profile. Both statements parse. The run looked
# slow rather than broken, and cost forty minutes before anyone read the log closely.
#
# So the launcher is executed for real, against a stub that reports what it was given.
set -uo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
LAUNCHER=${LAUNCHER:-$REPO/crates/partial-stateless-exex/scripts/run_f0_sequence.sh}
WORK=$(mktemp -d)
trap 'pkill -f "while sleep 300; do find .$WORK" 2>/dev/null; rm -rf "$WORK"' EXIT

cat > "$WORK/stub" <<'STUB'
#!/usr/bin/env bash
env | grep -E "^PS_" | sort
STUB
chmod +x "$WORK/stub"

# The function under test, with the two globals it reads from its script.
PRODUCER_BIN=$WORK/stub
NODE_FLAGS=(node --datadir "$WORK/nowhere")
eval "$(sed -n '/^start_producer() {/,/^}/p' "$LAUNCHER")"

arm=$WORK/arm
pid=$(start_producer "$arm" 1)
sleep 1

failed=0
check() { # <expected assignment>
    if grep -qx -- "$1" "$arm/producer.out"; then
        echo "  ok: $1"
    else
        echo "  FAIL: the producer never received $1" >&2
        failed=1
    fi
}
echo "==> the canonical profile reaches the producer"
check "PS_ENGINE_ACCESS=on"
check "PS_SHADOW_SAMPLE=50"
check "PS_ENGINE_PAYLOAD=on"
check "PS_STREAM_REORG_CHECKPOINT=always"
check "PS_ACCOUNT_WINDOW=90"
check "PS_STORAGE_WINDOW=60"
check "PS_WITNESS_V3=1"
check "PS_TRIE_REPR=exact"
check "PS_PARALLEL_INITIAL_PROOF=0"
check "PS_RETAIN_GENERATION=1"
check "PS_CANONICAL_REBUILD=0"
check "PS_SIDECAR_ROLE=builder"
check "PS_STREAM_DIR=$arm/spool"
check "PS_BOOTSTRAP_DIR=$arm/bootstrap"
check "PS_SIDECAR_DIR=$arm/sidecars"
check "PS_SHADOW_OUTPUT=$arm/access-shadow.jsonl"
# The per-arm argument, which is the whole point of the fsync ABBA.
check "PS_STREAM_FSYNC=1"

echo "==> and the ones that must be absent are absent"
for absent in PS_CAPTURE_DIR PS_VALIDATION_BENCH PS_POLICY_DATASET_CAPTURE_DIR \
              PS_WITNESS_BASELINE PS_RESOURCE_METRICS PS_TRIE_CACHE_DIAGNOSTICS \
              PS_FORCE_PREVIOUS_CACHE_SNAPSHOT; do
    if grep -q "^$absent=" "$arm/producer.out"; then
        echo "  FAIL: $absent reached the producer" >&2
        failed=1
    else
        echo "  ok: $absent is unset"
    fi
done

echo "==> the function returns a process id and nothing else"
if [[ "$pid" =~ ^[0-9]+$ ]]; then
    echo "  ok: pid '$pid'"
else
    echo "  FAIL: '$pid' is not a pid — the gate would signal the wrong thing" >&2
    failed=1
fi

[ "$failed" -eq 0 ] && echo "==> LAUNCHER OK" || { echo "==> LAUNCHER BROKEN" >&2; exit 1; }
