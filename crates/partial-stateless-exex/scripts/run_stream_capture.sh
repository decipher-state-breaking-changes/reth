#!/usr/bin/env bash
#
# Records one corpus: warm a coordinated pair from cold, export a snapshot at the first Ready,
# then write one commit frame per canonical block until asked to stop.
#
# The run is a plain `builder`. It is deliberately not the paired benchmark: `PS_VALIDATION_BENCH`
# triples the per-block cost, keeps the cache in memory only, and produces numbers about the *live*
# path -- none of which this corpus needs, because the A/B it feeds is standalone control against
# candidate over the recorded frames.
#
# What the timeline looks like on this host, from the S2-0 measurement:
#   ~40 s   node start, ExEx WAL load
#   ~12 min cold warm to Ready(H) at replay depth 61
#   ~3 min  whole-cache multiproof and snapshot export, in one long MDBX read transaction
#   then    one commit frame per block, roughly every 12 s
#
# Usage: run_stream_capture.sh <run-dir> [blocks-to-record]

set -euo pipefail

RUN_DIR="${1:?usage: run_stream_capture.sh <run-dir> [blocks]}"
WANT_BLOCKS="${2:-40}"
BINARY="${PS_BINARY:-/data/rust/target/release/reth-partial-stateless}"
DATADIR="${PS_DATADIR:-/data/reth_data}"
JWT="${PS_JWT:-/data/secrets/jwt.hex}"

[ -x "${BINARY}" ] || { echo "not executable: ${BINARY}" >&2; exit 2; }
mkdir -p "${RUN_DIR}"/{sidecars,bootstrap,stream}
LOG="${RUN_DIR}/capture.log"

# `find` rather than a glob: a glob that matches nothing exits non-zero, which under `pipefail`
# turns an empty spool -- the ordinary state at startup -- into a script failure.
count_frames() {
  find "${RUN_DIR}/stream" -maxdepth 1 -name "$1" -type f | wc -l
}

# A non-empty spool is refused by the recorder itself; failing here says so before the node spends
# fifteen minutes warming.
if [ "$(count_frames '*.frame')" -gt 0 ]; then
  echo "${RUN_DIR}/stream already holds frames; point this at a fresh directory" >&2
  exit 2
fi

echo "binary=$(sha256sum "${BINARY}" | cut -c1-16) run=${RUN_DIR} blocks=${WANT_BLOCKS}"

PS_SIDECAR_ROLE=builder \
PS_ENGINE_PAYLOAD=on \
PS_STREAM_DIR="${RUN_DIR}/stream" \
PS_BOOTSTRAP_DIR="${RUN_DIR}/bootstrap" \
PS_SIDECAR_DIR="${RUN_DIR}/sidecars" \
nohup "${BINARY}" node \
  --datadir "${DATADIR}" --authrpc.jwtsecret "${JWT}" \
  --minimal --http --http.api eth,net,web3,debug,trace \
  --authrpc.addr 127.0.0.1 --authrpc.port 8551 \
  --ws --ws.addr 127.0.0.1 --ws.port 8546 --ws.api eth,trace,debug,net --ws.origins localhost \
  > "${LOG}" 2>&1 &
NODE_PID=$!
echo "node pid=${NODE_PID} log=${LOG}"

# SIGTERM rather than SIGKILL: the recorder writes its End frame when the notification stream ends,
# and a corpus without one is indistinguishable from a corpus that was cut.
trap 'kill -TERM "${NODE_PID}" 2>/dev/null || true' INT TERM

while kill -0 "${NODE_PID}" 2>/dev/null; do
  commits=$(count_frames '*_commit.frame')
  if [ "${commits}" -ge "${WANT_BLOCKS}" ]; then
    echo "recorded ${commits} commits; stopping the node so it can close the stream"
    kill -TERM "${NODE_PID}"
    break
  fi
  sleep 15
done

wait "${NODE_PID}" 2>/dev/null || true

echo "frames: $(count_frames '*.frame'), spool bytes: $(du -sb "${RUN_DIR}/stream" | cut -f1)"
echo "ERROR lines: $(grep -c ' ERROR ' "${LOG}" || true)"
