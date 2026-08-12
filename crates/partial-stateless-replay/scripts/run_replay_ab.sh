#!/usr/bin/env bash
#
# Alternating control/candidate A/B over one recorded corpus.
#
# This is the comparison the recorded stream exists to make possible. Two builds consume
# byte-identical frames from byte-identical snapshots, so the live-run 5.7% non-identical-workload
# floor is gone from it — every block one arm replays is the same block the other arm replays,
# down to the bytes.
#
# What does not go away, and what the alternation is for: CPU frequency, allocator state, scheduler
# placement and file-cache order still differ between two processes run at two times. Running
# control-candidate-control-candidate rather than all-control-then-all-candidate is what keeps a
# monotonic host drift from being attributed to the change.
#
# Run it once with the SAME binary in both slots before trusting any result from it. That measures
# the method's own floor on this host, which is the number every later difference has to clear.
#
# Usage:
#   run_replay_ab.sh <spool-dir> <control-binary> <candidate-binary> [rounds] [out.jsonl]

set -euo pipefail

SPOOL="${1:?usage: run_replay_ab.sh <spool-dir> <control-bin> <candidate-bin> [rounds] [out]}"
CONTROL="${2:?control binary}"
CANDIDATE="${3:?candidate binary}"
ROUNDS="${4:-5}"
OUT="${5:-${SPOOL%/}-ab.jsonl}"

for binary in "${CONTROL}" "${CANDIDATE}"; do
  [ -x "${binary}" ] || { echo "not executable: ${binary}" >&2; exit 2; }
done
[ -d "${SPOOL}" ] || { echo "no such spool: ${SPOOL}" >&2; exit 2; }

if [ "${CONTROL}" = "${CANDIDATE}" ]; then
  echo "note: both arms are the same binary, so this run measures the method's floor" >&2
fi

echo "spool=${SPOOL} rounds=${ROUNDS} out=${OUT}"
echo "control=$(sha256sum "${CONTROL}" | cut -c1-16) candidate=$(sha256sum "${CANDIDATE}" | cut -c1-16)"

for round in $(seq 1 "${ROUNDS}"); do
  # Mutations are off in the timed arms. They are a correctness gate, they run three extra
  # admissions per block, and a timing that included them would be measuring the harness.
  echo "round ${round}/${ROUNDS}: control"
  "${CONTROL}" "${SPOOL}" --no-mutations --json "${OUT}" --label "control-r${round}"
  echo "round ${round}/${ROUNDS}: candidate"
  "${CANDIDATE}" "${SPOOL}" --no-mutations --json "${OUT}" --label "candidate-r${round}"
done

echo "wrote ${OUT}"
