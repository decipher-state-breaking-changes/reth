#!/usr/bin/env bash
#
# Build-profile guard for the standalone database-free validator.
#
# Three invariants, all of which have to hold for the standalone claim to mean anything, and none
# of which any test can observe:
#
#   1. No Reth provider or database implementation is reachable from the package's *normal*
#      dependency graph. This is what makes "database-free" a compile-time property rather than a
#      convention: the binary has no provider handle to misuse, even on an error, restart, gap, or
#      reorg branch. Trait and error crates (reth-storage-api, reth-storage-errors) are allowed on
#      purpose -- they carry no implementation. Dev-dependencies are excluded because test code may
#      legitimately build a provider-backed oracle; `-e normal` is what expresses that.
#
#   2. The keccak features the production binary runs are actually selected. Building one package
#      selects only that package's dependency graph, so a package that does not declare these is
#      silently built without them. That defect invalidated every absolute benchmark number in the
#      benchmark history before 2026-08-06, and a new standalone package is a fresh chance to
#      reintroduce it -- the standalone validator is built as its own binary, which is exactly the
#      build that would not inherit anything from the ExEx.
#
#   3. The signature-recovery backend the production binary runs is actually selected. This
#      package recovers senders itself, and without `secp256k1` on reth-primitives-traits the
#      recovery silently falls back to the pure-Rust k256 path. It was missing at first and nothing
#      showed it, because partial-stateless-exex pulls the feature in through the node graph and a
#      graph check on the ExEx would have passed. Same shape of defect as invariant 2, on a
#      different hot path.
#
# The three packages below are checked together because the claim is about the *binary*, and the
# binary is built from all three. `partial-stateless-stream` is where the frame format and the
# recorded oracle live, and `partial-stateless-replay` is the standalone process itself — the first
# thing that runs the validator core outside a reth node. A graph check that covered only the core
# would pass while the process that runs it linked a provider.
#
# Usage: check_validator_isolation.sh [package-name ...]

set -euo pipefail

if [ "$#" -gt 0 ]; then
  PACKAGES=("$@")
else
  PACKAGES=(
    partial-stateless-validator
    partial-stateless-stream
    partial-stateless-replay
  )
fi

# Implementations, not traits. reth-storage-api and reth-storage-errors are deliberately absent.
FORBIDDEN='^(reth-provider|reth-db|reth-db-common|reth-libmdbx|reth-mdbx-sys|reth-exex|reth-node-builder)$'
REQUIRED_FEATURES='feature "(asm-keccak|keccak-cache-global)"'
REQUIRED_RECOVERY='feature "secp256k1"'

status=0

check_package() {
  local PKG="$1"

echo "==> ${PKG}: normal dependency graph"
if ! deps="$(cargo tree -p "${PKG}" -e normal --prefix none 2>/dev/null)"; then
  echo "FAIL: cargo tree could not resolve package '${PKG}'" >&2
  status=2
  return
fi

if hits="$(printf '%s\n' "${deps}" | awk '{print $1}' | sort -u | grep -E "${FORBIDDEN}")"; then
  echo "FAIL: forbidden dependencies reachable from ${PKG}:" >&2
  printf '  %s\n' ${hits} >&2
  status=1
else
  echo "ok: no provider/database implementation crate is reachable"
fi

echo "==> ${PKG}: keccak build profile"
edges="$(cargo tree -p "${PKG}" -e features -i alloy-primitives 2>/dev/null \
  | grep -cE "${REQUIRED_FEATURES}" || true)"
if [ "${edges}" -eq 0 ]; then
  echo "FAIL: ${PKG} selects neither asm-keccak nor keccak-cache-global on alloy-primitives." >&2
  echo "      Any timing measured from this build describes a keccak production does not run." >&2
  status=1
else
  echo "ok: ${edges} asm-keccak/keccak-cache-global feature edges on alloy-primitives"
fi

echo "==> ${PKG}: signature recovery backend"
recovery="$(cargo tree -p "${PKG}" -e features -i reth-primitives-traits 2>/dev/null \
  | grep -cE "${REQUIRED_RECOVERY}" || true)"
if [ "${recovery}" -eq 0 ]; then
  echo "FAIL: ${PKG} does not select secp256k1 on reth-primitives-traits." >&2
  echo "      Sender recovery would fall back to k256, which production does not run." >&2
  status=1
else
  echo "ok: ${recovery} secp256k1 feature edges on reth-primitives-traits"
fi
}

for pkg in "${PACKAGES[@]}"; do
  check_package "${pkg}"
done

exit "${status}"
