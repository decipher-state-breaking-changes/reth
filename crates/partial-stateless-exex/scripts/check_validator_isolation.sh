#!/usr/bin/env bash
#
# Build-profile guard for the standalone database-free validator (Phase 4b, S0).
#
# Two invariants, both of which have to hold for the standalone claim to mean anything, and
# neither of which any test can observe:
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
#      archive before 2026-08-06 (plan section 1), and a new standalone package is a fresh chance
#      to reintroduce it -- S3 builds this package as its own binary, which is exactly the build
#      that would not inherit anything from the ExEx.
#
# Usage: check_validator_isolation.sh [package-name]

set -euo pipefail

PKG="${1:-partial-stateless-validator}"

# Implementations, not traits. reth-storage-api and reth-storage-errors are deliberately absent.
FORBIDDEN='^(reth-provider|reth-db|reth-db-common|reth-libmdbx|reth-mdbx-sys|reth-exex|reth-node-builder)$'
REQUIRED_FEATURES='feature "(asm-keccak|keccak-cache-global)"'

status=0

echo "==> ${PKG}: normal dependency graph"
if ! deps="$(cargo tree -p "${PKG}" -e normal --prefix none 2>/dev/null)"; then
  echo "FAIL: cargo tree could not resolve package '${PKG}'" >&2
  exit 2
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

exit "${status}"
