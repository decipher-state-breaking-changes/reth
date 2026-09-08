#!/usr/bin/env bash
#
# Build-profile guard for the standalone database-free validator.
#
# Five invariants, all of which have to hold for the standalone claim to mean anything, and none
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
#   4. The sparse trie's parallelism is actually compiled in. `std` on reth-trie-sparse is not a
#      portability knob: `is_reveal_parallelism_enabled` and `is_update_parallelism_enabled` are
#      `false` in nostd builds, so proof reveal and subtrie hash updates run serially. The
#      workspace declares the crate `default-features = false`, and the ExEx turns it on only by
#      accident of linking the node graph -- so, once more, a graph check on the ExEx would have
#      passed while the standalone binary ran a different trie. It did: every consumer cohort up to
#      2026-09-08 was measured serial, `account_trie_us` reported zero throughout because `timed()`
#      is gated on the same feature, and nothing failed. Third instance of invariant 2's defect.
#
# The packages below are checked together because the claim is about the *binaries*, and each is
# built from several of them. `partial-stateless-stream` is where the frame format and the recorded
# oracle live, and `partial-stateless-replay` is the standalone process itself — the first thing
# that runs the validator core outside a reth node. A graph check that covered only the core would
# pass while the process that runs it linked a provider.
#
# `partial-stateless-frontier` is here for a sharper reason than the others. It generates every
# cache policy's sidecars from a recorded witness, and the entire claim that a policy comparison is
# a fact about the policies rests on every proof request being answered from that recording. A
# provider reachable from its graph would let some requests be answered from a database instead,
# and the sidecars it produced would no longer be the ones the corpus determines — with nothing in
# the output to show which were which.
#
# Usage: check_validator_isolation.sh [package-name ...]
#
# All four are read off the *source* with `cargo tree`, which answers "would a build from this tree
# be correct?" -- not "is the binary about to be measured correct?". So there is a fifth check, on
# the artifact:
#
#   5. The built binary carries this commit. Every binary embeds `PS_BUILD_COMMIT` at compile time,
#      so a binary built from other code than the tree just checked is detectable, and that is the
#      whole gap: with `std` selected unconditionally in `partial-stateless/Cargo.toml` and the
#      keccak features in each binary crate's `default`, a binary built from a commit whose graph
#      passes invariants 2 to 4 has them. Commit plus graph is the property; nothing further needs
#      asking of the binary. It is not hypothetical -- on 2026-09-08 the graph check passed while
#      `target/release/ps-replay` predated the fix it was passing on, and a measurement was nearly
#      taken from it. §5 of the runbook already carried this check as a one-line `grep` on one
#      binary; this generalises it to all of them, `partial-stateless-exex` included, which no
#      other invariant reaches.
#
# The allocator is deliberately not checked here. It *is* a build-time choice that a commit does
# not imply, being `--features jemalloc` on the command line -- but `RunProvenance` already records
# it in every run manifest, and the runbook reads it there. A second copy in this script would be
# a second thing to keep true.
#
# `PS_ISOLATION_REQUIRE_BINARY=1` turns a missing binary from a warning into a failure. Use it
# wherever the artifact is about to be measured; the default warns, because the guard is also run
# before anything is built.
#
# `PS_ISOLATION_FEATURES` passes a feature list through to every `cargo tree` below, so an arm
# built with non-default features is checked as the graph it actually links rather than as the
# default one. A build profile that differs from the checked profile is precisely the defect
# invariants 2 to 4 exist to catch, and an allocator arm is a build profile.
#
# It is filtered per package before use. It is one string for a run spanning several packages, and
# they do not all declare the same features -- `jemalloc` is a binary crate's selection and the two
# library packages have no such feature. Passing it to a package that does not declare it makes
# `cargo tree` refuse the whole invocation, and a guard that fails for a reason unrelated to any
# invariant is a guard that gets skipped. The drop is announced rather than silent: a run believing
# it checked a profile it did not is the failure this script exists to prevent.
#
# Every feature check below passes `-e normal,build,features` rather than `-e features`. That is
# not cosmetic: `-e features` alone leaves cargo's dev edges in, so a feature reachable *only*
# through a dev-dependency satisfies the grep while the shipped binary does not have it -- the
# exact false pass these invariants exist to prevent, and the same reason invariant 1 has always
# used `-e normal`. Measured on this workspace 2026-09-08: `reth-trie` is a dev-dependency of
# `partial-stateless` and nothing else, and it appears twice under `-e features` and zero times
# under `-e normal,build,features`.

set -euo pipefail

DECLARED_FEATURES="$(cargo metadata --no-deps --format-version 1 2>/dev/null \
  | python3 -c 'import json,sys
for pkg in json.load(sys.stdin)["packages"]:
    print(pkg["name"], " ".join(sorted(pkg["features"])))' || true)"

FEATURE_ARGS=()
feature_args_for() {
  local PKG="$1"
  FEATURE_ARGS=()
  [ -n "${PS_ISOLATION_FEATURES:-}" ] || return 0

  local declared kept=() dropped=() feature
  declared=" $(printf '%s\n' "${DECLARED_FEATURES}" | awk -v p="${PKG}" '$1 == p {$1=""; print}') "
  for feature in ${PS_ISOLATION_FEATURES//,/ }; do
    if [ "${declared}" != "  " ] && [[ "${declared}" != *" ${feature} "* ]]; then
      dropped+=("${feature}")
    else
      kept+=("${feature}")
    fi
  done
  if [ ${#dropped[@]} -gt 0 ]; then
    echo "note: ${PKG} declares no ${dropped[*]}; checking its own graph instead" >&2
  fi
  if [ ${#kept[@]} -gt 0 ]; then
    FEATURE_ARGS=(--features "$(IFS=,; echo "${kept[*]}")")
  fi
}

if [ "$#" -gt 0 ]; then
  PACKAGES=("$@")
else
  PACKAGES=(
    partial-stateless-validator
    partial-stateless-stream
    partial-stateless-replay
    partial-stateless-frontier
    partial-stateless-exex
  )
fi

# Implementations, not traits. reth-storage-api and reth-storage-errors are deliberately absent.
FORBIDDEN='^(reth-provider|reth-db|reth-db-common|reth-libmdbx|reth-mdbx-sys|reth-exex|reth-node-builder)$'
REQUIRED_FEATURES='feature "(asm-keccak|keccak-cache-global)"'
REQUIRED_RECOVERY='feature "secp256k1"'
REQUIRED_TRIE_PARALLELISM='reth-trie-sparse feature "std"'

status=0

check_package() {
  local PKG="$1"

# Stage A cannot speak about the ExEx: invariant 1 forbids a provider and an ExEx *in a node*
# necessarily links one, and the features it resolves are the node's rather than its own -- which
# is exactly why "it happens to be right" has held there by accident three times. Invariant 5 can,
# and is the only thing that ever has.
if [ "${PKG}" = "partial-stateless-exex" ]; then
  echo "==> ${PKG}: dependency graph not checked (an ExEx in a node necessarily links a provider)"
  return
fi

feature_args_for "${PKG}"

echo "==> ${PKG}: normal dependency graph"
if ! deps="$(cargo tree -p "${PKG}" -e normal --prefix none "${FEATURE_ARGS[@]}" 2>/dev/null)"; then
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
edges="$(cargo tree -p "${PKG}" -e normal,build,features -i alloy-primitives "${FEATURE_ARGS[@]}" 2>/dev/null \
  | grep -cE "${REQUIRED_FEATURES}" || true)"
if [ "${edges}" -eq 0 ]; then
  echo "FAIL: ${PKG} selects neither asm-keccak nor keccak-cache-global on alloy-primitives." >&2
  echo "      Any timing measured from this build describes a keccak production does not run." >&2
  status=1
else
  echo "ok: ${edges} asm-keccak/keccak-cache-global feature edges on alloy-primitives"
fi

echo "==> ${PKG}: signature recovery backend"
recovery="$(cargo tree -p "${PKG}" -e normal,build,features -i reth-primitives-traits "${FEATURE_ARGS[@]}" 2>/dev/null \
  | grep -cE "${REQUIRED_RECOVERY}" || true)"
if [ "${recovery}" -eq 0 ]; then
  echo "FAIL: ${PKG} does not select secp256k1 on reth-primitives-traits." >&2
  echo "      Sender recovery would fall back to k256, which production does not run." >&2
  status=1
else
  echo "ok: ${recovery} secp256k1 feature edges on reth-primitives-traits"
fi

echo "==> ${PKG}: sparse trie parallelism"
parallelism="$(cargo tree -p "${PKG}" -e normal,build,features -i reth-trie-sparse "${FEATURE_ARGS[@]}" 2>/dev/null \
  | grep -cE "${REQUIRED_TRIE_PARALLELISM}" || true)"
if [ "${parallelism}" -eq 0 ]; then
  echo "FAIL: ${PKG} does not select std on reth-trie-sparse." >&2
  echo "      Proof reveal and subtrie hash updates would run serially, which production does not." >&2
  status=1
else
  echo "ok: ${parallelism} std feature edges on reth-trie-sparse"
fi
}

# Invariant 5: the artifact. `partial-stateless-validator` and `partial-stateless-stream` are
# libraries with no binary to check, and are absent on purpose -- their code ships inside the
# binaries below, so a defect in their build shows up in one of these rows.
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HEAD_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo "")"

binary_for() {
  case "$1" in
    partial-stateless-replay) echo "${REPO_ROOT}/target/release/ps-replay" ;;
    partial-stateless-frontier) echo "${REPO_ROOT}/target/release/ps-policy-frontier" ;;
    partial-stateless-exex) echo "${REPO_ROOT}/target/release/reth-partial-stateless" ;;
    *) echo "" ;;
  esac
}

check_binary() {
  local PKG BIN
  PKG="$1"
  BIN="$(binary_for "${PKG}")"
  [ -n "${BIN}" ] || return 0

  echo "==> ${PKG}: artifact (${BIN##*/})"
  if [ ! -x "${BIN}" ]; then
    if [ "${PS_ISOLATION_REQUIRE_BINARY:-0}" = "1" ]; then
      echo "FAIL: ${BIN} is missing; the graph above describes the source, nothing describes the artifact." >&2
      status=1
    else
      echo "warn: ${BIN} not built; the graph above describes the source, not any artifact" >&2
    fi
    return
  fi

  if [ -z "${HEAD_COMMIT}" ]; then
    echo "warn: not a git checkout; the artifact cannot be tied to a commit" >&2
    return
  fi

  # `grep -a` because the binary is not text. The commit is embedded by `option_env!` at compile
  # time, so this asks one question with one answer: does the artifact carry the commit whose graph
  # was just checked? A binary that does not is either stale or was built without the provenance
  # exports, and neither can be shown to be the code above -- so both fail, and the message names
  # both. There is no third state to detect: "no stamp at all" is not distinguishable by grep,
  # because a stripped binary contains 40-hex-digit byte sequences by chance.
  if grep -qa "${HEAD_COMMIT}" "${BIN}"; then
    echo "ok: the artifact was built from this commit (${HEAD_COMMIT:0:12})"
  else
    echo "FAIL: ${BIN} does not carry ${HEAD_COMMIT:0:12}." >&2
    echo "      Either it is stale, or it was built without PS_BUILD_COMMIT exported. Either way" >&2
    echo "      it is not the code the graph check above passed on. Rebuild with the three" >&2
    echo "      provenance exports before measuring." >&2
    status=1
  fi
}

for pkg in "${PACKAGES[@]}"; do
  check_package "${pkg}"
  check_binary "${pkg}"
done

exit "${status}"
