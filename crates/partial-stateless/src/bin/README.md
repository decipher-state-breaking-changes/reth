# `partial-stateless` binaries

Standalone command-line tools that operate on the artifacts produced by the
[`partial-stateless-exex`](../../../partial-stateless-exex). Each is a thin CLI
over the library in [`../`](../) — none of them needs a running node, so they are
cheap to run in CI or on a laptop with just the files the ExEx wrote to disk.

| Binary | Input | Answers |
| --- | --- | --- |
| [`sidecar_verifier`](./sidecar_verifier.rs) | `sidecar/*.bin` (+ optional reference witnesses) | Is the witness sidecar cryptographically anchored to the parent state root, and how complete is it? |
| [`cache_window_bench`](./cache_window_bench.rs) | `fixtures/accessed/*.bin` | How does the cache window trade off against hit ratio, and which state category is the hot-spot? |
| [`trie_repr_probe`](./trie_repr_probe.rs) | a recorded policy replay dataset | How many live bytes and how much clone time does each sparse-trie representation cost for the same revealed witness, measured through one counting allocator? |
| [`locality_scan`](./locality_scan.rs) | a recorded policy replay dataset | How much of what a block touches did the last `N` blocks already touch, by key count and by payload bytes, for accounts, slots and codes separately? |

Run any of them with `cargo run -p partial-stateless --bin <name> -- <args>`.
`trie_repr_probe` additionally needs `--features repr-probe`, which is what pulls in the arena
trie implementation it compares against; the library itself never enables it.

`cache_window_bench` and `locality_scan` overlap in subject and not in purpose. The bench sweeps a
window *grid* through a real `NetworkStateCache` and reports hit ratio and footprint per config,
which is how a window gets chosen. The scan makes one pass over the access sets and reports the
whole reuse-distance distribution, from which coverage at *every* window falls out at once — which
is how a chosen window gets explained. Use the bench to pick, the scan to justify.

---

## `sidecar_verifier`

Verifies the witness sidecars the ExEx emits, without trusting the producer or
the transport. Two independent checks:

1. **Crypto integrity** (default) — every missed account proof verifies against
   `parent_state_root`, every missed storage proof against its account
   `storageRoot`, and every supplied bytecode preimage hashes to a declared
   missed code hash. Proves the material that *is* present is anchored to the
   parent state root. Does **not** prove completeness.
2. **Coverage** (`--coverage`) — compares the sidecar against a reference
   `debug_executionWitness` (ground truth from a full node) for the same block,
   reporting how much of the state actually needed to re-execute is present.

```bash
# Crypto integrity over a directory of sidecars
cargo run -p partial-stateless --bin sidecar_verifier -- ./sidecar

# Coverage against reference witnesses (one witness_<N>.json per block N)
cargo run -p partial-stateless --bin sidecar_verifier -- \
    --coverage --witness-dir ./witnesses ./sidecar
```

Arguments:

| Flag | Meaning |
| --- | --- |
| `<path-or-dir> ...` | one or more sidecar `.bin` files or directories of them |
| `--coverage` | enable coverage mode (requires `--witness-dir`) |
| `--witness-dir <dir>` | dir of `witness_<block>.json` reference witnesses |

Neither mode re-executes the block; full re-execution (`ACCEPT_BLOCK`) is a
follow-up. Exit code is non-zero if any sidecar fails its check.

---

## `cache_window_bench`

Sweeps the cache eviction window against hit ratio over a **fixed, captured**
dataset, fully offline (no node, no EVM, no Merkle proofs). It replays the
per-block `BlockAccessedState` fixtures through a fresh `NetworkStateCache` for
every `(account_window, storage_window)` in a grid and reports hit ratio
(overall + per-category), cache footprint, and a hot-spot breakdown.

First capture a dataset once, by running the ExEx with `PS_CAPTURE_DIR` set (see
the [ExEx README](../../../partial-stateless-exex/README.md)). Then:

```bash
# Default 6×6 grid over ./fixtures/accessed
cargo run -p partial-stateless --bin cache_window_bench -- --fixtures ./fixtures/accessed

# Custom grid + baseline for the hot-spot breakdown
cargo run -p partial-stateless --bin cache_window_bench -- \
    --fixtures ./fixtures/accessed \
    --account-windows 30,60,90,128 --storage-windows 8,30,60 \
    --baseline 60,60 --out bench.csv
```

Arguments:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--fixtures <dir>` | `./fixtures/accessed` | directory of `accessed_*.bin` fixtures |
| `--account-windows a,b,c` | `8,16,30,60,90,128` | account `LastN` windows to sweep |
| `--storage-windows a,b,c` | `8,16,30,60,90,128` | storage/code `LastN` windows to sweep |
| `--warmup <N>` | largest window in the grid | blocks to warm before scoring |
| `--baseline aw,sw` | `60,60` | config used for the hot-spot breakdown |
| `--out <path>` | `./cache_window_bench.csv` | CSV output |

**Fairness note:** the cache is warmed over the first `--warmup` blocks (still
updated, but not scored) so every config is measured over the *same* fully-warmed
block range. Default warmup = the largest window, so even the widest window is
warm before scoring begins.

`LastNBlocksPolicy` is a *time* window (blocks), not a size cap — so "cache size"
is an emergent output reported per config, not an input you pin.

To smoke-test without a node, generate synthetic fixtures with the example:

```bash
cargo run -p partial-stateless --example gen_synth_fixtures -- /tmp/fix 300
cargo run -p partial-stateless --bin cache_window_bench -- --fixtures /tmp/fix
```

---

## `locality_scan`

Answers one question over a recorded policy replay dataset: **how much of what a block touches did
the last `N` blocks already touch?** Separately for accounts, storage slots and bytecodes, at every
`N` up to a bound, weighted both by key count and by payload bytes.

Fully offline and cheap — no EVM, no Merkle proof work, no node. One pass over the corpus, about
36 s for 1,200 records on a SATA SSD.

```bash
cargo run --release -p partial-stateless --bin locality_scan -- \
    --dataset /path/to/raw-block-witness-data \
    --warmup 120 --samples 1000 --max-window 120 \
    --out ./locality-out
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--dataset <dir>` | required | a recorded policy replay dataset (the directory holding `manifest.json`, `END.json` and `blocks/`) |
| `--warmup <n>` | required | leading canonical blocks that populate the history but are not scored |
| `--samples <n>` | required | measured blocks, taken immediately after the warm-up |
| `--max-window <n>` | `120` | widest window the curves report; warns if it reaches past `--warmup` (see below) |
| `--out <dir>` | required | output directory |

Three files come out. `locality-curve.csv` is the flat curve, one row per `N`, with coverage by key
count and by payload bytes for each category and for all three pooled. `locality-summary.json`
carries the full reuse-distance histograms, the totals, and the run's identity.
`locality-per-block.jsonl` carries per-block hit counts at a ladder of windows, which is what makes
the result checkable against a frontier run rather than merely plausible.

### The window is `N + 1` heights, not `N`

This is the one thing to get right before quoting a number from this tool.
[`LastNBlocksPolicy`](../policy.rs) retains an entry whose `last_accessed_block` is at or above
`current_block - window_size`, and eviction runs with the block that was *just applied*. So a cache
about to look up block `B` has applied through `B - 1` and holds the **closed** range
`[B - 1 - N, B - 1]` — `N + 1` distinct heights. [`readiness.rs`](../readiness.rs) says the same
thing in `required_replay_depth`, which is `window_size + 1`.

Coverage at window parameter `N` therefore counts a key whose reuse distance `d` satisfies
`d <= N + 1`, and that is what this tool computes. Recompute it as a union over `N` prior blocks
instead and the result is *not* the complement of a generator run's miss counts.

The same `+ 1` decides how much warm-up a curve needs. `--max-window N` reaches `N + 1` heights
back, so a warm-up of `N` leaves the first measured block one height short and biases the widest
windows low *at that block*. The tool warns rather than refuses, because the boundary case is the
one that actually gets run — holding a measured block set fixed pins the warm-up, and asking for
the full window then costs exactly one height at exactly one block:

```
locality_scan: WARNING --max-window 120 reaches 121 heights back but the warm-up supplies only
120; the first 1 measured block(s) are scored against an incomplete history and the widest
windows are biased low there. Reduce --max-window to 119 to remove the effect, or report the
shortfall.
```

Take one of the two exits it offers. Either drop to `--max-window 119` and the curve is clean, or
keep 120 and report the shortfall — its size is not recoverable from the data, since the missing
height is below the corpus floor, so what can be stated is a worst case: every miss at that block
could have been covered by the absent height.

### Checking a run against a frontier run

Because `ps-policy-frontier` reports `accessed_*` and `missed_*` per block per arm, and this
reports hits per block per window, the two are the same quantity and must agree exactly. The ladder
in `locality-per-block.jsonl` deliberately includes 30, 45, 60, 90 and 120 so that the arms a
frontier run typically uses can be checked directly:

```
hit_accounts[N=aw] == accessed_accounts - missed_accounts   (arm aw/sw)
hit_storage [N=sw] == accessed_storage  - missed_storage
hit_codes   [N=sw] == accessed_codes    - missed_codes
```

`measured_block_set_digest` in the summary uses the same construction the frontier's summary does,
so two runs that agree on it measured the same blocks. Check that first; the per-block equalities
mean nothing if the block sets differ.

### What it does and does not measure

The unit is a unique `(block, key)` **access-set incidence**. The input is the per-block
`BlockAccessedState`, whose fields are `HashMap`s, so a key touched forty times inside one block
counts once. This is inter-block locality — the only kind `LastNBlocksPolicy` can act on, since it
evicts on `last_accessed_block` — and it says nothing about per-transaction locality, transaction
ordering, or how hot a key is within the block that touches it.

Byte weighting is a **logical key-plus-value payload size** defined by this tool:
`20+8+32(+32)` per account, `20+32+32` per slot, `32+len` per bytecode. It is not a witness size,
not a proof-path cost, and deliberately not the cache's own `estimated_memory_bytes` accounting
(`112` / `104` / `52+len`). Accounts and slots are near-fixed-width, so their byte curves track
their key curves; bytecode is where the two diverge, which is the point of having both.

The overflow bucket is named `no_reuse_within_*` rather than "censored" because it is a mixture:
keys whose observed reuse distance exceeds the histogram, and keys with no earlier access anywhere
in the observed prefix. Only the second is unresolvable. Do not read it as a first-touch rate.

### Memory

Unlike the other dataset consumers this does **not** call `load_dataset`, which holds every record
at once and needs more memory than the corpus is large. Records are processed singly and reduced to
their access set immediately. The trade is that it verifies a strict subset of `load_dataset`'s
checks — per-record seal digest and schema, plus record count and the canonical `parent_hash` walk,
but not the manifest schema, the lifecycle log, the terminator's range cross-check or the
confirmation-depth claim. Point it at a corpus a generator has already accepted; use `load_dataset`
to vet a new one.
