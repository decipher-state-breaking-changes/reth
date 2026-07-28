# partial-stateless-exex

A reth [Execution Extension (ExEx)](../exex/exex) that drives the
[`partial-stateless`](../partial-stateless) library from a live node. It maintains
the network-level state cache as the chain advances and, per block, measures the
witness ("sidecar") a partially-stateless validator would need.

The binary is `reth-partial-stateless` — a full Ethereum node with the ExEx
installed.

## What it does per committed block

1. Re-executes the block against its parent state (`history_by_block_number`) and
   captures the `BlockAccessedState` (accounts, storage, bytecodes touched).
2. Computes the **cache miss** *before* updating the cache — this is what a
   validator joining at this block would have to be sent.
3. Applies the tentative `NetworkStateCache` transition (including `LastNBlocksPolicy` eviction),
   retaining a rollback record until the sparse-trie transition and sidecar checks succeed.
4. Generates a native V2 parent-state proof for cache misses plus execution-diff paths not already
   authenticated by the trie cache. It reveals that proof into one transactional sparse-trie
   session, fetches only newly discovered structural proof deltas, and resumes the same transition.
   The resulting flat, hash-deduplicated node witness is written with a JSON benchmark
   **manifest** to `./sidecar/`. Structural targets do not change the cache-miss manifest.
5. *(optional)* Runs the **provider-assisted sidecar preflight** — re-executes
   the block through a cache+witness-backed provider and checks the miss set plus
   cache-anchor transition.
6. *(optional)* Computes the **full-witness baseline** — a second multiproof over
   *all* accessed state, ignoring the cache — to report the reduction ratio.
7. Logs accessed/missed counts, miss ratio, witness size, and cache footprint.

The parent-state proof is revealed into a cloned local sparse trie. Storage and account
changes are applied locally and the computed post-state root is checked against the block header.
The tentative flat-cache membership produced in step 3 is then mirrored into the sparse trie:
inclusion paths are retained for existing values, while zero and nonexistent values retain the
terminal exclusion node. Unrelated decoded subtrees are blinded and an account's storage trie is
removed after its final cached slot expires. On failure, the value transition is rolled back and
the cloned trie is discarded. The sidecar carries no post-state proof.

The flat `NetworkStateCache` alone decides hits, misses, eviction, and cache anchors. Sparse-trie
shape is local validation state: additional revealed nodes do not change the sidecar miss manifest
or either cache anchor.

Sparse-trie snapshots currently have no branch-aware undo representation. On
`ChainReorged` and `ChainReverted`, both flat and trie caches are cold-reset so a
flat value cannot outlive its authenticated path. A builder can initialize from
the full provider while processing the new branch. A sidecar-only verifier cannot
recover from that cold reset without a synchronized joint cache snapshot or a
future bootstrap protocol.

## Run

```bash
cargo run -p partial-stateless-exex -- node --chain mainnet --datadir /path/to/data
```

The flat cache is persisted to `<datadir>/partial_stateless_cache.bin`, but the
matching sparse-trie snapshot is not yet persisted. A non-empty persisted value
cache is therefore cold-reset on restart.

### Configuration

The cache windows are set in `CacheConfig` ([main.rs](./src/main.rs)) — default
`account_window = 60`, `storage_window = 30` blocks. Adjust there and rebuild.
(Use [`cache_window_bench`](../partial-stateless/src/bin/README.md) to pick good
values offline before committing to them.)

Optional diagnostic/benchmark features are off by default and enabled per run via environment
variables, so the core sidecar generation path stays lean:

| Env var | Effect |
| --- | --- |
| `PS_SIDECAR_ROLE=builder\|builder-verifier\|verifier` | choose whether this ExEx writes sidecars, writes and preflights them, or consumes existing sidecars as a live verifier (default: `builder`) |
| `PS_SIDECAR_DIR=<dir>` | write sidecars in `<dir>` (default: `./sidecar`) |
| `PS_SIDECAR_VERIFIER_WAIT_MS=<ms>` | in `verifier` mode, wait up to this long for the block sidecar file to appear (default: `2000`) |
| `PS_CAPTURE_DIR=<dir>` | dump each block's `BlockAccessedState` fixture to `<dir>` (see below) |
| `PS_WITNESS_BASELINE=1` | also compute the full-witness baseline + reduction ratio (an extra, larger multiproof per block) |
| `PS_RESOURCE_METRICS=1` | capture per-thread CPU time + page faults around transition-witness construction (`cpu_time_ms`, `major_page_faults`, `minor_page_faults`) to separate compute-bound from disk-I/O-bound blocks |
| `PS_ENGINE_BENCH=1` | enable the lightweight Vanilla Engine V2 timing collector; usable by a standard Reth node without the ExEx |
| `PS_ENGINE_BENCH_OUTPUT=<file>` | JSONL destination for Vanilla Engine V2 timing records (default: `./engine_bench.jsonl`) |
| `PS_VALIDATION_BENCH=1` | enable in-memory DB-free Partial/Weak validation paired with same-block Vanilla Engine timing; requires `builder-verifier` |
| `PS_BENCH_OUTPUT=<file>` | JSONL destination for paired Partial/Weak benchmark records |
| `PS_TRIE_CACHE_DIAGNOSTICS=1` | validate retained account/storage paths and log trie shape, memory, and transition timings |

`PS_SIDECAR_ROLE=builder-verifier` is a single-process test mode: it keeps the
normal builder output path, but forces the same provider-assisted client preflight
before publishing each sidecar. Use this mode to observe cache-miss-only,
witness-integrity, state-root, and next-cache-anchor failures while the builder is
running.

`PS_SIDECAR_ROLE=verifier` is the live verifier mode. It does not build or publish
sidecars. For each canonical block it reads
`$PS_SIDECAR_DIR/block_<N>_<hash>.bin`, verifies it against the local previous
cache, re-executes with cache hits plus sidecar miss witnesses, and advances the
local cache only after verification succeeds. The verifier must start with a
cache synchronized to the parent block; the sidecar file alone is not enough to
reconstruct that previous cache. Because sparse-trie snapshots are not persisted,
the current binary cold-resets a persisted flat cache at startup; ordinary
mid-chain verifier restart/cold-start is therefore not implemented yet.

Preflight re-executes from cache hits plus sidecar misses, applies the execution
diff to a cloned local sparse trie, checks that root against the consensus block
root, and then cross-checks it with the full provider. It also verifies the miss
set and value-cache next anchor.

The manifest and verifier logs expose
`partial_state_trustless_verification_ready`. The readiness calculation includes
miss paths from the sidecar and cache-hit paths retained by the local sparse trie.

When the first processed block is not synchronized to the cache parent, the
builder obtains a local-only proof for the union of captured access paths and
execution-diff paths. It uses that proof to initialize both local caches and does
not publish a cache-coherent sidecar for that block. This is local ExEx startup,
not a protocol bootstrap mechanism for a new stateless node. Cold-EOA mempool
admission and new-node cache bootstrap remain out of scope.

When `PS_WITNESS_BASELINE` is unset, the manifest's `full_sidecar_baseline_stats`
and `reduction` are `null` and no baseline multiproof is computed. A baseline
failure is non-fatal — it never blocks the real (partial) sidecar.

When `PS_RESOURCE_METRICS` is unset, the partial stats' `cpu_time_ms`,
`major_page_faults`, and `minor_page_faults` are `null` and no `getrusage`
syscalls are made. The metrics are Linux-only (`RUSAGE_THREAD`); on other
platforms they log zeros. If comparing against the baseline, note that
`PS_WITNESS_BASELINE` runs first and can warm the OS page cache, deflating the
partial witness page-fault counts.

### Single-process paired execution benchmark

`scripts/run_live_paired_bench.py` supervises one `reth-partial-stateless` process. For each
canonical block, Engine V2 first records production Full-DB state access plus EVM execution. The
ExEx then builds the two witnesses and re-executes the same block through the DB-free Partial and
Weak providers. Partial-first and Weak-first order alternates by block.

The primary `state_access_execution_us` boundary is:

- Vanilla: parent provider construction, production prewarming/cache setup, DB-backed reads, and
  EVM execution.
- Partial/Weak: sidecar deserialize, context and witness commitment/self-consistency checks,
  materialization, witness-backed provider setup/lookups, and EVM execution.

Post-execution access capture, hashing/root work, cache maintenance, builder proof generation, file
I/O, and network transfer are outside the primary metric and are reported separately where
applicable. The secondary executor-call metric still includes state-provider reads made by the EVM:
Full DB/cache reads for Vanilla and in-memory witness/cache lookups for Partial and Weak. It excludes
Partial/Weak deserialize, cache-context validation, commitment checks, and materialization.

If builder-side preflight fails, the process remains fail-closed, but writes a reproducible diagnostic
bundle below `$PS_SIDECAR_DIR/preflight-failures/`. The bundle contains the exact sidecar, parent
value cache, a self-contained proof for all retained parent cache paths, and JSON metadata.

Benchmark mode serializes generated sidecars in memory for the current block only, does not persist
them or the cache, skips
warn-only root-completeness scans, and ignores capture, full-witness-baseline, resource, and trie
diagnostics flags. Production behavior is unchanged when `PS_VALIDATION_BENCH` is unset.

The default run excludes 60 successful warm-up blocks and collects 600 same-hash accepted samples.
It discards invalid pairs and any pair whose Partial/Weak interval overlaps the start of the next
Engine validation. A reorg/revert cold-resets the caches and restarts the warm-up; only samples after
the last reset are retained. The supervisor sends `SIGINT` after the target and writes `results.md`.
The output directory must be absent or empty.

```bash
python3 crates/partial-stateless-exex/scripts/run_live_paired_bench.py \
  --reth-bin ./target/release/reth-partial-stateless \
  --datadir /path/to/reth-data \
  --jwtsecret /path/to/jwt.hex \
  --output /path/to/benchmark-output \
  --warmup 60 \
  --samples 600 \
  -- \
  --minimal
```

The script only starts the node when the user invokes it; building or testing this crate does not
start a node. Raw records and logs are saved next to the report as `paired.jsonl`, `engine.jsonl`,
`resources.jsonl`, and `reth-partial-stateless.log`. The resource log samples process RSS and peak
RSS on each supervisor poll so long-run allocator regressions and OOM kills remain diagnosable.

### Transition-witness construction

The initial builder target set is the union of value-cache misses and mutation paths not already
authenticated by the persistent trie cache. It requests that set once through
`StateProofProvider::multiproof_v2`. Native V2 proof generation builds targeted storage proofs
first and reuses their roots when encoding account leaves, avoiding a second traversal of those
storage tries when a full storage proof is available.

The proof is revealed into one transactional sparse-trie clone. A deletion can expose a blinded
sibling or extension child whose node kind is needed for canonical branch compression. In that
case the transition reports all currently visible `(key, min_len)` targets, the builder subtracts
targets already requested, fetches only the delta, and resumes the unfinished session. There is no
full proof regeneration or transition replay. An empty delta is rejected as no progress, and a
128-round cap guards malformed or unexpectedly deep chains of structural dependencies.

Production sidecars use only `MptTransitionNodes`: deterministic, hash-deduplicated parent-state
RLP node preimages. Because this flat format carries no account/storage path context, storage
targets also include the account path needed to reconnect their storage root to the state root.
Legacy `MptMultiProof` sidecars remain decodable for compatibility.

For benchmark logs, `initial_provider_us` is the initial native V2 provider call,
`structural_provider_us` is the sum of later context/structural provider calls, and
`provider_calls` counts both. `partial_sidecar_stats.computation_time_ms` covers the initial proof,
transactional trie clone, proof deltas, transition, root check, and flat-witness decoding. It
excludes cache retention, optional trie-cache validation, and optional sidecar preflight, which
have separate timings or logs.

### Trie-shape diagnostics

`PS_TRIE_CACHE_DIAGNOSTICS=1` performs an O(retained paths) scan after each
builder transition. It checks exact flat/trie membership, a complete inclusion or
exclusion witness for every retained account and storage path (including zero and
nonexistent values), and equality of the recomputed sparse root and recorded
post-state root. Successful blocks log clone, update, retention, and validation
timings; memory; decoded account/storage node counts; and hashed-key prefix
coverage at depths zero through five.

The clone timing covers the single transactional trie-cache clone. The local-root timing covers the
resumable transition, including waits for any structural proof deltas. Retention is normal
per-block cache work. Full validation is diagnostic-only and is skipped when
`PS_TRIE_CACHE_DIAGNOSTICS` is unset.

Use `PS_SIDECAR_ROLE=builder-verifier` for a bounded correctness run.
Do not interpret prefix coverage as a literal MPT node count: Patricia extensions
compress nibble levels.

### Capturing a benchmark dataset

Set `PS_CAPTURE_DIR` to dump each block's `BlockAccessedState` as a fixture. This
reuses the exact execution path the live system uses, so the dataset is faithful —
and once captured, the offline `cache_window_bench` needs no node at all.

```bash
PS_CAPTURE_DIR=./fixtures/accessed \
    cargo run -p partial-stateless-exex -- node --chain mainnet --datadir /path/to/data
# let it run until ~300 accessed_*.bin files exist, then stop
```

Re-injecting *raw blocks* would not be reproducible — re-execution needs the parent
historical state present in the node DB at that exact height. The accessed-state
snapshot is the portable, self-contained artifact.

## Outputs

| Path | Contents |
| --- | --- |
| `<datadir>/partial_stateless_cache.bin` | persisted flat cache; cold-reset on restart until trie snapshots are persisted |
| `./sidecar/block_<N>_<hash>.bin` | witness sidecar (or `$PS_SIDECAR_DIR/block_<N>_<hash>.bin`) |
| `./sidecar/block_<N>_<hash>.manifest.json` | per-block benchmark manifest |
| `$PS_CAPTURE_DIR/accessed_<N>.bin` | captured fixture (when capture is enabled) |
