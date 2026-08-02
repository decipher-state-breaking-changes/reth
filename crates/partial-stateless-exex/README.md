# partial-stateless-exex

A reth [Execution Extension (ExEx)](../exex/exex) that drives the
[`partial-stateless`](../partial-stateless) library from a live node. It maintains
the network-level state cache as the chain advances and, per block, measures the
witness ("sidecar") a partially-stateless validator would need.

The binary is `reth-partial-stateless` — a full Ethereum node with the ExEx
installed. The crate is split into a library and a thin binary so that the
recovery, bootstrap, and admission paths can be tested from `tests/`: each of
those needs a state provider and an EVM, which `partial-stateless` deliberately
does not depend on.

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

## Reaching a usable cache

Sparse-trie snapshots still have no branch-aware undo representation, so nothing
here unwinds. Instead there is one primitive — a **provider-backed canonical
rebuild** ([`rebuild.rs`](./src/rebuild.rs)) — that produces an exact coordinated
pair at a given canonical block hash, from nothing, without consulting whatever
the caches held before. It replays `max_window + 1` heights ending at that block
to rebuild the flat cache, then authenticates the trie in one shot against the
block's canonical state root. Because it never reads the previous generation, the
same function serves a cold start, a reorg, and a revert alike; for a reorg,
replaying forward from the cached generation would be *incorrect* rather than
merely slow, since that generation sits on the abandoned branch.

| Situation | What happens |
| --- | --- |
| Cold start | Rebuild at the parent of the first notified block, then publish from its first child. Roughly `window + 1` historical executions and one multiproof, against about twelve minutes of live warming. |
| `ChainReorged` | Enter `Recovering`, rebuild at the common ancestor — the parent hash of the new chain's first block — then apply the new blocks through the normal path so the builder still produces a sidecar for each. |
| `ChainReverted` | Enter `Recovering` and rebuild at the new tip, addressed as the parent hash of the reverted chain's first block. |
| Gap or wrong-branch parent | Try a rebuild at the rejected block's own parent first; only fall back to a cold reset and live warming if that fails. |
| Rebuild unavailable or failing | Log it and warm from live blocks. A failed recovery is logged as such rather than sharing a code path with a clean cold start, and three consecutive failures stop further attempts for the run. |

The rebuild requires canonical state, so it does not replace the snapshot path: a
full node cold-starts by replay and needs no snapshot file, while a node without
the database cannot replay at all and needs one. It also assumes the last
`max_window + 1` heights are readable through `history_by_block_hash`, which holds
for a full node but not under aggressive pruning.

**No part of this is reachable from the measured verification path.** The
validator numbers only mean anything because the benchmark validates from
serialized sidecar bytes against the cache, trie cache, and witness alone. The
rebuild is cache *maintenance*, driven by the side that holds the database, and
runs between measured samples rather than inside one.

## Operator-trusted snapshot bootstrap

[`bootstrap_io.rs`](./src/bootstrap_io.rs) exports and imports the joint cache
snapshot the library can already build, verify, and restore. The importing side
authenticates everything against a `TrustedCheckpoint` — number, hash, canonical
state root, cache root, policy ID — that the operator supplies out of band, and
discards a package that disagrees with it. **A node bootstrapped this way trusts
whoever configured the checkpoint; this is not trustless new-node sync.**

`PS_BOOTSTRAP_SELF_TEST=<n>` closes the sync/bootstrap gate inside one process,
because two live runs cannot overlap on one datadir and sequencing them lets the
chain advance across the restart. The run warms normally, exports at `Ready(H)`,
restores a *second* coordinated pair from that package in the same process, and
then validates the next `n` blocks against both pairs through the same
provider-free path — asserting they agree on cache anchor, trie state root, trie
cache root, and retained paths. Miss-set agreement is structural rather than a
separate assertion: that verification path already checks the restored cache's
own expected miss set against the miss manifest the live pair built.

An imported snapshot is stale by the time the first notification arrives. A node
that can replay bridges the drift with a canonical rebuild; a node that cannot
stays Cold until a fresher snapshot is supplied. That is a real limitation of this
phase, which is why the gate above restores in-process rather than across a
restart.

## Run

```bash
cargo run -p partial-stateless-exex -- node --chain mainnet --datadir /path/to/data
```

The flat cache is persisted to `<datadir>/partial_stateless_cache.bin`, but the
matching sparse-trie snapshot is not yet persisted, so a non-empty persisted
value cache is still cold-reset on restart. A full node no longer pays for that:
the canonical rebuild above puts it back at `Ready` before the first notified
block is applied. Atomic value+trie+anchor persistence remains the thing that
would make a warm restart real.

### Configuration

The cache windows are set in `CacheConfig` ([lib.rs](./src/lib.rs)) — default
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
| `PS_PARALLEL_INITIAL_PROOF=1` | use Reth's proof workers for eligible initial V2 multiproofs; low-width target sets and later structural deltas stay serial |
| `PS_RESOURCE_METRICS=1` | capture process CPU time + page faults around transition-witness construction, including parallel proof workers (`cpu_time_ms`, `major_page_faults`, `minor_page_faults`) |
| `PS_ENGINE_BENCH=1` | enable the lightweight Vanilla Engine V2 timing collector; usable by a standard Reth node without the ExEx |
| `PS_ENGINE_BENCH_OUTPUT=<file>` | JSONL destination for Vanilla Engine V2 timing records (default: `./engine_bench.jsonl`) |
| `PS_VALIDATION_BENCH=1` | enable in-memory DB-free Partial/Weak validation paired with same-block Vanilla Engine timing; requires `builder-verifier` |
| `PS_BENCH_OUTPUT=<file>` | JSONL destination for paired Partial/Weak benchmark records |
| `PS_BUILDER_BENCH_OUTPUT=<file>` | JSONL destination for per-block builder proof, snapshot, commitment, and total-cost records |
| `PS_FORCE_PREVIOUS_CACHE_SNAPSHOT=1` | benchmark-only B2 control that recreates the old unconditional parent-cache clone |
| `PS_TRIE_CACHE_DIAGNOSTICS=1` | validate retained account/storage paths and log trie shape, memory, and transition timings |
| `PS_CANONICAL_REBUILD=0` | disable the provider-backed canonical rebuild and fall back to warming from live blocks (default: enabled) |
| `PS_BOOTSTRAP_DIR=<dir>` | where the snapshot package and its checkpoint live (default: `$PS_SIDECAR_DIR/bootstrap`) |
| `PS_BOOTSTRAP_EXPORT=1` | export a snapshot the first time the tracker reaches Ready |
| `PS_BOOTSTRAP_IMPORT=1` | restore from a snapshot at startup, ahead of the persisted flat cache |
| `PS_BOOTSTRAP_SELF_TEST=<n>` | export at the first Ready, restore a second pair in-process, and compare both pairs for the next `n` blocks (implies export) |

The initial parallel-proof gate currently requires at least two distinct storage tries and 64
total initial targets. Eligible one-shot calls use one account worker and a workload-bounded number
of storage workers; smaller calls and all later structural/context proof deltas stay on the serial
provider.

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
reconstruct that previous cache. A snapshot import (`PS_BOOTSTRAP_IMPORT=1`) is
how a verifier gets one, but an imported snapshot is stale by the time the first
notification arrives, and a verifier cannot replay to bridge the drift. Ordinary
mid-chain verifier restart therefore still needs a fresh snapshot per start.

Preflight re-executes from cache hits plus sidecar misses, applies the execution
diff to a cloned local sparse trie, checks that root against the consensus block
root, and then cross-checks it with the full provider. It also verifies the miss
set and value-cache next anchor.

The manifest and verifier logs expose
`partial_state_trustless_verification_ready`. The readiness calculation includes
miss paths from the sidecar and cache-hit paths retained by the local sparse trie.

## Cold-EOA admission

`partial_stateless::admit_cold_sender` turns a verified account proof into an
admission decision: it applies the cold precondition (the sender must be absent
from the account cache *at the current head*), verifies the proof, and then
applies balance, nonce, and EIP-3607 rules. It takes canonicality as a closure,
which [`cold_eoa.rs`](./src/cold_eoa.rs) supplies from two header reads. Keeping
the two apart is what makes the no-state-access property structural: the crate
holding the admission logic does not depend on `reth-provider` at all.

This is a caller-level path, not a transaction-pool integration. Reth's pooled
transaction type carries no proof field, so a real pool integration forces a
custom transaction type; that, along with p2p/RPC proof distribution, relay and
caching, DoS accounting, and pending-head/reorg behaviour, is deferred.

When `PS_WITNESS_BASELINE` is unset, the manifest's `full_sidecar_baseline_stats`
and `reduction` are `null` and no baseline multiproof is computed. A baseline
failure is non-fatal — it never blocks the real (partial) sidecar.

When `PS_RESOURCE_METRICS` is unset, the partial stats' `cpu_time_ms`,
`major_page_faults`, and `minor_page_faults` are `null` and no `getrusage`
syscalls are made. The metrics are Linux-only (`RUSAGE_SELF`); on other
platforms they log zeros. They include all process threads, so unrelated concurrent node work can
also contribute. If comparing against the baseline, note that
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
Engine validation. A reorg/revert cold-resets the caches and starts a new warm-up epoch. Samples
from earlier canonical heights remain eligible; orphaned heights at or above the reverted range and
each epoch’s warm-up are excluded. The cumulative target spans all epochs. The supervisor sends
`SIGINT` after the target and writes `results.md`.
The output directory must be absent or empty.

```bash
python3 crates/partial-stateless-exex/scripts/run_live_paired_bench.py \
  --reth-bin ./target/release/reth-partial-stateless \
  --datadir /path/to/reth-data \
  --jwtsecret /path/to/jwt.hex \
  --output /path/to/benchmark-output \
  --warmup 60 \
  --samples 600 \
  --parallel-initial-proof off \
  -- \
  --minimal
```

The parallel-proof setting is explicit and defaults to `off`; use
`--parallel-initial-proof on` for the B1 candidate. The script writes the clean primary report,
an overlap-retaining Engine report, and a structured builder report. Raw records and logs are saved
as `paired.jsonl`, `engine.jsonl`, `builder.jsonl`, `resources.jsonl`, and
`reth-partial-stateless.log`.

### Ordinary-builder and P0 comparison benchmark

`scripts/run_live_builder_bench.py` runs `PS_SIDECAR_ROLE=builder`, requires published sidecars,
and fails if an ordinary builder unexpectedly creates the previous-cache snapshot. Run the same
block replay twice with `--force-previous-cache-snapshot` off and on to isolate B2 on one binary:

```bash
python3 crates/partial-stateless-exex/scripts/run_live_builder_bench.py \
  --reth-bin ./target/release/reth-partial-stateless \
  --datadir /path/to/reth-data \
  --jwtsecret /path/to/jwt.hex \
  --output /path/to/builder-output \
  --warmup 60 \
  --samples 600 \
  --parallel-initial-proof off \
  -- \
  --minimal
```

Use `scripts/compare_p0_bench.py CONTROL/builder.jsonl CANDIDATE/builder.jsonl` to join the two
runs by block hash, reject witness-commitment differences, and report paired builder, initial-proof,
and snapshot ratios. Pass `--candidate-source parallel` when isolating B1.

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
