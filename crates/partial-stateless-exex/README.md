# partial-stateless-exex

A reth [Execution Extension (ExEx)](../exex/exex) that drives the
[`partial-stateless`](../partial-stateless) library from a live node. It maintains
the network-level state cache as the chain advances and, per block, measures the
witness ("sidecar") a partially-stateless validator would need.

Performance records distinguish sidecar generation, paired validation core, standalone
coordinated commit, and background writer costs. Paired-node resource samples include the
node and producer; standalone memory measurements require a separate validator process.

The binary is `reth-partial-stateless` — a full Ethereum node with the ExEx
installed. The crate is split into a library and a thin binary so that the
recovery, bootstrap, and admission paths can be tested from `tests/`: each of
those needs a state provider and an EVM, which `partial-stateless` deliberately
does not depend on.

## What it does per committed block

1. Obtains the block's `BlockAccessedState` (accounts, storage, bytecodes
   touched). With `PS_ENGINE_ACCESS=on` this is the artifact the consensus engine
   already produced when it validated the block, handed over by block hash, and
   no EVM runs here; otherwise — capture off, a handoff miss, or one of the
   sampled blocks kept for the differential comparison — the ExEx re-executes the
   block against its parent state (`history_by_block_number`) itself. Both paths
   produce the same value, and every later step is identical.
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

Committed blocks retain paired trie and flat-cache undo files. The default is **K=32**, with
**zero completed undo payloads retained in memory**: even depth-one recovery reads its file.
Each trie frame restores node preimages and displaced storage tries, including branch/extension
collapses and pruning. A reorg or revert within the available history checks lineage, generation
identity, state roots, cache policy and readiness before publishing the recovered pair.

When that recovery cannot apply, the role determines the fallback. An ExEx-attached full node
may use the **provider-backed canonical rebuild** ([`rebuild.rs`](./src/rebuild.rs)); the
standalone validator has no provider/database path and enters `NeedsSnapshot` until a recovery
checkpoint arrives. Missing or corrupt undo never partially changes the live caches.

The rebuild replays `max_window + 1` heights ending at the canonical target to reconstruct the
flat cache, then authenticates the trie against that block's state root. It requires those
heights to remain readable; aggressive pruning can prevent it. **Rebuild is opt-in under
`PS_CANONICAL_REBUILD=1`** because its whole-cache proof can stall processing. With rebuild
disabled or unavailable, the ExEx resets and warms from live blocks.

| Situation | Recovery |
| --- | --- |
| Cold start | Rebuild at the first notified block's parent if enabled; otherwise warm from live blocks. |
| `ChainReorged` | Undo the abandoned branch to the common ancestor, then apply the new blocks normally. If retained history cannot recover it, attempt rebuild. |
| `ChainReverted` | Undo the reverted branch to its parent; attempt rebuild on refusal. |
| Gap or wrong-branch parent | Attempt rebuild at the rejected block's parent, then reset/warm on failure. |
| Rebuild disabled, unavailable or failing | Log the outcome and warm. Three consecutive rebuild failures stop further attempts for the run. |

Undo is independent of the rebuild switch and cannot promote an under-warmed cache to Ready.
The recovered pair must still cover its policy window or be backed by an authenticated
checkpoint. K bounds recoverable execution blocks; **K=64 is supported but is not a finality
guarantee**. Undo files are disposable session data, not warm-restart checkpoints.

Recovery runs between block-verification samples. Historical depth-one timings do not measure
the current disk path.

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
that can replay bridges the drift with a canonical rebuild, which is the one
situation where turning it on is close to mandatory rather than a trade; a node
that cannot stays Cold until a fresher snapshot is supplied. That is a real limitation of this
phase, which is why the gate above restores in-process rather than across a
restart.

## Run

```bash
cargo run -p partial-stateless-exex -- node --chain mainnet --datadir /path/to/data
```

The flat cache is persisted to
`<datadir>/partial_stateless_cache-a<A>-s<S>.bin` for the selected windows, but the
matching sparse-trie snapshot is not yet persisted, so a non-empty persisted
value cache is still cold-reset on restart. A full node can buy its way out of
that with `PS_CANONICAL_REBUILD=1`, which puts the pair back at `Ready` before
the first notified block is applied — at the cost of the startup stall above.
Atomic value+trie+anchor persistence remains the thing that would make a warm
restart real, and free.

### Configuration

The cache windows default to `account_window = 60`, `storage_window = 30` blocks and are
selected at startup with `PS_ACCOUNT_WINDOW` and `PS_STORAGE_WINDOW`. Both must be positive
base-10 integers; invalid values fail startup. Changing a window does not require rebuilding,
but it does select a different cache-policy ID and persisted-cache filename. The optional
[`cache_window_bench`](../partial-stateless/src/bin/README.md)
can compare candidate windows offline.

Disk undo is enabled by default. Its settings survive snapshot restore, rebuild and reset.
Incompatible legacy settings fail startup; unset them to use the current profile. Optional
diagnostics and benchmark features still require explicit enablement.

| Env var | Effect |
| --- | --- |
| `PS_SIDECAR_ROLE=builder\|builder-verifier\|verifier` | choose whether this ExEx writes sidecars, writes and preflights them, or consumes existing sidecars as a live verifier (default: `builder`) |
| `PS_SIDECAR_DIR=<dir>` | write sidecars in `<dir>` (default: `./sidecar`) |
| `PS_ACCOUNT_WINDOW=<n>` / `PS_STORAGE_WINDOW=<n>` | inclusive Last-N account and storage/code cache windows (defaults: `60` / `30`). Both are runtime protocol parameters: they select the policy ID and persisted-cache filename; non-positive, signed, whitespace-padded, or non-decimal values fail startup |
| `PS_WITNESS_V3=1` | emit receiver-aware `MptTrimmedTransitionNodes` for a Ready cache; Cold/Warming and full-witness sidecars remain self-contained v2 (default: disabled; enabled explicitly by the frozen cohort profile) |
| `PS_RETAIN_DEPTH=<K>` | undo retention cap, default `32`, range `1..=64` execution blocks |
| `PS_UNDO_DIR=<dir>` | undo session directory parent, default `$PS_SIDECAR_DIR/undo`; failure to create/open/lock the store aborts startup |
| `PS_TRIE_REPR=exact` | required representation and default; legacy `parallel` is rejected |
| `PS_UNDO_LAYOUT=frames` | required layout and default; legacy `hybrid` is rejected |
| `PS_UNDO_RECORD=on` | recording is on by default; disabling it is rejected |
| `PS_RETAIN_GENERATION=1` | retention is on by default; disabling it is rejected |
| `PS_SIDECAR_VERIFIER_WAIT_MS=<ms>` | in `verifier` mode, wait up to this long for the block sidecar file to appear (default: `2000`) |
| `PS_CAPTURE_DIR=<dir>` | dump each block's accessed-state snapshot to `<dir>` (see below) |
| `PS_POLICY_DATASET_CAPTURE_DIR=<abs dir>` | capture the policy replay dataset into `<abs dir>`: raw payload, access set, and a policy-neutral full witness per block, so every cache policy can be generated offline later. Absolute paths only; refused alongside any measuring variable; requires `PS_ENGINE_ACCESS=on` and `PS_ENGINE_PAYLOAD=on` (see below) |
| `PS_POLICY_DATASET_MAX_BLOCKS=<n>` | **usable** blocks the capture records — reorg-abandoned ones stop counting and are replaced. Required whenever `PS_POLICY_DATASET_CAPTURE_DIR` is set; there is no default |
| `PS_POLICY_DATASET_CONFIRMATIONS=<n>` | canonical blocks the chain must advance past the recorded range before `END.json` is written (default: `96`, roughly three epochs). `0` disables the wait and leaves the tail reorg-exposed; the terminator records which was chosen |
| `PS_POLICY_DATASET_ALLOW_UNSTAMPED=1` | capture from a build that carries no `PS_BUILD_COMMIT`. Without it, such a run is a startup error: the manifest would record `build_commit: null`, and a corpus that cannot name the code that produced it is not evidence. The escape hatch hides nothing — the manifest still records `null` |
| `PS_WITNESS_BASELINE=1` | also compute the full-witness baseline + reduction ratio (an extra, larger multiproof per block) |
| `PS_PARALLEL_INITIAL_PROOF=1` | use Reth's proof workers for eligible initial V2 multiproofs; low-width target sets and later structural deltas stay serial |
| `PS_INITIAL_PROOF_AB=1` | benchmark-only: prove every eligible block's initial targets **both** ways — serial and wide — alternating which call runs first, and record both times plus whether they proved the same nodes. Implies the wide path. The block keeps the serial proof, so it publishes what a serial run would; it pays a second initial multiproof, so `builder_total_us`, process CPU and RSS in such a run are not a production builder's. Blocks below the wide gate, and blocks whose wide call failed, report no pair |
| `PS_RESOURCE_METRICS=1` | capture process CPU time + page faults around transition-witness construction, including parallel proof workers (`cpu_time_ms`, `major_page_faults`, `minor_page_faults`) |
| `PS_ENGINE_ACCESS=off\|shadow\|on` | reuse the engine's own execution instead of re-executing each block: `shadow` captures and compares while still re-executing, `on` consumes the artifact (default: `off`) |
| `PS_SHADOW_SAMPLE=<n>` | in `on` mode, re-execute one block in `n` anyway and compare it against the artifact, keeping the differential oracle alive; `0` disables sampling (default: `50`) |
| `PS_HANDOFF_CAPACITY=<n>` | artifacts the handoff retains before evicting the oldest insert (default: `4`) |
| `PS_HANDOFF_MAX_BYTES=<bytes>` | access-set byte budget for resident artifacts; excludes the shared execution outputs, so it is not an RSS cap (default: 256 MiB) |
| `PS_ENGINE_BENCH=1` | enable the lightweight Vanilla Engine V2 timing collector; usable by a standard Reth node without the ExEx |
| `PS_ENGINE_BENCH_OUTPUT=<file>` | JSONL destination for Vanilla Engine V2 timing records (default: `./engine_bench.jsonl`) |
| `PS_VALIDATION_BENCH=1` | enable in-memory DB-free Partial/Weak validation paired with same-block Vanilla Engine timing; requires `builder-verifier` |
| `PS_BENCH_OUTPUT=<file>` | JSONL destination for paired Partial/Weak benchmark records |
| `PS_BUILDER_BENCH_OUTPUT=<file>` | JSONL destination for per-block builder proof, snapshot, commitment, and total-cost records |
| `PS_FORCE_PREVIOUS_CACHE_SNAPSHOT=1` | benchmark-only control that recreates the old unconditional parent-cache clone, so its cost can be priced against the current conditional one |
| `PS_TRIE_CACHE_DIAGNOSTICS=1` | validate retained account/storage paths and log trie shape, memory, and transition timings |
| `PS_CANONICAL_REBUILD=1` | reach `Ready` by rebuilding the pair from canonical state at cold start and after a failed recovery, instead of warming over a policy window of live blocks (default: disabled) |
| `PS_BOOTSTRAP_DIR=<dir>` | where the snapshot package and its checkpoint live (default: `$PS_SIDECAR_DIR/bootstrap`) |
| `PS_BOOTSTRAP_EXPORT=1` | export a snapshot the first time the tracker reaches Ready |
| `PS_BOOTSTRAP_IMPORT=1` | restore from a snapshot at startup, ahead of the persisted flat cache |
| `PS_BOOTSTRAP_SELF_TEST=<n>` | export at the first Ready, restore a second pair in-process, and compare both pairs for the next `n` blocks (implies export; incompatible with `PS_STREAM_DIR`) |
| `PS_STREAM_DIR=<dir>` | record the event stream (manifest, checkpoint + snapshot chunks, commits, lifecycle frames) into `<dir>` as a live spool `ps-replay` can follow; implies the snapshot export. A run manifest with build/host provenance is appended beside the spool as `<dir>.run-manifest.jsonl` |
| `PS_STREAM_RESUME=1` | continue a non-empty spool as a new epoch after reading and checking every frame already in it (default: a non-empty spool is refused) |
| `PS_STREAM_FSYNC=0\|1` | `1` selects the power-loss durability profile: every frame is fsynced before its rename and the spool directory after it (one directory sync per checkpoint burst), and snapshot packages likewise. Default `0` keeps tmp+rename — durable across a process restart only. Anything else is a startup error |
| `PS_STREAM_PRODUCER=<name>` | producer identity stamped into the manifest (default: crate name + version) |
| `PS_STREAM_CHUNK_BYTES=<n>` | snapshot chunk size in the spool (default: 8 MiB) |
| `PS_STREAM_BUFFER_MAX_BYTES` / `PS_STREAM_BUFFER_MAX_FRAMES` | bounds on frames buffered while the snapshot export runs; overflow drops the buffer whole and fails the attempt (defaults: 256 MiB / 128) |
| `PS_STREAM_MAX_SPOOL_BYTES` / `PS_STREAM_MAX_SPOOL_FRAMES` | spool ceilings; reaching one closes the stream with `End(spool_limit)` (defaults: 64 GiB / 100000) |
| `PS_STREAM_EXPORT_RETRIES=<n>` | fresh export attempts after a genuine failure; a reorg fence does not consume one (default: 1) |
| `PS_STREAM_EXPORT_MAX_WORKERS=<n>` | live export workers allowed at once, abandoned ones included — each holds an MDBX read transaction for its whole multiproof. At the cap a fresh attempt waits for a slot; invalid values are a startup error (default: 4) |
| `PS_STREAM_REORG_CHECKPOINT=always\|never` | whether a branch change re-checkpoints the open stream at the block it recovered to (default: `always`; anything else is a startup error) |
| `PS_BUILD_COMMIT` / `PS_BUILD_DIRTY` / `PS_CARGO_LOCK_SHA256` | **compile-time**, not runtime: exported before `cargo build`, read by `option_env!`, and baked into the binary so its run manifests can name their own build (`git rev-parse HEAD`, `0`/`1` tree dirtiness, `sha256sum Cargo.lock`). Unset builds stamp `null` plus a note; the long gate requires a non-null commit from a clean tree, and a policy replay dataset capture refuses to start without one |

### Disk undo operation and standalone replay

No undo flags are needed for an ordinary ExEx run. To select a larger cap or a separate disk,
set `PS_RETAIN_DEPTH=64` and/or `PS_UNDO_DIR=/path/to/undo` before launching the node. Remove old
`PS_TRIE_REPR=parallel`, `PS_UNDO_LAYOUT=hybrid`, `PS_UNDO_RECORD=off` and
`PS_RETAIN_GENERATION=0` settings; they no longer select a supported ExEx mode.

`ps-replay` batch and follow use K=32, recording on, frames and `<spool-dir>/undo` by default:

```sh
ps-replay <spool-dir>
ps-replay --follow <spool-dir>
# Optional overrides, supported in both modes:
ps-replay <spool-dir> --retain-depth 64 --undo-dir /path/to/undo
```

Replay also accepts `PS_RETAIN_DEPTH`, `PS_UNDO_DIR`, `PS_UNDO_RECORD` and `PS_UNDO_LAYOUT`;
CLI flags override their environment values. `--undo-record on` and `--undo-layout frames`
are redundant; `off` and `hybrid` are rejected. A read-only spool needs a writable `--undo-dir`.
Listing/inspecting spool frames does not create an undo store.

After pending writes finish, each pair has at most K `.undo` files. Expiry, rollback and reset
release files. A writer may additionally have one temporary/finishing file; lock files are
metadata. Startup cleans inactive owned sessions and preserves active sessions. The writer has
one active and one queued bundle, plus encoding buffers and an outgoing bundle while enqueue
waits. These transient allocations and the per-block working clone are outside the zero-resident
history policy. Recovery also needs temporary copies of the live caches plus one decoded bundle;
no disk-path peak-RSS measurement is claimed.

Enqueue warns after 100 ms and refuses after 1 s. Recovery waits at most 5 s for a pending write;
these limits do not cancel filesystem syscalls. Write, file-integrity, lineage, generation,
root and readiness failures log their reason and block before the existing recovery fallback.
History-discard warnings include `dropped_generations` and `retained_depth`. A restore rejection
that preserves history reports zero dropped generations; retained depth alone does not prove
that every file is readable.

The first cold commit has no authenticated parent to recover. It skips retaining that parent
and logs `Initialized disk undo history; no recoverable parent` at INFO. Subsequent warming
commits accumulate undo normally. A later `full_fallback`, including while warming, still warns
and discards the history that can no longer be reached.

The startup retention log and per-commit debug log report the actual resident Full/Frame count.
Replay records that count as `undo_resident_blocks` in each committed block's timing record
(`null` for an attempt that did not commit); the manifest's `undo_resident_blocks_limit` is the
configured limit, not a measurement. Both exclude writer buffers and disk handles.
The benchmark report's **Newest retained generation** section describes one entry, not the
configured K or the whole deque. A disk handle counts as present while its file payload is
excluded from those memory figures.

When deploying, confirm K, layout, recording and directory in startup logs, then check ordinary
commits for file rotation and writer/coverage-loss warnings. No additional performance campaign
is required to enable the feature.

### Execution reuse and sidecar roles

`PS_ENGINE_ACCESS` shares one execution between the node and the ExEx. The engine
captures the access set at the same point of the same lifecycle the ExEx would,
after its own validation succeeds, and publishes it into a bounded store keyed by
block hash; the builder takes it by exact hash. Nothing waits: a contended publish
drops, a full store evicts its oldest insert, and any absence falls back to
re-executing the block, which is always correct and merely slower. Lookup is never
by height, so a reorg sibling can only ever be served its own artifact. In `on`
mode `PS_SHADOW_SAMPLE` keeps re-executing a fraction of blocks and comparing the
two access sets, so the equality that justifies the reuse stays under test instead
of being assumed after the initial `shadow` run.

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

With canonical rebuild disabled, a cold pair first spends `window_size + 1`
contiguous blocks (61 under the default policy) establishing an authenticated
`Ready` parent. No sidecar or paired record exists during this readiness
bootstrap. Because those live blocks already evolve both caches for a complete
policy window, the default paired sample warm-up is **0**. Progress reads
`bootstrap=N paired_sampling=no` until the first sidecar, then
`paired_sampling=yes sample_warmup=0/0`.

`--canonical-rebuild on` is different: it installs a minimal whole-cache
multiproof at Ready rather than evolving the trie through live blocks. Historical
Earlier rebuild measurements showed its revealed intermediate-node set converging for about
50 more blocks, so the runner automatically uses 60 paired warm-up records in that mode.
An explicit `--warmup N` overrides either default. When invoking either offline
analyzer directly, pass that run's selected value explicitly; the raw paired
records do not encode which Ready path established the cache.

The default run then collects 600 same-hash accepted samples. It discards invalid pairs and any
pair whose Partial/Weak interval overlaps the start of the next Engine validation. A warm retained-
generation branch switch removes orphaned samples but does not re-arm sample warm-up; only a cold
reset opens a new warm-up epoch. Samples from earlier canonical heights remain eligible, and the
cumulative target spans all epochs. The supervisor sends `SIGINT` after the target and writes
`results.md`.
The output directory must be absent or empty.

```bash
python3 crates/partial-stateless-exex/scripts/run_live_paired_bench.py \
  --reth-bin ./target/release/reth-partial-stateless \
  --datadir /path/to/reth-data \
  --jwtsecret /path/to/jwt.hex \
  --output /path/to/benchmark-output \
  --samples 600 \
  --parallel-initial-proof off \
  -- \
  --minimal
```

The parallel-proof setting is explicit and defaults to `off`; use
`--parallel-initial-proof on` to measure the parallel-proof candidate. The script writes the clean primary report,
an overlap-retaining Engine report, and a structured builder report. Raw records and logs are saved
as `paired.jsonl`, `engine.jsonl`, `builder.jsonl`, `resources.jsonl`, and
`reth-partial-stateless.log`.

### Ordinary-builder comparison benchmark

`scripts/run_live_builder_bench.py` runs `PS_SIDECAR_ROLE=builder`, requires published sidecars,
and fails if an ordinary builder unexpectedly creates the previous-cache snapshot. Run the same
block replay twice with `--force-previous-cache-snapshot` off and on to isolate the
previous-cache-snapshot cost on one binary:

```bash
python3 crates/partial-stateless-exex/scripts/run_live_builder_bench.py \
  --reth-bin ./target/release/reth-partial-stateless \
  --datadir /path/to/reth-data \
  --jwtsecret /path/to/jwt.hex \
  --output /path/to/builder-output \
  --samples 600 \
  --parallel-initial-proof off \
  -- \
  --minimal
```

Use `scripts/compare_p0_bench.py CONTROL/builder.jsonl CANDIDATE/builder.jsonl` to join the two
runs by block hash, reject witness-commitment differences, and report paired builder, initial-proof,
and snapshot ratios. Pass `--candidate-source parallel` when isolating the parallel initial proof.

### Transition-witness construction

The initial builder target set is the union of value-cache misses and mutation paths not already
authenticated by the persistent trie cache. It requests that set once through
`StateProofProvider::multiproof_v2`. Native V2 proof generation builds targeted storage proofs
first and reuses their roots when encoding account leaves, avoiding a second traversal of those
storage tries when a full storage proof is available.

With `PS_PARALLEL_INITIAL_PROOF=1`, the initial parallel-proof gate requires at least two distinct
storage tries and 64 total initial targets. Eligible calls use one account worker and a
workload-bounded number of storage workers; smaller calls and later structural proof deltas stay
serial. This provider option is independent of `PS_TRIE_REPR`, which must remain `exact`.

The proof is revealed into one transactional sparse-trie clone. A deletion can expose a blinded
sibling or extension child whose node kind is needed for canonical branch compression. In that
case the transition reports all currently visible `(key, min_len)` targets, the builder subtracts
targets already requested, fetches only the delta, and resumes the unfinished session. There is no
full proof regeneration or transition replay. An empty delta is rejected as no progress, and a
128-round cap guards malformed or unexpectedly deep chains of structural dependencies.

The self-contained v2 wire uses `MptTransitionNodes`: deterministic, hash-deduplicated
parent-state RLP node preimages. With `PS_WITNESS_V3=1`, a Ready builder instead emits
`MptTrimmedTransitionNodes`: the builder and validator share the same composite walk, prefer
the authenticated local trie, and carry only nodes consumed after a blinded frontier. The wire
binds the retention version/fingerprint, verifies every graft by its anchor hash, and rejects
missing or unconsumed nodes. Cold/Warming and full-witness paths degrade to v2. Legacy
`MptMultiProof` sidecars remain decodable for compatibility.

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

### Capturing accessed-state data

Set `PS_CAPTURE_DIR` to dump each block's `BlockAccessedState` as an accessed-state file. This
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

### Capturing a policy replay dataset

The accessed-state data above answers what each block *accessed*, which is all a cache hit/miss
sweep needs. Comparing what cache policies actually *cost* needs more: the real
sidecar each policy would have produced for each block, which needs the parent-state
proofs, which ordinarily needs the node database.

`PS_POLICY_DATASET_CAPTURE_DIR` records the one artifact that removes that
dependency — a **policy-neutral full transition witness** per block, proved against a
cold cache and an empty trie, so it names no window, no anchor, and no miss set. Every
policy's own witness is a subset of it, because every target a warm cache lets a policy
skip is a target the cold build already proved. With it plus the block's Engine payload
and access set, [`ps-policy-frontier`](../partial-stateless-frontier) generates and
validates every policy's real sidecar with no database at all.

```bash
cd /path/to/reth
PS_TARGET_DIR=$(cargo metadata --format-version 1 --no-deps | jq -r .target_directory)

# Exported before the build, not before the run: these are read at compile time, so a binary
# built without them carries no commit and a capture that used it would record `build_commit:
# null`. That is refused at startup rather than discovered hours later in the manifest.
export PS_BUILD_COMMIT=$(git rev-parse HEAD)
export PS_BUILD_DIRTY=$([ -z "$(git status --porcelain)" ] && echo 0 || echo 1)
export PS_CARGO_LOCK_SHA256=$(sha256sum Cargo.lock | cut -d' ' -f1)
cargo build --release -p partial-stateless-exex -p partial-stateless-frontier

PS_ENGINE_ACCESS=on \
PS_ENGINE_PAYLOAD=on \
PS_SHADOW_SAMPLE=0 \
PS_SIDECAR_ROLE=builder \
PS_HANDOFF_CAPACITY=32 \
PS_PAYLOAD_HANDOFF_CAPACITY=32 \
PS_POLICY_DATASET_CAPTURE_DIR=/abs/path/policy-dataset \
PS_POLICY_DATASET_MAX_BLOCKS=1200 \
PS_POLICY_DATASET_CONFIRMATIONS=96 \
    "$PS_TARGET_DIR/release/reth-partial-stateless" node \
    --chain mainnet \
    --datadir /path/to/data \
    --authrpc.jwtsecret /path/to/jwt.hex \
    --db.read-transaction-timeout 0
```

Both handoffs are widened from their default of 4, because a capture arrives late. The node
spends most of a minute reaching the point where the ExEx takes its first notification, and
the Engine taps keep handing off through their ring buffers meanwhile — so at the default the
first block the capture sees has often already been evicted, and it has neither the payload
nor the access set the corpus requires. 32 slots is roughly six minutes of chain at mainnet
block times. Before the first record, and only for a typed payload/access handoff miss, the
corpus starts later and files the skip in `lifecycle.jsonl`. Provider, encoding, filesystem,
and internal-invariant failures are fatal from the first block. After the first record even a
handoff miss is fatal, because it would leave a hole in a corpus that otherwise looks complete.

Run the node from somewhere other than the source tree. Some diagnostic output resolves
against the working directory, so a capture launched from the repo leaves files in it, which
then read as uncommitted changes the next time a build wants a clean tree.

Build and hash both `reth-partial-stateless` and `ps-policy-frontier` in the same stamped
shell before capture, then check the binaries rather than the build log: a `cargo build` run
from a shell where the stamps were unset can relink a cached unstamped artifact without even
changing its mtime, which leaves no other trace.

```bash
for bin in "$PS_TARGET_DIR/release/reth-partial-stateless" \
           "$PS_TARGET_DIR/release/ps-policy-frontier"; do
    grep -qa "$PS_BUILD_COMMIT" "$bin" || echo "UNSTAMPED: $bin"
done
```

Do not assume they are under `./target`: Cargo may select a shared
target directory through configuration or `CARGO_TARGET_DIR`. For an SSH-safe background
run, put the command and the `END.json` watcher in one `nohup` supervisor; the shell which
launched only the node is not a supervisor after the SSH session disappears.

**A capturing run is not a measurement run, and the contract says so in three places.**
The manifest carries `measurement_eligible: false` and `capture_overhead_excluded: true`;
startup refuses the capture alongside `PS_VALIDATION_BENCH`, `PS_BENCH_OUTPUT`,
`PS_BUILDER_BENCH_OUTPUT`, `PS_CAPTURE_DIR`, `PS_WITNESS_BASELINE`, or
`PS_RESOURCE_METRICS`; and the measurement launchers refuse to start while the variable
is set rather than quietly clearing somebody's running capture. The capture builds a
second, larger witness per block and writes it to disk, so anything measured beside it
would be measuring the capture.

The two Engine handoffs are required rather than optional, and so is `PS_SHADOW_SAMPLE=0`.
The recorded access set has to be the one production runs on (`PS_ENGINE_ACCESS=on`), and
the recorded payload has to be the one a consensus client actually sent
(`PS_ENGINE_PAYLOAD=on`) — a payload derived from a block this node already accepted hands
a later validator the answers its own admission checks exist to question. Sampling is
refused for the same reason on the other input: it re-executes one block in
`PS_SHADOW_SAMPLE` and records *that* block's own access set, so a corpus captured with it
on would claim every record came from the Engine while a fraction did not. Any record whose
provenance is not the Engine's — a sampled block, a handoff miss on a WAL replay, an
artifact that would not downcast — fails the capture rather than being written.

Nothing is lost by turning sampling off here. The capture re-executes every block
database-free against its own witness and compares access sets before writing it, and the
offline generator does the same again on another host. Both are stronger oracles than the
sampled comparison they replace.

Every captured block is proved before it is written: the full witness must re-execute
the block with no database, reconstruct the header's state root, and agree with this
node's own execution on gas, receipts root, and requests hash, and the access set the
re-execution observes must equal the recorded one. A block that fails any of those fails
the run, because an incomplete corpus is worse than no corpus — it looks like a complete
one. `END.json` is written last, and a dataset without it is refused as incomplete.

**Reaching the block budget is not finishing.** Writing `END.json` at the last file would
vouch for a tip that is still reorg-exposed, so the capture then stops building witnesses
and simply watches the chain advance `PS_POLICY_DATASET_CONFIRMATIONS` blocks past the range
it recorded. A reorg during that wait puts it back to work: the abandoned records stay on
disk but stop spending the budget, and the capture records replacements. The terminator
names a `usable_range` — the part the producer actually stands behind — and the loader drops
everything outside it, so a run that stopped early yields a shorter corpus rather than an
overstated one.

Both records at a contested height stay on disk, and the canonical set is **derived** from
the records rather than inferred from the log: the terminator names the hash at the top of
the usable range, and a reader walks `parent_hash` down from it. That is what makes a chain
which leaves a branch and later returns to it readable — an accumulated list of abandoned
hashes can only ever grow, so it would mark both branches abandoned and leave the contested
height empty.

`lifecycle.jsonl` records reorgs and resets so the exclusions can be *audited*, which is a
separate job from deciding them. A lifecycle event that cannot be written fails the whole
dataset rather than warning: a reorg that happened but was never logged leaves a corpus
indistinguishable, on disk, from one where nothing happened. The loader likewise refuses a
dataset with no log at all — every capture writes one before its first block, so its absence
means it was lost.

The capture root must be an empty or nonexistent directory. A directory with no records but
a leftover `END.json` is the dangerous case — a capture started there and killed would read
as complete, terminated by the previous run's verdict over this run's blocks — so it is
refused up front. On the reading side, the loader cross-checks the terminator's own record
count and block range against the files present, refuses a manifest from another schema
version, and checks the confirmation claim rather than taking it: a terminator that vouches
for a tip must record a canonical head at least `confirmations` above it, or the depth it
names is just a number in a file.

`END.json` closes the **producer** side of this contract; it is not by itself a dataset
acceptance result. Before preserving a capture as an experiment input, run the offline loader
and a small end-to-end replay. The loader verifies every record digest before applying
`--warmup` or `--samples`, so even a five-sample smoke checks the whole input first:

```bash
"$PS_TARGET_DIR/release/ps-policy-frontier" \
    --dataset /abs/path/policy-dataset \
    --arm weak --arm 60/30 --arm 90/60 --arm 120/45 \
    --warmup 121 --samples 5 \
    --out /abs/path/frontier-smoke
```

The acceptance gate is: a structurally valid terminator, successful loader verification,
successful database-free replay, and all requested arms completing. Never promote a capture
to a measurement corpus from the existence of `END.json` alone.

That gate exists because the first 1,200-block capture passed every producer-side check and
was still unusable. Its records carried a digest taken over their `bincode` serialization, and
a record holds the access set in `HashMap`s whose iteration order is seeded per process and
rebuilt on deserialization — so a record hashed one way when written and another when read
back, and the whole capture failed its own integrity check on load. Records now carry a digest
over an explicit, sorted, length-prefixed encoding, which is what the schema version at the
head of every record and manifest tracks; a capture from the superseded schema is refused at
its manifest.

**Schema-1 failure and accepted schema-2 replacement (2026-08-18--19).** The first 1,200-block
live capture closed cleanly and waited for 96 confirmations, but this acceptance smoke refused
block 25,781,091.
Schema 1 digests `bincode` bytes containing three unordered `HashMap`s in
`BlockAccessedState`; after deserialization their iteration order is not stable, so
re-serializing the same semantic record can produce a different digest. Hashing the exact body
bytes stored in the rejected file reproduces its recorded digest, which distinguishes this
from observed disk corruption. Its bulk data was discarded rather than migrated. The schema-2
recapture --- run `policy-frontier-20260818-223726` (`raw-block-witness-data`, retained in
the recording host's local artifact store) --- passed the
full loader, five-sample smoke, and 1,000-block four-policy replay and is the accepted evidence
input; its run root carries `RESULT_SHA256SUMS`.

## Outputs

| Path | Contents |
| --- | --- |
| `<datadir>/partial_stateless_cache-a<A>-s<S>.bin` | persisted flat cache for account/storage windows `<A>/<S>`; another policy uses another filename |
| `./sidecar/block_<N>_<hash>.bin` | witness sidecar (or `$PS_SIDECAR_DIR/block_<N>_<hash>.bin`) |
| `./sidecar/block_<N>_<hash>.manifest.json` | per-block benchmark manifest |
| `$PS_UNDO_DIR/undo-*/<sequence>.undo` | paired trie/flat undo bundles; default parent directory is `$PS_SIDECAR_DIR/undo`, capped at K completed files per pair after writes finish |
| `$PS_CAPTURE_DIR/accessed_<N>.bin` | captured accessed-state data (when capture is enabled) |
| `$PS_POLICY_DATASET_CAPTURE_DIR/manifest.json` | policy replay dataset identity, capture configuration, and the measurement disclaimer |
| `$PS_POLICY_DATASET_CAPTURE_DIR/blocks/block_<N>_<hash>.bin` | one captured block: payload, access set, policy-neutral full witness, roots, and a record digest |
| `$PS_POLICY_DATASET_CAPTURE_DIR/lifecycle.jsonl` | reorgs and resets seen under the capture; required, and a failed write fails the dataset |
| `$PS_POLICY_DATASET_CAPTURE_DIR/END.json` | written last; names the usable range, its tip hash, the confirmation depth, and the head that backs it. Its absence means the capture did not finish |
