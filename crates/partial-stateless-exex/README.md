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
3. Updates the `NetworkStateCache` (applies the `LastNBlocksPolicy` eviction).
4. Computes a Merkle multiproof for `cache_miss ∪ write_targets`, structurally
   completes it for trie insertions/deletions, and writes a **witness sidecar** +
   JSON benchmark **manifest** to `./sidecar/`.
5. *(optional)* Runs the **trustless sidecar preflight** — re-executes through a
   cache+witness-backed provider, applies the replayed diff to the sparse proof,
   checks the block state root, and then checks the cache-anchor transition.
6. *(optional)* Computes cold-only and full-witness baselines for payload
   comparison without affecting the generated sidecar.
7. Logs accessed/missed counts, miss ratio, witness size, and cache footprint.

On `ChainReorged` the old branch is rolled back newest-to-oldest before the new
canonical branch is applied. On `ChainReverted` the reverted blocks are rolled
back newest-to-oldest. If the required undo history is missing or pruned, the
cache is cold-reset before it advances again.

## Run

```bash
cargo run -p partial-stateless-exex -- node --chain mainnet --datadir /path/to/data
```

The warm cache is persisted to `<datadir>/partial_stateless_cache.bin` and reloaded
on restart (with a gap-tolerance check), so the cache survives node restarts and
short downtime without going cold.

### Configuration

The cache windows are set in `CacheConfig` ([main.rs](./src/main.rs)) — default
`account_window = 60`, `storage_window = 30` blocks. Adjust there and rebuild.
(Use [`cache_window_bench`](../partial-stateless/src/bin/README.md) to pick good
values offline before committing to them.)

Optional diagnostic/benchmark features are off by default and enabled per run
via environment variables, so the core sidecar generation path stays lean:

| Env var | Effect |
| --- | --- |
| `PS_SIDECAR_ROLE=builder\|builder-verifier\|verifier` | choose whether this ExEx writes sidecars, writes and preflights them, or consumes existing sidecars as a live verifier (default: `builder`) |
| `PS_SIDECAR_DIR=<dir>` | write sidecars in `<dir>` (default: `./sidecar`) |
| `PS_SIDECAR_VERIFIER_WAIT_MS=<ms>` | in `verifier` mode, wait up to this long for the block sidecar file to appear (default: `2000`) |
| `PS_CAPTURE_DIR=<dir>` | dump each block's `BlockAccessedState` fixture to `<dir>` (see below) |
| `PS_WITNESS_BASELINE=1` | also compute the previous cold-only and full-witness baselines (two extra multiproofs per block) |
| `PS_RESOURCE_METRICS=1` | capture per-thread CPU time + page faults around the sidecar multiproof (`cpu_time_ms`, `major_page_faults`, `minor_page_faults`) to separate compute-bound from disk-I/O-bound blocks |
| `PS_SIDECAR_PREFLIGHT=1` | run trustless validator preflight for each sidecar (an extra re-execution per block) |

`PS_SIDECAR_ROLE=builder-verifier` is a single-process test mode: it keeps the
normal builder output path, but forces the same trustless client preflight before
publishing each sidecar. Use this mode to observe cache-miss-only,
witness-integrity, state-root, and next-cache-anchor failures while the builder
is running.

`PS_SIDECAR_ROLE=verifier` is the live verifier mode. It does not build or publish
sidecars. For each canonical block it reads
`$PS_SIDECAR_DIR/block_<N>_<hash>.bin`, verifies it against the local previous
cache, re-executes with cache hits plus sidecar miss witnesses, and advances the
local cache only after verification succeeds. The verifier must start with a
cache synchronized to the parent block; the sidecar file alone is not enough to
reconstruct that previous cache.

`PS_SIDECAR_PREFLIGHT` gates the validator-like self-check. When enabled,
sidecar generation fails fast if the cache+witness-backed re-execution, state
proof, calculated state root, expected miss set, or next cache anchor check
fails. When unset, the sidecar still carries `prev_cache_anchor`,
`next_cache_anchor`, and `witness_commitment`, but this ExEx does not spend the
extra execution work to preflight them. The manifest records this as
`trustless_preflight: false`.

The state-root check does not use a full state provider. The actual proof targets
are the union of cache misses and replayed account/storage writes: cold paths
authenticate values absent from the cache, while warm read-only paths remain
omitted. The verifier anchors this proof to the canonical parent header root,
applies the replayed changes to a sparse trie, and requires the resulting root to
match the block header before advancing the cache.

### How the MPT multiproof is represented

Account addresses and storage slots are first Keccak-hashed and expanded into 64
nibbles. A legacy reth `MultiProof` stores account nodes in `account_subtree:
ProofNodes`, a map from the node's full nibble position from the trie root to its
RLP bytes. Storage proofs use the same representation in
`storages[hashed_address].subtree`, with a separate root and nibble namespace per
account.

The map key is not the node hash and is not the leaf's remaining path. Leaf and
extension RLPs carry their own compact-encoded remaining path. Branch RLPs carry
sixteen child references (inline RLP for small children, otherwise a 32-byte
hash) plus the optional branch value. `ProofRetainer` follows the target nibble
paths while `HashBuilder` rebuilds the canonical root, retaining path nodes and
deduplicating shared prefixes in the `ProofNodes` map.

An ordinary proof is enough to update an existing leaf value. Insertion and
deletion can split or collapse a branch and therefore require decoded structure
behind a sibling that the initial proof represented only by its hash. The
producer merges Reth's canonical transition witness into the initial proof. The
verifier has no provider fallback: it rejects an incomplete proof and only
advances the cache after the calculated post-state root matches the block
header.

This prototype intentionally does not implement cold-EOA mempool admission or
new-node cache bootstrap. Those follow-on flows should reuse the fork- and
policy-scoped `CacheAnchor` contract.

When `PS_WITNESS_BASELINE` is unset, the manifest's `full_sidecar_baseline_stats`
and `reduction` are `null` and no baseline multiproof is computed. When enabled,
the full result measures an all-access witness. A baseline failure is non-fatal
— it never blocks the real sidecar.

When `PS_RESOURCE_METRICS` is unset, the partial stats' `cpu_time_ms`,
`major_page_faults`, and `minor_page_faults` are `null` and no `getrusage`
syscalls are made. The metrics are Linux-only (`RUSAGE_THREAD`); on other
platforms they log zeros. If comparing against the baseline, note that
`PS_WITNESS_BASELINE` runs first and can warm the OS page cache, deflating the
partial proof's page-fault counts.

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
| `<datadir>/partial_stateless_cache.bin` | persisted warm cache |
| `./sidecar/block_<N>_<hash>.bin` | witness sidecar (or `$PS_SIDECAR_DIR/block_<N>_<hash>.bin`) |
| `./sidecar/block_<N>_<hash>.manifest.json` | per-block benchmark manifest |
| `$PS_CAPTURE_DIR/accessed_<N>.bin` | captured fixture (when capture is enabled) |
