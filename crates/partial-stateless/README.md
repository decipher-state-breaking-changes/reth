# partial-stateless

Library for the **Partial Statelessness**: a protocol-level state cache that
models the state subset every network validator is assumed to hold, plus the
machinery to compute what a block's witness ("sidecar") must contain when some of
its accessed state is *not* in that cache and to verify the resulting state root.

The `NetworkStateCache` holds values and bytecode. Sidecars keep value-cache misses
separate from the authenticated trie paths required to update state written by a
block.

This crate is node-independent — it operates on plain state access-sets, not on a
reth database. The [`partial-stateless-exex`](../partial-stateless-exex) drives it
from a live node; the [binaries](./src/bin/) and [example](./examples/) exercise
it offline.

## Mental model

```
                 (per block, from EVM execution)
  BlockAccessedState  ──compute_miss──▶  MissResult  ──▶  witness targets ──▶ sidecar
         │                  ▲
         └──on_block_executed──┘
              NetworkStateCache  (LastNBlocksPolicy: accounts vs storage/code)
```

A block's execution touches a set of accounts, storage slots, and bytecodes
(`BlockAccessedState`). The `NetworkStateCache` holds whatever was accessed within
the last *N* blocks. Its misses are exactly the values and bytecodes the sidecar
must supply; cache hits remain authoritative even if the sparse trie happens to
contain additional decoded nodes. The parent-state multiproof can also contain
structural targets needed for execution-diff paths or canonical branch/extension
transitions, so proof-target count is not always identical to value-cache miss
count. The sidecar carries no post-state proof: execution updates the authenticated
parent-state paths locally, and the result is checked against the consensus block
root.

## Modules

| Module | Responsibility |
| --- | --- |
| [`accessed_state`](./src/accessed_state.rs) | `BlockAccessedState` — the read/write set captured from revm's `State` after executing a block |
| [`network_cache`](./src/network_cache.rs) | `NetworkStateCache` — insert/refresh/evict entries, `compute_miss`, footprint stats |
| [`witness_check`](./src/witness_check.rs) | sidecar materialization, proof verification, and execution write-target extraction |
| [`policy`](./src/policy.rs) | `CachePolicy` trait + `LastNBlocksPolicy`; separate windows for accounts vs storage/code |
| [`witness`](./src/witness.rs) | turn a `MissResult` into `MultiProofTargets`, measure resulting proof size |
| [`sidecar`](./src/sidecar.rs) | serializable witness sidecar format + benchmark manifest |
| [`persistence`](./src/persistence.rs) | save/load the flat value cache |
| [`fixture`](./src/fixture.rs) | `AccessedStateFixture` — captured per-block access-sets for reproducible offline benchmarks |

## Trustless root verification

For each cache-coherent block, the proof target set is the union of value-cache
misses and state paths actually written by execution. Reth's canonical execution
witness supplies any extra branch or extension nodes needed for canonical trie
collapses and insertions. The producer merges that structure into one parent-state
multiproof.

A verifier authenticates the multiproof against the parent header, re-executes
using cached values plus cold witness values, applies the resulting account and
storage changes to a sparse trie, and compares the computed root with the block
header. The value cache advances only after the root and next cache anchor both
match.

## Binaries & examples

- [`src/bin/`](./src/bin/) — `sidecar_verifier` (trustless witness verification) and
  `cache_window_bench` (offline cache-window vs hit-ratio sweep). See the
  [bin README](./src/bin/README.md).
- [`examples/gen_synth_fixtures.rs`](./examples/gen_synth_fixtures.rs) — generate
  synthetic fixtures to smoke-test the benchmark without a node.

## Test

```bash
cargo test -p partial-stateless
```

Tests cover value-cache policies and rollback, sidecar commitments, account and
storage root updates, creation/deletion structure, integration flows, and reorg
behavior.
