# partial-stateless

Library for the **Partial Statelessness**: a protocol-level state cache that
models the state subset every network validator is assumed to hold, plus the
machinery to compute what a block's witness ("sidecar") must contain when some of
its accessed state is *not* in that cache.

The flat `NetworkStateCache` holds values and bytecode. The coordinated
`PartialTrieNodeCache` holds authenticated account paths and per-account storage
paths in a locally updated sparse trie.

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
contain additional decoded nodes. The state payload is a flat canonical transition
witness: it contains parent-state node preimages for cache misses, execution-diff
paths, and any branch/extension structure discovered while applying the complete
transition. The sidecar carries no post-state proof: execution updates storage and
account paths locally, and the result is checked against the consensus block root.

## Modules

| Module | Responsibility |
| --- | --- |
| [`accessed_state`](./src/accessed_state.rs) | `BlockAccessedState` — the read/write set captured from revm's `State` after executing a block |
| [`network_cache`](./src/network_cache.rs) | `NetworkStateCache` — insert/refresh/evict entries, `compute_miss`, footprint stats |
| [`trie_cache`](./src/trie_cache.rs) | persistent account/storage sparse trie, exclusion-aware path retention, shape metrics, and invariants |
| [`witness_check`](./src/witness_check.rs) | sidecar materialization, execution-diff application, and local post-state root calculation |
| [`policy`](./src/policy.rs) | `CachePolicy` trait + `LastNBlocksPolicy`; separate windows for accounts vs storage/code |
| [`witness`](./src/witness.rs) | turn a `MissResult` into `MultiProofTargets`, measure resulting proof size |
| [`sidecar`](./src/sidecar.rs) | serializable witness sidecar format + benchmark manifest |
| [`persistence`](./src/persistence.rs) | save/load the flat value cache; sparse-trie persistence is not implemented yet |
| [`fixture`](./src/fixture.rs) | `AccessedStateFixture` — captured per-block access-sets for reproducible offline benchmarks |

## Persistence limitation

Only the flat value cache is serialized. The sparse-trie snapshot and
branch-aware undo state are not persisted. The ExEx therefore cold-resets both
caches on restart, reorg, and revert so a value is never treated as a cache hit
without its authenticated path. A builder can initialize its local caches from a
full parent-state provider on the first unsynchronized block, but that local
initialization does not publish a cache-coherent bootstrap sidecar. A sidecar-only
verifier cannot cold-start or recover from a reset until the value and sparse-trie
caches can be restored together or a protocol bootstrap mechanism is added.

## Per-block trie synchronization

For each block, execution records touched parent-state values while the value-cache miss is
computed against the parent cache. Cache misses remain parent-value proof targets; they are not
inserted into the post-state diff. The builder starts one transactional sparse-trie session with
the real execution diff, reveals an initial native V2 proof for misses and initially
uncovered mutation paths, and then resumes the same session whenever a blinded structural child is
encountered. Paths already authenticated by the persistent trie cache are omitted from provider
targets. Each structural request fetches only the newly visible `(key, min_len)` delta, with no
transition replay and no accumulated-proof regeneration.

Storage wipe, upsert, removal, account upsert, and account removal ordering is preserved. The
session's final root is the builder root check; the normal builder path does not perform a second
DB-free transition replay. The root must equal the consensus block root before value-cache
membership drives sparse-trie retention. A failure rolls back the value-cache transition and
discards the trie clone; the two caches advance together only after the block checks succeed.

Retention keeps a complete lookup witness for every cached value. Existing accounts and nonzero
storage slots retain their inclusion paths. Nonexistent accounts and zero storage slots retain the
terminal branch, extension, or different leaf that proves exclusion. Unrelated decoded subtrees
are replaced with their authenticated hashes, so cache size follows the value-cache windows rather
than growing with every revealed proof. A per-account storage trie is discarded when its last
cached slot expires.

Only `NetworkStateCache` determines hits, misses, eviction, and cache anchors. Additional decoded
trie nodes never change the sidecar miss manifest or its anchors, and the sidecar carries no
post-state proof.

## Cache-aware structural witness generation

Leaf deletion can collapse a branch or extension. A blinded sibling hash commits to the parent
state but does not reveal the sibling's node kind, which is needed to construct the new canonical
shape. These structural dependencies cannot always be predicted from the transaction read set
alone.

The builder does not issue legacy leaf multiproofs or retry the transition from scratch. The
`StateProofProvider::multiproof_v2` API preserves native account/storage targets and `min_len`
through latest, historical, in-memory overlay, and delegating providers. A 128-round cap and
no-progress check remain as safety guards for chained structural gaps.

The production sidecar uses the flat transition-node format. Structural targets are normalized to
root-to-target paths (`min_len=0`) and their RLP preimages are hash-deduplicated. Native V2 proof
generation computes storage proofs first and reuses their storage roots while encoding account
leaves, avoiding a second traversal of targeted storage tries.

The builder session advances its persistent trie cache and checks the consensus root directly. A
separate serialized-sidecar re-execution is optional and runs only when `PS_SIDECAR_PREFLIGHT` is
set or the role is `builder-verifier`. Legacy multiproof and flat transition-node sidecars remain
decodable.

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

Tests cover value-cache policies and rollback, sidecar commitments, persistent
account updates, storage-root propagation, transactional sparse-trie cloning,
retained-path invariants, integration flows, and reorg behavior.
