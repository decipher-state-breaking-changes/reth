# partial-stateless-frontier

Generates **every cache policy's real sidecar for one identical block set**, offline,
with no state database. The binary is `ps-policy-frontier`.

## Why this exists

A cache-policy comparison measured across separate live runs cannot separate the policy
from the run. Two runs see different blocks, a different mempool, and a different machine
state, and nothing in the result says which of the three moved the number.

Replaying one recorded corpus through every policy fixes the block set. Host load and
measurement order still require control in the separate validation passes. The tool
reports a digest over the measured block hashes so two runs, on two hosts, can prove they
compared the same corpus.

## What makes it possible

The corpus is captured by a live node under
`PS_POLICY_DATASET_CAPTURE_DIR` (see [`partial-stateless-exex`](../partial-stateless-exex)).
Each record carries the block's Engine payload, the state it accessed, and a
**policy-neutral full transition witness** — the parent-state proof a validator holding
nothing at all would need, proved against a cold cache and an empty trie.

That witness names no policy, and it is a superset of what any policy's transition can
ask for: every target a warm cache lets a policy skip is a target the cold build already
proved. So a generator can answer any policy's proof requests by selecting from it,
which is exactly what a state database would have returned.

That argument is checked rather than assumed. `tests/live_offline_equivalence.rs` builds
a real parent trie, runs a warm-cache policy build against the whole trie and again
against the recorded subset alone, and requires the two sidecars to be **byte-identical**
— including a block whose transition takes a structural round, which is the case the
superset argument is really about.

## One construction core, two proof sources

Every sidecar this crate produces is built by the same functions a live node builder
uses, in `partial-stateless::transition_build`. The only thing that differs is where a
parent-state proof comes from:

```text
TransitionProofSource
├── RethStateProviderSource     (partial-stateless-exex) — the node's state database
└── RecordedFullWitnessSource   (this crate)             — the recorded full witness
```

Miss-target selection, structural discovery, node selection, trie-cache retention, the
witness commitment, the cache anchors, and the final serialization each have exactly one
implementation. Two builders would make a policy comparison a comparison of builders.

## Usage

```bash
ps-policy-frontier \
  --dataset /path/to/dataset \
  --arm weak --arm 60/30 --arm 90/60 --arm 120/45 \
  --warmup 121 --samples 1000 \
  --out /path/to/frontier-out
```

`--arm weak` is the no-cache baseline: a validator holding nothing when each block arrives.
During input generation it runs in the same rotation over the same blocks and validator
as every policy. Timing comparisons then replay the saved inputs in independent processes. Its sidecar is the recorded policy-neutral full witness — the same one the
block was executed against in step 2 — so Weak uses exactly the witness bytes the corpus was proved with. Without `--arm weak` the summary makes no
Partial-versus-Weak claim; `weak_baseline_present` says which happened.

Per block, in this order:

1. **Admit the payload** through the same untrusted-input boundary a standalone validator
   uses, so the block is derived from the payload rather than asserted by the record. The
   record's own block hash is then checked against it.
2. **Execute once, database-free**, against the recorded full witness — producing the post
   state and access set *this* run observed, neither of which is taken from the record.
3. **Compare that access set against the recorded one.** The producer captured its set from
   a live Engine; this one came from a witness-backed re-execution, possibly on another
   machine. Agreement is the evidence that the corpus describes the block it claims to.
4. **Build each policy's sidecar** through the shared construction core.
5. **Validate each sidecar** against that policy's own validator pair — a pair that has
   never seen anything but sidecars, so the result is a check and not a restatement.

Policy order rotates per block, so no policy keeps the slot that pays the cold read of
whatever the others then find warm.

Warm-up blocks do exactly the same work and simply do not count. The inclusive readiness
window requires `max(account_window, storage_window) + 1` blocks; common warm-up is 121
for 60/30, 90/60, and 120/45. Shorter warm-up is refused.

## Fail-closed inputs

`load_dataset` refuses a missing `END.json`, a capture that recorded its own failure, a
manifest or record from another schema version, a record whose digest does not cover it, a
terminator whose record count or block range disagrees with the files present, a height with
two surviving canonical records, a gap, and a parent hash that does not match the record
below. Each of those has a plausible-looking partial reading, and every partial reading
produces a comparison over a block set the report would describe wrongly.

It also drops every record outside the terminator's `usable_range` — the part the capture
carried far enough past to call settled. A run that stopped early therefore yields a shorter
corpus rather than one whose tail the producer never vouched for, and `--samples` fails
loudly when the confirmed range is too short instead of quietly reaching into it.

`END.json` is therefore necessary but not sufficient. Treat the first short invocation of
this binary as the dataset acceptance gate: the loader scans and verifies every record before
sampling, then the generator must complete database-free execution, access-set comparison,
sidecar construction, and validation for every requested arm.

**Schema-1 datasets are refused, and the reason is a defect in schema 1 rather than in the
data.** Version 1 took a record's digest over its `bincode` serialization, and a record holds
the access set in `HashMap`s whose iteration order is seeded per process and rebuilt on
deserialization. The digest was therefore not a function of the record: it came out one way
when a record was written and another when it was read back, so every schema-1 capture fails
its own integrity check. The first 1,200-block capture died exactly this way — a structurally
valid, 96-confirmation terminator, and a load that stopped at the first populated record.

Version 2 hashes an explicit, sorted, length-prefixed encoding instead, so the digest depends
on what a record contains and nothing else. The loader rejects a schema-1 manifest outright
rather than part-way through, because there is nothing in such a dataset it could check
records against.

## What the output supports, and what it does not

`frontier.jsonl` is one line per block; `frontier-summary.json` totals the measured
population per arm.

The summary names both pieces of code behind its numbers: `dataset_build_commit`, copied from
the corpus's manifest, and `generator_build_commit`, this binary's own. They move independently
— a corpus outlives many generator builds — so a result that named only one could not be traced
back. Either is `null` when that build carried no `PS_BUILD_COMMIT`, which is stamped in at
compile time; a capture refuses to start without one, while this tool records the absence and
runs, so ad-hoc analysis is still possible and still labelled.

The generation report supports same-block sidecar size, cache footprint, cache misses and
witness semantics. Its validation times are **diagnostic core measurements**:
`timing_boundary=generation_core_no_undo`, `operational_latency_eligible=false`.
They omit coordinated undo commit, and share CPU/cache with proof generation and other arms.
Obsolete flat undo is cleared after every successfully generated block.

Production builder latency still requires a live node: selecting recorded witness nodes is
not generating database multiproofs. `builder_latency_eligible` remains false.

Export `--prepare-inputs` and replay each arm with `--validate-inputs` to measure payload
decode through coordinated disk-undo commit in one standalone process per arm. Input file
reads and background writer completion are separate. It is an offline processing boundary;
live delivery and queue latency still come from `ps-replay --follow`.

`sidecar_digest` is over the sidecar's **semantic** content, not its bytes. A serialized
sidecar carries the wall-clock and resource measurements of the machine that built it, so
hashing the bytes would report two hosts as disagreeing about a sidecar they produced
identically. The digest normalizes exactly those fields and covers everything else,
including every size — a witness that is a different size is a different witness.

## Prepared inputs and replay streams

`--prepare-inputs DIR` saves payloads, sidecars, an explicit first-parent trust anchor, ordered
block identities and per-file digests. It publishes each arm's manifest only after all inputs
are saved. `--validate-inputs DIR` verifies these inputs from a cold pair and enforces N+1
warm-up before measurements. Partial uses Exact, recording, disk frames and K=32 by default;
Weak drops its caches after every block. A writer error or final-drain failure fails the pass.
These are benchmark inputs, not authenticated network bootstrap checkpoints.

A normal `ps-replay` spool still requires a proven snapshot package. The policy-neutral
corpus cannot prove all keys in a warmed cache at a single root, so exporting that spool
remains separate work. Prepared validation avoids inventing such a checkpoint.

## Isolation

"Database-free" is a property of the dependency graph, not a convention the code observes.
`crates/partial-stateless-exex/scripts/check_validator_isolation.sh` checks this package
along with the validator, the stream format, and the replay driver: no provider or MDBX
implementation crate is reachable from its normal graph, and the keccak and
signature-recovery features the production binary runs are actually selected.
