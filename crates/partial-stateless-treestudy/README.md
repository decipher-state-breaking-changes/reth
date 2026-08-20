# partial-stateless-treestudy

What a cache-complement witness would cost if Ethereum's state were committed in a different tree.

The partial-stateless system reduces witness transfer by keeping a bounded authenticated cache and
shipping only the complement. Every measured byte of that saving is against the hexary
Merkle-Patricia Trie, because that is what mainnet headers commit to. Two proposals would change the
commitment — [EIP-7864](https://eips.ethereum.org/EIPS/eip-7864)'s unified binary tree and
[EIP-6800](https://eips.ethereum.org/EIPS/eip-6800)'s Verkle tree — and both change the witness by
construction. This crate measures how much of the saving survives.

It is offline, corpus-driven, and touches nothing the live experiment uses. No provider, no state
database, no network, no changes to any other crate.

## What it does and does not measure

The cache policy is defined over addresses and slots, not over tree positions, so **which keys a
block misses is the same number whichever tree the state is committed in**. That is what makes the
comparison possible: the miss set is replayed exactly, through the production `NetworkStateCache` and
`LastNBlocksPolicy` rather than a reimplementation, and only the *cost of a miss* varies between
arms. The run checks that equality against the recorded frontier run rather than assuming it.

- **Measured, not modelled**: the access sequence, the miss set, the tree positions of every key the
  corpus touches, the multiproof structure over those positions, and the MPT arm's witness bytes.
- **Modelled**: the positions of the state the corpus does not name, as a uniform population of the
  measured size. Witness depth grows as `log2` of that size, so `--stems` sweeps it.
- **Not measured at all**: verification time. A hash-based binary proof and an elliptic-curve Verkle
  proof do not have comparable verification costs, and nothing here runs either scheme's
  cryptography.

There is no live arm and cannot be one: no chain commits a binary or Verkle root, so a validator has
no untrusted header to check a recomputed root against. What survives the move offline is everything
that does not need that anchor.

## One embedding per proposal

EIP-7864 and EIP-6800 are not the same tree with a different commitment, and the difference lands on
witness size. EIP-7864 prefixes every key with a storage type, so headers, code, and storage branch
among their own populations rather than among all of them, and its stems are variable-length — 33
bytes in the header and code regions, 65 in storage. EIP-6800 has one region, a fixed 31-byte stem,
and a header layout the EIP fixes at 64 storage slots and 128 code chunks rather than leaving open.
Each proposal therefore gets its own `TreeEmbedding` and the study never shares one between them;
`--header-layout` moves EIP-7864's layout only.

A block's witness is computed from a single target set covering state *and* code. An account's basic
data and its first code chunks share its header stem, so pricing them as two proofs and adding the
totals would charge that stem's outer path and identifier twice. Code is placed per account, not per
code hash: the cache dedups a bytecode by hash, but the trees hold it at each deploying account's own
leaves, and the run reports how many accounts that came to.

## Running it

```bash
cargo build --release -p partial-stateless-treestudy
ps-tree-study --dataset <policy dataset dir> --out <results dir> [--limit N] [--warmup N]
```

Useful options: `--frontier` (the recorded run to check the miss set against), `--stems` (modelled
whole-tree stem count), `--accounts` / `--slots` (measured state sizes), `--code-coverage` (fraction
of a contract's chunks a call runs), `--stem-occupancy` (overrides the occupancy otherwise derived
from `--slots` and `--stems`), `--header-layout` (`table` or `prose` — EIP-7864's constant table and
its prose disagree, so both are runnable), and `--mpt-storage-trie` (the MPT calibration's one fitted
parameter).

Stem occupancy is derived, not defaulted. A state of a given size spread over fewer stems fills each
of them more, so `--slots` and `--stems` together fix the mean occupancy of a storage stem and
sweeping the two independently would price states that do not exist. Untouched-but-existing suffixes
in an opened stem still cost sibling hashes, which is why the derived value is not zero except where
the stems are numerous enough to make it so.

Outputs `blocks.jsonl` (one record per block per arm), `summary.json`, and `census.json`.

## Layout

| Module | Responsibility |
| --- | --- |
| `corpus` | Streams the recorded dataset, resolving the canonical chain from the terminator's tip |
| `keys` | One `TreeEmbedding` per proposal, and the prefix arithmetic they are walked with |
| `population` | The background state per key region, sampled lazily from a deterministic PRF |
| `witness` | The multiproof accounting all backends share: held, derived, empty, carried |
| `mpt` | A hexary path model, kept to calibrate the modelling against the one measured tree |
| `study` | Replaying one corpus through one policy and pricing it under every tree |
| `report` | Totals, paired ratios, and the limits the numbers carry |

## Calibrating the model

The binary and Verkle numbers are modelled and nothing about them is checkable on its own, so the
same path machinery is run over the one tree whose real answer is in hand. The MPT model predicts
node *counts*, not bytes: counts are what the path model produces, and folding in RLP, inline nodes
under 32 bytes, and branch masks would let an encoding error cancel a structural one. It carries one
fitted parameter — a modelled per-account storage-trie size, which the unified trees have no
equivalent of — and a run reports the resulting agreement per arm rather than asserting a figure
here, because it is a property of the corpus and not of the code.

A second check needs no corpus at all: EIP-7864 states a branch of `32 * (k - 1) * log(N) / log(k)`
bytes, which fixes the proof-size reduction of arity 2 over arity 16 at a given state size. The
crate's path-hash total for the no-cache arm is the same quantity derived from real access sets, and
the two should land near each other.
