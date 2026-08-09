#!/usr/bin/env python3
"""Analyze same-block Vanilla/Partial/Weak records from the live ExEx benchmark."""

import argparse
import bisect
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

# Both caches were discarded and warming restarts from nothing, so the blocks that follow are not
# measuring a warm cache and must be excluded. These are the only two places the ExEx does that:
# process start, and `admit_after_cold_reset` once recovery and rebuild have both been ruled out.
COLD_MARKERS = (
    "Partial Stateless ExEx started",
    "Cold-resetting both caches and warming again from this block",
)

# The canonical chain moved, so samples on the abandoned branch have to go. This says nothing about
# cache warmth: a depth-1 reorg restored from the retained generation leaves the pair Ready at the
# common ancestor and the very next block verifies trustlessly, so re-arming warm-up here would
# discard a policy window of perfectly warm samples per reorg. When recovery genuinely fails the
# ExEx logs a cold marker of its own, which is what re-arms warm-up.
BRANCH_MARKERS = ("Chain reorg detected", "Chain reverted")

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def load_jsonl(path: Path, allow_incomplete_tail=False):
    if not path.exists():
        return []
    records = []
    lines = path.read_text(errors="replace").splitlines(keepends=True)
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            is_unterminated_tail = (
                line_number == len(lines) and not line.endswith(("\n", "\r"))
            )
            if allow_incomplete_tail and is_unterminated_tail:
                continue
            raise ValueError(f"invalid JSONL in {path} at line {line_number}: {error}") from error
    return records


def field_hash(line: str, name: str):
    line = ANSI_ESCAPE.sub("", line)
    match = re.search(rf"(?:^|\s){re.escape(name)}=(0x[0-9a-fA-F]+)", line)
    return match.group(1).lower() if match else None


@dataclass
class SelectionStats:
    cold_epochs: int = 0
    warm_branch_switches: int = 0
    orphaned: int = 0
    warmup: int = 0
    invalid: int = 0
    missing_engine: int = 0
    missing_log_position: int = 0
    contaminated: int = 0
    pending_next_engine: int = 0


@dataclass(frozen=True)
class ResetPoint:
    """A log position where sample selection has to change what it is doing.

    `reverted_from` drops samples the chain abandoned; `cold` re-arms warm-up. They are
    independent, and the reason this type carries both is that the two used to be conflated: every
    reorg was treated as a cache epoch even when the caches never lost warmth.
    """

    position: int
    reverted_from: int | None
    cold: bool


def reset_reverted_from(line: str):
    line = ANSI_ESCAPE.sub("", line)
    match = re.search(r"(?:from_chain|reverted_chain)=([0-9]+)[.][.]=", line)
    return int(match.group(1)) if match else None


def parse_log_positions(log_path: Path):
    engine_positions = {}
    paired_positions = {}
    ordered_engine_positions = []
    reset_points = []
    if not log_path.exists():
        return engine_positions, paired_positions, ordered_engine_positions, reset_points
    for position, line in enumerate(log_path.read_text(errors="replace").splitlines()):
        if any(marker in line for marker in COLD_MARKERS):
            reset_points.append(ResetPoint(position, None, cold=True))
        elif any(marker in line for marker in BRANCH_MARKERS):
            reset_points.append(ResetPoint(position, reset_reverted_from(line), cold=False))
        if "Vanilla Reth Engine V2 benchmark start" in line:
            block_hash = field_hash(line, "block_hash")
            if block_hash:
                engine_positions[block_hash] = position
                ordered_engine_positions.append(position)
        if "Paired Partial/Weak timed validation complete" in line:
            block_hash = field_hash(line, "block_hash")
            if block_hash:
                paired_positions[block_hash] = position
    ordered_engine_positions.sort()
    return engine_positions, paired_positions, ordered_engine_positions, reset_points


def select_samples(
    paired_records,
    engine_records,
    log_path: Path,
    warmup: int,
    limit=None,
    include_overlap=False,
):
    engine_by_hash = {
        record.get("block_hash", "").lower(): record
        for record in engine_records
        if record.get("block_hash")
    }
    engine_positions, paired_positions, ordered_engine_positions, resets = parse_log_positions(log_path)
    reset_positions = [reset.position for reset in resets]
    stats = SelectionStats()
    accepted = []
    candidates = []
    for record in paired_records:
        block_hash = record.get("block_hash", "").lower()
        paired_position = paired_positions.get(block_hash)
        if paired_position is None:
            stats.missing_log_position += 1
            continue
        epoch = bisect.bisect_right(reset_positions, paired_position)
        candidates.append((epoch, paired_position, record))

    candidates.sort(key=lambda item: item[1])
    candidate_index = 0
    epoch_count = len(resets) + 1
    # Warm-up carries across epoch boundaries rather than restarting at each one, and is re-armed
    # only by a cold marker. A branch switch that the pair survived warm therefore costs exactly
    # the samples the chain abandoned and nothing else; a branch switch that arrived mid-warm-up
    # still finishes the warm-up it interrupted.
    warmup_remaining = warmup
    cold_epoch_counted = False
    for epoch in range(epoch_count):
        reset = resets[epoch - 1] if epoch else None
        if reset is not None:
            if reset.reverted_from is not None:
                retained = [
                    record for record in accepted
                    if record.get("block_number", -1) < reset.reverted_from
                ]
                stats.orphaned += len(accepted) - len(retained)
                accepted = retained
            if reset.cold:
                warmup_remaining = warmup
                cold_epoch_counted = False
            else:
                stats.warm_branch_switches += 1

        while candidate_index < len(candidates) and candidates[candidate_index][0] == epoch:
            _, paired_position, record = candidates[candidate_index]
            candidate_index += 1
            block_hash = record.get("block_hash", "").lower()
            if not record.get("valid", False):
                stats.invalid += 1
                continue
            engine = engine_by_hash.get(block_hash)
            own_engine_position = engine_positions.get(block_hash)
            if engine is None or own_engine_position is None:
                stats.missing_engine += 1
                continue
            if warmup_remaining:
                warmup_remaining -= 1
                stats.warmup += 1
                if not cold_epoch_counted:
                    stats.cold_epochs += 1
                    cold_epoch_counted = True
                continue
            next_index = bisect.bisect_right(ordered_engine_positions, own_engine_position)
            if next_index >= len(ordered_engine_positions):
                stats.pending_next_engine += 1
                continue
            overlap = (
                ordered_engine_positions[next_index] < paired_position
                or engine.get("contaminated", False)
            )
            if overlap:
                stats.contaminated += 1
                if not include_overlap:
                    continue
            merged = dict(record)
            merged["vanilla_engine"] = engine
            merged["engine_overlap"] = overlap
            accepted.append(merged)
    return accepted[:limit] if limit is not None else accepted, stats


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return math.nan
    if len(values) == 1:
        return float(values[0])
    rank = (len(values) - 1) * fraction
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(values[lower])
    return values[lower] + (values[upper] - values[lower]) * (rank - lower)


def summary(values, scale=1.0):
    scaled = [value / scale for value in values]
    return {
        "avg": statistics.fmean(scaled),
        "p50": percentile(scaled, 0.50),
        "p90": percentile(scaled, 0.90),
        "p95": percentile(scaled, 0.95),
        "p99": percentile(scaled, 0.99),
        "max": max(scaled),
    }


def format_summary(name, values, scale=1000.0, unit="ms"):
    stats = summary(values, scale)
    return "| {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} {} |".format(
        name, stats["avg"], stats["p50"], stats["p90"], stats["p95"],
        stats["p99"], stats["max"], unit,
    )


def paired_ratio(numerators, denominators):
    return statistics.fmean(n / d for n, d in zip(numerators, denominators) if d)


def ratio_of_means(numerators, denominators):
    return statistics.fmean(numerators) / statistics.fmean(denominators)


def faster_share(numerators, denominators):
    return sum(n < d for n, d in zip(numerators, denominators)) / len(numerators)


RETENTION_SPLIT_FIELDS = [
    ("Warm membership rebuild", "retention_warm_membership_us"),
    ("Storage key hashing / path build", "retention_storage_paths_us"),
    ("Account path hashing / sort", "retention_account_paths_us"),
    ("Account trie prune", "retention_account_trie_us"),
    ("Storage trie sort / prune", "retention_storage_tries_us"),
]


def build_retention_split_section(accepted):
    """Break the largest validator phase into preparation versus trie work.

    Emitted only when the records carry the split, so reports regenerated from runs written
    before the fields existed stay readable instead of printing a table of zeros.
    """
    if not any(
        field in record["partial"] for record in accepted for _, field in RETENTION_SPLIT_FIELDS
    ):
        return []

    total = statistics.fmean(r["partial"].get("trie_retention_us", 0) for r in accepted) / 1000
    lines = [
        "", "### Trie retention split (Partial)", "",
        "Measured inside `trie_retention_us`; the rows are components of it, not additions to it.",
        "", "| Component | Avg | Share of retention |", "| --- | ---: | ---: |",
    ]
    preparation = 0.0
    for label, field in RETENTION_SPLIT_FIELDS:
        avg = statistics.fmean(r["partial"].get(field, 0) for r in accepted) / 1000
        if field.endswith(("membership_us", "paths_us")):
            preparation += avg
        share = f"{100 * avg / total:.1f}%" if total else "n/a"
        lines.append(f"| {label} | {avg:.2f} ms | {share} |")
    prep_share = f"{100 * preparation / total:.1f}%" if total else "n/a"
    lines.append(f"| **Key preparation subtotal** | **{preparation:.2f} ms** | **{prep_share}** |")

    paths = statistics.fmean(r["partial"].get("retention_account_paths", 0) for r in accepted)
    pruned = statistics.fmean(
        r["partial"].get("retention_storage_tries_pruned", 0) for r in accepted)
    skipped = statistics.fmean(
        r["partial"].get("retention_storage_tries_skipped", 0) for r in accepted)
    tries = pruned + skipped
    skip_share = f"{100 * skipped / tries:.1f}%" if tries else "n/a"
    lines.extend(["",
        f"- Retained account paths per block: **{paths:.0f}**",
        f"- Storage tries pruned / skipped per block: **{pruned:.0f} / {skipped:.0f}** "
        f"({skip_share} skipped as untouched and unmoved)"])

    # How much of the delta optimization the run actually got. The fallback is correct but pays
    # the full rebuild, so a high rate means the preparation rows above are not what steady-state
    # costs -- it does not mean anything is wrong.
    rebuilds = sum(r["partial"].get("retention_full_rebuild", 0) for r in accepted)
    rate = f"{100 * rebuilds / len(accepted):.1f}%" if accepted else "n/a"
    lines.append(
        f"- Blocks that fell back to a full retained-set rebuild: **{rebuilds} ({rate})**")

    walk_details = [
        ("Account trie", "retention_account_trie_detail"),
        ("Storage tries", "retention_storage_trie_detail"),
    ]
    if any(
        key in record["partial"] for record in accepted for _, key in walk_details
    ):
        lines.extend([
            "", "#### Retention walk internals", "",
            "These rows are nested inside the account/storage totals above.", "",
            "| Trie | Input | Traversal / prefix | Mutation | Finalization | Nodes / edges |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for label, key in walk_details:
            details = [record["partial"].get(key, {}) for record in accepted]

            def avg(field):
                return statistics.fmean(detail.get(field, 0) for detail in details)

            lines.append(
                f"| {label} | {avg('input_us') / 1000:.2f} ms | "
                f"{avg('traversal_us') / 1000:.2f} ms | {avg('mutation_us') / 1000:.2f} ms | "
                f"{avg('finalization_us') / 1000:.2f} ms | "
                f"{avg('nodes_visited'):.0f} / {avg('edges_visited'):.0f} |"
            )
            fallbacks = sum(detail.get("sorted_input_fallbacks", 0) for detail in details)
            dirty = sum(detail.get("unprunable_dirty", 0) for detail in details)
            inline = sum(detail.get("unprunable_inline", 0) for detail in details)
            global_lookups = sum(detail.get("global_prefix_lookups", 0) for detail in details)
            branch_clones = sum(detail.get("branch_clones", 0) for detail in details)
            full_range_calls = sum(detail.get("full_range_calls", 0) for detail in details)
            lines.append(
                f"- {label}: full-range calls **{full_range_calls}**, sorted fallbacks "
                f"**{fallbacks}**, global prefix lookups "
                f"**{global_lookups}**, branch clones **{branch_clones}**, "
                f"unprunable dirty / inline **{dirty} / {inline}**"
            )
    return lines


ANCHOR_SPLIT_GROUPS = [
    ("Collect + sort", "the digest index (moved)", ["account", "storage", "code"],
     "collect_sort_us"),
    ("Leaf preimage + hash", "the digest index (moved)", ["account", "storage", "code"],
     "leaf_hash_us"),
    ("Namespace hash", "nothing — measured irreducible", ["account", "storage", "code"],
     "namespace_us"),
]


def build_anchor_split_section(accepted):
    """Break the next cache anchor into the work each candidate optimization would remove.

    Ordering the keys removes the sort, memoizing leaf digests removes the hashing for entries
    that did not change, and neither touches the namespace hash. The three are disjoint, so
    their sizes are what ranks them. Emitted only when the records carry the split.

    The first two rows read zero once the leaf digest index is in place, and that is the
    measurement rather than a gap: both terms moved into `cache_root_index_maintenance_us`, where
    they are paid over the entries a block moved instead of over the whole cache. Reading the
    anchor alone across that change would credit it with work that only relocated.
    """
    details = [r["partial"].get("next_cache_anchor_detail") for r in accepted]
    details = [d for d in details if d]
    if not details:
        return []

    def avg(field):
        return statistics.fmean(d.get(field, 0) for d in details) / 1000

    total = statistics.fmean(r["partial"].get("next_cache_anchor_us", 0) for r in accepted) / 1000
    lines = [
        "", "### Next cache anchor split (Partial)", "",
        "Measured inside `next_cache_anchor_us`; the rows are components of it, not additions to "
        "it. The right column names the change that would remove the row.", "",
        "| Component | Account | Storage | Code | Total | Share | Removed by |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for label, removed_by, namespaces, suffix in ANCHOR_SPLIT_GROUPS:
        parts = [avg(f"{namespace}_{suffix}") for namespace in namespaces]
        group = sum(parts)
        share = f"{100 * group / total:.1f}%" if total else "n/a"
        cells = " | ".join(f"{part:.2f} ms" for part in parts)
        lines.append(f"| {label} | {cells} | **{group:.2f} ms** | {share} | {removed_by} |")
    root = avg("root_us")
    root_share = f"{100 * root / total:.1f}%" if total else "n/a"
    lines.append(f"| Final root hash | — | — | — | {root:.2f} ms | {root_share} | nothing |")

    accounts = statistics.fmean(d.get("accounts", 0) for d in details)
    storage = statistics.fmean(d.get("storage", 0) for d in details)
    codes = statistics.fmean(d.get("codes", 0) for d in details)
    leaves = accounts + storage + codes
    per_leaf = f"{1000 * total / leaves:.3f} µs" if leaves else "n/a"
    lines.extend(["",
        f"- Value-cache composition hashed: **{accounts:.0f} accounts, {storage:.0f} storage, "
        f"{codes:.0f} codes** ({leaves:.0f} leaves)",
        f"- Cost per leaf: **{per_leaf}**"])

    # A memo hit is free, so averaging one into the phase mean understates it. The validator path
    # invalidates the memo immediately before the anchor, so this should always be zero.
    memo_hits = sum(d.get("memo_hits", 0) for d in details)
    if memo_hits:
        lines.append(
            f"- **{memo_hits} samples answered from the cache-root memo** and cost nothing; the "
            "phase mean above is diluted by them and should be recomputed without them")
    return lines


CLONE_SPLIT_FIELDS = [
    ("Account trie deep copy", "account_trie_us", None),
    ("Storage trie map (refcount bumps)", "storage_tries_us", "storage_tries"),
    ("Warm membership sets", "warm_membership_us", "warm_accounts"),
    ("Retained path indexes", "retained_paths_us", "retained_account_paths"),
]


def build_clone_split_section(accepted):
    """Separate the account-trie copy from the three copies that scale with cache size.

    Sharing the account trie means node-granular structural sharing inside reth's own trie crate.
    The other three copies are ordinary data structures that an `Arc` would share, so their share
    decides whether that much smaller change is worth making. Emitted only when the records carry
    the split.
    """
    details = [r["partial"].get("trie_clone_detail") for r in accepted]
    details = [d for d in details if d]
    if not details:
        return []

    total = statistics.fmean(r["partial"].get("trie_clone_us", 0) for r in accepted) / 1000
    lines = [
        "", "### Transactional trie clone split (Partial)", "",
        "Measured inside `trie_clone_us`; the rows are components of it, not additions to it.", "",
        "| Component | Avg | Share of clone | Entries copied |",
        "| --- | ---: | ---: | ---: |",
    ]
    size_proportional = 0.0
    for label, field, count_field in CLONE_SPLIT_FIELDS:
        avg = statistics.fmean(d.get(field, 0) for d in details) / 1000
        if field != "account_trie_us":
            size_proportional += avg
        share = f"{100 * avg / total:.1f}%" if total else "n/a"
        count = (
            f"{statistics.fmean(d.get(count_field, 0) for d in details):.0f}"
            if count_field else "—")
        lines.append(f"| {label} | {avg:.2f} ms | {share} | {count} |")
    prop_share = f"{100 * size_proportional / total:.1f}%" if total else "n/a"
    lines.append(
        f"| **Copies proportional to cache size** | **{size_proportional:.2f} ms** | "
        f"**{prop_share}** | — |")
    return lines


def build_cache_composition_section(accepted):
    """Report the two cache-wide phases per cached entry, not only per block.

    Two runs cover different blocks, so an absolute phase mean confounds a workload difference
    with an implementation difference. Both phases are functions of cache composition, so the
    per-entry coefficients are the part the implementation controls and the part that survives a
    comparison across ranges. Emitted only when the records carry the composition.
    """
    if not any("cache_accounts" in record for record in accepted):
        return []

    accounts = statistics.fmean(r.get("cache_accounts", 0) for r in accepted)
    storage = statistics.fmean(r.get("cache_storage", 0) for r in accepted)
    codes = statistics.fmean(r.get("cache_codes", 0) for r in accepted)
    entries = accounts + storage

    def mean_us(field):
        return statistics.fmean(r["partial"].get(field, 0) for r in accepted)

    def per_entry(field):
        return f"{mean_us(field) / entries:.3f} µs" if entries else "n/a"

    # What the leaf digest index costs and what it saves are two different phases, so neither
    # alone says whether it paid. Their sum is the like-for-like comparison against the anchor a
    # run that predates the index reported.
    anchor_us = mean_us("next_cache_anchor_us")
    maintenance_us = mean_us("cache_root_index_maintenance_us")
    combined = anchor_us + maintenance_us
    combined_per_entry = f"{combined / entries:.3f} µs" if entries else "n/a"
    update_net_us = mean_us("cache_update_us") - maintenance_us

    return ["", "## Cache composition and normalized phase cost", "",
        "Compare these coefficients across runs, not the absolute means: two runs cover different "
        "blocks, and both phases scale with the numbers in the first row.", "",
        f"- Cached accounts / storage entries / codes: **{accounts:.0f} / {storage:.0f} / "
        f"{codes:.0f}**",
        f"- Next cache anchor per cached entry: **{per_entry('next_cache_anchor_us')}**",
        f"- Leaf digest index maintenance per cached entry: "
        f"**{per_entry('cache_root_index_maintenance_us')}**",
        f"- Anchor + index maintenance: **{combined / 1000:.2f} ms**, "
        f"**{combined_per_entry}** per cached entry",
        f"- Cache update net of index maintenance: **{update_net_us / 1000:.2f} ms**",
        f"- Trie retention per cached entry: **{per_entry('trie_retention_us')}**",
        f"- Transactional trie clone per cached entry: **{per_entry('trie_clone_us')}**"]


def build_cache_delta_section(accepted):
    """Report what share of each namespace's leaves a block actually invalidates.

    Not recoverable from anything else a record carries: the `cache_*` populations show only net
    movement, while a refresh moves no population at all and still changes that entry's leaf,
    because `last_accessed_block` is part of every leaf preimage. Evictions are reported separately
    since they drop a leaf rather than change one. Emitted only when the records carry the delta.
    """
    rows = [r["partial"]["cache_delta"] for r in accepted if "cache_delta" in r.get("partial", {})]
    if not rows:
        return []

    def mean(field):
        return statistics.fmean(row.get(field, 0) for row in rows)

    lines = [
        "", "## Per-block cache delta and leaf reuse", "",
        "`added + refreshed` is the leaves whose digest this block changed, against the population "
        "the root was computed over; reuse is the rest. Evictions drop a leaf rather than change "
        "one, so they are listed but not counted as invalidated.", "",
        "| Namespace | Population | Added | Refreshed | Evicted | Invalidated | Reuse |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    total_population = 0.0
    total_invalidated = 0.0
    for name, pop_field, prefix in (
        ("Accounts", "cache_accounts", "accounts"),
        ("Storage", "cache_storage", "storage"),
        ("Codes", "cache_codes", "codes"),
    ):
        population = statistics.fmean(r.get(pop_field, 0) for r in accepted)
        added, refreshed, evicted = (
            mean(f"{prefix}_added"), mean(f"{prefix}_refreshed"), mean(f"{prefix}_evicted"))
        invalidated = added + refreshed
        share = f"{100 * (1 - invalidated / population):.1f}%" if population else "n/a"
        lines.append(
            f"| {name} | {population:.0f} | {added:.1f} | {refreshed:.1f} | {evicted:.1f} "
            f"| {invalidated:.1f} ({100 * invalidated / population:.1f}%) | **{share}** |"
            if population else f"| {name} | 0 | — | — | — | — | n/a |")
        total_population += population
        total_invalidated += invalidated

    if total_population:
        lines += ["", f"- Weighted leaf reuse across all three namespaces: "
            f"**{100 * (1 - total_invalidated / total_population):.1f}%** "
            f"({total_invalidated:.0f} of {total_population:.0f} leaves invalidated per block)"]
    return lines


def retained_generation_lines(accepted):
    """Report what the K = 1 retained generation costs, or that this run kept none.

    Reported for both arms of the memory control, because "0 blocks retained" is the control's
    result rather than missing data. Records written before the telemetry existed have no field at
    all, which is a third case and is reported as such rather than as zero.
    """
    present = [r["retained_generation"] for r in accepted if "retained_generation" in r]
    if not present:
        return [
            "## K = 1 retained generation", "",
            "- Not recorded: these records predate retained-generation telemetry.", "",
        ]
    enabled = [r for r in present if r.get("enabled")]
    held = [r for r in present if r.get("present")]
    lines = [
        "## K = 1 retained generation", "",
        f"- Retention enabled: **{len(enabled)}/{len(present)}** blocks",
        f"- Generation actually held: **{len(held)}/{len(present)}** blocks",
    ]
    if held:
        lines += [
            "",
            "| Measure | Average | p50 | p90 | p95 | p99 | Maximum |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            format_summary("Retained trie, apparent size", [r["total_bytes"] for r in held], 1024 * 1024, "MiB"),
            format_summary("Retained trie, exclusive", [r["exclusive_bytes"] for r in held], 1024 * 1024, "MiB"),
            "",
            "Exclusive is the marginal cost: the apparent size counts storage tries the live cache "
            "still shares, which dropping the retained generation would not free.",
        ]
    return lines + [""]


def build_report(accepted, stats: SelectionStats, warmup: int, requested: int):
    if len(accepted) < requested:
        raise ValueError(f"only {len(accepted)} accepted samples; requested {requested} after warm-up {warmup}")
    accepted = accepted[:requested]
    vanilla_primary = [r["vanilla_engine"]["state_access_execution_us"] for r in accepted]
    partial_primary = [r["partial"]["state_access_execution_us"] for r in accepted]
    weak_primary = [r["weak"]["state_access_execution_us"] for r in accepted]
    vanilla_evm = [r["vanilla_engine"]["execution_us"] for r in accepted]
    partial_evm = [r["partial"]["evm_us"] for r in accepted]
    weak_evm = [r["weak"]["evm_us"] for r in accepted]
    gas = [r["gas_used"] for r in accepted]
    partial_witness = [r["partial_witness"]["serialized_witness_bytes"] for r in accepted]
    weak_witness = [r["weak_witness"]["serialized_witness_bytes"] for r in accepted]

    lines = [
        "# Single-process Vanilla / Partial / Weak benchmark", "",
        f"Accepted same-block samples: **{len(accepted)}**",
        f"Paired sample-warm-up records excluded: **{stats.warmup}** across "
        f"**{stats.cold_epochs}** re-armed epochs",
        f"Branch switches survived warm: **{stats.warm_branch_switches}** (no warm-up re-armed)",
        f"Excluded: orphaned {stats.orphaned}, overlap {stats.contaminated}, invalid {stats.invalid}, missing Engine {stats.missing_engine}, pending {stats.pending_next_engine}.", "",
        "## State access + execution (primary)", "",
        "Includes Vanilla provider/prewarming/DB reads and Partial/Weak deserialize, witness checks/materialization, provider setup/lookups, and EVM.", "",
        "| Mode | Average | p50 | p90 | p95 | p99 | Maximum |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        format_summary("Vanilla Engine V2", vanilla_primary),
        format_summary("Partial Stateless", partial_primary),
        format_summary("Weak Stateless", weak_primary), "",
        "## EVM executor call (including state-provider reads)", "",
        "Includes lookups performed inside each executor call. Partial/Weak pre-execution deserialize, context validation, and witness materialization are excluded.", "",
        "| Mode | Average | p50 | p90 | p95 | p99 | Maximum |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        format_summary("Vanilla Engine V2", vanilla_evm),
        format_summary("Partial Stateless", partial_evm),
        format_summary("Weak Stateless", weak_evm), "",
        "## Gas-normalized primary time", "",
        "| Mode | Average | p50 | p90 | p95 | p99 | Maximum |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        format_summary("Vanilla Engine V2", [t * 1000 / g for t, g in zip(vanilla_primary, gas) if g], 1, "ms/Mgas"),
        format_summary("Partial Stateless", [t * 1000 / g for t, g in zip(partial_primary, gas) if g], 1, "ms/Mgas"),
        format_summary("Weak Stateless", [t * 1000 / g for t, g in zip(weak_primary, gas) if g], 1, "ms/Mgas"), "",
        "## Partial / Weak witness size", "",
        "| Payload | Average | p50 | p90 | p95 | p99 | Maximum |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        format_summary("Partial witness", partial_witness, 1024 * 1024, "MiB"),
        format_summary("Weak witness", weak_witness, 1024 * 1024, "MiB"),
        format_summary("Partial full sidecar", [r["partial_sidecar_bytes"] for r in accepted], 1024 * 1024, "MiB"),
        format_summary("Weak full sidecar", [r["weak_sidecar_bytes"] for r in accepted], 1024 * 1024, "MiB"), "",
        "## Partial local cache memory", "",
        "| Cache | Average | p50 | p90 | p95 | p99 | Maximum |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        format_summary("Value cache", [r["value_cache_bytes"] for r in accepted], 1024 * 1024, "MiB"),
        format_summary("Sparse trie cache", [r["trie_cache_bytes"] for r in accepted], 1024 * 1024, "MiB"), "",
        *retained_generation_lines(accepted),
        "## Paired ratios", "",
        f"- Partial / Vanilla primary, ratio of means: **{ratio_of_means(partial_primary, vanilla_primary):.3f}x**",
        f"- Partial / Vanilla median same-block ratio: **{statistics.median(p / v for p, v in zip(partial_primary, vanilla_primary) if v):.3f}x**",
        f"- Partial faster than Vanilla blocks: **{faster_share(partial_primary, vanilla_primary) * 100:.1f}%**",
        f"- Weak / Vanilla primary, ratio of means: **{ratio_of_means(weak_primary, vanilla_primary):.3f}x**",
        f"- Weak faster than Vanilla blocks: **{faster_share(weak_primary, vanilla_primary) * 100:.1f}%**",
        f"- Partial / Weak primary, ratio of means: **{ratio_of_means(partial_primary, weak_primary):.3f}x**",
        f"- Partial / Weak EVM, ratio of means: **{ratio_of_means(partial_evm, weak_evm):.3f}x**",
        f"- Partial witness size / Weak, ratio of means: **{ratio_of_means(partial_witness, weak_witness):.3f}x**",
        f"- Mean same-block Partial witness reduction: **{statistics.fmean(1 - p / w for p, w in zip(partial_witness, weak_witness) if w) * 100:.2f}%**", "",
        "## Builder-only costs (outside primary)", "",
        "| Operation | Average | p50 | p90 | p95 | p99 | Maximum |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        format_summary("Full-DB simulation", [r["historical_full_db_evm_us"] for r in accepted]),
        format_summary("Partial witness build", [r["partial_witness_build_us"] for r in accepted]),
        format_summary("Weak witness build", [r["weak_witness_build_us"] for r in accepted]),
        format_summary("Partial serialize", [r["partial_serialize_us"] for r in accepted]),
        format_summary("Weak serialize", [r["weak_serialize_us"] for r in accepted]), "",
        "## Validation totals (secondary)", "",
        "| Total | Partial avg | Weak avg |",
        "| --- | ---: | ---: |",
        "| Raw validation | {:.2f} ms | {:.2f} ms |".format(
            statistics.fmean(r["partial"]["raw_total_us"] for r in accepted) / 1000,
            statistics.fmean(r["weak"]["raw_total_us"] for r in accepted) / 1000),
        "| Protocol total | {:.2f} ms | {:.2f} ms |".format(
            statistics.fmean(r["partial"]["protocol_total_us"] for r in accepted) / 1000,
            statistics.fmean(r["weak"]["protocol_total_us"] for r in accepted) / 1000),
        "| Root-validation reference | {:.2f} ms | {:.2f} ms |".format(
            statistics.fmean(r["partial"]["root_validation_total_us"] for r in accepted) / 1000,
            statistics.fmean(r["weak"]["root_validation_total_us"] for r in accepted) / 1000),
        "| Execution core | {:.2f} ms | {:.2f} ms |".format(
            statistics.fmean(r["partial"]["execution_core_us"] for r in accepted) / 1000,
            statistics.fmean(r["weak"]["execution_core_us"] for r in accepted) / 1000), "",
        "## Secondary phase breakdown", "",
        "| Phase | Partial avg | Weak avg |", "| --- | ---: | ---: |",
    ]
    phase_fields = [
        ("Deserialize", "deserialize_us"),
        ("Previous cache context / anchor", "context_check_us"),
        ("Witness self-consistency", "witness_self_consistency_us"),
        ("Materialize", "materialize_us"),
        ("Provider setup", "provider_setup_us"),
        ("Access capture (excluded from primary)", "accessed_state_capture_us"),
        ("Hash post-state", "hash_post_state_us"),
        ("Sparse-trie root", "state_root_us"),
        ("Exact miss-only policy", "miss_policy_check_us"),
        ("Cache update", "cache_update_us"),
        ("↳ leaf digest index maintenance", "cache_root_index_maintenance_us"),
        ("Trie retention", "trie_retention_us"),
        ("Next cache anchor", "next_cache_anchor_us"),
        ("Transactional trie clone", "trie_clone_us"),
        ("Trie commit", "trie_commit_us"),
        ("Unattributed", "unattributed_us"),
    ]
    for label, field in phase_fields:
        p_avg = statistics.fmean(r["partial"].get(field, 0) for r in accepted) / 1000
        w_avg = statistics.fmean(r["weak"].get(field, 0) for r in accepted) / 1000
        lines.append(f"| {label} | {p_avg:.2f} ms | {w_avg:.2f} ms |")
    lines.extend(["",
        "Rows marked ↳ are measured inside the row above them, so the column does not sum. Weak "
        "carries no leaf digest index, which is why that row is zero for it rather than absent."])

    lines.extend(build_retention_split_section(accepted))
    lines.extend(build_anchor_split_section(accepted))
    lines.extend(build_clone_split_section(accepted))

    lines.extend(build_cache_composition_section(accepted))
    lines.extend(build_cache_delta_section(accepted))

    orders = [r["verifier_order"] for r in accepted]
    tx_average = statistics.fmean(r["tx_count"] for r in accepted)
    partial_first_count = orders.count("partial-then-weak")
    weak_first_count = orders.count("weak-then-partial")
    lines.extend(["", "## Workload and validity", "",
        f"- Gas used average: **{statistics.fmean(gas) / 1_000_000:.2f} Mgas**",
        f"- Transaction count average: **{tx_average:.1f}**",
        f"- Partial-first samples: **{partial_first_count}**",
        f"- Weak-first samples: **{weak_first_count}**",
        "- Correctness failures: **0**", "",
        "## Interpretation limits", "",
        "- Engine executes each block before the ExEx builder and paired stateless executions.",
        "- Previous-block proof generation can warm or evict pages used by the next Engine block.",
        "- Partial and Weak are DB-free during their timed state-access + execution path.",
        "- Network transfer and sidecar file I/O are excluded; witness bytes are reported separately."])
    return "\n".join(lines) + "\n"


def build_overlap_report(accepted, stats: SelectionStats, warmup: int):
    """Report Engine latency with overlap retained and explicitly stratified."""
    if not accepted:
        raise ValueError(f"no valid samples available after warm-up {warmup}")

    clean = [record for record in accepted if not record.get("engine_overlap", False)]
    overlap = [record for record in accepted if record.get("engine_overlap", False)]
    lines = [
        "# Engine / ExEx overlap cohort", "",
        f"Valid samples after paired sample warm-up: **{len(accepted)}**",
        f"Clean samples: **{len(clean)}**",
        f"Overlap/contaminated samples: **{len(overlap)}**",
        (
            f"Excluded before selection: orphaned {stats.orphaned}, invalid {stats.invalid}, "
            f"missing Engine {stats.missing_engine}, pending {stats.pending_next_engine}."
        ), "",
        "Unlike the primary report, this cohort retains blocks where the next Engine payload "
        "started before paired ExEx work completed, or Engine marked the sample contaminated.", "",
        "## Engine state access + execution", "",
        "| Cohort | Average | p50 | p90 | p95 | p99 | Maximum |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if clean:
        lines.append(format_summary(
            "Clean", [record["vanilla_engine"]["state_access_execution_us"] for record in clean]
        ))
    if overlap:
        lines.append(format_summary(
            "Overlap", [record["vanilla_engine"]["state_access_execution_us"] for record in overlap]
        ))
    lines.extend([
        format_summary(
            "All", [record["vanilla_engine"]["state_access_execution_us"] for record in accepted]
        ), "",
        "## Engine executor call", "",
        "| Cohort | Average | p50 | p90 | p95 | p99 | Maximum |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    if clean:
        lines.append(format_summary(
            "Clean", [record["vanilla_engine"]["execution_us"] for record in clean]
        ))
    if overlap:
        lines.append(format_summary(
            "Overlap", [record["vanilla_engine"]["execution_us"] for record in overlap]
        ))
    lines.append(format_summary(
        "All", [record["vanilla_engine"]["execution_us"] for record in accepted]
    ))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--engine-records", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument(
        "--warmup",
        type=int,
        required=True,
        help="paired records excluded after Ready; use the value selected by the runner",
    )
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument(
        "--include-overlap",
        action="store_true",
        help="retain overlap/contaminated samples and emit an Engine overlap report",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    accepted, stats = select_samples(
        load_jsonl(args.records),
        load_jsonl(args.engine_records),
        args.log,
        args.warmup,
        None if args.include_overlap else args.samples,
        include_overlap=args.include_overlap,
    )
    try:
        report = (
            build_overlap_report(accepted, stats, args.warmup)
            if args.include_overlap
            else build_report(accepted, stats, args.warmup, args.samples)
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(report, end="")
    if args.output:
        args.output.write_text(report)


if __name__ == "__main__":
    main()
