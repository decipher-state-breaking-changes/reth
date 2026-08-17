#!/usr/bin/env python3
"""Turn a window sweep's miss counts into an estimated witness size, and pick candidates.

    estimate_witness_bytes.py --sweep sweep.csv \
        --fit <run>/producer.out --check <other-run>/producer.out \
        --baseline 60,30 [--json out.json]

`cache_window_bench` answers how often a window misses. It cannot answer what a miss costs to
carry, because it never builds a proof. This joins the two: a per-category size model fitted on
runs that really did build proofs, applied to the sweep's miss counts.

**These bytes are estimates and must never be cited as measurements.** A multiproof is a tree,
not a list: siblings are shared between targets, so the second account in a subtree costs a
fraction of the first, and where the targets land decides how much. Code bytes depend on *which*
contracts a block touched rather than how many. The model below is deliberately simple enough to
be wrong in a stated direction — it is a way to rank candidates for a live screen, and the live
screen is what produces a number.

## Pre-registered decision rule

Written before any curve from this project's own capture was looked at, which is the only thing
that makes it a rule rather than a description of whatever the data happened to show.

  * A candidate window *qualifies* when both hold:
      - its estimated witness bytes per block are at most 85% of the baseline's, and
      - its average cache footprint is at most twice the baseline's.
  * Among qualifying candidates, take the one with the **smallest account window**, not the one
    with the fewest bytes: memory is paid continuously by every validator, bytes are paid per
    block by the network, and the smallest window that buys the byte reduction is the one that
    spends least to get it.
  * If nothing qualifies, the operating point stays at the baseline.

The knee is reported separately and decides nothing: along the baseline's own account:storage
ratio, it is the smallest window past which one more step buys less than 3% of the baseline's
estimated bytes. Only along that ratio — a grid walked in two dimensions at once has no knee,
because "one more step" would mean two different things depending on which column moved.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, field

ANSI = re.compile(r"\x1b\[[0-9;]*m")

# The two producer log lines this reads, joined on the block they both name.
MISS_FIELDS = ("missed_accounts", "missed_storage", "missed_codes")
SIZE_FIELDS = ("account_proof_bytes", "storage_proof_bytes", "bytecode_bytes")

CATEGORIES = (
    ("account", "missed_accounts", "account_proof_bytes"),
    ("storage", "missed_storage", "storage_proof_bytes"),
    ("code", "missed_codes", "bytecode_bytes"),
)

QUALIFY_BYTES_FRACTION = 0.85
QUALIFY_MEMORY_FACTOR = 2.0
KNEE_MARGINAL_FRACTION = 0.03

REQUIRED_SWEEP_COLUMNS = (
    "account_window", "storage_window", "measured_blocks",
    "acc_accessed", "acc_hit", "sto_accessed", "sto_hit", "code_accessed", "code_hit",
    "avg_cache_mem_bytes",
)


@dataclass
class Observation:
    """One block that was really executed, with what it missed and what that cost."""

    block: int
    missed_accounts: int
    missed_storage: int
    missed_codes: int
    account_proof_bytes: int
    storage_proof_bytes: int
    bytecode_bytes: int


@dataclass
class Model:
    """Two shapes for one category, and how wrong each was on data it never saw."""

    category: str
    linear_k: float
    power_k: float
    power_p: float
    fit_n: int
    linear_fit_error: float | None = None
    power_fit_error: float | None = None
    linear_check_error: float | None = None
    power_check_error: float | None = None
    chosen: str = "linear"
    notes: list[str] = field(default_factory=list)

    def predict(self, misses: float) -> float:
        if misses <= 0:
            return 0.0
        if self.chosen == "power":
            return self.power_k * (misses ** self.power_p)
        return self.linear_k * misses


def parse_producer_log(path: str) -> list[Observation]:
    """Reads the miss line and the witness line and joins them per block.

    Both are emitted once per built sidecar and neither carries the other's fields, so a block
    that produced only one of them is dropped rather than half-counted.
    """
    misses: dict[int, dict[str, int]] = {}
    sizes: dict[int, dict[str, int]] = {}
    with open(path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = ANSI.sub("", raw)
            if "missed_accounts=" not in line and "witness_total_bytes=" not in line:
                continue
            fields = dict(
                token.split("=", 1)
                for token in line.split()
                if "=" in token and token.split("=", 1)[1].lstrip("-").isdigit()
            )
            block = fields.get("block")
            if block is None:
                continue
            block = int(block)
            if all(name in fields for name in MISS_FIELDS):
                misses[block] = {name: int(fields[name]) for name in MISS_FIELDS}
            if all(name in fields for name in SIZE_FIELDS):
                sizes[block] = {name: int(fields[name]) for name in SIZE_FIELDS}

    observations = []
    for block in sorted(set(misses) & set(sizes)):
        observations.append(
            Observation(
                block=block,
                **misses[block],
                **sizes[block],
            )
        )
    return observations


def fit_through_origin(points: list[tuple[float, float]]) -> float:
    """Least squares for `y = k*x`, which is the shape a per-item cost would have."""
    numerator = sum(x * y for x, y in points)
    denominator = sum(x * x for x, _ in points)
    return numerator / denominator if denominator else 0.0


def fit_power(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Least squares for `y = k*x^p` in log space.

    `p < 1` is the sharing the tree does: the second target in a subtree carries fewer new nodes
    than the first. Reporting the exponent is most of the value of fitting this at all.
    """
    usable = [(math.log(x), math.log(y)) for x, y in points if x > 0 and y > 0]
    if len(usable) < 2:
        return 0.0, 1.0
    n = len(usable)
    mean_x = sum(x for x, _ in usable) / n
    mean_y = sum(y for _, y in usable) / n
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in usable)
    variance = sum((x - mean_x) ** 2 for x, _ in usable)
    slope = covariance / variance if variance else 1.0
    intercept = mean_y - slope * mean_x
    return math.exp(intercept), slope


def median_relative_error(
    points: list[tuple[float, float]], predict
) -> float | None:
    errors = [abs(predict(x) - y) / y for x, y in points if y > 0]
    if not errors:
        return None
    errors.sort()
    middle = len(errors) // 2
    if len(errors) % 2:
        return errors[middle]
    return (errors[middle - 1] + errors[middle]) / 2


def build_models(
    fit: list[Observation], check: list[Observation]
) -> dict[str, Model]:
    models = {}
    for category, miss_field, byte_field in CATEGORIES:
        fit_points = [
            (float(getattr(o, miss_field)), float(getattr(o, byte_field))) for o in fit
        ]
        check_points = [
            (float(getattr(o, miss_field)), float(getattr(o, byte_field))) for o in check
        ]
        linear_k = fit_through_origin(fit_points)
        power_k, power_p = fit_power(fit_points)
        model = Model(
            category=category,
            linear_k=linear_k,
            power_k=power_k,
            power_p=power_p,
            fit_n=len(fit_points),
        )
        model.linear_fit_error = median_relative_error(fit_points, lambda x: linear_k * x)
        model.power_fit_error = median_relative_error(
            fit_points, lambda x: power_k * (x ** power_p) if x > 0 else 0.0
        )
        if check_points:
            model.linear_check_error = median_relative_error(
                check_points, lambda x: linear_k * x
            )
            model.power_check_error = median_relative_error(
                check_points, lambda x: power_k * (x ** power_p) if x > 0 else 0.0
            )
            # Chosen on data the fit never saw. Choosing on the fit set would pick the shape with
            # more freedom every time, which is how an estimator learns a run instead of a rule.
            if (
                model.power_check_error is not None
                and model.linear_check_error is not None
                and model.power_check_error < model.linear_check_error
            ):
                model.chosen = "power"
        else:
            model.notes.append(
                "no held-out run: the shape was chosen by default, not by evidence"
            )
        if model.power_p < 1.0:
            model.notes.append(
                f"exponent {model.power_p:.2f} < 1 — the tree is sharing nodes between targets, "
                "so a linear model overestimates a wider window's savings"
            )
        elif model.power_p > 1.05:
            model.notes.append(
                f"exponent {model.power_p:.2f} > 1 — this category is not sized by how many "
                "items were missed but by which ones, so the count is a proxy and the fit is "
                "the weakest of the three"
            )
        models[category] = model
    return models


def read_sweep(path: str) -> list[dict[str, float]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in REQUIRED_SWEEP_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(
                f"{path} is missing {', '.join(missing)} — it was written by a "
                "cache_window_bench that only recorded hit percentages. A percentage cannot be "
                "turned back into a miss count; re-run the sweep with a current build."
            )
        return [{k: float(v) for k, v in row.items()} for row in reader]


def estimate_rows(sweep: list[dict[str, float]], models: dict[str, Model]) -> list[dict]:
    rows = []
    for row in sweep:
        blocks = row["measured_blocks"] or 1
        misses = {
            "account": (row["acc_accessed"] - row["acc_hit"]) / blocks,
            "storage": (row["sto_accessed"] - row["sto_hit"]) / blocks,
            "code": (row["code_accessed"] - row["code_hit"]) / blocks,
        }
        per_category = {
            name: models[name].predict(value) for name, value in misses.items()
        }
        rows.append(
            {
                "account_window": int(row["account_window"]),
                "storage_window": int(row["storage_window"]),
                "measured_blocks": int(row["measured_blocks"]),
                "misses_per_block": misses,
                "estimated_bytes": per_category,
                "estimated_bytes_total": sum(per_category.values()),
                "avg_cache_mem_bytes": row["avg_cache_mem_bytes"],
                "overall_hit_pct": row.get("overall_hit_pct"),
            }
        )
    return rows


def apply_rule(rows: list[dict], baseline: tuple[int, int]) -> dict:
    """The pre-registered rule in the module docstring, applied mechanically."""
    base = next(
        (
            r
            for r in rows
            if (r["account_window"], r["storage_window"]) == baseline
        ),
        None,
    )
    if base is None:
        raise SystemExit(
            f"the sweep has no baseline row for {baseline[0]}/{baseline[1]}; "
            "the rule is relative to it and cannot be applied without it"
        )

    qualifying = [
        r
        for r in rows
        if r is not base
        and r["estimated_bytes_total"] <= QUALIFY_BYTES_FRACTION * base["estimated_bytes_total"]
        and r["avg_cache_mem_bytes"] <= QUALIFY_MEMORY_FACTOR * base["avg_cache_mem_bytes"]
    ]
    qualifying.sort(key=lambda r: (r["account_window"], r["storage_window"]))
    chosen = qualifying[0] if qualifying else base

    # The knee: the first window past which one more step buys less than the threshold. Walked
    # along the baseline's own account:storage ratio and nowhere else — the grid is a cross
    # product, so a lexicographic walk over it would compare 8/8 against 15/4 and call the
    # difference a step. Reported, never decisive: a knee is a property of the grid's spacing as
    # much as of the curve.
    diagonal = sorted(
        (
            r
            for r in rows
            if r["storage_window"] * base["account_window"]
            == base["storage_window"] * r["account_window"]
        ),
        key=lambda r: r["account_window"],
    )
    knee = None
    knee_status = "too few windows share the baseline's ratio to look for one"
    if len(diagonal) >= 3:
        knee_status = "still paying at the widest window in this grid — the grid is the limit, not the curve"
        for previous, nxt in zip(diagonal, diagonal[1:]):
            gain = previous["estimated_bytes_total"] - nxt["estimated_bytes_total"]
            if gain < KNEE_MARGINAL_FRACTION * base["estimated_bytes_total"]:
                knee = previous
                knee_status = "found"
                break

    return {
        "baseline": [base["account_window"], base["storage_window"]],
        "baseline_estimated_bytes": base["estimated_bytes_total"],
        "qualifying": [[r["account_window"], r["storage_window"]] for r in qualifying],
        "decision": [chosen["account_window"], chosen["storage_window"]],
        "decision_is_baseline": chosen is base,
        "knee": [knee["account_window"], knee["storage_window"]] if knee else None,
        "knee_status": knee_status,
        "ratio_windows": [[r["account_window"], r["storage_window"]] for r in diagonal],
        "thresholds": {
            "bytes_fraction": QUALIFY_BYTES_FRACTION,
            "memory_factor": QUALIFY_MEMORY_FACTOR,
            "knee_marginal_fraction": KNEE_MARGINAL_FRACTION,
        },
    }


def render(models, rows, rule, fit_paths, check_paths) -> str:
    out = []
    out.append("# Estimated witness size by cache window")
    out.append("")
    out.append("**Estimates, not measurements.** A multiproof shares sibling nodes between")
    out.append("targets, so bytes are sublinear in miss count and the sublinearity depends on")
    out.append("where the targets land in the trie. Rank candidates with this; measure the one")
    out.append("you pick.")
    out.append("")
    out.append("## Size model")
    out.append("")
    out.append(f"Fitted on: {', '.join(fit_paths) or '(none)'}")
    out.append(f"Held out:  {', '.join(check_paths) or '(none — no error bar)'}")
    out.append("")
    out.append("| category | shape | k | exponent | median error (fit) | median error (held out) |")
    out.append("|----------|-------|---|----------|--------------------|-------------------------|")
    for name, _, _ in CATEGORIES:
        m = models[name]
        fit_err = "—" if m.linear_fit_error is None else (
            f"{(m.power_fit_error if m.chosen == 'power' else m.linear_fit_error) * 100:.1f}%"
        )
        check_err = "—"
        chosen_check = m.power_check_error if m.chosen == "power" else m.linear_check_error
        if chosen_check is not None:
            check_err = f"{chosen_check * 100:.1f}%"
        k = m.power_k if m.chosen == "power" else m.linear_k
        exponent = f"{m.power_p:.3f}" if m.chosen == "power" else "1 (fixed)"
        out.append(f"| {name} | {m.chosen} | {k:,.1f} | {exponent} | {fit_err} | {check_err} |")
    out.append("")
    for name, _, _ in CATEGORIES:
        for note in models[name].notes:
            out.append(f"- {name}: {note}")
    out.append("")
    out.append("## Estimated cost per window")
    out.append("")
    out.append("| account/storage | hit % | misses/block (a/s/c) | est. bytes/block | vs baseline | avg cache MiB |")
    out.append("|-----------------|-------|----------------------|------------------|-------------|---------------|")
    base_bytes = rule["baseline_estimated_bytes"]
    for r in sorted(rows, key=lambda r: (r["account_window"], r["storage_window"])):
        m = r["misses_per_block"]
        delta = (r["estimated_bytes_total"] / base_bytes - 1) * 100 if base_bytes else 0.0
        marker = " **<-**" if [r["account_window"], r["storage_window"]] == rule["decision"] else ""
        out.append(
            f"| {r['account_window']}/{r['storage_window']}{marker} "
            f"| {r['overall_hit_pct'] or 0:.1f} "
            f"| {m['account']:.0f}/{m['storage']:.0f}/{m['code']:.0f} "
            f"| {r['estimated_bytes_total']:,.0f} "
            f"| {delta:+.1f}% "
            f"| {r['avg_cache_mem_bytes'] / (1024 * 1024):.1f} |"
        )
    out.append("")
    out.append("## Pre-registered rule")
    out.append("")
    out.append(
        f"Qualify at <= {QUALIFY_BYTES_FRACTION:.0%} of the baseline's bytes and "
        f"<= {QUALIFY_MEMORY_FACTOR:g}x its memory; among those take the smallest account window."
    )
    out.append("")
    out.append(f"- baseline: {rule['baseline'][0]}/{rule['baseline'][1]}")
    out.append(
        "- qualifying: "
        + (", ".join(f"{a}/{s}" for a, s in rule["qualifying"]) or "none")
    )
    out.append(f"- **decision: {rule['decision'][0]}/{rule['decision'][1]}**"
               + (" (unchanged)" if rule["decision_is_baseline"] else ""))
    out.append(
        "- knee along the baseline's ratio (reported, not decisive): "
        + (
            f"{rule['knee'][0]}/{rule['knee'][1]}"
            if rule["knee"]
            else rule["knee_status"]
        )
        + (
            ""
            if rule["knee"]
            else " ["
            + " -> ".join(f"{a}/{s}" for a, s in rule["ratio_windows"])
            + "]"
        )
    )
    out.append("")
    out.append("The decision names a candidate for a live confirmation screen. It is not a result.")
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", required=True, help="cache_window_bench CSV")
    parser.add_argument("--fit", action="append", default=[], help="producer log to fit on")
    parser.add_argument("--check", action="append", default=[], help="producer log held out")
    parser.add_argument("--baseline", default="60,30")
    parser.add_argument("--json", help="write the stable JSON here")
    parser.add_argument("--out", help="write the report here instead of stdout")
    args = parser.parse_args(argv)

    if not args.fit:
        raise SystemExit("--fit needs at least one producer log; there is no size model without one")

    fit = [o for path in args.fit for o in parse_producer_log(path)]
    check = [o for path in args.check for o in parse_producer_log(path)]
    if not fit:
        raise SystemExit(
            "no block in the fit logs carried both a miss line and a witness line; "
            "was this a run that built sidecars?"
        )

    models = build_models(fit, check)
    rows = estimate_rows(read_sweep(args.sweep), models)
    account, storage = (int(part) for part in args.baseline.split(","))
    rule = apply_rule(rows, (account, storage))
    report = render(models, rows, rule, args.fit, args.check)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": 1,
                    "status": "exploratory-estimate-never-cite-as-measurement",
                    "fit_blocks": len(fit),
                    "check_blocks": len(check),
                    "models": {
                        name: {
                            "chosen": m.chosen,
                            "linear_k": m.linear_k,
                            "power_k": m.power_k,
                            "power_p": m.power_p,
                            "linear_check_error": m.linear_check_error,
                            "power_check_error": m.power_check_error,
                            "notes": m.notes,
                        }
                        for name, m in models.items()
                    },
                    "windows": rows,
                    "rule": rule,
                },
                handle,
                indent=1,
            )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(report + "\n")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
