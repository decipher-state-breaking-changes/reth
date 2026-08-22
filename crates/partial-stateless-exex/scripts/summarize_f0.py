#!/usr/bin/env python3
"""Render F0 follower distributions and producer I/O totals without hiding skew."""

from __future__ import annotations

import json
import pathlib
import re
import sys
from collections.abc import Iterable, Sequence
from typing import Any


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PRODUCER_FIELDS = ("frames", "frame_write_us", "frame_fsync_us", "dir_syncs")


def format_number(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{value:,.0f}"


def distribution(metric: dict[str, Any]) -> str:
    """Keep a percentile beside its mean so a mixed population is visible."""
    if not metric:
        return "—"
    return "/".join(format_number(metric.get(field)) for field in ("mean", "p50"))


def ack_distribution(metric: dict[str, Any]) -> str:
    if not metric:
        return "—"
    return "/".join(format_number(metric.get(field)) for field in ("mean", "p50", "p95"))


def parse_producer_close(path: pathlib.Path) -> tuple[str, dict[str, int] | None]:
    """Return one successful close summary; zero or several are not silently selected."""
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return "missing", None
    closes = [ANSI_ESCAPE.sub("", line) for line in lines if "Closed the event stream" in line]
    if len(closes) != 1:
        return ("missing" if not closes else f"multiple({len(closes)})"), None

    line = closes[0]
    kind = re.search(r'\bkind="([^"]+)"', line)
    parsed: dict[str, int] = {}
    for field in PRODUCER_FIELDS:
        match = re.search(rf"\b{field}=(\d+)\b", line)
        if match is None:
            return "malformed", None
        parsed[field] = int(match.group(1))
    return (kind.group(1) if kind else "malformed"), parsed


def render_table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    rows = list(rows)
    widths = [max(len(str(row[index])) for row in [header, *rows]) for index in range(len(header))]
    line = lambda row: "  ".join(str(cell).ljust(width) for cell, width in zip(row, widths))
    return "\n".join(
        [line(header), "  ".join("-" * width for width in widths), *(line(row) for row in rows)]
    )


def summarize(base: pathlib.Path) -> str:
    follow_rows: list[tuple[str, ...]] = []
    producer_rows: list[tuple[str, ...]] = []
    for dist in sorted(base.glob("*/out/distributions.json")):
        arm_dir = dist.parent.parent
        arm = arm_dir.name
        try:
            data = json.loads(dist.read_text())
        except (OSError, json.JSONDecodeError) as err:
            follow_rows.append((arm, f"unreadable: {err}", "", "", "", ""))
        else:
            follow = data.get("follow") or {}
            summary = follow.get("summary") or {}
            steady = (follow.get("populations") or {}).get("phase:steady") or {}
            metrics = steady.get("metrics") or {}
            follow_rows.append(
                (
                    arm,
                    format_number(summary.get("blocks_verified")),
                    f"{summary.get('reorgs_applied', '—')}/{summary.get('reverts_applied', '—')}",
                    distribution(metrics.get("standalone_validation_us") or {}),
                    distribution(metrics.get("decision_latency_us[mtime]") or {}),
                    ack_distribution(summary.get("ack_write_us") or {}),
                )
            )

        close_kind, totals = parse_producer_close(arm_dir / "producer.out")
        if totals is None:
            producer_rows.append((arm, "—", "—", "—", "—", "—", close_kind))
            continue
        frames = totals["frames"]
        per_frame = lambda field: "—" if frames == 0 else format_number(totals[field] / frames)
        producer_rows.append(
            (
                arm,
                format_number(frames),
                format_number(totals["frame_write_us"]),
                format_number(totals["frame_fsync_us"]),
                per_frame("frame_write_us"),
                per_frame("frame_fsync_us"),
                f"{close_kind}; dir_syncs={totals['dir_syncs']}",
            )
        )

    if not follow_rows:
        return f"no F0 distributions found under {base}"

    follow = render_table(
        (
            "arm",
            "blocks",
            "reorg/revert",
            "primary mean/p50 us",
            "latency mean/p50 us",
            "ack write mean/p50/p95 us",
        ),
        follow_rows,
    )
    producer = render_table(
        (
            "arm",
            "frames",
            "write total us*",
            "sync total us",
            "write/frame us*",
            "sync/frame us",
            "producer close",
        ),
        producer_rows,
    )
    note = (
        "* write time includes serialization plus file write and is not a disk-media metric; "
        "sync/frame is fsync wall time."
    )
    return f"follower distributions\n{follow}\n\nproducer totals\n{producer}\n{note}"


def main(argv: Sequence[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <f0-run-dir>", file=sys.stderr)
        return 2
    print(summarize(pathlib.Path(argv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
