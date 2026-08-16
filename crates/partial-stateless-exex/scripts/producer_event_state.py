#!/usr/bin/env python3
"""Reduce a producer-events JSONL into per-cause lifecycle state.

The gate's quiesce decision used to compare grep counts of started vs settled events, which
has a real race: an `export_failed` with retries left *is* a settle by count, but the cause is
still pending — the retry spawns at the next Ready, and a SIGTERM inside that gap loses the
recovery checkpoint the run owes. This reducer replays the event log as the state machine it
records: one slot of armed cause, attempts per (epoch, cause_id), and a terminal state per
cause that says whether waiting can still change anything.

Contract (current epoch only — earlier epochs belong to producers that already exited):
  - `recheckpoint_armed {cause_id, armed}` opens a cause. armed=false (policy `never`) is
    terminal immediately. A later `recheckpoint_armed` supersedes an armed cause that never
    started an attempt: the pending slot holds one cause at a time.
  - `export_started {cause_id, attempt}` moves the cause in flight.
  - `export_failed {cause_id, retries_left}` with retries_left > 0 keeps the cause pending
    (a fresh attempt arms); with retries_left == 0 it is terminal.
  - `checkpoint_published` / `checkpoint_publication_skipped` / `export_fenced` are terminal
    for their cause_id (a fenced cause's successor arrives as its own recheckpoint_armed).
  - The initial export (cause_id 0) opens with its own export_started; it never arms.

Quiesced means: no armed cause without an attempt, no attempt in flight, no failed cause with
retries left. Exit 0 quiesced, 1 pending, 2 usage/parse error.

Also importable: `reduce_events(lines)` returns the per-cause state dict for the RB7 analyzer.
"""

import json
import sys

TERMINAL = ("published", "skipped", "fenced", "failed_final", "not_armed", "superseded")


def reduce_events(lines):
    """Returns (epoch, {cause_id: state}) for the highest epoch in the log.

    States: armed, in_flight, retrying, published, skipped, fenced, failed_final,
    not_armed, superseded.
    """
    events = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))
    if not events:
        return None, {}
    epoch = max(event.get("epoch", 1) for event in events)
    causes = {}
    armed_slot = None
    for event in events:
        if event.get("epoch", 1) != epoch:
            continue
        kind = event.get("kind")
        cause = event.get("cause_id")
        if kind == "recheckpoint_armed":
            # One slot: a cause that was armed but never attempted is superseded by the next.
            if armed_slot is not None and causes.get(armed_slot) == "armed":
                causes[armed_slot] = "superseded"
            if event.get("armed", True):
                causes[cause] = "armed"
                armed_slot = cause
            else:
                causes[cause] = "not_armed"
                armed_slot = None
        elif kind == "export_started":
            causes[cause] = "in_flight"
            if armed_slot == cause:
                armed_slot = None
        elif kind == "export_failed":
            if event.get("retries_left", 0) > 0:
                causes[cause] = "retrying"
            else:
                causes[cause] = "failed_final"
        elif kind == "checkpoint_published":
            causes[cause] = "published"
        elif kind == "checkpoint_publication_skipped":
            causes[cause] = "skipped"
        elif kind == "export_fenced":
            causes[cause] = "fenced"
    return epoch, causes


def main():
    if len(sys.argv) != 2:
        print("usage: producer_event_state.py <producer-events.jsonl>", file=sys.stderr)
        return 2
    try:
        with open(sys.argv[1]) as handle:
            epoch, causes = reduce_events(handle)
    except (OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    pending = {
        cause: state for cause, state in causes.items() if state in ("armed", "in_flight", "retrying")
    }
    print(
        json.dumps(
            {
                "epoch": epoch,
                "quiesced": not pending,
                "pending": pending,
                "causes": causes,
            },
            sort_keys=True,
        )
    )
    return 0 if not pending else 1


if __name__ == "__main__":
    sys.exit(main())
