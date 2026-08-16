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

Also importable, and the RB7 analyzer's front end:
  - `reduce_events(lines)` — the per-cause state dict above
  - `assemble_causes(lines)` — the same log keyed by (epoch, cause_id) and
    (epoch, cause_id, attempt), every lifecycle stamp attached where it belongs, so an
    interval between two of them can be measured without re-parsing the log
"""

import json
import sys

TERMINAL = ("published", "skipped", "fenced", "failed_final", "not_armed", "superseded")
PENDING = ("armed", "in_flight", "retrying")
#: Detection events open a cause with a branch change behind it; everything else opens with its
#: arming line. The distinction is the RB7 denominator's first column.
DETECTION_KINDS = {"reorg_detected": "reorg", "revert_detected": "revert"}


def load_epoch_events(lines):
    """Parses the log, returns (epoch, [events]) for the highest epoch it contains.

    Earlier epochs belong to producers that already exited; their causes cannot become pending
    again, and their monotonic stamps come from a different process origin.

    Each returned event carries a `mono_run` index. The monotonic clock is per-process, so two
    producer runs sharing one epoch number would otherwise let an interval be measured across a
    clock reset. A stamp that runs backwards opens a new run index, and an interval whose ends
    disagree on it is refused rather than reported.
    """
    events = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))
    if not events:
        return None, []
    epoch = max(event.get("epoch", 1) for event in events)
    epoch_events = [event for event in events if event.get("epoch", 1) == epoch]
    mono_run = 0
    highest = -1
    for event in epoch_events:
        mono = event.get("mono_elapsed_us")
        if mono is not None:
            if mono < highest:
                mono_run += 1
                highest = mono
            else:
                highest = max(highest, mono)
        event["mono_run"] = mono_run
    return epoch, epoch_events


def reduce_events(lines):
    """Returns (epoch, {cause_id: state}) for the highest epoch in the log.

    States: armed, in_flight, retrying, published, skipped, fenced, failed_final,
    not_armed, superseded.
    """
    epoch, events = load_epoch_events(lines)
    return epoch, reduce_states(events)


def reduce_states(events):
    """The state machine itself, over already-loaded events of one epoch."""
    causes = {}
    armed_slot = None
    for event in events:
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
    return causes


def assemble_causes(lines):
    """Returns (epoch, {cause_id: cause}) with every lifecycle stamp attached to its cause.

    A cause is the RB7 unit of account: one branch change (or one arming, for the causes that
    have no detection event of their own) with the attempts it armed, the checkpoint it
    published, and the first winning commit it freed. Attempts nest under it keyed by the
    envelope's attempt counter, so a retry's export duration is its own sample rather than an
    average smeared over the cause.

    Every stamp is `{mono_us, wall_ms, mono_run, ...event fields}`. Which clock an interval may
    use is a property of its two ends, not of this function: both ends inside the producer can
    difference `mono_us` (same `mono_run`), and an end that lives in another process has only
    `wall_ms`.
    """
    epoch, events = load_epoch_events(lines)
    states = reduce_states(events)
    causes = {}

    def slot(cause_id):
        return causes.setdefault(
            cause_id,
            {
                "cause_id": cause_id,
                # cause 0 is the stream-opening export: it is never armed and never detected.
                "origin_kind": "initial" if cause_id == 0 else "armed",
                "detection": None,
                "armed": None,
                "attempts": {},
                "published": None,
                "skipped": None,
                "first_winning": None,
                "first_winning_unmeasured": None,
            },
        )

    def stamp(event):
        out = {
            "mono_us": event.get("mono_elapsed_us"),
            "wall_ms": event.get("observed_at_ms"),
            "mono_run": event.get("mono_run", 0),
            "attempt": event.get("attempt"),
        }
        for key, value in event.items():
            if key not in ("schema_version", "benchmark", "kind", "epoch", "mono_run",
                           "mono_elapsed_us", "observed_at_ms", "attempt", "cause_id"):
                out[key] = value
        return out

    for event in events:
        kind = event.get("kind")
        cause_id = event.get("cause_id")
        if cause_id is None:
            continue
        cause = slot(cause_id)
        if kind in DETECTION_KINDS:
            cause["origin_kind"] = DETECTION_KINDS[kind]
            cause["detection"] = stamp(event)
        elif kind == "recheckpoint_armed":
            cause["armed"] = stamp(event)
        elif kind in ("export_started", "export_completed", "export_failed", "export_fenced"):
            attempt = cause["attempts"].setdefault(
                event.get("attempt"),
                {"attempt": event.get("attempt"), "started": None, "completed": None,
                 "failed": None, "fenced": None},
            )
            attempt[kind.removeprefix("export_")] = stamp(event)
        elif kind == "checkpoint_published":
            cause["published"] = stamp(event)
        elif kind == "checkpoint_publication_skipped":
            cause["skipped"] = stamp(event)
        elif kind == "first_winning_commit_published":
            cause["first_winning"] = stamp(event)
        elif kind == "first_winning_commit_unmeasured":
            cause["first_winning_unmeasured"] = stamp(event)

    for cause_id, cause in causes.items():
        cause["state"] = states.get(cause_id, "unknown")
        cause["attempts"] = [
            attempt for _, attempt in sorted(cause["attempts"].items(), key=lambda kv: kv[0] or 0)
        ]
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
    pending = {cause: state for cause, state in causes.items() if state in PENDING}
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
