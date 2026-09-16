#!/usr/bin/env python3
"""Check forced-reorg replay evidence and the clean build that produced it."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expect(row, **fields):
    for key, expected in fields.items():
        require(key in row and type(row[key]) is type(expected) and row[key] == expected,
                f"{key}: expected {expected!r}, got {row.get(key)!r}")


def check_applied(outcome, attempts, *, at, depth, layout):
    expected_frames = depth if layout == "frames" else depth - 1
    expect(outcome, at=at, depth=depth, outcome="applied", reapplied=depth,
           frames_applied=expected_frames, resumed_identical=True)
    require(outcome["resumed_identical"] is True, "resumed_identical must be true")
    expect(outcome["ancestor"], number=at - depth)
    sequences = outcome["reapplied_sequences"]
    require(len(sequences) == depth and len(set(sequences)) == depth,
            "missing reapplied sequence evidence")
    for offset, sequence in enumerate(sequences):
        rows = attempts[sequence]
        require(len(rows) >= 2, f"sequence {sequence} needs original and reapplied verdicts")
        require(all(row["number"] == at - depth + 1 + offset for row in rows),
                f"reapplied sequence {sequence} names the wrong block")


def check(records, *, at, depth, commits, expected_commit, allocator,
          mode="smoke", refusal_at=None, refusal_depth=None, checkpoint_sequence=None,
          checkpoints_skimmed=None, frames=None, layout=None, retain_depth=3):
    require(depth in (2, 3), "the smoke must consume frames: use depth 2 or 3")
    require(depth <= retain_depth <= 64, "retention must cover the applied reorg and be at most 64")
    require(re.fullmatch(r"[0-9a-f]{40}", expected_commit) is not None,
            "expected_commit must be a full lowercase Git commit hash")
    require(mode in ("smoke", "recovery"), "unknown evidence mode")
    manifests = [row for row in records if row.get("kind") == "run_manifest"]
    reports = [row for row in records if "blocks" in row]
    require(len(manifests) == 1 and len(reports) == 1,
            "expected one manifest and one report; use a fresh output")
    manifest, report = manifests[0], reports[0]
    schedule = [f"{depth}@{at}"]
    if mode == "recovery":
        require(all(value is not None for value in
                    (refusal_at, refusal_depth, checkpoint_sequence, checkpoints_skimmed, frames)),
                "recovery mode needs refusal placement, checkpoint sequence, skim count and frame inventory")
        require(refusal_depth > retain_depth and refusal_at > at,
                "refusal must be later and deeper than retention")
        schedule.append(f"{refusal_depth}@{refusal_at}")
    expect(manifest, benchmark="standalone_replay_v1", retain_depth=retain_depth,
           undo_record=True, forced_reorgs=schedule, allocator=allocator)
    require(manifest["undo_record"] is True, "undo recording must be enabled")
    # Runs written before the layout axis existed used the hybrid layout. The manifest is the
    # authority; an explicit argument pins it when a caller registered the expected arm in advance.
    manifest_layout = manifest.get("undo_layout", "hybrid")
    require(manifest_layout in ("hybrid", "frames"),
            f"unknown undo_layout {manifest_layout!r}")
    layout = manifest_layout if layout is None else layout
    require(manifest_layout == layout,
            f"undo_layout: expected {layout!r}, got {manifest.get('undo_layout')!r}")
    provenance = manifest["provenance"]
    require(provenance.get("build_dirty") is False, "build_dirty must be false")
    expect(provenance, build_commit=expected_commit)
    expect(report, benchmark="standalone_replay_v1", commits=commits,
           failures=0, disagreements=0, mutation_failures=0, winning_branch_incomplete=0,
           skipped_after_fault=0, skipped_awaiting_resync=0)
    for key in ("agreed", "continuous", "complete"):
        require(report.get(key) is True, f"replay {key} is not true")
    blocks = report["blocks"]
    attempts = defaultdict(list)
    for block in blocks:
        expect(block, verdict="accepted")
        require(isinstance(block.get("undo"), dict), "every attempt needs an undo timing block")
        attempts[block["sequence"]].append(block)
    require(len(attempts) == commits, f"expected {commits} distinct verified corpus sequences")
    outcomes = report["forced_reorgs"]
    require(len(outcomes) == len(schedule), "wrong forced-reorg outcome count")
    check_applied(outcomes[0], attempts, at=at, depth=depth, layout=layout)
    if mode == "smoke":
        require(len(blocks) == commits + depth, "wrong original/reapplied attempt count")
        require(all(len(attempts[sequence]) == 2 for sequence in outcomes[0]["reapplied_sequences"]),
                "every reapplied sequence needs exactly two verdicts")
        return

    refused = outcomes[1]
    expect(refused, at=refusal_at, depth=refusal_depth, outcome="refused",
           frames_applied=None, reapplied=0, resumed_identical=None)
    ancestor = refused["ancestor"]
    expect(ancestor, number=refusal_at - refusal_depth)
    skipped = refused["resumed_after_commits"]
    require(type(skipped) is int and skipped > 0, "refusal must skip commits before resuming")
    require(len(report["resyncs"]) == 1, "expected exactly one resync")
    resync = report["resyncs"][0]
    expect(resync, at_sequence=checkpoint_sequence, block=ancestor["number"],
           continuous=True, unverified=None, commits_skipped=0,
           commits_skipped_before_restore=skipped)
    require(resync["continuous"] is True, "resync must be continuous")
    rewind_count = refusal_depth + skipped
    expect(report, rewind_replayed_commits=rewind_count,
           checkpoints_skimmed=checkpoints_skimmed, closed=True)
    require(report["closed"] is True, "recovery corpus must be closed")
    require(len(blocks) == commits + depth + refusal_depth, "wrong recovery attempt count")

    require(all(frame["sequence"] == i for i, frame in enumerate(frames)),
            "frame inventory must be contiguous from sequence zero")
    checkpoint = frames[checkpoint_sequence]
    expect(checkpoint, kind="checkpoint", block=ancestor)
    chunks = checkpoint["snapshot_chunks"]
    require(type(chunks) is int and chunks >= 0, "invalid checkpoint snapshot_chunks")
    resume_at = checkpoint_sequence + 1 + chunks
    require(resume_at < len(frames), "inventory ends inside the recovery checkpoint")
    require(all(frame["kind"] == "snapshot_chunk" for frame in frames[checkpoint_sequence + 1:resume_at]),
            "checkpoint chunks do not cover resume_at")
    corpus_commits = {frame["sequence"]: frame for frame in frames if frame["kind"] == "commit"}
    require(set(attempts) == set(corpus_commits), "not every corpus commit was verified")
    for sequence, rows in attempts.items():
        number = corpus_commits[sequence]["block"]["number"]
        require(all(row["number"] == number for row in rows), "verdict block differs from inventory")
    firing = next(i for i, block in enumerate(blocks) if block["sequence"] == refused["sequence"])
    expect(blocks[firing], number=refusal_at)
    window = blocks[firing + 1:firing + 1 + rewind_count]
    require(len(window) == rewind_count, "rewind verdicts are missing")
    first_sequence = checkpoint_sequence - rewind_count
    parent_hash = ancestor["hash"]
    for offset, block in enumerate(window):
        sequence = first_sequence + offset
        expect(block, sequence=sequence, number=ancestor["number"] + 1 + offset)
        frame = corpus_commits[sequence]
        expect(frame, parent_hash=parent_hash)
        parent_hash = frame["block"]["hash"]
    after = blocks[firing + 1 + rewind_count:]
    expected_after = [seq for seq in corpus_commits if seq >= resume_at]
    require(expected_after and [block["sequence"] for block in after] == expected_after,
            "every commit past resume_at must be verified in order")


def read_jsonl(path):
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--mode", choices=("smoke", "recovery"), default="smoke")
    parser.add_argument("--at", type=int, required=True)
    parser.add_argument("--depth", type=int, default=2, choices=(2, 3))
    parser.add_argument("--retain-depth", type=int, default=3,
                        help="expected retention; defaults to 3 for historical smoke reports")
    parser.add_argument("--commits", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--allocator", choices=("jemalloc", "system", "snmalloc"), default="jemalloc")
    parser.add_argument("--layout", choices=("hybrid", "frames"))
    parser.add_argument("--refusal-at", type=int)
    parser.add_argument("--refusal-depth", type=int)
    parser.add_argument("--checkpoint-sequence", type=int)
    parser.add_argument("--checkpoints-skimmed", type=int)
    parser.add_argument("--frames", type=Path)
    args = parser.parse_args()
    try:
        check(read_jsonl(args.report), at=args.at, depth=args.depth, commits=args.commits,
              expected_commit=args.expected_commit, allocator=args.allocator, mode=args.mode,
              refusal_at=args.refusal_at, refusal_depth=args.refusal_depth,
              checkpoint_sequence=args.checkpoint_sequence, checkpoints_skimmed=args.checkpoints_skimmed,
              frames=read_jsonl(args.frames) if args.frames else None, layout=args.layout,
              retain_depth=args.retain_depth)
    except (ValueError, KeyError, TypeError, IndexError, StopIteration, OSError) as error:
        parser.exit(1, f"forced-reorg evidence failed: {error}\n")
    print(f"forced-reorg {args.mode} evidence passed: {args.commits} commits")


if __name__ == "__main__":
    main()
