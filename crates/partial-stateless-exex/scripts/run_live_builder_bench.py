#!/usr/bin/env python3
"""Run a bounded ordinary-builder benchmark without validator/preflight snapshots."""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from analyze_builder_bench import build_builder_report, select_builder_samples
from analyze_validation_bench import load_jsonl
from run_live_paired_bench import (
    DISABLED_DIAGNOSTICS,
    append_resource_sample,
    build_command,
    prepare_output,
    stop_process,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reth-bin",
        type=Path,
        default=Path("./target/release/reth-partial-stateless"),
    )
    parser.add_argument("--datadir", type=Path, required=True)
    parser.add_argument("--jwtsecret", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--shutdown-timeout", type=float, default=120.0)
    parser.add_argument(
        "--parallel-initial-proof",
        choices=("off", "on"),
        default="off",
        help="set PS_PARALLEL_INITIAL_PROOF deterministically (default: off)",
    )
    parser.add_argument(
        "--canonical-rebuild",
        choices=("off", "on"),
        default="off",
        help=(
            "set PS_CANONICAL_REBUILD deterministically (default: off). On, a cold or recovered "
            "pair reaches Ready by rebuilding from canonical state, which stalls the run once per "
            "cache epoch; off, it warms over a policy window of live blocks instead"
        ),
    )
    parser.add_argument(
        "--force-previous-cache-snapshot",
        action="store_true",
        help="recreate the old unconditional clone as the B2 control",
    )
    parser.add_argument("node_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.warmup < 0 or args.samples <= 0:
        parser.error("--warmup must be non-negative and --samples must be positive")
    if args.poll_seconds <= 0 or args.shutdown_timeout <= 0:
        parser.error("poll and shutdown timeouts must be positive")
    if args.node_args and args.node_args[0] == "--":
        args.node_args = args.node_args[1:]
    return args


def benchmark_environment(
    builder_path,
    engine_path,
    sidecar_dir,
    parallel_initial_proof,
    canonical_rebuild,
    force_previous_cache_snapshot,
):
    env = os.environ.copy()
    env.update(
        {
            "PS_SIDECAR_ROLE": "builder",
            "PS_SIDECAR_DIR": str(sidecar_dir),
            "PS_BUILDER_BENCH_OUTPUT": str(builder_path),
            "PS_ENGINE_BENCH": "1",
            "PS_ENGINE_BENCH_OUTPUT": str(engine_path),
            "PS_PARALLEL_INITIAL_PROOF": (
                "1" if parallel_initial_proof == "on" else "0"
            ),
            "PS_CANONICAL_REBUILD": "1" if canonical_rebuild == "on" else "0",
            "PS_FORCE_PREVIOUS_CACHE_SNAPSHOT": (
                "1" if force_previous_cache_snapshot else "0"
            ),
        }
    )
    env.pop("PS_VALIDATION_BENCH", None)
    env.pop("PS_BENCH_OUTPUT", None)
    for name in DISABLED_DIAGNOSTICS:
        env.pop(name, None)
    return env


def main():
    args = parse_args()
    prepare_output(args.output)
    reth_bin = args.reth_bin.resolve()
    if not reth_bin.is_file():
        raise SystemExit(f"reth binary not found: {reth_bin}")
    if not args.datadir.is_dir():
        raise SystemExit(f"datadir not found: {args.datadir}")
    if not args.jwtsecret.is_file():
        raise SystemExit(f"JWT secret not found: {args.jwtsecret}")

    builder_path = args.output / "builder.jsonl"
    engine_path = args.output / "engine.jsonl"
    log_path = args.output / "reth-partial-stateless.log"
    report_path = args.output / "results.md"
    resource_path = args.output / "resources.jsonl"
    sidecar_dir = args.output / "sidecars"
    env = benchmark_environment(
        builder_path,
        engine_path,
        sidecar_dir,
        args.parallel_initial_proof,
        args.canonical_rebuild,
        args.force_previous_cache_snapshot,
    )
    command = build_command(reth_bin, args.datadir, args.jwtsecret, args.node_args)
    print("Starting:", " ".join(command), flush=True)
    print(
        f"Collecting warm-up {args.warmup} plus {args.samples} published builder samples",
        flush=True,
    )

    process = None
    reached_target = False
    last_count = -1
    with log_path.open("wb") as log_file:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while True:
                exit_code = process.poll()
                accepted = select_builder_samples(
                    load_jsonl(builder_path, allow_incomplete_tail=True),
                    args.warmup,
                    args.samples,
                    require_published=True,
                )
                memory = append_resource_sample(resource_path, process.pid, len(accepted))
                if len(accepted) != last_count:
                    memory_summary = (
                        f" rss={memory[0] / 1024:.0f}MiB peak={memory[1] / 1024:.0f}MiB"
                        if memory is not None
                        else ""
                    )
                    print(
                        f"accepted={len(accepted)}/{args.samples}{memory_summary}",
                        flush=True,
                    )
                    last_count = len(accepted)
                if len(accepted) >= args.samples:
                    reached_target = True
                    break
                if exit_code is not None:
                    detail = (
                        " (SIGKILL; inspect resources.jsonl)"
                        if exit_code == -signal.SIGKILL
                        else ""
                    )
                    raise RuntimeError(
                        f"reth exited before the target with status {exit_code}{detail}; "
                        f"see {log_path}"
                    )
                time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            print("Interrupted; stopping reth", file=sys.stderr, flush=True)
            raise
        finally:
            if process is not None:
                stop_process(process, args.shutdown_timeout)

    if not reached_target:
        raise RuntimeError("benchmark stopped before reaching the sample target")
    if "Chain reorg detected" in log_path.read_text(errors="replace"):
        raise RuntimeError(
            "builder benchmark observed a reorg; repeat from a deterministic replay"
        )
    report = build_builder_report(
        load_jsonl(builder_path),
        args.warmup,
        args.samples,
        expect_snapshot=args.force_previous_cache_snapshot,
        require_published=True,
    )
    report_path.write_text(report)
    print(report, end="")
    print(f"Saved report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
