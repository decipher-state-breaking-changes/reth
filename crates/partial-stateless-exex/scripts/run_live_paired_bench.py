#!/usr/bin/env python3
"""Run one bounded reth-partial-stateless process and analyze 600 paired samples."""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from analyze_validation_bench import build_report, load_jsonl, select_samples

DISABLED_DIAGNOSTICS = (
    "PS_TRIE_CACHE_DIAGNOSTICS",
    "PS_RESOURCE_METRICS",
    "PS_WITNESS_BASELINE",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reth-bin", type=Path, default=Path("./target/release/reth-partial-stateless"))
    parser.add_argument("--datadir", type=Path, required=True)
    parser.add_argument("--jwtsecret", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--shutdown-timeout", type=float, default=120.0)
    parser.add_argument("node_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.warmup < 0 or args.samples <= 0:
        parser.error("--warmup must be non-negative and --samples must be positive")
    if args.poll_seconds <= 0 or args.shutdown_timeout <= 0:
        parser.error("poll and shutdown timeouts must be positive")
    if args.node_args and args.node_args[0] == "--":
        args.node_args = args.node_args[1:]
    return args


def prepare_output(path: Path):
    if path.exists() and not path.is_dir():
        raise RuntimeError(f"output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def stop_process(process, timeout):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise RuntimeError("reth did not stop after SIGINT") from error


def current_selection(paired_path, engine_path, log_path, warmup, samples):
    return select_samples(
        load_jsonl(paired_path),
        load_jsonl(engine_path),
        log_path,
        warmup,
        samples,
    )


def benchmark_environment(paired_path, engine_path, sidecar_dir):
    env = os.environ.copy()
    env.update(
        {
            "PS_SIDECAR_ROLE": "builder-verifier",
            "PS_SIDECAR_DIR": str(sidecar_dir),
            "PS_VALIDATION_BENCH": "1",
            "PS_BENCH_OUTPUT": str(paired_path),
            "PS_ENGINE_BENCH_OUTPUT": str(engine_path),
        }
    )
    for name in DISABLED_DIAGNOSTICS:
        env.pop(name, None)
    return env


def build_command(reth_bin, datadir, jwtsecret, node_args):
    return [
        str(reth_bin),
        "node",
        "--datadir",
        str(datadir),
        "--authrpc.jwtsecret",
        str(jwtsecret),
        *(node_args or ["--minimal"]),
    ]


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

    paired_path = args.output / "paired.jsonl"
    engine_path = args.output / "engine.jsonl"
    log_path = args.output / "reth-partial-stateless.log"
    report_path = args.output / "results.md"
    sidecar_dir = args.output / "sidecars"

    env = benchmark_environment(paired_path, engine_path, sidecar_dir)
    command = build_command(reth_bin, args.datadir, args.jwtsecret, args.node_args)
    print("Starting:", " ".join(command), flush=True)
    print(
        f"Collecting warm-up {args.warmup} plus {args.samples} accepted same-block samples",
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
                accepted, stats = current_selection(
                    paired_path, engine_path, log_path, args.warmup, args.samples
                )
                if len(accepted) != last_count:
                    print(
                        f"accepted={len(accepted)}/{args.samples} warmup={stats.warmup} "
                        f"overlap={stats.contaminated} pending={stats.pending_next_engine}",
                        flush=True,
                    )
                    last_count = len(accepted)
                if len(accepted) >= args.samples:
                    reached_target = True
                    break
                if exit_code is not None:
                    raise RuntimeError(
                        f"reth exited before the target with status {exit_code}; see {log_path}"
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
    accepted, stats = current_selection(
        paired_path, engine_path, log_path, args.warmup, args.samples
    )
    report = build_report(accepted, stats, args.warmup, args.samples)
    report_path.write_text(report)
    print(report, end="")
    print(f"Saved report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
