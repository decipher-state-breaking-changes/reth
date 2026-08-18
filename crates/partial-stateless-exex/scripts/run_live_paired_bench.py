#!/usr/bin/env python3
"""Run one bounded reth-partial-stateless process and analyze paired samples."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

from analyze_builder_bench import build_builder_report, select_builder_samples
from analyze_validation_bench import (
    build_overlap_report,
    build_report,
    load_jsonl,
    select_samples,
)

DISABLED_DIAGNOSTICS = (
    "PS_TRIE_CACHE_DIAGNOSTICS",
    "PS_RESOURCE_METRICS",
    "PS_WITNESS_BASELINE",
)

# Refused rather than unset. Everything in DISABLED_DIAGNOSTICS is a switch this launcher owns and
# may therefore clear on the operator's behalf; a policy-dataset capture is a separate job someone
# deliberately started, and silently turning it off would throw away hours of recording. Failing
# instead makes the operator choose which job this shell is running.
FORBIDDEN_ENV = ("PS_POLICY_DATASET_CAPTURE_DIR",)


def refuse_conflicting_env():
    """Abort if the shell is configured for a job this measurement cannot share a process with."""
    for name in FORBIDDEN_ENV:
        if os.environ.get(name):
            raise SystemExit(
                f"{name} is set; a policy replay dataset capture builds an extra full witness per "
                "block, so a measurement started beside it would be measuring the capture. Unset "
                "it, or run the capture on its own."
            )


def default_sample_warmup(canonical_rebuild):
    """Return the warm-up needed after the path that established Ready.

    Live policy-window bootstrap has already evolved the trie for a complete window. A canonical
    rebuild instead installs the minimum multiproof for the retained paths, whose extra revealed
    intermediate nodes converge over about 50 subsequent blocks.
    """
    return 60 if canonical_rebuild == "on" else 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reth-bin", type=Path, default=Path("./target/release/reth-partial-stateless"))
    parser.add_argument("--datadir", type=Path, required=True)
    parser.add_argument("--jwtsecret", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--warmup",
        type=int,
        help=(
            "paired records to exclude after Ready (default: 0 with live bootstrap, "
            "60 with --canonical-rebuild on)"
        ),
    )
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--shutdown-timeout", type=float, default=120.0)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="stop gracefully after this wall-clock budget and report what was collected (0: no limit)",
    )
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
        "--retain-generation",
        choices=("on", "off"),
        default="on",
        help=(
            "set PS_RETAIN_GENERATION deterministically (default: on, which is production). Off "
            "is the K = 1 memory control: the transition still copies the parent trie and still "
            "hands it back, but the copy is dropped instead of kept, so every reorg costs a full "
            "rebuild and the run must not be read for recovery timings"
        ),
    )
    parser.add_argument(
        "--handoff-capacity",
        type=int,
        default=None,
        help=(
            "set PS_HANDOFF_CAPACITY, the number of artifacts the handoff retains before evicting "
            "the oldest (default 4). Raising it trades resident memory for a lower miss rate when "
            "the builder falls briefly behind; read it together with resources.jsonl, since the "
            "byte budget bounds access sets only, not the execution outputs a deeper queue holds"
        ),
    )
    parser.add_argument(
        "--shadow-sample",
        type=int,
        default=None,
        help=(
            "with --engine-access on, re-execute one block in this many purely to compare against "
            "the artifact, keeping the correctness oracle live (default 50). 0 disables sampling "
            "entirely; use it only for a pure timing run, where the sampled blocks' re-execution "
            "would otherwise sit in the builder distribution being measured"
        ),
    )
    parser.add_argument(
        "--engine-access",
        choices=("off", "shadow", "on"),
        default="off",
        help=(
            "set PS_ENGINE_ACCESS (default: off, which is the baseline every measured run must "
            "use). 'shadow' makes the Engine capture its access set and the builder compare it "
            "against its own re-execution, writing access_shadow.jsonl; the builder still "
            "re-executes, so builder timings stay comparable but carry the capture's cost on the "
            "Engine side. 'on' makes the builder use the artifact instead of re-executing, which "
            "is what the item exists to do -- read builder timings from an 'on' run, never a "
            "'shadow' one, which by construction cannot show the improvement"
        ),
    )
    parser.add_argument("node_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.warmup is None:
        args.warmup = default_sample_warmup(args.canonical_rebuild)
    if args.warmup < 0 or args.samples <= 0:
        parser.error("--warmup must be non-negative and --samples must be positive")
    if args.poll_seconds <= 0 or args.shutdown_timeout <= 0:
        parser.error("poll and shutdown timeouts must be positive")
    if args.max_seconds < 0:
        parser.error("--max-seconds must be non-negative")
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


def current_selection(
    paired_path,
    engine_path,
    log_path,
    warmup,
    samples,
    allow_incomplete_tail=False,
):
    return select_samples(
        load_jsonl(paired_path, allow_incomplete_tail=allow_incomplete_tail),
        load_jsonl(engine_path, allow_incomplete_tail=allow_incomplete_tail),
        log_path,
        warmup,
        samples,
    )


class WarmingProgress(NamedTuple):
    blocks_seen: int
    bootstrap_blocks: int
    sidecars_constructed: int
    sampling_started: bool


def warming_progress(builder_path):
    """Separate readiness bootstrap from post-Ready sample warm-up.

    A cold cache builds no sidecar until the readiness tracker has an authenticated parent, which
    takes a full eviction window of contiguous blocks. Those blocks produce builder records with
    `sidecar_constructed` false and no paired samples at all. They are readiness bootstrap, not
    analysis warm-up: there is no paired observation that the selector could exclude yet.

    Construction, not publication, is the signal: paired mode serializes in memory and skips the
    sidecar file write, so `sidecar_published` is false for every sample it takes.
    """
    records = load_jsonl(builder_path, allow_incomplete_tail=True)
    constructed = len(select_builder_samples(records, 0))
    first_constructed = next(
        (
            index
            for index, record in enumerate(records)
            if record.get("sidecar_constructed", False)
        ),
        None,
    )
    bootstrap = len(records) if first_constructed is None else first_constructed
    return WarmingProgress(len(records), bootstrap, constructed, first_constructed is not None)


class ProcessMemory(NamedTuple):
    """One `/proc/<pid>/status` memory reading, in KiB.

    `anon_kib` is the memory the process actually holds: its heap, private and dirty. `file_kib`
    is page cache for mapped files, which for this node is overwhelmingly the MDBX mmap. Those
    file-backed pages are clean, the kernel drops them on demand at no I/O cost, and MDBX will
    map as much of a database far larger than RAM as it is allowed to. They therefore inflate
    `rss_kib` and `peak_rss_kib` without representing memory the run needs, and a total-RSS peak
    climbs for hours on cache warming alone. Report and compare the anon figures.
    """

    rss_kib: int
    peak_rss_kib: int
    anon_kib: int
    file_kib: int
    swap_kib: int


def process_memory_kib(pid):
    """Return the current `ProcessMemory`, or None once the process has exited."""
    try:
        fields = {}
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            name, separator, value = line.partition(":")
            if separator and name in {"VmRSS", "VmHWM", "RssAnon", "RssFile", "VmSwap"}:
                fields[name] = int(value.strip().split()[0])
        if "VmRSS" not in fields:
            return None
        return ProcessMemory(
            rss_kib=fields["VmRSS"],
            peak_rss_kib=fields.get("VmHWM", fields["VmRSS"]),
            anon_kib=fields.get("RssAnon", 0),
            file_kib=fields.get("RssFile", 0),
            swap_kib=fields.get("VmSwap", 0),
        )
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return None


class ResourceSampler:
    """Writes one `resources.jsonl` record per poll and carries the peak anon RSS.

    The kernel publishes a high-water mark for total RSS (`VmHWM`) but none for the anon subset,
    so the peak that actually bounds a run has to be accumulated across polls. That makes it a
    sampled maximum: a spike shorter than the poll interval can be missed, which is the price of
    tracking the figure that is not dominated by reclaimable page cache.
    """

    def __init__(self, path):
        self.path = path
        self.peak_anon_kib = 0

    def sample(self, pid, accepted):
        memory = process_memory_kib(pid)
        if memory is None:
            return None
        self.peak_anon_kib = max(self.peak_anon_kib, memory.anon_kib)
        record = {
            "timestamp_unix": time.time(),
            "pid": pid,
            "accepted": accepted,
            "rss_anon_kib": memory.anon_kib,
            "peak_rss_anon_kib": self.peak_anon_kib,
            "rss_kib": memory.rss_kib,
            "peak_rss_kib": memory.peak_rss_kib,
            "rss_file_kib": memory.file_kib,
            "swap_kib": memory.swap_kib,
        }
        with self.path.open("a") as output:
            json.dump(record, output, separators=(",", ":"))
            output.write("\n")
        return memory

    def progress_summary(self, memory):
        """Console fragment for one sample: anon leads, and the reclaimable rest is labelled."""
        if memory is None:
            return ""
        return (
            f" anon={memory.anon_kib / 1024:.0f}MiB"
            f" anon_peak={self.peak_anon_kib / 1024:.0f}MiB"
            f" rss={memory.rss_kib / 1024:.0f}MiB"
            f" file_cache={memory.file_kib / 1024:.0f}MiB"
        )


def benchmark_environment(
    paired_path,
    engine_path,
    builder_path,
    shadow_path,
    sidecar_dir,
    parallel_initial_proof,
    canonical_rebuild,
    retain_generation,
    engine_access,
    shadow_sample,
    handoff_capacity,
):
    env = os.environ.copy()
    env.update(
        {
            "PS_SIDECAR_ROLE": "builder-verifier",
            "PS_SIDECAR_DIR": str(sidecar_dir),
            "PS_ENGINE_BENCH": "1",
            "PS_VALIDATION_BENCH": "1",
            "PS_BENCH_OUTPUT": str(paired_path),
            "PS_ENGINE_BENCH_OUTPUT": str(engine_path),
            "PS_BUILDER_BENCH_OUTPUT": str(builder_path),
            "PS_PARALLEL_INITIAL_PROOF": "1" if parallel_initial_proof == "on" else "0",
            "PS_CANONICAL_REBUILD": "1" if canonical_rebuild == "on" else "0",
            "PS_RETAIN_GENERATION": "1" if retain_generation == "on" else "0",
            "PS_ENGINE_ACCESS": engine_access,
            "PS_SHADOW_OUTPUT": str(shadow_path),
        }
    )
    if shadow_sample is not None:
        env["PS_SHADOW_SAMPLE"] = str(shadow_sample)
    if handoff_capacity is not None:
        env["PS_HANDOFF_CAPACITY"] = str(handoff_capacity)
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
    refuse_conflicting_env()
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
    builder_path = args.output / "builder.jsonl"
    log_path = args.output / "reth-partial-stateless.log"
    report_path = args.output / "results.md"
    overlap_report_path = args.output / "results-overlap.md"
    builder_report_path = args.output / "results-builder.md"
    resource_path = args.output / "resources.jsonl"
    shadow_path = args.output / "access_shadow.jsonl"
    sidecar_dir = args.output / "sidecars"

    env = benchmark_environment(
        paired_path,
        engine_path,
        builder_path,
        shadow_path,
        sidecar_dir,
        args.parallel_initial_proof,
        args.canonical_rebuild,
        args.retain_generation,
        args.engine_access,
        args.shadow_sample,
        args.handoff_capacity,
    )
    command = build_command(reth_bin, args.datadir, args.jwtsecret, args.node_args)
    print("Starting:", " ".join(command), flush=True)
    print(
        "Waiting for cache readiness, then collecting "
        f"{args.warmup} sample-warm-up plus {args.samples} accepted same-block samples "
        f"(canonical_rebuild={args.canonical_rebuild}, engine_access={args.engine_access}, "
        f"shadow_sample={args.shadow_sample if args.shadow_sample is not None else 'default'}, "
        f"handoff_capacity={args.handoff_capacity if args.handoff_capacity is not None else 'default'})",
        flush=True,
    )

    process = None
    reached_target = False
    stopped_on_deadline = False
    deadline = time.monotonic() + args.max_seconds if args.max_seconds else None
    last_progress = None
    sampler = ResourceSampler(resource_path)
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
                    paired_path,
                    engine_path,
                    log_path,
                    args.warmup,
                    args.samples,
                    allow_incomplete_tail=True,
                )
                warming = warming_progress(builder_path)
                memory = sampler.sample(process.pid, len(accepted))
                progress = (len(accepted), warming.blocks_seen, warming.sidecars_constructed)
                if progress != last_progress:
                    print(
                        f"accepted={len(accepted)}/{args.samples} "
                        f"sample_warmup={stats.warmup}/{args.warmup} "
                        f"bootstrap={warming.bootstrap_blocks} "
                        f"paired_sampling={'yes' if warming.sampling_started else 'no'} "
                        f"overlap={stats.contaminated} pending={stats.pending_next_engine} "
                        f"blocks={warming.blocks_seen} built={warming.sidecars_constructed}"
                        f"{sampler.progress_summary(memory)}",
                        flush=True,
                    )
                    last_progress = progress
                if len(accepted) >= args.samples:
                    reached_target = True
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    stopped_on_deadline = True
                    print(
                        f"Wall-clock budget reached; stopping with {len(accepted)} accepted samples",
                        flush=True,
                    )
                    break
                if exit_code is not None:
                    detail = (
                        " (SIGKILL; likely OOM—inspect the kernel log and resources.jsonl)"
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

    if not reached_target and not stopped_on_deadline:
        raise RuntimeError("benchmark stopped before reaching the sample target")
    accepted, stats = current_selection(
        paired_path, engine_path, log_path, args.warmup, args.samples
    )
    if not accepted:
        warming = warming_progress(builder_path)
        raise RuntimeError(
            f"no paired samples were collected from {warming.blocks_seen} blocks "
            f"({warming.sidecars_constructed} built, {warming.bootstrap_blocks} bootstrap); "
            "the cache never became Ready, so the builder published nothing to measure — "
            f"see {log_path}"
        )
    # A deadline stop measures whatever the budget allowed; the sample target only bounds it.
    collected = len(accepted)
    report = build_report(accepted, stats, args.warmup, collected)
    report_path.write_text(report)
    overlap_accepted, overlap_stats = select_samples(
        load_jsonl(paired_path),
        load_jsonl(engine_path),
        log_path,
        args.warmup,
        include_overlap=True,
    )
    overlap_report = build_overlap_report(overlap_accepted, overlap_stats, args.warmup)
    overlap_report_path.write_text(overlap_report)
    builder_records = load_jsonl(builder_path)
    builder_report = build_builder_report(
        builder_records,
        args.warmup,
        len(select_builder_samples(builder_records, args.warmup)),
        expect_snapshot=True,
    )
    builder_report_path.write_text(builder_report)
    print(report, end="")
    print(f"Saved report: {report_path}", flush=True)
    print(f"Saved overlap report: {overlap_report_path}", flush=True)
    print(f"Saved builder report: {builder_report_path}", flush=True)


if __name__ == "__main__":
    main()
