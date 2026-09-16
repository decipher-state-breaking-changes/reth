#!/usr/bin/env python3
"""Sample a benchmark through exit and retain wait4's peak RSS.

Usage: measure_resources.py --out resources.jsonl [--undo-dir DIR] -- COMMAND ...
Heavy /proc and disk scans run outside the measured process. Samples are observations;
only wait4's maxrss is a process lifetime high-water mark. Host cache counters are global.
"""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def counters(path):
    result = {}
    try:
        for line in path.read_text().splitlines():
            key, _, value = line.partition(":")
            fields = value.split()
            if fields and fields[0].isdigit():
                result[key] = int(fields[0])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return result


def sample(pid, undo_dirs=(), proc=Path("/proc")):
    base = proc / str(pid)
    status = counters(base / "status")
    row = {"kind": "sample", "timestamp_unix": time.time(), "pid": pid,
           "process_present": "VmRSS" in status,
           "process_kib": {key: status.get(key) for key in ("VmRSS", "VmHWM", "RssAnon", "RssFile", "VmSwap")},
           "smaps_kib": counters(base / "smaps_rollup"), "io": counters(base / "io")}
    try:
        stat = (base / "stat").read_text().rsplit(")", 1)[1].split()
        hz = os.sysconf("SC_CLK_TCK")
        row["cpu_seconds"] = (int(stat[11]) + int(stat[12])) / hz
        row["minor_faults"] = int(stat[7])
        row["major_faults"] = int(stat[9])
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
        pass
    host = counters(proc / "meminfo")
    row["host_kib"] = {key: host.get(key) for key in ("MemAvailable", "Cached", "Buffers", "Dirty", "Writeback", "SwapFree")}
    disk = {"undo_files": 0, "temporary_files": 0, "logical_bytes": 0, "allocated_bytes": 0}
    for directory in undo_dirs:
        for path in Path(directory).rglob("*"):
            if path.suffix not in (".undo", ".tmp"):
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            disk["undo_files" if path.suffix == ".undo" else "temporary_files"] += 1
            disk["logical_bytes"] += stat.st_size
            disk["allocated_bytes"] += stat.st_blocks * 512
    row["disk"] = disk
    row["undo_directories"] = [str(directory) for directory in undo_dirs]
    return row


def run(command, output, undo_dirs, interval):
    with Path(output).open("x") as log:
        process = subprocess.Popen(command, start_new_session=True)
        def forward(signum, _frame):
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass
        previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            while True:
                log.write(json.dumps(sample(process.pid, undo_dirs), separators=(",", ":")) + "\n")
                log.flush()
                pid, status, usage = os.wait4(process.pid, os.WNOHANG)
                if pid:
                    process.returncode = os.waitstatus_to_exitcode(status)
                    log.write(json.dumps({"kind": "exit", "timestamp_unix": time.time(),
                        "pid": pid, "returncode": process.returncode, "maxrss_kib": usage.ru_maxrss,
                        "user_seconds": usage.ru_utime, "system_seconds": usage.ru_stime,
                        "minor_faults": usage.ru_minflt, "major_faults": usage.ru_majflt,
                        "input_blocks": usage.ru_inblock, "output_blocks": usage.ru_oublock}) + "\n")
                    return process.returncode
                time.sleep(interval)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            if process.returncode is None:
                process.terminate()
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--undo-dir", type=Path, action="append", default=[])
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.interval <= 0:
        parser.error("a command and positive sampling interval are required")
    code = run(command, args.out, args.undo_dir, args.interval)
    raise SystemExit(code if code >= 0 else 128 - code)


if __name__ == "__main__":
    main()
