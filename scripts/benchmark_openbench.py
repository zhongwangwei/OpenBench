#!/usr/bin/env python3
"""Benchmark OpenBench CLI runs with wall-time and peak-RSS sampling.

Examples:
    python scripts/benchmark_openbench.py test_river.yaml --dry-run
    python scripts/benchmark_openbench.py case.yaml --runs 3 --force --output-json bench.json

The script intentionally drives the public CLI (`python -m openbench`) so it can
benchmark the same path users run from shell/GUI wrappers. If psutil is
installed, peak RSS includes the child process tree; otherwise memory is reported
as unavailable while wall-clock timings are still captured.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

try:  # optional dependency in this repo, but keep the script usable without it
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - exercised only in minimal envs
    psutil = None


@dataclass
class CommandResult:
    label: str
    command: list[str]
    returncode: int
    seconds: float
    peak_rss_mb: float | None


@dataclass
class BenchmarkReport:
    config: str
    cwd: str
    runs: int
    results: list[CommandResult]
    summary: dict[str, dict[str, float | int | None]]


def _rss_tree_bytes(pid: int) -> int:
    if psutil is None:
        return 0
    try:
        proc = psutil.Process(pid)
        procs = [proc, *proc.children(recursive=True)]
    except psutil.Error:
        return 0
    total = 0
    for item in procs:
        try:
            total += int(item.memory_info().rss)
        except psutil.Error:
            continue
    return total


def _run_command(label: str, command: list[str], *, cwd: Path, env: dict[str, str]) -> CommandResult:
    start = time.perf_counter()
    peak = 0
    stop = threading.Event()
    process = subprocess.Popen(command, cwd=str(cwd), env=env)

    def sampler() -> None:
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, _rss_tree_bytes(process.pid))
            time.sleep(0.1)

    thread = threading.Thread(target=sampler, daemon=True)
    if psutil is not None:
        thread.start()
    try:
        returncode = process.wait()
    finally:
        stop.set()
        if psutil is not None:
            thread.join(timeout=1)
            peak = max(peak, _rss_tree_bytes(process.pid))

    seconds = time.perf_counter() - start
    return CommandResult(
        label=label,
        command=command,
        returncode=returncode,
        seconds=seconds,
        peak_rss_mb=(peak / 1024 / 1024) if peak else None,
    )


def _env_with_overrides(overrides: Iterable[str]) -> dict[str, str]:
    env = os.environ.copy()
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--env values must be KEY=VALUE, got: {item!r}")
        key, value = item.split("=", 1)
        env[key] = value
    return env


def _command(config: Path, action: str, *, force: bool, dry_run: bool) -> list[str]:
    cmd = [sys.executable, "-m", "openbench", action, str(config)]
    if action == "run":
        if force:
            cmd.append("--force")
        if dry_run:
            cmd.append("--dry-run")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark OpenBench CLI wall time and peak RSS.")
    parser.add_argument("config", type=Path, help="OpenBench YAML config to benchmark")
    parser.add_argument("--runs", type=int, default=1, help="Number of run iterations")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="Working directory for CLI commands")
    parser.add_argument("--force", action="store_true", help="Pass --force to openbench run")
    parser.add_argument("--dry-run", action="store_true", help="Benchmark run --dry-run instead of full evaluation")
    parser.add_argument("--skip-check", action="store_true", help="Skip the initial openbench check")
    parser.add_argument("--env", action="append", default=[], help="Environment override KEY=VALUE; may repeat")
    parser.add_argument("--output-json", type=Path, help="Write machine-readable JSON report")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be >= 1")

    cwd = args.cwd.resolve()
    config = args.config if args.config.is_absolute() else (cwd / args.config)
    env = _env_with_overrides(args.env)
    results: list[CommandResult] = []

    if not args.skip_check:
        results.append(_run_command("check", _command(config, "check", force=False, dry_run=False), cwd=cwd, env=env))
        if results[-1].returncode != 0:
            report = BenchmarkReport(str(config), str(cwd), args.runs, results, _summarize(results))
            _emit(report, args.output_json)
            return results[-1].returncode

    for index in range(args.runs):
        label = f"run[{index + 1}/{args.runs}]"
        result = _run_command(label, _command(config, "run", force=args.force, dry_run=args.dry_run), cwd=cwd, env=env)
        results.append(result)
        if result.returncode != 0:
            break

    report = BenchmarkReport(str(config), str(cwd), args.runs, results, _summarize(results))
    _emit(report, args.output_json)
    return next((result.returncode for result in results if result.returncode != 0), 0)


def _median(values: list[float | None]) -> float | None:
    numeric = [value for value in values if value is not None]
    return statistics.median(numeric) if numeric else None


def _summarize(results: list[CommandResult]) -> dict[str, dict[str, float | int | None]]:
    groups = {
        "check": [result for result in results if result.label == "check"],
        "run": [result for result in results if result.label.startswith("run[")],
    }
    summary: dict[str, dict[str, float | int | None]] = {}
    for label, items in groups.items():
        if not items:
            continue
        summary[label] = {
            "count": len(items),
            "seconds_median": _median([item.seconds for item in items]),
            "peak_rss_mb_median": _median([item.peak_rss_mb for item in items]),
        }
    return summary


def _emit(report: BenchmarkReport, output_json: Path | None) -> None:
    payload = asdict(report)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
