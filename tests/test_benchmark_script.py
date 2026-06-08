"""Smoke tests for the public CLI benchmark helper."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_benchmark_script_help_exits_successfully():
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_openbench.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, check=False)

    assert result.returncode == 0
    assert "Benchmark OpenBench CLI" in result.stdout


def test_benchmark_summary_reports_run_medians():
    import runpy

    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_openbench.py"
    module = runpy.run_path(str(script), run_name="benchmark_helper")
    CommandResult = module["CommandResult"]
    summarize = module["_summarize"]

    summary = summarize(
        [
            CommandResult("check", ["check"], 0, 10.0, 100.0),
            CommandResult("run[1/3]", ["run"], 0, 3.0, 30.0),
            CommandResult("run[2/3]", ["run"], 0, 1.0, None),
            CommandResult("run[3/3]", ["run"], 0, 2.0, 20.0),
        ]
    )

    assert summary["check"] == {"count": 1, "seconds_median": 10.0, "peak_rss_mb_median": 100.0}
    assert summary["run"] == {"count": 3, "seconds_median": 2.0, "peak_rss_mb_median": 25.0}
