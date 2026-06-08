"""Configuration-level preflight helpers for the local runner."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from openbench.config.adapter import build_runner_bindings
from openbench.config.schema import OpenBenchConfig


def _local_attr(name: str, default: Any = None) -> Any:
    local_runner = sys.modules.get("openbench.runner.local")
    if local_runner is not None and hasattr(local_runner, name):
        return getattr(local_runner, name)
    return default


def existing_output_preflight_errors(cfg: OpenBenchConfig) -> list[dict[str, Any]]:
    """Validate that existing task outputs satisfy the configured output contract."""
    make_phase_error = _local_attr("_make_phase_error")
    validate_inputs = _local_attr("_validate_comparison_only_inputs")

    bindings = build_runner_bindings(cfg)
    runner_cfg = bindings.runner_cfg
    output_dir = Path(runner_cfg.basedir) / runner_cfg.basename
    tasks = [
        {
            "var_name": source.var_name,
            "sim_source": source.sim_source,
            "ref_source": source.ref_source,
            "bindings": bindings,
            "output_requirements": {
                "metrics": list(runner_cfg.metrics),
                "scores": list(runner_cfg.scores),
            },
        }
        for source in bindings.iter_task_sources(cfg.evaluation.variables)
    ]
    if not tasks:
        return [
            make_phase_error(
                "preflight",
                "no evaluation tasks were queued; check evaluation.variables and reference mappings",
            )
        ]
    return validate_inputs(output_dir, tasks)


def comparison_only_preflight_errors(cfg: OpenBenchConfig) -> list[dict[str, Any]]:
    """Validate comparison-only prerequisites without creating output directories."""
    make_phase_error = _local_attr("_make_phase_error")
    if not cfg.comparison.enabled:
        return [
            make_phase_error(
                "preflight",
                "comparison-only mode requires comparison.enabled: true",
            )
        ]
    return existing_output_preflight_errors(cfg)
