"""Local evaluation runner.

Orchestrates the evaluation pipeline using the new config system
and the migrated core engine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from openbench.config.schema import OpenBenchConfig

logger = logging.getLogger(__name__)


def run_evaluation(cfg: OpenBenchConfig) -> dict[str, Any]:
    """Run evaluation from a validated config.

    This is the main entry point that replaces the old openbench.py script.
    It uses the config adapter to bridge between new and legacy formats.

    Args:
        cfg: Validated OpenBenchConfig instance.

    Returns:
        Summary dict with results.
    """
    from openbench.config.adapter import to_legacy_config

    legacy = to_legacy_config(cfg)
    general = legacy["general"]

    # Setup output directories
    basedir = Path(general["basedir"])
    basename = general["basename"]
    output_dir = basedir / basename
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting evaluation: {basename}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Variables: {list(legacy['evaluation_items'].keys())}")
    logger.info(f"Simulations: {list(cfg.simulation.keys())}")

    # For now, verify the pipeline is connected end-to-end.
    # Full evaluation engine integration will be completed when
    # the legacy evaluation code is fully wired to the new config.
    results = {
        "status": "success",
        "basename": basename,
        "output_dir": str(output_dir),
        "variables": list(legacy["evaluation_items"].keys()),
        "simulations": list(cfg.simulation.keys()),
        "metrics": list(legacy["metrics"].keys()),
    }

    logger.info("Evaluation pipeline connected successfully.")
    return results
