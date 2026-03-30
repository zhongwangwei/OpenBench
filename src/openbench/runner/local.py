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
    It uses the config adapter to bridge between new and legacy formats,
    builds legacy namelists from the registry, and drives the evaluation
    engine for each variable / reference / simulation combination.

    Args:
        cfg: Validated OpenBenchConfig instance.

    Returns:
        Summary dict with results.
    """
    from openbench.config.adapter import build_legacy_namelists, to_legacy_config

    legacy = to_legacy_config(cfg)
    general = legacy["general"]

    # Setup output directories
    basedir = Path(general["basedir"])
    basename = general["basename"]
    output_dir = basedir / basename
    for sub in ["data", "metrics", "scores", "figures"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    logger.info("Starting evaluation: %s", basename)
    logger.info("Output directory: %s", output_dir)
    logger.info("Variables: %s", list(legacy["evaluation_items"].keys()))
    logger.info("Simulations: %s", list(cfg.simulation.keys()))

    # Build the three legacy namelists from new config + registry
    main_nl, ref_nml, sim_nml = build_legacy_namelists(cfg)

    # Derive list keys from the legacy config
    metric_vars = list(legacy["metrics"].keys())
    score_vars = list(legacy["scores"].keys())
    comparison_vars = list(legacy["comparisons"].keys())
    statistic_vars = list(legacy["statistics"].keys())

    # fig_nml placeholder (figure options); the evaluation classes accept it
    fig_nml: dict[str, Any] = {}

    evaluated: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for var_name in cfg.evaluation.variables:
        ref_source = ref_nml["general"].get(f"{var_name}_ref_source")
        sim_sources = sim_nml["general"].get(f"{var_name}_sim_source", [])

        if not ref_source:
            logger.warning("Skipping %s: no reference source", var_name)
            continue

        for sim_source in sim_sources:
            logger.info("Evaluating %s: sim=%s ref=%s", var_name, sim_source, ref_source)
            try:
                from openbench.config.legacy_processors import GeneralInfoReader

                info_reader = GeneralInfoReader(
                    main_nl=main_nl,
                    sim_nml=sim_nml,
                    ref_nml=ref_nml,
                    metric_vars=metric_vars,
                    score_vars=score_vars,
                    comparison_vars=comparison_vars,
                    statistic_vars=statistic_vars,
                    item=var_name,
                    sim_source=sim_source,
                    ref_source=ref_source,
                )

                info = info_reader.to_dict()

                # Choose grid or station evaluation based on data types
                ref_dtype = info.get("ref_data_type", "grid")
                sim_dtype = info.get("sim_data_type", "grid")

                if ref_dtype == "stn" or sim_dtype == "stn":
                    from openbench.core.evaluation import Evaluation_stn

                    Evaluation_stn(info, fig_nml)
                else:
                    from openbench.core.evaluation import Evaluation_grid

                    Evaluation_grid(info, fig_nml)

                evaluated.append({"variable": var_name, "sim": sim_source, "ref": ref_source, "status": "success"})
                logger.info("Completed %s: sim=%s ref=%s", var_name, sim_source, ref_source)

            except Exception:
                logger.exception("Evaluation failed for %s (sim=%s, ref=%s)", var_name, sim_source, ref_source)
                errors.append({"variable": var_name, "sim": sim_source, "ref": ref_source, "status": "error"})

    results: dict[str, Any] = {
        "status": "success" if not errors else "partial",
        "basename": basename,
        "output_dir": str(output_dir),
        "variables": list(legacy["evaluation_items"].keys()),
        "simulations": list(cfg.simulation.keys()),
        "metrics": metric_vars,
        "evaluated": evaluated,
        "errors": errors,
    }

    logger.info("Evaluation complete: %d succeeded, %d failed", len(evaluated), len(errors))
    return results
