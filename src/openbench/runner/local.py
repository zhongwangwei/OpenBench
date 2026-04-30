"""Local evaluation runner.

Orchestrates the evaluation pipeline using the new config system
and the migrated core engine.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any

from openbench.config.adapter import BridgeRuntimeInfo, RunnerConfig
from openbench.config.schema import OpenBenchConfig

logger = logging.getLogger(__name__)


BRIDGE_RUNTIME_FIELDS = {
    "casedir",
    "ref_varname",
    "sim_varname",
    "ref_data_type",
    "sim_data_type",
    "compare_tim_res",
    "compare_grid_res",
    "compare_tzone",
    "unified_mask",
}


@dataclass(frozen=True)
class RuntimeContext:
    """Runner-owned context layered on top of bridge-provided fields."""

    bridge_info: BridgeRuntimeInfo
    ref_source: str
    sim_source: str
    ref_file_override: str | None = None

    def to_info(self) -> dict[str, Any]:
        info = self.bridge_info.to_info()
        info["ref_source"] = self.ref_source
        info["sim_source"] = self.sim_source
        if self.ref_file_override:
            info["ref_file_override"] = self.ref_file_override
        return info


def _coerce_bridge_runtime_info(bridge_info: BridgeRuntimeInfo | dict[str, Any]) -> BridgeRuntimeInfo:
    """Normalize adapter bridge payloads to the typed runtime-info wrapper."""
    if isinstance(bridge_info, BridgeRuntimeInfo):
        return bridge_info
    return BridgeRuntimeInfo(payload=dict(bridge_info))


def _build_runtime_context(task: dict[str, Any]) -> RuntimeContext:
    """Build runner-owned runtime context without mutating reader state."""
    bridge_info = _coerce_bridge_runtime_info(task["bindings"].build_runtime_info_for(
        task["var_name"], task["sim_source"], task["ref_source"]
    ))

    return RuntimeContext(
        bridge_info=bridge_info,
        ref_source=task["ref_source"],
        sim_source=task["sim_source"],
        ref_file_override=task.get("ref_file_override"),
    )


def _build_bridge_runtime_info(task: dict[str, Any]) -> dict[str, Any]:
    """Build the runner-owned bridge info dict for one task."""
    return _build_runtime_context(task).to_info()


def _evaluate_single(task: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a single variable+sim+ref pair.

    Args:
        task: Dict with keys: var_name, sim_source, ref_source, bindings,
              cache_key, config_hash, use_cache, cache_dir.

    Returns:
        Result dict with keys: variable, sim, ref, status, cache_key,
        config_hash, skipped.
    """
    var_name = task["var_name"]
    sim_source = task["sim_source"]
    ref_source = task["ref_source"]
    cache_key = task["cache_key"]
    config_hash = task["config_hash"]
    use_cache = task["use_cache"]
    cache_dir = task.get("cache_dir")
    bindings = task["bindings"]

    # Each worker process needs its own cache instance (for parallel safety)
    cache = None
    if use_cache and cache_dir is not None:
        from openbench.runner.cache import EvaluationCache

        cache = EvaluationCache(Path(cache_dir))
        if cache.is_cached(cache_key, config_hash):
            # Verify that output files actually exist before trusting cache.
            # Pattern MUST include ref_source — multi-ref configs have several
            # tasks per (var, sim) and an earlier ref's outputs would otherwise
            # let a later ref's cache check falsely pass with skipped=True even
            # though that ref had never been evaluated. Reuse _find_existing_outputs
            # which already encodes the correct (var, ref, sim) pattern.
            output_dir = Path(cache_dir)
            has_output = bool(_find_existing_outputs(output_dir, task))
            if has_output:
                logger.info("Cached, skipping %s: sim=%s ref=%s", var_name, sim_source, ref_source)
                return {
                    "variable": var_name,
                    "sim": sim_source,
                    "ref": ref_source,
                    "status": "success",
                    "cache_key": cache_key,
                    "config_hash": config_hash,
                    "skipped": True,
                }
            else:
                logger.warning(
                    "Cache stale (output missing), re-evaluating %s: sim=%s ref=%s",
                    var_name, sim_source, ref_source,
                )
                # invalidate() now takes an fcntl.flock; on NFS / locked
                # filesystems that can OSError or EPERM. Don't let cache
                # bookkeeping errors crash the worker — the evaluation
                # itself will simply re-run, which is the safe default.
                try:
                    cache.invalidate(cache_key)
                except Exception as inv_err:
                    logger.warning(
                        "Cache invalidate failed for %s (sim=%s ref=%s): %s — proceeding with re-evaluation",
                        var_name, sim_source, ref_source, inv_err,
                    )

    try:
        info = _build_bridge_runtime_info(task)
        evaluation_fig_nml = bindings.build_evaluation_fig_nml().to_fig_nml()

        # Step 1: Preprocess data (skip if already done by _preprocess_variable)
        if not task.get("ref_preprocessed"):
            from openbench.data.processing import DatasetProcessing

            dataset_processor = DatasetProcessing(info)
            dataset_processor.prepare_source("ref")
            dataset_processor.prepare_source("sim")

        # Step 2: Run evaluation
        ref_dtype = info.get("ref_data_type", "grid")
        sim_dtype = info.get("sim_data_type", "grid")

        if ref_dtype == "stn" or sim_dtype == "stn":
            from openbench.core.evaluation import Evaluation_stn

            evaluator = Evaluation_stn(info, evaluation_fig_nml)
            try:
                evaluator.make_evaluation_P()
            except (KeyError, TypeError) as viz_err:
                logger.warning("Metrics computed but visualization skipped: %s", viz_err)
        else:
            from openbench.core.evaluation import Evaluation_grid

            evaluator = Evaluation_grid(info, evaluation_fig_nml)
            try:
                evaluator.make_Evaluation()
            except (KeyError, TypeError) as viz_err:
                logger.warning("Metrics computed but visualization skipped: %s", viz_err)

        if cache is not None:
            # The evaluation already succeeded — its output files are on
            # disk. A failure to update the cache index (e.g. fcntl.flock
            # rejected on NFS) MUST NOT downgrade success to error,
            # otherwise GUI / CLI reports a false negative and the user
            # re-runs an already-completed evaluation. Log and continue.
            try:
                cache.mark_done(cache_key, config_hash)
            except Exception as mark_err:
                logger.warning(
                    "mark_done failed for %s (sim=%s ref=%s): %s — evaluation succeeded, cache index not updated",
                    var_name, sim_source, ref_source, mark_err,
                )

        logger.info("Completed %s: sim=%s ref=%s", var_name, sim_source, ref_source)
        return {
            "variable": var_name,
            "sim": sim_source,
            "ref": ref_source,
            "status": "success",
            "cache_key": cache_key,
            "config_hash": config_hash,
            "skipped": False,
        }

    except Exception as exc:
        logger.exception("Evaluation failed for %s (sim=%s, ref=%s)", var_name, sim_source, ref_source)
        return {
            "variable": var_name,
            "sim": sim_source,
            "ref": ref_source,
            "status": "error",
            "error": str(exc),
            "cache_key": cache_key,
            "config_hash": config_hash,
            "skipped": False,
        }


def _apply_unified_mask(info: dict, var_name: str, ref_source: str, sim_source: str, ref_override: str | None = None) -> None:
    """Apply unified mask: set ref to NaN wherever sim is NaN.

    This ensures evaluation metrics only cover grid cells where BOTH
    reference and simulation have valid data. Called for each sim source,
    so the mask accumulates — if any sim has NaN at a point, the ref
    gets NaN there too, ensuring consistent spatial coverage across
    all model comparisons.

    Only applies to grid data (station data is skipped).
    """
    import os

    import numpy as np

    casedir = info["casedir"]
    ref_varname = info.get("ref_varname", "")
    sim_varname = info.get("sim_varname", "")

    if ref_override:
        ref_path = ref_override
    else:
        ref_path = os.path.join(casedir, "data", f"{var_name}_ref_{ref_source}_{ref_varname}.nc")
    sim_path = os.path.join(casedir, "data", f"{var_name}_sim_{sim_source}_{sim_varname}.nc")

    ref_path = os.path.abspath(ref_path)
    sim_path = os.path.abspath(sim_path)

    if not os.path.exists(ref_path) or not os.path.exists(sim_path):
        logger.debug("Unified mask skipped: ref or sim file not found")
        return

    ref_ds = None
    sim_ds = None
    try:
        import xarray as xr

        ref_ds = xr.open_dataset(ref_path)
        sim_ds = xr.open_dataset(sim_path)

        o = ref_ds[ref_varname]
        s = sim_ds[sim_varname]

        # Convert types if needed
        try:
            from openbench.util.converttype import Convert_Type

            o = Convert_Type.convert_nc(o)
            s = Convert_Type.convert_nc(s)
        except ImportError:
            pass

        # Align time dimension. Length-equal but value-different time
        # vectors must NOT be coerced silently; that produced misaligned
        # masks where ref hour 03 was treated as sim hour 04 etc.
        if len(s["time"]) != len(o["time"]):
            logger.warning(
                "Unified mask: time length mismatch for %s (ref=%d, sim=%d), skipping",
                var_name, len(o["time"]), len(s["time"]),
            )
            return
        if not np.array_equal(s["time"].values, o["time"].values):
            logger.warning(
                "Unified mask: time values mismatch for %s (lengths equal but timestamps differ), skipping",
                var_name,
            )
            return
        # Lengths match AND values match — no copy needed; xarray ops will
        # broadcast natively.

        # Apply mask: NaN where either is NaN
        mask = np.isnan(s.values) | np.isnan(o.values)
        o_data = o.load()
        o_data.values[mask] = np.nan

        # Close datasets before writing
        ref_ds.close()
        ref_ds = None
        sim_ds.close()
        sim_ds = None

        # Write masked ref back
        o_data.to_netcdf(ref_path)
        logger.debug("Unified mask applied: %s (sim=%s)", var_name, sim_source)

        del o, s, mask, o_data

    except Exception:
        logger.exception("Unified mask failed for %s (sim=%s)", var_name, sim_source)
    finally:
        if ref_ds is not None:
            try:
                ref_ds.close()
            except Exception:
                pass
        if sim_ds is not None:
            try:
                sim_ds.close()
            except Exception:
                pass


def _make_phase_error(phase: str, message: str, **details: Any) -> dict[str, Any]:
    """Create a structured error entry for runner results."""
    error = {"phase": phase, "status": "error", "message": message}
    error.update(details)
    return error


def _find_existing_outputs(output_dir: Path, task: dict[str, Any]) -> list[Path]:
    """Find existing evaluation outputs for comparison-only mode."""
    pattern = f"{task['var_name']}_*{task['ref_source']}*{task['sim_source']}*"
    matches: list[Path] = []
    for subdir in ("metrics", "scores"):
        matches.extend((output_dir / subdir).glob(pattern))
    return matches


def _validate_comparison_only_inputs(output_dir: Path, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure comparison-only mode has pre-existing evaluation outputs."""
    errors = []
    for task in tasks:
        if _find_existing_outputs(output_dir, task):
            continue
        errors.append(
            _make_phase_error(
                "preflight",
                "missing prerequisite outputs for comparison-only mode",
                variable=task["var_name"],
                sim=task["sim_source"],
                ref=task["ref_source"],
            )
        )
    return errors


def run_evaluation(cfg: OpenBenchConfig, force: bool = False, comparison_only: bool = False) -> dict[str, Any]:
    """Run evaluation from a validated config.

    This is the main entry point that replaces the old openbench.py script.
    It uses the config adapter to bridge between new and legacy formats,
    builds legacy namelists from the registry, and drives the evaluation
    engine for each variable / reference / simulation combination.

    Supports variable-level parallelism via joblib when ``cfg.project.num_cores``
    is greater than 1, and an incremental cache that skips re-computation when
    the config for a variable hasn't changed (pass ``force=True`` to bypass).

    Args:
        cfg: Validated OpenBenchConfig instance.
        force: If True, bypass the incremental cache and re-run all evaluations.

    Returns:
        Summary dict with results.
    """
    # Disable HDF5 file locking for parallel reads.
    # HDF5 ≥ 1.14 locks files even for read-only access, which causes
    # "Resource temporarily unavailable" when multiple workers open the
    # same reference NC file.  All writes go to distinct output files,
    # so disabling the lock is safe.
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    from openbench.config.adapter import build_runner_bindings
    from openbench.runner.cache import EvaluationCache, make_cache_key

    bindings = build_runner_bindings(cfg)
    runner_cfg = bindings.runner_cfg
    general = runner_cfg.general

    # Setup output directories
    basedir = Path(runner_cfg.basedir)
    basename = runner_cfg.basename
    output_dir = basedir / basename
    for sub in ["data", "metrics", "scores", "figures", "comparisons", "reports", "scratch", "tmp"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    logger.info("Starting evaluation: %s", basename)
    logger.info("Output directory: %s", output_dir)
    logger.info("Variables: %s", list(runner_cfg.evaluation_items.keys()))
    logger.info("Simulations: %s", list(cfg.simulation.keys()))

    # Derive list keys from the legacy config
    metric_vars = list(runner_cfg.metrics)
    score_vars = list(runner_cfg.scores)
    comparison_vars = list(runner_cfg.comparisons)
    statistic_vars = list(runner_cfg.statistics)

    # Determine parallelism level (auto-detect when num_cores is None/0).
    # Currently unused at variable level — parallelism lives inside
    # DatasetProcessing (station processing, yearly combination).
    # Reserved for future variable-level parallel dispatch.
    num_cores: int = max(1, os.cpu_count() or 1)
    if hasattr(cfg, "project") and cfg.project is not None:
        num_cores = getattr(cfg.project, "num_cores", 0) or num_cores

    # Also honour force flag from cfg.project if not passed directly
    if not force and hasattr(cfg, "project") and cfg.project is not None:
        force = bool(getattr(cfg.project, "force", False))

    use_cache = not force

    # Build task list
    tasks: list[dict[str, Any]] = []
    for source in bindings.iter_task_sources(cfg.evaluation.variables):
        var_name = source.var_name
        sim_source = source.sim_source
        ref_source = source.ref_source
        cache_key = make_cache_key(var_name, sim_source, ref_source)
        config_hash = EvaluationCache.hash_config(
            {
                "variable": var_name,
                "sim_source": sim_source,
                "ref_source": ref_source,
                "metrics": metric_vars,
                "scores": score_vars,
                "comparisons": comparison_vars,
                "statistics": statistic_vars,
            }
        )
        tasks.append(
            {
                "var_name": var_name,
                "sim_source": sim_source,
                "ref_source": ref_source,
                "bindings": bindings,
                "cache_key": cache_key,
                "config_hash": config_hash,
                "use_cache": use_cache,
                "cache_dir": str(output_dir),
            }
        )
        logger.info("Queued %s: sim=%s ref=%s", var_name, sim_source, ref_source)

    # ─── Phase 1: Evaluation ───
    if comparison_only:
        logger.info("Comparison-only mode: skipping evaluation phase")
        # Validate up front: if NONE of the requested tasks have pre-existing
        # outputs, the comparison phase has nothing to work with — surface
        # that as a preflight error rather than silently completing.
        errors = _validate_comparison_only_inputs(output_dir, tasks)
        evaluated = []
        skipped = 0
        for t in tasks:
            if _find_existing_outputs(output_dir, t):
                evaluated.append({"variable": t["var_name"], "sim": t["sim_source"], "ref": t["ref_source"]})
            else:
                skipped += 1
                logger.info(
                    "Comparison-only: skipping %s/%s (no pre-existing outputs)",
                    t["var_name"], t["sim_source"],
                )
        if skipped:
            logger.info("Comparison-only: %d task(s) skipped, %d available", skipped, len(evaluated))
        # If every task is missing outputs, abort before downstream phases
        # so the caller sees a clean error instead of a "success-but-empty"
        # result that misleads scripted pipelines.
        if not evaluated:
            return {
                "status": "error",
                "basename": basename,
                "output_dir": str(output_dir),
                "variables": list(runner_cfg.evaluation_items.keys()),
                "simulations": list(cfg.simulation.keys()),
                "metrics": metric_vars,
                "evaluated": [],
                "errors": errors,
            }

    # ─── Pre-process data: parallel across variables, serial within each variable ───
    #
    # Strategy: group tasks by variable, preprocess each variable's tasks in parallel.
    # Within each variable: ref once → for each sim: sim + unified_mask (serial, mask accumulates).
    # Different variables write to different files → safe to parallelize.
    unified_mask = general.get("unified_mask", True)

    # Group tasks by variable
    from collections import defaultdict

    var_tasks: dict[str, list[dict]] = defaultdict(list)
    for task in tasks:
        var_tasks[task["var_name"]].append(task)

    time_alignment = cfg.project.time_alignment  # "intersection", "per_pair", "strict"

    def _preprocess_variable(var_name: str, vtasks: list[dict]) -> list[dict[str, Any]]:
        """Preprocess all tasks for one variable (serial within variable).

        Multi-reference + mixed data-type support: ref preprocessing dedupes
        on (ref_source, output_signature). The signature distinguishes pure
        grid×grid (which produces a shareable flat NC) from stn-involved
        runs (which write per-pair ``stn_<ref>_<sim>/`` and DELETE the flat
        as part of station extraction). Without this distinction, a sequence
        like grid×grid → stn×ref-grid → grid×grid would skip the third
        task's prep but find the flat NC already gone.

        For intersection/strict: unified_mask accumulates across sims for
        that ref_source (tracked separately).
        For per_pair: each sim gets its own ref copy (no cross-contamination).
        """
        from openbench.data.processing import DatasetProcessing

        # Dedupe by (ref_source, signature):
        #   - "_grid": pure grid×grid prep produces a flat NC reusable across
        #     multiple grid sims with the same ref
        #   - sim_source string: stn-involved prep writes a per-pair output
        #     dir (stn_<ref>_<sim>) and deletes the flat NC; cannot be shared
        preproc_done: set[tuple[str, str]] = set()
        # Track first-time-seen per ref_source for unified_mask accumulation
        refs_first_seen: set[str] = set()
        # For stn×stn symlink optimization: first stn dir per ref_source
        ref_stn_data_dirs: dict[str, str] = {}
        # ref_source -> flat ref NC path (only valid while a grid-only path lives there)
        ref_flat_paths: dict[str, str] = {}
        phase_errors: list[dict[str, Any]] = []

        for task in vtasks:
            ref_source = task["ref_source"]
            sim_source = task["sim_source"]
            try:
                info = _build_bridge_runtime_info(task)
                ref_dtype = info.get("ref_data_type", "grid")
                sim_dtype = info.get("sim_data_type", "grid")
                is_stn_path = (ref_dtype == "stn") or (sim_dtype == "stn")
                prep_key = (ref_source, sim_source if is_stn_path else "_grid")

                processor = DatasetProcessing(info)

                # Ref: dedupe by (ref_source, signature). Mixed-type configs
                # (same ref reused across grid and stn sims) correctly run
                # separate prep for the stn-side without skipping.
                if prep_key not in preproc_done:
                    logger.info(
                        "Preprocessing ref: %s (%s) [%s]",
                        var_name, ref_source,
                        "stn-pair" if is_stn_path else "grid",
                    )
                    processor.prepare_source("ref")
                    preproc_done.add(prep_key)
                    if is_stn_path:
                        # Remember first stn dir per ref for stn×stn symlink branch below
                        ref_stn_data_dirs.setdefault(
                            ref_source,
                            os.path.join(
                                info["casedir"], "data",
                                f"stn_{ref_source}_{sim_source}",
                            ),
                        )
                    else:
                        # Grid prep produced a flat NC; remember its path
                        ref_varname = info.get("ref_varname", "")
                        ref_flat_paths[ref_source] = os.path.join(
                            info["casedir"], "data",
                            f"{var_name}_ref_{ref_source}_{ref_varname}.nc",
                        )
                elif is_stn_path and ref_source in ref_stn_data_dirs:
                    # stn×stn (same ref, second sim): symlink ref files from
                    # the first stn dir to this sim's dir so we don't re-extract
                    # ref stations. Same-ref-same-stn-different-sim case.
                    this_data_dir = os.path.join(
                        info["casedir"], "data",
                        f"stn_{ref_source}_{sim_source}",
                    )
                    src_data_dir = ref_stn_data_dirs[ref_source]
                    if (
                        src_data_dir
                        and os.path.isdir(src_data_dir)
                        and this_data_dir != src_data_dir
                    ):
                        os.makedirs(this_data_dir, exist_ok=True)
                        for ref_file in os.listdir(src_data_dir):
                            if "_ref_" in ref_file and ref_file.endswith(".nc"):
                                src = os.path.abspath(os.path.join(src_data_dir, ref_file))
                                dst = os.path.join(this_data_dir, ref_file)
                                if not os.path.exists(dst):
                                    os.symlink(src, dst)

                # Sim: each task
                logger.info("Preprocessing sim: %s (%s)", var_name, sim_source)
                processor.prepare_source("sim")

                # Unified mask: ensure evaluation only covers cells where both ref and sim are valid.
                if unified_mask:
                    if ref_dtype != "stn" and sim_dtype != "stn":
                        ref_file_path_for_pair = ref_flat_paths.get(ref_source)
                        if time_alignment == "per_pair" and ref_file_path_for_pair:
                            # per_pair: each sim gets its own ref copy — no cross-contamination
                            ref_varname_m = info.get("ref_varname", "")
                            pair_ref = os.path.join(
                                info["casedir"], "data",
                                f"{var_name}_ref_{ref_source}_{sim_source}_{ref_varname_m}.nc",
                            )
                            if not os.path.exists(pair_ref):
                                import shutil
                                shutil.copy2(ref_file_path_for_pair, pair_ref)
                            _apply_unified_mask(info, var_name, ref_source, sim_source, ref_override=pair_ref)
                            # Record per-pair ref path so evaluation uses this copy
                            task["ref_file_override"] = pair_ref
                        else:
                            # intersection/strict: mask accumulates across sims onto shared ref
                            _apply_unified_mask(info, var_name, ref_source, sim_source)

                task["ref_preprocessed"] = True

            except Exception as exc:
                task["preprocess_failed"] = True
                phase_errors.append(
                    _make_phase_error(
                        "preprocess",
                        f"preprocessing failed: {exc}",
                        variable=var_name,
                        sim=sim_source,
                        ref=ref_source,
                    )
                )
                if isinstance(exc, (FileNotFoundError, ValueError)):
                    # Expected errors: show concise message without full traceback
                    logger.error(
                        "Preprocessing failed: %s (sim=%s, ref=%s): %s",
                        var_name, sim_source, ref_source, exc,
                    )
                else:
                    logger.exception(
                        "Preprocessing failed: %s (sim=%s, ref=%s)", var_name, sim_source, ref_source
                    )

        return phase_errors

    # Dispatch preprocessing + evaluation (skip in comparison_only mode)
    if not comparison_only:
        var_names = list(var_tasks.keys())
        # Preprocess + evaluate: serial across variables (like old openbench.py).
        # Parallelism lives *inside* station processing (Parallel n_jobs=num_cores)
        # and inside yearly file combination, not at the variable/task level.
        # This avoids nested-parallel deadlocks and I/O contention on shared
        # reference files that plagued the previous variable-level parallel dispatch.
        preprocess_errors: list[dict[str, Any]] = []
        for vn in var_names:
            preprocess_errors.extend(_preprocess_variable(vn, var_tasks[vn]))

        ready_tasks = [task for task in tasks if not task.get("preprocess_failed")]

        raw_results: list[dict[str, Any]] = [_evaluate_single(t) for t in ready_tasks]

        evaluated = []
        errors = list(preprocess_errors)
        for res in raw_results:
            if res["status"] == "success":
                evaluated.append({
                    "variable": res["variable"], "sim": res["sim"], "ref": res["ref"],
                    "status": "success", "skipped": res.get("skipped", False),
                })
            else:
                errors.append({
                    "variable": res["variable"], "sim": res["sim"], "ref": res["ref"],
                    "status": "error",
                })

    logger.info("Evaluation phase: %d succeeded, %d failed", len(evaluated), len(errors))

    # ─── Phase 2: Comparison ───
    if cfg.comparison.enabled and comparison_vars and evaluated:
        logger.info("Starting comparison phase: %s", comparison_vars)
        errors.extend(_run_comparison(bindings, comparison_vars, output_dir))

    # ─── Phase 2b: Groupby (IGBP / PFT / Climate Zone) ───
    if evaluated:
        errors.extend(_run_groupby(cfg, bindings, output_dir))

    # ─── Phase 3: Statistics ───
    # Statistics module operates on gridded NC files (spatial remap + aggregation).
    # Skip for purely station-based evaluations where metrics are CSV-only.
    grid_evidence = bindings.has_grid_evaluation(cfg.evaluation.variables)
    if cfg.statistics.enabled and statistic_vars and grid_evidence.has_grid:
        logger.info("Starting statistics phase: %s", statistic_vars)
        errors.extend(_run_statistics(bindings, statistic_vars))
    elif cfg.statistics.enabled and statistic_vars and not grid_evidence.has_grid:
        logger.info("Skipping statistics phase: not applicable for station-only evaluations")

    # ─── Phase 4: Report ───
    if cfg.project.generate_report:
        errors.extend(_run_report(bindings, output_dir))

    if errors and evaluated:
        status = "partial"
    elif errors:
        status = "error"
    else:
        status = "success"

    results: dict[str, Any] = {
        "status": status,
        "basename": basename,
        "output_dir": str(output_dir),
        "variables": list(runner_cfg.evaluation_items.keys()),
        "simulations": list(cfg.simulation.keys()),
        "metrics": metric_vars,
        "evaluated": evaluated,
        "errors": errors,
    }

    # Clean up per_pair temporary ref copies
    for task in tasks:
        pair_ref = task.get("ref_file_override")
        if pair_ref and os.path.isfile(pair_ref):
            try:
                os.remove(pair_ref)
            except OSError:
                pass

    logger.info("All phases complete: %d evaluated, %d errors", len(evaluated), len(errors))
    return results


# ─── Post-evaluation phases ───


def _run_comparison(bindings, comparison_vars, output_dir):
    """Run comparison visualizations (Taylor diagrams, heat maps, etc.)."""
    import gc
    phase_errors = []

    try:
        from openbench.core.comparison import ComparisonProcessing

        basedir = str(output_dir)
        context = bindings.build_comparison_context()
        namelists = context.namelists
        score_vars = context.score_vars
        metric_vars = context.metric_vars

        # Filter evaluation items to only those that were successfully evaluated
        # (i.e. have score/metric output files)
        scores_dir = output_dir / "scores"
        metrics_dir = output_dir / "metrics"
        all_items = context.evaluation_items
        evaluation_items = []
        for item in all_items:
            has_scores = scores_dir.exists() and any(
                f.name.startswith(f"{item}_") for f in scores_dir.iterdir() if f.is_file()
            )
            has_metrics = metrics_dir.exists() and any(
                f.name.startswith(f"{item}_") for f in metrics_dir.iterdir() if f.is_file()
            )
            if has_scores or has_metrics:
                evaluation_items.append(item)
            else:
                logger.info("Skipping comparison for '%s': no evaluation outputs found", item)
        if not evaluation_items:
            logger.warning("No evaluation items with data files, skipping comparison phase")
            return phase_errors

        ch = ComparisonProcessing(namelists.main, score_vars, metric_vars)

        comparison_fig = context.comparison_fig

        for cvar in comparison_vars:
            logger.info("Running %s comparison...", cvar)
            if cvar in ("Mean", "Median", "Max", "Min", "Sum"):
                method_name = "scenarios_Basic_comparison"
            else:
                method_name = f"scenarios_{cvar}_comparison"

            fig_opts = comparison_fig.get(cvar, {})

            if hasattr(ch, method_name):
                try:
                    getattr(ch, method_name)(
                        basedir, namelists.simulation, namelists.reference, evaluation_items,
                        score_vars, metric_vars, fig_opts,
                    )
                    logger.info("Completed %s comparison", cvar)
                except Exception:
                    logger.exception("Failed %s comparison", cvar)
                    phase_errors.append(_make_phase_error("comparison", f"{cvar} comparison failed"))
            else:
                logger.warning("Comparison method %s not found, skipping", method_name)
                phase_errors.append(_make_phase_error("comparison", f"{cvar} comparison method not found"))

            gc.collect()

    except ImportError:
        logger.warning("ComparisonProcessing not available, skipping comparison phase")
        phase_errors.append(_make_phase_error("comparison", "comparison processing is not available"))
    except Exception:
        logger.exception("Comparison phase failed")
        phase_errors.append(_make_phase_error("comparison", "comparison phase failed"))

    return phase_errors


def _run_groupby(cfg, bindings, output_dir):
    """Run land cover and climate zone groupby analysis."""
    import gc
    phase_errors = []

    basedir = str(output_dir)
    context = bindings.build_groupby_context()
    namelists = context.namelists
    evaluation_items = context.evaluation_items
    score_vars = context.score_vars
    metric_vars = context.metric_vars
    validation_fig = context.validation_fig

    if cfg.project.IGBP_groupby:
        try:
            from openbench.core.landcover_groupby import LC_groupby

            logger.info("Running IGBP land cover groupby...")
            lc = LC_groupby(namelists.main, score_vars, metric_vars)
            lc.scenarios_IGBP_groupby_comparison(
                basedir, namelists.simulation, namelists.reference, evaluation_items,
                score_vars, metric_vars, validation_fig,
            )
            gc.collect()
            logger.info("IGBP groupby complete")
        except Exception:
            logger.exception("IGBP groupby failed")
            phase_errors.append(_make_phase_error("groupby", "IGBP groupby failed"))

    if cfg.project.PFT_groupby:
        try:
            from openbench.core.landcover_groupby import LC_groupby

            logger.info("Running PFT groupby...")
            lc = LC_groupby(namelists.main, score_vars, metric_vars)
            lc.scenarios_PFT_groupby_comparison(
                basedir, namelists.simulation, namelists.reference, evaluation_items,
                score_vars, metric_vars, validation_fig,
            )
            gc.collect()
            logger.info("PFT groupby complete")
        except Exception:
            logger.exception("PFT groupby failed")
            phase_errors.append(_make_phase_error("groupby", "PFT groupby failed"))

    if cfg.project.climate_zone_groupby:
        try:
            from openbench.core.climatezone_groupby import CZ_groupby

            logger.info("Running climate zone groupby...")
            cz = CZ_groupby(namelists.main, score_vars, metric_vars)
            cz_fig = context.climate_zone_fig
            cz.scenarios_CZ_groupby_comparison(
                basedir, namelists.simulation, namelists.reference, evaluation_items,
                score_vars, metric_vars, cz_fig,
            )
            gc.collect()
            logger.info("Climate zone groupby complete")
        except Exception:
            logger.exception("Climate zone groupby failed")
            phase_errors.append(_make_phase_error("groupby", "climate zone groupby failed"))

    return phase_errors


def _run_statistics(bindings, statistic_vars):
    """Run statistical analysis."""
    import gc
    import os
    phase_errors = []

    try:
        from openbench.core.statistics.Mod_Statistics import StatisticsProcessing

        context = bindings.build_statistics_context(statistic_vars)
        main_nl = context.namelists.main
        stats_dir = context.stats_dir
        stats_nml = context.stats_nml
        os.makedirs(stats_dir, exist_ok=True)

        stats_handler = StatisticsProcessing(
            main_nl, stats_nml,
            stats_dir,
            num_cores=context.num_cores,
        )

        statistic_fig = context.statistic_fig

        for statistic in statistic_vars:
            logger.info("Running %s analysis...", statistic)
            if statistic in ("Mean", "Median", "Max", "Min", "Sum"):
                method_name = "scenarios_Basic_analysis"
            else:
                method_name = f"scenarios_{statistic}_analysis"

            if hasattr(stats_handler, method_name):
                try:
                    stat_fig = statistic_fig.get(statistic, {})
                    getattr(stats_handler, method_name)(statistic, stats_nml.get(statistic, {}), stat_fig)
                    logger.info("Completed %s analysis", statistic)
                except Exception:
                    logger.exception("Failed %s analysis", statistic)
                    phase_errors.append(_make_phase_error("statistics", f"{statistic} analysis failed"))
            else:
                logger.warning("Statistics method %s not found, skipping", method_name)
                phase_errors.append(_make_phase_error("statistics", f"{statistic} analysis method not found"))

            gc.collect()

    except ImportError:
        logger.warning("StatisticsProcessing not available, skipping statistics phase")
        phase_errors.append(_make_phase_error("statistics", "statistics processing is not available"))
    except Exception:
        logger.exception("Statistics phase failed")
        phase_errors.append(_make_phase_error("statistics", "statistics phase failed"))

    return phase_errors


def _run_report(bindings, output_dir):
    """Generate evaluation report."""
    phase_errors = []
    try:
        from openbench.util.report import ReportGenerator
        report_config = bindings.build_report_config().to_report_config()

        report_gen = ReportGenerator(report_config, str(output_dir))
        report_paths = report_gen.generate_report()

        if report_paths:
            for fmt, path in report_paths.items():
                logger.info("Report generated: %s → %s", fmt, path)
        else:
            logger.info("Report generation completed")

    except ImportError:
        logger.warning("ReportGenerator not available, skipping report generation")
        phase_errors.append(_make_phase_error("report", "report generation is not available"))
    except Exception:
        logger.exception("Report generation failed")
        phase_errors.append(_make_phase_error("report", "report generation failed"))

    return phase_errors
