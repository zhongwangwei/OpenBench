"""Task-hash payload construction for the local runner.

This module owns restart/cache invalidation inputs that are independent of the
runner orchestration loop: package/algorithm versions, regrid environment
signatures, raw-input file signatures, and the final per-task hash payload.
"""

from __future__ import annotations

import glob
import hashlib
import importlib
import inspect
import logging
import os
import shutil
from dataclasses import asdict, is_dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable

import openbench
from openbench.config.schema import OpenBenchConfig

logger = logging.getLogger(__name__)

OPENBENCH_ALGORITHM_VERSION = "2026-06-08.algorithm-source-v2"
ALGORITHM_SOURCE_MODULES = (
    "openbench.core.evaluation",
    "openbench.core.comparison",
    "openbench.core._comparison_basic",
    "openbench.core._comparison_common",
    "openbench.core._comparison_diagrams",
    "openbench.core._comparison_diff",
    "openbench.core._comparison_diff_grid",
    "openbench.core._comparison_diff_plot",
    "openbench.core._comparison_diff_station",
    "openbench.core._comparison_distributions",
    "openbench.core._comparison_heatmap",
    "openbench.core._comparison_helpers",
    "openbench.core._comparison_parallel",
    "openbench.core._comparison_portrait",
    "openbench.core._comparison_portrait_calculations",
    "openbench.core._comparison_portrait_seasonal",
    "openbench.core._comparison_relative",
    "openbench.core._comparison_smpi",
    "openbench.core._comparison_tail",
    "openbench.core._comparison_target",
    "openbench.core._comparison_taylor",
    "openbench.core.metrics",
    "openbench.core.scores",
    "openbench.core.statistics.Mod_Statistics",
    "openbench.core.statistics.base",
    "openbench.core.statistics.stat_Basic",
    "openbench.core.statistics.stat_False_Discovery_Rate",
    "openbench.core.statistics.stat_anova",
    "openbench.core.statistics.stat_autocorrelation",
    "openbench.core.statistics.stat_correlation",
    "openbench.core.statistics.stat_covariance",
    "openbench.core.statistics.stat_diff",
    "openbench.core.statistics.stat_functional_response",
    "openbench.core.statistics.stat_hellinger_distance",
    "openbench.core.statistics.stat_mann_kendall_trend_test",
    "openbench.core.statistics.stat_partial_least_squares_regression",
    "openbench.core.statistics.stat_resample",
    "openbench.core.statistics.stat_rolling",
    "openbench.core.statistics.stat_standard_deviation",
    "openbench.core.statistics.stat_three_cornered_hat",
    "openbench.core.statistics.stat_variance",
    "openbench.core.statistics.stat_z_score",
    "openbench.runner.masking",
    "openbench.data.climatology",
    "openbench.data.coordinates",
    "openbench.data.processing",
    "openbench.data.regrid",
    "openbench.data._processing_base",
    "openbench.data._processing_config",
    "openbench.data._processing_grid",
    "openbench.data._processing_grid_core",
    "openbench.data._processing_grid_regrid",
    "openbench.data._processing_selection",
    "openbench.data._processing_station",
    "openbench.data._processing_station_core",
    "openbench.data._processing_station_extract",
    "openbench.data._processing_time",
    "openbench.data._processing_time_adjustments",
    "openbench.data._processing_time_core",
    "openbench.data._processing_time_integrity",
    "openbench.data._processing_transforms",
    "openbench.data._processing_utils",
    "openbench.data._processing_yearly",
)


def source_specific_section(section: dict[str, Any], source: str) -> dict[str, Any]:
    """Return legacy namelist keys owned by one reference/simulation source."""
    prefix = f"{source}_"
    return {key: value for key, value in section.items() if str(key).startswith(prefix)}


def legacy_source_value(section: dict[str, Any], source: str, field: str, default: Any = "") -> Any:
    return section.get(f"{source}_{field}", default)


def file_sample_sha256(path: Path, *, chunk_bytes: int = 8192) -> str | None:
    """Return a cheap content digest sampled from head/middle/tail blocks.

    NetCDF rewrites often leave headers unchanged.  Hashing only the first
    block misses in-place data-section edits when size and mtime are preserved
    by tools such as rsync --times or NCO.  A sampled digest remains cheap for
    large climate files while catching edits outside the header.
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size <= chunk_bytes * 3:
                return hashlib.sha256(handle.read()).hexdigest()[:24]

            digest = hashlib.sha256()
            offsets = (0, max(0, size // 2 - chunk_bytes // 2), max(0, size - chunk_bytes))
            for offset in offsets:
                handle.seek(offset)
                digest.update(offset.to_bytes(8, "big", signed=False))
                digest.update(handle.read(chunk_bytes))
            return digest.hexdigest()[:24]
    except OSError as exc:
        logger.debug("Could not hash input file %s for cache signature: %s", path, exc)
        return None


def package_version(package: str) -> str | None:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return None


def openbench_version() -> str:
    """Return installed distribution version, falling back to package version."""
    for package in ("colm-openbench", "openbench"):
        version = package_version(package)
        if version:
            return version
    return getattr(openbench, "__version__", "unknown")


def algorithm_source_fingerprint() -> str:
    """Return a digest of source modules that define metric/score semantics."""
    digest = hashlib.sha256()
    for module_name in ALGORITHM_SOURCE_MODULES:
        digest.update(module_name.encode("utf-8"))
        try:
            module = importlib.import_module(module_name)
            source = inspect.getsource(module)
        except (ImportError, OSError, TypeError) as exc:
            logger.warning("Could not fingerprint algorithm source module %s: %s", module_name, exc)
            source = f"<unavailable:{type(exc).__name__}>"
        digest.update(source.encode("utf-8"))
    return digest.hexdigest()[:24]


def regrid_backend_signature() -> dict[str, Any]:
    """Return environment-sensitive regrid backend data for cache hashes."""
    cdo_path = shutil.which("cdo")
    return {
        "available_backends": [
            "openbench_conservative",
            "cdo_remapcon",
            "xesmf_conservative",
            "basic_interpolation",
        ],
        "openbench_conservative": True,
        "cdo": {"available": cdo_path is not None, "path": cdo_path},
        "xesmf": {"version": package_version("xesmf")},
        "scipy": {"version": package_version("scipy")},
    }


def selected_regrid_backend_signature(selected_backend: str, signature: dict[str, Any]) -> dict[str, Any]:
    """Return only the environment fields that can affect the selected backend."""
    selected = str(selected_backend or "openbench_conservative")
    if selected == "cdo_remapcon":
        return {"cdo": signature.get("cdo", {})}
    if selected == "xesmf_conservative":
        return {"xesmf": signature.get("xesmf", {})}
    if selected == "basic_interpolation":
        return {"scipy": signature.get("scipy", {})}
    return {"openbench_conservative": signature.get("openbench_conservative", True)}


def configured_regrid_backend(cfg: OpenBenchConfig, general: dict[str, Any]) -> str:
    """Return the deterministic regrid backend selected for this run."""
    return str(
        general.get("regrid_backend")
        or getattr(getattr(cfg, "project", None), "regrid_backend", None)
        or "openbench_conservative"
    )


def input_file_signature(section: dict[str, Any], source: str) -> dict[str, Any]:
    """Return cheap file metadata for raw inputs referenced by one source."""
    root = legacy_source_value(section, source, "dir")
    if not root:
        return {"files": []}

    root_path = Path(os.path.expanduser(os.path.expandvars(str(root))))
    candidates: set[Path] = set()

    fulllist = legacy_source_value(section, source, "fulllist")
    if fulllist:
        list_path = Path(os.path.expanduser(os.path.expandvars(str(fulllist))))
        if not list_path.is_absolute():
            list_path = root_path / list_path
        candidates.add(list_path)

    if root_path.exists():
        prefix = str(legacy_source_value(section, source, "prefix", "") or "")
        suffix = str(legacy_source_value(section, source, "suffix", "") or "")
        varname = str(legacy_source_value(section, source, "varname", "") or "")
        escaped_prefix = glob.escape(prefix)
        escaped_varname = glob.escape(varname)
        escaped_suffix = glob.escape(suffix)
        patterns = [
            f"{escaped_prefix}*{escaped_varname}*{escaped_suffix}*.nc",
            f"{escaped_prefix}*{escaped_varname}*{escaped_suffix}*.nc4",
            f"{escaped_prefix}*{escaped_varname}*{escaped_suffix}*.NC",
            f"{escaped_prefix}*{escaped_varname}*{escaped_suffix}*.NC4",
        ]
        try:
            for pattern in patterns:
                while "**" in pattern:
                    pattern = pattern.replace("**", "*")
                candidates.update(root_path.glob(pattern))
                candidates.update(root_path.rglob(pattern))
            if not candidates:
                # If registry metadata cannot express the on-disk layout
                # (year/region subdirectories, variable-free filenames, etc.),
                # hash all NetCDF inputs under the source root rather than
                # returning an empty signature that can falsely hit old cache.
                for pattern in ("*.nc", "*.nc4", "*.NC", "*.NC4"):
                    candidates.update(root_path.rglob(pattern))
        except OSError as exc:
            logger.debug("Could not stat input directory %s for cache signature: %s", root_path, exc)

    files: list[dict[str, Any]] = []
    for path in sorted(candidates, key=lambda item: str(item)):
        try:
            stat = path.stat()
        except OSError:
            continue
        if not path.is_file():
            continue
        files.append(
            {
                "path": str(path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
                "sample_sha256": file_sample_sha256(path),
            }
        )
    return {"files": files}


def stable_hash_data(value: Any) -> Any:
    """Convert dataclass-rich config objects into JSON-stable data."""
    if is_dataclass(value) and not isinstance(value, type):
        return stable_hash_data(asdict(value))
    if isinstance(value, dict):
        return {key: stable_hash_data(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [stable_hash_data(item) for item in value]
    return value


def shared_mask_peer_payload(
    *,
    cfg: OpenBenchConfig,
    bindings: Any,
    var_name: str,
    ref_source: str,
) -> dict[str, Any] | None:
    """Return peer simulation inputs that affect a shared unified mask."""
    alignment = getattr(cfg.project, "time_alignment", "intersection")
    if not getattr(cfg.project, "unified_mask", True) or alignment not in {"intersection", "strict"}:
        return None

    namelists = getattr(bindings, "namelists", None)
    ref_section: dict[str, Any] = {}
    sim_section: dict[str, Any] = {}
    sim_general: dict[str, Any] = {}
    if namelists is not None:
        ref_section = namelists.reference.get(var_name, {})
        sim_section = namelists.simulation.get(var_name, {})
        sim_general = namelists.simulation.get("general", {})

    ref_dtype = str(ref_section.get(f"{ref_source}_data_type", "grid"))
    if ref_dtype == "stn":
        return None

    sim_sources = sim_general.get(f"{var_name}_sim_source")
    if isinstance(sim_sources, str):
        candidate_sources = [sim_sources]
    elif sim_sources:
        candidate_sources = list(sim_sources)
    else:
        candidate_sources = list(cfg.simulation.keys())

    peers: list[dict[str, Any]] = []
    for peer_source in sorted(str(source) for source in candidate_sources):
        sim_entry = cfg.simulation.get(peer_source)
        if sim_entry is None:
            continue
        sim_dtype = str(
            sim_section.get(
                f"{peer_source}_data_type",
                getattr(sim_entry, "data_type", None) or "grid",
            )
        )
        if sim_dtype == "stn":
            continue
        peers.append(
            {
                "sim_source": peer_source,
                "config": stable_hash_data(sim_entry),
                "namelist": stable_hash_data(source_specific_section(sim_section, peer_source)),
                "inputs": input_file_signature(sim_section, peer_source),
            }
        )

    if not peers:
        return None

    return {
        "mode": alignment,
        "ref_source": ref_source,
        "ref_data_type": ref_dtype,
        "peers": peers,
    }


def task_hash_payload(
    *,
    cfg: OpenBenchConfig,
    bindings: Any,
    var_name: str,
    sim_source: str,
    ref_source: str,
    metric_vars: list[str],
    score_vars: list[str],
    comparison_vars: list[str],
    statistic_vars: list[str],
    openbench_version_fn: Callable[[], str] = openbench_version,
    regrid_backend_signature_fn: Callable[[], dict[str, Any]] = regrid_backend_signature,
) -> dict[str, Any]:
    """Build the runtime-sensitive payload used for one task's cache hash."""
    runner_cfg = bindings.runner_cfg
    general = runner_cfg.general
    namelists = getattr(bindings, "namelists", None)
    ref_section = {}
    sim_section = {}
    if namelists is not None:
        ref_section = source_specific_section(namelists.reference.get(var_name, {}), ref_source)
        sim_section = source_specific_section(namelists.simulation.get(var_name, {}), sim_source)

    general_keys = (
        "syear",
        "eyear",
        "min_year",
        "min_lat",
        "max_lat",
        "min_lon",
        "max_lon",
        "compare_tim_res",
        "compare_grid_res",
        "compare_tzone",
        "weight",
        "unified_mask",
        "time_alignment",
        "regrid_backend",
        "only_drawing",
    )

    sim_entry = cfg.simulation.get(sim_source)
    selected_regrid_backend = configured_regrid_backend(cfg, general)
    return {
        "variable": var_name,
        "sim_source": sim_source,
        "ref_source": ref_source,
        "metrics": metric_vars,
        "scores": score_vars,
        "comparisons": comparison_vars,
        "statistics": statistic_vars,
        "openbench": {
            "version": openbench_version_fn(),
            "algorithm_version": OPENBENCH_ALGORITHM_VERSION,
            "source_fingerprint": algorithm_source_fingerprint(),
        },
        "general": {key: general.get(key) for key in general_keys},
        "project": {
            "years": list(cfg.project.years),
            "lat_range": list(cfg.project.lat_range),
            "lon_range": list(cfg.project.lon_range),
            "tim_res": cfg.project.tim_res,
            "grid_res": cfg.project.grid_res,
            "timezone": cfg.project.timezone,
            "weight": cfg.project.weight,
            "min_year_threshold": cfg.project.min_year_threshold,
            "unified_mask": cfg.project.unified_mask,
            "time_alignment": cfg.project.time_alignment,
            "regrid_backend": cfg.project.regrid_backend,
            "only_drawing": cfg.project.only_drawing,
        },
        "regrid_backend": {
            "selected": selected_regrid_backend,
            "environment": selected_regrid_backend_signature(
                selected_regrid_backend,
                regrid_backend_signature_fn(),
            ),
        },
        "reference": {
            "data_root": cfg.reference.data_root,
            "source": cfg.reference.sources.get(var_name),
            "namelist": ref_section,
            "inputs": input_file_signature(
                namelists.reference.get(var_name, {}) if namelists is not None else {},
                ref_source,
            ),
        },
        "simulation": {
            "config": stable_hash_data(sim_entry),
            "namelist": sim_section,
            "inputs": input_file_signature(
                namelists.simulation.get(var_name, {}) if namelists is not None else {},
                sim_source,
            ),
        },
        "shared_unified_mask": shared_mask_peer_payload(
            cfg=cfg,
            bindings=bindings,
            var_name=var_name,
            ref_source=ref_source,
        ),
    }
