"""Uncertainty-aware post-processing for completed evaluation tasks."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import xarray as xr

from openbench.config.schema import UNCERTAINTY_METRIC_DIRECTIONS, OpenBenchConfig
from openbench.core.uncertainty import (
    bootstrap_grid_metric,
    bootstrap_network_metric,
    derived_seed,
    paired_grid_metric_difference,
    paired_network_metric_difference,
    verdict_from_reference_differences,
)
from openbench.runner.preflight import output_file_is_readable, task_output_data_types
from openbench.util.names import select_data_array
from openbench.util.netcdf import write_file_atomic, write_netcdf_atomic
from openbench.util.time import align_time_coordinates

logger = logging.getLogger(__name__)

MakePhaseError = Callable[..., dict[str, Any]]


def _write_json(path: Path, data: Any) -> None:
    def writer(temp_path: Path) -> None:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, allow_nan=False)
            handle.write("\n")

    write_file_atomic(path, writer, suffix=".tmp.json")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    write_file_atomic(path, lambda temp_path: frame.to_csv(temp_path, index=False), suffix=".tmp.csv")


def _source_varname(task: dict[str, Any], kind: str) -> str:
    source = str(task[f"{kind}_source"])
    section = getattr(task["bindings"].namelists, "reference" if kind == "ref" else "simulation")
    return str(section.get(str(task["var_name"]), {}).get(f"{source}_varname") or task["var_name"])


def _grid_arrays_and_weights(
    arrays: list[xr.DataArray],
    weight: str | None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    reference = arrays[-1]
    spatial_dims = [dim for dim in reference.dims if dim != "time"]
    if spatial_dims:
        template = xr.ones_like(reference.isel(time=0, drop=True), dtype=float)
        weights = template
        if weight in {"area", "mass"} and "lat" in reference.coords:
            weights = np.cos(np.deg2rad(reference["lat"])).clip(min=0).broadcast_like(template)
        if weight == "mass":
            weights = weights * abs(reference.mean("time", skipna=True))
        ordered = [array.transpose(*spatial_dims, "time") for array in arrays]
        values = [array.values.reshape(-1, array.sizes["time"]) for array in ordered]
        return values, weights.values.reshape(-1), reference["time"].values
    return (
        [array.values.reshape(1, array.sizes["time"]) for array in arrays],
        np.ones(1),
        reference["time"].values,
    )


def _common_grid_support(
    sim_a: xr.DataArray,
    ref_a: xr.DataArray,
    sim_b: xr.DataArray,
    ref_b: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray] | None:
    sim_a, ref_a, sim_b, ref_b = xr.align(sim_a, ref_a, sim_b, ref_b, join="inner")
    valid = np.isfinite(sim_a) & np.isfinite(ref_a) & np.isfinite(sim_b) & np.isfinite(ref_b)
    if not bool(valid.any()):
        return None
    valid_values = valid.values
    if not np.allclose(ref_a.values[valid_values], ref_b.values[valid_values]):
        return None
    return sim_a.where(valid), sim_b.where(valid), ref_a.where(valid)


def _load_grid_pair(
    task: dict[str, Any],
    output_dir: Path,
) -> tuple[xr.DataArray, xr.DataArray]:
    item = str(task["var_name"])
    sim_source = str(task["sim_source"])
    ref_source = str(task["ref_source"])
    sim_varname = _source_varname(task, "sim")
    ref_varname = _source_varname(task, "ref")
    sim_path = output_dir / "data" / f"{item}_sim_{sim_source}_{sim_varname}.nc"
    ref_path = Path(task.get("ref_file_override") or output_dir / "data" / f"{item}_ref_{ref_source}_{ref_varname}.nc")

    with xr.open_dataset(sim_path) as sim_ds, xr.open_dataset(ref_path) as ref_ds:
        sim = select_data_array(sim_ds, sim_varname, item)
        ref = select_data_array(ref_ds, ref_varname, item)
        sim, ref = xr.align(sim, ref, join="inner")
        if "time" not in sim.dims or sim.sizes.get("time", 0) == 0:
            raise ValueError("preprocessed grid pair has no common time samples")
        valid = np.isfinite(sim) & np.isfinite(ref)
        return sim.where(valid).load(), ref.where(valid).load()


def _load_station_pairs(
    task: dict[str, Any],
    output_dir: Path,
) -> dict[str, tuple[xr.DataArray, xr.DataArray]]:
    item = str(task["var_name"])
    folder = output_dir / "data" / f"stn_{task['ref_source']}_{task['sim_source']}"
    pairs: dict[str, tuple[xr.DataArray, xr.DataArray]] = {}
    runner_cfg = getattr(task["bindings"], "runner_cfg", None)
    resolution = getattr(runner_cfg, "general", {}).get("compare_tim_res") if runner_cfg is not None else None
    evaluation_path = output_dir / "metrics" / f"{item}_stn_{task['ref_source']}_{task['sim_source']}_evaluations.csv"
    station_rows = pd.read_csv(evaluation_path, dtype={"ID": str})
    if "ID" not in station_rows:
        raise ValueError(f"station evaluation output has no ID column: {evaluation_path}")

    for row in station_rows.to_dict(orient="records"):
        station_id = str(row["ID"])
        if {"use_syear", "use_eyear"} <= row.keys() and pd.notna(row["use_syear"]) and pd.notna(row["use_eyear"]):
            suffix = f"{station_id}_{int(row['use_syear'])}_{int(row['use_eyear'])}.nc"
            sim_path = folder / f"{item}_sim_{suffix}"
        else:
            matches = sorted(folder.glob(f"{item}_sim_{station_id}_*.nc"))
            if len(matches) != 1:
                logger.warning(
                    "Skipping station %s: expected one current simulation file, found %d",
                    station_id,
                    len(matches),
                )
                continue
            sim_path = matches[0]
            suffix = sim_path.name[len(f"{item}_sim_") :]
        ref_path = folder / f"{item}_ref_{suffix}"
        if not sim_path.is_file() or not ref_path.is_file():
            continue
        with xr.open_dataset(sim_path) as sim_ds, xr.open_dataset(ref_path) as ref_ds:
            sim = select_data_array(sim_ds, item).squeeze()
            ref = select_data_array(ref_ds, item).squeeze()
            sim, ref, normalized = align_time_coordinates(sim, ref, resolution)
            if normalized:
                logger.debug("Normalized station %s timestamps for uncertainty", station_id)
            if "time" not in sim.dims or sim.sizes.get("time", 0) == 0:
                continue
            pairs[station_id] = (sim.load(), ref.load())
    return pairs


def _bootstrap_rows(
    cfg: OpenBenchConfig,
    pair_data: dict[tuple[str, str, str], Any],
    pair_kinds: dict[tuple[str, str, str], str],
    pair_resolutions: dict[tuple[str, str, str], str | None],
    metrics: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    options = cfg.uncertainty
    for key, data in pair_data.items():
        item, ref_source, sim_source = key
        kind = pair_kinds[key]
        for metric in metrics:
            kwargs = {
                "n_resamples": options.n_resamples,
                "confidence_level": options.confidence_level,
                "block_length": options.block_length,
                "seed": derived_seed(options.seed, "aggregate", item, ref_source, sim_source, metric),
                "time_resolution": pair_resolutions[key],
            }
            if kind == "station":
                station_pairs = [(sim.values, ref.values, sim["time"].values) for sim, ref in data.values()]
                result = bootstrap_network_metric(station_pairs, metric, **kwargs)
                scope = "station_network"
            else:
                arrays, weights, time = _grid_arrays_and_weights(
                    [data[0], data[1]],
                    cfg.project.weight or "area",
                )
                result = bootstrap_grid_metric(
                    arrays[0],
                    arrays[1],
                    metric,
                    spatial_weights=weights,
                    time=time,
                    **kwargs,
                )
                scope = "evaluation_domain"
            rows.append(
                {
                    "variable": item,
                    "reference": ref_source,
                    "simulation": sim_source,
                    "metric": metric,
                    "scope": scope,
                    **result,
                }
            )
    return rows


def _aligned_station_triplets(
    first: dict[str, tuple[xr.DataArray, xr.DataArray]],
    second: dict[str, tuple[xr.DataArray, xr.DataArray]],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    triplets = []
    for station_id in sorted(first.keys() & second.keys()):
        sim_a, ref_a = first[station_id]
        sim_b, ref_b = second[station_id]
        sim_a, ref_a, sim_b, ref_b = xr.align(sim_a, ref_a, sim_b, ref_b, join="inner")
        if sim_a.sizes.get("time", 0) and np.allclose(ref_a.values, ref_b.values, equal_nan=True):
            triplets.append((sim_a.values, sim_b.values, ref_a.values, sim_a["time"].values))
        elif sim_a.sizes.get("time", 0):
            logger.warning(
                "Skipping station %s from paired verdict: reference values differ between models",
                station_id,
            )
    return triplets


def _verdicts(
    cfg: OpenBenchConfig,
    pair_data: dict[tuple[str, str, str], Any],
    pair_kinds: dict[tuple[str, str, str], str],
    pair_resolutions: dict[tuple[str, str, str], str | None],
    metrics: list[str],
) -> list[dict[str, Any]]:
    options = cfg.uncertainty
    verdicts: list[dict[str, Any]] = []
    variables = sorted({key[0] for key in pair_data})
    for item in variables:
        simulations = sorted({key[2] for key in pair_data if key[0] == item})
        for sim_a, sim_b in combinations(simulations, 2):
            references = sorted(
                {
                    key[1]
                    for key in pair_data
                    if key[0] == item and key[2] == sim_a and (item, key[1], sim_b) in pair_data
                }
            )
            for metric in metrics:
                differences = {}
                for ref_source in references:
                    key_a = (item, ref_source, sim_a)
                    key_b = (item, ref_source, sim_b)
                    kind = pair_kinds[key_a]
                    if pair_kinds[key_b] != kind:
                        differences[ref_source] = {
                            "status": "insufficient_data",
                            "reason": "simulation outputs use different spatial representations",
                        }
                        continue
                    kwargs = {
                        "n_resamples": options.n_resamples,
                        "confidence_level": options.confidence_level,
                        "block_length": options.block_length,
                        "seed": derived_seed(options.seed, "verdict", item, ref_source, sim_a, sim_b, metric),
                        "time_resolution": pair_resolutions[key_a],
                    }
                    if kind == "station":
                        triplets = _aligned_station_triplets(pair_data[key_a], pair_data[key_b])
                        differences[ref_source] = paired_network_metric_difference(triplets, metric, **kwargs)
                    else:
                        common = _common_grid_support(
                            pair_data[key_a][0],
                            pair_data[key_a][1],
                            pair_data[key_b][0],
                            pair_data[key_b][1],
                        )
                        if common is None:
                            differences[ref_source] = {
                                "status": "insufficient_data",
                                "reason": "models have no common support with an identical reference",
                            }
                            continue
                        arrays, weights, time = _grid_arrays_and_weights(
                            list(common),
                            cfg.project.weight or "area",
                        )
                        differences[ref_source] = paired_grid_metric_difference(
                            arrays[0],
                            arrays[1],
                            arrays[2],
                            metric,
                            spatial_weights=weights,
                            time=time,
                            **kwargs,
                        )
                verdict = verdict_from_reference_differences(
                    differences,
                    simulation_a=sim_a,
                    simulation_b=sim_b,
                )
                verdicts.append({"variable": item, "metric": metric, **verdict})
    return verdicts


def _metric_data_array(path: Path, metric: str) -> xr.DataArray:
    with xr.open_dataset(path) as dataset:
        return select_data_array(dataset, metric).load()


def _write_grid_spread(
    paths: list[tuple[str, xr.DataArray]],
    output_path: Path,
    *,
    axis: str,
    metric: str,
) -> None:
    names, arrays = zip(*paths)
    aligned = xr.align(*arrays, join="outer")
    stack = xr.concat(aligned, dim=xr.IndexVariable("member", list(names)))
    mean = stack.mean("member", skipna=True)
    count = stack.count("member")
    spread = stack.std("member", skipna=True).where(count >= 2)
    spread_name = "model_spread" if axis == "model" else "reference_sensitivity"
    mean_name = "ensemble_mean" if axis == "model" else "reference_mean"
    dataset = xr.Dataset(
        {
            mean_name: mean,
            spread_name: spread,
            "member_count": count,
            "coefficient_of_variation": xr.where(abs(mean) > 0, spread / abs(mean), np.nan),
        }
    )
    dataset.attrs.update({"uncertainty_axis": axis, "metric": metric, "members": ",".join(names)})
    write_netcdf_atomic(dataset, output_path)


def _write_station_spread(
    paths: list[tuple[str, Path]],
    output_path: Path,
    *,
    axis: str,
    metric: str,
) -> bool:
    series = []
    for name, path in paths:
        frame = pd.read_csv(path, dtype={"ID": str})
        if "ID" not in frame or metric not in frame:
            continue
        frame = frame.loc[frame["ID"].notna()]
        if axis == "reference":
            if not {"ref_lat", "ref_lon"} <= set(frame):
                continue
            lat = pd.to_numeric(frame["ref_lat"], errors="coerce").round(6)
            lon = pd.to_numeric(frame["ref_lon"], errors="coerce").round(6)
            located = lat.notna() & lon.notna()
            frame, lat, lon = frame.loc[located], lat.loc[located], lon.loc[located]
            index = pd.MultiIndex.from_arrays([lat, lon], names=["ref_lat", "ref_lon"])
        else:
            index = pd.Index(frame["ID"].astype(str), name="ID")
        values = pd.Series(pd.to_numeric(frame[metric], errors="coerce").values, index=index)
        series.append(values.groupby(level=list(range(values.index.nlevels))).mean().rename(name))
    if not series:
        output_path.unlink(missing_ok=True)
        return False
    members = pd.concat(series, axis=1)
    count = members.count(axis=1)
    comparable = count >= 2
    if not comparable.any():
        output_path.unlink(missing_ok=True)
        return False
    members = members.loc[comparable]
    count = count.loc[comparable]
    mean = members.mean(axis=1)
    spread_name = "model_spread" if axis == "model" else "reference_sensitivity"
    mean_name = "ensemble_mean" if axis == "model" else "reference_mean"
    spread = members.std(axis=1, ddof=0)
    result = pd.DataFrame(
        {
            mean_name: mean,
            spread_name: spread,
            "member_count": count,
            "coefficient_of_variation": spread / mean.abs().replace(0, np.nan),
        }
    ).reset_index()
    if "ID" not in result:
        result.insert(
            0,
            "ID",
            result["ref_lat"].map(lambda value: f"{value:.6f}")
            + ","
            + result["ref_lon"].map(lambda value: f"{value:.6f}"),
        )
    _write_csv(output_path, result)
    return True


def _write_spread_products(
    tasks: list[dict[str, Any]],
    output_dir: Path,
    metrics: list[str],
) -> dict[str, Any]:
    products: dict[str, Any] = {"model_spread": [], "reference_sensitivity": [], "skipped": []}
    uncertainty_dir = output_dir / "uncertainty"
    task_keys = {(str(task["var_name"]), str(task["ref_source"]), str(task["sim_source"])): task for task in tasks}
    variables = sorted({key[0] for key in task_keys})
    for item in variables:
        refs = sorted({key[1] for key in task_keys if key[0] == item})
        sims = sorted({key[2] for key in task_keys if key[0] == item})
        for metric in metrics:
            for ref_source in refs:
                members = [
                    (sim, task_keys[(item, ref_source, sim)]) for sim in sims if (item, ref_source, sim) in task_keys
                ]
                kinds = {"station" if "stn" in task_output_data_types(task) else "grid" for _, task in members}
                if len(members) < 2:
                    products["skipped"].append(
                        {
                            "axis": "model",
                            "variable": item,
                            "reference": ref_source,
                            "metric": metric,
                            "reason": "at least two simulations are required",
                        }
                    )
                elif len(kinds) > 1:
                    products["skipped"].append(
                        {
                            "axis": "model",
                            "variable": item,
                            "reference": ref_source,
                            "metric": metric,
                            "reason": "simulation outputs use different spatial representations",
                        }
                    )
                else:
                    task = members[0][1]
                    is_station = "stn" in task_output_data_types(task)
                    if is_station:
                        paths = [
                            (
                                sim,
                                output_dir / "metrics" / f"{item}_stn_{ref_source}_{sim}_evaluations.csv",
                            )
                            for sim, _ in members
                        ]
                        path = uncertainty_dir / "model_spread" / f"{item}_ref_{ref_source}_{metric}.csv"
                        if not _write_station_spread(paths, path, axis="model", metric=metric):
                            products["skipped"].append(
                                {
                                    "axis": "model",
                                    "variable": item,
                                    "reference": ref_source,
                                    "metric": metric,
                                    "reason": "fewer than two comparable station members",
                                }
                            )
                    else:
                        paths = [
                            (
                                sim,
                                _metric_data_array(
                                    output_dir / "metrics" / f"{item}_ref_{ref_source}_sim_{sim}_{metric}.nc",
                                    metric,
                                ),
                            )
                            for sim, _ in members
                        ]
                        path = uncertainty_dir / "model_spread" / f"{item}_ref_{ref_source}_{metric}.nc"
                        _write_grid_spread(paths, path, axis="model", metric=metric)
                    if path.is_file():
                        products["model_spread"].append(str(path.relative_to(output_dir)))

            for sim_source in sims:
                members = [
                    (ref, task_keys[(item, ref, sim_source)]) for ref in refs if (item, ref, sim_source) in task_keys
                ]
                kinds = {"station" if "stn" in task_output_data_types(task) else "grid" for _, task in members}
                if len(members) < 2:
                    products["skipped"].append(
                        {
                            "axis": "reference",
                            "variable": item,
                            "simulation": sim_source,
                            "metric": metric,
                            "reason": "at least two references are required",
                        }
                    )
                elif len(kinds) > 1:
                    products["skipped"].append(
                        {
                            "axis": "reference",
                            "variable": item,
                            "simulation": sim_source,
                            "metric": metric,
                            "reason": "reference outputs use different spatial representations",
                        }
                    )
                else:
                    task = members[0][1]
                    is_station = "stn" in task_output_data_types(task)
                    if is_station:
                        paths = [
                            (
                                ref,
                                output_dir / "metrics" / f"{item}_stn_{ref}_{sim_source}_evaluations.csv",
                            )
                            for ref, _ in members
                        ]
                        path = uncertainty_dir / "reference_sensitivity" / f"{item}_sim_{sim_source}_{metric}.csv"
                        if not _write_station_spread(paths, path, axis="reference", metric=metric):
                            products["skipped"].append(
                                {
                                    "axis": "reference",
                                    "variable": item,
                                    "simulation": sim_source,
                                    "metric": metric,
                                    "reason": "fewer than two colocated station references",
                                }
                            )
                    else:
                        paths = [
                            (
                                ref,
                                _metric_data_array(
                                    output_dir / "metrics" / f"{item}_ref_{ref}_sim_{sim_source}_{metric}.nc",
                                    metric,
                                ),
                            )
                            for ref, _ in members
                        ]
                        path = uncertainty_dir / "reference_sensitivity" / f"{item}_sim_{sim_source}_{metric}.nc"
                        _write_grid_spread(paths, path, axis="reference", metric=metric)
                    if path.is_file():
                        products["reference_sensitivity"].append(str(path.relative_to(output_dir)))
    return products


def _cached_uncertainty_outputs_complete(cfg: OpenBenchConfig, output_dir: Path) -> bool:
    uncertainty_dir = output_dir / "uncertainty"
    summary_path = uncertainty_dir / "summary.json"
    bootstrap_path = uncertainty_dir / "bootstrap_summary.csv"
    verdicts_path = uncertainty_dir / "verdicts.json"
    try:
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, ValueError):
        return False
    if not isinstance(summary, dict):
        return False
    if (
        summary.get("schema_version") != 1
        or not isinstance(summary.get("bootstrap"), list)
        or not isinstance(summary.get("verdicts"), list)
    ):
        return False
    if summary.get("config") != asdict(cfg.uncertainty) or not output_file_is_readable(bootstrap_path):
        return False
    try:
        with verdicts_path.open(encoding="utf-8") as handle:
            verdicts = json.load(handle)
    except (OSError, ValueError):
        return False
    if (
        not isinstance(verdicts, dict)
        or verdicts.get("schema_version") != 1
        or not isinstance(verdicts.get("verdicts"), list)
    ):
        return False
    products = summary.get("products", {})
    if not isinstance(products, dict):
        return False
    model_spread = products.get("model_spread", [])
    reference_sensitivity = products.get("reference_sensitivity", [])
    if not isinstance(model_spread, list) or not isinstance(reference_sensitivity, list):
        return False
    declared = [*model_spread, *reference_sensitivity]
    if not all(isinstance(path, str) for path in declared):
        return False
    return all(output_file_is_readable(output_dir / path) for path in declared)


def run_uncertainty(
    cfg: OpenBenchConfig,
    tasks: list[dict[str, Any]],
    output_dir: Path,
    metric_vars: list[str],
    *,
    make_phase_error_fn: MakePhaseError,
) -> list[dict[str, Any]]:
    """Generate aggregate bootstrap, spread/sensitivity products, and verdicts."""
    summary_path = output_dir / "uncertainty" / "summary.json"
    if (
        tasks
        and all(task.get("cache_skipped") for task in tasks)
        and _cached_uncertainty_outputs_complete(cfg, output_dir)
    ):
        logger.info("Reusing cached uncertainty outputs")
        return []
    summary_path.unlink(missing_ok=True)

    metrics = cfg.uncertainty.metrics or [metric for metric in metric_vars if metric in UNCERTAINTY_METRIC_DIRECTIONS]
    errors: list[dict[str, Any]] = []
    pair_data: dict[tuple[str, str, str], Any] = {}
    pair_kinds: dict[tuple[str, str, str], str] = {}
    pair_resolutions: dict[tuple[str, str, str], str | None] = {}
    for task in tasks:
        key = (str(task["var_name"]), str(task["ref_source"]), str(task["sim_source"]))
        try:
            is_station = "stn" in task_output_data_types(task)
            pair_kinds[key] = "station" if is_station else "grid"
            runner_cfg = getattr(task["bindings"], "runner_cfg", None)
            pair_resolutions[key] = (
                getattr(runner_cfg, "general", {}).get("compare_tim_res") if runner_cfg is not None else None
            )
            pair_data[key] = _load_station_pairs(task, output_dir) if is_station else _load_grid_pair(task, output_dir)
        except Exception as exc:
            logger.exception("Failed to load uncertainty inputs for %s", key)
            errors.append(
                make_phase_error_fn(
                    "uncertainty",
                    f"uncertainty input loading failed: {exc}",
                    variable=key[0],
                    ref=key[1],
                    sim=key[2],
                )
            )

    if not pair_data:
        return errors or [make_phase_error_fn("uncertainty", "no completed evaluation pairs were available")]

    try:
        bootstrap_rows = _bootstrap_rows(cfg, pair_data, pair_kinds, pair_resolutions, metrics)
        verdicts = _verdicts(cfg, pair_data, pair_kinds, pair_resolutions, metrics)
        completed_tasks = [
            task
            for task in tasks
            if (str(task["var_name"]), str(task["ref_source"]), str(task["sim_source"])) in pair_data
        ]
        products = _write_spread_products(
            completed_tasks,
            output_dir,
            metrics,
        )
        uncertainty_dir = output_dir / "uncertainty"
        _write_csv(uncertainty_dir / "bootstrap_summary.csv", pd.DataFrame(bootstrap_rows))
        _write_json(uncertainty_dir / "verdicts.json", {"schema_version": 1, "verdicts": verdicts})
        _write_json(
            summary_path,
            {
                "schema_version": 1,
                "config": asdict(cfg.uncertainty),
                "bootstrap": bootstrap_rows,
                "products": products,
                "verdicts": verdicts,
            },
        )
    except Exception as exc:
        logger.exception("Uncertainty phase failed")
        errors.append(make_phase_error_fn("uncertainty", f"uncertainty phase failed: {exc}"))
    return errors
