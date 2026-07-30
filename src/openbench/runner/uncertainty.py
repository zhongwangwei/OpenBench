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
    bootstrap_metric,
    bootstrap_network_metric,
    derived_seed,
    paired_metric_difference,
    paired_network_metric_difference,
    verdict_from_reference_differences,
)
from openbench.runner.preflight import task_output_data_types
from openbench.util.names import select_data_array
from openbench.util.netcdf import write_file_atomic, write_netcdf_atomic

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


def _spatial_means(
    sim: xr.DataArray,
    ref: xr.DataArray,
    weight: str | None,
) -> tuple[xr.DataArray, xr.DataArray]:
    spatial_dims = [dim for dim in sim.dims if dim != "time"]
    if not spatial_dims:
        return sim, ref
    if weight in {"area", "mass"} and "lat" in sim.coords and "lat" in spatial_dims:
        weights = np.cos(np.deg2rad(sim["lat"])).clip(min=0)
        if weight == "mass":
            weights = weights * abs(ref.mean("time", skipna=True))
        weights = weights.fillna(0)
        return sim.weighted(weights).mean(spatial_dims), ref.weighted(weights).mean(spatial_dims)
    return sim.mean(spatial_dims), ref.mean(spatial_dims)


def _load_grid_pair(
    task: dict[str, Any],
    output_dir: Path,
    weight: str | None,
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
        sim, ref = _spatial_means(sim.where(valid), ref.where(valid), weight)
        return sim.load(), ref.load()


def _load_station_pairs(
    task: dict[str, Any],
    output_dir: Path,
) -> dict[str, tuple[xr.DataArray, xr.DataArray]]:
    item = str(task["var_name"])
    folder = output_dir / "data" / f"stn_{task['ref_source']}_{task['sim_source']}"
    pairs: dict[str, tuple[xr.DataArray, xr.DataArray]] = {}
    prefix = f"{item}_sim_"
    for sim_path in sorted(folder.glob(f"{prefix}*.nc")):
        suffix = sim_path.name[len(prefix) :]
        ref_path = folder / f"{item}_ref_{suffix}"
        if not ref_path.is_file():
            continue
        with xr.open_dataset(sim_path) as sim_ds, xr.open_dataset(ref_path) as ref_ds:
            sim = select_data_array(sim_ds, item).squeeze()
            ref = select_data_array(ref_ds, item).squeeze()
            sim, ref = xr.align(sim, ref, join="inner")
            if "time" not in sim.dims or sim.sizes.get("time", 0) == 0:
                continue
            station_id = suffix.rsplit("_", 2)[0]
            pairs[station_id] = (sim.load(), ref.load())
    return pairs


def _bootstrap_rows(
    cfg: OpenBenchConfig,
    pair_data: dict[tuple[str, str, str], Any],
    pair_kinds: dict[tuple[str, str, str], str],
    metrics: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    options = cfg.uncertainty
    for (item, ref_source, sim_source), data in pair_data.items():
        kind = pair_kinds[(item, ref_source, sim_source)]
        for metric in metrics:
            kwargs = {
                "n_resamples": options.n_resamples,
                "confidence_level": options.confidence_level,
                "block_length": options.block_length,
                "seed": derived_seed(options.seed, "aggregate", item, ref_source, sim_source, metric),
            }
            if kind == "station":
                result = bootstrap_network_metric(list(data.values()), metric, **kwargs)
                scope = "station_network"
            else:
                result = bootstrap_metric(data[0].values, data[1].values, metric, **kwargs)
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
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    triplets = []
    for station_id in sorted(first.keys() & second.keys()):
        sim_a, ref_a = first[station_id]
        sim_b, ref_b = second[station_id]
        sim_a, ref_a, sim_b, ref_b = xr.align(sim_a, ref_a, sim_b, ref_b, join="inner")
        if sim_a.sizes.get("time", 0):
            triplets.append((sim_a.values, sim_b.values, ref_a.values))
    return triplets


def _verdicts(
    cfg: OpenBenchConfig,
    pair_data: dict[tuple[str, str, str], Any],
    pair_kinds: dict[tuple[str, str, str], str],
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
                    }
                    if kind == "station":
                        triplets = _aligned_station_triplets(pair_data[key_a], pair_data[key_b])
                        differences[ref_source] = paired_network_metric_difference(triplets, metric, **kwargs)
                    else:
                        sim_a_data, ref_a_data, sim_b_data, ref_b_data = xr.align(
                            pair_data[key_a][0],
                            pair_data[key_a][1],
                            pair_data[key_b][0],
                            pair_data[key_b][1],
                            join="inner",
                        )
                        differences[ref_source] = paired_metric_difference(
                            sim_a_data.values,
                            sim_b_data.values,
                            ref_a_data.values,
                            metric,
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
    spread = stack.std("member", skipna=True)
    count = stack.count("member")
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
) -> None:
    series = []
    for name, path in paths:
        frame = pd.read_csv(path)
        if "ID" not in frame or metric not in frame:
            continue
        series.append(pd.to_numeric(frame.set_index("ID")[metric], errors="coerce").rename(name))
    if not series:
        return
    members = pd.concat(series, axis=1)
    mean = members.mean(axis=1)
    spread_name = "model_spread" if axis == "model" else "reference_sensitivity"
    mean_name = "ensemble_mean" if axis == "model" else "reference_mean"
    spread = members.std(axis=1, ddof=0)
    result = pd.DataFrame(
        {
            "ID": members.index,
            mean_name: mean,
            spread_name: spread,
            "member_count": members.count(axis=1),
            "coefficient_of_variation": spread / mean.abs().replace(0, np.nan),
        }
    )
    _write_csv(output_path, result)


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
                    (sim, task_keys[(item, ref_source, sim)])
                    for sim in sims
                    if (item, ref_source, sim) in task_keys
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
                        _write_station_spread(paths, path, axis="model", metric=metric)
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
                    (ref, task_keys[(item, ref, sim_source)])
                    for ref in refs
                    if (item, ref, sim_source) in task_keys
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
                        _write_station_spread(paths, path, axis="reference", metric=metric)
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
    if tasks and all(task.get("cache_skipped") for task in tasks) and summary_path.is_file():
        logger.info("Reusing cached uncertainty outputs")
        return []

    metrics = cfg.uncertainty.metrics or [
        metric for metric in metric_vars if metric in UNCERTAINTY_METRIC_DIRECTIONS
    ]
    errors: list[dict[str, Any]] = []
    pair_data: dict[tuple[str, str, str], Any] = {}
    pair_kinds: dict[tuple[str, str, str], str] = {}
    for task in tasks:
        key = (str(task["var_name"]), str(task["ref_source"]), str(task["sim_source"]))
        try:
            is_station = "stn" in task_output_data_types(task)
            pair_kinds[key] = "station" if is_station else "grid"
            pair_data[key] = (
                _load_station_pairs(task, output_dir)
                if is_station
                else _load_grid_pair(task, output_dir, cfg.project.weight or "area")
            )
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
        bootstrap_rows = _bootstrap_rows(cfg, pair_data, pair_kinds, metrics)
        verdicts = _verdicts(cfg, pair_data, pair_kinds, metrics)
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
