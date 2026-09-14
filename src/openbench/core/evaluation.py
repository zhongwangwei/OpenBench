import gc
import importlib
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

from openbench.data._system_resources import effective_cpu_count, get_system_resources
from openbench.data.station_missing import StationDataUnavailable
from openbench.data.time_utils import align_station_times, normalize_station_time

try:
    from openbench.util.dataset_loader import open_dataset as open_dataset_chunked
except ImportError:
    # Fallback to standard xr.open_dataset; strip use_chunking if passed
    # (only the chunked loader understands it — bare xr.open_dataset
    # would raise TypeError: unexpected keyword argument).
    def open_dataset_chunked(path, *args, **kwargs):
        kwargs.pop("use_chunking", None)
        return xr.open_dataset(path, *args, **kwargs)


try:
    from openbench.util.parallel import (  # noqa: F401  feature-detection imports
        ParallelEngine,
        get_parallel_engine,
        parallel_decorator,
        parallel_map,
    )

    _HAS_PARALLEL_ENGINE = True
except ImportError:
    _HAS_PARALLEL_ENGINE = False
    ParallelEngine = None

    def get_parallel_engine(*args, **kwargs):
        return None

    def parallel_map(*args, **kwargs):
        # Fallback to sequential processing
        func = args[0]
        items = args[1]
        return [func(item) for item in items]


# Import CacheSystem - CacheSystem is mandatory for evaluation engine
try:
    from openbench.data.cache import get_cache_manager  # noqa: F401  feature detection

    _HAS_CACHE = True
except ImportError:
    raise RuntimeError(
        "CacheSystem is required for evaluation engine (务必使用CacheSystem). "
        "Please ensure openbench.data.cache is available."
    )

from openbench.util.converttype import Convert_Type
from openbench.util.netcdf import write_file_atomic as _write_file_atomic
from openbench.util.netcdf import write_netcdf_atomic as _write_netcdf_atomic
from openbench.core._visualization_bridge import visualization_callable
from openbench.core.metrics import metrics
from openbench.core.scores import scores


make_plot_index_grid = visualization_callable("make_plot_index_grid")
make_plot_index_stn = visualization_callable("make_plot_index_stn")
plot_stn = visualization_callable("plot_stn")

try:
    from openbench.data.climatology import ClimatologyProcessor, process_climatology_evaluation

    _HAS_CLIMATOLOGY = True
except ImportError:
    _HAS_CLIMATOLOGY = False
    ClimatologyProcessor = None

    def process_climatology_evaluation(*args, **kwargs):
        return args[0], args[1], args[2]


try:
    from openbench.util.output import ModularOutputManager, create_output_manager, save_evaluation_results

    _HAS_OUTPUT_MANAGER = True
except ImportError:
    _HAS_OUTPUT_MANAGER = False
    ModularOutputManager = object

    def create_output_manager(*args, **kwargs):
        return None

    def save_evaluation_results(*args, **kwargs):
        return ""


_MFM_METRIC_NAMES = {"MFM", "MFM_omega", "MFM_varphi", "MFM_eta"}


def _mfm_shared_metric_names(metric_names) -> set[str]:
    names = {name for name in metric_names if name in _MFM_METRIC_NAMES}
    return names if "MFM" in names and len(names) > 1 else set()


def _metric_worker_count(num_cores, metric_count: int, pair_nbytes: int = 0) -> int:
    """Return metric-level workers bounded by configured cores and metric count."""
    if metric_count <= 1:
        return 1
    try:
        requested = int(num_cores) if num_cores is not None else 0
    except (TypeError, ValueError):
        requested = 1
    available = effective_cpu_count(os.cpu_count() or 1)
    if requested <= 0:
        requested = available
    workers = min(max(1, requested), available, metric_count)
    if pair_nbytes > 0:
        available_bytes = get_system_resources()["available_memory_gb"] * 1024**3
        # Metric reductions commonly need several pair-sized temporaries.
        memory_workers = max(1, int((available_bytes * 0.25) // (pair_nbytes * 4)))
        workers = min(workers, memory_workers)
    return workers


def _apply_pairwise_valid_mask(s: xr.DataArray, o: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Mask sim/ref arrays to their shared finite support without eager loads."""
    valid = np.isfinite(s) & np.isfinite(o)
    return s.where(valid), o.where(valid)


def _has_any_valid_pair(s: xr.DataArray, o: xr.DataArray) -> bool:
    """Return whether sim/ref arrays share at least one finite pair."""
    valid = np.isfinite(s) & np.isfinite(o)
    try:
        any_valid = valid.any()
        if hasattr(any_valid, "compute"):
            any_valid = any_valid.compute()
        if hasattr(any_valid, "item"):
            return bool(any_valid.item())
        return bool(any_valid)
    except Exception as exc:
        logging.debug("Could not determine valid pair count before evaluation: %s", exc)
        return True


def _scalar_plot_value(value, *, label: str, station_id: object) -> float:
    """Return a scalar value for station plot annotations without crashing on arrays."""
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        logging.warning("Station %s %s is not numeric; using NaN for station plot", station_id, label)
        return float("nan")
    if array.size == 0:
        logging.warning("Station %s %s is empty; using NaN for station plot", station_id, label)
        return float("nan")
    if array.size == 1:
        return float(array.reshape(-1)[0])
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        logging.warning("Station %s %s has no finite values; using NaN for station plot", station_id, label)
        return float("nan")
    logging.warning(
        "Station %s %s returned %d values for a scalar plot annotation; using finite mean",
        station_id,
        label,
        array.size,
    )
    return float(np.nanmean(finite))


def _grid_output_array(value, template: xr.DataArray, name: str) -> xr.DataArray:
    """Return a named grid output without squeezing singleton lat/lon axes."""
    if isinstance(value, xr.DataArray):
        da = value.rename(name)
    else:
        spatial = template
        drop = {dim: 0 for dim in spatial.dims if dim not in {"lat", "lon"}}
        if drop:
            spatial = spatial.isel(drop=True, **drop)
        data = np.asarray(value)
        if data.shape == ():
            da = xr.full_like(spatial, float(data), dtype=float).rename(name)
        else:
            da = xr.DataArray(data, coords=[template.lat, template.lon], dims=["lat", "lon"], name=name)

    squeeze_dims = [dim for dim, size in da.sizes.items() if size == 1 and dim not in {"lat", "lon"}]
    if squeeze_dims:
        da = da.squeeze(squeeze_dims, drop=True)
    if "lat" in da.dims and "lon" in da.dims:
        da = da.transpose("lat", "lon", ...)
    return da


class Evaluation_grid(metrics, scores):
    def _calculate_metric(self, s, o, metric, shared_metrics=None):
        """Helper method for parallel metric calculation."""
        if not hasattr(self, metric):
            raise ValueError(f"No such metric: {metric}")
        if shared_metrics is not None and metric in shared_metrics:
            self._save_metric_array(metric, shared_metrics[metric])
        else:
            self.process_metric(metric, s, o)
        return metric

    def __init__(self, info, fig_nml):
        self.name = "Evaluation_grid"
        self.version = "0.1"
        self.release = "0.1"
        self.date = "Mar 2023"
        self.author = "Zhongwang Wei / zhongwang007@gmail.com"
        self.__dict__.update(info)
        self.fig_nml = fig_nml
        os.makedirs(self.casedir, exist_ok=True)

        # NOTE: the previous modular evaluation-engine experiment was not
        # wired into runtime execution. The active evaluation path is kept
        # here rather than shadowed by an unused public engine abstraction.

        if _HAS_OUTPUT_MANAGER:
            self.output_manager = create_output_manager(self.casedir)
            logging.debug("Output manager initialized")
        else:
            self.output_manager = None

        logging.info(" ")
        logging.info("╔═══════════════════════════════════════════════════════════════╗")
        logging.info("║                Evaluation processes starting!                 ║")
        logging.info("╚═══════════════════════════════════════════════════════════════╝")
        logging.info(" ")

    def _align_grid_times(self, s, o):
        """Align grid evaluation times without silently pairing mismatched timestamps."""
        if "time" not in s.coords or "time" not in o.coords:
            return s, o
        if len(s["time"]) == len(o["time"]) and np.array_equal(s["time"].values, o["time"].values):
            return s, o

        message = (
            f"time coordinate mismatch for {getattr(self, 'item', 'grid evaluation')} "
            f"(ref={len(o['time'])}, sim={len(s['time'])})"
        )
        if getattr(self, "time_alignment", "intersection") == "strict":
            raise ValueError(message)

        o_aligned, s_aligned = xr.align(o, s, join="inner")
        if "time" in o_aligned.sizes and o_aligned.sizes["time"] == 0:
            raise ValueError(f"{message}: no overlapping timestamps")
        logging.warning("%s; using %d overlapping timestamps", message, o_aligned.sizes.get("time", 0))
        return s_aligned, o_aligned

    def _save_metric_array(self, metric, pb_da, vkey=""):
        # ponytail: compute before HDF5 write; lazy source reads inside to_netcdf can hang on Windows/netCDF4.
        pb_da = pb_da.load()

        if self.output_manager:
            filename = f"{self.item}_ref_{self.ref_source}_sim_{self.sim_source}_{metric}{vkey}"
            metadata = {
                "metric": metric,
                "item": self.item,
                "ref_source": self.ref_source,
                "sim_source": self.sim_source,
                "variable_key": vkey,
            }
            self.output_manager.save_data(pb_da, "metrics", filename, "netcdf", metadata)
        else:
            output_path = os.path.join(
                self.casedir,
                "metrics",
                f"{self.item}_ref_{self.ref_source}_sim_{self.sim_source}_{metric}{vkey}.nc",
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            _write_netcdf_atomic(pb_da, output_path)
            logging.info(f"Saved metric {metric} to {output_path}")

    def process_metric(self, metric, s, o, vkey=""):
        pb = getattr(self, metric)(s, o)
        self._save_metric_array(metric, _grid_output_array(pb, o, metric), vkey)

    def _prepare_mfm_shared_metrics(self, s, o):
        shared_names = _mfm_shared_metric_names(self.metrics)
        if not shared_names:
            return None
        shared = self._MFM_shared_components(s, o)
        return xr.Dataset({name: _grid_output_array(shared[name], o, name) for name in shared_names}).load()

    def _process_metrics_in_order(self, s, o, shared_ds=None):
        if shared_ds is None:
            shared_ds = self._prepare_mfm_shared_metrics(s, o)

        for metric in self.metrics:
            if hasattr(self, metric):
                logging.info(f"Calculating metric: {metric}")
                if shared_ds is not None and metric in shared_ds:
                    self._save_metric_array(metric, shared_ds[metric])
                else:
                    self.process_metric(metric, s, o)
            else:
                logging.error(f"No such metric: {metric}; skipping")

    def process_score(self, score, s, o, vkey=""):
        pb = getattr(self, score)(s, o)
        pb_da = _grid_output_array(pb, o, score)
        # ponytail: compute before HDF5 write; lazy source reads inside to_netcdf can hang on Windows/netCDF4.
        pb_da = pb_da.load()

        if self.output_manager:
            filename = f"{self.item}_ref_{self.ref_source}_sim_{self.sim_source}_{score}{vkey}"
            metadata = {
                "score": score,
                "item": self.item,
                "ref_source": self.ref_source,
                "sim_source": self.sim_source,
                "variable_key": vkey,
            }
            self.output_manager.save_data(pb_da, "scores", filename, "netcdf", metadata)
        else:
            output_path = os.path.join(
                self.casedir, "scores", f"{self.item}_ref_{self.ref_source}_sim_{self.sim_source}_{score}{vkey}.nc"
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            _write_netcdf_atomic(pb_da, output_path)
            logging.info(f"Saved score {score} to {output_path}")

    def make_Evaluation(self, **kwargs):
        ref_ds = None
        sim_ds = None
        try:
            ref_path = getattr(self, "ref_file_override", None) or os.path.join(
                self.casedir, "data", f"{self.item}_ref_{self.ref_source}_{self.ref_varname}.nc"
            )
            sim_path = os.path.join(self.casedir, "data", f"{self.item}_sim_{self.sim_source}_{self.sim_varname}.nc")

            ref_ds = open_dataset_chunked(ref_path)
            sim_ds = open_dataset_chunked(sim_path)
            # Resolve the variable robustly: a fallback/convert may have stored
            # the data under the relabelled item name (e.g. Net_Ecosystem_Exchange)
            # rather than the configured varname (e.g. f_respc). Flat files hold a
            # single data variable, so fall back to the item name / sole variable.
            from openbench.util.names import select_data_array

            o = select_data_array(ref_ds, self.ref_varname, self.item)
            s = select_data_array(sim_ds, self.sim_varname, self.item)
            o = Convert_Type.convert_nc(o)
            s = Convert_Type.convert_nc(s)

            if _HAS_CLIMATOLOGY:
                original_metrics = self.metrics.copy() if hasattr(self.metrics, "copy") else list(self.metrics)
                original_scores = self.scores.copy() if hasattr(self.scores, "copy") else list(self.scores)

                all_evaluations = list(self.metrics) + list(self.scores)

                compare_tim_res = getattr(self, "compare_tim_res", None)
                syear = getattr(self, "syear", None)
                if syear:
                    try:
                        syear = int(syear)
                    except (ValueError, TypeError):
                        syear = None

                o_clim, s_clim, supported_evaluations = process_climatology_evaluation(
                    ref_ds,
                    sim_ds,
                    all_evaluations,
                    compare_tim_res=compare_tim_res,
                    syear=syear,
                    ref_tim_res=getattr(self, "ref_tim_res", None),
                    sim_tim_res=getattr(self, "sim_tim_res", None),
                )

                if o_clim is not None and s_clim is not None:
                    logging.info("=" * 80)
                    logging.info("CLIMATOLOGY EVALUATION MODE DETECTED")
                    logging.info("=" * 80)

                    o = select_data_array(o_clim, self.ref_varname, self.item)
                    s = select_data_array(s_clim, self.sim_varname, self.item)
                    o = Convert_Type.convert_nc(o)
                    s = Convert_Type.convert_nc(s)

                    supported_set = set(supported_evaluations)
                    skipped_metrics = set(original_metrics) - supported_set
                    skipped_scores = set(original_scores) - supported_set
                    if skipped_metrics:
                        raise ValueError(
                            f"Unsupported climatology metric(s) requested for {self.item}: {sorted(skipped_metrics)}"
                        )

                    if skipped_scores:
                        raise ValueError(
                            f"Unsupported climatology score(s) requested for {self.item}: {sorted(skipped_scores)}"
                        )

                    # Update metrics and scores after validating that no
                    # user-requested evaluation was silently dropped.
                    self.metrics = [m for m in self.metrics if m in supported_set]
                    self.scores = [sc for sc in self.scores if sc in supported_set]

                    logging.info("=" * 80)
                else:
                    s, o = self._align_grid_times(s, o)
            else:
                s, o = self._align_grid_times(s, o)

            if self.item == "Terrestrial_Water_Storage_Change":
                logging.info("Processing Terrestrial Water Storage Change...")
                # Calculate time difference on a derived object. Do not mutate
                # the source array view or write back to the preprocessed file;
                # repeated evaluations must not permanently re-difference TWS.
                s = s - s.shift(time=1)

            if not _has_any_valid_pair(s, o):
                logging.warning(
                    "Skipping %s evaluation for sim=%s ref=%s: no shared finite sim/ref pairs",
                    self.item,
                    self.sim_source,
                    self.ref_source,
                )
                return

            logging.info("=" * 80)

            # Parallel processing of metrics if configured and beneficial.
            # Honor project.num_cores instead of the old hard-coded max=4.
            metric_workers = _metric_worker_count(
                getattr(self, "num_cores", 1),
                len(self.metrics),
                int(s.nbytes + o.nbytes),
            )
            shared_metrics = self._prepare_mfm_shared_metrics(s, o)
            if _HAS_PARALLEL_ENGINE and metric_workers > 1:
                logging.info("Processing %d metrics in parallel with %d worker(s)", len(self.metrics), metric_workers)
                from functools import partial

                metric_func = partial(self._calculate_metric, s, o, shared_metrics=shared_metrics)
                metric_results = parallel_map(
                    metric_func,
                    self.metrics,
                    task_name="Calculating metrics",
                    show_progress=False,
                    max_workers=metric_workers,
                    backend="threading",
                )
                for metric, result in zip(self.metrics, metric_results):
                    if result is not None:
                        logging.info(f"Calculated metric: {metric}")
            else:
                # Sequential processing — log + skip unknown metrics to
                # match the parallel path (which never sys.exits). A typo
                # in one metric name should not abort an entire run.
                self._process_metrics_in_order(s, o, shared_metrics)

            # Process scores (usually fewer, so sequential is fine)
            for score in self.scores:
                if hasattr(self, score):
                    logging.info(f"Calculating score: {score}")
                    self.process_score(score, s, o)
                else:
                    logging.error(f"No such score: {score}; skipping")

            logging.info("=" * 80)
            make_plot_index_grid(self)
        finally:
            if ref_ds is not None:
                ref_ds.close()
            if sim_ds is not None:
                sim_ds.close()
            gc.collect()  # Final cleanup


class Evaluation_stn(metrics, scores):
    def __init__(self, info, fig_nml):
        self.name = "Evaluation_point"
        self.version = "0.1"
        self.release = "0.1"
        self.date = "Mar 2023"
        self.author = "Zhongwang Wei / zhongwang007@gmail.com"
        self.fig_nml = fig_nml
        self.__dict__.update(info)
        if isinstance(self.sim_varname, str):
            self.sim_varname = [self.sim_varname]
        if isinstance(self.ref_varname, str):
            self.ref_varname = [self.ref_varname]

        # See note in `__init__` of EvaluationGrid above — the modular
        # engine was never actually used; the dead station-side
        # assignment is removed for the same reason.

        if _HAS_OUTPUT_MANAGER:
            self.output_manager = create_output_manager(self.casedir)
            logging.debug("Output manager initialized")
        else:
            self.output_manager = None

        logging.info("Evaluation processes starting!")
        logging.info("=======================================")
        logging.info(" ")
        logging.info(" ")

    @staticmethod
    def _normalize_var_selector(selector):
        if isinstance(selector, (list, tuple)):
            return list(selector)
        if isinstance(selector, np.ndarray):
            return selector.tolist()
        return [selector]

    def _load_station_dataset(self, dataset, datasource):
        attr_name = "ref_varname" if datasource == "ref" else "sim_varname"
        selector = getattr(self, attr_name)
        selector_list = self._normalize_var_selector(selector)

        if not selector_list:
            raise KeyError(f"Variable selector for {attr_name} is empty")

        try:
            return dataset[selector_list]
        except KeyError:
            fallback = self._apply_station_custom_filter(dataset, datasource, attr_name, selector_list[0])
            if fallback is not None:
                return fallback

            data_vars = list(dataset.data_vars)
            if len(data_vars) == 1 and getattr(self, "allow_station_single_variable_fallback", False):
                logging.warning(
                    "Variable '%s' not found in %s dataset; using sole variable '%s'",
                    selector_list[0],
                    datasource,
                    data_vars[0],
                )
                return dataset[data_vars]

            available_vars = data_vars + list(dataset.coords)
            logging.error(
                "Variable '%s' not found in %s dataset. Available variables: %s",
                selector_list[0],
                datasource,
                available_vars,
            )
            raise

    def _apply_station_custom_filter(self, dataset, datasource, attr_name, canonical_name):
        source = self.sim_source if datasource == "sim" else self.ref_source
        try:
            model = getattr(self, f"{source}_model")
        except AttributeError:
            model = source

        try:
            custom_module = importlib.import_module(f"openbench.data.custom.{model}_filter")
        except ModuleNotFoundError as exc:
            module_name = f"openbench.data.custom.{model}_filter"
            if exc.name != module_name and not module_name.startswith(f"{exc.name}."):
                raise
            logging.warning(
                "Variable '%s' missing in %s dataset for %s, no custom filter available",
                canonical_name,
                datasource,
                model,
            )
            return None
        custom_filter = getattr(custom_module, f"filter_{model}")

        attr_value = getattr(self, attr_name)
        attr_is_sequence = isinstance(attr_value, (list, tuple, np.ndarray))
        original_attr = list(attr_value) if attr_is_sequence else attr_value

        try:
            logging.warning(
                "Variable '%s' missing in %s dataset for %s; applying custom fallback",
                canonical_name,
                datasource,
                model,
            )
            updated_self, filtered_data = custom_filter(self, dataset)
            if filtered_data is None:
                return None
            fallback_ds = self._convert_filtered_data_to_dataset(filtered_data, canonical_name)
            return fallback_ds
        finally:
            if attr_is_sequence:
                setattr(self, attr_name, list(original_attr))
            else:
                setattr(self, attr_name, original_attr)

    @staticmethod
    def _convert_filtered_data_to_dataset(filtered_data, canonical_name):
        if isinstance(filtered_data, xr.Dataset):
            if canonical_name in filtered_data:
                return filtered_data[[canonical_name]]
            data_vars = list(filtered_data.data_vars)
            if data_vars:
                return filtered_data[[data_vars[0]]]
            return None

        data_array = filtered_data
        if not isinstance(data_array, xr.DataArray):
            data_array = xr.DataArray(data_array)

        if not getattr(data_array, "name", None) or data_array.name != canonical_name:
            data_array = data_array.rename(canonical_name)

        return data_array.to_dataset(name=canonical_name)

    def _normalize_time_coordinate(self, data_array):
        return normalize_station_time(data_array, getattr(self, "compare_tim_res", ""))

    def _align_station_times(self, s, o, station_id):
        return align_station_times(s, o, station_id, getattr(self, "compare_tim_res", ""))

    def make_evaluation_parallel(self, station_list, iik):
        sim_ds = None
        ref_ds = None
        try:
            sim_path = os.path.join(
                self.casedir,
                "data",
                f"stn_{self.ref_source}_{self.sim_source}",
                f"{self.item}_sim_{station_list['ID'][iik]}_{station_list['use_syear'][iik]}_{station_list['use_eyear'][iik]}.nc",
            )
            ref_path = os.path.join(
                self.casedir,
                "data",
                f"stn_{self.ref_source}_{self.sim_source}",
                f"{self.item}_ref_{station_list['ID'][iik]}_{station_list['use_syear'][iik]}_{station_list['use_eyear'][iik]}.nc",
            )

            if not os.path.exists(sim_path) or not os.path.exists(ref_path):
                missing = [path for path in (sim_path, ref_path) if not os.path.exists(path)]
                reasons = []
                for path in missing:
                    marker = Path(path).with_suffix(".skip.txt")
                    if not marker.is_file():
                        raise FileNotFoundError(
                            f"Preprocessed station data missing without a recorded data gap: {path}"
                        )
                    reasons.append(f"{Path(path).name}: {marker.read_text(encoding='utf-8')}")
                raise StationDataUnavailable("; ".join(reasons))

            # Open datasets (station files are small, no chunking needed)
            sim_ds = open_dataset_chunked(sim_path, use_chunking=False)
            ref_ds = open_dataset_chunked(ref_path, use_chunking=False)
            s_ds = self._load_station_dataset(sim_ds, "sim")
            o_ds = self._load_station_dataset(ref_ds, "ref")
            s = s_ds.to_array()
            o = o_ds.to_array()
            # Keep even a single time step indexed so data-gap checks still apply.
            s = s.squeeze([dim for dim in s.dims if dim != "time" and s.sizes[dim] == 1])
            o = o.squeeze([dim for dim in o.dims if dim != "time" and o.sizes[dim] == 1])
            o = Convert_Type.convert_nc(o)
            s = Convert_Type.convert_nc(s)

            # Align by common timestamps to avoid dimension conflicts
            station_id = station_list["ID"][iik]
            s, o = self._align_station_times(s, o, station_id)
            if not _has_any_valid_pair(s, o):
                raise StationDataUnavailable("no shared finite sim/ref pairs")
            s, o = _apply_pairwise_valid_mask(s, o)

            row = {}
            shared_mfm_names = _mfm_shared_metric_names(self.metrics)
            shared_mfm = self._MFM_shared_components(s, o) if shared_mfm_names else {}
            for name in ("KGESS", "RMSE", "correlation"):
                value = getattr(self, name)(s, o)
                row[name] = value.values if hasattr(value, "values") else value

            for metric in self.metrics:
                if hasattr(self, metric):
                    # Defensive: a custom or partially-failing metric may
                    # return a plain scalar / None instead of an xr.DataArray.
                    # Take .values when available, otherwise the result
                    # itself; fall back to NaN for None so the row stays
                    # numeric and downstream pd.concat / mean works.
                    pb = shared_mfm[metric] if metric in shared_mfm_names else getattr(self, metric)(s, o)
                    if pb is None:
                        row[f"{metric}"] = np.nan
                    elif hasattr(pb, "values"):
                        row[f"{metric}"] = pb.values
                    else:
                        row[f"{metric}"] = pb
                else:
                    raise ValueError(f"No such metric: {metric}")

            for score in self.scores:
                if hasattr(self, score):
                    pb = getattr(self, score)(s, o)
                    if pb is None:
                        row[f"{score}"] = np.nan
                    elif hasattr(pb, "values"):
                        row[f"{score}"] = pb.values
                    else:
                        row[f"{score}"] = pb
                else:
                    raise ValueError(f"No such score: {score}")

            if "ref_lat" in station_list:
                lat_lon = [station_list["ref_lat"][iik], station_list["ref_lon"][iik]]
            else:
                lat_lon = [station_list["sim_lat"][iik], station_list["sim_lon"][iik]]
            plot_stn(
                self,
                s,
                o,
                station_list["ID"][iik],
                self.ref_varname,
                _scalar_plot_value(row["RMSE"], label="RMSE", station_id=station_list["ID"][iik]),
                _scalar_plot_value(row["KGESS"], label="KGESS", station_id=station_list["ID"][iik]),
                _scalar_plot_value(row["correlation"], label="correlation", station_id=station_list["ID"][iik]),
                lat_lon,
            )
            return row
        except StationDataUnavailable as exc:
            return {"_skip_reason": str(exc)}
        finally:
            if sim_ds is not None:
                sim_ds.close()
            if ref_ds is not None:
                ref_ds.close()
            gc.collect()  # Clean up memory after processing each station

    def make_evaluation_P(self):
        try:
            if hasattr(self, "ref_fulllist") and self.ref_fulllist and os.path.exists(self.ref_fulllist):
                stnlist = self.ref_fulllist
            else:
                stnlist = os.path.join(self.casedir, f"stn_{self.ref_source}_{self.sim_source}_list.txt")
            station_list = Convert_Type.convert_Frame(pd.read_csv(stnlist, header=0))

            station_indices = list(range(len(station_list["ID"])))
            n_jobs = getattr(self, "num_cores", -1)
            if n_jobs == 1:
                results = [self.make_evaluation_parallel(station_list, iik) for iik in station_indices]
            elif _HAS_PARALLEL_ENGINE:
                logging.info("Using enhanced parallel engine for station evaluation")

                from functools import partial

                eval_func = partial(self.make_evaluation_parallel, station_list)

                try:
                    max_workers = n_jobs if isinstance(n_jobs, int) and n_jobs > 0 else None
                    results = parallel_map(
                        eval_func,
                        station_indices,
                        max_workers=max_workers,
                        backend="concurrent",
                        task_name="Evaluating stations",
                        show_progress=True,
                    )
                except (PermissionError, OSError) as exc:
                    logging.warning(
                        "Parallel station evaluation unavailable (%s). Falling back to sequential execution.", exc
                    )
                    results = [self.make_evaluation_parallel(station_list, iik) for iik in station_indices]
            else:
                # Fallback to joblib — respect user core config if available
                try:
                    results = Parallel(n_jobs=n_jobs)(
                        delayed(self.make_evaluation_parallel)(station_list, iik) for iik in station_indices
                    )
                except (PermissionError, OSError) as exc:
                    logging.warning(
                        "Joblib station evaluation unavailable (%s). Falling back to sequential execution.", exc
                    )
                    results = [self.make_evaluation_parallel(station_list, iik) for iik in station_indices]

            if len(results) != len(station_indices) or any(not isinstance(r, dict) or not r for r in results):
                raise RuntimeError("Station evaluation returned incomplete or invalid worker results")
            skipped = [
                {"station": str(station_list.iloc[i]["ID"]), "reason": result["_skip_reason"]}
                for i, result in enumerate(results)
                if "_skip_reason" in result
            ]
            for entry in skipped:
                logging.warning("Skipping station %s: %s", entry["station"], entry["reason"])
            valid_indices = [i for i, result in enumerate(results) if "_skip_reason" not in result]
            self.station_summary = {"total": len(results), "succeeded": len(valid_indices), "skipped": skipped}
            # Keep unavailable sites durable for comparison-only runs without
            # changing the successful-only metric/score table contract.
            station_status = station_list.copy()
            station_status["status"] = ["unavailable" if "_skip_reason" in r else "ok" for r in results]
            station_status["reason"] = [r.get("_skip_reason", "") for r in results]
            status_path = os.path.join(
                self.casedir,
                "data",
                f"stn_{self.ref_source}_{self.sim_source}",
                f"{self.item}_evaluation_status.csv",
            )
            _write_file_atomic(status_path, lambda path: station_status.to_csv(path, index=False), suffix=".tmp.csv")
            if not valid_indices:
                raise RuntimeError(
                    f"Station evaluation produced no valid station results ({len(skipped)}/{len(results)} skipped)"
                )
            station_list = pd.concat(
                [
                    station_list.iloc[valid_indices].reset_index(drop=True),
                    pd.DataFrame([results[i] for i in valid_indices]),
                ],
                axis=1,
            )
            requested_columns = list((getattr(self, "metrics", None) or []) + (getattr(self, "scores", None) or []))
            if not requested_columns:
                requested_columns = ["KGESS", "RMSE", "correlation"]
            missing_columns = [col for col in requested_columns if col not in station_list.columns]
            if missing_columns:
                raise RuntimeError(f"Station evaluation missing requested column(s): {missing_columns}")
            station_list[requested_columns] = station_list[requested_columns].map(
                lambda value: value.item() if isinstance(value, np.ndarray) and value.ndim == 0 else value
            )
            numeric_results = station_list[requested_columns].apply(pd.to_numeric, errors="coerce")
            empty_columns = [
                col for col in requested_columns if not np.isfinite(numeric_results[col].to_numpy(dtype=float)).any()
            ]
            if empty_columns:
                raise RuntimeError(
                    f"Station evaluation produced no finite values for requested column(s): {empty_columns}"
                )

            logging.info("Evaluation finished")
            logging.info("=======================================")

            station_list = Convert_Type.convert_Frame(station_list)

            # Save requested station outputs.  Metrics and scores share the same
            # station rows, but ``scores: []`` must not create score artifacts.
            score_vars = getattr(self, "scores", []) or []
            metric_vars = getattr(self, "metrics", None)
            write_metrics = metric_vars is None or bool(metric_vars)
            if self.output_manager:
                if score_vars:
                    scores_filename = f"{self.item}_stn_{self.ref_source}_{self.sim_source}_evaluations"
                    scores_metadata = {
                        "type": "station_evaluations_scores",
                        "item": self.item,
                        "ref_source": self.ref_source,
                        "sim_source": self.sim_source,
                    }
                    self.output_manager.save_data(station_list, "scores", scores_filename, "csv", scores_metadata)

                if write_metrics:
                    metrics_filename = f"{self.item}_stn_{self.ref_source}_{self.sim_source}_evaluations"
                    metrics_metadata = {
                        "type": "station_evaluations_metrics",
                        "item": self.item,
                        "ref_source": self.ref_source,
                        "sim_source": self.sim_source,
                    }
                    self.output_manager.save_data(station_list, "metrics", metrics_filename, "csv", metrics_metadata)
            else:
                if score_vars:
                    scores_path = os.path.join(
                        self.casedir, "scores", f"{self.item}_stn_{self.ref_source}_{self.sim_source}_evaluations.csv"
                    )
                    logging.info(f"Saving scores to {scores_path}")
                    os.makedirs(os.path.dirname(scores_path), exist_ok=True)
                    _write_file_atomic(
                        scores_path,
                        lambda tmp_path: station_list.to_csv(tmp_path, index=False),
                        suffix=".tmp.csv",
                    )

                if write_metrics:
                    metrics_path = os.path.join(
                        self.casedir, "metrics", f"{self.item}_stn_{self.ref_source}_{self.sim_source}_evaluations.csv"
                    )
                    logging.info(f"Saving metrics to {metrics_path}")
                    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
                    _write_file_atomic(
                        metrics_path,
                        lambda tmp_path: station_list.to_csv(tmp_path, index=False),
                        suffix=".tmp.csv",
                    )

            make_plot_index_stn(self)
            if skipped:
                logging.warning(
                    "Station evaluation partial success: %d/%d succeeded, %d skipped",
                    len(valid_indices),
                    len(results),
                    len(skipped),
                )

        finally:
            gc.collect()  # Final cleanup
