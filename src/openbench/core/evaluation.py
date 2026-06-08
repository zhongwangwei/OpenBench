import gc
import importlib
import logging
import os

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

# Import chunked dataset loader for memory efficiency
try:
    from openbench.util.dataset_loader import open_dataset as open_dataset_chunked
except ImportError:
    # Fallback to standard xr.open_dataset; strip use_chunking if passed
    # (only the chunked loader understands it — bare xr.open_dataset
    # would raise TypeError: unexpected keyword argument).
    def open_dataset_chunked(path, *args, **kwargs):
        kwargs.pop("use_chunking", None)
        return xr.open_dataset(path, *args, **kwargs)


# Import parallel engine
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

# Check the platform
from openbench.util.converttype import Convert_Type
from openbench.util.netcdf import write_file_atomic as _write_file_atomic
from openbench.util.netcdf import write_netcdf_atomic as _write_netcdf_atomic
from openbench.core._visualization_bridge import visualization_callable
from openbench.core.metrics import metrics
from openbench.core.scores import scores


make_plot_index_grid = visualization_callable("make_plot_index_grid")
make_plot_index_stn = visualization_callable("make_plot_index_stn")
plot_stn = visualization_callable("plot_stn")

# Import climatology processor
try:
    from openbench.data.climatology import ClimatologyProcessor, process_climatology_evaluation

    _HAS_CLIMATOLOGY = True
except ImportError:
    _HAS_CLIMATOLOGY = False
    ClimatologyProcessor = None

    def process_climatology_evaluation(*args, **kwargs):
        return args[0], args[1], args[2]


# Import output manager
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


def _metric_worker_count(num_cores, metric_count: int) -> int:
    """Return metric-level workers bounded by configured cores and metric count."""
    if metric_count <= 1:
        return 1
    try:
        requested = int(num_cores) if num_cores is not None else 0
    except (TypeError, ValueError):
        requested = 1
    if requested <= 0:
        requested = max(1, os.cpu_count() or 1)
    return min(max(1, requested), metric_count)


def _apply_pairwise_valid_mask(s: xr.DataArray, o: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Mask sim/ref arrays to their shared non-missing support without eager loads.

    The old implementation used eager NumPy value copies and in-place
    assignment, which materialized chunked high-resolution grids before metric
    calculation. ``notnull``/``where`` keeps xarray/dask lazy until the metric
    result is actually written.
    """
    valid = s.notnull() & o.notnull()
    return s.where(valid), o.where(valid)


def _has_any_valid_pair(s: xr.DataArray, o: xr.DataArray) -> bool:
    """Return whether sim/ref arrays share at least one finite pair."""
    valid = s.notnull() & o.notnull()
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


class Evaluation_grid(metrics, scores):
    def _calculate_metric(self, s, o, metric):
        """Helper method for parallel metric calculation."""
        try:
            if hasattr(self, metric):
                self.process_metric(metric, s, o)
                return metric
            else:
                logging.error(f"No such metric: {metric}")
                return None
        except Exception as e:
            logging.error(f"Error calculating metric {metric}: {e}")
            return None

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

        # Initialize output manager if available
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

    def process_metric(self, metric, s, o, vkey=""):
        try:
            pb = getattr(self, metric)(s, o)
            pb = pb.squeeze()
            if isinstance(pb, xr.DataArray):
                pb_da = pb.rename(metric)
            else:
                pb_da = xr.DataArray(pb, coords=[o.lat, o.lon], dims=["lat", "lon"], name=metric)

            # Use output manager if available, otherwise fallback to original method
            if self.output_manager:
                filename = f"{self.item}_ref_{self.ref_source}_sim_{self.sim_source}_{metric}{vkey}"
                metadata = {
                    "metric": metric,
                    "item": self.item,
                    "ref_source": self.ref_source,
                    "sim_source": self.sim_source,
                    "variable_key": vkey,
                }
                output_path = self.output_manager.save_data(pb_da, "metrics", filename, "netcdf", metadata)
            else:
                # Original method
                output_path = os.path.join(
                    self.casedir,
                    "metrics",
                    f"{self.item}_ref_{self.ref_source}_sim_{self.sim_source}_{metric}{vkey}.nc",
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                _write_netcdf_atomic(pb_da, output_path)
                logging.info(f"Saved metric {metric} to {output_path}")
        finally:
            gc.collect()  # Clean up memory after processing each metric

    def process_score(self, score, s, o, vkey=""):
        try:
            pb = getattr(self, score)(s, o)
            pb = pb.squeeze()
            pb_da = xr.DataArray(pb, coords=[o.lat, o.lon], dims=["lat", "lon"], name=score)

            # Use output manager if available, otherwise fallback to original method
            if self.output_manager:
                filename = f"{self.item}_ref_{self.ref_source}_sim_{self.sim_source}_{score}{vkey}"
                metadata = {
                    "score": score,
                    "item": self.item,
                    "ref_source": self.ref_source,
                    "sim_source": self.sim_source,
                    "variable_key": vkey,
                }
                output_path = self.output_manager.save_data(pb_da, "scores", filename, "netcdf", metadata)
            else:
                # Original method
                output_path = os.path.join(
                    self.casedir, "scores", f"{self.item}_ref_{self.ref_source}_sim_{self.sim_source}_{score}{vkey}.nc"
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                _write_netcdf_atomic(pb_da, output_path)
                logging.info(f"Saved score {score} to {output_path}")
        finally:
            gc.collect()  # Clean up memory after processing each score

    def make_Evaluation(self, **kwargs):
        ref_ds = None
        sim_ds = None
        try:
            ref_path = getattr(self, "ref_file_override", None) or os.path.join(
                self.casedir, "data", f"{self.item}_ref_{self.ref_source}_{self.ref_varname}.nc"
            )
            sim_path = os.path.join(self.casedir, "data", f"{self.item}_sim_{self.sim_source}_{self.sim_varname}.nc")

            # Open datasets and keep references for proper cleanup
            ref_ds = open_dataset_chunked(ref_path)
            sim_ds = open_dataset_chunked(sim_path)
            o = ref_ds[f"{self.ref_varname}"]
            s = sim_ds[f"{self.sim_varname}"]
            o = Convert_Type.convert_nc(o)
            s = Convert_Type.convert_nc(s)

            # Process climatology if applicable
            if _HAS_CLIMATOLOGY:
                original_metrics = self.metrics.copy() if hasattr(self.metrics, "copy") else list(self.metrics)
                original_scores = self.scores.copy() if hasattr(self.scores, "copy") else list(self.scores)

                # Combine metrics and scores for filtering
                all_evaluations = list(self.metrics) + list(self.scores)

                # Get compare_tim_res and syear from instance attributes
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
                    # Climatology evaluation mode
                    logging.info("=" * 80)
                    logging.info("CLIMATOLOGY EVALUATION MODE DETECTED")
                    logging.info("=" * 80)

                    o = o_clim[f"{self.ref_varname}"]
                    s = s_clim[f"{self.sim_varname}"]
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
                    # Regular time series evaluation
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

            s, o = _apply_pairwise_valid_mask(s, o)
            logging.info("=" * 80)

            # Parallel processing of metrics if configured and beneficial.
            # Honor project.num_cores instead of the old hard-coded max=4.
            metric_workers = _metric_worker_count(getattr(self, "num_cores", 1), len(self.metrics))
            if _HAS_PARALLEL_ENGINE and metric_workers > 1:
                logging.info("Processing %d metrics in parallel with %d worker(s)", len(self.metrics), metric_workers)
                from functools import partial

                metric_func = partial(self._calculate_metric, s, o)
                metric_results = parallel_map(
                    metric_func,
                    self.metrics,
                    task_name="Calculating metrics",
                    show_progress=False,
                    max_workers=metric_workers,
                )
                # Process results
                for metric, result in zip(self.metrics, metric_results):
                    if result is not None:
                        logging.info(f"Calculated metric: {metric}")
            else:
                # Sequential processing — log + skip unknown metrics to
                # match the parallel path (which never sys.exits). A typo
                # in one metric name should not abort an entire run.
                for metric in self.metrics:
                    if hasattr(self, metric):
                        logging.info(f"Calculating metric: {metric}")
                        self.process_metric(metric, s, o)
                    else:
                        logging.error(f"No such metric: {metric}; skipping")

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
            # Close datasets to free memory and file handles
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

        # Initialize output manager if available
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
        # Get model name from _model attribute (e.g., TE-routing_model = "TE")
        source = self.sim_source if datasource == "sim" else self.ref_source
        try:
            model = getattr(self, f"{source}_model")
        except AttributeError:
            model = source

        try:
            custom_module = importlib.import_module(f"openbench.data.custom.{model}_filter")
            custom_filter = getattr(custom_module, f"filter_{model}")
        except (ImportError, AttributeError):
            logging.warning(
                "Variable '%s' missing in %s dataset for %s, no custom filter available",
                canonical_name,
                datasource,
                model,
            )
            return None

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
        """
        Normalize time coordinates to the configured comparison resolution.

        Ensures reference/simulation station series use identical timestamps even when
        source files encode different daily/hourly conventions (e.g., 00 UTC vs 12 UTC).
        """
        if not hasattr(data_array, "coords") or "time" not in data_array.coords:
            return data_array

        compare_res = str(getattr(self, "compare_tim_res", "") or "").strip().lower()
        if not compare_res:
            return data_array

        try:
            times = pd.to_datetime(data_array["time"].values)
        except Exception as err:
            logging.debug(f"Station time normalization skipped: {err}")
            return data_array

        if times.size == 0:
            return data_array

        normalized = None
        if compare_res in {"day", "d", "1d", "daily"}:
            normalized = (times.floor("D") + pd.Timedelta(hours=12)).values
        elif compare_res in {"hour", "h", "1h", "hourly"}:
            normalized = (times.floor("H") + pd.Timedelta(minutes=30)).values
        elif compare_res in {"month", "mon", "m", "1m", "monthly"}:
            normalized = (times.to_period("M").to_timestamp(how="start") + pd.Timedelta(days=14, hours=12)).values
        elif compare_res in {"year", "yr", "y", "1y", "annual", "yearly"}:
            normalized = (times.to_period("Y").to_timestamp(how="start") + pd.Timedelta(days=182, hours=12)).values
        else:
            return data_array

        try:
            data_array = data_array.assign_coords(time=("time", normalized))
        except Exception as err:
            logging.debug(f"Failed to assign normalized station times: {err}")
        return data_array

    def _align_station_times(self, s: xr.DataArray, o: xr.DataArray, station_id) -> tuple[xr.DataArray, xr.DataArray]:
        """Align station series exactly first; normalize only as a fallback."""
        s_times = pd.to_datetime(s["time"].values)
        o_times = pd.to_datetime(o["time"].values)
        common_times = np.intersect1d(
            s_times.values if hasattr(s_times, "values") else s_times,
            o_times.values if hasattr(o_times, "values") else o_times,
        )
        if common_times.size:
            return s.sel(time=common_times).sortby("time"), o.sel(time=common_times).sortby("time")

        s_norm = self._normalize_time_coordinate(s)
        o_norm = self._normalize_time_coordinate(o)
        s_times = pd.to_datetime(s_norm["time"].values)
        o_times = pd.to_datetime(o_norm["time"].values)
        common_times = np.intersect1d(
            s_times.values if hasattr(s_times, "values") else s_times,
            o_times.values if hasattr(o_times, "values") else o_times,
        )
        if common_times.size:
            logging.warning(
                "Station %s time coordinates required normalization before alignment; using %d overlapping steps",
                station_id,
                common_times.size,
            )
            return s_norm.sel(time=common_times).sortby("time"), o_norm.sel(time=common_times).sortby("time")
        raise ValueError(f"Station {station_id} has no overlapping time steps after exact or normalized alignment")

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

            # Check if both files exist before processing
            if not os.path.exists(sim_path) or not os.path.exists(ref_path):
                station_id = station_list["ID"][iik]
                logging.warning(f"Skipping station {station_id} - data files not found (time range mismatch)")
                return None

            # Open datasets (station files are small, no chunking needed)
            sim_ds = open_dataset_chunked(sim_path, use_chunking=False)
            ref_ds = open_dataset_chunked(ref_path, use_chunking=False)
            s_ds = self._load_station_dataset(sim_ds, "sim")
            o_ds = self._load_station_dataset(ref_ds, "ref")
            s = s_ds.to_array().squeeze()
            o = o_ds.to_array().squeeze()
            o = Convert_Type.convert_nc(o)
            s = Convert_Type.convert_nc(s)

            # Align by common timestamps to avoid dimension conflicts
            try:
                station_id = station_list["ID"][iik]
                s, o = self._align_station_times(s, o, station_id)
            except Exception as e:
                logging.debug(f"Time alignment fallback due to: {e}")
                station_id = station_list["ID"][iik]
                logging.warning(f"Skipping station {station_id} - failed to align time coordinates")
                return None
            if not _has_any_valid_pair(s, o):
                station_id = station_list["ID"][iik]
                logging.warning(f"Skipping station {station_id} - no shared finite sim/ref pairs")
                return None
            s, o = _apply_pairwise_valid_mask(s, o)

            row = {}
            # for based plot
            try:
                row["KGESS"] = self.KGESS(s, o).values
            except (ValueError, RuntimeError, AttributeError) as e:
                logging.warning("Station %s KGESS calculation failed: %s", station_id, e)
                row["KGESS"] = np.nan
            try:
                row["RMSE"] = self.RMSE(s, o).values
            except (ValueError, RuntimeError, AttributeError) as e:
                logging.warning("Station %s RMSE calculation failed: %s", station_id, e)
                row["RMSE"] = np.nan
            try:
                row["correlation"] = self.correlation(s, o).values
            except (ValueError, RuntimeError, AttributeError) as e:
                logging.warning("Station %s correlation calculation failed: %s", station_id, e)
                row["correlation"] = np.nan

            for metric in self.metrics:
                if hasattr(self, metric):
                    # Defensive: a custom or partially-failing metric may
                    # return a plain scalar / None instead of an xr.DataArray.
                    # Take .values when available, otherwise the result
                    # itself; fall back to NaN for None so the row stays
                    # numeric and downstream pd.concat / mean works.
                    pb = getattr(self, metric)(s, o)
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
        finally:
            # Close datasets to free memory and file handles
            if sim_ds is not None:
                sim_ds.close()
            if ref_ds is not None:
                ref_ds.close()
            gc.collect()  # Clean up memory after processing each station

    def make_evaluation_P(self):
        try:
            # Use ref_fulllist if available, otherwise use dataset-specific filename
            if hasattr(self, "ref_fulllist") and self.ref_fulllist and os.path.exists(self.ref_fulllist):
                stnlist = self.ref_fulllist
            else:
                stnlist = os.path.join(self.casedir, f"stn_{self.ref_source}_{self.sim_source}_list.txt")
            station_list = Convert_Type.convert_Frame(pd.read_csv(stnlist, header=0))

            # Use enhanced parallel engine if available
            station_indices = list(range(len(station_list["ID"])))
            n_jobs = getattr(self, "num_cores", -1)
            if n_jobs == 1:
                results = [self.make_evaluation_parallel(station_list, iik) for iik in station_indices]
            elif _HAS_PARALLEL_ENGINE:
                logging.info("Using enhanced parallel engine for station evaluation")

                # Create partial function with station_list
                from functools import partial

                eval_func = partial(self.make_evaluation_parallel, station_list)

                # Process stations in parallel
                try:
                    max_workers = n_jobs if isinstance(n_jobs, int) and n_jobs > 0 else None
                    results = parallel_map(
                        eval_func,
                        station_indices,
                        max_workers=max_workers,
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

            # Filter out None results from stations that failed or were skipped
            valid_results = [r if r is not None else {} for r in results]
            station_list = pd.concat([station_list, pd.DataFrame(valid_results)], axis=1)

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

        finally:
            gc.collect()  # Final cleanup
