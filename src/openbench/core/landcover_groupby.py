import logging
import os
import re

import numpy as np
import xarray as xr

from openbench.util.converttype import Convert_Type
from openbench.util.filenames import (
    groupby_class_netcdf_stem,
    groupby_pair_dirname,
    groupby_table_filename,
    join_filename_components,
)
from openbench.util.netcdf import write_file_atomic as _write_file_atomic
from openbench.util.netcdf import write_netcdf_atomic as _write_netcdf_atomic
from openbench.util.static_datasets import static_dataset_path

# Check the platform
from openbench.core._visualization_bridge import visualization_callable
from openbench.core.metrics import metrics
from openbench.core.scores import scores


make_LC_based_heat_map = visualization_callable("make_LC_based_heat_map")


def _open_dataset_safe(path: str, **kwargs) -> xr.Dataset:
    """Open dataset with fallback to decode_times=False if initial open fails."""
    try:
        return xr.open_dataset(path, **kwargs)
    except Exception as e:
        if kwargs.get("decode_times", True) is not False:
            logging.warning(f"Failed to open {path}: {e}. Retrying with decode_times=False")
            retry_kwargs = {k: v for k, v in kwargs.items() if k != "decode_times"}
            return xr.open_dataset(path, decode_times=False, **retry_kwargs)
        raise


def _write_lines_atomic(output_path: str, lines: list[str]) -> None:
    """Write a text table via same-directory temp file to avoid partial CSV/TXT outputs."""
    _write_file_atomic(output_path, lambda tmp_path: tmp_path.write_text("".join(lines)), suffix=".tmp.csv")


def _groupby_pair_dir(root: str, groupby_name: str, sim_source: str, ref_source: str) -> str:
    """Return safe LC/CZ groupby pair output directory."""
    return os.path.join(root, "comparisons", groupby_name, groupby_pair_dirname(sim_source, ref_source))


def _groupby_option_path(root: str, groupby_name: str, sim_source: str, ref_source: str) -> str:
    """Return safe groupby path value passed to downstream renderers."""
    return _groupby_pair_dir(root, groupby_name, sim_source, ref_source) + os.sep


def _evaluation_netcdf_path(
    casedir: str,
    category: str,
    evaluation_item: str,
    ref_source: str,
    sim_source: str,
    variable: str,
) -> str:
    """Return an evaluation NetCDF path, preferring legacy names but supporting safe names."""
    legacy_path = os.path.join(
        casedir,
        category,
        f"{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{variable}.nc",
    )
    if os.path.exists(legacy_path):
        return legacy_path
    safe_path = os.path.join(
        casedir,
        category,
        f"{join_filename_components(evaluation_item, 'ref', ref_source, 'sim', sim_source, variable)}.nc",
    )
    return safe_path


def _clip_metric_quantiles(ds: xr.Dataset, metric: str) -> xr.Dataset:
    """Clip metric outliers within the dataset currently being summarized."""
    ds = ds.where(np.isfinite(ds), np.nan)
    if metric not in ds:
        return ds
    dims = [dim for dim in ("lat", "lon") if dim in ds[metric].dims]
    if not dims:
        return ds
    q_value = ds[metric].quantile([0.05, 0.95], dim=dims, skipna=True)
    lower = q_value.sel(quantile=0.05)
    upper = q_value.sel(quantile=0.95)
    return ds.where((ds[metric] >= lower) & (ds[metric] <= upper), np.nan)


def _class_bundle_path(
    dir_path: str,
    evaluation_item: str,
    ref_source: str,
    sim_source: str,
    statistic: str,
    class_prefix: str,
) -> str:
    """Return the single class-dimension NetCDF path for one groupby statistic."""
    return os.path.join(
        dir_path,
        f"{groupby_class_netcdf_stem(evaluation_item, ref_source, sim_source, statistic, class_prefix)}__classes.nc",
    )


def _write_class_bundle_atomic(class_datasets: list[xr.Dataset], class_names: list[str], output_path: str) -> None:
    """Write all per-class datasets as one NetCDF with a ``class`` dimension."""
    if not class_datasets:
        return
    bundled = xr.concat(class_datasets, dim=xr.IndexVariable("class", class_names))
    bundled["class_id"] = ("class", np.arange(len(class_names), dtype=np.int32))
    _write_netcdf_atomic(bundled, output_path)


class LC_groupby(metrics, scores):
    def __init__(self, main_nml, scores, metrics):
        self.name = "StatisticsDataHandler"
        self.version = "0.3"
        self.release = "0.3"
        self.date = "June 2024"
        self.author = "Zhongwang Wei"
        self.main_nml = main_nml
        self.general_config = self.main_nml["general"]
        self._igbp_station_warning_shown = False  # Track if IGBP station data warning has been shown
        self._pft_station_warning_shown = False  # Track if PFT station data warning has been shown
        # update self based on self.general_config
        self.__dict__.update(self.general_config)
        # Extract remapping information from main namelist
        self.compare_grid_res = self.main_nml["general"]["compare_grid_res"]
        self.compare_tim_res = self.main_nml["general"].get("compare_tim_res", "Month").lower()
        self.casedir = os.path.join(self.main_nml["general"]["basedir"], self.main_nml["general"]["basename"])
        # Set default weight method to 'none'
        # Handle null/None values from config by defaulting to 'none'
        self.weight = self.main_nml["general"].get("weight", "none") or "none"
        # this should be done in read_namelist
        # adjust the time frequency
        # Check if climatology mode - skip frequency parsing
        if self.compare_tim_res in ["climatology-year", "climatology-month"]:
            logging.debug(
                f"LC_groupby: Climatology mode detected ({self.compare_tim_res}), skipping frequency conversion"
            )
        else:
            match = re.match(r"(\d*)\s*([a-zA-Z]+)", self.compare_tim_res)
            if not match:
                logging.error("Invalid time resolution format. Use '3month', '6hr', etc.")
                raise ValueError("Invalid time resolution format. Use '3month', '6hr', etc.")
            value, unit = match.groups()
            if not value:
                value = 1
            else:
                value = int(value)  # Convert the numerical value to an integer
        self.metrics = metrics
        self.scores = scores

    def scenarios_IGBP_groupby_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        def _IGBP_class_remap(self):
            from openbench.data.regrid import Grid, create_regridding_dataset

            with (
                static_dataset_path("IGBP.nc") as dataset_path,
                _open_dataset_safe(dataset_path, chunks={"lat": 2000, "lon": 2000}) as ds_file,
            ):
                ds = ds_file["IGBP"].load()  # Only take the class variable.
            ds = ds.sortby(["lat", "lon"])
            # ds = ds.rename({"lat": "latitude", "lon": "longitude"})
            new_grid = Grid(
                north=self.max_lat - self.compare_grid_res / 2,
                south=self.min_lat + self.compare_grid_res / 2,
                west=self.min_lon + self.compare_grid_res / 2,
                east=self.max_lon - self.compare_grid_res / 2,
                resolution_lat=self.compare_grid_res,
                resolution_lon=self.compare_grid_res,
            )
            target_dataset = create_regridding_dataset(new_grid)
            ds_regrid = ds.astype(int).regrid.most_common(target_dataset, values=np.arange(1, 18))
            IGBPtype_remap = f"{self.casedir}/comparisons/IGBP_groupby/IGBP_remap.nc"
            _write_netcdf_atomic(ds_regrid, IGBPtype_remap)
            self.IGBP_dir = IGBPtype_remap

        def _scenarios_IGBP_groupby(basedir, scores, metrics, sim_nml, ref_nml, evaluation_items):
            """
            Compare the IGBP class of the model output data and the reference data
            """
            with _open_dataset_safe(self.IGBP_dir) as igbp_ds:
                IGBPtype = igbp_ds["IGBP"].load()
            # convert IGBP type to int
            IGBPtype = IGBPtype.astype(int)

            igbp_class_names = {
                1: "evergreen_needleleaf_forest",
                2: "evergreen_broadleaf_forest",
                3: "deciduous_needleleaf_forest",
                4: "deciduous_broadleaf_forest",
                5: "mixed_forests",
                6: "closed_shrubland",
                7: "open_shrublands",
                8: "woody_savannas",
                9: "savannas",
                10: "grasslands",
                11: "permanent_wetlands",
                12: "croplands",
                13: "urban_and_built_up",
                14: "cropland_natural_vegetation_mosaic",
                15: "snow_and_ice",
                16: "barren_or_sparsely_vegetated",
                17: "water_bodies",
            }

            # read the simulation source and reference source
            for evaluation_item in evaluation_items:
                logging.info(f"Processing evaluation item: {evaluation_item}")
                sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
                ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]
                # if the sim_sources and ref_sources are not list, then convert them to list
                if isinstance(sim_sources, str):
                    sim_sources = [sim_sources]
                if isinstance(ref_sources, str):
                    ref_sources = [ref_sources]
                for ref_source in ref_sources:
                    for i, sim_source in enumerate(sim_sources):
                        ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                        sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]
                        ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]
                        if ref_data_type == "stn" or sim_data_type == "stn":
                            logging.warning(
                                "Skipping IGBP class comparison for %s ref=%s sim=%s: station data is not supported",
                                evaluation_item,
                                ref_source,
                                sim_source,
                            )
                            continue  # Skip processing for station data
                        else:
                            dir_path = _groupby_pair_dir(basedir, "IGBP_groupby", sim_source, ref_source)
                            if not os.path.exists(dir_path):
                                os.makedirs(dir_path)
                            if len(self.metrics) > 0:
                                output_file_path = os.path.join(
                                    dir_path, groupby_table_filename(evaluation_item, sim_source, ref_source, "metrics")
                                )
                                rows = []
                                # Print the table header with class names
                                header_values = ["metric"]
                                for igbp_class_name in igbp_class_names.values():
                                    header_values.append(igbp_class_name)
                                header_values.append("Overall")
                                rows.append("\t".join(header_values) + "\n")

                                # Calculate and print mean values
                                for metric in self.metrics:
                                    metric_file = _evaluation_netcdf_path(
                                        self.casedir, "metrics", evaluation_item, ref_source, sim_source, metric
                                    )
                                    # Skip if metric file doesn't exist (e.g., skipped in climatology mode)
                                    if not os.path.exists(metric_file):
                                        logging.debug(
                                            f"Skipping metric {metric} - file not found (possibly skipped in climatology mode)"
                                        )
                                        continue
                                    with _open_dataset_safe(metric_file) as ds_file:
                                        ds = Convert_Type.convert_nc(ds_file.load())

                                    # Clip within the scope being summarized.
                                    # Per-class summaries must not inherit a
                                    # global clip that silently drops valid
                                    # class-local tails.
                                    ds = ds.where(np.isfinite(ds), np.nan)
                                    overall_ds = _clip_metric_quantiles(ds, metric)

                                    overall_median = overall_ds[metric].median(skipna=True).values
                                    overall_median_str = (
                                        f"{overall_median:.3f}" if not np.isnan(overall_median) else "N/A"
                                    )

                                    row_values = [metric]
                                    class_datasets = []
                                    class_names = []
                                    for i in range(1, 18):
                                        ds1 = _clip_metric_quantiles(ds.where(IGBPtype == i), metric)
                                        igbp_class_name = igbp_class_names.get(i, f"IGBP_{i}")
                                        class_datasets.append(ds1)
                                        class_names.append(igbp_class_name)
                                        median_value = ds1[metric].median(skipna=True).values
                                        median_value_str = (
                                            f"{median_value:.3f}" if not np.isnan(median_value) else "N/A"
                                        )
                                        row_values.append(median_value_str)
                                    _write_class_bundle_atomic(
                                        class_datasets,
                                        class_names,
                                        _class_bundle_path(
                                            dir_path, evaluation_item, ref_source, sim_source, metric, "IGBP"
                                        ),
                                    )
                                    row_values.append(overall_median_str)
                                    rows.append("\t".join(row_values) + "\n")
                                _write_lines_atomic(output_file_path, rows)

                                selected_metrics = self.metrics
                                # selected_metrics = list(selected_metrics)
                                option["path"] = _groupby_option_path(
                                    self.casedir, "IGBP_groupby", sim_source, ref_source
                                )
                                option["item"] = [evaluation_item, sim_source, ref_source]
                                option["groupby"] = "IGBP_groupby"
                                make_LC_based_heat_map(output_file_path, selected_metrics, "metric", option)
                            else:
                                logging.debug("No metrics requested for IGBP class comparison")

                            if len(self.scores) > 0:
                                dir_path = _groupby_pair_dir(basedir, "IGBP_groupby", sim_source, ref_source)
                                if not os.path.exists(dir_path):
                                    os.makedirs(dir_path)
                                output_file_path2 = os.path.join(
                                    dir_path, groupby_table_filename(evaluation_item, sim_source, ref_source, "scores")
                                )

                                rows = []
                                # Print the table header with class names
                                header_values = ["score"]
                                for igbp_class_name in igbp_class_names.values():
                                    header_values.append(igbp_class_name)
                                header_values.append("Overall")
                                rows.append("\t".join(header_values) + "\n")

                                # Cache the mass-weight reference once per
                                # (sim, ref) pair instead of re-opening the
                                # same .nc file 18 times per IGBP class.
                                cached_mass_ref = None

                                # Calculate and print mean values
                                for score in self.scores:
                                    score_file = _evaluation_netcdf_path(
                                        self.casedir, "scores", evaluation_item, ref_source, sim_source, score
                                    )
                                    # Skip if score file doesn't exist (e.g., skipped in climatology mode)
                                    if not os.path.exists(score_file):
                                        logging.debug(
                                            f"Skipping score {score} - file not found (possibly skipped in climatology mode)"
                                        )
                                        continue
                                    with _open_dataset_safe(score_file) as ds_file:
                                        ds = Convert_Type.convert_nc(ds_file.load())

                                    if self.weight.lower() == "area":
                                        weights = np.cos(np.deg2rad(ds.lat))
                                        overall_mean = ds[score].weighted(weights).mean(skipna=True).values
                                    elif self.weight.lower() == "mass":
                                        # Reuse cached ref dataset across the
                                        # 17 IGBP-class iterations below; the
                                        # ref file does not change per score.
                                        if cached_mass_ref is None:
                                            with _open_dataset_safe(
                                                f"{self.casedir}/data/{evaluation_item}_ref_{ref_source}_{ref_varname}.nc"
                                            ) as ref_ds:
                                                cached_mass_ref = ref_ds[f"{ref_varname}"].load()
                                        o = cached_mass_ref

                                        # Calculate area weights (cosine of latitude)
                                        area_weights = np.cos(np.deg2rad(ds.lat))

                                        # Calculate absolute flux weights
                                        flux_weights = np.abs(o.mean("time"))

                                        # Combine area and flux weights
                                        combined_weights = area_weights * flux_weights

                                        # Normalize weights to sum to 1
                                        normalized_weights = combined_weights / combined_weights.sum()

                                        # Calculate weighted mean
                                        overall_mean = (
                                            ds[score].weighted(normalized_weights.fillna(0)).mean(skipna=True).values
                                        )
                                    else:
                                        overall_mean = ds[score].mean(skipna=True).values

                                    overall_mean_str = f"{overall_mean:.3f}" if not np.isnan(overall_mean) else "N/A"

                                    row_values = [score]
                                    class_datasets = []
                                    class_names = []
                                    for i in range(1, 18):
                                        ds1 = ds.where(IGBPtype == i)
                                        igbp_class_name = igbp_class_names.get(i, f"IGBP_{i}")
                                        class_datasets.append(ds1)
                                        class_names.append(igbp_class_name)

                                        if self.weight.lower() == "area":
                                            weights = np.cos(np.deg2rad(ds.lat))
                                            mean_value = ds1[score].weighted(weights).mean(skipna=True).values
                                        elif self.weight.lower() == "mass":
                                            # cached_mass_ref reused from outer score loop
                                            o = cached_mass_ref

                                            # Calculate area weights (cosine of latitude)
                                            area_weights = np.cos(np.deg2rad(ds.lat))

                                            # Calculate absolute flux weights
                                            flux_weights = np.abs(o.mean("time"))

                                            # Combine area and flux weights
                                            combined_weights = area_weights * flux_weights

                                            # Normalize weights to sum to 1
                                            normalized_weights = combined_weights / combined_weights.sum()

                                            # Calculate weighted mean
                                            mean_value = (
                                                ds1[score]
                                                .weighted(normalized_weights.fillna(0))
                                                .mean(skipna=True)
                                                .values
                                            )
                                        else:
                                            mean_value = ds1[score].mean(skipna=True).values

                                        mean_value_str = f"{mean_value:.3f}" if not np.isnan(mean_value) else "N/A"
                                        row_values.append(mean_value_str)
                                    _write_class_bundle_atomic(
                                        class_datasets,
                                        class_names,
                                        _class_bundle_path(
                                            dir_path, evaluation_item, ref_source, sim_source, score, "IGBP"
                                        ),
                                    )
                                    row_values.append(overall_mean_str)
                                    rows.append("\t".join(row_values) + "\n")
                                _write_lines_atomic(output_file_path2, rows)

                                selected_scores = self.scores
                                option["path"] = _groupby_option_path(
                                    self.casedir, "IGBP_groupby", sim_source, ref_source
                                )
                                option["item"] = [evaluation_item, sim_source, ref_source]
                                option["groupby"] = "IGBP_groupby"
                                make_LC_based_heat_map(output_file_path2, selected_scores, "score", option)
                                # print(f"IGBP class scores comparison results are saved to {output_file_path2}")
                            else:
                                logging.debug("No scores requested for IGBP class comparison")

        metricsdir_path = os.path.join(f"{casedir}", "comparisons", "IGBP_groupby")
        # if os.path.exists(metricsdir_path):
        #    shutil.rmtree(metricsdir_path)
        # print(f"Re-creating output directory: {metricsdir_path}")
        if not os.path.exists(metricsdir_path):
            os.makedirs(metricsdir_path)

        scoresdir_path = os.path.join(f"{casedir}", "comparisons", "IGBP_groupby")
        # if os.path.exists(scoresdir_path):
        #    shutil.rmtree(scoresdir_path)
        # print(f"Re-creating output directory: {scoresdir_path}")
        if not os.path.exists(scoresdir_path):
            os.makedirs(scoresdir_path)

        _IGBP_class_remap(self)
        _scenarios_IGBP_groupby(casedir, scores, metrics, sim_nml, ref_nml, evaluation_items)

    def scenarios_PFT_groupby_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        def _PFT_class_remap(self):
            """
            Compare the PFT class of the model output data and the reference data using xarray
            """
            from openbench.data.regrid import Grid, create_regridding_dataset

            with (
                static_dataset_path("PFT.nc") as dataset_path,
                _open_dataset_safe(dataset_path, chunks={"lat": 2000, "lon": 2000}) as ds_file,
            ):
                ds = ds_file["PFT"].load()
            ds = ds.sortby(["lat", "lon"])
            # ds = ds.rename({"lat": "latitude", "lon": "longitude"})
            new_grid = Grid(
                north=self.max_lat - self.compare_grid_res / 2,
                south=self.min_lat + self.compare_grid_res / 2,
                west=self.min_lon + self.compare_grid_res / 2,
                east=self.max_lon - self.compare_grid_res / 2,
                resolution_lat=self.compare_grid_res,
                resolution_lon=self.compare_grid_res,
            )
            target_dataset = create_regridding_dataset(new_grid)
            ds_regrid = ds.astype(int).regrid.most_common(target_dataset, values=np.arange(0, 16))
            PFTtype_remap = f"{self.casedir}/comparisons/PFT_groupby/PFT_remap.nc"
            _write_netcdf_atomic(ds_regrid, PFTtype_remap)
            self.PFT_dir = PFTtype_remap

        def _scenarios_PFT_groupby(basedir, scores, metrics, sim_nml, ref_nml, evaluation_items):
            """
            Compare the PFT class of the model output data and the reference data
            """
            with _open_dataset_safe(self.PFT_dir) as pft_ds:
                PFTtype = pft_ds["PFT"].load()
            # convert PFT type to int
            PFTtype = PFTtype.astype(int)
            PFT_class_names = {
                0: "bare_soil",
                1: "needleleaf_evergreen_temperate_tree",
                2: "needleleaf_evergreen_boreal_tree",
                3: "needleleaf_deciduous_boreal_tree",
                4: "broadleaf_evergreen_tropical_tree",
                5: "broadleaf_evergreen_temperate_tree",
                6: "broadleaf_deciduous_tropical_tree",
                7: "broadleaf_deciduous_temperate_tree",
                8: "broadleaf_deciduous_boreal_tree",
                9: "broadleaf_evergreen_shrub",
                10: "broadleaf_deciduous_temperate_shrub",
                11: "broadleaf_deciduous_boreal_shrub",
                12: "c3_arctic_grass",
                13: "c3_non-arctic_grass",
                14: "c4_grass",
                15: "c3_crop",
            }

            # read the simulation source and reference source
            for evaluation_item in evaluation_items:
                logging.info(f"now processing the evaluation item: {evaluation_item}")
                sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
                ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]
                # if the sim_sources and ref_sources are not list, then convert them to list
                if isinstance(sim_sources, str):
                    sim_sources = [sim_sources]
                if isinstance(ref_sources, str):
                    ref_sources = [ref_sources]
                for ref_source in ref_sources:
                    for sim_source in sim_sources:
                        ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                        sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]
                        ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]
                        if ref_data_type == "stn" or sim_data_type == "stn":
                            logging.warning(
                                "Skipping PFT class comparison for %s ref=%s sim=%s: station data is not supported",
                                evaluation_item,
                                ref_source,
                                sim_source,
                            )
                            continue  # Skip processing for station data
                        else:
                            dir_path = _groupby_pair_dir(basedir, "PFT_groupby", sim_source, ref_source)
                            if not os.path.exists(dir_path):
                                os.makedirs(dir_path)

                            if len(self.metrics) > 0:
                                output_file_path = os.path.join(
                                    dir_path, groupby_table_filename(evaluation_item, sim_source, ref_source, "metrics")
                                )
                                rows = []
                                # Print the table header with class names
                                header_values = ["metric"]
                                for PFT_class_name in PFT_class_names.values():
                                    header_values.append(PFT_class_name)
                                header_values.append("Overall")
                                rows.append("\t".join(header_values) + "\n")

                                # Calculate and print median values
                                for metric in self.metrics:
                                    metric_file = _evaluation_netcdf_path(
                                        self.casedir, "metrics", evaluation_item, ref_source, sim_source, metric
                                    )
                                    # Skip if metric file doesn't exist (e.g., skipped in climatology mode)
                                    if not os.path.exists(metric_file):
                                        logging.debug(
                                            f"Skipping metric {metric} - file not found (possibly skipped in climatology mode)"
                                        )
                                        continue
                                    with _open_dataset_safe(metric_file) as ds_file:
                                        ds = Convert_Type.convert_nc(ds_file.load())

                                    # Clip within the scope being summarized.
                                    # Per-class summaries must not inherit a
                                    # global clip that silently drops valid
                                    # class-local tails.
                                    ds = ds.where(np.isfinite(ds), np.nan)
                                    overall_ds = _clip_metric_quantiles(ds, metric)

                                    overall_median = overall_ds[metric].median(skipna=True).values
                                    overall_median_str = (
                                        f"{overall_median:.3f}" if not np.isnan(overall_median) else "N/A"
                                    )

                                    row_values = [metric]
                                    class_datasets = []
                                    class_names = []
                                    for i in range(0, 16):
                                        ds1 = _clip_metric_quantiles(ds.where(PFTtype == i), metric)
                                        PFT_class_name = PFT_class_names.get(i, f"PFT_{i}")
                                        class_datasets.append(ds1)
                                        class_names.append(PFT_class_name)
                                        median_value = ds1[metric].median(skipna=True).values
                                        median_value_str = (
                                            f"{median_value:.3f}" if not np.isnan(median_value) else "N/A"
                                        )
                                        row_values.append(median_value_str)
                                    _write_class_bundle_atomic(
                                        class_datasets,
                                        class_names,
                                        _class_bundle_path(
                                            dir_path, evaluation_item, ref_source, sim_source, metric, "PFT"
                                        ),
                                    )
                                    row_values.append(overall_median_str)
                                    rows.append("\t".join(row_values) + "\n")
                                _write_lines_atomic(output_file_path, rows)

                                selected_metrics = self.metrics
                                # selected_metrics = list(selected_metrics)
                                option["path"] = _groupby_option_path(
                                    self.casedir, "PFT_groupby", sim_source, ref_source
                                )
                                option["item"] = [evaluation_item, sim_source, ref_source]
                                option["groupby"] = "PFT_groupby"
                                make_LC_based_heat_map(output_file_path, selected_metrics, "metric", option)
                                # print(f"PFT class metrics comparison results are saved to {output_file_path}")
                            else:
                                logging.debug("No metrics requested for PFT class comparison")

                            if len(self.scores) > 0:
                                dir_path = _groupby_pair_dir(basedir, "PFT_groupby", sim_source, ref_source)
                                if not os.path.exists(dir_path):
                                    os.makedirs(dir_path)
                                output_file_path2 = os.path.join(
                                    dir_path, groupby_table_filename(evaluation_item, sim_source, ref_source, "scores")
                                )
                                rows = []
                                # Print the table header with class names
                                header_values = ["score"]
                                for PFT_class_name in PFT_class_names.values():
                                    header_values.append(PFT_class_name)
                                header_values.append("Overall")
                                rows.append("\t".join(header_values) + "\n")

                                # Cache mass-weight ref once per (sim, ref)
                                # pair; same pattern as IGBP branch above.
                                cached_mass_ref = None

                                # Calculate and print mean values
                                for score in self.scores:
                                    score_file = _evaluation_netcdf_path(
                                        self.casedir, "scores", evaluation_item, ref_source, sim_source, score
                                    )
                                    # Skip if score file doesn't exist (e.g., skipped in climatology mode)
                                    if not os.path.exists(score_file):
                                        logging.debug(
                                            f"Skipping score {score} - file not found (possibly skipped in climatology mode)"
                                        )
                                        continue
                                    with _open_dataset_safe(score_file) as ds_file:
                                        ds = Convert_Type.convert_nc(ds_file.load())

                                    # Calculate and write the overall mean first
                                    if self.weight.lower() == "area":
                                        weights = np.cos(np.deg2rad(ds.lat))
                                        overall_mean = ds[score].weighted(weights).mean(skipna=True).values
                                    elif self.weight.lower() == "mass":
                                        # Reuse cached ref dataset across the
                                        # 16 PFT-class iterations below.
                                        if cached_mass_ref is None:
                                            with _open_dataset_safe(
                                                f"{self.casedir}/data/{evaluation_item}_ref_{ref_source}_{ref_varname}.nc"
                                            ) as ref_ds:
                                                cached_mass_ref = ref_ds[f"{ref_varname}"].load()
                                        o = cached_mass_ref

                                        # Calculate area weights (cosine of latitude)
                                        area_weights = np.cos(np.deg2rad(ds.lat))

                                        # Calculate absolute flux weights
                                        flux_weights = np.abs(o.mean("time"))

                                        # Combine area and flux weights
                                        combined_weights = area_weights * flux_weights

                                        # Normalize weights to sum to 1
                                        normalized_weights = combined_weights / combined_weights.sum()

                                        # Calculate weighted mean
                                        overall_mean = (
                                            ds[score].weighted(normalized_weights.fillna(0)).mean(skipna=True).values
                                        )
                                    else:
                                        overall_mean = ds[score].mean(skipna=True).values

                                    overall_mean_str = f"{overall_mean:.3f}" if not np.isnan(overall_mean) else "N/A"

                                    row_values = [score]
                                    class_datasets = []
                                    class_names = []
                                    for i in range(0, 16):
                                        ds1 = ds.where(PFTtype == i)
                                        PFT_class_name = PFT_class_names.get(i, f"PFT_{i}")
                                        class_datasets.append(ds1)
                                        class_names.append(PFT_class_name)
                                        # Calculate mean value
                                        if self.weight.lower() == "area":
                                            weights = np.cos(np.deg2rad(ds.lat))
                                            mean_value = ds1[score].weighted(weights).mean(skipna=True).values
                                        elif self.weight.lower() == "mass":
                                            # cached_mass_ref reused from outer score loop
                                            o = cached_mass_ref

                                            # Calculate area weights (cosine of latitude)
                                            area_weights = np.cos(np.deg2rad(ds.lat))

                                            # Calculate absolute flux weights
                                            flux_weights = np.abs(o.mean("time"))

                                            # Combine area and flux weights
                                            combined_weights = area_weights * flux_weights

                                            # Normalize weights to sum to 1
                                            normalized_weights = combined_weights / combined_weights.sum()

                                            # Calculate weighted mean
                                            mean_value = (
                                                ds1[score]
                                                .weighted(normalized_weights.fillna(0))
                                                .mean(skipna=True)
                                                .values
                                            )
                                        else:
                                            mean_value = ds1[score].mean(skipna=True).values

                                        mean_value_str = f"{mean_value:.3f}" if not np.isnan(mean_value) else "N/A"
                                        row_values.append(mean_value_str)
                                    _write_class_bundle_atomic(
                                        class_datasets,
                                        class_names,
                                        _class_bundle_path(
                                            dir_path, evaluation_item, ref_source, sim_source, score, "PFT"
                                        ),
                                    )
                                    row_values.append(overall_mean_str)
                                    rows.append("\t".join(row_values) + "\n")
                                _write_lines_atomic(output_file_path2, rows)

                                selected_scores = self.scores
                                option["path"] = _groupby_option_path(
                                    self.casedir, "PFT_groupby", sim_source, ref_source
                                )
                                option["item"] = [evaluation_item, sim_source, ref_source]
                                option["groupby"] = "PFT_groupby"
                                make_LC_based_heat_map(output_file_path2, selected_scores, "score", option)
                                # print(f"PFT class scores comparison results are saved to {output_file_path2}")
                            else:
                                logging.debug("No scores requested for PFT class comparison")

        metricsdir_path = os.path.join(f"{casedir}", "comparisons", "PFT_groupby")
        # if os.path.exists(metricsdir_path):
        #     shutil.rmtree(metricsdir_path)
        # print(f"Re-creating output directory: {metricsdir_path}")
        if not os.path.exists(metricsdir_path):
            os.makedirs(metricsdir_path)

        scoresdir_path = os.path.join(f"{casedir}", "comparisons", "PFT_groupby")
        # if os.path.exists(scoresdir_path):
        #     shutil.rmtree(scoresdir_path)
        # print(f"Re-creating output directory: {scoresdir_path}")
        if not os.path.exists(scoresdir_path):
            os.makedirs(scoresdir_path)

        _PFT_class_remap(self)
        _scenarios_PFT_groupby(casedir, scores, metrics, sim_nml, ref_nml, evaluation_items)
