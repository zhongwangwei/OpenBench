import gc
import logging
import os
import re

import numpy as np
import pandas as pd
import xarray as xr

# Check the platform
from openbench.core.metrics import metrics
from openbench.core.scores import scores
from openbench.core.statistics import statistics_calculate
from openbench.util.converttype import Convert_Type
from openbench.util.filenames import (
    diff_grid_anomaly_filename,
    diff_grid_difference_filename,
    diff_station_anomaly_filename,
    diff_station_difference_filename,
    groupby_pair_dirname,
    groupby_table_filename,
    join_filename_components,
    legacy_groupby_pair_dirname,
    legacy_groupby_table_filename,
    relative_grid_score_filename,
    relative_station_scores_filename,
)

from openbench.visualization import (
    make_Correlation,
    make_CZ_based_heat_map,
    make_Functional_Response,
    make_geo_plot_index,
    make_LC_based_heat_map,
    make_Mann_Kendall_Trend_Test,
    make_plot_index_grid,
    make_plot_index_stn,
    make_scenarios_comparison_Diff_Plot,
    make_scenarios_comparison_Kernel_Density_Estimate,
    make_scenarios_comparison_parallel_coordinates,
    make_scenarios_comparison_Portrait_Plot_seasonal,
    make_scenarios_comparison_radar_map,
    make_scenarios_comparison_Relative_Score,
    make_scenarios_comparison_Ridgeline_Plot,
    make_scenarios_comparison_Single_Model_Performance_Index,
    make_scenarios_comparison_Target_Diagram,
    make_scenarios_comparison_Taylor_Diagram,
    make_scenarios_comparison_Whisker_Plot,
    make_scenarios_scores_comparison_heat_map,
    make_Standard_Deviation,
    make_stn_plot_index,
)


def _relative_station_scores_path(dir_path: str, evaluation_item: str, ref_source: str, sim_source: str) -> str:
    return os.path.join(dir_path, relative_station_scores_filename(evaluation_item, ref_source, sim_source))


def _legacy_relative_station_scores_path(dir_path: str, evaluation_item: str, ref_source: str, sim_source: str) -> str:
    return os.path.join(dir_path, f"{evaluation_item}_stn_{ref_source}_{sim_source}_relative_scores.csv")


def _relative_grid_score_path(dir_path: str, evaluation_item: str, ref_source: str, sim_source: str, score: str) -> str:
    return os.path.join(dir_path, relative_grid_score_filename(evaluation_item, ref_source, sim_source, score))


def _legacy_relative_grid_score_path(
    dir_path: str, evaluation_item: str, ref_source: str, sim_source: str, score: str
) -> str:
    return os.path.join(dir_path, f"{evaluation_item}_ref_{ref_source}_sim_{sim_source}_Relative{score}.nc")


def _diff_station_anomaly_path(
    dir_path: str, evaluation_item: str, ref_source: str, sim_source: str, item_type: str
) -> str:
    return os.path.join(dir_path, diff_station_anomaly_filename(evaluation_item, ref_source, sim_source, item_type))


def _legacy_diff_station_anomaly_path(
    dir_path: str, evaluation_item: str, ref_source: str, sim_source: str, item_type: str
) -> str:
    return os.path.join(dir_path, f"{evaluation_item}_stn_{ref_source}_sim_{sim_source}_{item_type}_anomaly.csv")


def _diff_station_difference_path(
    dir_path: str,
    evaluation_item: str,
    ref_source: str,
    sim1: str,
    sim_varname_1: str,
    sim2: str,
    sim_varname_2: str,
    item_type: str,
) -> str:
    return os.path.join(
        dir_path,
        diff_station_difference_filename(
            evaluation_item, ref_source, sim1, sim_varname_1, sim2, sim_varname_2, item_type
        ),
    )


def _legacy_diff_station_difference_path(
    dir_path: str,
    evaluation_item: str,
    ref_source: str,
    sim1: str,
    sim_varname_1: str,
    sim2: str,
    sim_varname_2: str,
    item_type: str,
) -> str:
    return os.path.join(
        dir_path,
        f"{evaluation_item}_stn_{ref_source}_{sim1}_{sim_varname_1}_vs_{sim2}_{sim_varname_2}_{item_type}_diff.csv",
    )


def _diff_grid_anomaly_path(
    dir_path: str, evaluation_item: str, ref_source: str, sim_source: str, item_type: str
) -> str:
    return os.path.join(dir_path, diff_grid_anomaly_filename(evaluation_item, ref_source, sim_source, item_type))


def _legacy_diff_grid_anomaly_path(
    dir_path: str, evaluation_item: str, ref_source: str, sim_source: str, item_type: str
) -> str:
    return os.path.join(dir_path, f"{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{item_type}_anomaly.nc")


def _diff_grid_difference_path(
    dir_path: str, evaluation_item: str, ref_source: str, sim1: str, sim2: str, item_type: str
) -> str:
    return os.path.join(dir_path, diff_grid_difference_filename(evaluation_item, ref_source, sim1, sim2, item_type))


def _legacy_diff_grid_difference_path(
    dir_path: str, evaluation_item: str, ref_source: str, sim1: str, sim2: str, item_type: str
) -> str:
    return os.path.join(dir_path, f"{evaluation_item}_ref_{ref_source}_{sim1}_vs_{sim2}_{item_type}_diff.nc")


def _groupby_pair_dir(root: str, groupby_name: str, sim_source: str, ref_source: str) -> str:
    return os.path.join(root, "comparisons", groupby_name, groupby_pair_dirname(sim_source, ref_source))


def _legacy_groupby_pair_dir(root: str, groupby_name: str, sim_source: str, ref_source: str) -> str:
    return os.path.join(root, "comparisons", groupby_name, legacy_groupby_pair_dirname(sim_source, ref_source))


def _groupby_table_path(
    root: str, groupby_name: str, evaluation_item: str, sim_source: str, ref_source: str, table_type: str
) -> str:
    return os.path.join(
        _groupby_pair_dir(root, groupby_name, sim_source, ref_source),
        groupby_table_filename(evaluation_item, sim_source, ref_source, table_type),
    )


def _legacy_groupby_table_path(
    root: str, groupby_name: str, evaluation_item: str, sim_source: str, ref_source: str, table_type: str
) -> str:
    return os.path.join(
        _legacy_groupby_pair_dir(root, groupby_name, sim_source, ref_source),
        legacy_groupby_table_filename(evaluation_item, sim_source, ref_source, table_type),
    )


def _only_drawing_missing_file(
    path: str, *, producer: str, candidate_paths: list[str] | None = None
) -> FileNotFoundError:
    """Build a user-actionable only_drawing missing-file error."""
    tried = f" Also tried legacy/fallback paths: {candidate_paths}." if candidate_paths else ""
    return FileNotFoundError(
        f"only_drawing missing required file: {path}. "
        f"{tried}Run the full {producer} first (set only_drawing=False) to generate required data files."
    )


def _require_only_drawing_file(
    path: str, *, producer: str, fallback_ext: str | None = None, fallback_paths: list[str] | None = None
) -> str:
    """Return a mandatory only_drawing input path, optionally trying legacy/fallback names."""
    if os.path.exists(path):
        return path
    candidate_paths = []
    if fallback_ext is not None:
        root, _ext = os.path.splitext(path)
        candidate_paths.append(f"{root}{fallback_ext}")
    if fallback_paths:
        candidate_paths.extend(fallback_paths)
        if fallback_ext is not None:
            for fallback_path in fallback_paths:
                root, _ext = os.path.splitext(fallback_path)
                candidate_paths.append(f"{root}{fallback_ext}")
    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path
    logging.error("File not found in only_drawing mode: %s", path)
    raise _only_drawing_missing_file(path, producer=producer, candidate_paths=candidate_paths)


def _skip_unrequested_only_drawing(*, figure: str, reason: str) -> None:
    """Log a debug-only skip for work not requested or not meaningful for this figure."""
    logging.debug("Skipping %s only_drawing: %s", figure, reason)


def _skip_optional_only_drawing(*, figure: str, reason: str) -> None:
    """Log a warning skip for optional derived drawings that cannot be rendered."""
    logging.warning("Skipping optional %s only_drawing output: %s", figure, reason)


def _unsupported_only_drawing(*, figure: str, reason: str) -> ValueError:
    """Build a fail-fast error for requested figures with unsupported inputs."""
    return ValueError(f"{figure} only_drawing cannot render requested figure: {reason}")


def _existing_only_drawing_file(
    path: str, *, fallback_ext: str | None = None, fallback_paths: list[str] | None = None
) -> str | None:
    """Return an existing only_drawing input path without raising when none exist."""
    if os.path.exists(path):
        return path
    candidate_paths = []
    if fallback_ext is not None:
        root, _ext = os.path.splitext(path)
        candidate_paths.append(f"{root}{fallback_ext}")
    if fallback_paths:
        candidate_paths.extend(fallback_paths)
        if fallback_ext is not None:
            for fallback_path in fallback_paths:
                root, _ext = os.path.splitext(fallback_path)
                candidate_paths.append(f"{root}{fallback_ext}")
    return next((candidate_path for candidate_path in candidate_paths if os.path.exists(candidate_path)), None)


def _require_groupby_only_drawing_file(
    path: str, *, fallback_ext: str | None = ".txt", fallback_paths: list[str] | None = None
) -> str:
    """Fail fast for requested groupby figures when their producer table is missing."""
    return _require_only_drawing_file(
        path,
        producer="comparison groupby",
        fallback_ext=fallback_ext,
        fallback_paths=fallback_paths,
    )


def _require_finite_values(values, *, path: str, variable: str) -> None:
    """Fail only_drawing before rendering stale/empty/partial precomputed outputs."""
    array = np.asarray(values)
    if not np.isfinite(array).any():
        raise ValueError(f"only_drawing input has no finite data for {variable}: {path}")


def _read_only_drawing_csv(path: str) -> pd.DataFrame:
    """Read comma/tab separated only_drawing input with delimiter auto-detection."""
    return Convert_Type.convert_Frame(pd.read_csv(path, sep=None, engine="python", header=0))


def _require_csv_columns(path: str, columns: list[str]) -> pd.DataFrame:
    """Read a precomputed CSV and require all columns used by the renderer."""
    df = _read_only_drawing_csv(path)
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"only_drawing input missing columns {missing}: {path}")
    return df


def _require_csv_finite(path: str, value_column: str, *, required_columns: list[str] | None = None) -> None:
    """Require a precomputed CSV value column to contain at least one finite value."""
    columns = [value_column, *(required_columns or [])]
    df = _require_csv_columns(path, columns)
    _require_finite_values(df[value_column].values, path=path, variable=value_column)


def _require_netcdf_finite(path: str, variable: str, *, producer: str = "comparison") -> None:
    """Require a precomputed NetCDF variable to exist and contain at least one finite value."""
    path = _require_only_drawing_file(path, producer=producer)
    with xr.open_dataset(path) as ds:
        if variable not in ds:
            raise KeyError(f"only_drawing input missing variable {variable}: {path}")
        data = ds[variable].load()
    _require_finite_values(data, path=path, variable=variable)


def _require_netcdf_variables(
    path: str, variables: list[str], *, finite_variables: list[str] | None = None, producer: str = "comparison"
) -> str:
    """Require precomputed NetCDF variables, optionally requiring finite values in selected variables."""
    path = _require_only_drawing_file(path, producer=producer)
    finite_variables = finite_variables or []
    with xr.open_dataset(path) as ds:
        for variable in variables:
            if variable not in ds:
                raise KeyError(f"only_drawing input missing variable {variable}: {path}")
        finite_data = {variable: ds[variable].load() for variable in finite_variables}
    for variable, data in finite_data.items():
        _require_finite_values(data, path=path, variable=variable)
    return path


class Evaluation_grid_only_drawing(metrics, scores):
    def __init__(self, info, fig_nml):
        self.name = "Evaluation_grid_only_drawing"
        self.version = "0.1"
        self.release = "0.1"
        self.date = "June 2025"
        self.author = "Xionghui Xu"
        self.__dict__.update(info)
        self.fig_nml = fig_nml

        logging.info(" ")
        logging.info("╔═══════════════════════════════════════════════════════════════╗")
        logging.info("║                Drawing evaluation processes starting!         ║")
        logging.info("╚═══════════════════════════════════════════════════════════════╝")
        logging.info(" ")

    def make_Evaluation(self, **kwargs):
        try:
            make_plot_index_grid(self)
        finally:
            gc.collect()  # Final cleanup


class Evaluation_stn_only_drawing(metrics, scores):
    def __init__(self, info, fig_nml):
        self.name = "Evaluation_point_only_drawing"
        self.version = "0.1"
        self.release = "0.1"
        self.date = "June 2025"
        self.author = "Xionghui Xu"
        self.__dict__.update(info)
        self.fig_nml = fig_nml
        if isinstance(self.sim_varname, str):
            self.sim_varname = [self.sim_varname]
        if isinstance(self.ref_varname, str):
            self.ref_varname = [self.ref_varname]

        logging.info(" ")
        logging.info("╔═══════════════════════════════════════════════════════════════╗")
        logging.info("║                  Evaluation processes starting!               ║")
        logging.info("╚═══════════════════════════════════════════════════════════════╝")
        logging.info(" ")

    def make_evaluation_P(self):
        try:
            make_plot_index_stn(self)
        finally:
            gc.collect()  # Final cleanup


class LC_groupby_only_drawing(metrics, scores):
    def __init__(self, main_nml, scores, metrics):
        self.name = "StatisticsDataHandler_only_drawing"
        self.version = "0.1"
        self.release = "0.1"
        self.date = "June 2025"
        self.author = "Xionghui Xu"
        self.main_nml = main_nml
        self.general_config = self.main_nml["general"]
        self._igbp_station_warning_shown = False  # Track if IGBP station data warning has been shown
        self._pft_station_warning_shown = False  # Track if PFT station data warning has been shown
        # update self based on self.general_config
        self.__dict__.update(self.general_config)
        # Extract remapping information from main namelist
        self.compare_grid_res = self.main_nml["general"]["compare_grid_res"]
        self.compare_tim_res = self.main_nml["general"].get("compare_tim_res", "1").lower()
        self.casedir = os.path.join(self.main_nml["general"]["basedir"], self.main_nml["general"]["basename"])
        # Set default weight method to 'none'
        self.weight = self.main_nml["general"].get("weight", "none")
        # this should be done in read_namelist
        # adjust the time frequency
        # Check if climatology mode - skip frequency parsing
        if self.compare_tim_res in ["climatology-year", "climatology-month"]:
            logging.debug(
                f"LC_groupby_only_drawing: Climatology mode detected ({self.compare_tim_res}), skipping frequency conversion"
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
        def _scenarios_IGBP_groupby(basedir, scores, metrics, sim_nml, ref_nml, evaluation_items):
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
                        if ref_data_type == "stn" or sim_data_type == "stn":
                            if not self._igbp_station_warning_shown:
                                logging.warning("warning: station data is not supported for IGBP class comparison")
                                self._igbp_station_warning_shown = True
                            pass
                        else:
                            if len(self.metrics) > 0:
                                output_file_path = _groupby_table_path(
                                    basedir, "IGBP_groupby", evaluation_item, sim_source, ref_source, "metrics"
                                )
                                metrics_file = _require_groupby_only_drawing_file(
                                    output_file_path,
                                    fallback_ext=".txt",
                                    fallback_paths=[
                                        _legacy_groupby_table_path(
                                            basedir, "IGBP_groupby", evaluation_item, sim_source, ref_source, "metrics"
                                        )
                                    ],
                                )

                                selected_metrics = self.metrics
                                option["path"] = (
                                    _groupby_pair_dir(self.casedir, "IGBP_groupby", sim_source, ref_source) + os.sep
                                )
                                option["item"] = [evaluation_item, sim_source, ref_source]
                                option["groupby"] = "IGBP_groupby"
                                make_LC_based_heat_map(metrics_file, selected_metrics, "metric", option)
                            else:
                                _skip_unrequested_only_drawing(
                                    figure="IGBP_groupby", reason="no metrics were requested"
                                )

                            if len(self.scores) > 0:
                                output_file_path2 = _groupby_table_path(
                                    basedir, "IGBP_groupby", evaluation_item, sim_source, ref_source, "scores"
                                )
                                legacy_root_scores_path = os.path.join(
                                    basedir,
                                    "comparisons",
                                    "IGBP_groupby",
                                    legacy_groupby_table_filename(evaluation_item, sim_source, ref_source, "scores"),
                                )
                                scores_file = _require_groupby_only_drawing_file(
                                    output_file_path2,
                                    fallback_ext=".txt",
                                    fallback_paths=[
                                        _legacy_groupby_table_path(
                                            basedir, "IGBP_groupby", evaluation_item, sim_source, ref_source, "scores"
                                        ),
                                        legacy_root_scores_path,
                                    ],
                                )

                                selected_scores = self.scores
                                option["path"] = (
                                    _groupby_pair_dir(self.casedir, "IGBP_groupby", sim_source, ref_source) + os.sep
                                )
                                option["item"] = [evaluation_item, sim_source, ref_source]
                                option["groupby"] = "IGBP_groupby"
                                make_LC_based_heat_map(scores_file, selected_scores, "score", option)
                            else:
                                _skip_unrequested_only_drawing(figure="IGBP_groupby", reason="no scores were requested")

        _scenarios_IGBP_groupby(casedir, scores, metrics, sim_nml, ref_nml, evaluation_items)

    def scenarios_PFT_groupby_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        def _scenarios_PFT_groupby(basedir, scores, metrics, sim_nml, ref_nml, evaluation_items):
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
                        if ref_data_type == "stn" or sim_data_type == "stn":
                            if not self._pft_station_warning_shown:
                                logging.warning("warning: station data is not supported for PFT class comparison")
                                self._pft_station_warning_shown = True
                        else:
                            if len(self.metrics) > 0:
                                output_file_path = _groupby_table_path(
                                    basedir, "PFT_groupby", evaluation_item, sim_source, ref_source, "metrics"
                                )
                                metrics_file = _require_groupby_only_drawing_file(
                                    output_file_path,
                                    fallback_ext=".txt",
                                    fallback_paths=[
                                        _legacy_groupby_table_path(
                                            basedir, "PFT_groupby", evaluation_item, sim_source, ref_source, "metrics"
                                        )
                                    ],
                                )

                                selected_metrics = self.metrics
                                option["path"] = (
                                    _groupby_pair_dir(self.casedir, "PFT_groupby", sim_source, ref_source) + os.sep
                                )
                                option["item"] = [evaluation_item, sim_source, ref_source]
                                option["groupby"] = "PFT_groupby"
                                make_LC_based_heat_map(metrics_file, selected_metrics, "metric", option)
                            else:
                                _skip_unrequested_only_drawing(figure="PFT_groupby", reason="no metrics were requested")

                            if len(self.scores) > 0:
                                output_file_path2 = _groupby_table_path(
                                    basedir, "PFT_groupby", evaluation_item, sim_source, ref_source, "scores"
                                )
                                scores_file = _require_groupby_only_drawing_file(
                                    output_file_path2,
                                    fallback_ext=".txt",
                                    fallback_paths=[
                                        _legacy_groupby_table_path(
                                            basedir, "PFT_groupby", evaluation_item, sim_source, ref_source, "scores"
                                        )
                                    ],
                                )

                                selected_scores = self.scores
                                option["path"] = (
                                    _groupby_pair_dir(self.casedir, "PFT_groupby", sim_source, ref_source) + os.sep
                                )
                                option["item"] = [evaluation_item, sim_source, ref_source]
                                option["groupby"] = "PFT_groupby"
                                make_LC_based_heat_map(scores_file, selected_scores, "score", option)
                            else:
                                _skip_unrequested_only_drawing(figure="PFT_groupby", reason="no scores were requested")

        _scenarios_PFT_groupby(casedir, scores, metrics, sim_nml, ref_nml, evaluation_items)


class CZ_groupby_only_drawing(metrics, scores):
    def __init__(self, main_nml, scores, metrics):
        self.name = "StatisticsDataHandler_only_drawing"
        self.version = "0.1"
        self.release = "0.1"
        self.date = "June 2025"
        self.author = "Xionghui Xu"
        self.main_nml = main_nml
        self.general_config = self.main_nml["general"]
        self._station_warning_shown = False  # Track if station data warning has been shown
        # update self based on self.general_config
        self.__dict__.update(self.general_config)
        # Extract remapping information from main namelist
        self.compare_grid_res = self.main_nml["general"]["compare_grid_res"]
        self.compare_tim_res = self.main_nml["general"].get("compare_tim_res", "1").lower()
        self.casedir = os.path.join(self.main_nml["general"]["basedir"], self.main_nml["general"]["basename"])
        # Set default weight method to 'none'
        self.weight = self.main_nml["general"].get("weight", "none")
        # this should be done in read_namelist
        # adjust the time frequency
        # Check if climatology mode - skip frequency parsing
        if self.compare_tim_res in ["climatology-year", "climatology-month"]:
            logging.debug(
                f"CZ_groupby_only_drawing: Climatology mode detected ({self.compare_tim_res}), skipping frequency conversion"
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

    def scenarios_CZ_groupby_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        """Render CZ groupby figures from existing groupby tables only.

        ``only_drawing`` must not remap ``Climate_zone.nc`` or recompute
        per-zone metrics/scores from evaluation NetCDF files. The full
        groupby phase owns those CSV/TXT intermediates; this method only
        reloads them and regenerates figures.
        """
        for evaluation_item in evaluation_items:
            logging.info(f"now processing the evaluation item: {evaluation_item}")
            sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
            ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]
            if isinstance(sim_sources, str):
                sim_sources = [sim_sources]
            if isinstance(ref_sources, str):
                ref_sources = [ref_sources]

            for ref_source in ref_sources:
                for sim_source in sim_sources:
                    ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                    sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]
                    if ref_data_type == "stn" or sim_data_type == "stn":
                        if not self._station_warning_shown:
                            logging.warning("warning: station data is not supported for Climate zone class comparison")
                            self._station_warning_shown = True
                        continue

                    option["path"] = _groupby_pair_dir(self.casedir, "CZ_groupby", sim_source, ref_source)
                    option["item"] = [evaluation_item, sim_source, ref_source]
                    option["groupby"] = "CZ_groupby"

                    if len(self.metrics) > 0:
                        metrics_path = _groupby_table_path(
                            casedir, "CZ_groupby", evaluation_item, sim_source, ref_source, "metrics"
                        )
                        metrics_file = _require_groupby_only_drawing_file(
                            metrics_path,
                            fallback_ext=".txt",
                            fallback_paths=[
                                _legacy_groupby_table_path(
                                    casedir, "CZ_groupby", evaluation_item, sim_source, ref_source, "metrics"
                                )
                            ],
                        )
                        make_CZ_based_heat_map(metrics_file, self.metrics, "metric", option)
                    else:
                        _skip_unrequested_only_drawing(figure="CZ_groupby", reason="no metrics were requested")

                    if len(self.scores) > 0:
                        scores_path = _groupby_table_path(
                            casedir, "CZ_groupby", evaluation_item, sim_source, ref_source, "scores"
                        )
                        scores_file = _require_groupby_only_drawing_file(
                            scores_path,
                            fallback_ext=".txt",
                            fallback_paths=[
                                _legacy_groupby_table_path(
                                    casedir, "CZ_groupby", evaluation_item, sim_source, ref_source, "scores"
                                )
                            ],
                        )
                        make_CZ_based_heat_map(scores_file, self.scores, "score", option)
                    else:
                        _skip_unrequested_only_drawing(figure="CZ_groupby", reason="no scores were requested")


class ComparisonProcessing_only_drawing(metrics, scores, statistics_calculate):
    def __init__(self, main_nml, scores, metrics):
        self.name = "ComparisonDataHandler_only_drawing"
        self.version = "0.1"
        self.release = "0.1"
        self.date = "June 2025"
        self.author = "Xionghui Xu"
        self.main_nml = main_nml
        self.general_config = self.main_nml["general"]
        # update self based on self.general_config
        self.__dict__.update(self.general_config)
        self.compare_nml = {}
        # Add default weight attribute
        self.weight = self.main_nml["general"].get("weight", "none")  # Default to 'none' if not specified
        self._igbp_station_warning_shown = False  # Track if IGBP station data warning has been shown

        # Extract remapping information from main namelist
        self.compare_grid_res = self.main_nml["general"]["compare_grid_res"]
        self.compare_tim_res = self.main_nml["general"].get("compare_tim_res", "1").lower()
        self.casedir = os.path.join(self.main_nml["general"]["basedir"], self.main_nml["general"]["basename"])
        # this should be done in read_namelist
        # adjust the time frequency
        # Check if climatology mode - skip frequency parsing
        if self.compare_tim_res in ["climatology-year", "climatology-month"]:
            logging.debug(
                f"ComparisonProcessing_only_drawing: Climatology mode detected ({self.compare_tim_res}), skipping frequency conversion"
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
            # Get the corresponding pandas frequency
            freq = self.freq_map.get(unit.lower())
            if not freq:
                raise ValueError(f"Unsupported time unit: {unit}")
            self.compare_tim_res = f"{value}{freq}E"

        self.metrics = metrics
        self.scores = scores

        # self.ref_source              =  ref_source
        # self.sim_source              =  sim_source

    def scenarios_IGBP_groupby_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        LC_groupby_only_drawing(self.main_nml, self.scores, self.metrics).scenarios_IGBP_groupby_comparison(
            casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option
        )

    def scenarios_PFT_groupby_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        LC_groupby_only_drawing(self.main_nml, self.scores, self.metrics).scenarios_PFT_groupby_comparison(
            casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option
        )

    def scenarios_HeatMap_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        try:
            dir_path = os.path.join(casedir, "comparisons", "HeatMap")
            for score in scores:
                output_file_path = os.path.join(dir_path, f"scenarios_{score}_comparison.csv")
                output_file_path = _require_only_drawing_file(
                    output_file_path, producer="comparison", fallback_ext=".txt"
                )
                make_scenarios_scores_comparison_heat_map(output_file_path, score, option)
        finally:
            gc.collect()  # Clean up memory after processing

    def scenarios_Taylor_Diagram_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        try:
            dir_path = os.path.join(casedir, "comparisons", "Taylor_Diagram")
            os.makedirs(dir_path, exist_ok=True)

            # read the simulation source and reference source
            for evaluation_item in evaluation_items:
                try:
                    # read the simulation source and reference source
                    sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
                    ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]
                    # if the sim_sources and ref_sources are not list, then convert them to list
                    if isinstance(sim_sources, str):
                        sim_sources = [sim_sources]
                    if isinstance(ref_sources, str):
                        ref_sources = [ref_sources]

                    for ref_source in ref_sources:
                        try:
                            output_file_path = os.path.join(
                                dir_path,
                                f"{join_filename_components('taylor_diagram', evaluation_item, ref_source)}.csv",
                            )
                            legacy_output_file_path = os.path.join(
                                dir_path, f"taylor_diagram_{evaluation_item}_{ref_source}.csv"
                            )
                            output_file_path = _require_only_drawing_file(
                                output_file_path,
                                producer="comparison",
                                fallback_ext=".txt",
                                fallback_paths=[legacy_output_file_path],
                            )
                            stds = np.zeros(len(sim_sources) + 1)
                            cors = np.zeros(len(sim_sources) + 1)
                            RMSs = np.zeros(len(sim_sources) + 1)
                            summary = pd.read_csv(output_file_path, sep="\t").iloc[0]
                            if "Reference_std" in summary and pd.notna(summary["Reference_std"]):
                                stds[0] = float(summary["Reference_std"])
                            else:
                                stds[0] = float(summary[f"{sim_sources[0]}_std_ref"])
                            for i, sim_source in enumerate(sim_sources):
                                stds[i + 1] = float(summary[f"{sim_source}_std"])
                                cors[i + 1] = float(summary[f"{sim_source}_COR"])
                                RMSs[i + 1] = float(summary[f"{sim_source}_RMS"])
                            make_scenarios_comparison_Taylor_Diagram(
                                casedir, evaluation_item, stds, RMSs, cors, ref_source, sim_sources, option
                            )
                        finally:
                            gc.collect()  # Clean up memory after processing each reference source
                finally:
                    gc.collect()  # Clean up memory after processing each evaluation item
        finally:
            gc.collect()  # Final cleanup for the entire method

    def scenarios_Target_Diagram_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        try:
            dir_path = os.path.join(casedir, "comparisons", "Target_Diagram")
            os.makedirs(dir_path, exist_ok=True)

            # read the simulation source and reference source
            for evaluation_item in evaluation_items:
                try:
                    # read the simulation source and reference source
                    sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
                    ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]
                    # if the sim_sources and ref_sources are not list, then convert them to list
                    if isinstance(sim_sources, str):
                        sim_sources = [sim_sources]
                    if isinstance(ref_sources, str):
                        ref_sources = [ref_sources]

                    for ref_source in ref_sources:
                        try:
                            output_file_path = os.path.join(
                                dir_path,
                                f"{join_filename_components('target_diagram', evaluation_item, ref_source)}.csv",
                            )
                            legacy_output_file_path = os.path.join(
                                dir_path, f"target_diagram_{evaluation_item}_{ref_source}.csv"
                            )
                            output_file_path = _require_only_drawing_file(
                                output_file_path,
                                producer="comparison",
                                fallback_ext=".txt",
                                fallback_paths=[legacy_output_file_path],
                            )
                            biases = np.zeros(len(sim_sources))
                            rmses = np.zeros(len(sim_sources))
                            crmsds = np.zeros(len(sim_sources))
                            with open(output_file_path, "r") as file:
                                lines = file.readlines()
                            second_row = lines[1].strip().split("\t")
                            values = [float(x) for x in second_row[2:] if x]
                            for i, sim_source in enumerate(sim_sources):
                                biases[i] = values[i * 3]
                                rmses[i] = values[i * 3 + 1]
                                crmsds[i] = values[i * 3 + 2]

                            # Target diagram expects (bias, crmsd, rmsd): centered CRMSD in the
                            # crmsd slot (x-axis uRMSD), total RMSD in the rmsd slot. Do NOT swap.
                            make_scenarios_comparison_Target_Diagram(
                                dir_path, evaluation_item, biases, crmsds, rmses, ref_source, sim_sources, option
                            )
                        finally:
                            gc.collect()  # Clean up memory after processing each reference source
                finally:
                    gc.collect()  # Clean up memory after processing each evaluation item
        finally:
            gc.collect()  # Final cleanup for the entire method

    def scenarios_Kernel_Density_Estimate_comparison(
        self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option
    ):
        try:
            dir_path = os.path.join(basedir, "comparisons", "Kernel_Density_Estimate")
            os.makedirs(dir_path, exist_ok=True)

            # fixme: add the Kernel Density Estimate
            for evaluation_item in evaluation_items:
                try:
                    sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
                    ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]
                    # if the sim_sources and ref_sources are not list, then convert them to list
                    if isinstance(sim_sources, str):
                        sim_sources = [sim_sources]
                    if isinstance(ref_sources, str):
                        ref_sources = [ref_sources]

                    for score in scores:
                        try:
                            # Skip nSpatialScore since it's a constant value
                            if score == "nSpatialScore":
                                _skip_optional_only_drawing(
                                    figure="Kernel Density Estimate", reason=f"{score} is a constant value"
                                )
                                continue
                            for ref_source in ref_sources:
                                try:
                                    datasets_filtered = []
                                    # create a numpy matrix to store the data
                                    for sim_source in sim_sources:
                                        try:
                                            ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                                            sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]

                                            if ref_data_type == "stn" or sim_data_type == "stn":
                                                ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]
                                                sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]
                                                if sim_varname is None or sim_varname == "":
                                                    sim_varname = evaluation_item
                                                if ref_varname is None or ref_varname == "":
                                                    ref_varname = evaluation_item

                                                file_path = os.path.join(
                                                    basedir,
                                                    "scores",
                                                    f"{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv",
                                                )
                                                if not os.path.exists(file_path):
                                                    _require_only_drawing_file(file_path, producer="evaluation")
                                                # read the file_path data and select the score
                                                df = pd.read_csv(file_path, sep=",", header=0)
                                                df = Convert_Type.convert_Frame(df)
                                                data = df[score].values
                                            else:
                                                file_path = os.path.join(
                                                    basedir,
                                                    "scores",
                                                    f"{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{score}.nc",
                                                )
                                                file_path = _require_only_drawing_file(file_path, producer="evaluation")
                                                with xr.open_dataset(file_path) as _ds:
                                                    ds = _ds.load()
                                                ds = Convert_Type.convert_nc(ds)
                                                data = ds[score].values
                                            datasets_filtered.append(
                                                data[~np.isnan(data)]
                                            )  # Filter out NaNs and append
                                        finally:
                                            gc.collect()  # Clean up memory after processing each simulation source

                                    try:
                                        make_scenarios_comparison_Kernel_Density_Estimate(
                                            dir_path,
                                            evaluation_item,
                                            ref_source,
                                            sim_sources,
                                            score,
                                            datasets_filtered,
                                            option,
                                        )
                                    except Exception:
                                        logging.exception(
                                            f"Error: {evaluation_item} {ref_source} {sim_sources} {score} Kernel Density Estimate failed!"
                                        )
                                        raise
                                finally:
                                    gc.collect()  # Clean up memory after processing each reference source
                        finally:
                            gc.collect()  # Clean up memory after processing each score

                    for metric in metrics:
                        try:
                            for ref_source in ref_sources:
                                try:
                                    datasets_filtered = []
                                    # create a numpy matrix to store the data
                                    for sim_source in sim_sources:
                                        try:
                                            ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                                            sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]

                                            if ref_data_type == "stn" or sim_data_type == "stn":
                                                ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]
                                                sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]
                                                if sim_varname is None or sim_varname == "":
                                                    sim_varname = evaluation_item
                                                if ref_varname is None or ref_varname == "":
                                                    ref_varname = evaluation_item

                                                file_path = os.path.join(
                                                    basedir,
                                                    "metrics",
                                                    f"{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv",
                                                )
                                                if not os.path.exists(file_path):
                                                    _require_only_drawing_file(file_path, producer="evaluation")
                                                # read the file_path data and select the metric
                                                df = pd.read_csv(file_path, sep=",", header=0)
                                                data = df[metric].values
                                            else:
                                                file_path = os.path.join(
                                                    basedir,
                                                    "metrics",
                                                    f"{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{metric}.nc",
                                                )
                                                file_path = _require_only_drawing_file(file_path, producer="evaluation")
                                                with xr.open_dataset(file_path) as _ds:
                                                    ds = _ds.load()
                                                ds = Convert_Type.convert_nc(ds)
                                                data = ds[metric].values

                                            data = data[~np.isinf(data)]
                                            if metric == "percent_bias":
                                                data = data[(data >= -100) & (data <= 100)]
                                            datasets_filtered.append(
                                                data[~np.isnan(data)]
                                            )  # Filter out NaNs and append
                                        finally:
                                            gc.collect()  # Clean up memory after processing each simulation source

                                    try:
                                        make_scenarios_comparison_Kernel_Density_Estimate(
                                            dir_path,
                                            evaluation_item,
                                            ref_source,
                                            sim_sources,
                                            metric,
                                            datasets_filtered,
                                            option,
                                        )
                                    except Exception:
                                        logging.exception(
                                            f"Error: {evaluation_item} {ref_source} {sim_sources} {metric} Kernel Density Estimate failed!"
                                        )
                                        raise
                                finally:
                                    gc.collect()  # Clean up memory after processing each reference source
                        finally:
                            gc.collect()  # Clean up memory after processing each metric
                finally:
                    gc.collect()  # Clean up memory after processing each evaluation item
        finally:
            gc.collect()  # Final cleanup for the entire method

    def scenarios_Parallel_Coordinates_comparison(
        self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option
    ):
        try:
            dir_path = os.path.join(basedir, "comparisons", "Parallel_Coordinates")
            os.makedirs(dir_path, exist_ok=True)
            output_file_path = os.path.join(dir_path, "Parallel_Coordinates_evaluations.csv")

            output_file_path = _require_only_drawing_file(output_file_path, producer="comparison", fallback_ext=".txt")
            df = pd.read_csv(output_file_path, sep="\t", header=0)
            df = df.dropna(axis=1, how="any")
            # If index in scores or metrics was dropped, then remove the corresponding scores or metrics
            scores = [score for score in scores if score in df.columns]
            metrics = [metric for metric in metrics if metric in df.columns]

            make_scenarios_comparison_parallel_coordinates(
                output_file_path, self.casedir, evaluation_items, scores, metrics, option
            )
        finally:
            gc.collect()  # Final cleanup for the entire method

    def scenarios_Portrait_Plot_seasonal_comparison(
        self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option
    ):
        try:
            dir_path = os.path.join(basedir, "comparisons", "Portrait_Plot_seasonal")
            os.makedirs(dir_path, exist_ok=True)
            output_file_path = os.path.join(dir_path, "Portrait_Plot_seasonal.csv")
            # Fallback to .txt if .csv not found
            if not os.path.exists(output_file_path):
                output_file_path_txt = output_file_path[:-4] + ".txt"
                if os.path.exists(output_file_path_txt):
                    output_file_path = output_file_path_txt
            output_file_path = _require_only_drawing_file(output_file_path, producer="comparison")
            df = _require_csv_columns(output_file_path, ["Item", "Reference", "Simulation"])
            for metric in metrics:
                for season in ["DJF", "MAM", "JJA", "SON"]:
                    column = f"{metric}_{season}"
                    if column in df.columns:
                        _require_finite_values(df[column].values, path=output_file_path, variable=column)
                    else:
                        raise KeyError(f"only_drawing input missing columns ['{column}']: {output_file_path}")
            for score in scores:
                for season in ["DJF", "MAM", "JJA", "SON"]:
                    column = f"{score}_{season}"
                    if column in df.columns:
                        _require_finite_values(df[column].values, path=output_file_path, variable=column)
                    else:
                        raise KeyError(f"only_drawing input missing columns ['{column}']: {output_file_path}")
            make_scenarios_comparison_Portrait_Plot_seasonal(
                output_file_path, self.casedir, evaluation_items, scores, metrics, option
            )
        finally:
            gc.collect()  # Final cleanup for the entire method

    def scenarios_Whisker_Plot_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        try:
            dir_path = os.path.join(basedir, "comparisons", "Whisker_Plot")
            os.makedirs(dir_path, exist_ok=True)

            for evaluation_item in evaluation_items:
                try:
                    sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
                    ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]
                    # If the sim_sources and ref_sources are not lists, convert them to lists
                    if isinstance(sim_sources, str):
                        sim_sources = [sim_sources]
                    if isinstance(ref_sources, str):
                        ref_sources = [ref_sources]

                    for score in scores:
                        try:
                            # Skip nSpatialScore since it's a constant value
                            if score == "nSpatialScore":
                                _skip_optional_only_drawing(
                                    figure="Whisker Plot", reason=f"{score} is a constant value"
                                )
                                continue
                            for ref_source in ref_sources:
                                try:
                                    datasets_filtered = []
                                    # Create a numpy matrix to store the data
                                    for sim_source in sim_sources:
                                        try:
                                            ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                                            sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]

                                            if ref_data_type == "stn" or sim_data_type == "stn":
                                                ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]
                                                sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]
                                                if sim_varname is None or sim_varname == "":
                                                    sim_varname = evaluation_item
                                                if ref_varname is None or ref_varname == "":
                                                    ref_varname = evaluation_item

                                                file_path = os.path.join(
                                                    basedir,
                                                    "scores",
                                                    f"{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv",
                                                )
                                                if not os.path.exists(file_path):
                                                    _require_only_drawing_file(file_path, producer="evaluation")
                                                # Read the file_path data and select the score
                                                df = pd.read_csv(file_path, sep=",", header=0)
                                                df = Convert_Type.convert_Frame(df)
                                                data = df[score].values
                                            else:
                                                file_path = os.path.join(
                                                    basedir,
                                                    "scores",
                                                    f"{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{score}.nc",
                                                )
                                                if not os.path.exists(file_path):
                                                    _require_only_drawing_file(file_path, producer="evaluation")
                                                file_path = _require_only_drawing_file(file_path, producer="evaluation")
                                                with xr.open_dataset(file_path) as _ds:
                                                    ds = _ds.load()
                                                ds = Convert_Type.convert_nc(ds)
                                                data = ds[score].values
                                            datasets_filtered.append(
                                                data[~np.isnan(data)]
                                            )  # Filter out NaNs and append
                                        finally:
                                            gc.collect()  # Clean up memory after processing each simulation source

                                    try:
                                        make_scenarios_comparison_Whisker_Plot(
                                            dir_path,
                                            evaluation_item,
                                            ref_source,
                                            sim_sources,
                                            score,
                                            datasets_filtered,
                                            option,
                                        )
                                    except Exception:
                                        logging.exception(
                                            f"Error: {evaluation_item} {ref_source} {sim_sources} {score} Whisker Plot failed!"
                                        )
                                        raise
                                finally:
                                    gc.collect()  # Clean up memory after processing each reference source
                        finally:
                            gc.collect()  # Clean up memory after processing each score

                    for metric in metrics:
                        try:
                            for ref_source in ref_sources:
                                try:
                                    datasets_filtered = []
                                    # Create a numpy matrix to store the data
                                    for sim_source in sim_sources:
                                        try:
                                            ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                                            sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]

                                            if ref_data_type == "stn" or sim_data_type == "stn":
                                                ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]
                                                sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]
                                                if sim_varname is None or sim_varname == "":
                                                    sim_varname = evaluation_item
                                                if ref_varname is None or ref_varname == "":
                                                    ref_varname = evaluation_item

                                                file_path = os.path.join(
                                                    basedir,
                                                    "metrics",
                                                    f"{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv",
                                                )
                                                if not os.path.exists(file_path):
                                                    _require_only_drawing_file(file_path, producer="evaluation")
                                                # Read the file_path data and select the metric
                                                df = pd.read_csv(file_path, sep=",", header=0)
                                                data = df[metric].values
                                            else:
                                                file_path = os.path.join(
                                                    basedir,
                                                    "metrics",
                                                    f"{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{metric}.nc",
                                                )
                                                if not os.path.exists(file_path):
                                                    _require_only_drawing_file(file_path, producer="evaluation")
                                                file_path = _require_only_drawing_file(file_path, producer="evaluation")
                                                with xr.open_dataset(file_path) as _ds:
                                                    ds = _ds.load()
                                                ds = Convert_Type.convert_nc(ds)
                                                data = ds[metric].values

                                            data = data[~np.isinf(data)]
                                            if metric == "percent_bias":
                                                data = data[(data >= -100) & (data <= 100)]
                                            datasets_filtered.append(
                                                data[~np.isnan(data)]
                                            )  # Filter out NaNs and append
                                        finally:
                                            gc.collect()  # Clean up memory after processing each simulation source

                                    try:
                                        make_scenarios_comparison_Whisker_Plot(
                                            dir_path,
                                            evaluation_item,
                                            ref_source,
                                            sim_sources,
                                            metric,
                                            datasets_filtered,
                                            option,
                                        )
                                    except Exception:
                                        logging.exception(
                                            f"Error: {evaluation_item} {ref_source} {sim_sources} {metric} Whisker Plot failed!"
                                        )
                                        raise
                                finally:
                                    gc.collect()  # Clean up memory after processing each reference source
                        finally:
                            gc.collect()  # Clean up memory after processing each metric
                finally:
                    gc.collect()  # Clean up memory after processing each evaluation item
        finally:
            gc.collect()  # Final cleanup for the entire method

    def scenarios_Relative_Score_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        try:
            dir_path = os.path.join(casedir, "comparisons", "Relative_Score")
            os.makedirs(dir_path, exist_ok=True)

            for evaluation_item in evaluation_items:
                try:
                    sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
                    ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]

                    if isinstance(sim_sources, str):
                        sim_sources = [sim_sources]
                    if isinstance(ref_sources, str):
                        ref_sources = [ref_sources]

                    for ref_source in ref_sources:
                        for sim_source in sim_sources:
                            try:
                                ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                                sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]
                                if ref_data_type == "stn" or sim_data_type == "stn":
                                    relative_file = _require_only_drawing_file(
                                        _relative_station_scores_path(
                                            dir_path, evaluation_item, ref_source, sim_source
                                        ),
                                        producer="comparison",
                                        fallback_paths=[
                                            _legacy_relative_station_scores_path(
                                                dir_path, evaluation_item, ref_source, sim_source
                                            )
                                        ],
                                    )
                                    for score in scores:
                                        _require_csv_finite(
                                            relative_file,
                                            f"relative_{score}_{sim_source}",
                                            required_columns=["ID"],
                                        )
                                    try:
                                        make_scenarios_comparison_Relative_Score(
                                            dir_path,
                                            evaluation_item,
                                            ref_source,
                                            sim_source,
                                            scores,
                                            "stn",
                                            self.main_nml["general"],
                                            option,
                                        )
                                    except Exception:
                                        logging.exception("Relative Score only_drawing renderer failed")
                                        raise
                                else:
                                    relative_files = {}
                                    for score in scores:
                                        relative_files[score] = _require_only_drawing_file(
                                            _relative_grid_score_path(
                                                dir_path, evaluation_item, ref_source, sim_source, score
                                            ),
                                            producer="comparison",
                                            fallback_paths=[
                                                _legacy_relative_grid_score_path(
                                                    dir_path, evaluation_item, ref_source, sim_source, score
                                                )
                                            ],
                                        )
                                    for score, relative_file in relative_files.items():
                                        _require_netcdf_finite(relative_file, f"relative_{score}")
                                    try:
                                        make_scenarios_comparison_Relative_Score(
                                            dir_path,
                                            evaluation_item,
                                            ref_source,
                                            sim_source,
                                            scores,
                                            "grid",
                                            self.main_nml["general"],
                                            option,
                                        )
                                    except Exception:
                                        logging.exception("Relative Score only_drawing renderer failed")
                                        raise
                            finally:
                                gc.collect()  # Clean up memory after processing each simulation source
                finally:
                    gc.collect()  # Clean up memory after processing each evaluation item
        finally:
            gc.collect()  # Final cleanup for the entire method

    def scenarios_Single_Model_Performance_Index_comparison(
        self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option
    ):
        make_scenarios_comparison_Single_Model_Performance_Index(basedir, evaluation_items, ref_nml, sim_nml, option)

    def scenarios_Ridgeline_Plot_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        dir_path = os.path.join(f"{basedir}", "comparisons", "Ridgeline_Plot")
        # if os.path.exists(dir_path):
        #    shutil.rmtree(dir_path)
        # print(f"Re-creating output directory: {dir_path}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        for evaluation_item in evaluation_items:
            sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
            ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]
            # if the sim_sources and ref_sources are not list, then convert them to list
            if isinstance(sim_sources, str):
                sim_sources = [sim_sources]
            if isinstance(ref_sources, str):
                ref_sources = [ref_sources]
            for score in scores:
                # Skip nSpatialScore since it's a constant value
                if score == "nSpatialScore":
                    _skip_optional_only_drawing(figure="Ridgeline Plot", reason=f"{score} is a constant value")
                    continue
                for ref_source in ref_sources:
                    datasets_filtered = []
                    # create a numpy matrix to store the data
                    for sim_source in sim_sources:
                        ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                        sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]
                        if isinstance(sim_sources, str):
                            sim_sources = [sim_sources]
                        if isinstance(ref_sources, str):
                            ref_sources = [ref_sources]
                        # create a numpy matrix to store the data

                        if ref_data_type == "stn" or sim_data_type == "stn":
                            ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]
                            sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]
                            if sim_varname is None or sim_varname == "":
                                sim_varname = evaluation_item
                            if ref_varname is None or ref_varname == "":
                                ref_varname = evaluation_item
                            file_path = os.path.join(
                                basedir, "scores", f"{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv"
                            )
                            if not os.path.exists(file_path):
                                _require_only_drawing_file(file_path, producer="evaluation")
                            # read the file_path data and select the score
                            df = pd.read_csv(file_path, sep=",", header=0)
                            df = Convert_Type.convert_Frame(df)
                            data = df[score].values
                        else:
                            file_path = os.path.join(
                                basedir, "scores", f"{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{score}.nc"
                            )
                            if not os.path.exists(file_path):
                                _require_only_drawing_file(file_path, producer="evaluation")
                            file_path = _require_only_drawing_file(file_path, producer="evaluation")
                            with xr.open_dataset(file_path) as _ds:
                                ds = _ds.load()
                            ds = Convert_Type.convert_nc(ds)
                            data = ds[score].values
                        datasets_filtered.append(data[~np.isnan(data)])  # Filter out NaNs and append

                    try:
                        make_scenarios_comparison_Ridgeline_Plot(
                            dir_path, evaluation_item, ref_source, sim_sources, score, datasets_filtered, option
                        )
                    except Exception:
                        logging.exception(
                            f"Error: {evaluation_item} {ref_source} {sim_sources} {score} Ridgeline_Plot failed!"
                        )
                        raise

            for metric in metrics:
                for ref_source in ref_sources:
                    dir_path = os.path.join(f"{basedir}", "comparisons", "Ridgeline_Plot")
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)

                    datasets_filtered = []
                    # create a numpy matrix to store the data
                    for sim_source in sim_sources:
                        ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                        sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]
                        if isinstance(sim_sources, str):
                            sim_sources = [sim_sources]
                        if isinstance(ref_sources, str):
                            ref_sources = [ref_sources]
                        # create a numpy matrix to store the data
                        if ref_data_type == "stn" or sim_data_type == "stn":
                            ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]
                            sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]
                            if sim_varname is None or sim_varname == "":
                                sim_varname = evaluation_item
                            if ref_varname is None or ref_varname == "":
                                ref_varname = evaluation_item
                            file_path = os.path.join(
                                basedir, "metrics", f"{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv"
                            )
                            if not os.path.exists(file_path):
                                _require_only_drawing_file(file_path, producer="evaluation")
                            # read the file_path data and select the score
                            df = pd.read_csv(file_path, sep=",", header=0)
                            data = df[metric].values
                        else:
                            file_path = os.path.join(
                                basedir, "metrics", f"{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{metric}.nc"
                            )
                            if not os.path.exists(file_path):
                                _require_only_drawing_file(file_path, producer="evaluation")
                            file_path = _require_only_drawing_file(file_path, producer="evaluation")
                            with xr.open_dataset(file_path) as _ds:
                                ds = _ds.load()
                            ds = Convert_Type.convert_nc(ds)
                            data = ds[metric].values
                        data = data[~np.isinf(data)]
                        if metric == "percent_bias":
                            data = data[(data >= -100) & (data <= 100)]
                        datasets_filtered.append(data[~np.isnan(data)])  # Filter out NaNs and append

                    try:
                        make_scenarios_comparison_Ridgeline_Plot(
                            dir_path, evaluation_item, ref_source, sim_sources, metric, datasets_filtered, option
                        )
                    except Exception:
                        logging.exception(
                            f"Error: {evaluation_item} {ref_source} {sim_sources} {metric} Ridgeline_Plot failed!"
                        )
                        raise

    def to_dict(self):
        return self.__dict__

    from openbench.data.coordinates import COORDINATE_MAP_WITH_VERTICAL

    coordinate_map = dict(COORDINATE_MAP_WITH_VERTICAL)

    freq_map = {
        "month": "M",
        "mon": "M",
        "monthly": "M",
        "day": "D",
        "daily": "D",
        "hour": "H",
        "Hour": "H",
        "hr": "H",
        "Hr": "H",
        "h": "H",
        "hourly": "H",
        "year": "Y",
        "yr": "Y",
        "yearly": "Y",
        "week": "W",
        "wk": "W",
        "weekly": "W",
    }

    def scenarios_Diff_Plot_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        """
        Compare metrics and scores between different simulations:
        1. Calculate ensemble mean across all simulations
        2. Calculate anomalies from ensemble mean for each simulation
        3. Calculate pairwise differences between simulations
        4. Plot the results
        Parameters:
            basedir: base directory path
            sim_nml: simulation namelist
            ref_nml: reference namelist
            evaluation_items: list of evaluation items
            scores: list of scores to compare
            metrics: list of metrics to compare
            option: additional options
        """
        dir_path = os.path.join(f"{basedir}", "comparisons", "Diff_Plot")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        for evaluation_item in evaluation_items:
            # Get simulation sources
            sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
            ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]

            # Convert to lists if needed
            if isinstance(sim_sources, str):
                sim_sources = [sim_sources]
            if isinstance(ref_sources, str):
                ref_sources = [ref_sources]

            for ref_source in ref_sources:
                # Skip if only one simulation source

                # Check data types for all simulation sources
                data_types = []
                for sim_source in sim_sources:
                    sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]
                    data_types.append(sim_data_type)

                # Check if both 'stn' and grid data exist
                if "stn" in data_types and any(dt != "stn" for dt in data_types):
                    raise _unsupported_only_drawing(
                        figure="Diff Plot",
                        reason=(
                            f"{evaluation_item} mixes station and gridded simulations; "
                            "all simulation sources must use the same data type"
                        ),
                    )

                ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                if ref_data_type == "stn":
                    for item_type in [*metrics, *scores]:
                        for sim_source in sim_sources:
                            anomaly_path = _require_only_drawing_file(
                                _diff_station_anomaly_path(
                                    dir_path, evaluation_item, ref_source, sim_source, item_type
                                ),
                                producer="comparison",
                                fallback_paths=[
                                    _legacy_diff_station_anomaly_path(
                                        dir_path, evaluation_item, ref_source, sim_source, item_type
                                    )
                                ],
                            )
                            _require_csv_finite(
                                anomaly_path,
                                f"{item_type}_anomaly",
                                required_columns=["ID", "lat", "lon"],
                            )

                        if len(sim_sources) >= 2:
                            for i, sim1 in enumerate(sim_sources):
                                sim_varname_1 = sim_nml[f"{evaluation_item}"][f"{sim1}_varname"]
                                for sim2 in sim_sources[i + 1 :]:
                                    sim_varname_2 = sim_nml[f"{evaluation_item}"][f"{sim2}_varname"]
                                    diff_path = _require_only_drawing_file(
                                        _diff_station_difference_path(
                                            dir_path,
                                            evaluation_item,
                                            ref_source,
                                            sim1,
                                            sim_varname_1,
                                            sim2,
                                            sim_varname_2,
                                            item_type,
                                        ),
                                        producer="comparison",
                                        fallback_paths=[
                                            _legacy_diff_station_difference_path(
                                                dir_path,
                                                evaluation_item,
                                                ref_source,
                                                sim1,
                                                sim_varname_1,
                                                sim2,
                                                sim_varname_2,
                                                item_type,
                                            )
                                        ],
                                    )
                                    _require_csv_finite(
                                        diff_path,
                                        f"{item_type}_diff",
                                        required_columns=["ID", "lat", "lon"],
                                    )
                else:
                    for item_type in [*metrics, *scores]:
                        for sim_source in sim_sources:
                            anomaly_path = _require_only_drawing_file(
                                _diff_grid_anomaly_path(dir_path, evaluation_item, ref_source, sim_source, item_type),
                                producer="comparison",
                                fallback_paths=[
                                    _legacy_diff_grid_anomaly_path(
                                        dir_path, evaluation_item, ref_source, sim_source, item_type
                                    )
                                ],
                            )
                            _require_netcdf_finite(anomaly_path, f"{item_type}_anomaly")

                        if len(sim_sources) >= 2:
                            for i, sim1 in enumerate(sim_sources):
                                for sim2 in sim_sources[i + 1 :]:
                                    diff_path = _require_only_drawing_file(
                                        _diff_grid_difference_path(
                                            dir_path, evaluation_item, ref_source, sim1, sim2, item_type
                                        ),
                                        producer="comparison",
                                        fallback_paths=[
                                            _legacy_diff_grid_difference_path(
                                                dir_path, evaluation_item, ref_source, sim1, sim2, item_type
                                            )
                                        ],
                                    )
                                    _require_netcdf_finite(diff_path, f"{item_type}_diff")
                # After calculating anomalies for metrics
                make_scenarios_comparison_Diff_Plot(
                    dir_path,
                    metrics,
                    scores,
                    evaluation_item,
                    ref_source,
                    sim_sources,
                    self.general_config,
                    sim_nml,
                    ref_data_type,
                    option,
                )

    def scenarios_Basic_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        """
        Calculate all the data (including input data,metrics,scores):
        1. Calculate ensemble mean, median, min, max
        2. Calculate sum value for each input
        4. Plot the results
        """
        basic_method = option["key"]
        dir_path = os.path.join(f"{basedir}", "comparisons", basic_method)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        for evaluation_item in evaluation_items:
            # Get simulation sources
            sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
            ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]

            # Convert to lists if needed
            if isinstance(sim_sources, str):
                sim_sources = [sim_sources]
            if isinstance(ref_sources, str):
                ref_sources = [ref_sources]

            for ref_source in ref_sources:
                ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]

                if ref_data_type == "stn":
                    try:
                        for sim_source in sim_sources:
                            output_path = (
                                f"{dir_path}/{evaluation_item}_stn_{ref_source}_{sim_source}_{basic_method}.csv"
                            )
                            output_path = _require_only_drawing_file(output_path, producer="comparison")
                            if basic_method != "nSpatialScore":
                                _require_csv_finite(output_path, "ref_value", required_columns=["sim_value"])
                                _require_csv_finite(output_path, "sim_value", required_columns=["ref_value"])
                            make_stn_plot_index(
                                output_path, basic_method, self.main_nml["general"], (ref_source, sim_source), option
                            )
                    except Exception as e:
                        logging.error(f"Error processing station {basic_method} calculations for {ref_source}: {e}")
                        raise
                else:
                    try:
                        output_path = os.path.join(
                            dir_path, f"{evaluation_item}_ref_{ref_source}_{ref_varname}_{basic_method}.nc"
                        )
                        # Skip global map plotting for nSpatialScore since it's constant globally
                        if basic_method != "nSpatialScore":
                            _require_netcdf_finite(output_path, basic_method)
                            make_geo_plot_index(output_path, basic_method, self.main_nml["general"], option)
                        else:
                            _skip_optional_only_drawing(
                                figure="Basic", reason=f"{basic_method} is a constant global value"
                            )
                    except Exception as e:
                        logging.error(f"Error processing Grid {basic_method} calculations for {ref_source}: {e}")
                        raise

            for sim_source in sim_sources:
                if len(sim_sources) < 2:
                    _skip_unrequested_only_drawing(
                        figure="Basic", reason=f"{evaluation_item} has fewer than two simulation sources"
                    )
                    continue

                sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]
                sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]
                if sim_data_type != "stn":
                    try:
                        output_path = os.path.join(
                            dir_path, f"{evaluation_item}_sim_{sim_source}_{sim_varname}_{basic_method}.nc"
                        )
                        # Skip global map plotting for nSpatialScore since it's constant globally
                        if basic_method != "nSpatialScore":
                            _require_netcdf_finite(output_path, basic_method)
                            make_geo_plot_index(output_path, basic_method, self.main_nml["general"], option)
                        else:
                            _skip_optional_only_drawing(
                                figure="Basic", reason=f"{basic_method} is a constant global value"
                            )
                    except Exception as e:
                        logging.error(f"Error processing station {basic_method} calculations for {sim_source}: {e}")
                        raise

    def scenarios_Mann_Kendall_Trend_Test_comparison(
        self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option
    ):
        method_name = "Mann_Kendall_Trend_Test"
        getattr(self, f"stat_{method_name.lower()}", None)
        dir_path = os.path.join(basedir, "comparisons", "Mann_Kendall_Trend_Test")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self.compare_nml["Mann_Kendall_Trend_Test"] = {}
        self.compare_nml["Mann_Kendall_Trend_Test"]["significance_level"] = option["significance_level"]
        for evaluation_item in evaluation_items:
            # Get simulation sources
            sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
            ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]

            # Convert to lists if needed
            if isinstance(sim_sources, str):
                sim_sources = [sim_sources]
            if isinstance(ref_sources, str):
                ref_sources = [ref_sources]

            for sim_source in sim_sources:
                # Skip if only one simulation source

                sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]
                sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]

                if sim_data_type != "stn":
                    try:
                        output_file = os.path.join(
                            dir_path, f"Mann_Kendall_Trend_Test_{evaluation_item}_sim_{sim_source}_{sim_varname}.nc"
                        )
                        output_file = _require_netcdf_variables(
                            output_file,
                            ["tau", "trend", "p_value"],
                            finite_variables=["tau", "trend"],
                            producer="statistics",
                        )
                        make_Mann_Kendall_Trend_Test(
                            output_file, method_name, sim_source, self.main_nml["general"], option
                        )
                    except FileNotFoundError:
                        raise
                    except Exception as e:
                        logging.error(
                            f"Error processing {method_name} calculations for {evaluation_item} {sim_source}: {e}"
                        )
                        raise
            for ref_source in ref_sources:
                ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]
                if ref_data_type != "stn":
                    try:
                        output_file = os.path.join(
                            dir_path, f"Mann_Kendall_Trend_Test_{evaluation_item}_ref_{ref_source}_{ref_varname}.nc"
                        )
                        output_file = _require_netcdf_variables(
                            output_file,
                            ["tau", "trend", "p_value"],
                            finite_variables=["tau", "trend"],
                            producer="statistics",
                        )
                        make_Mann_Kendall_Trend_Test(
                            output_file, method_name, ref_source, self.main_nml["general"], option
                        )
                    except FileNotFoundError:
                        raise
                    except Exception as e:
                        logging.error(
                            f"Error processing {method_name} calculations for {evaluation_item} {ref_source}: {e}"
                        )
                        raise

    def scenarios_Standard_Deviation_comparison(
        self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option
    ):
        try:
            method_name = "Standard_Deviation"
            getattr(self, f"stat_{method_name.lower()}", None)
            dir_path = os.path.join(basedir, "comparisons", method_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            for evaluation_item in evaluation_items:
                # Get simulation sources
                sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
                ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]

                # Convert to lists if needed
                if isinstance(sim_sources, str):
                    sim_sources = [sim_sources]
                if isinstance(ref_sources, str):
                    ref_sources = [ref_sources]

                for sim_source in sim_sources:
                    try:
                        sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]
                        sim_varname = sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]

                        if sim_data_type != "stn":
                            output_file = os.path.join(
                                dir_path, f"{method_name}_{evaluation_item}_sim_{sim_source}_{sim_varname}.nc"
                            )

                            _require_netcdf_finite(output_file, method_name, producer="statistics")
                            make_Standard_Deviation(
                                output_file, method_name, sim_source, self.main_nml["general"], option
                            )
                        else:
                            logging.info(
                                f"Skipping {method_name} drawing for {evaluation_item} {sim_source}: station data type."
                            )
                    except FileNotFoundError:
                        raise
                    except Exception as e:
                        logging.error(
                            f"Error processing {method_name} calculations for {evaluation_item} {sim_source}: {e}"
                        )
                        raise
                    finally:
                        # Clean up memory after each simulation source
                        gc.collect()

                for ref_source in ref_sources:
                    try:
                        ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                        ref_varname = ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]

                        if ref_data_type != "stn":
                            output_file = os.path.join(
                                dir_path, f"{method_name}_{evaluation_item}_ref_{ref_source}_{ref_varname}.nc"
                            )

                            _require_netcdf_finite(output_file, method_name, producer="statistics")
                            make_Standard_Deviation(
                                output_file, method_name, ref_source, self.main_nml["general"], option
                            )
                        else:
                            logging.info(
                                f"Skipping {method_name} drawing for {evaluation_item} {ref_source}: station data type."
                            )
                    except FileNotFoundError:
                        raise
                    except Exception as e:
                        logging.error(
                            f"Error processing {method_name} calculations for {evaluation_item} {ref_source}: {e}"
                        )
                        raise
                    finally:
                        # Clean up memory after each reference source
                        gc.collect()
        finally:
            # Ensure memory is cleaned up after the entire process
            gc.collect()

    def scenarios_Functional_Response_comparison(
        self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option
    ):
        self.compare_nml["Functional_Response"] = {}
        self.compare_nml["Functional_Response"]["nbins"] = option["nbins"]
        try:
            method_name = "Functional_Response"
            getattr(self, f"stat_{method_name.lower()}", None)
            dir_path = os.path.join(basedir, "comparisons", method_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            for evaluation_item in evaluation_items:
                # Get simulation sources
                sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
                ref_sources = ref_nml["general"][f"{evaluation_item}_ref_source"]

                # Convert to lists if needed
                if isinstance(sim_sources, str):
                    sim_sources = [sim_sources]
                if isinstance(ref_sources, str):
                    ref_sources = [ref_sources]

                for ref_source in ref_sources:
                    try:
                        ref_data_type = ref_nml[f"{evaluation_item}"][f"{ref_source}_data_type"]
                        ref_nml[f"{evaluation_item}"][f"{ref_source}_varname"]

                        if ref_data_type != "stn":
                            for sim_source in sim_sources:
                                try:
                                    sim_data_type = sim_nml[f"{evaluation_item}"][f"{sim_source}_data_type"]
                                    sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]
                                    if sim_data_type != "stn":
                                        output_file = os.path.join(
                                            dir_path,
                                            f"{method_name}_{evaluation_item}_ref_{ref_source}_sim_{sim_source}.nc",
                                        )

                                        _require_netcdf_finite(
                                            output_file, "functional_response_score", producer="statistics"
                                        )
                                        make_Functional_Response(
                                            output_file, method_name, sim_source, self.main_nml["general"], option
                                        )
                                except FileNotFoundError:
                                    raise
                                except Exception as e:
                                    logging.error(
                                        f"Error processing {method_name} calculations for {evaluation_item} {ref_source} {sim_source}: {e}"
                                    )
                                    raise
                                finally:
                                    # Clean up memory after each simulation source
                                    gc.collect()
                    finally:
                        # Clean up memory after each reference source
                        gc.collect()
        finally:
            # Ensure memory is cleaned up after the entire process
            gc.collect()

    def scenarios_RadarMap_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        try:
            dir_path = os.path.join(casedir, "comparisons", "RadarMap")
            for score in scores:
                output_file_path = os.path.join(dir_path, f"scenarios_{score}_comparison.csv")
                output_file_path = _require_only_drawing_file(
                    output_file_path, producer="comparison", fallback_ext=".txt"
                )
                make_scenarios_comparison_radar_map(output_file_path, score, option)
        finally:
            gc.collect()  # Clean up memory after processing

    def scenarios_Correlation_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        try:
            method_name = "Correlation"
            getattr(self, f"stat_{method_name.lower()}", None)
            dir_path = os.path.join(basedir, "comparisons", method_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            for evaluation_item in evaluation_items:
                # Get simulation sources
                sim_sources = sim_nml["general"][f"{evaluation_item}_sim_source"]
                # Convert to lists if needed
                if isinstance(sim_sources, str):
                    sim_sources = [sim_sources]
                if len(sim_sources) < 2:
                    _skip_unrequested_only_drawing(
                        figure="Correlation", reason=f"{evaluation_item} has fewer than two simulation sources"
                    )
                    continue

                for i, sim1 in enumerate(sim_sources):
                    for j, sim2 in enumerate(sim_sources[i + 1 :], i + 1):
                        sim_data_type1 = sim_nml[f"{evaluation_item}"][f"{sim1}_data_type"]
                        sim_data_type2 = sim_nml[f"{evaluation_item}"][f"{sim2}_data_type"]
                        if sim_data_type1 == "stn" or sim_data_type2 == "stn":
                            raise _unsupported_only_drawing(
                                figure="Correlation",
                                reason=f"{evaluation_item} uses station data; all simulation sources must be gridded",
                            )

                        try:
                            output_file = os.path.join(
                                dir_path, f"{method_name}_{evaluation_item}_{sim1}_and_{sim2}.nc"
                            )
                            _require_netcdf_finite(output_file, method_name, producer="statistics")
                            make_Correlation(output_file, method_name, self.main_nml["general"], option)
                        except FileNotFoundError:
                            raise
                        except Exception as e:
                            logging.error(
                                f"Error processing {method_name} calculations for {evaluation_item} {sim1} and {sim2}: {e}"
                            )
                            raise
                        finally:
                            # Clean up memory after each iteration
                            gc.collect()
        finally:
            # Ensure memory is cleaned up after the entire process
            gc.collect()
