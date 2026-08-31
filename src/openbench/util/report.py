#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report Generation Module for OpenBench
Generates comprehensive HTML and PDF evaluation reports with tables, figures, and detailed analysis
"""

import glob
import json
import os
import shutil
from datetime import datetime
from itertools import product
from typing import Any, Dict, List, Optional
from urllib.parse import quote, unquote

import numpy as np
import pandas as pd
import xarray as xr
from jinja2 import Environment, select_autoescape

from openbench.core.registry import IMPLEMENTED_METRIC_NAMES, IMPLEMENTED_SCORE_NAMES
from openbench.util.filenames import filename_component, join_filename_components


# HTML escaping is enabled by default to prevent XSS in user-controlled
# fields (e.g. evaluation_item names, file paths). Use a single shared
# Environment so templates parsed from strings inherit autoescape.
def _url_path(path: str) -> str:
    """URL-encode each path segment without treating literal %2F in filenames as a slash."""
    return "/".join(quote(segment, safe="") for segment in str(path).split("/"))


_jinja_env = Environment(autoescape=select_autoescape(default=True, default_for_string=True))
_jinja_env.filters["url_path"] = _url_path

# Report statistics read at most this many values per chunk. Exact medians are
# retained only when the complete result fits within the same bound.
_MAX_REPORT_STAT_POINTS = 1_000_000
_REPORT_FIGURE_SUFFIXES = (".jpg", ".jpeg", ".png", ".svg", ".webp")


def _is_report_figure(path: str) -> bool:
    """Accept real figure files, not hidden filesystem metadata sidecars."""
    return not os.path.basename(path).startswith(".") and path.lower().endswith(_REPORT_FIGURE_SUFFIXES)


def _remove_report_tree(path: str) -> None:
    def ignore_disappeared_entries(_func, _path, exc_info):
        if not isinstance(exc_info[1], FileNotFoundError):
            raise exc_info[1]

    shutil.rmtree(path, onerror=ignore_disappeared_entries)


def _remove_appledouble_files(path: str) -> None:
    """Remove macOS metadata sidecars from the generated report package."""
    for root, _dirs, files in os.walk(path):
        for name in files:
            if name.startswith("._"):
                try:
                    os.unlink(os.path.join(root, name))
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    logging.getLogger(__name__).warning("Could not remove report metadata sidecar %s: %s", name, exc)


# Import PDF generation libraries
try:
    from xhtml2pdf import pisa

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

import logging

# Setup logger
logger = logging.getLogger(__name__)


def _dedupe_paths(paths: List[str]) -> List[str]:
    """Return paths in first-seen order without duplicates."""
    seen = set()
    unique = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _decode_filename_component(value: str) -> str:
    """Decode a component produced by filename_component()."""
    return unquote(value)


class ReportGenerator:
    """Generate comprehensive evaluation reports in HTML and PDF formats"""

    def __init__(self, config: Dict[str, Any], output_dir: str):
        """
        Initialize the report generator

        Args:
            config: Configuration dictionary
            output_dir: Base output directory
        """
        self.config = config
        self.output_dir = output_dir
        self.report_dir = os.path.join(output_dir, "reports")
        self.metrics_dir = os.path.join(output_dir, "metrics")
        self.scores_dir = os.path.join(output_dir, "scores")
        self.comparisons_dir = os.path.join(output_dir, "comparisons")
        self.data_dir = os.path.join(output_dir, "data")
        self.uncertainty_dir = os.path.join(output_dir, "uncertainty")

        # Create reports directory if it doesn't exist
        os.makedirs(self.report_dir, exist_ok=True)

        # Report metadata - only include enabled evaluation items
        enabled_items = []
        if isinstance(config.get("evaluation_items"), dict):
            enabled_items = [item for item, enabled in config["evaluation_items"].items() if enabled]
        elif isinstance(config.get("evaluation_items"), list):
            enabled_items = config["evaluation_items"]

        # Get enabled metrics and scores from configuration
        self.enabled_metrics = [metric for metric, enabled in config.get("metrics", {}).items() if enabled]
        self.enabled_scores = [score for score, enabled in config.get("scores", {}).items() if enabled]

        # Get enabled comparisons from configuration
        self.enabled_comparisons = [comp for comp, enabled in config.get("comparisons", {}).items() if enabled]

        self.metadata = {
            "title": "OpenBench Evaluation Report",
            "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config_file": config.get("config_file", "N/A"),
            "evaluation_items": enabled_items,
        }

        logger.info(f"Report generator initialized with output directory: {self.report_dir}")

    def generate_report(self, report_name: str = "evaluation_report") -> Dict[str, str]:
        """
        Generate both HTML and PDF reports

        Args:
            report_name: Base name for the report files

        Returns:
            Dictionary with paths to generated reports
        """
        logger.info("Starting report generation...")

        # Collect all data
        report_data = self._collect_report_data()

        # Copy all relevant figures to report directory first
        self._copy_figures_to_report_dir()

        # Verify figure paths before generating reports
        self._verify_figure_paths(report_data)

        # Generate HTML report
        html_path = self._generate_html_report(report_data, report_name)

        # Generate PDF report (after figures are copied)
        pdf_path = self._generate_pdf_report(html_path, report_name)
        _remove_appledouble_files(self.report_dir)

        logger.info("Report generation completed successfully")
        logger.info(f"HTML report: {html_path}")
        if pdf_path:
            logger.info(f"PDF report: {pdf_path}")

        result = {"html": html_path}
        if pdf_path:
            result["pdf"] = pdf_path

        return result

    def _collect_report_data(self) -> Dict[str, Any]:
        """Collect all data needed for the report"""
        logger.info("Collecting report data...")

        report_data = {
            "metadata": self.metadata,
            "evaluation_items": {},
            "overall_summary": {},
            "comparisons": {},
            "climate_zone_analysis": {},
            "uncertainty": {},
        }

        # Collect data for each evaluation item
        for item in self.metadata["evaluation_items"]:
            logger.info(f"Collecting data for {item}")
            item_data = {
                "metrics": self._collect_metrics_data(item),
                "scores": self._collect_scores_data(item),
                "figures": self._collect_figures(item),
                "statistics": self._collect_statistics(item),
            }

            # Debug logging
            logger.info(f"Collected data for {item}:")
            logger.info(f"  - Figures: {list(item_data['figures'].keys())}")
            logger.info(f"  - Statistics: {list(item_data['statistics'].keys())}")
            for fig_type, figs in item_data["figures"].items():
                if figs:
                    logger.info(f"    {fig_type}: {len(figs)} figures")

            report_data["evaluation_items"][item] = item_data

        # Collect comparison data
        report_data["comparisons"] = self._collect_comparison_data()

        # Collect overall summary
        report_data["overall_summary"] = self._generate_overall_summary(report_data)

        # Collect groupby analysis summary
        report_data["groupby_summary"] = self._generate_groupby_analysis_summary(report_data)
        report_data["uncertainty"] = self._collect_uncertainty_data()

        return report_data

    def _collect_uncertainty_data(self) -> Dict[str, Any]:
        """Load the machine-readable uncertainty summary when available."""
        path = os.path.join(self.uncertainty_dir, "summary.json")
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError) as exc:
            logger.warning("Could not read uncertainty summary %s: %s", path, exc)
            return {}

    def _collect_metrics_data(self, item: str) -> Dict[str, Any]:
        """Collect metrics data for a specific evaluation item"""
        metrics_data = {}

        # Look for CSV files with evaluation results
        csv_files = [
            path
            for path in self._item_output_files(self.metrics_dir, item, (".csv",))
            if os.path.basename(path).endswith(("_evaluations.csv", "__evaluations.csv"))
        ]

        for csv_file in csv_files:
            key = os.path.basename(csv_file).replace("_evaluations.csv", "")
            try:
                row_count, summary = self._summarize_csv(csv_file)
                summary = {
                    name: values
                    for name, values in summary.items()
                    if name in self.enabled_metrics or name in {"use_syear", "use_eyear"}
                }
                metrics_data[key] = {"row_count": row_count, "summary": summary}
            except Exception as e:
                logger.warning(f"Error reading {csv_file}: {e}")

        # Generate comprehensive grid vs grid metric statistics from NetCDF files.
        grid_grid_stats = self._generate_grid_vs_grid_stats(item, include_scores=False)
        if grid_grid_stats:
            metrics_data.update(grid_grid_stats)
            return metrics_data

        # Look for individual NetCDF files with spatial metrics
        nc_files = self._item_output_files(self.metrics_dir, item, (".nc", ".nc4"))

        for nc_file in nc_files:
            if "_evaluations" not in nc_file:  # Skip CSV-related files
                key = os.path.splitext(os.path.basename(nc_file))[0]
                metric_type = self._extract_metric_type(key)
                if metric_type == "Unknown":
                    continue
                try:
                    with xr.open_dataset(nc_file) as ds:
                        # Get the main data variable (skip coordinate variables)
                        data_vars = [var for var in ds.data_vars if var not in ds.coords]
                        if data_vars:
                            main_var = ds[data_vars[0]]
                            stats = self._summarize_data_array(main_var)

                            if stats:
                                # Try to extract comparison pair from filename first, then fallback to config
                                comparison_pair = self._extract_comparison_pair(key)

                                # If we can extract ref and sim sources from filename, try to get better name from config
                                if "ref_" in key and "sim_" in key:
                                    try:
                                        # Extract ref and sim sources from filename
                                        parts = key.split("_")
                                        ref_source = None
                                        sim_source = None

                                        for i, part in enumerate(parts):
                                            if part == "ref" and i + 1 < len(parts):
                                                ref_parts = []
                                                j = i + 1
                                                while j < len(parts) and parts[j] != "sim":
                                                    ref_parts.append(parts[j])
                                                    j += 1
                                                ref_source = "_".join(ref_parts)

                                            elif part == "sim" and i + 1 < len(parts):
                                                sim_parts = []
                                                j = i + 1
                                                while j < len(parts) and not self._is_metric_or_score(parts[j]):
                                                    sim_parts.append(parts[j])
                                                    j += 1
                                                sim_source = "_".join(sim_parts)
                                                break

                                        if ref_source and sim_source:
                                            comparison_pair = self._get_comparison_pair_from_config(
                                                item, ref_source, sim_source
                                            )
                                    except Exception as e:
                                        logger.warning(f"Error extracting sources from filename {key}: {e}")

                                metrics_data[key] = {
                                    "summary": {
                                        metric_type: {
                                            "mean": stats["mean"],
                                            "std": stats["std"],
                                            "min": stats["min"],
                                            "max": stats["max"],
                                        }
                                    },
                                    "global_mean": stats["mean"],
                                    "global_std": stats["std"],
                                    "global_min": stats["min"],
                                    "global_max": stats["max"],
                                    "global_median": stats["median"],
                                    "median_omitted": stats["median_omitted"],
                                    "valid_points": stats["valid_points"],
                                    "total_points": stats["total_points"],
                                    "data_coverage": stats["coverage"],
                                    "shape": str(main_var.dims),
                                    "metric_type": metric_type,
                                    "comparison_pair": comparison_pair,
                                }
                            else:
                                logger.warning(f"No valid data found in {nc_file}")
                        else:
                            logger.warning(f"No data variables found in {nc_file}")
                except Exception as e:
                    logger.warning(f"Error reading {nc_file}: {e}")

        return metrics_data

    @staticmethod
    def _summarize_data_array(main_var: xr.DataArray, max_points: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Calculate exact bounded-memory statistics, omitting only large-array medians."""
        max_points = max_points or _MAX_REPORT_STAT_POINTS
        total_points = int(main_var.size)
        if total_points == 0:
            return None

        remaining = max_points
        chunk_sizes = []
        for size in reversed(main_var.shape):
            chunk_size = min(int(size), remaining)
            chunk_sizes.append(max(1, chunk_size))
            remaining = max(1, remaining // max(1, chunk_size))
        chunk_sizes.reverse()

        count = 0
        mean = 0.0
        m2 = 0.0
        minimum = float("inf")
        maximum = float("-inf")
        median_parts = [] if total_points <= max_points else None

        ranges = [range(0, int(size), chunk) for size, chunk in zip(main_var.shape, chunk_sizes)]
        for starts in product(*ranges):
            indexers = {
                dim: slice(start, min(start + chunk, int(size)))
                for dim, size, chunk, start in zip(main_var.dims, main_var.shape, chunk_sizes, starts)
            }
            values = np.asarray(main_var.isel(indexers).values).ravel()
            valid = values[np.isfinite(values)]
            if not len(valid):
                continue

            chunk_count = int(len(valid))
            chunk_mean = float(np.mean(valid))
            chunk_m2 = float(np.sum((valid - chunk_mean) ** 2))
            new_count = count + chunk_count
            delta = chunk_mean - mean
            mean += delta * chunk_count / new_count
            m2 += chunk_m2 + delta * delta * count * chunk_count / new_count
            count = new_count
            minimum = min(minimum, float(np.min(valid)))
            maximum = max(maximum, float(np.max(valid)))
            if median_parts is not None:
                median_parts.append(valid)

        if not count:
            return None

        return {
            "mean": mean,
            "std": float(np.sqrt(m2 / count)),
            "min": minimum,
            "max": maximum,
            "median": float(np.median(np.concatenate(median_parts))) if median_parts is not None else None,
            "median_omitted": median_parts is None,
            "coverage": count / total_points * 100,
            "valid_points": count,
            "total_points": total_points,
        }

    @staticmethod
    def _summarize_csv(csv_file: str, chunksize: int = 100_000) -> tuple[int, Dict[str, Any]]:
        """Return row count and numeric summaries without retaining the full CSV."""
        excluded = {
            "sim_lon",
            "sim_lat",
            "ref_lon",
            "ref_lat",
            "sim_syear",
            "sim_eyear",
            "ref_syear",
            "ref_eyear",
        }
        row_count = 0
        states: Dict[str, Dict[str, float | int]] = {}

        for chunk in pd.read_csv(csv_file, chunksize=chunksize):
            row_count += len(chunk)
            for column in chunk.columns:
                if column in excluded:
                    continue
                values = pd.to_numeric(chunk[column], errors="coerce").to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if not len(values):
                    continue

                chunk_count = int(len(values))
                chunk_mean = float(np.mean(values))
                chunk_m2 = float(np.sum((values - chunk_mean) ** 2))
                state = states.setdefault(
                    column,
                    {
                        "count": 0,
                        "mean": 0.0,
                        "m2": 0.0,
                        "min": float("inf"),
                        "max": float("-inf"),
                    },
                )
                old_count = int(state["count"])
                new_count = old_count + chunk_count
                delta = chunk_mean - float(state["mean"])
                state["mean"] = float(state["mean"]) + delta * chunk_count / new_count
                state["m2"] = float(state["m2"]) + chunk_m2 + delta * delta * old_count * chunk_count / new_count
                state["count"] = new_count
                state["min"] = min(float(state["min"]), float(np.min(values)))
                state["max"] = max(float(state["max"]), float(np.max(values)))

        summary = {}
        for column, state in states.items():
            count = int(state["count"])
            summary[column] = {
                "mean": float(state["mean"]),
                "std": float(np.sqrt(float(state["m2"]) / (count - 1))) if count > 1 else np.nan,
                "min": float(state["min"]),
                "max": float(state["max"]),
                "count": count,
            }
        return row_count, summary

    def _collect_scores_data(self, item: str) -> Dict[str, Any]:
        """Collect scores data for a specific evaluation item"""
        scores_data = {}

        # Similar to metrics collection
        csv_files = [
            path
            for path in self._item_output_files(self.scores_dir, item, (".csv",))
            if os.path.basename(path).endswith(("_evaluations.csv", "__evaluations.csv"))
        ]

        for csv_file in csv_files:
            key = os.path.basename(csv_file).replace("_evaluations.csv", "")
            try:
                row_count, summary = self._summarize_csv(csv_file)
                summary = {name: values for name, values in summary.items() if name in self.enabled_scores}
                scores_data[key] = {"row_count": row_count, "summary": summary}
            except Exception as e:
                logger.warning(f"Error reading {csv_file}: {e}")

        grid_grid_stats = self._generate_grid_vs_grid_stats(item, include_metrics=False)
        if grid_grid_stats:
            scores_data.update(grid_grid_stats)

        return scores_data

    def _collect_figures(self, item: str) -> Dict[str, List[str]]:
        """Collect all figures related to an evaluation item"""
        figures = {
            "metrics": [],
            "scores": [],
            "comparisons": [],
            "igbp_groupby": [],
            "pft_groupby": [],
            "climate_zone_groupby": [],
            "station_timeseries": [],
        }

        # Metrics/scores figures may use either legacy ``<item>_...`` names or
        # safe component-joined names. Avoid interpolating item into the glob so
        # items containing path separators cannot alter the search path.
        figures["metrics"] = [
            os.path.basename(f)
            for f in self._dir_files(self.metrics_dir, _REPORT_FIGURE_SUFFIXES)
            if self._filename_matches_item(f, item)
        ]

        figures["scores"] = [
            os.path.basename(f)
            for f in self._dir_files(self.scores_dir, _REPORT_FIGURE_SUFFIXES)
            if self._filename_matches_item(f, item)
        ]

        # Comparison figures (from various subdirectories)
        comparison_dirs = [
            "Taylor_Diagram",
            "Target_Diagram",
            "Whisker_Plot",
            "Ridgeline_Plot",
            "Kernel_Density_Estimate",
            "Parallel_Coordinates",
            "HeatMap",
            "RadarMap",
            "Relative_Score",
            "Diff_Plot",
            "Single_Model_Performance_Index",
            "Correlation",
            "Functional_Response",
            "Standard_Deviation",
            "Mann_Kendall_Trend_Test",
            "Portrait_Plot_seasonal",
        ]

        for comp_dir in comparison_dirs:
            if comp_dir in ["HeatMap", "RadarMap"]:
                # Aggregate score comparisons are rendered once in Overall Comparison.
                continue
            comp_path = os.path.join(self.comparisons_dir, comp_dir)
            # Avoid interpolating item into a glob: item names can contain
            # filesystem/glob metacharacters and similar prefixes must not collide.
            comp_files = [
                path
                for path in self._dir_files(comp_path, _REPORT_FIGURE_SUFFIXES)
                if self._comparison_figure_matches_item(path, comp_dir, item)
            ]
            figures["comparisons"].extend([f"{comp_dir}/{os.path.basename(f)}" for f in comp_files])

        figures["station_timeseries"] = self._collect_station_timeseries_figures(item)

        # IGBP groupby figures - now primarily in comparisons directory
        igbp_files = self._collect_groupby_figure_files(["IGBP_groupby"], item)

        # Format paths relative to the base directory
        figures["igbp_groupby"] = []
        for f in igbp_files:
            if self.comparisons_dir in f:
                rel_path = os.path.relpath(f, self.comparisons_dir).replace(os.sep, "/")
                figures["igbp_groupby"].append(f"comparisons/{rel_path}")

        if figures["igbp_groupby"]:
            logger.info(f"Found IGBP groupby figures: {figures['igbp_groupby']}")

        # PFT groupby figures - now primarily in comparisons directory
        pft_files = self._collect_groupby_figure_files(["PFT_groupby"], item)

        # Format paths relative to the base directory
        figures["pft_groupby"] = []
        for f in pft_files:
            if self.comparisons_dir in f:
                rel_path = os.path.relpath(f, self.comparisons_dir).replace(os.sep, "/")
                figures["pft_groupby"].append(f"comparisons/{rel_path}")

        if figures["pft_groupby"]:
            logger.info(f"Found PFT groupby figures: {figures['pft_groupby']}")

        # Climate zone groupby figures - now primarily in comparisons directory
        climate_files = self._collect_groupby_figure_files(["Climate_zone_groupby", "CZ_groupby"], item)

        # Format paths relative to the base directory
        figures["climate_zone_groupby"] = []
        for f in climate_files:
            if self.comparisons_dir in f:
                rel_path = os.path.relpath(f, self.comparisons_dir).replace(os.sep, "/")
                figures["climate_zone_groupby"].append(f"comparisons/{rel_path}")

        if figures["climate_zone_groupby"]:
            logger.info(f"Found Climate zone groupby figures: {figures['climate_zone_groupby']}")

        return figures

    def _filename_matches_item(self, path: str, item: str) -> bool:
        """Return whether a generated filename belongs to item under safe or legacy naming.

        Safe names use ``__`` component boundaries. Legacy names use ``<item>_``
        prefixes. Substring matching is intentionally avoided so ``Run`` cannot
        accidentally collect ``Runoff`` outputs.
        """
        stem = os.path.splitext(os.path.basename(path))[0]
        safe_item = filename_component(item)
        if stem == safe_item or stem.startswith(f"{safe_item}__"):
            return True
        if not stem.startswith(f"{item}_"):
            return False

        # Legacy names use "_" as both component separator and a legal
        # character inside item names. If the current item is a prefix of a
        # longer configured item, avoid attaching that longer item's legacy
        # output to the shorter one (e.g. Run vs Run_off).
        for other in self.metadata.get("evaluation_items", []):
            if other != item and len(str(other)) > len(str(item)) and stem.startswith(f"{other}_"):
                return False
        return True

    def _dir_files(self, directory: str, suffixes: tuple[str, ...]) -> List[str]:
        """List direct child files with suffixes without user-controlled glob patterns."""
        if not os.path.isdir(directory):
            return []
        matches: List[str] = []
        lowered_suffixes = tuple(s.lower() for s in suffixes)
        try:
            for name in sorted(os.listdir(directory)):
                path = os.path.join(directory, name)
                if os.path.isfile(path) and not name.startswith(".") and name.lower().endswith(lowered_suffixes):
                    matches.append(path)
        except OSError as exc:
            logger.warning("Could not list report directory %s: %s", directory, exc)
        return matches

    def _item_output_files(self, directory: str, item: str, suffixes: tuple[str, ...]) -> List[str]:
        """Return files that belong to an item under safe or legacy naming."""
        return [path for path in self._dir_files(directory, suffixes) if self._filename_matches_item(path, item)]

    def _filename_matches_pair(self, path: str, item: str, ref_source: str, sim_source: str) -> bool:
        """Return whether a metric/score filename belongs to an item/ref/sim pair."""
        stem = os.path.splitext(os.path.basename(path))[0]
        safe_prefix = join_filename_components(item, "ref", ref_source, "sim", sim_source)
        legacy_prefix = f"{item}_ref_{ref_source}_sim_{sim_source}_"
        return stem == safe_prefix or stem.startswith(f"{safe_prefix}__") or stem.startswith(legacy_prefix)

    def _filename_matches_data_role(self, path: str, item: str, role: str) -> bool:
        """Return whether a data filename belongs to item and role (ref/sim)."""
        stem = os.path.splitext(os.path.basename(path))[0]
        safe_prefix = join_filename_components(item, role)
        legacy_prefix = f"{item}_{role}_"
        return stem == safe_prefix or stem.startswith(f"{safe_prefix}__") or stem.startswith(legacy_prefix)

    def _comparison_figure_matches_item(self, path: str, comp_dir: str, item: str) -> bool:
        """Return whether a comparison figure belongs to this item."""
        if self._filename_matches_item(path, item):
            return True
        stem = os.path.splitext(os.path.basename(path))[0]
        safe_prefix = join_filename_components(comp_dir, item)
        if stem == safe_prefix or stem.startswith(f"{safe_prefix}__"):
            return True
        legacy_prefix = f"{comp_dir}_{item}_"
        if not stem.startswith(legacy_prefix):
            return False
        for other in self.metadata.get("evaluation_items", []):
            if other != item and len(str(other)) > len(str(item)) and stem.startswith(f"{comp_dir}_{other}_"):
                return False
        return True

    def _collect_station_timeseries_figures(self, item: str) -> List[str]:
        """Collect station timeseries figures under data/stn_* directories for this item."""
        if not os.path.isdir(self.data_dir):
            return []
        matches: List[str] = []
        try:
            station_dirs = [
                name
                for name in sorted(os.listdir(self.data_dir))
                if name.startswith("stn_") and os.path.isdir(os.path.join(self.data_dir, name))
            ]
        except OSError as exc:
            logger.warning("Could not list report data directory %s: %s", self.data_dir, exc)
            return []

        aliases = self._station_timeseries_aliases(item)
        for station_dir in station_dirs:
            if not self._station_dir_matches_item(station_dir, item):
                continue
            root_dir = os.path.join(self.data_dir, station_dir)
            for root, dirs, files in os.walk(root_dir):
                dirs.sort()
                for file in sorted(files):
                    if not _is_report_figure(file):
                        continue
                    stem = os.path.splitext(file)[0]
                    if not (stem.endswith("_timeseries") or stem.endswith("__timeseries")):
                        continue
                    if self._station_timeseries_matches_item(file, item, aliases):
                        rel_path = os.path.relpath(os.path.join(root, file), self.data_dir).replace(os.sep, "/")
                        matches.append(f"data/{rel_path}")
        return _dedupe_paths(matches)

    def _station_dir_matches_item(self, station_dir: str, item: str) -> bool:
        """Return whether a stn_<ref>_<sim> directory can belong to this item."""
        refs = self._get_reference_sources(item)
        sims = self._get_simulation_sources(item)
        if not refs or not sims:
            return True
        for ref_source in refs:
            for sim_source in sims:
                if station_dir == join_filename_components("stn", ref_source, sim_source):
                    return True
                if station_dir == f"stn_{ref_source}_{sim_source}":
                    return True
        return False

    def _station_timeseries_matches_item(self, filename: str, item: str, aliases: List[str]) -> bool:
        """Match station timeseries by item name or configured raw varname."""
        if self._filename_matches_item(filename, item):
            return True
        stem = os.path.splitext(os.path.basename(filename))[0]
        for alias in aliases:
            safe_alias = filename_component(alias)
            if stem == safe_alias or stem.startswith(f"{safe_alias}__"):
                return True
            if stem.startswith(f"{alias}_"):
                return True
        return False

    def _station_timeseries_aliases(self, item: str) -> List[str]:
        """Configured raw variable names that may prefix station timeseries files."""
        aliases: List[str] = []
        for nml_name, sources in (
            ("ref_nml", self._get_reference_sources(item)),
            ("sim_nml", self._get_simulation_sources(item)),
        ):
            section = self.config.get(nml_name, {}).get(item, {})
            if not isinstance(section, dict):
                continue
            for source in sources:
                varname = section.get(f"{source}_varname")
                if isinstance(varname, str) and varname and varname != item:
                    aliases.append(varname)
                elif isinstance(varname, list):
                    aliases.extend(str(name) for name in varname if str(name) and str(name) != item)
                for fallback in section.get(f"{source}_fallbacks", []) or []:
                    if isinstance(fallback, dict):
                        fallback_name = fallback.get("varname")
                        if isinstance(fallback_name, str) and fallback_name and fallback_name != item:
                            aliases.append(fallback_name)
            for key in ("varname", "stn_varname"):
                varname = section.get(key)
                if isinstance(varname, str) and varname and varname != item:
                    aliases.append(varname)
        return _dedupe_paths(aliases)

    def _collect_groupby_figure_files(self, groupby_dirs: List[str], item: str) -> List[str]:
        """Collect groupby figures recursively under safe and legacy pair directories."""
        matches: List[str] = []
        for groupby_dir in groupby_dirs:
            groupby_path = os.path.join(self.comparisons_dir, groupby_dir)
            if not os.path.exists(groupby_path):
                continue
            for root, _dirs, files in os.walk(groupby_path):
                for file in files:
                    if _is_report_figure(file) and self._filename_matches_item(file, item):
                        matches.append(os.path.join(root, file))
        return _dedupe_paths(matches)

    def _collect_statistics(self, item: str) -> Dict[str, Any]:
        """Collect statistical analysis results"""
        stats = {}

        # Check for statistical outputs in comparisons directory
        stat_dirs = ["Mean", "Median", "Min", "Max", "Standard_Deviation", "Mann_Kendall_Trend_Test"]

        for stat_dir in stat_dirs:
            stat_files = self._item_output_files(os.path.join(self.comparisons_dir, stat_dir), item, (".nc", ".nc4"))

            if stat_files:
                stats[stat_dir] = [os.path.basename(f) for f in stat_files]

        # Collect groupby statistics
        groupby_stats = self._collect_groupby_statistics(item)
        if groupby_stats:
            stats.update(groupby_stats)

        return stats

    def _collect_groupby_statistics(self, item: str) -> Dict[str, Any]:
        """Collect groupby statistics for IGBP, PFT, and Climate zone"""
        groupby_stats = {}

        # Define groupby types and their corresponding directories
        groupby_types = {
            "IGBP_groupby": ["IGBP_groupby"],
            "PFT_groupby": ["PFT_groupby"],
            "Climate_zone_groupby": ["Climate_zone_groupby", "CZ_groupby"],
        }

        for groupby_name, groupby_dirs in groupby_types.items():
            # Check in metrics, scores, and comparisons directories
            csv_files = []
            nc_files = []
            txt_files = []

            for groupby_dir in groupby_dirs:
                for base_dir in [self.metrics_dir, self.scores_dir, self.comparisons_dir]:
                    groupby_path = os.path.join(base_dir, groupby_dir)
                    if not os.path.exists(groupby_path):
                        continue

                    logger.info(f"Checking groupby directory: {groupby_path}")

                    for root, _dirs, files in os.walk(groupby_path):
                        for file in files:
                            path = os.path.join(root, file)
                            if file.endswith(".txt") and self._filename_matches_item(file, item):
                                txt_files.append(path)
                            elif file.endswith(".csv") and self._filename_matches_item(file, item):
                                csv_files.append(path)
                            elif file.endswith((".nc", ".nc4")) and self._filename_matches_item(file, item):
                                nc_files.append(path)

            txt_files = _dedupe_paths(txt_files)
            csv_files = _dedupe_paths(csv_files)
            nc_files = _dedupe_paths(nc_files)

            # Process txt files if found
            if txt_files:
                stats_data = []
                for txt_file in txt_files:
                    try:
                        # Read txt file and parse it
                        with open(txt_file, "r") as f:
                            content = f.read()
                        stats_data.append(
                            {
                                "file": os.path.basename(txt_file),
                                "content": content,
                                "summary": {"groups": self._extract_groups_from_txt(content)},
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Error reading {txt_file}: {e}")

                if stats_data:
                    groupby_stats[groupby_name] = {
                        "statistics": stats_data,
                        "description": self._get_groupby_description(groupby_name),
                    }

            # Process CSV files if found
            elif csv_files:
                stats_data = []
                for csv_file in csv_files:
                    try:
                        df = pd.read_csv(csv_file, sep=None, engine="python")
                        stats_data.append(
                            {
                                "file": os.path.basename(csv_file),
                                "row_count": int(len(df)),
                                "summary": self._generate_groupby_summary(df),
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Error reading {csv_file}: {e}")

                if stats_data:
                    if groupby_name not in groupby_stats:
                        groupby_stats[groupby_name] = {
                            "statistics": stats_data,
                            "description": self._get_groupby_description(groupby_name),
                        }
                    else:
                        groupby_stats[groupby_name]["statistics"].extend(stats_data)

            # Add spatial files info if found
            if nc_files:
                if groupby_name not in groupby_stats:
                    groupby_stats[groupby_name] = {
                        "statistics": [],
                        "description": self._get_groupby_description(groupby_name),
                    }

                groupby_stats[groupby_name]["spatial_files"] = [os.path.basename(f) for f in nc_files]

        return groupby_stats

    def _generate_groupby_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for groupby dataframe"""
        summary = {}

        # Identify numeric columns for statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if not df[col].isna().all():
                summary[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "median": float(df[col].median()),
                    "count": int(df[col].count()),
                }

        # Try to extract group information - check various possible column names
        group_columns = [
            "Group",
            "group",
            "IGBP",
            "igbp",
            "PFT",
            "pft",
            "Climate_Zone",
            "climate_zone",
            "Zone",
            "zone",
            "Type",
            "type",
            "Class",
            "class",
            "Category",
            "category",
        ]

        for col in group_columns:
            if col in df.columns:
                summary["groups"] = df[col].unique().tolist()
                summary["group_count"] = len(summary["groups"])
                summary["group_column"] = col

                # Add performance ranking if metrics are available
                if len(numeric_cols) > 0:
                    # Find the best and worst performing groups
                    metric_col = numeric_cols[0]  # Use first numeric column
                    group_performance = df.groupby(col)[metric_col].mean().sort_values()
                    summary["worst_performing_groups"] = group_performance.head(3).index.tolist()
                    summary["best_performing_groups"] = group_performance.tail(3).index.tolist()
                break

        return summary

    def _extract_groups_from_txt(self, content: str) -> List[str]:
        """Extract group names from txt file content"""
        groups = []
        # Look for lines that might contain group names
        lines = content.split("\n")
        for line in lines:
            # Common patterns for group names in txt files
            if "IGBP_" in line or "PFT_" in line or "CZ_" in line:
                # Extract the group name
                parts = line.split()
                for part in parts:
                    if "IGBP_" in part or "PFT_" in part or "CZ_" in part:
                        groups.append(part)
        return list(set(groups))  # Return unique groups

    def _get_groupby_description(self, groupby_name: str) -> str:
        """Get description for groupby analysis type"""
        descriptions = {
            "IGBP_groupby": "Analysis grouped by International Geosphere-Biosphere Programme (IGBP) land cover classification. This analysis evaluates model performance across different land cover types including forests, grasslands, croplands, and urban areas, providing insights into ecosystem-specific model behaviors.",
            "PFT_groupby": "Analysis grouped by Plant Functional Types (PFTs). This classification groups vegetation based on physiological and morphological characteristics, allowing assessment of model performance for different plant strategies such as evergreen vs. deciduous, C3 vs. C4 photosynthesis, and various growth forms.",
            "Climate_zone_groupby": "Analysis grouped by Köppen-Geiger climate zones. This classification divides the global land surface based on temperature and precipitation patterns, enabling evaluation of model performance under different climatic conditions from tropical to polar regions.",
        }
        return descriptions.get(groupby_name, f"{groupby_name} analysis")

    def _collect_comparison_data(self) -> Dict[str, Any]:
        """Collect overall comparison data based on enabled comparisons."""
        comparison_data = {}

        if not self.config.get("general", {}).get("comparison", True):
            logger.info("Comparison is disabled in configuration, skipping comparison data collection")
            return comparison_data

        figures = {}
        score = self.enabled_scores[0] if self.enabled_scores else "Overall_Score"
        comparison_mappings = {
            "HeatMap": {
                "data_files": [
                    os.path.join(self.comparisons_dir, "HeatMap", f"scenarios_{s}_comparison.csv")
                    for s in self.enabled_scores
                ]
                or [os.path.join(self.comparisons_dir, "HeatMap", "scenarios_Overall_Score_comparison.csv")],
                "figure_patterns": [f"*{score}*heatmap.*", "*heatmap.*"],
                "data_key": "heatmap",
            },
            "Parallel_Coordinates": {
                "data_files": [
                    os.path.join(self.comparisons_dir, "Parallel_Coordinates", "Parallel_Coordinates_evaluations.csv")
                ],
                "figure_patterns": [f"*{score}*.*", "*.jpg", "*.jpeg", "*.png", "*.svg", "*.webp"],
                "data_key": "parallel_coords",
            },
            "RadarMap": {
                "data_files": [
                    os.path.join(self.comparisons_dir, "RadarMap", f"scenarios_{s}_comparison.csv")
                    for s in self.enabled_scores
                ]
                or [os.path.join(self.comparisons_dir, "RadarMap", "scenarios_Overall_Score_comparison.csv")],
                "figure_patterns": [f"*{score}*radarmap.*", "*radarmap.*"],
                "data_key": "radar",
            },
        }

        for comp_type, mapping in comparison_mappings.items():
            if comp_type not in self.enabled_comparisons:
                continue

            for data_file in mapping["data_files"]:
                if os.path.exists(data_file):
                    try:
                        comparison_data[mapping["data_key"]] = pd.read_csv(
                            data_file, sep=None, engine="python"
                        ).to_dict(orient="records")
                    except Exception as e:
                        logger.warning(f"Error reading {data_file}: {e}")
                    break

            for pattern in mapping["figure_patterns"]:
                figure_path = self._find_figure(self.comparisons_dir, comp_type, pattern)
                if figure_path:
                    figures[mapping["data_key"]] = figure_path
                    break

        if figures:
            comparison_data["figures"] = figures
        if comparison_data:
            comparison_data["score"] = score

        return comparison_data

    def _generate_metrics_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for metrics dataframe"""
        summary = {}

        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col not in [
                "sim_lon",
                "sim_lat",
                "ref_lon",
                "ref_lat",
                "sim_syear",
                "sim_eyear",
                "ref_syear",
                "ref_eyear",
            ]:
                summary[col] = {
                    "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                    "std": float(df[col].std()) if not df[col].isna().all() else None,
                    "min": float(df[col].min()) if not df[col].isna().all() else None,
                    "max": float(df[col].max()) if not df[col].isna().all() else None,
                    "count": int(df[col].count()),
                }

        return summary

    def _generate_scores_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for scores dataframe"""
        # Similar to metrics summary
        return self._generate_metrics_summary(df)

    def _generate_overall_summary(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall summary of the evaluation"""
        summary = {
            "total_items": len(report_data["evaluation_items"]),
            "items": list(report_data["evaluation_items"].keys()),
            "overall_scores": {},
        }

        # Calculate average scores across all items
        for item, item_data in report_data["evaluation_items"].items():
            for score_key, score_data in item_data.get("scores", {}).items():
                score_names = ["Overall_Score"] if "Overall_Score" in self.enabled_scores else self.enabled_scores
                for score_name in score_names:
                    score_mean = None
                    if "summary" in score_data and score_name in score_data["summary"]:
                        score_mean = score_data["summary"][score_name].get("mean")
                    elif score_data.get("station_format") and score_name in score_data.get("metrics", {}):
                        score_mean = score_data["metrics"][score_name].get("mean")
                    if score_mean is not None and np.isfinite(score_mean):
                        summary["overall_scores"][f"{item}_{score_key}"] = score_mean
                        break

        # Calculate grand average if scores exist
        if summary["overall_scores"]:
            summary["grand_average"] = np.mean(list(summary["overall_scores"].values()))

        return summary

    def _generate_groupby_analysis_summary(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of all groupby analyses across evaluation items"""
        summary = {
            "has_igbp": False,
            "has_pft": False,
            "has_climate_zone": False,
            "igbp_items": [],
            "pft_items": [],
            "climate_zone_items": [],
            "total_groupby_analyses": 0,
        }

        # Check each evaluation item for groupby analyses
        for item, item_data in report_data.get("evaluation_items", {}).items():
            if item_data.get("figures", {}).get("igbp_groupby") or item_data.get("statistics", {}).get("IGBP_groupby"):
                summary["has_igbp"] = True
                summary["igbp_items"].append(item)
                summary["total_groupby_analyses"] += 1

            if item_data.get("figures", {}).get("pft_groupby") or item_data.get("statistics", {}).get("PFT_groupby"):
                summary["has_pft"] = True
                summary["pft_items"].append(item)
                summary["total_groupby_analyses"] += 1

            if item_data.get("figures", {}).get("climate_zone_groupby") or item_data.get("statistics", {}).get(
                "Climate_zone_groupby"
            ):
                summary["has_climate_zone"] = True
                summary["climate_zone_items"].append(item)
                summary["total_groupby_analyses"] += 1

        # Generate summary messages
        if summary["has_igbp"]:
            summary["igbp_message"] = f"IGBP land cover analysis performed for: {', '.join(summary['igbp_items'])}"
        if summary["has_pft"]:
            summary["pft_message"] = f"PFT analysis performed for: {', '.join(summary['pft_items'])}"
        if summary["has_climate_zone"]:
            summary["climate_zone_message"] = (
                f"Climate zone analysis performed for: {', '.join(summary['climate_zone_items'])}"
            )

        return summary

    def _extract_metric_type(self, filename: str) -> str:
        """Extract the exact metric/score component from a generated filename."""
        stem = os.path.splitext(os.path.basename(filename))[0]
        enabled = set(self.enabled_metrics + self.enabled_scores)
        candidates = sorted(set(IMPLEMENTED_METRIC_NAMES + IMPLEMENTED_SCORE_NAMES) | enabled, key=len, reverse=True)

        parts = stem.split("__")
        if len(parts) > 1:
            last = _decode_filename_component(parts[-1])
            return last if last in enabled else "Unknown"

        for name in candidates:
            if stem == name or stem.endswith(f"_{name}"):
                return name if name in enabled else "Unknown"
        return "Unknown"

    def _is_metric_or_score(self, text: str) -> bool:
        """Check if text is a metric or score name"""
        return text in self.enabled_metrics or text in self.enabled_scores

    def _extract_comparison_pair(self, filename: str) -> str:
        """Extract comparison pair from filename (reference vs simulation)"""
        # Extract reference and simulation sources from filename
        # Format: ItemName_ref_RefSource_sim_SimSource_Metric
        parts = filename.split("_")

        ref_source = "Unknown"
        sim_source = "Unknown"

        for i, part in enumerate(parts):
            if part == "ref" and i + 1 < len(parts):
                # Find continuous reference name (may contain underscores)
                ref_parts = []
                j = i + 1
                while j < len(parts) and parts[j] != "sim":
                    ref_parts.append(parts[j])
                    j += 1
                ref_source = "_".join(ref_parts)

            elif part == "sim" and i + 1 < len(parts):
                # Find continuous simulation name (may contain underscores)
                sim_parts = []
                j = i + 1
                while j < len(parts) and not self._is_metric_or_score(parts[j]):
                    sim_parts.append(parts[j])
                    j += 1
                sim_source = "_".join(sim_parts)
                break

        return f"{ref_source} vs {sim_source}"

    def _get_comparison_pair_from_config(self, item: str, ref_source: str, sim_source: str) -> str:
        """Get comparison pair from configuration instead of filename parsing"""
        try:
            # Get display names from configuration if available
            ref_display_name = self._get_source_display_name(item, ref_source, "ref")
            sim_display_name = self._get_source_display_name(item, sim_source, "sim")

            return f"{ref_display_name} vs {sim_display_name}"

        except Exception as e:
            logger.warning(f"Error getting comparison pair from config for {item}: {e}")
            return f"{ref_source} vs {sim_source}"

    def _get_source_display_name(self, item: str, source: str, source_type: str) -> str:
        """Get display name for a source from configuration"""
        try:
            config_key = f"{source_type}_nml"
            if config_key in self.config and item in self.config[config_key]:
                # Try to get a display name or description
                display_key = f"{source}_display_name"
                if display_key in self.config[config_key][item]:
                    return self.config[config_key][item][display_key]

                # Try to get from varname as display name
                varname_key = f"{source}_varname"
                if varname_key in self.config[config_key][item]:
                    return self.config[config_key][item][varname_key]

            # If no display name found, return the source name
            return source

        except Exception as e:
            logger.warning(f"Error getting display name for {source}: {e}")
            return source

    def _generate_grid_vs_grid_stats(
        self, item: str, include_metrics: bool = True, include_scores: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive grid vs grid statistics like station case format."""
        grid_stats = {}

        # Get reference and simulation sources from configuration
        ref_sources = self._get_reference_sources(item)
        sim_sources = self._get_simulation_sources(item)

        # Log what we found for debugging
        logger.info(f"Found reference sources for {item}: {ref_sources}")
        logger.info(f"Found simulation sources for {item}: {sim_sources}")

        # If no sources found from config, try to infer from existing files
        if not ref_sources or not sim_sources:
            logger.info(f"Attempting to infer sources from existing files for {item}")
            inferred_ref_sources, inferred_sim_sources = self._infer_sources_from_files(item)
            if not ref_sources:
                ref_sources = inferred_ref_sources
            if not sim_sources:
                sim_sources = inferred_sim_sources
            logger.info(f"After inference - ref_sources: {ref_sources}, sim_sources: {sim_sources}")

        # Generate all possible comparison pairs
        comparison_pairs = []
        for ref_source in ref_sources:
            for sim_source in sim_sources:
                comparison_pairs.append((ref_source, sim_source))

        for ref_source, sim_source in comparison_pairs:
            # Get year information from data files
            syear = self._get_year_info(item, ref_source, sim_source, "syear")
            eyear = self._get_year_info(item, ref_source, sim_source, "eyear")

            # Collect all metrics and scores for this pair
            pair_data = {}
            if syear is not None:
                pair_data["use_syear"] = {
                    "values": [syear],
                    "mean": float(syear),
                    "std": 0.0,
                    "min": float(syear),
                    "max": float(syear),
                    "median": float(syear),
                    "coverage": 100.0,
                }
            else:
                pair_data["use_syear"] = {
                    "values": [np.nan],
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "median": np.nan,
                    "coverage": 0.0,
                }
            if eyear is not None:
                pair_data["use_eyear"] = {
                    "values": [eyear],
                    "mean": float(eyear),
                    "std": 0.0,
                    "min": float(eyear),
                    "max": float(eyear),
                    "median": float(eyear),
                    "coverage": 100.0,
                }
            else:
                pair_data["use_eyear"] = {
                    "values": [np.nan],
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "median": np.nan,
                    "coverage": 0.0,
                }

            metrics_files = (
                [
                    path
                    for path in self._item_output_files(self.metrics_dir, item, (".nc", ".nc4"))
                    if self._filename_matches_pair(path, item, ref_source, sim_source)
                ]
                if include_metrics
                else []
            )
            scores_files = (
                [
                    path
                    for path in self._item_output_files(self.scores_dir, item, (".nc", ".nc4"))
                    if self._filename_matches_pair(path, item, ref_source, sim_source)
                ]
                if include_scores
                else []
            )

            all_files = metrics_files + scores_files

            if not all_files:
                continue  # No files for this comparison pair

            # Process each metric/score file
            for nc_file in all_files:
                metric_name = self._extract_metric_type(os.path.basename(nc_file))
                if metric_name == "Unknown":
                    continue

                try:
                    with xr.open_dataset(nc_file) as ds:
                        data_vars = [var for var in ds.data_vars if var not in ds.coords]
                        if data_vars:
                            main_var = ds[data_vars[0]]
                            stats = self._summarize_data_array(main_var)
                            if stats:
                                pair_data[metric_name] = stats

                except Exception as e:
                    logger.warning(f"Error reading {nc_file}: {e}")

            if len(pair_data) > 2:  # More than just the year entries
                # Calculate average data coverage across all metrics
                coverages = [
                    metric_info.get("coverage", 0.0)
                    for metric_name, metric_info in pair_data.items()
                    if metric_name not in {"use_syear", "use_eyear"}
                    if isinstance(metric_info, dict) and "coverage" in metric_info
                ]
                avg_coverage = float(np.mean(coverages)) if coverages else None

                # Get comparison pair from configuration
                comparison_pair = self._get_comparison_pair_from_config(item, ref_source, sim_source)
                pair_key = comparison_pair

                grid_stats[pair_key] = {
                    "comparison_pair": comparison_pair,
                    "station_format": True,  # Flag to use station-like display
                    "metrics": pair_data,
                    "data_coverage": avg_coverage,
                }

                logger.info(f"Generated comprehensive stats for {comparison_pair}")

        return grid_stats

    def _infer_sources_from_files(self, item: str) -> tuple[List[str], List[str]]:
        """Infer reference and simulation sources from existing files"""
        ref_sources = set()
        sim_sources = set()

        # Search in metrics and scores directories
        search_dirs = [self.metrics_dir, self.scores_dir]

        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                files = self._item_output_files(search_dir, item, (".nc", ".nc4"))

                for file_path in files:
                    filename = os.path.basename(file_path)
                    safe_ref, safe_sim = self._extract_sources_from_safe_filename(filename, item)
                    if safe_ref:
                        ref_sources.add(safe_ref)
                    if safe_sim:
                        sim_sources.add(safe_sim)
                    if safe_ref or safe_sim:
                        continue

                    parts = filename.split("_")

                    # Extract ref and sim sources
                    ref_source = None
                    sim_source = None

                    for i, part in enumerate(parts):
                        if part == "ref" and i + 1 < len(parts):
                            # Find continuous reference name
                            ref_parts = []
                            j = i + 1
                            while j < len(parts) and parts[j] != "sim":
                                ref_parts.append(parts[j])
                                j += 1
                            ref_source = "_".join(ref_parts)

                        elif part == "sim" and i + 1 < len(parts):
                            # Find continuous simulation name
                            sim_parts = []
                            j = i + 1
                            while j < len(parts) and not self._is_metric_or_score(parts[j]) and parts[j] != "Overall":
                                sim_parts.append(parts[j])
                                j += 1
                            sim_source = "_".join(sim_parts)
                            break

                    if ref_source:
                        ref_sources.add(ref_source)
                    if sim_source:
                        sim_sources.add(sim_source)

        return list(ref_sources), list(sim_sources)

    def _extract_sources_from_safe_filename(self, filename: str, item: str) -> tuple[Optional[str], Optional[str]]:
        """Extract ref/sim source names from safe component-joined filenames."""
        stem = os.path.splitext(os.path.basename(filename))[0]
        parts = stem.split("__")
        if not parts or parts[0] != filename_component(item):
            return None, None
        ref_source = None
        sim_source = None
        try:
            ref_idx = parts.index("ref")
            if ref_idx + 1 < len(parts):
                ref_source = _decode_filename_component(parts[ref_idx + 1])
        except ValueError:
            pass
        try:
            sim_idx = parts.index("sim")
            if sim_idx + 1 < len(parts):
                sim_source = _decode_filename_component(parts[sim_idx + 1])
        except ValueError:
            pass
        return ref_source, sim_source

    def _get_reference_sources(self, item: str) -> List[str]:
        """Get reference sources for an evaluation item from general configuration"""
        try:
            ref_sources = []

            # Get from general reference configuration
            if "ref_nml" in self.config and "general" in self.config["ref_nml"]:
                general = self.config["ref_nml"]["general"]
                value = general.get(f"{item}_ref_source", general.get(item))
                if isinstance(value, str) and "," in value:
                    ref_sources.extend(s.strip() for s in value.split(",") if s.strip())
                elif isinstance(value, str) and value.strip():
                    ref_sources.append(value.strip())
                elif isinstance(value, list):
                    ref_sources.extend(str(s).strip() for s in value if str(s).strip())

            logger.info(f"Found reference sources for {item}: {ref_sources}")
            return ref_sources

        except Exception as e:
            logger.warning(f"Error getting reference sources for {item}: {e}")
            return []

    def _get_simulation_sources(self, item: str) -> List[str]:
        """Get simulation sources for an evaluation item from general configuration"""
        try:
            sim_sources = []

            # Get from general simulation configuration
            if "sim_nml" in self.config and "general" in self.config["sim_nml"]:
                general = self.config["sim_nml"]["general"]
                value = general.get(f"{item}_sim_source", general.get("Case_lib"))
                if isinstance(value, str) and "," in value:
                    sim_sources.extend(s.strip() for s in value.split(",") if s.strip())
                elif isinstance(value, str) and value.strip():
                    sim_sources.append(value.strip())
                elif isinstance(value, list):
                    sim_sources.extend(str(s).strip() for s in value if str(s).strip())

            logger.info(f"Found simulation sources for {item}: {sim_sources}")
            return sim_sources

        except Exception as e:
            logger.warning(f"Error getting simulation sources for {item}: {e}")
            return []

    def _get_year_info(self, item: str, ref_source: str, sim_source: str, year_type: str) -> Optional[int]:
        """Get year information directly from NetCDF data files"""
        years_found = []

        for role in ("ref", "sim"):
            data_files = [
                path
                for path in self._item_output_files(self.data_dir, item, (".nc", ".nc4"))
                if self._filename_matches_data_role(path, item, role)
            ]
            for data_file in data_files:
                try:
                    with xr.open_dataset(data_file) as ds:
                        if "time" in ds.dims or "time" in ds.coords:
                            time_var = ds["time"]
                            if len(time_var) > 0:
                                time_values = pd.to_datetime(time_var.values)
                                if year_type == "syear":
                                    years_found.append(int(time_values.min().year))
                                else:
                                    years_found.append(int(time_values.max().year))
                except Exception:
                    pass

        if years_found:
            if year_type == "syear":
                return max(years_found)  # use_syear = max of all start years
            else:
                return min(years_found)  # use_eyear = min of all end years

        return None  # Return None if no year info found

    def _find_figure(self, base_dir: str, subdir: str, pattern: str) -> Optional[str]:
        """Find a figure matching the pattern."""
        search_path = os.path.join(base_dir, subdir, pattern)
        files = [f for f in sorted(glob.glob(search_path)) if _is_report_figure(f)]
        if files:
            return f"{subdir}/{os.path.basename(files[0])}"
        return None

    def _copy_figures_to_report_dir(self):
        """Copy all referenced figures to the report directory"""
        logger.info("Copying figures to report directory...")

        # Create figures subdirectory in reports
        figures_dir = os.path.join(self.report_dir, "figures")
        if os.path.exists(figures_dir):
            _remove_report_tree(figures_dir)
        os.makedirs(figures_dir, exist_ok=True)

        # Track copied files for debugging
        copied_count = 0

        # Copy metrics figures (including groupby subdirectories)
        if os.path.exists(self.metrics_dir):
            for root, dirs, files in os.walk(self.metrics_dir):
                for file in files:
                    if _is_report_figure(file):
                        src_file = os.path.join(root, file)
                        rel_path = os.path.relpath(src_file, self.metrics_dir).replace(os.sep, "/")
                        dst_file = os.path.join(figures_dir, "metrics", rel_path)
                        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                        shutil.copyfile(src_file, dst_file)
                        copied_count += 1
                        logger.debug(f"Copied metrics figure: {rel_path}")

        # Copy scores figures (including groupby subdirectories)
        if os.path.exists(self.scores_dir):
            for root, dirs, files in os.walk(self.scores_dir):
                for file in files:
                    if _is_report_figure(file):
                        src_file = os.path.join(root, file)
                        rel_path = os.path.relpath(src_file, self.scores_dir).replace(os.sep, "/")
                        dst_file = os.path.join(figures_dir, "scores", rel_path)
                        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                        shutil.copyfile(src_file, dst_file)
                        copied_count += 1
                        logger.debug(f"Copied scores figure: {rel_path}")

        # Copy comparison figures
        if os.path.exists(self.comparisons_dir):
            for root, dirs, files in os.walk(self.comparisons_dir):
                for file in files:
                    if _is_report_figure(file):
                        src_file = os.path.join(root, file)
                        rel_path = os.path.relpath(src_file, self.comparisons_dir).replace(os.sep, "/")
                        dst_file = os.path.join(figures_dir, "comparisons", rel_path)
                        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                        shutil.copyfile(src_file, dst_file)
                        copied_count += 1
                        logger.debug(f"Copied comparison figure: {rel_path}")

        # Copy station/data figures
        if os.path.exists(self.data_dir):
            for root, dirs, files in os.walk(self.data_dir):
                for file in files:
                    if _is_report_figure(file):
                        src_file = os.path.join(root, file)
                        rel_path = os.path.relpath(src_file, self.data_dir).replace(os.sep, "/")
                        dst_file = os.path.join(figures_dir, "data", rel_path)
                        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                        shutil.copyfile(src_file, dst_file)
                        copied_count += 1
                        logger.debug(f"Copied data figure: {rel_path}")

        logger.info(f"Copied {copied_count} figures to report directory")
        _remove_appledouble_files(figures_dir)

        output_figures_dir = os.path.join(self.output_dir, "figures")
        if os.path.exists(output_figures_dir):
            _remove_report_tree(output_figures_dir)
        shutil.copytree(figures_dir, output_figures_dir)
        logger.info(f"Published {copied_count} figures to output figures directory")

    def _verify_figure_paths(self, report_data: Dict[str, Any]):
        """Verify that all referenced figures exist in the report directory"""
        logger.info("Verifying figure paths...")
        figures_dir = os.path.join(self.report_dir, "figures")
        missing_figures = []
        total_figures = 0

        # Check figures for each evaluation item
        for item, item_data in report_data.get("evaluation_items", {}).items():
            figures = item_data.get("figures", {})

            # Check all figure types
            for fig_type, fig_list in figures.items():
                for fig_path in fig_list:
                    total_figures += 1
                    # Build the expected path in the reports directory
                    if fig_type == "metrics":
                        expected_path = os.path.join(figures_dir, "metrics", fig_path)
                    elif fig_type == "scores":
                        expected_path = os.path.join(figures_dir, "scores", fig_path)
                    elif fig_type == "comparisons":
                        expected_path = os.path.join(figures_dir, "comparisons", fig_path.replace("comparisons/", ""))
                    elif fig_type in ["igbp_groupby", "pft_groupby", "climate_zone_groupby", "station_timeseries"]:
                        # These are already prefixed with their source area (comparisons/ or data/).
                        expected_path = os.path.join(figures_dir, fig_path)
                    else:
                        expected_path = os.path.join(figures_dir, fig_path)

                    if not os.path.exists(expected_path):
                        missing_figures.append(f"{item}/{fig_type}: {fig_path} -> {expected_path}")
                        logger.warning(f"Missing figure: {expected_path}")

        # Check comparison figures
        comparisons = report_data.get("comparisons", {})
        if "figures" in comparisons:
            for fig_key, fig_path in comparisons["figures"].items():
                total_figures += 1
                expected_path = os.path.join(figures_dir, "comparisons", fig_path)
                if not os.path.exists(expected_path):
                    missing_figures.append(f"comparison/{fig_key}: {fig_path} -> {expected_path}")
                    logger.warning(f"Missing comparison figure: {expected_path}")

        if missing_figures:
            logger.warning(f"Found {len(missing_figures)} missing figures out of {total_figures} total figures:")
            for missing in missing_figures[:10]:  # Show first 10
                logger.warning(f"  - {missing}")
            if len(missing_figures) > 10:
                logger.warning(f"  ... and {len(missing_figures) - 10} more")
        else:
            logger.info(f"All {total_figures} figures verified successfully")

    def _generate_html_report(self, report_data: Dict[str, Any], report_name: str) -> str:
        """Generate HTML report from collected data"""
        logger.info("Generating HTML report...")

        # HTML template
        html_template = _jinja_env.from_string("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ metadata.title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f4f4f4;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 5px;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
        }
        .metadata {
            margin-top: 15px;
            font-size: 0.9em;
            opacity: 0.9;
        }
        .section {
            background-color: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        h2 {
            color: #2c3e50;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 10px;
        }
        h3 {
            color: #34495e;
            margin-top: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #34495e;
            color: white;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .figure-container {
            margin: 20px 0;
            text-align: center;
        }
        .figure-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .figure-caption {
            margin-top: 10px;
            font-size: 0.9em;
            color: #666;
            font-style: italic;
        }
        .summary-box {
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .metric-card {
            display: inline-block;
            background-color: #3498db;
            color: white;
            padding: 15px 25px;
            margin: 10px;
            border-radius: 5px;
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
        }
        .metric-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        .toc {
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }
        .toc h3 {
            margin-top: 0;
        }
        .toc ul {
            list-style-type: none;
            padding-left: 20px;
        }
        .toc a {
            color: #3498db;
            text-decoration: none;
        }
        .toc a:hover {
            text-decoration: underline;
        }
        @media print {
            body {
                background-color: white;
            }
            .section {
                box-shadow: none;
                border: 1px solid #ddd;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ metadata.title }}</h1>
        <div class="metadata">
            <p><strong>Generated:</strong> {{ metadata.generated_date }}</p>
            <p><strong>Configuration:</strong> {{ metadata.config_file }}</p>
            <p><strong>Evaluation Items:</strong> {{ ', '.join(metadata.evaluation_items) }}</p>
        </div>
    </div>
    
    <!-- Table of Contents -->
    <div class="section toc">
        <h3>Table of Contents</h3>
        <ul>
            <li><a href="#summary">Executive Summary</a></li>
            {% for item in metadata.evaluation_items %}
            <li><a href="#{{ item|replace(' ', '-')|lower }}">{{ item }} Analysis</a></li>
            {% endfor %}
            {% if uncertainty %}
            <li><a href="#uncertainty">Uncertainty-aware Evaluation</a></li>
            {% endif %}
            <li><a href="#overall-comparison">Overall Comparison</a></li>
            <li><a href="#appendix">Appendix</a></li>
        </ul>
    </div>
    
    <!-- Executive Summary -->
    <div class="section" id="summary">
        <h2>Executive Summary</h2>
        <div class="summary-box">
            <p>This report presents the comprehensive evaluation results from OpenBench Land Surface Model benchmarking system.</p>
            
            <div style="text-align: center; margin: 20px 0;">
                {% if overall_summary.grand_average is defined %}
                <div class="metric-card">
                    <div class="metric-value">{{ "%.3f"|format(overall_summary.grand_average) }}</div>
                    <div class="metric-label">Overall Average Score</div>
                </div>
                {% endif %}
                
                <div class="metric-card" style="background-color: #2ecc71;">
                    <div class="metric-value">{{ overall_summary.total_items }}</div>
                    <div class="metric-label">Evaluation Items</div>
                </div>
            </div>
            
            <h3>Key Findings</h3>
            <ul>
                {% for item in metadata.evaluation_items %}
                <li><strong>{{ item }}:</strong> Evaluation completed with multiple reference datasets</li>
                {% endfor %}
            </ul>
            
            {% if groupby_summary.total_groupby_analyses > 0 %}
            <h3>Groupby Analysis Summary</h3>
            <p>The evaluation includes detailed analysis across different classification schemes:</p>
            <ul>
                {% if groupby_summary.has_igbp %}
                <li><strong>IGBP Land Cover Analysis:</strong> {{ groupby_summary.igbp_message }}</li>
                {% endif %}
                {% if groupby_summary.has_pft %}
                <li><strong>Plant Functional Type Analysis:</strong> {{ groupby_summary.pft_message }}</li>
                {% endif %}
                {% if groupby_summary.has_climate_zone %}
                <li><strong>Climate Zone Analysis:</strong> {{ groupby_summary.climate_zone_message }}</li>
                {% endif %}
            </ul>
            <p>These groupby analyses provide insights into model performance across different land cover types, vegetation functional groups, and climatic conditions, helping identify systematic biases and areas for improvement.</p>
            {% endif %}
        </div>
    </div>
    
    {% if uncertainty %}
    <div class="section" id="uncertainty">
        <h2>Uncertainty-aware Evaluation</h2>
        <p>Confidence intervals use paired, gap-aware moving-block bootstrap resampling. Model spread and reference sensitivity are reported as separate axes.</p>

        {% if uncertainty.bootstrap %}
        <h3>Aggregate Metric Confidence Intervals</h3>
        <table>
            <thead>
                <tr>
                    <th>Variable</th><th>Reference</th><th>Simulation</th><th>Metric</th>
                    <th>Scope</th><th>Valid pairs</th><th>Segments</th>
                    <th>Estimate</th><th>Confidence Interval</th><th>Status</th>
                </tr>
            </thead>
            <tbody>
                {% for row in uncertainty.bootstrap %}
                <tr>
                    <td>{{ row.variable }}</td><td>{{ row.reference }}</td><td>{{ row.simulation }}</td>
                    <td>{{ row.metric }}</td><td>{{ row.scope }}</td>
                    <td>{{ row.valid_pair_count }}</td><td>{{ row.segment_count }}</td>
                    {% if row.status == 'available' %}
                    <td>{{ "%.4g"|format(row.estimate) }}</td>
                    <td>[{{ "%.4g"|format(row.lower) }}, {{ "%.4g"|format(row.upper) }}]</td>
                    {% else %}
                    <td>—</td><td>—</td>
                    {% endif %}
                    <td>{{ row.status|replace('_', ' ') }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}

        {% if uncertainty.verdicts %}
        <h3>Pairwise Model Verdicts</h3>
        <table>
            <thead>
                <tr><th>Variable</th><th>Metric</th><th>Model Pair</th><th>Verdict</th><th>Winner</th></tr>
            </thead>
            <tbody>
                {% for row in uncertainty.verdicts %}
                <tr>
                    <td>{{ row.variable }}</td><td>{{ row.metric }}</td>
                    <td>{{ row.simulation_a }} vs {{ row.simulation_b }}</td>
                    <td>{{ row.status|replace('_', ' ') }}</td><td>{{ row.winner or '—' }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}

        {% if uncertainty.products %}
        <h3>Spatial and Station Products</h3>
        <ul>
            {% for path in uncertainty.products.model_spread %}
            <li>Model spread: <a href="../{{ path|url_path }}">{{ path }}</a></li>
            {% endfor %}
            {% for path in uncertainty.products.reference_sensitivity %}
            <li>Reference sensitivity: <a href="../{{ path|url_path }}">{{ path }}</a></li>
            {% endfor %}
        </ul>
        {% endif %}
    </div>
    {% endif %}

    <!-- Individual Item Analysis -->
    {% for item, item_data in evaluation_items.items() %}
    <div class="section" id="{{ item|replace(' ', '-')|lower }}">
        <h2>{{ item }} Analysis</h2>
        
        <!-- Metrics Summary -->
        {% if item_data.metrics %}
        <h3>Evaluation Metrics</h3>
        {% for metric_key, metric_data in item_data.metrics.items() %}
            {% if metric_data.summary %}
            <h4>{{ metric_key }}</h4>
            <div class="summary-box">
                {% for metric_name, values in metric_data.summary.items() %}
                    {% if values.mean is not none and values.mean == values.mean %}
                    {% if metric_name in ['use_syear', 'use_eyear'] %}
                    <p><strong>{{ metric_name }}:</strong> {{ "%.0f"|format(values.mean) }}</p>
                    {% else %}
                    <p><strong>{{ metric_name }}:</strong> 
                       Mean = {{ "%.4f"|format(values.mean) }}, 
                       Std = {{ "%.4f"|format(values.std) }}, 
                       Range = [{{ "%.4f"|format(values.min) }}, {{ "%.4f"|format(values.max) }}]
                    </p>
                    {% endif %}
                    {% endif %}
                {% endfor %}
                {% if metric_data.median_omitted %}
                <p>Median omitted for large result; all other statistics were calculated from the complete dataset.</p>
                {% endif %}
            </div>
            {% elif metric_data.station_format %}
            <!-- Grid vs Grid comprehensive statistics (station format) -->
            <h4>{{ metric_data.comparison_pair }}</h4>
            <div class="summary-box">
                {% for metric_name, metric_values in metric_data.metrics.items() %}
                {% if metric_name in ['use_syear', 'use_eyear'] %}
                {% if metric_values.mean == metric_values.mean %}
                <p><strong>{{ metric_name }}:</strong> {{ "%.0f"|format(metric_values.mean) }}</p>
                {% endif %}
                {% else %}
                <p><strong>{{ metric_name }}:</strong> 
                   Mean = {{ "%.4f"|format(metric_values.mean) }}, 
                   Std = {{ "%.4f"|format(metric_values.std) }}, 
                   Median = {% if metric_values.median is not none %}{{ "%.4f"|format(metric_values.median) }}{% else %}omitted for large result{% endif %},
                   Range = [{{ "%.4f"|format(metric_values.min) }}, {{ "%.4f"|format(metric_values.max) }}]
                   {% if metric_data.data_coverage is not none %}, Data Coverage = {{ "%.1f"|format(metric_data.data_coverage) }}%{% endif %}
                </p>
                {% endif %}
                {% endfor %}
            </div>
            {% endif %}
        {% endfor %}
        {% endif %}
        
        <!-- Scores Summary -->
        {% if item_data.scores %}
        <h3>Evaluation Scores</h3>
        {% for score_key, score_data in item_data.scores.items() %}
            {% if score_data.summary %}
            <h4>{{ score_key }}</h4>
            <div class="summary-box">
                {% for score_name, values in score_data.summary.items() %}
                    {% if values.mean is not none and values.mean == values.mean %}
                    <p><strong>{{ score_name }}:</strong>
                       Mean = {{ "%.4f"|format(values.mean) }},
                       Std = {{ "%.4f"|format(values.std) }},
                       Range = [{{ "%.4f"|format(values.min) }}, {{ "%.4f"|format(values.max) }}]
                    </p>
                    {% endif %}
                {% endfor %}
            </div>
            {% elif score_data.station_format %}
            <h4>{{ score_data.comparison_pair }}</h4>
            <div class="summary-box">
                {% for score_name, score_values in score_data.metrics.items() %}
                {% if score_name not in ['use_syear', 'use_eyear'] %}
                <p><strong>{{ score_name }}:</strong>
                   Mean = {{ "%.4f"|format(score_values.mean) }},
                   Std = {{ "%.4f"|format(score_values.std) }},
                   Median = {% if score_values.median is not none %}{{ "%.4f"|format(score_values.median) }}{% else %}omitted for large result{% endif %},
                   Range = [{{ "%.4f"|format(score_values.min) }}, {{ "%.4f"|format(score_values.max) }}]
                   {% if score_data.data_coverage is not none %}, Data Coverage = {{ "%.1f"|format(score_data.data_coverage) }}%{% endif %}
                </p>
                {% endif %}
                {% endfor %}
            </div>
            {% endif %}
        {% endfor %}
        {% endif %}

        <!-- Metric Figures -->
        {% if item_data.figures.metrics %}
        <h3>Metric Visualizations</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px;">
            {% for fig in item_data.figures.metrics %}
            <div class="figure-container">
                <img src="figures/metrics/{{ fig|url_path }}" alt="{{ fig }}">
                <div class="figure-caption">{{ fig|replace('_', ' ')|replace('.jpg', '')|replace('.jpeg', '')|replace('.png', '')|replace('.svg', '')|replace('.webp', '') }}</div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        <!-- Score Figures -->
        {% if item_data.figures.scores %}
        <h3>Score Visualizations</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px;">
            {% for fig in item_data.figures.scores %}
            <div class="figure-container">
                <img src="figures/scores/{{ fig|url_path }}" alt="{{ fig }}">
                <div class="figure-caption">{{ fig|replace('_', ' ')|replace('.jpg', '')|replace('.jpeg', '')|replace('.png', '')|replace('.svg', '')|replace('.webp', '') }}</div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        <!-- Comparison Figures -->
        {% if item_data.figures.comparisons %}
        <h3>Detailed Comparisons</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px;">
            {% for fig in item_data.figures.comparisons %}
            <div class="figure-container">
                <img src="figures/comparisons/{{ fig|url_path }}" alt="{{ fig }}">
                <div class="figure-caption">{{ fig|replace('/', ' - ')|replace('_', ' ')|replace('.jpg', '')|replace('.jpeg', '')|replace('.png', '')|replace('.svg', '')|replace('.webp', '') }}</div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        <!-- Station Timeseries Figures -->
        {% if item_data.figures.station_timeseries %}
        <h3>Station Time Series</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px;">
            {% for fig in item_data.figures.station_timeseries %}
            <div class="figure-container">
                <img src="figures/{{ fig|url_path }}" alt="{{ fig }}">
                <div class="figure-caption">{{ fig|replace('/', ' - ')|replace('_', ' ')|replace('.jpg', '')|replace('.jpeg', '')|replace('.png', '')|replace('.svg', '')|replace('.webp', '') }}</div>
            </div>
            {% endfor %}
        </div>
        {% endif %}

        <!-- IGBP Groupby Analysis -->
        {% if item_data.figures.igbp_groupby or item_data.statistics.get('IGBP_groupby') %}
        <h3>IGBP Land Cover Classification Analysis</h3>
        
        <div class="summary-box">
            <p><strong>Analysis Overview:</strong> Performance evaluation across International Geosphere-Biosphere Programme (IGBP) land cover classes provides insights into model behavior across different vegetation types and land use categories.</p>
            <p><strong>IGBP Classes Include:</strong> Evergreen Needleleaf Forest, Evergreen Broadleaf Forest, Deciduous Needleleaf Forest, Deciduous Broadleaf Forest, Mixed Forest, Closed Shrublands, Open Shrublands, Woody Savannas, Savannas, Grasslands, Permanent Wetlands, Croplands, Urban and Built-up, Cropland/Natural Vegetation Mosaic, Snow and Ice, Barren or Sparsely Vegetated, and Water Bodies.</p>
        </div>
        
        {% if item_data.statistics.get('IGBP_groupby') %}
        <div class="summary-box">
            <p><strong>{{ item_data.statistics.IGBP_groupby.description }}</strong></p>
            {% if item_data.statistics.IGBP_groupby.statistics %}
                {% for stat_data in item_data.statistics.IGBP_groupby.statistics %}
                <h4>{{ stat_data.file|replace('_', ' ')|replace('.csv', '') }}</h4>
                {% if stat_data.summary.groups %}
                <p><strong>Groups analyzed:</strong> {{ ', '.join(stat_data.summary.groups) }} ({{ stat_data.summary.group_count|default(stat_data.summary.groups|length) }} groups)</p>
                <p><strong>Key Findings:</strong> The analysis reveals model performance variations across different land cover types, highlighting strengths and weaknesses in simulating specific ecosystems.</p>
                {% if stat_data.summary.best_performing_groups %}
                <p><strong>Best Performing Groups:</strong> {{ ', '.join(stat_data.summary.best_performing_groups) }}</p>
                {% endif %}
                {% if stat_data.summary.worst_performing_groups %}
                <p><strong>Areas for Improvement:</strong> {{ ', '.join(stat_data.summary.worst_performing_groups) }}</p>
                {% endif %}
                {% endif %}
                {% endfor %}
            {% endif %}
        </div>
        {% endif %}
        
        {% if item_data.figures.igbp_groupby %}
        <div style="display: grid; grid-template-columns: 1fr; gap: 20px; max-width: 800px; margin: 0 auto;">
            {% for fig in item_data.figures.igbp_groupby %}
            <div class="figure-container">
                <img src="figures/{{ fig|url_path }}" alt="{{ fig }}">
                <div class="figure-caption">{{ fig|replace('/', ' - ')|replace('_', ' ')|replace('.jpg', '')|replace('.jpeg', '')|replace('.png', '')|replace('.svg', '')|replace('.webp', '') }}</div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
        {% endif %}
        
        <!-- PFT Groupby Analysis -->
        {% if item_data.figures.pft_groupby or item_data.statistics.get('PFT_groupby') %}
        <h3>Plant Functional Type (PFT) Analysis</h3>
        
        <div class="summary-box">
            <p><strong>Analysis Overview:</strong> Plant Functional Type (PFT) classification groups vegetation based on physiological and structural characteristics, enabling detailed assessment of model performance for different plant strategies and ecosystem functions.</p>
            <p><strong>PFT Categories:</strong> Analysis typically includes Needleleaf Evergreen/Deciduous Trees, Broadleaf Evergreen/Deciduous Trees, Shrubs, C3/C4 Grasses, and Crops, each representing distinct plant functional strategies and environmental adaptations.</p>
        </div>
        
        {% if item_data.statistics.get('PFT_groupby') %}
        <div class="summary-box">
            <p><strong>{{ item_data.statistics.PFT_groupby.description }}</strong></p>
            {% if item_data.statistics.PFT_groupby.statistics %}
                {% for stat_data in item_data.statistics.PFT_groupby.statistics %}
                <h4>{{ stat_data.file|replace('_', ' ')|replace('.csv', '') }}</h4>
                {% if stat_data.summary.groups %}
                <p><strong>Groups analyzed:</strong> {{ ', '.join(stat_data.summary.groups) }} ({{ stat_data.summary.group_count|default(stat_data.summary.groups|length) }} groups)</p>
                <p><strong>Key Findings:</strong> PFT-based analysis reveals how well the model captures the distinct behaviors of different plant functional groups, particularly their responses to environmental conditions and resource availability.</p>
                {% if stat_data.summary.best_performing_groups %}
                <p><strong>Best Performing PFTs:</strong> {{ ', '.join(stat_data.summary.best_performing_groups) }}</p>
                {% endif %}
                {% if stat_data.summary.worst_performing_groups %}
                <p><strong>PFTs Requiring Attention:</strong> {{ ', '.join(stat_data.summary.worst_performing_groups) }}</p>
                {% endif %}
                {% endif %}
                {% endfor %}
            {% endif %}
        </div>
        {% endif %}
        
        {% if item_data.figures.pft_groupby %}
        <div style="display: grid; grid-template-columns: 1fr; gap: 20px; max-width: 800px; margin: 0 auto;">
            {% for fig in item_data.figures.pft_groupby %}
            <div class="figure-container">
                <img src="figures/{{ fig|url_path }}" alt="{{ fig }}">
                <div class="figure-caption">{{ fig|replace('/', ' - ')|replace('_', ' ')|replace('.jpg', '')|replace('.jpeg', '')|replace('.png', '')|replace('.svg', '')|replace('.webp', '') }}</div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
        {% endif %}
        
        <!-- Climate Zone Groupby Analysis -->
        {% if item_data.figures.climate_zone_groupby or item_data.statistics.get('Climate_zone_groupby') %}
        <h3>Climate Zone Classification Analysis</h3>
        
        <div class="summary-box">
            <p><strong>Analysis Overview:</strong> Climate zone analysis based on Köppen-Geiger classification evaluates model performance across different climatic regimes, revealing how well the model captures processes under varying temperature and precipitation conditions.</p>
            <p><strong>Climate Zones Include:</strong> Tropical (Af, Am, Aw), Dry (BWh, BWk, BSh, BSk), Temperate (Cfa, Cfb, Cfc, Csa, Csb, Csc, Cwa, Cwb, Cwc), Continental (Dfa, Dfb, Dfc, Dfd, Dsa, Dsb, Dsc, Dsd, Dwa, Dwb, Dwc, Dwd), and Polar (ET, EF) climates.</p>
        </div>
        
        {% if item_data.statistics.get('Climate_zone_groupby') %}
        <div class="summary-box">
            <p><strong>{{ item_data.statistics.Climate_zone_groupby.description }}</strong></p>
            {% if item_data.statistics.Climate_zone_groupby.statistics %}
                {% for stat_data in item_data.statistics.Climate_zone_groupby.statistics %}
                <h4>{{ stat_data.file|replace('_', ' ')|replace('.csv', '') }}</h4>
                {% if stat_data.summary.groups %}
                <p><strong>Groups analyzed:</strong> {{ ', '.join(stat_data.summary.groups) }} ({{ stat_data.summary.group_count|default(stat_data.summary.groups|length) }} groups)</p>
                <p><strong>Key Findings:</strong> Climate zone analysis identifies systematic biases and performance patterns across different climatic conditions, helping to understand model strengths in specific climate regimes and areas requiring improvement.</p>
                {% if stat_data.summary.best_performing_groups %}
                <p><strong>Best Performance in Climate Zones:</strong> {{ ', '.join(stat_data.summary.best_performing_groups) }}</p>
                {% endif %}
                {% if stat_data.summary.worst_performing_groups %}
                <p><strong>Challenging Climate Zones:</strong> {{ ', '.join(stat_data.summary.worst_performing_groups) }}</p>
                {% endif %}
                {% endif %}
                {% endfor %}
            {% endif %}
        </div>
        {% endif %}
        
        {% if item_data.figures.climate_zone_groupby %}
        <div style="display: grid; grid-template-columns: 1fr; gap: 20px; max-width: 800px; margin: 0 auto;">
            {% for fig in item_data.figures.climate_zone_groupby %}
            <div class="figure-container">
                <img src="figures/{{ fig|url_path }}" alt="{{ fig }}">
                <div class="figure-caption">{{ fig|replace('/', ' - ')|replace('_', ' ')|replace('.jpg', '')|replace('.jpeg', '')|replace('.png', '')|replace('.svg', '')|replace('.webp', '') }}</div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
        {% endif %}
    </div>
    {% endfor %}
    
    <!-- Overall Comparison -->
    {% if comparisons %}
    <div class="section" id="overall-comparison">
        <h2>Overall Comparison</h2>
        
        {% if comparisons.figures.heatmap %}
        <div class="figure-container">
            <img src="figures/comparisons/{{ comparisons.figures.heatmap|url_path }}" alt="{{ comparisons.score|replace('_', ' ') }} Heatmap">
            <div class="figure-caption">Figure: {{ comparisons.score|replace('_', ' ') }} Comparison Heatmap</div>
        </div>
        {% endif %}
        
        {% if comparisons.heatmap %}
        <h3>Score Comparison Table</h3>
        <table>
            <thead>
                <tr>
                    {% for key in comparisons.heatmap[0].keys() %}
                    <th>{{ key }}</th>
                    {% endfor %}
                </tr>
            </thead>
            <tbody>
                {% for row in comparisons.heatmap %}
                <tr>
                    {% for value in row.values() %}
                    <td>{{ value }}</td>
                    {% endfor %}
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}
        
        {% if comparisons.figures.radar %}
        <div class="figure-container">
            <img src="figures/comparisons/{{ comparisons.figures.radar|url_path }}" alt="Radar Map">
            <div class="figure-caption">Figure: Multi-dimensional Performance Radar Chart</div>
        </div>
        {% endif %}
    </div>
    {% endif %}
    
    <!-- Appendix -->
    <div class="section" id="appendix">
        <h2>Appendix</h2>
        <h3>Methodology</h3>
        <p>The evaluation was performed using the OpenBench Land Surface Model benchmarking system, which includes:</p>
        <ul>
            <li>Multiple evaluation metrics: RMSE, Correlation, KGESS, and various scoring methods</li>
            <li>Spatial and temporal analysis across different scales</li>
            <li>Climate zone-based grouping for regional analysis</li>
            <li>Comprehensive visualization suite for result interpretation</li>
        </ul>
        
        <h3>Data Sources</h3>
        <p>Reference datasets used in this evaluation:</p>
        <ul>
            <li>GLEAM: Global Land Evaporation Amsterdam Model</li>
            <li>ILAMB: International Land Model Benchmarking</li>
            <li>PLUMBER2: Protocol for Land Surface Model Benchmarking Evaluation</li>
        </ul>
    </div>
    
    <div style="text-align: center; padding: 20px; color: #666;">
        <p>Generated by OpenBench v2.0 | {{ metadata.generated_date }}</p>
    </div>
</body>
</html>""")

        # Render HTML
        html_content = html_template.render(**report_data)

        # Save HTML file
        html_path = os.path.join(self.report_dir, f"{report_name}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return html_path

    def _generate_pdf_report(self, html_path: str, report_name: str) -> Optional[str]:
        """
        Generate PDF report from HTML file using xhtml2pdf

        Args:
            html_path: Path to the HTML file
            report_name: Base name for the PDF file

        Returns:
            Path to generated PDF file, or None if generation failed
        """
        if not PDF_AVAILABLE:
            logger.warning("PDF generation not available. Please install the report extra.")
            logger.warning("Run: pip install 'colm-openbench[report]' (or install xhtml2pdf in your conda env)")
            return None

        try:
            logger.info("Generating PDF report using xhtml2pdf...")
            pdf_path = os.path.join(self.report_dir, f"{report_name}.pdf")

            # Read the HTML file
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            # Modify HTML content for better PDF generation
            # Convert relative image paths to absolute paths
            import re

            # Replace relative image paths with absolute paths
            def replace_img_src(match):
                src = match.group(1)

                # Skip if already an absolute path or URL
                if src.startswith(("http://", "https://", "file://", "/")):
                    return match.group(0)

                # Handle paths that start with 'figures/'
                if src.startswith("figures/"):
                    abs_path = os.path.join(self.report_dir, src)
                    abs_path = os.path.abspath(abs_path)  # Normalize the path
                    if os.path.exists(abs_path):
                        logger.debug(f"Converting image path: {src} -> {abs_path}")
                        return f'src="file://{abs_path}"'
                    else:
                        logger.warning(f"Image file not found: {abs_path}")
                        return f'src="file://{abs_path}"'  # Keep file:// prefix even if not found

                # Handle other relative paths
                elif not src.startswith("./"):
                    # If it's a relative path without ./, try to resolve it relative to report dir
                    potential_path = os.path.join(self.report_dir, src)
                    potential_path = os.path.abspath(potential_path)
                    if os.path.exists(potential_path):
                        logger.debug(f"Converting relative path: {src} -> {potential_path}")
                        return f'src="file://{potential_path}"'

                return match.group(0)

            html_content = re.sub(r'src="([^"]+)"', replace_img_src, html_content)

            # Debug: log some sample conversions
            sample_matches = re.findall(r'src="([^"]*figures[^"]*)"', html_content)
            if sample_matches:
                logger.debug(f"Sample converted image paths: {sample_matches[:5]}")  # Show first 5

            # Add PDF-specific CSS
            pdf_css = """
            <style type="text/css" media="print">
                @page {
                    margin: 2cm;
                    size: A4;
                }
                body {
                    font-size: 10pt;
                    line-height: 1.3;
                }
                .header {
                    page-break-inside: avoid;
                }
                .section {
                    page-break-inside: avoid;
                    margin-bottom: 1em;
                }
                .figure-container {
                    page-break-inside: avoid;
                    text-align: center;
                    margin: 1em 0;
                }
                table {
                    font-size: 8pt;
                    width: 100%;
                }
                th, td {
                    padding: 4px;
                    font-size: 8pt;
                }
                h2 {
                    page-break-after: avoid;
                    font-size: 14pt;
                }
                h3 {
                    page-break-after: avoid;
                    font-size: 12pt;
                }
                h4 {
                    page-break-after: avoid;
                    font-size: 11pt;
                }
            </style>
            """

            # Insert PDF CSS into HTML head
            html_content = html_content.replace("</head>", pdf_css + "</head>")

            # Generate PDF
            with open(pdf_path, "wb") as pdf_file:
                result = pisa.CreatePDF(
                    html_content, dest=pdf_file, encoding="utf-8", link_callback=self._link_callback
                )

                if result.err:
                    logger.error(f"PDF generation had errors: {result.err}")
                    return None

            logger.info(f"PDF report generated successfully: {pdf_path}")
            return pdf_path

        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            return None

    def _link_callback(self, uri, rel):
        """
        Callback function to handle local file links in PDF generation
        """
        # Handle file:// URLs
        if uri.startswith("file://"):
            path = uri[7:]  # Remove 'file://' prefix
            if os.path.exists(path):
                logger.debug(f"Link callback found file: {path}")
                return path
            else:
                logger.warning(f"Link callback file not found: {path}")
                return path  # Return path anyway, let PDF generator handle it

        # Handle relative paths directly
        elif uri.startswith("figures/"):
            abs_path = os.path.join(self.report_dir, uri)
            abs_path = os.path.abspath(abs_path)
            if os.path.exists(abs_path):
                logger.debug(f"Link callback resolved relative path: {uri} -> {abs_path}")
                return abs_path
            else:
                logger.warning(f"Link callback could not find relative path: {uri} -> {abs_path}")
                return abs_path

        # Handle other relative paths (like ./output/...)
        elif uri.startswith("./") and "figures" in uri:
            # Extract the figures part
            if "reports/figures" in uri:
                figures_part = uri.split("reports/figures/")[-1]
                abs_path = os.path.join(self.report_dir, "figures", figures_part)
                abs_path = os.path.abspath(abs_path)
                if os.path.exists(abs_path):
                    logger.debug(f"Link callback resolved complex relative path: {uri} -> {abs_path}")
                    return abs_path
                else:
                    logger.warning(f"Link callback could not find complex relative path: {uri} -> {abs_path}")
                    return abs_path

        return uri
