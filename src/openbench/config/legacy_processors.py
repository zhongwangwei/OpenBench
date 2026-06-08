# -*- coding: utf-8 -*-
"""
Configuration Processors for OpenBench

This module provides complex configuration processing classes that handle
OpenBench-specific data processing and analysis logic.

Author: Zhongwang Wei
Version: 2.0
Date: July 2025
"""

import logging
import os
import re
from copy import deepcopy
from typing import Any, Dict, Tuple

from openbench.util.exceptions import ConfigurationError

# Heavy dependencies for data processing
try:
    import numpy as np  # noqa: F401  feature-detection import
    import pandas as pd  # noqa: F401  feature-detection import
    import xarray as xr  # noqa: F401  feature-detection import
    from joblib import Parallel, delayed  # noqa: F401  feature-detection import

    _HAS_DATA_LIBS = True
except ImportError:
    _HAS_DATA_LIBS = False
    logging.warning("Data processing libraries (numpy, pandas, xarray, joblib) not available")

# Import caching - CacheSystem is mandatory for data processing modules
try:
    from openbench.data.cache import cached, get_cache_manager  # noqa: F401  feature detection

    _HAS_CACHE = True
except ImportError:
    _HAS_CACHE = False

    # Make CacheSystem optional for config processors since they're less compute-intensive
    def cached(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


class GeneralInfoReader:
    """
    Advanced configuration processor for OpenBench evaluation information.

    This class handles complex data processing tasks including time resolution
    normalization, station data filtering, and evaluation parameter processing.
    """

    # Keys that must never be overwritten by namelist data
    _PROTECTED_KEYS = frozenset(
        {
            "__class__",
            "__dict__",
            "__doc__",
            "__module__",
            "_safe_update",
            "_PROTECTED_KEYS",
            "_custom_filter_warnings_shown",
            "_time_resolution_warning_shown",
        }
    )

    def _safe_update(self, data: dict) -> None:
        """Update instance attributes from dict, skipping protected/method names."""
        for key, value in data.items():
            if key.startswith("__") or key in self._PROTECTED_KEYS or callable(getattr(self, key, None)):
                logging.debug("Skipping protected/method key in namelist: %s", key)
                continue
            setattr(self, key, value)

    def __init__(
        self,
        main_nl: Dict[str, Any],
        sim_nml: Dict[str, Any],
        ref_nml: Dict[str, Any],
        metric_vars: list,
        score_vars: list,
        comparison_vars: list,
        statistic_vars: list,
        item: str,
        sim_source: str,
        ref_source: str,
    ):
        """
        Initialize the GeneralInfoReader with configuration data.

        Args:
            main_nl: Main namelist containing general settings
            sim_nml: Simulation namelist
            ref_nml: Reference namelist
            metric_vars: List of metric variables
            score_vars: List of score variables
            comparison_vars: List of comparison variables
            statistic_vars: List of statistic variables
            item: The evaluation item being processed
            sim_source: The simulation data source
            ref_source: The reference data source
        """
        if not _HAS_DATA_LIBS:
            raise ConfigurationError("Data processing libraries required for GeneralInfoReader are not available")

        self.name = self.__class__.__name__

        # Per-instance warning dedupe state. Previously these were class
        # attributes which persisted for the lifetime of the Python process,
        # silently suppressing real configuration warnings on the second
        # and subsequent evaluations in GUI / test / multi-eval contexts.
        self._custom_filter_warnings_shown: set = set()
        self._time_resolution_warning_shown: bool = False

        # Frequency mapping for time resolution normalization
        self.freq_map = {
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

        # Coordinate mapping for standardization (shared + local extensions)
        from openbench.data.coordinates import COORDINATE_MAP_WITH_VERTICAL

        self.coordinate_map = dict(COORDINATE_MAP_WITH_VERTICAL)

        self.sim_source = sim_source
        self.ref_source = ref_source

        # Initialize processing from task-local copies. Fallback varname
        # population below mutates self.sim_nml/self.ref_nml for legacy
        # compatibility, but must not pollute the shared namelists reused by
        # subsequent evaluation tasks.
        sim_nml = deepcopy(sim_nml)
        ref_nml = deepcopy(ref_nml)
        self._initialize_attributes(main_nl, sim_nml, ref_nml, item, sim_source, ref_source)
        self._set_evaluation_variables(metric_vars, score_vars, comparison_vars, statistic_vars)
        self._process_data_types()
        self._process_time_resolutions()
        self._process_station_data()

        # Handle missing variable names
        if self.sim_varname is None or self.sim_varname == "":
            logging.warning(f"Warning: sim_varname is not specified in namelist. Using item name: {self.item}")
            self.sim_varname = self.item
            self.sim_nml[self.item][f"{self.sim_source}_varname"] = self.item

        if self.ref_varname is None or self.ref_varname == "":
            logging.warning(f"Warning: ref_varname is not specified in namelist. Using item name: {self.item}")
            self.ref_varname = self.item
            self.ref_nml[self.item][f"{self.ref_source}_varname"] = self.item

    def _initialize_attributes(
        self,
        main_nl: Dict[str, Any],
        sim_nml: Dict[str, Any],
        ref_nml: Dict[str, Any],
        item: str,
        sim_source: str,
        ref_source: str,
    ):
        """Initialize class attributes from namelists."""
        self._safe_update(main_nl["general"])
        self._safe_update(ref_nml.get(item, {}))
        self._safe_update(sim_nml.get(item, {}))
        self.casedir = os.path.join(self.basedir, self.basename)
        self.sim_nml, self.ref_nml, self.item = sim_nml, ref_nml, item
        self._set_source_attributes(sim_nml, ref_nml, item, sim_source, ref_source)
        self.min_year = main_nl["general"].get("min_year", 1)  # Default to 1 if not specified

    def _set_evaluation_variables(
        self, metric_vars: list, score_vars: list, comparison_vars: list, statistic_vars: list
    ):
        """Set evaluation-related variables."""
        self.metrics, self.scores = metric_vars, score_vars
        self.comparisons, self.statistics = comparison_vars, statistic_vars

    def _set_source_attributes(
        self, sim_nml: Dict[str, Any], ref_nml: Dict[str, Any], item: str, sim_source: str, ref_source: str
    ):
        """Set attributes specific to simulation and reference sources."""
        for source_type in ["ref", "sim"]:
            source = ref_source if source_type == "ref" else sim_source
            nml = ref_nml if source_type == "ref" else sim_nml
            attributes = [
                "data_type",
                "varname",
                "varunit",
                "data_groupby",
                "dir",
                "tim_res",
                "grid_res",
                "syear",
                "eyear",
                "convert",
            ]
            # Preserve native numeric types for attributes used arithmetically
            # downstream (year ranges, grid resolution). Coercing them to str
            # turned `max(ref_syear, sim_syear)` into lexicographic comparison
            # and `float(ref_grid_res)` into a runtime ValueError.
            numeric_attrs = {"syear", "eyear", "grid_res"}
            for attr in attributes:
                value = nml[item].get(f"{source}_{attr}")
                if value is None:
                    setattr(self, f"{source_type}_{attr}", "")
                elif attr in numeric_attrs and isinstance(value, (int, float, bool)):
                    setattr(self, f"{source_type}_{attr}", value)
                else:
                    setattr(self, f"{source_type}_{attr}", str(value))

            # Handle suffix and prefix separately to ensure they're always strings
            for attr in ["suffix", "prefix"]:
                value = nml[item].get(f"{source}_{attr}")
                setattr(self, f"{source_type}_{attr}", str(value) if value is not None else "")

            # Handle station data
            if nml[item].get(f"{source}_data_type") == "stn":
                try:
                    setattr(self, f"{source_type}_fulllist", str(nml[item][f"{source}_fulllist"]))
                except (KeyError, TypeError):
                    try:
                        setattr(self, f"{source_type}_fulllist", str(nml["general"][f"{source}_fulllist"]))
                    except (KeyError, TypeError) as e2:
                        setattr(self, f"{source_type}_fulllist", "")
                        logging.error(f"read {source_type}_fulllist namelist error: {e2}")

            # Handle uparea attributes for station data
            try:
                setattr(self, f"{source_type}_max_uparea", str(nml[item][f"{source}_max_uparea"]))
                setattr(self, f"{source_type}_min_uparea", str(nml[item][f"{source}_min_uparea"]))
            except (KeyError, TypeError, AttributeError):
                pass  # Optional attributes, skip if not present

    def _process_data_types(self):
        """Process and validate data types."""
        # Must mirror loader.VALID_SIM_DATA_TYPES — a third value here
        # would let unsupported types pass legacy validation that the
        # loader would correctly reject.
        valid_types = ["grid", "stn"]
        for data_type in [self.ref_data_type, self.sim_data_type]:
            if data_type not in valid_types:
                logging.warning(f"Unknown data type: {data_type}. Valid types: {valid_types}")

    def _process_time_resolutions(self):
        """Process and normalize time resolutions."""
        # Handle special case for GRDC data with missing time resolution
        if (
            self.ref_source == "GRDC"
            and self.ref_data_type == "stn"
            and (not self.ref_tim_res or self.ref_tim_res == "")
        ):
            # Set GRDC time resolution to match comparison resolution
            self.ref_tim_res = getattr(self, "compare_tim_res", "D")
            logging.info(f"GRDC reference time resolution set to comparison resolution: {self.ref_tim_res}")

        self.ref_tim_res_normalized, self.ref_freq = self._normalize_time_resolution(self.ref_tim_res)
        self.sim_tim_res_normalized, self.sim_freq = self._normalize_time_resolution(self.sim_tim_res)
        self._check_time_resolution_consistency()
        self._set_use_years()

    def _normalize_time_resolution(self, tim_res: str) -> Tuple[str, str]:
        """
        Normalize time resolution string.

        Args:
            tim_res: Time resolution string

        Returns:
            Tuple of (normalized_resolution, frequency_code)
        """
        if not tim_res:
            return "", ""

        tim_res = tim_res.lower().strip()

        # Handle numeric prefixes (e.g., "3hour" -> "3H")
        match = re.match(r"(\d+)(.+)", tim_res)
        if match:
            number, unit = match.groups()
            freq_code = self.freq_map.get(unit.lower(), unit)
            return f"{number}{freq_code}", f"{number}{freq_code}"

        # Handle standard resolutions
        freq_code = self.freq_map.get(tim_res, tim_res)
        return tim_res, freq_code

    def _check_time_resolution_consistency(self):
        """Check if time resolutions are consistent and valid."""
        if not self._is_valid_resolution(self.ref_freq) or not self._is_valid_resolution(self.sim_freq):
            logging.warning(f"Invalid time resolution detected: ref={self.ref_freq}, sim={self.sim_freq}")

        # Check if resolutions are compatible
        if self.ref_freq != self.sim_freq:
            logging.info(f"Different time resolutions: ref={self.ref_freq}, sim={self.sim_freq}")
            # Try to determine if they can be aligned
            try:
                ref_td = self._resolution_to_timedelta(self.ref_freq)
                sim_td = self._resolution_to_timedelta(self.sim_freq)
                if ref_td != sim_td:
                    # Only show warning once per instance
                    if not self._time_resolution_warning_shown:
                        logging.warning("Time resolution mismatch may cause alignment issues")
                        self._time_resolution_warning_shown = True
            except (ValueError, AttributeError, TypeError) as e:
                logging.warning(f"Could not compare time resolutions: {self.ref_freq} vs {self.sim_freq} ({e})")

    def _is_valid_resolution(self, resolution: str) -> bool:
        """Check if a resolution string is valid."""
        return bool(
            resolution
            and (
                resolution in self.freq_map.values()
                or any(resolution.endswith(freq) for freq in self.freq_map.values())
            )
        )

    def _resolution_to_timedelta(self, resolution: str) -> pd.Timedelta:
        """Convert resolution string to pandas Timedelta."""
        if not resolution:
            raise ValueError("Empty resolution string")

        # Handle frequency codes
        freq_map = {"H": "hours", "D": "days", "M": "months", "Y": "years", "W": "weeks"}

        match = re.match(r"(\d*)([HDMYW])", resolution)
        if match:
            number, unit = match.groups()
            number = int(number) if number else 1
            unit_name = freq_map.get(unit, unit)

            # Handle months and years specially
            if unit == "M":
                return pd.Timedelta(days=30 * number)  # Approximate
            elif unit == "Y":
                return pd.Timedelta(days=365 * number)  # Approximate
            else:
                return pd.Timedelta(**{unit_name: number})

        raise ValueError(f"Cannot parse resolution: {resolution}")

    def _process_station_data(self):
        """Process station data if applicable."""

        if self.ref_data_type == "stn" or self.sim_data_type == "stn":
            try:
                self._read_and_merge_station_lists()
            except Exception as e:
                # Preserve the original I/O / parse error in self for the
                # downstream "No stations selected" raise so the user sees
                # the real cause (missing file, bad CSV, mismatched columns)
                # rather than a misleading empty-list message.
                self._station_list_error = e
                logging.error(f"Error processing station data: {e}")
                # Set empty dataframe as fallback
                self.stn_list = pd.DataFrame()
            self._filter_stations()

            # Also initialize stn_info for backward compatibility
            if hasattr(self, "stn_list"):
                self.stn_info = self.stn_list
            else:
                self.stn_info = pd.DataFrame()

    @cached(key_prefix="station_lists", ttl=3600)
    def _read_and_merge_station_lists(self):
        """Read and merge station lists from reference and simulation sources.

        Note: If fulllist is empty, skip file reading and let custom filters
        populate the station list (e.g., GRDC_filter, CaMa_filter).
        """
        if self.ref_data_type == "stn" and self.sim_data_type == "stn":
            # Both ref and sim are station data
            # Only read if fulllist paths are provided
            if self.sim_fulllist and self.ref_fulllist:
                self.sim_stn_list = pd.read_csv(self.sim_fulllist, header=0)
                self.ref_stn_list = pd.read_csv(self.ref_fulllist, header=0)
                self._rename_station_columns()
                sim_root = getattr(self, "sim_dir", "")
                ref_root = getattr(self, "ref_dir", "")
                self._resolve_relative_paths(self.sim_stn_list, "sim_dir", self.sim_fulllist, sim_root)
                self._resolve_relative_paths(self.ref_stn_list, "ref_dir", self.ref_fulllist, ref_root)
                self.stn_list = self._match_station_lists(self.sim_stn_list, self.ref_stn_list)
            else:
                # Empty fulllist - rely on custom filter to populate station list
                logging.debug("fulllist is empty, will rely on custom filter to populate station list")
                self.stn_list = pd.DataFrame()
        elif self.sim_data_type == "stn":
            # Only sim is station data
            if self.sim_fulllist:
                self.sim_stn_list = pd.read_csv(self.sim_fulllist, header=0)
                self._rename_station_columns(sim_only=True)
                sim_root = getattr(self, "sim_dir", "")
                self._resolve_relative_paths(self.sim_stn_list, "sim_dir", self.sim_fulllist, sim_root)
                self.stn_list = self.sim_stn_list
            else:
                logging.debug("sim fulllist is empty, will rely on custom filter to populate station list")
                self.stn_list = pd.DataFrame()
        elif self.ref_data_type == "stn":
            # Only ref is station data (stn×grid case)
            if self.ref_fulllist:
                self.ref_stn_list = pd.read_csv(self.ref_fulllist, header=0)
                self._rename_station_columns(ref_only=True)
                ref_root = getattr(self, "ref_dir", "")
                self._resolve_relative_paths(self.ref_stn_list, "ref_dir", self.ref_fulllist, ref_root)
                self.stn_list = self.ref_stn_list
                # use_syear/use_eyear will be computed by _apply_default_filter()
                # Preserve Flag from CSV; default to True if missing
                if "Flag" not in self.stn_list.columns:
                    self.stn_list["Flag"] = True
            else:
                logging.debug("ref fulllist is empty, will rely on custom filter to populate station list")
                self.stn_list = pd.DataFrame()

    @staticmethod
    def _match_station_lists(
        sim_stn: pd.DataFrame, ref_stn: pd.DataFrame, spatial_threshold_deg: float = 0.01
    ) -> pd.DataFrame:
        """Match two station lists for stn×stn evaluation.

        Strategy:
        1. Inner join on ID (exact match) — standard, fast
        2. If ID match yields nothing, fall back to nearest-neighbor spatial matching
           within `spatial_threshold_deg` (~1 km at equator)

        Returns merged DataFrame with both sim and ref columns.
        """
        # 1. Try ID match
        merged = pd.merge(sim_stn, ref_stn, how="inner", on="ID")
        if len(merged) > 0:
            n_sim = len(sim_stn)
            n_ref = len(ref_stn)
            logging.info(
                "Station matching by ID: %d matches (sim=%d, ref=%d)",
                len(merged),
                n_sim,
                n_ref,
            )
            return merged

        # 2. Fallback: spatial nearest-neighbor matching
        logging.warning(
            "No ID matches between sim (%d stations) and ref (%d stations). "
            "Attempting spatial matching (threshold=%.4f°)...",
            len(sim_stn),
            len(ref_stn),
            spatial_threshold_deg,
        )

        # Find lat/lon columns
        sim_lat = sim_stn.get("sim_lat", sim_stn.get("lat"))
        sim_lon = sim_stn.get("sim_lon", sim_stn.get("lon"))
        ref_lat = ref_stn.get("ref_lat", ref_stn.get("lat"))
        ref_lon = ref_stn.get("ref_lon", ref_stn.get("lon"))

        if sim_lat is None or sim_lon is None or ref_lat is None or ref_lon is None:
            logging.error("Cannot perform spatial matching: lat/lon columns missing")
            return pd.DataFrame()

        import numpy as np

        rows = []
        ref_lats = ref_lat.values.astype(float)
        ref_lons = ref_lon.values.astype(float)

        for idx, sim_row in sim_stn.iterrows():
            slat = float(sim_row.get("sim_lat", sim_row.get("lat", np.nan)))
            slon = float(sim_row.get("sim_lon", sim_row.get("lon", np.nan)))
            if np.isnan(slat) or np.isnan(slon):
                continue

            # Distance in degrees (approximate, fast)
            dist = np.sqrt((ref_lats - slat) ** 2 + (ref_lons - slon) ** 2)
            min_idx = np.argmin(dist)
            min_dist = dist[min_idx]

            if min_dist <= spatial_threshold_deg:
                # Merge sim and ref rows
                ref_row = ref_stn.iloc[min_idx]
                combined = {}
                combined["ID"] = sim_row.get("ID", ref_row.get("ID", f"stn_{len(rows)}"))
                for col in sim_stn.columns:
                    if col != "ID":
                        combined[col] = sim_row[col]
                for col in ref_stn.columns:
                    if col != "ID" and col not in combined:
                        combined[col] = ref_row[col]
                rows.append(combined)

        if rows:
            result = pd.DataFrame(rows)
            logging.info("Spatial matching: %d matches found", len(result))
            return result
        else:
            logging.warning("Spatial matching found no matches within threshold")
            return pd.DataFrame()

    @staticmethod
    def _resolve_relative_paths(df: pd.DataFrame, col: str, csv_path: str, root_dir: str = "") -> None:
        """Resolve relative paths in a DataFrame column.

        Strategy: if the path is relative, try to find the file by:
        1. Resolving against the CSV directory
        2. If not found, use root_dir + basename (station files are often directly in root_dir)
        """
        if col not in df.columns or not csv_path:
            return
        csv_dir = os.path.dirname(os.path.abspath(csv_path))

        def _resolve(p):
            if pd.isna(p) or os.path.isabs(str(p)):
                return p
            # Try 1: resolve against CSV dir
            resolved = os.path.normpath(os.path.join(csv_dir, p))
            if os.path.exists(resolved):
                return resolved
            # Try 2: root_dir + basename
            if root_dir:
                resolved2 = os.path.join(root_dir, os.path.basename(p))
                if os.path.exists(resolved2):
                    return resolved2
            # Fallback: return the CSV-dir resolution (will fail later with clear error)
            return resolved

        df[col] = df[col].apply(_resolve)

    def _rename_station_columns(self, sim_only=False, ref_only=False):
        """Rename station columns to standard names."""

        def _ensure_side_time_columns(df: pd.DataFrame, side: str) -> None:
            syear_col = f"{side}_syear"
            eyear_col = f"{side}_eyear"
            if syear_col not in df.columns and "use_syear" in df.columns:
                df[syear_col] = df["use_syear"]
            if eyear_col not in df.columns and "use_eyear" in df.columns:
                df[eyear_col] = df["use_eyear"]

        if not ref_only and hasattr(self, "sim_stn_list"):
            self.sim_stn_list.rename(
                columns={
                    "SYEAR": "sim_syear",
                    "EYEAR": "sim_eyear",
                    "DIR": "sim_dir",
                    "LON": "sim_lon",
                    "LAT": "sim_lat",
                },
                inplace=True,
            )
            _ensure_side_time_columns(self.sim_stn_list, "sim")
        if not sim_only and hasattr(self, "ref_stn_list"):
            self.ref_stn_list.rename(
                columns={
                    "SYEAR": "ref_syear",
                    "EYEAR": "ref_eyear",
                    "DIR": "ref_dir",
                    "LON": "ref_lon",
                    "LAT": "ref_lat",
                },
                inplace=True,
            )
            _ensure_side_time_columns(self.ref_stn_list, "ref")

        # Also rename in combined station list if it exists
        if hasattr(self, "stn_list") and not self.stn_list.empty:
            # Standard column mapping for coordinates and other fields
            from openbench.data.coordinates import VERTICAL_COORDINATE_MAP

            column_mapping = {
                "id": "ID",
                "Id": "ID",
                "site_id": "ID",
                "site": "ID",
                "longitude": "lon",
                "Longitude": "lon",
                "LON": "lon",
                "Long": "lon",
                "latitude": "lat",
                "Latitude": "lat",
                "LAT": "lat",
                "Lat": "lat",
            }
            column_mapping.update(VERTICAL_COORDINATE_MAP)

            # Apply column renaming
            for old_name, new_name in column_mapping.items():
                if old_name in self.stn_list.columns:
                    self.stn_list.rename(columns={old_name: new_name}, inplace=True)

            # Handle side-specific station columns based on data types.
            if self.ref_data_type == "stn" and self.sim_data_type != "stn":
                self.stn_list.rename(
                    columns={"DIR": "ref_dir", "SYEAR": "ref_syear", "EYEAR": "ref_eyear"},
                    inplace=True,
                )
            elif self.sim_data_type == "stn" and self.ref_data_type != "stn":
                self.stn_list.rename(
                    columns={"DIR": "sim_dir", "SYEAR": "sim_syear", "EYEAR": "sim_eyear"},
                    inplace=True,
                )
            # Generated station lists commonly carry use_syear/use_eyear rather
            # than side-specific names; mirror them before filtering so the
            # branches below do not fail with KeyError.
            if self.ref_data_type == "stn":
                _ensure_side_time_columns(self.stn_list, "ref")
            if self.sim_data_type == "stn":
                _ensure_side_time_columns(self.stn_list, "sim")

    def _filter_stations(self):
        """Filter stations based on criteria."""
        if not hasattr(self, "stn_list") or self.stn_list is None:
            self.stn_list = pd.DataFrame()

        if self.ref_source.lower() != "grdc" and self.stn_list.empty:
            logging.debug("No station list available for filtering; attempting to generate one.")

        initial_count = len(self.stn_list)
        # Get custom filter if available
        custom_filter = self._get_custom_filter()
        if custom_filter is not None:
            logging.info(f"Applying custom filter for {self.ref_source}")
            try:
                custom_filter(self)
            except Exception as e:
                logging.error(f"Custom filter failed: {e}")
                self._apply_default_filter()
        else:
            # Apply default filters
            self._apply_default_filter()

        # Save the filtered station list
        if hasattr(self, "stn_list"):
            if self.stn_list.empty:
                cause = getattr(self, "_station_list_error", None)
                message = "No stations selected. Check filter criteria."
                if cause is not None:
                    message = f"{message} Station list error: {cause}"
                logging.error(message)
                raise ValueError(message)

            # Always write to a case-specific path to avoid overwriting the original ref CSV
            stn_list_path = os.path.join(self.casedir, f"stn_{self.ref_source}_{self.sim_source}_list.csv")
            os.makedirs(os.path.dirname(stn_list_path), exist_ok=True)
            self.ref_fulllist = stn_list_path
            self.stn_list.to_csv(stn_list_path, index=False)
            final_count = len(self.stn_list)
            logging.info(f"Station filtering: {initial_count} -> {final_count} stations")

    def _get_custom_filter(self):
        """Get custom filter function for the reference source.

        Priority:
        1. station_matching config in reference_catalog.yaml → StationMatcher engine
        2. Python filter file (built-in or user ~/.openbench/custom/)
        3. None → default filter
        """
        # Priority 1: station_matching config from reference catalog
        try:
            from openbench.data.registry.manager import get_registry

            mgr = get_registry()
            ref = mgr.get_reference(self.ref_source)
            if ref and ref.station_matching:
                sm = ref.station_matching

                def _station_matcher_filter(info):
                    from pathlib import Path
                    from openbench.data.station_matcher import run_station_matching

                    dataset_path = str(Path(info.ref_dir) / sm.dataset_file)
                    run_station_matching(
                        info,
                        dataset_path,
                        method=sm.method,
                        station_id_var=sm.station_id_var,
                        lon_var=sm.lon_var,
                        lat_var=sm.lat_var,
                        area_var=sm.area_var,
                        discharge_var=sm.discharge_var,
                        time_var=sm.time_var,
                        area_error_threshold=sm.area_error_threshold,
                        min_uparea=sm.min_uparea,
                        max_uparea=sm.max_uparea,
                        time_format=sm.time_format,
                    )

                return _station_matcher_filter
        except Exception as e:
            logging.debug("station_matching lookup failed for %s: %s", self.ref_source, e)

        # Priority 2: Python filter file
        try:
            from openbench.data.custom import load_filter

            filter_module = load_filter(self.ref_source)
            if filter_module:
                func = getattr(filter_module, f"filter_{self.ref_source}", None)
                if func:
                    return func
        except Exception:
            pass

        # Priority 3: no filter (warn once per instance per ref_source)
        if self.ref_source not in self._custom_filter_warnings_shown:
            logging.warning(f"Custom filter for {self.ref_source} not available. Using default filter.")
            self._custom_filter_warnings_shown.add(self.ref_source)
        return None

    def _apply_default_filter(self):
        """Apply default station filtering criteria.

        Station data always uses per-station time intersection (max starts,
        min ends) — the time_alignment option only affects grid evaluations.
        """
        if not hasattr(self, "stn_list") or self.stn_list.empty:
            return

        default_syear = self._safe_int(self.syear, 1990)
        default_eyear = self._safe_int(self.eyear, 2020)
        n = len(self.stn_list)

        if self.ref_data_type == "stn" and self.sim_data_type == "stn":
            sim_years = pd.to_numeric(self.stn_list["sim_syear"], errors="coerce")
            ref_years = pd.to_numeric(self.stn_list["ref_syear"], errors="coerce")
            syear_series = pd.Series([default_syear] * n)
            self.use_syear = pd.concat([sim_years, ref_years, syear_series], axis=1).max(axis=1)

            sim_eyears = pd.to_numeric(self.stn_list["sim_eyear"], errors="coerce")
            ref_eyears = pd.to_numeric(self.stn_list["ref_eyear"], errors="coerce")
            eyear_series = pd.Series([default_eyear] * n)
            self.use_eyear = pd.concat([sim_eyears, ref_eyears, eyear_series], axis=1).min(axis=1)

        elif self.sim_data_type == "stn":
            sim_years = pd.to_numeric(self.stn_list["sim_syear"], errors="coerce")
            ref_syear_val = self._safe_int(self.ref_syear, default_syear)
            ref_syear = pd.Series([ref_syear_val] * n)
            syear_series = pd.Series([default_syear] * n)
            self.use_syear = pd.concat([sim_years, ref_syear, syear_series], axis=1).max(axis=1)

            sim_eyears = pd.to_numeric(self.stn_list["sim_eyear"], errors="coerce")
            ref_eyear_val = self._safe_int(self.ref_eyear, default_eyear)
            ref_eyear = pd.Series([ref_eyear_val] * n)
            eyear_series = pd.Series([default_eyear] * n)
            self.use_eyear = pd.concat([sim_eyears, ref_eyear, eyear_series], axis=1).min(axis=1)

        elif self.ref_data_type == "stn":
            ref_years = pd.to_numeric(self.stn_list["ref_syear"], errors="coerce")
            sim_syear_val = self._safe_int(self.sim_syear, default_syear)
            sim_syear = pd.Series([sim_syear_val] * n)
            syear_series = pd.Series([default_syear] * n)
            self.use_syear = pd.concat([ref_years, sim_syear, syear_series], axis=1).max(axis=1)

            ref_eyears = pd.to_numeric(self.stn_list["ref_eyear"], errors="coerce")
            sim_eyear_val = self._safe_int(self.sim_eyear, default_eyear)
            sim_eyear = pd.Series([sim_eyear_val] * n)
            eyear_series = pd.Series([default_eyear] * n)
            self.use_eyear = pd.concat([ref_eyears, sim_eyear, eyear_series], axis=1).min(axis=1)

        self.stn_list["use_syear"] = self.use_syear
        self.stn_list["use_eyear"] = self.use_eyear

        # Apply basic filtering criteria based on time range validity
        # Only select stations where the time range is valid and meaningful
        valid_time_range = (self.stn_list["use_eyear"] - self.stn_list["use_syear"]) >= 0
        self.stn_list["Flag"] = valid_time_range

        # Apply geographical filters if available
        # Check for different possible longitude column names
        lon_col = None
        for col in ["lon", "LON", "longitude", "Longitude"]:
            if col in self.stn_list.columns:
                lon_col = col
                break

        lat_col = None
        for col in ["lat", "LAT", "latitude", "Latitude"]:
            if col in self.stn_list.columns:
                lat_col = col
                break

        if hasattr(self, "min_lon") and hasattr(self, "max_lon") and lon_col:
            lon_filter = (self.stn_list[lon_col] >= float(self.min_lon)) & (
                self.stn_list[lon_col] <= float(self.max_lon)
            )
            self.stn_list["Flag"] = self.stn_list["Flag"] & lon_filter

        if hasattr(self, "min_lat") and hasattr(self, "max_lat") and lat_col:
            lat_filter = (self.stn_list[lat_col] >= float(self.min_lat)) & (
                self.stn_list[lat_col] <= float(self.max_lat)
            )
            self.stn_list["Flag"] = self.stn_list["Flag"] & lat_filter

        # Apply minimum year criteria if available
        if hasattr(self, "min_year") and self.min_year:
            try:
                min_year_val = int(self.min_year)
                available_years = self.stn_list["use_eyear"] - self.stn_list["use_syear"] + 1
                year_filter = available_years >= min_year_val
                self.stn_list["Flag"] = self.stn_list["Flag"] & year_filter
            except (ValueError, AttributeError):
                pass

        # Filter by upstream area if available
        if hasattr(self, "ref_max_uparea") and hasattr(self, "ref_min_uparea"):
            try:
                max_uparea = float(self.ref_max_uparea) if self.ref_max_uparea else float("inf")
                min_uparea = float(self.ref_min_uparea) if self.ref_min_uparea else 0

                if "uparea" in self.stn_list.columns:
                    uparea_filter = (self.stn_list["uparea"] >= min_uparea) & (self.stn_list["uparea"] <= max_uparea)
                    self.stn_list["Flag"] = self.stn_list["Flag"] & uparea_filter
            except (ValueError, AttributeError):
                pass

        # For grid reference data (like GLEAM4.2a) used with station simulation data,
        # apply additional validation to ensure data availability
        if self.ref_data_type == "grid" and self.sim_data_type == "stn":
            # Check if reference data covers the station locations and time period
            # This is more conservative than flagging all stations as True
            ref_sy = self._safe_int(self.ref_syear, 1900)
            ref_ey = self._safe_int(self.ref_eyear, 2100)
            ref_time_coverage = (self.stn_list["use_syear"] >= ref_sy) & (self.stn_list["use_eyear"] <= ref_ey)
            self.stn_list["Flag"] = self.stn_list["Flag"] & ref_time_coverage

        # Keep only flagged stations
        self.stn_list = self.stn_list[self.stn_list["Flag"]]
        logging.info(f"Total number of stations selected: {len(self.stn_list)}")

        # Fail fast if no stations are selected; downstream processing cannot
        # evaluate an empty station list usefully.
        if len(self.stn_list) == 0:
            logging.error("No stations selected after filtering. Check filter criteria.")
            logging.error(f"Reference data type: {self.ref_data_type}, time range: {self.ref_syear}-{self.ref_eyear}")
            logging.error(f"Simulation data type: {self.sim_data_type}")
            raise ValueError("No stations selected after filtering. Check filter criteria.")

    @staticmethod
    def _safe_int(value, default=None):
        """Safely convert a value to integer, returning default if conversion fails."""
        if value is None or value == "" or (isinstance(value, str) and value.strip() == ""):
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def _set_use_years(self):
        """Set the use years based on the evaluation timeframe and time_alignment mode.

        Modes:
          - intersection (default): max(starts), min(ends) across ref, sim, config
          - per_pair: ref∩sim∩config for this specific pair (same formula, but the
            result is per-pair because each sim has its own syear/eyear)
          - strict: use config years exactly; warn if ref or sim doesn't fully cover
        """
        time_alignment = getattr(self, "time_alignment", "intersection")

        ref_sy = self._safe_int(self.ref_syear)
        sim_sy = self._safe_int(self.sim_syear)
        gen_sy = self._safe_int(self.syear, 1990)

        ref_ey = self._safe_int(self.ref_eyear)
        sim_ey = self._safe_int(self.sim_eyear)
        gen_ey = self._safe_int(self.eyear, 2020)

        if time_alignment == "strict":
            # Use config years exactly; validate coverage
            self.use_syear = gen_sy
            self.use_eyear = gen_ey
            if ref_sy is not None and ref_sy > gen_sy:
                logging.warning("strict mode: ref starts at %d but config requires %d", ref_sy, gen_sy)
            if ref_ey is not None and ref_ey < gen_ey:
                logging.warning("strict mode: ref ends at %d but config requires %d", ref_ey, gen_ey)
            if sim_sy is not None and sim_sy > gen_sy:
                logging.warning("strict mode: sim starts at %d but config requires %d", sim_sy, gen_sy)
            if sim_ey is not None and sim_ey < gen_ey:
                logging.warning("strict mode: sim ends at %d but config requires %d", sim_ey, gen_ey)
        else:
            # intersection and per_pair: same formula (max starts, min ends)
            # For per_pair the difference is that the runner calls this once per
            # sim-ref pair with that pair's own syear/eyear, so the result is
            # naturally per-pair.  For intersection across multiple sims, the
            # runner's _preprocess_variable iterates serially and the minyear/
            # maxyear from the first pair is reused (existing behaviour).
            syear_values = [v for v in [ref_sy, sim_sy, gen_sy] if v is not None]
            eyear_values = [v for v in [ref_ey, sim_ey, gen_ey] if v is not None]

            self.use_syear = max(syear_values) if syear_values else 1990
            self.use_eyear = min(eyear_values) if eyear_values else 2020

    # Removed dead methods: `to_dict`, `_check_station_file`, and the
    # `@cached`-decorated `_process_station_batch` had zero call sites
    # across the codebase. Their continued presence would (a) keep the
    # `joblib.Parallel` + `cached` decorator import surface live for no
    # benefit and (b) silently come back to life if anyone ever called
    # `_process_station_batch` and ran into the broken cache key state
    # documented in the audit notes.
