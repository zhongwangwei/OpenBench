"""Configuration, lifecycle, and dispatch helpers for dataset processing."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict

import pandas as pd
import xarray as xr

from openbench.data._processing_utils import performance_monitor
from openbench.data._system_resources import calculate_optimal_chunk_size, calculate_optimal_cores, get_system_resources
from openbench.util.converttype import Convert_Type
from openbench.util.netcdf import write_file_atomic

try:
    pd_version = tuple(int(x) for x in pd.__version__.split(".")[:2])
    USE_NEW_FREQ_ALIASES = pd_version >= (2, 2)
except (AttributeError, ValueError, IndexError):
    USE_NEW_FREQ_ALIASES = False

_HAS_INTERFACES = True


class ProcessingConfigMixin:
    def initialize_resource_parameters(self):
        """Initialize resource-related parameters based on system capabilities."""
        # Get system resources
        resources = get_system_resources()

        # Set default num_cores if not specified
        if not hasattr(self, "num_cores") or self.num_cores <= 0:
            self.num_cores = resources["cpu_count"]

        # Store resource information
        self.system_resources = resources

        # Default chunk size (will be adjusted per operation)
        self.default_chunks = {"time": "auto", "lat": "auto", "lon": "auto"}

    @staticmethod
    def validate_year(year, default=None, min_year=1900, max_year=2100):
        """Validate and convert year value to integer within valid range.

        Args:
            year: Year value to validate (can be int, float, str, or None)
            default: Default value if year is invalid
            min_year: Minimum valid year (default 1900)
            max_year: Maximum valid year (default 2100)

        Returns:
            Valid integer year value
        """
        # Handle None, empty string, or whitespace
        if year is None or (isinstance(year, str) and year.strip() == ""):
            return default if default is not None else min_year

        # Try to convert to integer
        try:
            year_int = int(float(year))
        except (ValueError, TypeError):
            logging.warning(f"Invalid year value: {year}. Using default: {default if default else min_year}")
            return default if default is not None else min_year

        # Validate range
        if year_int < min_year:
            logging.warning(f"Year {year_int} is before {min_year}. Adjusting to {min_year}.")
            return min_year
        if year_int > max_year:
            logging.warning(f"Year {year_int} is after {max_year}. Adjusting to {max_year}.")
            return max_year

        return year_int

    def get_optimal_chunks(self, dataset_size_gb: float) -> Dict[str, str]:
        """
        Get optimal chunk size for a dataset.

        Args:
            dataset_size_gb (float): Size of the dataset in GB

        Returns:
            dict: Optimal chunk sizes
        """
        return calculate_optimal_chunk_size(dataset_size_gb, self.system_resources["available_memory_gb"])

    def get_optimal_cores(self, dataset_size_gb: float) -> int:
        """
        Get optimal number of cores for processing.

        Args:
            dataset_size_gb (float): Size of the dataset in GB

        Returns:
            int: Optimal number of cores
        """
        return calculate_optimal_cores(
            self.system_resources["cpu_count"], self.system_resources["available_memory_gb"], dataset_size_gb
        )

    def _convert_legacy_freq_alias(self, freq: str) -> str:
        """
        Convert legacy pandas frequency aliases to new ones if needed.

        Args:
            freq (str): Frequency string that might contain legacy aliases

        Returns:
            str: Updated frequency string with new aliases if applicable
        """
        if not USE_NEW_FREQ_ALIASES:
            return freq

        # Map of legacy to new frequency aliases
        legacy_to_new = {
            "M": "ME",  # Month end
            "Y": "YE",  # Year end
            "Q": "QE",  # Quarter end
            "H": "h",  # Hour
            "T": "min",  # Minute
            "S": "s",  # Second
            "L": "ms",  # Millisecond
            "U": "us",  # Microsecond
            "N": "ns",  # Nanosecond
        }

        # Handle compound frequencies like '3M' -> '3ME'

        pattern = r"(\d*)([A-Z])"

        def replacer(match):
            number = match.group(1)
            letter = match.group(2)
            new_letter = legacy_to_new.get(letter, letter)
            return number + new_letter

        return re.sub(pattern, replacer, freq)

    def _is_climatology_mode(self) -> bool:
        """
        Check if compare_tim_res indicates climatology mode.

        Returns:
            bool: True if in climatology mode (climatology-year or climatology-month)
        """
        if not hasattr(self, "compare_tim_res") or not self.compare_tim_res:
            return False
        compare_tim_res_str = str(self.compare_tim_res).strip().lower()
        return compare_tim_res_str in ["climatology-year", "climatology-month"]

    @staticmethod
    def _is_climatology_frequency_value(tim_res: str) -> bool:
        return str(tim_res or "").strip().lower() in {"climatology-year", "climatology-month"}

    def _normalize_frequency(self, freq: str) -> str:
        """
        Convert human-readable frequency strings to pandas-compatible codes.

        Args:
            freq (str): Input frequency string (e.g., 'month', 'day', 'hour')

        Returns:
            str: Pandas-compatible frequency code (e.g., 'M', 'D', 'H')
        """
        # Use appropriate frequency aliases based on pandas version
        if USE_NEW_FREQ_ALIASES:
            freq_map = {
                "month": "ME",  # Month End (new alias)
                "mon": "ME",
                "monthly": "ME",
                "day": "D",
                "daily": "D",
                "hour": "h",  # Hour (lowercase in new pandas)
                "Hour": "h",
                "hr": "h",
                "Hr": "h",
                "h": "h",
                "hourly": "h",
                "year": "YE",  # Year End (new alias)
                "yr": "YE",
                "yearly": "YE",
                "week": "W",
                "wk": "W",
                "weekly": "W",
            }
        else:
            freq_map = {
                "month": "M",  # Month (old alias)
                "mon": "M",
                "monthly": "M",
                "day": "D",
                "daily": "D",
                "hour": "H",  # Hour (uppercase in old pandas)
                "Hour": "H",
                "hr": "H",
                "Hr": "H",
                "h": "H",
                "hourly": "H",
                "year": "Y",  # Year (old alias)
                "yr": "Y",
                "yearly": "Y",
                "week": "W",
                "wk": "W",
                "weekly": "W",
            }

        # Convert to lowercase for case-insensitive matching
        normalized_freq = freq.lower().strip()

        # Get mapped frequency or use original if no mapping found
        result_freq = freq_map.get(normalized_freq, freq)

        # Don't convert if we already got a mapped frequency from freq_map
        # Only convert if we're returning the original frequency
        if result_freq == freq:
            result_freq = self._convert_legacy_freq_alias(result_freq)

        return result_freq

    def initialize_attributes(self, config: Dict[str, Any]) -> None:
        # Set default values for optional config keys before updating
        self.debug_mode = False  # Default debug_mode to False
        for key, value in config.items():
            if key.startswith("__") or callable(getattr(self, key, None)):
                logging.debug("Skipping protected/method key in config: %s", key)
                continue
            setattr(self, key, value)
        self.sim_varname = [self.sim_varname] if isinstance(self.sim_varname, str) else self.sim_varname
        self.ref_varname = [self.ref_varname] if isinstance(self.ref_varname, str) else self.ref_varname
        # Handle both single values and Series for use_syear and use_eyear
        if hasattr(self.use_syear, "iloc"):
            self.minyear = int(self.use_syear.min())
            self.maxyear = int(self.use_eyear.max())
        else:
            self.minyear = int(self.use_syear)
            self.maxyear = int(self.use_eyear)

        essential_attrs = ["sim_tim_res", "ref_tim_res", "compare_tim_res"]
        for attr in essential_attrs:
            if not hasattr(self, attr):
                setattr(self, attr, config.get(attr, "M"))
                if self.debug_mode:
                    logging.warning(
                        f"Warning: '{attr}' was not provided in the config. Using value from 'tim_res': {getattr(self, attr)}"
                    )

        # Apply frequency normalization to timing resolution attributes
        if hasattr(self, "compare_tim_res"):
            original_freq = self.compare_tim_res
            self.compare_tim_res = self._normalize_frequency(self.compare_tim_res)
            if self.compare_tim_res != original_freq:
                logging.debug(f"Normalized frequency: {original_freq} -> {self.compare_tim_res}")

        # Also normalize other timing resolution attributes
        for attr in ["sim_tim_res", "ref_tim_res"]:
            if hasattr(self, attr):
                original_freq = getattr(self, attr)
                normalized_freq = self._normalize_frequency(original_freq)
                setattr(self, attr, normalized_freq)
                if normalized_freq != original_freq:
                    logging.debug(f"Normalized {attr}: {original_freq} -> {normalized_freq}")

    def setup_output_directories(self) -> None:
        if self.ref_data_type == "stn" or self.sim_data_type == "stn":
            # Try to load station list from multiple sources:
            # 1. Explicit fulllist path (ref or sim)
            # 2. Previously generated list file
            # 3. Auto-scan sim directory (generates list on the fly)
            stnlist_path = None

            # Priority 1: explicit fulllist
            if hasattr(self, "ref_fulllist") and self.ref_fulllist and os.path.exists(self.ref_fulllist):
                stnlist_path = self.ref_fulllist
            elif hasattr(self, "sim_fulllist") and self.sim_fulllist and os.path.exists(self.sim_fulllist):
                stnlist_path = self.sim_fulllist

            # Priority 2: previously generated list
            if not stnlist_path:
                generated_path = os.path.join(self.casedir, f"stn_{self.ref_source}_{self.sim_source}_list.txt")
                if os.path.exists(generated_path):
                    stnlist_path = generated_path

            # Priority 3: auto-scan station simulation directory
            if not stnlist_path and self.sim_data_type == "stn":
                sim_dir = getattr(self, "sim_dir", "")
                if sim_dir and os.path.isdir(sim_dir):
                    stnlist_path = self._auto_generate_station_list(sim_dir)

            if stnlist_path and os.path.exists(stnlist_path):
                self.station_list = Convert_Type.convert_Frame(pd.read_csv(stnlist_path, header=0))
                canonical_list_path = os.path.join(self.casedir, f"stn_{self.ref_source}_{self.sim_source}_list.txt")
                os.makedirs(os.path.dirname(canonical_list_path), exist_ok=True)
                write_file_atomic(
                    canonical_list_path,
                    lambda tmp_path: self.station_list.to_csv(tmp_path, index=False),
                    suffix=".tmp.csv",
                )
                # Evaluation_stn rebuilds runtime info later and falls back to this
                # case-level path, so keep it in sync with the list used here.
                self.ref_fulllist = canonical_list_path
            else:
                logging.warning("No station list found; station processing may fail")
                self.station_list = pd.DataFrame()

            output_dir = os.path.join(self.casedir, "data", f"stn_{self.ref_source}_{self.sim_source}")
            os.makedirs(output_dir, exist_ok=True)

    def _auto_generate_station_list(self, sim_dir: str) -> str | None:
        """Auto-scan a station simulation directory and generate a station list CSV."""
        try:
            from openbench.data.station_scanner import scan_station_sim_dir

            merge_dir = os.path.join(self.casedir, "scratch", f"merged_{self.sim_source}")
            df = scan_station_sim_dir(sim_dir, output_dir=merge_dir)

            # Save as CSV for reuse
            list_path = os.path.join(self.casedir, f"stn_{self.ref_source}_{self.sim_source}_list.txt")
            os.makedirs(os.path.dirname(list_path), exist_ok=True)
            write_file_atomic(list_path, lambda tmp_path: df.to_csv(tmp_path, index=False), suffix=".tmp.csv")
            logging.info("Auto-generated station list: %s (%d stations)", list_path, len(df))
            return list_path
        except Exception as e:
            logging.warning("Failed to auto-generate station list from %s: %s", sim_dir, e)
            return None

    def get_data_params(self, datasource: str) -> Dict[str, Any]:
        # Note: prefix_fallback is read directly from instance attributes
        # by _get_prefix_fallback_list(), not passed through params dict.
        return {
            "data_dir": getattr(self, f"{datasource}_dir"),
            "data_groupby": getattr(self, f"{datasource}_data_groupby").lower(),
            "varname": getattr(self, f"{datasource}_varname"),
            "tim_res": getattr(self, f"{datasource}_tim_res"),
            "varunit": getattr(self, f"{datasource}_varunit"),
            "prefix": getattr(self, f"{datasource}_prefix"),
            "suffix": getattr(self, f"{datasource}_suffix"),
            "datasource": datasource,
            "data_type": getattr(self, f"{datasource}_data_type"),
            "syear": getattr(self, f"{datasource}_syear"),
            "eyear": getattr(self, f"{datasource}_eyear"),
            "convert": getattr(self, f"{datasource}_convert", ""),
        }

    @performance_monitor
    def prepare_source(self, datasource: str) -> None:
        """Prepare one configured datasource label for runner execution."""
        logging.debug(f"Processing {datasource} data")
        self._preprocess(datasource)
        logging.debug(f"{datasource.capitalize()} data prepared!")

    @performance_monitor
    def process(self, data: xr.Dataset = None, **kwargs) -> xr.Dataset:
        """
        Process an xarray Dataset through the active in-memory validation path.

        Args:
            data: Input xarray Dataset
            **kwargs: Additional parameters

        Returns:
            Processed dataset
        """
        if isinstance(data, xr.Dataset):
            if _HAS_INTERFACES and hasattr(super(), "validate_input"):
                if not self.validate_input(data):
                    raise ValueError("Input dataset validation failed")

            # Apply basic processing steps
            processed_data = self.check_dataset(data)
            processed_data = self.check_coordinate(processed_data)

            return processed_data

        raise ValueError("process() expects an xarray.Dataset; use prepare_source() for runner datasource labels")

    @performance_monitor
    def _preprocess(self, datasource: str) -> None:
        data_params = self.get_data_params(datasource)
        state_names = (
            f"{datasource}_varname",
            f"{datasource}_varunit",
            f"_fb_convert_{datasource}",
        )
        state_before = {name: getattr(self, name, None) for name in state_names}
        state_existed = {name: hasattr(self, name) for name in state_names}

        # Propagate adapter-resolved convert expression so select_var can apply it.
        # Clear stale conversion state when this datasource has no conversion;
        # otherwise a fallback/convert from a previous variable can leak into the
        # next sequential preprocess call on the same DatasetProcessing instance.
        convert_expr = data_params.get("convert", "")
        if convert_expr:
            setattr(self, f"_fb_convert_{datasource}", convert_expr)
        elif hasattr(self, f"_fb_convert_{datasource}"):
            delattr(self, f"_fb_convert_{datasource}")

        try:
            if data_params["data_type"] != "stn":
                logging.debug(f"Processing {data_params['data_type']} data")
                self.process_grid_data(data_params)
            else:
                self.process_station_data(data_params)
        finally:
            for name in state_names:
                if state_existed[name]:
                    setattr(self, name, state_before[name])
                elif hasattr(self, name):
                    delattr(self, name)

    def check_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        if not isinstance(ds, xr.Dataset):
            logging.error("Input data must be a xarray dataset.")
            raise ValueError("Input data must be a xarray dataset.")
        return ds

    def check_units(self, input_units: str, target_units: str) -> bool:
        input_units_list = sorted(input_units.split())
        target_units_list = sorted(target_units.split())
        return input_units_list == target_units_list

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__
