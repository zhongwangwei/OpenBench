# -*- coding: utf-8 -*-
"""
Data validation for NetCDF files.

Validates file existence, variable names, time range, and spatial range.
Supports both local and remote (SSH) validation.
"""

import json
import logging
import os
import shlex
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from openbench.gui.widgets._ssh_worker import execute_responsive
from openbench.gui.path_utils import to_absolute_path, get_openbench_root

logger = logging.getLogger(__name__)


def safe_open(path: str):
    """Open xarray dataset, trying decode_times=False if default fails.

    Args:
        path: Path to NetCDF file

    Returns:
        xarray.Dataset
    """
    import xarray as xr

    try:
        return xr.open_dataset(path)
    except Exception:
        return xr.open_dataset(path, decode_times=False)


# String version of safe_open for embedding in remote scripts
SAFE_OPEN_CODE = '''
def safe_open(path):
    """Open dataset, trying decode_times=False if default fails."""
    try:
        return xr.open_dataset(path)
    except Exception:
        return xr.open_dataset(path, decode_times=False)
'''


@dataclass
class ValidationCheck:
    """Single validation check result."""

    name: str
    passed: bool
    message: str


@dataclass
class SourceValidationResult:
    """Validation result for a single data source."""

    var_name: str
    source_name: str
    checks: List[ValidationCheck] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return True if all checks passed."""
        return all(check.passed for check in self.checks)

    @property
    def failed_checks(self) -> List[ValidationCheck]:
        """Return list of failed checks."""
        return [check for check in self.checks if not check.passed]


@dataclass
class DataValidationReport:
    """Complete validation report for all sources."""

    results: List[SourceValidationResult] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        """Total number of sources validated."""
        return len(self.results)

    @property
    def passed_count(self) -> int:
        """Number of sources that passed all checks."""
        return sum(1 for r in self.results if r.is_valid)

    @property
    def failed_count(self) -> int:
        """Number of sources with failed checks."""
        return sum(1 for r in self.results if not r.is_valid)


class FilePathGenerator:
    """Generate file paths based on data_groupby setting."""

    def __init__(
        self,
        root_dir: str,
        sub_dir: str,
        prefix: str,
        suffix: str,
        data_groupby: str,
        syear: int,
        eyear: int,
        is_remote: bool = False,
        ssh_manager=None,
        remote_openbench_root: str = "",
    ):
        self.root_dir = root_dir
        self.sub_dir = sub_dir
        self.prefix = prefix or ""
        self.suffix = suffix or ""
        self.data_groupby = data_groupby
        self.syear = syear
        self.eyear = eyear
        self._is_remote = is_remote
        self._ssh_manager = ssh_manager
        self._remote_openbench_root = remote_openbench_root
        self.last_error: str | None = None

    def _get_base_dir(self) -> str:
        """Get the base directory path (root_dir + sub_dir)."""
        if self._is_remote:
            # Remote mode: use forward slashes and remote root
            root = self.root_dir.replace("\\", "/")
            if self.sub_dir:
                sub = self.sub_dir.replace("\\", "/")
                path = f"{root.rstrip('/')}/{sub.lstrip('/')}"
            else:
                path = root

            # Convert relative path to absolute using remote OpenBench root
            if not path.startswith("/") and self._remote_openbench_root:
                if path.startswith("./"):
                    path = path[2:]
                path = f"{self._remote_openbench_root.rstrip('/')}/{path}"
            return path
        else:
            # Local mode
            if self.sub_dir:
                path = os.path.join(self.root_dir, self.sub_dir)
            else:
                path = self.root_dir
            # Convert to absolute path using OpenBench root as base
            return to_absolute_path(path, get_openbench_root())

    def _build_path(self, filename: str) -> str:
        """Build full path with root_dir and sub_dir."""
        base_dir = self._get_base_dir()
        if self._is_remote:
            return f"{base_dir.rstrip('/')}/{filename}"
        return os.path.join(base_dir, filename)

    def get_sample_paths(self) -> List[str]:
        """Get sample file paths for validation.

        Uses glob pattern to find actual files matching prefix and suffix.
        Returns a small set of representative paths to check.
        """
        base_dir = self._get_base_dir()

        if self.data_groupby == "Single":
            # Exact match for single file
            filename = f"{self.prefix}{self.suffix}.nc"
            return [self._build_path(filename)]

        # For Year/Month/Day, use glob to find matching files
        # Pattern: {prefix}*{suffix}.nc
        pattern = f"{self.prefix}*{self.suffix}.nc"

        if self._is_remote and self._ssh_manager:
            # Remote mode: use SSH to list files
            matching_files = self._remote_glob(base_dir, pattern)
        else:
            # Local mode: use local glob
            import glob

            full_pattern = os.path.join(base_dir, pattern)
            matching_files = sorted(glob.glob(full_pattern))

        if matching_files:
            # Return first, middle, and last file as samples
            if len(matching_files) == 1:
                return matching_files
            elif len(matching_files) == 2:
                return matching_files
            else:
                mid = len(matching_files) // 2
                return [matching_files[0], matching_files[mid], matching_files[-1]]

        # If no files found via glob, return empty list
        # The validation will report "no files found"
        return []

    def _remote_glob(self, base_dir: str, pattern: str) -> List[str]:
        """Find files matching pattern on remote server via SSH."""
        self.last_error = None
        try:
            # Use find command to match files
            cmd = f"find {shlex.quote(base_dir)} -maxdepth 1 -name {shlex.quote(pattern)} -type f 2>/dev/null | sort"
            stdout, stderr, exit_code = execute_responsive(self._ssh_manager, cmd, timeout=30)
            if exit_code == 0 and stdout.strip():
                return [line.strip() for line in stdout.strip().split("\n") if line.strip()]
            if exit_code != 0:
                detail = stderr.strip() or stdout.strip() or f"exit code {exit_code}"
                self.last_error = f"Remote glob failed for {base_dir.rstrip('/')}/{pattern}: {detail}"
        except Exception as exc:
            self.last_error = f"Remote glob failed for {base_dir.rstrip('/')}/{pattern}: {exc}"
            logger.warning("%s", self.last_error)
        return []


class LocalNetCDFValidator:
    """Validate NetCDF files locally using xarray."""

    # Common dimension names
    TIME_DIMS = ["time", "Time", "TIME", "t", "date"]
    LAT_DIMS = ["lat", "latitude", "Lat", "LAT", "y"]
    LON_DIMS = ["lon", "longitude", "Lon", "LON", "x"]

    def check_file_exists(self, path: str) -> ValidationCheck:
        """Check if file exists."""
        exists = os.path.exists(path)
        if exists:
            return ValidationCheck("file_exists", True, f"File exists: {path}")
        return ValidationCheck("file_exists", False, f"File not found: {path}")

    def _open_dataset(self, path: str):
        """Open dataset using safe_open."""
        return safe_open(path)

    def check_variable(self, path: str, varname: str) -> ValidationCheck:
        """Check if variable exists in NetCDF file."""
        try:
            import xarray as xr  # noqa: F401  feature detection
        except ImportError:
            return ValidationCheck("variable_exists", False, "xarray required: pip install xarray netCDF4")

        try:
            with self._open_dataset(path) as ds:
                available_vars = list(ds.data_vars)

            if varname in available_vars:
                return ValidationCheck("variable_exists", True, f"Variable '{varname}' exists")
            return ValidationCheck(
                "variable_exists", False, f"Variable '{varname}' not found, available: {available_vars}"
            )
        except Exception as e:
            return ValidationCheck("variable_exists", False, f"Cannot read file: {e}")

    def _find_dim(self, ds, candidates: List[str]) -> Optional[str]:
        """Find a dimension by trying common names."""
        for name in candidates:
            if name in ds.dims or name in ds.coords:
                return name
        return None

    def check_time_range(self, path: str, syear: int, eyear: int) -> ValidationCheck:
        """Check if data time range covers required period."""
        try:
            import xarray as xr  # noqa: F401  feature detection
            import pandas as pd  # noqa: F401  feature detection
        except ImportError:
            return ValidationCheck("time_range", False, "xarray required: pip install xarray netCDF4")

        try:
            with self._open_dataset(path) as ds:
                time_dim = self._find_dim(ds, self.TIME_DIMS)

                if time_dim is None:
                    return ValidationCheck("time_range", False, f"Time dimension not found, tried: {self.TIME_DIMS}")

                time_vals = ds[time_dim].values

            # Convert to years - handle cftime objects
            try:
                time_years = pd.to_datetime(time_vals).year
            except (TypeError, ValueError):
                # cftime or other non-standard calendar - skip time check
                return ValidationCheck("time_range", True, "Time check skipped (non-standard calendar)")

            data_syear = int(time_years.min())
            data_eyear = int(time_years.max())

            if data_syear <= syear and data_eyear >= eyear:
                return ValidationCheck(
                    "time_range", True, f"Time range OK: data {data_syear}-{data_eyear}, required {syear}-{eyear}"
                )
            return ValidationCheck(
                "time_range",
                False,
                f"Time range insufficient: data {data_syear}-{data_eyear}, required {syear}-{eyear}",
            )
        except Exception as e:
            return ValidationCheck("time_range", False, f"Time check failed: {e}")

    def check_spatial_range(
        self, path: str, min_lat: float, max_lat: float, min_lon: float, max_lon: float
    ) -> ValidationCheck:
        """Check if data spatial range covers required area."""
        try:
            import xarray as xr  # noqa: F401  feature detection
        except ImportError:
            return ValidationCheck("spatial_range", False, "xarray required: pip install xarray netCDF4")

        try:
            with self._open_dataset(path) as ds:
                lat_dim = self._find_dim(ds, self.LAT_DIMS)
                lon_dim = self._find_dim(ds, self.LON_DIMS)

                if lat_dim is None or lon_dim is None:
                    return ValidationCheck("spatial_range", False, "Lat/lon dimensions not found")

                lat_vals = ds[lat_dim].values
                lon_vals = ds[lon_dim].values

            data_min_lat, data_max_lat = float(lat_vals.min()), float(lat_vals.max())
            data_min_lon, data_max_lon = float(lon_vals.min()), float(lon_vals.max())

            lat_ok = data_min_lat <= min_lat and data_max_lat >= max_lat
            lon_ok = data_min_lon <= min_lon and data_max_lon >= max_lon

            if lat_ok and lon_ok:
                return ValidationCheck("spatial_range", True, "Spatial range OK")

            msg_parts = []
            if not lat_ok:
                msg_parts.append(
                    f"Lat: data {data_min_lat:.1f}~{data_max_lat:.1f}, required {min_lat:.1f}~{max_lat:.1f}"
                )
            if not lon_ok:
                msg_parts.append(
                    f"Lon: data {data_min_lon:.1f}~{data_max_lon:.1f}, required {min_lon:.1f}~{max_lon:.1f}"
                )

            return ValidationCheck("spatial_range", False, "Spatial range insufficient: " + "; ".join(msg_parts))
        except Exception as e:
            return ValidationCheck("spatial_range", False, f"Spatial check failed: {e}")


class RemoteNetCDFValidator:
    """Validate NetCDF files on remote server via SSH."""

    # Python script template for remote execution
    INSPECT_SCRIPT = '''
import json
import sys
try:
    import xarray as xr
    import pandas as pd

    def safe_open(path):
        """Open dataset, trying decode_times=False if default fails."""
        try:
            return xr.open_dataset(path)
        except Exception:
            return xr.open_dataset(path, decode_times=False)

    ds = safe_open({path_json})
    result = {{"success": True}}
    result["variables"] = list(ds.data_vars)

    # Find time dimension and extract time range
    time_dims = ['time', 'Time', 'TIME', 't', 'date']
    for td in time_dims:
        if td in ds.dims or td in ds.coords:
            try:
                time_data = ds[td].values
                time_vals = pd.to_datetime(time_data)
                result["time_range"] = [int(time_vals.year.min()), int(time_vals.year.max())]
            except Exception as e:
                # If time conversion fails, skip time check
                result["time_error"] = str(e)
            break

    # Find lat/lon dimensions
    lat_dims = ['lat', 'latitude', 'Lat', 'LAT', 'y']
    lon_dims = ['lon', 'longitude', 'Lon', 'LON', 'x']
    for ld in lat_dims:
        if ld in ds.dims or ld in ds.coords:
            result["lat_range"] = [float(ds[ld].values.min()), float(ds[ld].values.max())]
            break
    for ld in lon_dims:
        if ld in ds.dims or ld in ds.coords:
            result["lon_range"] = [float(ds[ld].values.min()), float(ds[ld].values.max())]
            break

    ds.close()
    print(json.dumps(result))
except ImportError as e:
    print(json.dumps({{"success": False, "error": "xarray not installed"}}))
except Exception as e:
    print(json.dumps({{"success": False, "error": str(e)}}))
'''

    def __init__(self, ssh_manager, python_path: str = "", conda_env: str = ""):
        """Initialize with SSH manager.

        Args:
            ssh_manager: SSHManager instance for remote execution
            python_path: Path to Python interpreter on remote server
            conda_env: Conda environment name to activate before running
        """
        self._ssh = ssh_manager
        self._python_path = python_path or "python3"
        self._conda_env = conda_env

    def check_file_exists(self, path: str) -> ValidationCheck:
        """Check if file exists on remote server."""
        try:
            stdout, stderr, exit_code = execute_responsive(self._ssh, f"test -f {shlex.quote(path)}", timeout=10)
            if exit_code == 0:
                return ValidationCheck("file_exists", True, f"File exists: {path}")
            return ValidationCheck("file_exists", False, f"File not found: {path}")
        except Exception as e:
            return ValidationCheck("file_exists", False, f"Remote check failed: {e}")

    def _run_inspect_script(self, path: str) -> Optional[Dict[str, Any]]:
        """Run inspection script on remote server."""
        from openbench.gui.remote_python import build_remote_python_command

        script = self.INSPECT_SCRIPT.format(path_json=json.dumps(path))
        cmd = build_remote_python_command(script, python_path=self._python_path, conda_env=self._conda_env)

        try:
            stdout, stderr, exit_code = execute_responsive(self._ssh, cmd, timeout=30)
            if exit_code == 0 and stdout.strip():
                return json.loads(stdout.strip())
        except Exception:
            pass
        return None

    def check_variable(self, path: str, varname: str) -> ValidationCheck:
        """Check if variable exists in remote NetCDF file."""
        result = self._run_inspect_script(path)

        if result is None:
            return ValidationCheck("variable_exists", False, "Remote check failed")

        if not result.get("success"):
            error = result.get("error", "Unknown error")
            return ValidationCheck("variable_exists", False, f"Remote error: {error}")

        variables = result.get("variables", [])
        if varname in variables:
            return ValidationCheck("variable_exists", True, f"Variable '{varname}' exists")
        return ValidationCheck("variable_exists", False, f"Variable '{varname}' not found, available: {variables}")

    def check_time_range(self, path: str, syear: int, eyear: int) -> ValidationCheck:
        """Check time range on remote file."""
        result = self._run_inspect_script(path)

        if result is None or not result.get("success"):
            return ValidationCheck("time_range", False, "Remote time check failed")

        # Check for time conversion error
        if "time_error" in result:
            return ValidationCheck("time_range", True, "Time check skipped (non-standard calendar)")

        time_range = result.get("time_range")
        if time_range is None:
            return ValidationCheck("time_range", True, "Time check skipped (no time dimension)")

        data_syear, data_eyear = time_range
        if data_syear <= syear and data_eyear >= eyear:
            return ValidationCheck("time_range", True, f"Time range OK: data {data_syear}-{data_eyear}")
        return ValidationCheck(
            "time_range", False, f"Time range insufficient: data {data_syear}-{data_eyear}, required {syear}-{eyear}"
        )

    def check_spatial_range(
        self, path: str, min_lat: float, max_lat: float, min_lon: float, max_lon: float
    ) -> ValidationCheck:
        """Check spatial range on remote file."""
        result = self._run_inspect_script(path)

        if result is None or not result.get("success"):
            return ValidationCheck("spatial_range", False, "Remote spatial check failed")

        lat_range = result.get("lat_range")
        lon_range = result.get("lon_range")

        if lat_range is None or lon_range is None:
            return ValidationCheck("spatial_range", False, "Lat/lon dimensions not found")

        data_min_lat, data_max_lat = lat_range
        data_min_lon, data_max_lon = lon_range

        lat_ok = data_min_lat <= min_lat and data_max_lat >= max_lat
        lon_ok = data_min_lon <= min_lon and data_max_lon >= max_lon

        if lat_ok and lon_ok:
            return ValidationCheck("spatial_range", True, "Spatial range OK")

        return ValidationCheck("spatial_range", False, "Spatial range insufficient")


class DataValidator:
    """Main validator that orchestrates validation checks.

    Note: The is_remote parameter determines the validation METHOD (local xarray vs SSH),
    not storage abstraction. This is intentionally separate from ProjectStorage because
    validation requires actual file access/inspection which differs fundamentally between
    local filesystem (xarray) and remote execution (SSH + Python script).
    """

    def __init__(
        self,
        is_remote: bool = False,
        ssh_manager=None,
        remote_openbench_root: str = "",
        python_path: str = "",
        conda_env: str = "",
    ):
        """Initialize validator.

        Args:
            is_remote: If True, use remote validation via SSH. This determines how
                      files are accessed for validation, not storage abstraction.
            ssh_manager: SSHManager instance (required if is_remote=True)
            remote_openbench_root: Remote OpenBench root path (for remote mode)
            python_path: Python interpreter path for remote execution
            conda_env: Conda environment name for remote execution
        """
        self._is_remote = is_remote
        self._ssh_manager = ssh_manager
        self._remote_openbench_root = remote_openbench_root
        self.last_error: str | None = None

        if is_remote and ssh_manager:
            self._validator = RemoteNetCDFValidator(ssh_manager, python_path, conda_env)
        else:
            self._validator = LocalNetCDFValidator()

    def validate_source(
        self, var_name: str, source_name: str, source_config: Dict[str, Any], general_config: Dict[str, Any]
    ) -> SourceValidationResult:
        """Validate a single data source.

        Args:
            var_name: Variable name (e.g., "Evapotranspiration")
            source_name: Source name (e.g., "GLEAM_v4.2a")
            source_config: Source configuration dict
            general_config: General settings (syear, eyear, lat/lon range)

        Returns:
            SourceValidationResult with all checks
        """
        checks = []

        # Extract config values
        general = source_config.get("general", source_config)
        var_config = source_config.get("var_config", source_config)

        root_dir = general.get("root_dir") or general.get("dir", "")
        # sub_dir, prefix, suffix, varname can be in var_config or top level
        sub_dir = var_config.get("sub_dir") or source_config.get("sub_dir", "")
        prefix = var_config.get("prefix") or source_config.get("prefix", "")
        suffix = var_config.get("suffix") or source_config.get("suffix", "")
        varname = var_config.get("varname") or source_config.get("varname", "")
        data_groupby = general.get("data_groupby", "Year")
        data_type = general.get("data_type", "grid")

        # Use source-specific years if available, otherwise general config.
        # source_config is shaped {"general": {...}, "varname": ..., ...} so
        # `source_config.get("syear")` was always None — the actual per-source
        # value lives one level down under "general".
        _src_general = source_config.get("general", {}) or {}
        syear = _src_general.get("syear") or general.get("syear") or general_config.get("syear", 2000)
        eyear = _src_general.get("eyear") or general.get("eyear") or general_config.get("eyear", 2020)

        # For station data without prefix/suffix, skip file path validation
        # Station data files may not follow the standard naming pattern
        if data_type == "stn" and not prefix and not suffix:
            return SourceValidationResult(var_name, source_name, checks)

        # Generate file paths
        path_gen = FilePathGenerator(
            root_dir=root_dir,
            sub_dir=sub_dir,
            prefix=prefix,
            suffix=suffix,
            data_groupby=data_groupby,
            syear=syear,
            eyear=eyear,
            is_remote=self._is_remote,
            ssh_manager=self._ssh_manager,
            remote_openbench_root=self._remote_openbench_root,
        )
        sample_paths = path_gen.get_sample_paths()

        # Check file existence
        first_existing_path = None
        if not sample_paths:
            # No files found matching the pattern, or remote listing failed.
            base_dir = path_gen._get_base_dir()
            pattern = f"{prefix}*{suffix}.nc"
            if getattr(path_gen, "last_error", None):
                checks.append(ValidationCheck("file_exists", False, path_gen.last_error))
            else:
                checks.append(
                    ValidationCheck("file_exists", False, f"No files found matching pattern '{pattern}' in {base_dir}")
                )
        else:
            for path in sample_paths:
                check = self._validator.check_file_exists(path)
                checks.append(check)
                if check.passed and first_existing_path is None:
                    first_existing_path = path

        # If no files found, skip other checks
        if first_existing_path is None:
            return SourceValidationResult(var_name, source_name, checks)

        # Check variable name
        if varname:
            check = self._validator.check_variable(first_existing_path, varname)
            checks.append(check)

        # Check time range (only for grid data with Single groupby)
        # For Year/Month/Day groupby, each file only contains partial data
        if data_type == "grid" and data_groupby == "Single":
            check = self._validator.check_time_range(first_existing_path, int(syear), int(eyear))
            checks.append(check)

        return SourceValidationResult(var_name, source_name, checks)

    def validate_all(
        self, sources: Dict[str, Dict[str, Dict]], general_config: Dict[str, Any], progress_callback=None
    ) -> DataValidationReport:
        """Validate all data sources.

        Args:
            sources: Dict of {var_name: {source_name: source_config}}
            general_config: General settings
            progress_callback: Optional callback(current, total, var_name, source_name)

        Returns:
            DataValidationReport with all results
        """
        results = []
        total = sum(len(s) for s in sources.values())
        current = 0

        for var_name, var_sources in sources.items():
            for source_name, source_config in var_sources.items():
                if progress_callback:
                    progress_callback(current, total, var_name, source_name)

                result = self.validate_source(var_name, source_name, source_config, general_config)
                results.append(result)
                current += 1

        if progress_callback:
            progress_callback(total, total, "", "")

        return DataValidationReport(results=results)
