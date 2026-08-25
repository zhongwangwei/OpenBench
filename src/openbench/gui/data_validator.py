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

from openbench.data.coordinates import NC_SUFFIXES
from openbench.gui.remote_python import quote_remote_path
from openbench.gui.widgets._ssh_worker import execute_responsive
from openbench.gui.path_utils import to_absolute_path, get_openbench_root

logger = logging.getLogger(__name__)


def _has_nc_suffix(value: str) -> bool:
    return value.casefold().endswith(tuple(suffix.casefold() for suffix in NC_SUFFIXES))


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


def _longitude_range_covers(
    data_min: float,
    data_max: float,
    required_min: float,
    required_max: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Compare longitude intervals even when they use different 0/360 domains."""
    metadata = metadata or {}
    if metadata.get("lon_is_global"):
        return True
    coverage = metadata.get("lon_coverage")
    if coverage and required_max >= required_min:
        required_span = required_max - required_min
        if required_span >= 359.999:
            return False
        required_start = required_min % 360.0
        required_end = required_start + required_span
        coverage_start, coverage_end = coverage
        return any(
            coverage_start <= required_start + shift and coverage_end >= required_end + shift
            for shift in (-360.0, 0.0, 360.0)
        )
    return any(
        data_min + data_shift <= required_min + required_shift
        and data_max + data_shift >= required_max + required_shift
        for data_shift in (-360.0, 0.0, 360.0)
        for required_shift in (-360.0, 0.0, 360.0)
    )


def _longitude_metadata(values) -> Dict[str, Any]:
    """Describe a one-dimensional longitude coordinate on a circular domain."""
    try:
        import numpy as np

        array = np.asarray(values, dtype=float)
        if array.ndim != 1:
            return {}
        normalized = np.unique(np.mod(array[np.isfinite(array)], 360.0))
        if normalized.size < 2:
            return {}
        gaps = np.diff(np.concatenate((normalized, [normalized[0] + 360.0])))
        largest_index = int(np.argmax(gaps))
        smaller_gaps = np.delete(gaps, largest_index)
        resolution = float(np.median(smaller_gaps[smaller_gaps > 0])) if np.any(smaller_gaps > 0) else 0.0
        largest_gap = float(gaps[largest_index])
        coverage_start = float(normalized[(largest_index + 1) % normalized.size])
        coverage_end = float(normalized[largest_index])
        if coverage_end < coverage_start:
            coverage_end += 360.0
        return {
            "lon_is_global": bool(normalized.size >= 4 and resolution > 0 and largest_gap <= resolution * 1.5 + 1e-9),
            "lon_coverage": [coverage_start, coverage_end],
        }
    except (ImportError, TypeError, ValueError):
        return {}


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

    def _candidate_filenames(self) -> List[str]:
        base = f"{self.prefix}{self.suffix}"
        if _has_nc_suffix(base):
            return [base]
        return [f"{base}{ext}" for ext in NC_SUFFIXES]

    def _candidate_patterns(self) -> List[str]:
        base = f"{self.prefix}*{self.suffix}"
        if _has_nc_suffix(base):
            return [base]
        return [f"{base}{ext}" for ext in NC_SUFFIXES]

    def describe_pattern(self) -> str:
        patterns = self._candidate_patterns()
        return patterns[0] if len(patterns) == 1 else "{" + ",".join(patterns) + "}"

    def get_sample_paths(self) -> List[str]:
        """Get sample file paths for validation.

        Uses glob pattern to find actual files matching prefix and suffix.
        Returns a small set of representative paths to check.
        """
        base_dir = self._get_base_dir()

        if str(self.data_groupby).lower() == "single":
            candidates = [self._build_path(filename) for filename in self._candidate_filenames()]
            if self._is_remote:
                return candidates
            existing = [path for path in candidates if os.path.exists(path)]
            return existing or candidates[:1]

        # For Year/Month/Day, use glob to find matching files.
        patterns = self._candidate_patterns()

        if self._is_remote and self._ssh_manager:
            # Remote mode: use SSH to list files
            matching_files = self._remote_glob(base_dir, patterns)
        else:
            # Local mode: use local glob
            import glob

            matching_files = []
            for pattern in patterns:
                full_pattern = os.path.join(base_dir, pattern)
                matching_files.extend(glob.glob(full_pattern))
            matching_files = sorted(set(matching_files))

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

    def _remote_glob(self, base_dir: str, patterns: List[str]) -> List[str]:
        """Find files matching pattern on remote server via SSH."""
        self.last_error = None
        pattern_desc = patterns[0] if len(patterns) == 1 else "{" + ",".join(patterns) + "}"
        try:
            # Use find command to match files
            if len(patterns) == 1:
                name_expr = f"-name {shlex.quote(patterns[0])}"
            else:
                parts = []
                for index, pattern in enumerate(patterns):
                    if index:
                        parts.append("-o")
                    parts.extend(["-name", shlex.quote(pattern)])
                name_expr = r"\( " + " ".join(parts) + r" \)"
            cmd = f"find {quote_remote_path(base_dir)} -maxdepth 1 {name_expr} -type f 2>/dev/null | sort"
            stdout, stderr, exit_code = execute_responsive(self._ssh_manager, cmd, timeout=30)
            if exit_code == 0 and stdout.strip():
                return [line.strip() for line in stdout.strip().split("\n") if line.strip()]
            if exit_code != 0:
                detail = stderr.strip() or stdout.strip() or f"exit code {exit_code}"
                self.last_error = f"Remote glob failed for {base_dir.rstrip('/')}/{pattern_desc}: {detail}"
        except Exception as exc:
            self.last_error = f"Remote glob failed for {base_dir.rstrip('/')}/{pattern_desc}: {exc}"
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

    def inspect_file(self, path: str) -> Dict[str, Any]:
        """Read all metadata needed by validation while opening the file once."""
        try:
            import pandas as pd

            with self._open_dataset(path) as ds:
                result: Dict[str, Any] = {"success": True, "variables": list(ds.data_vars)}
                time_dim = self._find_dim(ds, self.TIME_DIMS)
                if time_dim is None:
                    result["time_missing"] = True
                else:
                    try:
                        time_years = pd.to_datetime(ds[time_dim].values).year
                        result["time_range"] = [int(time_years.min()), int(time_years.max())]
                    except (TypeError, ValueError) as exc:
                        result["time_error"] = str(exc)

                lat_dim = self._find_dim(ds, self.LAT_DIMS)
                lon_dim = self._find_dim(ds, self.LON_DIMS)
                if lat_dim is not None:
                    lat_vals = ds[lat_dim].values
                    result["lat_range"] = [float(lat_vals.min()), float(lat_vals.max())]
                if lon_dim is not None:
                    lon_vals = ds[lon_dim].values
                    result["lon_range"] = [float(lon_vals.min()), float(lon_vals.max())]
                    result.update(_longitude_metadata(lon_vals))
                return result
        except ImportError:
            return {"success": False, "error": "xarray required: pip install xarray netCDF4"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def check_variable(self, path: str, varname: str, inspection: Optional[Dict[str, Any]] = None) -> ValidationCheck:
        """Check if variable exists in NetCDF file."""
        result = inspection if inspection is not None else self.inspect_file(path)
        if not result.get("success"):
            return ValidationCheck(
                "variable_exists", False, f"Cannot read file: {result.get('error', 'Unknown error')}"
            )
        available_vars = result.get("variables", [])
        if varname in available_vars:
            return ValidationCheck("variable_exists", True, f"Variable '{varname}' exists")
        return ValidationCheck("variable_exists", False, f"Variable '{varname}' not found, available: {available_vars}")

    def _find_dim(self, ds, candidates: List[str]) -> Optional[str]:
        """Find a dimension by trying common names."""
        for name in candidates:
            if name in ds.dims or name in ds.coords:
                return name
        return None

    def check_time_range(
        self, path: str, syear: int, eyear: int, inspection: Optional[Dict[str, Any]] = None
    ) -> ValidationCheck:
        """Check if data time range covers required period."""
        result = inspection if inspection is not None else self.inspect_file(path)
        if not result.get("success"):
            return ValidationCheck("time_range", False, f"Time check failed: {result.get('error', 'Unknown error')}")
        if result.get("time_missing"):
            return ValidationCheck("time_range", False, f"Time dimension not found, tried: {self.TIME_DIMS}")
        if "time_error" in result:
            return ValidationCheck("time_range", True, "Time check skipped (non-standard calendar)")
        time_range = result.get("time_range")
        if time_range is None:
            return ValidationCheck("time_range", False, "Time check failed: no time range")
        data_syear, data_eyear = time_range
        if data_syear <= syear and data_eyear >= eyear:
            return ValidationCheck(
                "time_range", True, f"Time range OK: data {data_syear}-{data_eyear}, required {syear}-{eyear}"
            )
        return ValidationCheck(
            "time_range",
            False,
            f"Time range insufficient: data {data_syear}-{data_eyear}, required {syear}-{eyear}",
        )

    def check_spatial_range(
        self,
        path: str,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        inspection: Optional[Dict[str, Any]] = None,
    ) -> ValidationCheck:
        """Check if data spatial range covers required area."""
        result = inspection if inspection is not None else self.inspect_file(path)
        if not result.get("success"):
            return ValidationCheck(
                "spatial_range", False, f"Spatial check failed: {result.get('error', 'Unknown error')}"
            )
        lat_range = result.get("lat_range")
        lon_range = result.get("lon_range")
        if lat_range is None or lon_range is None:
            return ValidationCheck("spatial_range", False, "Lat/lon dimensions not found")
        data_min_lat, data_max_lat = lat_range
        data_min_lon, data_max_lon = lon_range
        lat_ok = data_min_lat <= min_lat and data_max_lat >= max_lat
        lon_ok = _longitude_range_covers(
            data_min_lon,
            data_max_lon,
            min_lon,
            max_lon,
            result,
        )
        if lat_ok and lon_ok:
            return ValidationCheck("spatial_range", True, "Spatial range OK")

        msg_parts = []
        if not lat_ok:
            msg_parts.append(f"Lat: data {data_min_lat:.1f}~{data_max_lat:.1f}, required {min_lat:.1f}~{max_lat:.1f}")
        if not lon_ok:
            msg_parts.append(f"Lon: data {data_min_lon:.1f}~{data_max_lon:.1f}, required {min_lon:.1f}~{max_lon:.1f}")
        return ValidationCheck("spatial_range", False, "Spatial range insufficient: " + "; ".join(msg_parts))


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
            lon_values = ds[ld].values
            result["lon_range"] = [float(lon_values.min()), float(lon_values.max())]
            try:
                import numpy as np
                array = np.asarray(lon_values, dtype=float)
                if array.ndim == 1:
                    normalized = np.unique(np.mod(array[np.isfinite(array)], 360.0))
                    if normalized.size > 1:
                        gaps = np.diff(np.concatenate((normalized, [normalized[0] + 360.0])))
                        largest_index = int(np.argmax(gaps))
                        smaller_gaps = np.delete(gaps, largest_index)
                        positive_gaps = smaller_gaps[smaller_gaps > 0]
                        resolution = float(np.median(positive_gaps)) if positive_gaps.size else 0.0
                        coverage_start = float(normalized[(largest_index + 1) % normalized.size])
                        coverage_end = float(normalized[largest_index])
                        if coverage_end < coverage_start:
                            coverage_end += 360.0
                        result["lon_is_global"] = bool(
                            normalized.size >= 4
                            and resolution > 0
                            and float(gaps[largest_index]) <= resolution * 1.5 + 1e-9
                        )
                        result["lon_coverage"] = [coverage_start, coverage_end]
            except Exception:
                pass
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
            stdout, stderr, exit_code = execute_responsive(self._ssh, f"test -f {quote_remote_path(path)}", timeout=10)
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

    def inspect_file(self, path: str) -> Dict[str, Any]:
        """Return all remotely inspected metadata in one SSH round trip."""
        result = self._run_inspect_script(path)
        return result if result is not None else {"success": False, "error": "Remote check failed"}

    def check_variable(self, path: str, varname: str, inspection: Optional[Dict[str, Any]] = None) -> ValidationCheck:
        """Check if variable exists in remote NetCDF file."""
        result = inspection if inspection is not None else self.inspect_file(path)

        if not result.get("success"):
            error = result.get("error", "Unknown error")
            return ValidationCheck("variable_exists", False, f"Remote error: {error}")

        variables = result.get("variables", [])
        if varname in variables:
            return ValidationCheck("variable_exists", True, f"Variable '{varname}' exists")
        return ValidationCheck("variable_exists", False, f"Variable '{varname}' not found, available: {variables}")

    def check_time_range(
        self, path: str, syear: int, eyear: int, inspection: Optional[Dict[str, Any]] = None
    ) -> ValidationCheck:
        """Check time range on remote file."""
        result = inspection if inspection is not None else self.inspect_file(path)

        if not result.get("success"):
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
        self,
        path: str,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        inspection: Optional[Dict[str, Any]] = None,
    ) -> ValidationCheck:
        """Check spatial range on remote file."""
        result = inspection if inspection is not None else self.inspect_file(path)

        if not result.get("success"):
            return ValidationCheck("spatial_range", False, "Remote spatial check failed")

        lat_range = result.get("lat_range")
        lon_range = result.get("lon_range")

        if lat_range is None or lon_range is None:
            return ValidationCheck("spatial_range", False, "Lat/lon dimensions not found")

        data_min_lat, data_max_lat = lat_range
        data_min_lon, data_max_lon = lon_range

        lat_ok = data_min_lat <= min_lat and data_max_lat >= max_lat
        lon_ok = _longitude_range_covers(
            data_min_lon,
            data_max_lon,
            min_lon,
            max_lon,
            result,
        )

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
        reference_data_root: str = "",
    ):
        """Initialize validator.

        Args:
            is_remote: If True, use remote validation via SSH. This determines how
                      files are accessed for validation, not storage abstraction.
            ssh_manager: SSHManager instance (required if is_remote=True)
            remote_openbench_root: Remote OpenBench root path (for remote mode)
            python_path: Python interpreter path for remote execution
            conda_env: Conda environment name for remote execution
            reference_data_root: Explicit runtime override for grid reference roots
        """
        self._is_remote = is_remote
        self._ssh_manager = ssh_manager
        self._remote_openbench_root = remote_openbench_root
        self._reference_data_root = reference_data_root
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
        if data_type != "stn" and self._reference_data_root:
            root_dir = self._reference_data_root

        data_syear = general.get("syear")
        data_eyear = general.get("eyear")
        syear = int(general_config.get("syear", 2000))
        eyear = int(general_config.get("eyear", 2020))
        if data_syear is not None and data_eyear is not None:
            try:
                data_syear, data_eyear = int(data_syear), int(data_eyear)
            except (TypeError, ValueError):
                pass
            else:
                if data_eyear < syear or data_syear > eyear:
                    checks.append(
                        ValidationCheck(
                            "time_range",
                            False,
                            f"Data years {data_syear}-{data_eyear} do not overlap required years {syear}-{eyear}",
                        )
                    )
                elif data_syear > syear or data_eyear < eyear:
                    checks.append(
                        ValidationCheck(
                            "time_range",
                            False,
                            f"Data years {data_syear}-{data_eyear} do not cover required years {syear}-{eyear}",
                        )
                    )

        # For station data without prefix/suffix, validate the reachable root
        # and station list rather than returning an empty successful result.
        if data_type == "stn" and not prefix and not suffix:
            base_dir = FilePathGenerator(
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
            )._get_base_dir()
            checks.append(self._check_directory_exists(base_dir))
            fulllist = general.get("fulllist") or source_config.get("fulllist") or var_config.get("fulllist")
            if fulllist:
                checks.append(self._check_file_exists(self._resolve_aux_path(fulllist, root_dir)))
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
            pattern = path_gen.describe_pattern()
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

        needs_time_check = data_type == "grid" and str(data_groupby).lower() == "single"
        needs_spatial_check = data_type == "grid" and all(
            key in general_config for key in ("min_lat", "max_lat", "min_lon", "max_lon")
        )
        inspection = (
            self._validator.inspect_file(first_existing_path)
            if varname or needs_time_check or needs_spatial_check
            else None
        )

        # Check variable name
        if varname:
            check = self._validator.check_variable(first_existing_path, varname, inspection)
            checks.append(check)

        # Check time range (only for grid data with Single groupby)
        # For Year/Month/Day groupby, each file only contains partial data
        if needs_time_check:
            check = self._validator.check_time_range(first_existing_path, int(syear), int(eyear), inspection)
            checks.append(check)

        if needs_spatial_check:
            checks.append(
                self._validator.check_spatial_range(
                    first_existing_path,
                    float(general_config["min_lat"]),
                    float(general_config["max_lat"]),
                    float(general_config["min_lon"]),
                    float(general_config["max_lon"]),
                    inspection,
                )
            )

        return SourceValidationResult(var_name, source_name, checks)

    def _resolve_aux_path(self, path: str, root_dir: str) -> str:
        if self._is_remote:
            normalized = str(path).replace("\\", "/")
            if normalized.startswith("/"):
                return normalized
            root = str(root_dir).replace("\\", "/")
            return f"{root.rstrip('/')}/{normalized}"
        if os.path.isabs(str(path)):
            return str(path)
        return os.path.join(to_absolute_path(str(root_dir), get_openbench_root()), str(path))

    def _check_directory_exists(self, path: str) -> ValidationCheck:
        if self._is_remote and self._ssh_manager:
            try:
                stdout, stderr, exit_code = execute_responsive(
                    self._ssh_manager, f"test -d {quote_remote_path(path)}", timeout=10
                )
                if exit_code == 0:
                    return ValidationCheck("directory_exists", True, f"Directory exists: {path}")
                detail = stderr.strip() or stdout.strip()
                suffix = f": {detail}" if detail else ""
                return ValidationCheck("directory_exists", False, f"Directory not found: {path}{suffix}")
            except Exception as exc:
                return ValidationCheck("directory_exists", False, f"Remote directory check failed: {exc}")
        if os.path.isdir(path):
            return ValidationCheck("directory_exists", True, f"Directory exists: {path}")
        return ValidationCheck("directory_exists", False, f"Directory not found: {path}")

    def _check_file_exists(self, path: str) -> ValidationCheck:
        return self._validator.check_file_exists(path)

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
