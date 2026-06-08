"""System resource sizing helpers for dataset processing."""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency fallback
    psutil = None


def get_system_resources():
    """
    Get system resources information with cross-platform compatibility.

    Returns:
        dict: Dictionary containing system resource information
    """
    import platform

    # Initialize default values
    result = {
        "total_memory_gb": 8,  # Default values
        "available_memory_gb": 4,
        "cpu_count": 4,
        "cpu_freq_mhz": 0,
    }

    try:
        # Get memory information - works on all platforms
        memory_info = psutil.virtual_memory()
        result["total_memory_gb"] = memory_info.total / (1024**3)
        result["available_memory_gb"] = memory_info.available / (1024**3)
    except Exception as e:
        logging.warning(f"Failed to get memory info: {e}")

    try:
        # Get CPU count - works on all platforms
        cpu_count = psutil.cpu_count(logical=False)
        if cpu_count is not None:
            result["cpu_count"] = cpu_count
        else:
            # Fallback to logical CPU count
            result["cpu_count"] = psutil.cpu_count(logical=True) or 4
    except Exception as e:
        logging.warning(f"Failed to get CPU count: {e}")

    # Get CPU frequency with platform-specific handling
    cpu_freq_from_psutil = False
    try:
        cpu_freq_info = psutil.cpu_freq()
        if cpu_freq_info is not None and hasattr(cpu_freq_info, "max") and cpu_freq_info.max:
            result["cpu_freq_mhz"] = cpu_freq_info.max
            cpu_freq_from_psutil = True
        elif cpu_freq_info is not None and hasattr(cpu_freq_info, "current") and cpu_freq_info.current:
            result["cpu_freq_mhz"] = cpu_freq_info.current
            cpu_freq_from_psutil = True
    except Exception as e:
        logging.debug(f"psutil.cpu_freq() failed: {e}")

    # If psutil didn't work, try platform-specific fallbacks
    if not cpu_freq_from_psutil:
        try:
            system = platform.system().lower()
            if system == "darwin":  # macOS
                result["cpu_freq_mhz"] = _get_macos_cpu_freq()
            elif system == "linux":
                result["cpu_freq_mhz"] = _get_linux_cpu_freq()
            elif system == "windows":
                result["cpu_freq_mhz"] = _get_windows_cpu_freq()
        except Exception as e:
            logging.debug(f"Platform-specific CPU frequency detection failed: {e}")
            # CPU frequency is optional, so we continue with 0

    return result


def _get_macos_cpu_freq():
    """Get CPU frequency on macOS."""
    try:
        import subprocess

        # For Apple Silicon Macs, try sysctl to get CPU frequency
        try:
            result = subprocess.run(["sysctl", "-n", "hw.cpufrequency_max"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip().isdigit():
                # Convert Hz to MHz
                return float(result.stdout.strip()) / 1000000
        except Exception as e:
            logger.debug("sysctl hw.cpufrequency_max failed: %s", e)

        # Try alternative sysctl commands for Apple Silicon
        try:
            result = subprocess.run(["sysctl", "-n", "hw.cpufrequency"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip().isdigit():
                return float(result.stdout.strip()) / 1000000
        except Exception as e:
            logger.debug("sysctl hw.cpufrequency failed: %s", e)

        # For Intel Macs or fallback, try system_profiler
        result = subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            import re

            for line in result.stdout.split("\n"):
                if "Processor Speed" in line:
                    # Extract frequency (e.g., "2.3 GHz" -> 2300)
                    match = re.search(r"(\d+\.?\d*)\s*GHz", line)
                    if match:
                        return float(match.group(1)) * 1000
                elif "Chip:" in line and "Apple" in line:
                    # For Apple Silicon, provide estimated frequencies based on chip model
                    if "M1" in line:
                        return 3200  # M1 estimated max frequency
                    elif "M2" in line:
                        return 3500  # M2 estimated max frequency
                    elif "M3" in line:
                        return 4000  # M3 estimated max frequency
                    elif "M4" in line:
                        return 4400  # M4 estimated max frequency

        return 0
    except Exception as e:
        logger.debug("macOS CPU frequency detection failed: %s", e)
        return 0


def _get_linux_cpu_freq():
    """Get CPU frequency on Linux."""
    try:
        # Try reading from /proc/cpuinfo
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("cpu MHz"):
                    return float(line.split(":")[1].strip())
        return 0
    except Exception as e:
        logger.debug("Linux CPU frequency detection failed: %s", e)
        return 0


def _get_windows_cpu_freq():
    """Get CPU frequency on Windows."""
    try:
        import subprocess

        # Try wmic command
        result = subprocess.run(
            ["wmic", "cpu", "get", "MaxClockSpeed", "/format:value"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "MaxClockSpeed=" in line:
                    freq = line.split("=")[1].strip()
                    if freq.isdigit():
                        return float(freq)  # Already in MHz
        return 0
    except Exception as e:
        logger.debug("Windows CPU frequency detection failed: %s", e)
        return 0


def calculate_optimal_chunk_size(dataset_size_gb: float, available_memory_gb: float) -> Dict[str, Any]:
    """
    Calculate optimal chunk size based on dataset size and available memory.

    For small datasets that fit in memory, returns no chunking (None).
    For larger datasets, lets xarray handle chunking via 'auto'.

    Args:
        dataset_size_gb: Size of the dataset in GB
        available_memory_gb: Available memory in GB

    Returns:
        dict: Chunk sizes for time/lat/lon dimensions, or None values for no chunking.
    """
    if dataset_size_gb < available_memory_gb * 0.3:
        # Dataset fits comfortably in memory — no chunking needed
        return {"time": None, "lat": None, "lon": None}
    # Large dataset — let xarray decide chunk sizes
    return {"time": "auto", "lat": "auto", "lon": "auto"}


def calculate_optimal_cores(cpu_count: int, available_memory_gb: float, dataset_size_gb: float) -> int:
    """
    Calculate optimal number of cores based on system resources and dataset size.

    Args:
        cpu_count (int): Number of CPU cores
        available_memory_gb (float): Available memory in GB
        dataset_size_gb (float): Size of the dataset in GB

    Returns:
        int: Optimal number of cores to use
    """
    # Calculate memory per core needed
    memory_per_core = dataset_size_gb / cpu_count

    # If memory per core is too high, reduce number of cores
    if memory_per_core > available_memory_gb * 0.8:
        optimal_cores = max(1, int(available_memory_gb * 0.8 / memory_per_core))
    else:
        # Leave one core free for system processes
        optimal_cores = max(1, cpu_count - 1)

    return optimal_cores


__all__ = [
    "get_system_resources",
    "calculate_optimal_chunk_size",
    "calculate_optimal_cores",
]
