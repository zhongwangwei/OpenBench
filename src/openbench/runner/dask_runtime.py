"""Dask/runtime-environment helpers for the local runner."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disable", "disabled", ""}:
        return False
    logger.warning("Ignoring invalid boolean %s=%r", name, value)
    return default


def env_positive_int(name: str, *, default: int, minimum: int = 1) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return max(minimum, int(default))
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Ignoring invalid integer %s=%r", name, value)
        return max(minimum, int(default))
    if parsed < minimum:
        logger.warning("Ignoring %s=%r below minimum %d", name, value, minimum)
        return max(minimum, int(default))
    return parsed


def dask_distributed_requested(dask_config: Any | None = None) -> bool:
    """Return dask opt-in state from env or project.dask config.

    Environment variables are process-level overrides and therefore take
    precedence over YAML. If no env override is present, use
    ``project.dask.enabled``.
    """
    if "OPENBENCH_DASK" in os.environ:
        return env_flag("OPENBENCH_DASK", default=False)
    if "OPENBENCH_DASK_DISTRIBUTED" in os.environ:
        return env_flag("OPENBENCH_DASK_DISTRIBUTED", default=False)
    return bool(getattr(dask_config, "enabled", False))


def task_uses_station_data(task: dict[str, Any]) -> bool:
    return (
        str(task.get("ref_data_type", "grid")).lower() == "stn"
        or str(task.get("sim_data_type", "grid")).lower() == "stn"
    )


def tasks_use_station_data(tasks: list[dict[str, Any]] | None) -> bool:
    return any(task_uses_station_data(task) for task in tasks or [])


def config_uses_station_data(cfg: Any) -> bool:
    """Return True when the configured evaluation includes station extraction.

    Dask distributed is high-ROI for lazy grid math, but station-heavy paths
    already parallelize over stations and do many small NetCDF writes. Starting
    a distributed scheduler there can add overhead or stall the combine/write
    phase, so use adapter runtime info to auto-avoid those tasks.
    """
    for sim in (getattr(cfg, "simulation", {}) or {}).values():
        if str(getattr(sim, "data_type", "grid")).lower() == "stn":
            return True
        for override in (getattr(sim, "variables", None) or {}).values():
            if str((override or {}).get("data_type", "grid")).lower() == "stn":
                return True

    try:
        from openbench.config.adapter import build_runner_bindings

        bindings = build_runner_bindings(cfg)
        for source in bindings.iter_task_sources(cfg.evaluation.variables):
            info = bindings.build_runtime_info_for(
                source.var_name,
                source.sim_source,
                source.ref_source,
            ).to_info()
            if (
                str(info.get("ref_data_type", "grid")).lower() == "stn"
                or str(info.get("sim_data_type", "grid")).lower() == "stn"
            ):
                return True
    except Exception as exc:
        logger.debug("Could not inspect dask station suitability from config: %s", exc)
    return False


def dask_station_guard_blocks(
    *,
    station_heavy: bool = False,
    tasks: list[dict[str, Any]] | None = None,
) -> bool:
    if not (station_heavy or tasks_use_station_data(tasks)):
        return False
    if env_flag("OPENBENCH_DASK_ALLOW_STATION", default=False):
        return False
    return True


def project_num_cores(cfg: Any) -> int:
    requested = getattr(getattr(cfg, "project", None), "num_cores", None)
    if requested:
        try:
            return max(1, int(requested))
        except (TypeError, ValueError):
            return max(1, os.cpu_count() or 1)
    return max(1, os.cpu_count() or 1)


def project_dask_config(cfg: Any) -> Any | None:
    project = getattr(cfg, "project", None)
    return getattr(project, "dask", None) if project is not None else None


def project_dask_local_directory(cfg: Any) -> str | None:
    project = getattr(cfg, "project", None)
    if project is None:
        return None
    dask_config = getattr(project, "dask", None)
    if os.environ.get("OPENBENCH_DASK_LOCAL_DIRECTORY"):
        return os.environ["OPENBENCH_DASK_LOCAL_DIRECTORY"]
    configured = getattr(dask_config, "local_directory", None)
    if configured:
        return str(Path(os.path.expandvars(str(configured))).expanduser())
    try:
        return str(Path(project.output_dir) / project.name / "scratch" / "dask")
    except Exception:
        return None


def project_io_config(cfg: Any) -> Any | None:
    project = getattr(cfg, "project", None)
    return getattr(project, "io", None) if project is not None else None


def io_env_defaults(io_config: Any | None) -> dict[str, str]:
    """Return project.io env defaults without overriding explicit env vars."""
    if io_config is None:
        return {}

    defaults: dict[str, str] = {}
    if getattr(io_config, "netcdf_compression", False):
        defaults["OPENBENCH_NETCDF_COMPRESSION"] = "1"
        defaults["OPENBENCH_NETCDF_COMP_LEVEL"] = str(getattr(io_config, "netcdf_compression_level", 1))

    mapping = {
        "mfdataset_batch_size": "OPENBENCH_MFDATASET_BATCH_SIZE",
        "mfdataset_auto_batch_min_files": "OPENBENCH_MFDATASET_AUTO_BATCH_MIN_FILES",
        "mfdataset_auto_batch_min_size_mb": "OPENBENCH_MFDATASET_AUTO_BATCH_MIN_SIZE_MB",
        "mfdataset_auto_batch_min_size": "OPENBENCH_MFDATASET_AUTO_BATCH_MIN_SIZE",
        "mfdataset_auto_batch_max_size": "OPENBENCH_MFDATASET_AUTO_BATCH_MAX_SIZE",
        "mfdataset_auto_batch_memory_fraction": "OPENBENCH_MFDATASET_AUTO_BATCH_MEMORY_FRACTION",
    }
    for attr, env_name in mapping.items():
        value = getattr(io_config, attr, None)
        if value is not None:
            defaults[env_name] = str(value)
    return defaults


@contextmanager
def temporary_env_defaults(defaults: dict[str, str]):
    """Set defaults for this run only when env does not already define them."""
    applied: list[str] = []
    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value
            applied.append(key)
    try:
        yield
    finally:
        for key in applied:
            os.environ.pop(key, None)


def dask_option(name: str, dask_config: Any | None, attr: str, default: Any = None) -> Any:
    if name in os.environ:
        return os.environ[name]
    value = getattr(dask_config, attr, None)
    return default if value is None else value


def start_optional_dask_client(
    num_cores: int,
    *,
    only_drawing: bool = False,
    comparison_only: bool = False,
    local_directory: str | None = None,
    dask_config: Any | None = None,
    station_heavy: bool = False,
    tasks: list[dict[str, Any]] | None = None,
) -> tuple[Any, Any | None] | None:
    """Start an opt-in dask.distributed scheduler for xarray lazy work.

    Disabled by default to preserve historical behavior. Enable with
    ``OPENBENCH_DASK=1`` (or the older ``OPENBENCH_DASK_DISTRIBUTED=1``).
    When enabled, startup/dependency failures are fatal because silent fallback
    makes performance runs non-reproducible.
    """
    if only_drawing or comparison_only:
        return None
    if not dask_distributed_requested(dask_config):
        return None
    if dask_station_guard_blocks(station_heavy=station_heavy, tasks=tasks):
        logger.warning(
            "Skipping dask.distributed for station-heavy/station-extraction workload; "
            "set OPENBENCH_DASK_ALLOW_STATION=1 to force it."
        )
        return None

    try:
        from distributed import Client, LocalCluster
    except ImportError as exc:
        raise RuntimeError(
            "OPENBENCH_DASK is enabled but dask.distributed is not installed; "
            "install distributed or disable OPENBENCH_DASK"
        ) from exc

    scheduler_address = dask_option("OPENBENCH_DASK_SCHEDULER", dask_config, "scheduler")
    if scheduler_address:
        try:
            client = Client(scheduler_address, set_as_default=True)
        except Exception as exc:
            raise RuntimeError(f"Could not connect to dask scheduler {scheduler_address!r}: {exc}") from exc
        logger.info("Dask distributed client connected: scheduler=%s", scheduler_address)
        return client, None

    configured_workers = getattr(dask_config, "n_workers", None) or num_cores
    default_workers = env_positive_int("OPENBENCH_DASK_WORKERS", default=configured_workers)
    workers = min(env_positive_int("OPENBENCH_DASK_N_WORKERS", default=default_workers), max(1, num_cores))
    threads_per_worker = env_positive_int(
        "OPENBENCH_DASK_THREADS_PER_WORKER",
        default=getattr(dask_config, "threads_per_worker", 1) or 1,
    )
    if "OPENBENCH_DASK_PROCESSES" in os.environ:
        processes = env_flag("OPENBENCH_DASK_PROCESSES", default=True)
    else:
        processes = bool(getattr(dask_config, "processes", True))
    memory_limit = dask_option("OPENBENCH_DASK_MEMORY_LIMIT", dask_config, "memory_limit", "auto")
    dashboard_address = dask_option("OPENBENCH_DASK_DASHBOARD_ADDRESS", dask_config, "dashboard_address")
    if dashboard_address is None:
        dashboard_address = None

    try:
        cluster = LocalCluster(
            n_workers=workers,
            threads_per_worker=threads_per_worker,
            processes=processes,
            memory_limit=memory_limit,
            dashboard_address=dashboard_address,
            scheduler_port=0,
            host="127.0.0.1",
            protocol="tcp",
            local_directory=local_directory,
            silence_logs=logging.WARNING,
        )
        client = Client(cluster, set_as_default=True)
        logger.info(
            "Dask distributed client started: workers=%d threads_per_worker=%d processes=%s dashboard=%s",
            workers,
            threads_per_worker,
            processes,
            getattr(client, "dashboard_link", None),
        )
        return client, cluster
    except Exception as exc:
        try:
            if "client" in locals():
                client.close()
            if "cluster" in locals():
                cluster.close()
        finally:
            raise RuntimeError(f"Could not start dask distributed client: {exc}") from exc


def close_optional_dask_client(handle: tuple[Any, Any | None] | None) -> None:
    if handle is None:
        return
    client, cluster = handle
    try:
        client.close()
    except Exception as exc:
        logger.warning("Could not close dask client: %s", exc)
    if cluster is not None:
        try:
            cluster.close()
        except Exception as exc:
            logger.warning("Could not close dask cluster: %s", exc)
