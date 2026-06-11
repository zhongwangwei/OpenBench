"""Background workers for GUI registry scans."""

from __future__ import annotations

import dataclasses
import json

from PySide6.QtCore import QThread, Signal


def _local_reference_names() -> set[str]:
    """Names already registered in the LOCAL catalog (the one we write to)."""
    from openbench.data.registry.manager import get_registry

    # get_registry() is cached and both registration paths invalidate the
    # cache after writing, so this is always fresh without a full re-parse.
    return {r.name for r in get_registry().list_references()}


def remote_scan_caveats(variants) -> str:
    """Describe what registration could not do for remote-scanned datasets."""
    station = sorted(
        {
            v.registry_name
            for v in variants
            if getattr(v, "data_type", "") == "stn" and not getattr(v, "remote_fulllist", "")
        }
    )
    if not station:
        return ""
    return (
        "Station fulllist generation failed on the remote host for: "
        + ", ".join(station)
        + ". Complete their fulllist manually in the Data Registry page."
    )


def scan_reference_datasets_remote(
    ssh_manager,
    data_root: str,
    *,
    python_path: str = "",
    conda_env: str = "",
    openbench_path: str = "",
    timeout: int = 900,
    should_abort=None,
):
    """Run reference registry discovery on the remote host and rehydrate groups.

    The remote script also performs the per-variable NetCDF inspection and
    data_groupby detection there (where the files actually live) and ships the
    results along, so local registration does not silently degrade. The
    "already registered" filter uses the LOCAL catalog names, because that is
    the catalog registration writes to.
    """
    from openbench.data.registry.scanner import DatasetGroup, ScannedDataset
    from openbench.gui.remote_python import run_remote_python_json

    existing_names = sorted(_local_reference_names())

    bootstrap = ""
    if openbench_path:
        root = openbench_path.rstrip("/")
        # Make a plain git checkout importable even when the pip-install
        # dependency step was skipped or failed on the remote host.
        bootstrap = (
            "import os\n"
            "import sys\n"
            f"for _path in ({json.dumps(root)}, {json.dumps(root + '/src')}):\n"
            "    _path = os.path.expanduser(_path)\n"  # '~/OpenBench' is the documented default
            "    if _path not in sys.path:\n"
            "        sys.path.insert(0, _path)\n"
        )

    script = f"""{bootstrap}
import dataclasses
import json

from openbench.data.registry.scanner import find_new_datasets

try:
    from openbench.data.registry.scanner import _detect_data_groupby, _expand_path, _inspect_nc_file
except ImportError:  # older remote checkout: scan still works, inspection degrades
    _detect_data_groupby = _expand_path = _inspect_nc_file = None

try:
    from openbench.data.coordinates import glob_nc as _glob_nc
    from openbench.data.registry.scanner import generate_station_list
except ImportError:  # older remote checkout: station fulllist degrades
    generate_station_list = _glob_nc = None


def _station_fulllist(variant):
    if generate_station_list is None or _expand_path is None:
        return ""
    nc_dir = _expand_path(variant.root_dir)
    first_sub = next(iter(variant.variables.values()), "")
    if first_sub:
        candidate = _expand_path(variant.root_dir) / first_sub
        if candidate.is_dir() and _glob_nc(candidate):
            nc_dir = candidate
        elif (candidate / "dataset").is_dir() and _glob_nc(candidate / "dataset"):
            nc_dir = candidate / "dataset"
    if not _glob_nc(nc_dir):
        return ""
    import pathlib

    lists_dir = pathlib.Path.home() / ".openbench" / "station_lists"
    lists_dir.mkdir(parents=True, exist_ok=True)
    output_csv = lists_dir / (variant.registry_name + ".csv")
    generate_station_list(nc_dir, output_csv)
    return str(output_csv)


def _json_default(value):
    item = getattr(value, "item", None)
    if callable(item):  # numpy scalars
        return item()
    return str(value)


groups = find_new_datasets({json.dumps(data_root)}, existing_names=set({json.dumps(existing_names)}))
payload = []
for group in groups:
    variants = {{}}
    for resolution, variant in group.variants.items():
        data = dataclasses.asdict(variant)
        inspections = {{}}
        if _inspect_nc_file is not None:
            for var_name, sub_dir in variant.variables.items():
                dataset_path = _expand_path(variant.root_dir) / sub_dir
                if dataset_path.is_dir():
                    file_glob = getattr(variant, "file_globs", {{}}).get(var_name)
                    inspections[var_name] = _inspect_nc_file(dataset_path, file_glob=file_glob)
        data["nc_inspections"] = inspections
        if _detect_data_groupby is not None:
            data["detected_data_groupby"] = _detect_data_groupby(variant)
        if variant.data_type == "stn":
            try:
                data["remote_fulllist"] = _station_fulllist(variant)
            except Exception:
                data["remote_fulllist"] = ""
        variants[resolution] = data
    payload.append({{"base_name": group.base_name, "variants": variants}})
print(json.dumps(payload, default=_json_default))
"""
    payload = run_remote_python_json(
        ssh_manager,
        script,
        python_path=python_path,
        conda_env=conda_env,
        timeout=timeout,
        should_abort=should_abort,
    )

    # Rehydrate tolerantly: the remote checkout may be a different OpenBench
    # version, so drop unknown fields instead of crashing on them.
    field_names = {f.name for f in dataclasses.fields(ScannedDataset)}
    groups = []
    for item in payload:
        variants = {}
        for resolution, variant in (item.get("variants") or {}).items():
            known = {key: value for key, value in variant.items() if key in field_names}
            try:
                variants[resolution] = ScannedDataset(**known)
            except TypeError as exc:
                raise RuntimeError(
                    "Remote scan returned data incompatible with this OpenBench "
                    f"version (local/remote version mismatch?): {exc}"
                ) from exc
        groups.append(DatasetGroup(base_name=item.get("base_name", ""), variants=variants))
    return groups


class FindDatasetsWorker(QThread):
    """Run registry discovery off the Qt main thread."""

    finished_with_result = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        data_root: str,
        parent=None,
        ssh_manager=None,
        python_path: str = "",
        conda_env: str = "",
        openbench_path: str = "",
    ):
        super().__init__(parent)
        self._data_root = data_root
        self._ssh_manager = ssh_manager
        self._python_path = python_path
        self._conda_env = conda_env
        self._openbench_path = openbench_path

    def run(self) -> None:  # pragma: no cover - exercised through GUI integration
        try:
            if self._ssh_manager is not None:
                result = scan_reference_datasets_remote(
                    self._ssh_manager,
                    self._data_root,
                    python_path=self._python_path,
                    conda_env=self._conda_env,
                    openbench_path=self._openbench_path,
                    should_abort=self.isInterruptionRequested,
                )
            else:
                from openbench.data.registry.scanner import find_new_datasets

                result = find_new_datasets(self._data_root)
            self.finished_with_result.emit(result)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
