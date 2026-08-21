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
    messages = []
    inspection = sorted(
        (
            v.registry_name,
            getattr(v, "remote_inspection_error", ""),
        )
        for v in variants
        if getattr(v, "remote_inspection_error", "")
    )
    if inspection:
        messages.append(
            "Remote NetCDF metadata/data_groupby inspection degraded for: "
            + ", ".join(f"{name} ({reason})" for name, reason in inspection)
            + "."
        )
    station = sorted(
        (
            v.registry_name,
            getattr(v, "remote_fulllist_error", "") or "reason not reported by the remote checkout",
        )
        for v in variants
        if getattr(v, "data_type", "") == "stn" and not getattr(v, "remote_fulllist", "")
    )
    if station:
        messages.append(
            "Station fulllist generation was unavailable for: "
            + ", ".join(f"{name} ({reason})" for name, reason in station)
            + ". Complete their fulllist manually in the Data Registry page."
        )
    return "\n".join(messages)


def unpack_scan_result(result):
    """Return ``(groups, skipped)`` while accepting legacy list results."""
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, []


def format_scan_skips(skipped) -> str:
    """Format unsupported reference folders for a GUI warning."""
    lines = [f"The scanner skipped {len(skipped)} unsupported folder(s):"]
    for item in skipped:
        path = getattr(item, "path", str(item))
        reason = getattr(item, "reason", "unsupported_layout")
        hint = getattr(item, "hint", "")
        lines.append(f"• {path}: {reason}")
        if hint:
            lines.append(f"  {hint}")
    return "\n".join(lines)


def scan_reference_datasets_remote(
    ssh_manager,
    data_root: str,
    *,
    python_path: str = "",
    conda_env: str = "",
    openbench_path: str = "",
    timeout: int = 900,
    should_abort=None,
    rescan: bool = False,
    only_names: set[str] | None = None,
    on_skip=None,
):
    """Run reference registry discovery on the remote host and rehydrate groups.

    The remote script performs expensive NetCDF inspection for new datasets,
    or for ``only_names`` when explicitly refreshing registered datasets.
    Local catalog names determine both discovery status and registration.
    """
    from openbench.data.registry.scanner import DatasetGroup, ScannedDataset, ScanSkip
    from openbench.gui.remote_python import run_remote_python_json

    registered_names = sorted(_local_reference_names())
    only_names_expr = "None" if only_names is None else f"set({json.dumps(sorted(only_names))})"

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
import inspect
import json

try:
    from openbench.data.registry.scanner import find_new_datasets
except ImportError:
    find_new_datasets = None

try:
    from openbench.data.registry.scanner import scan_reference_directory
except ImportError:
    scan_reference_directory = None

if find_new_datasets is None and scan_reference_directory is None:
    raise RuntimeError("remote OpenBench scanner API is unavailable")

try:
    from openbench.data.registry.scanner import _detect_data_groupby, _expand_path, _inspect_nc_file
    _inspection_import_error = ""
except ImportError as exc:  # older remote checkout: scan still works, inspection degrades
    _detect_data_groupby = _expand_path = _inspect_nc_file = None
    _inspection_import_error = "missing scanner metadata API: %s" % exc

try:
    from openbench.data.coordinates import glob_nc as _glob_nc
    from openbench.data.registry.scanner import generate_station_list, resolve_station_nc_dir
    _fulllist_import_error = ""
except ImportError as exc:  # older remote checkout: station fulllist degrades
    generate_station_list = _glob_nc = resolve_station_nc_dir = None
    _fulllist_import_error = "missing station-list API: %s" % exc


def _station_fulllist(variant):
    if generate_station_list is None or resolve_station_nc_dir is None or _glob_nc is None:
        return "", _fulllist_import_error or "station-list API unavailable"
    nc_dir = resolve_station_nc_dir(variant.root_dir, variant.variables)
    if not _glob_nc(nc_dir):
        return "", "no NetCDF files found in %s" % nc_dir
    import pathlib

    lists_dir = pathlib.Path.home() / ".openbench" / "station_lists"
    lists_dir.mkdir(parents=True, exist_ok=True)
    output_csv = lists_dir / (variant.registry_name + ".csv")
    generate_station_list(nc_dir, output_csv)
    return str(output_csv), ""


def _json_default(value):
    item = getattr(value, "item", None)
    if callable(item):  # numpy scalars
        return item()
    return str(value)


def _scan_with_skips(scan_fn, *args, **kwargs):
    parameters = inspect.signature(scan_fn).parameters
    kwargs = {{key: value for key, value in kwargs.items() if key in parameters}}
    if "on_skip" in parameters:
        kwargs["on_skip"] = skipped.append
    return scan_fn(*args, **kwargs)


skipped = []
registered_names = set({json.dumps(registered_names)})
only_names = {only_names_expr}
if {rescan!r}:
    scan_fn = scan_reference_directory or find_new_datasets
    scan_kwargs = {{}} if scan_reference_directory is not None else {{"existing_names": set()}}
else:
    scan_fn = find_new_datasets or scan_reference_directory
    scan_kwargs = {{"existing_names": registered_names}} if find_new_datasets is not None else {{}}
groups = _scan_with_skips(scan_fn, {json.dumps(data_root)}, **scan_kwargs)
payload = []
for group in groups:
    variants = {{}}
    for resolution, variant in group.variants.items():
        registry_name = variant.registry_name
        if only_names is not None and registry_name not in only_names:
            continue
        data = dataclasses.asdict(variant)
        if only_names is not None or registry_name not in registered_names:
            data["remote_inspection_error"] = _inspection_import_error
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
                    data["remote_fulllist"], data["remote_fulllist_error"] = _station_fulllist(variant)
                except Exception as exc:
                    data["remote_fulllist"] = ""
                    data["remote_fulllist_error"] = "generation failed: %s: %s" % (type(exc).__name__, exc)
        variants[resolution] = data
    if variants:
        payload.append({{"base_name": group.base_name, "variants": variants}})
print(json.dumps({{"groups": payload, "skipped": [dataclasses.asdict(item) for item in skipped]}}, default=_json_default))
"""
    payload = run_remote_python_json(
        ssh_manager,
        script,
        python_path=python_path,
        conda_env=conda_env,
        timeout=timeout,
        should_abort=should_abort,
    )

    raw_groups = payload.get("groups", []) if isinstance(payload, dict) else payload
    raw_skips = payload.get("skipped", []) if isinstance(payload, dict) else []
    if on_skip:
        skip_fields = {f.name for f in dataclasses.fields(ScanSkip)}
        for item in raw_skips:
            on_skip(ScanSkip(**{key: value for key, value in item.items() if key in skip_fields}))

    # Rehydrate tolerantly: the remote checkout may be a different OpenBench
    # version, so drop unknown fields instead of crashing on them.
    field_names = {f.name for f in dataclasses.fields(ScannedDataset)}
    groups = []
    for item in raw_groups:
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


def enrich_selected_remote_variants(
    ssh_manager,
    data_root: str,
    variants,
    *,
    existing_names: set[str],
    python_path: str = "",
    conda_env: str = "",
    openbench_path: str = "",
    parent=None,
):
    """Inspect registered variants only when the user explicitly selects them."""
    refresh_names = {variant.registry_name for variant in variants} & set(existing_names)
    if not refresh_names:
        return variants
    progress = None
    if parent is not None:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QProgressDialog

        progress = QProgressDialog("Inspecting selected reference datasets...", None, 0, 0, parent)
        progress.setWindowTitle("Inspecting")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)
        progress.show()
    try:
        groups = scan_reference_datasets_remote(
            ssh_manager,
            data_root,
            python_path=python_path,
            conda_env=conda_env,
            openbench_path=openbench_path,
            rescan=True,
            only_names=refresh_names,
        )
    finally:
        if progress is not None:
            progress.close()
            progress.deleteLater()
    refreshed = {
        variant.registry_name: variant
        for group in groups
        for variant in group.variants.values()
    }
    missing = refresh_names - set(refreshed)
    if missing:
        raise RuntimeError("Selected remote datasets disappeared during refresh: " + ", ".join(sorted(missing)))
    return [refreshed.get(variant.registry_name, variant) for variant in variants]


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
                skipped = []
                result = scan_reference_datasets_remote(
                    self._ssh_manager,
                    self._data_root,
                    python_path=self._python_path,
                    conda_env=self._conda_env,
                    openbench_path=self._openbench_path,
                    should_abort=self.isInterruptionRequested,
                    rescan=True,
                    on_skip=skipped.append,
                )
            else:
                from openbench.data.registry.scanner import scan_reference_directory

                skipped = []
                result = scan_reference_directory(self._data_root, on_skip=skipped.append)
            self.finished_with_result.emit((result, skipped))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
