import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PySide6 = pytest.importorskip("PySide6")

from openbench.remote.storage import RemoteStorage  # noqa: E402
from tests.gui_fakes import FakeButton, FakeControllerBase  # noqa: E402


class FakeSSH:
    is_connected = True

    def __init__(self, stdout_payload):
        self.stdout_payload = stdout_payload
        self.commands = []

    def execute(self, command, timeout=None):
        self.commands.append((command, timeout))
        return self.stdout_payload, "", 0


def test_remote_reference_scan_rehydrates_dataset_groups():
    from openbench.gui.pages._scan_worker import scan_reference_datasets_remote

    payload = json.dumps(
        [
            {
                "base_name": "RemoteSet",
                "variants": {
                    "LowRes": {
                        "name": "RemoteSet",
                        "resolution": "LowRes",
                        "category": "Water",
                        "data_type": "grid",
                        "root_dir": "/remote/ref/Grid/LowRes/Water",
                        "variables": {"Runoff": "Runoff/RemoteSet"},
                        "file_globs": {"Runoff": "*.nc4"},
                        "file_count": 2,
                        "tim_res": "Day",
                    }
                },
            }
        ]
    )
    ssh = FakeSSH(payload)

    groups = scan_reference_datasets_remote(
        ssh, "/remote/ref path", python_path="/opt/openbench/bin/python", conda_env="ob env"
    )

    assert len(groups) == 1
    assert groups[0].base_name == "RemoteSet"
    variant = groups[0].variants["LowRes"]
    assert variant.registry_name == "RemoteSet_LowRes"
    assert variant.root_dir == "/remote/ref/Grid/LowRes/Water"
    assert variant.variables == {"Runoff": "Runoff/RemoteSet"}
    command, timeout = ssh.commands[0]
    assert "conda activate" in command and "ob env" in command
    assert "base64 -d" in command
    assert "/opt/openbench/bin/python" in command
    assert timeout == 900


def test_remote_reference_scan_rebases_remote_openbench_ref_root_placeholder(monkeypatch):
    from openbench.gui.pages import _scan_worker

    _capture_remote_json(
        monkeypatch,
        result={
            "groups": [
                {
                    "base_name": "GLEAM",
                    "variants": {
                        "LowRes": {
                            "name": "GLEAM",
                            "resolution": "LowRes",
                            "category": "Water",
                            "data_type": "grid",
                            "root_dir": "${OPENBENCH_REF_ROOT}/Grid/LowRes/Water",
                            "variables": {"Runoff": "Runoff/GLEAM"},
                        }
                    },
                }
            ],
            "skipped": [],
            "data_root": "/home/remote/ref",
        },
    )
    monkeypatch.setattr(_scan_worker, "_local_reference_names", lambda: set())

    (group,) = _scan_worker.scan_reference_datasets_remote(object(), "/remote/ref")

    assert group.variants["LowRes"].root_dir == "/home/remote/ref/Grid/LowRes/Water"


def test_remote_reference_scan_script_sets_absolute_ref_root(monkeypatch):
    from openbench.gui.pages import _scan_worker

    captured = _capture_remote_json(monkeypatch, result={"groups": [], "skipped": [], "data_root": "/home/u/Reference"})
    monkeypatch.setattr(_scan_worker, "_local_reference_names", lambda: set())

    _scan_worker.scan_reference_datasets_remote(object(), "~/Reference")

    script = captured["script"]
    assert '_data_root = os.path.abspath(os.path.expanduser("~/Reference"))' in script
    assert 'os.environ["OPENBENCH_REF_ROOT"] = _data_root' in script
    assert "groups = _scan_with_skips(scan_fn, _data_root" in script
    assert '"data_root": _data_root' in script
    compile(script, "<remote-scan-script>", "exec")


def test_register_scanned_datasets_remote_writes_remote_user_registry(monkeypatch):
    from openbench.data.registry.scanner import ScannedDataset
    from openbench.gui.pages import _scan_worker

    captured = _capture_remote_json(
        monkeypatch,
        result={"catalog_path": "/home/u/.openbench/references/reference_catalog.yaml"},
    )
    dataset = ScannedDataset("Demo", "LowRes", "Water", "grid", "/remote/ref/Grid/LowRes/Water", {"Runoff": "Demo"})

    result = _scan_worker.register_scanned_datasets_remote(
        object(),
        [dataset],
        "~/Reference",
        python_path="/remote/python",
        conda_env="ob",
        openbench_path="~/OpenBench",
    )

    script = captured["script"]
    assert result["catalog_path"].endswith("reference_catalog.yaml")
    assert "register_scanned_datasets_batch(datasets)" in script
    assert '_data_root = os.path.abspath(os.path.expanduser("~/Reference"))' in script
    assert 'os.environ["OPENBENCH_REF_ROOT"] = _data_root' in script
    assert "remember_reference_root(_data_root)" in script
    remember_index = script.index("remember_reference_root(_data_root)")
    register_index = script.index("register_scanned_datasets_batch(datasets)")
    assert remember_index < register_index
    assert "/remote/ref/Grid/LowRes/Water" in script
    assert captured["python_path"] == "/remote/python"
    assert captured["conda_env"] == "ob"
    assert captured["should_abort"] is None
    compile(script, "<remote-register-script>", "exec")


def test_remote_register_script_payload_decodes_to_dataset_list(monkeypatch):
    import ast
    import re

    from openbench.data.registry.scanner import ScannedDataset
    from openbench.gui.pages import _scan_worker

    captured = _capture_remote_json(monkeypatch, result={})
    dataset = ScannedDataset("Demo", "LowRes", "Water", "grid", "/remote/ref", {"Runoff": "Demo"})

    _scan_worker.register_scanned_datasets_remote(object(), [dataset], "/remote/ref")

    match = re.search(r"for item in json\.loads\((.*)\):", captured["script"])
    assert match is not None
    decoded = json.loads(ast.literal_eval(match.group(1)))
    assert isinstance(decoded, list)
    assert decoded[0]["name"] == "Demo"
    assert decoded[0]["root_dir"] == "/remote/ref"


def test_nc_importer_variable_rows_shared_between_local_and_remote(qapp):
    """One extraction function backs both the local table and the remote script."""
    import xarray as xr

    from openbench.gui.widgets import nc_importer

    ds = xr.Dataset(
        {"tas": (("time", "lat", "lon"), [[[1.0, 2.0, 3.0]]])},
        coords={"time": [0], "lat": [0.0], "lon": [0.0, 1.0, 2.0]},
    )
    ds["tas"].attrs["units"] = "K"

    rows = nc_importer._variable_rows(ds)

    tas = next(r for r in rows if r["name"] == "tas")
    assert tas["units"] == "K"
    assert tas["is_coord"] is False
    # The remote inspector script embeds this exact function's source instead
    # of re-hardcoding the extraction rules.
    dlg = nc_importer.NCImporterDialog(ssh_manager=FakeSSH("{}"))
    monkeyed = {}
    dlg._python_path = ""
    dlg._conda_env = ""

    import openbench.gui.remote_python as rp

    original = rp.run_remote_python_json
    rp.run_remote_python_json = lambda ssh, script, **kw: (
        monkeyed.setdefault("script", script)
        or {
            "path": "x",
            "data_var_count": 0,
            "variables": [],
        }
    )
    try:
        dlg._open_remote_file("/remote/x.nc")
    finally:
        rp.run_remote_python_json = original

    assert "def _variable_rows" in monkeyed["script"]
    assert "remote_path = os.path.expanduser" in monkeyed["script"]
    assert "xr.open_dataset(remote_path)" in monkeyed["script"]


def test_nc_importer_opens_remote_netcdf_metadata(qapp):
    from openbench.gui.widgets.nc_importer import NCImporterDialog

    payload = json.dumps(
        {
            "path": "/remote/data/sample.nc",
            "data_var_count": 2,
            "variables": [
                {
                    "name": "tas",
                    "dtype": "float32",
                    "dims": [["time", 12], ["lat", 2], ["lon", 3]],
                    "units": "K",
                    "is_coord": False,
                },
                {
                    "name": "time_bnds",
                    "dtype": "float64",
                    "dims": [["time", 12], ["bnds", 2]],
                    "units": "",
                    "is_coord": True,
                },
            ],
        }
    )
    ssh = FakeSSH(payload)
    dlg = NCImporterDialog(ssh_manager=ssh, python_path="/opt/py/bin/python", conda_env="base")
    dlg.edit_path.setText("/remote/data/sample.nc")

    dlg._open_file()

    assert dlg.info_label.text() == "Opened: /remote/data/sample.nc  (2 data variables)"
    assert dlg.table.rowCount() == 2
    assert dlg.table.item(0, 1).text() == "tas"
    assert dlg.table.item(0, 3).text() == "time(12), lat(2), lon(3)"
    assert dlg.table.cellWidget(0, 0).isChecked() is True
    assert dlg.table.cellWidget(1, 0).isChecked() is False
    command, timeout = ssh.commands[0]
    assert "conda activate base" in command
    assert "| base64 -d | /opt/py/bin/python" in command
    assert timeout == 60


def test_remote_python_command_uses_conda_sh_when_base_derivable():
    from openbench.gui.remote_python import build_remote_python_command

    cmd = build_remote_python_command("print(1)", python_path="/opt/miniconda3/envs/ob/bin/python", conda_env="ob")

    assert ". /opt/miniconda3/etc/profile.d/conda.sh && conda activate ob && " in cmd
    assert "~/.bashrc" not in cmd
    assert cmd.endswith("| base64 -d | /opt/miniconda3/envs/ob/bin/python")


def test_remote_python_command_uses_nonstandard_conda_root():
    from openbench.gui.remote_python import build_remote_python_command

    cmd = build_remote_python_command("print(1)", python_path="/shared/apps/conda/envs/ob/bin/python", conda_env="ob")

    assert ". /shared/apps/conda/etc/profile.d/conda.sh && conda activate ob && " in cmd


def test_conda_wrapped_commands_stay_posix_under_sh_wrapper():
    """SSHManager wraps every command in `sh -c`; `source` is a bashism that
    dash/ash reject, so the conda activation chain must stay pure POSIX."""
    from openbench.gui.remote_python import build_remote_python_command
    from openbench.remote.ssh import SSHManager

    cmd = build_remote_python_command("print(1)", python_path="/opt/miniconda3/bin/python", conda_env="ob")
    final = SSHManager._shell_agnostic(cmd)

    assert "source " not in final
    assert final.startswith("sh -c ")


def test_conda_base_with_tilde_expands_to_remote_home():
    from openbench.gui.remote_python import build_remote_python_command

    cmd = build_remote_python_command("print(1)", python_path="~/miniconda3/envs/ob/bin/python", conda_env="ob")

    # shlex-quoting a literal '~' would make the shell look for a directory
    # named '~'; the conda base AND the runner python must both expand to
    # the remote $HOME.
    assert '. "$HOME"/miniconda3/etc/profile.d/conda.sh && conda activate ob && ' in cmd
    assert cmd.endswith('| base64 -d | "$HOME"/miniconda3/envs/ob/bin/python')
    assert "'~/" not in cmd


def test_runner_python_with_tilde_expands_without_conda_env():
    from openbench.gui.remote_python import build_remote_python_command

    cmd = build_remote_python_command("print(1)", python_path="~/venv/bin/python")

    assert cmd.endswith('| base64 -d | "$HOME"/venv/bin/python')
    assert "'~/" not in cmd


def test_remote_run_command_expands_tilde_paths():
    from openbench.gui.remote_runner import build_remote_run_command

    cmd = build_remote_run_command(
        python_path="~/miniconda3/envs/ob/bin/python",
        openbench_path="~/OpenBench",
        config_path="/tmp/openbench.yaml",
        conda_env="",
    )

    # '~/OpenBench' is the documented remote default; a shlex-quoted literal
    # tilde would make both cd and the interpreter path fail remotely.
    assert 'cd "$HOME"/OpenBench && ' in cmd
    assert "OPENBENCH_GUI_PROGRESS=1" in cmd
    assert '"$HOME"/miniconda3/envs/ob/bin/python -u -m openbench check' in cmd
    assert '"$HOME"/miniconda3/envs/ob/bin/python -u -m openbench run' in cmd
    assert "__OPENBENCH_PHASE__=run_started" in cmd
    assert "__OPENBENCH_PHASE__=run_completed" in cmd
    assert "'~/" not in cmd


def test_remote_python_command_falls_back_to_login_shell_without_conda_base():
    from openbench.gui.remote_python import build_remote_python_command

    cmd = build_remote_python_command("print(1)", conda_env="ob")

    # Non-interactive ~/.bashrc sourcing silently no-ops (interactivity
    # guard returns before conda init); a login shell at least runs the
    # profile chain, and && makes activation failure visible.
    assert cmd.startswith("bash -l -c ")
    assert "conda activate ob && " in cmd
    assert "~/.bashrc" not in cmd


def test_remote_python_command_without_env_is_bare_pipe():
    from openbench.gui.remote_python import build_remote_python_command

    cmd = build_remote_python_command("print(1)")

    assert cmd.startswith("printf %s ")
    assert cmd.endswith("| base64 -d | python3")


def test_run_remote_python_json_uploads_large_scripts_and_cleans_up(monkeypatch):
    import openbench.gui.remote_python as rp

    monkeypatch.setattr(rp, "_MAX_INLINE_SCRIPT_CHARS", 10)

    class SSH:
        def __init__(self):
            self.uploads = []
            self.commands = []

        def upload_file(self, local_path, remote_path):
            self.uploads.append((remote_path, Path(local_path).read_text(encoding="utf-8")))

        def execute(self, command, timeout=None):
            self.commands.append((command, timeout))
            if command.startswith("rm -f "):
                return "", "", 0
            return '{"ok": true}\n', "", 0

    script = "print('x')\n" * 20
    ssh = SSH()

    assert rp.run_remote_python_json(ssh, script, python_path="~/venv/bin/python", timeout=77) == {"ok": True}

    assert len(ssh.uploads) == 1
    remote_path, uploaded = ssh.uploads[0]
    assert remote_path.startswith("/tmp/openbench-python-")
    assert uploaded == script
    assert "base64 -d" not in ssh.commands[0][0]
    assert '"$HOME"/venv/bin/python' in ssh.commands[0][0]
    assert remote_path in ssh.commands[0][0]
    assert ssh.commands[0][1] == 77
    assert ssh.commands[-1][0] == f"rm -f {remote_path}"


def test_data_validator_inspect_uses_shared_remote_python_command(monkeypatch):
    from openbench.gui.data_validator import RemoteNetCDFValidator

    validator = RemoteNetCDFValidator.__new__(RemoteNetCDFValidator)
    validator._python_path = "/opt/miniconda3/bin/python"
    validator._conda_env = "ob"

    class SSH:
        def __init__(self):
            self.commands = []

        def execute(self, command, timeout=None):
            self.commands.append(command)
            return "{}", "", 0

    validator._ssh = SSH()

    assert validator._run_inspect_script("/remote/x.nc") == {}
    command = validator._ssh.commands[0]
    assert ". /opt/miniconda3/etc/profile.d/conda.sh && conda activate ob && " in command
    assert "source " not in command
    assert "~/.bashrc" not in command


def test_remote_grid_single_time_check_rejects_missing_time_dimension():
    from openbench.gui.data_validator import RemoteNetCDFValidator

    validator = RemoteNetCDFValidator(object())

    check = validator.check_time_range(
        "/remote/ref/runoff.nc",
        2000,
        2001,
        {"success": True, "variables": ["q"], "time_missing": True},
    )

    assert check.passed is False
    assert "Time dimension not found" in check.message


def test_data_validator_inspect_accepts_login_banner_before_json():
    from openbench.gui.data_validator import RemoteNetCDFValidator

    class SSH:
        def __init__(self):
            self.commands = []

        def execute(self, command, timeout=None):
            self.commands.append((command, timeout))
            return 'Last login: Fri Aug 28\n{"success": true, "variables": ["tas"]}\n', "", 0

    validator = RemoteNetCDFValidator(SSH())

    result = validator.inspect_file("/remote/x.nc")

    assert result == {"success": True, "variables": ["tas"]}
    assert validator._ssh.commands[0][1] == 30


def test_remote_reference_scan_reports_registry_load_failures(monkeypatch):
    from openbench.data.registry import manager as registry_manager
    from openbench.gui.pages import _scan_worker

    def fake_run(ssh_manager, script, **kwargs):
        exec(compile(script, "<remote-scan-script>", "exec"), {})

    def fail_registry():
        raise PermissionError("bad registry yaml")

    monkeypatch.setattr("openbench.gui.remote_python.run_remote_python_json", fake_run)
    monkeypatch.setattr(registry_manager, "get_registry", fail_registry)

    with pytest.raises(RuntimeError, match="remote registry load failed: PermissionError: bad registry yaml"):
        _scan_worker.scan_reference_datasets_remote(object(), "/remote/ref")


def test_remote_reference_scan_script_does_not_hide_registry_load_failures(monkeypatch):
    from openbench.gui.pages import _scan_worker

    captured = _capture_remote_json(monkeypatch, result={"groups": [], "skipped": [], "data_root": "/remote/ref"})

    _scan_worker.scan_reference_datasets_remote(object(), "/remote/ref")

    script = captured["script"]
    assert "remote registry load failed" in script
    assert "registered_names = set()" not in script


def _capture_remote_json(monkeypatch, result=None):
    captured = {}

    def fake_run(ssh_manager, script, *, python_path="", conda_env="", timeout=60, should_abort=None):
        captured["script"] = script
        captured["python_path"] = python_path
        captured["conda_env"] = conda_env
        captured["timeout"] = timeout
        captured["should_abort"] = should_abort
        return [] if result is None else result

    monkeypatch.setattr("openbench.gui.remote_python.run_remote_python_json", fake_run)
    return captured


def test_remote_scan_script_bootstraps_openbench_path_and_remote_names(monkeypatch):
    from openbench.gui.pages import _scan_worker

    captured = _capture_remote_json(monkeypatch)
    _scan_worker.scan_reference_datasets_remote(object(), "/remote/ref", openbench_path="/remote/openbench")

    script = captured["script"]
    assert "sys.path.insert" in script
    assert '"/remote/openbench/src"' in script
    assert "Already_LowRes" not in script
    assert "registered_names = {ref.name for ref in get_registry().list_references()}" in script
    assert '"existing_names"' in script
    assert captured["timeout"] == 900


def test_remote_scan_bootstrap_expands_tilde_openbench_path(monkeypatch):
    from openbench.gui.pages import _scan_worker

    captured = _capture_remote_json(monkeypatch)
    monkeypatch.setattr(_scan_worker, "_local_reference_names", lambda: set())

    _scan_worker.scan_reference_datasets_remote(object(), "/remote/ref", openbench_path="~/OpenBench")

    script = captured["script"]
    # Python never expands '~' in sys.path entries; the script must do it.
    assert "expanduser" in script
    compile(script, "<remote-scan-script>", "exec")


def test_find_datasets_worker_passes_interruption_probe(qapp, monkeypatch):
    from openbench.gui.pages import _scan_worker

    captured = {}
    monkeypatch.setattr(
        _scan_worker,
        "scan_reference_datasets_remote",
        lambda *args, **kwargs: captured.update(kwargs) or [],
    )

    worker = _scan_worker.FindDatasetsWorker("/remote/ref", ssh_manager=object())
    worker.run()

    assert callable(captured.get("should_abort"))
    assert captured["rescan"] is True
    assert callable(captured["on_skip"])


def test_local_gui_reference_scan_refreshes_existing_entries_and_keeps_skips(qapp, monkeypatch):
    from openbench.data.registry import scanner
    from openbench.gui.pages import _scan_worker

    skip = scanner.ScanSkip("Grid/LowRes/Water/Bad", "unsupported_layout", "Register it manually.")

    def fake_scan(root, on_skip=None):
        assert root == "/local/ref"
        on_skip(skip)
        return ["all scanned groups"]

    monkeypatch.setattr(scanner, "scan_reference_directory", fake_scan)
    results = []
    worker = _scan_worker.FindDatasetsWorker("/local/ref")
    worker.finished_with_result.connect(results.append)

    worker.run()

    assert results == [(["all scanned groups"], [skip])]


def test_remote_reference_scan_rehydrates_skipped_folders(monkeypatch):
    from openbench.gui.pages import _scan_worker

    _capture_remote_json(
        monkeypatch,
        result={
            "groups": [],
            "skipped": [
                {
                    "path": "Grid/LowRes/Water/Bad",
                    "reason": "unsupported_layout",
                    "hint": "Register it manually.",
                }
            ],
        },
    )
    monkeypatch.setattr(_scan_worker, "_local_reference_names", lambda: set())
    skipped = []

    groups = _scan_worker.scan_reference_datasets_remote(object(), "/remote/ref", on_skip=skipped.append)

    assert groups == []
    assert [(item.path, item.reason, item.hint) for item in skipped] == [
        ("Grid/LowRes/Water/Bad", "unsupported_layout", "Register it manually.")
    ]


def test_gui_reference_scan_skip_message_includes_remediation():
    from openbench.data.registry.scanner import ScanSkip
    from openbench.gui.pages._scan_worker import format_scan_skips

    message = format_scan_skips([ScanSkip("Grid/LowRes/Water/Bad", "unsupported_layout", "Register it manually.")])

    assert "Grid/LowRes/Water/Bad: unsupported_layout" in message
    assert "Register it manually." in message


def test_show_scan_incomplete_truncates_text_and_keeps_full_details(monkeypatch):
    from openbench.data.registry.scanner import ScanSkip
    from openbench.gui.pages import _scan_worker

    captured = {}

    class FakeBox:
        Warning = "warning"

        def __init__(self, parent):
            captured["parent"] = parent

        def setIcon(self, value):
            captured["icon"] = value

        def setWindowTitle(self, value):
            captured["title"] = value

        def setText(self, value):
            captured["text"] = value

        def setDetailedText(self, value):
            captured["details"] = value

        def exec(self):
            captured["exec"] = True

    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox", FakeBox)
    skipped = [ScanSkip(f"Bad{i}", "unsupported_layout", "Fix manually.") for i in range(25)]

    _scan_worker.show_scan_incomplete("parent", skipped)

    assert "Bad0" not in captured["text"]
    assert "25 unsupported folder" in captured["text"]
    assert "Details" in captured["text"]
    assert "Bad0" in captured["details"]
    assert "Bad24" in captured["details"]
    assert captured["exec"] is True


def test_enrich_selected_remote_variants_cancel_uses_event(monkeypatch):
    from types import SimpleNamespace

    from openbench.gui.pages import _scan_worker

    slots = []
    captured = {}

    class FakeSignal:
        def connect(self, slot):
            slots.append(slot)

    class FakeProgress:
        canceled = FakeSignal()

        def __init__(self, *args, **_kwargs):
            captured["args"] = args

        def setWindowTitle(self, *_args):
            pass

        def setWindowModality(self, *_args):
            pass

        def setMinimumDuration(self, *_args):
            pass

        def show(self):
            slots[0]()

        def close(self):
            captured["closed"] = True

        def deleteLater(self):
            captured["deleted"] = True

    def fake_scan(*_args, should_abort=None, **_kwargs):
        assert should_abort is not None and should_abort() is True
        raise RuntimeError("aborted")

    monkeypatch.setattr("PySide6.QtWidgets.QProgressDialog", FakeProgress)
    monkeypatch.setattr(_scan_worker, "scan_reference_datasets_remote", fake_scan)

    variant = SimpleNamespace(registry_name="Existing")
    try:
        _scan_worker.enrich_selected_remote_variants(
            object(), "/remote/ref", [variant], existing_names={"Existing"}, parent="parent"
        )
    except InterruptedError:
        pass
    else:
        raise AssertionError("cancel should stop enrichment")

    assert captured["args"][1] == "Cancel"
    assert captured["closed"] is True
    assert captured["deleted"] is True


def test_remote_scan_caveats_keeps_complete_long_station_details():
    from types import SimpleNamespace

    from openbench.gui.pages._scan_worker import remote_scan_caveats

    variants = [
        SimpleNamespace(
            registry_name=f"Station{i}",
            data_type="stn",
            remote_fulllist="",
            remote_fulllist_error="missing station-list API",
        )
        for i in range(20)
    ]

    message = remote_scan_caveats(variants)

    assert "Station fulllist generation was unavailable for:" in message
    for i in range(20):
        assert f"• Station{i}: missing station-list API" in message
    assert "and 12 more" not in message
    assert "Reason counts" not in message


def test_scan_complete_with_details_uses_qmessagebox_details(monkeypatch):
    from openbench.gui.pages import _scan_worker

    captured = {}

    class FakeBox:
        Information = "info"

        def __init__(self, parent):
            captured["parent"] = parent

        def setIcon(self, value):
            captured["icon"] = value

        def setWindowTitle(self, value):
            captured["title"] = value

        def setText(self, value):
            captured["text"] = value

        def setDetailedText(self, value):
            captured["details"] = value

        def exec(self):
            captured["exec"] = True

        @staticmethod
        def information(*_args):
            captured["information"] = True

    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox", FakeBox)

    details = "\n".join(f"Station{i} detail" for i in range(20))

    _scan_worker.show_scan_complete("parent", "Registered/updated 2 dataset(s).", details)

    assert captured == {
        "parent": "parent",
        "icon": "info",
        "title": "Scan Complete",
        "text": "Registered/updated 2 dataset(s).",
        "details": details,
        "exec": True,
    }
    assert "Station19 detail" in captured["details"]


def test_scan_complete_without_details_keeps_information_shortcut(monkeypatch):
    from openbench.gui.pages import _scan_worker

    captured = {}

    class FakeBox:
        Information = "info"

        def __init__(self, *_args):
            captured["constructed"] = True

        @staticmethod
        def information(*args):
            captured["information"] = args

    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox", FakeBox)

    _scan_worker.show_scan_complete("parent", "No supported reference datasets found.")

    assert captured == {"information": ("parent", "Scan Complete", "No supported reference datasets found.")}


def test_remote_ref_scan_register_worker_receives_remote_context(monkeypatch):
    from openbench.data.registry.scanner import DatasetGroup, ScannedDataset
    from openbench.gui.pages import page_ref_data
    from openbench.gui.pages.page_ref_data import PageRefData

    variant = ScannedDataset("Found", "LowRes", "Water", "grid", "/remote/ref/Grid/LowRes/Water", {"Runoff": "runoff"})
    group = DatasetGroup("Found", {"LowRes": variant})
    captured = {}

    class FakeDialog:
        def __init__(self, _groups, parent=None, *, existing_names=None):
            pass

        def exec(self):
            return True

        def get_selected(self):
            return [("Found", "LowRes", variant)]

    class FakeRegisterWorker:
        def __init__(self, variants, **kwargs):
            captured["variants"] = variants
            captured["kwargs"] = kwargs
            self.finished_with_result = FakeSignal()
            self.failed = FakeSignal()
            self.finished = FakeSignal()

        def start(self):
            captured["started"] = True

        def deleteLater(self):
            pass

    registry = SimpleNamespace(list_references=lambda: [], get_reference=lambda _name: None)
    monkeypatch.setattr(PageRefData, "_finish_scan_worker", lambda _self: None)
    monkeypatch.setattr("openbench.data.registry.manager.get_registry", lambda: registry)
    monkeypatch.setattr("openbench.gui.dialogs.data_discovery.DataDiscoveryDialog", FakeDialog)
    monkeypatch.setattr("openbench.gui.pages._scan_worker.RegisterScannedDatasetsWorker", FakeRegisterWorker)
    monkeypatch.setattr(page_ref_data, "QProgressDialog", FakeProgress)

    page = PageRefData.__new__(PageRefData)
    page.btn_scan = FakeButton()
    page._scan_remote_context = (
        "/remote/ref",
        {"ssh_manager": "ssh", "python_path": "/py", "conda_env": "ob", "openbench_path": "/ob"},
    )

    PageRefData._on_scan_data_root_finished(page, ([group], []))

    assert captured["variants"] == [variant]
    assert captured["kwargs"] == {
        "data_root": "/remote/ref",
        "ssh_manager": "ssh",
        "python_path": "/py",
        "conda_env": "ob",
        "openbench_path": "/ob",
    }
    assert captured["started"] is True


def test_remote_registry_scan_register_worker_receives_remote_context(monkeypatch):
    from openbench.data.registry.scanner import DatasetGroup, ScannedDataset
    from openbench.gui.pages import page_registry
    from openbench.gui.pages.page_registry import PageRegistry

    variant = ScannedDataset("Found", "LowRes", "Water", "grid", "/remote/ref/Grid/LowRes/Water", {"Runoff": "runoff"})
    group = DatasetGroup("Found", {"LowRes": variant})
    captured = {}

    class FakeDialog:
        def __init__(self, _groups, parent=None, *, existing_names=None):
            pass

        def exec(self):
            return True

        def get_selected(self):
            return [("Found", "LowRes", variant)]

    class FakeRegisterWorker:
        def __init__(self, variants, **kwargs):
            captured["variants"] = variants
            captured["kwargs"] = kwargs
            self.finished_with_result = FakeSignal()
            self.failed = FakeSignal()
            self.finished = FakeSignal()

        def start(self):
            captured["started"] = True

        def deleteLater(self):
            pass

    registry = SimpleNamespace(list_references=lambda: [], get_reference=lambda _name: None)
    monkeypatch.setattr(PageRegistry, "_finish_scan_worker", lambda _self: None)
    monkeypatch.setattr(page_registry, "_get_registry", lambda: registry)
    monkeypatch.setattr("openbench.gui.dialogs.data_discovery.DataDiscoveryDialog", FakeDialog)
    monkeypatch.setattr("openbench.gui.pages._scan_worker.RegisterScannedDatasetsWorker", FakeRegisterWorker)
    monkeypatch.setattr(page_registry, "QProgressDialog", FakeProgress)

    page = PageRegistry.__new__(PageRegistry)
    page._scan_remote_context = (
        "/remote/ref",
        {"ssh_manager": "ssh", "python_path": "/py", "conda_env": "ob", "openbench_path": "/ob"},
    )

    PageRegistry._on_scan_directory_finished(page, ([group], []))

    assert captured["variants"] == [variant]
    assert captured["kwargs"] == {
        "data_root": "/remote/ref",
        "ssh_manager": "ssh",
        "python_path": "/py",
        "conda_env": "ob",
        "openbench_path": "/ob",
    }
    assert captured["started"] is True


def test_remote_scan_script_attaches_remote_inspections(monkeypatch):
    from openbench.gui.pages import _scan_worker

    captured = _capture_remote_json(monkeypatch)
    monkeypatch.setattr(_scan_worker, "_local_reference_names", lambda: set())

    _scan_worker.scan_reference_datasets_remote(object(), "/remote/ref")

    script = captured["script"]
    assert "_inspect_nc_file" in script
    assert "_detect_data_groupby" in script
    assert "nc_inspections" in script
    assert "remote_inspection_error" in script
    assert "registry_name not in registered_names" in script
    assert "from openbench.data.registry.scanner import find_new_datasets, scan_reference_directory" not in script
    compile(script, "<remote-scan>", "exec")


def test_remote_scan_script_limits_registered_refresh_to_selected_names(monkeypatch):
    from openbench.data.registry.scanner import ScannedDataset
    from openbench.gui.pages import _scan_worker

    captured = _capture_remote_json(monkeypatch)
    monkeypatch.setattr(_scan_worker, "_local_reference_names", lambda: {"Existing_LowRes", "Other_LowRes"})
    existing = ScannedDataset("Existing", "LowRes", "Water", "grid", "/ref", {"Runoff": "existing"})

    _scan_worker.scan_reference_datasets_remote(
        object(),
        "/remote/ref",
        rescan=True,
        only_names={"Existing_LowRes"},
        selected_variants=[existing],
    )

    script = captured["script"]
    assert 'only_names = set(["Existing_LowRes"])' in script
    assert "if selected_variants is not None" in script
    assert "if only_names is not None and registry_name not in only_names" in script
    compile(script, "<selected-remote-refresh>", "exec")


def test_remote_refresh_enriches_only_selected_registered_variants(monkeypatch):
    from openbench.data.registry.scanner import DatasetGroup, ScannedDataset
    from openbench.gui.pages import _scan_worker

    existing = ScannedDataset("Existing", "LowRes", "Water", "grid", "/ref", {"Runoff": "existing"})
    new = ScannedDataset("New", "LowRes", "Water", "grid", "/ref", {"Runoff": "new"})
    enriched = ScannedDataset(
        "Existing",
        "LowRes",
        "Water",
        "grid",
        "/ref",
        {"Runoff": "existing"},
        nc_inspections={"Runoff": {"varname": "runoff"}},
    )
    captured = {}

    def fake_scan(*_args, **kwargs):
        captured.update(kwargs)
        return [DatasetGroup("Existing", {"LowRes": enriched})]

    monkeypatch.setattr(_scan_worker, "scan_reference_datasets_remote", fake_scan)

    variants = _scan_worker.enrich_selected_remote_variants(
        object(),
        "/remote/ref",
        [existing, new],
        existing_names={existing.registry_name},
    )

    assert captured["only_names"] == {existing.registry_name}
    assert captured["selected_variants"] == [existing]
    assert variants == [enriched, new]


def test_remote_scan_rehydration_ignores_unknown_fields(monkeypatch):
    from openbench.gui.pages import _scan_worker

    payload = [
        {
            "base_name": "X",
            "variants": {
                "LowRes": {
                    "name": "X",
                    "resolution": "LowRes",
                    "category": "Water",
                    "data_type": "grid",
                    "root_dir": "/r",
                    "field_from_a_newer_remote_version": 123,
                }
            },
        }
    ]
    _capture_remote_json(monkeypatch, result=payload)
    monkeypatch.setattr(_scan_worker, "_local_reference_names", lambda: set())

    groups = _scan_worker.scan_reference_datasets_remote(object(), "/remote/ref")

    assert groups[0].variants["LowRes"].name == "X"


def test_remote_scan_rehydration_reports_version_mismatch(monkeypatch):
    from openbench.gui.pages import _scan_worker

    payload = [{"base_name": "X", "variants": {"LowRes": {"name": "X"}}}]
    _capture_remote_json(monkeypatch, result=payload)
    monkeypatch.setattr(_scan_worker, "_local_reference_names", lambda: set())

    with pytest.raises(RuntimeError, match="version"):
        _scan_worker.scan_reference_datasets_remote(object(), "/remote/ref")


def test_remote_scan_script_generates_station_fulllists(monkeypatch):
    from openbench.gui.pages import _scan_worker

    captured = _capture_remote_json(monkeypatch)
    monkeypatch.setattr(_scan_worker, "_local_reference_names", lambda: set())

    _scan_worker.scan_reference_datasets_remote(object(), "/remote/ref")

    script = captured["script"]
    assert "generate_station_list" in script
    assert "remote_fulllist" in script
    assert "remote_fulllist_error" in script
    assert "station_lists" in script
    compile(script, "<remote-scan-script>", "exec")  # assembled f-string must be valid Python


def test_remote_scan_caveats_skips_station_datasets_with_remote_fulllist():
    from openbench.data.registry.scanner import ScannedDataset
    from openbench.gui.pages._scan_worker import remote_scan_caveats

    covered = ScannedDataset(
        name="Covered",
        resolution="Station",
        category="Water",
        data_type="stn",
        root_dir="/r",
        remote_fulllist="/remote/home/.openbench/station_lists/Covered.csv",
    )
    uncovered = ScannedDataset(
        name="Uncovered",
        resolution="Station",
        category="Water",
        data_type="stn",
        root_dir="/r",
        remote_fulllist_error="no NetCDF files found",
    )

    message = remote_scan_caveats([covered, uncovered])

    assert "Uncovered" in message
    assert "no NetCDF files found" in message
    assert "Covered," not in message and " Covered" not in message
    assert remote_scan_caveats([covered]) == ""


def test_remote_scan_caveats_flags_station_datasets():
    from openbench.data.registry.scanner import ScannedDataset
    from openbench.gui.pages._scan_worker import remote_scan_caveats

    grid = ScannedDataset(name="G", resolution="LowRes", category="Water", data_type="grid", root_dir="/r")
    stn = ScannedDataset(name="S", resolution="Station", category="Water", data_type="stn", root_dir="/r")

    message = remote_scan_caveats([grid, stn])

    assert "fulllist" in message
    assert "S" in message
    assert remote_scan_caveats([grid]) == ""


def test_remote_scan_caveats_reports_metadata_inspection_degradation():
    from openbench.data.registry.scanner import ScannedDataset
    from openbench.gui.pages._scan_worker import remote_scan_caveats

    grid = ScannedDataset(
        name="G",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir="/r",
        remote_inspection_error="missing scanner metadata API",
    )

    message = remote_scan_caveats([grid])

    assert "G_LowRes" in message
    assert "missing scanner metadata API" in message
    assert "data_groupby" in message


class RemoteController(FakeControllerBase):
    def __init__(self):
        self.config = {
            "general": {
                "remote": {
                    "python_path": "/remote/python",
                    "conda_env": "ob",
                    "openbench_path": "/remote/openbench",
                }
            }
        }
        self.storage = RemoteStorage("/remote/openbench", sync_engine=object())
        self.ssh_manager = SimpleNamespace(is_connected=True)

    def is_remote_mode(self):
        return True


def test_registry_netcdf_import_passes_remote_context(monkeypatch):
    from openbench.gui.pages.page_registry import PageRegistry

    captured = {}

    class FakeDialog:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def exec(self):
            return False

    monkeypatch.setattr("openbench.gui.widgets.nc_importer.NCImporterDialog", FakeDialog)

    page = PageRegistry.__new__(PageRegistry)
    page.controller = RemoteController()

    PageRegistry._import_model_from_nc(page)

    assert captured["ssh_manager"] is page.controller.ssh_manager
    assert captured["python_path"] == "/remote/python"
    assert captured["conda_env"] == "ob"
    assert captured["parent"] is page


class FakeSignal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)


class FakeProgress:
    def __init__(self, *args, **kwargs):
        self.closed = False

    def setWindowTitle(self, value):
        pass

    def setWindowModality(self, value):
        pass

    def setMinimumDuration(self, value):
        pass

    def setCancelButton(self, value):
        pass

    def show(self):
        pass

    def close(self):
        self.closed = True

    def deleteLater(self):
        pass


def test_registration_cancel_button_is_remote_only(monkeypatch):
    from openbench.gui.pages import page_ref_data, page_registry
    from openbench.gui.pages.page_ref_data import PageRefData
    from openbench.gui.pages.page_registry import PageRegistry

    cancel_labels = []

    class Progress(FakeProgress):
        def __init__(self, _message, cancel_label, *_args):
            super().__init__()
            self.canceled = FakeSignal()
            cancel_labels.append(cancel_label)

    class Worker:
        def __init__(self, *_args, **_kwargs):
            self.finished_with_result = FakeSignal()
            self.failed = FakeSignal()
            self.finished = FakeSignal()

        def start(self):
            pass

        def deleteLater(self):
            pass

    monkeypatch.setattr(page_ref_data, "QProgressDialog", Progress)
    monkeypatch.setattr(page_registry, "QProgressDialog", Progress)
    monkeypatch.setattr("openbench.gui.pages._scan_worker.RegisterScannedDatasetsWorker", Worker)

    ref_page = PageRefData.__new__(PageRefData)
    ref_page.btn_scan = FakeButton()
    registry_page = PageRegistry.__new__(PageRegistry)

    for page in (ref_page, registry_page):
        page._scan_remote_context = None
        page._start_register_worker(["local"])
        page._scan_remote_context = ("/remote/ref", {"ssh_manager": object()})
        page._start_register_worker(["remote"])

    assert cancel_labels == [None, "Cancel", None, "Cancel"]


def test_ref_scan_starts_remote_worker(monkeypatch):
    from openbench.gui.pages import page_ref_data
    from openbench.gui.pages.page_ref_data import PageRefData
    from tests.gui_fakes import FakeLineEdit

    captured = {}

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            self.finished_with_result = FakeSignal()
            self.failed = FakeSignal()
            self.finished = FakeSignal()
            self.started = False

        def start(self):
            self.started = True
            captured["started"] = True

        def deleteLater(self):
            pass

    monkeypatch.setattr(page_ref_data, "QProgressDialog", FakeProgress)
    monkeypatch.setattr("openbench.gui.path_utils._remote_directory_exists", lambda ssh, path: True)
    monkeypatch.setattr("openbench.gui.pages._scan_worker.FindDatasetsWorker", FakeWorker)

    page = PageRefData.__new__(PageRefData)
    page.controller = RemoteController()
    page.data_root_input = FakeLineEdit("/remote/ref")
    page.btn_scan = FakeButton()

    PageRefData._scan_data_root(page)

    assert captured["args"] == ("/remote/ref",)
    assert captured["kwargs"] == {
        "ssh_manager": page.controller.ssh_manager,
        "python_path": "/remote/python",
        "conda_env": "ob",
        "openbench_path": "/remote/openbench",
    }
    assert captured["started"] is True
    assert page.btn_scan.enabled is False


def test_ref_scan_disables_button_before_remote_existence_check(monkeypatch):
    from openbench.gui.pages.page_ref_data import PageRefData
    from tests.gui_fakes import FakeLineEdit

    page = PageRefData.__new__(PageRefData)
    page.controller = RemoteController()
    page.data_root_input = FakeLineEdit("/remote/ref")
    page.btn_scan = FakeButton()

    observed = {}

    def fake_exists(ssh, path):
        # The existence check spins a nested event loop; the button must
        # already be disabled or a second click re-enters the handler.
        observed["button_enabled_during_check"] = page.btn_scan.enabled
        return False  # then bail out without creating a worker

    monkeypatch.setattr("openbench.gui.path_utils._remote_directory_exists", fake_exists)
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.warning", lambda *args: None)

    PageRefData._scan_data_root(page)

    assert observed["button_enabled_during_check"] is False
    # Early-return path must hand the button back.
    assert page.btn_scan.enabled is True


def test_local_ref_scan_remembers_root_before_worker_starts(tmp_path, monkeypatch):
    from openbench.gui.pages import page_ref_data
    from openbench.gui.pages.page_ref_data import PageRefData
    from tests.gui_fakes import FakeLineEdit

    remembered = []
    captured = {}

    class FakeProgress:
        def __init__(self, *args, **kwargs):
            pass

        def setWindowTitle(self, *_args):
            pass

        def setWindowModality(self, *_args):
            pass

        def setMinimumDuration(self, *_args):
            pass

        def setCancelButton(self, *_args):
            pass

        def show(self):
            pass

    class FakeSignal:
        def connect(self, *_args):
            pass

    class FakeWorker:
        def __init__(self, root, **_kwargs):
            captured["root"] = root
            self.finished_with_result = FakeSignal()
            self.failed = FakeSignal()
            self.finished = FakeSignal()

        def start(self):
            captured["started"] = True

        def deleteLater(self):
            pass

    ref_root = tmp_path / "Reference"
    ref_root.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    page = PageRefData.__new__(PageRefData)
    page.controller = SimpleNamespace(storage=object())
    page.data_root_input = FakeLineEdit("~/Reference")
    page.btn_scan = FakeButton()
    page.save_to_config = lambda: captured.setdefault("saved", True)
    monkeypatch.setenv("OPENBENCH_REF_ROOT", "old-root")
    monkeypatch.setattr("openbench.gui.path_utils.remote_exec_context", lambda *_args: {})
    monkeypatch.setattr("openbench.config.user_settings.remember_reference_root", remembered.append)
    monkeypatch.setattr(page_ref_data, "QProgressDialog", FakeProgress)
    monkeypatch.setattr("openbench.gui.pages._scan_worker.FindDatasetsWorker", FakeWorker)

    PageRefData._scan_data_root(page)

    root = str(ref_root.resolve())
    assert remembered == [root]
    assert captured == {"saved": True, "root": root, "started": True}


def test_registry_scan_dialog_receives_registered_dataset_names(monkeypatch):
    from openbench.data.registry.scanner import DatasetGroup, ScannedDataset
    from openbench.gui.pages import page_registry
    from openbench.gui.pages.page_registry import PageRegistry

    variant = ScannedDataset("Existing", "LowRes", "Water", "grid", "/ref", {"Runoff": "runoff"})
    captured = {}

    class FakeDialog:
        def __init__(self, _groups, parent=None, *, existing_names=None):
            captured["existing_names"] = existing_names

        def exec(self):
            return False

    registry = SimpleNamespace(list_references=lambda: [SimpleNamespace(name=variant.registry_name)])
    monkeypatch.setattr(page_registry, "_get_registry", lambda: registry)
    monkeypatch.setattr(PageRegistry, "_finish_scan_worker", lambda _self: None)
    monkeypatch.setattr("openbench.gui.dialogs.data_discovery.DataDiscoveryDialog", FakeDialog)

    PageRegistry._on_scan_directory_finished(
        PageRegistry.__new__(PageRegistry),
        ([DatasetGroup("Existing", {"LowRes": variant})], []),
    )

    assert captured["existing_names"] == {variant.registry_name}


def test_ref_scan_filters_registry_list_to_datasets_found_on_disk(monkeypatch):
    from openbench.data.registry.scanner import DatasetGroup, ScannedDataset
    from openbench.gui.pages.page_ref_data import PageRefData

    found = ScannedDataset("Found", "LowRes", "Water", "grid", "/ref", {"Runoff": "runoff"})
    missing = SimpleNamespace(name="OpenBench_Missing")
    registry = SimpleNamespace(list_references=lambda: [SimpleNamespace(name=found.registry_name), missing])
    labels = []
    reloads = []

    class FakeDialog:
        def __init__(self, _groups, parent=None, *, existing_names=None):
            pass

        def exec(self):
            return False

    monkeypatch.setattr(PageRefData, "_finish_scan_worker", lambda _self: None)
    monkeypatch.setattr("openbench.data.registry.manager.get_registry", lambda: registry)
    monkeypatch.setattr("openbench.gui.dialogs.data_discovery.DataDiscoveryDialog", FakeDialog)

    page = PageRefData.__new__(PageRefData)
    page.registry_label = SimpleNamespace(setText=labels.append)
    page.load_from_config = lambda: reloads.append(True)

    PageRefData._on_scan_data_root_finished(page, ([DatasetGroup("Found", {"LowRes": found})], []))

    assert page._available_registry_names == {found.registry_name}
    assert labels == ["Registry: 1 datasets available"]
    assert reloads == [True]


def test_ref_scan_reports_connection_loss_distinctly(monkeypatch):
    """A dropped SSH session must not be reported as 'directory not found'."""
    from openbench.gui.pages.page_ref_data import PageRefData
    from openbench.remote.ssh import SSHConnectionError
    from tests.gui_fakes import FakeLineEdit

    def dead_exists(ssh, path):
        raise SSHConnectionError("session dropped")

    warnings = []
    monkeypatch.setattr("openbench.gui.path_utils._remote_directory_exists", dead_exists)
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.warning",
        lambda parent, title, message: warnings.append((title, message)),
    )

    page = PageRefData.__new__(PageRefData)
    page.controller = RemoteController()
    page.data_root_input = FakeLineEdit("/remote/ref")
    page.btn_scan = FakeButton()

    PageRefData._scan_data_root(page)

    assert warnings
    title, message = warnings[0]
    assert "onnect" in title + message  # mentions the connection
    assert "not found" not in message
    assert page.btn_scan.enabled is True  # button handed back


def test_ref_scan_ignores_reentrant_invocation(monkeypatch):
    from openbench.gui.pages import page_ref_data
    from openbench.gui.pages.page_ref_data import PageRefData
    from tests.gui_fakes import FakeLineEdit

    created = []
    monkeypatch.setattr(
        "openbench.gui.pages._scan_worker.FindDatasetsWorker",
        lambda *args, **kwargs: created.append(1),
    )
    monkeypatch.setattr(page_ref_data, "QProgressDialog", FakeProgress)
    monkeypatch.setattr("openbench.gui.path_utils._remote_directory_exists", lambda ssh, path: True)

    page = PageRefData.__new__(PageRefData)
    page.controller = RemoteController()
    page.data_root_input = FakeLineEdit("/remote/ref")
    page.btn_scan = FakeButton()
    page._scan_worker = object()  # a scan is already in flight

    PageRefData._scan_data_root(page)

    assert created == []


def test_registry_scan_ignores_reentrant_invocation(monkeypatch):
    from openbench.gui.pages import page_registry
    from openbench.gui.pages.page_registry import PageRegistry

    created = []
    monkeypatch.setattr(
        "openbench.gui.pages._scan_worker.FindDatasetsWorker",
        lambda *args, **kwargs: created.append(1),
    )
    monkeypatch.setattr(page_registry, "QProgressDialog", FakeProgress)
    monkeypatch.setattr(page_registry, "browse_directory", lambda *args, **kwargs: "/remote/ref")

    page = PageRegistry.__new__(PageRegistry)
    page.controller = RemoteController()
    page._scan_worker = object()  # a scan is already in flight

    PageRegistry._scan_directory(page)

    assert created == []


def test_nc_importer_open_file_is_guarded_against_reentry(qapp):
    from openbench.gui.widgets.nc_importer import NCImporterDialog

    ssh = FakeSSH("{}")
    dlg = NCImporterDialog(ssh_manager=ssh)
    dlg.edit_path.setText("/remote/data/sample.nc")
    dlg._busy = True

    dlg._open_file()

    assert ssh.commands == []


def test_nc_importer_disables_dialog_during_remote_open(qapp):
    from openbench.gui.widgets.nc_importer import NCImporterDialog

    payload = json.dumps({"path": "/remote/data/sample.nc", "data_var_count": 0, "variables": []})
    states = []

    class RecordingSSH(FakeSSH):
        dlg = None

        def execute(self, command, timeout=None):
            states.append(self.dlg.isEnabled())
            return super().execute(command, timeout=timeout)

    ssh = RecordingSSH(payload)
    dlg = NCImporterDialog(ssh_manager=ssh)
    ssh.dlg = dlg
    dlg.edit_path.setText("/remote/data/sample.nc")

    dlg._open_file()

    assert states and all(state is False for state in states)
    assert dlg.isEnabled() is True


def test_registry_scan_starts_remote_worker(monkeypatch):
    from openbench.gui.pages import page_registry
    from openbench.gui.pages.page_registry import PageRegistry

    captured = {}

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            self.finished_with_result = FakeSignal()
            self.failed = FakeSignal()
            self.finished = FakeSignal()

        def start(self):
            captured["started"] = True

        def deleteLater(self):
            pass

    monkeypatch.setattr(page_registry, "QProgressDialog", FakeProgress)
    monkeypatch.setattr(page_registry, "browse_directory", lambda *args, **kwargs: "/remote/ref")
    monkeypatch.setattr("openbench.gui.pages._scan_worker.FindDatasetsWorker", FakeWorker)

    page = PageRegistry.__new__(PageRegistry)
    page.controller = RemoteController()

    PageRegistry._scan_directory(page)

    assert captured["args"] == ("/remote/ref",)
    assert captured["kwargs"] == {
        "ssh_manager": page.controller.ssh_manager,
        "python_path": "/remote/python",
        "conda_env": "ob",
        "openbench_path": "/remote/openbench",
    }
    assert captured["started"] is True


def test_model_editor_inplace_save_expands_tilde_remote_path(qapp, monkeypatch):
    """Editing '~/OpenBench/.../Model.yaml' remotely and clicking Save must
    write through "$HOME", not a shlex-quoted literal tilde."""
    from openbench.gui.widgets import model_definition_editor as mde

    commands = []

    class SaveSSH:
        is_connected = True

        def execute(self, command, timeout=None):
            commands.append(command)
            return "", "", 0

    monkeypatch.setattr(mde.QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(mde.QMessageBox, "warning", staticmethod(lambda *a, **k: None))

    dlg = mde.ModelDefinitionEditor(file_path="~/OpenBench/nml/Mod/CoLM.yaml", ssh_manager=SaveSSH())
    dlg.model_name.setText("CoLM")

    dlg._save_inplace()

    assert commands
    assert '> "$HOME"/OpenBench/nml/Mod/CoLM.yaml' in commands[0]
    assert "'~/" not in commands[0]


def test_ref_scan_registers_selected_datasets_in_worker(monkeypatch):
    from openbench.data.registry.scanner import DatasetGroup, ScannedDataset
    from openbench.gui.pages import page_ref_data
    from openbench.gui.pages.page_ref_data import PageRefData
    from tests.gui_fakes import FakeButton

    variant = ScannedDataset("Found", "LowRes", "Water", "grid", "/ref", {"Runoff": "runoff"})
    variant.nc_inspections = {
        "Runoff": {
            "all_data_vars": [
                {"name": "wrong", "unit": "1"},
                {"name": "runoff", "unit": "mm"},
            ]
        }
    }
    group = DatasetGroup("Found", {"LowRes": variant})
    captured = {}

    class FakeDialog:
        def __init__(self, _groups, parent=None, *, existing_names=None):
            captured["existing_names"] = existing_names

        def exec(self):
            return True

        def get_selected(self):
            return [("Found", "LowRes", variant)]

    class FakeRegisterWorker:
        def __init__(self, variants):
            captured["variants"] = variants
            self.finished_with_result = FakeSignal()
            self.failed = FakeSignal()
            self.finished = FakeSignal()

        def start(self):
            captured["started"] = True

        def deleteLater(self):
            pass

    registry = SimpleNamespace(list_references=lambda: [], get_reference=lambda _name: None)
    monkeypatch.setattr(PageRefData, "_finish_scan_worker", lambda _self: None)
    monkeypatch.setattr("openbench.data.registry.manager.get_registry", lambda: registry)
    monkeypatch.setattr("openbench.gui.dialogs.data_discovery.DataDiscoveryDialog", FakeDialog)
    monkeypatch.setattr("openbench.gui.dialogs.data_discovery.choose_nc_variable", lambda *args: "runoff")
    monkeypatch.setattr("openbench.gui.pages._scan_worker.RegisterScannedDatasetsWorker", FakeRegisterWorker)
    monkeypatch.setattr(page_ref_data, "QProgressDialog", FakeProgress)

    page = PageRefData.__new__(PageRefData)
    page.btn_scan = FakeButton()

    PageRefData._on_scan_data_root_finished(page, ([group], []))

    assert captured["existing_names"] == set()
    assert captured["variants"] == [variant]
    assert captured["started"] is True
    assert page.btn_scan.enabled is False
    assert variant.nc_inspections["Runoff"]["varname"] == "runoff"
    assert variant.nc_inspections["Runoff"]["varunit"] == "mm"


def test_register_remote_passes_abort_callback_to_remote_python(monkeypatch):
    from openbench.data.registry.scanner import ScannedDataset
    from openbench.gui.pages import _scan_worker

    def aborting():
        return True

    captured = _capture_remote_json(monkeypatch, result={})
    dataset = ScannedDataset("Demo", "LowRes", "Water", "grid", "/remote/ref", {"Runoff": "Demo"})

    _scan_worker.register_scanned_datasets_remote(object(), [dataset], "/remote/ref", should_abort=aborting)

    assert captured["should_abort"] is aborting


def test_register_worker_remote_pre_cancel_skips_mutation(qapp, monkeypatch):
    from openbench.gui.pages import _scan_worker

    calls = []
    monkeypatch.setattr(_scan_worker, "register_scanned_datasets_remote", lambda *args, **kwargs: calls.append(args))

    worker = _scan_worker.RegisterScannedDatasetsWorker(["demo"], ssh_manager="ssh", data_root="/remote/ref")
    failures = []
    worker.failed.connect(failures.append)
    worker.requestInterruption()

    worker.run()

    assert calls == []
    assert failures == ["InterruptedError: Cancelled"]


def test_register_worker_remote_inflight_cancel_reaches_ssh_runner(qapp, monkeypatch):
    from openbench.gui.pages import _scan_worker

    captured = {}

    def fake_register_remote(*_args, **kwargs):
        captured["should_abort"] = kwargs["should_abort"]
        worker.requestInterruption()
        assert captured["should_abort"]() is True
        raise InterruptedError("Cancelled")

    monkeypatch.setattr(_scan_worker, "register_scanned_datasets_remote", fake_register_remote)

    worker = _scan_worker.RegisterScannedDatasetsWorker(["demo"], ssh_manager="ssh", data_root="/remote/ref")
    failures = []
    worker.failed.connect(failures.append)

    worker.run()

    assert failures == ["InterruptedError: Cancelled"]


def test_register_worker_writes_only_remote_registry(qapp, monkeypatch):
    from openbench.data.registry import scanner as scanner_module
    from openbench.gui.pages import _scan_worker

    calls = []

    def fake_remote(**kwargs):
        calls.append(("remote", kwargs))

    def fake_local(datasets):
        calls.append(("local", list(datasets)))
        return "/tmp/reference_catalog.yaml"

    def fake_register_remote(*_args, **kwargs):
        fake_remote(**kwargs)
        return {"catalog_path": "/remote/.openbench/references/reference_catalog.yaml"}

    monkeypatch.setattr(_scan_worker, "register_scanned_datasets_remote", fake_register_remote)
    monkeypatch.setattr(scanner_module, "register_scanned_datasets_batch", fake_local)

    worker = _scan_worker.RegisterScannedDatasetsWorker(
        ["demo"],
        ssh_manager="ssh",
        data_root="/remote/ref",
        python_path="/py",
        conda_env="ob",
        openbench_path="/ob",
    )
    results = []
    worker.finished_with_result.connect(results.append)

    worker.run()

    assert len(calls) == 1
    kind, kwargs = calls[0]
    assert kind == "remote"
    assert kwargs["python_path"] == "/py"
    assert kwargs["conda_env"] == "ob"
    assert kwargs["openbench_path"] == "/ob"
    assert callable(kwargs["should_abort"])
    assert results == [{"catalog_path": "/remote/.openbench/references/reference_catalog.yaml"}]


def test_register_scanned_datasets_worker_runs_off_gui_thread(qapp, monkeypatch):
    from PySide6.QtCore import QThread

    from openbench.data.registry import scanner as scanner_module
    from openbench.gui.pages._scan_worker import RegisterScannedDatasetsWorker

    ran_on_gui_thread = []

    def fake_register(datasets):
        assert datasets == ["demo"]
        ran_on_gui_thread.append(QThread.currentThread() == qapp.thread())
        return "/tmp/reference_catalog.yaml"

    monkeypatch.setattr(scanner_module, "register_scanned_datasets_batch", fake_register)

    worker = RegisterScannedDatasetsWorker(["demo"])
    worker.start()
    assert worker.wait(3000)
    worker.deleteLater()

    assert ran_on_gui_thread == [False]


def test_registry_scan_registers_selected_datasets_in_worker(monkeypatch):
    from openbench.data.registry.scanner import DatasetGroup, ScannedDataset
    from openbench.gui.pages import page_registry
    from openbench.gui.pages.page_registry import PageRegistry

    variant = ScannedDataset("Found", "LowRes", "Water", "grid", "/ref", {"Runoff": "runoff"})
    variant.nc_inspections = {
        "Runoff": {
            "all_data_vars": [
                {"name": "wrong", "unit": "1"},
                {"name": "runoff", "unit": "mm"},
            ]
        }
    }
    group = DatasetGroup("Found", {"LowRes": variant})
    captured = {}

    class FakeDialog:
        def __init__(self, _groups, parent=None, *, existing_names=None):
            captured["existing_names"] = existing_names

        def exec(self):
            return True

        def get_selected(self):
            return [("Found", "LowRes", variant)]

    class FakeRegisterWorker:
        def __init__(self, variants):
            captured["variants"] = variants
            self.finished_with_result = FakeSignal()
            self.failed = FakeSignal()
            self.finished = FakeSignal()

        def start(self):
            captured["started"] = True

        def deleteLater(self):
            pass

    registry = SimpleNamespace(list_references=lambda: [], get_reference=lambda _name: None)
    monkeypatch.setattr(PageRegistry, "_finish_scan_worker", lambda _self: None)
    monkeypatch.setattr(page_registry, "_get_registry", lambda: registry)
    monkeypatch.setattr("openbench.gui.dialogs.data_discovery.DataDiscoveryDialog", FakeDialog)
    monkeypatch.setattr("openbench.gui.dialogs.data_discovery.choose_nc_variable", lambda *args: "runoff")
    monkeypatch.setattr("openbench.gui.pages._scan_worker.RegisterScannedDatasetsWorker", FakeRegisterWorker)
    monkeypatch.setattr(page_registry, "QProgressDialog", FakeProgress)

    page = PageRegistry.__new__(PageRegistry)

    PageRegistry._on_scan_directory_finished(page, ([group], []))

    assert captured["existing_names"] == set()
    assert captured["variants"] == [variant]
    assert captured["started"] is True
    assert variant.nc_inspections["Runoff"]["varname"] == "runoff"
    assert variant.nc_inspections["Runoff"]["varunit"] == "mm"
