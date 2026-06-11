import json
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

PySide6 = pytest.importorskip("PySide6")

from openbench.remote.storage import RemoteStorage  # noqa: E402
from tests.gui_fakes import FakeControllerBase  # noqa: E402


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
    # named '~'; the conda base must expand to the remote $HOME instead.
    assert '. "$HOME"/miniconda3/etc/profile.d/conda.sh && conda activate ob && ' in cmd
    assert "'~/miniconda3'" not in cmd


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


def test_remote_scan_script_bootstraps_openbench_path_and_local_names(monkeypatch):
    from openbench.gui.pages import _scan_worker

    captured = _capture_remote_json(monkeypatch)
    monkeypatch.setattr(_scan_worker, "_local_reference_names", lambda: {"Already_LowRes"})

    _scan_worker.scan_reference_datasets_remote(object(), "/remote/ref", openbench_path="/remote/openbench")

    script = captured["script"]
    assert "sys.path.insert" in script
    assert '"/remote/openbench/src"' in script
    assert "Already_LowRes" in script
    assert "existing_names=" in script
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


def test_remote_scan_script_attaches_remote_inspections(monkeypatch):
    from openbench.gui.pages import _scan_worker

    captured = _capture_remote_json(monkeypatch)
    monkeypatch.setattr(_scan_worker, "_local_reference_names", lambda: set())

    _scan_worker.scan_reference_datasets_remote(object(), "/remote/ref")

    script = captured["script"]
    assert "_inspect_nc_file" in script
    assert "_detect_data_groupby" in script
    assert "nc_inspections" in script


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
    uncovered = ScannedDataset(name="Uncovered", resolution="Station", category="Water", data_type="stn", root_dir="/r")

    message = remote_scan_caveats([covered, uncovered])

    assert "Uncovered" in message
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


class FakeButton:
    def __init__(self):
        self.enabled = None

    def setEnabled(self, value):
        self.enabled = value


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
