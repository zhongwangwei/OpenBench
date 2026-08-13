import os

import pytest
import yaml

from openbench.gui.config_manager import ConfigManager
from openbench.gui.pages.page_preview import (
    PagePreview,
    RemoteNamelistSyncError,
    _materialize_local_station_sources,
)
from tests.gui_fakes import FakeControllerBase


class FakeSSH:
    def __init__(self, responses=None, fail=False):
        self.commands = []
        self.responses = responses or {}
        self.fail = fail

    def execute(self, command, timeout=30):
        self.commands.append(command)
        if self.fail:
            raise RuntimeError("ssh down")
        for needle, response in self.responses.items():
            if needle in command:
                return response
        return "", "missing", 1


class FakeSFTP:
    def __init__(self, *, fail_put_for=None):
        self.mkdir_calls = []
        self.put_calls = []
        self.fail_put_for = fail_put_for or set()

    def mkdir(self, path):
        self.mkdir_calls.append(path)

    def put(self, local, remote):
        self.put_calls.append((local, remote))
        if remote in self.fail_put_for:
            raise OSError(f"put failed for {remote}")


def _preview():
    return PagePreview.__new__(PagePreview)


class FakeSignal:
    def __init__(self):
        self.emitted = []

    def emit(self, *args):
        self.emitted.append(args)


class FakeLabel:
    def __init__(self):
        self.text = ""

    def setText(self, value):
        self.text = value


class FakeYamlPreview:
    def __init__(self):
        self.content = ""

    def set_content(self, value):
        self.content = value


class RecordingConfigManager:
    def __init__(self):
        self.calls = []

    def generate_config_yaml(self, config, **kwargs):
        self.calls.append((config, kwargs))
        return "project: {}\n"


class FakeController:
    config = {"general": {"basename": "demo"}}
    storage = None

    def get_output_dir(self):
        return "/case/output/demo"


def test_preview_uses_actual_case_output_dir_when_rendering_unified_yaml():
    preview = _preview()
    preview.controller = FakeController()
    preview.config_manager = RecordingConfigManager()
    preview.output_dir_label = FakeLabel()
    preview.config_preview = FakeYamlPreview()

    preview.load_from_config()

    assert preview.config_manager.calls == [(preview.controller.config, {"case_output_dir": "/case/output/demo"})]
    assert preview.output_dir_label.text == "/case/output/demo"
    assert preview.config_preview.content == "project: {}\n"


def test_local_station_sources_are_materialized_before_export(monkeypatch, tmp_path):
    root = tmp_path / "stations"
    root.mkdir()
    config = {
        "evaluation_items": {"Latent_Heat": True},
        "sim_data": {
            "general": {"Latent_Heat_sim_source": ["StationCase"]},
            "source_configs": {
                "StationCase": {
                    "general": {
                        "model_namelist": "CoLM2024",
                        "root_dir": str(root),
                        "data_type": "stn",
                        "tim_res": "Day",
                    }
                }
            },
        },
    }

    def fake_materialize(result, output_dir, **_kwargs):
        case = result.cases[0]
        fulllist = tmp_path / "output" / "station_data" / "StationCase" / "StationCase_stations.csv"
        fulllist.parent.mkdir(parents=True, exist_ok=True)
        fulllist.write_text("ID,sim_dir\nA,a.nc\n")
        case.fulllist = fulllist
        case.station_count = 1

    monkeypatch.setattr("openbench.data.sim_scanner.materialize_station_cases", fake_materialize)

    _materialize_local_station_sources(config, str(tmp_path / "output"))

    general = config["sim_data"]["source_configs"]["StationCase"]["general"]
    assert general["fulllist"].endswith("StationCase_stations.csv")
    exported = yaml.safe_load(ConfigManager().generate_config_yaml(config, case_output_dir=str(tmp_path / "output")))
    assert exported["simulation"]["StationCase"]["data_type"] == "stn"
    assert exported["simulation"]["StationCase"]["tim_res"] == "Day"
    assert exported["simulation"]["StationCase"]["fulllist"].endswith("StationCase_stations.csv")


def test_local_grid_sources_do_not_trigger_station_materialization(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(
        "openbench.data.sim_scanner.materialize_station_cases",
        lambda *_args, **_kwargs: called.append(True),
    )
    config = {
        "evaluation_items": {"Latent_Heat": True},
        "sim_data": {
            "general": {"Latent_Heat_sim_source": ["GridCase"]},
            "source_configs": {
                "GridCase": {"general": {"root_dir": "/grid", "data_type": "grid"}},
            },
        },
    }

    _materialize_local_station_sources(config, str(tmp_path / "output"))

    assert called == []


def test_remote_preview_uses_same_remote_path_transform_as_export():
    from openbench.remote.storage import RemoteStorage

    class FakeRemoteController(FakeControllerBase):
        config = {
            "general": {
                "basename": "demo",
                "remote": {"openbench_path": "/remote/openbench"},
            }
        }
        storage = RemoteStorage("/remote/project", sync_engine=object())

        def get_output_dir(self):
            return "/remote/output/demo"

    preview = _preview()
    preview.controller = FakeRemoteController()
    preview.config_manager = RecordingConfigManager()
    preview.output_dir_label = FakeLabel()
    preview.config_preview = FakeYamlPreview()

    preview.load_from_config()

    _, kwargs = preview.config_manager.calls[-1]
    assert kwargs["case_output_dir"] == "/remote/output/demo"
    assert kwargs["path_transform"]("Reference") == "/remote/openbench/Reference"


def test_resolve_remote_model_path_raises_on_ssh_failure():
    preview = _preview()

    with pytest.raises(RemoteNamelistSyncError, match="Failed to check remote model path"):
        preview._resolve_model_path(
            "/remote/models/CoLM.yaml", "/remote/openbench", is_remote=True, ssh_manager=FakeSSH(fail=True)
        )


def test_resolve_remote_model_path_returns_empty_when_no_candidate_exists():
    preview = _preview()

    assert (
        preview._resolve_model_path(
            "/remote/models/Missing.nml", "/remote/openbench", is_remote=True, ssh_manager=FakeSSH()
        )
        == ""
    )


def test_resolve_path_for_remote_rejects_ambiguous_existing_local_absolute_path(monkeypatch):
    preview = _preview()
    monkeypatch.setattr(
        "openbench.gui.pages.page_preview.os.path.exists",
        lambda path: path == "/data/local/reference",
    )

    with pytest.raises(RemoteNamelistSyncError, match="Ambiguous local absolute path"):
        preview._resolve_path_for_remote("/data/local/reference", "/remote/openbench")


def test_resolve_path_for_remote_keeps_paths_under_remote_root(monkeypatch):
    preview = _preview()
    monkeypatch.setattr("openbench.gui.pages.page_preview.os.path.exists", lambda path: True)

    assert (
        preview._resolve_path_for_remote("/remote/openbench/Reference", "/remote/openbench")
        == "/remote/openbench/Reference"
    )


def test_copy_remote_model_definition_reports_cat_failure(tmp_path):
    preview = _preview()
    dest = tmp_path / "models" / "CoLM.yaml"

    ok = preview._copy_model_definition_filtered(
        "/remote/models/CoLM.yaml",
        str(dest),
        ["Latent_Heat"],
        is_remote=True,
        ssh_manager=FakeSSH(responses={"cat": ("", "permission denied", 1)}),
    )

    assert ok is False
    assert not dest.exists()


def test_copy_remote_model_definition_writes_filtered_yaml(tmp_path):
    preview = _preview()
    dest = tmp_path / "models" / "CoLM.yaml"
    source = yaml.safe_dump(
        {"general": {"model": "CoLM"}, "Latent_Heat": {"varname": "lh"}, "Runoff": {"varname": "ro"}}
    )

    ok = preview._copy_model_definition_filtered(
        "/remote/models/CoLM.yaml",
        str(dest),
        ["Latent_Heat"],
        is_remote=True,
        ssh_manager=FakeSSH(responses={"cat": (source, "", 0)}),
    )

    assert ok is True
    assert yaml.safe_load(dest.read_text(encoding="utf-8")) == {
        "general": {"model": "CoLM"},
        "Latent_Heat": {"varname": "lh"},
    }


def test_upload_directory_uploads_root_files_and_creates_nested_empty_dirs(tmp_path):
    preview = _preview()
    local_dir = tmp_path / "nml"
    empty_dir = local_dir / "sim" / "models"
    empty_dir.mkdir(parents=True)
    root_file = local_dir / "main-demo.yaml"
    root_file.write_text("main: true\n", encoding="utf-8")

    sftp = FakeSFTP()
    preview._upload_directory(sftp, str(local_dir), "/remote/output/demo/nml")

    assert (str(root_file), "/remote/output/demo/nml/main-demo.yaml") in sftp.put_calls
    assert "/remote/output/demo/nml/sim" in sftp.mkdir_calls
    assert "/remote/output/demo/nml/sim/models" in sftp.mkdir_calls


def test_upload_directory_propagates_partial_file_upload_failure(tmp_path):
    preview = _preview()
    local_dir = tmp_path / "nml"
    local_dir.mkdir()
    file_path = local_dir / "main-demo.yaml"
    file_path.write_text("main: true\n", encoding="utf-8")
    sftp = FakeSFTP(fail_put_for={"/remote/output/demo/nml/main-demo.yaml"})

    with pytest.raises(OSError, match="put failed"):
        preview._upload_directory(sftp, str(local_dir), "/remote/output/demo/nml")


def test_remote_export_fails_when_unified_config_was_not_created(monkeypatch, tmp_path):
    from openbench.gui.pages import page_preview as preview_module

    class SSH:
        is_connected = True

        def execute(self, command, timeout=30):
            return "", "", 0

        def open_sftp(self):
            return FakeSFTP()

    class Controller(FakeControllerBase):
        config = {"general": {"basename": "demo"}}
        navigated = []

        def go_to_page(self, page):
            self.navigated.append(page)

    critical = []
    monkeypatch.setattr(preview_module, "get_remote_ssh_manager", lambda controller: SSH())
    monkeypatch.setattr(
        "openbench.gui.pages.page_preview.QMessageBox.critical",
        lambda parent, title, message: critical.append((title, message)),
    )

    preview = _preview()
    preview.controller = Controller()
    preview.run_requested = FakeSignal()
    preview._get_openbench_root = lambda: str(tmp_path)
    preview._export_for_remote = lambda local_dir, output_dir, openbench_root, remote_openbench_path: {}

    assert preview._export_and_run_remote("/remote/output/demo") is False
    assert preview.controller.navigated == []
    assert preview.run_requested.emitted == []
    assert critical and "openbench.yaml" in critical[-1][1]


def test_remote_export_sftp_put_failure_does_not_enter_run_page(monkeypatch, tmp_path):
    from openbench.gui.pages import page_preview as preview_module

    class SSH:
        is_connected = True

        def execute(self, command, timeout=30):
            return "", "", 0

        def open_sftp(self):
            return FakeSFTP(fail_put_for={"/remote/output/demo/openbench.yaml"})

    class Controller(FakeControllerBase):
        config = {"general": {"basename": "demo"}}
        navigated = []

        def go_to_page(self, page):
            self.navigated.append(page)

    def export_for_remote(local_dir, output_dir, openbench_root, remote_openbench_path):
        os.makedirs(os.path.join(local_dir, "nml"), exist_ok=True)
        config_path = tmp_path / "openbench.yaml"
        config_path.write_text("project: {}\n", encoding="utf-8")
        return {"config": str(config_path)}

    critical = []
    monkeypatch.setattr(preview_module, "get_remote_ssh_manager", lambda controller: SSH())
    monkeypatch.setattr(
        "openbench.gui.pages.page_preview.QMessageBox.critical",
        lambda parent, title, message: critical.append((title, message)),
    )

    preview = _preview()
    preview.controller = Controller()
    preview.run_requested = FakeSignal()
    preview._get_openbench_root = lambda: str(tmp_path)
    preview._export_for_remote = export_for_remote

    assert preview._export_and_run_remote("/remote/output/demo") is False
    assert preview.controller.navigated == []
    assert preview.run_requested.emitted == []
    assert critical and "put failed" in critical[-1][1]


def test_remote_export_nml_upload_failure_does_not_enter_run_page(monkeypatch, tmp_path):
    from openbench.gui.pages import page_preview as preview_module

    class SSH:
        is_connected = True

        def execute(self, command, timeout=30):
            return "", "", 0

        def open_sftp(self):
            return FakeSFTP(fail_put_for={"/remote/output/demo/nml/main-demo.yaml"})

    class Controller(FakeControllerBase):
        config = {"general": {"basename": "demo"}}
        navigated = []

        def go_to_page(self, page):
            self.navigated.append(page)

    def export_for_remote(local_dir, output_dir, openbench_root, remote_openbench_path):
        nml_dir = os.path.join(local_dir, "nml")
        os.makedirs(nml_dir, exist_ok=True)
        main_path = os.path.join(nml_dir, "main-demo.yaml")
        with open(main_path, "w", encoding="utf-8") as f:
            f.write("main: true\n")
        config_path = os.path.join(local_dir, "openbench.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("project: {}\n")
        return {"config": config_path}

    critical = []
    monkeypatch.setattr(preview_module, "get_remote_ssh_manager", lambda controller: SSH())
    monkeypatch.setattr(
        "openbench.gui.pages.page_preview.QMessageBox.critical",
        lambda parent, title, message: critical.append((title, message)),
    )

    preview = _preview()
    preview.controller = Controller()
    preview.run_requested = FakeSignal()
    preview._get_openbench_root = lambda: str(tmp_path)
    preview._export_for_remote = export_for_remote

    assert preview._export_and_run_remote("/remote/output/demo") is False
    assert preview.controller.navigated == []
    assert preview.run_requested.emitted == []
    assert critical and "put failed" in critical[-1][1]


def test_remote_export_marks_direct_sftp_uploads_synced_in_remote_storage_cache(monkeypatch, tmp_path):
    from openbench.gui.pages import page_preview as preview_module
    from openbench.remote.storage import RemoteStorage
    from openbench.remote.sync import SyncEngine, SyncStatus

    class SSH:
        is_connected = True

        def execute(self, command, timeout=30):
            return "", "", 0

        def open_sftp(self):
            return FakeSFTP()

    ssh = SSH()
    sync = SyncEngine(ssh, "/remote/project")
    sync.write("output/demo/openbench.yaml", "old config")
    sync.write("output/demo/nml/main-demo.yaml", "old main")

    class Controller(FakeControllerBase):
        config = {"general": {"basename": "demo"}}
        storage = RemoteStorage("/remote/project", sync)
        navigated = []

        def go_to_page(self, page):
            self.navigated.append(page)

    def export_for_remote(local_dir, output_dir, openbench_root, remote_openbench_path):
        nml_dir = os.path.join(local_dir, "nml")
        os.makedirs(nml_dir, exist_ok=True)
        main_path = os.path.join(nml_dir, "main-demo.yaml")
        with open(main_path, "w", encoding="utf-8") as f:
            f.write("new main")
        config_path = os.path.join(local_dir, "openbench.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("new config")
        return {"config": config_path}

    monkeypatch.setattr(preview_module, "get_remote_ssh_manager", lambda controller: ssh)

    preview = _preview()
    preview.controller = Controller()
    preview.run_requested = FakeSignal()
    preview._get_openbench_root = lambda: str(tmp_path)
    preview._export_for_remote = export_for_remote

    assert preview._export_and_run_remote("/remote/project/output/demo") is True
    assert sync.get_pending_count() == 0
    assert sync.get_sync_status("output/demo/openbench.yaml") is SyncStatus.SYNCED
    assert sync.get_sync_status("output/demo/nml/main-demo.yaml") is SyncStatus.SYNCED
    assert sync.read("output/demo/openbench.yaml") == "new config"
    assert sync.read("output/demo/nml/main-demo.yaml") == "new main"
