import shlex
from pathlib import Path
from types import SimpleNamespace

from openbench.gui.pages import page_sim_data
from tests.gui_fakes import FakeLineEdit as _Text


def test_gui_sim_scan_helpers_find_nc4_history_dir(tmp_path: Path):
    case_dir = tmp_path / "CaseA"
    history = case_dir / "history"
    history.mkdir(parents=True)
    (history / "hist_2000.nc4").write_text("placeholder")

    assert page_sim_data._find_nc_dir(str(case_dir)) == str(history)
    assert page_sim_data._detect_prefix(str(case_dir)) == "hist_"


def test_remote_sim_scan_helpers_quote_paths_and_find_nc4():
    commands = []

    class FakeSSH:
        def execute(self, command, timeout=30):
            commands.append(command)
            if "test -d" in command:
                return "dir\n", "", 0
            if "history" in command:
                return "/remote/project/Case A/history/hist_2000.nc4\n", "", 0
            return "", "", 1

    ssh = FakeSSH()
    case_dir = "/remote/project/Case A"

    assert page_sim_data._remote_find_nc_dir(ssh, case_dir) == f"{case_dir}/history"
    assert page_sim_data._remote_detect_prefix(ssh, case_dir) == "hist_"
    assert any(shlex.quote(f"{case_dir}/history") in command for command in commands)


def test_remote_scan_includes_root_and_nested_directories():
    commands = []

    class FakeSSH:
        def execute(self, command, timeout=30):
            commands.append(command)
            return "/remote/Simulation\n/remote/Simulation/LSMs/CLM5\n", "", 0

    found = page_sim_data._remote_list_dirs(FakeSSH(), "/remote/Simulation")

    assert found == ["/remote/Simulation", "/remote/Simulation/LSMs/CLM5"]
    assert "-mindepth 0 -maxdepth 5" in commands[0]


def test_remote_sim_scan_rehydrates_scanner_metadata_and_fulllist(monkeypatch):
    captured = {}

    def fake_remote_json(ssh_manager, script, *, python_path="", conda_env="", timeout=60, should_abort=None):
        captured.update(
            {
                "script": script,
                "python_path": python_path,
                "conda_env": conda_env,
                "timeout": timeout,
                "should_abort": should_abort,
            }
        )
        return {
            "cases": [
                {
                    "label": "StationCase",
                    "root_dir": "/remote/sim/StationCase",
                    "source_root": "/remote/sim",
                    "model": "CoLM2024",
                    "prefix": "",
                    "suffix": "",
                    "files": [],
                    "variables": ["Latent_Heat"],
                    "variable_overrides": {},
                    "data_type": "stn",
                    "grid_res": None,
                    "tim_res": "Day",
                    "data_groupby": "Single",
                    "fulllist": "/home/alice/.openbench/sim_station_lists/abc/StationCase/StationCase_stations.csv",
                    "station_layout": "nested_multi",
                }
            ]
        }

    monkeypatch.setattr("openbench.gui.remote_python.run_remote_python_json", fake_remote_json)

    discovered, metadata = page_sim_data.scan_simulation_cases_remote(
        object(),
        "/remote/sim",
        python_path="/remote/conda/env/bin/python",
        conda_env="ob",
        openbench_path="~/OpenBench",
        should_abort=lambda: False,
    )

    assert discovered == [("StationCase", "/remote/sim/StationCase", "")]
    assert metadata["StationCase"]["model"] == "CoLM2024"
    assert metadata["StationCase"]["data_type"] == "stn"
    assert metadata["StationCase"]["tim_res"] == "Day"
    assert metadata["StationCase"]["data_groupby"] == "Single"
    assert metadata["StationCase"]["fulllist"].endswith("StationCase_stations.csv")
    assert "materialize_station_cases" in captured["script"]
    assert "num_workers=1" in captured["script"]
    assert "expanduser" in captured["script"]
    assert captured["python_path"] == "/remote/conda/env/bin/python"
    assert captured["conda_env"] == "ob"
    assert captured["timeout"] == 900


def test_remote_model_match_uses_case_label_and_leaves_unknown_blank():
    models = ["LEM2", "CLM5", "CoLM2024"]

    assert page_sim_data._model_from_case_label("CLM5", models) == "CLM5"
    assert page_sim_data._model_from_case_label("UnknownCase", models) == ""


def test_remote_rehydrate_does_not_use_local_registry_for_blank_model(monkeypatch):
    import openbench.data.sim_scanner as sim_scanner

    calls = []

    def fake_resolve(model_name, root, nc_dir, label, variables):
        calls.append((model_name, root, nc_dir, label, variables))
        return "VariableModel", "variables", None

    monkeypatch.setattr(sim_scanner, "_resolve_model", fake_resolve)

    discovered, metadata = page_sim_data._rehydrate_simulation_cases(
        {
            "cases": [
                {
                    "label": "CaseA",
                    "root_dir": "/remote/sim/CaseA",
                    "source_root": "/remote/sim",
                    "model": "UNRESOLVED",
                    "variables": ["Runoff", "Latent_Heat"],
                }
            ]
        }
    )

    assert discovered == [("CaseA", "/remote/sim/CaseA", "")]
    assert metadata["CaseA"]["model"] == ""
    assert calls == []


def test_remote_rehydrate_keeps_unresolved_model_blank(monkeypatch):
    import openbench.data.sim_scanner as sim_scanner

    monkeypatch.setattr(
        sim_scanner,
        "_resolve_model",
        lambda *_args: ("UNRESOLVED", "unresolved", "model"),
    )

    _discovered, metadata = page_sim_data._rehydrate_simulation_cases(
        {
            "cases": [
                {
                    "label": "CaseA",
                    "root_dir": "/remote/sim/CaseA",
                    "source_root": "/remote/sim",
                    "model": "",
                    "variables": ["unknown"],
                }
            ]
        }
    )

    assert metadata["CaseA"]["model"] == ""


class _Controller:
    def __init__(self):
        self.config = {
            "evaluation_items": {"Runoff": True},
            "sim_data": {
                "general": {"legacy": "keep", "Runoff_sim_source": ["CaseA"]},
                "def_nml": {"CaseA": "/old/def.yaml"},
                "source_configs": {
                    "CaseA": {
                        "general": {"fulllist": "/sim/list.csv", "legacy_general": "keep"},
                        "variables": {"Runoff": {"varname": "q"}},
                    }
                },
            },
        }
        self.updated = None

    def update_section(self, name, value):
        self.updated = (name, value)


def test_save_to_config_preserves_simulation_source_metadata():
    controller = _Controller()
    page = SimpleNamespace(
        controller=controller,
        get_selected_cases=lambda: [{"label": "CaseA", "model": "CoLM2024", "nc_dir": "/sim", "prefix": "hist_"}],
        _prefix_input=_Text(""),
        _data_type_combo=_Text("stn"),
        _grid_res_input=_Text("0.5"),
        _tim_res_combo=_Text("Day"),
        _data_groupby_combo=_Text("Single"),
        _suffix_input=_Text(".nc"),
        _root_input=_Text("/root"),
        _get_available_variables=lambda: set(),
    )

    page_sim_data.PageSimData.save_to_config(page)

    assert controller.updated[0] == "sim_data"
    saved = controller.updated[1]
    assert saved["general"]["legacy"] == "keep"
    assert saved["def_nml"] == {"CaseA": "/old/def.yaml"}
    assert saved["source_configs"]["CaseA"]["general"]["fulllist"] == "/sim/list.csv"
    assert saved["source_configs"]["CaseA"]["general"]["legacy_general"] == "keep"
    assert saved["source_configs"]["CaseA"]["variables"] == {"Runoff": {"varname": "q"}}


def test_remote_sim_helpers_expand_tilde_paths():
    """'~/Simulation' typed in the root field must reach the remote shell as
    "$HOME"/Simulation, not a shlex-quoted literal tilde."""
    commands = []

    class FakeSSH:
        def execute(self, command, timeout=30):
            commands.append(command)
            return "", "", 1

    page_sim_data._remote_is_dir(FakeSSH(), "~/Simulation")

    assert commands
    assert '"$HOME"/Simulation' in commands[0]
    assert "'~/" not in commands[0]


def test_local_gui_sim_scan_runs_off_gui_thread(qapp, monkeypatch, tmp_path):
    from PySide6.QtCore import QThread

    ran_on_gui_thread = []

    def fake_scan(root):
        assert root == str(tmp_path)
        ran_on_gui_thread.append(QThread.currentThread() == qapp.thread())
        return [], {}

    monkeypatch.setattr(page_sim_data, "_scan_local_cases", fake_scan)
    monkeypatch.setattr(page_sim_data.QMessageBox, "information", lambda *args: None)
    page = SimpleNamespace(
        controller=SimpleNamespace(storage=object()),
        _root_input=_Text(str(tmp_path)),
        _clear_cases=lambda: None,
    )

    page_sim_data.PageSimData._do_scan_flow(page)

    assert ran_on_gui_thread == [False]


def test_remote_sim_scan_helpers_find_uppercase_nc4():
    class FakeSSH:
        def execute(self, command, timeout=30):
            if "test -d" in command:
                return "dir\n", "", 0
            if "history" in command:
                return "/remote/Case/history/HIST_2000.NC4\n", "", 0
            return "", "", 0

    assert page_sim_data._remote_find_nc_dir(FakeSSH(), "/remote/Case") == "/remote/Case/history"
    assert page_sim_data._remote_detect_prefix(FakeSSH(), "/remote/Case") == "HIST_"


def test_remote_sim_scan_helpers_raise_find_errors():
    class FakeSSH:
        def execute(self, command, timeout=30):
            return "", "permission denied", 1

    try:
        page_sim_data._remote_list_nc_files(FakeSSH(), "/remote/private")
    except RuntimeError as exc:
        assert "permission denied" in str(exc)
    else:
        raise AssertionError("expected remote find failure to raise")


def test_validate_data_requires_local_netcdf_files(monkeypatch, tmp_path):
    warnings = []
    monkeypatch.setattr(page_sim_data.QMessageBox, "warning", lambda *args: warnings.append(args))
    page = SimpleNamespace(
        controller=SimpleNamespace(storage=object()),
        get_selected_cases=lambda: [{"label": "CaseA", "nc_dir": str(tmp_path), "model": "CoLM2024"}],
    )

    page_sim_data.PageSimData._validate_data(page)

    assert "no NetCDF files found" in warnings[0][2]


def test_validate_data_requires_configured_file_pattern(monkeypatch, tmp_path):
    (tmp_path / "other.nc").touch()
    warnings = []
    infos = []
    monkeypatch.setattr(page_sim_data.QMessageBox, "warning", lambda *args: warnings.append(args))
    monkeypatch.setattr(page_sim_data.QMessageBox, "information", lambda *args: infos.append(args))
    page = SimpleNamespace(
        controller=SimpleNamespace(storage=object(), config={"general": {"syear": 2001, "eyear": 2001}}),
        get_selected_cases=lambda: [
            {
                "label": "CaseA",
                "nc_dir": str(tmp_path),
                "model": "CoLM2024",
                "prefix": "missing_",
                "suffix": ".nc",
                "data_groupby": "Year",
            }
        ],
    )

    page_sim_data.PageSimData._validate_data(page)

    assert not infos
    assert warnings
    assert "No files found matching pattern" in warnings[0][2]
    assert "missing_*.nc" in warnings[0][2]


def test_validate_data_checks_each_variable_file_pattern(monkeypatch, tmp_path):
    (tmp_path / "runoff_2001.nc").touch()
    warnings = []
    monkeypatch.setattr(page_sim_data.QMessageBox, "warning", lambda *args: warnings.append(args))
    monkeypatch.setattr(page_sim_data.QMessageBox, "information", lambda *args: None)
    page = SimpleNamespace(
        controller=SimpleNamespace(storage=object(), config={"general": {"syear": 2001, "eyear": 2001}}),
        get_selected_cases=lambda: [
            {
                "label": "CaseA",
                "nc_dir": str(tmp_path),
                "model": "CoLM2024",
                "prefix": "",
                "suffix": "",
                "data_groupby": "Year",
                "variables": {
                    "Runoff": {"prefix": "runoff_", "suffix": ".nc"},
                    "Latent_Heat": {"prefix": "heat_", "suffix": ".nc"},
                },
            }
        ],
    )

    page_sim_data.PageSimData._validate_data(page)

    assert warnings
    assert "CaseA (Latent_Heat)" in warnings[0][2]
    assert "heat_*.nc" in warnings[0][2]


def test_remote_station_validation_checks_fulllist_not_case_root(monkeypatch):
    from openbench.gui.data_validator import ValidationCheck
    from openbench.remote.storage import RemoteStorage

    calls = []
    infos = []

    class SSH:
        is_connected = True

    class FakeRemoteValidator:
        def __init__(self, ssh_manager):
            self.ssh_manager = ssh_manager

        def check_file_exists(self, path):
            calls.append(path)
            return ValidationCheck("file_exists", True, f"File exists: {path}")

    monkeypatch.setattr(page_sim_data, "get_remote_ssh_manager", lambda _controller: SSH())
    monkeypatch.setattr(page_sim_data, "_remote_find_nc_dir", lambda *_args: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr("openbench.gui.data_validator.RemoteNetCDFValidator", FakeRemoteValidator)
    monkeypatch.setattr(page_sim_data.QMessageBox, "information", lambda *args: infos.append(args))
    monkeypatch.setattr(page_sim_data.QMessageBox, "warning", lambda *args: None)

    page = SimpleNamespace(
        controller=SimpleNamespace(
            storage=RemoteStorage("/remote/project", object()),
            config={"general": {"syear": 2001, "eyear": 2001}},
            remote_settings=lambda: {"openbench_path": "/remote/openbench"},
        ),
        get_selected_cases=lambda: [
            {
                "label": "StationCase",
                "nc_dir": "/remote/sim/StationCase",
                "model": "CoLM2024",
                "data_type": "stn",
                "fulllist": "/remote/home/.openbench/sim_station_lists/StationCase.csv",
            }
        ],
    )

    page_sim_data.PageSimData._validate_data(page)

    assert calls == ["/remote/home/.openbench/sim_station_lists/StationCase.csv"]
    assert infos and infos[0][1] == "Validation OK"


def test_remote_station_validation_rejects_partial_materialization(monkeypatch):
    from openbench.remote.storage import RemoteStorage

    warnings = []
    infos = []
    payload = {
        "cases": [
            {
                "label": "StationCase",
                "root_dir": "/remote/sim/StationCase",
                "model": "CoLM2024",
                "data_type": "stn",
                "fulllist": "/remote/home/.openbench/sim_station_lists/StationCase.csv",
                "station_layout": "nested_multi",
                "station_dropped_sites": ["bad_a", "bad_b"],
                "unresolved": ["station_partial"],
            }
        ]
    }
    _discovered, metadata = page_sim_data._rehydrate_simulation_cases(payload)

    class SSH:
        is_connected = True

    class FakeRemoteValidator:
        def __init__(self, ssh_manager):
            self.ssh_manager = ssh_manager

        def check_file_exists(self, path):
            raise AssertionError("partial station must fail before fulllist existence check")

    monkeypatch.setattr(page_sim_data, "get_remote_ssh_manager", lambda _controller: SSH())
    monkeypatch.setattr("openbench.gui.data_validator.RemoteNetCDFValidator", FakeRemoteValidator)
    monkeypatch.setattr(page_sim_data.QMessageBox, "information", lambda *args: infos.append(args))
    monkeypatch.setattr(page_sim_data.QMessageBox, "warning", lambda *args: warnings.append(args))

    page = SimpleNamespace(
        controller=SimpleNamespace(
            storage=RemoteStorage("/remote/project", object()),
            config={"general": {"syear": 2001, "eyear": 2001}},
            remote_settings=lambda: {},
        ),
        get_selected_cases=lambda: [
            {
                "label": "StationCase",
                "nc_dir": "/remote/sim/StationCase",
                "model": "CoLM2024",
                "data_type": "stn",
                "fulllist": metadata["StationCase"]["fulllist"],
                "station_materialize_error": metadata["StationCase"]["station_materialize_error"],
            }
        ],
    )

    page_sim_data.PageSimData._validate_data(page)

    assert not infos
    assert warnings
    assert "dropped 2 site(s): bad_a, bad_b" in warnings[0][2]


def test_save_to_config_only_assigns_variables_to_supporting_cases(monkeypatch):
    controller = _Controller()
    controller.config["evaluation_items"] = {"Runoff": True, "Latent_Heat": True}
    monkeypatch.setattr(
        page_sim_data,
        "_get_model_variables",
        lambda model: ["Runoff"] if model == "RunoffModel" else [],
    )
    page = SimpleNamespace(
        controller=controller,
        get_selected_cases=lambda: [
            {"label": "RunoffCase", "model": "RunoffModel", "nc_dir": "/sim/r", "prefix": "", "variables": {}},
            {
                "label": "ManualHeatCase",
                "model": "UnknownModel",
                "nc_dir": "/sim/h",
                "prefix": "",
                "variables": {"Latent_Heat": {"varname": "lh"}},
            },
        ],
        _prefix_input=_Text(""),
        _data_type_combo=_Text("grid"),
        _grid_res_input=_Text("0.5"),
        _tim_res_combo=_Text("Month"),
        _data_groupby_combo=_Text("Year"),
        _suffix_input=_Text(".nc"),
        _root_input=_Text("/sim"),
    )

    page_sim_data.PageSimData.save_to_config(page)

    general = controller.updated[1]["general"]
    assert general["Runoff_sim_source"] == ["RunoffCase"]
    assert general["Latent_Heat_sim_source"] == ["ManualHeatCase"]


def test_simulation_case_uses_readable_card_layout(qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QFormLayout, QGroupBox, QSizePolicy

    from openbench.gui.controller import WizardController
    from openbench.gui.pages.page_sim_data import PageSimData

    page = PageSimData(WizardController())
    page._model_names = ["CLM5"]
    path = r"G:\Cases_for_openbench\Simulation\LSMs\CLM5"
    page._add_case_row(
        "CLM5",
        path,
        "prefix_",
        model_name="CLM5",
        suffix=".nc",
        scan_metadata={"data_type": "grid", "tim_res": "Month", "grid_res": 1.8947},
    )

    scan_group = next(group for group in page.findChildren(QGroupBox) if group.title() == "Scan for Cases")
    case = page._cases[0]

    assert PageSimData.CONTENT_EXPAND is True
    assert scan_group.layout().fieldGrowthPolicy() == QFormLayout.AllNonFixedFieldsGrow
    assert page._case_scroll.minimumHeight() >= 220
    assert page._case_layout.alignment() & Qt.AlignTop
    assert case["path_input"].text() == path
    assert case["path_input"].isReadOnly()
    assert case["path_input"].sizePolicy().horizontalPolicy() == QSizePolicy.Expanding
    assert case["status_label"].text() == "Model: CLM5"
    assert case["model_combo"].isHidden()
    assert case["pattern_widget"].isHidden()


def test_simulation_case_model_picker_updates_hidden_state(qapp, monkeypatch):
    from openbench.gui.controller import WizardController
    from openbench.gui.pages.page_sim_data import PageSimData

    page = PageSimData(WizardController())
    page._model_names = ["CLM5", "CoLM2024"]
    page._add_case_row(
        "CaseA",
        "/sim/CaseA",
        "hist_",
        model_name="CLM5",
        scan_metadata={"data_type": "grid", "tim_res": "Month", "grid_res": 1.0},
    )
    case = page._cases[0]

    monkeypatch.setattr(
        page_sim_data.QInputDialog,
        "getItem",
        lambda *args: ("CoLM2024", True),
    )

    page._choose_case_model(case)

    assert case["model_combo"].currentData() == "CoLM2024"
    assert case["status_label"].text() == "Model: CoLM2024"
    assert page.get_selected_cases()[0]["model"] == "CoLM2024"


def test_remote_saved_model_survives_registry_disconnect(qapp):
    from types import SimpleNamespace

    from openbench.gui.controller import WizardController
    from openbench.remote.storage import RemoteStorage

    controller = WizardController()
    controller.storage = RemoteStorage("/remote/project", SimpleNamespace())
    controller.ssh_manager = None
    controller._config["evaluation_items"] = {"Runoff": True}
    controller._config["sim_data"] = {
        "general": {"Runoff_sim_source": ["CaseA"]},
        "_scanned_cases": [
            {
                "label": "CaseA",
                "nc_dir": "/remote/sim/CaseA",
                "prefix": "hist_",
                "checked": True,
                "model": "RemoteOnly",
            }
        ],
    }
    page = page_sim_data.PageSimData(controller)

    page.load_from_config()

    assert page._cases[0]["model_combo"].currentData() == "RemoteOnly"
    page.save_to_config()
    assert controller.config["sim_data"]["_scanned_cases"][0]["model"] == "RemoteOnly"
    assert controller.config["sim_data"]["source_configs"]["CaseA"]["general"]["model_namelist"] == "RemoteOnly"
