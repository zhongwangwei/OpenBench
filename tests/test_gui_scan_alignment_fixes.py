"""Regression tests for GUI/CLI scan alignment and run-exit fixes.

Covers four reported GUI defects:
1. Loading a CLI openbench.yaml left the ref/sim scan-root fields empty.
2. GUI sim scan derived the prefix from the first file only, so
   one-file-per-variable cases (e.g. TE) loaded one variable's file for all
   variables; per-variable overrides were never exported.
3. `openbench run` could hang after printing the final summary (lingering
   worker pools), so the GUI/remote runners never saw the process exit.
4. Remote export rejected bare registry model names
   ("Remote model definition not found: CoLM2024").
"""

from types import SimpleNamespace

import yaml

from openbench.gui import path_utils
from openbench.gui.config_manager import (
    ConfigManager,
    is_builtin_model,
    model_definition_from_registry,
    registry_model_profile,
)
from openbench.gui.pages import page_sim_data
from tests.gui_fakes import FakeLineEdit as _Text

# ---------------------------------------------------------------------------
# Bug 1 — scan root inference
# ---------------------------------------------------------------------------


def test_infer_common_scan_root_posix_history_layout():
    assert (
        path_utils.infer_common_scan_root(
            [
                "/tera04/zhwei/Test20260106/Case01/history",
                "/tera04/zhwei/Test20260106/Case02/history",
                "/tera04/zhwei/Test20260106/Case03/history",
            ]
        )
        == "/tera04/zhwei/Test20260106"
    )


def test_infer_common_scan_root_windows_paths_without_history_leaf():
    assert (
        path_utils.infer_common_scan_root(
            [
                r"F:\streamlit\Cases_for_Openbench\LSMs\TE",
                "F:/streamlit/Cases_for_Openbench/LSMs/CLM5",
            ]
        )
        == "F:/streamlit/Cases_for_Openbench/LSMs"
    )


def test_infer_common_scan_root_single_case_and_empty_input():
    assert path_utils.infer_common_scan_root(["/data/Sim/Case01/history"]) == "/data/Sim"
    assert path_utils.infer_common_scan_root([]) == ""
    assert path_utils.infer_common_scan_root(["", None]) == ""


def test_unified_to_gui_config_infers_sim_scan_root():
    config = {
        "project": {"name": "demo"},
        "evaluation": {"variables": ["Evapotranspiration"]},
        "reference": {"Evapotranspiration": "GLEAM"},
        "simulation": {
            "_defaults": {"model": "CoLM2024"},
            "Case01": {"root_dir": "/tera04/zhwei/Tests/Case01/history"},
            "Case02": {"root_dir": "/tera04/zhwei/Tests/Case02/history"},
        },
    }

    gui_config = ConfigManager().unified_to_gui_config(config)

    assert gui_config["sim_data"]["_scan_root"] == "/tera04/zhwei/Tests"


def test_sim_page_load_from_config_falls_back_to_inferred_scan_root():
    page = SimpleNamespace(
        controller=SimpleNamespace(
            config={
                "sim_data": {
                    "general": {"Runoff_sim_source": ["Case01"]},
                    "source_configs": {
                        "Case01": {"general": {"root_dir": "/srv/sims/Case01/history"}},
                        "Case02": {"general": {"root_dir": "/srv/sims/Case02/history"}},
                    },
                }
            }
        ),
        _root_input=_Text(""),
        _clear_cases=lambda: None,
        _add_case_row=lambda *args, **kwargs: None,
        _settings_group=SimpleNamespace(setVisible=lambda _v: None),
    )

    page_sim_data.PageSimData.load_from_config(page)

    assert page._root_input.text() == "/srv/sims"


def test_ref_page_data_root_inferred_from_registry(monkeypatch):
    from openbench.gui.pages import page_ref_data

    class FakeRegistry:
        def get_reference(self, name):
            resolution = "LowRes" if name == "GLEAM" else "MidRes"
            return SimpleNamespace(root_dir=f"/vol/Reference/Grid/{resolution}")

    import openbench.data.registry.manager as manager_module

    monkeypatch.setattr(manager_module, "get_registry", lambda: FakeRegistry())

    inferred = page_ref_data._infer_ref_data_root(
        {"Evapotranspiration_ref_source": "GLEAM", "Albedo_ref_source": ["MODIS"]},
        ["Evapotranspiration", "Albedo"],
    )

    assert inferred == "/vol/Reference"


def test_ref_page_data_root_inferred_from_windows_registry_tree(monkeypatch):
    from openbench.gui.pages import page_ref_data

    class FakeRegistry:
        def get_reference(self, _name):
            return SimpleNamespace(root_dir="G:/Cases_for_openbench/Reference/Grid/MidRes")

    import openbench.data.registry.manager as manager_module

    monkeypatch.setattr(manager_module, "get_registry", lambda: FakeRegistry())

    inferred = page_ref_data._infer_ref_data_root(
        {"Latent_Heat_ref_source": "ERA5LAND_MidRes"},
        ["Latent_Heat"],
    )

    assert inferred == "G:/Cases_for_openbench/Reference"


def test_gui_reference_validation_uses_explicit_runtime_root(tmp_path):
    from openbench.gui.data_validator import DataValidator

    registered_root = tmp_path / "registered" / "Grid" / "MidRes"
    override_root = tmp_path / "override" / "Grid" / "MidRes"
    sub_dir = "Heat/Latent_Heat/ERA5LAND"
    data_dir = override_root / sub_dir
    data_dir.mkdir(parents=True)
    (data_dir / "ERA5LAND_2003.nc").touch()

    result = DataValidator(reference_data_root=str(override_root)).validate_source(
        "Latent_Heat",
        "ERA5LAND_MidRes",
        {
            "general": {"root_dir": str(registered_root), "data_type": "grid", "data_groupby": "Year"},
            "sub_dir": sub_dir,
            "prefix": "ERA5LAND_",
            "varname": "slhf",
        },
        {"syear": 2003, "eyear": 2003},
    )

    file_check = next(check for check in result.checks if check.name == "file_exists")
    assert file_check.passed is True
    assert str(override_root) in file_check.message


def test_gui_reference_validation_rejects_non_overlapping_years(tmp_path):
    from openbench.gui.data_validator import DataValidator

    data_root = tmp_path / "ref"
    data_root.mkdir()
    (data_root / "runoff_2015.nc").touch()

    result = DataValidator().validate_source(
        "Runoff",
        "DemoRef",
        {
            "general": {
                "root_dir": str(data_root),
                "data_type": "grid",
                "data_groupby": "Year",
                "syear": 2015,
                "eyear": 2020,
            },
            "prefix": "runoff_",
            "varname": "runoff",
        },
        {"syear": 2001, "eyear": 2002},
    )

    time_check = next(check for check in result.checks if check.name == "time_range")
    assert time_check.passed is False
    assert "do not overlap" in time_check.message


# ---------------------------------------------------------------------------
# Bug 2 — one-file-per-variable scan alignment with the CLI scanner
# ---------------------------------------------------------------------------


def test_gui_local_scan_accepts_a_case_directory_as_the_root(tmp_path):
    case_root = tmp_path / "Simulation" / "LSMs" / "CoLM2024"
    case_root.mkdir(parents=True)
    (case_root / "hist_2000.nc").touch()

    discovered, metadata = page_sim_data._scan_local_cases(str(case_root))

    assert discovered == [("CoLM2024", str(case_root), "hist_")]
    assert metadata["CoLM2024"]["model"] == "CoLM2024"


def test_gui_local_scan_discovers_nested_cases_instead_of_only_direct_children(tmp_path):
    root = tmp_path / "Simulation"
    colm = root / "LSMs" / "CoLM2024"
    clm = root / "LSMs" / "CLM5" / "history"
    colm.mkdir(parents=True)
    clm.mkdir(parents=True)
    (colm / "hist_2000.nc").touch()
    (clm / "hist_2000.nc").touch()

    discovered, metadata = page_sim_data._scan_local_cases(str(root))

    assert [case[0] for case in discovered] == ["CLM5", "CoLM2024"]
    assert metadata["CLM5"]["model"] == "CLM5"
    assert metadata["CoLM2024"]["model"] == "CoLM2024"


def test_gui_local_scan_does_not_display_one_variable_stream_as_case_prefix(tmp_path):
    case_root = tmp_path / "Simulation" / "LSMs" / "TE"
    case_root.mkdir(parents=True)
    for name in _TE_FILES:
        (case_root / name).touch()

    discovered, metadata = page_sim_data._scan_local_cases(str(case_root))

    assert discovered == [("TE", str(case_root), "YEE2_JRA-55_")]
    assert metadata["TE"]["multi_stream"] is True


def test_gui_local_scan_keeps_detected_case_metadata(monkeypatch, tmp_path):
    case_root = tmp_path / "StationCase"
    case_root.mkdir()
    scanned = SimpleNamespace(
        label="StationCase",
        root_dir=case_root,
        source_root=tmp_path,
        model="CoLM2024",
        prefix="",
        suffix="",
        variables=["Latent_Heat"],
        variable_overrides={},
        data_type="stn",
        grid_res=None,
        tim_res="Day",
        data_groupby="Single",
        fulllist=None,
        station_layout="flat",
    )
    monkeypatch.setattr(
        "openbench.data.sim_scanner.scan_simulation_roots",
        lambda *_args, **_kwargs: SimpleNamespace(cases=[scanned]),
    )

    _, metadata = page_sim_data._scan_local_cases(str(tmp_path))

    assert metadata["StationCase"] | {} == {
        "files": [],
        "suffix": "",
        "multi_stream": False,
        "model": "CoLM2024",
        "variables": ["Latent_Heat"],
        "variable_overrides": {},
        "data_type": "stn",
        "grid_res": None,
        "tim_res": "Day",
        "data_groupby": "Single",
        "fulllist": "",
        "station_layout": "flat",
        "source_root": str(tmp_path),
    }


def test_scan_confirmation_keeps_unchecked_cases_and_assigns_models_per_case(qapp):
    from openbench.gui.dialogs.scan_confirm import ScanConfirmDialog

    dialog = ScanConfirmDialog(
        discovered=[("CLM5", "/sim/CLM5", "hist_"), ("CoLM2024", "/sim/CoLM2024", "hist_")],
        model_names=["LEM2", "CLM5", "CoLM2024"],
        auto_model="",
        match_info="",
        nc_var_count=0,
        case_models={"CLM5": "CLM5", "CoLM2024": "CoLM2024"},
    )
    dialog._checkboxes[1].setChecked(False)

    results = dialog.get_results()

    assert [(item["model"], item["checked"]) for item in results] == [
        ("CLM5", True),
        ("CoLM2024", False),
    ]


def test_scan_confirmation_leaves_unresolved_model_blank(qapp):
    from openbench.gui.dialogs.scan_confirm import ScanConfirmDialog

    dialog = ScanConfirmDialog(
        discovered=[("Unknown", "/sim/Unknown", "")],
        model_names=["LEM2", "CLM5"],
        auto_model="",
        match_info="",
        nc_var_count=0,
    )

    assert dialog.get_results()[0]["model"] == ""


def test_scan_confirmation_does_not_claim_model_is_unresolved_when_case_model_was_detected(qapp):
    from PySide6.QtWidgets import QLabel

    from openbench.gui.dialogs.scan_confirm import ScanConfirmDialog

    dialog = ScanConfirmDialog(
        discovered=[("CaseA", "/sim/CaseA", "")],
        model_names=["CoLM2024"],
        auto_model="",
        match_info="CaseA: CoLM2024",
        nc_var_count=10,
        case_models={"CaseA": "CoLM2024"},
    )

    text = " ".join(label.text() for label in dialog.findChildren(QLabel))
    assert "No model auto-detected" not in text


_TE_FILES = [
    "YEE2_JRA-55_alb_Mon_2000.nc",
    "YEE2_JRA-55_alb_Mon_2001.nc",
    "YEE2_JRA-55_lai_Mon_2000.nc",
    "YEE2_JRA-55_lai_Mon_2001.nc",
]


def test_case_file_patterns_single_stream_uses_date_split():
    prefix, suffix, multi = page_sim_data._case_file_patterns(["hist_2000.nc4", "hist_2001.nc4"])
    assert (prefix, suffix, multi) == ("hist_", "", False)


def test_case_file_patterns_one_file_per_variable_is_multi_stream():
    prefix, suffix, multi = page_sim_data._case_file_patterns(_TE_FILES)
    assert multi is True
    assert prefix == "YEE2_JRA-55_"  # common prefix, not the first file's
    assert suffix == ""


def _fake_te_registry(monkeypatch):
    mapping_alb = SimpleNamespace(varname="alb", fallbacks=None, compute=None)
    mapping_lai = SimpleNamespace(varname="lai", fallbacks=None, compute=None)
    profile = SimpleNamespace(variables={"Albedo": mapping_alb, "Leaf_Area_Index": mapping_lai})

    class FakeRegistry:
        def get_model(self, name):
            return profile if name == "TE" else None

    import openbench.data.registry.manager as manager_module

    monkeypatch.setattr(manager_module, "get_registry", lambda: FakeRegistry())


def test_filename_variable_overrides_map_each_variable_to_its_stream(monkeypatch):
    _fake_te_registry(monkeypatch)

    overrides = page_sim_data._filename_variable_overrides(_TE_FILES, "TE")

    assert overrides["Albedo"]["prefix"] == "YEE2_JRA-55_alb_Mon_"
    assert overrides["Leaf_Area_Index"]["prefix"] == "YEE2_JRA-55_lai_Mon_"


def test_filename_variable_overrides_skip_single_stream(monkeypatch):
    _fake_te_registry(monkeypatch)
    assert page_sim_data._filename_variable_overrides(["hist_2000.nc", "hist_2001.nc"], "TE") == {}


class _FakeCheck:
    def isChecked(self):
        return True


class _FakeCombo:
    def __init__(self, value):
        self._value = value

    def currentData(self):
        return self._value


def _selected_cases(case):
    page = SimpleNamespace(_prefix_input=_Text(""), _suffix_input=_Text(""), _cases=[case])
    return page_sim_data.PageSimData.get_selected_cases(page)


def test_get_selected_cases_uses_per_case_scan_metadata_when_shared_overrides_are_blank():
    case = {
        "checkbox": _FakeCheck(),
        "model_combo": _FakeCombo("CoLM2024"),
        "label": "StationCase",
        "nc_dir": "/sims/StationCase",
        "auto_prefix": "",
        "auto_suffix": "",
        "variable_overrides": {},
        "scan_metadata": {
            "data_type": "stn",
            "tim_res": "Day",
            "grid_res": None,
            "data_groupby": "Single",
            "station_layout": "flat",
        },
    }
    page = SimpleNamespace(
        _prefix_input=_Text("global_"),
        _suffix_input=_Text(".nc"),
        _data_type_combo=_FakeCombo(""),
        _tim_res_combo=_FakeCombo(""),
        _data_groupby_combo=_FakeCombo(""),
        _grid_res_input=_Text(""),
        _cases=[case],
    )

    (selected,) = page_sim_data.PageSimData.get_selected_cases(page)

    assert selected["data_type"] == "stn"
    assert selected["tim_res"] == "Day"
    assert selected["grid_res"] is None
    assert selected["data_groupby"] == "Single"
    assert selected["station_layout"] == "flat"
    assert selected["prefix"] == ""
    assert selected["suffix"] == ""


def test_gui_tim_res_options_cover_all_config_values():
    from openbench.config.loader import VALID_TIM_RES_VALUES

    assert set(page_sim_data.SIM_TIM_RES_OPTIONS) == VALID_TIM_RES_VALUES


def test_manual_grid_res_override_is_exported_as_a_number():
    assert page_sim_data._grid_res_value("0.25") == 0.25
    assert page_sim_data._grid_res_value(0.5) == 0.5


def test_get_selected_cases_suppresses_case_prefix_for_multi_stream():
    case = {
        "checkbox": _FakeCheck(),
        "model_combo": _FakeCombo("TE"),
        "label": "TE",
        "nc_dir": "/sims/TE/history",
        "auto_prefix": "YEE2_JRA-55_",
        "auto_suffix": "",
        "variable_overrides": {
            "Albedo": {"prefix": "YEE2_JRA-55_alb_Mon_"},
            "Leaf_Area_Index": {"prefix": "YEE2_JRA-55_lai_Mon_"},
        },
        "multi_stream": True,
    }

    (selected,) = _selected_cases(case)

    # Mirrors cli/sim._case_prefix_is_safe_to_write: a case-level prefix would
    # apply one stream's files to every variable, so it must not be exported.
    assert selected["prefix"] == ""
    assert selected["variables"]["Albedo"]["prefix"] == "YEE2_JRA-55_alb_Mon_"


def test_get_selected_cases_keeps_prefix_for_single_stream():
    case = {
        "checkbox": _FakeCheck(),
        "model_combo": _FakeCombo("CLM5"),
        "label": "CLM5",
        "nc_dir": "/sims/CLM5/history",
        "auto_prefix": "hist_",
        "auto_suffix": "",
        "variable_overrides": {},
        "multi_stream": False,
    }

    (selected,) = _selected_cases(case)

    assert selected["prefix"] == "hist_"
    assert selected["variables"] == {}


def test_get_selected_cases_uses_per_case_pattern_edits():
    case = {
        "checkbox": _FakeCheck(),
        "model_combo": _FakeCombo("GLDAS"),
        "label": "GLDAS",
        "nc_dir": "/sims/GLDAS",
        "auto_prefix": "wrong_",
        "auto_suffix": "",
        "prefix_input": _Text("GLDAS_NOAH025_M.A"),
        "suffix_input": _Text(".021"),
        "case_pattern_edited": True,
        "variable_overrides": {},
        "multi_stream": False,
    }

    (selected,) = _selected_cases(case)

    assert selected["prefix"] == "GLDAS_NOAH025_M.A"
    assert selected["suffix"] == ".021"


def test_simulation_variable_editor_exposes_prefix_and_suffix(qapp):
    from openbench.gui.widgets.variable_editor import VariableEditorDialog

    dialog = VariableEditorDialog(
        mode="simulation",
        variable_name="Albedo",
        varname="alb",
        prefix="YEE2_JRA-55_alb_M",
        suffix="GLB050",
    )

    data = dialog.get_data()
    assert data["variable_name"] == "Albedo"
    assert data["varname"] == "alb"
    assert data["prefix"] == "YEE2_JRA-55_alb_M"
    assert data["suffix"] == "GLB050"
    assert "sub_dir" not in data


def test_variable_pattern_edit_preserves_inferred_metadata_and_can_clear_fields():
    overrides = {
        "Albedo": {
            "varname": "alb",
            "prefix": "old_",
            "suffix": "old.nc",
            "grid_res": 0.5,
        }
    }

    edited = page_sim_data._apply_variable_pattern_edit(
        overrides,
        "Albedo",
        {
            "variable_name": "Albedo",
            "varname": "alb",
            "varunit": "1",
            "prefix": "new_",
            "suffix": "",
        },
    )

    assert edited == {
        "Albedo": {
            "varname": "alb",
            "varunit": "1",
            "prefix": "new_",
            "grid_res": 0.5,
        }
    }


def test_generate_config_yaml_exports_per_variable_overrides_without_case_prefix():
    config = {
        "general": {"basename": "demo", "basedir": "/out", "syear": 2000, "eyear": 2001},
        "evaluation_items": {"Albedo": True, "Leaf_Area_Index": True},
        "metrics": {"RMSE": True},
        "scores": {},
        "comparisons": {},
        "statistics": {},
        "ref_data": {
            "general": {
                "data_root": "/ref",
                "Albedo_ref_source": "MODIS",
                "Leaf_Area_Index_ref_source": "MODIS",
            }
        },
        "sim_data": {
            "general": {"Albedo_sim_source": ["TE"], "Leaf_Area_Index_sim_source": ["TE"]},
            "source_configs": {
                "TE": {
                    "general": {
                        "model_namelist": "TE",
                        "root_dir": "/sims/TE/history",
                        "data_groupby": "month",
                        "prefix": "",
                        "suffix": "",
                    },
                    "variables": {
                        "Albedo": {"prefix": "YEE2_JRA-55_alb_Mon_"},
                        "Leaf_Area_Index": {"prefix": "YEE2_JRA-55_lai_Mon_"},
                    },
                }
            },
        },
    }

    data = yaml.safe_load(ConfigManager().generate_config_yaml(config))

    entry = data["simulation"]["TE"]
    assert "prefix" not in entry
    assert entry["variables"]["Albedo"]["prefix"] == "YEE2_JRA-55_alb_Mon_"
    assert entry["variables"]["Leaf_Area_Index"]["prefix"] == "YEE2_JRA-55_lai_Mon_"


# ---------------------------------------------------------------------------
# Bug 3 — run command must terminate once results are final
# ---------------------------------------------------------------------------


def test_release_worker_pools_is_safe_to_call():
    from openbench.cli.run import _release_worker_pools

    _release_worker_pools()  # must never raise, with or without joblib installed


def test_arm_exit_watchdog_starts_daemon_timer_in_standalone_cli(monkeypatch):
    import threading

    from openbench.cli import run as run_module

    created = {}

    class FakeTimer:
        def __init__(self, timeout, callback):
            created["timeout"] = timeout
            created["callback"] = callback
            self.daemon = False

        def start(self):
            created["started"] = True
            created["daemon"] = self.daemon

    monkeypatch.setattr(threading, "Timer", FakeTimer)
    monkeypatch.setattr(run_module, "_is_standalone_cli_process", lambda: True)

    run_module._arm_exit_watchdog(0, timeout=33.0)

    assert created["started"] is True
    assert created["daemon"] is True  # must not keep a healthy process alive
    assert created["timeout"] == 33.0


def test_arm_exit_watchdog_never_arms_inside_embedding_processes(monkeypatch):
    """pytest/CliRunner/API embedders must not be os._exit'ed by a timer."""
    import threading

    from openbench.cli import run as run_module

    def _fail(*_args, **_kwargs):
        raise AssertionError("watchdog timer must not be created in an embedded process")

    monkeypatch.setattr(threading, "Timer", _fail)

    assert run_module._is_standalone_cli_process() is False  # we run under pytest
    run_module._arm_exit_watchdog(0)  # must be a no-op


# ---------------------------------------------------------------------------
# Bug 4 — bare registry model names in (remote) export
# ---------------------------------------------------------------------------


def test_registry_model_profile_resolves_builtin_name_not_paths():
    assert registry_model_profile("CoLM2024") is not None
    assert registry_model_profile("./nml/models/CoLM.nml") is None
    assert registry_model_profile("C:/Users/me/CoLM2024.yaml") is None
    assert registry_model_profile("") is None


def test_model_definition_from_registry_filters_selected_items():
    content = model_definition_from_registry("CoLM2024", ["Evapotranspiration"])

    assert content is not None
    assert content["general"]["model"] == "CoLM2024"
    assert "Evapotranspiration" in content
    assert "varname" in content["Evapotranspiration"]


def test_is_builtin_model_distinguishes_user_models():
    assert is_builtin_model("CoLM2024") is True
    assert is_builtin_model("LEM2-definitely-not-builtin") is False


def test_remote_model_sync_accepts_registry_name(monkeypatch, tmp_path):
    """A bare registry model name must export without remote file lookups."""
    from openbench.gui.pages import page_preview as page_preview_module
    from openbench.gui.pages.page_preview import PagePreview

    monkeypatch.setattr(page_preview_module, "get_remote_ssh_manager", lambda _controller: None)

    page = PagePreview.__new__(PagePreview)
    staged = []
    page._stage_remote_registry_model = lambda name, ssh: staged.append(name)
    page.controller = SimpleNamespace(config={}, storage=None)

    config = {
        "evaluation_items": {"Evapotranspiration": True},
        "sim_data": {
            "source_configs": {
                "Case01": {
                    "general": {"model_namelist": "CoLM2024", "root_dir": "/remote/sims/Case01"},
                }
            }
        },
        "ref_data": {"source_configs": {}},
    }

    PagePreview._sync_namelists_for_remote(page, config, str(tmp_path), "/remote/out", "/remote/openbench")

    model_file = tmp_path / "nml" / "sim" / "models" / "CoLM2024.yaml"
    assert model_file.exists()
    written = yaml.safe_load(model_file.read_text())
    assert written["general"]["model"] == "CoLM2024"
    assert staged == ["CoLM2024"]
