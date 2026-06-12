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
            return SimpleNamespace(root_dir=f"/vol/Reference/Grid/{name}")

    import openbench.data.registry.manager as manager_module

    monkeypatch.setattr(manager_module, "get_registry", lambda: FakeRegistry())

    inferred = page_ref_data._infer_ref_data_root(
        {"Evapotranspiration_ref_source": "GLEAM", "Albedo_ref_source": ["MODIS"]},
        ["Evapotranspiration", "Albedo"],
    )

    assert inferred == "/vol/Reference/Grid"


# ---------------------------------------------------------------------------
# Bug 2 — one-file-per-variable scan alignment with the CLI scanner
# ---------------------------------------------------------------------------

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
