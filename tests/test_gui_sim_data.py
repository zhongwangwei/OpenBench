import shlex
from pathlib import Path
from types import SimpleNamespace

from openbench.gui.pages import page_sim_data


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


class _Text:
    def __init__(self, value):
        self._value = value

    def text(self):
        return self._value

    def currentText(self):
        return self._value


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
