import pytest

from tests.gui_fakes import FakeButton  # noqa: E402

pytest.importorskip("PySide6")

from openbench.gui.main_window import MainWindow  # noqa: E402
from openbench.gui.pages.page_preview import PagePreview  # noqa: E402
from openbench.gui.widgets.validation_dialog import ValidationWorker  # noqa: E402


class FakeController:
    current_page = "run_monitor"

    def prev_page(self):
        return "preview"

    def next_page(self):
        return None


class RunningRunner:
    def isRunning(self):
        return True


def test_main_window_disables_navigation_while_runner_is_active():
    window = MainWindow.__new__(MainWindow)
    window.controller = FakeController()
    window.btn_back = FakeButton()
    window.btn_next = FakeButton()
    window.btn_rerun = FakeButton()
    window.pages = {"run_monitor": type("RunPage", (), {"_runner": RunningRunner()})()}

    window._update_buttons()

    assert window.btn_back.enabled is False
    assert window.btn_next.enabled is False
    assert window.btn_rerun.enabled is False
    assert window.btn_rerun.visible is True


def test_next_saves_current_page_before_navigation():
    events = []

    class Controller:
        current_page = "sim_data"

        def go_next(self):
            events.append("next")
            return True

    class Page:
        def validate(self):
            events.append("validate")
            return True

        def save_to_config(self):
            events.append("save")

    window = MainWindow.__new__(MainWindow)
    window.controller = Controller()
    window.pages = {"sim_data": Page()}
    window._runner_is_active = lambda: False
    window._save_current_page = lambda trigger_sync=False: window.pages["sim_data"].save_to_config()

    window._on_next_clicked()

    assert events == ["validate", "save", "next"]


def test_preview_ignores_duplicate_run_request_while_export_in_progress():
    preview = PagePreview.__new__(PagePreview)
    preview._export_in_progress = True
    calls = []
    preview._export_and_run_once = lambda: calls.append("run") or True

    assert preview.export_and_run() is False
    assert calls == []


def test_validation_worker_uses_snapshot_not_live_config_dicts():
    seen = {}

    class Validator:
        def validate_all(self, sources, general_config, progress_callback):
            seen["sources"] = sources
            seen["general"] = general_config
            return object()

    sources = {"Runoff::RefA": {"general": {"root_dir": "/old"}}}
    general = {"basedir": "/old-out"}
    worker = ValidationWorker(Validator(), sources, general)

    sources["Runoff::RefA"]["general"]["root_dir"] = "/new"
    general["basedir"] = "/new-out"
    worker.run()

    assert seen["sources"]["Runoff::RefA"]["general"]["root_dir"] == "/old"
    assert seen["general"]["basedir"] == "/old-out"


def test_progress_parser_comparison_stage_uses_specific_increment():
    from openbench.gui.progress_parser import parse_progress_line

    constants = {
        "PROGRESS_INIT": 5,
        "PROGRESS_WORK": 80,
        "PROGRESS_MAX": 95,
        "PROGRESS_INCREMENT": 2,
    }
    state = {
        "completed_eval_tasks": set(),
        "completed_groupby_tasks": set(),
        "completed_comparison_tasks": set(),
        "total_tasks": 0,
        "num_comparisons": 0,
        "num_variables": 0,
    }

    progress, _var, stage = parse_progress_line("Comparison", 5, state, constants)

    assert stage == "Comparison"
    assert progress == 7
