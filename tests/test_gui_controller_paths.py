import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from openbench.gui.controller import WizardController  # noqa: E402
from openbench.remote.storage import LocalStorage, RemoteStorage  # noqa: E402


def _controller(config, storage=None, project_root=""):
    controller = WizardController.__new__(WizardController)
    controller._config = config
    controller._storage = storage
    controller._project_root = project_root
    return controller


def test_remote_default_output_dir_uses_remote_openbench_path_not_local_root():
    controller = _controller(
        {
            "general": {
                "basename": "demo",
                "basedir": "./output",
                "remote": {"openbench_path": "/remote/openbench"},
            }
        },
        storage=RemoteStorage("/remote/openbench", sync_engine=object()),
        project_root="/local/source/tree",
    )

    assert controller.get_output_dir() == "/remote/openbench/output/demo"


def test_remote_relative_output_dir_uses_remote_storage_root_when_openbench_path_missing():
    controller = _controller(
        {"general": {"basename": "demo", "basedir": "runs"}},
        storage=RemoteStorage("/remote/project", sync_engine=object()),
        project_root="/local/source/tree",
    )

    assert controller.get_output_dir() == "/remote/project/runs/demo"


def test_local_relative_output_dir_uses_selected_relative_basedir():
    controller = _controller(
        {"general": {"basename": "demo", "basedir": "runs"}},
        storage=LocalStorage("/local/source/tree"),
        project_root="/local/source/tree",
    )

    assert controller.get_output_dir() == os.path.join("/local/source/tree", "runs", "demo")
