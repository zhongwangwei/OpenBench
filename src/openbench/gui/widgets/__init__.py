# -*- coding: utf-8 -*-
"""UI Widgets package."""

from openbench.gui.widgets.path_selector import PathSelector
from openbench.gui.widgets.checkbox_group import CheckboxGroup
from openbench.gui.widgets.yaml_preview import YamlPreview
from openbench.gui.widgets.progress_dashboard import ProgressDashboard, TaskStatus, TaskInfo
from openbench.gui.widgets.data_source_editor import DataSourceEditor
from openbench.gui.widgets.model_definition_editor import ModelDefinitionEditor
from openbench.gui.widgets.remote_config import RemoteConfigWidget
from openbench.gui.widgets.sync_status import SyncStatusWidget
from openbench.gui.widgets.path_completer import PathCompleter

__all__ = [
    "PathSelector",
    "CheckboxGroup",
    "YamlPreview",
    "ProgressDashboard",
    "TaskStatus",
    "TaskInfo",
    "DataSourceEditor",
    "ModelDefinitionEditor",
    "RemoteConfigWidget",
    "SyncStatusWidget",
    "PathCompleter",
]
