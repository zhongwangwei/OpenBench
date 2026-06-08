from pathlib import Path

from openbench.gui import config_manager as config_manager_module
from openbench.gui import path_utils
from openbench.gui.config_manager import ConfigManager


def test_resolve_model_path_does_not_search_install_root_v2_nml_tree(tmp_path, monkeypatch):
    """v3 must not infer model definitions from install-root nml/nml-yaml."""
    install_root = tmp_path / "OpenBench"
    legacy_model_dir = install_root / "nml" / "nml-yaml" / "Mod_variables_definition"
    legacy_model_dir.mkdir(parents=True)
    (legacy_model_dir / "CoLM.yaml").write_text("general: {}\n")
    monkeypatch.setattr(config_manager_module, "get_openbench_root", lambda: str(install_root))

    missing_requested_path = tmp_path / "case" / "nml" / "sim" / "models" / "CoLM.nml"

    assert ConfigManager()._resolve_model_path(str(missing_requested_path)) is None


def test_resolve_model_path_keeps_adjacent_nml_to_yaml_compatibility(tmp_path):
    """Legacy .nml references may still resolve to an adjacent .yaml file."""
    yaml_path = tmp_path / "CoLM.yaml"
    yaml_path.write_text("general: {}\n")

    assert ConfigManager()._resolve_model_path(str(tmp_path / "CoLM.nml")) == str(yaml_path)


def test_path_utils_legacy_v2_markers_are_explicitly_named():
    """Old nml markers should be quarantined as legacy conversion markers."""
    assert "/nml/nml-yaml/" in path_utils._LEGACY_V2_PATH_MARKERS
    assert "/nml/nml-yaml/" not in path_utils._OPENBENCH_ROOT_MARKERS


def test_path_utils_accepts_colm_openbench_distribution_name(tmp_path):
    """Project-root detection must follow the PyPI/conda distribution name."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "colm-openbench"\n', encoding="utf-8")

    assert path_utils.looks_like_openbench_root(str(tmp_path)) is True


def test_gui_widgets_do_not_prefer_v2_nml_yaml_defaults():
    """Dialogs should not actively default to v2 nml/nml-yaml directories."""
    data_source_source = Path("src/openbench/gui/widgets/data_source_editor.py").read_text(encoding="utf-8")
    model_editor_source = Path("src/openbench/gui/widgets/model_definition_editor.py").read_text(encoding="utf-8")

    assert "OpenBench/nml/nml-yaml" not in data_source_source
    assert '"nml", "nml-yaml"' not in data_source_source
    assert "Ref_variables_definition_LowRes" not in data_source_source
    assert '"nml", "nml-yaml"' not in model_editor_source


def test_gui_metric_and_score_options_come_from_core_registry():
    """GUI must not advertise metrics/scores the core cannot execute."""
    from openbench.core.registry import IMPLEMENTED_METRICS, IMPLEMENTED_SCORES
    from openbench.gui.pages.page_metrics import METRICS_ITEMS
    from openbench.gui.pages.page_scores import SCORES_ITEMS

    gui_metrics = {name for group in METRICS_ITEMS.values() for name in group}
    gui_scores = {name for group in SCORES_ITEMS.values() for name in group}

    assert gui_metrics <= IMPLEMENTED_METRICS
    assert gui_scores <= IMPLEMENTED_SCORES
    assert {"dr", "APFB", "br2", "cp", "smpi"} <= gui_metrics
    assert {"SMPI", "MSE", "LNSE", "The_Ideal_Point_score"} & (gui_metrics | gui_scores) == set()
    assert {"index_agreement", "nSeasonalityScore"} <= gui_scores


def test_dead_page_options_duplicate_is_removed():
    assert not Path("src/openbench/gui/pages/page_options.py").exists()


def test_dead_page_variables_duplicate_is_removed():
    assert not Path("src/openbench/gui/pages/page_variables.py").exists()
