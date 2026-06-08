"""Architecture cleanup regression checks."""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_data_processing_utilities_live_outside_processing_god_module():
    processing_source = (ROOT / "src/openbench/data/processing.py").read_text(encoding="utf-8")
    utils_source = (ROOT / "src/openbench/data/_processing_utils.py").read_text(encoding="utf-8")

    config_source = (ROOT / "src/openbench/data/_processing_config.py").read_text(encoding="utf-8")
    time_adjustments_source = (ROOT / "src/openbench/data/_processing_time_adjustments.py").read_text(encoding="utf-8")

    assert "from openbench.data._processing_utils import" in config_source
    assert "from openbench.data._processing_utils import" in time_adjustments_source
    assert "def performance_monitor(" in utils_source
    assert "def parse_time_offset(" in utils_source
    assert "def get_coordinate_map(" in utils_source
    assert "def performance_monitor(" not in processing_source


def test_data_processing_time_integrity_lives_outside_processing_god_module():
    processing_source = (ROOT / "src/openbench/data/processing.py").read_text(encoding="utf-8")
    time_source = (ROOT / "src/openbench/data/_processing_time.py").read_text(encoding="utf-8")
    integrity_source = (ROOT / "src/openbench/data/_processing_time_integrity.py").read_text(encoding="utf-8")
    adjustment_source = (ROOT / "src/openbench/data/_processing_time_adjustments.py").read_text(encoding="utf-8")

    assert "from openbench.data._processing_time import TimeIntegrityMixin" in processing_source
    assert (
        "class BaseDatasetProcessing(BaseProcessingMixin, SelectionMixin, TimeIntegrityMixin, BaseProcessor):"
        in processing_source
    )
    assert "class TimeIntegrityMixin(TimeCoreMixin, TimeIntegrityWorkflowMixin, TimeAdjustmentMixin):" in time_source
    assert "def make_time_integrity(" in integrity_source
    assert "def apply_model_specific_time_adjustment(" in adjustment_source
    assert "normalize_cftime_axis(" in integrity_source
    assert "def make_time_integrity(" not in processing_source


def test_data_processing_selection_lives_outside_processing_god_module():
    processing_source = (ROOT / "src/openbench/data/processing.py").read_text(encoding="utf-8")
    selection_source = (ROOT / "src/openbench/data/_processing_selection.py").read_text(encoding="utf-8")

    assert "from openbench.data._processing_selection import SelectionMixin" in processing_source
    assert (
        "class BaseDatasetProcessing(BaseProcessingMixin, SelectionMixin, TimeIntegrityMixin, BaseProcessor):"
        in processing_source
    )
    assert "def select_var(" in selection_source
    assert "def _find_data_files(" in selection_source
    assert "decode_nonstandard_time(" in selection_source
    assert "def select_var(" not in processing_source


def test_data_processing_base_lifecycle_lives_outside_processing_god_module():
    processing_source = (ROOT / "src/openbench/data/processing.py").read_text(encoding="utf-8")
    base_source = (ROOT / "src/openbench/data/_processing_base.py").read_text(encoding="utf-8")
    config_source = (ROOT / "src/openbench/data/_processing_config.py").read_text(encoding="utf-8")
    transform_source = (ROOT / "src/openbench/data/_processing_transforms.py").read_text(encoding="utf-8")
    yearly_source = (ROOT / "src/openbench/data/_processing_yearly.py").read_text(encoding="utf-8")

    assert "from openbench.data._processing_base import BaseProcessingMixin" in processing_source
    assert (
        "class BaseDatasetProcessing(BaseProcessingMixin, SelectionMixin, TimeIntegrityMixin, BaseProcessor):"
        in processing_source
    )
    assert (
        "class BaseProcessingMixin(ProcessingConfigMixin, ProcessingTransformMixin, YearlyPreprocessingMixin):"
        in base_source
    )
    assert "def initialize_attributes(" in config_source
    assert "def setup_output_directories(" in config_source
    assert "def process_units(" in transform_source
    assert "def preprocess_yearly_files(" in yearly_source
    for name in ("initialize_attributes", "setup_output_directories", "process_units", "preprocess_yearly_files"):
        assert f"def {name}(" not in processing_source


def test_data_processing_station_and_grid_classes_are_thin_wrappers():
    processing_source = (ROOT / "src/openbench/data/processing.py").read_text(encoding="utf-8")
    station_source = (ROOT / "src/openbench/data/_processing_station.py").read_text(encoding="utf-8")
    station_core_source = (ROOT / "src/openbench/data/_processing_station_core.py").read_text(encoding="utf-8")
    grid_source = (ROOT / "src/openbench/data/_processing_grid.py").read_text(encoding="utf-8")
    grid_regrid_source = (ROOT / "src/openbench/data/_processing_grid_regrid.py").read_text(encoding="utf-8")

    assert "from openbench.data._processing_station import StationProcessingMixin" in processing_source
    assert "from openbench.data._processing_grid import GridProcessingMixin" in processing_source
    assert "class StationDatasetProcessing(StationProcessingMixin, BaseDatasetProcessing):" in processing_source
    assert "class GridDatasetProcessing(GridProcessingMixin, BaseDatasetProcessing):" in processing_source
    assert "class StationProcessingMixin(StationProcessingCoreMixin, StationExtractionMixin):" in station_source
    assert "class GridProcessingMixin(GridProcessingCoreMixin, GridRegridMixin):" in grid_source
    assert "def process_single_station_data(" in station_core_source
    assert "def remap_data(" in grid_regrid_source
    assert "def process_single_station_data(" not in processing_source
    assert "def remap_data(" not in processing_source
