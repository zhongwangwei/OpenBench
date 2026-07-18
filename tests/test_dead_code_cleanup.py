"""Regression checks for detached utility cleanup."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_api_service_module_is_removed():
    """Detached API service module should not remain in the package tree."""
    assert not (ROOT / "src/openbench/util/api_service.py").exists()


def test_legacy_config_check_module_is_removed():
    """The v2 config-check helper should not remain as a legacy utility shell."""
    assert not (ROOT / "src/openbench/util/config_check.py").exists()


def test_legacy_config_modules_are_removed():
    """Detached legacy config manager/reader modules should not remain in the package tree."""
    assert not (ROOT / "src/openbench/config/legacy_manager.py").exists()
    assert not (ROOT / "src/openbench/config/legacy_readers.py").exists()


def test_general_info_reader_no_longer_inherits_namelist_reader():
    """GeneralInfoReader should stand alone once legacy reader modules are removed."""
    from openbench.config.runtime_info import GeneralInfoReader

    assert "NamelistReader" not in {base.__name__ for base in GeneralInfoReader.__mro__[1:]}


def test_detached_evaluation_engine_module_is_removed():
    """Unwired modular evaluation engine should not remain as a shadow path."""
    assert not (ROOT / "src/openbench/core/evaluation_engine.py").exists()


def test_evaluation_module_does_not_import_detached_engine():
    """evaluation.py should not expose the removed modular engine feature flag."""
    import openbench.core.evaluation as evaluation_module

    assert not hasattr(evaluation_module, "_HAS_MODULAR_ENGINE")
    assert not hasattr(evaluation_module, "create_evaluation_engine")


def test_evaluation_module_does_not_keep_unused_engine_aliases():
    """evaluation.py should not retain unused compatibility aliases from evaluation_engine."""
    import openbench.core.evaluation as evaluation_module

    assert not hasattr(evaluation_module, "GridEvaluationEngine")
    assert not hasattr(evaluation_module, "StationEvaluationEngine")
    assert not hasattr(evaluation_module, "ModularEvaluationEngine")
    assert not hasattr(evaluation_module, "evaluate_datasets")


def test_processing_module_no_longer_exposes_process_legacy():
    """Processing should no longer keep the detached legacy wrapper entrypoint."""
    import openbench.data.processing as processing_module

    assert not hasattr(processing_module.BaseDatasetProcessing, "process_legacy")


def test_runtime_modules_no_longer_special_case_legacy_list_varnames():
    """Legacy list-varname compatibility should be normalized before runtime hot paths."""
    processing_source = (ROOT / "src/openbench/data/processing.py").read_text(encoding="utf-8")
    adapter_source = (ROOT / "src/openbench/config/adapter.py").read_text(encoding="utf-8")

    assert "Try legacy list fallback" not in processing_source
    assert "Legacy list fallback" not in adapter_source


def test_runner_runtime_context_no_longer_falls_back_to_free_runtime_info_helper():
    """Runner should require bindings for runtime info instead of calling a free adapter helper."""
    runner_source = (ROOT / "src/openbench/runner/local.py").read_text(encoding="utf-8")

    assert "from openbench.config.adapter import build_runtime_info" not in runner_source
    assert "bridge_info = build_runtime_info(task)" not in runner_source


def test_adapter_no_longer_exports_free_runtime_info_helper():
    """Runtime info assembly should live on RunnerBindings, not as a module-level helper."""
    adapter_source = (ROOT / "src/openbench/config/adapter.py").read_text(encoding="utf-8")

    assert "\ndef build_runtime_info(" not in adapter_source


def test_adapter_no_longer_exposes_runner_bindings_compat_properties():
    """RunnerBindings should stop re-exporting raw legacy sections as compatibility properties."""
    adapter_source = (ROOT / "src/openbench/config/adapter.py").read_text(encoding="utf-8")

    assert "\n    def main_nl(" not in adapter_source
    assert "\n    def ref_nml(" not in adapter_source
    assert "\n    def sim_nml(" not in adapter_source
    assert "\n    def fig_nml(" not in adapter_source


def test_run_cli_no_longer_uses_legacy_dump_helper_name():
    """CLI debug dumping should use runner/debug naming all the way through."""
    run_source = (ROOT / "src/openbench/cli/run.py").read_text(encoding="utf-8")

    assert "_dump_legacy_config" not in run_source
    assert "runner_config.yaml" in run_source


def test_conventions_describe_adapter_output_without_legacy_bridge_wording():
    """Conventions should describe current adapter/runtime surfaces without legacy-bridge framing."""
    conventions = (ROOT / "src/openbench/CONVENTIONS.md").read_text(encoding="utf-8")

    assert "Legacy config adapter" not in conventions
    assert "Legacy bridge code uses" not in conventions


def test_no_config_legacy_modules_remain():
    """Active config modules should not be mislabeled as legacy."""
    legacy_modules = sorted(path.name for path in (ROOT / "src/openbench/config").glob("legacy_*.py"))
    assert legacy_modules == []


def test_orphaned_data_file_processing_module_is_removed():
    """file_processing.py only kept the old Lib_FileProcessing shell.

    It had no live imports in src/ or tests/; active data preparation now
    lives in the split _processing_* modules behind openbench.data.processing.
    """
    assert not (ROOT / "src/openbench/data/file_processing.py").exists()


# ---------------------------------------------------------------------------
# util/ orphaned modules — removed in batch after audit confirmed zero internal
# imports. Each was a generic helper written speculatively that never got
# adopted; replacement code lives elsewhere (e.g., progress.py → dask
# ProgressBar; fileio.py → direct xr.open_dataset; memory.py → psutil in
# data/processing.py). Historical migration doc references in
# docs/superpowers/plans/ remain as record but are not live imports.
# ---------------------------------------------------------------------------


def test_orphaned_util_cache_cleanup_module_is_removed():
    assert not (ROOT / "src/openbench/util/cache_cleanup.py").exists()


def test_orphaned_util_directory_module_is_removed():
    assert not (ROOT / "src/openbench/util/directory.py").exists()


def test_orphaned_util_fileio_module_is_removed():
    """fileio.py defined safe_open_netcdf / safe_save_netcdf etc. — never
    called anywhere in src/ or tests/. Replacement: direct xr.open_dataset
    with try/finally or with-blocks at call sites.
    """
    assert not (ROOT / "src/openbench/util/fileio.py").exists()


def test_orphaned_util_memory_module_is_removed():
    """memory.py provided memory monitoring with platform-color formatting.
    Replacement: psutil-based monitoring lives in data/processing.py
    (lines 280-440); no consumer needed the standalone module.
    """
    assert not (ROOT / "src/openbench/util/memory.py").exists()


def test_orphaned_util_progress_module_is_removed():
    """progress.py wrapped progress-bar primitives. Replacement: dask
    diagnostics ProgressBar used directly in data/processing.py.
    """
    assert not (ROOT / "src/openbench/util/progress.py").exists()
