"""Architecture cleanup regression checks."""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_runner_hashing_logic_lives_outside_local_god_module():
    local_source = (ROOT / "src/openbench/runner/local.py").read_text(encoding="utf-8")
    hashing_source = (ROOT / "src/openbench/runner/hashing.py").read_text(encoding="utf-8")

    assert "from openbench.runner import hashing as _runner_hashing" in local_source
    assert "def task_hash_payload(" in hashing_source
    assert "def input_file_signature(" in hashing_source
    assert "def _task_hash_payload(" in local_source  # compatibility wrapper only
    assert "def input_file_signature(" not in local_source


def test_runner_context_logic_lives_outside_local_god_module():
    local_source = (ROOT / "src/openbench/runner/local.py").read_text(encoding="utf-8")
    context_source = (ROOT / "src/openbench/runner/context.py").read_text(encoding="utf-8")

    assert "from openbench.runner import context as _runner_context" in local_source
    assert "class RuntimeContext" in context_source
    assert "def build_bridge_runtime_info(" in context_source
    assert "class RuntimeContext" not in local_source


def test_runner_preflight_logic_lives_outside_local_god_module():
    local_source = (ROOT / "src/openbench/runner/local.py").read_text(encoding="utf-8")
    preflight_source = (ROOT / "src/openbench/runner/preflight.py").read_text(encoding="utf-8")

    facade_source = (ROOT / "src/openbench/runner/preflight_facade.py").read_text(encoding="utf-8")

    assert "from openbench.runner import preflight_facade as _runner_preflight_facade" in local_source
    assert "def validate_comparison_only_inputs(" in preflight_source
    assert "def missing_expected_outputs(" in preflight_source
    assert "def validate_comparison_only_inputs(" in facade_source
    assert '"_validate_comparison_only_inputs": (_runner_preflight_facade' in local_source
    assert "def _validate_comparison_only_inputs(" not in local_source
    assert "def missing_expected_outputs(" not in local_source


def test_runner_pair_ref_logic_lives_outside_local_god_module():
    local_source = (ROOT / "src/openbench/runner/local.py").read_text(encoding="utf-8")
    pair_ref_source = (ROOT / "src/openbench/runner/pair_ref.py").read_text(encoding="utf-8")

    assert "from openbench.runner import pair_ref as _runner_pair_ref" in local_source
    assert "def clone_or_link_ref_for_pair(" in pair_ref_source
    assert "def cleanup_pair_ref_overrides(" in pair_ref_source
    assert "def _clone_or_link_ref_for_pair(" in local_source  # compatibility wrapper only
    assert "def clone_or_link_ref_for_pair(" not in local_source


def test_runner_dask_runtime_logic_lives_outside_local_god_module():
    local_source = (ROOT / "src/openbench/runner/local.py").read_text(encoding="utf-8")
    dask_source = (ROOT / "src/openbench/runner/dask_runtime.py").read_text(encoding="utf-8")

    assert "from openbench.runner import dask_runtime as _runner_dask_runtime" in local_source
    assert "def start_optional_dask_client(" in dask_source
    assert "def temporary_env_defaults(" in dask_source
    assert '"_start_optional_dask_client": (_runner_dask_runtime' in local_source
    assert "def _start_optional_dask_client(" not in local_source
    assert "def start_optional_dask_client(" not in local_source


def test_runner_task_execution_logic_lives_outside_local_god_module():
    local_source = (ROOT / "src/openbench/runner/local.py").read_text(encoding="utf-8")
    task_source = (ROOT / "src/openbench/runner/task_execution.py").read_text(encoding="utf-8")

    assert "from openbench.runner import task_execution as _runner_task_execution" in local_source
    assert "def evaluate_single(" in task_source
    assert "def _evaluate_single(" in local_source  # compatibility wrapper only
    assert "DatasetProcessing(info)" in task_source
    assert "def evaluate_single(" not in local_source


def test_runner_masking_logic_lives_outside_local_god_module():
    local_source = (ROOT / "src/openbench/runner/local.py").read_text(encoding="utf-8")
    masking_source = (ROOT / "src/openbench/runner/masking.py").read_text(encoding="utf-8")

    assert "from openbench.runner import masking as _runner_masking" in local_source
    assert "def apply_unified_mask(" in masking_source
    assert "def _apply_unified_mask(" in local_source  # compatibility wrapper only
    assert "np.array_equal" in masking_source
    assert "np.array_equal" not in local_source


def test_runner_postprocessing_logic_lives_outside_local_god_module():
    local_source = (ROOT / "src/openbench/runner/local.py").read_text(encoding="utf-8")
    post_source = (ROOT / "src/openbench/runner/postprocessing.py").read_text(encoding="utf-8")

    assert "from openbench.runner import postprocessing as _runner_postprocessing" in local_source
    for name in ("run_comparison", "run_groupby", "run_statistics", "run_report"):
        assert f"def {name}(" in post_source
    assert "ComparisonProcessing(namelists.main" in post_source
    assert "ComparisonProcessing(namelists.main" not in local_source
    assert "def _run_comparison(" in local_source


def test_runner_orchestration_logic_lives_outside_local_god_module():
    local_source = (ROOT / "src/openbench/runner/local.py").read_text(encoding="utf-8")
    orchestration_source = (ROOT / "src/openbench/runner/orchestration.py").read_text(encoding="utf-8")

    assert "from openbench.runner import orchestration as _runner_orchestration" in local_source
    assert "def run_evaluation_impl(" in orchestration_source
    assert "build_runner_bindings(cfg)" in orchestration_source
    assert "def _run_evaluation_impl(" in local_source  # compatibility wrapper only
    assert "Phase 1: Evaluation" not in local_source


def test_runner_preprocessing_logic_lives_outside_local_god_module():
    local_source = (ROOT / "src/openbench/runner/local.py").read_text(encoding="utf-8")
    preprocessing_source = (ROOT / "src/openbench/runner/preprocessing.py").read_text(encoding="utf-8")

    assert "from openbench.runner import preprocessing as _runner_preprocessing" in local_source
    assert "def preprocess_variable(" in preprocessing_source
    assert "DatasetProcessing(info)" in preprocessing_source
    assert 'prepare_source("ref")' in preprocessing_source
    assert "def _preprocess_variable(" not in local_source
    assert "DatasetProcessing(info)" not in local_source


def test_runner_task_planning_lives_outside_local_god_module():
    local_source = (ROOT / "src/openbench/runner/local.py").read_text(encoding="utf-8")
    planning_source = (ROOT / "src/openbench/runner/task_planning.py").read_text(encoding="utf-8")

    assert "from openbench.runner import task_planning as _runner_task_planning" in local_source
    assert "def build_evaluation_tasks(" in planning_source
    assert "def collect_cached_results(" in planning_source
    assert "EvaluationCache.hash_config(" in planning_source
    assert "EvaluationCache.hash_config(" not in local_source


def test_runner_evaluation_dispatch_lives_outside_local_god_module():
    local_source = (ROOT / "src/openbench/runner/local.py").read_text(encoding="utf-8")
    dispatch_source = (ROOT / "src/openbench/runner/evaluation_dispatch.py").read_text(encoding="utf-8")

    assert "from openbench.runner import evaluation_dispatch as _runner_evaluation_dispatch" in local_source
    assert "def evaluate_ready_tasks(" in dispatch_source
    assert "def task_level_parallel_safe(" in dispatch_source
    assert "ProcessPoolExecutor" in dispatch_source
    assert "def _evaluate_ready_tasks(" in local_source  # compatibility wrapper only
    assert "Task-level process parallelism disabled" not in local_source


def test_runner_cache_and_config_preflight_live_outside_local_god_module():
    local_source = (ROOT / "src/openbench/runner/local.py").read_text(encoding="utf-8")
    cache_source = (ROOT / "src/openbench/runner/cache_state.py").read_text(encoding="utf-8")
    config_preflight_source = (ROOT / "src/openbench/runner/config_preflight.py").read_text(encoding="utf-8")

    assert "from openbench.runner import cache_state as _runner_cache_state" in local_source
    assert "from openbench.runner import config_preflight as _runner_config_preflight" in local_source
    assert "def cached_task_result(" in cache_source
    assert "def existing_output_preflight_errors(" in config_preflight_source
    assert "def _cached_task_result(" in local_source  # compatibility wrapper only
    assert "EvaluationCache(cache_dir)" not in local_source
    assert "build_runner_bindings(cfg)" not in local_source
