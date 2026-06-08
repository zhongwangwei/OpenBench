"""Architecture cleanup regression checks."""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_runtime_dependencies_declare_direct_imports_without_unused_platformdirs():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = "\n".join(pyproject["project"]["dependencies"]).lower()
    conda_reqs = (ROOT / "conda/meta.yaml").read_text(encoding="utf-8").lower()

    for package in ("tqdm", "packaging"):
        assert package in dependencies
        assert package in conda_reqs
    assert "platformdirs" not in dependencies
    assert "platformdirs" not in conda_reqs


def test_core_public_api_does_not_export_unwired_evaluation_engine():
    import openbench
    import openbench.core as core
    import openbench.core.statistics as statistics

    assert core.__version__ == openbench.__version__
    assert statistics.__version__ == openbench.__version__
    for name in ("ModularEvaluationEngine", "GridEvaluationEngine", "StationEvaluationEngine"):
        assert name not in core.__all__
        assert not hasattr(core, name)


def test_unused_interface_abstractions_are_removed():
    import openbench.util.interfaces as interfaces

    for name in (
        "IVisualizationEngine",
        "IConfigurationManager",
        "IOrchestrator",
        "IResourceManager",
        "IEvaluationEngine",
        "IMetricsCalculator",
        "BaseEvaluator",
        "ProcessingPipeline",
    ):
        assert not hasattr(interfaces, name)


def test_detached_architecture_modules_are_removed_or_facades():
    assert not (ROOT / "src/openbench/_resources.py").exists()
    assert not (ROOT / "src/openbench/core/evaluation_engine.py").exists()
    assert not (ROOT / "src/openbench/data/pipeline.py").exists()

    wrapper = ROOT / "src/openbench/visualization/Mod_Only_Drawing.py"
    implementation = ROOT / "src/openbench/visualization/only_drawing.py"
    assert wrapper.exists()
    assert implementation.exists()
    assert len(wrapper.read_text(encoding="utf-8").splitlines()) < 40


def test_god_object_files_have_focused_helper_modules():
    assert (ROOT / "src/openbench/core/_comparison_helpers.py").exists()
    assert (ROOT / "src/openbench/data/_system_resources.py").exists()
    assert (ROOT / "src/openbench/data/registry/_filename_dates.py").exists()


def test_core_modules_do_not_import_visualization_at_module_import_time():
    # Probe the import graph in a subprocess so it never mutates this process's
    # sys.modules.  Popping/re-importing core modules in-process would corrupt
    # module identity for other tests that bound names at import time (e.g.
    # `from ...stat_anova import stat_anova`), silently breaking their patches.
    code = (
        "import sys\n"
        "import openbench.core.comparison  # noqa: F401\n"
        "import openbench.core.evaluation  # noqa: F401\n"
        "assert 'openbench.visualization' not in sys.modules, "
        "'importing openbench.core pulled in openbench.visualization'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_core_visualization_imports_are_lazy_not_module_top_level():
    for rel in (
        "src/openbench/core/comparison.py",
        "src/openbench/core/evaluation.py",
        "src/openbench/core/landcover_groupby.py",
        "src/openbench/core/climatezone_groupby.py",
    ):
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        for node in tree.body:
            assert not (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and node.module.startswith("openbench.visualization")
            ), rel


def test_core_visualization_lazy_imports_use_shared_bridge():
    for rel in (
        "src/openbench/core/comparison.py",
        "src/openbench/core/evaluation.py",
        "src/openbench/core/landcover_groupby.py",
        "src/openbench/core/climatezone_groupby.py",
    ):
        source = (ROOT / rel).read_text(encoding="utf-8")
        assert "from openbench.core._visualization_bridge import visualization_callable" in source


def test_user_selectable_metrics_exclude_known_unsafe_physics_metrics():
    from openbench.core.registry import IMPLEMENTED_METRICS, METRICS_ITEMS

    gui_metrics = {name for group in METRICS_ITEMS.values() for name in group}

    assert {"ubKGE", "kappa_coeff"}.isdisjoint(IMPLEMENTED_METRICS)
    assert {"ubKGE", "kappa_coeff"}.isdisjoint(gui_metrics)


def test_internal_import_errors_are_not_silently_downgraded():
    processing_source = (ROOT / "src/openbench/data/processing.py").read_text(encoding="utf-8")
    core_source = (ROOT / "src/openbench/core/__init__.py").read_text(encoding="utf-8")

    assert "BaseProcessor = object" not in processing_source
    assert "DataProcessingError = Exception" not in processing_source
    assert "Evaluation_grid = None" not in core_source
    assert "ComparisonProcessing = None" not in core_source


def test_callers_use_public_registry_scanner_helpers():
    sim_source = (ROOT / "src/openbench/data/sim_scanner.py").read_text(encoding="utf-8")
    cli_sources = (ROOT / "src/openbench/cli/data.py").read_text(encoding="utf-8") + (
        ROOT / "src/openbench/cli/_profile_rescue.py"
    ).read_text(encoding="utf-8")

    for source in (sim_source, cli_sources):
        assert "_inspect_nc_file" not in source
        assert "_filename_split_match" not in source
        assert "inspect_nc_file" in source
    assert "filename_split_match" in sim_source


def test_cache_error_messages_reference_current_cache_module():
    for rel in (
        "src/openbench/core/metrics.py",
        "src/openbench/core/evaluation.py",
        "src/openbench/data/processing.py",
    ):
        source = (ROOT / rel).read_text(encoding="utf-8")
        assert "openbench.data.Mod_CacheSystem" not in source
        assert "openbench.data.cache" in source


def test_dead_regrid_cdo_and_vendored_cmaps_setup_are_removed():
    assert not (ROOT / "src/openbench/data/regrid/regrid_cdo.py").exists()
    assert not (ROOT / "src/openbench/visualization/cmaps/setup.py").exists()
    regrid_init = (ROOT / "src/openbench/data/regrid/__init__.py").read_text(encoding="utf-8")
    assert "regridder_cdo" not in regrid_init


def test_wheel_registry_resources_do_not_need_force_include():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "tool.hatch.build.targets.wheel.force-include" not in pyproject
    assert "src/openbench/data/registry/reference_catalog.yaml" not in pyproject
    assert "src/openbench/visualization/cmaps/setup.py" not in pyproject


def test_version_single_source_of_truth_and_conda_parity():
    """Version must live only in __init__; pyproject is dynamic, conda must match (M4)."""
    import re

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    # pyproject derives the version dynamically from the package __init__.
    assert "version" in pyproject["project"].get("dynamic", []), "pyproject must declare dynamic version"
    assert pyproject["tool"]["hatch"]["version"]["path"] == "src/openbench/__init__.py"
    assert "version" not in pyproject["project"], "pyproject must not hard-code a static version"

    init_src = (ROOT / "src/openbench/__init__.py").read_text(encoding="utf-8")
    init_version = re.search(r'__version__\s*=\s*"([^"]+)"', init_src).group(1)

    import openbench

    assert openbench.__version__ == init_version

    conda_src = (ROOT / "conda/meta.yaml").read_text(encoding="utf-8")
    conda_version = re.search(r'{%\s*set\s+version\s*=\s*"([^"]+)"\s*%}', conda_src).group(1)
    assert conda_version == init_version, (
        f"conda/meta.yaml version {conda_version!r} != __init__ {init_version!r} — keep them in sync"
    )
