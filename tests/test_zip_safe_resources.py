"""Regression checks for wheel/zip-safe package resource access."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cmaps_builtin_resources_do_not_depend_on_package_file_paths():
    """Built-in cmaps resources must be read via importlib.resources."""
    cmaps_source = (ROOT / "src/openbench/visualization/cmaps/cmaps.py").read_text(encoding="utf-8")
    template_source = (ROOT / "src/openbench/visualization/cmaps/cmaps.template").read_text(encoding="utf-8")

    for source in (cmaps_source, template_source):
        assert "os.path.dirname(os.path.abspath(__file__))" not in source
        assert "os.path.join(CMAPSFILE_DIR" not in source

    assert "from importlib.resources import files" in cmaps_source
    assert "_cmap_resource(" in cmaps_source


def test_fignml_builder_uses_traversable_resources_not_path_exists():
    """Figure YAML loading must work when package files live inside a wheel zip."""
    adapter_source = (ROOT / "src/openbench/config/adapter.py").read_text(encoding="utf-8")

    assert 'files("openbench.data.fignml")' in adapter_source
    assert 'package_path("openbench.data.fignml")' not in adapter_source
    assert 'resource_path("openbench.data.fignml"' not in adapter_source


def test_registry_manager_uses_traversable_resources_for_builtins():
    """Built-in registry YAML must load from zipped wheels."""
    manager_source = (ROOT / "src/openbench/data/registry/manager.py").read_text(encoding="utf-8")

    assert 'files("openbench.data.registry")' in manager_source
    assert 'package_path("openbench.data.registry")' not in manager_source


def test_legacy_comparison_groupby_does_not_hardcode_package_static_dataset_paths():
    """Old comparison path should delegate groupby handling instead of resolving static datasets itself."""
    comparison_source = (ROOT / "src/openbench/core/_comparison_common.py").read_text(encoding="utf-8")

    assert "os.path.abspath(__file__)" not in comparison_source
    assert 'os.path.join(package_dir, "data", "IGBP.nc")' not in comparison_source
    assert 'os.path.join(package_dir, "dataset", "PFT.nc")' not in comparison_source
    assert "LC_groupby(self.main_nml, self.scores, self.metrics)" in comparison_source
    assert '"IGBP.nc"' not in comparison_source
    assert '"PFT.nc"' not in comparison_source


def test_comparison_groupby_methods_stay_delegation_only():
    """Guard against reintroducing dead LC/PFT implementation branches in comparison.py."""
    comparison_source = (ROOT / "src/openbench/core/_comparison_common.py").read_text(encoding="utf-8")

    assert "LC_groupby(self.main_nml, self.scores, self.metrics)" in comparison_source
    for legacy_marker in (
        "IGBP_remap",
        "PFT_remap",
        "IGBPtype",
        "PFTtype",
        "make_LC_based_heat_map",
        "_resolve_static_dataset",
        "_igbp_station_warning_shown",
        "_pft_station_warning_shown",
    ):
        assert legacy_marker not in comparison_source


def test_lc_cz_groupby_use_shared_static_dataset_resolver():
    """LC/CZ groupby should not maintain divergent static dataset resolution logic."""
    landcover_source = (ROOT / "src/openbench/core/landcover_groupby.py").read_text(encoding="utf-8")
    climate_source = (ROOT / "src/openbench/core/climatezone_groupby.py").read_text(encoding="utf-8")
    resolver_source = (ROOT / "src/openbench/util/static_datasets.py").read_text(encoding="utf-8")

    for source in (landcover_source, climate_source):
        assert "def _resolve_static_dataset" not in source
        assert "static_dataset_path(" in source

    assert "from importlib.resources import as_file, files" in resolver_source
    assert "OPENBENCH_DATASET_DIR" in resolver_source


def test_packaged_static_dataset_resolution_matches_cli_check_and_runtime(tmp_path, monkeypatch):
    """Packaged static datasets must pass preflight and still open through the runtime resolver."""
    from contextlib import contextmanager
    from types import SimpleNamespace

    import numpy as np
    import xarray as xr

    import openbench.cli.check as check_module
    import openbench.util.static_datasets as static_datasets

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(static_datasets.STATIC_DATASET_ENV, raising=False)

    package_root = tmp_path / "package" / "openbench"
    dataset_root = package_root / "dataset"
    dataset_root.mkdir(parents=True)
    xr.Dataset(
        {"IGBP": (("lat", "lon"), np.array([[1]], dtype=np.int32))},
        coords={"lat": [0.25], "lon": [0.25]},
    ).to_netcdf(dataset_root / "IGBP.nc")

    class FakeTraversable:
        def __init__(self, path: Path):
            self.path = path

        def __truediv__(self, child: str) -> "FakeTraversable":
            return FakeTraversable(self.path / child)

        def is_file(self) -> bool:
            return self.path.is_file()

    @contextmanager
    def fake_as_file(traversable: FakeTraversable):
        yield traversable.path

    monkeypatch.setattr(static_datasets, "files", lambda package: FakeTraversable(package_root))
    monkeypatch.setattr(static_datasets, "as_file", fake_as_file)

    assert static_datasets.static_dataset_exists("IGBP.nc") is True
    with static_datasets.static_dataset_path("IGBP.nc") as dataset_path:
        with xr.open_dataset(dataset_path) as dataset:
            assert int(dataset["IGBP"].values[0, 0]) == 1

    cfg = SimpleNamespace(project=SimpleNamespace(IGBP_groupby=True, PFT_groupby=False, climate_zone_groupby=False))
    assert check_module._groupby_static_dataset_findings(cfg) == []


def test_missing_static_dataset_message_matches_runtime_fallback(tmp_path, monkeypatch):
    """CLI miss messages and runtime fallback should point at the same legacy candidate."""
    from types import SimpleNamespace

    import openbench.cli.check as check_module
    import openbench.util.static_datasets as static_datasets

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(static_datasets.STATIC_DATASET_ENV, raising=False)

    class MissingTraversable:
        def __truediv__(self, child: str) -> "MissingTraversable":
            return self

        def is_file(self) -> bool:
            return False

    monkeypatch.setattr(static_datasets, "files", lambda package: MissingTraversable())

    cfg = SimpleNamespace(project=SimpleNamespace(IGBP_groupby=True, PFT_groupby=False, climate_zone_groupby=False))
    errors = check_module._groupby_static_dataset_findings(cfg)

    assert len(errors) == 1
    assert "IGBP_groupby requires static dataset IGBP.nc" in errors[0]
    assert "dataset/IGBP.nc" in errors[0]
    with static_datasets.static_dataset_path("IGBP.nc") as dataset_path:
        assert Path(dataset_path) == Path("dataset/IGBP.nc")


def test_custom_filter_resource_access_is_zip_safe():
    """Built-in custom filters must not be enumerated via package __file__."""
    init_source = (ROOT / "src/openbench/cli/init_cmd.py").read_text(encoding="utf-8")
    scanner_source = (ROOT / "src/openbench/data/registry/scanner.py").read_text(encoding="utf-8")

    assert "Path(custom_package.__file__).parent" not in init_source
    assert "Path(custom_package.__file__).parent" not in scanner_source
    assert 'files("openbench.data.custom")' in init_source
    assert 'files("openbench.data.custom")' in scanner_source


def test_gui_stylesheet_resource_access_is_zip_safe():
    """GUI styles must load without relying on a filesystem package directory."""
    app_source = (ROOT / "src/openbench/gui/app.py").read_text(encoding="utf-8")

    assert "os.path.dirname(__file__)" not in app_source
    assert 'files("openbench.gui")' in app_source
    assert "as_file" in app_source


def test_gui_root_detection_does_not_use_package_file_paths():
    """GUI install-root discovery should not depend on package __file__ paths."""
    path_utils_source = (ROOT / "src/openbench/gui/path_utils.py").read_text(encoding="utf-8")
    config_manager_source = (ROOT / "src/openbench/gui/config_manager.py").read_text(encoding="utf-8")

    assert "os.path.abspath(_ob.__file__)" not in path_utils_source
    assert "os.path.abspath(__file__)" not in config_manager_source
    assert 'files("openbench")' in path_utils_source


def test_scanner_reference_profiles_do_not_use_legacy_path_helpers():
    """Scanner package profiles should load via Traversable, not Path helpers."""
    scanner_source = (ROOT / "src/openbench/data/registry/scanner.py").read_text(encoding="utf-8")

    assert "resource_path(" not in scanner_source
    assert "open_resource(" not in scanner_source
    assert 'files("openbench.data.registry")' in scanner_source


def test_no_legacy_resources_abstraction_module():
    """Package resource access should use importlib.resources directly."""
    assert not (ROOT / "src/openbench/_resources.py").exists()


def test_ci_runs_zipimport_smoke_against_built_wheel():
    """CI should exercise the built wheel as a zip on PYTHONPATH."""
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "python -m build --wheel" in ci
    assert 'PYTHONPATH="$WHEEL" python -m openbench --help' in ci
    assert 'PYTHONPATH="$WHEEL" python -m openbench model list' in ci
    assert "ensure_user_registry_overlays(tmpdir)" in ci
    assert '_has_custom_station_filter("GEBA")' in ci
    assert "_prepare_stylesheet()" in ci
    assert "get_reference_profile" in ci
    assert "build_fig_nml()" in ci
    assert "cmaps.N3gauss" in ci
