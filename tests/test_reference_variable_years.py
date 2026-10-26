"""Variable-specific reference coverage must survive editing and reach consumers."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml

from openbench.config import adapter
from openbench.data.registry import manager
from openbench.data.registry.schema import VariableMapping


def _reference():
    return manager._build_reference(
        {
            "name": "CoverageRef",
            "data_type": "grid",
            "tim_res": "Month",
            "data_groupby": "Year",
            "years": [1950, 2022],
            "variables": {
                "Latent_Heat": {"varname": "le", "varunit": "W m-2", "years": [1980, 2022]},
                "Net_Radiation": {"varname": "rn", "varunit": "W m-2", "years": [1970, 2020]},
                "Sensible_Heat": {"varname": "h", "varunit": "W m-2"},
            },
        }
    )


def test_reference_years_roundtrip_and_partial_overlay():
    ref = _reference()
    assert ref.variables["Latent_Heat"].years == [1980, 2022]
    assert "years" not in VariableMapping("v", "K").to_dict()
    rebuilt = manager._build_reference(ref.to_dict())
    merged = manager._deep_merge_reference(rebuilt, {"variables": {"Latent_Heat": {"varunit": "J"}}})
    assert merged.variables["Latent_Heat"].years == [1980, 2022]
    cleared = manager._deep_merge_reference(merged, {"variables": {"Latent_Heat": {"years": None}}})
    from openbench.data.registry.schema import reference_years

    assert reference_years(cleared, cleared.variables["Latent_Heat"]) == [1950, 2022]


@pytest.mark.parametrize(
    "override,expected",
    [
        ({}, [1980, 2022]),
        ({"years": [2000, 2010]}, [2000, 2010]),
        ({"years": [2000, 2010], "variables": {"latent_heat": {"years": [2002, 2008]}}}, [2002, 2008]),
        ({"variables": {"Latent_Heat": {"years": [1990, 2005]}}}, [1990, 2005]),
        ({"variables": {"Latent_Heat": {"years": None}}}, [1950, 2022]),
    ],
)
def test_effective_years_override_precedence_without_mutation(override, expected):
    from openbench.data.registry.schema import reference_years

    ref = _reference()
    effective, mapping = adapter._apply_reference_override(ref, ref.variables["Latent_Heat"], "Latent_Heat", override)
    assert reference_years(effective, mapping) == expected
    assert ref.variables["Latent_Heat"].years == [1980, 2022]
    assert ref.years == [1950, 2022]
    assert reference_years(ref, ref.variables["Sensible_Heat"]) == ref.years


def test_saved_reference_keeps_variable_years(monkeypatch, tmp_path):
    catalog = tmp_path / "references" / "reference_catalog.yaml"
    monkeypatch.setattr(manager, "get_writable_reference_catalog_path", lambda: catalog)
    registry = manager.RegistryManager(user_dir=tmp_path)
    registry.save_reference("CoverageRef", _reference())
    raw = yaml.safe_load(catalog.read_text())["CoverageRef"]
    assert raw["variables"]["Latent_Heat"]["years"] == [1980, 2022]
    assert manager.RegistryManager(user_dir=tmp_path).get_reference("CoverageRef").variables["Latent_Heat"].years == [
        1980,
        2022,
    ]


def test_adapter_emits_independent_binding_years(monkeypatch):
    from openbench.config.schema import (
        EvaluationConfig,
        OpenBenchConfig,
        ProjectConfig,
        ReferenceConfig,
        SimulationEntry,
    )

    ref = _reference()
    registry = SimpleNamespace(
        get_model=lambda name: None, get_resolution_variants=lambda name: {}, get_reference=lambda name, **kwargs: ref
    )
    monkeypatch.setattr(manager, "get_registry", lambda: registry)
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="coverage", output_dir="./out", years=[1950, 2022]),
        evaluation=EvaluationConfig(variables=list(ref.variables)),
        reference=ReferenceConfig(sources={v: ref.name for v in ref.variables}),
        simulation={"Sim": SimulationEntry(model="M", root_dir="/sim", tim_res="Month", grid_res=0.5)},
    )
    _, nml, _ = adapter.build_legacy_namelists(cfg)
    for variable, expected in [
        ("Latent_Heat", [1980, 2022]),
        ("Net_Radiation", [1970, 2020]),
        ("Sensible_Heat", [1950, 2022]),
    ]:
        assert [nml[variable]["CoverageRef_syear"], nml[variable]["CoverageRef_eyear"]] == expected


@pytest.mark.parametrize(
    "alignment,expected", [("intersection", [1980, 1981, 1982]), ("strict", list(range(1978, 1983)))]
)
def test_cli_probes_effective_years_but_strict_keeps_project(monkeypatch, tmp_path, alignment, expected):
    import importlib

    check = importlib.import_module("openbench.cli.check")
    ref = _reference()
    (tmp_path / "le.nc").touch()
    monkeypatch.setattr(
        check, "_effective_reference", lambda *args: (ref, ref.variables["Latent_Heat"], str(tmp_path), str(tmp_path))
    )
    seen = []
    monkeypatch.setattr(check, "data_file_findings", lambda *a, **kw: (seen.append(kw["years"]) or [], []))
    cfg = SimpleNamespace(project=SimpleNamespace(years=[1978, 1982], time_alignment=alignment), simulation={})
    check._reference_file_findings(cfg, SimpleNamespace(var_name="Latent_Heat"), registry=object())
    assert seen == [expected]


def test_gui_and_remote_keep_variable_years():
    pytest.importorskip("PySide6")
    from openbench.gui.pages.page_ref_data import _registry_source_data
    from openbench.gui.pages.page_registry import _merge_reference_editor_dataset
    from openbench.gui.remote_registry import _build_remote_reference

    ref = _reference()
    edited = replace(ref, variables={"Latent_Heat": VariableMapping("edited", "J")})
    merged = _merge_reference_editor_dataset(ref, edited)
    assert merged.variables["Latent_Heat"].years == [1980, 2022]
    remote = _build_remote_reference(merged.to_dict())
    assert remote.variables["Latent_Heat"].years == [1980, 2022]
    source = _registry_source_data("latent_heat", ref.name, SimpleNamespace(get_reference=lambda name: ref))
    assert [source["general"]["syear"], source["general"]["eyear"]] == [1980, 2022]


@pytest.mark.parametrize("years", [[1980], [2022, 1980]])
def test_malformed_variable_window_is_not_silently_replaced(years):
    from openbench.data.registry.schema import reference_years

    ref = _reference()
    mapping = replace(ref.variables["Latent_Heat"], years=years)
    rebuilt = manager._build_reference(replace(ref, variables={"Latent_Heat": mapping}).to_dict())
    assert reference_years(rebuilt, rebuilt.variables["Latent_Heat"]) == years


def test_cli_metadata_and_init_warning_use_binding_window(monkeypatch, capsys):
    import importlib

    check = importlib.import_module("openbench.cli.check")
    init = importlib.import_module("openbench.cli.init_cmd")
    ref = _reference()
    mapping = ref.variables["Latent_Heat"]
    monkeypatch.setattr(check, "_effective_reference", lambda *args: (ref, mapping, None, None))
    cfg = SimpleNamespace(project=SimpleNamespace(years=[1970, 1979]))
    errors, _ = check._reference_metadata_findings(
        cfg, SimpleNamespace(status="ok", ref_ds=ref, resolved_name=ref.name), "Month", file_checks=False
    )
    assert any("[1980, 2022] do not overlap" in error for error in errors)
    init._warn_reference_year_coverage({"Latent_Heat": ref}, (1970, 1979))
    assert "1980-2022" in capsys.readouterr().out


def test_init_file_validation_only_requires_binding_years(tmp_path):
    import importlib

    import xarray as xr

    init = importlib.import_module("openbench.cli.init_cmd")
    ref = _reference()
    ref = replace(
        ref, root_dir=str(tmp_path), variables={"Latent_Heat": replace(ref.variables["Latent_Heat"], prefix="le_")}
    )
    for year in range(1980, 1983):
        xr.Dataset({"le": ("time", [1.0])}, coords={"time": [f"{year}-01-01"]}).to_netcdf(tmp_path / f"le_{year}.nc")
    init._validate_selected_reference_data({"Latent_Heat": ref}, (1978, 1982))


def test_save_cleared_bundled_variable_years_survives_reload(monkeypatch, tmp_path):
    catalog = tmp_path / "references" / "reference_catalog.yaml"
    monkeypatch.setattr(manager, "get_writable_reference_catalog_path", lambda: catalog)
    registry = manager.RegistryManager(user_dir=tmp_path)
    ref = registry.get_reference("ERA5LAND_LowRes")
    assert ref.variables["Latent_Heat"].years == [1980, 2022]
    cleared = manager._deep_merge_reference(ref, {"variables": {"Latent_Heat": {"years": None}}})
    registry.save_reference(ref.name, cleared)
    reloaded = manager.RegistryManager(user_dir=tmp_path).get_reference(ref.name)
    assert reloaded.variables["Latent_Heat"].years is None
    assert yaml.safe_load(catalog.read_text())[ref.name]["variables"]["Latent_Heat"]["years"] is None


@pytest.mark.parametrize("dataset_years", [None, [2000, 2010]])
@pytest.mark.parametrize("clear_binding", [False, True])
def test_gui_edit_one_binding_does_not_change_other_binding_years(monkeypatch, dataset_years, clear_binding):
    pytest.importorskip("PySide6")
    from openbench.data.registry.schema import reference_years
    from openbench.gui.config_manager import ConfigManager
    from openbench.gui.pages.page_ref_data import PageRefData, _registry_source_data
    from tests.test_gui_save_preserves_fields import FakeController, FakeLoadCombo, FakeText

    ref = _reference()
    registry = SimpleNamespace(get_reference=lambda name: ref)
    monkeypatch.setattr(manager, "get_registry", lambda: registry)
    variables = ["Latent_Heat", "Net_Radiation"]
    config = {
        "general": {"basename": "coverage", "basedir": "/out"},
        "evaluation_items": {var: True for var in variables},
        "ref_data": {
            "general": {f"{var}_ref_source": ref.name for var in variables},
            "source_configs": {
                f"{var}::{ref.name}": _registry_source_data(var, ref.name, registry) for var in variables
            },
        },
    }
    source = config["ref_data"]["source_configs"][f"Latent_Heat::{ref.name}"]
    source.update(varname="edited_le", _explicit_override=True)
    if dataset_years:
        source["general"].update(syear=dataset_years[0], eyear=dataset_years[1])
        source["_years_scope"] = "dataset"
    if clear_binding:
        source["years"] = None
    converter = ConfigManager()
    exported = yaml.safe_load(converter.generate_config_yaml(config))
    override = exported["reference"]["overrides"][ref.name]
    assert override.get("years") == dataset_years
    for var, expected in [("Latent_Heat", [1980, 2022]), ("Net_Radiation", [1970, 2020])]:
        effective, mapping = adapter._apply_reference_override(ref, ref.variables[var], var, override)
        expected = ref.years if clear_binding and var == "Latent_Heat" else expected
        assert reference_years(effective, mapping) == (dataset_years or expected)
    # Reload through the page's registry merge before a second save.
    controller = FakeController(converter.unified_to_gui_config(exported))
    controller.storage = object()
    page = PageRefData.__new__(PageRefData)
    page.controller = controller
    page.data_root_input = FakeText("")
    page._source_configs = {}
    page._var_combos = {var: FakeLoadCombo([("Choose", None), (ref.name, ref.name)]) for var in variables}
    page._var_advanced_fields = {
        var: {key: FakeText("") for key in ("varname", "varunit", "prefix", "suffix", "sub_dir")} for var in variables
    }
    page._rebuild_variable_groups = lambda: None
    page.load_from_config()
    second = yaml.safe_load(converter.generate_config_yaml(controller.config))
    saved_override = second["reference"]["overrides"][ref.name]
    if clear_binding:
        assert "years" in saved_override["variables"]["Latent_Heat"]
        assert saved_override["variables"]["Latent_Heat"]["years"] is None
    assert saved_override.get("years") == dataset_years
    for var, expected in [("Latent_Heat", [1980, 2022]), ("Net_Radiation", [1970, 2020])]:
        effective, mapping = adapter._apply_reference_override(ref, ref.variables[var], var, saved_override)
        expected = ref.years if clear_binding and var == "Latent_Heat" else expected
        assert reference_years(effective, mapping) == (dataset_years or expected)


@pytest.mark.parametrize("variable_scope", [True, False])
@pytest.mark.parametrize("years_override", ["absent", None, [1990, 2000]])
@pytest.mark.parametrize("shared_dataset_override", [False, True])
def test_split_namelist_exports_preserve_binding_years_locally_and_remotely(
    tmp_path, variable_scope, years_override, shared_dataset_override
):
    pytest.importorskip("PySide6")
    from openbench.gui.config_manager import ConfigManager
    from openbench.gui.pages.page_preview import PagePreview
    from openbench.gui.pages.page_ref_data import _registry_source_data

    ref = _reference()
    variables = ["Latent_Heat", "Net_Radiation"]
    registry = SimpleNamespace(get_reference=lambda name: ref)
    sources = [_registry_source_data(var, ref.name, registry) for var in variables]
    if not variable_scope:
        for source in sources:
            source.pop("_years_scope", None)
            source["general"].update(syear=2000, eyear=2010)
    dataset_years = None if variable_scope else [2000, 2010]
    if variable_scope and shared_dataset_override:
        dataset_years = [2005, 2015]
        sources[1]["_years_scope"] = "dataset"
        sources[1]["general"].update(syear=2005, eyear=2015)
    if years_override != "absent":
        sources[0]["years"] = years_override
    local_dir = tmp_path / "local"
    ConfigManager()._copy_data_namelists(
        {ref.name: ""},
        str(local_dir),
        variables,
        str(tmp_path),
        "ref",
        {f"{var}::{ref.name}": source for var, source in zip(variables, sources)},
    )
    local_text = (local_dir / f"{ref.name}.yaml").read_text()
    page = PagePreview.__new__(PagePreview)
    merged = page._merge_source_configs(sources, variables)
    remote_path = tmp_path / "remote.yaml"
    page._write_source_config_remote(merged, str(remote_path), variables, "/remote")
    remote_text = remote_path.read_text()
    for text in [local_text, remote_text]:
        exported = yaml.safe_load(text)
        assert "_years_scope" not in text
        assert "_dataset_years" not in text
        if variable_scope or years_override != "absent":
            assert "syear" not in exported["general"]
            assert "eyear" not in exported["general"]
            for var, years in [("Latent_Heat", [1980, 2022]), ("Net_Radiation", [1970, 2020])]:
                if dataset_years:
                    years = dataset_years
                if var == "Latent_Heat" and years_override != "absent":
                    years = years_override or dataset_years or ref.years
                assert [exported[var]["syear"], exported[var]["eyear"]] == years
                assert "years" not in exported[var]
        else:
            assert [exported["general"]["syear"], exported["general"]["eyear"]] == [2000, 2010]
            assert all("syear" not in exported[var] for var in variables)


def test_legacy_per_variable_time_ranges_keep_their_independent_windows():
    from openbench.gui.config_manager import _source_binding_time_range

    sources = [
        {"general": {"per_var_time_range": True, "syear": 1980, "eyear": 2022}},
        {"general": {"per_var_time_range": True, "syear": 1970, "eyear": 2020}},
    ]
    for source in sources:
        assert _source_binding_time_range(source, sources) == source["general"]


def test_split_null_override_without_parent_coverage_fails_closed():
    from openbench.gui.config_manager import _source_binding_time_range

    source = {"_years_scope": "variable", "general": {"syear": 1980, "eyear": 2022}, "years": None}
    with pytest.raises(ValueError, match="reload the reference"):
        _source_binding_time_range(source, [source])
