"""Tests for RegistryManager."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import openbench.cli.check as check_module
import openbench.config as config_module
import openbench.data.registry.manager as registry_manager_module
from openbench.config.adapter import build_legacy_namelists
from openbench.config.schema import (
    ComparisonConfig,
    EvaluationConfig,
    OpenBenchConfig,
    ProjectConfig,
    ReferenceConfig,
    SimulationEntry,
)
from openbench.data.registry.manager import RegistryManager, _auto_resolve_variant
from openbench.data.registry.schema import ModelProfile, ReferenceDataset, VariableMapping


def _load_builtin_yaml(filename: str) -> dict:
    import yaml

    path = registry_manager_module.REGISTRY_DIR / filename
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_legacy_yaml_dir(dirname: str) -> dict[str, dict]:
    import yaml

    root = _legacy_openbench_wei_root() / "nml" / "nml-yaml" / dirname
    if not root.exists():
        pytest.skip(f"legacy OpenBench-wei definitions not available: {root}")

    return {path.stem: yaml.safe_load(path.read_text(encoding="utf-8")) or {} for path in sorted(root.glob("*.yaml"))}


def _legacy_openbench_wei_root() -> Path:
    return Path(__file__).resolve().parents[2] / "bk" / "external" / "OpenBench-wei"


def _legacy_variable_names(data: dict) -> set[str]:
    return {
        _canonical_variable_name(name)
        for name, value in data.items()
        if name != "general" and isinstance(value, dict) and ("varname" in value or "varunit" in value)
    }


def _canonical_variable_name(name: str) -> str:
    legacy_canopy_name = "Canopy_" + "Interception"
    renamed = {
        legacy_canopy_name: "Canopy_Evaporation",
    }
    return renamed.get(name, name)


def _catalog_variable_names(data: dict) -> set[str]:
    return {name for name in data.get("variables", {})}


def _reference_entry_for_legacy_lowres(catalog: dict, legacy_name: str) -> str | None:
    aliases = {
        "CRU_TS_4.08_LowRes_Precipitation": "CRU_TS_4.08_LowRes",
        "GIEMS-MC": "GIEMS_MC_LowRes",
    }
    candidates = [
        aliases.get(legacy_name),
        legacy_name,
        f"{legacy_name}_LowRes",
        f"{legacy_name.replace('-', '_')}_LowRes",
    ]
    return next((name for name in candidates if name and name in catalog), None)


def test_builtin_reference_catalog_includes_legacy_lowres_variable_definitions():
    catalog = _load_builtin_yaml("reference_catalog.yaml")
    missing_entries = []
    missing_variables = {}

    for legacy_name, legacy_data in _load_legacy_yaml_dir("Ref_variables_definition_LowRes").items():
        entry_name = _reference_entry_for_legacy_lowres(catalog, legacy_name)
        if entry_name is None:
            missing_entries.append(legacy_name)
            continue

        missing = _legacy_variable_names(legacy_data) - _catalog_variable_names(catalog[entry_name])
        if missing:
            missing_variables[f"{legacy_name}->{entry_name}"] = sorted(missing)

    assert missing_entries == []
    assert missing_variables == {}


def test_legacy_lowres_aliases_use_current_reference_tree_subdirs():
    catalog = _load_builtin_yaml("reference_catalog.yaml")
    data_root = _legacy_openbench_wei_root() / "dataset" / "Reference" / "Grid" / "LowRes"
    if not data_root.exists():
        pytest.skip(f"legacy reference data tree not available: {data_root}")

    expected_subdirs = {
        ("AH4GUC_LowRes", "Urban_Anthropogenic_Heat_Flux"): "Anth/Urban/AH4GUC",
        ("ERA5LAND_LowRes", "Surface_Wind_Speed"): "Meteo/Surface_Wind_Speed/ERA5LAND",
        ("ETMonitor_LowRes", "Urban_Latent_Heat_Flux"): "Anth/Urban/ETMonitor",
        ("GGMSEUD_LowRes", "Total_Irrigation_Amount"): "Anth/Crop/GGMSEUD",
        ("GIWUED_LowRes", "Total_Irrigation_Amount"): "Anth/Crop/GIWUED",
        ("GLEAM_v4.2a_LowRes", "Canopy_Transpiration"): "Water/Transpiration/GLEAM_v4.2a",
        ("GRFR_LowRes", "Runoff"): "Water/Total_Runoff/GRFR",
        ("MCD43A3_LowRes", "Urban_Albedo"): "Anth/Urban/MCD43A3",
        ("MODIS_LST_LowRes", "Urban_Surface_Temperature"): "Anth/Urban/MODIS_LST",
        ("TEMP_Zhang_etal_2022_LowRes", "Urban_Air_Temperature_Max"): "Anth/Urban/TEMP_Zhang_etal_2022",
        ("TEMP_Zhang_etal_2022_LowRes", "Urban_Air_Temperature_Min"): "Anth/Urban/TEMP_Zhang_etal_2022",
        ("TRIMS_LowRes", "Urban_Surface_Temperature"): "Anth/Urban/TRIMS",
    }

    mismatches = {}
    for (entry_name, variable_name), expected in expected_subdirs.items():
        actual = catalog[entry_name]["variables"][variable_name].get("sub_dir")
        if actual != expected or not (data_root / expected).exists():
            mismatches[f"{entry_name}.{variable_name}"] = actual

    assert mismatches == {}


def test_builtin_reference_catalog_omits_scanner_placeholder_variables():
    catalog = _load_builtin_yaml("reference_catalog.yaml")
    placeholders = {
        "AH4GUC_LowRes": {"Urban"},
        "ETMonitor_LowRes": {"Urban"},
        "GGMSEUD_LowRes": {"Crop"},
        "GIWUED_LowRes": {"Crop"},
        "GRFR_LowRes": {"Total_Runoff"},
        "MCD43A3_LowRes": {"Urban"},
        "MODIS_LST_LowRes": {"Urban"},
        "ResOpsUS": {"Dam"},
        "TEMP_Zhang_etal_2022_LowRes": {"Urban"},
        "TRIMS_LowRes": {"Urban"},
    }

    offenders = {
        name: sorted(disallowed & set(catalog[name].get("variables", {})))
        for name, disallowed in placeholders.items()
        if disallowed & set(catalog[name].get("variables", {}))
    }

    assert offenders == {}


def test_fluxnet_plumber2_does_not_advertise_latent_heat_as_evapotranspiration():
    """FLUXNET/PLUMBER2 Qle_cor is an energy flux, not an ET depth flux."""

    catalog = _load_builtin_yaml("reference_catalog.yaml")
    profiles = _load_builtin_yaml("reference_profiles.yaml")

    assert "Evapotranspiration" not in catalog["FLUXNET_PLUMBER2"]["variables"]
    assert "Evapotranspiration" not in profiles["FLUXNET_PLUMBER2"]["variables"]
    assert catalog["FLUXNET_PLUMBER2"]["variables"]["Latent_Heat"]["varname"] == "Qle_cor"
    assert catalog["FLUXNET_PLUMBER2"]["variables"]["Latent_Heat"]["varunit"].lower() == "w m-2"


def test_builtin_reference_catalog_includes_legacy_station_variable_definitions():
    catalog = _load_builtin_yaml("reference_catalog.yaml")
    aliases = {"GRDC_daily": "GRDC_Daily"}
    missing_entries = []
    missing_variables = {}

    for legacy_name, legacy_data in _load_legacy_yaml_dir("Ref_variables_definition_station").items():
        entry_name = aliases.get(legacy_name, legacy_name)
        if entry_name not in catalog:
            missing_entries.append(legacy_name)
            continue

        missing = _legacy_variable_names(legacy_data) - set(catalog[entry_name].get("variables", {}))
        if missing:
            missing_variables[f"{legacy_name}->{entry_name}"] = sorted(missing)

    assert missing_entries == []
    assert missing_variables == {}


def test_plain_reference_profiles_do_not_override_scanned_file_paths():
    profiles = _load_builtin_yaml("reference_profiles.yaml")
    path_keys = {"sub_dir", "prefix", "suffix", "fulllist", "data_groupby"}
    offenders = {}

    for profile_name, profile in profiles.items():
        if not isinstance(profile, dict) or profile.get("scan"):
            continue
        for variable_name, variable in (profile.get("variables") or {}).items():
            if not isinstance(variable, dict):
                continue
            present = sorted(path_keys & set(variable))
            if present:
                offenders[f"{profile_name}.{variable_name}"] = present

    assert offenders == {}


def test_builtin_model_catalog_includes_legacy_model_variable_definitions():
    catalog = _load_builtin_yaml("model_catalog.yaml")
    missing_entries = []
    missing_variables = {}

    for legacy_name, legacy_data in _load_legacy_yaml_dir("Mod_variables_definition").items():
        if legacy_name == "empty":
            continue
        if legacy_name not in catalog:
            missing_entries.append(legacy_name)
            continue

        missing = _legacy_variable_names(legacy_data) - _catalog_variable_names(catalog[legacy_name])
        if missing:
            missing_variables[legacy_name] = sorted(missing)

    assert missing_entries == []
    assert missing_variables == {}


def test_reference_loader_skips_reserved_overlay_files(tmp_path, caplog):
    """reference_profiles.yaml belongs to scanner profiles, not reference entries."""
    import logging

    import yaml

    references_dir = tmp_path / "references"
    references_dir.mkdir()
    (references_dir / "reference_profiles.yaml").write_text(
        yaml.dump(
            {
                "DemoProfile": {
                    "variables": {
                        "Runoff": {"varname": "ro", "varunit": "mm day-1"},
                    }
                }
            }
        )
    )

    with caplog.at_level(logging.WARNING):
        RegistryManager(user_dir=tmp_path)

    assert "reference_profiles.yaml" not in caplog.text


def test_user_reference_overlay_partial_variable_update_preserves_existing_fields(tmp_path):
    import yaml

    references_dir = tmp_path / "references"
    references_dir.mkdir()
    (references_dir / "reference_catalog.yaml").write_text(
        yaml.safe_dump(
            {
                "CARE_LowRes": {
                    "variables": {
                        "Surface_Downward_LW_Radiation": {
                            "varunit": "PATCHED_UNIT",
                        }
                    }
                }
            }
        )
    )

    mgr = RegistryManager(user_dir=tmp_path)
    ref = mgr.get_reference("CARE_LowRes")
    mapping = ref.variables["Surface_Downward_LW_Radiation"]

    assert mapping.varname == "LWDR"
    assert mapping.varunit == "PATCHED_UNIT"
    assert mapping.sub_dir == "Heat/Surface_Downward_LW_Radiation/CARE"


def test_user_reference_overlay_empty_new_variable_uses_blank_varname(tmp_path):
    import yaml

    references_dir = tmp_path / "references"
    references_dir.mkdir()
    (references_dir / "reference_catalog.yaml").write_text(
        yaml.safe_dump(
            {
                "CARE_LowRes": {
                    "variables": {
                        "Custom_Diagnostic": {},
                    }
                }
            }
        )
    )

    mgr = RegistryManager(user_dir=tmp_path)
    ref = mgr.get_reference("CARE_LowRes")
    mapping = ref.variables["Custom_Diagnostic"]

    assert mapping.varname == ""
    assert mapping.varunit == ""


def test_user_model_overlay_partial_variable_update_preserves_existing_fields(tmp_path):
    import yaml

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "model_catalog.yaml").write_text(
        yaml.safe_dump(
            {
                "CoLM2024": {
                    "variables": {
                        "Gross_Primary_Productivity": {
                            "varunit": "PATCHED_UNIT",
                        }
                    }
                }
            }
        )
    )

    mgr = RegistryManager(user_dir=tmp_path)
    model = mgr.get_model("CoLM2024")
    mapping = model.variables["Gross_Primary_Productivity"]

    assert mapping.varname == "f_gpp"
    assert mapping.varunit == "PATCHED_UNIT"
    assert mapping.fallbacks[0].varname == "f_assim"


def test_user_model_overlay_empty_new_variable_uses_blank_varname(tmp_path):
    import yaml

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "model_catalog.yaml").write_text(
        yaml.safe_dump(
            {
                "CoLM2024": {
                    "variables": {
                        "Custom_Diagnostic": {},
                    }
                }
            }
        )
    )

    mgr = RegistryManager(user_dir=tmp_path)
    model = mgr.get_model("CoLM2024")
    mapping = model.variables["Custom_Diagnostic"]

    assert mapping.varname == ""
    assert mapping.varunit == ""


def test_user_model_overlay_delete_variables_tombstone_removes_bundled_variable(tmp_path):
    import yaml

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "model_catalog.yaml").write_text(yaml.safe_dump({"CoLM2024": {"_delete_variables": ["Snow_Depth"]}}))

    mgr = RegistryManager(user_dir=tmp_path)
    model = mgr.get_model("CoLM2024")

    assert "Snow_Depth" not in model.variables
    assert "Evapotranspiration" in model.variables


def test_save_model_rejects_case_insensitive_catalog_conflict(tmp_path, monkeypatch):
    import yaml

    catalog_path = tmp_path / "model_catalog.yaml"
    catalog_path.write_text(
        yaml.safe_dump(
            {
                "MyModel": {
                    "name": "MyModel",
                    "description": "demo",
                    "data_type": "grid",
                    "tim_res": "Month",
                    "variables": {"Runoff": {"varname": "ro", "varunit": "mm"}},
                }
            }
        )
    )
    monkeypatch.setattr(
        registry_manager_module,
        "get_writable_model_catalog_path",
        lambda: catalog_path,
    )
    mgr = RegistryManager(user_dir=tmp_path)
    profile = ModelProfile(
        name="mymodel",
        description="conflict",
        data_type="grid",
        tim_res="Month",
        variables={"Runoff": VariableMapping(varname="ro2", varunit="mm")},
    )

    with pytest.raises(ValueError, match="case-insensitive"):
        mgr.save_model("mymodel", profile)


def test_builtin_reference_catalog_has_no_developer_station_list_paths():
    catalog = _load_builtin_yaml("reference_catalog.yaml")
    bad = {
        name: data["fulllist"]
        for name, data in catalog.items()
        if isinstance(data, dict)
        and isinstance(data.get("fulllist"), str)
        and (
            data["fulllist"].startswith("/Volumes/")
            or "/src/openbench/data/registry/station_lists/" in data["fulllist"]
        )
    }

    assert bad == {}


def test_packaged_station_lists_do_not_embed_developer_absolute_paths():
    root = registry_manager_module.REGISTRY_DIR / "station_lists"
    bad = {}
    for path in sorted(root.glob("*.csv")):
        text = path.read_text(encoding="utf-8")
        matches = [needle for needle in ("/Volumes/", "/Users/", "/tera11/") if needle in text]
        if matches:
            bad[path.name] = matches

    assert bad == {}


def test_gleam_open_water_profile_subdir_matches_catalog_variants():
    catalog = _load_builtin_yaml("reference_catalog.yaml")
    profiles = _load_builtin_yaml("reference_profiles.yaml")

    profile_subdir = profiles["GLEAM_v4.2a"]["variables"]["Open_Water_Evaporation"]["sub_dir"]
    catalog_subdirs = {
        variant: catalog[variant]["variables"]["Open_Water_Evaporation"]["sub_dir"]
        for variant in ("GLEAM_v4.2a_LowRes", "GLEAM_v4.2a_MidRes")
    }

    assert catalog_subdirs == {
        "GLEAM_v4.2a_LowRes": profile_subdir,
        "GLEAM_v4.2a_MidRes": profile_subdir,
    }


def test_station_catalog_entries_have_a_station_list_matching_or_filter():
    import openbench.data.custom as custom_package

    catalog = _load_builtin_yaml("reference_catalog.yaml")
    custom_dir = Path(custom_package.__file__).parent
    incomplete = []

    for name, data in catalog.items():
        if not isinstance(data, dict) or data.get("data_type") != "stn":
            continue
        has_station_source = data.get("fulllist") or data.get("station_matching")
        has_custom_filter = (custom_dir / f"{name}_filter.py").exists()
        if not has_station_source and not has_custom_filter:
            incomplete.append(name)

    assert incomplete == []


def test_streamflow_aggregate_entries_use_existing_station_matching_aliases():
    catalog = _load_builtin_yaml("reference_catalog.yaml")
    profiles = _load_builtin_yaml("reference_profiles.yaml")
    expected = {
        "Daily": ("Station/Water/StreamFlow/Daily", "OpenBench_Streamflow_Daily.nc"),
        "Hourly": ("Station/Water/StreamFlow/Hourly", "OpenBench_Streamflow_Hourly_full.nc"),
        "Monthly": ("Station/Water/StreamFlow/Monthly", "OpenBench_Streamflow_Monthly_full.nc"),
    }

    for name, (root_suffix, dataset_file) in expected.items():
        entry = catalog[name]
        assert "fulllist" not in entry
        assert entry["root_dir"] == f"${{OPENBENCH_REF_ROOT}}/{root_suffix}"
        assert entry["station_matching"]["dataset_file"] == dataset_file
        assert profiles[name]["station_matching"]["dataset_file"] == dataset_file


def test_expand_env_path_uses_persisted_reference_root_when_env_unset(
    tmp_path: Path,
    monkeypatch,
    caplog,
):
    import logging

    import yaml

    home = tmp_path / "home"
    settings_dir = home / ".openbench"
    settings_dir.mkdir(parents=True)
    (settings_dir / "settings.yaml").write_text(yaml.safe_dump({"reference_root": str(tmp_path / "Reference")}))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("OPENBENCH_REF_ROOT", raising=False)
    registry_manager_module._UNRESOLVED_ENV_VARS_WARNED.clear()

    with caplog.at_level(logging.WARNING):
        expanded = registry_manager_module._expand_env_path(
            "${OPENBENCH_REF_ROOT}/Grid/LowRes",
            context="Demo.root_dir",
        )

    assert expanded.replace("\\", "/") == (tmp_path / "Reference" / "Grid" / "LowRes").as_posix()
    assert "OPENBENCH_REF_ROOT" not in caplog.text


def test_profile_station_matching_dataset_files_match_catalog():
    catalog = _load_builtin_yaml("reference_catalog.yaml")
    profiles = _load_builtin_yaml("reference_profiles.yaml")
    mismatches = {}

    for name, profile in profiles.items():
        if not isinstance(profile, dict) or name not in catalog:
            continue
        profile_matching = profile.get("station_matching")
        catalog_matching = catalog[name].get("station_matching")
        if profile_matching and catalog_matching:
            profile_file = profile_matching.get("dataset_file")
            catalog_file = catalog_matching.get("dataset_file")
            if profile_file != catalog_file:
                mismatches[name] = (profile_file, catalog_file)

    assert mismatches == {}


def test_era5land_profile_covers_catalog_variables():
    catalog = _load_builtin_yaml("reference_catalog.yaml")
    profiles = _load_builtin_yaml("reference_profiles.yaml")

    catalog_variables = set()
    for variant in ("ERA5LAND_LowRes", "ERA5LAND_MidRes"):
        catalog_variables.update(catalog[variant]["variables"])

    profile_variables = set(profiles["ERA5LAND"]["variables"])
    assert catalog_variables <= profile_variables


def test_list_references():
    mgr = RegistryManager()
    refs = mgr.list_references()
    assert isinstance(refs, list)
    assert len(refs) >= 1
    names = [r.name for r in refs]
    assert "GLEAM_v4.2a_LowRes" in names


def test_get_reference_exact():
    mgr = RegistryManager()
    ref = mgr.get_reference("GLEAM_v4.2a_LowRes")
    assert ref is not None
    assert ref.name == "GLEAM_v4.2a_LowRes"
    assert ref.data_type == "grid"
    assert "Evapotranspiration" in ref.variables
    assert ref.variables["Evapotranspiration"].varname == "E"


def test_fluxnet_plumber2_registry_preserves_legacy_tim_res_and_units():
    mgr = RegistryManager()
    ref = mgr.get_reference("FLUXNET_PLUMBER2")
    assert ref is not None
    assert ref.tim_res == "Hour"
    assert ref.variables["Surface_Upward_LW_Radiation"].varname == "LWup"
    assert ref.variables["Gross_Primary_Productivity"].varunit == "mumolCO2 m-2 s-1"
    assert ref.variables["Net_Ecosystem_Exchange"].varunit == "mumolCO2 m-2 s-1"
    assert "Ecosystem_Respiration" not in ref.variables


def test_plumber2s_registry_preserves_legacy_carbon_flux_units():
    mgr = RegistryManager()
    ref = mgr.get_reference("PLUMBER2S")
    assert ref is not None
    assert ref.variables["Surface_Upward_LW_Radiation"].varname == "LWup"
    assert ref.variables["Gross_Primary_Productivity"].varunit == "mumolCO2 m-2 s-1"
    assert ref.variables["Net_Ecosystem_Exchange"].varunit == "mumolCO2 m-2 s-1"


def test_get_reference_auto_resolve():
    """Base name auto-resolves when sim context is provided."""
    mgr = RegistryManager()
    ref = mgr.get_reference("CARE", sim_tim_res="Month", sim_grid_res=0.25)
    assert ref is not None
    assert ref.name == "CARE_MidRes"


def test_get_reference_exact_variant_name_wins_over_auto_resolve():
    mgr = RegistryManager()
    ref = mgr.get_reference("CARE_LowRes", sim_tim_res="Month", sim_grid_res=0.25)
    assert ref is not None
    assert ref.name == "CARE_LowRes"


def test_get_reference_base_name_no_context():
    """Base name without sim context: returns exact match if exists, None otherwise."""
    mgr = RegistryManager()
    # If standalone entry exists, exact match returns it
    ref = mgr.get_reference("GLEAM_v4.2a")
    if ref is not None:
        assert ref.name == "GLEAM_v4.2a"
    # Non-existent base name with variants but no context → None
    ref2 = mgr.get_reference("TotallyFakeDataset")
    assert ref2 is None


def test_get_reference_base_name_requires_context_when_only_variants_exist():
    mgr = RegistryManager()
    assert mgr.get_reference("CARE") is None

    ref = mgr.get_reference("CARE", sim_tim_res="Month", sim_grid_res=0.25)
    assert ref is not None
    assert ref.name == "CARE_MidRes"


def test_get_reference_auto_resolve_prefers_lower_time_waste_on_spatial_tie():
    mgr = RegistryManager()
    mgr._references = {
        "demo_lowres": ReferenceDataset(
            name="Demo_LowRes",
            description="",
            category="Water",
            data_type="grid",
            tim_res="Month",
            data_groupby="Year",
            timezone=0,
            years=[2000, 2001],
            variables={},
            grid_res=0.25,
        ),
        "demo_midres": ReferenceDataset(
            name="Demo_MidRes",
            description="",
            category="Water",
            data_type="grid",
            tim_res="Day",
            data_groupby="Year",
            timezone=0,
            years=[2000, 2001],
            variables={},
            grid_res=0.25,
        ),
    }

    ref = mgr.get_reference("Demo", sim_tim_res="Month", sim_grid_res=0.25)
    assert ref is not None
    assert ref.name == "Demo_LowRes"


def test_check_scans_later_simulation_fallbacks_while_adapter_stops_at_first_entry(monkeypatch, tmp_path):
    s1 = tmp_path / "s1"
    s2 = tmp_path / "s2"
    s1.mkdir()
    s2.mkdir()
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="fallback", output_dir="/out", years=[2000, 2001]),
        evaluation=EvaluationConfig(variables=["Evapotranspiration"]),
        reference=ReferenceConfig(sources={"Evapotranspiration": "CARE"}),
        simulation={
            "First": SimulationEntry(model="FirstModel", root_dir=str(s1)),
            "Second": SimulationEntry(
                model="SecondModel",
                root_dir=str(s2),
                tim_res="Month",
                grid_res=0.25,
            ),
        },
        comparison=ComparisonConfig(),
    )

    class CheckRegistryManager:
        def get_reference(self, name, sim_tim_res=None, sim_grid_res=None):
            check_calls.append((name, sim_tim_res, sim_grid_res))
            if sim_tim_res is None and sim_grid_res is None:
                return None
            return SimpleNamespace(
                name="CARE_MidRes",
                data_type="grid",
                tim_res=sim_tim_res,
                grid_res=sim_grid_res,
                root_dir=str(tmp_path),
                variables={"Evapotranspiration": SimpleNamespace(varname="E", varunit="mm")},
            )

        def get_resolution_variants(self, name):
            if name == "CARE":
                return {"MidRes": SimpleNamespace(name="CARE_MidRes", data_type="grid", tim_res="Month", grid_res=0.25)}
            return {}

    class AdapterRegistryManager:
        def get_reference(self, name, sim_tim_res=None, sim_grid_res=None):
            adapter_calls.append((name, sim_tim_res, sim_grid_res))
            return None

        def get_resolution_variants(self, name):
            return {}

        def get_model(self, name):
            return None

    check_calls = []
    adapter_calls = []

    check_mgr = CheckRegistryManager()
    adapter_mgr = AdapterRegistryManager()

    monkeypatch.setattr(config_module, "load_config", lambda _path: cfg)
    monkeypatch.setattr(config_module, "ConfigError", Exception)
    # check.py imports get_registry from registry_manager_module at call time
    monkeypatch.setattr(registry_manager_module, "get_registry", lambda: check_mgr)
    monkeypatch.setattr(check_module.click, "secho", lambda *args, **kwargs: None)
    monkeypatch.setattr(check_module.click, "echo", lambda *args, **kwargs: None)

    check_module.check.callback("/tmp/fallback.yaml")

    # adapter uses get_registry() too
    monkeypatch.setattr(registry_manager_module, "get_registry", lambda: adapter_mgr)

    main_nl, ref_nml, sim_nml = build_legacy_namelists(cfg)

    # check.py now always passes sim context (unified with adapter logic)
    assert check_calls == [
        ("CARE", "Month", 0.25),
    ]
    # After unification, adapter uses the same resolver as check — same resolution context
    assert adapter_calls == [("CARE", "Month", 0.25)]
    assert ref_nml["general"]["Evapotranspiration_ref_source"] == "CARE"
    assert sim_nml["Evapotranspiration"]["First_varname"] == "Evapotranspiration"
    assert main_nl["general"]["basename"] == "fallback"


def test_auto_resolve_variant_applies_time_filter_grid_priority_and_secondary_waste_penalty():
    variants = {
        "LowRes": ReferenceDataset(
            name="Demo_LowRes",
            description="",
            category="Water",
            data_type="grid",
            tim_res="Month",
            data_groupby="Year",
            timezone=0,
            years=[2000, 2001],
            variables={},
            grid_res=0.10,
        ),
        "MidRes": ReferenceDataset(
            name="Demo_MidRes",
            description="",
            category="Water",
            data_type="grid",
            tim_res="Day",
            data_groupby="Year",
            timezone=0,
            years=[2000, 2001],
            variables={},
            grid_res=0.50,
        ),
        "HigRes": ReferenceDataset(
            name="Demo_HigRes",
            description="",
            category="Water",
            data_type="grid",
            tim_res="Hour",
            data_groupby="Year",
            timezone=0,
            years=[2000, 2001],
            variables={},
            grid_res=0.25,
        ),
    }

    ref, reason = _auto_resolve_variant(variants, sim_tim_res="Day", sim_grid_res=0.25)
    assert ref is not None
    assert ref.name == "Demo_HigRes"
    assert reason  # should contain decision trace


def test_auto_resolve_variant_handles_climatology_time_resolution_aliases():
    variants = {
        "LowRes": ReferenceDataset(
            name="Demo_LowRes",
            description="",
            category="Water",
            data_type="grid",
            tim_res="Year",
            data_groupby="Year",
            timezone=0,
            years=[2000, 2001],
            variables={},
            grid_res=0.25,
        ),
        "MidRes": ReferenceDataset(
            name="Demo_MidRes",
            description="",
            category="Water",
            data_type="grid",
            tim_res="Month",
            data_groupby="Year",
            timezone=0,
            years=[2000, 2001],
            variables={},
            grid_res=0.25,
        ),
    }

    monthly, _ = _auto_resolve_variant(
        variants,
        sim_tim_res="climatology-month",
        sim_grid_res=0.25,
    )
    yearly, _ = _auto_resolve_variant(
        variants,
        sim_tim_res="climatology-year",
        sim_grid_res=0.25,
    )

    assert monthly.name == "Demo_MidRes"
    assert yearly.name == "Demo_LowRes"


def test_get_reference_not_found():
    mgr = RegistryManager()
    ref = mgr.get_reference("NonExistentDataset")
    assert ref is None


def test_list_models():
    mgr = RegistryManager()
    models = mgr.list_models()
    assert isinstance(models, list)
    assert len(models) >= 1
    names = [m.name for m in models]
    assert "CoLM2024" in names


def test_get_model():
    mgr = RegistryManager()
    model = mgr.get_model("CoLM2024")
    assert model is not None
    assert model.name == "CoLM2024"
    assert "Evapotranspiration" in model.variables
    assert model.variables["Evapotranspiration"].varname == "f_fevpa"


def test_legacy_colm_profile_matches_colm2024_runtime_mapping():
    mgr = RegistryManager()

    legacy = mgr.get_model("CoLM")
    modern = mgr.get_model("CoLM2024")

    assert legacy is not None
    assert modern is not None
    assert legacy.name == "CoLM"
    assert legacy.data_type == modern.data_type
    assert legacy.grid_res == modern.grid_res
    assert legacy.tim_res == modern.tim_res
    assert legacy.time_offset == modern.time_offset
    assert {name: mapping.to_dict() for name, mapping in legacy.variables.items()} == {
        name: mapping.to_dict() for name, mapping in modern.variables.items()
    }
    assert legacy.variables["Canopy_Evaporation"].compute == "ds['f_fevpl'] - ds['f_etr']"


def test_colm2024_routing_prefix_fallback_variables_include_unitcat_varnames():
    model = RegistryManager().get_model("CoLM2024")

    expected = {
        "Dam_Elevation": ["f_sfcelv"],
        "Dam_Storage": ["volresv"],
        "Dam_Water_Elevation": ["f_sfcelv"],
        "Depth_Of_Surface_Water": ["f_flddph"],
        "Inundation_Area": ["f_floodarea"],
        "Inundation_Fraction": ["f_floodfrc"],
        "River_Water_Level": ["f_sfcelv", "f_wdpth_ucat"],
        "Streamflow_Ocean": ["f_discharge_rivermouth"],
        "Total_Water_Storage": ["f_storge"],
    }

    for variable, fallback_names in expected.items():
        mapping = model.variables[variable]
        actual = [mapping.varname] + [fallback.varname for fallback in mapping.fallbacks or []]
        assert all(name in actual for name in fallback_names)
        assert mapping.prefix_fallback == ["_cama_", "_unitcat_"]


def test_registry_manager_normalizes_legacy_list_varnames_into_fallbacks(tmp_path: Path):
    # RegistryManager looks for user models at user_dir/models/model_catalog.yaml
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    catalog = models_dir / "model_catalog.yaml"
    catalog.write_text(
        """
LegacyModel:
  name: LegacyModel
  description: legacy list varname test
  data_type: grid
  grid_res: 0.5
  tim_res: Month
  variables:
    Runoff:
      varname:
        - runoff_primary
        - runoff_fallback
      varunit: mm day-1
""".strip()
    )

    mgr = RegistryManager(user_dir=tmp_path)
    model = mgr.get_model("LegacyModel")

    assert model is not None
    assert model.variables["Runoff"].varname == "runoff_primary"
    assert model.variables["Runoff"].fallbacks is not None
    assert [fb.varname for fb in model.variables["Runoff"].fallbacks] == ["runoff_fallback"]
    assert model.variables["Runoff"].fallbacks[0].varunit == "mm day-1"


def test_clm5_model_preserves_legacy_carbon_flux_mappings():
    mgr = RegistryManager()
    model = mgr.get_model("CLM5")
    assert model is not None
    assert model.variables["Gross_Primary_Productivity"].varunit == "g m-2 s-1"
    assert model.variables["Net_Ecosystem_Exchange"].varname == "NEE"
    assert model.variables["Net_Ecosystem_Exchange"].varunit == "g m-2 s-1"
    assert model.variables["Ecosystem_Respiration"].varname == "ER"
    assert model.variables["Ecosystem_Respiration"].varunit == "g m-2 s-1"


def test_bcc_avim_model_preserves_legacy_carbon_flux_placeholders():
    mgr = RegistryManager()
    model = mgr.get_model("BCC_AVIM")
    assert model is not None
    assert model.variables["Gross_Primary_Productivity"].varname == ""
    assert model.variables["Gross_Primary_Productivity"].varunit == "g m-2 s-1"
    assert model.variables["Ecosystem_Respiration"].varname == ""
    assert model.variables["Ecosystem_Respiration"].varunit == "mol m-2 s-1"


def test_get_model_not_found():
    mgr = RegistryManager()
    model = mgr.get_model("NonExistentModel")
    assert model is None


def test_registry_manager_initializes_last_resolve_reason(tmp_path):
    mgr = RegistryManager(user_dir=tmp_path / "does-not-exist")

    assert mgr.last_resolve_reason == ""


def test_references_for_variable():
    mgr = RegistryManager()
    refs = mgr.references_for_variable("Evapotranspiration")
    assert len(refs) >= 1
    assert any("GLEAM" in r.name for r in refs)


def test_references_for_unknown_variable():
    mgr = RegistryManager()
    refs = mgr.references_for_variable("UnknownVariable")
    assert refs == []


def test_get_resolution_variants():
    mgr = RegistryManager()
    variants = mgr.get_resolution_variants("GLEAM_v4.2a")
    assert isinstance(variants, dict)
    assert "LowRes" in variants


def test_delete_builtin_reference_writes_tombstone_and_survives_reload(tmp_path, monkeypatch):
    """Deleting a built-in-only reference must persist across a manager reload (M1)."""
    # Writable overlay path must match where __init__ merges the user catalog from:
    # <user_dir>/references/reference_catalog.yaml.
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    catalog_path = refs_dir / "reference_catalog.yaml"
    monkeypatch.setattr(
        registry_manager_module,
        "get_writable_reference_catalog_path",
        lambda: catalog_path,
    )

    builtin = _load_builtin_yaml("reference_catalog.yaml")
    target = next(name for name, data in builtin.items() if isinstance(data, dict) and not data.get("_deleted"))

    mgr = RegistryManager(user_dir=tmp_path)
    assert mgr.get_reference(target) is not None

    mgr.delete_reference(target)
    assert mgr.get_reference(target) is None
    # A tombstone must have been written to the user overlay catalog.
    import yaml

    written = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    assert any(isinstance(v, dict) and v.get("_deleted") for v in written.values())

    # A fresh manager (reload) must NOT resurrect the deleted built-in reference.
    reloaded = RegistryManager(user_dir=tmp_path)
    assert reloaded.get_reference(target) is None
