from __future__ import annotations

import pytest
import yaml

from openbench.data.registry import manager as registry_manager_module
from openbench.data.registry.manager import RegistryManager, _build_model
from openbench.data.registry.schema import ModelProfile, ReferenceDataset, VariableMapping
from openbench.gui.remote_registry import RemoteRegistrySnapshot
from openbench.util.names import canonical_variable_name


@pytest.mark.parametrize(
    ("name", "canonical"),
    [
        ("Transpiration", "Canopy_Transpiration"),
        (" canopy_transpiration ", "Canopy_Transpiration"),
        ("RUNOFF", "Total_Runoff"),
        ("total_runoff", "Total_Runoff"),
        ("CANOPY_INTERCEPTION", "Canopy_Evaporation"),
        (" canopy_evaporation ", "Canopy_Evaporation"),
        ("Surface_Runoff", "Surface_Runoff"),
        ("Subsurface_Runoff", "Subsurface_Runoff"),
        ("Canopy_Evaporation_Canopy_Transpiration", "Canopy_Evaporation_Canopy_Transpiration"),
    ],
)
def test_water_aliases_normalize_only_equivalent_logical_names(name, canonical):
    assert canonical_variable_name(name) == canonical


def _ref(name: str, variables: dict[str, VariableMapping]) -> ReferenceDataset:
    return ReferenceDataset(
        name=name,
        description="test",
        category="Water",
        data_type="grid",
        tim_res="Month",
        data_groupby="Year",
        timezone=0,
        years=[2000, 2001],
        variables=variables,
    )


def _mapping(varname: str = "", unit: str = "") -> VariableMapping:
    return VariableMapping(varname=varname, varunit=unit)


def test_builtin_soil_evaporation_aliases_resolve_to_same_reference_variable(tmp_path):
    registry = RegistryManager(user_dir=tmp_path)

    soil_refs = {ref.name: ref for ref in registry.references_for_variable("Soil_Evaporation")}
    bare_refs = {ref.name: ref for ref in registry.references_for_variable("Bare_Soil_Evaporation")}

    assert soil_refs == bare_refs
    assert soil_refs["GLEAM_v4.2a_LowRes"].variables["Soil_Evaporation"].varname == "Eb"


def test_builtin_model_aliases_keep_native_varnames(tmp_path):
    registry = RegistryManager(user_dir=tmp_path)

    assert registry.get_model("CoLM2024").variables["Soil_Evaporation"].varname == "f_fevpg"
    assert registry.get_model("TE").variables["Soil_Evaporation"].varname == "EBFLX"
    assert "Bare_Soil_Evaporation" not in registry.get_model("TE").variables


@pytest.mark.parametrize(
    ("old", "canonical", "reference", "native"),
    [
        ("Transpiration", "Canopy_Transpiration", "GLEAM_v4.3a_LowRes", "Et"),
        ("Runoff", "Total_Runoff", "GRFR_LowRes", "ro"),
        ("Canopy_Interception", "Canopy_Evaporation", "GLEAM_v4.3a_LowRes", "Ei"),
    ],
)
def test_builtin_water_aliases_share_reference_candidates(tmp_path, old, canonical, reference, native):
    registry = RegistryManager(user_dir=tmp_path)
    old_refs = {ref.name: ref for ref in registry.references_for_variable(old.lower())}
    canonical_refs = {ref.name: ref for ref in registry.references_for_variable(canonical.upper())}

    assert old_refs == canonical_refs
    assert old_refs[reference].variables[canonical].varname == native
    assert all(old not in ref.variables for ref in canonical_refs.values())


def test_model_water_aliases_preserve_total_surface_and_compute_mappings(tmp_path):
    registry = RegistryManager(user_dir=tmp_path)
    colm = registry.get_model("CoLM2024")
    assert colm.variables["Total_Runoff"] == _mapping("f_rnof", "mm s-1")
    assert colm.variables["Surface_Runoff"] == _mapping("f_rsur", "mm s-1")
    assert colm.variables["Canopy_Transpiration"] == _mapping("f_etr", "mm s-1")
    assert "RUNOFF" in registry.get_model("TE").variables["Total_Runoff"].compute
    assert "isel(lev=0)" in registry.get_model("TE").variables["Total_Runoff"].compute
    assert registry.get_model("GLDAS").variables["Total_Runoff"].compute == "ds['Qs_acc'] + ds['Qsb_acc']"
    for name in ("CLM5", "ELM", "E3SM", "CoLM2024", "GLDAS", "TE"):
        variables = registry.get_model(name).variables
        assert "Runoff" not in variables
        assert "Transpiration" not in variables


@pytest.mark.parametrize("name", ["CLM5", "ELM", "E3SM"])
def test_clm_family_total_runoff_keeps_native_rate_units(name):
    catalog = yaml.safe_load((registry_manager_module.REGISTRY_DIR / "model_catalog.yaml").read_text())
    model = _build_model(catalog[name])

    assert model.variables["Total_Runoff"] == _mapping("QRUNOFF", "mm s-1")
    assert model.variables["Surface_Runoff"] == _mapping("QOVER", "mm s-1")
    assert model.variables["Subsurface_Runoff"] == _mapping("QDRAI", "mm s-1")


def test_gldas_total_runoff_does_not_count_snowmelt_as_an_extra_runoff_component():
    import xarray as xr

    from openbench.data.compute import execute_compute

    catalog = yaml.safe_load((registry_manager_module.REGISTRY_DIR / "model_catalog.yaml").read_text())
    mapping = _build_model(catalog["GLDAS"]).variables["Total_Runoff"]
    source = xr.Dataset({"Qs_acc": ("time", [1.0]), "Qsb_acc": ("time", [2.0]), "Qsm_acc": ("time", [100.0])})

    result = execute_compute(source, mapping.compute, "Total_Runoff")

    assert result.item() == 3.0
    assert mapping.varunit == "mm 3hour-1"


def test_unknown_variable_names_keep_prior_case_sensitive_behavior():
    model = _build_model(
        {
            "name": "demo",
            "variables": {
                "Custom_Variable": {"varname": "A", "varunit": "1"},
                "custom_variable": {"varname": "B", "varunit": "1"},
            },
        }
    )

    assert set(model.variables) == {"Custom_Variable", "custom_variable"}


@pytest.mark.parametrize(
    ("old", "canonical"),
    [
        ("Bare_Soil_Evaporation", "Soil_Evaporation"),
        ("Transpiration", "Canopy_Transpiration"),
        ("Runoff", "Total_Runoff"),
        ("Canopy_Interception", "Canopy_Evaporation"),
    ],
)
def test_alias_build_prefers_populated_mapping_and_rejects_populated_conflict(old, canonical):
    model = _build_model(
        {
            "name": "demo",
            "variables": {
                canonical: {"varname": "", "varunit": ""},
                old: {"varname": "EBFLX", "varunit": "kg m-2 s-1"},
            },
        }
    )
    assert model.variables == {canonical: _mapping("EBFLX", "kg m-2 s-1")}

    try:
        _build_model(
            {
                "name": "bad",
                "variables": {
                    canonical: {"varname": "A", "varunit": "mm day-1"},
                    old: {"varname": "B", "varunit": "mm day-1"},
                },
            }
        )
    except ValueError as exc:
        assert "Conflicting variable alias definitions" in str(exc)
    else:  # pragma: no cover - keeps assertion message clearer than pytest.raises here
        raise AssertionError("conflicting populated aliases should fail")


@pytest.mark.parametrize(
    ("old", "canonical"),
    [
        ("Bare_Soil_Evaporation", "Soil_Evaporation"),
        ("Transpiration", "Canopy_Transpiration"),
        ("Runoff", "Total_Runoff"),
        ("Canopy_Interception", "Canopy_Evaporation"),
    ],
)
def test_overlay_delete_uses_alias_name(tmp_path, old, canonical):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "model_catalog.yaml").write_text(
        yaml.safe_dump({"TE": {"name": "TE", "_delete_variables": [old]}}),
        encoding="utf-8",
    )

    model = RegistryManager(user_dir=tmp_path).get_model("TE")

    assert canonical not in model.variables
    assert old not in model.variables


@pytest.mark.parametrize(
    ("old", "canonical"),
    [
        ("Bare_Soil_Evaporation", "Soil_Evaporation"),
        ("Transpiration", "Canopy_Transpiration"),
        ("Runoff", "Total_Runoff"),
        ("Canopy_Interception", "Canopy_Evaporation"),
    ],
)
def test_save_canonicalizes_aliases_in_memory_and_catalog(tmp_path, monkeypatch, old, canonical):
    monkeypatch.setattr(registry_manager_module, "get_user_config_dir", lambda: tmp_path)
    registry = RegistryManager(user_dir=tmp_path)

    registry.save_reference("AliasRef", _ref("AliasRef", {old: _mapping("Eb", "mm day-1")}))
    registry.save_model(
        "AliasModel",
        ModelProfile(name="AliasModel", description="test", variables={old: _mapping("EBFLX")}),
    )

    assert registry.get_reference("AliasRef").variables == {canonical: _mapping("Eb", "mm day-1")}
    assert registry.get_model("AliasModel").variables == {canonical: _mapping("EBFLX")}
    ref_catalog = yaml.safe_load((tmp_path / "references" / "reference_catalog.yaml").read_text())
    model_catalog = yaml.safe_load((tmp_path / "models" / "model_catalog.yaml").read_text())
    assert canonical in ref_catalog["AliasRef"]["variables"]
    assert canonical in model_catalog["AliasModel"]["variables"]


@pytest.mark.parametrize(
    ("old", "canonical"),
    [
        ("Bare_Soil_Evaporation", "Soil_Evaporation"),
        ("Transpiration", "Canopy_Transpiration"),
        ("Runoff", "Total_Runoff"),
        ("Canopy_Interception", "Canopy_Evaporation"),
    ],
)
def test_remote_snapshot_canonicalizes_aliases(monkeypatch, old, canonical):
    payload = {
        "references": [
            _ref("RemoteRef", {old: _mapping("Eb", "mm day-1")}).to_dict(),
        ],
        "models": [
            ModelProfile(
                name="RemoteModel",
                description="test",
                variables={old: _mapping("EBFLX")},
            ).to_dict(),
        ],
    }
    snapshot = RemoteRegistrySnapshot(object(), ("host",), payload)
    monkeypatch.setattr(snapshot, "_ensure_current_target", lambda: None)

    assert snapshot.references_for_variable(old)[0].variables == {canonical: _mapping("Eb", "mm day-1")}
    assert snapshot.references_for_variable(canonical)[0].name == "RemoteRef"
    assert snapshot.get_model("RemoteModel").variables == {canonical: _mapping("EBFLX")}


def test_remote_snapshot_rejects_conflicting_total_and_surface_runoff_aliases():
    payload = {
        "models": [
            ModelProfile(
                name="RemoteModel",
                description="test",
                variables={"Runoff": _mapping("surface"), "Total_Runoff": _mapping("total")},
            ).to_dict()
        ]
    }
    with pytest.raises(ValueError, match="Conflicting variable alias definitions"):
        RemoteRegistrySnapshot(object(), ("host",), payload)


def test_builtin_canopy_evaporation_keeps_evaporation_not_intercepted_precipitation():
    # CLM5 WaterfluxType.F90:399-425 at fdfc03a0abec838ec0b66663a6a2f5e619cd2611
    # binds QINTR to qflx_prec_intr_patch, QVEGE to qflx_evap_can_patch (both mm/s).
    catalog = yaml.safe_load((registry_manager_module.REGISTRY_DIR / "model_catalog.yaml").read_text())
    models = {name: _build_model(raw) for name, raw in catalog.items()}
    for name, model in models.items():
        assert "Canopy_Interception" not in model.variables, name
    for name in ("CLM5", "ELM", "E3SM"):
        assert models[name].variables["Canopy_Evaporation"] == _mapping("QVEGE", "mm s-1")
        assert "Canopy_Evaporation_Canopy_Transpiration" in models[name].variables
    colm = models["CoLM2024"].variables["Canopy_Evaporation"]
    assert colm.compute == "ds['f_fevpl'] - ds['f_etr']"
    assert colm.varname == "Canopy_Evaporation"
    assert colm.varunit == "mm s-1"


def test_custom_canopy_interception_precipitation_conflict_is_not_silently_dropped():
    with pytest.raises(ValueError, match="Conflicting variable alias definitions"):
        _build_model(
            {
                "name": "Custom",
                "variables": {
                    "Canopy_Interception": {"varname": "QINTR", "varunit": "mm s-1"},
                    "Canopy_Evaporation": {"varname": "QVEGE", "varunit": "mm s-1"},
                },
            }
        )
