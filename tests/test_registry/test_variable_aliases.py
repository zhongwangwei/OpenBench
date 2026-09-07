from __future__ import annotations

import yaml

from openbench.data.registry import manager as registry_manager_module
from openbench.data.registry.manager import RegistryManager, _build_model
from openbench.data.registry.schema import ModelProfile, ReferenceDataset, VariableMapping
from openbench.gui.remote_registry import RemoteRegistrySnapshot


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


def test_alias_build_prefers_populated_mapping_and_rejects_populated_conflict():
    model = _build_model(
        {
            "name": "demo",
            "variables": {
                "Soil_Evaporation": {"varname": "", "varunit": ""},
                "Bare_Soil_Evaporation": {"varname": "EBFLX", "varunit": "kg m-2 s-1"},
            },
        }
    )
    assert model.variables == {"Soil_Evaporation": _mapping("EBFLX", "kg m-2 s-1")}

    try:
        _build_model(
            {
                "name": "bad",
                "variables": {
                    "Soil_Evaporation": {"varname": "A", "varunit": "mm day-1"},
                    "Bare_Soil_Evaporation": {"varname": "B", "varunit": "mm day-1"},
                },
            }
        )
    except ValueError as exc:
        assert "Conflicting variable alias definitions" in str(exc)
    else:  # pragma: no cover - keeps assertion message clearer than pytest.raises here
        raise AssertionError("conflicting populated aliases should fail")


def test_overlay_delete_uses_alias_name(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "model_catalog.yaml").write_text(
        yaml.safe_dump({"TE": {"name": "TE", "_delete_variables": ["Bare_Soil_Evaporation"]}}),
        encoding="utf-8",
    )

    model = RegistryManager(user_dir=tmp_path).get_model("TE")

    assert "Soil_Evaporation" not in model.variables
    assert "Bare_Soil_Evaporation" not in model.variables


def test_save_canonicalizes_aliases_in_memory_and_catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(registry_manager_module, "get_user_config_dir", lambda: tmp_path)
    registry = RegistryManager(user_dir=tmp_path)

    registry.save_reference("AliasRef", _ref("AliasRef", {"Bare_Soil_Evaporation": _mapping("Eb", "mm day-1")}))
    registry.save_model(
        "AliasModel",
        ModelProfile(name="AliasModel", description="test", variables={"Bare_Soil_Evaporation": _mapping("EBFLX")}),
    )

    assert registry.get_reference("AliasRef").variables == {"Soil_Evaporation": _mapping("Eb", "mm day-1")}
    assert registry.get_model("AliasModel").variables == {"Soil_Evaporation": _mapping("EBFLX")}
    ref_catalog = yaml.safe_load((tmp_path / "references" / "reference_catalog.yaml").read_text())
    model_catalog = yaml.safe_load((tmp_path / "models" / "model_catalog.yaml").read_text())
    assert "Soil_Evaporation" in ref_catalog["AliasRef"]["variables"]
    assert "Soil_Evaporation" in model_catalog["AliasModel"]["variables"]


def test_remote_snapshot_canonicalizes_aliases(monkeypatch):
    payload = {
        "references": [
            _ref("RemoteRef", {"Bare_Soil_Evaporation": _mapping("Eb", "mm day-1")}).to_dict(),
        ],
        "models": [
            ModelProfile(
                name="RemoteModel",
                description="test",
                variables={"Bare_Soil_Evaporation": _mapping("EBFLX")},
            ).to_dict(),
        ],
    }
    snapshot = RemoteRegistrySnapshot(object(), ("host",), payload)
    monkeypatch.setattr(snapshot, "_ensure_current_target", lambda: None)

    assert snapshot.references_for_variable("Bare_Soil_Evaporation")[0].variables == {
        "Soil_Evaporation": _mapping("Eb", "mm day-1")
    }
    assert snapshot.references_for_variable("Soil_Evaporation")[0].name == "RemoteRef"
    assert snapshot.get_model("RemoteModel").variables == {"Soil_Evaporation": _mapping("EBFLX")}
