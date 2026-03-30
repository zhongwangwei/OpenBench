"""Tests for RegistryManager."""

from openbench.data.registry.manager import RegistryManager


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


def test_get_reference_auto_resolve():
    """Base name auto-resolves when sim context is provided."""
    mgr = RegistryManager()
    ref = mgr.get_reference("GLEAM_v4.2a", sim_tim_res="Month", sim_grid_res=0.5)
    assert ref is not None
    assert "GLEAM_v4.2a" in ref.name


def test_get_reference_base_name_no_context():
    """Base name without sim context returns None when variants exist."""
    mgr = RegistryManager()
    variants = mgr.get_resolution_variants("GLEAM_v4.2a")
    if variants:
        # Has variants but no context → None
        ref = mgr.get_reference("GLEAM_v4.2a")
        assert ref is None


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


def test_get_model_not_found():
    mgr = RegistryManager()
    model = mgr.get_model("NonExistentModel")
    assert model is None


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
