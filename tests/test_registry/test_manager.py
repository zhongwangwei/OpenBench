"""Tests for RegistryManager."""

from openbench.data.registry.manager import RegistryManager, _auto_resolve_variant
from openbench.data.registry.schema import ReferenceDataset


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

    ref = _auto_resolve_variant(variants, sim_tim_res="Day", sim_grid_res=0.25)
    assert ref is not None
    assert ref.name == "Demo_HigRes"


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
