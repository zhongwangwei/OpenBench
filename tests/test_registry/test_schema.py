"""Tests for registry schema dataclasses."""

from openbench.data.registry.schema import ModelProfile, ReferenceDataset, StationMatchingConfig, VariableMapping


def test_variable_mapping():
    v = VariableMapping(varname="E", varunit="mm day-1", prefix="E_", suffix="_GLEAM")
    assert v.varname == "E"
    assert v.sub_dir is None


def test_variable_mapping_with_subdir():
    v = VariableMapping(
        varname="E",
        varunit="mm day-1",
        prefix="E_",
        suffix="_GLEAM",
        sub_dir="Evapotranspiration/GLEAM_v4.2a",
    )
    assert v.sub_dir == "Evapotranspiration/GLEAM_v4.2a"


def test_reference_dataset():
    ref = ReferenceDataset(
        name="GLEAM_v4.2a",
        description="Global Land Evaporation Amsterdam Model v4.2a",
        category="Water",
        data_type="grid",
        grid_res=0.25,
        tim_res="Month",
        data_groupby="Year",
        timezone=0,
        years=[1980, 2023],
        variables={
            "Evapotranspiration": VariableMapping(
                varname="E",
                varunit="mm day-1",
                prefix="E_",
                suffix="_GLEAM_v4.2a",
                sub_dir="Evapotranspiration/GLEAM_v4.2a",
            ),
        },
    )
    assert ref.name == "GLEAM_v4.2a"
    assert ref.data_type == "grid"
    assert "Evapotranspiration" in ref.variables
    assert ref.fulllist is None


def test_reference_dataset_station():
    ref = ReferenceDataset(
        name="GRDC_Monthly",
        description="GRDC Monthly Streamflow",
        category="Water",
        data_type="stn",
        tim_res="Month",
        data_groupby="single",
        timezone=0,
        years=[1950, 2023],
        variables={
            "Streamflow": VariableMapping(varname="streamflow", varunit="m3 s-1"),
        },
        fulllist="GRDC_Monthly.csv",
    )
    assert ref.data_type == "stn"
    assert ref.fulllist == "GRDC_Monthly.csv"
    assert ref.grid_res is None


def test_reference_dataset_to_dict_preserves_station_matching_and_provenance():
    ref = ReferenceDataset(
        name="GRDC_Monthly",
        description="GRDC Monthly Streamflow",
        category="Water",
        data_type="stn",
        tim_res="Month",
        data_groupby="single",
        timezone=0,
        years=[1950, 2023],
        variables={"Streamflow": VariableMapping(varname="streamflow", varunit="m3 s-1")},
        station_matching=StationMatchingConfig(method="direct", dataset_file="stations.nc"),
        _provenance={"station_matching": "profile"},
    )

    data = ref.to_dict()

    assert data["station_matching"]["method"] == "direct"
    assert data["station_matching"]["dataset_file"] == "stations.nc"
    assert data["_provenance"] == {"station_matching": "profile"}


def test_model_profile():
    m = ModelProfile(
        name="CoLM2024",
        description="Common Land Model 2024",
        data_type="grid",
        grid_res=0.5,
        tim_res="Month",
        variables={
            "Evapotranspiration": VariableMapping(varname="ET", varunit="mm day-1"),
            "Latent_Heat": VariableMapping(varname="Qle", varunit="W m-2"),
        },
    )
    assert m.name == "CoLM2024"
    assert len(m.variables) == 2
    assert m.variables["Evapotranspiration"].varname == "ET"
