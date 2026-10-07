"""Test legacy namelist building from new config."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import openbench.config.adapter as adapter_module
from openbench.config.adapter import build_legacy_namelists
from openbench.config.loader import load_config
from openbench.config.runtime_info import GeneralInfoReader
from openbench.config.schema import (
    EvaluationConfig,
    OpenBenchConfig,
    ProjectConfig,
    ReferenceConfig,
    SimulationEntry,
)


def test_non_streamflow_reference_ignores_stale_station_matching(monkeypatch, caplog):
    import logging

    import openbench.data.custom as custom_module
    import openbench.data.registry.manager as registry_manager_module

    reference = SimpleNamespace(
        station_matching=SimpleNamespace(dataset_file="flux.nc"),
        variables={"Latent_Heat": SimpleNamespace()},
    )
    registry = SimpleNamespace(get_reference=lambda _name: reference)
    monkeypatch.setattr(registry_manager_module, "get_registry", lambda: registry)
    monkeypatch.setattr(custom_module, "load_filter", lambda _name: None)

    reader = GeneralInfoReader.__new__(GeneralInfoReader)
    reader.ref_source = "OpenBench_FLUX_Daily"
    reader._custom_filter_warnings_shown = set()

    with caplog.at_level(logging.WARNING):
        assert reader._get_custom_filter() is None

    assert "Ignoring station_matching for reference OpenBench_FLUX_Daily" in caplog.text


def _write_streamflow_stations(path, areas):
    import numpy as np
    import pandas as pd
    import xarray as xr

    n = len(areas)
    xr.Dataset(
        {
            "station": ("station", np.arange(n)),
            "lon": ("station", np.full(n, 10.0)),
            "lat": ("station", np.full(n, 20.0)),
            "area": ("station", np.asarray(areas)),
            "cama_lon_15min": ("station", np.full(n, 10.0)),
            "cama_lat_15min": ("station", np.full(n, 20.0)),
            "cama_alloc_err_15min": ("station", np.zeros(n)),
            "discharge": (("station", "time"), np.ones((n, 2))),
        },
        coords={"time": pd.date_range("2000-01-01", periods=2, freq="D")},
    ).to_netcdf(path)


def _streamflow_matching_reader(monkeypatch, tmp_path, dataset_file, sim_grid_res):
    import openbench.data.registry.manager as registry_manager_module
    from openbench.data.registry.schema import StationMatchingConfig

    reference = SimpleNamespace(
        station_matching=StationMatchingConfig(dataset_file=dataset_file),
        variables={"Streamflow": SimpleNamespace()},
    )
    registry = SimpleNamespace(get_reference=lambda _name: reference)
    monkeypatch.setattr(registry_manager_module, "get_registry", lambda: registry)

    reader = GeneralInfoReader.__new__(GeneralInfoReader)
    reader.__dict__.update(
        item="Streamflow",
        ref_source="RiverStn",
        sim_source="SimA",
        ref_dir=str(tmp_path),
        casedir=str(tmp_path / "case"),
        sim_grid_res=sim_grid_res,
        ref_data_type="stn",
        sim_data_type="grid",
        syear=2000,
        eyear=2000,
        sim_syear=2000,
        sim_eyear=2000,
        min_year=0,
        min_lon=-180,
        max_lon=180,
        min_lat=-90,
        max_lat=90,
        _custom_filter_warnings_shown=set(),
    )
    return reader


@pytest.mark.parametrize(
    ("sim_grid_res", "areas", "error"),
    [
        (0.5, [5000.0], "no minimum upstream area"),
        (0.25, [1000.0], "No stations passed"),
    ],
)
def test_station_matching_failure_propagates_instead_of_using_loaded_station_list(
    monkeypatch, tmp_path, sim_grid_res, areas, error
):
    """A failed built-in match must not fall back to stations it did not vet."""
    import pandas as pd

    _write_streamflow_stations(tmp_path / "stations.nc", areas)
    reader = _streamflow_matching_reader(monkeypatch, tmp_path, "stations.nc", sim_grid_res)
    # A station list already read from a fulllist that the matcher never vetted.
    reader.stn_list = pd.DataFrame(
        {"ID": ["OLD"], "lon": [10.0], "lat": [20.0], "ref_syear": [2000], "ref_eyear": [2000]}
    )

    with pytest.raises(ValueError, match=error):
        reader._filter_stations()


def test_station_matching_uses_dist_subset_when_full_dataset_is_missing(monkeypatch, tmp_path):
    import pandas as pd

    _write_streamflow_stations(tmp_path / "Flow_Daily_dist.nc", [5000.0])
    reader = _streamflow_matching_reader(monkeypatch, tmp_path, "Flow_Daily_full.nc", 0.25)
    reader.stn_list = pd.DataFrame()

    reader._filter_stations()

    assert reader.stn_list["ID"].tolist() == ["0"]


def test_station_matching_names_both_dataset_files_when_neither_exists(monkeypatch, tmp_path):
    import pandas as pd

    reader = _streamflow_matching_reader(monkeypatch, tmp_path, "Flow_Daily_full.nc", 0.25)
    reader.stn_list = pd.DataFrame()

    with pytest.raises(FileNotFoundError, match="Flow_Daily_full.nc.*Flow_Daily_dist.nc"):
        reader._filter_stations()


@pytest.mark.parametrize(
    ("item", "has_filter", "reports_error"),
    [
        ("Streamflow", False, False),
        ("Latent_Heat", True, False),
        ("Latent_Heat", False, True),
    ],
)
def test_missing_reference_fulllist_reports_only_when_runtime_cannot_generate_it(
    monkeypatch, caplog, item, has_filter, reports_error
):
    """Streamflow and custom filters handle missing lists without a diagnostic."""
    import logging

    filter_checks = []

    def fake_get_custom_filter(_reader):
        filter_checks.append(True)
        return (lambda _info: None) if has_filter else None

    monkeypatch.setattr(GeneralInfoReader, "_get_custom_filter", fake_get_custom_filter)

    reader = GeneralInfoReader.__new__(GeneralInfoReader)
    reader.item = item
    reader.ref_source = "RefA"
    reader._custom_filter_warnings_shown = set()
    ref_nml = {"general": {}, item: {"RefA_data_type": "stn"}}
    sim_nml = {"general": {}, item: {"SimA_data_type": "grid"}}

    with caplog.at_level(logging.DEBUG):
        reader._set_source_attributes(sim_nml, ref_nml, item, "SimA", "RefA")

    records = [record for record in caplog.records if "ref_fulllist namelist" in record.message]
    assert len(records) == int(reports_error)
    assert len(filter_checks) == int(item.casefold() != "streamflow")
    if reports_error:
        assert records[0].levelname == "ERROR"
        assert "namelist error" in records[0].message


def test_adapter_declares_legacy_root_section_contract():
    """The adapter should publish the legacy top-level sections it still emits."""
    assert adapter_module.LEGACY_ROOT_SECTION_KEYS == {
        "general",
        "evaluation_items",
        "metrics",
        "scores",
        "comparisons",
        "statistics",
    }


def test_build_legacy_namelists():
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="test", output_dir="./output", years=[2004, 2010]),
        evaluation=EvaluationConfig(variables=["Evapotranspiration"]),
        reference=ReferenceConfig(sources={"Evapotranspiration": "GLEAM_v4.2a_LowRes"}),
        simulation={"CoLM2024": SimulationEntry(model="CoLM2024", root_dir="/data/CoLM2024")},
    )

    main_nl, ref_nml, sim_nml = build_legacy_namelists(cfg)

    # Check main_nl
    assert main_nl["general"]["basename"] == "test"
    assert "Evapotranspiration" in main_nl["evaluation_items"]

    # Check ref_nml
    assert ref_nml["general"]["Evapotranspiration_ref_source"] == "GLEAM_v4.2a_LowRes"
    assert "Evapotranspiration" in ref_nml
    assert ref_nml["Evapotranspiration"]["GLEAM_v4.2a_LowRes_varname"] == "E"
    assert ref_nml["Evapotranspiration"]["GLEAM_v4.2a_LowRes_data_type"] == "grid"
    assert ref_nml["Evapotranspiration"]["GLEAM_v4.2a_LowRes_varunit"] == "mm day-1"
    assert ref_nml["Evapotranspiration"]["GLEAM_v4.2a_LowRes_prefix"] == "E_"
    assert ref_nml["Evapotranspiration"]["GLEAM_v4.2a_LowRes_suffix"] == "_GLEAM_v4.2a"
    assert ref_nml["Evapotranspiration"]["GLEAM_v4.2a_LowRes_tim_res"] == "Month"
    assert ref_nml["Evapotranspiration"]["GLEAM_v4.2a_LowRes_grid_res"] in (0.25, 0.5)
    # years come from registry (may be auto-detected or config fallback)
    assert ref_nml["Evapotranspiration"]["GLEAM_v4.2a_LowRes_syear"] >= 1980
    assert ref_nml["Evapotranspiration"]["GLEAM_v4.2a_LowRes_eyear"] >= 2005
    assert ref_nml["Evapotranspiration"]["GLEAM_v4.2a_LowRes_timezone"] == 0

    # Check sim_nml
    assert "Evapotranspiration_sim_source" in sim_nml["general"]
    assert "CoLM2024" in sim_nml["general"]["Evapotranspiration_sim_source"]
    assert "Evapotranspiration" in sim_nml
    assert sim_nml["Evapotranspiration"]["CoLM2024_varname"] == "f_fevpa"
    assert sim_nml["Evapotranspiration"]["CoLM2024_varunit"] in ("mm day-1", "mm s-1")
    assert sim_nml["Evapotranspiration"]["CoLM2024_data_type"] == "grid"
    assert sim_nml["Evapotranspiration"]["CoLM2024_grid_res"] == 0.5
    assert sim_nml["Evapotranspiration"]["CoLM2024_dir"] == "/data/CoLM2024"


def test_build_namelists_multiple_simulations():
    """Multiple sim entries produce a list of sim sources per variable."""
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="multi", output_dir="/out", years=[2000, 2020], tim_res="Month", grid_res=0.5),
        evaluation=EvaluationConfig(variables=["Evapotranspiration"]),
        reference=ReferenceConfig(sources={"Evapotranspiration": "GLEAM_v4.2a_LowRes"}),
        simulation={
            "CoLM2024": SimulationEntry(model="CoLM2024", root_dir="/d1"),
            "CLM5": SimulationEntry(model="CLM5", root_dir="/d2"),
        },
    )

    main_nl, ref_nml, sim_nml = build_legacy_namelists(cfg)

    sources = sim_nml["general"]["Evapotranspiration_sim_source"]
    assert "CoLM2024" in sources
    assert "CLM5" in sources
    assert len(sources) == 2


def test_general_info_reader_fallback_varname_does_not_mutate_shared_namelists():
    main_nl = {
        "general": {
            "basename": "case",
            "basedir": "/tmp",
            "syear": 2000,
            "eyear": 2001,
            "compare_tim_res": "Month",
            "compare_grid_res": 1.0,
        }
    }
    sim_nml = {
        "Runoff": {
            "SimA_varname": "",
            "SimA_data_type": "grid",
            "SimA_varunit": "m3 s-1",
            "SimA_data_groupby": "Year",
            "SimA_dir": "/sim",
            "SimA_tim_res": "Month",
            "SimA_grid_res": 1.0,
            "SimA_syear": 2000,
            "SimA_eyear": 2001,
        }
    }
    ref_nml = {
        "Runoff": {
            "RefA_varname": "",
            "RefA_data_type": "grid",
            "RefA_varunit": "m3 s-1",
            "RefA_data_groupby": "Year",
            "RefA_dir": "/ref",
            "RefA_tim_res": "Month",
            "RefA_grid_res": 1.0,
            "RefA_syear": 2000,
            "RefA_eyear": 2001,
        }
    }

    reader = GeneralInfoReader(
        main_nl=main_nl,
        sim_nml=sim_nml,
        ref_nml=ref_nml,
        metric_vars=[],
        score_vars=[],
        comparison_vars=[],
        statistic_vars=[],
        item="Runoff",
        sim_source="SimA",
        ref_source="RefA",
    )

    assert reader.sim_nml["Runoff"]["SimA_varname"] == "Runoff"
    assert reader.ref_nml["Runoff"]["RefA_varname"] == "Runoff"
    assert sim_nml["Runoff"]["SimA_varname"] == ""
    assert ref_nml["Runoff"]["RefA_varname"] == ""


def test_build_namelists_inline_override():
    """Inline variable overrides take precedence over model profile."""
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="override", output_dir="/out", years=[2005, 2015]),
        evaluation=EvaluationConfig(variables=["Evapotranspiration"]),
        reference=ReferenceConfig(sources={"Evapotranspiration": "GLEAM_v4.2a_LowRes"}),
        simulation={
            "MyModel": SimulationEntry(
                model="CoLM2024",
                root_dir="/custom",
                variables={
                    "Evapotranspiration": {
                        "varname": "custom_et",
                        "varunit": "kg m-2 s-1",
                    }
                },
            ),
        },
    )

    _, _, sim_nml = build_legacy_namelists(cfg)

    assert sim_nml["Evapotranspiration"]["MyModel_varname"] == "custom_et"
    assert sim_nml["Evapotranspiration"]["MyModel_varunit"] == "kg m-2 s-1"
    # dir still comes from root_dir
    assert sim_nml["Evapotranspiration"]["MyModel_dir"] == "/custom"


def test_build_namelists_inline_variable_metadata_overrides_entry_metadata():
    """Variable-level scan overrides must reach the runtime lookup metadata."""
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="override-meta", output_dir="/out", years=[2005, 2015]),
        evaluation=EvaluationConfig(variables=["Streamflow"]),
        reference=ReferenceConfig(sources={"Streamflow": "GRDC"}),
        simulation={
            "TE": SimulationEntry(
                model="TE",
                root_dir="/custom",
                data_groupby="Year",
                grid_res=0.5,
                tim_res="Month",
                prefix="YEE2_JRA-55_CDSTM_M",
                suffix="_GLB050",
                variables={
                    "Streamflow": {
                        "prefix": "YEE2_JRA-55_outflw_M",
                        "suffix": "_GLB025",
                        "grid_res": 0.25,
                        "tim_res": "3month",
                        "data_groupby": "Year",
                    }
                },
            ),
        },
    )

    _, _, sim_nml = build_legacy_namelists(cfg)

    section = sim_nml["Streamflow"]
    assert section["TE_varname"] == "outflw"
    assert section["TE_prefix"] == "YEE2_JRA-55_outflw_M"
    assert section["TE_suffix"] == "_GLB025"
    assert section["TE_grid_res"] == 0.25
    assert section["TE_tim_res"] == "3month"
    assert section["TE_data_groupby"] == "Year"


def test_build_namelists_legacy_colm_keeps_computed_canopy_interception_mapping():
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="legacy-colm", output_dir="/out", years=[2000, 2020]),
        evaluation=EvaluationConfig(variables=["Canopy_Evaporation"]),
        reference=ReferenceConfig(sources={"Canopy_Evaporation": "GLEAM_v4.2a_LowRes"}),
        simulation={
            "LegacyCoLM": SimulationEntry(
                model="CoLM",
                root_dir="/data/colm",
            ),
        },
    )

    _, _, sim_nml = build_legacy_namelists(cfg)

    section = sim_nml["Canopy_Evaporation"]
    assert section["LegacyCoLM_model"] == "CoLM"
    assert section["LegacyCoLM_varname"] == "Canopy_Evaporation"


def test_build_namelists_colm2024_streamflow_passes_prefix_fallbacks():
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="colm-routing", output_dir="/out", years=[2000, 2020]),
        evaluation=EvaluationConfig(variables=["Streamflow"]),
        reference=ReferenceConfig(sources={"Streamflow": "GRDC"}),
        simulation={
            "CaseA": SimulationEntry(
                model="CoLM2024",
                root_dir="/data/colm",
                prefix="Mediterranean_hist_",
                data_groupby="Month",
            ),
        },
    )

    _, _, sim_nml = build_legacy_namelists(cfg)

    section = sim_nml["Streamflow"]
    assert section["CaseA_varname"] == "outflw"
    assert section["CaseA_prefix"] == "Mediterranean_hist_"
    assert section["CaseA_prefix_fallback"] == ["_cama_", "_unitcat_"]


def test_build_namelists_unknown_model_with_inline():
    """An unknown model name still works if inline variables are provided."""
    cfg = OpenBenchConfig(
        project=ProjectConfig(name="unknown", output_dir="/out", years=[2000, 2010]),
        evaluation=EvaluationConfig(variables=["Evapotranspiration"]),
        reference=ReferenceConfig(sources={"Evapotranspiration": "GLEAM_v4.2a_LowRes"}),
        simulation={
            "Custom": SimulationEntry(
                model="UnknownModel",
                root_dir="/x",
                variables={
                    "Evapotranspiration": {"varname": "et", "varunit": "mm/d"},
                },
            ),
        },
    )

    _, _, sim_nml = build_legacy_namelists(cfg)

    assert sim_nml["Evapotranspiration"]["Custom_varname"] == "et"
    assert sim_nml["Evapotranspiration"]["Custom_varunit"] == "mm/d"


def test_reference_fulllist_relative_to_root_dir_is_resolved(monkeypatch, tmp_path):
    """Registry fulllist values may be root_dir-relative; legacy processing needs an existing path."""
    import openbench.data.registry.manager as manager_module
    from openbench.data.registry.schema import ReferenceDataset, VariableMapping

    ref = ReferenceDataset(
        name="DemoStn",
        description="Demo station",
        category="Water",
        data_type="stn",
        tim_res="Day",
        data_groupby="single",
        timezone=0,
        years=[2000, 2001],
        root_dir=str(tmp_path),
        variables={
            "Runoff": VariableMapping(varname="Q", varunit="m3 s-1"),
        },
        fulllist="list/subset.csv",
    )

    class FakeRegistry:
        last_resolve_reason = ""

        def get_reference(self, name, sim_tim_res=None, sim_grid_res=None):
            return ref if name == "DemoStn" else None

        def get_resolution_variants(self, name):
            return {}

        def get_model(self, name):
            return None

    monkeypatch.setattr(manager_module, "get_registry", lambda: FakeRegistry())

    cfg = OpenBenchConfig(
        project=ProjectConfig(name="fulllist", output_dir="/out", years=[2000, 2001]),
        evaluation=EvaluationConfig(variables=["Runoff"]),
        reference=ReferenceConfig(sources={"Runoff": "DemoStn"}),
        simulation={
            "Custom": SimulationEntry(
                model="UnknownModel",
                root_dir="/sim",
                variables={
                    "Runoff": {"varname": "runoff", "varunit": "m3 s-1"},
                },
            ),
        },
    )

    _, ref_nml, _ = build_legacy_namelists(cfg)

    assert ref_nml["Runoff"]["DemoStn_fulllist"].replace("\\", "/") == (tmp_path / "list" / "subset.csv").as_posix()


def test_generated_sim_station_fulllist_columns_are_accepted_by_legacy_reader(tmp_path):
    """sim scan writes use_syear/use_eyear; legacy station filtering needs sim_syear/sim_eyear."""
    import pandas as pd

    (tmp_path / "case").mkdir()
    sim_file = tmp_path / "US-AAA.nc"
    sim_file.write_text("placeholder")
    fulllist = tmp_path / "sim_stations.csv"
    pd.DataFrame(
        [
            {
                "ID": "US-AAA",
                "sim_lon": 10.0,
                "sim_lat": 45.0,
                "use_syear": 2001,
                "use_eyear": 2002,
                "sim_dir": str(sim_file),
            }
        ]
    ).to_csv(fulllist, index=False)

    main_nl = {
        "general": {
            "basedir": str(tmp_path),
            "basename": "case",
            "syear": 2000,
            "eyear": 2005,
            "min_year": 0,
        }
    }
    sim_nml = {
        "general": {},
        "Runoff": {
            "SimA_data_type": "stn",
            "SimA_varname": "Q",
            "SimA_varunit": "m3 s-1",
            "SimA_data_groupby": "Single",
            "SimA_dir": str(tmp_path),
            "SimA_tim_res": "Day",
            "SimA_grid_res": "",
            "SimA_syear": 2000,
            "SimA_eyear": 2005,
            "SimA_prefix": "",
            "SimA_suffix": "",
            "SimA_fulllist": str(fulllist),
        },
    }
    ref_nml = {
        "general": {},
        "Runoff": {
            "RefG_data_type": "grid",
            "RefG_varname": "Q",
            "RefG_varunit": "m3 s-1",
            "RefG_data_groupby": "Single",
            "RefG_dir": str(tmp_path),
            "RefG_tim_res": "Day",
            "RefG_grid_res": 0.5,
            "RefG_syear": 1999,
            "RefG_eyear": 2006,
            "RefG_prefix": "",
            "RefG_suffix": "",
        },
    }

    info = GeneralInfoReader(main_nl, sim_nml, ref_nml, [], [], [], [], "Runoff", "SimA", "RefG")

    assert list(info.stn_list["ID"]) == ["US-AAA"]
    assert int(info.stn_list.iloc[0]["sim_syear"]) == 2001
    assert int(info.stn_list.iloc[0]["sim_eyear"]) == 2002
    assert int(info.stn_list.iloc[0]["use_syear"]) == 2001
    assert int(info.stn_list.iloc[0]["use_eyear"]) == 2002


def test_station_min_year_threshold_counts_inclusive_years(tmp_path):
    import pandas as pd

    sim_file = tmp_path / "US-ARM.nc"
    sim_file.write_text("placeholder")
    fulllist = tmp_path / "sim_stations.csv"
    pd.DataFrame(
        [
            {
                "ID": "US-ARM",
                "sim_lon": -97.4888,
                "sim_lat": 36.6058,
                "use_syear": 2004,
                "use_eyear": 2005,
                "sim_dir": str(sim_file),
            }
        ]
    ).to_csv(fulllist, index=False)

    main_nl = {
        "general": {
            "basedir": str(tmp_path),
            "basename": "case",
            "syear": 2004,
            "eyear": 2005,
            "min_year": 2,
        }
    }
    sim_nml = {
        "general": {},
        "Latent_Heat": {
            "stn_data_type": "stn",
            "stn_varname": "f_lfevpa",
            "stn_varunit": "W m-2",
            "stn_data_groupby": "Single",
            "stn_dir": str(tmp_path),
            "stn_tim_res": "Day",
            "stn_grid_res": "",
            "stn_syear": 2004,
            "stn_eyear": 2005,
            "stn_prefix": "",
            "stn_suffix": "",
            "stn_fulllist": str(fulllist),
        },
    }
    ref_nml = {
        "general": {},
        "Latent_Heat": {
            "FLUXCOM_LowRes_data_type": "grid",
            "FLUXCOM_LowRes_varname": "le",
            "FLUXCOM_LowRes_varunit": "W m-2",
            "FLUXCOM_LowRes_data_groupby": "Single",
            "FLUXCOM_LowRes_dir": str(tmp_path),
            "FLUXCOM_LowRes_tim_res": "Month",
            "FLUXCOM_LowRes_grid_res": 0.5,
            "FLUXCOM_LowRes_syear": 2004,
            "FLUXCOM_LowRes_eyear": 2005,
            "FLUXCOM_LowRes_prefix": "",
            "FLUXCOM_LowRes_suffix": "",
        },
    }

    info = GeneralInfoReader(main_nl, sim_nml, ref_nml, [], [], [], [], "Latent_Heat", "stn", "FLUXCOM_LowRes")

    assert list(info.stn_list["ID"]) == ["US-ARM"]


def test_integration_fixture_uses_lowres_registry_root_without_duplicate_category():
    fixture = Path("test_data/openbench.yaml")
    if not fixture.exists():
        pytest.skip("local integration fixture is not present")

    cfg = load_config(fixture)

    _, ref_nml, _ = build_legacy_namelists(cfg)

    source = ref_nml["general"]["Evapotranspiration_ref_source"]
    ref_dir = ref_nml["Evapotranspiration"][f"{source}_dir"].replace("\\", "/")
    assert source == "GLEAM_v4.2a_LowRes"
    assert "/Water/Water/" not in ref_dir
    assert ref_dir.endswith("OpenBench-wei/dataset/Reference/Grid/LowRes/Water/Evapotranspiration/GLEAM_v4.2a")


def test_sediment_reference_matches_each_item_from_its_own_variable(monkeypatch, tmp_path):
    import numpy as np
    import pandas as pd
    import xarray as xr

    import openbench.data.registry.manager as registry_manager_module
    from openbench.data.registry.schema import StationMatchingConfig

    xr.Dataset(
        {
            "station": ("station", np.array(["S1", "S2"], dtype=object)),
            "lon": ("station", [10.0, 11.0]),
            "lat": ("station", [20.0, 21.0]),
            "upstream_area": ("station", [5000.0, 6000.0]),
            "cama_lon_15min": ("station", [10.0, 11.0]),
            "cama_lat_15min": ("station", [20.0, 21.0]),
            "cama_alloc_err_15min": ("station", [0.0, 0.0]),
            "discharge": (("station", "time"), [[1.0, 2.0], [np.nan, np.nan]]),
            "ssc": (("station", "time"), [[30.0, 40.0], [50.0, 60.0]]),
        },
        coords={"time": pd.date_range("2000-01-01", periods=2, freq="D")},
    ).to_netcdf(tmp_path / "Sed_Daily_full.nc")
    reference = SimpleNamespace(
        station_matching=StationMatchingConfig(dataset_file="Sed_Daily_full.nc", area_var="upstream_area"),
        variables={
            "Discharge_for_Sediment": SimpleNamespace(),
            "Suspended_Sediment_Concentration": SimpleNamespace(),
            "Suspended_Sediment_Load": SimpleNamespace(),
        },
    )
    registry = SimpleNamespace(get_reference=lambda _name: reference)
    monkeypatch.setattr(registry_manager_module, "get_registry", lambda: registry)

    matched = {}
    for item, varname in (("Discharge_for_Sediment", "discharge"), ("Suspended_Sediment_Concentration", "ssc")):
        reader = _streamflow_matching_reader(monkeypatch, tmp_path, "Sed_Daily_full.nc", 0.25)
        monkeypatch.setattr(registry_manager_module, "get_registry", lambda: registry)
        reader.__dict__.update(item=item, ref_source="Sed_Daily", ref_varname=varname, stn_list=pd.DataFrame())
        reader._filter_stations()
        matched[item] = reader.stn_list

    # Only S1 has discharge; both stations have SSC. Each item reads its own variable,
    # and its station files sit in a scratch directory of their own.
    assert matched["Discharge_for_Sediment"]["ID"].tolist() == ["S1"]
    assert matched["Suspended_Sediment_Concentration"]["ID"].tolist() == ["S1", "S2"]
    q_file = matched["Discharge_for_Sediment"]["ref_dir"].iloc[0]
    ssc_file = matched["Suspended_Sediment_Concentration"]["ref_dir"].iloc[0]
    assert "Discharge_for_Sediment" in q_file and "Suspended_Sediment_Concentration" in ssc_file
    with xr.open_dataset(ssc_file) as station_ds:
        assert list(station_ds.data_vars) == ["ssc"]
