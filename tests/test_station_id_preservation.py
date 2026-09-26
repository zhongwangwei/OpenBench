"""Station IDs are labels, not numbers, even when every ID contains digits."""

import pandas as pd
import xarray as xr


def _fulllist(path, nc_name):
    pd.DataFrame(
        {
            "ID": ["0000000009463", "0000000009464"],
            "LON": [10.0, 11.0],
            "LAT": [20.0, 21.0],
            "SYEAR": [2000, 2000],
            "EYEAR": [2000, 2000],
            "DIR": [nc_name, nc_name],
        }
    ).to_csv(path, index=False)


def test_station_fulllist_keeps_leading_zero_ids_through_merge_and_selection(tmp_path):
    from openbench.data.processing import StationDatasetProcessing

    nc = tmp_path / "stations.nc"
    xr.Dataset(
        {"wse": (("time", "station"), [[1.0, 2.0]])},
        coords={"time": pd.date_range("2000-01-01", periods=1), "station": ["0000000009463", "0000000009464"]},
    ).to_netcdf(nc)
    sim_list, ref_list = tmp_path / "sim.csv", tmp_path / "ref.csv"
    _fulllist(sim_list, nc.name)
    _fulllist(ref_list, nc.name)

    proc = StationDatasetProcessing.__new__(StationDatasetProcessing)
    proc.ref_data_type = proc.sim_data_type = "stn"
    proc.ref_source, proc.sim_source = "Ref", "Sim"
    proc.ref_fulllist, proc.sim_fulllist = str(ref_list), str(sim_list)
    proc.ref_dir = proc.sim_dir = str(tmp_path)
    proc.ref_varname = proc.sim_varname = "wse"
    proc.casedir = str(tmp_path / "case")
    proc.setup_output_directories()

    assert proc.station_list["ID"].tolist() == ["0000000009463", "0000000009464"]
    with xr.open_dataset(nc) as ds:
        selected = proc._select_merged_station_data(ds, proc.station_list.iloc[1], "ref")
        assert selected["wse"].item() == 2.0


def test_runtime_station_fulllist_keeps_leading_zero_ids(tmp_path):
    from openbench.config.runtime_info import GeneralInfoReader

    sim_list, ref_list = tmp_path / "sim.csv", tmp_path / "ref.csv"
    _fulllist(sim_list, "stations.nc")
    _fulllist(ref_list, "stations.nc")
    reader = GeneralInfoReader.__new__(GeneralInfoReader)
    reader.ref_data_type = reader.sim_data_type = "stn"
    reader.ref_fulllist, reader.sim_fulllist = str(ref_list), str(sim_list)
    reader.ref_dir = reader.sim_dir = str(tmp_path)

    reader._read_and_merge_station_lists()

    assert reader.stn_list["ID"].tolist() == ["0000000009463", "0000000009464"]


def test_station_evaluation_keeps_leading_zero_ids(tmp_path, monkeypatch):
    from openbench.core.evaluation import Evaluation_stn

    station_list = tmp_path / "stn_Ref_Sim_list.txt"
    pd.DataFrame(
        {
            "ID": ["0000000009463", "0000000009464"],
            "sim_lon": [10.0, 11.0],
            "sim_lat": [20.0, 21.0],
            "use_syear": [2000, 2000],
            "use_eyear": [2000, 2000],
        }
    ).to_csv(station_list, index=False)
    evaluator = Evaluation_stn.__new__(Evaluation_stn)
    evaluator.casedir = str(tmp_path)
    evaluator.item, evaluator.ref_source, evaluator.sim_source = "Water_Surface_Elevation", "Ref", "Sim"
    evaluator.ref_fulllist = str(station_list)
    evaluator.num_cores = 1
    evaluator.output_manager = None
    evaluator.metrics, evaluator.scores = ["bias"], []
    seen_ids = []

    def evaluate(stations, index):
        seen_ids.append(stations.iloc[index]["ID"])
        return {"bias": 0.0}

    evaluator.make_evaluation_parallel = evaluate
    monkeypatch.setattr("openbench.core.evaluation.make_plot_index_stn", lambda *_: None)
    evaluator.make_evaluation_P()

    assert seen_ids == ["0000000009463", "0000000009464"]


def test_station_sidecar_metadata_matches_leading_zero_id(tmp_path):
    from openbench.data.station_scanner import _load_station_metadata, _station_metadata_row

    (tmp_path / "station_case.csv").write_text("ID,LON,LAT\n0000000009463,16.1374,10.7639\n", encoding="utf-8")

    metadata = _load_station_metadata(tmp_path)
    row = _station_metadata_row(metadata, "0000000009463", tmp_path / "station.nc")

    assert row is not None
    assert row["ID"] == "0000000009463"
    assert row["LON"] == 16.1374
    assert row["LAT"] == 10.7639


def test_station_id_key_ignores_zero_padding_only_for_digit_ids():
    from openbench.util.station_ids import station_id_key

    assert station_id_key("0000000009463") == station_id_key(9463) == station_id_key(" 9463 ") == "9463"
    assert station_id_key("000") == "0"
    assert station_id_key("US-Ha1") == "US-Ha1"
    assert station_id_key("0A12") == "0A12"


def _merged_processor(tmp_path, sim_ids, ref_ids, station_coord):
    from openbench.data.processing import StationDatasetProcessing

    nc = tmp_path / "stations.nc"
    xr.Dataset(
        {"wse": (("time", "station"), [[1.0, 2.0]])},
        coords={"time": pd.date_range("2000-01-01", periods=1), "station": station_coord},
    ).to_netcdf(nc)
    sim_list, ref_list = tmp_path / "sim.csv", tmp_path / "ref.csv"
    for path, ids in ((sim_list, sim_ids), (ref_list, ref_ids)):
        pd.DataFrame(
            {
                "ID": ids,
                "LON": [10.0, 11.0],
                "LAT": [20.0, 21.0],
                "SYEAR": [2000, 2000],
                "EYEAR": [2000, 2000],
                "DIR": [nc.name, nc.name],
            }
        ).to_csv(path, index=False)

    proc = StationDatasetProcessing.__new__(StationDatasetProcessing)
    proc.ref_data_type = proc.sim_data_type = "stn"
    proc.ref_source, proc.sim_source = "Ref", "Sim"
    proc.ref_fulllist, proc.sim_fulllist = str(ref_list), str(sim_list)
    proc.ref_dir = proc.sim_dir = str(tmp_path)
    proc.ref_varname = proc.sim_varname = "wse"
    proc.casedir = str(tmp_path / "case")
    proc.setup_output_directories()
    return proc, nc


def test_padded_station_ids_select_integer_station_coordinates(tmp_path):
    proc, nc = _merged_processor(
        tmp_path,
        ["0000000009463", "0000000009464"],
        ["0000000009463", "0000000009464"],
        station_coord=[9463, 9464],
    )

    assert proc.station_list["ID"].tolist() == ["0000000009463", "0000000009464"]
    with xr.open_dataset(nc) as ds:
        assert proc._select_merged_station_data(ds, proc.station_list.iloc[1], "ref")["wse"].item() == 2.0


def test_station_fulllists_merge_padded_and_unpadded_ids(tmp_path):
    proc, nc = _merged_processor(
        tmp_path,
        ["9463", "9464"],
        ["0000000009463", "0000000009464"],
        station_coord=["0000000009463", "0000000009464"],
    )

    # The merged list keeps the simulation spelling, as a plain merge on ID did.
    assert proc.station_list["ID"].tolist() == ["9463", "9464"]
    assert "ID_refdup" not in proc.station_list.columns
    with xr.open_dataset(nc) as ds:
        assert proc._select_merged_station_data(ds, proc.station_list.iloc[0], "ref")["wse"].item() == 1.0


def test_runtime_station_lists_match_padded_and_unpadded_ids_by_id(tmp_path, caplog):
    from openbench.config.runtime_info import GeneralInfoReader

    sim_list, ref_list = tmp_path / "sim.csv", tmp_path / "ref.csv"
    _fulllist(sim_list, "stations.nc")
    pd.DataFrame(
        {
            "ID": ["9463", "9464"],
            "LON": [50.0, 51.0],  # far apart: only an ID match can pair these
            "LAT": [-20.0, -21.0],
            "SYEAR": [2000, 2000],
            "EYEAR": [2000, 2000],
            "DIR": ["stations.nc", "stations.nc"],
        }
    ).to_csv(ref_list, index=False)
    reader = GeneralInfoReader.__new__(GeneralInfoReader)
    reader.ref_data_type = reader.sim_data_type = "stn"
    reader.ref_fulllist, reader.sim_fulllist = str(ref_list), str(sim_list)
    reader.ref_dir = reader.sim_dir = str(tmp_path)

    with caplog.at_level("WARNING"):
        reader._read_and_merge_station_lists()

    assert reader.stn_list["ID"].tolist() == ["0000000009463", "0000000009464"]
    assert "Attempting spatial matching" not in caplog.text


def test_station_sidecar_metadata_keeps_zeros_for_any_id_column_case(tmp_path):
    from openbench.data.station_scanner import _load_station_metadata, _station_metadata_row

    (tmp_path / "station_case.csv").write_text("Id,LON,LAT\n0000000009463,16.1374,10.7639\n", encoding="utf-8")

    metadata = _load_station_metadata(tmp_path)
    assert metadata["Id"].tolist() == ["0000000009463"]
    row = _station_metadata_row(metadata, "9463", tmp_path / "station.nc")
    assert row is not None
    assert row["LAT"] == 10.7639


def test_hydroweb_finds_station_file_named_by_unpadded_id(tmp_path):
    from types import SimpleNamespace

    from openbench.data.custom.HydroWeb_filter import process_station

    river = tmp_path / "output" / "river"
    river.mkdir(parents=True)
    xr.Dataset(coords={"time": pd.date_range("2000-01-01", "2003-12-31", freq="D")}).to_netcdf(
        river / "hydroprd_river_9463.nc"
    )
    info = SimpleNamespace(
        compare_tim_res="D",
        ref_dir=str(tmp_path),
        debug_mode=False,
        sim_syear=2000,
        sim_eyear=2003,
        syear=2000,
        eyear=2003,
        min_year=1,
        min_lon=-180,
        max_lon=180,
        min_lat=-90,
        max_lat=90,
    )

    result = process_station({"ID": "0000000009463", "lon": 10.0, "lat": 20.0}, info)

    assert result["Flag"] is True
    assert result["ref_dir"] == f"{tmp_path}/output/river/hydroprd_river_9463.nc"
