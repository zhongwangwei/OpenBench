"""WSE-only correction and unchanged GeographicLib grid decoding safeguards."""

import numpy as np
import pytest
import xarray as xr

from openbench.data.vertical_datum import (
    VerticalDatumError,
    _sample_geoid,
    harmonize_station_vertical_datum,
    normalize_vertical_datum_spec,
)


def pgm(tmp_path, model="EGM2008", values=None, offset=-10, scale=0.5):
    path = tmp_path / (model + ".pgm")
    values = np.array(values if values is not None else [[20] * 4, [40, 60, 80, 100], [120] * 4], dtype=">u2")
    path.write_bytes(
        (
            f"P5\n# Description {model} geoid\n# Offset {offset}\n# Scale {scale}\n# Vertical_Datum WGS84\n4 3\n65535\n"
        ).encode()
        + values.tobytes()
    )
    return str(path)


def test_pgm_interpolation(tmp_path):
    path = pgm(tmp_path)
    assert _sample_geoid(path, "EGM2008", 0, 0) == 10
    assert _sample_geoid(path, "EGM2008", 45, 45) == 7.5
    assert _sample_geoid(path, "EGM2008", 0, -45) == 25
    assert _sample_geoid(path, "EGM2008", 0, 360) == 10
    assert _sample_geoid(path, "EGM2008", 90, 12) == pytest.approx(0)
    assert _sample_geoid(path, "EGM2008", -90, 211) == 50


def test_bad_grid(tmp_path):
    path = pgm(tmp_path)
    with pytest.raises(VerticalDatumError):
        _sample_geoid(path, "EGM96", 0, 0)
    from pathlib import Path

    p = Path(path)
    p.write_bytes(p.read_bytes()[:-1])
    with pytest.raises(VerticalDatumError):
        _sample_geoid(path, "EGM2008", 0, 0)


@pytest.mark.parametrize(
    "old,new",
    [
        (b"P5", b"P2"),
        (b"65535", b"255"),
        (b"4 3", b"5 3"),
        (b"# Offset -10\n", b""),
        (b"# Scale 0.5", b"# Scale 0"),
        (b"# Scale 0.5", b"# Scale nan"),
        (b"EGM2008 geoid", b"Unknown geoid"),
    ],
)
def test_malformed_headers(tmp_path, old, new):
    from pathlib import Path

    path = Path(pgm(tmp_path))
    path.write_bytes(path.read_bytes().replace(old, new))
    with pytest.raises(VerticalDatumError):
        _sample_geoid(str(path), "EGM2008", 0, 0)


def test_binary_whitespace_and_big_endian(tmp_path):
    path = pgm(tmp_path, values=[[0x0A20] * 4] * 3, offset=0, scale=1)
    assert _sample_geoid(path, "EGM2008", 0, 0) == 2592


def test_missing_grid(tmp_path):
    with pytest.raises(VerticalDatumError, match="Cannot read geoid"):
        _sample_geoid(str(tmp_path / "absent.pgm"), "EGM2008", 0, 0)


@pytest.mark.parametrize("comment", [b"# Origin 90S 0E\n", b"# Origin 90N 180E\n", b"# Vertical_Datum GRS80\n"])
def test_incompatible_optional_grid_metadata(tmp_path, comment):
    from pathlib import Path

    path = Path(pgm(tmp_path))
    content = path.read_bytes().replace(b"# Vertical_Datum WGS84\n", b"")
    path.write_bytes(content.replace(b"P5\n", b"P5\n" + comment))
    with pytest.raises(VerticalDatumError):
        _sample_geoid(str(path), "EGM2008", 0, 0)


def test_grid_header_refresh_with_preserved_mtime(tmp_path):
    import os
    from pathlib import Path

    path = Path(pgm(tmp_path))
    assert _sample_geoid(str(path), "EGM2008", 0, 0) == 10
    before = path.stat()
    path.write_bytes(path.read_bytes().replace(b"# Offset -10", b"# Offset -20"))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    assert _sample_geoid(str(path), "EGM2008", 0, 0) == 0


@pytest.mark.parametrize("comments", [b"", b"# Origin 90N 0E\n# Vertical_Datum WGS84\n"])
def test_supported_optional_grid_metadata(tmp_path, comments):
    from pathlib import Path

    path = Path(pgm(tmp_path))
    content = path.read_bytes().replace(b"# Vertical_Datum WGS84\n", b"")
    path.write_bytes(content.replace(b"P5\n", b"P5\n" + comments))
    assert _sample_geoid(str(path), "EGM2008", 0, 0) == 10


def spec(tmp_path):
    return {"source_geoid": pgm(tmp_path), "target_geoid": pgm(tmp_path, "EGM96", [[20] * 4] * 3)}


def heights(**attrs):
    return xr.DataArray([10.0, np.nan], dims="time", name="wse", attrs={"units": "m", **attrs})


def run(ref, config, sim=None, **kwargs):
    return harmonize_station_vertical_datum(
        heights() if sim is None else sim,
        ref,
        ref_config=config,
        station=kwargs.pop("station", {"ref_lat": 0, "ref_lon": 0}),
        item=kwargs.pop("item", "Water_Surface_Elevation"),
        **kwargs,
    )


@pytest.mark.parametrize("dataset", [False, True])
def test_minimal_wse_correction_no_model_declaration_no_mutation(tmp_path, dataset):
    config = spec(tmp_path)
    ref = heights(description="Unrelated depth words do not override explicit WSE configuration")
    sim = heights()
    if dataset:
        ref, sim = ref.to_dataset(), sim.to_dataset()
    before_ref, before_sim = ref.copy(deep=True), sim.copy(deep=True)
    returned_sim, result = run(ref, config, sim)
    assert returned_sim is sim
    corrected = result.wse if dataset else result
    np.testing.assert_allclose(corrected, [20.0, np.nan], equal_nan=True)
    assert corrected.attrs["openbench_vertical_datum_offset_m"] == 10
    assert corrected.attrs["openbench_vertical_datum"] == "EGM96"
    xr.testing.assert_identical(ref, before_ref)
    xr.testing.assert_identical(sim, before_sim)
    with pytest.raises(VerticalDatumError, match="already corrected"):
        run(result, config, sim)


@pytest.mark.parametrize("item", ["Water_Surface_Elevation", "Streamflow", "Dam_Elevation", "River_Water_Level"])
def test_default_is_noop_for_all_variables(item):
    ref, sim = xr.DataArray([1]), xr.DataArray([2])
    actual_sim, actual_ref = run(ref, None, sim, item=item, station={})
    assert actual_sim is sim and actual_ref is ref


@pytest.mark.parametrize(
    "item", ["Dam_Water_Elevation", "River_Water_Level", "Lake_Water_Level", "Dam_Elevation", "Streamflow"]
)
def test_only_water_surface_elevation_may_opt_in(tmp_path, item):
    with pytest.raises(VerticalDatumError, match="only Water_Surface_Elevation"):
        run(heights(), spec(tmp_path), item=item)


@pytest.mark.parametrize("units", [None, "cm", "mm", "m2"])
@pytest.mark.parametrize("side", ["ref", "sim"])
def test_metres_required_on_processed_series(tmp_path, side, units):
    ref, sim = heights(), heights()
    (ref if side == "ref" else sim).attrs["units"] = units
    with pytest.raises(VerticalDatumError, match="metres"):
        run(ref, spec(tmp_path), sim)


@pytest.mark.parametrize("metadata", [{"vertical_datum": "EGM96"}, {"openbench_vertical_datum": "ellipsoidal"}])
def test_explicit_observation_datum_conflict_fails(tmp_path, metadata):
    with pytest.raises(VerticalDatumError, match="conflicts"):
        run(heights(**metadata), spec(tmp_path))


@pytest.mark.parametrize(
    "station",
    [
        {},
        {"sim_lat": 0, "sim_lon": 0},
        {"ref_lat": 91, "ref_lon": 0},
        {"ref_lat": 0, "ref_lon": 361},
        {"ref_lat": float("nan"), "ref_lon": 0},
        {"ref_lat": True, "ref_lon": 0},
    ],
)
def test_reference_coordinates_required(tmp_path, station):
    with pytest.raises(VerticalDatumError, match="Reference WGS84"):
        run(heights(), spec(tmp_path), station=station)


def test_model_side_configuration_rejected_even_without_reference_config():
    with pytest.raises(VerticalDatumError, match="not simulation"):
        run(heights(), None, sim_config={"source": "EGM96"})


@pytest.mark.parametrize(
    "value",
    [
        True,
        [],
        {},
        {"source": "EGM2008"},
        {"source_geoid": "x", "target_geoid": "y", "source": "EGM96"},
        {"source_geoid": "x", "target_geoid": "y", "quantity": "depth"},
        {"source_geoid": "x", "target_geoid": "y", "extra": 1},
    ],
)
def test_invalid_spec(value):
    with pytest.raises(VerticalDatumError):
        normalize_vertical_datum_spec(value)


def test_optional_fixed_fields_are_normalized_without_mutating_input():
    value = {"source_geoid": "08.pgm", "target_geoid": "96.pgm", "source": "egm2008"}
    result = normalize_vertical_datum_spec(value)
    assert result["source"] == "EGM2008" and result["target"] == "EGM96"
    assert value["source"] == "egm2008" and "target" not in value


def test_geoid_sampling_does_not_cache_memory_mappings(tmp_path, monkeypatch):
    from pathlib import Path

    def forbidden_map(*args, **kwargs):
        raise AssertionError("Geoid samples must not leave a Windows file mapping open")

    monkeypatch.setattr(np, "memmap", forbidden_map)
    path = Path(pgm(tmp_path))
    assert _sample_geoid(str(path), "EGM2008", 0, 0) == 10
    # Updating and deleting a sampled grid must remain possible while OpenBench runs.
    path.write_bytes(path.read_bytes().replace(b"# Offset -10", b"# Offset -20"))
    assert _sample_geoid(str(path), "EGM2008", 0, 0) == 0
    path.unlink()
    assert not path.exists()
