"""Preflight follows runtime selection, including unusable named files."""

from pathlib import Path

import pytest
import xarray as xr

from openbench.cli.check import data_file_findings


def _write(path, **variables):
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({name: ("time", [value]) for name, value in variables.items()}).to_netcdf(path)


@pytest.mark.parametrize("groupby", ["Year", "Single"])
def test_unusable_named_file_does_not_hide_compute_branch_conflict(tmp_path, groupby):
    prefix = "runoff_" if groupby == "Year" else "runoff_all"
    _write(tmp_path / ("runoff_2001.nc" if groupby == "Year" else "runoff_all.nc"), unrelated=0)
    for branch in ("0p25", "0p5"):
        _write(tmp_path / branch / "input_2001.nc", rain=1, snow=2)
    errors, _ = data_file_findings(
        "Simulation",
        str(tmp_path),
        prefix=prefix,
        suffix="",
        data_groupby=groupby,
        years=[2001],
        compute="ds['rain'] + ds['snow']",
    )
    assert len(errors) == 1 and "several folders" in errors[0]


@pytest.mark.parametrize("prefix", ["runoff_", "input_"])
def test_same_named_complementary_compute_inputs_are_allowed(tmp_path, prefix):
    _write(tmp_path / "rain" / "input_2001.nc", rain=1)
    _write(tmp_path / "snow" / "input_2001.nc", snow=2)
    assert data_file_findings(
        "Reference",
        str(tmp_path),
        prefix=prefix,
        suffix="",
        data_groupby="Year",
        years=[2001],
        compute="ds['rain'] + ds['snow']",
    ) == ([], [])


def test_compute_preflight_reuses_one_inventory_across_years(tmp_path, monkeypatch):
    from openbench.data import file_lookup

    for year in (2001, 2002, 2003):
        _write(tmp_path / f"input_{year}.nc", rain=1, snow=2)
    original = file_lookup.netcdf_inventory
    calls = []

    def counted(directory):
        calls.append(directory)
        return original(directory)

    monkeypatch.setattr(file_lookup, "netcdf_inventory", counted)
    assert data_file_findings(
        "Reference",
        str(tmp_path),
        prefix="runoff_",
        suffix="",
        data_groupby="Year",
        years=[2001, 2002, 2003],
        compute="ds['rain'] + ds['snow']",
    ) == ([], [])
    assert calls == [str(tmp_path)]


@pytest.mark.parametrize("groupby", ["Year", "Month", "Day"])
@pytest.mark.parametrize("command", ["check", "dry-run"])
@pytest.mark.parametrize("complementary", [False, True])
def test_cli_and_runtime_agree_on_compute_branches(tmp_path, monkeypatch, command, complementary, groupby):
    from click.testing import CliRunner

    from openbench.cli.main import cli
    from openbench.data.processing import DatasetProcessing
    from tests.test_cli_check_preflight import (
        _base_config,
        _install_registry,
        _model,
        _ref,
        _Registry,
        _var,
        _write_config,
    )

    root = tmp_path / "sim"
    _write(root / "runoff_2001.nc", unrelated=0)
    _write(root / "rain" / "input_2001.nc", rain=1)
    _write(root / "snow" / "input_2001.nc", **({"snow": 2} if complementary else {"rain": 100, "snow": 100}))
    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    (ref_root / "2001.nc").touch()
    mapping = _var("runoff")
    mapping.prefix = "runoff_"
    mapping.compute = "ds['rain'] + ds['snow']"
    _install_registry(
        monkeypatch,
        _Registry(
            {"DemoRef": _ref("DemoRef", str(ref_root))},
            {"KnownModel": _model(variables={"Runoff": mapping})},
        ),
    )
    cfg = _base_config(tmp_path, project={"years": [2001, 2001]})
    cfg["simulation"]["CaseA"]["data_groupby"] = groupby
    config = _write_config(tmp_path, cfg)
    args = ["check", str(config)] if command == "check" else ["run", str(config), "--dry-run"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == (0 if complementary and groupby == "Year" else 1), result.output
    if groupby != "Year":
        assert "each file" in result.output
    assert ("same names in several folders" in result.output) == (not complementary)

    processor = object.__new__(DatasetProcessing)
    processor.__dict__.update(item="Runoff", sim_source="Sim", Sim_model="KnownModel")
    selected = processor._find_data_files(str(root), "runoff_", 2001, "", "sim", ["runoff"])
    assert {path.relative_to(root).as_posix() for path in map(Path, selected)} == {
        "rain/input_2001.nc",
        "snow/input_2001.nc",
    }
    if complementary:
        from openbench.data.compute import execute_compute

        frames = []
        for path in selected:
            with xr.open_dataset(path) as ds:
                frames.append(ds.load())
        assert execute_compute(xr.merge(frames), mapping.compute, "Runoff").values.tolist() == [3]


def test_inventory_is_shared_within_check_and_released_afterwards(tmp_path, monkeypatch):
    from openbench.cli import check as check_module
    from openbench.data import file_lookup

    _write(tmp_path / "input_2001.nc", rain=1, snow=2)
    original = file_lookup.netcdf_inventory
    calls = []

    def counted(directory):
        calls.append(directory)
        return original(directory)

    monkeypatch.setattr(file_lookup, "netcdf_inventory", counted)
    kwargs = dict(prefix="runoff_", suffix="", data_groupby="Year", years=[2001], compute="ds['rain'] + ds['snow']")
    with check_module._reference_resolution_cache():
        assert data_file_findings("Reference", str(tmp_path), **kwargs) == ([], [])
        assert data_file_findings("Simulation", str(tmp_path), **kwargs) == ([], [])
    assert calls == [str(tmp_path)]
    assert check_module._FILE_INVENTORIES is None
    (tmp_path / "input_2001.nc").unlink()
    with check_module._reference_resolution_cache():
        assert data_file_findings("Reference", str(tmp_path), **kwargs)[1]
    assert calls == [str(tmp_path), str(tmp_path)]


@pytest.mark.parametrize("expression", ["ds['rain'] + ds['snow']", "ds.get('rain') + ds.get('snow')"])
@pytest.mark.parametrize("groupby", ["Month", "Day"])
@pytest.mark.parametrize("prefix", ["input_", "runoff_"])
def test_per_file_compute_rejects_split_inputs(tmp_path, groupby, prefix, expression):
    _write(tmp_path / "rain" / "input_200101.nc", rain=1)
    _write(tmp_path / "snow" / "input_200101.nc", snow=2)
    errors, _ = data_file_findings(
        "Simulation",
        str(tmp_path),
        prefix=prefix,
        suffix="",
        data_groupby=groupby,
        years=[2001],
        compute=expression,
        candidate_varnames=["runoff"],
    )
    assert errors and "each file" in errors[0]
    assert "snow" in errors[0] and "rain" in errors[0]


@pytest.mark.parametrize(
    "expression,variables",
    [
        ("ds['rain'] + ds['snow']", {"rain": 1, "snow": 2}),
        ("ds['rain'] + ds['snow']", {"runoff": 3}),
        ("ds['rain'] + ds['snow']", {"Total_Runoff": 3}),
        ("ds.sum_prefix('rain_', 2)", {"rain_1": 1, "rain_2": 2}),
        ("ds['rain'] if 0 > 1 > ds['snow'] else ds['rain']", {"rain": 1}),
        ("ds['rain'] if 'rain' in ds else ds['snow']", {"rain": 1}),
        ("ds.get('snow', default=ds['rain'])", {"rain": 1}),
    ],
)
def test_per_file_compute_keeps_complete_optional_and_raw_inputs(tmp_path, expression, variables):
    _write(tmp_path / "input_200101.nc", **variables)
    assert data_file_findings(
        "Simulation",
        str(tmp_path),
        prefix="input_",
        suffix="",
        data_groupby="Month",
        years=[2001],
        compute=expression,
        candidate_varnames=["runoff"],
        standard_varname="Total_Runoff",
    ) == ([], [])


@pytest.mark.parametrize(
    "expression,variables,primary",
    [
        ("ds['rain'] + ds['snow']", {"rain": 1, "Total_Runoff": 3}, "rain"),
        ("ds.sum_prefix('rain_', 2)", {"rain_1": 1, "Total_Runoff": 3}, "runoff"),
    ],
)
def test_per_file_compute_cannot_use_unreachable_or_forbidden_raw_fallback(tmp_path, expression, variables, primary):
    _write(tmp_path / "input_200101.nc", **variables)
    errors, _ = data_file_findings(
        "Simulation",
        str(tmp_path),
        prefix="input_",
        suffix="",
        data_groupby="Month",
        years=[2001],
        compute=expression,
        candidate_varnames=[primary],
        standard_varname="Total_Runoff",
    )
    assert errors and "each file" in errors[0]


def test_month_compute_check_reads_only_the_first_and_last_file(tmp_path, monkeypatch):
    import xarray as xr

    from openbench.cli.check import data_file_findings
    from openbench.data import file_lookup

    for month in range(1, 13):
        _write(tmp_path / f"runoff_2001-{month:02d}.nc", rain=1, snow=1)
    file_lookup._variable_names.cache_clear()
    reads = []
    real = xr.open_dataset
    monkeypatch.setattr(xr, "open_dataset", lambda path, *a, **k: reads.append(str(path)) or real(path, *a, **k))

    errors, _warnings = data_file_findings(
        "Simulation",
        str(tmp_path),
        prefix="runoff_",
        suffix="",
        data_groupby="Month",
        years=[2001],
        compute="ds['rain'] + ds['snow']",
        candidate_varnames=["runoff"],
    )

    assert not errors
    assert sorted(set(reads)) == [str(tmp_path / "runoff_2001-01.nc"), str(tmp_path / "runoff_2001-12.nc")]
