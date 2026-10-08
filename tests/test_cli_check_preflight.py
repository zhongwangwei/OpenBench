from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml
from click.testing import CliRunner

from openbench.cli.main import cli

runner = CliRunner()


def _var(
    varname: str = "ro",
    *,
    varunit: str = "mm day-1",
    sub_dir: str = "",
    fulllist: str | None = None,
):
    return SimpleNamespace(
        varname=varname,
        varunit=varunit,
        sub_dir=sub_dir,
        prefix="",
        suffix="",
        fulllist=fulllist,
        max_uparea=None,
        min_uparea=None,
        fallbacks=None,
        prefix_fallback=None,
    )


def _ref(
    name: str,
    root_dir: str,
    *,
    variable: str = "Runoff",
    data_type: str = "grid",
    tim_res: str = "Day",
    grid_res: float | None = 0.5,
    data_groupby: str = "Year",
    years: list[int] | None = None,
    var_map=None,
    fulllist: str | None = None,
    provenance: dict | None = None,
    station_matching=None,
):
    return SimpleNamespace(
        name=name,
        description=f"{name} reference",
        category="Water",
        data_type=data_type,
        tim_res=tim_res,
        grid_res=grid_res,
        data_groupby=data_groupby,
        timezone=0,
        years=years if years is not None else [2000, 2010],
        root_dir=root_dir,
        fulllist=fulllist,
        variables={variable: var_map or _var()},
        station_matching=station_matching,
        _provenance=provenance or {},
    )


def _model(
    name: str = "KnownModel",
    *,
    variables: dict | None = None,
    data_type: str = "grid",
    tim_res: str = "Day",
    grid_res: float | None = 0.5,
):
    return SimpleNamespace(
        name=name,
        description=f"{name} model",
        data_type=data_type,
        tim_res=tim_res,
        grid_res=grid_res,
        variables={} if variables is None else variables,
    )


class _Registry:
    def __init__(self, refs: dict[str, object], models: dict[str, object] | None = None):
        self.refs = {k.lower(): v for k, v in refs.items()}
        self.models = {k.lower(): v for k, v in (models or {}).items()}
        self.last_resolve_reason = ""

    def get_reference(self, name, **kwargs):
        return self.refs.get(str(name).lower())

    def get_resolution_variants(self, name):
        return {}

    def get_model(self, name):
        key = str(name).lower()
        if key == "colm" and "colm2024" in self.models:
            return self.models["colm2024"]
        return self.models.get(key)

    def list_models(self):
        return list(self.models.values())


def _write_config(tmp_path, data):
    path = tmp_path / "openbench.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _write_fake_netcdf(path):
    import numpy as np
    import xarray as xr

    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    var = "Overall_Score" if stem.endswith("_Overall_Score") else stem.rsplit("_", 1)[-1]
    xr.Dataset({var: ("sample", np.array([1.0]))}).to_netcdf(path)


def _write_time_netcdf(path, years):
    import numpy as np
    import pandas as pd
    import xarray as xr

    path.parent.mkdir(parents=True, exist_ok=True)
    times = pd.to_datetime([f"{year}-01-01" for year in years])
    xr.Dataset({"runoff": ("time", np.ones(len(times)))}, coords={"time": times}).to_netcdf(path)


def _touch_year_files(*roots, years=(2001, 2002)):
    """Name data files the way preprocessing looks them up for an empty prefix/suffix."""
    for root in roots:
        for year in years:
            (root / f"{year}.nc").touch()


def _base_config(tmp_path, *, simulation=None, reference=None, project=None, variables=None):
    return {
        "project": {
            "name": "case",
            "output_dir": str(tmp_path / "out"),
            "years": [2001, 2002],
            **(project or {}),
        },
        "evaluation": {"variables": variables or ["Runoff"]},
        "reference": reference or {"Runoff": "DemoRef"},
        "simulation": simulation
        or {
            "CaseA": {
                "model": "KnownModel",
                "root_dir": str(tmp_path / "sim"),
                "tim_res": "Day",
                "grid_res": 0.5,
            }
        },
    }


def _install_registry(monkeypatch, registry):
    from openbench.data.registry import manager as mgr_mod

    monkeypatch.setattr(mgr_mod, "get_registry", lambda: registry)


def test_false_discovery_rate_is_a_valid_statistic_name():
    from openbench.cli.check import _figlib_names

    assert "False_Discovery_Rate" in _figlib_names("statistic_nml")


def test_check_rejects_unknown_model_and_missing_model_variable(tmp_path, monkeypatch):
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    (ref_root / "runoff.nc").touch()
    config = _base_config(
        tmp_path,
        simulation={
            "MissingModelCase": {
                "model": "MissingModel",
                "root_dir": str(sim_root),
                "tim_res": "Day",
                "grid_res": 0.5,
            },
            "MissingVariableCase": {
                "model": "KnownModel",
                "root_dir": str(sim_root),
                "tim_res": "Day",
                "grid_res": 0.5,
            },
        },
    )
    path = _write_config(tmp_path, config)
    _install_registry(
        monkeypatch,
        _Registry(
            {"DemoRef": _ref("DemoRef", str(ref_root))},
            {"KnownModel": _model("KnownModel", variables={"OtherVar": _var("other")})},
        ),
    )

    result = runner.invoke(cli, ["check", str(path)])

    assert result.exit_code == 1
    assert "Model 'MissingModel' is not registered" in result.output
    assert "Variable 'Runoff' is not defined in model profile 'KnownModel'" in result.output


def test_check_variable_filters_validation_scope(tmp_path, monkeypatch):
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    (ref_root / "runoff.nc").touch()
    _touch_year_files(ref_root, sim_root)
    config = _base_config(
        tmp_path,
        variables=["Runoff", "MissingVar"],
        reference={"Runoff": "DemoRef", "MissingVar": "DemoRef"},
        simulation={
            "CaseA": {
                "model": "KnownModel",
                "root_dir": str(sim_root),
                "tim_res": "Day",
                "grid_res": 0.5,
            }
        },
    )
    path = _write_config(tmp_path, config)
    _install_registry(
        monkeypatch,
        _Registry(
            {"DemoRef": _ref("DemoRef", str(ref_root))},
            {"KnownModel": _model("KnownModel", variables={"Runoff": _var("runoff")})},
        ),
    )

    result = runner.invoke(cli, ["check", "--variable", "Runoff", str(path)])

    assert result.exit_code == 0, result.output
    assert "MissingVar" not in result.output
    assert "1 variables" in result.output


def test_check_variable_filter_is_case_insensitive(tmp_path, monkeypatch):
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    (ref_root / "runoff.nc").touch()
    _touch_year_files(ref_root, sim_root)
    config = _base_config(
        tmp_path,
        variables=["Runoff", "MissingVar"],
        reference={"Runoff": "DemoRef", "MissingVar": "DemoRef"},
        simulation={
            "CaseA": {
                "model": "KnownModel",
                "root_dir": str(sim_root),
                "tim_res": "Day",
                "grid_res": 0.5,
            }
        },
    )
    path = _write_config(tmp_path, config)
    _install_registry(
        monkeypatch,
        _Registry(
            {"DemoRef": _ref("DemoRef", str(ref_root))},
            {"KnownModel": _model("KnownModel", variables={"Runoff": _var("runoff")})},
        ),
    )

    result = runner.invoke(cli, ["check", "--variable", "runoff", str(path)])

    assert result.exit_code == 0, result.output
    assert "MissingVar" not in result.output
    assert "1 variables" in result.output


def test_run_variable_filter_uses_canonical_config_spelling(tmp_path, monkeypatch):
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    (ref_root / "runoff.nc").touch()
    _touch_year_files(ref_root, sim_root)
    config = _base_config(tmp_path, variables=["Runoff"], reference={"Runoff": "DemoRef"})
    path = _write_config(tmp_path, config)
    _install_registry(
        monkeypatch,
        _Registry(
            {"DemoRef": _ref("DemoRef", str(ref_root))},
            {"KnownModel": _model("KnownModel", variables={"Runoff": _var("runoff")})},
        ),
    )

    result = runner.invoke(cli, ["run", "--dry-run", "--variable", "RUNOFF", str(path)])

    assert result.exit_code == 0, result.output
    assert "Variables: Runoff" in result.output


def test_check_accepts_inline_only_custom_model(tmp_path, monkeypatch):
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    (ref_root / "runoff.nc").touch()
    _touch_year_files(ref_root, sim_root)
    config = _base_config(
        tmp_path,
        simulation={
            "InlineCase": {
                "model": "CustomInlineModel",
                "root_dir": str(sim_root),
                "tim_res": "Day",
                "grid_res": 0.5,
                "variables": {
                    "Runoff": {
                        "varname": "ro",
                        "varunit": "mm day-1",
                    }
                },
            }
        },
    )
    path = _write_config(tmp_path, config)
    _install_registry(monkeypatch, _Registry({"DemoRef": _ref("DemoRef", str(ref_root))}, {}))

    result = runner.invoke(cli, ["check", str(path)])

    assert result.exit_code == 0, result.output
    assert "using inline variable mappings" in result.output


def test_check_rejects_unregistered_reference_by_default(tmp_path, monkeypatch):
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    config = _base_config(tmp_path)
    path = _write_config(tmp_path, config)
    _install_registry(
        monkeypatch,
        _Registry({}, {"KnownModel": _model("KnownModel", variables={"Runoff": _var("runoff")})}),
    )

    check_result = runner.invoke(cli, ["check", str(path)])
    dry_run_result = runner.invoke(cli, ["run", "--dry-run", str(path)])

    for result in (check_result, dry_run_result):
        assert result.exit_code == 1
        assert "Runoff → DemoRef" in result.output
        assert "not in registry" in result.output


def test_check_flags_station_fulllist_time_and_year_problems(tmp_path, monkeypatch):
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    (ref_root / "runoff.nc").touch()
    missing_list = tmp_path / "missing_stations.csv"
    config = _base_config(
        tmp_path,
        project={"tim_res": "Day"},
        simulation={
            "StationCase": {
                "model": "StationModel",
                "root_dir": str(sim_root),
                "data_type": "stn",
                "tim_res": "Day",
                "data_groupby": "Single",
                "fulllist": str(missing_list),
            }
        },
    )
    path = _write_config(tmp_path, config)
    _install_registry(
        monkeypatch,
        _Registry(
            {
                "DemoRef": _ref(
                    "DemoRef",
                    str(ref_root),
                    data_type="stn",
                    tim_res="Month",
                    data_groupby="Single",
                    years=[2015, 2020],
                )
            },
            {
                "StationModel": _model(
                    "StationModel",
                    data_type="stn",
                    tim_res="Day",
                    variables={"Runoff": _var("runoff")},
                )
            },
        ),
    )

    result = runner.invoke(cli, ["check", str(path)])

    assert result.exit_code == 1
    assert "Reference 'DemoRef' time resolution Month is coarser than target Day" in result.output
    assert "Reference 'DemoRef' years [2015, 2020] do not overlap project years [2001, 2002]" in result.output
    assert "Station fulllist does not exist" in result.output
    assert "Station reference 'DemoRef' has no fulllist" in result.output


def test_check_rejects_simulation_data_outside_project_years(tmp_path, monkeypatch):
    sim_root = tmp_path / "sim"
    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    (ref_root / "runoff.nc").touch()
    _write_time_netcdf(sim_root / "runoff.nc", [2015, 2016])
    path = _write_config(
        tmp_path,
        _base_config(
            tmp_path,
            simulation={
                "CaseA": {
                    "model": "KnownModel",
                    "root_dir": str(sim_root),
                    "data_groupby": "Single",
                }
            },
        ),
    )
    _install_registry(
        monkeypatch,
        _Registry(
            {"DemoRef": _ref("DemoRef", str(ref_root))},
            {"KnownModel": _model("KnownModel", variables={"Runoff": _var("runoff")})},
        ),
    )

    result = runner.invoke(cli, ["check", str(path)])

    assert result.exit_code == 1
    assert "Simulation 'CaseA' data years [2015, 2016] do not overlap project years [2001, 2002]" in result.output


def test_check_resolves_relative_station_simulation_fulllist_against_root_dir(tmp_path, monkeypatch):
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    (ref_root / "runoff.nc").touch()
    (sim_root / "stations.csv").write_text("ID,SYEAR,EYEAR,LON,LAT,DIR\nS1,2001,2002,10,20,\n")
    (ref_root / "stations.csv").write_text("ID,SYEAR,EYEAR,LON,LAT,DIR\nS1,2001,2002,10,20,\n")
    config = _base_config(
        tmp_path,
        simulation={
            "StationCase": {
                "model": "StationModel",
                "root_dir": str(sim_root),
                "data_type": "stn",
                "tim_res": "Day",
                "data_groupby": "Single",
                "fulllist": "stations.csv",
            }
        },
    )
    path = _write_config(tmp_path, config)
    _install_registry(
        monkeypatch,
        _Registry(
            {
                "DemoRef": _ref(
                    "DemoRef",
                    str(ref_root),
                    data_type="stn",
                    fulllist="stations.csv",
                )
            },
            {
                "StationModel": _model(
                    "StationModel",
                    data_type="stn",
                    tim_res="Day",
                    variables={"Runoff": _var("runoff")},
                )
            },
        ),
    )

    result = runner.invoke(cli, ["check", str(path)])

    assert result.exit_code == 0, result.output
    assert "Station fulllist does not exist" not in result.output


def test_check_and_run_dry_run_report_inline_station_fulllist_label(tmp_path, monkeypatch):
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    (ref_root / "runoff.nc").touch()
    config = _base_config(
        tmp_path,
        simulation={
            "StationCase": {
                "model": "StationModel",
                "root_dir": str(sim_root),
                "data_type": "stn",
                "tim_res": "Day",
                "variables": {
                    "Runoff": {
                        "varname": "runoff",
                        "fulllist": "$OPENBENCH_MISSING_STATION_LIST/sites.csv",
                    }
                },
            }
        },
    )
    path = _write_config(tmp_path, config)
    _install_registry(
        monkeypatch,
        _Registry(
            {"DemoRef": _ref("DemoRef", str(ref_root))},
            {
                "StationModel": _model(
                    "StationModel",
                    data_type="stn",
                    tim_res="Day",
                    variables={"Runoff": _var("runoff")},
                )
            },
        ),
    )

    check_result = runner.invoke(cli, ["check", str(path)])
    dry_run_result = runner.invoke(cli, ["run", "--dry-run", str(path)])

    assert check_result.exit_code == 1
    assert dry_run_result.exit_code == 1
    expected = "simulation.StationCase.variables.Runoff.fulllist contains unresolved environment variable"
    assert expected in check_result.output
    assert expected in dry_run_result.output


def test_check_reports_model_alias_extended_provenance_and_missing_ref_years(tmp_path, monkeypatch):
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    (ref_root / "runoff.nc").touch()
    _touch_year_files(ref_root, sim_root)
    config = _base_config(
        tmp_path,
        simulation={
            "AliasCase": {
                "model": "colm",
                "root_dir": str(sim_root),
                "tim_res": "Day",
                "grid_res": 0.5,
            }
        },
    )
    path = _write_config(tmp_path, config)
    ref = _ref(
        "DemoRef",
        str(ref_root),
        provenance={"data_type": "default", "years": "default"},
    )
    ref.years = None
    _install_registry(
        monkeypatch,
        _Registry(
            {"DemoRef": ref},
            {"CoLM2024": _model("CoLM2024", variables={"Runoff": _var("runoff")})},
        ),
    )

    result = runner.invoke(cli, ["check", str(path)])

    assert result.exit_code == 0, result.output
    assert "model alias 'colm' resolved to 'CoLM2024'" in result.output
    assert "data_type: grid (default" in result.output
    assert "years: None (default" in result.output
    assert "has no registered years" in result.output


def test_check_validates_metric_score_comparison_statistics_and_option_names(tmp_path, monkeypatch):
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    (ref_root / "runoff.nc").touch()
    output_file = tmp_path / "not-a-directory"
    output_file.write_text("x")
    config = _base_config(
        tmp_path,
        project={
            "output_dir": str(output_file),
            "timezone": 99,
        },
    )
    config["metrics"] = ["not_metric"]
    config["scores"] = ["NotScore"]
    config["comparison"] = {"enabled": True, "items": ["Taylor_diagram"]}
    config["statistics"] = {"enabled": True, "items": ["FakeStat"]}
    config["simulation"]["CaseA"]["data_groupby"] = "BadGroup"
    path = _write_config(tmp_path, config)
    _install_registry(
        monkeypatch,
        _Registry(
            {"DemoRef": _ref("DemoRef", str(ref_root))},
            {"KnownModel": _model("KnownModel", variables={"Runoff": _var("runoff")})},
        ),
    )

    result = runner.invoke(cli, ["check", str(path)])

    assert result.exit_code == 1
    assert "Unknown metric 'not_metric'" in result.output
    assert "Unknown score 'NotScore'" in result.output
    assert "Unknown comparison item 'Taylor_diagram'" in result.output
    assert "Unknown statistics item 'FakeStat'" in result.output
    assert "timezone 99 is outside" in result.output
    assert "data_groupby 'BadGroup' is invalid" in result.output
    assert "project.output_dir exists but is not a directory" in result.output


def test_run_dry_run_validates_model_resolution_like_check(tmp_path, monkeypatch):
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    (ref_root / "runoff.nc").touch()
    config = _base_config(
        tmp_path,
        simulation={
            "MissingModelCase": {
                "model": "MissingModel",
                "root_dir": str(sim_root),
                "tim_res": "Day",
                "grid_res": 0.5,
            },
            "MissingVariableCase": {
                "model": "KnownModel",
                "root_dir": str(sim_root),
                "tim_res": "Day",
                "grid_res": 0.5,
            },
        },
    )
    path = _write_config(tmp_path, config)
    _install_registry(
        monkeypatch,
        _Registry(
            {"DemoRef": _ref("DemoRef", str(ref_root))},
            {"KnownModel": _model("KnownModel", variables={"OtherVar": _var("other")})},
        ),
    )

    result = runner.invoke(cli, ["run", "--dry-run", str(path)])

    assert result.exit_code == 1
    assert "Model 'MissingModel' is not registered" in result.output
    assert "Variable 'Runoff' is not defined in model profile 'KnownModel'" in result.output


def test_run_validates_model_resolution_before_runner_like_dry_run(tmp_path, monkeypatch):
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    (ref_root / "runoff.nc").touch()
    config = _base_config(
        tmp_path,
        simulation={
            "MissingModelCase": {
                "model": "MissingModel",
                "root_dir": str(sim_root),
                "tim_res": "Day",
                "grid_res": 0.5,
            },
            "MissingVariableCase": {
                "model": "KnownModel",
                "root_dir": str(sim_root),
                "tim_res": "Day",
                "grid_res": 0.5,
            },
        },
    )
    path = _write_config(tmp_path, config)
    _install_registry(
        monkeypatch,
        _Registry(
            {"DemoRef": _ref("DemoRef", str(ref_root))},
            {"KnownModel": _model("KnownModel", variables={"OtherVar": _var("other")})},
        ),
    )
    called = []
    monkeypatch.setattr("openbench.runner.local.run_evaluation", lambda *args, **kwargs: called.append(True))

    result = runner.invoke(cli, ["run", str(path)])

    assert result.exit_code == 1
    assert "Model 'MissingModel' is not registered" in result.output
    assert "Variable 'Runoff' is not defined in model profile 'KnownModel'" in result.output
    assert called == []


def test_check_comparison_only_skips_missing_simulation_roots(tmp_path, monkeypatch):
    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    (ref_root / "runoff.nc").touch()
    config = _base_config(
        tmp_path,
        simulation={
            "CaseA": {
                "model": "KnownModel",
                "root_dir": str(tmp_path / "missing-sim"),
                "tim_res": "Day",
                "grid_res": 0.5,
            }
        },
    )
    config["comparison"] = {"enabled": True, "items": ["HeatMap"]}
    path = _write_config(tmp_path, config)
    _install_registry(
        monkeypatch,
        _Registry(
            {"DemoRef": _ref("DemoRef", str(ref_root))},
            {"KnownModel": _model("KnownModel", variables={"Runoff": _var("runoff")})},
        ),
    )
    output_dir = tmp_path / "out" / "case"
    stem = "Runoff_ref_DemoRef_sim_CaseA"
    _write_fake_netcdf(output_dir / "metrics" / f"{stem}_bias.nc")
    _write_fake_netcdf(output_dir / "metrics" / f"{stem}_RMSE.nc")
    _write_fake_netcdf(output_dir / "metrics" / f"{stem}_correlation.nc")
    _write_fake_netcdf(output_dir / "scores" / f"{stem}_Overall_Score.nc")

    result = runner.invoke(cli, ["check", "--comparison-only", str(path)])

    assert result.exit_code == 0, result.output
    assert "Simulation root does not exist" not in result.output


def test_check_comparison_only_skips_missing_reference_roots(tmp_path, monkeypatch):
    config = _base_config(
        tmp_path,
        simulation={
            "CaseA": {
                "model": "KnownModel",
                "root_dir": str(tmp_path / "missing-sim"),
                "tim_res": "Day",
                "grid_res": 0.5,
            }
        },
    )
    config["comparison"] = {"enabled": True, "items": ["HeatMap"]}
    path = _write_config(tmp_path, config)
    _install_registry(
        monkeypatch,
        _Registry(
            {"DemoRef": _ref("DemoRef", str(tmp_path / "missing-ref"))},
            {"KnownModel": _model("KnownModel", variables={"Runoff": _var("runoff")})},
        ),
    )
    output_dir = tmp_path / "out" / "case"
    stem = "Runoff_ref_DemoRef_sim_CaseA"
    _write_fake_netcdf(output_dir / "metrics" / f"{stem}_bias.nc")
    _write_fake_netcdf(output_dir / "metrics" / f"{stem}_RMSE.nc")
    _write_fake_netcdf(output_dir / "metrics" / f"{stem}_correlation.nc")
    _write_fake_netcdf(output_dir / "scores" / f"{stem}_Overall_Score.nc")

    result = runner.invoke(cli, ["check", "--comparison-only", str(path)])

    assert result.exit_code == 0, result.output
    assert "Reference root does not exist" not in result.output
    assert "Simulation root does not exist" not in result.output


def test_check_comparison_only_checks_existing_outputs_like_run_dry_run(tmp_path, monkeypatch):
    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    (ref_root / "runoff.nc").touch()
    config = _base_config(
        tmp_path,
        simulation={
            "CaseA": {
                "model": "KnownModel",
                "root_dir": str(tmp_path / "missing-sim"),
                "tim_res": "Day",
                "grid_res": 0.5,
            }
        },
        project={"name": "comparison_missing_outputs"},
    )
    config["comparison"] = {"enabled": True, "items": ["HeatMap"]}
    path = _write_config(tmp_path, config)
    _install_registry(
        monkeypatch,
        _Registry(
            {"DemoRef": _ref("DemoRef", str(ref_root))},
            {"KnownModel": _model("KnownModel", variables={"Runoff": _var("runoff")})},
        ),
    )

    check_result = runner.invoke(cli, ["check", "--comparison-only", str(path)])
    dry_run_result = runner.invoke(cli, ["run", "--dry-run", "--comparison-only", str(path)])

    for result in (check_result, dry_run_result):
        assert result.exit_code == 1
        assert "missing prerequisite outputs" in result.output
        assert "Simulation root does not exist" not in result.output


def test_check_only_drawing_checks_existing_outputs_and_skips_raw_sim_root(tmp_path, monkeypatch):
    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    (ref_root / "runoff.nc").touch()
    config = _base_config(
        tmp_path,
        simulation={
            "CaseA": {
                "model": "KnownModel",
                "root_dir": str(tmp_path / "missing-sim"),
                "tim_res": "Day",
                "grid_res": 0.5,
            }
        },
        project={"name": "only_drawing_missing_outputs", "only_drawing": True},
    )
    config["metrics"] = ["bias"]
    config["scores"] = ["Overall_Score"]
    path = _write_config(tmp_path, config)
    _install_registry(
        monkeypatch,
        _Registry(
            {"DemoRef": _ref("DemoRef", str(ref_root))},
            {"KnownModel": _model("KnownModel", variables={"Runoff": _var("runoff")})},
        ),
    )

    check_result = runner.invoke(cli, ["check", str(path)])
    dry_run_result = runner.invoke(cli, ["run", "--dry-run", str(path)])

    for result in (check_result, dry_run_result):
        assert result.exit_code == 1
        assert "missing prerequisite outputs" in result.output
        assert "Simulation root does not exist" not in result.output


def test_check_comparison_only_skips_station_simulation_fulllist_preflight(tmp_path, monkeypatch):
    """comparison-only consumes existing outputs, so station simulation fulllists are not prerequisites."""
    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    (ref_root / "runoff.nc").touch()
    config = _base_config(
        tmp_path,
        simulation={
            "CaseA": {
                "model": "StationModel",
                "root_dir": str(tmp_path / "missing-sim"),
                "data_type": "stn",
                "tim_res": "Day",
                "grid_res": 0.5,
                "fulllist": "missing_sites.csv",
            }
        },
    )
    config["comparison"] = {"enabled": True, "items": ["HeatMap"]}
    path = _write_config(tmp_path, config)
    _install_registry(
        monkeypatch,
        _Registry(
            {"DemoRef": _ref("DemoRef", str(ref_root), data_type="grid")},
            {
                "StationModel": _model(
                    "StationModel",
                    data_type="stn",
                    variables={"Runoff": _var("runoff")},
                )
            },
        ),
    )
    output_dir = tmp_path / "out" / "case"
    filename = "Runoff_stn_DemoRef_CaseA_evaluations.csv"
    (output_dir / "metrics").mkdir(parents=True)
    (output_dir / "scores").mkdir()
    (output_dir / "metrics" / filename).write_text("station,bias,RMSE,correlation\nS1,1.0,1.0,1.0\n")
    (output_dir / "scores" / filename).write_text("station,Overall_Score\nS1,1.0\n")

    result = runner.invoke(cli, ["check", "--comparison-only", str(path)])

    assert result.exit_code == 0, result.output
    assert "Station fulllist does not exist" not in result.output
    assert "Simulation root does not exist" not in result.output


def test_check_detects_recursive_include(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("!include b.yaml\n")
    b.write_text("!include a.yaml\n")

    result = runner.invoke(cli, ["check", str(a)])

    assert result.exit_code == 1
    assert "Recursive !include detected" in result.output


def test_check_rejects_missing_requested_reference_resolution(tmp_path, monkeypatch):
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    mid_root = tmp_path / "Grid" / "MidRes"
    low_data = tmp_path / "Grid" / "LowRes" / "Water" / "Runoff" / "Demo"
    low_data.mkdir(parents=True)
    (low_data / "runoff.nc").touch()
    config = _base_config(
        tmp_path,
        reference={"data_root": str(mid_root), "Runoff": "DemoRef"},
    )
    path = _write_config(tmp_path, config)
    _install_registry(
        monkeypatch,
        _Registry(
            {
                "DemoRef": _ref(
                    "DemoRef",
                    str(mid_root),
                    var_map=_var("ro", sub_dir="Water/Runoff/Demo"),
                )
            },
            {"KnownModel": _model("KnownModel", variables={"Runoff": _var("runoff")})},
        ),
    )

    result = runner.invoke(cli, ["check", str(path)])

    assert result.exit_code == 1, result.output
    assert str(mid_root) in result.output
    assert str(tmp_path / "Grid" / "LowRes") not in result.output
    assert "Reference data path does not exist" in result.output


def test_check_reports_multi_reference_task_count_and_effective_root(tmp_path, monkeypatch):
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    (ref_root / "runoff.nc").touch()
    _touch_year_files(ref_root, sim_root)
    config = _base_config(
        tmp_path,
        reference={"Runoff": ["RefA", "RefB"]},
    )
    path = _write_config(tmp_path, config)
    _install_registry(
        monkeypatch,
        _Registry(
            {
                "RefA": _ref("RefA", str(ref_root)),
                "RefB": _ref("RefB", str(ref_root)),
            },
            {"KnownModel": _model("KnownModel", variables={"Runoff": _var("runoff")})},
        ),
    )

    result = runner.invoke(cli, ["check", str(path)])

    assert result.exit_code == 0, result.output
    assert "Evaluation tasks: 2" in result.output
    assert f"effective root: {ref_root}" in result.output


def test_check_strict_reference_flag_upgrades_default_provenance(tmp_path, monkeypatch):
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    (ref_root / "runoff.nc").touch()
    config = _base_config(tmp_path)
    path = _write_config(tmp_path, config)
    _install_registry(
        monkeypatch,
        _Registry(
            {
                "DemoRef": _ref(
                    "DemoRef",
                    str(ref_root),
                    provenance={"tim_res": "default", "grid_res": "nc"},
                )
            },
            {"KnownModel": _model("KnownModel", variables={"Runoff": _var("runoff")})},
        ),
    )

    result = runner.invoke(cli, ["check", "--strict-reference", str(path)])

    assert result.exit_code == 1
    assert "unconfirmed default" in result.output


def test_check_model_profile_variable_lookup_is_case_insensitive(tmp_path, monkeypatch):
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    (ref_root / "qle.nc").touch()
    _touch_year_files(ref_root, sim_root)
    config = _base_config(
        tmp_path,
        variables=["latent_heat"],
        reference={"latent_heat": "DemoRef"},
        simulation={
            "CaseA": {
                "model": "KnownModel",
                "root_dir": str(sim_root),
                "tim_res": "Day",
                "grid_res": 0.5,
            }
        },
    )
    path = _write_config(tmp_path, config)
    _install_registry(
        monkeypatch,
        _Registry(
            {"DemoRef": _ref("DemoRef", str(ref_root), variable="Latent_Heat", var_map=_var("Qle"))},
            {"KnownModel": _model("KnownModel", variables={"Latent_Heat": _var("Qle")})},
        ),
    )

    result = runner.invoke(cli, ["check", str(path)])

    assert result.exit_code == 0, result.output
    assert "Variable 'latent_heat' is not defined" not in result.output


def test_check_rejects_comparison_only_with_only_drawing(tmp_path, monkeypatch):
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    (ref_root / "runoff.nc").touch()
    config = _base_config(
        tmp_path,
        project={"only_drawing": True},
        simulation={
            "CaseA": {
                "model": "KnownModel",
                "root_dir": str(sim_root),
                "tim_res": "Day",
                "grid_res": 0.5,
            }
        },
    )
    config["comparison"] = {"enabled": True, "items": ["Mean"]}
    path = _write_config(tmp_path, config)
    _install_registry(
        monkeypatch,
        _Registry(
            {"DemoRef": _ref("DemoRef", str(ref_root))},
            {"KnownModel": _model("KnownModel", variables={"Runoff": _var("ro")})},
        ),
    )

    result = runner.invoke(cli, ["check", "--comparison-only", str(path)])

    assert result.exit_code == 1
    assert "--comparison-only conflicts with project.only_drawing=true" in result.output


def test_resolve_variable_filters_reports_duplicates_and_unknowns_together():
    from openbench.cli._names import resolve_variable_filters

    try:
        resolve_variable_filters(("runoff", "Runoff", "Missing"), ["Runoff"])
    except Exception as exc:
        message = str(exc)
    else:
        raise AssertionError("expected ClickException")

    assert "must be unique" in message
    assert "not in evaluation.variables" in message


def test_load_config_accepts_project_dask_block(tmp_path):
    from openbench.config.loader import load_config

    config = _base_config(
        tmp_path,
        project={
            "dask": {
                "enabled": True,
                "n_workers": 2,
                "threads_per_worker": 1,
                "processes": False,
                "memory_limit": "1GB",
                "dashboard_address": ":0",
                "local_directory": str(tmp_path / "dask"),
            }
        },
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    loaded = load_config(path)

    assert loaded.project.dask.enabled is True
    assert loaded.project.dask.n_workers == 2
    assert loaded.project.dask.processes is False


def test_load_config_rejects_invalid_project_dask_block(tmp_path):
    from openbench.config.loader import ConfigError, load_config

    config = _base_config(tmp_path, project={"dask": {"enabled": "yes"}})
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ConfigError, match="project.dask.enabled must be a boolean"):
        load_config(path)


def test_load_config_accepts_project_io_block(tmp_path):
    from openbench.config.loader import load_config

    config = _base_config(
        tmp_path,
        project={
            "io": {
                "netcdf_compression": True,
                "netcdf_compression_level": 4,
                "mfdataset_batch_size": 25,
                "mfdataset_auto_batch_min_files": 100,
                "mfdataset_auto_batch_min_size_mb": 512,
                "mfdataset_auto_batch_min_size": 5,
                "mfdataset_auto_batch_max_size": 80,
                "mfdataset_auto_batch_memory_fraction": 0.5,
            }
        },
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    loaded = load_config(path)

    assert loaded.project.io.netcdf_compression is True
    assert loaded.project.io.netcdf_compression_level == 4
    assert loaded.project.io.mfdataset_batch_size == 25
    assert loaded.project.io.mfdataset_auto_batch_min_files == 100
    assert loaded.project.io.mfdataset_auto_batch_min_size_mb == 512
    assert loaded.project.io.mfdataset_auto_batch_memory_fraction == 0.5


def test_load_config_accepts_project_io_auto_mfdataset_batch_size(tmp_path):
    from openbench.config.loader import load_config

    config = _base_config(tmp_path, project={"io": {"mfdataset_batch_size": "auto"}})
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    loaded = load_config(path)

    assert loaded.project.io.mfdataset_batch_size is None


def test_load_config_rejects_invalid_project_io_block(tmp_path):
    from openbench.config.loader import ConfigError, load_config

    config = _base_config(tmp_path, project={"io": {"netcdf_compression_level": 12}})
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ConfigError, match="project.io.netcdf_compression_level"):
        load_config(path)


def test_check_rejects_explicit_empty_metrics_and_scores(tmp_path, monkeypatch):
    from openbench.config.loader import ConfigError, load_config

    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    (ref_root / "runoff.nc").touch()
    config = _base_config(tmp_path)
    config["metrics"] = []
    config["scores"] = []
    path = _write_config(tmp_path, config)
    _install_registry(
        monkeypatch,
        _Registry(
            {"DemoRef": _ref("DemoRef", str(ref_root))},
            {"KnownModel": _model("KnownModel", variables={"Runoff": _var("runoff")})},
        ),
    )

    result = runner.invoke(cli, ["check", str(path)])

    assert result.exit_code == 1
    assert "metrics and scores cannot both be empty" in result.output

    with pytest.raises(ConfigError, match="metrics and scores cannot both be empty"):
        load_config(path)


def test_check_reports_reference_file_named_by_the_catalog_but_absent(tmp_path, monkeypatch):
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    _touch_year_files(sim_root)
    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    (ref_root / "CN05.1_Tm_1961_2021_daily_0P25.nc").touch()
    var_map = _var("tm")
    var_map.prefix, var_map.suffix = "CN05.1_Tm_", "_2021_daily_0P25"
    ref = _ref("CN05", str(ref_root), data_groupby="Single", years=[1961, 2021], var_map=var_map)
    path = _write_config(tmp_path, _base_config(tmp_path, reference={"Runoff": "CN05"}))
    _install_registry(
        monkeypatch,
        _Registry({"CN05": ref}, {"KnownModel": _model("KnownModel", variables={"Runoff": _var("runoff")})}),
    )

    result = runner.invoke(cli, ["check", str(path)])

    assert result.exit_code == 1, result.output
    assert "Reference data file not found" in result.output
    assert "CN05.1_Tm__2021_daily_0P25.nc" in result.output

    var_map.prefix = "CN05.1_Tm_1961"
    result = runner.invoke(cli, ["check", str(path)])

    assert result.exit_code == 0, result.output
    assert "data file" not in result.output


def test_check_reports_simulation_files_named_unlike_the_config(tmp_path, monkeypatch):
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    (sim_root / "run_2001.nc").touch()
    (sim_root / "run_2002.nc").touch()
    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    _touch_year_files(ref_root)
    simulation = {
        "CaseA": {
            "model": "KnownModel",
            "root_dir": str(sim_root),
            "tim_res": "Day",
            "grid_res": 0.5,
            "prefix": "case_",
        }
    }
    path = _write_config(tmp_path, _base_config(tmp_path, simulation=simulation))
    _install_registry(
        monkeypatch,
        _Registry(
            {"DemoRef": _ref("DemoRef", str(ref_root))},
            {"KnownModel": _model("KnownModel", variables={"Runoff": _var("runoff")})},
        ),
    )

    result = runner.invoke(cli, ["check", str(path)])

    assert result.exit_code == 1, result.output
    assert "Simulation data files not found" in result.output
    assert "case_<year>*.nc for 2001–2002" in result.output

    simulation["CaseA"]["prefix"] = "run_"
    path = _write_config(tmp_path, _base_config(tmp_path, simulation=simulation))
    result = runner.invoke(cli, ["check", str(path)])

    assert result.exit_code == 0, result.output


def test_check_only_warns_for_missing_files_of_computed_variables(tmp_path, monkeypatch):
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    (sim_root / "inputs.nc").touch()
    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    _touch_year_files(ref_root)
    simulation = {
        "CaseA": {
            "model": "KnownModel",
            "root_dir": str(sim_root),
            "tim_res": "Day",
            "grid_res": 0.5,
            "variables": {"Runoff": {"varname": "runoff", "varunit": "mm", "compute": "runoff = a + b"}},
        }
    }
    path = _write_config(tmp_path, _base_config(tmp_path, simulation=simulation))
    _install_registry(
        monkeypatch,
        _Registry(
            {"DemoRef": _ref("DemoRef", str(ref_root))},
            {"KnownModel": _model("KnownModel", variables={"Runoff": _var("runoff")})},
        ),
    )

    result = runner.invoke(cli, ["check", str(path)])

    assert result.exit_code == 0, result.output
    assert "preprocessing will look for files holding the compute inputs" in result.output


def test_simulation_checks_project_endpoints_despite_inferred_coverage(tmp_path, monkeypatch):
    from openbench.cli import check as check_module
    from openbench.config import load_config

    sim = tmp_path / "sim"
    sim.mkdir()
    (sim / "2002.nc").touch()
    cfg = load_config(str(_write_config(tmp_path, _base_config(tmp_path))))
    monkeypatch.setattr(check_module, "_simulation_data_years", lambda *args, **kwargs: [2002, 2002])
    result = check_module._simulation_findings(
        cfg, _Registry({}, {"KnownModel": _model(variables={"Runoff": _var()})}), comparison_only=False
    )
    assert any("2001" in error and "Runoff" in error for error in result["CaseA"]["errors"])


def test_check_missing_middle_year_reuses_inventory(tmp_path, monkeypatch):
    from openbench.cli import check as check_module
    from openbench.config import load_config
    from openbench.data import file_lookup

    sim = tmp_path / "sim"
    sim.mkdir()
    _touch_year_files(sim, years=(2001, 2003))
    cfg = load_config(str(_write_config(tmp_path, _base_config(tmp_path, project={"years": [2001, 2003]}))))
    walks = []
    original = file_lookup.netcdf_inventory

    def inventory(directory):
        walks.append(directory)
        return original(directory)

    monkeypatch.setattr(file_lookup, "netcdf_inventory", inventory)
    monkeypatch.setattr(check_module, "_simulation_data_years", lambda *args, **kwargs: [2001, 2003])
    findings = check_module._simulation_findings(
        cfg, _Registry({}, {"KnownModel": _model(variables={"Runoff": _var()})}), comparison_only=False
    )
    assert any("2002" in error for error in findings["CaseA"]["errors"])
    assert len(walks) == 1


@pytest.mark.parametrize("strict", [False, True])
def test_grid_sim_station_ref_checks_only_station_overlap(tmp_path, monkeypatch, strict):
    from openbench.cli import check as check_module
    from openbench.config import load_config

    sim = tmp_path / "sim"
    sim.mkdir()
    _touch_year_files(sim, years=(2002,))
    ref = _ref("DemoRef", str(tmp_path), data_type="stn", years=[2002, 2002])
    cfg = load_config(
        str(
            _write_config(
                tmp_path, _base_config(tmp_path, project={"time_alignment": "strict" if strict else "intersection"})
            )
        )
    )
    monkeypatch.setattr(check_module, "_simulation_data_years", lambda *args, **kwargs: [2002, 2002])
    findings = check_module._simulation_findings(
        cfg, _Registry({"DemoRef": ref}, {"KnownModel": _model(variables={"Runoff": _var()})}), comparison_only=False
    )
    assert findings["CaseA"]["errors"] == []


@pytest.mark.parametrize("columns", ["SYEAR,EYEAR", "use_syear,use_eyear", "sim_syear,sim_eyear"])
def test_grid_ref_station_sim_uses_station_list_years(tmp_path, columns):
    from openbench.cli.check import _reference_file_findings
    from openbench.config import load_config

    data = tmp_path / "reference"
    sim = tmp_path / "sim"
    data.mkdir()
    sim.mkdir()
    _touch_year_files(data, years=(2002,))
    (sim / "stations.csv").write_text(f"ID,{columns},LON,LAT\nS1,2002,2002,10,20\n")
    cfg = load_config(
        str(
            _write_config(
                tmp_path,
                _base_config(
                    tmp_path,
                    simulation={
                        "CaseA": {
                            "model": "KnownModel",
                            "data_type": "stn",
                            "root_dir": str(sim),
                            "fulllist": "stations.csv",
                        }
                    },
                ),
            )
        )
    )
    ref = _ref("DemoRef", str(data), years=[2001, 2002])
    resolved = SimpleNamespace(
        status="ok",
        ref_ds=ref,
        var_map=ref.variables["Runoff"],
        var_name="Runoff",
        source_name="DemoRef",
        resolved_name="DemoRef",
    )
    assert _reference_file_findings(cfg, resolved, _Registry({"DemoRef": ref})) == ([], [])


@pytest.mark.parametrize("columns", ["SYEAR,EYEAR", "use_syear,use_eyear", "ref_syear,ref_eyear"])
def test_grid_sim_generated_station_ref_uses_station_years(tmp_path, monkeypatch, columns):
    from openbench.cli import check as check_module
    from openbench.config import load_config

    sim = tmp_path / "sim"
    sim.mkdir()
    _touch_year_files(sim, years=(2002,))
    (tmp_path / "stations.csv").write_text(f"ID,{columns},LON,LAT\nS1,2002,2002,10,20\n")
    ref = _ref("DemoRef", str(tmp_path), data_type="stn", years=[2001, 2002], fulllist="stations.csv")
    cfg = load_config(str(_write_config(tmp_path, _base_config(tmp_path))))
    monkeypatch.setattr(check_module, "_simulation_data_years", lambda *args, **kwargs: [2002, 2002])
    findings = check_module._simulation_findings(
        cfg, _Registry({"DemoRef": ref}, {"KnownModel": _model(variables={"Runoff": _var()})}), comparison_only=False
    )
    assert findings["CaseA"]["errors"] == []


@pytest.mark.parametrize("strict, missing", [(False, False), (True, True)])
def test_reference_year_directories_and_strict_endpoint_checks(tmp_path, strict, missing):
    from openbench.cli.check import _reference_data_findings, _reference_file_findings
    from openbench.config import load_config

    data = tmp_path / "reference"
    (data / "2002" / "01").mkdir(parents=True)
    (data / "2002" / "01" / "runoff.nc").touch()
    cfg = load_config(
        str(
            _write_config(
                tmp_path, _base_config(tmp_path, project={"time_alignment": "strict" if strict else "intersection"})
            )
        )
    )
    mapping = _var()
    mapping.prefix = "runoff"
    ref = _ref("DemoRef", str(data), years=[2002, 2002], var_map=mapping)
    resolved = SimpleNamespace(
        status="ok", ref_ds=ref, var_map=mapping, var_name="Runoff", source_name="DemoRef", resolved_name="DemoRef"
    )
    assert _reference_data_findings(cfg, resolved)[0:2] == ([], [])
    errors, warnings = _reference_file_findings(cfg, resolved)
    assert bool(errors) == missing
    assert not warnings
    if missing:
        assert "2001" in errors[0]


@pytest.mark.parametrize("override_exists", [True, False])
def test_reference_root_override_controls_directory_checks(tmp_path, override_exists):
    from openbench.cli.check import _reference_data_findings
    from openbench.config import load_config
    from openbench.data.registry.schema import ReferenceDataset, VariableMapping

    base, override = tmp_path / "base", tmp_path / "override"
    (override if override_exists else base).mkdir()
    ((override if override_exists else base) / "data.nc").touch()
    cfg = load_config(
        str(
            _write_config(
                tmp_path,
                _base_config(
                    tmp_path,
                    reference={
                        "data_root": str(base),
                        "Runoff": "DemoRef",
                        "overrides": {"DemoRef": {"root_dir": str(override)}},
                    },
                ),
            )
        )
    )
    mapping = VariableMapping(varname="runoff", varunit="mm day-1")
    ref = ReferenceDataset(
        name="DemoRef",
        description="",
        category="Water",
        data_type="grid",
        tim_res="Day",
        data_groupby="Year",
        timezone=0,
        years=[2001, 2002],
        root_dir=str(base),
        variables={"Runoff": mapping},
    )
    resolved = SimpleNamespace(
        status="ok", ref_ds=ref, var_map=mapping, var_name="Runoff", source_name="DemoRef", resolved_name="DemoRef"
    )
    errors, warnings, info = _reference_data_findings(cfg, resolved)
    assert bool(errors) != override_exists
    assert str(override) in " ".join(errors + info)
    assert not warnings


def test_empty_inline_subdir_and_nested_simulation_use_runtime_layout(tmp_path):
    from openbench.cli.check import _effective_sim_values, _simulation_file_findings
    from openbench.config import load_config

    cfg = load_config(
        str(
            _write_config(
                tmp_path,
                _base_config(
                    tmp_path,
                    simulation={
                        "CaseA": {
                            "model": "KnownModel",
                            "root_dir": str(tmp_path),
                            "variables": {"Runoff": {"sub_dir": ""}},
                        }
                    },
                ),
            )
        )
    )
    profile = _model(variables={"Runoff": _var(sub_dir="inherited")})
    values = _effective_sim_values(cfg.simulation["CaseA"], profile, "Runoff")
    assert values["dir"] == str(tmp_path)
    assert values["sub_dir"] == ""
    (tmp_path / "2001" / "01").mkdir(parents=True)
    (tmp_path / "2001" / "01" / "runoff.nc").touch()
    layout = {
        "dir": str(tmp_path),
        "prefix": "runoff",
        "suffix": "",
        "data_groupby": "Year",
        "prefix_fallback": None,
        "compute": None,
    }
    assert _simulation_file_findings(layout, (2001,)) == ([], [])


def test_check_help_discloses_every_required_year():
    result = runner.invoke(cli, ["check", "--help"])
    assert result.exit_code == 0
    assert "every required evaluation year" in result.output


def test_check_and_init_share_station_dataset_validation(tmp_path):
    import click

    from openbench.cli.check import _reference_data_findings
    from openbench.cli.init_cmd import _validate_selected_reference_data
    from openbench.config import load_config
    from openbench.data.registry.schema import ReferenceDataset, StationMatchingConfig, VariableMapping

    mapping = VariableMapping(varname="discharge", varunit="m3 s-1", sub_dir="unused_station_subdir")
    ref = ReferenceDataset(
        name="DemoRef",
        description="",
        category="Water",
        data_type="stn",
        tim_res="Day",
        data_groupby="Single",
        timezone=0,
        years=[2001, 2002],
        root_dir=str(tmp_path),
        variables={"Runoff": mapping},
        station_matching=StationMatchingConfig(dataset_file="daily_full.nc"),
    )
    cfg = load_config(str(_write_config(tmp_path, _base_config(tmp_path))))
    resolved = SimpleNamespace(
        status="ok", ref_ds=ref, var_map=mapping, var_name="Runoff", source_name="DemoRef", resolved_name="DemoRef"
    )
    assert "daily_full.nc" in _reference_data_findings(cfg, resolved)[0][0]
    with pytest.raises(click.ClickException, match="daily_full.nc"):
        _validate_selected_reference_data({"Runoff": ref})
    (tmp_path / "daily_dist.nc").touch()
    assert _reference_data_findings(cfg, resolved)[0:2] == ([], [])
    _validate_selected_reference_data({"Runoff": ref})


def test_reference_directory_keeps_years_at_the_root(tmp_path):
    from openbench.config.adapter import _find_nc_dir

    for year in (2001, 2002):
        (tmp_path / str(year)).mkdir()
        (tmp_path / str(year) / f"ref_{year}.nc").touch()
    (tmp_path / "ancillary").mkdir()
    (tmp_path / "ancillary" / "metadata.nc").touch()
    assert _find_nc_dir(str(tmp_path), str(tmp_path), None) == str(tmp_path)


def test_missing_year_messages_list_ranges_once_per_directory():
    from openbench.cli.check import data_file_findings

    errors, warnings = data_file_findings(
        "Reference", "/data/E", prefix="E_", suffix="_v1", data_groupby="Year", years=list(range(2001, 2021))
    )

    assert warnings == []
    assert errors == [
        "Reference data files not found in /data/E: E_<year>*_v1.nc for 2001–2020 "
        "(prefix 'E_', suffix '_v1', data_groupby Year)"
    ]


def test_empty_grid_simulation_directory_is_an_error(tmp_path, monkeypatch):
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    _touch_year_files(ref_root)
    path = _write_config(tmp_path, _base_config(tmp_path))
    _install_registry(
        monkeypatch,
        _Registry(
            {"DemoRef": _ref("DemoRef", str(ref_root))},
            {"KnownModel": _model("KnownModel", variables={"Runoff": _var("runoff")})},
        ),
    )

    result = runner.invoke(cli, ["check", str(path)])

    assert result.exit_code == 1, result.output
    assert "Simulation data directory has no NetCDF files" in result.output


def test_check_resolves_each_reference_directory_once(tmp_path, monkeypatch):
    import openbench.cli.check as check_module

    ref_root = tmp_path / "ref"
    ref_root.mkdir()
    _touch_year_files(ref_root)
    simulation = {}
    for index in range(3):
        root = tmp_path / f"sim{index}"
        root.mkdir()
        _touch_year_files(root)
        simulation[f"Case{index}"] = {"model": "KnownModel", "root_dir": str(root), "tim_res": "Day", "grid_res": 0.5}
    path = _write_config(tmp_path, _base_config(tmp_path, simulation=simulation))
    _install_registry(
        monkeypatch,
        _Registry(
            {"DemoRef": _ref("DemoRef", str(ref_root))},
            {"KnownModel": _model("KnownModel", variables={"Runoff": _var("runoff")})},
        ),
    )
    calls = []
    resolve = check_module._resolve_effective_reference

    def counted(cfg, resolved_ref):
        calls.append(resolved_ref.resolved_name)
        return resolve(cfg, resolved_ref)

    monkeypatch.setattr(check_module, "_resolve_effective_reference", counted)

    result = runner.invoke(cli, ["check", str(path)])

    assert result.exit_code == 0, result.output
    assert calls == ["DemoRef"]
    assert check_module._EFFECTIVE_REFERENCES is None  # the cache lives for one check only


def test_reference_naming_error_says_where_the_value_was_set(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from openbench.cli.check import _naming_origin_hint

    cfg = SimpleNamespace(reference=SimpleNamespace(overrides={}))
    resolved = SimpleNamespace(
        source_name="CN05.1_MidRes", resolved_name="CN05.1_MidRes", var_name="Surface_Air_Temperature"
    )
    stale = SimpleNamespace(
        prefix="CN05.1_Tm_", suffix="_2021_daily_0P25", sub_dir="Meteo/Surface_Air_Temperature/CN05.1"
    )
    current = SimpleNamespace(
        prefix="CN05.1_Tm_1961", suffix="_2021_daily_0P25", sub_dir="Meteo/Surface_Air_Temperature/CN05.1"
    )

    assert "registry reset CN05.1_MidRes" in _naming_origin_hint(cfg, resolved, stale)
    assert _naming_origin_hint(cfg, resolved, current) is None
    cfg.reference.overrides = {"CN05.1_MidRes": {"variables": {"Surface_Air_Temperature": {"prefix": "CN05.1_Tm_"}}}}
    assert "reference.overrides" in _naming_origin_hint(cfg, resolved, stale)

    # A separate overlay file is not edited by `registry reset`; name the file instead.
    cfg.reference.overrides = {}
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("OPENBENCH_HOME", raising=False)
    legacy = tmp_path / ".openbench" / "references" / "legacy.yaml"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("CN05.1_MidRes:\n  variables:\n    Surface_Air_Temperature:\n      prefix: CN05.1_Tm_\n")
    hint = _naming_origin_hint(cfg, resolved, stale)
    assert str(legacy) in hint and "registry reset" not in hint


def test_single_file_inside_a_dated_folder_is_found(tmp_path):
    from openbench.config.adapter import _find_nc_dir

    (tmp_path / "2024-05").mkdir()
    (tmp_path / "2024-05" / "all_years.nc").touch()

    assert _find_nc_dir(str(tmp_path), str(tmp_path), None, "Single") == str(tmp_path / "2024-05")
    assert _find_nc_dir(str(tmp_path), str(tmp_path), None, "Year") == str(tmp_path)


def test_reference_root_with_tilde_is_expanded_for_check_and_runtime(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from openbench.config.adapter import reference_data_dir

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    (tmp_path / "ref").mkdir()
    cfg = SimpleNamespace(reference=SimpleNamespace(data_root=None))
    ref = SimpleNamespace(data_type="grid", root_dir="~/ref", data_groupby="Year")

    assert reference_data_dir(cfg, ref, SimpleNamespace(sub_dir=None), None) == (str(tmp_path / "ref"),) * 2


def test_only_drawing_skips_raw_reference_checks(tmp_path, monkeypatch):
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    ref_root = tmp_path / "ref"
    ref_root.mkdir()  # empty raw directory: drawing reads existing results only
    path = _write_config(tmp_path, _base_config(tmp_path, project={"only_drawing": True}))
    _install_registry(
        monkeypatch,
        _Registry(
            {"DemoRef": _ref("DemoRef", str(ref_root))},
            {"KnownModel": _model("KnownModel", variables={"Runoff": _var("runoff")})},
        ),
    )

    result = runner.invoke(cli, ["check", str(path)])

    assert "has no NetCDF files" not in result.output
