from pathlib import Path

import yaml

from openbench.gui.config_manager import ConfigManager


def _config():
    return {
        "general": {
            "basename": "demo",
            "basedir": "/tmp/openbench-out",
            "statistics": True,
        },
        "statistics": {
            "Mean": True,
            "Median": False,
            "Hellinger_Distance": True,
        },
        "evaluation_items": {"Runoff": True},
        "metrics": {"bias": True},
        "scores": {},
        "comparisons": {},
    }


def test_generate_main_nml_references_case_local_support_namelists_for_wheel_layout(tmp_path):
    site_packages = tmp_path / "venv" / "site-packages"
    package = site_packages / "openbench"
    (package / "cli").mkdir(parents=True)
    (package / "cli" / "main.py").write_text("# marker\n")
    (package / "data" / "registry").mkdir(parents=True)

    output_dir = tmp_path / "case-output" / "demo"
    cm = ConfigManager()
    main = yaml.safe_load(cm.generate_main_nml(_config(), str(site_packages), str(output_dir)))

    assert main["general"]["statistics_nml"] == str(output_dir / "nml" / "stats-demo.yaml")
    assert main["general"]["figure_nml"] == str(output_dir / "nml" / "fig-demo.yaml")
    assert "site-packages/nml/nml-yaml" not in main["general"]["statistics_nml"]
    assert "site-packages/nml/nml-yaml" not in main["general"]["figure_nml"]


def test_write_legacy_support_namelists_creates_files_referenced_by_main(tmp_path):
    output_dir = tmp_path / "case-output" / "demo"
    nml_dir = output_dir / "nml"
    cm = ConfigManager()

    main = yaml.safe_load(cm.generate_main_nml(_config(), str(tmp_path), str(output_dir)))
    written = cm.write_legacy_support_namelists(_config(), str(nml_dir))

    stats_path = Path(main["general"]["statistics_nml"])
    figure_path = Path(main["general"]["figure_nml"])

    assert stats_path.exists()
    assert figure_path.exists()
    assert written == {"statistics": str(stats_path), "figure": str(figure_path)}

    stats = yaml.safe_load(stats_path.read_text(encoding="utf-8"))
    assert sorted(k for k in stats if k != "general") == ["Hellinger_Distance", "Mean"]
    assert "Median" not in stats

    figure = yaml.safe_load(figure_path.read_text(encoding="utf-8"))
    assert "Comparison" in figure
    assert "Statistic" in figure


def _runnable_config(tmp_path):
    sim_root = tmp_path / "sim"
    sim_root.mkdir(exist_ok=True)
    return {
        "general": {
            "basename": "demo",
            "basedir": str(tmp_path / "out"),
            "syear": 2001,
            "eyear": 2002,
            "compare_tim_res": "Month",
            "compare_grid_res": 0.5,
            "comparison": False,
            "statistics": False,
            "IGBP_groupby": False,
            "PFT_groupby": False,
            "Climate_zone_groupby": False,
        },
        "evaluation_items": {"Evapotranspiration": True},
        "ref_data": {
            "general": {
                "data_root": "Reference",
                "Evapotranspiration_ref_source": "ET_Xu_etal_2025_LowRes",
            }
        },
        "sim_data": {
            "general": {"Evapotranspiration_sim_source": ["CaseA"]},
            "source_configs": {
                "CaseA": {
                    "general": {
                        "model_namelist": "CoLM2024",
                        "root_dir": "Simulation/CaseA",
                        "data_type": "grid",
                        "tim_res": "Month",
                        "grid_res": 0.5,
                    }
                }
            },
        },
        "metrics": {"bias": True},
        "scores": {},
        "comparisons": {},
        "statistics": {},
    }


def test_generate_config_yaml_preserves_exact_reference_source_names(tmp_path):
    data = yaml.safe_load(ConfigManager().generate_config_yaml(_runnable_config(tmp_path)))

    assert data["reference"]["Evapotranspiration"] == "ET_Xu_etal_2025_LowRes"


def test_generate_config_yaml_preserves_gui_default_min_year_and_none_weight(tmp_path):
    config = _runnable_config(tmp_path)
    for key in (
        "compare_tim_res",
        "compare_grid_res",
        "compare_tzone",
        "min_year",
        "num_cores",
        "weight",
        "IGBP_groupby",
        "PFT_groupby",
        "Climate_zone_groupby",
    ):
        config["general"].pop(key, None)

    data = yaml.safe_load(ConfigManager().generate_config_yaml(config))

    assert data["project"]["min_year_threshold"] == 1.0
    assert data["project"]["weight"] == "none"
    assert data["project"]["tim_res"] == "month"
    assert data["project"]["grid_res"] == 2.0
    assert data["project"]["timezone"] == 0.0
    assert data["project"]["num_cores"] == 4
    assert data["project"]["IGBP_groupby"] is True
    assert data["project"]["PFT_groupby"] is True
    assert data["project"]["climate_zone_groupby"] is True


def test_generate_main_nml_accepts_null_weight(tmp_path):
    config = _runnable_config(tmp_path)
    config["general"]["weight"] = None

    data = yaml.safe_load(ConfigManager().generate_main_nml(config, str(tmp_path), str(tmp_path / "out" / "demo")))

    assert data["general"]["weight"] == "None"


def test_generated_gui_config_loads_with_gui_default_min_year_and_none_weight(tmp_path):
    from openbench.config.loader import load_config

    config_path = tmp_path / "openbench.yaml"
    config_path.write_text(ConfigManager().generate_config_yaml(_runnable_config(tmp_path)), encoding="utf-8")

    loaded = load_config(config_path)

    assert loaded.project.min_year_threshold == 1.0
    assert loaded.project.weight == "none"


def test_export_all_uses_actual_case_output_dir_for_project_output_dir(tmp_path):
    from openbench.config.loader import load_config

    config = _runnable_config(tmp_path)
    config["general"]["basedir"] = "./output"
    case_dir = tmp_path / "OpenBench" / "output" / "demo"

    files = ConfigManager().export_all(config, str(case_dir), openbench_root=str(tmp_path / "OpenBench"))

    exported = yaml.safe_load(Path(files["config"]).read_text(encoding="utf-8"))
    assert exported["project"]["output_dir"] == str(case_dir.parent)
    assert load_config(files["config"]).project.output_dir == str(case_dir.parent)


def test_preview_export_load_and_runner_output_dir_stay_aligned(tmp_path):
    from openbench.config.loader import load_config

    config = _runnable_config(tmp_path)
    config["general"]["basedir"] = "./output"
    case_dir = tmp_path / "OpenBench" / "output" / "demo"
    manager = ConfigManager()

    preview_yaml = yaml.safe_load(manager.generate_config_yaml(config, case_output_dir=str(case_dir)))
    files = manager.export_all(config, str(case_dir), openbench_root=str(tmp_path / "OpenBench"))
    exported_yaml = yaml.safe_load(Path(files["config"]).read_text(encoding="utf-8"))
    loaded = load_config(files["config"])

    assert preview_yaml["project"]["output_dir"] == str(case_dir.parent)
    assert exported_yaml["project"]["output_dir"] == preview_yaml["project"]["output_dir"]
    assert loaded.project.output_dir == str(case_dir.parent)
    assert Path(loaded.project.output_dir) / loaded.project.name == case_dir


def test_validate_rejects_missing_simulation_mapping_and_config(tmp_path):
    config = _runnable_config(tmp_path)
    config["sim_data"] = {"general": {}, "source_configs": {}}

    errors = ConfigManager().validate(config)

    assert "Simulation data source required for Evapotranspiration" in errors

    config = _runnable_config(tmp_path)
    config["sim_data"]["source_configs"] = {}

    errors = ConfigManager().validate(config)

    assert "Simulation source 'CaseA' is missing configuration" in errors


def test_validate_requires_selected_items_when_comparison_or_statistics_enabled(tmp_path):
    config = _runnable_config(tmp_path)
    config["general"]["comparison"] = True
    config["general"]["statistics"] = True

    errors = ConfigManager().validate(config)

    assert "At least one comparison item must be selected when comparison is enabled" in errors
    assert "At least one statistics item must be selected when statistics is enabled" in errors


def test_generate_config_yaml_does_not_enable_empty_comparison_or_statistics(tmp_path):
    config = _runnable_config(tmp_path)
    config["general"]["comparison"] = True
    config["general"]["statistics"] = True

    data = yaml.safe_load(ConfigManager().generate_config_yaml(config))

    assert "comparison" not in data
    assert "statistics" not in data


def test_generate_config_yaml_remote_case_dir_overrides_output_and_transforms_paths(tmp_path):
    config = _runnable_config(tmp_path)

    def to_remote(path: str) -> str:
        return f"/remote/project/{path.strip('/')}"

    data = yaml.safe_load(
        ConfigManager().generate_config_yaml(
            config,
            case_output_dir="/remote/output/demo",
            path_transform=to_remote,
        )
    )

    assert data["project"]["output_dir"] == "/remote/output"
    assert data["reference"]["data_root"] == "/remote/project/Reference"
    assert data["simulation"]["CaseA"]["root_dir"] == "/remote/project/Simulation/CaseA"


def test_unified_config_converts_to_gui_internal_shape():
    unified = {
        "project": {
            "name": "demo",
            "output_dir": "/out",
            "years": [2001, 2002],
            "tim_res": "Month",
            "grid_res": 0.5,
            "timezone": 8,
            "weight": None,
            "IGBP_groupby": True,
            "climate_zone_groupby": True,
        },
        "evaluation": {"variables": ["Evapotranspiration"]},
        "reference": {
            "data_root": "/ref",
            "Evapotranspiration": ["GLEAM", "ERA5LAND"],
        },
        "simulation": {
            "_defaults": {"data_type": "grid", "tim_res": "Month"},
            "CaseA": {"model": "CoLM2024", "root_dir": "/sim/CaseA", "grid_res": 0.5},
        },
        "metrics": ["bias"],
        "scores": ["Overall_Score"],
        "comparison": {"enabled": True, "items": ["Taylor_Diagram"]},
        "statistics": {"enabled": True, "items": ["ANOVA"]},
    }

    gui_config = ConfigManager().unified_to_gui_config(unified)

    assert gui_config["general"]["basename"] == "demo"
    assert gui_config["general"]["basedir"] == "/out"
    assert gui_config["general"]["syear"] == 2001
    assert gui_config["general"]["eyear"] == 2002
    assert gui_config["general"]["weight"] == "none"
    assert gui_config["general"]["Climate_zone_groupby"] is True
    assert gui_config["evaluation_items"] == {"Evapotranspiration": True}
    assert gui_config["ref_data"]["general"]["data_root"] == "/ref"
    assert gui_config["ref_data"]["general"]["Evapotranspiration_ref_source"] == ["GLEAM", "ERA5LAND"]
    assert gui_config["sim_data"]["general"]["Evapotranspiration_sim_source"] == ["CaseA"]
    assert gui_config["sim_data"]["source_configs"]["CaseA"]["general"]["model"] == "CoLM2024"
    assert gui_config["metrics"] == {"bias": True}
    assert gui_config["scores"] == {"Overall_Score": True}
    assert gui_config["comparisons"] == {"Taylor_Diagram": True}
    assert gui_config["statistics"] == {"ANOVA": True}


def test_generate_config_yaml_preserves_simulation_variables_and_fulllist(tmp_path):
    config = _runnable_config(tmp_path)
    config["evaluation_items"] = {"Runoff": True}
    config["sim_data"]["general"] = {"Runoff_sim_source": ["CaseA"]}
    config["sim_data"]["source_configs"]["CaseA"] = {
        "general": {
            "model_namelist": "CoLM2024",
            "root_dir": "/sim",
            "data_type": "stn",
            "fulllist": "/sim/list.csv",
            "tim_res": "Day",
        },
        "variables": {"Runoff": {"varname": "q", "varunit": "m3 s-1"}},
    }

    data = yaml.safe_load(ConfigManager().generate_config_yaml(config))

    assert data["simulation"]["CaseA"]["fulllist"] == "/sim/list.csv"
    assert data["simulation"]["CaseA"]["variables"] == {"Runoff": {"varname": "q", "varunit": "m3 s-1"}}


def test_generate_and_load_config_yaml_preserves_project_dask(tmp_path):
    from openbench.config.loader import load_config

    config = _runnable_config(tmp_path)
    config["general"]["dask"] = {
        "enabled": True,
        "n_workers": 3,
        "threads_per_worker": 2,
        "processes": False,
        "memory_limit": "2GB",
        "dashboard_address": ":0",
        "local_directory": str(tmp_path / "dask"),
    }

    config_path = tmp_path / "openbench.yaml"
    config_path.write_text(ConfigManager().generate_config_yaml(config), encoding="utf-8")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    loaded = load_config(config_path)

    assert data["project"]["dask"]["enabled"] is True
    assert data["project"]["dask"]["n_workers"] == 3
    assert loaded.project.dask.enabled is True
    assert loaded.project.dask.n_workers == 3
    assert loaded.project.dask.threads_per_worker == 2
    assert loaded.project.dask.processes is False
    assert loaded.project.dask.local_directory == str(tmp_path / "dask")


def test_generate_and_load_config_yaml_preserves_project_io(tmp_path):
    from openbench.config.loader import load_config

    config = _runnable_config(tmp_path)
    config["general"]["io"] = {
        "netcdf_compression": True,
        "netcdf_compression_level": 1,
        "mfdataset_batch_size": 25,
        "mfdataset_auto_batch_min_files": 100,
        "mfdataset_auto_batch_min_size_mb": 512,
        "mfdataset_auto_batch_min_size": 5,
        "mfdataset_auto_batch_max_size": 80,
        "mfdataset_auto_batch_memory_fraction": 0.5,
    }

    config_path = tmp_path / "openbench.yaml"
    config_path.write_text(ConfigManager().generate_config_yaml(config), encoding="utf-8")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    loaded = load_config(config_path)

    assert data["project"]["io"]["netcdf_compression"] is True
    assert data["project"]["io"]["mfdataset_batch_size"] == 25
    assert loaded.project.io.netcdf_compression is True
    assert loaded.project.io.netcdf_compression_level == 1
    assert loaded.project.io.mfdataset_auto_batch_max_size == 80


def test_unified_to_gui_config_preserves_project_dask():
    gui = ConfigManager().unified_to_gui_config(
        {
            "project": {
                "name": "case",
                "output_dir": "./output",
                "years": [2000, 2001],
                "dask": {"enabled": True, "n_workers": 2},
            },
            "evaluation": {"variables": []},
            "reference": {},
            "simulation": {},
        }
    )

    assert gui["general"]["dask"] == {"enabled": True, "n_workers": 2}


def test_unified_to_gui_config_preserves_project_io():
    gui = ConfigManager().unified_to_gui_config(
        {
            "project": {
                "name": "case",
                "output_dir": "./output",
                "years": [2000, 2001],
                "io": {"netcdf_compression": True, "mfdataset_batch_size": 25},
            },
            "evaluation": {"variables": []},
            "reference": {},
            "simulation": {},
        }
    )

    assert gui["general"]["io"] == {"netcdf_compression": True, "mfdataset_batch_size": 25}
