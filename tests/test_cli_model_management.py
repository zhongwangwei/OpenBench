from __future__ import annotations

import json
from pathlib import Path

import yaml
from click.testing import CliRunner

from openbench.cli.main import cli
from openbench.data.registry.manager import RegistryManager

runner = CliRunner()


def _catalog(home):
    return home / ".openbench" / "models" / "model_catalog.yaml"


def _load_model(home, name):
    return RegistryManager(user_dir=home / ".openbench").get_model(name)


def test_model_register_alias_updates_canonical_profile_after_reload(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    result = runner.invoke(cli, ["model", "register", "colm", "-v", "Audit_Var:audit_var:1"])

    assert result.exit_code == 0, result.output
    descriptor = yaml.safe_load(_catalog(home).read_text(encoding="utf-8"))
    assert "CoLM2024" in descriptor
    assert "CoLM" not in descriptor
    assert "alias 'colm' resolved to 'CoLM2024'" in result.output
    assert "Audit_Var" in _load_model(home, "CoLM2024").variables
    assert "Audit_Var" in _load_model(home, "colm").variables


def test_model_remove_var_from_bundled_profile_persists_after_reload(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    result = runner.invoke(cli, ["model", "remove-var", "CoLM2024", "Snow_Depth"])

    assert result.exit_code == 0, result.output
    descriptor = yaml.safe_load(_catalog(home).read_text(encoding="utf-8"))["CoLM2024"]
    assert "Snow_Depth" in descriptor["_delete_variables"]
    assert "Snow_Depth" not in _load_model(home, "CoLM2024").variables


def test_model_remove_var_works_on_sparse_overlay_of_bundled_profile(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    _catalog(home).parent.mkdir(parents=True)
    _catalog(home).write_text(yaml.safe_dump({"CoLM2024": {"description": "mine"}}))

    result = runner.invoke(cli, ["model", "remove-var", "CoLM2024", "Snow_Depth"])

    assert result.exit_code == 0, result.output
    descriptor = yaml.safe_load(_catalog(home).read_text(encoding="utf-8"))["CoLM2024"]
    assert descriptor == {"description": "mine", "_delete_variables": ["Snow_Depth"]}
    model = _load_model(home, "CoLM2024")
    assert "Snow_Depth" not in model.variables
    assert model.description == "mine"


def test_model_remove_var_from_differently_cased_sparse_overlay(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBENCH_HOME", str(tmp_path))
    catalog = _catalog(tmp_path)
    catalog.parent.mkdir(parents=True)
    catalog.write_text(
        yaml.safe_dump({"BCC_AVIM": {"variables": {"snow_depth": {"varname": "custom_snow", "varunit": "m"}}}})
    )
    result = runner.invoke(cli, ["model", "remove-var", "BCC_AVIM", "Snow_Depth"])
    assert result.exit_code == 0, result.output
    assert not any(name.lower() == "snow_depth" for name in _load_model(tmp_path, "BCC_AVIM").variables)


def test_model_description_edit_preserves_disabled_time_offset(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBENCH_HOME", str(tmp_path))
    catalog = _catalog(tmp_path)
    catalog.parent.mkdir(parents=True)
    catalog.write_text(yaml.safe_dump({"BCC_AVIM": {"time_offset": None}}))
    result = runner.invoke(cli, ["model", "register", "BCC_AVIM", "--description", "changed description"])
    assert result.exit_code == 0, result.output
    assert yaml.safe_load(catalog.read_text())["BCC_AVIM"]["description"] == "changed description"
    assert yaml.safe_load(catalog.read_text())["BCC_AVIM"]["time_offset"] is None
    assert not _load_model(tmp_path, "BCC_AVIM").time_offset


def test_model_register_on_sparse_overlay_keeps_bundled_fields(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    _catalog(home).parent.mkdir(parents=True)
    _catalog(home).write_text(yaml.safe_dump({"VIC5": {"description": "mine"}}))

    result = runner.invoke(cli, ["model", "register", "VIC5", "-v", "Audit_Var:audit_var:1"])

    assert result.exit_code == 0, result.output
    descriptor = yaml.safe_load(_catalog(home).read_text(encoding="utf-8"))["VIC5"]
    assert descriptor == {
        "description": "mine",
        "variables": {"Audit_Var": {"varname": "audit_var", "varunit": "1"}},
    }
    model = _load_model(home, "VIC5")
    assert model.tim_res == "Day"
    assert "Audit_Var" in model.variables


def test_model_remove_var_from_bundled_profile_matches_variable_ignoring_case(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    result = runner.invoke(cli, ["model", "remove-var", "CoLM2024", "snow_depth"])

    assert result.exit_code == 0, result.output
    descriptor = yaml.safe_load(_catalog(home).read_text(encoding="utf-8"))["CoLM2024"]
    assert "Snow_Depth" in descriptor["_delete_variables"]
    assert "Snow_Depth" not in _load_model(home, "CoLM2024").variables


def test_model_remove_var_matches_variable_ignoring_case(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    create = runner.invoke(
        cli,
        [
            "model",
            "register",
            "CaseVarModel",
            "--data-type",
            "grid",
            "--grid-res",
            "0.5",
            "--tim-res",
            "Month",
            "-v",
            "Latent_Heat:Qle:W m-2",
        ],
    )
    assert create.exit_code == 0, create.output

    result = runner.invoke(cli, ["model", "remove-var", "casevarmodel", "latent_heat"])

    assert result.exit_code == 0, result.output
    assert "Latent_Heat" not in _load_model(home, "CaseVarModel").variables


def test_model_delete_removes_user_profile(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    create = runner.invoke(
        cli,
        [
            "model",
            "register",
            "ScratchModel",
            "--data-type",
            "grid",
            "--grid-res",
            "0.5",
            "--tim-res",
            "Month",
            "-v",
            "Runoff:ro:mm day-1",
        ],
    )
    assert create.exit_code == 0, create.output

    result = runner.invoke(cli, ["model", "delete", "ScratchModel", "--yes"])

    assert result.exit_code == 0, result.output
    assert _load_model(home, "ScratchModel") is None


def test_model_register_extended_attributes_show_and_reload(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    result = runner.invoke(
        cli,
        [
            "model",
            "register",
            "AttrModel",
            "--data-type",
            "grid",
            "--grid-res",
            "0.5",
            "--tim-res",
            "Month",
            "-v",
            "GPP:gpp:gC",
            "--var-attr",
            "GPP:compute=value * 12.011",
            "--var-attr",
            "GPP:sub_dir=carbon",
            "--var-attr",
            "GPP:prefix_fallback=case_,hist_",
            "--time-offset",
            "Month=-15 days",
        ],
    )

    assert result.exit_code == 0, result.output
    model = _load_model(home, "AttrModel")
    assert model.variables["GPP"].compute == "value * 12.011"
    assert model.variables["GPP"].sub_dir == "carbon"
    assert model.variables["GPP"].prefix_fallback == ["case_", "hist_"]
    assert model.time_offset == {"Month": "-15 days"}

    show = runner.invoke(cli, ["model", "show", "AttrModel", "--format", "json"])
    assert show.exit_code == 0, show.output
    payload = json.loads(show.output)
    assert payload["time_offset"] == {"Month": "-15 days"}
    assert payload["variables"]["GPP"]["sub_dir"] == "carbon"


def test_model_register_accepts_case_insensitive_time_resolution(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    result = runner.invoke(
        cli,
        [
            "model",
            "register",
            "LowerTime",
            "--data-type",
            "grid",
            "--grid-res",
            "0.5",
            "--tim-res",
            "month",
            "-v",
            "Runoff:ro:mm",
        ],
    )

    assert result.exit_code == 0, result.output
    assert _load_model(home, "LowerTime").tim_res == "Month"


def test_model_register_allows_fallback_conversion_peer_variables(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    result = runner.invoke(
        cli,
        [
            "model",
            "register",
            "PeerFallback",
            "--data-type",
            "grid",
            "--grid-res",
            "0.5",
            "--tim-res",
            "Month",
            "-v",
            "Net_Ecosystem_Exchange:nee:g m-2 s-1",
            "-v",
            "Gross_Primary_Productivity:f_assim:mol m-2 s-1",
            "-f",
            "Net_Ecosystem_Exchange:f_respc:mol m-2 s-1:value * 12.011 - f_assim * 12.011",
        ],
    )

    assert result.exit_code == 0, result.output
    model = _load_model(home, "PeerFallback")
    assert model.variables["Net_Ecosystem_Exchange"].fallbacks[0].convert == ("value * 12.011 - f_assim * 12.011")


def test_model_validate_accepts_bundled_peer_variable_fallbacks(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))

    result = runner.invoke(cli, ["model", "validate", "CoLM2024"])

    assert result.exit_code == 0, result.output
    assert "looks valid" in result.output


def test_model_register_time_offset_merges_default_and_variable_offsets_order_independently(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    result = runner.invoke(
        cli,
        [
            "model",
            "register",
            "OffsetModel",
            "--data-type",
            "grid",
            "--grid-res",
            "0.5",
            "--tim-res",
            "Day",
            "-v",
            "Runoff:ro:mm day-1",
            "--time-offset",
            "Day:Runoff=0",
            "--time-offset",
            "Day=-1",
        ],
    )

    assert result.exit_code == 0, result.output
    descriptor = yaml.safe_load(_catalog(home).read_text(encoding="utf-8"))["OffsetModel"]
    assert descriptor["time_offset"] == {"Day": {"Runoff": "0", "default": "-1"}}
    assert _load_model(home, "OffsetModel").time_offset == {"Day": {"Runoff": "0", "default": "-1"}}


def test_model_register_time_offset_preserves_existing_default_when_adding_variable_offset(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    create = runner.invoke(
        cli,
        [
            "model",
            "register",
            "OffsetUpdateModel",
            "--data-type",
            "grid",
            "--grid-res",
            "0.5",
            "--tim-res",
            "Day",
            "-v",
            "Runoff:ro:mm day-1",
            "--time-offset",
            "Day=-1",
        ],
    )
    assert create.exit_code == 0, create.output

    result = runner.invoke(
        cli,
        [
            "model",
            "register",
            "OffsetUpdateModel",
            "--time-offset",
            "Day:Runoff=0",
        ],
    )

    assert result.exit_code == 0, result.output
    descriptor = yaml.safe_load(_catalog(home).read_text(encoding="utf-8"))["OffsetUpdateModel"]
    assert descriptor["time_offset"] == {"Day": {"default": "-1", "Runoff": "0"}}


def test_model_register_var_attr_can_create_new_profile_noninteractively(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    result = runner.invoke(
        cli,
        [
            "model",
            "register",
            "ComputedModel",
            "--data-type",
            "grid",
            "--grid-res",
            "0.5",
            "--tim-res",
            "Month",
            "--var-attr",
            "Runoff:compute=ds['rain'] + ds['snow']",
        ],
    )

    assert result.exit_code == 0, result.output
    mapping = _load_model(home, "ComputedModel").variables["Runoff"]
    assert mapping.varname == "Runoff"
    assert mapping.varunit == ""
    assert mapping.compute == "ds['rain'] + ds['snow']"


def test_model_register_named_variable_syntax_preserves_colons(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    result = runner.invoke(
        cli,
        [
            "model",
            "register",
            "ColonModel",
            "--data-type",
            "grid",
            "--grid-res",
            "0.5",
            "--tim-res",
            "Month",
            "-v",
            'Temperature name=tas unit="degC:lower" prefix="C:/openbench/case:" suffix=":monthly"',
        ],
    )

    assert result.exit_code == 0, result.output
    mapping = _load_model(home, "ColonModel").variables["Temperature"]
    assert mapping.varname == "tas"
    assert mapping.varunit == "degC:lower"
    assert mapping.prefix == "C:/openbench/case:"
    assert mapping.suffix == ":monthly"


def test_model_show_history_reports_user_catalog_backups(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    create = runner.invoke(
        cli,
        [
            "model",
            "register",
            "HistoryModel",
            "--data-type",
            "grid",
            "--grid-res",
            "0.5",
            "--tim-res",
            "Month",
            "-v",
            "Runoff:ro:mm",
        ],
    )
    assert create.exit_code == 0, create.output
    update = runner.invoke(cli, ["model", "register", "HistoryModel", "-v", "GPP:gpp:gC"])
    assert update.exit_code == 0, update.output

    result = runner.invoke(cli, ["model", "show", "HistoryModel", "--history", "--format", "json"])

    assert result.exit_code == 0, result.output
    history = json.loads(result.output)
    assert history
    assert history[0]["catalog_name"] == "HistoryModel"
    assert history[0]["variables"] == 1


def test_model_export_import_and_rename_roundtrip(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    create = runner.invoke(
        cli,
        [
            "model",
            "register",
            "PortableModel",
            "--data-type",
            "grid",
            "--grid-res",
            "1.0",
            "--tim-res",
            "Day",
            "-v",
            "Runoff:ro:mm day-1",
        ],
    )
    assert create.exit_code == 0, create.output
    exported = tmp_path / "portable.yaml"

    export_result = runner.invoke(cli, ["model", "export", "PortableModel", "-o", str(exported)])
    assert export_result.exit_code == 0, export_result.output
    delete_result = runner.invoke(cli, ["model", "delete", "PortableModel", "--yes"])
    assert delete_result.exit_code == 0, delete_result.output
    import_result = runner.invoke(cli, ["model", "import", str(exported), "--yes"])
    assert import_result.exit_code == 0, import_result.output
    rename_result = runner.invoke(cli, ["model", "rename", "PortableModel", "PortableModelV2", "--yes"])

    assert rename_result.exit_code == 0, rename_result.output
    assert _load_model(home, "PortableModel") is None
    assert _load_model(home, "PortableModelV2") is not None


def test_model_import_preserves_existing_catalog_key_case(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    catalog = _catalog(home)
    catalog.parent.mkdir(parents=True)
    catalog.write_text(
        yaml.safe_dump(
            {
                "CLM5": {
                    "name": "CLM5",
                    "description": "existing",
                    "data_type": "grid",
                    "grid_res": 1.0,
                    "tim_res": "Month",
                    "variables": {"Runoff": {"varname": "ro", "varunit": "mm"}},
                }
            }
        )
    )
    imported = tmp_path / "clm5.yaml"
    imported.write_text(
        yaml.safe_dump(
            {
                "name": "clm5",
                "description": "imported",
                "data_type": "grid",
                "grid_res": 0.5,
                "tim_res": "Day",
                "variables": {"Runoff": {"varname": "runoff", "varunit": "mm day-1"}},
            }
        )
    )

    result = runner.invoke(cli, ["model", "import", str(imported), "--yes"])

    assert result.exit_code == 0, result.output
    descriptor = yaml.safe_load(catalog.read_text(encoding="utf-8"))
    assert list(descriptor) == ["CLM5"]
    # The overlay of a bundled profile stores only changed fields; the name
    # comes from the catalog key, so the import must not rename it to "clm5".
    assert descriptor["CLM5"].get("name", "CLM5") == "CLM5"
    assert descriptor["CLM5"]["variables"]["Runoff"] == {"varname": "runoff", "varunit": "mm day-1"}


def test_model_status_supports_json(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))

    result = runner.invoke(cli, ["model", "status", "--format", "json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "model_profiles" in payload
    assert "user_catalog" in payload


def test_model_list_empty_registry_reports_message(monkeypatch):
    import openbench.data.registry as registry_pkg

    class EmptyRegistry:
        def list_models(self):
            return []

    monkeypatch.setattr(registry_pkg, "RegistryManager", lambda: EmptyRegistry())

    result = runner.invoke(cli, ["model", "list"])

    assert result.exit_code == 0, result.output
    assert "No model profiles registered." in result.output


def test_model_import_reports_invalid_profile_without_traceback(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    bad_profile = tmp_path / "bad-profile.yaml"
    bad_profile.write_text(
        yaml.safe_dump(
            {
                "name": "BadModel",
                "variables": {
                    "Runoff": "not-a-mapping",
                },
            }
        )
    )

    result = runner.invoke(cli, ["model", "import", str(bad_profile)])

    assert result.exit_code == 1
    assert "Invalid model profile" in result.output
    assert not isinstance(result.exception, AttributeError)
    assert not _catalog(home).exists()


def test_model_import_reports_malformed_yaml_without_traceback(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    bad_profile = tmp_path / "malformed-profile.yaml"
    bad_profile.write_text("name: BadModel\nvariables: [\n")

    result = runner.invoke(cli, ["model", "import", str(bad_profile)])

    assert result.exit_code == 1
    assert "Failed to read model profile YAML" in result.output
    assert "Traceback" not in result.output
    assert not isinstance(result.exception, yaml.YAMLError)
    assert not _catalog(home).exists()


def test_model_register_rejects_invalid_name_and_stn_grid_res(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    bad_name = runner.invoke(cli, ["model", "register", "../bad", "-v", "Runoff:ro:mm"])
    stn_grid = runner.invoke(
        cli,
        [
            "model",
            "register",
            "StationModel",
            "--data-type",
            "stn",
            "--grid-res",
            "0.5",
            "-v",
            "Runoff:ro:mm",
        ],
    )

    assert bad_name.exit_code == 1
    assert "model name must be" in bad_name.output
    assert stn_grid.exit_code == 1
    assert "--grid-res is not valid for station model profiles" in stn_grid.output


def test_model_register_rejects_grid_res_update_for_existing_station_profile(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    create = runner.invoke(
        cli,
        [
            "model",
            "register",
            "StationOnly",
            "--data-type",
            "stn",
            "--tim-res",
            "Day",
            "-v",
            "Runoff:ro:mm",
        ],
    )
    assert create.exit_code == 0, create.output

    result = runner.invoke(cli, ["model", "register", "StationOnly", "--grid-res", "0.5"])

    assert result.exit_code == 1
    assert "--grid-res is not valid for station model profiles" in result.output


def test_model_register_rejects_nonpositive_grid_res(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    result = runner.invoke(
        cli,
        [
            "model",
            "register",
            "BadGridModel",
            "--data-type",
            "grid",
            "--grid-res",
            "0",
            "--tim-res",
            "Month",
            "-v",
            "Runoff:ro:mm",
        ],
    )

    assert result.exit_code == 1
    assert "--grid-res must be a positive value" in result.output


def test_model_register_warns_when_creating_user_overlay_over_bundled(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    result = runner.invoke(cli, ["model", "register", "CoLM2024", "-v", "Audit_Var:audit_var:1"])

    assert result.exit_code == 0, result.output
    assert "Creating user overlay over bundled model profile 'CoLM2024'" in result.output


def test_model_register_interactive_can_capture_extended_fields(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    result = runner.invoke(
        cli,
        [
            "model",
            "register",
            "ManualInteractiveModel",
            "--data-type",
            "grid",
            "--grid-res",
            "0.5",
            "--tim-res",
            "Month",
        ],
        input=("Interactive description\nMonth=-15 days\nGPP\ngpp\ngC\ncarbon\ncase_\n.nc\n\n"),
    )

    assert result.exit_code == 0, result.output
    descriptor = yaml.safe_load(_catalog(home).read_text(encoding="utf-8"))["ManualInteractiveModel"]
    assert descriptor["description"] == "Interactive description"
    assert descriptor["time_offset"] == {"Month": "-15 days"}
    assert descriptor["variables"]["GPP"]["sub_dir"] == "carbon"
    assert descriptor["variables"]["GPP"]["prefix"] == "case_"
    assert descriptor["variables"]["GPP"]["suffix"] == ".nc"


def test_model_register_rejects_orphan_fallback(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    create = runner.invoke(
        cli,
        [
            "model",
            "register",
            "FallbackTargetModel",
            "--data-type",
            "grid",
            "--grid-res",
            "0.5",
            "--tim-res",
            "Month",
            "-v",
            "Runoff:ro:mm",
        ],
    )
    assert create.exit_code == 0, create.output

    result = runner.invoke(
        cli,
        [
            "model",
            "register",
            "FallbackTargetModel",
            "-f",
            "GPP:gpp_alt:gC:value * 1.0",
        ],
    )

    assert result.exit_code == 1
    assert "Fallback target 'GPP' has no primary variable" in result.output


def test_model_register_append_only_can_append_fallback_to_existing_variable(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    create = runner.invoke(
        cli,
        [
            "model",
            "register",
            "FallbackModel",
            "--data-type",
            "grid",
            "--grid-res",
            "0.5",
            "--tim-res",
            "Month",
            "-v",
            "GPP:gpp:gC",
        ],
    )
    assert create.exit_code == 0, create.output

    result = runner.invoke(
        cli,
        [
            "model",
            "register",
            "FallbackModel",
            "--append-only",
            "-f",
            "GPP:gpp_alt:gC:value * 1.0",
        ],
    )

    assert result.exit_code == 0, result.output
    fallbacks = _load_model(home, "FallbackModel").variables["GPP"].fallbacks
    assert fallbacks and fallbacks[0].varname == "gpp_alt"


def test_model_show_long_compute_suggests_structured_output(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    create = runner.invoke(
        cli,
        [
            "model",
            "register",
            "LongComputeModel",
            "--data-type",
            "grid",
            "--grid-res",
            "0.5",
            "--tim-res",
            "Month",
            "-v",
            "GPP:gpp:gC",
            "--var-attr",
            "GPP:compute=value * 12.011 + value * 0.001 + value * 0.0001",
        ],
    )
    assert create.exit_code == 0, create.output

    result = runner.invoke(cli, ["model", "show", "LongComputeModel"])

    assert result.exit_code == 0, result.output
    assert "use --format json/yaml to see full" in result.output


def test_model_list_and_show_report_incomplete_variable_mappings(tmp_path, monkeypatch):
    home = tmp_path / "home"
    catalog = _catalog(home)
    catalog.parent.mkdir(parents=True)
    catalog.write_text(
        yaml.safe_dump(
            {
                "IncompleteModel": {
                    "name": "IncompleteModel",
                    "description": "demo",
                    "data_type": "grid",
                    "grid_res": 0.5,
                    "tim_res": "Month",
                    "variables": {"Runoff": {"varname": "ro", "varunit": ""}},
                }
            }
        )
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    listed = runner.invoke(cli, ["model", "list"])
    shown = runner.invoke(cli, ["model", "show", "IncompleteModel"])

    assert listed.exit_code == 0, listed.output
    assert shown.exit_code == 0, shown.output
    assert "⚠" in listed.output
    assert "missing varunit" in shown.output


def test_model_alias_and_validate_commands(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    create = runner.invoke(
        cli,
        [
            "model",
            "register",
            "AliasTarget",
            "--data-type",
            "grid",
            "--grid-res",
            "0.5",
            "--tim-res",
            "Month",
            "-v",
            "Runoff:ro:mm",
        ],
    )
    assert create.exit_code == 0, create.output

    alias_result = runner.invoke(cli, ["model", "alias", "at", "AliasTarget"])
    validate_result = runner.invoke(cli, ["model", "validate", "at"])

    assert alias_result.exit_code == 0, alias_result.output
    assert validate_result.exit_code == 0, validate_result.output
    assert _load_model(home, "at").name == "AliasTarget"


def test_model_alias_file_is_not_loaded_as_model_profile(tmp_path, monkeypatch, caplog):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    create = runner.invoke(
        cli,
        [
            "model",
            "register",
            "AliasTarget",
            "--data-type",
            "grid",
            "--grid-res",
            "0.5",
            "--tim-res",
            "Month",
            "-v",
            "Runoff:ro:mm",
        ],
    )
    assert create.exit_code == 0, create.output
    alias_result = runner.invoke(cli, ["model", "alias", "at", "AliasTarget"])
    assert alias_result.exit_code == 0, alias_result.output

    caplog.set_level("WARNING", logger="openbench.data.registry.manager")
    caplog.clear()
    loaded = RegistryManager(user_dir=home / ".openbench").get_model("at")

    assert loaded is not None
    assert loaded.name == "AliasTarget"
    assert "aliases.yaml" not in caplog.text


def test_model_alias_reports_corrupt_alias_file_without_traceback(tmp_path, monkeypatch):
    home = tmp_path / "home"
    alias_path = home / ".openbench" / "models" / "aliases.yaml"
    alias_path.parent.mkdir(parents=True)
    alias_path.write_text("bad: [\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    result = runner.invoke(cli, ["model", "alias"])

    assert result.exit_code == 1
    assert "Failed to read model aliases YAML" in result.output
    assert "Traceback" not in result.output
    assert not isinstance(result.exception, yaml.YAMLError)


def test_model_status_and_path_commands_report_user_catalog(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    create = runner.invoke(
        cli,
        [
            "model",
            "register",
            "PathTarget",
            "--data-type",
            "grid",
            "--grid-res",
            "0.5",
            "--tim-res",
            "Month",
            "-v",
            "Runoff:ro:mm",
        ],
    )
    assert create.exit_code == 0, create.output

    status = runner.invoke(cli, ["model", "status"])
    path = runner.invoke(cli, ["model", "path", "PathTarget"])

    assert status.exit_code == 0, status.output
    assert path.exit_code == 0, path.output
    assert "model profiles available" in status.output
    assert str(_catalog(home)) in status.output
    assert f"user({_catalog(home)})" in path.output


def test_model_show_reports_corrupt_user_catalog_without_traceback(tmp_path, monkeypatch):
    home = tmp_path / "home"
    catalog = _catalog(home)
    catalog.parent.mkdir(parents=True)
    catalog.write_text("bad: [\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    result = runner.invoke(cli, ["model", "show", "CoLM2024"])

    assert result.exit_code == 1
    assert "Failed to read model catalog YAML" in result.output
    assert "Traceback" not in result.output
    assert not isinstance(result.exception, yaml.YAMLError)


def test_model_delete_refuses_bundled_profile(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    result = runner.invoke(cli, ["model", "delete", "CoLM2024", "--yes"])

    assert result.exit_code == 1
    assert "bundled" in result.output.lower()


def test_model_delete_handles_missing_profile(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    result = runner.invoke(cli, ["model", "delete", "NoSuchModel", "--yes"])

    assert result.exit_code == 1
    assert "Model not found in user catalog" in result.output


def test_model_import_rejects_yaml_without_name(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    imported = tmp_path / "missing_name.yaml"
    imported.write_text(yaml.safe_dump({"variables": {"Runoff": {"varname": "ro", "varunit": "mm"}}}))

    result = runner.invoke(cli, ["model", "import", str(imported), "--yes"])

    assert result.exit_code == 1
    assert "must be a mapping with a 'name' field" in result.output


def test_model_rename_refuses_existing_target(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    for name in ("RenameSource", "RenameTarget"):
        create = runner.invoke(
            cli,
            [
                "model",
                "register",
                name,
                "--data-type",
                "grid",
                "--grid-res",
                "0.5",
                "--tim-res",
                "Month",
                "-v",
                "Runoff:ro:mm",
            ],
        )
        assert create.exit_code == 0, create.output

    result = runner.invoke(cli, ["model", "rename", "RenameSource", "RenameTarget", "--yes"])

    assert result.exit_code == 1
    assert "Model already exists in user catalog" in result.output


def test_model_alias_rejects_unknown_canonical(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    result = runner.invoke(cli, ["model", "alias", "missing_alias", "NoSuchModel"])

    assert result.exit_code == 1
    assert "Canonical model not found" in result.output


def test_model_validate_flags_missing_varunit(tmp_path, monkeypatch):
    home = tmp_path / "home"
    catalog = _catalog(home)
    catalog.parent.mkdir(parents=True)
    catalog.write_text(
        yaml.safe_dump(
            {
                "InvalidModel": {
                    "name": "InvalidModel",
                    "description": "demo",
                    "data_type": "grid",
                    "grid_res": 0.5,
                    "tim_res": "Month",
                    "variables": {"Runoff": {"varname": "ro", "varunit": ""}},
                }
            }
        )
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    result = runner.invoke(cli, ["model", "validate", "InvalidModel"])

    assert result.exit_code == 1
    assert "missing varunit" in result.output


def test_model_export_reports_write_failure(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    def fail_write_text(self, data, *args, **kwargs):
        if self.name == "readonly.yaml":
            raise OSError("permission denied")
        return original_write_text(self, data, *args, **kwargs)

    original_write_text = Path.write_text
    monkeypatch.setattr(Path, "write_text", fail_write_text)

    result = runner.invoke(
        cli,
        ["model", "export", "CoLM2024", "-o", str(tmp_path / "readonly.yaml")],
    )

    assert result.exit_code == 1
    assert "permission denied" in result.output


def test_parse_var_attr_allows_colon_in_standard_variable_name():
    from openbench.cli.model import _parse_var_attrs

    target = {}
    existing = {"soil:moisture": {"varname": "sm", "varunit": "kg m-2"}}

    _parse_var_attrs(("soil:moisture:sub_dir=hydro:daily",), target, existing)

    assert target["soil:moisture"]["sub_dir"] == "hydro:daily"


def test_model_register_and_validate_accept_compute_source_variable_names(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    result = runner.invoke(
        cli,
        [
            "model",
            "register",
            "ComputeModel",
            "--data-type",
            "grid",
            "--grid-res",
            "0.5",
            "--tim-res",
            "Month",
            "-v",
            "Leaf_Area_Index:lai:1",
            "--var-attr",
            "Leaf_Area_Index:compute=lai + 1.0",
        ],
    )

    assert result.exit_code == 0, result.output
    validate = runner.invoke(cli, ["model", "validate", "ComputeModel"])
    assert validate.exit_code == 0, validate.output


def test_model_alias_edits_and_removal_survive_registry_reload(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBENCH_HOME", str(tmp_path))
    for args in (
        ["-v", "Bare_Soil_Evaporation:custom_soil:mm"],
        ["--var-attr", "Bare_Soil_Evaporation:prefix=soil_"],
        ["-f", "Bare_Soil_Evaporation:alternate_soil:mm"],
    ):
        result = runner.invoke(cli, ["model", "register", "CoLM2024", *args])
        assert result.exit_code == 0, result.output
    mapping = _load_model(tmp_path, "CoLM2024").variables["Soil_Evaporation"]
    assert mapping.varname == "custom_soil"
    assert mapping.prefix == "soil_"
    assert mapping.fallbacks[0].varname == "alternate_soil"
    removed = runner.invoke(cli, ["model", "remove-var", "CoLM2024", "Bare_Soil_Evaporation"])
    assert removed.exit_code == 0, removed.output
    assert "Soil_Evaporation" not in _load_model(tmp_path, "CoLM2024").variables


def test_model_native_replacement_clears_inherited_compute(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBENCH_HOME", str(tmp_path))
    baseline = _load_model(tmp_path, "CoLM2024")
    variable = next(name for name, mapping in baseline.variables.items() if mapping.compute)
    result = runner.invoke(cli, ["model", "register", "CoLM2024", "-v", f"{variable}:native:1"])
    assert result.exit_code == 0, result.output
    mapping = _load_model(tmp_path, "CoLM2024").variables[variable]
    assert mapping.varname == "native"
    assert mapping.compute is None
    assert not mapping.fallbacks


def test_model_time_offset_edit_keeps_other_resolutions_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBENCH_HOME", str(tmp_path))
    catalog = _catalog(tmp_path)
    catalog.parent.mkdir(parents=True)
    catalog.write_text(yaml.safe_dump({"BCC_AVIM": {"time_offset": None}}))
    result = runner.invoke(cli, ["model", "register", "BCC_AVIM", "--time-offset", "Day=0"])
    assert result.exit_code == 0, result.output
    offsets = _load_model(tmp_path, "BCC_AVIM").time_offset
    assert offsets["Day"] == "0"
    assert not offsets.get("Hour")
    assert not offsets.get("Month")
    result = runner.invoke(cli, ["model", "register", "BCC_AVIM", "--description", "later edit"])
    assert result.exit_code == 0, result.output
    assert _load_model(tmp_path, "BCC_AVIM").time_offset == offsets


def test_model_register_restores_deleted_bundled_profile_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBENCH_HOME", str(tmp_path))
    baseline = _load_model(tmp_path, "VIC5")
    catalog = _catalog(tmp_path)
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text(yaml.safe_dump({"VIC5": {"_deleted": True}}))
    result = runner.invoke(cli, ["model", "register", "VIC5", "--description", "restored"])
    assert result.exit_code == 0, result.output
    restored = _load_model(tmp_path, "VIC5")
    assert restored.tim_res == baseline.tim_res
    assert restored.variables == baseline.variables
    assert restored.description == "restored"


def test_model_station_conversion_clears_bundled_grid_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBENCH_HOME", str(tmp_path))
    result = runner.invoke(cli, ["model", "register", "CoLM2024", "--data-type", "stn"])
    assert result.exit_code == 0, result.output
    profile = _load_model(tmp_path, "CoLM2024")
    assert profile.data_type == "stn"
    assert profile.grid_res is None


def test_model_te_metadata_edit_keeps_valid_soil_alias(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBENCH_HOME", str(tmp_path))
    result = runner.invoke(cli, ["model", "register", "TE", "--description", "my TE"])
    assert result.exit_code == 0, result.output
    profile = _load_model(tmp_path, "TE")
    assert profile.description == "my TE"
    assert profile.variables["Soil_Evaporation"].varname == "EBFLX"


def test_model_export_import_preserves_deletions_and_disabled_offsets(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBENCH_HOME", str(tmp_path))
    catalog = _catalog(tmp_path)
    catalog.parent.mkdir(parents=True)
    catalog.write_text(yaml.safe_dump({"BCC_AVIM": {"time_offset": None, "_delete_variables": ["Snow_Depth"]}}))
    before = _load_model(tmp_path, "BCC_AVIM")
    exported = tmp_path / "export.yaml"
    result = runner.invoke(cli, ["model", "export", "BCC_AVIM", "-o", str(exported)])
    assert result.exit_code == 0, result.output
    catalog.write_text("{}\n")
    result = runner.invoke(cli, ["model", "import", str(exported), "--yes"])
    assert result.exit_code == 0, result.output
    after = _load_model(tmp_path, "BCC_AVIM")
    assert after == before


def test_model_export_import_preserves_partial_complete_offsets(tmp_path, monkeypatch):
    from dataclasses import replace

    from openbench.data.registry.manager import RegistryManager

    monkeypatch.setenv("OPENBENCH_HOME", str(tmp_path))
    manager = RegistryManager()
    before = replace(manager.get_model("BCC_AVIM"), time_offset={"Day": "0"})
    manager.save_model(before.name, before)
    exported = tmp_path / "export.yaml"
    result = runner.invoke(cli, ["model", "export", before.name, "-o", str(exported)])
    assert result.exit_code == 0, result.output
    _catalog(tmp_path).write_text("{}\n")
    result = runner.invoke(cli, ["model", "import", str(exported), "--yes"])
    assert result.exit_code == 0, result.output
    assert _load_model(tmp_path, before.name) == before


def test_model_rename_refuses_bundled_destination(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBENCH_HOME", str(tmp_path))
    catalog = _catalog(tmp_path)
    catalog.parent.mkdir(parents=True)
    catalog.write_text(yaml.safe_dump({"Mine": {"variables": {"Runoff": {"varname": "ro", "varunit": "mm"}}}}))
    result = runner.invoke(cli, ["model", "rename", "Mine", "CoLM2024", "--yes"])
    assert result.exit_code == 1
    assert "bundled catalog" in result.output
    assert "Mine" in yaml.safe_load(catalog.read_text())


def test_model_metadata_edit_preserves_null_variable_deletion(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBENCH_HOME", str(tmp_path))
    catalog = _catalog(tmp_path)
    catalog.parent.mkdir(parents=True)
    catalog.write_text(yaml.safe_dump({"CLM5": {"variables": {"Snow_Depth": None}}}))
    result = runner.invoke(cli, ["model", "register", "CLM5", "--description", "updated"])
    assert result.exit_code == 0, result.output
    profile = _load_model(tmp_path, "CLM5")
    assert "Snow_Depth" not in profile.variables
    assert profile.description == "updated"


def test_model_metadata_edit_keeps_legacy_list_sparse(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBENCH_HOME", str(tmp_path))
    catalog = _catalog(tmp_path)
    catalog.parent.mkdir(parents=True)
    catalog.write_text(yaml.safe_dump({"CaMa": {"variables": {"Dam_Outflow": {"varname": ["a", "b"]}}}}))
    before = _load_model(tmp_path, "CaMa")
    result = runner.invoke(cli, ["model", "register", "CaMa", "--description", "updated"])
    assert result.exit_code == 0, result.output
    after = _load_model(tmp_path, "CaMa")
    assert after.variables == before.variables
    mapping = yaml.safe_load(catalog.read_text())["CaMa"]["variables"]["Dam_Outflow"]
    assert mapping["varname"] == ["a", "b"]
    assert "fallbacks" not in mapping


def test_model_history_resolves_sparse_profile_counts(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBENCH_HOME", str(tmp_path))
    catalog = _catalog(tmp_path)
    catalog.parent.mkdir(parents=True)
    catalog.with_suffix(".yaml.bak").write_text(yaml.safe_dump({"VIC5": {"description": "historical"}}))
    result = runner.invoke(cli, ["model", "show", "VIC5", "--history", "--format", "json"])
    assert result.exit_code == 0, result.output
    entry = json.loads(result.output)[0]
    assert entry["description"] == "historical"
    assert entry["variables"] == len(_load_model(tmp_path, "VIC5").variables)


def test_model_rename_explains_restoring_bundled_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBENCH_HOME", str(tmp_path))
    catalog = _catalog(tmp_path)
    catalog.parent.mkdir(parents=True)
    catalog.write_text(yaml.safe_dump({"VIC5": {"description": "custom"}}))
    result = runner.invoke(cli, ["model", "rename", "VIC5", "MyVIC", "--yes"])
    assert result.exit_code == 0, result.output
    assert "bundled defaults again" in result.output
    assert _load_model(tmp_path, "MyVIC").description == "custom"
    assert _load_model(tmp_path, "VIC5").description != "custom"


def test_model_conflicting_alias_input_is_rejected_before_write(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBENCH_HOME", str(tmp_path))
    result = runner.invoke(
        cli,
        [
            "model",
            "register",
            "CoLM2024",
            "-v",
            "Bare_Soil_Evaporation:first:mm",
            "-v",
            "Soil_Evaporation:second:mm",
        ],
    )
    assert result.exit_code == 1
    assert "Conflicting variable alias" in result.output
    catalog = _catalog(tmp_path)
    assert not catalog.exists() or not yaml.safe_load(catalog.read_text())


def test_model_register_fallback_after_variable_replacement(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    replaced = runner.invoke(cli, ["model", "register", "CaMa", "-v", "Dam_Outflow:dout:m3 s-1"])
    assert replaced.exit_code == 0, replaced.output
    added = runner.invoke(cli, ["model", "register", "CaMa", "-f", "Dam_Outflow:outflw:m3 s-1"])

    assert added.exit_code == 0, added.output
    mapping = _load_model(home, "CaMa").variables["Dam_Outflow"]
    assert mapping.varname == "dout"
    assert [fb.varname for fb in mapping.fallbacks] == ["outflw"]


def test_gui_save_under_builtin_alias_updates_the_named_profile(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    from dataclasses import replace

    registry = RegistryManager(user_dir=home / ".openbench")
    alias_copy = registry.get_model("CoLM")
    registry.save_model("CoLM", replace(alias_copy, description="edited via alias"))

    catalog = yaml.safe_load(_catalog(home).read_text(encoding="utf-8"))
    assert list(catalog) == ["CoLM2024"]
    assert _load_model(home, "CoLM2024").description == "edited via alias"
    assert _load_model(home, "colm").description == "edited via alias"


def test_model_import_under_builtin_alias_updates_the_named_profile(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    exported = _load_model(home, "CoLM2024").to_dict()
    exported.update(name="colm", description="imported via alias")
    source = tmp_path / "colm.yaml"
    source.write_text(yaml.safe_dump(exported), encoding="utf-8")

    result = runner.invoke(cli, ["model", "import", str(source), "--yes"])

    assert result.exit_code == 0, result.output
    assert list(yaml.safe_load(_catalog(home).read_text(encoding="utf-8"))) == ["CoLM2024"]
    assert _load_model(home, "CoLM2024").description == "imported via alias"


def test_model_register_back_to_grid_restores_bundled_resolution(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    bundled = _load_model(home, "CLM5").grid_res

    to_stn = runner.invoke(cli, ["model", "register", "CLM5", "--data-type", "stn"])
    assert to_stn.exit_code == 0, to_stn.output
    assert _load_model(home, "CLM5").grid_res is None
    to_grid = runner.invoke(cli, ["model", "register", "CLM5", "--data-type", "grid"])

    assert to_grid.exit_code == 0, to_grid.output
    assert _load_model(home, "CLM5").grid_res == bundled
    assert "grid_res" not in (yaml.safe_load(_catalog(home).read_text(encoding="utf-8")).get("CLM5") or {})


def test_model_show_history_lists_valid_backups_next_to_an_invalid_one(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    assert runner.invoke(cli, ["model", "register", "VIC5", "--description", "first"]).exit_code == 0
    assert runner.invoke(cli, ["model", "register", "VIC5", "--description", "second"]).exit_code == 0
    broken = _catalog(home).with_name(f"{_catalog(home).name}.20000101-000000-000000.bak")
    broken.write_text(
        yaml.safe_dump(
            {"VIC5": {"variables": {"Soil_Evaporation": {"varname": "a"}, "Bare_Soil_Evaporation": {"varname": "b"}}}}
        ),
        encoding="utf-8",
    )

    result = runner.invoke(cli, ["model", "show", "VIC5", "--history", "--format", "json"])

    assert result.exit_code == 0, result.output
    entries = json.loads(result.output)
    assert [entry["variables"] for entry in entries if entry["backup"] == str(broken)] == ["invalid"]
    assert any(entry["variables"] != "invalid" for entry in entries)


def test_real_profiles_keep_their_name_when_an_alias_spells_the_same(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    from dataclasses import replace

    registry = RegistryManager(user_dir=home / ".openbench")
    assert registry.model_write_target("CaMaFlood") == "CaMaFlood"  # also an alias of CaMa
    assert registry.model_write_target("cama-flood") == "CaMa"
    assert registry.model_write_target("colm") == "CoLM2024"
    cama_before = registry.get_model("CaMa")

    registry.save_model("CaMaFlood", replace(registry.get_model("CaMaFlood"), description="edited"))

    assert list(yaml.safe_load(_catalog(home).read_text(encoding="utf-8"))) == ["CaMaFlood"]
    assert _load_model(home, "CaMaFlood").description == "edited"
    assert _load_model(home, "CaMa") == cama_before
    result = runner.invoke(cli, ["model", "register", "CaMaFlood", "--description", "cli edit"])
    assert result.exit_code == 0, result.output
    assert _load_model(home, "CaMaFlood").description == "cli edit"
    assert _load_model(home, "CaMa") == cama_before


def test_model_alias_cannot_take_an_existing_profile_name(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    result = runner.invoke(cli, ["model", "alias", "clm5", "CoLM2024"])

    assert result.exit_code != 0
    assert "already a model profile name" in result.output


def test_model_import_over_a_bundled_profile_asks_first(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    source = tmp_path / "clm5.yaml"
    source.write_text(
        yaml.safe_dump(
            {"name": "CLM5", "description": "partial", "variables": {"Runoff": {"varname": "q", "varunit": "mm"}}}
        )
    )

    declined = runner.invoke(cli, ["model", "import", str(source)], input="n\n")

    assert "Overwrite existing profile 'CLM5'?" in declined.output
    assert not _catalog(home).exists() or not yaml.safe_load(_catalog(home).read_text(encoding="utf-8"))


def test_per_variable_offset_keeps_cleared_bundled_offsets_cleared(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    _catalog(home).parent.mkdir(parents=True)
    _catalog(home).write_text(yaml.safe_dump({"CLM5": {"time_offset": {"Month": None}}}))

    result = runner.invoke(cli, ["model", "register", "CLM5", "--time-offset", "Month:Latent_Heat=-2 days"])

    assert result.exit_code == 0, result.output
    assert _load_model(home, "CLM5").time_offset["Month"] == {"Latent_Heat": "-2 days"}


def test_rename_refuses_an_overlay_the_registry_cannot_merge(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    _catalog(home).parent.mkdir(parents=True)
    broken = {
        "TE": {
            "description": "mine",
            "variables": {"Soil_Evaporation": {"varname": "a"}, "Bare_Soil_Evaporation": {"varname": "b"}},
        }
    }
    _catalog(home).write_text(yaml.safe_dump(broken))

    result = runner.invoke(cli, ["model", "rename", "TE", "TE2", "--yes"])

    assert result.exit_code != 0
    assert "Conflicting variable alias definitions" in result.output
    assert yaml.safe_load(_catalog(home).read_text(encoding="utf-8")) == broken


def test_history_marks_non_mapping_backups_and_resolves_aliases(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    assert runner.invoke(cli, ["model", "register", "colm", "--description", "one"]).exit_code == 0
    assert runner.invoke(cli, ["model", "register", "colm", "--description", "two"]).exit_code == 0
    listed = _catalog(home).with_name(f"{_catalog(home).name}.20000101-000000-000000.bak")
    listed.write_text("- a\n- b\n", encoding="utf-8")

    result = runner.invoke(cli, ["model", "show", "colm", "--history", "--format", "json"])

    assert result.exit_code == 0, result.output
    entries = json.loads(result.output)
    assert any(entry["backup"] == str(listed) and entry["variables"] == "invalid" for entry in entries)
    assert any(entry["catalog_name"] == "CoLM2024" for entry in entries)


def test_repeating_a_fallback_does_not_stack_copies(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    for _ in range(2):
        result = runner.invoke(cli, ["model", "register", "CaMa", "-f", "Dam_Outflow:outflw2:m3 s-1"])
        assert result.exit_code == 0, result.output

    fallbacks = [fb.varname for fb in _load_model(home, "CaMa").variables["Dam_Outflow"].fallbacks]
    assert fallbacks.count("outflw2") == 1
