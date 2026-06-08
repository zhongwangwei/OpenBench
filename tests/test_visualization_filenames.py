import unicodedata
from pathlib import Path, PurePath

import pytest

from openbench.util.filenames import (
    diff_grid_anomaly_filename,
    diff_grid_difference_filename,
    diff_station_anomaly_filename,
    diff_station_difference_filename,
    groupby_class_netcdf_filename,
    groupby_class_netcdf_stem,
    groupby_pair_dirname,
    groupby_table_filename,
    legacy_groupby_pair_dirname,
    legacy_groupby_table_filename,
    relative_grid_score_filename,
    relative_station_scores_filename,
)
from openbench.visualization._filenames import filename_component, join_filename_components

ROOT = Path(__file__).resolve().parents[1]


def test_join_filename_components_keeps_underscore_boundaries_distinct():
    assert join_filename_components("A_B", "C") != join_filename_components("A", "B_C")


def test_filename_component_escapes_path_and_reserved_characters():
    component = filename_component("Ref/A:B*C?D__E%")

    assert "/" not in PurePath(component).parts
    assert "__" not in component
    assert "%2F" in component
    assert "%3A" in component
    assert "%2A" in component
    assert "%3F" in component
    assert "%5F%5F" in component
    assert "%25" in component


@pytest.mark.parametrize("name", ["CON", "nul", "COM1.txt", "LPT9"])
def test_filename_component_escapes_windows_reserved_device_names(name):
    component = filename_component(name)

    assert component.upper().split(".", 1)[0] not in {"CON", "NUL", "COM1", "LPT9"}
    assert component != name


def test_filename_component_escapes_trailing_dot_space_and_normalizes_unicode():
    assert filename_component("case.").endswith("%2E")
    assert filename_component("case ") == "case%20"
    assert filename_component("A\u030a") == filename_component(unicodedata.normalize("NFC", "A\u030a"))


def test_filename_component_bounds_very_long_components_with_hash_suffix():
    component = filename_component("x" * 400)

    assert len(component) < 230
    assert component.startswith("x" * 180)
    assert "__sha256-" in component


def test_filename_component_keeps_drive_and_unc_paths_inside_one_component():
    drive = filename_component(r"C:\data\file.nc")
    unc = filename_component(r"\\server\share\data.nc")

    assert "/" not in drive and "\\" not in drive
    assert "/" not in unc and "\\" not in unc
    assert "%3A" in drive and "%5C" in drive
    assert "%5C%5Cserver" in unc


def test_only_drawing_file_lookup_falls_back_to_legacy_names(tmp_path):
    from openbench.visualization.Mod_Only_Drawing import _require_only_drawing_file

    safe_path = tmp_path / "taylor_diagram__Runoff__RefA.csv"
    legacy_path = tmp_path / "taylor_diagram_Runoff_RefA.csv"
    legacy_path.write_text("legacy")

    assert _require_only_drawing_file(str(safe_path), producer="comparison", fallback_paths=[str(legacy_path)]) == str(
        legacy_path
    )


def test_only_drawing_file_lookup_falls_back_to_legacy_txt_names(tmp_path):
    from openbench.visualization.Mod_Only_Drawing import _require_only_drawing_file

    safe_path = tmp_path / "target_diagram__Runoff__RefA.csv"
    legacy_txt_path = tmp_path / "target_diagram_Runoff_RefA.txt"
    legacy_txt_path.write_text("legacy txt")

    legacy_csv_path = tmp_path / "target_diagram_Runoff_RefA.csv"

    assert _require_only_drawing_file(
        str(safe_path),
        producer="comparison",
        fallback_ext=".txt",
        fallback_paths=[str(legacy_csv_path)],
    ) == str(legacy_txt_path)


def test_relative_score_intermediate_filenames_are_safe_components():
    assert relative_station_scores_filename("Run/off", "Ref:A", "Sim*B") == (
        "Run%2Foff__stn__Ref%3AA__Sim%2AB__relative_scores.csv"
    )
    assert relative_grid_score_filename("Run_off", "Ref", "Sim", "Overall_Score") == (
        "Run_off__ref__Ref__sim__Sim__RelativeOverall_Score.nc"
    )


def test_diff_plot_intermediate_filenames_are_safe_components():
    assert diff_station_anomaly_filename("Run/off", "Ref:A", "Sim*B", "bias") == (
        "Run%2Foff__stn__Ref%3AA__sim__Sim%2AB__bias__anomaly.csv"
    )
    assert diff_station_difference_filename("Run", "Ref", "Sim_1", "var/a", "Sim", "var:b", "score") == (
        "Run__stn__Ref__Sim_1__var%2Fa__vs__Sim__var%3Ab__score__diff.csv"
    )
    assert diff_grid_anomaly_filename("Run/off", "Ref:A", "Sim*B", "bias") == (
        "Run%2Foff__ref__Ref%3AA__sim__Sim%2AB__bias__anomaly.nc"
    )
    assert diff_grid_difference_filename("Run", "Ref", "Sim_1", "Sim", "Overall_Score") == (
        "Run__ref__Ref__Sim_1__vs__Sim__Overall_Score__diff.nc"
    )


def test_groupby_output_filenames_are_safe_and_legacy_names_are_distinct():
    assert groupby_pair_dirname("Sim/A___B", "Ref:C*") == "Sim%2FA%5F%5F_B__Ref%3AC%2A"
    assert groupby_table_filename("Run/off", "Sim/A___B", "Ref:C*", "scores") == (
        "Run%2Foff__Sim%2FA%5F%5F_B__Ref%3AC%2A__scores.csv"
    )
    assert groupby_class_netcdf_filename("Run/off", "Ref:C*", "Sim/A___B", "bias", "CZ", "A/B") == (
        "Run%2Foff__ref__Ref%3AC%2A__sim__Sim%2FA%5F%5F_B__bias__CZ__A%2FB.nc"
    )
    assert legacy_groupby_pair_dirname("SimA", "RefA") == "SimA___RefA"
    assert legacy_groupby_table_filename("Runoff", "SimA", "RefA", "scores") == ("Runoff_SimA___RefA_scores.csv")


def test_basic_plot_outputs_use_safe_filename_components():
    source = (ROOT / "src/openbench/visualization/Fig_Basic_Plot.py").read_text(encoding="utf-8")

    assert "join_filename_components(self.item, 'ref', self.ref_source, 'sim', self.sim_source, xitem)" in source
    assert "join_filename_components(key[0], ID, 'timeseries')" in source
    assert "join_filename_components(self.item, 'stn', self.ref_source, self.sim_source, varname)" in source
    assert (
        "self.casedir}/{k}/{self.item}_ref_{self.ref_source}_sim_{self.sim_source}_{xitem}.{option['saving_format']}"
        not in source
    )
    assert (
        "self.casedir}/{s_m}/{self.item}_stn_{self.ref_source}_{self.sim_source}_{varname}.{option['saving_format']}"
        not in source
    )
    assert (
        "data/stn_{self.ref_source}_{self.sim_source}/{key[0]}_{ID}_timeseries.{option['saving_format']}" not in source
    )


def test_statistical_figure_suffixes_escape_dynamic_titles():
    files = [
        ROOT / "src/openbench/visualization/Fig_ANOVA.py",
        ROOT / "src/openbench/visualization/Fig_Mann_Kendall_Trend_Test.py",
        ROOT / "src/openbench/visualization/Fig_Partial_Least_Squares_Regression.py",
        ROOT / "src/openbench/visualization/Fig_Three_Cornered_Hat.py",
        ROOT / "src/openbench/visualization/Fig_stn_plot_index.py",
    ]

    for path in files:
        source = path.read_text(encoding="utf-8")
        assert "filename_component(" in source, path
        assert "_{title}.{option['saving_format']}" not in source, path
        assert "_{type}.{option['saving_format']}" not in source, path


def test_pls_statistical_map_uses_netcdf_stem_for_figure_filename():
    source = (ROOT / "src/openbench/visualization/Fig_Partial_Least_Squares_Regression.py").read_text(encoding="utf-8")

    assert "file2 = file[:-3]" in source
    assert "f\"{file2}_{filename_component(title)}.{option['saving_format']}\"" in source
    assert "f\"{file}_{filename_component(title)}.{option['saving_format']}\"" not in source


def test_lc_heatmap_finds_safe_groupby_class_netcdf_files(tmp_path):
    from openbench.visualization.Fig_LC_based_heat_map import (
        _groupby_class_netcdf_files,
        _require_groupby_class_netcdf_files,
    )

    safe_stem = groupby_class_netcdf_stem("Run/off", "Ref:C*", "Sim/A___B", "bias", "CZ")
    safe_file = tmp_path / f"{safe_stem}__A%2FB.nc"
    legacy_file = tmp_path / "Runoff_ref_RefA_sim_SimA_bias_CZ_Af.nc"
    safe_file.write_text("safe")
    legacy_file.write_text("legacy")

    assert _groupby_class_netcdf_files(
        {"path": str(tmp_path) + "/", "item": ["Run/off", "Sim/A___B", "Ref:C*"], "groupby": "CZ_groupby"},
        "bias",
    ) == [str(safe_file)]
    assert _groupby_class_netcdf_files(
        {"path": str(tmp_path) + "/", "item": ["Runoff", "SimA", "RefA"], "groupby": "CZ_groupby"},
        "bias",
    ) == [str(legacy_file)]

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="heatmap missing per-class NetCDF inputs"):
        _require_groupby_class_netcdf_files(
            {"path": str(empty_dir) + "/", "item": ["Runoff", "SimA", "RefA"], "groupby": "CZ_groupby"},
            "bias",
        )


def test_lc_heatmap_opens_safe_groupby_class_bundle_as_distribution(tmp_path):
    import numpy as np
    import xarray as xr

    from openbench.visualization.Fig_LC_based_heat_map import _open_groupby_class_distribution

    safe_stem = groupby_class_netcdf_stem("Runoff", "RefA", "SimA", "bias", "CZ")
    xr.Dataset(
        {"bias": (("class", "lat", "lon"), np.arange(8, dtype=float).reshape(2, 2, 2))},
        coords={"class": ["C1", "C2"], "lat": [0, 1], "lon": [10, 11]},
    ).to_netcdf(tmp_path / f"{safe_stem}__classes.nc")

    combined = _open_groupby_class_distribution(
        {"path": str(tmp_path) + "/", "item": ["Runoff", "SimA", "RefA"], "groupby": "CZ_groupby"},
        "bias",
    )

    assert "time" in combined.dims
    assert "class" not in combined.dims
    assert combined.sizes["time"] == 2
    np.testing.assert_array_equal(combined["bias"].values, np.arange(8, dtype=float).reshape(2, 2, 2))


def test_relative_score_renderers_fall_back_to_legacy_intermediate_names(tmp_path):
    from openbench.visualization.Fig_Relative_Score import (
        _relative_grid_score_path,
        _relative_station_scores_path,
    )

    legacy_stn = tmp_path / "Runoff_stn_RefA_SimA_relative_scores.csv"
    legacy_grid = tmp_path / "Runoff_ref_RefA_sim_SimA_RelativeOverall_Score.nc"
    legacy_stn.write_text("legacy station")
    legacy_grid.write_text("legacy grid")

    assert _relative_station_scores_path(str(tmp_path), "Runoff", "RefA", "SimA") == str(legacy_stn)
    assert _relative_grid_score_path(str(tmp_path), "Runoff", "RefA", "SimA", "Overall_Score") == str(legacy_grid)


def test_diff_plot_renderer_falls_back_to_legacy_intermediate_names(tmp_path):
    from openbench.visualization.Fig_Diff_Plot import _diff_input_filename

    sim_nml = {"Runoff": {"SimA_varname": "flow", "SimB_varname": "flow"}}
    legacy_grid = tmp_path / "Runoff_ref_RefA_sim_SimA_bias_anomaly.nc"
    legacy_station = tmp_path / "Runoff_stn_RefA_SimA_flow_vs_SimB_flow_bias_diff.csv"
    legacy_grid.write_text("legacy grid")
    legacy_station.write_text("legacy station")

    assert _diff_input_filename(str(tmp_path), "anomaly", "bias", "Runoff", "RefA", "SimA", sim_nml, "grid") == (
        legacy_grid.name
    )
    assert (
        _diff_input_filename(str(tmp_path), "difference", "bias", "Runoff", "RefA", ("SimA", "SimB"), sim_nml, "stn")
        == legacy_station.name
    )


def test_only_drawing_legacy_fallback_matrix_covers_migration_without_cli_tool():
    source = (ROOT / "src/openbench/visualization/only_drawing.py").read_text(encoding="utf-8")

    assert "legacy_groupby_pair_dirname" in source
    assert "legacy_groupby_table_filename" in source
    assert "taylor_diagram_{evaluation_item}_{ref_source}.csv" in source
    assert "target_diagram_{evaluation_item}_{ref_source}.csv" in source
    assert "_legacy_diff_grid_anomaly_path" in source
    assert "_legacy_diff_station_difference_path" in source
    assert "_legacy_relative_station_scores_path" in source
    assert "_legacy_relative_grid_score_path" in source
