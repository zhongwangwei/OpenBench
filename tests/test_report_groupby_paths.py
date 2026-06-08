from __future__ import annotations

from openbench.util.filenames import (
    groupby_class_netcdf_filename,
    groupby_pair_dirname,
    groupby_table_filename,
    join_filename_components,
    legacy_groupby_pair_dirname,
    legacy_groupby_table_filename,
)
from openbench.util.report import ReportGenerator


def test_report_collects_safe_groupby_figures_and_statistics_recursively(tmp_path):
    case_dir = tmp_path / "case"
    group_dir = case_dir / "comparisons" / "CZ_groupby" / groupby_pair_dirname("Sim/A___B", "Ref:C*")
    group_dir.mkdir(parents=True)

    safe_table = group_dir / groupby_table_filename("Run/off", "Sim/A___B", "Ref:C*", "scores")
    safe_table.write_text("score\tAf\tOverall\nOverall_Score\t0.8\t0.8\n")
    safe_fig = group_dir / f"{safe_table.stem}_heatmap.jpg"
    safe_fig.write_bytes(b"fake jpg")
    safe_nc = group_dir / groupby_class_netcdf_filename("Run/off", "Ref:C*", "Sim/A___B", "Overall_Score", "CZ", "Af")
    safe_nc.write_bytes(b"fake nc")

    legacy_root = case_dir / "comparisons" / "CZ_groupby"
    legacy_table = legacy_root / legacy_groupby_table_filename("Runoff", "SimA", "RefA", "scores")
    legacy_table.write_text("score\tAf\tOverall\nOverall_Score\t0.7\t0.7\n")
    legacy_pair_dir = legacy_root / legacy_groupby_pair_dirname("SimB", "RefB")
    legacy_pair_dir.mkdir()
    legacy_pair_table = legacy_pair_dir / legacy_groupby_table_filename("Runoff", "SimB", "RefB", "scores")
    legacy_pair_table.write_text("score\tAf\tOverall\nOverall_Score\t0.6\t0.6\n")

    generator = ReportGenerator(
        {
            "evaluation_items": ["Run/off"],
            "metrics": {},
            "scores": {"Overall_Score": True},
            "comparisons": {},
            "general": {"comparison": True},
        },
        str(case_dir),
    )

    figures = generator._collect_figures("Run/off")
    assert figures["climate_zone_groupby"] == [
        f"comparisons/CZ_groupby/{group_dir.name}/{safe_fig.name}",
    ]

    stats = generator._collect_groupby_statistics("Run/off")
    assert stats["Climate_zone_groupby"]["statistics"][0]["file"] == safe_table.name
    assert stats["Climate_zone_groupby"]["spatial_files"] == [safe_nc.name]

    legacy_stats = generator._collect_groupby_statistics("Runoff")
    legacy_files = {entry["file"] for entry in legacy_stats["Climate_zone_groupby"]["statistics"]}
    assert legacy_files == {legacy_table.name, legacy_pair_table.name}


def test_report_copies_verifies_and_url_encodes_safe_groupby_figures(tmp_path):
    case_dir = tmp_path / "case"
    group_dir = case_dir / "comparisons" / "CZ_groupby" / groupby_pair_dirname("Sim/A___B", "Ref:C*")
    group_dir.mkdir(parents=True)

    table = group_dir / groupby_table_filename("Run/off", "Sim/A___B", "Ref:C*", "scores")
    table.write_text("score\tAf\tOverall\nOverall_Score\t0.8\t0.8\n")
    figure = group_dir / f"{table.stem}_heatmap.jpg"
    figure.write_bytes(b"fake jpg")

    generator = ReportGenerator(
        {
            "evaluation_items": ["Run/off"],
            "metrics": {},
            "scores": {"Overall_Score": True},
            "comparisons": {},
            "general": {"comparison": True},
        },
        str(case_dir),
    )
    report_data = generator._collect_report_data()

    generator._copy_figures_to_report_dir()
    generator._verify_figure_paths(report_data)
    html_path = generator._generate_html_report(report_data, "groupby_safe")

    assert (case_dir / "reports" / "figures" / "comparisons" / "CZ_groupby" / group_dir.name / figure.name).exists()

    html = open(html_path, encoding="utf-8").read()
    assert "Run%252Foff" in html
    assert "Ref%253AC%252A" in html
    assert f"figures/comparisons/CZ_groupby/{group_dir.name}/{figure.name}" not in html


def test_report_groupby_item_matching_is_component_bounded(tmp_path):
    case_dir = tmp_path / "case"
    group_dir = case_dir / "comparisons" / "CZ_groupby" / groupby_pair_dirname("SimA", "RefA")
    group_dir.mkdir(parents=True)

    wrong_table = group_dir / groupby_table_filename("Runoff", "SimA", "RefA", "scores")
    wrong_table.write_text("score\tAf\tOverall\nOverall_Score\t0.8\t0.8\n")
    wrong_fig = group_dir / f"{wrong_table.stem}_heatmap.jpg"
    wrong_fig.write_bytes(b"fake jpg")

    generator = ReportGenerator(
        {
            "evaluation_items": ["Run"],
            "metrics": {},
            "scores": {"Overall_Score": True},
            "comparisons": {},
            "general": {"comparison": True},
        },
        str(case_dir),
    )

    figures = generator._collect_figures("Run")
    assert figures["climate_zone_groupby"] == []
    assert generator._collect_groupby_statistics("Run") == {}


def test_report_collects_safe_metric_score_figures_without_item_glob_injection(tmp_path):
    case_dir = tmp_path / "case"
    metrics_dir = case_dir / "metrics"
    scores_dir = case_dir / "scores"
    metrics_dir.mkdir(parents=True)
    scores_dir.mkdir(parents=True)

    metric_fig = metrics_dir / f"{join_filename_components('Run/off', 'ref', 'Ref:A', 'sim', 'Sim*', 'bias')}.jpg"
    score_fig = (
        scores_dir / f"{join_filename_components('Run/off', 'ref', 'Ref:A', 'sim', 'Sim*', 'Overall_Score')}.jpg"
    )
    wrong_prefix_fig = metrics_dir / "Runoff_ref_RefA_sim_SimA_bias.jpg"
    metric_fig.write_bytes(b"fake jpg")
    score_fig.write_bytes(b"fake jpg")
    wrong_prefix_fig.write_bytes(b"fake jpg")

    generator = ReportGenerator(
        {
            "evaluation_items": ["Run/off"],
            "metrics": {"bias": True},
            "scores": {"Overall_Score": True},
            "comparisons": {},
            "general": {"comparison": True},
        },
        str(case_dir),
    )

    figures = generator._collect_figures("Run/off")
    assert figures["metrics"] == [metric_fig.name]
    assert figures["scores"] == [score_fig.name]
    assert generator._collect_figures("Run")["metrics"] == []


def test_report_collects_non_groupby_outputs_without_glob_injection_or_prefix_collision(tmp_path):
    import numpy as np
    import xarray as xr

    case_dir = tmp_path / "case"
    metrics_dir = case_dir / "metrics"
    scores_dir = case_dir / "scores"
    stats_dir = case_dir / "comparisons" / "Mean"
    figs_dir = case_dir / "comparisons" / "Taylor_Diagram"
    for directory in (metrics_dir, scores_dir, stats_dir, figs_dir):
        directory.mkdir(parents=True)

    item = "Run*"
    ref = "Ref:A"
    sim = "Sim?1"
    metric_file = metrics_dir / f"{join_filename_components(item, 'ref', ref, 'sim', sim, 'bias')}.nc"
    score_file = scores_dir / f"{join_filename_components(item, 'ref', ref, 'sim', sim, 'Overall_Score')}.nc"
    wrong_prefix = metrics_dir / f"{join_filename_components('Runoff', 'ref', ref, 'sim', sim, 'bias')}.nc"
    ds = xr.Dataset({"bias": (("lat", "lon"), np.array([[1.0]]))}, coords={"lat": [0.0], "lon": [0.0]})
    ds.to_netcdf(metric_file)
    ds.to_netcdf(score_file)
    ds.to_netcdf(wrong_prefix)

    stat_file = stats_dir / f"{join_filename_components(item, 'ref', ref, 'sim', sim, 'Mean')}.nc"
    stat_file.write_bytes(b"placeholder")
    fig_file = figs_dir / f"{join_filename_components(item, 'ref', ref, 'sim', sim, 'Taylor_Diagram')}.jpg"
    fig_file.write_bytes(b"fake jpg")
    wrong_fig = figs_dir / f"{join_filename_components('Runoff', 'ref', ref, 'sim', sim, 'Taylor_Diagram')}.jpg"
    wrong_fig.write_bytes(b"fake jpg")

    generator = ReportGenerator(
        {
            "evaluation_items": [item],
            "metrics": {"bias": True},
            "scores": {"Overall_Score": True},
            "comparisons": {"Taylor_Diagram": True},
            "general": {"comparison": True},
        },
        str(case_dir),
    )

    stats = generator._generate_grid_vs_grid_stats(item)
    assert list(stats) == [f"{ref} vs {sim}"]
    assert stats[f"{ref} vs {sim}"]["metrics"]["bias"]["mean"] == 1.0
    assert stats[f"{ref} vs {sim}"]["metrics"]["Overall_Score"]["mean"] == 1.0
    assert generator._generate_grid_vs_grid_stats("Run") == {}
    assert generator._collect_statistics(item)["Mean"] == [stat_file.name]
    assert generator._collect_figures(item)["comparisons"] == [f"Taylor_Diagram/{fig_file.name}"]


def test_report_legacy_item_matching_avoids_configured_underscore_prefix_collision(tmp_path):
    case_dir = tmp_path / "case"
    metrics_dir = case_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "Run_off_ref_RefA_sim_SimA_bias.jpg").write_bytes(b"fake jpg")
    (metrics_dir / "Run_ref_RefA_sim_SimA_bias.jpg").write_bytes(b"fake jpg")

    generator = ReportGenerator(
        {
            "evaluation_items": ["Run", "Run_off"],
            "metrics": {"bias": True},
            "scores": {},
            "comparisons": {},
            "general": {"comparison": True},
        },
        str(case_dir),
    )

    assert generator._collect_figures("Run")["metrics"] == ["Run_ref_RefA_sim_SimA_bias.jpg"]
    assert generator._collect_figures("Run_off")["metrics"] == ["Run_off_ref_RefA_sim_SimA_bias.jpg"]
