from __future__ import annotations

from pathlib import Path

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


def test_report_grid_stats_do_not_store_full_value_arrays(tmp_path):
    import numpy as np
    import xarray as xr

    case_dir = tmp_path / "case"
    metrics_dir = case_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    item = "Runoff"
    ref = "RefA"
    sim = "SimA"
    metric_file = metrics_dir / f"{join_filename_components(item, 'ref', ref, 'sim', sim, 'bias')}.nc"
    xr.Dataset({"bias": (("lat", "lon"), np.array([[1.0, 2.0], [np.nan, 4.0]]))}).to_netcdf(metric_file)

    generator = ReportGenerator(
        {
            "evaluation_items": [item],
            "metrics": {"bias": True},
            "scores": {},
            "ref_nml": {"general": {f"{item}_ref_source": ref}},
            "sim_nml": {"general": {"Case_lib": sim}},
            "comparisons": {},
            "general": {"comparison": False},
        },
        str(case_dir),
    )

    stats = generator._generate_grid_vs_grid_stats(item)
    metric = stats[f"{ref} vs {sim}"]["metrics"]["bias"]
    assert "values" not in metric
    assert metric["mean"] == 7 / 3


def test_report_csv_collection_streams_summary_without_full_records(tmp_path):
    case_dir = tmp_path / "case"
    metrics_dir = case_dir / "metrics"
    scores_dir = case_dir / "scores"
    metrics_dir.mkdir(parents=True)
    scores_dir.mkdir(parents=True)
    (metrics_dir / "Runoff_evaluations.csv").write_text("station,bias\nA,1\nB,3\n", encoding="utf-8")
    (scores_dir / "Runoff_evaluations.csv").write_text("station,Overall_Score\nA,0\nB,0\n", encoding="utf-8")

    generator = ReportGenerator(
        {
            "evaluation_items": ["Runoff"],
            "metrics": {"bias": True},
            "scores": {"Overall_Score": True},
            "comparisons": {},
        },
        str(case_dir),
    )
    data = generator._collect_metrics_data("Runoff")

    assert data["Runoff"]["row_count"] == 2
    assert "data" not in data["Runoff"]
    assert data["Runoff"]["summary"]["bias"] == {
        "mean": 2.0,
        "std": 2**0.5,
        "min": 1.0,
        "max": 3.0,
        "count": 2,
    }
    assert generator._summarize_csv(str(metrics_dir / "Runoff_evaluations.csv"), chunksize=1) == (
        2,
        data["Runoff"]["summary"],
    )
    report_data = generator._collect_report_data()
    assert report_data["overall_summary"]["grand_average"] == 0.0
    html = Path(generator._generate_html_report(report_data, "csv_summary")).read_text(encoding="utf-8")
    assert "Overall Average Score" in html
    assert "0.000" in html


def test_report_streams_large_netcdf_statistics(tmp_path, monkeypatch):
    import numpy as np
    import xarray as xr

    import openbench.util.report as report_module

    case_dir = tmp_path / "case"
    metrics_dir = case_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    metric_file = metrics_dir / f"{join_filename_components('Runoff', 'ref', 'RefA', 'sim', 'SimA', 'bias')}.nc"
    xr.Dataset({"bias": (("lat", "lon"), np.array([[1.0, 2.0], [np.nan, 4.0]]))}).to_netcdf(metric_file)
    monkeypatch.setattr(report_module, "_MAX_REPORT_STAT_POINTS", 2)

    generator = ReportGenerator(
        {
            "evaluation_items": ["Runoff"],
            "metrics": {"bias": True},
            "scores": {},
            "ref_nml": {"general": {"Runoff_ref_source": "RefA"}},
            "sim_nml": {"general": {"Case_lib": "SimA"}},
            "comparisons": {},
        },
        str(case_dir),
    )

    data = generator._collect_metrics_data("Runoff")
    metric = data["RefA vs SimA"]["metrics"]["bias"]
    assert metric["mean"] == 7 / 3
    assert metric["std"] == np.std([1.0, 2.0, 4.0])
    assert metric["min"] == 1.0
    assert metric["max"] == 4.0
    assert metric["coverage"] == 75.0
    assert metric["median"] is None
    assert metric["median_omitted"] is True

    report_data = generator._collect_report_data()
    html = Path(generator._generate_html_report(report_data, "large_nc")).read_text(encoding="utf-8")
    assert "Mean = 2.3333" in html
    assert "Median = omitted for large result" in html
    assert "Summary omitted" not in html


def test_report_netcdf_summary_never_reads_more_than_chunk_limit():
    import numpy as np

    class LoadedChunk:
        def __init__(self, values):
            self.values = values

    class ChunkedArray:
        dims = ("lat", "lon")

        def __init__(self):
            self.data = np.arange(12, dtype=float).reshape(3, 4)
            self.shape = self.data.shape
            self.size = self.data.size
            self.chunk_sizes = []

        @property
        def values(self):
            raise AssertionError("the complete array must not be loaded eagerly")

        def isel(self, indexers):
            chunk = self.data[tuple(indexers[dim] for dim in self.dims)]
            self.chunk_sizes.append(chunk.size)
            return LoadedChunk(chunk)

    values = ChunkedArray()
    stats = ReportGenerator._summarize_data_array(values, max_points=5)

    assert max(values.chunk_sizes) <= 5
    assert stats["mean"] == 5.5
    assert stats["median"] is None


def test_report_does_not_invent_correlation_for_large_real_result(tmp_path, monkeypatch):
    import numpy as np
    import xarray as xr

    import openbench.util.report as report_module

    case_dir = tmp_path / "case"
    metrics_dir = case_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    correlation_file = metrics_dir / join_filename_components("Runoff", "ref", "RefA", "sim", "SimA", "correlation")
    xr.Dataset({"correlation": (("lat", "lon"), np.array([[0.1, 0.2], [np.nan, 0.9]]))}).to_netcdf(
        correlation_file.with_suffix(".nc")
    )
    monkeypatch.setattr(report_module, "_MAX_REPORT_STAT_POINTS", 2)

    generator = ReportGenerator(
        {
            "evaluation_items": ["Runoff"],
            "metrics": {"correlation": True},
            "scores": {},
            "ref_nml": {"general": {"Runoff_ref_source": "RefA"}},
            "sim_nml": {"general": {"Case_lib": "SimA"}},
            "comparisons": {},
        },
        str(case_dir),
    )

    metric = generator._generate_grid_vs_grid_stats("Runoff")["RefA vs SimA"]["metrics"]["correlation"]
    assert metric["mean"] == 0.4
    assert metric["coverage"] == 75.0
    assert metric["median"] is None
    assert metric["median_omitted"] is True


def test_report_summaries_ignore_infinite_values(tmp_path):
    import numpy as np
    import pandas as pd
    import xarray as xr

    nc = xr.DataArray([1.0, np.inf, -np.inf, np.nan], dims=["x"])
    stats = ReportGenerator._summarize_data_array(nc)
    assert stats is not None
    assert stats["mean"] == 1.0
    assert stats["valid_points"] == 1

    csv = tmp_path / "values.csv"
    pd.DataFrame({"bias": [1.0, np.inf, -np.inf, np.nan]}).to_csv(csv, index=False)
    _rows, summary = ReportGenerator._summarize_csv(str(csv))
    assert summary["bias"]["mean"] == 1.0
    assert summary["bias"]["count"] == 1


def test_report_extracts_exact_metric_and_score_names(tmp_path):
    generator = ReportGenerator(
        {
            "evaluation_items": ["Runoff"],
            "metrics": {"correlation": True, "correlation_R2": True, "RMSE": True},
            "scores": {"nRMSEScore": True, "Overall_Score": True},
        },
        str(tmp_path / "case"),
    )

    assert generator._extract_metric_type("Runoff_ref_Ref_sim_Sim_correlation_R2.nc") == "correlation_R2"
    assert generator._extract_metric_type("Runoff_ref_Ref_sim_Sim_nRMSEScore.nc") == "nRMSEScore"
    assert (
        generator._extract_metric_type(
            f"{join_filename_components('Run/off', 'ref', 'Ref', 'sim', 'Sim', 'Overall_Score')}.nc"
        )
        == "Overall_Score"
    )

    only_short = ReportGenerator(
        {
            "evaluation_items": ["Runoff"],
            "metrics": {"correlation_R2": True},
            "scores": {},
        },
        str(tmp_path / "case2"),
    )
    assert only_short._extract_metric_type("Runoff_ref_Ref_sim_Sim_ubcorrelation_R2.nc") == "Unknown"


def test_report_reads_v3_source_keys_and_keeps_grid_scores_out_of_metrics(tmp_path):
    import numpy as np
    import xarray as xr

    case_dir = tmp_path / "case"
    metrics_dir = case_dir / "metrics"
    scores_dir = case_dir / "scores"
    metrics_dir.mkdir(parents=True)
    scores_dir.mkdir(parents=True)

    item = "Runoff"
    ref = "RefA"
    sim = "SimA"
    xr.Dataset({"bias": (("lat", "lon"), np.array([[1.0, 3.0]]))}).to_netcdf(
        metrics_dir / f"{join_filename_components(item, 'ref', ref, 'sim', sim, 'bias')}.nc"
    )
    xr.Dataset({"Overall_Score": (("lat", "lon"), np.array([[0.2, 0.8]]))}).to_netcdf(
        scores_dir / f"{join_filename_components(item, 'ref', ref, 'sim', sim, 'Overall_Score')}.nc"
    )
    xr.Dataset({"RMSE": (("lat", "lon"), np.array([[99.0, 99.0]]))}).to_netcdf(
        metrics_dir / f"{join_filename_components(item, 'ref', ref, 'sim', sim, 'RMSE')}.nc"
    )

    generator = ReportGenerator(
        {
            "evaluation_items": [item],
            "metrics": {"bias": True},
            "scores": {"Overall_Score": True},
            "ref_nml": {"general": {f"{item}_ref_source": ref}},
            "sim_nml": {"general": {f"{item}_sim_source": sim}},
            "comparisons": {},
        },
        str(case_dir),
    )

    report_data = generator._collect_report_data()
    item_data = report_data["evaluation_items"][item]
    assert item_data["metrics"][f"{ref} vs {sim}"]["metrics"] == {
        "bias": item_data["metrics"][f"{ref} vs {sim}"]["metrics"]["bias"],
        "use_eyear": item_data["metrics"][f"{ref} vs {sim}"]["metrics"]["use_eyear"],
        "use_syear": item_data["metrics"][f"{ref} vs {sim}"]["metrics"]["use_syear"],
    }
    assert "Overall_Score" not in item_data["metrics"][f"{ref} vs {sim}"]["metrics"]
    assert "Unknown" not in item_data["metrics"][f"{ref} vs {sim}"]["metrics"]
    assert item_data["scores"][f"{ref} vs {sim}"]["metrics"]["Overall_Score"]["mean"] == 0.5
    assert report_data["overall_summary"]["grand_average"] == 0.5

    html = Path(generator._generate_html_report(report_data, "grid_scores")).read_text(encoding="utf-8")
    assert "Evaluation Scores" in html
    assert "Overall_Score" in html
    assert "0.500" in html


def test_report_collects_comparison_csv_for_selected_score_and_non_jpg_figures(tmp_path):
    case_dir = tmp_path / "case"
    heatmap_dir = case_dir / "comparisons" / "HeatMap"
    radar_dir = case_dir / "comparisons" / "RadarMap"
    parallel_dir = case_dir / "comparisons" / "Parallel_Coordinates"
    for directory in (heatmap_dir, radar_dir, parallel_dir):
        directory.mkdir(parents=True)

    (heatmap_dir / "scenarios_nRMSEScore_comparison.csv").write_text("item,score\nRunoff,0.9\n", encoding="utf-8")
    (radar_dir / "scenarios_nRMSEScore_comparison.csv").write_text("item,score\nRunoff,0.8\n", encoding="utf-8")
    (parallel_dir / "Parallel_Coordinates_evaluations.csv").write_text(
        "item\tnRMSEScore\nRunoff\t0.7\n", encoding="utf-8"
    )
    (heatmap_dir / "case_heatmap.png").write_bytes(b"png")
    (radar_dir / "case_radarmap.svg").write_bytes(b"svg")
    (parallel_dir / "case_nRMSEScore.webp").write_bytes(b"webp")

    generator = ReportGenerator(
        {
            "evaluation_items": ["Runoff"],
            "metrics": {},
            "scores": {"nRMSEScore": True},
            "comparisons": {"HeatMap": True, "RadarMap": True, "Parallel_Coordinates": True},
            "general": {"comparison": True},
        },
        str(case_dir),
    )

    data = generator._collect_comparison_data()
    assert data["heatmap"] == [{"item": "Runoff", "score": 0.9}]
    assert data["radar"] == [{"item": "Runoff", "score": 0.8}]
    assert data["parallel_coords"] == [{"item": "Runoff", "nRMSEScore": 0.7}]
    assert data["score"] == "nRMSEScore"
    assert data["figures"] == {
        "heatmap": "HeatMap/case_heatmap.png",
        "parallel_coords": "Parallel_Coordinates/case_nRMSEScore.webp",
        "radar": "RadarMap/case_radarmap.svg",
    }

    html = Path(
        generator._generate_html_report(
            {
                "metadata": generator.metadata,
                "evaluation_items": {
                    "Runoff": {
                        "metrics": {},
                        "scores": {},
                        "figures": {
                            "metrics": [],
                            "scores": [],
                            "comparisons": ["HeatMap/case_heatmap.png"],
                            "igbp_groupby": [],
                            "pft_groupby": [],
                            "climate_zone_groupby": [],
                        },
                        "statistics": {},
                    }
                },
                "overall_summary": {"total_items": 1},
                "comparisons": data,
                "groupby_summary": {"total_groupby_analyses": 0},
            },
            "non_jpg_caption",
        )
    ).read_text(encoding="utf-8")
    assert "case heatmap</div>" in html
    assert "case heatmap.png</div>" not in html
    assert "nRMSEScore Comparison Heatmap" in html

    generator._copy_figures_to_report_dir()
    assert (case_dir / "reports" / "figures" / "comparisons" / "HeatMap" / "case_heatmap.png").exists()


def test_report_station_csv_metric_and_score_summaries_do_not_cross(tmp_path):
    case_dir = tmp_path / "case"
    metrics_dir = case_dir / "metrics"
    scores_dir = case_dir / "scores"
    metrics_dir.mkdir(parents=True)
    scores_dir.mkdir(parents=True)

    (metrics_dir / "Runoff_evaluations.csv").write_text(
        "station,bias,Overall_Score,RMSE,correlation\nA,1,0.2,5,0.9\nB,3,0.8,7,0.8\n",
        encoding="utf-8",
    )
    (scores_dir / "Runoff_evaluations.csv").write_text(
        "station,bias,Overall_Score,RMSE,correlation\nA,1,0.2,5,0.9\nB,3,0.8,7,0.8\n",
        encoding="utf-8",
    )

    generator = ReportGenerator(
        {
            "evaluation_items": ["Runoff"],
            "metrics": {"bias": True},
            "scores": {"Overall_Score": True},
            "comparisons": {},
        },
        str(case_dir),
    )

    report_data = generator._collect_report_data()
    item_data = report_data["evaluation_items"]["Runoff"]
    assert set(item_data["metrics"]["Runoff"]["summary"]) == {"bias"}
    assert set(item_data["scores"]["Runoff"]["summary"]) == {"Overall_Score"}

    html = Path(generator._generate_html_report(report_data, "station_no_cross")).read_text(encoding="utf-8")
    metrics_section = html.split("Evaluation Scores", 1)[0]
    scores_section = html.split("Evaluation Scores", 1)[1].split("Metric Visualizations", 1)[0]
    assert "<strong>Overall_Score:</strong>" not in metrics_section
    assert "<strong>bias:</strong>" not in scores_section


def test_report_collects_station_timeseries_for_current_item_only(tmp_path):
    case_dir = tmp_path / "case"
    good_dir = case_dir / "data" / "stn_RefA_SimA"
    nested = good_dir / "nested"
    nested.mkdir(parents=True)
    safe_fig = good_dir / f"{join_filename_components('Run/off', 'US-A', 'timeseries')}.jpg"
    legacy_fig = nested / "Runoff_US-B_timeseries.png"
    wrong_item = good_dir / "Runoff_US-C_timeseries.jpg"
    near_prefix = good_dir / "Run_US-D_timeseries.jpg"
    for path in (safe_fig, legacy_fig, wrong_item, near_prefix):
        path.write_bytes(b"fake image")

    generator = ReportGenerator(
        {
            "evaluation_items": ["Run/off", "Runoff"],
            "metrics": {},
            "scores": {},
            "comparisons": {},
        },
        str(case_dir),
    )

    figures = generator._collect_figures("Run/off")
    assert figures["station_timeseries"] == [f"data/stn_RefA_SimA/{safe_fig.name}"]

    runoff_figures = generator._collect_figures("Runoff")
    assert runoff_figures["station_timeseries"] == [
        "data/stn_RefA_SimA/Runoff_US-C_timeseries.jpg",
        "data/stn_RefA_SimA/nested/Runoff_US-B_timeseries.png",
    ]


def test_report_copies_verifies_and_renders_station_timeseries(tmp_path):
    case_dir = tmp_path / "case"
    data_dir = case_dir / "data" / "stn_RefA_SimA"
    data_dir.mkdir(parents=True)
    figure = data_dir / f"{join_filename_components('Run/off', 'Site 1', 'timeseries')}.jpg"
    figure.write_bytes(b"fake image")

    generator = ReportGenerator(
        {
            "evaluation_items": ["Run/off"],
            "metrics": {},
            "scores": {},
            "comparisons": {},
        },
        str(case_dir),
    )
    report_data = generator._collect_report_data()

    generator._copy_figures_to_report_dir()
    generator._verify_figure_paths(report_data)
    html_path = generator._generate_html_report(report_data, "station_timeseries")

    copied = case_dir / "reports" / "figures" / "data" / "stn_RefA_SimA" / figure.name
    assert copied.exists()
    html = Path(html_path).read_text(encoding="utf-8")
    assert "Station Time Series" in html
    assert f"figures/data/stn_RefA_SimA/{figure.name}" not in html
    assert "Run%252Foff" in html


def test_report_ignores_hidden_sidecars_and_replaces_stale_figure_copies(tmp_path):
    case_dir = tmp_path / "case"
    metrics_dir = case_dir / "metrics"
    copied_dir = case_dir / "reports" / "figures" / "metrics"
    output_figures_dir = case_dir / "figures"
    metrics_dir.mkdir(parents=True)
    copied_dir.mkdir(parents=True)
    output_figures_dir.mkdir(parents=True)
    (metrics_dir / "Runoff_bias.jpg").write_bytes(b"figure")
    (metrics_dir / "._Runoff_bias.jpg").write_bytes(b"sidecar")
    (copied_dir / "stale.jpg").write_bytes(b"stale")
    (copied_dir / "._stale.jpg").write_bytes(b"sidecar")
    (output_figures_dir / "stale.jpg").write_bytes(b"stale")

    generator = ReportGenerator(
        {"evaluation_items": ["Runoff"], "metrics": {}, "scores": {}, "comparisons": {}},
        str(case_dir),
    )
    generator._copy_figures_to_report_dir()

    assert (copied_dir / "Runoff_bias.jpg").read_bytes() == b"figure"
    assert not (copied_dir / "._Runoff_bias.jpg").exists()
    assert not (copied_dir / "stale.jpg").exists()
    assert not (copied_dir / "._stale.jpg").exists()
    assert (output_figures_dir / "metrics" / "Runoff_bias.jpg").read_bytes() == b"figure"
    assert not (output_figures_dir / "stale.jpg").exists()


def test_report_renders_taylor_and_heatmap_once_without_heatmap_item_duplicate(tmp_path):
    case_dir = tmp_path / "case"
    heatmap_dir = case_dir / "comparisons" / "HeatMap"
    taylor_dir = case_dir / "comparisons" / "Taylor_Diagram"
    heatmap_dir.mkdir(parents=True)
    taylor_dir.mkdir(parents=True)
    heatmap = heatmap_dir / "scenarios_Overall_Score_comparison_heatmap.jpg"
    taylor = taylor_dir / f"{join_filename_components('Taylor_Diagram', 'Latent_Heat', 'RefA')}.jpg"
    heatmap.write_bytes(b"heatmap")
    taylor.write_bytes(b"taylor")
    (heatmap_dir / "scenarios_Overall_Score_comparison.csv").write_text("item\tscore\nLatent_Heat\t0.9\n")

    generator = ReportGenerator(
        {
            "evaluation_items": ["Latent_Heat"],
            "metrics": {},
            "scores": {"Overall_Score": True},
            "comparisons": {"HeatMap": True, "Taylor_Diagram": True},
            "general": {"comparison": True},
        },
        str(case_dir),
    )
    report_data = generator._collect_report_data()

    assert report_data["evaluation_items"]["Latent_Heat"]["figures"]["comparisons"] == [f"Taylor_Diagram/{taylor.name}"]
    assert report_data["comparisons"]["figures"] == {"heatmap": f"HeatMap/{heatmap.name}"}

    generator._copy_figures_to_report_dir()
    html = Path(generator._generate_html_report(report_data, "comparison_once")).read_text(encoding="utf-8")

    assert html.count("figures/comparisons/HeatMap/") == 1
    assert html.count("figures/comparisons/Taylor_Diagram/") == 1
    assert heatmap.name in html
    assert taylor.name in html


def test_report_assigns_station_timeseries_by_configured_varname_alias_and_pair(tmp_path):
    case_dir = tmp_path / "case"
    site_dir = case_dir / "data" / join_filename_components("stn", "FLUXNET_PLUMBER2", "Site")
    other_dir = case_dir / "data" / join_filename_components("stn", "FLUXNET_PLUMBER2", "OtherSite")
    site_dir.mkdir(parents=True)
    other_dir.mkdir(parents=True)
    qle = site_dir / "Qle_cor__IT-Isp__timeseries.jpg"
    qh = site_dir / "Qh_cor__IT-Isp__timeseries.jpg"
    wrong_pair = other_dir / "Qle_cor__IT-Isp__timeseries.jpg"
    for path in (qle, qh, wrong_pair):
        path.write_bytes(b"fake image")

    config = {
        "evaluation_items": ["Latent_Heat", "Sensible_Heat"],
        "metrics": {},
        "scores": {},
        "comparisons": {},
        "ref_nml": {
            "general": {
                "Latent_Heat_ref_source": "FLUXNET_PLUMBER2",
                "Sensible_Heat_ref_source": "FLUXNET_PLUMBER2",
            },
            "Latent_Heat": {"FLUXNET_PLUMBER2_varname": "Qle_cor"},
            "Sensible_Heat": {"FLUXNET_PLUMBER2_varname": "Qh_cor"},
        },
        "sim_nml": {
            "general": {
                "Latent_Heat_sim_source": ["Site"],
                "Sensible_Heat_sim_source": ["Site"],
            },
            "Latent_Heat": {"Site_varname": "Latent_Heat"},
            "Sensible_Heat": {"Site_varname": "Sensible_Heat"},
        },
    }
    generator = ReportGenerator(config, str(case_dir))

    latent = generator._collect_figures("Latent_Heat")["station_timeseries"]
    sensible = generator._collect_figures("Sensible_Heat")["station_timeseries"]

    assert latent == [f"data/{site_dir.name}/{qle.name}"]
    assert sensible == [f"data/{site_dir.name}/{qh.name}"]
