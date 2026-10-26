"""Station comparisons must use the same read-only datum correction as evaluation."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openbench.core._comparison_helpers import _load_station_pair
from tests.test_station_vertical_datum_runtime import _enable, _evaluation


def test_shared_pair_matches_evaluation_without_accumulation(tmp_path, monkeypatch):
    ev, rows, directory = _evaluation(tmp_path, monkeypatch)
    _enable(ev, tmp_path)
    before = {p: p.read_bytes() for p in directory.glob("*.nc")}
    bias = ev.make_evaluation_parallel(rows, 0)["bias"]
    for _ in range(2):
        s, o = _load_station_pair(ev, tmp_path, ev.item, "Ref", "Sim", rows.iloc[0], "height", "height")
        assert float((s - o).mean()) == pytest.approx(bias)
        np.testing.assert_allclose(o, [5, 6, 7])
    assert all(p.read_bytes() == content for p, content in before.items())


def _namelists(ev):
    item = ev.item
    sim = {"general": {f"{item}_sim_source": "Sim"}, item: {"Sim_data_type": "grid", "Sim_varname": "height"}}
    ref = {
        "general": {f"{item}_ref_source": "Ref"},
        item: {
            "Ref_data_type": "stn",
            "Ref_varname": "height",
            "Ref_vertical_datum": json.dumps(ev.ref_vertical_datum),
        },
    }
    return sim, ref


def test_target_and_mean_agree_with_metrics(tmp_path, monkeypatch):
    import openbench.core.comparison as comparison

    ev, rows, directory = _evaluation(tmp_path, monkeypatch)
    _enable(ev, tmp_path)
    ev.make_evaluation_P()
    sim, ref = _namelists(ev)
    monkeypatch.setattr(comparison, "make_stn_plot_index", lambda *a: None)
    monkeypatch.setattr(comparison, "make_scenarios_comparison_Target_Diagram", lambda *a: None)
    monkeypatch.setattr(
        comparison.ComparisonProcessing,
        "_run_parallel_or_serial",
        lambda self, tasks: [fn(*args, **kw) for fn, args, kw in tasks],
    )
    main = {"general": {"basedir": str(tmp_path), "basename": "", "compare_grid_res": 1, "compare_tim_res": "Day"}}
    handler = comparison.ComparisonProcessing(main, [], ["bias"])
    before = {p: p.read_bytes() for p in directory.glob("*.nc")}
    for _ in range(2):
        handler.scenarios_Basic_comparison(str(tmp_path), sim, ref, [ev.item], [], ["bias"], {"key": "Mean"})
        handler.scenarios_Target_Diagram_comparison(str(tmp_path), sim, ref, [ev.item], [], ["bias"], {})
        mean = pd.read_csv(tmp_path / f"comparisons/Mean/{ev.item}_stn_Ref_Sim_Mean.csv")
        target = pd.read_csv(tmp_path / f"comparisons/Target_Diagram/target_diagram_{ev.item}_stn_Ref_Sim.csv")
        assert mean.ref_value.iloc[0] == 6
        assert mean.sim_value.iloc[0] == 2
        assert target.bias.iloc[0] == mean.sim_value.iloc[0] - mean.ref_value.iloc[0] == -4
    assert all(p.read_bytes() == content for p, content in before.items())


def test_source_selection_relative_paths_and_reference_coordinates(tmp_path, monkeypatch):
    ev, rows, _ = _evaluation(tmp_path, monkeypatch)
    _enable(ev, tmp_path)
    sim, ref = _namelists(ev)
    ref[ev.item]["Ref_vertical_datum"] = json.dumps({"source_geoid": "08.pgm", "target_geoid": "96.pgm"})
    ref[ev.item]["Ref_dir"] = str(tmp_path)
    ref[ev.item]["Other_vertical_datum"] = {"source_geoid": "missing", "target_geoid": "missing"}
    seen = []

    def sample(path, model, lat, lon):
        seen.append((path, lat, lon))
        return 10 if model == "EGM2008" else 6

    monkeypatch.setattr("openbench.data.vertical_datum._sample_geoid", sample)
    row = rows.iloc[0].copy()
    row["sim_lat"], row["sim_lon"] = 45, 90
    handler = SimpleNamespace(compare_tim_res="Day", ref_vertical_datum={"stale": True})
    s, o = _load_station_pair(
        handler, tmp_path, ev.item, "Ref", "Sim", row, "height", "height", ref_nml=ref, sim_nml=sim
    )
    assert float((s - o).mean()) == -4
    assert seen == [(str(tmp_path / "08.pgm"), 0, 0), (str(tmp_path / "96.pgm"), 0, 0)]
    del ref[ev.item]["Ref_vertical_datum"]
    s, o = _load_station_pair(
        handler, tmp_path, ev.item, "Ref", "Sim", row, "height", "height", ref_nml=ref, sim_nml=sim
    )
    assert float((s - o).mean()) == 0


def test_all_only_drawing_comparisons_reject_datum_before_touching_outputs(tmp_path, monkeypatch):
    from openbench.data.vertical_datum import VerticalDatumError
    from openbench.visualization.only_drawing import ComparisonProcessing_only_drawing

    ev, _, _ = _evaluation(tmp_path, monkeypatch)
    _enable(ev, tmp_path)
    sim, ref = _namelists(ev)
    ref["general"]["Runoff_ref_source"] = "Plain"
    ref["Runoff"] = {}
    stale = tmp_path / "comparisons" / "old.csv"
    stale.parent.mkdir()
    stale.write_text("unchanged")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    draw = ComparisonProcessing_only_drawing.__new__(ComparisonProcessing_only_drawing)
    methods = [name for name in dir(draw) if name.startswith("scenarios_") and name.endswith("_comparison")]
    assert len(methods) == 19
    for name in methods:
        with pytest.raises(VerticalDatumError, match="only_drawing=False"):
            getattr(draw, name)(str(tmp_path), sim, ref, ["Runoff", ev.item], [], ["bias"], {})
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_only_drawing_guard_ignores_unselected_sources_and_items():
    from openbench.core._comparison_helpers import _require_uncorrected_comparison_cache

    item = "Water_Surface_Elevation"
    ref = {"general": {f"{item}_ref_source": "Plain"}, item: {"Other_vertical_datum": {"source": "EGM2008"}}}
    _require_uncorrected_comparison_cache(ref, [item])
    ref["general"][f"{item}_ref_source"] = "Other"
    _require_uncorrected_comparison_cache(ref, ["Runoff"])


@pytest.mark.parametrize(
    "comparison_enabled,generate_report,selected,comparison_only",
    [
        (True, False, True, False),
        (False, True, True, False),
        (False, False, True, False),
        (True, True, False, False),
        (False, True, False, False),
        (False, False, True, True),
        (True, True, True, True),
        (True, True, False, True),
    ],
)
def test_runner_rejects_before_evaluation_or_reporting(
    tmp_path, monkeypatch, comparison_enabled, generate_report, selected, comparison_only
):
    import openbench.config.adapter as adapter
    import openbench.runner.local as local
    from openbench.data.vertical_datum import VerticalDatumError
    from openbench.runner.orchestration import run_evaluation_impl

    item = "Water_Surface_Elevation"
    ref = {
        "general": {f"{item}_ref_source": "Ref" if selected else "Plain"},
        item: {"Ref_vertical_datum": {"source_geoid": "08.pgm", "target_geoid": "96.pgm"}},
    }
    runner = SimpleNamespace(
        general={"only_drawing": not comparison_only},
        basedir=str(tmp_path),
        basename="case",
        evaluation_items={"Runoff": True, item: True},
        metrics=["bias"],
        scores=[],
        comparisons=["HeatMap"],
        statistics=[],
    )
    bindings = SimpleNamespace(runner_cfg=runner, namelists=SimpleNamespace(reference=ref))
    cfg = SimpleNamespace(
        project=SimpleNamespace(num_cores=1, force=False, generate_report=generate_report),
        comparison=SimpleNamespace(enabled=comparison_enabled),
        simulation={"Sim": {}},
    )
    monkeypatch.setattr(adapter, "build_runner_bindings", lambda cfg: bindings)

    def task_boundary(**kwargs):
        raise RuntimeError("task construction reached")

    monkeypatch.setattr(local, "_build_evaluation_tasks", task_boundary)
    for name in ("_evaluate_ready_tasks", "_run_comparison", "_run_report"):
        monkeypatch.setattr(local, name, lambda *a, **kw: pytest.fail("must fail before evaluation/post phases"))
    cache = tmp_path / "case/comparisons/stale.csv"
    cache.parent.mkdir(parents=True)
    cache.write_text("unchanged")
    report = tmp_path / "case/reports/old.html"
    report.parent.mkdir()
    report.write_text("unchanged report")
    if (comparison_only or comparison_enabled or generate_report) and selected:
        with pytest.raises(VerticalDatumError, match="only_drawing=False"):
            run_evaluation_impl(cfg, comparison_only=comparison_only)
    else:
        with pytest.raises(RuntimeError, match="task construction reached"):
            run_evaluation_impl(cfg, comparison_only=comparison_only)
    assert cache.read_text() == "unchanged"
    assert report.read_text() == "unchanged report"
