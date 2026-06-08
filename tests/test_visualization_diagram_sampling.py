import numpy as np


def test_limit_diagram_points_preserves_taylor_reference_and_samples_simulations():
    from openbench.visualization._diagram_sampling import limit_diagram_points

    series, labels = limit_diagram_points(
        [np.arange(7), np.arange(10, 17)],
        ["s0", "s1", "s2", "s3", "s4", "s5"],
        {"max_diagram_points": 3},
        has_reference=True,
    )

    assert [item.tolist() for item in series] == [[0, 1, 3, 5], [10, 11, 13, 15]]
    assert labels == ["s0", "s2", "s4"]


def test_limit_diagram_points_full_mode_keeps_all_points():
    from openbench.visualization._diagram_sampling import limit_diagram_points

    series, labels = limit_diagram_points(
        [np.arange(10)],
        [f"s{i}" for i in range(10)],
        {"plotting_mode": "full", "max_diagram_points": 2},
    )

    assert series[0].tolist() == list(range(10))
    assert labels == [f"s{i}" for i in range(10)]


def test_taylor_wrapper_limits_plotted_simulation_points(tmp_path, monkeypatch):
    import openbench.visualization.Fig_taylor_diagram as fig_taylor
    from openbench.config.adapter import build_fig_nml

    captured = {}

    def fake_taylor(*args, **kwargs):
        stds, rms, cors = args[-3:]
        captured["stds"] = np.asarray(stds)
        captured["rms"] = np.asarray(rms)
        captured["cors"] = np.asarray(cors)
        captured["markers"] = kwargs["markers"]

    monkeypatch.setattr(fig_taylor, "taylor_diagram", fake_taylor)
    monkeypatch.setattr(fig_taylor, "save_figure", lambda *args, **kwargs: None)

    option = build_fig_nml()["Comparison"]["Taylor_Diagram"]
    option["max_diagram_points"] = 3
    fig_taylor.make_scenarios_comparison_Taylor_Diagram(
        str(tmp_path),
        "Runoff",
        np.arange(1, 8, dtype=float),
        np.arange(10, 17, dtype=float),
        np.linspace(0.1, 0.7, 7),
        "RefA",
        [f"s{i}" for i in range(6)],
        option,
    )

    assert captured["stds"].tolist() == [1.0, 2.0, 4.0, 6.0]
    assert captured["rms"].tolist() == [10.0, 11.0, 13.0, 15.0]
    assert captured["cors"].tolist() == [0.1, 0.2, 0.4, 0.6]
    assert list(captured["markers"]) == ["s0", "s2", "s4"]


def test_taylor_wrapper_normalizes_crmsd_with_standard_deviation(tmp_path, monkeypatch):
    import openbench.visualization.Fig_taylor_diagram as fig_taylor
    from openbench.config.adapter import build_fig_nml

    captured = {}

    def fake_taylor(*args, **kwargs):
        stds, rms, cors = args[-3:]
        captured["stds"] = np.asarray(stds)
        captured["rms"] = np.asarray(rms)
        captured["cors"] = np.asarray(cors)

    monkeypatch.setattr(fig_taylor, "taylor_diagram", fake_taylor)
    monkeypatch.setattr(fig_taylor, "save_figure", lambda *args, **kwargs: None)

    option = build_fig_nml()["Comparison"]["Taylor_Diagram"]
    option["Normalized"] = True
    ref_std = 2.0
    sim_std = 4.0
    corr = 0.5
    crmsd = np.sqrt(sim_std**2 + ref_std**2 - 2.0 * ref_std * sim_std * corr)

    fig_taylor.make_scenarios_comparison_Taylor_Diagram(
        str(tmp_path),
        "Runoff",
        np.array([ref_std, sim_std]),
        np.array([0.0, crmsd]),
        np.array([1.0, corr]),
        "RefA",
        ["sim"],
        option,
    )

    np.testing.assert_allclose(captured["stds"], [1.0, 2.0])
    np.testing.assert_allclose(captured["rms"], [0.0, crmsd / ref_std])
    np.testing.assert_allclose(
        captured["rms"][1] ** 2,
        captured["stds"][1] ** 2
        + captured["stds"][0] ** 2
        - 2.0 * captured["stds"][0] * captured["stds"][1] * captured["cors"][1],
    )


def test_target_wrapper_limits_plotted_points(tmp_path, monkeypatch):
    import openbench.visualization.Fig_target_diagram as fig_target
    from openbench.config.adapter import build_fig_nml

    captured = {}

    def fake_target(*args, **kwargs):
        bias, crmsd, rmsd = args[-3:]
        captured["bias"] = np.asarray(bias)
        captured["crmsd"] = np.asarray(crmsd)
        captured["rmsd"] = np.asarray(rmsd)
        captured["labels"] = kwargs["markerLabel"]
        captured["markers"] = kwargs["markers"]

    monkeypatch.setattr(fig_target, "target_diagram", fake_target)
    monkeypatch.setattr(fig_target, "save_figure", lambda *args, **kwargs: None)

    option = build_fig_nml()["Comparison"]["Target_Diagram"]
    option["max_diagram_points"] = 3
    fig_target.make_scenarios_comparison_Target_Diagram(
        str(tmp_path),
        "Runoff",
        np.arange(6, dtype=float),
        np.arange(10, 16, dtype=float),
        np.arange(20, 26, dtype=float),
        "RefA",
        [f"s{i}" for i in range(6)],
        option,
    )

    assert captured["bias"].tolist() == [0.0, 2.0, 4.0]
    assert captured["crmsd"].tolist() == [10.0, 12.0, 14.0]
    assert captured["rmsd"].tolist() == [20.0, 22.0, 24.0]
    assert captured["labels"] == ["s0", "s2", "s4"]
    assert list(captured["markers"]) == ["s0", "s2", "s4"]


def test_default_taylor_target_options_expose_diagram_limits():
    from openbench.config.adapter import build_fig_nml

    fig_nml = build_fig_nml()

    for section_name in ("Taylor_Diagram", "Target_Diagram"):
        section = fig_nml["Comparison"][section_name]
        assert section["plotting_mode"] == "balanced"
        assert section["max_diagram_points"] == 200
        assert section["diagram_sample_method"] == "stride"
