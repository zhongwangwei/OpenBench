from pathlib import Path

import pytest

from openbench.config.adapter import build_fig_nml
from openbench.visualization._geo_axes import add_rivers


class _Axis:
    def __init__(self):
        self.calls = []

    def add_feature(self, feature, **kwargs):
        self.calls.append((feature, kwargs))
        return "artist"


def test_map_defaults_propagate_and_station_yaml_overrides():
    figures = build_fig_nml()

    assert figures["map_features"]["river_resolution"] == "10m"
    assert figures["make_geo_plot_index"]["river_resolution"] == "10m"
    assert figures["Comparison"]["Diff_Plot"]["river_resolution"] == "10m"
    assert figures["Statistic"]["Correlation"]["river_resolution"] == "10m"
    assert figures["make_stn_plot_index"]["river_color"] == "#4088b8"
    assert figures["make_stn_plot_index"]["river_linewidth"] == 0.5


def test_add_rivers_uses_configured_style():
    axis = _Axis()
    result = add_rivers(
        axis,
        {
            "river_resolution": "50m",
            "river_color": "navy",
            "river_linewidth": 0.7,
            "river_alpha": 0.4,
        },
    )

    assert result == "artist"
    feature, kwargs = axis.calls[0]
    assert feature.scale == "50m"
    assert feature.kwargs["edgecolor"] == "navy"
    assert kwargs == {"linewidth": 0.7, "alpha": 0.4}


def test_show_rivers_false_adds_nothing():
    axis = _Axis()
    assert add_rivers(axis, {"show_rivers": False}) is None
    assert axis.calls == []


@pytest.mark.parametrize(
    "option, message",
    [
        ({"show_rivers": "false"}, "show_rivers"),
        ({"river_resolution": "25m"}, "river_resolution"),
        ({"river_color": "not-a-color"}, "river_color"),
        ({"river_linewidth": -0.1}, "river_linewidth"),
        ({"river_alpha": 1.1}, "river_alpha"),
    ],
)
def test_invalid_river_options_fail_clearly(option, message):
    with pytest.raises(ValueError, match=message):
        add_rivers(_Axis(), option)


def test_river_feature_is_not_hardcoded_in_plot_modules():
    directory = Path(__file__).parents[1] / "src" / "openbench" / "visualization"
    offenders = []
    for path in directory.glob("Fig_*.py"):
        if "rivers_lake_centerlines" in path.read_text(encoding="utf-8"):
            offenders.append(path.name)
    assert offenders == []
