"""Station maps retain the statistical range, including constant values."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from openbench.visualization import Fig_stn_plot_index as plotting


@pytest.mark.parametrize(
    "method,column,limits",
    [
        ("Mann_Kendall_Trend_Test", "sim_tau", (-1.0, 1.0)),
        ("Mann_Kendall_Trend_Test", "ref_trend", (-1.0, 1.0)),
        ("Correlation", "Correlation", (-1.0, 1.0)),
        ("Functional_Response", "functional_response_score", (0.0, 1.0)),
        ("Standard_Deviation", "ref_value", (0.0, 1.0)),
    ],
)
@pytest.mark.parametrize("manual", [False, True])
def test_station_statistic_uses_natural_or_explicit_limits(tmp_path, monkeypatch, method, column, limits, manual):
    path = tmp_path / "stations.csv"
    pd.DataFrame({"ID": ["A", "B"], "ref_lon": [10.0, 20.0], "ref_lat": [10.0, 20.0], column: [1.0, np.nan]}).to_csv(
        path, index=False
    )
    config = Path(plotting.__file__).parents[1] / "data/fignml" / f"{method}.yaml"
    option = yaml.safe_load(config.read_text())["general"]
    option.update(vmin_max_on=manual, vmin=-2.0, vmax=2.0)
    figures = []
    monkeypatch.setattr(plotting, "save_figure", lambda fig, *a, **kw: figures.append(fig))
    plotting.make_stn_plot_index(str(path), method, option, ("A / B",), option, value_columns=(column,))
    norm = figures[0].axes[0].collections[0].norm
    assert (norm.vmin, norm.vmax) == ((-2.0, 2.0) if manual else limits)
    assert len(figures[0].axes[0].collections[1].get_offsets()) == 1
