import logging
import os
from openbench.visualization._rc_isolation import with_isolated_rc  # noqa: E402
from openbench.visualization._figure_io import save_figure

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch

from openbench.util.converttype import Convert_Type

from .Fig_toolbox import get_index
from ._validation import finite_min_max


def _read_comparison_file(file):
    """
    Read comparison file with fallback logic.
    Try .csv first, then .txt if not found.
    Auto-detect separator (tab or comma).
    """
    file_to_read = file

    if not os.path.exists(file):
        if file.endswith(".csv"):
            alt_file = file[:-4] + ".txt"
            if os.path.exists(alt_file):
                logging.info(f"File {file} not found, using {alt_file}")
                file_to_read = alt_file
        elif file.endswith(".txt"):
            alt_file = file[:-4] + ".csv"
            if os.path.exists(alt_file):
                logging.info(f"File {file} not found, using {alt_file}")
                file_to_read = alt_file

    if not os.path.exists(file_to_read):
        raise FileNotFoundError(f"Neither {file} nor alternative extension found")

    with open(file_to_read, "r") as f:
        first_line = f.readline()

    sep = "\t" if "\t" in first_line else ","
    df = pd.read_csv(file_to_read, sep=sep, header=0)
    # Remove index name to prevent it from appearing in visualizations
    df.index.name = None
    return df


@with_isolated_rc
def make_scenarios_comparison_radar_map(file, score, option):
    # Read file with fallback and auto-detection
    df = _read_comparison_file(file)
    df = Convert_Type.convert_Frame(df)
    list1 = df.iloc[:, 0].tolist()
    list2 = df.iloc[:, 1].tolist()
    ref_dataname = [f"{item1.replace('_', ' ')}\n({item2})" for item1, item2 in zip(list1, list2)]
    df.set_index("Item", inplace=True)
    df = df.iloc[:, 1:].astype("float32")
    min_value, max_value = finite_min_max(df.values, label=f"{score} radar map")
    # fillna(0) keeps the polygon plottable, but a real "0 score" and an
    # "evaluation-failed NaN" then look identical on the radar — warn so the
    # user knows the filled cells are not valid performance, just placeholders.
    nan_count = int(df.isna().sum().sum())
    if nan_count > 0:
        logging.warning(
            "Fig_radarmap (%s): %d NaN cell(s) in the score matrix filled with 0; "
            "the corresponding sim/variable did not produce a valid score and the "
            "radar polygon should NOT be read as 'performance = 0'.",
            score,
            nan_count,
        )
    df = df.fillna(0)

    cmap, mticks, norm, bnd, extend = get_index(min_value, max_value, option.get("cmap", "Spectral"), score)
    if mticks[-1] > 0.5:
        mticks_interval = 0.2
    else:
        mticks_interval = mticks[1] - mticks[0]
    mticks = np.arange(0, max_value + mticks_interval, mticks_interval).round(1)

    if not option["vmin_max_on"]:
        option["vmax"], option["vmin"] = mticks[-1], mticks[0]

    params = {
        "figure.figsize": [option["x_wise"], option["y_wise"]],
        "figure.dpi": option["dpi"],
        "figure.autolayout": True,
        "savefig.format": option["saving_format"],
        "font.family": option["font_family"],
        "font.size": option["font_size"],
        "axes.edgecolor": "white",
        "axes.titlesize": option["titlesize"],
        "axes.titleweight": "bold",
        "axes.titlepad": 10.0,
        "axes.linewidth": option["linewidth"],
        "patch.linewidth": option["patch_linewidth"],
        "lines.linestyle": option["lines_linestyle"],
        "xtick.labelsize": option["xtick_labelsize"],
        "xtick.major.pad": 5,
        "legend.handlelength": 2.0,
        "legend.handleheight": 2.0,
        "legend.fontsize": option["legend_fontsize"],
        "legend.frameon": True,
    }
    rcParams.update(params)

    series_count = len(df.columns)
    color_positions = np.linspace(0.1, 0.9, max(series_count, 1))
    series_colors = [cmap(position) for position in color_positions]

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    angles = [n / float(len(ref_dataname)) * 2 * np.pi for n in range(len(ref_dataname))]
    angles += angles[:1]

    ax.set_title(score.replace("_", " "))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(ref_dataname)
    ax.set_rlabel_position(0)
    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            label.set_horizontalalignment("center")
        elif 0 < angle < np.pi:
            label.set_horizontalalignment("left")
        else:
            label.set_horizontalalignment("right")
    ax.set_yticks([])
    ax.set_yticklabels("")
    ax.set_ylim(mticks[0], mticks[-1])
    for i in range(len(mticks)):
        grid_values = [mticks[i]] * len(angles)
        ax.plot(angles, grid_values, color="grey")
        if i != len(mticks) - 1:
            if mticks[i] - 0.02 > 0:
                ax.text(np.deg2rad(0), mticks[i] - 0.02, str(mticks[i]), ha="center", va="center")
            else:
                ax.text(np.deg2rad(0), mticks[i], str(mticks[i]), ha="center", va="center")

    for i in range(len(df.columns)):
        values = df.iloc[:, i].tolist()
        values += values[:1]
        ax.plot(angles, values, color=series_colors[i], linestyle="-")
        ax.fill(angles, values, color=series_colors[i], alpha=0.8)

    legend_elements = [
        Patch(facecolor=series_colors[i], edgecolor="black", label=df.columns[i]) for i in range(len(df.columns))
    ]
    title_font = FontProperties(weight="bold")
    ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(1, 0.1),
        title=option["legend_title"],
        title_fontproperties=title_font,
    )
    file2 = os.path.splitext(file)[0]
    save_figure(fig, f"{file2}_radarmap.{option['saving_format']}")
    plt.close(fig)
