import logging
from openbench.visualization._rc_isolation import with_isolated_rc  # noqa: E402
from ._sampling import sample_distribution_series
from openbench.visualization._figure_io import save_figure
from openbench.visualization._filenames import join_filename_components

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.artist import setp
from matplotlib.lines import Line2D


@with_isolated_rc
def make_scenarios_comparison_Whisker_Plot(
    basedir, evaluation_item, ref_source, sim_sources, varname, datasets_filtered, option
):
    option = option.copy()
    font = {"family": option["font"]}
    matplotlib.rc("font", **font)

    params = {
        "axes.linewidth": option["axes_linewidth"],
        "xtick.labelsize": option["xtick"],
        "xtick.direction": "in",
        "ytick.labelsize": option["ytick"],
        "ytick.direction": "in",
        "savefig.bbox": "tight",
        "axes.unicode_minus": False,
        "text.usetex": False,
    }
    rcParams.update(params)
    # Create the heatmap using Matplotlib

    facecolors = [
        "#ac8ab1",
        "#e3b855",
        "#7dacb0",
        "#a8501a",
        "#5ca44e",
        "#953658",
        "#ead693",
        "#9b5d7e",
        "#a2ad87",
        "#1e4f9e",
    ]
    fig, ax = plt.subplots(1, figsize=(option["x_wise"] * len(sim_sources) / 2, option["y_wise"]))

    for spine in ax.spines.values():
        spine.set_linewidth(option["line_width"])

    def colors(color):
        hex_pattern = r"^#([0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})$"
        import re

        if bool(re.match(hex_pattern, f"#{color}")):
            return f"#{color}"
        else:
            return color

    for key, value in option.items():
        if "color" in key:
            option[key] = colors(value)

    datasets_filtered = sample_distribution_series(datasets_filtered, option)

    boxprops = dict(linewidth=option["boxpropslinewidth"])
    if option["patch_artist"]:
        boxprops = dict(linewidth=option["boxpropslinewidth"], edgecolor=option["boxpropsedgecolor"])
    if varname in ["KGE", "KGESS", "NSE"]:
        for i, data in enumerate(datasets_filtered):
            lower_bound = np.min(data)
            if lower_bound < -1:
                datasets_filtered[i] = np.where(data < -1, -1, data).tolist()

    # Create the whisker plot
    bp = ax.boxplot(
        datasets_filtered,
        labels=[f"{i}" for i in sim_sources],
        vert=option["vert"],
        showfliers=option["showfliers"],
        flierprops=dict(
            marker=option["flierpropsmarker"],
            markerfacecolor=option["flierpropsmarkerfacecolor"],
            markersize=option["flierpropsmarkersize"],
            markeredgecolor=option["flierpropsmarkeredgecolor"],
            markeredgewidth=option["flierpropsmarkeredgewidth"],
        ),
        widths=option["box_widths"],
        showmeans=option["box_showmeans"],
        meanline=option["meanline"],
        meanprops=dict(
            linestyle=option["meanpropslinestyle"],
            linewidth=option["meanpropslinewidth"],
            color=option["meanpropscolor"],
        ),
        medianprops=dict(
            linestyle=option["medianpropslinestyle"],
            linewidth=option["medianpropslinewidth"],
            color=option["medianpropscolor"],
        ),
        patch_artist=option["patch_artist"],
        boxprops=boxprops,
        whiskerprops=dict(
            linestyle=option["whiskerpropslinestyle"],
            linewidth=option["whiskerpropslinewidth"],
            color=option["whiskerpropscolor"],
        ),
        capprops=dict(
            linestyle=option["cappropslinestyle"], linewidth=option["cappropslinewidth"], color=option["cappropscolor"]
        ),
    )

    for box, color in zip(bp["boxes"], facecolors[: len(sim_sources)]):
        if hasattr(box, "set_facecolor"):
            box.set_facecolor(color)

    # Create the whisker plot
    def remove_outliers(data_list):
        q1, q3 = np.percentile(data_list, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 2.5 * iqr
        upper_bound = q3 + 2.5 * iqr
        return [lower_bound, upper_bound]

    try:
        bound = [remove_outliers(d) for d in datasets_filtered]
        max_value = max([d[1] for d in bound])
        min_value = min([d[0] for d in bound])

        if varname in [
            "RMSE",
            "CRMSD",
            "MSE",
            "ubRMSE",
            "nRMSE",
            "mean_absolute_error",
            "ssq",
            "ve",
            "absolute_percent_bias",
        ]:
            min_value = 0
        elif varname in ["NSE", "LNSE", "ubNSE", "rNSE", "wNSE", "wsNSE"]:
            max_value = 1
        elif "Score" in varname:
            max_value = 1
            min_value = 0
    except Exception:
        min_value, max_value = None, None
        logging.warning(
            f"Could not estimate whisker axis bounds for {evaluation_item} {ref_source} {varname}; using Matplotlib autoscaling.",
            exc_info=True,
        )

    if option["vert"]:
        setp(ax.get_xticklabels(), rotation=option["x_rotation"], ha=option["ha"])

        ax.xaxis.set_ticks_position("both")
        ax.yaxis.set_ticks_position("both")
        ax.tick_params(axis="x", color="k", width=1.5, length=4, which="major")
        ax.tick_params(axis="y", color="k", width=1.5, length=4, which="major")

        # Add labels and title
        ax.set_xlabel(option["xticklabel"], fontsize=option["xtick"] + 1)
        ylabel = option["yticklabel"]
        if not option["yticklabel"] or len(option["yticklabel"]) == 0:
            ylabel = f"{varname}"
        ax.set_ylabel(ylabel.replace("_", " "), fontsize=option["ytick"] + 1, weight="bold")

        if option["grid"]:
            ax.yaxis.grid(True, linestyle=option["grid_style"], alpha=0.7, linewidth=option["grid_linewidth"])

        if option["limit_on"]:
            if option["value_min"] is None or option["value_max"] is None:
                raise ValueError(
                    "limit_on=True requires both value_min and value_max; got "
                    f"value_min={option['value_min']!r}, value_max={option['value_max']!r}"
                )
            if option["value_min"] > option["value_max"]:
                raise ValueError("Invalid limit settings: value_min must be less than or equal to value_max")
            else:
                ax.set(ylim=(option["value_min"], option["value_max"]))
                # ax.set(ylim=(dmin, dmax))
        else:
            ax.set(ylim=(min_value, max_value))
    else:
        setp(ax.get_yticklabels(), rotation=option["y_rotation"], ha=option["ha"])

        xlabel = option["xticklabel"]
        if not option["xticklabel"] or len(option["xticklabel"]) == 0:
            xlabel = f"{varname}"
        ax.set_xlabel(xlabel, fontsize=option["xtick"] + 1)
        ax.set_ylabel(option["yticklabel"], fontsize=option["ytick"] + 1)

        if option["grid"]:
            ax.xaxis.grid(True, linestyle=option["grid_style"], alpha=0.7, linewidth=option["grid_linewidth"])

        if option["limit_on"]:
            if option["value_min"] is None or option["value_max"] is None:
                raise ValueError(
                    "limit_on=True requires both value_min and value_max; got "
                    f"value_min={option['value_min']!r}, value_max={option['value_max']!r}"
                )
            if option["value_min"] > option["value_max"]:
                raise ValueError("Invalid limit settings: value_min must be less than or equal to value_max")
            else:
                ax.set(xlim=(option["value_min"], option["value_max"]))
                # ax.set(xlim=(dmin, dmax))
        else:
            ax.set(xlim=(min_value, max_value))

    title = option["title"]
    if not option["title"] or len(option["title"]) == 0:
        title = f"{evaluation_item.replace('_', ' ')}"
    ax.set_title(title, fontsize=option["title_fontsize"], weight="bold")

    legend_elements = [
        Line2D([0], [0], linestyle="-", color=option["medianpropscolor"], label="Median"),
        Line2D([0], [0], linestyle="--", color=option["meanpropscolor"], label="Mean"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="best",
        frameon=False,
        handlelength=1.75,
        handleheight=1,
        handletextpad=0.5,
        labelspacing=0.6,
        prop={"size": 12},
    )

    output_file_path = f"{basedir}/{join_filename_components('Whisker_Plot', evaluation_item, ref_source, varname)}.{option['saving_format']}"
    save_figure(fig, output_file_path, format=f"{option['saving_format']}", dpi=option["dpi"], bbox_inches="tight")
    plt.close(fig)  # Close the figure to free up memory

    return
