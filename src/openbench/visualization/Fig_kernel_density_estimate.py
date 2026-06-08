import hashlib
import logging
import math
from openbench.visualization._rc_isolation import with_isolated_rc  # noqa: E402
from ._sampling import sample_series_for_plot
from openbench.visualization._figure_io import save_figure
from openbench.visualization._filenames import join_filename_components
from openbench.visualization._validation import finite_values

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from scipy.stats import gaussian_kde

logger = logging.getLogger(__name__)


@with_isolated_rc
def make_scenarios_comparison_Kernel_Density_Estimate(
    basedir, evaluation_item, ref_source, sim_sources, varname, datasets_filtered, option
):
    if varname != "nSpatialScore":
        font = {"family": option["font"]}
        matplotlib.rc("font", **font)

        params = {
            "axes.linewidth": option["axes_linewidth"],
            "font.size": option["fontsize"],
            "xtick.labelsize": option["xtick"],
            "xtick.direction": "out",
            "ytick.labelsize": option["ytick"],
            "ytick.direction": "out",
            "savefig.bbox": "tight",
            "axes.unicode_minus": False,
            "text.usetex": False,
        }
        rcParams.update(params)

        # create a figure and axis
        fig = plt.figure(figsize=(option["x_wise"], option["y_wise"]))
        ax = fig.add_subplot(111)

        for spine in ax.spines.values():
            spine.set_linewidth(option["line_width"])

        # Generate colors using a colormap
        MLINES = generate_lines(sim_sources, option)
        # Create a list to store line objects for the legend
        lines = []
        # Plot each KDE
        for i, data in enumerate(datasets_filtered):
            sim_source = sim_sources[i]

            try:
                data = sample_series_for_plot(data, option, purpose="kde")
                data = finite_values(data, label=f"{evaluation_item}/{ref_source}/{sim_source}/{varname} KDE")
                lower_bound = np.min(data)
                filtered_data = data
                if varname in ["KGE", "KGESS", "NSE"]:
                    if lower_bound < -1:
                        filtered_data = np.where(data < -1, -1, data)
                elif varname in ["MFM"]:
                    filtered_data = np.where(data < 0, 0, data)
                kde = gaussian_kde(filtered_data)
                covariance_matrix = kde.covariance
                covariance_matrix += np.eye(covariance_matrix.shape[0]) * 1e-6  # Regularization
                kde.covariance = covariance_matrix

                # Set x-axis range for KGE and KGESS
                x_values = np.linspace(filtered_data.min(), filtered_data.max(), 100)

                density = kde(x_values)

                # Store the line object
                (line,) = ax.plot(
                    x_values,
                    density,
                    color=MLINES[sim_source]["lineColor"],
                    linestyle=MLINES[sim_source]["linestyle"],
                    linewidth=MLINES[sim_source]["linewidth"] * 2,
                    label=sim_source,
                )
                lines.append(line)  # Add the line object to the list
                ax.fill_between(
                    x_values, density, color=MLINES[sim_source]["lineColor"], alpha=MLINES[sim_source]["alpha"]
                )

            except Exception:
                logger.exception(
                    "%s %s %s %s Kernel Density Estimate failed",
                    evaluation_item,
                    ref_source,
                    sim_source,
                    varname,
                )
                raise

        # Add labels and legend
        if varname == "percent_bias":
            legend_title = "Percent Bias showing value between [-100,100]"
        else:
            legend_title = ""

        ncol = option["ncol"]
        if not option["set_legend"]:
            ncol = int(math.ceil(len(sim_sources) / 10))
            ax.legend(
                shadow=False,
                frameon=False,
                fontsize=option["fontsize"],
                title=legend_title,
                loc=option["loc"],
                ncol=ncol,
                handlelength=1,
                handleheight=1,
            )
        else:
            ax.legend(
                shadow=False,
                frameon=False,
                fontsize=option["fontsize"],
                title=legend_title,
                bbox_to_anchor=(option["bbox_to_anchor_x"], option["bbox_to_anchor_y"]),
                ncol=ncol,
                handlelength=1,
                handleheight=1,
            )

        if option["grid"]:
            ax.grid(linestyle=option["grid_style"], alpha=0.7, linewidth=option["grid_linewidth"])

        if not option["title"]:
            title = f"Kernel Density Estimate of {evaluation_item.replace('_', ' ')}"
        else:
            title = option["title"]

        if not option["xticklabel"]:
            xticklabel = f"{varname.replace('_', '')}"
        else:
            xticklabel = option["xticklabel"]

        if not option["yticklabel"]:
            yticklabel = "KDE Density"
        else:
            yticklabel = option["yticklabel"]

        ax.set_xlabel(xticklabel, fontsize=option["xtick"] + 1, weight="bold")
        ax.set_ylabel(yticklabel, fontsize=option["ytick"] + 1, weight="bold")
        ax.set_title(title, fontsize=option["title_fontsize"], weight="bold", loc="center")

        # Encode the sim_sources set into the filename so that two consecutive
        # runs over [A,B] and [A,B,C] don't overwrite each other's KDE PNG.
        # Use a short stable hash to keep filenames bounded; the full list is
        # logged below at INFO so users can map the token back if needed.
        if sim_sources:
            sim_key = "+".join(sorted(sim_sources))
            sim_token = hashlib.sha1(sim_key.encode("utf-8")).hexdigest()[:8]
            logger.info("Fig_kde: token %s = sim_sources [%s]", sim_token, sim_key)
            filename_parts = ("Kernel_Density_Estimate", evaluation_item, ref_source, varname, sim_token)
        else:
            filename_parts = ("Kernel_Density_Estimate", evaluation_item, ref_source, varname)
        output_file_path = f"{basedir}/{join_filename_components(*filename_parts)}.{option['saving_format']}"
        save_figure(fig, output_file_path, format=f"{option['saving_format']}", dpi=option["dpi"], bbox_inches="tight")
        # Close the figure so matplotlib's figure manager releases its
        # reference; otherwise figures accumulate across batch calls.
        plt.close(fig)


def generate_lines(data_names, option):
    import itertools

    import matplotlib.colors as mcolors

    lines = {}
    # add colors and symbols
    hex_colors = [
        "#468cc8",
        "#90278c",
        "#b48abc",
        "#f9931d",
        "#fcdf15",
        "#d02030",
        "#252162",
        "#a9dae0",
        "#e57f99",
        "#019d88",
        "#afb2b3",
    ]
    # hex_colors = cm.Set3(np.linspace(0, 1, len(data_names) + 1))
    colors = itertools.cycle([mcolors.rgb2hex(color) for color in hex_colors])

    if not option["linestyle"]:
        linestyle = "solid"
    else:
        linestyle = option["linestyle"]

    if not option["linewidth"]:
        linewidth = 1.5
    else:
        linewidth = option["linewidth"]

    if not option["alpha"]:
        alpha = 0.3
    else:
        alpha = option["alpha"]

    for name in data_names:
        color = next(colors)
        lines[name] = {"lineColor": color, "linestyle": linestyle, "linewidth": linewidth, "alpha": alpha}
    return lines
