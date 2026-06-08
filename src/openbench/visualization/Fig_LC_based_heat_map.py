import logging
import math
import os
from openbench.visualization._rc_isolation import with_isolated_rc  # noqa: E402
from openbench.visualization._figure_io import save_figure

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import rcParams

from openbench.util.converttype import Convert_Type
from openbench.util.filenames import groupby_class_netcdf_stem

logger = logging.getLogger(__name__)


def _annotation_color(value, *, high=0.8, low=0.2):
    """Return readable annotation color for diverging heat-map cells."""
    try:
        return "white" if float(value) > high or float(value) < low else "black"
    except (TypeError, ValueError):
        return "black"


def _add_custom_colorbar_axes(fig, option):
    return fig.add_axes(
        [
            option["colorbar_left"],
            option["colorbar_bottom"],
            option["colorbar_width"],
            option["colorbar_height"],
        ]
    )


def _groupby_class_netcdf_files(option, statistic):
    """Return safe per-class NetCDF files, with legacy name fallback."""
    import glob

    selected_item, sim_source, ref_source = option["item"][0], option["item"][1], option["item"][2]
    groupby_prefix = str(option.get("groupby", "")).split("_", maxsplit=1)[0]
    safe_stem = groupby_class_netcdf_stem(selected_item, ref_source, sim_source, statistic, groupby_prefix)
    safe_pattern = glob.escape(os.path.join(option["path"], safe_stem)) + "__*.nc"
    legacy_stem = os.path.join(
        option["path"], f"{selected_item}_ref_{ref_source}_sim_{sim_source}_{statistic}_{groupby_prefix}_"
    )
    legacy_pattern = glob.escape(legacy_stem) + "*.nc"
    files = glob.glob(safe_pattern)
    files.extend(path for path in glob.glob(legacy_pattern) if path not in files)
    return files


def _require_groupby_class_netcdf_files(option, statistic):
    """Return per-class NetCDF files or fail with a clear producer/consumer path error."""
    files = _groupby_class_netcdf_files(option, statistic)
    if not files:
        item, sim_source, ref_source = option["item"][0], option["item"][1], option["item"][2]
        groupby = option.get("groupby", "groupby")
        raise FileNotFoundError(
            f"{groupby} heatmap missing per-class NetCDF inputs for "
            f"item={item!r}, sim={sim_source!r}, ref={ref_source!r}, statistic={statistic!r} "
            f"under {option['path']!r}. Run the full groupby producer first or check safe/legacy filenames."
        )
    return files


def _open_groupby_class_distribution(option, statistic):
    """Open class NetCDF inputs as a single distribution axis for quantiles.

    New producers write one ``__classes.nc`` with a ``class`` dimension; older
    runs wrote one file per class.  Normalize both to the historical synthetic
    ``time`` axis consumed by this plotting code.
    """
    files = _require_groupby_class_netcdf_files(option, statistic)
    if len(files) == 1:
        with xr.open_dataset(files[0]) as dataset:
            ds = dataset.load()
        if "class" in ds.dims:
            return ds.rename({"class": "time"})
        logger.warning(
            "LC/CZ heatmap quantile clip for %s degenerates: only one legacy per-class NetCDF matched; "
            "colour scale will reflect a single sample, not a distribution.",
            statistic,
        )
        return ds.expand_dims(dim={"time": [0]})

    datasets = []
    for path in files:
        with xr.open_dataset(path) as dataset:
            datasets.append(dataset.load())
    for idx, ds in enumerate(datasets):
        datasets[idx] = ds.expand_dims(dim={"time": [idx]})
    return xr.concat(datasets, dim="time")


def _read_metrics_file(file):
    """
    Read metrics/scores file with fallback logic.
    Try .csv first, then .txt if not found.
    Auto-detect separator (tab or comma).
    """
    # Try the given file path first
    file_to_read = file

    # If file doesn't exist, try alternative extension
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

    # Read first line to detect separator
    with open(file_to_read, "r") as f:
        first_line = f.readline()

    # Auto-detect separator: if tabs present, use tab; otherwise use comma
    if "\t" in first_line:
        sep = "\t"
    else:
        sep = ","

    # Read the file manually to handle inconsistent column counts from trailing tabs
    with open(file_to_read, "r") as f:
        lines = f.readlines()

    # Clean lines by stripping trailing tabs/whitespace
    cleaned_lines = []
    for line in lines:
        # Remove trailing tabs and whitespace, then add newline back
        cleaned_line = line.rstrip("\t \n\r") + "\n"
        cleaned_lines.append(cleaned_line)

    # Write cleaned content to a temporary string buffer
    from io import StringIO

    cleaned_content = "".join(cleaned_lines)

    # Read the cleaned content
    df = pd.read_csv(StringIO(cleaned_content), sep=sep, header=0, index_col=0)

    # Check if first row is "FullName" (old format with ID + FullName rows) - if so, skip ID row
    if len(df.index) > 0 and df.index[0] == "FullName":
        df = pd.read_csv(StringIO(cleaned_content), sep=sep, skiprows=1, header=0, index_col=0)

    # Drop any unnamed columns that may result from trailing tabs
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Remove the index name as it shouldn't appear in visualizations
    df.index.name = None

    return df


@with_isolated_rc
def make_LC_based_heat_map(file, selected_metrics, lb, option):
    option = option.copy()
    selected_metrics = list(selected_metrics)
    # Convert the data to a DataFrame with fallback and auto-detection
    df = _read_metrics_file(file)

    # Convert string values to numeric, replacing 'N/A' with NaN
    df = df.apply(pd.to_numeric, errors="coerce")
    df = Convert_Type.convert_Frame(df)

    # Select the desired metrics
    # selected_metrics = ['nBiasScore', 'nRMSEScore', 'nPhaseScore', 'nIavScore', 'nSpatialScore', 'overall_score']
    df_selected = df.loc[selected_metrics]

    # Mapping from numeric column IDs to IGBP class names
    igbp_id_to_name = {
        "1": "evergreen_needleleaf_forest",
        "2": "evergreen_broadleaf_forest",
        "3": "deciduous_needleleaf_forest",
        "4": "deciduous_broadleaf_forest",
        "5": "mixed_forests",
        "6": "closed_shrubland",
        "7": "open_shrublands",
        "8": "woody_savannas",
        "9": "savannas",
        "10": "grasslands",
        "11": "permanent_wetlands",
        "12": "croplands",
        "13": "urban_and_built_up",
        "14": "cropland_natural_vegetation_mosaic",
        "15": "snow_and_ice",
        "16": "barren_or_sparsely_vegetated",
        "17": "water_bodies",
        "All": "Overall",
    }

    shorter = {
        "PFT_groupby": {
            "bare_soil": "BS",
            "needleleaf_evergreen_temperate_tree": "NETT",
            "needleleaf_evergreen_boreal_tree": "NEBT",
            "needleleaf_deciduous_boreal_tree": "NDBT",
            "broadleaf_evergreen_tropical_tree": "BETT",
            "broadleaf_evergreen_temperate_tree": "BETT",
            "broadleaf_deciduous_tropical_tree": "BDTT",
            "broadleaf_deciduous_temperate_tree": "BDTT",
            "broadleaf_deciduous_boreal_tree": "BDBT",
            "broadleaf_evergreen_shrub": "BES",
            "broadleaf_deciduous_temperate_shrub": "BDTS",
            "broadleaf_deciduous_boreal_shrub": "BDBS",
            "c3_arctic_grass": "C3AG",
            "c3_non-arctic_grass": "C3NAG",
            "c4_grass": "C4G",
            "c3_crop": "C3C",
            "Overall": "Overall",
        },
        "IGBP_groupby": {
            "evergreen_needleleaf_forest": "ENF",
            "evergreen_broadleaf_forest": "EBF",
            "deciduous_needleleaf_forest": "DNF",
            "deciduous_broadleaf_forest": "DBF",
            "mixed_forests": "MF",
            "closed_shrubland": "CSH",
            "open_shrublands": "OSH",
            "woody_savannas": "WSA",
            "savannas": "SAV",
            "grasslands": "GRA",
            "permanent_wetlands": "WET",
            "croplands": "CRO",
            "urban_and_built_up": "URB",
            "cropland_natural_vegetation_mosaic": "CVM",
            "snow_and_ice": "SNO",
            "barren_or_sparsely_vegetated": "BSV",
            "water_bodies": "WAT",
            "Overall": "Overall",
        },
    }

    def get_short_label(column, groupby):
        """Get short label for column, handling both numeric IDs and class names."""
        # First try direct lookup
        if column in shorter.get(groupby, {}):
            return shorter[groupby][column]
        # For IGBP, try mapping numeric ID to class name first
        if groupby == "IGBP_groupby" and str(column) in igbp_id_to_name:
            class_name = igbp_id_to_name[str(column)]
            return shorter[groupby].get(class_name, str(column))
        return str(column)

    font = {"family": "DejaVu Sans"}
    # font = {'family': option['font']}
    matplotlib.rc("font", **font)
    params = {
        "axes.linewidth": option["axes_linewidth"],
        "font.size": option["fontsize"],
        "xtick.labelsize": option["xtick"],
        "xtick.direction": "out",
        "ytick.labelsize": option["ytick"],
        "grid.linewidth": 1,
        "ytick.direction": "out",
        "savefig.bbox": "tight",
        "axes.unicode_minus": False,
        "text.usetex": False,
    }
    rcParams.update(params)

    if lb == "score":
        # Create the heatmap using Matplotlib
        fig, ax = plt.subplots(figsize=(option["x_wise"], option["y_wise"]))
        if option["vmin_max_on"]:
            vmin, vmax = option["vmin"], option["vmax"]
        else:
            vmin, vmax = 0, 1
        if not option["cmap"]:
            option["cmap"] = "coolwarm"
        im = ax.imshow(df_selected, cmap=option["cmap"], vmin=vmin, vmax=vmax)

        ax.set_yticks(range(len(df_selected.index)))
        ax.set_xticks(range(len(df_selected.columns)))
        ax.set_yticklabels(
            [index.replace("_", " ") for index in df_selected.index], rotation=option["y_rotation"], ha=option["y_ha"]
        )
        if option["x_ticklabel"] == "Normal":
            ax.set_xticklabels(
                [columns.replace("_", " ").title() for columns in df_selected.columns],
                rotation=option["x_rotation"],
                ha=option["x_ha"],
            )
        else:
            item = option["groupby"]
            ax.set_xticklabels(
                [get_short_label(column, item) for column in df_selected.columns],
                rotation=option["x_rotation"],
                ha=option["x_ha"],
            )

        ax.set_ylabel("Scores", fontsize=option["ytick"] + 1)
        ax.set_xlabel(option["xlabel"], fontsize=option["xtick"] + 1)

        if len(option["title"]) == 0:
            option["title"] = f"Heatmap of {lb}"
        ax.set_title(option["title"], fontsize=option["title_size"])

        for i in range(len(df_selected.index)):
            for j in range(len(df_selected.columns)):
                ax.text(
                    j,
                    i,
                    f"{df_selected.iloc[i, j]:{option['ticks_format']}}",
                    ha="center",
                    va="center",
                    color=_annotation_color(df_selected.iloc[i, j]),
                    fontsize=option["fontsize"],
                )

        pos = ax.get_position()  # .bounds
        left, right, bottom, width, height = pos.x0, pos.x1, pos.y0, pos.width, pos.height

        if not option["colorbar_position_set"]:
            if option["colorbar_position"] == "vertical":
                cbar_ax = fig.add_axes([right + 0.05, bottom, 0.03, height])  # right + 0.2
            else:
                xlabel = ax.xaxis.label
                xticks = ax.get_xticklabels()
                max_xtick_height = 0
                for xtick in xticks:
                    bbox = xtick.get_window_extent()  # 获取每个 xtick 的包围框
                    bbox_transformed = bbox.transformed(fig.transFigure.inverted())  # 将像素转换为图坐标
                    max_xtick_height = max(max_xtick_height, bbox_transformed.height)
                if xlabel is not None:
                    bbox = xlabel.get_window_extent()  # 获取每个 xtick 的包围框
                    bbox_transformed = bbox.transformed(fig.transFigure.inverted())  # 将像素转换为图坐标
                    x_height = bbox_transformed.height
                    cbar_ax = fig.add_axes(
                        [left + width / 6, bottom - max_xtick_height - x_height - 0.1, width / 3 * 2, 0.04]
                    )
                else:
                    cbar_ax = fig.add_axes([left + width / 6, bottom - max_xtick_height - 0.1, width / 3 * 2, 0.04])
        else:
            cbar_ax = _add_custom_colorbar_axes(fig, option)
        cbar = fig.colorbar(
            im,
            cax=cbar_ax,
            label=option["colorbar_label"],
            orientation=option["colorbar_position"],
            extend=option["extend"],
        )
    elif len(df_selected.index) == 1 and lb != "score":
        fig, ax = plt.subplots(figsize=(option["x_wise"], option["y_wise"]))

        metric = df_selected.index[0]
        logger.info(metric)
        combined_dataset = _open_groupby_class_distribution(option, metric)
        quantiles = combined_dataset.quantile([0.05, 0.2, 0.8, 0.95], dim=["time", "lat", "lon"])
        # consider 0.05 and 0.95 value as the max/min value
        custom_vmin_vmax = {}
        if not option["vmin_max_on"]:
            if metric in ["bias", "percent_bias", "rSD", "PBIAS_HF", "PBIAS_LF"]:
                custom_vmin_vmax[metric] = [
                    quantiles[metric][0].values,
                    quantiles[metric][-1].values,
                    quantiles[metric][2].values,
                    quantiles[metric][1].values,
                ]
            elif metric in ["NSE", "KGE", "KGESS", "correlation", "kappa_coeff", "rSpearman"]:
                custom_vmin_vmax[metric] = [-1, 1, 0.8, -0.8]
            elif metric in ["LNSE", "ubNSE", "rNSE", "wNSE", "wsNSE"]:
                custom_vmin_vmax[metric] = [quantiles[metric][0].values, 1, 0.8, quantiles[metric][1].values]
            elif metric in [
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
                custom_vmin_vmax[metric] = [-1, quantiles[metric][-1].values, quantiles[metric][2].values, -0.8]
            else:
                custom_vmin_vmax[metric] = [0, 1, 0.8, 0.2]
        else:
            custom_vmin_vmax[metric] = [option["vmin"], option["vmax"], 0.8, 0.2]

        if not option["cmap"]:
            option["cmap"] = "coolwarm"

        vmin, vmax = custom_vmin_vmax[metric][0], custom_vmin_vmax[metric][1]
        x1, x2 = custom_vmin_vmax[metric][2], custom_vmin_vmax[metric][3]
        im = ax.imshow(df_selected, cmap=option["cmap"], vmin=vmin, vmax=vmax)

        ax.set_yticks(range(len(df_selected.index)))
        ax.set_xticks(range(len(df_selected.columns)))
        ax.set_yticklabels(
            [index.replace("_", " ") for index in df_selected.index], rotation=option["y_rotation"], ha=option["y_ha"]
        )
        if option["x_ticklabel"] == "Normal":
            ax.set_xticklabels(
                [columns.replace("_", " ").title() for columns in df_selected.columns],
                rotation=option["x_rotation"],
                ha=option["x_ha"],
            )
        else:
            item = option["groupby"]
            ax.set_xticklabels(
                [get_short_label(column, item) for column in df_selected.columns],
                rotation=option["x_rotation"],
                ha=option["x_ha"],
            )

        ax.set_ylabel("Metrics", fontsize=option["ytick"] + 1)
        ax.set_xlabel(option["xlabel"], fontsize=option["xtick"] + 1)

        if len(option["title"]) == 0:
            option["title"] = f"Heatmap of {lb}"
        ax.set_title(option["title"], fontsize=option["title_size"])

        for i in range(len(df_selected.index)):
            for j in range(len(df_selected.columns)):
                ax.text(
                    j,
                    i,
                    f"{df_selected.iloc[i, j]:{option['ticks_format']}}",
                    ha="center",
                    va="center",
                    color=_annotation_color(df_selected.iloc[i, j], high=x1, low=x2),
                    fontsize=option["fontsize"],
                )

        pos = ax.get_position()  # .bounds
        left, right, bottom, width, height = pos.x0, pos.x1, pos.y0, pos.width, pos.height

        if not option["colorbar_position_set"]:
            if option["colorbar_position"] == "vertical":
                cbar_ax = fig.add_axes([right + 0.05, bottom, 0.03, height])  # right + 0.2
            else:
                xlabel = ax.xaxis.label
                xticks = ax.get_xticklabels()
                max_xtick_height = 0
                for xtick in xticks:
                    bbox = xtick.get_window_extent()  # 获取每个 xtick 的包围框
                    bbox_transformed = bbox.transformed(fig.transFigure.inverted())  # 将像素转换为图坐标
                    max_xtick_height = max(max_xtick_height, bbox_transformed.height)
                if xlabel is not None:
                    bbox = xlabel.get_window_extent()  # 获取每个 xtick 的包围框
                    bbox_transformed = bbox.transformed(fig.transFigure.inverted())  # 将像素转换为图坐标
                    x_height = bbox_transformed.height
                    cbar_ax = fig.add_axes(
                        [left + width / 6, bottom - max_xtick_height - x_height - 0.1, width / 3 * 2, 0.04]
                    )
                else:
                    cbar_ax = fig.add_axes([left + width / 6, bottom - max_xtick_height - 0.1, width / 3 * 2, 0.04])
        else:
            cbar_ax = _add_custom_colorbar_axes(fig, option)
        cbar = fig.colorbar(
            im,
            cax=cbar_ax,
            label=option["colorbar_label"],
            orientation=option["colorbar_position"],
            extend=option["extend"],
        )
    else:
        mfigsize = (len(shorter[option["groupby"]]), len(df_selected.index))
        fig, axes = plt.subplots(nrows=len(df_selected.index), ncols=1, figsize=mfigsize, sharex=True)
        fig.text(-0.01, 0.5, "Metrics", va="center", rotation="vertical", fontsize=option["ytick"] + 1)
        fig.subplots_adjust(hspace=0)
        # get the minimal and maximal value
        if not option["cmap"]:
            option["cmap"] = "coolwarm"
        custom_vmin_vmax = {}
        for i, (metric, row_data) in enumerate(df_selected.iterrows()):
            combined_dataset = _open_groupby_class_distribution(option, metric)
            quantiles = combined_dataset.quantile([0.05, 0.2, 0.8, 0.95], dim=["time", "lat", "lon"])
            # consider 0.05 and 0.95 value as the max/min value

            if not option["vmin_max_on"]:
                if metric in ["bias", "percent_bias", "rSD", "PBIAS_HF", "PBIAS_LF"]:
                    custom_vmin_vmax[metric] = [
                        quantiles[metric][0].values,
                        quantiles[metric][-1].values,
                        quantiles[metric][2].values,
                        quantiles[metric][1].values,
                    ]
                elif metric in ["NSE", "KGE", "KGESS", "correlation", "kappa_coeff", "rSpearman"]:
                    custom_vmin_vmax[metric] = [-1, 1, 0.8, -0.8]
                elif metric in ["LNSE", "ubNSE", "rNSE", "wNSE", "wsNSE"]:
                    custom_vmin_vmax[metric] = [quantiles[metric][0].values, 1, 0.8, quantiles[metric][1].values]
                elif metric in [
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
                    custom_vmin_vmax[metric] = [-1, quantiles[metric][-1].values, quantiles[metric][2].values, -0.8]
                else:
                    custom_vmin_vmax[metric] = [0, 1, 0.8, 0.2]
            else:
                custom_vmin_vmax[metric] = [option["vmin"], option["vmax"], 0.8, 0.2]

        for i, (row_name, row_data) in enumerate(df_selected.iterrows()):
            vmin, vmax = custom_vmin_vmax[row_name][0], custom_vmin_vmax[row_name][1]
            x1, x2 = custom_vmin_vmax[row_name][2], custom_vmin_vmax[row_name][3]
            im = axes[i].imshow(row_data.values.reshape(1, -1), cmap=option["cmap"], vmin=vmin, vmax=vmax)
            # Add numbers to each cell
            for j, value in enumerate(row_data):
                axes[i].text(
                    j,
                    0,
                    f"{df_selected.iloc[i, j]:{option['ticks_format']}}",
                    ha="center",
                    va="center",
                    color="white" if df_selected.iloc[i, j] > x1 or df_selected.iloc[i, j] < x2 else "black",
                    fontsize=option["fontsize"] - 1,
                )

            pos = axes[i].get_position()  # .bounds
            left, right, bottom, width, height = pos.x0, pos.x1, pos.y0, pos.width, pos.height
            cbar_ax = fig.add_axes(
                [right + 0.02, bottom + height / 2, width * 2 / len(shorter[option["groupby"]]), height / 4]
            )
            cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", extend=option["extend"])
            cbar.set_ticks([math.ceil(vmin), (vmin + vmax) / 2, math.floor(vmax)])
            cbar.set_ticklabels([f"{vmin:.1f}", f"{(vmin + vmax) / 2:.1f}", f"{vmax:.1f}"])
            cbar.ax.tick_params(labelsize=9)

            if i < len(df_selected.index) - 1:
                axes[i].get_xaxis().set_visible(False)

            if i == 0:
                axes[i].spines["bottom"].set_visible(False)
            elif 0 < i < len(df_selected.index) - 1:
                axes[i].spines["top"].set_visible(False)
                axes[i].spines["bottom"].set_visible(False)
            else:
                axes[i].spines["top"].set_visible(False)

            axes[i].set_yticks([0])
            axes[i].set_yticklabels(
                [selected_metrics[i].replace("_", " ")], rotation=option["y_rotation"], ha=option["y_ha"]
            )

        # 设置 x 轴标签
        axes[-1].set_xticks(np.arange(len(df_selected.columns)))
        if option["x_ticklabel"] == "Normal":
            axes[-1].set_xticklabels(
                [columns.replace("_", " ").title() for columns in df_selected.columns],
                rotation=option["x_rotation"],
                ha=option["x_ha"],
            )
        else:
            item = option["groupby"]
            axes[-1].set_xticklabels(
                [get_short_label(column, item) for column in df_selected.columns],
                rotation=option["x_rotation"],
                ha=option["x_ha"],
            )

        axes[-1].set_xlabel(option["xlabel"], fontsize=option["xtick"] + 1)
        axes[0].set_title(option["title"], fontsize=option["title_size"])

    file2 = file[:-4]
    save_figure(
        fig, f"{file2}_heatmap.{option['saving_format']}", format=f"{option['saving_format']}", dpi=option["dpi"]
    )
    # Close only the figure created by this renderer; closing "all" would
    # destroy unrelated figures owned by callers or concurrent renderers.
    plt.close(fig)
    # plt.show()


@with_isolated_rc
def make_CZ_based_heat_map(file, selected_metrics, lb, option):
    option = option.copy()
    selected_metrics = list(selected_metrics)
    # Convert the data to a DataFrame with fallback and auto-detection
    df = _read_metrics_file(file)

    # Convert string values to numeric, replacing 'N/A' with NaN
    df = df.apply(pd.to_numeric, errors="coerce")
    df = Convert_Type.convert_Frame(df)

    # Select the desired metrics
    df_selected = df.loc[selected_metrics]

    font = {"family": "DejaVu Sans"}
    matplotlib.rc("font", **font)
    params = {
        "axes.linewidth": option["axes_linewidth"],
        "font.size": option["fontsize"],
        "xtick.labelsize": option["xtick"],
        "xtick.direction": "out",
        "ytick.labelsize": option["ytick"],
        "grid.linewidth": 1,
        "ytick.direction": "out",
        "savefig.bbox": "tight",
        "axes.unicode_minus": False,
        "text.usetex": False,
    }
    rcParams.update(params)

    if lb == "score":
        # Create the heatmap using Matplotlib
        fig, axes = plt.subplots(nrows=2, figsize=(option["x_wise"], option["y_wise"]))
        if option["vmin_max_on"]:
            vmin, vmax = option["vmin"], option["vmax"]
        else:
            vmin, vmax = 0, 1
        if not option["cmap"]:
            option["cmap"] = "coolwarm"
        axes[0].imshow(df_selected.iloc[:, :16], cmap=option["cmap"], vmin=vmin, vmax=vmax)
        im2 = axes[1].imshow(df_selected.iloc[:, 16:], cmap=option["cmap"], vmin=vmin, vmax=vmax)

        axes[0].set_xticks(range(len(df_selected.columns[:16])))
        axes[0].set_xticklabels(
            [columns.replace("_", " ").title() for columns in df_selected.columns[:16]],
            rotation=option["x_rotation"],
            ha=option["x_ha"],
        )
        for i in range(len(df_selected.index)):
            for j in range(len(df_selected.columns[:16])):
                axes[0].text(
                    j,
                    i,
                    f"{df_selected.iloc[i, j]:{option['ticks_format']}}",
                    ha="center",
                    va="center",
                    color=_annotation_color(df_selected.iloc[i, j]),
                    fontsize=option["fontsize"],
                )

        axes[1].set_xticks(range(len(df_selected.columns[16:])))
        axes[1].set_xticklabels(
            [columns.replace("_", " ").title() for columns in df_selected.columns[16:]],
            rotation=option["x_rotation"],
            ha=option["x_ha"],
        )
        for i in range(len(df_selected.index)):
            for j in range(16, 16 + len(df_selected.columns[16:])):
                axes[1].text(
                    j - 16,
                    i,
                    f"{df_selected.iloc[i, j]:{option['ticks_format']}}",
                    ha="center",
                    va="center",
                    color=_annotation_color(df_selected.iloc[i, j]),
                    fontsize=option["fontsize"],
                )
        if len(option["title"]) == 0:
            option["title"] = f"Heatmap of {lb}"
        axes[0].set_title(option["title"], fontsize=option["title_size"])

        for ax in axes.flat:
            ax.set_yticks(range(len(df_selected.index)))
            ax.set_yticklabels(
                [index.replace("_", " ") for index in df_selected.index],
                rotation=option["y_rotation"],
                ha=option["y_ha"],
            )
            ax.set_ylabel("Scores", fontsize=option["ytick"] + 1)
            ax.set_xlabel(option["xlabel"], fontsize=option["xtick"] + 1)

        pos0 = axes[0].get_position()  # 第一行
        pos1 = axes[-1].get_position()  # 最后一行（这里就是第二行）
        left, right, bottom, width, height = pos1.x0, pos1.x1, pos1.y0, pos1.width, pos0.y1 - pos1.y0

        if not option["colorbar_position_set"]:
            if option["colorbar_position"] == "vertical":
                cbar_ax = fig.add_axes([right + 0.05, bottom, 0.03, height])  # right + 0.2
            else:
                xlabel = axes[1].xaxis.label
                xticks = axes[1].get_xticklabels()
                max_xtick_height = 0
                for xtick in xticks:
                    bbox = xtick.get_window_extent()  # 获取每个 xtick 的包围框
                    bbox_transformed = bbox.transformed(fig.transFigure.inverted())  # 将像素转换为图坐标
                    max_xtick_height = max(max_xtick_height, bbox_transformed.height)
                if xlabel is not None:
                    bbox = xlabel.get_window_extent()  # 获取每个 xtick 的包围框
                    bbox_transformed = bbox.transformed(fig.transFigure.inverted())  # 将像素转换为图坐标
                    x_height = bbox_transformed.height
                    cbar_ax = fig.add_axes(
                        [left + width / 6, bottom - max_xtick_height - x_height - 0.1, width / 3 * 2, 0.04]
                    )
                else:
                    cbar_ax = fig.add_axes([left + width / 6, bottom - max_xtick_height - 0.1, width / 3 * 2, 0.04])
        else:
            cbar_ax = _add_custom_colorbar_axes(fig, option)
        cbar = fig.colorbar(
            im2,
            cax=cbar_ax,
            label=option["colorbar_label"],
            orientation=option["colorbar_position"],
            extend=option["extend"],
        )

    elif len(df_selected.index) == 1 and lb != "score":
        fig, axes = plt.subplots(nrows=2, figsize=(option["x_wise"], option["y_wise"]))
        metric = df_selected.index[0]
        combined_dataset = _open_groupby_class_distribution(option, metric)
        quantiles = combined_dataset.quantile([0.05, 0.2, 0.8, 0.95], dim=["time", "lat", "lon"])
        # consider 0.05 and 0.95 value as the max/min value
        custom_vmin_vmax = {}
        if not option["vmin_max_on"]:
            if metric in ["bias", "percent_bias", "rSD", "PBIAS_HF", "PBIAS_LF"]:
                custom_vmin_vmax[metric] = [
                    quantiles[metric][0].values,
                    quantiles[metric][-1].values,
                    quantiles[metric][2].values,
                    quantiles[metric][1].values,
                ]
            elif metric in ["NSE", "KGE", "KGESS", "correlation", "kappa_coeff", "rSpearman"]:
                custom_vmin_vmax[metric] = [-1, 1, 0.8, -0.8]
            elif metric in ["LNSE", "ubNSE", "rNSE", "wNSE", "wsNSE"]:
                custom_vmin_vmax[metric] = [quantiles[metric][0].values, 1, 0.8, quantiles[metric][1].values]
            elif metric in [
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
                custom_vmin_vmax[metric] = [-1, quantiles[metric][-1].values, quantiles[metric][2].values, -0.8]
            else:
                custom_vmin_vmax[metric] = [0, 1, 0.8, 0.2]
        else:
            custom_vmin_vmax[metric] = [option["vmin"], option["vmax"], 0.8, 0.2]

        if not option["cmap"]:
            option["cmap"] = "coolwarm"

        vmin, vmax = custom_vmin_vmax[metric][0], custom_vmin_vmax[metric][1]
        x1, x2 = custom_vmin_vmax[metric][2], custom_vmin_vmax[metric][3]

        axes[0].imshow(df_selected.iloc[:, :16], cmap=option["cmap"], vmin=vmin, vmax=vmax)
        im2 = axes[1].imshow(df_selected.iloc[:, 16:], cmap=option["cmap"], vmin=vmin, vmax=vmax)

        for ax in axes.flat:
            ax.set_yticks(range(len(df_selected.index)))
            ax.set_yticklabels(
                [index.replace("_", " ") for index in df_selected.index],
                rotation=option["y_rotation"],
                ha=option["y_ha"],
            )
            ax.set_ylabel("Scores", fontsize=option["ytick"] + 1)
            ax.set_xlabel(option["xlabel"], fontsize=option["xtick"] + 1)

        axes[0].set_xticks(range(len(df_selected.columns[:16])))
        axes[0].set_xticklabels(
            [columns.replace("_", " ").title() for columns in df_selected.columns[:16]],
            rotation=option["x_rotation"],
            ha=option["x_ha"],
        )
        for i in range(len(df_selected.index)):
            for j in range(len(df_selected.columns[:16])):
                axes[0].text(
                    j,
                    i,
                    f"{df_selected.iloc[i, j]:{option['ticks_format']}}",
                    ha="center",
                    va="center",
                    color=_annotation_color(df_selected.iloc[i, j]),
                    fontsize=option["fontsize"],
                )
        #
        axes[1].set_xticks(range(len(df_selected.columns[16:])))
        axes[1].set_xticklabels(
            [columns.replace("_", " ").title() for columns in df_selected.columns[16:]],
            rotation=option["x_rotation"],
            ha=option["x_ha"],
        )
        for i in range(len(df_selected.index)):
            for j in range(16, 16 + len(df_selected.columns[16:])):
                axes[1].text(
                    j - 16,
                    i,
                    f"{df_selected.iloc[i, j]:{option['ticks_format']}}",
                    ha="center",
                    va="center",
                    color=_annotation_color(df_selected.iloc[i, j]),
                    fontsize=option["fontsize"],
                )
        if len(option["title"]) == 0:
            option["title"] = f"Heatmap of {lb}"
        axes[0].set_title(option["title"], fontsize=option["title_size"])

        pos0 = axes[0].get_position()  # 第一行
        pos1 = axes[-1].get_position()  # 最后一行（这里就是第二行）
        left, right, bottom, width, height = pos1.x0, pos1.x1, pos1.y0, pos1.width, pos0.y1 - pos1.y0

        if not option["colorbar_position_set"]:
            if option["colorbar_position"] == "vertical":
                cbar_ax = fig.add_axes([right + 0.05, bottom, 0.03, height])  # right + 0.2
            else:
                xlabel = axes[1].xaxis.label
                xticks = axes[1].get_xticklabels()
                max_xtick_height = 0
                for xtick in xticks:
                    bbox = xtick.get_window_extent()  # 获取每个 xtick 的包围框
                    bbox_transformed = bbox.transformed(fig.transFigure.inverted())  # 将像素转换为图坐标
                    max_xtick_height = max(max_xtick_height, bbox_transformed.height)
                if xlabel is not None:
                    bbox = xlabel.get_window_extent()  # 获取每个 xtick 的包围框
                    bbox_transformed = bbox.transformed(fig.transFigure.inverted())  # 将像素转换为图坐标
                    x_height = bbox_transformed.height
                    cbar_ax = fig.add_axes(
                        [left + width / 6, bottom - max_xtick_height - x_height - 0.1, width / 3 * 2, 0.04]
                    )
                else:
                    cbar_ax = fig.add_axes([left + width / 6, bottom - max_xtick_height - 0.1, width / 3 * 2, 0.04])
        else:
            cbar_ax = _add_custom_colorbar_axes(fig, option)
        cbar = fig.colorbar(
            im2,
            cax=cbar_ax,
            label=option["colorbar_label"],
            orientation=option["colorbar_position"],
            extend=option["extend"],
        )
    else:
        mfigsize = (15, len(df_selected.index) * 2)

        from matplotlib.gridspec import GridSpec

        nrows_original = len(df_selected.index)
        nrows_total = 2 * nrows_original

        fig = plt.figure(figsize=mfigsize)
        gs = GridSpec(nrows_total + 1, 1)  # +1 是为了在两块之间添加间隔
        axes_part1 = []
        for i in range(nrows_original):
            if i == 0:
                ax = fig.add_subplot(gs[i])
            else:
                ax = fig.add_subplot(gs[i], sharex=axes_part1[0])
            axes_part1.append(ax)

        width_scale = 16 / 15

        axes_part2 = []
        for i in range(nrows_original):
            gs_pos = nrows_original + 1 + i  # +1 是为了跳过间隔行
            if i == 0:
                ax = fig.add_subplot(gs[gs_pos])
            else:
                ax = fig.add_subplot(gs[gs_pos], sharex=axes_part2[0])
            pos = ax.get_position()
            new_width = pos.width * width_scale
            ax.set_position([pos.x0, pos.y0, new_width, pos.height])
            axes_part2.append(ax)

        fig.subplots_adjust(hspace=0)  # 每部分内部无间隔
        fig.text(-0.01, 0.5, "Metrics", va="center", rotation="vertical", fontsize=option["ytick"] + 1)

        # get the minimal and maximal value
        if not option["cmap"]:
            option["cmap"] = "coolwarm"

        custom_vmin_vmax = {}
        df_1 = df_selected.iloc[:, :16]
        for i, (metric, row_data) in enumerate(df_1.iterrows()):
            combined_dataset = _open_groupby_class_distribution(option, metric)
            quantiles = combined_dataset.quantile([0.05, 0.2, 0.8, 0.95], dim=["time", "lat", "lon"])
            # consider 0.05 and 0.95 value as the max/min value

            if not option["vmin_max_on"]:
                if metric in ["bias", "percent_bias", "rSD", "PBIAS_HF", "PBIAS_LF"]:
                    custom_vmin_vmax[metric] = [
                        quantiles[metric][0].values,
                        quantiles[metric][-1].values,
                        quantiles[metric][2].values,
                        quantiles[metric][1].values,
                    ]
                elif metric in ["NSE", "KGE", "KGESS", "correlation", "kappa_coeff", "rSpearman"]:
                    custom_vmin_vmax[metric] = [-1, 1, 0.8, -0.8]
                elif metric in ["LNSE", "ubNSE", "rNSE", "wNSE", "wsNSE"]:
                    custom_vmin_vmax[metric] = [quantiles[metric][0].values, 1, 0.8, quantiles[metric][1].values]
                elif metric in [
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
                    custom_vmin_vmax[metric] = [-1, quantiles[metric][-1].values, quantiles[metric][2].values, -0.8]
                else:
                    custom_vmin_vmax[metric] = [0, 1, 0.8, 0.2]
            else:
                custom_vmin_vmax[metric] = [option["vmin"], option["vmax"], 0.8, 0.2]

        for i, (row_name, row_data) in enumerate(df_1.iterrows()):
            vmin, vmax = custom_vmin_vmax[row_name][0], custom_vmin_vmax[row_name][1]
            x1, x2 = custom_vmin_vmax[row_name][2], custom_vmin_vmax[row_name][3]
            im = axes_part1[i].imshow(row_data.values.reshape(1, -1), cmap=option["cmap"], vmin=vmin, vmax=vmax)
            for j, value in enumerate(row_data):
                axes_part1[i].text(
                    j,
                    0,
                    f"{df_1.iloc[i, j]:{option['ticks_format']}}",
                    ha="center",
                    va="center",
                    color="white" if df_1.iloc[i, j] > x1 or df_1.iloc[i, j] < x2 else "black",
                    fontsize=option["fontsize"] - 1,
                )

            pos = axes_part1[i].get_position()  # .bounds
            left, right, bottom, width, height = pos.x0, pos.x1, pos.y0, pos.width, pos.height
            cbar_ax = fig.add_axes([right + 0.02, bottom + height / 2, width * 2 / len(df_1.columns), height / 4])
            cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", extend=option["extend"])
            cbar.set_ticks([math.ceil(vmin), (vmin + vmax) / 2, math.floor(vmax)])
            cbar.set_ticklabels([f"{vmin:.1f}", f"{(vmin + vmax) / 2:.1f}", f"{vmax:.1f}"])
            cbar.ax.tick_params(labelsize=9)

            if i < len(df_1.index) - 1:
                axes_part1[i].get_xaxis().set_visible(False)

            if i == 0:
                axes_part1[i].spines["bottom"].set_visible(False)
            elif 0 < i < len(df_1.index) - 1:
                axes_part1[i].spines["top"].set_visible(False)
                axes_part1[i].spines["bottom"].set_visible(False)
            else:
                axes_part1[i].spines["top"].set_visible(False)

            axes_part1[i].set_yticks([0])
            axes_part1[i].set_yticklabels(
                [selected_metrics[i].replace("_", " ")], rotation=option["y_rotation"], ha=option["y_ha"]
            )

        # 设置 x 轴标签
        axes_part1[-1].set_xticks(np.arange(len(df_1.columns)))

        axes_part1[-1].set_xticklabels(
            [columns.replace("_", " ").title() for columns in df_1.columns],
            rotation=option["x_rotation"],
            ha=option["x_ha"],
        )
        axes_part1[-1].set_xlabel(option["xlabel"], fontsize=option["xtick"] + 1)
        axes_part1[0].set_title(option["title"], fontsize=option["title_size"])

        df_2 = df_selected.iloc[:, 16:]
        for i, (metric, row_data) in enumerate(df_2.iterrows()):
            combined_dataset = _open_groupby_class_distribution(option, metric)
            quantiles = combined_dataset.quantile([0.05, 0.2, 0.8, 0.95], dim=["time", "lat", "lon"])
            # consider 0.05 and 0.95 value as the max/min value

            if not option["vmin_max_on"]:
                if metric in ["bias", "percent_bias", "rSD", "PBIAS_HF", "PBIAS_LF"]:
                    custom_vmin_vmax[metric] = [
                        quantiles[metric][0].values,
                        quantiles[metric][-1].values,
                        quantiles[metric][2].values,
                        quantiles[metric][1].values,
                    ]
                elif metric in ["NSE", "KGE", "KGESS", "correlation", "kappa_coeff", "rSpearman"]:
                    custom_vmin_vmax[metric] = [-1, 1, 0.8, -0.8]
                elif metric in ["LNSE", "ubNSE", "rNSE", "wNSE", "wsNSE"]:
                    custom_vmin_vmax[metric] = [quantiles[metric][0].values, 1, 0.8, quantiles[metric][1].values]
                elif metric in [
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
                    custom_vmin_vmax[metric] = [-1, quantiles[metric][-1].values, quantiles[metric][2].values, -0.8]
                else:
                    custom_vmin_vmax[metric] = [0, 1, 0.8, 0.2]
            else:
                custom_vmin_vmax[metric] = [option["vmin"], option["vmax"], 0.8, 0.2]
        for i, (row_name, row_data) in enumerate(df_2.iterrows()):
            vmin, vmax = custom_vmin_vmax[row_name][0], custom_vmin_vmax[row_name][1]
            x1, x2 = custom_vmin_vmax[row_name][2], custom_vmin_vmax[row_name][3]
            im = axes_part2[i].imshow(row_data.values.reshape(1, -1), cmap=option["cmap"], vmin=vmin, vmax=vmax)
            for j, value in enumerate(row_data):
                axes_part2[i].text(
                    j,
                    0,
                    f"{df_2.iloc[i, j]:{option['ticks_format']}}",
                    ha="center",
                    va="center",
                    color="white" if df_2.iloc[i, j] > x1 or df_2.iloc[i, j] < x2 else "black",
                    fontsize=option["fontsize"] - 1,
                )

            pos = axes_part2[i].get_position()  # .bounds
            left, right, bottom, width, height = pos.x0, pos.x1, pos.y0, pos.width, pos.height
            cbar_ax = fig.add_axes([right + 0.02, bottom + height / 2, width * 2 / len(df_2.columns), height / 4])
            cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", extend=option["extend"])
            cbar.set_ticks([math.ceil(vmin), (vmin + vmax) / 2, math.floor(vmax)])
            cbar.set_ticklabels([f"{vmin:.1f}", f"{(vmin + vmax) / 2:.1f}", f"{vmax:.1f}"])
            cbar.ax.tick_params(labelsize=9)

            if i < len(df_2.index) - 1:
                axes_part2[i].get_xaxis().set_visible(False)

            if i == 0:
                axes_part2[i].spines["bottom"].set_visible(False)
            elif 0 < i < len(df_2.index) - 1:
                axes_part2[i].spines["top"].set_visible(False)
                axes_part2[i].spines["bottom"].set_visible(False)
            else:
                axes_part2[i].spines["top"].set_visible(False)

            axes_part2[i].set_yticks([0])
            axes_part2[i].set_yticklabels(
                [selected_metrics[i].replace("_", " ")], rotation=option["y_rotation"], ha=option["y_ha"]
            )

        axes_part2[-1].set_xticks(np.arange(len(df_2.columns)))

        axes_part2[-1].set_xticklabels(
            [columns.replace("_", " ").title() for columns in df_2.columns],
            rotation=option["x_rotation"],
            ha=option["x_ha"],
        )
        axes_part2[-1].set_xlabel(option["xlabel"], fontsize=option["xtick"] + 1)

    file2 = file[:-4]
    save_figure(
        fig, f"{file2}_heatmap.{option['saving_format']}", format=f"{option['saving_format']}", dpi=option["dpi"]
    )
    # Close only the figure created by this renderer; closing "all" would
    # destroy unrelated figures owned by callers or concurrent renderers.
    plt.close(fig)
