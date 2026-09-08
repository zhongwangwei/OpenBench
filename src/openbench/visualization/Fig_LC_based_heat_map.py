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
from .Fig_toolbox import get_index

logger = logging.getLogger(__name__)


def _metric_color_scale(option, metric):
    """Use Basic's metric ranges and Fig_toolbox's cmap, ticks, norm and extend."""
    if option["vmin_max_on"]:
        vmin, vmax = option["vmin"], option["vmax"]
    else:
        combined_dataset = _open_groupby_class_distribution(option, metric)
        data = combined_dataset[metric]
        finite = data.values[np.isfinite(data.values)]
        low, high = np.quantile(finite, [0.05, 0.95]) if finite.size else (0, 1)
        if metric in ["bias", "percent_bias", "rSD", "PBIAS_HF", "PBIAS_LF"]:
            vmin, vmax = math.floor(low), math.ceil(high)
            if metric == "percent_bias":
                vmin, vmax = max(-100, vmin), min(100, vmax)
        elif metric in ["NSE", "KGE", "KGESS", "correlation", "kappa_coeff", "rSpearman"]:
            vmin, vmax = -1, 1
        elif metric in ["LNSE", "ubNSE", "rNSE", "wNSE", "wsNSE"]:
            vmin, vmax = math.floor(low), 1
        elif metric in ["RMSE", "CRMSD", "MSE", "ubRMSE", "nRMSE", "mean_absolute_error",
                        "ssq", "ve", "absolute_percent_bias"]:
            vmin, vmax = 0, math.ceil(high)
        else:
            vmin, vmax = 0, 1
    return get_index(vmin, vmax, option.get("cmap") or "coolwarm", metric)


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

    metadata = {}
    with open(file_to_read, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
    while lines and lines[0].startswith("#"):
        key, separator, value = lines.pop(0)[1:].partition(":")
        if separator:
            metadata[key.strip()] = value.strip()
    sep = "\t" if lines and "\t" in lines[0] else ","

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

    counts = {}
    if "n_valid" in df.index:
        if df.index[-1] != "n_valid" or list(df.index).count("n_valid") != 1:
            raise ValueError(f"{file}: n_valid must appear once as the final row")
        counts = pd.to_numeric(df.loc["n_valid"], errors="raise").to_dict()
        df = df.drop(index="n_valid")
    df.attrs["metadata"] = metadata
    df.attrs["n_valid"] = counts
    return df


@with_isolated_rc
def make_LC_based_heat_map(file, selected_metrics, lb, option):
    option = option.copy()
    selected_metrics = list(selected_metrics)
    # Convert the data to a DataFrame with fallback and auto-detection
    df = _read_metrics_file(file)
    metadata = df.attrs.get("metadata", {})
    counts = df.attrs.get("n_valid", {})
    selected_metrics = [name for name in selected_metrics if name != "n_valid"]

    def class_label(column, label):
        count = counts.get(column)
        return f"{label}\nn={int(count):,}" if count is not None else label

    def metric_label(metric):
        label = metric.replace("_", " ")
        if lb == "score" or not metadata:
            return label
        from types import SimpleNamespace
        from .Fig_Basic_Plot import determine_display_unit
        from .Fig_toolbox import process_unit
        unit = determine_display_unit(SimpleNamespace(
            ref_varunit=metadata.get("ref_unit", ""), sim_varunit=metadata.get("sim_unit", ""),
            item=option.get("item", [""])[0],
        ))
        return f"{label}\n{process_unit(unit, unit, metric)}"

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

    column_labels = [
        class_label(column, column.replace("_", " ").title()
                    if option.get("x_ticklabel", "Normal") == "Normal" else get_short_label(column, option["groupby"]))
        for column in df_selected.columns
    ]
    row_labels = [metric_label(metric) for metric in df_selected.index]
    _draw_groupby_heatmap(file, df_selected, column_labels, row_labels, lb, option)


@with_isolated_rc
def make_CZ_based_heat_map(file, selected_metrics, lb, option):
    option = option.copy()
    option.setdefault("groupby", "CZ_groupby")
    make_LC_based_heat_map(file, selected_metrics, lb, option)


def _groupby_layout(column_labels, row_labels, lb, option):
    """Measure labels; keep LC continuous and split CZ after the first 16 columns.

    Options are minimum canvas sizes. Label extents determine cell spacing,
    margins at the output DPI, including the final n_valid label.
    """
    if not column_labels or not row_labels:
        raise ValueError("Groupby heatmap needs at least one class and statistic")
    panels = [slice(0, len(column_labels))]
    if option.get("groupby") == "CZ_groupby" and len(column_labels) > 16:
        panels = [slice(0, 16), slice(16, len(column_labels))]
    fig = plt.figure(figsize=(2, 2), dpi=option["dpi"])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def measure(labels, fontsize, rotation, ha):
        sizes = []
        for label in labels:
            text = fig.text(0, 0, label, fontsize=fontsize, rotation=rotation,
                            ha=ha, va="top", multialignment="center" if ha == "center" else "left")
            box = text.get_window_extent(renderer)
            sizes.append((box.width / fig.dpi, box.height / fig.dpi))
            text.remove()
        return max(size[0] for size in sizes), max(size[1] for size in sizes)

    x_width, x_height = measure(column_labels, option["xtick"], option["x_rotation"], "center")
    y_width, y_height = measure(row_labels, option["ytick"], option["y_rotation"], option["y_ha"])
    _, xlabel_height = measure([option["xlabel"]], option["xtick"] + 1, 0, "center")
    _, title_height = measure([option["title"] or f"Heatmap of {lb}"], option["title_size"], 0, "center")

    per_row_colorbar = lb != "score" or len(row_labels) == 1
    horizontal_colorbar = not per_row_colorbar and option["colorbar_position"] == "horizontal"
    left = y_width + 0.85
    # A single physical cell size fits both axes' rotated labels. Centered
    # category labels stay within the cell width, including the Overall label.
    cell_width = max(0.8, x_width + 0.16, y_height + 0.2, option["fontsize"] / 18)
    overhang = 0
    colorbar_width = 2.5 if per_row_colorbar else (1.2 if not horizontal_colorbar else 0)
    right = overhang + colorbar_width + 0.6
    max_columns = max(panel.stop - panel.start for panel in panels)
    width = max(float(option["x_wise"]), left + max_columns * cell_width + right)
    cell_width = (width - left - right) / max_columns
    # Scale text against the initial canvas expansion, once. Keep cell size
    # fixed so larger fonts do not recursively inflate the whole canvas.
    font_scale = min(1.8, max(1.0, width / max(float(option["x_wise"]), 1.0)))
    option["fontsize"] *= min(font_scale, max(1.0, cell_width * 72 * 0.28 / option["fontsize"]))
    option["title_size"] *= font_scale
    option["xtick"] *= max(1.0, min(font_scale, (cell_width - 0.1) / max(x_width, 0.01)))
    option["ytick"] *= max(1.0, min(font_scale, (cell_width - 0.12) / max(y_height, 0.01))) * 1.3
    option["colorbar_fontsize"] = max(11.0, 9 * font_scale)
    option["colorbar_auto_width"] = 2.1 * font_scale
    option["colorbar_auto_height"] = 0.18 * font_scale
    x_width, x_height = measure(column_labels, option["xtick"], option["x_rotation"], "center")
    y_width, y_height = measure(row_labels, option["ytick"], option["y_rotation"], option["y_ha"])
    _, xlabel_height = measure([option["xlabel"]], option["xtick"] + 1, 0, "center")
    _, title_height = measure([option["title"] or f"Heatmap of {lb}"], option["title_size"], 0, "center")
    left = y_width + 0.85
    right = option["colorbar_auto_width"] + 1.0 if per_row_colorbar else right
    width = left + max_columns * cell_width + right
    tick_space = x_height + xlabel_height + 0.4
    top = title_height + 0.5
    bottom = 0.3 + (0.9 if horizontal_colorbar else 0)
    row_height = cell_width
    row_count = len(row_labels) * len(panels)
    gap = 0.45
    label_space = len(panels) * tick_space + (len(panels) - 1) * gap
    height = max(float(option["y_wise"]),
                 top + bottom + row_count * row_height + label_space)
    fig.set_size_inches(width, height)
    panel_height = len(row_labels) * row_height
    panel_boxes = []
    upper = height - top
    for panel in panels:
        panel_boxes.append((left, upper - panel_height, (panel.stop - panel.start) * cell_width, panel_height))
        upper -= panel_height + tick_space + gap
    return fig, panels, panel_boxes, row_height, overhang


def _draw_groupby_heatmap(file, data, column_labels, row_labels, lb, option):
    """One renderer for LC/CZ, scores, and single/multiple metric panels."""
    option = option.copy()
    fig, panels, panel_boxes, row_height, overhang = _groupby_layout(column_labels, row_labels, lb, option)
    width, height = fig.get_size_inches()

    def add_axes(box):
        x, y, w, h = box
        return fig.add_axes([x / width, y / height, w / width, h / height])

    try:
        scales = {metric: _metric_color_scale(option, metric) for metric in data.index} if lb != "score" else {}
        per_row_colorbar = lb != "score" or len(data.index) == 1
        images = []
        for panel_index, (panel, box) in enumerate(zip(panels, panel_boxes)):
            left, bottom, panel_width, panel_height = box
            # Scores share a color scale; metrics retain a scale for each row.
            row_groups = [slice(i, i + 1) for i in range(len(data.index))] if per_row_colorbar else [slice(0, len(data.index))]
            for rows in row_groups:
                y = bottom + panel_height - rows.stop * row_height
                ax = add_axes((left, y, panel_width, (rows.stop - rows.start) * row_height))
                values = data.iloc[rows, panel]
                if lb == "score":
                    vmin, vmax = (option["vmin"], option["vmax"]) if option["vmin_max_on"] else (0, 1)
                    im = ax.imshow(values, cmap=option["cmap"] or "coolwarm", vmin=vmin, vmax=vmax, aspect="equal")
                    high, low = 0.8, 0.2
                else:
                    cmap, ticks, norm, _bounds, extend = scales[data.index[rows.start]]
                    im = ax.imshow(values, cmap=cmap, norm=norm, aspect="equal")
                    high, low = norm.vmin + 0.8 * (norm.vmax - norm.vmin), norm.vmin + 0.2 * (norm.vmax - norm.vmin)
                images.append(im)
                ax.tick_params(axis="x", labelsize=option["xtick"])
                ax.tick_params(axis="y", labelsize=option["ytick"])
                ax.set_yticks(range(len(values.index)))
                ax.set_yticklabels(row_labels[rows], rotation=option["y_rotation"], ha=option["y_ha"])
                ax.set_xticks(range(len(values.columns)))
                if rows.stop == len(data.index):
                    ax.set_xticklabels(column_labels[panel], rotation=option["x_rotation"],
                                       ha="center", va="top", multialignment="center")
                    ax.set_xlabel(option["xlabel"], fontsize=option["xtick"] + 1)
                else:
                    ax.xaxis.set_visible(False)
                    ax.spines["bottom"].set_visible(False)
                if rows.start:
                    ax.spines["top"].set_visible(False)
                if panel_index == 0 and rows.start == 0:
                    title = option["title"] or (f"Heatmap of {lb}" if lb == "score" or len(data.index) == 1 else "")
                    ax.set_title(title, fontsize=option["title_size"])
                for i in range(len(values.index)):
                    for j in range(len(values.columns)):
                        value = values.iloc[i, j]
                        ax.text(j, i, f"{value:{option['ticks_format']}}", ha="center", va="center",
                                color=_annotation_color(value, high=high, low=low),
                                fontsize=option["fontsize"] - (1 if lb != "score" and len(data.index) > 1 else 0))
                if per_row_colorbar:
                    cax = add_axes((left + panel_width + overhang + 0.25, y + row_height / 2,
                                    option["colorbar_auto_width"], option["colorbar_auto_height"]))
                    colorbar_options = {"extend": option["extend"]} if lb == "score" else {
                        "ticks": ticks, "extend": extend
                    }
                    cbar = fig.colorbar(im, cax=cax, orientation="horizontal",
                                        label=option["colorbar_label"] if len(data.index) == 1 else "",
                                        **colorbar_options)
                    cbar.ax.tick_params(labelsize=option["colorbar_fontsize"])
                    cbar.ax.xaxis.label.set_size(option["colorbar_fontsize"])

        fig.text(0.2 / width, 0.5, "Scores" if lb == "score" else "Metrics", rotation=90,
                 ha="left", va="center", fontsize=option["ytick"] + 1)
        if not per_row_colorbar:
            if option["colorbar_position_set"]:
                cax = _add_custom_colorbar_axes(fig, option)
            elif option["colorbar_position"] == "vertical":
                left, bottom, panel_width, panel_height = panel_boxes[0]
                cax = add_axes((left + panel_width + overhang + 0.3, bottom, 0.25, panel_height))
            else:
                left, _bottom, panel_width, _panel_height = panel_boxes[0]
                cax = add_axes((left + panel_width / 6, 0.55, panel_width * 2 / 3, 0.2))
            colorbar_options = {"extend": option["extend"]} if lb == "score" else {
                "ticks": scales[data.index[0]][1], "extend": scales[data.index[0]][4]
            }
            cbar = fig.colorbar(images[0], cax=cax, label=option["colorbar_label"],
                         orientation=option["colorbar_position"], **colorbar_options)
            cbar.ax.tick_params(labelsize=option["colorbar_fontsize"])
            cbar.ax.xaxis.label.set_size(option["colorbar_fontsize"])
            cbar.ax.yaxis.label.set_size(option["colorbar_fontsize"])
        save_figure(fig, f"{file[:-4]}_heatmap.{option['saving_format']}",
                    format=option["saving_format"], dpi=option["dpi"])
    finally:
        plt.close(fig)
