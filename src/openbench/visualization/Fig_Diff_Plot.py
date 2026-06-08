import logging
import math
import os
from openbench.visualization._rc_isolation import with_isolated_rc  # noqa: E402
from openbench.visualization._figure_io import save_figure

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Plot settings
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib import rcParams

from openbench.util.converttype import Convert_Type
from openbench.util.filenames import (
    diff_grid_anomaly_filename,
    diff_grid_difference_filename,
    diff_station_anomaly_filename,
    diff_station_difference_filename,
)

from .Fig_toolbox import get_index, process_unit
from ._downsample import downsample_for_plot
from ._validation import finite_min_max, finite_values

logger = logging.getLogger(__name__)


def _legacy_diff_filename(data_type, item_type, evaluation_item, ref_source, sim_source, sim_nml, ref_data_type):
    if ref_data_type == "stn":
        if data_type == "anomaly":
            return f"{evaluation_item}_stn_{ref_source}_sim_{sim_source}_{item_type}_anomaly.csv"
        sim1, sim2 = sim_source
        sim_varname_1 = sim_nml[f"{evaluation_item}"][f"{sim1}_varname"]
        sim_varname_2 = sim_nml[f"{evaluation_item}"][f"{sim2}_varname"]
        return (
            f"{evaluation_item}_stn_{ref_source}_{sim1}_{sim_varname_1}_vs_{sim2}_{sim_varname_2}_{item_type}_diff.csv"
        )

    if data_type == "anomaly":
        return f"{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{item_type}_anomaly.nc"
    sim1, sim2 = sim_source
    return f"{evaluation_item}_ref_{ref_source}_{sim1}_vs_{sim2}_{item_type}_diff.nc"


def _safe_diff_filename(data_type, item_type, evaluation_item, ref_source, sim_source, sim_nml, ref_data_type):
    if ref_data_type == "stn":
        if data_type == "anomaly":
            return diff_station_anomaly_filename(evaluation_item, ref_source, sim_source, item_type)
        sim1, sim2 = sim_source
        sim_varname_1 = sim_nml[f"{evaluation_item}"][f"{sim1}_varname"]
        sim_varname_2 = sim_nml[f"{evaluation_item}"][f"{sim2}_varname"]
        return diff_station_difference_filename(
            evaluation_item, ref_source, sim1, sim_varname_1, sim2, sim_varname_2, item_type
        )

    if data_type == "anomaly":
        return diff_grid_anomaly_filename(evaluation_item, ref_source, sim_source, item_type)
    sim1, sim2 = sim_source
    return diff_grid_difference_filename(evaluation_item, ref_source, sim1, sim2, item_type)


def _diff_input_filename(
    basedir, data_type, item_type, evaluation_item, ref_source, sim_source, sim_nml, ref_data_type
):
    """Return the safe Diff Plot input filename, falling back to pre-migration legacy names."""
    safe_filename = _safe_diff_filename(
        data_type, item_type, evaluation_item, ref_source, sim_source, sim_nml, ref_data_type
    )
    if os.path.exists(os.path.join(basedir, safe_filename)):
        return safe_filename
    return _legacy_diff_filename(data_type, item_type, evaluation_item, ref_source, sim_source, sim_nml, ref_data_type)


@with_isolated_rc
def plot_grid_map(basedir, filename, main_nml, metric, xitem, option):
    option = option.copy()
    font = {"family": option["font"]}
    matplotlib.rc("font", **font)

    params = {
        "axes.labelsize": option["labelsize"],
        "grid.linewidth": 0.2,
        "font.size": option["labelsize"],
        "xtick.labelsize": option["xtick"],
        "xtick.direction": "out",
        "ytick.labelsize": option["ytick"],
        "ytick.direction": "out",
        "savefig.bbox": "tight",
        "axes.unicode_minus": False,
        "text.usetex": False,
    }
    rcParams.update(params)

    # Set the region of the map based on self.Max_lat, self.Min_lat, self.Max_lon, self.Min_lon
    with xr.open_dataset(f"{basedir}/{filename}") as _ds:
        ds = _ds.load()
    ds = Convert_Type.convert_nc(ds)

    data = downsample_for_plot(ds[xitem], option)
    finite_values(data, label=f"Diff Plot grid {filename}/{xitem}")

    # Extract variables after plot-only downsampling.
    ilat = data.lat.values
    ilon = data.lon.values
    lat, lon = np.meshgrid(ilat[::-1], ilon)

    var = data.transpose("lon", "lat")[:, ::-1].values
    if not option["vmin_max_on"]:
        if metric in [
            "bias",
            "percent_bias",
            "rSD",
            "PBIAS_HF",
            "PBIAS_LF",
            "NSE",
            "KGE",
            "KGESS",
            "correlation",
            "kappa_coeff",
            "rSpearman",
        ]:
            min_value, max_value = finite_min_max(data, label=f"Diff Plot grid {filename}/{xitem}", percentile=(5, 95))
            max_value = math.ceil(max_value)
            min_value = math.floor(min_value)
            if metric == "percent_bias":
                if max_value > 100:
                    max_value = 100
                if min_value < -100:
                    min_value = -100
        else:
            min_value, max_value = finite_min_max(var, label=f"Diff Plot grid {filename}/{xitem}")
    else:
        min_value, max_value = option["vmin"], option["vmax"]

    cmap, mticks, norm, bnd, extend = get_index(min_value, max_value, option["cmap"])
    option["vmin"], option["vmax"] = mticks[0], mticks[-1]
    # `get_index` already computed `extend` against the *original* min/max
    # (before we overwrote option["vmin"]/vmax with mticks bounds); reuse it
    # instead of re-deriving from the now-clamped bounds, which made the
    # "min"/"both" branches unreachable.
    option["extend"] = extend

    fig = plt.figure(figsize=(option["x_wise"], option["y_wise"]))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    extent = (ilon[0], ilon[-1], ilat[0], ilat[-1])

    if ilat[0] - ilat[-1] < 0:
        origin = "lower"
    else:
        origin = "upper"

    if option["show_method"] == "interpolate":
        cs = ax.contourf(lon, lat, var, levels=bnd, cmap=cmap, norm=norm, extend=extend)
    else:
        cs = ax.imshow(var.T, cmap=cmap, vmin=mticks[0], vmax=mticks[-1], extent=extent, origin=origin)

    for spine in ax.spines.values():
        spine.set_linewidth(option["line_width"])

    coastline = cfeature.NaturalEarthFeature("physical", "coastline", "110m", edgecolor="0.6", facecolor="none")
    rivers = cfeature.NaturalEarthFeature(
        "physical", "rivers_lake_centerlines", "110m", edgecolor="0.6", facecolor="none"
    )
    ax.add_feature(cfeature.LAND, facecolor="0.9")
    ax.add_feature(coastline, linewidth=0.6)
    ax.add_feature(cfeature.LAKES, alpha=1, facecolor="white", edgecolor="white")
    ax.add_feature(rivers, linewidth=0.5)
    ax.gridlines(
        draw_labels=False,
        linestyle=":",
        linewidth=0.5,
        color="grey",
        alpha=0.8,
        xlocs=np.arange(option["max_lon"], option["min_lon"], -60)[:0:-1],
        ylocs=np.arange(option["max_lat"], option["min_lat"], -30)[:0:-1],
    )

    if not option["set_lat_lon"]:
        ax.set_extent(
            [main_nml["min_lon"], main_nml["max_lon"], main_nml["min_lat"], main_nml["max_lat"]], crs=ccrs.PlateCarree()
        )
        ax.set_xticks(np.arange(main_nml["max_lon"], main_nml["min_lon"], -60)[:0:-1], crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(main_nml["max_lat"], main_nml["min_lat"], -30)[:0:-1], crs=ccrs.PlateCarree())
    else:
        ax.set_extent(
            [option["min_lon"], option["max_lon"], option["min_lat"], option["max_lat"]], crs=ccrs.PlateCarree()
        )
        ax.set_xticks(np.arange(option["max_lon"], option["min_lon"], -60)[:0:-1], crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(option["max_lat"], option["min_lat"], -30)[:0:-1], crs=ccrs.PlateCarree())
    ax.tick_params(axis="x", color="#969696", width=1.5, length=4, which="major")
    ax.tick_params(axis="y", color="#969696", width=1.5, length=4, which="major")
    ax.set_adjustable("datalim")
    ax.set_aspect("equal", adjustable="box")
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    ax.set_xlabel(option["xticklabel"], fontsize=option["xtick"] + 1, labelpad=20)
    ax.set_ylabel(option["yticklabel"], fontsize=option["ytick"] + 1, labelpad=40)
    ax.set_title(option["title"], fontsize=option["title_size"], weight="bold")

    if not option["colorbar_position_set"]:
        pos = ax.get_position()
        left, right, bottom, width, height = pos.x0, pos.x1, pos.y0, pos.width, pos.height
        if (
            (option["min_lat"] < -60)
            & (option["max_lat"] > 89)
            & (option["min_lon"] < -179)
            & (option["max_lon"] > 179)
        ):
            if option["colorbar_position"] == "horizontal":
                cbaxes = fig.add_axes([left + 0.03, bottom + 0.14, 0.15, 0.02])
            else:
                cbaxes = fig.add_axes([left + 0.015, bottom + 0.08, 0.02, height / 3])
        else:
            if option["colorbar_position"] == "horizontal":
                if len(option["xticklabel"]) == 0:
                    cbaxes = fig.add_axes([left + width / 8, bottom - 0.1, width / 4 * 3, 0.03])
                else:
                    cbaxes = fig.add_axes([left + width / 8, bottom - 0.15, width / 4 * 3, 0.03])
            else:
                cbaxes = fig.add_axes([right + 0.01, bottom, 0.015, height])
    else:
        cbaxes = fig.add_axes(
            [option["colorbar_left"], option["colorbar_bottom"], option["colorbar_width"], option["colorbar_height"]]
        )
    cb = fig.colorbar(
        cs,
        cax=cbaxes,
        ticks=mticks,
        spacing="uniform",
        label="",
        extend=extend,
        orientation=option["colorbar_position"],
    )
    cb.solids.set_edgecolor("face")

    filename2 = filename[:-3]
    save_figure(
        fig, f"{basedir}/{filename2}.{option['saving_format']}", format=f"{option['saving_format']}", dpi=option["dpi"]
    )
    plt.close(fig)


@with_isolated_rc
def plot_stn_map(basedir, filename, stn_lon, stn_lat, metric, main_nml, var, varname, option):
    option = option.copy()
    font = {"family": option["font"]}
    matplotlib.rc("font", **font)

    params = {
        "axes.labelsize": option["labelsize"],
        "grid.linewidth": 0.2,
        "font.size": option["labelsize"],
        "xtick.labelsize": option["xtick"],
        "xtick.direction": "out",
        "ytick.labelsize": option["ytick"],
        "ytick.direction": "out",
        "savefig.bbox": "tight",
        "axes.unicode_minus": False,
        "text.usetex": False,
    }
    rcParams.update(params)
    finite_values(metric, label=f"Diff Plot station {filename}/{varname}")

    if not option["vmin_max_on"]:
        if var in [
            "bias",
            "percent_bias",
            "rSD",
            "PBIAS_HF",
            "PBIAS_LF",
            "NSE",
            "KGE",
            "KGESS",
            "correlation",
            "kappa_coeff",
            "rSpearman",
        ]:
            min_value, max_value = finite_min_max(
                metric, label=f"Diff Plot station {filename}/{varname}", percentile=(5, 95)
            )
            max_value = math.ceil(max_value)
            min_value = math.floor(min_value)
            if var == "percent_bias":
                if max_value > 100:
                    max_value = 100
                if min_value < -100:
                    min_value = -100
        else:
            min_value, max_value = finite_min_max(metric, label=f"Diff Plot station {filename}/{varname}")
    else:
        min_value, max_value = option["vmin"], option["vmax"]

    cmap, mticks, norm, bnd, extend = get_index(min_value, max_value, option["cmap"])
    option["vmin"], option["vmax"] = mticks[0], mticks[-1]

    fig = plt.figure(figsize=(option["x_wise"], option["y_wise"]))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    cs = ax.scatter(
        stn_lon,
        stn_lat,
        s=option["markersize"],
        c=metric,
        cmap=cmap,
        norm=norm,
        vmin=mticks[0],
        vmax=mticks[-1],
        marker=option["marker"],
        linewidths=0.5,
        edgecolors="black",
        alpha=0.9,
    )
    coastline = cfeature.NaturalEarthFeature("physical", "coastline", "110m", edgecolor="0.6", facecolor="none")
    rivers = cfeature.NaturalEarthFeature(
        "physical", "rivers_lake_centerlines", "110m", edgecolor="0.6", facecolor="none"
    )
    ax.add_feature(cfeature.LAND, facecolor="0.9")
    ax.add_feature(coastline, linewidth=0.6)
    ax.add_feature(cfeature.LAKES, alpha=1, facecolor="white", edgecolor="white")
    ax.add_feature(rivers, linewidth=0.5)
    ax.gridlines(
        draw_labels=False,
        linestyle=":",
        linewidth=0.5,
        color="grey",
        alpha=0.8,
        xlocs=np.arange(option["max_lon"], option["min_lon"], -60)[:0:-1],
        ylocs=np.arange(option["max_lat"], option["min_lat"], -30)[:0:-1],
    )

    if not option["set_lat_lon"]:
        ax.set_extent(
            [main_nml["min_lon"], main_nml["max_lon"], main_nml["min_lat"], main_nml["max_lat"]], crs=ccrs.PlateCarree()
        )
        ax.set_xticks(np.arange(main_nml["max_lon"], main_nml["min_lon"], -60)[:0:-1], crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(main_nml["max_lat"], main_nml["min_lat"], -30)[:0:-1], crs=ccrs.PlateCarree())
    else:
        ax.set_extent(
            [option["min_lon"], option["max_lon"], option["min_lat"], option["max_lat"]], crs=ccrs.PlateCarree()
        )
        ax.set_xticks(np.arange(option["max_lon"], option["min_lon"], -60)[:0:-1], crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(option["max_lat"], option["min_lat"], -30)[:0:-1], crs=ccrs.PlateCarree())

    ax.tick_params(axis="x", color="#969696", width=1.5, length=4, which="major")
    ax.tick_params(axis="y", color="#969696", width=1.5, length=4, which="major")
    ax.set_adjustable("datalim")
    ax.set_aspect("equal", adjustable="box")
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    ax.set_xlabel(option["xticklabel"], fontsize=option["xtick"] + 1, labelpad=20)
    ax.set_ylabel(option["yticklabel"], fontsize=option["ytick"] + 1, labelpad=50)
    ax.set_title(option["title"], fontsize=option["title_size"], weight="bold")

    if not option["colorbar_position_set"]:
        pos = ax.get_position()
        left, right, bottom, width, height = pos.x0, pos.x1, pos.y0, pos.width, pos.height
        if (
            (option["min_lat"] < -60)
            & (option["max_lat"] > 89)
            & (option["min_lon"] < -179)
            & (option["max_lon"] > 179)
        ):
            if option["colorbar_position"] == "horizontal":
                cbaxes = fig.add_axes([left + 0.03, bottom + 0.14, 0.15, 0.02])
            else:
                cbaxes = fig.add_axes([left + 0.015, bottom + 0.08, 0.02, height / 3])
        else:
            if option["colorbar_position"] == "horizontal":
                if len(option["xticklabel"]) == 0:
                    cbaxes = fig.add_axes([left + width / 8, bottom - 0.1, width / 4 * 3, 0.03])
                else:
                    cbaxes = fig.add_axes([left + width / 8, bottom - 0.15, width / 4 * 3, 0.03])
            else:
                cbaxes = fig.add_axes([right + 0.01, bottom, 0.015, height])
    else:
        cbaxes = fig.add_axes(
            [option["colorbar_left"], option["colorbar_bottom"], option["colorbar_width"], option["colorbar_height"]]
        )

    cb = fig.colorbar(
        cs,
        cax=cbaxes,
        ticks=mticks,
        spacing="uniform",
        label="",
        extend=option["extend"],
        orientation=option["colorbar_position"],
    )
    cb.solids.set_edgecolor("face")
    # cb.set_label('%s' % (varname), position=(0.5, 1.5), labelpad=-35)
    filename2 = filename[:-4]
    save_figure(
        fig, f"{basedir}/{filename2}.{option['saving_format']}", format=f"{option['saving_format']}", dpi=option["dpi"]
    )
    plt.close(fig)


# Add plotting function for anomalies and differences
def plot_diff_results(
    basedir, data_type, item_type, evaluation_item, ref_source, sim_source, main_nml, sim_nml, ref_data_type, option
):
    """
    Plot anomalies or differences for metrics/scores
    data_type: 'anomaly' or 'difference'
    item_type: 'metric' or 'score'
    """
    plot_option = option.copy()
    filename = _diff_input_filename(
        basedir, data_type, item_type, evaluation_item, ref_source, sim_source, sim_nml, ref_data_type
    )

    # plot_option.update(option)
    # Set plot parameters based on data type
    if data_type == "anomaly":
        plot_option["title"] = f"{evaluation_item} {item_type} anomaly for {sim_source}"
        # if not plot_option['colorbar_label']:
        unit = sim_nml[f"{evaluation_item}"][f"{sim_source}_varunit"]
        plot_option["colorbar_label"] = process_unit(unit, "", item_type)
    else:
        plot_option["title"] = f"{evaluation_item} {item_type} difference {sim_source[0]} vs {sim_source[1]}"
        # if not plot_option['colorbar_label']:
        unit = sim_nml[f"{evaluation_item}"][f"{sim_source[0]}_varunit"]
        plot_option["colorbar_label"] = process_unit(unit, "", item_type)

    if not plot_option["cmap"]:
        plot_option["cmap"] = "RdBu_r"  # Diverging colormap for anomalies/differences

    # For station data
    if ref_data_type == "stn":
        data = pd.read_csv(f"{basedir}/{filename}", header=0)
        data = Convert_Type.convert_Frame(data)
        lon_select = data["lon"].values
        lat_select = data["lat"].values
        plotvar = data[f"{item_type}_{'anomaly' if data_type == 'anomaly' else 'diff'}"].values
        plot_stn_map(
            basedir,
            filename,
            lon_select,
            lat_select,
            plotvar,
            main_nml,
            item_type,
            f"{data_type}_{item_type}",
            plot_option,
        )

    # For gridded data
    else:  # xarray Dataset
        plot_grid_map(
            basedir,
            filename,
            main_nml,
            item_type,
            f"{item_type}_{'anomaly' if data_type == 'anomaly' else 'diff'}",
            plot_option,
        )


def make_scenarios_comparison_Diff_Plot(
    basedir, metrics, scores, evaluation_item, ref_source, sim_sources, main_nml, sim_nml, ref_data_type, option
):
    for metric in metrics:
        for sim_source in sim_sources:
            # try:
            plot_diff_results(
                basedir,
                "anomaly",
                metric,
                evaluation_item,
                ref_source,
                sim_source,
                main_nml,
                sim_nml,
                ref_data_type,
                option,
            )
        # except:
        #     logging.error(f'{evaluation_item}:{metric} - {ref_source} {sim_source} anomaly error')
        # After calculating differences for metrics
        if len(sim_sources) >= 2:
            for i, sim1 in enumerate(sim_sources):
                for j, sim2 in enumerate(sim_sources[i + 1 :], i + 1):
                    try:
                        plot_diff_results(
                            basedir,
                            "difference",
                            metric,
                            evaluation_item,
                            ref_source,
                            (sim1, sim2),
                            main_nml,
                            sim_nml,
                            ref_data_type,
                            option,
                        )
                    except Exception:
                        logging.exception(
                            f"{evaluation_item}:{metric} - {ref_source} {sim1} vs {sim2} difference error"
                        )
                        raise

    for score in scores:
        # After calculating anomalies for scores
        for sim_source in sim_sources:
            try:
                plot_diff_results(
                    basedir,
                    "anomaly",
                    score,
                    evaluation_item,
                    ref_source,
                    sim_source,
                    main_nml,
                    sim_nml,
                    ref_data_type,
                    option,
                )
            except Exception:
                logging.exception(f"{evaluation_item}:{score} - {ref_source} {sim_source} anomaly error")
                raise

        # After calculating differences for scores
        if len(sim_sources) >= 2:
            for i, sim1 in enumerate(sim_sources):
                for j, sim2 in enumerate(sim_sources[i + 1 :], i + 1):
                    try:
                        plot_diff_results(
                            basedir,
                            "difference",
                            score,
                            evaluation_item,
                            ref_source,
                            (sim1, sim2),
                            main_nml,
                            sim_nml,
                            ref_data_type,
                            option,
                        )
                    except Exception:
                        logging.exception(f"{evaluation_item}:{score} - {ref_source} {sim1} vs {sim2} difference error")
                        raise
