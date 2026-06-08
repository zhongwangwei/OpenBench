import logging
import os
from openbench.visualization._rc_isolation import with_isolated_rc  # noqa: E402
from openbench.visualization._figure_io import save_figure

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib import rcParams

from openbench.util.converttype import Convert_Type
from openbench.util.filenames import (
    relative_grid_score_filename,
    relative_station_score_plot_stem,
    relative_station_scores_filename,
)

from .Fig_toolbox import get_index, tick_length
from ._downsample import downsample_for_plot, lat_lon_plot_args
from ._validation import finite_min_max, finite_values

logger = logging.getLogger(__name__)


def _first_existing_path(*paths: str) -> str:
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]


def _relative_station_scores_path(output_dir, evaluation_item, ref_source, sim_source):
    return _first_existing_path(
        os.path.join(output_dir, relative_station_scores_filename(evaluation_item, ref_source, sim_source)),
        os.path.join(output_dir, f"{evaluation_item}_stn_{ref_source}_{sim_source}_relative_scores.csv"),
    )


def _relative_grid_score_path(output_dir, evaluation_item, ref_source, sim_source, score):
    return _first_existing_path(
        os.path.join(output_dir, relative_grid_score_filename(evaluation_item, ref_source, sim_source, score)),
        os.path.join(output_dir, f"{evaluation_item}_ref_{ref_source}_sim_{sim_source}_Relative{score}.nc"),
    )


@with_isolated_rc
def make_stn_plot_index(file, method_name, metric, stn_lat, stn_lon, main_nml, option):
    option = option.copy()
    finite_values(metric, label=f"Relative Score station {method_name}")

    if not option["cmap"]:
        option["cmap"] = "coolwarm"
    min_value, max_value = finite_min_max(metric, label=f"Relative Score station {method_name}", percentile=(5, 95))
    cmap, mticks, norm, bnd, extend = get_index(min_value, max_value, option["cmap"], method_name)
    if not option["vmin_max_on"]:
        option["vmax"], option["vmin"] = mticks[-1], mticks[0]

    option["extend"] = extend

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

    fig = plt.figure(figsize=(option["x_wise"], option["y_wise"]))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    cs = ax.scatter(
        stn_lon,
        stn_lat,
        s=option["markersize"],
        c=metric,
        cmap=cmap,
        vmin=mticks[0],
        vmax=mticks[-1],
        marker=option["marker"],
        linewidths=0.5,
        edgecolors="black",
        alpha=0.9,
        zorder=10,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(option["line_width"])
    coastline = cfeature.NaturalEarthFeature("physical", "coastline", "110m", edgecolor="0.6", facecolor="none")
    rivers = cfeature.NaturalEarthFeature(
        "physical", "rivers_lake_centerlines", "110m", edgecolor="0.6", facecolor="none"
    )
    ax.add_feature(cfeature.LAND, facecolor="0.9")
    ax.add_feature(coastline, linewidth=0.6)
    ax.add_feature(cfeature.LAKES, alpha=1, facecolor="white", edgecolor="white", zorder=9)
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
    title = option["title"]

    ax.set_title(title, fontsize=option["title_size"], weight="bold")

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

    filename2 = file[:-4]
    save_figure(fig, f"{filename2}.{option['saving_format']}", format=f"{option['saving_format']}", dpi=option["dpi"])
    plt.close(fig)


def prepare_stn(output_dir, evaluation_item, ref_source, sim_source, scores, main_nml, option):
    # read the data
    file = _relative_station_scores_path(output_dir, evaluation_item, ref_source, sim_source)
    if os.path.exists(file):
        df = pd.read_csv(file, header=0)
        df = Convert_Type.convert_Frame(df)
        # loop the keys in self.variables to get the metric output
        min_metric = -999.0
        max_metric = 1000.0

        for score in scores:
            var = f"relative_{score}_{sim_source}"
            ind0 = df[df["%s" % (var)] > min_metric].index
            data_select0 = df.loc[ind0]
            ind1 = data_select0[data_select0["%s" % (var)] < max_metric].index
            data_select = data_select0.loc[ind1]
            try:
                stn_lon = data_select["ref_lon"].values
                stn_lat = data_select["ref_lat"].values
            except Exception:
                stn_lon = data_select["sim_lon"].values
                stn_lat = data_select["sim_lat"].values
            metric = data_select["%s" % (var)].values
            output_file = os.path.join(
                output_dir, f"{relative_station_score_plot_stem(evaluation_item, ref_source, sim_source, score)}.csv"
            )
            try:
                make_stn_plot_index(output_file, score, metric, stn_lat, stn_lon, main_nml, option)
            except Exception:
                logging.exception(f"ERROR: relative {score} {ref_source} {sim_source}")
                raise


@with_isolated_rc
def make_geo_plot_index(file, data, ilat, ilon, main_nml, option):
    option = option.copy()
    data = downsample_for_plot(data, option)
    finite_values(data, label=f"Relative Score grid {file}")
    data, ilat, ilon, lon, lat, extent, origin = lat_lon_plot_args(data)

    if not option["cmap"]:
        option["cmap"] = "coolwarm"
    min_value, max_value = finite_min_max(data, label=f"Relative Score grid {file}")
    cmap, mticks, norm, bnd, extend = get_index(min_value, max_value, option["cmap"], "Overall_Score")
    if not option["vmin_max_on"]:
        try:
            option["vmax"], option["vmin"] = mticks[-1], mticks[0]
        except Exception:
            option["vmax"], option["vmin"] = mticks[0] + 1, mticks[0]

    option["extend"] = extend

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

    fig = plt.figure(figsize=(option["x_wise"], option["y_wise"]))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    if option["show_method"] == "interpolate":
        cs = ax.contourf(lon, lat, data, levels=bnd, cmap=cmap, norm=norm, extend=extend)
    else:
        cs = ax.imshow(
            data.values if hasattr(data, "values") else data,
            cmap=cmap,
            vmin=mticks[0],
            vmax=mticks[-1],
            extent=extent,
            origin=origin,
        )

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

    if option["title"] is None:
        option["title"] = "Correlation Results"
    ax.set_xlabel(option["xticklabel"], fontsize=option["xtick"] + 1, labelpad=20)
    ax.set_ylabel(option["yticklabel"], fontsize=option["ytick"] + 1, labelpad=40)
    ax.set_title(option["title"], fontsize=option["title_size"])

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
                ax.text(-130, -40, option["colorbar_label"], fontsize=16, weight="bold", ha="center", va="bottom")
            else:
                cbaxes = fig.add_axes([left + 0.015, bottom + 0.08, 0.02, height / 3])
                ax.text(
                    -160 + 7 * tick_length(np.median(mticks)),
                    -40,
                    option["colorbar_label"],
                    fontsize=16,
                    weight="bold",
                    ha="left",
                    va="center",
                )
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
        label=option["colorbar_label"],
        extend=extend,
        orientation=option["colorbar_position"],
    )
    cb.solids.set_edgecolor("face")

    file2 = os.path.splitext(file)[0]
    save_figure(fig, f"{file2}.{option['saving_format']}", format=f"{option['saving_format']}", dpi=option["dpi"])
    plt.close(fig)


def prepare_geo(output_dir, evaluation_item, ref_source, sim_source, scores, main_nml, option):
    for score in scores:
        filename = _relative_grid_score_path(output_dir, evaluation_item, ref_source, sim_source, score)
        if os.path.exists(filename):
            logger.info(filename)
            with xr.open_dataset(filename) as ds:
                ds = Convert_Type.convert_nc(ds)
                data = ds[f"relative_{score}"]
                ilat = ds.lat.values
                ilon = ds.lon.values
                data = data.where(np.isfinite(data), np.nan)
                try:
                    make_geo_plot_index(filename, data, ilat, ilon, main_nml, option)
                except Exception:
                    logging.exception(f"ERROR: relative {score} {ref_source} {sim_source}")
                    raise


def make_scenarios_comparison_Relative_Score(
    output_dir, evaluation_item, ref_source, sim_source, scores, data_type, main_nml, option
):
    if data_type == "stn":
        prepare_stn(output_dir, evaluation_item, ref_source, sim_source, scores, main_nml, option)
    else:
        prepare_geo(output_dir, evaluation_item, ref_source, sim_source, scores, main_nml, option)
