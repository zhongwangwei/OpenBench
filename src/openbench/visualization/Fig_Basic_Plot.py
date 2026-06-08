import logging
import os
from openbench.visualization._rc_isolation import with_isolated_rc  # noqa: E402
from openbench.visualization._figure_io import save_figure
from openbench.util.filenames import join_filename_components

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Try to import cftime for datetime conversion
try:
    import cftime

    _HAS_CFTIME = True
except ImportError:
    _HAS_CFTIME = False
from openbench.data.unit import UnitProcessing
from openbench.util.converttype import Convert_Type

from .Fig_toolbox import convert_unit
from ._downsample import downsample_for_plot, lat_lon_plot_args
from ._validation import finite_min_max


def convert_cftime_to_pandas(data_array):
    """
    Convert cftime datetime objects to pandas datetime for plotting compatibility.

    Args:
        data_array (xr.DataArray): DataArray with potentially cftime datetime index

    Returns:
        xr.DataArray: DataArray with pandas datetime index
    """
    if "time" not in data_array.coords:
        return data_array

    time_coord = data_array.coords["time"]

    # Check if we have cftime objects
    if _HAS_CFTIME and hasattr(time_coord.values, "__iter__"):
        try:
            # Try to detect if we have cftime objects
            first_time = time_coord.values.flat[0] if hasattr(time_coord.values, "flat") else time_coord.values[0]
            if isinstance(first_time, cftime.datetime):
                # Convert cftime to pandas datetime
                pd_times = pd.to_datetime(
                    [
                        f"{t.year:04d}-{t.month:02d}-{t.day:02d}T{t.hour:02d}:{t.minute:02d}:{t.second:02d}"
                        for t in time_coord.values
                    ]
                )
                # Create a new DataArray with converted time coordinate
                return data_array.assign_coords(time=pd_times)
        except (AttributeError, TypeError, IndexError):
            # If conversion fails, try xarray's built-in conversion
            try:
                return data_array.assign_coords(time=pd.to_datetime(time_coord.values))
            except Exception:  # If all else fails, return original
                pass

    return data_array


from .Fig_toolbox import get_index, process_unit

logger = logging.getLogger(__name__)


def determine_display_unit(self):
    """
    Determine the consistent display unit for plotting.
    Handles unit standardization between reference and simulation data.
    """
    display_unit = "Unknown"
    if hasattr(self, "ref_varunit") and self.ref_varunit:
        ref_unit = self.ref_varunit.strip() if self.ref_varunit else ""
        sim_unit = self.sim_varunit.strip() if hasattr(self, "sim_varunit") and self.sim_varunit else ""

        # Special case: For evapotranspiration, standardize to mm day-1
        if "evapotranspiration" in self.item.lower():
            display_unit = convert_unit("mm day-1")
            logging.info("Using standardized unit for evapotranspiration: mm day-1")
        elif ref_unit == sim_unit or not sim_unit:
            # If units are the same or sim unit is missing, use ref unit
            display_unit = convert_unit(ref_unit)
        else:
            # If units differ, try to convert to a common base unit
            try:
                ref_data, ref_base = UnitProcessing.convert_unit(None, ref_unit.lower())
                sim_data, sim_base = UnitProcessing.convert_unit(None, sim_unit.lower())
                if ref_base == sim_base:
                    display_unit = convert_unit(ref_base)
                    logging.info(f"Converted both units to common base: {ref_base}")
                else:
                    # Fallback: use reference unit
                    display_unit = convert_unit(ref_unit)
                    logging.warning(f"Unit mismatch: ref={ref_unit}, sim={sim_unit}. Using ref unit.")
            except Exception:
                display_unit = convert_unit(ref_unit)
                logging.warning(f"Failed to convert units. Using ref unit: {ref_unit}")

    return display_unit


def make_plot_index_grid(self):
    key = self.ref_varname

    for metric in self.metrics:
        option = self.fig_nml["make_geo_plot_index"].copy()
        logger.info(f"plotting metric: {metric}")
        # Determine the display unit with consistent handling
        display_unit = determine_display_unit(self)
        option["colorbar_label"] = metric.replace("_", "\n") + "\n" + process_unit(display_unit, display_unit, metric)
        # Set default extend option if not specified
        if "extend" not in option:
            option["extend"] = "both"  # Default value

        try:
            import math

            with xr.open_dataset(
                f"{self.casedir}/metrics/{self.item}_ref_{self.ref_source}_sim_{self.sim_source}_{metric}.nc"
            ) as _ds:
                ds = _ds[metric].load()
            ds = Convert_Type.convert_nc(ds)
            quantiles = ds.quantile([0.05, 0.95], dim=["lat", "lon"])
            del ds
            if not option["vmin_max_on"]:
                if metric in ["bias", "percent_bias", "rSD", "PBIAS_HF", "PBIAS_LF"]:
                    option["vmax"] = math.ceil(quantiles[1].values)
                    option["vmin"] = math.floor(quantiles[0].values)
                    if metric == "percent_bias":
                        if option["vmax"] > 100:
                            option["vmax"] = 100
                        if option["vmin"] < -100:
                            option["vmin"] = -100
                elif metric in ["NSE", "KGE", "KGESS", "correlation", "kappa_coeff", "rSpearman"]:
                    option["vmin"], option["vmax"] = -1, 1
                elif metric in ["LNSE", "ubNSE", "rNSE", "wNSE", "wsNSE"]:
                    option["vmin"], option["vmax"] = math.floor(quantiles[0].values), 1
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
                    option["vmin"], option["vmax"] = 0, math.ceil(quantiles[1].values)
                else:
                    option["vmin"], option["vmax"] = 0, 1

            cmap, mticks, norm, bnd, extend = get_index(option["vmin"], option["vmax"], option["cmap"], metric)
            option["extend"] = extend
            plot_map_grid(self, cmap, norm, bnd, metric, "metrics", mticks, option)
        except Exception:
            logger.exception(f"ERROR: {key} {metric} plotting error, please check!")
            raise
    # print("\033[1;32m" + "=" * 80 + "\033[0m")
    for score in self.scores:
        # Skip global map plotting for nSpatialScore since it's constant globally
        if score == "nSpatialScore":
            logger.warning(f"skipping global map plotting for score: {score} (constant globally)")
            continue

        option = self.fig_nml["make_geo_plot_index"].copy()
        logger.info(f"plotting score: {score}")
        option["colorbar_label"] = score.replace("_", "\n")
        if not option["vmin_max_on"]:
            option["vmin"], option["vmax"] = 0, 1

        cmap, mticks, norm, bnd, extend = get_index(option["vmin"], option["vmax"], option["cmap"], score)
        option["extend"] = extend
        plot_map_grid(self, cmap, norm, bnd, score, "scores", mticks, option)


@with_isolated_rc
def plot_map_grid(self, colormap, normalize, levels, xitem, k, mticks, option):
    option = option.copy()
    # Plot settings
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import numpy as np
    import xarray as xr
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
    from matplotlib import rcParams

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
    with xr.open_dataset(
        f"{self.casedir}/{k}/{self.item}_ref_{self.ref_source}_sim_{self.sim_source}_{xitem}.nc"
    ) as _ds:
        ds = _ds.load()
    ds = Convert_Type.convert_nc(ds)

    data = downsample_for_plot(ds[xitem], option)
    data, ilat, ilon, lon, lat, extent, origin = lat_lon_plot_args(data)

    var = data.values
    min_value, max_value = finite_min_max(var, label=f"{xitem} grid map")
    if min_value < option["vmin"] and max_value > option["vmax"]:
        option["extend"] = "both"
    elif min_value > option["vmin"] and max_value > option["vmax"]:
        option["extend"] = "max"
    elif min_value < option["vmin"] and max_value < option["vmax"]:
        option["extend"] = "min"
    else:
        option["extend"] = "neither"

    fig = plt.figure(figsize=(option["x_wise"], option["y_wise"]))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    if option["show_method"] == "interpolate":
        cs = ax.contourf(lon, lat, var, levels=levels, cmap=colormap, norm=normalize, extend=option["extend"])
    else:
        cs = ax.imshow(var, cmap=colormap, vmin=mticks[0], vmax=mticks[-1], extent=extent, origin=origin)

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
        ax.set_extent([self.min_lon, self.max_lon, self.min_lat, self.max_lat], crs=ccrs.PlateCarree())
        ax.set_xticks(np.arange(self.max_lon, self.min_lon, -60)[:0:-1], crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(self.max_lat, self.min_lat, -30)[:0:-1], crs=ccrs.PlateCarree())
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
                # ax.text(-130, -40, option['colorbar_label'], fontsize=16, weight='bold', ha='center', va='bottom')
            else:
                cbaxes = fig.add_axes([left + 0.015, bottom + 0.08, 0.02, height / 3])
                # ax.text(left + 0.02, bottom + 0.08+height / 6, option['colorbar_label'], fontsize=16, weight='bold',
                #         ha='left', va='center')
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
        spacing="uniform",  # label= option['colorbar_label'],
        extend=option["extend"],
        orientation=option["colorbar_position"],
    )
    cb.set_label(
        option["colorbar_label"],
        rotation=0,  # 横向显示
        fontsize=16,
        weight="bold",
        labelpad=10,  # 增大 label 和 colorbar 的间距
        ha="left",  # 水平居中
        va="bottom",  # 垂直底部对齐
    )
    cb.solids.set_edgecolor("face")

    output_name = (
        f"{join_filename_components(self.item, 'ref', self.ref_source, 'sim', self.sim_source, xitem)}"
        f".{option['saving_format']}"
    )
    save_figure(
        fig,
        os.path.join(self.casedir, k, output_name),
        format=f"{option['saving_format']}",
        dpi=option["dpi"],
    )
    plt.close(fig)


@with_isolated_rc
def plot_stn(self, sim, obs, ID, key, RMSE, KGESS, correlation, lat_lon):
    option = self.fig_nml["plot_stn"].copy()
    import matplotlib
    import matplotlib.pyplot as plt
    from pylab import rcParams
    ### Plot settings

    # font = {'family': 'Times-Roman'}
    font = {"family": "DejaVu Sans"}
    matplotlib.rc("font", **font)

    params = {
        "axes.labelsize": option["labelsize"],
        "font.size": option["fontsize"],
        "legend.fontsize": option["fontsize"],
        "legend.frameon": False,
        "xtick.labelsize": option["xtick"],
        "xtick.direction": "out",
        "ytick.labelsize": option["ytick"],
        "ytick.direction": "out",
        "savefig.bbox": "tight",
        "axes.unicode_minus": False,
        "text.usetex": False,
    }
    rcParams.update(params)

    lines = [option["obs_lineswidth"], option["sim_lineswidth"]]
    alphas = [option["obs_alphas"], option["sim_alphas"]]
    linestyles = [option["obs_linestyle"], option["sim_linestyle"]]

    hex_pattern = r"^#([0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})$"
    import re

    if bool(re.match(hex_pattern, f"#{option['obs_linecolor']}")) and bool(
        re.match(hex_pattern, f"#{option['sim_linecolor']}")
    ):
        colors = [f"#{option['obs_linecolor']}", f"#{option['sim_linecolor']}"]
    else:
        colors = [option["obs_linecolor"], option["sim_linecolor"]]
    markers = [option["obs_marker"], option["sim_marker"]]
    markersizes = [option["obs_markersize"], option["sim_markersize"]]

    fig, ax = plt.subplots(1, 1, figsize=(option["x_wise"], option["y_wise"]))
    # Guard zero-length inputs: if both series are empty, dividing lines/markers
    # by 0 below would raise ZeroDivisionError and abort the figure.
    max_time_len = max(1, max(len(sim), len(obs)))

    # Convert cftime to pandas datetime for plotting compatibility
    obs_plot = convert_cftime_to_pandas(obs)
    sim_plot = convert_cftime_to_pandas(sim)

    obs_plot.plot.line(
        x="time",
        ax=ax,
        label="Obs",
        linewidth=lines[0] / max_time_len,
        linestyle=linestyles[0],
        alpha=alphas[0],
        color=colors[0],
        marker=markers[0],
        markersize=markersizes[0] / max_time_len,
    )
    sim_plot.plot.line(
        x="time",
        ax=ax,
        label="Sim",
        linewidth=lines[1] / max_time_len,
        linestyle=linestyles[1],
        alpha=alphas[1],
        color=colors[1],
        marker=markers[1],
        markersize=markersizes[1] / max_time_len,
        add_legend=True,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(option["line_width"])

    # set ylabel to be the same as the variable name
    # Use consistent unit determination logic
    display_unit = determine_display_unit(self)

    ax.set_ylabel(f"{key[0]} ({display_unit})", fontsize=option["ytick"] + 4, fontweight="bold")
    ax.set_xlabel("Date", fontsize=option["xtick"] + 4, fontweight="bold")
    # ax.tick_params(axis='both', top='off', labelsize=16)

    # ax.scatter([], [], color='black', marker='o', label=overall_label)
    ax.legend(loc="best", shadow=False, labelspacing=option["labelspacing"], fontsize=option["fontsize"])
    # add RMSE,KGE,correlation in two digital to the legend in left top
    ax.text(
        0.6,
        1.08,
        f"RMSE: {RMSE:.2f}   R: {correlation:.2f}   KGESS: {KGESS:.2f}",
        transform=ax.transAxes,
        fontsize=option["fontsize"] - 4,
        verticalalignment="top",
    )
    if not option["title"]:
        lat = f"{abs(lat_lon[0]):.2f}°{'N' if lat_lon[0] > 0 else ('S' if lat_lon[0] < 0 else '')}"
        lon = f"{abs(lat_lon[1]):.2f}°{'E' if lat_lon[1] > 0 else ('W' if lat_lon[1] < 0 else '')}"
        option["title"] = f"ID: {str(ID).title()}  ({lat}, {lon})"
    ax.set_title(option["title"], fontsize=option["title_size"], fontweight="bold", x=0, y=1.08, ha="left", va="top")
    if option["grid"]:
        ax.grid(linestyle=option["grid_linestyle"], alpha=0.7, linewidth=option["grid_width"])

    # plt.tight_layout()
    output_dir = os.path.join(self.casedir, "data", join_filename_components("stn", self.ref_source, self.sim_source))
    output_name = f"{join_filename_components(key[0], ID, 'timeseries')}.{option['saving_format']}"
    save_figure(
        fig,
        os.path.join(output_dir, output_name),
        format=f"{option['saving_format']}",
        dpi=option["dpi"],
    )
    plt.close(fig)


@with_isolated_rc
def plot_stn_map(self, stn_lon, stn_lat, metric, cmap, norm, varname, s_m, mticks, option):
    option = option.copy()
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib
    import matplotlib.pyplot as plt
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
    from pylab import rcParams

    ### Plot settings
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
    # Fail before silently omitting a requested station map.
    min_value, max_value = finite_min_max(metric, label=f"{varname} station map")

    fig = plt.figure(figsize=(option["x_wise"], option["y_wise"]))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    # set the region of the map based on self.Max_lat, self.Min_lat, self.Max_lon, self.Min_lon
    if min_value < option["vmin"] and max_value > option["vmax"]:
        option["extend"] = "both"
    elif min_value > option["vmin"] and max_value > option["vmax"]:
        option["extend"] = "max"
    elif min_value < option["vmin"] and max_value < option["vmax"]:
        option["extend"] = "min"
    else:
        option["extend"] = "neither"

    cs = ax.scatter(
        stn_lon,
        stn_lat,
        s=option["markersize"],
        c=metric,
        cmap=cmap,
        norm=norm,
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
        ax.set_extent([self.min_lon, self.max_lon, self.min_lat, self.max_lat], crs=ccrs.PlateCarree())
        ax.set_xticks(np.arange(self.max_lon, self.min_lon, -60)[:0:-1], crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(self.max_lat, self.min_lat, -30)[:0:-1], crs=ccrs.PlateCarree())
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
                # ax.text(-130, -40, option['colorbar_label'], fontsize=16, weight='bold', ha='center', va='bottom')
            else:
                cbaxes = fig.add_axes([left + 0.015, bottom + 0.08, 0.02, height / 3])
                # ax.text(-160 + 7 * tick_length(np.median(mticks)), -40, option['colorbar_label'], fontsize=16, weight='bold', ha='left', va='center')
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
    cb.set_label(
        option["colorbar_label"],
        rotation=0,  # 横向显示
        fontsize=16,
        weight="bold",
        labelpad=10,  # 增大 label 和 colorbar 的间距
        ha="left",  # 水平居中
        va="bottom",  # 垂直底部对齐
    )
    cb.solids.set_edgecolor("face")
    # cb.set_label('%s' % (varname), position=(0.5, 1.5), labelpad=-35)
    output_name = (
        f"{join_filename_components(self.item, 'stn', self.ref_source, self.sim_source, varname)}"
        f".{option['saving_format']}"
    )
    save_figure(
        fig,
        os.path.join(self.casedir, s_m, output_name),
        format=f"{option['saving_format']}",
        dpi=option["dpi"],
    )
    plt.close(fig)


def make_plot_index_stn(self):
    # read the data
    station_eval_name = f"{self.item}_stn_{self.ref_source}_{self.sim_source}_evaluations.csv"
    csv_candidates = []
    if self.metrics:
        csv_candidates.append(os.path.join(self.casedir, "metrics", station_eval_name))
    if self.scores:
        csv_candidates.append(os.path.join(self.casedir, "scores", station_eval_name))
    csv_candidates.extend(
        [
            os.path.join(self.casedir, "metrics", station_eval_name),
            os.path.join(self.casedir, "scores", station_eval_name),
        ]
    )
    csv_path = next((path for path in csv_candidates if os.path.exists(path)), None)
    if csv_path is None:
        raise FileNotFoundError(f"Station evaluation CSV not found in metrics/ or scores/: {station_eval_name}")
    df = pd.read_csv(csv_path, header=0)
    df = Convert_Type.convert_Frame(df)

    # loop the keys in self.variables to get the metric output
    for metric in self.metrics:
        option = self.fig_nml["make_stn_plot_index"].copy()
        option["extend"] = self.fig_nml["make_geo_plot_index"].get("extend", "both")
        logger.info(f"plotting metric: {metric}")
        # Determine the display unit with consistent handling (same logic as grid)
        display_unit = determine_display_unit(self)
        option["colorbar_label"] = metric.replace("_", "\n") + "\n" + process_unit(display_unit, display_unit, metric)
        min_metric = -999.0
        max_metric = 100000.0
        # print(df['%s'%(metric)])
        ind0 = df[df["%s" % (metric)] > min_metric].index
        data_select0 = df.loc[ind0]
        # print(data_select0[data_select0['%s'%(metric)] < max_metric])
        ind1 = data_select0[data_select0["%s" % (metric)] < max_metric].index
        data_select = data_select0.loc[ind1]

        try:
            lon_select = data_select["ref_lon"].values
            lat_select = data_select["ref_lat"].values
        except Exception:
            lon_select = data_select["sim_lon"].values
            lat_select = data_select["sim_lat"].values
        plotvar = data_select["%s" % (metric)].values
        if not np.isfinite(plotvar).any():
            logger.warning("skipping station map for metric %s: no finite data", metric)
            continue
        vmin, vmax = finite_min_max(plotvar, label=f"{metric} station map", percentile=(5, 95))

        try:
            import math

            if not option["vmin_max_on"]:
                if metric in ["bias", "percent_bias", "rSD", "PBIAS_HF", "PBIAS_LF"]:
                    option["vmax"] = math.ceil(vmax)
                    option["vmin"] = math.floor(vmin)
                    if option["vmax"] > 100:
                        option["vmax"] = 100
                    if option["vmin"] < -100:
                        option["vmin"] = -100
                elif metric in ["NSE", "KGE", "KGESS", "correlation", "kappa_coeff", "rSpearman"]:
                    option["vmin"], option["vmax"] = -1, 1
                elif metric in ["LNSE", "ubNSE", "rNSE", "wNSE", "wsNSE"]:
                    option["vmin"], option["vmax"] = math.floor(vmin), 1
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
                    option["vmin"], option["vmax"] = 0, math.ceil(vmax)
                else:
                    option["vmin"], option["vmax"] = 0, 1
        except Exception:
            option["vmin"], option["vmax"] = 0, 1

        cmap, mticks, norm, bnd, extend = get_index(option["vmin"], option["vmax"], option["cmap"], metric)
        option["extend"] = extend
        plot_stn_map(self, lon_select, lat_select, plotvar, cmap, norm, metric, "metrics", mticks, option)

    for score in self.scores:
        # Skip global map plotting for nSpatialScore since it's constant globally
        if score == "nSpatialScore":
            logger.warning(f"skipping station map plotting for score: {score} (constant globally)")
            continue

        option = self.fig_nml["make_stn_plot_index"].copy()
        logger.info(f"plotting score: {score}")
        option["colorbar_label"] = score.replace("_", "\n")
        min_score = -999.0
        max_score = 100000.0
        # print(df['%s'%(score)])
        ind0 = df[df["%s" % (score)] > min_score].index
        data_select0 = df.loc[ind0]
        # print(data_select0[data_select0['%s'%(score)] < max_score])
        ind1 = data_select0[data_select0["%s" % (score)] < max_score].index
        data_select = data_select0.loc[ind1]
        # if key=='discharge':
        #    #ind2 = data_select[abs(data_select['err']) < 0.001].index
        #    #data_select = data_select.loc[ind2]
        #    ind3 = data_select[abs(data_select['area1']) > 1000.].index
        #    data_select = data_select.loc[ind3]
        try:
            lon_select = data_select["ref_lon"].values
            lat_select = data_select["ref_lat"].values
        except Exception:
            lon_select = data_select["sim_lon"].values
            lat_select = data_select["sim_lat"].values
        plotvar = data_select["%s" % (score)].values
        if not np.isfinite(plotvar).any():
            logger.warning("skipping station map for score %s: no finite data", score)
            continue

        if not option["vmin_max_on"]:
            option["vmin"], option["vmax"] = 0, 1

        cmap, mticks, norm, bnd, extend = get_index(option["vmin"], option["vmax"], option["cmap"], score)
        option["extend"] = extend

        plot_stn_map(self, lon_select, lat_select, plotvar, cmap, norm, score, "scores", mticks, option)


@with_isolated_rc
def make_Basic(file, method_name, data_sources, main_nml, option):
    option = option.copy()
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import numpy as np
    import xarray as xr
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
    from matplotlib import rcParams
    # filename_parts = [method_name] + data_sources
    # filename = "_".join(filename_parts) + "_output"
    # file = os.path.join(output_dir, f"{method_name}", filename)

    with xr.open_dataset(file) as _ds:
        ds = _ds.load()
    ds = Convert_Type.convert_nc(ds)
    data = downsample_for_plot(ds[method_name], option)
    data, ilat, ilon, lon, lat, extent, origin = lat_lon_plot_args(data)

    min_value, max_value = finite_min_max(data, label=f"{method_name} basic map")
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

    if option["show_method"] == "interpolate":
        cs = ax.contourf(lon, lat, data, levels=bnd, cmap=cmap, norm=norm, extend=extend)
    else:
        cs = ax.imshow(data.values, cmap=cmap, vmin=mticks[0], vmax=mticks[-1], extent=extent, origin=origin)

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
                # ax.text(-130, -40, option['colorbar_label'], fontsize=16, weight='bold', ha='center', va='bottom')
            else:
                cbaxes = fig.add_axes([left + 0.015, bottom + 0.08, 0.02, height / 3])
                # ax.text(-160 + 7 * tick_length(np.median(mticks)), -40, option['colorbar_label'], fontsize=16, weight='bold', ha='left', va='center')
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
    cb.set_label(
        option["colorbar_label"],
        rotation=0,  # 横向显示
        fontsize=16,
        weight="bold",
        labelpad=10,  # 增大 label 和 colorbar 的间距
        ha="left",  # 水平居中
        va="bottom",  # 垂直底部对齐
    )
    cb.solids.set_edgecolor("face")

    save_figure(fig, f"{file}.{option['saving_format']}", format=f"{option['saving_format']}", dpi=option["dpi"])
    plt.close(fig)
