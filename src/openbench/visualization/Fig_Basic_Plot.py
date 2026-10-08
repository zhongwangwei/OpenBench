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
from ._geo_axes import configure_geo_axis
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


def colorbar_extend(min_value, max_value, vmin, vmax):
    """Return the colour-bar extend for data spanning [min_value, max_value].

    Each end is checked on its own: data equal to a bound lie inside the
    range, so a minimum of exactly ``vmin`` must not hide values above ``vmax``.
    """
    below = min_value < vmin
    above = max_value > vmax
    if below and above:
        return "both"
    if below:
        return "min"
    if above:
        return "max"
    return "neither"


def rounded_limits(low, high):
    """Colour-bar limits from the 5th and 95th percentiles, rounded outward.

    Limits are rounded to integers. When both percentiles are smaller than one
    in magnitude (CH4 fluxes in gC m-2 day-1, area fractions) integer rounding
    would stretch the colour bar to [-1, 1] or [0, 1] and leave the map in one
    colour, so they are rounded at the first significant digit of the larger
    percentile instead.
    """
    import math

    largest = max(abs(low), abs(high))
    if largest >= 1 or largest == 0:
        return math.floor(low), math.ceil(high)
    exponent = math.floor(math.log10(largest))
    step = 10.0**exponent
    return round(math.floor(low / step) * step, -exponent), round(math.ceil(high / step) * step, -exponent)


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
        else:
            # Label the unit the data were converted to, not the declared one
            ref_display = UnitProcessing.display_unit(ref_unit, self.item)
            sim_display = UnitProcessing.display_unit(sim_unit, self.item) if sim_unit else ref_display
            if sim_display.lower() != ref_display.lower():
                logging.warning(f"Unit mismatch: ref={ref_unit}, sim={sim_unit}. Using ref unit.")
            display_unit = convert_unit(ref_display)

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
                    option["vmin"], option["vmax"] = rounded_limits(
                        float(quantiles[0].values), float(quantiles[1].values)
                    )
                    if metric == "percent_bias":
                        if option["vmax"] > 100:
                            option["vmax"] = 100
                        if option["vmin"] < -100:
                            option["vmin"] = -100
                elif metric in [
                    "NSE",
                    "KGE",
                    "KGESS",
                    "KGEln",
                    "dr",
                    "cp",
                    "APFB",
                    "correlation",
                    "kappa_coeff",
                    "rSpearman",
                ]:
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
                    option["vmin"], option["vmax"] = 0, rounded_limits(0, float(quantiles[1].values))[1]
                else:
                    option["vmin"], option["vmax"] = 0, 1

            cmap, mticks, norm, bnd, extend = get_index(option["vmin"], option["vmax"], option["cmap"], metric)
            option["extend"] = extend
            plot_map_grid(self, cmap, norm, bnd, metric, "metrics", mticks, option)
        except Exception:
            logger.exception(f"ERROR: {key} {metric} plotting error, please check!")
            raise
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
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import xarray as xr
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

    with xr.open_dataset(
        f"{self.casedir}/{k}/{self.item}_ref_{self.ref_source}_sim_{self.sim_source}_{xitem}.nc"
    ) as _ds:
        ds = _ds.load()
    ds = Convert_Type.convert_nc(ds)

    data = downsample_for_plot(ds[xitem], option)
    data, ilat, ilon, lon, lat, extent, origin = lat_lon_plot_args(data)

    var = data.values
    min_value, max_value = finite_min_max(var, label=f"{xitem} grid map")
    option["extend"] = colorbar_extend(min_value, max_value, option["vmin"], option["vmax"])

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
    configure_geo_axis(
        ax,
        option,
        (self.min_lon, self.max_lon, self.min_lat, self.max_lat),
        gridline_kwargs={"linestyle": ":", "linewidth": 0.5, "color": "grey", "alpha": 0.8},
    )
    ax.tick_params(axis="x", color="#969696", width=1.5, length=4, which="major")
    ax.tick_params(axis="y", color="#969696", width=1.5, length=4, which="major")
    ax.set_adjustable("datalim")
    ax.set_aspect("equal", adjustable="box")

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


# Older plot_stn options gave line widths and marker sizes as totals that were
# divided by the series length, calibrated for 144 steps (12 years of months).
_LEGACY_STN_SERIES_LENGTH = 144
_LEGACY_STN_LINEWIDTH_MIN = 20.0
_LEGACY_STN_MARKERSIZE_MIN = 40.0


def _stn_line_style(option, side, n_points):
    """Return (linewidth, marker, markersize) in points for one station series.

    Widths and sizes do not depend on the series length, so stations with short
    and long records are drawn alike. Markers are drawn only for series of at
    most ``marker_max_points`` steps; on denser series they hide the line.
    """
    width = float(option[f"{side}_lineswidth"])
    size = float(option[f"{side}_markersize"])
    if width > _LEGACY_STN_LINEWIDTH_MIN:
        width /= _LEGACY_STN_SERIES_LENGTH
    if size > _LEGACY_STN_MARKERSIZE_MIN:
        size /= _LEGACY_STN_SERIES_LENGTH
    marker = option[f"{side}_marker"] if n_points <= int(option.get("marker_max_points", 200)) else None
    return width, marker, size


def _isolated_indices(values):
    """Indices of finite values whose neighbours are both missing (or absent)."""
    finite = np.isfinite(np.asarray(values, dtype=float))
    before = np.concatenate(([False], finite[:-1]))
    after = np.concatenate((finite[1:], [False]))
    return np.flatnonzero(finite & ~before & ~after)


def _stn_series_style(option, side, values, n_points):
    """Line keyword arguments for one station series.

    A value with missing neighbours forms no line segment, so on a dense
    series drawn without markers it would vanish; those values keep their
    marker. Lines are never drawn across missing values.
    """
    width, marker, size = _stn_line_style(option, side, n_points)
    style = {"linewidth": width, "marker": marker, "markersize": size}
    if marker is None:
        isolated = _isolated_indices(values)
        if isolated.size:
            style.update(marker=option[f"{side}_marker"], markevery=isolated.tolist())
    return style


@with_isolated_rc
def plot_stn(self, sim, obs, ID, key, RMSE, KGESS, correlation, lat_lon):
    option = self.fig_nml["plot_stn"].copy()
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.transforms import offset_copy
    from pylab import rcParams

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
    fig, ax = plt.subplots(1, 1, figsize=(option["x_wise"], option["y_wise"]))

    # Convert cftime to pandas datetime for plotting compatibility
    obs_plot = convert_cftime_to_pandas(obs)
    sim_plot = convert_cftime_to_pandas(sim)
    n_points = max(len(sim), len(obs))
    obs_style = _stn_series_style(option, "obs", obs_plot.values, n_points)
    sim_style = _stn_series_style(option, "sim", sim_plot.values, n_points)

    obs_plot.plot.line(
        x="time",
        ax=ax,
        label="Obs",
        linestyle=linestyles[0],
        alpha=alphas[0],
        color=colors[0],
        **obs_style,
    )
    sim_plot.plot.line(
        x="time",
        ax=ax,
        label="Sim",
        linestyle=linestyles[1],
        alpha=alphas[1],
        color=colors[1],
        add_legend=True,
        **sim_style,
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
    # The metrics sit on their own row just above the axes, right-aligned, and
    # the title on the row above them, so a long title cannot run into them.
    metrics_size = option["fontsize"] - 4
    metrics_gap = 4  # points between the axes and the metrics row
    ax.text(
        1.0,
        1.0,
        f"RMSE: {RMSE:.2f}   R: {correlation:.2f}   KGESS: {KGESS:.2f}",
        transform=offset_copy(ax.transAxes, fig=fig, y=metrics_gap, units="points"),
        fontsize=metrics_size,
        horizontalalignment="right",
        verticalalignment="bottom",
    )
    if not option["title"]:
        lat = f"{abs(lat_lon[0]):.2f}°{'N' if lat_lon[0] > 0 else ('S' if lat_lon[0] < 0 else '')}"
        lon = f"{abs(lat_lon[1]):.2f}°{'E' if lat_lon[1] > 0 else ('W' if lat_lon[1] < 0 else '')}"
        option["title"] = f"ID: {ID}  ({lat}, {lon})"
    # xarray titles the axes with the series' scalar coordinates
    # ("lat = ..., lon = ..., variable = ...") in the centre slot; the left
    # title below would be drawn over it.
    ax.set_title("")
    ax.set_title("", loc="right")
    ax.set_title(
        option["title"],
        fontsize=option["title_size"],
        fontweight="bold",
        loc="left",
        pad=metrics_gap + 1.6 * metrics_size,
    )
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
    from pylab import rcParams

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
    option["extend"] = colorbar_extend(min_value, max_value, option["vmin"], option["vmax"])

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
    configure_geo_axis(
        ax,
        option,
        (self.min_lon, self.max_lon, self.min_lat, self.max_lat),
        gridline_kwargs={"linestyle": ":", "linewidth": 0.5, "color": "grey", "alpha": 0.8},
    )
    ax.tick_params(axis="x", color="#969696", width=1.5, length=4, which="major")
    ax.tick_params(axis="y", color="#969696", width=1.5, length=4, which="major")
    ax.set_adjustable("datalim")
    ax.set_aspect("equal", adjustable="box")

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

    for metric in self.metrics:
        option = self.fig_nml["make_stn_plot_index"].copy()
        option["extend"] = self.fig_nml["make_geo_plot_index"].get("extend", "both")
        logger.info(f"plotting metric: {metric}")
        # Determine the display unit with consistent handling (same logic as grid)
        display_unit = determine_display_unit(self)
        option["colorbar_label"] = metric.replace("_", "\n") + "\n" + process_unit(display_unit, display_unit, metric)
        min_metric = -999.0
        max_metric = 100000.0
        ind0 = df[df["%s" % (metric)] > min_metric].index
        data_select0 = df.loc[ind0]
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
                    option["vmin"], option["vmax"] = rounded_limits(vmin, vmax)
                    if option["vmax"] > 100:
                        option["vmax"] = 100
                    if option["vmin"] < -100:
                        option["vmin"] = -100
                elif metric in [
                    "NSE",
                    "KGE",
                    "KGESS",
                    "KGEln",
                    "dr",
                    "cp",
                    "APFB",
                    "correlation",
                    "kappa_coeff",
                    "rSpearman",
                ]:
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
                    option["vmin"], option["vmax"] = 0, rounded_limits(0, vmax)[1]
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
        ind0 = df[df["%s" % (score)] > min_score].index
        data_select0 = df.loc[ind0]
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
    import xarray as xr
    from matplotlib import rcParams

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
    configure_geo_axis(
        ax,
        option,
        (main_nml["min_lon"], main_nml["max_lon"], main_nml["min_lat"], main_nml["max_lat"]),
        gridline_kwargs={"linestyle": ":", "linewidth": 0.5, "color": "grey", "alpha": 0.8},
    )
    ax.tick_params(axis="x", color="#969696", width=1.5, length=4, which="major")
    ax.tick_params(axis="y", color="#969696", width=1.5, length=4, which="major")
    ax.set_adjustable("datalim")
    ax.set_aspect("equal", adjustable="box")

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
