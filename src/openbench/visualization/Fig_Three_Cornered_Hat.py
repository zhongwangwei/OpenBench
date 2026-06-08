import math
import os
from openbench.visualization._rc_isolation import with_isolated_rc  # noqa: E402
from openbench.visualization._figure_io import save_figure
from openbench.util.filenames import filename_component

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib import colors, rcParams

from openbench.util.converttype import Convert_Type
from openbench.visualization._validation import finite_min_max


def get_index(vmin, vmax, colormap):
    def get_ticks(vmin, vmax):
        if 2 >= vmax - vmin > 1:
            colorbar_ticks = 0.2
        elif 5 >= vmax - vmin > 2:
            colorbar_ticks = 0.5
        elif 10 >= vmax - vmin > 5:
            colorbar_ticks = 1
        elif 100 >= vmax - vmin > 10:
            colorbar_ticks = 5
        elif 100 >= vmax - vmin > 50:
            colorbar_ticks = 20
        elif 200 >= vmax - vmin > 100:
            colorbar_ticks = 20
        elif 500 >= vmax - vmin > 200:
            colorbar_ticks = 50
        elif 1000 >= vmax - vmin > 500:
            colorbar_ticks = 100
        elif 2000 >= vmax - vmin > 1000:
            colorbar_ticks = 200
        elif 10000 >= vmax - vmin > 2000:
            colorbar_ticks = 10 ** math.floor(math.log10(vmax - vmin)) / 2
        else:
            colorbar_ticks = 0.10
        return colorbar_ticks

    colorbar_ticks = get_ticks(vmin, vmax)
    ticks = matplotlib.ticker.MultipleLocator(base=colorbar_ticks)
    mticks = ticks.tick_values(vmin=vmin, vmax=vmax)
    mticks = [
        round(tick, 2) if isinstance(tick, float) and len(str(tick).split(".")[1]) > 2 else tick for tick in mticks
    ]
    if mticks[0] < vmin and mticks[-1] < vmax:
        mticks = mticks[1:]
    elif mticks[0] > vmin and mticks[-1] > vmax:
        mticks = mticks[:-1]
    elif mticks[0] < vmin and mticks[-1] > vmax:
        mticks = mticks[1:-1]

    cmap = matplotlib.colormaps.get_cmap(colormap)
    bnd = np.arange(vmin, vmax + colorbar_ticks / 2, colorbar_ticks / 2)
    norm = colors.BoundaryNorm(bnd, cmap.N)
    return mticks, norm, bnd


@with_isolated_rc
def map(file, method_name, data_sources, ilon, ilat, data, title, main_nml, option):
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

    # filename_parts = [method_name] + data_sources
    # filename = "_".join(filename_parts) + "_output"
    # file = os.path.join(output_dir, f"{method_name}", filename)

    fig = plt.figure(figsize=(option["x_wise"], option["y_wise"]))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    if not option["cmap"]:
        option["cmap"] = "coolwarm"
    mticks, norm, bnd = get_index(option["vmin"], option["vmax"], option["cmap"])

    extent = (ilon[0], ilon[-1], ilat[0], ilat[-1])

    if ilat[0] - ilat[-1] < 0:
        origin = "lower"
    else:
        origin = "upper"

    lon, lat = np.meshgrid(ilon, ilat)
    if option["show_method"] == "interpolate":
        cs = ax.contourf(lon, lat, data, levels=bnd, cmap=option["cmap"], norm=norm, extend=option["extend"])
    else:
        cs = ax.imshow(
            data, cmap=option["cmap"], vmin=option["vmin"], vmax=option["vmax"], extent=extent, origin=origin
        )

    coastline = cfeature.NaturalEarthFeature("physical", "coastline", "50m", edgecolor="0.6", facecolor="none")
    rivers = cfeature.NaturalEarthFeature(
        "physical", "rivers_lake_centerlines", "110m", edgecolor="0.6", facecolor="none"
    )
    ax.add_feature(cfeature.LAND, facecolor="0.9")
    ax.add_feature(coastline, linewidth=0.6)
    ax.add_feature(cfeature.LAKES, alpha=1, facecolor="white", edgecolor="white")
    ax.add_feature(rivers, linewidth=0.5)
    ax.gridlines(draw_labels=False, linestyle=":", linewidth=0.5, color="grey", alpha=0.8)

    if not option["set_lat_lon"]:
        ax.set_extent(
            [main_nml["min_lon"], main_nml["max_lon"], main_nml["min_lat"], main_nml["max_lat"]], crs=ccrs.PlateCarree()
        )
        ax.set_xticks(np.arange(main_nml["max_lon"], main_nml["min_lon"], -60)[::-1], crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(main_nml["max_lat"], main_nml["min_lat"], -30)[::-1], crs=ccrs.PlateCarree())
    else:
        ax.set_extent(
            [option["min_lon"], option["max_lon"], option["min_lat"], option["max_lat"]], crs=ccrs.PlateCarree()
        )
        ax.set_xticks(np.arange(option["max_lon"], option["min_lon"], -60)[::-1], crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(option["max_lat"], option["min_lat"], -30)[::-1], crs=ccrs.PlateCarree())
    ax.set_adjustable("datalim")
    ax.set_aspect("equal", adjustable="box")

    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    if not option["title"]:
        option["title"] = title
    ax.set_xlabel(option["xticklabel"], fontsize=option["xtick"] + 1, labelpad=20)
    ax.set_ylabel(option["yticklabel"], fontsize=option["ytick"] + 1, labelpad=40)
    ax.set_title(option["title"], fontsize=option["title_size"])

    if not option["colorbar_position_set"]:
        pos = ax.get_position()  # .bounds
        left, right, bottom, width, height = pos.x0, pos.x1, pos.y0, pos.width, pos.height
        if option["colorbar_position"] == "horizontal":
            if len(option["xticklabel"]) == 0:
                cbaxes = fig.add_axes([left + width / 6, bottom - 0.12, width / 3 * 2, 0.04])
            else:
                cbaxes = fig.add_axes([left + width / 6, bottom - 0.17, width / 3 * 2, 0.04])
        else:
            cbaxes = fig.add_axes([right + 0.05, bottom, 0.03, height])
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
        orientation=option["colorbar_position"],
    )
    cb.solids.set_edgecolor("face")
    # 绘制地图
    file2 = os.path.splitext(file)[0]
    save_figure(
        fig,
        f"{file2}_{filename_component(title)}.{option['saving_format']}",
        format=f"{option['saving_format']}",
        dpi=option["dpi"],
    )
    plt.close(fig)


def make_Three_Cornered_Hat(file, method_name, data_sources, main_nml, statistic_nml, option):
    option = option.copy()
    with xr.open_dataset(f"{file}") as _ds:
        ds = _ds.load()
    ds = Convert_Type.convert_nc(ds)
    ilat = ds.lat.values
    ilon = ds.lon.values

    relative_uncertainty = ds.relative_uncertainty
    uncertainty = ds.uncertainty
    variables = ds.variable.values
    # Iterate by positional index so the label arithmetic (`var_idx + 1`)
    # works regardless of whether the `variable` coord values are ints or
    # strings; relative_uncertainty[variable] still works in both cases via
    # xarray's getitem.
    for var_idx, variable in enumerate(variables):
        relative_data = relative_uncertainty[variable]
        option["extend"] = "both"
        relative_min, relative_max = finite_min_max(
            relative_data,
            label=f"Three Cornered Hat relative_uncertainty variable {var_idx + 1}",
        )
        option["vmin"], option["vmax"] = math.floor(relative_min), math.ceil(relative_max)
        map(
            file,
            method_name,
            data_sources,
            ilon,
            ilat,
            relative_data,
            f"relative_uncertainty_variable{var_idx + 1}",
            main_nml,
            option,
        )

        uncertainty_data = uncertainty[variable]
        option["extend"] = "both"
        uncertainty_min, uncertainty_max = finite_min_max(
            uncertainty_data,
            label=f"Three Cornered Hat uncertainty variable {var_idx + 1}",
        )
        option["vmin"], option["vmax"] = math.floor(uncertainty_min), math.ceil(uncertainty_max)
        map(
            file,
            method_name,
            data_sources,
            ilon,
            ilat,
            uncertainty_data,
            f"uncertainty_variable{var_idx + 1}",
            main_nml,
            option,
        )
