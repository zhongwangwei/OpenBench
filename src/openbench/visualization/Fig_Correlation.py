import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib import rcParams
from openbench.visualization._rc_isolation import with_isolated_rc  # noqa: E402
from openbench.visualization._figure_io import save_figure

from openbench.util.converttype import Convert_Type

from .Fig_toolbox import get_index


@with_isolated_rc
def make_Correlation(file, method_name, main_nml, option):
    option = option.copy()
    with xr.open_dataset(f"{file}") as _ds:
        ds = _ds.load()
    ds = Convert_Type.convert_nc(ds)
    data = ds.Correlation
    ilat = ds.lat.values
    ilon = ds.lon.values
    lon, lat = np.meshgrid(ilon, ilat)

    option["vmin"], option["vmax"] = -1, 1
    if not option["extend"]:
        option["extend"] = "neither"
    if not option["cmap"]:
        option["cmap"] = "coolwarm"

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

    cmap, mticks, norm, bnd, extend = get_index(option["vmin"], option["vmax"], option["cmap"])
    extent = (ilon[0], ilon[-1], ilat[0], ilat[-1])

    if ilat[0] - ilat[-1] < 0:
        origin = "lower"
    else:
        origin = "upper"

    if option["show_method"] == "interpolate":
        cs = ax.contourf(lon, lat, data, levels=bnd, cmap=cmap, norm=norm, extend=extend)
    else:
        cs = ax.imshow(data, cmap=cmap, vmin=mticks[0], vmax=mticks[-1], extent=extent, origin=origin)

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

    if option["title"] is None:
        option["title"] = "Correlation Results"
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

    file2 = os.path.splitext(file)[0]
    save_figure(fig, f"{file2}.{option['saving_format']}", format=f"{option['saving_format']}", dpi=option["dpi"])
    plt.close(fig)
