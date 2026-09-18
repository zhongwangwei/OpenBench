import logging
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib import rcParams
from openbench.visualization._rc_isolation import with_isolated_rc  # noqa: E402
from openbench.visualization._figure_io import save_figure
from openbench.util.filenames import filename_component

from openbench.util.converttype import Convert_Type

from .Fig_toolbox import get_index
from ._validation import finite_min_max

logger = logging.getLogger(__name__)


@with_isolated_rc
def make_stn_plot_index(file, method_name, main_nml, sources, option, *, value_columns=("ref_value", "sim_value")):
    option = option.copy()
    # Snapshot the (already-copied) option so each loop iteration starts
    # from a fresh dict; the previous code mutated option["vmin"]/vmax/extend
    # inside the first iteration and the second iteration then inherited
    # those values whenever the vmin_max_on branch didn't overwrite them.
    option_base = option.copy()
    df = pd.read_csv(file, header=0)
    df = Convert_Type.convert_Frame(df)
    if len(value_columns) != len(sources):
        raise ValueError("Station plot columns and source labels must have the same length")
    for type, source in zip(value_columns, sources):
        option = option_base.copy()
        available = np.isfinite(df[type])
        data_select = df.loc[available]
        coord_prefix = "ref" if "ref_lon" in df else "sim"
        stn_lon = data_select[f"{coord_prefix}_lon"].values
        stn_lat = data_select[f"{coord_prefix}_lat"].values
        metric = data_select[type].values
        missing = df.loc[~available]

        if not option["cmap"]:
            option["cmap"] = "coolwarm"
        min_value, max_value = (
            finite_min_max(metric, label=f"{method_name} station map/{type}", percentile=(5, 95))
            if metric.size
            else (0.0, 1.0)
        )
        limits = None
        if option["vmin_max_on"]:
            limits = (option["vmin"], option["vmax"])
        elif method_name in {"Correlation", "Mann_Kendall_Trend_Test"}:
            limits = (-1.0, 1.0)
        elif method_name == "Functional_Response":
            limits = (0.0, 1.0)
        elif method_name == "Standard_Deviation":
            upper = float(np.percentile(metric, 95)) if metric.size else 1.0
            limits = (0.0, upper if upper > 0 else 1.0)
        cmap, mticks, _norm, _bnd, extend = get_index(min_value, max_value, option["cmap"], type)
        if limits is not None:
            if not np.isfinite(limits).all() or limits[0] >= limits[1]:
                raise ValueError(f"Station map requires finite increasing color limits: {limits}")
            mticks = np.linspace(*limits, 5)
            low = bool(metric.size and metric.min() < limits[0])
            high = bool(metric.size and metric.max() > limits[1])
            extend = "both" if low and high else "min" if low else "max" if high else "neither"

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
            s=option.get("markersize", 20),
            c=metric,
            cmap=cmap,
            vmin=mticks[0],
            vmax=mticks[-1],
            marker=option.get("marker", "o"),
            linewidths=0.5,
            edgecolors="black",
            alpha=0.9,
            zorder=10,
        )

        if not missing.empty:
            ax.scatter(
                missing[f"{coord_prefix}_lon"],
                missing[f"{coord_prefix}_lat"],
                s=option.get("markersize", 20),
                c="0.6",
                marker="x",
                zorder=10,
                label=f"Unavailable (n={len(missing)})",
            )
            ax.legend(loc="lower right", fontsize=option["xtick"])
        if not metric.size:
            ax.text(0.5, 0.5, "No valid station data", transform=ax.transAxes, ha="center")

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
                [main_nml["min_lon"], main_nml["max_lon"], main_nml["min_lat"], main_nml["max_lat"]],
                crs=ccrs.PlateCarree(),
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
        if not option["title"]:
            title = f"{source} {method_name}"
            if type not in ("ref_value", "sim_value"):
                title += f" {type}"
        ax.set_title(title, fontsize=option["title_size"], weight="bold")
        if metric.size:
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
                    [
                        option["colorbar_left"],
                        option["colorbar_bottom"],
                        option["colorbar_width"],
                        option["colorbar_height"],
                    ]
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
        save_figure(
            fig,
            f"{filename2}_{filename_component(type)}.{option['saving_format']}",
            format=f"{option['saving_format']}",
            dpi=option["dpi"],
        )
        plt.close(fig)
