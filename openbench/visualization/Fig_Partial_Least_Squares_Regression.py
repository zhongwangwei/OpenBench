import math
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib import cm
from matplotlib import colors
from matplotlib import rcParams
try:
    from openbench.util.Mod_Converttype import Convert_Type
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from openbench.util.Mod_Converttype import Convert_Type

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
    mticks = [round(tick, 2) if isinstance(tick, float) and len(str(tick).split('.')[1]) > 2 else tick for tick in
              mticks]
    if mticks[0] < vmin and mticks[-1] < vmax:
        mticks = mticks[1:]
    elif mticks[0] > vmin and mticks[-1] > vmax:
        mticks = mticks[:-1]
    elif mticks[0] < vmin and mticks[-1] > vmax:
        mticks = mticks[1:-1]

    cmap = cm.get_cmap(colormap)
    bnd = np.arange(vmin, vmax + colorbar_ticks / 2, colorbar_ticks / 2)
    norm = colors.BoundaryNorm(bnd, cmap.N)
    return mticks, norm, bnd


def map(file, method_name, data_sources, ilon, ilat, data, title, main_nml, option):
    font = {'family': option['font']}
    matplotlib.rc('font', **font)

    params = {'backend': 'ps',
              'axes.labelsize': option['labelsize'],
              'grid.linewidth': 0.2,
              'font.size': option['labelsize'],
              'xtick.labelsize': option['xtick'],
              'xtick.direction': 'out',
              'ytick.labelsize': option['ytick'],
              'ytick.direction': 'out',
              'savefig.bbox': 'tight',
              'axes.unicode_minus': False,
              'text.usetex': False}
    rcParams.update(params)

    fig = plt.figure(figsize=(option['x_wise'], option['y_wise']))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    if not option['cmap']:
        option['cmap'] = 'coolwarm'
    mticks, norm, bnd = get_index(option['vmin'], option['vmax'], option['cmap'])

    extent = (ilon[0], ilon[-1], ilat[0], ilat[-1])

    if ilat[0] - ilat[-1] < 0:
        origin = 'lower'
    else:
        origin = 'upper'

    if option['show_method'] == 'interpolate':
        lon, lat = np.meshgrid(ilon, ilat)
        if 'intercepts' in title:
            cs = ax.contourf(lon, lat, data, cmap=option['cmap'], extend=option['extend'])
        else:
            cs = ax.contourf(lon, lat, data, levels=bnd, cmap=option['cmap'], norm=norm, extend=option['extend'])
    else:
        if 'intercepts' in title:
            cs = ax.imshow(data, cmap=option['cmap'], extent=extent, origin='lower')
        else:
            cs = ax.imshow(data, cmap=option['cmap'], vmin=option['vmin'], vmax=option['vmax'], extent=extent, origin='lower')
        # cs = ax.imshow(data, cmap=option['cmap'], vmin=option['vmin'], vmax=option['vmax'], extent=extent, origin=origin)


    coastline = cfeature.NaturalEarthFeature(
        'physical', 'coastline', '50m', edgecolor='0.6', facecolor='none')
    rivers = cfeature.NaturalEarthFeature(
        'physical', 'rivers_lake_centerlines', '110m', edgecolor='0.6', facecolor='none')
    ax.add_feature(cfeature.LAND, facecolor='0.9')
    ax.add_feature(coastline, linewidth=0.6)
    ax.add_feature(cfeature.LAKES, alpha=1, facecolor='white', edgecolor='white')
    ax.add_feature(rivers, linewidth=0.5)
    ax.gridlines(draw_labels=False, linestyle=':', linewidth=0.5, color='grey', alpha=0.8)

    if not option['set_lat_lon']:
        ax.set_extent([main_nml['min_lon'], main_nml['max_lon'], main_nml['min_lat'],
                       main_nml['max_lat']], crs=ccrs.PlateCarree())
        ax.set_xticks(np.arange(main_nml['max_lon'], main_nml['min_lon'], -60)[::-1],
                      crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(main_nml['max_lat'], main_nml['min_lat'], -30)[::-1],
                      crs=ccrs.PlateCarree())
    else:
        ax.set_extent([option['min_lon'], option['max_lon'], option['min_lat'], option['max_lat']], crs=ccrs.PlateCarree())
        ax.set_xticks(np.arange(option['max_lon'], option['min_lon'], -60)[::-1], crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(option['max_lat'], option['min_lat'], -30)[::-1], crs=ccrs.PlateCarree())
    ax.set_adjustable('datalim')
    ax.set_aspect('equal', adjustable='box')

    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    if option['title'] is None:
        option['title'] = f'Mann-Kendall Test Results ({title})'
    ax.set_xlabel(option['xticklabel'], fontsize=option['xtick'] + 1, labelpad=20)
    ax.set_ylabel(option['yticklabel'], fontsize=option['ytick'] + 1, labelpad=40)
    plt.title(option['title'], fontsize=option['title_size'])

    if not option['colorbar_position_set']:
        pos = ax.get_position()  # .bounds
        left, right, bottom, width, height = pos.x0, pos.x1, pos.y0, pos.width, pos.height
        if option['colorbar_position'] == 'horizontal':
            if len(option['xticklabel']) == 0:
                cbaxes = fig.add_axes([left + width / 6, bottom - 0.12, width / 3 * 2, 0.04])
            else:
                cbaxes = fig.add_axes([left + width / 6, bottom - 0.17, width / 3 * 2, 0.04])
        else:
            cbaxes = fig.add_axes([right + 0.05, bottom, 0.03, height])
    else:
        cbaxes = fig.add_axes(
            [option["colorbar_left"], option["colorbar_bottom"], option["colorbar_width"], option["colorbar_height"]])

    cb = fig.colorbar(cs, cax=cbaxes, ticks=mticks, spacing='uniform', label=option['colorbar_label'],
                      orientation=option['colorbar_position'])
    cb.solids.set_edgecolor("face")
    # 绘制地图
    file2 = file[:-3]
    plt.savefig(f'{file}_{title}.{option["saving_format"]}', format=f'{option["saving_format"]}',
                dpi=option['dpi'])
    plt.close()


def make_Partial_Least_Squares_Regression(file, method_name, data_sources, main_nml, statistic_nml,
                                          option):  # outpath, source

    # filename_parts = [method_name] + data_sources
    # filename = "_".join(filename_parts) + "_output"
    # file = os.path.join(output_dir, f"{method_name}", filename)

    info = {'best_n_components': 'both',
            'coefficients': 'both',
            'intercepts': 'both',
            'p_values': 'neither',
            'r_squared': 'neither',
            'anomaly': 'both',
            }
    nX = statistic_nml[f"{data_sources[0]}_nX"]
    ds = xr.open_dataset(f"{file}")
    ds = Convert_Type.convert_nc(ds)
    ilat = ds.lat.values
    ilon = ds.lon.values

    for var in ds.data_vars:
        if var not in ['best_n_components', 'r_squared']:
            for x in range(nX):
                data = ds[var][x]
                fvar = f'{var}_X{x + 1}'

                option['extend'] = info[var]
                if var == 'p_values':
                    option['vmin'], option['vmax'] = 0, 1
                else:
                    option['vmin'], option['vmax'] = math.floor(data.min(skipna=True).values), math.ceil(
                        data.max(skipna=True).values)
                map(file, method_name, data_sources, ilon, ilat, data, fvar, main_nml, option)
        else:
            data = ds[var]
            fvar = var

            option['extend'] = info[var]
            if var == 'r_squared':
                option['vmin'], option['vmax'] = 0, 1
            else:
                option['vmin'], option['vmax'] = math.floor(data.min(skipna=True).values), math.ceil(
                    data.max(skipna=True).values)
            map(file, method_name, data_sources, ilon, ilat, data, fvar, main_nml, option)
