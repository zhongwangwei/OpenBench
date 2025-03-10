import warnings
from matplotlib import colors
from matplotlib import cm
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import math
import os
import pandas as pd
import matplotlib
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib import rcParams


def get_index(vmin, vmax, colormap):
    def get_ticks(vmin, vmax):
        if 2 >= vmax - vmin > 1:
            colorbar_ticks = 0.2
        elif 5 >= vmax - vmin > 2:
            colorbar_ticks = 0.5
        elif 10 >= vmax - vmin > 5:
            colorbar_ticks = 1
        elif 20 >= vmax - vmin > 10:
            colorbar_ticks = 2
        elif 50 >= vmax - vmin > 20:
            colorbar_ticks = 5
        elif 100 >= vmax - vmin > 50:
            colorbar_ticks = 10
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

    # Calculate ticks
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

    return cmap, mticks, norm, bnd


def make_stn_plot_index(file, method_name, main_nml, sources, option):
    # read the data
    df = pd.read_csv(file, header=0)
    # loop the keys in self.variables to get the metric output
    for type, source in zip(['ref_value', 'sim_value'], sources):
        min_metric = -999.0
        max_metric = 100000.0

        ind0 = df[df['%s' % (type)] > min_metric].index
        data_select0 = df.loc[ind0]
        ind1 = data_select0[data_select0['%s' % (type)] < max_metric].index
        data_select = data_select0.loc[ind1]

        try:
            stn_lon = data_select['ref_lon'].values
            stn_lat = data_select['ref_lat'].values
        except:
            stn_lon = data_select['sim_lon'].values
            stn_lat = data_select['sim_lat'].values
        metric = data_select['%s' % (type)].values

        if not option['cmap']:
            option['cmap'] = 'coolwarm'
        min_value, max_value = np.percentile(metric, 5), np.percentile(metric, 95)
        cmap, mticks, norm, bnd = get_index(min_value, max_value, option['cmap'])
        if not option['vmin_max_on']:
            option['vmax'], option['vmin'] = mticks[-1], mticks[0]

        if min_value < option['vmin'] and max_value > option['vmax']:
            option['extend'] = 'both'
        elif min_value > option['vmin'] and max_value > option['vmax']:
            option['extend'] = 'max'
        elif min_value < option['vmin'] and max_value < option['vmax']:
            option['extend'] = 'min'
        else:
            option['extend'] = 'neither'

        # Add check for empty or all-NaN array
        if len(metric) == 0 or np.all(np.isnan(metric)):
            print(f"Warning: No valid data for {method_name}. Skipping plot.")
            return

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

        cs = ax.scatter(stn_lon, stn_lat, s=option['markersize'], c=metric, cmap=cmap, vmin=option['vmin'], vmax=option['vmax'],
                        marker=option['marker'],
                        edgecolors='none', alpha=0.9)
        coastline = cfeature.NaturalEarthFeature(
            'physical', 'coastline', '50m', edgecolor='0.6', facecolor='none')
        rivers = cfeature.NaturalEarthFeature(
            'physical', 'rivers_lake_centerlines', '110m', edgecolor='0.6', facecolor='none')
        ax.add_feature(cfeature.LAND, facecolor='0.8')
        ax.add_feature(coastline, linewidth=0.6)
        ax.add_feature(cfeature.LAKES, alpha=1, facecolor='white', edgecolor='white')
        ax.add_feature(rivers, linewidth=0.5)
        ax.gridlines(draw_labels=False, linestyle=':', linewidth=0.5, color='grey', alpha=0.8)

        if not option['set_lat_lon']:
            ax.set_extent([main_nml['min_lon'], main_nml['max_lon'], main_nml['min_lat'],
                           main_nml['max_lat']])
            ax.set_xticks(np.arange(main_nml['max_lon'], main_nml['min_lon'], -60)[::-1],
                          crs=ccrs.PlateCarree())
            ax.set_yticks(np.arange(main_nml['max_lat'], main_nml['min_lat'], -30)[::-1],
                          crs=ccrs.PlateCarree())
        else:
            ax.set_extent([option['min_lon'], option['max_lon'], option['min_lat'], option['max_lat']])
            ax.set_xticks(np.arange(option['max_lon'], option['min_lon'], -60)[::-1], crs=ccrs.PlateCarree())
            ax.set_yticks(np.arange(option['max_lat'], option['min_lat'], -30)[::-1], crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter()
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)

        ax.set_xlabel(option['xticklabel'], fontsize=option['xtick'] + 1, labelpad=20)
        ax.set_ylabel(option['yticklabel'], fontsize=option['ytick'] + 1, labelpad=50)
        title = option['title']
        if not option['title']:
            title = f'{source} {method_name}'
        plt.title(title, fontsize=option['title_size'])

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
                          extend=option['extend'],
                          orientation=option['colorbar_position'])
        cb.solids.set_edgecolor("face")

        filename2 = file[:-4]
        plt.savefig(f'{filename2}_{type}.{option["saving_format"]}', format=f'{option["saving_format"]}', dpi=option['dpi'])
        plt.close()
