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
try:
    from openbench.util.Mod_Converttype import Convert_Type
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from openbench.util.Mod_Converttype import Convert_Type

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
import cmaps
from .Fig_toolbox import get_index, convert_unit, get_colormap, process_unit, tick_length

def make_stn_plot_index(file, method_name, main_nml, sources, option):
    # read the data
    df = pd.read_csv(file, header=0)
    df = Convert_Type.convert_Frame(df)
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
        cmap, mticks, norm, bnd, extend = get_index(min_value, max_value, option['cmap'])
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

        cs = ax.scatter(stn_lon, stn_lat, s=option['markersize'], c=metric, cmap=cmap, norm=norm, vmin=option['vmin'], vmax=option['vmax'],
                        marker=option['marker'], linewidths=0.5,
                        edgecolors='black', alpha=0.9, zorder=10)
        
        for spine in ax.spines.values():
            spine.set_linewidth(option['line_width'])

        coastline = cfeature.NaturalEarthFeature(
            'physical', 'coastline', '110m', edgecolor='0.6', facecolor='none')
        rivers = cfeature.NaturalEarthFeature(
            'physical', 'rivers_lake_centerlines', '110m', edgecolor='0.6', facecolor='none')
        ax.add_feature(cfeature.LAND, facecolor='0.9')
        ax.add_feature(coastline, linewidth=0.6)
        ax.add_feature(cfeature.LAKES, alpha=1, facecolor='white', edgecolor='white', zorder=9)
        ax.add_feature(rivers, linewidth=0.5)
        ax.gridlines(draw_labels=False, linestyle=':', linewidth=0.5, color='grey', alpha=0.8,
            xlocs=np.arange(option['max_lon'], option['min_lon'], -60)[:0:-1],
            ylocs=np.arange(option['max_lat'], option['min_lat'], -30)[:0:-1])

        if not option['set_lat_lon']:
            ax.set_extent([main_nml['min_lon'], main_nml['max_lon'], main_nml['min_lat'],
                           main_nml['max_lat']], crs=ccrs.PlateCarree())
            ax.set_xticks(np.arange(main_nml['max_lon'], main_nml['min_lon'], -60)[:0:-1],
                          crs=ccrs.PlateCarree())
            ax.set_yticks(np.arange(main_nml['max_lat'], main_nml['min_lat'], -30)[:0:-1],
                          crs=ccrs.PlateCarree())
        else:
            ax.set_extent([option['min_lon'], option['max_lon'], option['min_lat'], option['max_lat']], crs=ccrs.PlateCarree())
            ax.set_xticks(np.arange(option['max_lon'], option['min_lon'], -60)[:0:-1], crs=ccrs.PlateCarree())
            ax.set_yticks(np.arange(option['max_lat'], option['min_lat'], -30)[:0:-1], crs=ccrs.PlateCarree())
        ax.tick_params(axis='x', color="#969696", width=1.5, length=4,which='major')  
        ax.tick_params(axis='y', color="#969696", width=1.5, length=4,which='major')
        ax.set_adjustable('datalim')
        ax.set_aspect('equal', adjustable='box')

        lon_formatter = LongitudeFormatter()
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)

        ax.set_xlabel(option['xticklabel'], fontsize=option['xtick'] + 1, labelpad=20)
        ax.set_ylabel(option['yticklabel'], fontsize=option['ytick'] + 1, labelpad=40)
        title = option['title']
        if not option['title']:
            title = f'{source} {method_name}'
        plt.title(title, fontsize=option['title_size'], weight='bold')
    if not option['colorbar_position_set']:
        pos = ax.get_position()
        left, right, bottom, width, height = pos.x0, pos.x1, pos.y0, pos.width, pos.height
        if (option['min_lat']<-60) & (option['max_lat']>89) & (option['min_lon']<-179) & (option['max_lon']>179):
            if option['colorbar_position'] == 'horizontal':
                cbaxes = fig.add_axes([left + 0.03, bottom+ 0.14, 0.15, 0.02])
            else:
                cbaxes = fig.add_axes([left + 0.015, bottom + 0.08, 0.02, height/3])
        else:
            if option['colorbar_position'] == 'horizontal':
                if len(option['xticklabel']) == 0:
                    cbaxes = fig.add_axes([left + width / 8, bottom - 0.1, width/4*3, 0.03])
                else:
                    cbaxes = fig.add_axes([left + width / 8, bottom - 0.15, width/4*3, 0.03])
            else:
                cbaxes = fig.add_axes([right + 0.01, bottom, 0.015, height])
    else:
        cbaxes = fig.add_axes(
            [option["colorbar_left"], option["colorbar_bottom"], option["colorbar_width"], option["colorbar_height"]])


        cb = fig.colorbar(cs, cax=cbaxes, ticks=mticks, spacing='uniform', label='',
                          extend=option['extend'],
                          orientation=option['colorbar_position'])
        cb.solids.set_edgecolor("face")

        filename2 = file[:-4]
        plt.savefig(f'{filename2}_{type}.{option["saving_format"]}', format=f'{option["saving_format"]}', dpi=option['dpi'])
        plt.close()
