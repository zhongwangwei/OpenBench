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
from Mod_Converttype import Convert_Type

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
import cmaps
from .Fig_toolbox import get_index, convert_unit, get_colormap, tick_length

def map(file, method_name, data_sources, ilon, ilat, data, title, p_value, significant, main_nml, option):
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

    cmap, mticks, norm, bnd, extend = get_index(option['vmin'], option['vmax'], option['cmap'])

    extent = (ilon[0], ilon[-1], ilat[0], ilat[-1])

    if ilat[0] - ilat[-1] < 0:
        origin = 'lower'
    else:
        origin = 'upper'
    lon, lat = np.meshgrid(ilon, ilat)

    if option['show_method'] == 'interpolate':
        cs = ax.contourf(lon, lat, data, levels=bnd, cmap=cmap, norm=norm, extend=extend)
    else:
        cs = ax.imshow(data, cmap=cmap, vmin=mticks[0], vmax=mticks[-1], extent=extent, origin=origin)

    for spine in ax.spines.values():
        spine.set_linewidth(option['line_width'])

    coastline = cfeature.NaturalEarthFeature(
        'physical', 'coastline', '110m', edgecolor='0.6', facecolor='none')
    rivers = cfeature.NaturalEarthFeature(
        'physical', 'rivers_lake_centerlines', '110m', edgecolor='0.6', facecolor='none')
    ax.add_feature(cfeature.LAND, facecolor='0.9')
    ax.add_feature(coastline, linewidth=0.6)
    ax.add_feature(cfeature.LAKES, alpha=1, facecolor='white', edgecolor='white')
    ax.add_feature(rivers, linewidth=0.5)
    ax.gridlines(draw_labels=False, linestyle=':', linewidth=0.5, color='grey', alpha=0.8,
        xlocs=np.arange(option['max_lon'], option['min_lon'], -60)[:0:-1],
        ylocs=np.arange(option['max_lat'], option['min_lat'], -30)[:0:-1])

    p = ax.contourf(lon, lat, p_value, levels=[0, significant, 1],
                    hatches=['.....', None], colors="none", add_colorbar=False,
                    zorder=3)

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

    if not option['title'] :
        option['title'] = f'Mann-Kendall Test Results ({title}) on significant level: {significant:.3f}'
    ax.set_xlabel(option['xticklabel'], fontsize=option['xtick'] + 1, labelpad=20)
    ax.set_ylabel(option['yticklabel'], fontsize=option['ytick'] + 1, labelpad=40)
    plt.title(option['title'], fontsize=option['title_size'], weight='bold', loc='left')

    if not option['colorbar_position_set']:
        pos = ax.get_position()
        left, right, bottom, width, height = pos.x0, pos.x1, pos.y0, pos.width, pos.height
        if (option['min_lat']<-60) & (option['max_lat']>89) & (option['min_lon']<-179) & (option['max_lon']>179):
            if option['colorbar_position'] == 'horizontal':
                cbaxes = fig.add_axes([left + 0.03, bottom+ 0.14, 0.15, 0.02])
                ax.text(-130,-40,option['colorbar_label'],fontsize=16, weight='bold', ha='center' ,va='bottom')
            else:
                cbaxes = fig.add_axes([left + 0.015, bottom + 0.08, 0.02, height/3])
                ax.text(-160+7*tick_length(np.median(mticks)),-40,option['colorbar_label'],fontsize=16, weight='bold', ha='left' ,va='center')                
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

    cb = fig.colorbar(cs, cax=cbaxes, ticks=mticks, spacing='uniform', label=option['colorbar_label'], extend=extend,
                      orientation=option['colorbar_position'])
    cb.solids.set_edgecolor("face")

    file2 = file[:-3]
    plt.savefig(f'{file2}_{title}.{option["saving_format"]}', format=f'{option["saving_format"]}',
                dpi=option['dpi'])
    plt.close()


def make_Mann_Kendall_Trend_Test(file, method_name, data_sources, main_nml, option):
    ds = xr.open_dataset(f"{file}")
    ds = Convert_Type.convert_nc(ds)
    trend = ds.trend
    if trend.ndim == 3 and trend.shape[0] == 1:
        trend = trend.squeeze(axis=0)
    tau = ds.tau
    if tau.ndim == 3 and tau.shape[0] == 1:
        tau = tau.squeeze(axis=0)
    # significant = ds.significance
    p_value = ds.p_value
    if p_value.ndim == 3 and p_value.shape[0] == 1:
        p_value = p_value.squeeze(axis=0)

    significant = option['significance_level']

    ilat = ds.lat.values
    ilon = ds.lon.values

    if not option['extend']:
        option['extend'] = 'both'
    try:
        option['vmin'], option['vmax'] = math.floor(tau.min(skipna=True).values), math.ceil(tau.max(skipna=True).values)
    except:
        option['vmin'], option['vmax'] = -1, 1
    map(file, method_name, data_sources, ilon, ilat, tau, 'tau', p_value, significant, main_nml, option)

    option['vmin'], option['vmax'] = -1, 1
    map(file, method_name, data_sources, ilon, ilat, trend, 'Trend', p_value, significant, main_nml, option)
