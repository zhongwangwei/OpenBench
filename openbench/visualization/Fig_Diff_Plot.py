import itertools
import sys
import math
import matplotlib
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import numpy as np
import pandas as pd
from matplotlib import rcParams
import logging
# Plot settings
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from Mod_Converttype import Convert_Type
from matplotlib import rcParams

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
import cmaps
from .Fig_toolbox import get_index, convert_unit, get_colormap, process_unit, tick_length

def plot_grid_map(basedir, filename, main_nml, metric, xitem, option):
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

    # Set the region of the map based on self.Max_lat, self.Min_lat, self.Max_lon, self.Min_lon
    ds = xr.open_dataset(f'{basedir}/{filename}')
    ds = Convert_Type.convert_nc(ds)

    # Extract variables
    ilat = ds.lat.values
    ilon = ds.lon.values
    lat, lon = np.meshgrid(ilat[::-1], ilon)

    var = ds[xitem].transpose("lon", "lat")[:, ::-1].values
    if not option["vmin_max_on"]:
        if metric in ['bias', 'percent_bias', 'rSD', 'PBIAS_HF', 'PBIAS_LF', 'NSE', 'KGE', 'KGESS', 'correlation', 'kappa_coeff',
                      'rSpearman']:
            quantiles = ds[xitem].quantile([0.05, 0.95], dim=['lat', 'lon'])
            max_value = math.ceil(quantiles[1].values)
            min_value = math.floor(quantiles[0].values)
            if metric == 'percent_bias':
                if max_value > 100:
                    max_value = 100
                if min_value < -100:
                    min_value = -100
        else:
            min_value, max_value = np.nanmin(var), np.nanmax(var)
        if min_value == max_value:
            max_value = max_value + 1

    cmap, mticks, norm, bnd, extend = get_index(min_value, max_value, option['cmap'])
    option['vmin'], option['vmax'] = mticks[0], mticks[-1]
    if min_value < option['vmin'] and max_value > option['vmax']:
        option['extend'] = 'both'
    elif min_value > option['vmin'] and max_value > option['vmax']:
        option['extend'] = 'max'
    elif min_value < option['vmin'] and max_value < option['vmax']:
        option['extend'] = 'min'
    else:
        option['extend'] = 'neither'

    fig = plt.figure(figsize=(option['x_wise'], option['y_wise']))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    extent = (ilon[0], ilon[-1], ilat[0], ilat[-1])

    if ilat[0] - ilat[-1] < 0:
        origin = 'lower'
    else:
        origin = 'upper'

    if option['show_method'] == 'interpolate':
        cs = ax.contourf(lon, lat, var, levels=bnd, cmap=cmap, norm=norm, extend=extend)
    else:
        cs = ax.imshow(ds[xitem].values, cmap=cmap, vmin=mticks[0], vmax=mticks[-1], extent=extent,
                       origin=origin)

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
    plt.title(option['title'], fontsize=option['title_size'], weight='bold')

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
                      extend=extend, orientation=option['colorbar_position'])
    cb.solids.set_edgecolor("face")

    filename2 = filename[:-3]
    plt.savefig(f'{basedir}/{filename2}.{option["saving_format"]}', format=f'{option["saving_format"]}', dpi=option['dpi'])
    plt.close()


def plot_stn_map(basedir, filename, stn_lon, stn_lat, metric, main_nml, var, varname, option):
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
    # Add check for empty or all-NaN array
    if len(metric) == 0 or np.all(np.isnan(metric)):
        print(f"Warning: No valid data for {varname}. Skipping plot.")
        return

    if not option["vmin_max_on"]:
        if var in ['bias', 'percent_bias', 'rSD', 'PBIAS_HF', 'PBIAS_LF', 'NSE', 'KGE', 'KGESS', 'correlation', 'kappa_coeff',
                   'rSpearman']:
            quantiles = [np.nanpercentile(metric, 5), np.nanpercentile(metric, 95)]
            max_value = math.ceil(quantiles[1])
            min_value = math.floor(quantiles[0])
            if var == 'percent_bias':
                if max_value > 100:
                    max_value = 100
                if min_value < -100:
                    min_value = -100
        else:
            min_value, max_value = np.nanmin(metric), np.nanmax(metric)
        if min_value == max_value:
            max_value = max_value + 1

    cmap, mticks, norm, bnd, extend = get_index(min_value, max_value, option['cmap'])
    option['vmin'], option['vmax'] = mticks[0], mticks[-1]


    fig = plt.figure(figsize=(option['x_wise'], option['y_wise']))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    cs = ax.scatter(stn_lon, stn_lat, s=option['markersize'], c=metric, cmap=cmap, norm=norm, vmin=option['vmin'], vmax=option['vmax'],
                    marker=option['marker'], linewidths=0.5, edgecolors='black', alpha=0.9)
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
    ax.set_ylabel(option['yticklabel'], fontsize=option['ytick'] + 1, labelpad=50)
    plt.title(option['title'], fontsize=option['title_size'], weight='bold')

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
    # cb.set_label('%s' % (varname), position=(0.5, 1.5), labelpad=-35)
    filename2 = filename[:-4]
    plt.savefig(f'{basedir}/{filename2}.{option["saving_format"]}', format=f'{option["saving_format"]}', dpi=option['dpi'])
    plt.close()


# Add plotting function for anomalies and differences
def plot_diff_results(basedir, data_type, item_type, evaluation_item, ref_source, sim_source, main_nml, sim_nml, ref_data_type,
                      option):
    """
    Plot anomalies or differences for metrics/scores
    data_type: 'anomaly' or 'difference'
    item_type: 'metric' or 'score'
    """
    plot_option = option
    if ref_data_type == 'stn':
        if data_type == 'anomaly':
            filename = f'{evaluation_item}_stn_{ref_source}_sim_{sim_source}_{item_type}_anomaly.csv'
        else:
            sim1, sim2 = sim_source
            sim_varname_1 = sim_nml[f'{evaluation_item}'][f'{sim1}_varname']
            sim_varname_2 = sim_nml[f'{evaluation_item}'][f'{sim2}_varname']
            filename = f'{evaluation_item}_stn_{ref_source}_{sim1}_{sim_varname_1}_vs_{sim2}_{sim_varname_2}_{item_type}_diff.csv'
    else:
        if data_type == 'anomaly':
            filename = f'{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{item_type}_anomaly.nc'
        else:
            sim1, sim2 = sim_source
            filename = f'{evaluation_item}_ref_{ref_source}_{sim1}_vs_{sim2}_{item_type}_diff.nc'

    # plot_option.update(option)
    # Set plot parameters based on data type
    if data_type == 'anomaly':
        plot_option['title'] = f'{evaluation_item} {item_type} anomaly for {sim_source}'
        # if not plot_option['colorbar_label']:
        unit = sim_nml[f'{evaluation_item}'][f'{sim_source}_varunit']
        plot_option['colorbar_label'] = process_unit(unit, '', item_type)
    else:
        plot_option['title'] = f'{evaluation_item} {item_type} difference {sim_source[0]} vs {sim_source[1]}'
        # if not plot_option['colorbar_label']:
        unit = sim_nml[f'{evaluation_item}'][f'{sim_source[0]}_varunit']
        plot_option['colorbar_label'] = process_unit(unit, '', item_type)

    if not plot_option['cmap']:
        plot_option['cmap'] = 'RdBu_r'  # Diverging colormap for anomalies/differences

    # For station data
    if ref_data_type == 'stn':
        data = pd.read_csv(f'{basedir}/{filename}', header=0)
        data = Convert_Type.convert_Frame(data)
        lon_select = data['lon'].values
        lat_select = data['lat'].values
        plotvar = data[f'{item_type}_{"anomaly" if data_type == "anomaly" else "diff"}'].values
        plot_stn_map(basedir, filename, lon_select, lat_select, plotvar, main_nml, item_type, f'{data_type}_{item_type}',
                     plot_option)

    # For gridded data
    else:  # xarray Dataset
        plot_grid_map(basedir, filename, main_nml, item_type,
                      f'{item_type}_{"anomaly" if data_type == "anomaly" else "diff"}', plot_option)


def make_scenarios_comparison_Diff_Plot(basedir, metrics, scores, evaluation_item, ref_source, sim_sources, main_nml, sim_nml,
                                        ref_data_type, option):
    for metric in metrics:
        for sim_source in sim_sources:
            # try:
            plot_diff_results(basedir, 'anomaly', metric, evaluation_item, ref_source, sim_source, main_nml, sim_nml,
                              ref_data_type, option)
        # except:
        #     logging.error(f'{evaluation_item}:{metric} - {ref_source} {sim_source} anomaly error')
        # After calculating differences for metrics
        if len(sim_sources) >= 2:
            for i, sim1 in enumerate(sim_sources):
                for j, sim2 in enumerate(sim_sources[i + 1:], i + 1):
                    try:
                        plot_diff_results(basedir, 'difference', metric, evaluation_item, ref_source,
                                          (sim1, sim2), main_nml, sim_nml, ref_data_type, option)
                    except:
                        logging.error(f'{evaluation_item}:{metric} - {ref_source} {sim1} vs {sim2} anomaly error')

    for score in scores:
        # After calculating anomalies for scores
        for sim_source in sim_sources:
            try:
                plot_diff_results(basedir, 'anomaly', score, evaluation_item, ref_source, sim_source, main_nml, sim_nml, ref_data_type
                                  , option)
            except:
                logging.error(f'{evaluation_item}:{score} - {ref_source} {sim_source} anomaly error')

        # After calculating differences for scores
        if len(sim_sources) >= 2:
            for i, sim1 in enumerate(sim_sources):
                for j, sim2 in enumerate(sim_sources[i + 1:], i + 1):
                    try:
                        plot_diff_results(basedir, 'difference', score, evaluation_item, ref_source, (sim1, sim2), main_nml, sim_nml,
                                          ref_data_type, option)
                    except:
                        logging.error(f'{evaluation_item}:{score} - {ref_source} {sim1} vs {sim2} anomaly error')
