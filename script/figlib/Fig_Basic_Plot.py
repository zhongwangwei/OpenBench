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
from Mod_Converttype import Convert_Type
from .Fig_toolbox import convert_unit
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import sys
cmaps_parent_path = os.path.abspath('./script/figlib/')
sys.path.append(cmaps_parent_path)
import cmaps
from Fig_toolbox import get_index, convert_unit, get_colormap, process_unit, tick_length


def make_plot_index_grid(self):
    key = self.ref_varname

    for metric in self.metrics:
        option = self.fig_nml['make_geo_plot_index']
        print(f'plotting metric: {metric}')
        unit = convert_unit(self.ref_varunit)
        option['colorbar_label'] = metric.replace('_', '\n') +'\n'+ process_unit( unit, self.sim_varunit, metric)
        # Set default extend option if not specified
        if 'extend' not in option:
            option['extend'] = 'both'  # Default value

        try:
            import math
            ds = xr.open_dataset(
                f'{self.casedir}/output/metrics/{self.item}_ref_{self.ref_source}_sim_{self.sim_source}_{metric}.nc')[metric]
            ds = Convert_Type.convert_nc(ds)
            quantiles = ds.quantile([0.05, 0.95], dim=['lat', 'lon'])
            del ds
            if not option["vmin_max_on"]:
                if metric in ['bias', 'percent_bias', 'rSD', 'PBIAS_HF', 'PBIAS_LF']:
                    option["vmax"] = math.ceil(quantiles[1].values)
                    option["vmin"] = math.floor(quantiles[0].values)
                    if metric == 'percent_bias':
                        if option["vmax"] > 100:
                            option["vmax"] = 100
                        if option["vmin"] < -100:
                            option["vmin"] = -100
                elif metric in [ 'NSE', 'KGE', 'KGESS', 'correlation', 'kappa_coeff', 'rSpearman']:
                    option["vmin"], option["vmax"] = -1, 1
                elif metric in ['LNSE', 'ubNSE', 'rNSE', 'wNSE', 'wsNSE']:
                    option["vmin"], option["vmax"] = math.floor(quantiles[1].values), 1
                elif metric in ['RMSE', 'CRMSD', 'MSE', 'ubRMSE', 'nRMSE', 'mean_absolute_error', 'ssq', 've',
                                'absolute_percent_bias']:
                    option["vmin"], option["vmax"] = 0, math.ceil(quantiles[1].values)
                else:
                    option["vmin"], option["vmax"] = 0, 1

            cmap, mticks, norm, bnd, extend = get_index(option['vmin'], option['vmax'], option['cmap'])
            plot_map_grid(self, cmap, norm, bnd, metric, 'metrics', mticks, option)
        except:
            print(f"ERROR: {key} {metric} ploting error, please check!")

    # print("\033[1;32m" + "=" * 80 + "\033[0m")
    for score in self.scores:
        option = self.fig_nml['make_geo_plot_index']
        print(f'plotting score: {score}')
        option['colorbar_label'] = score.replace('_', '\n')
        if not option["vmin_max_on"]:
            option["vmin"], option["vmax"] = 0, 1

        cmap, mticks, norm, bnd, extend = get_index(option['vmin'], option['vmax'], option['cmap'])
        plot_map_grid(self, cmap, norm, bnd, score, 'scores', mticks, option)
    print("\033[1;32m" + "=" * 80 + "\033[0m")

def plot_map_grid(self, colormap, normalize, levels, xitem, k, mticks, option):
    # Plot settings
    import numpy as np
    import xarray as xr
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    from matplotlib import rcParams

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
    ds = xr.open_dataset(f'{self.casedir}/output/{k}/{self.item}_ref_{self.ref_source}_sim_{self.sim_source}_{xitem}.nc')
    ds = Convert_Type.convert_nc(ds)

    # Extract variables
    ilat = ds.lat.values
    ilon = ds.lon.values
    lat, lon = np.meshgrid(ilat[::-1], ilon)

    var = ds[xitem].transpose("lon", "lat")[:, ::-1].values
    min_value, max_value = np.nanmin(var), np.nanmax(var)
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
        cs = ax.contourf(lon, lat, var, levels=levels, cmap=colormap, norm=normalize, extend=option['extend'])
    else:
        cs = ax.imshow(ds[xitem].values, cmap=colormap, vmin=option['vmin'], vmax=option['vmax'], extent=extent,
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
        ax.set_extent([self.min_lon, self.max_lon, self.min_lat, self.max_lat], crs=ccrs.PlateCarree())
        ax.set_xticks(np.arange(self.max_lon, self.min_lon, -60)[:0:-1], crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(self.max_lat, self.min_lat, -30)[:0:-1], crs=ccrs.PlateCarree())
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
    plt.title(option['title'], fontsize=option['title_size'])

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

    cb = fig.colorbar(cs, cax=cbaxes, ticks=mticks, spacing='uniform', label='',
                        extend=option['extend'],
                        orientation=option['colorbar_position'])
    cb.solids.set_edgecolor("face")

    plt.savefig(
        f'{self.casedir}/output/{k}/{self.item}_ref_{self.ref_source}_sim_{self.sim_source}_{xitem}.{option["saving_format"]}',
        format=f'{option["saving_format"]}', dpi=option['dpi'])
    plt.close()



def plot_stn(self, sim, obs, ID, key, RMSE, KGESS, correlation, lat_lon):
    option = self.fig_nml['plot_stn']
    from pylab import rcParams
    import matplotlib
    import matplotlib.pyplot as plt
    ### Plot settings

    # font = {'family': 'Times-Roman'}
    font = {'family': 'DejaVu Sans'}
    matplotlib.rc('font', **font)

    params = {'backend': 'ps',
                'axes.labelsize': option['labelsize'],
                'font.size': option['fontsize'],
                'legend.fontsize': option['fontsize'],
                'legend.frameon': False,
                'xtick.labelsize': option['xtick'],
                'xtick.direction': 'out',
                'ytick.labelsize': option['ytick'],
                'ytick.direction': 'out',
                'savefig.bbox': 'tight',
                'axes.unicode_minus': False,
                'text.usetex': False}
    rcParams.update(params)

    legs = ['Obs', 'Sim']
    lines = [option['obs_lineswidth'], option['sim_lineswidth']]
    alphas = [option['obs_alphas'], option['sim_alphas']]
    linestyles = [option['obs_linestyle'], option['sim_linestyle']]

    hex_pattern = r'^#([0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})$'
    import re
    if bool(re.match(hex_pattern, f"#{option['obs_linecolor']}")) and bool(
            re.match(hex_pattern, f"#{option['sim_linecolor']}")):
        colors = [f"#{option['obs_linecolor']}", f"#{option['sim_linecolor']}"]
    else:
        colors = [option['obs_linecolor'], option['sim_linecolor']]
    markers = [option['obs_marker'], option['sim_marker']]
    markersizes = [option['obs_markersize'], option['sim_markersize']]

    fig, ax = plt.subplots(1, 1, figsize=(option['x_wise'], option['y_wise']))
    max_time_len = max(len(sim), len(obs))

    obs.plot.line(x='time', label='Obs', linewidth=lines[0]/max_time_len, linestyle=linestyles[0], alpha=alphas[0], color=colors[0],
                    marker=markers[0], markersize=markersizes[0]/max_time_len)
    sim.plot.line(x='time', label='Sim', linewidth=lines[1]/max_time_len, linestyle=linestyles[1], alpha=alphas[1], color=colors[1],
                    marker=markers[1], markersize=markersizes[1]/max_time_len, add_legend=True)

    for spine in ax.spines.values():
        spine.set_linewidth(option['line_width'])

    # set ylabel to be the same as the variable name
    unit = convert_unit(self.ref_varunit)
    ax.set_ylabel(f"{key[0]} ({unit})", fontsize=option['ytick'] + 4, fontweight='bold')
    ax.set_xlabel('Date', fontsize=option['xtick'] + 4, fontweight='bold')
    # ax.tick_params(axis='both', top='off', labelsize=16)

    overall_label = f' RMSE: {RMSE:.2f} R: {correlation:.2f} KGESS: {KGESS:.2f} '
    # ax.scatter([], [], color='black', marker='o', label=overall_label)
    ax.legend(loc='best', shadow=False, labelspacing=option['labelspacing'] ,fontsize=option['fontsize'])
    # add RMSE,KGE,correlation in two digital to the legend in left top
    ax.text(0.6, 1.08, f'RMSE: {RMSE:.2f}   R: {correlation:.2f}   KGESS: {KGESS:.2f}', transform=ax.transAxes,
            fontsize=option['fontsize']-4, verticalalignment='top')
    if len(option['title']) == 0:
        lat = f"{abs(lat_lon[0]):.2f}°{'N' if lat_lon[0] > 0 else ('S' if lat_lon[0] < 0 else '')}"
        lon = f"{abs(lat_lon[1]):.2f}°{'E' if lat_lon[1] > 0 else ('W' if lat_lon[1] < 0 else '')}"
        option['title'] = f'ID: {str(ID).title()}  ({lat}, {lon})'
    ax.set_title(option['title'], fontsize=option['title_size'], fontweight='bold', x=0, y=1.08, 
        ha='left', va='top')
    if option['grid']:
        ax.grid(linestyle=option['grid_linestyle'], alpha=0.7, linewidth=option['grid_width'])

    # plt.tight_layout()
    plt.savefig(
        f'{self.casedir}/output/data/stn_{self.ref_source}_{self.sim_source}/{key[0]}_{ID}_timeseries.{option["saving_format"]}',
        format=f'{option["saving_format"]}', dpi=option['dpi'])
    plt.close(fig)

def plot_stn_map(self, stn_lon, stn_lat, metric, cmap, norm, varname, s_m, mticks, option):
    from pylab import rcParams
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import matplotlib
    import matplotlib.pyplot as plt
    ### Plot settings
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

    fig = plt.figure(figsize=(option['x_wise'], option['y_wise']))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    # set the region of the map based on self.Max_lat, self.Min_lat, self.Max_lon, self.Min_lon
    min_value, max_value = np.nanmin(metric), np.nanmax(metric)
    if min_value < option['vmin'] and max_value > option['vmax']:
        option['extend'] = 'both'
    elif min_value > option['vmin'] and max_value > option['vmax']:
        option['extend'] = 'max'
    elif min_value < option['vmin'] and max_value < option['vmax']:
        option['extend'] = 'min'
    else:
        option['extend'] = 'neither'

    cs = ax.scatter(stn_lon, stn_lat, s=option['markersize'], c=metric, cmap=cmap,norm=norm, marker=option['marker'],
                    linewidths=0.5, edgecolors='black', alpha=0.9, zorder=10)

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
        ax.set_extent([self.min_lon, self.max_lon, self.min_lat, self.max_lat], crs=ccrs.PlateCarree())
        ax.set_xticks(np.arange(self.max_lon, self.min_lon, -60)[:0:-1], crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(self.max_lat, self.min_lat, -30)[:0:-1], crs=ccrs.PlateCarree())
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

    cb = fig.colorbar(cs, cax=cbaxes, ticks=mticks, spacing='uniform', label='',
                        extend=option['extend'],
                        orientation=option['colorbar_position'])
    cb.solids.set_edgecolor("face")
    # cb.set_label('%s' % (varname), position=(0.5, 1.5), labelpad=-35)
    plt.savefig(
        f'{self.casedir}/output/{s_m}/{self.item}_stn_{self.ref_source}_{self.sim_source}_{varname}.{option["saving_format"]}',
        format=f'{option["saving_format"]}', dpi=option['dpi'])
    plt.close()

def make_plot_index_stn(self):

    # read the data
    df = pd.read_csv(f'{self.casedir}/output/scores/{self.item}_stn_{self.ref_source}_{self.sim_source}_evaluations.csv',
                        header=0)
    df = Convert_Type.convert_Frame(df)

    # loop the keys in self.variables to get the metric output
    for metric in self.metrics:
        option = self.fig_nml['make_stn_plot_index']
        if 'extend' not in self.fig_nml['make_geo_plot_index']:
            self.fig_nml['make_geo_plot_index']['extend'] = 'both'  # Default value
        option['extend'] = self.fig_nml['make_geo_plot_index']['extend']
        print(f'plotting metric: {metric}')
        unit = convert_unit(self.ref_varunit)
        option['colorbar_label'] = metric.replace('_', '\n')+'\n' + process_unit( unit, self.sim_varunit, metric)
        min_metric = -999.0
        max_metric = 100000.0
        # print(df['%s'%(metric)])
        ind0 = df[df['%s' % (metric)] > min_metric].index
        data_select0 = df.loc[ind0]
        # print(data_select0[data_select0['%s'%(metric)] < max_metric])
        ind1 = data_select0[data_select0['%s' % (metric)] < max_metric].index
        data_select = data_select0.loc[ind1]

        try:
            lon_select = data_select['ref_lon'].values
            lat_select = data_select['ref_lat'].values
        except:
            lon_select = data_select['sim_lon'].values
            lat_select = data_select['sim_lat'].values
        plotvar = data_select['%s' % (metric)].values

        try:
            import math
            vmin, vmax = np.percentile(plotvar, 5), np.percentile(plotvar, 95)
            if not option["vmin_max_on"]:
                if metric in ['bias', 'percent_bias', 'rSD', 'PBIAS_HF', 'PBIAS_LF']:
                    option["vmax"] = math.ceil(vmax)
                    option["vmin"] = math.floor(vmin)
                    if option["vmax"] > 100:
                        option["vmax"] = 100
                    if option["vmin"] < -100:
                        option["vmin"] = -100
                elif metric in ['NSE', 'KGE', 'KGESS', 'correlation', 'kappa_coeff', 'rSpearman']:
                    option["vmin"], option["vmax"] = -1, 1
                elif metric in ['LNSE', 'ubNSE', 'rNSE', 'wNSE', 'wsNSE']:
                    option["vmin"], option["vmax"] = math.floor(vmin), 1
                elif metric in ['RMSE', 'CRMSD', 'MSE', 'ubRMSE', 'nRMSE', 'mean_absolute_error', 'ssq', 've',
                                'absolute_percent_bias']:
                    option["vmin"], option["vmax"] = 0, math.ceil(vmax)
                else:
                    option["vmin"], option["vmax"] = 0, 1
        except:
            option["vmin"], option["vmax"] = 0, 1

        cmap, mticks, norm, bnd, extend = get_index(option['vmin'], option['vmax'], option['cmap'])
        plot_stn_map(self,lon_select, lat_select, plotvar, cmap, norm, metric, 'metrics', mticks, option)

    for score in self.scores:
        option = self.fig_nml['make_stn_plot_index']
        print(f'plotting score: {score}')
        option['colorbar_label'] = score.replace('_', '\n')
        min_score = -999.0
        max_score = 100000.0
        # print(df['%s'%(score)])
        ind0 = df[df['%s' % (score)] > min_score].index
        data_select0 = df.loc[ind0]
        # print(data_select0[data_select0['%s'%(score)] < max_score])
        ind1 = data_select0[data_select0['%s' % (score)] < max_score].index
        data_select = data_select0.loc[ind1]
        # if key=='discharge':
        #    #ind2 = data_select[abs(data_select['err']) < 0.001].index
        #    #data_select = data_select.loc[ind2]
        #    ind3 = data_select[abs(data_select['area1']) > 1000.].index
        #    data_select = data_select.loc[ind3]
        try:
            lon_select = data_select['ref_lon'].values
            lat_select = data_select['ref_lat'].values
        except:
            lon_select = data_select['sim_lon'].values
            lat_select = data_select['sim_lat'].values
        plotvar = data_select['%s' % (score)].values

        if not option["vmin_max_on"]:
            option["vmin"], option["vmax"] = 0, 1

        cmap, mticks, norm, bnd, extend = get_index(option['vmin'], option['vmax'], option['cmap'])

        plot_stn_map(self,lon_select, lat_select, plotvar, cmap, norm, score, 'scores', mticks, option)

def make_Basic(file, method_name, data_sources, main_nml,
                                 option):
    import numpy as np
    import xarray as xr
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    from matplotlib import rcParams
    # filename_parts = [method_name] + data_sources
    # filename = "_".join(filename_parts) + "_output"
    # file = os.path.join(output_dir, f"{method_name}", filename)

    ds = xr.open_dataset(file)
    ds = Convert_Type.convert_nc(ds)
    data = ds[method_name]
    ilat = ds.lat.values
    ilon = ds.lon.values
    lon, lat = np.meshgrid(ilon, ilat)

    min_value, max_value = np.nanmin(data), np.nanmax(data)
    cmap, mticks, norm, bnd, extend = get_index(in_value, max_value, option['cmap'])
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

    extent = (ilon[0], ilon[-1], ilat[0], ilat[-1])
    if ilat[0] - ilat[-1] < 0:
        origin = 'lower'
    else:
        origin = 'upper'

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

    if option['title'] is None:
        option['title'] = f'Correlation Results'
    ax.set_xlabel(option['xticklabel'], fontsize=option['xtick'] + 1, labelpad=20)
    ax.set_ylabel(option['yticklabel'], fontsize=option['ytick'] + 1, labelpad=40)
    plt.title(option['title'], fontsize=option['title_size'])

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

    cb = fig.colorbar(cs, cax=cbaxes, ticks=mticks, spacing='uniform', label='', extend=extend,
                      orientation=option['colorbar_position'])
    cb.solids.set_edgecolor("face")

    plt.savefig(f'{file}.{option["saving_format"]}', format=f'{option["saving_format"]}',
                dpi=option['dpi'])
    plt.close()

