import os
import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

# Check the platform
from Mod_Metrics import metrics
from Mod_Scores import scores

warnings.simplefilter(action='ignore', category=RuntimeWarning)
from matplotlib import colors
from matplotlib import cm


class Evaluation_grid(metrics, scores):
    def __init__(self, info, fig_nml):
        self.name = 'Evaluation_grid'
        self.version = '0.1'
        self.release = '0.1'
        self.date = 'Mar 2023'
        self.author = "Zhongwang Wei / zhongwang007@gmail.com"
        self.__dict__.update(info)
        self.fig_nml = fig_nml
        os.makedirs(self.casedir + '/output/', exist_ok=True)

        print(" ")
        print("\033[1;32m╔═══════════════════════════════════════════════════════════════╗\033[0m")
        print("\033[1;32m║                Evaluation processes starting!                 ║\033[0m")
        print("\033[1;32m╚═══════════════════════════════════════════════════════════════╝\033[0m")
        print("\n")

    def process_metric(self, metric, s, o, vkey=''):
        pb = getattr(self, metric)(s, o)
        pb = pb.squeeze()
        pb_da = xr.DataArray(pb, coords=[o.lat, o.lon], dims=['lat', 'lon'], name=metric)
        pb_da.to_netcdf(
            f'{self.casedir}/output/metrics/{self.item}_ref_{self.ref_source}_sim_{self.sim_source}_{metric}{vkey}.nc')

    def process_score(self, score, s, o, vkey=''):
        pb = getattr(self, score)(s, o)
        pb = pb.squeeze()
        pb_da = xr.DataArray(pb, coords=[o.lat, o.lon], dims=['lat', 'lon'], name=score)
        pb_da.to_netcdf(f'{self.casedir}/output/scores/{self.item}_ref_{self.ref_source}_sim_{self.sim_source}_{score}{vkey}.nc')

    def make_Evaluation(self, **kwargs):
        o = xr.open_dataset(f'{self.casedir}/output/data/{self.item}_ref_{self.ref_source}_{self.ref_varname}.nc')[
            f'{self.ref_varname}']
        s = xr.open_dataset(f'{self.casedir}/output/data/{self.item}_sim_{self.sim_source}_{self.sim_varname}.nc')[
            f'{self.sim_varname}']

        s['time'] = o['time']

        mask1 = np.isnan(s) | np.isnan(o)
        s.values[mask1] = np.nan
        o.values[mask1] = np.nan
        print("\033[1;32m" + "=" * 80 + "\033[0m")
        for metric in self.metrics:
            if hasattr(self, metric):
                print(f'calculating metric: {metric}')
                self.process_metric(metric, s, o)
            else:
                print('No such metric')
                sys.exit(1)

        for score in self.scores:
            if hasattr(self, score):
                print(f'calculating score: {score}')
                self.process_score(score, s, o)
            else:
                print('No such score')
                sys.exit(1)

        print("\033[1;32m" + "=" * 80 + "\033[0m")

        return

    def make_plot_index(self):
        option = self.fig_nml['make_geo_plot_index']
        key = self.ref_varname
        for metric in self.metrics:
            print(f'plotting metric: {metric}')
            option['colorbar_label'] = metric.replace('_', ' ')

            try:
                import math

                ds = xr.open_dataset(
                    f'{self.casedir}/output/metrics/{self.item}_ref_{self.ref_source}_sim_{self.sim_source}_{metric}.nc')[metric]
                quantiles = ds.quantile([0.05, 0.95], dim=['lat', 'lon'])
                if not option["vmin_max_on"]:
                    if metric in ['bias', 'percent_bias', 'rSD', 'PBIAS_HF', 'PBIAS_LF']:
                        option["vmax"] = math.ceil(quantiles[1].values)
                        option["vmin"] = math.floor(quantiles[0].values)
                        if metric == 'percent_bias':
                            if option["vmax"] > 500:
                                option["vmax"] = 500
                            if option["vmin"] < -500:
                                option["vmin"] = -500
                    elif metric in ['KGE', 'KGESS', 'correlation', 'kappa_coeff', 'rSpearman']:
                        option["vmin"], option["vmax"] = -1, 1
                    elif metric in ['NSE', 'LNSE', 'ubNSE', 'rNSE', 'wNSE', 'wsNSE']:
                        option["vmin"], option["vmax"] = math.floor(quantiles[1].values), 1
                    elif metric in ['RMSE', 'CRMSD', 'MSE', 'ubRMSE', 'nRMSE', 'mean_absolute_error', 'ssq', 've',
                                    'absolute_percent_bias']:
                        option["vmin"], option["vmax"] = 0, math.ceil(quantiles[1].values)
                    else:
                        option["vmin"], option["vmax"] = 0, 1

                if 2 >= option["vmax"] - option["vmin"] > 1:
                    option['colorbar_ticks'] = 0.2
                elif 10 >= option["vmax"] - option["vmin"] > 5:
                    option['colorbar_ticks'] = 1
                elif 100 >= option["vmax"] - option["vmin"] > 10:
                    option['colorbar_ticks'] = 5
                elif 200 >= option["vmax"] - option["vmin"] > 100:
                    option['colorbar_ticks'] = 20
                elif 500 >= option["vmax"] - option["vmin"] > 200:
                    option['colorbar_ticks'] = 50
                elif 1000 >= option["vmax"] - option["vmin"] > 500:
                    option['colorbar_ticks'] = 100
                elif 2000 >= option["vmax"] - option["vmin"] > 1000:
                    option['colorbar_ticks'] = 200
                elif 10000 >= option["vmax"] - option["vmin"] > 2000:
                    option['colorbar_ticks'] = 10 ** math.floor(math.log10(option["vmax"] - option["vmin"])) / 2
                else:
                    option['colorbar_ticks'] = 0.1

                ticks = matplotlib.ticker.MultipleLocator(base=option['colorbar_ticks'])
                mticks = ticks.tick_values(vmin=option['vmin'], vmax=option['vmax'])
                if mticks[0] < option['vmin']:
                    mticks = mticks[1:]
                if mticks[-1] > option['vmax']:
                    mticks = mticks[:-1]
                option['vmax'], option['vmin'] = mticks[-1], mticks[0]

                if option['cmap'] is not None:
                    cmap = cm.get_cmap(option['cmap'])
                    # bnd = np.linspace(mticks[0], mticks[-1], 11)
                    bnd = np.arange(mticks[0], mticks[-1] + option['colorbar_ticks'] / 2, option['colorbar_ticks'] / 2)
                    norm = colors.BoundaryNorm(bnd, cmap.N)
                else:
                    cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4',
                             '#313695']
                    cmap = colors.ListedColormap(cpool)
                    # bnd = np.linspace(mticks[0], mticks[-1], 11)
                    bnd = np.arange(mticks[0], mticks[-1] + option['colorbar_ticks'] / 2, option['colorbar_ticks'] / 2)
                    norm = colors.BoundaryNorm(bnd, cmap.N)

                self.plot_map(cmap, norm, bnd, metric, 'metrics', mticks, option)
            except:
                print(f"ERROR: {key} {metric} ploting error, please check!")

        # print("\033[1;32m" + "=" * 80 + "\033[0m")
        for score in self.scores:
            print(f'plotting score: {score}')
            option['colorbar_label'] = score.replace('_', ' ')
            if not option["vmin_max_on"]:
                option["vmin"], option["vmax"] = 0, 1
                option['extend'] = 'neither'

            if 2 >= option["vmax"] - option["vmin"] > 1:
                option['colorbar_ticks'] = 0.2
            elif 10 >= option["vmax"] - option["vmin"] > 5:
                option['colorbar_ticks'] = 1
            elif 100 >= option["vmax"] - option["vmin"] > 10:
                option['colorbar_ticks'] = 5
            elif 200 >= option["vmax"] - option["vmin"] > 100:
                option['colorbar_ticks'] = 20
            elif 500 >= option["vmax"] - option["vmin"] > 200:
                option['colorbar_ticks'] = 50
            elif 1000 >= option["vmax"] - option["vmin"] > 500:
                option['colorbar_ticks'] = 100
            elif 2000 >= option["vmax"] - option["vmin"] > 1000:
                option['colorbar_ticks'] = 200
            elif 10000 >= option["vmax"] - option["vmin"] > 2000:
                option['colorbar_ticks'] = 10 ** math.floor(math.log10(option["vmax"] - option["vmin"])) / 2
            else:
                option['colorbar_ticks'] = 0.1

            ticks = matplotlib.ticker.MultipleLocator(base=option['colorbar_ticks'])
            mticks = ticks.tick_values(vmin=option['vmin'], vmax=option['vmax'])
            if mticks[0] < option['vmin']:
                mticks = mticks[1:]
            if mticks[-1] > option['vmax']:
                mticks = mticks[:-1]
            option['vmax'], option['vmin'] = mticks[-1], mticks[0]


            if option['cmap'] is not None:
                cmap = cm.get_cmap(option['cmap'])
                # bnd = np.linspace(mticks[0], mticks[-1], 11)
                bnd = np.arange(mticks[0], mticks[-1] + option['colorbar_ticks'] / 2, option['colorbar_ticks'] / 2)
                norm = colors.BoundaryNorm(bnd, cmap.N)
            else:
                cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4',
                         '#313695']
                cmap = colors.ListedColormap(cpool)
                # bnd = np.linspace(mticks[0], mticks[-1], 11)
                bnd = np.arange(mticks[0], mticks[-1] + option['colorbar_ticks'] / 2, option['colorbar_ticks'] / 2)
                norm = colors.BoundaryNorm(bnd, cmap.N)

            self.plot_map(cmap, norm, bnd, score, 'scores', mticks, option)
        print("\033[1;32m" + "=" * 80 + "\033[0m")

    def plot_map(self, colormap, normalize, levels, xitem, k, mticks, option):
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

        # Extract variables
        lat = ds.lat.values
        lon = ds.lon.values
        lat, lon = np.meshgrid(lat[::-1], lon)

        var = ds[xitem].transpose("lon", "lat")[:, ::-1].values

        fig = plt.figure(figsize=(option['x_wise'], option['y_wise']))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        cs = ax.contourf(lon, lat, var, levels=levels, cmap=colormap, norm=normalize, extend=option['extend'])
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
            ax.set_extent([self.min_lon, self.max_lon, self.min_lat, self.max_lat])
            ax.set_xticks(np.arange(self.max_lon, self.min_lon, -60)[::-1], crs=ccrs.PlateCarree())
            ax.set_yticks(np.arange(self.max_lat, self.min_lat, -30)[::-1], crs=ccrs.PlateCarree())
        else:
            ax.set_extent([option['min_lon'], option['max_lon'], option['min_lat'], option['max_lat']])
            ax.set_xticks(np.arange(option['max_lon'], option['min_lon'], -60)[::-1], crs=ccrs.PlateCarree())
            ax.set_yticks(np.arange(option['max_lat'], option['min_lat'], -30)[::-1], crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter()
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)

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

        plt.savefig(
            f'{self.casedir}/output/{k}/{self.item}_ref_{self.ref_source}_sim_{self.sim_source}_{xitem}.{option["saving_format"]}',
            format=f'{option["saving_format"]}', dpi=option['dpi'])
        plt.close()


class Evaluation_stn(metrics, scores):
    def __init__(self, info, fig_nml):
        self.name = 'Evaluation_point'
        self.version = '0.1'
        self.release = '0.1'
        self.date = 'Mar 2023'
        self.author = "Zhongwang Wei / zhongwang007@gmail.com"
        self.fig_nml = fig_nml
        self.__dict__.update(info)
        if isinstance(self.sim_varname, str): self.sim_varname = [self.sim_varname]
        if isinstance(self.ref_varname, str): self.ref_varname = [self.ref_varname]

        print('Evaluation processes starting!')
        print("=======================================")
        print(" ")
        print(" ")

    def make_evaluation(self):
        # read station information
        stnlist = f"{self.casedir}/stn_list.txt"
        station_list = pd.read_csv(stnlist, header=0)

        # loop the keys in self.variables to get the metric output
        for metric in self.metrics:
            station_list[f'{metric}'] = [-9999.0] * len(station_list['ID'])
        for score in self.scores:
            station_list[f'{score}'] = [-9999.0] * len(station_list['ID'])
        for iik in range(len(station_list['ID'])):
            s = xr.open_dataset(
                f"{self.casedir}/output/data/stn_{self.ref_source}_{self.sim_source}/sim_{station_list['ID'][iik]}" + f"_{station_list['use_syear'][iik]}" + f"_{station_list['use_eyear'][iik]}.nc")[
                self.sim_varname]
            o = xr.open_dataset(
                f"{self.casedir}/output/data/stn_{self.ref_source}_{self.sim_source}/ref_{station_list['ID'][iik]}" + f"_{station_list['use_syear'][iik]}" + f"_{station_list['use_eyear'][iik]}.nc")[
                self.ref_varname]
            s['time'] = o['time']
            mask1 = np.isnan(s) | np.isnan(o)
            s.values[mask1] = np.nan
            o.values[mask1] = np.nan

            for metric in self.metrics:
                if hasattr(self, metric):
                    pb = getattr(self, metric)(s, o)
                    station_list.loc[iik, f'{metric}'] = pb.values
                #  self.plot_stn(s.squeeze(),o.squeeze(),station_list['ID'][iik],self.ref_varname, float(station_list['RMSE'][iik]), float(station_list['KGE'][iik]),float(station_list['correlation'][iik]))
                else:
                    print('No such metric')
                    sys.exit(1)

            for score in self.scores:
                if hasattr(self, score):
                    pb = getattr(self, score)(s, o)

                else:
                    print('No such score')
                    sys.exit(1)

            print("=======================================")
            print(" ")
            print(" ")

        print('Comparison dataset prepared!')
        print("=======================================")
        print(" ")
        print(" ")
        print(f"send {self.ref_varname} evaluation to {self.ref_varname}_{self.sim_varname}_metric.csv'")
        station_list.to_csv(
            f'{self.casedir}/output/metrics/stn_{self.ref_source}_{self.sim_source}/{self.ref_varname}_{self.sim_varname}_metric.csv',
            index=False)
        station_list.to_csv(
            f'{self.casedir}/output/scores/stn_{self.ref_source}_{self.sim_source}/{self.ref_varname}_{self.sim_varname}_scores.csv',
            index=False)

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

        obs.plot.line(x='time', label='Obs', linewidth=lines[0], linestyle=linestyles[0], alpha=alphas[0], color=colors[0],
                      marker=markers[0], markersize=markersizes[0])
        sim.plot.line(x='time', label='Sim', linewidth=lines[0], linestyle=linestyles[1], alpha=alphas[1], color=colors[1],
                      marker=markers[1], markersize=markersizes[1], add_legend=True)
        # set ylabel to be the same as the variable name
        ax.set_ylabel(f"{key[0]} [{self.ref_varunit}]", fontsize=option['ytick'] + 1)
        ax.set_xlabel('Date', fontsize=option['xtick'] + 1)
        # ax.tick_params(axis='both', top='off', labelsize=16)

        overall_label = f' RMSE: {RMSE:.2f}\n R: {correlation:.2f}\n KGESS: {KGESS:.2f} '
        ax.scatter([], [], color='black', marker='o', label=overall_label)
        ax.legend(loc='best', shadow=False, fontsize=option['fontsize'])
        # add RMSE,KGE,correlation in two digital to the legend in left top
        # ax.text(0.01, 0.95, f'RMSE: {RMSE:.2f}\n R: {correlation:.2f}\n KGESS: {KGESS:.2f} ', transform=ax.transAxes,
        #         fontsize=option['fontsize'], verticalalignment='top')
        if len(option['title']) == 0:
            option['title'] = f'ID: {str(ID).title()},  Lat: {lat_lon[0]:.2f},  Lon:{lat_lon[1]:.2f}'
        ax.set_title(option['title'], fontsize=option['title_size'])
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
        fig = plt.figure(figsize=(option['x_wise'], option['y_wise']))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        # set the region of the map based on self.Max_lat, self.Min_lat, self.Max_lon, self.Min_lon
        cs = ax.scatter(stn_lon, stn_lat, s=option['markersize'], c=metric, cmap=cmap, norm=norm, marker=option['marker'],
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
            ax.set_extent([self.min_lon, self.max_lon, self.min_lat, self.max_lat])
            ax.set_xticks(np.arange(self.max_lon, self.min_lon, -60)[::-1], crs=ccrs.PlateCarree())
            ax.set_yticks(np.arange(self.max_lat, self.min_lat, -30)[::-1], crs=ccrs.PlateCarree())
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
        # cb.set_label('%s' % (varname), position=(0.5, 1.5), labelpad=-35)
        plt.savefig(
            f'{self.casedir}/output/{s_m}/{self.item}_stn_{self.ref_source}_{self.sim_source}_{varname}.{option["saving_format"]}',
            format=f'{option["saving_format"]}', dpi=option['dpi'])
        plt.close()

    def make_plot_index(self):
        option = self.fig_nml['make_stn_plot_index']
        # read the data
        df = pd.read_csv(f'{self.casedir}/output/scores/{self.item}_stn_{self.ref_source}_{self.sim_source}_evaluations.csv',
                         header=0)
        # loop the keys in self.variables to get the metric output
        for metric in self.metrics:
            print(f'plotting metric: {metric}')
            option['colorbar_label'] = metric.replace('_', ' ')
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
                    elif metric in ['KGE', 'KGESS', 'correlation', 'kappa_coeff', 'rSpearman']:
                        option["vmin"], option["vmax"] = -1, 1
                    elif metric in ['NSE', 'LNSE', 'ubNSE', 'rNSE', 'wNSE', 'wsNSE']:
                        option["vmin"], option["vmax"] = math.floor(vmin), 1
                    elif metric in ['RMSE', 'CRMSD', 'MSE', 'ubRMSE', 'nRMSE', 'mean_absolute_error', 'ssq', 've',
                                    'absolute_percent_bias']:
                        option["vmin"], option["vmax"] = 0, math.ceil(vmax)
                    else:
                        option["vmin"], option["vmax"] = 0, 1
            except:
                option["vmin"], option["vmax"] = 0, 1

            if 2 >= option["vmax"] - option["vmin"] > 1:
                option['colorbar_ticks'] = 0.2
            elif 10 >= option["vmax"] - option["vmin"] > 5:
                option['colorbar_ticks'] = 1
            elif 100 >= option["vmax"] - option["vmin"] > 10:
                option['colorbar_ticks'] = 5
            elif 200 >= option["vmax"] - option["vmin"] > 100:
                option['colorbar_ticks'] = 20
            elif 500 >= option["vmax"] - option["vmin"] > 200:
                option['colorbar_ticks'] = 50
            elif 1000 >= option["vmax"] - option["vmin"] > 500:
                option['colorbar_ticks'] = 100
            elif 2000 >= option["vmax"] - option["vmin"] > 1000:
                option['colorbar_ticks'] = 200
            elif 10000 >= option["vmax"] - option["vmin"] > 2000:
                option['colorbar_ticks'] = 10 ** math.floor(math.log10(option["vmax"] - option["vmin"])) / 2
            else:
                option['colorbar_ticks'] = 0.1

            ticks = matplotlib.ticker.MultipleLocator(base=option['colorbar_ticks'])
            mticks = ticks.tick_values(vmin=option['vmin'], vmax=option['vmax'])
            if mticks[0] < option['vmin']:
                mticks = mticks[1:]
            if mticks[-1] > option['vmax']:
                mticks = mticks[:-1]

            if option['cmap'] is not None:
                cmap = cm.get_cmap(option['cmap'])
                # bnd = np.linspace(mticks[0], mticks[-1], 11)
                bnd = np.arange(mticks[0], mticks[-1] + option['colorbar_ticks'] / 2, option['colorbar_ticks'] / 2)
                norm = colors.BoundaryNorm(bnd, cmap.N)
            else:
                cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4',
                         '#313695']
                cmap = colors.ListedColormap(cpool)
                # bnd = np.linspace(mticks[0], mticks[-1], 11)
                bnd = np.arange(mticks[0], mticks[-1] + option['colorbar_ticks'] / 2, option['colorbar_ticks'] / 2)
                norm = colors.BoundaryNorm(bnd, cmap.N)
            self.plot_stn_map(lon_select, lat_select, plotvar, cmap, norm, metric, 'metrics', mticks, option)

        for score in self.scores:
            print(f'plotting score: {score}')
            option['colorbar_label'] = score.replace('_', ' ')
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
                option['extend'] = 'neither'

            if 2 >= option["vmax"] - option["vmin"] > 1:
                option['colorbar_ticks'] = 0.2
            elif 10 >= option["vmax"] - option["vmin"] > 5:
                option['colorbar_ticks'] = 1
            elif 100 >= option["vmax"] - option["vmin"] > 10:
                option['colorbar_ticks'] = 5
            elif 200 >= option["vmax"] - option["vmin"] > 100:
                option['colorbar_ticks'] = 20
            elif 500 >= option["vmax"] - option["vmin"] > 200:
                option['colorbar_ticks'] = 50
            elif 1000 >= option["vmax"] - option["vmin"] > 500:
                option['colorbar_ticks'] = 100
            elif 2000 >= option["vmax"] - option["vmin"] > 1000:
                option['colorbar_ticks'] = 200
            elif 10000 >= option["vmax"] - option["vmin"] > 2000:
                option['colorbar_ticks'] = 10 ** math.floor(math.log10(option["vmax"] - option["vmin"])) / 2
            else:
                option['colorbar_ticks'] = 0.1

            ticks = matplotlib.ticker.MultipleLocator(base=option['colorbar_ticks'])
            mticks = ticks.tick_values(vmin=option['vmin'], vmax=option['vmax'])
            if mticks[0] < option['vmin']:
                mticks = mticks[1:]
            if mticks[-1] > option['vmax']:
                mticks = mticks[:-1]

            if option['cmap'] is not None:
                cmap = cm.get_cmap(option['cmap'])
                # bnd = np.linspace(mticks[0], mticks[-1], 11)
                bnd = np.arange(mticks[0], mticks[-1] + option['colorbar_ticks'] / 2, option['colorbar_ticks'] / 2)
                norm = colors.BoundaryNorm(bnd, cmap.N)
            else:
                cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4',
                         '#313695']
                cmap = colors.ListedColormap(cpool)
                # bnd = np.linspace(mticks[0], mticks[-1], 11)
                bnd = np.arange(mticks[0], mticks[-1] + option['colorbar_ticks'] / 2, option['colorbar_ticks'] / 2)
                norm = colors.BoundaryNorm(bnd, cmap.N)

            self.plot_stn_map(lon_select, lat_select, plotvar, cmap, norm, score, 'scores', mticks, option)

    def make_evaluation_parallel(self, station_list, iik):
        s = xr.open_dataset(
            f"{self.casedir}/output/data/stn_{self.ref_source}_{self.sim_source}/{self.item}_sim_{station_list['ID'][iik]}" + f"_{station_list['use_syear'][iik]}" + f"_{station_list['use_eyear'][iik]}.nc")[
            self.sim_varname].to_array().squeeze()
        o = xr.open_dataset(
            f"{self.casedir}/output/data/stn_{self.ref_source}_{self.sim_source}/{self.item}_ref_{station_list['ID'][iik]}" + f"_{station_list['use_syear'][iik]}" + f"_{station_list['use_eyear'][iik]}.nc")[
            self.ref_varname].to_array().squeeze()

        s['time'] = o['time']
        mask1 = np.isnan(s) | np.isnan(o)
        s.values[mask1] = np.nan
        o.values[mask1] = np.nan
        # remove the nan values
        # s=s.dropna(dim='time').astype(np.float32)
        # o=o.dropna(dim='time').astype(np.float32)
        row = {}
        # for based plot
        try:
            row['KGESS'] = self.KGESS(s, o).values
        except:
            row['KGESS'] = -9999.0
        try:
            row['RMSE'] = self.rmse(s, o).values
        except:
            row['RMSE'] = -9999.0
        try:
            row['correlation'] = self.correlation(s, o).values
        except:
            row['correlation'] = -9999.0

        for metric in self.metrics:
            if hasattr(self, metric):
                pb = getattr(self, metric)(s, o)
                if pb.values is not None:
                    row[f'{metric}'] = pb.values
                else:
                    row[f'{metric}'] = -9999.0
                    if 'ref_lat' in station_list:
                        lat_lon = [station_list['ref_lat'][iik], station_list['ref_lon'][iik]]
                    else:
                        lat_lon = [station_list['sim_lat'][iik], station_list['sim_lon'][iik]]
                    self.plot_stn(s.squeeze(), o.squeeze(), station_list['ID'][iik], self.ref_varname,
                                  float(station_list['RMSE'][iik]), float(station_list['KGE'][iik]),
                                  float(station_list['correlation'][iik]), lat_lon)
            else:
                print(f'No such metric: {metric}')
                sys.exit(1)

        for score in self.scores:
            if hasattr(self, score):
                pb2 = getattr(self, score)(s, o)
                # if pb2.values is not None:
                if pb2.values is not None:
                    row[f'{score}'] = pb2.values
                else:
                    row[f'{score}'] = -9999.0
            else:
                print('No such score')
                sys.exit(1)

        if 'ref_lat' in station_list:
            lat_lon = [station_list['ref_lat'][iik], station_list['ref_lon'][iik]]
        else:
            lat_lon = [station_list['sim_lat'][iik], station_list['sim_lon'][iik]]
        self.plot_stn(s, o, station_list['ID'][iik], self.ref_varname, float(row['RMSE']), float(row['KGESS']),
                      float(row['correlation']), lat_lon)
        return row
        # return station_list

    def make_evaluation_P(self):
        stnlist = f"{self.casedir}/stn_list.txt"
        station_list = pd.read_csv(stnlist, header=0)
        num_cores = os.cpu_count()  ##用来计算现在可以获得多少cpu核心。 也可以用multipocessing.cpu_count(),或者随意设定<=cpu核心数的数值
        # shutil.rmtree(f'{self.casedir}/output',ignore_errors=True)
        # creat tmp directory
        # os.makedirs(f'{self.casedir}/output', exist_ok=True)

        # loop the keys in self.variables
        # loop the keys in self.variables to get the metric output
        # for metric in self.metrics.keys():
        #    station_list[f'{metric}']=[-9999.0] * len(station_list['ID'])
        # if self.ref_source.lower() == 'grdc':

        results = Parallel(n_jobs=-1)(
            delayed(self.make_evaluation_parallel)(station_list, iik) for iik in range(len(station_list['ID'])))
        station_list = pd.concat([station_list, pd.DataFrame(results)], axis=1)

        print('Evaluation finished')
        print("=======================================")
        print(" ")
        print(" ")
        print(f"send evaluation to {self.casedir}/output/scores/{self.item}_stn_{self.ref_source}_{self.sim_source}_evaluations")
        print(f"send evaluation to {self.casedir}/output/metrics/{self.item}_stn_{self.ref_source}_{self.sim_source}_evaluations")

        station_list.to_csv(f'{self.casedir}/output/scores/{self.item}_stn_{self.ref_source}_{self.sim_source}_evaluations.csv',
                            index=False)
        station_list.to_csv(f'{self.casedir}/output/metrics/{self.item}_stn_{self.ref_source}_{self.sim_source}_evaluations.csv',
                            index=False)
