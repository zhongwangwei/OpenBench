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
warnings.simplefilter(action='ignore', category=RuntimeWarning)


def process_unit_grid(self,ref_unit,sim_unit,metric):
    all_metrics_units = {
        'percent_bias': '%',  # Percent Bias
        'absolute_percent_bias': '%',  # Absolute Percent Bias
        'bias': 'Same as input data',  # Bias
        'mean_absolute_error': 'Same as input data',  # Mean Absolute Error
        'RMSE': 'Same as input data',  # Root Mean Squared Error
        'MSE': 'Square of input data unit',  # Mean Squared Error
        'ubRMSE': 'Same as input data',  # Unbiased Root Mean Squared Error
        'CRMSD': 'Same as input data',  # Centered Root Mean Square Difference
        'nrmse': 'Unitless',  # Normalized Root Mean Square Error
        'L': 'Unitless',  # Likelihood
        'correlation': 'Unitless',  # correlation coefficient
        'correlation_R2': 'Unitless',  # correlation coefficient R2
        'NSE': 'Unitless',  # Nash Sutcliffe efficiency coefficient
        'LNSE': 'Unitless',  # natural logarithm of NSE coefficient
        'KGE': 'Unitless',  # Kling-Gupta Efficiency
        'KGESS': 'Unitless',  # Normalized Kling-Gupta Efficiency
        'kappa_coeff': 'Unitless',  # Kappa coefficient
        'rv': 'Unitless',  # Relative variability (amplitude ratio)
        'ubNSE': 'Unitless',  # Unbiased Nash Sutcliffe efficiency coefficient
        'ubKGE': 'Unitless',  # Unbiased Kling-Gupta Efficiency
        'ubcorrelation': 'Unitless',  # Unbiased correlation
        'ubcorrelation_R2': 'Unitless',  # correlation coefficient R2
        'pc_max': '%',  # the bias of the maximum value
        'pc_min': '%',  # the bias of the minimum value
        'pc_ampli': '%',  # the bias of the amplitude value
        'rSD': 'Unitless',  # Ratio of standard deviations
        'PBIAS_HF': '%',  # Percent bias of flows ≥ Q98 (Yilmaz et al., 2008)
        'PBIAS_LF': '%',  # Percent bias of flows ≤ Q30(Yilmaz et al., 2008)
        'SMPI': 'Unitless',  # https://docs.esmvaltool.org/en/latest/recipes/recipe_smpi.html
        'ggof': 'Unitless',  # Graphical Goodness of Fit
        'gof': 'Unitless',  # Numerical Goodness-of-fit measures
        'KGEkm': 'Unitless',  # Kling-Gupta Efficiency with knowable-moments
        'KGElf': 'Unitless',  # Kling-Gupta Efficiency for low values
        'KGEnp': 'Unitless',  # Non-parametric version of the Kling-Gupta Efficiency
        'md': 'Unitless',  # Modified Index of Agreement
        'mNSE': 'Unitless',  # Modified Nash-Sutcliffe efficiency
        'pbiasfdc': '%',  # Percent Bias in the Slope of the Midsegment of the Flow Duration Curve
        'pfactor': '%',  # the percent of observations that are within the given uncertainty bounds.
        'rd': 'Unitless',  # Relative Index of Agreement
        'rfactor': 'Unitless',
        # the average width of the given uncertainty bounds divided by the standard deviation of the observations.
        'rNSE': 'Unitless',  # Relative Nash-Sutcliffe efficiency
        'rSpearman': 'Unitless',  # Spearman's rank correlation coefficient
        'rsr': 'Unitless',  # Ratio of RMSE to the standard deviation of the observations
        'sKGE': 'Unitless',  # Split Kling-Gupta Efficiency
        'ssq': 'Square of input data unit',  # Sum of the Squared Residuals
        'valindex': 'Unitless',  # Valid Indexes
        've': 'Unitless',  # Volumetric Efficiency
        'wNSE': 'Unitless',  # Weighted Nash-Sutcliffe efficiency
        'wsNSE': 'Unitless',  # Weighted seasonal Nash-Sutcliffe Efficiency
        'index_agreement': 'Unitless',  # Index of agreement
    }

    unit = all_metrics_units[metric]
    if unit == 'Unitless':
        return '[Unitless]'
    elif unit == '%':
        return '[%]'
    elif unit == 'Same as input data':
        return f'[{ref_unit}]'
    elif unit == 'Square of input data unit':
        return rf'[${ref_unit}^{{2}}$]'
    else:
        print('Warning: Missing metric unit!')
        return '[None]'

def make_plot_index_grid(self):
    key = self.ref_varname

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
    for metric in self.metrics:
        option = self.fig_nml['make_geo_plot_index']
        print(f'plotting metric: {metric}')
        option['colorbar_label'] = metric.replace('_', ' ') +' '+ process_unit_grid(self, self.ref_varunit,self.sim_varunit, metric)
        # Set default extend option if not specified
        if 'extend' not in option:
            option['extend'] = 'both'  # Default value

        try:
            import math
            ds = xr.open_dataset(
                f'{self.casedir}/output/metrics/{self.item}_ref_{self.ref_source}_sim_{self.sim_source}_{metric}.nc')[metric]
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

            option['colorbar_ticks'] = get_ticks(option["vmin"], option["vmax"])

            ticks = matplotlib.ticker.MultipleLocator(base=option['colorbar_ticks'])
            mticks = ticks.tick_values(vmin=option['vmin'], vmax=option['vmax'])
            mticks = [round(tick, 2) if isinstance(tick, float) and len(str(tick).split('.')[1]) > 2 else tick for tick in
                        mticks]
            if mticks[0] < option['vmin'] and mticks[-1] < option['vmax']:
                mticks = mticks[1:]
            elif mticks[0] > option['vmin'] and mticks[-1] > option['vmax']:
                mticks = mticks[:-1]
            elif mticks[0] < option['vmin'] and mticks[-1] > option['vmax']:
                mticks = mticks[1:-1]
            option['vmax'], option['vmin'] = mticks[-1], mticks[0]

            if option['cmap'] is not None:
                cmap = cm.get_cmap(option['cmap'])
                bnd = np.arange(option['vmin'], option['vmax'] + option['colorbar_ticks'] / 2, option['colorbar_ticks'] / 2)
                norm = colors.BoundaryNorm(bnd, cmap.N)
            else:
                cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4',
                            '#313695']
                cmap = colors.ListedColormap(cpool)
                bnd = np.arange(option['vmin'], option['vmax'] + option['colorbar_ticks'] / 2, option['colorbar_ticks'] / 2)
                norm = colors.BoundaryNorm(bnd, cmap.N)
            plot_map_grid(self, cmap, norm, bnd, metric, 'metrics', mticks, option)
        except:
            print(f"ERROR: {key} {metric} ploting error, please check!")

    # print("\033[1;32m" + "=" * 80 + "\033[0m")
    for score in self.scores:
        option = self.fig_nml['make_geo_plot_index']
        print(f'plotting score: {score}')
        option['colorbar_label'] = score.replace('_', ' ')
        if not option["vmin_max_on"]:
            option["vmin"], option["vmax"] = 0, 1

        option['colorbar_ticks'] = get_ticks(option["vmin"], option["vmax"])

        ticks = matplotlib.ticker.MultipleLocator(base=option['colorbar_ticks'])
        mticks = ticks.tick_values(vmin=option['vmin'], vmax=option['vmax'])
        mticks = [round(tick, 2) if isinstance(tick, float) and len(str(tick).split('.')[1]) > 2 else tick for tick in mticks]
        if mticks[0] < option['vmin'] and mticks[-1] < option['vmax']:
            mticks = mticks[1:]
        elif mticks[0] > option['vmin'] and mticks[-1] > option['vmax']:
            mticks = mticks[:-1]
        elif mticks[0] < option['vmin'] and mticks[-1] > option['vmax']:
            mticks = mticks[1:-1]
        option['vmax'], option['vmin'] = mticks[-1], mticks[0]

        if option['cmap'] is not None:
            cmap = cm.get_cmap(option['cmap'])
            bnd = np.arange(option["vmin"], option["vmax"] + option['colorbar_ticks'] / 2, option['colorbar_ticks'] / 2)
            norm = colors.BoundaryNorm(bnd, cmap.N)
        else:
            cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4',
                        '#313695']
            cmap = colors.ListedColormap(cpool)
            bnd = np.arange(option["vmin"], option["vmax"] + option['colorbar_ticks'] / 2, option['colorbar_ticks'] / 2)
            norm = colors.BoundaryNorm(bnd, cmap.N)

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

    if option['show_method'] == 'imshow':
        cs = ax.imshow(ds[xitem].values, cmap=colormap, vmin=option['vmin'], vmax=option['vmax'], extent=extent,
                        origin=origin)
    elif option['show_method'] == 'contourf':
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
                        extend=option['extend'],
                        orientation=option['colorbar_position'])
    cb.solids.set_edgecolor("face")
    # cb.set_label('%s' % (varname), position=(0.5, 1.5), labelpad=-35)
    plt.savefig(
        f'{self.casedir}/output/{s_m}/{self.item}_stn_{self.ref_source}_{self.sim_source}_{varname}.{option["saving_format"]}',
        format=f'{option["saving_format"]}', dpi=option['dpi'])
    plt.close()

def process_unit_stn(self,ref_unit,sim_unit,metric):
    all_metrics_units = {
        'percent_bias': '%',  # Percent Bias
        'absolute_percent_bias': '%',  # Absolute Percent Bias
        'bias': 'Same as input data',  # Bias
        'mean_absolute_error': 'Same as input data',  # Mean Absolute Error
        'RMSE': 'Same as input data',  # Root Mean Squared Error
        'MSE': 'Square of input data unit',  # Mean Squared Error
        'ubRMSE': 'Same as input data',  # Unbiased Root Mean Squared Error
        'CRMSD': 'Same as input data',  # Centered Root Mean Square Difference
        'nrmse': 'Unitless',  # Normalized Root Mean Square Error
        'L': 'Unitless',  # Likelihood
        'correlation': 'Unitless',  # correlation coefficient
        'correlation_R2': 'Unitless',  # correlation coefficient R2
        'NSE': 'Unitless',  # Nash Sutcliffe efficiency coefficient
        'LNSE': 'Unitless',  # natural logarithm of NSE coefficient
        'KGE': 'Unitless',  # Kling-Gupta Efficiency
        'KGESS': 'Unitless',  # Normalized Kling-Gupta Efficiency
        'kappa_coeff': 'Unitless',  # Kappa coefficient
        'rv': 'Unitless',  # Relative variability (amplitude ratio)
        'ubNSE': 'Unitless',  # Unbiased Nash Sutcliffe efficiency coefficient
        'ubKGE': 'Unitless',  # Unbiased Kling-Gupta Efficiency
        'ubcorrelation': 'Unitless',  # Unbiased correlation
        'ubcorrelation_R2': 'Unitless',  # correlation coefficient R2
        'pc_max': '%',  # the bias of the maximum value
        'pc_min': '%',  # the bias of the minimum value
        'pc_ampli': '%',  # the bias of the amplitude value
        'rSD': 'Unitless',  # Ratio of standard deviations
        'PBIAS_HF': '%',  # Percent bias of flows ≥ Q98 (Yilmaz et al., 2008)
        'PBIAS_LF': '%',  # Percent bias of flows ≤ Q30(Yilmaz et al., 2008)
        'SMPI': 'Unitless',  # https://docs.esmvaltool.org/en/latest/recipes/recipe_smpi.html
        'ggof': 'Unitless',  # Graphical Goodness of Fit
        'gof': 'Unitless',  # Numerical Goodness-of-fit measures
        'KGEkm': 'Unitless',  # Kling-Gupta Efficiency with knowable-moments
        'KGElf': 'Unitless',  # Kling-Gupta Efficiency for low values
        'KGEnp': 'Unitless',  # Non-parametric version of the Kling-Gupta Efficiency
        'md': 'Unitless',  # Modified Index of Agreement
        'mNSE': 'Unitless',  # Modified Nash-Sutcliffe efficiency
        'pbiasfdc': '%',  # Percent Bias in the Slope of the Midsegment of the Flow Duration Curve
        'pfactor': '%',  # the percent of observations that are within the given uncertainty bounds.
        'rd': 'Unitless',  # Relative Index of Agreement
        'rfactor': 'Unitless',
        # the average width of the given uncertainty bounds divided by the standard deviation of the observations.
        'rNSE': 'Unitless',  # Relative Nash-Sutcliffe efficiency
        'rSpearman': 'Unitless',  # Spearman's rank correlation coefficient
        'rsr': 'Unitless',  # Ratio of RMSE to the standard deviation of the observations
        'sKGE': 'Unitless',  # Split Kling-Gupta Efficiency
        'ssq': 'Square of input data unit',  # Sum of the Squared Residuals
        'valindex': 'Unitless',  # Valid Indexes
        've': 'Unitless',  # Volumetric Efficiency
        'wNSE': 'Unitless',  # Weighted Nash-Sutcliffe efficiency
        'wsNSE': 'Unitless',  # Weighted seasonal Nash-Sutcliffe Efficiency
        'index_agreement': 'Unitless',  # Index of agreement
    }

    unit = all_metrics_units[metric]
    if unit == 'Unitless':
        return '[Unitless]'
    elif unit == '%':
        return '[%]'
    elif unit == 'Same as input data':
        return f'[{ref_unit}]'
    elif unit == 'Square of input data unit':
        return rf'[${ref_unit}^{{2}}$]'
    else:
        print('Warning: Missing metric unit!')
        return '[None]'

def make_plot_index_stn(self):

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

    # read the data
    df = pd.read_csv(f'{self.casedir}/output/scores/{self.item}_stn_{self.ref_source}_{self.sim_source}_evaluations.csv',
                        header=0)
    # loop the keys in self.variables to get the metric output
    for metric in self.metrics:
        option = self.fig_nml['make_stn_plot_index']
        if 'extend' not in self.fig_nml['make_geo_plot_index']:
            self.fig_nml['make_geo_plot_index']['extend'] = 'both'  # Default value
        option['extend'] = self.fig_nml['make_geo_plot_index']['extend']
        print(f'plotting metric: {metric}')
        option['colorbar_label'] = metric.replace('_', ' ')+' ' + process_unit_stn(self, self.ref_varunit,self.sim_varunit, metric)
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

        option['colorbar_ticks'] = get_ticks(option["vmin"], option["vmax"])

        ticks = matplotlib.ticker.MultipleLocator(base=option['colorbar_ticks'])
        mticks = ticks.tick_values(vmin=option['vmin'], vmax=option['vmax'])
        mticks = [round(tick, 2) if isinstance(tick, float) and len(str(tick).split('.')[1]) > 2 else tick for tick in mticks]
        if mticks[0] < option['vmin'] and mticks[-1] < option['vmax']:
            mticks = mticks[1:]
        elif mticks[0] > option['vmin'] and mticks[-1] > option['vmax']:
            mticks = mticks[:-1]
        elif mticks[0] < option['vmin'] and mticks[-1] > option['vmax']:
            mticks = mticks[1:-1]
        option['vmax'], option['vmin'] = mticks[-1], mticks[0]

        if option['cmap'] is not None:
            cmap = cm.get_cmap(option['cmap'])
            bnd = np.arange(option["vmin"], option["vmax"] + option['colorbar_ticks'] / 2, option['colorbar_ticks'] / 2)
            norm = colors.BoundaryNorm(bnd, cmap.N)
        else:
            cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4',
                        '#313695']
            cmap = colors.ListedColormap(cpool)
            # bnd = np.linspace(mticks[0], mticks[-1], 11)
            bnd = np.arange(option["vmin"], option["vmax"] + option['colorbar_ticks'] / 2, option['colorbar_ticks'] / 2)
            norm = colors.BoundaryNorm(bnd, cmap.N)
        plot_stn_map(self,lon_select, lat_select, plotvar, cmap, norm, metric, 'metrics', mticks, option)

    for score in self.scores:
        option = self.fig_nml['make_stn_plot_index']
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

        option['colorbar_ticks'] = get_ticks(option["vmin"], option["vmax"])

        ticks = matplotlib.ticker.MultipleLocator(base=option['colorbar_ticks'])
        mticks = ticks.tick_values(vmin=option['vmin'], vmax=option['vmax'])
        mticks = [round(tick, 2) if isinstance(tick, float) and len(str(tick).split('.')[1]) > 2 else tick for tick in mticks]
        if mticks[0] < option['vmin'] and mticks[-1] < option['vmax']:
            mticks = mticks[1:]
        elif mticks[0] > option['vmin'] and mticks[-1] > option['vmax']:
            mticks = mticks[:-1]
        elif mticks[0] < option['vmin'] and mticks[-1] > option['vmax']:
            mticks = mticks[1:-1]
        option['vmax'], option['vmin'] = mticks[-1], mticks[0]

        if option['cmap'] is not None:
            cmap = cm.get_cmap(option['cmap'])
            bnd = np.arange(option["vmin"], option["vmax"] + option['colorbar_ticks'] / 2, option['colorbar_ticks'] / 2)
            norm = colors.BoundaryNorm(bnd, cmap.N)
        else:
            cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4',
                        '#313695']
            cmap = colors.ListedColormap(cpool)
            bnd = np.arange(option["vmin"], option["vmax"] + option['colorbar_ticks'] / 2, option['colorbar_ticks'] / 2)
            norm = colors.BoundaryNorm(bnd, cmap.N)

        plot_stn_map(self,lon_select, lat_select, plotvar, cmap, norm, score, 'scores', mticks, option)


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

def make_Basic(output_dir, method_name, data_sources, main_nml,
                                 option):
    import numpy as np
    import xarray as xr
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    from matplotlib import rcParams
    filename_parts = [method_name] + data_sources
    filename = "_".join(filename_parts) + "_output"
    file = os.path.join(output_dir, f"{method_name}", filename)

    ds = xr.open_dataset(f"{file}.nc")
    data = ds[method_name]
    ilat = ds.lat.values
    ilon = ds.lon.values
    lon, lat = np.meshgrid(ilon, ilat)

    if not option['cmap']:
        option['cmap'] = 'coolwarm'
    min_value, max_value = np.nanmin(data), np.nanmax(data)
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

    if option['show_method'] == 'imshow':
        cs = ax.imshow(data, cmap=option['cmap'], vmin=option['vmin'], vmax=option['vmax'], extent=extent, origin=origin)
    elif option['show_method'] == 'contourf':
        cs = ax.contourf(lon, lat, data, levels=bnd, cmap=option['cmap'], norm=norm, extend=option['extend'])

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

    if option['title'] is None:
        option['title'] = f'Correlation Results'
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

    plt.savefig(f'{file}.{option["saving_format"]}', format=f'{option["saving_format"]}',
                dpi=option['dpi'])
    plt.close()

