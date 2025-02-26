import itertools
import sys
import matplotlib
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import numpy as np
import pandas as pd
from matplotlib import rcParams

# Plot settings
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from matplotlib import rcParams


def process_unit(ref_unit, metric):
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
    if metric not in all_metrics_units.keys():
        return '[None]'
    else:
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


def get_index(vmin, vmax, colormap):
    import math
    def get_ticks(vmin, vmax):
        if 2 >= vmax - vmin > 1:
            colorbar_ticks = 0.2
        elif 5 >= vmax - vmin > 2:
            colorbar_ticks = 0.5
        elif 10 >= vmax - vmin > 5:
            colorbar_ticks = 1
        elif 50 >= vmax - vmin > 10:
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


def plot_grid_map(basedir, filename, main_nml, xitem, option):
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

    # Extract variables
    ilat = ds.lat.values
    ilon = ds.lon.values
    lat, lon = np.meshgrid(ilat[::-1], ilon)

    var = ds[xitem].transpose("lon", "lat")[:, ::-1].values
    max_value = max(abs(np.nanmin(var)), np.nanmax(var))
    min_value = max_value * -1
    cmap, mticks, norm, bnd = get_index(min_value, max_value, option['cmap'])
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

    if option['show_method'] == 'imshow':
        cs = ax.imshow(ds[xitem].values, cmap=cmap, vmin=option['vmin'], vmax=option['vmax'], extent=extent,
                       origin=origin)
    elif option['show_method'] == 'contourf':
        cs = ax.contourf(lon, lat, var, levels=bnd, cmap=cmap, norm=norm, extend=option['extend'])

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
                      extend=option['extend'], orientation=option['colorbar_position'])
    cb.solids.set_edgecolor("face")

    filename2 = filename[:-3]
    plt.savefig(f'{basedir}/{filename2}.{option["saving_format"]}', format=f'{option["saving_format"]}', dpi=option['dpi'])
    plt.close()


def plot_stn_map(basedir, filename, stn_lon, stn_lat, metric, main_nml, varname, option):
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

    max_value = max(abs(np.nanmin(metric)), np.nanmax(metric))
    min_value = max_value * -1
    cmap, mticks, norm, bnd = get_index(min_value, max_value, option['cmap'])
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
        plot_option['colorbar_label'] = process_unit(unit, item_type)
    else:
        plot_option['title'] = f'{evaluation_item} {item_type} difference {sim_source[0]} vs {sim_source[1]}'
        # if not plot_option['colorbar_label']:
        unit = sim_nml[f'{evaluation_item}'][f'{sim_source[0]}_varunit']
        plot_option['colorbar_label'] = process_unit(unit, item_type)

    if not plot_option['cmap']:
        plot_option['cmap'] = 'RdBu_r'  # Diverging colormap for anomalies/differences

    # For station data
    if ref_data_type == 'stn':
        data = pd.read_csv(f'{basedir}/{filename}', header=0)
        lon_select = data['lon'].values
        lat_select = data['lat'].values
        plotvar = data[f'{item_type}_{"anomaly" if data_type == "anomaly" else "diff"}'].values
        plot_stn_map(basedir, filename, lon_select, lat_select, plotvar, main_nml, f'{data_type}_{item_type}', plot_option)

    # For gridded data
    else:  # xarray Dataset
        plot_grid_map(basedir, filename, main_nml,
                      f'{item_type}_{"anomaly" if data_type == "anomaly" else "diff"}', plot_option)


def make_scenarios_comparison_Diff_Plot(basedir, metrics, scores, evaluation_item, ref_source, sim_sources, main_nml, sim_nml,
                                        ref_data_type, option):
    for metric in metrics:
        for sim_source in sim_sources:
            plot_diff_results(basedir, 'anomaly', metric, evaluation_item, ref_source, sim_source, main_nml, sim_nml,
                              ref_data_type, option)
        # After calculating differences for metrics
        for i, sim1 in enumerate(sim_sources):
            for j, sim2 in enumerate(sim_sources[i + 1:], i + 1):
                plot_diff_results(basedir, 'difference', metric, evaluation_item, ref_source,
                                  (sim1, sim2), main_nml, sim_nml, ref_data_type, option)

    for score in scores:
        # After calculating anomalies for scores
        for sim_source in sim_sources:
            plot_diff_results(basedir, 'anomaly', score, evaluation_item, ref_source, sim_source, main_nml, sim_nml, ref_data_type
                              , option)

        # After calculating differences for scores
        for i, sim1 in enumerate(sim_sources):
            for j, sim2 in enumerate(sim_sources[i + 1:], i + 1):
                plot_diff_results(basedir, 'difference', score, evaluation_item, ref_source, (sim1, sim2), main_nml, sim_nml,
                                  ref_data_type, option)
