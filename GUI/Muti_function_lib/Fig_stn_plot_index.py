import streamlit
import xarray as xr
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
from matplotlib import rcParams
from matplotlib import ticker
import math
import matplotlib.colors as clr
import itertools

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from io import BytesIO
import streamlit as st
from PIL import Image

font = {'family': 'DejaVu Sans'}
matplotlib.rc('font', **font)

params = {'backend': 'ps',
          'axes.labelsize': 10,
          'grid.linewidth': 0.2,
          'font.size': 12,
          'legend.fontsize': 12,
          'legend.frameon': False,
          'xtick.labelsize': 12,
          'xtick.direction': 'out',
          'ytick.labelsize': 12,
          'ytick.direction': 'out',
          'savefig.bbox': 'tight',
          'axes.unicode_minus': False,
          'text.usetex': False}
rcParams.update(params)


def make_stn_plot_index(casedir, ref, sim, item, metric, selected_item, option):
    # read the data
    df = pd.read_csv(f'{casedir}/{item}/{selected_item}_stn_{ref}_{sim}_evaluations.csv', header=0)
    # for metric in metrics:
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
        option['colorbar_ticks'] = 10 ** math.floor(math.log10(option["vmax"] - option["vmin"]))/2
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
        cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
        cmap = colors.ListedColormap(cpool)
        # bnd = np.linspace(mticks[0], mticks[-1], 11)
        bnd = np.arange(mticks[0], mticks[-1] + option['colorbar_ticks'] / 2, option['colorbar_ticks'] / 2)
        norm = colors.BoundaryNorm(bnd, cmap.N)
    plot_stn_map(lon_select, lat_select, plotvar, cmap, norm, ref, sim, selected_item, metric, mticks, option)


def plot_stn_map(loc_lon, loc_lat, metric_data, cmap, norm, ref, sim, selected_item, metric, mticks, option):
    from pylab import rcParams
    import matplotlib
    import matplotlib.pyplot as plt
    ### Plot settings

    font = {'family': option['font']}
    # font = {'family' : 'Myriad Pro'}
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

    cs = ax.scatter(loc_lon, loc_lat, s=option['markersize'], c=metric_data, cmap=cmap, norm=norm, marker=option['marker'],
                    edgecolors='none', alpha=0.9)

    coastline = cfeature.NaturalEarthFeature(
        'physical', 'coastline', '50m', edgecolor='0.6', facecolor='none')
    rivers = cfeature.NaturalEarthFeature(
        'physical', 'rivers_lake_centerlines', '110m', edgecolor='0.6', facecolor='none')
    ax.add_feature(cfeature.LAND, facecolor='0.8')
    ax.add_feature(coastline, linewidth=0.6)
    ax.add_feature(cfeature.LAKES, alpha=1, facecolor='white',edgecolor='white')
    ax.add_feature(rivers, linewidth=0.5)

    ax.gridlines(draw_labels=False, linestyle=':', linewidth=0.5, color='grey', alpha=0.8)

    ax.set_extent([option['min_lon'], option['max_lon'], option['min_lat'], option['max_lat']])
    ax.set_xticks(np.arange(option['max_lon'], option['min_lon'], -60)[::-1], crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(option['max_lat'], option['min_lat'], -30)[::-1], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    ax.set_xlabel(option['xticklabel'], fontsize=option['xtick'] + 1)
    ax.set_ylabel(option['yticklabel'], fontsize=option['ytick'] + 1)
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
                      orientation=option['colorbar_position'], extend=option['extend'])

    cb.solids.set_edgecolor("face")
    # cb.set_label('%s' % (metric), position=(0.5, 1.7), labelpad=-35)

    file = f'{selected_item}_stn_{ref}_{sim}_{metric}'
    st.pyplot(fig)

    # 将图像保存到 BytesIO 对象
    buffer = BytesIO()
    fig.savefig(buffer, format=option['saving_format'], dpi=option['dpi'])
    buffer.seek(0)
    buffer.seek(0)
    st.download_button('Download image', buffer, file_name=f'{file}.{option["saving_format"]}',
                       mime=f"image/{option['saving_format']}",
                       type="secondary", disabled=False, use_container_width=False, key=metric)
