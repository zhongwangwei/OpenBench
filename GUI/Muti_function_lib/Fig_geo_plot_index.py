import pandas as pd
import numpy as np
import xarray as xr

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
from matplotlib import rcParams
from matplotlib import ticker
import matplotlib.colors as clr
from matplotlib.pyplot import MultipleLocator

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import math
import itertools

from io import BytesIO
import streamlit as st
from PIL import Image


def get_ticks(vmin, vmax):
    if 2 >= vmax - vmin > 1:
        colorbar_ticks = 0.2
    elif 5 >= vmax - vmin > 2:
        colorbar_ticks = 0.5
    elif 10 >= vmax - vmin > 5:
        colorbar_ticks = 1.
    elif 20 >= vmax - vmin > 10:
        colorbar_ticks = 5.
    elif 50 >= vmax - vmin > 20:
        colorbar_ticks = 10.
    elif 80 >= vmax - vmin > 50:
        colorbar_ticks = 15.
    elif 100 >= vmax - vmin > 80:
        colorbar_ticks = 20.
    elif 200 >= vmax - vmin > 100:
        colorbar_ticks = 40.
    elif 500 >= vmax - vmin > 200:
        colorbar_ticks = 50.
    elif 1000 >= vmax - vmin > 500:
        colorbar_ticks = 100.
    elif 2000 >= vmax - vmin > 1000:
        colorbar_ticks = 200.
    elif 10000 >= vmax - vmin > 2000:
        colorbar_ticks = 10 ** math.floor(math.log10(vmax - vmin)) / 2
    else:
        colorbar_ticks = 0.10
    return colorbar_ticks


def get_index(vmin, vmax, colormap, option):
    if not option['vmin_max_on']:
        colorbar_ticks = get_ticks(vmin, vmax)
    else:
        colorbar_ticks = option['colorbar_ticks']
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

    try:
        norm = colors.BoundaryNorm(bnd, cmap.N)
    except:
        if cmap.N < len(bnd):
            bnd = np.linspace(vmin, vmax + colorbar_ticks / 2, 15)
            norm = colors.BoundaryNorm(bnd, cmap.N)
    return mticks, norm, bnd, cmap


def plot_geo_map(Figure_show, casedir, item, ref, sim, selected_item, xitem, option):
    font = {'family': option['font']}
    matplotlib.rc('font', **font)

    params = {'backend': 'ps',
              'axes.labelsize': option['labelsize'],
              'grid.linewidth': 0.2,
              'font.size': option['labelsize'],
              'xtick.labelsize': option['xticksize'],
              'xtick.direction': 'out',
              'ytick.labelsize': option['yticksize'],
              'ytick.direction': 'out',
              'savefig.bbox': 'tight',
              'axes.unicode_minus': False,
              'text.usetex': False}
    rcParams.update(params)

    mticks, normalize, levels, colormap = get_index(option['vmin'], option['vmax'], option['cmap'], option)
    ilat, ilon, var, data = prepare_geo_plot_index(casedir, ref, sim, item, selected_item, xitem, 'value')
    fig = plt.figure(figsize=(option['x_wise'], option['y_wise']))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    if ilat[0] - ilat[-1] < 0:
        option['origin'] = 'lower'
    else:
        option['origin'] = 'upper'

    if option['map'] == 'interpolate':
        lon, lat = np.meshgrid(ilon, ilat)
        cs = ax.contourf(lon, lat, var, levels=levels, cmap=colormap, norm=normalize, extend=option['extend'])
    else:
        extent = (ilon[0], ilon[-1], ilat[0], ilat[-1])
        cs = ax.imshow(data, cmap=colormap, vmin=option['vmin'], vmax=option['vmax'], extent=extent,
                       origin=option['origin'])
    ax.set_extent([option['min_lon'], option['max_lon'], option['min_lat'], option['max_lat']], crs=ccrs.PlateCarree())
    ax.set_adjustable('datalim')  # 固定数据范围，不自动扩展
    ax.set_aspect('equal', adjustable='box')  # 保持数据比例

    coastline = cfeature.NaturalEarthFeature(
        'physical', 'coastline', '50m', edgecolor='0.6', facecolor='none')
    rivers = cfeature.NaturalEarthFeature(
        'physical', 'rivers_lake_centerlines', '110m', edgecolor='0.6', facecolor='none')
    ax.add_feature(cfeature.LAND, facecolor='0.8')
    ax.add_feature(coastline, linewidth=0.6)
    ax.add_feature(cfeature.LAKES, alpha=1, facecolor='white', edgecolor='white')
    ax.add_feature(rivers, linewidth=0.5)
    ax.gridlines(draw_labels=False, linestyle=':', linewidth=0.5, color='grey', alpha=0.8)

    ax.set_xticks(np.arange(option['max_lon'], option['min_lon'], -1 * option["xtick"])[::-1], crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(option['max_lat'], option['min_lat'], -1 * option["ytick"])[::-1], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    ax.set_xlabel(option['xticklabel'], fontsize=option['xticksize'] + 1)
    ax.set_ylabel(option['yticklabel'], fontsize=option['yticksize'] + 1)
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

    cb = fig.colorbar(cs, cax=cbaxes, spacing='uniform', label=option['colorbar_label'], extend=option['extend'],
                      orientation=option['colorbar_position'], ticks=mticks)  # , ticks=mticks
    cb.solids.set_edgecolor("face")

    file = f'{selected_item}_ref_{ref}_sim_{sim}_{xitem}'
    with Figure_show:
        st.pyplot(fig)

    # 将图像保存到 BytesIO 对象
    buffer = BytesIO()
    fig.savefig(buffer, format=option['saving_format'], dpi=option['dpi'])
    buffer.seek(0)
    buffer.seek(0)
    st.download_button('Download image', buffer, file_name=f'{file}.{option["saving_format"]}',
                       mime=f"image/{option['saving_format']}",
                       type="secondary", disabled=False, use_container_width=False, key=file)


def prepare_geo_plot_index(path, ref, sim, item, selected_item, metric, function):
    ds = xr.open_dataset(f'{path}/{item}/{selected_item}_ref_{ref}_sim_{sim}_{metric}.nc')[metric]
    import math
    if function == 'min_max':
        quantiles = ds.quantile([0.05, 0.95], dim=['lat', 'lon'], skipna=True)
        vmax = math.ceil(quantiles[1].values)
        vmin = math.floor(quantiles[0].values)
        if vmax > 100: vmax = 100
        if vmin < -100: vmin = -100
        if metric in ['bias', 'percent_bias', 'rSD', 'PBIAS_HF', 'PBIAS_LF']:
            colorbar_ticks = get_ticks(vmin, vmax)
            return vmin, vmax, colorbar_ticks
        elif metric in ['KGE', 'KGESS', 'correlation', 'kappa_coeff', 'rSpearman']:
            colorbar_ticks = get_ticks(-1, 1)
            return -1, 1, colorbar_ticks
        elif metric in ['NSE', 'LNSE', 'ubNSE', 'rNSE', 'wNSE', 'wsNSE']:
            colorbar_ticks = get_ticks(-1, 1)
            return vmin, 1, colorbar_ticks
        elif metric in ['RMSE', 'CRMSD', 'MSE', 'ubRMSE', 'nRMSE', 'mean_absolute_error', 'ssq', 've',
                        'absolute_percent_bias']:
            colorbar_ticks = get_ticks(vmin, 1)
            return 0, vmax, colorbar_ticks
        else:
            colorbar_ticks = get_ticks(0, 1)
            return 0, 1, colorbar_ticks
    else:
        ilat = ds.lat.values
        ilon = ds.lon.values
        var = ds.values
        return ilat, ilon, var, ds


def make_geo_plot_index(item, metric, selected_item, ref, sim, path):
    key_value = f"{selected_item}_{metric}_{ref}_{sim}_"
    option = {}
    Figure_show = st.container()
    Labels_tab, Scale_tab, Map_tab, Save_tab = st.tabs(['Labels', 'Scale', 'Map', 'Save'])
    with Labels_tab:
        col1, col2, col3, col4 = st.columns((3.5, 3, 3, 3))
        with col1:
            option['title'] = st.text_input('Title', value=f'', label_visibility="visible", key=f"{key_value}title")
            option['title_size'] = st.number_input("Title label size", min_value=0, value=20,
                                                   key=f"{key_value}_titlesize")
        with col2:
            option['xticklabel'] = st.text_input('X tick labels', value='Longitude', label_visibility="visible",
                                                 key=f"{key_value}xticklabel")
            option['xticksize'] = st.number_input("xtick label size", min_value=0, value=17, key=f"{key_value}xticksize")

        with col3:
            option['yticklabel'] = st.text_input('Y tick labels', value='Latitude', label_visibility="visible",
                                                 key=f"{key_value}yticklabel")
            option['yticksize'] = st.number_input("ytick label size", min_value=0, value=17, key=f"{key_value}yticksize")

        with col4:
            option['labelsize'] = st.number_input("labelsize", min_value=0, value=17, key=f"{key_value}labelsize")
            option['axes_linewidth'] = st.number_input("axes linewidth", min_value=0, value=1,
                                                       key=f"{key_value}axes_linewidth")

    with Map_tab:
        col1, col2, col3, col4 = st.columns(4)
        option["min_lon"] = col1.number_input(f"minimal longitude", value=st.session_state['generals']['min_lon'],
                                              key=f"{key_value}min_lon")
        option["max_lon"] = col2.number_input(f"maximum longitude", value=st.session_state['generals']['max_lon'],
                                              key=f"{key_value}max_lon")
        option["min_lat"] = col3.number_input(f"minimal latitude", value=st.session_state['generals']['min_lat'],
                                              key=f"{key_value}min_lat")
        option["max_lat"] = col4.number_input(f"maximum latitude", value=st.session_state['generals']['max_lat'],
                                              key=f"{key_value}max_lat")
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        option["map"] = col1.selectbox(f"Draw map", ['None', 'interpolate'],
                                       index=0, placeholder="Choose an option", label_visibility="visible",
                                       key=f"{key_value}map")
        option["xtick"] = col2.number_input(f"Set x tick scale", value=60., min_value=0., max_value=360., step=10.,
                                            key=f"{key_value}xtick")
        option["ytick"] = col3.number_input(f"Set y tick scale", value=30., min_value=0., max_value=180., step=10.,
                                            key=f"{key_value}ytick")

    with Scale_tab:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            option['cmap'] = st.selectbox('Colorbar',
                                          ['coolwarm', 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn',
                                           'BuGn_r',
                                           'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r',
                                           'Grays', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges',
                                           'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r',
                                           'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r',
                                           'PuBu_r',
                                           'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r',
                                           'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r',
                                           'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r',
                                           'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r',
                                           'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r',
                                           'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r',
                                           'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r',
                                           'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag',
                                           'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey',
                                           'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow',
                                           'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r',
                                           'gist_yerg', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray',
                                           'gray_r', 'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r',
                                           'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r',
                                           'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow',
                                           'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer',
                                           'summer_r',
                                           'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c',
                                           'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight',
                                           'twilight_r',
                                           'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter',
                                           'winter_r'], index=0, placeholder="Choose an option",
                                          label_visibility="visible", key=f"{key_value}cmap")
        with col2:
            option['colorbar_label'] = st.text_input('colorbar label', value=metric.replace('_', ' '),
                                                     label_visibility="visible", key=f"{key_value}colorbar_label")
        with col3:
            option["colorbar_position"] = st.selectbox('colorbar position', ['horizontal', 'vertical'],  # 'Season',
                                                       index=0, placeholder="Choose an option",
                                                       label_visibility="visible", key=f"{key_value}colorbar_position")
        with col4:
            option["extend"] = st.selectbox(f"colorbar extend", ['neither', 'both', 'min', 'max'],
                                            index=0, placeholder="Choose an option", label_visibility="visible",
                                            key=f"{key_value}extend")

        if option["colorbar_position"] == 'vertical':
            left, bottom, right, top = 0.94, 0.24, 0.02, 0.5
        else:
            left, bottom, right, top = 0.26, 0.14, 0.5, 0.03
        col1, col2, col3 = st.columns(3)
        option['colorbar_position_set'] = col1.toggle('Setting colorbar position', value=False,
                                                      key=f"{key_value}colorbar_position_set")
        if option['colorbar_position_set']:
            col1, col2, col3, col4 = st.columns(4)
            option["colorbar_left"] = col1.number_input(f"colorbar left", value=left, key=f"{key_value}colorbar_left")
            option["colorbar_bottom"] = col2.number_input(f"colorbar bottom", value=bottom, key=f"{key_value}colorbar_bottom")
            option["colorbar_width"] = col3.number_input(f"colorbar width", value=right, key=f"{key_value}colorbar_width")
            option["colorbar_height"] = col4.number_input(f"colorbar height", value=top, key=f"{key_value}colorbar_height")

        col1, col2, col3, col4 = st.columns(4)
        option["vmin_max_on"] = col1.toggle('Fit to data', value=False, key=f"{key_value}vmin_max_on")

        error = False
        try:
            vmin, vmax, colorbar_ticks = prepare_geo_plot_index(path, ref, sim, item, selected_item, metric, 'min_max')
            if option["vmin_max_on"]:
                option["colorbar_ticks"] = col2.number_input(f"Colorbar Ticks locater", value=colorbar_ticks, step=0.1,
                                                             key=f"{key_value}colorbar_ticks")
                try:
                    option["vmin"] = col3.number_input(f"colorbar min", value=vmin, key=f"{key_value}vmin")
                    option["vmax"] = col4.number_input(f"colorbar max", value=vmax, key=f"{key_value}vmax")
                except ValueError:
                    st.error(f"Max value must larger than min value.")
            else:
                option["colorbar_ticks"] = 0.5
                option["vmin"] = vmin
                option["vmax"] = vmax
        except Exception as e:
            st.error(f"Error: {e}")
            error = True

    with Save_tab:
        col1, col2, col3 = st.columns(3)
        with col1:
            option["x_wise"] = st.number_input(f"X Length", min_value=0, value=10, key=f"{key_value}x_wise")
            option['saving_format'] = st.selectbox('Image saving format', ['png', 'jpg', 'eps'],
                                                   index=1, placeholder="Choose an option", label_visibility="visible",
                                                   key=f"{key_value}saving_format")
        with col2:
            option["y_wise"] = st.number_input(f"y Length", min_value=0, value=6)
            option['font'] = st.selectbox('Image saving format',
                                          ['Times new roman', 'Arial', 'Courier New', 'Comic Sans MS', 'Verdana',
                                           'Helvetica',
                                           'Georgia', 'Tahoma', 'Trebuchet MS', 'Lucida Grande'],
                                          index=0, placeholder="Choose an option", label_visibility="visible",
                                          key=f"{key_value}y_wise")
        with col3:
            option['dpi'] = st.number_input(f"Figure dpi", min_value=0, value=300, key=f"{key_value}dpi")
    st.divider()
    if not error:
        plot_geo_map(Figure_show, path, item, ref, sim, selected_item, metric, option)
    else:
        st.error(f'Please check File: {selected_item}_ref_{ref}_sim_{sim}_{metric}.nc')
