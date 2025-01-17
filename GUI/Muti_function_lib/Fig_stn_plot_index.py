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



def plot_stn_map(loc_lon, loc_lat, metric_data, cmap, norm, ref, sim, selected_item, metric, mticks, option):
    from pylab import rcParams
    import matplotlib
    import matplotlib.pyplot as plt

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

    fig = plt.figure(figsize=(option['x_wise'], option['y_wise']))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.set_extent([option['min_lon'], option['max_lon'], option['min_lat'], option['max_lat']], crs=ccrs.PlateCarree())

    cs = ax.scatter(loc_lon, loc_lat, s=option['markersize'], c=metric_data, cmap=cmap, vmin=option['vmin'],vmax=option['vmax'], marker=option['marker'],
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
    ax.set_xticks(np.arange(option['max_lon'], option['min_lon'], -1*option["xtick"])[::-1], crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(option['max_lat'], option['min_lat'], -1*option["ytick"])[::-1], crs=ccrs.PlateCarree())
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
    cb = fig.colorbar(cs, cax=cbaxes, ticks=mticks, spacing='uniform', label=option['colorbar_label'],
                      orientation=option['colorbar_position'], extend=option['extend'])

    cb.solids.set_edgecolor("face")

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

def prepare_stn_plot_index(casedir, ref, sim, item, metric, selected_item, option):
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
    elif 5 >= option["vmax"] - option["vmin"] > 2:
        option['colorbar_ticks'] = 0.5
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
        bnd = np.arange(mticks[0], mticks[-1] + option['colorbar_ticks'] / 2, option['colorbar_ticks'] / 2)
        norm = colors.BoundaryNorm(bnd, cmap.N)
    else:
        cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
        cmap = colors.ListedColormap(cpool)
        bnd = np.arange(mticks[0], mticks[-1] + option['colorbar_ticks'] / 2, option['colorbar_ticks'] / 2)
        norm = colors.BoundaryNorm(bnd, cmap.N)
    plot_stn_map(lon_select, lat_select, plotvar, cmap, norm, ref, sim, selected_item, metric, mticks, option)


def make_stn_plot_index(item, metric, selected_item, ref, sim, path):
    key_value = f"{selected_item}_{metric}_{ref}_{sim}_"
    option = {}
    with st.container(height=None, border=True):
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
        st.divider()

        with st.expander('More info', expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            # min_lon, max_lon, min_lat, max_lat
            option["min_lon"] = col1.number_input(f"minimal longitude", value=st.session_state['generals']['min_lon'])
            option["max_lon"] = col2.number_input(f"maximum longitude", value=st.session_state['generals']['max_lon'])
            option["min_lat"] = col3.number_input(f"minimal latitude", value=st.session_state['generals']['min_lat'])
            option["max_lat"] = col4.number_input(f"maximum latitude", value=st.session_state['generals']['max_lat'])

            option["xtick"] = col1.number_input(f"Set x tick scale", value=60, min_value=0, max_value=360, step=10,key=f"{key_value}xtick")
            option["ytick"] = col2.number_input(f"Set y tick scale", value=30, min_value=0, max_value=180, step=10,key=f"{key_value}ytick")

            st.divider()
            col1, col2, col3 = st.columns((3, 2, 2))
            option["vmin_max_on"] = col1.toggle('Setting max min', value=False, key=f"{key_value}vmin_max_on")
            error = False
            try:
                import math
                df = pd.read_csv(f'{path}/{item}/{selected_item}_stn_{ref}_{sim}_evaluations.csv', header=0)
                # df = pd.read_csv(f'{path}/{item}/{selected_item}_stn_{ref}_{sim}_evaluations.csv', header=0)
                min_metric = -999.0
                max_metric = 100000.0
                ind0 = df[df['%s' % (metric)] > min_metric].index
                data_select0 = df.loc[ind0]
                ind1 = data_select0[data_select0['%s' % (metric)] < max_metric].index
                data_select = data_select0.loc[ind1]
                plotvar = data_select['%s' % (metric)].values

                try:
                    vmin, vmax = np.percentile(plotvar, 5), np.percentile(plotvar, 95)
                    if metric in ['bias', 'percent_bias', 'rSD', 'PBIAS_HF', 'PBIAS_LF']:
                        vmax = math.ceil(vmax)
                        vmin = math.floor(vmin)
                    elif metric in ['KGE', 'KGESS', 'correlation', 'kappa_coeff', 'rSpearman']:
                        vmin, vmax = -1, 1
                    elif metric in ['NSE', 'LNSE', 'ubNSE', 'rNSE', 'wNSE', 'wsNSE']:
                        vmin, vmax = math.floor(vmin), 1
                    elif metric in ['RMSE', 'CRMSD', 'MSE', 'ubRMSE', 'nRMSE', 'mean_absolute_error', 'ssq', 've',
                                    'absolute_percent_bias']:
                        vmin, vmax = 0, math.ceil(vmax)
                    else:
                        vmin, vmax = 0, 1
                except:
                    vmin, vmax = 0, 1

                if option["vmin_max_on"]:
                    try:
                        option["vmin"] = col2.number_input(f"colorbar min", value=vmin)
                        option["vmax"] = col3.number_input(f"colorbar max", value=vmax)

                        min_value, max_value = np.nanmin(plotvar), np.nanmax(plotvar)
                        if min_value < option['vmin'] and max_value > option['vmax']:
                            oextend = 'both'
                        elif min_value > option['vmin'] and max_value > option['vmax']:
                            extend = 'max'
                        elif min_value < option['vmin'] and max_value < option['vmax']:
                            extend = 'min'
                        else:
                            extend = 'neither'
                    except ValueError:
                        st.error(f"Max value must larger than min value.")
                else:
                    option["vmin"] = vmin
                    option["vmax"] = vmax
                    extend = 'neither'
            except Exception as e:
                st.error(f"Error: {e}")
                error = True

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                option['cmap'] = st.selectbox('Colorbar',
                                              ['coolwarm',
                                               'coolwarm_r', 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r',
                                               'BuGn', 'BuGn_r',
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
                                               'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm',
                                               'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag',
                                               'flag_r',
                                               'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey',
                                               'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow',
                                               'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r',
                                               'gist_yerg', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray',
                                               'gray_r',
                                               'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r',
                                               'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r',
                                               'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow',
                                               'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer',
                                               'summer_r',
                                               'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c',
                                               'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight',
                                               'twilight_r',
                                               'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter',
                                               'winter_r'], index=0, placeholder="Choose an option",
                                              label_visibility="visible")
            with col2:
                option['colorbar_label'] = st.text_input('colorbar label', value=metric.replace("_", " "),
                                                         label_visibility="visible")
            with col3:
                option["colorbar_position"] = st.selectbox('colorbar position', ['horizontal', 'vertical'],  # 'Season',
                                                           index=0, placeholder="Choose an option",
                                                           label_visibility="visible")
                if option["colorbar_position"] == 'vertical':
                    left, bottom, right, top = 0.94, 0.24, 0.02, 0.5
                else:
                    left, bottom, right, top = 0.26, 0.14, 0.5, 0.03

            def get_extend(extend):
                my_list = ['neither', 'both', 'min', 'max']
                index = my_list.index(extend.lower())
                return index

            with col4:
                option["extend"] = st.selectbox(f"colorbar extend", ['neither', 'both', 'min', 'max'],
                                                index=get_extend(extend), placeholder="Choose an option",
                                                label_visibility="visible",
                                                key=f"{key_value}extend")

            option['marker'] = col1.selectbox(f'Marker style',
                                              ['.', 'x', 'o', ">", '<', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X', "+",
                                               "^", "v"],
                                              index=2,
                                              placeholder="Choose an option",
                                              label_visibility="visible")

            option['markersize'] = col2.number_input(f"Markersize", min_value=0, value=15, step=1)

            col1, col2, col3 = st.columns(3)
            if option["colorbar_position"] == 'vertical':
                left, bottom, right, top = 0.94, 0.24, 0.02, 0.5
            else:
                left, bottom, right, top = 0.26, 0.14, 0.5, 0.03
            option['colorbar_position_set'] = col1.toggle('Setting colorbar position', value=False,
                                                          key=f"{key_value}colorbar_position_set")
            if option['colorbar_position_set']:
                col1, col2, col3, col4 = st.columns(4)
                option["colorbar_left"] = col1.number_input(f"colorbar left", value=left)
                option["colorbar_bottom"] = col2.number_input(f"colorbar bottom", value=bottom)
                option["colorbar_width"] = col3.number_input(f"colorbar width", value=right)
                option["colorbar_height"] = col4.number_input(f"colorbar height", value=top)

        col1, col2, col3 = st.columns(3)
        with col1:
            option["x_wise"] = st.number_input(f"X Length", min_value=0, value=10)
            option['saving_format'] = st.selectbox('Image saving format', ['png', 'jpg', 'eps'],
                                                   index=1, placeholder="Choose an option", label_visibility="visible")
        with col2:
            option["y_wise"] = st.number_input(f"y Length", min_value=0, value=6)
            option['font'] = st.selectbox('Image saving format',
                                          ['Times new roman', 'Arial', 'Courier New', 'Comic Sans MS', 'Verdana',
                                           'Helvetica',
                                           'Georgia', 'Tahoma', 'Trebuchet MS', 'Lucida Grande'],
                                          index=0, placeholder="Choose an option", label_visibility="visible")
        with col3:
            option['dpi'] = st.number_input(f"Figure dpi", min_value=0, value=300)
    if not error:
        prepare_stn_plot_index(path, ref, sim, item, metric, selected_item, option)
    else:
        st.error(f'Please check File: {selected_item}_stn_{ref}_{sim}_evaluations.csv')