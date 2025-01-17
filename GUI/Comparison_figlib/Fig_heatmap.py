import os
import xarray as xr
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import rcParams
from matplotlib import ticker
import math
import matplotlib.colors as clr
import itertools

from io import BytesIO
import streamlit as st


def draw_scenarios_scores_comparison_heat_map(file, score, items, cases, option):
    # Convert the data to a DataFrame
    # read the data from the file using csv, remove the first row, then set the index to the first column
    df = pd.read_csv(file, sep='\s+', header=0)
    # exclude the first column
    df.set_index('Item', inplace=True)
    ref_dataname = df.iloc[:, 0:]
    df = df.iloc[:, 1:]
    if items is not None:
        df = df.loc[items]
        ref_dataname = ref_dataname.loc[items]
    if cases is not None:
        cases = sorted(cases)
        df = df.loc[:, cases]
    font = {'family': option['font']}
    matplotlib.rc('font', **font)

    params = {'backend': 'ps',
              'axes.linewidth': option['axes_linewidth'],
              'font.size': option['fontsize'],
              'xtick.labelsize': option['xticksize'],
              'xtick.direction': 'out',
              'ytick.labelsize': option['yticksize'],
              'grid.linewidth': 0.2,
              'ytick.direction': 'out',
              'savefig.bbox': 'tight',
              'axes.unicode_minus': False,
              'text.usetex': False}
    rcParams.update(params)

    # Create the heatmap using Matplotlib
    fig, ax = plt.subplots(figsize=(option['x_wise'], option['y_wise']))
    im = ax.imshow(df, cmap=option['cmap'], vmin=option['min'], vmax=option['max'])

    # Add colorbar
    # Add labels and title
    ax.set_yticks(range(len(df.index)))
    ax.set_xticks(range(len(df.columns)))
    ax.set_yticklabels([y.replace('_', ' ') for y in df.index], rotation=option['y_rotation'], ha=option['y_ha'])
    ax.set_xticklabels(df.columns, rotation=option['x_rotation'], ha=option['x_ha'])

    ax.set_ylabel(option['ylabel'], fontsize=option['yticksize'] + 1)
    ax.set_xlabel(option['xlabel'], fontsize=option['xticksize'] + 1)

    ax.set_title(option['title'], fontsize=option['title_size'])
    # Add numbers to each cell
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            ax.text(j, i, f'{df.iloc[i, j]:{option["ticks_format"][1:]}}', ha='center', va='center',
                    color='white' if df.iloc[i, j] > 0.8 or df.iloc[i, j] < 0.2 else 'black', fontsize=option['fontsize'])

    for i in range(df.shape[0] + 1):
        ax.axhline(i - 0.5, color='white', linewidth=0.5)  # 水平线
    for j in range(df.shape[1]):
        ax.axvline(j - 0.5, color='white', linewidth=0.5)  # 垂直线

    max_tick_width = 0
    for i in range(len(ref_dataname.index)):
        text_obj = ax.text(len(df.columns) - 0.3, i, f'{ref_dataname.iloc[i, 0]}', ha='left', va='center', color='black',
                           fontsize=option['yticksize'])
        # add the small ticks to the right of the heatmap
        ax.text(len(df.columns) - 0.5, i, '-', ha='left', va='center', color='black', fontsize=option['fontsize'])

        text_position = text_obj.get_position()
        bbox = text_obj.get_window_extent()
        bbox_in_fig_coords = bbox.transformed(fig.transFigure.inverted())
        max_tick_width = max(max_tick_width, bbox_in_fig_coords.width)
    # add the colorbar, and make it shrink to the size of the heatmap, and the location is at the right of reference data name

    # Create dynamically positioned and sized colorbar
    pos = ax.get_position()  # .bounds
    left, right, bottom, width, height = pos.x0, pos.x1, pos.y0, pos.width, pos.height

    if option["colorbar_position"] == 'vertical':
        cbar_ax = fig.add_axes([right + max_tick_width + 0.05, bottom, 0.03, height])  # right + 0.2
    else:
        xlabel = ax.xaxis.label
        xticks = ax.get_xticklabels()
        max_xtick_height = 0
        for xtick in xticks:
            bbox = xtick.get_window_extent()  # 获取每个 xtick 的包围框
            bbox_transformed = bbox.transformed(fig.transFigure.inverted())  # 将像素转换为图坐标
            max_xtick_height = max(max_xtick_height, bbox_transformed.height)
        if xlabel is not None:
            bbox = xlabel.get_window_extent()  # 获取每个 xtick 的包围框
            bbox_transformed = bbox.transformed(fig.transFigure.inverted())  # 将像素转换为图坐标
            x_height = bbox_transformed.height
            cbar_ax = fig.add_axes([left, bottom - max_xtick_height - x_height - 0.1, width, 0.05])
        else:
            cbar_ax = fig.add_axes([left, bottom - max_xtick_height - 0.1, width, 0.05])

    cbar = fig.colorbar(im, cax=cbar_ax, extend=option['extend'], orientation=option[
        'colorbar_position'], )  # , pad=option['colorbar_pad'],shrink=option['colorbar_shrink']

    file2 = f'scenarios_{score}_comparison'

    st.pyplot(fig)

    # 将图像保存到 BytesIO 对象
    buffer = BytesIO()
    fig.savefig(buffer, format=option['saving_format'], dpi=option['dpi'])
    buffer.seek(0)
    buffer.seek(0)
    st.download_button('Download image', buffer, file_name=f'{file2}.{option["saving_format"]}',
                       mime=f"image/{option['saving_format']}",
                       type="secondary", disabled=False, use_container_width=False)


def make_scenarios_scores_comparison_heat_map(dir_path, score, selected_items, sim):
    iscore = score.replace('_', ' ')
    option = {}
    item = 'heatmap'
    with st.container(height=None, border=True):
        col1, col2, col3, col4 = st.columns((3.5, 3, 3, 3))
        with col1:
            option['title'] = st.text_input('Title', value=f'Heatmap of {iscore}', label_visibility="visible",
                                            key=f"{item}_title")
            option['title_size'] = st.number_input("Title label size", min_value=0, value=20, key=f"{item}_title_size")

        with col2:
            option['xlabel'] = st.text_input('X labels', value='Simulations', label_visibility="visible",
                                             key=f"{item}_xlabel")
            option['xticksize'] = st.number_input("Xtick label size", min_value=0, value=17, key=f"{item}_xlabelsize")

        with col3:
            option['ylabel'] = st.text_input('Y labels', value='References', label_visibility="visible",
                                             key=f"{item}_ylabel")
            option['yticksize'] = st.number_input("Ytick label size", min_value=0, value=17, key=f"{item}_ylabelsize")

        with col4:
            option['fontsize'] = st.number_input("Fontsize", min_value=0, value=17, key=f"{item}_fontsize")
            option['axes_linewidth'] = st.number_input("axes linewidth", min_value=0, value=1,
                                                       key=f"{item}_axes_linewidth")
        st.divider()
        with st.expander("More info", expanded=False):
            col1, col2, col3, col4 = st.columns(4)

            option["x_rotation"] = col1.number_input(f"x rotation", min_value=-90, max_value=90, value=45,
                                                     key=f'{item}_x_rotation')
            option['x_ha'] = col2.selectbox('x ha', ['right', 'left', 'center'], key=f'{item}_x_ha',
                                            index=0, placeholder="Choose an option", label_visibility="visible")

            option["y_rotation"] = col3.number_input(f"y rotation", min_value=-90, max_value=90, value=45,
                                                     key=f'{item}_y_rotation')
            option['y_ha'] = col4.selectbox('y ha', ['right', 'left', 'center'], key=f'{item}_y_ha',
                                            index=0, placeholder="Choose an option", label_visibility="visible")
            option['ticks_format'] = col1.selectbox('Tick Format',
                                                    ['%f', '%G', '%.1f', '%.1G', '%.2f', '%.2G',
                                                     '%.3f', '%.3G'],
                                                    index=2, placeholder="Choose an option", label_visibility="visible",
                                                    key=f"{item}_ticks_format")

            option['cmap'] = col2.selectbox('Colorbar',
                                            ['coolwarm', 'coolwarm_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn',
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
                                             'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'copper', 'copper_r',
                                             'cubehelix', 'cubehelix_r', 'flag', 'flag_r',
                                             'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey',
                                             'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow',
                                             'gist_rainbow_r', 'gray', 'gray_r',
                                             'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r',
                                             'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r',
                                             'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow',
                                             'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer',
                                             'summer_r',
                                             'terrain', 'terrain_r', 'viridis', 'viridis_r', 'winter',
                                             'winter_r'], index=0, placeholder="Choose an option", key=f'{item}_cmap',
                                            label_visibility="visible")

            option["colorbar_position"] = col3.selectbox('colorbar position', ['horizontal', 'vertical'],
                                                         key=f"{item}_colorbar_position",
                                                         index=0, placeholder="Choose an option",
                                                         label_visibility="visible")
            option["extend"] = col4.selectbox(f"colorbar extend", ['neither', 'both', 'min', 'max'],
                                              index=0, placeholder="Choose an option", label_visibility="visible",
                                              key=f"{item}_extend")
            col1, col2, col3 = st.columns(3)
            option['xlimit_on'] = col1.toggle('Setting the max-min value manually', value=False,
                                              key=f'{item}_limit_on')
            if option['xlimit_on']:
                df = pd.read_csv(dir_path, sep='\s+', header=0)
                df.set_index('Item', inplace=True)
                df = df.iloc[:, 1:]
                max_value, min_value = df.max().max(), df.min().min()
                option["max"] = col3.number_input(f"x ticks max", key=f"{item}_max",
                                                  value=max_value)
                option["min"] = col2.number_input(f"x ticks min", key=f"{item}_min",
                                                  value=min_value)
            else:
                option["max"] = 1.
                option["min"] = 0.

        def get_cases(items, title):
            case_item = {}
            for item in items:
                case_item[item] = True
            with st.popover(title, use_container_width=True):
                st.subheader(f"Showing {title}", divider=True)
                if title != 'cases':
                    for item in case_item:
                        case_item[item] = st.checkbox(item.replace("_", " "), key=f"{item}__heatmap",
                                                      value=case_item[item])
                else:
                    for item in case_item:
                        case_item[item] = st.checkbox(item, key=f"{item}__heatmap",
                                                      value=case_item[item])
            return [item for item, value in case_item.items() if value]

        items = [k for k in selected_items]
        cases = list(
            set([value for key in selected_items for value in sim['general'][f"{key}_sim_source"] if value]))
        col1, col2 = st.columns(2)
        with col1:
            items = get_cases(items, 'Selected items')
        with col2:
            cases = get_cases(cases, 'cases')

        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            option["x_wise"] = st.number_input(f"X Length", min_value=0, value=10, key=f"{item}_x_wise")
            option['saving_format'] = st.selectbox('Image saving format', ['png', 'jpg', 'eps'],
                                                   index=1, placeholder="Choose an option", label_visibility="visible",
                                                   key=f"{item}_saving_format")
        with col2:
            option["y_wise"] = st.number_input(f"y Length", min_value=0, value=6, key=f"{item}_y_wise")
            option['font'] = st.selectbox('Image saving format',
                                          ['Times new roman', 'Arial', 'Courier New', 'Comic Sans MS', 'Verdana',
                                           'Helvetica',
                                           'Georgia', 'Tahoma', 'Trebuchet MS', 'Lucida Grande'],
                                          index=0, placeholder="Choose an option", label_visibility="visible",
                                          key=f"{item}_font")
        with col3:
            option['dpi'] = st.number_input(f"Figure dpi", min_value=0, value=300, key=f"{item}_dpi")
    draw_scenarios_scores_comparison_heat_map(dir_path, score, items, cases, option)


def draw_LC_based_heat_map(option, file, selected_metrics, selected_item, score, ref_source, sim_source, item, items):
    # st.write( selected_item, score, ref_source, sim_source, item)
    # Convert the data to a DataFrame
    # read the data from the file using csv, remove the first row, then set the index to the first column
    df = pd.read_csv(file, sep='\s+', skiprows=1, header=0)
    df.set_index('FullName', inplace=True)
    # Select the desired metrics
    if selected_item is not None:
        selected_metrics = list(selected_metrics)
        df_selected = df.loc[selected_metrics]
    else:
        df_selected = df.loc[items]
        selected_metrics = items

    font = {'family': option['font']}
    matplotlib.rc('font', **font)

    params = {'backend': 'ps',
              'axes.linewidth': option['axes_linewidth'],
              'font.size': option['fontsize'],
              'xtick.labelsize': option['xticksize'],
              'xtick.direction': 'out',
              'ytick.labelsize': option['yticksize'],
              'grid.linewidth': 0.2,
              'ytick.direction': 'out',
              'savefig.bbox': 'tight',
              'axes.unicode_minus': False,
              'text.usetex': False}
    rcParams.update(params)
    # Create the heatmap using Matplotlib

    shorter = {
        'PFT_groupby':
            {
                "bare_soil": "BS",
                "needleleaf_evergreen_temperate_tree": "NETT",
                "needleleaf_evergreen_boreal_tree": "NEBT",
                "needleleaf_deciduous_boreal_tree": "NDBT",
                "broadleaf_evergreen_tropical_tree": "BETT",
                "broadleaf_evergreen_temperate_tree": "BETT",
                "broadleaf_deciduous_tropical_tree": "BDTT",
                "broadleaf_deciduous_temperate_tree": "BDTT",
                "broadleaf_deciduous_boreal_tree": "BDBT",
                "broadleaf_evergreen_shrub": "BES",
                "broadleaf_deciduous_temperate_shrub": "BDTS",
                "broadleaf_deciduous_boreal_shrub": "BDBS",
                "c3_arctic_grass": "C3AG",
                "c3_non-arctic_grass": "C3NAG",
                "c4_grass": "C4G",
                "c3_crop": "C3C",
                "Overall": 'Overall'
            },
        'IGBP_groupby': {
            "evergreen_needleleaf_forest": 'ENF',
            "evergreen_broadleaf_forest": 'EBF',
            "deciduous_needleleaf_forest": 'DNF',
            "deciduous_broadleaf_forest": 'DBF',
            "mixed_forests": 'MF',
            "closed_shrubland": 'CSH',
            "open_shrublands": 'OSH',
            "woody_savannas": 'WSA',
            "savannas": 'SAV',
            "grasslands": 'GRA',
            "permanent_wetlands": 'WET',
            "croplands": 'CRO',
            "urban_and_built_up": 'URB',
            "cropland_natural_vegetation_mosaic": 'CVM',
            "snow_and_ice": 'SNO',
            "barren_or_sparsely_vegetated": 'BSV',
            "water_bodies": 'WAT',
            "Overall": 'Overall',
        }
    }

    if score == 'scores':
        fig, ax = plt.subplots(figsize=(option['x_wise'], option['y_wise']))
        im = ax.imshow(df_selected, cmap=option['cmap'], vmin=0, vmax=1)
        # Add colorbar
        # cbar = ax.figure.colorbar(im, ax=ax, label='Score', shrink=0.5)
        # Add labels and title
        ax.set_yticks(range(len(df_selected.index)))
        ax.set_xticks(range(len(df_selected.columns)))
        ax.set_yticklabels([index.replace('_', ' ') for index in df_selected.index], rotation=option['y_rotation'],
                           ha=option['y_ha'])

        if option["x_ticklabel"] == 'Normal':
            ax.set_xticklabels([columns.replace('_', ' ').title() for columns in df_selected.columns],
                               rotation=option['x_rotation'],
                               ha=option['x_ha'])
        else:
            ax.set_xticklabels([shorter[item][column] for column in df_selected.columns], rotation=option['x_rotation'],
                               ha=option['x_ha'])

        ax.set_ylabel(option['ylabel'], fontsize=option['yticksize'] + 1)
        ax.set_xlabel(option['xlabel'], fontsize=option['xticksize'] + 1)
        ax.set_title(option['title'], fontsize=option['title_size'])

        # Add numbers to each cell
        for i in range(len(df_selected.index)):
            for j in range(len(df_selected.columns)):
                ax.text(j, i, f'{df_selected.iloc[i, j]:{option["ticks_format"][1:]}}', ha='center', va='center',
                        color='white' if df_selected.iloc[i, j] > 0.8 else 'black' or df_selected.iloc[i, j] < 0.2,
                        fontsize=option['fontsize'])
        if option['show_colorbar']:
            pos = ax.get_position()  # .bounds
            left, right, bottom, width, height = pos.x0, pos.x1, pos.y0, pos.width, pos.height

            if option["colorbar_position"] == 'vertical':
                cbar_ax = fig.add_axes([right + 0.05, bottom, 0.03, height])  # right + 0.2
            else:
                xticks = ax.get_xticklabels()
                max_xtick_height = 0
                for xtick in xticks:
                    bbox = xtick.get_window_extent()  # 获取每个 xtick 的包围框
                    bbox_transformed = bbox.transformed(fig.transFigure.inverted())  # 将像素转换为图坐标
                    max_xtick_height = max(max_xtick_height, bbox_transformed.height)

                if len(option['xlabel']) != 0:
                    xlabel = ax.xaxis.label
                    bbox = xlabel.get_window_extent()  # 获取每个 xtick 的包围框
                    bbox_transformed = bbox.transformed(fig.transFigure.inverted())  # 将像素转换为图坐标
                    x_height = bbox_transformed.height
                    cbar_ax = fig.add_axes([left + width / 6, bottom - max_xtick_height - x_height - 0.1, width / 3 * 2, 0.05])
                else:
                    cbar_ax = fig.add_axes([left / 6, bottom - max_xtick_height - 0.1, width / 3 * 2, 0.05])
            cbar = fig.colorbar(im, cax=cbar_ax, orientation=option['colorbar_position'], extend='neither')
    elif len(df_selected.index) == 1 and score != 'scores':
        fig, ax = plt.subplots(figsize=(option['x_wise'], option['y_wise']))
        metric = df_selected.index[0]
        import glob
        files = glob.glob(f'{option["path"]}/{selected_item}_ref_{ref_source}_sim_{sim_source}_{metric}*.nc')
        datasets = [xr.open_dataset(file) for file in files]
        for t, ds in enumerate(datasets):
            datasets[t] = ds.expand_dims(dim={'time': [t]})  # 为每个文件添加一个新的'time'维度

        combined_dataset = xr.concat(datasets, dim='time')
        quantiles = combined_dataset.quantile([0.05, 0.2, 0.8, 0.95], dim=['time', 'lat', 'lon'])
        # consider 0.05 and 0.95 value as the max/min value
        custom_vmin_vmax = {}

        if metric in ['bias', 'percent_bias', 'rSD', 'PBIAS_HF', 'PBIAS_LF']:
            custom_vmin_vmax[metric] = [quantiles[metric][0].values, quantiles[metric][-1].values,
                                        quantiles[metric][2].values, quantiles[metric][1].values]
        elif metric in ['KGE', 'KGESS', 'correlation', 'kappa_coeff', 'rSpearman']:
            custom_vmin_vmax[metric] = [-1, 1, 0.8, -0.8]
        elif metric in ['NSE', 'LNSE', 'ubNSE', 'rNSE', 'wNSE', 'wsNSE']:
            custom_vmin_vmax[metric] = [quantiles[metric][0].values, 1, 0.8, quantiles[metric][1].values]
        elif metric in ['RMSE', 'CRMSD', 'MSE', 'ubRMSE', 'nRMSE', 'mean_absolute_error', 'ssq', 've',
                        'absolute_percent_bias']:
            custom_vmin_vmax[metric] = [-1, quantiles[metric][-1].values, quantiles[metric][2].values, -0.8]
        else:
            custom_vmin_vmax[metric] = [0, 1, 0.8, 0.2]

        if not option['cmap']:
            option['cmap'] = 'coolwarm'

        vmin, vmax = custom_vmin_vmax[metric][0], custom_vmin_vmax[metric][1]
        x1, x2 = custom_vmin_vmax[metric][2], custom_vmin_vmax[metric][3]
        im = ax.imshow(df_selected, cmap=option['cmap'], vmin=vmin, vmax=vmax)
        for j in range(df_selected.shape[1] + 1):
            ax.axvline(j - 0.5, color='white', linewidth=0.5)  # 垂直线
        ax.set_yticks(range(len(df_selected.index)))
        ax.set_xticks(range(len(df_selected.columns)))
        ax.set_yticklabels([index.replace('_', ' ') for index in df_selected.index], rotation=option['y_rotation'],
                           ha=option['y_ha'])
        if option["x_ticklabel"] == 'Normal':
            ax.set_xticklabels([columns.replace('_', ' ').title() for columns in df_selected.columns],
                               rotation=option['x_rotation'],
                               ha=option['x_ha'])
        else:
            item = option['groupby']
            ax.set_xticklabels([shorter[item][column] for column in df_selected.columns], rotation=option['x_rotation'],
                               ha=option['x_ha'])

        ax.set_ylabel('Metrics', fontsize=option['yticksize'] + 1)
        ax.set_xlabel(option['xlabel'], fontsize=option['xticksize'] + 1)

        if len(option['title']) == 0:
            option['title'] = f'Heatmap of {lb}'
        ax.set_title(option['title'], fontsize=option['title_size'])

        for i in range(len(df_selected.index)):
            for j in range(len(df_selected.columns)):
                ax.text(j, i, f'{df_selected.iloc[i, j]:{option["ticks_format"][1:]}}', ha='center', va='center',
                        color='white' if df_selected.iloc[i, j] > x1 else 'black' or df_selected.iloc[i, j] < x2,
                        fontsize=option['fontsize'])

        pos = ax.get_position()  # .bounds
        left, right, bottom, width, height = pos.x0, pos.x1, pos.y0, pos.width, pos.height
        xlabel = ax.xaxis.label
        xticks = ax.get_xticklabels()
        max_xtick_height = 0
        for xtick in xticks:
            bbox = xtick.get_window_extent()  # 获取每个 xtick 的包围框
            bbox_transformed = bbox.transformed(fig.transFigure.inverted())  # 将像素转换为图坐标
            max_xtick_height = max(max_xtick_height, bbox_transformed.height)
        if xlabel is not None:
            bbox = xlabel.get_window_extent()  # 获取每个 xtick 的包围框
            bbox_transformed = bbox.transformed(fig.transFigure.inverted())  # 将像素转换为图坐标
            x_height = bbox_transformed.height
            cbar_ax = fig.add_axes(
                [left + width / 6, bottom - max_xtick_height - x_height - 0.4, width / 3 * 2, 0.2])
        else:
            cbar_ax = fig.add_axes([left + width / 6, bottom - max_xtick_height - 0.1, width / 3 * 2, 0.04])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal',
                            extend=option['extend'])
    else:
        fig, axes = plt.subplots(nrows=len(df_selected.index), ncols=1, figsize=(option['x_wise'], option['y_wise']), sharex=True)
        fig.text(0.03, 0.5, 'Metrics', va='center', rotation='vertical', fontsize=option['yticksize'] + 1)
        plt.subplots_adjust(hspace=0.01)
        # get the minimal and maximal value
        custom_vmin_vmax = {}
        for i, (metric, row_data) in enumerate(df_selected.iterrows()):
            if metric in ['bias', 'mae', 'ubRMSE', 'apb', 'RMSE', 'L', 'percent_bias', 'apb']:
                import glob
                files = glob.glob(f'{option["path"]}/{selected_item}_ref_{ref_source}_sim_{sim_source}_{metric}*.nc')
                datasets = [xr.open_dataset(file) for file in files]
                for t, ds in enumerate(datasets):
                    datasets[t] = ds.expand_dims(dim={'time': [t]})  # 为每个文件添加一个新的'time'维度

                combined_dataset = xr.concat(datasets, dim='time')
                quantiles = combined_dataset.quantile([0.05, 0.2, 0.8, 0.95], dim=['time', 'lat', 'lon'])
                # consider 0.05 and 0.95 value as the max/min value
                vmin = quantiles[metric][0].values
                vmax = quantiles[metric][-1].values
                x1 = quantiles[metric][2].values
                x2 = quantiles[metric][1].values
                del quantiles, combined_dataset, datasets
                custom_vmin_vmax[metric] = [vmin, vmax, x1, x2]
            else:
                custom_vmin_vmax[metric] = [-1, 1, 0.8, -0.8]

        for i, (row_name, row_data) in enumerate(df_selected.iterrows()):
            vmin, vmax = custom_vmin_vmax[row_name][0], custom_vmin_vmax[row_name][1]
            x1, x2 = custom_vmin_vmax[row_name][2], custom_vmin_vmax[row_name][3]
            im = axes[i].imshow(row_data.values.reshape(1, -1), cmap=option['cmap'],
                                vmin=vmin, vmax=vmax)
            # Add numbers to each cell
            for j, value in enumerate(row_data):
                axes[i].text(j, 0, f'{df_selected.iloc[i, j]:{option["ticks_format"][1:]}}', ha='center', va='center',
                             color='white' if df_selected.iloc[i, j] > x1 or df_selected.iloc[i, j] < x2 else 'black',
                             fontsize=option['fontsize'])
                axes[i].axvline(j - 0.5, color='white', linewidth=0.5)  # 垂直线

            if option['show_colorbar']:
                pos = axes[i].get_position()  # .bounds
                left, right, bottom, width, height = pos.x0, pos.x1, pos.y0, pos.width, pos.height
                cbar_ax = fig.add_axes([right + 0.02, bottom + height / 2, height / 3 * 2, height / 4])
                cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', extend=option['extend'])
                cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])
                cbar.set_ticklabels([f'{vmin:.1f}', f'{(vmin + vmax) / 2:.1f}', f'{vmax:.1f}'])
                cbar.ax.tick_params(labelsize=18)

            if i < len(df_selected.index) - 1:
                axes[i].get_xaxis().set_visible(False)

            if i == 0:
                axes[i].spines['bottom'].set_visible(False)
            elif 0 < i < len(df_selected.index) - 1:
                axes[i].spines['top'].set_visible(False)
                axes[i].spines['bottom'].set_visible(False)
            else:
                axes[i].spines['top'].set_visible(False)

            axes[i].set_yticks([0])
            axes[i].set_yticklabels([selected_metrics[i].replace('_', ' ')], rotation=option['y_rotation'], ha=option['y_ha'])

        # 设置 x 轴标签
        axes[-1].set_xticks(np.arange(len(df_selected.columns)))
        if option["x_ticklabel"] == 'Normal':
            axes[-1].set_xticklabels([columns.replace('_', ' ').title() for columns in df_selected.columns],
                                     rotation=option['x_rotation'],
                                     ha=option['x_ha'])
        else:
            axes[-1].set_xticklabels([shorter[item][column] for column in df_selected.columns], rotation=option['x_rotation'],
                                     ha=option['x_ha'])

        axes[-1].set_xlabel(option['xlabel'], fontsize=option['xticksize'] + 1)
        axes[0].set_title(option['title'], fontsize=option['title_size'])

    file2 = f'{selected_item}_{sim_source}___{ref_source}_{score}'

    st.pyplot(fig)

    # 将图像保存到 BytesIO 对象
    buffer = BytesIO()
    fig.savefig(buffer, format=option['saving_format'], dpi=option['dpi'])
    buffer.seek(0)
    buffer.seek(0)
    st.download_button('Download image', buffer, file_name=f'{file2}_{item}.{option["saving_format"]}',
                       mime=f"image/{option['saving_format']}",
                       type="secondary", disabled=False, use_container_width=False, key=item)


def make_LC_based_heat_map(item, file, selected_item, score, sim_source, ref_source, dir_path, metrics, scores):
    option = {}
    option['path'] = os.path.dirname(file)

    with st.container(height=None, border=True):
        col1, col2, col3, col4 = st.columns((3.5, 3, 3, 3))
        with col1:
            option['title'] = st.text_input('Title', value=f'Heatmap of {score}', label_visibility="visible",
                                            key=f"{item}_title")
            option['title_size'] = st.number_input("Title label size", min_value=0, value=20, key=f"{item}_title_size")

        with col2:
            option['xlabel'] = st.text_input('X labels', value=sim_source, label_visibility="visible",
                                             key=f"{item}_xlabel")
            option['xticksize'] = st.number_input("X ticks size", min_value=0, value=17, key=f"{item}_xlabelsize")

        with col3:
            option['ylabel'] = st.text_input('Y labels', value=ref_source, label_visibility="visible",
                                             key=f"{item}_ylabel")
            option['yticksize'] = st.number_input("Y ticks size", min_value=0, value=17, key=f"{item}_ylabelsize")

        with col4:
            option['fontsize'] = st.number_input("Fontsize", min_value=0, value=17, step=1, key=f"{item}_fontsize",
                                                 help='Control label size on each ceil')
            option['axes_linewidth'] = st.number_input("axes linewidth", min_value=0, value=1,
                                                       key=f"{item}_axes_linewidth")

        col1, col2, col3 = st.columns((2, 1, 1))

        def get_cases(items, title):
            case_item = {}
            for item in items:
                case_item[item] = True
            with st.popover(f"Select {title}", use_container_width=True):
                st.subheader(f"Showing {title}", divider=True)
                for item in case_item:
                    case_item[item] = st.checkbox(item.replace("_", " "), key=f"{item}__heatmap_groupby",
                                                  value=case_item[item])
                selected = [item for item, value in case_item.items() if value]
                if len(selected) > 0:
                    return selected
                else:
                    st.error('You must choose one item!')

        if score == 'scores':
            selected_metrics = [k for k, v in scores.items() if v]
        else:
            selected_metrics = [k for k, v in metrics.items() if v]
        with col1:
            selected_metrics = get_cases(selected_metrics, score.title())

        st.divider()

        set_colorbar = st.expander("More info", expanded=False)
        col1, col2, col3, col4 = set_colorbar.columns(4)

        option["x_rotation"] = col1.number_input(f"x rotation", min_value=-90, max_value=90, value=45,
                                                 key=f"{item}_x_rotation")
        option['x_ha'] = col2.selectbox('x ha', ['right', 'left', 'center'],
                                        index=0, placeholder="Choose an option", label_visibility="visible",
                                        key=f"{item}_x_ha")
        option["y_rotation"] = col3.number_input(f"y rotation", min_value=-90, max_value=90, value=45,
                                                 key=f"{item}_y_rotation")
        option['y_ha'] = col4.selectbox('y ha', ['right', 'left', 'center'],
                                        index=0, placeholder="Choose an option", label_visibility="visible",
                                        key=f"{item}_y_ha")

        col1, col2, col3 = set_colorbar.columns(3)
        option['show_colorbar'] = col1.toggle('Showing colorbar?', value=True, key=f"{item}colorbar_on")
        option['cmap'] = col2.selectbox('Colorbar',
                                        ['coolwarm', 'coolwarm_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r',
                                         'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r',
                                         'Grays', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges',
                                         'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r',
                                         'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r',
                                         'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r',
                                         'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r',
                                         'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r',
                                         'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r',
                                         'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r',
                                         'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r',
                                         'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'copper', 'copper_r',
                                         'cubehelix', 'cubehelix_r', 'flag', 'flag_r',
                                         'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey',
                                         'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow',
                                         'gist_rainbow_r', 'gray', 'gray_r',
                                         'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r',
                                         'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r',
                                         'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow',
                                         'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r',
                                         'terrain', 'terrain_r', 'viridis', 'viridis_r', 'winter',
                                         'winter_r'], index=0, placeholder="Choose an option", key=f'{item}_cmap',
                                        label_visibility="visible")

        if option['show_colorbar']:
            if score == 'metrics':
                option["extend"] = col3.selectbox(f"colorbar extend", ['neither', 'both', 'min', 'max'],
                                                  index=1, placeholder="Choose an option", label_visibility="visible",
                                                  key=f"{item}_extend")
            else:
                option["colorbar_position"] = col3.selectbox('colorbar position', ['horizontal', 'vertical'],
                                                             key=f"{item}_colorbar_position",
                                                             index=0, placeholder="Choose an option",
                                                             label_visibility="visible")
        col1, col2, col3 = set_colorbar.columns(3)
        option["x_ticklabel"] = col1.selectbox('X tick labels Format', ['Normal', 'Shorter'],  # 'Season',
                                               index=1, placeholder="Choose an option", label_visibility="visible",
                                               key=f"{item}_x_ticklabel")
        option['ticks_format'] = col2.selectbox('Tick Format',
                                                ['%f', '%G', '%.1f', '%.1G', '%.2f', '%.2G',
                                                 '%.3f', '%.3G'],
                                                index=4, placeholder="Choose an option", label_visibility="visible",
                                                key=f"{item}_ticks_format")
        st.divider()

        if score == 'scores':
            items = [k for k, v in scores.items() if v]
        else:
            items = [k for k, v in metrics.items() if v]

        col1, col2, col3 = st.columns(3)
        with col1:
            if item == 'PFT_groupby':
                x_value = 17
            else:
                x_value = 18
            option["x_wise"] = st.number_input(f"X Length", min_value=0, value=x_value, key=f"{item}_x_wise")
            option['saving_format'] = st.selectbox('Image saving format', ['png', 'jpg', 'eps'],
                                                   index=1, placeholder="Choose an option",
                                                   label_visibility="visible",
                                                   key=f"{item}_saving_format")
        with col2:
            try:
                option["y_wise"] = st.number_input(f"y Length", min_value=0, value=len(selected_metrics),
                                                   key=f"{item}_y_wise")
            except:
                option["y_wise"] = 1
            option['font'] = st.selectbox('Image saving format',
                                          ['Times new roman', 'Arial', 'Courier New', 'Comic Sans MS', 'Verdana',
                                           'Helvetica',
                                           'Georgia', 'Tahoma', 'Trebuchet MS', 'Lucida Grande'],
                                          index=0, placeholder="Choose an option", label_visibility="visible",
                                          key=f"{item}_font")
        with col3:
            option['dpi'] = st.number_input(f"Figure dpi", min_value=0, value=300, key=f"{item}_dpi'")

    if selected_metrics:
        draw_LC_based_heat_map(option, file, selected_metrics, selected_item, score, ref_source, sim_source, item, items)
