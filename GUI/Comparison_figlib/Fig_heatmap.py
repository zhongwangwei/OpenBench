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


def make_scenarios_scores_comparison_heat_map(file, score, items, cases, option):
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
    im = ax.imshow(df, cmap=option['cmap'], vmin=0, vmax=1)

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

    cbar = fig.colorbar(im, cax=cbar_ax, orientation=option[
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


def make_LC_based_heat_map(option, file, selected_metrics, selected_item, score, ref_source, sim_source, item, items):
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
        path = f'{option["path"]}/{sim_source}___{ref_source}/'
        files = glob.glob(f'{path}{selected_item}_ref_{ref_source}_sim_{sim_source}_{metric}*.nc')
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
        path = f'{option["path"]}/{sim_source}___{ref_source}/'

        fig, axes = plt.subplots(nrows=len(df_selected.index), ncols=1, figsize=(option['x_wise'], option['y_wise']), sharex=True)
        fig.text(0.03, 0.5, 'Metrics', va='center', rotation='vertical', fontsize=option['yticksize'] + 1)
        plt.subplots_adjust(hspace=0)
        # get the minimal and maximal value
        custom_vmin_vmax = {}
        for i, (metric, row_data) in enumerate(df_selected.iterrows()):
            if metric in ['bias', 'mae', 'ubRMSE', 'apb', 'RMSE', 'L', 'percent_bias', 'apb']:
                import glob
                files = glob.glob(f'{path}{selected_item}_ref_{ref_source}_sim_{sim_source}_{metric}*.nc')
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
