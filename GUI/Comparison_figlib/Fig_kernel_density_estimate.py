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
from matplotlib import cm
from scipy.stats import gaussian_kde

from io import BytesIO
import streamlit as st


def make_scenarios_comparison_Kernel_Density_Estimate(option, selected_item, ref_source, sim_sources, datasets_filtered, varname):
    font = {'family': option['font']}
    matplotlib.rc('font', **font)

    params = {'backend': 'ps',
              'axes.linewidth': option['axes_linewidth'],
              'font.size': option["fontsize"],
              'xtick.labelsize': option['xticksize'],
              'xtick.direction': 'out',
              'ytick.labelsize': option['yticksize'],
              'ytick.direction': 'out',
              'savefig.bbox': 'tight',
              'axes.unicode_minus': False,
              'text.usetex': False}
    rcParams.update(params)
    # Create the heatmap using Matplotlib
    # fig, ax = plt.subplots(figsize=(option['x_wise'], option['y_wise']))

    fig = plt.figure(figsize=(option['x_wise'], option['y_wise']))
    ax = fig.add_subplot(111)  # 添加子图

    lines = []

    for i, sim_source in enumerate(sim_sources):
        data = datasets_filtered[i]
        try:
            lower_bound, upper_bound = np.percentile(data, 5), np.percentile(data, 95)
            if varname in ['bias', 'rSD', 'PBIAS_HF', 'PBIAS_LF']:
                filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
            elif varname in ['KGE', 'KGESS', 'NSE', 'LNSE', 'ubNSE', 'rNSE', 'wNSE', 'wsNSE']:
                filtered_data = data[(data >= lower_bound)]
            elif varname in ['RMSE', 'CRMSD', 'MSE', 'ubRMSE', 'nRMSE', 'mean_absolute_error', 'ssq', 've',
                             'absolute_percent_bias']:
                filtered_data = data[(data <= upper_bound)]
            else:
                filtered_data = data


            kde = gaussian_kde(filtered_data)
            covariance_matrix = kde.covariance
            covariance_matrix += np.eye(covariance_matrix.shape[0]) * 1e-6  # Regularization
            kde.covariance = covariance_matrix

            x_values = np.linspace(filtered_data.min(), filtered_data.max(), 100)
            density = kde(x_values)

            # Store the line object
            line, = ax.plot(x_values, density, color=option['MARKERS'][sim_source]['lineColor'],
                            linestyle=option['MARKERS'][sim_source]['linestyle'],
                            linewidth=option['MARKERS'][sim_source]['linewidth'],
                            label=sim_source)
            lines.append(line)  # Add the line object to the list
            ax.fill_between(x_values, density, color=option['MARKERS'][sim_source]['lineColor'],
                            alpha=option['MARKERS'][sim_source]['alpha'])
        except Exception as e:
            st.error(f"{selected_item} {ref_source} {sim_source} {varname} Kernel Density Estimate failed!")

    if not option["legend_on"]:
        ax.legend(shadow=False, frameon=False, fontsize=option['fontsize'],
                  loc=option["loc"], ncol=option["ncol"])

    else:
        ax.legend( shadow=False, frameon=False, fontsize=option['fontsize'],
                  bbox_to_anchor=(option["bbox_to_anchor_x"], option["bbox_to_anchor_y"]), ncol=option["ncol"])

    if option['grid']:
        ax.grid(linestyle=option['grid_style'], alpha=0.7, linewidth=option['grid_linewidth'])  # 绘制图中虚线 透明度0.3
    if option['minmax']:
        ax.set_xlim(option['xmin'], option['xmax'])

    plt.xlabel(option['xticklabel'], fontsize=option['xticksize']+1)
    plt.ylabel(option['yticklabel'], fontsize=option['yticksize']+1)
    plt.title(option['title'], fontsize=option['title_fontsize'])


    try:
        del datasets_filtered, lines, kde, covariance_matrix, x_values, density, line
    except:
        del datasets_filtered, data, lines


    st.pyplot(fig)

    file2 = f"Kernel_Density_Estimate_{selected_item}_{ref_source}_{varname}"
    # 将图像保存到 BytesIO 对象
    buffer = BytesIO()
    fig.savefig(buffer, format=option['saving_format'], dpi=option['dpi'])
    buffer.seek(0)
    buffer.seek(0)
    st.download_button('Download image', buffer, file_name=f'{file2}.{option["saving_format"]}',
                       mime=f"image/{option['saving_format']}",
                       type="secondary", disabled=False, use_container_width=False)
