import os
import numpy as np
import xarray as xr
import pandas as pd
from joblib import Parallel, delayed
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams
import scipy.stats as sts
from io import BytesIO
import streamlit as st


def make_scenarios_comparison_Single_Model_Performance_Index(option, file, evaluation_items, ref_nml):
    st.json(option)
    # Read the SMPI data
    font = {'family': option['font']}
    matplotlib.rc('font', **font)

    params = {'backend': 'ps',
              'axes.linewidth': 1,
              'font.size': option['fontsize'],
              'xtick.labelsize': option['xticksize'],
              'xtick.direction': 'out',
              'ytick.direction': 'out',
              'savefig.bbox': 'tight',
              'axes.unicode_minus': False,
              'text.usetex': False}
    rcParams.update(params)
    df = pd.read_csv(file, sep='\t')

    # Prepare the subplot grid
    n_items = len(evaluation_items)

    fig, axs = plt.subplots(n_items, 1, figsize=(option['x_wise'], option['y_wise']), sharey=True, squeeze=False)  # sharey=True,

    fig.subplots_adjust(hspace=option["hspace"], wspace=option["wspace"])

    # Calculate overall min and max I² values for consistent x-axis range
    min_I2 = max(0, df['SMPI'].min() - 0.5)
    max_I2 = min(5, df['SMPI'].max() + 0.5)

    # Create a color map for subplots
    # color_map = plt.cm.get_cmap('tab20')

    for i, item in enumerate(evaluation_items):
        ref_sources = ref_nml['general'][f'{item}_ref_source']
        if isinstance(ref_sources, str):
            ref_sources = [ref_sources]
        ax = axs[i, 0]

        for j, ref_source in enumerate(ref_sources):

            # Filter data for this item and reference source
            item_data = df[(df['Item'] == item) & (df['Reference'] == ref_source)]
            if item_data.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                continue

            I2_values = item_data['SMPI'].tolist()
            labels = item_data['Simulation'].tolist()

            # Calculate confidence intervals
            mean = np.mean(I2_values)
            sem = sts.sem(I2_values)
            conf_interval = sem * sts.t.ppf((1 + 0.95) / 2., len(I2_values) - 1)
            sizes = [150 * conf_interval] * len(I2_values)  # Reduced circle size

            # Get color for this subplot
            # color = color_map(i * max_ref_sources + j)

            # Plot
            for k, (value, size) in enumerate(zip(I2_values, sizes)):
                ax.scatter(value, 0, s=size*option["n"], facecolors=option['COLORS'][item], edgecolors=option['COLORS'][item], alpha=0.8)
                ax.scatter(value, 0, s=size * 0.01*option["n"], facecolors='white', edgecolors='none')

            # Annotate labels
            for k, value in enumerate(I2_values):
                ax.annotate(
                    str(k + 1),  # Use numbers starting from 1
                    (value, 0),
                    xytext=(0, 18),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    fontsize=option['fontsize'],
                    rotation=45
                )

            # Mean (black circle)
            ax.scatter(mean, 0, color="black", s=50, marker="o", alpha=0.6)
            ax.scatter(mean, 0, color="white", s=50 * 0.01, marker="o", alpha=0.6)
            # Add mean label
            ax.annotate(
                'Mean',
                (mean, 0),
                xytext=(0, -15),  # Position the label below the mean point
                textcoords='offset points',
                ha='center',
                va='top',
                # fontsize=8,
                # fontweight='bold',
                rotation=-45
            )

            # Set up axes and ticks
            ax.spines["bottom"].set_position("zero")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.xaxis.set_ticks_position("bottom")
            ax.tick_params(axis="x", direction="inout", which="both", length=20, width=1.5)  # , labelsize=8
            ax.tick_params(axis="x", which="minor", length=10)
            ax.set_xlim([min_I2, max_I2])
            ax.set_xticks(np.arange(min_I2, max_I2 + 0.5, 0.5))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(0.25))

            # Set titles
            # if i == 0:
            #    ax.set_title(f"Reference: {ref_source}", fontsize=16)
            if not option["var_loc"]:
                if j == 0:
                    ax.text(option["x_posi"], option["y_posi"], item.replace('_', ' '), rotation=option["y_rotation"], va='center', ha=option['y_ha'],
                            transform=ax.transAxes, fontsize=option['yticksize'])
        if option["var_loc"]:
            ax.text(option["x_posi"], option["y_posi"], item.replace('_', ' '), rotation=option["x_rotation"], va='center', ha=option['x_ha'],
                        transform=ax.transAxes, fontsize=option['xlabelsize'])

    # Overall title
    # fig.suptitle("Single Model Performance Index Comparison", fontsize=16, y=1.02)

    # X-axis label
    if not option["var_loc"]:
        fig.supxlabel(option['xlabel'], ha=option['x_ha'], fontsize=option['xticksize']+1, rotation=option["x_rotation"])
    else:
        fig.supylabel(option['ylabel'], ha=option['y_ha'], fontsize=option['yticksize']+1, rotation=option["y_rotation"])

    # plt.savefig(f"{basedir}/output/comparisons/Single_Model_Performance_Index/SMPI_comparison_plot_comprehensive.png", dpi=300,
    #             bbox_inches='tight')

    st.pyplot(fig)

    # 将图像保存到 BytesIO 对象
    buffer = BytesIO()
    fig.savefig(buffer, format=option['saving_format'], dpi=option['dpi'])
    buffer.seek(0)
    buffer.seek(0)
    st.download_button('Download image', buffer, file_name=f'SMPI_comparison_plot_comprehensive.{option["saving_format"]}',
                       mime=f"image/{option['saving_format']}",
                       type="secondary", disabled=False, use_container_width=False)
