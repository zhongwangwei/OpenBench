import numpy as np
import math
from scipy.stats import gaussian_kde
import matplotlib.cm as cm
import matplotlib
import matplotlib as mpl
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from io import BytesIO
import streamlit as st


def make_scenarios_comparison_Ridgeline_Plot(option, evaluation_item, ref_source, sim_sources, datasets_filtered, varname):
    # st.write(option)
    font = {'family': option['font']}
    matplotlib.rc('font', **font)

    params = {'backend': 'ps',
              'axes.linewidth': option['axes_linewidth'],
              'font.size': option["fontsize"],
              'xtick.labelsize': option['xticksize'],
              'xtick.direction': 'out',
              'ytick.direction': 'out',
              'savefig.bbox': 'tight',
              'axes.unicode_minus': False,
              'text.usetex': False}
    rcParams.update(params)

    fig, axes = plt.subplots(figsize=(option['x_wise'], option['y_wise']))

    n_plots = len(sim_sources)
    global_min = option['global_min']
    global_max = option['global_max']
    x_range = np.linspace(global_min, global_max, 200)
    dx = x_range[1] - x_range[0]
    # Adjust these parameters to control spacing and overlap
    y_shift_increment = 0.5
    scale_factor = 0.8

    if option['colormap']:
        cmap = mpl.colormaps[f'{option["cmap"]}'].resampled(256)(range(256))
        R = np.broadcast_to(cmap[:, 0], (200, 256))
        G = np.broadcast_to(cmap[:, 1], (200, 256))
        B = np.broadcast_to(cmap[:, 2], (200, 256))
        alpha = np.full_like(R, 1)
        rgba = np.stack((R, G, B, alpha), axis=0)

    for i, (data, sim_source) in enumerate(zip(datasets_filtered, sim_sources)):

        filtered_data = data[(data >= global_min) & (data <= global_max)]

        kde = gaussian_kde(filtered_data)
        y_range = kde(x_range)

        # Scale and shift the densities
        y_range = y_range * scale_factor / y_range.max()
        y_shift = i * y_shift_increment

        # Plot the KDE

        if option['colormap']:
            path = axes.fill_between(x_range, y_shift, y_range + y_shift, facecolor='none', lw=0.35, edgecolor='k', zorder=n_plots - i)
            patch = PathPatch(path._paths[0], visible=False, transform=axes.transData)
            ai = axes.imshow(rgba.transpose((1, 2, 0)), extent=[0, 1, 0, 1], transform=axes.transAxes, alpha=option['alpha'])
            ai.set_clip_path(patch)
            axes.plot(x_range, y_range + y_shift, color='black', linewidth=option['linewidth'])
        else:
            axes.fill_between(x_range, y_shift, y_range + y_shift, alpha=option['MARKERS'][sim_source]['alpha'],
                              color=option['MARKERS'][sim_source]['lineColor'], zorder=n_plots - i)
            axes.plot(x_range, y_range + y_shift, color='black', linewidth=option['MARKERS'][sim_source]['linewidth'])

        # Add labels
        axes.text(global_min, y_shift + 0.2, sim_source, fontweight='bold', ha='left', va='center')

        # Calculate and plot median
        median = np.median(filtered_data)
        index_closest = (np.abs(x_range - median)).argmin()
        y_target = y_range[index_closest]
        axes.vlines(median, y_shift, y_shift + y_target, color='black', linestyle=option['vlinestyle'],
                    linewidth=option['vlinewidth'], zorder=n_plots + 1)

        # Add median value text
        axes.text(median, y_shift + y_target, f'{median:.2f}', ha='center', va='bottom', fontsize=option["fontsize"],
                  zorder=n_plots + 2)

    # Customize the plot
    axes.set_yticks([])
    axes.set_xlabel(option['xticklabel'], fontsize=option['xticksize'] + 1)
    axes.set_title(option['title'], fontsize=option['title_fontsize'],pad=30)

    # Remove top and right spines
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['left'].set_visible(False)

    # Extend the bottom spine to the left
    axes.spines['bottom'].set_position(('data', -0.2))

    # Set y-axis limits
    axes.set_ylim(-0.2, (n_plots - 1) * y_shift_increment + scale_factor)
    axes.set_xlim(global_min-dx, global_max+dx)

    st.pyplot(fig)

    file2 = f"Ridgeline_Plot_{evaluation_item}_{ref_source}_{varname}"
    # 将图像保存到 BytesIO 对象
    buffer = BytesIO()
    fig.savefig(buffer, format=option['saving_format'], dpi=option['dpi'])
    buffer.seek(0)
    buffer.seek(0)
    st.download_button('Download image', buffer, file_name=f'{file2}.{option["saving_format"]}',
                       mime=f"image/{option['saving_format']}",
                       type="secondary", disabled=False, use_container_width=False)
