import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams


def make_scenarios_scores_comparison_heat_map(file, score, option):
    # Convert the data to a DataFrame
    # read the data from the file using csv, remove the first row, then set the index to the first column
    df = pd.read_csv(file, sep='\s+', header=0)
    # exclude the first column
    df.set_index('Item', inplace=True)
    ref_dataname = df.iloc[:, 0:]
    df = df.iloc[:, 1:]

    font = {'family': 'DejaVu Sans'}
    # font = {'family': option['font']}
    matplotlib.rc('font', **font)
    # Create the heatmap using Matplotlib
    params = {'backend': 'ps',
              'axes.linewidth': option['axes_linewidth'],
              'font.size': option['fontsize'],
              'xtick.labelsize': option['xtick'],
              'xtick.direction': 'out',
              'ytick.labelsize': option['ytick'],
              'grid.linewidth': 0.2,
              'ytick.direction': 'out',
              'savefig.bbox': 'tight',
              'axes.unicode_minus': False,
              'text.usetex': False}
    rcParams.update(params)

    fig, ax = plt.subplots(figsize=(option['x_wise'], option['y_wise']))
    if option['vmin_max_on']:
        vmin, vmax = option['vmin'], option['vmax']
    else:
        vmin, vmax = 0, 1
    if not option['cmap']:
        option['cmap'] = 'coolwarm'
    im = ax.imshow(df, cmap=option['cmap'], vmin=vmin, vmax=vmax)

    # Add colorbar
    # Add labels and title
    ax.set_yticks(range(len(df.index)))
    ax.set_xticks(range(len(df.columns)))
    ax.set_yticklabels([y.replace('_', ' ') for y in df.index], rotation=option['y_rotation'], ha=option['y_ha'])
    ax.set_xticklabels(df.columns, rotation=option['x_rotation'], ha=option['x_ha'])

    ax.set_ylabel(option['ylabel'], fontsize=option['ytick'] + 1)
    ax.set_xlabel(option['xlabel'], fontsize=option['xtick'] + 1)

    if len(option['title']) == 0:
        option['title'] = f'Heatmap of {score}'
    ax.set_title(option['title'], fontsize=option['title_size'])

    # Add numbers to each cell
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            ax.text(j, i, f'{df.iloc[i, j]:{option["ticks_format"]}}', ha='center', va='center',
                    color='white' if df.iloc[i, j] > 0.8 or df.iloc[i, j] < 0.2 else 'black', fontsize=option['fontsize'])

    max_tick_width = 0
    for i in range(len(ref_dataname.index)):
        text_obj = ax.text(len(df.columns) - 0.3, i, f'{ref_dataname.iloc[i, 0]}', ha='left', va='center', color='black',
                           fontsize=option['ytick'])
        # add the small ticks to the right of the heatmap
        ax.text(len(df.columns) - 0.5, i, '-', ha='left', va='center', color='black')
        text_position = text_obj.get_position()
        bbox = text_obj.get_window_extent()
        bbox_in_fig_coords = bbox.transformed(fig.transFigure.inverted())
        max_tick_width = max(max_tick_width, bbox_in_fig_coords.width)
    # add the colorbar, and make it shrink to the size of the heatmap, and the location is at the right of reference data name

    # Create dynamically positioned and sized colorbar
    pos = ax.get_position()  # .bounds
    left, right, bottom, width, height = pos.x0, pos.x1, pos.y0, pos.width, pos.height
    if not option['colorbar_position_set']:
        if option["colorbar_position"] == 'vertical':
            if len(df.index) < 6:
                cbar_ax = fig.add_axes([right + max_tick_width + 0.05, bottom, 0.03, height])  # right + 0.2
            else:
                cbar_ax = fig.add_axes([right + max_tick_width + 0.05, bottom + height / 6, 0.03, height / 3 * 2])  # right + 0.2
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
                if len(df.columns) < 6:
                    cbar_ax = fig.add_axes([left, bottom - max_xtick_height - x_height - 0.1, width, 0.05])
                else:
                    cbar_ax = fig.add_axes([left + width / 6, bottom - max_xtick_height - x_height - 0.1, width / 3 * 2, 0.05])
            else:
                if len(df.columns) < 6:
                    cbar_ax = fig.add_axes([left, bottom - max_xtick_height - x_height - 0.1, width, 0.05])
                else:
                    cbar_ax = fig.add_axes([left + width / 6, bottom - max_xtick_height - 0.1, width / 3 * 2, 0.05])
    else:
        cbar_ax = fig.add_axes(option["colorbar_left"], option["colorbar_bottom"], option["colorbar_width"],
                               option["colorbar_height"])

    cbar = fig.colorbar(im, cax=cbar_ax, label=option['colorbar_label'], orientation=option['colorbar_position'],
                        extend=option['extend'])

    # plt.tight_layout()

    file2 = file[:-4]
    plt.savefig(f'{file2}_heatmap.{option["saving_format"]}', format=f'{option["saving_format"]}', dpi=option['dpi'])
