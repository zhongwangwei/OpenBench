import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
try:
    from openbench.util.Mod_Converttype import Convert_Type
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from openbench.util.Mod_Converttype import Convert_Type

import os
import sys
# Add the local visualization path for cmaps
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
import cmaps
from .Fig_toolbox import get_index, convert_unit, get_colormap
 
def make_scenarios_scores_comparison_heat_map(file, score, option):
    # Convert the data to a DataFrame
    # read the data from the file using csv, remove the first row, then set the index to the first column
    # df = pd.read_csv(file, sep=r'\s+', header=0)
    # df = Convert_Type.convert_Frame(df)
    df = pd.read_csv(file)
    # exclude the first column
    df.set_index('Item', inplace=True)
    ref_dataname = df.iloc[:, 0:]
    df = df.iloc[:, 1:].astype('float32')

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

    option['x_wise'] = len(df.index)
    option['y_wise'] = len(df.columns)
    # Add minimum size constraints for small datasets
    if option['x_wise'] < 3:
        option['x_wise'] = max(3, option['x_wise'])
    if option['y_wise'] < 3:
        option['y_wise'] = max(3, option['y_wise'])
        
    # Adjust font sizes for small datasets
    if len(df.index) <= 2 or len(df.columns) <= 2:
        option['fontsize'] = min(option['fontsize'], 16)
        option['xtick'] = min(option['xtick'], 18)
        option['ytick'] = min(option['ytick'], 18)
        option['title_size'] = min(option['title_size'], 24)

    fig, ax = plt.subplots(figsize=((option['x_wise']+len(df.columns))*0.9, (option['y_wise']+len(df.index))*0.75))
    if option['vmin_max_on']:
        vmin, vmax = option['vmin'], option['vmax']
    else:
        vmin, vmax = 0, 1

    im = ax.imshow(df, cmap=cmaps.MPL_RdBu_r, vmin=vmin, vmax=vmax)

    # Add colorbar
    # Add labels and title
    ref_dataname_name = ref_dataname.iloc[:, 0].tolist()
    ax.set_yticks(range(len(df.index)))
    ax.set_xticks(range(len(df.columns)))
    ax.set_yticklabels(
        [f"{y.replace('_', ' ')}\n({x.replace('_', ' ')})" for y, x in zip(df.index, ref_dataname_name)],
        rotation=option['y_rotation'],
        ha=option['y_ha']
    )
    ax.set_xticklabels([x for x in df.columns], rotation=option['x_rotation'], ha=option['x_ha'])

    ax.set_ylabel(option['ylabel'], fontsize=option['ytick'] + 1, weight='bold')
    ax.set_xlabel(option['xlabel'], fontsize=option['xtick'] + 1, weight='bold')

    title = option['title']
    if len(option['title']) == 0:
        # title = f'Heatmap of {score}'
        title = f'{score}'
    ax.set_title(title.replace("_", " "), fontsize=option['title_size'], weight='bold')

    # Add numbers to each cell
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            ax.text(j, i, f'{df.iloc[i, j]:{option["ticks_format"]}}', ha='center', va='center',
                    color='white' if df.iloc[i, j] > 0.8 or df.iloc[i, j] < 0.2 else 'black', fontsize=option['fontsize'])

    max_tick_width = 0
    # for i in range(len(ref_dataname.index)):
    #     ref_dataname_name = ref_dataname.iloc[i, 0].replace("_", " ")
    #     text_obj = ax.text(len(df.columns) - 0.3, i, f'{ref_dataname_name}', ha='left', va='center', color='black',
    #                        fontsize=option['ytick'])
    #     # add the small ticks to the right of the heatmap
    #     ax.text(len(df.columns) - 0.5, i, '-', ha='left', va='center', color='black')
    #     text_position = text_obj.get_position()
    #     bbox = text_obj.get_window_extent()
    #     bbox_in_fig_coords = bbox.transformed(fig.transFigure.inverted())
    #     max_tick_width = max(max_tick_width, bbox_in_fig_coords.width)
    # add the colorbar, and make it shrink to the size of the heatmap, and the location is at the right of reference data name

    # Create dynamically positioned and sized colorbar
    pos = ax.get_position()  # .bounds
    left, right, bottom, width, height = pos.x0, pos.x1, pos.y0, pos.width, pos.height
    if not option['colorbar_position_set']:
        if option["colorbar_position"] == 'vertical':
            if len(df.index) < 6:
                cbar_ax = fig.add_axes([right + max_tick_width + 0.05, bottom, 0.03, height]) 
            else:
                cbar_ax = fig.add_axes([right + max_tick_width +0.01 , bottom, 0.03, height])
        else:
            xlabel = ax.xaxis.label
            xticks = ax.get_xticklabels()
            max_xtick_height = 0
            for xtick in xticks:
                bbox = xtick.get_window_extent() 
                bbox_transformed = bbox.transformed(fig.transFigure.inverted())  
                max_xtick_height = max(max_xtick_height, bbox_transformed.height)
            if xlabel is not None:
                bbox = xlabel.get_window_extent()  
                bbox_transformed = bbox.transformed(fig.transFigure.inverted()) 
                x_height = bbox_transformed.height
                if len(df.columns) < 6:
                    cbar_ax = fig.add_axes([left, bottom - max_xtick_height - x_height - 0.1, width, 0.02])
                else:
                    cbar_ax = fig.add_axes([left + width / 6, bottom - max_xtick_height - x_height - 0.1, width / 3 * 2, 0.02])
            else:
                if len(df.columns) < 6:
                    cbar_ax = fig.add_axes([left, bottom - max_xtick_height - x_height - 0.1, width, 0.02])
                else:
                    cbar_ax = fig.add_axes([left + width / 6, bottom - max_xtick_height - 0.1, width / 3 * 2, 0.02])
    else:
        cbar_ax = fig.add_axes(option["colorbar_left"], option["colorbar_bottom"], option["colorbar_width"],
                               option["colorbar_height"])

    cbar = fig.colorbar(im, cax=cbar_ax, label=option['colorbar_label'], orientation=option['colorbar_position'],
                        extend=option['extend'])

    # plt.tight_layout()

    file2 = file[:-4]
    plt.savefig(f'{file2}_heatmap.{option["saving_format"]}', format=f'{option["saving_format"]}', dpi=option['dpi'])
