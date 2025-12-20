import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties

try:
    from openbench.util.Mod_Converttype import Convert_Type
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from openbench.util.Mod_Converttype import Convert_Type
from .Fig_toolbox import get_index, convert_unit, get_colormap


def _read_comparison_file(file):
    """
    Read comparison file with fallback logic.
    Try .csv first, then .txt if not found.
    Auto-detect separator (tab or comma).
    """
    file_to_read = file

    if not os.path.exists(file):
        if file.endswith('.csv'):
            alt_file = file[:-4] + '.txt'
            if os.path.exists(alt_file):
                logging.info(f"File {file} not found, using {alt_file}")
                file_to_read = alt_file
        elif file.endswith('.txt'):
            alt_file = file[:-4] + '.csv'
            if os.path.exists(alt_file):
                logging.info(f"File {file} not found, using {alt_file}")
                file_to_read = alt_file

    if not os.path.exists(file_to_read):
        raise FileNotFoundError(f"Neither {file} nor alternative extension found")

    with open(file_to_read, 'r') as f:
        first_line = f.readline()

    sep = '\t' if '\t' in first_line else ','
    df = pd.read_csv(file_to_read, sep=sep, header=0)
    # Remove index name to prevent it from appearing in visualizations
    df.index.name = None
    return df


def make_scenarios_comparison_radar_map(file, score, option):
    # Read file with fallback and auto-detection
    df = _read_comparison_file(file)
    df = Convert_Type.convert_Frame(df)
    list1 = df.iloc[:, 0].tolist()
    list2 = df.iloc[:, 1].tolist()
    ref_dataname = [f"{item1.replace('_', ' ')}\n({item2})" for item1, item2 in zip(list1, list2)]
    df.set_index('Item', inplace=True)
    df = df.iloc[:, 1:].astype('float32')
    min_value, max_value = np.nanmin(df.values), np.nanmax(df.values)
    df = df.fillna(0)

    cmap, mticks, norm, bnd, extend = get_index(min_value, max_value)
    if mticks[-1] > 0.5:
        mticks_interval = 0.2
    else:
        mticks_interval = mticks[1] - mticks[0]
    mticks = np.arange(0, max_value + mticks_interval, mticks_interval).round(1)

    if not option['vmin_max_on']:
        option['vmax'], option['vmin'] = mticks[-1], mticks[0]

    params = {'backend': 'ps',
              'figure.figsize': [option['x_wise'], option['y_wise']],
              'figure.dpi': option['dpi'],
              'figure.autolayout': True,
              'savefig.format': option['saving_format'],
              'font.family': option['font_family'],
              'font.size': option['font_size'],
              'axes.edgecolor': 'white',
              'axes.titlesize': option['titlesize'],
              'axes.titleweight': 'bold',
              'axes.titlepad': 10.0,
              'axes.linewidth': option['linewidth'],
              'patch.linewidth': option['patch_linewidth'],
              'lines.linestyle': option['lines_linestyle'],
              'xtick.labelsize': option['xtick_labelsize'],
              'xtick.major.pad': 5,
              'legend.handlelength': 2.0,
              'legend.handleheight': 2.0,
              'legend.fontsize': option['legend_fontsize'],
              'legend.frameon': True,
              }
    rcParams.update(params)

    colors = ['#2887c5', '#d35625', '#e5b51f', '#80328c', '#50bee2', '#9e1d32', '#75a641', '#2b2f79']

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    angles = [n / float(len(ref_dataname)) * 2 * np.pi for n in range(len(ref_dataname))]
    angles += angles[:1]

    plt.title(score.replace('_', " "))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(ref_dataname)
    ax.set_rlabel_position(0)
    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')
    ax.set_yticks([])
    ax.set_yticklabels("")
    ax.set_ylim(mticks[0], mticks[-1])
    for i in range(len(mticks)):
        grid_values = [mticks[i]] * len(angles)
        ax.plot(angles, grid_values, color='grey')
        if i != len(mticks) - 1:
            if mticks[i] - 0.02 > 0:
                ax.text(np.deg2rad(0), mticks[i] - 0.02, str(mticks[i]), ha='center', va='center')
            else:
                ax.text(np.deg2rad(0), mticks[i], str(mticks[i]), ha='center', va='center')

    for i in range(len(df.columns)):
        values = df.iloc[:, i].tolist()
        values += values[:1]
        ax.plot(angles, values, colors[i], linestyle='-')
        ax.fill(angles, values, colors[i], alpha=0.8)

    legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=df.columns[i]) for i in range(len(df.columns))]
    title_font = FontProperties(weight='bold')
    plt.legend(
        handles=legend_elements,
        bbox_to_anchor=(1, 0.1),
        title=option['legend_title'],
        title_fontproperties=title_font,
    )
    file2 = file[:-4]
    plt.savefig(f'{file2}_radarmap.{option["saving_format"]}')
