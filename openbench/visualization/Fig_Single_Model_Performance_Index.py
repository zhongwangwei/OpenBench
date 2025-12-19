import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib import rcParams
try:
    from openbench.util.Mod_Converttype import Convert_Type
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from openbench.util.Mod_Converttype import Convert_Type
import math
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def make_scenarios_comparison_Single_Model_Performance_Index(basedir, evaluation_items, ref_nml, sim_nml, option):
    # Read the SMPI data
    font = {'family': option['font']}
    matplotlib.rc('font', **font)

    params = {'backend': 'ps',
              'axes.linewidth': 1,
              'font.size': option['fontsize'],
              'xtick.labelsize': option['xtick'],
              'xtick.direction': 'out',
              # 'ytick.labelsize': option['ytick'],
              'ytick.direction': 'out',
              'savefig.bbox': 'tight',
              'axes.unicode_minus': False,
              'text.usetex': False}
    rcParams.update(params)

    data_path = f"{basedir}/comparisons/Single_Model_Performance_Index/SMPI_comparison.csv"
    # df = pd.read_csv(data_path, sep='\t')
    df = pd.read_csv(data_path, header=0)
    df = Convert_Type.convert_Frame(df)
    # Prepare the subplot grid
    n_items = len(evaluation_items)

    fig, axs = plt.subplots(n_items, 1, figsize=(option['x_wise'], option['y_wise']/3*n_items), sharey=True, squeeze=False)  # sharey=True,
    # MCOLORS = generate_colors(evaluation_items)

    fig.subplots_adjust(left=0., right=1, 
                    bottom=0, top=0.82, hspace=option["hspace"], wspace=option["wspace"])

    # Calculate overall min and max I² values for consistent x-axis range
    # min_I2 = max(0, df['SMPI'].min() - 0.5)
    # max_I2 = min(5, df['SMPI'].max() + 0.5)
    min_I2 = 0

    # Create a color map for subplots
    # color_map = plt.cm.get_cmap('tab20')

    for i, item in enumerate(evaluation_items):
        smpi_max = df[(df['Item'] == item)]['SMPI'].max()
        max_I2 = math.ceil(max(1, smpi_max))

        ref_sources = ref_nml['general'][f'{item}_ref_source']
        if isinstance(ref_sources, str):
            ref_sources = [ref_sources]
        ax = axs[i, 0]

        for j, ref_source in enumerate(ref_sources):

            # Filter data for this item and reference source
            item_data = df[(df['Item'] == item) & (df['Reference'] == ref_source)]

            if item_data.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes, fontsize=option['fontsize'])
                continue

            I2_values = item_data['SMPI'].tolist()
            labels = item_data['Simulation'].tolist()

            # Calculate confidence intervals
            mean = np.mean(I2_values)
            sem = st.sem(I2_values)
            conf_interval = sem * st.t.ppf((1 + 0.95) / 2., len(I2_values) - 1)
            sizes = [150 * conf_interval] * len(I2_values)  # Reduced circle size

            # Get color for this subplot

            # Plot
            for k, (value, size) in enumerate(zip(I2_values, sizes)):
                if np.isnan(size):
                    size = 2
                MCOLORS = generate_colors(k,j)
                ax.scatter(value, 0, s=size * option["n"], facecolors=MCOLORS["selected_color"]["Color"],
                           edgecolors=MCOLORS["selected_color"]["Color"],
                           alpha=option['markeraplha'])
                ax.scatter(value, 0, s=10, facecolors='white', edgecolors='none')

            ax.text(0+0.4*j, 1, f'ref{j+1}: {ref_source.replace("_"," ")}', ha='left', va='center', transform=ax.transAxes, fontsize=option['fontsize'])
            # Annotate labels
            # for k, value in enumerate(I2_values):
            #     ax.annotate(
            #         str(k + 1),  # Use numbers starting from 1
            #         (value, 0),
            #         xytext=(0, 18),
            #         textcoords='offset points',
            #         ha='center',
            #         va='bottom',
            #         fontsize=option['fontsize'],
            #         rotation=45
            #     )

            # Mean (black circle)
            MCOLORS = generate_colors(k+1,j)
            ax.scatter(mean, 0, color=MCOLORS["selected_color"]["Color"], s=50, marker="s", alpha=0.6)
            ax.scatter(mean, 0, color="white", s=2, marker="s", alpha=0.6)
            # Add mean label
            # ax.annotate(
            #     'Mean',
            #     (mean, 0),
            #     xytext=(0, -15),  # Position the label below the mean point
            #     textcoords='offset points',
            #     ha='center',
            #     va='top',
            #     # fontsize=8,
            #     # fontweight='bold',
            #     rotation=-45

            # )

            # Set up axes and ticks
            ax.spines["bottom"].set_position("zero")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.xaxis.set_ticks_position("bottom")
            ax.tick_params(axis="x", direction="inout", which="both", length=10, width=1.5)  # , labelsize=8
            ax.tick_params(axis="x", which="minor", length=5)
            ax.set_xlim([min_I2, max_I2])
            if max_I2 > 2:
                ax.set_xticks(np.arange(min_I2, max_I2 + 0.5, 0.5))
                ax.xaxis.set_minor_locator(plt.MultipleLocator(0.25))
            elif max_I2 > 1:
                ax.set_xticks(np.arange(min_I2, max_I2 + 0.25, 0.25))
                ax.xaxis.set_minor_locator(plt.MultipleLocator(0.125))
            else:
                ax.set_xticks(np.arange(min_I2, max_I2 + 0.125, 0.125))
                ax.xaxis.set_minor_locator(plt.MultipleLocator(0.0625))

            # Set titles
            # if i == 0:
            #    ax.set_title(f"Reference: {ref_source}", fontsize=16)
        ax.text(option["x_posi"], option["y_posi"], item.replace('_', ' '), rotation=option["y_rotation"], va='center',
                ha=option['y_ha'], transform=ax.transAxes, fontsize=option['ylabelsize'], weight='bold')

    # Overall title
        if i == (len(evaluation_items)-1):
            for k, value in enumerate(I2_values):
                legend_elements = []
                for j, ref_source in enumerate(ref_sources):
                
                    MCOLORS = generate_colors(k,j)
                    # legend_elements.append(Patch(facecolor=MCOLORS["selected_color"]["Color"], 
                    #                 linewidth=1.5, edgecolor='black', label=''))
                    legend_elements.append(Line2D([0], [0], marker='o', 
                              color=MCOLORS["selected_color"]["Color"], 
                              linestyle='None', 
                              markersize=24, 
                              markeredgecolor='None', 
                              markerfacecolor=MCOLORS["selected_color"]["Color"], 
                              label=''))
                    if k==0:
                        fig.text(0.04, -0.06-j*0.07, f"ref{str(j+1)}", va='center',
                            ha='right', fontsize=option['fontsize'])

                    # item_data = df[(df['Item'] == item) & (df['Reference'] == ref_source)]
                    # labels = item_data['Simulation'].tolist()


                fig.legend(
                    handles=legend_elements,
                    bbox_to_anchor=(0.125+k*0.07, 0.045),
                    frameon=False,
                    title=f'sim{str(k+1)}',
                    fontsize=option['fontsize'],
                    handlelength=2, 
                    handleheight=2,
                    handletextpad=0.5,  
                    labelspacing= 0.5
                )
                fig.text(0.701, -0.06-k*0.07, f"sim{str(k+1)}: {labels[k]}", va='center',
                        ha='left', fontsize=option['fontsize'])

            legend_elements = []
            for j, ref_source in enumerate(ref_sources):
                MCOLORS = generate_colors(k+1,j)
                # legend_elements.append(Patch(facecolor=MCOLORS["selected_color"]["Color"], 
                #                     linewidth=1.5, edgecolor='black', label=''))
                legend_elements.append(Line2D([0], [0], marker='s', 
                              color=MCOLORS["selected_color"]["Color"], 
                              linestyle='None', 
                              markersize=24, 
                              markeredgecolor='None', 
                              markerfacecolor=MCOLORS["selected_color"]["Color"], 
                              label=''))
            fig.legend(
                handles=legend_elements,
                bbox_to_anchor=(0.125+k*0.07+0.07, 0.045),
                frameon=False,
                title=f'mean',
                fontsize=option['fontsize'],
                handlelength=2, 
                handleheight=2,
                handletextpad=0.5,  
                labelspacing= 0.5
            )

    legend_elements = [Line2D([0], [0], marker='o', color='#bfbfbf', 
            label="$\\text{         point size} \\propto \\sigma_{\\text{CI}}$\n    (larger size indicates\n  greater variability across\nsimulations for a reference)",
            linestyle='None', markersize=20*len(ref_sources), markeredgecolor='None', markerfacecolor='#bfbfbf')]
    fig.legend(
        handles=legend_elements,
        bbox_to_anchor=(0.67, 0.015*len(ref_sources)),
        frameon=False,
        title=f'',
        fontsize=option['fontsize'],
        handlelength=2, 
        handleheight=2,
        handletextpad=0.5,  
        labelspacing= 0.5,
    )

    fig.suptitle("Single Model Performance Index Comparison", fontsize=16, weight="bold", y=0.95)

    plt.savefig(
        f'{basedir}/comparisons/Single_Model_Performance_Index/SMPI_comparison_plot_comprehensive.{option["saving_format"]}',
        format=f'{option["saving_format"]}', dpi=option['dpi'], bbox_inches='tight')
    plt.close()


def generate_colors(color_index, list_index,  name="selected_color"):
    """
    Generate a color dictionary based on list_index and color_index.
    
    Parameters:
    - list_index (int): Index to select a color list from colorslist (0 to 4).
    - color_index (int): Index to select a color from the chosen color list.
    - name (str): Optional name for the color dictionary key (default: "selected_color").
    
    Returns:
    - dict: A dictionary with the format {name: {"Color": color}}.
    """
    # Define color lists
    colors1 = ['#a30327', '#d53126', '#f0734b', '#fbae62', '#fde28e']
    colors2 = ['#303692', '#4376b7', '#76aecf', '#2c75ea', '#aadae7']
    colors3 = ['#727a5f', '#95a07c', '#b8c699', '#aadae7', '#e8ffc4']
    colors4 = ['#673e54', '#8d5573', '#b36c92', '#d983b1', '#ff99c9']
    colors5 = ['#b6a772', '#dcca89', '#fff09e', '#fff69e', '#fffc9e']
    colorslist = [colors1, colors2, colors3, colors4, colors5]
    
    # Validate inputs
    if not isinstance(list_index, int) or list_index < 0 or list_index >= len(colorslist):
        raise ValueError(f"list_index must be an integer between 0 and {len(colorslist)-1}")
    
    selected_colors = colorslist[list_index]
    
    if not isinstance(color_index, int) or color_index < 0 or color_index >= len(selected_colors):
        raise ValueError(f"color_index must be an integer between 0 and {len(selected_colors)-1}")
    
    # Select the color
    selected_color = selected_colors[color_index]
    
    # Create the output dictionary
    gcolors = {
        name: {
            "Color": selected_color
        }
    }
    
    return gcolors
