import os
import numpy as np
import xarray as xr
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as st
def make_scenarios_comparison_Single_Model_Performance_Index(basedir, evaluation_items, ref_nml, sim_nml):
    # Read the SMPI data
    data_path = f"{basedir}/output/comparisons/Single_Model_Performance_Index/SMPI_comparison.txt"
    df = pd.read_csv(data_path, sep='\t')
    
    # Prepare the subplot grid
    n_items = len(evaluation_items)
    max_ref_sources = max(len(ref_nml['general'][f'{item}_ref_source']) if isinstance(ref_nml['general'][f'{item}_ref_source'], list) 
                          else 1 for item in evaluation_items)
    
    #fig, axs = plt.subplots(n_items, max_ref_sources, figsize=(6*max_ref_sources, 2.5*n_items), squeeze=False)
    fig, axs = plt.subplots(n_items, 1, figsize=(10, 4), sharey=True, squeeze=False)
    colors = cm.get_cmap("tab10", n_items)

    fig.subplots_adjust(hspace=-0.9, wspace=0.1)

    
    # Calculate overall min and max I² values for consistent x-axis range
    min_I2 = max(0, df['SMPI'].min() - 0.5)
    max_I2 = min(5, df['SMPI'].max() + 0.5)
    
    # Create a color map for subplots
    #color_map = plt.cm.get_cmap('tab20')
    
    for i, item in enumerate(evaluation_items):
        ref_sources = ref_nml['general'][f'{item}_ref_source']
        if isinstance(ref_sources, str):
            ref_sources = [ref_sources]
        ax = axs[i,0]

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
            sem = st.sem(I2_values)
            conf_interval = sem * st.t.ppf((1 + 0.95) / 2., len(I2_values) - 1)
            sizes = [150 * conf_interval] * len(I2_values)  # Reduced circle size

            
            # Get color for this subplot
            #color = color_map(i * max_ref_sources + j)
            
            # Plot
            for k, (value, size) in enumerate(zip(I2_values, sizes)):
                ax.scatter(value, 0, s=size, facecolors=colors(i), edgecolors=colors(i), alpha=0.8)
                ax.scatter(value, 0, s=size*0.01, facecolors='white', edgecolors='none')
            
            # Annotate labels
            for k, value in enumerate(I2_values):
                ax.annotate(
                    str(k + 1),  # Use numbers starting from 1
                    (value, 0),
                    xytext=(0, 18),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    rotation=45
                )
            
            
            # Mean (black circle)
            ax.scatter(mean, 0, color="black", s=50, marker="o", alpha=0.6)
            ax.scatter(mean, 0, color="white", s=50*0.01, marker="o", alpha=0.6)
            # Add mean label
            ax.annotate(
                'Mean',
                (mean, 0),
                xytext=(0, -15),  # Position the label below the mean point
                textcoords='offset points',
                ha='center',
                va='top',
                fontsize=8,
                #fontweight='bold',
                rotation=-45

            )
            
            # Set up axes and ticks
            ax.spines["bottom"].set_position("zero")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.xaxis.set_ticks_position("bottom")
            ax.tick_params(axis="x", direction="inout", which="both", length=20, width=1.5, labelsize=8)
            ax.tick_params(axis="x", which="minor", length=10)
            ax.set_xlim([min_I2, max_I2])
            ax.set_xticks(np.arange(min_I2, max_I2 + 0.5, 0.5))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(0.25))
            
            # Set titles
            #if i == 0:
            #    ax.set_title(f"Reference: {ref_source}", fontsize=16)
            if j == 0:
                ax.text(-0.1, 0.5, item, rotation=90, va='center', ha='right', transform=ax.transAxes, fontsize=12)


    
    # Overall title
    #fig.suptitle("Single Model Performance Index Comparison", fontsize=16, y=1.02)
    
    # X-axis label
    fig.text(0.5, 0.01, "Single Model Performance Index", ha='center', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{basedir}/output/comparisons/Single_Model_Performance_Index/SMPI_comparison_plot_comprehensive.png", dpi=300, bbox_inches='tight')
    plt.close()