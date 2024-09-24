import xarray as xr
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from   matplotlib import colors
from   mpl_toolkits.basemap import Basemap
from   matplotlib import rcParams
from   matplotlib import ticker
import math
import matplotlib.colors as clr
import itertools
from matplotlib import cm
from scipy.stats import gaussian_kde

def make_scenarios_comparison_Kernel_Density_Estimate(basedir,evaluation_item,ref_source,sim_sources,varname, datasets_filtered):

    
    #create a figure and axis
    plt.figure(figsize=(8, 6))
    xlabel=f'{varname}'
    ylabel='KDE Density'

    # Generate colors using a colormap
    colors = cm.tab10(np.linspace(0, 1, len(datasets_filtered)))  # Adjust colormap as desired
    # Create a list to store line objects for the legend
    lines = []
    # Plot each KDE
    for i, data in enumerate(datasets_filtered):            
        kde = gaussian_kde(data)
        covariance_matrix = kde.covariance 
        covariance_matrix += np.eye(covariance_matrix.shape[0]) * 1e-6 # Regularization
        kde.covariance = covariance_matrix
        x_values = np.linspace(data.min(), data.max(), 100)
        density = kde(x_values)

        # Store the line object
        line, = plt.plot(x_values, density, color=colors[i])
        lines.append(line)  # Add the line object to the list
        plt.fill_between(x_values, density, color=colors[i], alpha=0.5)

    # Add labels and legend
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(lines, [f'{i}' for i in (sim_sources)])
    plt.title(f"Kernel Density Estimate of {evaluation_item}")
    # plt.title("(b) Sensible Heat Flux", fontsize=18, pad=25, loc='left')
    output_file_path = f"{basedir}/Kernel_Density_Estimate_{evaluation_item}_{ref_source}_{varname}.jpg"
    plt.savefig(output_file_path, format='jpg', dpi=300, bbox_inches='tight') 
    del data, datasets_filtered, colors, lines, kde, covariance_matrix, x_values, density, line 
    return
