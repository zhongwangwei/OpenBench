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

def make_scenarios_comparison_Whisker_Plot(basedir,evaluation_item,ref_source,sim_sources,varname, datasets_filtered):
    # Create a figure and axis
    plt.figure(figsize=(8, 6))
    xlabel = f'{varname}'
    ylabel = 'Value'

    # Create the whisker plot
    plt.boxplot(datasets_filtered, labels=[f'{i}' for i in sim_sources])

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Whisker Plot of {evaluation_item}")

    output_file_path = f"{basedir}/Whisker_Plot_{evaluation_item}_{ref_source}_{varname}.jpg"
    plt.savefig(output_file_path, format='jpg', dpi=300, bbox_inches='tight')
    
    plt.close()  # Close the figure to free up memory
    
    return

