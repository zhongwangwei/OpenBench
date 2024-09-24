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
def make_scenarios_scores_comparison_heat_map(file,score):
   # Convert the data to a DataFrame
   #read the data from the file using csv, remove the first row, then set the index to the first column
   df = pd.read_csv(file, sep='\s+', header=0)
   #exclude the first column
   df.set_index('Item', inplace=True)
   ref_dataname= df.iloc[:, 0:]
   df = df.iloc[:, 1:]

   # Create the heatmap using Matplotlib
   fig, ax = plt.subplots(figsize=(10, 6))
   im = ax.imshow(df, cmap='coolwarm', vmin=0, vmax=1)

   # Add colorbar
   # Add labels and title
   ax.set_yticks(range(len(df.index)))
   ax.set_xticks(range(len(df.columns)))
   ax.set_yticklabels(df.index, rotation=45, ha='right')
   ax.set_xticklabels(df.columns, rotation=45, ha='right')

   #ax.set_ylabel('Metrics')
   #ax.set_xlabel('Land Cover Classes')
   ax.set_title(f'Heatmap of {score}')
   # Add numbers to each cell
   for i in range(len(df.index)):
      for j in range(len(df.columns)):
            ax.text(j, i, f'{df.iloc[i, j]:.2f}', ha='center', va='center', color='white' if df.iloc[i, j] > 0.8 or df.iloc[i, j] < 0.2 else 'black', fontsize=12)


   for i in range(len(ref_dataname.index)):
      ax.text(len(df.columns)-0.3, i, f'{ref_dataname.iloc[i, 0]}', ha='left', va='center', color='black')
      #add the small ticks to the right of the heatmap
      ax.text(len(df.columns)-0.5, i, '-', ha='left', va='center', color='black')
   #add the colorbar, and make it shrink to the size of the heatmap, and the location is at the right of reference data name 
   
   # Create dynamically positioned and sized colorbar
   cbar_ax = fig.add_axes([0.3, -0.05, 0.5, 0.05]) 
   cbar = fig.colorbar(im, cax=cbar_ax, label='Score', orientation='horizontal')

   plt.tight_layout()



   file2=file[:-4]
   plt.savefig(f'{file2}_heatmap.png', format='png', dpi=300)
