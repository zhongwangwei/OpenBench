import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from   matplotlib import rcParams


font = {'family': 'DejaVu Sans'}
matplotlib.rc('font', **font)

params = {'backend': 'ps',
            'axes.labelsize': 10,
            'grid.linewidth': 0.2,
            'font.size': 12,
            'legend.fontsize': 12,
            'legend.frameon': False,
            'xtick.labelsize': 12,
            'xtick.direction': 'out',
            'ytick.labelsize': 12,
            'ytick.direction': 'out',
            'savefig.bbox': 'tight',
            'axes.unicode_minus': False,
            'text.usetex': False}
rcParams.update(params)

def make_LC_based_heat_map(file,selected_metrics,lb,vmin,vmax):
    selected_metrics = list(selected_metrics)
    # Convert the data to a DataFrame
    #read the data from the file using csv, remove the first row, then set the index to the first column
    df = pd.read_csv(file, sep='\s+', skiprows=1, header=0)
    df.set_index('FullName', inplace=True)
    # Select the desired metrics
    #selected_metrics = ['nBiasScore', 'nRMSEScore', 'nPhaseScore', 'nIavScore', 'nSpatialScore', 'overall_score']
    df_selected = df.loc[selected_metrics]

    # Create the heatmap using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(df_selected, cmap='coolwarm', vmin=0, vmax=1)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, label='Score', shrink=0.5)
    # Add labels and title
    ax.set_yticks(range(len(df_selected.index)))
    ax.set_xticks(range(len(df_selected.columns)))
    ax.set_yticklabels(selected_metrics, rotation=45, ha='right')
    ax.set_xticklabels(df_selected.columns, rotation=45, ha='right')
    ax.set_ylabel('Metrics')
    ax.set_xlabel('Land Cover Classes')
    ax.set_title('Heatmap of Selected Metrics')

    # Add numbers to each cell
    for i in range(len(df_selected.index)):
        for j in range(len(df_selected.columns)):
            ax.text(j, i, f'{df_selected.iloc[i, j]:.2f}', ha='center', va='center', color='white' if df_selected.iloc[i, j] > 0.8 or df_selected.iloc[i, j] < 0.2 else 'black', fontsize=8)

    plt.tight_layout()

    file2=file[:-4]
    plt.savefig(f'{file2}_heatmap.png', format='png', dpi=300)
    #plt.show() 