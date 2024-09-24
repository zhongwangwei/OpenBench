import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.cm as cm

def make_scenarios_comparison_Ridgeline_Plot(basedir, evaluation_item, ref_source, sim_sources, varname, datasets_filtered):
    fig, axes = plt.subplots(figsize=(10, 8))
    n_plots = len(datasets_filtered)
    
    # Generate colors using a colormap
    colors = cm.viridis(np.linspace(0, 1, n_plots))

    # Find global min and max for x-axis
    global_min = min(data.min() for data in datasets_filtered)
    global_max = max(data.max() for data in datasets_filtered)
    x_range = np.linspace(global_min, global_max, 200)

    # Adjust these parameters to control spacing and overlap
    y_shift_increment = 0.5
    scale_factor = 0.8

    for i, (data, sim_source) in enumerate(zip(datasets_filtered, sim_sources)):
        kde = gaussian_kde(data)
        y_range = kde(x_range)
        
        # Scale and shift the densities
        y_range = y_range * scale_factor / y_range.max()
        y_shift = i * y_shift_increment

        # Plot the KDE
        axes.fill_between(x_range, y_shift, y_range + y_shift, alpha=0.8, color=colors[i], zorder=n_plots - i)
        axes.plot(x_range, y_range + y_shift, color='black', linewidth=0.5)

        # Add labels
        axes.text(global_min, y_shift + 0.2, sim_source, fontweight='bold', ha='left', va='center')

        # Calculate and plot median
        median = np.median(data)
        axes.vlines(median, y_shift, y_shift + y_range.max(), color='black', linestyle=':', linewidth=1.5, zorder=n_plots + 1)
        
        # Add median value text
        axes.text(median, y_shift + y_range.max(), f'{median:.2f}', ha='center', va='bottom', fontsize=16, zorder=n_plots + 2)

    # Customize the plot
    axes.set_yticks([])
    axes.set_xlabel(varname)
    axes.set_title(f"Ridgeline Plot of {evaluation_item}")

    # Remove top and right spines
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['left'].set_visible(False)

    # Extend the bottom spine to the left
    axes.spines['bottom'].set_position(('data', -0.2))

    # Set y-axis limits
    axes.set_ylim(-0.2, (n_plots - 1) * y_shift_increment + scale_factor)

    # Adjust layout and save
    plt.tight_layout()
    output_file_path = f"{basedir}/Ridgeline_Plot_{evaluation_item}_{ref_source}_{varname}.jpg"
    plt.savefig(output_file_path, format='jpg', dpi=300, bbox_inches='tight')

    # Clean up
    plt.close()

    return