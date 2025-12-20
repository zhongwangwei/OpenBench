import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from scipy.stats import gaussian_kde


def make_scenarios_comparison_Ridgeline_Plot(basedir, evaluation_item, ref_source, sim_sources, varname, datasets_filtered,
                                             option):
    if varname != 'nSpatialScore':
        font = {'family': option['font']}
        matplotlib.rc('font', **font)

        params = {'backend': 'ps',
                  'axes.linewidth': option['axes_linewidth'],
                #   'font.size': option["fontsize"],
                  'xtick.labelsize': option['xtick'],
                  'xtick.direction': 'out',
                  'ytick.direction': 'out',
                  'savefig.bbox': 'tight',
                  'axes.unicode_minus': False,
                  'text.usetex': False}
        rcParams.update(params)

        n_plots = len(sim_sources)
        # create a figure and axis
        fig, axes = plt.subplots(figsize=(option['x_wise'], option['y_wise']*len(sim_sources)/2))

        # Generate colors using a colormap
        MLINES = generate_lines(sim_sources, option)

        # Find global min and max for x-axis
        def remove_outliers(data_list):
            q1, q3 = np.percentile(data_list, [1.5, 98.5])
            return [q1, q3]

        bound = [remove_outliers(d) for d in datasets_filtered]
        global_max = max([d[1] for d in bound])
        global_min = min([d[0] for d in bound])

        if varname in ['NSE', 'KGE', 'KGESS', 'ubKGE', 'rKGE', 'wKGE', 'wsKGE']:
            if global_min < -1:
                global_min = -1
            global_max = 1
        elif varname in ['MFM']:
            global_min = 0
            global_max = 1
        x_range = np.linspace(global_min, global_max, 200)
        dx = x_range[1] - x_range[0]
        # Adjust these parameters to control spacing and overlap
        y_shift_increment = 0.1
        scale_factor = 0.15

        for i, (data, sim_source) in enumerate(zip(datasets_filtered, sim_sources)):
            filtered_data = data
            if varname in ['KGE', 'NSE', 'KGESS']:
                filtered_data = np.where(data < -1, -1, data)
            elif varname in ['MFM']:
                filtered_data = np.where(data < 0, 0, data)

            kde = gaussian_kde(filtered_data)
            y_range = kde(x_range)

            # Scale and shift the densities
            y_range = y_range * scale_factor / y_range.max()
            y_shift = i * y_shift_increment

            # Plot the KDE
            axes.fill_between(x_range, y_shift, y_range + y_shift, color=MLINES[sim_source]['lineColor'],
                                alpha=MLINES[sim_source]['alpha'], zorder=n_plots - i)
            axes.plot(x_range, y_range + y_shift, color="k", 
                            alpha=MLINES[sim_source]['alpha'], linewidth=MLINES[sim_source]['linewidth']*1.5, )

            # Add labels
            axes.text(global_min-(global_max-global_min)/100, y_shift, sim_source, 
                    fontsize=option["fontsize"], fontweight='bold', ha='right', va='center')
            # Calculate and plot median
            median = np.median(data)
            if varname in ['KGE', 'NSE', 'KGESS'] and median <= global_min:
                pass
            else:
                index_closest = (np.abs(x_range - median)).argmin()
                y_target = y_range[index_closest]
                axes.vlines(median, y_shift, y_shift + y_target, color='black', linestyle=option['vlinestyle'],
                            linewidth=option['vlinewidth']*1.5, zorder=n_plots + 1)

                # Add median value text
                axes.text(median, y_shift + y_target*1.02, f'{median:.2f}', ha='center', va='bottom', fontsize=option["fontsize"],
                            zorder=n_plots + 2)

        # Customize the plot
        axes.set_yticks([])
        xlabel = option['xlabel']
        title = option['title']

        if not option['xlabel']:
            xlabel = varname.replace('_', ' ')
            if varname == 'percent_bias':
                xlabel = varname.replace('_', ' ') + f' (showing value between [-100,100])'
        if not option['title']:
            title = f'{evaluation_item.replace("_", " ")}'
        axes.set_xlabel(xlabel, fontsize=option['xtick'] + 4, weight='bold')
        axes.tick_params(axis='x', color="#969696", width=1.5, length=4,which='major')  
        axes.set_title(title, fontsize=option['title_fontsize'], pad=30, weight='bold')

        # Remove top and right spines
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.spines['left'].set_visible(False)

        # Extend the bottom spine to the left
        axes.spines['bottom'].set_visible(False)

        # Set y-axis limits
        axes.set_ylim(0, (n_plots - 1) * y_shift_increment + scale_factor)
        axes.set_xlim(global_min - dx, global_max + dx)

        # Adjust layout and save
        plt.tight_layout()
        output_file_path = f"{basedir}/Ridgeline_Plot_{evaluation_item}_{ref_source}_{varname}.{option['saving_format']}"
        plt.savefig(output_file_path, format=f'{option["saving_format"]}', dpi=option['dpi'], bbox_inches='tight')

        # Clean up
        plt.close()
        return

def generate_lines(data_names, option):
    import itertools
    import matplotlib.colors as mcolors
    lines = {}
    # add colors and symbols
    hex_colors = ['#b1c5e1', '#c2b3d4', '#a7dc8f', '#f09f99', '#be9f98', 
                '#edbad0', '#dddb97', '#a9dae0', '#f8e9b9', '#7f7f7f','#f4be83',]
    # hex_colors = cm.vivid(np.linspace(0, 1, len(data_names) + 1))
    colors = itertools.cycle([mcolors.rgb2hex(color) for color in hex_colors])

    if not option['linewidth']:
        linewidth = 1.5
    else:
        linewidth = option['linewidth']

    if not option['alpha']:
        alpha = 0.3
    else:
        alpha = option['alpha']

    for name in data_names:
        color = next(colors)
        lines[name] = {
            "lineColor": color,
            "linewidth": linewidth,
            "alpha": alpha
        }
    return lines
