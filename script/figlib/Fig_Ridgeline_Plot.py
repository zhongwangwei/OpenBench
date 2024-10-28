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
                  'font.size': option["fontsize"],
                  'xtick.labelsize': option['xtick'],
                  'xtick.direction': 'out',
                  'ytick.direction': 'out',
                  'savefig.bbox': 'tight',
                  'axes.unicode_minus': False,
                  'text.usetex': False}
        rcParams.update(params)
        if varname != 'nSpatialScore':
            n_plots = len(sim_sources)
            # create a figure and axis
            fig, axes = plt.subplots(figsize=(option['x_wise'], option['y_wise']))

            # Generate colors using a colormap
            MLINES = generate_lines(sim_sources, option)

            # Find global min and max for x-axis
            def remove_outliers(data_list):
                q1, q3 = np.percentile(data_list, [1.5, 98.5])
                return [q1, q3]

            bound = [remove_outliers(d) for d in datasets_filtered]
            global_max = max([d[1] for d in bound])
            global_min = min([d[0] for d in bound])

            if varname in ['RMSE', 'CRMSD', 'MSE', 'ubRMSE', 'nRMSE', 'mean_absolute_error', 'ssq', 've',
                           'absolute_percent_bias']:
                global_min = global_min * 0 - 0.2
            elif varname in ['NSE', 'LNSE', 'ubNSE', 'rNSE', 'wNSE', 'wsNSE']:
                global_max = global_max * 0 + 0.2

            # global_min = min(data.min() for data in datasets_filtered)
            # global_max = max(data.max() for data in datasets_filtered)
            x_range = np.linspace(global_min, global_max, 200)

            # Adjust these parameters to control spacing and overlap
            y_shift_increment = 0.5
            scale_factor = 0.8

            for i, (data, sim_source) in enumerate(zip(datasets_filtered, sim_sources)):
                filtered_data = data[(data >= global_min) & (data <= global_max)]

                kde = gaussian_kde(filtered_data)
                y_range = kde(x_range)

                # Scale and shift the densities
                y_range = y_range * scale_factor / y_range.max()
                y_shift = i * y_shift_increment

                # Plot the KDE
                axes.fill_between(x_range, y_shift, y_range + y_shift, color=MLINES[sim_source]['lineColor'],
                                  alpha=MLINES[sim_source]['alpha'], zorder=n_plots - i)
                axes.plot(x_range, y_range + y_shift, color='black', linewidth=MLINES[sim_source]['linewidth'], )

                # Add labels
                axes.text(global_min, y_shift + 0.2, sim_source, fontweight='bold', ha='left', va='center')
                # Calculate and plot median
                median = np.median(filtered_data)
                index_closest = (np.abs(x_range - median)).argmin()
                y_target = y_range[index_closest]
                axes.vlines(median, y_shift, y_shift + y_target, color='black', linestyle=option['vlinestyle'],
                            linewidth=option['vlinewidth'], zorder=n_plots + 1)

                # Add median value text
                axes.text(median, y_shift + y_target, f'{median:.2f}', ha='center', va='bottom', fontsize=option["fontsize"],
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
                title = f'Ridgeline Plot of {evaluation_item.replace("_", " ")}'
            axes.set_xlabel(xlabel, fontsize=option['xtick'] + 1)
            axes.set_title(title, fontsize=option['title_fontsize'], pad=30)

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
    hex_colors = ['#4C6EF5', '#F9C74F', '#90BE6D', '#5BC0EB', '#43AA8B', '#F3722C', '#855456', '#F9AFAF', '#F8961E'
        , '#277DA1', '#5A189A']
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
