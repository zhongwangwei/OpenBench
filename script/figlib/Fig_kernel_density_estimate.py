import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from scipy.stats import gaussian_kde


def make_scenarios_comparison_Kernel_Density_Estimate(basedir, evaluation_item, ref_source, sim_sources, varname,
                                                      datasets_filtered, option):
    font = {'family': option['font']}
    matplotlib.rc('font', **font)

    params = {'backend': 'ps',
              'axes.linewidth': option['axes_linewidth'],
              'font.size': option["fontsize"],
              'xtick.labelsize': option['xtick'],
              'xtick.direction': 'out',
              'ytick.labelsize': option['ytick'],
              'ytick.direction': 'out',
              'savefig.bbox': 'tight',
              'axes.unicode_minus': False,
              'text.usetex': False}
    rcParams.update(params)

    # create a figure and axis
    fig = plt.figure(figsize=(option['x_wise'], option['y_wise']))
    ax = fig.add_subplot(111)  # 添加子图



    # Generate colors using a colormap
    MLINES = generate_lines(sim_sources, option)
    # colors = cm.Set3(np.linspace(0, 1, len(datasets_filtered)))
    # Create a list to store line objects for the legend
    lines = []
    # Plot each KDE
    for i, data in enumerate(datasets_filtered):
        sim_source = sim_sources[i]

        try:
            lower_bound, upper_bound = np.percentile(data, 1.5), np.percentile(data, 98.5)
            if varname in ['bias', 'percent_bias', 'rSD', 'PBIAS_HF', 'PBIAS_LF']:
                filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
            elif varname in ['KGE', 'KGESS', 'NSE', 'LNSE', 'ubNSE', 'rNSE', 'wNSE', 'wsNSE']:
                filtered_data = data[(data >= lower_bound)]
            elif varname in ['RMSE', 'CRMSD', 'MSE', 'ubRMSE', 'nRMSE', 'mean_absolute_error', 'ssq', 've',
                             'absolute_percent_bias']:
                filtered_data = data[(data <= upper_bound)]
            else:
                filtered_data = data

            kde = gaussian_kde(filtered_data)
            covariance_matrix = kde.covariance
            covariance_matrix += np.eye(covariance_matrix.shape[0]) * 1e-6  # Regularization
            kde.covariance = covariance_matrix
            x_values = np.linspace(filtered_data.min(), filtered_data.max(), 100)
            density = kde(x_values)

            # Store the line object
            line, = plt.plot(x_values, density, color=MLINES[sim_source]['lineColor'],
                             linestyle=MLINES[sim_source]['linestyle'],
                             linewidth=MLINES[sim_source]['linewidth'],
                             label=sim_source
                             )
            lines.append(line)  # Add the line object to the list
            plt.fill_between(x_values, density, color=MLINES[sim_source]['lineColor'],
                             alpha=MLINES[sim_source]['alpha'])

        except:
            print(
                f"Error: {evaluation_item} {ref_source} {sim_source} {varname} Kernel Density Estimate failed!")  # ValueError: array must not contain infs or NaNs

    # Add labels and legend
    if not option["set_legend"]:
        ax.legend(shadow=False, frameon=False, fontsize=option['fontsize'],
                  loc=option["loc"], ncol=option["ncol"])
    else:
        ax.legend(shadow=False, frameon=False, fontsize=option['fontsize'],
                  bbox_to_anchor=(option["bbox_to_anchor_x"], option["bbox_to_anchor_y"]), ncol=option["ncol"])

    if option['grid']:
        ax.grid(linestyle=option['grid_style'], alpha=0.7, linewidth=option['grid_linewidth'])  # 绘制图中虚线 透明度0.3

    if not option['title']:
        option['title'] = f"Kernel Density Estimate of {evaluation_item.replace('_', ' ')}"

    if not option['xticklabel']:
        option['xticklabel'] = f'{varname.replace('_', '')}'
    if not option['yticklabel']:
        option['yticklabel'] = 'KDE Density'
    if not option['title']:
        option['title'] = f"Kernel Density Estimate of {evaluation_item.replace("_", " ")}"

    plt.xlabel(option['xticklabel'], fontsize=option['xtick'] + 1)
    plt.ylabel(option['yticklabel'], fontsize=option['ytick'] + 1)
    plt.title(option['title'], fontsize=option['title_fontsize'])

    output_file_path = f"{basedir}/Kernel_Density_Estimate_{evaluation_item}_{ref_source}_{varname}.{option["saving_format"]}"
    plt.savefig(output_file_path, format=f'{option["saving_format"]}', dpi=option['dpi'], bbox_inches='tight')

    if len(lines) > 0:
        del data, datasets_filtered, MLINES, lines, kde, covariance_matrix, x_values, density, line
    else:
        del data, datasets_filtered, MLINES, lines
    return


def generate_lines(data_names, option):
    import itertools
    import matplotlib.colors as mcolors
    lines = {}
    # add colors and symbols
    hex_colors = ['#4C6EF5', '#F9C74F', '#90BE6D', '#5BC0EB', '#43AA8B', '#F3722C', '#855456', '#F9AFAF', '#F8961E'
        , '#277DA1', '#5A189A']
    # hex_colors = cm.Set3(np.linspace(0, 1, len(data_names) + 1))
    colors = itertools.cycle([mcolors.rgb2hex(color) for color in hex_colors])

    if not option['linestyle']:
        linestyle = 'solid'
    else:
        linestyle = option['linestyle']

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
            "linestyle": linestyle,
            "linewidth": linewidth,
            "alpha": alpha
        }
    return lines
