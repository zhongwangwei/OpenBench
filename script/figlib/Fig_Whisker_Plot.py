import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


def make_scenarios_comparison_Whisker_Plot(basedir, evaluation_item, ref_source, sim_sources, varname, datasets_filtered, option):
    font = {'family': option['font']}
    matplotlib.rc('font', **font)

    params = {'backend': 'ps',
              'axes.linewidth': option['axes_linewidth'],
              'xtick.labelsize': option['xtick'],
              'xtick.direction': 'out',
              'ytick.labelsize': option['ytick'],
              'ytick.direction': 'out',
              'savefig.bbox': 'tight',
              'axes.unicode_minus': False,
              'text.usetex': False}
    rcParams.update(params)
    # Create the heatmap using Matplotlib
    fig, ax = plt.subplots(1, figsize=(option['x_wise'], option['y_wise']))

    def colors(color):
        hex_pattern = r'^#([0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})$'
        import re
        if bool(re.match(hex_pattern, f"#{color}")):
            return f"#{color}"
        else:
            return color

    for key, value in option.items():
        if 'color' in key:
            option[key] = colors(value)

    boxprops = dict(linewidth=option["boxpropslinewidth"])
    if option["patch_artist"]:
        boxprops = dict(linewidth=option["boxpropslinewidth"], facecolor=option["boxpropsfacecolor"],
                        edgecolor=option["boxpropsedgecolor"])
    if varname in ['KGE', 'KGESS', 'NSE']:
        for i, data in enumerate(datasets_filtered):
            lower_bound = np.percentile(data, 5)
            if lower_bound < -1:
                datasets_filtered[i] = np.where(data < -1, -1, data).tolist()

    # Create the whisker plot
    plt.boxplot(datasets_filtered, labels=[f'{i}' for i in sim_sources],
                vert=option['vert'],
                showfliers=option['showfliers'],
                flierprops=dict(marker=option["flierpropsmarker"], markerfacecolor=option["flierpropsmarkerfacecolor"],
                                markersize=option["flierpropsmarkersize"],
                                markeredgecolor=option["flierpropsmarkeredgecolor"]),

                widths=option["box_widths"],

                showmeans=option["box_showmeans"],
                meanline=option["meanline"],
                meanprops=dict(linestyle=option["meanpropslinestyle"], linewidth=option["meanpropslinewidth"],
                               color=option["meanpropscolor"]),

                medianprops=dict(linestyle=option["medianpropslinestyle"], linewidth=option["medianpropslinewidth"],
                                 color=option["medianpropscolor"]),

                patch_artist=option["patch_artist"],
                boxprops=boxprops,

                whiskerprops=dict(linestyle=option["whiskerpropslinestyle"], linewidth=option["whiskerpropslinewidth"],
                                  color=option["whiskerpropscolor"]),
                capprops=dict(linestyle=option["cappropslinestyle"], linewidth=option["cappropslinewidth"],
                              color=option["cappropscolor"]),
                )

    # Create the whisker plot
    def remove_outliers(data_list):
        # 计算 IQR 并去除异常值
        q1, q3 = np.percentile(data_list, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr - iqr
        upper_bound = q3 + 1.5 * iqr + iqr
        return [lower_bound, upper_bound]

    try:
        bound = [remove_outliers(d) for d in datasets_filtered]
        max_value = max([d[1] for d in bound])
        min_value = min([d[0] for d in bound])

        if varname in ['RMSE', 'CRMSD', 'MSE', 'ubRMSE', 'nRMSE', 'mean_absolute_error', 'ssq', 've',
                       'absolute_percent_bias']:
            min_value = min_value * 0 - 0.2
        elif varname in ['NSE', 'LNSE', 'ubNSE', 'rNSE', 'wNSE', 'wsNSE']:
            max_value = max_value * 0 + 0.2

    except:
        min_value, max_value = None, None
        print(
            f"Error: {evaluation_item} {ref_source} {varname} Kernel Density Estimate failed!")  # ValueError: array must not contain infs or NaNs

    if option['vert']:
        plt.xticks(rotation=option['x_rotation'], ha=option['ha'])

        # Add labels and title
        plt.xlabel(option['xticklabel'], fontsize=option['xtick'] + 1)
        ylabel = option['yticklabel']
        if not option['yticklabel'] or len(option['yticklabel']) == 0:
            ylabel = f'{varname}'
        plt.ylabel(ylabel, fontsize=option['ytick'] + 1)

        if option['grid']:
            ax.yaxis.grid(True, linestyle=option['grid_style'], alpha=0.7, linewidth=option['grid_linewidth'])

        if option['limit_on']:
            if option['value_min'] > option['value_max']:
                print('Error: make sure your max min value was setting right!')
                exit()
            else:
                ax.set(ylim=(option['value_min'], option['value_max']))
                # ax.set(ylim=(dmin, dmax))
        else:
            ax.set(ylim=(min_value, max_value))
    else:
        plt.yticks(rotation=option['y_rotation'], ha=option['ha'])

        xlabel = option['xticklabel']
        if not option['xticklabel'] or len(option['xticklabel']) == 0:
            xlabel = f'{varname}'
        plt.xlabel(xlabel, fontsize=option['xtick'] + 1)
        plt.ylabel(option['yticklabel'], fontsize=option['ytick'] + 1)

        if option['grid']:
            ax.xaxis.grid(True, linestyle=option['grid_style'], alpha=0.7, linewidth=option['grid_linewidth'])

        if option['limit_on']:
            if option['value_min'] > option['value_max']:
                print('Error: make sure your max min value was setting right!')
                exit()
            else:
                ax.set(xlim=(option['value_min'], option['value_max']))
                # ax.set(xlim=(dmin, dmax))
        else:
            ax.set(xlim=(min_value, max_value))

    title = option['title']
    if not option['title'] or len(option['title']) == 0:
        title = f'Whisker Plot of {evaluation_item.replace("_", " ")}'
    plt.title(title, fontsize=option['title_fontsize'])

    output_file_path = f"{basedir}/Whisker_Plot_{evaluation_item}_{ref_source}_{varname}.{option['saving_format']}"
    plt.savefig(output_file_path, format=f'{option["saving_format"]}', dpi=option['dpi'], bbox_inches='tight')
    plt.close()  # Close the figure to free up memory

    return
