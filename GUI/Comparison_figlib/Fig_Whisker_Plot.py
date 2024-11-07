import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from io import BytesIO
import streamlit as st


def make_scenarios_comparison_Whisker_Plot(option, selected_item, ref_source, sim_sources, datasets_filtered, varname):
    font = {'family': option['font']}
    matplotlib.rc('font', **font)

    params = {'backend': 'ps',
              'axes.linewidth': option['axes_linewidth'],
              'font.size': option["fontsize"],
              'xtick.labelsize': option['xticksize'],
              'xtick.direction': 'out',
              'ytick.labelsize': option['yticksize'],
              'ytick.direction': 'out',
              'savefig.bbox': 'tight',
              'axes.unicode_minus': False,
              'text.usetex': False}
    rcParams.update(params)
    # Create the heatmap using Matplotlib
    fig, ax = plt.subplots(1, figsize=(option['x_wise'], option['y_wise']))


    bplot = plt.boxplot(datasets_filtered, labels=sim_sources,
                vert=option['vert'],
                showfliers=option['showfliers'],
                widths=option["box_widths"],
                showmeans=option["box_showmeans"],
                meanline=option["meanline"],
                meanprops=option["meanprops"],
                medianprops=option["mediaprops"],
                flierprops=option["flierprops"],
                patch_artist=option["patch_artist"],
                boxprops=option["boxprops"],
                whiskerprops=option["whiskerprops"],
                capprops=option["capprops"],
                )
    if option["patch_artist"]:
        for patch, color in zip(bplot['boxes'], option["colors"]):
            patch.set_facecolor(color)


    if option['vert']:
        plt.xticks(rotation=option['x_rotation'], ha=option['x_ha'])

        # Add labels and title
        plt.xlabel(option['xticklabel'], fontsize=option['xticksize']+1)
        plt.ylabel(option['yticklabel'], fontsize=option['yticksize']+1)
        plt.title(option['title'], fontsize=option['title_fontsize'])
        # ax.yaxis.grid(True)
        if option['grid']:
            ax.yaxis.grid(True, linestyle=option['grid_style'], alpha=0.7, linewidth=option['grid_linewidth'])

        if option['ylimit_on']:
            if option['y_min'] > option['y_max']:
                st.error('make sure your max min value was setting right!')
                exit()
            else:
                ax.set(ylim=(option['y_min'], option['y_max']))
    else:
        plt.yticks(rotation=option['y_rotation'], ha=option['y_ha'])
        plt.xlabel(option['xticklabel'], fontsize=option['xticksize']+1)
        plt.ylabel(option['yticklabel'], fontsize=option['yticksize']+1)
        plt.title(option['title'], fontsize=option['title_fontsize'])
        if option['grid']:
            ax.xaxis.grid(True, linestyle=option['grid_style'], alpha=0.7, linewidth=option['grid_linewidth'])

        if option['xlimit_on']:
            if option['x_min'] > option['x_max']:
                st.error('make sure your max min value was setting right!')
                exit()
            else:
                ax.set(xlim=(option['x_min'], option['x_max']))

    if varname == 'percent_bias':
        legend_title = 'Percent Bias showing value between [-100,100]'
        ax.legend(title=legend_title,shadow=False, frameon=False, fontsize=15, loc='best')
    elif varname in ['bias', 'rSD', 'PBIAS_HF', 'PBIAS_LF']:
        legend_title = f'{varname.replace("_"," ")} showing value under 5% and 95% quantile'
        ax.legend(title=legend_title,shadow=False, frameon=False, fontsize=15, loc='best')
    elif varname in ['KGE', 'KGESS', 'NSE', 'LNSE', 'ubNSE', 'rNSE', 'wNSE', 'wsNSE']:
        legend_title = f'{varname.replace("_"," ")} showing value above 5% quantile'
        ax.legend(title=legend_title,shadow=False, frameon=False, fontsize=15, loc='best')
    elif varname in ['RMSE', 'CRMSD', 'MSE', 'ubRMSE', 'nRMSE', 'mean_absolute_error', 'ssq', 've',
                   'absolute_percent_bias']:
        legend_title = f'{varname.replace("_"," ")} showing value under 95% quantile'
        ax.legend(title=legend_title,shadow=False, frameon=False, fontsize=15, loc='best')


    st.pyplot(fig)

    file2 = f"Whisker_Plot_{selected_item}_{ref_source}_{varname}"
    # 将图像保存到 BytesIO 对象
    buffer = BytesIO()
    fig.savefig(buffer, format=option['saving_format'], dpi=option['dpi'])
    buffer.seek(0)
    buffer.seek(0)
    st.download_button('Download image', buffer, file_name=f'{file2}.{option["saving_format"]}',
                       mime=f"image/{option['saving_format']}",
                       type="secondary", disabled=False, use_container_width=False)
