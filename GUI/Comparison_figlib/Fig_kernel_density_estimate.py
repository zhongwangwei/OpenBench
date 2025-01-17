import xarray as xr
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import rcParams
from matplotlib import ticker
import math
import matplotlib.colors as clr
import itertools
from matplotlib import cm
from scipy.stats import gaussian_kde

from io import BytesIO
import streamlit as st


def draw_scenarios_comparison_Kernel_Density_Estimate(option, selected_item, ref_source, sim_sources, datasets_filtered, varname):
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
    # fig, ax = plt.subplots(figsize=(option['x_wise'], option['y_wise']))

    fig = plt.figure(figsize=(option['x_wise'], option['y_wise']))
    ax = fig.add_subplot(111)  # 添加子图

    lines = []

    for i, sim_source in enumerate(sim_sources):
        data = datasets_filtered[i]
        try:
            lower_bound, upper_bound = np.percentile(data, 5), np.percentile(data, 95)
            if varname in ['KGE', 'KGESS', 'NSE', 'LNSE', 'ubNSE', 'rNSE', 'wNSE', 'wsNSE']:
                if lower_bound < -1:
                    filtered_data = np.where(data < -1, -1, data)
            else:
                filtered_data = data
            kde = gaussian_kde(filtered_data)
            covariance_matrix = kde.covariance
            covariance_matrix += np.eye(covariance_matrix.shape[0]) * 1e-6  # Regularization
            kde.covariance = covariance_matrix


            x_values = np.linspace(filtered_data.min(), filtered_data.max(), 100)
            density = kde(x_values)

            # Store the line object
            line, = ax.plot(x_values, density, color=option['MARKERS'][sim_source]['lineColor'],
                            linestyle=option['MARKERS'][sim_source]['linestyle'],
                            linewidth=option['MARKERS'][sim_source]['linewidth'],
                            label=sim_source)
            lines.append(line)  # Add the line object to the list
            ax.fill_between(x_values, density, color=option['MARKERS'][sim_source]['lineColor'],
                            alpha=option['MARKERS'][sim_source]['alpha'])
        except Exception as e:
            st.error(f"{selected_item} {ref_source} {sim_source} {varname} Kernel Density Estimate failed!")

    if not option["legend_on"]:
        ax.legend(shadow=False, frameon=False, fontsize=option['fontsize'],
                  loc=option["loc"], ncol=option["ncol"])

    else:
        ax.legend(shadow=False, frameon=False, fontsize=option['fontsize'],
                  bbox_to_anchor=(option["bbox_to_anchor_x"], option["bbox_to_anchor_y"]), ncol=option["ncol"])

    if option['grid']:
        ax.grid(linestyle=option['grid_style'], alpha=0.7, linewidth=option['grid_linewidth'])  # 绘制图中虚线 透明度0.3
    if option['minmax']:
        ax.set_xlim(option['xmin'], option['xmax'])

    plt.xlabel(option['xticklabel'], fontsize=option['xticksize'] + 1)
    plt.ylabel(option['yticklabel'], fontsize=option['yticksize'] + 1)
    plt.title(option['title'], fontsize=option['title_fontsize'])

    try:
        del datasets_filtered, lines, kde, covariance_matrix, x_values, density, line
    except:
        del datasets_filtered, data, lines

    st.pyplot(fig)

    file2 = f"Kernel_Density_Estimate_{selected_item}_{ref_source}_{varname}"
    # 将图像保存到 BytesIO 对象
    buffer = BytesIO()
    fig.savefig(buffer, format=option['saving_format'], dpi=option['dpi'])
    buffer.seek(0)
    buffer.seek(0)
    st.download_button('Download image', buffer, file_name=f'{file2}.{option["saving_format"]}',
                       mime=f"image/{option['saving_format']}",
                       type="secondary", disabled=False, use_container_width=False)


def make_scenarios_comparison_Kernel_Density_Estimate(dir_path, selected_item, score, ref_source, ref, sim, scores):
    item = 'Kernel_Density_Estimate'
    option = {}

    with st.container(height=None, border=True):
        col1, col2, col3, col4 = st.columns((3.5, 3, 3, 3))
        with col1:
            option['title'] = st.text_input('Title',
                                            value=f'Kernel Density Estimate of {selected_item.replace("_", " ")}',
                                            label_visibility="visible", key=f'kde_title')
            option['title_fontsize'] = st.number_input("Title label size", min_value=0, value=20,
                                                       key=f'kde_title_fontsize')

        with col2:
            option['xticklabel'] = st.text_input('X tick labels', value=score.replace("_", " "),
                                                 label_visibility="visible",
                                                 key=f'kde_xticklabel')
            if score == 'percent_bias':
                option['xticklabel'] = option['xticklabel'] + ' (showing value between [-100, 100])'
            option['xticksize'] = st.number_input("xtick label size", min_value=0, value=17, key=f'kde_xticksize')

        with col3:
            option['yticklabel'] = st.text_input('Y tick labels', value='KDE Density', label_visibility="visible",
                                                 key=f'kde_yticklabel')
            option['yticksize'] = st.number_input("ytick label size", min_value=0, value=17, key=f'kde_yticksize')

        with col4:
            option['fontsize'] = st.number_input("Font size", min_value=0, value=17, key=f'kde_labelsize')
            option['axes_linewidth'] = st.number_input("axes linewidth", min_value=0, value=1, key=f'kde_axes_linewidth')

        st.divider()
        ref_data_type = ref[ref_source]['general'][f'data_type']

        from matplotlib import cm
        import matplotlib.colors as mcolors
        import itertools
        colors = cm.Set3(np.linspace(0, 1, len(sim['general'][f'{selected_item}_sim_source']) + 1))
        hex_colors = itertools.cycle([mcolors.rgb2hex(color) for color in colors])

        def get_cases(items, title):
            case_item = {}
            for item in items:
                case_item[item] = True
            import itertools
            with st.popover(f"Selecting {title}", use_container_width=True):
                st.subheader(f"Showing {title}", divider=True)
                cols = itertools.cycle(st.columns(2))
                for item in case_item:
                    col = next(cols)
                    case_item[item] = col.checkbox(item, key=f'{item}__Kernel_Density_Estimate',
                                                   value=case_item[item])
                if len([item for item, value in case_item.items() if value]) > 0:
                    return [item for item, value in case_item.items() if value]
                else:
                    st.error('You must choose one item!')

        sim_sources = sim['general'][f'{selected_item}_sim_source']
        sim_sources = get_cases(sim_sources, 'cases')

        with st.expander("Other information setting", expanded=False):
            markers = {}
            datasets_filtered = []
            col1, col2, col3, col4 = st.columns(4)
            col1.write('##### :blue[Line colors]')
            col2.write('##### :blue[Lines Style]')
            col3.write('##### :blue[Line width]')
            col4.write('##### :blue[Line alpha]')
            for sim_source in sim_sources:
                st.write(f"Case: {sim_source}")
                col1, col2, col3, col4 = st.columns((1.1, 2, 2, 2))
                markers[sim_source] = {}
                markers[sim_source]['lineColor'] = col1.color_picker(f'{sim_source} Line colors', value=next(hex_colors),
                                                                     key=f'kde {sim_source} colors',
                                                                     disabled=False,
                                                                     label_visibility="collapsed")
                markers[sim_source]['linestyle'] = col2.selectbox(f'{sim_source} Lines Style',
                                                                  ['solid', 'dotted', 'dashed', 'dashdot'],
                                                                  key=f'{sim_source} Line Style',
                                                                  index=None, placeholder="Choose an option",
                                                                  label_visibility="collapsed")
                markers[sim_source]['linewidth'] = col3.number_input(f"{sim_source} Line width", min_value=0., value=1.5,
                                                                     step=0.1, label_visibility="collapsed")
                markers[sim_source]['alpha'] = col4.number_input(f"{sim_source} fill line alpha",
                                                                 key=f'{sim_source} alpha',
                                                                 min_value=0., value=0.3, step=0.1,
                                                                 max_value=1., label_visibility="collapsed")

                sim_data_type = sim[sim_source]['general'][f'data_type']
                if ref_data_type == 'stn' or sim_data_type == 'stn':
                    ref_varname = ref[f'{selected_item}'][f'{ref_source}_varname']
                    sim_varname = sim[f'{selected_item}'][f'{sim_source}_varname']
                    file_path = f"{dir_path}/output/scores/{selected_item}_stn_{ref_source}_{sim_source}_evaluations.csv"
                    df = pd.read_csv(file_path, sep=',', header=0)
                    data = df[score].values
                else:
                    if score in scores:
                        file_path = f"{dir_path}/output/scores/{selected_item}_ref_{ref_source}_sim_{sim_source}_{score}.nc"
                    else:
                        file_path = f"{dir_path}/output/metrics/{selected_item}_ref_{ref_source}_sim_{sim_source}_{score}.nc"
                    try:
                        ds = xr.open_dataset(file_path)
                        data = ds[score].values
                    except FileNotFoundError:
                        st.error(f"File {file_path} not found. Please check the file path.")
                    except KeyError as e:
                        st.error(f"Key error: {e}. Please check the keys in the option dictionary.")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                if score == 'percent_bias':
                    data = data[(data <= 100) & (data >= -100)]
                data = data[~np.isinf(data)]
                datasets_filtered.append(data[~np.isnan(data)])  # Filter out NaNs and append

            option['MARKERS'] = markers
            option["legend_on"] = st.toggle('Turn on to set the location of the legend manually', value=False,
                                            key=f'kde_legend_on')
            col1, col2, col3, col4 = st.columns(4)
            option["ncol"] = col1.number_input("N cols", value=1, min_value=1, format='%d', key=f'kde_ncol')
            if not option["legend_on"]:
                option["loc"] = col2.selectbox("Legend location",
                                               ['best', 'right', 'left', 'upper left', 'upper right', 'lower left',
                                                'lower right',
                                                'upper center',
                                                'lower center', 'center left', 'center right'], index=0,
                                               placeholder="Choose an option",
                                               label_visibility="visible", key=f'kde_loc')
            else:
                option["bbox_to_anchor_x"] = col3.number_input("X position of legend", value=1.5, key=f'kde_bbox_to_anchor_x')
                option["bbox_to_anchor_y"] = col4.number_input("Y position of legend", value=1., key=f'kde_bbox_to_anchor_y')
            col1, col2, col3 = st.columns(3)
            with col1:
                option['grid'] = st.toggle("Showing grid", value=False, label_visibility="visible", key=f'kde_grid')
            if option['grid']:
                option['grid_style'] = col2.selectbox('Grid Line Style', ['solid', 'dotted', 'dashed', 'dashdot'],
                                                      index=2, placeholder="Choose an option", label_visibility="visible",
                                                      key=f'kde_grid_style')
                option['grid_linewidth'] = col3.number_input("grid linewidth", min_value=0., value=1.,
                                                             key=f'kde_grid_linewidth')
            col1, col2, col3 = st.columns(3)
            with col1:
                option['minmax'] = st.toggle("Turn on to set X axis manually", value=False, label_visibility="visible",
                                             key=f'kde_minmax')
            if option['minmax']:
                def remove_outliers(data_list):
                    q1, q3 = np.percentile(data_list, [5, 95])
                    return [q1, q3]

                bound = [remove_outliers(d) for d in datasets_filtered]
                global_max = max([d[1] for d in bound])
                global_min = min([d[0] for d in bound])
                option['xmin'] = col2.number_input("X minimum value", value=float(math.floor(global_min)), step=0.1,
                                                   key=f'kde_xmin')
                option['xmax'] = col3.number_input("X maximum value", value=float(math.ceil(global_max)), step=0.1,
                                                   key=f'kde_xmax')

        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            option["x_wise"] = st.number_input(f"X Length", min_value=0, value=10, key=f'kde_x_wise')
            option['saving_format'] = st.selectbox('Image saving format', ['png', 'jpg', 'eps'],
                                                   index=1, placeholder="Choose an option", label_visibility="visible",
                                                   key=f'kde_saving_format')
        with col2:
            option["y_wise"] = st.number_input(f"y Length", min_value=0, value=6, key=f'kde_y_wise')
            option['font'] = st.selectbox('Image saving format',
                                          ['Times new roman', 'Arial', 'Courier New', 'Comic Sans MS', 'Verdana',
                                           'Helvetica',
                                           'Georgia', 'Tahoma', 'Trebuchet MS', 'Lucida Grande'],
                                          index=0, placeholder="Choose an option", label_visibility="visible",
                                          key=f'kde_font')
        with col3:
            option['dpi'] = st.number_input(f"Figure dpi", min_value=0, value=300, key=f'kde_dpi')

    if sim_sources:
        if score == 'nSpatialScore':
            st.info(f'{score} is not supported for {item.replace("_", " ")}!', icon="ℹ️")
        else:
            draw_scenarios_comparison_Kernel_Density_Estimate(option, selected_item, ref_source,
                                                              sim_sources, datasets_filtered,
                                                              score)
