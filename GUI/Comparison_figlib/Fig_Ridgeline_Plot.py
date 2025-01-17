import math
from scipy.stats import gaussian_kde
import matplotlib.cm as cm
import matplotlib
import matplotlib as mpl
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import itertools
from io import BytesIO
import streamlit as st
import xarray as xr
import numpy as np


def draw_scenarios_comparison_Ridgeline_Plot(option, evaluation_item, ref_source, sim_sources, datasets_filtered, varname):
    # st.write(option)
    font = {'family': option['font']}
    matplotlib.rc('font', **font)

    params = {'backend': 'ps',
              'axes.linewidth': option['axes_linewidth'],
              'font.size': option["fontsize"],
              'xtick.labelsize': option['xticksize'],
              'xtick.direction': 'out',
              'ytick.direction': 'out',
              'savefig.bbox': 'tight',
              'axes.unicode_minus': False,
              'text.usetex': False}
    rcParams.update(params)

    fig, axes = plt.subplots(figsize=(option['x_wise'], option['y_wise']))

    n_plots = len(sim_sources)
    global_min = option['global_min']
    global_max = option['global_max']
    if varname in ['KGE', 'NSE', 'KGESS']:
        if global_min < -1:
            global_min = -1
    x_range = np.linspace(global_min, global_max, 200)
    dx = x_range[1] - x_range[0]
    # Adjust these parameters to control spacing and overlap
    y_shift_increment = 0.5
    scale_factor = 0.8

    if option['colormap']:
        cmap = mpl.colormaps[f'{option["cmap"]}'].resampled(256)(range(256))
        R = np.broadcast_to(cmap[:, 0], (200, 256))
        G = np.broadcast_to(cmap[:, 1], (200, 256))
        B = np.broadcast_to(cmap[:, 2], (200, 256))
        alpha = np.full_like(R, 1)
        rgba = np.stack((R, G, B, alpha), axis=0)

    for i, (data, sim_source) in enumerate(zip(datasets_filtered, sim_sources)):
        filtered_data= data
        if varname in ['KGE', 'NSE', 'KGESS']:
            filtered_data = np.where(data < -1, -1, data)
        kde = gaussian_kde(filtered_data)
        y_range = kde(x_range)

        # Scale and shift the densities
        y_range = y_range * scale_factor / y_range.max()
        y_shift = i * y_shift_increment

        # Plot the KDE

        if option['colormap']:
            path = axes.fill_between(x_range, y_shift, y_range + y_shift, facecolor='none', lw=0.35, edgecolor='k',
                                     zorder=n_plots - i)
            patch = PathPatch(path._paths[0], visible=False, transform=axes.transData)
            ai = axes.imshow(rgba.transpose((1, 2, 0)), extent=[0, 1, 0, 1], transform=axes.transAxes, alpha=option['alpha'])
            ai.set_clip_path(patch)
            axes.plot(x_range, y_range + y_shift, color='black', linewidth=option['linewidth'])
        else:
            axes.fill_between(x_range, y_shift, y_range + y_shift, alpha=option['MARKERS'][sim_source]['alpha'],
                              color=option['MARKERS'][sim_source]['lineColor'], zorder=n_plots - i)
            axes.plot(x_range, y_range + y_shift, color='black', linewidth=option['MARKERS'][sim_source]['linewidth'])

        # Add labels
        axes.text(global_min, y_shift + 0.2, sim_source, fontweight='bold', ha='left', va='center')

        # Calculate and plot median
        median = np.median(data)
        if varname in ['KGE', 'NSE', 'KGESS'] and median <= global_min:
            pass
        else:
            index_closest = (np.abs(x_range - median)).argmin()
            y_target = y_range[index_closest]
            axes.vlines(median, y_shift, y_shift + y_target, color='black', linestyle=option['vlinestyle'],
                        linewidth=option['vlinewidth'], zorder=n_plots + 1)

            # Add median value text
            axes.text(median, y_shift + y_target, f'{median:.2f}', ha='center', va='bottom', fontsize=option["fontsize"],
                      zorder=n_plots + 2)

    # Customize the plot
    axes.set_yticks([])
    axes.set_xlabel(option['xticklabel'], fontsize=option['xticksize'] + 1)
    axes.set_title(option['title'], fontsize=option['title_fontsize'], pad=30)

    # Remove top and right spines
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['left'].set_visible(False)

    # Extend the bottom spine to the left
    axes.spines['bottom'].set_position(('data', -0.2))

    # Set y-axis limits
    axes.set_ylim(-0.2, (n_plots - 1) * y_shift_increment + scale_factor)
    axes.set_xlim(global_min - dx, global_max + dx)

    st.pyplot(fig)

    file2 = f"Ridgeline_Plot_{evaluation_item}_{ref_source}_{varname}"
    # 将图像保存到 BytesIO 对象
    buffer = BytesIO()
    fig.savefig(buffer, format=option['saving_format'], dpi=option['dpi'])
    buffer.seek(0)
    buffer.seek(0)
    st.download_button('Download image', buffer, file_name=f'{file2}.{option["saving_format"]}',
                       mime=f"image/{option['saving_format']}",
                       type="secondary", disabled=False, use_container_width=False)


def make_scenarios_comparison_Ridgeline_Plot(dir_path, selected_item, score, ref_source, ref, sim, scores):
    item = 'Ridgeline_Plot'
    option = {}

    with st.container(height=None, border=True):
        col1, col2, col3 = st.columns((3.5, 3, 3))
        with col1:
            option['title'] = st.text_input('Title', value=f'Ridgeline Plot of {selected_item.replace("_", " ")}',
                                            label_visibility="visible", key=f'Ridgeline_Plot_title')
            option['title_fontsize'] = st.number_input("Title label size", min_value=0, value=20,
                                                       key=f'Ridgeline_Plot_title_fontsize')
        with col2:
            xlabel = score.replace("_", " ")
            if score == 'percent_bias':
                xlabel = score.replace('_', ' ') + f' (showing value between [-100,100])'
            option['xticklabel'] = st.text_input('X tick labels', value=xlabel,
                                                 label_visibility="visible",
                                                 key=f'Ridgeline_Plot_xticklabel')
            option['xticksize'] = st.number_input("xtick label size", min_value=0, value=17, key=f'Ridgeline_Plot_xtick')

        with col3:
            option['fontsize'] = st.number_input("labelsize", min_value=0, value=17, key=f'Ridgeline_Plot_labelsize')
            option['axes_linewidth'] = st.number_input("axes linewidth", min_value=0, value=1,
                                                       key=f'Ridgeline_Plot_axes_linewidth')

        st.divider()
        ref_data_type = ref[ref_source]['general'][f'data_type']

        from matplotlib import cm
        import matplotlib.colors as mcolors
        colors = ['#4C6EF5', '#F9C74F', '#90BE6D', '#5BC0EB', '#43AA8B', '#F3722C', '#855456', '#F9AFAF', '#F8961E'
            , '#277DA1', '#5A189A']
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
                    case_item[item] = col.checkbox(item, key=f'{item}__Ridgeline_Plot',
                                                   value=case_item[item])
                if len([item for item, value in case_item.items() if value]) > 0:
                    return [item for item, value in case_item.items() if value]
                else:
                    st.error('You must choose one item!')

        sim_sources = sim['general'][f'{selected_item}_sim_source']
        sim_sources = get_cases(sim_sources, 'cases')
        option['colormap'] = False

        with st.expander("Colors setting", expanded=False):
            markers = {}
            datasets_filtered = []

            option['colormap'] = st.toggle('Use colormap?', key=f'{item}_colormap', value=option['colormap'])

            col1, col2, col3 = st.columns(3)
            col1.write('##### :blue[Line colors]')
            col2.write('##### :blue[Line width]')
            col3.write('##### :blue[Line alpha]')
            if option['colormap']:
                option['cmap'] = col1.selectbox('Colorbar',
                                                ['coolwarm', 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r',
                                                 'BuGn',
                                                 'BuGn_r',
                                                 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r',
                                                 'Grays', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r',
                                                 'Oranges',
                                                 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r',
                                                 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r',
                                                 'PuBu_r',
                                                 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r',
                                                 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn',
                                                 'RdYlGn_r',
                                                 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r',
                                                 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r',
                                                 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r',
                                                 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r',
                                                 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r',
                                                 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag',
                                                 'flag_r',
                                                 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey',
                                                 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow',
                                                 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r',
                                                 'gist_yerg', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray',
                                                 'gray_r',
                                                 'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet',
                                                 'jet_r',
                                                 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r',
                                                 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow',
                                                 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer',
                                                 'summer_r',
                                                 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c',
                                                 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight',
                                                 'twilight_r',
                                                 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter',
                                                 'winter_r'], index=0, placeholder="Choose an option",
                                                label_visibility="collapsed")
                option['linewidth'] = col2.number_input(f"Line width", min_value=0., value=1.5,
                                                        key=f'{item} Line width',
                                                        label_visibility="collapsed", step=0.1)
                option['alpha'] = col3.number_input(f"fill line alpha",
                                                    label_visibility="collapsed",
                                                    key=f'{item} alpha',
                                                    min_value=0., value=0.3, max_value=1.)
            else:
                for sim_source in sim_sources:
                    st.write(f"Case: {sim_source}")
                    col1, col2, col3 = st.columns((1.1, 2, 2))
                    markers[sim_source] = {}
                    markers[sim_source]['lineColor'] = col1.color_picker(f'{sim_source} Line colors', value=next(hex_colors),
                                                                         key=f'{item} {sim_source} colors', disabled=False,
                                                                         label_visibility="collapsed")
                    markers[sim_source]['linewidth'] = col2.number_input(f"{sim_source} Line width", min_value=0., value=1.5,
                                                                         key=f'{item} {sim_source} Line width',
                                                                         label_visibility="collapsed", step=0.1)
                    markers[sim_source]['alpha'] = col3.number_input(f"{sim_source} fill line alpha",
                                                                     label_visibility="collapsed",
                                                                     key=f'{item} {sim_source} alpha',
                                                                     min_value=0., value=0.3, max_value=1.)

            for sim_source in sim_sources:
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
                data = data[~np.isinf(data)]
                if score == 'percent_bias':
                    data = data[(data >= -100) & (data <= 100)]
                datasets_filtered.append(data[~np.isnan(data)])  # Filter out NaNs and append

            option['xlimit_on'] = True
            x_disable = False

            def remove_outliers(data_list):
                q1, q3 = np.percentile(data_list, [5, 95])
                return [q1, q3]

            try:
                bound = [remove_outliers(d) for d in datasets_filtered]
                max_value = max([d[1] for d in bound])
                min_value = min([d[0] for d in bound])
                # if score in ['RMSE', 'CRMSD', 'MSE', 'ubRMSE', 'nRMSE', 'mean_absolute_error', 'ssq', 've',
                #              'absolute_percent_bias']:
                #     min_value = min_value * 0 - 0.2
                # elif score in ['NSE', 'LNSE', 'ubNSE', 'rNSE', 'wNSE', 'wsNSE']:
                #     max_value = max_value * 0 + 0.2
                # elif score in ['KGE', 'NSE', 'KGESS']:
                #     if min_value < -1:
                #         min_value = min_value * 0 -1.0
                #     max_value =max_value* 0 +1.0
            except Exception as e:
                st.error(f"An error occurred: {e}, find minmax error!")
                option['xlimit_on'] = False
                x_disable = True
            st.divider()
            col1, col2, col3 = st.columns((3, 2, 2))
            option['xlimit_on'] = col1.toggle('Turn on to set the min-max value legend manually', value=option['xlimit_on'],
                                              disabled=x_disable, key=f'{item}_xlimit_on')
            if option['xlimit_on']:
                try:
                    option["global_max"] = col3.number_input(f"x ticks max", key=f'{item}_x_max',
                                                             value=max_value.astype(float))
                    option['global_min'] = col2.number_input(f"x ticks min", key=f'{item}_x_min',
                                                             value=min_value.astype(float))
                except Exception as e:
                    option['xlimit_on'] = False
            else:
                option["global_max"] = max_value.astype(float)  # max(data.max() for data in datasets_filtered)
                option['global_min'] = min_value.astype(float)  # min(data.min() for data in datasets_filtered)

            st.divider()
            col1, col2, col3 = st.columns((3, 3, 2))
            col1.write('##### :green[V Line Style]')
            col2.write('##### :green[V Line Width]')
            option['vlinestyle'] = col1.selectbox(f'V Line Style',
                                                  ['solid', 'dotted', 'dashed', 'dashdot'],
                                                  key=f'{item} {sim_source} Line Style',
                                                  index=1, placeholder="Choose an option",
                                                  label_visibility="collapsed")
            option['vlinewidth'] = col2.number_input(f"V Line width", min_value=0., value=1.5,
                                                     step=0.1, label_visibility="collapsed")

        option['MARKERS'] = markers
        st.divider()

        col1, col2, col3 = st.columns(3)
        with col1:
            option["x_wise"] = st.number_input(f"X Length", min_value=0, value=10, key=f'Ridgeline_Plot_x_wise')
            option['saving_format'] = st.selectbox('Image saving format', ['png', 'jpg', 'eps'],
                                                   index=1, placeholder="Choose an option", label_visibility="visible",
                                                   key=f'Ridgeline_Plot_saving_format')
        with col2:
            option["y_wise"] = st.number_input(f"y Length", min_value=0, value=len(sim_sources) * 2,
                                               key=f'Ridgeline_Plot_y_wise')
            option['font'] = st.selectbox('Image saving format',
                                          ['Times new roman', 'Arial', 'Courier New', 'Comic Sans MS', 'Verdana',
                                           'Helvetica',
                                           'Georgia', 'Tahoma', 'Trebuchet MS', 'Lucida Grande'],
                                          index=0, placeholder="Choose an option", label_visibility="visible",
                                          key=f'Ridgeline_Plot_font')
        with col3:
            option['dpi'] = st.number_input(f"Figure dpi", min_value=0, value=300, key=f'Ridgeline_Plot_dpi')
    draw_scenarios_comparison_Ridgeline_Plot(option, selected_item, ref_source,
                                             sim_sources, datasets_filtered, score)
