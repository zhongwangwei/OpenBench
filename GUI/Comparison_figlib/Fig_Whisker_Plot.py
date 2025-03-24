import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from io import BytesIO
import streamlit as st
import xarray as xr
import itertools


def draw_scenarios_comparison_Whisker_Plot(Figure_show, option, selected_item, ref_source, sim_sources, datasets_filtered, varname):
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
        plt.xlabel(option['xticklabel'], fontsize=option['xticksize'] + 1)
        plt.ylabel(option['yticklabel'], fontsize=option['yticksize'] + 1)
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
        plt.xlabel(option['xticklabel'], fontsize=option['xticksize'] + 1)
        plt.ylabel(option['yticklabel'], fontsize=option['yticksize'] + 1)
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
        ax.legend(title=legend_title, shadow=False, frameon=False, fontsize=15, loc='best')
    elif varname in ['bias', 'rSD', 'PBIAS_HF', 'PBIAS_LF']:
        legend_title = f'{varname.replace("_", " ")} showing value under 5% and 95% quantile'
        ax.legend(title=legend_title, shadow=False, frameon=False, fontsize=15, loc='best')
    elif varname in ['KGE', 'KGESS', 'NSE', 'LNSE', 'ubNSE', 'rNSE', 'wNSE', 'wsNSE']:
        legend_title = f'{varname.replace("_", " ")} showing value above 5% quantile'
        ax.legend(title=legend_title, shadow=False, frameon=False, fontsize=15, loc='best')
    elif varname in ['RMSE', 'CRMSD', 'MSE', 'ubRMSE', 'nRMSE', 'mean_absolute_error', 'ssq', 've',
                     'absolute_percent_bias']:
        legend_title = f'{varname.replace("_", " ")} showing value under 95% quantile'
        ax.legend(title=legend_title, shadow=False, frameon=False, fontsize=15, loc='best')

    Figure_show.pyplot(fig)

    file2 = f"Whisker_Plot_{selected_item}_{ref_source}_{varname}"
    buffer = BytesIO()
    fig.savefig(buffer, format=option['saving_format'], dpi=option['dpi'])
    buffer.seek(0)
    buffer.seek(0)
    st.download_button('Download image', buffer, file_name=f'{file2}.{option["saving_format"]}',
                       mime=f"image/{option['saving_format']}",
                       type="secondary", disabled=False, use_container_width=False)


def make_scenarios_comparison_Whisker_Plot(dir_path, selected_item, score, ref_source, self):
    Figure_show = st.container()
    Labels_tab,  Var_tab,Scale_tab, Line_tab, Save_tab = st.tabs(['Labels',  'Variables', 'Scale','Lines','Save'])

    option = {}
    item = 'Whisker_Plot'
    with Labels_tab:
        col1, col2, col3, col4 = st.columns((3.5, 3, 3, 3))
        with col1:
            option['title'] = st.text_input('Title', value=f'Whisker Plot of {selected_item.replace("_", " ")}',
                                            label_visibility="visible", key=f'Whisker_title')
            option['title_fontsize'] = st.number_input("Title label size", min_value=0, value=20,
                                                       key=f'Whisker_title_fontsize')
        with col2:
            option['xticklabel'] = st.text_input('X tick labels', label_visibility="visible",
                                                 key=f'Whisker_xticklabel')
            option['xticksize'] = st.number_input("xtick label size", min_value=0, value=17, key=f'Whisker_xtick')

        with col3:
            option['yticklabel'] = st.text_input('Y tick labels', value=score.replace("_", " "),
                                                 label_visibility="visible",
                                                 key=f'Whisker_yticklabel')
            option['yticksize'] = st.number_input("ytick label size", min_value=0, value=17, key=f'Whisker_ytick')

        with col4:
            option['fontsize'] = st.number_input("labelsize", min_value=0, value=17, key=f'Whisker_labelsize')
            option['axes_linewidth'] = st.number_input("axes linewidth", min_value=0., value=1.,
                                                       key=f'Whisker_axes_linewidth')

    with Var_tab:
        def get_cases(items, title):
            case_item = {}
            for item in items:
                case_item[item] = True
            import itertools
            color = '#9DA79A'
            st.markdown(f"""
            <div style="font-size:20px; font-weight:bold; color:{color}; border-bottom:3px solid {color}; padding: 5px;">
                 Showing {title}....
            </div>
            """, unsafe_allow_html=True)
            st.write('')
            cols = itertools.cycle(st.columns(2))
            for item in case_item:
                col = next(cols)
                case_item[item] = col.checkbox(item, key=f'{item}__Whisker_Plot',
                                               value=case_item[item])
            if len([item for item, value in case_item.items() if value]) > 0:
                return [item for item, value in case_item.items() if value]
            else:
                st.error('You must choose one item!')

        sim_sources = self.sim['general'][f'{selected_item}_sim_source']
        sim_sources = get_cases(sim_sources, 'cases')

    with Scale_tab:
        ref_data_type = self.ref[ref_source]['general'][f'data_type']
        datasets_filtered = []
        if sim_sources:
            for sim_source in sim_sources:
                sim_data_type = self.sim[sim_source]['general'][f'data_type']
                if ref_data_type == 'stn' or sim_data_type == 'stn':
                    file_path = f"{dir_path}/output/scores/{selected_item}_stn_{ref_source}_{sim_source}_evaluations.csv"
                    df = pd.read_csv(file_path, sep=',', header=0)
                    data = df[score].values
                else:
                    if score in self.scores:
                        file_path = f"{dir_path}/output/scores/{selected_item}_ref_{ref_source}_sim_{sim_source}_{score}.nc"
                    else:
                        file_path = f"{dir_path}/output/metrics/{selected_item}_ref_{ref_source}_sim_{sim_source}_{score}.nc"
                    try:
                        ds = xr.open_dataset(file_path)
                        data = ds[score].values
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                if score == 'percent_bias':
                    data = data[(data <= 100) & (data >= -100)]
                data = data[~np.isinf(data)]
                data = data[~np.isnan(data)]
                datasets_filtered.append(data)  # Filter out NaNs and append

            def remove_outliers(data_list):
                q1, q3 = np.percentile(data_list, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr - iqr
                upper_bound = q3 + 1.5 * iqr + iqr
                return [lower_bound, upper_bound]

            bound = [remove_outliers(d) for d in datasets_filtered]
            max_value = max([d[1] for d in bound])
            min_value = min([d[0] for d in bound])


        col1, col2, col3 = st.columns((3, 2, 2))
        option['vert'] = col1.toggle('Turn on to display Vertical Whisker Plot?', value=True, key=f'Whisker_Plot_vert')
        if option['vert']:
            option["x_rotation"] = col2.number_input(f"x rotation", min_value=-90, max_value=90, value=45)
            option['x_ha'] = col3.selectbox('x ha', ['right', 'left', 'center'],
                                            index=0, placeholder="Choose an option", label_visibility="visible")
            # ----------------------------------------------
            col1, col2, col3 = st.columns((3, 2, 2))
            option['ylimit_on'] = col1.toggle('Setting the max-min value manually', value=False,
                                              key=f'Whisker_Plot_limit_on')
            if option['ylimit_on']:
                try:
                    option["y_max"] = col3.number_input(f"y ticks max", key='Whisker_Plot_y_max',
                                                        value=max_value.astype(float))
                    option['y_min'] = col2.number_input(f"y ticks min", key='Whisker_Plot_y_min',
                                                        value=min_value.astype(float))
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    option['ylimit_on'] = False
        else:
            option["y_rotation"] = col2.number_input(f"y rotation", min_value=-90, max_value=90, value=0)
            option['y_ha'] = col3.selectbox('y ha', ['right', 'left', 'center'],
                                            index=0, placeholder="Choose an option", label_visibility="visible")
            col1, col2, col3 = st.columns((3, 2, 2))
            option['xlimit_on'] = col1.toggle('Setting the max-min value manually', value=False,
                                              key=f'Whisker_Plot_limit_on')
            if option['xlimit_on']:
                try:
                    option["x_max"] = col3.number_input(f"x ticks max", key='Whisker_Plot_x_max',
                                                        value=max_value.astype(float))
                    option['x_min'] = col2.number_input(f"x ticks min", key='Whisker_Plot_x_min',
                                                        value=min_value.astype(float))
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    option['xlimit_on'] = False
        col1, col2, col3 = st.columns((3, 2, 2))
        option['grid'] = col1.toggle("Turn on to showing grid", value=True, label_visibility="visible",
                                     key=f'Whisker_grid')
        if option['grid']:
            option['grid_style'] = col2.selectbox('Line Style', ['solid', 'dotted', 'dashed', 'dashdot'],
                                                  index=2, placeholder="Choose an option", label_visibility="visible",
                                                  key=f'Whisker_grid_style')
            option['grid_linewidth'] = col3.number_input("Linewidth", min_value=0., value=1.,
                                                         key=f'Whisker_grid_linewidth')

    with Line_tab:
        col1, col2, col3 = st.columns((3, 2, 2))
        option['showfliers'] = col1.toggle('Turn on to show fliers?', value=True, key=f'Whisker_Plot_showfliers')
        if option['showfliers']:
            Marker = ['.', 'x', 'o', ">", '<', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X', "+", "^", "v"]
            col1, col2, col3, col4 = st.columns((1.4, 1.4, 2, 2))
            col1.write('###### :orange[Marker color]')
            col2.write('###### :orange[Edge color]')
            col3.write('###### :orange[Marker style]')
            col4.write('###### :orange[Marker size]')
            col1, col2, col3, col4 = st.columns((1.2, 1.2, 2, 2))
            fliermarkerfacecolor = col1.color_picker(f'flier marker color', value='#FFFFFF',
                                                     key=f'fliermarkerfacecolor',
                                                     disabled=False,
                                                     label_visibility="collapsed")

            fliermarkeredgecolor = col2.color_picker(f'flier marker edge color', value='#000000',
                                                     key=f'fliermarkeredgecolor',
                                                     disabled=False,
                                                     label_visibility="collapsed")
            flier_marker = col3.selectbox(f'flier marker style', Marker,
                                          index=2, placeholder="Choose an option", label_visibility="collapsed")
            fliermarkersize = col4.number_input(f"flier marker size", key='Whisker_Plot_fliermarkersize', value=10.,
                                                label_visibility="collapsed")
            option["flierprops"] = dict(marker=flier_marker, markerfacecolor=fliermarkerfacecolor,
                                        markersize=fliermarkersize,
                                        markeredgecolor=fliermarkeredgecolor)
        else:
            option["flierprops"] = {}
        st.divider()

        # ----------------------------------------------
        col1, col2 = st.columns((3, 3))
        option["box_showmeans"] = col1.toggle('Draw box meanline? ', value=False, key="box_meanline")
        if option["box_showmeans"]:
            option["means_style"] = col2.selectbox('means style', ['marker', 'line'],
                                                   index=1, placeholder="Choose an option", label_visibility="visible")

        col1, col2, col3 = st.columns((2.5, 2.5, 1.5))
        col1.write('###### :green[Line style]')
        col2.write('###### :green[Line widths]')
        col3.write('###### :green[color]')

        st.write('###### Median')
        col1, col2, col3 = st.columns((2.5, 2.5, 1.5))
        medialinestyle = col1.selectbox(f'media line style', ['-', '--', '-.', ':'],
                                        index=0, placeholder="Choose an option", label_visibility="collapsed")
        medialinewidth = col2.number_input(f"media line widths", key='Whisker_Plot_media_line_widths', value=1.,
                                           label_visibility="collapsed")
        medialinecolor = col3.color_picker(f'media line color', value='#FD5900',
                                           key=f'medialinecolor',
                                           disabled=False,
                                           label_visibility="collapsed")
        option["mediaprops"] = dict(linestyle=medialinestyle, linewidth=medialinewidth, color=medialinecolor)

        if option["box_showmeans"]:
            mcol1, mcol2, mcol3, mcol4 = st.columns((2.5, 2.5, 0.75, 0.75))
            mcol1.write('###### Mean')
            col1, col2, col3 = st.columns((2.5, 2.5, 1.5))
            if option["means_style"] == 'line':
                option["meanline"] = True
                linestyle = col1.selectbox(f'mean line style', ['-', '--', '-.', ':'],
                                           index=0, placeholder="Choose an option", label_visibility="collapsed")
                linewidth = col2.number_input(f"mean line widths", key='Whisker_Plot_mean_line_widths', value=1.,
                                              label_visibility="collapsed")
                linecolor = col3.color_picker(f'mean line color', value='#0050FD',
                                              key=f'markerfacecolor',
                                              disabled=False, label_visibility="collapsed")
                option["meanprops"] = dict(linestyle=linestyle, linewidth=linewidth, color=linecolor)
            elif option["means_style"] == 'marker':
                col1, col2, col3 = st.columns((2.5, 2.5, 1.5))
                option["meanline"] = False
                Marker = ['.', 'x', 'o', ">", '<', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X', "+", "^", "v"]
                mean_marker = col1.selectbox(f'mean marker style', Marker,
                                             index=0, placeholder="Choose an option", label_visibility="collapsed")
                mcol2.write('Marker size')
                markersize = col2.number_input(f"mean marker size", key='Whisker_Plot_markersize', value=4.,
                                               label_visibility="collapsed")
                mcol3.write('Edge')
                mcol4.write('Marker')
                col31, col32 = col3.columns(2)
                markeredgecolor = col31.color_picker(f'mean marker edge color', value='#000000',
                                                     key=f'markeredgecolor',
                                                     disabled=False,
                                                     label_visibility="collapsed")
                markerfacecolor = col32.color_picker(f'mean marker color', value='#007CFD',
                                                     key=f'markerfacecolor',
                                                     disabled=False,
                                                     label_visibility="collapsed")
                option["meanprops"] = dict(marker=mean_marker, markeredgecolor=markeredgecolor,
                                           markerfacecolor=markerfacecolor, markersize=markersize)
        else:
            option["meanline"] = False
            option["meanprops"] = {}

        st.divider()

        col1, col2 = st.columns((3, 3))
        option["patch_artist"] = col1.toggle('patch artist? ', value=False, key="patch_artist",
                                             help='If False produces boxes with the Line2D artist. Otherwise, boxes are drawn with Patch artists.')
        col1, col2 = st.columns((2.5, 1))
        col1.write('###### :green[Widths]')
        col2.write('###### :green[Color]')
        mcol1, mcol2, mcol3 = st.columns((2, 2, 1.5))
        mcol1.write('###### Box')
        mcol2.write('###### Box Line')
        col1, col2, col3 = st.columns((2, 2, 1.5))
        option["box_widths"] = col1.number_input(f"box widths", key='Whisker_Plot_box_widths', value=0.5,
                                                 label_visibility="collapsed")
        boxlinewidth = col2.number_input(f"box line widths", key='Whisker_Plot_box_line_widths', value=1.,
                                         label_visibility="collapsed")
        if option["patch_artist"]:
            mcol3.write('###### Box Edge')
            boxedgecolor = col3.color_picker(f'box edge color', value='#000000',
                                             key=f'boxedgecolor',
                                             disabled=False, label_visibility="collapsed")
            option["boxprops"] = dict(linewidth=boxlinewidth, edgecolor=boxedgecolor)  # , facecolor=boxfacecolor
            if sim_sources:
                from matplotlib import cm
                import matplotlib.colors as mcolors
                hex_colors = cm.Set3(np.linspace(0, 1, len(sim_sources) + 1))
                colors = itertools.cycle([mcolors.rgb2hex(color) for color in hex_colors])
                st.write('##### :blue[box color]')
                cols = itertools.cycle(st.columns(3))
                option["colors"] = []
                for sim_source in sim_sources:
                    col = next(cols)
                    mcolor = next(colors)
                    color = col.color_picker(f'{sim_source}', value=mcolor,
                                             key=f'{sim_source}_boxedgecolor',
                                             disabled=False, label_visibility="visible")
                    option["colors"].append(color)
        else:
            option["boxprops"] = dict(linewidth=boxlinewidth)

        mcol1, mcol2, mcol3, mcol4 = st.columns((2.5, 2.5, 1.5, 1.5))
        mcol1.write('###### Cap line')
        mcol2.write('###### whisker line')
        col1, col2, col3, col4 = st.columns((2.5, 2.5, 1.5, 1.5))
        caplinewidth = col1.number_input(f"cap line widths", key='Whisker_Plot_cap_line_widths', value=1., step=0.1,
                                         label_visibility="collapsed")
        whiskerlinewidth = col2.number_input(f"whisker line widths", key='Whisker_Plot_whisker_line_widths', value=1.,
                                             step=0.1, label_visibility="collapsed")
        mcol3.write('###### Cap line')
        mcol4.write('###### whisker line')
        caplinecolor = col3.color_picker(f'cap line color', value='#000000',
                                         key=f'caplinecolor', disabled=False, label_visibility="collapsed")
        whiskerlinecolor = col4.color_picker(f'whisker line color', value='#000000', key=f'whiskerlinecolor',
                                             disabled=False, label_visibility="collapsed")

        option["whiskerprops"] = dict(linestyle='-', linewidth=whiskerlinewidth, color=whiskerlinecolor)
        option["capprops"] = dict(linestyle='-', linewidth=caplinewidth, color=caplinecolor)



    with Save_tab:
        col1, col2, col3 = st.columns(3)
        with col1:
            option["x_wise"] = st.number_input(f"X Length", min_value=0, value=10, key=f'Whisker_x_wise')
            option['saving_format'] = st.selectbox('Image saving format', ['png', 'jpg', 'eps'],
                                                   index=1, placeholder="Choose an option", label_visibility="visible",
                                                   key=f'Whisker_saving_format')
        with col2:
            option["y_wise"] = st.number_input(f"y Length", min_value=0, value=6, key=f'Whisker_y_wise')
            option['font'] = st.selectbox('Image saving format',
                                          ['Times new roman', 'Arial', 'Courier New', 'Comic Sans MS', 'Verdana',
                                           'Helvetica',
                                           'Georgia', 'Tahoma', 'Trebuchet MS', 'Lucida Grande'],
                                          index=0, placeholder="Choose an option", label_visibility="visible",
                                          key=f'Whisker_font')
        with col3:
            option['dpi'] = st.number_input(f"Figure dpi", min_value=0, value=300, key=f'Whisker_dpi')

    st.divider()
    if sim_sources:
        draw_scenarios_comparison_Whisker_Plot(Figure_show, option, selected_item, ref_source, sim_sources, datasets_filtered, score)
    else:
        st.error('You must choose at least one case!')
