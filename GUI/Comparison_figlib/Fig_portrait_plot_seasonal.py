import sys
import matplotlib
from matplotlib import rcParams
import matplotlib.collections as collections
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd

from io import BytesIO
import streamlit as st


def make_scenarios_comparison_Portrait_Plot_seasonal(self, file, selected_item, score, ref_source, item):
    st.cache_data.clear()
    option = {}
    Figure_show = st.container()
    Labels_tab, Scale_tab, Var_tab, Save_tab = st.tabs(['Labels', 'Scale', 'Variables', 'Save'])
    with Labels_tab:
        col1, col2, col3, col4 = st.columns((3.5, 3, 3, 3))
        with col1:
            option['title'] = st.text_input('Title', label_visibility="visible",
                                            key=f"{item} title")
            option['title_size'] = st.number_input("Title label size", min_value=0, value=20, key=f"{item}_title_size")

        with col2:
            option['xticklabel'] = st.text_input('X tick labels', value='Simulation', label_visibility="visible",
                                                 key=f"{item}_xticklabel")
            option['xticksize'] = st.number_input("xtick size", min_value=0, value=17, key=f"{item}_xtick")

        with col3:
            option['yticklabel'] = st.text_input('Y tick labels', value=f'{score.title()}', label_visibility="visible",
                                                 key=f"{item}_yticklabel")
            option['yticksize'] = st.number_input("ytick size", min_value=0, value=17, key=f"{item}_ytick")

        with col4:
            option['axes_linewidth'] = st.number_input("axes linewidth", min_value=0., value=1., step=0.1,
                                                       key=f"{item}_axes_linewidth")
            option['fontsize'] = st.number_input("Font size", min_value=0, value=15, step=1, key=f"{item}_fontsize")

    with Var_tab:
        def get_cases(items, title):
            case_item = {}
            for item in items:
                case_item[item] = True

            color = '#9DA79A'
            st.markdown(f"""
            <div style="font-size:20px; font-weight:bold; color:{color}; border-bottom:3px solid {color}; padding: 5px;">
                 Showing {title}....
            </div>
            """, unsafe_allow_html=True)
            st.write('')
            if title != 'cases':
                for item in case_item:
                    case_item[item] = st.checkbox(item.replace("_", " "), key=f"{item}__Portrait_Plot_seasonal_variable",
                                                  value=case_item[item])
                if len([item for item, value in case_item.items() if value]) > 0:
                    return [item for item, value in case_item.items() if value]
                else:
                    st.error("You must choose one items!")
            else:
                for item in case_item:
                    case_item[item] = st.checkbox(item, key=f"{item}__Portrait_Plot_seasonal_variable",
                                                  value=case_item[item])
                if len([item for item, value in case_item.items() if value]) > 0:
                    return [item for item, value in case_item.items() if value]
                else:
                    st.error("You must choose one items!")

        if score == 'metrics':
            items = [k for k, v in self.metrics.items() if v]
        else:
            items = [k for k, v in self.scores.items() if v]
        cases = [k for k in self.sim['general'][f"{selected_item}_sim_source"]]
        col1, col2 = st.columns(2)
        with col1:
            items = get_cases(items, f'Selected {score.title()}')
        with col2:
            cases = get_cases(cases, 'cases')

    with Scale_tab:

        col1, col2, col3, col4 = st.columns(4)

        option["x_rotation"] = col1.number_input(f"x rotation", min_value=-90, max_value=90, value=45,
                                                 key=f"{item}_x_rotation")
        option['x_ha'] = col2.selectbox('x ha', ['right', 'left', 'center'],
                                        index=1, placeholder="Choose an option", label_visibility="visible",
                                        key=f"{item}_x_ha")
        option["y_rotation"] = col3.number_input(f"y rotation", min_value=-90, max_value=90, value=0,
                                                 key=f"{item}_y_rotation")
        option['y_ha'] = col4.selectbox('y ha', ['right', 'left', 'center'],
                                        index=0, placeholder="Choose an option", label_visibility="visible",
                                        key=f"{item}_y_ha")

        col1, col2, col3, col4 = st.columns(4)
        option['colorbar_off'] = col1.toggle('Turn off colorbar?', value=False)
        option['cmap'] = col2.selectbox('Colorbar',
                                        ['coolwarm', 'coolwarm_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn',
                                         'BuGn_r',
                                         'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu',
                                         'GnBu_r',
                                         'Grays', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r',
                                         'Oranges',
                                         'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1',
                                         'Pastel1_r',
                                         'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r',
                                         'PuBu_r',
                                         'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu',
                                         'RdBu_r',
                                         'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn',
                                         'RdYlGn_r',
                                         'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r',
                                         'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu',
                                         'YlGnBu_r',
                                         'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot',
                                         'afmhot_r',
                                         'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg',
                                         'brg_r',
                                         'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'copper',
                                         'copper_r',
                                         'cubehelix', 'cubehelix_r', 'flag', 'flag_r',
                                         'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey',
                                         'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow',
                                         'gist_rainbow_r', 'gray', 'gray_r',
                                         'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet',
                                         'jet_r',
                                         'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean',
                                         'ocean_r',
                                         'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow',
                                         'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer',
                                         'summer_r',
                                         'terrain', 'terrain_r', 'viridis', 'viridis_r', 'winter',
                                         'winter_r'], index=0, placeholder="Choose an option", key=f'{item}_cmap',
                                        label_visibility="visible")
        if not option['colorbar_off']:
            with col3:
                option['colorbar_label'] = st.text_input('colorbar label', value=score.title(),
                                                         label_visibility="visible",
                                                         key=f"{item}_colorbar_label")
            with col4:
                option["colorbar_position"] = st.selectbox('colorbar position', ['horizontal', 'vertical'],  # 'Season',
                                                           index=0, placeholder="Choose an option",
                                                           label_visibility="visible",
                                                           key=f"{item}_colorbar_position")
                if option["colorbar_position"] == 'vertical':
                    pad_value = 0.05
                else:
                    pad_value = 0.15
        else:
            option['colorbar_label'] = ''
            option["colorbar_position"] = 'vertical'

        if score == 'scores':
            col1, col2, col3, col4 = st.columns(4)
            option["extend"] = 'neither'
            vmin, vmax, colorbar_ticks = 0., 1., 0.2
            option["vmin_max_on"] = col1.toggle('Fit to data', value=False, key=f"{item}_vmin_max_on")
            try:
                if option["vmin_max_on"]:
                    option["nstep"] = col2.number_input(f"Colorbar Ticks locater", value=colorbar_ticks, step=0.1,
                                                        key=f"{item}_colorbar_ticks")
                    try:
                        option["vmin"] = col3.number_input(f"colorbar min", value=vmin, key=f"{item}_vmin")
                        option["vmax"] = col4.number_input(f"colorbar max", value=vmax, key=f"{item}_vmax")
                    except ValueError:
                        st.error(f"Max value must larger than min value.")
                else:
                    option["nstep"] = colorbar_ticks
                    option["vmin"] = vmin
                    option["vmax"] = vmax
            except Exception as e:
                st.error(f"Error: {e}")
                error = True

        st.write("##### :blue[Legend]")
        col1, col2, col3, col4 = st.columns(4)
        option["legend_box_x"] = col1.number_input(f"x position", value=1.1, step=0.1, key=f"{item}_legend_box_x")
        option["legend_box_y"] = col2.number_input(f"y position", value=1.2, step=0.1, key=f"{item}_legend_box_y")
        option["legend_box_size"] = col3.number_input(f"legend box size", value=1.0, step=0.1,
                                                      key=f"{item}_legend_box_size")
        option["legend_lw"] = col4.number_input(f"Line width", value=1.0, step=0.1, key=f"{item}_legend_lw")
        option["legend_fontsize"] = col1.number_input(f"Box fontsize", value=12.5, step=0.2,
                                                      key=f"{item}_legend_fontsize")
    with Save_tab:
        col1, col2, col3 = st.columns(3)
        x_lenth = 10
        y_lenth = 5
        if score == 'metrics':
            x_lenth = len(cases)
            y_lenth = len(items)
        with col1:
            option["x_wise"] = st.number_input(f"X Length", min_value=0, value=x_lenth, key=f"{item}_x_wise")
            option['saving_format'] = st.selectbox('Image saving format', ['png', 'jpg', 'eps'],
                                                   index=1, placeholder="Choose an option", label_visibility="visible",
                                                   key=f"{item}_saving_format")
        with col2:
            option["y_wise"] = st.number_input(f"y Length", min_value=0, value=y_lenth, key=f"{item}_y_wise")
            option['font'] = st.selectbox('Image saving format',
                                          ['Times new roman', 'Arial', 'Courier New', 'Comic Sans MS', 'Verdana',
                                           'Helvetica',
                                           'Georgia', 'Tahoma', 'Trebuchet MS', 'Lucida Grande'],
                                          index=0, placeholder="Choose an option", label_visibility="visible",
                                          key=f"{item}_font")
        with col3:
            option['dpi'] = st.number_input(f"Figure dpi", min_value=0, value=300, key=f"{item}_dpi'")

    st.divider()
    if items is not None and cases is not None:
        if score == 'metrics':
            draw_scenarios_comparison_Portrait_Plot_seasonal_metrics(Figure_show, option, file, selected_item, ref_source, items, cases,
                                                                     score)
        else:
            draw_scenarios_comparison_Portrait_Plot_seasonal(Figure_show, option, file, selected_item, ref_source, items, cases,
                                                             score)
    elif not items:
        st.error('Metircs items is None!')
    elif not cases:
        st.error('Simulation cases is None!')


def make_scenarios_comparison_Portrait_Plot_seasonal_by_score(self, file, score, item):
    st.cache_data.clear()
    option = {}
    Figure_show = st.container()
    Labels_tab, Scale_tab, Var_tab, Save_tab = st.tabs(['Labels', 'Scale', 'Variables', 'Save'])
    with Labels_tab:
        col1, col2, col3, col4 = st.columns((3, 3, 3, 3))
        with col1:
            option['title'] = st.text_input('Title', label_visibility="visible",
                                            key=f"{item} title")
            option['title_size'] = st.number_input("Title label size", min_value=0, value=18, key=f"{item}_title_size")

        with col2:
            option['xticklabel'] = st.text_input('X tick labels', value='Simulation', label_visibility="visible",
                                                 key=f"{item}_xticklabel")
            option['xticksize'] = st.number_input("xtick size", min_value=0, value=15, key=f"{item}_xtick")

        with col3:
            option['yticklabel'] = st.text_input('Y tick labels', value=f'{score.replace("_", " ")}',
                                                 label_visibility="visible",
                                                 key=f"{item}_yticklabel")
            option['yticksize'] = st.number_input("ytick size", min_value=0, value=15, key=f"{item}_ytick")

        with col4:
            option['axes_linewidth'] = st.number_input("axes linewidth", min_value=0., value=1., step=0.1,
                                                       key=f"{item}_axes_linewidth")
            option['fontsize'] = st.number_input("Font size", min_value=0, value=15, step=1, key=f"{item}_fontsize")
    with Var_tab:

        def get_cases(items, title):
            case_item = {}
            for item in items:
                case_item[item] = True
            color = '#9DA79A'
            st.markdown(f"""
            <div style="font-size:20px; font-weight:bold; color:{color}; border-bottom:3px solid {color}; padding: 5px;">
                 Showing {title}....
            </div>
            """, unsafe_allow_html=True)
            st.write('')
            if title != 'cases':
                for item in case_item:
                    case_item[item] = st.checkbox(item.replace("_", " "), key=f"{item}__Portrait_Plot_seasonal_score",
                                                  value=case_item[item])
                if len([item for item, value in case_item.items() if value]) > 0:
                    return [item for item, value in case_item.items() if value]
                else:
                    st.error("You must choose one item!")
            else:
                for item in case_item:
                    case_item[item] = st.checkbox(item, key=f"{item}__Portrait_Plot_seasonal_score",
                                                  value=case_item[item])
                if len([item for item, value in case_item.items() if value]) > 0:
                    return [item for item, value in case_item.items() if value]
                else:
                    st.error("You must choose one item!")

        items = [k for k in self.selected_items]
        cases = list(
            set([value for key in self.selected_items for value in self.sim['general'][f"{key}_sim_source"] if value]))
        col1, col2 = st.columns(2)
        with col1:
            items = get_cases(items, 'Selected items')
        with col2:
            cases = get_cases(cases, 'cases')
    with Scale_tab:
        col1, col2, col3, col4 = st.columns(4)
        option["x_rotation"] = col1.number_input(f"x rotation", min_value=-90, max_value=90, value=45,
                                                 key=f"{item}_x_rotation")
        option['x_ha'] = col2.selectbox('x ha', ['right', 'left', 'center'],
                                        index=1, placeholder="Choose an option", label_visibility="visible",
                                        key=f"{item}_x_ha")
        option["y_rotation"] = col3.number_input(f"y rotation", min_value=-90, max_value=90, value=0,
                                                 key=f"{item}_y_rotation")
        option['y_ha'] = col4.selectbox('y ha', ['right', 'left', 'center'],
                                        index=0, placeholder="Choose an option", label_visibility="visible",
                                        key=f"{item}_y_ha")

        col1, col2, col3, col4 = st.columns(4)
        option['colorbar_off'] = col1.toggle('Turn off colorbar?', value=False)
        option['cmap'] = col2.selectbox('Colorbar',
                                        ['coolwarm', 'coolwarm_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn',
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
                                         'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'copper',
                                         'copper_r',
                                         'cubehelix', 'cubehelix_r', 'flag', 'flag_r',
                                         'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey',
                                         'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow',
                                         'gist_rainbow_r', 'gray', 'gray_r',
                                         'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet',
                                         'jet_r',
                                         'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r',
                                         'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow',
                                         'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer',
                                         'summer_r',
                                         'terrain', 'terrain_r', 'viridis', 'viridis_r', 'winter',
                                         'winter_r'], index=0, placeholder="Choose an option", key=f'{item}_cmap',
                                        label_visibility="visible")
        if not option['colorbar_off']:
            with col4:
                option["extend"] = st.selectbox(f"colorbar extend", ['neither', 'both', 'min', 'max'],
                                                index=0, placeholder="Choose an option", label_visibility="visible",
                                                key=f"{item}_extend")
            with col3:
                option["colorbar_position"] = st.selectbox('colorbar position', ['horizontal', 'vertical'],  # 'Season',
                                                           index=0, placeholder="Choose an option",
                                                           label_visibility="visible",
                                                           key=f"{item}_colorbar_position")
        if score in self.scores:
            option["extend"] = 'neither'

        col1, col2, col3, col4 = st.columns(4)
        vmin, vmax, colorbar_ticks = get_index(file, cases, items, score)
        option["vmin_max_on"] = col1.toggle('Fit to data', value=False, key=f"{item}_vmin_max_on")
        try:
            if option["vmin_max_on"]:
                option["colorbar_ticks"] = col2.number_input(f"Colorbar Ticks locater", value=colorbar_ticks, step=0.1,
                                                             key=f"{item}_colorbar_ticks")
                try:
                    option["vmin"] = col3.number_input(f"colorbar min", value=vmin, key=f"{item}_vmin")
                    option["vmax"] = col4.number_input(f"colorbar max", value=vmax, key=f"{item}_vmax")
                except ValueError:
                    st.error(f"Max value must larger than min value.")
            else:
                option["colorbar_ticks"] = colorbar_ticks
                option["vmin"] = vmin
                option["vmax"] = vmax
        except Exception as e:
            st.error(f"Error: {e}")
            error = True

        st.divider()
        st.write("##### :blue[Legend]")
        col1, col2, col3, col4 = st.columns(4)
        option["legend_box_x"] = col1.number_input(f"x position", value=1.1, step=0.1, key=f"{item}_legend_box_x")
        option["legend_box_y"] = col2.number_input(f"y position", value=1.2, step=0.1, key=f"{item}_legend_box_y")
        option["legend_box_size"] = col3.number_input(f"legend box size", value=1.0, step=0.1,
                                                      key=f"{item}_legend_box_size")
        option["legend_lw"] = col4.number_input(f"Line width", value=1.0, step=0.1, key=f"{item}_legend_lw")
        option["legend_fontsize"] = col1.number_input(f"Box fontsize", value=12.5, step=0.2,
                                                      key=f"{item}_legend_fontsize")
    with Save_tab:
        col1, col2, col3 = st.columns(3)
        x_lenth = 10
        y_lenth = 6
        if items and cases:
            x_lenth = len(items)
            y_lenth = len(cases)
        with col1:
            option["x_wise"] = st.number_input(f"X Length", min_value=0, value=x_lenth, key=f"{item}_x_wise")
            option['saving_format'] = st.selectbox('Image saving format', ['png', 'jpg', 'eps'],
                                                   index=1, placeholder="Choose an option", label_visibility="visible",
                                                   key=f"{item}_saving_format")
        with col2:
            option["y_wise"] = st.number_input(f"y Length", min_value=0, value=y_lenth, key=f"{item}_y_wise")
            option['font'] = st.selectbox('Image saving format',
                                          ['Times new roman', 'Arial', 'Courier New', 'Comic Sans MS', 'Verdana',
                                           'Helvetica',
                                           'Georgia', 'Tahoma', 'Trebuchet MS', 'Lucida Grande'],
                                          index=0, placeholder="Choose an option", label_visibility="visible",
                                          key=f"{item}_font")
        with col3:
            option['dpi'] = st.number_input(f"Figure dpi", min_value=0, value=300, key=f"{item}_dpi'")

    st.divider()
    if items and cases:
        draw_scenarios_comparison_Portrait_Plot_seasonal_by_score(Figure_show, option, file, score, items, cases)
    elif not items:
        st.error('Metircs items is None!')
    elif not cases:
        st.error('Simulation cases is None!')


def draw_scenarios_comparison_Portrait_Plot_seasonal(Figure_show, option, file, evaluation_items, reference, scores, cases, var):
    # ----------------------------------------------------------------------------------#
    #                                                                                  #
    #                                                                                  #
    #                               Start the main loop                                #
    #                                                                                  #
    #                                                                                  #
    # ----------------------------------------------------------------------------------#
    df = pd.read_csv(file, sep='\s+', header=0)

    unique_items = df['Item'].unique()
    item_references = df.groupby('Item')['Reference'].unique()
    sim_sources = df['Simulation'].unique()
    item = evaluation_items
    references = item_references[evaluation_items]
    # simulations = sim_sources[cases]

    data_score = np.zeros((4, len(scores), len(cases)))

    for k, season in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
        for i, score in enumerate(scores):
            for j, sim_source in enumerate(cases):
                try:
                    data_score[k, i, j] = \
                        df.loc[(df['Item'] == item) & (df['Reference'] == reference) & (df['Simulation'] == sim_source)][
                            f'{score}_{season}'].iloc[0]
                except IndexError:
                    data_score[k, i, j] = np.nan

    xaxis_labels = cases
    yaxis_labels = [score.replace("_", " ") for score in scores]

    font = {'family': option['font']}
    matplotlib.rc('font', **font)

    params = {'backend': 'ps',
              'axes.linewidth': option['axes_linewidth'],
              'font.size': 15,
              'xtick.labelsize': option['xticksize'],
              'xtick.direction': 'out',
              'ytick.labelsize': option['yticksize'],
              'ytick.direction': 'out',
              'savefig.bbox': 'tight',
              'axes.unicode_minus': False,
              'text.usetex': False}
    rcParams.update(params)
    # Set figure size
    figsize = (option['x_wise'], option['y_wise'])

    fig, ax, cbar = portrait_plot(data_score,
                                  xaxis_labels=xaxis_labels,
                                  yaxis_labels=yaxis_labels,
                                  cbar_label=option['colorbar_label'],
                                  box_as_square=True,
                                  vrange=(option["vmin"], option["vmax"]),
                                  figsize=figsize,
                                  xaxis_fontsize=option['xticksize'],
                                  colorbar_off=option['colorbar_off'],
                                  cmap=option['cmap'],
                                  colorbar_ticks=option['nstep'],
                                  cbar_kw={"orientation": option["colorbar_position"], 'extend': option["extend"]},
                                  cbar_label_fontsize=option["fontsize"],
                                  cbar_tick_fontsize=option["fontsize"],
                                  missing_color='grey',
                                  legend_on=True,
                                  legend_labels=['DJF', 'MAM', 'JJA', 'SON'],
                                  legend_box_xy=(option["legend_box_x"], option["legend_box_y"]),
                                  legend_box_size=option["legend_box_size"],
                                  legend_lw=option["legend_lw"],
                                  legend_fontsize=option["legend_fontsize"])
    # Rotate x-axis labels for better readability
    ax.set_xticklabels(xaxis_labels, rotation=option['x_rotation'], ha=option['x_ha'], fontsize=option['xticksize'])
    ax.set_yticklabels(yaxis_labels, rotation=option['y_rotation'], ha=option['y_ha'], fontsize=option['yticksize'])
    ax.set_ylabel(option['yticklabel'], fontsize=option['yticksize'] + 1)
    ax.set_xlabel(option['xticklabel'], fontsize=option['xticksize'] + 1)
    ax.set_title(option['title'], fontsize=option['title_size'])

    Figure_show.pyplot(fig)

    # # Save the plot
    filename = f'{item}_{reference}'
    # 将图像保存到 BytesIO 对象
    buffer = BytesIO()
    fig.savefig(buffer, format=option['saving_format'], dpi=option['dpi'])  # f
    buffer.seek(0)
    buffer.seek(0)
    st.download_button('Download image', buffer, file_name=f'{filename}_{var}.{option["saving_format"]}',
                       mime=f"image/{option['saving_format']}",
                       type="secondary", disabled=False, use_container_width=False)

    # delete the variables
    del df, unique_items, item_references, sim_sources, item, references, data_score, xaxis_labels, yaxis_labels, figsize, fig, ax, filename


def draw_scenarios_comparison_Portrait_Plot_seasonal_metrics(Figure_show, option, file, evaluation_items, reference, scores, cases, var):
    # ----------------------------------------------------------------------------------#
    #                                                                                  #
    #                                                                                  #
    #                               Start the main loop                                #
    #                                                                                  #
    #                                                                                  #
    # ----------------------------------------------------------------------------------#

    df = pd.read_csv(file, sep='\s+', header=0)

    unique_items = df['Item'].unique()
    item_references = df.groupby('Item')['Reference'].unique()
    sim_sources = df['Simulation'].unique()
    item = evaluation_items
    references = item_references[evaluation_items]
    # simulations = sim_sources[cases]

    font = {'family': option['font']}
    matplotlib.rc('font', **font)

    params = {'backend': 'ps',
              'axes.linewidth': option['axes_linewidth'],
              'font.size': 15,
              'xtick.labelsize': option['xticksize'],
              'xtick.direction': 'out',
              'ytick.labelsize': option['yticksize'],
              'ytick.direction': 'out',
              'savefig.bbox': 'tight',
              'axes.unicode_minus': False,
              'text.usetex': False}
    rcParams.update(params)
    # Set figure size
    figsize = (len(cases), len(scores))
    figure, axes = plt.subplots(nrows=len(scores), ncols=1, figsize=figsize, sharex=True)

    plt.subplots_adjust(hspace=0)  # -0.91

    data_score = np.zeros((4, len(scores), 1, len(cases)))
    for k, season in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
        for i, score in enumerate(scores):
            for j, sim_source in enumerate(cases):
                try:
                    data_score[k, i, 0, j] = \
                        df.loc[(df['Item'] == item) & (df['Reference'] == reference) & (df['Simulation'] == sim_source)][
                            f'{score}_{season}'].iloc[0]
                except IndexError:
                    data_score[k, i, 0, j] = np.nan
    min_ytick_left = 0
    for i, score in enumerate(scores):
        xaxis_labels = cases
        yaxis_labels = [scores[i].replace("_", " ")]
        vmin, vmax, colorbar_ticks = get_min_max(data_score[:, i, :], score)
        fig, ax, cbar = portrait_plot(data_score[:, i, :],
                                      fig=figure,
                                      ax=axes[i],
                                      xaxis_labels=xaxis_labels,
                                      yaxis_labels=yaxis_labels,
                                      box_as_square=True,
                                      figsize=figsize,
                                      vrange=(vmin, vmax),
                                      colorbar_ticks=colorbar_ticks,
                                      xaxis_fontsize=option['xticksize'],
                                      yaxis_fontsize=option['yticksize'],
                                      colorbar_off=option['colorbar_off'],
                                      cmap=option['cmap'],
                                      cbar_kw={"orientation": option["colorbar_position"]},
                                      cbar_label_fontsize=option["fontsize"],
                                      cbar_tick_fontsize=option["fontsize"],
                                      missing_color='grey',
                                      legend_on=False,
                                      legend_labels=['DJF', 'MAM', 'JJA', 'SON'],
                                      legend_box_xy=(option["legend_box_x"], option["legend_box_y"]),
                                      legend_box_size=option["legend_box_size"],
                                      legend_lw=option["legend_lw"],
                                      legend_fontsize=option["legend_fontsize"],
                                      use_axes=True,
                                      ifigure=i,
                                      )
        axes[i] = ax
        axes[i].set_yticklabels(yaxis_labels, rotation=option['y_rotation'], ha=option['y_ha'], fontsize=option['yticksize'])

    # Rotate x-axis labels for better readability
    ax.set_xticklabels(xaxis_labels, rotation=option['x_rotation'], ha=option['x_ha'], fontsize=option['xticksize'])
    ax.set_xlabel(option['xticklabel'], fontsize=option['xticksize'] + 1)
    axes[0].set_title(option['title'], fontsize=option['title_size'])

    add_legend(
        4,
        axes[0],
        (option["legend_box_x"], option["legend_box_y"]),
        option["legend_box_size"],
        labels=['DJF', 'MAM', 'JJA', 'SON'],
        lw=option["legend_lw"],
        fontsize=option["legend_fontsize"], )
    Figure_show.pyplot(figure)

    # Save the plot
    filename = f'{item}_{references}'
    # 将图像保存到 BytesIO 对象
    buffer = BytesIO()
    figure.savefig(buffer, format=option['saving_format'], dpi=option['dpi'])  # f
    buffer.seek(0)
    buffer.seek(0)
    st.download_button('Download image', buffer, file_name=f'{filename}_{var}.{option["saving_format"]}',
                       mime=f"image/{option['saving_format']}",
                       type="secondary", disabled=False, use_container_width=False)

    # delete the variables
    del df, unique_items, item_references, sim_sources, item, references, data_score, xaxis_labels, yaxis_labels, figsize, fig, ax, filename


def draw_scenarios_comparison_Portrait_Plot_seasonal_by_score(Figure_show, option, file, score, select_items, sim_cases):
    # ----------------------------------------------------------------------------------#
    #                                                                                  #
    #                                                                                  #
    #                               Start the main loop                                #
    #                                                                                  #
    #                                                                                  #
    # ----------------------------------------------------------------------------------#
    df = pd.read_csv(file, sep='\s+', header=0)
    df = df[df['Simulation'].isin(sim_cases)]

    filtered_df = df.groupby("Item")[["Reference"]].agg(lambda x: list(x.unique())).reset_index()
    filtered_df = filtered_df[filtered_df['Item'].isin(select_items)]
    unique_items = filtered_df['Item'].unique()
    sim_sources = df['Simulation'].unique()
    all_combinations = list(itertools.product(*filtered_df['Reference']))

    if 'Portrait_Plot_seasonal_score_next' not in st.session_state:
        st.session_state['Portrait_Plot_seasonal_score_next'] = 0

    if st.session_state.get('seasonal_Next_item', False):
        st.session_state['Portrait_Plot_seasonal_score_next'] = (st.session_state.get('Portrait_Plot_seasonal_score_next',
                                                                                      0) + 1) % len(all_combinations)
        i = st.session_state['Portrait_Plot_seasonal_score_next']
    else:
        i = 0
    item_combination = all_combinations[i]
    Figure_show.write(f"##### :green[Showing for {', '.join(item_combination)}]")
    # Iterate over each `item_combination` in the generated combinations.
    # Create a boolean mask to filter rows where `Item` and `Reference` match the current combination.
    mask = pd.Series(False, index=df.index)
    for i, item in enumerate(unique_items):
        mask |= (df['Item'] == item) & (df['Reference'] == item_combination[i])

    # Filter the DataFrame based on the boolean mask.
    filtered_df = df[mask]
    # Initialize data_score array
    data_score = np.zeros((4, len(unique_items), len(sim_sources)))
    # Fill data_score array with corresponding values
    for k, season in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
        for i, uitem in enumerate(unique_items):
            for j, sim_source in enumerate(sim_sources):
                try:
                    data_score[k, i, j] = \
                        filtered_df.loc[(filtered_df['Item'] == uitem) & (filtered_df['Simulation'] == sim_source)][
                            f'{score}_{season}'].iloc[0]
                except IndexError:
                    data_score[k, i, j] = np.nan

    # Set x-axis and y-axis labels
    xaxis_labels = sim_sources
    yaxis_labels = [unique_item.replace("_", " ") for unique_item in unique_items]

    # Set figure size
    font = {'family': option['font']}
    matplotlib.rc('font', **font)

    params = {'backend': 'ps',
              'axes.linewidth': option['axes_linewidth'],
              'font.size': 15,
              'xtick.labelsize': option['xticksize'],
              'xtick.direction': 'out',
              'ytick.labelsize': option['yticksize'],
              'ytick.direction': 'out',
              'savefig.bbox': 'tight',
              'axes.unicode_minus': False,
              'text.usetex': False}
    rcParams.update(params)
    # Set figure size
    figsize = (option['x_wise'], option['y_wise'])
    fig, ax, cbar = portrait_plot(data_score,
                                  xaxis_labels=xaxis_labels,
                                  yaxis_labels=yaxis_labels,
                                  # cbar_label=option['colorbar_label'],
                                  box_as_square=True,
                                  vrange=(option["vmin"], option["vmax"]),
                                  figsize=figsize,
                                  xaxis_fontsize=option['xticksize'],
                                  colorbar_off=option['colorbar_off'],
                                  cmap=option['cmap'],
                                  cbar_kw={"orientation": option["colorbar_position"], 'extend': option["extend"]},
                                  cbar_label_fontsize=option["fontsize"],
                                  cbar_tick_fontsize=option["fontsize"],
                                  colorbar_ticks=option["colorbar_ticks"],
                                  missing_color='grey',
                                  legend_on=True,
                                  legend_labels=['DJF', 'MAM', 'JJA', 'SON'],
                                  legend_box_xy=(option["legend_box_x"], option["legend_box_y"]),
                                  legend_box_size=option["legend_box_size"],
                                  legend_lw=option["legend_lw"],
                                  legend_fontsize=option["legend_fontsize"],
                                  logo_off=True)

    # Rotate x-axis labels for better readability
    ax.set_xticklabels(xaxis_labels, rotation=option['x_rotation'], ha=option['x_ha'], fontsize=option['xticksize'])
    ax.set_yticklabels(yaxis_labels, rotation=option['y_rotation'], ha=option['y_ha'], fontsize=option['yticksize'])
    ax.set_ylabel(option['yticklabel'], fontsize=option['yticksize'] + 1)
    ax.set_xlabel(option['xticklabel'], fontsize=option['xticksize'] + 1)
    ax.set_title(option['title'], fontsize=option['title_size'])

    # Save the plot
    Figure_show.pyplot(fig)

    filename = f'{score}_{"_".join(item_combination)}'
    # # Save the plot
    buffer = BytesIO()
    fig.savefig(buffer, format=option['saving_format'], dpi=option['dpi'])  # f
    buffer.seek(0)
    buffer.seek(0)
    col1, col2, col3 = Figure_show.columns(3)
    st.download_button('Download image', buffer, file_name=f'{filename}.{option["saving_format"]}',
                       mime=f"image/{option['saving_format']}",
                       type="secondary", disabled=False, use_container_width=False)
    next_disable = False
    if len(all_combinations) <= 1:
        next_disable = True
    col3.button(f':point_right: Next Case', key='seasonal_Next_item', disabled=next_disable,
                help='Press to change reference items')
    del df, filtered_df, unique_items, sim_sources, all_combinations, score, item_combination, mask, data_score, xaxis_labels, yaxis_labels, figsize, fig, ax, filename

    # -------------------------------------------------------------------------------------------------------------------
    # end of the function

    # -------------------------------------------------------------------------------------------------------------------


def get_ticks(vmin, vmax):
    if 2 >= vmax - vmin > 1:
        colorbar_ticks = 0.2
    elif 5 >= vmax - vmin > 2:
        colorbar_ticks = 0.5
    elif 10 >= vmax - vmin > 5:
        colorbar_ticks = 1.
    elif 100 >= vmax - vmin > 10:
        colorbar_ticks = 5.
    elif 100 >= vmax - vmin > 50:
        colorbar_ticks = 20.
    elif 200 >= vmax - vmin > 100:
        colorbar_ticks = 20.
    elif 500 >= vmax - vmin > 200:
        colorbar_ticks = 50.
    elif 1000 >= vmax - vmin > 500:
        colorbar_ticks = 100.
    elif 2000 >= vmax - vmin > 1000:
        colorbar_ticks = 200.
    elif 10000 >= vmax - vmin > 2000:
        colorbar_ticks = 10 ** math.floor(math.log10(vmax - vmin)) / 2
    else:
        colorbar_ticks = 0.10
    return colorbar_ticks


def get_min_max(data, score):
    vmin, vmax = np.percentile(data, 5), np.percentile(data, 95)
    if vmin < -100: vmin = -100
    if vmax > 100: vmax = 100
    if score in ['RMSE', 'CRMSD', 'MSE', 'ubRMSE', 'nRMSE', 'mae', 'ssq', 've', 'apb']:
        vmin, vmax = 0, vmax
    elif score in ['NSE', 'LNSE', 'ubNSE', 'rNSE', 'wNSE', 'wsNSE']:  # [-@,1]
        vmin, vmax = vmin, 1
    elif score in ['percent_bias', 'bias', 'pc_bias', 'rSD', 'PBIAS_HF', 'PBIAS_LF']:  # [-@,@]
        vmin, vmax = vmin, vmax
    elif score in ['KGE', 'KGESS', 'correlation', 'kappa_coeff', 'rSpearman']:  # [-1,1]
        vmin, vmax = -1, 1
    elif score in st.session_state.scores:
        vmin, vmax = 0, 1
    colorbar_ticks = get_ticks(vmin, vmax)
    return vmin, vmax, colorbar_ticks


def get_index(file, sim_cases, select_items, score):
    df = pd.read_csv(file, sep='\s+', header=0)
    unique_items = df['Item'].unique()
    item_references = df.groupby('Item')['Reference'].unique()
    sim_sources = df['Simulation'].unique()
    df = df[df['Simulation'].isin(sim_cases)]

    filtered_df = df.groupby("Item")[["Reference"]].agg(lambda x: list(x.unique())).reset_index()
    filtered_df = filtered_df[filtered_df['Item'].isin(select_items)]
    all_combinations = list(itertools.product(*filtered_df['Reference']))

    def remove_outliers(item_combination):
        mask = pd.Series(False, index=df.index)
        for i, item in enumerate(unique_items):
            mask |= (df['Item'] == item) & (df['Reference'] == item_combination[i])

        filtered_df = df[mask]
        data_score = np.zeros((4, len(unique_items), len(sim_sources)))
        for k, season in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
            for i, uitem in enumerate(unique_items):
                for j, sim_source in enumerate(sim_sources):
                    try:
                        data_score[k, i, j] = \
                            filtered_df.loc[(filtered_df['Item'] == uitem) & (filtered_df['Simulation'] == sim_source)][
                                f'{score}_{season}'].iloc[0]
                    except IndexError:
                        data_score[k, i, j] = np.nan
        vmin, vmax = np.percentile(data_score[~np.isnan(data_score)], 5), np.percentile(data_score[~np.isnan(data_score)], 95)
        return vmin, vmax

    bound = [remove_outliers(item_combination) for item_combination in all_combinations]
    vmax = max([d[1] for d in bound])
    vmin = min([d[0] for d in bound])

    if vmin < -100: vmin = -100
    if vmax > 100: vmax = 100

    if score in ['RMSE', 'CRMSD', 'MSE', 'ubRMSE', 'nRMSE', 'mae', 'ssq', 've', 'apb']:
        vmin, vmax = 0, vmax
    elif score in ['NSE', 'LNSE', 'ubNSE', 'rNSE', 'wNSE', 'wsNSE']:  # [-@,1]
        vmin, vmax = vmin, 1
    elif score in ['percent_bias', 'bias', 'pc_bias', 'rSD', 'PBIAS_HF', 'PBIAS_LF']:  # [-@,@]
        vmin, vmax = vmin, vmax
    elif score in ['KGE', 'KGESS', 'correlation', 'kappa_coeff', 'rSpearman']:  # [-1,1]
        vmin, vmax = -1, 1
    elif score in st.session_state.scores:
        vmin, vmax = 0, 1
    colorbar_ticks = get_ticks(vmin, vmax)

    return vmin, vmax, colorbar_ticks


def portrait_plot(
        data,
        xaxis_labels,
        yaxis_labels,
        fig=None,
        ax=None,
        annotate=False,
        annotate_data=None,
        annotate_textcolors=("black", "white"),
        annotate_textcolors_threshold=(-2, 2),
        annotate_fontsize=15,
        annotate_format="{x:.2f}",
        figsize=(12, 10),
        vrange=None,
        xaxis_fontsize=15,
        yaxis_fontsize=15,
        xaxis_tick_labels_top_and_bottom=False,
        xticklabel_rotation=45,
        inner_line_color="k",
        inner_line_width=0.5,
        cmap="RdBu_r",
        cmap_bounds=None,
        cbar_label=None,
        cbar_label_fontsize=15,
        cbar_tick_fontsize=12,
        colorbar_ticks=0.2,
        cbar_kw={},
        colorbar_off=False,
        missing_color="grey",
        invert_yaxis=True,
        box_as_square=False,
        legend_on=False,
        legend_labels=None,
        legend_box_xy=None,
        legend_box_size=None,
        legend_lw=1,
        legend_fontsize=14,
        logo_rect=None,
        logo_off=False,
        debug=False,
        use_axes=False,
        ifigure=None,
):
    """
    Parameters
    ----------
    - `data`: 2d numpy array, a list of 2d numpy arrays, or a 3d numpy array (i.e. stacked 2d numpy arrays)
    - `xaxis_labels`: list of strings, labels for xaixs. Number of list element must consistent to x-axis,
                    or 0 (empty list) to turn off xaxis tick labels
    - `yaxis_labels`: list of strings, labels for yaxis. Number of list element must consistent to y-axis,
                    or 0 (empty list) to turn off yaxis tick labels
    - `fig`: `matplotlib.figure` instance to which the portrait plot is plotted.
            If not provided, use current axes or create a new one.  Optional.
    - `ax`: `matplotlib.axes.Axes` instance to which the portrait plot is plotted.
            If not provided, use current axes or create a new one.  Optional.
    - `annotate`: bool, default=False, add annotating text if true,
                but work only for heatmap style map (i.e., no triangles)
    - `annotate_data`: 2d numpy array, default=None. If None, the image's data is used.  Optional.
    - `annotate_textcolors`: Tuple. A pair of colors for annotation text. Default is ("black", "white")
    - `annotate_textcolors_threshold`: Tuple or float. Value in data units according to which the colors from textcolors are applied. Default=(-2, 2)
    - `annotate_fontsize`: number (int/float), default=15. Font size for annotation
    - `annotate_format`: format for annotate value, default="{x:.2f}"
    - `figsize`: tuple of two numbers (width, height), default=(12, 10), figure size in inches
    - `vrange`: tuple of two numbers, range of value for colorbar.  Optional.
    - `xaxis_fontsize`: number, default=15, font size for xaxis tick labels.  Optional.
    - `yaxis_fontsize`: number, default=15, font size for yaxis tick labels.  Optional.
    - `xaxis_tick_labels_top_and_bottom`: bool, default=False, if true duplicate xaxis tick label to the other side.  Optional.
    - `xticklabel_rotation`: int or float, default=45, degree of angle to rotate x-axis tick label.  Optional
    - `inner_line_color`: string, default="k" (black), color for inner lines (triangle edge lines).  Optional.
    - `inner_line_width`: float, default=0.5, line width for inner lines (triangle edge lines).  Optional.
    - `cmap`: string, default="RdBu_r", name of matplotlib colormap.  Optional.
    - `cmap_bounds`: list of numbers.  If given, discrete colors are applied.  Optional.
    - `cbar_label`: string, default=None, label for colorbar.  Optional.
    - `cbar_label_fontsize`: number, default=15, font size for colorbar labels.  Optional.
    - `cbar_tick_fontsize`: number, default=12, font size for colorbar tick labels.  Optional.
    - `cbar_kw`: A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    - `colorbar_off`: Trun off colorbar if True.  Optional.
    - `missing_color`: color, default="grey", `matplotlib.axes.Axes.set_facecolor` parameter.  Optional.
    - `invert_yaxis`: bool, default=True, place y=0 at top on the plot.  Optional.
    - `box_as_square`: bool, default=False, make each box as square.  Optional.
    - `legend_on`: bool, default=False, show legend (only for 2 or 4 triangles portrait plot).  Optional.
    - `legend_labels`: list of strings, legend labels for triangls.  Optional.
    - `legend_box_xy`: tuple of numbers, position of legend box's upper-left corner.  Optional.
                    (lower-left if `invert_yaxis=False`), in `axes` coordinate.  Optional.
    - `legend_box_size`: number, size of legend box.  Optional.
    - `legend_lw`: number, line width of legend, default=1.  Optional.
    - `legend_fontsize`: number, font size for legend, default=14.  Optional.
    - `logo_rect`: sequence of float. The dimensions [left, bottom, width, height] of the the PMP logo.  Optional.
                All quantities are in fractions of figure width and height.  Optional
    - `logo_off`: bool, default=False, turn off PMP logo.  Optional.
    - `debug`: bool, default=False, if true print more message when running that help debugging.  Optional.

    Return
    ------
    - `fig`: matplotlib component for figure
    - `ax`: matplotlib component for axis
    - `cbar`: matplotlib component for colorbar (not returned if colorbar_off=True)

    Author: Jiwoo Lee @ LLNL (2021. 7)
    Last update: 2022. 10
    """

    # ----------------
    # Prepare plotting
    # ----------------
    data, num_divide = prepare_data(data, xaxis_labels, yaxis_labels, debug=debug)

    if num_divide not in [1, 2, 4]:
        sys.exit("Error: Number of (stacked) array is not 1, 2, or 4.")

    if annotate:
        if annotate_data is None:
            annotate_data = data
            num_divide_annotate = num_divide
        else:
            annotate_data, num_divide_annotate = prepare_data(
                annotate_data, xaxis_labels, yaxis_labels, debug=debug
            )
            if num_divide_annotate != num_divide:
                sys.exit("Error: annotate_data does not have same size as data")

    # ----------------
    # Ready to plot!!
    # ----------------
    if fig is None:
        fig = plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.add_subplot(111)

    ax.set_facecolor(missing_color)

    if "extend" in list(cbar_kw.keys()):
        extend = cbar_kw["extend"]
    else:
        extend = "neither"

    if vrange is None:
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
        if use_axes:
            vmin, vmax = np.percentile(data, [5, 95])
            min_value, max_value = np.nanmin(data), np.nanmax(data)
            if min_value < vmin and max_value > vmax:
                extend = 'both'
            elif min_value > vmin and max_value > vmax:
                extend = 'max'
            elif min_value < vmin and max_value < vmax:
                extend = 'min'
            else:
                extend = 'neither'
    else:
        vmin = min(vrange)
        vmax = max(vrange)
        if "extend" in list(cbar_kw.keys()):
            extend = cbar_kw["extend"]

    # Normalize colorbar
    if cmap_bounds is None:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        cmap = plt.get_cmap(cmap)
        norm = matplotlib.colors.BoundaryNorm(cmap_bounds, cmap.N)
    # [1] Heatmap-style portrait plot (no triangles)
    if num_divide == 1:
        ax, im = heatmap(
            data,
            xaxis_labels,
            yaxis_labels,
            ax=ax,
            invert_yaxis=invert_yaxis,
            cmap=cmap,
            edgecolors="k",
            linewidth=0.5,
            norm=norm,
        )
        if annotate:
            if annotate_data is not None:
                if annotate_data.shape != data.shape:
                    sys.exit("Error: annotate_data has different size than data")
            else:
                annotate_data = data
            ax = annotate_heatmap(
                im,
                ax=ax,
                data=data,
                annotate_data=annotate_data,
                valfmt=annotate_format,
                textcolors=annotate_textcolors,
                threshold=annotate_textcolors_threshold,
                fontsize=annotate_fontsize,
            )

    # [2] Two triangle portrait plot
    elif num_divide == 2:
        # data order is upper, lower
        upper = data[0]
        lower = data[1]
        ax, im = triamatrix_wrap_up(
            upper,
            lower,
            ax,
            xaxis_labels=xaxis_labels,
            yaxis_labels=yaxis_labels,
            cmap=cmap,
            invert_yaxis=invert_yaxis,
            norm=norm,
            inner_line_color=inner_line_color,
            inner_line_width=inner_line_width,
        )

    # [4] Four triangle portrait plot
    elif num_divide == 4:
        # data order is clockwise from top: top, right, bottom, left
        top = data[0]
        right = data[1]
        bottom = data[2]
        left = data[3]
        ax, im = quatromatrix(
            top,
            right,
            bottom,
            left,
            ax=ax,
            tripcolorkw={
                "cmap": cmap,
                "norm": norm,
                "edgecolors": inner_line_color,
                "linewidth": inner_line_width,
            },
            xaxis_labels=xaxis_labels,
            yaxis_labels=yaxis_labels,
            invert_yaxis=invert_yaxis,
        )

    pos = ax.get_position()
    left, right, bottom, width, height = pos.x0, pos.x1, pos.y0, pos.width, pos.height

    # X-axis tick labels
    if xaxis_tick_labels_top_and_bottom:
        # additional x-axis tick labels
        ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
    else:
        # Let the horizontal axes labeling appear on top.
        if use_axes:
            if ifigure == 0:
                ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
            else:
                ax.tick_params(top=False, bottom=False, labeltop=False, labelbottom=False)
        else:
            ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    """
    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(),
        fontsize=xaxis_fontsize,
        rotation=-30,
        ha="right",
        rotation_mode="anchor",
    )
    """
    # Rotate and align top ticklabels
    plt.setp(
        [tick.label2 for tick in ax.xaxis.get_major_ticks()],
        rotation=xticklabel_rotation,
        ha="left",
        va="center",
        rotation_mode="anchor",
        fontsize=xaxis_fontsize,
    )

    if xaxis_tick_labels_top_and_bottom:
        # Rotate and align bottom ticklabels
        plt.setp(
            [tick.label1 for tick in ax.xaxis.get_major_ticks()],
            rotation=xticklabel_rotation,
            ha="right",
            va="center",
            rotation_mode="anchor",
            fontsize=xaxis_fontsize,
        )

    # Set font size for yaxis tick labels
    plt.setp(ax.get_yticklabels(), fontsize=yaxis_fontsize)

    # Legend
    if legend_on:
        if legend_labels is None:
            sys.exit("Error: legend_labels was not provided.")
        else:
            if not use_axes:
                add_legend(
                    num_divide,
                    ax,
                    legend_box_xy,
                    legend_box_size,
                    labels=legend_labels,
                    lw=legend_lw,
                    fontsize=legend_fontsize,
                )

    if box_as_square:
        ax.set_aspect("equal")

    if not colorbar_off:
        if not use_axes:
            ticks = matplotlib.ticker.MultipleLocator(base=colorbar_ticks)
            mticks = ticks.tick_values(vmin=vmin, vmax=vmax)
            pos = ax.get_position()
            left, right, bottom, width, height = pos.x0, pos.x1, pos.y0, pos.width, pos.height
            if cbar_kw["orientation"] == 'vertical':
                if len(yaxis_labels) <= 6:
                    cbar_ax = fig.add_axes([right + 0.05, bottom, 0.03, height])  # right + 0.2
                else:
                    cbar_ax = fig.add_axes([right + 0.05, bottom + height / 6, 0.03, height / 3 * 2])  # right + 0.2
            else:
                if len(xaxis_labels) <= 6:
                    cbar_ax = fig.add_axes([left, bottom - 0.15, width, 0.03])
                else:
                    cbar_ax = fig.add_axes([left + width / 6, bottom - 0.15, width / 3 * 2, 0.05])
            cbar = ax.figure.colorbar(im, cax=cbar_ax, ticks=mticks, **cbar_kw, )
        else:
            ticks = matplotlib.ticker.MultipleLocator(base=colorbar_ticks * 2)
            mticks = ticks.tick_values(vmin=vmin, vmax=vmax)
            cbar_kw["orientation"] = "horizontal"
            w = height * 1.5
            if len(xaxis_labels) <= 5:
                w = height * 2
            cbar_ax = fig.add_axes([right + 0.08, bottom + height / 2, w, height / 4])
            # Label for colorbar
            cbar = ax.figure.colorbar(im, cax=cbar_ax, ticks=mticks, **cbar_kw)
        if cbar_label is not None:
            if "orientation" in list(cbar_kw.keys()):
                if cbar_kw["orientation"] == "horizontal":
                    rotation = 0
                    ha = "center"
                    va = "top"
                    cbar.ax.set_xlabel(
                        cbar_label,
                        rotation=rotation,
                        ha=ha,
                        va=va,
                        fontsize=cbar_label_fontsize,
                    )
                else:
                    rotation = -90
                    ha = "center"
                    va = "bottom"
                    cbar.ax.set_ylabel(
                        cbar_label,
                        rotation=rotation,
                        ha=ha,
                        va=va,
                        fontsize=cbar_label_fontsize,
                    )
            else:
                rotation = -90
                ha = "center"
                va = "bottom"
                cbar.ax.set_ylabel(
                    cbar_label,
                    rotation=rotation,
                    ha=ha,
                    va=va,
                    fontsize=cbar_label_fontsize,
                )
            cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
            return fig, ax, cbar
        else:
            rotation = 0
            ha = "center"
            va = "top"
            cbar.ax.set_xlabel(
                cbar_label,
                rotation=rotation,
                ha=ha,
                va=va,
                fontsize=cbar_label_fontsize,
            )
            cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
            return fig, ax, cbar
    else:
        return fig, ax, 'cbar'


# ======================================================================
# Prepare data
# ----------------------------------------------------------------------
def prepare_data(data, xaxis_labels, yaxis_labels, debug=False):
    # In case data was given as list of arrays, convert it to numpy (stacked) array
    if isinstance(data, list):
        if debug:
            print("data type is list")
            print("len(data):", len(data))
        if len(data) == 1:  # list has only 1 array as element
            if isinstance(data[0], np.ndarray) and (len(data[0].shape) == 2):
                data = data[0]
                num_divide = 1
            else:
                sys.exit("Error: Element of given list is not in np.ndarray type")
        else:  # list has more than 1 arrays as elements
            data = np.stack(data)
            num_divide = len(data)

    # Now, data is expected to be  a numpy array (whether given or converted from list)
    if debug:
        print("data.shape:", data.shape)

    if data.shape[-1] != len(xaxis_labels) and len(xaxis_labels) > 0:
        sys.exit("Error: Number of elements in xaxis_label mismatchs to the data")
    if data.shape[-2] != len(yaxis_labels) and len(yaxis_labels) > 0:
        sys.exit("Error: Number of elements in yaxis_label mismatchs to the data")

    if isinstance(data, np.ndarray):
        # data = np.squeeze(data)
        if len(data.shape) == 2:
            num_divide = 1
            st.write(num_divide)
        elif len(data.shape) == 3:
            num_divide = data.shape[0]
        else:
            print("data.shape:", data.shape)
            sys.exit("Error: data.shape is not right")
    else:
        sys.exit("Error: Converted or given data is not in np.ndarray type")

    if debug:
        print("num_divide:", num_divide)

    return data, num_divide


# ======================================================================
# Portrait plot 1: heatmap-style (no triangle)
# (Inspired from: https://matplotlib.org/devdocs/gallery/images_contours_and_fields/image_annotated_heatmap.html)
# ----------------------------------------------------------------------
def heatmap(data, xaxis_labels, yaxis_labels, ax=None, invert_yaxis=False, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    yaxis_labels
        A list or array of length M with the labels for the rows.
    xaxis_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    invert_yaxis
        A bool to decide top-down or bottom-up order on y-axis
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if invert_yaxis:
        ax.invert_yaxis()

    # Plot the heatmap
    im = ax.pcolormesh(data, **kwargs)

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_xticklabels(xaxis_labels)
    ax.set_yticklabels(yaxis_labels)
    ax.tick_params(which="minor", bottom=False, left=False)

    return ax, im


def annotate_heatmap(
        im,
        ax,
        data=None,
        annotate_data=None,
        valfmt="{x:.2f}",
        textcolors=("black", "white"),
        threshold=None,
        **textkw,
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    ax
        Matplotlib axis
    data
        Data used to color in the image.  If None, the image's data is used.  Optional.
    annotate_data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array().reshape(im._meshHeight, im._meshWidth)

    if annotate_data is None:
        annotate_data = data

    if threshold is None:
        threshold = (data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if type(threshold) is tuple:
                kw.update(
                    color=textcolors[
                        int(
                            (data[i, j] > max(threshold))
                            or (data[i, j] < min(threshold))
                        )
                    ]
                )
            else:
                kw.update(color=textcolors[int(data[i, j] > threshold)])
            text = ax.text(j + 0.5, i + 0.5, valfmt(annotate_data[i, j], None), **kw)
            texts.append(text)

    return ax


# ======================================================================
# Portrait plot 2 (two triangles)
# (Inspired from: https://stackoverflow.com/questions/44291155/plotting-two-distance-matrices-together-on-same-plot)
# ----------------------------------------------------------------------
def triamatrix_wrap_up(
        upper,
        lower,
        ax,
        xaxis_labels,
        yaxis_labels,
        cmap="viridis",
        vmin=-3,
        vmax=3,
        norm=None,
        invert_yaxis=True,
        inner_line_color="k",
        inner_line_width=0.5,
):
    # Colorbar range
    if norm is None:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    # Triangles
    im = triamatrix(
        upper,
        ax,
        rot=270,
        cmap=cmap,
        norm=norm,
        edgecolors=inner_line_color,
        lw=inner_line_width,
    )
    im = triamatrix(
        lower,
        ax,
        rot=90,
        cmap=cmap,
        norm=norm,
        edgecolors=inner_line_color,
        lw=inner_line_width,
    )
    ax.set_xlim(-0.5, upper.shape[1] - 0.5)
    ax.set_ylim(-0.5, upper.shape[0] - 0.5)

    if invert_yaxis:
        ax.invert_yaxis()

    ax.set_xticks(np.arange(upper.shape[1]))
    ax.set_yticks(np.arange(upper.shape[0]))

    ax.set_xticklabels(xaxis_labels)
    ax.set_yticklabels(yaxis_labels)

    return ax, im


def triatpos(pos=(0, 0), rot=0):
    r = np.array([[-1, -1], [1, -1], [1, 1], [-1, -1]]) * 0.5
    rm = [
        [np.cos(np.deg2rad(rot)), -np.sin(np.deg2rad(rot))],
        [np.sin(np.deg2rad(rot)), np.cos(np.deg2rad(rot))],
    ]
    r = np.dot(rm, r.T).T
    r[:, 0] += pos[0]
    r[:, 1] += pos[1]
    return r


def triamatrix(a, ax, rot=0, cmap="viridis", **kwargs):
    segs = []
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            segs.append(triatpos((j, i), rot=rot))
    col = collections.PolyCollection(segs, cmap=cmap, **kwargs)
    col.set_array(a.flatten())
    ax.add_collection(col)
    return col


# ======================================================================
# Portrait plot 4 (four triangles)
# (Inspired from: https://stackoverflow.com/questions/44666679/something-like-plt-matshow-but-with-triangles)
# ----------------------------------------------------------------------
def quatromatrix(
        top,
        right,
        bottom,
        left,
        ax=None,
        tripcolorkw={},
        xaxis_labels=None,
        yaxis_labels=None,
        invert_yaxis=True,
):
    if ax is None:
        ax = plt.gca()

    n = left.shape[0]
    m = left.shape[1]

    a = np.array([[0, 0], [0, 1], [0.5, 0.5], [1, 0], [1, 1]])
    tr = np.array([[0, 1, 2], [0, 2, 3], [2, 3, 4], [1, 2, 4]])

    A = np.zeros((n * m * 5, 2))
    Tr = np.zeros((n * m * 4, 3))

    for i in range(n):
        for j in range(m):
            k = i * m + j
            A[k * 5: (k + 1) * 5, :] = np.c_[a[:, 0] + j, a[:, 1] + i]
            Tr[k * 4: (k + 1) * 4, :] = tr + k * 5

    if invert_yaxis:
        ax.invert_yaxis()
        C = np.c_[
            left.flatten(), top.flatten(), right.flatten(), bottom.flatten()
        ].flatten()
    else:
        C = np.c_[
            left.flatten(), bottom.flatten(), right.flatten(), top.flatten()
        ].flatten()

    # Prevent coloring missing data
    C = np.ma.array(C, mask=np.isnan(C))

    tripcolor = ax.tripcolor(A[:, 0], A[:, 1], Tr, facecolors=C, **tripcolorkw)

    ax.margins(0)

    if xaxis_labels is not None:
        x_loc = list_between_elements(np.arange(left.shape[1] + 1))
        ax.set_xticks(x_loc)
        ax.set_xticklabels(xaxis_labels)
    if yaxis_labels is not None:
        y_loc = list_between_elements(np.arange(left.shape[0] + 1))
        ax.set_yticks(y_loc)
        ax.set_yticklabels(yaxis_labels)

    return ax, tripcolor


def list_between_elements(a):
    a_between = []
    for i in range(len(a)):
        try:
            tmp = (a[i] + a[i + 1]) / 2.0
            a_between.append(tmp)
        except Exception:
            pass
    return a_between


# ======================================================================
# Portrait plot legend (four/two triangles)
# ======================================================================
def add_legend(
        num_divide, ax, box_xy=None, box_size=None, labels=None, lw=1, fontsize=14
):
    if box_xy is None:
        box_x = ax.get_xlim()[1] * 1.25
        box_y = ax.get_ylim()[1]
    else:
        # Convert axes coordinate to data coordinate
        # Ref: https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html
        box_x, box_y = ax.transLimits.inverted().transform(box_xy)

    if box_size is None:
        box_size = 1.5

    if num_divide == 4:
        if labels is None:
            labels = ["TOP", "RIGHT", "BOTTOM", "LEFT"]
        ax.add_patch(
            plt.Polygon(
                [
                    [box_x, box_y],
                    [box_x + box_size / 2.0, box_y + box_size / 2],
                    [box_x + box_size, box_y],
                ],
                color="k",
                fill=False,
                clip_on=False,
                lw=lw,
            )
        )
        ax.add_patch(
            plt.Polygon(
                [
                    [box_x + box_size, box_y],
                    [box_x + box_size / 2.0, box_y + box_size / 2],
                    [box_x + box_size, box_y + box_size],
                ],
                color="k",
                fill=False,
                clip_on=False,
                lw=lw,
            )
        )
        ax.add_patch(
            plt.Polygon(
                [
                    [box_x + box_size, box_y + box_size],
                    [box_x + box_size / 2.0, box_y + box_size / 2],
                    [box_x, box_y + box_size],
                ],
                color="k",
                fill=False,
                clip_on=False,
                lw=lw,
            )
        )
        ax.add_patch(
            plt.Polygon(
                [
                    [box_x, box_y],
                    [box_x + box_size / 2.0, box_y + box_size / 2],
                    [box_x, box_y + box_size],
                ],
                color="k",
                fill=False,
                clip_on=False,
                lw=lw,
            )
        )
        ax.text(
            box_x + box_size * 0.5,
            box_y + box_size * 0.2,
            labels[0],
            ha="center",
            va="center",
            fontsize=fontsize,
        )
        ax.text(
            box_x + box_size * 0.8,
            box_y + box_size * 0.5,
            labels[1],
            ha="center",
            va="center",
            fontsize=fontsize,
        )
        ax.text(
            box_x + box_size * 0.5,
            box_y + box_size * 0.8,
            labels[2],
            ha="center",
            va="center",
            fontsize=fontsize,
        )
        ax.text(
            box_x + box_size * 0.2,
            box_y + box_size * 0.5,
            labels[3],
            ha="center",
            va="center",
            fontsize=fontsize,
        )
    elif num_divide == 2:
        if labels is None:
            labels = ["UPPER", "LOWER"]
        ax.add_patch(
            plt.Polygon(
                [[box_x, box_y], [box_x, box_y + box_size], [box_x + box_size, box_y]],
                color="k",
                fill=False,
                clip_on=False,
                lw=lw,
            )
        )
        ax.add_patch(
            plt.Polygon(
                [
                    [box_x + box_size, box_y + box_size],
                    [box_x, box_y + box_size],
                    [box_x + box_size, box_y],
                ],
                color="k",
                fill=False,
                clip_on=False,
                lw=lw,
            )
        )
        ax.text(
            box_x + box_size * 0.05,
            box_y + box_size * 0.2,
            labels[0],
            ha="left",
            va="center",
            fontsize=fontsize,
        )
        ax.text(
            box_x + box_size * 0.95,
            box_y + box_size * 0.8,
            labels[1],
            ha="right",
            va="center",
            fontsize=fontsize,
        )
