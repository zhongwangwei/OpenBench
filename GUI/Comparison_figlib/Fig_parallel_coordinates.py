import sys
import matplotlib
from matplotlib import rcParams
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
from matplotlib.cbook import flatten
import itertools

from io import BytesIO
import streamlit as st


def make_scenarios_comparison_parallel_coordinates(self, file, selected_item, score, ref_source, item):
    option = {}
    Figure_show = st.container()
    Labels_tab, Scale_tab, Var_tab, Save_tab = st.tabs(['Labels', 'Scale', 'Variables', 'Save'])

    with Labels_tab:
        option['title'] = st.text_input('Title',
                                        value=f"Parallel Coordinates Plot - {selected_item.replace('_', ' ')} - {ref_source}",
                                        label_visibility="visible", key=f"{item} title")

        col1, col2, col3, col4 = st.columns((3.5, 3, 3, 3))
        with col1:
            option['title_size'] = st.number_input("Title label size", min_value=0, value=20, key=f"{item}_title_size")
            option['fontsize'] = st.number_input("Font size", min_value=0, value=15, step=1, key=f"{item}_fontsize")

        with col2:
            option['xticklabel'] = st.text_input('X tick labels', value='', label_visibility="visible",
                                                 key=f"{item}_xticklabel")
            option['xtick'] = st.number_input("xtick label size", min_value=0, value=17, key=f"{item}_xtick")

        with col3:
            option['yticklabel'] = st.text_input('Y tick labels', value=f'', label_visibility="visible",
                                                 key=f"{item}_yticklabel")
            option['ytick'] = st.number_input("ytick label size", min_value=0, value=17, key=f"{item}_ytick")

        with col4:
            option['axes_linewidth'] = st.number_input("axes linewidth", min_value=0., value=1., step=0.1,
                                                       key=f"{item}_axes_linewidth")

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
                    case_item[item] = st.checkbox(item.replace("_", " "), key=f"{item}__Parallel_Coordinates_variable", value=case_item[item])
                if len([item for item, value in case_item.items() if value]) > 0:
                    return [item for item, value in case_item.items() if value]
                else:
                    st.error("You must choose one items!")
            else:
                for item in case_item:
                    case_item[item] = st.checkbox(item, key=f"{item}__Parallel_Coordinates_variable",
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
        option["legend_off"] = st.toggle('Turn off the legend?', value=False, key=f"{item}_legend_off")
        col1, col2, col3, col4 = st.columns((4, 4, 4, 4))
        if not option['legend_off']:
            option["legend_loc"] = col1.selectbox("Legend location",
                                                  ['upper center', 'best', 'upper right', 'upper left', 'lower left',
                                                   'lower right', 'right', 'center left', 'center right', 'lower center',
                                                   'center'], index=0,
                                                  placeholder="Choose an option",
                                                  label_visibility="visible", key=f"{item}_loc")
            option["legend_ncol"] = col2.number_input("N cols", value=3, min_value=1, format='%d', key=f"{item}_ncol")

            option["bbox_to_anchor_x"] = col3.number_input("X position of legend", value=0.5, step=0.1,
                                                           key=f"{item}_bbox_to_anchor_x")
            option["bbox_to_anchor_y"] = col4.number_input("Y position of legend", value=-0.14, step=0.1,
                                                           key=f"{item}_bbox_to_anchor_y")
        else:
            option["bbox_to_anchor_x"] = 0.5
            option["bbox_to_anchor_y"] = -0.15
            option["legend_ncol"] = 1
            option["legend_loc"] = 'best'

        option["models_to_highlight_by_line"] = st.toggle('Models highlight by line?', value=True,
                                                          key=f"{item}_models_to_highlight_by_line",
                                                          help='Turn off to show figure by markers')
        option["models_to_highlight_markers_size"] = 22
        option['models_to_highlight_markers_alpha'] = 1.0
        if not option["models_to_highlight_by_line"]:
            col1, col2, col3 = st.columns((4, 4, 4))
            col1.write("###### :orange[Marker style]")
            option['markers'] = col1.selectbox(f'Marker',
                                               ['.', 'x', 'o', '<', '8', 's', 'p', '*', 'h', 'H', 'D',
                                                'd', 'P', 'X'], key=f'{item} Marker',
                                               index=2, placeholder="Choose an option",
                                               label_visibility="collapsed")
            col2.write("###### :orange[Marker size]")
            option["models_to_highlight_markers_size"] = col2.number_input(f"markers size", value=22, step=1,
                                                                           key=f"{item}_markers_size",
                                                                           label_visibility="collapsed")
            col3.write("###### :orange[Marker alpha]")
            option['models_to_highlight_markers_alpha'] = col3.number_input(f"markers alpha",
                                                                            label_visibility="collapsed",
                                                                            key=f'{item} alpha',
                                                                            min_value=0., value=0.8, max_value=1.)

        colors = {}
        import matplotlib.colors as mcolors
        import itertools
        colors_list = [plt.get_cmap('Set3_r')(c) for c in np.linspace(0, 1, 11)]
        hex_colors = itertools.cycle([mcolors.rgb2hex(color) for color in colors_list])
        st.write("###### :orange[Colors]")
        cols = itertools.cycle(st.columns(3))
        if cases:
            for item_select in cases:
                col = next(cols)
                colors[f"{item_select}"] = col.color_picker(f'{item_select}', value=next(hex_colors),
                                                            key=f'{item} {item_select} colors', help=None,
                                                            on_change=None,
                                                            args=None, kwargs=None, disabled=False,
                                                            label_visibility="visible")

        option['colors'] = [color for color in colors.values()]

    with Save_tab:
        col1, col2, col3 = st.columns(3)
        with col1:
            x_lenth = 15
            if items:
                x_lenth = len(items) * 3
            option["x_wise"] = st.number_input(f"X Length", min_value=0, value=x_lenth, key=f"{item}_x_wise")
            option['saving_format'] = st.selectbox('Image saving format', ['png', 'jpg', 'eps'],
                                                   index=1, placeholder="Choose an option", label_visibility="visible",
                                                   key=f"{item}_saving_format")
        with col2:
            option["y_wise"] = st.number_input(f"y Length", min_value=0, value=5, key=f"{item}_y_wise")
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
        draw_scenarios_comparison_parallel_coordinates(Figure_show, option, file, selected_item, ref_source, items, cases, score)
    elif not items:
        st.error('Metircs items is None!')
    elif not cases:
        st.error('Simulation cases is None!')


def make_scenarios_comparison_parallel_coordinates_by_score(self, file, score, item):
    option = {}
    Figure_show = st.container()
    Labels_tab, Scale_tab, Var_tab, Save_tab = st.tabs(['Labels', 'Scale', 'Variables', 'Save'])
    with Labels_tab:
        option['title'] = st.text_input('Title',
                                        value=f"Parallel Coordinates Plot - {score.replace('_', ' ')}",
                                        label_visibility="visible", key=f"{item} title")
        col1, col2, col3, col4 = st.columns((3, 3, 3, 3))
        with col1:
            option['title_size'] = st.number_input("Title label size", min_value=0, value=20, key=f"{item}_title_size")
            option["fontsize"] = st.number_input(f"Font size", value=15., step=1., key=f"{item}_legend_fontsize")
        with col2:
            option['xticklabel'] = st.text_input('X tick labels', value='', label_visibility="visible",
                                                 key=f"{item}_xticklabel")
            option['xtick'] = st.number_input("xtick label size", min_value=0, value=17, key=f"{item}_xtick")

        with col3:
            option['yticklabel'] = st.text_input('Y tick labels', value=score.replace("_", " "),
                                                 label_visibility="visible",
                                                 key=f"{item}_yticklabel")
            option['ytick'] = st.number_input("ytick label size", min_value=0, value=17, key=f"{item}_ytick")

        with col4:
            option['axes_linewidth'] = st.number_input("axes linewidth", min_value=0., value=1., step=0.1,
                                                       key=f"{item}_axes_linewidth")

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
                    case_item[item] = st.checkbox(item.replace("_", " "), key=f"{item}__Parallel_Coordinates_score",
                                                  value=case_item[item])
                if len([item for item, value in case_item.items() if value]) > 0:
                    return [item for item, value in case_item.items() if value]
                else:
                    st.error("You must choose one item!")
            else:
                for item in case_item:
                    case_item[item] = st.checkbox(item, key=f"{item}__Parallel_Coordinates_score",
                                                  value=case_item[item])
                if len([item for item, value in case_item.items() if value]) > 0:
                    return [item for item, value in case_item.items() if value]
                else:
                    st.error("You must choose one item!")

        items = [k for k in self.selected_items]
        cases = list(set([value for key in self.selected_items for value in self.sim['general'][f"{key}_sim_source"] if value]))
        col1, col2 = st.columns(2)
        with col1:
            items = get_cases(items, 'Selected items')
        with col2:
            cases = get_cases(cases, 'cases')

    with Scale_tab:

        col1, col2, col3, col4 = st.columns((4, 4, 4, 4))
        option["x_rotation"] = col1.number_input(f"x rotation", min_value=-90, max_value=90, value=15,
                                                 key=f"{item}_x_rotation")
        option['x_ha'] = col2.selectbox('x ha', ['right', 'left', 'center'],
                                        index=0, placeholder="Choose an option", label_visibility="visible",
                                        key=f"{item}_x_ha")

        option["legend_off"] = st.toggle('Turn off the legend?', value=False, key=f"{item}_legend_off")
        col1, col2, col3, col4 = st.columns((4, 4, 4, 4))
        if not option['legend_off']:
            option["legend_loc"] = col1.selectbox("Legend location",
                                                  ['best', 'right', 'left', 'upper left', 'upper right', 'lower left',
                                                   'lower right',
                                                   'upper center',
                                                   'lower center', 'center left', 'center right'], index=7,
                                                  placeholder="Choose an option",
                                                  label_visibility="visible", key=f"{item}_loc")
            option["legend_ncol"] = col2.number_input("N cols", value=4, min_value=1, format='%d', key=f"{item}_ncol")

            option["bbox_to_anchor_x"] = col3.number_input("X position of legend", value=0.5, step=0.1,
                                                           key=f"{item}_bbox_to_anchor_x")
            option["bbox_to_anchor_y"] = col4.number_input("Y position of legend", value=-0.25, step=0.1,
                                                           key=f"{item}_bbox_to_anchor_y")
        else:
            option["bbox_to_anchor_x"] = 0.5
            option["bbox_to_anchor_y"] = -0.55
            option["legend_ncol"] = 1
            option["legend_loc"] = 'best'

        option["models_to_highlight_by_line"] = st.toggle('Models highlight by line?', value=True,
                                                          key=f"{item}_models_to_highlight_by_line",
                                                          help='Turn off to show figure by markers')
        option["models_to_highlight_markers_size"] = 22
        option['models_to_highlight_markers_alpha'] = 1.0
        if not option["models_to_highlight_by_line"]:
            col1, col2, col3 = st.columns((4, 4, 4))
            col1.write("###### :orange[Marker style]")
            option['markers'] = col1.selectbox(f'Marker',
                                               ['.', 'x', 'o', '<', '8', 's', 'p', '*', 'h', 'H', 'D',
                                                'd', 'P', 'X'], key=f'{item} Marker',
                                               index=2, placeholder="Choose an option",
                                               label_visibility="collapsed")
            col2.write("###### :orange[Marker size]")
            option["models_to_highlight_markers_size"] = col2.number_input(f"markers size", value=22, step=1,
                                                                           key=f"{item}_markers_size",
                                                                           label_visibility="collapsed")
            col3.write("###### :orange[Marker alpha]")
            option['models_to_highlight_markers_alpha'] = col3.number_input(f"markers alpha",
                                                                            label_visibility="collapsed",
                                                                            key=f'{item} alpha',
                                                                            min_value=0., value=0.8, max_value=1.)

        colors = {}

        import matplotlib.colors as mcolors

        colors_list = [plt.get_cmap('Set3_r')(c) for c in np.linspace(0, 1, 11)]
        hex_colors = itertools.cycle([mcolors.rgb2hex(color) for color in colors_list])
        st.write("###### :orange[Colors]")
        cols = itertools.cycle(st.columns(3))
        if cases:
            for item_select in cases:
                col = next(cols)
                colors[f"{item_select}"] = col.color_picker(f'{item_select}', value=next(hex_colors),
                                                            key=f'{item} {item_select} colors', help=None,
                                                            on_change=None,
                                                            args=None, kwargs=None, disabled=False,
                                                            label_visibility="visible")
        option['colors'] = [color for color in colors.values()]

    with Save_tab:
        col1, col2, col3 = st.columns(3)
        with col1:
            x_lenth = 15
            if items:
                x_lenth = len(items) * 3
            option["x_wise"] = st.number_input(f"X Length", min_value=0, value=x_lenth, key=f"{item}_x_wise")
            option['saving_format'] = st.selectbox('Image saving format', ['png', 'jpg', 'eps'],
                                                   index=1, placeholder="Choose an option", label_visibility="visible",
                                                   key=f"{item}_saving_format")
        with col2:
            option["y_wise"] = st.number_input(f"y Length", min_value=0, value=5, key=f"{item}_y_wise")
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
        draw_scenarios_comparison_parallel_coordinates_by_score(Figure_show, option, file, score, items, cases)
    elif not items:
        st.error('Metircs items is None!')
    elif not cases:
        st.error('Simulation cases is None!')


def draw_scenarios_comparison_parallel_coordinates(Figure_show, option, file, evaluation_item, ref_source, scores, sim_cases, var):
    # ----------------------------------------------------------------------------------#
    #                                                                                  #
    #                                                                                  #
    #                               Start the main loop                                #
    #                                                                                  #
    #                                                                                  #
    # ----------------------------------------------------------------------------------#
    #
    df = pd.read_csv(file, sep='\s+',
                     header=0)
    evaluation_items = df['Item'].unique()
    item_references = df.groupby('Item')['Reference'].unique()
    sim_sources = df['Simulation'].unique()

    df_selected = df.loc[(df['Item'] == evaluation_item) & (df['Reference'] == ref_source)]
    df_selected = df_selected[df_selected['Simulation'].isin(sim_cases)]

    model_names = df_selected['Simulation'].values

    data = df_selected[scores].dropna(axis=1, how='any').values

    # Set figure size
    font = {'family': option['font']}
    matplotlib.rc('font', **font)

    params = {'backend': 'ps',
              'axes.linewidth': option['axes_linewidth'],
              'font.size': 15,
              'xtick.labelsize': option['xtick'],
              'xtick.direction': 'out',
              'ytick.labelsize': option['ytick'],
              'ytick.direction': 'out',
              'savefig.bbox': 'tight',
              'axes.unicode_minus': False,
              'text.usetex': False}
    rcParams.update(params)
    # Set figure size
    figsize = (option['x_wise'], option['y_wise'])

    # Create the parallel coordinate plot

    fig, ax = parallel_coordinate_plot(data, list(scores), model_names,
                                       models_to_highlight=model_names,
                                       models_to_highlight_by_line=option["models_to_highlight_by_line"],
                                       models_to_highlight_markers=['.'],
                                       models_to_highlight_markers_size=option["models_to_highlight_markers_size"],
                                       models_to_highlight_markers_alpha=option['models_to_highlight_markers_alpha'],
                                       debug=False,
                                       figsize=figsize,
                                       colormap='tab20',
                                       xtick_labelsize=option['xtick'],
                                       ytick_labelsize=option['ytick'],
                                       legend_off=option["legend_off"],
                                       legend_ncol=option["legend_ncol"],
                                       legend_bbox_to_anchor=(option["bbox_to_anchor_x"], option["bbox_to_anchor_y"]),
                                       legend_loc=option["legend_loc"],
                                       legend_fontsize=option['fontsize'],
                                       option=option,
                                       )

    ax.set_ylabel(option['yticklabel'], fontsize=option['ytick'] + 1)
    ax.set_xlabel(option['xticklabel'], fontsize=option['xtick'] + 1)
    ax.set_title(option['title'], fontsize=option['title_size'])

    Figure_show.pyplot(fig)

    # # Save the plot
    filename = f'Parallel_Coordinates_Plot_{var}_{evaluation_item}_{ref_source}'
    buffer = BytesIO()
    fig.savefig(buffer, format=option['saving_format'], dpi=option['dpi'])  # f
    buffer.seek(0)
    buffer.seek(0)

    st.download_button('Download image', buffer, file_name=f'{filename}.{option["saving_format"]}',
                       mime=f"image/{option['saving_format']}",
                       type="secondary", disabled=False, use_container_width=False)


def draw_scenarios_comparison_parallel_coordinates_by_score(Figure_show, option, file, score, select_items, sim_cases):
    # Read the data from the file
    df = pd.read_csv(file, sep='\s+', header=0)

    filtered_df = df.groupby("Item")[["Reference"]].agg(lambda x: list(x.unique())).reset_index()

    filtered_df = filtered_df[filtered_df['Item'].isin(select_items)]
    unique_items = filtered_df['Item'].unique()
    sim_sources = df['Simulation'].unique()
    all_combinations = list(itertools.product(*filtered_df['Reference']))

    if 'Parallel_Coordinates_score_next' not in st.session_state:
        st.session_state['Parallel_Coordinates_score_next'] = 0

    if st.session_state.get('Next_item', False):
        st.session_state['Parallel_Coordinates_score_next'] = (st.session_state.get('Parallel_Coordinates_score_next',
                                                                                    0) + 1) % len(all_combinations)
        i = st.session_state['Parallel_Coordinates_score_next']
    else:
        i = 0

    item_combination = all_combinations[i]
    Figure_show.write(f"##### :green[Showing for {', '.join(item_combination)}]")
    mask = pd.Series(False, index=df.index)
    for j, item in enumerate(unique_items):
        mask |= (df['Item'] == item) & (df['Reference'] == item_combination[j])

    filtered_df = df[mask]

    # Set figure size
    font = {'family': option['font']}
    matplotlib.rc('font', **font)

    params = {'backend': 'ps',
              'axes.linewidth': option['axes_linewidth'],
              'font.size': 15,
              'xtick.labelsize': option['xtick'],
              'xtick.direction': 'out',
              'ytick.labelsize': option['ytick'],
              'ytick.direction': 'out',
              'savefig.bbox': 'tight',
              'axes.unicode_minus': False,
              'text.usetex': False}
    rcParams.update(params)
    # Set figure size
    figsize = (option['x_wise'], option['y_wise'])

    # Create an empty list to store the data for each simulation
    data_list = []
    model_names = []
    #
    # Iterate over each simulation
    for sim_source in sim_cases:
        # Select rows where 'Simulation' matches the current simulation
        df_selected = filtered_df.loc[filtered_df['Simulation'] == sim_source]

        # Extract the score values for each item in the current simulation
        score_values = df_selected.set_index('Item')[score].reindex(unique_items).values.reshape(1, -1)
        data_list.append(score_values)
        model_names.append(sim_source)

    # Concatenate the data from all simulations
    data = np.concatenate(data_list, axis=0)

    fig, ax = parallel_coordinate_plot(data, unique_items, model_names,
                                       models_to_highlight=model_names,
                                       models_to_highlight_by_line=option["models_to_highlight_by_line"],
                                       models_to_highlight_markers=True,
                                       models_to_highlight_markers_size=option["models_to_highlight_markers_size"],
                                       models_to_highlight_markers_alpha=option['models_to_highlight_markers_alpha'],
                                       debug=False,
                                       figsize=figsize,
                                       colormap='tab20',
                                       xtick_labelsize=option['xtick'],
                                       ytick_labelsize=option['ytick'],
                                       legend_off=option["legend_off"],
                                       legend_ncol=option["legend_ncol"],
                                       legend_bbox_to_anchor=(option["bbox_to_anchor_x"], option["bbox_to_anchor_y"]),
                                       legend_loc=option["legend_loc"],
                                       legend_fontsize=option["fontsize"],
                                       option=option,
                                       )

    # Set the title of the plot
    option['title'] = option['title'] + f" \n References: {', '.join(all_combinations[i])}"
    ax.set_ylabel(option['yticklabel'], fontsize=option['ytick'] + 1)
    ax.set_xlabel(option['xticklabel'], fontsize=option['xtick'] + 1)
    ax.set_title(option['title'], fontsize=option['title_size'])
    Figure_show.pyplot(fig)

    # # Save the plot
    filename = f"Parallel_Coordinates_Plot_{score}_{'_'.join(item_combination)}"
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
    col3.button(f':point_right: Next Case', key='Next_item', disabled=next_disable, help='Press to change reference items')


def _quick_qc(data, model_names, metric_names, model_names2=None):
    # Quick initial QC
    if data.shape[0] != len(model_names):
        sys.exit(
            "Error: data.shape[0], "
            + str(data.shape[0])
            + ", mismatch to len(model_names), "
            + str(len(model_names))
        )
    if data.shape[1] != len(metric_names):
        sys.exit(
            "Error: data.shape[1], "
            + str(data.shape[1])
            + ", mismatch to len(metric_names), "
            + str(len(metric_names))
        )
    if model_names2 is not None:
        # Check: model_names2 should be a subset of model_names
        for model in model_names2:
            if model not in model_names:
                sys.exit(
                    "Error: model_names2 should be a subset of model_names, but "
                    + model
                    + " is not in model_names"
                )
    # print("Passed a quick QC")


def _data_transform(
        data,
        metric_names,
        model_names,
        model_names2=None,
        group1_name="group1",
        group2_name="group2",
        vertical_center=None,
        ymax=None,
        ymin=None,
):
    # Data to plot
    ys = data  # stacked y-axis values
    N = ys.shape[1]  # number of vertical axis (i.e., =len(metric_names))

    if ymax is None:
        ymaxs = np.nanmax(ys, axis=0)  # maximum (ignore nan value)
    else:
        try:
            if isinstance(ymax, str) and ymax == "percentile":
                ymaxs = np.nanpercentile(ys, 95, axis=0)
            else:
                ymaxs = np.repeat(ymax, N)
        except ValueError:
            print(f"Invalid input for ymax: {ymax}")

    if ymin is None:
        ymins = np.nanmin(ys, axis=0)  # minimum (ignore nan value)
    else:
        try:
            if isinstance(ymin, str) and ymin == "percentile":
                ymins = np.nanpercentile(ys, 5, axis=0)
            else:
                ymins = np.repeat(ymin, N)
        except ValueError:
            print(f"Invalid input for ymin: {ymin}")

    ymeds = np.nanmedian(ys, axis=0)  # median
    ymean = np.nanmean(ys, axis=0)  # mean

    if vertical_center is not None:
        if vertical_center == "median":
            ymids = ymeds
        elif vertical_center == "mean":
            ymids = ymean
        elif isinstance(vertical_center, float) or isinstance(vertical_center, int):
            ymids = np.repeat(vertical_center, N)
        else:
            raise ValueError(f"vertical center {vertical_center} unknown.")

        for i in range(0, N):
            distance_from_middle = max(
                abs(ymaxs[i] - ymids[i]), abs(ymids[i] - ymins[i])
            )
            ymaxs[i] = ymids[i] + distance_from_middle
            ymins[i] = ymids[i] - distance_from_middle

    dys = ymaxs - ymins
    if ymin is None:
        ymins -= dys * 0.05  # add 5% padding below and above
    if ymax is None:
        ymaxs += dys * 0.05
    dys = ymaxs - ymins

    # Handle the case when ymins and ymaxs are the same for a particular axis
    zero_range_indices = np.where(dys == 0)[0]
    if len(zero_range_indices) > 0:
        for idx in zero_range_indices:
            if ymins[idx] == 0:
                ymaxs[idx] = 1
            else:
                ymins[idx] -= np.abs(ymins[idx]) * 0.05
                ymaxs[idx] += np.abs(ymaxs[idx]) * 0.05
        dys = ymaxs - ymins

    # Transform all data to be compatible with the main axis
    zs = np.zeros_like(ys)
    zs[:, 0] = ys[:, 0]
    zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

    if vertical_center is not None:
        zs_middle = (ymids[:] - ymins[:]) / dys[:] * dys[0] + ymins[0]
    else:
        zs_middle = (ymaxs[:] - ymins[:]) / 2 / dys[:] * dys[0] + ymins[0]

    if model_names2 is not None:
        print("Models in the second group:", model_names2)

    # Pandas dataframe for seaborn plotting
    df_stacked = _to_pd_dataframe(
        data,
        metric_names,
        model_names,
        model_names2=model_names2,
        group1_name=group1_name,
        group2_name=group2_name,
    )
    df2_stacked = _to_pd_dataframe(
        zs,
        metric_names,
        model_names,
        model_names2=model_names2,
        group1_name=group1_name,
        group2_name=group2_name,
    )

    return zs, zs_middle, N, ymins, ymaxs, df_stacked, df2_stacked


def _to_pd_dataframe(
        data,
        metric_names,
        model_names,
        model_names2=None,
        group1_name="group1",
        group2_name="group2",
):
    # Pandas dataframe for seaborn plotting
    df = pd.DataFrame(data, columns=metric_names, index=model_names)
    # Stack
    # df_stacked = df.stack(dropna=False).reset_index()
    # df_stacked = df.stack(dropna=False, future_stack=True).reset_index()
    df_stacked = df.stack(future_stack=True).reset_index()
    df_stacked = df_stacked.rename(
        columns={"level_0": "Model", "level_1": "Metric", 0: "value"}
    )
    df_stacked = df_stacked.assign(group=group1_name)
    if model_names2 is not None:
        for model2 in model_names2:
            df_stacked["group"] = np.where(
                (df_stacked.Model == model2), group2_name, df_stacked.group
            )
    return df_stacked


def parallel_coordinate_plot(
        data,
        metric_names,
        model_names,
        models_to_highlight=list(),
        models_to_highlight_by_line=True,
        models_to_highlight_colors=None,
        models_to_highlight_labels=None,
        models_to_highlight_markers=["s", "o", "^", "*", ],
        models_to_highlight_markers_size=22,
        models_to_highlight_markers_alpha=1.0,
        fig=None,
        ax=None,
        figsize=(15, 5),
        show_boxplot=False,
        show_violin=False,
        violin_colors=("lightgrey", "pink"),
        violin_label=None,
        title=None,
        identify_all_models=True,
        xtick_labelsize=None,
        ytick_labelsize=None,
        colormap="viridis",
        num_color=20,
        legend_off=False,
        legend_ncol=6,
        legend_bbox_to_anchor=(0.5, -0.14),
        legend_loc="upper center",
        legend_fontsize=10,
        logo_rect=None,
        logo_off=False,
        model_names2=None,
        group1_name="group1",
        group2_name="group2",
        comparing_models=None,
        fill_between_lines=False,
        fill_between_lines_colors=("red", "green"),
        arrow_between_lines=False,
        arrow_between_lines_colors=("red", "green"),
        arrow_alpha=1,
        arrow_width=0.05,
        arrow_linewidth=0,
        arrow_head_width=0.15,
        arrow_head_length=0.15,
        vertical_center=None,
        vertical_center_line=False,
        vertical_center_line_label=None,
        ymax=None,
        ymin=None,
        debug=False,
        option={},
):
    """
    Parameters
    ----------
    - `data`: 2-d numpy array for metrics
    - `metric_names`: list, names of metrics for individual vertical axes (axis=1)
    - `model_names`: list, name of models for markers/lines (axis=0)
    - `models_to_highlight`: list, default=None, List of models to highlight as lines or marker
    - `models_to_highlight_by_line`: bool, default=True, highlight as lines. If False, as marker
    - `models_to_highlight_colors`: list, default=None, List of colors for models to highlight as lines
    - `models_to_highlight_labels`: list, default=None, List of string labels for models to highlight as lines
    - `models_to_highlight_markers`: list, matplotlib markers for models to highlight if as marker
    - `models_to_highlight_markers_size`: float, size of matplotlib markers for models to highlight if as marker
    - `fig`: `matplotlib.figure` instance to which the parallel coordinate plot is plotted.
             If not provided, use current axes or create a new one.  Optional.
    - `ax`: `matplotlib.axes.Axes` instance to which the parallel coordinate plot is plotted.
             If not provided, use current axes or create a new one.  Optional.
    - `figsize`: tuple (two numbers), default=(15,5), image size
    - `show_boxplot`: bool, default=False, show box and wiskers plot
    - `show_violin`: bool, default=False, show violin plot
    - `violin_colors`: tuple or list containing two strings for colors of violin. Default=("lightgrey", "pink")
    - `violin_label`: string to label the violin plot, when violin plot is not splited. Default is None.
    - `title`: string, default=None, plot title
    - `identify_all_models`: bool, default=True. Show and identify all models using markers
    - `xtick_labelsize`: number, fontsize for x-axis tick labels (optional)
    - `ytick_labelsize`: number, fontsize for x-axis tick labels (optional)
    - `colormap`: string, default='viridis', matplotlib colormap
    - `num_color`: integer, default=20, how many color to use.
    - `legend_off`: bool, default=False, turn off legend
    - `legend_ncol`: integer, default=6, number of columns for legend text
    - `legend_bbox_to_anchor`: tuple, defulat=(0.5, -0.14), set legend box location
    - `legend_loc`: string, default="upper center", set legend box location
    - `legend_fontsize`: float, default=8, legend font size
    - `logo_rect`: sequence of float. The dimensions [left, bottom, width, height] of the new Axes.
                All quantities are in fractions of figure width and height.  Optional.
    - `logo_off`: bool, default=False, turn off PMP logo
    - `model_names2`: list of string, should be a subset of `model_names`.  If given, violin plot will be split into 2 groups. Optional.
    - `group1_name`: string, needed for violin plot legend if splited to two groups, for the 1st group. Default is 'group1'.
    - `group2_name`: string, needed for violin plot legend if splited to two groups, for the 2nd group. Default is 'group2'.
    - `comparing_models`: tuple or list containing two strings for models to compare with colors filled between the two lines.
    - `fill_between_lines`: bool, default=False, fill color between lines for models in comparing_models
    - `fill_between_lines_colors`: tuple or list containing two strings of colors for filled between the two lines. Default=('red', 'green')
    - `arrow_between_lines`: bool, default=False, place arrows between two lines for models in comparing_models
    - `arrow_between_lines_colors`: tuple or list containing two strings of colors for arrow between the two lines. Default=('red', 'green')
    - `arrow_alpha`: float, default=1, transparency of arrow (faction between 0 to 1)
    - `arrow_width`: float, default is 0.05, width of arrow
    - `arrow_linewidth`: float, default is 0, width of arrow edge line
    - `arrow_head_width`: float, default is 0.15, widht of arrow head
    - `arrow_head_length`: float, default is 0.15, length of arrow head
    - `vertical_center`: string ("median", "mean")/float/integer, default=None, adjust range of vertical axis to set center of vertical axis as median, mean, or given number
    - `vertical_center_line`: bool, default=False, show median as line
    - `vertical_center_line_label`: str, default=None, label in legend for the horizontal vertical center line. If not given, it will be automatically assigned. It can be turned off by "off"
    - `ymax`: int or float or string ('percentile'), default=None, specify value of vertical axis top. If percentile, 95th percentile or extended for top
    - `ymin`: int or float or string ('percentile'), default=None, specify value of vertical axis bottom. If percentile, 5th percentile or extended for bottom

    Return
    ------
    - `fig`: matplotlib component for figure
    - `ax`: matplotlib component for axis

    Author: Jiwoo Lee @ LLNL (2021. 7)
    Update history:
    2021-07 Plotting code created. Inspired by https://stackoverflow.com/questions/8230638/parallel-coordinates-plot-in-matplotlib
    2022-09 violin plots added
    2023-03 median centered option added
    2023-04 vertical center option diversified (median, mean, or given number)
    2024-03 parameter added for violin plot label
    2024-04 parameters added for arrow and option added for ymax/ymin setting
    """
    params = {
        "legend.fontsize": "large",
        "axes.labelsize": "x-large",
        "axes.titlesize": "x-large",
        "xtick.labelsize": "x-large",
        "ytick.labelsize": "x-large",
    }
    pylab.rcParams.update(params)

    # Quick initial QC
    _quick_qc(data, model_names, metric_names, model_names2=model_names2)

    # Transform data for plotting
    zs, zs_middle, N, ymins, ymaxs, df_stacked, df2_stacked = _data_transform(
        data,
        metric_names,
        model_names,
        model_names2=model_names2,
        group1_name=group1_name,
        group2_name=group2_name,
        vertical_center=vertical_center,
        ymax=ymax,
        ymin=ymin,
    )

    if debug:
        print("ymins:", ymins)
        print("ymaxs:", ymaxs)

    # Prepare plot
    if N > 20:
        if xtick_labelsize is None:
            xtick_labelsize = "large"
        if ytick_labelsize is None:
            ytick_labelsize = "large"
    else:
        if xtick_labelsize is None:
            xtick_labelsize = "x-large"
        if ytick_labelsize is None:
            ytick_labelsize = "x-large"
    params = {
        "legend.fontsize": "large",
        "axes.labelsize": "x-large",
        "axes.titlesize": "x-large",
        "xtick.labelsize": xtick_labelsize,
        "ytick.labelsize": ytick_labelsize,
    }
    pylab.rcParams.update(params)

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    axes = [ax] + [ax.twinx() for i in range(N - 1)]

    for i, ax_y in enumerate(axes):
        ax_y.set_ylim(ymins[i], ymaxs[i])
        ax_y.spines["top"].set_visible(False)
        ax_y.spines["bottom"].set_visible(False)
        if ax_y == ax:
            ax_y.spines["left"].set_position(("data", i))
        if ax_y != ax:
            ax_y.spines["left"].set_visible(False)
            ax_y.yaxis.set_ticks_position("right")
            ax_y.spines["right"].set_position(("data", i))

    # Population distribuion on each vertical axis
    if show_boxplot or show_violin:
        y = [zs[:, i] for i in range(N)]
        y_filtered = [
            y_i[~np.isnan(y_i)] for y_i in y
        ]  # Remove NaN value for box/violin plot

        # Box plot
        if show_boxplot:
            box = ax.boxplot(
                y_filtered, positions=range(N), patch_artist=True, widths=0.15
            )
            for item in ["boxes", "whiskers", "fliers", "medians", "caps"]:
                plt.setp(box[item], color="darkgrey")
            plt.setp(box["boxes"], facecolor="None")
            plt.setp(box["fliers"], markeredgecolor="darkgrey")

        # Violin plot
        if show_violin:
            if model_names2 is None:
                # matplotlib for regular violin plot
                violin = ax.violinplot(
                    y_filtered,
                    positions=range(N),
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )
                for pc in violin["bodies"]:
                    if isinstance(violin_colors, tuple) or isinstance(
                            violin_colors, list
                    ):
                        violin_color = violin_colors[0]
                    else:
                        violin_color = violin_colors
                    pc.set_facecolor(violin_color)
                    pc.set_edgecolor("None")
                    pc.set_alpha(0.8)
            else:
                # seaborn for split violin plot
                violin = sns.violinplot(
                    data=df2_stacked,
                    x="Metric",
                    y="value",
                    ax=ax,
                    hue="group",
                    split=True,
                    linewidth=0.1,
                    scale="count",
                    scale_hue=False,
                    palette={
                        group1_name: violin_colors[0],
                        group2_name: violin_colors[1],
                    },
                )

    # Line or marker
    if 'colors' in option:
        colors = option['colors']
    else:
        colors = [plt.get_cmap(colormap)(c) for c in np.linspace(0, 1, num_color)]
    if 'markers' in option:
        marker_types = [option['markers']]
        markers = list(flatten([[marker] * len(colors) for marker in marker_types]))
    else:
        marker_types = ["o", "s", "*", "^", "X", "D", "p"]
        markers = list(flatten([[marker] * len(colors) for marker in marker_types]))

    colors *= len(marker_types)
    mh_index = 0
    for j, model in enumerate(model_names):
        # to just draw straight lines between the axes:
        if model in models_to_highlight:
            if models_to_highlight_colors is not None:
                color = models_to_highlight_colors[mh_index]
            else:
                color = colors[j]

            if models_to_highlight_labels is not None:
                label = models_to_highlight_labels[mh_index]
            else:
                label = model

            if models_to_highlight_by_line:
                ax.plot(range(N), zs[j, :], "-", c=color, label=label, lw=3, markersize=models_to_highlight_markers_size, )
            else:
                ax.plot(
                    range(N),
                    zs[j, :],
                    markers[mh_index],
                    c=color,
                    label=label,
                    markersize=models_to_highlight_markers_size,
                    alpha=models_to_highlight_markers_alpha,
                )

            mh_index += 1
        else:
            if identify_all_models:
                ax.plot(
                    range(N),
                    zs[j, :],
                    markers[j],
                    c=colors[j],
                    label=model,
                    clip_on=False,
                    markersize=models_to_highlight_markers_size
                )

    if vertical_center_line:
        if vertical_center_line_label is None:
            vertical_center_line_label = str(vertical_center)
        elif vertical_center_line_label == "off":
            vertical_center_line_label = None
        ax.plot(range(N), zs_middle, "-", c="k", label=vertical_center_line_label, lw=1)

    # Compare two models
    if comparing_models is not None:
        if isinstance(comparing_models, tuple) or (
                isinstance(comparing_models, list) and len(comparing_models) == 2
        ):
            x = range(N)
            m1 = model_names.index(comparing_models[0])
            m2 = model_names.index(comparing_models[1])
            y1 = zs[m1, :]
            y2 = zs[m2, :]

            # Fill between lines
            if fill_between_lines:
                ax.fill_between(
                    x,
                    y1,
                    y2,
                    where=(y2 > y1),
                    facecolor=fill_between_lines_colors[0],
                    interpolate=False,
                    alpha=0.5,
                )
                ax.fill_between(
                    x,
                    y1,
                    y2,
                    where=(y2 < y1),
                    facecolor=fill_between_lines_colors[1],
                    interpolate=False,
                    alpha=0.5,
                )

            # Add vertical arrows
            if arrow_between_lines:
                for xi, yi1, yi2 in zip(x, y1, y2):
                    if yi2 > yi1:
                        arrow_color = arrow_between_lines_colors[0]
                    elif yi2 < yi1:
                        arrow_color = arrow_between_lines_colors[1]
                    else:
                        arrow_color = None
                    arrow_length = yi2 - yi1
                    ax.arrow(
                        xi,
                        yi1,
                        0,
                        arrow_length,
                        color=arrow_color,
                        length_includes_head=True,
                        alpha=arrow_alpha,
                        width=arrow_width,
                        linewidth=arrow_linewidth,
                        head_width=arrow_head_width,
                        head_length=arrow_head_length,
                        zorder=999,
                    )
    if 'x_rotation' in option:
        x_rotation = option["x_rotation"]
    else:
        x_rotation = 0

    if 'x_ha' in option:
        x_ha = option['x_ha']
    else:
        x_ha = 'center'

    ax.set_xlim(-0.5, N - 0.5)
    ax.set_xticks(range(N))
    ax.set_xticklabels([name.replace('_', ' ') for name in metric_names], fontsize=xtick_labelsize, rotation=x_rotation, ha=x_ha)
    ax.tick_params(axis="x", which="major", pad=7)
    ax.spines["right"].set_visible(False)

    if not legend_off:
        if violin_label is not None:
            # Get all lines for legend
            lines = [violin["bodies"][0]] + ax.lines
            # Get labels for legend
            labels = [violin_label] + [line.get_label() for line in ax.lines]
            # Remove unnessasary lines that its name starts with '_' to avoid the burden of warning message
            lines = [aa for aa, bb in zip(lines, labels) if not bb.startswith("_")]
            labels = [bb for bb in labels if not bb.startswith("_")]
            # Add legend
            ax.legend(
                lines,
                labels,
                loc=legend_loc,
                ncol=legend_ncol,
                bbox_to_anchor=legend_bbox_to_anchor,
                fontsize=legend_fontsize,
            )
        else:
            # Add legend
            ax.legend(
                loc=legend_loc,
                ncol=legend_ncol,
                bbox_to_anchor=legend_bbox_to_anchor,
                fontsize=legend_fontsize,
            )

    return fig, ax
