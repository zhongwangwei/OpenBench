import os
import numpy as np
import xarray as xr
import pandas as pd
from joblib import Parallel, delayed
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams
import scipy.stats as sts
from io import BytesIO
import streamlit as st
import itertools


def draw_scenarios_comparison_Single_Model_Performance_Index(Figure_show, option, file, evaluation_items, ref_nml, cases):
    # Read the SMPI data
    font = {'family': option['font']}
    matplotlib.rc('font', **font)

    params = {'backend': 'ps',
              'axes.linewidth': 1,
              'font.size': option['fontsize'],
              'xtick.labelsize': option['xticksize'],
              'xtick.direction': 'out',
              'ytick.direction': 'out',
              'savefig.bbox': 'tight',
              'axes.unicode_minus': False,
              'text.usetex': False}
    rcParams.update(params)
    df = pd.read_csv(file, sep='\t')

    # Prepare the subplot grid
    n_items = len(evaluation_items)

    fig, axs = plt.subplots(n_items, 1, figsize=(option['x_wise'], option['y_wise']), sharey=True, squeeze=False)  # sharey=True,

    fig.subplots_adjust(hspace=option["hspace"], wspace=option["wspace"])

    # Calculate overall min and max I² values for consistent x-axis range
    min_I2 = max(0, df['SMPI'].min() - 0.5)
    max_I2 = min(5, df['SMPI'].max() + 0.5)

    # Create a color map for subplots
    # color_map = plt.cm.get_cmap('tab20')

    for i, item in enumerate(evaluation_items):
        ref_sources = ref_nml['general'][f'{item}_ref_source']
        if isinstance(ref_sources, str):
            ref_sources = [ref_sources]
        ax = axs[i, 0]

        for j, ref_source in enumerate(ref_sources):

            # Filter data for this item and reference source
            item_data = df[(df['Item'] == item) & (df['Reference'] == ref_source)]
            if item_data.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                continue

            I2_values = item_data['SMPI'].tolist()
            labels = item_data['Simulation'].tolist()

            # Calculate confidence intervals
            mean = np.mean(I2_values)
            sem = sts.sem(I2_values)
            conf_interval = sem * sts.t.ppf((1 + 0.95) / 2., len(I2_values) - 1)
            sizes = [150 * conf_interval] * len(I2_values)  # Reduced circle size

            for k, (value, size, label) in enumerate(zip(I2_values, sizes, labels)):
                ax.scatter(value, 0, s=size * option["n"], facecolors=option['COLORS'][label], edgecolors=option['COLORS'][label], label=label,
                           alpha=0.8)
                ax.scatter(value, 0, s=size * 0.01 * option["n"], facecolors='white', edgecolors='none')

            # Annotate labels
            for k, value in enumerate(I2_values):
                ax.annotate(
                    str(k + 1),  # Use numbers starting from 1
                    (value, 0),
                    xytext=(0, 18),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    fontsize=option['fontsize'],
                    rotation=45
                )

            # Mean (black circle)
            ax.scatter(mean, 0, color="black", s=50, marker="o", alpha=0.6)
            ax.scatter(mean, 0, color="white", s=50 * 0.01, marker="o", alpha=0.6)
            # Add mean label
            ax.annotate(
                'Mean',
                (mean, 0),
                xytext=(0, -15),  # Position the label below the mean point
                textcoords='offset points',
                ha='center',
                va='top',
                rotation=-45
            )

            # Set up axes and ticks
            ax.spines["bottom"].set_position("zero")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.xaxis.set_ticks_position("bottom")
            ax.tick_params(axis="x", direction="inout", which="both", length=20, width=1.5)  # , labelsize=8
            ax.tick_params(axis="x", which="minor", length=10)
            ax.set_xlim([min_I2, max_I2])
            ax.set_xticks(np.arange(min_I2, max_I2 + 0.5, 0.5))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(0.25))

            if not option["var_loc"]:
                if j == 0:
                    ax.text(option["x_posi"], option["y_posi"], item.replace('_', ' '), rotation=option["y_rotation"],
                            va='center', ha=option['y_ha'],
                            transform=ax.transAxes, fontsize=option['yticksize'])
        if option["var_loc"]:
            ax.text(option["x_posi"], option["y_posi"], item.replace('_', ' '), rotation=option["x_rotation"], va='center',
                    ha=option['x_ha'], transform=ax.transAxes, fontsize=option['xticksize'])

    if not option["legend_on"]:
        ax.legend(shadow=False, frameon=False, fontsize=option['fontsize'],
                  loc=option["loc"], ncol=option["ncol"])
    else:
        ax.legend(shadow=False, frameon=False, fontsize=option['fontsize'],
                  bbox_to_anchor=(option["bbox_to_anchor_x"], option["bbox_to_anchor_y"]), ncol=option["ncol"])

    # X-axis label
    if not option["var_loc"]:
        fig.supxlabel(option['xlabel'], ha=option['x_ha'], fontsize=option['xticksize'] + 1, rotation=option["x_rotation"])
    else:
        fig.supylabel(option['ylabel'], ha=option['y_ha'], fontsize=option['yticksize'] + 1, rotation=option["y_rotation"])

    Figure_show.pyplot(fig)

    # 将图像保存到 BytesIO 对象
    buffer = BytesIO()
    fig.savefig(buffer, format=option['saving_format'], dpi=option['dpi'])
    buffer.seek(0)
    buffer.seek(0)
    st.download_button('Download image', buffer, file_name=f'SMPI_comparison_plot_comprehensive.{option["saving_format"]}',
                       mime=f"image/{option['saving_format']}",
                       type="secondary", disabled=False, use_container_width=False)


def make_scenarios_comparison_Single_Model_Performance_Index(dir_path, selected_items, ref, item):
    file = dir_path + "/SMPI_comparison.txt"
    option = {}
    Figure_show = st.container()
    Labels_tab, Var_tab, Scale_tab, Save_tab = st.tabs(['Labels', 'Variables', 'Scale', 'Save'])

    with Labels_tab:
        col1, col2, col3, col4 = st.columns((4, 3, 3, 3))
        option['title'] = col1.text_input('Title',
                                          value=f"",
                                          label_visibility="visible",
                                          key=f"{item} title")
        option['title_size'] = col2.number_input("Title label size", min_value=0, value=15, key=f"{item}_title_size")
        option['fontsize'] = col3.number_input("Font size", min_value=0, value=15,
                                               key=f'{item}_fontsize')
        option['tick'] = col4.number_input("Tick size", min_value=0, value=15,
                                           key=f'{item}_ticksize')
        col1, col2, col3 = st.columns((2, 1, 1))
        option["var_loc"] = col1.toggle('Put Variable labels in X axis?', value=False, key=f"{item}_var_loc")
        option['yticksize'] = col2.number_input("Y ticks size", min_value=0, value=17, key=f"{item}_ytick")
        option['xticksize'] = col3.number_input("X ticks size", min_value=0, value=17, key=f"{item}_xtick")
        if not option["var_loc"]:
            option['xlabel'] = st.text_input('X label', value='Single Model Performance Index',
                                             label_visibility="visible",
                                             key=f"{item}_xlabel")
            option['ylabel'] = ''
        else:
            option['ylabel'] = st.text_input('Y labels', value='Single Model Performance Index',
                                             label_visibility="visible",
                                             key=f"{item}_ylabel")
            option['xlabel'] = ''
        st.divider()
        col1, col2, col3, col4 = st.columns((3, 3, 3, 3))
        if not option["var_loc"]:
            option["x_rotation"] = col1.number_input(f"x rotation", min_value=-90, max_value=90, value=0,
                                                     key=f"{item}_x_rotation")
            option['x_ha'] = col2.selectbox('x ha', ['right', 'left', 'center'],
                                            index=2, placeholder="Choose an option", label_visibility="visible",
                                            key=f"{item}_x_ha")
            option["y_rotation"] = col3.number_input(f"Y rotation", min_value=-90, max_value=90, value=0,
                                                     key=f"{item}_y_rotation")
            option['y_ha'] = col4.selectbox('y ha', ['right', 'left', 'center'],
                                            index=0, placeholder="Choose an option", label_visibility="visible",
                                            key=f"{item}_y_ha")
            option["x_posi"] = col1.number_input(f"Y label position of X", value=-0.02, step=0.05,
                                                 key=f"{item}x_posi")
            option["y_posi"] = col2.number_input(f"Y label position of Y", value=0.5, step=0.05,
                                                 key=f"{item}y_posi")
        else:
            option["x_rotation"] = col1.number_input(f"x rotation", min_value=-90, max_value=90, value=0,
                                                     key=f"{item}_x_rotation")
            option['x_ha'] = col2.selectbox('x ha', ['right', 'left', 'center'],
                                            index=2, placeholder="Choose an option", label_visibility="visible",
                                            key=f"{item}_x_ha")
            option["y_rotation"] = col3.number_input(f"Y rotation", min_value=-90, max_value=90, value=90,
                                                     key=f"{item}_y_rotation")
            option['y_ha'] = col4.selectbox('y ha', ['right', 'left', 'center'],
                                            index=0, placeholder="Choose an option", label_visibility="visible",
                                            key=f"{item}_y_ha")
            option["x_posi"] = col1.number_input(f"X label position of X", value=0.5, step=0.05,
                                                 key=f"{item}x_posi")
            option["y_posi"] = col2.number_input(f"X label position of Y", value=-1.1, step=0.05,
                                                 key=f"{item}y_posi")
        option["n"] = col3.number_input(f"marker size multi", value=1., step=0.1,
                                        key=f"{item}_n")

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
            for item in case_item:
                case_item[item] = st.checkbox(item.replace("_", " "), key=f"{item}__Single_Model_Performance_Index",
                                              value=case_item[item])
            if len([item for item, value in case_item.items() if value]) > 0:
                return [item for item, value in case_item.items() if value]
            else:
                st.error('You must choose one item!')

        if isinstance(selected_items, str): selected_items = [selected_items]
        items = get_cases(selected_items, 'Selected Variables')

    with Scale_tab:
        st.write("##### :blue[Marker Color]")
        import matplotlib.colors as mcolors
        import matplotlib.cm as cm
        df = pd.read_csv(file, sep='\t')
        sim_sources = df['Simulation'].unique()
        hex_colors = [cm.get_cmap('Set3')(c) for c in np.linspace(0, 1, len(sim_sources))]
        colors = itertools.cycle([mcolors.rgb2hex(color) for color in hex_colors])
        cols = itertools.cycle(st.columns(4))
        markers = {}
        for sim_source in sim_sources:
            col = next(cols)
            col.write(f":orange[{sim_source}]")
            markers[sim_source] = col.color_picker(f'{sim_source} colors', value=next(colors),
                                                   key=f"{item}_{sim_source}_colors",
                                                   disabled=False,
                                                   label_visibility="collapsed")

        st.divider()
        option["legend_on"] = st.toggle('Turn on to set the location of the legend manually', value=True,
                                        key=f'{item}_legend_on')
        col1, col2, col3, col4 = st.columns(4)
        option["ncol"] = col1.number_input("N cols", value=1, min_value=1, format='%d', key=f'{item}_ncol')
        if not option["legend_on"]:
            option["loc"] = col2.selectbox("Legend location",
                                           ['best', 'right', 'left', 'upper left', 'upper right', 'lower left',
                                            'lower right',
                                            'upper center',
                                            'lower center', 'center left', 'center right'], index=0,
                                           placeholder="Choose an option",
                                           label_visibility="visible", key=f'{item}_loc')
        else:
            option["bbox_to_anchor_x"] = col3.number_input("X position of legend", value=1.4, key=f'{item}_bbox_to_anchor_x', step=0.1)
            option["bbox_to_anchor_y"] = col4.number_input("Y position of legend", value=3.5, key=f'{item}_bbox_to_anchor_y', step=0.1)

        col1, col2, col3 = st.columns((3, 3, 2))
        option["hspace"] = col1.number_input(f"hspace", step=0.1, value=0.0)
        option["wspace"] = col2.number_input(f"wspace", min_value=0., max_value=1.0, value=0.1)

        option['COLORS'] = markers

    with Save_tab:
        col1, col2, col3 = st.columns(3)
        with col1:
            option["x_wise"] = st.number_input(f"X Length", min_value=0, value=10, key=f"{item}_x_wise")
            option['saving_format'] = st.selectbox('Image saving format', ['png', 'jpg', 'eps'],
                                                   index=1, placeholder="Choose an option", label_visibility="visible",
                                                   key=f"{item}_saving_format")
        with col2:
            if not option["var_loc"]:
                y_wide = len(items)
            else:
                y_wide = len(items) * 2
            option["y_wise"] = st.number_input(f"y Length", min_value=0, value=y_wide, key=f"{item}_y_wise")
            option['font'] = st.selectbox('Image saving format',
                                          ['Times new roman', 'Arial', 'Courier New', 'Comic Sans MS', 'Verdana',
                                           'Helvetica',
                                           'Georgia', 'Tahoma', 'Trebuchet MS', 'Lucida Grande'],
                                          index=0, placeholder="Choose an option", label_visibility="visible",
                                          key=f"{item}_font")
        with col3:
            option['dpi'] = st.number_input(f"Figure dpi", min_value=0, value=300, key=f"{item}_dpi'")

    st.divider()
    if items:
        draw_scenarios_comparison_Single_Model_Performance_Index(Figure_show, option, file, items, ref, sim_sources)
