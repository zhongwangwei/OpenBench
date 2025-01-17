import os
import itertools
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
import matplotlib
from io import BytesIO
import streamlit as st



def ref_lines(option: dict, showing_items, selected_item):
    font = {'family': option['font']}
    matplotlib.rc('font', **font)
    params = {'backend': 'ps',
              'axes.labelsize': option['fontsize'],
              'grid.linewidth': 0.2,
              'font.size': option['fontsize'],
              'legend.fontsize': option['fontsize'],
              'legend.frameon': False,
              'xtick.labelsize': option['xticksize'],
              'xtick.direction': 'out',
              'ytick.labelsize': option['yticksize'],
              'ytick.direction': 'out',
              'savefig.bbox': 'tight',
              'axes.unicode_minus': False,
              'text.usetex': False}
    rcParams.update(params)
    ref_var, sim_var = option['vars']
    ref_unit, sim_unit = option['units']
    try:
        fig, ax = plt.subplots(1, figsize=(option['x_wise'], option['y_wise']))
        for i in range(len(showing_items["ID"])):
            id = showing_items["ID"].values[i]
            use_syear = showing_items["Start year"].values[i]
            use_eyear = showing_items["End year"].values[i]
            try:
                filename = option['data_path'] + f'{selected_item}_ref_{id}_{use_syear}_{use_eyear}.nc'
                ref_data = xr.open_dataset(filename)[ref_var]

                if 'lat' in ref_data.dims and 'lon' in ref_data.dims:
                    ref_data = ref_data.squeeze('lat').squeeze('lon')
                # if option["resample_option"]:
                #     compare_tim_res_map = {'month': '1M', 'day': '1D', 'year': '1Y'}
                #     ref_data = ref_data.resample(time=compare_tim_res_map[option["resample_option"].lower()]).mean()
                # xtime = 'time'
                # if option["groubly_option"]:
                #     ref_data = ref_data.groupby(f'time.{option["groubly_option"].lower()}').mean()
                #     xtime = option["groubly_option"].lower()

                ref_data.plot.line(x='time', linewidth=option[f"{id}"]['linewidth'], linestyle=option[id]['linestyle'],
                                   marker=option[id]['marker'],
                                   markersize=option[id]['markersize'], alpha=0.9,
                                   label=id, color=option[id]['color'], ax=ax)
            except FileNotFoundError:
                st.error(f"File {filename} not found. Please check the file path.")
            except KeyError as e:
                st.error(f"Key error: {e}. Please check the keys in the option dictionary.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        ax.legend(shadow=False, frameon=False, fontsize=option['fontsize'], loc='best')
        ax.set_title(option['title'], fontsize=option['title_size'])
        ax.set_xlabel("Time", fontsize=option['xticksize'] + 1)
        ax.set_ylabel(f"{selected_item} [{ref_unit}]", fontsize=option['yticksize'] + 1)
        if option['grid']:
            ax.grid(linestyle=option['grid_style'], alpha=0.7, linewidth=1.5)  # 绘制图中虚线 透明度0.3
        st.pyplot(fig)

        # 将图像保存到 BytesIO 对象
        buffer = BytesIO()
        fig.savefig(buffer, format=option['saving_format'], dpi=300)
        buffer.seek(0)
        st.download_button('Download image', buffer, file_name=f'Reference.{option["saving_format"]}',
                           mime=f"image/{option['saving_format']}",
                           type="secondary", disabled=False, use_container_width=False)
    except:
        st.error('Please make sure your files were generated correctly.')

def sim_lines(option: dict, showing_items, selected_item):
    font = {'family': option['font']}
    matplotlib.rc('font', **font)
    params = {'backend': 'ps',
              'axes.labelsize': option['fontsize'],
              'grid.linewidth': 0.2,
              'font.size': option['fontsize'],
              'legend.fontsize': option['fontsize'],
              'legend.frameon': False,
              'xtick.labelsize': option['xticksize'],
              'xtick.direction': 'out',
              'ytick.labelsize': option['yticksize'],
              'ytick.direction': 'out',
              'savefig.bbox': 'tight',
              'axes.unicode_minus': False,
              'text.usetex': False}
    rcParams.update(params)
    ref_var, sim_var = option['vars']
    ref_unit, sim_unit = option['units']
    fig, ax = plt.subplots(1, figsize=(option['x_wise'], option['y_wise']))
    try:
        for i in range(len(showing_items["ID"])):
            id = showing_items["ID"].values[i]
            use_syear = showing_items["Start year"].values[i]
            use_eyear = showing_items["End year"].values[i]
            try:
                sim_data = xr.open_dataset(option['data_path'] + f'{selected_item}_sim_{id}_{use_syear}_{use_eyear}.nc')[sim_var]

                if 'lat' in sim_data.dims and 'lon' in sim_data.dims:
                    sim_data = sim_data.squeeze('lat').squeeze('lon')
                sim_data.plot.line(x='time', linewidth=option[f"{id}"]['linewidth'], linestyle=option[id]['linestyle'], marker=option[id]['marker'],
                                   markersize=option[id]['markersize'], alpha=0.9,
                                   label=id, color=option[id]['color'], ax=ax)
            except FileNotFoundError:
                st.error(
                    f"File {option['data_path'] + f'{selected_item}_sim_{id}_{use_syear}_{use_eyear}.nc'} not found. Please check the file path.")

        ax.legend(shadow=False, frameon=False, fontsize=option['fontsize'], loc='best')
        ax.set_title(option['title'], fontsize=option['title_size'])
        ax.set_xlabel("Time", fontsize=option['xticksize'] + 1)
        ax.set_ylabel(f"{selected_item} [{sim_unit}]", fontsize=option['yticksize'] + 1)
        if option['grid']:
            ax.grid(linestyle=option['grid_style'], alpha=0.7, linewidth=1.5)  # 绘制图中虚线 透明度0.3
        st.pyplot(fig)

        # 将图像保存到 BytesIO 对象
        buffer = BytesIO()
        fig.savefig(buffer, format=option['saving_format'], dpi=300)
        buffer.seek(0)
        st.download_button('Download image', buffer, file_name=f'Simulation.{option["saving_format"]}',
                           mime=f"image/{option['saving_format']}",
                           type="secondary", disabled=False, use_container_width=False)

    except:
        st.error('Please make sure your files were generated correctly.')

def each_line(option: dict, showing_items, selected_item):
    font = {'family': option['font']}
    matplotlib.rc('font', **font)
    params = {'backend': 'ps',
              'axes.labelsize': option['fontsize'],
              'grid.linewidth': 0.2,
              'font.size': option['fontsize'],
              'legend.fontsize': option['fontsize'],
              'legend.frameon': False,
              'xtick.labelsize': option['xticksize'],
              'xtick.direction': 'out',
              'ytick.labelsize': option['yticksize'],
              'ytick.direction': 'out',
              'savefig.bbox': 'tight',
              'axes.unicode_minus': False,
              'text.usetex': False}
    rcParams.update(params)
    ref_var, sim_var = option['vars']
    ref_unit, sim_unit = option['units']

    def switch_button_index(select):
        if select is None:
            return 0
        my_list = showing_items["ID"]
        index = my_list.index(select)
        return index

    if 'next_option' not in st.session_state:
        st.session_state['next_option'] = 0
    if st.session_state.get('Next_Site', False):
        st.session_state['next_option'] = (st.session_state.get('next_option', 0) + 1) % len(showing_items["ID"])
        i = st.session_state['next_option']
    else:
        i = 0

    id = showing_items["ID"].values[i]
    fig, ax = plt.subplots(1, figsize=(option['x_wise'], option['y_wise']))
    use_syear = showing_items["Start year"].values[i]
    use_eyear = showing_items["End year"].values[i]


    try:
        ref_data = xr.open_dataset(option['data_path'] + f'{selected_item}_ref_{id}_{use_syear}_{use_eyear}.nc')[ref_var]
        if 'lat' in ref_data.dims and 'lon' in ref_data.dims:
            ref_data = ref_data.squeeze('lat').squeeze('lon')
        ref_data.plot.line(x='time', linewidth=option['site_ref']['lineWidth'], linestyle=option['site_ref']['linestyle'], marker=option['site_ref']['marker'],
                           markersize=option['site_ref']['markersize'], alpha=0.9,
                           label=id + ' Reference', color=option['site_ref']['color'], ax=ax)
    except FileNotFoundError:
        st.error(
            f"File {option['data_path'] + f'{selected_item}_ref_{id}_{use_syear}_{use_eyear}.nc'} not found. Please check the file path.")
    except KeyError as e:
        st.error(f"Key error: {e}. Please check the keys in the option dictionary.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    try:
        sim_data = xr.open_dataset(option['data_path'] + f'{selected_item}_sim_{id}_{use_syear}_{use_eyear}.nc')[sim_var]

        if 'lat' in sim_data.dims and 'lon' in sim_data.dims:
            sim_data = sim_data.squeeze('lat').squeeze('lon')
        sim_data.plot.line(x='time', linewidth=option['site_sim']['lineWidth'], linestyle=option['site_sim']['linestyle'], marker=option['site_sim']['marker'],
                           markersize=option['site_sim']['markersize'], alpha=0.9,
                           label=id + ' Simulation', color=option['site_sim']['color'], ax=ax)
    except FileNotFoundError:
        st.error(
            f"File {option['data_path'] + f'{selected_item}_sim_{id}_{use_syear}_{use_eyear}.nc'} not found. Please check the file path.")
    except KeyError as e:
        st.error(f"Key error: {e}. Please check the keys in the option dictionary.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    overall_score_label = f'Overall Scores: {showing_items["Overall_Score"].values[i]:.3f}'
    ax.scatter([], [], color='black', marker='o', label=overall_score_label)

    ax.legend(shadow=False, frameon=False, fontsize=option['fontsize'], loc='best')

    if len(option['title']) == 0:
        option['title'] = f'Lat: {showing_items["Latitude"].values[i]},    Lon:{showing_items["Longitude"].values[i]}'
    ax.set_title(option['title'], fontsize=option['title_size'])
    ax.set_xlabel("Time", fontsize=option['xticksize'] + 1)
    ax.set_ylabel(f"{selected_item.replace('_', ' ')} [{sim_unit}]",
                  fontsize=option['yticksize'] + 1)
    if option['grid']:
        ax.grid(linestyle=option['grid_style'], alpha=0.7, linewidth=1.5)  # 绘制图中虚线 透明度0.3
    st.pyplot(fig)

    buffer = BytesIO()
    fig.savefig(buffer, format=option['saving_format'], dpi=300)
    buffer.seek(0)
    col1, col2 = st.columns((2, .5))
    with col1:
        st.download_button('Download image', buffer, file_name=id + f'.{option["saving_format"]}',
                           mime=f"image/{option['saving_format']}",
                           type="secondary", disabled=False, use_container_width=False)
    with col2:
        st.button(f':point_right: Next Site', key='Next_Site')


def make_stn_lines(showing_items, selected_item, refselect, simselect, path, vars, unit):
    option = {}
    option['data_path'] = path + f'/data/stn_{refselect}_{simselect}/'
    option['vars'] = vars
    option['units'] = unit
    with st.container(border=True):
        st.write('##### :orange[Replot Type]')
        col1, col2 = st.columns((2, 1.5))
        option['plot_type'] = col1.radio('Please choose your type', ['sim lines', 'ref lines', 'each site'],
                                         index=None, label_visibility="collapsed", horizontal=True)
        st.divider()
        if option['plot_type'] == 'sim lines':
            title = 'Simulation'
            ylabel = f"{selected_item} [{unit[1]}]"
        elif option['plot_type'] == 'ref lines':
            title = 'Reference'
            ylabel = f"{selected_item} [{unit[0]}]"
        else:
            title = ''
            ylabel = ''


        col1, col2, col3 = st.columns(3)
        with col1:

            option['title'] = st.text_input('Title',value=title, label_visibility="visible")
            option['title_size'] = st.number_input("Title label size", min_value=0, value=20)
            option['fontsize'] = st.number_input("Font size", min_value=0, value=17)

        with col2:
            option['xlabel'] = st.text_input('X label',value='Time', label_visibility="visible")
            option['xticksize'] = st.number_input("xtick size", min_value=0, value=17)


        with col3:
            option['ylabel'] = st.text_input('Y label',value=ylabel, label_visibility="visible")
            option['yticksize'] = st.number_input("ytick size", min_value=0, value=17)

        col1, col2 = st.columns((2, 1.5))
        option['grid'] = col1.toggle("Showing grid?", value=False, label_visibility="visible")
        if option['grid']:
            option['grid_style'] = col2.selectbox('Grid Line Style', ['solid', 'dotted', 'dashed', 'dashdot'],
                                                  index=2, placeholder="Choose an option", label_visibility="visible")


        import matplotlib.colors as mcolors
        from matplotlib import cm

        hex_colors = ['#4C6EF5', '#F9C74F', '#90BE6D', '#5BC0EB', '#43AA8B', '#F3722C', '#855456', '#F9AFAF',
                      '#F8961E', '#277DA1', '#5A189A']
        colors = itertools.cycle([mcolors.rgb2hex(color) for color in hex_colors])
        if option['plot_type']:
            with st.expander("Edited Lines", expanded=False):
                if option['plot_type'] != 'each site':
                    col1, col2, col3, col4, col5 = st.columns((1.5, 2, 2, 2, 2))
                    col1.write('##### :blue[Colors]')
                    col2.write('##### :blue[LineWidth]')
                    col3.write('##### :blue[Line Style]')
                    col4.write('##### :blue[Marker]')
                    col5.write('##### :blue[Markersize]')

                    for i in range(len(showing_items["ID"])):
                        id = showing_items["ID"].values[i]
                        st.write(id)
                        col1, col2, col3, col4, col5 = st.columns((1.2, 2, 2, 2, 2))
                        option[id] = {}
                        color = next(colors)
                        with col1:
                            option[f"{id}"]['color'] = st.color_picker(f'{id} colors', value=color, key=None, help=None,
                                                                       on_change=None,
                                                                       args=None, kwargs=None, disabled=False,
                                                                       label_visibility="collapsed")
                        with col2:
                            option[f"{id}"]['linewidth'] = st.number_input(f"{id} LineWidth", min_value=0., value=2.,
                                                                           step=0.1,
                                                                           label_visibility="collapsed")
                        with col3:
                            option[f"{id}"]['linestyle'] = st.selectbox(f'{id} Line Style',
                                                                        ['solid', 'dotted', 'dashed', 'dashdot'],
                                                                        index=None, placeholder="Choose an option",
                                                                        label_visibility="collapsed")
                        with col4:
                            option[f"{id}"]['marker'] = st.selectbox(f'{id} Line Marker',
                                                                     ['.', 'x', 'o', '<', '8', 's', 'p', '*', 'h', 'H', 'D',
                                                                      'd',
                                                                      'P',
                                                                      'X'],
                                                                     index=None, placeholder="Choose an option",
                                                                     label_visibility="collapsed")
                        with col5:
                            option[f"{id}"]['markersize'] = st.number_input(f"{id} Markersize", min_value=0, value=10,
                                                                            label_visibility="collapsed")
                elif option['plot_type'] == 'each site':
                    col1, col2, col3, col4, col5 = st.columns((1.4, 2, 2, 2, 2))
                    col1.write('##### :blue[Colors]')
                    col2.write('##### :blue[LineWidth]')
                    col3.write('##### :blue[Linestyle]')
                    col4.write('##### :blue[Marker]')
                    col5.write('##### :blue[Markersize]')

                    ids = ['site_ref', 'site_sim']
                    titles = ['Reference', 'Simulation']
                    colors = ['#F96969', '#599AD4']
                    for id, title, color in zip(ids, titles, colors):
                        option[id] = {}
                        st.write(f'###### :green[{title} Line]')
                        col1, col2, col3, col4, col5 = st.columns((1.2, 2, 2, 2, 2))
                        with col1:
                            option[f"{id}"]['color'] = st.color_picker(f'{id} ref colors', value=color, key=None, help=None,
                                                                       on_change=None,
                                                                       args=None, kwargs=None, disabled=False,
                                                                       label_visibility="collapsed")
                        with col2:
                            option[f"{id}"]['lineWidth'] = st.number_input(f"{id} lineWidth", min_value=0., value=2.,
                                                                           step=0.1,
                                                                           label_visibility="collapsed")
                        with col3:
                            option[f"{id}"]['linestyle'] = st.selectbox(f'{id} Line Style',
                                                                        ['solid', 'dotted', 'dashed', 'dashdot'],
                                                                        index=None, placeholder="Choose an option",
                                                                        label_visibility="collapsed")
                        with col4:
                            option[f"{id}"]['marker'] = st.selectbox(f'{id} Line Marker',
                                                                     ['.', 'x', 'o', '<', '8', 's', 'p', '*', 'h', 'H', 'D',
                                                                      'd', 'P',
                                                                      'X'],
                                                                     index=None, placeholder="Choose an option",
                                                                     label_visibility="collapsed")
                        with col5:
                            option[f"{id}"]['markersize'] = st.number_input(f"{id} Markersize", min_value=0, value=10,
                                                                            label_visibility="collapsed")

        col1, col2, col3 = st.columns(3)
        with col1:
            option["x_wise"] = st.number_input(f"X Length", min_value=0, value=17)
            option['font'] = st.selectbox('Image saving format',
                                            ['Times new roman', 'Arial', 'Courier New', 'Comic Sans MS', 'Verdana',
                                             'Helvetica',
                                             'Georgia', 'Tahoma', 'Trebuchet MS', 'Lucida Grande'],
                                            index=0, placeholder="Choose an option", label_visibility="visible",
                                            key=f"font")
        with col2:
            option["y_wise"] = st.number_input(f"y Length", min_value=0, value=6)
        with col3:
            option['saving_format'] = st.selectbox('Image saving format', ['png', 'jpg', 'eps'],
                                                   index=1, placeholder="Choose an option", label_visibility="visible")


    if option['plot_type'] == 'each site':
        each_line(option, showing_items, selected_item)
    elif option['plot_type'] == 'ref lines':
        ref_lines(option, showing_items, selected_item)
    elif option['plot_type'] == 'sim lines':
        sim_lines(option, showing_items, selected_item)
    else:
        st.warning('Please choose first!')


def geo_Compare_lines(option: dict, selected_item, ref, sim):
    font = {'family': option['font']}
    matplotlib.rc('font', **font)
    params = {'backend': 'ps',
              'axes.labelsize': option['labelsize'],
              'axes.linewidth': option['axes_linewidth'],
              'grid.linewidth': option['grid_linewidth'],
              'font.size': option['labelsize'],
              'legend.fontsize': option['labelsize'],
              'legend.frameon': False,
              'xtick.labelsize': option['xticksize'],
              'xtick.direction': 'out',
              'ytick.labelsize': option['yticksize'],
              'ytick.direction': 'out',
              'savefig.bbox': 'tight',
              'axes.unicode_minus': False,
              'text.usetex': False}
    rcParams.update(params)

    if option['plot_type'] == 'ref lines':
        namelist = ref['general'][f'{selected_item}_ref_source']
        diffs = ['ref'] * len(namelist)
        varlist = []
        for item in namelist:
            varlist.append(ref[selected_item][f'{item}_varname'])

    elif option['plot_type'] == 'sim lines':
        namelist = sim['general'][f'{selected_item}_sim_source']
        diffs = ['sim'] * len(namelist)
        varlist = []
        for item in namelist:
            varlist.append(sim[selected_item][f'{item}_varname'])

    else:
        namelist = ref['general'][f'{selected_item}_ref_source'] + sim['general'][f'{selected_item}_sim_source']
        diffs = ['ref'] * len(ref['general'][f'{selected_item}_ref_source']) + ['sim'] * len(
            sim['general'][f'{selected_item}_sim_source'])
        varlist = []
        for item in ref['general'][f'{selected_item}_ref_source']:
            varlist.append(ref[selected_item][f'{item}_varname'])
        for item in sim['general'][f'{selected_item}_sim_source']:
            varlist.append(sim[selected_item][f'{item}_varname'])

    fig, ax = plt.subplots(1, figsize=(option['x_wise'], option['y_wise']))
    try:
        for var, name, diff in zip(varlist, namelist, diffs):
            filename = f'{option["data_path"]}/{selected_item}_{diff}_{name}_{var}.nc'
            try:
                ds = xr.open_dataset(filename)[var]
                ds = ds.groupby(f'time.{option["showing_option"].lower()}').mean(...)
                ds.plot.line(x=f'{option["showing_option"].lower()}', linewidth=option['linewidth'],
                             linestyle=option[name]['linestyle'], marker=option[name]['marker'],
                             markersize=option[name]['markersize'], alpha=0.9, label=f"{name}",
                             color=option[name]['color'], ax=ax
                             )
            except FileNotFoundError:
                st.error(f"File {filename} not found. Please check the file path.")
            except KeyError as e:
                st.error(f"Key error: {e}. Please check the keys in the option dictionary.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        # if option["showing_option"].lower() == 'month':
        #     ax.set_xticks(ds.month, style='italic')
        if option["showing_option"].lower() == 'month':
            ax.set_xticks(ds.month, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                          style='italic')
        elif option["showing_option"].lower() == 'day':
            ax.set_xticks(ds.day, ['28 days', '29 days', '30 days', '31 days'], style='italic')

        if not option["legend_on"]:
            ax.legend(shadow=False, frameon=False, fontsize=option['labelsize'], loc=option["loc"], ncol=option["ncol"])
        else:
            ax.legend(shadow=False, frameon=False, fontsize=option['labelsize'],
                      bbox_to_anchor=(option["bbox_to_anchor_x"], option["bbox_to_anchor_y"]), ncol=option["ncol"])

        # matplotlib.legend.Legend(parent, handles, labels, *, loc=None, numpoints=None, markerscale=None, markerfirst=True,
        # reverse=False, scatterpoints=None, scatteryoffsets=None, prop=None, fontsize=None, labelcolor=None, borderpad=None,
        # labelspacing=None, handlelength=None, handleheight=None, handletextpad=None, borderaxespad=None, columnspacing=None,
        # ncols=1, mode=None, fancybox=None, shadow=None, title=None, title_fontsize=None, framealpha=None, edgecolor=None,
        # facecolor=None, bbox_to_anchor=None, bbox_transform=None, frameon=None, handler_map=None, title_fontproperties=None,
        # alignment='center', ncol=1, draggable=False)

        ax.set_title(option['title'], fontsize=option['title_size'])
        ax.set_xlabel(option['xticklabel'], fontsize=option['xticksize'] + 1)
        if len(option['yticklabel']) > 0:
            ax.set_ylabel(option['yticklabel'], fontsize=option['yticksize'] + 1)
        if option['grid']:
            ax.grid(linestyle=option['grid_style'], alpha=0.7, linewidth=1.5)  # 绘制图中虚线 透明度0.3
        st.pyplot(fig)

        buffer = BytesIO()
        fig.savefig(buffer, format=option['saving_format'], dpi=300)
        buffer.seek(0)
        st.download_button('Download image', buffer, file_name=f'Reference.{option["saving_format"]}',
                           mime=f"image/{option['saving_format']}",
                           type="secondary", disabled=False, use_container_width=False)
    except:
        st.error('Please make sure your files were generated correctly.')
