# -*- coding: utf-8 -*-
import os
import glob
import math
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pylab import rcParams
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
from io import BytesIO
import itertools
from itertools import chain
from Namelist_lib.namelist_read import NamelistReader
from Muti_function_lib import *
from Comparison_figlib import *

font = {'family': 'Times new roman'}
matplotlib.rc('font', **font)


def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = (time_end - time_start) / 60.
        print('%s cost time: %.3f min' % (func.__name__, time_spend))
        return result

    return func_wrapper


class visualization_validation:
    def __init__(self):
        self.author = "Qingchen Xu/xuqingchen0@gmail.com"
        self.coauthor = "Zhongwang Wei/@gmail.com"

        # self.classification = initial.classification()

        # ------------------------
        self.ref = st.session_state.ref_data
        self.sim = st.session_state.sim_data
        # ----------------------------
        self.generals = st.session_state.generals
        self.evaluation_items = st.session_state.evaluation_items
        self.metrics = st.session_state.metrics
        self.scores = st.session_state.scores
        self.comparisons = st.session_state.comparisons
        self.statistics = st.session_state.statistics

        self.selected_items = st.session_state.selected_items
        self.tittles = st.session_state.tittles

    def set_errors(self):
        # st.json(st.session_state, expanded=False)
        e = RuntimeError('This is an exception of type RuntimeError.'
                         'No data was found for visualization under {st.session_state.,'
                         'check that the path is correct or run the validation first.')
        st.exception(e)
        # '在该路径下并未找到用于可视化的数据，请检查路径是否正确或先运行验证'

    def visualizations(self):
        def on_change(key):
            selection = st.session_state[key]

        visual_select = option_menu(None, ["Metrics", "Scores", "Comparisons"],  # , 'Statistics'
                                    icons=['list-task', 'easel', "list-task", 'easel'],
                                    on_change=on_change, key='visual_forshow', orientation="horizontal")

        case_path = os.path.join(self.generals['basedir'], self.generals['basename'], "output")

        showing_item = []
        if visual_select == "Metrics":
            if self.generals['evaluation']:
                showing_item = [k for k, v in self.metrics.items() if v] + ['PFT_groupby', 'IGBP_groupby']
                if not showing_item:
                    st.info('No metrics selected!')
                else:
                    st.subheader("Select Metrics item to show", divider=True)
            else:
                st.info('No Metrics selected!')

        elif visual_select == "Scores":
            if self.generals['evaluation']:
                showing_item = [k for k, v in self.scores.items() if v] + ['PFT_groupby', 'IGBP_groupby']
                if not showing_item:
                    st.info('No Scores selected!')
                else:
                    st.subheader("Select Score item to show", divider=True)
            else:
                st.info('No Scores selected!')

        elif visual_select == "Comparisons":
            if self.generals['comparison']:
                showing_item = [k for k, v in self.comparisons.items() if v]
                if not showing_item:
                    st.info('No Comparisons selected!')
                else:
                    st.subheader("Select comparison item to show", divider=True)
            else:
                st.info('No Comparisons selected!')

        elif visual_select == "Statistics":
            if self.generals['statistics']:
                showing_item = [k for k, v in self.statistics.items() if v]
                if not showing_item:
                    st.info('No statistics selected!')
            else:
                st.info('No statistics selected!')

        if showing_item:
            item = st.radio('showing_item', [k.replace("_", " ") for k in showing_item], index=None, horizontal=True,
                            label_visibility='collapsed')
            if item:
                self.__step5_make_show_tab(case_path, visual_select, item.replace(" ", "_"))

        # for itab, item in zip(st.tabs([k.replace("_", " ") for k in showing_item]), showing_item):
        #     with itab:
        #         self.__step5_make_show_tab(case_path, visual_select, item)

    def __step5_make_show_tab(self, case_path, visual_select, item):
        @st.cache_data
        def load_image(path):
            image = Image.open(path)
            return image

        if (visual_select == "Metrics") | (visual_select == "Scores"):
            st.cache_data.clear()
            st.divider()
            selected_item = st.radio('selected_items', [k.replace("_", " ") for k in self.selected_items], index=None,
                                     horizontal=True,
                                     key=f'{item}_item')
            st.divider()

            if selected_item:
                selected_item = selected_item.replace(" ", "_")
                sim_sources = self.sim['general'][f'{selected_item}_sim_source']
                ref_sources = self.ref['general'][f'{selected_item}_ref_source']
                if isinstance(sim_sources, str): sim_sources = [sim_sources]
                if isinstance(ref_sources, str): ref_sources = [ref_sources]
                for ref_source in ref_sources:
                    for sim_source in sim_sources:
                        if (self.ref[ref_source]['general'][f'data_type'] != 'stn') & (
                                self.sim[sim_source]['general'][f'data_type'] != 'stn'):
                            if item not in ['PFT_groupby', 'IGBP_groupby']:
                                filenames = os.path.join(case_path, visual_select.lower(),
                                                         f'{selected_item}_ref_{ref_source}_sim_{sim_source}_{item}*')
                            else:
                                filenames = os.path.join(case_path, visual_select.lower(), item, f'{sim_source}___{ref_source}',
                                                         f'{selected_item}_{sim_source}___{ref_source}_{visual_select.lower()}_heatmap*')
                            filename = glob.glob(filenames)
                            filtered_list = [f for f in filename if not f.endswith('.nc')]
                            try:
                                image = load_image(filtered_list[0])
                                st.image(image, caption=f'Reference: {ref_source}, Simulation: {sim_source}',
                                         use_container_width=True)
                            except:
                                st.error(f'Missing Figure for Reference: {ref_source}, Simulation: {sim_source}', icon="⚠")
                        else:
                            if item not in ['PFT_groupby', 'IGBP_groupby']:
                                filenames = os.path.join(case_path, visual_select.lower(),
                                                         f'{selected_item}_stn_{ref_source}_{sim_source}_{item}*')
                                filename = glob.glob(filenames)
                                filtered_list = [f for f in filename if not f.endswith('.csv')]
                                try:
                                    image = load_image(filtered_list[0])
                                    st.image(image, caption=f'Reference: {ref_source}, Simulation: {sim_source}',
                                             use_container_width=True)
                                except:
                                    st.error(f'Missing Figure for Reference: {ref_source}, Simulation: {sim_source}', icon="⚠")
                            else:
                                st.info(
                                    f'Reference: {ref_source}, Simulation: {sim_source}---Heatmap groupby is not supported for station data!',
                                    icon="👋")

        elif visual_select == "Comparisons":
            st.cache_data.clear()
            st.divider()
            figure_path = str(os.path.join(case_path, visual_select.lower(), item))

            if item == "HeatMap":
                st.write('#### :blue[Select Scores!]')
                iscore = st.radio("HeatMap", [k.replace("_", " ") for k, v in self.scores.items() if v],
                                  index=None, horizontal=True, key=f'{item}', label_visibility='collapsed')
                st.divider()
                if iscore:
                    score = iscore.replace(' ', '_')
                    filename = glob.glob(os.path.join(figure_path, f'scenarios_{score}_comparison_heatmap.*'))[0]
                    if os.path.exists(filename):
                        image = load_image(filename)
                        st.image(image, caption=f'Scores: {iscore}', use_container_width=True)
                    else:
                        st.error(f'Missing Figure for Scores: {iscore}', icon="⚠")

            elif (item == "Taylor_Diagram") | (item == "Target_Diagram"):
                st.write('##### :blue[Select Variables]')
                selected_item = st.radio(item,
                                         [i.replace("_", " ") for i in self.selected_items], index=None,
                                         horizontal=True, key=f'{item}', label_visibility="collapsed")
                st.divider()
                if selected_item:
                    selected_item = selected_item.replace(" ", "_")
                    ref_sources = self.ref['general'][f'{selected_item}_ref_source']
                    if isinstance(ref_sources, str): ref_sources = [ref_sources]
                    for ref_source in ref_sources:
                        filename = glob.glob(os.path.join(figure_path, f'{item}_{selected_item}_{ref_source}.*'))
                        filename = [f for f in filename if not f.endswith('.txt')][0]
                        if os.path.exists(filename):
                            image = load_image(filename)
                            st.image(image, caption=f'Reference: {ref_source}', use_container_width=True)
                        else:
                            st.error(f'Missing Figure for {selected_item.replace("_", " ")} Reference: {ref_source}', icon="⚠")

            elif item == "Portrait_Plot_seasonal":
                col1, col2 = st.columns((1, 2))
                col1.write("##### :green[Please choose!]")
                showing_format = col1.radio(
                    "Portrait_Plot_seasonal", ["***Variables***", "***Matrics***"],
                    captions=["Showing by Variables.", "Showing by Matrics."], index=None, horizontal=False,
                    label_visibility="collapsed")

                if showing_format == '***Variables***':
                    col21, col22 = col2.columns(2)
                    col21.write("##### :green[Select Variables!]")
                    iselected_item = col21.radio("Portrait_Plot_seasonal", [i.replace("_", " ") for i in self.selected_items],
                                                 index=None,
                                                 horizontal=False, key=f'{item}_item', label_visibility="collapsed")
                    col22.write("###### :green[Select Matrics or scores!]")
                    mm = col22.radio("Portrait_Plot_seasonal", ['metrics', 'scores'], index=None, horizontal=False,
                                     key=f'{item}_score',
                                     label_visibility="collapsed")
                    st.divider()
                    if iselected_item and mm:
                        selected_item = iselected_item.replace(" ", "_")
                        ref_sources = self.ref['general'][f'{selected_item}_ref_source']
                        if isinstance(ref_sources, str): ref_sources = [ref_sources]
                        for ref_source in ref_sources:
                            filename = glob.glob(os.path.join(figure_path, f'{selected_item}_{ref_source}_{mm}.*'))
                            try:
                                image = load_image(filename[0])
                                st.image(image, caption=f'Reference: {ref_source}', use_container_width="auto")
                            except:
                                st.error(f'Missing Figure for Reference: {ref_source}', icon="⚠")
                elif showing_format == '***Matrics***':
                    df = pd.read_csv(figure_path + "/Portrait_Plot_seasonal.txt", sep='\s+', header=0)
                    import itertools
                    filtered_df = df.groupby("Item")[["Reference"]].agg(lambda x: list(x.unique())).reset_index()
                    all_combinations = list(itertools.product(*filtered_df['Reference']))
                    col2.write("##### :green[Select Matrics or scores!!]")
                    score = col2.radio("Portrait_Plot_seasonal", [k.replace("_", " ") for k, v in self.scores.items() if v],
                                       index=None, horizontal=True, key=f'{item}_score', label_visibility="collapsed")
                    st.divider()
                    if score:
                        score = score.replace(" ", "_")
                        for item_combination in all_combinations:
                            filename = glob.glob(os.path.join(figure_path, f'{score}_{"_".join(item_combination)}.*'))
                            try:
                                image = load_image(filename[0])
                                st.image(image, caption=f'Reference: {", ".join(item_combination)}', use_container_width="auto")
                            except:
                                st.error(f'Missing Figure for Reference:{", ".join(item_combination)}', icon="⚠")

            elif item == "Parallel_Coordinates":
                col1, col2 = st.columns((1, 2))
                col1.write("##### :green[Please choose!]")
                showing_format = col1.radio(
                    "Parallel_Coordinates", ["***Variables***", "***Matrics***"],
                    captions=["Showing by Variables.", "Showing by Matrics."], index=None, horizontal=False, key=item,
                    label_visibility="collapsed")

                if showing_format == '***Variables***':
                    col21, col22 = col2.columns(2)
                    col21.write("##### :green[Select Variables!]")
                    iselected_item = col21.radio("Parallel_Coordinates", [i.replace("_", " ") for i in self.selected_items],
                                                 index=None,
                                                 horizontal=False, key=f'{item}_item', label_visibility="collapsed")
                    col22.write("###### :green[Select Matrics or scores!]")
                    mm = col22.radio("Parallel_Coordinates", ['metrics', 'scores'], index=None, horizontal=False,
                                     key=f'{item}_score',
                                     label_visibility="collapsed")
                    st.divider()
                    if iselected_item and mm:
                        selected_item = iselected_item.replace(" ", "_")
                        ref_sources = self.ref['general'][f'{selected_item}_ref_source']
                        if isinstance(ref_sources, str): ref_sources = [ref_sources]
                        for ref_source in ref_sources:
                            filename = glob.glob(
                                os.path.join(figure_path, f'Parallel_Coordinates_Plot_{mm}_{selected_item}_{ref_source}.*'))
                            try:
                                image = load_image(filename[0])
                                st.image(image, caption=f'Reference: {ref_source.replace(" ", "_")}', use_container_width="auto")
                            except:
                                st.error(f'Missing Figure for Reference: {ref_source.replace(" ", "_")}', icon="⚠")
                elif showing_format == '***Matrics***':
                    df = pd.read_csv(figure_path + "/Parallel_Coordinates_evaluations.txt", sep='\s+', header=0)
                    import itertools
                    filtered_df = df.groupby("Item")[["Reference"]].agg(lambda x: list(x.unique())).reset_index()
                    all_combinations = list(itertools.product(*filtered_df['Reference']))
                    col2.write("##### :green[Select Matrics or scores!!]")
                    iscore = col2.radio("###### Matrics and scores!", [k.replace("_", " ") for k, v in self.scores.items() if v],
                                        index=None, horizontal=True, key=f'{item}_score', label_visibility="collapsed")
                    st.divider()
                    if iscore:
                        score = iscore.replace(" ", "_")
                        for item_combination in all_combinations:
                            filename = glob.glob(
                                os.path.join(figure_path, f'Parallel_Coordinates_Plot_{score}_{"_".join(item_combination)}.*'))
                            try:
                                image = load_image(filename[0])
                                st.image(image, caption=f'References: {", ".join(item_combination)}', use_container_width="auto")
                            except:
                                st.error(f'Missing Figure for Reference: {", ".join(item_combination)}', icon="⚠")

            elif (item == "Kernel_Density_Estimate") | (item == "Whisker_Plot") | (item == "Ridgeline_Plot"):
                col1, col2 = st.columns((1.5, 2.5))
                col1.write('##### :blue[Select Variables]')
                iselected_item = col1.radio(item, [i.replace("_", " ") for i in self.selected_items], index=None,
                                            horizontal=False,
                                            key=f'{item}_item', label_visibility="collapsed")
                col2.write('##### :blue[Select Matrics and scores]')
                imm = col2.radio(item,
                                 [k.replace("_", " ") for k, v in dict(chain(self.metrics.items(), self.scores.items())).items()
                                  if v],
                                 index=None, horizontal=True, key=f'{item}_score', label_visibility="collapsed")
                st.divider()
                if iselected_item and imm:
                    selected_item = iselected_item.replace(" ", "_")
                    mm = imm.replace(" ", "_")
                    ref_sources = self.ref['general'][f'{selected_item}_ref_source']
                    if isinstance(ref_sources, str): ref_sources = [ref_sources]
                    for ref_source in ref_sources:
                        ffname = f'{iselected_item}: Reference -- {ref_source.replace("_", " ")} --{imm}'
                        filenames = glob.glob(os.path.join(figure_path, f'{item}_{selected_item}_{ref_source}_{mm}.*'))
                        try:
                            image = load_image(filenames[0])
                            st.image(image, caption=ffname, use_container_width="auto")
                        except:
                            if mm == 'nSpatialScore':
                                st.info(f'{mm} is not supported for {item.replace("_", " ")}!', icon="ℹ️")
                            else:
                                st.error(f'Missing Figure for {ffname}', icon="⚠")

            elif item == "Single_Model_Performance_Index":
                filename = glob.glob(os.path.join(figure_path, f'SMPI_comparison_plot_comprehensive.*'))
                try:
                    image = load_image(filename[0])
                    st.image(image, caption='SMIP', use_container_width="auto")
                except:
                    st.error(f'Missing Figure for SMIP', icon="⚠")

            elif item == "Relative_Score":
                st.info(f'Relative_Score not ready yet!', icon="ℹ️")

            elif item == "Diff_Plot":
                col1, col2, col3 = st.columns((1, 1.2, 2))
                col1.write("##### :green[Please choose!]")
                showing_format = col1.radio(
                    "Parallel_Coordinates", ["***Anomaly***", "***Differentiate***"], index=None, horizontal=False, key=item,
                    label_visibility="collapsed")
                col2.write("##### :green[Please choose!]")
                iselected_item = col2.radio("Parallel_Coordinates", [f'***{i.replace("_", " ")}***' for i in self.selected_items],
                                            index=None, horizontal=False, key=f'{item}_item', label_visibility="collapsed")
                col3.write("##### :green[Please choose!]")
                mm = col3.radio("metric", [f'***{key.replace("_", " ")}***' for key, value in self.metrics.items() if value] + [
                    f'***{key.replace("_", " ")}***' for key, value in self.scores.items() if value],
                                index=None, horizontal=True,
                                key=f'{item}_metrics', label_visibility="collapsed")

                st.divider()
                if showing_format and iselected_item and mm:
                    mm = mm.replace("***", "").replace(" ", "_")
                    selected_item = iselected_item.replace("***", "").replace(" ", "_")
                    ref_sources = self.ref['general'][f'{selected_item}_ref_source']
                    if isinstance(ref_sources, str): ref_sources = [ref_sources]
                    sim_sources = self.sim['general'][f'{selected_item}_sim_source']
                    if isinstance(sim_sources, str): sim_sources = [sim_sources]

                    col1, col2 = st.columns(2)
                    col1.write("##### :green[Please choose Reference!]")
                    ref_source = col1.radio(
                        "Reference", [i for i in ref_sources],
                        index=None, horizontal=False, key=f'{item}_Reference',
                        label_visibility="collapsed")
                    col2.write("##### :green[Please choose!]")
                    col21, col22 = col2.columns(2)
                    sim_source = col21.radio("sim_sources",
                                             [i for i in sim_sources],
                                             index=None, horizontal=False, key=f'{item}_simulation', label_visibility="collapsed")
                    st.divider()
                    if ref_source and sim_source:
                        ref_data_type = self.ref[ref_source]['general']['data_type']
                        if showing_format == '***Anomaly***':
                            if ref_data_type != 'stn':
                                file = f'{selected_item}_ref_{ref_source}_sim_{sim_source}_{mm}_anomaly'
                                filename = glob.glob(
                                    os.path.join(figure_path,
                                                 f'{selected_item}_ref_{ref_source}_sim_{sim_source}_{mm}_anomaly.*'))
                                filename = [f for f in filename if not f.endswith('.nc')]
                            else:
                                file = f'{selected_item}_stn_{ref_source}_sim_{sim_source}_{mm}_anomaly*'
                                filename = glob.glob(os.path.join(figure_path, file))
                                filename = [f for f in filename if not f.endswith('.csv')]
                            try:
                                image = load_image(filename[0])
                                st.image(image, caption=f'File: {file}', use_container_width="auto")
                            except:
                                st.error(f'Missing Figure for File: {file}', icon="⚠")
                        else:
                            sim_source1 = col22.radio("sim_sources1",
                                                      [i for i in sim_sources if i != sim_source],
                                                      index=None, horizontal=False, key=f'{item}_simulation1',
                                                      label_visibility="collapsed")
                            if sim_source1:
                                if ref_data_type != 'stn':
                                    file = f'{selected_item}_ref_{ref_source}_{sim_source}_vs_{sim_source1}_{mm}_diff.*'
                                    filenames = glob.glob(os.path.join(figure_path, file))
                                    if len(filenames) == 0:
                                        file = f'{selected_item}_ref_{ref_source}_{sim_source1}_vs_{sim_source}_{mm}_diff.*'
                                        filenames = glob.glob(os.path.join(figure_path, file))
                                    filenames = [f for f in filenames if not f.endswith('.nc')]
                                else:
                                    sim_varname_1 = self.sim[sim_source][selected_item][f'varname']
                                    sim_varname_2 = self.sim[sim_source1][selected_item][f'varname']

                                    file = f"{selected_item}_stn_{ref_source}_{sim_source}_{sim_varname_1}_vs_{sim_source1}_{sim_varname_2}_{mm}_diff.*"
                                    filenames = glob.glob(os.path.join(figure_path, file))
                                    if len(filenames) == 0:
                                        file = f"{selected_item}_stn_{ref_source}_{sim_source1}_{sim_varname_2}_vs_{sim_source}_{sim_varname_1}_{mm}_diff.*"
                                        filenames = glob.glob(os.path.join(figure_path, file))
                                    filenames = [f for f in filenames if not f.endswith('.csv')]

                                for filename in filenames:
                                    try:
                                        image = load_image(filename)
                                        st.image(image, caption=f'File: {filename.replace(f"{figure_path}", "")[1:]}',
                                                 use_container_width="auto")
                                    except:
                                        st.error(f'Missing Figure for File: {filename.replace(f"{figure_path}", "")[1:]}',
                                                 icon="⚠")

                                if len(filenames) == 0:
                                    st.error(f'Missing Figure for File: {selected_item}_ref_{ref_source}_{sim_source}_{mm}_diff',
                                             icon="⚠")

            elif item in ['Mean', 'Median', 'Max', 'Min', 'Sum']:
                col1, col2, col3 = st.columns(3)
                col1.write("##### :green[Items!]")
                iselected_item = col1.radio("Basic Plot", [f'***{i.replace("_", " ")}***' for i in self.selected_items],
                                            index=None, horizontal=False, key=f'{item}_item', label_visibility="collapsed")
                col2.write("##### :green[Please choose!]")
                type = col2.radio("type", ['***Reference***', '***Simulation***'],
                                  index=None, horizontal=False, key=f'{item}_type', label_visibility="collapsed")
                if iselected_item:
                    selected_item = iselected_item.replace("***", "").replace(" ", "_")

                if type == '***Reference***':
                    itype = 'ref'
                    sources = self.ref['general'][f'{selected_item}_ref_source']
                elif type == '***Simulation***':
                    itype = 'sim'
                    sources = self.sim['general'][f'{selected_item}_sim_source']
                else:
                    sources = None
                    st.info('Please choose showing type!')

                if iselected_item and type:
                    col3.write("##### :green[Sources!]")
                    source = col3.radio("col3", [source for source in sources],
                                        index=None, horizontal=False, key=f'{item}_source', label_visibility="collapsed")
                    st.divider()
                    if source and type == '***Reference***':
                        data_type = self.ref[source]['general']['data_type']
                    elif source and type == '***Simulation***':
                        data_type = self.sim[source]['general']['data_type']

                    if source:
                        if data_type != 'stn':
                            filenames = glob.glob(
                                os.path.join(figure_path, f'{selected_item}_{itype}_{source}_*_{item}.*'))
                            filenames = [f for f in filenames if not f.endswith('.nc')]
                            for filename in filenames:
                                try:
                                    image = load_image(filename)
                                    st.image(image, caption=f'File: {filename.replace(f"{figure_path}/", "")}',
                                             use_container_width="auto")
                                except:
                                    st.error(f'Missing Figure for File: {filename.replace(f"{figure_path}/", "")}', icon="⚠")

                            if len(filenames) == 0:
                                st.error(f'Missing Figure for File: {selected_item} {itype} {source} {item}',
                                         icon="⚠")
                        elif data_type == 'stn':

                            if type == '***Reference***':
                                sources1 = self.sim['general'][f'{selected_item}_sim_source']
                            elif type == '***Simulation***':
                                sources1 = self.ref['general'][f'{selected_item}_ref_source']
                            else:
                                sources1 = None
                                st.info('Please choose showing type!')
                            source1 = col3.radio("sources1", [source for source in sources1],
                                                 index=None, horizontal=False, key=f'{item}_source1',
                                                 label_visibility="collapsed")
                            if source1:

                                try:
                                    filenames = glob.glob(
                                        os.path.join(figure_path, f'{selected_item}_stn*{source}_{source1}*{item}_*.*'))
                                except:
                                    filenames = glob.glob(
                                        os.path.join(figure_path, f'{selected_item}_stn*{source1}_{source}*{item}_*.*'))

                                filenames = [f for f in filenames if not f.endswith('.csv')]
                                for filename in filenames:
                                    try:
                                        image = load_image(filename)
                                        st.image(image, caption=f'File: {filename.replace(f"{figure_path}/", "")}',
                                                 use_container_width="auto")
                                    except:
                                        st.error(f'Missing Figure for File: {filename.replace(f"{figure_path}/", "")}', icon="⚠")

                                if len(filenames) == 0:
                                    st.error(f'Missing Figure for File: {selected_item} {source} {item}', icon="⚠")

            elif item == "Mann_Kendall_Trend_Test":
                st.markdown(f"""
                <div style="font-size:22px; font-weight:bold; color:#68838B; border-bottom:3px solid #68838B; padding: 5px;">
                    Select Cases!
                </div>""", unsafe_allow_html=True)
                st.write(' ')

                col1, col2, col3 = st.columns(3)
                iselected_item = col1.radio("Mann_Kendall_Trend_Test_item",
                                            [f'{i.replace("_", " ")}' for i in self.selected_items],
                                            index=None, horizontal=False, key=f'{item}_item', label_visibility="collapsed")
                type = col2.radio("Mann_Kendall_Trend_Test_type", ['Reference', 'Simulation'],
                                  index=None, horizontal=False, key=f'{item}_type', label_visibility="collapsed")

                if iselected_item and type:
                    selected_item = iselected_item.replace("***", "").replace(" ", "_")
                    if type == 'Reference':
                        itype = 'ref'
                        sources = self.ref['general'][f'{selected_item}_ref_source']
                    elif type == 'Simulation':
                        itype = 'sim'
                        sources = self.sim['general'][f'{selected_item}_sim_source']

                    source = col3.radio("Mann_Kendall_Trend_Test_source", [source for source in sources],
                                        index=None, horizontal=False, key=f'{item}_source', label_visibility="collapsed")
                    st.divider()
                    if source:
                        if type == 'Reference':
                            data_type = self.ref[source]['general']['data_type']
                        elif type == 'Simulation':
                            data_type = self.sim[source]['general']['data_type']

                        if data_type == 'stn':
                            st.info('Function for station data is still on develop!')
                        else:
                            tau = glob.glob(
                                os.path.join(figure_path, f'Mann_Kendall_Trend_Test_{selected_item}_{itype}_{source}*tau.*'))[0]
                            trend = glob.glob(
                                os.path.join(figure_path, f'Mann_Kendall_Trend_Test_{selected_item}_{itype}_{source}*Trend.*'))[0]
                            if os.path.exists(tau):
                                image = load_image(tau)
                                st.image(image, caption=f'Case: {selected_item} {source} tau', use_container_width=True)
                            else:
                                st.error(f'Missing Figure for Case: {selected_item} {source}', icon="⚠")
                            if os.path.exists(trend):
                                image = load_image(trend)
                                st.image(image, caption=f'Case: {selected_item} {source} trend', use_container_width=True)
                            else:
                                st.error(f'Missing Figure for Case: {selected_item} {source}', icon="⚠")

            elif item == "Correlation":
                st.markdown(f"""
                <div style="font-size:22px; font-weight:bold; color:#68838B; border-bottom:3px solid #68838B; padding: 5px;">
                    Select Cases!
                </div>""", unsafe_allow_html=True)
                st.write(' ')

                col1, col2 = st.columns(2)
                iselected_item = col1.radio("Correlation_item", [f'{i.replace("_", " ")}' for i in self.selected_items],
                                            index=None, horizontal=False, key=f'{item}_item', label_visibility="collapsed")

                if iselected_item:
                    selected_item = iselected_item.replace("***", "").replace(" ", "_")
                    sources = self.sim['general'][f'{selected_item}_sim_source']
                    if isinstance(sources, str): sources = [sources]
                    source = col2.radio("Correlation_source", [source for source in sources],
                                        index=None, horizontal=False, key=f'{item}_source', label_visibility="collapsed")

                    st.divider()
                    if source:
                        filename = glob.glob(os.path.join(figure_path, f'Correlation_{selected_item}*{source}*.*'))
                        filename = [f for f in filename if not f.endswith('.nc')]
                        for file in filename:
                            try:
                                image = load_image(file)
                                st.image(image, caption=f'Case: {file.replace(f"{figure_path}", "")[1:].replace("_", " ")}',
                                         use_container_width=True)
                            except:
                                st.error(f'Missing Figure for Case: {file.replace(f"{figure_path}", "")[1:].replace("_", " ")}',
                                         icon="⚠")

            elif item == "Standard_Deviation":
                st.markdown(f"""
                <div style="font-size:22px; font-weight:bold; color:#68838B; border-bottom:3px solid #68838B; padding: 5px;">
                    Select Cases!
                </div>""", unsafe_allow_html=True)
                st.write(' ')

                col1, col2, col3 = st.columns(3)
                iselected_item = col1.radio("Standard_Deviation_item",
                                            [f'{i.replace("_", " ")}' for i in self.selected_items],
                                            index=None, horizontal=False, key=f'{item}_item', label_visibility="collapsed")
                type = col2.radio("Standard_Deviation_type", ['Reference', 'Simulation'],
                                  index=None, horizontal=False, key=f'{item}_type', label_visibility="collapsed")

                if iselected_item and type:
                    selected_item = iselected_item.replace("***", "").replace(" ", "_")
                    if type == 'Reference':
                        itype = 'ref'
                        sources = self.ref['general'][f'{selected_item}_ref_source']
                    elif type == 'Simulation':
                        itype = 'sim'
                        sources = self.sim['general'][f'{selected_item}_sim_source']

                    if isinstance(sources, str): sources = [sources]
                    source = col3.radio("Standard_Deviation_source", [source for source in sources],
                                        index=None, horizontal=False, key=f'{item}_source', label_visibility="collapsed")
                    st.divider()
                    if source:
                        if type == 'Reference':
                            data_type = self.ref[source]['general']['data_type']
                        elif type == 'Simulation':
                            data_type = self.sim[source]['general']['data_type']

                        if data_type == 'stn':
                            st.info('Function for station data is still on develop!')
                        else:
                            filename = glob.glob(
                                os.path.join(figure_path, f'Standard_Deviation_{selected_item}_{itype}_{source}*.*'))
                            filename = [f for f in filename if not f.endswith('.nc')]
                            try:
                                image = load_image(filename[0])
                                st.image(image, caption=f'Case: {selected_item} {source}', use_container_width=True)
                            except:
                                st.error(f'Missing Figure for Case: {selected_item} {source}', icon="⚠")

            elif item == "Functional_Response":
                st.markdown(f"""
                <div style="font-size:22px; font-weight:bold; color:#68838B; border-bottom:3px solid #68838B; padding: 5px;">
                    Select Cases!
                </div>""", unsafe_allow_html=True)
                st.write(' ')

                col1, col2, col3 = st.columns(3)
                iselected_item = col1.radio("Functional_Response_item",
                                            [f'{i.replace("_", " ")}' for i in self.selected_items],
                                            index=None, horizontal=False, key=f'{item}_item', label_visibility="collapsed")
                if iselected_item:
                    selected_item = iselected_item.replace("***", "").replace(" ", "_")
                    ref_sources = self.ref['general'][f'{selected_item}_ref_source']
                    sim_sources = self.sim['general'][f'{selected_item}_sim_source']
                    if isinstance(ref_sources, str): ref_sources = [ref_sources]
                    if isinstance(sim_sources, str): sim_sources = [sim_sources]
                    ref_source = col2.radio("Functional_Response_refsource", [source for source in ref_sources],
                                            index=None, horizontal=False, key=f'{item}_refsource', label_visibility="collapsed")
                    sim_source = col3.radio("Functional_Response_simsource", [source for source in sim_sources],
                                            index=None, horizontal=False, key=f'{item}_simsource', label_visibility="collapsed")
                    st.divider()
                    if ref_source and sim_source:
                        filename = glob.glob(
                            os.path.join(figure_path, f'Functional_Response_{selected_item}_ref_{ref_source}_sim_{sim_source}.*'))
                        filename = [f for f in filename if not f.endswith('.nc')]
                        try:
                            image = load_image(filename[0])
                            st.image(image, caption=f'Case: {selected_item} ref:{ref_source} sim:{sim_source}',
                                     use_container_width=True)
                        except:
                            st.error(f'Missing Figure for Case: {selected_item} ref:{ref_source} sim:{sim_source}', icon="⚠")


class visualization_replot_files:
    def __init__(self):
        self.author = "Qingchen Xu/xuqingchen0@gmail.com"
        self.coauthor = "Zhongwang Wei/@gmail.com"

        # self.classification = initial.classification()

        # ------------------------
        self.ref = st.session_state.ref_data
        self.sim = st.session_state.sim_data
        # ----------------------------
        self.generals = st.session_state.generals
        self.evaluation_items = st.session_state.evaluation_items
        self.metrics = st.session_state.metrics
        self.scores = st.session_state.scores
        self.comparisons = st.session_state.comparisons
        self.statistics = st.session_state.statistics

        self.selected_items = st.session_state.selected_items
        self.tittles = st.session_state.tittles
        self.nl = NamelistReader()

    def Showing_for_files(self):
        # st.write('files')

        case_path = os.path.join(self.generals['basedir'], self.generals['basename'], "output")

        color = '#9DA79A'
        st.markdown(f"""
        <div style="font-size:20px; font-weight:bold; color:{color}; border-bottom:3px solid {color}; padding: 5px;">
             What's your choice?....
        </div>
        """, unsafe_allow_html=True)
        st.write('')
        selected_item = st.radio(
            "#### What's your choice?", [selected_item.replace("_", " ") for selected_item in self.selected_items], index=None,
            horizontal=True, label_visibility='collapsed')

        if selected_item:
            selected_item = selected_item.replace(" ", "_")
            col1, col2 = st.columns(2)
            with col1:
                color = '#C48E8E'
                st.markdown(f"""
                <div style="font-size:18px; font-weight:bold; color:{color}; padding: 0px;">
                     > Reference
                </div>
                """, unsafe_allow_html=True)

                if len(self.ref['general'][f'{selected_item}_ref_source']) == 1:
                    ref_index = 0
                else:
                    ref_index = None

                return_refselect = st.radio("###### > Reference", self.ref['general'][f'{selected_item}_ref_source'],
                                            index=ref_index, horizontal=False, label_visibility='collapsed')
            with col2:
                color = '#68838B'
                st.markdown(f"""
                <div style="font-size:18px; font-weight:bold; color:{color}; padding: 2px;">
                     > Simulation
                </div>
                """, unsafe_allow_html=True)
                if len(self.sim['general'][f'{selected_item}_sim_source']) == 1:
                    sim_index = 0
                else:
                    sim_index = None
                return_simselect = st.radio("###### > Simulation", self.sim['general'][f'{selected_item}_sim_source'],
                                            index=sim_index, horizontal=False, label_visibility='collapsed')
            st.divider()
            if (return_refselect is not None) & (return_simselect is not None):
                ref_dt = self.ref[return_refselect]['general'][f'data_type']
                sim_dt = self.sim[return_simselect]['general'][f'data_type']

                if ref_dt != 'stn' and sim_dt != 'stn':
                    self._geo_geo_visual(selected_item, return_refselect, return_simselect, case_path)
                elif ref_dt != 'stn' and sim_dt == 'stn':
                    st.write(ref_dt, sim_dt)
                    self._geo_stn_visual(selected_item, return_refselect, return_simselect, case_path)
                elif ref_dt == 'stn' and sim_dt != 'stn':
                    self._stn_geo_visual(selected_item, return_refselect, return_simselect, case_path)
                elif ref_dt == 'stn' and sim_dt == 'stn':
                    self._stn_stn_visual(selected_item, return_refselect, return_simselect, case_path)
                else:
                    st.write("Error: ref_data_type and sim_data_type are not defined!")

    def _geo_geo_visual(self, selected_item, refselect, simselect, path):
        left, right = st.columns((2.5, 5))
        with left:
            plot_type = st.selectbox('Please choose your type',
                                     ['Geo metrics replot', 'Time average', 'Other Functions'],
                                     # 'Time average', 'Compare lines',
                                     index=None, placeholder="Choose an option", label_visibility="visible")
        if plot_type == 'Time average':
            self.__generate_image_geo_time_average(selected_item, refselect, simselect, path)
        elif plot_type == 'Compare lines':
            self.__generate_image_geo_Compare_lines(selected_item, path)
        elif plot_type == 'Geo metrics replot':
            with right:
                select_to_plot = st.selectbox('Metrics',
                                              [k for k, v in dict(chain(self.metrics.items(), self.scores.items())).items() if
                                               v], placeholder="Choose an option", label_visibility="visible",
                                              key='_geo_geo_Site_metrics_replot')

            if select_to_plot in self.metrics:
                mm = 'metrics'
            elif select_to_plot in self.scores:
                mm = 'scores'
            self.__generate_image_geo_index(mm, select_to_plot, selected_item, refselect, simselect, path)
        elif plot_type == 'Other Functions':
            st.info('Some functions are not perfect yet, Coming soon...', icon="ℹ️")

    def _geo_stn_visual(self, selected_item, refselect, simselect, path):
        left, right = st.columns((2.5, 5))
        with left:
            plot_type = st.selectbox('Please choose your type', ['Site metrics replot', 'Site replot', 'Other Functions'],
                                     # 'Hist-plot',
                                     index=None, placeholder="Choose an option", label_visibility="visible")
        if plot_type == 'Site replot':
            self._showing_stn_data(selected_item, refselect, simselect, path, ('grid', 'stn'))
        elif plot_type == 'Other Functions':
            st.info('Some functions are not perfect yet, Coming soon...', icon="ℹ️")

    def _stn_geo_visual(self, selected_item, refselect, simselect, path):
        # streamlit # hist plot 可以做
        left, right = st.columns((2.5, 5))
        with left:
            plot_type = st.selectbox('Please choose your type', ['Site metrics replot', 'Site replot', 'Other Functions'],
                                     # 'Hist-plot', 'Site data',
                                     index=None, placeholder="Choose an option", label_visibility="visible")
        if plot_type == 'Site replot':
            self._showing_stn_data(selected_item, refselect, simselect, path, ('stn', 'grid'))
        elif plot_type == 'Site metrics replot':
            with right:
                select_to_plot = st.selectbox('Metrics',
                                              [k for k, v in dict(chain(self.metrics.items(), self.scores.items())).items() if
                                               v], placeholder="Choose an option", label_visibility="visible",
                                              key='_stn_geo_Site_metrics_replot')
                if select_to_plot in self.metrics:
                    mm = 'metrics'
                elif select_to_plot in self.scores:
                    mm = 'scores'
            self.__generate_image_stn_index(mm, select_to_plot, selected_item, refselect, simselect, path)


        elif plot_type == 'Hist-plot':
            with right:
                select_to_plot = st.multiselect('Metrics',
                                                [k for k, v in dict(chain(self.metrics.items(), self.scores.items())).items()
                                                 if
                                                 v],
                                                placeholder="Choose an option", label_visibility="visible")
        elif plot_type == 'Other Functions':
            st.info('Some functions are not perfect yet, Coming soon...', icon="ℹ️")

    def _stn_stn_visual(self, selected_item, refselect, simselect, path):
        left, right = st.columns((2.5, 5))
        with left:
            plot_type = st.selectbox('Please choose your type', ['Site metrics replot', 'Site replot', 'Other Functions'],
                                     index=None, placeholder="Choose an option", label_visibility="visible")
        if plot_type == 'Site replot':
            self._showing_stn_data(selected_item, refselect, simselect, path, ('stn', 'stn'))
        elif plot_type == 'Hist-plot':
            with right:
                select_to_plot = st.multiselect('Metrics',
                                                [k for k, v in dict(chain(self.metrics.items(), self.scores.items())).items()
                                                 if
                                                 v],
                                                placeholder="Choose an option", label_visibility="visible")
        elif plot_type == 'Other Functions':
            st.info('Some functions are not perfect yet, Coming soon...', icon="ℹ️")

    def _showing_stn_data(self, selected_item, refselect, simselect, path, data_type):
        stn_data = pd.read_csv(path + '/scores/' + f'{selected_item}_stn_{refselect}_{simselect}_evaluations.csv',
                               header=0)
        del_col = ['sim_dir', 'ref_dir', 'sim_syear', 'sim_eyear', 'ref_syear', 'ref_eyear', 'sim_lat', 'sim_lon', 'Flag']
        stn_data.drop(columns=[col for col in stn_data.columns if col in del_col], inplace=True)
        new_names = {
            'ref_lon': 'Longitude',
            'ref_lat': 'Latitude',
            'use_syear': 'Start year',
            'use_eyear': 'End year'
        }
        ref_type, sim_type = data_type
        ref_var = self.ref[refselect][selected_item][f"varname"]
        ref_unit = self.ref[refselect][selected_item][f"varunit"]
        if sim_type == 'stn':
            sim_var = self.sim[simselect][selected_item][f"varname"]
            sim_unit = self.sim[refselect][selected_item][f"varunit"]
        else:
            try:
                sim_var = self.sim[simselect][selected_item][f"varname"]
                sim_unit = self.sim[refselect][selected_item][f"varunit"]
            except:
                nml = self.nl.read_namelist(self.sim[simselect]['general']['model_namelist'])
                sim_var = nml[selected_item][f"varname"]
                sim_unit = nml[selected_item][f"varunit"]

        stn_data.rename(columns=new_names, inplace=True)
        mean_data = stn_data.mean(skipna=True, numeric_only=True)
        mean_data = pd.Series(['Mean'] + mean_data.to_list(), index=stn_data.columns, name='Mean')
        stn_data.insert(0, f'Select', False)
        ms = [k for k, v in dict(chain(self.metrics.items(), self.scores.items())).items() if v]

        def data_editor_change(key, editor_key):
            """Callback function of data_editor. """
            # st.write(key, editor_key)
            st.session_state[key] = apply_de_change(st.session_state[editor_key])
            st.session_state['Submit_button'] = False

        def apply_de_change(changes):
            """Apply changes of data_editor."""
            last_edited = st.session_state['new']
            edited_rows = changes.get('edited_rows')
            edited_indexs = [k for k, v in edited_rows.items() if v['Select']]

            unique_values, counts = np.unique(last_edited + edited_indexs, return_counts=True)
            different_values = unique_values[counts == 1]

            st.session_state['new'] = edited_indexs
            return different_values

        df_key = 'stn_data'  # value_key of data_editor
        df_editor_key = '__stn_data'

        if 'new' not in st.session_state or df_editor_key not in st.session_state or st.session_state[df_editor_key][
            'edited_rows'] is None:  # initialize session_state.value_key
            st.session_state['new'] = []
        if df_key not in st.session_state:
            st.session_state[df_key] = None

        if 'Submit_button' not in st.session_state:  # initialize session_state.value_key
            st.session_state['Submit_button'] = False

        def Submit_button():
            st.session_state['Submit_button'] = True

        with st.container(border=True):
            chart_df = st.data_editor(
                stn_data.copy(),
                key=df_editor_key,  # set editor_key
                on_change=data_editor_change,  # callback function
                args=(df_key, df_editor_key),
                column_config={
                    'Select': st.column_config.CheckboxColumn(
                        "Select",
                        default=False),
                },
                num_rows='fixed',
                use_container_width=True,
                hide_index=False)
            if not st.session_state['Submit_button']:
                if st.session_state[df_key] is not None:
                    idx = st.session_state[df_key][0]
                    id = chart_df.loc[idx, "ID"]
                    use_syear = chart_df.loc[idx, "Start year"]
                    use_eyear = chart_df.loc[idx, "End year"]
                    data_path = path + f'/data/stn_{refselect}_{simselect}/'
                    try:
                        ref_data = xr.open_dataset(data_path + f'{selected_item}_ref_{id}_{use_syear}_{use_eyear}.nc')[ref_var]
                        sim_data = xr.open_dataset(data_path + f'{selected_item}_sim_{id}_{use_syear}_{use_eyear}.nc')[sim_var]
                        if 'lat' in ref_data.dims and 'lon' in ref_data.dims:
                            ref_data = ref_data.squeeze('lat').squeeze('lon')
                        if 'lat' in sim_data.dims and 'lon' in sim_data.dims:
                            sim_data = sim_data.squeeze('lat').squeeze('lon')
                        chart_data = pd.DataFrame({'Time': ref_data.time,
                                                   "Simulation": sim_data.values,
                                                   "Reference": ref_data.values, })
                        st.write(f'###### :blue[{id}]')

                        if len(ms) > 4:
                            scores = [k for k, v in self.scores.items() if v]
                            m_datas = chart_df.loc[idx, scores].to_list()
                            scores = np.delete(scores, np.where(np.isnan(m_datas)))
                            m_datas = np.delete(m_datas, np.where(np.isnan(m_datas)))
                            cols = st.columns(len(scores))
                            for i, col in enumerate(cols):
                                col.metric(scores[i], value=f"{m_datas[i]:.3f}",
                                           delta=f"{m_datas[i] - mean_data[f'{scores[i]}']:.3f}",
                                           delta_color="normal", help='Compare to mean value in all sites',
                                           label_visibility="visible")

                        else:
                            cols = st.columns(len(ms))
                            for i, col in enumerate(cols):
                                m_data = chart_df.loc[idx, f'{ms[i]}']
                                col.metric(ms[i], value=f"{m_data:.3f}", delta=f"{m_data - mean_data[f'{ms[i]}']:.3f}",
                                           delta_color="normal", help='Compare to mean value in all sites',
                                           label_visibility="visible")

                        st.line_chart(chart_data, x='Time', y=["Simulation", "Reference"], color=['#F96969', '#599AD4'])
                        del ref_data, sim_data, chart_data

                    except FileNotFoundError:
                        st.error(f"{id} File not found. Please check the file path.")
            st.button('Submitted', on_click=Submit_button)
        if st.session_state['Submit_button']:
            showing_items = chart_df[chart_df['Select']]
            self.__generate_image_stn_lines(showing_items, selected_item, refselect, simselect,
                                            path, (ref_var, sim_var), (ref_unit, sim_unit))

    def __generate_image_stn_index(self, item, metric, selected_item, ref, sim, path):
        make_stn_plot_index(item, metric, selected_item, ref, sim, path)

    def __generate_image_stn_lines(self, showing_items, selected_item, refselect, simselect, path, vars, unit):
        make_stn_lines(showing_items, selected_item, refselect, simselect, path, vars, unit)

    def __generate_image_geo_index(self, item, metric, selected_item, ref, sim, path):
        make_geo_plot_index(item, metric, selected_item, ref, sim, path)

    def __generate_image_geo_time_average(self, selected_item, refselect, simselect, path):
        make_geo_time_average(selected_item, refselect, simselect, path, self.ref, self.sim, self.nl)

    def __generate_image_geo_Compare_lines(self, selected_item, path):
        option = {}

        col1, col2, col3, col4 = st.columns((3.5, 3, 3, 3))
        with col1:
            option['title'] = st.text_input('Title', label_visibility="visible")
            option['title_size'] = st.number_input("Title label size", min_value=0, value=20)

        with col2:
            option['xticklabel'] = st.text_input('X tick labels', value='Time', label_visibility="visible")
            option['xtick'] = st.number_input("xtick label size", min_value=0, value=17)
        with col3:
            option['yticklabel'] = st.text_input('Y tick labels', label_visibility="visible")
            option['ytick'] = st.number_input("ytick label size", min_value=0, value=17)

        with col4:
            option['grid'] = st.toggle("Showing grid", value=False, label_visibility="visible")
            if option['grid']:
                option['grid_style'] = st.selectbox('Grid Line Style', ['solid', 'dotted', 'dashed', 'dashdot'],
                                                    index=2, placeholder="Choose an option", label_visibility="visible")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            option['labelsize'] = st.number_input("labelsize", min_value=0, value=17)
        with col2:
            option['axes_linewidth'] = st.number_input("axes linewidth", min_value=0, value=1)
        with col3:
            option['grid_linewidth'] = st.number_input("grid linewidth", min_value=0, value=1)
        with col4:
            option['linewidth'] = st.number_input("Lines width", min_value=0, value=1)
        st.divider()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            option["x_wise"] = st.number_input(f"X Length", min_value=0, value=17)
        with col2:
            option["y_wise"] = st.number_input(f"y Length", min_value=0, value=6)
        with col3:
            option["showing_option"] = st.selectbox('Showing option', ['Year', 'Month', 'Day'],  # 'Season',
                                                    index=2, placeholder="Choose an option", label_visibility="visible")
        st.divider()

        col1, col2 = st.columns((2, 1))
        with col1:
            option['plot_type'] = st.radio('Please choose your type', ['sim lines', 'ref lines', 'each site'],
                                           index=None, label_visibility="visible", horizontal=True)
        with col2:
            option['saving_format'] = st.selectbox('Image saving format', ['png', 'jpg', 'eps'],
                                                   index=1, placeholder="Choose an option", label_visibility="visible")

        if option['plot_type'] == 'sim lines':
            if len(option['title']) == 0:
                option['title'] = 'Simulation'
            items = self.sim['general'][f'{selected_item}_sim_source']
        elif option['plot_type'] == 'ref lines':
            if len(option['title']) == 0:
                option['title'] = 'Reference'
            items = self.ref['general'][f'{selected_item}_ref_source']
        else:
            items = list(chain(self.ref['general'][f'{selected_item}_ref_source'],
                               self.sim['general'][f'{selected_item}_sim_source']))

        with st.expander("Edited Lines", expanded=True):
            col1, col2, col3, col4 = st.columns((4, 4, 4, 4))
            for i, item_select in enumerate(items):
                option[item_select] = {}
                with col1:
                    option[f"{item_select}"]['color'] = st.color_picker(f'{item_select} colors', value=None,
                                                                        key=f'{item_select} colors', help=None,
                                                                        on_change=None,
                                                                        args=None, kwargs=None, disabled=False,
                                                                        label_visibility="visible")
                with col2:
                    option[f"{item_select}"]['linestyle'] = st.selectbox(f'{item_select} Line Style',
                                                                         ['solid', 'dotted', 'dashed', 'dashdot'],
                                                                         key=f'{item_select} Line Style',
                                                                         index=None, placeholder="Choose an option",
                                                                         label_visibility="visible")

                with col3:
                    option[f"{item_select}"]['marker'] = st.selectbox(f'{item_select} Line Marker',
                                                                      ['.', 'x', 'o', '<', '8', 's', 'p', '*', 'h', 'H', 'D',
                                                                       'd',
                                                                       'P',
                                                                       'X'], key=f'{item_select} Line Marker',
                                                                      index=None, placeholder="Choose an option",
                                                                      label_visibility="visible")
                with col4:
                    option[f"{item_select}"]['markersize'] = st.number_input(f"{item_select} Markersize",
                                                                             key=f'{item_select} Markersize', min_value=0,
                                                                             value=10)
            option["legend_on"] = st.toggle('Setting or auto', value=False)
            col1, col2, col3, col4 = st.columns((4, 4, 4, 4))
            option["ncol"] = col1.number_input("N cols", value=1, min_value=1, format='%d')
            if not option["legend_on"]:
                option["loc"] = col2.selectbox("Legend location",
                                               ['best', 'right', 'left', 'upper left', 'upper right', 'lower left',
                                                'lower right',
                                                'upper center',
                                                'lower center', 'center left', 'center right'], index=0,
                                               placeholder="Choose an option",
                                               label_visibility="visible")
            else:
                option["bbox_to_anchor_x"] = col3.number_input("X position of legend", value=0.5)
                option["bbox_to_anchor_y"] = col4.number_input("Y position of legend", value=0.)
            # x、y、width、height

        option['data_path'] = path + f'/data/'
        geo_Compare_lines(option, selected_item, self.ref, self.sim)


class visualization_replot_Comparison:
    def __init__(self):
        self.author = "Qingchen Xu/xuqingchen0@gmail.com"
        self.coauthor = "Zhongwang Wei/@gmail.com"

        # self.classification = initial.classification()
        self.nl = NamelistReader()
        # ------------------------
        self.ref = st.session_state.ref_data
        self.sim = st.session_state.sim_data
        # ----------------------------
        self.generals = st.session_state.generals
        self.evaluation_items = st.session_state.evaluation_items
        self.metrics = st.session_state.metrics
        self.scores = st.session_state.scores
        self.comparisons = st.session_state.comparisons
        self.statistics = st.session_state.statistics

        self.selected_items = st.session_state.selected_items
        self.tittles = st.session_state.tittles

    # -=========================================================
    def Comparison_replot(self):

        case_path = os.path.join(self.generals['basedir'], self.generals['basename'], "output", "comparisons")
        if not self.generals['comparison']:
            st.info('You haven\'t selected a comparison module!')
        else:
            showing_item = [k for k, v in self.comparisons.items() if v]
            if not showing_item:
                st.info('No comparison item selected!')

            if self.generals['evaluation']:
                showing_item = ['PFT_groupby', 'IGBP_groupby'] + showing_item
            tabs = st.tabs([k.replace("_", " ") for k in showing_item])
            for i, item in enumerate(showing_item):
                with tabs[i]:
                    self._prepare(case_path, item)

    def _prepare(self, case_path, item):
        st.cache_data.clear()
        dir_path = os.path.join(case_path, item)

        if (item == "IGBP_groupby") | (item == "PFT_groupby"):
            col1, col2 = st.columns(2)
            col1.write('##### :green[Select Variables!]')
            iselected_item = col1.radio("###### Variables!", [i.replace("_", " ") for i in self.selected_items], index=None,
                                        horizontal=False, label_visibility="collapsed",
                                        key=f'{item}_item')
            col2.write("##### :green[Select Matrics or scores!]")
            mm = col2.radio("###### Matrics and scores!", ['metrics', 'scores'], label_visibility="collapsed",
                            index=None, horizontal=False, key=f'{item}_score')
            if iselected_item and mm:
                selected_item = iselected_item.replace(" ", "_")
                st.divider()
                sim_sources = self.sim['general'][f'{selected_item}_sim_source']
                ref_sources = self.ref['general'][f'{selected_item}_ref_source']
                if isinstance(sim_sources, str): sim_sources = [sim_sources]
                if isinstance(ref_sources, str): ref_sources = [ref_sources]
                if len(sim_sources) > 4 or len(ref_sources) > 4:
                    set_sourese = st.expander("", expanded=True)
                    col1, col2 = set_sourese.columns(2)
                else:
                    col1, col2 = st.columns(2)
                col1.write('##### :blue[Select Simulation]')
                sim_source = col1.radio("###### Select your Simulation!", sim_sources, index=0, horizontal=False,
                                        key=f'{item}_sim_source', label_visibility="collapsed")
                col2.write('##### :blue[Select Reference]')
                ref_source = col2.radio("###### Select your reference!", ref_sources, index=0, horizontal=False,
                                        key=f'{item}_ref_source', label_visibility="collapsed")
                st.divider()
                if (self.ref[ref_source]['general'][f'data_type'] != 'stn') & (
                        self.sim[sim_source]['general'][f'data_type'] != 'stn'):
                    path = os.path.abspath(os.path.join(case_path, '../', mm, item, f'{sim_source}___{ref_source}'))
                    heatmap_groupby_file = f"{path}/{selected_item}_{sim_source}___{ref_source}_{mm}.txt"
                    try:
                        self.__heatmap_groupby(item, heatmap_groupby_file, selected_item, mm, sim_source, ref_source, dir_path)
                    except FileNotFoundError:
                        st.error(f'Missing File for Reference: {ref_source} Simulation: {sim_source}', icon="⚠")
                else:
                    st.info(
                        f'Reference: {ref_source}, Simulation: {sim_source}---Heatmap groupby is not supported for station data!',
                        icon="👋")

        if item == "HeatMap":
            st.write('#### Select Scores to replot')
            iscore = st.radio("HeatMap", [k.replace("_", " ") for k, v in self.scores.items() if v],
                              index=None, horizontal=True, key=f'{item}', label_visibility="collapsed")
            if iscore:
                score = iscore.replace(" ", "_")
                heatmap_file = f"{dir_path}/scenarios_{score}_comparison.txt"
                try:
                    self.__heatmap(heatmap_file, score)
                except FileNotFoundError:
                    st.error(f'Missing File for Score: {iscore}', icon="⚠")
        elif (item == "Taylor_Diagram"):
            st.cache_data.clear()
            st.write('##### :blue[Select Variables]')
            iselected_item = st.radio("Taylor_Diagram", [i.replace("_", " ") for i in self.selected_items], index=None,
                                      horizontal=True, key=f'{item}', label_visibility="collapsed")
            if iselected_item:
                selected_item = iselected_item.replace(" ", "_")
                ref_sources = self.ref['general'][f'{selected_item}_ref_source']
                if isinstance(ref_sources, str): ref_sources = [ref_sources]
                st.write('##### :green[Select your reference!]')
                ref_source = st.radio("Taylor_Diagram", ref_sources, index=0, horizontal=True, key=f'{item}_ref_source',
                                      label_visibility="collapsed")
                st.divider()
                if ref_source:
                    taylor_diagram_file = f"{dir_path}/taylor_diagram_{selected_item}_{ref_source}.txt"
                    try:
                        self.__taylor(taylor_diagram_file, selected_item, ref_source)
                    except FileNotFoundError:
                        st.error(f'Missing File for {iselected_item} Reference: {ref_source}', icon="⚠")
        elif (item == "Target_Diagram"):
            st.cache_data.clear()
            st.write('##### :blue[Select Variables]')
            iselected_item = st.radio("Target_Diagram", [i.replace("_", " ") for i in self.selected_items], index=None,
                                      horizontal=True, key=f'{item}', label_visibility="collapsed")
            if iselected_item:
                selected_item = iselected_item.replace(" ", "_")
                ref_sources = self.ref['general'][f'{selected_item}_ref_source']
                if isinstance(ref_sources, str): ref_sources = [ref_sources]
                st.write('##### :green[Select your reference!]')
                ref_source = st.radio("Target_Diagram", ref_sources, index=0, horizontal=True, key=f'{item}_ref_source',
                                      label_visibility="collapsed")
                st.divider()
                if ref_source:
                    target_diagram_file = f"{dir_path}/target_diagram_{selected_item}_{ref_source}.txt"
                    try:
                        self.__target(target_diagram_file, selected_item, ref_source)
                    except FileNotFoundError:
                        st.error(f'Missing File for {iselected_item} Reference: {ref_source}', icon="⚠")
        elif item == "Portrait_Plot_seasonal":
            st.cache_data.clear()
            col1, col2 = st.columns((1, 2))
            col1.write("##### :green[Please choose!]")
            showing_format = col1.radio("Portrait_Plot_seasonal", ["***Variables***", "***Matrics***"],
                                        captions=["Showing by Variables.", "Showing by Matrics."], index=None, horizontal=False,
                                        label_visibility="collapsed")
            figure_path = os.path.join(case_path, item)
            if showing_format == '***Variables***':
                col21, col22 = col2.columns(2)
                col21.write("##### :green[Select Variables!]")
                iselected_item = col21.radio("Portrait_Plot_seasonal", [i.replace("_", " ") for i in self.selected_items],
                                             index=None,
                                             horizontal=False, key=f'{item}_item', label_visibility="collapsed")
                col22.write("###### :green[Select Matrics or scores!]")
                mm = col22.radio("Portrait_Plot_seasonal", ['metrics', 'scores'], index=None, horizontal=False,
                                 key=f'{item}_score',
                                 label_visibility="collapsed")
                if iselected_item and mm:
                    selected_item = iselected_item.replace(" ", "_")
                    ref_sources = self.ref['general'][f'{selected_item}_ref_source']
                    if isinstance(ref_sources, str): ref_sources = [ref_sources]
                    st.write('##### :blue[Reference sources!]')
                    ref_source = st.radio("Portrait_Plot_seasonal", ref_sources, index=0, horizontal=True,
                                          key=f'{item}_ref_sources',
                                          label_visibility="collapsed")
                    st.divider()
                    try:
                        self.__Portrait_Plot_seasonal_variable(figure_path + "/Portrait_Plot_seasonal.txt",
                                                               selected_item, mm, ref_source, item + '_var')
                    except FileNotFoundError:
                        st.error(f'Missing File for {iselected_item} Reference: {ref_source}')

            elif showing_format == '***Matrics***':
                col2.write("##### :green[Select Matrics or scores!!]")
                iscore = col2.radio("Portrait_Plot_seasonal", [k.replace("_", " ") for k, v in
                                                               dict(chain(self.metrics.items(), self.scores.items())).items()
                                                               if v],
                                    index=None, horizontal=True, key=f'{item}_score', label_visibility="collapsed")
                st.divider()
                if iscore:
                    score = iscore.replace(" ", "_")
                    try:
                        st.write('### :red[Add button switch to next one!]')
                        self.__Portrait_Plot_seasonal_score(figure_path + "/Portrait_Plot_seasonal.txt",
                                                            score, item + '_score')
                    except FileNotFoundError:
                        st.error(f'Missing File for {iscore}')
        elif item == "Parallel_Coordinates":
            st.cache_data.clear()
            col1, col2 = st.columns((1, 2))
            col1.write("##### :green[Please choose!]")
            showing_format = col1.radio(
                "Parallel_Coordinates", ["***Variables***", "***Matrics***"],
                captions=["Showing by Variables.", "Showing by Matrics."], index=None, horizontal=False, key=item,
                label_visibility="collapsed")

            if showing_format == '***Variables***':
                col21, col22 = col2.columns(2)
                col21.write("##### :green[Select Variables!]")
                iselected_item = col21.radio("Parallel_Coordinates", [i.replace("_", " ") for i in self.selected_items],
                                             index=None,
                                             horizontal=False, key=f'{item}_item', label_visibility="collapsed")
                col22.write("###### :green[Select Matrics or scores!]")
                mm = col22.radio("Parallel_Coordinates", ['metrics', 'scores'], index=None, horizontal=False, key=f'{item}_score',
                                 label_visibility="collapsed")

                if iselected_item and mm:
                    selected_item = iselected_item.replace(" ", "_")
                    ref_sources = self.ref['general'][f'{selected_item}_ref_source']
                    if isinstance(ref_sources, str): ref_sources = [ref_sources]
                    st.write('##### :green[Reference sources!]')
                    ref_source = st.radio("Parallel_Coordinates", ref_sources, index=0, horizontal=True,
                                          key=f'{item}_ref_sources',
                                          label_visibility="collapsed")
                    st.divider()
                    try:
                        self.__Parallel_Coordinates_variable(dir_path + "/Parallel_Coordinates_evaluations.txt",
                                                             selected_item, mm, ref_source, item + '_var')
                    except FileNotFoundError:
                        st.error(f'Missing File for {iselected_item} Reference: {ref_source}')

            elif showing_format == '***Matrics***':
                col2.write("##### :green[Select Matrics or scores!!]")
                iscore = col2.radio("###### Matrics and scores!", [k.replace("_", " ") for k, v in
                                                                   dict(chain(self.metrics.items(), self.scores.items())).items()
                                                                   if v],
                                    index=None, horizontal=True, key=f'{item}_score', label_visibility="collapsed")
                st.divider()
                if iscore:
                    score = iscore.replace(" ", "_")
                    try:
                        self.__Parallel_Coordinates_score(dir_path + "/Parallel_Coordinates_evaluations.txt",
                                                          score, item + '_score')
                    except FileNotFoundError:
                        st.error(f'Missing File for {iscore}')
        elif item == "Kernel_Density_Estimate":
            st.cache_data.clear()
            col1, col2 = st.columns((1.5, 2.5))
            col1.write('##### :blue[Select Variables]')
            iselected_item = col1.radio(item, [i.replace("_", " ") for i in self.selected_items], index=None, horizontal=False,
                                        key=f'{item}_item', label_visibility="collapsed")
            col2.write('##### :blue[Select Matrics and scores]')
            imm = col2.radio("Kernel_Density_Estimate",
                             [k.replace("_", " ") for k, v in dict(chain(self.metrics.items(), self.scores.items())).items()
                              if v],
                             index=None, horizontal=True, key=f'{item}_score', label_visibility="collapsed")
            if iselected_item and imm:
                selected_item = iselected_item.replace(" ", "_")
                mm = imm.replace(" ", "_")
                ref_sources = self.ref['general'][f'{selected_item}_ref_source']
                if isinstance(ref_sources, str): ref_sources = [ref_sources]
                st.write('##### :orange[Select your reference!]')
                ref_source = st.radio("Kernel_Density_Estimate", ref_sources, index=0, horizontal=True, key=f'{item}_ref_source',
                                      label_visibility="collapsed")
                # st.divider()
                if ref_source:
                    self.__Kernel_Density_Estimate(f"{self.generals['basedir']}/{self.generals['basename']}", selected_item,
                                                   mm, ref_source)
        elif (item == "Whisker_Plot"):
            st.cache_data.clear()
            col1, col2 = st.columns((1.5, 2.5))
            col1.write('##### :blue[Select Variables]')
            iselected_item = col1.radio(item, [i.replace("_", " ") for i in self.selected_items], index=None, horizontal=False,
                                        key=f'{item}_item', label_visibility="collapsed")
            col2.write('##### :blue[Select Matrics and scores]')
            imm = col2.radio("Whisker_Plot",
                             [k.replace("_", " ") for k, v in dict(chain(self.metrics.items(), self.scores.items())).items()
                              if v],
                             index=None, horizontal=True, key=f'{item}_score', label_visibility="collapsed")
            if iselected_item and imm:
                selected_item = iselected_item.replace(" ", "_")
                mm = imm.replace(" ", "_")
                ref_sources = self.ref['general'][f'{selected_item}_ref_source']
                if isinstance(ref_sources, str): ref_sources = [ref_sources]
                st.write('##### :orange[Select your reference!]')
                ref_source = st.radio("Whisker_Plot", ref_sources, index=0, horizontal=False, key=f'{item}_ref_source',
                                      label_visibility="collapsed")
                st.divider()
                if ref_source:
                    self.__Whisker_Plot(f"{self.generals['basedir']}/{self.generals['basename']}", selected_item, mm,
                                        ref_source)
        elif (item == "Ridgeline_Plot"):
            st.cache_data.clear()
            col1, col2 = st.columns((1.5, 2.5))
            col1.write('##### :blue[Select Variables]')
            iselected_item = col1.radio(item, [i.replace("_", " ") for i in self.selected_items], index=None, horizontal=False,
                                        key=f'{item}_item', label_visibility="collapsed")
            col2.write('##### :blue[Select Matrics and scores]')
            imm = col2.radio("Ridgeline_Plot",
                             [k.replace("_", " ") for k, v in dict(chain(self.metrics.items(), self.scores.items())).items()
                              if v],
                             index=None, horizontal=True, key=f'{item}_score', label_visibility="collapsed")
            if iselected_item and imm:
                selected_item = iselected_item.replace(" ", "_")
                mm = imm.replace(" ", "_")
                ref_sources = self.ref['general'][f'{selected_item}_ref_source']
                if isinstance(ref_sources, str): ref_sources = [ref_sources]
                st.write('##### :orange[Select your reference!]')
                ref_source = st.radio("Ridgeline_Plot", ref_sources, index=0, horizontal=False, key=f'{item}_ref_source',
                                      label_visibility="collapsed")
                st.divider()
                if ref_source:
                    self.__Ridgeline_Plot(f"{self.generals['basedir']}/{self.generals['basename']}", selected_item, mm,
                                          ref_source)
        elif item == "Single_Model_Performance_Index":
            try:
                self.__Single_Model_Performance_Index(dir_path + "/SMPI_comparison.txt", self.selected_items, self.ref, item)
            except FileNotFoundError:
                st.error(f'Missing SMIP', icon="⚠")
        elif item == "Diff_Plot":
            col1, col2, col3 = st.columns((1, 1.2, 2))
            col1.write("##### :green[Please choose!]")
            data_type = col1.radio(
                "Parallel_Coordinates", ["Anomaly", "Difference"], index=None, horizontal=False, key=item,
                label_visibility="collapsed")
            col2.write("##### :green[Please choose!]")
            iselected_item = col2.radio("Parallel_Coordinates", [f'{i.replace("_", " ")}' for i in self.selected_items],
                                        index=None, horizontal=False, key=f'{item}_item', label_visibility="collapsed")
            col3.write("##### :green[Please choose!]")
            mm = col3.radio("metric", [f'{key.replace("_", " ")}' for key, value in self.metrics.items() if value] + [
                f'{key.replace("_", " ")}' for key, value in self.scores.items() if value],
                            index=None, horizontal=True,
                            key=f'{item}_metrics', label_visibility="collapsed")

            st.divider()
            if data_type and iselected_item and mm:
                selected_item = iselected_item.replace(" ", "_")
                mm = mm.replace(" ", "_")
                ref_sources = self.ref['general'][f'{selected_item}_ref_source']
                if isinstance(ref_sources, str): ref_sources = [ref_sources]
                sim_sources = self.sim['general'][f'{selected_item}_sim_source']
                if isinstance(sim_sources, str): sim_sources = [sim_sources]

                if len(sim_sources) > 4 or len(ref_sources) > 4:
                    set_sourese = st.expander("", expanded=True)
                    col1, col2 = set_sourese.columns((1.8, 2.2))
                else:
                    col1, col2 = st.columns((1.8, 2.2))

                col1.write("##### :green[Please choose Reference!]")
                ref_source = col1.radio(
                    "Reference", [i for i in ref_sources],
                    index=None, horizontal=False, key=f'{item}_Reference',
                    label_visibility="collapsed")
                col2.write("##### :green[Please choose!]")
                col21, col22 = col2.columns(2)
                sim_source = col21.radio("sim_sources",
                                         [i for i in sim_sources],
                                         index=None, horizontal=False, key=f'{item}_simulation', label_visibility="collapsed")
                st.divider()
                if ref_source and sim_source:
                    ref_data_type = self.ref[ref_source]['general'][f'data_type']
                    if ref_data_type == 'stn':
                        if data_type == 'Anomaly':
                            file = f'{selected_item}_stn_{ref_source}_sim_{sim_source}_{mm}_anomaly.csv'
                            self.__Diff_Plot(dir_path, file, selected_item, mm, ref_source, sim_source,
                                             data_type, ref_data_type)
                        else:
                            sim_source1 = col22.radio("sim_sources1",
                                                      [i for i in sim_sources if i != sim_source],
                                                      index=None, horizontal=False, key=f'{item}_simulation1',
                                                      label_visibility="collapsed")
                            if sim_source1:
                                sim_varname_1 = self.sim[sim_source][selected_item][f'varname']
                                sim_varname_2 = self.sim[sim_source1][selected_item][f'varname']
                                file = f'{selected_item}_stn_{ref_source}_{sim_source}_{sim_varname_1}_vs_{sim_source1}_{sim_varname_2}_{mm}_diff.csv'
                                if not os.path.exists(os.path.join(dir_path, file)):
                                    file = f'{selected_item}_stn_{ref_source}_{sim_source1}_{sim_varname_2}_vs_{sim_source}_{sim_varname_1}_{mm}_diff.csv'
                                self.__Diff_Plot(dir_path, file, selected_item, mm, ref_source, (sim_source, sim_source1),
                                                 data_type, ref_data_type)
                    else:
                        if data_type == 'Anomaly':
                            file = f'{selected_item}_ref_{ref_source}_sim_{sim_source}_{mm}_anomaly.nc'
                            self.__Diff_Plot(dir_path, file, selected_item, mm, ref_source, sim_source, data_type, ref_data_type)
                        else:
                            sim_source1 = col22.radio("sim_sources1",
                                                      [i for i in sim_sources if i != sim_source],
                                                      index=None, horizontal=False, key=f'{item}_simulation1',
                                                      label_visibility="collapsed")
                            if sim_source1:
                                file = f'{selected_item}_ref_{ref_source}_{sim_source}_vs_{sim_source1}_{mm}_diff.nc'
                                if not os.path.exists(os.path.join(dir_path, file)):
                                    file = f'{selected_item}_ref_{ref_source}_{sim_source1}_vs_{sim_source}_{mm}_diff.nc'
                                self.__Diff_Plot(dir_path, file, selected_item, mm, ref_source, (sim_source, sim_source1),
                                                 data_type, ref_data_type)
        elif item in ['Mean', 'Median', 'Max', 'Min', 'Sum']:
            col1, col2, col3 = st.columns(3)
            col1.write("##### :green[Items!]")
            iselected_item = col1.radio("Basic Plot", [f'{i.replace("_", " ")}' for i in self.selected_items],
                                        index=None, horizontal=False, key=f'{item}_item', label_visibility="collapsed")
            col2.write("##### :green[Please choose!]")
            type = col2.radio("type", ['Reference', 'Simulation'],
                              index=None, horizontal=False, key=f'{item}_type', label_visibility="collapsed")

            if iselected_item and type:
                selected_item = iselected_item.replace(" ", "_")
                if type == 'Reference':
                    itype = 'ref'
                    sources = self.ref['general'][f'{selected_item}_ref_source']
                elif type == 'Simulation':
                    itype = 'sim'
                    sources = self.sim['general'][f'{selected_item}_sim_source']

                col3.write("##### :green[Sources!]")
                source = col3.radio("col3", [source for source in sources],
                                    index=None, horizontal=False, key=f'{item}_source', label_visibility="collapsed")
                st.divider()

                if source:
                    try:
                        data_type = self.ref[source]['general']['data_type']
                    except:
                        data_type = self.sim[source]['general']['data_type']

                    if data_type != 'stn':
                        filenames = glob.glob(
                            os.path.join(dir_path, f'{selected_item}_{itype}_{source}_*_{item}.*'))
                        filenames = [f for f in filenames if f.endswith('.nc')][0]
                        self.__Basic_Plot(dir_path, os.path.basename(filenames), selected_item, source, item, data_type)
                    else:
                        if type == 'Reference':
                            sources1 = self.sim['general'][f'{selected_item}_sim_source']
                        elif type == 'Simulation':
                            sources1 = self.ref['general'][f'{selected_item}_ref_source']

                        source1 = col3.radio("sources1", [source for source in sources1],
                                             index=None, horizontal=False, key=f'{item}_source1',
                                             label_visibility="collapsed")
                        if source1:
                            filenames = f'{selected_item}_stn_{source}_{source1}_{item}.csv'
                            self.__Basic_Plot(dir_path, filenames, selected_item, (source, source1), item, data_type)
        elif item == "Mann_Kendall_Trend_Test":
            col1, col2, col3 = st.columns(3)
            iselected_item = col1.radio("Mann_Kendall_Trend_Test_item",
                                        [f'{i.replace("_", " ")}' for i in self.selected_items],
                                        index=None, horizontal=False, key=f'{item}_item', label_visibility="collapsed")
            type = col2.radio("Mann_Kendall_Trend_Test_type", ['Reference', 'Simulation'],
                              index=None, horizontal=False, key=f'{item}_type', label_visibility="collapsed")

            if iselected_item and type:
                itype = type[:3].lower()
                selected_item = iselected_item.replace(" ", "_")
                try:
                    sources = self.ref['general'][f'{selected_item}_ref_source']
                except:
                    sources = self.sim['general'][f'{selected_item}_sim_source']
                if isinstance(sources, str): sources = [sources]
                source = col3.radio("Mann_Kendall_Trend_Test_source", [source for source in sources],
                                    index=None, horizontal=False, key=f'{item}_source', label_visibility="collapsed")
                st.divider()
                if source:
                    try:
                        data_type = self.ref[source]['general']['data_type']
                        varname = self.ref[source][selected_item]['varname']
                    except:
                        data_type = self.sim[source]['general']['data_type']
                        varname = self.sim[source][selected_item]['varname']

                    if data_type == 'stn':
                        st.info('Function for station data is still on develop!')
                    else:
                        file = f'Mann_Kendall_Trend_Test_{selected_item}_{itype}_{source}_{varname}.nc'
                        try:
                            self.__Mann_Kendall_Trend_Test(dir_path, file, selected_item, source)
                        except:
                            st.error(f'Missing File for Case: {selected_item} {source}', icon="⚠")
        elif item == "Correlation":
            col1, col2 = st.columns((1.8, 2.2))
            iselected_item = col1.radio("Correlation_item", [f'{i.replace("_", " ")}' for i in self.selected_items],
                                        index=None, horizontal=False, key=f'{item}_item', label_visibility="collapsed")
            col21, col22 = col2.columns(2)
            if iselected_item:
                selected_item = iselected_item.replace(" ", "_")
                sim_sources = self.sim['general'][f'{selected_item}_sim_source']
                if isinstance(sim_sources, str): sim_sources = [sim_sources]
                sim_source = col21.radio("Correlation_source", [source for source in sim_sources],
                                         index=None, horizontal=False, key=f'{item}_source', label_visibility="collapsed")

                st.divider()
                if sim_source:
                    sim_source1 = col22.radio("sim_sources1",
                                              [i for i in sim_sources if i != sim_source],
                                              index=None, horizontal=False, key=f'{item}_simulation1',
                                              label_visibility="collapsed")
                    if sim_source1:
                        file = f'Correlation_{selected_item}_{sim_source}_and_{sim_source1}.nc'
                        if not os.path.exists(os.path.join(dir_path, file)):
                            file = f'Correlation_{selected_item}_{sim_source1}_and_{sim_source}.nc'
                        self.__Correlation(dir_path, file, selected_item, (sim_source, sim_source1))
        elif item == "Standard_Deviation":
            st.cache_data.clear()
            col1, col2, col3 = st.columns(3)
            iselected_item = col1.radio("Standard_Deviation_item",
                                        [f'{i.replace("_", " ")}' for i in self.selected_items],
                                        index=None, horizontal=False, key=f'{item}_item', label_visibility="collapsed")
            type = col2.radio("Standard_Deviation_type", ['Reference', 'Simulation'],
                              index=None, horizontal=False, key=f'{item}_type', label_visibility="collapsed")

            if iselected_item and type:
                selected_item = iselected_item.replace("***", "").replace(" ", "_")
                itype = type[:3].lower()
                try:
                    sources = self.ref['general'][f'{selected_item}_ref_source']
                except:
                    sources = self.sim['general'][f'{selected_item}_sim_source']

                if isinstance(sources, str): sources = [sources]
                source = col3.radio("Standard_Deviation_source", [source for source in sources],
                                    index=None, horizontal=False, key=f'{item}_source', label_visibility="collapsed")
                st.divider()
                if source:
                    try:
                        data_type = self.ref[source]['general']['data_type']
                        varname = self.ref[source][selected_item]['varname']
                    except:
                        data_type = self.sim[source]['general']['data_type']
                        varname = self.sim[source][selected_item]['varname']

                    if data_type == 'stn':
                        st.info('Function for station data is still on develop!')
                    else:
                        file = f'Standard_Deviation_{selected_item}_{itype}_{source}_{varname}.nc'
                        self.__Standard_Deviation(dir_path, file, selected_item, source)
        elif item == "Functional_Response":

            col1, col2, col3 = st.columns(3)
            iselected_item = col1.radio("Functional_Response_item",
                                        [f'{i.replace("_", " ")}' for i in self.selected_items],
                                        index=None, horizontal=False, key=f'{item}_item', label_visibility="collapsed")
            if iselected_item:
                selected_item = iselected_item.replace("***", "").replace(" ", "_")
                ref_sources = self.ref['general'][f'{selected_item}_ref_source']
                sim_sources = self.sim['general'][f'{selected_item}_sim_source']
                if isinstance(ref_sources, str): ref_sources = [ref_sources]
                if isinstance(sim_sources, str): sim_sources = [sim_sources]
                ref_source = col2.radio("Functional_Response_refsource", [source for source in ref_sources],
                                        index=None, horizontal=False, key=f'{item}_refsource', label_visibility="collapsed")
                sim_source = col3.radio("Functional_Response_simsource", [source for source in sim_sources],
                                        index=None, horizontal=False, key=f'{item}_simsource', label_visibility="collapsed")
                st.divider()
                if ref_source and sim_source:
                    data_type = self.ref[ref_source]['general']['data_type']
                    if data_type != 'stn':
                        file = f'Functional_Response_{selected_item}_ref_{ref_source}_sim_{sim_source}.nc'
                        self.__Functional_Response(dir_path, file, selected_item, ref_source, sim_source)
                    else:
                        st.info('Function for station data is still on develop!')




        elif item == "Relative_Score":
            st.info(f'Relative_Score not ready yet!', icon="ℹ️")

    def __heatmap(self, dir_path, score):
        st.cache_data.clear()
        make_scenarios_scores_comparison_heat_map(dir_path, score, self.selected_items, self.sim)

    def __heatmap_groupby(self, item, file, selected_item, score, sim_source, ref_source, dir_path):
        st.cache_data.clear()
        make_LC_based_heat_map(item, file, selected_item, score, sim_source, ref_source, dir_path, self.metrics, self.scores)

    def __taylor(self, dir_path, selected_item, ref_source):
        st.cache_data.clear()
        option = {}
        with st.container(height=None, border=True):
            col1, col2, col3, col4 = st.columns((3, 3, 3, 3))
            option['title'] = col1.text_input('Title', value=f'{selected_item.replace("_", " ")}', label_visibility="visible",
                                              key=f"taylor_title")
            option['title_size'] = col2.number_input("Title label size", min_value=0, value=18, key=f"taylor_title_size")
            option['axes_linewidth'] = col3.number_input("axes linewidth", min_value=0, value=1, key=f"taylor_axes_linewidth")
            option['fontsize'] = col4.number_input("font size", min_value=0, value=16, key=f"taylor_fontsize")

            option['STDlabelsize'] = col1.number_input("STD label size", min_value=0, value=16, key=f"taylor_STD_size")
            option['CORlabelsize'] = col2.number_input("COR label size", min_value=0, value=16, key=f"taylor_COR_size")
            option['RMSlabelsize'] = col3.number_input("RMS label size", min_value=0, value=16, key=f"taylor_RMS_size")
            option['Normalized'] = col4.toggle('Normalized', value=True, key=f"taylor_Normalized")

            def get_cases(items, title):
                case_item = {}
                for item in items:
                    case_item[item] = True
                import itertools
                with st.popover(title, use_container_width=True):
                    st.subheader(f"Showing {title}", divider=True)
                    cols = itertools.cycle(st.columns(2))
                    for item in case_item:
                        col = next(cols)
                        case_item[item] = col.checkbox(item, key=f'{item}__taylor',
                                                       value=case_item[item])
                return [item for item, value in case_item.items() if value]

            sim_sources = self.sim['general'][f'{selected_item}_sim_source']
            sim_sources = get_cases(sim_sources, 'cases')
            st.divider()

            with st.expander("Markers setting", expanded=False):
                col1, col2, col3, col4 = st.columns((3, 3, 3, 3))

                col1.write('##### :blue[Ticksize]')
                col2.write('##### :blue[Line style]')
                col3.write('##### :blue[Line width]')
                col4.write('##### :blue[Line color]')

                with col1:
                    option['ticksizeSTD'] = st.number_input("STD", min_value=0, value=14, step=1, key=f"taylor_ticksizeSTD",
                                                            label_visibility='visible')
                    option['ticksizeCOR'] = st.number_input("R", min_value=0, value=14, step=1, key=f"taylor_ticksizeCOR",
                                                            label_visibility='visible')
                    option['ticksizeRMS'] = st.number_input("RMSD", min_value=0, value=14, step=1,
                                                            key=f"taylor_ticksizeRMS", label_visibility='visible')
                    option['markersizeobs'] = st.number_input("Observation marker size", min_value=0, value=10, step=1,
                                                              key=f"taylor_markersizeobs")
                    option['markerobs'] = st.selectbox(f'Observation Marker',
                                                       ['.', 'x', 'o', ">", '<', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X',
                                                        "+", "^", "v"], index=2, placeholder="Choose an option",
                                                       label_visibility="visible")

                with col2:
                    option['styleSTD'] = st.selectbox(f'STD', ['solid', 'dotted', 'dashed', 'dashdot', ':', '-', '--'],
                                                      index=5, placeholder="Choose an option",
                                                      key=f"taylor_styleSTD", label_visibility='visible')
                    option['styleCOR'] = st.selectbox(f'R', ['solid', 'dotted', 'dashed', 'dashdot', ':', '-', '--'],
                                                      index=6, placeholder="Choose an option", label_visibility='visible')
                    option['styleRMS'] = st.selectbox(f'RMSD',
                                                      ['solid', 'dotted', 'dashed', 'dashdot', ':', '-', '--'],
                                                      index=4, placeholder="Choose an option", label_visibility='visible')
                    option['styleOBS'] = st.selectbox(f'Observation',
                                                      ['solid', 'dotted', 'dashed', 'dashdot', ':', '-', '--'],
                                                      index=5, placeholder="Choose an option",
                                                      label_visibility="visible")
                with col3:
                    option['widthSTD'] = st.number_input("STD", min_value=0., value=1., key=f"taylor_widthSTD",
                                                         label_visibility='visible')
                    option['widthCOR'] = st.number_input("R", min_value=0., value=1., key=f"taylor_widthCOR",
                                                         label_visibility='visible')
                    option['widthRMS'] = st.number_input("RMSD", min_value=0., value=2., key=f"taylor_widthRMS",
                                                         label_visibility='visible')
                    option['widthOBS'] = st.number_input("Observation", min_value=0., value=1.)

                with col4:
                    option['colSTD'] = st.text_input("STD", value='k', label_visibility='visible', key=f"taylor_colSTD")
                    option['colCOR'] = st.text_input("R ", value='k', label_visibility='visible', key=f"taylor_colCOR")
                    option['colRMS'] = st.text_input("RMSD ", value='green', label_visibility='visible', key=f"taylor_colRMS")
                    option['colOBS'] = st.text_input("Observation", value='m', label_visibility="visible")

                st.divider()
                stds, RMSs, cors = [], [], []
                df = pd.read_csv(dir_path, sep='\s+', header=0)
                df.set_index('Item', inplace=True)

                stds.append(df['Reference_std'].values[0])
                RMSs.append(np.array(0))
                cors.append(np.array(0))

                import matplotlib.colors as mcolors
                from matplotlib import cm

                hex_colors = ['#4C6EF5', '#F9C74F', '#90BE6D', '#5BC0EB', '#43AA8B', '#F3722C', '#855456', '#F9AFAF',
                              '#F8961E'
                    , '#277DA1', '#5A189A']
                # hex_colors = cm.Set3(np.linspace(0, 1, len(self.sim['general'][f'{selected_item}_sim_source']) + 1))
                colors = itertools.cycle([mcolors.rgb2hex(color) for color in hex_colors])
                symbols = itertools.cycle(["+", ".", "o", "*", "x", "s", "D", "^", "v", ">", "<", "p"])
                markers = {}
                col1, col2, col3, col4 = st.columns((1.8, 2, 2, 2))
                col1.write('##### :blue[Colors]')
                col2.write('##### :blue[Marker]')
                col3.write('##### :blue[Markersize]')
                col4.write('##### :blue[FaceColor]')

                for sim_source in sim_sources:
                    st.write('Case: ', sim_source)
                    col1, col2, col3, col4 = st.columns((1, 2, 2, 2))
                    stds.append(df[f'{sim_source}_std'].values[0])
                    RMSs.append(df[f'{sim_source}_RMS'].values[0])
                    cors.append(df[f'{sim_source}_COR'].values[0])
                    markers[sim_source] = {}
                    with col1:
                        markers[sim_source]['labelColor'] = st.color_picker(f'{sim_source}Colors', value=next(colors),
                                                                            key=None,
                                                                            help=None,
                                                                            on_change=None, args=None, kwargs=None,
                                                                            disabled=False,
                                                                            label_visibility="collapsed")
                        markers[sim_source]['edgeColor'] = markers[sim_source]['labelColor']
                    with col2:
                        Marker = ['.', 'x', 'o', ">", '<', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X', "+", "^", "v"]
                        markers[sim_source]['symbol'] = st.selectbox(f'{sim_source}Marker', Marker,
                                                                     index=Marker.index(next(symbols)),
                                                                     placeholder="Choose an15 option",
                                                                     label_visibility="collapsed")
                    with col3:
                        markers[sim_source]['size'] = st.number_input(f"{sim_source}Markersize", min_value=0, value=10,
                                                                      label_visibility="collapsed")

                    with col4:
                        markers[sim_source]['faceColor'] = st.selectbox(f'{sim_source}FaceColor', ['w', 'b', 'k', 'r', 'none'],
                                                                        index=0, placeholder="Choose an option",
                                                                        label_visibility="collapsed")
                st.info('If you choose Facecolor as "none", then markers will not be padded.')
            option['MARKERS'] = markers

            legend_on = st.toggle('Turn on to set the location of the legend manually', value=False, key=f'taylor_legend_on')
            col1, col2, col3, col4 = st.columns((4, 4, 4, 4))
            if legend_on:
                if len(self.sim['general'][f'{selected_item}_sim_source']) < 6:
                    bbox_to_anchor_x = col1.number_input("X position of legend", value=1.5, step=0.1,
                                                         key=f'taylor_bbox_to_anchor_x')
                    bbox_to_anchor_y = col2.number_input("Y position of legend", value=1., step=0.1,
                                                         key=f'taylor_bbox_to_anchor_y')
                else:
                    bbox_to_anchor_x = col1.number_input("X position of legend", value=1.1, step=0.1,
                                                         key=f'taylor_bbox_to_anchor_x')
                    bbox_to_anchor_y = col2.number_input("Y position of legend", value=0.25, step=0.1,
                                                         key=f'taylor_bbox_to_anchor_y')
            else:
                bbox_to_anchor_x = 1.4
                bbox_to_anchor_y = 1.1
            option['set_legend'] = dict(legend_on=legend_on, bbox_to_anchor_x=bbox_to_anchor_x, bbox_to_anchor_y=bbox_to_anchor_y)
            st.divider()

            col1, col2, col3 = st.columns(3)
            with col1:
                option["x_wise"] = st.number_input(f"X Length", min_value=0, value=6, key=f"taylor_x_wise")
                option['saving_format'] = st.selectbox('Image saving format', ['png', 'jpg', 'eps'],
                                                       index=1, placeholder="Choose an option", label_visibility="visible",
                                                       key=f"taylor_saving_format")
            with col2:
                option["y_wise"] = st.number_input(f"y Length", min_value=0, value=6, key=f"taylor_y_wise")
                option['font'] = st.selectbox('Image saving format',
                                              ['Times new roman', 'Arial', 'Courier New', 'Comic Sans MS', 'Verdana',
                                               'Helvetica',
                                               'Georgia', 'Tahoma', 'Trebuchet MS', 'Lucida Grande'],
                                              index=0, placeholder="Choose an option", label_visibility="visible",
                                              key=f"taylor_font")
            with col3:
                option['dpi'] = st.number_input(f"Figure dpi", min_value=0, value=300, key=f"taylor_dpi")

        if option['Normalized']:
            make_scenarios_comparison_Taylor_Diagram(option, selected_item, np.array(stds) / df['Reference_std'].values[0],
                                                     np.array(RMSs), np.array(cors),
                                                     ref_source,
                                                     self.sim['general'][f'{selected_item}_sim_source'])
        else:
            make_scenarios_comparison_Taylor_Diagram(option, selected_item, np.array(stds), np.array(RMSs), np.array(cors),
                                                     ref_source,
                                                     self.sim['general'][f'{selected_item}_sim_source'])

    def __target(self, dir_path, selected_item, ref_source):
        st.cache_data.clear()
        option = {}
        #
        with (st.container(height=None, border=True)):
            col1, col2, col3, col4 = st.columns((3, 3, 3, 3))

            option['title'] = col1.text_input('Title', value=f'{selected_item.replace("_", " ")}', label_visibility="visible",
                                              key=f"target_title")
            option['title_size'] = col2.number_input("Title label size", min_value=0, value=18, key=f"target_title_size")
            option['axes_linewidth'] = col3.number_input("axes linewidth", min_value=0, value=1, key=f"target_axes_linewidth")
            option['fontsize'] = col4.number_input("font size", min_value=0, value=15, key=f"target_labelsize")
            option['xticksize'] = col1.number_input("X tick size", min_value=0, value=15, key=f"target_xticksize")
            option['yticksize'] = col2.number_input("Y tick size", min_value=0, value=15, key=f"target_yticksize")
            option['Normalized'] = col3.toggle('Normalized', value=True, key=f"target_Normalized")

            def get_cases(items, title):
                case_item = {}
                for item in items:
                    case_item[item] = True
                import itertools
                with st.popover(title, use_container_width=True):
                    st.subheader(f"Showing {title}", divider=True)
                    cols = itertools.cycle(st.columns(2))
                    for item in case_item:
                        col = next(cols)
                        case_item[item] = col.checkbox(item, key=f'{item}__target',
                                                       value=case_item[item])
                return [item for item, value in case_item.items() if value]

            sim_sources = self.sim['general'][f'{selected_item}_sim_source']
            sim_sources = get_cases(sim_sources, 'cases')
            st.divider()

            with st.expander("Markers setting", expanded=False):
                col1, col2, col3, col4 = st.columns((3, 3, 3, 3))
                option['circlelabelsize'] = col1.number_input("circle label size", min_value=0, value=14,
                                                              key=f"target_circletick")
                option['circlestyle'] = col2.selectbox(f'circle Line Style',
                                                       ['solid', 'dotted', 'dashed', 'dashdot', ':', '-', '--', '-.'],
                                                       index=7, placeholder="Choose an option",
                                                       label_visibility="visible", key=f"target_stylecircle")
                option['widthcircle'] = col3.number_input("circle line width", min_value=0., value=1., step=0.1,
                                                          key=f"target_widthcircle")
                option['circlecolor'] = col4.text_input("circle color", value='k', label_visibility="visible",
                                                        key=f"target_colcircle")
                st.divider()
                biases = np.zeros(len(sim_sources))
                rmses = np.zeros(len(sim_sources))
                crmsds = np.zeros(len(sim_sources))
                df = pd.read_csv(dir_path, sep='\s+', header=0)
                df.set_index('Item', inplace=True)

                import matplotlib.colors as mcolors
                import itertools

                hex_colors = ['#4C6EF5', '#F9C74F', '#90BE6D', '#5BC0EB', '#43AA8B', '#F3722C', '#855456', '#F9AFAF',
                              '#F8961E'
                    , '#277DA1', '#5A189A']
                colors = itertools.cycle([mcolors.rgb2hex(color) for color in hex_colors])
                symbols = itertools.cycle(["+", ".", "o", "*", "x", "s", "D", "^", "v", ">", "<", "p"])
                markers = {}

                col1, col2, col3, col4 = st.columns((1.5, 2, 2, 2))
                col1.write('##### :blue[Colors]')
                col2.write('##### :blue[Marker]')
                col3.write('##### :blue[Markersize]')
                col4.write('##### :blue[FaceColor]')

                for i, sim_source in enumerate(sim_sources):
                    st.write('Case: ', sim_source)
                    col1, col2, col3, col4 = st.columns((1, 2, 2, 2))

                    biases[i] = df[f'{sim_source}_bias'].values[0]
                    rmses[i] = df[f'{sim_source}_rmsd'].values[0]
                    crmsds[i] = df[f'{sim_source}_crmsd'].values[0]
                    markers[sim_source] = {}
                    with col1:
                        markers[sim_source]['labelColor'] = st.color_picker(f'{sim_source} colors', value=next(colors),
                                                                            key=f"target_{sim_source}_colors",
                                                                            disabled=False,
                                                                            label_visibility="collapsed")
                        markers[sim_source]['edgeColor'] = markers[sim_source]['labelColor']
                    with col2:
                        Marker = ['.', 'x', 'o', ">", '<', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X', "+", "^", "v"]
                        markers[sim_source]['symbol'] = st.selectbox(f'{sim_source} Marker', Marker,
                                                                     key=f"target_{sim_source}_Marker",
                                                                     index=Marker.index(next(symbols)),
                                                                     placeholder="Choose an option",
                                                                     label_visibility="collapsed")
                    with col3:
                        markers[sim_source]['size'] = st.number_input(f"{sim_source} Markersize", min_value=0, value=10,
                                                                      key=f"target_{sim_source} Markersize",
                                                                      label_visibility="collapsed")

                    with col4:
                        markers[sim_source]['faceColor'] = st.selectbox(f'{sim_source} faceColor', ['w', 'b', 'k', 'r', 'none'],
                                                                        index=0, placeholder="Choose an option",
                                                                        key=f"target_{sim_source} faceColor",
                                                                        label_visibility="collapsed")
            option['MARKERS'] = markers

            legend_on = st.toggle('Turn on to set the location of the legend manually', value=False, key=f'target_legend_on')
            if legend_on:
                col1, col2, col3, col4 = st.columns((4, 4, 4, 4))
                if len(self.sim['general'][f'{selected_item}_sim_source']) < 6:
                    bbox_to_anchor_x = col1.number_input("X position of legend", value=1.5, step=0.1,
                                                         key=f'target_bbox_to_anchor_x')
                    bbox_to_anchor_y = col2.number_input("Y position of legend", value=1., step=0.1,
                                                         key=f'target_bbox_to_anchor_y')
                else:
                    bbox_to_anchor_x = col1.number_input("X position of legend", value=1.1, step=0.1,
                                                         key=f'target_bbox_to_anchor_x')
                    bbox_to_anchor_y = col2.number_input("Y position of legend", value=0.25, step=0.1,
                                                         key=f'target_bbox_to_anchor_y')
            else:
                bbox_to_anchor_x = 1.4
                bbox_to_anchor_y = 1.1
            option['set_legend'] = dict(legend_on=legend_on, bbox_to_anchor_x=bbox_to_anchor_x, bbox_to_anchor_y=bbox_to_anchor_y)
            st.divider()

            col1, col2, col3 = st.columns(3)
            with col1:
                option["x_wise"] = st.number_input(f"X Length", min_value=0, value=6, key=f"target_x_wise")
                option['saving_format'] = st.selectbox('Image saving format', ['png', 'jpg', 'eps'],
                                                       index=1, placeholder="Choose an option", label_visibility="visible",
                                                       key=f"target_saving_format")
            with col2:
                option["y_wise"] = st.number_input(f"y Length", min_value=0, value=6, key=f"target_y_wise")
                option['font'] = st.selectbox('Image saving format',
                                              ['Times new roman', 'Arial', 'Courier New', 'Comic Sans MS', 'Verdana',
                                               'Helvetica',
                                               'Georgia', 'Tahoma', 'Trebuchet MS', 'Lucida Grande'],
                                              index=0, placeholder="Choose an option", label_visibility="visible",
                                              key=f"target_font")
            with col3:
                option['dpi'] = st.number_input(f"Figure dpi", min_value=0, value=300, key=f"target_dpi")

        make_scenarios_comparison_Target_Diagram(option, selected_item, biases, crmsds, rmses,
                                                 ref_source, self.sim['general'][f'{selected_item}_sim_source'])

    def __Kernel_Density_Estimate(self, dir_path, selected_item, score, ref_source):
        st.cache_data.clear()
        make_scenarios_comparison_Kernel_Density_Estimate(dir_path, selected_item, score, ref_source, self.ref, self.sim,
                                                          self.scores)

    def __Whisker_Plot(self, dir_path, selected_item, score, ref_source):
        st.cache_data.clear()
        make_scenarios_comparison_Whisker_Plot(dir_path, selected_item, score, ref_source, self.ref, self.sim, self.scores)

    def __Portrait_Plot_seasonal_variable(self, file, selected_item, score, ref_source, item):
        st.cache_data.clear()
        option = {}

        with st.container(height=None, border=True):
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

            col1, col2, col3 = st.columns((1, 2, 1))
            with col2:
                set_label = st.expander("More info", expanded=False)

            st.divider()

            def get_cases(items, title):
                case_item = {}
                for item in items:
                    case_item[item] = True
                with st.popover(title, use_container_width=True):
                    st.subheader(f"Showing {title}", divider=True)
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

            with st.expander("More information setting", expanded=False):
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
                    option["extend"] = 'neither'
                option["vmin"] = 0
                option["vmax"] = 1
                option["nstep"] = 0.1

                st.write("##### :blue[Legend]")
                col1, col2, col3, col4 = st.columns(4)
                option["legend_box_x"] = col1.number_input(f"x position", value=1.1, step=0.1, key=f"{item}_legend_box_x")
                option["legend_box_y"] = col2.number_input(f"y position", value=1.2, step=0.1, key=f"{item}_legend_box_y")
                option["legend_box_size"] = col3.number_input(f"legend box size", value=1.0, step=0.1,
                                                              key=f"{item}_legend_box_size")
                option["legend_lw"] = col4.number_input(f"Line width", value=1.0, step=0.1, key=f"{item}_legend_lw")
                option["legend_fontsize"] = col1.number_input(f"Box fontsize", value=12.5, step=0.2,
                                                              key=f"{item}_legend_fontsize")

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

        if items is not None and cases is not None:
            if score == 'metrics':
                make_scenarios_comparison_Portrait_Plot_seasonal_metrics(option, file, selected_item, ref_source, items, cases,
                                                                         score)
            else:
                make_scenarios_comparison_Portrait_Plot_seasonal(option, file, selected_item, ref_source, items, cases,
                                                                 score)
        elif not items:
            st.error('Metircs items is None!')
        elif not cases:
            st.error('Simulation cases is None!')

    def __Portrait_Plot_seasonal_score(self, file, score, item):
        st.cache_data.clear()
        option = {}

        with st.container(height=None, border=True):
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

            st.divider()

            def get_cases(items, title):
                case_item = {}
                for item in items:
                    case_item[item] = True
                with st.popover(title, use_container_width=True):
                    st.subheader(f"Showing {title}", divider=True)
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

            with st.expander("More info", expanded=False):
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

        # make_scenarios_comparison_Portrait_Plot_seasonal_by_score(option, file, score)

        if items and cases:
            make_scenarios_comparison_Portrait_Plot_seasonal_by_score(option, file, score, items, cases)
        elif not items:
            st.error('Metircs items is None!')
        elif not cases:
            st.error('Simulation cases is None!')

    def __Parallel_Coordinates_variable(self, file, selected_item, score, ref_source, item):
        st.cache_data.clear()

        option = {}

        with st.container(height=None, border=True):
            col1, col2, col3, col4 = st.columns((3.5, 3, 3, 3))
            with col1:
                option['title'] = st.text_input('Title', value=f'', label_visibility="visible",
                                                key=f"{item} title")
                option['title_size'] = st.number_input("Title label size", min_value=0, value=20, key=f"{item}_title_size")

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
                option['fontsize'] = st.number_input("Font size", min_value=0, value=15, step=1, key=f"{item}_fontsize")

            def get_cases(items, title):
                case_item = {}
                for item in items:
                    case_item[item] = True
                with st.popover(title, use_container_width=True):
                    st.subheader(f"Showing {title}", divider=True)
                    if title != 'cases':
                        for item in case_item:
                            case_item[item] = st.checkbox(item.replace("_", " "), key=f"{item}__Parallel_Coordinates_variable",
                                                          value=case_item[item])
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

            st.divider()

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

            with st.expander("More information", expanded=False):
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

            st.divider()

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

        if items and cases:
            make_scenarios_comparison_parallel_coordinates(option, file, selected_item, ref_source, items, cases, score)
        elif not items:
            st.error('Metircs items is None!')
        elif not cases:
            st.error('Simulation cases is None!')

    def __Parallel_Coordinates_score(self, file, score, item):
        st.cache_data.clear()
        option = {}
        with st.container(height=None, border=True):

            option['title'] = st.text_input('Title',
                                            value=f"Parallel Coordinates Plot - {score.replace('_', ' ')}",
                                            label_visibility="visible",
                                            key=f"{item} title")
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

            st.divider()

            def get_cases(items, title):
                case_item = {}
                for item in items:
                    case_item[item] = True
                with st.popover(title, use_container_width=True):
                    st.subheader(f"Showing {title}", divider=True)
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
            cases = list(
                set([value for key in self.selected_items for value in self.sim['general'][f"{key}_sim_source"] if value]))
            col1, col2 = st.columns(2)
            with col1:
                items = get_cases(items, 'Selected items')
            with col2:
                cases = get_cases(cases, 'cases')

            with st.expander("More info", expanded=False):
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

            st.divider()

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
        if items and cases:
            make_scenarios_comparison_parallel_coordinates_by_score(option, file, score, items, cases)
        elif not items:
            st.error('Metircs items is None!')
        elif not cases:
            st.error('Simulation cases is None!')

    def __Single_Model_Performance_Index(self, file, selected_items, ref, item):
        st.cache_data.clear()
        make_scenarios_comparison_Single_Model_Performance_Index(file, selected_items, ref, item)

    def __Ridgeline_Plot(self, dir_path, selected_item, score, ref_source):
        st.cache_data.clear()
        make_scenarios_comparison_Ridgeline_Plot(dir_path, selected_item, score, ref_source, self.ref, self.sim, self.scores)

    def __Diff_Plot(self, dir_path, file, selected_item, score, ref_source, sim_source, showing_format, ref_data_type):
        st.cache_data.clear()
        option = {}
        ref_unit = self.ref[ref_source][selected_item]['varunit']
        make_scenarios_comparison_Diff_Plot(dir_path, file, selected_item, score, ref_source, sim_source, showing_format,
                                            ref_data_type, ref_unit, option)

    def __Basic_Plot(self, dir_path, file, selected_item, source, item, data_type):
        st.cache_data.clear()
        option = {}
        if data_type != 'stn':
            try:
                unit = self.ref[source][selected_item]['varunit']
            except:
                unit = self.sim[source][selected_item]['varunit']
        else:
            try:
                unit = self.ref[source[0]][selected_item]['varunit']
            except:
                unit = self.sim[source[0]][selected_item]['varunit']

        make_Basic_Plot(dir_path, file, selected_item, source, item, unit, data_type, option)

    def __Mann_Kendall_Trend_Test(self, dir_path, file, selected_item, source):
        st.cache_data.clear()
        option = {}
        figure_nml = self.nl.read_namelist(st.session_state['generals']['figure_nml'])
        item_nml = self.nl.read_namelist(figure_nml['comparison_nml']['Mann_Kendall_Trend_Test_source'])
        option['significance_level'] = item_nml['general']['significance_level']
        del figure_nml, item_nml
        make_Mann_Kendall_Trend_Test_Plot(dir_path, file, selected_item, source, option)

    def __Correlation(self, dir_path, file, selected_item, sources):
        st.cache_data.clear()
        option = {}
        make_Correlation_Plot(dir_path, file, selected_item, sources, option)

    def __Standard_Deviation(self, dir_path, file, selected_item, source):
        st.cache_data.clear()
        option = {}
        make_Standard_Deviation_Plot(dir_path, file, selected_item, source, option)

    def __Functional_Response(self, dir_path, file, selected_item,ref_source, sim_source):
        st.cache_data.clear()
        option = {}
        make_Functional_Response_Plot(dir_path, file, selected_item, ref_source, sim_source, option)
