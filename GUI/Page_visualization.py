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

        if showing_item:
            if len(showing_item) > 8:
                item = st.pills('showing', [k.replace("_", " ") for k in showing_item], selection_mode="single", default=None, label_visibility="collapsed")
            else:
                item = st.radio('showing_item', [k.replace("_", " ") for k in showing_item], index=None, horizontal=True,
                                label_visibility='collapsed')
            if item:
                self.__step5_make_show_tab(case_path, visual_select, item.replace(" ", "_"))

    def __step5_make_show_tab(self, case_path, visual_select, item):
        @st.cache_data
        def load_image(path):
            image = Image.open(path)
            return image

        if (visual_select == "Metrics") | (visual_select == "Scores"):
            st.cache_data.clear()
            st.divider()
            st.write('##### :orange[Evaluation Items]')
            selected_item = st.radio('selected_items', [k.replace("_", " ") for k in self.selected_items], index=None,
                                     horizontal=True, key=f'{item}_item', label_visibility='collapsed')

            if selected_item:
                st.divider()
                selected_item = selected_item.replace(" ", "_")
                sim_sources = self.sim['general'][f'{selected_item}_sim_source']
                ref_sources = self.ref['general'][f'{selected_item}_ref_source']
                if isinstance(sim_sources, str): sim_sources = [sim_sources]
                if isinstance(ref_sources, str): ref_sources = [ref_sources]
                for ref_source in ref_sources:
                    for sim_source in sim_sources:
                        if (self.ref[ref_source]['general'][f'data_type'] != 'stn') & (self.sim[sim_source]['general'][f'data_type'] != 'stn'):
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
            st.divider()
            figure_path = str(os.path.join(case_path, visual_select.lower(), item))

            if item == "HeatMap":
                st.cache_data.clear()
                st.write('#### :blue[Select Scores!]')
                iscore = st.radio("HeatMap", [k.replace("_", " ") for k, v in self.scores.items() if v],
                                  index=None, horizontal=True, key=f'{item}', label_visibility='collapsed')
                st.divider()
                if iscore:
                    score = iscore.replace(' ', '_')
                    filename = glob.glob(os.path.join(figure_path, f'scenarios_{score}_comparison_heatmap.*'))
                    filename = [f for f in filename if not f.endswith('.txt')][0]
                    if os.path.exists(filename):
                        image = load_image(filename)
                        st.image(image, caption=f'Scores: {iscore}', use_container_width=True)
                    else:
                        st.error(f'Missing Figure for Scores: {iscore}', icon="⚠")

            elif (item == "Taylor_Diagram") | (item == "Target_Diagram"):
                st.cache_data.clear()
                st.write('##### :blue[Select Variables]')
                selected_item = st.radio(item, [i.replace("_", " ") for i in self.selected_items], index=None,
                                         horizontal=True, key=f'{item}', label_visibility="collapsed")
                st.divider()
                if selected_item:
                    selected_item = selected_item.replace(" ", "_")
                    ref_sources = self.ref['general'][f'{selected_item}_ref_source']
                    if isinstance(ref_sources, str): ref_sources = [ref_sources]
                    for ref_source in ref_sources:
                        filename = glob.glob(os.path.join(figure_path, f'*_{selected_item}_{ref_source}.*'))
                        filename = [f for f in filename if not f.endswith('.txt')][0]
                        if os.path.exists(filename):
                            image = load_image(filename)
                            st.image(image, caption=f'Reference: {ref_source}', use_container_width=True)
                        else:
                            st.error(f'Missing Figure for {selected_item.replace("_", " ")} Reference: {ref_source}', icon="⚠")

            elif item == "Portrait_Plot_seasonal":
                st.cache_data.clear()
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
                    df = pd.read_csv(figure_path + "/Portrait_Plot_seasonal.txt", sep=r'\s+', header=0)
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
                    df = pd.read_csv(figure_path + "/Parallel_Coordinates_evaluations.txt", sep=r'\s+', header=0)
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
                st.cache_data.clear()
                col1, col2 = st.columns((1.5, 2.5))
                col1.write('##### :blue[Select Variables]')
                iselected_item = col1.radio(item, [i.replace("_", " ") for i in self.selected_items], index=None,
                                            horizontal=False, key=f'{item}_item', label_visibility="collapsed")
                col2.write('##### :blue[Select Matrics and scores]')
                imm = col2.radio(item, [k.replace("_", " ") for k, v in dict(chain(self.metrics.items(), self.scores.items())).items() if v],
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
                st.cache_data.clear()
                filename = glob.glob(os.path.join(figure_path, f'SMPI_comparison_plot_comprehensive.*'))
                try:
                    image = load_image(filename[0])
                    st.image(image, caption='SMIP', use_container_width="auto")
                except:
                    st.error(f'Missing Figure for SMIP', icon="⚠")

            elif item == "Relative_Score":
                st.cache_data.clear()
                st.write('##### :orange[Evaluation Items]')
                selected_item = st.radio('selected_items', [k.replace("_", " ") for k in self.selected_items], index=None,
                                         horizontal=True, key=f'{item}_item', label_visibility='collapsed')

                if selected_item:
                    st.divider()
                    selected_item = selected_item.replace(" ", "_")
                    sim_sources = self.sim['general'][f'{selected_item}_sim_source']
                    ref_sources = self.ref['general'][f'{selected_item}_ref_source']
                    if isinstance(sim_sources, str): sim_sources = [sim_sources]
                    if isinstance(ref_sources, str): ref_sources = [ref_sources]
                    for ref_source in ref_sources:
                        for sim_source in sim_sources:
                            if (self.ref[ref_source]['general'][f'data_type'] != 'stn') & (self.sim[sim_source]['general'][f'data_type'] != 'stn'):
                                filenames = glob.glob(os.path.join(figure_path, f'{selected_item}_ref_{ref_source}_sim_{sim_source}_Relative*'))
                                filtered_list = [f for f in filenames if not f.endswith('.nc')]
                                for filename in filtered_list:
                                    try:
                                        image = load_image(filename)
                                        relative_part = filename.split("Relative")[-1].split(".")[0]
                                        st.image(image, caption=f'Reference: {ref_source}, Simulation: {sim_source} Relative:{relative_part}',
                                                 use_container_width=True)
                                    except:
                                        st.error(f'Missing Figure for Reference: {ref_source}, Simulation: {sim_source} Relative:{relative_part}', icon="⚠")
                            else:
                                filenames = glob.glob(os.path.join(figure_path, f'{selected_item}_stn_{ref_source}_{sim_source}_relative_*'))
                                filtered_list = [f for f in filenames if not f.endswith('.csv')]
                                for filename in filtered_list:
                                    try:
                                        image = load_image(filename)
                                        relative_part = filename.split("relative")[-1].split(".")[0]
                                        st.image(image, caption=f'Reference: {ref_source}, Simulation: {sim_source} Relative:{relative_part}',
                                                 use_container_width=True)
                                    except:
                                        st.error(f'Missing Figure for Reference: {ref_source}, Simulation: {sim_source} Relative:{relative_part}', icon="⚠")


            elif item == "Diff_Plot":
                st.cache_data.clear()
                col1, col2, col3 = st.columns((1, 1.2, 2))
                col1.write("##### :green[Showing Format!]")
                showing_format = col1.radio(
                    "Diff_Plot", ["***Anomaly***", "***Differentiate***"], index=None, horizontal=False, key=item,
                    label_visibility="collapsed")

                col2.write("##### :green[Evaluation Item!]")
                iselected_item = col2.radio("Parallel_Coordinates", [f'***{i.replace("_", " ")}***' for i in self.selected_items],
                                            index=None, horizontal=False, key=f'{item}_item', label_visibility="collapsed")

                col3.write("##### :green[Choose metric and scores!]")
                mm = col3.radio("metric", [f'***{k.replace("_", " ")}***' for k, v in dict(chain(self.metrics.items(), self.scores.items())).items() if v],
                                index=None, horizontal=True, key=f'{item}_metrics', label_visibility="collapsed")

                if showing_format and iselected_item and mm:
                    st.divider()
                    mm = mm.replace("***", "").replace(" ", "_")
                    selected_item = iselected_item.replace("***", "").replace(" ", "_")
                    ref_sources = self.ref['general'][f'{selected_item}_ref_source']
                    sim_sources = self.sim['general'][f'{selected_item}_sim_source']
                    if isinstance(ref_sources, str): ref_sources = [ref_sources]
                    if isinstance(sim_sources, str): sim_sources = [sim_sources]

                    if showing_format == '***Anomaly***':
                        for ref_source in ref_sources:
                            ref_data_type = self.ref[ref_source]['general']['data_type']
                            for sim_source in sim_sources:
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
                                    st.image(image, caption=f'File: {selected_item} {ref_source} {sim_source} {mm}', use_container_width="auto")
                                except:
                                    st.error(f'Missing Figure for File: {selected_item} {ref_source} {sim_source} {mm}', icon="⚠")
                    else:
                        for ref_source in ref_sources:
                            ref_data_type = self.ref[ref_source]['general']['data_type']
                            for i, sim1 in enumerate(sim_sources):
                                for j, sim2 in enumerate(sim_sources[i + 1:], i + 1):
                                    if ref_data_type != 'stn':
                                        file = f'{selected_item}_ref_{ref_source}_{sim1}_vs_{sim2}_{mm}_diff.*'
                                        filenames = glob.glob(os.path.join(figure_path, file))
                                        if len(filenames) == 0:
                                            file = f'{selected_item}_ref_{ref_source}_{sim2}_vs_{sim1}_{mm}_diff.*'
                                            filenames = glob.glob(os.path.join(figure_path, file))
                                        filenames = [f for f in filenames if not f.endswith('.nc')]
                                    else:
                                        sim_varname_1 = self.sim[sim_source][selected_item][f'varname']
                                        sim_varname_2 = self.sim[sim_source1][selected_item][f'varname']
                                        file = f"{selected_item}_stn_{ref_source}_{sim1}_{sim_varname_1}_vs_{sim2}_{sim_varname_2}_{mm}_diff.*"
                                        filenames = glob.glob(os.path.join(figure_path, file))
                                        if len(filenames) == 0:
                                            file = f"{selected_item}_stn_{ref_source}_{sim2}_{sim_varname_2}_vs_{sim1}_{sim_varname_1}_{mm}_diff.*"
                                            filenames = glob.glob(os.path.join(figure_path, file))
                                        filenames = [f for f in filenames if not f.endswith('.csv')]

                                    for filename in filenames:
                                        try:
                                            image = load_image(filename)
                                            st.image(image, caption=f'File: {selected_item} {ref_source} {sim1} vs {sim2} {mm} diff',
                                                     use_container_width="auto")
                                        except:
                                            st.error(f'Missing Figure for File: {selected_item} {ref_source} {sim1} vs {sim2} {mm} diff',
                                                     icon="⚠")

                                    if len(filenames) == 0:
                                        st.error(f'Missing Figure for File: {selected_item} {ref_source} {sim1} vs {sim2} {mm} diff',
                                                 icon="⚠")

            elif item in ['Mean', 'Median', 'Max', 'Min', 'Sum']:
                st.cache_data.clear()
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
                            if source1:
                                filenames = glob.glob(
                                    os.path.join(figure_path, f'{selected_item}_stn*{source}*{item}_*.*'))
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
                st.cache_data.clear()
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

                    show_type = col3.radio("Mann_Kendall_Trend_Test_source", ['Trend', 'tau'],
                                           index=None, horizontal=False, key=f'{item}_source', label_visibility="collapsed")
                    st.divider()
                    if show_type:
                        for source in sources:
                            if type == 'Reference':
                                data_type = self.ref[source]['general']['data_type']
                            elif type == 'Simulation':
                                data_type = self.sim[source]['general']['data_type']
                            if data_type == 'stn':
                                st.info('Function for station data is still on develop!')
                            else:
                                filenames = glob.glob(
                                    os.path.join(figure_path, f'Mann_Kendall_Trend_Test_{selected_item}_{itype}_{source}*{show_type}.*'))
                                for filename in filenames:
                                    if os.path.exists(filename):
                                        image = load_image(filename)
                                        st.image(image, caption=f'Case: {selected_item} {source} {show_type}', use_container_width=True)
                                    else:
                                        st.error(f'Missing Figure for Case: {selected_item} {source}', icon="⚠")

            elif item == "Correlation":
                st.cache_data.clear()
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
                st.cache_data.clear()
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

                    st.divider()
                    for source in sources:
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
                st.cache_data.clear()
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
                    st.divider()
                    for ref_source in ref_sources:
                        for sim_source in sim_sources:
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
            ref_sources = self.ref['general'][f'{selected_item}_ref_source']
            sim_sources = self.sim['general'][f'{selected_item}_sim_source']
            if isinstance(ref_sources, str): ref_sources = [ref_sources]
            if isinstance(sim_sources, str): sim_sources = [sim_sources]
            if len(ref_sources) > 6 or len(sim_sources) > 6:
                container = st.expander('Sources', expanded=True)
                col1, col2 = container.columns(2)
            else:
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

                return_refselect = st.radio("###### > Reference", ref_sources,
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
                return_simselect = st.radio("###### > Simulation", sim_sources,
                                            index=sim_index, horizontal=False, label_visibility='collapsed')

            if (return_refselect is not None) & (return_simselect is not None):
                st.divider()
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
        st.cache_data.clear()
        left, right = st.columns((3, 5))
        with left:
            st.write('##### :orange[Please choose your type]')
            plot_type = st.selectbox('Please choose your type',
                                     ['Geo metrics replot', 'Time average', 'Other Functions'],
                                     index=None, placeholder="Choose an option", label_visibility="collapsed")
        if plot_type == 'Time average':
            make_geo_time_average(selected_item, refselect, simselect, os.path.join(path, 'data'), self.ref, self.sim)
        elif plot_type == 'Compare lines':
            self.__generate_image_geo_Compare_lines(selected_item, path)
        elif plot_type == 'Geo metrics replot':
            with right:
                st.write('##### :orange[Showing metrics or scores]')
                select_to_plot = st.selectbox('Metrics', [k for k, v in dict(chain(self.metrics.items(), self.scores.items())).items() if v],
                                              placeholder="Choose an option", label_visibility="collapsed",
                                              key='_geo_geo_Site_metrics_replot')

            if select_to_plot in self.metrics:
                mm = 'metrics'
            elif select_to_plot in self.scores:
                mm = 'scores'
            make_geo_plot_index(mm, select_to_plot, selected_item, refselect, simselect, path)
        elif plot_type == 'Other Functions':
            st.info('Some functions are not perfect yet, Coming soon...', icon="ℹ️")

    def _geo_stn_visual(self, selected_item, refselect, simselect, path):
        st.cache_data.clear()
        left, right = st.columns((3, 5))
        with left:
            st.write('##### :orange[Please choose your type]')
            plot_type = st.selectbox('Please choose your type', ['Site metrics replot', 'Site replot', 'Other Functions'],
                                     index=None, placeholder="Choose an option", label_visibility="collapsed")
        if plot_type == 'Site replot':
            showing_stn_data(self, right, selected_item, refselect, simselect, path, ('grid', 'stn'))
        elif plot_type == 'Other Functions':
            st.info('Some functions are not perfect yet, Coming soon...', icon="ℹ️")

    def _stn_geo_visual(self, selected_item, refselect, simselect, path):
        st.cache_data.clear()
        # streamlit # hist plot 可以做
        left, right = st.columns((3, 5))
        with left:
            st.write('##### :orange[Please choose your type]')
            plot_type = st.selectbox('Please choose your type', ['Site metrics replot', 'Site replot', 'Other Functions'],
                                     index=None, placeholder="Choose an option", label_visibility="collapsed")
        if plot_type == 'Site replot':
            showing_stn_data(self, right, selected_item, refselect, simselect, path, ('stn', 'grid'))
        elif plot_type == 'Site metrics replot':
            with right:
                select_to_plot = st.selectbox('Metrics',
                                              [k for k, v in dict(chain(self.metrics.items(), self.scores.items())).items() if
                                               v], placeholder="Choose an option", label_visibility="visible", key='_stn_geo_Site_metrics_replot')
                if select_to_plot in self.metrics:
                    mm = 'metrics'
                elif select_to_plot in self.scores:
                    mm = 'scores'
            make_stn_plot_index(mm, select_to_plot, selected_item, refselect, simselect, path)


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
        st.cache_data.clear()
        left, right = st.columns((3, 5))
        with left:
            st.write('##### :orange[Please choose your type]')
            plot_type = st.selectbox('Please choose your type', ['Site metrics replot', 'Site replot', 'Other Functions'],
                                     index=None, placeholder="Choose an option", label_visibility="collapsed")
        if plot_type == 'Site replot':
            showing_stn_data(self, right, selected_item, refselect, simselect, path, ('stn', 'stn'))
        elif plot_type == 'Hist-plot':
            with right:
                select_to_plot = st.multiselect('Metrics',
                                                [k for k, v in dict(chain(self.metrics.items(), self.scores.items())).items() if v],
                                                placeholder="Choose an option", label_visibility="visible")
        elif plot_type == 'Other Functions':
            st.info('Some functions are not perfect yet, Coming soon...', icon="ℹ️")

    def __generate_image_geo_Compare_lines(self, selected_item, path):
        st.cache_data.clear()
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

        option['data_path'] = path + f'/dataset/'
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
        self.output = os.path.join(self.generals['basedir'], self.generals['basename'], "output")
        self.comparisons_path = os.path.join(self.generals['basedir'], self.generals['basename'], "output", "comparisons")
        self.data_path = os.path.join(self.generals['basedir'], self.generals['basename'], "output", "data")
        self.metrics_path = os.path.join(self.generals['basedir'], self.generals['basename'], "output", "metrics")
        self.scores_path = os.path.join(self.generals['basedir'], self.generals['basename'], "output", "scores")

    # -=========================================================
    def Comparison_replot(self):
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
                    self._prepare(item)

    def _prepare(self, item):
        dir_path = os.path.join(self.comparisons_path, item)
        if (item == "IGBP_groupby") | (item == "PFT_groupby"):
            Comparison_item = 'heatmap_groupby'
        elif item in ['Mean', 'Median', 'Max', 'Min', 'Sum']:
            Comparison_item = 'Basic_Plot'
        else:
            Comparison_item = item
        getattr(self, f"{Comparison_item}")(dir_path, item)

    def heatmap_groupby(self, dir_path, item):
        st.cache_data.clear()
        col1, col2 = st.columns(2)
        col1.write('##### :green[Select Variables!]')
        iselected_item = col1.radio("###### Variables!", [i.replace("_", " ") for i in self.selected_items], index=None,
                                    horizontal=False, label_visibility="collapsed", key=f'{item}_Variables')
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
            if (self.ref[ref_source]['general'][f'data_type'] != 'stn') & (self.sim[sim_source]['general'][f'data_type'] != 'stn'):
                heatmap_groupby_file = os.path.join(str(self.output), mm, item, f'{sim_source}___{ref_source}',
                                                    f"{selected_item}_{sim_source}___{ref_source}_{mm}.txt")
                try:
                    make_LC_based_heat_map(item, heatmap_groupby_file, selected_item, mm, sim_source, ref_source, self.metrics, self.scores)
                except FileNotFoundError:
                    st.error(f'Missing File for Reference: {ref_source} Simulation: {sim_source}', icon="⚠")
            else:
                st.info(
                    f'Reference: {ref_source}, Simulation: {sim_source}---Heatmap groupby is not supported for station data!',
                    icon="👋")

    def HeatMap(self, dir_path, item):
        st.cache_data.clear()
        st.write('#### Select Scores to replot')
        iscore = st.radio("HeatMap", [k.replace("_", " ") for k, v in self.scores.items() if v],
                          index=None, horizontal=True, key=f'{item}', label_visibility="collapsed")
        if iscore:
            st.divider()
            score = iscore.replace(" ", "_")
            heatmap_file = f"{dir_path}/scenarios_{score}_comparison.txt"
            try:
                make_scenarios_scores_comparison_heat_map(heatmap_file, score, self.selected_items, self.sim)
            except FileNotFoundError:
                st.error(f'Missing File for Score: {iscore}', icon="⚠")

    def Taylor_Diagram(self, dir_path, item):
        st.cache_data.clear()
        col1, col2 = st.columns(2)
        col1.write('##### :blue[Select Variables]')
        iselected_item = col1.radio("Taylor_Diagram", [i.replace("_", " ") for i in self.selected_items], index=None,
                                    horizontal=False, key=f'{item}', label_visibility="collapsed")
        if iselected_item:
            selected_item = iselected_item.replace(" ", "_")
            ref_sources = self.ref['general'][f'{selected_item}_ref_source']
            if isinstance(ref_sources, str): ref_sources = [ref_sources]
            col2.write('##### :green[Select your reference!]')
            ref_source = col2.radio("Taylor_Diagram", ref_sources, index=0, horizontal=False, key=f'{item}_ref_source',
                                    label_visibility="collapsed")
            st.divider()
            if ref_source:
                taylor_diagram_file = f"{dir_path}/taylor_diagram_{selected_item}_{ref_source}.txt"
                try:
                    make_scenarios_comparison_Taylor_Diagram(self, taylor_diagram_file, selected_item, ref_source)
                except FileNotFoundError:
                    st.error(f'Missing File for {iselected_item} Reference: {ref_source}', icon="⚠")

    def Target_Diagram(self, dir_path, item):
        col1, col2 = st.columns(2)
        col1.write('##### :blue[Select Variables]')
        iselected_item = col1.radio("Target_Diagram", [i.replace("_", " ") for i in self.selected_items], index=None,
                                    horizontal=False, key=f'{item}', label_visibility="collapsed")
        if iselected_item:
            selected_item = iselected_item.replace(" ", "_")
            ref_sources = self.ref['general'][f'{selected_item}_ref_source']
            if isinstance(ref_sources, str): ref_sources = [ref_sources]
            col2.write('##### :green[Select your reference!]')
            ref_source = col2.radio("Target_Diagram", ref_sources, index=0, horizontal=False, key=f'{item}_ref_source',
                                    label_visibility="collapsed")
            st.divider()
            if ref_source:
                target_diagram_file = f"{dir_path}/target_diagram_{selected_item}_{ref_source}.txt"
                try:
                    make_scenarios_comparison_Target_Diagram(self, target_diagram_file, selected_item, ref_source)
                except FileNotFoundError:
                    st.error(f'Missing File for {iselected_item} Reference: {ref_source}', icon="⚠")

    def Kernel_Density_Estimate(self, dir_path, item):
        st.cache_data.clear()
        col1, col2 = st.columns((1.5, 2.5))
        col1.write('##### :blue[Select Variables]')
        iselected_item = col1.radio(item, [i.replace("_", " ") for i in self.selected_items], index=None, horizontal=False,
                                    key=f'{item}_item', label_visibility="collapsed")
        col2.write('##### :blue[Select Matrics and scores]')
        imm = col2.radio("Kernel_Density_Estimate", [k.replace("_", " ") for k, v in dict(chain(self.metrics.items(), self.scores.items())).items()
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
            if ref_source:
                st.divider()
                make_scenarios_comparison_Kernel_Density_Estimate(f"{self.generals['basedir']}/{self.generals['basename']}", selected_item,
                                                                  mm, ref_source, self)

    def Whisker_Plot(self, dir_path, item):
        st.cache_data.clear()
        col1, col2 = st.columns((1.5, 2.5))
        col1.write('##### :blue[Select Variables]')
        iselected_item = col1.radio(item, [i.replace("_", " ") for i in self.selected_items], index=None, horizontal=False,
                                    key=f'{item}_item', label_visibility="collapsed")
        col2.write('##### :blue[Select Matrics and scores]')
        imm = col2.radio("Whisker_Plot",
                         [k.replace("_", " ") for k, v in dict(chain(self.metrics.items(), self.scores.items())).items()
                          if v], index=None, horizontal=True, key=f'{item}_score', label_visibility="collapsed")
        if iselected_item and imm:
            selected_item = iselected_item.replace(" ", "_")
            mm = imm.replace(" ", "_")
            ref_sources = self.ref['general'][f'{selected_item}_ref_source']
            if isinstance(ref_sources, str): ref_sources = [ref_sources]
            st.write('##### :orange[Select your reference!]')
            ref_source = st.radio("Whisker_Plot", ref_sources, index=0, horizontal=False, key=f'{item}_ref_source',
                                  label_visibility="collapsed")
            if ref_source:
                st.divider()
                make_scenarios_comparison_Whisker_Plot(f"{self.generals['basedir']}/{self.generals['basename']}", selected_item, mm, ref_source, self)

    def Parallel_Coordinates(self, dir_path, item):
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
                                         index=None, horizontal=False, key=f'{item}_item', label_visibility="collapsed")
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
                    make_scenarios_comparison_parallel_coordinates(self, dir_path + "/Parallel_Coordinates_evaluations.txt", selected_item, mm, ref_source,
                                                                   item + '_var')
                except FileNotFoundError:
                    st.error(f'Missing File for {iselected_item} Reference: {ref_source}')

        elif showing_format == '***Matrics***':
            col2.write("##### :green[Select Matrics or scores!!]")
            iscore = col2.radio("###### Matrics and scores!", [k.replace("_", " ") for k, v in
                                                               dict(chain(self.metrics.items(), self.scores.items())).items() if v],
                                index=None, horizontal=True, key=f'{item}_score', label_visibility="collapsed")
            st.divider()
            if iscore:
                score = iscore.replace(" ", "_")
                try:
                    make_scenarios_comparison_parallel_coordinates_by_score(self, dir_path + "/Parallel_Coordinates_evaluations.txt", score, item + '_score')
                except FileNotFoundError:
                    st.error(f'Missing File for {iscore}')

    def Portrait_Plot_seasonal(self, dir_path, item):
        col1, col2 = st.columns((1, 2))
        col1.write("##### :green[Please choose!]")
        showing_format = col1.radio("Portrait_Plot_seasonal", ["***Variables***", "***Matrics***"],
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
                    make_scenarios_comparison_Portrait_Plot_seasonal(self, dir_path + "/Portrait_Plot_seasonal.txt",
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
                    make_scenarios_comparison_Portrait_Plot_seasonal_by_score(self, dir_path + "/Portrait_Plot_seasonal.txt",
                                                                              score, item + '_score')
                except FileNotFoundError:
                    st.error(f'Missing File for {iscore}')

    def Single_Model_Performance_Index(self, dir_path, item):
        st.cache_data.clear()
        make_scenarios_comparison_Single_Model_Performance_Index(dir_path, self.selected_items, self.ref, item)

    def Ridgeline_Plot(self, dir_path, item):
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
                make_scenarios_comparison_Ridgeline_Plot(f"{self.generals['basedir']}/{self.generals['basename']}", selected_item, mm,
                                                         ref_source, self)

    def Diff_Plot(self, dir_path, item):
        st.cache_data.clear()
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
                ref_unit = self.ref[ref_source][selected_item]['varunit']
                ref_data_type = self.ref[ref_source]['general'][f'data_type']
                if ref_data_type == 'stn':
                    if data_type == 'Anomaly':
                        file = f'{selected_item}_stn_{ref_source}_sim_{sim_source}_{mm}_anomaly.csv'
                        option = {}
                        make_scenarios_comparison_Diff_Plot(dir_path, file, selected_item, mm, ref_source, sim_source, data_type,
                                                            ref_data_type, ref_unit, option)
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
                            option = {}
                            make_scenarios_comparison_Diff_Plot(dir_path, file, selected_item, mm, ref_source, (sim_source, sim_source1), data_type,
                                                                ref_data_type, ref_unit, option)
                else:
                    if data_type == 'Anomaly':
                        file = f'{selected_item}_ref_{ref_source}_sim_{sim_source}_{mm}_anomaly.nc'
                        option = {}
                        make_scenarios_comparison_Diff_Plot(dir_path, file, selected_item, mm, ref_source, sim_source, data_type,
                                                            ref_data_type, ref_unit, option)
                    else:
                        sim_source1 = col22.radio("sim_sources1",
                                                  [i for i in sim_sources if i != sim_source],
                                                  index=None, horizontal=False, key=f'{item}_simulation1',
                                                  label_visibility="collapsed")
                        if sim_source1:
                            file = f'{selected_item}_ref_{ref_source}_{sim_source}_vs_{sim_source1}_{mm}_diff.nc'
                            if not os.path.exists(os.path.join(dir_path, file)):
                                file = f'{selected_item}_ref_{ref_source}_{sim_source1}_vs_{sim_source}_{mm}_diff.nc'
                            option = {}
                            make_scenarios_comparison_Diff_Plot(dir_path, file, selected_item, mm, ref_source, (sim_source, sim_source1), data_type,
                                                                ref_data_type, ref_unit, option)

    def Basic_Plot(self, dir_path, item):
        st.cache_data.clear()
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
                    filenames = [f for f in filenames if f.endswith('.nc')]
                    if len(filenames) == 1:
                        make_Basic_Plot(dir_path, os.path.basename(filenames[0]), selected_item, source, item, data_type, self)
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
                        make_Basic_Plot(dir_path, filenames, selected_item, (source, source1), item, data_type, self)

    def Mann_Kendall_Trend_Test(self, dir_path, item):
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
                        make_Mann_Kendall_Trend_Test_Plot(dir_path, file, selected_item, source, self)
                    except:
                        st.error(f'Missing File for Case: {selected_item} {source}', icon="⚠")

    def Correlation(self, dir_path, item):
        st.cache_data.clear()
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
                    make_Correlation_Plot(dir_path, file, selected_item, (sim_source, sim_source1))

    def Standard_Deviation(self, dir_path, item):
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
            if type == 'Reference':
                sources = self.ref['general'][f'{selected_item}_ref_source']
            else:
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
                    make_Standard_Deviation_Plot(dir_path, file, selected_item, source)

    def Functional_Response(self, dir_path, item):
        st.cache_data.clear()
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
            if ref_source and sim_source:
                st.divider()
                data_type = self.ref[ref_source]['general']['data_type']
                if data_type != 'stn':
                    file = f'Functional_Response_{selected_item}_ref_{ref_source}_sim_{sim_source}.nc'
                    make_Functional_Response_Plot(dir_path, file, selected_item, ref_source, sim_source)
                else:
                    st.info('Function for station data is still on develop!')

    def Relative_Score(self, dir_path, item):
        st.cache_data.clear()
        st.write('##### :orange[Evaluation Items]')
        selected_item = st.radio('selected_items', [k.replace("_", " ") for k in self.selected_items], index=None,
                                 horizontal=True, key=f'{item}_item', label_visibility='collapsed')
        if selected_item:
            st.divider()
            col1, col2, col3 = st.columns(3)
            selected_item = selected_item.replace(" ", "_")
            sim_sources = self.sim['general'][f'{selected_item}_sim_source']
            ref_sources = self.ref['general'][f'{selected_item}_ref_source']
            if isinstance(sim_sources, str): sim_sources = [sim_sources]
            if isinstance(ref_sources, str): ref_sources = [ref_sources]
            col1.write('##### :orange[Reference]')
            col2.write('##### :orange[Simulation]')
            col3.write('##### :orange[Scores]')
            ref_source = col1.radio("Relative_Score_refsource", [source for source in ref_sources],
                                    index=None, horizontal=False, key=f'{item}_ref_source', label_visibility="collapsed")
            sim_source = col2.radio("Relative_Score_simsource", [source for source in sim_sources],
                                    index=None, horizontal=False, key=f'{item}_sim_source', label_visibility="collapsed")
            iscore = col3.radio("Relative_Score_score", [key.replace("_", " ") for key, value in self.scores.items() if value],
                                index=None, horizontal=False, key=f'{item}_score', label_visibility="collapsed")
            if ref_source and sim_source and iscore:
                score = iscore.replace(" ", "_")
                ref_type = self.ref[ref_source]['general'][f'data_type']
                sim_type = self.sim[sim_source]['general'][f'data_type']
                if (ref_type != 'stn') & (sim_type != 'stn'):
                    try:
                        filename = f'{selected_item}_ref_{ref_source}_sim_{sim_source}_Relative{score}.nc'
                        make_Relative_Score_Plot(dir_path, filename, selected_item, ref_source, sim_source, score, 'grid')
                    except:
                        st.error(f'Missing Figure for Reference: {ref_source}, Simulation: {sim_source} Relative:{score}', icon="⚠")
                else:
                    try:
                        filename = f'{selected_item}_stn_{ref_source}_{sim_source}_relative_scores.csv'
                        make_Relative_Score_Plot(dir_path, filename, selected_item, ref_source, sim_source, score, 'stn')
                    except:
                        st.error(f'Missing Figure for Reference: {ref_source}, Simulation: {sim_source} Relative:{score}', icon="⚠")
