import os
import glob
import math
import streamlit as st
from PIL import Image
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
import matplotlib
from io import BytesIO
from streamlit_option_menu import option_menu
import itertools
from itertools import chain
from Namelist_lib.namelist_read import NamelistReader, GeneralInfoReader, UpdateNamelist, UpdateFigNamelist
from Muti_function_lib import ref_lines, sim_lines, each_line
from Muti_function_lib import make_stn_plot_index, geo_Compare_lines
from Muti_function_lib import geo_single_average, geo_average_diff
from Muti_function_lib import make_geo_plot_index

from Comparison_figlib.Fig_portrait_plot_seasonal import make_scenarios_comparison_Portrait_Plot_seasonal
from Comparison_figlib.Fig_portrait_plot_seasonal import make_scenarios_comparison_Portrait_Plot_seasonal_metrics
from Comparison_figlib.Fig_portrait_plot_seasonal import make_scenarios_comparison_Portrait_Plot_seasonal_by_score
from Comparison_figlib.Fig_heatmap import make_scenarios_scores_comparison_heat_map
from Comparison_figlib.Fig_heatmap import make_LC_based_heat_map
from Comparison_figlib.Fig_parallel_coordinates import make_scenarios_comparison_parallel_coordinates
from Comparison_figlib.Fig_parallel_coordinates import make_scenarios_comparison_parallel_coordinates_by_score
from Comparison_figlib.Fig_taylor_diagram import make_scenarios_comparison_Taylor_Diagram
from Comparison_figlib.Fig_target_diagram import make_scenarios_comparison_Target_Diagram
from Comparison_figlib.Fig_kernel_density_estimate import make_scenarios_comparison_Kernel_Density_Estimate
from Comparison_figlib.Fig_Whisker_Plot import make_scenarios_comparison_Whisker_Plot
from Comparison_figlib.Fig_Single_Model_Performance_Index import make_scenarios_comparison_Single_Model_Performance_Index
from Comparison_figlib.Fig_Ridgeline_Plot import make_scenarios_comparison_Ridgeline_Plot

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

        visual_select = option_menu(None, ["Metrics", "Scores", "Comparisons"], #, 'Statistics'
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
                                         use_column_width=True)
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
                                             use_column_width=True)
                                except:
                                    st.error(f'Missing Figure for Reference: {ref_source}, Simulation: {sim_source}', icon="⚠")
                            else:
                                st.info(
                                    f'Reference: {ref_source}, Simulation: {sim_source}---Heatmap groupby is not supported for station data!',
                                    icon="👋")

        elif visual_select == "Comparisons":
            st.cache_data.clear()
            st.divider()
            figure_path = os.path.join(case_path, visual_select.lower(), item)

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
                        st.image(image, caption=f'Scores: {iscore}', use_column_width=True)
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
                            st.image(image, caption=f'Reference: {ref_source}', use_column_width=True)
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
                                st.image(image, caption=f'Reference: {ref_source}', use_column_width="auto")
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
                                st.image(image, caption=f'Reference: {", ".join(item_combination)}', use_column_width="auto")
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
                                st.image(image, caption=f'Reference: {ref_source.replace(" ", "_")}', use_column_width="auto")
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
                                st.image(image, caption=f'References: {", ".join(item_combination)}', use_column_width="auto")
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
                            st.image(image, caption=ffname, use_column_width="auto")
                        except:
                            if mm == 'nSpatialScore':
                                st.info(f'{mm} is not supported for {item.replace("_", " ")}!', icon="ℹ️")
                            else:
                                st.error(f'Missing Figure for {ffname}', icon="⚠")

            elif item == "Single_Model_Performance_Index":
                filename = glob.glob(os.path.join(figure_path, f'SMPI_comparison_plot_comprehensive.*'))
                try:
                    image = load_image(filename[0])
                    st.image(image, caption='SMIP', use_column_width="auto")
                except:
                    st.error(f'Missing Figure for SMIP', icon="⚠")

            elif item == "Relative_Score":
                st.info(f'Relative_Score not ready yet!', icon="ℹ️")

        elif visual_select == "Statistics":
            st.info(f'Statistics not ready yet!', icon="ℹ️")


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

        selected_item = st.radio(
            "#### What's your choice?", [selected_item.replace("_", " ") for selected_item in self.selected_items], index=None,
            horizontal=True)

        if selected_item:
            selected_item = selected_item.replace(" ", "_")
            col1, col2 = st.columns(2)
            with col1:
                if len(self.ref['general'][f'{selected_item}_ref_source']) == 1:
                    ref_index = 0
                else:
                    ref_index = None
                return_refselect = st.radio("###### > Reference", self.ref['general'][f'{selected_item}_ref_source'],
                                            index=ref_index, horizontal=False)
            with col2:
                if len(self.sim['general'][f'{selected_item}_sim_source']) == 1:
                    sim_index = 0
                else:
                    sim_index = None
                return_simselect = st.radio("###### > Simulation", self.sim['general'][f'{selected_item}_sim_source'],
                                            index=sim_index, horizontal=False)
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
        #             exit()
        # def _get_options(option: dict, **kwargs) -> dict:
        #         st.divider()
        # TODO: 这个部分还没做好，我记得之前看过有多级选择的，再找一找
        # TODO: 初步预想，做一个近似于panoply的页面，如果是区域的，那么可以进行时间平均和区域平均两种，上传数据，添加经纬度某点选择
        # TODO: 站点的话选择性稍小一些，但是可以根据timeres，进一步平均？？？？
        # TODO: 及直接上传所选站点数据，因为做数据之前是需要有excle表格的，所以可以先上传，根据df选择。。。。需要可视化的站点
        # TODO: 多个模式之间进行对比（ref）：区分站点与全球；站点：站点个数选择，不要超过。。。。；全球：区域平均结果

        # st.write(st.session_state.item_checkbox)
        # st.json(st.session_state.visual_item, expanded=False)
        # st.info('Some functions are not perfect yet, Coming soon...', icon="ℹ️")

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

                        # ms = visual_item['metrics'] + visual_item['scores']
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

                        st.line_chart(chart_data, x='Time', y=["Simulation", "Reference"], color=["#FF0000", "#0000FF"])
                        del ref_data, sim_data, chart_data

                    except FileNotFoundError:
                        st.error(f"{id} File not found. Please check the file path.")
            st.button('Submitted', on_click=Submit_button)
        if st.session_state['Submit_button']:
            showing_items = chart_df[chart_df['Select']]
            self.__generate_image_stn_lines(showing_items, selected_item, refselect, simselect,
                                            path, (ref_var, sim_var), (ref_unit, sim_unit))

        # st.write('1、 添加一键清除按钮')
        # st.write('2、 展示全部数据序列？？可能比较耗内存就是')
        # st.write('3、 添加该站点的质量')
        # st.write('4、 或许可以根据站点评估指标添加一个直观的打分系统，就像zetero的那种')
        # st.write('5、 .......')

    # ------------------------------------------------------
    def __generate_image_histplot(self, visual_item, select_to_plot, df):
        with st.expander("Edited", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                title = st.text_input('Title', value=visual_item['ref'], label_visibility="visible")
                title_size = st.number_input("Title label size", min_value=0, value=20)

            with col2:
                xtick = st.number_input("xtick label size", min_value=0, value=17)
                grid = st.toggle("Showing grid", value=False, label_visibility="visible")

            with col3:
                #
                ytick = st.number_input("ytick label size", min_value=0, value=17)
                if grid:
                    grid_style = st.selectbox('Grid Line Style', ['solid', 'dotted', 'dashed', 'dashdot'],
                                              index=2, placeholder="Choose an option", label_visibility="visible")

        fig, ax = plt.subplots(1, figsize=(15, 7))
        params = {'backend': 'ps',
                  'axes.labelsize': 20,
                  'grid.linewidth': 0.2,
                  'font.size': 20,
                  'legend.fontsize': 20,
                  'legend.frameon': False,
                  'xtick.labelsize': xtick,
                  'xtick.direction': 'out',
                  'ytick.labelsize': ytick,
                  'ytick.direction': 'out',
                  'savefig.bbox': 'tight',
                  'axes.unicode_minus': False,
                  'text.usetex': False}
        rcParams.update(params)

        # labels = select_to_plot
        # colors_list = sns.color_palette(f"Set3", n_colors=len(labels), desat=.7).as_hex()
        # nbegin = 0
        # nend = 1
        # if ('KGE' in select_to_plot) | ('KGESS' in select_to_plot) | ('correlation' in select_to_plot) | (
        #         'NSE' in select_to_plot) | ('kappa_coeff' in select_to_plot):
        #     nbegin = -1
        # if ('RMSE' in select_to_plot) | ('apb' in select_to_plot) | ('ubRMSE' in select_to_plot) | ('mae' in select_to_plot):
        #     nend = 2
        #
        # llist1 = ['NSE', 'KGE', 'KGESS', 'kappa_coeff']
        # llist2 = ['apb', 'RMSE', 'ubRMSE', 'mae', ]
        # llist3 = ['pc_bias', 'bias', 'L']
        # st.write(df.head())
        # st.write(sns.load_dataset('penguins'))
        # for metric in select_to_plot:
        #     # data = np.delete(df[metric], np.argwhere(np.isnan(df[metric])), axis=0)
        #     # sns.histplot(data, bins=np.arange(nbegin, nend, 0.05), element="bars", stat="probability", common_norm=True,
        #     #              kde=True,warn_singular=True,
        #     #              color=colors_list[1], ax=ax, edgecolor='k', linewidth=1.5,
        #     #              line_kws={'color': colors_list[1], 'linestyle': '--', 'linewidth': 3.5,
        #     #                        'label': f"{metric} KDE"})  # label={file},
        #     sns.displot(df[metric], kind='kde', warn_singular=True)
        #
        # ax.legend(shadow=False, frameon=False, fontsize=20, loc='best')
        # ax.set_title(title, fontsize=title_size)
        # ax.set_xlabel("Time", fontsize=xtick + 1)
        # # ax.set_ylabel(f"{selected_item} [{sim_setting[f'{sim_case}_varunit']}]", fontsize=ytick + 1)
        # if grid:
        #     ax.grid(True, linestyle='--', linewidth=0.3, color='grey', alpha=0.5)  # 绘制图中虚线 透明度0.3
        # st.pyplot(fig)
        # ax4.grid(True, linestyle='--', linewidth=0.3, color='grey', alpha=0.5)
        # ax4.set_ylabel('Density', fontsize=30)
        # ax4.set_xlabel(f'KGESS', fontsize=30)
        # ax4.set_xticks(np.arange(-1, 1.2, 0.2))
        #
        # ax4.legend(fontsize=30, loc='best')  # 绘制表示框，右下角绘制
        # plt.tight_layout()
        # plt.savefig(f"{pathout}/fig2_kde.png", dpi=300)
        # plt.savefig(f"{pathout}/fig2_kde.pdf", dpi=300)
        # plt.savefig(f"{pathout}/fig2_kde.eps", dpi=300)
        # plt.savefig(f"{pathout}/fig2_kde.tif", dpi=300)
        # plt.close(fig4)
        return

    def __generate_image_geo_time_average(self, selected_item, refselect, simselect, path):
        option = {}

        with st.container(border=True):

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                option['title'] = st.text_input('Title', label_visibility="visible")
                option['title_size'] = st.number_input("Title label size", min_value=0, value=20)
            with col2:
                option['xticklabel'] = st.text_input('X tick labels', value='Longitude', label_visibility="visible")
                option['xtick'] = st.number_input("xtick label size", min_value=0, value=17)
            with col3:
                option['yticklabel'] = st.text_input('Y tick labels', value='Latitude', label_visibility="visible")
                option['ytick'] = st.number_input("ytick label size", min_value=0, value=17)
            with col4:
                option['fontsize'] = st.number_input("Font size", min_value=0, value=17)
                option['axes_linewidth'] = st.number_input("axes linewidth", min_value=0, value=1)

            st.divider()
            col1, col2 = st.columns((2, 1))
            with col1:
                option['plot_type'] = st.radio('Please choose your type',
                                               ['Simulation average', 'Reference average', 'Differentiate'],
                                               index=None, label_visibility="visible", horizontal=True)

            with st.expander('More info', expanded=True):
                col1, col2, col3 = st.columns((1.5, 1, 1))
                option['grid'] = col1.toggle("Showing grid?", value=False, label_visibility="visible")
                if option['grid']:
                    option['grid_style'] = col2.selectbox('Grid Line Style', ['solid', 'dotted', 'dashed', 'dashdot'],
                                                          index=2, placeholder="Choose an option", label_visibility="visible")
                    option['grid_linewidth'] = col3.number_input("grid linewidth", min_value=0, value=1)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    option['max_lat'] = col1.number_input("Max latitude: ", value=float(st.session_state['generals']["max_lat"]),
                                                          key="geo_time_average_max_lat",
                                                          min_value=-90.0, max_value=90.0)
                    option['min_lat'] = col2.number_input("Min latitude: ", value=float(st.session_state['generals']["min_lat"]),
                                                          key="geo_time_average_min_lat",
                                                          min_value=-90.0, max_value=90.0)
                    option['max_lon'] = col3.number_input("Max Longitude: ", value=float(st.session_state['generals']["max_lon"]),
                                                          key="geo_time_average_max_lon",
                                                          min_value=-180.0, max_value=180.0)
                    option['min_lon'] = col4.number_input("Min Longitude: ", value=float(st.session_state['generals']["min_lon"]),
                                                          key="geo_time_average_min_lon",
                                                          min_value=-180.0, max_value=180.0)

                def get_ticks(vmin, vmax):
                    if 2 >= vmax - vmin > 1:
                        colorbar_ticks = 0.2
                    elif 5 >= vmax - vmin > 2:
                        colorbar_ticks = 0.5
                    elif 10 >= vmax - vmin > 5:
                        colorbar_ticks = 1
                    elif 100 >= vmax - vmin > 10:
                        colorbar_ticks = 5
                    elif 100 >= vmax - vmin > 50:
                        colorbar_ticks = 20
                    elif 200 >= vmax - vmin > 100:
                        colorbar_ticks = 20
                    elif 500 >= vmax - vmin > 200:
                        colorbar_ticks = 50
                    elif 1000 >= vmax - vmin > 500:
                        colorbar_ticks = 100
                    elif 2000 >= vmax - vmin > 1000:
                        colorbar_ticks = 200
                    elif 10000 >= vmax - vmin > 2000:
                        colorbar_ticks = 10 ** math.floor(math.log10(vmax - vmin)) / 2
                    else:
                        colorbar_ticks = 0.10
                    return colorbar_ticks

                ref_type, sim_type = ('grid', 'grid')
                ref_var = self.ref[refselect][selected_item][f"varname"]
                try:
                    sim_var = self.sim[simselect][selected_item][f"varname"]
                except:
                    nml = self.nl.read_namelist(self.sim[simselect]['general']['model_namelist'])
                    sim_var = nml[selected_item][f"varname"]

                option['data_path'] = path + f'/data/'
                key_value = 'geo_time_average'


                import math
                if option['plot_type'] == 'Differentiate':
                    sim_vmin_max_on = False
                    sim_error = False
                    try:
                        var = sim_var
                        filename = f'{option["data_path"]}/{selected_item}_sim_{simselect}_{sim_var}.nc'
                        if len(option['title']) == 0:
                            option['title'] = 'Simulation'
                        ds = xr.open_dataset(filename)
                        ds_sim = ds[var].mean('time', skipna=True)
                        sim_vmin = math.floor(np.nanmin(ds_sim))
                        sim_vmax = math.floor(np.nanmax(ds_sim))
                    except Exception as e:
                        st.error(f"Error: {e}")
                        sim_error = True

                    if not sim_error:
                        col1, col2, col3 = st.columns((4, 2, 2))
                        option["sim_vmin_max_on"] = col1.toggle('Setting Simulation max min', value=sim_vmin_max_on,
                                                                key=f"{key_value}sim_vmin_max_on")
                        if option["sim_vmin_max_on"]:
                            try:
                                option["sim_vmin"] = col2.number_input(f"colorbar min", value=sim_vmin)
                                option["sim_vmax"] = col3.number_input(f"colorbar max", value=sim_vmax)
                            except ValueError:
                                st.error(f"Max value must larger than min value.")
                        else:
                            option["sim_vmin"] = sim_vmin
                            option["sim_vmax"] = sim_vmax
                        sim_colorbar_ticks = get_ticks(option["sim_vmin"], option["sim_vmax"])
                    else:
                        sim_colorbar_ticks = 0.5

                    ref_vmin_max_on = False
                    ref_error = False
                    try:
                        var = ref_var
                        filename = f'{option["data_path"]}/{selected_item}_ref_{refselect}_{ref_var}.nc'
                        if len(option['title']) == 0:
                            option['title'] = 'Reference'
                        ds = xr.open_dataset(filename)

                        ds_ref = ds[var].mean('time', skipna=True)
                        ref_vmin = math.floor(np.nanmin(ds_ref))
                        ref_vmax = math.floor(np.nanmax(ds_ref))
                    except Exception as e:
                        st.error(f"Error: {e}")
                        ref_error = True

                    if not ref_error:
                        col1, col2, col3 = st.columns((4, 2, 2))
                        option["ref_vmin_max_on"] = col1.toggle('Setting Reference max min', value=ref_vmin_max_on,
                                                                key=f"{key_value}ref_vmin_max_on")
                        if option["ref_vmin_max_on"]:
                            try:
                                option["ref_vmin"] = col2.number_input(f"colorbar min", value=ref_vmin)
                                option["ref_vmax"] = col3.number_input(f"colorbar max", value=ref_vmax)
                            except ValueError:
                                st.error(f"Max value must larger than min value.")
                        else:
                            option["ref_vmin"] = ref_vmin
                            option["ref_vmax"] = ref_vmax
                        ref_colorbar_ticks = get_ticks(option["ref_vmin"], option["ref_vmax"])
                    else:
                        ref_colorbar_ticks = 0.5

                    diff_vmin_max_on = False
                    diff_error = False
                    try:
                        diff = ds_ref - ds_sim
                        diff_vmin = math.floor(np.nanmin(diff))
                        diff_vmax = math.floor(np.nanmax(diff))
                    except Exception as e:
                        st.error(f"Error: {e}")
                        ref_error = True
                    except Exception as e:
                        st.error(f"Error: {e}")
                        diff_error = True

                    if not diff_error:
                        col1, col2, col3 = st.columns((4, 2, 2))
                        option["diff_vmin_max_on"] = col1.toggle('Setting Difference max min', value=ref_vmin_max_on,
                                                                key=f"{key_value}diff_vmin_max_on")
                        if option["diff_vmin_max_on"]:
                            try:
                                option["diff_vmin"] = col2.number_input(f"colorbar min", value=diff_vmin)
                                option["diff_vmax"] = col3.number_input(f"colorbar max", value=diff_vmax)
                            except ValueError:
                                st.error(f"Max value must larger than min value.")
                        else:
                            option["diff_vmin"] = diff_vmin
                            option["diff_vmax"] = diff_vmax
                        diff_colorbar_ticks = get_ticks(option["diff_vmin"], option["diff_vmax"])
                    else:
                        diff_colorbar_ticks = 0.5

                    st.write('##### :blue[Colorbar Ticks locater]')
                    col1, col2, col3 = st.columns((3, 3, 3))
                    option["sim_colorbar_ticks"] = col1.number_input(f"Simulation", value=float(sim_colorbar_ticks), step=0.1)
                    option["ref_colorbar_ticks"] = col2.number_input(f"Reference", value=float(ref_colorbar_ticks), step=0.1)
                    option["diff_colorbar_ticks"] = col3.number_input(f"Difference", value=float(diff_colorbar_ticks), step=0.1)
                elif option['plot_type'] == 'Simulation average' or option['plot_type'] == 'Reference average':
                    vmin_max_on = False
                    error = False
                    try:
                        if option['plot_type'] == 'Simulation average':
                            var = sim_var
                            filename = f'{option["data_path"]}/{selected_item}_sim_{simselect}_{sim_var}.nc'
                            if len(option['title']) == 0:
                                option['title'] = 'Simulation'

                            ds = xr.open_dataset(filename)
                            ds = ds[var].mean('time', skipna=True)
                            vmin = math.floor(np.nanmin(ds))
                            vmax = math.floor(np.nanmax(ds))
                        elif option['plot_type'] == 'Reference average':
                            var = ref_var
                            filename = f'{option["data_path"]}/{selected_item}_ref_{refselect}_{ref_var}.nc'
                            if len(option['title']) == 0:
                                option['title'] = 'Reference'
                            ds = xr.open_dataset(filename)

                            ds = ds[var].mean('time', skipna=True)
                            vmin = math.floor(np.nanmin(ds))
                            vmax = math.floor(np.nanmax(ds))
                    except Exception as e:
                        st.error(f"Error: {e}")
                        error = True

                    if not error:
                        col1, col2, col3 = st.columns(3)
                        option["vmin_max_on"] = col1.toggle('Setting max min', value=vmin_max_on, key=f"{key_value}vmin_max_on")
                        if option["vmin_max_on"]:
                            try:
                                option["vmin"] = col2.number_input(f"colorbar min", value=vmin)
                                option["vmax"] = col3.number_input(f"colorbar max", value=vmax)
                            except ValueError:
                                st.error(f"Max value must larger than min value.")
                        else:
                            option["vmin"] = vmin
                            option["vmax"] = vmax
                        colorbar_ticks = get_ticks(option["vmin"], option["vmax"])
                    else:
                        colorbar_ticks = 0.5
                st.divider()


                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    option['cpool'] = st.selectbox('Colorbar',
                                                   ['RdYlGn', 'RdYlGn_r', 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG',
                                                    'BrBG_r', 'BuGn', 'BuGn_r',
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
                                                    'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm',
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
                                                   label_visibility="visible")
                with col2:
                    option["colorbar_position"] = st.selectbox('colorbar position', ['horizontal', 'vertical'],  # 'Season',
                                                               index=0, placeholder="Choose an option",
                                                               label_visibility="visible")

                with col3:
                    option["extend"] = st.selectbox(f"colorbar extend", ['neither', 'both', 'min', 'max'],
                                                    index=0, placeholder="Choose an option", label_visibility="visible",
                                                    key=f"geo_time_average_extend")
                if option['plot_type'] == 'Simulation average' or option['plot_type'] == 'Reference average':
                    with col4:
                        option["colorbar_ticks"] = st.number_input(f"Colorbar Ticks locater", value=float(colorbar_ticks), step=0.1)

            col1, col2, col3 = st.columns(3)
            option["x_wise"] = col1.number_input(f"X Length", min_value=0, value=13)
            option["y_wise"] = col2.number_input(f"y Length", min_value=0, value=7)

            option["hspace"] = col1.number_input(f"hspace", min_value=0., max_value=1.0, value=0.45, step=0.1)
            option["wspace"] = col2.number_input(f"wspace", min_value=0., max_value=1.0, value=0.25, step=0.1, )

            option['saving_format'] = col3.selectbox('Image saving format', ['png', 'jpg', 'eps'],
                                                     index=1, placeholder="Choose an option", label_visibility="visible")

        if option['plot_type'] == 'Simulation average' or option['plot_type'] == 'Reference average':
            geo_single_average(option, selected_item, refselect, simselect, self.ref, self.sim, var, filename)
        elif option['plot_type'] == 'Differentiate':
            geo_average_diff(option, selected_item, refselect, simselect, self.ref, self.sim, ref_var, sim_var)
        else:
            st.error('please choose first!')


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

    def __generate_image_stn_lines(self, showing_items, selected_item, refselect, simselect, path, vars, unit):
        option = {}
        with st.container(border=True):

            col1, col2, col3 = st.columns(3)
            with col1:
                option['title'] = st.text_input('Title', label_visibility="visible")
                option['title_size'] = st.number_input("Title label size", min_value=0, value=20)

            with col2:
                option['xticksize'] = st.number_input("xtick size", min_value=0, value=17)
                option['fontsize'] = st.number_input("Font size", min_value=0, value=17)

            with col3:
                option['yticksize'] = st.number_input("ytick size", min_value=0, value=17)

            col1, col2 = st.columns((2, 1.5))
            option['grid'] = col1.toggle("Showing grid?", value=False, label_visibility="visible")
            if option['grid']:
                option['grid_style'] = col2.selectbox('Grid Line Style', ['solid', 'dotted', 'dashed', 'dashdot'],
                                                      index=2, placeholder="Choose an option", label_visibility="visible")

            st.write('##### :orange[Replot Type]')
            col1, col2 = st.columns((2, 1.5))
            option['plot_type'] = col1.radio('Please choose your type', ['sim lines', 'ref lines', 'each site'],
                                             index=None, label_visibility="collapsed", horizontal=True)
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

            if 'resample_disable' not in st.session_state:
                st.session_state.resample_disable = False
            if 'groubly_disable' not in st.session_state:
                st.session_state.groubly_disable = False
            if ('resample_option' not in st.session_state and 'groubly_option' not in st.session_state) or (
                    st.session_state['resample_option'] is None and st.session_state['groubly_option'] is None):
                st.session_state.groubly_disable = False
                st.session_state.resample_disable = False

            def data_editor_change(key, editor_key):
                st.write(st.session_state[key])
                if key == 'resample_option' and st.session_state[key] is not None:
                    st.session_state.groubly_disable = True
                    st.session_state.resample_disable = False
                elif key == 'groubly_option' and st.session_state[key] is not None:
                    st.session_state.groubly_disable = False
                    st.session_state.resample_disable = True
                else:
                    st.session_state.groubly_disable = False
                    st.session_state.resample_disable = False

            col1, col2, col3 = st.columns(3)
            with col1:
                option["x_wise"] = st.number_input(f"X Length", min_value=0, value=17)
            with col2:
                option["y_wise"] = st.number_input(f"y Length", min_value=0, value=6)
            with col3:
                option['saving_format'] = st.selectbox('Image saving format', ['png', 'jpg', 'eps'],
                                                       index=1, placeholder="Choose an option", label_visibility="visible")
            #     option["resample_option"] = st.selectbox('Mean option', ['Year', 'Month', 'Day'], key='resample_option',
            #                                              on_change=data_editor_change,
            #                                              disabled=st.session_state.resample_disable,
            #                                              args=('resample_option', 'resample_option'),  # 'Season',
            #                                              index=None, placeholder="Choose an option",
            #                                              label_visibility="visible")
            #
            # with col4:
            #     option["groubly_option"] = st.selectbox('Groubly option', ['Year', 'Month', 'Day'], key='groubly_option',
            #                                             on_change=data_editor_change,
            #                                             args=('groubly_option', 'groubly_option'),  # 'Season',
            #                                             disabled=st.session_state.groubly_disable,
            #                                             index=None, placeholder="Choose an option",
            #                                             label_visibility="visible")
            # st.write(st.session_state['resample_option'], st.session_state['groubly_option'])

        if len(option['title']) == 0 and option['plot_type'] == 'sim lines':
            option['title'] = 'Simulation'
        elif len(option['title']) == 0 and option['plot_type'] == 'ref lines':
            option['title'] = 'Reference'

        option['data_path'] = path + f'/data/stn_{refselect}_{simselect}/'
        option['vars'] = vars
        option['units'] = unit

        if option['plot_type'] == 'each site':
            each_line(option, showing_items, selected_item)
        elif option['plot_type'] == 'ref lines':
            ref_lines(option, showing_items, selected_item)
        elif option['plot_type'] == 'sim lines':
            sim_lines(option, showing_items, selected_item)
        else:
            st.warning('Please choose first!')

    def __generate_image_stn_index(self, item, metric, selected_item, ref, sim, path):

        key_value = f"{selected_item}_{metric}_{ref}_{sim}_"
        option = {}
        with st.container(height=None, border=True):
            col1, col2, col3, col4 = st.columns((3.5, 3, 3, 3))
            with col1:
                option['title'] = st.text_input('Title', value=f'', label_visibility="visible", key=f"{key_value}title")
                option['title_size'] = st.number_input("Title label size", min_value=0, value=20,
                                                       key=f"{key_value}_titlesize")
            with col2:
                option['xticklabel'] = st.text_input('X tick labels', value='Longitude', label_visibility="visible",
                                                     key=f"{key_value}xticklabel")
                option['xtick'] = st.number_input("xtick label size", min_value=0, value=17, key=f"{key_value}xtick")

            with col3:
                option['yticklabel'] = st.text_input('Y tick labels', value='Latitude', label_visibility="visible",
                                                     key=f"{key_value}yticklabel")
                option['ytick'] = st.number_input("ytick label size", min_value=0, value=17, key=f"{key_value}ytick")

            with col4:
                option['labelsize'] = st.number_input("labelsize", min_value=0, value=17, key=f"{key_value}labelsize")
                option['axes_linewidth'] = st.number_input("axes linewidth", min_value=0, value=1,
                                                           key=f"{key_value}axes_linewidth")

            st.divider()

            with st.expander('More info',expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                # min_lon, max_lon, min_lat, max_lat
                option["min_lon"] = col1.number_input(f"minimal longitude", value=st.session_state['generals']['min_lon'])
                option["max_lon"] = col2.number_input(f"maximum longitude", value=st.session_state['generals']['max_lon'])
                option["min_lat"] = col3.number_input(f"minimal latitude", value=st.session_state['generals']['min_lat'])
                option["max_lat"] = col4.number_input(f"maximum latitude", value=st.session_state['generals']['max_lat'])
                st.divider()
                col1, col2, col3 = st.columns((3,2,2))
                option["vmin_max_on"] = col1.toggle('Setting max min', value=False, key=f"{key_value}vmin_max_on")
                error = False
                try:
                    import math
                    df = pd.read_csv(f'{path}/{item}/{selected_item}_stn_{ref}_{sim}_evaluations.csv', header=0)
                    # df = pd.read_csv(f'{path}/{item}/{selected_item}_stn_{ref}_{sim}_evaluations.csv', header=0)
                    min_metric = -999.0
                    max_metric = 100000.0
                    ind0 = df[df['%s' % (metric)] > min_metric].index
                    data_select0 = df.loc[ind0]
                    ind1 = data_select0[data_select0['%s' % (metric)] < max_metric].index
                    data_select = data_select0.loc[ind1]
                    plotvar = data_select['%s' % (metric)].values

                    try:
                        vmin, vmax = np.percentile(plotvar, 5), np.percentile(plotvar, 95)
                        if metric in ['bias', 'percent_bias', 'rSD', 'PBIAS_HF', 'PBIAS_LF']:
                            vmax = math.ceil(vmax)
                            vmin = math.floor(vmin)
                        elif metric in ['KGE', 'KGESS', 'correlation', 'kappa_coeff', 'rSpearman']:
                            vmin, vmax = -1, 1
                        elif metric in ['NSE', 'LNSE', 'ubNSE', 'rNSE', 'wNSE', 'wsNSE']:
                            vmin, vmax = math.floor(vmin), 1
                        elif metric in ['RMSE', 'CRMSD', 'MSE', 'ubRMSE', 'nRMSE', 'mean_absolute_error', 'ssq', 've',
                                        'absolute_percent_bias']:
                            vmin, vmax = 0, math.ceil(vmax)
                        else:
                            vmin, vmax = 0, 1
                    except:
                        vmin, vmax = 0, 1

                    if option["vmin_max_on"]:
                        try:
                            option["vmin"] = col2.number_input(f"colorbar min", value=vmin)
                            option["vmax"] = col3.number_input(f"colorbar max", value=vmax)

                            min_value, max_value = np.nanmin(plotvar), np.nanmax(plotvar)
                            if min_value < option['vmin'] and max_value > option['vmax']:
                                oextend = 'both'
                            elif min_value > option['vmin'] and max_value > option['vmax']:
                                extend = 'max'
                            elif min_value < option['vmin'] and max_value < option['vmax']:
                                extend = 'min'
                            else:
                                extend = 'neither'
                        except ValueError:
                            st.error(f"Max value must larger than min value.")
                    else:
                        option["vmin"] = vmin
                        option["vmax"] = vmax
                        extend = 'neither'
                except Exception as e:
                    st.error(f"Error: {e}")
                    error = True




                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    option['cmap'] = st.selectbox('Colorbar',
                                                  ['coolwarm',
                                                   'coolwarm_r','Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r',
                                                   'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r',
                                                   'Grays', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges',
                                                   'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r',
                                                   'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r',
                                                   'PuBu_r',
                                                   'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r',
                                                   'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r',
                                                   'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r',
                                                   'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r',
                                                   'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r',
                                                   'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r',
                                                   'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm',
                                                   'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag',
                                                   'flag_r',
                                                   'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey',
                                                   'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow',
                                                   'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r',
                                                   'gist_yerg', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray',
                                                   'gray_r',
                                                   'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r',
                                                   'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r',
                                                   'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow',
                                                   'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer',
                                                   'summer_r',
                                                   'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c',
                                                   'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight',
                                                   'twilight_r',
                                                   'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter',
                                                   'winter_r'], index=0, placeholder="Choose an option",
                                                  label_visibility="visible")
                with col2:
                    option['colorbar_label'] = st.text_input('colorbar label', value=metric.replace("_"," "), label_visibility="visible")
                with col3:
                    option["colorbar_position"] = st.selectbox('colorbar position', ['horizontal', 'vertical'],  # 'Season',
                                                               index=0, placeholder="Choose an option",
                                                               label_visibility="visible")
                    if option["colorbar_position"] == 'vertical':
                        left, bottom, right, top = 0.94, 0.24, 0.02, 0.5
                    else:
                        left, bottom, right, top = 0.26, 0.14, 0.5, 0.03

                def get_extend(extend):
                    my_list = ['neither', 'both', 'min', 'max']
                    index = my_list.index(extend.lower())
                    return index

                with col4:
                    option["extend"] = st.selectbox(f"colorbar extend", ['neither', 'both', 'min', 'max'],
                                                    index=get_extend(extend), placeholder="Choose an option", label_visibility="visible",
                                                    key=f"{key_value}extend")

                option['marker'] = col1.selectbox(f'Marker style',
                                                  ['.', 'x', 'o', ">", '<', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X', "+",
                                                   "^", "v"],
                                                  index=2,
                                                  placeholder="Choose an option",
                                                  label_visibility="visible")

                option['markersize'] = col2.number_input(f"Markersize", min_value=0, value=15, step=1)

                col1, col2, col3 = st.columns(3)
                if option["colorbar_position"] == 'vertical':
                    left, bottom, right, top = 0.94, 0.24, 0.02, 0.5
                else:
                    left, bottom, right, top = 0.26, 0.14, 0.5, 0.03
                option['colorbar_position_set'] = col1.toggle('Setting colorbar position', value=False,
                                                              key=f"{key_value}colorbar_position_set")
                if option['colorbar_position_set']:
                    col1, col2, col3, col4 = st.columns(4)
                    option["colorbar_left"] = col1.number_input(f"colorbar left", value=left)
                    option["colorbar_bottom"] = col2.number_input(f"colorbar bottom", value=bottom)
                    option["colorbar_width"] = col3.number_input(f"colorbar width", value=right)
                    option["colorbar_height"] = col4.number_input(f"colorbar height", value=top)


            col1, col2, col3 = st.columns(3)
            with col1:
                option["x_wise"] = st.number_input(f"X Length", min_value=0, value=10)
                option['saving_format'] = st.selectbox('Image saving format', ['png', 'jpg', 'eps'],
                                                       index=1, placeholder="Choose an option", label_visibility="visible")
            with col2:
                option["y_wise"] = st.number_input(f"y Length", min_value=0, value=6)
                option['font'] = st.selectbox('Image saving format',
                                              ['Times new roman', 'Arial', 'Courier New', 'Comic Sans MS', 'Verdana',
                                               'Helvetica',
                                               'Georgia', 'Tahoma', 'Trebuchet MS', 'Lucida Grande'],
                                              index=0, placeholder="Choose an option", label_visibility="visible")
            with col3:
                option['dpi'] = st.number_input(f"Figure dpi", min_value=0, value=300)
        if not error:
            make_stn_plot_index(path, ref, sim, item, metric, selected_item, option)
        else:
            st.error(f'Please check File: {selected_item}_stn_{ref}_{sim}_evaluations.csv')

    def __generate_image_geo_index(self, item, metric, selected_item, ref, sim, path):

        key_value = f"{selected_item}_{metric}_{ref}_{sim}_"
        option = {}
        with st.container(height=None, border=True):
            col1, col2, col3, col4 = st.columns((3.5, 3, 3, 3))
            with col1:
                option['title'] = st.text_input('Title', value=f'', label_visibility="visible", key=f"{key_value}title")
                option['title_size'] = st.number_input("Title label size", min_value=0, value=20,
                                                       key=f"{key_value}_titlesize")
            with col2:
                option['xticklabel'] = st.text_input('X tick labels', value='Longitude', label_visibility="visible",
                                                     key=f"{key_value}xticklabel")
                option['xtick'] = st.number_input("xtick label size", min_value=0, value=17, key=f"{key_value}xtick")

            with col3:
                option['yticklabel'] = st.text_input('Y tick labels', value='Latitude', label_visibility="visible",
                                                     key=f"{key_value}yticklabel")
                option['ytick'] = st.number_input("ytick label size", min_value=0, value=17, key=f"{key_value}ytick")

            with col4:
                option['labelsize'] = st.number_input("labelsize", min_value=0, value=17, key=f"{key_value}labelsize")
                option['axes_linewidth'] = st.number_input("axes linewidth", min_value=0, value=1,
                                                           key=f"{key_value}axes_linewidth")

            st.divider()

            col1, col2, col3, col4 = st.columns(4)
            # min_lon, max_lon, min_lat, max_lat
            option["min_lon"] = col1.number_input(f"minimal longitude", value=st.session_state['generals']['min_lon'])
            option["max_lon"] = col2.number_input(f"maximum longitude", value=st.session_state['generals']['max_lon'])
            option["min_lat"] = col3.number_input(f"minimal latitude", value=st.session_state['generals']['min_lat'])
            option["max_lat"] = col4.number_input(f"maximum latitude", value=st.session_state['generals']['max_lat'])

            with col1:
                option['cmap'] = st.selectbox('Colorbar',
                                              ['coolwarm', 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn',
                                               'BuGn_r',
                                               'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r',
                                               'Grays', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges',
                                               'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r',
                                               'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r',
                                               'PuBu_r',
                                               'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r',
                                               'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r',
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
                                               'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r',
                                               'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r',
                                               'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow',
                                               'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer',
                                               'summer_r',
                                               'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c',
                                               'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight',
                                               'twilight_r',
                                               'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter',
                                               'winter_r'], index=0, placeholder="Choose an option",
                                              label_visibility="visible")
            with col2:
                option['colorbar_label'] = st.text_input('colorbar label', value=metric.replace('_', ' '),
                                                         label_visibility="visible")
            with col3:
                option["colorbar_position"] = st.selectbox('colorbar position', ['horizontal', 'vertical'],  # 'Season',
                                                           index=0, placeholder="Choose an option",
                                                           label_visibility="visible")

            with col4:
                option["extend"] = st.selectbox(f"colorbar extend", ['neither', 'both', 'min', 'max'],
                                                index=0, placeholder="Choose an option", label_visibility="visible",
                                                key=f"{key_value}extend")

            if option["colorbar_position"] == 'vertical':
                left, bottom, right, top = 0.94, 0.24, 0.02, 0.5
            else:
                left, bottom, right, top = 0.26, 0.14, 0.5, 0.03
            col1, col2, col3 = st.columns(3)
            option['colorbar_position_set'] = col1.toggle('Setting colorbar position', value=False,
                                                          key=f"{key_value}colorbar_position_set")
            if option['colorbar_position_set']:
                col1, col2, col3, col4 = st.columns(4)
                option["colorbar_left"] = col1.number_input(f"colorbar left", value=left)
                option["colorbar_bottom"] = col2.number_input(f"colorbar bottom", value=bottom)
                option["colorbar_width"] = col3.number_input(f"colorbar width", value=right)
                option["colorbar_height"] = col4.number_input(f"colorbar height", value=top)

            col1, col2, col3, col4 = st.columns(4)
            option["vmin_max_on"] = col1.toggle('Setting max min', value=False, key=f"{key_value}vmin_max_on")
            option["colorbar_ticks"] = col2.number_input(f"Colorbar Ticks locater", value=0.5, step=0.1)
            error = False
            try:
                ds = xr.open_dataset(f'{path}/{item}/{selected_item}_ref_{ref}_sim_{sim}_{metric}.nc')[metric]
                quantiles = ds.quantile([0.05, 0.95], dim=['lat', 'lon'])
                import math
                if metric in ['bias', 'percent_bias', 'rSD', 'PBIAS_HF', 'PBIAS_LF']:
                    vmax = math.ceil(quantiles[1].values)
                    vmin = math.floor(quantiles[0].values)
                elif metric in ['KGE', 'KGESS', 'correlation', 'kappa_coeff', 'rSpearman']:
                    vmin, vmax = -1, 1
                elif metric in ['NSE', 'LNSE', 'ubNSE', 'rNSE', 'wNSE', 'wsNSE']:
                    vmin, vmax = math.floor(quantiles[1].values), 1
                elif metric in ['RMSE', 'CRMSD', 'MSE', 'ubRMSE', 'nRMSE', 'mean_absolute_error', 'ssq', 've',
                                'absolute_percent_bias']:
                    vmin, vmax = 0, math.ceil(quantiles[1].values)
                else:
                    vmin, vmax = 0, 1

                if option["vmin_max_on"]:
                    try:
                        option["vmin"] = col3.number_input(f"colorbar min", value=vmin)
                        option["vmax"] = col4.number_input(f"colorbar max", value=vmax)
                    except ValueError:
                        st.error(f"Max value must larger than min value.")
                else:
                    option["vmin"] = vmin
                    option["vmax"] = vmax
            except Exception as e:
                st.error(f"Error: {e}")
                error = True
            st.divider()

            col1, col2, col3 = st.columns(3)
            with col1:
                option["x_wise"] = st.number_input(f"X Length", min_value=0, value=10)
                option['saving_format'] = st.selectbox('Image saving format', ['png', 'jpg', 'eps'],
                                                       index=1, placeholder="Choose an option", label_visibility="visible")
            with col2:
                option["y_wise"] = st.number_input(f"y Length", min_value=0, value=6)
                option['font'] = st.selectbox('Image saving format',
                                              ['Times new roman', 'Arial', 'Courier New', 'Comic Sans MS', 'Verdana',
                                               'Helvetica',
                                               'Georgia', 'Tahoma', 'Trebuchet MS', 'Lucida Grande'],
                                              index=0, placeholder="Choose an option", label_visibility="visible")
            with col3:
                option['dpi'] = st.number_input(f"Figure dpi", min_value=0, value=300)
        if not error:
            make_geo_plot_index(path, ref, sim, item, metric, selected_item, option)
        else:
            st.error(f'Please check File: {selected_item}_ref_{ref}_sim_{sim}_{metric}.nc')


class visualization_replot_Comparison:
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

    # -=========================================================
    def Comparison_replot(self):

        case_path = os.path.join(self.generals['basedir'], self.generals['basename'], "output", "comparisons")
        if not self.generals['comparison']:
            st.info('You haven\'t selected a comparison module!')
        else:
            showing_item = [k for k, v in self.comparisons.items() if v]
            if not showing_item:
                st.info('No comparison item selected!')

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
                    st.write(taylor_diagram_file)
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

        elif item == "Relative_Score":
            st.info(f'Relative_Score not ready yet!', icon="ℹ️")

    def __heatmap(self, dir_path, score):
        iscore = score.replace('_', ' ')
        option = {}
        item = 'heatmap'
        with st.container(height=None, border=True):
            col1, col2, col3, col4 = st.columns((3.5, 3, 3, 3))
            with col1:
                option['title'] = st.text_input('Title', value=f'Heatmap of {iscore}', label_visibility="visible",
                                                key=f"{item}_title")
                option['title_size'] = st.number_input("Title label size", min_value=0, value=20, key=f"{item}_title_size")

            with col2:
                option['xlabel'] = st.text_input('X labels', value='Simulations', label_visibility="visible",
                                                 key=f"{item}_xlabel")
                option['xticksize'] = st.number_input("Xtick label size", min_value=0, value=17, key=f"{item}_xlabelsize")

            with col3:
                option['ylabel'] = st.text_input('Y labels', value='References', label_visibility="visible",
                                                 key=f"{item}_ylabel")
                option['yticksize'] = st.number_input("Ytick label size", min_value=0, value=17, key=f"{item}_ylabelsize")

            with col4:
                option['fontsize'] = st.number_input("Fontsize", min_value=0, value=17, key=f"{item}_fontsize")
                option['axes_linewidth'] = st.number_input("axes linewidth", min_value=0, value=1,
                                                           key=f"{item}_axes_linewidth")

            col1, col2, col3 = st.columns((2, 1, 1))
            with col1:
                set_label = st.expander("More info", expanded=False)
                col11, col22 = set_label.columns(2)

                option["x_rotation"] = col11.number_input(f"x rotation", min_value=-90, max_value=90, value=45,
                                                          key=f'{item}_x_rotation')
                option['x_ha'] = col11.selectbox('x ha', ['right', 'left', 'center'], key=f'{item}_x_ha',
                                                 index=0, placeholder="Choose an option", label_visibility="visible")
                option['ticks_format'] = col11.selectbox('Tick Format',
                                                         ['%f', '%G', '%.1f', '%.1G', '%.2f', '%.2G',
                                                          '%.3f', '%.3G'],
                                                         index=2, placeholder="Choose an option", label_visibility="visible",
                                                         key=f"{item}_ticks_format")

                option["y_rotation"] = col22.number_input(f"y rotation", min_value=-90, max_value=90, value=45,
                                                          key=f'{item}_y_rotation')
                option['y_ha'] = col22.selectbox('y ha', ['right', 'left', 'center'], key=f'{item}_y_ha',
                                                 index=0, placeholder="Choose an option", label_visibility="visible")

            with col2:
                option['cmap'] = st.selectbox('Colorbar',
                                              ['coolwarm', 'coolwarm_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn',
                                               'BuGn_r',
                                               'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r',
                                               'Grays', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges',
                                               'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r',
                                               'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r',
                                               'PuBu_r',
                                               'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r',
                                               'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r',
                                               'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r',
                                               'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r',
                                               'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r',
                                               'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r',
                                               'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'copper', 'copper_r',
                                               'cubehelix', 'cubehelix_r', 'flag', 'flag_r',
                                               'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey',
                                               'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow',
                                               'gist_rainbow_r', 'gray', 'gray_r',
                                               'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r',
                                               'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r',
                                               'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow',
                                               'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer',
                                               'summer_r',
                                               'terrain', 'terrain_r', 'viridis', 'viridis_r', 'winter',
                                               'winter_r'], index=0, placeholder="Choose an option", key=f'{item}_cmap',
                                              label_visibility="visible")
            with col3:
                option["colorbar_position"] = st.selectbox('colorbar position', ['horizontal', 'vertical'],
                                                           key=f"{item}_colorbar_position",
                                                           index=0, placeholder="Choose an option",
                                                           label_visibility="visible")

            def get_cases(items, title):
                case_item = {}
                for item in items:
                    case_item[item] = True
                with st.popover(title, use_container_width=True):
                    st.subheader(f"Showing {title}", divider=True)
                    if title != 'cases':
                        for item in case_item:
                            case_item[item] = st.checkbox(item.replace("_", " "), key=f"{item}__heatmap",
                                                          value=case_item[item])
                    else:
                        for item in case_item:
                            case_item[item] = st.checkbox(item, key=f"{item}__heatmap",
                                                          value=case_item[item])
                return [item for item, value in case_item.items() if value]

            items = [k for k in self.selected_items]
            cases = list(
                set([value for key in self.selected_items for value in self.sim['general'][f"{key}_sim_source"] if value]))
            col1, col2 = st.columns(2)
            with col1:
                items = get_cases(items, 'Selected items')
            with col2:
                cases = get_cases(cases, 'cases')

            col1, col2, col3 = st.columns(3)
            with col1:
                option["x_wise"] = st.number_input(f"X Length", min_value=0, value=10, key=f"{item}_x_wise")
                option['saving_format'] = st.selectbox('Image saving format', ['png', 'jpg', 'eps'],
                                                       index=1, placeholder="Choose an option", label_visibility="visible",
                                                       key=f"{item}_saving_format")
            with col2:
                option["y_wise"] = st.number_input(f"y Length", min_value=0, value=6, key=f"{item}_y_wise")
                option['font'] = st.selectbox('Image saving format',
                                              ['Times new roman', 'Arial', 'Courier New', 'Comic Sans MS', 'Verdana',
                                               'Helvetica',
                                               'Georgia', 'Tahoma', 'Trebuchet MS', 'Lucida Grande'],
                                              index=0, placeholder="Choose an option", label_visibility="visible",
                                              key=f"{item}_font")
            with col3:
                option['dpi'] = st.number_input(f"Figure dpi", min_value=0, value=300, key=f"{item}_dpi")

        make_scenarios_scores_comparison_heat_map(dir_path, score, items, cases, option)

    def __heatmap_groupby(self, item, file, selected_item, score, sim_source, ref_source, dir_path):
        option = {}
        option['path'] = dir_path

        with st.container(height=None, border=True):
            col1, col2, col3, col4 = st.columns((3.5, 3, 3, 3))
            with col1:
                option['title'] = st.text_input('Title', value=f'Heatmap of {score}', label_visibility="visible",
                                                key=f"{item}_title")
                option['title_size'] = st.number_input("Title label size", min_value=0, value=20, key=f"{item}_title_size")

            with col2:
                option['xlabel'] = st.text_input('X labels', value=sim_source, label_visibility="visible",
                                                 key=f"{item}_xlabel")
                option['xticksize'] = st.number_input("X ticks size", min_value=0, value=17, key=f"{item}_xlabelsize")

            with col3:
                option['ylabel'] = st.text_input('Y labels', value=ref_source, label_visibility="visible",
                                                 key=f"{item}_ylabel")
                option['yticksize'] = st.number_input("Y ticks size", min_value=0, value=17, key=f"{item}_ylabelsize")

            with col4:
                option['fontsize'] = st.number_input("Fontsize", min_value=0, value=17, step=1, key=f"{item}_fontsize",
                                                     help='Control label size on each ceil')
                option['axes_linewidth'] = st.number_input("axes linewidth", min_value=0, value=1,
                                                           key=f"{item}_axes_linewidth")

            col1, col2, col3 = st.columns((1, 1, 2))
            option["x_ticklabel"] = col1.selectbox('X tick labels Format', ['Normal', 'Shorter'],  # 'Season',
                                                   index=1, placeholder="Choose an option", label_visibility="visible",
                                                   key=f"{item}_x_ticklabel")
            option['ticks_format'] = col2.selectbox('Tick Format',
                                                    ['%f', '%G', '%.1f', '%.1G', '%.2f', '%.2G',
                                                     '%.3f', '%.3G'],
                                                    index=4, placeholder="Choose an option", label_visibility="visible",
                                                    key=f"{item}_ticks_format")

            def get_cases(items, title):
                case_item = {}
                for item in items:
                    case_item[item] = True
                with st.popover(f"Select {title}", use_container_width=True):
                    st.subheader(f"Showing {title}", divider=True)
                    for item in case_item:
                        case_item[item] = st.checkbox(item.replace("_", " "), key=f"{item}__heatmap_groupby",
                                                      value=case_item[item])
                    selected = [item for item, value in case_item.items() if value]
                    if len(selected) > 0:
                        return selected
                    else:
                        st.error('You must choose one item!')

            if score == 'scores':
                selected_metrics = [k for k, v in self.scores.items() if v]
            else:
                selected_metrics = [k for k, v in self.metrics.items() if v]
            with col3:
                selected_metrics = get_cases(selected_metrics, score.title())

            st.divider()

            set_colorbar = st.expander("More info", expanded=False)
            col1, col2, col3, col4 = set_colorbar.columns(4)

            option["x_rotation"] = col1.number_input(f"x rotation", min_value=-90, max_value=90, value=45,
                                                     key=f"{item}_x_rotation")
            option['x_ha'] = col2.selectbox('x ha', ['right', 'left', 'center'],
                                            index=0, placeholder="Choose an option", label_visibility="visible",
                                            key=f"{item}_x_ha")
            option["y_rotation"] = col3.number_input(f"y rotation", min_value=-90, max_value=90, value=45,
                                                     key=f"{item}_y_rotation")
            option['y_ha'] = col4.selectbox('y ha', ['right', 'left', 'center'],
                                            index=0, placeholder="Choose an option", label_visibility="visible",
                                            key=f"{item}_y_ha")

            col1, col2, col3 = set_colorbar.columns(3)
            option['cmap'] = col1.selectbox('Colorbar',
                                            ['coolwarm', 'coolwarm_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r',
                                             'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r',
                                             'Grays', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges',
                                             'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r',
                                             'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r',
                                             'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r',
                                             'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r',
                                             'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r',
                                             'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r',
                                             'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r',
                                             'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r',
                                             'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'copper', 'copper_r',
                                             'cubehelix', 'cubehelix_r', 'flag', 'flag_r',
                                             'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey',
                                             'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow',
                                             'gist_rainbow_r', 'gray', 'gray_r',
                                             'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r',
                                             'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r',
                                             'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow',
                                             'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r',
                                             'terrain', 'terrain_r', 'viridis', 'viridis_r', 'winter',
                                             'winter_r'], index=0, placeholder="Choose an option", key=f'{item}_cmap',
                                            label_visibility="visible")

            option['show_colorbar'] = col2.toggle('Showing colorbar?', value=True, key=f"{item}colorbar_on")

            if option['show_colorbar']:
                if score == 'metrics':
                    option["extend"] = col3.selectbox(f"colorbar extend", ['neither', 'both', 'min', 'max'],
                                                      index=1, placeholder="Choose an option", label_visibility="visible",
                                                      key=f"{item}_extend")
                else:
                    option["colorbar_position"] = col3.selectbox('colorbar position', ['horizontal', 'vertical'],
                                                                 key=f"{item}_colorbar_position",
                                                                 index=0, placeholder="Choose an option",
                                                                 label_visibility="visible")

            st.divider()

            if score == 'scores':
                items = [k for k, v in self.scores.items() if v]
            else:
                items = [k for k, v in self.metrics.items() if v]

            col1, col2, col3 = st.columns(3)
            with col1:
                if item == 'PFT_groupby':
                    x_value = 17
                else:
                    x_value = 18
                option["x_wise"] = st.number_input(f"X Length", min_value=0, value=x_value, key=f"{item}_x_wise")
                option['saving_format'] = st.selectbox('Image saving format', ['png', 'jpg', 'eps'],
                                                       index=1, placeholder="Choose an option",
                                                       label_visibility="visible",
                                                       key=f"{item}_saving_format")
            with col2:
                try:
                    option["y_wise"] = st.number_input(f"y Length", min_value=0, value=len(selected_metrics),
                                                       key=f"{item}_y_wise")
                except:
                    option["y_wise"] = 1
                option['font'] = st.selectbox('Image saving format',
                                              ['Times new roman', 'Arial', 'Courier New', 'Comic Sans MS', 'Verdana',
                                               'Helvetica',
                                               'Georgia', 'Tahoma', 'Trebuchet MS', 'Lucida Grande'],
                                              index=0, placeholder="Choose an option", label_visibility="visible",
                                              key=f"{item}_font")
            with col3:
                option['dpi'] = st.number_input(f"Figure dpi", min_value=0, value=300, key=f"{item}_dpi'")

        if selected_metrics:
            make_LC_based_heat_map(option, file, selected_metrics, selected_item, score, ref_source, sim_source, item, items)

    def __taylor(self, dir_path, selected_item, ref_source):
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
                        markers[sim_source]['faceColor'] = st.selectbox(f'{sim_source} faceColor', ['w', 'b', 'k', 'r','none'],
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

        st.json(option, expanded=False)
        make_scenarios_comparison_Target_Diagram(option, selected_item, biases, crmsds, rmses,
                                                 ref_source, self.sim['general'][f'{selected_item}_sim_source'])

    def __Kernel_Density_Estimate(self, dir_path, selected_item, score, ref_source):
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
            ref_data_type = self.ref[ref_source]['general'][f'data_type']

            from matplotlib import cm
            import matplotlib.colors as mcolors
            import itertools
            colors = cm.Set3(np.linspace(0, 1, len(self.sim['general'][f'{selected_item}_sim_source']) + 1))
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

            sim_sources = self.sim['general'][f'{selected_item}_sim_source']
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

                    sim_data_type = self.sim[sim_source]['general'][f'data_type']
                    if ref_data_type == 'stn' or sim_data_type == 'stn':
                        ref_varname = self.ref[f'{selected_item}'][f'{ref_source}_varname']
                        sim_varname = self.sim[f'{selected_item}'][f'{sim_source}_varname']
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
                make_scenarios_comparison_Kernel_Density_Estimate(option, selected_item, ref_source,
                                                                  sim_sources, datasets_filtered,
                                                                  score)

    def __Whisker_Plot(self, dir_path, selected_item, score, ref_source):
        option = {}
        item = 'Whisker_Plot'
        with st.container(height=None, border=True):
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

            st.divider()
            ref_data_type = self.ref[ref_source]['general'][f'data_type']

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
                        case_item[item] = col.checkbox(item, key=f'{item}__Whisker_Plot',
                                                       value=case_item[item])
                    if len([item for item, value in case_item.items() if value]) > 0:
                        return [item for item, value in case_item.items() if value]
                    else:
                        st.error('You must choose one item!')

            sim_sources = self.sim['general'][f'{selected_item}_sim_source']
            sim_sources = get_cases(sim_sources, 'cases')

            datasets_filtered = []
            if sim_sources:
                for sim_source in sim_sources:
                    sim_data_type = self.sim[sim_source]['general'][f'data_type']
                    if ref_data_type == 'stn' or sim_data_type == 'stn':
                        ref_varname = self.ref[f'{selected_item}'][f'{ref_source}_varname']
                        sim_varname = self.sim[f'{selected_item}'][f'{sim_source}_varname']

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
                        except FileNotFoundError:
                            st.error(f"File {file_path} not found. Please check the file path.")
                        except KeyError as e:
                            st.error(f"Key error: {e}. Please check the keys in the option dictionary.")
                        except Exception as e:
                            st.error(f"An error occurred: {e}")
                    if score == 'percent_bias':
                        data = data[(data <= 100) & (data >= -100)]
                    data = data[~np.isinf(data)]
                    data = data[~np.isnan(data)]
                    try:
                        lower_bound, upper_bound = np.percentile(data, 5), np.percentile(data, 95)
                        if score in ['bias', 'rSD', 'PBIAS_HF', 'PBIAS_LF']:
                            data = data[(data >= lower_bound) & (data <= upper_bound)]
                        elif score in ['KGE', 'KGESS', 'NSE', 'LNSE', 'ubNSE', 'rNSE', 'wNSE', 'wsNSE']:
                            data = data[(data >= lower_bound)]
                        elif score in ['RMSE', 'CRMSD', 'MSE', 'ubRMSE', 'nRMSE', 'mean_absolute_error', 'ssq', 've',
                                       'absolute_percent_bias']:
                            data = data[(data <= upper_bound)]
                    except Exception as e:
                        st.error(f"{selected_item} {ref_source} {sim_source} {selected_item} failed!")
                    datasets_filtered.append(data)  # Filter out NaNs and append

                # ----------------------------------------------
                # Create the whisker plot
                def remove_outliers(data_list):
                    q1, q3 = np.percentile(data_list, [25, 75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr - iqr
                    upper_bound = q3 + 1.5 * iqr + iqr
                    return [lower_bound, upper_bound]

                bound = [remove_outliers(d) for d in datasets_filtered]
                max_value = max([d[1] for d in bound])
                min_value = min([d[0] for d in bound])

                if score in ['RMSE', 'CRMSD']:
                    min_value = min_value * 0 - 0.2
                # elif score in ['KGE', 'KGESS']:
                #     max_value = max_value * 0 + 1
                #     min_value = min_value * 0 - 1.0

            with st.expander("More info", expanded=False):
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

                st.divider()
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

                st.divider()
                col1, col2, col3 = st.columns((3, 2, 2))
                option['grid'] = col1.toggle("Turn on to showing grid", value=True, label_visibility="visible",
                                             key=f'Whisker_grid')
                if option['grid']:
                    option['grid_style'] = col2.selectbox('Line Style', ['solid', 'dotted', 'dashed', 'dashdot'],
                                                          index=2, placeholder="Choose an option", label_visibility="visible",
                                                          key=f'Whisker_grid_style')
                    option['grid_linewidth'] = col3.number_input("Linewidth", min_value=0., value=1.,
                                                                 key=f'Whisker_grid_linewidth')

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

        if sim_sources:
            make_scenarios_comparison_Whisker_Plot(option, selected_item, ref_source,
                                                   sim_sources, datasets_filtered, score)
        else:
            st.error('You must choose at least one case!')

    def __Portrait_Plot_seasonal_variable(self, file, selected_item, score, ref_source, item):
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
        option = {}
        with st.container(height=None, border=True):
            col1, col2, col3, col4 = st.columns((3.5, 3, 3, 3))
            with col1:
                option['title'] = st.text_input('Title',
                                                value=f"Parallel Coordinates Plot - {score.replace('_', ' ')}",
                                                label_visibility="visible",
                                                key=f"{item} title")
                option['title_size'] = st.number_input("Title label size", min_value=0, value=20, key=f"{item}_title_size")

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
                option["fontsize"] = col1.number_input(f"Font size", value=15., step=1., key=f"{item}_legend_fontsize")

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

                colors = {}

                import matplotlib.colors as mcolors

                colors_list = [plt.get_cmap('tab20')(c) for c in np.linspace(0, 1, 20)]
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
        option = {}
        with st.container(height=None, border=True):
            col1, col2, col3, col4 = st.columns((4, 3, 3, 3))
            # with col1:
            option['title'] = col1.text_input('Title',
                                              value=f"",
                                              label_visibility="visible",
                                              key=f"{item} title")
            option['title_size'] = col2.number_input("Title label size", min_value=0, value=15, key=f"{item}_title_size")
            option['fontsize'] = col3.number_input("Font size", min_value=0, value=15,
                                                   key=f'{item}_fontsize')
            option['tick'] = col4.number_input("Tick size", min_value=0, value=15,
                                               key=f'{item}_ticksize')
            col1, col2, col3, col4 = st.columns((3, 3, 3, 3))
            col1, col2, col3 = st.columns((2, 1, 1))
            option["var_loc"] = col1.toggle('Put Variable labels in X axis?', value=False, key=f"{item}_var_loc")
            option['yticksize'] = col2.number_input("Y ticks size", min_value=0, value=17, key=f"{item}_ytick")
            option['xticksize'] = col3.number_input("X ticks size", min_value=0, value=17, key=f"{item}_xtick")
            if not option["var_loc"]:
                option['xlabel'] = col1.text_input('X label', value='Single Model Performance Index',
                                                   label_visibility="visible",
                                                   key=f"{item}_xlabel")
                option['ylabel'] = ''
            else:
                option['ylabel'] = col1.text_input('Y labels', value='Single Model Performance Index',
                                                   label_visibility="visible",
                                                   key=f"{item}_ylabel")
                option['xlabel'] = ''

            st.divider()

            def get_cases(items, title):
                case_item = {}
                for item in items:
                    case_item[item] = True
                with st.popover(title, use_container_width=True):
                    st.subheader(f"Showing Variables", divider=True)
                    for item in case_item:
                        case_item[item] = st.checkbox(item.replace("_", " "), key=f"{item}__Single_Model_Performance_Index",
                                                      value=case_item[item])
                    if len([item for item, value in case_item.items() if value]) > 0:
                        return [item for item, value in case_item.items() if value]
                    else:
                        st.error('You must choose one item!')

            if isinstance(selected_items, str): selected_items = [selected_items]
            items = get_cases(selected_items, 'Selected Variables')

            with st.expander("Other informations setting", expanded=False):
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
                st.divider()
                st.write("##### :blue[Marker Color]")
                import matplotlib.colors as mcolors
                import matplotlib.cm as cm
                hex_colors = [cm.get_cmap('tab10')(c) for c in np.linspace(0, 1, len(items))]
                colors = itertools.cycle([mcolors.rgb2hex(color) for color in hex_colors])
                cols = itertools.cycle(st.columns(4))
                markers = {}
                for selected_item in items:
                    col = next(cols)
                    col.write(f":orange[{selected_item.replace('_', ' ')}]")
                    markers[selected_item] = col.color_picker(f'{selected_item} colors', value=next(colors),
                                                              key=f"{item}_{selected_item}_colors",
                                                              disabled=False,
                                                              label_visibility="collapsed")
                st.divider()
                col1, col2, col3 = st.columns((3, 3, 2))
                option["hspace"] = col1.number_input(f"hspace", step=0.1, value=0.0)
                option["wspace"] = col2.number_input(f"wspace", min_value=0., max_value=1.0, value=0.1)

            option['COLORS'] = markers
            st.divider()
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

        if items:
            make_scenarios_comparison_Single_Model_Performance_Index(option, file, items, ref)

    def __Ridgeline_Plot(self, dir_path, selected_item, score, ref_source):
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
            ref_data_type = self.ref[ref_source]['general'][f'data_type']

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

            sim_sources = self.sim['general'][f'{selected_item}_sim_source']
            sim_sources = get_cases(sim_sources, 'cases')
            option['colormap']=False

            with st.expander("Colors setting", expanded=False):
                markers = {}
                datasets_filtered = []

                option['colormap'] = st.toggle('Use colormap?',  key=f'{item}_colormap',value=option['colormap'])

                col1, col2, col3 = st.columns(3)
                col1.write('##### :blue[Line colors]')
                col2.write('##### :blue[Line width]')
                col3.write('##### :blue[Line alpha]')
                if option['colormap']:
                    option['cmap'] = col1.selectbox('Colorbar',
                                                  ['coolwarm', 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn',
                                                   'BuGn_r',
                                                   'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r',
                                                   'Grays', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges',
                                                   'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r',
                                                   'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r',
                                                   'PuBu_r',
                                                   'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r',
                                                   'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r',
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
                                                   'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r',
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
                    sim_data_type = self.sim[sim_source]['general'][f'data_type']
                    if ref_data_type == 'stn' or sim_data_type == 'stn':
                        ref_varname = self.ref[f'{selected_item}'][f'{ref_source}_varname']
                        sim_varname = self.sim[f'{selected_item}'][f'{sim_source}_varname']

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
                    if score in ['RMSE', 'CRMSD', 'MSE', 'ubRMSE', 'nRMSE', 'mean_absolute_error', 'ssq', 've',
                                 'absolute_percent_bias']:
                        min_value = min_value * 0 - 0.2
                    elif score in ['NSE', 'LNSE', 'ubNSE', 'rNSE', 'wNSE', 'wsNSE']:
                        max_value = max_value * 0 + 0.2
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
        # st.json(option,expanded=False)
        make_scenarios_comparison_Ridgeline_Plot(option, selected_item, ref_source,
                                                 sim_sources, datasets_filtered,
                                                 score)
