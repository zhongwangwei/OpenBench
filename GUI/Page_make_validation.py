# -*- coding: utf-8 -*-
import glob, shutil
import os
import sys
import itertools
import platform
import posixpath
from posixpath import normpath
import time
import streamlit as st
from PIL import Image
from io import StringIO
from collections import ChainMap
import xarray as xr
import numpy as np
from Namelist_lib.namelist_read import NamelistReader
from Namelist_lib.namelist_info import initial_setting
from Namelist_lib.find_path import FindPath


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


class make_initional:
    def __init__(self, initial):
        self.author = "Qingchen Xu/xuqingchen0@gmail.com"
        self.initial = initial
        self.classification = initial.classification()
        self.sim_info = initial.sim_info()
        self.ref_source = initial.ref_source()
        self.ref_info = initial.ref_info()
        # ------------------------
        # self.main = initial.main()
        # self.ref = initial.ref()
        # self.sim = initial.sim()
        # ----------------------------
        self.generals = initial.generals()
        self.evaluation_items = initial.evaluation_items()
        self.metrics = initial.metrics()
        self.scores = initial.scores()
        self.comparisons = initial.comparisons()
        self.statistics = initial.statistics()
        self.nl = NamelistReader()
        self.path_finder = FindPath()

    def find_paths_in_dict(self, d):

        if platform.system() == "Windows":
            sep = '\\'
        else:
            sep = posixpath.sep

        paths = []
        for key, value in d.items():
            if isinstance(value, dict):
                paths.extend(self.find_paths_in_dict(value))
            elif isinstance(value, str):
                if 'path' in key.lower() or '/' in value or '\\' in value:
                    if sep == '\\':
                        path.replace(os.sep, "\\")
                        d[key] = value.replace(os.sep, '\\')
                    else:
                        d[key] = value.replace(os.sep, "/")
                    paths.append((key, value))
        return paths

    def home(self):
        st.subheader("Welcome to the Evaluation page", divider=True)
        st.write(
            "##### :green[Choose whether to upload a file or create a new initial setup, and please press Next step button]")  # 请选择是上传文件，还是新建一个初始设置
        genre = st.radio(
            " What's your choice?", ["***Upload***", "***Setting***"],
            captions=["Upload your case and change a little.", "No need to upload, staring setting."], index=1,
            horizontal=True, label_visibility='collapsed')

        def define_step1():
            st.session_state.step1_general = True

        if genre == '***Upload***':
            if 'main_nml' not in st.session_state:
                st.session_state['main_nml'] = None
            st.session_state['main_nml'] = self.path_finder.get_file(st.session_state['main_nml'], 'upload_file', 'nml',
                                                                     [None, None])

            if st.session_state.main_nml is not None:
                st.code(f'Upload Namelist: {st.session_state.main_nml}', language='shell', wrap_lines=True)

            if st.session_state['main_nml']:
                if 'main_change' in st.session_state:
                    del st.session_state['main_change']
                if not os.path.exists(st.session_state['main_nml']):
                    e = FileNotFoundError(f'[Errno 2] No such file or directory: "{st.session_state["main_nml"]}"')
                    st.exception(e)
                    check = True
                else:
                    main_data = self.nl.read_namelist(st.session_state['main_nml'])
                    ref_path = main_data['general']["reference_nml"]
                    sim_path = main_data['general']["simulation_nml"]
                    if (not os.path.exists(ref_path)) | (not os.path.exists(sim_path)):
                        if not os.path.exists(ref_path):
                            e = FileNotFoundError(f'[Errno 2] No such file or directory: "{ref_path}"')
                            st.exception(e)
                        if not os.path.exists(sim_path):
                            e = FileNotFoundError(f'[Errno 2] No such file or directory: "{sim_path}"')
                            st.exception(e)
                        check = True
                    else:
                        check = self.__initial_nml(main_data)
                    st.session_state.step1_initial = 'Upload'

            else:
                st.warning('Please input your file path!')
                check = True

        elif genre == '***Setting***':
            if 'main_change' in st.session_state:
                del st.session_state['main_change']
            check = False
            st.session_state.step1_initial = 'Setting'
            if st.session_state:
                for key, value in st.session_state.items():
                    if key in ['main_path', 'validation_path', 'main_data', 'sim_data', 'ref_data']:
                        del st.session_state[key]
                    elif isinstance(value, int) & ('count' in key):
                        st.session_state[key] = 0
                    elif isinstance(value, bool) & (key not in ['switch_button']):
                        st.session_state[key] = False
                    st.session_state.main_path = os.getcwd()
                    st.session_state.validation_path = os.path.abspath(os.path.join(os.getcwd(), 'CoLM-Evaluation'))
                    st.session_state.main_data = self.initial.main()
                    st.session_state.sim_data = self.initial.sim()
                    st.session_state.ref_data = self.initial.ref()
        else:
            st.write("You didn\'t select.")
            check = True
        st.divider()

        col1, col2, col3, col4 = st.columns(4)
        with col4:
            st.button('Next step :soon: ', on_click=define_step1, disabled=check)

    def __initial_nml(self, main_data):
        ref_path = main_data['general']["reference_nml"]
        sim_path = main_data['general']["simulation_nml"]
        check = True

        st.session_state.main_data = self.initial.main()
        st.session_state.ref_data = self.initial.ref()
        st.session_state.sim_data = self.initial.sim()

        st.session_state.main_data = main_data
        st.session_state.generals = main_data['general']
        st.session_state.evaluation_items = main_data['evaluation_items']
        st.session_state.metrics = main_data['metrics']
        st.session_state.scores = main_data['scores']
        st.session_state.comparisons = main_data['comparisons']
        st.session_state.statistics = main_data['statistics']
        if isinstance(main_data['general']['compare_tzone'], str):
            if main_data['general']['compare_tzone'] == 'UTC':
                st.session_state.generals['compare_tzone'] = 0.0
        elif isinstance(main_data['general']['compare_tzone'], int):
            st.session_state.generals['compare_tzone'] = float(main_data['general']['compare_tzone'])

        if self.__upload_ref_check(main_data, ref_path):
            st.session_state.ref_data = self.nl.read_namelist(ref_path)
            check = False
        else:
            check = True
        if self.__upload_sim_check(main_data, sim_path):
            st.session_state.sim_data = self.nl.read_namelist(sim_path)
            check = False
        else:
            check = True

        st.session_state.step1 = True
        st.session_state.step1_main_nml = True
        st.session_state.step2 = True
        st.session_state.step2_ref_nml = True
        st.session_state.step3 = True
        st.session_state.step3_sim_nml = True
        return check

    def __upload_ref_check(self, main_data, ref_path):
        ref_data = self.nl.read_namelist(ref_path)
        selected_items = [k for k, v in main_data['evaluation_items'].items() if v]

        error = 0
        sources = []

        for key in selected_items:
            if isinstance(ref_data['general'][f"{key}_ref_source"], str): ref_data['general'][f"{key}_ref_source"] = [
                ref_data['general'][f"{key}_ref_source"]]
            for value in ref_data['general'][f"{key}_ref_source"]:
                if value not in sources:
                    sources.append(value)

        for source in sources:
            if not os.path.exists(ref_data['def_nml'][source]):
                e = FileNotFoundError(f'[Errno 2] No such file or directory: "{ref_data["def_nml"][source]}"')
                st.exception(e)
                error = +1

        if error == 0:
            ref_sources = self.nl.read_namelist('./GUI/Namelist_lib/Reference_lib.nml')
            for selected_item in selected_items:
                if isinstance(ref_sources['general'][selected_item], str): ref_sources['general'][selected_item] = [
                    ref_sources['general'][selected_item]]
                for item in ref_data['general'][f"{selected_item}_ref_source"]:
                    if item not in ref_sources['general'][selected_item]:
                        ref_sources['general'][selected_item].append(item)
                        ref_sources['def_nml'][item] = ref_data['def_nml'][item]

            with open('./GUI/Namelist_lib/Reference_lib.nml', 'w') as f1:
                lines = []
                end_line = "/\n\n\n"
                lines.append("&general\n")
                max_key_length = max(len(key) for key in ref_sources['general'].keys())
                for key in list(ref_sources['general'].keys()):
                    value = ref_sources['general'][f'{key}']
                    if isinstance(value, str): value = [value]
                    lines.append(f"    {key:<{max_key_length}} = {', '.join(value)}\n")
                lines.append(end_line)

                lines.append("&def_nml\n")
                max_key_length = max(len(key) for key in ref_sources['def_nml'].keys())
                for key in list(ref_sources['def_nml'].keys()):
                    lines.append(f"    {key:<{max_key_length}} = {ref_sources['def_nml'][f'{key}']}\n")
                lines.append(end_line)
                for line in lines:
                    f1.write(line)
            return True
        else:
            return False

    def __upload_sim_check(self, main_data, sim_path):
        sim_data = self.nl.read_namelist(sim_path)
        selected_items = [k for k, v in main_data['evaluation_items'].items() if v]
        error = 0
        sources = []
        Mods = {}
        for key in selected_items:
            if isinstance(sim_data['general'][f"{key}_sim_source"], str): sim_data['general'][f"{key}_sim_source"] = [
                sim_data['general'][f"{key}_sim_source"]]
            for value in sim_data['general'][f"{key}_sim_source"]:
                if value not in sources:
                    sources.append(value)

        for source in sources:
            if not os.path.exists(sim_data['def_nml'][source]):
                e = FileNotFoundError(f'[Errno 2] No such file or directory: "{sim_data["def_nml"][source]}"')
                st.exception(e)
                error = +1
            else:
                Mod_path = self.nl.read_namelist(sim_data['def_nml'][source])['general']['model_namelist']
                Mod = self.nl.read_namelist(Mod_path)['general']['model']
                if Mod not in Mods:
                    Mods[Mod] = Mod_path

        if error == 0:
            sim_sources = self.nl.read_namelist('./GUI/Namelist_lib/Simulation_lib.nml')
            if isinstance(sim_sources['general']['Case_lib'], str): sim_sources['general']['Case_lib'] = [
                sim_sources['general']['Case_lib']]
            for selected_item in selected_items:
                if isinstance(sim_data['general'][f"{selected_item}_sim_source"], str): sim_data['general'][
                    f"{selected_item}_sim_source"] = [
                    sim_data['general'][f"{selected_item}_sim_source"]]
                for item in sim_data['general'][f"{selected_item}_sim_source"]:
                    if item not in sim_sources['general']['Case_lib']:
                        sim_sources['general']['Case_lib'].append(item)
                        sim_sources['def_nml'][item] = sim_data['def_nml'][item]
            for Mod in Mods:
                if Mod not in sim_sources['def_Mod']:
                    sim_sources['def_Mod'][Mod] = Mods[Mod]

            with open('./GUI/Namelist_lib/Simulation_lib.nml', 'w') as f1:
                lines = []
                end_line = "/\n\n\n"
                lines.append("&general\n")
                max_key_length = max(len(key) for key in sim_sources['general'].keys())
                for key in list(sim_sources['general'].keys()):
                    value = sim_sources['general'][f'{key}']
                    if isinstance(value, str): value = [value]
                    lines.append(f"    {key:<{max_key_length}} = {', '.join(value)}\n")
                lines.append(end_line)

                lines.append("&def_nml\n")
                max_key_length = max(len(key) for key in sim_sources['def_nml'].keys())
                for key in list(sim_sources['def_nml'].keys()):
                    lines.append(f"    {key:<{max_key_length}} = {sim_sources['def_nml'][f'{key}']}\n")
                lines.append(end_line)

                lines.append("&def_Mod\n")
                max_key_length = max(len(key) for key in sim_sources['def_Mod'].keys())
                for key in list(sim_sources['def_Mod'].keys()):
                    lines.append(f"    {key:<{max_key_length}} = {sim_sources['def_Mod'][f'{key}']}\n")
                lines.append(end_line)
                for line in lines:
                    f1.write(line)
            return True
        else:
            return False

    def step1_general(self):
        if 'main_change' not in st.session_state:
            st.session_state.main_change = {'general': False,
                                            'metrics': False,
                                            'scores': False,
                                            'evaluation': False,
                                            'comparisons': False,
                                            'statistics': False,
                                            }
        General = self.generals
        if st.session_state.generals:
            General = st.session_state.generals
        st.subheader('General setting Info....', divider=True)

        def compare_tres_index(compare_tres_value):
            my_list = ['hour', 'day', 'month', 'year']
            index = my_list.index(compare_tres_value)
            return index

        def data_editor_change(key, editor_key):
            a = General[key]
            General[key] = st.session_state[key]
            if a != st.session_state[key]:
                st.session_state.main_change['general'] = True

        col1, col2, col3, col4 = st.columns(4)
        col1.write('###### :green[Basename]')
        col2.write('###### :green[Time Resolution]')
        col3.write('###### :green[Time zone]')
        col4.write('###### :green[Grid Resolution]')

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.text_input('Basename: ', value=General["basename"],
                          key="basename",
                          on_change=data_editor_change,  # callback function
                          args=("basename", "basename"),
                          placeholder="Set your basename...",
                          label_visibility='collapsed')
        with col2:
            st.selectbox(f'Compare Time Resolution: ',
                         options=('hour', 'day', 'month', 'year'),
                         index=compare_tres_index(General["compare_tim_res"].lower()),
                         key="compare_tim_res",
                         on_change=data_editor_change,  # callback function
                         args=("compare_tim_res", "compare_tim_res"),
                         placeholder=f"Set your Time Resolution (default={General['compare_tim_res']})...",
                         label_visibility='collapsed')
        with col3:
            st.number_input("Compare Time zone: ", value=General["compare_tzone"],
                            key="compare_tzone",
                            on_change=data_editor_change,  # callback function
                            args=("compare_tzone", "compare_tzone"),
                            min_value=-12.0,
                            max_value=12.0,
                            label_visibility='collapsed')
        with col4:
            st.number_input("Compare Geo Resolution: ", value=General["compare_grid_res"],
                            key="compare_grid_res",
                            on_change=data_editor_change,  # callback function
                            args=("compare_grid_res", "compare_grid_res"),
                            min_value=0.0,
                            placeholder="Compare Geo Resolution...",
                            label_visibility='collapsed')
        st.divider()
        st.write('##### :blue[Extent]')  # 范围
        col1, col2, col3, col4 = st.columns(4)
        col1.write('###### :green[Max latitude]')
        col2.write('###### :green[Min latitude]')
        col3.write('###### :green[Max Longitude]')
        col4.write('###### :green[Min Longitude]')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.number_input("Max latitude: ", value=float(General["max_lat"]),
                            key="max_lat",
                            on_change=data_editor_change,  # callback function
                            args=("max_lat", "max_lat"),
                            min_value=-90.0,
                            max_value=90.0,
                            label_visibility='collapsed')
        with col2:
            st.number_input("Min latitude: ", value=float(General["min_lat"]),
                            key="min_lat",
                            on_change=data_editor_change,  # callback function
                            args=("min_lat", "min_lat"),
                            min_value=-90.0,
                            max_value=90.0,
                            label_visibility='collapsed')
        with col3:
            st.number_input("Max Longitude: ", value=float(General["max_lon"]),
                            key="max_lon",
                            on_change=data_editor_change,  # callback function
                            args=("max_lon", "max_lon"),
                            min_value=-180.0,
                            max_value=180.0,
                            label_visibility='collapsed')
        with col4:
            st.number_input("Min Longitude: ", value=float(General["min_lon"]),
                            key="min_lon",
                            on_change=data_editor_change,  # callback function
                            args=("min_lon", "min_lon"),
                            min_value=-180.0,
                            max_value=180.0,
                            label_visibility='collapsed')

        st.divider()
        st.write('###### :green[Base case directory]')
        if General["basedir"] is None or General["basedir"] == '': General["basedir"] = '/'
        if os.path.isdir(os.path.abspath(General["basedir"])):
            General["basedir"] = os.path.abspath(General["basedir"])
        else:
            General["basedir"] = '/'
        General["basedir"] = self.path_finder.find_path(General["basedir"], "basedir", ['main_change', 'general'])
        st.code(f"Current Path: {General['basedir']}", language='shell', wrap_lines=True)

        col1, col2 = st.columns((1, 2))
        with col1:
            st.write('###### :green[Num Cores]')
        with col2:
            st.write('###### :green[Weight for metrics and scores]')

        def Weight_index(Weight_value):
            my_list = ['none', 'area', 'mass']
            index = my_list.index(Weight_value)
            return index

        col1, col2, col3 = st.columns(3)
        with col1:
            st.number_input("Num Cores: ", value=General["num_cores"],
                            key="num_cores",
                            on_change=data_editor_change,  # callback function
                            args=("num_cores", "num_cores"),
                            placeholder="how many core will be used in Parallel computing...",
                            format='%d', label_visibility='collapsed')
        with col2:
            st.selectbox(f'weight for metrics and scores',
                         options=('None', 'area', 'mass'),
                         index=Weight_index(General["weight"].lower()),
                         key="weight",
                         on_change=data_editor_change,  # callback function
                         args=("weight", "weight"),
                         placeholder=f"Set your weight (default={General['weight']})...",
                         label_visibility='collapsed')
        st.write(':information_source: How many cores will be used in Parallel computing? '
                 'Recommend core number is 5.')

        col1, col2, col3 = st.columns(3)
        col1.write('###### :green[Start year]')
        col2.write('###### :green[End year]')
        col3.write('###### :green[Minimum year]')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.number_input("Start year: ", format='%04d', value=General["syear"], step=int(1),
                            key="syear",
                            on_change=data_editor_change,  # callback function
                            args=("syear", "syear"),
                            placeholder="Start year...", label_visibility='collapsed')

        with col2:
            st.number_input("End year: ", format='%04d', value=General["eyear"], step=int(1),
                            key="eyear",
                            on_change=data_editor_change,  # callback function
                            args=("eyear", "eyear"),
                            placeholder="End year...", label_visibility='collapsed')

        with col3:
            st.number_input("Minimum year: ", value=General["min_year"], step=1.0,
                            key="min_year",
                            on_change=data_editor_change,  # callback function
                            args=("min_year", "min_year"),
                            placeholder="Minimum year...", label_visibility='collapsed')
        st.write('')
        col1, col2, col3 = st.columns(3)
        col1.checkbox('Running Evaluation? ', value=General['evaluation'],
                      key="evaluation",
                      on_change=data_editor_change,  # callback function
                      args=("evaluation", "evaluation"),
                      )
        col2.checkbox('Running Comparison? ', value=General['comparison'],
                      key="comparison",
                      on_change=data_editor_change,  # callback function
                      args=("comparison", "comparison"),
                      )
        col3.checkbox('Debug Model? ', value=General['debug_mode'],
                      key="debug_mode",
                      on_change=data_editor_change,  # callback function
                      args=("debug_mode", "debug_mode")
                      )
        General['statistics'] = False

        col1.checkbox('IGBP groupby? ', value=General['IGBP_groupby'],
                      key="IGBP_groupby",
                      on_change=data_editor_change,  # callback function
                      args=("IGBP_groupby", "IGBP_groupby"),
                      )
        col2.checkbox('PFT groupby? ', value=General['PFT_groupby'],
                      key="PFT_groupby",
                      on_change=data_editor_change,  # callback function
                      args=("PFT_groupby", "PFT_groupby"),
                      )
        col3.checkbox('Unified mask? ', value=General['unified_mask'],
                      key="unified_mask",
                      on_change=data_editor_change,  # callback function
                      args=("unified_mask", "unified_mask")
                      )

        # st.write(
        # ':information_source: You can choose which one to run, if only :point_up: running comparison you should prepare the files first.')
        st.divider()

        # showing for detail ===============================================
        st.session_state.casepath = f'{os.path.join(General["basedir"], General["basename"])}'
        if not General["reference_nml"] or st.session_state.step1_initial == 'Setting':
            General["reference_nml"] = f"{st.session_state.openbench_path}/nml/ref-{General['basename']}.nml"
        if not General["simulation_nml"] or st.session_state.step1_initial == 'Setting':
            General["simulation_nml"] = f"{st.session_state.openbench_path}/nml/sim-{General['basename']}.nml"
        if not General["statistics_nml"] or st.session_state.step1_initial == 'Setting':
            General["statistics_nml"] = f"{st.session_state.openbench_path}/nml/stats-{General['basename']}.nml"
        General["figure_nml"] = f"{st.session_state.openbench_path}/nml/figlib.nml"
        paths = self.find_paths_in_dict(General)

        st.session_state.step1_main_check_general = self.__step1_check_general(General)

        st.session_state['generals'] = General

        def define_step1():
            st.session_state.step1_general = False

        def define_step2():
            st.session_state.step1_metrics = True

        col1, col2, col5, col4 = st.columns(4)
        with col1:
            st.button(':back: Previous step', on_click=define_step1, help='Beck to Home page')

        with col4:
            st.button('Next step :soon: ', on_click=define_step2, help='Go to Metrics page')

    def __step1_check_general(self, Generals):

        check_state = 0
        for key, value in Generals.items():
            if isinstance(value, str):
                if len(value) <= 1:
                    st.warning(f'{key} should be a string longer than one, please check {key}.', icon="⚠")
                    check_state += 1
            elif isinstance(value, bool):
                if value not in [True, False]:
                    st.warning(f'{key} should be True or False, please check {key}.', icon="⚠")
                    check_state += 1
            elif isinstance(value, int):
                if key in ['num_cores', 'syear', 'eyear']:
                    if not isinstance(value, int):
                        st.warning(f'{key} should be in integer format, please check {key}.', icon="⚠")
                        check_state += 1
            elif isinstance(value, float):
                if key in ['min_year', 'max_lat', 'min_lat', 'max_lon', 'min_lon']:
                    if not isinstance(value, float) or not (value == value):  # 检查是否是 NaN
                        st.warning(f'{key} should be in float format, please check {key}.', icon="⚠")
                        check_state += 1
            elif value is None:
                if key in ['compare_tres', 'compare_tzone']:
                    st.warning(f'{key} It can not be empty, please check {key}.', icon="⚠")
                    check_state += 1
            else:
                st.warning(f'Unsupported data types {type(value)} for {key}, please check {key}.', icon="⚠")
                check_state += 1
        if Generals['eyear'] < Generals['syear']:
            st.warning(f" End year should be larger than Start year, please check.", icon="⚠")
            check_state += 1

        def get_os_type(path):
            os_type = False
            if path:
                if ('\\' in path) and (':' in path):
                    if sys.platform.startswith('win'):
                        os_type = True
                elif ('/' in path) & (path[0] == '/'):
                    if sys.platform.startswith('linux') | (not sys.platform.startswith('win')):
                        os_type = True
                elif (('./' in path) | ('.\\' in path)) & (path[0] == '.'):
                    os_type = True
                return os_type
            else:
                return os_type

        basedir_type = get_os_type(Generals['basedir'])
        if not basedir_type:
            check_state += 1
            st.error('Please make sure your path is right!', icon="⚠")

        if check_state > 0:
            return False
        if check_state == 0:
            return True
        # return step1_main_check~

    # =============================================

    def step1_metrics(self):

        metrics = self.metrics
        scores = self.scores

        if st.session_state.metrics:
            metrics = st.session_state.metrics
        if st.session_state.scores:
            scores = st.session_state.scores

        def metrics_editor_change(key, editor_key):
            metrics[key] = st.session_state[key]
            st.session_state.main_change['metrics'] = True

        def scores_editor_change(key, editor_key):
            scores[key] = st.session_state[key]
            st.session_state.main_change['scores'] = True

        st.subheader('Metrics and Scores setting ....', divider=True)
        st.write('##### :orange[Select Metrics]')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.checkbox("Absolute Percent Bias (absolute_percent_bias)", key="absolute_percent_bias",
                        on_change=metrics_editor_change, args=("absolute_percent_bias", "absolute_percent_bias"),
                        value=metrics["absolute_percent_bias"])
            st.checkbox("Bias (bias)", key="bias", on_change=metrics_editor_change, args=("bias", "bias"),
                        value=metrics["bias"])
            st.checkbox("correlation coefficient (correlation)", key="correlation", on_change=metrics_editor_change,
                        args=("correlation", "correlation"), value=metrics["correlation"])
            st.checkbox("correlation coefficient R2 (correlation_R2)", key="correlation_R2",
                        on_change=metrics_editor_change,
                        args=("correlation_R2", "correlation_R2"), value=metrics["correlation_R2"])
            st.checkbox("Index of agreement (index_agreement)", key="index_agreement", on_change=metrics_editor_change,
                        args=("index_agreement", "index_agreement"), value=metrics["index_agreement"])
        with col2:
            st.checkbox("Kling-Gupta Efficiency (KGE)", key="KGE", on_change=metrics_editor_change, args=("KGE", "KGE"),
                        value=metrics["KGE"])
            st.checkbox("Likelihood (L)", key="L", on_change=metrics_editor_change, args=("L", "L"), value=metrics["L"])
            st.checkbox("Mean Absolute Error (mean_absolute_error)", key="mean_absolute_error",
                        on_change=metrics_editor_change,
                        args=("mean_absolute_error", "mean_absolute_error"), value=metrics["mean_absolute_error"])
            st.checkbox("Mean Squared Error (MSE)", key="MSE", on_change=metrics_editor_change, args=("MSE", "MSE"),
                        value=metrics["MSE"])
        with col3:
            st.checkbox("Nash Sutcliffe efficiency coefficient (NSE)", key="NSE", on_change=metrics_editor_change,
                        args=("NSE", "NSE"), value=metrics["NSE"])
            st.checkbox("Normalized Kling-Gupta Efficiency (KGESS)", key="KGESS", on_change=metrics_editor_change,
                        args=("KGESS", "KGESS"), value=metrics["KGESS"])
            st.checkbox("Percent Bias (percent_bias)", key="percent_bias", on_change=metrics_editor_change,
                        args=("percent_bias", "percent_bias"), value=metrics["percent_bias"])
            st.checkbox("Root Mean Squared Error (RMSE)", key="RMSE", on_change=metrics_editor_change,
                        args=("RMSE", "RMSE"),
                        value=metrics["RMSE"])

        with st.expander("More metrics", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.checkbox("Centered Root Mean Square Difference (CRMSD)", key="CRMSD",
                            on_change=metrics_editor_change,
                            args=("CRMSD", "CRMSD"), value=metrics["CRMSD"])
                st.checkbox("correlation coefficient R2 (ubcorrelation_R2)", key="ubcorrelation_R2",
                            on_change=metrics_editor_change, args=("ubcorrelation_R2", "ubcorrelation_R2"),
                            value=metrics["ubcorrelation_R2"])
                st.checkbox("Graphical Goodness of Fit (ggof)", key="ggof", on_change=metrics_editor_change,
                            args=("ggof", "ggof"), value=metrics["ggof"])
                st.checkbox("SMPI (SMPI)", key="SMPI", on_change=metrics_editor_change, args=("SMPI", "SMPI"),
                            value=metrics["SMPI"])
                st.checkbox("Kappa coefficient (kappa_coeff)", key="kappa_coeff", on_change=metrics_editor_change,
                            args=("kappa_coeff", "kappa_coeff"), value=metrics["kappa_coeff"])
                st.checkbox("Kling-Gupta Efficiency for low values (KGElf)", key="KGElf",
                            on_change=metrics_editor_change,
                            args=("KGElf", "KGElf"), value=metrics["KGElf"])
                st.checkbox("Kling-Gupta Efficiency with knowable-moments (KGEkm)", key="KGEkm",
                            on_change=metrics_editor_change,
                            args=("KGEkm", "KGEkm"), value=metrics["KGEkm"])
                st.checkbox("Modified Index of Agreement (md)", key="md", on_change=metrics_editor_change,
                            args=("md", "md"),
                            value=metrics["md"])
                st.checkbox("Modified Nash-Sutcliffe efficiency (mNSE)", key="mNSE", on_change=metrics_editor_change,
                            args=("mNSE", "mNSE"), value=metrics["mNSE"])
                st.checkbox("natural logarithm of NSE coefficient (LNSE)", key="LNSE", on_change=metrics_editor_change,
                            args=("LNSE", "LNSE"), value=metrics["LNSE"])
                st.checkbox("Non-parametric version of the Kling-Gupta Efficiency (KGEnp)", key="KGEnp",
                            on_change=metrics_editor_change, args=("KGEnp", "KGEnp"), value=metrics["KGEnp"])
                st.checkbox("Normalized Root Mean Square Error (nrmse)", key="nrmse", on_change=metrics_editor_change,
                            args=("nrmse", "nrmse"), value=metrics["nrmse"])
                st.checkbox("Numerical Goodness-of-fit measures (gof)", key="gof", on_change=metrics_editor_change,
                            args=("gof", "gof"), value=metrics["gof"])
                st.checkbox("Percent Bias in the Slope of the Midsegment of the Flow Duration Curve (pbiasfdc)",
                            key="pbiasfdc",
                            on_change=metrics_editor_change, args=("pbiasfdc", "pbiasfdc"), value=metrics["pbiasfdc"])
            with col2:
                st.checkbox("Percent bias of flows ≤ Q30(Yilmaz et al., 2008) (PBIAS_LF)", key="PBIAS_LF",
                            on_change=metrics_editor_change, args=("PBIAS_LF", "PBIAS_LF"), value=metrics["PBIAS_LF"])
                st.checkbox("Percent bias of flows ≥ Q98 (Yilmaz et al., 2008) (PBIAS_HF)", key="PBIAS_HF",
                            on_change=metrics_editor_change, args=("PBIAS_HF", "PBIAS_HF"), value=metrics["PBIAS_HF"])
                st.checkbox("Ratio of RMSE to the standard deviation of the observations (rsr)", key="rsr",
                            on_change=metrics_editor_change, args=("rsr", "rsr"), value=metrics["rsr"])
                st.checkbox("Ratio of Standard Deviations (rSD)", key="rSD", on_change=metrics_editor_change,
                            args=("rSD", "rSD"),
                            value=metrics["rSD"])
                st.checkbox("Relative Index of Agreement (rd)", key="rd", on_change=metrics_editor_change,
                            args=("rd", "rd"),
                            value=metrics["rd"])
                st.checkbox("Relative Nash-Sutcliffe efficiency (rNSE)", key="rNSE", on_change=metrics_editor_change,
                            args=("rNSE", "rNSE"), value=metrics["rNSE"])
                st.checkbox("Relative variability (amplitude ratio) (rv)", key="rv", on_change=metrics_editor_change,
                            args=("rv", "rv"), value=metrics["rv"])
                st.checkbox("Spearman’s rank correlation coefficient (rSpearman)", key="rSpearman",
                            on_change=metrics_editor_change, args=("rSpearman", "rSpearman"),
                            value=metrics["rSpearman"])
                st.checkbox("Split Kling-Gupta Efficiency (sKGE)", key="sKGE", on_change=metrics_editor_change,
                            args=("sKGE", "sKGE"), value=metrics["sKGE"])
                st.checkbox("Sum of the Squared Residuals (ssq)", key="ssq", on_change=metrics_editor_change,
                            args=("ssq", "ssq"),
                            value=metrics["ssq"])
                st.checkbox(
                    "the average width of the given uncertainty bounds divided by the standard deviation of the observations. (rfactor)",
                    key="rfactor", on_change=metrics_editor_change, args=("rfactor", "rfactor"),
                    value=metrics["rfactor"])

            with col3:
                st.checkbox("the bias of the amplitude value (pc_ampli)", key="pc_ampli",
                            on_change=metrics_editor_change,
                            args=("pc_ampli", "pc_ampli"), value=metrics["pc_ampli"])
                st.checkbox("the bias of the maximum value (pc_max)", key="pc_max", on_change=metrics_editor_change,
                            args=("pc_max", "pc_max"), value=metrics["pc_max"])
                st.checkbox("the bias of the minimum value (pc_min)", key="pc_min", on_change=metrics_editor_change,
                            args=("pc_min", "pc_min"), value=metrics["pc_min"])
                st.checkbox("the percent of observations that are within the given uncertainty bounds. (pfactor)",
                            key="pfactor",
                            on_change=metrics_editor_change, args=("pfactor", "pfactor"), value=metrics["pfactor"])
                st.checkbox("Unbiased correlation (ubcorrelation)", key="ubcorrelation",
                            on_change=metrics_editor_change,
                            args=("ubcorrelation", "ubcorrelation"), value=metrics["ubcorrelation"])
                st.checkbox("Unbiased Kling-Gupta Efficiency (ubKGE)", key="ubKGE", on_change=metrics_editor_change,
                            args=("ubKGE", "ubKGE"), value=metrics["ubKGE"])
                st.checkbox("Unbiased Nash Sutcliffe efficiency coefficient (ubNSE)", key="ubNSE",
                            on_change=metrics_editor_change, args=("ubNSE", "ubNSE"), value=metrics["ubNSE"])
                st.checkbox("Unbiased Root Mean Squared Error (ubRMSE)", key="ubRMSE", on_change=metrics_editor_change,
                            args=("ubRMSE", "ubRMSE"), value=metrics["ubRMSE"])
                st.checkbox("Valid Indexes (valindex)", key="valindex", on_change=metrics_editor_change,
                            args=("valindex", "valindex"), value=metrics["valindex"])
                st.checkbox("Volumetric Efficiency (ve)", key="ve", on_change=metrics_editor_change, args=("ve", "ve"),
                            value=metrics["ve"])
                st.checkbox("Weighted Nash-Sutcliffe efficiency (wNSE)", key="wNSE", on_change=metrics_editor_change,
                            args=("wNSE", "wNSE"), value=metrics["wNSE"])
                st.checkbox("Weighted seasonal Nash-Sutcliffe Efficiency (wsNSE)", key="wsNSE",
                            on_change=metrics_editor_change,
                            args=("wsNSE", "wsNSE"), value=metrics["wsNSE"])

        st.divider()
        st.write('##### :orange[Select Scores]')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.checkbox('Bias Score from ILAMB (nBiasScore)',
                        key="nBiasScore",
                        on_change=scores_editor_change,
                        args=("nBiasScore", "nBiasScore"),
                        value=scores["nBiasScore"])
            st.checkbox('Interannual Variability Score from ILAMB (nIavScore)',
                        key="nIavScore",
                        on_change=scores_editor_change,
                        args=("nIavScore", "nIavScore"),
                        value=scores["nIavScore"])
            st.checkbox('Overall Score from ILAMB (Overall_Score)',
                        key="Overall_Score",
                        on_change=scores_editor_change,
                        args=("Overall_Score", "Overall_Score"),
                        value=scores["Overall_Score"])
        with col2:
            st.checkbox('RMSE Score from ILAMB (nRMSEScore)',
                        key="nRMSEScore",
                        on_change=scores_editor_change,
                        args=("nRMSEScore", "nRMSEScore"),
                        value=scores["nRMSEScore"])
            st.checkbox('Spatial distribution score (nSpatialScore)',
                        key="nSpatialScore",
                        on_change=scores_editor_change,
                        args=("nSpatialScore", "nSpatialScore"),
                        value=scores["nSpatialScore"])
        with col3:
            st.checkbox('Phase Score from ILAMB (nPhaseScore)',
                        key="nPhaseScore",
                        on_change=scores_editor_change,
                        args=("nPhaseScore", "nPhaseScore"),
                        value=scores["nPhaseScore"])
            st.checkbox('The Ideal Point score',
                        key="The_Ideal_Point_score",
                        on_change=scores_editor_change,
                        args=("The_Ideal_Point_score", "The_Ideal_Point_score"),
                        value=scores["The_Ideal_Point_score"])

        st.session_state.step1_main_check_metrics_scores = self.__step1_check_metrics_scores(metrics, scores)
        st.session_state.metrics = metrics
        st.session_state.scores = scores

        def define_step1():
            st.session_state.step1_metrics = False

        def define_step2():
            st.session_state.step1_evaluation = True

        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.button(':back: Previous step', on_click=define_step1, help='Beck to General page')
        with col4:
            st.button('Next step :soon: ', on_click=define_step2, help='Go to Evaluation Items page')

    def __step1_check_metrics_scores(self, metrics, scores):
        check_dict = ChainMap(metrics, scores)
        check_state = 0

        metrics_all_false = True
        metrics_select = []
        scores_select = []
        for key, value in check_dict.items():
            if isinstance(value, bool):
                if value not in [True, False]:
                    st.warning(f'{key} should be True or False, please check {key}.', icon="⚠")
                    check_state += 1
                if value:
                    metrics_all_false = False
                    if key in metrics:
                        metrics_select.append(key)
                    if key in scores:
                        scores_select.append(key)
        if metrics_all_false:
            st.error(f'Please choose at least one Metrics or Scores!', icon="🚨")
            check_state += 1
        else:
            # with st.container(border=True):
            if len(metrics_select) >= 1:
                m_select = ", ".join(m for m in metrics_select)
                st.info(f'Make sure your select Metrics is: \n:red[{m_select}]', icon="ℹ️")
            if len(scores_select) >= 1:
                s_select = ", ".join(s for s in scores_select)
                st.info(f'Make sure your select Scores is: \n:red[{s_select}]', icon="ℹ️")
        # st.info(f"Make sure your selected Evaluation Item is:    \n{formatted_keys}", icon="ℹ️")
        if check_state > 0:
            return False
        if check_state == 0:
            return True
        # return check state~

    # ===============================================

    def step1_evaluation(self):
        self.__step1_set_Evaluation_Items(self.evaluation_items)

        def define_step1():
            st.session_state.step1_evaluation = False

        def define_step2():
            st.session_state.step1_comparison = True

        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.button(':back: Previous step', on_click=define_step1, help='Beck to seting metrisc and scores page')
        with col4:
            st.button('Next step :soon: ', on_click=define_step2, help='Go to Comaprison Items page')

    def __step1_set_Evaluation_Items(self, Evaluation_Items):
        check_list = []
        if st.session_state.evaluation_items:
            Evaluation_Items = st.session_state.evaluation_items

        def Evaluation_Items_editor_change(key, editor_key):
            Evaluation_Items[key] = st.session_state[key]
            st.session_state.main_change['evaluation'] = True

        st.subheader("Evaluation Items ....", divider=True)
        col1, col2 = st.columns(2)
        with col1:
            # st.subheader("", divider=True, )
            st.write('##### :blue[Ecosystem and Carbon Cycle]')
            st.checkbox("Gross Primary Productivity", key="Gross_Primary_Productivity",
                        on_change=Evaluation_Items_editor_change,
                        args=("Gross_Primary_Productivity", "Gross_Primary_Productivity"),
                        value=Evaluation_Items["Gross_Primary_Productivity"])
            st.checkbox("Ecosystem Respiration", key="Ecosystem_Respiration", on_change=Evaluation_Items_editor_change,
                        args=("Ecosystem_Respiration", "Ecosystem_Respiration"),
                        value=Evaluation_Items["Ecosystem_Respiration"])
            st.checkbox("Net Ecosystem Exchange", key="Net_Ecosystem_Exchange",
                        on_change=Evaluation_Items_editor_change,
                        args=("Net_Ecosystem_Exchange", "Net_Ecosystem_Exchange"),
                        value=Evaluation_Items["Net_Ecosystem_Exchange"])
            st.checkbox("Leaf Area Index", key="Leaf_Area_Index", on_change=Evaluation_Items_editor_change,
                        args=("Leaf_Area_Index", "Leaf_Area_Index"), value=Evaluation_Items["Leaf_Area_Index"])
            st.checkbox("Biomass", key="Biomass", on_change=Evaluation_Items_editor_change, args=("Biomass", "Biomass"),
                        value=Evaluation_Items["Biomass"])
            st.checkbox("Burned Area", key="Burned_Area", on_change=Evaluation_Items_editor_change,
                        args=("Burned_Area", "Burned_Area"), value=Evaluation_Items["Burned_Area"])
            st.checkbox("Soil Carbon", key="Soil_Carbon", on_change=Evaluation_Items_editor_change,
                        args=("Soil_Carbon", "Soil_Carbon"), value=Evaluation_Items["Soil_Carbon"])
            st.checkbox("Nitrogen Fixation", key="Nitrogen_Fixation", on_change=Evaluation_Items_editor_change,
                        args=("Nitrogen_Fixation", "Nitrogen_Fixation"), value=Evaluation_Items["Nitrogen_Fixation"])
            st.checkbox("Methane", key="Methane", on_change=Evaluation_Items_editor_change, args=("Methane", "Methane"),
                        value=Evaluation_Items["Methane"])
            st.checkbox("Veg Cover In Fraction", key="Veg_Cover_In_Fraction", on_change=Evaluation_Items_editor_change,
                        args=("Veg_Cover_In_Fraction", "Veg_Cover_In_Fraction"),
                        value=Evaluation_Items["Veg_Cover_In_Fraction"])
            st.checkbox("Leaf Greenness", key="Leaf_Greenness", on_change=Evaluation_Items_editor_change,
                        args=("Leaf_Greenness", "Leaf_Greenness"), value=Evaluation_Items["Leaf_Greenness"])

        with col2:
            # st.subheader(":blue[]", divider=True)
            st.write('##### :blue[Radiation and Energy Cycle]')
            st.checkbox("Net Radiation", key="Net_Radiation", on_change=Evaluation_Items_editor_change,
                        args=("Net_Radiation", "Net_Radiation"), value=Evaluation_Items["Net_Radiation"])
            st.checkbox("Latent Heat", key="Latent_Heat", on_change=Evaluation_Items_editor_change,
                        args=("Latent_Heat", "Latent_Heat"), value=Evaluation_Items["Latent_Heat"])
            st.checkbox("Sensible Heat", key="Sensible_Heat", on_change=Evaluation_Items_editor_change,
                        args=("Sensible_Heat", "Sensible_Heat"), value=Evaluation_Items["Sensible_Heat"])
            st.checkbox("Ground Heat", key="Ground_Heat", on_change=Evaluation_Items_editor_change,
                        args=("Ground_Heat", "Ground_Heat"), value=Evaluation_Items["Ground_Heat"])
            st.checkbox("Albedo", key="Albedo", on_change=Evaluation_Items_editor_change, args=("Albedo", "Albedo"),
                        value=Evaluation_Items["Albedo"])
            st.checkbox("Surface Upward SW Radiation", key="Surface_Upward_SW_Radiation",
                        on_change=Evaluation_Items_editor_change,
                        args=("Surface_Upward_SW_Radiation", "Surface_Upward_SW_Radiation"),
                        value=Evaluation_Items["Surface_Upward_SW_Radiation"])
            st.checkbox("Surface Upward LW Radiation", key="Surface_Upward_LW_Radiation",
                        on_change=Evaluation_Items_editor_change,
                        args=("Surface_Upward_LW_Radiation", "Surface_Upward_LW_Radiation"),
                        value=Evaluation_Items["Surface_Upward_LW_Radiation"])
            st.checkbox("Surface Net SW Radiation", key="Surface_Net_SW_Radiation",
                        on_change=Evaluation_Items_editor_change,
                        args=("Surface_Net_SW_Radiation", "Surface_Net_SW_Radiation"),
                        value=Evaluation_Items["Surface_Net_SW_Radiation"])
            st.checkbox("Surface Net LW Radiation", key="Surface_Net_LW_Radiation",
                        on_change=Evaluation_Items_editor_change,
                        args=("Surface_Net_LW_Radiation", "Surface_Net_LW_Radiation"),
                        value=Evaluation_Items["Surface_Net_LW_Radiation"])
            st.checkbox("Surface Soil Temperature", key="Surface_Soil_Temperature",
                        on_change=Evaluation_Items_editor_change,
                        args=("Surface_Soil_Temperature", "Surface_Soil_Temperature"),
                        value=Evaluation_Items["Surface_Soil_Temperature"])
            st.checkbox("Root Zone Soil Temperature", key="Root_Zone_Soil_Temperature",
                        on_change=Evaluation_Items_editor_change,
                        args=("Root_Zone_Soil_Temperature", "Root_Zone_Soil_Temperature"),
                        value=Evaluation_Items["Root_Zone_Soil_Temperature"])

        col1, col2 = st.columns((1, 2))
        with col1:
            st.subheader(":blue[]", divider=True)
            st.write('##### :blue[Forcings]')
            st.checkbox("Diurnal Temperature Range", key="Diurnal_Temperature_Range",
                        on_change=Evaluation_Items_editor_change,
                        args=("Diurnal_Temperature_Range", "Diurnal_Temperature_Range"),
                        value=Evaluation_Items["Diurnal_Temperature_Range"])
            st.checkbox("Diurnal Max Temperature", key="Diurnal_Max_Temperature",
                        on_change=Evaluation_Items_editor_change,
                        args=("Diurnal_Max_Temperature", "Diurnal_Max_Temperature"),
                        value=Evaluation_Items["Diurnal_Max_Temperature"])
            st.checkbox("Diurnal Min Temperature", key="Diurnal_Min_Temperature",
                        on_change=Evaluation_Items_editor_change,
                        args=("Diurnal_Min_Temperature", "Diurnal_Min_Temperature"),
                        value=Evaluation_Items["Diurnal_Min_Temperature"])
            st.checkbox("Surface Downward SW Radiation", key="Surface_Downward_SW_Radiation",
                        on_change=Evaluation_Items_editor_change,
                        args=("Surface_Downward_SW_Radiation", "Surface_Downward_SW_Radiation"),
                        value=Evaluation_Items["Surface_Downward_SW_Radiation"])
            st.checkbox("Surface Downward LW Radiation", key="Surface_Downward_LW_Radiation",
                        on_change=Evaluation_Items_editor_change,
                        args=("Surface_Downward_LW_Radiation", "Surface_Downward_LW_Radiation"),
                        value=Evaluation_Items["Surface_Downward_LW_Radiation"])
            st.checkbox("Surface Relative Humidity", key="Surface_Relative_Humidity",
                        on_change=Evaluation_Items_editor_change,
                        args=("Surface_Relative_Humidity", "Surface_Relative_Humidity"),
                        value=Evaluation_Items["Surface_Relative_Humidity"])
            st.checkbox("Surface Specific Humidity", key="Surface_Specific_Humidity",
                        on_change=Evaluation_Items_editor_change,
                        args=("Surface_Specific_Humidity", "Surface_Specific_Humidity"),
                        value=Evaluation_Items["Surface_Specific_Humidity"])
            st.checkbox("Precipitation", key="Precipitation", on_change=Evaluation_Items_editor_change,
                        args=("Precipitation", "Precipitation"), value=Evaluation_Items["Precipitation"])
            st.checkbox("Surface Air Temperature", key="Surface_Air_Temperature",
                        on_change=Evaluation_Items_editor_change,
                        args=("Surface_Air_Temperature", "Surface_Air_Temperature"),
                        value=Evaluation_Items["Surface_Air_Temperature"])

        with col2:
            st.subheader(":blue[]", divider=True)
            st.write('##### :blue[Hydrology Cycle]')
            col21, col22 = st.columns(2)
            with col21:
                st.checkbox("Evapotranspiration", key="Evapotranspiration", on_change=Evaluation_Items_editor_change,
                            args=("Evapotranspiration", "Evapotranspiration"),
                            value=Evaluation_Items["Evapotranspiration"])
                st.checkbox("Canopy Transpiration", key="Canopy_Transpiration",
                            on_change=Evaluation_Items_editor_change,
                            args=("Canopy_Transpiration", "Canopy_Transpiration"),
                            value=Evaluation_Items["Canopy_Transpiration"])
                st.checkbox("Canopy Interception", key="Canopy_Interception", on_change=Evaluation_Items_editor_change,
                            args=("Canopy_Interception", "Canopy_Interception"),
                            value=Evaluation_Items["Canopy_Interception"])
                st.checkbox("Ground Evaporation", key="Ground_Evaporation", on_change=Evaluation_Items_editor_change,
                            args=("Ground_Evaporation", "Ground_Evaporation"),
                            value=Evaluation_Items["Ground_Evaporation"])
                st.checkbox("Water Evaporation", key="Water_Evaporation", on_change=Evaluation_Items_editor_change,
                            args=("Water_Evaporation", "Water_Evaporation"),
                            value=Evaluation_Items["Water_Evaporation"])
                st.checkbox("Soil Evaporation", key="Soil_Evaporation", on_change=Evaluation_Items_editor_change,
                            args=("Soil_Evaporation", "Soil_Evaporation"), value=Evaluation_Items["Soil_Evaporation"])
                st.checkbox("Total Runoff", key="Total_Runoff", on_change=Evaluation_Items_editor_change,
                            args=("Total_Runoff", "Total_Runoff"), value=Evaluation_Items["Total_Runoff"])
                st.checkbox("Terrestrial Water Storage Change", key="Terrestrial_Water_Storage_Change",
                            on_change=Evaluation_Items_editor_change,
                            args=("Terrestrial_Water_Storage_Change", "Terrestrial_Water_Storage_Change"),
                            value=Evaluation_Items["Terrestrial_Water_Storage_Change"])
                st.checkbox("Snow Water Equivalent", key="Snow_Water_Equivalent",
                            on_change=Evaluation_Items_editor_change,
                            args=("Snow_Water_Equivalent", "Snow_Water_Equivalent"),
                            value=Evaluation_Items["Snow_Water_Equivalent"])
            with col22:
                st.checkbox("Surface Snow Cover In Fraction", key="Surface_Snow_Cover_In_Fraction",
                            on_change=Evaluation_Items_editor_change,
                            args=("Surface_Snow_Cover_In_Fraction", "Surface_Snow_Cover_In_Fraction"),
                            value=Evaluation_Items["Surface_Snow_Cover_In_Fraction"])
                st.checkbox("Snow Depth", key="Snow_Depth", on_change=Evaluation_Items_editor_change,
                            args=("Snow_Depth", "Snow_Depth"), value=Evaluation_Items["Snow_Depth"])
                st.checkbox("Permafrost", key="Permafrost", on_change=Evaluation_Items_editor_change,
                            args=("Permafrost", "Permafrost"), value=Evaluation_Items["Permafrost"])
                st.checkbox("Surface Soil Moisture", key="Surface_Soil_Moisture",
                            on_change=Evaluation_Items_editor_change,
                            args=("Surface_Soil_Moisture", "Surface_Soil_Moisture"),
                            value=Evaluation_Items["Surface_Soil_Moisture"])
                st.checkbox("Root Zone Soil Moisture", key="Root_Zone_Soil_Moisture",
                            on_change=Evaluation_Items_editor_change,
                            args=("Root_Zone_Soil_Moisture", "Root_Zone_Soil_Moisture"),
                            value=Evaluation_Items["Root_Zone_Soil_Moisture"])
                st.checkbox("Water Table Depth", key="Water_Table_Depth", on_change=Evaluation_Items_editor_change,
                            args=("Water_Table_Depth", "Water_Table_Depth"),
                            value=Evaluation_Items["Water_Table_Depth"])
                st.checkbox("Water Storage In Aquifer", key="Water_Storage_In_Aquifer",
                            on_change=Evaluation_Items_editor_change,
                            args=("Water_Storage_In_Aquifer", "Water_Storage_In_Aquifer"),
                            value=Evaluation_Items["Water_Storage_In_Aquifer"])
                st.checkbox("Depth Of Surface Water", key="Depth_Of_Surface_Water",
                            on_change=Evaluation_Items_editor_change,
                            args=("Depth_Of_Surface_Water", "Depth_Of_Surface_Water"),
                            value=Evaluation_Items["Depth_Of_Surface_Water"])
                st.checkbox("Groundwater Recharge Rate", key="Groundwater_Recharge_Rate",
                            on_change=Evaluation_Items_editor_change,
                            args=("Groundwater_Recharge_Rate", "Groundwater_Recharge_Rate"),
                            value=Evaluation_Items["Groundwater_Recharge_Rate"])
        st.subheader(":blue[]", divider=True)
        st.write('##### :blue[Human Activity]')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write('###### :blue[---urban---]')
            st.checkbox("Urban Anthropogenic Heat Flux", key="Urban_Anthropogenic_Heat_Flux",
                        on_change=Evaluation_Items_editor_change,
                        args=("Urban_Anthropogenic_Heat_Flux", "Urban_Anthropogenic_Heat_Flux"),
                        value=Evaluation_Items["Urban_Anthropogenic_Heat_Flux"])
            st.checkbox("Urban Albedo", key="Urban_Albedo", on_change=Evaluation_Items_editor_change,
                        args=("Urban_Albedo", "Urban_Albedo"), value=Evaluation_Items["Urban_Albedo"])
            st.checkbox("Urban Surface Temperature", key="Urban_Surface_Temperature",
                        on_change=Evaluation_Items_editor_change,
                        args=("Urban_Surface_Temperature", "Urban_Surface_Temperature"),
                        value=Evaluation_Items["Urban_Surface_Temperature"])
            st.checkbox("Urban Air Temperature Max", key="Urban_Air_Temperature_Max",
                        on_change=Evaluation_Items_editor_change,
                        args=("Urban_Air_Temperature_Max", "Urban_Air_Temperature_Max"),
                        value=Evaluation_Items["Urban_Air_Temperature_Max"])
            st.checkbox("Urban Air Temperature Min", key="Urban_Air_Temperature_Min",
                        on_change=Evaluation_Items_editor_change,
                        args=("Urban_Air_Temperature_Min", "Urban_Air_Temperature_Min"),
                        value=Evaluation_Items["Urban_Air_Temperature_Min"])
            st.checkbox("Urban Latent Heat Flux", key="Urban_Latent_Heat_Flux",
                        on_change=Evaluation_Items_editor_change,
                        args=("Urban_Latent_Heat_Flux", "Urban_Latent_Heat_Flux"),
                        value=Evaluation_Items["Urban_Latent_Heat_Flux"])
            st.write('###### :blue[---Crop---]')
            st.checkbox("Crop Yield Rice", key="Crop_Yield_Rice", on_change=Evaluation_Items_editor_change,
                        args=("Crop_Yield_Rice", "Crop_Yield_Rice"), value=Evaluation_Items["Crop_Yield_Rice"])
            st.checkbox("Crop Yield Corn", key="Crop_Yield_Corn", on_change=Evaluation_Items_editor_change,
                        args=("Crop_Yield_Corn", "Crop_Yield_Corn"), value=Evaluation_Items["Crop_Yield_Corn"])
            st.checkbox("Crop Yield Wheat", key="Crop_Yield_Wheat", on_change=Evaluation_Items_editor_change,
                        args=("Crop_Yield_Wheat", "Crop_Yield_Wheat"), value=Evaluation_Items["Crop_Yield_Wheat"])
            st.checkbox("Crop Yield Maize", key="Crop_Yield_Maize", on_change=Evaluation_Items_editor_change,
                        args=("Crop_Yield_Maize", "Crop_Yield_Maize"), value=Evaluation_Items["Crop_Yield_Maize"])
        with col2:
            st.checkbox("Crop Yield Soybean", key="Crop_Yield_Soybean", on_change=Evaluation_Items_editor_change,
                        args=("Crop_Yield_Soybean", "Crop_Yield_Soybean"), value=Evaluation_Items["Crop_Yield_Soybean"])
            st.checkbox("Crop Heading DOY Corn", key="Crop_Heading_DOY_Corn", on_change=Evaluation_Items_editor_change,
                        args=("Crop_Heading_DOY_Corn", "Crop_Heading_DOY_Corn"),
                        value=Evaluation_Items["Crop_Heading_DOY_Corn"])
            st.checkbox("Crop Heading DOY Wheat", key="Crop_Heading_DOY_Wheat",
                        on_change=Evaluation_Items_editor_change,
                        args=("Crop_Heading_DOY_Wheat", "Crop_Heading_DOY_Wheat"),
                        value=Evaluation_Items["Crop_Heading_DOY_Wheat"])
            st.checkbox("Crop Maturity DOY Corn", key="Crop_Maturity_DOY_Corn",
                        on_change=Evaluation_Items_editor_change,
                        args=("Crop_Maturity_DOY_Corn", "Crop_Maturity_DOY_Corn"),
                        value=Evaluation_Items["Crop_Maturity_DOY_Corn"])
            st.checkbox("Crop Maturity DOY Wheat", key="Crop_Maturity_DOY_Wheat",
                        on_change=Evaluation_Items_editor_change,
                        args=("Crop_Maturity_DOY_Wheat", "Crop_Maturity_DOY_Wheat"),
                        value=Evaluation_Items["Crop_Maturity_DOY_Wheat"])
            st.checkbox("Crop V3 DOY Corn", key="Crop_V3_DOY_Corn", on_change=Evaluation_Items_editor_change,
                        args=("Crop_V3_DOY_Corn", "Crop_V3_DOY_Corn"), value=Evaluation_Items["Crop_V3_DOY_Corn"])
            st.checkbox("Crop Emergence DOY Wheat", key="Crop_Emergence_DOY_Wheat",
                        on_change=Evaluation_Items_editor_change,
                        args=("Crop_Emergence_DOY_Wheat", "Crop_Emergence_DOY_Wheat"),
                        value=Evaluation_Items["Crop_Emergence_DOY_Wheat"])
            st.checkbox("Total Irrigation Amount", key="Total_Irrigation_Amount",
                        on_change=Evaluation_Items_editor_change,
                        args=("Total_Irrigation_Amount", "Total_Irrigation_Amount"),
                        value=Evaluation_Items["Total_Irrigation_Amount"])
            st.write('###### :blue[---Dam---]')
            st.checkbox("Dam Inflow", key="Dam_Inflow", on_change=Evaluation_Items_editor_change,
                        args=("Dam_Inflow", "Dam_Inflow"),
                        value=Evaluation_Items["Dam_Inflow"])
            st.checkbox("Dam Outflow", key="Dam_Outflow", on_change=Evaluation_Items_editor_change,
                        args=("Dam_Outflow", "Dam_Outflow"), value=Evaluation_Items["Dam_Outflow"])
            st.checkbox("Dam Water Storage", key="Dam_Water_Storage", on_change=Evaluation_Items_editor_change,
                        args=("Dam_Water_Storage", "Dam_Water_Storage"), value=Evaluation_Items["Dam_Water_Storage"])
        with col3:
            st.checkbox("Dam Water Elevation", key="Dam_Water_Elevation", on_change=Evaluation_Items_editor_change,
                        args=("Dam_Water_Elevation", "Dam_Water_Elevation"),
                        value=Evaluation_Items["Dam_Water_Elevation"])
            st.write('###### :blue[---Lake---]')
            st.checkbox("Lake Temperature", key="Lake_Temperature", on_change=Evaluation_Items_editor_change,
                        args=("Lake_Temperature", "Lake_Temperature"), value=Evaluation_Items["Lake_Temperature"])
            st.checkbox("Lake Ice Fraction Cover", key="Lake_Ice_Fraction_Cover",
                        on_change=Evaluation_Items_editor_change,
                        args=("Lake_Ice_Fraction_Cover", "Lake_Ice_Fraction_Cover"),
                        value=Evaluation_Items["Lake_Ice_Fraction_Cover"])
            st.checkbox("Lake Water Level", key="Lake_Water_Level", on_change=Evaluation_Items_editor_change,
                        args=("Lake_Water_Level", "Lake_Water_Level"), value=Evaluation_Items["Lake_Water_Level"])
            st.checkbox("Lake Water Area", key="Lake_Water_Area", on_change=Evaluation_Items_editor_change,
                        args=("Lake_Water_Area", "Lake_Water_Area"), value=Evaluation_Items["Lake_Water_Area"])
            st.checkbox("Lake Water Volume", key="Lake_Water_Volume", on_change=Evaluation_Items_editor_change,
                        args=("Lake_Water_Volume", "Lake_Water_Volume"), value=Evaluation_Items["Lake_Water_Volume"])
            st.write('###### :blue[---River---]')
            st.checkbox("Streamflow", key="Streamflow", on_change=Evaluation_Items_editor_change,
                        args=("Streamflow", "Streamflow"),
                        value=Evaluation_Items["Streamflow"])
            st.checkbox("Inundation Fraction", key="Inundation_Fraction", on_change=Evaluation_Items_editor_change,
                        args=("Inundation_Fraction", "Inundation_Fraction"),
                        value=Evaluation_Items["Inundation_Fraction"])
            st.checkbox("Inundation Area", key="Inundation_Area", on_change=Evaluation_Items_editor_change,
                        args=("Inundation_Area", "Inundation_Area"), value=Evaluation_Items["Inundation_Area"])
            st.checkbox("River Water Level", key="River_Water_Level", on_change=Evaluation_Items_editor_change,
                        args=("River_Water_Level", "River_Water_Level"), value=Evaluation_Items["River_Water_Level"])
        # -------------------------------
        check_list.append(self.__step1_check_evaluation(Evaluation_Items))

        if all(check_list):
            st.session_state.step1_main_check_evaluation = True
        else:
            st.session_state.step1_main_check_evaluation = False
        st.session_state.evaluation_items = Evaluation_Items

    def __step1_check_evaluation(self, Evaluation_Items):
        check_state = 0

        ei_all_false = True
        ei_select = {}
        for key, value in Evaluation_Items.items():
            if isinstance(value, bool):
                if value not in [True, False]:
                    st.warning(f'{key} should be True or False, please check {key}.', icon="⚠")
                    check_state += 1
                if value:
                    ei_all_false = False
                    ei_select[key] = value

        if ei_all_false:
            st.error(f'Please choose at least one Evaluation Item!', icon="⚠")
            check_state += 1
        else:
            formatted_keys = ", \n".join(key.replace('_', ' ') for key in ei_select.keys())
            st.info(f"Make sure your selected Evaluation Item is:      \n:red[{formatted_keys}]", icon="ℹ️")

        if check_state > 0:
            return False
        if check_state == 0:
            return True

    def step1_comparison(self):
        self.__step1_set_Comparison_Items(self.comparisons, self.statistics)
        st.divider()
        step1_disable = False
        if st.session_state.step1_main_check_general & st.session_state.step1_main_check_metrics_scores & (
                st.session_state.step1_main_check_evaluation):
            st.session_state.step1_main_check = True
            st.session_state.step1_main_nml = False
        else:
            step1_disable = True
            st.session_state.step1_main_check = False
            if not st.session_state.step1_main_check_general:
                st.error('There exist error in general page, please check first!', icon="🚨")
            if not st.session_state.step1_main_check_metrics_scores:
                st.error('There exist error in metrics and scores page, please check first!', icon="🚨")
            if not st.session_state.step1_main_check_evaluation:
                st.error('There exist error in evaluation page, please check first!', icon="🚨")
            st.session_state.step1_main_nml = False
        st.session_state['main_data'] = {'general': st.session_state['generals'],
                                         'metrics': st.session_state['metrics'],
                                         'scores': st.session_state['scores'],
                                         'evaluation_items': st.session_state['evaluation_items'],
                                         'comparisons': st.session_state['comparisons'],
                                         'statistics': st.session_state['statistics'],
                                         }

        def define_step1():
            st.session_state.step1_comparison = False

        def define_step2(make_contain, make):
            if st.session_state.main_change:
                with make_contain:
                    if any(v for v in st.session_state.main_change.values()):
                        self._step1_main_nml()
            st.session_state.step2_set = False
            if not st.session_state.step1_main_check:
                st.session_state.step2_set = False
                st.session_state.step1 = False
                make_contain.error('There are some error in Pages, please check!')
            else:
                st.session_state.step2_set = True
                st.session_state.step1 = True
                for key in st.session_state.main_change.keys():
                    st.session_state.main_change[key] = False

        make_contain = st.container()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.button(':back: Previous step', on_click=define_step1, help='Beck to Metrics page')
        with col4:
            st.button('Next step :soon: ', on_click=define_step2, args=(make_contain, 'make_contain'), disabled=step1_disable,
                      help='Go to Simulation page')

    def __step1_set_Comparison_Items(self, comparisons, statistics):
        check_list = []

        if st.session_state.comparisons:
            comparisons = st.session_state.comparisons
        if st.session_state.statistics:
            statistics = st.session_state.statistics

        def comparisons_editor_change(key, editor_key):
            comparisons[key] = st.session_state[key]
            st.session_state.main_change['comparisons'] = True

        st.subheader("Comparisons Items ....", divider=True)
        st.markdown(f"""
        <div style="font-size:22px; font-weight:bold; color:#C48E8E; border-bottom:3px solid #C48E8E; padding: 5px;">
            Basic Items
        </div>""", unsafe_allow_html=True)
        st.write(' ')
        col1, col2 = st.columns(2)
        with col1:
            st.checkbox('Mean values', key="Mean",
                        on_change=comparisons_editor_change,
                        args=("Mean", "Mean"),
                        value=comparisons['Mean'])
            st.checkbox('Min values', key="Min",
                        on_change=comparisons_editor_change,
                        args=("Min", "Min"),
                        value=comparisons['Min'])
            st.checkbox('Sum values', key="Sum",
                        on_change=comparisons_editor_change,
                        args=("Sum", "Sum"),
                        value=comparisons['Sum'])
        with col2:
            st.checkbox('Median values', key="Median",
                        on_change=comparisons_editor_change,
                        args=("Median", "Median"),
                        value=comparisons['Median'])
            st.checkbox('Max values', key="Max",
                        on_change=comparisons_editor_change,
                        args=("Max", "Max"),
                        value=comparisons['Max'])

        st.markdown(f"""
        <div style="font-size:22px; font-weight:bold; color:#C48E8E; border-bottom:3px solid #C48E8E; padding: 5px;">
            Other Items
        </div>""", unsafe_allow_html=True)
        st.write(' ')
        col1, col2 = st.columns(2)
        with col1:
            st.checkbox('HeatMap', key="HeatMap",
                        on_change=comparisons_editor_change,
                        args=("HeatMap", "HeatMap"),
                        value=comparisons['HeatMap'])
            st.checkbox('Taylor Diagram', key="Taylor_Diagram",
                        on_change=comparisons_editor_change,
                        args=("Taylor_Diagram", "Taylor_Diagram"),
                        value=comparisons['Taylor_Diagram'])
            st.checkbox('Target Diagram', key="Target_Diagram",
                        on_change=comparisons_editor_change,
                        args=("Target_Diagram", "Target_Diagram"),
                        value=comparisons['Target_Diagram'])
            st.checkbox('Kernel Density Estimate',
                        key="Kernel_Density_Estimate",
                        on_change=comparisons_editor_change,
                        args=("Kernel_Density_Estimate", "Kernel_Density_Estimate"),
                        value=comparisons['Kernel_Density_Estimate'])
            st.checkbox('Whisker Plot', key="Whisker_Plot",
                        on_change=comparisons_editor_change,
                        args=("Whisker_Plot", "Whisker_Plot"),
                        value=comparisons['Whisker_Plot'])
            st.checkbox('Diff Plot', key="Diff_Plot",
                        on_change=comparisons_editor_change,
                        args=("Diff_Plot", "Diff_Plot"),
                        value=comparisons['Diff_Plot'])
            st.checkbox('Correlation', key="Correlation",
                        on_change=comparisons_editor_change,
                        args=("Correlation", "Correlation"),
                        value=comparisons['Correlation'])
            st.checkbox('Standard Deviation', key="Standard_Deviation",
                        on_change=comparisons_editor_change,
                        args=("Standard_Deviation", "Standard_Deviation"),
                        value=comparisons['Standard_Deviation'])
        with col2:
            st.checkbox('Parallel Coordinates',
                        key="Parallel_Coordinates",
                        on_change=comparisons_editor_change,
                        args=("Parallel_Coordinates", "Parallel_Coordinates"),
                        value=comparisons['Parallel_Coordinates'])
            st.checkbox('Portrait Plot seasonal',
                        key="Portrait_Plot_seasonal",
                        on_change=comparisons_editor_change,
                        args=("Portrait_Plot_seasonal", "Portrait_Plot_seasonal"),
                        value=comparisons['Portrait_Plot_seasonal'])
            st.checkbox('Single Model Performance Index',
                        key="Single_Model_Performance_Index",
                        on_change=comparisons_editor_change,
                        args=("Single_Model_Performance_Index", "Single_Model_Performance_Index"),
                        value=comparisons['Single_Model_Performance_Index'])
            st.checkbox('Relative Score',
                        key="Relative_Score",
                        on_change=comparisons_editor_change,
                        args=("Relative_Score", "Relative_Score"),
                        value=comparisons['Relative_Score'])
            st.checkbox('Ridgeline Plot',
                        key="Ridgeline_Plot",
                        on_change=comparisons_editor_change,
                        args=("Ridgeline_Plot", "Ridgeline_Plot"),
                        value=comparisons['Ridgeline_Plot'])
            st.checkbox('Mann Kendall Trend Test',
                        key="Mann_Kendall_Trend_Test",
                        on_change=comparisons_editor_change,
                        args=("Mann_Kendall_Trend_Test", "Mann_Kendall_Trend_Test"),
                        value=comparisons['Mann_Kendall_Trend_Test'])
            st.checkbox('Functional Response', key="Functional_Response",
                        on_change=comparisons_editor_change,
                        args=("Functional_Response", "Functional_Response"),
                        value=comparisons['Functional_Response'])

            # Statistics_disable = True
            # st.subheader("Statistics Items ....", divider=True)
            # st.checkbox('Mann Kendall Trend Test',
            #             key="Mann_Kendall_Trend_Test",
            #             on_change=statistics_editor_change,
            #             args=("Mann_Kendall_Trend_Test", "Mann_Kendall_Trend_Test"),
            #             disabled=Statistics_disable,
            #             value=statistics['Mann_Kendall_Trend_Test'])
            # st.checkbox('Correlation', key="Correlation",
            #             on_change=statistics_editor_change,
            #             args=("Correlation", "Correlation"),
            #             disabled=Statistics_disable,
            #             value=statistics['Correlation'])
            # st.checkbox('Standard Deviation', key="Standard_Deviation",
            #             on_change=statistics_editor_change,
            #             args=("Standard_Deviation", "Standard_Deviation"),
            #             disabled=Statistics_disable,
            #             value=statistics['Standard_Deviation'])
            # st.checkbox('Z Score', key="Z_Score",
            #             on_change=statistics_editor_change,
            #             args=("Z_Score", "Z_Score"),
            #             disabled=Statistics_disable,
            #             value=statistics['Z_Score'])
            #
            # st.checkbox('Functional Response', key="Functional_Response",
            #             on_change=statistics_editor_change,
            #             args=("Functional_Response", "Functional_Response"),
            #             disabled=Statistics_disable,
            #             value=statistics['Functional_Response'])
            # st.checkbox('Hellinger Distance', key="Hellinger_Distance",
            #             on_change=statistics_editor_change,
            #             args=("Hellinger_Distance", "Hellinger_Distance"),
            #             disabled=Statistics_disable,
            #             value=statistics['Hellinger_Distance'])
            # st.checkbox('Partial Least Squares Regression', key="Partial_Least_Squares_Regression",
            #             on_change=statistics_editor_change,
            #             args=("Partial_Least_Squares_Regression", "Partial_Least_Squares_Regression"),
            #             disabled=Statistics_disable,
            #             value=statistics['Partial_Least_Squares_Regression'])
            # st.checkbox('Three Cornered Hat', key="Three_Cornered_Hat",
            #             on_change=statistics_editor_change,
            #             args=("Three_Cornered_Hat", "Three_Cornered_Hat"),
            #             disabled=Statistics_disable,
            #             value=statistics['Three_Cornered_Hat'])
        st.session_state.step1_main_check_comparison = self.__step1_check_comparisons(comparisons)
        st.session_state.comparisons = comparisons
        st.session_state.statistics = statistics

    def __step1_check_comparisons(self, comparisons):
        # Generals_check = False
        check_state = 0

        ec_select = {}
        score_all_false = False
        for key, value in comparisons.items():
            if isinstance(value, bool):
                if value:
                    ec_select[key] = value
                    score_all_false = False
                    if key == 'HeatMap':
                        if not any(st.session_state.scores.values()):
                            st.warning(f'HeatMap need scores, Please choose at least one Scores!', icon="⚠")
                            score_all_false = True
        if not st.session_state['generals']['comparison'] or not any(comparisons.values()):
            st.warning(f'Please make sure choose to select comparison', icon="⚠")

        if score_all_false:
            check_state += 1
        else:
            formatted_keyc = ", \n".join(key.replace('_', ' ') for key in ec_select.keys())
            st.info(f"Make sure your selected comparisons Item is:      :red[{formatted_keyc}] ", icon="ℹ️")

        if check_state > 0:
            return False
        if check_state == 0:
            return True

    def _step1_main_nml(self):
        if st.session_state.step1_main_check & (not st.session_state.step1_main_nml):
            st.code(f"Make sure your namelist path is: \n{st.session_state.openbench_path}", wrap_lines=True)
            if not os.path.exists(st.session_state.casepath):
                os.makedirs(st.session_state.casepath)
            classification = self.classification
            if st.session_state.step1_initial == 'Setting':
                st.session_state[
                    'main_nml'] = f"{st.session_state.openbench_path}/nml/main-{st.session_state['generals']['basename']}.nml"
            st.session_state.step1_main_nml = self.__step1_make_main_namelist(st.session_state['main_nml'],
                                                                              st.session_state['generals'],
                                                                              st.session_state['metrics'],
                                                                              st.session_state['scores'],
                                                                              st.session_state['evaluation_items'],
                                                                              st.session_state['comparisons'],
                                                                              st.session_state['statistics'],
                                                                              classification)

        if st.session_state.step1_main_nml:
            st.success("😉 Make file successfully!!! \n Please press to Next step")
            for key in st.session_state.main_change.keys():
                st.session_state.main_change[key] = False
        st.session_state['main_data'] = {'general': st.session_state['generals'],
                                         'metrics': st.session_state['metrics'],
                                         'scores': st.session_state['scores'],
                                         'evaluation_items': st.session_state['evaluation_items'],
                                         'comparisons': st.session_state['comparisons'],
                                         'statistics': st.session_state['statistics'],
                                         }

    def __step1_make_main_namelist(self, file_path, Generals, metrics, scores, Evaluation_Items, comparisons,
                                   statistics,
                                   classification):
        """
        Write a namelist from a text file.

        Args:
            file_path (str): Path to the text file.

        """
        with st.spinner('Making namelist... Please wait.'):
            if st.session_state.step1_main_check:
                with open(file_path, 'w') as f:
                    lines = []
                    end_line = "/\n\n\n"

                    lines.append("&general\n")

                    max_key_length = max(len(key) for key in Generals.keys()) + 1
                    for key in list(Generals.keys()):
                        lines.append(f"    {key:<{max_key_length}}= {Generals[f'{key}']}\n")
                    lines.append(end_line)

                    lines.append("&evaluation_items\n")
                    lines.append(
                        "  #========================Evaluation_Items====================\n"
                        "  #*******************Ecosystem and Carbon Cycle****************\n")
                    max_key_length = max(len(key) for key in Evaluation_Items.keys()) + 1
                    for key in list(sorted(classification["Ecosystem and Carbon Cycle"], key=None, reverse=False)):
                        lines.append(f"    {key:<{max_key_length}}= {Evaluation_Items[f'{key}']}\n")

                    lines.append("  #**************************************************************\n\n\n"
                                 "  #*******************      Hydrology Cycle      ****************\n")
                    for key in list(sorted(classification["Hydrology Cycle"], key=None, reverse=False)):
                        lines.append(f"    {key:<{max_key_length}}= {Evaluation_Items[f'{key}']}\n")

                    lines.append("  #**************************************************************\n\n\n"
                                 "  #*******************  Radiation and Energy Cycle  *************\n")
                    for key in list(sorted(classification["Radiation and Energy Cycle"], key=None, reverse=False)):
                        lines.append(f"    {key:<{max_key_length}}= {Evaluation_Items[f'{key}']}\n")

                    lines.append("  #**************************************************************\n\n\n"
                                 "  #*******************         Forcings      **********************\n")
                    for key in list(sorted(classification["Forcings"], key=None, reverse=False)):
                        lines.append(f"    {key:<{max_key_length}}= {Evaluation_Items[f'{key}']}\n")

                    lines.append("  #**************************************************************\n\n\n"
                                 "  #*******************         Human Activity      **********************\n")
                    for key in list(sorted(classification["Human Activity"], key=None, reverse=False)):
                        lines.append(f"    {key:<{max_key_length}}= {Evaluation_Items[f'{key}']}\n")
                    lines.append(end_line)

                    lines.append("&metrics\n")
                    max_key_length = max(len(key) for key in metrics.keys()) + 1
                    for key, value in metrics.items():
                        lines.append(f"    {key:<{max_key_length}}= {value}\n")
                    lines.append(end_line)

                    max_key_length = max(len(key) for key in scores.keys()) + 1
                    lines.append("&scores\n")
                    for key, value in scores.items():
                        lines.append(f"    {key:<{max_key_length}}= {value}\n")
                    lines.append(end_line)

                    max_key_length = max(len(key) for key in comparisons.keys()) + 1
                    lines.append("&comparisons\n")
                    for key, value in comparisons.items():
                        lines.append(f"    {key:<{max_key_length}}= {value}\n")
                    lines.append(end_line)

                    max_key_length = max(len(key) for key in statistics.keys()) + 1
                    lines.append("&statistics\n")
                    for key, value in statistics.items():
                        lines.append(f"    {key:<{max_key_length}}= {value}\n")
                    lines.append(end_line)

                    for line in lines:
                        f.write(line)

                    del max_key_length
                    time.sleep(0.8)

                    return True
            else:
                return False

    def __step1_check_data(self, Generals, metrics, Evaluation_Items):
        # Generals_check = False
        check_state = 0
        for key, value in Generals.items():
            if isinstance(value, str):
                if len(value) <= 1:
                    st.warning(f'{key} should be a string longer than one, please check {key}.', icon="⚠")
                    check_state += 1
            elif isinstance(value, bool):
                if value not in [True, False]:
                    st.warning(f'{key} should be True or False, please check {key}.', icon="⚠")
                    check_state += 1
            elif isinstance(value, int):
                if key in ['num_cores', 'syear', 'eyear']:
                    if not isinstance(value, int):
                        st.warning(f'{key} should be in integer format, please check {key}.', icon="⚠")
                        check_state += 1
            elif isinstance(value, float):
                if key in ['min_year', 'max_lat', 'min_lat', 'max_lon', 'min_lon']:
                    if not isinstance(value, float) or not (value == value):  # 检查是否是 NaN
                        st.warning(f'{key} should be in float format, please check {key}.', icon="⚠")
                        check_state += 1
            elif value is None:
                if key in ['compare_tres', 'compare_tzone']:
                    st.warning(f'{key} It can not be empty, please check {key}.', icon="⚠")
                    check_state += 1
            else:
                st.warning(f'Unsupported data types {type(value)} for {key}, please check {key}.', icon="⚠")
                check_state += 1
        if Generals['eyear'] < Generals['syear']:
            st.warning(f" End year should be larger than Start year, please check.", icon="⚠")
            check_state += 1

        ei_all_false = True
        for key, value in Evaluation_Items.items():
            if isinstance(value, bool):
                if value not in [True, False]:
                    st.warning(f'{key} should be True or False, please check {key}.', icon="⚠")
                    check_state += 1
                if value:
                    ei_all_false = False

        if ei_all_false:
            st.warning(f'Please choose at least one Evaluation Item!', icon="⚠")
            check_state += 1

        metrics_all_false = True
        for key, value in metrics.items():
            if isinstance(value, bool):
                if value not in [True, False]:
                    st.warning(f'{key} should be True or False, please check {key}.', icon="⚠")
                    check_state += 1
                if value:
                    metrics_all_false = False
        if metrics_all_false:
            st.warning(f'Please choose at least one Metrics!', icon="⚠")
            check_state += 1

        if check_state > 0:
            return False
        if check_state == 0:
            return True
        # return check state~




