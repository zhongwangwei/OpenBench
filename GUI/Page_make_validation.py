import glob, shutil
import os
import time
import streamlit as st
from PIL import Image
from io import StringIO
from collections import ChainMap
import xarray as xr
# from streamlit_tags import st_tags
import numpy as np
from Namelist_lib.namelist_read import NamelistReader, GeneralInfoReader, UpdateNamelist, UpdateFigNamelist
from Namelist_lib.namelist_info import initial_setting
import sys
import itertools

from mpl_toolkits.axisartist.angle_helper import select_step


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

    def find_paths_in_dict(self, d):
        paths = []
        for key, value in d.items():
            if isinstance(value, dict):
                paths.extend(self.find_paths_in_dict(value))
            elif isinstance(value, str):
                if 'path' in key.lower() or '/' in value or '\\' in value:
                    if sys.platform.startswith('win'):
                        d[key] = value.replace('/', '\\')
                    elif sys.platform.startswith('linux') | sys.platform.startswith('macos'):
                        d[key] = value.replace('\\', '/')
                        d[key] = value.replace("'\'", "/")
                        d[key] = value.replace("'//'", "/")
                    paths.append((key, value))
            # if sys.platform.startswith('linux'):
        return paths

    def home(self):
        st.subheader("Welcome to the Evaluation page",divider=True)
        st.write(
            "##### :green[Choose whether to upload a file or create a new initial setup, and please press Next step button]")  # 请选择是上传文件，还是新建一个初始设置
        genre = st.radio(
            " What's your choice?", ["***Upload***", "***Setting***"],
            captions=["Upload your case and change a little.", "No need to upload, staring setting."], index=1,
            horizontal=True,label_visibility='collapsed')

        def define_step1():
            st.session_state.step1_general = True

        if genre == '***Upload***':
            st.write('###### Please Upload :red[Main.nml] Filepath:')
            st.session_state['main_nml'] = st.text_input(' :red[Main] Filepath: ',
                                                         value='',
                                                         placeholder="Please input your file path...",label_visibility='collapsed')
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
        st.subheader('General setting Info....',divider=True)

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

        st.write('###### :green[Base case directory]')
        st.text_input('Base case directory: ', value=General["basedir"],
                      key="basedir",
                      on_change=data_editor_change,
                      args=("basedir", "basedir"),
                      placeholder="Set your case directory...",
                          label_visibility='collapsed')
        # ===============================================
        st.divider()
        st.write('###### :green[Num Cores]')
        left, right = st.columns((1, 3))
        with left:
            st.number_input("Num Cores: ", value=General["num_cores"],
                            key="num_cores",
                            on_change=data_editor_change,  # callback function
                            args=("num_cores", "num_cores"),
                            placeholder="how many core will be used in Parallel computing...",
                            format='%d',label_visibility='collapsed')
        with right:
            st.write(':information_source: How many cores will be used in Parallel computing? '
                     'Recommend core number is 5.')

        col1, col2, col3 = st.columns(3)
        col1.write('###### :green[Start year]')
        col2.write('###### :green[Start year]')
        col3.write('###### :green[Minimum year]')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.number_input("Start year: ", format='%04d', value=General["syear"], step=int(1),
                            key="syear",
                            on_change=data_editor_change,  # callback function
                            args=("syear", "syear"),
                            placeholder="Start year...",label_visibility='collapsed')

        with col2:
            st.number_input("End year: ", format='%04d', value=General["eyear"], step=int(1),
                            key="eyear",
                            on_change=data_editor_change,  # callback function
                            args=("eyear", "eyear"),
                            placeholder="End year...",label_visibility='collapsed')

        with col3:
            st.number_input("Minimum year: ", value=General["min_year"], step=1.0,
                            key="min_year",
                            on_change=data_editor_change,  # callback function
                            args=("min_year", "min_year"),
                            placeholder="Minimum year...",label_visibility='collapsed')
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
        col3.checkbox('Running Statistics? ', value=General['statistics'],
                      key="statistics",
                      on_change=data_editor_change,  # callback function
                      args=("statistics", "statistics")
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
        General["statistics_nml"] = f"{st.session_state.openbench_path}/nml/stats.nml"
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
                elif (('./' in path)|('.\\' in path)) & (path[0] == '.'):
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

        st.subheader('Metrics and Scores setting ....',divider=True)
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
        self.__step1_set_Evaluation_Items(self.evaluation_items, self.comparisons, self.statistics)
        st.divider()
        step1_disable=False
        if st.session_state.step1_main_check_general & st.session_state.step1_main_check_metrics_scores & (
                st.session_state.step1_main_check_evaluation):
            st.session_state.step1_main_check = True
            st.session_state.step1_main_nml = False
        else:
            step1_disable=True
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
            st.session_state.step1_evaluation = False

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
            st.button('Next step :soon: ', on_click=define_step2, args=(make_contain, 'make_contain'),disabled=step1_disable,
                      help='Go to Simulation page')

    
    def __step1_set_Evaluation_Items(self, Evaluation_Items, comparisons, statistics):
        check_list = []
        if st.session_state.evaluation_items:
            Evaluation_Items = st.session_state.evaluation_items
        if st.session_state.comparisons:
            comparisons = st.session_state.comparisons
        if st.session_state.statistics:
            statistics = st.session_state.statistics

        def Evaluation_Items_editor_change(key, editor_key):
            Evaluation_Items[key] = st.session_state[key]
            st.session_state.main_change['evaluation'] = True

        def comparisons_editor_change(key, editor_key):
            comparisons[key] = st.session_state[key]
            st.session_state.main_change['comparisons'] = True

        def statistics_editor_change(key, editor_key):
            statistics[key] = st.session_state[key]
            st.session_state.main_change['statistics'] = True

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
                st.checkbox("Terrestrial Water Storage Anomaly", key="Terrestrial_Water_Storage_Anomaly",
                            on_change=Evaluation_Items_editor_change,
                            args=("Terrestrial_Water_Storage_Anomaly", "Terrestrial_Water_Storage_Anomaly"),
                            value=Evaluation_Items["Terrestrial_Water_Storage_Anomaly"])
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
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Comparisons Items ....",divider=True)
            # st.checkbox('IGBP groupby', key="IGBP_groupby",
            #             on_change=comparisons_editor_change,
            #             args=("IGBP_groupby", "IGBP_groupby"),
            #             value=comparisons['IGBP_groupby'])
            # st.checkbox('PFT groupby', key="PFT_groupby",
            #             on_change=comparisons_editor_change,
            #             args=("PFT_groupby", "PFT_groupby"),
            #             value=comparisons['PFT_groupby'])
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

        with col2:
            Statistics_disable = True
            st.subheader("Statistics Items ....",divider=True)
            st.checkbox('Mann Kendall Trend Test',
                        key="Mann_Kendall_Trend_Test",
                        on_change=statistics_editor_change,
                        args=("Mann_Kendall_Trend_Test", "Mann_Kendall_Trend_Test"),
                        disabled=Statistics_disable,
                        value=statistics['Mann_Kendall_Trend_Test'])
            st.checkbox('Correlation', key="Correlation",
                        on_change=statistics_editor_change,
                        args=("Correlation", "Correlation"),
                        disabled=Statistics_disable,
                        value=statistics['Correlation'])
            st.checkbox('Standard Deviation', key="Standard_Deviation",
                        on_change=statistics_editor_change,
                        args=("Standard_Deviation", "Standard_Deviation"),
                        disabled=Statistics_disable,
                        value=statistics['Standard_Deviation'])
            st.checkbox('Functional Response', key="Functional_Response",
                        on_change=statistics_editor_change,
                        args=("Functional_Response", "Functional_Response"),
                        disabled=Statistics_disable,
                        value=statistics['Functional_Response'])
            st.checkbox('Hellinger Distance', key="Hellinger_Distance",
                        on_change=statistics_editor_change,
                        args=("Hellinger_Distance", "Hellinger_Distance"),
                        disabled=Statistics_disable,
                        value=statistics['Hellinger_Distance'])
            st.checkbox('Partial Least Squares Regression', key="Partial_Least_Squares_Regression",
                        on_change=statistics_editor_change,
                        args=("Partial_Least_Squares_Regression", "Partial_Least_Squares_Regression"),
                        disabled=Statistics_disable,
                        value=statistics['Partial_Least_Squares_Regression'])
            st.checkbox('Three Cornered Hat', key="Three_Cornered_Hat",
                        on_change=statistics_editor_change,
                        args=("Three_Cornered_Hat", "Three_Cornered_Hat"),
                        disabled=Statistics_disable,
                        value=statistics['Three_Cornered_Hat'])
        check_list.append(self.__step1_check_comparisons_statistics(comparisons, statistics))
        if all(check_list):
            st.session_state.step1_main_check_evaluation = True
        else:
            st.session_state.step1_main_check_evaluation = False
        st.session_state.evaluation_items = Evaluation_Items
        st.session_state.comparisons = comparisons
        st.session_state.statistics = statistics

    
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

    def __step1_check_comparisons_statistics(self, comparisons, statistics):
        # Generals_check = False
        check_state = 0

        ec_select = {}
        score_all_false = False
        for key, value in comparisons.items():
            if isinstance(value, bool):
                if value:
                    ec_select[key] = value
                    score_all_false = False
                    # score_all_false = True
                    if key in ['HeatMap', 'IGBP_groupby', 'PFT_groupby']:
                        if not any(st.session_state.scores.values()):
                            st.warning(f'HeatMap need scores, Please choose at least one Scores!', icon="⚠")
                            score_all_false = True
        if not st.session_state['generals']['comparison'] and any(comparisons.values()):
            st.warning(f'Please make sure choose to select comparison', icon="⚠")

        es_select = {}
        for key, value in statistics.items():
            if isinstance(value, bool):
                if value:
                    es_select[key] = value
                    score_all_false = False
        if score_all_false:
            check_state += 1
        else:
            formatted_keyc = ", \n".join(key.replace('_', ' ') for key in ec_select.keys())
            formatted_keys = ", \n".join(key.replace('_', ' ') for key in es_select.keys())
            st.info(f"Make sure your selected comparisons Item is:      :red[{formatted_keyc}] "
                    f" \n Make sure your selected Statistics Item is:      :red[{formatted_keys}]", icon="ℹ️")
        # TODO: whether need to add more check to satisfy needs.

        if check_state > 0:
            return False
        if check_state == 0:
            return True
        # return check state~

    
    def _step1_main_nml(self):
        if st.session_state.step1_main_check & (not st.session_state.step1_main_nml):
            st.code(f"Make sure your namelist path is: \n{st.session_state.openbench_path}")
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

                    max_key_length = max(len(key) for key in Generals.keys())
                    for key in list(Generals.keys()):
                        lines.append(f"    {key:<{max_key_length}}={Generals[f'{key}']}\n")
                    lines.append(end_line)

                    lines.append("&evaluation_items\n")
                    lines.append(
                        "  #========================Evaluation_Items====================\n"
                        "  #*******************Ecosystem and Carbon Cycle****************\n")
                    max_key_length = max(len(key) for key in Evaluation_Items.keys())
                    for key in list(sorted(classification["Ecosystem and Carbon Cycle"], key=None, reverse=False)):
                        lines.append(f"    {key:<{max_key_length}}={Evaluation_Items[f'{key}']}\n")

                    lines.append("  #**************************************************************\n\n\n"
                                 "  #*******************      Hydrology Cycle      ****************\n")
                    for key in list(sorted(classification["Hydrology Cycle"], key=None, reverse=False)):
                        lines.append(f"    {key:<{max_key_length}}={Evaluation_Items[f'{key}']}\n")

                    lines.append("  #**************************************************************\n\n\n"
                                 "  #*******************  Radiation and Energy Cycle  *************\n")
                    for key in list(sorted(classification["Radiation and Energy Cycle"], key=None, reverse=False)):
                        lines.append(f"    {key:<{max_key_length}}={Evaluation_Items[f'{key}']}\n")

                    lines.append("  #**************************************************************\n\n\n"
                                 "  #*******************         Forcings      **********************\n")
                    for key in list(sorted(classification["Forcings"], key=None, reverse=False)):
                        lines.append(f"    {key:<{max_key_length}}={Evaluation_Items[f'{key}']}\n")

                    lines.append("  #**************************************************************\n\n\n"
                                 "  #*******************         Human Activity      **********************\n")
                    for key in list(sorted(classification["Human Activity"], key=None, reverse=False)):
                        lines.append(f"    {key:<{max_key_length}}={Evaluation_Items[f'{key}']}\n")
                    lines.append(end_line)

                    lines.append("&metrics\n")
                    max_key_length = max(len(key) for key in metrics.keys())
                    for key, value in metrics.items():
                        lines.append(f"    {key:<{max_key_length}}={value}\n")
                    lines.append(end_line)

                    max_key_length = max(len(key) for key in scores.keys())
                    lines.append("&scores\n")
                    for key, value in scores.items():
                        lines.append(f"    {key:<{max_key_length}}={value}\n")
                    lines.append(end_line)

                    max_key_length = max(len(key) for key in comparisons.keys())
                    lines.append("&comparisons\n")
                    for key, value in comparisons.items():
                        lines.append(f"    {key:<{max_key_length}}={value}\n")
                    lines.append(end_line)

                    max_key_length = max(len(key) for key in statistics.keys())
                    lines.append("&statistics\n")
                    for key, value in statistics.items():
                        lines.append(f"    {key:<{max_key_length}}={value}\n")
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


class make_reference:
    def __init__(self, initial):
        self.author = "Qingchen Xu/xuqingchen23@163.com"
        self.classification = initial.classification()
        self.nl = NamelistReader()
        self.ref_sources = self.nl.read_namelist('./GUI/Namelist_lib/Reference_lib.nml')
        self.initial = initial
        # self.ref_source = initial.ref_source()
        # self.ref_initial = initial.ref_info()

        self.evaluation_items = st.session_state.evaluation_items
        self.selected_items = [k for k, v in self.evaluation_items.items() if v]

        self.tittles = [k.replace('_', ' ') for k, v in self.evaluation_items.items() if v]

        self.ref = initial.ref()

        self.lib_path = os.path.join(st.session_state.openbench_path, 'nml', 'Ref_variables_defination')

        st.session_state.selected_items = self.selected_items
        st.session_state.tittles = self.tittles

    def find_paths_in_dict(self, d):
        paths = []
        for key, value in d.items():
            if isinstance(value, dict):
                paths.extend(self.find_paths_in_dict(value))
            elif isinstance(value, str):
                if 'path' in key.lower() or '/' in value or '\\' in value:
                    if sys.platform.startswith('win'):
                        d[key] = value.replace('/', '\\')
                    elif sys.platform.startswith('linux') | sys.platform.startswith('macos'):
                        d[key] = value.replace('\\', '/')
                        d[key] = value.replace("'\'", "/")
                        d[key] = value.replace("'//'", "/")
                    paths.append((key, value))
            # if sys.platform.startswith('linux'):
        return paths

    
    def step2_set(self):
        # print('Create ref namelist -------------------')
        if 'ref_change' not in st.session_state:
            st.session_state.ref_change = {'general': False}
        st.subheader(f'Select your Reference source', divider=True)
        selected_items = self.selected_items
        Reference_lib = self.ref_sources

        ref_general = self.ref['general']
        if st.session_state.ref_data['general']:
            ref_general = st.session_state.ref_data['general']

        def ref_data_change(key, editor_key):
            ref_general[key] = st.session_state[editor_key]
            st.session_state.ref_change['general'] = True

        for selected_item in selected_items:
            item = f"{selected_item}_ref_source"
            if item not in ref_general:
                ref_general[item] = []
            if isinstance(ref_general[item], str): ref_general[item] = [ref_general[item]]

            label_text = f"<span style='font-size: 20px;'>{selected_item.replace('_', ' ')} reference cases ....</span>"
            st.markdown(f":blue[{label_text}]", unsafe_allow_html=True)
            if len(Reference_lib['general'][selected_item]) == 0:
                st.warning(
                    f"Sorry we didn't offer reference data for {selected_item.replace('_', ' ')}, please upload!")
            # col1, col2 = st.columns((2.5, 1.5))
            st.multiselect("Reference offered",
                           [value for value in Reference_lib['general'][selected_item]],
                           default=[value for value in ref_general[item]],
                           key=f"{item}_multi",
                           on_change=ref_data_change,
                           args=(item, f"{item}_multi"),
                           placeholder="Choose an option",
                           label_visibility="collapsed")

        st.session_state.step2_set_check = self.__step2_setcheck(ref_general)

        sources = list(set([value for key in selected_items for value in ref_general[f"{key}_ref_source"] if value]))
        st.session_state.ref_data['def_nml'] = {}
        for source in sources:
            st.session_state.ref_data['def_nml'][source] = Reference_lib['def_nml'][source]
            if source not in st.session_state.ref_change:
                st.session_state.ref_change[source] = False

        formatted_keys = " \n".join(
            f'{key.replace("_", " ")}: {", ".join(value for value in ref_general[f"{key}_ref_source"] if value)}' for
            key in
            selected_items)
        sourced_key = " \n".join(f"{source}: {Reference_lib['def_nml'][source]}" for source in sources)
        st.code(f'''{formatted_keys}\n\n{sourced_key}''', language="shell", line_numbers=True, wrap_lines=True)

        col1, col, col3 = st.columns(3)

        def define_new_refnml():
            st.session_state.step2_make_newnamelist = True

        col1.button('Add new reference namelist', on_click=define_new_refnml)

        def define_clear_sources():
            st.session_state.step2_mange_sources = True

        col3.button('Manage Reference sources', on_click=define_clear_sources)

        st.session_state.ref_data['general'] = ref_general

        def define_step1():
            st.session_state.step2_set = False

        def define_step2():
            st.session_state.step2_make = True

        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.button(':back: Previous step', on_click=define_step1, help='Go to Evalution page')

        with col4:
            st.button('Next step :soon: ', on_click=define_step2, help='Go to Making page')

    
    def __step2_setcheck(self, general_ref):
        check_state = 0

        ref_all_false = True
        for selected_item in self.selected_items:
            key = f'{selected_item}_ref_source'
            if (general_ref[key] == []) or (general_ref[key] is None) or (len(general_ref[key]) == 0):
                st.warning(f'Please choose at least one source data in {key.replace("_", " ")}!', icon="⚠")
                check_state += 1
            if selected_item not in st.session_state.step2_errorlist:
                st.session_state.step2_errorlist[selected_item] = []

        if check_state > 0:
            st.session_state.step2_errorlist[selected_item].append(1)
            st.session_state.step2_errorlist[selected_item] = list(
                np.unique(st.session_state.step2_errorlist[selected_item]))
            return False
        if check_state == 0:
            if (selected_item in st.session_state.step2_errorlist) & (
                    1 in st.session_state.step2_errorlist[selected_item]):
                st.session_state.step2_errorlist[selected_item] = list(
                    filter(lambda x: x != 1, st.session_state.step2_errorlist[selected_item]))
                st.session_state.step2_errorlist[selected_item] = list(
                    np.unique(st.session_state.step2_errorlist[selected_item]))
            return True

    
    def step2_make_new_refnml(self):
        if 'step2_add_nml' not in st.session_state:
            st.session_state.step2_add_nml = False
        form = st.radio("#### Set reference form 👇", ["Composite", "Single"], key="ref_form",
                        label_visibility='visible',
                        horizontal=True,
                        captions=['If reference has multi-variables', 'If reference has only variable'])
        file = st.radio("Set reference file 👇",
                        ['Composite', 'Crop', 'Dam', 'Ecosystem', 'Energy', 'Forcing', 'Hydrology', 'Lake', 'River',
                         'Urban'],
                        key="ref_savefile", label_visibility='visible',
                        horizontal=True)
        ref_save_path = os.path.join(self.lib_path, file)
        st.divider()

        newlib = {}  # st.session_state['new_lib']

        def variables_change(key, editor_key):
            newlib[key] = st.session_state[editor_key]

        def ref_main_change(key, editor_key):
            if 'general' not in newlib:
                newlib['general'] = {}
            newlib['general'][key] = st.session_state[editor_key]

        def ref_info_change(key, editor_key):
            newlib[key] = st.session_state[f"{key}_{editor_key}"]

        def get_var(col):
            if 'reflib_item' not in st.session_state:
                st.session_state['reflib_item'] = self.initial.evaluation_items()
            Evaluation_Items = st.session_state['reflib_item']
            # c = col.empty()
            col.write('')
            with col.popover("Variables items", use_container_width=True):
                def Evaluation_Items_editor_change(key, editor_key):
                    Evaluation_Items[key] = st.session_state[key]

                st.subheader("Mod Variables ....", divider=True)
                st.write('##### :blue[Ecosystem and Carbon Cycle]')
                # st.subheader("", divider=True, )
                st.checkbox("Gross Primary Productivity", key="Gross_Primary_Productivity",
                            on_change=Evaluation_Items_editor_change,
                            args=("Gross_Primary_Productivity", "Gross_Primary_Productivity"),
                            value=Evaluation_Items["Gross_Primary_Productivity"])
                st.checkbox("Ecosystem Respiration", key="Ecosystem_Respiration",
                            on_change=Evaluation_Items_editor_change,
                            args=("Ecosystem_Respiration", "Ecosystem_Respiration"),
                            value=Evaluation_Items["Ecosystem_Respiration"])
                st.checkbox("Net Ecosystem Exchange", key="Net_Ecosystem_Exchange",
                            on_change=Evaluation_Items_editor_change,
                            args=("Net_Ecosystem_Exchange", "Net_Ecosystem_Exchange"),
                            value=Evaluation_Items["Net_Ecosystem_Exchange"])
                st.checkbox("Leaf Area Index", key="Leaf_Area_Index", on_change=Evaluation_Items_editor_change,
                            args=("Leaf_Area_Index", "Leaf_Area_Index"), value=Evaluation_Items["Leaf_Area_Index"])
                st.checkbox("Biomass", key="Biomass", on_change=Evaluation_Items_editor_change,
                            args=("Biomass", "Biomass"),
                            value=Evaluation_Items["Biomass"])
                st.checkbox("Burned Area", key="Burned_Area", on_change=Evaluation_Items_editor_change,
                            args=("Burned_Area", "Burned_Area"), value=Evaluation_Items["Burned_Area"])
                st.checkbox("Soil Carbon", key="Soil_Carbon", on_change=Evaluation_Items_editor_change,
                            args=("Soil_Carbon", "Soil_Carbon"), value=Evaluation_Items["Soil_Carbon"])
                st.checkbox("Nitrogen Fixation", key="Nitrogen_Fixation", on_change=Evaluation_Items_editor_change,
                            args=("Nitrogen_Fixation", "Nitrogen_Fixation"),
                            value=Evaluation_Items["Nitrogen_Fixation"])
                st.checkbox("Methane", key="Methane", on_change=Evaluation_Items_editor_change,
                            args=("Methane", "Methane"),
                            value=Evaluation_Items["Methane"])
                st.checkbox("Veg Cover In Fraction", key="Veg_Cover_In_Fraction",
                            on_change=Evaluation_Items_editor_change,
                            args=("Veg_Cover_In_Fraction", "Veg_Cover_In_Fraction"),
                            value=Evaluation_Items["Veg_Cover_In_Fraction"])
                st.checkbox("Leaf Greenness", key="Leaf_Greenness", on_change=Evaluation_Items_editor_change,
                            args=("Leaf_Greenness", "Leaf_Greenness"), value=Evaluation_Items["Leaf_Greenness"])

                st.write('##### :blue[Radiation and Energy Cycle]')
                # st.subheader(":blue[]", divider=True)
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

                st.write('##### :blue[Forcings]')
                # st.subheader(":blue[]", divider=True)
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

                st.write('##### :blue[Hydrology Cycle]')
                # st.subheader(":blue[]", divider=True)

                st.checkbox("Evapotranspiration", key="Evapotranspiration", on_change=Evaluation_Items_editor_change,
                            args=("Evapotranspiration", "Evapotranspiration"),
                            value=Evaluation_Items["Evapotranspiration"])
                st.checkbox("Canopy Transpiration", key="Canopy_Transpiration",
                            on_change=Evaluation_Items_editor_change,
                            args=("Canopy_Transpiration", "Canopy_Transpiration"),
                            value=Evaluation_Items["Canopy_Transpiration"])
                st.checkbox("Canopy Interception", key="Canopy_Interception",
                            on_change=Evaluation_Items_editor_change,
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
                st.checkbox("Terrestrial Water Storage Anomaly", key="Terrestrial_Water_Storage_Anomaly",
                            on_change=Evaluation_Items_editor_change,
                            args=("Terrestrial_Water_Storage_Anomaly", "Terrestrial_Water_Storage_Anomaly"),
                            value=Evaluation_Items["Terrestrial_Water_Storage_Anomaly"])
                st.checkbox("Snow Water Equivalent", key="Snow_Water_Equivalent",
                            on_change=Evaluation_Items_editor_change,
                            args=("Snow_Water_Equivalent", "Snow_Water_Equivalent"),
                            value=Evaluation_Items["Snow_Water_Equivalent"])

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

                st.write('##### :blue[Human Activity]')
                # st.subheader(":blue[]", divider=True)

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

                st.checkbox("Crop Yield Soybean", key="Crop_Yield_Soybean", on_change=Evaluation_Items_editor_change,
                            args=("Crop_Yield_Soybean", "Crop_Yield_Soybean"),
                            value=Evaluation_Items["Crop_Yield_Soybean"])
                st.checkbox("Crop Heading DOY Corn", key="Crop_Heading_DOY_Corn",
                            on_change=Evaluation_Items_editor_change,
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
                            args=("Dam_Water_Storage", "Dam_Water_Storage"),
                            value=Evaluation_Items["Dam_Water_Storage"])

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
                            args=("Lake_Water_Volume", "Lake_Water_Volume"),
                            value=Evaluation_Items["Lake_Water_Volume"])
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
                            args=("River_Water_Level", "River_Water_Level"),
                            value=Evaluation_Items["River_Water_Level"])

            return [item for item, value in Evaluation_Items.items() if value]

        if st.session_state['ref_form'] == 'Composite':
            newlib = {}
            info_list = ['sub_dir', 'varname', 'varunit', 'prefix', 'suffix', 'syear', 'eyear']
            col1, col2 = st.columns((1, 2))
            newlib['Ref_libname'] = col1.text_input(f'Reference lib name: ', value='',
                                                    key=f"Ref_libname",
                                                    on_change=variables_change,
                                                    args=(f"Ref_libname", 'Ref_libname'),
                                                    placeholder=f"Set your Reference lib...")
            newlib['variables'] = get_var(col2)

            newlib['general'] = {}
            newlib['general']['root_dir'] = st.text_input(f'Set Data Dictionary: ',
                                                          value='',
                                                          key=f"new_lib_root_dir",
                                                          on_change=ref_main_change,
                                                          args=(f"root_dir", 'new_lib_root_dir'),
                                                          placeholder=f"Set your Reference Dictionary...")
            col1, col2, col3 = st.columns(3)
            newlib['general']['timezone'] = col1.number_input(f"Set Time zone: ",
                                                              value=0.0,
                                                              key=f"new_lib_timezone",
                                                              on_change=ref_main_change,
                                                              args=(f"timezone", 'new_lib_timezone'),
                                                              min_value=-12.0,
                                                              max_value=12.0)
            newlib['general']['data_type'] = col2.selectbox(f'Set Data type: ',
                                                            options=('stn', 'Grid'),
                                                            index=1,
                                                            key=f"new_lib_data_type",
                                                            on_change=ref_main_change,
                                                            args=(f"data_type", 'new_lib_data_type'),
                                                            placeholder=f"Set your Reference Data type...")
            newlib['general']['data_groupby'] = col3.selectbox(f'Set Data groupby: ',
                                                               options=('hour', 'day', 'month', 'year', 'single'),
                                                               index=4,
                                                               key=f"new_lib_data_groupby",
                                                               on_change=ref_main_change,
                                                               args=(f"data_groupby", 'new_lib_data_groupby'),
                                                               placeholder=f"Set your Reference Data groupby...")
            newlib['general']['tim_res'] = col1.selectbox(f'Set Time Resolution: ',
                                                          options=('hour', 'day', 'month', 'year'),
                                                          index=0,
                                                          key=f"new_lib_tim_res",
                                                          on_change=ref_main_change,
                                                          args=(f"tim_res", 'new_lib_tim_res'),
                                                          placeholder=f"Set your Reference Time Resolution ...")
            if newlib['general']['data_type'] == 'Grid':
                newlib['general']['grid_res'] = col2.number_input(f"Set Geo Resolution: ",
                                                                  value=0.5,
                                                                  min_value=0.0,
                                                                  key=f"new_lib_grid_res",
                                                                  on_change=ref_main_change,
                                                                  args=(f"grid_res", 'new_lib_grid_res'),
                                                                  placeholder="Set your Reference Geo Resolution...")
                year = st.toggle(f'Reference {newlib["Ref_libname"]} share the same start and end year?',
                                 value=True)
                if year:
                    info_list = ['sub_dir', 'varname', 'varunit', 'prefix', 'suffix']
                    col1, col2, col3 = st.columns(3)
                    newlib['general']['syear'] = col1.number_input(f"Set Start year: ",
                                                                   value=2000,
                                                                   format='%04d', step=int(1),
                                                                   key=f"new_lib_syear",
                                                                   on_change=ref_main_change,
                                                                   args=(f"syear", 'new_lib_syear'),
                                                                   placeholder="Set your Reference Start year...")
                    newlib['general']['eyear'] = col2.number_input(f"Set End year: ",
                                                                   value=2001,
                                                                   format='%04d', step=int(1),
                                                                   key=f"new_lib_eyear",
                                                                   on_change=ref_main_change,
                                                                   args=(f"eyear", 'new_lib_eyear'),
                                                                   placeholder="Set your Reference End year...")
            else:
                info_list = ['varname', 'varunit']
                newlib['general'][f"fulllist"] = st.text_input(f'Set Station Fulllist File: ',
                                                               value='',
                                                               key=f"new_lib_fulllist",
                                                               on_change=ref_main_change,
                                                               args=(f"fulllist", 'new_lib_fulllist'),
                                                               placeholder=f"Set your Reference Fulllist file...")
                newlib['general']['grid_res'] = ''
                newlib['general']['syear'] = ''
                newlib['general']['eyear'] = ''

            newlib['info_list'] = st.multiselect("Add info", info_list, default=['varname', 'varunit'],
                                                 key=f"variables_info_list",
                                                 on_change=variables_change,
                                                 args=('info_list', f"variables_info_list"),
                                                 placeholder="Choose an option",
                                                 label_visibility="visible")

            info_lists = {'sub_dir': {'title': 'Set Data Sub-Dictionary', 'value': ''},
                          'varname': {'title': 'Set varname', 'value': ''},
                          'varunit': {'title': 'Set varunit', 'value': ''},
                          'prefix': {'title': 'Set prefix', 'value': ''},
                          'suffix': {'title': 'Set suffix', 'value': ''},
                          'syear': {'title': 'Set syear', 'value': 2000},
                          'eyear': {'title': 'Set eyear', 'value': 2001}}
            if newlib['variables']:
                for var in self.evaluation_items:
                    if var in newlib['variables']:
                        newlib[var] = {}
                    if var in newlib and var not in newlib['variables']:
                        del newlib[var]

                import itertools
                for variable in newlib['variables']:
                    with st.container(height=None, border=True):
                        st.write(f"##### :blue[{variable.replace('_', ' ')}]")
                        cols = itertools.cycle(st.columns(3))
                        for info in st.session_state['variables_info_list']:
                            col = next(cols)
                            newlib[variable][info] = col.text_input(info_lists[info]['title'],
                                                                    value=info_lists[info]['value'],
                                                                    key=f"{variable}_{info}",
                                                                    on_change=ref_info_change,
                                                                    args=(variable, info),
                                                                    placeholder=f"Set your Reference Fulllist file...")

        else:
            newlib = {}
            info_list = ['varname', 'varunit', 'prefix', 'suffix']
            col1, col2 = st.columns((1, 2))
            newlib['Ref_libname'] = col1.text_input(f'Reference lib name: ', value='',
                                                    key=f"Ref_libname",
                                                    on_change=variables_change,
                                                    args=(f"Ref_libname", 'Ref_libname'),
                                                    placeholder=f"Set your Reference lib...")
            newlib['variables'] = col2.selectbox("Variable selected",
                                                 [e.replace('_', ' ') for e in self.evaluation_items], index=None,
                                                 key=f"variable_select",
                                                 on_change=variables_change,
                                                 args=('variables', f"variable_select"),
                                                 placeholder="Choose an option",
                                                 label_visibility="visible")

            if st.session_state['variable_select']:
                newlib['general'] = {}
                newlib['general']['root_dir'] = st.text_input(f'Set Data Dictionary: ',
                                                              value='',
                                                              key=f"new_lib_root_dir",
                                                              on_change=ref_main_change,
                                                              args=(f"root_dir", 'new_lib_root_dir'),
                                                              placeholder=f"Set your Reference Dictionary...")
                col1, col2, col3 = st.columns(3)
                newlib['general']['timezone'] = col1.number_input(f"Set Time zone: ",
                                                                  value=0.0,
                                                                  key=f"new_lib_timezone",
                                                                  on_change=ref_main_change,
                                                                  args=(f"timezone", 'new_lib_timezone'),
                                                                  min_value=-12.0,
                                                                  max_value=12.0)
                newlib['general']['data_type'] = col2.selectbox(f'Set Data type: ',
                                                                options=('stn', 'Grid'),
                                                                index=1,
                                                                key=f"new_lib_data_type",
                                                                on_change=ref_main_change,
                                                                args=(f"data_type", 'new_lib_data_type'),
                                                                placeholder=f"Set your Reference Data type...")
                newlib['general']['data_groupby'] = col3.selectbox(f'Set Data groupby: ',
                                                                   options=('hour', 'day', 'month', 'year', 'single'),
                                                                   index=4,
                                                                   key=f"new_lib_data_groupby",
                                                                   on_change=ref_main_change,
                                                                   args=(f"data_groupby", 'new_lib_data_groupby'),
                                                                   placeholder=f"Set your Reference Data groupby...")
                newlib['general']['tim_res'] = col1.selectbox(f'Set Time Resolution: ',
                                                              options=('hour', 'day', 'month', 'year'),
                                                              index=0,
                                                              key=f"new_lib_tim_res",
                                                              on_change=ref_main_change,
                                                              args=(f"tim_res", 'new_lib_tim_res'),
                                                              placeholder=f"Set your Reference Time Resolution ...")
                if newlib['general']['data_type'] == 'Grid':
                    newlib['general']['grid_res'] = col2.number_input(f"Set Geo Resolution: ",
                                                                      value=0.5,
                                                                      min_value=0.0,
                                                                      key=f"new_lib_grid_res",
                                                                      on_change=ref_main_change,
                                                                      args=(f"grid_res", 'new_lib_grid_res'),
                                                                      placeholder="Set your Reference Geo Resolution...")
                    newlib['general']['syear'] = col1.number_input(f"Set Start year: ",
                                                                   value=2000,
                                                                   format='%04d', step=int(1),
                                                                   key=f"new_lib_syear",
                                                                   on_change=ref_main_change,
                                                                   args=(f"syear", 'new_lib_syear'),
                                                                   placeholder="Set your Reference Start year...")
                    newlib['general']['eyear'] = col2.number_input(f"Set End year: ",
                                                                   value=2001,
                                                                   format='%04d', step=int(1),
                                                                   key=f"new_lib_eyear",
                                                                   on_change=ref_main_change,
                                                                   args=(f"eyear", 'new_lib_eyear'),
                                                                   placeholder="Set your Reference End year...")
                else:
                    info_list = ['varname', 'varunit']
                    newlib['general'][f"fulllist"] = st.text_input(f'Set Station Fulllist File: ',
                                                                   value='',
                                                                   key=f"new_lib_fulllist",
                                                                   on_change=ref_main_change,
                                                                   args=(f"fulllist", 'new_lib_fulllist'),
                                                                   placeholder=f"Set your Reference Fulllist file...")
                    newlib['general']['grid_res'] = ''
                    newlib['general']['syear'] = ''
                    newlib['general']['eyear'] = ''

                newlib['info_list'] = st.multiselect("Add info", info_list, default=info_list,
                                                     key=f"variables_info_list",
                                                     on_change=variables_change,
                                                     args=('info_list', f"variables_info_list"),
                                                     placeholder="Choose an option",
                                                     label_visibility="visible")

                info_lists = {'sub_dir': {'title': 'Set Data Sub-Dictionary', 'value': ''},
                              'varname': {'title': 'Set varname', 'value': ''},
                              'varunit': {'title': 'Set varunit', 'value': ''},
                              'prefix': {'title': 'Set prefix', 'value': ''},
                              'suffix': {'title': 'Set suffix', 'value': ''},
                              'syear': {'title': 'Set syear', 'value': 2000},
                              'eyear': {'title': 'Set eyear', 'value': 2001}}

                for var in self.evaluation_items:
                    if var.replace('_', ' ') in newlib['variables']:
                        newlib[var] = {}
                    if var in newlib and var.replace('_', ' ') not in newlib['variables']:
                        del newlib[var]

                import itertools
                variable = newlib['variables'].replace(' ', '_')
                with st.container(height=None, border=True):
                    st.write(f"##### :blue[{variable.replace('_', ' ')}]")
                    cols = itertools.cycle(st.columns(3))
                    for info in st.session_state['variables_info_list']:
                        col = next(cols)
                        newlib[variable][info] = col.text_input(info_lists[info]['title'],
                                                                value=info_lists[info]['value'],
                                                                key=f"{variable}_{info}",
                                                                on_change=ref_info_change,
                                                                args=(variable, info),
                                                                placeholder=f"Set your Reference Fulllist file...")

        disable = False
        if not newlib['Ref_libname']:
            st.error('Please input your Reference lib name First!')
            disable = True
        elif not newlib['general']['root_dir']:
            st.error('Please input your Reference Dictionary First!')
            disable = True
        elif newlib['general']['data_type'] == 'stn' and not newlib['general'][f"fulllist"]:
            st.error('Please input your Reference fulllist First!')
            disable = True

        def define_add():
            st.session_state.step2_add_nml = True

        col1, col2, col3 = st.columns(3)
        if col1.button('Make namelist', on_click=define_add, help='Press this button to add new reference namelist',
                       disabled=disable):
            self.__step2_make_ref_lib_namelist(ref_save_path, newlib, form)
            st.success("😉 Make file successfully!!! \n Please press to Next step")

        st.divider()

        def define_back(make_contain, warn):
            if not disable:
                if not st.session_state.step2_add_nml:
                    with make_contain:
                        self.__step2_make_ref_lib_namelist(ref_save_path, newlib, form)
                    make_contain.success("😉 Make file successfully!!! \n Please press to Next step")
                del_vars = ['reflib_item']
                for del_var in del_vars:
                    del st.session_state[del_var]
                st.session_state.step2_add_nml = False

            else:
                make_contain.warning('Making file failed or No file is processing!')
                time.sleep(0.8)
            st.session_state.step2_make_newnamelist = False

        make_contain = st.container()
        st.button('Finish and back to select page', on_click=define_back, args=(make_contain, 'make_contain'),
                  help='Finish add and go back to select page')

    
    def __step2_make_ref_lib_namelist(self, file_path, newlib, fname):
        """
        Write a namelist from a text file.

        Args:
            file_path (str): Path to the text file.

        """
        with st.spinner('Making namelist... Please wait.'):
            with open(file_path + f'/{newlib["Ref_libname"]}.nml', 'w') as f:
                lines = []
                end_line = "/\n\n\n"

                lines.append("&general\n")
                max_key_length = max(len(key) for key in newlib['general'].keys())
                for key in list(newlib['general'].keys()):
                    lines.append(f"    {key:<{max_key_length}} = {newlib['general'][f'{key}']}\n")
                lines.append(end_line)

                max_key_length = max(len(key) for key in newlib['info_list'])
                if isinstance(newlib['variables'], str): newlib['variables'] = [newlib['variables']]
                for variable in newlib['variables']:
                    variable = variable.replace(' ', '_')
                    lines.append(f"&{variable}\n")
                    for info in newlib['info_list']:
                        lines.append(f"    {info:<{max_key_length}} = {newlib[variable][info]}\n")
                    lines.append(end_line)
                for line in lines:
                    f.write(line)
                time.sleep(2)

            if fname == 'Composite':
                for variable in newlib['variables']:
                    variable = variable.replace(' ', '_')
                    if isinstance(self.ref_sources['general'][variable], str): self.ref_sources['general'][variable] = [
                        self.ref_sources['general'][variable]]
                    if newlib["Ref_libname"] not in self.ref_sources['general'][variable]:
                        self.ref_sources['general'][variable].append(newlib["Ref_libname"])
                self.ref_sources['def_nml'][newlib["Ref_libname"]] = file_path + f'/{newlib["Ref_libname"]}.nml'

            else:
                variable = variable.replace(' ', '_')
                if isinstance(self.ref_sources['general'][variable], str): self.ref_sources['general'][variable] = [
                    self.ref_sources['general'][variable]]
                if newlib["Ref_libname"] not in self.ref_sources['general'][variable]:
                    self.ref_sources['general'][variable].append(newlib["Ref_libname"])
                self.ref_sources['def_nml'][newlib["Ref_libname"]] = file_path + f'/{newlib["Ref_libname"]}.nml'

            with open('./GUI/Namelist_lib/Reference_lib.nml', 'w') as f1:
                lines = []
                end_line = "/\n\n\n"
                lines.append("&general\n")
                max_key_length = max(len(key) for key in self.ref_sources['general'].keys())
                for key in list(self.ref_sources['general'].keys()):
                    value = self.ref_sources['general'][f'{key}']
                    if isinstance(value, str): value = [value]
                    lines.append(f"    {key:<{max_key_length}} = {', '.join(value)}\n")
                lines.append(end_line)

                lines.append("&def_nml\n")
                max_key_length = max(len(key) for key in self.ref_sources['def_nml'].keys())
                for key in list(self.ref_sources['def_nml'].keys()):
                    lines.append(f"    {key:<{max_key_length}} = {self.ref_sources['def_nml'][f'{key}']}\n")
                lines.append(end_line)
                for line in lines:
                    f1.write(line)
            time.sleep(0.5)
            # return True

    def step2_mange_sources(self):
        if 'step2_remove' not in st.session_state:
            st.session_state['step2_remove'] = False
        Ref_lib = self.nl.read_namelist('./GUI/Namelist_lib/Reference_lib.nml')

        # st.write(Ref_lib['general'])

        def find_source(item, source):
            found_source = False
            for key, values in Ref_lib['general'].items():
                if isinstance(values, str): values = [values]
                if key != item and source in values:
                    return False
            if not found_source:
                return True

        def get_cases(Ref_lib):
            case_item = {}
            for item in self.selected_items:
                case_item[item] = {}
                for source in Ref_lib['general'][item]:
                    case_item[item][source] = False

            st.subheader("Reference sources ....", divider=True)
            for item in case_item.keys():
                st.write(f"##### :blue[{item}]")
                cols = itertools.cycle(st.columns(3))
                for source in case_item[item].keys():
                    col = next(cols)
                    case_item[item][source] = col.checkbox(source, key=f'{item}_{source}',
                                                           value=case_item[item][source])
                    if case_item[item][source]:
                        single_source = find_source(item, source)
                        if not single_source:
                            st.warning(f"{item} {source} exist in other variables make sure you want to delete!")
            if len([f"{item}, {key}" for item in case_item.keys() for key, value in case_item[item].items() if value]) > 0:
                sources = {}
                for item in case_item.keys():
                    if any(case_item[item].values()):
                        sources[item] = {}
                    for key, value in case_item[item].items():
                        if value:
                            sources[item][key] = True
                return True, sources
            else:
                return False, {}

        cases, sources = get_cases(Ref_lib)

        def remove(sources):
            with st.spinner('Making namelist... Please wait.'):
                for item in sources:
                    for source in sources[item]:
                        remove_nml = find_source(item, source)
                        if remove_nml and source in Ref_lib['def_nml']:
                            if os.path.exists(Ref_lib['def_nml'][source]):
                                data = Ref_lib['def_nml'].pop(source)
                                try:
                                    os.remove(data)
                                    st.code(f"Remove file: {data}")
                                except:
                                    st.warning(f"{source} already removed or file doesn't exist!")
                            else:
                                st.warning(f"{source} already removed or file doesn't exist!")

                with open('./GUI/Namelist_lib/Reference_lib.nml', 'w') as f1:
                    lines = []
                    end_line = "/\n\n\n"
                    lines.append("&general\n")

                    max_key_length = max(len(key) for key in Ref_lib['general'].keys())
                    for key in list(Ref_lib['general'].keys()):
                        values = Ref_lib['general'][f'{key}']
                        if isinstance(values, str): values = [values]
                        values = [value for value in values if key not in sources or value not in sources[key]]
                        lines.append(f"    {key:<{max_key_length}} = {', '.join(values)}\n")
                    lines.append(end_line)

                    lines.append("&def_nml\n")
                    max_key_length = max(len(key) for key in Ref_lib['def_nml'].keys())
                    for key in list(Ref_lib['def_nml'].keys()):
                        lines.append(f"    {key:<{max_key_length}} = {Ref_lib['def_nml'][f'{key}']}\n")
                    lines.append(end_line)
                    for line in lines:
                        f1.write(line)
                time.sleep(0.8)

        def define_remove():
            st.session_state.step2_remove = True

        disable = False
        if not cases:
            disable = True

        remove_contain = st.container()

        if st.button('Remove cases', on_click=define_remove, help='Press to remove cases', disabled=disable):
            with remove_contain:
                remove(sources)

        st.divider()

        def define_back(remove_contain, warn):
            if not disable:
                if not st.session_state.step2_remove:
                    with remove_contain:
                        remove(sources)
                st.session_state.step2_remove = False
            st.session_state.step2_mange_sources = False

        st.button('Finish and back to select page', on_click=define_back, args=(remove_contain, 'remove_contain'),
                  help='Finish add and go back to select page')

    
    def step2_make(self):
        if 'step2_make_check' not in st.session_state:
            st.session_state.step2_make_check = False

        selected_items = self.selected_items
        tittles = self.tittles
        ref_general = st.session_state.ref_data['general']
        def_nml = st.session_state.ref_data['def_nml']

        if ref_general and def_nml:
            st.session_state.step2_check = []
            for i, (source, path), tab in zip(range(len(def_nml)), def_nml.items(), st.tabs(def_nml.keys())):
                try:
                    if source not in st.session_state.ref_data:
                        st.session_state.ref_data[source] = self.nl.read_namelist(path)
                    tab.subheader(f':blue[{source} Reference checking ....]', divider=True)
                    with tab:  # .expander(f"###### Cases: {source}", expanded=True)
                        self.__step2_make_ref_info(i, source, st.session_state.ref_data[source], path, ref_general)  # self.ref
                except Exception as e:
                    st.error(f'Error: {e}')
            if all(st.session_state.step2_check):
                st.session_state.step2_make_check = True
            else:
                st.session_state.step2_make_check = False
        else:
            st.error('Please select your Reference first!')
            st.session_state.step2_make_check = False

        def define_step1():
            st.session_state.step2_make = False

        def define_step2():
            st.session_state.step2_tomake_nml = True

        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.button(':back: Previous step', on_click=define_step1, help='Go to Reference set page',
                      use_container_width=True)
        with col4:
            st.button('Next step :soon: ', on_click=define_step2, help='Go to Reference nml page',
                      use_container_width=True)

    
    def __step2_make_ref_info(self, i, source, source_lib, file, ref_general):
        # st.write(source)

        def ref_editor_change(key, editor_key, source):
            source_lib[key][editor_key] = st.session_state[f"{source}_{key}_{editor_key}"]
            st.session_state.ref_change[source] = True

        import itertools
        with st.container(height=None, border=True):
            key = 'general'
            source_lib[key][f"root_dir"] = st.text_input(f'{i}. Set Data Dictionary: ',
                                                         value=source_lib[key][f"root_dir"],
                                                         key=f"{source}_{key}_root_dir",
                                                         on_change=ref_editor_change,
                                                         args=(key, "root_dir", source),
                                                         placeholder=f"Set your Reference Dictionary...")
            if source_lib[key]['data_type'] == 'stn':
                source_lib[key][f"fulllist"] = st.text_input(f'{i}. Set Fulllist File: ',
                                                             value=source_lib[key][f"fulllist"],
                                                             key=f"{source}_{key}_fulllist",
                                                             on_change=ref_editor_change,
                                                             args=(key, "fulllist", source),
                                                             placeholder=f"Set your Reference Fulllist file...")

            cols = itertools.cycle(st.columns(2))
            for key, values in source_lib.items():
                if key != 'general' and key in self.selected_items:
                    if source in ref_general[f'{key}_ref_source']:
                        col = next(cols)
                        if 'sub_dir' in source_lib[key].keys():
                            source_lib[key][f"sub_dir"] = col.text_input(
                                f'{i}. Set {key.replace("_", " ")} Sub-Data Dictionary: ',
                                value=source_lib[key][f"sub_dir"],
                                key=f"{source}_{key}_sub_dir",
                                on_change=ref_editor_change,
                                args=(key, "sub_dir", source),
                                placeholder=f"Set your Reference Dictionary...")

            st.session_state.step2_check.append(self.__step2_makecheck(source_lib, source))

    
    def __step2_makecheck(self, source_lib, source):
        error_state = 0
        warning_state = 0
        general_list = ["root_dir", "timezone", "data_type", "data_groupby", "tim_res"]
        grid_list = ["grid_res", "syear", "eyear"]
        stn_list = ["fulllist"]
        info_list = ['sub_dir', 'varname', 'varunit', 'prefix', 'suffix', 'syear', 'eyear']
        for key in general_list:
            if isinstance(source_lib['general'][key], str):
                if len(source_lib['general'][key]) < 1:
                    st.error(f'general: {key} should be a string longer than one, please check {key}.',
                             icon="⚠")
                    error_state += 1
        if source_lib['general']["data_type"] == 'grid':
            if "grid_res" in source_lib['general'].keys():
                if isinstance(source_lib['general']["grid_res"], float) | isinstance(source_lib['general']["grid_res"],
                                                                                     int):
                    if source_lib['general']["grid_res"] <= 0:
                        st.error(
                            f"general: Geo Resolution should be larger than zero when data_type is 'geo', please check.",
                            icon="⚠")
                        error_state += 1
            if "syear" in source_lib['general'].keys() and "eyear" in source_lib['general'].keys():
                if isinstance(source_lib['general']["syear"], int) and isinstance(source_lib['general']["eyear"], int):
                    if source_lib['general']["syear"] > source_lib['general']["eyear"]:
                        st.error(f'general: End year should be larger than Start year, please check.',
                                 icon="⚠")
                        error_state += 1
        else:
            if not source_lib['general']["fulllist"]:
                st.error(f"general : Fulllist should not be empty when data_type is 'stn'.",
                         icon="⚠")
                error_state += 1

        for var in source_lib.keys():
            if var != 'general':
                for key in source_lib[var].keys():
                    if key in ['sub_dir', 'varname', 'varunit']:  #
                        if var in self.selected_items and source in st.session_state.ref_data['general'][
                            f'{var}_ref_source']:
                            if len(source_lib[var][key]) < 1:
                                st.error(f'{var}: {key} should be a string longer than one, please check.',
                                         icon="⚠")
                                error_state += 1
                        if len(source_lib[var][key]) < 1:
                            warning_state += 1
                    elif key in ['prefix', 'suffix']:
                        if len(source_lib[var]['prefix']) < 1 and len(source_lib[var]['suffix']) < 1:
                            warning_state += 1
                    elif key in ['syear', 'eyear']:
                        if not isinstance(source_lib[var][key], int):
                            warning_state += 1
                        if source_lib[var]["syear"] > source_lib[var]["eyear"]:
                            warning_state += 1

        if warning_state > 0:
            st.warning(f"Some mistake in source,please check your file!",
                       icon="⚠")

        if source not in st.session_state.step2_errorlist:
            st.session_state.step2_errorlist[source] = []
        if error_state > 0:
            st.session_state.step2_errorlist[source].append(2)
            st.session_state.step2_errorlist[source] = list(np.unique(st.session_state.step2_errorlist[source]))
            return False
        if error_state == 0:
            if (source in st.session_state.step2_errorlist) & (2 in st.session_state.step2_errorlist[source]):
                st.session_state.step2_errorlist[source] = list(
                    filter(lambda x: x != 2, st.session_state.step2_errorlist[source]))
                st.session_state.step2_errorlist[source] = list(np.unique(st.session_state.step2_errorlist[source]))
            return True

    
    def step2_ref_nml(self):
        step2_disable=False
        if st.session_state.step2_set_check & st.session_state.step2_make_check:
            for source, path in st.session_state.ref_data['def_nml'].items():
                source_lib = st.session_state.ref_data[source]
                st.subheader(source, divider=True)
                path_info = f'Root Dictionary: {source_lib["general"][f"root_dir"]}'
                key = 'general'
                if source_lib[key]['data_type'] == 'stn':
                    path_info = path_info + f'\nFulllist File: {source_lib["general"][f"fulllist"]}'

                for key, values in source_lib.items():
                    if key != 'general' and key in self.selected_items:
                        if source in st.session_state.ref_data['general'][f'{key}_ref_source']:
                            if 'sub_dir' in source_lib[key].keys():
                                path_info = path_info + f'\n{key.replace("_", " ")} Sub-Data Dictionary: {source_lib[key][f"sub_dir"]}'
                st.code(f'''{path_info}''', language='shell', line_numbers=True, wrap_lines=True)
            st.session_state.step2_ref_check = True
            st.session_state.step2_ref_nml = False
        else:
            step2_disable=True
            if not st.session_state.step2_set_check:
                formatted_keys = ", ".join(
                    key.replace('_', ' ') for key, value in st.session_state.step2_errorlist.items() if 1 in value)
                st.error(
                    f'There exist error in set page, please check {formatted_keys} first! Set your reference data.',
                    icon="🚨")
            if not st.session_state.step2_make_check:
                formatted_keys = ", ".join(
                    key.replace('_', ' ') for key, value in st.session_state.step2_errorlist.items() if 2 in value)
                st.error(f'There exist error in Making page, please check {formatted_keys} first!', icon="🚨")
            st.session_state.step2_ref_nml = False
            st.session_state.step2_ref_check = False

        def write_nml(nml_dict, output_file):
            """
            将字典数据重新写回 .nml 文件。
            """
            with open(output_file, 'w') as f:
                # 确保 'general' 部分总是第一个
                if 'general' in nml_dict:
                    f.write(f'&general\n')
                    max_key_length = max(len(key) for key in nml_dict['general'].keys())
                    for key, value in nml_dict['general'].items():
                        f.write(f'  {key:<{max_key_length}} = {value}\n')
                    f.write('/\n\n')

                # 写入其他部分
                for section, variables in nml_dict.items():
                    max_key_length = max(len(key) for key in variables.keys())
                    if section == 'general':
                        continue  # 'general' 已经处理过了
                    f.write(f'&{section}\n')
                    for key, value in variables.items():
                        f.write(f'  {key:<{max_key_length}} = {value}\n')
                    f.write('/\n\n')
            del f

        def make():
            for key, value in st.session_state.ref_change.items():
                if key == 'general' and value:
                    if st.session_state.step2_ref_check & (not st.session_state.step2_ref_nml):
                        st.session_state.step2_ref_nml = self.__step2_make_ref_namelist(
                            st.session_state.generals['reference_nml'], self.selected_items, st.session_state.ref_data)
                    if st.session_state.step2_ref_nml:
                        st.success("😉 Make file successfully!!!")
                        st.session_state.ref_change[key] = False
                elif key != 'general' and value:
                    file = st.session_state.ref_data['def_nml'][key]
                    source_lib = st.session_state.ref_data[key]
                    write_nml(source_lib, file)
                    st.session_state.ref_change[key] = False

        def define_step1():
            st.session_state.step2_tomake_nml = False

        def define_step2(make_contain, smake):
            if not st.session_state.step2_ref_check:
                st.session_state.step3_set = False
                st.session_state.step2 = False
            else:
                with make_contain:
                    if any(v for v in st.session_state.ref_change.values()):
                        make()
                st.session_state.step3_set = True
                st.session_state.step2 = True

        make_contain = st.container()

        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.button(':back: Previous step', on_click=define_step1, help='Go to Reference Making page')
        with col4:
            st.button('Next step :soon: ', on_click=define_step2, args=(make_contain, 'make'),disabled=step2_disable,
                      help='Go to Simulation page')


    def __step2_make_ref_namelist(self, file_path, selected_items, ref_data):
        general = ref_data['general']
        def_nml = ref_data['def_nml']

        with st.spinner('Making namelist... Please wait.'):
            if st.session_state.step2_ref_check:
                st.write("Making namelist...")
                with open(file_path, 'w') as f:
                    lines = []
                    end_line = "/\n\n\n"

                    lines.append("&general\n")
                    max_key_length = max(len(f"{key}") for key in general.keys())
                    for item in selected_items:
                        key = f"{item}_ref_source"
                        lines.append(f"    {key:<{max_key_length}} = {','.join(general[f'{item}_ref_source'])}\n")
                    lines.append(end_line)

                    lines.append("&def_nml\n")
                    max_key_length = max(len(key) for key in def_nml.keys())
                    for key, value in def_nml.items():
                        lines.append(f"    {key:<{max_key_length}} = {value}\n")
                    lines.append(end_line)
                    for line in lines:
                        f.write(line)
                    time.sleep(2)

                    return True

    # 
    # def __step2_make_ref_namelist(self, file_path, selected_items, ref_data):
    #     general = st.session_state.ref_data['general']
    #     norm_list = ['_timezone', '_data_type', '_data_groupby', '_dir', '_varname', '_varunit', '_fulllist',
    #                  '_tim_res',
    #                  '_geo_res',
    #                  '_suffix', '_prefix', '_syear', '_eyear']
    #     streamflow_list = ['_timezone', '_data_type', '_data_groupby', '_dir', '_varname', '_varunit', '_fulllist',
    #                        '_tim_res',
    #                        '_geo_res',
    #                        '_suffix', '_prefix', '_syear', '_eyear', '_max_uparea', '_min_uparea']
    #
    #     with st.spinner('Making namelist... Please wait.'):
    #         if st.session_state.step2_ref_check:
    #             st.write("Making namelist...")
    #             with open(file_path, 'w') as f:
    #                 lines = []
    #                 end_line = "/\n\n\n"
    #
    #                 lines.append("&general\n")
    #                 for item in selected_items:
    #                     lines.append(f"    {item}_ref_source = {','.join(ref_data['general'][f'{item}_ref_source'])}\n")
    #                 lines.append(end_line)
    #                 # for key in list(ref_data['general'].keys()):
    #                 #     if key[:-11] in selected_items:
    #                 #         lines.append(f"    {key} = {','.join(ref_data['general'][key])}\n")
    #                 # lines.append(end_line)
    #
    #                 for item in selected_items:
    #                     if item == 'Streamflow':
    #                         lines.append(f"&{item}\n")
    #                         for casename in ref_data['general'][f'{item}_ref_source']:
    #                             lines.append(f"\n#source: {casename}\n")
    #                             for key in streamflow_list:
    #                                 if (key in ['_geo_res', '_suffix', '_prefix', '_syear', '_eyear']) & (
    #                                         st.session_state.ref_data[item][f"{casename}_data_type"] == 'stn'):
    #                                     lines.append(f"    {casename}{key} =  \n")
    #                                 elif (key in ['_fulllist']) & (
    #                                         st.session_state.ref_data[item][f"{casename}_data_type"] == 'geo'):
    #                                     lines.append(f"    {casename}{key} =  \n")
    #                                 else:
    #                                     lines.append(f"    {casename}{key} =  {ref_data[item][f'{casename}{key}']}\n")
    #                         lines.append(end_line)
    #                     else:
    #                         lines.append(f"&{item}\n")
    #                         for casename in ref_data['general'][f'{item}_ref_source']:
    #                             lines.append(f"\n#source: {casename}\n")
    #                             for key in norm_list:
    #                                 if (key in ['_geo_res', '_suffix', '_prefix', '_syear', '_eyear']) & (
    #                                         st.session_state.ref_data[item][f"{casename}_data_type"] == 'stn'):
    #                                     lines.append(f"    {casename}{key} =  \n")
    #                                 elif (key in ['_fulllist']) & (
    #                                         st.session_state.ref_data[item][f"{casename}_data_type"] == 'geo'):
    #                                     lines.append(f"    {casename}{key} =  \n")
    #                                 else:
    #                                     lines.append(f"    {casename}{key} =  {ref_data[item][f'{casename}{key}']}\n")
    #                         lines.append(end_line)
    #
    #                 for line in lines:
    #                     f.write(line)
    #                 time.sleep(2)
    #
    #                 return True


class make_simulation():
    def __init__(self, initial):
        self.author = "Qingchen Xu/xuqingchen23@163.com"
        self.classification = initial.classification()
        # self.sim_initial = initial.sim_info()
        # self.sim = st.session_state.sim_data
        self.sim = initial.sim()
        self.evaluation_items = st.session_state.evaluation_items
        self.selected_items = [k for k, v in self.evaluation_items.items() if v]
        self.tittles = [k.replace('_', ' ') for k, v in self.evaluation_items.items() if v]
        self.initial = initial
        self.nl = NamelistReader()
        self.sim_sources = self.nl.read_namelist('./GUI/Namelist_lib/Simulation_lib.nml')
        self.Mod_variables_defination = os.path.join(st.session_state.openbench_path, 'nml', 'Mod_variables_defination')
        self.lib_path = os.path.join(st.session_state.openbench_path, 'nml', 'user')

    def find_paths_in_dict(self, d):
        paths = []
        for key, value in d.items():
            if isinstance(value, dict):
                paths.extend(self.find_paths_in_dict(value))
            elif isinstance(value, str):
                if 'path' in key.lower() or '/' in value or '\\' in value:
                    if sys.platform.startswith('win'):
                        d[key] = value.replace('/', '\\')
                    elif sys.platform.startswith('linux') | sys.platform.startswith('macos'):
                        d[key] = value.replace('\\', '/')
                        d[key] = value.replace("'\'", "/")
                        d[key] = value.replace("'//'", "/")
                    paths.append((key, value))
            # if sys.platform.startswith('linux'):
        return paths

    
    def step3_set(self):
        # print('Create ref namelist -------------------')
        if 'sim_change' not in st.session_state:
            st.session_state.sim_change = {'general': False}

        st.subheader(f'Select your simulation cases', divider=True)
        selected_items = self.selected_items
        Simulation_lib = self.nl.read_namelist('./GUI/Namelist_lib/Simulation_lib.nml')

        if 'add_mode' not in st.session_state:
            st.session_state['add_mod'] = False

        sim_general = self.sim['general']
        if st.session_state.sim_data['general']:
            sim_general = st.session_state.sim_data['general']

        def sim_data_change(key, editor_key):
            sim_general[key] = st.session_state[editor_key]
            st.session_state.sim_change['general'] = True

        for selected_item in selected_items:
            item = f"{selected_item}_sim_source"
            if item not in sim_general:
                sim_general[item] = []
            if isinstance(sim_general[item], str): sim_general[item] = [sim_general[item]]

            label_text = f"<span style='font-size: 20px;'>{selected_item.replace('_', ' ')} simulation cases ....</span>"
            st.markdown(f":blue[{label_text}]", unsafe_allow_html=True)
            if len(Simulation_lib['general']['Case_lib']) == 0:
                st.warning(
                    f"Sorry we didn't offer simulation data, please upload!")

            st.multiselect("simulation offered",
                           [value for value in Simulation_lib['general']['Case_lib']],
                           default=[value for value in sim_general[item]],
                           key=f"{item}_multi",
                           on_change=sim_data_change,
                           args=(item, f"{item}_multi"),
                           placeholder="Choose an option",
                           label_visibility="collapsed")

        st.session_state.step3_set_check = self.__step3_setcheck(sim_general)

        sources = list(set([value for key in selected_items for value in sim_general[f"{key}_sim_source"] if value]))
        st.session_state.sim_data['def_nml'] = {}
        for source in sources:
            st.session_state.sim_data['def_nml'][source] = Simulation_lib['def_nml'][source]
            st.session_state.sim_change[source] = False

        formatted_keys = " \n".join(
            f'{key.replace("_", " ")}: {", ".join(value for value in sim_general[f"{key}_sim_source"] if value)}' for
            key in
            selected_items)
        sourced_key = " \n".join(f"{source}: {Simulation_lib['def_nml'][source]}" for source in sources)
        st.code(f'''{formatted_keys}\n\n{sourced_key}''', language="shell", line_numbers=True, wrap_lines=True)

        #

        col1, col, col3 = st.columns(3)

        def define_new_simnml():
            st.session_state.step3_make_newnamelist = True

        col1.button('Add new simulation namelist', on_click=define_new_simnml)

        def define_clear_cases():
            st.session_state.step3_mange_cases = True

        col3.button('Manage simulation cases', on_click=define_clear_cases)

        st.session_state.sim_data['general'] = sim_general

        st.divider()

        def define_step1():
            st.session_state.step3_set = False

        def define_step2():
            st.session_state.step3_make = True

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.button(':back: Previous step', on_click=define_step1, help='Go to Refernece page')
        with col4:
            st.button('Next step :soon: ', on_click=define_step2, help='Go to making page')

    
    def __step3_setcheck(self, general_sim):
        check_state = 0

        ref_all_false = True
        for selected_item in self.selected_items:
            key = f"{selected_item}_sim_source"
            if general_sim[key] is None or len(general_sim[key]) == 0:
                st.warning(f'Please set at least one case in {key.replace("_", " ")}!', icon="⚠")
                check_state += 1
            if selected_item not in st.session_state.step3_errorlist:
                st.session_state.step3_errorlist[selected_item] = []

        if check_state > 0:
            st.session_state.step3_errorlist[selected_item].append(1)
            st.session_state.step3_errorlist[selected_item] = list(
                np.unique(st.session_state.step3_errorlist[selected_item]))
            return False
        if check_state == 0:
            if (selected_item in st.session_state.step3_errorlist) & (
                    1 in st.session_state.step3_errorlist[selected_item]):
                st.session_state.step3_errorlist[selected_item] = list(
                    filter(lambda x: x != 1, st.session_state.step3_errorlist[selected_item]))
                st.session_state.step3_errorlist[selected_item] = list(
                    np.unique(st.session_state.step3_errorlist[selected_item]))
            return True

    
    def step3_make_new_simnml(self):
        if 'step3_add_nml' not in st.session_state:
            st.session_state.step3_add_nml = True
        # st.write("#### :red[前两天学习的时候，有同学提出了疑问，如果在simulation里面，但是不同变量在不同文件里面的情况，这个需要修改一下对应的代码，类似于reference"
        #          "的那种]")
        path = os.path.join(st.session_state.openbench_path, 'nml', 'user')
        folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

        file = st.radio("Set Simulation file 👇",
                        ["user"] + [f"user/{folder}" for folder in folders] + ['New folder'],
                        key="sim_savefile", label_visibility='visible',
                        horizontal=True)
        if file == 'New folder':
            col1, col2, col3 = st.columns(3)
            name = col1.text_input(f'Simulation file name: ', value='',
                                   key=f"Sim_filename",
                                   placeholder=f"Set your Simulation file...")
            file = f"user/{name}"
        sim_save_path = os.path.join(st.session_state.openbench_path, 'nml', file)

        st.divider()

        def variables_change(key, editor_key):
            newlib[key] = st.session_state[editor_key]

        def sim_main_change(key, editor_key):
            newlib['general'][key] = st.session_state[editor_key]

        newlib = {}
        col1, col2, col3, col4 = st.columns((1, 1.5, 0.8, 0.8))
        newlib['Sim_casename'] = col1.text_input(f'Simulation case name: ', value='',
                                                 key=f"Sim_casename",
                                                 on_change=variables_change,
                                                 args=(f"Sim_casename", 'Sim_casename'),
                                                 placeholder=f"Set your Simulation case...")
        newlib['Mod'] = col2.selectbox("Mod variables selected", sorted(self.sim_sources['def_Mod'].keys()), index=None,
                                       key=f"Mod_select",
                                       on_change=variables_change,
                                       args=('Mod', f"Mod_select"),
                                       placeholder="Choose an option",
                                       label_visibility="visible")
        col3.write(':point_down: press to add')
        col4.write('new model')

        def define_new_mod():
            st.session_state.add_mod = True

        def define_finish():
            st.session_state.add_mod = False
            del_vars = ['mode_item']
            for del_var in del_vars:
                del st.session_state[del_var]

        col3.button('Add new Mod', on_click=define_new_mod)
        col4.button('Finish add', on_click=define_finish)

        def get_var(col, name):
            if name not in st.session_state:
                st.session_state[name] = self.initial.evaluation_items()
            Evaluation_Items = st.session_state[name]
            col.write('')
            with col.popover("Variables items", use_container_width=True):
                def Evaluation_Items_editor_change(key, editor_key):
                    Evaluation_Items[key] = st.session_state[key]

                st.subheader("Mod Variables ....", divider=True)
                st.write('##### :blue[Ecosystem and Carbon Cycle]')
                # st.subheader("", divider=True, )
                st.checkbox("Gross Primary Productivity", key="Gross_Primary_Productivity",
                            on_change=Evaluation_Items_editor_change,
                            args=("Gross_Primary_Productivity", "Gross_Primary_Productivity"),
                            value=Evaluation_Items["Gross_Primary_Productivity"])
                st.checkbox("Ecosystem Respiration", key="Ecosystem_Respiration",
                            on_change=Evaluation_Items_editor_change,
                            args=("Ecosystem_Respiration", "Ecosystem_Respiration"),
                            value=Evaluation_Items["Ecosystem_Respiration"])
                st.checkbox("Net Ecosystem Exchange", key="Net_Ecosystem_Exchange",
                            on_change=Evaluation_Items_editor_change,
                            args=("Net_Ecosystem_Exchange", "Net_Ecosystem_Exchange"),
                            value=Evaluation_Items["Net_Ecosystem_Exchange"])
                st.checkbox("Leaf Area Index", key="Leaf_Area_Index", on_change=Evaluation_Items_editor_change,
                            args=("Leaf_Area_Index", "Leaf_Area_Index"), value=Evaluation_Items["Leaf_Area_Index"])
                st.checkbox("Biomass", key="Biomass", on_change=Evaluation_Items_editor_change,
                            args=("Biomass", "Biomass"),
                            value=Evaluation_Items["Biomass"])
                st.checkbox("Burned Area", key="Burned_Area", on_change=Evaluation_Items_editor_change,
                            args=("Burned_Area", "Burned_Area"), value=Evaluation_Items["Burned_Area"])
                st.checkbox("Soil Carbon", key="Soil_Carbon", on_change=Evaluation_Items_editor_change,
                            args=("Soil_Carbon", "Soil_Carbon"), value=Evaluation_Items["Soil_Carbon"])
                st.checkbox("Nitrogen Fixation", key="Nitrogen_Fixation", on_change=Evaluation_Items_editor_change,
                            args=("Nitrogen_Fixation", "Nitrogen_Fixation"),
                            value=Evaluation_Items["Nitrogen_Fixation"])
                st.checkbox("Methane", key="Methane", on_change=Evaluation_Items_editor_change,
                            args=("Methane", "Methane"),
                            value=Evaluation_Items["Methane"])
                st.checkbox("Veg Cover In Fraction", key="Veg_Cover_In_Fraction",
                            on_change=Evaluation_Items_editor_change,
                            args=("Veg_Cover_In_Fraction", "Veg_Cover_In_Fraction"),
                            value=Evaluation_Items["Veg_Cover_In_Fraction"])
                st.checkbox("Leaf Greenness", key="Leaf_Greenness", on_change=Evaluation_Items_editor_change,
                            args=("Leaf_Greenness", "Leaf_Greenness"), value=Evaluation_Items["Leaf_Greenness"])

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

                st.write('##### :blue[Forcings]')
                # st.subheader(":blue[]", divider=True)
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

                st.write('##### :blue[Hydrology Cycle]')
                # st.subheader(":blue[]", divider=True)

                st.checkbox("Evapotranspiration", key="Evapotranspiration", on_change=Evaluation_Items_editor_change,
                            args=("Evapotranspiration", "Evapotranspiration"),
                            value=Evaluation_Items["Evapotranspiration"])
                st.checkbox("Canopy Transpiration", key="Canopy_Transpiration",
                            on_change=Evaluation_Items_editor_change,
                            args=("Canopy_Transpiration", "Canopy_Transpiration"),
                            value=Evaluation_Items["Canopy_Transpiration"])
                st.checkbox("Canopy Interception", key="Canopy_Interception",
                            on_change=Evaluation_Items_editor_change,
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
                st.checkbox("Terrestrial Water Storage Anomaly", key="Terrestrial_Water_Storage_Anomaly",
                            on_change=Evaluation_Items_editor_change,
                            args=("Terrestrial_Water_Storage_Anomaly", "Terrestrial_Water_Storage_Anomaly"),
                            value=Evaluation_Items["Terrestrial_Water_Storage_Anomaly"])
                st.checkbox("Snow Water Equivalent", key="Snow_Water_Equivalent",
                            on_change=Evaluation_Items_editor_change,
                            args=("Snow_Water_Equivalent", "Snow_Water_Equivalent"),
                            value=Evaluation_Items["Snow_Water_Equivalent"])

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

                st.write('##### :blue[Human Activity]')
                # st.subheader(":blue[]", divider=True)

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

                st.checkbox("Crop Yield Soybean", key="Crop_Yield_Soybean", on_change=Evaluation_Items_editor_change,
                            args=("Crop_Yield_Soybean", "Crop_Yield_Soybean"),
                            value=Evaluation_Items["Crop_Yield_Soybean"])
                st.checkbox("Crop Heading DOY Corn", key="Crop_Heading_DOY_Corn",
                            on_change=Evaluation_Items_editor_change,
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
                            args=("Dam_Water_Storage", "Dam_Water_Storage"),
                            value=Evaluation_Items["Dam_Water_Storage"])

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
                            args=("Lake_Water_Volume", "Lake_Water_Volume"),
                            value=Evaluation_Items["Lake_Water_Volume"])
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
                            args=("River_Water_Level", "River_Water_Level"),
                            value=Evaluation_Items["River_Water_Level"])

            return [item for item, value in Evaluation_Items.items() if value]

        if st.session_state.add_mod:
            mod_lib = {}
            mod_lib['general'] = {}
            with st.container(height=None, border=False):
                st.divider()
                st.write(':blue[Add Mod variables defination]', )

                col1, col2, col3 = st.columns((3, 4, 2))
                mod_lib['general']['model'] = col1.text_input(f'Model name: ', value='',
                                                              placeholder=f"Set your Model name...")
                variables = get_var(col2, 'mode_item')
                for var in self.evaluation_items:
                    if var in variables:
                        mod_lib[var] = {}
                    if var in mod_lib and var not in variables:
                        del mod_lib[var]

                info_lists = {
                    'varname': {'title': 'Set varname', 'value': ''},
                    'varunit': {'title': 'Set varunit', 'value': ''}
                }
                with st.expander(':blue[Add Mod variables defination]', expanded=True, icon=None):
                    import itertools
                    info_list = ['varname', 'varunit']
                    tcols = itertools.cycle(st.columns(2))
                    for variable in variables:
                        tcol = next(tcols)
                        tcol.write(f"##### :blue[{variable.replace('_', ' ')}]")
                        cols = itertools.cycle(tcol.columns(2))
                        for info in info_list:
                            col = next(cols)
                            mod_lib[variable][info] = col.text_input(info_lists[info]['title'],
                                                                     value=info_lists[info]['value'],
                                                                     key=f"{variable}_{info}_model",
                                                                     placeholder=f"Set your Model Var...")
                        tcol.divider()
                col3.write('''''')
                if col3.button('Make namelist', help='Yes, this is the one.'):
                    make_Mod = self.__step3_make_Mod_namelist(self.Mod_variables_defination, mod_lib, variables)
                    if make_Mod:
                        st.success("😉 Make file successfully!!! \n Please press to Next step")

        def get_info(col, ilist):
            if isinstance(ilist, str): ilist = [ilist]
            case_item = {}
            for item in ilist:
                case_item[item] = False

            col.write('')
            with col.popover("Variables Infos", use_container_width=True):
                st.subheader(f"Showing Infos", divider=True)
                for item in ilist:
                    case_item[item] = st.checkbox(item, key=f"{item}__sim_Infos",
                                                  value=case_item[item])
                return [item for item, value in case_item.items() if value]

        newlib['general'] = {}
        if newlib['Mod']:
            info_list = ['sub_dir', 'varname', 'varunit', 'prefix', 'suffix']
            st.divider()
            newlib['general']['model_namelist'] = self.sim_sources['def_Mod'][newlib['Mod']]
            col1, col2, col3 = st.columns(3)
            newlib['general']['timezone'] = col1.number_input(f"Set Time zone: ",
                                                              value=0.0,
                                                              key=f"new_simlib_timezone",
                                                              on_change=sim_main_change,
                                                              args=(f"timezone", 'new_simlib_timezone'),
                                                              min_value=-12.0,
                                                              max_value=12.0)
            newlib['general']['data_type'] = col2.selectbox(f'Set Data type: ',
                                                            options=('stn', 'Grid'),
                                                            index=1,
                                                            key=f"new_simlib_data_type",
                                                            on_change=sim_main_change,
                                                            args=(f"data_type", 'new_simlib_data_type'),
                                                            placeholder=f"Set your Simulation Data type...")
            newlib['general']['data_groupby'] = col3.selectbox(f'Set Data groupby: ',
                                                               options=('hour', 'day', 'month', 'year', 'single'),
                                                               index=4,
                                                               key=f"new_simlib_data_groupby",
                                                               on_change=sim_main_change,
                                                               args=(f"data_groupby", 'new_simlib_data_groupby'),
                                                               placeholder=f"Set your Simulation Data groupby...")
            newlib['general']['tim_res'] = col1.selectbox(f'Set Time Resolution: ',
                                                          options=('hour', 'day', 'month', 'year'),
                                                          index=0,
                                                          key=f"new_simlib_tim_res",
                                                          on_change=sim_main_change,
                                                          args=(f"tim_res", 'new_simlib_tim_res'),
                                                          placeholder=f"Set your Simulation Time Resolution ...")
            if newlib['general']['data_type'] == 'Grid':
                newlib['general']['grid_res'] = col2.number_input(f"Set Geo Resolution: ",
                                                                  value=0.5,
                                                                  min_value=0.0,
                                                                  key=f"new_simlib_grid_res",
                                                                  on_change=sim_main_change,
                                                                  args=(f"grid_res", 'new_simlib_grid_res'),
                                                                  placeholder="Set your Simulation Geo Resolution...")

                newlib['general']['syear'] = col1.number_input(f"Set Start year: ",
                                                               value=2000,
                                                               format='%04d', step=int(1),
                                                               key=f"new_simlib_syear",
                                                               on_change=sim_main_change,
                                                               args=(f"syear", 'new_simlib_syear'),
                                                               placeholder="Set your Simulation Start year...")
                newlib['general']['eyear'] = col2.number_input(f"Set End year: ",
                                                               value=2001,
                                                               format='%04d', step=int(1),
                                                               key=f"new_simlib_eyear",
                                                               on_change=sim_main_change,
                                                               args=(f"eyear", 'new_simlib_eyear'),
                                                               placeholder="Set your Simulation End year...")
                ffix = st.toggle(f'Simulation {newlib["Sim_casename"]} share the same prefix and suffix?', value=True)
                col1, col2, col3 = st.columns(3)
                if ffix:
                    info_list = ['sub_dir', 'varname', 'varunit']
                    newlib['general']['prefix'] = col1.text_input(f'Set File prefix: ',
                                                                  value='',
                                                                  key=f"new_simlib_prefix",
                                                                  on_change=sim_main_change,
                                                                  args=(f"prefix", 'new_simlib_prefix'),
                                                                  placeholder=f"Set your Simulation prefix...")
                    newlib['general']['suffix'] = col2.text_input(f'Set File suffix: ',
                                                                  value='',
                                                                  key=f"new_simlib_suffix",
                                                                  on_change=sim_main_change,
                                                                  args=(f"suffix", 'new_simlib_suffix'),
                                                                  placeholder=f"Set your Simulation suffix...")
                newlib['general']['dir'] = st.text_input(f'Set Data Dictionary: ',
                                                         value='',
                                                         key=f"new_simlib_root_dir",
                                                         on_change=sim_main_change,
                                                         args=(f"root_dir", 'new_simlib_root_dir'),
                                                         placeholder=f"Set your Simulation Dictionary...")
                newlib['general'][f"fulllist"] = ''
            else:
                info_list = ['sub_dir', 'varname', 'varunit']
                newlib['general']['dir'] = st.text_input(f'Set Data Dictionary: ',
                                                         value='',
                                                         key=f"new_simlib_root_dir",
                                                         on_change=sim_main_change,
                                                         args=(f"root_dir", 'new_simlib_root_dir'),
                                                         placeholder=f"Set your Simulation Dictionary...")

                newlib['general'][f"fulllist"] = st.text_input(f'Set Station Fulllist File: ',
                                                               value='',
                                                               key=f"new_simlib_fulllist",
                                                               on_change=sim_main_change,
                                                               args=(f"fulllist", 'new_simlib_fulllist'),
                                                               placeholder=f"Set your Simulation Fulllist file...")
                newlib['general']['grid_res'] = ''
                newlib['general']['syear'] = ''
                newlib['general']['eyear'] = ''
                newlib['general']['suffix'] = ''
                newlib['general']['prefix'] = ''
            st.divider()
            st.write(f':point_down: If some item is differ to {newlib["Mod"]} variables, press to change')
            col1, col2 = st.columns((2, 1.2))

            newlib['variables'] = get_var(col1, 'case_items')
            newlib['info_list'] = get_info(col2, info_list)
            if newlib['variables'] and newlib['info_list']:
                info_lists = {'sub_dir': {'title': 'Set Data Sub-Dictionary', 'value': ''},
                              'varname': {'title': 'Set varname', 'value': ''},
                              'varunit': {'title': 'Set varunit', 'value': ''},
                              'prefix': {'title': 'Set prefix', 'value': ''},
                              'suffix': {'title': 'Set suffix', 'value': ''},
                              'syear': {'title': 'Set syear', 'value': 2000},
                              'eyear': {'title': 'Set eyear', 'value': 2001}}

                for var in self.evaluation_items:
                    if var in newlib['variables']:
                        newlib[var] = {}
                    if var in newlib and var not in newlib['variables']:
                        del newlib[var]
                import itertools
                for variable in newlib['variables']:
                    with st.container(height=None, border=True):
                        st.write(f"##### :blue[{variable.replace('_', ' ')}]")
                        cols = itertools.cycle(st.columns(3))
                        for info in newlib['info_list']:
                            col = next(cols)
                            newlib[variable][info] = col.text_input(info_lists[info]['title'],
                                                                    value=info_lists[info]['value'],
                                                                    key=f"{variable}_{info}_sim",
                                                                    placeholder=f"Set your Simulation Var...")

        disable = False
        check = self.__step3_check_newcase_namelist(newlib)
        if not check:
            disable = True

        def define_add():
            st.session_state.step3_add_nml = True

        col1, col2, col3 = st.columns(3)
        if col1.button('Make namelist', help='Yes, this is the one.', disabled=disable, on_click=define_add):
            self.__step3_make_case_namelist(sim_save_path, newlib)
            st.success("😉 Make file successfully!!! \n Please press to Next step")

        st.divider()

        def define_back(make_contain, warn):
            if not disable:
                if not st.session_state.step3_add_nml:
                    with make_contain:
                        self.__step3_make_case_namelist(sim_save_path, newlib)
                    make_contain.success("😉 Make file successfully!!! \n Please press to Next step")
                st.session_state.step3_add_nml = False
            else:
                make_contain.warning('Making file failed or No file is processing!')
                time.sleep(0.8)
            st.session_state.step3_make_newnamelist = False
            del_vars = ['case_items']
            for del_var in del_vars:
                del st.session_state[del_var]

        make_contain = st.container()
        st.button('Finish and back to select page', on_click=define_back, args=(make_contain, 'make_contain'),
                  help='Finish add and go back to select page')

    def __step3_make_Mod_namelist(self, path, mod_lib, variables):
        """
        Write a namelist from a text file.

        Args:
            file_path (str): Path to the text file.

        """
        if mod_lib["general"]["model"] == '':
            st.error('Please define your model first!')
            return False
        elif len(variables) == 0:
            st.error('Please choose variables first!')
            return False
        else:
            with st.spinner('Making namelist... Please wait.'):
                with open(path + f'/{mod_lib["general"]["model"]}.nml', 'w') as f:
                    lines = []
                    end_line = "/\n\n\n"

                    lines.append("&general\n")
                    lines.append(f"    model = {mod_lib['general']['model']}\n")
                    lines.append(end_line)

                    for variable in variables:
                        lines.append(f"&{variable}\n")
                        for info in ['varname', 'varunit']:
                            lines.append(f"    {info} = {mod_lib[variable][info]}\n")
                        lines.append(end_line)
                    for line in lines:
                        f.write(line)
                    time.sleep(2)

                self.sim_sources['def_Mod'][mod_lib["general"]["model"]] = path + f'/{mod_lib["general"]["model"]}.nml'
                with open('./GUI/Namelist_lib/Simulation_lib.nml', 'w') as f1:
                    lines = []
                    end_line = "/\n\n\n"
                    lines.append("&general\n")
                    max_key_length = max(len(key) for key in self.sim_sources['general'].keys())
                    for key in list(self.sim_sources['general'].keys()):
                        value = self.sim_sources['general'][f'{key}']
                        if isinstance(value, str): value = [value]
                        lines.append(f"    {key:<{max_key_length}} = {', '.join(value)}\n")
                    lines.append(end_line)

                    lines.append("&def_nml\n")
                    max_key_length = max(len(key) for key in self.sim_sources['def_nml'].keys())
                    for key in list(self.sim_sources['def_nml'].keys()):
                        lines.append(f"    {key:<{max_key_length}} = {self.sim_sources['def_nml'][f'{key}']}\n")
                    lines.append(end_line)

                    lines.append("&def_Mod\n")
                    max_key_length = max(len(key) for key in self.sim_sources['def_Mod'].keys())
                    for key in list(self.sim_sources['def_Mod'].keys()):
                        lines.append(f"    {key:<{max_key_length}} = {self.sim_sources['def_Mod'][f'{key}']}\n")
                    lines.append(end_line)

                    for line in lines:
                        f1.write(line)
                time.sleep(2)
                return False

    def __step3_make_case_namelist(self, sim_save_path, newlib):
        """
        Write a namelist from a text file.

        Args:
            file_path (str): Path to the text file.

        """

        with st.spinner('Making namelist... Please wait.'):
            with open(sim_save_path + f'/{newlib["Sim_casename"]}.nml', 'w') as f:
                lines = []
                end_line = "/\n\n\n"

                lines.append("&general\n")
                max_key_length = max(len(key) for key in newlib['general'].keys())
                for key in list(newlib['general'].keys()):
                    lines.append(f"    {key:<{max_key_length}} = {newlib['general'][f'{key}']}\n")
                lines.append(end_line)

                if newlib['variables']:
                    if isinstance(newlib['variables'], str): newlib['variables'] = [newlib['variables']]
                    for variable in newlib['variables']:
                        variable = variable.replace(' ', '_')
                        lines.append(f"&{variable}\n")
                        for info in newlib['info_list']:
                            lines.append(f"    {info} = {newlib[variable][info]}\n")
                        lines.append(end_line)
                for line in lines:
                    f.write(line)
                time.sleep(2)

            if isinstance(self.sim_sources['general']['Case_lib'], str): self.sim_sources['general']['Case_lib'] = [
                self.sim_sources['general']['Case_lib']]
            if newlib["Sim_casename"] not in self.sim_sources['general']['Case_lib']:
                self.sim_sources['general']['Case_lib'].append(newlib["Sim_casename"])
            self.sim_sources['def_nml'][newlib["Sim_casename"]] = sim_save_path + f'/{newlib["Sim_casename"]}.nml'

            with open('./GUI/Namelist_lib/Simulation_lib.nml', 'w') as f1:
                lines = []
                end_line = "/\n\n\n"
                lines.append("&general\n")
                max_key_length = max(len(key) for key in self.sim_sources['general'].keys())
                for key, value in self.sim_sources['general'].items():
                    if isinstance(value, str): value = [value]
                    lines.append(f"    {key:<{max_key_length}} = {', '.join(value)}\n")
                lines.append(end_line)

                lines.append("&def_nml\n")
                max_key_length = max(len(key) for key in self.sim_sources['def_nml'].keys())
                for key in list(self.sim_sources['def_nml'].keys()):
                    lines.append(f"    {key:<{max_key_length}} = {self.sim_sources['def_nml'][f'{key}']}\n")
                lines.append(end_line)

                lines.append("&def_Mod\n")
                max_key_length = max(len(key) for key in self.sim_sources['def_Mod'].keys())
                for key in list(self.sim_sources['def_Mod'].keys()):
                    lines.append(f"    {key:<{max_key_length}} = {self.sim_sources['def_Mod'][f'{key}']}\n")
                lines.append(end_line)

                for line in lines:
                    f1.write(line)
            time.sleep(0.8)
        # return True

    def __step3_check_newcase_namelist(self, newlib):
        check_state = 0

        general = newlib['general']
        model_key = "Sim_casename"
        timezone_key = "timezone"
        data_groupby_key = "data_groupby"
        dir_key = "dir"
        tim_res_key = "tim_res"

        # 这之后的要区分检查--------------------------------
        data_type_key = "data_type"
        fulllist_key = "fulllist"
        geo_res_key = "grid_res"
        suffix_key = "suffix"
        prefix_key = "prefix"
        syear_key = "syear"
        eyear_key = "eyear"

        if len(newlib[model_key]) <= 1:
            st.error(f'{model_key} should be a string longer than one, please check {model_key}.',
                     icon="⚠")
            check_state += 1

        if newlib['Mod']:
            for key in [timezone_key, data_groupby_key, dir_key, tim_res_key]:
                if isinstance(general[key], str):
                    if len(general[key]) < 1:
                        st.error(f'{key} should be a string longer than one, please check {key}.',
                                 icon="⚠")
                        check_state += 1
                elif isinstance(general[key], float) | isinstance(general[key], int):
                    if general[key] < -12. or general[key] > 12.:
                        st.error(f'"please check {key}.', icon="⚠")
                        check_state += 1

            if general[data_type_key] == "Grid":
                if (general[geo_res_key] == 0.0):
                    st.error(f"Geo Resolution should be larger than zero when data_type is 'geo', please check.", icon="⚠")
                    check_state += 1
                elif (suffix_key in general) or (prefix_key in general):
                    if isinstance(general[suffix_key], str) | (isinstance(general[prefix_key], str)):
                        if len(general[suffix_key]) == 0 & len(general[prefix_key]) == 0:
                            st.error(f'"suffix or prefix should be a string longer than one, please check.', icon="⚠")
                            check_state += 1
                elif general[eyear_key] < general[syear_key]:
                    st.error(f" year should be larger than Start year, please check.", icon="⚠")
                    check_state += 1
            elif general[data_type_key] == "stn":
                if not general[fulllist_key]:
                    st.error(f"Fulllist should not be empty when data_type is 'stn'.", icon="⚠")
                    check_state += 1

            model_var = self.nl.read_namelist(general['model_namelist'])
            for selected_item in self.selected_items:
                if selected_item not in model_var.keys():
                    st.warning(f"{selected_item.replace('_', ' ')} not in {newlib['Mod']}, please make sure!", icon="⚠")

            if newlib['variables'] and newlib['info_list']:
                for variable in newlib['variables']:
                    warning = 0
                    if (suffix_key in newlib[variable]) and (prefix_key in newlib[variable]):
                        if isinstance(newlib[variable][suffix_key], str) | (isinstance(newlib[variable][prefix_key], str)):
                            if len(newlib[variable][suffix_key]) == 0 and len(newlib[variable][prefix_key]) == 0:
                                st.error(f'{variable} "suffix or prefix should be a string longer than one, please check.',
                                         icon="⚠")
                    for info in newlib['info_list']:
                        if isinstance(newlib[variable][info], str) and info not in [suffix_key, prefix_key]:
                            if len(newlib[variable][info]) == 0:
                                warning += 1
                    if warning > 0:
                        st.warning(f'{variable} exist warning, please check.', icon="⚠")



        else:
            st.error(f"Please select Model first.", icon="⚠")
            check_state += 1

        if check_state > 0:
            return False
        if check_state == 0:
            return True

    
    def step3_mange_simcases(self):
        if 'step3_remove' not in st.session_state:
            st.session_state['step3_remove'] = False
        sim_sources = self.nl.read_namelist('./GUI/Namelist_lib/Simulation_lib.nml')

        def get_cases(sim_sources):
            case_item = {}
            for item in sim_sources['general']['Case_lib']:
                case_item[item] = False

            st.subheader("Simulation Cases ....", divider=True)
            cols = itertools.cycle(st.columns(2))
            for item in case_item:
                col = next(cols)
                case_item[item] = col.checkbox(item, key=item,
                                               value=case_item[item])
            return [item for item, value in case_item.items() if value]

        cases = get_cases(sim_sources)

        def remove(cases):
            with st.spinner('Making namelist... Please wait.'):
                for case in cases:
                    if case in sim_sources['def_nml']:
                        if os.path.exists(sim_sources['def_nml'][case]):
                            os.remove(sim_sources['def_nml'][case])
                            st.code(f"Remove file: {sim_sources['def_nml'][case]}")
                        else:
                            st.warning(f"{case} already removed or file doesn't exist!")
                    else:
                        st.warning(f"{case} already removed or file doesn't exist!")

                with open('./GUI/Namelist_lib/Simulation_lib.nml', 'w') as f1:
                    lines = []
                    end_line = "/\n\n\n"
                    lines.append("&general\n")
                    if isinstance(sim_sources['general']['Case_lib'], str): sim_sources['general'][
                        'Case_lib'] = [
                        sim_sources['general']['Case_lib']]
                    value = [case for case in sim_sources['general']['Case_lib'] if case not in cases]
                    if isinstance(value, str): value = [value]
                    lines.append(f"    Case_lib = {', '.join(value)}\n")
                    lines.append(end_line)

                    lines.append("&def_nml\n")
                    max_key_length = max(len(key) for key in sim_sources['def_nml'].keys())
                    for key, value in sim_sources['def_nml'].items():
                        if key not in cases:
                            lines.append(f"    {key:<{max_key_length}} = {value}\n")
                    lines.append(end_line)

                    lines.append("&def_Mod\n")
                    max_key_length = max(len(key) for key in sim_sources['def_Mod'].keys())
                    for key in list(sim_sources['def_Mod'].keys()):
                        lines.append(f"    {key:<{max_key_length}} = {sim_sources['def_Mod'][f'{key}']}\n")
                    lines.append(end_line)

                    for line in lines:
                        f1.write(line)
                    # time.sleep(2)
            for item in st.session_state.sim_data['general'].keys():
                for case in cases:
                    if case in st.session_state.sim_data['general'][item]:
                        st.session_state.sim_change['general']=True
                st.session_state.sim_data['general'][item] = [value for value in st.session_state.sim_data['general'][item] if value not in cases]


        def define_remove():
            st.session_state.step3_remove = True

        disable = False
        if not cases:
            disable = True

        remove_contain = st.container()

        if st.button('Remove cases', on_click=define_remove, help='Press to remove cases', disabled=disable):
            with remove_contain:
                remove(cases)

        st.divider()

        def define_back(remove_contain, warn):
            if not disable:
                if not st.session_state.step3_remove:
                    with remove_contain:
                        remove(cases)
                st.session_state.step3_remove = False
            # else:
            # remove_contain.error('Remove file failed!')
            # time.sleep(0.8)
            st.session_state.step3_mange_cases = False

        st.button('Finish and back to select page', on_click=define_back, args=(remove_contain, 'remove_contain'),
                  help='Finish add and go back to select page')

    
    def step3_make(self):
        sim_general = st.session_state.sim_data['general']
        def_nml = st.session_state.sim_data['def_nml']

        if sim_general and def_nml:
            st.session_state.step3_check = []
            for i, (source, path), tab in zip(range(len(def_nml)), def_nml.items(), st.tabs(def_nml.keys())):
                if source not in st.session_state.sim_data:
                    st.session_state.sim_data[source] = self.nl.read_namelist(path)
                tab.subheader(f':blue[{source} Simulation checking ....]', divider=True)
                with tab:
                    self.__step3_make_sim_info(i, source, st.session_state.sim_data[source], path, sim_general)  # self.ref

            if all(st.session_state.step3_check):
                st.session_state.step3_make_check = True
            else:
                st.session_state.step3_make_check = False
        else:
            st.error('Please select your case first!')
            st.session_state.step3_make_check = False
            st.divider()

        def define_step1():
            st.session_state.step3_make = False

        def define_step2():
            st.session_state.step3_nml = True

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.button(':back: Previous step', on_click=define_step1, help='Go to Simulation set page')
        with col4:
            st.button('Next step :soon: ', on_click=define_step2, help='Go to Simulation nml page')

    
    def __step3_make_sim_info(self, i, source, source_lib, file, sim_general):
        def sim_editor_change(key, editor_key, source):
            source_lib[key][editor_key] = st.session_state[f"{source}_{key}_{editor_key}"]
            st.session_state.sim_change[source] = True

        def set_data_type(compare_tres_value):
            my_list = ['stn', 'grid']
            index = my_list.index(compare_tres_value.lower())
            return index

        def set_data_groupby(compare_tres_value):
            my_list = ['hour', 'day', 'month', 'year', 'single']
            index = my_list.index(compare_tres_value.lower())
            return index

        import itertools
        with st.container(height=None, border=True):
            key = 'general'
            cols = itertools.cycle(st.columns(3))
            for item in source_lib[key].keys():
                if item not in ["model_namelist", "dir", "fulllist"]:
                    col = next(cols)
                    if item in ['prefix', 'suffix']:
                        source_lib[key][item] = col.text_input(f'{i}. Set {item}: ',
                                                               value=source_lib[key][item],
                                                               key=f"{source}_{key}_{item}",
                                                               on_change=sim_editor_change,
                                                               args=(key, item, source),
                                                               placeholder=f"Set your Simulation {item}...")
                    elif item in ['timezone', 'grid_res']:
                        source_lib[key][item] = col.number_input(f"{i}. Set {item}: ",
                                                                 value=float(source_lib[key][item]),
                                                                 key=f"{source}_{key}_{item}",
                                                                 on_change=sim_editor_change,
                                                                 args=(key, item, source),
                                                                 placeholder=f"Set your Simulation {item}...")
                    elif item in ['syear', 'eyear']:
                        source_lib[key][item] = col.number_input(f"{i}. Set {item}:",
                                                                 format='%04d', step=int(1),
                                                                 value=source_lib[key][item],
                                                                 key=f"{source}_{key}_{item}",
                                                                 on_change=sim_editor_change,
                                                                 args=(key, item, source),
                                                                 placeholder=f"Set your Simulation {item}...")
                    elif item == 'tim_res':
                        source_lib[key][item] = col.selectbox(f'{i}. Set Time Resolution: ',
                                                              options=('hour', 'day', 'month', 'year'),
                                                              index=set_data_groupby(source_lib[key][item]),
                                                              placeholder=f"Set your Simulation Time Resolution (default={source_lib[key][item]})...")
                    elif item == 'data_groupby':
                        source_lib[key][item] = col.selectbox(f'{i}. Set Data groupby: ',
                                                              options=('hour', 'day', 'month', 'year', 'single'),
                                                              index=set_data_groupby(source_lib[key][item]),
                                                              placeholder=f"Set your Simulation Data groupby (default={source_lib[key][item]})...")
                    elif item == 'data_type':
                        source_lib[key][item] = col.selectbox(f'{i}. Set Data type: ',
                                                              options=('stn', 'grid'),
                                                              index=set_data_type(source_lib[key][item]),
                                                              placeholder=f"Set your Simulation Data type (default={source_lib[key][item]})...")

            source_lib[key][f"dir"] = st.text_input(f'{i}. Set Data Dictionary: ',
                                                    value=source_lib[key][f"dir"],
                                                    key=f"{source}_{key}_dir",
                                                    on_change=sim_editor_change,
                                                    args=(key, "dir", source),
                                                    placeholder=f"Set your Simulation Dictionary...")
            if source_lib[key]['data_type'] == 'stn':
                source_lib[key][f"fulllist"] = st.text_input(f'{i}. Set Fulllist File: ',
                                                             value=source_lib[key][f"fulllist"],
                                                             key=f"{source}_{key}_fulllist",
                                                             on_change=sim_editor_change,
                                                             args=(key, "fulllist", source),
                                                             placeholder=f"Set your Simulation Fulllist file...")

            for key, values in source_lib.items():
                if key != 'general' and key in self.selected_items:
                    if source in sim_general[f'{key}_sim_source'] and len(source_lib[key]) > 0:
                        st.divider()
                        st.write(f'##### :blue[{key.replace("_", " ")}]')
                        cols = itertools.cycle(st.columns(2))
                        for info in source_lib[key].keys():
                            col = next(cols)
                            source_lib[key][info] = col.text_input(
                                f'{i}. Set {info}: ',
                                value=source_lib[key][info],
                                key=f"{source}_{key}_{info}",
                                on_change=sim_editor_change,
                                args=(key, info, source),
                                placeholder=f"Set your Reference Dictionary...")

            st.session_state.step3_check.append(self.__step3_makecheck(source_lib, source))

            # st.divider()

            # def write_nml(nml_dict, output_file):
            #     """
            #     将字典数据重新写回 .nml 文件。
            #     """
            #     with open(output_file, 'w') as f:
            #         # 确保 'general' 部分总是第一个
            #         if 'general' in nml_dict:
            #             f.write(f'&general\n')
            #             max_key_length = max(len(key) for key in nml_dict['general'].keys())
            #             for key, value in nml_dict['general'].items():
            #                 f.write(f'  {key:<{max_key_length}} = {value}\n')
            #             f.write('/\n\n')
            #
            #         # 写入其他部分
            #         for section, variables in nml_dict.items():
            #             if section == 'general':
            #                 continue  # 'general' 已经处理过了
            #             f.write(f'&{section}\n')
            #             for key, value in variables.items():
            #                 f.write(f'  {key} = {value}\n')
            #             f.write('/\n\n')
            #     del f
            #
            # col1, col2 = st.columns((2, 1))
            # col1.write(
            #     ':information_source: If you change :red[Dictionary Path], Press this button to save :point_right:')
            # if col2.button('Make namelist', help='Yes, this is the one.', key=f'{i}_remake', use_container_width=True):
            #     write_nml(source_lib, file)
            #     st.success("😉 Make file successfully!!! \n Please press to Next step")

    
    def __step3_makecheck(self, source_lib, source):
        error_state = 0

        info_list = ['varname', 'varunit', ]
        general = source_lib['general']
        model_key = "model_namelist"
        timezone_key = "timezone"
        data_groupby_key = "data_groupby"
        dir_key = "dir"
        tim_res_key = "tim_res"

        # 这之后的要区分检查--------------------------------
        data_type_key = "data_type"
        fulllist_key = "fulllist"
        geo_res_key = "grid_res"
        suffix_key = "suffix"
        prefix_key = "prefix"
        syear_key = "syear"
        eyear_key = "eyear"

        if len(general[model_key]) <= 1:
            st.error(f'{model_key} should be a string longer than one, please check {model_key}.',
                     icon="⚠")
            error_state += 1

        for key in [timezone_key, data_groupby_key, dir_key, tim_res_key]:
            if isinstance(general[key], str):
                if len(general[key]) <= 1:
                    st.error(f'{key} should be a string longer than one, please check {key}.',
                             icon="⚠")
                    error_state += 1
            elif isinstance(general[key], float) | isinstance(general[key], int):
                if general[key] < -12. or general[key] > 12.:
                    st.error(f'"please check {key}.', icon="⚠")
                    error_state += 1

        if general[data_type_key] == "Grid":
            if (general[geo_res_key] == 0.0):
                st.error(f"Geo Resolution should be larger than zero when data_type is 'geo', please check.", icon="⚠")
                error_state += 1
            elif (isinstance(general[suffix_key], str)) | (isinstance(general[prefix_key], str)):
                if (len(general[suffix_key]) == 0) & (len(general[prefix_key]) == 0):
                    st.error(f'"suffix or prefix should be a string longer than one, please check.', icon="⚠")
                    error_state += 1
            elif general[eyear_key] < general[syear_key]:
                st.error(f" year should be larger than Start year, please check.", icon="⚠")
                error_state += 1
            for key in [syear_key, eyear_key]:
                if isinstance(general[key], int):
                    st.error(f'"please check {key}.', icon="⚠")
                    error_state += 1

        elif general[data_type_key] == "stn":
            if not general[fulllist_key]:
                st.error(f"Fulllist should not be empty when data_type is 'stn'.", icon="⚠")
                error_state += 1

        # model_var = self.nl.read_namelist(general['model_namelist'])
        # for selected_item in self.selected_items:
        #     if selected_item not in model_var.keys():
        #         st.warning(f"{selected_item.replace('_', ' ')} not in {general[model_key]}, please make sure!", icon="⚠")

        if source not in st.session_state.step3_errorlist:
            st.session_state.step3_errorlist[source] = []
        if error_state > 0:
            st.session_state.step3_errorlist[source].append(2)
            st.session_state.step3_errorlist[source] = list(np.unique(st.session_state.step3_errorlist[source]))
            return False
        if error_state == 0:
            if (source in st.session_state.step3_errorlist) & (2 in st.session_state.step3_errorlist[source]):
                st.session_state.step3_errorlist[source] = list(
                    filter(lambda x: x != 2, st.session_state.step3_errorlist[source]))
                st.session_state.step3_errorlist[source] = list(np.unique(st.session_state.step3_errorlist[source]))
            return True

    
    def step3_sim_nml(self):

        step3_disable=False

        Mod = self.sim_sources['def_Mod']
        if st.session_state.step3_set_check & st.session_state.step3_make_check:
            for source, path in st.session_state.sim_data['def_nml'].items():
                source_lib = st.session_state.sim_data[source]
                st.subheader(source, divider=True)
                for key, value in Mod.items():
                    if source_lib["general"][f"model_namelist"] == value:
                        model_info = f'Model: {key}'
                path_info = f'Root Dictionary: {source_lib["general"][f"dir"]}'
                key = 'general'
                if source_lib[key]['data_type'] == 'stn':
                    path_info = path_info + f'\nFulllist File: {source_lib["general"][f"fulllist"]}'

                st.code(f'''{model_info}\n{path_info}''', language='shell', line_numbers=True, wrap_lines=True)
            st.session_state.step3_sim_nml = False
            st.session_state.step3_sim_check = True
        else:
            step3_disable=True
            if not st.session_state.step3_set_check:
                formatted_keys = ", ".join(
                    key.replace('_', ' ') for key, value in st.session_state.step3_errorlist.items() if 1 in value)
                st.error(
                    f'There exist error in set page, please check {formatted_keys} first! Set your simulation case.',
                    icon="🚨")
            if not st.session_state.step3_make_check:
                formatted_keys = ", ".join(
                    key.replace('_', ' ') for key, value in st.session_state.step3_errorlist.items() if 2 in value)
                st.error(f'There exist error in Making page, please check {formatted_keys} first!', icon="🚨")
            st.session_state.step3_sim_nml = False
            st.session_state.step3_sim_check = False

        def write_nml(nml_dict, output_file):
            """
            将字典数据重新写回 .nml 文件。
            """
            with open(output_file, 'w') as f:
                # 确保 'general' 部分总是第一个
                if 'general' in nml_dict:
                    f.write(f'&general\n')
                    max_key_length = max(len(key) for key in nml_dict['general'].keys())
                    for key, value in nml_dict['general'].items():
                        f.write(f'  {key:<{max_key_length}} = {value}\n')
                    f.write('/\n\n')

                # 写入其他部分
                for section, variables in nml_dict.items():
                    if section == 'general':
                        continue  # 'general' 已经处理过了
                    f.write(f'&{section}\n')
                    for key, value in variables.items():
                        f.write(f'  {key} = {value}\n')
                    f.write('/\n\n')
            del f

        def make():
            for key, value in st.session_state.sim_change.items():
                if key == 'general' and value:
                    if st.session_state.step3_sim_check & (not st.session_state.step3_sim_nml):
                        st.session_state.step3_sim_nml = self.__step3_make_sim_namelist(
                            st.session_state.generals['simulation_nml'], self.selected_items, st.session_state.sim_data)
                    if st.session_state.step3_sim_nml:
                        st.success("😉 Make file successfully!!!")
                        st.session_state.sim_change[key] = False
                elif key != 'general' and value:
                    file = st.session_state.sim_data['def_nml'][key]
                    source_lib = st.session_state.sim_data[key]
                    write_nml(source_lib, file)
                    st.session_state.sim_change[key] = False

        def define_step1():
            st.session_state.step3_nml = False

        def define_step2(make_contain, smake):
            st.write()
            if not st.session_state.step3_sim_check:
                st.session_state.step4_set = False
            else:
                with make_contain:
                    if any(v for v in st.session_state.sim_change.values()):
                        make()
                if st.session_state.get('switch_button1', True):
                    st.session_state.step4_set = True
                    st.session_state.step3 = True
                    st.session_state.switch_button1_onclick = +1
                    st.session_state['menu_option'] = (switch_button_index(st.session_state.selected) + 1) % 4

        def switch_button_index(select):
            my_list = ["Home", "Evaluation", "Running", 'Visualization']
            index = my_list.index(select)
            return index

        st.divider()
        make_contain = st.container()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.button(':back: Previous step', on_click=define_step1, help='Go to Simulation Making page')
        with col4:
            st.button('Next step :soon: ', help='Press to go to Run page', on_click=define_step2,
                      args=(make_contain, 'make'),disabled=step3_disable,
                      key='switch_button1')

    def __step3_make_sim_namelist(self, file_path, selected_items, sim_data):
        general = sim_data['general']
        def_nml = sim_data['def_nml']

        with st.spinner('Making namelist... Please wait.'):
            if st.session_state.step3_sim_check:
                with open(file_path, 'w') as f:
                    lines = []
                    end_line = "/\n\n\n"

                    lines.append("&general\n")
                    max_key_length = max(len(f"{key}") for key in general.keys())
                    for item in selected_items:
                        key = f"{item}_sim_source"
                        lines.append(f"    {key:<{max_key_length}} = {','.join(general[f'{item}_sim_source'])}\n")
                    lines.append(end_line)

                    lines.append("&def_nml\n")
                    max_key_length = max(len(key) for key in def_nml.keys())
                    for key, value in def_nml.items():
                        lines.append(f"    {key:<{max_key_length}} = {value}\n")
                    lines.append(end_line)
                    for line in lines:
                        f.write(line)
                    time.sleep(2)
                    return True
