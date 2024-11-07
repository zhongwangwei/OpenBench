# -*- coding: utf-8 -*-
import glob
import os
import re
import subprocess
import time
import streamlit as st
from PIL import Image
from collections import ChainMap
import xarray as xr
import numpy as np
from tqdm import tqdm
from stqdm import stqdm
from time import sleep
import pandas as pd
import sys
import io


def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = (time_end - time_start) / 60.
        st.write('%s cost time: %.3f min' % (func.__name__, time_spend))
        return result

    return func_wrapper


class run_validation:
    def __init__(self):
        self.author = "Qingchen Xu/xuqingchen0@gmail.com"
        self.coauthor = "Zhongwang Wei/@gmail.com"

    def set_errors(self):
        # st.json(st.session_state, expanded=False)
        e = RuntimeError('This is an exception of type RuntimeError.'
                         'No data has been uploaded or validation data has not been set.')
        st.exception(e)
        # st.error('No data has been uploaded or validation data has not been set.', icon="🚨")

    def run(self):
        # if 'run_onclick' not in st.session_state:
        #     st.session_state.run_onclick = 0
        self.__print_welcome_message()

        st.divider()
        col1, col4 = st.columns((2, 1.7))
        col1.write(':point_down: Press button to running :orange[Openbench]')
        col4.write(' Click to passing through the running steps :point_down:')  # 点击按钮调过运行步骤

        if 'status' not in st.session_state:
            st.session_state.status = ''

        # def define_run():
        #     st.session_state.run_onclick += 1

        if "status_message" not in st.session_state:
            st.session_state["status_message"] = "***Running Pages...***"

        col1, col2, col4 = st.columns(3)
        st.divider()

        if col1.button('Run', use_container_width=True):
            status = st.status(label="***Running Evaluation...***", expanded=False)
            st.session_state.status = self.Openbench_processing(status)
            st.info('More info please check task_log.txt')
        elif col4.button('Pass', use_container_width=True):
            st.session_state.status = 'complete'
            # st.session_state['status_message'] = "***Running Pages...***"

        # if st.session_state.run_onclick and st.session_state.step4_run and st.session_state['status_message'] != "***Running Pages...***" and st.session_state.status!='':
        #     status = st.status(label="***Running Pages...***", expanded=False)
        #     with open("task_log.txt", "r", encoding='utf-8') as f:
        #         for line in f:
        #             return_status = self.__process_line(line, status)
        #         if st.session_state['status_message'] == f"***:red[Evaluation Error]***":
        #             status.update(label=st.session_state['status_message'], state="error", expanded=False)
        #         elif st.session_state['status_message'] == f"***Evaluation done***":
        #             status.update(label=st.session_state['status_message'], state="complete", expanded=False)


        if st.session_state.status == 'complete':
            st.session_state.step4_run = True
            st.success("Done!")

        elif st.session_state.status == 'error':
            st.error("There is error in your setting, please check!")
            st.session_state.step4_run = False
            st.session_state.step4 = False

        st.divider()

        def switch_button_index(select):
            my_list = ["Home", "Validation", "Running", 'Visualization']
            index = my_list.index(select)
            return index

        if st.session_state.status == 'Running':
            next_button_disable1 = True
            next_button_disable2 = True
        else:
            next_button_disable1 = False
            if not st.session_state.step4_run:
                next_button_disable2 = True
            else:
                next_button_disable2 = False

        def define_evaluation():
            if st.session_state.get('switch_button2', False):
                st.session_state.switch_button2_onclick = +1
                st.session_state['menu_option'] = (switch_button_index(st.session_state.selected) - 1) % 4
                st.session_state.step4_run = False
                st.session_state.step4 = False

        def define_visual():
            if st.session_state.get('switch_button3', False):
                st.session_state.step4 = True
                st.session_state.switch_button3_onclick = +1
                st.session_state['menu_option'] = (switch_button_index(st.session_state.selected) + 1) % 4

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.button(':point_left: Previous step', disabled=next_button_disable1, key='switch_button2',
                      on_click=define_evaluation,
                      help='Press back to Evaluation page')
        with col4:
            st.button('Next step :point_right: ', disabled=next_button_disable2, key='switch_button3', on_click=define_visual,
                      help='Press go to Visualization page')

        @timer
        def _lunch_errors(self, run_err):
            e = RuntimeError(run_err)
            st.exception(e)

    def __print_welcome_message(self):
        """Print a more beautiful welcome message and ASCII art."""
        st.subheader('Welcome to Running Page!', divider=True)
        st.code(f'''
        \n\n
        {"=" * 80}
           ____                   ____                  _
          / __ \\                 |  _ \\                | |
         | |  | |_ __   ___ _ __ | |_) | ___ _ __   ___| |__
         | |  | | '_ \\ / _ \\ '_ \\|  _ < / _ \\ '_ \\ / __| '_ \\
         | |__| | |_) |  __/ | | | |_) |  __/ | | | (__| | | |
          \\____/| .__/ \\___|_| |_|____/ \\___|_| |_|\\___|_| |_|
                | |
                |_|                                           
        {"=" * 80}
        Welcome to OpenBench: The Open Land Surface Model Benchmark Evaluation System!
        {"=" * 80}
        This system evaluate various land surface model outputs against reference data.
        Key Features:
          • Multi-model support
          • Comprehensive variable evaluation
          • Advanced metrics and scoring
          • Customizable benchmarking
        {"=" * 80}

        \n
        ''',
                language='python',
                # line_numbers=True,
                )

        #        Initializing OpenBench Evaluation System...
        # {"=" * 80}

    @timer
    def Openbench_processing(self,status):
        st.divider()
        p = subprocess.Popen(
            f'python -u {st.session_state.openbench_path}/script/openbench.py {st.session_state["main_nml"]}',
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
            # text=True,
            # encoding='utf-8',
            # errors='ignore',
        )

        # st.session_state.run_string = ''
        log_file = open("task_log.txt", "w", encoding='utf-8')
        i = 0
        for line in p.stdout:
            if re.search(r'\033\[[0-9;]*m', line):
                line = re.sub(r'\033\[[0-9;]*m', '', line)
            
            if i <= 15:
                pass
            else:
                return_status = self.__process_line(line, status)
                if return_status:
                    return status._current_state
            log_file.write(line)
            i = i + 1

        log_file.close()
        if status._current_state != "error":
            sleep(1)
            st.session_state['status_message'] = f"***Evaluation done***"
            status.update(label=f"***Evaluation done***", state="complete", expanded=False)
        elif status._current_state == "error":
            st.session_state['status_message'] = f"***:red[Evaluation Error]***"
            sleep(0.5)
            status.update(label=f"***:red[Evaluation Error]***", state="error", expanded=False)

        return status._current_state

    def __process_line(self,line,status):
        eskip_next_line = False
        wskip_next_line = False
        error_keywords = ["error", "failed", "exception", "traceback"]
        error_keywords1 = ['File "', '", line']
        error_pattern = re.compile("|".join(error_keywords), re.IGNORECASE)
        error_file_pattern = re.compile("|".join(error_keywords1), re.IGNORECASE)
        python_error_pattern = re.compile(r"(raise|Error|Exception)")
        custom_error_pattern = re.compile(r"Error: .+ failed!")
        stop_next_line = False
        warning_keywords = ['Warning']
        warning_pattern = re.compile("|".join(warning_keywords), re.IGNORECASE)

        if error_pattern.search(line):
            status.update(label=f":red[{line.strip()}]", state="error", expanded=True)
            status.write(f"***:red[{line.strip()}]***")
            if python_error_pattern.search(line) and not custom_error_pattern.search(line):
                st.session_state['status_message'] = f"***:red[Evaluation Error]***"
                status.update(label=f"***:red[Evaluation Error]***", state="error", expanded=False)
                return True

        elif error_file_pattern.search(line.strip()):
            status.update(label=f":red[{line.strip()}]", state="error", expanded=True)
            status.write(f"***:red[{line.strip()}]***")
            eskip_next_line = True
        elif eskip_next_line:
            status.update(label=f":red[{line.strip()}]", state="error", expanded=True)
            status.write(f"***:red[{line.strip()}]***")
            eskip_next_line = False

        elif warning_pattern.search(line.strip()) and not wskip_next_line:
            status.write(f"***:orange[{line.strip()}]***")
            wskip_next_line = True
        elif wskip_next_line:
            status.write(f"***:orange[{line.strip()}]***")
            wskip_next_line = False
        else:
            status.update(label=f"***{line.strip()}***", state="running", expanded=False)
            status.write(f"***{line.strip()}***")
        return False