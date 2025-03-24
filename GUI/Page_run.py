# -*- coding: utf-8 -*-
import glob
import os
import re
import sys
import io
import subprocess
import time
from time import sleep
import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime import get_instance


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


class run_validation():
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
        self.__print_welcome_message()

        st.divider()
        col1, col4 = st.columns((2, 1.7))
        col1.write(':point_down: Press button to running :orange[Openbench]')
        col4.write(' Click to passing through the running steps :point_down:')  # 点击按钮调过运行步骤

        if 'status' not in st.session_state:
            st.session_state.status = ''


        if "status_message" not in st.session_state:
            st.session_state["status_message"] = "***Running Pages...***"

        col1, col2, col4 = st.columns(3)
        st.divider()

        if col1.button('Run', use_container_width=True):
            status = st.status(label="***Running Evaluation...***", expanded=False)
            st.session_state.status = self.Openbench_processing(status)
            st.info(f'More info please check {st.session_state.running_log_file}')
        if col4.button('Pass', use_container_width=True):
            st.session_state.status = 'complete'


        if st.session_state.status == 'complete':
            st.session_state.step4_run = True
            st.success("Done!")

        elif st.session_state.status == 'error':
            st.error("There is error in your setting, please check!")
            st.session_state.step4_run = False
            st.session_state.step4 = False

        # st.divider()

        if st.session_state.status == 'Running':
            next_button_disable1 = True
            next_button_disable2 = True
        else:
            next_button_disable1 = False
            if not st.session_state.step4_run:
                next_button_disable2 = True
            else:
                next_button_disable2 = False

        return next_button_disable1, next_button_disable2

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
    def Openbench_processing(self, status):
        st.divider()

        p = subprocess.Popen(
            f'python -u {st.session_state.openbench_path}/script/openbench.py {st.session_state["main_nml"]}',
            shell=True,
            stdout=subprocess.PIPE,  # 捕获 stdout
            stderr=subprocess.STDOUT,  # 将 stderr 重定向到 stdout
            bufsize=1,  # 行缓冲
            universal_newlines=True,  # 文本模式
        )
        if 'running_log_file' not in st.session_state:
            st.session_state.running_log_file = ''

        i = 0
        for line in p.stdout:
            if re.search(r'\033\[[0-9;]*m', line):
                line = re.sub(r'\033\[[0-9;]*m', '', line)
            if i <= 15:
                pass
            else:
                if "OpenBench Log File: " in line:
                    st.session_state.running_log_file = line.replace('OpenBench Log File: ', '')
                return_status = self.__process_line(line, status)
                if return_status:
                    return status._current_state
            i = i + 1


        if status._current_state != "error":
            sleep(1)
            st.session_state['status_message'] = f"***Evaluation done***"
            status.update(label=f"***Evaluation done***", state="complete", expanded=False)
        elif status._current_state == "error":
            st.session_state['status_message'] = f"***:red[Evaluation Error]***"
            sleep(0.5)
            status.update(label=f"***:red[Evaluation Error]***", state="error", expanded=False)
        return status._current_state

    def __process_line(self, line, status):
        eskip_next_line = False
        wskip_next_line = False

        error_keywords = [" - ERROR -","error", "failed", "exception", "traceback"]
        error_keywords1 = ['File "', '", line']
        error_pattern = re.compile("|".join(error_keywords), re.IGNORECASE)
        error_file_pattern = re.compile("|".join(error_keywords1), re.IGNORECASE)

        python_error_pattern = re.compile(r"(raise|Error|Exception)")
        custom_error_pattern = re.compile(r"Error: .+ failed!")
        
        stop_next_line = False
        warning_keywords = ['Warning']
        warning_pattern = re.compile("|".join(warning_keywords), re.IGNORECASE)
        log_warning_pattern = re.compile("|".join([' - WARNNING -']), re.IGNORECASE)



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

        elif log_warning_pattern.search(line.strip()) and not wskip_next_line:
            status.write(f"***:orange[{line.strip()}]***")

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
