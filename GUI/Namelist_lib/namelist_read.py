import numpy as np
import os
import glob
import pandas as pd
import importlib
import re
import sys
from typing import Dict, Any, Tuple, List, Union

import streamlit as st
import xarray as xr
from Namelist_lib.find_path import FindPath


class NamelistReader():
    """
    A class for reading and processing namelist files.
    """

    def __init__(self):
        """
        Initialize the NamelistReader with metadata and error settings.
        """
        self.name = 'namelist_read'
        self.version = '0.1'
        self.release = '0.1'
        self.date = 'Mar 2023'
        self.author = "Zhongwang Wei / zhongwang007@gmail.com"

        # Ignore all numpy warnings
        np.seterr(all='ignore')

    @staticmethod
    def strtobool(val: str) -> int:
        """
        Convert a string representation of truth to 1 (true) or 0 (false).

        Args:
            val (str): The string to convert.

        Returns:
            int: 1 for true values, 0 for false values.

        Raises:
            ValueError: If the input string is not a valid truth value.
        """
        val = val.lower()
        if val in ('y', 'yes', 't', 'true', 'on', '1'):
            return 1
        elif val in ('n', 'no', 'f', 'false', 'off', '0'):
            return 0
        else:
            raise ValueError(f"Invalid truth value: {val}")

    @staticmethod
    def select_variables(namelist: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select variables from namelist if the value is truthy.

        Args:
            namelist (Dict[str, Any]): The namelist dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing only the truthy values.
        """
        return {k: v for k, v in namelist.items() if v}

    def read_namelist(self, file_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Read a namelist from a text file.

        Args:
            file_path (str): The path to the namelist file.

        Returns:
            Dict[str, Dict[str, Any]]: A nested dictionary representing the namelist structure.
        """
        namelist = {}
        current_dict = None

        def parse_value(key: str, value: str) -> Union[bool, int, float, list, str]:
            """
            Parse a string value into its appropriate type.

            Args:
                key (str): The key of the value being parsed.
                value (str): The string value to parse.

            Returns:
                Union[bool, int, float, list, str]: The parsed value.
            """
            value = value.strip()
            if key in ['suffix', 'prefix']:
                return value  # Return as string for suffix and prefix
            if value.lower() in ['true', 'false']:
                return bool(self.strtobool(value))
            elif value.replace('-', '', 1).isdigit():
                return int(value)
            elif value.replace('.', '', 1).replace('-', '', 1).isdigit():
                return float(value)
            elif ',' in value:
                return [v.strip() for v in value.split(',')]
            else:
                return value

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('&'):
                    dict_name = line[1:]
                    current_dict = {}
                    namelist[dict_name] = current_dict
                elif line.startswith('/'):
                    current_dict = None
                elif current_dict is not None:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.split('#')[0].strip()  # Remove inline comments
                    current_dict[key] = parse_value(key, value)

        return namelist

    def Update_namelist(self, nml: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        source = self.read_namelist(nml['general'][f"model_namelist"])
        for key in nml.keys():
            if key != 'general':
                try:
                    for source_key, source_value in source[key].items():
                        nml[key][source_key] = source_value
                except:
                    pass
        return nml


class Process_home_nml(NamelistReader):
    def __init__(self, initial):
        self.initial = initial
        self.path_finder = FindPath()

    def Upload(self):
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
                main_data = self.read_namelist(st.session_state['main_nml'])
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

        return check

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
            st.session_state.ref_data = self.read_namelist(ref_path)
            check = False
        else:
            check = True
        if self.__upload_sim_check(main_data, sim_path):
            st.session_state.sim_data = self.read_namelist(sim_path)
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
        ref_data = self.read_namelist(ref_path)
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
            ref_sources = self.read_namelist('./GUI/Namelist_lib/Reference_lib.nml')
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
        sim_data = self.read_namelist(sim_path)
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
                Mod_path = self.read_namelist(sim_data['def_nml'][source])['general']['model_namelist']
                Mod = self.read_namelist(Mod_path)['general']['model']
                if Mod not in Mods:
                    Mods[Mod] = Mod_path

        if error == 0:
            sim_sources = self.read_namelist('./GUI/Namelist_lib/Simulation_lib.nml')
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

    def Setting(self):
        if 'main_change' in st.session_state:
            del st.session_state['main_change']
        check = False

        st.session_state.step1_initial = 'Setting'
        if st.session_state:
            for key, value in st.session_state.items():
                if key in ['namelist_path', 'main_data', 'sim_data', 'ref_data','generals','evaluation_items','metrics','scores','comparisons','statistics']:
                    del st.session_state[key]
                elif isinstance(value, int) & ('count' in key):
                    st.session_state[key] = 0
                elif isinstance(value, bool) & (key not in ['switch_button']):
                    st.session_state[key] = False
            st.session_state.namelist_path = os.path.join(st.session_state.openbench_path, 'nml')
            st.session_state.main_data = self.initial.main()
            st.session_state.sim_data = self.initial.sim()
            st.session_state.ref_data = self.initial.ref()
            st.session_state.generals = self.initial.generals()
            st.session_state.evaluation_items = self.initial.evaluation_items()
            st.session_state.metrics = self.initial.metrics()
            st.session_state.scores = self.initial.scores()
            st.session_state.comparisons = self.initial.comparisons()
            st.session_state.statistics = self.initial.statistics()

        return check



