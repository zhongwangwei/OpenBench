import glob, shutil
import os, re
import time
import streamlit as st
from PIL import Image
from io import StringIO
from collections import ChainMap
import xarray as xr
# from streamlit_tags import st_tags
from Namelist_lib.find_path import FindPath
import numpy as np
from Namelist_lib.namelist_read import NamelistReader, GeneralInfoReader, UpdateNamelist, UpdateFigNamelist
from Namelist_lib.namelist_info import initial_setting
import sys
import subprocess
import itertools
from posixpath import normpath
from mpl_toolkits.axisartist.angle_helper import select_step
from Statistic_figlib.Fig_Mann_Kendall_Trend_Test import make_Mann_Kendall_Trend_Test
from Statistic_figlib.Fig_Correlation import make_Correlation
from Statistic_figlib.Fig_Standard_Deviation import make_Standard_Deviation
from Statistic_figlib.Fig_Hellinger_Distance import make_Hellinger_Distance
from Statistic_figlib.Fig_Z_Score import make_Z_Score
from Statistic_figlib.Fig_Functional_Response import make_Functional_Response
from Statistic_figlib.Fig_Partial_Least_Squares_Regression import make_Partial_Least_Squares_Regression


class process_info():
    def __init__(self):
        self.author = "Qingchen Xu/xuqingchen23@163.com"
        self.nl = NamelistReader()
        # self.ref_sources = self.nl.read_namelist('./GUI/Namelist_lib/Reference_lib.nml')
        self.initial = initial_setting()
        self.stat_data = self.initial.stat()
        self.classification = self.initial.classification()
        self.main_data = st.session_state.main_data
        self.ref_data = st.session_state.ref_data
        self.sim_data = st.session_state.sim_data
        self.evaluation_items = st.session_state.evaluation_items
        self.selected_items = [k for k, v in self.evaluation_items.items() if v]
        self.statistics = self.initial.statistics()
        self.stat_info = self.initial.stat_list()
        self.stat_list = {
            'Mann_Kendall_Trend_Test': 'MK_Test',
            'Correlation': 'Corr',
            'Standard_Deviation': 'SD',
            'Z_Score': 'Z_Score',
            'Functional_Response': 'FR',
            'Hellinger_Distance': 'HD',
            'Partial_Least_Squares_Regression': 'PLSR',
            'Three_Cornered_Hat': 'TCH'
        }
        self.stat_class = {
            'Mann_Kendall_Trend_Test': 'Single',
            'Correlation': 'Multi',
            'Standard_Deviation': 'Single',
            'Z_Score': 'Single',
            'Functional_Response': 'Single',
            'Hellinger_Distance': 'Multi',
            'Partial_Least_Squares_Regression': 'PLSR',
            'Three_Cornered_Hat': 'Multi'
        }
        self.set_default = self.initial.stat_default()

    # ============================================================
    def statistic_one_items(self, statistic_item, stat_items):
        s_item = self.stat_list[statistic_item]
        st.session_state.stat_items[statistic_item] = {}
        select_list = []
        for selected_item in self.selected_items:
            st.session_state.stat_items[statistic_item][selected_item] = {}
            ref_list, sim_list = self.ref_data['general'][f'{selected_item}_ref_source'], self.sim_data['general'][
                f'{selected_item}_sim_source']
            if isinstance(ref_list, str): ref_list = [ref_list]
            if isinstance(sim_list, str): sim_list = [sim_list]
            for source in ref_list + sim_list:
                st.session_state.stat_items[statistic_item][selected_item][source] = False
                if f"{s_item}_{selected_item}_{source}" in stat_items:
                    st.session_state.stat_items[statistic_item][selected_item][source] = True
                    select_list.append(f"{s_item}_{selected_item}_{source}")
        st.session_state.stat_items[statistic_item]['Data'] = {}
        unique_items = list(set(stat_items).symmetric_difference(set(select_list)))
        if unique_items:
            for item in unique_items:
                st.session_state.stat_items[statistic_item][item] = True
        del unique_items, select_list

    def get_Mann_Kendall_Trend_Test(self, statistic_item, stat_MK):
        if isinstance(stat_MK, str): stat_MK = [stat_MK]
        if statistic_item not in st.session_state.stat_items:
            self.statistic_one_items(statistic_item, stat_MK)

        case_item = st.session_state.stat_items[statistic_item]

        def stat_data_change(key, editor_key):
            if editor_key is not None:
                case_item[key][editor_key] = st.session_state[f"{key}_{editor_key}_Mann_Kendall_Trend_Test"]
            else:
                case_item[key] = st.session_state[f"{key}_Mann_Kendall_Trend_Test"]
            st.session_state.stat_change['general'] = True

        def data_text_change(key, editor_key, col):
            custom_input = st.session_state[editor_key]
            if '，' in custom_input:
                custom_input = custom_input.replace('，', ',')
            selected_options = []
            for option in custom_input.split(','):
                if len(option.strip()) > 0 and option.strip() not in case_item[key]:
                    case_item[key][option.strip()] = True
                elif option.strip() in case_item[key]:
                    col.warning(f'{option.strip()} has already been selected, please change!')
            st.session_state[editor_key] = ''
            del selected_options

        with st.popover("Mann Kendall Trend Test Infos", use_container_width=True):
            for selected_item in self.selected_items:
                st.subheader(f"Showing {selected_item}", divider=True)
                cols = itertools.cycle(st.columns(2))
                for item in case_item[selected_item].keys():
                    col = next(cols)
                    case_item[selected_item][item] = col.checkbox(item, key=f"{selected_item}_{item}_Mann_Kendall_Trend_Test",
                                                                  on_change=stat_data_change,
                                                                  args=(selected_item, item),
                                                                  value=case_item[selected_item][item])
            st.subheader(f"New items", divider=True)

            col1, col2 = st.columns((2, 2.5))
            col1.text_input("Add more", value='',
                            key=f"MK_Test_text",
                            on_change=data_text_change,
                            args=('Data', f"MK_Test_text", col2),
                            placeholder='Press Enter to add more',
                            type='default',
                            help='Using "," to separate',
                            label_visibility="visible")
            cols = itertools.cycle(st.columns(2))
            for item, value in case_item.items():
                if isinstance(value, bool):
                    col = next(cols)
                    case_item[item] = col.checkbox(item, key=f"{item}_Mann_Kendall_Trend_Test",
                                                   # on_change=stat_data_change,
                                                   # args=(item, None),
                                                   value=case_item[item])
            for item in case_item['Data'].keys():
                col = next(cols)
                case_item['Data'][item] = col.checkbox(item, key=f"Data_{item}_Mann_Kendall_Trend_Test",
                                                       on_change=stat_data_change,
                                                       args=('Data', item),
                                                       value=case_item['Data'][item])

    def get_Standard_Deviation(self, statistic_item, stat_SD):
        if isinstance(stat_SD, str): stat_SD = [stat_SD]
        if statistic_item not in st.session_state.stat_items:
            self.statistic_one_items(statistic_item, stat_SD)

        case_item = st.session_state.stat_items[statistic_item]

        def stat_data_change(key, editor_key):
            case_item[key][editor_key] = st.session_state[f"{key}_{editor_key}_Standard_Deviation"]
            st.session_state.stat_change['general'] = True

        def data_text_change(key, editor_key, col):
            custom_input = st.session_state[editor_key]
            if '，' in custom_input:
                custom_input = custom_input.replace('，', ',')
            selected_options = []
            for option in custom_input.split(','):
                if len(option.strip()) > 0 and option.strip() not in case_item[key]:
                    case_item[key][option.strip()] = True
                elif option.strip() in case_item[key]:
                    col.warning(f'{option.strip()} has already been selected, please change!')
            st.session_state[editor_key] = ''
            del selected_options

        with st.popover("Standard Deviation Infos", use_container_width=True):
            for selected_item in self.selected_items:
                st.subheader(f"Showing {selected_item}", divider=True)
                cols = itertools.cycle(st.columns(2))
                for item in case_item[selected_item].keys():
                    col = next(cols)
                    case_item[selected_item][item] = col.checkbox(item, key=f"{selected_item}_{item}_Standard_Deviation",
                                                                  on_change=stat_data_change,
                                                                  args=(selected_item, item),
                                                                  value=case_item[selected_item][item])
            st.subheader(f"New items", divider=True)

            col1, col2 = st.columns((2, 2.5))
            col1.text_input("Add more", value='',
                            key=f"SD_input",
                            on_change=data_text_change,
                            args=('Data', f"SD_input", col2),
                            placeholder='Press Enter to add more',
                            type='default',
                            help='Using "," to separate',
                            label_visibility="visible")
            cols = itertools.cycle(st.columns(2))
            for item, value in case_item.items():
                if isinstance(value, bool):
                    col = next(cols)
                    case_item[item] = col.checkbox(item, key=f"{item}_Standard_Deviation",
                                                   value=case_item[item])
            for item in case_item['Data'].keys():
                col = next(cols)
                case_item['Data'][item] = col.checkbox(item, key=f"Data_{item}_Standard_Deviation",
                                                       on_change=stat_data_change,
                                                       args=('Data', item),
                                                       value=case_item['Data'][item])

    def get_Z_Score(self, statistic_item, stat_Z_Score):
        if isinstance(stat_Z_Score, str): stat_Z_Score = [stat_Z_Score]
        if statistic_item not in st.session_state.stat_items:
            self.statistic_one_items(statistic_item, stat_Z_Score)

        case_item = st.session_state.stat_items[statistic_item]

        def stat_data_change(key, editor_key):
            if editor_key is not None:
                case_item[key][editor_key] = st.session_state[f"{key}_{editor_key}_Z_Score"]
            else:
                case_item[key] = st.session_state[f"{key}_Z_Score"]
            st.session_state.stat_change['general'] = True

        def data_text_change(key, editor_key, col):
            custom_input = st.session_state[editor_key]
            if '，' in custom_input:
                custom_input = custom_input.replace('，', ',')
            selected_options = []
            for option in custom_input.split(','):
                if len(option.strip()) > 0 and option.strip() not in case_item[key]:
                    case_item[key][option.strip()] = True
                elif option.strip() in case_item[key]:
                    col.warning(f'{option.strip()} has already been selected, please change!')
            st.session_state[editor_key] = ''
            del selected_options

        with st.popover("Z Score Infos", use_container_width=True):
            for selected_item in self.selected_items:
                st.subheader(f"Showing {selected_item}", divider=True)
                cols = itertools.cycle(st.columns(2))
                for item in case_item[selected_item].keys():
                    col = next(cols)
                    case_item[selected_item][item] = col.checkbox(item, key=f"{selected_item}_{item}_Z_Score",
                                                                  on_change=stat_data_change,
                                                                  args=(selected_item, item),
                                                                  value=case_item[selected_item][item])
            st.subheader(f"New items", divider=True)

            col1, col2 = st.columns((2, 2.5))
            col1.text_input("Add more", value='',
                            key=f"Z_Score_input",
                            on_change=data_text_change,
                            args=('Data', f"Z_Score_input", col2),
                            placeholder='Press Enter to add more',
                            type='default',
                            help='Using "," to separate',
                            label_visibility="visible")
            cols = itertools.cycle(st.columns(2))
            for item, value in case_item.items():
                if isinstance(value, bool):
                    col = next(cols)
                    case_item[item] = col.checkbox(item, key=f"{item}_Z_Score",
                                                   value=case_item[item])
            for item in case_item['Data'].keys():
                col = next(cols)
                case_item['Data'][item] = col.checkbox(item, key=f"Data_{item}_Z_Score",
                                                       on_change=stat_data_change,
                                                       args=('Data', item),
                                                       value=case_item['Data'][item])

    def statistic_multi_items(self, statistic_item, stat_items):
        s_item = self.stat_list[statistic_item]
        st.session_state.stat_items[statistic_item] = {}
        select_list = []

        def match_item(target_string, options):
            return [option for option in options if option in target_string]

        for selected_item in self.selected_items:
            st.session_state.stat_items[statistic_item][selected_item] = {}
            st.session_state.stat_items[statistic_item][selected_item]['select_list'] = []
            ref_list, sim_list = self.ref_data['general'][f'{selected_item}_ref_source'], self.sim_data['general'][
                f'{selected_item}_sim_source']
            if isinstance(ref_list, str): ref_list = [ref_list]
            if isinstance(sim_list, str): sim_list = [sim_list]
            st.session_state.stat_items[statistic_item][selected_item]['options'] = ref_list + sim_list

            import re
            pattern = re.compile(rf"{s_item}_{selected_item}*")  # .* 表示任意多个字符
            for item in stat_items:
                if pattern.match(item):
                    option = match_item(item, st.session_state.stat_items[statistic_item][selected_item]['options'])
                    select_list.append(item)
                    if frozenset(option) not in [frozenset(sl) for sl in
                                                 st.session_state.stat_items[statistic_item][selected_item]['select_list']]:
                        st.session_state.stat_items[statistic_item][selected_item][
                            f"{item.replace(f'{s_item}_{selected_item}_', '')}"] = True
                        st.session_state.stat_items[statistic_item][selected_item]['select_list'].append(option)
        st.session_state.stat_items[statistic_item]['Data'] = {}
        unique_items = list(set(stat_items).symmetric_difference(set(select_list)))
        if unique_items:
            for item in unique_items:
                st.session_state.stat_items[statistic_item][item] = True
        del unique_items, select_list

    def get_Correlation(self, statistic_item, stat_Corr):
        if isinstance(stat_Corr, str): stat_Corr = [stat_Corr]
        if statistic_item not in st.session_state.stat_items:
            self.statistic_multi_items(statistic_item, stat_Corr)
        case_item = st.session_state.stat_items[statistic_item]

        def stat_data_change(key, editor_key):
            case_item[key][editor_key] = st.session_state[f"{key}_{editor_key}_Corr"]
            st.session_state.stat_change['general'] = True

        def stat_submit_change(key, warn_container):
            if len(st.session_state[f"{key}_Corr_multi"]) == 2:
                sitem = '_'.join(st.session_state[f"{key}_Corr_multi"])
                if frozenset(st.session_state[f"{key}_Corr_multi"]) not in [frozenset(item) for item in
                                                                            case_item[key]['select_list']]:
                    case_item[key][sitem] = True
                    case_item[key]['select_list'].append(st.session_state[f"{key}_Corr_multi"])
                else:
                    warn_container.warning(f"Multiple items selected for {sitem}")
            elif len(st.session_state[f"{key}_Corr_multi"]) <= 1:
                warn_container.warning('Please select at least 2 items')

        def stat_data_submit(key, warn_container):
            stat_submit_change(key, warn_container)
            st.session_state[f"{key}_multi"] = []
            st.session_state.stat_change['general'] = True

        def data_text_change(key, editor_key, col):
            custom_input = st.session_state[editor_key]
            if '，' in custom_input:
                custom_input = custom_input.replace('，', ',')
            selected_options = []
            for option in custom_input.split(','):
                if len(option.strip()) > 0 and option.strip() not in case_item[key]:
                    case_item[key][option.strip()] = True
                elif option.strip() in case_item[key]:
                    col.warning(f'{option.strip()} has already been selected, please change!')
            st.session_state[editor_key] = ''
            del selected_options

        with st.popover("Variables Infos", use_container_width=True):
            for selected_item in self.selected_items:
                st.subheader(f"Showing {selected_item.replace('_', ' ')}", divider=True)
                col1, col2 = st.columns((2.5, 1))
                warn_container = st.container()
                col1.multiselect(f"{selected_item} offered",
                                 [value for value in st.session_state.stat_items[statistic_item][selected_item]['options']],
                                 default=None,
                                 key=f"{selected_item}_Corr_multi",
                                 max_selections=2,
                                 placeholder="Choose an option",
                                 label_visibility="collapsed")
                col2.button('Submit', key=f"{selected_item}_Corr_submit", on_click=stat_data_submit,
                            args=(selected_item, warn_container), use_container_width=True)

                cols = itertools.cycle(st.columns(2))
                for skey, svalue in case_item[selected_item].items():
                    if isinstance(svalue, bool):
                        col = next(cols)
                        case_item[selected_item][skey] = col.checkbox(skey, key=f"{selected_item}_{skey}_Corr",
                                                                      on_change=stat_data_change,
                                                                      args=(selected_item, skey),
                                                                      value=case_item[selected_item][skey])

            st.divider()

            st.subheader(f"New items", divider=True)
            col1, col2 = st.columns((2, 2.5))
            col1.text_input("Add more", value='',
                            key=f"Corr_input",
                            on_change=data_text_change,
                            args=('Data', f"Corr_input", col2),
                            placeholder='Press Enter to add more',
                            type='default',
                            help='Using "," to separate',
                            label_visibility="visible")

            cols = itertools.cycle(st.columns(2))
            for item, value in case_item.items():
                if isinstance(value, bool):
                    col = next(cols)
                    case_item[item] = col.checkbox(item, key=f"{item}_Corr",
                                                   value=case_item[item])
            for item in case_item['Data'].keys():
                col = next(cols)
                case_item['Data'][item] = col.checkbox(item, key=f"Data_{item}_Corr",
                                                       on_change=stat_data_change,
                                                       args=('Data', item),
                                                       value=case_item['Data'][item])

    def get_Functional_Response(self, statistic_item, stat_FD):
        if isinstance(stat_FD, str): stat_FD = [stat_FD]
        if statistic_item not in st.session_state.stat_items:
            self.statistic_multi_items(statistic_item, stat_FD)
        case_item = st.session_state.stat_items[statistic_item]

        def stat_data_change(key, editor_key):
            case_item[key][editor_key] = st.session_state[f"{key}_{editor_key}_FD"]
            st.session_state.stat_change['general'] = True

        def stat_submit_change(key, warn_container):
            if len(st.session_state[f"{key}_FD_multi"]) == 2:
                sitem = '_'.join(st.session_state[f"{key}_FD_multi"])
                if frozenset(st.session_state[f"{key}_FD_multi"]) not in [frozenset(item) for item in
                                                                          case_item[key]['select_list']]:
                    case_item[key][sitem] = True
                    case_item[key]['select_list'].append(st.session_state[f"{key}_FD_multi"])
                else:
                    warn_container.warning(f"Multiple items selected for {sitem}")
            elif len(st.session_state[f"{key}_FD_multi"]) <= 1:
                warn_container.warning('Please select at least 2 items')

        def stat_data_submit(key, warn_container):
            stat_submit_change(key, warn_container)
            st.session_state[f"{key}_multi"] = []
            st.session_state.stat_change['general'] = True

        def data_text_change(key, editor_key, col):
            custom_input = st.session_state[editor_key]
            if '，' in custom_input:
                custom_input = custom_input.replace('，', ',')
            selected_options = []
            for option in custom_input.split(','):
                if len(option.strip()) > 0 and option.strip() not in case_item[key]:
                    case_item[key][option.strip()] = True
                elif option.strip() in case_item[key]:
                    col.warning(f'{option.strip()} has already been selected, please change!')
            st.session_state[editor_key] = ''
            del selected_options

        with st.popover("Variables Infos", use_container_width=True):
            for selected_item in self.selected_items:
                st.subheader(f"Showing {selected_item.replace('_', ' ')}", divider=True)
                col1, col2 = st.columns((2.5, 1))
                warn_container = st.container()
                col1.multiselect(f"{selected_item} offered",
                                 [value for value in st.session_state.stat_items[statistic_item][selected_item]['options']],
                                 default=None,
                                 key=f"{selected_item}_FD_multi",
                                 max_selections=2,
                                 placeholder="Choose an option",
                                 label_visibility="collapsed")
                col2.button('Submit', key=f"{selected_item}_FD_submit", on_click=stat_data_submit,
                            args=(selected_item, warn_container), use_container_width=True)

                cols = itertools.cycle(st.columns(2))
                for skey, svalue in case_item[selected_item].items():
                    if isinstance(svalue, bool):
                        col = next(cols)
                        case_item[selected_item][skey] = col.checkbox(skey, key=f"{selected_item}_{skey}_FD",
                                                                      on_change=stat_data_change,
                                                                      args=(selected_item, skey),
                                                                      value=case_item[selected_item][skey])

            st.divider()

            st.subheader(f"New items", divider=True)
            col1, col2 = st.columns((2, 2.5))
            col1.text_input("Add more", value='',
                            key=f"FD_input",
                            on_change=data_text_change,
                            args=('Data', f"FD_input", col2),
                            placeholder='Press Enter to add more',
                            type='default',
                            help='Using "," to separate',
                            label_visibility="visible")

            cols = itertools.cycle(st.columns(2))
            for item, value in case_item.items():
                if isinstance(value, bool):
                    col = next(cols)
                    case_item[item] = col.checkbox(item, key=f"{item}_FD",
                                                   value=case_item[item])
            for item in case_item['Data'].keys():
                col = next(cols)
                case_item['Data'][item] = col.checkbox(item, key=f"Data_{item}_FD",
                                                       on_change=stat_data_change,
                                                       args=('Data', item),
                                                       value=case_item['Data'][item])

    def get_Hellinger_Distance(self, statistic_item, stat_HD):
        if isinstance(stat_HD, str): stat_HD = [stat_HD]
        if statistic_item not in st.session_state.stat_items:
            self.statistic_multi_items(statistic_item, stat_HD)
        case_item = st.session_state.stat_items[statistic_item]

        def stat_data_change(key, editor_key):
            case_item[key][editor_key] = st.session_state[f"{key}_{editor_key}_HD"]
            st.session_state.stat_change['general'] = True

        def stat_submit_change(key, warn_container):
            if len(st.session_state[f"{key}_HD_multi"]) == 2:
                sitem = '_'.join(st.session_state[f"{key}_HD_multi"])
                if frozenset(st.session_state[f"{key}_HD_multi"]) not in [frozenset(item) for item in
                                                                          case_item[key]['select_list']]:
                    case_item[key][sitem] = True
                    case_item[key]['select_list'].append(st.session_state[f"{key}_HD_multi"])
                else:
                    warn_container.warning(f"Multiple items selected for {sitem}")
            elif len(st.session_state[f"{key}_HD_multi"]) <= 1:
                warn_container.warning('Please select at least 2 items')

        def stat_data_submit(key, warn_container):
            stat_submit_change(key, warn_container)
            st.session_state[f"{key}_HD_multi"] = []
            st.session_state.stat_change['general'] = True

        def data_text_change(key, editor_key, col):
            custom_input = st.session_state[editor_key]
            if '，' in custom_input:
                custom_input = custom_input.replace('，', ',')
            selected_options = []
            for option in custom_input.split(','):
                if len(option.strip()) > 0 and option.strip() not in case_item[key]:
                    case_item[key][option.strip()] = True
                elif option.strip() in case_item[key]:
                    col.warning(f'{option.strip()} has already been selected, please change!')
            st.session_state[editor_key] = ''
            del selected_options

        with st.popover("Variables Infos", use_container_width=True):
            for selected_item in self.selected_items:
                st.subheader(f"Showing {selected_item.replace('_', ' ')}", divider=True)
                col1, col2 = st.columns((2.5, 1))
                warn_container = st.container()
                col1.multiselect(f"{selected_item} offered",
                                 [value for value in st.session_state.stat_items[statistic_item][selected_item]['options']],
                                 default=None,
                                 key=f"{selected_item}_HD_multi",
                                 max_selections=2,
                                 placeholder="Choose an option",
                                 label_visibility="collapsed")
                col2.button('Submit', key=f"{selected_item}_HD_submit", on_click=stat_data_submit,
                            args=(selected_item, warn_container), use_container_width=True)

                cols = itertools.cycle(st.columns(2))
                for skey, svalue in case_item[selected_item].items():
                    if isinstance(svalue, bool):
                        col = next(cols)
                        case_item[selected_item][skey] = col.checkbox(skey, key=f"{selected_item}_{skey}_HD",
                                                                      on_change=stat_data_change,
                                                                      args=(selected_item, skey),
                                                                      value=case_item[selected_item][skey])

            st.divider()

            st.subheader(f"New items", divider=True)
            col1, col2 = st.columns((2, 2.5))
            col1.text_input("Add more", value='',
                            key=f"HD_input",
                            on_change=data_text_change,
                            args=('Data', f"HD_input", col2),
                            placeholder='Press Enter to add more',
                            type='default',
                            help='Using "," to separate',
                            label_visibility="visible")

            cols = itertools.cycle(st.columns(2))
            for item, value in case_item.items():
                if isinstance(value, bool):
                    col = next(cols)
                    case_item[item] = col.checkbox(item, key=f"{item}_HD",
                                                   value=case_item[item])
            for item in case_item['Data'].keys():
                col = next(cols)
                case_item['Data'][item] = col.checkbox(item, key=f"Data_{item}_HD",
                                                       on_change=stat_data_change,
                                                       args=('Data', item),
                                                       value=case_item['Data'][item])

    def get_Three_Cornered_Hat(self, statistic_item, stat_TCH):
        if isinstance(stat_TCH, str): stat_TCH = [stat_TCH]
        if statistic_item not in st.session_state.stat_items:
            self.statistic_multi_items(statistic_item, stat_TCH)
        case_item = st.session_state.stat_items[statistic_item]

        def stat_data_change(key, editor_key):
            case_item[key][editor_key] = st.session_state[f"{key}_{editor_key}_TCH"]
            st.session_state.stat_change['general'] = True

        def stat_submit_change(key, warn_container):
            if len(st.session_state[f"{key}_TCH_multi"]) == 3:
                sitem = '_'.join(st.session_state[f"{key}_TCH_multi"])
                if frozenset(st.session_state[f"{key}_TCH_multi"]) not in [frozenset(item) for item in
                                                                           case_item[key]['select_list']]:
                    case_item[key][sitem] = True
                    case_item[key]['select_list'].append(st.session_state[f"{key}_TCH_multi"])
                else:
                    warn_container.warning(f"Multiple items selected for {sitem}")
            elif len(st.session_state[f"{key}_TCH_multi"]) <= 2:
                warn_container.warning('Please select at least 3 items')

        def stat_data_submit(key, warn_container):
            stat_submit_change(key, warn_container)
            st.session_state[f"{key}_TCH_multi"] = []
            st.session_state.stat_change['general'] = True

        def data_text_change(key, editor_key, col):
            custom_input = st.session_state[editor_key]
            if '，' in custom_input:
                custom_input = custom_input.replace('，', ',')
            selected_options = []
            for option in custom_input.split(','):
                if len(option.strip()) > 0 and option.strip() not in case_item[key]:
                    case_item[key][option.strip()] = True
                elif option.strip() in case_item[key]:
                    col.warning(f'{option.strip()} has already been selected, please change!')
            st.session_state[editor_key] = ''
            del selected_options

        with st.popover("Variables Infos", use_container_width=True):
            for selected_item in self.selected_items:
                st.subheader(f"Showing {selected_item.replace('_', ' ')}", divider=True)
                col1, col2 = st.columns((2.5, 1))
                warn_container = st.container()
                col1.multiselect(f"{selected_item} offered",
                                 [value for value in st.session_state.stat_items[statistic_item][selected_item]['options']],
                                 default=None,
                                 key=f"{selected_item}_TCH_multi",
                                 max_selections=None,
                                 placeholder="Choose an option",
                                 label_visibility="collapsed")
                col2.button('Submit', key=f"{selected_item}_TCH_submit", on_click=stat_data_submit,
                            args=(selected_item, warn_container), use_container_width=True)

                cols = itertools.cycle(st.columns(2))
                for skey, svalue in case_item[selected_item].items():
                    if isinstance(svalue, bool):
                        col = next(cols)
                        case_item[selected_item][skey] = col.checkbox(skey, key=f"{selected_item}_{skey}_TCH",
                                                                      on_change=stat_data_change,
                                                                      args=(selected_item, skey),
                                                                      value=case_item[selected_item][skey])

            st.divider()

            st.subheader(f"New items", divider=True)
            col1, col2 = st.columns((2, 2.5))
            col1.text_input("Add more", value='',
                            key=f"TCH_input",
                            on_change=data_text_change,
                            args=('Data', f"TCH_input", col2),
                            placeholder='Press Enter to add more',
                            type='default',
                            help='Using "," to separate',
                            label_visibility="visible")

            cols = itertools.cycle(st.columns(2))
            for item, value in case_item.items():
                if isinstance(value, bool):
                    col = next(cols)
                    case_item[item] = col.checkbox(item, key=f"{item}_TCH",
                                                   value=case_item[item])
            for item in case_item['Data'].keys():
                col = next(cols)
                case_item['Data'][item] = col.checkbox(item, key=f"Data_{item}_TCH",
                                                       on_change=stat_data_change,
                                                       args=('Data', item),
                                                       value=case_item['Data'][item])

    def get_Partial_Least_Squares_Regression(self, statistic_item, stat_PLSR):
        if isinstance(stat_PLSR, str): stat_PLSR = [stat_PLSR]
        if statistic_item not in st.session_state.stat_items:
            self.statistic_multi_items(statistic_item, stat_PLSR)
        case_item = st.session_state.stat_items[statistic_item]

        def stat_data_change(key, editor_key):
            case_item[key][editor_key] = st.session_state[f"{key}_{editor_key}_PLSR"]
            st.session_state.stat_change['general'] = True

        def stat_submit_change(key, warn_container):
            if len(st.session_state[f"{key}_PLSR_multi"]) >= 3:
                sitem = '_'.join(st.session_state[f"{key}_PLSR_multi"])
                if frozenset(st.session_state[f"{key}_PLSR_multi"]) not in [frozenset(item) for item in
                                                                            case_item[key]['select_list']]:
                    case_item[key][sitem] = True
                    case_item[key]['select_list'].append(st.session_state[f"{key}_PLSR_multi"])
                else:
                    warn_container.warning(f"Multiple items selected for {sitem}")
            elif len(st.session_state[f"{key}_PLSR_multi"]) <= 2:
                warn_container.warning('Please select at least 3 items')

        def stat_data_submit(key, warn_container):
            stat_submit_change(key, warn_container)
            st.session_state[f"{key}_PLSR_multi"] = []
            st.session_state.stat_change['general'] = True

        def data_text_change(key, editor_key, col):
            custom_input = st.session_state[editor_key]
            if '，' in custom_input:
                custom_input = custom_input.replace('，', ',')
            selected_options = []
            for option in custom_input.split(','):
                if len(option.strip()) > 0 and option.strip() not in case_item[key]:
                    case_item[key][option.strip()] = True
                elif option.strip() in case_item[key]:
                    col.warning(f'{option.strip()} has already been selected, please change!')
            st.session_state[editor_key] = ''
            del selected_options

        with st.popover("Variables Infos", use_container_width=True):
            for selected_item in self.selected_items:
                st.subheader(f"Showing {selected_item.replace('_', ' ')}", divider=True)
                col1, col2 = st.columns((2.5, 1))
                warn_container = st.container()
                col1.multiselect(f"{selected_item} offered",
                                 [value for value in st.session_state.stat_items[statistic_item][selected_item]['options']],
                                 default=None,
                                 key=f"{selected_item}_PLSR_multi",
                                 max_selections=None,
                                 placeholder="Choose an option",
                                 label_visibility="collapsed")
                col2.button('Submit', key=f"{selected_item}_PLSR_submit", on_click=stat_data_submit,
                            args=(selected_item, warn_container), use_container_width=True)

                cols = itertools.cycle(st.columns(2))
                for skey, svalue in case_item[selected_item].items():
                    if isinstance(svalue, bool):
                        col = next(cols)
                        case_item[selected_item][skey] = col.checkbox(skey, key=f"{selected_item}_{skey}_PLSR",
                                                                      on_change=stat_data_change,
                                                                      args=(selected_item, skey),
                                                                      value=case_item[selected_item][skey])
            st.info('Note! The :red[First] selected item will be set as :red[Y data]!')
            st.divider()

            st.subheader(f"New items", divider=True)
            col1, col2 = st.columns((2, 2.5))
            col1.text_input("Add more", value='',
                            key=f"PLSR_input",
                            on_change=data_text_change,
                            args=('Data', f"PLSR_input", col2),
                            placeholder='Press Enter to add more',
                            type='default',
                            help='Using "," to separate',
                            label_visibility="visible")

            cols = itertools.cycle(st.columns(2))
            for item, value in case_item.items():
                if isinstance(value, bool):
                    col = next(cols)
                    case_item[item] = col.checkbox(item, key=f"{item}_PLSR",
                                                   value=case_item[item])
            for item in case_item['Data'].keys():
                col = next(cols)
                case_item['Data'][item] = col.checkbox(item, key=f"Data_{item}_PLSR",
                                                       on_change=stat_data_change,
                                                       args=('Data', item),
                                                       value=case_item['Data'][item])


class visualization_statistic:
    def __init__(self):
        self.author = "Qingchen Xu/xuqingchen23@163.com"
        self.nl = NamelistReader()
        # self.ref_sources = self.nl.read_namelist('./GUI/Namelist_lib/Reference_lib.nml')
        self.initial = initial_setting()
        self.stat_data = self.initial.stat()
        self.classification = self.initial.classification()
        # self.generals = st.session_state.generals
        self.main_data = st.session_state.main_data
        self.ref_data = st.session_state.ref_data
        self.sim_data = st.session_state.sim_data
        self.evaluation_items = st.session_state.evaluation_items
        self.selected_items = [k for k, v in self.evaluation_items.items() if v]
        # self.statistics = self.initial.statistics()
        self.stat_info = self.initial.stat_list()
        self.stat_list = {
            'Mann_Kendall_Trend_Test': 'MK_Test',
            'Correlation': 'Corr',
            'Standard_Deviation': 'SD',
            'Z_Score': 'Z_Score',
            'Functional_Response': 'FR',
            'Hellinger_Distance': 'HD',
            'Partial_Least_Squares_Regression': 'PLSR',
            'Three_Cornered_Hat': 'TCH'
        }
        self.stat_class = {
            'Mann_Kendall_Trend_Test': 'Single',
            'Correlation': 'Multi',
            'Standard_Deviation': 'Single',
            'Z_Score': 'Single',
            'Functional_Response': 'Single',
            'Hellinger_Distance': 'Multi',
            'Partial_Least_Squares_Regression': 'PLSR',
            'Three_Cornered_Hat': 'Multi'
        }
        self.set_default = self.initial.stat_default()

    def set_errors(self):
        e = RuntimeError('This is an exception of type RuntimeError.'
                         'No data was found for visualization under {st.session_state.,'
                         'check that the path is correct or run the validation first.')
        st.exception(e)

    def visualizations(self):
        generals = st.session_state.main_data['general']
        case_path = os.path.join(os.path.abspath(generals['basedir']), generals['basename'], "output")  # generals['basename'],
        showing_item = []
        if generals['statistics']:
            showing_item = st.session_state.statistic_items
            # showing_item = [k for k, v in st.session_state.statistics.items() if v]
            if not showing_item:
                st.info('No statistics selected!')
        else:
            st.info('No statistics selected!')

        morandi_colors = {
            "豆沙色": "#C48E8E",
            # "雾霾蓝": "#A2B9C7",,
            "灰蓝色": "#68838B",
            "天空灰蓝": "#8A9EB6",
            "枯叶黄": "#A28F79",
            # "鼠尾草绿": "#B5C4B1",,
            "浅橄榄绿": "#9DA79A",
            "军灰绿": "#6B7169",
            # "淡灰紫": "#B4A7B6",,
            "丁香紫": "#9A94BC",
            "灰紫色": "#6E617F",
            # "燕麦色": "#D4C9B9",
            # "奶咖色": "#C2B49A",

        }

        color = "#C48E8E"
        stat_general = st.session_state.stat_data['general']
        if showing_item:
            st.markdown(f"""
            <div style="font-size:22px; font-weight:bold; color:{color}; border-bottom:3px solid {color}; padding: 5px;">
                Showing Statistics items....
            </div>""", unsafe_allow_html=True)
            st.write(' ')
            item = st.radio('showing_item', [k.replace("_", " ") for k in showing_item], index=None, horizontal=True,
                            label_visibility='collapsed')

            if item:
                self.__make_show_tab(case_path, item.replace(" ", "_"), stat_general[f'{item.replace(" ", "_")}_data_source'])

    def __make_show_tab(self, case_path, item, item_general):
        @st.cache_data
        def load_image(path):
            image = Image.open(path)
            return image

        st.cache_data.clear()
        figure_path = os.path.join(case_path, 'statistics', item)
        if item == "Mann_Kendall_Trend_Test":
            st.markdown(f"""
            <div style="font-size:22px; font-weight:bold; color:#68838B; border-bottom:3px solid #68838B; padding: 5px;">
                Select Cases!
            </div>""", unsafe_allow_html=True)
            st.write(' ')
            # st.write('#### :blue[Select Cases!]')
            icase = st.radio("Mann_Kendall_Trend_Test", [k for k in item_general],
                             index=None, horizontal=True, key=f'{item}', label_visibility='collapsed')
            st.divider()
            if icase:
                tau = glob.glob(os.path.join(figure_path, f'Mann_Kendall_Trend_Test_{icase}_output_tau.*'))[0]
                trend = glob.glob(os.path.join(figure_path, f'Mann_Kendall_Trend_Test_{icase}_output_Trend.*'))[0]
                if os.path.exists(tau):
                    image = load_image(tau)
                    st.image(image, caption=f'Case: {icase} tau', use_column_width=True)
                else:
                    st.error(f'Missing Figure for Case: {icase}', icon="⚠")

                if os.path.exists(trend):
                    image = load_image(trend)
                    st.image(image, caption=f'Case: {icase} trend', use_column_width=True)
                else:
                    st.error(f'Missing Figure for Case: {icase}', icon="⚠")

        elif (item == "Correlation"):
            st.markdown(f"""
            <div style="font-size:22px; font-weight:bold; color:#68838B; border-bottom:3px solid #68838B; padding: 5px;">
                Select Cases!
            </div>""", unsafe_allow_html=True)
            st.write(' ')
            icase = st.radio("Correlation", [k for k in item_general],
                             index=None, horizontal=True, key=f'{item}', label_visibility='collapsed')
            st.divider()
            if icase:
                filename = glob.glob(os.path.join(figure_path, f'Correlation_{icase}_output.*'))
                filename = [f for f in filename if not f.endswith('.nc')]
                try:
                    image = load_image(filename[0])
                    st.image(image, caption=f'Case: {icase}', use_column_width=True)
                except:
                    st.error(f'Missing Figure for Case: {icase}', icon="⚠")

        elif item == "Standard_Deviation":
            st.markdown(f"""
            <div style="font-size:22px; font-weight:bold; color:#68838B; border-bottom:3px solid #68838B; padding: 5px;">
                Select Cases!
            </div>""", unsafe_allow_html=True)
            st.write(' ')
            icase = st.radio("Standard_Deviation", [k for k in item_general],
                             index=None, horizontal=True, key=f'{item}', label_visibility='collapsed')
            st.divider()
            if icase:
                filename = glob.glob(os.path.join(figure_path, f'Standard_Deviation_{icase}_output.*'))
                filename = [f for f in filename if not f.endswith('.nc')]
                try:
                    image = load_image(filename[0])
                    st.image(image, caption=f'Case: {icase}', use_column_width=True)
                except:
                    st.error(f'Missing Figure for Case: {icase}', icon="⚠")

        elif item == "Z_Score":
            st.info(f'Z_Score not ready yet!', icon="ℹ️")
            # st.markdown(f"""
            # <div style="font-size:22px; font-weight:bold; color:#68838B; border-bottom:3px solid #68838B; padding: 5px;">
            #     Select Cases!
            # </div>""", unsafe_allow_html=True)
            # st.write(' ')
            # icase = st.radio("Z_Score", [k for k in item_general],
            #                  index=None, horizontal=True, key=f'{item}', label_visibility='collapsed')
            # st.divider()
            # if icase:
            #
            #     filename = glob.glob(os.path.join(figure_path, f'Z_Score_{icase}_output.*'))
            #     filename = [f for f in filename if not f.endswith('.nc')]
            #     try:
            #         image = load_image(filename[0])
            #         st.image(image, caption=f'Case: {icase}', use_column_width=True)
            #     except:
            #         st.error(f'Missing Figure for Case: {icase}', icon="⚠")

        elif item == "Functional_Response":
            st.markdown(f"""
            <div style="font-size:22px; font-weight:bold; color:#68838B; border-bottom:3px solid #68838B; padding: 5px;">
                Select Cases!
            </div>""", unsafe_allow_html=True)
            st.write(' ')
            icase = st.radio("Functional Response", [k for k in item_general],
                             index=None, horizontal=True, key=f'{item}', label_visibility='collapsed')
            st.divider()
            if icase:
                filename = glob.glob(os.path.join(figure_path, f'Functional_Response_{icase}_output.*'))
                filename = [f for f in filename if not f.endswith('.nc')]
                try:
                    image = load_image(filename[0])
                    st.image(image, caption=f'Case: {icase}', use_column_width=True)
                except:
                    st.error(f'Missing Figure for Case: {icase}', icon="⚠")


        elif (item == "Hellinger_Distance"):
            st.markdown(f"""
            <div style="font-size:22px; font-weight:bold; color:#68838B; border-bottom:3px solid #68838B; padding: 5px;">
                Select Cases!
            </div>""", unsafe_allow_html=True)
            st.write(' ')
            icase = st.radio("Hellinger _Distance", [k for k in item_general],
                             index=None, horizontal=True, key=f'{item}', label_visibility='collapsed')
            st.divider()
            if icase:
                filename = glob.glob(os.path.join(figure_path, f'Hellinger_Distance_{icase}_output.*'))
                filename = [f for f in filename if not f.endswith('.nc')]
                try:
                    image = load_image(filename[0])
                    st.image(image, caption=f'Case: {icase}', use_column_width=True)
                except:
                    st.error(f'Missing Figure for Case: {icase}', icon="⚠")


        elif item == "Partial_Least_Squares_Regression":
            st.markdown(f"""
            <div style="font-size:22px; font-weight:bold; color:#68838B; border-bottom:3px solid #68838B; padding: 5px;">
                Select Cases!
            </div>""", unsafe_allow_html=True)
            st.write(' ')
            icase = st.radio("Partial Least Squares Regression", [k for k in item_general],
                             index=None, horizontal=True, key=f'{item}', label_visibility='collapsed')
            iclass = st.radio("Partial Least Squares Regression class",
                              ['best_n_components', 'coefficients', 'intercepts', 'p_values', 'r_squared', 'anomaly', ],
                              index=None, horizontal=True, key=f'{item}_class', label_visibility='collapsed')
            st.divider()
            if icase and iclass:
                filename = glob.glob(os.path.join(figure_path, f'Partial_Least_Squares_Regression_{icase}_output_{iclass}*.*'))
                filename = [f for f in filename if not f.endswith('.nc')]
                pattern = re.compile(rf'Partial_Least_Squares_Regression_{icase}_output_{iclass}_X(\d+)')
                for file in filename:
                    try:
                        image = load_image(file)
                        if (match := pattern.search(file)):
                            x_number = int(match.group(1))
                            st.image(image, caption=f'Case: {icase}, {iclass} X{x_number}', use_column_width=True)
                        else:
                            st.image(image, caption=f'Case: {icase}, {iclass}', use_column_width=True)
                    except:
                        st.error(f'Missing Figure for Case: {icase}, {iclass}', icon="⚠")
        elif item == "Three_Cornered_Hat":
            st.info(f'Three_Cornered_Hat not ready yet!', icon="ℹ️")


class visualization_replot_statistic:
    def __init__(self):
        self.author = "Qingchen Xu/xuqingchen23@163.com"
        self.nl = NamelistReader()
        # self.ref_sources = self.nl.read_namelist('./GUI/Namelist_lib/Reference_lib.nml')
        self.initial = initial_setting()
        self.stat_data = self.initial.stat()
        self.classification = self.initial.classification()
        # self.generals = st.session_state.generals
        self.main_data = st.session_state.main_data
        self.ref_data = st.session_state.ref_data
        self.sim_data = st.session_state.sim_data
        self.evaluation_items = st.session_state.evaluation_items
        self.selected_items = [k for k, v in self.evaluation_items.items() if v]
        # self.statistics = self.initial.statistics()
        self.stat_info = self.initial.stat_list()
        self.stat_list = {
            'Mann_Kendall_Trend_Test': 'MK_Test',
            'Correlation': 'Corr',
            'Standard_Deviation': 'SD',
            'Z_Score': 'Z_Score',
            'Functional_Response': 'FR',
            'Hellinger_Distance': 'HD',
            'Partial_Least_Squares_Regression': 'PLSR',
            'Three_Cornered_Hat': 'TCH'
        }
        self.stat_class = {
            'Mann_Kendall_Trend_Test': 'Single',
            'Correlation': 'Multi',
            'Standard_Deviation': 'Single',
            'Z_Score': 'Single',
            'Functional_Response': 'Single',
            'Hellinger_Distance': 'Multi',
            'Partial_Least_Squares_Regression': 'PLSR',
            'Three_Cornered_Hat': 'Multi'
        }
        self.set_default = self.initial.stat_default()

    # -=========================================================
    def replot_statistic(self):
        generals = st.session_state.main_data['general']
        case_path = os.path.join(os.path.abspath(generals['basedir']), generals['basename'], "output")
        showing_item = []
        if generals['statistics']:
            showing_item = st.session_state.statistic_items
            if not showing_item:
                st.info('No statistics selected!')
        else:
            st.info('You haven\'t selected a statistic module!')

        tabs = st.tabs([f':orange[{k.replace("_", " ")}]' for k in showing_item])
        stat_general = st.session_state.stat_data['general']

        for i, item in enumerate(showing_item):
            with tabs[i]:
                self._prepare(case_path, item, stat_general[f'{item.replace(" ", "_")}_data_source'])

    def _prepare(self, case_path, item, item_general):
        item_path = os.path.join(case_path, 'statistics', item)
        if item == "Mann_Kendall_Trend_Test":
            if isinstance(item_general, str): item_general = [item_general]
            icase = st.radio("Mann_Kendall_Trend_Test", [k for k in item_general],
                             index=None, horizontal=True, key=f'{item}', label_visibility='collapsed')
            st.divider()
            if icase:
                try:
                    file = glob.glob(os.path.join(item_path, f'Mann_Kendall_Trend_Test_{icase}_output.nc'))[0]
                    self.__Mann_Kendall_Trend_Test(item, file, icase, item_path)
                except FileNotFoundError:
                    st.error(f'Missing File for Case: {icase}', icon="⚠")
        elif (item == "Correlation"):

            icase = st.radio("Correlation", [k for k in item_general],
                             index=None, horizontal=True, key=f'{item}', label_visibility='collapsed')
            st.divider()
            if icase:
                try:
                    file = glob.glob(os.path.join(item_path, f'Correlation_{icase}_output.nc'))[0]
                    self.__Correlation(item, file, icase, item_path)
                except:
                    st.error(f'Missing File for Case: {icase}', icon="⚠")

        elif item == "Standard_Deviation":

            icase = st.radio("Standard_Deviation", [k for k in item_general],
                             index=None, horizontal=True, key=f'{item}', label_visibility='collapsed')
            st.divider()
            if icase:
                try:
                    file = glob.glob(os.path.join(item_path, f'Standard_Deviation_{icase}_output.nc'))[0]
                    self.__Standard_Deviation(item, file, icase, item_path)
                except:
                    st.error(f'Missing File for Case: {icase}', icon="⚠")

        elif item == "Z_Score":
            st.info(f'Z_Score not ready yet!', icon="ℹ️")
            # icase = st.radio("Z_Score", [k for k in item_general],
            #                  index=None, horizontal=True, key=f'{item}', label_visibility='collapsed')
            # st.divider()
            # if icase:
            #     file = glob.glob(os.path.join(item_path, f'Z_Score_{icase}_output.nc'))[0]
            #     try:
            #         self.__Z_Score(item, file, icase, item_path)
            #     except:
            #         st.error(f'Missing File for Case: {icase}', icon="⚠")
        #
        elif item == "Functional_Response":
            icase = st.radio("Functional Response", [k for k in item_general],
                             index=None, horizontal=True, key=f'{item}', label_visibility='collapsed')
            st.divider()
            if icase:
                try:
                    file = glob.glob(os.path.join(item_path, f'Functional_Response_{icase}_output.nc'))[0]
                    self.__Functional_Response(item, file, icase, item_path)
                except:
                    st.error(f'Missing File for Case: {icase}', icon="⚠")

        elif (item == "Hellinger_Distance"):
            icase = st.radio("Hellinger Distance", [k for k in item_general],
                             index=None, horizontal=True, key=f'{item}', label_visibility='collapsed')
            st.divider()
            if icase:
                try:
                    file = glob.glob(os.path.join(item_path, f'Hellinger_Distance_{icase}_output.nc'))[0]
                    self.__Hellinger_Distance(item, file, icase, item_path)
                except:
                    st.error(f'Missing File for Case: {icase}', icon="⚠")

        elif item == "Partial_Least_Squares_Regression":
            icase = st.radio("Partial_Least_Squares_Regression", [k for k in item_general],
                             index=None, horizontal=True, key=f'{item}', label_visibility='collapsed')
            st.divider()
            if icase:
                try:
                    file = glob.glob(os.path.join(item_path, f'Partial_Least_Squares_Regression_{icase}_output.nc'))[0]
                    self.__Partial_Least_Squares_Regression(item, file, icase, item_path)
                except:
                    st.error(f'Missing File for Case: {icase}', icon="⚠")

        elif item == "Three_Cornered_Hat":
            st.info(f'Three_Cornered_Hat not ready yet!', icon="ℹ️")

    def __Mann_Kendall_Trend_Test(self, item, file, icase, case_path):
        st.cache_data.clear()
        option = {}
        make_Mann_Kendall_Trend_Test(case_path, item, icase, file, st.session_state.stat_data[item], option)

    def __Correlation(self, item, file, icase, case_path):
        st.cache_data.clear()
        option = {}
        make_Correlation(case_path, item, icase, file, st.session_state.stat_data[item], option)

    def __Standard_Deviation(self, item, file, icase, case_path):
        st.cache_data.clear()
        option = {}
        make_Standard_Deviation(case_path, item, icase, file, st.session_state.stat_data[item], option)

    def __Z_Score(self, item, file, icase, case_path):
        st.cache_data.clear()
        option = {}
        make_Z_Score(case_path, item, icase, file, st.session_state.stat_data[item], option)

    def __Functional_Response(self, item, file, icase, case_path):
        st.cache_data.clear()
        option = {}
        make_Functional_Response(case_path, item, icase, file, st.session_state.stat_data[item], option)

    def __Hellinger_Distance(self, item, file, icase, case_path):
        st.cache_data.clear()
        option = {}
        make_Hellinger_Distance(case_path, item, icase, file, st.session_state.stat_data[item], option)

    def __Partial_Least_Squares_Regression(self, item, file, icase, case_path):
        st.cache_data.clear()
        option = {}
        make_Partial_Least_Squares_Regression(case_path, item, icase, file, st.session_state.stat_data[item], option)


class Process_stastic(process_info, visualization_statistic, visualization_replot_statistic):
    def __init__(self, initial):
        self.author = "Qingchen Xu/xuqingchen23@163.com"
        self.nl = NamelistReader()
        self.path_finder = FindPath()
        # self.ref_sources = self.nl.read_namelist('./GUI/Namelist_lib/Reference_lib.nml')
        self.initial = initial
        self.stat_data = initial.stat()
        # self.generals = st.session_state.generals
        self.classification = initial.classification()
        self.main_data = st.session_state.main_data
        self.ref_data = st.session_state.ref_data
        self.sim_data = st.session_state.sim_data
        self.evaluation_items = st.session_state.evaluation_items
        self.selected_items = [k for k, v in self.evaluation_items.items() if v]
        self.statistics = initial.statistics()
        self.stat_info = initial.stat_list()
        self.stat_list = {
            'Mann_Kendall_Trend_Test': 'MK_Test',
            'Correlation': 'Corr',
            'Standard_Deviation': 'SD',
            'Z_Score': 'Z_Score',
            'Functional_Response': 'FR',
            'Hellinger_Distance': 'HD',
            'Partial_Least_Squares_Regression': 'PLSR',
            'Three_Cornered_Hat': 'TCH'
        }
        self.stat_class = {
            'Mann_Kendall_Trend_Test': 'Single',
            'Correlation': 'Multi',
            'Standard_Deviation': 'Single',
            'Z_Score': 'Single',
            'Functional_Response': 'Multi',
            'Hellinger_Distance': 'Multi',
            'Partial_Least_Squares_Regression': 'Multi',
            'Three_Cornered_Hat': 'Multi'
        }
        self.set_default = initial.stat_default()
        st.session_state['generals']['statistics'] = True
        st.session_state.main_data['general']['statistics'] = True

    def switch_button_index(self, select):
        my_list = ["Home", "Evaluation", "Running", 'Visualization', 'Statistics']
        index = my_list.index(select)
        return index

    def set_errors(self):
        # st.json(st.session_state, expanded=False)
        e = RuntimeError('This is an exception of type RuntimeError.'
                         'No data was found for statistic under {st.session_state.,'
                         'check that the namelist path is correct or run the Evaluation first.')
        st.exception(e)

    def statistic_set(self):
        if 'stat_change' not in st.session_state:
            st.session_state.stat_change = {'general': False}
        if 'stat_errorlist' not in st.session_state:
            st.session_state.stat_errorlist = {'set': {}}

        if self.main_data['general']["statistics_nml"] or st.session_state.step1_initial == 'Upload':
            st.session_state.stat_data = self.nl.read_namelist(self.main_data['general']["statistics_nml"])
            if self.__upload_stat_check(self.main_data, self.main_data['general']["statistics_nml"]):
                st.session_state.stat_data = self.nl.read_namelist(self.main_data['general']["statistics_nml"])
                check = False
            else:
                check = True
            st.session_state.step6_stat_nml = True

        self._select_items()

        self._select_cases()
        # st.write('###### :red[识别顺序打乱呢？？还是直接不管了？]')
        st.divider()

        def define_step2(make_contain, make):
            if st.session_state.main_change['statistics']:
                with make_contain:
                    self._main_nml()
            if not st.session_state.step6_stat_setect_check:
                st.session_state.step6_stat_set = False
                make_contain.error('There are some error in Pages, please check!')
            else:
                for key in st.session_state.main_change.keys():
                    st.session_state.main_change[key] = False
                st.session_state.step6_stat_set = True

        def define_visual():
            if st.session_state.get('switch_button6', False):
                st.session_state.switch_button6_onclick = +1
                st.session_state['menu_option'] = (self.switch_button_index(st.session_state.selected) - 1) % 5

        make_contain = st.container()
        col1, col2, col3, col4 = st.columns((1.5, 1, 1, 1.2))
        with col1:
            st.button(':point_left: Visualization Page', key='switch_button6', on_click=define_visual, use_container_width=True,
                      help='Press go to Visualization page')
        with col4:
            st.button('Next step :soon: ', on_click=define_step2, help='Go to Statistic make page', args=(make_contain, 'make'))

    def __upload_stat_check(self, main_data, stat_path):
        stat_data = self.nl.read_namelist(stat_path)

        selected_items = [k for k, v in main_data['statistics'].items() if v]

        error = 0
        sources = []

        for key in selected_items:
            if isinstance(stat_data['general'][f"{key}_data_source"], str): stat_data['general'][f"{key}_data_source"] = [
                stat_data['general'][f"{key}_data_source"]]
            for value in stat_data['general'][f"{key}_data_source"]:
                if value not in sources:
                    sources.append(value)
        # TODO: Update!!!!
        if error == 0:
            return True
        else:
            return False

    def _select_items(self):
        def statistics_editor_change(key, editor_key):
            statistics[key] = st.session_state[key]
            st.session_state.main_change['statistics'] = True

        statistics = self.statistics
        if st.session_state.statistics:
            statistics = st.session_state.statistics
        st.subheader("Statistics Items ....", divider=True)
        # statistics['ANOVA']=False
        col1, col2 = st.columns(2)
        with col1:
            st.checkbox('Mann Kendall Trend Test',
                        key="Mann_Kendall_Trend_Test",
                        on_change=statistics_editor_change,
                        args=("Mann_Kendall_Trend_Test", "Mann_Kendall_Trend_Test"),
                        value=statistics['Mann_Kendall_Trend_Test'])
            st.checkbox('Correlation', key="Correlation",
                        on_change=statistics_editor_change,
                        args=("Correlation", "Correlation"),
                        value=statistics['Correlation'])
            st.checkbox('Standard Deviation', key="Standard_Deviation",
                        on_change=statistics_editor_change,
                        args=("Standard_Deviation", "Standard_Deviation"),
                        value=statistics['Standard_Deviation'])
            st.checkbox('Z Score', key="Z_Score",
                        on_change=statistics_editor_change,
                        args=("Z_Score", "Z_Score"),
                        value=statistics['Z_Score'])
            st.checkbox('ANOVA', key="ANOVA",
                        on_change=statistics_editor_change,
                        disabled=True,
                        args=("ANOVA", "ANOVA"),
                        value=statistics['ANOVA'])
        with col2:
            st.checkbox('Functional Response', key="Functional_Response",
                        on_change=statistics_editor_change,
                        args=("Functional_Response", "Functional_Response"),
                        value=statistics['Functional_Response'])
            st.checkbox('Hellinger Distance', key="Hellinger_Distance",
                        on_change=statistics_editor_change,
                        args=("Hellinger_Distance", "Hellinger_Distance"),
                        value=statistics['Hellinger_Distance'])
            st.checkbox('Partial Least Squares Regression', key="Partial_Least_Squares_Regression",
                        on_change=statistics_editor_change,
                        args=("Partial_Least_Squares_Regression", "Partial_Least_Squares_Regression"),
                        value=statistics['Partial_Least_Squares_Regression'])
            st.checkbox('Three Cornered Hat', key="Three_Cornered_Hat",
                        disabled=True,
                        on_change=statistics_editor_change,
                        args=("Three_Cornered_Hat", "Three_Cornered_Hat"),
                        value=statistics['Three_Cornered_Hat'])

        st.session_state.step6_stat_setect_check = self.__stat_check_items(statistics)
        st.session_state.statistics = statistics
        st.divider()

        st.session_state.statistic_items = [k for k, v in st.session_state.statistics.items() if v]
        for statistic_item in st.session_state.statistic_items:
            st.session_state.stat_change[statistic_item] = False

    def __stat_check_items(self, statistics):
        check_state = 0
        score_all_false = False
        es_select = {}
        for key, value in statistics.items():
            if isinstance(value, bool):
                if value:
                    es_select[key] = value
                    score_all_false = False
        if not any(statistics.values()):
            st.warning(f'Please choose at least one Statistic items!', icon="⚠")
            score_all_false = True

        if score_all_false:
            check_state += 1
        else:
            formatted_keys = ", \n".join(key.replace('_', ' ') for key in es_select.keys())
            st.info(f" \n Make sure your selected Statistics Item is:      :red[{formatted_keys}]", icon="ℹ️")

        if check_state > 0:
            return False
        if check_state == 0:
            return True
        # return check state~

    def _main_nml(self):
        if st.session_state.step6_stat_setect_check:
            st.code(f"Make sure your namelist path is: \n{st.session_state.openbench_path}")
            if not os.path.exists(st.session_state.casepath):
                os.makedirs(st.session_state.casepath)
            classification = self.classification
            st.session_state['generals']['evaluation'] = False
            st.session_state['generals']['comparison'] = False
            st.session_state['generals']['statistics'] = True
            main_nml = self.__make_main_namelist(st.session_state['main_nml'],
                                                 st.session_state['generals'],
                                                 st.session_state['metrics'],
                                                 st.session_state['scores'],
                                                 st.session_state['evaluation_items'],
                                                 st.session_state['comparisons'],
                                                 st.session_state['statistics'],
                                                 classification)

        if main_nml:
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

    def __make_main_namelist(self, file_path, Generals, metrics, scores, Evaluation_Items, comparisons,
                             statistics, classification):
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
                        lines.append(f"    {key:<{max_key_length}} = {Generals[f'{key}']}\n")
                    lines.append(end_line)

                    lines.append("&evaluation_items\n")
                    lines.append(
                        "  #========================Evaluation_Items====================\n"
                        "  #*******************Ecosystem and Carbon Cycle****************\n")
                    max_key_length = max(len(key) for key in Evaluation_Items.keys())
                    for key in list(sorted(classification["Ecosystem and Carbon Cycle"], key=None, reverse=False)):
                        lines.append(f"    {key:<{max_key_length}} = {Evaluation_Items[f'{key}']}\n")

                    lines.append("  #**************************************************************\n\n\n"
                                 "  #*******************      Hydrology Cycle      ****************\n")
                    for key in list(sorted(classification["Hydrology Cycle"], key=None, reverse=False)):
                        lines.append(f"    {key:<{max_key_length}} = {Evaluation_Items[f'{key}']}\n")

                    lines.append("  #**************************************************************\n\n\n"
                                 "  #*******************  Radiation and Energy Cycle  *************\n")
                    for key in list(sorted(classification["Radiation and Energy Cycle"], key=None, reverse=False)):
                        lines.append(f"    {key:<{max_key_length}} = {Evaluation_Items[f'{key}']}\n")

                    lines.append("  #**************************************************************\n\n\n"
                                 "  #*******************         Forcings      **********************\n")
                    for key in list(sorted(classification["Forcings"], key=None, reverse=False)):
                        lines.append(f"    {key:<{max_key_length}} = {Evaluation_Items[f'{key}']}\n")

                    lines.append("  #**************************************************************\n\n\n"
                                 "  #*******************         Human Activity      **********************\n")
                    for key in list(sorted(classification["Human Activity"], key=None, reverse=False)):
                        lines.append(f"    {key:<{max_key_length}} = {Evaluation_Items[f'{key}']}\n")
                    lines.append(end_line)

                    lines.append("&metrics\n")
                    max_key_length = max(len(key) for key in metrics.keys())
                    for key, value in metrics.items():
                        lines.append(f"    {key:<{max_key_length}} = {value}\n")
                    lines.append(end_line)

                    max_key_length = max(len(key) for key in scores.keys())
                    lines.append("&scores\n")
                    for key, value in scores.items():
                        lines.append(f"    {key:<{max_key_length}} = {value}\n")
                    lines.append(end_line)

                    max_key_length = max(len(key) for key in comparisons.keys())
                    lines.append("&comparisons\n")
                    for key, value in comparisons.items():
                        lines.append(f"    {key:<{max_key_length}} = {value}\n")
                    lines.append(end_line)

                    max_key_length = max(len(key) for key in statistics.keys())
                    lines.append("&statistics\n")
                    for key, value in statistics.items():
                        lines.append(f"    {key:<{max_key_length}} = {value}\n")
                    lines.append(end_line)

                    for line in lines:
                        f.write(line)

                    del max_key_length
                    time.sleep(0.8)

                    return True
            else:
                return False

    def _select_cases(self):
        stat_general = self.stat_data
        if st.session_state.stat_data['general']:
            stat_general = st.session_state.stat_data['general']

        if 'stat_items' not in st.session_state:
            st.session_state.stat_items = {}

        for statistic_item in st.session_state.statistic_items:
            item = f"{statistic_item}_data_source"
            if item not in stat_general:
                stat_general[item] = []
            if isinstance(stat_general[item], str): stat_general[item] = [stat_general[item]]

            label_text = f"<span style='font-size: 20px;'>{statistic_item.replace('_', ' ')} cases ....</span>"
            st.markdown(f":blue[{label_text}]", unsafe_allow_html=True)
            method_function = getattr(self, f"get_{statistic_item}", None)
            method_function(statistic_item, stat_general[item])
            s_item = self.stat_list[statistic_item]
            mvalues1 = [
                f"{s_item}_{ikey}_{mkey}" for ikey, value in st.session_state.stat_items[statistic_item].items() if
                not isinstance(value, bool) if any(value.values()) for
                mkey, mvalue in value.items() if isinstance(mvalue, bool) and mvalue]
            mvalues2 = [
                f"{ikey}" for ikey, value in st.session_state.stat_items[statistic_item].items() if isinstance(value, bool) if
                value]
            if mvalues1 or mvalues2:
                stat_general[item] = mvalues1 + mvalues2
            elif not mvalues1 and not mvalues2:
                stat_general[item] = []

        formatted_keys = " \n---------------------------------------------------------------------\n".join(
            f'{key.replace("_", " ")}: {", ".join(value for value in stat_general[f"{key}_data_source"] if value)}'
            for key in st.session_state.statistic_items)
        st.code(formatted_keys, language='shell', line_numbers=True, wrap_lines=True)
        st.session_state.step6_stat_set_check = self.__stat_setcheck(stat_general)

    def __stat_setcheck(self, stat_general):
        check_state = 0
        stat_all_false = True
        for statistic_item in st.session_state.statistic_items:
            key = f"{statistic_item}_data_source"
            if stat_general[key] is None or len(stat_general[key]) == 0:
                st.warning(f'Please set at least one case in {statistic_item.replace("_", " ")}!', icon="⚠")
                check_state += 1
            if statistic_item not in st.session_state.stat_errorlist['set']:
                st.session_state.stat_errorlist['set'][statistic_item] = []

        if check_state > 0:
            st.session_state.stat_errorlist['set'][statistic_item].append(1)
            st.session_state.stat_errorlist['set'][statistic_item] = list(
                np.unique(st.session_state.stat_errorlist['set'][statistic_item]))
            return False
        if check_state == 0:
            if (statistic_item in st.session_state.stat_errorlist['set']) & (
                    1 in st.session_state.stat_errorlist['set'][statistic_item]):
                st.session_state.stat_errorlist['set'][statistic_item] = list(
                    filter(lambda x: x != 1, st.session_state.stat_errorlist['set'][statistic_item]))
                st.session_state.stat_errorlist['set'][statistic_item] = list(
                    np.unique(st.session_state.stat_errorlist['set'][statistic_item]))
            return True

    # ------------------------------------------------------

    def statistic_make(self):

        if 'step6_make_check' not in st.session_state:
            st.session_state.step6_make_check = False
        statistic_items = st.session_state.statistic_items
        stat_general = st.session_state.stat_data['general']

        if statistic_items:
            st.session_state.step6_check = []
            for i, statistic_item, tab in zip(range(len(statistic_items)), statistic_items, st.tabs(statistic_items)):
                if statistic_item not in st.session_state.stat_data:
                    st.session_state.stat_data[statistic_item] = {}
                if statistic_item not in st.session_state.stat_errorlist:
                    st.session_state.stat_errorlist[statistic_item] = {}
                self.__step6_make_stat_info(statistic_item, tab, stat_general,
                                            st.session_state.stat_data[statistic_item],
                                            st.session_state.stat_items[statistic_item])  # self.ref
            if all(st.session_state.step6_check):
                st.session_state.step6_make_check = True
            else:
                st.session_state.step6_make_check = False
        else:
            st.error('Please select your case first!')
            st.session_state.step6_make_check = False

        step6_disable = False

        def contains_value(nested_list, target):
            return any(target in sublist for sublist in nested_list)

        if st.session_state.step6_stat_set_check & st.session_state.step6_make_check:
            st.session_state.step6_stat_nml = False
            st.session_state.step6_stat_check = True
        else:
            step6_disable = True
            if not st.session_state.step6_stat_set_check:
                formatted_keys = ", ".join(
                    key.replace('_', ' ') for key, value in st.session_state.stat_errorlist['set'].items() if 1 in value)
                st.error(
                    f'There exist error in set page, please check {formatted_keys} first! Set your Statistic case.',
                    icon="🚨")
            if not st.session_state.step6_make_check:
                formatted_keys = ", ".join(
                    key.replace('_', ' ') for key, value in st.session_state.stat_errorlist.items() if
                    contains_value(value.values(), 2))
                st.error(f'There exist error in Making page, please check {formatted_keys} first!', icon="🚨")
            st.session_state.step6_stat_nml = False
            st.session_state.step6_stat_check = False

        st.divider()

        def make():
            if st.session_state.step6_stat_check & (not st.session_state.step6_stat_nml):
                st.session_state.step6_stat_nml = self.__step6_make_stat_namelist(
                    st.session_state.generals['statistics_nml'], statistic_items, st.session_state.stat_data)
                if st.session_state.step6_stat_nml:
                    st.success("😉 Make file successfully!!!")
                    for key in st.session_state.stat_change.keys():
                        st.session_state.stat_change[key] = False

        def define_step1():
            st.session_state.step6_stat_set = False

        def define_step2(make_contain, smake):
            if not st.session_state.step6_stat_check:
                st.session_state.step6_stat_make = False
            else:
                with make_contain:
                    if any(v for v in st.session_state.stat_change.values()):
                        make()
                st.session_state.step6_stat_make = True

        make_contain = st.container()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.button(':back: Previous step', on_click=define_step1, help='Go to Statistic Set page')
        with col4:
            st.button('Next step :soon: ', on_click=define_step2, help='Go to Statistic running page',
                      args=(make_contain, 'make'), disabled=step6_disable, )

    def __step6_make_stat_info(self, statistic_item, tab, stat_general, item_data, stat_items):
        morandi_colors = {
            "豆沙色": "#C48E8E",
            # "雾霾蓝": "#A2B9C7",,
            "灰蓝色": "#68838B",
            "天空灰蓝": "#8A9EB6",
            "枯叶黄": "#A28F79",
            # "鼠尾草绿": "#B5C4B1",,
            "浅橄榄绿": "#9DA79A",
            "军灰绿": "#6B7169",
            # "淡灰紫": "#B4A7B6",,
            "丁香紫": "#9A94BC",
            "灰紫色": "#6E617F",
            # "燕麦色": "#D4C9B9",
            # "奶咖色": "#C2B49A",

        }

        def item_editor_change(key1, key2):
            item_data[key1] = st.session_state[f"{key1}_{key2}"]
            st.session_state.stat_change[key1] = True

        info_list = self.stat_info[statistic_item]
        if 'other' in info_list.keys():
            cols = itertools.cycle(tab.columns(2))
            for key in info_list['other']:
                if key == 'nbins':
                    f_format = '%d'
                    f_step = int(1)
                    if key not in item_data:
                        item_data[key] = 25
                elif key == 'max_components':
                    f_format = '%d'
                    f_step = int(1)
                    if key not in item_data:
                        item_data[key] = 2
                elif key == 'n_splits':
                    f_format = '%d'
                    f_step = int(1)
                    if key not in item_data:
                        item_data[key] = 5
                elif key == 'n_jobs':
                    f_format = '%d'
                    f_step = int(1)
                    if key not in item_data:
                        item_data[key] = -1
                else:
                    f_format = '%f'
                    f_step = 0.05
                    item_data[key] = 0.05

                col = next(cols)
                item_data[key] = col.number_input(f"Set {key}:",
                                                  format=f_format, step=f_step,
                                                  value=item_data[key],
                                                  key=f"{statistic_item}_{key}",
                                                  on_change=item_editor_change,
                                                  args=(statistic_item, key),
                                                  placeholder=f"Set your {statistic_item.replace('_', ' ')} {key.replace('_', ' ')}...")
            tab.divider()
        colors = itertools.cycle(morandi_colors.values())
        for j, item in enumerate(stat_general[f"{statistic_item}_data_source"]):
            # tab.markdown(f"""
            # <div style='background-color: #B5C4B1; padding: 5px; border-radius: 10px;'>
            #     <p style="font-size:18px; font-weight:bold; color:#6E617F; border-bottom:3px solid #6E617F; padding: 5px;">
            #     {item.replace(f"{self.stat_list[statistic_item]}_", "")} ....
            # </div>
            # """, unsafe_allow_html=True)
            color = next(colors)
            tab.markdown(f"""
            <div style="font-size:18px; font-weight:bold; color:{color}; border-bottom:3px solid {color}; padding: 5px;">
                {item.replace(f"{self.stat_list[statistic_item]}_", "")} ....
            </div>
            """, unsafe_allow_html=True)
            tab.write('')

            with tab.expander(f"###### Cases: {item}", expanded=False):
                self.__make_info(statistic_item, item_data, item, info_list['general'], stat_items)

    def __make_info(self, statistic_item, item_data, item, info_list, stat_items):
        s_item = self.stat_list[statistic_item]

        def get_items(iclass, item):
            if iclass == 'Single':
                for ikey, value in stat_items.items():
                    if not isinstance(value, bool) and any(value.values()):
                        for mkey, mvalue in value.items():
                            if isinstance(mvalue, bool) and mvalue:
                                if item == f"{s_item}_{ikey}_{mkey}":
                                    return ikey, mkey
                    elif isinstance(value, bool) and value:
                        return ikey, None
            elif iclass == 'Multi':
                for ikey, value in stat_items.items():
                    if isinstance(value, bool) and value:
                        return ikey, None
                    elif isinstance(value, dict) and any(value.values()):
                        for mkey, mvalue in value.items():
                            if isinstance(mvalue, bool) and mvalue:
                                if item == f"{s_item}_{ikey}_{mkey}" and 'select_list' in value.keys():
                                    for select in value['select_list']:
                                        all_combinations = itertools.permutations(select)
                                        if mkey in ['_'.join(i) for i in all_combinations]:
                                            return ikey, select
                                elif item == f"{s_item}_Data_{mkey}" and 'select_list' not in value.keys():
                                    return ikey, None

        variable, source = get_items(self.stat_class[statistic_item], item)

        if self.stat_class[statistic_item] == 'Single':
            self.__make_single_items(variable, source, statistic_item, item_data, item, info_list)
        elif self.stat_class[statistic_item] == 'Multi':
            self.__make_Multi_items(variable, source, statistic_item, item_data, item, info_list)

    def __make_single_items(self, variable, source, statistic_item, item_data, item, info_list):
        def stat_editor_change(key1, key2, key3):
            item_data[f"{key2}_{key3}"] = st.session_state[f"{key1}_{key2}_{key3}"]
            st.session_state.stat_change[key1] = True

        def set_data_type(compare_tres_value):
            my_list = ['stn', 'grid']
            index = my_list.index(compare_tres_value.lower())
            return index

        def set_data_groupby(compare_tres_value):
            my_list = ['hour', 'day', 'month', 'year', 'single']
            index = my_list.index(compare_tres_value.lower())
            return index

        if variable != 'Data' and source is not None:
            if source in st.session_state.ref_data['general'][f"{variable}_ref_source"]:
                var_data = st.session_state.ref_data[source]
            elif source in st.session_state.sim_data['general'][f"{variable}_sim_source"]:
                var_data = st.session_state.sim_data[source]
            for i_info in info_list:
                if f"{item}_{i_info}" not in item_data:
                    if 'varname' not in var_data[variable].keys():
                        varname = self.nl.read_namelist(var_data['general']['model_namelist'])[variable]['varname']
                    else:
                        varname = var_data[variable]['varname']
                    if i_info == 'varname' and i_info not in var_data[variable].keys():
                        item_data[f"{item}_{i_info}"] = self.nl.read_namelist(var_data['general']['model_namelist'])[variable][
                            i_info]
                    elif i_info == 'dir':
                        item_data[f"{item}_dir"] = os.path.join(st.session_state.main_data['general']['basedir'],
                                                                st.session_state.main_data['general']['basename'], 'output',
                                                                'data')
                    elif i_info in ['syear', 'eyear']:
                        item_data[f"{item}_{i_info}"] = st.session_state.main_data['general'][i_info]
                    elif i_info == 'prefix':
                        if source in st.session_state.ref_data['general'][f"{variable}_ref_source"]:
                            item_data[f"{item}_{i_info}"] = f'{variable}_ref_{source}_{varname}'
                        elif source in st.session_state.sim_data['general'][f"{variable}_sim_source"]:
                            item_data[f"{item}_{i_info}"] = f'{variable}_sim_{source}_{varname}'
                    elif i_info == 'suffix':
                        item_data[f"{item}_{i_info}"] = ''
                    elif i_info == 'data_groupby':
                        item_data[f"{item}_{i_info}"] = 'single'
                    elif i_info == 'grid_res':
                        item_data[f"{item}_grid_res"] = st.session_state.main_data['general']['compare_grid_res']
                    else:
                        if i_info in var_data['general'].keys():
                            item_data[f"{item}_{i_info}"] = var_data['general'][i_info]
                        elif i_info in var_data[variable].keys():
                            item_data[f"{item}_{i_info}"] = var_data[variable][i_info]

        elif variable == 'Data':
            for i_info in info_list:
                if f"{item}_{i_info}" not in item_data:
                    item_data[f"{item}_{i_info}"] = self.set_default[i_info]
        else:
            for i_info in info_list:
                if f"{item}_{i_info}" not in item_data:
                    item_data[f"{item}_{i_info}"] = self.set_default[i_info]

        import itertools
        cols = itertools.cycle(st.columns(3))
        for i_info in sorted(info_list, key=str.lower):
            if i_info not in ["dir", "fulllist"] and f"{item}_{i_info}" in item_data.keys():
                col = next(cols)
                if i_info in ['prefix', 'suffix', 'varname']:
                    item_data[f"{item}_{i_info}"] = col.text_input(f'Set {i_info}: ',
                                                                   value=item_data[f"{item}_{i_info}"],
                                                                   key=f"{statistic_item}_{item}_{i_info}",
                                                                   on_change=stat_editor_change,
                                                                   args=(statistic_item, item, i_info),
                                                                   placeholder=f"Set your {statistic_item.replace('_', ' ')} {i_info.replace('_', ' ')}...")
                elif i_info in ['timezone', 'grid_res']:
                    item_data[f"{item}_{i_info}"] = col.number_input(f"Set {i_info}: ",
                                                                     value=float(item_data[f"{item}_{i_info}"]),
                                                                     key=f"{statistic_item}_{item}_{i_info}",
                                                                     on_change=stat_editor_change,
                                                                     args=(statistic_item, item, i_info),
                                                                     placeholder=f"Set your Simulation {item}...")
                elif i_info in ['syear', 'eyear']:
                    item_data[f"{item}_{i_info}"] = col.number_input(f" Set {i_info}:",
                                                                     format='%04d', step=int(1),
                                                                     value=item_data[f"{item}_{i_info}"],
                                                                     key=f"{statistic_item}_{item}_{i_info}",
                                                                     on_change=stat_editor_change,
                                                                     args=(statistic_item, item, i_info),
                                                                     placeholder=f"Set your Simulation {item}...")
                elif i_info == 'tim_res':
                    item_data[f"{item}_{i_info}"] = col.selectbox(f' Set Time Resolution: ',
                                                                  options=('hour', 'day', 'month', 'year'),
                                                                  index=set_data_groupby(item_data[f"{item}_{i_info}"]),
                                                                  key=f"{statistic_item}_{item}_{i_info}",
                                                                  placeholder=f"Set your Simulation Time Resolution (default={item_data[f'{item}_{i_info}']})...")
                elif i_info == 'data_groupby':
                    item_data[f"{item}_{i_info}"] = col.selectbox(f' Set Data groupby: ',
                                                                  options=('hour', 'day', 'month', 'year', 'single'),
                                                                  index=set_data_groupby(item_data[f"{item}_{i_info}"]),
                                                                  key=f"{statistic_item}_{item}_{i_info}",
                                                                  placeholder=f"Set your Simulation Data groupby (default={item_data[f'{item}_{i_info}']})...")
                elif i_info == 'data_type':
                    item_data[f"{item}_{i_info}"] = col.selectbox(f' Set Data type: ',
                                                                  options=('stn', 'grid'),
                                                                  index=set_data_type(item_data[f"{item}_{i_info}"]),
                                                                  key=f"{statistic_item}_{item}_{i_info}",
                                                                  placeholder=f"Set your Simulation Data type (default={item_data[f'{item}_{i_info}']})...")
        # item_data[f"{item}_dir"] = st.text_input(f' Set Data Dictionary: ',
        #                                          value=item_data[f"{item}_dir"],
        #                                          key=f"{statistic_item}_{item}_dir",
        #                                          on_change=stat_editor_change,
        #                                          args=(statistic_item, item, "dir",),
        #                                          placeholder=f"Set your Simulation Dictionary...")
        if not item_data[f"{item}_dir"]: item_data[f"{item}_dir"] = '/'
        try:
            item_data[f"{item}_dir"] = self.path_finder.find_path(item_data[f"{item}_dir"], f"{statistic_item}_{item}_dir",
                                                                  ['stat_change', statistic_item])
            st.code(f"Set Data Dictionary: {item_data[f'{item}_dir']}", language='shell')
        except PermissionError as e:
            if e:
                item_data[f"{item}_dir"] = '/'

        if item_data[f"{item}_data_type"] == 'stn':
            item_data[f"{item}_fulllist"] = st.text_input(f'{i}. Set Fulllist File: ',
                                                          value=item_data[f"{item}_fulllist"],
                                                          key=f"{statistic_item}_{item}_fulllist",
                                                          on_change=stat_editor_change,
                                                          args=(statistic_item, item, "fulllist"),
                                                          placeholder=f"Set your Simulation Fulllist file...")
        else:
            item_data[f"{item}_fulllist"] = ''

        st.session_state.step6_check.append(self.__step6_makecheck(item_data, item, statistic_item))

    def __make_Multi_items(self, variable, sources, statistic_item, item_data, item, info_list):
        def stat_editor_change(key1, key2, key3):
            item_data[f"{key2}_{key3}"] = st.session_state[f"{key1}_{key2}_{key3}"]
            st.session_state.stat_change[key1] = True

        def set_data_type(compare_tres_value):
            my_list = ['stn', 'grid']
            index = my_list.index(compare_tres_value.lower())
            return index

        def set_data_groupby(compare_tres_value):
            my_list = ['hour', 'day', 'month', 'year', 'single']
            index = my_list.index(compare_tres_value.lower())
            return index

        def get_x(item_data, item):
            source_config = {k: v for k, v in item_data.items() if k.startswith(f"{item}_X")}
            pattern = re.compile(rf'{item}_X(\d+)')
            x_numbers = sorted(set([int(match.group(1)) for key in source_config.keys() if (match := pattern.search(key))]))

            try:
                max_x = max(x_numbers)
                return max_x
            except Exception as e:
                st.error(f"发生错误: {e}")

        if sources is not None:
            if statistic_item == 'Three_Cornered_Hat' and f"{item}_nX" not in item_data:
                col1, col2, col3 = st.columns(3)
                item_data[f"{item}_nX"] = col1.number_input(f"Set n data: ",
                                                            value=int(len(sources)),
                                                            key=f"{statistic_item}_{item}_nX",
                                                            on_change=stat_editor_change,
                                                            disabled=True,
                                                            args=(statistic_item, f"{item}", 'n'),
                                                            placeholder=f"Set your Simulation {item}...")
                n = item_data[f"{item}X"]
            elif statistic_item == 'Partial_Least_Squares_Regression':
                col1, col2, col3 = st.columns(3)
                item_data[f"{item}_nX"] = col1.number_input(f"Set nX data: ",
                                                            min_value=2,
                                                            value=int(len(sources) - 1),
                                                            key=f"{statistic_item}_{item}_nX",
                                                            on_change=stat_editor_change,
                                                            args=(statistic_item, f"{item}", 'n'),
                                                            placeholder=f"Set your Simulation {item}...")
                n = item_data[f"{item}_nX"]
            else:
                n = len(sources)
        else:
            if statistic_item == 'Three_Cornered_Hat':
                if f"{item}_nX" not in item_data:
                    item_data[f"{item}_nX"] = get_x(item_data, item)
                col1, col2, col3 = st.columns(3)
                item_data[f"{item}_nX"] = col1.number_input(f"Set n data: ",
                                                            min_value=3,
                                                            value=int(item_data[f"{item}_nX"]),
                                                            key=f"{statistic_item}_{item}_nX",
                                                            on_change=stat_editor_change,
                                                            args=(statistic_item, f"{item}", 'n'),
                                                            placeholder=f"Set your Simulation {item}...")
                n = item_data[f"{item}_nX"]
            elif statistic_item == 'Partial_Least_Squares_Regression':
                if f"{item}_nX" not in item_data:
                    item_data[f"{item}_nX"] = get_x(item_data, item)
                col1, col2, col3 = st.columns(3)
                item_data[f"{item}_nX"] = col1.number_input(f"Set nX data: ",
                                                            min_value=2,
                                                            value=int(item_data[f"{item}_nX"]),
                                                            key=f"{statistic_item}_{item}_nX",
                                                            on_change=stat_editor_change,
                                                            args=(statistic_item, f"{item}", 'n'),
                                                            placeholder=f"Set your Simulation {item}...")
                n = item_data[f"{item}_nX"]
            else:
                n = 2

        if statistic_item != 'Partial_Least_Squares_Regression':
            if variable != 'Data' and sources is not None:
                for i, source in enumerate(sources):
                    i = i + 1
                    if source in st.session_state.ref_data['general'][f"{variable}_ref_source"]:
                        var_data = st.session_state.ref_data[source]
                    elif source in st.session_state.sim_data['general'][f"{variable}_sim_source"]:
                        var_data = st.session_state.sim_data[source]

                    for i_info in info_list:
                        if f"{item}{i}_{i_info}" not in item_data:
                            if 'varname' not in var_data[variable].keys():
                                varname = self.nl.read_namelist(var_data['general']['model_namelist'])[variable]['varname']
                            else:
                                varname = var_data[variable]['varname']
                            if i_info == 'varname' and i_info not in var_data[variable].keys():
                                item_data[f"{item}{i}_{i_info}"] = \
                                self.nl.read_namelist(var_data['general']['model_namelist'])[variable][
                                    i_info]
                            elif i_info == 'dir':
                                item_data[f"{item}{i}_dir"] = os.path.join(st.session_state.main_data['general']['basedir'],
                                                                           st.session_state.main_data['general']['basename'],
                                                                           'output',
                                                                           'data')
                            elif i_info in ['syear', 'eyear']:
                                item_data[f"{item}{i}_{i_info}"] = st.session_state.main_data['general'][i_info]
                            elif i_info == 'prefix':
                                if source in st.session_state.ref_data['general'][f"{variable}_ref_source"]:
                                    item_data[f"{item}{i}_{i_info}"] = f'{variable}_ref_{source}_{varname}'
                                elif source in st.session_state.sim_data['general'][f"{variable}_sim_source"]:
                                    item_data[f"{item}{i}_{i_info}"] = f'{variable}_sim_{source}_{varname}'
                            elif i_info == 'suffix':
                                item_data[f"{item}{i}_{i_info}"] = ''
                            elif i_info == 'data_groupby':
                                item_data[f"{item}{i}_{i_info}"] = 'single'
                            elif i_info == 'grid_res':
                                item_data[f"{item}{i}_grid_res"] = st.session_state.main_data['general']['compare_grid_res']
                            else:
                                if i_info in var_data['general'].keys():
                                    item_data[f"{item}{i}_{i_info}"] = var_data['general'][i_info]
                                elif i_info in var_data[variable].keys():
                                    item_data[f"{item}{i}_{i_info}"] = var_data[variable][i_info]


            else:
                for i in range(1, n + 1):
                    for i_info in info_list:
                        if f"{item}{i}_{i_info}" not in item_data:
                            item_data[f"{item}{i}_{i_info}"] = self.set_default[i_info]

            for i in range(1, n + 1):
                if sources is not None:
                    st.write(f'##### :violet[{sources[i - 1]}]')
                else:
                    st.write(f'##### :violet[Input Data {i}]')
                import itertools
                cols = itertools.cycle(st.columns(3))
                for i_info in sorted(info_list, key=str.lower):
                    if i_info not in ["dir", "fulllist"] and f"{item}{i}_{i_info}" in item_data.keys():
                        col = next(cols)
                        if i_info in ['prefix', 'suffix', 'varname']:
                            item_data[f"{item}{i}_{i_info}"] = col.text_input(f'Set {i_info}: ',
                                                                              value=item_data[f"{item}{i}_{i_info}"],
                                                                              key=f"{statistic_item}_{item}{i}_{i_info}",
                                                                              on_change=stat_editor_change,
                                                                              args=(statistic_item, f"{item}{i}", i_info),
                                                                              placeholder=f"Set your {statistic_item.replace('_', ' ')} {i_info.replace('_', ' ')}...")
                        elif i_info in ['timezone', 'grid_res']:
                            item_data[f"{item}{i}_{i_info}"] = col.number_input(f"Set {i_info}: ",
                                                                                value=float(item_data[f"{item}{i}_{i_info}"]),
                                                                                key=f"{statistic_item}_{item}{i}_{i_info}",
                                                                                on_change=stat_editor_change,
                                                                                args=(statistic_item, f"{item}{i}", i_info),
                                                                                placeholder=f"Set your Simulation {item}...")
                        elif i_info in ['syear', 'eyear']:
                            item_data[f"{item}{i}_{i_info}"] = col.number_input(f" Set {i_info}:",
                                                                                format='%04d', step=int(1),
                                                                                value=item_data[f"{item}{i}_{i_info}"],
                                                                                key=f"{statistic_item}_{item}{i}_{i_info}",
                                                                                on_change=stat_editor_change,
                                                                                args=(statistic_item, f"{item}{i}", i_info),
                                                                                placeholder=f"Set your Simulation {item}...")
                        elif i_info == 'tim_res':
                            item_data[f"{item}{i}_{i_info}"] = col.selectbox(f' Set Time Resolution: ',
                                                                             options=('hour', 'day', 'month', 'year'),
                                                                             index=set_data_groupby(
                                                                                 item_data[f"{item}{i}_{i_info}"]),
                                                                             key=f"{statistic_item}_{item}{i}_{i_info}",
                                                                             placeholder=f"Set your Simulation Time Resolution (default={item_data[f'{item}{i}_{i_info}']})...")
                        elif i_info == 'data_groupby':
                            item_data[f"{item}{i}_{i_info}"] = col.selectbox(f' Set Data groupby: ',
                                                                             options=('hour', 'day', 'month', 'year', 'single'),
                                                                             index=set_data_groupby(
                                                                                 item_data[f"{item}{i}_{i_info}"]),
                                                                             key=f"{statistic_item}_{item}{i}_{i_info}",
                                                                             placeholder=f"Set your Simulation Data groupby (default={item_data[f'{item}{i}_{i_info}']})...")
                        elif i_info == 'data_type':
                            item_data[f"{item}{i}_{i_info}"] = col.selectbox(f' Set Data type: ',
                                                                             options=('stn', 'grid'),
                                                                             index=set_data_type(
                                                                                 item_data[f"{item}{i}_{i_info}"]),
                                                                             key=f"{statistic_item}_{item}{i}_{i_info}",
                                                                             placeholder=f"Set your Simulation Data type (default={item_data[f'{item}{i}_{i_info}']})...")
                if not item_data[f"{item}{i}_dir"]: item_data[f"{item}{i}_dir"] = '/'
                try:
                    item_data[f"{item}{i}_dir"] = self.path_finder.find_path(item_data[f"{item}{i}_dir"],
                                                                             f"{statistic_item}_{item}{i}_dir",
                                                                             ['stat_change', statistic_item])
                    st.code(f"Set Data Dictionary: {item_data[f'{item}{i}_dir']}", language='shell')
                except PermissionError as e:
                    if e:
                        item_data[f"{item}{i}_dir"] = '/'

                if item_data[f"{item}{i}_data_type"] == 'stn':
                    if not item_data[f"{item}{i}_fulllist"]: item_data[f"{item}{i}_fulllist"] = None
                    item_data[f"{item}{i}_fulllist"] = self.path_finder.get_file(item_data[f"{item}{i}_fulllist"],
                                                                                 f"{statistic_item}_{item}{i}_fulllist",
                                                                                 'csv',
                                                                                 ['stat_change', statistic_item])
                    st.code(f"Set Fulllist File: {item_data[f'{item}{i}_fulllist']}", language='shell')
                    # item_data[f"{item}{i}_fulllist"] = st.text_input(f'Set Fulllist File: ',
                    #                                                  value=item_data[f"{item}{i}_fulllist"],
                    #                                                  key=f"{statistic_item}_{item}{i}_fulllist",
                    #                                                  on_change=stat_editor_change,
                    #                                                  args=(statistic_item, f"{item}{i}", "fulllist"),
                    #                                                  placeholder=f"Set your Simulation Fulllist file...")
                else:
                    item_data[f"{item}{i}_fulllist"] = ''
                st.divider()
                st.session_state.step6_check.append(self.__step6_makecheck(item_data, f"{item}{i}", statistic_item))
        else:
            if variable != 'Data' and sources is not None:
                for i, source in enumerate(sources):
                    if source in st.session_state.ref_data['general'][f"{variable}_ref_source"]:
                        var_data = st.session_state.ref_data[source]
                    elif source in st.session_state.sim_data['general'][f"{variable}_sim_source"]:
                        var_data = st.session_state.sim_data[source]
                    if i == 0:
                        for i_info in info_list:
                            if f"{item}_Y_{i_info}" not in item_data:
                                if 'varname' not in var_data[variable].keys():
                                    varname = self.nl.read_namelist(var_data['general']['model_namelist'])[variable]['varname']
                                else:
                                    varname = var_data[variable]['varname']
                                if i_info == 'varname' and i_info not in var_data[variable].keys():
                                    item_data[f"{item}_Y_{i_info}"] = \
                                        self.nl.read_namelist(var_data['general']['model_namelist'])[variable][
                                            i_info]
                                elif i_info == 'dir':
                                    item_data[f"{item}_Y_dir"] = os.path.join(st.session_state.main_data['general']['basedir'],
                                                                              st.session_state.main_data['general']['basename'],
                                                                              'output',
                                                                              'data')
                                elif i_info in ['syear', 'eyear']:
                                    item_data[f"{item}_Y_{i_info}"] = st.session_state.main_data['general'][i_info]
                                elif i_info == 'prefix':
                                    if source in st.session_state.ref_data['general'][f"{variable}_ref_source"]:
                                        item_data[f"{item}_Y_{i_info}"] = f'{variable}_ref_{source}_{varname}'
                                    elif source in st.session_state.sim_data['general'][f"{variable}_sim_source"]:
                                        item_data[f"{item}_Y_{i_info}"] = f'{variable}_sim_{source}_{varname}'
                                elif i_info == 'suffix':
                                    item_data[f"{item}_Y_{i_info}"] = ''
                                elif i_info == 'data_groupby':
                                    item_data[f"{item}_Y_{i_info}"] = 'single'
                                elif i_info == 'grid_res':
                                    item_data[f"{item}_Y_grid_res"] = st.session_state.main_data['general']['compare_grid_res']
                                else:
                                    if i_info in var_data['general'].keys():
                                        item_data[f"{item}_Y_{i_info}"] = var_data['general'][i_info]
                                    elif i_info in var_data[variable].keys():
                                        item_data[f"{item}_Y_{i_info}"] = var_data[variable][i_info]

                    else:
                        for i_info in info_list:
                            if f"{item}_X{i}_{i_info}" not in item_data:
                                if 'varname' not in var_data[variable].keys():
                                    varname = self.nl.read_namelist(var_data['general']['model_namelist'])[variable]['varname']
                                else:
                                    varname = var_data[variable]['varname']
                                if i_info == 'varname' and i_info not in var_data[variable].keys():
                                    item_data[f"{item}_X{i}_{i_info}"] = \
                                        self.nl.read_namelist(var_data['general']['model_namelist'])[variable][
                                            i_info]
                                elif i_info == 'dir':
                                    item_data[f"{item}_X{i}_dir"] = os.path.join(st.session_state.main_data['general']['basedir'],
                                                                                 st.session_state.main_data['general']['basename'],
                                                                                 'output',
                                                                                 'data')
                                elif i_info in ['syear', 'eyear']:
                                    item_data[f"{item}_X{i}_{i_info}"] = st.session_state.main_data['general'][i_info]
                                elif i_info == 'prefix':
                                    if source in st.session_state.ref_data['general'][f"{variable}_ref_source"]:
                                        item_data[f"{item}_X{i}_{i_info}"] = f'{variable}_ref_{source}_{varname}'
                                    elif source in st.session_state.sim_data['general'][f"{variable}_sim_source"]:
                                        item_data[f"{item}_X{i}_{i_info}"] = f'{variable}_sim_{source}_{varname}'
                                elif i_info == 'suffix':
                                    item_data[f"{item}_X{i}_{i_info}"] = ''
                                elif i_info == 'data_groupby':
                                    item_data[f"{item}_X{i}_{i_info}"] = 'single'
                                elif i_info == 'grid_res':
                                    item_data[f"{item}_X{i}_grid_res"] = st.session_state.main_data['general']['compare_grid_res']
                                else:
                                    if i_info in var_data['general'].keys():
                                        item_data[f"{item}_X{i}_{i_info}"] = var_data['general'][i_info]
                                    elif i_info in var_data[variable].keys():
                                        item_data[f"{item}_X{i}_{i_info}"] = var_data[variable][i_info]
            else:
                for i_info in info_list:
                    if f"{item}_Y_{i_info}" not in item_data:
                        item_data[f"{item}_Y_{i_info}"] = self.set_default[i_info]
                for i in range(1, n + 1):
                    for i_info in info_list:
                        if f"{item}_X{i}_{i_info}" not in item_data:
                            item_data[f"{item}_X{i}_{i_info}"] = self.set_default[i_info]

            if sources is not None:
                st.write(f'##### :violet[{sources[0]}]')
            else:
                st.write(f'##### :violet[Input Data Y]')
            import itertools
            cols = itertools.cycle(st.columns(3))
            for i_info in sorted(info_list, key=str.lower):
                if i_info not in ["dir", "fulllist"] and f"{item}_Y_{i_info}" in item_data.keys():
                    col = next(cols)
                    if i_info in ['prefix', 'suffix', 'varname']:
                        item_data[f"{item}_Y_{i_info}"] = col.text_input(f'Set {i_info}: ',
                                                                         value=item_data[f"{item}_Y_{i_info}"],
                                                                         key=f"{statistic_item}_{item}_Y_{i_info}",
                                                                         on_change=stat_editor_change,
                                                                         args=(statistic_item, f"{item}_Y", i_info),
                                                                         placeholder=f"Set your {statistic_item.replace('_', ' ')} {i_info.replace('_', ' ')}...")
                    elif i_info in ['timezone', 'grid_res']:
                        item_data[f"{item}_Y_{i_info}"] = col.number_input(f"Set {i_info}: ",
                                                                           value=float(item_data[f"{item}_Y_{i_info}"]),
                                                                           key=f"{statistic_item}_{item}_Y_{i_info}",
                                                                           on_change=stat_editor_change,
                                                                           args=(statistic_item, f"{item}_Y", i_info),
                                                                           placeholder=f"Set your Simulation {item}...")
                    elif i_info in ['syear', 'eyear']:
                        item_data[f"{item}_Y_{i_info}"] = col.number_input(f" Set {i_info}:",
                                                                           format='%04d', step=int(1),
                                                                           value=item_data[f"{item}_Y_{i_info}"],
                                                                           key=f"{statistic_item}_{item}_Y_{i_info}",
                                                                           on_change=stat_editor_change,
                                                                           args=(statistic_item, f"{item}_Y", i_info),
                                                                           placeholder=f"Set your Simulation {item}...")
                    elif i_info == 'tim_res':
                        item_data[f"{item}_Y_{i_info}"] = col.selectbox(f' Set Time Resolution: ',
                                                                        options=('hour', 'day', 'month', 'year'),
                                                                        index=set_data_groupby(
                                                                            item_data[f"{item}_Y_{i_info}"]),
                                                                        key=f"{statistic_item}_{item}_Y_{i_info}",
                                                                        placeholder=f"Set your Simulation Time Resolution (default={item_data[f'{item}_Y_{i_info}']})...")
                    elif i_info == 'data_groupby':
                        item_data[f"{item}_Y_{i_info}"] = col.selectbox(f' Set Data groupby: ',
                                                                        options=('hour', 'day', 'month', 'year', 'single'),
                                                                        index=set_data_groupby(
                                                                            item_data[f"{item}_Y_{i_info}"]),
                                                                        key=f"{statistic_item}_{item}_Y_{i_info}",
                                                                        placeholder=f"Set your Simulation Data groupby (default={item_data[f'{item}_Y_{i_info}']})...")
                    elif i_info == 'data_type':
                        item_data[f"{item}_Y_{i_info}"] = col.selectbox(f' Set Data type: ',
                                                                        options=('stn', 'grid'),
                                                                        index=set_data_type(
                                                                            item_data[f"{item}_Y_{i_info}"]),
                                                                        key=f"{statistic_item}_{item}_Y_{i_info}",
                                                                        placeholder=f"Set your Simulation Data type (default={item_data[f'{item}_Y_{i_info}']})...")
            # item_data[f"{item}_Y_dir"] = st.text_input(f' Set Data Dictionary: ',
            #                                            value=item_data[f"{item}_Y_dir"],
            #                                            key=f"{statistic_item}_{item}_Y_dir",
            #                                            on_change=stat_editor_change,
            #                                            args=(statistic_item, f"{item}_Y", "dir",),
            #                                            placeholder=f"Set your Simulation Dictionary...")
            if not item_data[f"{item}_Y_dir"]: item_data[f"{item}_Y_dir"] = '/'
            try:
                item_data[f"{item}_Y_dir"] = self.path_finder.find_path(item_data[f"{item}_Y_dir"],
                                                                        f"{statistic_item}_{item}_Y_dir",
                                                                        ['stat_change', statistic_item])
                st.code(f"Set Data Dictionary: {item_data[f'{item}_Y_dir']}", language='shell')
            except PermissionError as e:
                if e:
                    item_data[f"{item}_Y_dir"] = '/'

            if item_data[f"{item}_Y_data_type"] == 'stn':
                if not item_data[f"{item}_Y_fulllist"]: item_data[f"{item}_Y_fulllist"] = None
                item_data[f"{item}_Y_fulllist"] = self.path_finder.get_file(item_data[f"{item}_Y_fulllist"],
                                                                            f"{statistic_item}_{item}_Y_fulllist",
                                                                            'csv',
                                                                            ['stat_change', statistic_item])
                st.code(f"Set Fulllist File: {item_data[f'{item}_Y_fulllist']}", language='shell')
                # item_data[f"{item}_Y_fulllist"] = st.text_input(f'Set Fulllist File: ',
                #                                                 value=item_data[f"{item}_Y_fulllist"],
                #                                                 key=f"{statistic_item}_{item}_Y_fulllist",
                #                                                 on_change=stat_editor_change,
                #                                                 args=(statistic_item, f"{item}_Y", "fulllist"),
                #                                                 placeholder=f"Set your Simulation Fulllist file...")
            else:
                item_data[f"{item}_Y_fulllist"] = ''
            st.divider()
            st.session_state.step6_check.append(self.__step6_makecheck(item_data, f"{item}_Y", statistic_item))
            for i in range(1, n + 1):
                if sources is not None:
                    st.write(f'##### :violet[{sources[i]}]')
                else:
                    st.write(f'##### :violet[Input Data X{i}]')
                import itertools
                cols = itertools.cycle(st.columns(3))
                for i_info in sorted(info_list, key=str.lower):
                    if i_info not in ["dir", "fulllist"] and f"{item}_X{i}_{i_info}" in item_data.keys():
                        col = next(cols)
                        if i_info in ['prefix', 'suffix', 'varname']:
                            item_data[f"{item}_X{i}_{i_info}"] = col.text_input(f'Set {i_info}: ',
                                                                                value=item_data[f"{item}_X{i}_{i_info}"],
                                                                                key=f"{statistic_item}_{item}_X{i}_{i_info}",
                                                                                on_change=stat_editor_change,
                                                                                args=(statistic_item, f"{item}_X{i}", i_info),
                                                                                placeholder=f"Set your {statistic_item.replace('_', ' ')} {i_info.replace('_', ' ')}...")
                        elif i_info in ['timezone', 'grid_res']:
                            item_data[f"{item}_X{i}_{i_info}"] = col.number_input(f"Set {i_info}: ",
                                                                                  value=float(item_data[f"{item}_X{i}_{i_info}"]),
                                                                                  key=f"{statistic_item}_{item}_X{i}_{i_info}",
                                                                                  on_change=stat_editor_change,
                                                                                  args=(statistic_item, f"{item}_X{i}", i_info),
                                                                                  placeholder=f"Set your Simulation {item}...")
                        elif i_info in ['syear', 'eyear']:
                            item_data[f"{item}_X{i}_{i_info}"] = col.number_input(f" Set {i_info}:",
                                                                                  format='%04d', step=int(1),
                                                                                  value=item_data[f"{item}_X{i}_{i_info}"],
                                                                                  key=f"{statistic_item}_{item}_X{i}_{i_info}",
                                                                                  on_change=stat_editor_change,
                                                                                  args=(statistic_item, f"{item}_X{i}", i_info),
                                                                                  placeholder=f"Set your Simulation {item}...")
                        elif i_info == 'tim_res':
                            item_data[f"{item}_X{i}_{i_info}"] = col.selectbox(f' Set Time Resolution: ',
                                                                               options=('hour', 'day', 'month', 'year'),
                                                                               index=set_data_groupby(
                                                                                   item_data[f"{item}_X{i}_{i_info}"]),
                                                                               key=f"{statistic_item}_{item}_X{i}_{i_info}",
                                                                               placeholder=f"Set your Simulation Time Resolution (default={item_data[f'{item}_X{i}_{i_info}']})...")
                        elif i_info == 'data_groupby':
                            item_data[f"{item}_X{i}_{i_info}"] = col.selectbox(f' Set Data groupby: ',
                                                                               options=('hour', 'day', 'month', 'year', 'single'),
                                                                               index=set_data_groupby(
                                                                                   item_data[f"{item}_X{i}_{i_info}"]),
                                                                               key=f"{statistic_item}_{item}_X{i}_{i_info}",
                                                                               placeholder=f"Set your Simulation Data groupby (default={item_data[f'{item}_X{i}_{i_info}']})...")
                        elif i_info == 'data_type':
                            item_data[f"{item}_X{i}_{i_info}"] = col.selectbox(f' Set Data type: ',
                                                                               options=('stn', 'grid'),
                                                                               index=set_data_type(
                                                                                   item_data[f"{item}_X{i}_{i_info}"]),
                                                                               key=f"{statistic_item}_{item}_X{i}_{i_info}",
                                                                               placeholder=f"Set your Simulation Data type (default={item_data[f'{item}_X{i}_{i_info}']})...")
                if f"{item}_X{i}_dir" not in item_data: item_data[f"{item}_X{i}_dir"] = '/'
                if not item_data[f"{item}_X{i}_dir"]: item_data[f"{item}_X{i}_dir"] = '/'
                try:
                    item_data[f"{item}_X{i}_dir"] = self.path_finder.find_path(item_data[f"{item}_X{i}_dir"],
                                                                               f"{statistic_item}_{item}_X{i}_dir",
                                                                               ['stat_change', statistic_item])
                    st.code(f"Set Data Dictionary: {item_data[f'{item}_X{i}_dir']}", language='shell')
                except PermissionError as e:
                    if e:
                        item_data[f"{item}_X{i}_dir"] = '/'
                if item_data[f"{item}_X{i}_data_type"] == 'stn':
                    if f"{item}_X{i}_fulllist" not in item_data: item_data[f"{item}_X{i}_fulllist"] = None
                    if not item_data[f"{item}_X{i}_fulllist"]: item_data[f"{item}_X{i}_fulllist"] = None
                    item_data[f"{item}_X{i}_fulllist"] = self.path_finder.get_file(item_data[f"{item}_X{i}_fulllist"],
                                                                                   f"{statistic_item}_{item}_X{i}_fulllist",
                                                                                   'csv',
                                                                                   ['stat_change', statistic_item])
                    st.code(f"Set Fulllist File: {item_data[f'{item}_X{i}_fulllist']}", language='shell')

                    # item_data[f"{item}_X{i}_fulllist"] = st.text_input(f'Set Fulllist File: ',
                    #                                                    value=item_data[f"{item}_X{i}_fulllist"],
                    #                                                    key=f"{statistic_item}_{item}_X{i}_fulllist",
                    #                                                    on_change=stat_editor_change,
                    #                                                    args=(statistic_item, f"{item}_X{i}", "fulllist"),
                    #                                                    placeholder=f"Set your Simulation Fulllist file...")
                else:
                    item_data[f"{item}_X{i}_fulllist"] = ''
                st.divider()
                st.session_state.step6_check.append(self.__step6_makecheck(item_data, f"{item}_X{i}", statistic_item))

    def __step6_makecheck(self, source_lib, item, source):
        error_state = 0

        timezone_key = item + "_timezone"
        data_groupby_key = item + "_data_groupby"
        dir_key = item + "_dir"
        tim_res_key = item + "_tim_res"
        varname_key = item + "_varname"
        # 这之后的要区分检查--------------------------------
        data_type_key = item + "_data_type"
        fulllist_key = item + "_fulllist"
        geo_res_key = item + "_grid_res"
        suffix_key = item + "_suffix"
        prefix_key = item + "_prefix"
        syear_key = item + "_syear"
        eyear_key = item + "_eyear"

        for key in [timezone_key, data_groupby_key, dir_key, tim_res_key, varname_key]:
            if isinstance(source_lib[key], str):
                if len(source_lib[key]) < 1:
                    st.error(f'{key} should be a string longer than one, please check {key}.',
                             icon="⚠")
                    error_state += 1
            elif isinstance(source_lib[key], float) | isinstance(source_lib[key], int):
                if source_lib[key] < -12. or source_lib[key] > 12.:
                    st.error(f'"please check {key}.', icon="⚠")
                    error_state += 1

        if source_lib[data_type_key] == "Grid":
            if (source_lib[geo_res_key] == 0.0):
                st.error(f"Geo Resolution should be larger than zero when data_type is 'geo', please check.", icon="⚠")
                error_state += 1
            elif (isinstance(source_lib[suffix_key], str)) | (isinstance(source_lib[prefix_key], str)):
                if (len(source_lib[suffix_key]) == 0) & (len(source_lib[prefix_key]) == 0):
                    st.error(f'"suffix or prefix should be a string longer than one, please check.', icon="⚠")
                    error_state += 1
            elif source_lib[eyear_key] < source_lib[syear_key]:
                st.error(f" year should be larger than Start year, please check.", icon="⚠")
                error_state += 1
            for key in [syear_key, eyear_key]:
                if isinstance(source_lib[key], int):
                    st.error(f'"please check {key}.', icon="⚠")
                    error_state += 1

        elif source_lib[data_type_key] == "stn":
            if not source_lib[fulllist_key]:
                st.error(f"Fulllist should not be empty when data_type is 'stn'.", icon="⚠")
                error_state += 1

        if item not in st.session_state.stat_errorlist[source]:
            st.session_state.stat_errorlist[source][item] = []
        if error_state > 0:
            st.session_state.stat_errorlist[source][item].append(2)
            st.session_state.stat_errorlist[source][item] = list(np.unique(st.session_state.stat_errorlist[source][item]))
            return False
        if error_state == 0:
            if (item in st.session_state.stat_errorlist[source]) & (2 in st.session_state.stat_errorlist[source][item]):
                st.session_state.stat_errorlist[source][item] = list(
                    filter(lambda x: x != 2, st.session_state.stat_errorlist[source][item]))
                st.session_state.stat_errorlist[source][item] = list(np.unique(st.session_state.stat_errorlist[source][item]))
            return True

    def __step6_make_stat_namelist(self, file_path, statistic_items, stat_data):
        general = stat_data['general']
        with st.spinner('Making namelist... Please wait.'):
            with open(file_path, 'w') as f:
                lines = []
                end_line = "/\n\n\n"

                lines.append("&general\n")
                max_key_length = max(len(f"{key}") for key in general.keys())
                for statistic_item in statistic_items:
                    key = f"{statistic_item}_data_source"
                    lines.append(f"    {key:<{max_key_length}} = {', '.join(general[key])}\n")
                lines.append(end_line)

                for statistic_item in statistic_items:
                    info_list = self.stat_info[statistic_item]
                    add_list = info_list['general']

                    lines.append(f"&{statistic_item}\n")
                    if self.stat_class[statistic_item] == 'Single':
                        for casename in general[f'{statistic_item}_data_source']:
                            lines.append(f"\n#casename: {casename}\n")
                            for key in add_list:
                                # if (key in ['geo_res', 'suffix', 'prefix', 'syear', 'eyear']) & (
                                #         st.session_state.stat_data[statistic_item][f"{casename}_data_type"] == 'stn'):
                                #     lines.append(f"    {casename}{key} =  \n")
                                # elif (key in ['fulllist']) & (
                                #         st.session_state.sim_data[statistic_item][f"{casename}_data_type"] == 'geo'):
                                #     lines.append(f"    {casename}{key} =  \n")
                                # else:
                                lines.append(f"    {casename}_{key} =  {stat_data[statistic_item][f'{casename}_{key}']}\n")
                    elif self.stat_class[statistic_item] == 'Multi':
                        for casename in general[f'{statistic_item}_data_source']:
                            lines.append(f"\n#casename: {casename}\n")
                            if statistic_item == 'Partial_Least_Squares_Regression':
                                n = stat_data[statistic_item][f'{casename}_nX']
                                lines.append(
                                    f"    {casename}_nX =  {stat_data[statistic_item][f'{casename}_nX']}\n")
                                for key in add_list:
                                    lines.append(
                                        f"    {casename}_Y_{key} =  {stat_data[statistic_item][f'{casename}_Y_{key}']}\n")
                                lines.append("\n")

                                for i in range(1, n + 1):
                                    for key in add_list:
                                        lines.append(
                                            f"    {casename}_X{i}_{key} =  {stat_data[statistic_item][f'{casename}_X{i}_{key}']}\n")
                                    lines.append("\n")
                            else:
                                if statistic_item == 'Three_Cornered_Hat':
                                    n = stat_data[statistic_item][f'{casename}_nX']
                                    lines.append(
                                        f"    {casename}_nX =  {stat_data[statistic_item][f'{casename}_nX']}\n")
                                else:
                                    n = 2
                                for i in range(1, n + 1):
                                    for key in add_list:
                                        lines.append(
                                            f"    {casename}{i}_{key} =  {stat_data[statistic_item][f'{casename}{i}_{key}']}\n")
                                    lines.append("\n")
                    if 'other' in info_list.keys():
                        for key in info_list['other']:
                            lines.append(f"    {key} =  {stat_data[statistic_item][f'{key}']}\n")
                    lines.append(end_line)

                for line in lines:
                    f.write(line)
                time.sleep(0.8)
                return True

    # Not Strat yet!-------------------------------
    def statistic_run(self):

        self.__print_welcome_message()

        st.divider()
        col1, col4 = st.columns((2, 1.7))
        col1.write(':point_down: Press button to running :orange[Openbench]')
        col4.write(' Click to passing through the running steps :point_down:')  # 点击按钮调过运行步骤

        if 'stat_status' not in st.session_state:
            st.session_state.stat_status = ''
        if 'step6_run' not in st.session_state:
            st.session_state.step6_run = False
        if "status_message" not in st.session_state:
            st.session_state["status_message"] = "***Running Pages...***"

        col1, col2, col4 = st.columns(3)
        st.divider()

        if col1.button('Run', use_container_width=True):
            status = st.status(label="***Running Evaluation...***", expanded=False)
            st.session_state.stat_status = self.Openbench_processing(status)
            st.info('More info please check task_log_stat.txt')
        elif col4.button('Pass', use_container_width=True):
            st.session_state.stat_status = 'complete'

        if st.session_state.stat_status == 'complete':
            st.session_state.step6_run = True
            st.success("Done!")
        elif st.session_state.status == 'error':
            st.error("There is error in your setting, please check!")
            st.session_state.step6_run = False

        if st.session_state.status == 'Running':
            next_button_disable1 = True
            next_button_disable2 = True
        else:
            next_button_disable1 = False
            if not st.session_state.step6_run:
                next_button_disable2 = True
            else:
                next_button_disable2 = False

        def define_step1():
            st.session_state.step6_stat_make = False

        def define_step2():
            st.session_state.step6_stat_run = True

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.button(':back: Previous step', on_click=define_step1, help='Go to Statistic make page',
                      disabled=next_button_disable1)
        with col4:
            st.button('Next step :soon: ', on_click=define_step2, help='Go to Statistic Show figures page',
                      disabled=next_button_disable2)

    def Openbench_processing(self, status):
        st.divider()
        p = subprocess.Popen(
            f'python -u {st.session_state.openbench_path}/script/openbench.py {st.session_state["main_nml"]}',
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,

        )

        log_file = open("task_log_stat.txt", "w", encoding='utf-8')
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
            time.sleep(1)
            st.session_state['status_message'] = f"***Evaluation done***"
            status.update(label=f"***Evaluation done***", state="complete", expanded=False)
        elif status._current_state == "error":
            st.session_state['status_message'] = f"***:red[Evaluation Error]***"
            time.sleep(0.5)
            status.update(label=f"***:red[Evaluation Error]***", state="error", expanded=False)

        return status._current_state

    def __process_line(self, line, status):
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

        if warning_pattern.search(line.strip()) and error_pattern.search(line) and not wskip_next_line:
            status.write(f"***:orange[{line.strip()}]***")
            wskip_next_line = True
        elif wskip_next_line:
            status.write(f"***:orange[{line.strip()}]***")
            wskip_next_line = False
        elif error_pattern.search(line):
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

    def __print_welcome_message(self):
        """Print a more beautiful welcome message and ASCII art."""
        st.subheader('Welcome to Statistic Running Page!', divider=True)
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

    def statistic_show(self):
        # st.write('show')
        # if not st.session_state.step6_stat_run:
        #     visualization_statistic.set_errors()
        self.visualizations()

        def define_step1():
            st.session_state.step6_stat_run = False

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.button(':back: Previous step', on_click=define_step1, help='Go to Statistic running page')

    def statistic_replot(self):
        self.replot_statistic()

        def define_step1():
            st.session_state.step6_stat_run = False

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.button(':back: Previous step', on_click=define_step1, help='Go to Statistic running page')
