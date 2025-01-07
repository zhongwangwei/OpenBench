import os.path
import platform
import posixpath
import itertools
from posixpath import normpath
from pathlib import Path
import streamlit as st


class FindPath:
    def __init__(self):
        self.author = "Qingchen Xu/xuqingchen0@gmail.com"

    def __update_path(self, path: str, key: str):
        def path_change(key, ipath):
            selected_dir = st.session_state[key]
            if selected_dir == '<- Back SPACE':  # 返回上一级
                st.session_state['find_path'][key] = os.path.dirname(ipath)
            else:  # 进入选择的子目录
                st.session_state['find_path'][key] = os.path.join(ipath, selected_dir)

        subdirectories = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
        with st.popover(f"Find Path", use_container_width=True):
            st.code(f"Current Path: {st.session_state['find_path'][key]}")
            options = sorted(subdirectories) + (['<- Back SPACE'] if path != '/' else [])
            st.radio(
                label="Select Directory:",
                options=options,
                index=None,
                key=key,
                on_change=path_change,
                args=(key, path),
                label_visibility="collapsed"
            )

    # @staticmethod
    def find_path(self, path: str, key: str):
        if 'find_path' not in st.session_state:
            st.session_state['find_path'] = {}
        if 'find_option' not in st.session_state:
            st.session_state['find_option'] = None
        if key not in st.session_state['find_path']:
            st.session_state['find_path'][key] = path
        if path is None:
            if platform.system() == 'Windows':
                st.warning(f'Input path:{path} is not a directory. Reset path as "C:"'.format(path=path))
                st.session_state['find_path'][key] = 'C:'
            else:
                st.warning(f'Input path:{path} is not a directory. Reset path as "/"'.format(path=path))
                st.session_state['find_path'][key] = '/'
        else:
            if not os.path.exists(os.path.abspath(path)):
                if platform.system() == 'Windows':
                    st.warning(f'Input path: :red[{path}] is not exists. Find path from "C:"'.format(path=path))
                    st.session_state['find_path'][key] = 'C:'
                else:
                    st.warning(f'Input path: :red[{path}] is not exists. Find path from "/"'.format(path=path))
                    st.session_state['find_path'][key] = '/'
            elif not os.path.isdir(os.path.abspath(path)):
                if platform.system() == 'Windows':
                    st.warning(f'Input path: :red[{path}] is not a directory. Reset path as "C:"'.format(path=path))
                    st.session_state['find_path'][key] = 'C:'
                else:
                    st.warning(f'Input path: :red[{path}]is not a directory. Reset path as "/"'.format(path=path))
                    st.session_state['find_path'][key] = '/'

        self.__update_path(st.session_state['find_path'][key], key)
        return st.session_state['find_path'][key]

    def __update_sub_path(self, path: str, key: str, root_path: str):
        def path_change(key, ipath):
            selected_dir = st.session_state[key]
            if selected_dir == '<- Back SPACE':  # 返回上一级
                st.session_state['find_path'][key] = os.path.dirname(ipath)
            else:  # 进入选择的子目录
                st.session_state['find_path'][key] = os.path.join(ipath, selected_dir)

        if not os.path.exists(os.path.join(root_path, path)):
            subdirectories = [name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]
        else:
            subdirectories = [name for name in os.listdir(os.path.join(root_path, path)) if
                              os.path.isdir(os.path.join(root_path, path, name))]

        with st.popover(f"Find Path", use_container_width=True):
            st.code(f"Current Sub Path: {st.session_state['find_path'][key].replace(root_path, '')}")
            options = sorted(subdirectories) + (
                ['<- Back SPACE'] if st.session_state['find_path'][key] != root_path else [])
            st.radio(
                label="Select Directory:",
                options=options,
                index=None,
                key=key,
                on_change=path_change,
                args=(key, path),
                label_visibility="collapsed"
            )

    def find_subdirectories(self, path: str, root_key: str, sub_key):
        root_path = st.session_state['find_path'][root_key]
        if sub_key not in st.session_state['find_path']:
            if not os.path.exists(os.path.join(root_path, path)):
                st.warning(f'Input path:{path} is not a directory. Refind path from "{root_path}".')
                st.session_state['find_path'][sub_key] = root_path  # os.path.join(root_path, path)
            else:
                st.session_state['find_path'][sub_key] = os.path.join(root_path, path)

        self.__update_sub_path(path, sub_key, root_path)
        st.write(st.session_state['find_path'][sub_key].replace(root_path, ''))

    # def __update_file(self, path: str, key: str, end_type: str):
    #     def ifile_change(key, ipath):
    #         selected_dir = st.session_state[key]
    #         if selected_dir == '<- Back SPACE':  # 返回上一级
    #             st.session_state['find_file_ipath'][key] = os.path.dirname(ipath)
    #         else:
    #             st.session_state['find_file_ipath'][key] = os.path.join(ipath, selected_dir)
    #
    #     subdirectories = [
    #         name for name in os.listdir(path)
    #         if os.path.isdir(os.path.join(path, name))
    #     ]
    #     with st.popover(f"Find File", use_container_width=True):
    #         st.code(f"Current Path: {st.session_state['find_file_ipath'][key]}")
    #         col1, col2 = st.columns(2)
    #         options = sorted(subdirectories) + (['<- Back SPACE'] if path != '/' else [])
    #         col1.radio(
    #             label="Select Directory:",
    #             options=options,
    #             index=None,
    #             key=key,
    #             on_change=ifile_change,
    #             args=(key, path),
    #             label_visibility="collapsed"
    #         )
    #         selected_dir = st.session_state[key]
    #         if selected_dir == '<- Back SPACE':  # 返回上一级
    #             st.session_state['find_file_ipath'][key] = os.path.dirname(path)
    #         else:
    #             st.session_state['find_file_ipath'][key] = os.path.join(path, selected_dir)
    #
    #         st.write(st.session_state[key], st.session_state['find_file_ipath'][key])
    #         #
    #         # subdirfiles = [file for file in os.listdir(path) if file.endswith(end_type)]
    #         #
    #         # if subdirfiles:
    #         #     ifile = col2.radio('file', sorted(subdirfiles), index=None, key=f'{key}_ifile', help=None,
    #         #                        disabled=False, horizontal=False, captions=None, label_visibility="collapsed")
    #         #     if ifile:
    #         #         st.session_state['find_file'][key]['file'] = ifile
    #
    # def get_file(self, path: str, key: str, file_type: str):
    #     if 'find_file_ipath' not in st.session_state:
    #         st.session_state['find_file_ipath'] = {}
    #         st.session_state['find_file_ifile'] = {}
    #     if key not in st.session_state['find_file_ipath']:
    #         st.session_state['find_file_ipath'][key] = path
    #         st.session_state['find_file_ifile'][key] = None
    #
    #     if path is None:
    #         if platform.system() == 'Windows':
    #             st.warning(f'Not setting input path. Reset path as "C:"'.format(path=path))
    #             st.session_state['find_file_ipath'][key] = 'C:'
    #         else:
    #             st.warning(f'Not setting input path. Reset path as "/"'.format(path=path))
    #             st.session_state['find_file_ipath'][key] = '/'
    #     else:
    #         if not os.path.exists(os.path.abspath(path)):
    #             if platform.system() == 'Windows':
    #                 st.warning(f'Input path: :red[{path}] is not exists. Find path from "C:"'.format(path=path))
    #                 st.session_state['find_file_ipath'][key] = 'C:'
    #             else:
    #                 st.warning(f'Input path: :red[{path}] is not exists. Find path from "/"'.format(path=path))
    #                 st.session_state['find_file_ipath'][key] = '/'
    #         elif not os.path.isdir(os.path.abspath(path)):
    #             if platform.system() == 'Windows':
    #                 st.warning(f'Input path: :red[{path}] is not a directory. Reset path as "C:"'.format(path=path))
    #                 st.session_state['find_file_ipath'][key] = 'C:'
    #             else:
    #                 st.warning(f'Input path: :red[{path}]is not a directory. Reset path as "/"'.format(path=path))
    #                 st.session_state['find_file_ipath'][key] = '/'
    #     st.write(st.session_state['find_file_ipath'][key])
    #     self.__update_file(st.session_state['find_file_ipath'][key], key, file_type)
    #     # st.write(normpath(os.path.join(st.session_state['find_file_ipath'][key], st.session_state['find_file_ifile'][key])))

    def __update_file(self, path: str, key: str, file_type: str):
        def path_change(key, ipath):
            selected_dir = st.session_state[key]
            if selected_dir == '<- Back SPACE':  # 返回上一级
                st.session_state['find_path'][key] = os.path.dirname(ipath)
            else:  # 进入选择的子目录
                st.session_state['find_path'][key] = os.path.join(ipath, selected_dir)

        subdirectories = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
        with st.popover(f"Find Path", use_container_width=True):
            st.code(f"Current Path: {st.session_state['find_path'][key]}")
            col1, col2 = st.columns(2)
            options = sorted(subdirectories) + (['<- Back SPACE'] if path != '/' else [])
            col1.radio(
                label="Select Directory:",
                options=options,
                index=None,
                key=key,
                on_change=path_change,
                args=(key, path),
                label_visibility="collapsed"
            )
            subdirfiles = [file for file in os.listdir(path) if file.endswith(file_type)]

            if subdirfiles:
                ifile = col2.radio('file', sorted(subdirfiles), index=None, key=f'{key}_ifile', help=None,
                                   disabled=False, horizontal=False, captions=None, label_visibility="collapsed")
                if ifile:
                    st.session_state['find_file'][key] = ifile
                else:
                    st.session_state['find_file'][key] = None

    # @staticmethod
    def get_file(self, path: str, key: str, file_type: str):
        if 'find_path' not in st.session_state:
            st.session_state['find_path'] = {}
        if 'find_file' not in st.session_state:
            st.session_state['find_file'] = {}

        # if ipath.endswith(file_type):
        #     file_path = Path(ipath)
        #     path = file_path.parent
        # else:
        #     path = ipath
        if key not in st.session_state['find_path']:
            st.session_state['find_path'][key] = path
        if key not in st.session_state['find_file']:
            st.session_state['find_file'][key] = None

        if st.session_state['find_path'][key] is None:
            if platform.system() == 'Windows':
                st.warning(f'Not setting input path. Reset path as "C:"')
                st.session_state['find_path'][key] = 'C:'
            else:
                st.warning(f'Not setting input path. Reset path as "/"')
                st.session_state['find_path'][key] = '/'
        else:
            if not os.path.exists(os.path.abspath(st.session_state['find_path'][key])):
                if platform.system() == 'Windows':
                    st.warning(f'Input path: :red[{st.session_state["find_path"][key]}] is not exists. Find path from "C:"')
                    st.session_state['find_path'][key] = 'C:'
                else:
                    st.warning(f'Input path: :red[{st.session_state["find_path"][key]}] is not exists. Find path from "/"')
                    st.session_state['find_path'][key] = '/'
            elif not os.path.isdir(os.path.abspath(st.session_state["find_path"][key])):
                if platform.system() == 'Windows':
                    st.warning(f'Input path: :red[{st.session_state["find_path"][key]}] is not a directory. Reset path as "C:"')
                    st.session_state['find_path'][key] = 'C:'
                else:
                    st.warning(f'Input path: :red[{st.session_state["find_path"][key]}]is not a directory. Reset path as "/"')
                    st.session_state['find_path'][key] = '/'

        self.__update_file(st.session_state['find_path'][key], key, file_type)
        if st.session_state['find_file'][key] is not None:
            return normpath(os.path.join(st.session_state['find_path'][key], st.session_state['find_file'][key]))
