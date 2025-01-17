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

    def has_permission(self, path):
        return os.access(path, os.R_OK)  # 检查是否有读取权限

    def __update_path(self, path: str, key: str, change_list):
        def path_change(key, ipath, change_list):
            selected_dir = st.session_state[key]
            if selected_dir == '<- Back SPACE':  # 返回上一级
                st.session_state['find_path'][key] = os.path.dirname(ipath)
            else:  # 进入选择的子目录
                if self.has_permission(os.path.join(ipath, selected_dir)):
                    st.session_state['find_path'][key] = os.path.join(ipath, selected_dir)
            if change_list[0] is not None and change_list[1] is not None:
                st.session_state[change_list[0]][change_list[1]] = True

        subdirectories = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
        with st.popover(f"Find Path", use_container_width=True):
            st.code(f"Current Path: {st.session_state['find_path'][key]}")
            options = (['<- Back SPACE'] if path != '/' else []) + sorted(subdirectories)
            st.radio(
                label="Select Directory:",
                options=options,
                index=None,
                key=key,
                on_change=path_change,
                args=(key, path, change_list),
                label_visibility="collapsed"
            )

    # @staticmethod
    def find_path(self, path: str, key: str, change_list: list):
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

        self.__update_path(st.session_state['find_path'][key], key, change_list)
        return st.session_state['find_path'][key]

    def __update_sub_path(self, sub_path: str, key: str, root_path: str, change_list: list):
        def path_change(key, ipath, change_list):
            selected_dir = st.session_state[key]
            if selected_dir == '<- Back SPACE':  # 返回上一级
                st.session_state['find_path'][key] = os.path.dirname(ipath)
            else:
                if self.has_permission(os.path.join(ipath, selected_dir)):
                    st.session_state['find_path'][key] = os.path.join(ipath, selected_dir)
            if change_list[0] is not None and change_list[1] is not None:
                st.session_state[change_list[0]][change_list[1]] = True

        if not os.path.exists(os.path.join(sub_path)):
            subdirectories = [name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]
        else:
            subdirectories = [name for name in os.listdir(os.path.join(sub_path)) if
                              os.path.isdir(os.path.join(sub_path, name))]

        with st.popover(f"Find Path", use_container_width=True):
            st.code(f"Current Sub Path: {os.path.relpath(st.session_state['find_path'][key], root_path)}")
            options = (
                          ['<- Back SPACE'] if not os.path.samefile(st.session_state['find_path'][key],
                                                                    root_path) else []) + sorted(subdirectories)
            st.radio(
                label="Select Directory:",
                options=options,
                index=None,
                key=key,
                on_change=path_change,
                args=(key, st.session_state['find_path'][key], change_list),
                label_visibility="collapsed"
            )

    def find_subdirectories(self, sub_path: str, root_key: str, sub_key, change_list: list):
        root_path = st.session_state['find_path'][root_key]
        if sub_key not in st.session_state['find_path']:
            st.session_state['find_path'][sub_key] = os.path.join(root_path, sub_path)
        if platform.system() == "Windows":
            sep = '\\'
        else:
            sep = posixpath.sep

        if sub_path == sep:
            sub_path = ''
            st.session_state['find_path'][sub_key] = root_path
        else:
            if not os.path.exists(st.session_state['find_path'][sub_key]):
                st.warning(f'Input path:{sub_path} is not a directory. Refind path from "{root_path}".')
                st.session_state['find_path'][sub_key] = root_path
            elif not st.session_state['find_path'][sub_key].startswith(root_path):
                st.session_state['find_path'][sub_key] = root_path

        self.__update_sub_path(st.session_state['find_path'][sub_key], sub_key, root_path, change_list)
        if not os.path.samefile(st.session_state['find_path'][sub_key], root_path):
            sub_dir = os.path.relpath(st.session_state['find_path'][sub_key], root_path)
            if sub_dir.endswith(sep):
                return sub_dir
            else:
                return sub_dir + sep
        else:
            st.warning('Please select sub-Dictionary!')
            return None
            # return os.path.relpath(st.session_state['find_path'][sub_key], root_path)

    def __update_file(self, path: str, key: str, file_type: str, change_list):
        def path_change(key, ipath, change_list):
            selected_dir = st.session_state[key]
            if selected_dir == '<- Back SPACE':  # 返回上一级
                st.session_state['find_path'][key] = os.path.dirname(ipath)
            else:  # 进入选择的子目录
                if self.has_permission(os.path.join(ipath, selected_dir)):
                    st.session_state['find_path'][key] = os.path.join(ipath, selected_dir)
            if change_list[0] is not None and change_list[1] is not None:
                st.session_state[change_list[0]][change_list[1]] = True

        subdirectories = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
        with st.popover(f"Find Path", use_container_width=True):
            st.code(f"Current Path: {st.session_state['find_path'][key]}")
            col1, col2 = st.columns(2)
            options = (['<- Back SPACE'] if path != '/' else []) + sorted(subdirectories)
            col1.radio(
                label="Select Directory:",
                options=options,
                index=None,
                key=key,
                on_change=path_change,
                args=(key, path, change_list),
                label_visibility="collapsed"
            )
            subdirfiles = [file for file in os.listdir(path) if file.endswith(file_type)]

            def get_index(subdirfiles, f_value):
                if f_value in subdirfiles:
                    index = subdirfiles.index(f_value)
                    return index
                else:
                    return None

            if st.session_state['find_file'][key] is not None:
                index = get_index(subdirfiles, st.session_state['find_file'][key])
            else:
                index = None

            if subdirfiles:
                ifile = col2.radio('file', sorted(subdirfiles), index=index, key=f'{key}_ifile', help=None,
                                   disabled=False, horizontal=False, captions=None, label_visibility="collapsed")
                if ifile is not None:
                    st.session_state['find_file'][key] = ifile
                else:
                    if st.session_state['find_file'][key] is not None:
                        file = normpath(os.path.join(st.session_state['find_path'][key], st.session_state['find_file'][key]))
                        if not os.path.isfile(file):
                            st.session_state['find_file'][key] = None

    # @staticmethod
    def get_file(self, path: str, key: str, file_type: str, change_list: list):
        if 'find_path' not in st.session_state:
            st.session_state['find_path'] = {}
        if 'find_file' not in st.session_state:
            st.session_state['find_file'] = {}

        if key not in st.session_state['find_path']:
            if path is not None:
                st.session_state['find_path'][key] = os.path.dirname(path)
            else:
                st.session_state['find_path'][key] = path
        if key not in st.session_state['find_file']:
            if path is not None:
                st.session_state['find_file'][key] = os.path.basename(path)
            else:
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
        self.__update_file(st.session_state['find_path'][key], key, file_type, change_list)
        if st.session_state['find_file'][key] is not None:
            return normpath(os.path.join(st.session_state['find_path'][key], st.session_state['find_file'][key]))
