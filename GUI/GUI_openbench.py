# -*- coding: utf-8 -*-
import os
import streamlit as st
from streamlit_option_menu import option_menu
from Namelist_lib.namelist_read import NamelistReader
from Namelist_lib.namelist_info import initial_setting
from Page_control import Pages_control
from Page_make_validation import make_initional, make_reference, make_simulation
from Page_run import run_validation
from Page_visualization import visualization_validation, visualization_replot_files, visualization_replot_Comparison

st.set_page_config(page_title="Home Pages", page_icon="🌍", layout="centered")


def info_2025_01_17():
    st.write('##### :blue[Update in 2025-01-17]')

    with st.expander("Show update (2025-01-17)"):
        st.code('''
                The code has been updated, including solving the instability of upload paths and files,
                adding a debug button to the Evaluation page.
                Updated code fixing some functions of the visualization page,
                and adding some new function options. 
                This update fixes some Statistical issues, including information misalignment after selecting a data source, 
                and issues with the Plotting module.
        ''', language='shell', line_numbers=True)


def info_2023_11_14():
    st.write('##### :orange[Introduction for connect host]')

    with st.expander("Show method"):
        st.write("""###### In loaclhost""")
        st.code("""
        $ ssh-keygen -t rsa
        $ ssh-copy-id -i ~/.ssh/id_rsa.pub username@172.16.102.36
        $ cd ~/.ssh
        $ touch config
        """, language='shell', line_numbers=True)
        st.write(""" ###### Place the following text in the config file:""")
        st.code('''
        Host land
            Hostname 172.16.102.86
            User Username
            IdentityFile C:/Users/Administrator/.ssh/id_rsa
            ForwardX11 yes
        Host tms15
            Hostname 192.168.6.115
            User Username
            IdentityFile ~/.ssh/id_rsa
            ForwardX11 yes
            ProxyJump land
        ''', language='shell', line_numbers=True)
        st.write("""###### If you using the :red[macos system], """)
        st.code("""Replace "C:/Users/Administrator/.ssh/id_rsa" with "~/.ssh/id_rsa" """, language='shell', line_numbers=True)
        st.write('###### Connect to server :red[tms15]')
        st.code("""
        $ ssh -L 8000:127.0.0.1:8501 tms15
        """, language='shell', line_numbers=True)
        st.write('###### Open the terminal or the command prompt.')
        st.code("""
        $ conda create -n openbench python=3.12
        $ conda activate openbench
        """, language='shell', line_numbers=True)
        st.write('###### Installation packages depends on')
        st.code("""
        $ conda install -c conda-forge numpy pandas xarray matplotlib cartopy scipy dask joblib netCDF4 flox
        $ pip install streamlit
        $ pip install streamlit_option_menu
        $ pip install stqdm 
        """, language='shell', line_numbers=True)
        st.write('######  Use the xesmf and Cdo packages under Linux or Mac OS')
        st.code("""
        $ conda install -c fonda-forge xesmf cdo
        """, language='shell', line_numbers=True)
        st.write('###### Evaluation installation')
        st.code("""
        $ python -c "import numpy, pandas, xarray, matplotlib, cartopy,
         scipy ,dask ,joblib ,streamlit;
         print('All packages imported successfully!')"
        """, language='shell', line_numbers=True)

        st.write("###### If you do not install, then copy the '.bashrc_openbench' file and source it.")
        st.code("""
        $ cp /home/xuqch3/.bashrc ~/.bashrc_openbench
        $ source ~/.bashrc_openbench
        """, language='shell', line_numbers=True)

        st.code("""
        $ cd your_code_file
        $ source ./GUI/run.sh or streamlit run ./GUI/GUI_openbench.py
        """, language='shell', line_numbers=True)
        st.code("""
        If your connected by 'ssh -L 8000:127.0.0.1:8501 tms15', then copy http://127.0.0.1:8000/
        Otherwise please copy the 'Network URL' to your browser.
        """, language='shell', line_numbers=True)
        # st.code(code, language='shell')


def info_2023_11_30():
    st.write('##### :blue[Update in 2023-11-30]')

    with st.expander("Show update (2023-11-30)"):
        st.code('''
                The code has been updated, including updates to different Liszt crafting,
                    and added controls for the components used.
                The check namelist section has been updated to be more detailed than 
                    before, and it cannot be output in case of errors.
                The Run Verification Code section has been updatedto output the 
                    results to the page for inspection by consumers.
                The validation code has been updated to add the ability to compare 
                    multiple modes at the same time.
        ''', language='shell', line_numbers=True)
        # st.write("""If you are using the macos system, """)


def openbench_flowchart():
    st.write('##### :green[Flowchart of Openbench]')
    with st.expander("Show update (2024-11-07)"):
        st.image('./GUI/Namelist_lib/flowchart.jpg')


def print_welcome_message():
    """Print a more beautiful welcome message and ASCII art."""

    st.text(f'''\n
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
                This system evaluate various land surface model outputs against reference data.\n
                Key Features:
                  • Multi-model support
                  • Comprehensive variable evaluation
                  • Advanced metrics and scoring
                  • Customizable benchmarking
                \n
                Graphical User Interface:
                  • Efficient access, more intuitive experience
                  • Improved visualization effect.
                \n
                ''')

    # st.code(f'''
    #         \n\n
    #         {"=" * 80}
    #            ____                   ____                  _
    #           / __ \\                 |  _ \\                | |
    #          | |  | |_ __   ___ _ __ | |_) | ___ _ __   ___| |__
    #          | |  | | '_ \\ / _ \\ '_ \\|  _ < / _ \\ '_ \\ / __| '_ \\
    #          | |__| | |_) |  __/ | | | |_) |  __/ | | | (__| | | |
    #           \\____/| .__/ \\___|_| |_|____/ \\___|_| |_|\\___|_| |_|
    #                 | |
    #                 |_|
    #         {"=" * 80}
    #         Welcome to OpenBench: The Open Land Surface Model Benchmark Evaluation System!
    #         {"=" * 80}
    #         This system evaluate various land surface model outputs against reference data.
    #         Key Features:
    #           • Multi-model support
    #           • Comprehensive variable evaluation
    #           • Advanced metrics and scoring
    #           • Customizable benchmarking
    #         {"=" * 80}
    #         \n
    #         ''',
    #         language='python',
    #         # line_numbers=True,
    #         )


def show_info():
    st.subheader('Welcome to The Open Source Land Surface Model Benchmarking System Graphical User Interface!', divider=True)

    # st.divider()
    info_2025_01_17()
    openbench_flowchart()
    info_2023_11_14()
    info_2023_11_30()
    # info_2023_11_01()


def initial_st(initial_information):
    # if 'pickle_file' not in st.session_state:
    #     st.session_state.pickle_file = 'cachefile'
    if 'clear_state_onclick' not in st.session_state:
        st.session_state.clear_state_onclick = 0
    if 'switch_button1_onclick' not in st.session_state:
        st.session_state.switch_button1_onclick = 0
    if 'switch_button2_onclick' not in st.session_state:
        st.session_state.switch_button2_onclick = 0
    if 'switch_button3_onclick' not in st.session_state:
        st.session_state.switch_button3_onclick = 0
    if 'switch_button4_onclick' not in st.session_state:
        st.session_state.switch_button4_onclick = 0
    if 'switch_button5_onclick' not in st.session_state:
        st.session_state.switch_button5_onclick = 0
    if 'switch_button6_onclick' not in st.session_state:
        st.session_state.switch_button6_onclick = 0
    if 'clear_state' not in st.session_state:
        st.session_state['clear_state'] = False
    if 'find_path' not in st.session_state:
        st.session_state['find_path'] = {}

    if 'selected' not in st.session_state:
        st.session_state.selected = 'Home'
    if 'menu_option' not in st.session_state:
        st.session_state.menu_option = 0
    if 'menu_4' not in st.session_state:
        st.session_state.menu_4 = 'Home'

    if 'main_path' not in st.session_state:
        st.session_state.main_path = os.getcwd()
    if 'openbench_path' not in st.session_state:
        st.session_state.openbench_path = os.path.abspath(os.path.join(os.getcwd()))
    if 'main_data' not in st.session_state:
        st.session_state.main_data = initial_information.main()
    if 'sim_data' not in st.session_state:
        st.session_state.sim_data = initial_information.sim()
    if 'ref_data' not in st.session_state:
        st.session_state.ref_data = initial_information.ref()
    if 'stat_data' not in st.session_state:
        st.session_state.stat_data = initial_information.stat()
    if 'generals' not in st.session_state:
        st.session_state.generals = {}
    if 'evaluation_items' not in st.session_state:
        st.session_state.evaluation_items = initial_information.evaluation_items()
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {}
    if 'scores' not in st.session_state:
        st.session_state.scores = {}
    if 'comparisons' not in st.session_state:
        st.session_state.comparisons = {}
    if 'statistics' not in st.session_state:
        st.session_state.statistics = {}

    if 'ref_data_general' not in st.session_state:
        st.session_state.ref_data_general = {}
    # TODO: change name

    if 'step1_initial' not in st.session_state:
        st.session_state.step1_initial = None
    if 'step1_general' not in st.session_state:
        st.session_state.step1_general = False
    if 'step1_main_check_general' not in st.session_state:
        st.session_state.step1_main_check_general = False
    if 'step1_main_check' not in st.session_state:
        st.session_state.step1_main_check = False
    if 'step1_main_nml' not in st.session_state:
        st.session_state.step1_main_nml = False
    if 'step1_metrics' not in st.session_state:
        st.session_state.step1_metrics = False
    if 'step1_main_check_metrics_scores' not in st.session_state:
        st.session_state.step1_main_check_metrics_scores = False
    if 'step1_evaluation' not in st.session_state:
        st.session_state.step1_evaluation = False
    if 'step1_main_check_evaluation' not in st.session_state:
        st.session_state.step1_main_check_evaluation = False
    if 'step1_nml_count' not in st.session_state:
        st.session_state.step1_nml_count = 0

    if 'step2_set' not in st.session_state:
        st.session_state.step2_set = False
    if 'step2_ref_check' not in st.session_state:
        st.session_state.step2_ref_check = False
    if 'step2_ref_nml' not in st.session_state:
        st.session_state.step2_ref_nml = False
    if 'step2_set_check' not in st.session_state:
        st.session_state.step2_set_check = False
    if 'selected_items' not in st.session_state:
        st.session_state.selected_items = []
        st.session_state.tittles = []
    if 'step2_errorlist' not in st.session_state:
        st.session_state.step2_errorlist = {}
    if 'step2_make' not in st.session_state:
        st.session_state.step2_make = False
    if 'step2_make_newnamelist' not in st.session_state:
        st.session_state.step2_make_newnamelist = False
    if 'step2_mange_sources' not in st.session_state:
        st.session_state.step2_mange_sources = False
    if 'step2_tomake_nml' not in st.session_state:
        st.session_state.step2_tomake_nml = False
    if 'step2_nml_count' not in st.session_state:
        st.session_state.step2_nml_count = 0

    if 'step3_set' not in st.session_state:
        st.session_state.step3_set = False
    if 'step3_make_newnamelist' not in st.session_state:
        st.session_state.step3_make_newnamelist = False
    if 'step3_mange_cases' not in st.session_state:
        st.session_state.step3_mange_cases = False
    if 'step3_make' not in st.session_state:
        st.session_state.step3_make = False
    if 'step3_nml' not in st.session_state:
        st.session_state.step3_nml = False
    if 'step3_nml_count' not in st.session_state:
        st.session_state.step3_nml_count = 0
    if 'step3_make_check' not in st.session_state:
        st.session_state.step3_make_check = False
    if 'step3_sim_check' not in st.session_state:
        st.session_state.step3_sim_check = False
    if 'step3_sim_nml' not in st.session_state:
        st.session_state.step3_sim_nml = False
    if 'step3_set_check' not in st.session_state:
        st.session_state.step3_set_check = False
    if 'step3_errorlist' not in st.session_state:
        st.session_state.step3_errorlist = {}

    if 'run_string' not in st.session_state:
        st.session_state.run_string = ''
    if 'run_err' not in st.session_state:
        st.session_state.run_err = ''
    if 'step4_set' not in st.session_state:
        st.session_state.step4_set = False
    if 'step5_figures' not in st.session_state:
        st.session_state.step5_figures = False
    if 'step5_files' not in st.session_state:
        st.session_state.step5_files = False
    if 'step5_Comparison' not in st.session_state:
        st.session_state.step5_Comparison = False

    if 'step6_stat_set' not in st.session_state:
        st.session_state.step6_stat_set = False
    if 'step6_stat_setect_check' not in st.session_state:
        st.session_state.step6_stat_setect_check = False
    if 'step6_stat_set_check' not in st.session_state:
        st.session_state.step6_stat_set_check = False
    if 'step6_stat_make' not in st.session_state:
        st.session_state.step6_stat_make = False
    if 'step6_stat_nml' not in st.session_state:
        st.session_state.step6_stat_nml = False
    if 'step6_stat_check' not in st.session_state:
        st.session_state.step6_stat_check = False
    if 'step6_stat_run' not in st.session_state:
        st.session_state.step6_stat_run = False
    if 'step6_stat_figures' not in st.session_state:
        st.session_state.step6_stat_figures = False
    if 'step6_stat_show' not in st.session_state:
        st.session_state.step6_stat_show = False
    if 'step6_stat_replot' not in st.session_state:
        st.session_state.step6_stat_replot = False

    if 'step1' not in st.session_state:
        st.session_state.step1 = False
    if 'step2' not in st.session_state:
        st.session_state.step2 = False
    if 'step3' not in st.session_state:
        st.session_state.step3 = False
    if 'step4' not in st.session_state:
        st.session_state.step4 = False
    if 'filenames' not in st.session_state:
        st.session_state.filenames = {}
    if 'item_checkbox' not in st.session_state:
        st.session_state.item_checkbox = {}


def initial():
    if 'clear_state' not in st.session_state:
        st.session_state['clear_state'] = False
    if 'switch_button1_onclick' not in st.session_state:
        st.session_state.switch_button1_onclick = 0
    if 'switch_button2_onclick' not in st.session_state:
        st.session_state.switch_button2_onclick = 0
    if 'switch_button3_onclick' not in st.session_state:
        st.session_state.switch_button3_onclick = 0
    if 'switch_button4_onclick' not in st.session_state:
        st.session_state.switch_button4_onclick = 0
    if 'switch_button5_onclick' not in st.session_state:
        st.session_state.switch_button5_onclick = 0
    if 'switch_button6_onclick' not in st.session_state:
        st.session_state.switch_button6_onclick = 0
    if 'clear_state_onclick' not in st.session_state:
        st.session_state.clear_state_onclick = 0
    if 'selected' not in st.session_state:
        st.session_state.selected = None
    if 'menu_option' not in st.session_state:
        st.session_state.menu_option = None


def define_step_st():
    def define1_step0():
        st.session_state.step1_general = False
        st.session_state.step1_metrics = False
        st.session_state.step1_evaluation = False
        st.session_state.step2_set = False
        st.session_state.step3_set = False

    def define1_step1():
        st.session_state.step1_general = True
        st.session_state.step1_metrics = False
        st.session_state.step1_evaluation = False
        st.session_state.step2_set = False
        st.session_state.step3_set = False

    def define2_step1():
        st.session_state.step1_general = True
        st.session_state.step1_metrics = True
        st.session_state.step1_evaluation = False
        st.session_state.step2_set = False
        st.session_state.step3_set = False

    def define3_step1():
        st.session_state.step1_general = True
        st.session_state.step1_metrics = True
        st.session_state.step1_evaluation = True
        st.session_state.step2_set = False
        st.session_state.step3_set = False

    def define1_step2():
        st.session_state.step2_set = True
        st.session_state.step2_make_newnamelist = False
        st.session_state.step2_mange_sources = False
        st.session_state.step2_make = False
        st.session_state.step2_tomake_nml = False
        st.session_state.step3_set = False

    def define2_step2():
        st.session_state.step2_set = True
        st.session_state.step2_make_newnamelist = False
        st.session_state.step2_mange_sources = False
        st.session_state.step2_make = True
        st.session_state.step2_tomake_nml = False
        st.session_state.step3_set = False

    def define3_step2():
        st.session_state.step2_set = True
        st.session_state.step2_make_newnamelist = False
        st.session_state.step2_mange_sources = False
        st.session_state.step2_make = True
        st.session_state.step2_tomake_nml = True
        st.session_state.step3_set = False

    def define1_step3():
        st.session_state.step2_set = True
        # ------------------------------------
        st.session_state.step3_set = True
        st.session_state.step3_make_newnamelist = False
        st.session_state.step3_mange_cases = False
        st.session_state.step3_make = False
        st.session_state.step3_nml = False
        st.session_state.step4_run = False

    def define2_step3():
        st.session_state.step2_set = True
        st.session_state.step3_set = True
        st.session_state.step3_make_newnamelist = False
        st.session_state.step3_mange_cases = False
        st.session_state.step3_make = True
        st.session_state.step3_nml = False
        st.session_state.step4_run = False

    def define3_step3():
        st.session_state.step2_set = True
        st.session_state.step3_set = True
        st.session_state.step3_make_newnamelist = False
        st.session_state.step3_mange_cases = False
        st.session_state.step3_make = True
        st.session_state.step3_nml = True
        st.session_state.step4_run = False

    def define5_figures():
        st.session_state.step5_figures = True
        st.session_state.step5_files = False
        st.session_state.step5_Comparison = False

    def define5_files():
        st.session_state.step5_figures = False
        st.session_state.step5_files = True
        st.session_state.step5_Comparison = False

    def define5_Comparison():
        st.session_state.step5_figures = False
        st.session_state.step5_files = False
        st.session_state.step5_Comparison = True

    def define6_show():
        # st.session_state.step6_stat_set = False
        # st.session_state.step6_stat_make = False
        st.session_state.step6_stat_run = True
        # st.session_state.step6_stat_figures = True
        st.session_state.step6_stat_show = True
        st.session_state.step6_stat_replot = False

    def define6_replot():
        st.session_state.step6_stat_show = False
        st.session_state.step6_stat_replot = True

    return {
        'step1': {
            'initial': define1_step0,
            'define1': define1_step1,
            'define2': define2_step1,
            'define3': define3_step1
        },
        'step2': {
            'define1': define1_step2,
            'define2': define2_step2,
            'define3': define3_step2
        },
        'step3': {
            'define1': define1_step3,
            'define2': define2_step3,
            'define3': define3_step3
        },
        'step5': {
            'figures': define5_figures,
            'files': define5_files,
            'Comparison': define5_Comparison
        },
        'step6': {
            'show': define6_show,
            'replot': define6_replot
        }
    }


step_functions = define_step_st()


def on_click_handler(step_func):
    step_func()


if __name__ == "__main__":

    initial_information = initial_setting()
    initial_st(initial_information)


    def switch_button_index(select):
        if select is None:
            return 0
        my_list = ["Home", "Evaluation", "Running", 'Visualization', 'Statistics']
        index = my_list.index(select)
        return index


    def define_clear():
        st.session_state.selected = None
        st.session_state.clear_state_onclick = +1
        st.session_state['menu_option'] = switch_button_index(st.session_state.selected)
        st.session_state['clear_state'] = True


    def on_change(key):
        selection = st.session_state[key]
        st.session_state['menu_option'] = switch_button_index(selection)


    if st.session_state.get('switch_button', False):
        st.session_state['menu_option'] = (switch_button_index(st.session_state.selected) + 1) % 5
        manual_select = st.session_state['menu_option']
    else:
        manual_select = None

    with st.sidebar:
        st.logo('./GUI/Namelist_lib/None.png', size='large', icon_image='./GUI/Namelist_lib/Openbench_Logo.png')
        st.image('./GUI/Namelist_lib/Openbench_Logo_n.png')
        if st.session_state.clear_state_onclick > 0:
            if st.session_state.clear_state:
                manual_select = st.session_state['menu_option']
        if st.session_state.switch_button1_onclick > 0:
            if 'switch_button1' in st.session_state.keys():
                if st.session_state.switch_button1:
                    manual_select = st.session_state['menu_option']
        if st.session_state.switch_button2_onclick > 0:
            if 'switch_button2' in st.session_state.keys():
                if st.session_state.switch_button2:
                    manual_select = st.session_state['menu_option']
        if 'switch_button3' in st.session_state.keys():
            if st.session_state.switch_button3 & (st.session_state.switch_button3_onclick > 0):
                manual_select = st.session_state['menu_option']
        if 'switch_button4' in st.session_state.keys():
            if st.session_state.switch_button4 & (st.session_state.switch_button4_onclick > 0):
                manual_select = st.session_state['menu_option']
        if 'switch_button5' in st.session_state.keys():
            if st.session_state.switch_button5 & (st.session_state.switch_button5_onclick > 0):
                manual_select = st.session_state['menu_option']
        if 'switch_button6' in st.session_state.keys():
            if st.session_state.switch_button6 & (st.session_state.switch_button6_onclick > 0):
                manual_select = st.session_state['menu_option']

        st.session_state['selected'] = option_menu(
            menu_title="Main Menu",
            options=["Home", "Evaluation", "Running", 'Visualization', 'Statistics'],
            icons=['house-heart', 'list-check', "list-task", "easel"],
            menu_icon="cast",
            default_index=0,
            on_change=on_change,
            manual_select=manual_select,
            key='menu_4'
        )
        st.button(f':point_right: Move to Next', key='switch_button')

    define_step_st()
    with st.sidebar:
        if st.session_state.selected == "Evaluation":
            st.divider()
            st.button(':zero: Initial', on_click=lambda: on_click_handler(step_functions['step1']['initial']),
                      help='Beck to Initial page', use_container_width=True)
            # st.divider()
            # st.button(':one: General', on_click=lambda: on_click_handler(step_functions['step1']['define1']),
            #           help='Beck to General page', use_container_width=True)
            # # st.button(':one: Metrics and Scores', on_click=lambda: on_click_handler(step_functions['step1']['define2']),
            # #           help='Go to Metrics page', use_container_width=True)
            # # st.button(':one: Evaluation', on_click=lambda: on_click_handler(step_functions['step1']['define3']),
            # #           help='Go to Evaluation page', use_container_width=True)
            # if st.session_state.step1:
            #     st.divider()
            #     st.button(':two:  Reference Setting', on_click=lambda: on_click_handler(step_functions['step2']['define1']),
            #               help='Go to Reference page', use_container_width=True)
            # #     st.button(':two:  Reference Making', on_click=lambda: on_click_handler(step_functions['step2']['define2']),
            # #               help='Go to Reference page', use_container_width=True)
            # #     st.button(':two:  Reference namelist', on_click=lambda: on_click_handler(step_functions['step2']['define3']),
            # #               help='Go to Reference page', use_container_width=True)
            # if st.session_state.step1 & st.session_state.step2:
            #     st.divider()
            #     st.button(':three:  Simulation Setting', on_click=lambda: on_click_handler(step_functions['step3']['define1']),
            #               help='Go to Reference page', use_container_width=True)
            #     st.button(':three:  Simulation Making', on_click=lambda: on_click_handler(step_functions['step3']['define2']),
            #               help='Go to Reference page', use_container_width=True)
            #     st.button(':three:  Simulation namelist', on_click=lambda: on_click_handler(step_functions['step3']['define3']),
            #               help='Go to Reference page', use_container_width=True)
            st.session_state.step4_run = False
        if st.session_state.selected == "Visualization":
            st.divider()
            st.button(':bar_chart: Showing Figures', on_click=lambda: on_click_handler(step_functions['step5']['figures']),
                      help='Page to displays the default drawing of the system', use_container_width=True)
            st.button(':books: Metrics/Scores items replot', on_click=lambda: on_click_handler(step_functions['step5']['files']),
                      help='Page for custom processing of data and images', use_container_width=True)
            st.button(':balloon: Comparison items replot',
                      on_click=lambda: on_click_handler(step_functions['step5']['Comparison']),
                      help='Page for redraw the image', use_container_width=True)
        if st.session_state.selected == "Statistics":
            if st.session_state.step6_stat_run:
                st.divider()
                st.button(':bar_chart: Showing Figures', on_click=lambda: on_click_handler(step_functions['step6']['show']),
                          help='Page to displays the default drawing of the system', use_container_width=True)
                st.button(':balloon: Statistics items replot',
                          on_click=lambda: on_click_handler(step_functions['step6']['replot']),
                          help='Page for redraw the image', use_container_width=True)

    with st.sidebar:
        st.divider()
        initial()
        if st.button(':large_red_square: Clear state', on_click=define_clear):
            if st.session_state:
                for key in st.session_state.keys():
                    del st.session_state[key]
            initial_st(initial_information)

    Page_control = Pages_control()
    if st.session_state.selected == "Home":
        show_info()
    if st.session_state.selected == "Evaluation":
        Page_control.main()
    if st.session_state.selected == "Running":
        Page_control.run()
    if st.session_state.selected == "Visualization":
        Page_control.visual()
    if st.session_state.selected == "Statistics":
        Page_control.Stats()
