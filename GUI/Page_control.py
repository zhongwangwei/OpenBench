import streamlit as st
# from Namelist_lib.namelist_read import NamelistReader, GeneralInfoReader, UpdateNamelist, UpdateFigNamelist
from Namelist_lib.namelist_info import initial_setting
from Page_make_validation import make_initional, make_reference, make_simulation
from Page_run import run_validation
from Page_visualization import visualization_validation, visualization_replot_files, visualization_replot_Comparison


# st.set_page_config(page_title="Validation Pages", page_icon="🌍", layout="wide")

class Pages_control:
    def __init__(self):
        self.author = "Qingchen Xu/xuqingchen0@gmail.com"
        self.coauthor = "Zhongwang Wei/@gmail.com"

    def main(self):
        # print('Create validation namelist -------------------')

        initial_information = initial_setting()
        # nl = namelist_read()

        nml_data = make_initional(initial_information)
        ref_data = make_reference(initial_information)
        sim_data = make_simulation(initial_information)
        # run_page = run_validation()
        if (
                not st.session_state.step1_general) & (
                not st.session_state.step1_metrics) & (
                not st.session_state.step1_evaluation) & (
                not st.session_state.step2_set):
            nml_data.home()

        if (
                st.session_state.step1_general) & (
                not st.session_state.step1_metrics) & (
                not st.session_state.step1_evaluation) & (
                not st.session_state.step2_set):
            nml_data.step1_general()

        if (
                st.session_state.step1_general) & (
                st.session_state.step1_metrics) & (
                not st.session_state.step1_evaluation) & (
                not st.session_state.step2_set):
            nml_data.step1_metrics()

        if (
                st.session_state.step1_general) & (
                st.session_state.step1_metrics) & (
                st.session_state.step1_evaluation) & (
                not st.session_state.step2_set):
            nml_data.step1_evaluation()

        if (
                st.session_state.step2_set & (
                not st.session_state.step2_make_newnamelist) & (
                not st.session_state.step2_mange_sources) & (
                not st.session_state.step2_make) & (
                not st.session_state.step2_tomake_nml) & (
                not st.session_state.step3_set)):
            ref_data.step2_set()

        if (
                st.session_state.step2_set) & (
                st.session_state.step2_make_newnamelist) & (
                not st.session_state.step2_mange_sources) & (
                not st.session_state.step2_make) & (
                not st.session_state.step2_tomake_nml) & (
                not st.session_state.step3_set):
            ref_data.step2_make_new_refnml()

        if (
                st.session_state.step2_set) & (
                not st.session_state.step2_make_newnamelist) & (
                st.session_state.step2_mange_sources) & (
                not st.session_state.step2_make) & (
                not st.session_state.step2_tomake_nml) & (
                not st.session_state.step3_set):
            ref_data.step2_mange_sources()

        if (
                st.session_state.step2_set) & (
                st.session_state.step2_make) & (
                not st.session_state.step2_tomake_nml) & (
                not st.session_state.step3_set):
            ref_data.step2_make()

        if (
                st.session_state.step2_set) & (
                st.session_state.step2_make) & (
                st.session_state.step2_tomake_nml) & (
                not st.session_state.step3_set):
            ref_data.step2_ref_nml()

        if (
                st.session_state.step3_set & (
                not st.session_state.step3_make_newnamelist) & (
                not st.session_state.step3_mange_cases) & (
                not st.session_state.step3_make) & (
                not st.session_state.step3_nml) & (
                not st.session_state.step4_run)
        ):
            sim_data.step3_set()

        if (
                st.session_state.step3_set & (
                st.session_state.step3_make_newnamelist) & (
                not st.session_state.step3_mange_cases) & (
                not st.session_state.step3_make) & (
                not st.session_state.step3_nml) & (
                not st.session_state.step4_run)
        ):
            sim_data.step3_make_new_simnml()

        if (
                st.session_state.step3_set & (
                not st.session_state.step3_make_newnamelist) & (
                st.session_state.step3_mange_cases) & (
                not st.session_state.step3_make) & (
                not st.session_state.step3_nml) & (
                not st.session_state.step4_run)
        ):
            sim_data.step3_mange_simcases()

        if (
                st.session_state.step3_set &
                st.session_state.step3_make & (
                not st.session_state.step3_nml) & (
                not st.session_state.step4_run)
        ):
            sim_data.step3_make()

        if (
                st.session_state.step3_set &
                st.session_state.step3_make &
                st.session_state.step3_nml & (
                not st.session_state.step4_run)
        ):
            sim_data.step3_sim_nml()

        # if st.session_state.step3_nml & st.session_state.step4_set & (not st.session_state.step4_run):
        #     run_page.run()


    def run(self):

        run_pages = run_validation()
        if not st.session_state.step3 or not st.session_state.step4_set:
            run_pages.set_errors()
        if st.session_state.step3 and st.session_state.step4_set:  # & (not st.session_state.step4_run): #& (not st.session_state.step4_check)
            run_pages.run()
        # TODO: 暂时不加入check部分，之后可以添加选择项，根据选择定向检查某一部分或者直接不检查。

    def visual(self):


        def switch_button_index(select):
            my_list = ["Home", "Evaluation", "Running", 'Visualization']
            index = my_list.index(select)
            return index

        visual_pages = visualization_validation()
        replot_file_pages = visualization_replot_files()
        Comparison_replot_pages = visualization_replot_Comparison()

        if not st.session_state.step4:
            visual_pages.set_errors()
        if st.session_state.step4:  # & (not st.session_state.step4_run): #& (not st.session_state.step4_check)
            if st.session_state.step5_figures & (not st.session_state.step5_files) & (not st.session_state.step5_Comparison):
                visual_pages.visualizations()
            if (not st.session_state.step5_figures) & (not st.session_state.step5_files) & (not st.session_state.step5_Comparison):
                visual_pages.visualizations()
            if st.session_state.step5_files & (not st.session_state.step5_figures) & (not st.session_state.step5_Comparison):
                replot_file_pages.Showing_for_files()
            if (not st.session_state.step5_figures) & (not st.session_state.step5_files) & (st.session_state.step5_Comparison):
                Comparison_replot_pages.Comparison_replot()
            # st.divider()
            if 'switch_button4_onclick' not in st.session_state:
                st.session_state.switch_button4_onclick = 0

            def define_run():
                if st.session_state.get('switch_button4', False):
                    st.session_state.switch_button4_onclick = +1
                    st.session_state['menu_option'] = (switch_button_index(st.session_state.selected) - 1) % 4

            st.divider()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.button(':point_left: Previous step', key='switch_button4', on_click=define_run,
                          help='Press twice to go to Run page')



