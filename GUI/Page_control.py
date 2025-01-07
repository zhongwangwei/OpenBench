import streamlit as st

# from Namelist_lib.namelist_read import NamelistReader, GeneralInfoReader, UpdateNamelist, UpdateFigNamelist
from Namelist_lib.namelist_info import initial_setting
from Page_make_validation import make_initional, make_reference, make_simulation
from Page_run import run_validation
from Page_statistic import Process_stastic
from Page_visualization import visualization_validation, visualization_replot_files, visualization_replot_Comparison


# st.set_page_config(page_title="Validation Pages", page_icon="🌍", layout="wide")

class Pages_control:
    def __init__(self):
        self.author = "Qingchen Xu/xuqingchen0@gmail.com"
        self.coauthor = "Zhongwang Wei/@gmail.com"

    def switch_button_index(self, select):
        my_list = ["Home", "Evaluation", "Running", 'Visualization', 'Statistics']
        index = my_list.index(select)
        return index

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

    def run(self):

        run_pages = run_validation()
        if not st.session_state.step3 or not st.session_state.step4_set:
            run_pages.set_errors()
            next_button_disable1, next_button_disable2 = False, False
        if st.session_state.step3 and st.session_state.step4_set:  # & (not st.session_state.step4_run): #& (not st.session_state.step4_check)
            next_button_disable1, next_button_disable2 = run_pages.run()

        # TODO: 暂时不加入check部分，之后可以添加选择项，根据选择定向检查某一部分或者直接不检查。
        def define_evaluation():
            if st.session_state.get('switch_button2', False):
                st.session_state.switch_button2_onclick = +1
                st.session_state['menu_option'] = (self.switch_button_index(st.session_state.selected) - 1) % 5
                st.session_state.step4_run = False
                st.session_state.step4 = False

        def define_visual():
            if st.session_state.get('switch_button3', False):
                st.session_state.step4 = True
                st.session_state.switch_button3_onclick = +1
                st.session_state['menu_option'] = (self.switch_button_index(st.session_state.selected) + 1) % 5

        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.button(':point_left: Evaluation Page', disabled=next_button_disable1, key='switch_button2',
                      on_click=define_evaluation,
                      help='Press back to Evaluation page')
        with col3:
            st.button('Visualization Page :point_right: ', disabled=next_button_disable2, key='switch_button3',
                      on_click=define_visual, use_container_width=True,
                      help='Press go to Visualization page')

    def visual(self):

        visual_pages = visualization_validation()
        replot_file_pages = visualization_replot_files()
        Comparison_replot_pages = visualization_replot_Comparison()

        if not st.session_state.step4:
            visual_pages.set_errors()
        if st.session_state.step4:  # & (not st.session_state.step4_run): #& (not st.session_state.step4_check)
            if st.session_state.step5_figures & (not st.session_state.step5_files) & (not st.session_state.step5_Comparison):
                visual_pages.visualizations()
            if (not st.session_state.step5_figures) & (not st.session_state.step5_files) & (
                    not st.session_state.step5_Comparison):
                visual_pages.visualizations()
            if st.session_state.step5_files & (not st.session_state.step5_figures) & (not st.session_state.step5_Comparison):
                replot_file_pages.Showing_for_files()
            if (not st.session_state.step5_figures) & (not st.session_state.step5_files) & (st.session_state.step5_Comparison):
                Comparison_replot_pages.Comparison_replot()

        def define_run():
            if st.session_state.get('switch_button4', False):
                st.session_state.switch_button4_onclick = +1
                st.session_state['menu_option'] = (self.switch_button_index(st.session_state.selected) - 1) % 5

        def define_stat():
            if st.session_state.get('switch_button5', False):
                st.session_state.switch_button5_onclick = +1
                st.session_state['menu_option'] = (self.switch_button_index(st.session_state.selected) + 1) % 5

        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.button(':point_left: Running Page', key='switch_button4', on_click=define_run,
                      help='Press twice to go to Run page')
        with col4:
            st.button(':point_right: Statistic Page', key='switch_button5', on_click=define_stat,
                      help='Press twice to go to Statistic page')

    def Stats(self):

        initial_information = initial_setting()
        statistic_process = Process_stastic(initial_information)
        if not st.session_state.step4:
            statistic_process.set_errors()
        if st.session_state.step4:  # & (not st.session_state.step4_run): #& (not st.session_state.step4_check)
            # st.write(st.session_state.step6_stat_set, st.session_state.step6_stat_make, st.session_state.step6_stat_run)
            if (
                    not st.session_state.step6_stat_set) & (
                    not st.session_state.step6_stat_make) & (
                    not st.session_state.step6_stat_run):
                statistic_process.statistic_set()
            if (
                    st.session_state.step6_stat_set & (
                    not st.session_state.step6_stat_make) & (
                    not st.session_state.step6_stat_run)
            ):
                statistic_process.statistic_make()
            if (
                    st.session_state.step6_stat_set &
                    st.session_state.step6_stat_make & (
                    not st.session_state.step6_stat_run)
            ):
                statistic_process.statistic_run()
            # if (
            #         st.session_state.step6_stat_set &
            #         st.session_state.step6_stat_make &
            #         st.session_state.step6_stat_run
            # ):
            #     statistic_process.Comparison_figures()
            if (
                    st.session_state.step6_stat_set &
                    st.session_state.step6_stat_make &
                    st.session_state.step6_stat_run & (
                    not st.session_state.step6_stat_show) & (
                    not st.session_state.step6_stat_replot)
            ):
                statistic_process.statistic_show()
            if (
                    st.session_state.step6_stat_set &
                    st.session_state.step6_stat_make &
                    st.session_state.step6_stat_run &
                    st.session_state.step6_stat_show & (
                    not st.session_state.step6_stat_replot)
            ):
                statistic_process.statistic_show()
            if (
                    st.session_state.step6_stat_set &
                    st.session_state.step6_stat_make &
                    st.session_state.step6_stat_run & (
                    not st.session_state.step6_stat_show) &
                    st.session_state.step6_stat_replot
            ):
                statistic_process.statistic_replot()

        # def define_visual():
        #     if st.session_state.get('switch_button6', False):
        #         st.session_state.switch_button6_onclick = +1
        #         st.session_state['menu_option'] = (self.switch_button_index(st.session_state.selected) - 1) % 5
        #
        # st.divider()
        # col1, col2, col3 = st.columns(3)
        # with col1:
        #     st.button(':point_left: Visualization Page', key='switch_button6', on_click=define_visual,use_container_width=True,
        #               help='Press go to Visualization page')
