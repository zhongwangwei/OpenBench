# -*- coding: utf-8 -*-
"""
Land Surface Model 2024 Benchmark Evaluation System

This script is the main entry point for the evaluation system. It handles the
configuration, data processing, Evaluation, comparison, and statistical analysis
of land surface model outputs against reference data.

Author: Zhongwang Wei (zhongwang007@gmail.com)
Version: 0.1
Release: 0.1
Date: Mar 2023
"""

import sys
import os
import shutil
import time
from Mod_Namelist import NamelistReader, GeneralInfoReader, UpdateNamelist, UpdateFigNamelist
from Mod_Evaluation import Evaluation_grid, Evaluation_stn
from Mod_DatasetProcessing import DatasetProcessing
from Mod_Comparison  import ComparisonProcessing
from Mod_Statistics import StatisticsProcessing

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'

def print_welcome_message():
    """Print a more beautiful welcome message and ASCII art."""
    print("\n\n")
    print("=" * 80)
    print("""
    \033[1;36m  
   ____                   ____                  _     
  / __ \\                 |  _ \\                | |    
 | |  | |_ __   ___ _ __ | |_) | ___ _ __   ___| |__  
 | |  | | '_ \\ / _ \\ '_ \\|  _ < / _ \\ '_ \\ / __| '_ \\ 
 | |__| | |_) |  __/ | | | |_) |  __/ | | | (__| | | |
  \\____/| .__/ \\___|_| |_|____/ \\___|_| |_|\\___|_| |_|
        | |                                           
        |_|                                           \033[0m
    """)
    print("\033[1;32m" + "=" * 80 + "\033[0m")
    print("\033[1;33mWelcome to OpenBench: The Open Land Surface Model Benchmark Evaluation System!\033[0m")
    print("\033[1;32m" + "=" * 80 + "\033[0m")
    print("\n\033[1mThis system evaluate various land surface model outputs against reference data.\033[0m")
    print("\n\033[1;34mKey Features:\033[0m")
    print("  • Multi-model support")
    print("  • Comprehensive variable evaluation")
    print("  • Advanced metrics and scoring")
    print("  • Customizable benchmarking")
    print("\n\033[1;32m" + "=" * 80 + "\033[0m")
    print("\033[1;35mInitializing OpenBench Evaluation System...\033[0m")
   # input("\033[1mPress Enter to begin the benchmarking process...\033[0m")
    print("\033[1;32m" + "=" * 80 + "\033[0m")
    print("\n")

def setup_directories(main_nl):
    """Create necessary directories for the evaluation process."""
    base_path = os.path.join(main_nl['general']["basedir"], main_nl['general']['basename'])
    directories = {
        'tmp': os.path.join(base_path, 'tmp'),
        'scratch': os.path.join(base_path, 'scratch'),
        'metrics': os.path.join(base_path, 'output', 'metrics'),
        'scores': os.path.join(base_path, 'output', 'scores'),
        'data': os.path.join(base_path, 'output', 'data')
    }
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    return directories

def load_namelists(nl, main_nl):
    """Load reference, simulation, and statistics namelists."""
    ref_nml = nl.read_namelist(main_nl["general"]["reference_nml"])
    sim_nml = nl.read_namelist(main_nl["general"]["simulation_nml"])
    try:
        stats_nml = nl.read_namelist(main_nl["general"]["statistics_nml"])
    except:
        stats_nml = None
    fig_nml = nl.read_namelist(main_nl["general"]["figure_nml"])
    return ref_nml, sim_nml, stats_nml, fig_nml

def run_evaluation(main_nl, sim_nml, ref_nml, evaluation_items, metric_vars, score_vars, comparison_vars, statistic_vars,fig_nml):
    """Run the evaluation process for each item."""
    for evaluation_item in evaluation_items:
        print(f"Start running {evaluation_item} evaluation...")
        if evaluation_item not in sim_nml:
            print(f"Error: {evaluation_item} is not in the simulation namelist!")
            exit(1)
        
        sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
        ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']
        
        # Ensure sources are lists
        sim_sources = [sim_sources] if isinstance(sim_sources, str) else sim_sources
        ref_sources = [ref_sources] if isinstance(ref_sources, str) else ref_sources
        
        for ref_source in ref_sources:
            onetimeref=True
            for sim_source in sim_sources:
                process_evaluation(onetimeref,main_nl, sim_nml, ref_nml, metric_vars, score_vars, comparison_vars, statistic_vars, evaluation_item, sim_source, ref_source,fig_nml)
                onetimeref=False

def process_evaluation(onetimeref,main_nl, sim_nml, ref_nml, metric_vars, score_vars, comparison_vars, statistic_vars, evaluation_item, sim_source, ref_source,fig_nml):
    """Process a single evaluation for given sources."""
    print(f"Processing {evaluation_item} - ref: {ref_source} - sim: {sim_source}")
    general_info_object = GeneralInfoReader(main_nl, sim_nml, ref_nml, metric_vars, score_vars, comparison_vars, statistic_vars, evaluation_item, sim_source, ref_source)
    general_info = general_info_object.to_dict()
    
    # Add ref_source and sim_source to general_info
    general_info['ref_source'] = ref_source
    general_info['sim_source'] = sim_source

    dataset_processer = DatasetProcessing(general_info)
    if general_info['ref_data_type'] == 'stn' or general_info['sim_data_type'] == 'stn':
        onetimeref=True
    if onetimeref==True:
        dataset_processer.process('ref')
    else:
        print("Skip processing ref data")
    dataset_processer.process('sim')
    
    # Clear scratch directory
    scratch_dir = os.path.join(main_nl['general']["basedir"], main_nl['general']['basename'], 'scratch')
    shutil.rmtree(scratch_dir, ignore_errors=True)
    os.makedirs(scratch_dir)
    
    # Run Evaluation
    if general_info['ref_data_type'] == 'stn' or general_info['sim_data_type'] == 'stn':
        evaluater = Evaluation_stn(general_info,fig_nml)
        evaluater.make_evaluation_P()
    else:
        evaluater = Evaluation_grid(general_info,fig_nml)
        evaluater.make_Evaluation()
    
    evaluater.make_plot_index()

def run_comparison(main_nl, sim_nml, ref_nml, evaluation_items, score_vars, metric_vars, comparison_vars,fig_nml):
    """Run the comparison process for each comparison variable."""
    basedir = os.path.join(main_nl['general']['basedir'], main_nl['general']['basename'])
    ch = ComparisonProcessing(main_nl, score_vars, metric_vars)
    
    for cvar in comparison_vars:
        print("\033[1;32m" + "=" * 80 + "\033[0m")
        print(f"********************Start running {cvar} comparison...******************")
        comparison_method = f'scenarios_{cvar}_comparison'
        if hasattr(ch, comparison_method):
            getattr(ch, comparison_method)(basedir, sim_nml, ref_nml, evaluation_items, score_vars, metric_vars)
        else:
            print(f"Error: The {cvar} module does not exist!")
            print(f"Please add the {cvar} function in the Comparison_handle class!")
            exit(1)
        print(f"<<<<<<<<<<<<<<<<<<<<<<<<<Done running {cvar} comparison...<<<<<<<<<<<<<<<<<<<<<<")
        print("\033[1;32m" + "=" * 80 + "\033[0m")

def run_statistics(main_nl, stats_nml, statistic_vars):
    """Run statistical analysis for each statistic variable."""
    if not statistic_vars:
        return
    
    print("Running statistical analysis...")
    basedir = os.path.join(main_nl['general']['basedir'], main_nl['general']['basename'])
    stats_handler = StatisticsProcessing(
        main_nl, stats_nml,
        os.path.join(basedir, 'output', 'statistics'),
        num_cores=main_nl['general']['num_cores']
    )
    
    for statistic in statistic_vars:
        print(f"Running {statistic} analysis...")
        stats_handler.run_analysis(statistic)
    
    print("Statistical analysis completed.")

def main():
    """Main function to orchestrate the evaluation process."""
    print_welcome_message()
    
    # Initialize NamelistReader and load main namelist
    nl = NamelistReader()
    main_nl = nl.read_namelist(sys.argv[1])

    # Setup directories
    setup_directories(main_nl)
    
    # Load namelists
    ref_nml, sim_nml, stats_nml, fig_nml = load_namelists(nl, main_nl)
    
    # Select variables based on main namelist settings
    evaluation_items = nl.select_variables(main_nl['evaluation_items']).keys()
    metric_vars = nl.select_variables(main_nl['metrics']).keys()
    score_vars = nl.select_variables(main_nl['scores']).keys()
    comparison_vars = nl.select_variables(main_nl['comparsions']).keys()
    statistic_vars = nl.select_variables(main_nl['statistics']).keys()
    
    # Update namelists
    UpdateNamelist(main_nl, sim_nml, ref_nml, evaluation_items)

    UpdateFigNamelist(main_nl, fig_nml, comparison_vars)
    # Run evaluation if enabled
    if main_nl['general']['evaluation']:
        start_time = time.time()
        run_evaluation(main_nl, sim_nml, ref_nml, evaluation_items, metric_vars, score_vars, comparison_vars, statistic_vars,fig_nml['Validation'])
        end_time = time.time()
        evaluation_time = (end_time - start_time)/60
        print(f"\n\033[1;36mEvaluation process completed in {evaluation_time:.2f} minutes.\033[0m")

    # Run comparison if enabled
    if main_nl['general']['comparison']:
        start_time = time.time()
        run_comparison(main_nl, sim_nml, ref_nml, evaluation_items, score_vars, metric_vars, comparison_vars,fig_nml['Comparison'])
        end_time = time.time()
        comparison_time = (end_time - start_time)/60
        print(f"\n\033[1;36mComparison process completed in {comparison_time:.2f} minutes.\033[0m")

    # Run statistics if enabled
    if main_nl['general']['statistics']:
        start_time = time.time()
        run_statistics(main_nl, stats_nml, statistic_vars)
        end_time = time.time()
        statistic_time = (end_time - start_time)/60
        print(f"\n\033[1;36mStatistics process completed in {statistic_time:.2f} minutes.\033[0m")

if __name__ == '__main__':
    main()
