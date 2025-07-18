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

import os
import shutil
import sys
import time
import glob
import xarray as xr
import numpy as np
from Mod_Comparison import ComparisonProcessing
from Mod_DatasetProcessing import DatasetProcessing
from Mod_Evaluation import Evaluation_grid, Evaluation_stn
from Mod_Landcover_Groupby import LC_groupby
from Mod_ClimateZone_Groupby import CZ_groupby
from config import NamelistReader, GeneralInfoReader, UpdateNamelist, UpdateFigNamelist
from Mod_Statistics import StatisticsProcessing
from Mod_Only_Drawing import Evaluation_grid_only_drawing, Evaluation_stn_only_drawing, LC_groupby_only_drawing, CZ_groupby_only_drawing, ComparisonProcessing_only_drawing
from Mod_Preprocessing import check_required_nml, run_files_check
from Mod_Converttype import Convert_Type
import logging
from datetime import datetime

# Import enhanced logging system
try:
    from Mod_LoggingSystem import setup_logging, get_logging_manager, configure_library_logging
    _HAS_ENHANCED_LOGGING = True
except ImportError:
    _HAS_ENHANCED_LOGGING = False

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
    print("  * Multi-model support")
    print("  * Comprehensive variable evaluation")
    print("  * Advanced metrics and scoring")
    print("  * Customizable benchmarking")
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
        'data': os.path.join(base_path, 'output', 'data'),
        'log': os.path.join(base_path, 'log')
    }
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)

    # Configure logging
    log_file = os.path.join(directories['log'], f'openbench_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    print('OpenBench Log File: {}'.format(log_file))
    
    if _HAS_ENHANCED_LOGGING:
        # Use enhanced logging system
        logging_manager = setup_logging(
            level=logging.INFO,
            console=True,
            file=True,
            structured=False,  # Can be enabled for JSON logs
            async_mode=False,  # Can be enabled for better performance
            base_dir=directories['log']
        )
        
        # Configure library logging to reduce noise
        configure_library_logging()
        
        # Add context information
        logging_manager.add_context(
            app_name="OpenBench",
            version="0.1",
            case_dir=base_path
        )
        
        # Set specific log file
        logging_manager.add_file_handler(
            filename=f'openbench_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            formatter=logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
    else:
        # Fallback to standard logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(stream=sys.stdout),
            ]
        )
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


def run_evaluation(main_nl, sim_nml, ref_nml, evaluation_items, metric_vars, score_vars, comparison_vars, statistic_vars,
                   fig_nml):
    """Run the evaluation process for each item."""
    for evaluation_item in evaluation_items:
        logging.info(f"Start running {evaluation_item} evaluation...")

        sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
        ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']
        # Ensure sources are lists
        sim_sources = [sim_sources] if isinstance(sim_sources, str) else sim_sources
        ref_sources = [ref_sources] if isinstance(ref_sources, str) else ref_sources
        # Rearrange reference sources to put station data first
        ref_sources = sorted(ref_sources, key=lambda x: 0 if ref_nml[evaluation_item].get(f'{x}_data_type') == 'stn' else 1)
        # Rearrange simulation sources to put station data first
        sim_sources = sorted(sim_sources, key=lambda x: 0 if sim_nml[evaluation_item].get(f'{x}_data_type') == 'stn' else 1)
        
        if not main_nl['general']['only_drawing']:
            for ref_source in ref_sources:
                onetimeref = True
                for sim_source in sim_sources:
                    process_mask(onetimeref, main_nl, sim_nml, ref_nml, metric_vars, score_vars, comparison_vars, statistic_vars,
                                evaluation_item, sim_source, ref_source, fig_nml)
                    onetimeref = False

        for ref_source in ref_sources:
            onetimeref = True
            for sim_source in sim_sources:
                process_evaluation(onetimeref, main_nl, sim_nml, ref_nml, metric_vars, score_vars, comparison_vars,
                                statistic_vars, evaluation_item, sim_source, ref_source, fig_nml)
                onetimeref = False

    main_nl['general']['IGBP_groupby'] = main_nl['general'].get('IGBP_groupby', 'True')
    main_nl['general']['PFT_groupby'] = main_nl['general'].get('PFT_groupby', 'True')
    if main_nl['general']['IGBP_groupby']:
        if main_nl['general']['only_drawing']:
            LC = LC_groupby_only_drawing(main_nl, score_vars, metric_vars)
        else:
            LC = LC_groupby(main_nl, score_vars, metric_vars)
        basedir = os.path.join(main_nl['general']['basedir'], main_nl['general']['basename'])
        LC.scenarios_IGBP_groupby_comparison(basedir, sim_nml, ref_nml, evaluation_items, score_vars, metric_vars,
                                             fig_nml['IGBP_groupby'])
    if main_nl['general']['PFT_groupby']:
        if main_nl['general']['only_drawing']:
            LC = LC_groupby_only_drawing(main_nl, score_vars, metric_vars)
        else:
            LC = LC_groupby(main_nl, score_vars, metric_vars)
        basedir = os.path.join(main_nl['general']['basedir'], main_nl['general']['basename'])
        LC.scenarios_PFT_groupby_comparison(basedir, sim_nml, ref_nml, evaluation_items, score_vars, metric_vars,
                                            fig_nml['PFT_groupby'])
    if main_nl['general']['Climate_zone_groupby']:
        if main_nl['general']['only_drawing']:
            CZ = CZ_groupby_only_drawing(main_nl, score_vars, metric_vars)
        else:
            CZ = CZ_groupby(main_nl, score_vars, metric_vars)
        basedir = os.path.join(main_nl['general']['basedir'], main_nl['general']['basename'])
        CZ.scenarios_CZ_groupby_comparison(basedir, sim_nml, ref_nml, evaluation_items, score_vars, metric_vars,
                                            fig_nml['Climate_zone_groupby'])

def process_mask(onetimeref, main_nl, sim_nml, ref_nml, metric_vars, score_vars, comparison_vars, statistic_vars, evaluation_item,
                 sim_source, ref_source, fig_nml):
    logging.info(f"Processing {evaluation_item} - ref: {ref_source} - sim: {sim_source}")
    general_info_object = GeneralInfoReader(main_nl, sim_nml, ref_nml, metric_vars, score_vars, comparison_vars, statistic_vars,
                                            evaluation_item, sim_source, ref_source)
    general_info = general_info_object.to_dict()
    # Add ref_source and sim_source to general_info
    general_info['ref_source'] = ref_source
    general_info['sim_source'] = sim_source

    dataset_processer = DatasetProcessing(general_info)
    if general_info['ref_data_type'] == 'stn' or general_info['sim_data_type'] == 'stn':
        onetimeref = True
    if onetimeref == True:
        dataset_processer.process('ref')
    else:
        logging.info("Skip processing ref data")
    dataset_processer.process('sim')

    # Clear scratch directory
    scratch_dir = os.path.join(main_nl['general']["basedir"], main_nl['general']['basename'], 'scratch')
    shutil.rmtree(scratch_dir, ignore_errors=True)
    logging.info(f"Re-creating output directory: {scratch_dir}")
    os.makedirs(scratch_dir)
    if main_nl['general']['unified_mask']:
        if general_info['ref_data_type'] == 'stn' or general_info['sim_data_type'] == 'stn':
            pass
        else:
            # Mask the observation data with simulation data to ensure consistent coverage
            logging.info("Mask the observation data with all simulation datasets to ensure consistent coverage")
            import time
            
            def wait_for_file(file_path, max_wait_time=30, check_interval=1):
                """Wait for a file to exist and be readable."""
                start_time = time.time()
                elapsed = 0
                while elapsed < max_wait_time:
                    if os.path.exists(file_path):
                        try:
                            size = os.path.getsize(file_path)
                            if size > 0:
                                logging.info(f"File found after {elapsed:.1f}s: {file_path} ({size} bytes)")
                                return True
                            else:
                                logging.debug(f"File exists but empty at {elapsed:.1f}s: {file_path}")
                        except (OSError, IOError) as e:
                            logging.debug(f"Error checking file at {elapsed:.1f}s: {e}")
                    else:
                        logging.debug(f"File still not found at {elapsed:.1f}s: {file_path}")
                    
                    time.sleep(check_interval)
                    elapsed = time.time() - start_time
                
                logging.warning(f"File not found after {max_wait_time}s: {file_path}")
                return False
            
            ref_file_path = f'{general_info["casedir"]}/output/data/{evaluation_item}_ref_{ref_source}_{general_info["ref_varname"]}.nc'
            # Convert to absolute path to ensure consistency
            ref_file_path_abs = os.path.abspath(ref_file_path)
            
            try:
                o = xr.open_dataset(ref_file_path_abs)[f'{general_info["ref_varname"]}']
            except FileNotFoundError as e:
                logging.info(f"Ref data file not found, processing now: {ref_file_path_abs}")
                logging.info(f"Processing ref data")
                dataset_processer.process('ref')
                
                # Check immediately after processing
                if os.path.exists(ref_file_path_abs):
                    logging.info(f"Ref file created immediately after processing: {ref_file_path_abs}")
                    o = xr.open_dataset(ref_file_path_abs)[f'{general_info["ref_varname"]}']
                else:
                    # Wait for the file to be created and available
                    logging.info(f"Waiting for ref data file to be created: {ref_file_path_abs}")
                    if wait_for_file(ref_file_path_abs):
                        o = xr.open_dataset(ref_file_path_abs)[f'{general_info["ref_varname"]}']
                    else:
                        logging.error(f"Timeout waiting for ref data file: {ref_file_path_abs}")
                        raise FileNotFoundError(f"Ref data file not available after processing: {ref_file_path_abs}")
            o = Convert_Type.convert_nc(o)
            
            sim_file_path = f'{general_info["casedir"]}/output/data/{evaluation_item}_sim_{sim_source}_{general_info["sim_varname"]}.nc'
            # Convert to absolute path to ensure consistency
            sim_file_path_abs = os.path.abspath(sim_file_path)
            
            try:
                s = xr.open_dataset(sim_file_path_abs)[f'{general_info["sim_varname"]}']
            except FileNotFoundError as e:
                logging.info(f"Sim data file not found, processing now: {sim_file_path_abs}")
                logging.info(f"Processing sim data")
                dataset_processer.process('sim')
                
                # Check immediately after processing
                if os.path.exists(sim_file_path_abs):
                    logging.info(f"Sim file created immediately after processing: {sim_file_path_abs}")
                    s = xr.open_dataset(sim_file_path_abs)[f'{general_info["sim_varname"]}']
                else:
                    # Wait for the sim file to be created and available
                    logging.info(f"Waiting for sim data file to be created: {sim_file_path_abs}")
                    if wait_for_file(sim_file_path_abs):
                        s = xr.open_dataset(sim_file_path_abs)[f'{general_info["sim_varname"]}']
                    else:
                        logging.error(f"Timeout waiting for sim data file: {sim_file_path_abs}")
                        raise FileNotFoundError(f"Sim data file not available after processing: {sim_file_path_abs}")
            s = Convert_Type.convert_nc(s)
            s['time'] = o['time']
            mask1 = np.isnan(s) | np.isnan(o)
            o.values[mask1] = np.nan
            # Use absolute path for final save as well
            o.to_netcdf(ref_file_path_abs)


def process_evaluation(onetimeref, main_nl, sim_nml, ref_nml, metric_vars, score_vars, comparison_vars, statistic_vars,
                       evaluation_item, sim_source, ref_source, fig_nml):
    """Process a single evaluation for given sources."""
    general_info_object = GeneralInfoReader(main_nl, sim_nml, ref_nml, metric_vars, score_vars, comparison_vars, statistic_vars,
                                            evaluation_item, sim_source, ref_source)
    general_info = general_info_object.to_dict()
    # Run Evaluation
    if general_info['ref_data_type'] == 'stn' or general_info['sim_data_type'] == 'stn':
        if main_nl['general']['only_drawing']:
            evaluater = Evaluation_stn_only_drawing(general_info, fig_nml)
        else:
            evaluater = Evaluation_stn(general_info, fig_nml)
        evaluater.make_evaluation_P()

    else:
        if main_nl['general']['only_drawing']:
            evaluater = Evaluation_grid_only_drawing(general_info, fig_nml)
        else:
            evaluater = Evaluation_grid(general_info, fig_nml)
        evaluater.make_Evaluation()


def run_comparison(main_nl, sim_nml, ref_nml, evaluation_items, score_vars, metric_vars, comparison_vars, fig_nml):
    """Run the comparison process for each comparison variable."""
    logging.info(" ")
    logging.info("╔═══════════════════════════════════════════════════════════════╗")
    logging.info("║                Comparison processes starting!                 ║")
    logging.info("╚═══════════════════════════════════════════════════════════════╝")
    logging.info(" ")
    basedir = os.path.join(main_nl['general']['basedir'], main_nl['general']['basename'])
    if main_nl['general']['only_drawing']:
        ch = ComparisonProcessing_only_drawing(main_nl, score_vars, metric_vars)
    else:
        ch = ComparisonProcessing(main_nl, score_vars, metric_vars)

    for cvar in comparison_vars:
        logging.info("\033[1;32m" + "=" * 80 + "\033[0m")
        logging.info(f"********************Start running {cvar} comparison...******************")
        if cvar in ['Mean', 'Median', 'Max', 'Min', 'Sum']:
            comparison_method = f'scenarios_Basic_comparison'
        else:
            comparison_method = f'scenarios_{cvar}_comparison'
        if hasattr(ch, comparison_method):
            getattr(ch, comparison_method)(basedir, sim_nml, ref_nml, evaluation_items, score_vars, metric_vars, fig_nml[cvar])
        else:
            logging.error(f"Error: The {cvar} module does not exist!")
            logging.error(f"Please add the {cvar} function in the Comparison_handle class!")
            exit(1)
        logging.info(f"<<<<<<<<<<<<<<<<<<<<<<<<<Done running {cvar} comparison...<<<<<<<<<<<<<<<<<<<<<<")
        logging.info("\033[1;32m" + "=" * 80 + "\033[0m")


def run_statistics(main_nl, stats_nml, statistic_vars, fig_nml):
    """Run statistical analysis for each statistic variable."""
    if not statistic_vars:
        return

    logging.info("Running statistical analysis...")
    basedir = os.path.join(main_nl['general']['basedir'], main_nl['general']['basename'])
    stats_handler = StatisticsProcessing(
        main_nl, stats_nml,
        os.path.join(basedir, 'output', 'statistics'),
        num_cores=main_nl['general']['num_cores']
    )

    for statistic in statistic_vars:
        logging.info("\033[1;32m" + "=" * 80 + "\033[0m")
        logging.info(f"********************Start running {statistic} analysis...******************")
        if statistic in ['Mean', 'Median', 'Max', 'Min', 'Sum']:
            statistic_method = f'scenarios_Basic_analysis'
        else:
            statistic_method = f'scenarios_{statistic}_analysis'
        if hasattr(stats_handler, statistic_method):
            getattr(stats_handler, statistic_method)(statistic, stats_nml[statistic], fig_nml[statistic])
        else:
            logging.error(f"Error: The {statistic} module does not exist!")
            logging.error(f"Please add the {statistic} function in the stats_handler class!")
            exit(1)
    logging.info("Statistical analysis completed.")


def main():
    """Main function to orchestrate the evaluation process."""
    print_welcome_message()

    # Initialize NamelistReader and load main namelist
    nl = NamelistReader()
    main_nl = nl.read_namelist(sys.argv[1])

    # Setup directories
    setup_directories(main_nl)

    logging.info("Starting OpenBench evaluation process...")

    # Load namelists
    ref_nml, sim_nml, stats_nml, fig_nml = load_namelists(nl, main_nl)

    # Select variables based on main namelist settings
    evaluation_items = nl.select_variables(main_nl['evaluation_items']).keys()
    metric_vars = nl.select_variables(main_nl['metrics']).keys()
    score_vars = nl.select_variables(main_nl['scores']).keys()
    comparison_vars = nl.select_variables(main_nl['comparisons']).keys()
    statistic_vars = nl.select_variables(main_nl['statistics']).keys()

    # Check required files before proceeding
    check_required_nml(main_nl, sim_nml, ref_nml, evaluation_items)

    # Update namelists
    UpdateNamelist(main_nl, sim_nml, ref_nml, evaluation_items)
    UpdateFigNamelist(main_nl, fig_nml, comparison_vars, statistic_vars)
    if main_nl['general']['statistics'] and not main_nl['general']['evaluation'] and not main_nl['general']['comparison']:
        pass
    else:
        run_files_check(main_nl, sim_nml, ref_nml, evaluation_items, metric_vars, score_vars, comparison_vars, statistic_vars,
                        fig_nml)

    # xu output nml information:
    # logging.info(f"xu output nml information before running")
    # logging.info(f"main_nl: {main_nl}, type: {type(main_nl)}")
    # logging.info(f"ref_nml: {ref_nml}, type: {type(ref_nml)}")
    # logging.info(f"sim_nml: {sim_nml}, type: {type(sim_nml)}")
    # logging.info(f"stats_nml: {stats_nml}, type: {type(stats_nml)}")
    # logging.info(f"fig_nml: {fig_nml}, type: {type(fig_nml)}")
    # logging.info(f"evaluation_items: {evaluation_items}, type: {type(evaluation_items)}")
    # logging.info(f"metric_vars: {metric_vars}, type: {type(metric_vars)}")
    # logging.info(f"score_vars: {score_vars}, type: {type(score_vars)}")
    # logging.info(f"comparison_vars: {comparison_vars}, type: {type(comparison_vars)}")
    # logging.info(f"statistic_vars: {statistic_vars}, type: {type(statistic_vars)}")

    main_nl['general']['only_drawing'] = main_nl['general'].get('only_drawing', 'True')

    # Run evaluation if enabled
    if main_nl['general']['evaluation']:
        start_time = time.time()
        run_evaluation(main_nl, sim_nml, ref_nml, evaluation_items, metric_vars, score_vars, comparison_vars, statistic_vars,
                       fig_nml['Validation'])
        end_time = time.time()
        evaluation_time = (end_time - start_time) / 60
        logging.info(f"Evaluation process completed in {evaluation_time:.2f} minutes.")

    # Run comparison if enabled
    if main_nl['general']['comparison']:
        start_time = time.time()
        run_comparison(main_nl, sim_nml, ref_nml, evaluation_items, score_vars, metric_vars, comparison_vars,
                       fig_nml['Comparison'])
        end_time = time.time()
        comparison_time = (end_time - start_time) / 60
        logging.info(f"Comparison process completed in {comparison_time:.2f} minutes.")

    # Run statistics if enabled
    if main_nl['general']['statistics']:
        start_time = time.time()
        run_statistics(main_nl, stats_nml, statistic_vars, fig_nml['Statistic'])
        end_time = time.time()
        statistic_time = (end_time - start_time) / 60
        logging.info(f"Statistics process completed in {statistic_time:.2f} minutes.")

    logging.info("OpenBench evaluation process completed successfully.")


if __name__ == '__main__':
    main()
