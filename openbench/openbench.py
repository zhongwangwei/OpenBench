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

import gc
import glob
import logging
import os
import platform
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import xarray as xr

# Try to import psutil for memory monitoring, use fallback if not available
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# Add parent directory to Python path for direct script execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# DatasetProcessing will be imported when needed to avoid circular imports
from openbench.config import NamelistReader, GeneralInfoReader, UpdateNamelist, UpdateFigNamelist
from openbench.visualization.Mod_Only_Drawing import Evaluation_grid_only_drawing, Evaluation_stn_only_drawing, LC_groupby_only_drawing, CZ_groupby_only_drawing, ComparisonProcessing_only_drawing
from openbench.data.Mod_Preprocessing import check_required_nml, run_files_check
from openbench.util.Mod_Converttype import Convert_Type
from openbench.util.Mod_ReportGenerator import ReportGenerator
from openbench.util.Mod_FileIO import safe_open_netcdf, safe_save_netcdf
from openbench.util.Mod_DatasetLoader import open_dataset as open_dataset_chunked
from openbench.util.Mod_ProgressMonitor import HeartbeatMonitor
from openbench.util.Mod_PreValidation import run_pre_validation, ValidationError
from openbench.util.Mod_ConfigCheck import (
    get_platform_colors,
    print_config_summary,
    perform_early_validation
)
from openbench.util.Mod_MemoryManager import (
    cleanup_memory,
    initialize_memory_management,
    get_memory_usage,
    log_memory_usage,
    cleanup_old_outputs
)
from openbench.util.Mod_DirectoryUtils import reset_directory
from openbench.util.Mod_CacheCleanup import cleanup_all_cache
from openbench.data.Mod_DatasetProcessing import DatasetProcessing


def _ensure_sysconf_accessible():
    """Patch os.sysconf to avoid sandbox PermissionError for semaphore queries.

    On Windows, os.sysconf does not exist, so this function is a no-op.
    On Unix/Linux, it patches os.sysconf to handle PermissionError gracefully.
    """
    # Skip on platforms without sysconf (e.g., Windows)
    if not hasattr(os, 'sysconf'):
        return

    if getattr(os, "_openbench_sysconf_patched", False):
        return

    original_sysconf = os.sysconf

    def safe_sysconf(name):
        if name == 'SC_SEM_NSEMS_MAX':
            try:
                return original_sysconf(name)
            except PermissionError:
                logging.warning(
                    "SC_SEM_NSEMS_MAX not accessible; using conservative fallback value."
                )
                return 256
        return original_sysconf(name)

    os.sysconf = safe_sysconf
    os._openbench_sysconf_patched = True


_ensure_sysconf_accessible()

# Import enhanced logging system
try:
    from openbench.util.Mod_LoggingSystem import setup_logging, get_logging_manager, configure_library_logging
    _HAS_ENHANCED_LOGGING = True
except ImportError:
    _HAS_ENHANCED_LOGGING = False

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'


def print_welcome_message():
    """Print a cross-platform compatible welcome message."""
    colors = get_platform_colors()
    width = 80
    
    print("\n")
    print("=" * width)
    
    # ASCII art with platform-aware colors
    print(f"""{colors['cyan']}
   ____                   ____                  _     
  / __ \\                 |  _ \\                | |    
 | |  | |_ __   ___ _ __ | |_) | ___ _ __   ___| |__  
 | |  | | '_ \\ / _ \\ '_ \\|  _ < / _ \\ '_ \\ / __| '_ \\ 
 | |__| | |_) |  __/ | | | |_) |  __/ | | | (__| | | |
  \\____/| .__/ \\___|_| |_|____/ \\___|_| |_|\\___|_| |_|
        | |                                           
        |_|                                           {colors['reset']}
    """)
    
    print(f"{colors['green']}" + "=" * width + f"{colors['reset']}")
    print(f"{colors['yellow']}{colors['bold']}Welcome to OpenBench: The Open Land Surface Model Benchmark Evaluation System!{colors['reset']}")
    print(f"{colors['green']}" + "=" * width + f"{colors['reset']}")
    
    # System information
    system_info = f"Running on {platform.system()} {platform.release()}"
    print(f"\n{colors['blue']}{colors['bold']}System Info:{colors['reset']} {system_info}")
    
    print(f"\n{colors['bold']}This system evaluates various land surface model outputs against reference data.{colors['reset']}")
    
    print(f"\n{colors['blue']}{colors['bold']}Key Features:{colors['reset']}")
    features = [
        "🌍 Multi-model support",
        "📊 Comprehensive variable evaluation", 
        "📈 Advanced metrics and scoring",
        "⚙️ Customizable benchmarking",
        "🚀 Enhanced parallel processing",
        "💾 Intelligent caching system"
    ]
    
    for feature in features:
        # Use simple bullets for non-Unicode terminals
        if not colors['unicode_support']:
            feature = feature.replace('🌍', '*').replace('📊', '*').replace('📈', '*').replace('⚙️', '*').replace('🚀', '*').replace('💾', '*')
        print(f"  {feature}")
    
    print(f"\n{colors['green']}" + "=" * width + f"{colors['reset']}")
    print(f"{colors['magenta']}{colors['bold']}Initializing OpenBench Evaluation System...{colors['reset']}")
    print(f"{colors['green']}" + "=" * width + f"{colors['reset']}")
    print("")


def setup_directories(main_nl):
    """Create necessary directories for the evaluation process."""
    import os, sys, io, logging
    from datetime import datetime

    base_path = os.path.join(main_nl['general']["basedir"], main_nl['general']['basename'])
    directories = {
        'scratch': os.path.join(base_path, 'scratch'),
        'metrics': os.path.join(base_path, 'metrics'),
        'scores': os.path.join(base_path, 'scores'),
        'data': os.path.join(base_path, 'data'),
        'log': os.path.join(base_path, 'log')
    }
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)

    # ---- Make stdout/stderr UTF-8 tolerant BEFORE any logging config
    try:
        enc = (getattr(sys.stdout, "encoding", "") or "").lower()
        if hasattr(sys.stdout, "buffer") and "utf-8" not in enc:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
        enc = (getattr(sys.stderr, "encoding", "") or "").lower()
        if hasattr(sys.stderr, "buffer") and "utf-8" not in enc:
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
    except Exception:
        pass

    # Configure logging
    log_file = os.path.join(directories['log'], f'openbench_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    print(f'OpenBench Log File: {log_file}')

    # Simple logging configuration for reliability
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Console handler (only show warnings and above)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)

    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)

    # Log initial message to verify logging is working
    logging.info("OpenBench logging initialized successfully")

    return directories

def load_namelists(nl, main_nl):
    """Load reference, simulation, and statistics namelists."""
    ref_nml = nl.read_namelist(main_nl["general"]["reference_nml"])
    sim_nml = nl.read_namelist(main_nl["general"]["simulation_nml"])
    try:
        stats_nml = nl.read_namelist(main_nl["general"]["statistics_nml"])
    except (KeyError, FileNotFoundError, IOError) as e:
        logging.debug(f"Statistics namelist not available: {e}")
        stats_nml = None
    fig_nml = nl.read_namelist(main_nl["general"]["figure_nml"])
    return ref_nml, sim_nml, stats_nml, fig_nml


def print_phase_header(phase_name, icon=""):
    """Print a cross-platform compatible phase header."""
    colors = get_platform_colors()
    width = 60

    # Use simple text for non-Unicode terminals
    if not colors['unicode_support']:
        icon = icon.replace('🔍', '[EVAL]').replace('📈', '[COMP]').replace('📊', '[STAT]')
    
    print(f"\n{colors['green']}" + "=" * width + f"{colors['reset']}")
    print(f"{colors['bold']}{colors['cyan']}{icon} {phase_name.upper()}{colors['reset']}")
    print(f"{colors['green']}" + "=" * width + f"{colors['reset']}")

def print_item_progress(item_name, icon="📊", ref_dataset=None, sim_dataset=None):
    """Print progress for individual items with dataset information."""
    colors = get_platform_colors()

    # Use simple text for non-Unicode terminals
    if not colors['unicode_support']:
        icon = icon.replace('📊', '*').replace('📋', '*').replace('📈', '*')
    
    print(f"  {icon} {colors['blue']}Processing {item_name}...{colors['reset']}")
    
    # Print dataset information if provided
    if ref_dataset:
        print(f"    Reference dataset: {ref_dataset}")
    if sim_dataset:
        print(f"    Simulation dataset: {sim_dataset}")

def format_dataset_info(source_name, data_config, evaluation_item):
    """Format dataset information showing name and type (station/grid)."""
    data_type = data_config[evaluation_item].get(f'{source_name}_data_type', 'unknown')
    data_type_display = 'station' if data_type == 'stn' else 'grid' if data_type == 'grid' else data_type
    return f"{source_name} ({data_type_display})"

# Global set to track displayed dataset combinations
displayed_combinations = set()

def run_evaluation(main_nl, sim_nml, ref_nml, evaluation_items, metric_vars, score_vars, comparison_vars, statistic_vars,
                   fig_nml):
    """Run the evaluation process for each item."""
    print_phase_header("EVALUATION PHASE", "🔍")

    for evaluation_item in evaluation_items:
        print_item_progress(evaluation_item, "📊")
        logging.info(f"Start running {evaluation_item} evaluation...")

        # Note: scratch directory reset is now disabled because each ref/sim combination
        # has its own subdirectory (e.g., scratch/GSIM_colm-routing/). Filters create their
        # own subdirs as needed, so resetting the parent would delete data from other combos.
        # scratch_dir = os.path.join(main_nl['general']["basedir"], main_nl['general']['basename'], 'scratch')
        # gc.collect()
        # time.sleep(0.5)
        # try:
        #     reset_directory(scratch_dir, timeout=60.0, max_retries=5, retry_delay=2.0)
        #     logging.info(f"Scratch directory reset for {evaluation_item}")
        # except Exception as e:
        #     logging.warning(f"Scratch directory reset encountered issues: {e}")

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

        # Clean up memory after data preprocessing
        gc.collect()
        logging.debug(f"Memory cleaned after preprocessing {evaluation_item}")

        for ref_source in ref_sources:
            onetimeref = True
            for sim_source in sim_sources:
                # Display dataset information for each combination
                combination_key = f"{evaluation_item}_{ref_source}_{sim_source}"
                if combination_key not in displayed_combinations:
                    ref_dataset_info = format_dataset_info(ref_source, ref_nml, evaluation_item)
                    sim_dataset_info = format_dataset_info(sim_source, sim_nml, evaluation_item)
                    # Get colors for data type highlighting
                    colors = get_platform_colors()
                    
                    ref_data_type = ref_nml[evaluation_item].get(f'{ref_source}_data_type')
                    sim_data_type = sim_nml[evaluation_item].get(f'{sim_source}_data_type')
                    
                    ref_type = "Station" if ref_data_type == 'stn' else "Grid"
                    sim_type = "Station" if sim_data_type == 'stn' else "Grid"
                    
                    # Color code: Station = Orange/Yellow, Grid = Cyan/Blue
                    ref_color = colors['yellow'] if ref_data_type == 'stn' else colors['cyan']
                    sim_color = colors['yellow'] if sim_data_type == 'stn' else colors['cyan']
                    
                    print(f"    Reference: {ref_source} ({ref_color}{ref_type}{colors['reset']})")
                    print(f"    Simulation: {sim_source} ({sim_color}{sim_type}{colors['reset']})")
                    displayed_combinations.add(combination_key)
                
                process_evaluation(onetimeref, main_nl, sim_nml, ref_nml, metric_vars, score_vars, comparison_vars,
                                statistic_vars, evaluation_item, sim_source, ref_source, fig_nml)
                onetimeref = False

                # Clean up memory after each evaluation
                gc.collect()

        # Clean up memory after completing all evaluations for this item
        gc.collect()
        logging.info(f"Completed evaluation for {evaluation_item}")

    main_nl['general']['IGBP_groupby'] = main_nl['general'].get('IGBP_groupby', True)
    main_nl['general']['PFT_groupby'] = main_nl['general'].get('PFT_groupby', True)
    if main_nl['general']['IGBP_groupby']:
        if main_nl['general']['only_drawing']:
            LC = LC_groupby_only_drawing(main_nl, score_vars, metric_vars)
        else:
            from openbench.core.comparison.Mod_Landcover_Groupby import LC_groupby
            LC = LC_groupby(main_nl, score_vars, metric_vars)
        basedir = os.path.join(main_nl['general']['basedir'], main_nl['general']['basename'])
        LC.scenarios_IGBP_groupby_comparison(basedir, sim_nml, ref_nml, evaluation_items, score_vars, metric_vars,
                                             fig_nml['IGBP_groupby'])
    if main_nl['general']['PFT_groupby']:
        if main_nl['general']['only_drawing']:
            LC = LC_groupby_only_drawing(main_nl, score_vars, metric_vars)
        else:
            from openbench.core.comparison.Mod_Landcover_Groupby import LC_groupby
            LC = LC_groupby(main_nl, score_vars, metric_vars)
        basedir = os.path.join(main_nl['general']['basedir'], main_nl['general']['basename'])
        LC.scenarios_PFT_groupby_comparison(basedir, sim_nml, ref_nml, evaluation_items, score_vars, metric_vars,
                                            fig_nml['PFT_groupby'])
    if main_nl['general']['Climate_zone_groupby']:
        if main_nl['general']['only_drawing']:
            CZ = CZ_groupby_only_drawing(main_nl, score_vars, metric_vars)
        else:
            from openbench.core.comparison.Mod_ClimateZone_Groupby import CZ_groupby
            CZ = CZ_groupby(main_nl, score_vars, metric_vars)
        basedir = os.path.join(main_nl['general']['basedir'], main_nl['general']['basename'])
        CZ.scenarios_CZ_groupby_comparison(basedir, sim_nml, ref_nml, evaluation_items, score_vars, metric_vars,
                                            fig_nml['Climate_zone_groupby'])

    # Clean up all groupby objects to release file handles
    if 'LC' in locals():
        del LC
    if 'CZ' in locals():
        del CZ

    # Force garbage collection to ensure all file handles are released
    gc.collect()
    logging.debug("Groupby objects cleaned up")

def process_mask(onetimeref, main_nl, sim_nml, ref_nml, metric_vars, score_vars, comparison_vars, statistic_vars, evaluation_item,
                 sim_source, ref_source, fig_nml):
    try:
        # Note: scratch directory is now reset once per evaluation_item in run_evaluation()
        # to prevent deleting files needed by parallel processes
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
        if onetimeref:
            dataset_processer.process('ref')
        else:
            logging.debug("Skip processing ref data")
        dataset_processer.process('sim')
        if main_nl['general']['unified_mask']:
            if general_info['ref_data_type'] == 'stn' or general_info['sim_data_type'] == 'stn':
                pass
            else:
                # Mask the observation data with simulation data to ensure consistent coverage
                logging.debug("Mask the observation data with all simulation datasets to ensure consistent coverage")

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

                ref_file_path = f'{general_info["casedir"]}/data/{evaluation_item}_ref_{ref_source}_{general_info["ref_varname"]}.nc'
                # Convert to absolute path to ensure consistency
                ref_file_path_abs = os.path.abspath(ref_file_path)

                ref_ds = None
                sim_ds = None
                try:
                    # Open datasets with chunking for memory efficiency
                    try:
                        ref_ds = open_dataset_chunked(ref_file_path_abs)
                        o = ref_ds[f'{general_info["ref_varname"]}']
                    except FileNotFoundError as e:
                        logging.debug(f"Ref data file not found, processing now: {ref_file_path_abs}")
                        logging.debug(f"Processing ref data")
                        dataset_processer.process('ref')

                        # Try again after processing
                        ref_ds = open_dataset_chunked(ref_file_path_abs)
                        o = ref_ds[f'{general_info["ref_varname"]}']
                    o = Convert_Type.convert_nc(o)

                    sim_file_path = f'{general_info["casedir"]}/data/{evaluation_item}_sim_{sim_source}_{general_info["sim_varname"]}.nc'
                    # Convert to absolute path to ensure consistency
                    sim_file_path_abs = os.path.abspath(sim_file_path)

                    try:
                        sim_ds = open_dataset_chunked(sim_file_path_abs)
                        s = sim_ds[f'{general_info["sim_varname"]}']
                    except FileNotFoundError as e:
                        logging.debug(f"Sim data file not found, processing now: {sim_file_path_abs}")
                        logging.debug(f"Processing sim data")
                        dataset_processer.process('sim')

                        # Try again after processing
                        sim_ds = open_dataset_chunked(sim_file_path_abs)
                        s = sim_ds[f'{general_info["sim_varname"]}']
                    s = Convert_Type.convert_nc(s)

                    # Align time dimension if compatible
                    if len(s['time']) == len(o['time']):
                        s['time'] = o['time']
                    else:
                        logging.warning(f"Time dimension mismatch: sim has {len(s['time'])} points, ref has {len(o['time'])} points")

                    # Create mask and apply it
                    mask1 = np.isnan(s) | np.isnan(o)
                    o.values[mask1] = np.nan

                    # Load data into memory before closing datasets
                    o_data = o.load()

                    # Close parent datasets first
                    if ref_ds is not None:
                        ref_ds.close()
                        ref_ds = None
                    if sim_ds is not None:
                        sim_ds.close()
                        sim_ds = None

                    # Now save the modified data
                    o_data.to_netcdf(ref_file_path_abs)

                    # Clean up variables
                    del o, s, mask1, o_data

                finally:
                    # Ensure datasets are closed even if there's an error
                    if ref_ds is not None:
                        try:
                            ref_ds.close()
                        except Exception:
                            pass
                    if sim_ds is not None:
                        try:
                            sim_ds.close()
                        except Exception:
                            pass

    finally:
        # Ensure cleanup even if there's an error
        gc.collect()


def process_evaluation(onetimeref, main_nl, sim_nml, ref_nml, metric_vars, score_vars, comparison_vars, statistic_vars,
                       evaluation_item, sim_source, ref_source, fig_nml):
    """Process a single evaluation for given sources."""
    # Start heartbeat monitor for this evaluation
    monitor_name = f"Evaluating {evaluation_item} ({ref_source} vs {sim_source})"

    with HeartbeatMonitor(monitor_name, heartbeat_interval=30):
        try:
            general_info_object = GeneralInfoReader(main_nl, sim_nml, ref_nml, metric_vars, score_vars, comparison_vars, statistic_vars,
                                                    evaluation_item, sim_source, ref_source)
            general_info = general_info_object.to_dict()

            # Run Evaluation
            if general_info['ref_data_type'] == 'stn' or general_info['sim_data_type'] == 'stn':
                if main_nl['general']['only_drawing']:
                    evaluater = Evaluation_stn_only_drawing(general_info, fig_nml)
                else:
                    from openbench.core.evaluation.Mod_Evaluation import Evaluation_stn
                    evaluater = Evaluation_stn(general_info, fig_nml)
                evaluater.make_evaluation_P()

            else:
                if main_nl['general']['only_drawing']:
                    evaluater = Evaluation_grid_only_drawing(general_info, fig_nml)
                else:
                    from openbench.core.evaluation.Mod_Evaluation import Evaluation_grid
                    evaluater = Evaluation_grid(general_info, fig_nml)
                evaluater.make_Evaluation()

        finally:
            # Clean up evaluater object and collect garbage
            if 'evaluater' in locals():
                del evaluater
            gc.collect()


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
        from openbench.core.comparison.Mod_Comparison import ComparisonProcessing
        ch = ComparisonProcessing(main_nl, score_vars, metric_vars)

    print_phase_header("COMPARISON PHASE", "📈")

    for cvar in comparison_vars:
        print_item_progress(f"{cvar} comparison", "📋")
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

        # Clean up memory after each comparison
        gc.collect()
        logging.debug(f"Memory cleaned after {cvar} comparison")

    # Final cleanup after all comparisons
    gc.collect()
    logging.debug("Memory cleaned after all comparisons")


def run_statistics(main_nl, stats_nml, statistic_vars, fig_nml):
    """Run statistical analysis for each statistic variable."""
    if not statistic_vars:
        return

    # Import here to avoid circular imports
    from openbench.core.statistic.Mod_Statistics import StatisticsProcessing
    
    logging.info("Running statistical analysis...")
    basedir = os.path.join(main_nl['general']['basedir'], main_nl['general']['basename'])
    stats_handler = StatisticsProcessing(
        main_nl, stats_nml,
        os.path.join(basedir, 'statistics'),
        num_cores=main_nl['general']['num_cores']
    )

    print_phase_header("STATISTICS PHASE", "📊")

    for statistic in statistic_vars:
        print_item_progress(f"{statistic} analysis", "📈")
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

        # Clean up memory after each statistic
        gc.collect()
        logging.debug(f"Memory cleaned after {statistic} analysis")

    logging.info("Statistical analysis completed.")

    # Final cleanup after all statistics
    gc.collect()
    logging.debug("Memory cleaned after all statistics")


def main():
    """Main function to orchestrate the evaluation process."""
    # Clean up Python cache files before starting
    colors = get_platform_colors()
    cache_icon = "🧹" if colors['unicode_support'] else"[CACHE]"
    print(f"\n{cache_icon} {colors['cyan']}Cleaning up Python cache files...{colors['reset']}")

    # Clean __pycache__ directories (silent mode before logging setup)
    try:
        stats = cleanup_all_cache(verbose=False)
        if stats['pycache_dirs_removed'] > 0 or stats['total_files_removed'] > 0:
            print(f"  ✓ Removed {stats['pycache_dirs_removed']} __pycache__ directories")
            print(f"  ✓ Removed {stats['total_files_removed']} cached files")
        else:
            print(f"  ✓ No cache files found (already clean)")
    except Exception as e:
        print(f"  ⚠ Cache cleanup warning: {e}")

    # Clean up memory garbage before starting (silent mode before logging setup)
    cleanup_memory(verbose=False)

    print_welcome_message()

    # Display initial memory cleanup
    clean_icon = "🧹" if colors['unicode_support'] else"[CLEAN]"
    print(f"\n{clean_icon} {colors['cyan']}Performing initial memory cleanup...{colors['reset']}")

    # Perform early validation: command line args, config file existence, external configs
    # This consolidates all early-stage checks and provides clear error messages
    config_file = perform_early_validation()

    # Initialize NamelistReader and load main namelist
    nl = NamelistReader()
    main_nl = nl.read_namelist(config_file)

    # Set default values for optional general config keys
    # This ensures missing keys default to False rather than causing KeyError
    general_defaults = {
        'evaluation': False,
        'comparison': False,
        'statistics': False,
        'only_drawing': False,  # Default to running full evaluation, not just plotting
        'unified_mask': False,
        'IGBP_groupby': False,
        'PFT_groupby': False,
        'Climate_zone_groupby': False,
    }
    for key, default_value in general_defaults.items():
        if key not in main_nl['general']:
            main_nl['general'][key] = default_value

    # Ensure optional top-level sections exist
    section_defaults = ['evaluation_items', 'metrics', 'scores', 'comparisons', 'statistics']
    for section in section_defaults:
        if section not in main_nl:
            main_nl[section] = {}

    # Clean up old outputs before running
    # clean_level options: 'tmp' (default), 'all', 'none'
    clean_level = main_nl['general'].get('clean_level', 'tmp')
    cleanup_old_outputs(main_nl, clean_level=clean_level)

    # Setup directories
    setup_directories(main_nl)

    # Initialize memory management and perform detailed cleanup
    colors = get_platform_colors()
    mem_icon = "💾" if colors['unicode_support'] else"[MEM]"
    print(f"\n{mem_icon} {colors['cyan']}Initializing memory management...{colors['reset']}")
    initialize_memory_management()
    cleanup_memory()  # This will log detailed cleanup info to log file

    rocket = "🚀" if colors['unicode_support'] else">>>"
    print(f"{rocket} {colors['bold']}{colors['green']}Starting OpenBench evaluation process...{colors['reset']}")
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

    # Update namelists - this merges external configs (def_nml) into main namelists
    UpdateNamelist(main_nl, sim_nml, ref_nml, evaluation_items)
    UpdateFigNamelist(main_nl, fig_nml, comparison_vars, statistic_vars)

    # Run comprehensive pre-validation AFTER UpdateNamelist
    # At this point, all external configs have been merged, so we can validate the final configuration
    colors = get_platform_colors()
    validate_icon = "🔍" if colors['unicode_support'] else"[VALIDATE]"
    print(f"\n{validate_icon} {colors['bold']}{colors['cyan']}Running comprehensive pre-validation...{colors['reset']}")
    logging.info("=" * 80)
    logging.info("Starting comprehensive pre-validation (after UpdateNamelist)")
    logging.info("=" * 80)

    try:
        # Run pre-validation with all checks
        # Pre-validation runs AFTER UpdateNamelist to validate the merged configuration
        # - Path existence validation
        # - Namelist completeness checking
        # - Unit validation (units now directly in nml[item][f'{source}_varunit'])
        # - Dimension compatibility checking
        skip_data_check = main_nl['general'].get('skip_data_precheck', False)
        validation_success = run_pre_validation(
            main_nl, sim_nml, ref_nml, list(evaluation_items),
            skip_data_check=skip_data_check
        )

        if not validation_success:
            error_icon = "❌" if colors['unicode_support'] else"[ERROR]"
            print(f"\n{error_icon} {colors['red']}{colors['bold']}Pre-validation failed. Please fix the errors above.{colors['reset']}")
            sys.exit(1)

        check_icon = "✅" if colors['unicode_support'] else"[OK]"
        print(f"{check_icon} {colors['bold']}{colors['green']}Pre-validation completed successfully{colors['reset']}")
        logging.info("Pre-validation completed successfully")

    except ValidationError as e:
        error_icon = "❌" if colors['unicode_support'] else"[ERROR]"
        print(f"\n{error_icon} {colors['red']}{colors['bold']}Validation Error: {str(e)}{colors['reset']}")
        logging.error(f"Validation failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        error_icon = "❌" if colors['unicode_support'] else"[ERROR]"
        print(f"\n{error_icon} {colors['red']}{colors['bold']}Unexpected validation error: {str(e)}{colors['reset']}")
        logging.error(f"Unexpected validation error: {str(e)}", exc_info=True)
        sys.exit(1)

    # Print configuration summary after validation
    print_config_summary(main_nl, sim_nml, ref_nml, evaluation_items, metric_vars, score_vars, comparison_vars, statistic_vars)

    if main_nl['general']['statistics'] and not main_nl['general']['evaluation'] and not main_nl['general']['comparison']:
        pass
    else:
        run_files_check(main_nl, sim_nml, ref_nml, evaluation_items, metric_vars, score_vars, comparison_vars, statistic_vars,
                        fig_nml)

    main_nl['general']['only_drawing'] = main_nl['general'].get('only_drawing', False)

    # Run evaluation if enabled
    if main_nl['general']['evaluation']:
        start_time = time.time()
        run_evaluation(main_nl, sim_nml, ref_nml, evaluation_items, metric_vars, score_vars, comparison_vars, statistic_vars,
                       fig_nml['Validation'])
        end_time = time.time()
        evaluation_time = (end_time - start_time) / 60
        colors = get_platform_colors()
        check = "✅" if colors['unicode_support'] else"[OK]"
        print(f"\n{check} {colors['bold']}{colors['green']}Evaluation process completed in {evaluation_time:.2f} minutes.{colors['reset']}")
        print(f"{colors['green']}" + "=" * 60 + f"{colors['reset']}")
        logging.info(f"Evaluation process completed in {evaluation_time:.2f} minutes.")
        # Clean up memory after evaluation
        cleanup_memory()

    # Run comparison if enabled
    if main_nl['general']['comparison']:
        colors = get_platform_colors()
        chart = "📈" if colors['unicode_support'] else">>>"
        print(f"{chart} {colors['bold']}{colors['blue']}Running comparison process...{colors['reset']}")
        start_time = time.time()
        run_comparison(main_nl, sim_nml, ref_nml, evaluation_items, score_vars, metric_vars, comparison_vars,
                       fig_nml['Comparison'])
        end_time = time.time()
        comparison_time = (end_time - start_time) / 60
        colors = get_platform_colors()
        check = "✅" if colors['unicode_support'] else"[OK]"
        print(f"\n{check} {colors['bold']}{colors['green']}Comparison process completed in {comparison_time:.2f} minutes.{colors['reset']}")
        print(f"{colors['green']}" + "=" * 60 + f"{colors['reset']}")
        logging.info(f"Comparison process completed in {comparison_time:.2f} minutes.")
        # Clean up memory after comparison
        cleanup_memory()

    # Run statistics if enabled
    if main_nl['general']['statistics']:
        colors = get_platform_colors()
        chart = "📊" if colors['unicode_support'] else">>>"
        print(f"{chart} {colors['bold']}{colors['blue']}Running statistics process...{colors['reset']}")
        start_time = time.time()
        run_statistics(main_nl, stats_nml, statistic_vars, fig_nml['Statistic'])
        end_time = time.time()
        statistic_time = (end_time - start_time) / 60
        colors = get_platform_colors()
        check = "✅" if colors['unicode_support'] else"[OK]"
        print(f"\n{check} {colors['bold']}{colors['green']}Statistics process completed in {statistic_time:.2f} minutes.{colors['reset']}")
        print(f"{colors['green']}" + "=" * 60 + f"{colors['reset']}")
        logging.info(f"Statistics process completed in {statistic_time:.2f} minutes.")
        # Clean up memory after statistics
        cleanup_memory()

    # Generate comprehensive report if enabled
    if main_nl['general'].get('generate_report', True):
        colors = get_platform_colors()
        report_icon = "📝" if colors['unicode_support'] else">>>"
        print(f"\n{report_icon} {colors['bold']}{colors['blue']}Generating comprehensive evaluation report...{colors['reset']}")
        start_time = time.time()
        
        try:
            # Prepare configuration for report generator
            report_config = {
                'config_file': sys.argv[1],
                'evaluation_items': list(evaluation_items),
                'metrics': dict(main_nl.get('metrics', {})),
                'scores': dict(main_nl.get('scores', {})),
                'comparisons': dict(main_nl.get('comparisons', {})),
                'statistics': dict(main_nl.get('statistics', {})),
                'general': dict(main_nl.get('general', {})),
                'ref_nml': dict(ref_nml) if ref_nml else {},
                'sim_nml': dict(sim_nml) if sim_nml else {}
            }
            
            # Initialize report generator
            basedir = os.path.join(main_nl['general']['basedir'], main_nl['general']['basename'])
            output_basedir = basedir
            report_gen = ReportGenerator(report_config, output_basedir)
            
            # Generate reports
            report_paths = report_gen.generate_report()
            
            end_time = time.time()
            report_time = (end_time - start_time) / 60
            
            check = "✅" if colors['unicode_support'] else"[OK]"
            print(f"\n{check} {colors['bold']}{colors['green']}Report generation completed in {report_time:.2f} minutes.{colors['reset']}")
            if report_paths.get('html'):
                print(f"   📄 HTML Report: {report_paths['html']}")
            if report_paths.get('pdf'):
                print(f"   📄 PDF Report: {report_paths['pdf']}")
            
            logging.info(f"Report generation completed in {report_time:.2f} minutes.")
            
        except Exception as e:
            logging.error(f"Error generating report: {e}")
            logging.warning("Continuing without report generation...")
    
    # Final memory cleanup
    cleanup_memory()

    # Clean up scratch directory after evaluation is complete
    scratch_dir = os.path.join(main_nl['general']["basedir"], main_nl['general']['basename'], 'scratch')
    if os.path.exists(scratch_dir):
        try:
            shutil.rmtree(scratch_dir)
            logging.info(f"Cleaned up scratch directory: {scratch_dir}")
        except Exception as e:
            logging.warning(f"Failed to clean up scratch directory: {e}")

    colors = get_platform_colors()
    party = "🎉" if colors['unicode_support'] else"[SUCCESS]"
    print(f"\n{colors['green']}" + "=" * 60 + f"{colors['reset']}")
    print(f"{party} {colors['bold']}{colors['green']}OpenBench evaluation process completed successfully!{colors['reset']}")
    print(f"{colors['green']}" + "=" * 60 + f"{colors['reset']}")
    logging.info("OpenBench evaluation process completed successfully.")


if __name__ == '__main__':
    main()
