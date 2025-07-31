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
import platform
import xarray as xr
import numpy as np
import gc

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
from openbench.Mod_ReportGenerator import ReportGenerator
import logging
from datetime import datetime

# Import enhanced logging system
try:
    from openbench.Mod_LoggingSystem import setup_logging, get_logging_manager, configure_library_logging
    _HAS_ENHANCED_LOGGING = True
except ImportError:
    _HAS_ENHANCED_LOGGING = False

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'


def cleanup_memory():
    """Comprehensive memory cleanup function."""
    try:
        # Get memory info before cleanup
        if _HAS_PSUTIL:
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
        else:
            memory_before = 0  # Fallback when psutil is not available
        
        # Clear numpy and xarray caches
        try:
            # Use a more robust approach that avoids accessing deprecated internals
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                
                # Try new numpy structure first (numpy >= 1.20)
                try:
                    if hasattr(np, '_core') and hasattr(np._core, '_internal'):
                        if hasattr(np._core._internal, 'clear_cache'):
                            np._core._internal.clear_cache()
                        elif hasattr(np._core._internal, '_clear_cache'):
                            np._core._internal._clear_cache()
                    else:
                        # Fallback to older numpy structure
                        if hasattr(np, 'core') and hasattr(np.core, '_internal'):
                            if hasattr(np.core._internal, 'clear_cache'):
                                np.core._internal.clear_cache()
                except (AttributeError, TypeError):
                    # No cache clearing available in this numpy version
                    pass
        except (AttributeError, ImportError, Exception):
            # Skip cache clearing if not available or fails
            pass
        
        # Clear xarray global options and caches
        if hasattr(xr, 'set_options'):
            # Reset xarray options to defaults
            xr.set_options(keep_attrs=False)
        
        # Force garbage collection multiple times
        for _ in range(3):
            collected = gc.collect()
            
        # Force memory defragmentation if available
        if hasattr(gc, 'set_threshold'):
            # Temporarily lower GC thresholds to be more aggressive
            old_thresholds = gc.get_threshold()
            gc.set_threshold(10, 5, 5)
            gc.collect()
            # Restore original thresholds
            gc.set_threshold(*old_thresholds)
        
        # Clear any cached modules if safe to do so
        import sys
        modules_to_clear = []
        for module_name in list(sys.modules.keys()):
            if (module_name.startswith('matplotlib.') or 
                module_name.startswith('scipy.') or
                module_name.startswith('pandas.')):
                # Don't actually remove, just note for potential cleanup
                pass
        
        # Get memory info after cleanup
        if _HAS_PSUTIL:
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_freed = memory_before - memory_after
            
            logging.info(f"Memory cleanup completed:")
            logging.info(f"  - Memory before: {memory_before:.1f} MB")
            logging.info(f"  - Memory after: {memory_after:.1f} MB")
            if memory_freed > 0:
                logging.info(f"  - Memory freed: {memory_freed:.1f} MB")
            else:
                logging.info(f"  - Memory usage: {abs(memory_freed):.1f} MB (may have increased due to logging)")
        else:
            logging.info(f"Memory cleanup completed (psutil not available for detailed monitoring)")
            logging.info(f"  - Garbage collection performed")
            logging.info(f"  - Cache clearing attempted")
        
    except Exception as e:
        logging.warning(f"Memory cleanup encountered an issue: {e}")
        # Still perform basic garbage collection
        gc.collect()


def initialize_memory_management():
    """Initialize memory management settings for optimal performance."""
    try:
        # Configure garbage collection for better memory management
        gc.set_threshold(700, 10, 10)  # More aggressive collection
        
        # Enable garbage collection debugging if needed (disable in production)
        # gc.set_debug(gc.DEBUG_STATS)
        
        # Configure numpy for memory efficiency
        if hasattr(np, 'seterr'):
            np.seterr(all='ignore')  # Ignore numpy warnings to reduce memory overhead
        
        # Configure xarray for memory efficiency  
        if hasattr(xr, 'set_options'):
            xr.set_options(
                keep_attrs=False,  # Don't keep attributes to save memory
                display_style='text',  # Use text display to save memory
            )
        
        logging.info("Memory management initialized with optimized settings")
        
    except Exception as e:
        logging.warning(f"Failed to initialize memory management settings: {e}")


def get_platform_colors():
    """Get color codes based on platform support."""
    system = platform.system().lower()
    
    # Check if terminal supports colors
    supports_color = (
        hasattr(sys.stdout, 'isatty') and sys.stdout.isatty() and
        os.environ.get('TERM', '') != 'dumb' and
        ('COLORTERM' in os.environ or 
         os.environ.get('TERM', '').endswith(('color', '256color', 'truecolor')))
    )
    
    # Windows Command Prompt has limited ANSI support
    if system == 'windows' and not os.environ.get('WT_SESSION'):
        supports_color = False
    
    if supports_color:
        return {
            'reset': '\033[0m',
            'bold': '\033[1m',
            'cyan': '\033[1;36m',
            'green': '\033[1;32m',
            'yellow': '\033[1;33m',
            'blue': '\033[1;34m',
            'magenta': '\033[1;35m',
            'red': '\033[1;31m'
        }
    else:
        # Fallback to no colors for compatibility
        return {
            'reset': '', 'bold': '', 'cyan': '', 'green': '',
            'yellow': '', 'blue': '', 'magenta': '', 'red': ''
        }

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
        "⚙️  Customizable benchmarking",
        "🚀 Enhanced parallel processing",
        "💾 Intelligent caching system"
    ]
    
    for feature in features:
        # Use simple bullets for non-Unicode terminals
        if not colors['reset']:  # No color support likely means limited Unicode
            feature = feature.replace('🌍', '*').replace('📊', '*').replace('📈', '*').replace('⚙️', '*').replace('🚀', '*').replace('💾', '*')
        print(f"  {feature}")
    
    print(f"\n{colors['green']}" + "=" * width + f"{colors['reset']}")
    print(f"{colors['magenta']}{colors['bold']}Initializing OpenBench Evaluation System...{colors['reset']}")
    print(f"{colors['green']}" + "=" * width + f"{colors['reset']}")
    print("")


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
        # Use enhanced logging system with separate console and file levels
        logging_manager = setup_logging(
            level=logging.INFO,  # File level: INFO and above
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
        
        # Set specific log file with INFO level
        logging_manager.add_file_handler(
            filename=f'openbench_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            formatter=logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'),
            level=logging.INFO
        )
        
        # Configure console handler to show only WARNING and above
        console_handler = None
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream.name == '<stdout>':
                console_handler = handler
                break
        
        if console_handler:
            console_handler.setLevel(logging.WARNING)
    else:
        # Fallback to standard logging with separate levels for console and file
        # Clear any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler: INFO and above
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Console handler: WARNING and above
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(formatter)
        
        # Configure root logger
        logging.root.setLevel(logging.INFO)  # Overall level
        logging.root.addHandler(file_handler)
        logging.root.addHandler(console_handler)
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


def print_phase_header(phase_name, icon=""):
    """Print a cross-platform compatible phase header."""
    colors = get_platform_colors()
    width = 60
    
    # Use simple text for non-Unicode terminals
    if not colors['reset']:
        icon = icon.replace('🔍', '[EVAL]').replace('📈', '[COMP]').replace('📊', '[STAT]')
    
    print(f"\n{colors['green']}" + "=" * width + f"{colors['reset']}")
    print(f"{colors['bold']}{colors['cyan']}{icon} {phase_name.upper()}{colors['reset']}")
    print(f"{colors['green']}" + "=" * width + f"{colors['reset']}")

def print_item_progress(item_name, icon="📊", ref_dataset=None, sim_dataset=None):
    """Print progress for individual items with dataset information."""
    colors = get_platform_colors()
    
    # Use simple text for non-Unicode terminals
    if not colors['reset']:
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

    main_nl['general']['IGBP_groupby'] = main_nl['general'].get('IGBP_groupby', 'True')
    main_nl['general']['PFT_groupby'] = main_nl['general'].get('PFT_groupby', 'True')
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

def process_mask(onetimeref, main_nl, sim_nml, ref_nml, metric_vars, score_vars, comparison_vars, statistic_vars, evaluation_item,
                 sim_source, ref_source, fig_nml):
    logging.info(f"Processing {evaluation_item} - ref: {ref_source} - sim: {sim_source}")
    general_info_object = GeneralInfoReader(main_nl, sim_nml, ref_nml, metric_vars, score_vars, comparison_vars, statistic_vars,
                                            evaluation_item, sim_source, ref_source)
    general_info = general_info_object.to_dict()
    # Add ref_source and sim_source to general_info
    general_info['ref_source'] = ref_source
    general_info['sim_source'] = sim_source

    from openbench.data.Mod_DatasetProcessing import DatasetProcessing
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
        os.path.join(basedir, 'output', 'statistics'),
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
    logging.info("Statistical analysis completed.")


def main():
    """Main function to orchestrate the evaluation process."""
    print_welcome_message()

    # Initialize NamelistReader and load main namelist
    nl = NamelistReader()
    main_nl = nl.read_namelist(sys.argv[1])

    # Setup directories
    setup_directories(main_nl)

    # Initialize memory management and perform initial cleanup
    initialize_memory_management()
    cleanup_memory()

    colors = get_platform_colors()
    rocket = "🚀" if colors['reset'] else ">>>"
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
        colors = get_platform_colors()
        check = "✅" if colors['reset'] else "[OK]"
        print(f"\n{check} {colors['bold']}{colors['green']}Evaluation process completed in {evaluation_time:.2f} minutes.{colors['reset']}")
        print(f"{colors['green']}" + "=" * 60 + f"{colors['reset']}")
        logging.info(f"Evaluation process completed in {evaluation_time:.2f} minutes.")
        # Clean up memory after evaluation
        cleanup_memory()

    # Run comparison if enabled
    if main_nl['general']['comparison']:
        colors = get_platform_colors()
        chart = "📈" if colors['reset'] else ">>>"
        print(f"{chart} {colors['bold']}{colors['blue']}Running comparison process...{colors['reset']}")
        start_time = time.time()
        run_comparison(main_nl, sim_nml, ref_nml, evaluation_items, score_vars, metric_vars, comparison_vars,
                       fig_nml['Comparison'])
        end_time = time.time()
        comparison_time = (end_time - start_time) / 60
        colors = get_platform_colors()
        check = "✅" if colors['reset'] else "[OK]"
        print(f"\n{check} {colors['bold']}{colors['green']}Comparison process completed in {comparison_time:.2f} minutes.{colors['reset']}")
        print(f"{colors['green']}" + "=" * 60 + f"{colors['reset']}")
        logging.info(f"Comparison process completed in {comparison_time:.2f} minutes.")
        # Clean up memory after comparison
        cleanup_memory()

    # Run statistics if enabled
    if main_nl['general']['statistics']:
        colors = get_platform_colors()
        chart = "📊" if colors['reset'] else ">>>"
        print(f"{chart} {colors['bold']}{colors['blue']}Running statistics process...{colors['reset']}")
        start_time = time.time()
        run_statistics(main_nl, stats_nml, statistic_vars, fig_nml['Statistic'])
        end_time = time.time()
        statistic_time = (end_time - start_time) / 60
        colors = get_platform_colors()
        check = "✅" if colors['reset'] else "[OK]"
        print(f"\n{check} {colors['bold']}{colors['green']}Statistics process completed in {statistic_time:.2f} minutes.{colors['reset']}")
        print(f"{colors['green']}" + "=" * 60 + f"{colors['reset']}")
        logging.info(f"Statistics process completed in {statistic_time:.2f} minutes.")
        # Clean up memory after statistics
        cleanup_memory()

    # Generate comprehensive report if enabled
    if main_nl['general'].get('generate_report', True):
        colors = get_platform_colors()
        report_icon = "📝" if colors['reset'] else ">>>"
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
                'statistics': dict(main_nl.get('statistics', {}))
            }
            
            # Initialize report generator
            basedir = os.path.join(main_nl['general']['basedir'], main_nl['general']['basename'])
            output_basedir = os.path.join(basedir, "output")
            report_gen = ReportGenerator(report_config, output_basedir)
            
            # Generate reports
            report_paths = report_gen.generate_report()
            
            end_time = time.time()
            report_time = (end_time - start_time) / 60
            
            check = "✅" if colors['reset'] else "[OK]"
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
    
    colors = get_platform_colors()
    party = "🎉" if colors['reset'] else "[SUCCESS]"
    print(f"\n{colors['green']}" + "=" * 60 + f"{colors['reset']}")
    print(f"{party} {colors['bold']}{colors['green']}OpenBench evaluation process completed successfully!{colors['reset']}")
    print(f"{colors['green']}" + "=" * 60 + f"{colors['reset']}")
    logging.info("OpenBench evaluation process completed successfully.")


if __name__ == '__main__':
    main()
