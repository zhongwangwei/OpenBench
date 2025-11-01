"""
Configuration Checking and Validation Utilities

This module provides utilities for validating configuration files before processing begins.
It includes early validation to catch common errors like missing configuration files.

Author: OpenBench Development Team
Date: November 2025
"""

import os
import sys
import platform
import logging
from pathlib import Path
from typing import Dict, List, Set, Any, Optional


def get_platform_colors() -> Dict[str, str]:
    """
    Get color codes based on platform support.

    Returns:
        Dict[str, str]: Dictionary of color codes for terminal output
    """
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


def check_config_files_exist(main_config_path: str, check_external: bool = True) -> Dict[str, Any]:
    """
    Check if all required configuration files exist before processing.

    This function performs early validation to catch missing configuration files
    before any processing begins, providing clear error messages to users.

    Args:
        main_config_path: Path to the main configuration file
        check_external: Whether to check external config files referenced in main config

    Returns:
        Dict containing validation results with keys:
            - 'valid': bool indicating if all checks passed
            - 'errors': list of error messages
            - 'warnings': list of warning messages
            - 'files_checked': list of files that were checked

    Raises:
        FileNotFoundError: If critical configuration files are missing
    """
    colors = get_platform_colors()
    errors = []
    warnings = []
    files_checked = []

    # Check main configuration file
    main_path = Path(main_config_path)
    if not main_path.exists():
        errors.append(f"Main configuration file not found: {main_config_path}")
        return {
            'valid': False,
            'errors': errors,
            'warnings': warnings,
            'files_checked': files_checked
        }

    files_checked.append(str(main_path.resolve()))

    # If requested, check external configuration files referenced in main config
    if check_external:
        try:
            # Import here to avoid circular dependencies
            from openbench.config.manager import ConfigManager

            config_mgr = ConfigManager()
            main_config = config_mgr.load_config(main_config_path, validate=False)

            # Check for external config file references
            external_configs = []
            general = main_config.get('general', {})

            # Common external config keys
            external_config_keys = [
                'reference_nml',
                'simulation_nml',
                'statistics_nml',
                'figlib_nml'
            ]

            for key in external_config_keys:
                if key in general:
                    config_path = general[key]
                    if config_path:  # Only check non-empty paths
                        external_configs.append((key, config_path))

            # Validate each external config file
            for config_name, config_path in external_configs:
                # Convert to absolute path if relative
                # Relative paths in config files are relative to current working directory,
                # not relative to the config file location
                config_path_obj = Path(config_path)

                # If it's not absolute, resolve it relative to current working directory
                if not config_path_obj.is_absolute():
                    config_path_obj = config_path_obj.resolve()

                if not config_path_obj.exists():
                    errors.append(
                        f"{config_name} file not found: {config_path}\n"
                        f"  Referenced in: {main_config_path}"
                    )
                else:
                    files_checked.append(str(config_path_obj))

        except Exception as e:
            warnings.append(f"Could not validate external configs: {str(e)}")

    # Determine overall validity
    valid = len(errors) == 0

    return {
        'valid': valid,
        'errors': errors,
        'warnings': warnings,
        'files_checked': files_checked
    }


def print_config_validation_results(results: Dict[str, Any]) -> None:
    """
    Print configuration validation results in a user-friendly format.

    Args:
        results: Dictionary returned by check_config_files_exist()
    """
    colors = get_platform_colors()

    print(f"\n{colors['cyan']}{'=' * 80}{colors['reset']}")
    print(f"{colors['bold']}{colors['cyan']}🔍 Configuration File Validation{colors['reset']}")
    print(f"{colors['cyan']}{'=' * 80}{colors['reset']}\n")

    # Show files checked
    if results['files_checked']:
        print(f"{colors['bold']}{colors['blue']}Files Checked:{colors['reset']}")
        for file_path in results['files_checked']:
            print(f"  {colors['green']}✓{colors['reset']} {file_path}")
        print()

    # Show warnings
    if results['warnings']:
        print(f"{colors['bold']}{colors['yellow']}Warnings:{colors['reset']}")
        for warning in results['warnings']:
            print(f"  {colors['yellow']}⚠{colors['reset']} {warning}")
        print()

    # Show errors
    if results['errors']:
        print(f"{colors['bold']}{colors['red']}Errors Found:{colors['reset']}")
        for error in results['errors']:
            print(f"  {colors['red']}✗{colors['reset']} {error}")
        print()
        print(f"{colors['cyan']}{'=' * 80}{colors['reset']}\n")
        return False

    # Success message
    print(f"{colors['bold']}{colors['green']}✅ All configuration files exist and are accessible{colors['reset']}")
    print(f"{colors['cyan']}{'=' * 80}{colors['reset']}\n")

    return True


def validate_config_before_processing(main_config_path: str, exit_on_error: bool = True) -> bool:
    """
    Validate configuration files before processing begins.

    This is the main entry point for early configuration validation.
    It checks that all configuration files exist and are accessible.

    Args:
        main_config_path: Path to the main configuration file
        exit_on_error: Whether to exit the program if validation fails

    Returns:
        bool: True if validation passed, False otherwise
    """
    colors = get_platform_colors()

    try:
        # Run validation checks
        results = check_config_files_exist(main_config_path, check_external=True)

        # Print results
        success = print_config_validation_results(results)

        # Handle errors
        if not success:
            if exit_on_error:
                print(f"{colors['red']}❌ Configuration validation failed. Please fix the errors above.{colors['reset']}\n")
                sys.exit(1)
            return False

        return True

    except Exception as e:
        print(f"{colors['red']}❌ Error during configuration validation: {str(e)}{colors['reset']}\n")
        if exit_on_error:
            sys.exit(1)
        return False


def perform_early_validation(config_file_path: Optional[str] = None) -> str:
    """
    Perform early validation of configuration file before any processing.

    This function consolidates all early-stage configuration checks:
    1. Command line argument validation
    2. Main configuration file existence
    3. External configuration files existence

    Args:
        config_file_path: Path to configuration file (from command line args if None)

    Returns:
        str: Path to the validated configuration file

    Raises:
        SystemExit: If validation fails
    """
    colors = get_platform_colors()

    # Step 1: Validate command line arguments
    if config_file_path is None:
        if len(sys.argv) < 2:
            error_icon = "❌" if colors['reset'] else "[ERROR]"
            print(f"\n{error_icon} {colors['red']}{colors['bold']}Error: Configuration file path required{colors['reset']}")
            print(f"\n{colors['bold']}Usage:{colors['reset']} python openbench.py <config_file_path>")
            print(f"\n{colors['bold']}Examples:{colors['reset']}")
            print(f"  python openbench.py nml/nml-json/main-Debug.json")
            print(f"  python openbench.py nml/nml-yaml/main-Debug.yaml")
            print(f"  python openbench.py nml/nml-Fortran/main-Debug.nml")
            sys.exit(1)
        config_file_path = sys.argv[1]

    # Step 2: Check main configuration file exists
    if not os.path.exists(config_file_path):
        error_icon = "❌" if colors['reset'] else "[ERROR]"
        print(f"\n{error_icon} {colors['red']}{colors['bold']}Error: Configuration file not found{colors['reset']}")
        print(f"  File: {config_file_path}")
        sys.exit(1)

    # Step 3: Validate all referenced configuration files exist
    rocket = "🔍" if colors['reset'] else ">>>"
    print(f"\n{rocket} {colors['bold']}{colors['cyan']}Validating configuration files...{colors['reset']}")

    validate_config_before_processing(config_file_path, exit_on_error=True)

    return config_file_path


def print_config_summary(main_nl, sim_nml, ref_nml, evaluation_items, metric_vars, score_vars, comparison_vars, statistic_vars):
    """
    Print a comprehensive summary of the configuration after validation.

    This shows users exactly what will be evaluated before processing starts.

    Args:
        main_nl: Main configuration namelist
        sim_nml: Simulation configuration namelist
        ref_nml: Reference configuration namelist
        evaluation_items: List of variables to evaluate
        metric_vars: Set of metrics to calculate
        score_vars: Set of scoring methods to apply
        comparison_vars: Set of comparison analyses to perform
        statistic_vars: Set of statistical analyses to perform
    """
    colors = get_platform_colors()
    width = 80

    print("\n" + colors['cyan'] + "=" * width + colors['reset'])
    print(f"{colors['bold']}{colors['cyan']}📋 CONFIGURATION SUMMARY{colors['reset']}")
    print(colors['cyan'] + "=" * width + colors['reset'])

    # General settings
    print(f"\n{colors['bold']}{colors['blue']}General Settings:{colors['reset']}")
    print(f"  Output directory: {main_nl['general']['basedir']}/{main_nl['general']['basename']}")
    print(f"  Time period: {main_nl['general']['syear']} - {main_nl['general']['eyear']}")
    print(f"  Spatial domain: Lat[{main_nl['general']['min_lat']}°, {main_nl['general']['max_lat']}°], Lon[{main_nl['general']['min_lon']}°, {main_nl['general']['max_lon']}°]")

    # Enabled modules
    print(f"\n{colors['bold']}{colors['blue']}Enabled Modules:{colors['reset']}")
    modules = []
    if main_nl['general'].get('evaluation', False):
        modules.append(f"{colors['green']}✓{colors['reset']} Evaluation")
    if main_nl['general'].get('comparison', False):
        modules.append(f"{colors['green']}✓{colors['reset']} Comparison")
    if main_nl['general'].get('statistics', False):
        modules.append(f"{colors['green']}✓{colors['reset']} Statistics")
    print(f"  {' | '.join(modules)}")

    # Groupby options
    groupby_opts = []
    if main_nl['general'].get('IGBP_groupby', False):
        groupby_opts.append("IGBP")
    if main_nl['general'].get('PFT_groupby', False):
        groupby_opts.append("PFT")
    if main_nl['general'].get('Climate_zone_groupby', False):
        groupby_opts.append("Climate Zone")
    if groupby_opts:
        print(f"  Group-by analysis: {', '.join(groupby_opts)}")

    # Evaluation items and data sources
    print(f"\n{colors['bold']}{colors['blue']}Evaluation Items ({len(evaluation_items)}):{colors['reset']}")
    for i, item in enumerate(evaluation_items, 1):
        print(f"\n  {colors['bold']}{i}. {item}{colors['reset']}")

        # Reference sources
        ref_sources = ref_nml['general'].get(f'{item}_ref_source', [])
        if isinstance(ref_sources, str):
            ref_sources = [ref_sources]

        print(f"     {colors['yellow']}Reference Data:{colors['reset']}")
        for ref_src in ref_sources:
            if item in ref_nml and f'{ref_src}_data_type' in ref_nml[item]:
                data_type = ref_nml[item][f'{ref_src}_data_type']
                varname = ref_nml[item].get(f'{ref_src}_varname', 'N/A')
                unit = ref_nml[item].get(f'{ref_src}_varunit', 'N/A')
                data_type_icon = "📍" if data_type == 'stn' else "🌐"
                if not colors['reset']:
                    data_type_icon = "[STN]" if data_type == 'stn' else "[GRID]"
                print(f"       {data_type_icon} {ref_src}: {varname} ({unit})")

        # Simulation sources
        sim_sources = sim_nml['general'].get(f'{item}_sim_source', [])
        if isinstance(sim_sources, str):
            sim_sources = [sim_sources]

        print(f"     {colors['cyan']}Simulation Data:{colors['reset']}")
        for sim_src in sim_sources:
            if item in sim_nml and f'{sim_src}_data_type' in sim_nml[item]:
                data_type = sim_nml[item][f'{sim_src}_data_type']
                varname = sim_nml[item].get(f'{sim_src}_varname', 'N/A')
                unit = sim_nml[item].get(f'{sim_src}_varunit', 'N/A')
                data_type_icon = "📍" if data_type == 'stn' else "🌐"
                if not colors['reset']:
                    data_type_icon = "[STN]" if data_type == 'stn' else "[GRID]"
                print(f"       {data_type_icon} {sim_src}: {varname} ({unit})")

    # Metrics
    if metric_vars:
        print(f"\n{colors['bold']}{colors['blue']}Metrics ({len(metric_vars)}):{colors['reset']}")
        metrics_list = ', '.join(metric_vars)
        # Wrap long lines
        if len(metrics_list) > 70:
            metrics = list(metric_vars)
            line = "  "
            for metric in metrics:
                if len(line) + len(metric) + 2 > 78:
                    print(line)
                    line = "  " + metric
                else:
                    line += (", " if line != "  " else "") + metric
            if line != "  ":
                print(line)
        else:
            print(f"  {metrics_list}")

    # Scores
    if score_vars:
        print(f"\n{colors['bold']}{colors['blue']}Scoring Methods ({len(score_vars)}):{colors['reset']}")
        scores_list = ', '.join(score_vars)
        print(f"  {scores_list}")

    # Comparisons
    if comparison_vars and main_nl['general'].get('comparison', False):
        print(f"\n{colors['bold']}{colors['blue']}Comparison Analyses ({len(comparison_vars)}):{colors['reset']}")
        comp_list = ', '.join(comparison_vars)
        # Wrap long lines
        if len(comp_list) > 70:
            comps = list(comparison_vars)
            line = "  "
            for comp in comps:
                if len(line) + len(comp) + 2 > 78:
                    print(line)
                    line = "  " + comp
                else:
                    line += (", " if line != "  " else "") + comp
            if line != "  ":
                print(line)
        else:
            print(f"  {comp_list}")

    # Statistics
    if statistic_vars and main_nl['general'].get('statistics', False):
        print(f"\n{colors['bold']}{colors['blue']}Statistical Analyses ({len(statistic_vars)}):{colors['reset']}")
        stats_list = ', '.join(statistic_vars)
        print(f"  {stats_list}")

    print("\n" + colors['cyan'] + "=" * width + colors['reset'])
    print(f"{colors['bold']}{colors['green']}Ready to start processing...{colors['reset']}")
    print(colors['cyan'] + "=" * width + colors['reset'] + "\n")

    # Also log to file
    logging.info("=" * 80)
    logging.info("CONFIGURATION SUMMARY")
    logging.info("=" * 80)
    logging.info(f"Output: {main_nl['general']['basedir']}/{main_nl['general']['basename']}")
    logging.info(f"Period: {main_nl['general']['syear']}-{main_nl['general']['eyear']}")
    logging.info(f"Evaluation items: {', '.join(evaluation_items)}")
    logging.info(f"Metrics: {', '.join(metric_vars)}")
    logging.info(f"Scores: {', '.join(score_vars)}")
    if comparison_vars:
        logging.info(f"Comparisons: {', '.join(comparison_vars)}")
    if statistic_vars:
        logging.info(f"Statistics: {', '.join(statistic_vars)}")
    logging.info("=" * 80)
