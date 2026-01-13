# -*- coding: utf-8 -*-
"""
Pre-validation Module for OpenBench

This module provides comprehensive validation before starting the evaluation process:
1. Path validation and conversion to absolute paths
2. Namelist validation
3. Data dimension matching validation
4. Unit validation with Lib_Unit.py integration

Author: Zhongwang Wei (zhongwang007@gmail.com)
Version: 1.0
Date: November 2025
"""

import os
import sys
import logging
import xarray as xr
import numpy as np
from typing import Dict, Any, List, Tuple, Union, Optional
from pathlib import Path
import warnings
import re
import importlib.util


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class PreValidator:
    """
    Comprehensive pre-validation system for OpenBench.

    This class performs all necessary validation before starting the evaluation process:
    - Path validation and conversion to absolute paths
    - Namelist schema validation
    - Data dimension matching
    - Unit validation with unknown unit detection
    """

    def __init__(self, main_nl: Dict[str, Any], sim_nml: Dict[str, Any],
                 ref_nml: Dict[str, Any], evaluation_items: List[str]):
        """
        Initialize the PreValidator.

        Args:
            main_nl: Main namelist configuration
            sim_nml: Simulation namelist configuration
            ref_nml: Reference namelist configuration
            evaluation_items: List of evaluation items to process
        """
        self.main_nl = main_nl
        self.sim_nml = sim_nml
        self.ref_nml = ref_nml
        self.evaluation_items = evaluation_items
        self.validation_errors = []
        self.validation_warnings = []
        self.unknown_units = set()

        # Get base directory for resolving relative paths
        self.base_dir = os.getcwd()

    def validate_all(self, skip_data_check: bool = False) -> bool:
        """
        Run all validation checks.

        Note: This runs AFTER namelists are already loaded by NamelistReader.
        We validate the loaded configuration without modifying it.

        Args:
            skip_data_check: If True, skip data dimension and unit checks (only validate paths and namelists)

        Returns:
            bool: True if all validations pass, False otherwise

        Raises:
            ValidationError: If critical validation fails
        """
        logging.info("=" * 80)
        logging.info("🔍 Starting comprehensive pre-validation...")
        logging.info("=" * 80)

        # Step 1: Path validation (check existence, not conversion)
        logging.info("\n📁 Step 1: Validating paths...")
        try:
            self._validate_paths()
            logging.info("✅ Path validation completed successfully")
        except Exception as e:
            error_msg = f"❌ Path validation failed: {str(e)}"
            logging.error(error_msg)
            self.validation_errors.append(error_msg)
            raise ValidationError(error_msg)

        # Step 2: Namelist validation
        logging.info("\n📋 Step 2: Validating namelist configurations...")
        try:
            self._validate_namelists()
            logging.info("✅ Namelist validation completed successfully")
        except Exception as e:
            error_msg = f"❌ Namelist validation failed: {str(e)}"
            logging.error(error_msg)
            self.validation_errors.append(error_msg)
            raise ValidationError(error_msg)

        if not skip_data_check:
            # Step 3: Unit validation
            logging.info("\n📏 Step 3: Validating data units...")
            try:
                self._validate_units()
                if self.unknown_units:
                    self._handle_unknown_units()
                logging.info("✅ Unit validation completed")
            except Exception as e:
                error_msg = f"❌ Unit validation failed: {str(e)}"
                logging.error(error_msg)
                self.validation_errors.append(error_msg)
                raise ValidationError(error_msg)

            # Step 4: Data dimension validation
            logging.info("\n📊 Step 4: Validating data dimensions...")
            try:
                self._validate_dimensions()
                logging.info("✅ Dimension validation completed")
            except Exception as e:
                error_msg = f"❌ Dimension validation failed: {str(e)}"
                logging.error(error_msg)
                self.validation_errors.append(error_msg)
                raise ValidationError(error_msg)

            # Step 5: Variable existence validation
            logging.info("\n🔍 Step 5: Validating variable existence in data files...")
            try:
                self._validate_variable_existence()
                logging.info("✅ Variable existence validation completed")
            except Exception as e:
                error_msg = f"❌ Variable existence validation failed: {str(e)}"
                logging.error(error_msg)
                self.validation_errors.append(error_msg)
                raise ValidationError(error_msg)

        # Summary
        logging.info("\n" + "=" * 80)
        if self.validation_errors:
            logging.error(f"❌ Validation completed with {len(self.validation_errors)} error(s)")
            for error in self.validation_errors:
                logging.error(f"  - {error}")
            return False
        elif self.validation_warnings:
            logging.warning(f"⚠️  Validation completed with {len(self.validation_warnings)} warning(s)")
            for warning in self.validation_warnings:
                logging.warning(f"  - {warning}")
        else:
            logging.info("✅ All validations passed successfully!")
        logging.info("=" * 80)

        return True

    def _validate_paths(self):
        """
        Validate that all required paths exist.

        Note: Paths should already be absolute after NamelistReader processing.
        We only check for existence here, not convert.
        """
        self._check_critical_paths()

    def _check_critical_paths(self):
        """
        Check that all critical paths exist after conversion.
        """
        logging.info("  Checking critical paths...")

        # Check main namelist paths
        critical_files = [
            ('reference_nml', self.main_nl['general'].get('reference_nml')),
            ('simulation_nml', self.main_nl['general'].get('simulation_nml')),
            ('figure_nml', self.main_nl['general'].get('figure_nml')),
        ]

        # Check statistics namelist if enabled
        if self.main_nl['general'].get('statistics', False):
            critical_files.append(('statistics_nml', self.main_nl['general'].get('statistics_nml')))

        # Check base directory
        basedir = self.main_nl['general'].get('basedir')
        if basedir and not os.path.exists(basedir):
            try:
                os.makedirs(basedir, exist_ok=True)
                logging.info(f"  Created base directory: {basedir}")
            except Exception as e:
                raise ValidationError(f"Cannot create base directory {basedir}: {str(e)}")

        # Check critical files
        missing_files = []
        for name, path in critical_files:
            if path is None:
                missing_files.append(f"{name} (not specified)")
            elif not os.path.exists(path):
                missing_files.append(f"{name}: {path}")

        if missing_files:
            error_msg = "Missing critical configuration files:\n" + "\n".join(f"    - {f}" for f in missing_files)
            raise ValidationError(error_msg)

        logging.info("  ✓ All critical paths exist")

    def _validate_def_nml_consistency(self):
        """
        Validate that all source names in general sections have corresponding def_nml entries.

        This catches configuration errors where a source is referenced in
        general.*_ref_source or general.*_sim_source but doesn't have a
        corresponding entry in def_nml.
        """
        logging.info("  Checking def_nml consistency...")

        errors = []

        # Check reference sources
        for item in self.evaluation_items:
            ref_source_key = f'{item}_ref_source'
            if ref_source_key in self.ref_nml.get('general', {}):
                ref_sources = self.ref_nml['general'][ref_source_key]
                ref_sources = [ref_sources] if isinstance(ref_sources, str) else ref_sources

                # Check each ref_source has a def_nml entry
                def_nml = self.ref_nml.get('def_nml', {})
                for source in ref_sources:
                    if source not in def_nml:
                        errors.append(
                            f"Reference source '{source}' for {item} is not defined in ref_nml['def_nml']"
                        )
                    else:
                        # Also check if the file path exists
                        config_file = def_nml[source]
                        if not os.path.exists(config_file):
                            errors.append(
                                f"Configuration file for reference source '{source}' not found: {config_file}"
                            )

        # Check simulation sources
        for item in self.evaluation_items:
            sim_source_key = f'{item}_sim_source'
            if sim_source_key in self.sim_nml.get('general', {}):
                sim_sources = self.sim_nml['general'][sim_source_key]
                sim_sources = [sim_sources] if isinstance(sim_sources, str) else sim_sources

                # Check each sim_source has a def_nml entry
                def_nml = self.sim_nml.get('def_nml', {})
                for source in sim_sources:
                    if source not in def_nml:
                        errors.append(
                            f"Simulation source '{source}' for {item} is not defined in sim_nml['def_nml']"
                        )
                    else:
                        # Also check if the file path exists
                        config_file = def_nml[source]
                        if not os.path.exists(config_file):
                            errors.append(
                                f"Configuration file for simulation source '{source}' not found: {config_file}"
                            )

        if errors:
            error_msg = "def_nml consistency errors:\n" + "\n".join(f"    - {e}" for e in errors)
            raise ValidationError(error_msg)

        logging.info("  ✓ All source names have corresponding def_nml entries")

    def _validate_namelists(self):
        """
        Validate namelist configurations for completeness and correctness.
        """
        # Check main namelist required fields
        required_main_fields = {
            'general': ['basedir', 'basename', 'reference_nml', 'simulation_nml', 'figure_nml'],
        }

        for section, fields in required_main_fields.items():
            if section not in self.main_nl:
                raise ValidationError(f"Missing required section '{section}' in main namelist")
            for field in fields:
                if field not in self.main_nl[section]:
                    raise ValidationError(f"Missing required field '{field}' in main namelist section '{section}'")

        # Validate def_nml consistency for reference and simulation sources
        self._validate_def_nml_consistency()

        # Validate each evaluation item
        for item in self.evaluation_items:
            self._validate_evaluation_item(item)

    def _validate_evaluation_item(self, item: str):
        """
        Validate configuration for a specific evaluation item.

        Args:
            item: Evaluation item name
        """
        logging.info(f"  Validating configuration for: {item}")

        # Check reference sources
        ref_source_key = f'{item}_ref_source'
        if ref_source_key not in self.ref_nml['general']:
            raise ValidationError(f"Missing reference source for {item} in reference namelist")

        ref_sources = self.ref_nml['general'][ref_source_key]
        ref_sources = [ref_sources] if isinstance(ref_sources, str) else ref_sources

        if not ref_sources:
            raise ValidationError(f"No reference sources defined for {item}")

        # Check simulation sources
        sim_source_key = f'{item}_sim_source'
        if sim_source_key not in self.sim_nml['general']:
            raise ValidationError(f"Missing simulation source for {item} in simulation namelist")

        sim_sources = self.sim_nml['general'][sim_source_key]
        sim_sources = [sim_sources] if isinstance(sim_sources, str) else sim_sources

        if not sim_sources:
            raise ValidationError(f"No simulation sources defined for {item}")

        # Validate each source has required configuration
        for ref_source in ref_sources:
            self._validate_source_config(item, ref_source, self.ref_nml, 'reference')

        for sim_source in sim_sources:
            self._validate_source_config(item, sim_source, self.sim_nml, 'simulation')

    def _check_filter_handles_item(self, model: str, item: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a filter file handles a specific evaluation item.

        Args:
            model: Model name (e.g., 'CoLM', 'CLM5')
            item: Evaluation item name (e.g., 'Precipitation')

        Returns:
            Tuple of (handles_item: bool, filter_path: Optional[str])
        """
        # Construct filter file path
        filter_filename = f"{model}_filter.py"
        filter_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data', 'custom', filter_filename
        )

        if not os.path.exists(filter_path):
            return False, None

        try:
            # Read filter file content
            with open(filter_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check if the filter handles this item
            # Look for patterns like: if info.item == "Precipitation":
            patterns = [
                rf'if\s+info\.item\s*==\s*["\']({re.escape(item)})["\']',
                rf'elif\s+info\.item\s*==\s*["\']({re.escape(item)})["\']',
            ]

            for pattern in patterns:
                if re.search(pattern, content):
                    return True, filter_path

            return False, None

        except Exception as e:
            logging.debug(f"Error checking filter {filter_path}: {e}")
            return False, None

    def _validate_source_config(self, item: str, source: str, nml: Dict[str, Any], source_type: str):
        """
        Validate configuration for a specific data source after UpdateNamelist.

        After UpdateNamelist runs, all external configs have been merged, so we validate
        the merged configuration directly.

        Args:
            item: Evaluation item name
            source: Source name
            nml: Namelist containing the merged source configuration
            source_type: Type of source ('reference' or 'simulation')
        """
        # After UpdateNamelist, check if the item section exists
        if item not in nml:
            raise ValidationError(f"Missing section '{item}' in {source_type} namelist after UpdateNamelist")

        # Required fields (after UpdateNamelist merge)
        required_fields = [
            f'{source}_data_type',
            f'{source}_varname',
            f'{source}_dir',
            f'{source}_varunit',  # Note: varunit after UpdateNamelist
        ]

        # Get model name if available (for filter checking)
        model_name = nml[item].get(f'{source}_model', None)

        # Check if filter handles this item (only for simulation sources with model info)
        filter_handles_item = False
        filter_path = None
        if source_type == 'simulation' and model_name:
            filter_handles_item, filter_path = self._check_filter_handles_item(model_name, item)

        missing_fields = []
        for field in required_fields:
            value = nml[item].get(field)
            if value is None or value == '':
                # Skip varname and varunit validation if filter handles this item
                if filter_handles_item and field in [f'{source}_varname', f'{source}_varunit']:
                    logging.info(f"  ✓ {field} will be handled by filter: {filter_path}")
                    continue
                missing_fields.append(field)

        if missing_fields:
            raise ValidationError(
                f"Missing required fields for {source} in {item} {source_type} configuration:\n" +
                "\n".join(f"    - {f}" for f in missing_fields)
            )

        # Check data type is valid
        data_type = nml[item][f'{source}_data_type']
        if data_type not in ['grid', 'stn']:
            raise ValidationError(
                f"Invalid data type '{data_type}' for {source} in {item}. Must be 'grid' or 'stn'"
            )

        # Check data directory exists
        data_dir = nml[item][f'{source}_dir']
        if not os.path.exists(data_dir):
            raise ValidationError(
                f"Data directory does not exist: {data_dir}\n"
                f"  ({source_type} {source} for {item})"
            )

    def _validate_units(self):
        """
        Validate units for all evaluation items and detect unknown units.
        """
        # Import unit library to check against known units
        try:
            from openbench.data.Lib_Unit import UnitProcessing
        except ImportError:
            logging.warning("Cannot import Lib_Unit.py, skipping detailed unit validation")
            return

        # Build known units set from Lib_Unit.py
        known_units = self._get_known_units()

        # Check units for all evaluation items
        for item in self.evaluation_items:
            ref_sources = self.ref_nml['general'][f'{item}_ref_source']
            ref_sources = [ref_sources] if isinstance(ref_sources, str) else ref_sources

            sim_sources = self.sim_nml['general'][f'{item}_sim_source']
            sim_sources = [sim_sources] if isinstance(sim_sources, str) else sim_sources

            for ref_source in ref_sources:
                ref_unit = self._get_unit_from_config(item, ref_source, self.ref_nml)
                if ref_unit and ref_unit.lower() not in known_units:
                    self.unknown_units.add(ref_unit)
                    logging.warning(f"  Unknown unit '{ref_unit}' for reference source {ref_source} in {item}")

            for sim_source in sim_sources:
                sim_unit = self._get_unit_from_config(item, sim_source, self.sim_nml)
                if sim_unit and sim_unit.lower() not in known_units:
                    self.unknown_units.add(sim_unit)
                    logging.warning(f"  Unknown unit '{sim_unit}' for simulation source {sim_source} in {item}")

    def _get_known_units(self) -> set:
        """
        Extract known units from Lib_Unit.py.

        Returns:
            set: Set of known unit strings (lowercase)
        """
        try:
            from openbench.data.Lib_Unit import UnitProcessing
            # We need to extract units from the conversion_factors in convert_unit method
            # Since it's defined in the method, we'll create a temporary instance and extract from source
            import inspect
            source = inspect.getsource(UnitProcessing.convert_unit)

            # Parse conversion_factors dictionary from source
            # This is a simplified approach - extract units from the structure
            known_units = set()

            # These are the base units and all their conversions from Lib_Unit.py
            conversion_factors = {
                'gc m-2 day-1': ['gc m-2 s-1', 'g c m-2 s-1', 'g c m-2 day-1', 'g m-2 s-1', 'mol m-2 s-1', 'mumolco2 m-2 s-1'],
                'mm day-1': ['kg m-2 s-1', 'mm s-1', 'mm hr-1', 'mm h-1', 'mm hour-1', 'mm mon-1', 'mm m-1', 'mm month-1', 'w m-2 heat', 'mm 3hour-1'],
                'w m-2': ['mj m-2 day-1', 'mj m-2 d-1'],
                'unitless': ['percent', 'percentage', '%', 'g kg-1', 'fraction', 'm3 m-3', 'm2 m-2', 'g g-1', '1', '0'],
                'k': ['c', 'degc', 'degreec', 'degree c', 'celsius', 'f', 'degf', 'degreef', 'degree f', 'fahrenheit'],
                'm3 s-1': ['m3 day-1', 'm3 d-1', 'l s-1'],
                'mcm': ['m3', 'km3', 'million cubic meters'],
                'mm year-1': ['m year-1', 'cm year-1', 'kg m-2', 'mm month-1', 'mm mon-1', 'mm day-1'],
                'm': ['cm', 'mm'],
                'km2': ['m2'],
                'm s-1': ['km h-1'],
                't ha-1': ['kg ha-1'],
                'kg c m-2': ['g c m-2'],
                'kgc m-2': ['gc m-2', 'g c m-2'],
            }

            # Add all base units and their conversions
            for base_unit, conversions in conversion_factors.items():
                known_units.add(base_unit.lower())
                for conv_unit in conversions:
                    known_units.add(conv_unit.lower())

            logging.info(f"  Loaded {len(known_units)} known units from Lib_Unit.py")
            return known_units

        except Exception as e:
            logging.warning(f"Could not extract known units from Lib_Unit.py: {str(e)}")
            return set()

    def _handle_unknown_units(self):
        """
        Handle unknown units by prompting user and providing guidance.
        """
        logging.warning("\n" + "=" * 80)
        logging.warning("⚠️  UNKNOWN UNITS DETECTED")
        logging.warning("=" * 80)
        logging.warning("\nThe following units are not recognized in Lib_Unit.py:")
        for unit in sorted(self.unknown_units):
            logging.warning(f"  - {unit}")

        logging.warning("\n📝 To add these units to Lib_Unit.py, follow these steps:")
        logging.warning("1. Open the file: openbench/data/Lib_Unit.py")
        logging.warning("2. Locate the 'conversion_factors' dictionary in the convert_unit() method")
        logging.warning("3. Add your unit conversion following the existing pattern:")
        logging.warning("\n   Example for adding a new unit:")
        logging.warning("   'base_unit': {")
        logging.warning("       'your_new_unit': lambda x: x * conversion_factor,")
        logging.warning("   },")
        logging.warning("\n4. Save the file and run the validation again")
        logging.warning("\n" + "=" * 80)

        # Prompt user to continue
        print("\n⚠️  Unknown units detected. See log for details.")
        print("Do you want to continue anyway? (yes/no): ", end='')

        try:
            response = input().strip().lower()
            if response not in ['yes', 'y']:
                logging.info("User chose to abort due to unknown units")
                raise ValidationError("Validation aborted: Unknown units detected")
            logging.info("User chose to continue despite unknown units")
        except EOFError:
            # Non-interactive mode, continue with warning
            logging.warning("Non-interactive mode: continuing with unknown units")

    def _validate_dimensions(self):
        """
        Validate that data dimensions match between reference and simulation data.

        This performs a sample check on the first evaluation item to ensure
        dimensions will be compatible during processing.
        """
        logging.info("  Checking data dimensions (sample validation)...")

        # Sample check on first evaluation item
        if not self.evaluation_items:
            logging.warning("  No evaluation items to validate dimensions")
            return

        sample_item = list(self.evaluation_items)[0]
        logging.info(f"  Performing dimension check on sample item: {sample_item}")

        ref_sources = self.ref_nml['general'][f'{sample_item}_ref_source']
        ref_sources = [ref_sources] if isinstance(ref_sources, str) else ref_sources

        sim_sources = self.sim_nml['general'][f'{sample_item}_sim_source']
        sim_sources = [sim_sources] if isinstance(sim_sources, str) else sim_sources

        # Check first ref-sim pair
        ref_source = ref_sources[0]
        sim_source = sim_sources[0]

        try:
            ref_info = self._get_dimension_info(sample_item, ref_source, self.ref_nml, 'reference')
            sim_info = self._get_dimension_info(sample_item, sim_source, self.sim_nml, 'simulation')

            # Compare data types
            if ref_info['data_type'] != sim_info['data_type']:
                logging.info(f"  ℹ️  Different data types: reference={ref_info['data_type']}, simulation={sim_info['data_type']}")
                logging.info(f"     This is acceptable - OpenBench can handle mixed grid/station data")

            # If both are grid data, check spatial compatibility
            if ref_info['data_type'] == 'grid' and sim_info['data_type'] == 'grid':
                if ref_info['dimensions'] and sim_info['dimensions']:
                    logging.info(f"  Reference dimensions: {ref_info['dimensions']}")
                    logging.info(f"  Simulation dimensions: {sim_info['dimensions']}")

                    # Different dimensions is OK - OpenBench will regrid
                    if ref_info['dimensions'] != sim_info['dimensions']:
                        logging.info(f"  ℹ️  Dimensions differ - automatic regridding will be performed")

            logging.info("  ✓ Dimension compatibility check passed")

        except Exception as e:
            # Dimension check is informational - don't fail if data can't be loaded yet
            logging.warning(f"  ⚠️  Could not fully validate dimensions: {str(e)}")
            logging.warning(f"     This is non-critical - dimensions will be checked during processing")

    def _get_unit_from_config(self, item: str, source: str, nml: Dict[str, Any]) -> Optional[str]:
        """
        Get unit from configuration after UpdateNamelist has merged external configs.

        After UpdateNamelist runs, all units are directly available in the namelist as:
        - nml[item][f'{source}_varunit']

        Args:
            item: Evaluation item name
            source: Source name
            nml: Namelist configuration (after UpdateNamelist)

        Returns:
            Unit string or None if not found
        """
        # After UpdateNamelist, units are directly in nml[item][f'{source}_varunit']
        if item in nml:
            # Try varunit first (standard format after UpdateNamelist)
            unit_value = nml[item].get(f'{source}_varunit', '')
            # Convert to string if it's not already (handle integer values like 0)
            if unit_value is not None and unit_value != '':
                unit = str(unit_value).strip()
                if unit:
                    return unit

            # Fallback to _unit for backward compatibility
            unit_value = nml[item].get(f'{source}_unit', '')
            if unit_value is not None and unit_value != '':
                unit = str(unit_value).strip()
                if unit:
                    return unit

        return None

    def _get_dimension_info(self, item: str, source: str, nml: Dict[str, Any],
                           source_type: str) -> Dict[str, Any]:
        """
        Get dimension information for a data source after UpdateNamelist.

        After UpdateNamelist, all config info is directly in nml[item][f'{source}_*'].

        Args:
            item: Evaluation item name
            source: Source name
            nml: Namelist configuration (after UpdateNamelist)
            source_type: Type ('reference' or 'simulation')

        Returns:
            Dict containing dimension information
        """
        # After UpdateNamelist, all info is in nml[item]
        if item not in nml:
            return {'data_type': 'unknown', 'dimensions': None, 'shape': None}

        # Get configuration from merged namelist
        data_type = nml[item].get(f'{source}_data_type', 'grid')
        data_dir = nml[item].get(f'{source}_dir', '')
        varname = nml[item].get(f'{source}_varname', '')
        data_groupby = nml[item].get(f'{source}_data_groupby', 'single')
        if isinstance(data_groupby, str):
            data_groupby = data_groupby.lower()
        prefix = nml[item].get(f'{source}_prefix', '')
        suffix = nml[item].get(f'{source}_suffix', '')

        info = {
            'data_type': data_type,
            'dimensions': None,
            'shape': None,
        }

        # For grid data, try to get a sample file
        if data_type == 'grid':
            sample_file = None
            if data_groupby == 'single':
                sample_file = os.path.join(data_dir, f"{prefix}{suffix}.nc")
            elif data_groupby == 'year':
                # Try to find any year file
                import glob
                pattern = os.path.join(data_dir, f"{prefix}*{suffix}.nc")
                files = glob.glob(pattern)
                # Filter files: only keep files where prefix is followed by digit (year), not letters
                # This prevents matching "prefix_cama_year.nc" when we want "prefix_year.nc"
                if files:
                    prefix_escaped = re.escape(prefix)
                    suffix_escaped = re.escape(suffix) if suffix else ''
                    file_pattern = re.compile(rf'^{prefix_escaped}\d[^a-zA-Z]*{suffix_escaped}\.nc$')
                    filtered_files = [f for f in files if file_pattern.match(os.path.basename(f))]
                    files = filtered_files if filtered_files else files  # Fallback to original
                if files:
                    sample_file = files[0]

            if sample_file and os.path.exists(sample_file):
                try:
                    with xr.open_dataset(sample_file) as ds:
                        if varname in ds:
                            var = ds[varname]
                            info['dimensions'] = list(var.dims)
                            info['shape'] = var.shape
                        else:
                            logging.warning(f"  Variable '{varname}' not found in {sample_file}")
                except Exception as e:
                    logging.warning(f"  Could not read sample file {sample_file}: {str(e)}")

        return info

    def _validate_variable_existence(self):
        """
        Validate that required variables exist in data files or custom filters.

        For each evaluation item and data source:
        1. Check if varname exists in a sample data file
        2. If not found (or varname is empty), check if custom filter exists
        3. If custom filter exists, verify it handles this evaluation item
        4. Report errors if variable cannot be found
        """
        logging.info("  Checking variable existence in data files and custom filters...")

        missing_variables = []

        for item in self.evaluation_items:
            logging.info(f"  Validating variables for: {item}")

            # Check reference sources
            ref_sources = self.ref_nml['general'][f'{item}_ref_source']
            ref_sources = [ref_sources] if isinstance(ref_sources, str) else ref_sources

            for ref_source in ref_sources:
                result = self._check_variable_availability(
                    item, ref_source, self.ref_nml, 'reference'
                )
                if not result['available']:
                    missing_variables.append(result)

            # Check simulation sources
            sim_sources = self.sim_nml['general'][f'{item}_sim_source']
            sim_sources = [sim_sources] if isinstance(sim_sources, str) else sim_sources

            for sim_source in sim_sources:
                result = self._check_variable_availability(
                    item, sim_source, self.sim_nml, 'simulation'
                )
                if not result['available']:
                    missing_variables.append(result)

        # Report missing variables
        if missing_variables:
            error_msg = self._format_missing_variables_error(missing_variables)
            raise ValidationError(error_msg)

        logging.info("  ✓ All required variables are available")

    def _check_variable_availability(self, item: str, source: str,
                                     nml: Dict[str, Any], source_type: str) -> Dict[str, Any]:
        """
        Check if a variable is available in data file or custom filter.

        Returns:
            Dict with keys: 'available', 'item', 'source', 'source_type', 'reason'
        """
        result = {
            'available': False,
            'item': item,
            'source': source,
            'source_type': source_type,
            'reason': ''
        }

        # Get variable name from config (handle None values)
        varname = ''

        # Try from nml[item] (if UpdateNamelist has run)
        if item in nml:
            varname_raw = nml[item].get(f'{source}_varname', '')
            varname = str(varname_raw).strip() if varname_raw is not None else ''

        # If varname is empty, use item name as default and rely on custom filter
        if not varname:
            logging.info(f"    Variable name empty for {source_type} {source} in {item}, checking custom filter...")
            filter_available = self._check_custom_filter(item, source, nml, source_type)
            if filter_available:
                result['available'] = True
                logging.info(f"    ✓ Custom filter available for {source}")
            else:
                result['reason'] = f"Variable name is empty and no custom filter found"
            return result

        # Try to find and check a sample data file
        try:
            sample_file = self._get_sample_data_file(item, source, nml)
            if sample_file and os.path.exists(sample_file):
                # Check if variable exists in file
                try:
                    import xarray as xr
                    # Try with default time decoding first, then fallback to decode_times=False
                    # for datasets with non-standard time formats (e.g., 'months since 1800 01')
                    try:
                        ds = xr.open_dataset(sample_file)
                    except (ValueError, OSError) as time_err:
                        if 'time' in str(time_err).lower() or 'decode' in str(time_err).lower() or 'calendar' in str(time_err).lower():
                            logging.debug(f"    Retrying with decode_times=False due to: {time_err}")
                            ds = xr.open_dataset(sample_file, decode_times=False)
                        else:
                            raise
                    
                    with ds:
                        if varname in ds:
                            result['available'] = True
                            logging.info(f"    ✓ Variable '{varname}' found in data file")
                            return result
                        else:
                            # Variable not in file, check custom filter
                            logging.info(f"    Variable '{varname}' not in data file, checking custom filter...")
                            filter_available = self._check_custom_filter(item, source, nml, source_type)
                            if filter_available:
                                result['available'] = True
                                logging.info(f"    ✓ Custom filter can generate '{varname}'")
                            else:
                                available_vars = list(ds.data_vars) + list(ds.coords)
                                result['reason'] = f"Variable '{varname}' not in data file and no custom filter found. Available variables: {available_vars[:10]}"
                            return result
                except Exception as e:
                    logging.debug(f"    Could not read sample file {sample_file}: {str(e)}")
                    # Can't verify from file, check filter
                    filter_available = self._check_custom_filter(item, source, nml, source_type)
                    if filter_available:
                        result['available'] = True
                        logging.info(f"    ✓ Custom filter available (couldn't verify from file)")
                    else:
                        result['reason'] = f"Could not verify variable in data file: {str(e)}"
                    return result
            else:
                # No sample file found, must have custom filter
                logging.info(f"    No sample data file found, checking custom filter...")
                filter_available = self._check_custom_filter(item, source, nml, source_type)
                if filter_available:
                    result['available'] = True
                    logging.info(f"    ✓ Custom filter available")
                else:
                    result['reason'] = f"No sample data file found and no custom filter available"
                return result
        except Exception as e:
            logging.warning(f"    Error checking variable availability: {str(e)}")
            # As fallback, check if custom filter exists
            filter_available = self._check_custom_filter(item, source, nml, source_type)
            if filter_available:
                result['available'] = True
                logging.info(f"    ✓ Custom filter available (as fallback)")
            else:
                result['reason'] = f"Error during validation: {str(e)}"
            return result

    def _check_custom_filter(self, item: str, source: str,
                            nml: Dict[str, Any], source_type: str) -> bool:
        """
        Check if a custom filter exists and handles the evaluation item.

        Returns:
            True if custom filter exists and handles the item, False otherwise
        """
        import importlib
        import inspect

        # Determine filter name
        if source_type == 'simulation':
            # For simulation, try to get model name
            model = None

            # First try from {source}_model attribute (if UpdateNamelist has run)
            if item in nml:
                model = nml[item].get(f'{source}_model')

            # If not available, try reading from source definition file -> model_namelist
            if model is None or (isinstance(model, str) and not model.strip()):
                # Get source definition file path from def_nml
                source_def_path = nml.get('def_nml', {}).get(source)
                if source_def_path and os.path.exists(source_def_path):
                    try:
                        import yaml
                        with open(source_def_path, 'r') as f:
                            source_def = yaml.safe_load(f)
                        # Get model_namelist path from source definition
                        model_namelist_path = source_def.get('general', {}).get('model_namelist')
                        if model_namelist_path and os.path.exists(model_namelist_path):
                            with open(model_namelist_path, 'r') as f:
                                model_nml = yaml.safe_load(f)
                            model = model_nml.get('general', {}).get('model')
                    except Exception:
                        pass

            # Final fallback to source name
            if model is None or (isinstance(model, str) and not model.strip()):
                filter_name = source
            else:
                filter_name = str(model).strip()
        else:
            # For reference, use source name
            filter_name = source

        # Try to import the filter module
        try:
            filter_module = importlib.import_module(f"openbench.data.custom.{filter_name}_filter")
            filter_func = getattr(filter_module, f"filter_{filter_name}", None)

            if filter_func is None:
                logging.debug(f"    Filter module exists but no filter_{filter_name} function found")
                return False

            # Check if the filter function handles this item
            # by examining the source code
            source_code = inspect.getsource(filter_func)

            # Look for references to this item in the filter code
            if f'info.item == "{item}"' in source_code or f"info.item == '{item}'" in source_code:
                logging.debug(f"    Custom filter {filter_name}_filter handles {item}")
                return True

            # For dedicated filters (filter name matches the item-specific source),
            # assume it handles that item even without explicit checks
            # For example: GRDC filter for Streamflow, CaMa filter for Streamflow, etc.
            if source_type == 'reference' and filter_name.upper() in ['GRDC', 'CAMA']:
                logging.debug(f"    Dedicated filter {filter_name}_filter assumed to handle {item}")
                return True

            logging.debug(f"    Custom filter {filter_name}_filter exists but doesn't handle {item}")
            return False

        except ImportError:
            logging.debug(f"    No custom filter module found: {filter_name}_filter")
            return False
        except Exception as e:
            logging.debug(f"    Error checking custom filter: {str(e)}")
            return False

    def _get_sample_data_file(self, item: str, source: str, nml: Dict[str, Any]) -> Optional[str]:
        """
        Get a sample data file path for checking variable existence.

        Returns:
            Path to sample file or None if not found
        """
        import glob

        data_dir = nml[item].get(f'{source}_dir', '')
        if not data_dir or not os.path.exists(data_dir):
            return None

        data_groupby = nml[item].get(f'{source}_data_groupby', 'single')
        if isinstance(data_groupby, str):
            data_groupby = data_groupby.lower()

        prefix = nml[item].get(f'{source}_prefix', '')
        suffix = nml[item].get(f'{source}_suffix', '')

        # Helper function to filter files: only keep files where the part after prefix contains no letters before digits
        # This prevents matching files like "prefix_cama_year" when we want "prefix_year"
        def filter_files_no_letters_before_year(files, prefix, suffix):
            """Filter files to exclude those with letters between prefix and year."""
            if not files:
                return files
            filtered = []
            prefix_escaped = re.escape(prefix)
            suffix_escaped = re.escape(suffix) if suffix else ''
            # Pattern: prefix + (year starting with digit) + (only digits and symbols, no letters) + suffix + .nc
            # Match files like: prefix2006-01.nc, prefix2006.nc but not prefix_cama_2006.nc
            pattern = re.compile(rf'^{prefix_escaped}\d[^a-zA-Z]*{suffix_escaped}\.nc$')
            for f in files:
                filename = os.path.basename(f)
                if pattern.match(filename):
                    filtered.append(f)
            return filtered if filtered else files  # Return original if no matches (fallback)

        # Try to find a sample file based on data_groupby
        if data_groupby == 'single':
            sample_file = os.path.join(data_dir, f"{prefix}{suffix}.nc")
            if os.path.exists(sample_file):
                return sample_file
        elif data_groupby == 'year':
            # Find any year file
            pattern = os.path.join(data_dir, f"{prefix}*{suffix}.nc")
            files = glob.glob(pattern)
            files = filter_files_no_letters_before_year(files, prefix, suffix)
            if files:
                return files[0]
        elif data_groupby == 'month':
            # Find any month file
            pattern = os.path.join(data_dir, f"{prefix}*{suffix}.nc")
            files = glob.glob(pattern)
            files = filter_files_no_letters_before_year(files, prefix, suffix)
            if files:
                return files[0]

        # If still not found, try any .nc file in directory
        pattern = os.path.join(data_dir, "*.nc")
        files = glob.glob(pattern)
        if files:
            return files[0]

        return None

    def _format_missing_variables_error(self, missing_variables: List[Dict[str, Any]]) -> str:
        """
        Format a clear error message for missing variables.
        """
        lines = ["\n" + "=" * 80]
        lines.append("❌ MISSING VARIABLES DETECTED")
        lines.append("=" * 80)
        lines.append("\nThe following variables could not be found in data files or custom filters:\n")

        for mv in missing_variables:
            lines.append(f"  Item: {mv['item']}")
            lines.append(f"    {mv['source_type'].title()} source: {mv['source']}")
            lines.append(f"    Reason: {mv['reason']}")
            lines.append("")

        lines.append("📝 To fix this issue:")
        lines.append("1. Verify variable names in your configuration files")
        lines.append("2. Check that data files contain the specified variables")
        lines.append("3. If variables need to be computed from raw data:")
        lines.append("   - Create a custom filter in: openbench/data/custom/<source>_filter.py")
        lines.append("   - Implement filter_<source>(info, ds) function")
        lines.append("   - Handle the evaluation item: if info.item == '<item>': ...")
        lines.append("=" * 80)

        return "\n".join(lines)


def run_pre_validation(main_nl: Dict[str, Any], sim_nml: Dict[str, Any],
                       ref_nml: Dict[str, Any], evaluation_items: List[str],
                       skip_data_check: bool = False) -> bool:
    """
    Convenience function to run pre-validation.

    Args:
        main_nl: Main namelist configuration
        sim_nml: Simulation namelist configuration
        ref_nml: Reference namelist configuration
        evaluation_items: List of evaluation items
        skip_data_check: If True, skip data dimension and unit checks

    Returns:
        bool: True if validation passes

    Raises:
        ValidationError: If validation fails
    """
    validator = PreValidator(main_nl, sim_nml, ref_nml, evaluation_items)
    return validator.validate_all(skip_data_check=skip_data_check)
