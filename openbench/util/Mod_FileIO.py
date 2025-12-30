# -*- coding: utf-8 -*-
"""
Safe File I/O Utilities for OpenBench

This module provides safe wrappers for file operations with comprehensive
error handling, validation, and logging.

Author: Zhongwang Wei
Version: 1.0
Date: January 2025
"""

import os
import logging
import time
from typing import Optional, Union, Any, Dict
import xarray as xr
import numpy as np

# Import exception handling
try:
    from openbench.util.Mod_Exceptions import (
        FileSystemError,
        DataProcessingError,
        ValidationError
    )
except ImportError:
    # Fallback if exceptions module not available
    class FileSystemError(Exception):
        pass
    class DataProcessingError(Exception):
        pass
    class ValidationError(Exception):
        pass


def validate_file_path(file_path: str,
                       check_readable: bool = True,
                       check_writable: bool = False,
                       create_parent: bool = False) -> None:
    """
    Validate file path with comprehensive checks.

    Args:
        file_path: Path to the file
        check_readable: Whether to check read permission
        check_writable: Whether to check write permission
        create_parent: Whether to create parent directory if missing

    Raises:
        FileSystemError: If validation fails
    """
    # Convert to absolute path
    file_path = os.path.abspath(file_path)

    # Check if parent directory exists
    parent_dir = os.path.dirname(file_path)
    if not os.path.exists(parent_dir):
        if create_parent:
            try:
                os.makedirs(parent_dir, exist_ok=True)
                logging.info(f"Created parent directory: {parent_dir}")
            except Exception as e:
                raise FileSystemError(
                    f"Failed to create parent directory: {parent_dir}",
                    context={'parent_dir': parent_dir, 'file_path': file_path},
                    original_error=e
                )
        else:
            raise FileSystemError(
                f"Parent directory does not exist: {parent_dir}",
                context={'parent_dir': parent_dir, 'file_path': file_path}
            )

    # Check if file exists (for read operations)
    if check_readable:
        if not os.path.exists(file_path):
            raise FileSystemError(
                f"File not found: {file_path}",
                context={'file_path': file_path}
            )

        if not os.path.isfile(file_path):
            raise FileSystemError(
                f"Path exists but is not a file: {file_path}",
                context={'file_path': file_path, 'is_dir': os.path.isdir(file_path)}
            )

        # Check file size (be tolerant of race conditions)
        try:
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise FileSystemError(
                    f"File is empty: {file_path}",
                    context={'file_path': file_path, 'size': 0}
                )
        except (OSError, IOError) as e:
            # File exists but size check failed (possible race condition)
            logging.debug(f"File size check failed for {file_path}: {e}")

        # Check read permission
        if not os.access(file_path, os.R_OK):
            raise FileSystemError(
                f"No read permission for file: {file_path}",
                context={'file_path': file_path}
            )

    # Check write permission (for write operations)
    if check_writable:
        if os.path.exists(file_path):
            if not os.access(file_path, os.W_OK):
                raise FileSystemError(
                    f"No write permission for file: {file_path}",
                    context={'file_path': file_path}
                )
        else:
            # Check if parent directory is writable
            if not os.access(parent_dir, os.W_OK):
                raise FileSystemError(
                    f"No write permission for directory: {parent_dir}",
                    context={'parent_dir': parent_dir, 'file_path': file_path}
                )


def safe_open_netcdf(file_path: str,
                     variable_name: Optional[str] = None,
                     timeout: int = 30,
                     validate: bool = True) -> Union[xr.Dataset, xr.DataArray]:
    """
    Safely open NetCDF file with comprehensive error handling.

    Args:
        file_path: Path to NetCDF file
        variable_name: Optional variable name to extract
        timeout: Maximum time to wait for file (seconds)
        validate: Whether to validate file before opening

    Returns:
        xr.Dataset or xr.DataArray

    Raises:
        FileSystemError: If file validation fails
        DataProcessingError: If file cannot be opened or processed
    """
    # Convert to absolute path
    file_path = os.path.abspath(file_path)

    # Check if file exists first
    if not os.path.exists(file_path):
        # File doesn't exist, wait for it
        logging.debug(f"File does not exist, waiting: {file_path}")
        if not _wait_for_file(file_path, max_wait_time=timeout):
            raise FileSystemError(
                f"File not available after {timeout}s: {file_path}",
                context={'file_path': file_path, 'timeout': timeout}
            )

    # Validate file if requested (only after we know it exists)
    if validate:
        try:
            validate_file_path(file_path, check_readable=True)
        except FileSystemError as e:
            # Validation failed but file exists - log and continue
            logging.debug(f"File validation failed but file exists: {e}")
            # Don't raise - xr.open_dataset will catch real issues

    # Try to open the dataset
    try:
        logging.debug(f"Opening NetCDF file: {file_path}")
        ds = xr.open_dataset(file_path)
    except FileNotFoundError as e:
        raise FileSystemError(
            f"File not found when opening: {file_path}",
            context={'file_path': file_path},
            original_error=e
        )
    except PermissionError as e:
        raise FileSystemError(
            f"Permission denied when opening: {file_path}",
            context={'file_path': file_path},
            original_error=e
        )
    except (OSError, IOError) as e:
        raise DataProcessingError(
            f"I/O error when opening NetCDF file: {file_path}",
            context={'file_path': file_path},
            original_error=e
        )
    except Exception as e:
        # Try with decode_times=False as fallback
        logging.warning(f"Failed to open {file_path} with default time decoding: {e}. Retrying with decode_times=False")
        try:
            ds = xr.open_dataset(file_path, decode_times=False)
        except Exception as e2:
            raise DataProcessingError(
                f"Failed to open NetCDF file: {file_path}",
                context={'file_path': file_path, 'error_type': type(e).__name__},
                original_error=e2
            )

    # If variable name is specified, extract it
    if variable_name:
        try:
            return _extract_variable(ds, variable_name, file_path)
        except Exception as e:
            # Close dataset on error
            if hasattr(ds, 'close'):
                ds.close()
            raise

    return ds


def _extract_variable(ds: xr.Dataset,
                     variable_name: str,
                     file_path: str) -> xr.DataArray:
    """
    Extract variable from dataset with validation.

    Args:
        ds: xarray Dataset
        variable_name: Name of variable to extract
        file_path: File path (for error messages)

    Returns:
        xr.DataArray

    Raises:
        ValidationError: If variable not found
    """
    # Check if variable exists
    all_vars = list(ds.data_vars) + list(ds.coords)

    if variable_name not in all_vars:
        available_data_vars = list(ds.data_vars)
        available_coords = list(ds.coords)

        error_msg = f"Variable '{variable_name}' not found in dataset: {file_path}"
        logging.error(error_msg)
        logging.error(f"  Available data variables: {available_data_vars}")
        logging.error(f"  Available coordinates: {available_coords}")

        raise ValidationError(
            error_msg,
            context={
                'file_path': file_path,
                'requested_variable': variable_name,
                'available_data_vars': available_data_vars,
                'available_coords': available_coords
            }
        )

    # Extract variable
    try:
        var_data = ds[variable_name]
        logging.debug(f"Extracted variable '{variable_name}' from {file_path}")
        return var_data
    except Exception as e:
        raise DataProcessingError(
            f"Failed to extract variable '{variable_name}' from {file_path}",
            context={'file_path': file_path, 'variable_name': variable_name},
            original_error=e
        )


def validate_variable_in_dataset(ds: Union[xr.Dataset, xr.DataArray],
                                 variable_name: str,
                                 context: str = "dataset") -> None:
    """
    Validate that a variable exists in a dataset.

    Args:
        ds: xarray Dataset or DataArray
        variable_name: Variable name to check
        context: Context description for error messages

    Raises:
        ValidationError: If variable not found
    """
    if isinstance(ds, xr.DataArray):
        # For DataArray, check if name matches
        if ds.name != variable_name:
            raise ValidationError(
                f"DataArray name '{ds.name}' does not match expected '{variable_name}'",
                context={'context': context, 'expected': variable_name, 'actual': ds.name}
            )
        return

    # For Dataset, check both data_vars and coords
    all_vars = list(ds.data_vars) + list(ds.coords)

    if variable_name not in all_vars:
        available_data_vars = list(ds.data_vars)
        available_coords = list(ds.coords)

        raise ValidationError(
            f"Variable '{variable_name}' not found in {context}",
            context={
                'context': context,
                'requested_variable': variable_name,
                'available_data_vars': available_data_vars,
                'available_coords': available_coords
            }
        )


def safe_save_netcdf(ds: Union[xr.Dataset, xr.DataArray],
                     file_path: str,
                     create_parent: bool = True,
                     backup_existing: bool = False) -> None:
    """
    Safely save NetCDF file with error handling.

    Args:
        ds: xarray Dataset or DataArray to save
        file_path: Output file path
        create_parent: Whether to create parent directory
        backup_existing: Whether to backup existing file

    Raises:
        FileSystemError: If file system operations fail
        DataProcessingError: If save operation fails
    """
    # Convert to absolute path
    file_path = os.path.abspath(file_path)

    # Validate output path
    try:
        validate_file_path(
            file_path,
            check_readable=False,
            check_writable=True,
            create_parent=create_parent
        )
    except FileSystemError as e:
        logging.error(f"Output path validation failed: {e}")
        raise

    # Backup existing file if requested
    if backup_existing and os.path.exists(file_path):
        backup_path = f"{file_path}.backup"
        try:
            import shutil
            shutil.copy2(file_path, backup_path)
            logging.info(f"Backed up existing file to: {backup_path}")
        except Exception as e:
            logging.warning(f"Failed to create backup: {e}")

    # Save to temporary file first
    temp_path = f"{file_path}.tmp"

    try:
        logging.debug(f"Saving to temporary file: {temp_path}")
        ds.to_netcdf(temp_path)

        # Verify temporary file was created and has content
        if not os.path.exists(temp_path):
            raise DataProcessingError(
                f"Temporary file was not created: {temp_path}",
                context={'temp_path': temp_path, 'final_path': file_path}
            )

        if os.path.getsize(temp_path) == 0:
            raise DataProcessingError(
                f"Temporary file is empty: {temp_path}",
                context={'temp_path': temp_path, 'final_path': file_path}
            )

        # Move temporary file to final location
        import shutil
        shutil.move(temp_path, file_path)
        logging.info(f"Successfully saved NetCDF file: {file_path}")

    except Exception as e:
        # Clean up temporary file on error
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logging.debug(f"Cleaned up temporary file: {temp_path}")
            except Exception:
                pass

        raise DataProcessingError(
            f"Failed to save NetCDF file: {file_path}",
            context={'file_path': file_path, 'temp_path': temp_path},
            original_error=e
        )


def _wait_for_file(file_path: str,
                   max_wait_time: int = 30,
                   check_interval: float = 1.0) -> bool:
    """
    Wait for a file to exist and be readable.

    Args:
        file_path: Path to file
        max_wait_time: Maximum time to wait (seconds)
        check_interval: Time between checks (seconds)

    Returns:
        True if file is available, False if timeout
    """
    start_time = time.time()

    while (time.time() - start_time) < max_wait_time:
        if os.path.exists(file_path):
            try:
                size = os.path.getsize(file_path)
                if size > 0:
                    # Check if file is accessible (but don't read it, as macOS extended attributes can cause issues)
                    if os.access(file_path, os.R_OK):
                        elapsed = time.time() - start_time
                        if elapsed > 0.1:  # Only log if we actually waited
                            logging.info(f"File available after {elapsed:.1f}s: {file_path} ({size} bytes)")
                        return True
            except (OSError, IOError) as e:
                # File exists but not ready yet
                logging.debug(f"File not ready: {e}")
                pass

        time.sleep(check_interval)

    logging.warning(f"File not available after {max_wait_time}s: {file_path}")
    return False


def check_dataset_integrity(ds: Union[xr.Dataset, xr.DataArray],
                           required_dims: Optional[list] = None,
                           required_vars: Optional[list] = None,
                           context: str = "dataset") -> None:
    """
    Check dataset integrity and completeness.

    Args:
        ds: xarray Dataset or DataArray
        required_dims: List of required dimensions
        required_vars: List of required variables (for Dataset only)
        context: Context description for error messages

    Raises:
        ValidationError: If integrity checks fail
    """
    # Check dimensions
    if required_dims:
        missing_dims = set(required_dims) - set(ds.dims)
        if missing_dims:
            raise ValidationError(
                f"Missing required dimensions in {context}: {missing_dims}",
                context={
                    'context': context,
                    'required_dims': required_dims,
                    'available_dims': list(ds.dims),
                    'missing_dims': list(missing_dims)
                }
            )

    # Check variables (for Dataset only)
    if isinstance(ds, xr.Dataset) and required_vars:
        missing_vars = set(required_vars) - set(ds.data_vars)
        if missing_vars:
            raise ValidationError(
                f"Missing required variables in {context}: {missing_vars}",
                context={
                    'context': context,
                    'required_vars': required_vars,
                    'available_vars': list(ds.data_vars),
                    'missing_vars': list(missing_vars)
                }
            )

    # Check for NaN values
    if isinstance(ds, xr.DataArray):
        if ds.isnull().all():
            raise ValidationError(
                f"Dataset contains only NaN values: {context}",
                context={'context': context}
            )
    elif isinstance(ds, xr.Dataset):
        all_nan_vars = []
        for var in ds.data_vars:
            if ds[var].isnull().all():
                all_nan_vars.append(var)

        if all_nan_vars:
            logging.warning(f"Variables with all NaN values in {context}: {all_nan_vars}")
