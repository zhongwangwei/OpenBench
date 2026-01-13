# -*- coding: utf-8 -*-
"""
Enhanced Configuration Manager for OpenBench

This module provides an enhanced centralized configuration management system that supports
multiple file formats (JSON, YAML, Fortran NML) with validation, templates, and error handling.

Author: Zhongwang Wei
Version: 2.0
Date: July 2025
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

# Import the existing Fortran NML reader
try:
    from .readers import NamelistReader as FortranNMLReader
except ImportError:
    FortranNMLReader = None

# Configuration schemas not needed - using built-in validation
_HAS_SCHEMAS = False
validate_config_file = None
get_schema_for_file = None

# Import unified exception handling
try:
    from ..Mod_Exceptions import ConfigurationError, error_handler, validate_file_exists
    _HAS_EXCEPTIONS = True
except ImportError:
    _HAS_EXCEPTIONS = False
    # Fallback to basic ConfigurationError
    class ConfigurationError(Exception):
        """Basic configuration error for fallback."""
        def __init__(self, message: str, file_path: Optional[str] = None, context: Optional[Dict] = None):
            self.message = message
            self.file_path = file_path
            self.context = context or {}
            super().__init__(self.format_message())
        
        def format_message(self) -> str:
            msg = f"Configuration Error: {self.message}"
            if self.file_path:
                msg += f" (File: {self.file_path})"
            if self.context:
                msg += f" (Context: {self.context})"
            return msg


class ConfigManager:
    """
    Enhanced unified configuration manager supporting multiple file formats.
    
    This class provides a centralized way to load, validate, manage, and generate
    configuration files across the OpenBench system.
    """
    
    def __init__(self):
        """Initialize the ConfigManager."""
        self.name = 'ConfigManager'
        self.version = '2.0'
        self.author = "OpenBench Contributors"
        
        # File format loaders
        self._loaders = {
            '.json': self._load_json,
            '.yaml': self._load_yaml,
            '.yml': self._load_yaml,
            '.nml': self._load_nml
        }
        
        # Cache for loaded configurations
        self._config_cache = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config_path: Union[str, Path], use_cache: bool = True, validate: bool = True) -> Dict[str, Any]:
        """
        Load configuration from file with automatic format detection and validation.
        
        Args:
            config_path: Path to configuration file
            use_cache: Whether to use cached configuration if available
            validate: Whether to validate configuration against schema
            
        Returns:
            Dictionary containing configuration data
            
        Raises:
            ConfigurationError: If file cannot be loaded, parsed, or validated
        """
        config_path = Path(config_path).resolve()
        config_str = str(config_path)
        
        # Check cache first
        if use_cache and config_str in self._config_cache:
            self.logger.debug(f"Using cached configuration for {config_path}")
            return self._config_cache[config_str]
        
        # Validate file exists
        if not config_path.exists():
            raise ConfigurationError(
                f"Configuration file not found", 
                context={'file_path': config_str}
            )
        
        # Detect file format
        file_format = self._detect_format(config_path)
        
        # Warn about deprecated formats
        if file_format == '.nml':
            self.logger.warning(
                f"\n" + "="*80 + "\n"
                f"⚠️  DEPRECATION WARNING: Fortran NML format (.nml) is deprecated!\n"
                f"    The Fortran NML format is no longer being updated.\n"
                f"    Please switch to YAML format (.yaml) for configuration files.\n"
                f"    File: {config_path}\n" + "="*80
            )
        elif file_format == '.json':
            self.logger.warning(
                f"\n" + "="*80 + "\n"
                f"⚠️  DEPRECATION WARNING: JSON format (.json) is deprecated!\n"
                f"    The JSON format is no longer being updated.\n"
                f"    Please switch to YAML format (.yaml) for configuration files.\n"
                f"    File: {config_path}\n" + "="*80
            )
        
        try:
            # Load configuration
            loader = self._loaders.get(file_format)
            if not loader:
                raise ConfigurationError(f"Unsupported file format: {file_format}", config_str)
            
            config = loader(config_path)
            
            # Validate configuration consistency
            if validate:
                warnings = self._validate_config_consistency(config)
                if warnings:
                    self.logger.warning(f"Configuration validation warnings for {config_path}: {'; '.join(warnings)}")
            
            # Cache the configuration
            if use_cache:
                self._config_cache[config_str] = config
            
            self.logger.info(f"Successfully loaded configuration from {config_path}")
            return config
            
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Failed to load configuration: {str(e)}", config_str)
    
    def _detect_format(self, config_path: Path) -> str:
        """
        Detect configuration file format based on extension.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            File format extension (e.g., '.json', '.yaml', '.nml')
        """
        suffix = config_path.suffix.lower()
        
        if suffix in self._loaders:
            return suffix
        
        # Default to .nml for backward compatibility
        self.logger.warning(f"Unknown file extension {suffix}, defaulting to Fortran namelist format")
        return '.nml'
    
    def _load_json(self, config_path: Path) -> Dict[str, Any]:
        """Load JSON configuration file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON format: {str(e)}", str(config_path))
        except Exception as e:
            raise ConfigurationError(f"Failed to read JSON file: {str(e)}", str(config_path))
    
    def _load_yaml(self, config_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML format: {str(e)}", str(config_path))
        except Exception as e:
            raise ConfigurationError(f"Failed to read YAML file: {str(e)}", str(config_path))
    
    def _load_nml(self, config_path: Path) -> Dict[str, Any]:
        """Load Fortran namelist configuration file."""
        if FortranNMLReader is None:
            raise ConfigurationError("Fortran NML reader not available", str(config_path))
        
        try:
            reader = FortranNMLReader()
            # Use the existing method from the original NamelistReader
            return reader.read(str(config_path))
        except Exception as e:
            raise ConfigurationError(f"Failed to read Fortran NML file: {str(e)}", str(config_path))
    
    def _validate_config_consistency(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration consistency and return warnings instead of raising exceptions.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            List of validation warning messages
        """
        warnings = []
        
        # Basic structure validation
        if not isinstance(config, dict):
            warnings.append("Configuration must be a dictionary")
            return warnings
        
        # Check general section
        general = config.get('general', {})
        if general:
            # Time period validation
            start_year = general.get('start_year')
            end_year = general.get('end_year')
            if start_year and end_year:
                if end_year < start_year:
                    warnings.append(f"end_year ({end_year}) should be >= start_year ({start_year})")
            
            # Only drawing validation
            only_drawing = general.get('only_drawing', False)
            if only_drawing:
                # Check if visualization-related items are available
                comparisons = config.get('comparisons', {})
                if not any(comparisons.values()) if comparisons else True:
                    warnings.append("only_drawing is enabled but no comparison plots are enabled")
        
        # Check evaluation items
        eval_items = config.get('evaluation_items', {})
        if eval_items and isinstance(eval_items, dict):
            enabled_items = [k for k, v in eval_items.items() if v]
            if not enabled_items:
                warnings.append("No evaluation items are enabled")
        
        # Check metrics
        metrics = config.get('metrics', {})
        if metrics and isinstance(metrics, dict):
            enabled_metrics = [k for k, v in metrics.items() if v]
            if not enabled_metrics:
                warnings.append("No metrics are enabled")
        
        return warnings
    
    def validate_config(self, config: Dict[str, Any], required_sections: Optional[List[str]] = None) -> bool:
        """
        Validate configuration structure.
        
        Args:
            config: Configuration dictionary to validate
            required_sections: List of required configuration sections
            
        Returns:
            True if configuration is valid
            
        Raises:
            ConfigurationError: If validation fails
        """
        if not isinstance(config, dict):
            raise ConfigurationError("Configuration must be a dictionary")
        
        if required_sections:
            missing_sections = [section for section in required_sections if section not in config]
            if missing_sections:
                raise ConfigurationError(f"Missing required configuration sections: {missing_sections}")
        
        return True
    
    def get_config_value(self, config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation (e.g., 'general.basename').
        
        Args:
            config: Configuration dictionary
            key_path: Dot-separated path to configuration value
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple configuration dictionaries.
        
        Args:
            *configs: Variable number of configuration dictionaries
            
        Returns:
            Merged configuration dictionary
        """
        merged = {}
        
        for config in configs:
            if not isinstance(config, dict):
                continue
            
            for key, value in config.items():
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    # Recursively merge nested dictionaries
                    merged[key] = self.merge_configs(merged[key], value)
                else:
                    merged[key] = value
        
        return merged
    
    def create_basic_config(self) -> Dict[str, Any]:
        """
        Create a basic configuration template.
        
        Returns:
            Basic configuration dictionary
        """
        return {
            "general": {
                "basedir": "./output",
                "basename": "Debug",
                "num_cores": 4,
                "evaluation": True,
                "comparison": False,
                "statistics": False,
                "only_drawing": False,
                "debug_mode": False,
                "start_year": 2004,
                "end_year": 2005,
                "weight": None
            },
            "evaluation_items": {
                "Evapotranspiration": True,
                "Latent_Heat": True,
                "Sensible_Heat": True
            },
            "metrics": {
                "bias": True,
                "RMSE": True,
                "KGE": True,
                "KGESS": True,
                "CRMSD": True,
                "mean_absolute_error": True,
                "absolute_percent_bias": True,
                "percent_bias": True
            },
            "scores": {
                "nBiasScore": True,
                "nRMSEScore": True,
                "nPhaseScore": True,
                "nIavScore": True,
                "nSpatialScore": True,
                "Overall_Score": True
            },
            "comparisons": {
                "HeatMap": False,
                "Taylor_Diagram": False,
                "Target_Diagram": False
            },
            "statistics": {}
        }
    
    def create_advanced_config(self) -> Dict[str, Any]:
        """
        Create an advanced configuration template.
        
        Returns:
            Advanced configuration dictionary
        """
        basic = self.create_basic_config()
        
        # Enable more evaluation items
        basic["evaluation_items"].update({
            "Gross_Primary_Productivity": True,
            "Net_Primary_Productivity": True,
            "Soil_Moisture": True,
            "Snow_Water_Equivalent": True
        })
        
        # Enable more metrics
        basic["metrics"].update({
            "NSE": True,
            "correlation": True,
            "correlation_R2": True,
            "ubRMSE": True,
            "index_agreement": True
        })
        
        # Enable comparisons
        basic["comparisons"].update({
            "HeatMap": True,
            "Taylor_Diagram": True,
            "Target_Diagram": True,
            "Kernel_Density_Estimate": True,
            "Single_Model_Performance_Index": True
        })
        
        # Enable statistics
        basic["statistics"] = {
            "Mean": True,
            "Standard_Deviation": True,
            "Correlation": True
        }
        
        # More processing cores for advanced config
        basic["general"]["num_cores"] = 8
        basic["general"]["comparison"] = True
        basic["general"]["statistics"] = True
        
        return basic
    
    def create_debug_config(self) -> Dict[str, Any]:
        """
        Create a debug configuration template.
        
        Returns:
            Debug configuration dictionary
        """
        return {
            "general": {
                "basedir": "./output",
                "basename": "Debug",
                "num_cores": 1,
                "evaluation": True,
                "comparison": False,
                "statistics": False,
                "only_drawing": False,
                "debug_mode": True,
                "start_year": 2004,
                "end_year": 2004,
                "weight": None
            },
            "evaluation_items": {
                "Evapotranspiration": True
            },
            "metrics": {
                "bias": True,
                "RMSE": True,
                "KGE": True
            },
            "scores": {
                "nBiasScore": True,
                "Overall_Score": True
            },
            "comparisons": {},
            "statistics": {}
        }
    
    def create_config_template(self, template_type: str = 'basic') -> Dict[str, Any]:
        """
        Create configuration template based on type.
        
        Args:
            template_type: Type of template ('basic', 'advanced', 'debug')
            
        Returns:
            Configuration template dictionary
            
        Raises:
            ConfigurationError: If template type is unknown
        """
        templates = {
            'basic': self.create_basic_config,
            'advanced': self.create_advanced_config,
            'debug': self.create_debug_config
        }
        
        if template_type not in templates:
            raise ConfigurationError(f"Unknown template type: {template_type}. Available: {list(templates.keys())}")
        
        return templates[template_type]()
    
    def save_config(self, config: Dict[str, Any], output_path: Union[str, Path], format_type: Optional[str] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary to save
            output_path: Path to save configuration
            format_type: Force specific format ('json', 'yaml'), otherwise infer from extension
            
        Raises:
            ConfigurationError: If saving fails
        """
        output_path = Path(output_path)
        
        # Determine format
        if format_type:
            file_format = f".{format_type.lower()}"
        else:
            file_format = output_path.suffix.lower()
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if file_format == '.json':
                self._save_json(config, output_path)
            elif file_format in ['.yaml', '.yml']:
                self._save_yaml(config, output_path)
            else:
                raise ConfigurationError(f"Unsupported save format: {file_format}")
            
            self.logger.info(f"Configuration saved to: {output_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {str(e)}")
    
    def _save_json(self, config: Dict[str, Any], output_path: Path) -> None:
        """Save configuration as JSON."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def _save_yaml(self, config: Dict[str, Any], output_path: Path) -> None:
        """Save configuration as YAML."""
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get configuration schema description.
        
        Returns:
            Configuration schema with descriptions
        """
        return {
            "general": {
                "description": "General configuration settings",
                "fields": {
                    "basedir": "Base output directory path",
                    "basename": "Base name for output files",
                    "num_cores": "Number of CPU cores to use for parallel processing",
                    "evaluation": "Enable evaluation module",
                    "comparison": "Enable comparison module",
                    "statistics": "Enable statistics module",
                    "only_drawing": "Only generate plots without computation",
                    "debug_mode": "Enable debug mode with verbose logging",
                    "start_year": "Start year for analysis period",
                    "end_year": "End year for analysis period",
                    "weight": "Weighting scheme for aggregation"
                }
            },
            "evaluation_items": {
                "description": "Variables to evaluate",
                "examples": ["Evapotranspiration", "Latent_Heat", "Sensible_Heat", "Gross_Primary_Productivity"]
            },
            "metrics": {
                "description": "Statistical metrics to compute",
                "examples": ["bias", "RMSE", "KGE", "correlation", "NSE"]
            },
            "scores": {
                "description": "Normalized scores to compute",
                "examples": ["nBiasScore", "nRMSEScore", "Overall_Score"]
            },
            "comparisons": {
                "description": "Comparison plots to generate",
                "examples": ["HeatMap", "Taylor_Diagram", "Target_Diagram"]
            },
            "statistics": {
                "description": "Statistical analyses to perform",
                "examples": ["Mean", "Standard_Deviation", "Correlation"]
            }
        }
    
    def clear_cache(self) -> None:
        """Clear configuration cache."""
        self._config_cache.clear()
        self.logger.debug("Configuration cache cleared")
    
    def list_supported_formats(self) -> List[str]:
        """Return list of supported configuration file formats."""
        return list(self._loaders.keys())


# Global configuration manager instance
config_manager = ConfigManager()


def load_config(config_path: Union[str, Path], use_cache: bool = True) -> Dict[str, Any]:
    """
    Convenience function to load configuration using the global config manager.
    
    Args:
        config_path: Path to configuration file
        use_cache: Whether to use cached configuration if available
        
    Returns:
        Dictionary containing configuration data
    """
    return config_manager.load_config(config_path, use_cache)


def validate_config(config: Dict[str, Any], required_sections: Optional[List[str]] = None) -> bool:
    """
    Convenience function to validate configuration using the global config manager.
    
    Args:
        config: Configuration dictionary to validate
        required_sections: List of required configuration sections
        
    Returns:
        True if configuration is valid
    """
    return config_manager.validate_config(config, required_sections)


def create_config_template(template_type: str = 'basic') -> Dict[str, Any]:
    """
    Convenience function to create configuration template.
    
    Args:
        template_type: Type of template ('basic', 'advanced', 'debug')
        
    Returns:
        Configuration template dictionary
    """
    return config_manager.create_config_template(template_type)