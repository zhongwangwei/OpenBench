# -*- coding: utf-8 -*-
"""
OpenBench Unified API

This module provides a unified, high-level interface for OpenBench operations,
making it easy to configure and run complete evaluations.

Author: Zhongwang Wei
Version: 2.0
Date: July 2025
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import json

# Import all OpenBench modules
try:
    from openbench.config import ConfigManager, load_config
    from openbench.util.Mod_Exceptions import OpenBenchError, ConfigurationError, error_handler
    from openbench.core.evaluation.Mod_EvaluationEngine import create_evaluation_engine
    from openbench.util.Mod_OutputManager import ModularOutputManager
    from openbench.util.Mod_LoggingSystem import get_logging_manager
    from openbench.util.Mod_ParallelEngine import ParallelEngine
    from openbench.data.Mod_CacheSystem import get_cache_manager
    from openbench.data.Mod_DataPipeline import create_standard_pipeline
    from openbench.util.Mod_APIService import create_api_service
    _HAS_MODULES = True
except ImportError as e:
    _HAS_MODULES = False
    print(f"Warning: Some OpenBench modules not available: {e}")
    
    # Fallback definitions
    OpenBenchError = Exception
    ConfigurationError = Exception
    def error_handler(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Import legacy OpenBench components (disabled to avoid dependency conflicts)
try:
    # Skip these imports to avoid cmaps dependency issues
    # from Mod_Evaluation import ModelEvaluation
    # from Mod_Statistics import statistics_calculate  
    # from Mod_Comparison import ComparisonProcessing
    _HAS_LEGACY = False  # Temporarily disable to avoid import issues
except ImportError:
    _HAS_LEGACY = False

# Data handling imports
try:
    import xarray as xr
    import numpy as np
    import pandas as pd
    _HAS_DATA_LIBS = True
except ImportError:
    _HAS_DATA_LIBS = False


class OpenBench:
    """
    Unified OpenBench interface for land surface model evaluation.
    
    This class provides a simple, high-level API for configuring and running
    complete OpenBench evaluations with all the power of the modular system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize OpenBench instance.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.results = {}
        self._initialized = False
        
        # Initialize components
        if _HAS_MODULES:
            self.config_manager = ConfigManager()
            self.output_manager = ModularOutputManager()
            self.parallel_engine = ParallelEngine()
            self.logger = get_logging_manager().get_logger("OpenBench")
            
            # Try to get cache manager
            try:
                self.cache_manager = get_cache_manager()
            except:
                self.cache_manager = None
        else:
            self.config_manager = None
            self.output_manager = None
            self.parallel_engine = None
            self.logger = logging.getLogger("OpenBench")
            self.cache_manager = None
        
        # Legacy evaluation components
        if _HAS_LEGACY:
            self.model_evaluation = None  # Will be initialized when needed
            self.statistics = None
            self.comparison = None
        
        # API service
        self.api_service = None
        
        # Evaluation engines
        self.evaluation_engines = {}
        
        self.logger.info("OpenBench instance created")
    
    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> 'OpenBench':
        """
        Create OpenBench instance from configuration file.
        
        Supports the following configuration formats:
        - JSON (.json)
        - YAML (.yaml, .yml)
        - Fortran Namelist (.nml)
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configured OpenBench instance
            
        Examples:
            >>> # Load from JSON
            >>> ob = OpenBench.from_config('config/main.json')
            
            >>> # Load from YAML
            >>> ob = OpenBench.from_config('config/main.yaml')
            
            >>> # Load from Fortran Namelist
            >>> ob = OpenBench.from_config('config/main.nml')
        """
        instance = cls()
        instance.load_config(config_path)
        return instance
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OpenBench':
        """
        Create OpenBench instance from configuration dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Configured OpenBench instance
        """
        instance = cls(config_dict)
        instance._initialize_from_config()
        return instance
    
    @error_handler(reraise=True)
    def load_config(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from file.
        
        Supports JSON (.json), YAML (.yaml/.yml), and Fortran Namelist (.nml) formats.
        
        Args:
            config_path: Path to configuration file
        """
        if not _HAS_MODULES:
            # Fallback: support all three formats
            config_path = Path(config_path)
            if config_path.exists():
                try:
                    # JSON format
                    if config_path.suffix == '.json':
                        self.logger.warning(
                            f"\n" + "="*80 + "\n"
                            f"⚠️  DEPRECATION WARNING: JSON format (.json) is deprecated!\n"
                            f"    The JSON format is no longer being updated.\n"
                            f"    Please switch to YAML format (.yaml) for configuration files.\n"
                            f"    File: {config_path}\n" + "="*80
                        )
                        import json
                        with open(config_path, 'r') as f:
                            self.config = json.load(f)
                        self._initialize_from_config()
                        self.logger.info(f"Configuration loaded from {config_path} (JSON, fallback mode)")
                        return
                    
                    # YAML format
                    elif config_path.suffix in ['.yaml', '.yml']:
                        try:
                            import yaml
                            with open(config_path, 'r') as f:
                                self.config = yaml.safe_load(f)
                            self._initialize_from_config()
                            self.logger.info(f"Configuration loaded from {config_path} (YAML, fallback mode)")
                            return
                        except ImportError:
                            self.logger.warning("PyYAML not available for fallback YAML loading")
                    
                    # Fortran Namelist format
                    elif config_path.suffix == '.nml':
                        self.logger.warning(
                            f"\n" + "="*80 + "\n"
                            f"⚠️  DEPRECATION WARNING: Fortran NML format (.nml) is deprecated!\n"
                            f"    The Fortran NML format is no longer being updated.\n"
                            f"    Please switch to YAML format (.yaml) for configuration files.\n"
                            f"    File: {config_path}\n" + "="*80
                        )
                        try:
                            import f90nml
                            nml = f90nml.read(str(config_path))
                            # Convert namelist to dict
                            self.config = {k: dict(v) if hasattr(v, 'items') else v 
                                         for k, v in nml.items()}
                            self._initialize_from_config()
                            self.logger.info(f"Configuration loaded from {config_path} (NML, fallback mode)")
                            return
                        except ImportError:
                            self.logger.warning("f90nml not available for fallback NML loading")
                            
                except Exception as e:
                    self.logger.warning(f"Fallback config loading failed: {e}")
            
            # If fallback fails, create minimal config with proper structure
            self.config = {
                "general": {
                    "basename": "debug",
                    "basedir": "./output",
                    "reference_nml": "./nml/ref-Debug.yaml",
                    "simulation_nml": "./nml/sim-Debug.yaml",
                    "figure_nml": "./nml/figlib.yaml",
                    "evaluation": True,
                    "comparison": True,
                    "statistics": False
                },
                "config_path": str(config_path), 
                "fallback_mode": True
            }
            self._initialize_from_config()
            self.logger.warning(f"Using minimal config for {config_path}")
            return
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        # Use ConfigManager which already supports all three formats
        self.config = self.config_manager.load_config(str(config_path))
        self._initialize_from_config()
        
        self.logger.info(f"Configuration loaded from {config_path}")
    
    def _initialize_from_config(self) -> None:
        """Initialize components based on configuration."""
        if not self.config:
            return
        
        # Setup logging if configured
        if 'logging' in self.config and _HAS_MODULES:
            log_config = self.config['logging']
            if log_config.get('level'):
                logging.getLogger().setLevel(getattr(logging, log_config['level'].upper()))
        
        # Initialize evaluation engines based on config
        engine_configs = self.config.get('engines', {})
        for engine_name, engine_config in engine_configs.items():
            if _HAS_MODULES:
                self.evaluation_engines[engine_name] = create_evaluation_engine(
                    engine_config.get('type', 'modular'),
                    **engine_config.get('parameters', {})
                )
        
        # Setup default engines if none configured
        if not self.evaluation_engines and _HAS_MODULES:
            self.evaluation_engines = {
                'modular': create_evaluation_engine('modular'),
                'grid': create_evaluation_engine('grid'),
                'station': create_evaluation_engine('station')
            }
        
        # Initialize legacy components if needed
        if _HAS_LEGACY and self.config.get('use_legacy', True):
            try:
                self.model_evaluation = ModelEvaluation()
            except:
                pass
        
        self._initialized = True
        self.logger.info("OpenBench components initialized")
    
    @error_handler(reraise=True)
    def run(
        self, 
        simulation_data: Optional[Union[str, xr.Dataset]] = None,
        reference_data: Optional[Union[str, xr.Dataset]] = None,
        metrics: Optional[List[str]] = None,
        engine_type: str = 'modular',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run complete OpenBench evaluation.
        
        Args:
            simulation_data: Simulation data path or dataset
            reference_data: Reference data path or dataset  
            metrics: List of metrics to calculate
            engine_type: Evaluation engine type
            **kwargs: Additional evaluation parameters
            
        Returns:
            Complete evaluation results
        """
        if not self._initialized:
            self._initialize_from_config()
        
        # Use legacy evaluation if configured or if modular system unavailable
        if (self.config.get('use_legacy', False) and _HAS_LEGACY) or not _HAS_MODULES:
            return self._run_legacy_evaluation(**kwargs)
        
        # Use modular evaluation system
        return self._run_modular_evaluation(
            simulation_data, reference_data, metrics, engine_type, **kwargs
        )
    
    def _run_legacy_evaluation(self, **kwargs) -> Dict[str, Any]:
        """Run evaluation using legacy system."""
        if not _HAS_LEGACY:
            raise ImportError("Legacy evaluation system not available")
        
        self.logger.info("Running legacy OpenBench evaluation")
        
        try:
            # Initialize legacy components if not done
            if not self.model_evaluation:
                self.model_evaluation = ModelEvaluation()
            
            # Run evaluation using the configuration
            config_file = kwargs.get('config_file')
            if config_file:
                # This would call the original openbench.py workflow
                from openbench.openbench import main
                results = main(config_file)
            else:
                # Use the current config
                results = self.model_evaluation.run_evaluation(self.config)
            
            self.results = {
                'evaluation_type': 'legacy',
                'results': results,
                'metadata': {
                    'engine': 'legacy',
                    'config': self.config
                }
            }
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Legacy evaluation failed: {e}")
            raise OpenBenchError(f"Legacy evaluation failed: {e}")
    
    def _run_modular_evaluation(
        self,
        simulation_data: Optional[Union[str, xr.Dataset]],
        reference_data: Optional[Union[str, xr.Dataset]], 
        metrics: Optional[List[str]],
        engine_type: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Run evaluation using modular system."""
        if not _HAS_MODULES:
            raise ImportError("Modular evaluation system not available")
        
        self.logger.info("Running modular OpenBench evaluation")

        # Track datasets we open so we can close them
        datasets_to_close = []

        try:
            # Load data if paths provided
            if isinstance(simulation_data, str):
                simulation_data = xr.open_dataset(simulation_data)
                datasets_to_close.append(simulation_data)
            if isinstance(reference_data, str):
                reference_data = xr.open_dataset(reference_data)
                datasets_to_close.append(reference_data)

            # Use data from config if not provided
            if simulation_data is None and 'simulation' in self.config:
                sim_path = self.config['simulation'].get('path')
                if sim_path:
                    simulation_data = xr.open_dataset(sim_path)
                    datasets_to_close.append(simulation_data)

            if reference_data is None and 'reference' in self.config:
                ref_path = self.config['reference'].get('path')
                if ref_path:
                    reference_data = xr.open_dataset(ref_path)
                    datasets_to_close.append(reference_data)

            # Use metrics from config if not provided
            if metrics is None:
                metrics = self.config.get('metrics', ['bias', 'RMSE', 'correlation'])

            # Get evaluation engine
            engine = self.evaluation_engines.get(engine_type)
            if not engine:
                engine = create_evaluation_engine(engine_type)
                self.evaluation_engines[engine_type] = engine

            # Run evaluation
            if simulation_data is not None and reference_data is not None:
                eval_results = engine.evaluate(
                    simulation_data, reference_data, metrics, **kwargs
                )
            else:
                # Run full evaluation using config
                eval_results = self._run_full_config_evaluation(engine, **kwargs)

            # Process and save results
            self.results = {
                'evaluation_type': 'modular',
                'engine_type': engine_type,
                'results': eval_results,
                'metadata': {
                    'metrics': metrics,
                    'config': self.config,
                    'engine': engine_type
                }
            }

            # Save results if output manager available
            if self.output_manager:
                output_path = kwargs.get('output_path', './output/evaluation_results.json')
                self.save_results(output_path)

            self.logger.info("Modular evaluation completed successfully")
            return self.results

        except Exception as e:
            self.logger.error(f"Modular evaluation failed: {e}")
            raise OpenBenchError(f"Modular evaluation failed: {e}")
        finally:
            # Close any datasets we opened
            for ds in datasets_to_close:
                try:
                    ds.close()
                except Exception:
                    pass
    
    def _run_full_config_evaluation(self, engine, **kwargs) -> Dict[str, Any]:
        """Run full evaluation based on configuration."""
        if not _HAS_LEGACY:
            raise ImportError("Full config evaluation requires legacy components")
        
        # This integrates with the existing evaluation workflow
        # Use the configuration to determine what evaluations to run
        
        results = {}
        
        # Run grid evaluations if configured
        if 'grid_evaluation' in self.config:
            grid_config = self.config['grid_evaluation']
            # Implementation would call grid evaluation modules
            results['grid'] = {}
        
        # Run station evaluations if configured  
        if 'station_evaluation' in self.config:
            station_config = self.config['station_evaluation']
            # Implementation would call station evaluation modules
            results['station'] = {}
        
        # Run comparisons if configured
        if 'comparisons' in self.config:
            comparison_config = self.config['comparisons']
            # Implementation would call comparison modules
            results['comparisons'] = {}
        
        return results
    
    def save_results(self, output_path: str, format_type: str = 'json') -> None:
        """
        Save evaluation results to file.
        
        Args:
            output_path: Output file path
            format_type: Output format ('json', 'csv', 'netcdf')
        """
        if not self.results:
            self.logger.warning("No results to save")
            return
        
        if self.output_manager:
            self.output_manager.save_data(
                self.results,
                category='evaluation_results',
                filename=os.path.basename(output_path),
                format_type=format_type
            )
        else:
            # Fallback to basic JSON save
            if format_type == 'json':
                with open(output_path, 'w') as f:
                    json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {output_path}")
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available evaluation metrics."""
        if self.evaluation_engines:
            engine = list(self.evaluation_engines.values())[0]
            return engine.get_supported_metrics()
        elif _HAS_MODULES:
            engine = create_evaluation_engine()
            return engine.get_supported_metrics()
        else:
            return ['bias', 'RMSE', 'correlation', 'NSE']
    
    def get_available_engines(self) -> List[str]:
        """Get list of available evaluation engines."""
        if self.evaluation_engines:
            return list(self.evaluation_engines.keys())
        else:
            return ['modular', 'grid', 'station']
    
    def create_api_service(self, **config) -> 'APIService':
        """
        Create API service for remote access.
        
        Args:
            **config: API service configuration
            
        Returns:
            Configured API service instance
        """
        if not _HAS_MODULES:
            raise ImportError("API service requires OpenBench modules")
        
        # Merge API config with OpenBench config
        api_config = {**self.config.get('api', {}), **config}
        
        self.api_service = create_api_service()
        self.api_service.config.update(api_config)
        
        self.logger.info("API service created")
        return self.api_service
    
    def start_api_service(self, **config) -> None:
        """
        Start API service for remote access.
        
        Args:
            **config: API service configuration
        """
        if not self.api_service:
            self.create_api_service(**config)
        
        self.logger.info("Starting OpenBench API service")
        self.api_service.run(**config)
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate current configuration.
        
        Returns:
            Validation results
        """
        if not self.config_manager:
            return {"valid": False, "message": "Config manager not available"}
        
        try:
            self.config_manager.validate_config(self.config)
            return {"valid": True, "message": "Configuration is valid"}
        except Exception as e:
            return {"valid": False, "message": str(e)}
    
    @staticmethod
    def detect_config_format(config_path: Union[str, Path]) -> str:
        """
        Detect configuration file format from extension.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Format string: 'json', 'yaml', 'nml', or 'unknown'
        """
        config_path = Path(config_path)
        suffix = config_path.suffix.lower()
        
        if suffix == '.json':
            return 'json'
        elif suffix in ['.yaml', '.yml']:
            return 'yaml'
        elif suffix == '.nml':
            return 'nml'
        else:
            return 'unknown'
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()
    
    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """
        Update configuration.
        
        Args:
            config_updates: Configuration updates to apply
        """
        self.config.update(config_updates)
        self._initialize_from_config()
        self.logger.info("Configuration updated")
    
    def get_results(self) -> Dict[str, Any]:
        """Get evaluation results."""
        return self.results.copy()
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        if self.cache_manager:
            self.cache_manager.clear_all()
            self.logger.info("Cache cleared")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and status."""
        info = {
            'version': '2.0.0',
            'modules_available': _HAS_MODULES,
            'legacy_available': _HAS_LEGACY,
            'data_libs_available': _HAS_DATA_LIBS,
            'config_loaded': bool(self.config),
            'initialized': self._initialized,
            'available_engines': self.get_available_engines(),
            'available_metrics': self.get_available_metrics()
        }
        
        # Add system resources if available
        try:
            import psutil
            info['system'] = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            }
        except ImportError:
            pass
        
        return info
    
    def __repr__(self) -> str:
        """String representation of OpenBench instance."""
        status = "initialized" if self._initialized else "not initialized"
        config_status = "loaded" if self.config else "not loaded"
        return f"OpenBench(status={status}, config={config_status})"
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        # Cleanup resources
        if self.cache_manager:
            self.cache_manager.cleanup()
        
        # Close any open datasets
        if hasattr(self, '_datasets'):
            for dataset in self._datasets:
                if hasattr(dataset, 'close'):
                    dataset.close()


# Convenience functions for quick usage
def run_evaluation(
    config_path: Union[str, Path],
    simulation_data: Optional[Union[str, xr.Dataset]] = None,
    reference_data: Optional[Union[str, xr.Dataset]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Quick evaluation function.
    
    Supports JSON, YAML, and Fortran Namelist configuration formats.
    
    Args:
        config_path: Configuration file path (.json, .yaml, .yml, or .nml)
        simulation_data: Optional simulation data
        reference_data: Optional reference data
        **kwargs: Additional parameters
        
    Returns:
        Evaluation results
        
    Examples:
        >>> # Run with JSON config
        >>> results = run_evaluation('config/main.json')
        
        >>> # Run with YAML config
        >>> results = run_evaluation('config/main.yaml')
        
        >>> # Run with Fortran Namelist
        >>> results = run_evaluation('config/main.nml')
    """
    with OpenBench.from_config(config_path) as ob:
        return ob.run(simulation_data, reference_data, **kwargs)


def create_openbench(config: Optional[Union[str, Dict[str, Any]]] = None) -> OpenBench:
    """
    Create OpenBench instance with flexible configuration.
    
    Args:
        config: Configuration file path (supports .json, .yaml, .yml, .nml) or dictionary
        
    Returns:
        OpenBench instance
        
    Examples:
        >>> # Create from JSON file
        >>> ob = create_openbench('config/main.json')
        
        >>> # Create from YAML file
        >>> ob = create_openbench('config/main.yaml')
        
        >>> # Create from Fortran Namelist
        >>> ob = create_openbench('config/main.nml')
        
        >>> # Create from dictionary
        >>> ob = create_openbench({'general': {'basename': 'test'}})
    """
    if isinstance(config, str):
        return OpenBench.from_config(config)
    elif isinstance(config, dict):
        return OpenBench.from_dict(config)
    else:
        return OpenBench()