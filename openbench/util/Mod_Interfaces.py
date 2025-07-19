# -*- coding: utf-8 -*-
"""
Core Interface Abstractions for OpenBench

This module defines abstract base classes and interfaces for core OpenBench
components to promote modularity, testability, and maintainability.

Author: Zhongwang Wei
Version: 1.0
Date: July 2025
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import xarray as xr
import pandas as pd
from pathlib import Path


class IOutputManager(ABC):
    """Abstract interface for output management."""
    
    @abstractmethod
    def save_data(self, data: Any, category: str, filename: str, **kwargs) -> str:
        """Save data to output directory."""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get list of supported output formats."""
        pass
    
    @abstractmethod
    def register_formatter(self, formatter: 'IOutputFormatter') -> None:
        """Register an output formatter."""
        pass


class IOutputFormatter(ABC):
    """Abstract interface for output formatters."""
    
    @abstractmethod
    def get_name(self) -> str:
        """Get formatter name."""
        pass
    
    @abstractmethod
    def get_extension(self) -> str:
        """Get file extension."""
        pass
    
    @abstractmethod
    def format_data(self, data: Any, output_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Format and save data."""
        pass


class IDataProcessor(ABC):
    """Abstract interface for data processing operations."""
    
    @abstractmethod
    def process(self, data: xr.Dataset, **kwargs) -> xr.Dataset:
        """
        Process a dataset with specified operations.
        
        Args:
            data: Input dataset
            **kwargs: Additional processing parameters
            
        Returns:
            Processed dataset
        """
        pass
    
    @abstractmethod
    def validate_input(self, data: xr.Dataset) -> bool:
        """
        Validate input data format and content.
        
        Args:
            data: Dataset to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass


class IDataLoader(ABC):
    """Abstract interface for data loading operations."""
    
    @abstractmethod
    def load(self, source: Union[str, Path, List[str]], **kwargs) -> xr.Dataset:
        """
        Load data from source(s).
        
        Args:
            source: Data source path(s)
            **kwargs: Additional loading parameters
            
        Returns:
            Loaded dataset
        """
        pass
    
    @abstractmethod
    def supports_format(self, file_path: Union[str, Path]) -> bool:
        """
        Check if loader supports the given file format.
        
        Args:
            file_path: Path to file to check
            
        Returns:
            True if format is supported
        """
        pass


class IEvaluationEngine(ABC):
    """Abstract interface for evaluation engines."""
    
    @abstractmethod
    def evaluate(
        self, 
        simulation: xr.Dataset, 
        reference: xr.Dataset, 
        metrics: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate simulation against reference data.
        
        Args:
            simulation: Simulation dataset
            reference: Reference dataset
            metrics: List of metrics to calculate
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing evaluation results
        """
        pass
    
    @abstractmethod
    def get_supported_metrics(self) -> List[str]:
        """
        Get list of supported metrics.
        
        Returns:
            List of metric names
        """
        pass
    
    @abstractmethod
    def validate_datasets(self, simulation: xr.Dataset, reference: xr.Dataset) -> bool:
        """
        Validate that datasets are compatible for evaluation.
        
        Args:
            simulation: Simulation dataset
            reference: Reference dataset
            
        Returns:
            True if datasets are compatible
        """
        pass


class IMetricsCalculator(ABC):
    """Abstract interface for metrics calculation."""
    
    @abstractmethod
    def calculate(self, simulation: xr.Dataset, reference: xr.Dataset) -> float:
        """
        Calculate metric value.
        
        Args:
            simulation: Simulation data
            reference: Reference data
            
        Returns:
            Calculated metric value
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get metric name.
        
        Returns:
            Metric name
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """
        Get metric description.
        
        Returns:
            Metric description
        """
        pass


class IVisualizationEngine(ABC):
    """Abstract interface for visualization engines."""
    
    @abstractmethod
    def create_plot(
        self, 
        data: Union[xr.Dataset, pd.DataFrame], 
        plot_type: str,
        **kwargs
    ) -> Any:
        """
        Create a plot from data.
        
        Args:
            data: Data to plot
            plot_type: Type of plot to create
            **kwargs: Additional plotting parameters
            
        Returns:
            Plot object or figure
        """
        pass
    
    @abstractmethod
    def save_plot(self, plot: Any, output_path: Union[str, Path], **kwargs) -> None:
        """
        Save plot to file.
        
        Args:
            plot: Plot object to save
            output_path: Output file path
            **kwargs: Additional save parameters
        """
        pass
    
    @abstractmethod
    def get_supported_plot_types(self) -> List[str]:
        """
        Get list of supported plot types.
        
        Returns:
            List of plot type names
        """
        pass


class IConfigurationManager(ABC):
    """Abstract interface for configuration management."""
    
    @abstractmethod
    def load_config(self, config_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            **kwargs: Additional loading parameters
            
        Returns:
            Configuration dictionary
        """
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Get supported configuration file formats.
        
        Returns:
            List of format extensions
        """
        pass


class IOrchestrator(ABC):
    """Abstract interface for workflow orchestration."""
    
    @abstractmethod
    def execute(self, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a workflow based on configuration.
        
        Args:
            workflow_config: Workflow configuration
            
        Returns:
            Execution results
        """
        pass
    
    @abstractmethod
    def validate_workflow(self, workflow_config: Dict[str, Any]) -> bool:
        """
        Validate workflow configuration.
        
        Args:
            workflow_config: Workflow to validate
            
        Returns:
            True if valid
        """
        pass


class IResourceManager(ABC):
    """Abstract interface for resource management."""
    
    @abstractmethod
    def allocate_resources(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Allocate computational resources for a task.
        
        Args:
            task_config: Task configuration
            
        Returns:
            Resource allocation information
        """
        pass
    
    @abstractmethod
    def release_resources(self, allocation_id: str) -> None:
        """
        Release allocated resources.
        
        Args:
            allocation_id: ID of allocation to release
        """
        pass
    
    @abstractmethod
    def get_resource_status(self) -> Dict[str, Any]:
        """
        Get current resource status.
        
        Returns:
            Resource status information
        """
        pass


class BaseProcessor(IDataProcessor):
    """Base implementation of data processor with common functionality."""
    
    def __init__(self, name: str):
        """
        Initialize base processor.
        
        Args:
            name: Processor name
        """
        self.name = name
        self._metadata = {}
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set processor metadata."""
        self._metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get processor metadata."""
        return self._metadata.get(key, default)
    
    def validate_input(self, data: xr.Dataset) -> bool:
        """Basic input validation."""
        if not isinstance(data, xr.Dataset):
            return False
        
        # Check if dataset has data
        if len(data.data_vars) == 0:
            return False
        
        return True
    
    def process(self, data: xr.Dataset, **kwargs) -> xr.Dataset:
        """
        Basic process implementation that returns data unchanged.
        Subclasses should override this method.
        """
        return data


class BaseEvaluator(IEvaluationEngine):
    """Base implementation of evaluation engine with common functionality."""
    
    def __init__(self, name: str):
        """
        Initialize base evaluator.
        
        Args:
            name: Evaluator name
        """
        self.name = name
        self._metrics_registry = {}
    
    def register_metric(self, metric: IMetricsCalculator) -> None:
        """
        Register a metric calculator.
        
        Args:
            metric: Metric calculator to register
        """
        self._metrics_registry[metric.get_name()] = metric
    
    def get_supported_metrics(self) -> List[str]:
        """Get list of registered metrics."""
        return list(self._metrics_registry.keys())
    
    def validate_datasets(self, simulation: xr.Dataset, reference: xr.Dataset) -> bool:
        """Basic dataset validation."""
        # Check if both are datasets
        if not isinstance(simulation, xr.Dataset) or not isinstance(reference, xr.Dataset):
            return False
        
        # Check if they have compatible dimensions
        sim_dims = set(simulation.dims.keys())
        ref_dims = set(reference.dims.keys())
        
        if not sim_dims.intersection(ref_dims):
            return False
        
        return True


class ProcessingPipeline:
    """Pipeline for chaining data processing operations."""
    
    def __init__(self, name: str = "default"):
        """
        Initialize processing pipeline.
        
        Args:
            name: Pipeline name
        """
        self.name = name
        self._processors = []
    
    def add_processor(self, processor: IDataProcessor) -> 'ProcessingPipeline':
        """
        Add a processor to the pipeline.
        
        Args:
            processor: Processor to add
            
        Returns:
            Self for method chaining
        """
        self._processors.append(processor)
        return self
    
    def process(self, data: xr.Dataset, **kwargs) -> xr.Dataset:
        """
        Process data through the entire pipeline.
        
        Args:
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            Processed data
        """
        result = data
        
        for processor in self._processors:
            if not processor.validate_input(result):
                raise ValueError(f"Invalid input for processor: {processor.name}")
            
            result = processor.process(result, **kwargs)
        
        return result
    
    def get_processor_count(self) -> int:
        """Get number of processors in pipeline."""
        return len(self._processors)


class ComponentRegistry:
    """Registry for managing component instances."""
    
    def __init__(self):
        """Initialize component registry."""
        self._components = {}
    
    def register(self, name: str, component: Any, category: str = "default") -> None:
        """
        Register a component.
        
        Args:
            name: Component name
            component: Component instance
            category: Component category
        """
        if category not in self._components:
            self._components[category] = {}
        
        self._components[category][name] = component
    
    def get(self, name: str, category: str = "default") -> Optional[Any]:
        """
        Get a registered component.
        
        Args:
            name: Component name
            category: Component category
            
        Returns:
            Component instance or None if not found
        """
        return self._components.get(category, {}).get(name)
    
    def list_components(self, category: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List all registered components.
        
        Args:
            category: Specific category to list (optional)
            
        Returns:
            Dictionary mapping categories to component names
        """
        if category:
            return {category: list(self._components.get(category, {}).keys())}
        
        return {cat: list(comps.keys()) for cat, comps in self._components.items()}
    
    def unregister(self, name: str, category: str = "default") -> bool:
        """
        Unregister a component.
        
        Args:
            name: Component name
            category: Component category
            
        Returns:
            True if component was removed
        """
        if category in self._components and name in self._components[category]:
            del self._components[category][name]
            return True
        
        return False


# Global component registry instance
component_registry = ComponentRegistry()