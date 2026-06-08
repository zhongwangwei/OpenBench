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
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import xarray as xr


class BaseComponent:
    """Minimal base class for OpenBench components.

    Provides a name + metadata convention that ModularOutputManager and
    similar component classes extend. Previously the name was referenced
    by util/output.py without being defined here, which silently disabled
    type checking via the import-fallback path.
    """

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self._metadata: Dict[str, Any] = {}

    def get_metadata(self) -> Dict[str, Any]:
        """Return component metadata."""
        return dict(self._metadata)


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
    def register_formatter(self, formatter: "IOutputFormatter") -> None:
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
