import logging
from typing import Any, Dict

import xarray as xr  # noqa: F401 - compatibility re-export for tests/monkeypatching
from joblib import Parallel, delayed  # noqa: F401 - compatibility re-export for tests/monkeypatching

from openbench.data._processing_base import BaseProcessingMixin
from openbench.data._processing_grid import GridProcessingMixin
from openbench.data._processing_grid_regrid import REGRID_ALGORITHM_VERSION, REGRID_BACKENDS  # noqa: F401
from openbench.data._processing_selection import SelectionMixin
from openbench.data._processing_station import StationProcessingMixin
from openbench.data._processing_time import TimeIntegrityMixin
from openbench.util.converttype import Convert_Type  # noqa: F401 - compatibility re-export
from openbench.util.interfaces import BaseProcessor
from openbench.util.netcdf import write_netcdf_atomic as _write_netcdf_atomic  # noqa: F401

logger = logging.getLogger(__name__)
_HAS_INTERFACES = True

# Import caching system (required for data processing)
try:
    from openbench.data.cache import DataCache, get_cache_manager
except ImportError:
    raise ImportError(
        "CacheSystem is required for data processing modules. "
        "Please ensure openbench.data.cache is available. "
        "This module provides essential caching functionality for data processing performance."
    )


# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.getLogger("xarray").setLevel(logging.WARNING)


class BaseDatasetProcessing(BaseProcessingMixin, SelectionMixin, TimeIntegrityMixin, BaseProcessor):
    def __init__(self, config: Dict[str, Any]):
        # Initialize base processor if available
        if _HAS_INTERFACES:
            BaseProcessor.__init__(self, name=config.get("name", "BaseDatasetProcessing"))

        self.initialize_attributes(config)
        self.setup_output_directories()
        self.initialize_resource_parameters()


class StationDatasetProcessing(StationProcessingMixin, BaseDatasetProcessing):
    """Station processing implementation assembled from focused mixins."""


class GridDatasetProcessing(GridProcessingMixin, BaseDatasetProcessing):
    """Grid processing implementation assembled from focused mixins."""


class DatasetProcessing(StationDatasetProcessing, GridDatasetProcessing):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Cache system will be initialized on-demand to avoid serialization issues
        self._cache_initialized = False

    def _get_cache(self):
        """Get cache manager with lazy initialization."""
        if not self._cache_initialized:
            try:
                self.cache_manager = get_cache_manager()
                self.data_cache = DataCache(self.cache_manager)
                self._cache_initialized = True
                logging.debug("Cache system lazily initialized")
            except Exception as e:
                logging.error(f"Failed to initialize required cache system: {e}")
                raise RuntimeError(f"CacheSystem initialization failed: {e}")
        return self.cache_manager

    def prepare_source(self, datasource: str) -> None:
        super().prepare_source(datasource)
        # Add any additional processing specific to this class if needed
