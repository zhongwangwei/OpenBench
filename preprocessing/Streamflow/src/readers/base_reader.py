"""Abstract base reader for all data source readers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import logging

from ..models import StationDataset


class BaseReader(ABC):
    """Base class for dataset readers."""

    source_name: str = "unknown"

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def read_all(self, config: Dict) -> List[StationDataset]:
        """Read all data from this source.

        Returns one StationDataset per time resolution found.
        E.g., GRDC returns [StationDataset(daily), StationDataset(monthly)].
        """
        pass
