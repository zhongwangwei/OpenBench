"""Station processing helper facade."""

from __future__ import annotations

from openbench.data._processing_station_core import StationProcessingCoreMixin
from openbench.data._processing_station_extract import StationExtractionMixin


class StationProcessingMixin(StationProcessingCoreMixin, StationExtractionMixin):
    """Station dataset preprocessing and grid-to-station extraction helpers."""

    pass
