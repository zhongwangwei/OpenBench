from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class Station:
    """Virtual station data structure."""
    id: str
    name: str
    lon: float
    lat: float
    source: str
    elevation: float = 0.0
    num_observations: int = 0
    egm08: Optional[float] = None
    egm96: Optional[float] = None
    cama_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if station coordinates are valid."""
        return -180 <= self.lon <= 180 and -90 <= self.lat <= 90

    def set_cama_result(self, resolution: str, result: Dict[str, Any]):
        """Set CaMa allocation result for a resolution."""
        self.cama_results[resolution] = result


class StationList:
    """Collection of stations with filtering and grouping."""

    def __init__(self):
        self._stations: List[Station] = []

    def add(self, station: Station):
        self._stations.append(station)

    def __len__(self):
        return len(self._stations)

    def __iter__(self):
        return iter(self._stations)

    def filter_by_source(self, source: str) -> 'StationList':
        """Filter stations by source."""
        result = StationList()
        for s in self._stations:
            if s.source == source:
                result.add(s)
        return result

    def filter_valid(self) -> 'StationList':
        """Filter only valid stations."""
        result = StationList()
        for s in self._stations:
            if s.is_valid():
                result.add(s)
        return result

    def get_sources(self) -> List[str]:
        """Get unique sources."""
        return list(set(s.source for s in self._stations))
