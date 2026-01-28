"""
Core modules for WSE Pipeline
"""

from .station import Station, StationList
from .geoid_calculator import GeoidCalculator
from .cama_allocator import CamaAllocator, StationAllocation, AllocationResult, HAS_ALLOCATE_VS

__all__ = [
    'Station',
    'StationList',
    'GeoidCalculator',
    'CamaAllocator',
    'StationAllocation',
    'AllocationResult',
    'HAS_ALLOCATE_VS',
]

# Conditionally export AllocateVS if available
try:
    from .allocate_vs import AllocateVS
    __all__.append('AllocateVS')
except ImportError:
    pass
