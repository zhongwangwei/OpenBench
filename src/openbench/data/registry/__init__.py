"""Dataset registry and model profiles.

Public API:
    RegistryManager - loads and queries reference datasets and model profiles
    ReferenceDataset, ModelProfile, VariableMapping - schema dataclasses
"""

from openbench.data.registry.manager import RegistryManager
from openbench.data.registry.schema import ModelProfile, ReferenceDataset, VariableMapping

__all__ = [
    "RegistryManager",
    "ReferenceDataset",
    "ModelProfile",
    "VariableMapping",
]
