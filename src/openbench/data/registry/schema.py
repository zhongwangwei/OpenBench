"""Dataclasses for registry descriptors (reference datasets and model profiles)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class VariableMapping:
    """How a variable is stored in a dataset or model output.

    varname can be a string or a list of strings (fallback chain).
    E.g., varname=["f_gpp", "f_assim"] means try f_gpp first, fall back to f_assim.
    """

    varname: Union[str, list[str]]
    varunit: str
    prefix: str = ""
    suffix: str = ""
    sub_dir: Optional[str] = None
    fulllist: Optional[str] = None
    max_uparea: Optional[float] = None
    min_uparea: Optional[float] = None


@dataclass
class ReferenceDataset:
    """Descriptor for a reference dataset in the registry."""

    name: str
    description: str
    category: str  # Water, Carbon, Energy, Meteorology, Crop, Urban, Lake
    data_type: str  # grid, stn
    tim_res: str
    data_groupby: str
    timezone: int | float
    years: list[int]  # [start, end]
    variables: dict[str, VariableMapping]
    grid_res: Optional[float] = None
    fulllist: Optional[str] = None
    root_dir: Optional[str] = None  # Set when data is downloaded/located


@dataclass
class ModelProfile:
    """Descriptor for a simulation model's variable mappings."""

    name: str
    description: str
    data_type: str = "grid"
    grid_res: Optional[float] = None
    tim_res: str = "Month"
    variables: dict[str, VariableMapping] = field(default_factory=dict)
