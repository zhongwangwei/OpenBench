"""Dataclasses for registry descriptors (reference datasets and model profiles)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class FallbackVar:
    """A fallback variable with its own unit and conversion expression."""

    varname: str
    varunit: str = ""
    convert: str = ""  # Python expression, e.g., "value * 12.011"


@dataclass
class VariableMapping:
    """How a variable is stored in a dataset or model output.

    Primary variable: varname + varunit.
    Optional fallbacks: list of FallbackVar, tried in order when primary is missing.
    Each fallback has its own unit and a conversion expression to convert
    to the primary unit.

    Example in YAML:
        Gross_Primary_Productivity:
          varname: f_gpp
          varunit: "g m-2 s-1"
          fallbacks:
            - varname: f_assim
              varunit: "mol m-2 s-1"
              convert: "value * 12.011"
    """

    varname: Union[str, list[str]]  # str for primary, list[str] for legacy fallback format
    varunit: str
    prefix: str = ""
    suffix: str = ""
    sub_dir: Optional[str] = None
    fulllist: Optional[str] = None
    max_uparea: Optional[float] = None
    min_uparea: Optional[float] = None
    fallbacks: Optional[list[FallbackVar]] = None  # Fallback variables with unit conversion
    compute: Optional[str] = None  # Python expression to compute from other vars, e.g., "ds['a'] + ds['b']"
    filter: Optional[str] = None  # Custom filter module name for complex computations


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
