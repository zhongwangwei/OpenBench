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

    def to_dict(self) -> dict:
        d: dict = {"varname": self.varname}
        if self.varunit:
            d["varunit"] = self.varunit
        if self.convert:
            d["convert"] = self.convert
        return d


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
    compute: Optional[str] = None  # Python expression to compute variable from dataset
    prefix_fallback: Optional[list[str]] = None  # Alternative file prefix suffixes, e.g., ["_cama_", "_unitcat_"]

    def to_dict(self) -> dict:
        d: dict = {"varname": self.varname, "varunit": self.varunit}
        if self.prefix:
            d["prefix"] = self.prefix
        if self.suffix:
            d["suffix"] = self.suffix
        if self.sub_dir:
            d["sub_dir"] = self.sub_dir
        if self.fulllist:
            d["fulllist"] = self.fulllist
        if self.max_uparea is not None:
            d["max_uparea"] = self.max_uparea
        if self.min_uparea is not None:
            d["min_uparea"] = self.min_uparea
        if self.fallbacks:
            d["fallbacks"] = [fb.to_dict() for fb in self.fallbacks]
        if self.compute:
            d["compute"] = self.compute
        if self.prefix_fallback:
            d["prefix_fallback"] = self.prefix_fallback
        return d


@dataclass
class StationMatchingConfig:
    """Configuration for the built-in station matching engine.

    Stored in reference_catalog.yaml under ``station_matching:``.

    Attributes:
        method: Matching algorithm — ``cama_allocation`` or ``direct``
        dataset_file: NC file name in the dataset root directory
        station_id_var: Variable name for station IDs
        lon_var: Variable name for station longitudes
        lat_var: Variable name for station latitudes
        area_var: Variable name for upstream areas (empty string if none)
        discharge_var: Variable name for discharge/streamflow data
        time_var: Variable name for time coordinate
        area_error_threshold: Max fractional CaMA allocation error
        min_uparea: Minimum upstream area (km²)
        max_uparea: Maximum upstream area (km²)
        time_format: Special time format (e.g. ``YYYYMM``) or None
    """

    method: str = "cama_allocation"
    dataset_file: str = ""
    station_id_var: str = "station"
    lon_var: str = "lon"
    lat_var: str = "lat"
    area_var: str = "area"
    discharge_var: str = "discharge"
    time_var: str = "time"
    area_error_threshold: float = 0.2
    min_uparea: float = 1000.0
    max_uparea: float = float("inf")
    time_format: Optional[str] = None


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
    station_matching: Optional[StationMatchingConfig] = None
    _provenance: Optional[dict] = None  # field → source ("profile"/"scan"/"default"/"nc"/"existing")

    def to_dict(self) -> dict:
        d: dict = {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "data_type": self.data_type,
            "tim_res": self.tim_res,
            "data_groupby": self.data_groupby,
            "timezone": self.timezone,
        }
        if self.root_dir:
            d["root_dir"] = self.root_dir
        if self.years:
            d["years"] = self.years
        if self.grid_res is not None:
            d["grid_res"] = self.grid_res
        if self.fulllist:
            d["fulllist"] = self.fulllist
        d["variables"] = {name: vm.to_dict() for name, vm in self.variables.items()}
        return d


@dataclass
class ModelProfile:
    """Descriptor for a simulation model's variable mappings.

    time_offset: per-resolution time shift applied to timestamps.
        Example: {"Month": "-15 days", "Day": "-1 days"}
        Supports: "N days", "N months", "N hours"
    """

    name: str
    description: str
    data_type: str = "grid"
    grid_res: Optional[float] = None
    tim_res: str = "Month"
    variables: dict[str, VariableMapping] = field(default_factory=dict)
    time_offset: Optional[dict[str, str]] = None  # {"Month": "-15 days", "Day": "0"}

    def to_dict(self) -> dict:
        d: dict = {
            "name": self.name,
            "description": self.description,
            "data_type": self.data_type,
        }
        if self.grid_res is not None:
            d["grid_res"] = self.grid_res
        if self.tim_res:
            d["tim_res"] = self.tim_res
        d["variables"] = {name: vm.to_dict() for name, vm in self.variables.items()}
        if self.time_offset:
            d["time_offset"] = self.time_offset
        return d
