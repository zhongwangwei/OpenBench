"""Model-specific time-offset rules with actual-file provenance."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
import xarray as xr

from openbench.data._processing_utils import parse_time_offset as _parse_time_offset

logger = logging.getLogger(__name__)

_RESOLUTION_NAMES = {
    "h": "Hour",
    "hr": "Hour",
    "hour": "Hour",
    "hourly": "Hour",
    "d": "Day",
    "day": "Day",
    "daily": "Day",
    "m": "Month",
    "me": "Month",
    "mon": "Month",
    "month": "Month",
    "monthly": "Month",
    "y": "Year",
    "ye": "Year",
    "yr": "Year",
    "year": "Year",
    "yearly": "Year",
    "annual": "Year",
    "a": "Year",
}
_RESERVED = frozenset({"Hour", "Day", "Month", "Year", "default"})


def _variable_override(group, native_name, item_name):
    """Return an explicit variable override, ignoring default fallbacks."""
    if not isinstance(group, dict):
        return None
    for name in (native_name, item_name):
        if not name:
            continue
        for key, value in group.items():
            if str(key).lower() == "default" or value is None:
                continue
            if name in (part.strip().casefold() for part in str(key).split(",")):
                return str(value)
    return None


def _group_offset(group, resolution, native_name, item_name):
    if isinstance(group, str):
        return group
    if not isinstance(group, dict):
        return None
    nested = group.get(resolution)
    if nested is not None:
        return _group_offset(nested, resolution, native_name, item_name)
    override = _variable_override(group, native_name, item_name)
    if override is not None:
        return override
    default = group.get("default")
    return str(default) if isinstance(default, str) else None


def resolve_time_offset(profile, resolution, ds, item, source_files=None, file_context=None):
    """Return (offset, rule) without losing the actual NetCDF provenance."""
    offsets = profile.time_offset or {}
    native_name = ds.name if isinstance(ds, xr.DataArray) else next(iter(ds.data_vars), "")
    native = str(native_name or "").strip().casefold()
    logical = str(item or "").strip().casefold()
    generic = offsets.get(resolution)
    explicit = _variable_override(generic, native, logical)

    if isinstance(source_files, (str, Path)):
        files = [str(source_files)]
    else:
        files = [str(x) for x in (source_files or [])]
    # Without an actual path, preserve the legacy config-prefix matching for
    # non-CoLM profiles. It is not used when the real file path is supplied.
    files = files or ([str(file_context)] if file_context else [])
    choices = []
    for filename in files:
        basename = Path(filename).name.casefold()
        patterns = sorted(
            (key for key, val in offsets.items() if key not in _RESERVED and isinstance(val, dict)),
            key=lambda x: (-len(x), x),
        )
        matching = next((key for key in patterns if key.casefold() in basename), None)
        if explicit is not None:
            choices.append((explicit, "variable"))
        elif matching is not None:
            selected = _group_offset(offsets[matching], resolution, native, logical)
            choices.append((selected if selected is not None else "0", matching))
        else:
            choices.append((None, "unmatched"))

    chosen = {(value, rule) for value, rule in choices if value is not None}
    # Reject only different effective offsets; one or more provenance rules
    # yielding exactly the same correction is safe to combine.
    effective_offsets = {value for value, _ in chosen}
    if len(effective_offsets) > 1:
        raise ValueError(f"Mixed time-offset sources in one dataset: {choices}")

    if chosen and all(value is not None for value, _ in choices):
        return choices[0]
    if chosen and any(value is None for value, _ in choices):
        # A mixed group with both known and unknown provenance is unsafe.
        raise ValueError(f"Mixed known and unknown time-offset sources: {choices}")

    if explicit is not None:
        return explicit, "variable"
    generic_value = _group_offset(generic, resolution, native, logical)
    if generic_value is not None:
        return generic_value, resolution
    fallback = _group_offset(offsets.get("default"), resolution, native, logical)
    return (fallback if fallback is not None else "0"), "default"


class TimeAdjustmentMixin:
    """Shift timestamps according to the source model and actual input file."""

    def apply_model_specific_time_adjustment(
        self,
        ds: xr.Dataset,
        datasource: str,
        syear: int,
        eyear: int,
        tim_res: str,
        *,
        source_files=None,
    ) -> xr.Dataset:
        source = self.sim_source if datasource == "sim" else self.ref_source
        model = getattr(self, f"{source}_model", source)

        from openbench.data.registry.manager import get_registry

        profile = get_registry().get_model(model)
        if source_files is None:
            # Retain the historical no-provenance normalization for other
            # models and legacy direct callers. Actual-file CoLM data is
            # never rewritten to an artificial HH:30 before offset resolution.
            try:
                ds = ds.assign_coords(time=pd.to_datetime(ds["time"].dt.strftime("%Y-%m-%dT%H:30:00")))
            except (ValueError, AttributeError, TypeError):
                logger.debug("Legacy time normalization skipped for %s", model)
        if not profile or not profile.time_offset:
            return ds

        match = re.match(r"\d*\s*([a-zA-Z]+)", str(tim_res).strip())
        resolution = _RESOLUTION_NAMES.get(match.group(1).lower()) if match else None
        if resolution is None:
            raise ValueError(f"Unsupported time resolution for model offset: {tim_res!r}")

        context = f"{getattr(self, f'{datasource}_dir', '')}/{getattr(self, f'{datasource}_prefix', '')}"
        offset_str, rule = resolve_time_offset(
            profile,
            resolution,
            ds,
            getattr(self, "item", ""),
            source_files=source_files,
            file_context=context,
        )

        # Preserve original timestamps when no offset is selected. A blanket
        # HH:30 normalization changes even CoLM's already centred daily time.
        if offset_str.strip() in {"0", "+0", "-0"}:
            logger.debug(
                "Time offset model=%s item=%s rule=%s offset=0 files=%s",
                model,
                getattr(self, "item", ""),
                rule,
                source_files,
            )
            return ds

        offset = _parse_time_offset(offset_str)
        ds = ds.assign_coords(time=pd.DatetimeIndex(ds["time"].values) + offset)
        logger.info(
            "Time offset model=%s item=%s rule=%s offset=%s files=%s",
            model,
            getattr(self, "item", ""),
            rule,
            offset_str,
            source_files,
        )
        return ds
