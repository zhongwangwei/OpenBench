"""Time-coordinate validation helpers for dataset processing."""

from __future__ import annotations

import logging
import re

import pandas as pd
import xarray as xr

from openbench.data._processing_utils import parse_time_offset as _parse_time_offset

logger = logging.getLogger(__name__)


class TimeAdjustmentMixin:
    """Split temporal processing helpers."""

    def apply_model_specific_time_adjustment(
        self, ds: xr.Dataset, datasource: str, syear: int, eyear: int, tim_res: str
    ) -> xr.Dataset:
        # Get model name from _model attribute (e.g., TE-routing_model = "TE")
        source = self.sim_source if datasource == "sim" else self.ref_source
        try:
            model = getattr(self, f"{source}_model")
        except AttributeError:
            model = source
        # Universal time adjustment from model profile YAML

        # Step 1: normalize timestamp format
        try:
            ds["time"] = pd.to_datetime(ds["time"].dt.strftime("%Y-%m-%dT%H:30:00"))
        except Exception:
            logging.debug("Time normalization skipped for %s", model)

        # Step 2: apply model-specific time offset from profile
        #
        # time_offset YAML structure:
        #   default:           # applies to all files
        #     Month: "-1 months"
        #     Day: "-1 days"
        #   _cama_:            # applies when file path contains "_cama_"
        #     Day: "-1 days"
        #
        # Resolution priority: file-path pattern match > default
        try:
            from openbench.data.registry.manager import get_registry

            mgr = get_registry()
            profile = mgr.get_model(model)
            if profile and profile.time_offset:
                match = re.match(r"(\d*)\s*([a-zA-Z]+)", tim_res)
                if match:
                    _, time_unit = match.groups()
                    key_map = {
                        "m": "Month",
                        "me": "Month",
                        "month": "Month",
                        "mon": "Month",
                        "d": "Day",
                        "day": "Day",
                        "h": "Hour",
                        "hour": "Hour",
                        "y": "Year",
                        "year": "Year",
                    }
                    res_key = key_map.get(time_unit.lower(), "")

                    # Build file context string for pattern matching
                    current_prefix = getattr(self, f"{datasource}_prefix", "")
                    current_dir = getattr(self, f"{datasource}_dir", "")
                    file_context = f"{current_dir}/{current_prefix}"

                    def _offset_from_resolution_group(group):
                        """Resolve either a string or {item-pattern/default: offset} group."""
                        if isinstance(group, str):
                            return group
                        if not isinstance(group, dict):
                            return None

                        item_name = str(getattr(self, "item", "")).strip().lower()
                        for item_key, item_offset in group.items():
                            if str(item_key).lower() == "default":
                                continue
                            item_names = [part.strip().lower() for part in str(item_key).split(",")]
                            if item_name and item_name in item_names:
                                return str(item_offset)
                        if "default" in group:
                            return str(group["default"])
                        return None

                    def _offset_from_pattern_group(group):
                        """Resolve {Month/Day/Hour: offset} or nested resolution groups."""
                        if isinstance(group, str):
                            return group
                        if not isinstance(group, dict):
                            return None
                        if res_key and res_key in group:
                            return _offset_from_resolution_group(group[res_key])
                        return _offset_from_resolution_group(group)

                    # Supported YAML shapes:
                    # 1. Per-resolution legacy/catalog:
                    #      Month: "-15 days"
                    #      Day: {default: "-1 days", Streamflow: "0"}
                    # 2. Pattern-grouped:
                    #      default: {Month: "-1 months", Day: "-1 days"}
                    #      _cama_: {Day: "-1 days"}
                    offset_str = None

                    if isinstance(profile.time_offset, dict):
                        if res_key and res_key in profile.time_offset:
                            offset_str = _offset_from_resolution_group(profile.time_offset[res_key])

                        if offset_str is None:
                            for group_key, group_val in profile.time_offset.items():
                                if group_key in {"default", res_key}:
                                    continue
                                if isinstance(group_val, dict) and str(group_key) in file_context:
                                    offset_str = _offset_from_pattern_group(group_val)
                                    logging.debug(
                                        "Time offset: matched file pattern '%s' in '%s'",
                                        group_key,
                                        file_context,
                                    )
                                    break

                        if offset_str is None:
                            offset_str = _offset_from_pattern_group(profile.time_offset.get("default"))

                    if offset_str is None:
                        offset_str = "0"

                    if offset_str and offset_str != "0":
                        offset = _parse_time_offset(offset_str)
                        if offset:
                            ds["time"] = pd.DatetimeIndex(ds["time"].values) + offset
                            logging.debug(
                                "Applied time offset %s for %s/%s (%s)",
                                offset_str,
                                model,
                                getattr(self, "item", "?"),
                                res_key,
                            )
        except Exception as e:
            logging.debug("Time offset skipped for %s: %s", model, e)

        return ds
