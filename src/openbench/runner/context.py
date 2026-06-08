"""Runtime context construction for local-runner tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openbench.config.adapter import BridgeRuntimeInfo

BRIDGE_RUNTIME_FIELDS = {
    "casedir",
    "ref_varname",
    "sim_varname",
    "ref_data_type",
    "sim_data_type",
    "compare_tim_res",
    "compare_grid_res",
    "compare_tzone",
    "regrid_backend",
    "unified_mask",
}


@dataclass(frozen=True)
class RuntimeContext:
    """Runner-owned context layered on top of bridge-provided fields."""

    bridge_info: BridgeRuntimeInfo
    ref_source: str
    sim_source: str
    ref_file_override: str | None = None

    def to_info(self) -> dict[str, Any]:
        info = self.bridge_info.to_info()
        info["ref_source"] = self.ref_source
        info["sim_source"] = self.sim_source
        if self.ref_file_override:
            info["ref_file_override"] = self.ref_file_override
        return info


def coerce_bridge_runtime_info(bridge_info: BridgeRuntimeInfo | dict[str, Any]) -> BridgeRuntimeInfo:
    """Normalize adapter bridge payloads to the typed runtime-info wrapper."""
    if isinstance(bridge_info, BridgeRuntimeInfo):
        return bridge_info
    return BridgeRuntimeInfo(payload=dict(bridge_info))


def build_runtime_context(task: dict[str, Any]) -> RuntimeContext:
    """Build runner-owned runtime context without mutating reader state."""
    bridge_info = coerce_bridge_runtime_info(
        task["bindings"].build_runtime_info_for(task["var_name"], task["sim_source"], task["ref_source"])
    )

    return RuntimeContext(
        bridge_info=bridge_info,
        ref_source=task["ref_source"],
        sim_source=task["sim_source"],
        ref_file_override=task.get("ref_file_override"),
    )


def build_bridge_runtime_info(task: dict[str, Any]) -> dict[str, Any]:
    """Build the runner-owned bridge info dict for one task."""
    return build_runtime_context(task).to_info()
