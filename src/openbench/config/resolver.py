"""Unified reference resolution — single source of truth.

All entry points (check, adapter, GUI) call these functions instead of
implementing their own resolution logic. This eliminates the "check passes
but runtime binds differently" class of bugs.

Two modes:
  - strict (strict_reference=True):  unresolved references are hard errors.
  - lenient (default):               unresolved references produce warnings
                                      and fall back to minimal defaults.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from openbench.config.schema import OpenBenchConfig

if TYPE_CHECKING:
    from openbench.data.registry.schema import ReferenceDataset, VariableMapping

logger = logging.getLogger(__name__)


@dataclass
class ResolvedReference:
    """Result of resolving a reference for one variable."""

    var_name: str
    source_name: str           # user-specified name (may be base name)
    resolved_name: str         # actual registry entry name after auto-resolve
    ref_ds: Optional[ReferenceDataset]
    var_map: Optional[VariableMapping]
    status: str                # "ok", "not_found", "no_variable", "ambiguous"
    provenance: str = ""       # "registry" = from registry, "fallback" = minimal defaults
    message: str = ""          # human-readable explanation


@dataclass
class TargetResolutionContext:
    """Effective target resolution used for reference auto-resolution."""

    tim_res: Optional[str]
    grid_res: Optional[float]
    source: str
    message: str = ""


def _consistent_value(values: list[object]) -> object | None:
    """Return the shared non-None value if all populated values agree."""
    populated = [v for v in values if v is not None]
    if not populated:
        return None
    first = populated[0]
    if all(v == first for v in populated[1:]):
        return first
    return None


def derive_target_resolution_context(cfg: OpenBenchConfig) -> TargetResolutionContext:
    """Derive the effective target resolution for reference auto-resolve.

    Priority:
      1. User-specified comparison resolution
      2. Shared simulation resolution if all simulations agree
      3. Otherwise: raise a configuration error because the target is ambiguous
    """
    proj_tim_res = cfg.project.tim_res
    proj_grid_res = cfg.project.grid_res
    sim_entries = list(cfg.simulation.values())

    sim_tim_res = _consistent_value([entry.tim_res for entry in sim_entries])
    sim_grid_res = _consistent_value([entry.grid_res for entry in sim_entries])

    target_tim_res = proj_tim_res if proj_tim_res is not None else sim_tim_res
    target_grid_res = proj_grid_res if proj_grid_res is not None else sim_grid_res

    conflicting_tim = proj_tim_res is None and any(
        entry.tim_res is not None and entry.tim_res != sim_tim_res for entry in sim_entries
    )
    conflicting_grid = proj_grid_res is None and any(
        entry.grid_res is not None and entry.grid_res != sim_grid_res for entry in sim_entries
    )

    if conflicting_tim or conflicting_grid:
        from openbench.config import ConfigError

        details = []
        if conflicting_tim:
            details.append(
                "simulation tim_res values differ; set project.tim_res explicitly"
            )
        if conflicting_grid:
            details.append(
                "simulation grid_res values differ; set project.grid_res explicitly"
            )
        raise ConfigError(
            "Reference resolution is ambiguous across simulations: " + "; ".join(details)
        )

    source = "project" if proj_tim_res is not None or proj_grid_res is not None else "simulation"
    return TargetResolutionContext(
        tim_res=target_tim_res,
        grid_res=target_grid_res,
        source=source,
    )


def resolve_reference(
    var_name: str,
    source_name: str,
    registry,
    target_tim_res: Optional[str] = None,
    target_grid_res: Optional[float] = None,
) -> ResolvedReference:
    """Resolve a single variable's reference binding.

    This is the ONE place that decides: given a user-specified source name
    and simulation context, what ReferenceDataset and VariableMapping to use.
    """
    ref_ds = registry.get_reference(
        source_name, sim_tim_res=target_tim_res, sim_grid_res=target_grid_res
    )

    if ref_ds is None:
        variants = registry.get_resolution_variants(source_name)
        if variants:
            variant_names = [v.name for v in variants.values()]
            return ResolvedReference(
                var_name=var_name,
                source_name=source_name,
                resolved_name=source_name,
                ref_ds=None,
                var_map=None,
                status="ambiguous",
                provenance="",
                message=f"'{source_name}' has multiple resolutions: {variant_names}. "
                        f"Specify one explicitly or provide simulation context for auto-resolve.",
            )
        return ResolvedReference(
            var_name=var_name,
            source_name=source_name,
            resolved_name=source_name,
            ref_ds=None,
            var_map=None,
            status="not_found",
            provenance="",
            message=f"'{source_name}' not found in registry.",
        )

    resolved_name = ref_ds.name
    resolve_reason = getattr(registry, "last_resolve_reason", "") or ""
    if resolved_name != source_name:
        logger.info(
            "Reference auto-resolved: %s → %s (tim_res=%s, grid_res=%s) — %s",
            source_name, resolved_name, target_tim_res, target_grid_res, resolve_reason,
        )

    var_map = ref_ds.variables.get(var_name) if hasattr(ref_ds, "variables") else None
    if var_map is None:
        available = list(ref_ds.variables.keys())[:10] if hasattr(ref_ds, "variables") else []
        return ResolvedReference(
            var_name=var_name,
            source_name=source_name,
            resolved_name=resolved_name,
            ref_ds=ref_ds,
            var_map=None,
            status="no_variable",
            provenance="registry",
            message=f"Variable '{var_name}' not in {resolved_name}. Available: {available}",
        )

    # Warn if key metadata fields are from low-confidence sources
    ds_prov = getattr(ref_ds, "_provenance", None)
    if ds_prov is None and hasattr(ref_ds, "__dict__"):
        # _provenance may be stored as a plain dict attribute from YAML loading
        ds_prov = ref_ds.__dict__.get("_provenance")
    prov_warnings = _check_provenance_confidence(resolved_name, ds_prov)

    # Combine auto-resolve reason and provenance warnings into message
    message_parts = []
    if resolve_reason:
        message_parts.append(resolve_reason)
    if prov_warnings:
        message_parts.extend(prov_warnings)

    return ResolvedReference(
        var_name=var_name,
        source_name=source_name,
        resolved_name=resolved_name,
        ref_ds=ref_ds,
        var_map=var_map,
        status="ok",
        provenance="registry",
        message="; ".join(message_parts) if message_parts else "",
    )


# ---------------------------------------------------------------------------
# Provenance confidence tiers
# ---------------------------------------------------------------------------
#
# Each source has a confidence level that determines downstream behaviour:
#   HIGH:   profile, existing, nc  — authoritative, no warnings
#   MEDIUM: scan                   — usable but inferred from directory structure
#   LOW:    default                — fallback only, strict mode treats as error
#
# Fields checked: tim_res, grid_res (the two that most affect downstream binding).

PROVENANCE_HIGH = frozenset({"profile", "existing", "nc"})
PROVENANCE_MEDIUM = frozenset({"scan"})
PROVENANCE_LOW = frozenset({"default"})

# Fields whose provenance matters for binding correctness
_PROVENANCE_FIELDS = ("tim_res", "grid_res")


def _check_provenance_confidence(resolved_name: str, ds_prov: dict | None) -> list[str]:
    """Check provenance of key fields and return warning messages.

    Returns a list of human-readable warning strings (empty if all high-confidence).
    """
    if not isinstance(ds_prov, dict):
        return []

    warnings = []
    for fld in _PROVENANCE_FIELDS:
        source = ds_prov.get(fld)
        if not source:
            continue
        if source in PROVENANCE_LOW:
            warnings.append(f"{fld}: unconfirmed default (register a profile to confirm)")
            logger.debug(
                "Reference %s: %s=%s from '%s' (low confidence)",
                resolved_name, fld, source, source,
            )
        elif source in PROVENANCE_MEDIUM:
            logger.debug(
                "Reference %s: %s inferred from directory structure (medium confidence)",
                resolved_name, fld,
            )
    return warnings


def _strict_provenance_errors(resolved_name: str, ds_prov: dict | None) -> list[str]:
    """Return strict-mode errors for low-confidence provenance fields."""
    warnings = _check_provenance_confidence(resolved_name, ds_prov)
    return [f"{resolved_name}: {warning}" for warning in warnings if "unconfirmed default" in warning]


def resolve_all_references(
    cfg: OpenBenchConfig,
    registry,
    strict: Optional[bool] = None,
) -> list[ResolvedReference]:
    """Resolve all reference bindings for a config.

    Args:
        cfg: The configuration.
        registry: RegistryManager instance.
        strict: If True, unresolved references raise ConfigError.
                If None, reads from cfg.project.strict_reference.
    """
    if strict is None:
        strict = getattr(cfg.project, "strict_reference", False)

    ctx = derive_target_resolution_context(cfg)

    results = []
    errors = []

    for var_name in cfg.evaluation.variables:
        source_value = cfg.reference.sources.get(var_name)
        if not source_value:
            r = ResolvedReference(
                var_name=var_name,
                source_name="",
                resolved_name="",
                ref_ds=None,
                var_map=None,
                status="not_found",
                provenance="",
                message=f"No reference source configured for variable '{var_name}'.",
            )
            results.append(r)
            if strict:
                errors.append(r.message)
            continue

        # Normalize to list — single string and list both produce one or more
        # ResolvedReference entries (one per source). v2.x compatibility.
        if isinstance(source_value, str):
            source_names = [source_value]
        else:
            source_names = list(source_value)

        for source_name in source_names:
            r = resolve_reference(
                var_name, source_name, registry,
                target_tim_res=ctx.tim_res, target_grid_res=ctx.grid_res,
            )

            # In strict mode, non-ok results are hard errors
            if strict and r.status != "ok":
                errors.append(r.message)
            elif strict and r.status == "ok":
                ds_prov = getattr(r.ref_ds, "_provenance", None)
                if ds_prov is None and hasattr(r.ref_ds, "__dict__"):
                    ds_prov = r.ref_ds.__dict__.get("_provenance")
                errors.extend(_strict_provenance_errors(r.resolved_name, ds_prov))

            results.append(r)

    if errors:
        from openbench.config import ConfigError

        msg = "Reference resolution errors (strict_reference=true):\n" + "\n".join(f"  - {e}" for e in errors)
        raise ConfigError(msg)

    return results
