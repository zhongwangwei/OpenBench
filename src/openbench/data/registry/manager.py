"""RegistryManager: loads and queries reference datasets and model profiles.

All catalogs live in one place: the registry directory inside the package
(src/openbench/data/registry/). User registrations are written directly
to the same catalog files when the directory is writable (editable install).
If the package directory is read-only (pip install), a fallback user
directory is used.

Loading order (later entries override earlier):
1. Built-in catalog:  <package>/data/registry/reference_catalog.yaml
2. Fallback user dir: ~/.openbench/reference_catalog.yaml  (only if exists)
3. Fallback individuals: ~/.openbench/references/*.yaml    (only if exists)
Same for model_catalog.yaml / models/*.yaml.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import yaml

from openbench.data.registry.schema import FallbackVar, ModelProfile, ReferenceDataset, VariableMapping

logger = logging.getLogger(__name__)

# The single authoritative registry directory (inside the package)
REGISTRY_DIR = Path(__file__).parent


def get_writable_registry_dir() -> Path:
    """Return the registry directory for writing.

    If the package registry directory is writable, use it directly.
    Otherwise fall back to a user directory.
    """
    if os.access(REGISTRY_DIR, os.W_OK):
        return REGISTRY_DIR

    # Fallback for read-only installs
    try:
        from platformdirs import user_config_dir

        fallback = Path(user_config_dir("openbench"))
    except ImportError:
        fallback = Path.home() / ".openbench"

    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


class RegistryManager:
    """Manages reference dataset and model profile descriptors."""

    def __init__(self, user_dir: Optional[Path] = None):
        self._references: dict[str, ReferenceDataset] = {}
        self._models: dict[str, ModelProfile] = {}

        # Primary: package registry directory
        self._load_reference_catalog(REGISTRY_DIR / "reference_catalog.yaml")
        self._load_reference_dir(REGISTRY_DIR / "references")
        self._load_model_catalog(REGISTRY_DIR / "model_catalog.yaml")
        self._load_model_dir(REGISTRY_DIR / "models")

        # Fallback: user directory (only if different from package dir and exists)
        if user_dir is None:
            try:
                from platformdirs import user_config_dir

                user_dir = Path(user_config_dir("openbench"))
            except ImportError:
                user_dir = Path.home() / ".openbench"

        if user_dir.exists() and user_dir.resolve() != REGISTRY_DIR.resolve():
            self._load_reference_catalog(user_dir / "reference_catalog.yaml")
            self._load_reference_dir(user_dir / "references")
            self._load_model_catalog(user_dir / "model_catalog.yaml")
            self._load_model_dir(user_dir / "models")

    # --- Loading ---

    def _load_reference_catalog(self, path: Path) -> None:
        """Load all references from a single catalog YAML file."""
        if not path.exists():
            return
        try:
            with open(path) as f:
                catalog = yaml.safe_load(f) or {}
            for name, data in catalog.items():
                try:
                    ref = _build_reference(data)
                    self._references[name.lower()] = ref
                except Exception as e:
                    logger.warning("Failed to load reference '%s' from %s: %s", name, path.name, e)
        except Exception as e:
            logger.warning("Failed to read reference catalog %s: %s", path, e)

    def _load_reference_dir(self, directory: Path) -> None:
        """Load references from individual YAML files in a directory."""
        if not directory.exists():
            return
        for path in sorted(directory.glob("*.yaml")):
            try:
                with open(path) as f:
                    data = yaml.safe_load(f)
                if data and "name" in data:
                    ref = _build_reference(data)
                    self._references[data["name"].lower()] = ref
            except Exception as e:
                logger.warning("Failed to load reference from %s: %s", path.name, e)

    def _load_model_catalog(self, path: Path) -> None:
        """Load all models from a single catalog YAML file."""
        if not path.exists():
            return
        try:
            with open(path) as f:
                catalog = yaml.safe_load(f) or {}
            for name, data in catalog.items():
                try:
                    m = _build_model(data)
                    self._models[name.lower()] = m
                except Exception as e:
                    logger.warning("Failed to load model '%s' from %s: %s", name, path.name, e)
        except Exception as e:
            logger.warning("Failed to read model catalog %s: %s", path, e)

    def _load_model_dir(self, directory: Path) -> None:
        """Load models from individual YAML files in a directory."""
        if not directory.exists():
            return
        for path in sorted(directory.glob("*.yaml")):
            try:
                with open(path) as f:
                    data = yaml.safe_load(f)
                if data and "name" in data:
                    m = _build_model(data)
                    self._models[data["name"].lower()] = m
            except Exception as e:
                logger.warning("Failed to load model from %s: %s", path.name, e)

    # --- Resolution suffixes ---
    RESOLUTION_SUFFIXES = ("_LowRes", "_MidRes", "_HigRes")

    # --- Queries ---

    def list_references(self) -> list[ReferenceDataset]:
        return sorted(self._references.values(), key=lambda r: r.name)

    def get_reference(
        self,
        name: str,
        sim_tim_res: Optional[str] = None,
        sim_grid_res: Optional[float] = None,
    ) -> Optional[ReferenceDataset]:
        """Get a reference dataset by name, with optional auto-resolve.

        Exact name (e.g., 'GLEAM_v4.2a_LowRes') always returns directly.

        Base name (e.g., 'GLEAM_v4.2a') with resolution variants:
          - If sim_tim_res/sim_grid_res provided: picks the best match
          - Otherwise: returns None (use get_resolution_variants to see options)

        Auto-resolve rules:
          1. Time: ref frequency must be >= simulation frequency
          2. Space: prefer closest grid_res to simulation
          3. Tie-break: prefer lowest sufficient frequency (avoid waste)
        """
        # Case-insensitive lookup
        key = name.lower()

        # Exact match — always wins
        if key in self._references:
            return self._references[key]

        # Base name — try auto-resolve if sim resolution is provided
        variants = self.get_resolution_variants(name)
        if not variants:
            return None

        if sim_tim_res is None and sim_grid_res is None:
            # No simulation context — can't auto-resolve
            return None

        return _auto_resolve_variant(
            variants, sim_tim_res=sim_tim_res, sim_grid_res=sim_grid_res
        )

    def get_resolution_variants(self, base_name: str) -> dict[str, ReferenceDataset]:
        """Find all resolution variants of a dataset.

        Args:
            base_name: Base dataset name without resolution suffix (e.g., 'GLEAM_v4.2a')

        Returns:
            Dict mapping resolution label to ReferenceDataset.
            E.g., {'LowRes': ..., 'MidRes': ..., 'HigRes': ...}
        """
        variants = {}
        base_key = base_name.lower()

        for suffix in self.RESOLUTION_SUFFIXES:
            full_key = f"{base_key}{suffix.lower()}"
            if full_key in self._references:
                label = suffix[1:]  # Strip leading underscore
                variants[label] = self._references[full_key]

        # Also check if base_name itself is a standalone entry (no resolution suffix)
        if base_key in self._references and not variants:
            variants["default"] = self._references[base_key]

        return variants

    def list_models(self) -> list[ModelProfile]:
        return sorted(self._models.values(), key=lambda m: m.name)

    def get_model(self, name: str) -> Optional[ModelProfile]:
        return self._models.get(name.lower())

    def references_for_variable(self, variable: str) -> list[ReferenceDataset]:
        return [ref for ref in self._references.values() if variable in ref.variables]


def _auto_resolve_variant(
    variants: dict[str, ReferenceDataset],
    sim_tim_res: Optional[str] = None,
    sim_grid_res: Optional[float] = None,
) -> Optional[ReferenceDataset]:
    """Pick the best reference variant matching simulation resolution.

    Rules:
      1. Time frequency must be >= simulation (ref can be same or higher, not lower)
      2. Among valid time matches, prefer closest spatial resolution to simulation
      3. Tie-break on spatial: prefer lowest sufficient time frequency (avoid waste)

    Examples:
      sim=Month/0.5° → prefer ref Month/0.5° over Day/0.1°
      sim=Day/0.5°   → must use ref Day or higher; prefer Day/0.5° over Hour/0.1°
      sim=Month/0.1° → prefer ref Month/0.1° (or closest available)
    """
    from openbench.data.registry.scanner import _tim_res_rank

    sim_rank = _tim_res_rank(sim_tim_res) if sim_tim_res else -1

    # Step 1: filter to refs with sufficient time frequency
    candidates = []
    for label, ref in variants.items():
        ref_rank = _tim_res_rank(ref.tim_res)
        if sim_rank >= 0 and ref_rank < sim_rank:
            continue  # Too low frequency — skip
        candidates.append((label, ref, ref_rank))

    if not candidates:
        # No variant has sufficient time frequency — fall back to all
        candidates = [(l, r, _tim_res_rank(r.tim_res)) for l, r in variants.items()]

    if not candidates:
        return None

    # Step 2: score by spatial closeness + time waste penalty
    def _score(item):
        label, ref, ref_rank = item
        # Spatial distance: how far is ref.grid_res from sim.grid_res
        ref_grid = ref.grid_res or 0.25
        sim_grid = sim_grid_res or 0.25
        spatial_diff = abs(ref_grid - sim_grid)

        # Time waste: higher frequency than needed is wasteful
        # Lower penalty = better
        time_excess = max(0, ref_rank - sim_rank) if sim_rank >= 0 else 0

        # Combined: spatial closeness is primary, time waste is secondary
        return (spatial_diff, time_excess)

    candidates.sort(key=_score)
    return candidates[0][1]


def _parse_fallbacks(raw_list: list | None) -> list[FallbackVar] | None:
    """Parse fallback variable definitions from YAML."""
    if not raw_list:
        return None
    return [
        FallbackVar(
            varname=fb["varname"],
            varunit=fb.get("varunit", ""),
            convert=fb.get("convert", ""),
        )
        for fb in raw_list
        if isinstance(fb, dict) and "varname" in fb
    ]


def _build_reference(data: dict) -> ReferenceDataset:
    """Build a ReferenceDataset from a raw dict."""
    variables = {}
    for var_name, var_data in data.get("variables", {}).items():
        variables[var_name] = VariableMapping(
            varname=var_data["varname"],
            varunit=var_data.get("varunit", ""),
            prefix=var_data.get("prefix", ""),
            suffix=var_data.get("suffix", ""),
            sub_dir=var_data.get("sub_dir"),
            fulllist=var_data.get("fulllist"),
            max_uparea=var_data.get("max_uparea"),
            min_uparea=var_data.get("min_uparea"),
            fallbacks=_parse_fallbacks(var_data.get("fallbacks")),
        )

    # Validate required fields
    name = data.get("name")
    if not name:
        raise ValueError("Reference dataset missing 'name' field")
    data_type = data.get("data_type")
    if not data_type:
        raise ValueError(f"Reference '{name}' missing 'data_type' field")
    tim_res = data.get("tim_res")
    if not tim_res:
        raise ValueError(f"Reference '{name}' missing 'tim_res' field")

    # Validate grid_res
    grid_res = data.get("grid_res")
    if grid_res is not None:
        grid_res = float(grid_res)
        if grid_res <= 0:
            logger.warning("Reference '%s' has invalid grid_res=%s, ignoring", name, grid_res)
            grid_res = None

    # Validate years
    years = data.get("years", [])
    if years and len(years) >= 2 and years[0] > years[1]:
        logger.warning("Reference '%s' has start year > end year: %s", name, years)

    return ReferenceDataset(
        name=name,
        description=data.get("description", ""),
        category=data.get("category", ""),
        data_type=data_type,
        tim_res=tim_res,
        data_groupby=data.get("data_groupby", "Year"),
        timezone=data.get("timezone", 0),
        years=years,
        variables=variables,
        grid_res=grid_res,
        fulllist=data.get("fulllist"),
        root_dir=data.get("root_dir"),
    )


def _build_model(data: dict) -> ModelProfile:
    """Build a ModelProfile from a raw dict."""
    variables = {}
    for var_name, var_data in data.get("variables", {}).items():
        variables[var_name] = VariableMapping(
            varname=var_data["varname"],
            varunit=var_data.get("varunit", ""),
            prefix=var_data.get("prefix", ""),
            suffix=var_data.get("suffix", ""),
            sub_dir=var_data.get("sub_dir"),
            fallbacks=_parse_fallbacks(var_data.get("fallbacks")),
            compute=var_data.get("compute"),
        )

    return ModelProfile(
        name=data["name"],
        description=data.get("description", ""),
        data_type=data.get("data_type", "grid"),
        grid_res=data.get("grid_res"),
        tim_res=data.get("tim_res", "Month"),
        variables=variables,
        time_offset=data.get("time_offset"),
    )
