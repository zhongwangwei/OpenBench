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

from openbench.data.registry.schema import (
    FallbackVar, ModelProfile, ReferenceDataset, StationMatchingConfig, VariableMapping,
)

logger = logging.getLogger(__name__)

# The single authoritative registry directory (inside the package)
REGISTRY_DIR = Path(__file__).parent


def _get_user_dir() -> Path:
    """Return the user config directory (~/.openbench)."""
    return Path.home() / ".openbench"


def get_writable_registry_dir() -> Path:
    """Return the registry directory for writing.

    If the package registry directory is writable, use it directly.
    Otherwise fall back to the user directory.
    """
    if os.access(REGISTRY_DIR, os.W_OK):
        return REGISTRY_DIR
    fallback = _get_user_dir()
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def get_writable_model_catalog_path() -> Path:
    """Return the path for writing model catalog entries.

    Writable install: <package>/data/registry/model_catalog.yaml
    Read-only install: ~/.openbench/models/model_catalog.yaml
    """
    if os.access(REGISTRY_DIR, os.W_OK):
        return REGISTRY_DIR / "model_catalog.yaml"
    fallback = _get_user_dir() / "models"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback / "model_catalog.yaml"


def get_writable_reference_catalog_path() -> Path:
    """Return the path for writing reference catalog entries.

    Writable install: <package>/data/registry/reference_catalog.yaml
    Read-only install: ~/.openbench/references/reference_catalog.yaml
    """
    if os.access(REGISTRY_DIR, os.W_OK):
        return REGISTRY_DIR / "reference_catalog.yaml"
    fallback = _get_user_dir() / "references"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback / "reference_catalog.yaml"


_REGISTRY_CACHE: Optional["RegistryManager"] = None

_UNRESOLVED_ENV_VARS_WARNED: set = set()


def _expand_env_path(value, context: str = "") -> Optional[str]:
    """Expand $VAR / ${VAR} in a path string and warn (once) if unresolved.

    The bundled reference_catalog.yaml uses ${OPENBENCH_REF_ROOT}/... so
    paths are not hard-coded to a single developer's machine. End users
    override by setting the env var. If the var is unset the literal
    placeholder is returned so the resulting "file not found" error at
    I/O time clearly names the missing variable rather than producing a
    silently-empty path.
    """
    if not isinstance(value, str) or "$" not in value:
        return value
    expanded = os.path.expandvars(value)
    if "$" in expanded:
        # Extract the first unresolved var for the warning
        import re as _re
        match = _re.search(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", expanded)
        var_name = match.group(1) if match else "<unknown>"
        # Dedupe by var name so one missing env var = one warning, not 100+
        if var_name not in _UNRESOLVED_ENV_VARS_WARNED:
            _UNRESOLVED_ENV_VARS_WARNED.add(var_name)
            logger.warning(
                "Reference catalog references env var $%s which is unset. "
                "Set it (e.g., export %s=/path/to/data) so registry paths "
                "resolve. First unresolved entry: %s",
                var_name, var_name, context or value,
            )
    return expanded


def get_registry() -> "RegistryManager":
    """Get a cached RegistryManager instance (avoids re-reading YAML on every call)."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        _REGISTRY_CACHE = RegistryManager()
    return _REGISTRY_CACHE


def clear_registry_cache() -> None:
    """Clear the cached instance (e.g., after registering new datasets)."""
    global _REGISTRY_CACHE
    _REGISTRY_CACHE = None


class RegistryManager:
    """Manages reference dataset and model profile descriptors."""

    def __init__(self, user_dir: Optional[Path] = None):
        self._references: dict[str, ReferenceDataset] = {}
        self._models: dict[str, ModelProfile] = {}
        self._var_index: dict[str, list[str]] = {}  # variable → [ref_keys]

        # Primary: package registry directory
        self._load_reference_catalog(REGISTRY_DIR / "reference_catalog.yaml")
        self._load_reference_dir(REGISTRY_DIR / "references")
        self._load_model_catalog(REGISTRY_DIR / "model_catalog.yaml")
        self._load_model_dir(REGISTRY_DIR / "models")

        # User overlay: deep-merge user directory on top of built-in
        # (only if different from package dir and exists)
        if user_dir is None:
            user_dir = Path.home() / ".openbench"

        if user_dir.exists() and user_dir.resolve() != REGISTRY_DIR.resolve():
            # ~/.openbench/references/reference_catalog.yaml + individual YAML files
            self._merge_reference_catalog(user_dir / "references" / "reference_catalog.yaml")
            self._merge_reference_dir(user_dir / "references")
            # ~/.openbench/models/model_catalog.yaml + individual YAML files
            self._merge_model_catalog(user_dir / "models" / "model_catalog.yaml")
            self._merge_model_dir(user_dir / "models")

        # Build variable → references index (O(1) lookup instead of O(n) scan)
        self._build_var_index()

    def _build_var_index(self) -> None:
        """Pre-build variable → reference keys index for fast lookup."""
        self._var_index.clear()
        for key, ref in self._references.items():
            for var_name in ref.variables:
                self._var_index.setdefault(var_name, []).append(key)

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

    # --- User overlay (deep merge) ---

    def _merge_reference_catalog(self, path: Path) -> None:
        """Deep-merge user reference catalog on top of built-in entries."""
        if not path.exists():
            return
        try:
            with open(path) as f:
                catalog = yaml.safe_load(f) or {}
            for name, data in catalog.items():
                try:
                    key = name.lower()
                    if key in self._references:
                        self._references[key] = _deep_merge_reference(self._references[key], data)
                        logger.debug("Merged user overlay for reference '%s'", name)
                    else:
                        ref = _build_reference(data)
                        self._references[key] = ref
                        logger.debug("Added new user reference '%s'", name)
                except Exception as e:
                    logger.warning("Failed to merge user reference '%s': %s", name, e)
        except Exception as e:
            logger.warning("Failed to read user reference catalog %s: %s", path, e)

    def _merge_reference_dir(self, directory: Path) -> None:
        """Deep-merge individual user reference YAML files."""
        if not directory.exists():
            return
        for path in sorted(directory.glob("*.yaml")):
            try:
                with open(path) as f:
                    data = yaml.safe_load(f)
                if not data:
                    continue
                # Support both {Name: {...}} and {name: ..., variables: ...} formats
                if "name" in data:
                    entries = {data["name"]: data}
                else:
                    entries = data
                for name, entry in entries.items():
                    key = name.lower()
                    if key in self._references:
                        self._references[key] = _deep_merge_reference(self._references[key], entry)
                        logger.debug("Merged user overlay for reference '%s' from %s", name, path.name)
                    else:
                        ref = _build_reference(entry)
                        self._references[key] = ref
                        logger.debug("Added new user reference '%s' from %s", name, path.name)
            except Exception as e:
                logger.warning("Failed to merge user reference from %s: %s", path.name, e)

    def _merge_model_catalog(self, path: Path) -> None:
        """Deep-merge user model catalog on top of built-in entries."""
        if not path.exists():
            return
        try:
            with open(path) as f:
                catalog = yaml.safe_load(f) or {}
            for name, data in catalog.items():
                try:
                    key = name.lower()
                    if key in self._models:
                        self._models[key] = _deep_merge_model(self._models[key], data)
                        logger.debug("Merged user overlay for model '%s'", name)
                    else:
                        m = _build_model(data)
                        self._models[key] = m
                        logger.debug("Added new user model '%s'", name)
                except Exception as e:
                    logger.warning("Failed to merge user model '%s': %s", name, e)
        except Exception as e:
            logger.warning("Failed to read user model catalog %s: %s", path, e)

    def _merge_model_dir(self, directory: Path) -> None:
        """Deep-merge individual user model YAML files."""
        if not directory.exists():
            return
        for path in sorted(directory.glob("*.yaml")):
            try:
                with open(path) as f:
                    data = yaml.safe_load(f)
                if not data:
                    continue
                if "name" in data:
                    entries = {data["name"]: data}
                else:
                    entries = data
                for name, entry in entries.items():
                    key = name.lower()
                    if key in self._models:
                        self._models[key] = _deep_merge_model(self._models[key], entry)
                        logger.debug("Merged user overlay for model '%s' from %s", name, path.name)
                    else:
                        m = _build_model(entry)
                        self._models[key] = m
                        logger.debug("Added new user model '%s' from %s", name, path.name)
            except Exception as e:
                logger.warning("Failed to merge user model from %s: %s", path.name, e)

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

        After a call, ``last_resolve_reason`` contains the decision trace
        (empty string for exact matches, human-readable for auto-resolve).
        """
        # Case-insensitive lookup
        key = name.lower()

        # Try alias first
        key = self.REFERENCE_ALIASES.get(key, key)

        # Exact match — always wins
        if key in self._references:
            self.last_resolve_reason = ""
            return self._references[key]

        # Base name — try auto-resolve if sim resolution is provided
        variants = self.get_resolution_variants(key)
        if not variants:
            self.last_resolve_reason = ""
            return None

        if sim_tim_res is None and sim_grid_res is None:
            # No simulation context — can't auto-resolve
            self.last_resolve_reason = ""
            return None

        ref, reason = _auto_resolve_variant(
            variants, sim_tim_res=sim_tim_res, sim_grid_res=sim_grid_res
        )
        self.last_resolve_reason = reason
        return ref

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
        base_key = self.REFERENCE_ALIASES.get(base_key, base_key)

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

    # Reserved for future use — add entries like {"old_name": "new_name"} when needed
    REFERENCE_ALIASES: dict[str, str] = {}

    # Alias map: alternative names → canonical name (all lowercase)
    MODEL_ALIASES: dict[str, str] = {
        "colm": "colm2024",
        "cama-flood": "cama",
        "camaflood": "cama",
        "cama_flood": "cama",
    }

    def get_model(self, name: str) -> Optional[ModelProfile]:
        key = name.lower()
        # Try direct lookup first
        result = self._models.get(key)
        if result:
            return result
        # Try alias
        canonical = self.MODEL_ALIASES.get(key)
        if canonical:
            return self._models.get(canonical)
        return None

    def references_for_variable(self, variable: str) -> list[ReferenceDataset]:
        """Get all reference datasets that support a given variable (O(1) index lookup)."""
        return [self._references[k] for k in self._var_index.get(variable, []) if k in self._references]

    # --- Write methods ---

    def save_model(self, name: str, profile: ModelProfile) -> None:
        """Save or update a model profile to the catalog."""
        catalog_path = get_writable_model_catalog_path()
        catalog = self._read_catalog(catalog_path)
        catalog[name] = profile.to_dict()
        self._write_catalog(catalog_path, catalog)
        self._models[name.lower()] = profile
        logger.info("Saved model '%s' to %s", name, catalog_path)

    def delete_model(self, name: str) -> None:
        """Delete a model profile from the catalog."""
        catalog_path = get_writable_model_catalog_path()
        catalog = self._read_catalog(catalog_path)
        removed = catalog.pop(name, None) or catalog.pop(name.lower(), None)
        if removed is not None:
            self._write_catalog(catalog_path, catalog)
        self._models.pop(name.lower(), None)
        logger.info("Deleted model '%s'", name)

    def save_reference(self, name: str, dataset: ReferenceDataset) -> None:
        """Save or update a reference dataset to the catalog."""
        catalog_path = get_writable_reference_catalog_path()
        catalog = self._read_catalog(catalog_path)
        catalog[name] = dataset.to_dict()
        self._write_catalog(catalog_path, catalog)
        self._references[name.lower()] = dataset
        self._build_var_index()
        logger.info("Saved reference '%s' to %s", name, catalog_path)

    def delete_reference(self, name: str) -> None:
        """Delete a reference dataset from the catalog."""
        catalog_path = get_writable_reference_catalog_path()
        catalog = self._read_catalog(catalog_path)
        removed = catalog.pop(name, None) or catalog.pop(name.lower(), None)
        if removed is not None:
            self._write_catalog(catalog_path, catalog)
        self._references.pop(name.lower(), None)
        self._build_var_index()
        logger.info("Deleted reference '%s'", name)

    @staticmethod
    def _read_catalog(path: Path) -> dict:
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f) or {}
        return {}

    @staticmethod
    def _write_catalog(path: Path, catalog: dict) -> None:
        import tempfile
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                yaml.dump(catalog, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            os.replace(tmp, str(path))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


def _auto_resolve_variant(
    variants: dict[str, ReferenceDataset],
    sim_tim_res: Optional[str] = None,
    sim_grid_res: Optional[float] = None,
) -> tuple[Optional[ReferenceDataset], str]:
    """Pick the best reference variant matching simulation resolution.

    Rules:
      1. Time frequency must be >= simulation (ref can be same or higher, not lower)
      2. Among valid time matches, prefer closest spatial resolution to simulation
      3. Tie-break on spatial: prefer lowest sufficient time frequency (avoid waste)

    Returns:
        (best_ref, reason) — reason is a human-readable decision trace.
    """
    from openbench.data.registry.scanner import _tim_res_rank

    sim_rank = _tim_res_rank(sim_tim_res) if sim_tim_res else -1
    reasons = []

    # Step 1: filter to refs with sufficient time frequency
    candidates = []
    for label, ref in variants.items():
        ref_rank = _tim_res_rank(ref.tim_res)
        if sim_rank >= 0 and ref_rank < sim_rank:
            continue  # Too low frequency — skip
        candidates.append((label, ref, ref_rank))

    if not candidates:
        logger.warning(
            "Auto-resolve: no variant has sufficient time frequency for sim_tim_res=%s. "
            "Falling back to all %d variants.",
            sim_tim_res, len(variants),
        )
        candidates = [(l, r, _tim_res_rank(r.tim_res)) for l, r in variants.items()]
        reasons.append(f"no variant >= {sim_tim_res}, fell back to all {len(variants)}")

    if not candidates:
        return None, "no candidates available"

    if len(candidates) < len(variants):
        dropped = len(variants) - len(candidates)
        reasons.append(f"filtered {dropped}/{len(variants)} variants (tim_res < {sim_tim_res})")

    # Step 2: score by spatial closeness + time waste penalty
    def _score(item):
        label, ref, ref_rank = item
        if not sim_grid_res:
            spatial_diff = 0.0
        elif ref.grid_res:
            spatial_diff = abs(ref.grid_res - sim_grid_res)
        else:
            spatial_diff = 999.0

        time_excess = max(0, ref_rank - sim_rank) if sim_rank >= 0 else 0
        return (spatial_diff, time_excess)

    candidates.sort(key=_score)

    # Prefer variants whose root_dir actually exists on disk
    best_label, best, best_rank = candidates[0]
    chosen_reason = f"best match: {best.name} (tim_res={best.tim_res}, grid_res={best.grid_res})"

    if best.root_dir and not Path(best.root_dir).is_dir():
        for _, ref, _ in candidates[1:]:
            if ref.root_dir and Path(ref.root_dir).is_dir():
                logger.info(
                    "Auto-resolve: preferred %s has no data on disk, using %s instead",
                    best.name, ref.name,
                )
                reasons.append(f"{best.name} not on disk, switched to {ref.name}")
                return ref, "; ".join(reasons + [f"selected {ref.name}"])
        logger.warning("Auto-resolve: %s selected but root_dir not found: %s", best.name, best.root_dir)
        reasons.append(f"{best.name} selected but root_dir missing")

    reasons.append(chosen_reason)
    return best, "; ".join(reasons)


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


def _normalize_legacy_varname_list(var_data: dict) -> tuple[str, list[FallbackVar] | None]:
    """Normalize legacy list-valued varname entries into primary + fallbacks."""
    raw_varname = var_data["varname"]
    parsed_fallbacks = _parse_fallbacks(var_data.get("fallbacks")) or []

    if not isinstance(raw_varname, list):
        return raw_varname, parsed_fallbacks or None

    if not raw_varname:
        return "", parsed_fallbacks or None

    primary = raw_varname[0]
    legacy_fallbacks = [
        FallbackVar(varname=name, varunit=var_data.get("varunit", ""))
        for name in raw_varname[1:]
    ]
    combined = legacy_fallbacks + parsed_fallbacks
    return primary, combined or None


def _build_reference(data: dict) -> ReferenceDataset:
    """Build a ReferenceDataset from a raw dict."""
    variables = {}
    for var_name, var_data in data.get("variables", {}).items():
        primary_varname, fallbacks = _normalize_legacy_varname_list(var_data)
        variables[var_name] = VariableMapping(
            varname=primary_varname,
            varunit=var_data.get("varunit", ""),
            prefix=var_data.get("prefix", ""),
            suffix=var_data.get("suffix", ""),
            sub_dir=var_data.get("sub_dir"),
            fulllist=var_data.get("fulllist"),
            max_uparea=var_data.get("max_uparea"),
            min_uparea=var_data.get("min_uparea"),
            fallbacks=fallbacks,
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

    # Parse station_matching config if present
    sm_data = data.get("station_matching")
    station_matching = None
    if sm_data and isinstance(sm_data, dict):
        station_matching = StationMatchingConfig(
            method=sm_data.get("method", "cama_allocation"),
            dataset_file=sm_data.get("dataset_file", ""),
            station_id_var=sm_data.get("station_id_var", "station"),
            lon_var=sm_data.get("lon_var", "lon"),
            lat_var=sm_data.get("lat_var", "lat"),
            area_var=sm_data.get("area_var", "area"),
            discharge_var=sm_data.get("discharge_var", "discharge"),
            time_var=sm_data.get("time_var", "time"),
            area_error_threshold=float(sm_data.get("area_error_threshold", 0.2)),
            min_uparea=float(sm_data.get("min_uparea", 1000.0)),
            max_uparea=float(sm_data.get("max_uparea", float("inf"))),
            time_format=sm_data.get("time_format"),
        )

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
        fulllist=_expand_env_path(data.get("fulllist"), context=f"{name}.fulllist"),
        root_dir=_expand_env_path(data.get("root_dir"), context=f"{name}.root_dir"),
        station_matching=station_matching,
        _provenance=data.get("_provenance"),
    )


def _build_model(data: dict) -> ModelProfile:
    """Build a ModelProfile from a raw dict."""
    variables = {}
    for var_name, var_data in data.get("variables", {}).items():
        primary_varname, fallbacks = _normalize_legacy_varname_list(var_data)
        variables[var_name] = VariableMapping(
            varname=primary_varname,
            varunit=var_data.get("varunit", ""),
            prefix=var_data.get("prefix", ""),
            suffix=var_data.get("suffix", ""),
            sub_dir=var_data.get("sub_dir"),
            fallbacks=fallbacks,
            compute=var_data.get("compute"),
            prefix_fallback=var_data.get("prefix_fallback"),
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


def _deep_merge_model(existing: ModelProfile, overlay: dict) -> ModelProfile:
    """Deep-merge a user overlay dict into an existing ModelProfile.

    - Scalar fields (description, data_type, etc.): overlay wins if present
    - variables: per-variable merge — new vars added, existing vars overwritten
    - time_offset: deep-merge dict keys
    """
    # Start from existing values
    name = overlay.get("name", existing.name)
    description = overlay.get("description", existing.description)
    data_type = overlay.get("data_type", existing.data_type)
    grid_res = overlay.get("grid_res", existing.grid_res)
    tim_res = overlay.get("tim_res", existing.tim_res)

    # Deep-merge variables: keep all existing, overlay updates/adds
    variables = dict(existing.variables)
    for var_name, var_data in overlay.get("variables", {}).items():
        primary_varname, fallbacks = _normalize_legacy_varname_list(var_data)
        variables[var_name] = VariableMapping(
            varname=primary_varname,
            varunit=var_data.get("varunit", ""),
            prefix=var_data.get("prefix", ""),
            suffix=var_data.get("suffix", ""),
            sub_dir=var_data.get("sub_dir"),
            fallbacks=fallbacks,
            compute=var_data.get("compute"),
            prefix_fallback=var_data.get("prefix_fallback"),
        )

    # Deep-merge time_offset
    time_offset = dict(existing.time_offset) if existing.time_offset else {}
    if "time_offset" in overlay:
        user_to = overlay["time_offset"]
        if user_to is None:
            time_offset = {}
        elif isinstance(user_to, dict):
            for k, v in user_to.items():
                if isinstance(v, dict) and isinstance(time_offset.get(k), dict):
                    time_offset[k] = {**time_offset[k], **v}
                else:
                    time_offset[k] = v

    return ModelProfile(
        name=name,
        description=description,
        data_type=data_type,
        grid_res=grid_res,
        tim_res=tim_res,
        variables=variables,
        time_offset=time_offset or None,
    )


def _deep_merge_reference(existing: ReferenceDataset, overlay: dict) -> ReferenceDataset:
    """Deep-merge a user overlay dict into an existing ReferenceDataset.

    - Scalar fields: overlay wins if present
    - variables: per-variable merge — new vars added, existing vars overwritten
    - _provenance / station_matching: rebuild from overlay if present, else
      keep the existing object so a second-pass dir-glob load doesn't strip
      these fields when re-merging the catalog file.
    """
    name = overlay.get("name", existing.name)
    description = overlay.get("description", existing.description)
    category = overlay.get("category", existing.category)
    data_type = overlay.get("data_type", existing.data_type)
    tim_res = overlay.get("tim_res", existing.tim_res)
    data_groupby = overlay.get("data_groupby", existing.data_groupby)
    timezone = overlay.get("timezone", existing.timezone)
    years = overlay.get("years", existing.years)
    grid_res = overlay.get("grid_res", existing.grid_res)
    fulllist = _expand_env_path(overlay.get("fulllist", existing.fulllist), context=f"{name}.fulllist")
    root_dir = _expand_env_path(overlay.get("root_dir", existing.root_dir), context=f"{name}.root_dir")
    provenance = overlay.get("_provenance", existing._provenance)

    # Rebuild station_matching only if overlay supplies one; otherwise keep
    # the already-parsed object on `existing` (overlay station_matching is a
    # raw dict at this point and would need re-parsing).
    sm_overlay = overlay.get("station_matching")
    if isinstance(sm_overlay, dict):
        station_matching = StationMatchingConfig(
            method=sm_overlay.get("method", "cama_allocation"),
            dataset_file=sm_overlay.get("dataset_file", ""),
            station_id_var=sm_overlay.get("station_id_var", "station"),
            lon_var=sm_overlay.get("lon_var", "lon"),
            lat_var=sm_overlay.get("lat_var", "lat"),
            area_var=sm_overlay.get("area_var", "area"),
            discharge_var=sm_overlay.get("discharge_var", "discharge"),
            time_var=sm_overlay.get("time_var", "time"),
            area_error_threshold=float(sm_overlay.get("area_error_threshold", 0.2)),
            min_uparea=float(sm_overlay.get("min_uparea", 1000.0)),
            max_uparea=float(sm_overlay.get("max_uparea", float("inf"))),
            time_format=sm_overlay.get("time_format"),
        )
    else:
        station_matching = existing.station_matching

    # Deep-merge variables
    variables = dict(existing.variables)
    for var_name, var_data in overlay.get("variables", {}).items():
        primary_varname, fallbacks = _normalize_legacy_varname_list(var_data)
        variables[var_name] = VariableMapping(
            varname=primary_varname,
            varunit=var_data.get("varunit", ""),
            prefix=var_data.get("prefix", ""),
            suffix=var_data.get("suffix", ""),
            sub_dir=var_data.get("sub_dir"),
            fulllist=var_data.get("fulllist"),
            max_uparea=var_data.get("max_uparea"),
            min_uparea=var_data.get("min_uparea"),
            fallbacks=fallbacks,
            compute=var_data.get("compute"),
        )

    return ReferenceDataset(
        name=name,
        description=description,
        category=category,
        data_type=data_type,
        tim_res=tim_res,
        data_groupby=data_groupby,
        timezone=timezone,
        years=years,
        variables=variables,
        grid_res=grid_res,
        fulllist=fulllist,
        root_dir=root_dir,
        station_matching=station_matching,
        _provenance=provenance,
    )
