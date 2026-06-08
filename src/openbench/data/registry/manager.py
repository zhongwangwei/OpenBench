"""RegistryManager: loads and queries reference datasets and model profiles.

All catalogs live in one place: the registry directory inside the package
(src/openbench/data/registry/). User registrations are written directly
to the same catalog files when the directory is writable (editable install).
If the package directory is read-only (pip install), a fallback user
directory is used.

Loading order (later entries override earlier):
1. Built-in catalog:  <package>/data/registry/reference_catalog.yaml
2. Fallback user dir: ~/.openbench/references/reference_catalog.yaml  (only if exists)
3. Fallback individuals: ~/.openbench/references/*.yaml    (only if exists)
Same for model_catalog.yaml / models/*.yaml.
"""

from __future__ import annotations

from copy import deepcopy
from importlib.resources import files
import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml

from openbench.config.user_settings import resolve_reference_root
from openbench.data.registry.schema import (
    FallbackVar,
    ModelProfile,
    ReferenceDataset,
    StationMatchingConfig,
    VariableMapping,
)
from openbench.util.names import get_mapping_key_case_insensitive, normalize_name

logger = logging.getLogger(__name__)

# The single authoritative registry directory (inside the package). Keep this
# as an importlib.resources Traversable; converting it to Path breaks when
# OpenBench is imported directly from a zipped wheel.
REGISTRY_DIR = files("openbench.data.registry")

_RESERVED_REFERENCE_FILES = {"reference_catalog.yaml", "reference_profiles.yaml"}
_RESERVED_MODEL_FILES = {"model_catalog.yaml", "aliases.yaml"}
_MODEL_EQUIVALENT_ALIASES = {
    "colm": ("colm2024", "CoLM"),
}


def canonical_model_key(name: str) -> str:
    """Return the canonical registry key for model names that are behavioral aliases."""
    key = normalize_name(name)
    return _MODEL_EQUIVALENT_ALIASES.get(key, (key, ""))[0]


def _get_user_dir() -> Path:
    """Return the user config directory (~/.openbench)."""
    return Path.home() / ".openbench"


def get_writable_model_catalog_path() -> Path:
    """Return the user overlay path for writing model catalog entries.

    The package registry remains the bundled default catalog. User-created
    or edited model profiles are always written to ~/.openbench/models/.
    """
    fallback = _get_user_dir() / "models"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback / "model_catalog.yaml"


def get_writable_reference_catalog_path() -> Path:
    """Return the user overlay path for writing reference catalog entries.

    The package registry remains the bundled default catalog. User-created
    or scanned reference entries are always written to ~/.openbench/references/.
    """
    fallback = _get_user_dir() / "references"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback / "reference_catalog.yaml"


def get_writable_reference_profiles_path() -> Path:
    """Return the user overlay path for writing reference profile entries.

    The package registry remains the bundled default profile set. User-created
    scanner profiles are always written to ~/.openbench/references/.
    """
    fallback = _get_user_dir() / "references"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback / "reference_profiles.yaml"


def get_legacy_reference_profiles_path() -> Path:
    """Return the old user profile path kept for read compatibility."""
    return _get_user_dir() / "reference_profiles.yaml"


_REGISTRY_CACHE: Optional["RegistryManager"] = None

_UNRESOLVED_ENV_VARS_WARNED: set = set()


def _is_file_resource(path: Any) -> bool:
    """Return True for either a filesystem Path or a Traversable file."""
    if hasattr(path, "is_file"):
        return path.is_file()
    return Path(path).is_file()


def _is_dir_resource(path: Any) -> bool:
    """Return True for either a filesystem Path or a Traversable directory."""
    if hasattr(path, "is_dir"):
        return path.is_dir()
    return Path(path).is_dir()


def _iter_yaml_resources(directory: Any) -> list[Any]:
    """List YAML children from a filesystem Path or Traversable directory."""
    if not _is_dir_resource(directory):
        return []
    if hasattr(directory, "glob"):
        return sorted(directory.glob("*.yaml"))
    return sorted(
        (
            child
            for child in directory.iterdir()
            if getattr(child, "name", "").endswith(".yaml") and _is_file_resource(child)
        ),
        key=lambda child: child.name,
    )


def _open_yaml_resource(path: Any):
    """Open a filesystem Path or Traversable YAML resource as text."""
    return path.open("r", encoding="utf-8")


def _same_concrete_path(left: Any, right: Any) -> bool:
    """Best-effort concrete path comparison, false for virtual resources."""
    try:
        return Path(left).resolve() == Path(right).resolve()
    except (TypeError, ValueError, OSError):
        return False


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
    if "OPENBENCH_REF_ROOT" in value and "OPENBENCH_REF_ROOT" not in os.environ:
        ref_root = resolve_reference_root()
        if ref_root:
            value = value.replace("${OPENBENCH_REF_ROOT}", ref_root).replace("$OPENBENCH_REF_ROOT", ref_root)
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
                var_name,
                var_name,
                context or value,
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
        self._model_aliases: dict[str, str] = dict(self.MODEL_ALIASES)
        self._var_index: dict[str, list[str]] = {}  # variable → [ref_keys]
        self.last_resolve_reason = ""

        # Primary: package registry directory
        self._load_reference_catalog(REGISTRY_DIR / "reference_catalog.yaml")
        self._load_reference_dir(REGISTRY_DIR / "references")
        self._load_model_catalog(REGISTRY_DIR / "model_catalog.yaml")
        self._load_model_dir(REGISTRY_DIR / "models")

        # User overlay: deep-merge user directory on top of built-in
        # (only if different from package dir and exists)
        if user_dir is None:
            user_dir = Path.home() / ".openbench"

        if user_dir.exists() and not _same_concrete_path(user_dir, REGISTRY_DIR):
            # ~/.openbench/references/reference_catalog.yaml + individual YAML files
            self._merge_reference_catalog(user_dir / "references" / "reference_catalog.yaml")
            self._merge_reference_dir(user_dir / "references")
            # ~/.openbench/models/model_catalog.yaml + individual YAML files
            self._merge_model_catalog(user_dir / "models" / "model_catalog.yaml")
            self._merge_model_dir(user_dir / "models")
            self._load_user_model_aliases(user_dir / "models" / "aliases.yaml")

        self._sync_model_equivalent_aliases()

        # Build variable → references index (O(1) lookup instead of O(n) scan)
        self._build_var_index()

    def _sync_model_equivalent_aliases(self) -> None:
        """Keep legacy model names behaviorally identical to their canonical profile."""
        for alias_key, (source_key, alias_name) in _MODEL_EQUIVALENT_ALIASES.items():
            source = self._models.get(source_key)
            if source is None:
                continue
            alias = deepcopy(source)
            alias.name = alias_name
            self._models[alias_key] = alias

    def _build_var_index(self) -> None:
        """Pre-build variable → reference keys index for fast lookup."""
        self._var_index.clear()
        for key, ref in self._references.items():
            for var_name in ref.variables:
                self._var_index.setdefault(normalize_name(var_name), []).append(key)

    # --- Loading ---

    def _load_reference_catalog(self, path: Any) -> None:
        """Load all references from a single catalog YAML file."""
        if not _is_file_resource(path):
            return
        try:
            with _open_yaml_resource(path) as f:
                catalog = yaml.safe_load(f) or {}
            for name, data in catalog.items():
                try:
                    ref = _build_reference(data)
                    self._references[normalize_name(name)] = ref
                except Exception as e:
                    logger.warning("Failed to load reference '%s' from %s: %s", name, path.name, e)
        except Exception as e:
            logger.warning("Failed to read reference catalog %s: %s", path, e)

    def _load_reference_dir(self, directory: Any) -> None:
        """Load references from individual YAML files in a directory."""
        for path in _iter_yaml_resources(directory):
            try:
                with _open_yaml_resource(path) as f:
                    data = yaml.safe_load(f)
                if data and "name" in data:
                    ref = _build_reference(data)
                    self._references[normalize_name(data["name"])] = ref
            except Exception as e:
                logger.warning("Failed to load reference from %s: %s", path.name, e)

    def _load_model_catalog(self, path: Any) -> None:
        """Load all models from a single catalog YAML file."""
        if not _is_file_resource(path):
            return
        try:
            with _open_yaml_resource(path) as f:
                catalog = yaml.safe_load(f) or {}
            for name, data in catalog.items():
                try:
                    m = _build_model(data)
                    self._models[normalize_name(name)] = m
                except Exception as e:
                    logger.warning("Failed to load model '%s' from %s: %s", name, path.name, e)
        except Exception as e:
            logger.warning("Failed to read model catalog %s: %s", path, e)

    def _load_model_dir(self, directory: Any) -> None:
        """Load models from individual YAML files in a directory."""
        for path in _iter_yaml_resources(directory):
            try:
                with _open_yaml_resource(path) as f:
                    data = yaml.safe_load(f)
                if data and "name" in data:
                    m = _build_model(data)
                    self._models[normalize_name(data["name"])] = m
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
                    key = normalize_name(name)
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
            if path.name in _RESERVED_REFERENCE_FILES:
                continue
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
                    key = normalize_name(name)
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
                    key = normalize_name(name)
                    if isinstance(data, dict) and data.get("_deleted"):
                        self._models.pop(key, None)
                        logger.debug("Deleted model '%s' via user overlay tombstone", name)
                        continue
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

    def _load_user_model_aliases(self, path: Path) -> None:
        """Load user-defined model aliases from ~/.openbench/models/aliases.yaml."""
        if not path.exists():
            return
        try:
            with open(path) as f:
                aliases = yaml.safe_load(f) or {}
            if not isinstance(aliases, dict):
                logger.warning("Model aliases file must be a mapping: %s", path)
                return
            for alias, canonical in aliases.items():
                alias_key = normalize_name(alias)
                canonical_key = normalize_name(canonical)
                if alias_key and canonical_key:
                    self._model_aliases[alias_key] = canonical_key
        except Exception as e:
            logger.warning("Failed to read model aliases %s: %s", path, e)

    def _merge_model_dir(self, directory: Path) -> None:
        """Deep-merge individual user model YAML files."""
        if not directory.exists():
            return
        for path in sorted(directory.glob("*.yaml")):
            if path.name in _RESERVED_MODEL_FILES:
                continue
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
                    key = normalize_name(name)
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
        key = normalize_name(name)

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

        ref, reason = _auto_resolve_variant(variants, sim_tim_res=sim_tim_res, sim_grid_res=sim_grid_res)
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
        base_key = normalize_name(base_name)
        base_key = self.REFERENCE_ALIASES.get(base_key, base_key)

        for suffix in self.RESOLUTION_SUFFIXES:
            full_key = f"{base_key}{normalize_name(suffix)}"
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
        key = normalize_name(name)
        # Try direct lookup first
        result = self._models.get(key)
        if result:
            return result
        # Try alias
        canonical = self._model_aliases.get(key)
        if canonical:
            return self._models.get(canonical)
        return None

    def references_for_variable(self, variable: str) -> list[ReferenceDataset]:
        """Get all reference datasets that support a given variable (O(1) index lookup)."""
        return [self._references[k] for k in self._var_index.get(normalize_name(variable), []) if k in self._references]

    # --- Write methods ---

    def save_model(self, name: str, profile: ModelProfile) -> None:
        """Save or update a model profile to the catalog."""
        catalog_path = get_writable_model_catalog_path()
        catalog = self._read_catalog(catalog_path)
        existing_key = next(
            (candidate for candidate in catalog if normalize_name(candidate) == normalize_name(name)),
            None,
        )
        if existing_key is not None and existing_key != name:
            raise ValueError(
                f"Model name '{name}' conflicts with existing catalog entry '{existing_key}' case-insensitively"
            )
        catalog[name] = profile.to_dict()
        self._write_catalog(catalog_path, catalog)
        self._models[normalize_name(name)] = profile
        logger.info("Saved model '%s' to %s", name, catalog_path)

    def delete_model(self, name: str) -> None:
        """Delete a model profile from the catalog."""
        catalog_path = get_writable_model_catalog_path()
        catalog = self._read_catalog(catalog_path)
        key = normalize_name(name)
        canonical = self._model_aliases.get(canonical_model_key(name), canonical_model_key(name))
        catalog_key = None
        for candidate in list(catalog):
            if normalize_name(candidate) in {key, canonical}:
                catalog_key = candidate
                break
        if catalog_key is not None:
            catalog.pop(catalog_key, None)
            self._write_catalog(catalog_path, catalog)
        elif canonical in self._models:
            profile = self._models[canonical]
            catalog[profile.name] = {"name": profile.name, "_deleted": True}
            self._write_catalog(catalog_path, catalog)
        self._models.pop(canonical, None)
        self._models.pop(key, None)
        logger.info("Deleted model '%s'", name)

    def save_reference(self, name: str, dataset: ReferenceDataset) -> None:
        """Save or update a reference dataset to the catalog."""
        catalog_path = get_writable_reference_catalog_path()
        catalog = self._read_catalog(catalog_path)
        catalog[name] = dataset.to_dict()
        self._write_catalog(catalog_path, catalog)
        self._references[normalize_name(name)] = dataset
        self._build_var_index()
        logger.info("Saved reference '%s' to %s", name, catalog_path)

    def delete_reference(self, name: str) -> None:
        """Delete a reference dataset from the catalog."""
        catalog_path = get_writable_reference_catalog_path()
        catalog = self._read_catalog(catalog_path)
        catalog_key = get_mapping_key_case_insensitive(catalog, name)
        removed = catalog.pop(catalog_key, None) if catalog_key is not None else None
        if removed is not None:
            self._write_catalog(catalog_path, catalog)
        self._references.pop(normalize_name(name), None)
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
            sim_tim_res,
            len(variants),
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
                    best.name,
                    ref.name,
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
    raw_varname = var_data.get("varname", "")
    parsed_fallbacks = _parse_fallbacks(var_data.get("fallbacks")) or []

    if not isinstance(raw_varname, list):
        return raw_varname, parsed_fallbacks or None

    if not raw_varname:
        return "", parsed_fallbacks or None

    primary = raw_varname[0]
    legacy_fallbacks = [FallbackVar(varname=name, varunit=var_data.get("varunit", "")) for name in raw_varname[1:]]
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
            compute=var_data.get("compute"),
            prefix_fallback=var_data.get("prefix_fallback"),
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


def _merge_variable_mapping(
    existing: VariableMapping | None, overlay: dict, *, include_reference_fields: bool
) -> VariableMapping:
    """Merge a partial YAML variable overlay into an existing mapping."""
    merged = existing.to_dict() if existing is not None else {}
    merged.update(overlay)

    primary_varname, fallbacks = _normalize_legacy_varname_list(merged)
    kwargs = {
        "varname": primary_varname,
        "varunit": merged.get("varunit", ""),
        "prefix": merged.get("prefix", ""),
        "suffix": merged.get("suffix", ""),
        "sub_dir": merged.get("sub_dir"),
        "fallbacks": fallbacks,
        "compute": merged.get("compute"),
        "prefix_fallback": merged.get("prefix_fallback"),
    }
    if include_reference_fields:
        kwargs.update(
            {
                "fulllist": merged.get("fulllist"),
                "max_uparea": merged.get("max_uparea"),
                "min_uparea": merged.get("min_uparea"),
            }
        )
    return VariableMapping(**kwargs)


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

    # Deep-merge variables: keep all existing, overlay updates/adds.
    # ``_delete_variables`` is a user-overlay tombstone list used by the CLI to
    # remove variables from bundled profiles, where omission alone cannot mean
    # deletion because overlays are merged.
    variables = dict(existing.variables)
    for deleted in overlay.get("_delete_variables", []) or []:
        delete_key = get_mapping_key_case_insensitive(variables, deleted)
        if delete_key is not None:
            variables.pop(delete_key, None)
    for var_name, var_data in overlay.get("variables", {}).items():
        variable_key = get_mapping_key_case_insensitive(variables, var_name) or var_name
        variables[variable_key] = _merge_variable_mapping(
            variables.get(variable_key),
            var_data,
            include_reference_fields=False,
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
        variable_key = get_mapping_key_case_insensitive(variables, var_name) or var_name
        if var_data is None:
            variables.pop(variable_key, None)
        else:
            variables[variable_key] = _merge_variable_mapping(
                variables.get(variable_key),
                var_data,
                include_reference_fields=True,
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
