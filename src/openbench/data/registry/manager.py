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

import os
from pathlib import Path
from typing import Optional

import yaml

from openbench.data.registry.schema import ModelProfile, ReferenceDataset, VariableMapping

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
                    self._references[name] = _build_reference(data)
                except Exception:
                    pass
        except Exception:
            pass

    def _load_reference_dir(self, directory: Path) -> None:
        """Load references from individual YAML files in a directory."""
        if not directory.exists():
            return
        for path in sorted(directory.glob("*.yaml")):
            try:
                with open(path) as f:
                    data = yaml.safe_load(f)
                if data and "name" in data:
                    self._references[data["name"]] = _build_reference(data)
            except Exception:
                pass

    def _load_model_catalog(self, path: Path) -> None:
        """Load all models from a single catalog YAML file."""
        if not path.exists():
            return
        try:
            with open(path) as f:
                catalog = yaml.safe_load(f) or {}
            for name, data in catalog.items():
                try:
                    self._models[name] = _build_model(data)
                except Exception:
                    pass
        except Exception:
            pass

    def _load_model_dir(self, directory: Path) -> None:
        """Load models from individual YAML files in a directory."""
        if not directory.exists():
            return
        for path in sorted(directory.glob("*.yaml")):
            try:
                with open(path) as f:
                    data = yaml.safe_load(f)
                if data and "name" in data:
                    self._models[data["name"]] = _build_model(data)
            except Exception:
                pass

    # --- Resolution suffixes ---
    RESOLUTION_SUFFIXES = ("_LowRes", "_MidRes", "_HigRes")

    # --- Queries ---

    def list_references(self) -> list[ReferenceDataset]:
        return sorted(self._references.values(), key=lambda r: r.name)

    def get_reference(self, name: str) -> Optional[ReferenceDataset]:
        """Get a reference dataset by exact name.

        Use the full name including resolution suffix:
            'GLEAM_v4.2a_LowRes', 'GLEAM_v4.2a_MidRes'

        Base name without suffix (e.g., 'GLEAM_v4.2a') only works if
        there is an exact entry with that name (no resolution variants).
        """
        return self._references.get(name)

    def get_resolution_variants(self, base_name: str) -> dict[str, ReferenceDataset]:
        """Find all resolution variants of a dataset.

        Args:
            base_name: Base dataset name without resolution suffix (e.g., 'GLEAM_v4.2a')

        Returns:
            Dict mapping resolution label to ReferenceDataset.
            E.g., {'LowRes': ..., 'MidRes': ..., 'HigRes': ...}
        """
        variants = {}

        for suffix in self.RESOLUTION_SUFFIXES:
            full_name = f"{base_name}{suffix}"
            if full_name in self._references:
                label = suffix[1:]  # Strip leading underscore
                variants[label] = self._references[full_name]

        # Also check if base_name itself is a standalone entry (no resolution suffix)
        if base_name in self._references and not variants:
            variants["default"] = self._references[base_name]

        return variants

    def list_models(self) -> list[ModelProfile]:
        return sorted(self._models.values(), key=lambda m: m.name)

    def get_model(self, name: str) -> Optional[ModelProfile]:
        return self._models.get(name)

    def references_for_variable(self, variable: str) -> list[ReferenceDataset]:
        return [ref for ref in self._references.values() if variable in ref.variables]


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
        )

    return ReferenceDataset(
        name=data["name"],
        description=data.get("description", ""),
        category=data.get("category", ""),
        data_type=data["data_type"],
        tim_res=data["tim_res"],
        data_groupby=data.get("data_groupby", "Year"),
        timezone=data.get("timezone", 0),
        years=data.get("years", []),
        variables=variables,
        grid_res=data.get("grid_res"),
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
        )

    return ModelProfile(
        name=data["name"],
        description=data.get("description", ""),
        data_type=data.get("data_type", "grid"),
        grid_res=data.get("grid_res"),
        tim_res=data.get("tim_res", "Month"),
        variables=variables,
    )
