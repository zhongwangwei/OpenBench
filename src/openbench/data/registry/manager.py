"""RegistryManager: loads and queries reference datasets and model profiles.

Loading order (later entries override earlier):
1. Built-in catalog:  src/openbench/data/registry/reference_catalog.yaml
2. Built-in individuals: src/openbench/data/registry/references/*.yaml  (legacy)
3. User catalog:      ~/.openbench/reference_catalog.yaml
4. User individuals:  ~/.openbench/references/*.yaml
Same order for model_catalog.yaml / models/*.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from openbench.data.registry.schema import ModelProfile, ReferenceDataset, VariableMapping


class RegistryManager:
    """Manages reference dataset and model profile descriptors."""

    def __init__(self, user_dir: Optional[Path] = None):
        self._references: dict[str, ReferenceDataset] = {}
        self._models: dict[str, ModelProfile] = {}

        # Built-in (shipped with package)
        builtin_dir = Path(__file__).parent
        self._load_reference_catalog(builtin_dir / "reference_catalog.yaml")
        self._load_reference_dir(builtin_dir / "references")
        self._load_model_catalog(builtin_dir / "model_catalog.yaml")
        self._load_model_dir(builtin_dir / "models")

        # User-defined (override built-in)
        if user_dir is None:
            try:
                from platformdirs import user_config_dir

                user_dir = Path(user_config_dir("openbench"))
            except ImportError:
                user_dir = Path.home() / ".openbench"

        if user_dir.exists():
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

    # --- Queries ---

    def list_references(self) -> list[ReferenceDataset]:
        return sorted(self._references.values(), key=lambda r: r.name)

    def get_reference(self, name: str) -> Optional[ReferenceDataset]:
        return self._references.get(name)

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
