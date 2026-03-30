"""RegistryManager: loads and queries reference datasets and model profiles.

Loads YAML descriptors from two locations:
1. Built-in: src/openbench/data/registry/references/ and models/
2. User-defined: ~/.openbench/references/ and ~/.openbench/models/
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

        # Built-in descriptors (shipped with package)
        builtin_dir = Path(__file__).parent
        self._load_references(builtin_dir / "references")
        self._load_models(builtin_dir / "models")

        # User-defined descriptors (override built-in)
        if user_dir is None:
            try:
                from platformdirs import user_config_dir

                user_dir = Path(user_config_dir("openbench"))
            except ImportError:
                user_dir = Path.home() / ".openbench"

        if user_dir.exists():
            self._load_references(user_dir / "references")
            self._load_models(user_dir / "models")

    def _load_references(self, directory: Path) -> None:
        if not directory.exists():
            return
        for path in sorted(directory.glob("*.yaml")):
            try:
                ref = _load_reference_yaml(path)
                self._references[ref.name] = ref
            except Exception:
                pass

    def _load_models(self, directory: Path) -> None:
        if not directory.exists():
            return
        for path in sorted(directory.glob("*.yaml")):
            try:
                model = _load_model_yaml(path)
                self._models[model.name] = model
            except Exception:
                pass

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


def _load_reference_yaml(path: Path) -> ReferenceDataset:
    with open(path) as f:
        data = yaml.safe_load(f)

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


def _load_model_yaml(path: Path) -> ModelProfile:
    with open(path) as f:
        data = yaml.safe_load(f)

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
