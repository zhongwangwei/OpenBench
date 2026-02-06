"""YAML configuration loading with env var substitution."""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..exceptions import ConfigurationError


class ConfigLoader:
    """Loads and merges YAML configuration files."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)

    def load_global(self) -> Dict[str, Any]:
        """Load global configuration from global.yaml.

        Returns:
            Parsed configuration dictionary.

        Raises:
            ConfigurationError: If global.yaml is not found.
        """
        path = self.base_dir / "global.yaml"
        if not path.exists():
            raise ConfigurationError(f"Global config not found: {path}")
        return self._load_yaml(path)

    def load_dataset(self, path: Path) -> Dict[str, Any]:
        """Load a single dataset configuration file.

        Args:
            path: Path to the dataset YAML file.

        Returns:
            Parsed configuration dictionary.
        """
        return self._load_yaml(path)

    def load_all_datasets(self, datasets_dir: Path) -> List[Dict[str, Any]]:
        """Load all dataset configs from a directory, excluding files starting with '_'.

        Args:
            datasets_dir: Directory containing dataset YAML files.

        Returns:
            List of parsed configuration dictionaries.
        """
        configs = []
        for f in sorted(datasets_dir.glob("*.yaml")):
            if f.name.startswith("_"):
                continue
            configs.append(self._load_yaml(f))
        return configs

    def load_priorities(self) -> Dict[str, Any]:
        """Load priorities configuration.

        Returns:
            Parsed priorities dictionary, or defaults if file not found.
        """
        path = self.base_dir / "priorities.yaml"
        if not path.exists():
            return {"priorities": {}, "default_priority": 50, "excluded_from_merge": []}
        return self._load_yaml(path)

    def load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules configuration.

        Returns:
            Parsed validation rules dictionary, or empty dict if file not found.
        """
        path = self.base_dir / "validation_rules.yaml"
        if not path.exists():
            return {}
        return self._load_yaml(path)

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load and parse a YAML file with env var substitution.

        Args:
            path: Path to the YAML file.

        Returns:
            Parsed dictionary with environment variables resolved.
        """
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        resolved = self._resolve_env_vars(raw)
        return yaml.safe_load(resolved) or {}

    def _resolve_env_vars(self, text: str) -> str:
        """Resolve ${VAR_NAME:default} patterns in text.

        Supports:
            ${VAR} - replaced by env var, kept as-is if not set
            ${VAR:default} - replaced by env var, or default if not set

        Args:
            text: Raw text with possible env var references.

        Returns:
            Text with env vars resolved.
        """
        def replacer(match):
            var_name = match.group(1)
            default = match.group(3)
            return os.environ.get(
                var_name, default if default is not None else match.group(0)
            )

        return re.sub(r"\$\{(\w+)(:([^}]*))?\}", replacer, text)
