# -*- coding: utf-8 -*-
"""
Wizard Configuration Manager for .wizard.yaml files.

Manages wizard-only settings that are stored separately from the main
OpenBench configuration. This includes remote execution settings,
UI preferences, and other wizard-specific configurations.
"""

import os
from typing import Dict, Any, Optional

import yaml


class WizardConfigManager:
    """
    Manages wizard-specific configuration stored in .wizard.yaml files.

    The .wizard.yaml file stores settings that are specific to the wizard
    application and should NOT be included in the main OpenBench config.
    This includes:
    - Remote execution settings (SSH host, auth type, jump nodes, etc.)
    - Python environment paths for remote execution
    - Execution mode preferences (local vs remote)
    - Other wizard-specific UI settings

    Usage:
        manager = WizardConfigManager()

        # Load existing config
        config = manager.load("/path/to/project")

        # Modify and save
        config["remote"]["host"] = "user@server.example.com"
        manager.save("/path/to/project", config)
    """

    WIZARD_CONFIG_FILENAME = ".wizard.yaml"

    def __init__(self):
        """Initialize the Wizard Config Manager."""
        pass

    def get_config_path(self, output_dir: str) -> str:
        """
        Get the full path to the .wizard.yaml file.

        Args:
            output_dir: Project output directory path

        Returns:
            Full path to .wizard.yaml file
        """
        return os.path.join(output_dir, self.WIZARD_CONFIG_FILENAME)

    def load(self, output_dir: str) -> Dict[str, Any]:
        """
        Load wizard configuration from .wizard.yaml in the project directory.

        Args:
            output_dir: Project output directory path

        Returns:
            Configuration dictionary. Returns default config if file doesn't exist.
        """
        config_path = self.get_config_path(output_dir)

        if not os.path.exists(config_path):
            return self._get_default_config()

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if config is None:
                return self._get_default_config()

            # Merge with defaults to ensure all keys exist
            return self._merge_with_defaults(config)

        except yaml.YAMLError as e:
            # Log error and return defaults
            print(f"Warning: Failed to parse {config_path}: {e}")
            return self._get_default_config()
        except Exception as e:
            print(f"Warning: Failed to load {config_path}: {e}")
            return self._get_default_config()

    def save(self, output_dir: str, config: Dict[str, Any]) -> None:
        """
        Save wizard configuration to .wizard.yaml in the project directory.

        Args:
            output_dir: Project output directory path
            config: Configuration dictionary to save

        Raises:
            OSError: If directory creation or file writing fails
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        config_path = self.get_config_path(output_dir)

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                config,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                indent=2
            )

    def exists(self, output_dir: str) -> bool:
        """
        Check if .wizard.yaml exists in the project directory.

        Args:
            output_dir: Project output directory path

        Returns:
            True if .wizard.yaml exists, False otherwise
        """
        return os.path.exists(self.get_config_path(output_dir))

    def delete(self, output_dir: str) -> bool:
        """
        Delete .wizard.yaml from the project directory.

        Args:
            output_dir: Project output directory path

        Returns:
            True if file was deleted, False if it didn't exist
        """
        config_path = self.get_config_path(output_dir)

        if os.path.exists(config_path):
            os.remove(config_path)
            return True
        return False

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get the default wizard configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            "execution": {
                "mode": "local",  # "local" or "remote"
            },
            "remote": {
                "host": "",                    # user@host[:port] format
                "auth_type": "password",       # "password" or "key"
                "key_file": "",                # Path to SSH key file
                "use_jump": False,             # Whether to use jump/compute node
                "jump_node": "",               # Jump/compute node name
                "jump_auth": "none",           # "none" (internal trust) or "password"
                "python_path": "",             # Python interpreter path on remote
                "conda_env": "",               # Conda environment name
                "openbench_path": "",          # OpenBench installation path on remote
            },
            "ui": {
                "last_tab": 0,                 # Last selected tab index
                "window_geometry": None,       # Window position and size
            },
        }

    def _merge_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge loaded config with defaults to ensure all keys exist.

        Args:
            config: Loaded configuration dictionary

        Returns:
            Configuration dictionary with all default keys present
        """
        defaults = self._get_default_config()
        return self._deep_merge(defaults, config)

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries, with override values taking precedence.

        Args:
            base: Base dictionary with default values
            override: Dictionary with values to override

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    # Convenience methods for common operations

    def get_execution_mode(self, output_dir: str) -> str:
        """
        Get the execution mode (local or remote).

        Args:
            output_dir: Project output directory path

        Returns:
            Execution mode string ("local" or "remote")
        """
        config = self.load(output_dir)
        return config.get("execution", {}).get("mode", "local")

    def set_execution_mode(self, output_dir: str, mode: str) -> None:
        """
        Set the execution mode.

        Args:
            output_dir: Project output directory path
            mode: Execution mode ("local" or "remote")
        """
        config = self.load(output_dir)
        if "execution" not in config:
            config["execution"] = {}
        config["execution"]["mode"] = mode
        self.save(output_dir, config)

    def get_remote_config(self, output_dir: str) -> Dict[str, Any]:
        """
        Get remote execution configuration.

        Args:
            output_dir: Project output directory path

        Returns:
            Remote configuration dictionary
        """
        config = self.load(output_dir)
        return config.get("remote", self._get_default_config()["remote"])

    def set_remote_config(self, output_dir: str, remote_config: Dict[str, Any]) -> None:
        """
        Set remote execution configuration.

        Args:
            output_dir: Project output directory path
            remote_config: Remote configuration dictionary
        """
        config = self.load(output_dir)
        config["remote"] = remote_config
        self.save(output_dir, config)

    def is_remote_execution_enabled(self, output_dir: str) -> bool:
        """
        Check if remote execution is enabled.

        Args:
            output_dir: Project output directory path

        Returns:
            True if execution mode is "remote", False otherwise
        """
        return self.get_execution_mode(output_dir) == "remote"

    def get_remote_host(self, output_dir: str) -> str:
        """
        Get the remote host string.

        Args:
            output_dir: Project output directory path

        Returns:
            Remote host string (user@host[:port] format)
        """
        remote_config = self.get_remote_config(output_dir)
        return remote_config.get("host", "")

    def get_remote_python_path(self, output_dir: str) -> str:
        """
        Get the Python interpreter path on the remote server.

        Args:
            output_dir: Project output directory path

        Returns:
            Python interpreter path
        """
        remote_config = self.get_remote_config(output_dir)
        return remote_config.get("python_path", "")

    def get_remote_openbench_path(self, output_dir: str) -> str:
        """
        Get the OpenBench installation path on the remote server.

        Args:
            output_dir: Project output directory path

        Returns:
            OpenBench installation path
        """
        remote_config = self.get_remote_config(output_dir)
        return remote_config.get("openbench_path", "")
