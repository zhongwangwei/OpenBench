# core/connection_manager.py
# -*- coding: utf-8 -*-
"""
SSH connection configuration manager.

Saves and loads connection profiles from ~/.openbench_wizard/connections.yaml

Usage:
    This module provides persistent storage for SSH connection profiles.
    It is intended to be used by RemoteConfigWidget to populate the connection
    dropdown list and save new connections. The ProjectSelectorDialog indirectly
    uses this through RemoteConfigWidget.

    Example:
        manager = ConnectionManager()
        connections = manager.list_connections()
        manager.save_connection(name="Server", host="user@example.com")
"""

import os
from typing import Dict, List, Optional, Any
from pathlib import Path

import yaml


class ConnectionManager:
    """Manages saved SSH connection profiles."""

    DEFAULT_PATH = os.path.expanduser("~/.openbench_wizard/connections.yaml")

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize connection manager.

        Args:
            config_path: Path to connections config file.
                        Defaults to ~/.openbench_wizard/connections.yaml
        """
        self._config_path = config_path or self.DEFAULT_PATH
        self._connections: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        """Load connections from file."""
        if os.path.exists(self._config_path):
            try:
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
                self._connections = data.get("connections", [])
            except Exception:
                self._connections = []
        else:
            self._connections = []

    def _save(self):
        """Save connections to file."""
        # Ensure directory exists
        dir_path = os.path.dirname(self._config_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        data = {"connections": self._connections}
        with open(self._config_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def list_connections(self) -> List[Dict[str, Any]]:
        """Get list of saved connections."""
        return list(self._connections)

    def get_connection(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a connection by name.

        Args:
            name: Connection name

        Returns:
            Connection dict or None if not found
        """
        for conn in self._connections:
            if conn.get("name") == name:
                return dict(conn)
        return None

    def save_connection(
        self,
        name: str,
        host: str,
        auth_type: str = "key",
        key_file: Optional[str] = None,
        jump_node: Optional[str] = None,
        **kwargs
    ):
        """
        Save or update a connection.

        Args:
            name: Display name for connection
            host: Host string (user@host:port)
            auth_type: "key" or "password"
            key_file: Path to SSH key file
            jump_node: Optional jump node host string
            **kwargs: Additional connection parameters
        """
        conn = {
            "name": name,
            "host": host,
            "auth_type": auth_type,
        }

        if key_file:
            conn["key_file"] = key_file
        if jump_node:
            conn["jump_node"] = jump_node

        conn.update(kwargs)

        # Update existing or add new
        for i, existing in enumerate(self._connections):
            if existing.get("name") == name:
                self._connections[i] = conn
                self._save()
                return

        self._connections.append(conn)
        self._save()

    def delete_connection(self, name: str) -> bool:
        """
        Delete a connection.

        Args:
            name: Connection name

        Returns:
            True if deleted, False if not found
        """
        for i, conn in enumerate(self._connections):
            if conn.get("name") == name:
                del self._connections[i]
                self._save()
                return True
        return False
