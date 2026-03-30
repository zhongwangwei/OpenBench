# -*- coding: utf-8 -*-
"""
Credential Manager for secure password storage.

Stores SSH credentials with encryption.
"""

import os
import json
import hashlib
import getpass
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)


class CredentialManager:
    """Manage encrypted credential storage."""

    CREDENTIALS_FILE = "credentials.json"
    SALT_FILE = "encryption_salt.bin"
    SALT_SIZE = 32  # 256 bits

    def __init__(self, config_dir: Optional[str] = None):
        """Initialize credential manager.

        Args:
            config_dir: Config directory path (default: ~/.openbench_wizard)
        """
        if config_dir is None:
            config_dir = os.path.join(os.path.expanduser("~"), ".openbench_wizard")
        self._config_dir = config_dir
        self._credentials_path = os.path.join(config_dir, self.CREDENTIALS_FILE)
        self._salt_path = os.path.join(config_dir, self.SALT_FILE)
        self._fernet = self._create_fernet()

    def _get_or_create_salt(self) -> bytes:
        """Get existing salt or create a new random salt.

        The salt is stored in a separate file and created once per installation.
        This ensures each installation has a unique salt while maintaining
        backward compatibility (existing installations keep their salt).

        Returns:
            Salt bytes (32 bytes)
        """
        os.makedirs(self._config_dir, exist_ok=True)

        if os.path.exists(self._salt_path):
            try:
                with open(self._salt_path, 'rb') as f:
                    salt = f.read()
                if len(salt) == self.SALT_SIZE:
                    return salt
                else:
                    logger.warning("Invalid salt file size, regenerating")
            except Exception as e:
                logger.warning(f"Failed to read salt file: {e}")

        # Generate new random salt
        salt = os.urandom(self.SALT_SIZE)

        try:
            with open(self._salt_path, 'wb') as f:
                f.write(salt)
            # Set restrictive permissions (Unix only)
            try:
                os.chmod(self._salt_path, 0o600)
            except (OSError, AttributeError):
                pass  # Windows doesn't support chmod
            logger.info("Generated new encryption salt")
        except Exception as e:
            logger.warning(f"Failed to save salt file: {e}")
            # Fall back to a deterministic salt based on machine ID
            # This is less secure but ensures functionality
            machine_id = self._get_encryption_key()
            salt = hashlib.sha256(machine_id.encode()).digest()

        return salt

    def _create_fernet(self) -> Fernet:
        """Create Fernet cipher using machine-specific key.

        Returns:
            Fernet cipher instance
        """
        # Derive key from machine identifier with random salt
        machine_id = self._get_encryption_key()
        salt = self._get_or_create_salt()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(machine_id.encode()))
        return Fernet(key)

    def _get_encryption_key(self) -> str:
        """Get machine identifier for key derivation.

        Derives a key from machine identifiers to ensure credentials
        can only be decrypted on the same machine by the same user.

        Returns:
            Machine identifier string
        """
        import uuid
        # Use MAC address + username as identifier
        mac = uuid.getnode()
        user = getpass.getuser()
        return f"{mac}:{user}"

    def _load_credentials(self) -> Dict[str, Any]:
        """Load credentials from file.

        Returns:
            Credentials dictionary
        """
        if not os.path.exists(self._credentials_path):
            return {"servers": {}}

        try:
            with open(self._credentials_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {"servers": {}}

    def _save_credentials(self, data: Dict[str, Any]) -> None:
        """Save credentials to file.

        Args:
            data: Credentials dictionary
        """
        os.makedirs(self._config_dir, exist_ok=True)
        with open(self._credentials_path, 'w') as f:
            json.dump(data, f, indent=2)
        # Set file permissions to 600 (user only)
        os.chmod(self._credentials_path, 0o600)

    def save_credential(
        self,
        host: str,
        auth_type: str,
        password: Optional[str] = None,
        key_file: Optional[str] = None,
        jump_node: Optional[str] = None,
        jump_auth: str = "none"
    ) -> None:
        """Save credential for a host.

        Args:
            host: Host string (user@host)
            auth_type: Authentication type (password/key)
            password: Password (will be encrypted)
            key_file: SSH key file path
            jump_node: Jump/compute node name
            jump_auth: Jump authentication type
        """
        data = self._load_credentials()

        encrypted_password = None
        if password:
            encrypted_password = self._fernet.encrypt(password.encode()).decode()

        data["servers"][host] = {
            "auth_type": auth_type,
            "password": encrypted_password,
            "key_file": key_file,
            "jump_node": jump_node,
            "jump_auth": jump_auth
        }

        self._save_credentials(data)

    def get_credential(self, host: str) -> Optional[Dict[str, Any]]:
        """Get credential for a host.

        Args:
            host: Host string

        Returns:
            Credential dictionary or None
        """
        data = self._load_credentials()
        cred = data.get("servers", {}).get(host)

        if cred is None:
            return None

        # Make a copy to avoid modifying the stored data
        cred = cred.copy()

        if cred.get("password"):
            try:
                decrypted = self._fernet.decrypt(cred["password"].encode()).decode()
                cred["password"] = decrypted
            except Exception:
                cred["password"] = None

        return cred

    def delete_credential(self, host: str) -> None:
        """Delete credential for a host.

        Args:
            host: Host string
        """
        data = self._load_credentials()
        if host in data.get("servers", {}):
            del data["servers"][host]
            self._save_credentials(data)

    def clear_all(self) -> None:
        """Clear all saved credentials."""
        self._save_credentials({"servers": {}})

    def list_hosts(self) -> List[str]:
        """List all saved hosts.

        Returns:
            List of host strings
        """
        data = self._load_credentials()
        return list(data.get("servers", {}).keys())
