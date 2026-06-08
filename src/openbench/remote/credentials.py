# -*- coding: utf-8 -*-
"""
Credential Manager for secure password storage.

Stores SSH credentials with encryption.
"""

import base64
import getpass
import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class CredentialStorageError(RuntimeError):
    """Raised when saved credential storage cannot be safely read or updated."""


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

    def _ensure_config_dir(self) -> None:
        os.makedirs(self._config_dir, exist_ok=True)
        try:
            os.chmod(self._config_dir, 0o700)
        except (OSError, AttributeError):
            pass

    def _get_or_create_salt(self) -> bytes:
        """Get existing salt or create a new random salt."""
        self._ensure_config_dir()

        if os.path.exists(self._salt_path):
            try:
                with open(self._salt_path, "rb") as f:
                    salt = f.read()
                if len(salt) == self.SALT_SIZE:
                    try:
                        os.chmod(self._salt_path, 0o600)
                    except (OSError, AttributeError):
                        pass
                    return salt
                logger.warning("Invalid salt file size, regenerating")
            except Exception as e:
                logger.warning("Failed to read salt file: %s", e)

        salt = os.urandom(self.SALT_SIZE)

        try:
            self._atomic_write(self._salt_path, salt, binary=True)
            logger.info("Generated new encryption salt")
        except Exception as e:
            logger.warning(
                "Failed to save salt file (%s). Using an ephemeral in-memory "
                "salt for this session — credentials saved now will NOT be "
                "decryptable in future sessions.",
                e,
            )

        return salt

    def _create_fernet(self) -> Fernet:
        """Create Fernet cipher using machine-specific key."""
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
        """Get machine identifier for key derivation."""
        import platform
        import uuid

        machine_id = ""
        for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
            try:
                with open(path, "r") as f:
                    candidate = f.read().strip()
                if candidate:
                    machine_id = candidate
                    break
            except OSError:
                continue

        if not machine_id:
            # Prefer MAC-derived UUID over platform.node() — node() mirrors
            # the current hostname, which flips when the user moves between
            # Wi-Fi networks / DHCP leases and silently invalidates every
            # stored password. uuid.getnode() returns a stable NIC MAC; if
            # no MAC is available it returns a random 48-bit value with the
            # multicast bit set, which we then fall back through to hostname.
            node_id = uuid.getnode()
            if node_id and not (node_id & (1 << 40)):
                machine_id = format(node_id, "012x")
            else:
                machine_id = platform.node() or "openbench-unknown-host"

        user = getpass.getuser()
        return f"{machine_id}:{user}"

    def _read_credentials_file(self) -> Dict[str, Any]:
        if not os.path.exists(self._credentials_path):
            return {"servers": {}}
        try:
            with open(self._credentials_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise CredentialStorageError(f"Failed to load credentials from {self._credentials_path}: {e}") from e
        if not isinstance(data, dict):
            raise CredentialStorageError(f"Credentials file {self._credentials_path} must contain a JSON object")
        servers = data.setdefault("servers", {})
        if not isinstance(servers, dict):
            raise CredentialStorageError(f"Credentials file {self._credentials_path} has invalid 'servers' section")
        return data

    def _load_credentials(self) -> Dict[str, Any]:
        """Load credentials for read-only operations.

        Corrupt/unreadable files are reported and treated as empty for reads,
        but update operations use `_load_credentials_for_update()` so a later
        save cannot silently overwrite the damaged file.
        """
        try:
            return self._read_credentials_file()
        except CredentialStorageError as e:
            logger.warning("%s; treating as empty for this read.", e)
            return {"servers": {}}

    def _load_credentials_for_update(self) -> Dict[str, Any]:
        """Load credentials before mutation, refusing to clobber corrupt data."""
        try:
            return self._read_credentials_file()
        except CredentialStorageError as e:
            raise CredentialStorageError(
                f"Refusing to overwrite unreadable or corrupt credentials file {self._credentials_path}. "
                "Move it aside or delete it before saving new credentials."
            ) from e

    def _atomic_write(self, path: str, content: str | bytes, *, binary: bool = False) -> None:
        self._ensure_config_dir()
        mode = "wb" if binary else "w"
        kwargs = {} if binary else {"encoding": "utf-8"}
        fd, tmp_path = tempfile.mkstemp(
            dir=self._config_dir,
            prefix=f".{os.path.basename(path)}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, mode, **kwargs) as f:
                f.write(content)
            try:
                os.chmod(tmp_path, 0o600)
            except (OSError, AttributeError):
                pass
            os.replace(tmp_path, path)
            try:
                os.chmod(path, 0o600)
            except (OSError, AttributeError):
                pass
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _save_credentials(self, data: Dict[str, Any]) -> None:
        """Save credentials to file atomically."""
        self._atomic_write(self._credentials_path, json.dumps(data, indent=2))

    def save_credential(
        self,
        host: str,
        auth_type: str,
        password: Optional[str] = None,
        key_file: Optional[str] = None,
        jump_node: Optional[str] = None,
        jump_auth: str = "none",
    ) -> None:
        """Save credential for a host."""
        data = self._load_credentials_for_update()

        encrypted_password = None
        if password:
            encrypted_password = self._fernet.encrypt(password.encode()).decode()

        data["servers"][host] = {
            "auth_type": auth_type,
            "password": encrypted_password,
            "key_file": key_file,
            "jump_node": jump_node,
            "jump_auth": jump_auth,
        }

        self._save_credentials(data)

    def get_credential(self, host: str) -> Optional[Dict[str, Any]]:
        """Get credential for a host."""
        data = self._load_credentials()
        cred = data.get("servers", {}).get(host)

        if cred is None:
            return None

        cred = cred.copy()

        if cred.get("password"):
            try:
                decrypted = self._fernet.decrypt(cred["password"].encode()).decode()
                cred["password"] = decrypted
            except Exception as e:
                logger.warning(
                    "Failed to decrypt saved password for host %r (%s). "
                    "Re-save the credential to regenerate it under the "
                    "current machine identity.",
                    host,
                    e,
                )
                cred["password"] = None

        return cred

    def delete_credential(self, host: str) -> None:
        """Delete credential for a host."""
        data = self._load_credentials_for_update()
        if host in data.get("servers", {}):
            del data["servers"][host]
            self._save_credentials(data)

    def clear_all(self) -> None:
        """Clear all saved credentials."""
        if os.path.exists(self._credentials_path):
            self._load_credentials_for_update()
        self._save_credentials({"servers": {}})

    def list_hosts(self) -> List[str]:
        """List all saved hosts."""
        data = self._load_credentials()
        return list(data.get("servers", {}).keys())
