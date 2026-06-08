"""Small persistent user settings for OpenBench CLI behavior."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml

SETTINGS_FILE_NAME = "settings.yaml"


def get_user_config_dir(user_dir: str | Path | None = None) -> Path:
    """Return the per-user OpenBench config directory."""
    return Path(user_dir).expanduser() if user_dir is not None else Path.home() / ".openbench"


def get_user_settings_path(user_dir: str | Path | None = None) -> Path:
    """Return the persistent settings file path."""
    return get_user_config_dir(user_dir) / SETTINGS_FILE_NAME


def load_user_settings(user_dir: str | Path | None = None) -> dict[str, Any]:
    """Load user settings, treating missing or invalid files as empty."""
    path = get_user_settings_path(user_dir)
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def save_user_settings(settings: dict[str, Any], user_dir: str | Path | None = None) -> Path:
    """Atomically save user settings."""
    path = get_user_settings_path(user_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(settings, sort_keys=True)
    with NamedTemporaryFile("w", delete=False, dir=path.parent, prefix=f".{path.name}.", suffix=".tmp") as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)
    return path


def get_persisted_reference_root(user_dir: str | Path | None = None) -> str | None:
    """Return the saved reference root without consulting process env vars."""
    value = load_user_settings(user_dir).get("reference_root")
    if not isinstance(value, str) or not value.strip():
        return None
    return str(Path(os.path.expandvars(os.path.expanduser(value))).resolve())


def resolve_reference_root(user_dir: str | Path | None = None) -> str | None:
    """Return OPENBENCH_REF_ROOT, falling back to the persisted reference root."""
    env_value = os.environ.get("OPENBENCH_REF_ROOT")
    if env_value:
        return str(Path(os.path.expandvars(os.path.expanduser(env_value))).resolve())
    return get_persisted_reference_root(user_dir)


def remember_reference_root(ref_root: str | Path, user_dir: str | Path | None = None) -> Path:
    """Persist the default reference root for later commands."""
    settings = load_user_settings(user_dir)
    settings["reference_root"] = str(Path(ref_root).expanduser().resolve())
    return save_user_settings(settings, user_dir)
