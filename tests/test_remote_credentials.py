import json
import os

import pytest

from openbench.remote.credentials import CredentialManager, CredentialStorageError


def _manager(tmp_path, monkeypatch):
    monkeypatch.setattr(CredentialManager, "_get_encryption_key", lambda self: "stable-machine:user")
    return CredentialManager(config_dir=str(tmp_path))


def test_credentials_roundtrip_encrypts_password_and_restricts_permissions(tmp_path, monkeypatch):
    mgr = _manager(tmp_path, monkeypatch)

    mgr.save_credential("alice@example.org", "password", password="secret")

    raw = json.loads((tmp_path / "credentials.json").read_text(encoding="utf-8"))
    stored_password = raw["servers"]["alice@example.org"]["password"]
    assert stored_password != "secret"
    assert "secret" not in (tmp_path / "credentials.json").read_text(encoding="utf-8")
    assert mgr.get_credential("alice@example.org")["password"] == "secret"

    if os.name == "posix":
        assert ((tmp_path / "credentials.json").stat().st_mode & 0o777) == 0o600
        assert ((tmp_path / "encryption_salt.bin").stat().st_mode & 0o777) == 0o600


def test_corrupt_credentials_are_not_overwritten_on_save(tmp_path, monkeypatch):
    mgr = _manager(tmp_path, monkeypatch)
    credentials_path = tmp_path / "credentials.json"
    credentials_path.write_text("not-json", encoding="utf-8")

    with pytest.raises(CredentialStorageError, match="Refusing to overwrite"):
        mgr.save_credential("bob@example.org", "password", password="new-secret")

    assert credentials_path.read_text(encoding="utf-8") == "not-json"


def test_reading_corrupt_credentials_returns_empty_without_destroying_file(tmp_path, monkeypatch):
    mgr = _manager(tmp_path, monkeypatch)
    credentials_path = tmp_path / "credentials.json"
    credentials_path.write_text("not-json", encoding="utf-8")

    assert mgr.get_credential("bob@example.org") is None
    assert mgr.list_hosts() == []
    assert credentials_path.read_text(encoding="utf-8") == "not-json"


def test_save_credential_roundtrips_compute_node_key_and_password(tmp_path):
    mgr = CredentialManager(config_dir=str(tmp_path))

    mgr.save_credential(
        "alice@example.org",
        "key",
        key_file="/keys/login",
        jump_node="node001",
        jump_auth="key",
        node_key_file="/keys/node",
    )

    cred = mgr.get_credential("alice@example.org")
    assert cred["key_file"] == "/keys/login"
    assert cred["jump_node"] == "node001"
    assert cred["jump_auth"] == "key"
    assert cred["node_key_file"] == "/keys/node"

    mgr.save_credential(
        "alice@example.org",
        "password",
        password="login-secret",
        jump_node="node002",
        jump_auth="password",
        node_password="node-secret",
    )

    cred = mgr.get_credential("alice@example.org")
    assert cred["password"] == "login-secret"
    assert cred["jump_node"] == "node002"
    assert cred["jump_auth"] == "password"
    assert cred["node_password"] == "node-secret"
