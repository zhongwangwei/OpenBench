"""Helpers for running small Python inspectors on a connected remote host."""

from __future__ import annotations

import base64
import json
import os
import shlex
import tempfile
import uuid
from typing import Any


# Canonical implementation lives in the remote layer so non-GUI modules
# (sync engine, SSH manager) can use it; re-exported here for the GUI imports.
from openbench.remote.ssh import quote_remote_path  # noqa: F401

_MAX_INLINE_SCRIPT_CHARS = 60_000


def _looks_like_conda_base(path: str) -> bool:
    name = path.rstrip("/").rsplit("/", 1)[-1].lower()
    return name in {"conda", "mamba"} or name.startswith(
        ("miniconda", "anaconda", "miniforge", "mambaforge", "micromamba")
    )


def conda_env_from_python_path(python_path: str) -> tuple[str, str] | None:
    """Return ``(environment name, Conda base)`` for an identifiable Conda Python."""
    path = (python_path or "").strip()
    suffix = "/bin/python"
    if not path.endswith(suffix):
        return None
    prefix = path[: -len(suffix)]
    if "/envs/" in prefix:
        base, env_name = prefix.rsplit("/envs/", 1)
        if _looks_like_conda_base(base) and env_name and "/" not in env_name:
            return env_name, base
        return None
    if _looks_like_conda_base(prefix):
        return "base", prefix
    return None


def wrap_with_conda_env(inner: str, python_path: str = "", conda_env: str = "") -> str:
    """Wrap a shell command so it runs inside the given conda environment.

    Sourcing ``~/.bashrc`` does NOT work for this: under a non-interactive
    ssh exec the interactivity guard returns before the conda init block, so
    activation silently no-ops and the command runs against the wrong
    interpreter. Instead, derive the conda base from ``python_path`` and
    source ``conda.sh`` directly; fall back to a login shell. ``&&`` chaining
    makes an activation failure fail the command loudly.
    """
    if not conda_env:
        return inner
    q_env = shlex.quote(conda_env)
    conda_info = conda_env_from_python_path(python_path)
    if conda_info:
        # SSHManager wraps everything in `sh -c`, so use the POSIX dot
        # command (`source` is a bashism that dash/ash reject).
        q_base = quote_remote_path(conda_info[1])
        return f". {q_base}/etc/profile.d/conda.sh && conda activate {q_env} && {inner}"
    return f"bash -l -c {shlex.quote(f'conda activate {q_env} && {inner}')}"


def build_remote_python_command(script: str, python_path: str = "", conda_env: str = "") -> str:
    """Return a shell command that pipes ``script`` into remote Python safely."""
    python = python_path or "python3"
    script_b64 = base64.b64encode(script.encode("utf-8")).decode("ascii")
    runner = f"printf %s {shlex.quote(script_b64)} | base64 -d | {quote_remote_path(python)}"
    return wrap_with_conda_env(runner, python_path=python_path, conda_env=conda_env)


def _execute(ssh_manager, command: str, *, timeout: int, should_abort=None):
    try:
        from openbench.gui.widgets._ssh_worker import execute_responsive

        return execute_responsive(ssh_manager, command, timeout=timeout, should_abort=should_abort)
    except ImportError:  # pragma: no cover - GUI extra not installed
        if should_abort is None:
            return ssh_manager.execute(command, timeout=timeout)
        return ssh_manager.execute(command, timeout=timeout, should_abort=should_abort)


def _parse_json_stdout(stdout: str) -> Any:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Remote Python command returned no JSON output")
    text = lines[-1]
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Remote Python command returned invalid JSON: {exc}") from exc


def _remote_python_file_command(remote_path: str, python_path: str = "", conda_env: str = "") -> str:
    python = python_path or "python3"
    runner = f"{quote_remote_path(python)} {quote_remote_path(remote_path)}"
    return wrap_with_conda_env(runner, python_path=python_path, conda_env=conda_env)


def _upload_remote_script(ssh_manager, script: str) -> str:
    remote_path = f"/tmp/openbench-python-{uuid.uuid4().hex}.py"
    local_path = ""
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".py", delete=False) as handle:
            local_path = handle.name
            handle.write(script)
        ssh_manager.upload_file(local_path, remote_path)
    finally:
        if local_path:
            try:
                os.unlink(local_path)
            except OSError:
                pass
    return remote_path


def run_remote_python_json(
    ssh_manager,
    script: str,
    *,
    python_path: str = "",
    conda_env: str = "",
    timeout: int = 60,
    should_abort=None,
) -> Any:
    """Execute ``script`` remotely and parse a JSON value from stdout."""
    remote_script = ""
    if len(script) > _MAX_INLINE_SCRIPT_CHARS and hasattr(ssh_manager, "upload_file"):
        remote_script = _upload_remote_script(ssh_manager, script)
        command = _remote_python_file_command(remote_script, python_path=python_path, conda_env=conda_env)
    else:
        command = build_remote_python_command(script, python_path=python_path, conda_env=conda_env)
    try:
        stdout, stderr, exit_code = _execute(ssh_manager, command, timeout=timeout, should_abort=should_abort)
    finally:
        if remote_script:
            try:
                _execute(ssh_manager, f"rm -f {quote_remote_path(remote_script)}", timeout=30)
            except Exception:
                pass
    if exit_code != 0:
        detail = (stderr or stdout or "").strip()
        raise RuntimeError(f"Remote Python command failed with exit code {exit_code}: {detail}")
    return _parse_json_stdout(stdout)
