"""Helpers for running small Python inspectors on a connected remote host."""

from __future__ import annotations

import base64
import json
import re
import shlex
from typing import Any

_CONDA_BASE_PATTERN = re.compile(r"(.*?/(?:miniconda|miniforge|anaconda|mambaforge)[^/]*)")


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
    match = _CONDA_BASE_PATTERN.search(python_path or "")
    if match:
        q_base = shlex.quote(match.group(1))
        return f"source {q_base}/etc/profile.d/conda.sh && conda activate {q_env} && {inner}"
    return f"bash -l -c {shlex.quote(f'conda activate {q_env} && {inner}')}"


def build_remote_python_command(script: str, python_path: str = "", conda_env: str = "") -> str:
    """Return a shell command that pipes ``script`` into remote Python safely."""
    python = python_path or "python3"
    script_b64 = base64.b64encode(script.encode("utf-8")).decode("ascii")
    runner = f"printf %s {shlex.quote(script_b64)} | base64 -d | {shlex.quote(python)}"
    return wrap_with_conda_env(runner, python_path=python_path, conda_env=conda_env)


def run_remote_python_json(
    ssh_manager,
    script: str,
    *,
    python_path: str = "",
    conda_env: str = "",
    timeout: int = 60,
) -> Any:
    """Execute ``script`` remotely and parse a JSON value from stdout."""
    command = build_remote_python_command(script, python_path=python_path, conda_env=conda_env)
    try:
        from openbench.gui.widgets._ssh_worker import execute_responsive

        stdout, stderr, exit_code = execute_responsive(ssh_manager, command, timeout=timeout)
    except ImportError:  # pragma: no cover - GUI extra not installed
        stdout, stderr, exit_code = ssh_manager.execute(command, timeout=timeout)
    if exit_code != 0:
        detail = (stderr or stdout or "").strip()
        raise RuntimeError(f"Remote Python command failed with exit code {exit_code}: {detail}")
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Remote Python command returned no JSON output")
    text = lines[-1]
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Remote Python command returned invalid JSON: {exc}") from exc
