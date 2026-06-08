"""Build-artifact contents test.

Guards against accidentally shipping runtime scratch files
(`.bak`, `.lock`, `.tmp`, `.nc`, `__pycache__`, output/cache trees, …)
inside the wheel or sdist tarball. Earlier alpha builds shipped a
4 GB `output/` tree (sdist 13 GB) and stale `reference_catalog.yaml.bak`
+ `.lock` files; this test catches both classes of regression.

The test is opt-in: it requires a `dist/` directory containing exactly
one `*.whl` and one `*.tar.gz`. CI builds wheel+sdist explicitly before
running pytest; local developers can either run `python -m build` first
or skip via `-k 'not test_build_artifacts'`.
"""

from __future__ import annotations

import re
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

# Forbidden patterns (regex matched against each archive member name).
# `.bak`/`.lock`/`.tmp`: registry runtime scratch
# `.nc`/`.nc4`: NetCDF datasets that shouldn't ship — EXCEPT the bundled
#   classification masks under openbench/dataset/ (see _ALLOWED_NC below).
# `__pycache__`/`.pyc`: compiled bytecode
# `output/` and `cache/` under data/custom: per-run scratch trees
_FORBIDDEN = [
    re.compile(r"\.bak$"),
    re.compile(r"\.lock$"),
    re.compile(r"\.tmp$"),
    re.compile(r"\.nc4?$"),
    re.compile(r"__pycache__"),
    re.compile(r"\.pyc$"),
    re.compile(r"\.DS_Store$"),
    re.compile(r"data/custom/output(/|$)"),
    re.compile(r"data/custom/cache(/|$)"),
]

# Allowlist: the standard classification masks intentionally bundled with the
# package (compressed integer masks, ~1.1 MB total) consumed by the *_groupby
# evaluation paths. Matches both wheel members (`openbench/dataset/IGBP.nc`)
# and sdist members (`<pkg>-<ver>/src/openbench/dataset/IGBP.nc`).
_ALLOWED = [
    re.compile(r"openbench/dataset/(IGBP|PFT|Climate_zone)\.nc$"),
]

# The dataset masks that MUST ship in the wheel (positive guard).
_REQUIRED_DATASET_MASKS = {
    "openbench/dataset/IGBP.nc",
    "openbench/dataset/PFT.nc",
    "openbench/dataset/Climate_zone.nc",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _find_exactly_one(pattern: str) -> Path | None:
    candidates = sorted((_project_root() / "dist").glob(pattern))
    if not candidates:
        return None
    assert len(candidates) == 1, (
        f"expected exactly one {pattern} artifact under dist/; remove stale artifacts before testing: {candidates}"
    )
    return candidates[0]


def _wheel_members() -> list[str]:
    whl = _find_exactly_one("colm_openbench-*.whl")
    if whl is None:
        pytest.skip("no built wheel under dist/ — run `python -m build --wheel` first")
    with zipfile.ZipFile(whl) as zf:
        return zf.namelist()


def _sdist_members() -> list[str]:
    tarball = _find_exactly_one("colm_openbench-*.tar.gz")
    if tarball is None:
        pytest.skip("no built sdist under dist/ — run `python -m build --sdist` first")
    with tarfile.open(tarball, "r:gz") as tf:
        return tf.getnames()


def _violations(members: list[str]) -> list[str]:
    return [m for m in members if any(p.search(m) for p in _FORBIDDEN) and not any(a.search(m) for a in _ALLOWED)]


def _package_resource_files() -> list[Path]:
    package_root = _project_root() / "src" / "openbench"
    return [
        path
        for path in package_root.rglob("*")
        if path.is_file()
        and path.suffix not in {".py", ".pyc", ".pyo"}
        and not any(part == "__pycache__" for part in path.parts)
        and path.name != ".DS_Store"
    ]


def test_wheel_has_no_forbidden_files() -> None:
    bad = _violations(_wheel_members())
    assert not bad, (
        "Wheel contains forbidden runtime artifacts (update pyproject.toml "
        "[tool.hatch.build.targets.wheel].exclude):\n  " + "\n  ".join(bad[:50])
    )


def test_sdist_has_no_forbidden_files() -> None:
    bad = _violations(_sdist_members())
    assert not bad, (
        "Sdist contains forbidden runtime artifacts (update pyproject.toml "
        "[tool.hatch.build.targets.sdist].exclude):\n  " + "\n  ".join(bad[:50])
    )


def test_wheel_contains_required_data_files() -> None:
    """Sanity: the registry YAML files MUST ship in the wheel."""
    required = {
        "openbench/data/registry/reference_catalog.yaml",
        "openbench/data/registry/reference_profiles.yaml",
        "openbench/data/registry/model_catalog.yaml",
    }
    members = set(_wheel_members())
    missing = required - members
    assert not missing, f"Wheel missing required files: {missing}"


def test_wheel_contains_bundled_classification_masks() -> None:
    """The IGBP/PFT/Köppen masks MUST ship so *_groupby works out of the box."""
    members = set(_wheel_members())
    missing = _REQUIRED_DATASET_MASKS - members
    assert not missing, f"Wheel missing bundled classification masks: {missing}"


def test_wheel_contains_all_package_resource_files() -> None:
    """Every non-Python resource in src/openbench should be shipped."""
    package_root = _project_root() / "src" / "openbench"
    expected = {"openbench/" + path.relative_to(package_root).as_posix() for path in _package_resource_files()}
    members = set(_wheel_members())
    missing = sorted(expected - members)

    assert not missing, "Wheel missing package resources:\n  " + "\n  ".join(missing[:50])


def test_no_generated_artifacts_are_tracked_under_package_or_tests() -> None:
    """Ignored generated files may exist locally, but must never be tracked."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "-i", "--exclude-standard", "src", "tests"],
        cwd=_project_root(),
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        pytest.skip(f"git tracked-ignore check unavailable: {result.stderr.strip()}")
    tracked_ignored = [line for line in result.stdout.splitlines() if line.strip()]
    assert not tracked_ignored, "Generated/ignored files are tracked:\n  " + "\n  ".join(tracked_ignored[:50])


def test_ci_runs_build_artifact_contents_check() -> None:
    """CI must build both archives and run these artifact-content tests."""
    ci = (_project_root() / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "python -m build --wheel" in ci
    assert "python -m build --sdist" in ci
    assert "unexpected wheel NetCDF members" in ci
    assert "unexpected sdist NetCDF members" in ci


def test_publish_runs_build_artifact_contents_check_before_upload() -> None:
    """Release workflow should fail before upload if wheel/sdist are polluted."""
    publish = (_project_root() / ".github/workflows/publish.yml").read_text(encoding="utf-8")

    assert "python -m build" in publish
    assert "unexpected wheel NetCDF members" in publish
    assert "unexpected sdist NetCDF members" in publish
