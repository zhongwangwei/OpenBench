"""Build-artifact contents test.

Guards against accidentally shipping runtime scratch files
(`.bak`, `.lock`, `.tmp`, `.nc`, `__pycache__`, output/cache trees, …)
inside the wheel or sdist tarball. Earlier alpha builds shipped a
4 GB `output/` tree (sdist 13 GB) and stale `reference_catalog.yaml.bak`
+ `.lock` files; this test catches both classes of regression.

The test is opt-in: it requires a `dist/` directory containing exactly
one `*.whl` and one `*.tar.gz`. Set OPENBENCH_DIST_DIR to select fresh builds
outside the checkout; explicitly requested missing artifacts fail rather than
skip. CI builds wheel+sdist before running these same checks.
"""

from __future__ import annotations

import os
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
    re.compile(r"\.nc4?$", re.IGNORECASE),
    re.compile(r"__pycache__"),
    re.compile(r"\.pyc$"),
    re.compile(r"\.DS_Store$"),
    re.compile(r"data/custom/output(/|$)"),
    re.compile(r"data/custom/cache(/|$)"),
]

# The archive-specific prefix is checked too: a copy under tests/ or scratch/
# is not a package resource, even when it has the same classification filename.
_MASK_MEMBER = r"openbench/dataset/(IGBP|PFT|Climate_zone)\.nc"

# The dataset masks that MUST ship in the wheel (positive guard).
_REQUIRED_DATASET_MASKS = {
    "openbench/dataset/IGBP.nc",
    "openbench/dataset/PFT.nc",
    "openbench/dataset/Climate_zone.nc",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _find_exactly_one(pattern: str) -> Path | None:
    explicit_dir = os.environ.get("OPENBENCH_DIST_DIR")
    directory = Path(explicit_dir) if explicit_dir is not None else _project_root() / "dist"
    candidates = sorted(directory.glob(pattern))
    if not candidates and explicit_dir is None:
        return None
    assert len(candidates) == 1, f"expected exactly one {pattern} artifact under {directory}; found: {candidates}"
    return candidates[0]


def test_artifact_directory_can_be_selected_explicitly(tmp_path, monkeypatch):
    wheel = tmp_path / "colm_openbench-3.0.0-py3-none-any.whl"
    wheel.touch()
    monkeypatch.setenv("OPENBENCH_DIST_DIR", str(tmp_path))
    assert _find_exactly_one("colm_openbench-*.whl") == wheel


def test_explicit_artifact_directory_cannot_skip_missing_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBENCH_DIST_DIR", str(tmp_path))
    with pytest.raises(AssertionError, match="exactly one"):
        _find_exactly_one("colm_openbench-*.whl")


@pytest.mark.parametrize(
    "member",
    ["openbench/dataset/unexpected.nc", "openbench/dataset/unexpected.NC4", "openbench/data/registry/catalog.yaml.bak"],
)
def test_archive_guard_rejects_unapproved_data_and_scratch_files(member):
    assert _violations([member, "openbench/dataset/IGBP.nc"]) == [member]


@pytest.mark.parametrize(
    "member",
    [
        "tests/openbench/dataset/IGBP.nc",
        "scratch/openbench/dataset/PFT.nc",
        "colm_openbench-3.0.0b15/tests/openbench/dataset/IGBP.nc",
        "colm_openbench-3.0.0b15/src/openbench/dataset/IGBP.nc",
    ],
)
def test_wheel_rejects_masks_outside_exact_package_location(member):
    assert _violations([member]) == [member]


def test_sdist_only_allows_masks_under_its_package_source_tree():
    allowed = "colm_openbench-3.0.0b15/src/openbench/dataset/IGBP.nc"
    forbidden = [
        "colm_openbench-3.0.0b15/tests/openbench/dataset/IGBP.nc",
        "colm_openbench-3.0.0b15/scratch/openbench/dataset/PFT.nc",
        "openbench/dataset/IGBP.nc",
    ]
    assert _violations([allowed, *forbidden], sdist=True) == forbidden


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


def _package_members(archive_members) -> set[str]:
    members = archive_members()
    if archive_members is _wheel_members:
        return set(members)
    # Only strip the sdist's <distribution>/src/ prefix, not arbitrary folders.
    parts = (member.split("/", 2) for member in members)
    return {part[2] for part in parts if len(part) == 3 and part[1] == "src"}


def _violations(members: list[str], *, sdist: bool = False) -> list[str]:
    prefix = r"colm_openbench-[^/]+/src/" if sdist else ""
    allowed = re.compile(prefix + _MASK_MEMBER)
    return [m for m in members if any(p.search(m) for p in _FORBIDDEN) and not allowed.fullmatch(m)]


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
    bad = _violations(_sdist_members(), sdist=True)
    assert not bad, (
        "Sdist contains forbidden runtime artifacts (update pyproject.toml "
        "[tool.hatch.build.targets.sdist].exclude):\n  " + "\n  ".join(bad[:50])
    )


def test_sdist_contains_only_release_inputs() -> None:
    root_files = {"README.md", "LICENSE", "pyproject.toml", "CHANGELOG.md", "PKG-INFO", ".gitignore"}
    paths = [member.split("/", 1)[1] for member in _sdist_members()]
    unexpected = [
        path for path in paths if path not in root_files and not path.startswith(("src/openbench/", "tests/"))
    ]
    assert not unexpected, f"Sdist contains files outside the release inputs: {unexpected}"


@pytest.mark.parametrize("archive_members", [_wheel_members, _sdist_members], ids=["wheel", "sdist"])
def test_archives_contain_required_data_files(archive_members) -> None:
    """Sanity: the registry YAML files MUST ship in the wheel."""
    required = {
        "openbench/data/registry/reference_catalog.yaml",
        "openbench/data/registry/reference_profiles.yaml",
        "openbench/data/registry/model_catalog.yaml",
    }
    missing = required - _package_members(archive_members)
    assert not missing, f"Archive missing required files: {missing}"


@pytest.mark.parametrize("archive_members", [_wheel_members, _sdist_members], ids=["wheel", "sdist"])
def test_archives_contain_bundled_classification_masks(archive_members) -> None:
    """The IGBP/PFT/Köppen masks MUST ship so *_groupby works out of the box."""
    missing = _REQUIRED_DATASET_MASKS - _package_members(archive_members)
    assert not missing, f"Archive missing bundled classification masks: {missing}"


@pytest.mark.parametrize("archive_members", [_wheel_members, _sdist_members], ids=["wheel", "sdist"])
def test_archives_contain_all_package_resource_files(archive_members) -> None:
    """Every non-Python resource in src/openbench should be shipped."""
    package_root = _project_root() / "src" / "openbench"
    expected = {"openbench/" + path.relative_to(package_root).as_posix() for path in _package_resource_files()}
    missing = sorted(expected - _package_members(archive_members))

    assert not missing, "Archive missing package resources:\n  " + "\n  ".join(missing[:50])


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


@pytest.mark.parametrize("name", ["ci.yml", "publish.yml"])
def test_workflows_use_shared_artifact_gate(name) -> None:
    """Repository CI/release must run the shared gate after building archives."""
    path = _project_root() / ".github/workflows" / name
    if not path.exists() and (_project_root() / "PKG-INFO").is_file():
        pytest.skip("CI workflow wiring is checked in the repository, not the sdist")
    workflow = path.read_text(encoding="utf-8")
    build_at = workflow.index("python -m build")
    gate_at = workflow.index("python -m pytest tests/test_build_artifacts.py -q")
    assert build_at < gate_at
    assert "OPENBENCH_DIST_DIR: dist" in workflow
    assert "import tarfile" not in workflow  # The archive rules have one owner.
    if name == "publish.yml":
        assert gate_at < workflow.index("uses: actions/upload-artifact")
