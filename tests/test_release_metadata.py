"""Keep PyPI and Conda release metadata aligned."""

import sys
from pathlib import Path

import pytest
import yaml
from jinja2 import Environment, StrictUndefined
from packaging.requirements import Requirement
from packaging.version import Version

from openbench import __version__

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def metadata():
    recipe_path = ROOT / "conda/meta.yaml"
    if not recipe_path.is_file():
        pytest.skip("Conda recipe is a repository-only release input")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    template = Environment(undefined=StrictUndefined).from_string(recipe_path.read_text(encoding="utf-8"))
    recipe = yaml.safe_load(template.render(PYTHON=sys.executable))
    return project, recipe


def test_pypi_and_conda_versions_and_release_status_agree(metadata):
    project, recipe = metadata
    assert recipe["package"] == {"name": project["name"], "version": __version__}
    status = "4 - Beta" if Version(__version__).is_prerelease else "5 - Production/Stable"
    assert f"Development Status :: {status}" in project["classifiers"]
    assert recipe["source"]["url"].endswith(f"/colm_openbench-{__version__}.tar.gz")
    assert len(recipe["source"]["sha256"]) == 64


def test_conda_installs_the_declared_runtime_without_pip_dependency_resolution(metadata):
    project, recipe = metadata
    requirements = dict(item.lower().split(maxsplit=1) for item in recipe["requirements"]["run"])
    assert requirements.pop("python") == project["requires-python"]
    aliases = {"matplotlib": "matplotlib-base"}
    for item in project["dependencies"]:
        dependency = Requirement(item)
        name = aliases.get(dependency.name.lower(), dependency.name.lower())
        assert requirements.pop(name) == str(dependency.specifier)
        if dependency.name == "dask" and "distributed" in dependency.extras:
            assert requirements.pop("distributed") == str(dependency.specifier)
    assert not requirements
    assert "--no-deps" in recipe["build"]["script"]
    assert "--no-build-isolation" in recipe["build"]["script"]
    assert recipe["build"]["entry_points"] == [f"openbench = {project['scripts']['openbench']}"]
    assert {"python -m pip check", "openbench --version", "openbench smoke-test"} <= set(recipe["test"]["commands"])


def test_distributions_declare_and_include_vendored_licenses(metadata):
    project, recipe = metadata
    assert project["license"] == recipe["about"]["license"] == "MIT AND GPL-3.0-only AND LicenseRef-NCL-6.3.0"
    license_files = {
        "LICENSE",
        "src/openbench/visualization/cmaps/LICENSE",
        "src/openbench/visualization/cmaps/colormaps/ncar_ncl/Copyright",
        "src/openbench/visualization/cmaps/colormaps/ncar_ncl/NCL_source_license.txt",
    }
    assert set(project["license-files"]) == set(recipe["about"]["license_file"]) == license_files
    assert all((ROOT / path).is_file() for path in license_files)
