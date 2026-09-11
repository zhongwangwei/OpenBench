# Releasing OpenBench 3.0

This is a preparation checklist, not a record of publication or a version change.
Do not overwrite an existing release or reuse old local `dist/` artifacts.

## 1. Finalize the release inputs

- Review the complete diff, including scientific and remote-workflow fixes.
- Set `src/openbench/__init__.py::__version__` to the intended release version;
  Hatch reads this single source for package metadata and the CLI.
- Move the relevant `CHANGELOG.md` Unreleased notes into a dated release entry.
- Update the README's beta notice and package development-status classifier
  only when declaring the stable release. Do not claim conda-forge availability
  before it exists.
- Update `conda/meta.yaml` version and SHA-256 from the **final sdist**, not a
  previous beta. Its source URL must point to that exact published archive.

## 2. Validate the source checkout

Run with the project's development, GUI and remote test dependencies installed:

```sh
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -m pytest tests -q -rs
python -m ruff check src tests
python -m ruff format --check src tests
git diff --check
```

Review skips rather than treating them as passes. Require the existing GitHub CI
matrix (Linux, macOS and Windows; Python 3.10–3.12) before a stable release. Local
headless tests do not prove real SSH/HPC connectivity or every optional backend.

## 3. Build and inspect fresh artifacts

Install the build tools in the release environment (`build`, `twine`, `pytest`).
The following shell commands use a new directory and leave existing `dist/` alone:

```sh
ARTIFACT_DIR="$(mktemp -d)"
python -m build --outdir "$ARTIFACT_DIR"
OPENBENCH_DIST_DIR="$ARTIFACT_DIR" python -m pytest tests/test_build_artifacts.py -q
python -m twine check "$ARTIFACT_DIR"/*
```

On Windows, use a new temporary directory and set `OPENBENCH_DIST_DIR` through
PowerShell's `$env:OPENBENCH_DIST_DIR`. Plain `python -m build` builds the wheel
**from the sdist**, so missing source-distribution resources are exercised too.
The shared artifact gate requires registry resources and classification masks
in both archives and rejects scratch files and unapproved NetCDF files.

## 4. Exercise the actual wheel

Install the wheel in a fresh virtual environment, run outside the source checkout,
and confirm `openbench.__file__` points into that environment rather than `src/`.
Do not use editable installs for release evidence.

```sh
python -m pip install /path/to/fresh/colm_openbench-VERSION-py3-none-any.whl
python -c "import openbench; print(openbench.__version__, openbench.__file__)"
openbench --version
openbench --help
openbench model list
openbench smoke-test
openbench smoke-test --run
```

Keep the artifact hashes and test/smoke logs with the release evidence. If a
validation venv shares existing scientific dependencies, record that limitation;
it is an installed-wheel check, not proof of a clean dependency resolution.

## 5. Publish deliberately

Only after review and CI pass: use the approved release/tag and upload process.
Upload the exact verified wheel and sdist with `python -m twine upload`, using the
configured credential store rather than putting tokens in commands or logs.
Verify PyPI's version-specific JSON metadata and download hashes after upload.
The manual publish workflow runs the same artifact gate before uploading and
requires separately configured PyPI/TestPyPI environments. No local cleanup or
build command should trigger publication.

## 6. Validate Conda installation

The README documents both Conda-plus-PyPI installation and a native local Conda
build. Test in a fresh environment, not `base`. The recipe's pip step must not
resolve runtime dependencies or fetch build tools outside Conda's host environment.

Before publishing, a temporary copy of the recipe may point to the verified local
sdist for build/testing; the committed recipe must retain its PyPI URL and the
same SHA-256. After PyPI publication, verify that URL resolves to the same archive.
Run the recipe's dependency, CLI and bundled-data smoke checks. Do not claim a
conda-forge release until the feedstock and channel package actually exist.
