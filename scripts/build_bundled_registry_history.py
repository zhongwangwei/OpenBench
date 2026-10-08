#!/usr/bin/env python
"""Regenerate src/openbench/data/registry/bundled_history.json from git history.

Run from a git checkout whenever reference_catalog.yaml or model_catalog.yaml
changes (tests/test_registry_bundled_history.py fails until you do):

    python scripts/build_bundled_registry_history.py [--ref main]

Released catalog history reachable from ``--ref`` (default: main, else
origin/main), plus the working tree, is recorded as digests and merged with the
digests already in the file, so regenerating never forgets a shipped value.
Shallow clones are refused because they would silently miss history. Matches
warn about possibly outdated overrides; they never authorize replacing a
user's settings.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from openbench.data.registry.bundled_history import (  # noqa: E402
    HISTORY_FILENAME,
    HISTORY_FORMAT,
    iter_entry_digests,
)

REGISTRY = "src/openbench/data/registry"
SOURCES = {
    "references": (f"{REGISTRY}/reference_catalog.yaml", f"{REGISTRY}/references/"),
    "models": (f"{REGISTRY}/model_catalog.yaml", f"{REGISTRY}/models/"),
}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout


def _entries(text: str) -> dict:
    """Return {name: entry} from a catalog or a single-entry descriptor file."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return {}
    if not isinstance(data, dict):
        return {}
    if isinstance(data.get("name"), str):
        return {data["name"]: data}
    return {name: entry for name, entry in data.items() if isinstance(entry, dict)}


def _resolve_ref(requested: str | None) -> str:
    if _git("rev-parse", "--is-shallow-repository").strip() == "true":
        sys.exit("Shallow clone: catalog history is incomplete. Run `git fetch --unshallow` first.")
    candidates = [requested] if requested else ["main", "origin/main"]
    for ref in candidates:
        verify = ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"]
        if subprocess.run(verify, cwd=REPO, capture_output=True).returncode == 0:
            return ref
    sys.exit(f"No such git ref: {' or '.join(candidates)}. Pass --ref with the release branch.")


def _existing_digests() -> dict[str, set[str]]:
    path = REPO / REGISTRY / HISTORY_FILENAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("format") != HISTORY_FORMAT:
        return {}
    return {kind: set(data.get(kind) or ()) for kind in SOURCES}


def _historical_texts(catalog: str, per_file_dir: str, ref: str) -> list[str]:
    texts = []
    commits = _git("log", ref, "--format=%H", "--", catalog, per_file_dir).split()
    for commit in commits:
        listed = _git("ls-tree", "-r", "--name-only", commit, "--", catalog, per_file_dir).split()
        for path in listed:
            if path == catalog or (path.startswith(per_file_dir) and path.endswith(".yaml")):
                texts.append(_git("show", f"{commit}:{path}"))
    current = REPO / catalog
    if current.exists():
        texts.append(current.read_text(encoding="utf-8"))
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", help="git ref whose history counts as released (default: main, else origin/main)")
    ref = _resolve_ref(parser.parse_args().ref)
    previous = _existing_digests()
    history: dict = {"format": HISTORY_FORMAT}
    for kind, (catalog, per_file_dir) in SOURCES.items():
        digests: set[str] = set(previous.get(kind, ()))
        for text in _historical_texts(catalog, per_file_dir, ref):
            for name, entry in _entries(text).items():
                digests.update(iter_entry_digests(kind, name, entry))
        history[kind] = sorted(digests)
        print(f"{kind}: {len(digests)} digests ({len(digests) - len(previous.get(kind, ()))} new) from {ref}")
    out = REPO / REGISTRY / HISTORY_FILENAME
    out.write_text(json.dumps(history, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)} ({out.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
