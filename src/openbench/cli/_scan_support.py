"""Support helpers for ``openbench ref scan`` kept out of the Click command module."""

from __future__ import annotations

import sys
from fnmatch import fnmatch
from pathlib import Path

import click

from openbench.cli import _profile_rescue


def _data_attr(name: str, fallback=None):
    """Resolve monkeypatch-friendly attributes from openbench.cli.data."""
    data_module = sys.modules.get("openbench.cli.data")
    if data_module is not None and hasattr(data_module, name):
        return getattr(data_module, name)
    return fallback


def print_scan_skip_report(skipped) -> None:
    click.secho(
        f"Unsupported folder(s) skipped by scanner: {len(skipped)}",
        fg="yellow",
        bold=True,
    )
    for item in skipped:
        path = getattr(item, "path", str(item))
        reason = getattr(item, "reason", "unsupported_layout")
        hint = getattr(item, "hint", "")
        click.echo(f"  - {path}: {reason}")
        if hint:
            click.echo(f"    {hint}")


def scan_skip_keys(skipped) -> set[tuple[str, str]]:
    return {
        (
            getattr(item, "path", str(item)),
            getattr(item, "reason", "unsupported_layout"),
        )
        for item in skipped
    }


def filter_scan_groups(groups, pattern: str):
    filtered = []
    for group in groups:
        variants = {
            res: variant
            for res, variant in group.variants.items()
            if fnmatch(group.base_name, pattern) or fnmatch(variant.registry_name, pattern)
        }
        if variants:
            filtered.append(type(group)(base_name=group.base_name, variants=variants))
    return filtered


def filter_scan_skips(skipped, pattern: str):
    filtered = []
    for item in skipped:
        raw_path = getattr(item, "path", str(item))
        path = Path(raw_path)
        candidates = [raw_path, path.name, *path.parts]
        if any(fnmatch(str(candidate), pattern) for candidate in candidates):
            filtered.append(item)
    return filtered


def format_scan_skip_key(key: tuple[str, str]) -> str:
    path, reason = key
    return f"{path} ({reason})"


def create_profiles_for_scan_skips(skipped, ref_root: Path) -> int:
    pending = []
    profile_supported = _data_attr("_profile_rescue_supported", _profile_rescue._profile_rescue_supported)
    prompt_profile = _data_attr(
        "_prompt_reference_profile_for_scan_skip",
        _profile_rescue._prompt_reference_profile_for_scan_skip,
    )
    for item in skipped:
        if not profile_supported(item):
            path = getattr(item, "path", str(item))
            reason = getattr(item, "reason", "unsupported_layout")
            click.secho(
                f"Cannot infer a reference profile for {path}: {reason}. "
                "Use [i] to ignore it, [s] to skip for now, or register manually.",
                fg="yellow",
            )
            continue
        profile_name, profile = prompt_profile(item, ref_root)
        pending.append((profile_name, profile))
    updated = _data_attr("_write_reference_profiles", _profile_rescue._write_reference_profiles)(pending)
    for profile_name, _profile in pending:
        click.secho(f"Updated reference profile: {profile_name}", fg="green")
    return updated


def create_ignore_profiles_for_scan_skips(skipped) -> int:
    pending = []
    ignore_profile_name = _data_attr("_ignore_profile_name", _profile_rescue._ignore_profile_name)
    for item in skipped:
        skip_path = getattr(item, "path", str(item))
        pending.append(
            (
                ignore_profile_name(skip_path),
                {"scan": {"layout": "ignore", "root_sub_dir": skip_path}},
            )
        )
    updated = _data_attr("_write_reference_profiles", _profile_rescue._write_reference_profiles)(pending)
    for profile_name, _profile in pending:
        click.secho(f"Updated ignore profile: {profile_name}", fg="green")
    return updated
