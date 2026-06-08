"""Tests for per-pair reference copy-on-write preparation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

import openbench.runner.local as local_runner


def test_clone_or_link_ref_for_pair_prefers_first_success(tmp_path, monkeypatch):
    src = tmp_path / "src.nc"
    dst = tmp_path / "nested" / "dst.nc"
    src.write_text("source")
    calls: list[str] = []

    def clonefile(source: str, target: str) -> bool:
        calls.append("clonefile")
        Path(target).write_text(Path(source).read_text())
        return True

    def should_not_run(*_args):  # pragma: no cover - failure path only
        raise AssertionError("fallback should not run after clonefile succeeds")

    monkeypatch.setattr(local_runner, "_try_clonefile", clonefile)
    monkeypatch.setattr(local_runner, "_try_reflink", should_not_run)
    monkeypatch.setattr(local_runner, "_try_hardlink", should_not_run)
    monkeypatch.setattr(local_runner, "_try_symlink", should_not_run)

    strategy = local_runner._clone_or_link_ref_for_pair(str(src), str(dst))

    assert strategy == "clonefile"
    assert calls == ["clonefile"]
    assert dst.read_text() == "source"


def test_clone_or_link_ref_for_pair_falls_back_to_copy2(tmp_path, monkeypatch):
    src = tmp_path / "src.nc"
    dst = tmp_path / "dst.nc"
    src.write_text("source")
    calls: list[str] = []

    def fail_with_partial(name: str):
        def _fail(_source: str, target: str) -> bool:
            calls.append(name)
            Path(target).write_text("partial")
            raise OSError(name)

        return _fail

    def copy2(source: str, target: str):
        calls.append("copy2")
        Path(target).write_text(Path(source).read_text())
        return target

    monkeypatch.setattr(local_runner, "_try_clonefile", fail_with_partial("clonefile"))
    monkeypatch.setattr(local_runner, "_try_reflink", fail_with_partial("reflink"))
    monkeypatch.setattr(local_runner, "_try_hardlink", fail_with_partial("hardlink"))
    monkeypatch.setattr(local_runner, "_try_symlink", fail_with_partial("symlink"))
    monkeypatch.setattr(local_runner.shutil, "copy2", copy2)

    strategy = local_runner._clone_or_link_ref_for_pair(str(src), str(dst))

    assert strategy == "copy2"
    assert calls == ["clonefile", "reflink", "hardlink", "symlink", "copy2"]
    assert dst.read_text() == "source"


def test_clone_or_link_ref_for_pair_replaces_existing_stale_file(tmp_path, monkeypatch):
    src = tmp_path / "src.nc"
    dst = tmp_path / "dst.nc"
    src.write_text("fresh")
    dst.write_text("stale")

    monkeypatch.setattr(local_runner, "_try_clonefile", lambda *_args: False)
    monkeypatch.setattr(local_runner, "_try_reflink", lambda *_args: False)
    monkeypatch.setattr(local_runner, "_try_hardlink", lambda *_args: False)
    monkeypatch.setattr(local_runner, "_try_symlink", lambda *_args: False)

    strategy = local_runner._clone_or_link_ref_for_pair(str(src), str(dst))

    assert strategy == "copy2"
    assert dst.read_text() == "fresh"


def test_clone_or_link_ref_for_pair_replaces_broken_symlink(tmp_path, monkeypatch):
    src = tmp_path / "src.nc"
    dst = tmp_path / "dst.nc"
    src.write_text("source")
    dst.symlink_to(tmp_path / "missing.nc")

    monkeypatch.setattr(local_runner, "_try_clonefile", lambda *_args: False)
    monkeypatch.setattr(local_runner, "_try_reflink", lambda *_args: False)
    monkeypatch.setattr(local_runner, "_try_hardlink", lambda *_args: False)
    monkeypatch.setattr(local_runner, "_try_symlink", lambda *_args: False)

    strategy = local_runner._clone_or_link_ref_for_pair(str(src), str(dst))

    assert strategy == "copy2"
    assert not dst.is_symlink()
    assert dst.read_text() == "source"


def _write_pair_mask_inputs(tmp_path):
    casedir = tmp_path / "case"
    data_dir = casedir / "data"
    data_dir.mkdir(parents=True)
    ref = data_dir / "Runoff_ref_Ref_rv.nc"
    pair_ref = data_dir / "Runoff_ref_Ref_Sim_rv.nc"
    sim = data_dir / "Runoff_sim_Sim_sv.nc"
    coords = {
        "time": np.array(["2000-01-01", "2000-02-01"], dtype="datetime64[ns]"),
        "lat": [1.0],
        "lon": [2.0],
    }
    xr.Dataset({"rv": (("time", "lat", "lon"), np.ones((2, 1, 1)))}, coords=coords).to_netcdf(ref)
    sim_values = np.ones((2, 1, 1))
    sim_values[1, 0, 0] = np.nan
    xr.Dataset({"sv": (("time", "lat", "lon"), sim_values)}, coords=coords).to_netcdf(sim)
    return casedir, ref, pair_ref


def _apply_pair_mask(casedir: Path, pair_ref: Path) -> None:
    local_runner._apply_unified_mask(
        {"casedir": str(casedir), "ref_varname": "rv", "sim_varname": "sv", "time_alignment": "per_pair"},
        "Runoff",
        "Ref",
        "Sim",
        ref_override=str(pair_ref),
    )


def test_pair_ref_hardlink_is_replaced_not_shared_when_masked(tmp_path, monkeypatch):
    """Hardlink pair refs are safe because masking atomically replaces the pair path."""
    casedir, ref, pair_ref = _write_pair_mask_inputs(tmp_path)

    monkeypatch.setattr(local_runner, "_try_clonefile", lambda *_args: False)
    monkeypatch.setattr(local_runner, "_try_reflink", lambda *_args: False)

    def hardlink(source: str, target: str) -> bool:
        Path(target).hardlink_to(source)
        return True

    monkeypatch.setattr(local_runner, "_try_hardlink", hardlink)
    monkeypatch.setattr(local_runner, "_try_symlink", lambda *_args: False)

    strategy = local_runner._clone_or_link_ref_for_pair(str(ref), str(pair_ref))
    assert strategy == "hardlink"
    assert ref.stat().st_ino == pair_ref.stat().st_ino

    _apply_pair_mask(casedir, pair_ref)

    with xr.open_dataset(ref) as ref_ds, xr.open_dataset(pair_ref) as pair_ds:
        assert np.isfinite(ref_ds["rv"].values).all()
        assert np.isnan(pair_ds["rv"].values[1, 0, 0])


def test_pair_ref_symlink_is_replaced_not_shared_when_masked(tmp_path, monkeypatch):
    """Symlink pair refs are also safe under atomic replace."""
    casedir, ref, pair_ref = _write_pair_mask_inputs(tmp_path)

    monkeypatch.setattr(local_runner, "_try_clonefile", lambda *_args: False)
    monkeypatch.setattr(local_runner, "_try_reflink", lambda *_args: False)
    monkeypatch.setattr(local_runner, "_try_hardlink", lambda *_args: False)

    def symlink(source: str, target: str) -> bool:
        Path(target).symlink_to(source)
        return True

    monkeypatch.setattr(local_runner, "_try_symlink", symlink)

    strategy = local_runner._clone_or_link_ref_for_pair(str(ref), str(pair_ref))
    assert strategy == "symlink"
    assert pair_ref.is_symlink()

    _apply_pair_mask(casedir, pair_ref)

    with xr.open_dataset(ref) as ref_ds, xr.open_dataset(pair_ref) as pair_ds:
        assert not pair_ref.is_symlink()
        assert np.isfinite(ref_ds["rv"].values).all()
        assert np.isnan(pair_ds["rv"].values[1, 0, 0])
