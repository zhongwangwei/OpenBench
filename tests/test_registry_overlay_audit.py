"""Tests for registry overlay hygiene: classify, behavior-preserving prune, throttle."""

from __future__ import annotations

import copy

import yaml

from openbench.data.registry import manager as registry_manager
from openbench.data.registry import overlay_audit as oa


def _bundled_ref(name: str) -> dict:
    bundled = yaml.safe_load((registry_manager.REGISTRY_DIR / "reference_catalog.yaml").read_text())
    key = next(k for k in bundled if k.lower() == name.lower())
    return copy.deepcopy(bundled[key]), key


def _write_overlay(user_dir, entries: dict) -> None:
    refs = user_dir / "references"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "reference_catalog.yaml").write_text(yaml.safe_dump(entries, sort_keys=False))
    models = user_dir / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / "model_catalog.yaml").write_text("{}\n")


def test_classify_four_kinds(tmp_path):
    clara, _ = _bundled_ref("CLARA_3_LowRes")
    era, _ = _bundled_ref("ERA5LAND_LowRes")

    # stale full-copy: full bundled copy of ERA with one variable's unit changed
    stale_era = copy.deepcopy(era)
    a_var = next(iter(stale_era["variables"]))
    stale_era["variables"][a_var] = {**stale_era["variables"][a_var], "varunit": "STALE"}

    overlay = {
        "CLARA_3_LowRes": copy.deepcopy(clara),  # redundant (verbatim copy)
        "ERA5LAND_LowRes": stale_era,  # stale full-copy
        "GRFR_LowRes": {"description": "tweaked"},  # sparse delta (one field differs)
        "My_Custom_Ref": {"description": "x", "data_type": "grid", "variables": {"Latent_Heat": {"varname": "LE"}}},
    }
    _write_overlay(tmp_path, overlay)

    audit = oa.audit_overlays(tmp_path)
    kinds = {e.name: e.kind for e in audit.references.entries}
    assert kinds["CLARA_3_LowRes"] == oa.REDUNDANT
    assert kinds["ERA5LAND_LowRes"] == oa.STALE_FULLCOPY
    assert kinds["GRFR_LowRes"] == oa.DELTA
    assert kinds["My_Custom_Ref"] == oa.CUSTOM


def test_prune_is_behavior_preserving(tmp_path):
    clara, _ = _bundled_ref("CLARA_3_LowRes")
    gleam, _ = _bundled_ref("GLEAM_v4.2a_LowRes")

    redundant = copy.deepcopy(clara)
    stale = copy.deepcopy(gleam)
    # change one var unit to make it a stale full-copy
    a_var = next(iter(stale["variables"]))
    stale["variables"][a_var] = {**stale["variables"][a_var], "varunit": "CHANGED"}

    _write_overlay(tmp_path, {"CLARA_3_LowRes": redundant, "GLEAM_v4.2a_LowRes": stale})

    def merged_units():
        mgr = registry_manager.RegistryManager(user_dir=tmp_path)
        out = {}
        for name in ("CLARA_3_LowRes", "GLEAM_v4.2a_LowRes"):
            r = mgr.get_reference(name)
            out[name] = {vk: (vv.varname, vv.varunit) for vk, vv in r.variables.items()}
        return out

    before = merged_units()
    results = oa.prune_overlays(tmp_path, dry_run=False)
    after = merged_units()

    assert before == after  # behavior preserved
    pruned = yaml.safe_load((tmp_path / "references" / "reference_catalog.yaml").read_text())
    assert "CLARA_3_LowRes" not in pruned  # redundant dropped
    assert "GLEAM_v4.2a_LowRes" in pruned  # stale kept but minimized
    assert set(pruned["GLEAM_v4.2a_LowRes"].keys()) == {"variables"}  # reduced to just the delta
    ref_result = next(r for r in results if r.label == "references")
    assert ref_result.removed == ["CLARA_3_LowRes"]
    assert ref_result.minimized == ["GLEAM_v4.2a_LowRes"]
    assert ref_result.backup is not None


def test_prune_dry_run_writes_nothing(tmp_path):
    clara, _ = _bundled_ref("CLARA_3_LowRes")
    _write_overlay(tmp_path, {"CLARA_3_LowRes": copy.deepcopy(clara)})
    original = (tmp_path / "references" / "reference_catalog.yaml").read_text()
    oa.prune_overlays(tmp_path, dry_run=True)
    assert (tmp_path / "references" / "reference_catalog.yaml").read_text() == original


def test_notice_throttle_and_suppress(tmp_path, monkeypatch):
    clara, _ = _bundled_ref("CLARA_3_LowRes")
    _write_overlay(tmp_path, {"CLARA_3_LowRes": copy.deepcopy(clara)})

    monkeypatch.delenv(oa._SUPPRESS_ENV, raising=False)
    # First call: bloat present -> message emitted
    msg1 = oa.maybe_emit_overlay_notice(tmp_path)
    assert msg1 and "registry overlay shadows" in msg1
    # Second call: fingerprint unchanged -> throttled (silent)
    assert oa.maybe_emit_overlay_notice(tmp_path) is None
    # Changing the overlay re-triggers
    _write_overlay(tmp_path, {"CLARA_3_LowRes": copy.deepcopy(clara), "ERA5LAND_LowRes": copy.deepcopy(clara)})
    assert oa.maybe_emit_overlay_notice(tmp_path) is not None

    # Suppress env disables entirely
    monkeypatch.setenv(oa._SUPPRESS_ENV, "1")
    _write_overlay(
        tmp_path,
        {"CLARA_3_LowRes": {"variables": {"Surface_Albedo": {"varunit": "x"}}}, "GRFR_LowRes": copy.deepcopy(clara)},
    )
    assert oa.maybe_emit_overlay_notice(tmp_path) is None


def test_notice_silent_when_clean(tmp_path, monkeypatch):
    monkeypatch.delenv(oa._SUPPRESS_ENV, raising=False)
    _write_overlay(tmp_path, {})  # empty overlay
    assert oa.maybe_emit_overlay_notice(tmp_path) is None
    # a deliberate sparse delta is NOT bloat -> still silent
    _write_overlay(tmp_path, {"CLARA_3_LowRes": {"variables": {"Surface_Albedo": {"varunit": "1"}}}})
    assert oa.maybe_emit_overlay_notice(tmp_path) is None
