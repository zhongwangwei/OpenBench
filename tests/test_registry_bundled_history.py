"""Bundled-catalog history: sparse overlay writes and the one-time legacy sync."""

from __future__ import annotations

import copy
import hashlib
from contextlib import contextmanager

import pytest
import yaml

from openbench.data.registry import bundled_history
from openbench.data.registry import manager as registry_manager
from openbench.data.registry import overlay_audit as oa
from openbench.data.registry.manager import REGISTRY_DIR, RegistryManager
from openbench.data.registry.scanner import _backup_then_write, _merge_descriptor_overlay


def _bundled_ref(name: str) -> dict:
    catalog = yaml.safe_load((REGISTRY_DIR / "reference_catalog.yaml").read_text(encoding="utf-8"))
    return copy.deepcopy(catalog[name])


def _write_overlays(user_dir, references: dict, models: dict | None = None):
    ref_path = user_dir / "references" / "reference_catalog.yaml"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.write_text(yaml.safe_dump(references, sort_keys=False))
    model_path = user_dir / "models" / "model_catalog.yaml"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(yaml.safe_dump(models or {}, sort_keys=False))
    return ref_path


def test_history_records_every_current_bundled_value():
    history = bundled_history.load_history()
    for kind, filename in (("references", "reference_catalog.yaml"), ("models", "model_catalog.yaml")):
        catalog = yaml.safe_load((REGISTRY_DIR / filename).read_text(encoding="utf-8"))
        missing = sorted(
            {
                name
                for name, entry in catalog.items()
                for digest in bundled_history.iter_entry_digests(kind, name, entry)
                if digest not in history[kind]
            }
        )
        assert not missing, (
            f"{filename} changed without updating bundled_history.json (entries: {missing[:5]}). "
            "Run: python scripts/build_bundled_registry_history.py"
        )


def test_sync_sparsifies_without_replacing_historical_values(tmp_path):
    legacy = _bundled_ref("CN05.1_MidRes")
    # The bundled catalog shipped this prefix before it was fixed to CN05.1_Tm_1961.
    legacy["variables"]["Surface_Air_Temperature"]["prefix"] = "CN05.1_Tm_"
    # Values the bundled catalog never held are user edits.
    legacy["variables"]["Precipitation"]["varunit"] = "kg m-2 s-1"
    legacy["root_dir"] = "/data/mine/MidRes"
    custom = {
        "name": "My_Custom",
        "data_type": "grid",
        "tim_res": "Month",
        "variables": {"Precipitation": {"varname": "pr", "varunit": "mm"}},
    }
    tombstone = {"name": "CLARA_3_LowRes", "_deleted": True}
    ref_path = _write_overlays(tmp_path, {"CN05.1_MidRes": legacy, "My_Custom": custom, "CLARA_3_LowRes": tombstone})

    results = oa.sync_legacy_overlays(tmp_path)

    assert [(r.label, r.compacted) for r in results] == [("references", ["CN05.1_MidRes"])]
    assert results[0].backup.exists()
    assert yaml.safe_load(results[0].backup.read_text())["CN05.1_MidRes"] == legacy
    assert yaml.safe_load(ref_path.read_text()) == {
        "CN05.1_MidRes": {
            "root_dir": "/data/mine/MidRes",
            "variables": {
                "Precipitation": {"varunit": "kg m-2 s-1"},
                "Surface_Air_Temperature": {"prefix": "CN05.1_Tm_"},
            },
        },
        "My_Custom": custom,
        "CLARA_3_LowRes": tombstone,
    }
    ref = RegistryManager(user_dir=tmp_path).get_reference("CN05.1_MidRes")
    assert ref.variables["Surface_Air_Temperature"].prefix == "CN05.1_Tm_"
    assert ref.variables["Precipitation"].varunit == "kg m-2 s-1"
    assert ref.root_dir == "/data/mine/MidRes"
    assert "without changing overrides" in oa.format_sync_notice(results)


def test_sync_runs_once_so_later_edits_stay(tmp_path):
    ref_path = _write_overlays(tmp_path, {})
    assert oa.sync_legacy_overlays(tmp_path) == []
    # Written after the sync, so deliberate even though it repeats an old bundled value.
    later = {"CN05.1_MidRes": {"variables": {"Surface_Air_Temperature": {"prefix": "CN05.1_Tm_"}}}}
    ref_path.write_text(yaml.safe_dump(later))

    assert oa.sync_legacy_overlays(tmp_path) == []
    assert yaml.safe_load(ref_path.read_text()) == later


def test_sync_preserves_intentional_historical_prefix(tmp_path):
    entry = {"root_dir": "/data/renamed", "variables": {"Surface_Air_Temperature": {"prefix": "CN05.1_Tm_"}}}
    path = _write_overlays(tmp_path, {"CN05.1_MidRes": entry})
    oa.sync_legacy_overlays(tmp_path)
    assert yaml.safe_load(path.read_text())["CN05.1_MidRes"] == entry


def test_sparse_paths_remain_pinned_when_environment_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBENCH_REF_ROOT", "/data/pinned")
    entry = {"root_dir": "/data/pinned/Grid/MidRes"}
    path = _write_overlays(tmp_path, {"CN05.1_MidRes": entry})
    oa.sync_legacy_overlays(tmp_path)
    assert yaml.safe_load(path.read_text())["CN05.1_MidRes"] == entry
    monkeypatch.setenv("OPENBENCH_REF_ROOT", "/data/moved")
    assert RegistryManager(user_dir=tmp_path).get_reference("CN05.1_MidRes").root_dir == entry["root_dir"]


def test_sync_skips_missing_user_dir(tmp_path):
    assert oa.sync_legacy_overlays(tmp_path / "absent") == []
    assert not (tmp_path / "absent").exists()


def test_overlay_writes_keep_only_changed_fields(tmp_path, monkeypatch):
    catalog = tmp_path / "references" / "reference_catalog.yaml"
    catalog.parent.mkdir()
    monkeypatch.setattr(registry_manager, "get_writable_reference_catalog_path", lambda: catalog)
    edited = _bundled_ref("CN05.1_MidRes")
    edited["variables"]["Surface_Air_Temperature"]["varunit"] = "K"

    _backup_then_write(catalog, {"CN05.1_MidRes": edited})

    assert yaml.safe_load(catalog.read_text()) == {
        "CN05.1_MidRes": {"variables": {"Surface_Air_Temperature": {"varunit": "K"}}}
    }
    # Other catalogs are written verbatim.
    other = tmp_path / "elsewhere.yaml"
    _backup_then_write(other, {"CN05.1_MidRes": edited})
    assert yaml.safe_load(other.read_text()) == {"CN05.1_MidRes": edited}


def test_sparse_delta_keeps_overlay_behavior():
    bundled = _bundled_ref("CN05.1_MidRes")
    overlay = copy.deepcopy(bundled)
    overlay["years"] = [1961, 2020]
    overlay["variables"]["Precipitation"]["prefix"] = "Pre_"

    delta = oa._sparse_delta(bundled, overlay, name="CN05.1_MidRes")

    assert delta == {"years": [1961, 2020], "variables": {"Precipitation": {"prefix": "Pre_"}}}
    base = oa._bundled_object("references", "CN05.1_MidRes", bundled)
    assert oa._same_behavior("references", base, delta, overlay)


def test_merge_descriptor_overlay_merges_variables_per_field():
    base = {"variables": {"A": {"varname": "a", "varunit": "1", "prefix": "p"}, "B": {"varname": "b"}}}
    overlay = {"variables": {"A": {"varunit": "K"}, "B": None}}

    merged = _merge_descriptor_overlay(base, overlay)

    assert merged["variables"] == {"A": {"varname": "a", "varunit": "K", "prefix": "p"}}


def test_historical_delta_warns_after_compaction_and_can_reset(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from openbench.cli.main import cli

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    base = tmp_path / ".openbench"
    legacy = _bundled_ref("CN05.1_MidRes")
    legacy["variables"]["Surface_Air_Temperature"]["prefix"] = "CN05.1_Tm_"
    path = _write_overlays(base, {"CN05.1_MidRes": legacy})
    original = path.read_bytes()
    runner = CliRunner()

    diff = runner.invoke(cli, ["registry", "diff"])
    assert diff.exit_code == 0, diff.output
    assert "matches older bundled defaults" in diff.output
    assert "deliberate" not in diff.output
    assert path.read_bytes() == original
    assert not oa._sync_marker(path).exists()
    assert runner.invoke(cli, ["registry", "prune", "--dry-run"]).exit_code == 0
    assert path.read_bytes() == original
    assert not oa._sync_marker(path).exists()

    oa.sync_legacy_overlays(base)
    assert "may be outdated" in oa.maybe_emit_overlay_notice(base)
    assert oa.maybe_emit_overlay_notice(base) is None
    assert (
        yaml.safe_load(path.read_text())["CN05.1_MidRes"]["variables"]["Surface_Air_Temperature"]["prefix"]
        == "CN05.1_Tm_"
    )
    reset = runner.invoke(cli, ["registry", "reset", "CN05.1_MidRes", "--yes"])
    assert reset.exit_code == 0, reset.output
    assert yaml.safe_load(path.read_text()) == {}
    assert "Backup:" in reset.output


def test_sync_resets_only_proven_untouched_seed(tmp_path):
    legacy = _bundled_ref("CN05.1_MidRes")
    legacy["variables"]["Surface_Air_Temperature"]["prefix"] = "CN05.1_Tm_"
    path = _write_overlays(tmp_path, {"CN05.1_MidRes": legacy})
    original = path.read_bytes()
    manifest = tmp_path / ".seeded_defaults.yaml"
    manifest.write_text(
        yaml.safe_dump({"references/reference_catalog.yaml": {"sha256": hashlib.sha256(original).hexdigest()}})
    )

    results = oa.sync_legacy_overlays(tmp_path)

    assert results[0].reset_seeded
    assert results[0].backup.read_bytes() == original
    assert yaml.safe_load(path.read_text()) == {}
    record = yaml.safe_load(manifest.read_text())["references/reference_catalog.yaml"]
    assert record["kind"] == "empty-overlay"
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_sync_preserves_edited_seed(tmp_path):
    path = _write_overlays(tmp_path, {"CN05.1_MidRes": {"description": "mine"}})
    (tmp_path / ".seeded_defaults.yaml").write_text(
        yaml.safe_dump({"references/reference_catalog.yaml": {"sha256": "old-hash"}})
    )
    oa.sync_legacy_overlays(tmp_path)
    assert yaml.safe_load(path.read_text()) == {"CN05.1_MidRes": {"description": "mine"}}


def test_sync_suppression_and_absent_catalog_have_no_side_effects(tmp_path, monkeypatch):
    path = _write_overlays(tmp_path, {"CN05.1_MidRes": _bundled_ref("CN05.1_MidRes")})
    original = path.read_bytes()
    monkeypatch.setenv("OPENBENCH_NO_REGISTRY_CHECK", "1")
    assert oa.sync_legacy_overlays(tmp_path) == []
    assert path.read_bytes() == original
    assert not oa._sync_marker(path).exists()
    monkeypatch.delenv("OPENBENCH_NO_REGISTRY_CHECK")
    absent = tmp_path / "absent"
    (absent / "models").mkdir(parents=True)
    assert oa.sync_legacy_overlays(absent) == []
    assert list((absent / "models").iterdir()) == []


def test_sync_readonly_failure_is_quiet(tmp_path, monkeypatch, caplog):
    from openbench.data.registry import scanner

    _write_overlays(tmp_path, {"CN05.1_MidRes": _bundled_ref("CN05.1_MidRes")})

    @contextmanager
    def denied(_path):
        raise PermissionError("read only")
        yield

    monkeypatch.setattr(scanner, "_catalog_write_lock", denied)
    assert oa.sync_legacy_overlays(tmp_path) == []
    assert oa.sync_legacy_overlays(tmp_path) == []
    assert not caplog.records


def test_incomparable_overlay_is_preserved_without_success_notice(tmp_path, caplog):
    invalid = {"variables": {"Soil_Evaporation": {"varname": "a"}, "Bare_Soil_Evaporation": {"varname": "b"}}}
    _write_overlays(tmp_path, {}, {"CoLM2024": invalid})
    path = tmp_path / "models" / "model_catalog.yaml"
    original = path.read_bytes()
    assert oa.sync_legacy_overlays(tmp_path) == []
    assert path.read_bytes() == original
    assert "could not be compared" in caplog.text
    assert "now follow" not in caplog.text


def test_duplicate_names_keep_sequential_merge_semantics(tmp_path):
    name = "CN05.1_MidRes"
    prefix = _bundled_ref(name)["variables"]["Surface_Air_Temperature"]["prefix"]
    entries = {
        name: {"variables": {"Surface_Air_Temperature": {"prefix": "mine_"}}},
        name.lower(): {"variables": {"Surface_Air_Temperature": {"prefix": prefix}}},
    }
    path = _write_overlays(tmp_path, entries)
    before = RegistryManager(user_dir=tmp_path).get_reference(name)
    assert oa.sparsify_overlay_catalog("references", entries) == entries
    oa.sync_legacy_overlays(tmp_path)
    oa.prune_overlays(tmp_path)
    after = RegistryManager(user_dir=tmp_path).get_reference(name)
    assert before == after
    assert yaml.safe_load(path.read_text()) == entries


def test_sparse_delta_preserves_meaningful_name_and_handles_alias_paths():
    bundled = {
        "name": "Demo",
        "variables": {"Bare_Soil_Evaporation": {"varname": "q", "varunit": "mm", "sub_dir": "data"}},
    }
    overlay = {"name": "Different", "variables": {"Soil_Evaporation": {"sub_dir": "data"}}}
    assert oa._sparse_delta(bundled, overlay, kind="models", name="Demo") == {"name": "Different"}


def test_sparse_time_offsets_keep_only_changed_nested_fields():
    bundled = {
        "name": "Demo",
        "variables": {},
        "time_offset": {"Hour": "-1 hours", "Day": {"default": "-1 days", "special": "-2 days"}},
    }
    overlay = {"time_offset": {"Hour": "-1 hours", "Day": {"default": "0", "special": "-2 days"}}}
    assert oa._sparse_delta(bundled, overlay, kind="models", name="Demo") == {"time_offset": {"Day": {"default": "0"}}}


@pytest.mark.skipif(__import__("os").name == "nt", reason="POSIX permission bits")
def test_catalog_atomic_writes_preserve_existing_permissions(tmp_path):
    from openbench.data.registry.scanner import _atomic_yaml_write

    path = tmp_path / "other.yaml"
    path.write_text("{}\n")
    path.chmod(0o660)
    _atomic_yaml_write(path, {"x": 1})
    assert path.stat().st_mode & 0o777 == 0o660
    RegistryManager._write_catalog(path, {"x": 2})
    assert path.stat().st_mode & 0o777 == 0o660


def test_overlay_classification_does_not_create_user_directories(tmp_path, monkeypatch):
    from openbench.config.user_settings import get_user_config_dir

    monkeypatch.setenv("HOME", str(tmp_path / "absent-home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "absent-home"))
    other = tmp_path / "other" / "reference_catalog.yaml"
    assert oa.overlay_kind_for_path(other) is None
    assert not get_user_config_dir().exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--help"],
        ["run", "missing.yaml", "--dry-run"],
        ["check", "missing.yaml"],
        ["version"],
        ["model", "list"],
        ["model", "show", "CLM5"],
        ["ref", "status"],
        ["registry", "diff"],
    ],
)
def test_read_only_invocations_leave_the_overlay_untouched(tmp_path, monkeypatch, argv):
    from click.testing import CliRunner

    from openbench.cli.main import cli

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    base = tmp_path / ".openbench"
    path = _write_overlays(base, {"CN05.1_MidRes": _bundled_ref("CN05.1_MidRes")})  # compaction would drop it
    original = path.read_bytes()

    CliRunner().invoke(cli, argv)

    assert path.read_bytes() == original
    assert not oa._sync_marker(path).exists()
    assert oa._sync_suspensions == 0  # the suspension ends with the command


def test_commands_that_change_the_registry_still_compact_once(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from openbench.cli.main import cli

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    base = tmp_path / ".openbench"
    path = _write_overlays(base, {"CN05.1_MidRes": _bundled_ref("CN05.1_MidRes")})

    result = CliRunner().invoke(cli, ["model", "register", "VIC5", "--description", "mine"])

    assert result.exit_code == 0, result.output
    assert oa._sync_marker(path).exists()
    assert "CN05.1_MidRes" not in (yaml.safe_load(path.read_text()) or {})


def test_sync_leaves_a_symlinked_overlay_alone(tmp_path):
    shared = tmp_path / "team" / "reference_catalog.yaml"
    shared.parent.mkdir()
    shared.write_text(yaml.safe_dump({"CN05.1_MidRes": _bundled_ref("CN05.1_MidRes")}, sort_keys=False))
    base = tmp_path / ".openbench"
    (base / "references").mkdir(parents=True)
    (base / "models").mkdir()
    link = base / "references" / "reference_catalog.yaml"
    try:
        link.symlink_to(shared)
    except OSError:
        pytest.skip("symlinks are not available")
    before = shared.read_bytes()

    assert oa.sync_legacy_overlays(base) == []

    assert link.is_symlink()
    assert shared.read_bytes() == before
    assert not oa._sync_marker(link).exists()


def _cli_home(tmp_path, monkeypatch):
    from click.testing import CliRunner

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("OPENBENCH_REF_ROOT", raising=False)
    return CliRunner(), tmp_path / ".openbench"


def test_reset_reports_overrides_in_separate_files_instead_of_success(tmp_path, monkeypatch):
    from openbench.cli.main import cli

    runner, base = _cli_home(tmp_path, monkeypatch)
    catalog = _write_overlays(base, {"CN05.1_MidRes": {"variables": {"Surface_Air_Temperature": {"prefix": "X_"}}}})
    separate = base / "references" / "CN05.1_MidRes.yaml"
    separate.write_text(yaml.safe_dump({"name": "CN05.1_MidRes", "variables": {"Precipitation": {"prefix": "Y_"}}}))

    diff = runner.invoke(cli, ["registry", "diff"])
    assert str(separate) in diff.output
    reset = runner.invoke(cli, ["registry", "reset", "CN05.1_MidRes", "--yes"])

    assert reset.exit_code != 0
    assert "still overridden by" in reset.output and str(separate) in reset.output
    assert yaml.safe_load(catalog.read_text()) == {}  # the catalog override itself was removed
    assert separate.exists()


def test_reset_detects_kind_and_skips_the_prompt_when_nothing_to_remove(tmp_path, monkeypatch):
    from openbench.cli.main import cli

    runner, base = _cli_home(tmp_path, monkeypatch)
    _write_overlays(base, {}, {"BCC_AVIM": {"description": "mine"}})

    untouched = runner.invoke(cli, ["registry", "reset", "VIC5"])  # no prompt answer needed
    assert untouched.exit_code == 0, untouched.output
    assert "already follows the bundled catalog" in untouched.output

    reset = runner.invoke(cli, ["registry", "reset", "BCC_AVIM"], input="y\n")
    assert reset.exit_code == 0, reset.output
    assert "description: mine" in reset.output  # shown before confirming
    assert RegistryManager(user_dir=base).get_model("BCC_AVIM").description != "mine"


def test_reset_reports_a_corrupt_catalog_without_traceback(tmp_path, monkeypatch):
    from openbench.cli.main import cli

    runner, base = _cli_home(tmp_path, monkeypatch)
    catalog = _write_overlays(base, {})
    catalog.write_text("CN05.1_MidRes: [unclosed\n")

    result = runner.invoke(cli, ["registry", "reset", "CN05.1_MidRes", "--yes"])

    assert result.exit_code == 1
    assert "Failed to load existing catalog" in result.output
    assert not isinstance(result.exception, (RuntimeError, KeyError))


def test_reset_explains_entries_no_longer_bundled(tmp_path, monkeypatch):
    from openbench.cli.main import cli

    runner, base = _cli_home(tmp_path, monkeypatch)
    old = {
        "InteractiveModel": {
            "name": "InteractiveModel",
            "description": "InteractiveModel model profile",
            "data_type": "grid",
            "grid_res": 0.5,
            "tim_res": "Month",
            "variables": {"Runoff": {"varname": "runoff_primary", "varunit": "mm day-1"}},
        }
    }
    _write_overlays(base, {}, old)

    diff = runner.invoke(cli, ["registry", "diff"])
    assert "no longer bundled" in diff.output
    reset = runner.invoke(cli, ["registry", "reset", "InteractiveModel", "--kind", "models", "--yes"])
    assert reset.exit_code != 0
    assert "no longer in the bundled catalog" in reset.output


def test_case_variant_keys_are_reported_and_notice_lists_every_problem(tmp_path, monkeypatch):
    from openbench.cli.main import cli

    runner, base = _cli_home(tmp_path, monkeypatch)
    stale = _bundled_ref("CN05.1_MidRes")
    stale["variables"]["Surface_Air_Temperature"]["prefix"] = "CN05.1_Tm_"
    _write_overlays(
        base,
        {
            "CN05.1_MidRes": stale,
            "GLEAM_v4.2a_LowRes": {"description": "x"},
            "gleam_v4.2a_lowres": {"description": "y"},
        },
    )

    diff = runner.invoke(cli, ["registry", "diff"])
    assert "differ only in case" in diff.output
    notice = oa.maybe_emit_overlay_notice(base)
    assert "shadows" in notice and "may be outdated" in notice and "differ only in case" in notice


def test_registry_inspection_does_not_warn_about_unset_reference_root(tmp_path, monkeypatch, caplog):
    from openbench.cli.main import cli
    from openbench.data.registry import manager

    runner, base = _cli_home(tmp_path, monkeypatch)
    monkeypatch.setattr(manager, "_UNRESOLVED_ENV_VARS_WARNED", set())
    _write_overlays(base, {"CN05.1_MidRes": _bundled_ref("CN05.1_MidRes")})

    with caplog.at_level("WARNING"):
        runner.invoke(cli, ["registry", "diff"])
        oa.sync_legacy_overlays(base)

    assert not [record for record in caplog.records if "is unset" in record.getMessage()]


def test_legacy_list_varname_copy_is_compacted_without_duplicate_fallbacks(tmp_path):
    from openbench.data.registry.manager import REGISTRY_DIR

    bundled = yaml.safe_load((REGISTRY_DIR / "model_catalog.yaml").read_text(encoding="utf-8"))
    copied = {"CaMa": {"variables": {"Dam_Outflow": copy.deepcopy(bundled["CaMa"]["variables"]["Dam_Outflow"])}}}
    _write_overlays(tmp_path, {}, copied)

    mapping = RegistryManager(user_dir=tmp_path).get_model("CaMa").variables["Dam_Outflow"]
    assert [fallback.varname for fallback in mapping.fallbacks] == ["outflw"]
    assert oa.sparsify_overlay_catalog("models", copied) == {}


def test_overlay_hints_name_outdated_overrides(tmp_path):
    stale = _bundled_ref("CN05.1_MidRes")
    stale["variables"]["Surface_Air_Temperature"]["prefix"] = "CN05.1_Tm_"
    _write_overlays(tmp_path, {"CN05.1_MidRes": stale})

    hints = oa.overlay_hints(tmp_path)

    assert any("CN05.1_MidRes" in hint and "may be outdated" in hint for hint in hints)
    assert any("copies the bundled catalog" in hint for hint in hints)
    assert oa.overlay_hints(tmp_path / "absent") == []


def test_registry_page_shows_overlay_hints_locally_and_from_remote_snapshot(qapp, monkeypatch):
    from PySide6.QtWidgets import QLabel

    from openbench.gui.pages.page_registry import PageRegistry
    from openbench.gui.remote_registry import RemoteRegistrySnapshot

    page = PageRegistry.__new__(PageRegistry)
    page.overlay_hint_label = QLabel()
    snapshot = RemoteRegistrySnapshot.__new__(RemoteRegistrySnapshot)
    snapshot._references, snapshot._models, snapshot._model_aliases, snapshot._var_index = {}, {}, {}, {}
    snapshot._replace({"references": [], "models": [], "overlay_hints": ["CN05.1_MidRes: prefix may be outdated"]})
    page._registry = lambda: snapshot

    page._refresh_overlay_hints()
    assert not page.overlay_hint_label.isHidden()
    assert "CN05.1_MidRes" in page.overlay_hint_label.text()

    snapshot._replace({"references": [], "models": []})  # an older server sends no hints
    page._refresh_overlay_hints()
    assert page.overlay_hint_label.isHidden()


def test_remote_snapshot_script_survives_servers_without_overlay_hints():
    from openbench.gui.remote_registry import _SNAPSHOT_SCRIPT

    assert "except Exception:\n    hints = []" in _SNAPSHOT_SCRIPT
    compile(_SNAPSHOT_SCRIPT, "<snapshot>", "exec")


def test_reset_detects_the_kind_of_entries_no_longer_bundled(tmp_path, monkeypatch):
    from openbench.cli.main import cli

    runner, base = _cli_home(tmp_path, monkeypatch)
    shipped = {  # as a 2026 release bundled it (a test had written it into the catalog)
        "name": "InteractiveModel",
        "description": "InteractiveModel model profile",
        "data_type": "grid",
        "grid_res": 0.5,
        "tim_res": "Month",
        "variables": {"Runoff": {"varname": "runoff_primary", "varunit": "mm day-1"}},
    }
    _write_overlays(base, {}, {"InteractiveModel": shipped})

    result = runner.invoke(cli, ["registry", "reset", "InteractiveModel", "--yes"])

    assert result.exit_code != 0
    assert "no longer in the bundled catalog" in result.output


def test_reset_refuses_a_linked_overlay(tmp_path, monkeypatch):
    from openbench.cli.main import cli

    runner, base = _cli_home(tmp_path, monkeypatch)
    shared = tmp_path / "team" / "reference_catalog.yaml"
    shared.parent.mkdir()
    shared.write_text(yaml.safe_dump({"CN05.1_MidRes": {"description": "team"}}))
    (base / "references").mkdir(parents=True)
    (base / "models").mkdir()
    try:
        (base / "references" / "reference_catalog.yaml").symlink_to(shared)
    except OSError:
        pytest.skip("symlinks are not available")
    before = shared.read_bytes()

    result = runner.invoke(cli, ["registry", "reset", "CN05.1_MidRes", "--yes"])

    assert result.exit_code != 0
    assert "links to" in result.output
    assert (base / "references" / "reference_catalog.yaml").is_symlink()
    assert shared.read_bytes() == before


def test_reset_of_an_equivalent_alias_resets_its_profile(tmp_path, monkeypatch):
    from openbench.cli.main import cli

    runner, base = _cli_home(tmp_path, monkeypatch)
    _write_overlays(base, {}, {"CoLM2024": {"description": "mine"}})

    result = runner.invoke(cli, ["registry", "reset", "CoLM", "--yes"])

    assert result.exit_code == 0, result.output
    assert "resetting CoLM2024" in result.output
    assert RegistryManager(user_dir=base).get_model("CoLM2024").description != "mine"


def test_compaction_keeps_keys_the_registry_does_not_merge():
    sparse = oa.sparsify_overlay_catalog("models", {"CoLM2024": {"notes": "why I changed it", "data_groupby": "Year"}})

    assert sparse == {"CoLM2024": {"notes": "why I changed it", "data_groupby": "Year"}}


def test_sync_skips_overlays_below_a_linked_directory_or_read_only(tmp_path):
    import os

    shared = tmp_path / "team_refs"
    shared.mkdir()
    (shared / "reference_catalog.yaml").write_text(yaml.safe_dump({"CN05.1_MidRes": _bundled_ref("CN05.1_MidRes")}))
    base = tmp_path / ".openbench"
    base.mkdir()
    (base / "models").mkdir()
    (base / "models" / "model_catalog.yaml").write_text(yaml.safe_dump({"CLM5": {"description": "x", "name": "CLM5"}}))
    try:
        (base / "references").symlink_to(shared, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are not available")
    before = (shared / "reference_catalog.yaml").read_bytes()
    os.chmod(base / "models" / "model_catalog.yaml", 0o444)
    try:
        assert oa.sync_legacy_overlays(base) == []
    finally:
        os.chmod(base / "models" / "model_catalog.yaml", 0o644)
    assert (shared / "reference_catalog.yaml").read_bytes() == before


def test_diff_reports_an_unreadable_overlay_instead_of_crashing(tmp_path, monkeypatch):
    from openbench.cli.main import cli

    runner, base = _cli_home(tmp_path, monkeypatch)
    catalog = _write_overlays(base, {})
    catalog.write_text("- a\n- b\n")

    result = runner.invoke(cli, ["registry", "diff"])

    assert result.exit_code == 0, result.output
    assert "is not a mapping of entries" in result.output
    assert "Overlay is clean" not in result.output


@pytest.mark.parametrize("argv", [["model"], ["model", "lst"], ["model", "alias"], ["smoke-test", "--help"], ["sim"]])
def test_bare_groups_and_unknown_subcommands_do_not_compact(tmp_path, monkeypatch, argv):
    from openbench.cli.main import cli

    runner, base = _cli_home(tmp_path, monkeypatch)
    path = _write_overlays(base, {"CN05.1_MidRes": _bundled_ref("CN05.1_MidRes")})
    original = path.read_bytes()

    runner.invoke(cli, argv)

    assert path.read_bytes() == original
    assert not oa._sync_marker(path).exists()


def test_registry_page_caps_long_hint_lists(qapp):
    from PySide6.QtWidgets import QLabel

    from openbench.gui.pages.page_registry import PageRegistry

    class Snapshot:
        overlay_hints = [f"Entry{index}: prefix matches older bundled defaults" for index in range(40)]

    page = PageRegistry.__new__(PageRegistry)
    page.overlay_hint_label = QLabel()
    page._registry = lambda: Snapshot()

    page._refresh_overlay_hints()

    assert page.overlay_hint_label.text().count("⚠") == 3
    assert "and 37 more" in page.overlay_hint_label.text()
    assert page.overlay_hint_label.toolTip().count("Entry") == 40


def test_separate_file_overrides_are_audited_everywhere(tmp_path, monkeypatch):
    from openbench.cli.main import cli

    runner, base = _cli_home(tmp_path, monkeypatch)
    _write_overlays(base, {})
    legacy = base / "references" / "legacy.yaml"
    stale = {"CN05.1_MidRes": {"variables": {"Surface_Air_Temperature": {"prefix": "CN05.1_Tm_"}}}}
    legacy.write_text(yaml.safe_dump(stale))
    (base / "references" / "mine.yaml").write_text(
        yaml.safe_dump({"name": "My_Data", "data_type": "grid", "variables": {"Precipitation": {"varname": "pr"}}})
    )

    diff = runner.invoke(cli, ["registry", "diff"])
    hints = oa.overlay_hints(base)
    notice = oa.maybe_emit_overlay_notice(base)

    assert diff.exit_code == 0, diff.output
    assert "Surface_Air_Temperature.prefix matches older bundled defaults" in diff.output
    assert "1 entry in separate overlay files still override" in diff.output
    assert "Overlay is clean" not in diff.output
    assert [hint for hint in hints if str(legacy) in hint and "older bundled defaults" in hint]
    assert not [hint for hint in hints if "mine.yaml" in hint]
    assert notice and "separate overlay files" in notice


def test_editing_only_a_separate_file_is_checked_again(tmp_path, monkeypatch):
    _runner, base = _cli_home(tmp_path, monkeypatch)
    _write_overlays(base, {})
    legacy = base / "references" / "legacy.yaml"
    legacy.write_text(yaml.safe_dump({"CN05.1_MidRes": {"root_dir": "/data/mine"}}))
    assert oa.maybe_emit_overlay_notice(base) is None  # a deliberate delta; the check is recorded

    legacy.write_text(
        yaml.safe_dump({"CN05.1_MidRes": {"variables": {"Surface_Air_Temperature": {"prefix": "CN05.1_Tm_"}}}})
    )

    assert "separate overlay files" in (oa.maybe_emit_overlay_notice(base) or "")


def test_custom_entries_in_separate_files_keep_the_overlay_clean(tmp_path, monkeypatch):
    from openbench.cli.main import cli

    runner, base = _cli_home(tmp_path, monkeypatch)
    _write_overlays(base, {})
    (base / "references" / "mine.yaml").write_text(
        yaml.safe_dump({"name": "My_Data", "data_type": "grid", "variables": {"Precipitation": {"varname": "pr"}}})
    )

    diff = runner.invoke(cli, ["registry", "diff"])

    assert "My_Data" in diff.output and "not in bundled" in diff.output
    assert "Overlay is clean" in diff.output


def _separate_temperature_file(base):
    legacy = base / "references" / "legacy.yaml"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(
        yaml.safe_dump({"CN05.1_MidRes": {"variables": {"Surface_Air_Temperature": {"prefix": "CN05.1_Tm_1961"}}}})
    )
    return legacy


def test_save_refuses_a_change_a_separate_file_would_undo(tmp_path, monkeypatch):
    from dataclasses import replace

    from openbench.data.registry.manager import RegistryManager

    _runner, base = _cli_home(tmp_path, monkeypatch)
    legacy = _separate_temperature_file(base)
    catalog = base / "references" / "reference_catalog.yaml"
    manager = RegistryManager(user_dir=base)
    ref = manager.get_reference("CN05.1_MidRes")
    temperature = ref.variables["Surface_Air_Temperature"]

    for edited in (
        replace(ref, variables={"Precipitation": ref.variables["Precipitation"]}),
        replace(ref, variables={**ref.variables, "Surface_Air_Temperature": replace(temperature, prefix="X_")}),
    ):
        with pytest.raises(oa.SeparateFileOverride) as caught:
            manager.save_reference(ref.name, edited)
        assert str(legacy) in str(caught.value) and "variables.Surface_Air_Temperature" in str(caught.value)
        assert not catalog.exists()

    manager.save_reference(ref.name, replace(ref, description="mine"))  # a field the file does not set
    assert yaml.safe_load(catalog.read_text()) == {"CN05.1_MidRes": {"description": "mine"}}


def test_delete_refuses_an_entry_a_separate_file_would_recreate(tmp_path, monkeypatch):
    from openbench.data.registry.manager import RegistryManager

    _runner, base = _cli_home(tmp_path, monkeypatch)
    _separate_temperature_file(base)

    with pytest.raises(oa.SeparateFileOverride, match="whole entry"):
        RegistryManager(user_dir=base).delete_reference("CN05.1_MidRes")


def test_delete_of_an_overridden_bundled_entry_survives_a_reload(tmp_path, monkeypatch):
    from openbench.data.registry.manager import RegistryManager

    _runner, base = _cli_home(tmp_path, monkeypatch)
    _write_overlays(base, {"CN05.1_MidRes": {"description": "mine"}}, models={"CLM5": {"description": "mine"}})

    manager = RegistryManager(user_dir=base)
    manager.delete_reference("CN05.1_MidRes")
    manager.delete_model("CLM5")

    reloaded = RegistryManager(user_dir=base)
    assert reloaded.get_reference("CN05.1_MidRes") is None
    assert reloaded.get_model("CLM5") is None


def test_cli_edit_undone_by_a_separate_file_fails_cleanly(tmp_path, monkeypatch):
    from openbench.cli.main import cli

    runner, base = _cli_home(tmp_path, monkeypatch)
    (base / "models").mkdir(parents=True)
    (base / "models" / "legacy.yaml").write_text(yaml.safe_dump({"CLM5": {"description": "from a file"}}))

    result = runner.invoke(cli, ["model", "register", "CLM5", "--description", "changed"])

    assert result.exit_code != 0
    assert "legacy.yaml" in result.output and "description" in result.output
    assert "Traceback" not in result.output
    assert not (base / "models" / "model_catalog.yaml").exists()


def test_batch_writes_only_warn_about_separate_files(tmp_path, monkeypatch, caplog):
    from openbench.data.registry.scanner import _backup_then_write

    _runner, base = _cli_home(tmp_path, monkeypatch)
    _separate_temperature_file(base)
    catalog = base / "references" / "reference_catalog.yaml"
    change = {"CN05.1_MidRes": {"variables": {"Surface_Air_Temperature": {"prefix": "X_"}}}}

    with caplog.at_level("WARNING"):
        _backup_then_write(catalog, change, separate_files="warn")

    assert yaml.safe_load(catalog.read_text()) == change
    assert "legacy.yaml" in caplog.text


def _separate_variable_files(base):
    (base / "references").mkdir(parents=True, exist_ok=True)
    (base / "models").mkdir(parents=True, exist_ok=True)
    (base / "references" / "legacy.yaml").write_text(
        yaml.safe_dump({"CN05.1_MidRes": {"variables": {"Evapotranspiration": {"varname": "et", "varunit": "mm"}}}})
    )
    (base / "models" / "legacy.yaml").write_text(
        yaml.safe_dump({"CLM5": {"variables": {"Extra_Var": {"varname": "extra", "varunit": "1"}}}})
    )


def test_deleting_a_variable_only_a_separate_file_adds_is_refused(tmp_path, monkeypatch):
    from dataclasses import replace

    from openbench.data.registry.manager import RegistryManager

    _runner, base = _cli_home(tmp_path, monkeypatch)
    _separate_variable_files(base)
    manager = RegistryManager(user_dir=base)
    ref = manager.get_reference("CN05.1_MidRes")
    model = manager.get_model("CLM5")

    with pytest.raises(oa.SeparateFileOverride, match="variables.Evapotranspiration"):
        manager.save_reference(
            ref.name, replace(ref, variables={k: v for k, v in ref.variables.items() if k != "Evapotranspiration"})
        )
    with pytest.raises(oa.SeparateFileOverride, match="variables.Extra_Var"):
        manager.save_model(
            "CLM5", replace(model, variables={k: v for k, v in model.variables.items() if k != "Extra_Var"})
        )

    # Keeping what the file adds while editing something else saves normally.
    manager.save_reference(ref.name, replace(ref, description="mine"))
    assert RegistryManager(user_dir=base).get_reference(ref.name).description == "mine"


def test_cli_remove_var_of_a_file_only_variable_is_refused(tmp_path, monkeypatch):
    from openbench.cli.main import cli

    runner, base = _cli_home(tmp_path, monkeypatch)
    _separate_variable_files(base)

    result = runner.invoke(cli, ["model", "remove-var", "CLM5", "Extra_Var"])

    assert result.exit_code != 0
    assert "variables.Extra_Var" in result.output and "Traceback" not in result.output
    assert not (base / "models" / "model_catalog.yaml").exists()


def test_deleting_a_dataset_only_a_separate_file_defines_is_refused(tmp_path, monkeypatch):
    from openbench.data.registry.manager import RegistryManager

    _runner, base = _cli_home(tmp_path, monkeypatch)
    (base / "references").mkdir(parents=True)
    (base / "references" / "mine.yaml").write_text(
        yaml.safe_dump({"name": "My_Data", "data_type": "grid", "tim_res": "Month", "variables": {}})
    )

    with pytest.raises(oa.SeparateFileOverride, match="whole entry"):
        RegistryManager(user_dir=base).delete_reference("My_Data")
