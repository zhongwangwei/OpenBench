"""Tests for resolve_all_references() — the single source of truth for
reference binding.

Covers:
  - exact name
  - base name auto-resolve
  - ambiguous (base name, no sim context)
  - not_found
  - no_variable
  - strict vs lenient mode
  - entry-point smoke tests (check, run --dry-run, adapter)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pytest

from openbench.config.resolver import (
    ResolvedReference,
    resolve_all_references,
    resolve_reference,
)
from openbench.config.schema import (
    EvaluationConfig,
    OpenBenchConfig,
    ProjectConfig,
    ReferenceConfig,
    SimulationEntry,
)
from openbench.data.registry.schema import ReferenceDataset, VariableMapping


# ---------------------------------------------------------------------------
# Fixtures: minimal config and fake registry
# ---------------------------------------------------------------------------

def _make_cfg(
    variables: list[str],
    reference: dict[str, str],
    strict: bool = False,
    sim_tim_res: str | None = None,
    sim_grid_res: float | None = None,
    comparison_tim_res: str | None = None,
    comparison_grid_res: float | None = None,
) -> OpenBenchConfig:
    sim_entry = SimulationEntry(
        model="test_model",
        root_dir="/tmp/sim",
        tim_res=sim_tim_res,
        grid_res=sim_grid_res,
    )
    return OpenBenchConfig(
        project=ProjectConfig(
            name="test", output_dir="/tmp/out", years=[2000, 2010],
            strict_reference=strict,
            tim_res=comparison_tim_res,
            grid_res=comparison_grid_res,
        ),
        evaluation=EvaluationConfig(variables=variables),
        reference=ReferenceConfig(sources=reference),
        simulation={"sim1": sim_entry},
    )


def _make_cfg_multi_sim(
    variables: list[str],
    reference: dict[str, str],
    simulations: dict[str, tuple[str | None, float | None]],
    strict: bool = False,
    comparison_tim_res: str | None = None,
    comparison_grid_res: float | None = None,
) -> OpenBenchConfig:
    sim_entries = {
        label: SimulationEntry(
            model="test_model",
            root_dir=f"/tmp/{label}",
            tim_res=tim_res,
            grid_res=grid_res,
        )
        for label, (tim_res, grid_res) in simulations.items()
    }
    return OpenBenchConfig(
        project=ProjectConfig(
            name="test", output_dir="/tmp/out", years=[2000, 2010],
            strict_reference=strict,
            tim_res=comparison_tim_res,
            grid_res=comparison_grid_res,
        ),
        evaluation=EvaluationConfig(variables=variables),
        reference=ReferenceConfig(sources=reference),
        simulation=sim_entries,
    )


class FakeRegistry:
    """Minimal registry that returns pre-configured datasets."""

    def __init__(self, datasets: dict[str, ReferenceDataset]):
        self._datasets = {k.lower(): v for k, v in datasets.items()}

    def get_reference(
        self,
        name: str,
        sim_tim_res: str | None = None,
        sim_grid_res: float | None = None,
    ) -> ReferenceDataset | None:
        key = name.lower()
        if key in self._datasets:
            return self._datasets[key]
        variants = self.get_resolution_variants(name)
        if variants and (sim_tim_res or sim_grid_res is not None):
            if sim_grid_res is not None:
                for ref in variants.values():
                    if ref.grid_res == sim_grid_res:
                        return ref
            if sim_tim_res is not None:
                for ref in variants.values():
                    if ref.tim_res == sim_tim_res:
                        return ref
            for suffix in ("LowRes", "MidRes", "HigRes", "default"):
                if suffix in variants:
                    return variants[suffix]
        return None

    def get_resolution_variants(self, base_name: str) -> dict[str, ReferenceDataset]:
        key = base_name.lower()
        variants = {}
        for suffix in ("_lowres", "_midres", "_higres"):
            full = f"{key}{suffix}"
            if full in self._datasets:
                variants[suffix[1:]] = self._datasets[full]
        return variants

    def get_model(self, name: str):
        return None


def _make_ref(
    name: str,
    variables: dict[str, VariableMapping] | None = None,
    tim_res: str = "Month",
    grid_res: float | None = 0.5,
    data_type: str = "grid",
) -> ReferenceDataset:
    if variables is None:
        variables = {}
    return ReferenceDataset(
        name=name,
        description=f"{name} test",
        category="Water",
        data_type=data_type,
        tim_res=tim_res,
        data_groupby="Year",
        timezone=0,
        years=[2000, 2010],
        variables=variables,
        grid_res=grid_res,
    )


def _var(varname: str, varunit: str = "") -> VariableMapping:
    return VariableMapping(varname=varname, varunit=varunit)


# ---------------------------------------------------------------------------
# resolve_reference unit tests
# ---------------------------------------------------------------------------


class TestResolveReference:
    """Tests for the single-variable resolve_reference() function."""

    def test_exact_name_ok(self):
        reg = FakeRegistry({
            "GLEAM_v4.2a_LowRes": _make_ref(
                "GLEAM_v4.2a_LowRes",
                variables={"Evapotranspiration": _var("E", "mm day-1")},
            ),
        })
        r = resolve_reference("Evapotranspiration", "GLEAM_v4.2a_LowRes", reg)
        assert r.status == "ok"
        assert r.resolved_name == "GLEAM_v4.2a_LowRes"
        assert r.var_map.varname == "E"

    def test_base_name_auto_resolve(self):
        reg = FakeRegistry({
            "GLEAM_v4.2a_LowRes": _make_ref(
                "GLEAM_v4.2a_LowRes",
                variables={"Evapotranspiration": _var("E")},
            ),
        })
        r = resolve_reference(
            "Evapotranspiration", "GLEAM_v4.2a", reg,
            target_tim_res="Month", target_grid_res=0.5,
        )
        assert r.status == "ok"
        assert r.resolved_name == "GLEAM_v4.2a_LowRes"

    def test_ambiguous_no_sim_context(self):
        reg = FakeRegistry({
            "TEST_LowRes": _make_ref("TEST_LowRes"),
            "TEST_MidRes": _make_ref("TEST_MidRes"),
        })
        r = resolve_reference("SomeVar", "TEST", reg)
        assert r.status == "ambiguous"
        assert "multiple resolutions" in r.message.lower() or "resolutions" in r.message.lower()

    def test_not_found(self):
        reg = FakeRegistry({})
        r = resolve_reference("SomeVar", "NonExistent", reg)
        assert r.status == "not_found"
        assert "not found" in r.message.lower()

    def test_no_variable(self):
        reg = FakeRegistry({
            "ERA5": _make_ref("ERA5", variables={"Precipitation": _var("tp")}),
        })
        r = resolve_reference("Evapotranspiration", "ERA5", reg)
        assert r.status == "no_variable"
        assert "Evapotranspiration" in r.message


# ---------------------------------------------------------------------------
# resolve_all_references tests
# ---------------------------------------------------------------------------


class TestResolveAllReferences:
    """Tests for the multi-variable resolve_all_references() function."""

    def test_all_ok(self):
        reg = FakeRegistry({
            "GLEAM_v4.2a_LowRes": _make_ref(
                "GLEAM_v4.2a_LowRes",
                variables={
                    "Evapotranspiration": _var("E"),
                    "Surface_Soil_Moisture": _var("SMs"),
                },
            ),
        })
        cfg = _make_cfg(
            variables=["Evapotranspiration", "Surface_Soil_Moisture"],
            reference={
                "Evapotranspiration": "GLEAM_v4.2a_LowRes",
                "Surface_Soil_Moisture": "GLEAM_v4.2a_LowRes",
            },
        )
        results = resolve_all_references(cfg, reg)
        assert all(r.status == "ok" for r in results)
        assert len(results) == 2

    def test_mixed_statuses_lenient(self):
        reg = FakeRegistry({
            "GLEAM_v4.2a_LowRes": _make_ref(
                "GLEAM_v4.2a_LowRes",
                variables={"Evapotranspiration": _var("E")},
            ),
        })
        cfg = _make_cfg(
            variables=["Evapotranspiration", "Missing_Var"],
            reference={
                "Evapotranspiration": "GLEAM_v4.2a_LowRes",
                "Missing_Var": "NonExistent",
            },
            strict=False,
        )
        results = resolve_all_references(cfg, reg)
        statuses = {r.var_name: r.status for r in results}
        assert statuses["Evapotranspiration"] == "ok"
        assert statuses["Missing_Var"] == "not_found"

    def test_strict_mode_raises_on_not_found(self):
        from openbench.config import ConfigError

        reg = FakeRegistry({})
        cfg = _make_cfg(
            variables=["Evapotranspiration"],
            reference={"Evapotranspiration": "NonExistent"},
            strict=True,
        )
        with pytest.raises(ConfigError, match="strict_reference"):
            resolve_all_references(cfg, reg)

    def test_strict_mode_raises_on_no_variable(self):
        from openbench.config import ConfigError

        reg = FakeRegistry({
            "ERA5": _make_ref("ERA5", variables={"Precipitation": _var("tp")}),
        })
        cfg = _make_cfg(
            variables=["Evapotranspiration"],
            reference={"Evapotranspiration": "ERA5"},
            strict=True,
        )
        with pytest.raises(ConfigError, match="strict_reference"):
            resolve_all_references(cfg, reg)

    def test_strict_mode_raises_on_ambiguous(self):
        from openbench.config import ConfigError

        reg = FakeRegistry({
            "TEST_LowRes": _make_ref("TEST_LowRes"),
            "TEST_MidRes": _make_ref("TEST_MidRes"),
        })
        cfg = _make_cfg(
            variables=["SomeVar"],
            reference={"SomeVar": "TEST"},
            strict=True,
        )
        with pytest.raises(ConfigError, match="strict_reference"):
            resolve_all_references(cfg, reg)

    def test_strict_mode_raises_on_low_confidence_provenance(self):
        from openbench.config import ConfigError

        ref = _make_ref(
            "GLEAM_v4.2a_LowRes",
            variables={"Evapotranspiration": _var("E")},
        )
        ref._provenance = {"tim_res": "default", "grid_res": "nc"}
        reg = FakeRegistry({"GLEAM_v4.2a_LowRes": ref})
        cfg = _make_cfg(
            variables=["Evapotranspiration"],
            reference={"Evapotranspiration": "GLEAM_v4.2a_LowRes"},
            strict=True,
        )

        with pytest.raises(ConfigError, match="strict_reference"):
            resolve_all_references(cfg, reg)

    def test_no_reference_configured(self):
        reg = FakeRegistry({})
        cfg = _make_cfg(
            variables=["Evapotranspiration"],
            reference={},  # no ref configured
        )
        results = resolve_all_references(cfg, reg, strict=False)
        assert results[0].status == "not_found"
        assert results[0].source_name == ""

    def test_auto_resolve_uses_comparison_resolution(self):
        reg = FakeRegistry({
            "GLEAM_v4.2a_LowRes": _make_ref(
                "GLEAM_v4.2a_LowRes",
                variables={"Evapotranspiration": _var("E")},
            ),
        })
        cfg = _make_cfg(
            variables=["Evapotranspiration"],
            reference={"Evapotranspiration": "GLEAM_v4.2a"},
            comparison_tim_res="Month",
            comparison_grid_res=0.5,
        )
        results = resolve_all_references(cfg, reg)
        assert results[0].status == "ok"
        assert results[0].resolved_name == "GLEAM_v4.2a_LowRes"

    def test_auto_resolve_uses_sim_resolution_fallback(self):
        reg = FakeRegistry({
            "GLEAM_v4.2a_LowRes": _make_ref(
                "GLEAM_v4.2a_LowRes",
                variables={"Evapotranspiration": _var("E")},
            ),
        })
        cfg = _make_cfg(
            variables=["Evapotranspiration"],
            reference={"Evapotranspiration": "GLEAM_v4.2a"},
            sim_tim_res="Month",
            sim_grid_res=0.5,
        )
        results = resolve_all_references(cfg, reg)
        assert results[0].status == "ok"

    def test_explicit_comparison_resolution_wins_with_mixed_simulations(self):
        reg = FakeRegistry({
            "CARE_LowRes": _make_ref("CARE_LowRes", variables={"Evapotranspiration": _var("E")}, tim_res="Day", grid_res=0.1),
            "CARE_MidRes": _make_ref("CARE_MidRes", variables={"Evapotranspiration": _var("E")}, tim_res="Month", grid_res=0.25),
        })
        cfg = _make_cfg_multi_sim(
            variables=["Evapotranspiration"],
            reference={"Evapotranspiration": "CARE"},
            simulations={
                "sim_daily": ("Day", 0.1),
                "sim_monthly": ("Month", 0.25),
            },
            comparison_tim_res="Month",
            comparison_grid_res=0.25,
        )

        results = resolve_all_references(cfg, reg)
        assert results[0].status == "ok"
        assert results[0].resolved_name == "CARE_MidRes"

    def test_multi_sim_resolution_conflict_requires_explicit_comparison_target(self):
        from openbench.config import ConfigError

        reg = FakeRegistry({
            "CARE_LowRes": _make_ref("CARE_LowRes", variables={"Evapotranspiration": _var("E")}, tim_res="Day", grid_res=0.1),
            "CARE_MidRes": _make_ref("CARE_MidRes", variables={"Evapotranspiration": _var("E")}, tim_res="Month", grid_res=0.25),
        })
        cfg = _make_cfg_multi_sim(
            variables=["Evapotranspiration"],
            reference={"Evapotranspiration": "CARE"},
            simulations={
                "sim_daily": ("Day", 0.1),
                "sim_monthly": ("Month", 0.25),
            },
            strict=True,
        )

        with pytest.raises(ConfigError, match="project"):
            resolve_all_references(cfg, reg)


# ---------------------------------------------------------------------------
# Entry-point smoke tests
# ---------------------------------------------------------------------------


class TestCheckEntrySmoke:
    """Verify check.py wires through to resolver and consumes provenance."""

    def test_check_invokes_resolver(self, tmp_path):
        """check command should call resolve_all_references and display results."""
        from unittest.mock import patch

        from click.testing import CliRunner

        from openbench.cli.check import check

        fake_ref = _make_ref(
            "TestRef",
            variables={"Evapotranspiration": _var("E")},
        )
        fake_resolved = ResolvedReference(
            var_name="Evapotranspiration",
            source_name="TestRef",
            resolved_name="TestRef",
            ref_ds=fake_ref,
            var_map=_var("E"),
            status="ok",
            provenance="registry",
        )

        cfg = _make_cfg(
            variables=["Evapotranspiration"],
            reference={"Evapotranspiration": "TestRef"},
        )

        config_file = tmp_path / "test.yaml"
        config_file.write_text("dummy: true")

        # Patch at the source modules (lazy imports inside function body)
        with patch("openbench.config.load_config", return_value=cfg), \
             patch("openbench.config.resolver.resolve_all_references", return_value=[fake_resolved]) as mock_resolve, \
             patch("openbench.data.registry.manager.get_registry", return_value=FakeRegistry({})):
            runner = CliRunner()
            result = runner.invoke(check, [str(config_file)])

        mock_resolve.assert_called_once()
        assert "Evapotranspiration" in result.output


class TestRunDryRunSmoke:
    """Verify run --dry-run wires through to resolver."""

    def test_dry_run_invokes_resolver(self, tmp_path):
        from unittest.mock import patch

        from click.testing import CliRunner

        from openbench.cli.run import run

        fake_ref = _make_ref("TestRef", variables={"ET": _var("E")})
        fake_resolved = ResolvedReference(
            var_name="ET",
            source_name="TestRef",
            resolved_name="TestRef",
            ref_ds=fake_ref,
            var_map=_var("E"),
            status="ok",
            provenance="registry",
        )

        cfg = _make_cfg(
            variables=["ET"],
            reference={"ET": "TestRef"},
        )

        config_file = tmp_path / "test.yaml"
        config_file.write_text("dummy: true")

        with patch("openbench.config.load_config", return_value=cfg), \
             patch("openbench.config.resolver.resolve_all_references", return_value=[fake_resolved]) as mock_resolve, \
             patch("openbench.data.registry.manager.get_registry", return_value=FakeRegistry({})):
            runner = CliRunner()
            result = runner.invoke(run, [str(config_file), "--dry-run"])

        mock_resolve.assert_called_once()
        assert "ET" in result.output


class TestAdapterSmoke:
    """Verify adapter.build_legacy_namelists wires through to resolver."""

    def test_adapter_calls_resolver(self):
        from unittest.mock import MagicMock, patch

        from openbench.config.adapter import build_legacy_namelists

        fake_ref = _make_ref(
            "TestRef",
            variables={"Evapotranspiration": _var("E", "mm day-1")},
            data_type="grid",
        )
        fake_ref.root_dir = "/data/ref"

        fake_resolved = ResolvedReference(
            var_name="Evapotranspiration",
            source_name="TestRef",
            resolved_name="TestRef",
            ref_ds=fake_ref,
            var_map=_var("E", "mm day-1"),
            status="ok",
            provenance="registry",
        )

        cfg = _make_cfg(
            variables=["Evapotranspiration"],
            reference={"Evapotranspiration": "TestRef"},
        )

        fake_registry = FakeRegistry({})
        fake_model = MagicMock()
        fake_model.variables = {}
        fake_model.data_type = "grid"
        fake_model.grid_res = 0.5
        fake_model.tim_res = "Month"
        fake_model.time_offset = None

        with patch("openbench.config.resolver.resolve_all_references", return_value=[fake_resolved]) as mock_resolve, \
             patch("openbench.data.registry.manager.get_registry", return_value=fake_registry), \
             patch.object(fake_registry, "get_model", return_value=fake_model):
            try:
                main_nl, ref_nml, sim_nml = build_legacy_namelists(cfg)
            except (KeyError, AttributeError, TypeError):
                pass  # adapter may fail on missing fields; we only test wiring

        mock_resolve.assert_called_once()


def test_resolve_all_references_multi_source_per_variable():
    """v3 schema accepts list[str] per variable; resolver returns one ResolvedReference per source.

    Regression for the v3.0a1 multi-ref capability lost during schema simplification.
    """
    from openbench.config.schema import (
        OpenBenchConfig, ProjectConfig, EvaluationConfig,
        ReferenceConfig, SimulationEntry,
    )
    from openbench.config.resolver import resolve_all_references
    from unittest.mock import MagicMock

    cfg = OpenBenchConfig(
        project=ProjectConfig(
            name="test", output_dir="/tmp/x", years=[2010, 2014],
            tim_res="Month", grid_res=0.5,
        ),
        evaluation=EvaluationConfig(variables=["ET"]),
        reference=ReferenceConfig(sources={"ET": ["GLEAM", "FLUXCOM"]}),
        simulation={"M": SimulationEntry(model="M", root_dir="/d")},
    )

    # Mock registry to return distinct ReferenceDataset per source
    registry = MagicMock()
    fake_ds = MagicMock()
    fake_ds.name = "fake"
    fake_ds.variables = {"ET": MagicMock()}
    fake_ds._provenance = {}
    registry.get_reference.return_value = fake_ds
    registry.last_resolve_reason = ""

    results = resolve_all_references(cfg, registry, strict=False)

    # 2 ResolvedReference for the same variable
    assert len(results) == 2
    assert all(r.var_name == "ET" for r in results)
    source_names = [r.source_name for r in results]
    assert source_names == ["GLEAM", "FLUXCOM"]
