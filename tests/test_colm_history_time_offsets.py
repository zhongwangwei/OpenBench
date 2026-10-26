"""More detailed history-provenance and cache regression coverage."""

from types import SimpleNamespace

import pandas as pd
import pytest
import xarray as xr

from openbench.data._processing_time_adjustments import (
    TimeAdjustmentMixin,
    resolve_time_offset,
)
from openbench.data._processing_utils import parse_time_offset
from openbench.data.registry.schema import ModelProfile


def test_longest_filename_pattern_and_directory_name_ignored():
    profile = ModelProfile(
        name="Demo",
        description="Test history filename provenance",
        time_offset={
            "_hist_": {"Day": "0"},
            "_hist_cama_": {"Day": "-1 days"},
            "default": {"Day": "0"},
        },
    )
    data = xr.DataArray([1], dims="time", coords={"time": [pd.Timestamp("2001-01-02")]}, name="x")
    offset, matched = resolve_time_offset(
        profile, "Day", data, "x", source_files="/tmp/_hist_cama_/run_hist_unitcat_2001.nc"
    )
    assert (offset, matched) == ("0", "_hist_")
    offset, matched = resolve_time_offset(profile, "Day", data, "x", source_files="/tmp/run_hist_cama_2001.nc")
    assert (offset, matched) == ("-1 days", "_hist_cama_")


@pytest.mark.parametrize("text", ["bad", "-1 fortnights", "-1 days later", "1.5 hours"])
def test_invalid_nonzero_time_offset_fails(text):
    with pytest.raises(ValueError, match="Invalid time offset"):
        parse_time_offset(text)


def test_time_offset_years_supported():
    assert pd.Timestamp("2002-01-01") + parse_time_offset("-1 years") == pd.Timestamp("2001-01-01")


def test_legacy_resolution_specific_rule_still_supported(monkeypatch):
    from openbench.data.registry import manager

    profile = ModelProfile(
        name="Demo",
        description="Test legacy resolution offsets",
        time_offset={"Day": {"default": "-1 days", "special": "0"}},
    )
    monkeypatch.setattr(manager, "get_registry", lambda: SimpleNamespace(get_model=lambda model: profile))
    processor = SimpleNamespace(sim_source="Demo", item="Any")
    data = xr.DataArray([42], dims="time", coords={"time": [pd.Timestamp("2001-01-02 12:00")]}, name="a")
    shifted = TimeAdjustmentMixin.apply_model_specific_time_adjustment(processor, data, "sim", 2001, 2001, "Day")
    # Legacy no-source callers also normalize timestamps to HH:30.
    assert pd.Timestamp(shifted.time.values[0]) == pd.Timestamp("2001-01-01 12:30")


def test_cache_hash_records_effective_model_offset(monkeypatch):
    from openbench.data.registry import manager
    from openbench.runner import hashing

    state = {"offset": "0"}
    monkeypatch.setattr(
        manager,
        "get_registry",
        lambda: SimpleNamespace(
            get_model=lambda name: SimpleNamespace(time_offset={"Day": {"default": state["offset"]}})
        ),
    )
    first = hashing.effective_model_time_offset("CoLM2024")
    state["offset"] = "-1 days"
    second = hashing.effective_model_time_offset("CoLM2024")
    assert first != second
    assert (
        hashing.EvaluationCache.hash_config(first) != hashing.EvaluationCache.hash_config(second)
        if hasattr(hashing, "EvaluationCache")
        else first != second
    )
