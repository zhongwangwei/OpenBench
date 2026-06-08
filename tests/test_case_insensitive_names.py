from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
import yaml

from openbench.config import ConfigError, load_config
from openbench.data.compute import execute_compute
from openbench.data.registry.manager import RegistryManager
from openbench.data.registry.scanner import _child_dir_case_insensitive
from openbench.data.station_matcher import _require_dataset_field
from openbench.util.exceptions import DataProcessingError
from openbench.util.names import (
    AmbiguousNameError,
    NameResolutionError,
    get_xarray_key_case_insensitive,
    resolve_many_case_insensitive,
    resolve_name_case_insensitive,
)


def test_config_normalizes_reference_and_simulation_variable_keys(tmp_path):
    config_path = tmp_path / "openbench.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {
                    "name": "case",
                    "output_dir": str(tmp_path / "out"),
                    "years": [2001, 2002],
                },
                "evaluation": {"variables": ["Latent_Heat"]},
                "reference": {"latent_heat": "FluxRef"},
                "simulation": {
                    "CaseA": {
                        "model": "DemoModel",
                        "root_dir": str(tmp_path / "sim"),
                        "variables": {
                            "LATENT_HEAT": {"varname": "qle", "varunit": "W m-2"},
                        },
                    }
                },
            },
            sort_keys=False,
        )
    )

    cfg = load_config(config_path)

    assert cfg.reference.sources == {"Latent_Heat": "FluxRef"}
    assert cfg.simulation["CaseA"].variables == {"Latent_Heat": {"varname": "qle", "varunit": "W m-2"}}


def test_compute_dataset_lookup_is_exact_first_case_insensitive():
    ds = xr.Dataset({"Qle": ("time", np.array([1.0, 2.0]))})

    result = execute_compute(ds, "ds['qle'] * 2", "Latent_Heat")

    assert result.name == "Qle"
    assert result.values.tolist() == [2.0, 4.0]


def test_reference_variable_index_matches_ignoring_case(tmp_path):
    catalog_dir = tmp_path / ".openbench" / "references"
    catalog_dir.mkdir(parents=True)
    (catalog_dir / "reference_catalog.yaml").write_text(
        yaml.safe_dump(
            {
                "DemoRef": {
                    "name": "DemoRef",
                    "description": "demo",
                    "category": "Energy",
                    "data_type": "grid",
                    "tim_res": "Month",
                    "data_groupby": "Year",
                    "timezone": 0,
                    "years": [2001, 2002],
                    "root_dir": str(tmp_path / "ref"),
                    "variables": {
                        "Latent_Heat": {"varname": "Qle", "varunit": "W m-2"},
                    },
                }
            }
        )
    )

    refs = RegistryManager(user_dir=tmp_path / ".openbench").references_for_variable("latent_heat")

    assert "DemoRef" in [ref.name for ref in refs]


def test_config_duplicate_variable_keys_ignoring_case_error(tmp_path):
    config_path = tmp_path / "openbench.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {
                    "name": "case",
                    "output_dir": str(tmp_path / "out"),
                    "years": [2001, 2002],
                },
                "evaluation": {"variables": ["Latent_Heat"]},
                "reference": {
                    "Latent_Heat": "FluxRef",
                    "latent_heat": "OtherRef",
                },
                "simulation": {
                    "CaseA": {
                        "model": "DemoModel",
                        "root_dir": str(tmp_path / "sim"),
                    }
                },
            },
            sort_keys=False,
        )
    )

    with pytest.raises(ConfigError, match="duplicate variable keys ignoring case"):
        load_config(config_path)


def test_reference_dataset_names_use_casefold_for_non_ascii(tmp_path):
    catalog_dir = tmp_path / ".openbench" / "references"
    catalog_dir.mkdir(parents=True)
    (catalog_dir / "reference_catalog.yaml").write_text(
        yaml.safe_dump(
            {
                "Straße": {
                    "name": "Straße",
                    "description": "demo",
                    "category": "Other",
                    "data_type": "grid",
                    "tim_res": "Month",
                    "data_groupby": "Year",
                    "timezone": 0,
                    "years": [2001, 2002],
                    "root_dir": str(tmp_path / "ref"),
                    "variables": {
                        "Runoff": {"varname": "ro", "varunit": "mm day-1"},
                    },
                }
            }
        )
    )

    ref = RegistryManager(user_dir=tmp_path / ".openbench").get_reference("Strasse")

    assert ref is not None
    assert ref.name == "Straße"


def test_ambiguous_error_message_is_not_double_nested():
    with pytest.raises(AmbiguousNameError) as excinfo:
        resolve_name_case_insensitive("QLE", ["Qle", "qle"], label="--variable value")

    message = str(excinfo.value)
    assert message.count("ambiguous ignoring case") == 1
    assert "Qle, qle" in message


def test_duplicate_many_error_message_shows_raw_inputs_and_canonical_name():
    with pytest.raises(NameResolutionError) as excinfo:
        resolve_many_case_insensitive(
            ["Latent_Heat", "LATENT_HEAT"],
            ["Latent_Heat", "Runoff"],
            label="--variable",
        )

    message = str(excinfo.value)
    assert message == "duplicate --variable values ignoring case: Latent_Heat, LATENT_HEAT -> Latent_Heat"


def test_xarray_lookup_prefers_data_var_over_coord_case_match():
    ds = xr.Dataset(
        {"Lat": ("x", np.array([1.0]))},
        coords={"lat": ("x", np.array([2.0]))},
    )

    assert get_xarray_key_case_insensitive(ds, "LAT") == "Lat"


def test_station_matcher_missing_field_raises_openbench_error(tmp_path):
    ds = xr.Dataset({"station": ("x", np.array([1]))})

    with pytest.raises(DataProcessingError, match="lon"):
        _require_dataset_field(ds, "lon", "lon_var", tmp_path / "stations.nc")


def test_ref_scan_raises_on_ambiguous_casefold_root_dirs():
    class FakeChild:
        def __init__(self, name: str):
            self.name = name

        def is_dir(self):
            return True

        def __str__(self):
            return f"/fake/{self.name}"

    class FakeExact(FakeChild):
        def exists(self):
            return False

    class FakeParent:
        def __truediv__(self, name: str):
            return FakeExact(name)

        def iterdir(self):
            return [FakeChild("grid"), FakeChild("GRID")]

        def __str__(self):
            return "/fake"

    with pytest.raises(AmbiguousNameError, match="Grid"):
        _child_dir_case_insensitive(FakeParent(), "Grid")
