# Sub-project 3: Data Registry + Model Profiles

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the data registry system that resolves reference dataset names (e.g., `GLEAM_v4.2a`) to their variable mappings and data paths, and model profile system that resolves model names (e.g., `CoLM2024`) to their variable mappings. Wire up `openbench data` and `openbench model` CLI commands.

**Architecture:** Registry is a Python class (`RegistryManager`) that loads YAML descriptor files from two locations: built-in (shipped with package) and user-defined (`~/.openbench/`). Reference descriptors are converted from the existing 44 grid + 28 station variable definition files in OpenBench-wei. Model profiles are created for known LSMs. A batch conversion script converts the old definition files to new format.

**Tech Stack:** Python dataclasses, PyYAML, platformdirs, click, pytest

**Spec:** `docs/superpowers/specs/2026-03-30-openbench-unification-design.md` (Section 6)

**Working Directory:** `/Volumes/Data01/Openbench`

---

## File Structure

| Action | Path | Responsibility |
|---|---|---|
| Create | `src/openbench/data/registry/manager.py` | RegistryManager class |
| Create | `src/openbench/data/registry/schema.py` | Dataclasses for reference/model descriptors |
| Create | `src/openbench/data/registry/converter.py` | Batch convert old definition files to new format |
| Create | `src/openbench/data/registry/references/` | 72 reference descriptor YAML files |
| Create | `src/openbench/data/registry/models/` | Model profile YAML files |
| Modify | `src/openbench/data/registry/__init__.py` | Public API exports |
| Modify | `src/openbench/cli/data.py` | Wire `openbench data list/status/path` |
| Modify | `src/openbench/cli/model.py` | Wire `openbench model list/show` |
| Create | `tests/test_registry/__init__.py` | Test package |
| Create | `tests/test_registry/test_manager.py` | RegistryManager tests |
| Create | `tests/test_registry/test_converter.py` | Converter tests |

---

### Task 1: Define Registry Schema Dataclasses

**Files:**
- Create: `src/openbench/data/registry/schema.py`
- Create: `tests/test_registry/__init__.py`
- Create: `tests/test_registry/test_schema.py`

- [ ] **Step 1: Write `tests/test_registry/__init__.py`**

```python
"""Registry test package."""
```

- [ ] **Step 2: Write `tests/test_registry/test_schema.py`**

```python
"""Tests for registry schema dataclasses."""

from openbench.data.registry.schema import ModelProfile, ReferenceDataset, VariableMapping


def test_variable_mapping():
    v = VariableMapping(varname="E", varunit="mm day-1", prefix="E_", suffix="_GLEAM")
    assert v.varname == "E"
    assert v.sub_dir is None


def test_variable_mapping_with_subdir():
    v = VariableMapping(
        varname="E",
        varunit="mm day-1",
        prefix="E_",
        suffix="_GLEAM",
        sub_dir="Evapotranspiration/GLEAM_v4.2a",
    )
    assert v.sub_dir == "Evapotranspiration/GLEAM_v4.2a"


def test_reference_dataset():
    ref = ReferenceDataset(
        name="GLEAM_v4.2a",
        description="Global Land Evaporation Amsterdam Model v4.2a",
        category="Water",
        data_type="grid",
        grid_res=0.25,
        tim_res="Month",
        data_groupby="Year",
        timezone=0,
        years=[1980, 2023],
        variables={
            "Evapotranspiration": VariableMapping(
                varname="E", varunit="mm day-1", prefix="E_", suffix="_GLEAM_v4.2a",
                sub_dir="Evapotranspiration/GLEAM_v4.2a",
            ),
        },
    )
    assert ref.name == "GLEAM_v4.2a"
    assert ref.data_type == "grid"
    assert "Evapotranspiration" in ref.variables
    assert ref.fulllist is None


def test_reference_dataset_station():
    ref = ReferenceDataset(
        name="GRDC_Monthly",
        description="GRDC Monthly Streamflow",
        category="Water",
        data_type="stn",
        tim_res="Month",
        data_groupby="single",
        timezone=0,
        years=[1950, 2023],
        variables={
            "Streamflow": VariableMapping(varname="streamflow", varunit="m3 s-1"),
        },
        fulllist="GRDC_Monthly.csv",
    )
    assert ref.data_type == "stn"
    assert ref.fulllist == "GRDC_Monthly.csv"
    assert ref.grid_res is None


def test_model_profile():
    m = ModelProfile(
        name="CoLM2024",
        description="Common Land Model 2024",
        data_type="grid",
        grid_res=0.5,
        tim_res="Month",
        variables={
            "Evapotranspiration": VariableMapping(varname="ET", varunit="mm day-1"),
            "Latent_Heat": VariableMapping(varname="Qle", varunit="W m-2"),
        },
    )
    assert m.name == "CoLM2024"
    assert len(m.variables) == 2
    assert m.variables["Evapotranspiration"].varname == "ET"
```

- [ ] **Step 3: Write `src/openbench/data/registry/schema.py`**

```python
"""Dataclasses for registry descriptors (reference datasets and model profiles)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VariableMapping:
    """How a variable is stored in a dataset or model output."""

    varname: str
    varunit: str
    prefix: str = ""
    suffix: str = ""
    sub_dir: Optional[str] = None
    fulllist: Optional[str] = None
    max_uparea: Optional[float] = None
    min_uparea: Optional[float] = None


@dataclass
class ReferenceDataset:
    """Descriptor for a reference dataset in the registry."""

    name: str
    description: str
    category: str  # Water, Carbon, Energy, Meteorology, Crop, Urban, Lake
    data_type: str  # grid, stn
    tim_res: str
    data_groupby: str
    timezone: int | float
    years: list[int]  # [start, end]
    variables: dict[str, VariableMapping]
    grid_res: Optional[float] = None
    fulllist: Optional[str] = None
    root_dir: Optional[str] = None  # Set when data is downloaded/located


@dataclass
class ModelProfile:
    """Descriptor for a simulation model's variable mappings."""

    name: str
    description: str
    data_type: str = "grid"
    grid_res: Optional[float] = None
    tim_res: str = "Month"
    variables: dict[str, VariableMapping] = field(default_factory=dict)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_registry/ -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/openbench/data/registry/schema.py tests/test_registry/
git commit -m "feat(registry): add schema dataclasses for reference datasets and model profiles"
```

---

### Task 2: Implement RegistryManager

**Files:**
- Create: `src/openbench/data/registry/manager.py`
- Create: `tests/test_registry/test_manager.py`
- Create: `src/openbench/data/registry/references/GLEAM_v4.2a.yaml` (test fixture)
- Create: `src/openbench/data/registry/models/CoLM2024.yaml` (test fixture)

- [ ] **Step 1: Create a minimal reference descriptor for testing**

`src/openbench/data/registry/references/GLEAM_v4.2a.yaml`:
```yaml
name: GLEAM_v4.2a
description: "Global Land Evaporation Amsterdam Model v4.2a"
category: Water
data_type: grid
grid_res: 0.25
tim_res: Month
data_groupby: Year
timezone: 0
years: [1980, 2023]
variables:
  Evapotranspiration:
    varname: E
    varunit: "mm day-1"
    prefix: "E_"
    suffix: "_GLEAM_v4.2a"
    sub_dir: "Evapotranspiration/GLEAM_v4.2a"
  Canopy_Evaporation:
    varname: Ei
    varunit: "mm day-1"
    prefix: "Ei_"
    suffix: "_GLEAM_v4.2a"
    sub_dir: "Canopy_Evaporation/GLEAM_v4.2a"
  Canopy_Transpiration:
    varname: Et
    varunit: "mm day-1"
    prefix: "Et_"
    suffix: "_GLEAM_v4.2a"
    sub_dir: "Canopy_Transpiration/GLEAM_v4.2a"
  Bare_Soil_Evaporation:
    varname: Eb
    varunit: "mm day-1"
    prefix: "Eb_"
    suffix: "_GLEAM_v4.2a"
    sub_dir: "Bare_Soil_Evaporation/GLEAM_v4.2a"
  Root_Zone_Soil_Moisture:
    varname: SMroot
    varunit: "m3 m-3"
    prefix: "SMroot_"
    suffix: "_GLEAM_v4.2a"
    sub_dir: "Root_Zone_Soil_Moisture/GLEAM_v4.2a"
  Surface_Soil_Moisture:
    varname: SMsurf
    varunit: "m3 m-3"
    prefix: "SMsurf_"
    suffix: "_GLEAM_v4.2a"
    sub_dir: "Surface_Soil_Moisture/GLEAM_v4.2a"
  Sensible_Heat:
    varname: H
    varunit: "W m-2"
    prefix: "H_"
    suffix: "_GLEAM_v4.2a"
    sub_dir: "Sensible_Heat/GLEAM_v4.2a"
```

- [ ] **Step 2: Create a minimal model profile for testing**

`src/openbench/data/registry/models/CoLM2024.yaml`:
```yaml
name: CoLM2024
description: "Common Land Model 2024"
data_type: grid
grid_res: 0.5
tim_res: Month
variables:
  Evapotranspiration:
    varname: f_fevpa
    varunit: "mm day-1"
  Latent_Heat:
    varname: f_lfevpa
    varunit: "W m-2"
  Sensible_Heat:
    varname: f_fsena
    varunit: "W m-2"
  Gross_Primary_Productivity:
    varname: f_assim
    varunit: "mol m-2 s-1"
  Net_Radiation:
    varname: f_srnet
    varunit: "W m-2"
  Ground_Heat:
    varname: f_grnd_flux
    varunit: "W m-2"
  Surface_Soil_Temperature:
    varname: f_t_grnd
    varunit: "K"
  Surface_Soil_Moisture:
    varname: f_h2osoi
    varunit: "m3 m-3"
  Total_Runoff:
    varname: f_rnof
    varunit: "mm day-1"
  Surface_Albedo:
    varname: f_alb
    varunit: "-"
```

- [ ] **Step 3: Write `tests/test_registry/test_manager.py`**

```python
"""Tests for RegistryManager."""

from openbench.data.registry.manager import RegistryManager


def test_list_references():
    mgr = RegistryManager()
    refs = mgr.list_references()
    assert isinstance(refs, list)
    assert len(refs) >= 1
    # GLEAM should be in the built-in list
    names = [r.name for r in refs]
    assert "GLEAM_v4.2a" in names


def test_get_reference():
    mgr = RegistryManager()
    ref = mgr.get_reference("GLEAM_v4.2a")
    assert ref is not None
    assert ref.name == "GLEAM_v4.2a"
    assert ref.data_type == "grid"
    assert ref.grid_res == 0.25
    assert "Evapotranspiration" in ref.variables
    assert ref.variables["Evapotranspiration"].varname == "E"


def test_get_reference_not_found():
    mgr = RegistryManager()
    ref = mgr.get_reference("NonExistentDataset")
    assert ref is None


def test_list_models():
    mgr = RegistryManager()
    models = mgr.list_models()
    assert isinstance(models, list)
    assert len(models) >= 1
    names = [m.name for m in models]
    assert "CoLM2024" in names


def test_get_model():
    mgr = RegistryManager()
    model = mgr.get_model("CoLM2024")
    assert model is not None
    assert model.name == "CoLM2024"
    assert "Evapotranspiration" in model.variables
    assert model.variables["Evapotranspiration"].varname == "f_fevpa"


def test_get_model_not_found():
    mgr = RegistryManager()
    model = mgr.get_model("NonExistentModel")
    assert model is None


def test_references_for_variable():
    mgr = RegistryManager()
    refs = mgr.references_for_variable("Evapotranspiration")
    assert len(refs) >= 1
    assert any(r.name == "GLEAM_v4.2a" for r in refs)


def test_references_for_unknown_variable():
    mgr = RegistryManager()
    refs = mgr.references_for_variable("UnknownVariable")
    assert refs == []
```

- [ ] **Step 4: Write `src/openbench/data/registry/manager.py`**

```python
"""RegistryManager: loads and queries reference datasets and model profiles.

Loads YAML descriptors from two locations:
1. Built-in: src/openbench/data/registry/references/ and models/
2. User-defined: ~/.openbench/references/ and ~/.openbench/models/
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from openbench.data.registry.schema import ModelProfile, ReferenceDataset, VariableMapping


class RegistryManager:
    """Manages reference dataset and model profile descriptors."""

    def __init__(self, user_dir: Optional[Path] = None):
        """Initialize registry, loading all descriptors.

        Args:
            user_dir: Override for user config directory. Defaults to ~/.openbench/.
        """
        self._references: dict[str, ReferenceDataset] = {}
        self._models: dict[str, ModelProfile] = {}

        # Built-in descriptors (shipped with package)
        builtin_dir = Path(__file__).parent
        self._load_references(builtin_dir / "references")
        self._load_models(builtin_dir / "models")

        # User-defined descriptors (override built-in)
        if user_dir is None:
            try:
                from platformdirs import user_config_dir

                user_dir = Path(user_config_dir("openbench"))
            except ImportError:
                user_dir = Path.home() / ".openbench"

        if user_dir.exists():
            self._load_references(user_dir / "references")
            self._load_models(user_dir / "models")

    def _load_references(self, directory: Path) -> None:
        """Load all reference descriptors from a directory."""
        if not directory.exists():
            return
        for path in sorted(directory.glob("*.yaml")):
            try:
                ref = _load_reference_yaml(path)
                self._references[ref.name] = ref
            except Exception:
                pass  # Skip malformed files silently

    def _load_models(self, directory: Path) -> None:
        """Load all model profiles from a directory."""
        if not directory.exists():
            return
        for path in sorted(directory.glob("*.yaml")):
            try:
                model = _load_model_yaml(path)
                self._models[model.name] = model
            except Exception:
                pass  # Skip malformed files silently

    def list_references(self) -> list[ReferenceDataset]:
        """List all available reference datasets."""
        return sorted(self._references.values(), key=lambda r: r.name)

    def get_reference(self, name: str) -> Optional[ReferenceDataset]:
        """Get a reference dataset by name."""
        return self._references.get(name)

    def list_models(self) -> list[ModelProfile]:
        """List all available model profiles."""
        return sorted(self._models.values(), key=lambda m: m.name)

    def get_model(self, name: str) -> Optional[ModelProfile]:
        """Get a model profile by name."""
        return self._models.get(name)

    def references_for_variable(self, variable: str) -> list[ReferenceDataset]:
        """Find all reference datasets that provide a given variable."""
        return [
            ref
            for ref in self._references.values()
            if variable in ref.variables
        ]


def _load_reference_yaml(path: Path) -> ReferenceDataset:
    """Load a reference dataset descriptor from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    variables = {}
    for var_name, var_data in data.get("variables", {}).items():
        variables[var_name] = VariableMapping(
            varname=var_data["varname"],
            varunit=var_data.get("varunit", ""),
            prefix=var_data.get("prefix", ""),
            suffix=var_data.get("suffix", ""),
            sub_dir=var_data.get("sub_dir"),
            fulllist=var_data.get("fulllist"),
            max_uparea=var_data.get("max_uparea"),
            min_uparea=var_data.get("min_uparea"),
        )

    return ReferenceDataset(
        name=data["name"],
        description=data.get("description", ""),
        category=data.get("category", ""),
        data_type=data["data_type"],
        tim_res=data["tim_res"],
        data_groupby=data.get("data_groupby", "Year"),
        timezone=data.get("timezone", 0),
        years=data.get("years", []),
        variables=variables,
        grid_res=data.get("grid_res"),
        fulllist=data.get("fulllist"),
        root_dir=data.get("root_dir"),
    )


def _load_model_yaml(path: Path) -> ModelProfile:
    """Load a model profile from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    variables = {}
    for var_name, var_data in data.get("variables", {}).items():
        variables[var_name] = VariableMapping(
            varname=var_data["varname"],
            varunit=var_data.get("varunit", ""),
            prefix=var_data.get("prefix", ""),
            suffix=var_data.get("suffix", ""),
            sub_dir=var_data.get("sub_dir"),
        )

    return ModelProfile(
        name=data["name"],
        description=data.get("description", ""),
        data_type=data.get("data_type", "grid"),
        grid_res=data.get("grid_res"),
        tim_res=data.get("tim_res", "Month"),
        variables=variables,
    )
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_registry/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/openbench/data/registry/ tests/test_registry/
git commit -m "feat(registry): add RegistryManager with reference and model profile loading"
```

---

### Task 3: Batch Convert Old Reference Definitions

**Files:**
- Create: `src/openbench/data/registry/converter.py`
- Create: `tests/test_registry/test_converter.py`

This task converts all 44 grid reference definition files from the old format into new registry descriptor YAML files.

- [ ] **Step 1: Write `tests/test_registry/test_converter.py`**

```python
"""Tests for reference definition converter."""

import tempfile
from pathlib import Path

from openbench.data.registry.converter import convert_old_reference


def test_convert_gleam():
    """Convert old GLEAM definition to new format."""
    old_path = Path("OpenBench-wei/nml/nml-yaml/Ref_variables_definition_LowRes/GLEAM_v4.2a.yaml")
    if not old_path.exists():
        return  # Skip if old code not present

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "GLEAM_v4.2a.yaml"
        convert_old_reference(old_path, out_path, name="GLEAM_v4.2a", category="Water")

        assert out_path.exists()

        import yaml

        with open(out_path) as f:
            data = yaml.safe_load(f)

        assert data["name"] == "GLEAM_v4.2a"
        assert data["data_type"] == "grid"
        assert "variables" in data
        assert "Evapotranspiration" in data["variables"]
        assert data["variables"]["Evapotranspiration"]["varname"] == "E"
```

- [ ] **Step 2: Write `src/openbench/data/registry/converter.py`**

```python
"""Convert old-format reference variable definition files to new registry format.

Old format: separate YAML files with 'general' section + per-variable sections.
New format: unified registry descriptor with 'variables' dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml


def convert_old_reference(
    old_path: Path,
    output_path: Path,
    name: str,
    category: str = "",
    description: str = "",
) -> None:
    """Convert a single old-format reference definition to new registry format.

    Args:
        old_path: Path to the old YAML definition file.
        output_path: Where to write the new descriptor.
        name: Dataset name for the registry.
        category: Dataset category (Water, Carbon, Energy, etc.).
        description: Human-readable description.
    """
    with open(old_path) as f:
        old = yaml.safe_load(f)

    general = old.get("general", {})

    # Extract variables (everything except 'general')
    variables: dict[str, Any] = {}
    for key, value in old.items():
        if key != "general" and isinstance(value, dict):
            var_entry: dict[str, Any] = {"varname": value.get("varname", "")}
            if value.get("varunit"):
                var_entry["varunit"] = value["varunit"]
            if value.get("prefix"):
                var_entry["prefix"] = value["prefix"]
            if value.get("suffix"):
                var_entry["suffix"] = value["suffix"]
            if value.get("sub_dir"):
                var_entry["sub_dir"] = value["sub_dir"]
            if value.get("fulllist"):
                var_entry["fulllist"] = value["fulllist"]
            if value.get("max_uparea"):
                var_entry["max_uparea"] = value["max_uparea"]
            if value.get("min_uparea"):
                var_entry["min_uparea"] = value["min_uparea"]
            variables[key] = var_entry

    new_descriptor: dict[str, Any] = {
        "name": name,
        "description": description or f"{name} reference dataset",
        "category": category,
        "data_type": general.get("data_type", "grid"),
        "tim_res": general.get("tim_res", "Month"),
        "data_groupby": general.get("data_groupby", "Year"),
        "timezone": general.get("timezone", 0),
        "years": [general.get("syear", 2000), general.get("eyear", 2020)],
        "variables": variables,
    }

    if general.get("grid_res"):
        new_descriptor["grid_res"] = general["grid_res"]
    if general.get("fulllist"):
        new_descriptor["fulllist"] = general["fulllist"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(new_descriptor, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def batch_convert_references(
    source_dir: Path,
    output_dir: Path,
    category_map: Optional[dict[str, str]] = None,
) -> list[str]:
    """Convert all old-format reference definitions in a directory.

    Args:
        source_dir: Directory containing old YAML definition files.
        output_dir: Where to write new descriptors.
        category_map: Optional mapping of dataset name to category.

    Returns:
        List of converted dataset names.
    """
    if category_map is None:
        category_map = {}

    converted = []
    for old_file in sorted(source_dir.glob("*.yaml")):
        name = old_file.stem
        category = category_map.get(name, _guess_category(old_file))
        output_path = output_dir / f"{name}.yaml"
        try:
            convert_old_reference(old_file, output_path, name=name, category=category)
            converted.append(name)
        except Exception as e:
            print(f"Warning: Failed to convert {old_file.name}: {e}")

    return converted


def _guess_category(path: Path) -> str:
    """Guess dataset category from its variable names."""
    with open(path) as f:
        data = yaml.safe_load(f)

    variables = [k for k in data.keys() if k != "general"]

    water_vars = {
        "Evapotranspiration", "Runoff", "Total_Runoff", "Streamflow",
        "Soil_Moisture", "Surface_Soil_Moisture", "Root_Zone_Soil_Moisture",
        "Snow_Depth", "Snow_Water_Equivalent", "Precipitation",
        "Canopy_Evaporation", "Canopy_Transpiration", "Bare_Soil_Evaporation",
        "Terrestrial_Water_Storage_Change", "Water_Table_Depth",
        "Groundwater_Recharge_Rate", "Inundation_Fraction", "Inundation_Area",
        "Depth_Of_Surface_Water",
    }
    carbon_vars = {
        "Gross_Primary_Productivity", "Net_Ecosystem_Exchange",
        "Ecosystem_Respiration", "Biomass", "Leaf_Area_Index",
        "Soil_Carbon", "Net_Primary_Production", "Veg_Cover_In_Fraction",
        "Burned_Area", "Methane", "Wetland_Methane_Emission",
    }
    energy_vars = {
        "Latent_Heat", "Sensible_Heat", "Net_Radiation", "Ground_Heat",
        "Surface_Upward_SW_Radiation", "Surface_Upward_LW_Radiation",
        "Surface_Net_SW_Radiation", "Surface_Net_LW_Radiation",
        "Surface_Downward_SW_Radiation", "Surface_Downward_LW_Radiation",
        "Surface_Albedo",
    }

    for v in variables:
        if v in water_vars:
            return "Water"
        if v in carbon_vars:
            return "Carbon"
        if v in energy_vars:
            return "Energy"

    return "Other"
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_registry/test_converter.py -v`
Expected: PASS

- [ ] **Step 4: Run the batch converter to generate all reference descriptors**

```bash
cd /Volumes/Data01/Openbench
python -c "
from pathlib import Path
from openbench.data.registry.converter import batch_convert_references

# Grid references
grid_dir = Path('OpenBench-wei/nml/nml-yaml/Ref_variables_definition_LowRes')
out_dir = Path('src/openbench/data/registry/references')
if grid_dir.exists():
    converted = batch_convert_references(grid_dir, out_dir)
    print(f'Converted {len(converted)} grid reference descriptors')

# Station references
stn_dir = Path('OpenBench-wei/nml/nml-yaml/Ref_variables_definition_stn')
if stn_dir.exists():
    converted = batch_convert_references(stn_dir, out_dir)
    print(f'Converted {len(converted)} station reference descriptors')
"
```

- [ ] **Step 5: Verify converted files**

Run:
```bash
ls src/openbench/data/registry/references/ | wc -l
python -c "
from openbench.data.registry.manager import RegistryManager
mgr = RegistryManager()
refs = mgr.list_references()
print(f'{len(refs)} references loaded')
for r in refs[:5]:
    print(f'  {r.name}: {r.category}, {r.data_type}, {len(r.variables)} vars')
"
```

- [ ] **Step 6: Commit**

```bash
git add src/openbench/data/registry/converter.py src/openbench/data/registry/references/ tests/test_registry/test_converter.py
git commit -m "feat(registry): batch convert 70+ reference definitions to new format"
```

---

### Task 4: Create Model Profiles

**Files:**
- Create: `src/openbench/data/registry/models/CLM5.yaml`
- Create: `src/openbench/data/registry/models/ERA5-Land.yaml`

- [ ] **Step 1: Create CLM5 model profile**

`src/openbench/data/registry/models/CLM5.yaml`:
```yaml
name: CLM5
description: "Community Land Model version 5"
data_type: grid
grid_res: 1.25
tim_res: Month
variables:
  Evapotranspiration:
    varname: QFLX_EVAP_TOT
    varunit: "mm day-1"
  Latent_Heat:
    varname: EFLX_LH_TOT
    varunit: "W m-2"
  Sensible_Heat:
    varname: FSH
    varunit: "W m-2"
  Gross_Primary_Productivity:
    varname: GPP
    varunit: "gC m-2 s-1"
  Net_Ecosystem_Exchange:
    varname: NEE
    varunit: "gC m-2 s-1"
  Net_Radiation:
    varname: FSA
    varunit: "W m-2"
  Total_Runoff:
    varname: QRUNOFF
    varunit: "mm day-1"
  Surface_Soil_Moisture:
    varname: SOILWATER_10CM
    varunit: "kg m-2"
  Surface_Soil_Temperature:
    varname: TSOI_10CM
    varunit: "K"
  Snow_Water_Equivalent:
    varname: H2OSNO
    varunit: "mm"
  Leaf_Area_Index:
    varname: TLAI
    varunit: "m2 m-2"
  Surface_Albedo:
    varname: ASA
    varunit: "-"
```

- [ ] **Step 2: Create ERA5-Land model profile**

`src/openbench/data/registry/models/ERA5-Land.yaml`:
```yaml
name: ERA5-Land
description: "ECMWF ERA5-Land Reanalysis"
data_type: grid
grid_res: 0.1
tim_res: Month
variables:
  Evapotranspiration:
    varname: e
    varunit: "m"
  Latent_Heat:
    varname: slhf
    varunit: "J m-2"
  Sensible_Heat:
    varname: sshf
    varunit: "J m-2"
  Surface_Soil_Temperature:
    varname: stl1
    varunit: "K"
  Surface_Soil_Moisture:
    varname: swvl1
    varunit: "m3 m-3"
  Snow_Water_Equivalent:
    varname: sd
    varunit: "m of water equivalent"
  Snow_Depth:
    varname: sde
    varunit: "m"
  Total_Runoff:
    varname: ro
    varunit: "m"
  Surface_Net_SW_Radiation:
    varname: ssr
    varunit: "J m-2"
  Surface_Net_LW_Radiation:
    varname: str
    varunit: "J m-2"
  Precipitation:
    varname: tp
    varunit: "m"
  Surface_Air_Temperature:
    varname: t2m
    varunit: "K"
```

- [ ] **Step 3: Verify models load**

Run:
```bash
python -c "
from openbench.data.registry.manager import RegistryManager
mgr = RegistryManager()
for m in mgr.list_models():
    print(f'{m.name}: {len(m.variables)} variables')
"
```

Expected:
```
CLM5: 12 variables
CoLM2024: 10 variables
ERA5-Land: 12 variables
```

- [ ] **Step 4: Commit**

```bash
git add src/openbench/data/registry/models/
git commit -m "feat(registry): add CLM5 and ERA5-Land model profiles"
```

---

### Task 5: Wire CLI Commands

**Files:**
- Modify: `src/openbench/data/registry/__init__.py`
- Modify: `src/openbench/cli/data.py`
- Modify: `src/openbench/cli/model.py`

- [ ] **Step 1: Update `src/openbench/data/registry/__init__.py`**

```python
"""Dataset registry and model profiles.

Public API:
    RegistryManager - loads and queries reference datasets and model profiles
    ReferenceDataset, ModelProfile, VariableMapping - schema dataclasses
"""

from openbench.data.registry.manager import RegistryManager
from openbench.data.registry.schema import ModelProfile, ReferenceDataset, VariableMapping

__all__ = [
    "RegistryManager",
    "ReferenceDataset",
    "ModelProfile",
    "VariableMapping",
]
```

- [ ] **Step 2: Update `src/openbench/cli/data.py`**

```python
"""openbench data commands."""

import click


@click.group()
def data():
    """Manage reference datasets."""


@data.command("list")
@click.option("--variable", default=None, help="Filter by variable name.")
def list_datasets(variable):
    """List all available reference datasets."""
    from openbench.data.registry import RegistryManager

    mgr = RegistryManager()

    if variable:
        refs = mgr.references_for_variable(variable)
        if not refs:
            click.echo(f"No datasets found for variable: {variable}")
            return
    else:
        refs = mgr.list_references()

    click.secho(f"{'Name':<30} {'Category':<12} {'Type':<6} {'Res':<8} {'Years':<14} {'Variables'}", bold=True)
    click.echo("─" * 100)
    for r in refs:
        res = f"{r.grid_res}°" if r.grid_res else "stn"
        years = f"{r.years[0]}-{r.years[1]}" if r.years else "?"
        nvars = len(r.variables)
        click.echo(f"{r.name:<30} {r.category:<12} {r.data_type:<6} {res:<8} {years:<14} {nvars}")

    click.echo(f"\nTotal: {len(refs)} datasets")


@data.command()
@click.argument("names", nargs=-1, required=True)
def download(names):
    """Download reference datasets by name."""
    click.echo("Dataset download not yet implemented (requires hosted data repository).")
    click.echo(f"Requested: {', '.join(names)}")


@data.command()
def status():
    """Show local dataset cache status."""
    from openbench.data.registry import RegistryManager

    mgr = RegistryManager()
    refs = mgr.list_references()
    click.echo(f"Registry: {len(refs)} datasets available")
    click.echo("Local cache: not yet implemented")


@data.command()
@click.argument("name")
def path(name):
    """Print local path for a dataset."""
    from openbench.data.registry import RegistryManager

    mgr = RegistryManager()
    ref = mgr.get_reference(name)
    if ref is None:
        click.secho(f"Dataset not found: {name}", fg="red")
        raise SystemExit(1)
    if ref.root_dir:
        click.echo(ref.root_dir)
    else:
        click.echo(f"No local path configured for {name}")


@data.command()
@click.argument("name")
def optimize(name):
    """Convert dataset to zarr for faster reads."""
    click.echo(f"Not yet implemented. Dataset: {name}")
```

- [ ] **Step 3: Update `src/openbench/cli/model.py`**

```python
"""openbench model commands."""

import click


@click.group()
def model():
    """Manage model profiles."""


@model.command("list")
def list_models():
    """List all available model profiles."""
    from openbench.data.registry import RegistryManager

    mgr = RegistryManager()
    models = mgr.list_models()

    click.secho(f"{'Name':<20} {'Type':<6} {'Res':<8} {'Variables'}", bold=True)
    click.echo("─" * 60)
    for m in models:
        res = f"{m.grid_res}°" if m.grid_res else "?"
        click.echo(f"{m.name:<20} {m.data_type:<6} {res:<8} {len(m.variables)}")

    click.echo(f"\nTotal: {len(models)} model profiles")


@model.command()
@click.argument("name")
def show(name):
    """Show variable mappings for a model profile."""
    from openbench.data.registry import RegistryManager

    mgr = RegistryManager()
    m = mgr.get_model(name)
    if m is None:
        click.secho(f"Model profile not found: {name}", fg="red")
        raise SystemExit(1)

    click.secho(f"{m.name}", bold=True)
    click.echo(f"Description: {m.description}")
    click.echo(f"Type: {m.data_type}, Resolution: {m.grid_res}°, Time: {m.tim_res}")
    click.echo()
    click.secho(f"{'Variable':<35} {'File varname':<20} {'Unit'}", bold=True)
    click.echo("─" * 70)
    for var_name, mapping in sorted(m.variables.items()):
        click.echo(f"{var_name:<35} {mapping.varname:<20} {mapping.varunit}")


@model.command()
def create():
    """Interactively create a new model profile."""
    click.echo("Not yet implemented.")
```

- [ ] **Step 4: Test CLI commands**

Run:
```bash
openbench data list
openbench data list --variable Evapotranspiration
openbench model list
openbench model show CoLM2024
```

- [ ] **Step 5: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/openbench/data/registry/__init__.py src/openbench/cli/data.py src/openbench/cli/model.py
git commit -m "feat(cli): wire openbench data and model commands to registry"
```

---

### Task 6: Final Verification and Lint

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`

- [ ] **Step 2: Run lint**

Run:
```bash
ruff check src/ tests/
ruff format src/ tests/
ruff format --check src/ tests/
```

- [ ] **Step 3: End-to-end test**

Run:
```bash
openbench data list | head -20
openbench model list
openbench model show CLM5
openbench data list --variable Latent_Heat
```

- [ ] **Step 4: Commit any fixes**

```bash
git add -A && git commit -m "fix: lint and format cleanup for registry system"
```

- [ ] **Step 5: Tag milestone**

```bash
git tag -a v3.0.0a3 -m "Sub-project 3 complete: data registry and model profiles"
```
