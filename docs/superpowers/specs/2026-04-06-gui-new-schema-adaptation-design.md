# GUI New Schema Adaptation — Design Spec

**Date:** 2026-04-06
**Status:** Approved
**Scope:** Adapt GUI to output new unified YAML config format

## Context

The OpenBench config schema was restructured:
- `options` section merged into `project`
- `comparison.tim_res/grid_res/timezone/weight` moved to `project`
- `reference` changed from flat dict to `ReferenceConfig` with `data_root` + `sources`
- Reference datasets support auto-resolution (e.g., `GLEAM_v4.2a` → `GLEAM_v4.2a_LowRes`)
- Single file `openbench.yaml` replaces the old three-file namelist format

The GUI (~15,000 lines, PySide6) still uses the legacy dict-based config internally and outputs the old three-file format. This spec adapts the GUI to output the new format with minimal internal changes.

## Strategy

**Only change the output layer.** The GUI's internal dict data model, page structure, signal mechanism, and validation framework remain unchanged. Changes are confined to:

1. `ConfigManager` — new export function
2. `PageGeneral` — UI label text
3. `PageRefData` — dataset selector from registry
4. `PagePreview` — single file display

## 1. ConfigManager Export Layer

### What changes

Replace three export functions with one:

```
generate_main_nml()  ─┐
generate_ref_nml()   ─┼─→  generate_config_yaml()  →  openbench.yaml
generate_sim_nml()   ─┘
```

### Key mapping (internal dict → new YAML)

```
general["basename"]          → project.name
general["basedir"]           → project.output_dir
general["syear"]             → project.years[0]
general["eyear"]             → project.years[1]
general["compare_tim_res"]   → project.tim_res
general["compare_grid_res"]  → project.grid_res
general["compare_tzone"]     → project.timezone
general["weight"]            → project.weight
general["num_cores"]         → project.num_cores
general["unified_mask"]      → project.unified_mask
general["time_alignment"]    → project.time_alignment
general["generate_report"]   → project.generate_report
general["IGBP_groupby"]      → project.IGBP_groupby
general["PFT_groupby"]       → project.PFT_groupby
general["Climate_zone_groupby"] → project.climate_zone_groupby
general["min_lat/max_lat"]   → project.lat_range
general["min_lon/max_lon"]   → project.lon_range
general["min_year"]          → project.min_year_threshold

evaluation_items (true keys)  → evaluation.variables (list)

ref_data["general"]["data_root"]  → reference.data_root
ref_data per-variable source name → reference.{var}: {source_name}

sim_data entries               → simulation section with _defaults extraction
  general fields (model, data_type, grid_res, tim_res, data_groupby)
  per-case: root_dir, prefix, suffix, variables overrides

metrics (true keys)   → metrics (list)
scores (true keys)    → scores (list)

general["comparison"] + comparisons (true keys) → comparison.enabled + comparison.items
general["statistics"] + statistics (true keys)  → statistics.enabled + statistics.items
```

### Implementation

- Add `generate_config_yaml() -> str` method to `ConfigManager`
- Returns YAML string of the new single-file format
- `sync_namelists()` writes to `{basedir}/{basename}/openbench.yaml` instead of three files
- Keep old `generate_*_nml()` methods (not deleted, just unused by default)

### File: `config_manager.py`

New method ~100 lines. Touches: export path logic, `sync_namelists()`, `cleanup_unused_namelists()`.

## 2. PageGeneral UI Labels

### What changes

Text-only changes, no logic changes:

| Current label | New label |
|---------------|-----------|
| Compare Time Resolution | Time Resolution |
| Compare Grid Resolution | Grid Resolution |
| Compare Timezone | Timezone |

### File: `pages/page_general.py`

~3 string edits.

## 3. PageRefData Dataset Selector

### What changes

Replace manual field entry with registry-based dataset selector per variable.

**Current flow:**
```
User picks variable → manually fills root_dir, sub_dir, prefix, suffix, varname, varunit
```

**New flow:**
```
User picks variable → dropdown shows registry datasets for that variable
  → selection auto-fills all fields from catalog
  → "Advanced" expander allows manual override
```

### UI structure per variable

```
┌─ Latent_Heat ──────────────────────────────┐
│  Dataset: [ ERA5LAND              ▼ ]      │
│  (auto-resolved to ERA5LAND_LowRes)        │
│                                             │
│  ▶ Advanced (collapsed by default)          │
│    varname: slhf                            │
│    varunit: J m-2                           │
│    sub_dir: Heat/Latent_Heat/ERA5LAND       │
│    prefix: ERA5LAND_                        │
│    suffix: _050_monthly                     │
└─────────────────────────────────────────────┘
```

### Data source

```python
from openbench.data.registry.manager import get_registry
registry = get_registry()
# For a given variable, list all datasets that support it:
datasets = registry.references_for_variable("Latent_Heat")
# Returns list of ReferenceDataset objects with .name, .variables, etc.
```

### Top-level data_root

- Existing `data_root_input` text field stays at top of PageRefData
- Value written to `reference.data_root` in export

### Export mapping

When exporting, for each variable:
- If user selected a registry dataset → write just the base name (e.g., `GLEAM_v4.2a`)
- If user manually configured → write full name or inline overrides

### File: `pages/page_ref_data.py`

Major rework of the per-variable config section (~300 lines changed out of 949). The data scanning and registration UI at the top can remain.

## 4. PagePreview Single File

### What changes

- Remove three-tab layout (main / ref / sim)
- Single YAML preview of the complete `openbench.yaml`
- Export button writes one file instead of three
- "Copy to clipboard" button

### File: `pages/page_preview.py`

~100 lines simplified (remove tab management, simplify export).

## 5. Runner Integration

### What changes

The runner currently passes the three NML file paths to the backend. After this change:

- `EvaluationRunner` passes the single `openbench.yaml` path to `openbench run`
- `RemoteRunner` uploads one config file instead of three
- No changes to progress tracking or log parsing

### Files: `runner.py`, `remote_runner.py`

~10 lines each (path construction).

## 6. What Does NOT Change

- GUI internal dict data model (all pages read/write same dict keys)
- 11 page classes' structure, signals, and validation
- `WizardController` state management
- `DataValidator` NC file validation
- Theme and styling
- Widget library (DataSourceEditor, RemoteConfigWidget, etc.)
- `WizardConfigManager` (.wizard.yaml for runtime settings)

## File Change Summary

| File | Change | Effort |
|------|--------|--------|
| `config_manager.py` | New `generate_config_yaml()`, update `sync_namelists()` | Medium |
| `pages/page_general.py` | 3 label string changes | Trivial |
| `pages/page_ref_data.py` | Dataset selector dropdown per variable | Medium |
| `pages/page_preview.py` | Single tab, single file export | Small |
| `runner.py` | Config file path change | Trivial |
| `remote_runner.py` | Upload path change | Trivial |

Total estimated: ~500 lines changed across 6 files.

## Validation

After implementation:
1. GUI generates valid `openbench.yaml` that passes `openbench check`
2. PageRefData dropdown correctly lists registry datasets per variable
3. Preview shows correct single-file YAML
4. Local and remote run work with the new config path
5. Existing projects can still be loaded (old format → internal dict → new export)
