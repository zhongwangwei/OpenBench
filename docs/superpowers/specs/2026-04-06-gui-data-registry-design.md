# GUI Data Registry Management — Design Spec

**Date:** 2026-04-06
**Status:** Draft
**Scope:** Add a GUI page for managing registered models and reference datasets

## Context

Users currently register models and reference datasets via CLI commands or manual YAML editing. When a desired model or dataset is missing from the dropdown in the GUI, there's no way to add it without leaving the GUI.

This spec adds a "Data Registry" management page to the GUI with full CRUD operations, plus quick-access buttons from the Sim/Ref pages.

## Page Location

- New page in sidebar: **"Data Registry"** — positioned after General, before Simulation Data
- Page order becomes: General → **Data Registry** → Simulation Data → Reference Data → Evaluation → ...
- Quick-access: Sim page model dropdown gets a "⚙" button → jumps to Registry (Models tab). Ref page dataset dropdown gets a "⚙" button → jumps to Registry (Datasets tab).

## Page Layout

Two tabs: **Models** and **Reference Datasets**.

### Tab 1: Models

```
┌─ Models ─────────────────────────────────────────────────────┐
│                                                               │
│  Model List                              Actions              │
│  ┌──────────────────────────────────┐                        │
│  │ ● CoLM2024         (47 vars)    │   [+ New Model]        │
│  │   CLM5              (16 vars)    │   [Import from NC]     │
│  │   CaMa              (8 vars)     │   [Delete]             │
│  │   NoahMP5           (34 vars)    │                        │
│  │   ...                            │                        │
│  └──────────────────────────────────┘                        │
│                                                               │
│  ── Model Editor: CoLM2024 ──────────────────────────────── │
│                                                               │
│  Name: [CoLM2024          ]                                  │
│  Description: [Community Land Model 2024        ]            │
│  data_type: [grid ▼]  grid_res: [0.5]                       │
│                                                               │
│  Variables:                                                   │
│  ┌─────────────────────────┬────────────┬───────────┬──────┐ │
│  │ Variable                │ varname    │ varunit   │ comp │ │
│  ├─────────────────────────┼────────────┼───────────┼──────┤ │
│  │ Gross_Primary_Product...│ f_assim    │ mol m-2.. │      │ │
│  │ Evapotranspiration      │ f_fevpa    │ mm s-1    │      │ │
│  │ Latent_Heat             │ f_lfevpa   │ W m-2     │      │ │
│  │ Surface_Soil_Moisture   │            │ m3 m-3    │  ✓   │ │
│  │ ...                     │            │           │      │ │
│  └─────────────────────────┴────────────┴───────────┴──────┘ │
│                                                               │
│  [+ Add Variable]  [Remove Selected]  [Edit Variable...]     │
│                                                               │
│  [Save]  [Revert]                                            │
└───────────────────────────────────────────────────────────────┘
```

### Tab 2: Reference Datasets

```
┌─ Reference Datasets ─────────────────────────────────────────┐
│                                                               │
│  Dataset List                            Actions              │
│  ┌──────────────────────────────────┐                        │
│  │ ● ERA5LAND_LowRes   (grid, 4v)  │   [+ New Dataset]      │
│  │   GLEAM_v4.2a_LowRes (grid, 8v) │   [Scan Directory]     │
│  │   GRDC_Monthly       (stn, 1v)  │   [Delete]             │
│  │   ...                            │                        │
│  └──────────────────────────────────┘                        │
│                                                               │
│  ── Dataset Editor: ERA5LAND_LowRes ────────────────────── │
│                                                               │
│  Name: [ERA5LAND_LowRes   ]                                 │
│  Description: [ERA5-Land reanalysis (0.5 degree)   ]        │
│  data_type: [grid ▼]  tim_res: [Month ▼]  grid_res: [0.5]  │
│  root_dir: [/Volumes/work/Reference/Grid/LowRes   ] [Browse]│
│  data_groupby: [Year ▼]  timezone: [0]                      │
│                                                               │
│  Variables:                                                   │
│  ┌───────────────────────┬──────────┬─────────┬─────────────┐│
│  │ Variable              │ varname  │ varunit │ sub_dir      ││
│  ├───────────────────────┼──────────┼─────────┼─────────────┤│
│  │ Latent_Heat           │ slhf     │ J m-2   │ Heat/Late...││
│  │ Sensible_Heat         │ sshf     │ J m-2   │ Heat/Sens...││
│  │ ...                   │          │         │             ││
│  └───────────────────────┴──────────┴─────────┴─────────────┘│
│                                                               │
│  [+ Add Variable]  [Remove Selected]  [Edit Variable...]     │
│                                                               │
│  [Save]  [Revert]                                            │
└───────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Model Registration

**Manual:** Click "+ New Model" → empty editor form → fill name, add variables one by one.

**Import from NC:** Click "Import from NC" → file dialog → opens NC file → shows all variables/dims:
```
┌─ Import Variables from NC File ─────────────────────┐
│                                                       │
│  File: /path/to/Case01_hist_2004-01.nc               │
│  Dimensions: time(12), lat(360), lon(720), soil(10)  │
│                                                       │
│  ☑ f_assim        float32  (time,lat,lon)            │
│  ☑ f_fevpa        float32  (time,lat,lon)            │
│  ☑ f_lfevpa       float32  (time,lat,lon)            │
│  ☐ f_xy_prc       float32  (time,lat,lon)  ← forcing │
│  ☐ lat            float64  (lat)           ← coord   │
│  ☐ lon            float64  (lon)           ← coord   │
│  ...                                                  │
│                                                       │
│  Auto-detect: coordinates and forcing variables are   │
│  unchecked by default. Only model output is checked.  │
│                                                       │
│  [Select All]  [Deselect All]  [Import Selected]     │
└───────────────────────────────────────────────────────┘
```

After import, the variable table is populated with NC varnames. User then maps each to an OpenBench standard variable name (dropdown with all known variable names from the evaluation system).

### 2. Reference Dataset Registration

**Manual:** Click "+ New Dataset" → empty editor.

**Scan Directory:** Click "Scan Directory" → browse to data root → scans subdirectory structure to discover variables and file patterns. This reuses the existing `openbench data register --scan` logic.

### 3. Variable Editor Dialog

When editing a variable (click "Edit Variable..." or double-click a row):

```
┌─ Edit Variable Mapping ──────────────────────────────┐
│                                                       │
│  OpenBench Variable: [Latent_Heat            ▼]      │
│    (dropdown of all standard variable names)          │
│                                                       │
│  NC Variable Name: [f_lfevpa                ]        │
│  Unit: [W m-2                               ]        │
│                                                       │
│  ── For reference datasets only ──                    │
│  sub_dir: [Heat/Latent_Heat/ERA5LAND        ]        │
│  prefix:  [ERA5LAND_                        ]        │
│  suffix:  [_050_monthly                     ]        │
│                                                       │
│  ── Advanced (optional) ──                            │
│  Compute expression:                                  │
│  [                                                ]   │
│  (e.g., ds['var1'] + ds['var2'])                     │
│                                                       │
│  Fallbacks:                                           │
│  [+ Add Fallback]                                     │
│  1. varname: f_discharge  unit: m3 s-1               │
│                                                       │
│  [OK]  [Cancel]                                      │
└───────────────────────────────────────────────────────┘
```

### 4. Persistence

All changes write directly to:
- `src/openbench/data/registry/model_catalog.yaml`
- `src/openbench/data/registry/reference_catalog.yaml`

Via the existing `RegistryManager` API — need to add write methods:

```python
class RegistryManager:
    # Existing read methods...
    
    # New write methods:
    def save_model(self, name: str, profile: ModelProfile) -> None
    def delete_model(self, name: str) -> None
    def save_reference(self, name: str, dataset: ReferenceDataset) -> None
    def delete_reference(self, name: str) -> None
```

These serialize to YAML and write atomically (temp file + rename).

### 5. Quick Access from Sim/Ref Pages

**Simulation Data page:**
- Each case's model dropdown gets a small "⚙" (gear) button next to it
- Clicking it navigates to Data Registry → Models tab
- After registering, returning to Sim page refreshes the model dropdown

**Reference Data page:**
- Each variable's dataset dropdown gets a small "⚙" button
- Same pattern: navigates to Data Registry → Datasets tab

## Implementation Components

### New Files

| File | Purpose |
|------|---------|
| `gui/pages/page_registry.py` | Main registry management page (~600 lines) |
| `gui/widgets/variable_editor.py` | Variable mapping editor dialog (~200 lines) |
| `gui/widgets/nc_importer.py` | NC file import dialog (~250 lines) |

### Modified Files

| File | Change |
|------|--------|
| `data/registry/manager.py` | Add `save_model()`, `delete_model()`, `save_reference()`, `delete_reference()` |
| `data/registry/schema.py` | Add `to_dict()` serialization methods to `ModelProfile`, `ReferenceDataset` |
| `gui/controller.py` | Add "registry" to `ALL_PAGES` |
| `gui/main_window.py` | Import and register `PageRegistry` |
| `gui/pages/page_sim_data.py` | Add "⚙" button next to model combos |
| `gui/pages/page_ref_data.py` | Add "⚙" button next to dataset combos |
| `gui/pages/__init__.py` | Export `PageRegistry` |

### Standard Variable Names

The variable editor's "OpenBench Variable" dropdown is populated from the evaluation system's known variable list. Source: `evaluation_items` keys from the default config, or a dedicated constant list.

## NC Import Auto-Detection

When importing from NC file, automatically:
1. **Skip coordinates**: variables whose name matches `lat`, `lon`, `time`, `level`, etc.
2. **Skip 1D variables**: only include variables with at least 2 dimensions (time + space)
3. **Read units**: from `units` attribute in NC metadata
4. **Suggest OpenBench variable name**: fuzzy match NC varname against known variable names (e.g., `f_lfevpa` → suggest "Latent_Heat" if model profile has this mapping)

## Workflow Examples

### Example 1: Register a new model

1. User goes to Simulation Data, scans directory, but model dropdown doesn't have their model
2. Clicks "⚙" → jumps to Data Registry → Models tab
3. Clicks "Import from NC" → selects a simulation output file
4. NC variables are listed with checkboxes → user selects relevant output variables
5. For each selected variable, maps to OpenBench standard name via dropdown
6. Fills model name, clicks Save
7. Returns to Sim page → new model appears in dropdown

### Example 2: Register a new reference dataset

1. User goes to Reference Data page, but variable dropdown doesn't show the dataset they want
2. Clicks "⚙" → jumps to Data Registry → Datasets tab
3. Clicks "Scan Directory" → browses to data root
4. Scanner discovers variables from subdirectory structure and NC files
5. User reviews, adjusts settings, clicks Save
6. Returns to Ref page → new dataset appears in dropdown

## Validation

- Model name uniqueness enforced on save
- Dataset name uniqueness enforced on save
- Variable name must be from the standard OpenBench variable list (with option to add custom)
- NC file must be readable by xarray
- Warn before deleting a model/dataset that's in use by an active project
