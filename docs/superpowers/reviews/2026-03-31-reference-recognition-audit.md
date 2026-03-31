# Reference Recognition Audit

## Task 1 Baseline

### Classification Baseline

Problems:
- wrong recognition
- missed recognition
- wrong variant resolution
- wrong variable recognition
- wrong time-resolution recognition
- inconsistent source semantics that can produce incorrect results

Improvement items:
- manual confirmation gaps
- weak defaults
- missing tests
- maintainability or performance issues that do not directly change results

## Baseline Coverage

- Covered: exact reference lookup, auto-resolve with simulation context, resolution-variant listing, and variable-based reference listing.
- Partially covered: no-context base-name handling and not-found behavior, but only at the `get_reference()` boundary.
- Not covered: directory scanning, descriptor registration, time-resolution inference, station-list generation, or scan/import behavior.
- Not covered: merge semantics for persisted registry descriptors, and runtime consumption paths outside `RegistryManager`.

## RegistryManager Promises Under Review

- Exact-name lookup should win immediately, including case-insensitive matching and alias normalization.
- Base-name lookup should only auto-resolve when simulation time or grid context is provided.
- Variant lookup is keyed off `_LowRes`, `_MidRes`, and `_HigRes` suffixes, with standalone entries treated separately.
- Auto-resolution currently promises sufficient time frequency plus closest grid resolution, with lower waste as a tie-breaker.
- Variable lookup is a simple membership check across loaded reference descriptors.

These promises are documented in `src/openbench/data/registry/manager.py` and will be checked against the broader registry flow in later tasks.

## Discovery Gate

### Confirmed problem: 3-hour datasets are misclassified as hourly

- Classification: Problem
- Code location: `src/openbench/data/registry/scanner.py:421-433`
- Trigger: a dataset directory whose first `.nc` stem contains `3hourly`
- Outcome: `_detect_tim_res()` returns `Hour` because the generic `hourly` branch runs before the more specific `3hour` branch
- Evidence: `pytest -q tests/test_registry/test_scanner_tim_res.py::test_detect_tim_res_prefers_3hour_over_hourly` failed on `sample_3hourly.nc` with `AssertionError: assert 'Hour' == '3Hour'`

### Confirmed limitation: grid discovery stops at the dataset directory plus one child layer

- Classification: Improvement item
- Code location: `src/openbench/data/registry/scanner.py:100-162`
- Trigger: a dataset tree with `.nc` files only below a grandchild of the dataset directory
- Outcome: the scanner checks `dataset_dir` and then only its immediate child directories for `.nc` files; deeper descendants are not discovered
- Evidence: `pytest -q tests/test_registry/test_scanner_tim_res.py::test_scan_reference_directory_misses_grandchildren` returned no dataset for `DatasetC`

### Scope clarification: Composite is intentionally skipped

- Classification: Improvement item
- Code location: `src/openbench/data/registry/scanner.py:114-119`
- Trigger: `Grid/<Res>/Composite/...`
- Outcome: the scanner emits a progress note and skips the category instead of treating it as a standard discovered dataset
- Evidence: `pytest -q tests/test_registry/test_scanner_tim_res.py::test_scan_reference_directory_skips_composite` does not return `DatasetD`, matching the explicit `Skipping Composite/ (register manually)` branch

### Confirmed limitation: nested child NC search stops at one level

- Classification: Improvement item
- Code location: `src/openbench/data/registry/scanner.py:130-138`
- Trigger: `.nc` files stored deeper than one child below the dataset directory
- Outcome: only immediate child directories are inspected for `.nc` files; grandchildren and deeper descendants are ignored
- Evidence: `pytest -q tests/test_registry/test_scanner_tim_res.py::test_scan_reference_directory_discovers_one_level_nested_children` discovered `DatasetB` with one nested child, while `pytest -q tests/test_registry/test_scanner_tim_res.py::test_scan_reference_directory_misses_grandchildren` returned no dataset for `DatasetC`

### Confirmed assumption: variable discovery is path-driven, not content-driven

- Classification: Improvement item
- Code location: `src/openbench/data/registry/scanner.py:122-162` and `src/openbench/data/registry/scanner.py:173-196`
- Trigger: variable folders whose names do not map 1:1 to the desired logical variable identity
- Outcome: the scanner uses the folder name as the variable key and records only the first subdirectory seen for that key; it does not infer variables from file contents during discovery
- Evidence: grid scanning stores `scanned.variables[var_name] = sub_dir` only on first sight of a folder name, and station scanning uses `var_name` directly as the key

## Registration Gate

### Confirmed problem: registration persists unverified default metadata as authoritative facts

- Classification: Problem
- Code location: `src/openbench/data/registry/scanner.py:266-281`
- Trigger: registering a scanned dataset without confirmed year bounds or other curated top-level metadata
- Outcome: `register_scanned_dataset()` writes `tim_res: Month`, `data_groupby: Year`, `timezone: 0`, `years: [1980, 2023]`, and a grid `grid_res` derived only from the resolution label, even when those values were not verified during scanning
- Evidence: `tests/test_registry/test_scanner_registration.py::test_register_scanned_dataset_does_not_persist_unverified_default_years` failed because the emitted YAML contained `1980` and `2023`; the code hard-codes those defaults before any scan-derived confirmation at lines 272-281

### Confirmed merge boundary: variable metadata is preserved only for exact scanned-variable keys

- Classification: Improvement item
- Code location: `src/openbench/data/registry/scanner.py:289-331`
- Trigger: an `existing_descriptor` whose `variables` mapping uses a different key than the scanned folder name
- Outcome: variable-level `varname`, `varunit`, `prefix`, and `suffix` are preserved only when `existing_descriptor["variables"]` contains the exact scanned variable key; top-level fields are rebuilt from scan-time defaults or scan inspection, so `root_dir`, `grid_res`, `tim_res`, `data_groupby`, `timezone`, and `years` are overwritten rather than merged
- Evidence: `tests/test_registry/test_scanner_registration.py::test_register_scanned_dataset_merges_existing_variable_descriptor_by_scanned_variable_key` passed and showed preserved `varname: ET`, `varunit: mm`, `prefix: pre_`, `suffix: _suf` while `root_dir` was rewritten to the scanned path and `grid_res` to `0.5`

### Cleared suspicion: merge does not accidentally match by variable varname

- Classification: Cleared suspicion
- Code location: `src/openbench/data/registry/scanner.py:289-301`
- Trigger: an existing descriptor that stores curated metadata under `variables["ET"]` while the scanned dataset key is `Evapotranspiration`
- Outcome: no merge occurs; the registration keeps the scan-derived key and does not apply metadata from an alias-style varname entry
- Evidence: `tests/test_registry/test_scanner_registration.py::test_register_scanned_dataset_does_not_match_existing_variable_descriptors_by_varname` passed, and the code checks `if var_name in existing_vars` rather than matching on `varname` or path suffix

### Confirmed behavior: multi-variable NetCDF registration is callback-gated

- Classification: Improvement item
- Code location: `src/openbench/data/registry/scanner.py:309-319`
- Trigger: an inspected NetCDF file exposes 2+ data variables
- Outcome: `on_multi_var` is called with `(var_name, sub_dir, all_vars)` to choose a `varname`; if no callback is supplied, the first discovered variable remains authoritative
- Evidence: code inspection of the `len(all_vars) > 1 and on_multi_var` branch; this is an explicit confirmation hook, not an automatic reconciliation step

### Confirmed problem: caller-side descriptor selection can shadow variant-specific metadata with base-name matches

- Classification: Problem
- Code location: `src/openbench/cli/data.py:361-376`, `src/openbench/gui/app.py:101-118`, `src/openbench/gui/pages/page_ref_data.py:252-265`
- Trigger: a scanned variant has both a standalone base-name descriptor and a resolution-specific registry entry available in the registry
- Outcome: the CLI scan path evaluates `mgr.get_reference(variant.name)` before `mgr.get_reference(variant.registry_name)`, so a truthy base-name match prevents the registry-name fallback; the GUI app and page scan path only query `variant.name` / `base_name`, so they never consider the registry-name descriptor at all
- Evidence: `tests/test_registry/test_scanner_registration.py::test_cli_scan_prefers_base_name_existing_descriptor_before_registry_name` passed and recorded only the base-name lookup; the cited source lines show the exact short-circuit and base-name-only lookup behavior

## Resolution Gate

### Cleared suspicion: exact registry names still win before any base-name resolution

- Classification: Cleared suspicion
- Code location: `src/openbench/data/registry/manager.py:152-192`
- Trigger: querying an exact variant name such as `CARE_LowRes`, even when simulation resolution hints are present
- Outcome: exact-name lookup returns immediately and bypasses variant auto-resolve; base names without context return `None`, while base names with context can resolve to the best variant
- Evidence: `pytest -q /Volumes/Data01/Openbench/tests/test_registry/test_manager.py` passed with `14 passed`, including `test_get_reference_exact_variant_name_wins_over_auto_resolve`, `test_get_reference_base_name_requires_context_when_only_variants_exist`, and `test_get_reference_auto_resolve`

### Confirmed behavior: auto-resolve filters low-frequency candidates, then prioritizes closest grid with waste as a secondary penalty

- Classification: Confirmed behavior
- Code location: `src/openbench/data/registry/manager.py:249-301`
- Trigger: a base-name query has one insufficient-frequency candidate, one valid but farther-grid candidate, and one valid closer-grid candidate with higher-than-needed frequency
- Outcome: `_auto_resolve_variant()` drops the insufficient-frequency option when valid candidates exist, chooses the closest-grid valid candidate, and treats higher-than-needed frequency as only a secondary penalty
- Evidence: `test_auto_resolve_variant_applies_time_filter_grid_priority_and_secondary_waste_penalty` passed in the same `pytest -q /Volumes/Data01/Openbench/tests/test_registry/test_manager.py` run

### Confirmed problem: CLI check and config adapter diverge when comparison resolution is unset and the first simulation entry is incomplete

- Classification: Problem
- Code location: `src/openbench/cli/check.py:32-45`, `src/openbench/config/adapter.py:275-287`
- Trigger: `cfg.comparison.tim_res` and/or `cfg.comparison.grid_res` are unset, and the first simulation entry lacks one of the fallback resolution values while a later simulation entry provides it
- Outcome: `check.py` scans simulation entries until it finds populated `tim_res` and `grid_res`, but `to_legacy_config()` in `config/adapter.py` stops after the first simulation entry regardless of whether that entry actually supplies values; the adapter can therefore leave the derived target resolution unset when `check.py` would recover it
- Evidence: `pytest -q /Volumes/Data01/Openbench/tests/test_registry/test_manager.py` passed with `16 passed`, including `test_check_scans_later_simulation_fallbacks_while_adapter_stops_at_first_entry`, which captured `check_calls == [("CARE", None, None), ("CARE", "Month", 0.25)]` and `adapter_calls == [("CARE", None, None)]`

## Consumption Gate

### Confirmed behavior: GUI variable selection stores exact registry names, and the adapter performs the runtime bind

- Classification: Improvement item
- Code location: `src/openbench/gui/pages/page_variables.py:131-140`, `src/openbench/config/adapter.py:278-295`, `src/openbench/data/registry/manager.py:152-192`
- Trigger: a user selects a reference source in the Variables & References page, or a base-name source is carried into `cfg.reference[var_name]`
- Outcome: the GUI writes `ref.name` into config, and `to_legacy_config()` later resolves that string through `registry.get_reference(ref_source_name, sim_tim_res, sim_grid_res)`; exact names bind immediately, while base names only auto-resolve when simulation context is available
- Evidence: `PageVariables._populate_ref_combo()` stores `ref.name` as combo item data and `load_from_config()` matches the same string; `config/adapter.py` reads `cfg.reference[var_name]` and passes it to `get_reference()`, whose implementation returns exact matches before any auto-resolve branch.

### Confirmed problem: GUI persistence stores a concrete variant name while runtime still accepts unresolved base names

- Classification: Improvement item
- Code location: `src/openbench/gui/pages/page_ref_data.py:300-331`, `src/openbench/gui/pages/page_ref_data.py:337-469`, `src/openbench/cli/check.py:32-79`, `src/openbench/config/adapter.py:289-295`
- Trigger: a dataset has multiple `_LowRes/_MidRes/_HigRes` variants and the user selects it through the GUI registry picker, or later supplies a base-name reference in config
- Outcome: the GUI picker resolves the selection to a full registry name and stores that exact `source_name`; runtime paths still accept an unresolved base name and defer binding to `get_reference()` using comparison/simulation context, so the persisted GUI choice and the runtime bind point are not the same abstraction
- Evidence: `_pick_resolution()` maps the dialog selection back to a full registry name and `_add_from_registry()` persists `source_name` verbatim; `check.py` still calls `get_reference(source, sim_tim_res=target_tim_res, sim_grid_res=target_grid_res)` for base-name references, and `to_legacy_config()` resolves `cfg.reference[var_name]` the same way.

### Confirmed UX/docs inconsistency: `data show` advertises auto-selection, but it is only guidance

- Classification: Improvement item
- Code location: `src/openbench/cli/data.py:240-270`, `src/openbench/cli/check.py:32-79`
- Trigger: a user reads `openbench data show` output for a multi-resolution dataset
- Outcome: `data show` presents `reference: <base_name>  # auto-select best resolution` as usage guidance, but the actual auto-resolution and failure messaging happen in `check.py`, not in `data show` itself
- Evidence: `data show` prints the base-name example and exits, while `check.py` derives target resolution, prints `auto-resolved to <resolved.name>` when binding succeeds, or prints `Please specify one:` with explicit variants when it cannot bind.
