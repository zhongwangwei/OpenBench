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
