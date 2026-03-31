# Reference Recognition Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit the new reference auto-recognition flow for correctness, produce evidence-backed findings, and deliver targeted redesign recommendations without mixing in the legacy manual path.

**Architecture:** Review the flow in four gates: discovery, registration, resolution, and runtime consumption. For each gate, first write a focused failing-or-gap-oriented test or reproducible inspection step, then confirm the current behavior, record evidence, and only then write the audit conclusion and redesign recommendation.

**Tech Stack:** Python, pytest, ripgrep, git, OpenBench registry/config modules

---

## File Structure

### Files To Read During Execution

- [`docs/superpowers/specs/2026-03-31-reference-recognition-audit-design.md`](/Volumes/Data01/Openbench/docs/superpowers/specs/2026-03-31-reference-recognition-audit-design.md): approved scope, classification rules, output contract
- [`src/openbench/data/registry/scanner.py`](/Volumes/Data01/Openbench/src/openbench/data/registry/scanner.py): dataset discovery, registration, station list generation
- [`src/openbench/data/registry/manager.py`](/Volumes/Data01/Openbench/src/openbench/data/registry/manager.py): reference lookup, variant resolution, variable listing
- [`src/openbench/config/adapter.py`](/Volumes/Data01/Openbench/src/openbench/config/adapter.py): runtime resolution into legacy namelists
- [`src/openbench/cli/check.py`](/Volumes/Data01/Openbench/src/openbench/cli/check.py): config-time validation and auto-resolve reporting
- [`src/openbench/cli/data.py`](/Volumes/Data01/Openbench/src/openbench/cli/data.py): scan command and registry display semantics
- [`src/openbench/gui/pages/page_variables.py`](/Volumes/Data01/Openbench/src/openbench/gui/pages/page_variables.py): variable-to-reference selection surface
- [`src/openbench/gui/pages/page_ref_data.py`](/Volumes/Data01/Openbench/src/openbench/gui/pages/page_ref_data.py): registry scan/import flow and grouped resolution UI
- [`tests/test_registry/test_manager.py`](/Volumes/Data01/Openbench/tests/test_registry/test_manager.py): current reference-resolution coverage baseline

### Files To Create Or Modify

- Create: [`docs/superpowers/reviews/2026-03-31-reference-recognition-audit.md`](/Volumes/Data01/Openbench/docs/superpowers/reviews/2026-03-31-reference-recognition-audit.md)
- Create if needed: targeted regression tests under [`tests/test_registry/`](/Volumes/Data01/Openbench/tests/test_registry)
- Modify only if the audit requires proof-by-test for a confirmed behavior gap

## Task 1: Establish Baseline Coverage And Known Promises

**Files:**
- Read: [`docs/superpowers/specs/2026-03-31-reference-recognition-audit-design.md`](/Volumes/Data01/Openbench/docs/superpowers/specs/2026-03-31-reference-recognition-audit-design.md)
- Read: [`tests/test_registry/test_manager.py`](/Volumes/Data01/Openbench/tests/test_registry/test_manager.py)
- Read: [`src/openbench/data/registry/manager.py`](/Volumes/Data01/Openbench/src/openbench/data/registry/manager.py)

- [ ] **Step 1: Summarize what the approved spec says counts as a problem vs an improvement item**

Write a short scratch summary with these headings:

```text
Problems:
- wrong recognition
- missed recognition
- wrong variant resolution

Improvement items:
- manual confirmation gaps
- weak defaults
- missing tests
```

- [ ] **Step 2: Inspect existing tests to list covered and uncovered claims**

Run: `sed -n '1,260p' tests/test_registry/test_manager.py`
Expected: only registry lookup behavior is directly covered; scanner/registration coverage is absent or near-absent

- [ ] **Step 3: Record the baseline coverage gap in the audit draft**

Add a section to the audit draft with this shape:

```markdown
## Baseline Coverage

- Covered: exact reference lookup, auto-resolve, resolution variants, variable listing
- Not covered: directory scanning, descriptor registration, time-resolution inference, station-list generation
```

- [ ] **Step 4: Commit audit scaffolding if a new draft file was created**

```bash
git add docs/superpowers/reviews/2026-03-31-reference-recognition-audit.md
git commit -m "docs: scaffold reference recognition audit report"
```

## Task 2: Audit Discovery Gate

**Files:**
- Read: [`src/openbench/data/registry/scanner.py`](/Volumes/Data01/Openbench/src/openbench/data/registry/scanner.py)
- Test: [`tests/test_registry/`](/Volumes/Data01/Openbench/tests/test_registry)

- [ ] **Step 1: Write a focused failing test for time-resolution inference ordering**

Create a test with this shape:

```python
from pathlib import Path

from openbench.data.registry.scanner import _detect_tim_res


def test_detect_tim_res_prefers_specific_3hour_over_hourly(tmp_path: Path):
    dataset_dir = tmp_path / "foo_3hourly"
    dataset_dir.mkdir()
    (dataset_dir / "var_2001_3hourly.nc").touch()

    assert _detect_tim_res(dataset_dir) == "3Hour"
```

- [ ] **Step 2: Run the targeted test to verify current behavior**

Run: `pytest tests/test_registry/test_scanner_tim_res.py::test_detect_tim_res_prefers_specific_3hour_over_hourly -v`
Expected: FAIL if current behavior misclassifies as `Hour`, otherwise PASS and the suspicion is cleared

- [ ] **Step 3: Inspect scanner discovery assumptions beyond the test**

Read and note:
- directory depth assumptions
- `Composite` handling
- nested child NC search depth
- variable discovery rules that exclude certain dimensions/variables

- [ ] **Step 4: Add confirmed findings or cleared suspicions to the audit draft**

Use this format:

```markdown
## Discovery Gate

### Finding: [title]
- Classification: Problem | Improvement Item
- Code: `src/openbench/data/registry/scanner.py`
- Trigger:
- Outcome:
- Evidence:
```

- [ ] **Step 5: Commit tests or draft updates if new evidence was added**

```bash
git add tests/test_registry/test_scanner_tim_res.py docs/superpowers/reviews/2026-03-31-reference-recognition-audit.md
git commit -m "test: capture scanner time resolution audit case"
```

## Task 3: Audit Registration Gate

**Files:**
- Read: [`src/openbench/data/registry/scanner.py`](/Volumes/Data01/Openbench/src/openbench/data/registry/scanner.py)
- Test: [`tests/test_registry/`](/Volumes/Data01/Openbench/tests/test_registry)

- [ ] **Step 1: Write a failing test for registration default metadata being persisted as facts**

Create a test with this shape:

```python
from openbench.data.registry.scanner import ScannedDataset, register_scanned_dataset


def test_register_scanned_dataset_does_not_write_unverified_default_years(tmp_path):
    catalog = tmp_path / "reference_catalog.yaml"
    scanned = ScannedDataset(
        name="Demo",
        resolution="LowRes",
        category="Water",
        data_type="grid",
        root_dir=str(tmp_path),
        variables={"Evapotranspiration": "Water/Evapotranspiration/Demo"},
    )

    register_scanned_dataset(scanned, catalog_path=catalog)

    text = catalog.read_text()
    assert "1980" not in text
    assert "2023" not in text
```

- [ ] **Step 2: Run the targeted test to confirm whether the concern is real**

Run: `pytest tests/test_registry/test_scanner_registration.py::test_register_scanned_dataset_does_not_write_unverified_default_years -v`
Expected: FAIL if unknown metadata is currently materialized as fixed values

- [ ] **Step 3: Inspect merge semantics with existing descriptors**

Verify:
- whether existing variable mappings are preserved
- whether non-variable metadata is preserved or silently replaced
- whether base-name vs registry-name matching can merge against the wrong prior descriptor

- [ ] **Step 4: Record each registration-stage conclusion in the audit draft**

At minimum capture:
- default years
- default `data_groupby`
- grid resolution inferred from resolution label only
- user confirmation path for multi-variable NC files

- [ ] **Step 5: Commit tests or draft updates if evidence changed**

```bash
git add tests/test_registry/test_scanner_registration.py docs/superpowers/reviews/2026-03-31-reference-recognition-audit.md
git commit -m "test: capture registry registration audit cases"
```

## Task 4: Audit Resolution Gate

**Files:**
- Read: [`src/openbench/data/registry/manager.py`](/Volumes/Data01/Openbench/src/openbench/data/registry/manager.py)
- Read: [`src/openbench/cli/check.py`](/Volumes/Data01/Openbench/src/openbench/cli/check.py)
- Test: [`tests/test_registry/test_manager.py`](/Volumes/Data01/Openbench/tests/test_registry/test_manager.py)

- [ ] **Step 1: Write a failing test for base-name and variant-name semantic consistency**

Create a test with this shape:

```python
from openbench.data.registry.manager import RegistryManager


def test_get_reference_base_name_without_context_never_silently_picks_variant():
    mgr = RegistryManager()
    ref = mgr.get_reference("GLEAM_v4.2a")

    assert ref is None or ref.name == "GLEAM_v4.2a"
```

- [ ] **Step 2: Write a focused test for auto-resolve tie-breaking semantics**

Create a test that proves:
- insufficient time frequency is excluded when possible
- closest grid resolution wins among valid candidates
- higher-than-needed frequency is treated as secondary penalty

- [ ] **Step 3: Run the targeted resolution tests**

Run: `pytest tests/test_registry/test_manager.py -k "reference or variants" -v`
Expected: evidence of whether current rules match the stated comments in `_auto_resolve_variant()`

- [ ] **Step 4: Compare resolution behavior across CLI and adapter**

Inspect:
- `RegistryManager.get_reference()`
- `cli/check.py`
- `config/adapter.py`

Record any mismatch between:
- when base names are accepted
- when variant names are required
- how target time/grid resolution is derived

- [ ] **Step 5: Commit tests or draft updates if new evidence was added**

```bash
git add tests/test_registry/test_manager.py docs/superpowers/reviews/2026-03-31-reference-recognition-audit.md
git commit -m "test: add reference variant resolution audit cases"
```

## Task 5: Audit Consumption Gate

**Files:**
- Read: [`src/openbench/config/adapter.py`](/Volumes/Data01/Openbench/src/openbench/config/adapter.py)
- Read: [`src/openbench/gui/pages/page_variables.py`](/Volumes/Data01/Openbench/src/openbench/gui/pages/page_variables.py)
- Read: [`src/openbench/gui/pages/page_ref_data.py`](/Volumes/Data01/Openbench/src/openbench/gui/pages/page_ref_data.py)
- Read: [`src/openbench/cli/data.py`](/Volumes/Data01/Openbench/src/openbench/cli/data.py)

- [ ] **Step 1: Trace how a source name chosen in each entry point becomes a runtime binding**

Build a scratch matrix:

```text
Entry point | accepted value | grouped variants? | runtime resolver | risk
```

- [ ] **Step 2: Confirm whether GUI and CLI expose the same multi-resolution semantics**

Check specifically:
- grouped base names in `page_ref_data.py`
- flat reference listing in `page_variables.py`
- auto-resolve messaging in `cli/check.py`
- scan/registration merge behavior in `cli/data.py`

- [ ] **Step 3: Add any semantic mismatch to the audit draft**

Use this format:

```markdown
### Finding: Cross-entry-point semantic mismatch
- Entry points:
- User-visible expectation:
- Actual binding behavior:
- Classification:
```

- [ ] **Step 4: Commit draft updates if needed**

```bash
git add docs/superpowers/reviews/2026-03-31-reference-recognition-audit.md
git commit -m "docs: capture reference consumption audit findings"
```

## Task 6: Produce Final Audit Judgment And Redesign Recommendations

**Files:**
- Modify: [`docs/superpowers/reviews/2026-03-31-reference-recognition-audit.md`](/Volumes/Data01/Openbench/docs/superpowers/reviews/2026-03-31-reference-recognition-audit.md)
- Read: [`docs/superpowers/specs/2026-03-31-reference-recognition-audit-design.md`](/Volumes/Data01/Openbench/docs/superpowers/specs/2026-03-31-reference-recognition-audit-design.md)

- [ ] **Step 1: Convert notes into final findings with strict classification**

Ensure every finding is labeled:
- `Problem`
- `Improvement Item`
- `Cleared Suspicion`

- [ ] **Step 2: Write the overall trust judgment**

Choose one and justify it:

```text
Usable
Usable but fragile
Not trustworthy without fixes
```

- [ ] **Step 3: Write redesign recommendations in three layers**

Required headings:

```markdown
## Minimal Repairs
## Moderate Cleanup
## Structural Redesign
```

- [ ] **Step 4: Add a recommended execution order**

Example format:

```markdown
1. Fix misclassification bugs
2. Remove dangerous default persistence
3. Add regression tests
4. Unify variant semantics across entry points
```

- [ ] **Step 5: Run a final grep for placeholders**

Run: `rg -n "TODO|TBD|placeholder" docs/superpowers/reviews/2026-03-31-reference-recognition-audit.md`
Expected: no matches

- [ ] **Step 6: Commit the completed audit report**

```bash
git add docs/superpowers/reviews/2026-03-31-reference-recognition-audit.md tests/test_registry
git commit -m "docs: complete reference recognition audit"
```

## Task 7: Verification Before Handoff

**Files:**
- Read: [`docs/superpowers/reviews/2026-03-31-reference-recognition-audit.md`](/Volumes/Data01/Openbench/docs/superpowers/reviews/2026-03-31-reference-recognition-audit.md)

- [ ] **Step 1: Run all audit-related tests**

Run: `pytest tests/test_registry -v`
Expected: all newly added audit/regression tests pass

- [ ] **Step 2: Verify git working tree is clean**

Run: `git status --short`
Expected: no output

- [ ] **Step 3: Prepare user-facing summary**

The summary must include:
- audit report path
- confirmed top problems
- top redesign recommendation
- tests run and their result

- [ ] **Step 4: Final handoff commit if verification changes were needed**

```bash
git add docs/superpowers/reviews/2026-03-31-reference-recognition-audit.md tests/test_registry
git commit -m "chore: finalize reference recognition audit handoff"
```
