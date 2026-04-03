# Main-Flow Audit Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute a deep audit of the active local main flow in `src/openbench`, produce the first module-level audit summary, and prepare the findings for user-directed incremental fixes.

**Architecture:** The plan follows the real local execution path rather than a file-by-file sweep. It first establishes the active call graph and test boundaries, then audits `CLI`, `config`, `registry`, `runner`, and `core/data` in flow order, and finally consolidates confirmed issues, high-risk risks, and modification priorities into a single review artifact.

**Tech Stack:** Python, pytest, Click, ripgrep, git, Markdown

---

## File Structure

**Primary spec input:**
- `docs/superpowers/specs/2026-04-03-main-flow-audit-design.md`

**Inspect:**
- `src/openbench/cli/main.py`
- `src/openbench/cli/run.py`
- `src/openbench/cli/check.py`
- `src/openbench/config/loader.py`
- `src/openbench/config/schema.py`
- `src/openbench/config/adapter.py`
- `src/openbench/config/resolver.py`
- `src/openbench/data/registry/manager.py`
- `src/openbench/data/registry/scanner.py`
- `src/openbench/data/registry/schema.py`
- `src/openbench/runner/local.py`
- `src/openbench/data/processing.py`
- `src/openbench/data/pipeline.py`
- `src/openbench/core/evaluation.py`
- `src/openbench/core/evaluation_engine.py`
- `tests/test_cli_stubs.py`
- `tests/test_cli_integration.py`
- `tests/test_config/`
- `tests/test_registry/`
- `tests/test_runner/test_local.py`
- `tests/test_processing_registry_cache.py`

**Create:**
- `docs/superpowers/reviews/2026-04-03-main-flow-audit.md`

**Optional follow-up reference only after user selects a fix slice:**
- the module and test files named in the audit findings

### Task 1: Establish Active Main-Flow Map and Test Baseline

**Files:**
- Inspect: `docs/superpowers/specs/2026-04-03-main-flow-audit-design.md`
- Inspect: `src/openbench/cli/main.py`
- Inspect: `src/openbench/cli/run.py`
- Inspect: `src/openbench/runner/local.py`
- Inspect: `tests/test_cli_stubs.py`
- Inspect: `tests/test_cli_integration.py`
- Inspect: `tests/test_runner/test_local.py`

- [ ] **Step 1: Re-read the approved spec and extract audit constraints**

Run:

```bash
sed -n '1,260p' docs/superpowers/specs/2026-04-03-main-flow-audit-design.md
```

Expected: scope, evidence standard, output structure, and post-audit change protocol are clear before any audit notes are written

- [ ] **Step 2: Map the active local entry path**

Run:

```bash
sed -n '1,220p' src/openbench/cli/main.py
sed -n '1,260p' src/openbench/cli/run.py
sed -n '240,620p' src/openbench/runner/local.py
```

Expected: a concrete local flow map from CLI entry to runner phases is documented in working notes

- [ ] **Step 3: Confirm the available regression baseline**

Run:

```bash
pytest --collect-only -q
pytest -q tests/test_cli_stubs.py tests/test_cli_integration.py tests/test_runner/test_local.py
```

Expected: test collection succeeds and the current targeted baseline passes before audit conclusions are published

- [ ] **Step 4: Create the audit review document scaffold**

Create `docs/superpowers/reviews/2026-04-03-main-flow-audit.md` with these top-level sections:

```markdown
# Main-Flow Audit

Date: 2026-04-03
Scope: `src/openbench` local main flow

## Overall Assessment
## Architectural Conclusions
## Module Audit
## Cross-Module Issues
## Recommended Change Priority
```

- [ ] **Step 5: Commit the audit scaffold and baseline evidence**

```bash
git add docs/superpowers/reviews/2026-04-03-main-flow-audit.md
git commit -m "docs: add main-flow audit scaffold"
```

### Task 2: Audit CLI and Config Boundary

**Files:**
- Inspect: `src/openbench/cli/main.py`
- Inspect: `src/openbench/cli/run.py`
- Inspect: `src/openbench/cli/check.py`
- Inspect: `src/openbench/config/loader.py`
- Inspect: `src/openbench/config/schema.py`
- Inspect: `src/openbench/config/adapter.py`
- Inspect: `src/openbench/config/resolver.py`
- Inspect: `tests/test_cli_stubs.py`
- Inspect: `tests/test_cli_integration.py`
- Inspect: `tests/test_config/`
- Modify: `docs/superpowers/reviews/2026-04-03-main-flow-audit.md`

- [ ] **Step 1: Read CLI command wiring and error/reporting paths**

Run:

```bash
sed -n '1,220p' src/openbench/cli/main.py
sed -n '1,260p' src/openbench/cli/run.py
sed -n '1,260p' src/openbench/cli/check.py
```

Expected: command registration, lazy loading, overrides, dry-run behavior, and failure reporting paths are explicit

- [ ] **Step 2: Read config construction and bridge layers**

Run:

```bash
sed -n '1,260p' src/openbench/config/loader.py
sed -n '1,260p' src/openbench/config/schema.py
sed -n '1,340p' src/openbench/config/adapter.py
sed -n '1,320p' src/openbench/config/resolver.py
```

Expected: validation boundaries, defaults/include behavior, adapter assumptions, and resolver coupling are clear

- [ ] **Step 3: Cross-check tests that constrain CLI and config behavior**

Run:

```bash
sed -n '1,260p' tests/test_cli_stubs.py
sed -n '1,260p' tests/test_cli_integration.py
rg -n "load_config|resolve_all_references|build_legacy_namelists|time_alignment|strict_reference" tests/test_config tests/test_cli_stubs.py tests/test_cli_integration.py
```

Expected: covered and uncovered CLI/config behaviors are identified

- [ ] **Step 4: Write CLI and config findings into the audit document**

Update `docs/superpowers/reviews/2026-04-03-main-flow-audit.md` so the `Module Audit` section contains:

```markdown
### CLI
- Overall judgment:
- Confirmed issues:
- High-risk risks:
- Maintainability issues:
- Performance opportunities:
- Recommended modification order:

### Config
- Overall judgment:
- Confirmed issues:
- High-risk risks:
- Maintainability issues:
- Performance opportunities:
- Recommended modification order:
```

- [ ] **Step 5: Sanity-check evidence before moving on**

Run:

```bash
rg -n "^### CLI|^### Config|confirmed|high-risk|Recommended modification order" docs/superpowers/reviews/2026-04-03-main-flow-audit.md
```

Expected: both sections exist and distinguish confirmed issues from high-risk risks

### Task 3: Audit Registry Resolution and Cache Behavior

**Files:**
- Inspect: `src/openbench/data/registry/manager.py`
- Inspect: `src/openbench/data/registry/scanner.py`
- Inspect: `src/openbench/data/registry/schema.py`
- Inspect: `tests/test_registry/`
- Inspect: `tests/test_processing_registry_cache.py`
- Modify: `docs/superpowers/reviews/2026-04-03-main-flow-audit.md`

- [ ] **Step 1: Read registry load, resolve, and cache paths**

Run:

```bash
sed -n '1,340p' src/openbench/data/registry/manager.py
sed -n '1,420p' src/openbench/data/registry/scanner.py
sed -n '1,260p' src/openbench/data/registry/schema.py
```

Expected: catalog precedence, cache invalidation, and auto-resolve behavior are mapped

- [ ] **Step 2: Read registry tests for current guarantees**

Run:

```bash
sed -n '1,260p' tests/test_registry/test_manager.py
sed -n '1,260p' tests/test_registry/test_resolver.py
sed -n '1,260p' tests/test_registry/test_scanner_registration.py
sed -n '1,220p' tests/test_processing_registry_cache.py
```

Expected: existing guarantees around resolution, provenance, scanning, and registry caching are explicit

- [ ] **Step 3: Run the targeted registry baseline**

Run:

```bash
pytest -q tests/test_registry/test_manager.py tests/test_registry/test_resolver.py tests/test_registry/test_scanner_registration.py tests/test_processing_registry_cache.py
```

Expected: PASS

- [ ] **Step 4: Write registry findings into the audit document**

Update `docs/superpowers/reviews/2026-04-03-main-flow-audit.md` with:

```markdown
### Registry
- Overall judgment:
- Confirmed issues:
- High-risk risks:
- Maintainability issues:
- Performance opportunities:
- Recommended modification order:
```

- [ ] **Step 5: Re-check that registry findings cite test or code evidence**

Run:

```bash
rg -n "^### Registry|confirmed|high-risk|cache|resolve|scanner" docs/superpowers/reviews/2026-04-03-main-flow-audit.md
```

Expected: registry section contains evidence-backed findings, not generic cleanup suggestions

### Task 4: Audit Runner Orchestration and Phase Coupling

**Files:**
- Inspect: `src/openbench/runner/local.py`
- Inspect: `src/openbench/config/legacy_processors.py`
- Inspect: `tests/test_runner/test_local.py`
- Inspect: `tests/test_cli_integration.py`
- Modify: `docs/superpowers/reviews/2026-04-03-main-flow-audit.md`

- [ ] **Step 1: Read the full local runner orchestration**

Run:

```bash
sed -n '1,260p' src/openbench/runner/local.py
sed -n '260,760p' src/openbench/runner/local.py
```

Expected: preprocessing, evaluation, comparison, statistics, report, and partial-failure handling are all mapped

- [ ] **Step 2: Read the bridge object used by runner task assembly**

Run:

```bash
rg -n "class GeneralInfoReader|def to_dict" src/openbench/config/legacy_processors.py
sed -n '1,260p' src/openbench/config/legacy_processors.py
```

Expected: runner dependence on legacy bridge state and output conventions is explicit

- [ ] **Step 3: Cross-check runner tests and CLI integration coverage**

Run:

```bash
sed -n '1,260p' tests/test_runner/test_local.py
rg -n "comparison_only|preprocess|partial|run_evaluation|dump_config" tests/test_runner/test_local.py tests/test_cli_integration.py
pytest -q tests/test_runner/test_local.py tests/test_cli_integration.py
```

Expected: current safeguards and obvious orchestration gaps are identified

- [ ] **Step 4: Write runner findings into the audit document**

Update `docs/superpowers/reviews/2026-04-03-main-flow-audit.md` with:

```markdown
### Runner
- Overall judgment:
- Confirmed issues:
- High-risk risks:
- Maintainability issues:
- Performance opportunities:
- Recommended modification order:
```

- [ ] **Step 5: Verify the runner section distinguishes correctness from structural debt**

Run:

```bash
rg -n "^### Runner|partial|comparison|cache|mask|legacy" docs/superpowers/reviews/2026-04-03-main-flow-audit.md
```

Expected: runner issues are separated into bug, maintainability, and performance buckets

### Task 5: Audit Core/Data Execution Path

**Files:**
- Inspect: `src/openbench/data/processing.py`
- Inspect: `src/openbench/data/pipeline.py`
- Inspect: `src/openbench/core/evaluation.py`
- Inspect: `src/openbench/core/evaluation_engine.py`
- Inspect: `tests/test_compute.py`
- Inspect: `tests/test_core/`
- Modify: `docs/superpowers/reviews/2026-04-03-main-flow-audit.md`

- [ ] **Step 1: Read active data processing entry points**

Run:

```bash
sed -n '1,260p' src/openbench/data/processing.py
sed -n '1,260p' src/openbench/data/pipeline.py
```

Expected: active preprocessing contracts, heavy imports, cache assumptions, and repeated I/O patterns are clear

- [ ] **Step 2: Read active evaluation entry points**

Run:

```bash
sed -n '1,260p' src/openbench/core/evaluation.py
sed -n '1,260p' src/openbench/core/evaluation_engine.py
```

Expected: actual runtime path vs secondary or legacy-style engine code is distinguishable

- [ ] **Step 3: Read test coverage for core/data behavior**

Run:

```bash
sed -n '1,240p' tests/test_compute.py
sed -n '1,240p' tests/test_core/test_metrics.py
sed -n '1,240p' tests/test_core/test_scores.py
pytest -q tests/test_compute.py tests/test_core/test_metrics.py tests/test_core/test_scores.py
```

Expected: known protections and obvious untested execution surfaces are identified

- [ ] **Step 4: Write core/data findings into the audit document**

Update `docs/superpowers/reviews/2026-04-03-main-flow-audit.md` with:

```markdown
### Core/Data
- Overall judgment:
- Confirmed issues:
- High-risk risks:
- Maintainability issues:
- Performance opportunities:
- Recommended modification order:
```

- [ ] **Step 5: Check that active-path findings are not diluted by out-of-scope legacy modules**

Run:

```bash
rg -n "^### Core/Data|processing|evaluation|pipeline|legacy" docs/superpowers/reviews/2026-04-03-main-flow-audit.md
```

Expected: findings stay anchored to the active local main flow

### Task 6: Consolidate Cross-Module Conclusions and Fix Queue

**Files:**
- Modify: `docs/superpowers/reviews/2026-04-03-main-flow-audit.md`
- Inspect: `docs/superpowers/specs/2026-04-03-main-flow-audit-design.md`

- [ ] **Step 1: Derive cross-module findings from the completed module sections**

Add to `docs/superpowers/reviews/2026-04-03-main-flow-audit.md`:

```markdown
## Cross-Module Issues
- Shared issue:
- Affected layers:
- Why it matters:
- Confidence:
- Recommended action:
```

- [ ] **Step 2: Build a user-facing modification priority queue**

Add to `docs/superpowers/reviews/2026-04-03-main-flow-audit.md`:

```markdown
## Recommended Change Priority
1. [P0/P1 issue group]
2. [P1/P2 issue group]
3. [P2 maintainability group]
4. [P2/P3 performance group]
```

Expected: the queue is ordered for incremental fixes rather than one large refactor

- [ ] **Step 3: Review the audit document against the approved spec**

Run:

```bash
sed -n '1,260p' docs/superpowers/specs/2026-04-03-main-flow-audit-design.md
sed -n '1,320p' docs/superpowers/reviews/2026-04-03-main-flow-audit.md
```

Expected: the audit document matches the approved scope, structure, and evidence standard

- [ ] **Step 4: Perform a final targeted regression check before publishing findings**

Run:

```bash
pytest -q tests/test_cli_stubs.py tests/test_cli_integration.py tests/test_registry/test_manager.py tests/test_registry/test_resolver.py tests/test_registry/test_scanner_registration.py tests/test_processing_registry_cache.py tests/test_runner/test_local.py tests/test_compute.py tests/test_core/test_metrics.py tests/test_core/test_scores.py
```

Expected: PASS

- [ ] **Step 5: Commit the completed audit document**

```bash
git add docs/superpowers/reviews/2026-04-03-main-flow-audit.md
git commit -m "docs: add main-flow audit findings"
```

### Task 7: Prepare First Fix-Slice Handoff

**Files:**
- Inspect: `docs/superpowers/reviews/2026-04-03-main-flow-audit.md`

- [ ] **Step 1: Extract the first recommended issue group**

Read the top item under `## Recommended Change Priority` and write down:

```markdown
- target module:
- issue group:
- confidence:
- expected user-visible impact:
```

- [ ] **Step 2: Convert that issue group into a bounded change slice**

Prepare a short handoff with:

```markdown
- scope:
- files likely to change:
- tests to add or update:
- regression risks:
- verification commands:
```

- [ ] **Step 3: Present the audit summary and ask the user which slice to modify first**

Expected: the user receives the module-level audit plus a clear first-slice recommendation, and implementation still waits for explicit user direction
