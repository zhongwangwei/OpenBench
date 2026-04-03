# Main-Flow Audit Design

Date: 2026-04-03
Scope: `src/openbench` current production path only

## Objective

Perform a deep code audit of the local execution path in `src/openbench`, starting from the overall workflow and drilling down module by module, before making any code changes. The audit must give equal weight to:

- bug discovery and correctness risk
- maintainability and structural clarity
- performance and unnecessary runtime cost

The first deliverable is a module-level audit summary, not code changes. Implementation happens later, only after the audit results are reviewed and the next change slice is explicitly selected.

## In Scope

Only the current local main flow and its corresponding tests:

- `openbench.cli`
- `openbench.config`
- `openbench.data.registry`
- `openbench.runner`
- `openbench.core`
- `openbench.data`
- matching tests under `tests/`

## Out of Scope

The following are intentionally excluded from the first audit pass:

- `src/openbench/gui`
- `src/openbench/remote`
- historical or sidecar directories in the repository root such as `OpenBench-wei` and `openbench-wizard`
- unrelated cleanup outside the local main flow

## Audit Strategy

Use a main-flow-driven audit rather than a file-by-file sweep.

The review follows the real execution path:

`CLI -> config -> registry -> runner -> core/data`

The audit first traces how a local run is assembled and executed, then maps findings back into the affected module. This is intended to surface both local problems and cross-module issues such as:

- mismatched assumptions between layers
- incorrect or inconsistent error propagation
- duplicated I/O or repeated configuration work
- hidden state coupling through legacy adapters
- test coverage gaps around real workflow boundaries

## Deliverable Structure

The first review output will use this fixed structure:

1. Overall assessment
2. Architectural conclusions
3. Module audit
4. Cross-module issues
5. Recommended change priority

Each module section will use the same template:

- overall judgment
- confirmed issues
- high-risk but not yet reproduced risks
- maintainability issues
- performance opportunities
- recommended modification order

Modules to report separately:

- CLI
- config
- registry
- runner
- core/data

## Evidence Standard

Every finding must be explicitly labeled so confirmed faults are not mixed with plausible but unverified risks.

### Confirmed issues

A finding is confirmed only when it is supported by one or more of:

- a concrete code path with an identifiable failure mode
- a control-flow or state-management flaw visible from the implementation
- an inconsistency between module contract and actual behavior
- existing test evidence that exposes or constrains the behavior

### High-risk hidden issues

A finding is high-risk when it has not been directly reproduced, but there is a strong reason to expect instability based on:

- shared mutable state
- repeated file I/O on shared outputs
- weak interface boundaries across legacy bridge layers
- missing validation or partial exception handling
- code structure that suggests fragile behavior under realistic inputs

These findings must include:

- trigger condition
- likely impact
- why current tests may not catch it

## Classification Model

Each item in the audit should carry four labels:

- category: `bug`, `maintainability`, or `performance`
- confidence: `confirmed` or `high-risk`
- impact: correctness, testability, extensibility, or runtime cost
- action: fix now, queue next, add tests first, or defer

Severity should also be implied through priority:

- `P0/P1`: can directly corrupt results, break execution, mis-handle state, or produce wrong outputs
- `P2`: does not always fail immediately, but creates material maintenance or scaling risk
- `P3`: localized cleanup or design polish with limited immediate impact

## Review Focus By Layer

### CLI

Review:

- command wiring and lazy loading behavior
- override handling
- dry-run and failure reporting consistency
- mismatch risk between user-facing command behavior and runner expectations

### Config

Review:

- schema construction and validation boundaries
- include/default merge behavior
- adapter behavior between new config objects and legacy namelists
- error clarity and assumptions that leak into downstream layers

### Registry

Review:

- catalog loading precedence
- cache behavior and invalidation
- exact-match vs auto-resolve behavior
- variant selection logic and ambiguity handling

### Runner

Review:

- orchestration boundaries
- preprocessing/evaluation/comparison phase coupling
- cache key and config hash behavior
- shared output handling
- comparison-only mode and partial-failure behavior

### Core/Data

Review:

- data preprocessing contracts
- evaluation entry points and adapter coupling
- repeated dataset loading or conversion work
- failure isolation around expensive operations
- boundaries between active execution code and legacy-heavy modules

## Testing Expectations

The audit must use existing tests as evidence, but should not treat a green test suite as proof that the workflow is safe.

The review should explicitly identify:

- where tests already protect current behavior
- where behavior depends on untested integration paths
- where future fixes should be accompanied by new tests

## Post-Audit Change Protocol

No bulk rewrite follows the first audit.

After the module-level audit is delivered, changes proceed incrementally:

1. Select one module or one bounded issue group from the audit.
2. Define a small modification slice with change scope, expected behavior, regression risk, and verification plan.
3. Review that slice before editing code.
4. Implement the slice, add or adjust tests, and verify.
5. Return to the audit list for the next selected slice.

This protocol is intended to keep bug fixes, structural cleanup, and performance work from being mixed into one uncontrolled refactor.

## Success Criteria

This design is successful if it produces:

- a trustworthy first-pass audit of the local main flow
- a clear separation between confirmed problems and high-risk risks
- a review format that can drive user-directed incremental edits
- a modification sequence that preserves evidence and minimizes regression risk
