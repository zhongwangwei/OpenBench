# CLI/Local Dead Code Cleanup Design

Date: 2026-04-03

## Summary

This cleanup removes modules that are no longer part of the current `CLI/local`
execution path and still encode obsolete execution assumptions such as
`openbench/openbench.py` script discovery or old namelist updater flows.

The cleanup is intentionally conservative:

- Remove clearly dead code and detached old tool modules.
- Do not change the active `openbench run` / `openbench check` path.
- Do not remove the current legacy bridge used by the local runner.
- Do not touch GUI/remote old-path logic in this batch.

## Scope

Delete these modules:

- `src/openbench/runner/wizard_runner_reference.py`
- `src/openbench/config/legacy_updaters.py`
- `src/openbench/data/preprocessing.py`

Keep these modules for now because they still participate in the active bridge
between the new YAML config system and the old execution engine:

- `src/openbench/config/adapter.py`
- `src/openbench/config/legacy_processors.py`
- `src/openbench/config/legacy_manager.py`
- `src/openbench/config/legacy_readers.py`
- `src/openbench/config/migration.py`
- `src/openbench/runner/local.py`

## Why These Files

### `runner/wizard_runner_reference.py`

This module is not referenced by the current repository call graph and still
assumes the old `openbench/openbench.py` entry script model. That is no longer
the current CLI contract.

### `config/legacy_updaters.py`

This module implements old namelist update/merge behavior and is not part of
the current `resolver -> adapter -> local runner` flow.

### `data/preprocessing.py`

This is an old preprocessing utility module detached from the active local
runner path. It also contains a stale import expectation for
`openbench.config.GeneralInfoReader`, which is not exported by the current
package API. That makes it both dead weight and a latent maintenance trap.

## Non-Goals

- Replacing the legacy bridge used by the current local runner.
- Rewriting the evaluation engine to consume the new config model directly.
- Cleaning GUI/remote compatibility layers.
- Deleting top-level tool modules that may still serve external consumers
  without stronger evidence, such as `util/api_service.py`.

## Implementation Plan

1. Delete the three target modules.
2. Re-scan the repository for residual imports, symbol references, and stale
   comments tied to those modules.
3. Apply only the minimal follow-up edits required to remove broken imports or
   references.
4. Leave all active CLI/local bridge code untouched.

## Validation

Run:

- `rg` checks for deleted module names and exported symbols
- `pytest -q tests/test_runner/test_local.py`
- `pytest -q tests/test_cli_stubs.py`
- `pytest -q tests/test_cli_integration.py`

Success criteria:

- No in-repo references remain to the deleted modules.
- CLI/local tests continue to pass.
- The active runner path still imports and executes normally.

## Risks

- External scripts outside this repository may still import the deleted
  modules. This cleanup does not preserve that compatibility.
- Some dead modules may have indirect documentation references that require
  small follow-up cleanup after deletion.

## Follow-Up

After this batch, a separate review can decide whether to remove the remaining
legacy bridge from `adapter`, `legacy_processors`, and `runner/local` by
teaching the execution engine to consume the new config model directly.
