# Reference Recognition Audit Design

## Goal

Deeply inspect the reference data recognition flow in the new automatic path and determine whether it has correctness problems, whether its behavior is reasonable, and whether it is efficient enough to trust operationally.

This work is explicitly scoped to:

- directory scanning
- registry registration
- config-time and run-time reference resolution

This work explicitly excludes the old manual `ref_data` compatibility path as a primary review target. Old-path code may be mentioned only when it affects the new path's semantics.

## User-Approved Scope

The user confirmed the following priorities and constraints:

- Primary evaluation dimension: correctness
- Review only the new path:
  `scan_reference_directory()` -> `register_scanned_dataset()` -> `RegistryManager` / config adapter resolution
- Distinguish between:
  - actual problems: wrong recognition, missed recognition, wrong variant resolution
  - improvement items: automation gaps, weak defaults, missing tests, maintainability issues
- Final output should contain both:
  - an audit report
  - targeted redesign / repair recommendations

## Review Questions

The audit will answer these questions:

1. Can the scanner reliably discover datasets that follow the intended directory structure?
2. Can the scanner avoid misclassifying time resolution, variables, and dataset identity?
3. Does registration preserve facts discovered during scanning without turning guesses into authoritative metadata?
4. Does registry lookup resolve the same dataset identity that scanning and registration intended?
5. Can a source selected through GUI or YAML resolve to an unexpected variant at runtime?
6. Are the assumptions made in each phase explicit, consistent, and testable?

## System Under Review

The relevant flow is:

1. Directory discovery in [`src/openbench/data/registry/scanner.py`](/Volumes/Data01/Openbench/src/openbench/data/registry/scanner.py)
2. Registry write-back in [`src/openbench/data/registry/scanner.py`](/Volumes/Data01/Openbench/src/openbench/data/registry/scanner.py)
3. Registry loading and resolution in [`src/openbench/data/registry/manager.py`](/Volumes/Data01/Openbench/src/openbench/data/registry/manager.py)
4. Config-time and runtime adaptation in [`src/openbench/config/adapter.py`](/Volumes/Data01/Openbench/src/openbench/config/adapter.py) and [`src/openbench/cli/check.py`](/Volumes/Data01/Openbench/src/openbench/cli/check.py)
5. New-path selection surfaces in [`src/openbench/gui/pages/page_variables.py`](/Volumes/Data01/Openbench/src/openbench/gui/pages/page_variables.py), [`src/openbench/gui/pages/page_ref_data.py`](/Volumes/Data01/Openbench/src/openbench/gui/pages/page_ref_data.py), and [`src/openbench/cli/data.py`](/Volumes/Data01/Openbench/src/openbench/cli/data.py)

## Audit Method

The review will be performed in four gates.

### 1. Discovery Gate

Inspect scanner behavior for:

- supported directory depth and naming assumptions
- grid vs station branching
- multi-variable NetCDF inspection
- time-resolution inference
- handling of non-standard directory structures such as nested data folders or skipped categories

Primary question:
Can valid datasets be missed or misclassified before registration even begins?

### 2. Registration Gate

Inspect registration behavior for:

- descriptor construction
- merge behavior with existing descriptors
- default values for years, grid resolution, grouping, timezone, and variable metadata
- whether inferred values are stored as if they were confirmed facts

Primary question:
Does registration faithfully represent what scanning knows, and clearly separate facts from guesses?

### 3. Resolution Gate

Inspect registry resolution behavior for:

- exact-name lookup
- base-name multi-resolution lookup
- automatic variant selection
- variable-based reference listing
- consistency between lookup and registered naming conventions

Primary question:
Can the same logical dataset resolve differently depending on where the name is used?

### 4. Consumption Gate

Inspect how resolved references are consumed by:

- config validation
- legacy namelist generation
- GUI selection surfaces
- CLI information and scan commands

Primary question:
Can the system accept a source name that appears valid but binds to the wrong reference metadata at runtime?

## Classification Rules

### Problems

An item is a problem if it can cause any of the following:

- wrong dataset recognition
- missed dataset recognition
- wrong variable recognition
- wrong time-resolution recognition
- wrong variant selection at runtime
- inconsistent source semantics across discovery, registration, and resolution that can produce incorrect results

### Improvement Items

An item is an improvement item if it does not directly force an incorrect result but still weakens confidence, including:

- manual confirmation required for safe use
- defaults that hide uncertainty
- duplicated logic across GUI / CLI / adapter
- missing tests around claimed behavior
- performance or maintainability issues without direct correctness impact

## Evidence Standard

Each audit finding should include:

- the specific code location(s)
- the triggering condition
- the likely outcome
- whether the conclusion is directly proven by code or inferred from behavior coupling

No finding should rely on style preference alone.

## Current High-Priority Suspicion List

These are not yet final findings, but they are strong audit candidates already visible from the code:

1. Time-resolution detection likely has ordering-sensitive misclassification risk.
   In [`src/openbench/data/registry/scanner.py`](/Volumes/Data01/Openbench/src/openbench/data/registry/scanner.py), `_detect_tim_res()` checks general hourly markers before more specific `3hour` markers.

2. Registration stores strong defaults where data may actually be unknown.
   In [`src/openbench/data/registry/scanner.py`](/Volumes/Data01/Openbench/src/openbench/data/registry/scanner.py), `register_scanned_dataset()` assigns defaults for `years`, `data_groupby`, and grid resolution even when those values may not be confirmed.

3. Discovery and runtime resolution are coupled only by convention, not by validation.
   Scanning infers metadata from path and file patterns, but runtime resolution trusts registry metadata as authoritative.

4. Multi-resolution semantics differ across entry points.
   GUI and CLI surfaces do not present identical source-selection semantics for grouped resolution variants.

5. Scanner and registration behavior appear under-tested relative to their correctness burden.
   Existing tests heavily cover registry querying but provide little or no direct coverage for scanning and registration.

## Output Plan

The final deliverable will have two parts.

### Part A: Audit Report

Structure:

1. Flow summary
2. Confirmed problems
3. High-risk improvement items
4. Evidence and trigger conditions
5. Overall judgment on trustworthiness of the current flow

### Part B: Redesign Recommendations

Recommendations will be grouped into three levels:

1. Minimal repairs
   - fix clear misclassification / misresolution behavior
   - add targeted tests
   - remove or weaken dangerous defaults

2. Moderate cleanup
   - unify dataset identity and variant-selection semantics across scanner, registry, GUI, and adapter
   - separate inferred metadata from confirmed metadata

3. Structural redesign
   - decouple candidate discovery from authoritative registration
   - require explicit normalization / confirmation before persisting descriptors

The likely default recommendation is "minimal repairs plus selected moderate cleanup" unless the audit shows systemic semantic drift.

## Non-Goals

This work will not:

- redesign the entire reference-data UX
- replace the registry format unless the audit proves it necessary
- fold old manual `ref_data` flows into the new path review except where they directly distort new-path behavior
- prioritize stylistic refactoring over correctness

## Approval State

This design has been validated interactively with the user and is ready for spec review and then implementation planning of the audit work.
