# Main-Flow Audit

Date: 2026-04-03
Scope: `src/openbench` local main flow

## Overall Assessment

## Architectural Conclusions

## Module Audit

### Registry
- Overall judgment: Registry loading and resolution are mostly solid and the targeted baseline is green (`66 passed`). Catalog precedence is deterministic, auto-resolve is test-backed, and ambiguity is handled in the resolver layer. The main gap is cache invalidation outside the CLI-owned write paths.
- Confirmed issues:
  - Finding: Public scanner registration APIs write new registry data but do not invalidate the singleton registry cache, so a long-lived process can keep serving stale references after `register_scanned_dataset()` or `register_scanned_datasets_batch()`.
    - Category: bug
    - Confidence: confirmed
    - Impact: correctness
    - Action: fix now
- High-risk risks:
  - Finding: `RegistryManager.get_reference()` collapses "not found" and "base name with variants but no simulation context" into the same `None` result, and `openbench.config.resolver.resolve_reference()` has to reconstruct ambiguity by calling `get_resolution_variants()` again.
    - Category: bug
    - Confidence: high-risk
    - Trigger condition: a new caller uses `RegistryManager` directly instead of going through the resolver wrapper.
    - Likely impact: ambiguous references can be misclassified as missing and silently fall back to defaults.
    - Why current tests may not catch it: `tests/test_registry/test_resolver.py` verifies the wrapper behavior, but there is no direct test for consumers of `RegistryManager.get_reference()` alone.
    - Impact: testability
    - Action: add tests first
  - Finding: `_auto_resolve_variant()` performs filesystem `Path(...).is_dir()` checks during lookup to prefer an on-disk variant, so repeated auto-resolve calls pay runtime I/O at resolution time.
    - Category: performance
    - Confidence: high-risk
    - Trigger condition: repeated lookups in GUI/CLI sessions with many registry bindings.
    - Likely impact: lookup latency grows with registry size and filesystem latency.
    - Why current tests may not catch it: the registry tests validate selection semantics, not runtime cost under repeated resolution.
    - Impact: runtime cost
    - Action: queue next
- Maintainability issues:
  - Finding: Resolution rules are split across `src/openbench/data/registry/manager.py` and `src/openbench/config/resolver.py`, with `None`-return ambiguity inferred by a second lookup instead of being explicit in the registry API.
    - Category: maintainability
    - Confidence: high-risk
    - Impact: extensibility
    - Action: add tests first
- Performance opportunities:
  - Finding: `RegistryManager` already builds a variable index, but registry reloads are still eager and cache invalidation is caller-driven; if registry mutation becomes more frequent, the reload path should be centralized around explicit invalidation hooks.
    - Category: performance
    - Confidence: high-risk
    - Impact: runtime cost
    - Action: queue next
- Recommended modification order:
  1. Add coverage for cache invalidation after direct scanner registration so stale singleton reads are impossible to miss.
  2. Add a direct registry-resolution test for the ambiguous base-name case, separate from the resolver wrapper.
  3. If lookup volume matters, move the on-disk variant preference out of `_auto_resolve_variant()` or cache the existence check.

## Cross-Module Issues

## Recommended Change Priority
