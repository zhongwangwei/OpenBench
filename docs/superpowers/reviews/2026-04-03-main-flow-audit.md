# Main-Flow Audit

Date: 2026-04-03
Scope: `src/openbench` local main flow

## Overall Assessment

## Architectural Conclusions

## Module Audit

### Registry
- Overall judgment: Registry loading and resolution are in decent shape and the targeted baseline is green (`66 passed`). Catalog precedence and auto-resolve behavior are test-backed. The strongest issue in this area is stale singleton state after direct scanner registration writes.
- Confirmed issues:
  - Finding: Public scanner registration APIs write new registry data but do not invalidate the singleton registry cache, so a long-lived process can keep serving stale references after `register_scanned_dataset()` or `register_scanned_datasets_batch()`.
    - Category: bug
    - Confidence: confirmed
    - Impact: correctness
    - Action: fix now
- High-risk risks:
  - Finding: No additional registry issue in this pass cleared the evidence bar to classify above the confirmed cache invalidation bug.
    - Category: maintainability
    - Confidence: high-risk
    - Trigger condition: future registry changes add more direct callers outside the resolver and CLI-owned write paths.
    - Likely impact: behavior can drift between entry points before tests make the contract explicit.
    - Why current tests may not catch it: current tests are strongest at resolver behavior and CLI cache clears, but thinner on direct `RegistryManager` caller contracts.
    - Impact: testability
    - Action: add tests first
- Maintainability issues:
  - Finding: Resolution rules are split between `src/openbench/data/registry/manager.py` and `src/openbench/config/resolver.py`, which makes the registry API contract less explicit than the resolver contract.
    - Category: maintainability
    - Confidence: high-risk
    - Impact: extensibility
    - Action: add tests first
- Performance opportunities:
  - Finding: `RegistryManager` already builds a variable index, but registry invalidation is still caller-driven rather than centralized around the write APIs.
    - Category: performance
    - Confidence: high-risk
    - Impact: runtime cost
    - Action: queue next
- Recommended modification order:
  1. Add coverage for cache invalidation after direct scanner registration so stale singleton reads are impossible to miss.
  2. Add direct `RegistryManager` contract tests where behavior is currently inferred through the resolver layer.

## Cross-Module Issues

## Recommended Change Priority
