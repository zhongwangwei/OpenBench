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

### Runner
- Overall judgment: The orchestration is mostly wired correctly and the targeted runner/cache/CLI tests are green (`24 passed`), but the phase boundaries are too loose. The biggest risks are comparison-only overreach and cache reuse across configuration changes that alter preprocessing output.
- Confirmed issues:
  - Finding: `run_evaluation(..., comparison_only=True)` still allows groupby, statistics, and report phases to run whenever those options are enabled, because the code reconstructs a synthetic `evaluated` list after preflight and later phase gates only check `evaluated`.
    Category: bug
    Confidence: confirmed
    Impact: correctness
    Action: fix now
  - Finding: The incremental cache hash does not include preprocessing-affecting options such as `time_alignment`, `unified_mask`, or the derived comparison resolution inputs, so a change in those settings can incorrectly reuse stale evaluation outputs.
    Category: bug
    Confidence: confirmed
    Impact: correctness
    Action: fix now
- High-risk risks:
  - Finding: Comparison-only preflight accepts any loose glob match in `metrics/` or `scores/`, so stale or partial artifacts can satisfy the prerequisite check and let the runner proceed with incomplete inputs.
    Category: bug
    Confidence: high-risk
    Trigger condition: an output directory contains old files whose names happen to match `"{var}_*{ref}*{sim}*"`.
    Likely impact: comparison phases can run against incomplete or stale evaluation products, or fail later with less actionable errors.
    Why current tests may not catch it: the current comparison-only tests only prove that one placeholder file is enough to pass preflight; they do not assert completeness or freshness of the prerequisite set.
    Impact: correctness
    Action: add tests first
- Maintainability issues:
  - Finding: `GeneralInfoReader.to_dict()` exposes the live bridge object state, and the runner mutates that dictionary directly (`ref_source`, `sim_source`, `ref_file_override`, `ref_preprocessed`). That makes runner behavior depend on legacy processor internals and undocumented output keys like `casedir`, `ref_varname`, and `sim_varname`.
    Category: maintainability
    Confidence: confirmed
    Impact: extensibility
    Action: queue next
- Performance opportunities:
  - Finding: `EvaluationCache.is_cached()` and `mark_done()` reload the JSON cache from disk on every task, which keeps the logic simple but adds repeated file I/O on larger runs.
    Category: performance
    Confidence: confirmed
    Impact: runtime cost
    Action: queue next
- Recommended modification order:
  1. Make comparison-only mode phase-exclusive so it cannot trigger groupby/statistics/report work.
  2. Extend the cache hash to include the preprocessing and resolution knobs that affect generated outputs.
  3. Add tests for comparison-only completeness and cache invalidation across time-alignment and unified-mask changes.
