"""Static metric and score applicability checks for CLI preflight."""

from __future__ import annotations

from typing import Any

# Keep these defaults consistent with the normal evaluation configuration.
_DEFAULT_METRICS = ("bias", "RMSE", "correlation")
_DEFAULT_SCORES = ("Overall_Score",)

# A single annual sample cannot provide the temporal variation required by
# these calculations. Data-dependent requirements remain runtime checks.
_SINGLE_ANNUAL_SAMPLE_METRICS = frozenset(
    {
        "RSR",
        "ubRMSE",
        "CRMSD",
        "L",
        "correlation",
        "correlation_R2",
        "NSE",
        "KGE",
        "KGESS",
        "rv",
        "ubNSE",
        "ubcorrelation",
        "ubcorrelation_R2",
        "pc_ampli",
        "rSD",
        "rSpearman",
        "mNSE",
        "rNSE",
        "mKGE",
        "KGEkm",
        "KGEln",
        "KGElf",
        "br2",
        "cp",
        "LNSE",
        "MFM_omega",
        "MFM_varphi",
        "MFM_eta",
        "MFM",
    }
)
_SINGLE_ANNUAL_SAMPLE_SCORES = frozenset({"nBiasScore", "nRMSEScore"})


def effective_metric_tim_res(cfg: Any, registry: Any = None) -> str | None:
    """Return the configured or resolver-derived comparison time resolution."""
    configured = getattr(cfg.project, "tim_res", None)
    if configured:
        return str(configured)

    from openbench.config.resolver import derive_target_resolution_context

    return derive_target_resolution_context(cfg, registry).tim_res


def metric_score_errors(cfg: Any, effective_tim_res: str | None) -> list[str]:
    """Return only configuration-proven metric/score calculation failures."""
    from openbench.core.registry import IMPLEMENTED_METRICS, IMPLEMENTED_SCORES

    selected_metrics = tuple(dict.fromkeys(cfg.metrics if cfg.metrics is not None else _DEFAULT_METRICS))
    selected_scores = tuple(dict.fromkeys(cfg.scores if cfg.scores is not None else _DEFAULT_SCORES))
    selected = [
        name
        for name in (*selected_metrics, *selected_scores)
        if name in IMPLEMENTED_METRICS or name in IMPLEMENTED_SCORES
    ]

    first, last = cfg.project.years
    n_years = last - first + 1
    resolution = str(effective_tim_res or "").strip().lower()
    errors: list[str] = []

    if resolution in {"climatology-month", "climatology-year"}:
        from openbench.data.climatology import ClimatologyProcessor

        processor = ClimatologyProcessor()
        clim_type = "annual" if resolution == "climatology-year" else "monthly"
        samples = 1 if clim_type == "annual" else 12
        for name in selected:
            if not processor.is_metric_supported(name, clim_type=clim_type, time_points=samples):
                errors.append(
                    f"{name} cannot be calculated: effective time resolution {effective_tim_res!r} "
                    "is unsupported by the current climatology evaluation path; select a supported "
                    "metric/score or change project.tim_res."
                )
        return errors

    if n_years < 2 and "nIavScore" in selected_scores:
        errors.append(
            f"nIavScore cannot be calculated: project.years [{first}, {last}] contains one year; "
            "at least 2 years are required to estimate interannual variability."
        )

    if resolution in {"year", "yearly"}:
        for name in selected_scores:
            if name in {"nPhaseScore", "nSeasonalityScore"}:
                errors.append(
                    f"{name} cannot be calculated: Year resolution has no 12-month seasonal cycle; "
                    "use Month or Day data."
                )

        if n_years < 2:
            for name in selected_metrics:
                if name in _SINGLE_ANNUAL_SAMPLE_METRICS:
                    errors.append(
                        f"{name} cannot be calculated: project.years [{first}, {last}] at Year resolution "
                        "provides only one temporal sample; configure at least 2 years or use a finer "
                        "time resolution."
                    )
            for name in selected_scores:
                if name in _SINGLE_ANNUAL_SAMPLE_SCORES:
                    errors.append(
                        f"{name} cannot be calculated: project.years [{first}, {last}] at Year resolution "
                        "provides only one temporal sample; configure at least 2 years or use a finer "
                        "time resolution."
                    )

    return list(dict.fromkeys(errors))
