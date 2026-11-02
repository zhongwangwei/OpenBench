"""Metric and score applicability notes for openbench check.

This is a configuration-only advisory: a time span cannot establish the number
of finite aligned samples, nonzero denominators, or valid station/grid points.
Do not interpret a successful check as proof that every selected metric is
computable. The calculation functions remain the authority for those checks.
"""

from __future__ import annotations

# Keep these defaults consistent with the normal evaluation configuration.
_DEFAULT_METRICS = ("bias", "RMSE", "correlation")
_DEFAULT_SCORES = ("Overall_Score",)

# Names intentionally match core.registry. Notes describe the current
# implementation rather than a generic textbook variant of each metric.
METRIC_LIMITS = {
    "percent_bias": "Observed time sum must be nonzero; cancellation makes signed flux/anomaly totals unstable.",
    "absolute_percent_bias": "Observed time sum must be nonzero; near-zero totals amplify error.",
    "RMSE": "Requires finite aligned pairs; large errors receive extra weight.",
    "MSE": "Requires finite aligned pairs; squared units and outliers complicate comparison.",
    "NRMSE": "Requires nonzero absolute observed temporal mean; unstable near zero.",
    "RSR": "Requires non-constant observations and nonzero observed variance.",
    "RSS": "Requires finite aligned pairs; depends on sample count, unlike RMSE.",
    "NMAE": "Requires nonzero sum of absolute observed values.",
    "ubRMSE": "Requires temporal samples; with one pair, demeaning yields a trivial zero.",
    "CRMSD": "Requires temporal variability in both series for the correlation-based formula.",
    "mean_absolute_error": "Requires finite aligned pairs; retains the original variable units.",
    "bias": "Requires finite aligned pairs; positive and negative errors can cancel.",
    "L": "Requires non-constant observations and nonzero observed variance.",
    "correlation": "Requires >=2 finite aligned pairs and non-constant simulation and observation series.",
    "correlation_R2": "Same sample/variability requirements as correlation; the sign is lost by squaring.",
    "NSE": "Requires non-constant observations and nonzero observed variance.",
    "KGE": "Requires non-constant series, nonzero observed mean, and nonzero observed standard deviation.",
    "KGESS": "Inherits KGE's nonzero-mean, variance, and correlation requirements.",
    "index_agreement": "Requires a nonzero agreement denominator; a degenerate perfect-constant pair is undefined.",
    "rv": "Requires non-constant observations with nonzero standard deviation.",
    "ubNSE": "Requires non-constant observations with nonzero variance; uses demeaned series.",
    "ubcorrelation": "Requires varying series; demeaning does not change Pearson correlation.",
    "ubcorrelation_R2": "Requires varying series; demeaning does not change Pearson R-squared.",
    "pc_max": "Observed temporal maximum must be nonzero; extremes are sensitive to sample length.",
    "pc_min": "Observed temporal minimum must be nonzero; extremes are sensitive to sample length.",
    "pc_ampli": "Observed temporal max-minus-min range must be nonzero.",
    "rSD": "Requires non-constant observations and nonzero observed standard deviation.",
    "PBIAS_HF": "Uses observed upper 2% tail (Q98); needs enough high-flow pairs and nonzero tail sum.",
    "PBIAS_LF": "Uses observed lower 30% (Q30); selected observations must be strictly positive.",
    "pbiasfdc": "FDC quantiles must be positive, with a nonzero observed log-slope.",
    "rSpearman": "Requires >=2 finite pairs with varying ranks; many ties reduce information.",
    "MIA": "Requires nonzero modified-agreement denominator.",
    "RIA": "Requires strictly positive observed pairs and nonzero relative-agreement denominator.",
    "VE": "Requires nonnegative observed pairs and a positive observed total.",
    "LNSE": "Requires all finite aligned pairs to be strictly positive and log-observation variance >0.",
    "mNSE": "Requires non-constant observations and nonzero absolute-deviation sum.",
    "rNSE": "Requires nonzero observed pairs, nonzero observed mean and variance.",
    "mKGE": "Requires varying series and nonzero means for coefficient-of-variation ratios.",
    "KGEkm": "Requires varying series and nonzero means for CV and standard-deviation ratios.",
    "KGEln": "Requires varying series and strictly positive standard-deviation and mean ratios.",
    "KGElf": "Uses observed Q30 low-flow subset; that subset needs enough varied, valid pairs.",
    "APFB": "Requires valid dated annual flow peaks and nonzero observed peaks; complete years are preferable.",
    "br2": "Requires >=2 varying finite pairs for correlation and linear regression.",
    "cp": "Requires valid consecutive-time triples and nonzero observed first-difference energy.",
    "dr": "Requires nonzero observed absolute-deviation sum.",
    "MFM_omega": "Needs >=3 finite pairs and nonzero observed mean; FFT phase needs >=4 equally spaced samples.",
    "MFM_varphi": "Needs >=3 finite pairs; entropy depends on histogram binning and sample size.",
    "MFM_eta": "Needs >=3 finite pairs; histogram overlap depends on binning and sample size.",
    "MFM": "Needs >=3 finite pairs and nonzero observed mean; FFT phase needs >=4 equally spaced samples.",
}

SCORE_LIMITS = {
    "nBiasScore": "Requires non-constant observations and nonzero observed temporal standard deviation.",
    "nRMSEScore": "Requires non-constant observations and nonzero observed temporal standard deviation.",
    "nPhaseScore": "Requires all 12 calendar months and nonzero seasonal amplitude in both series.",
    "nIavScore": "Requires >=2 independent years with overlapping months; 3-5+ years recommended; observed IAV must be nonzero.",
    "nSpatialScore": "Requires >=2 valid spatial points/stations and nonzero spatial std in both fields.",
    "Overall_Score": "Combines bias, RMSE, phase, IAV and spatial scores; unavailable components are omitted and weights renormalized.",
    "nSeasonalityScore": "Requires all 12 calendar months and nonzero observed seasonal amplitude.",
}

# Explicit annual (one value per year) time series with a single year have
# only one temporal point. These names require >=2 meaningful time points.
_MULTI_TIME_METRICS = frozenset({
    "RSR", "ubRMSE", "CRMSD", "L", "correlation", "correlation_R2",
    "NSE", "KGE", "KGESS", "rv", "ubNSE", "ubcorrelation",
    "ubcorrelation_R2", "pc_ampli", "rSD", "rSpearman", "mNSE",
    "rNSE", "mKGE", "KGEkm", "KGEln", "KGElf", "br2", "cp",
    "LNSE", "MFM_omega", "MFM_varphi", "MFM_eta", "MFM",
})
_MULTI_TIME_SCORES = frozenset({
    "nBiasScore", "nRMSEScore", "nPhaseScore", "nIavScore",
    "nSeasonalityScore", "Overall_Score",
})


def metric_score_findings(cfg) -> tuple[list[str], list[str]]:
    """Return warnings and per-selected-item notes without inspecting data.

    Static findings deliberately do not fail openbench check: this is a
    first-stage advisory, not a runtime feasibility guarantee.
    """
    from openbench.core.registry import IMPLEMENTED_METRICS, IMPLEMENTED_SCORES

    selected_metrics = tuple(dict.fromkeys(cfg.metrics if cfg.metrics is not None else _DEFAULT_METRICS))
    selected_scores = tuple(dict.fromkeys(cfg.scores if cfg.scores is not None else _DEFAULT_SCORES))
    selected = [name for name in (*selected_metrics, *selected_scores)
                if name in IMPLEMENTED_METRICS or name in IMPLEMENTED_SCORES]

    notes = [f"{name}: {METRIC_LIMITS.get(name) or SCORE_LIMITS.get(name)}" for name in selected]
    warnings: list[str] = []
    first, last = cfg.project.years
    n_years = last - first + 1
    resolution = str(getattr(cfg.project, "tim_res", None) or "").strip().lower()

    if n_years < 2:
        if "nIavScore" in selected_scores:
            warnings.append(
                f"nIavScore: project.years [{first}, {last}] contain only one year; "
                "interannual variability needs >=2 independent years (3-5+ recommended)."
            )
        if "Overall_Score" in selected_scores:
            warnings.append(
                "Overall_Score: single-year evaluation cannot reliably estimate the IAV component; "
                "the effective components/weights may differ between evaluations."
            )
    elif n_years < 3 and "nIavScore" in selected_scores:
        warnings.append("nIavScore: two years are the minimum, not a robust IAV estimate (3-5+ recommended).")

    if resolution in {"year", "yearly"}:
        for name in selected_scores:
            if name in {"nPhaseScore", "nSeasonalityScore"}:
                warnings.append(f"{name}: yearly resolution has no 12-month seasonal cycle; use Month/Day data.")
            elif name == "Overall_Score":
                warnings.append("Overall_Score: yearly resolution has no monthly phase/seasonality cycle.")
        if n_years < 2:
            for name in (*selected_metrics, *selected_scores):
                if name in _MULTI_TIME_METRICS or name in _MULTI_TIME_SCORES:
                    warnings.append(f"{name}: a single Year-resolution sample cannot establish temporal variability.")

    if resolution in {"climatology-month", "climatology-year"}:
        from openbench.data.climatology import ClimatologyProcessor

        processor = ClimatologyProcessor()
        clim_type = "annual" if resolution == "climatology-year" else "monthly"
        samples = 1 if clim_type == "annual" else 12
        for name in selected:
            if not processor.is_metric_supported(name, clim_type=clim_type, time_points=samples):
                warnings.append(f"{name}: {resolution} is unsupported by the current climatology evaluation path.")
        if resolution == "climatology-month" and "Overall_Score" in selected_scores:
            warnings.append(
                "Overall_Score: monthly climatology has one representative annual cycle; "
                "its IAV component is unavailable and the composite may be reweighted."
            )

    if resolution == "day" and ("nIavScore" in selected_scores or "Overall_Score" in selected_scores):
        warnings.append(
            "nIavScore/Overall_Score: the current IAV formula groups raw daily samples by month; "
            "within-month daily variability can affect the reported IAV."
        )

    if not resolution:
        notes.insert(0, "project.tim_res is unset; exact target-resolution compatibility must be verified after resolution.")
    notes.append("The paired-data checks (finite samples, zero denominators, spatial spread) are not performed here.")
    return list(dict.fromkeys(warnings)), notes
