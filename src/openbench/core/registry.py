"""Shared registry for implemented evaluation metric and score names."""

from __future__ import annotations

from collections.abc import Iterable

from openbench.core.metrics import metrics
from openbench.core.scores import scores

# Public helpers / known placeholders that should not be advertised as user
# selectable evaluation products.  Keep this exclusion in core so GUI, CLI, and
# visualization code share one source of truth instead of each carrying its own
# stale hard-coded copy.
_METRIC_EXCLUDE = {
    "rm_mean",
    "rSD",
    "PBIAS_HF",
    "PBIAS_LF",
    # SMPI returns (estimate, lower, upper) and belongs to the dedicated
    # comparison workflow, not the single-DataArray evaluation contract.
    "smpi",
    # Unsafe as general continuous-field metrics: ubKGE can degenerate after
    # mean removal, while kappa_coeff only accepts categorical labels. Keep the
    # methods importable for compatibility but do not expose them through
    # GUI/CLI selectable metric registries.
    "ubKGE",
    "kappa_coeff",
}

_SCORE_EXCLUDE = {"index_agreement"}


def _public_callable_names(cls: type, *, exclude: Iterable[str] = ()) -> list[str]:
    excluded = set(exclude)
    return [
        name
        for name, member in cls.__dict__.items()
        if callable(member) and not name.startswith("_") and name not in excluded
    ]


IMPLEMENTED_METRIC_NAMES = tuple(_public_callable_names(metrics, exclude=_METRIC_EXCLUDE))
IMPLEMENTED_SCORE_NAMES = tuple(_public_callable_names(scores, exclude=_SCORE_EXCLUDE))

IMPLEMENTED_METRICS = set(IMPLEMENTED_METRIC_NAMES)
IMPLEMENTED_SCORES = set(IMPLEMENTED_SCORE_NAMES)

METRIC_LABELS = {
    "percent_bias": "Percent Bias (PBIAS)",
    "absolute_percent_bias": "Absolute Percent Bias (APB)",
    "RMSE": "Root Mean Squared Error (RMSE)",
    "ubRMSE": "Unbiased Root Mean Squared Error (ubRMSE)",
    "CRMSD": "Centered Root Mean Square Difference (CRMSD)",
    "mean_absolute_error": "Mean Absolute Error (MAE)",
    "bias": "Bias (BIAS)",
    "L": "Likelihood (L)",
    "correlation": "Correlation Coefficient (r)",
    "correlation_R2": "Coefficient of Determination (R²)",
    "NSE": "Nash–Sutcliffe Efficiency (NSE)",
    "KGE": "Kling–Gupta Efficiency (KGE)",
    "KGESS": "Kling–Gupta Efficiency Skill Score (KGESS)",
    "rv": "Relative Variability (RV)",
    "ubNSE": "Unbiased Nash–Sutcliffe Efficiency (ubNSE)",
    "ubcorrelation": "Unbiased Correlation Coefficient (ubr)",
    "ubcorrelation_R2": "Unbiased Coefficient of Determination (ubR²)",
    "pc_max": "Relative Maximum Deviation (PCmax)",
    "pc_min": "Relative Minimum Deviation (PCmin)",
    "pc_ampli": "Relative Amplitude Deviation (PCamp)",
    "APFB": "Annual Peak Flow Bias (APFB)",
    "br2": "Slope-adjusted Coefficient of Determination (br²)",
    "cp": "Coefficient of Persistence (CP)",
    "dr": "Refined Index of Agreement (dr)",
    "MFM_omega": "Model Fidelity Metric Phase Component (MFM-ω)",
    "MFM_varphi": "Model Fidelity Metric Variability Component (MFM-φ)",
    "MFM_eta": "Model Fidelity Metric Distribution Component (MFM-η)",
    "MFM": "Model Fidelity Metric (MFM)",
    "index_agreement": "Index of Agreement (IOA)",
}

SCORE_LABELS = {
    "nBiasScore": "Normalized Bias Score (nBiasScore)",
    "nRMSEScore": "Normalized RMSE Score (nRMSEScore)",
    "nPhaseScore": "Normalized Phase Score (nPhaseScore)",
    "nIavScore": "Normalized Interannual Variability Score (nIAVScore)",
    "nSpatialScore": "Normalized Spatial Score (nSpatialScore)",
    "Overall_Score": "Overall Score (OS)",
    "nSeasonalityScore": "Normalized Seasonality Score (nSeasonalityScore)",
}


def _filtered(items: Iterable[str], valid: set[str]) -> list[str]:
    return [item for item in items if item in valid]


METRICS_ITEMS = {
    "Basic Metrics": _filtered(
        [
            "bias",
            "percent_bias",
            "absolute_percent_bias",
            "mean_absolute_error",
            "RMSE",
            "ubRMSE",
            "CRMSD",
        ],
        IMPLEMENTED_METRICS,
    ),
    "Correlation": _filtered(
        [
            "correlation",
            "correlation_R2",
            "ubcorrelation",
            "ubcorrelation_R2",
        ],
        IMPLEMENTED_METRICS,
    ),
    "Efficiency": _filtered(
        [
            "NSE",
            "KGE",
            "KGESS",
            "ubNSE",
            "L",
            "index_agreement",
        ],
        IMPLEMENTED_METRICS,
    ),
    "Hydrology": _filtered(["br2", "cp", "dr", "APFB"], IMPLEMENTED_METRICS),
    "Other": [],
}

_CATEGORIZED_METRICS = {item for values in METRICS_ITEMS.values() for item in values}
METRICS_ITEMS["Other"] = [item for item in IMPLEMENTED_METRIC_NAMES if item not in _CATEGORIZED_METRICS]

SCORES_ITEMS = {
    "ILAMB Scoring System": _filtered(
        [
            "nBiasScore",
            "nRMSEScore",
            "nPhaseScore",
            "nIavScore",
            "nSpatialScore",
            "Overall_Score",
        ],
        IMPLEMENTED_SCORES,
    ),
    "Other": [],
}

_CATEGORIZED_SCORES = {item for values in SCORES_ITEMS.values() for item in values}
SCORES_ITEMS["Other"] = [item for item in IMPLEMENTED_SCORE_NAMES if item not in _CATEGORIZED_SCORES]
