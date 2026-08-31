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
    # These methods are implemented for programmatic use, but the current
    # evaluation config cannot provide their extra inputs or scalarize their
    # vector output.
    "valindex",
    "wNSE",
    "wsNSE",
    "sKGE",
    # The appendix names quantile-based KGEnp components without fixing their
    # formulas. Keep the canonical Pool et al. implementation importable, but
    # do not present one scientific convention as the guide's only definition.
    "KGEnp",
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
    "MSE": "Mean Squared Error (MSE)",
    "RMSE": "Root Mean Squared Error (RMSE)",
    "NRMSE": "Normalized Root Mean Squared Error (NRMSE)",
    "RSR": "RMSE–Observation Standard Deviation Ratio (RSR)",
    "RSS": "Residual Sum of Squares (RSS)",
    "NMAE": "Normalized Mean Absolute Error (NMAE)",
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
    "rSD": "Ratio of Standard Deviations (rSD)",
    "ubNSE": "Unbiased Nash–Sutcliffe Efficiency (ubNSE)",
    "ubcorrelation": "Unbiased Correlation Coefficient (ubr)",
    "ubcorrelation_R2": "Unbiased Coefficient of Determination (ubR²)",
    "pc_max": "Relative Maximum Deviation (PCmax)",
    "pc_min": "Relative Minimum Deviation (PCmin)",
    "pc_ampli": "Relative Amplitude Deviation (PCamp)",
    "APFB": "Annual Peak Flow Bias (APFB)",
    "PBIAS_HF": "Percent Bias of High Flows (PBIAS-HF)",
    "PBIAS_LF": "Percent Bias of Low Flows (PBIAS-LF)",
    "pbiasfdc": "Flow Duration Curve Midsegment Slope Bias (PBIAS-FDC)",
    "br2": "Slope-adjusted Coefficient of Determination (br²)",
    "cp": "Coefficient of Persistence (CP)",
    "rSpearman": "Spearman Rank Correlation Coefficient (rSpearman)",
    "MIA": "Modified Index of Agreement (MIA)",
    "RIA": "Relative Index of Agreement (RIA)",
    "dr": "Refined Index of Agreement (dr)",
    "VE": "Volumetric Efficiency (VE)",
    "LNSE": "Log Nash–Sutcliffe Efficiency (LNSE)",
    "mNSE": "Modified Nash–Sutcliffe Efficiency (mNSE)",
    "rNSE": "Relative Nash–Sutcliffe Efficiency (rNSE)",
    "mKGE": "Modified Kling–Gupta Efficiency (mKGE)",
    "KGEkm": "Known-moments Kling–Gupta Efficiency (KGEkm)",
    "KGElf": "Low-flow Kling–Gupta Efficiency (KGElf)",
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
            "MSE",
            "RMSE",
            "NRMSE",
            "RSR",
            "RSS",
            "NMAE",
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
            "rSpearman",
        ],
        IMPLEMENTED_METRICS,
    ),
    "Efficiency": _filtered(
        [
            "NSE",
            "LNSE",
            "mNSE",
            "rNSE",
            "KGE",
            "KGESS",
            "mKGE",
            "KGEkm",
            "KGElf",
            "ubNSE",
            "L",
        ],
        IMPLEMENTED_METRICS,
    ),
    "Agreement": _filtered(["index_agreement", "MIA", "RIA", "dr"], IMPLEMENTED_METRICS),
    "Hydrology": _filtered(
        ["PBIAS_HF", "PBIAS_LF", "pbiasfdc", "APFB", "VE", "br2", "cp"],
        IMPLEMENTED_METRICS,
    ),
    "Variability": _filtered(["rv", "rSD", "pc_max", "pc_min", "pc_ampli"], IMPLEMENTED_METRICS),
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
