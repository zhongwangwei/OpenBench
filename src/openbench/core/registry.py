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
    "index_agreement",
    # Unsafe as general continuous-field metrics: ubKGE can degenerate after
    # mean removal, and kappa_coeff silently bins continuous values via int
    # casting.  Keep the methods importable for compatibility but do not expose
    # them through GUI/CLI selectable metric registries.
    "ubKGE",
    "kappa_coeff",
}


def _public_callable_names(cls: type, *, exclude: Iterable[str] = ()) -> list[str]:
    excluded = set(exclude)
    return [
        name
        for name, member in cls.__dict__.items()
        if callable(member) and not name.startswith("_") and name not in excluded
    ]


IMPLEMENTED_METRIC_NAMES = tuple(_public_callable_names(metrics, exclude=_METRIC_EXCLUDE))
IMPLEMENTED_SCORE_NAMES = tuple(_public_callable_names(scores))

IMPLEMENTED_METRICS = set(IMPLEMENTED_METRIC_NAMES)
IMPLEMENTED_SCORES = set(IMPLEMENTED_SCORE_NAMES)


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
