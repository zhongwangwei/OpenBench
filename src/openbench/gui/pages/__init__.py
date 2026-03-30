# -*- coding: utf-8 -*-
"""Pages package."""

from openbench.gui.pages.base_page import BasePage
from openbench.gui.pages.page_runtime import PageRuntime
from openbench.gui.pages.page_general import PageGeneral
from openbench.gui.pages.page_evaluation import PageEvaluation
from openbench.gui.pages.page_metrics import PageMetrics
from openbench.gui.pages.page_scores import PageScores
from openbench.gui.pages.page_comparisons import PageComparisons
from openbench.gui.pages.page_statistics import PageStatistics
from openbench.gui.pages.page_ref_data import PageRefData
from openbench.gui.pages.page_sim_data import PageSimData
from openbench.gui.pages.page_variables import PageVariables
from openbench.gui.pages.page_options import PageOptions
from openbench.gui.pages.page_preview import PagePreview
from openbench.gui.pages.page_run_monitor import PageRunMonitor

__all__ = [
    "BasePage",
    "PageRuntime",
    "PageGeneral",
    "PageEvaluation",
    "PageMetrics",
    "PageScores",
    "PageComparisons",
    "PageStatistics",
    "PageRefData",
    "PageSimData",
    "PageVariables",
    "PageOptions",
    "PagePreview",
    "PageRunMonitor",
]
