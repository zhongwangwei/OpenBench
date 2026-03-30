# -*- coding: utf-8 -*-
"""
Options page — metrics, scores, comparisons, and statistics in tabs.
Merges the old PageMetrics, PageScores, PageComparisons, PageStatistics.
"""

from PySide6.QtWidgets import QTabWidget, QWidget, QVBoxLayout, QCheckBox

from openbench.gui.pages.base_page import BasePage
from openbench.gui.widgets import CheckboxGroup


METRICS_ITEMS = {
    "Basic Metrics": [
        "bias", "percent_bias", "absolute_percent_bias",
        "mean_absolute_error", "RMSE", "MSE", "ubRMSE", "CRMSD", "nrmse",
    ],
    "Correlation": [
        "correlation", "correlation_R2", "rSpearman",
        "ubcorrelation", "ubcorrelation_R2",
    ],
    "Efficiency": [
        "NSE", "LNSE", "KGE", "KGESS", "ubNSE", "ubKGE",
        "mNSE", "rNSE", "wNSE", "wsNSE", "sKGE",
        "KGEkm", "KGElf", "KGEnp",
    ],
    "Other": [
        "L", "kappa_coeff", "rv", "pc_max", "pc_min", "pc_ampli",
        "rSD", "PBIAS_HF", "PBIAS_LF", "SMPI", "ggof", "gof", "md",
        "pbiasfdc", "pfactor", "rd", "rfactor", "rsr", "ssq",
        "valindex", "ve", "index_agreement", "MFM",
    ],
}

SCORES_ITEMS = {
    "ILAMB Scoring System": [
        "nBiasScore", "nRMSEScore", "nPhaseScore",
        "nIavScore", "nSpatialScore", "Overall_Score",
    ],
    "Other": ["The_Ideal_Point_score"],
}

COMPARISON_ITEMS = {
    "Diagrams": [
        "Taylor_Diagram",
        "Target_Diagram",
        "Whisker_Plot",
        "Parallel_Coordinates",
        "Portrait_Plot_seasonal",
        "Ridgeline_Plot",
    ],
    "Plots": ["HeatMap", "Kernel_Density_Estimate", "Diff_Plot", "RadarMap"],
    "Statistics": [
        "Single_Model_Performance_Index",
        "Relative_Score",
        "Mann_Kendall_Trend_Test",
        "Correlation",
        "Standard_Deviation",
        "Functional_Response",
    ],
    "Aggregation": ["Mean", "Median", "Min", "Max", "Sum"],
}

STATISTICS_ITEMS = {
    "Basic Statistics": ["Mean", "Median", "Min", "Max", "Sum", "Standard_Deviation"],
    "Advanced Statistics": [
        "Mann_Kendall_Trend_Test",
        "Correlation",
        "Functional_Response",
        "Z_Score",
        "Hellinger_Distance",
        "Three_Cornered_Hat",
        "Partial_Least_Squares_Regression",
        "ANOVA",
    ],
}


class PageOptions(BasePage):
    """Options page with tabbed metrics, scores, comparisons, statistics."""

    PAGE_ID = "options"
    PAGE_TITLE = "Options"
    PAGE_SUBTITLE = "Configure metrics, scores, comparisons, and statistical analyses"
    CONTENT_EXPAND = True

    def _setup_content(self):
        """Setup tabbed content."""
        self.tabs = QTabWidget()

        # Metrics tab
        self.metrics_group = CheckboxGroup(METRICS_ITEMS)
        self.metrics_group.selection_changed.connect(self._on_change)
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout()
        metrics_layout.addWidget(self.metrics_group)
        metrics_widget.setLayout(metrics_layout)
        self.tabs.addTab(metrics_widget, "Metrics")

        # Scores tab
        self.scores_group = CheckboxGroup(SCORES_ITEMS)
        self.scores_group.selection_changed.connect(self._on_change)
        scores_widget = QWidget()
        scores_layout = QVBoxLayout()
        scores_layout.addWidget(self.scores_group)
        scores_widget.setLayout(scores_layout)
        self.tabs.addTab(scores_widget, "Scores")

        # Comparisons tab with enable toggle
        comp_widget = QWidget()
        comp_layout = QVBoxLayout()
        self.comparison_enabled = QCheckBox("Enable comparison visualizations")
        self.comparison_enabled.stateChanged.connect(self._on_comparison_toggled)
        comp_layout.addWidget(self.comparison_enabled)
        self.comparisons_group = CheckboxGroup(COMPARISON_ITEMS)
        self.comparisons_group.selection_changed.connect(self._on_change)
        self.comparisons_group.setEnabled(False)
        comp_layout.addWidget(self.comparisons_group)
        comp_widget.setLayout(comp_layout)
        self.tabs.addTab(comp_widget, "Comparisons")

        # Statistics tab with enable toggle
        stats_widget = QWidget()
        stats_layout = QVBoxLayout()
        self.statistics_enabled = QCheckBox("Enable statistical analyses")
        self.statistics_enabled.stateChanged.connect(self._on_statistics_toggled)
        stats_layout.addWidget(self.statistics_enabled)
        self.statistics_group = CheckboxGroup(STATISTICS_ITEMS)
        self.statistics_group.selection_changed.connect(self._on_change)
        self.statistics_group.setEnabled(False)
        stats_layout.addWidget(self.statistics_group)
        stats_widget.setLayout(stats_layout)
        self.tabs.addTab(stats_widget, "Statistics")

        self.content_layout.addWidget(self.tabs)

    def _on_change(self, _=None):
        self.save_to_config()

    def _on_comparison_toggled(self):
        self.comparisons_group.setEnabled(self.comparison_enabled.isChecked())
        self.save_to_config()

    def _on_statistics_toggled(self):
        self.statistics_group.setEnabled(self.statistics_enabled.isChecked())
        self.save_to_config()

    def load_from_config(self):
        """Load from config."""
        config = self.controller.config
        self.metrics_group.set_selection(config.get("metrics", {}))
        self.scores_group.set_selection(config.get("scores", {}))

        # Comparison
        general = config.get("general", {})
        comp_enabled = general.get("comparison", False)
        self.comparison_enabled.setChecked(comp_enabled)
        self.comparisons_group.setEnabled(comp_enabled)
        self.comparisons_group.set_selection(config.get("comparisons", {}))

        # Statistics
        stats_enabled = general.get("statistics", False)
        self.statistics_enabled.setChecked(stats_enabled)
        self.statistics_group.setEnabled(stats_enabled)
        self.statistics_group.set_selection(config.get("statistics", {}))

    def save_to_config(self):
        """Save all tabs to config."""
        self.controller.update_section("metrics", self.metrics_group.get_selection())
        self.controller.update_section("scores", self.scores_group.get_selection())
        self.controller.update_section("comparisons", self.comparisons_group.get_selection())
        self.controller.update_section("statistics", self.statistics_group.get_selection())

        # Update general flags
        general = self.controller.config.get("general", {})
        general["comparison"] = self.comparison_enabled.isChecked()
        general["statistics"] = self.statistics_enabled.isChecked()
        self.controller.update_section("general", general)

    def validate(self) -> bool:
        """Check at least one metric or score is selected."""
        self.save_to_config()
        metrics = self.controller.config.get("metrics", {})
        scores = self.controller.config.get("scores", {})
        selected = [k for k, v in {**metrics, **scores}.items() if v]
        if not selected:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Validation", "Please select at least one metric or score.")
            return False
        return True
