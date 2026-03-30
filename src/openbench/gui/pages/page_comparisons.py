# -*- coding: utf-8 -*-
"""
Comparisons selection page (conditional).
"""

from openbench.gui.pages.base_page import BasePage
from openbench.gui.widgets import CheckboxGroup


COMPARISON_ITEMS = {
    "Diagrams": [
        "Taylor_Diagram", "Target_Diagram", "Whisker_Plot",
        "Parallel_Coordinates", "Portrait_Plot_seasonal",
        "Ridgeline_Plot"
    ],
    "Plots": [
        "HeatMap", "Kernel_Density_Estimate", "Diff_Plot", "RadarMap"
    ],
    "Statistics": [
        "Single_Model_Performance_Index", "Relative_Score",
        "Mann_Kendall_Trend_Test", "Correlation", "Standard_Deviation",
        "Functional_Response"
    ],
    "Aggregation": [
        "Mean", "Median", "Min", "Max", "Sum"
    ],
}


class PageComparisons(BasePage):
    """Comparisons selection page."""

    PAGE_ID = "comparisons"
    PAGE_TITLE = "Comparisons"
    PAGE_SUBTITLE = "Select comparison visualizations and analyses"
    CONTENT_EXPAND = True  # Allow content to fill available space

    def _setup_content(self):
        """Setup page content."""
        self.checkbox_group = CheckboxGroup(COMPARISON_ITEMS)
        self.checkbox_group.selection_changed.connect(self._on_selection_changed)
        self.content_layout.addWidget(self.checkbox_group)

    def _on_selection_changed(self, selection):
        """Handle selection changes."""
        self.save_to_config()

    def load_from_config(self):
        """Load from config."""
        comparisons = self.controller.config.get("comparisons", {})
        self.checkbox_group.set_selection(comparisons)

    def save_to_config(self):
        """Save to config."""
        selection = self.checkbox_group.get_selection()
        self.controller.update_section("comparisons", selection)
