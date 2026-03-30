# -*- coding: utf-8 -*-
"""
Statistics selection page (conditional).
"""

from openbench.gui.pages.base_page import BasePage
from openbench.gui.widgets import CheckboxGroup


STATISTICS_ITEMS = {
    "Basic Statistics": [
        "Mean", "Median", "Min", "Max", "Sum", "Standard_Deviation"
    ],
    "Advanced Statistics": [
        "Mann_Kendall_Trend_Test", "Correlation", "Functional_Response",
        "Z_Score", "Hellinger_Distance", "Three_Cornered_Hat",
        "Partial_Least_Squares_Regression", "ANOVA"
    ],
}


class PageStatistics(BasePage):
    """Statistics selection page."""

    PAGE_ID = "statistics"
    PAGE_TITLE = "Statistics"
    PAGE_SUBTITLE = "Select statistical analyses"
    CONTENT_EXPAND = True  # Allow content to fill available space

    def _setup_content(self):
        """Setup page content."""
        self.checkbox_group = CheckboxGroup(STATISTICS_ITEMS)
        self.checkbox_group.selection_changed.connect(self._on_selection_changed)
        self.content_layout.addWidget(self.checkbox_group)

    def _on_selection_changed(self, selection):
        """Handle selection changes."""
        self.save_to_config()

    def load_from_config(self):
        """Load from config."""
        statistics = self.controller.config.get("statistics", {})
        self.checkbox_group.set_selection(statistics)

    def save_to_config(self):
        """Save to config."""
        selection = self.checkbox_group.get_selection()
        self.controller.update_section("statistics", selection)
