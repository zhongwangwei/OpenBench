# -*- coding: utf-8 -*-
"""
Metrics selection page.
"""

from openbench.gui.pages.base_page import BasePage
from openbench.gui.widgets import CheckboxGroup
from openbench.core.registry import METRICS_ITEMS


class PageMetrics(BasePage):
    """Metrics selection page."""

    PAGE_ID = "metrics"
    PAGE_TITLE = "Metrics"
    PAGE_SUBTITLE = "Select evaluation metrics"
    CONTENT_EXPAND = True  # Allow content to fill available space

    def _setup_content(self):
        """Setup page content."""
        self.checkbox_group = CheckboxGroup(METRICS_ITEMS)
        self.checkbox_group.selection_changed.connect(self._on_selection_changed)
        self.content_layout.addWidget(self.checkbox_group)

    def _on_selection_changed(self, selection):
        """Handle selection changes."""
        self.save_to_config()

    def load_from_config(self):
        """Load from config."""
        metrics = self.controller.config.get("metrics", {})
        self.checkbox_group.set_selection(metrics)

    def save_to_config(self):
        """Save to config."""
        selection = self.checkbox_group.get_selection()
        self.controller.update_section("metrics", selection)

    def validate(self) -> bool:
        """Validate page input - check combined metrics + scores selection."""
        from openbench.gui.validation import FieldValidator, ValidationManager

        # Save current selection first
        self.save_to_config()

        # Get combined selection
        combined = self.controller.get_combined_metrics_scores_selection()

        error = FieldValidator.selection_required(
            combined, "metrics_scores", "Please select at least one metric or score", page_id=self.PAGE_ID
        )

        if error:
            manager = ValidationManager(self)
            if not manager.show_error_and_focus(error):
                return False

        return True
