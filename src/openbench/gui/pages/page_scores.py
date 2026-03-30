# -*- coding: utf-8 -*-
"""
Scores selection page.
"""

from openbench.gui.pages.base_page import BasePage
from openbench.gui.widgets import CheckboxGroup


SCORES_ITEMS = {
    "ILAMB Scoring System": ["nBiasScore", "nRMSEScore", "nPhaseScore", "nIavScore", "nSpatialScore", "Overall_Score"],
    "Other": ["The_Ideal_Point_score"],
}


class PageScores(BasePage):
    """Scores selection page."""

    PAGE_ID = "scores"
    PAGE_TITLE = "Scores"
    PAGE_SUBTITLE = "Select scoring methods"
    CONTENT_EXPAND = True  # Allow content to fill available space

    def _setup_content(self):
        """Setup page content."""
        self.checkbox_group = CheckboxGroup(SCORES_ITEMS)
        self.checkbox_group.selection_changed.connect(self._on_selection_changed)
        self.content_layout.addWidget(self.checkbox_group)

    def _on_selection_changed(self, selection):
        """Handle selection changes."""
        self.save_to_config()

    def load_from_config(self):
        """Load from config."""
        scores = self.controller.config.get("scores", {})
        self.checkbox_group.set_selection(scores)

    def save_to_config(self):
        """Save to config."""
        selection = self.checkbox_group.get_selection()
        self.controller.update_section("scores", selection)

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
