# -*- coding: utf-8 -*-
"""
Evaluation Items selection page.
"""

from openbench.gui.pages.base_page import BasePage
from openbench.gui.widgets import CheckboxGroup


# Evaluation items grouped by category
EVALUATION_ITEMS = {
    "Carbon Cycle": [
        "Biomass",
        "Ecosystem_Respiration",
        "Gross_Primary_Productivity",
        "Leaf_Area_Index",
        "Methane",
        "Net_Ecosystem_Exchange",
        "Nitrogen_Fixation",
        "Soil_Carbon",
    ],
    "Water Cycle": [
        "Canopy_Interception",
        "Canopy_Transpiration",
        "Evapotranspiration",
        "Permafrost",
        "Root_Zone_Soil_Moisture",
        "Snow_Depth",
        "Snow_Water_Equivalent",
        "Soil_Evaporation",
        "Surface_Snow_Cover_In_Fraction",
        "Surface_Soil_Moisture",
        "Terrestrial_Water_Storage_Change",
        "Total_Runoff",
        "Water_Evaporation",
    ],
    "Energy Cycle": [
        "Surface_Albedo",
        "Ground_Heat",
        "Latent_Heat",
        "Net_Radiation",
        "Root_Zone_Soil_Temperature",
        "Sensible_Heat",
        "Surface_Net_LW_Radiation",
        "Surface_Net_SW_Radiation",
        "Surface_Soil_Temperature",
        "Surface_Upward_LW_Radiation",
        "Surface_Upward_SW_Radiation",
    ],
    "Atmospheric": [
        "Diurnal_Max_Temperature",
        "Diurnal_Min_Temperature",
        "Diurnal_Temperature_Range",
        "Precipitation",
        "Surface_Air_Temperature",
        "Surface_Downward_LW_Radiation",
        "Surface_Downward_SW_Radiation",
        "Surface_Relative_Humidity",
        "Surface_Specific_Humidity",
    ],
    "Agriculture": [
        "Crop_Emergence_DOY_Wheat",
        "Crop_Heading_DOY_Corn",
        "Crop_Heading_DOY_Wheat",
        "Crop_Maturity_DOY_Corn",
        "Crop_Maturity_DOY_Wheat",
        "Crop_V3_DOY_Corn",
        "Crop_Yield_Corn",
        "Crop_Yield_Maize",
        "Crop_Yield_Rice",
        "Crop_Yield_Soybean",
        "Crop_Yield_Wheat",
        "Total_Irrigation_Amount",
    ],
    "Water Bodies": [
        "Dam_Inflow",
        "Dam_Outflow",
        "Dam_Water_Elevation",
        "Dam_Water_Storage",
        "Inundation_Area",
        "Inundation_Fraction",
        "Lake_Ice_Fraction_Cover",
        "Lake_Temperature",
        "Lake_Water_Area",
        "Lake_Water_Level",
        "Lake_Water_Volume",
        "River_Water_Level",
        "Streamflow",
    ],
    "Urban": [
        "Urban_Air_Temperature_Max",
        "Urban_Air_Temperature_Min",
        "Urban_Albedo",
        "Urban_Anthropogenic_Heat_Flux",
        "Urban_Latent_Heat_Flux",
        "Urban_Surface_Temperature",
    ],
}


class PageEvaluation(BasePage):
    """Evaluation Items selection page."""

    PAGE_ID = "evaluation_items"
    PAGE_TITLE = "Evaluation Items"
    PAGE_SUBTITLE = "Select the variables to evaluate"
    CONTENT_EXPAND = True  # Allow content to fill available space

    def _setup_content(self):
        """Setup page content."""
        self.checkbox_group = CheckboxGroup(EVALUATION_ITEMS)
        self.checkbox_group.selection_changed.connect(self._on_selection_changed)
        self.content_layout.addWidget(self.checkbox_group)

    def _on_selection_changed(self, _selection):
        """Handle selection changes."""
        self.save_to_config()

    def load_from_config(self):
        """Load from config."""
        eval_items = self.controller.config.get("evaluation_items", {})
        self.checkbox_group.set_selection(eval_items)

    def save_to_config(self):
        """Save to config."""
        selection = self.checkbox_group.get_selection()
        self.controller.update_section("evaluation_items", selection)

        # Trigger namelist sync (evaluation items affect filtering)
        self.controller.sync_namelists()

    def validate(self) -> bool:
        """Validate page input."""
        from openbench.gui.validation import FieldValidator, ValidationManager

        selection = self.checkbox_group.get_selection()
        error = FieldValidator.selection_required(
            selection, "evaluation_items", "Please select at least one evaluation item", page_id=self.PAGE_ID
        )

        if error:
            manager = ValidationManager(self)
            if not manager.show_error_and_focus(error):
                return False

        self.save_to_config()
        return True
