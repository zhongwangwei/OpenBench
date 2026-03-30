# -*- coding: utf-8 -*-
"""
Variables & Reference Data page — select evaluation variables and their reference sources.
Merges the old PageEvaluation and PageRefData into one unified view.
"""

from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QWidget, QLabel, QComboBox,
    QCheckBox, QGroupBox, QScrollArea, QFrame, QPushButton,
)
from PySide6.QtCore import Qt

from openbench.gui.pages.base_page import BasePage


# Evaluation items grouped by category
EVALUATION_ITEMS = {
    "Carbon Cycle": [
        "Biomass", "Ecosystem_Respiration", "Gross_Primary_Productivity",
        "Leaf_Area_Index", "Methane", "Net_Ecosystem_Exchange",
        "Nitrogen_Fixation", "Soil_Carbon", "Net_Primary_Production",
        "Burned_Area", "Veg_Cover_In_Fraction", "Leaf_Greenness",
        "Wetland_Methane_Emission",
    ],
    "Water Cycle": [
        "Canopy_Interception", "Canopy_Transpiration", "Evapotranspiration",
        "Permafrost", "Root_Zone_Soil_Moisture", "Snow_Depth",
        "Snow_Water_Equivalent", "Soil_Evaporation",
        "Surface_Snow_Cover_In_Fraction", "Surface_Soil_Moisture",
        "Terrestrial_Water_Storage_Change", "Total_Runoff",
        "Water_Evaporation", "Bare_Soil_Evaporation",
        "Open_Water_Evaporation", "Precipitation", "Runoff",
        "Transpiration", "Inundation_Area", "Inundation_Fraction",
        "Groundwater_Recharge_Rate", "Water_Table_Depth",
        "Depth_Of_Surface_Water",
    ],
    "Energy Cycle": [
        "Surface_Albedo", "Ground_Heat", "Latent_Heat", "Net_Radiation",
        "Root_Zone_Soil_Temperature", "Sensible_Heat",
        "Surface_Net_LW_Radiation", "Surface_Net_SW_Radiation",
        "Surface_Soil_Temperature", "Surface_Upward_LW_Radiation",
        "Surface_Upward_SW_Radiation", "Surface_Downward_LW_Radiation",
        "Surface_Downward_SW_Radiation",
    ],
    "Meteorology": [
        "Diurnal_Max_Temperature", "Diurnal_Min_Temperature",
        "Diurnal_Temperature_Range", "Surface_Air_Temperature",
        "Surface_Relative_Humidity", "Surface_Specific_Humidity",
        "Surface_Wind_Speed", "Cloud_Cover", "Ground_Frost_Frequency",
    ],
    "Crop": [
        "Crop_Yield_Corn", "Crop_Yield_Maize", "Crop_Yield_Rice",
        "Crop_Yield_Soybean", "Crop_Yield_Wheat",
        "Total_Irrigation_Amount",
    ],
    "Hydrology": [
        "Streamflow", "Dam_Inflow", "Dam_Outflow", "Dam_Water_Storage",
        "Lake_Temperature", "Lake_Water_Level",
        "River_Water_Level",
    ],
}


class PageVariables(BasePage):
    """Variables & Reference selection page.

    Each variable has a checkbox (enable/disable) and a combo box
    to select the reference data source from the registry.
    """

    PAGE_ID = "variables"
    PAGE_TITLE = "Variables & References"
    PAGE_SUBTITLE = "Select evaluation variables and their reference data sources"
    CONTENT_EXPAND = True

    def _setup_content(self):
        """Setup page content with variable checkboxes and reference combos."""
        self._checkboxes = {}  # var_name -> QCheckBox
        self._ref_combos = {}  # var_name -> QComboBox
        self._registry = None

        # Load registry
        try:
            from openbench.data.registry import RegistryManager
            self._registry = RegistryManager()
        except Exception:
            pass

        # Build UI for each category
        for category, variables in EVALUATION_ITEMS.items():
            group = QGroupBox(category)
            group_layout = QVBoxLayout()
            group_layout.setSpacing(4)

            for var_name in variables:
                row = QHBoxLayout()
                row.setSpacing(8)

                # Checkbox for variable
                cb = QCheckBox(var_name.replace("_", " "))
                cb.setObjectName(var_name)
                cb.stateChanged.connect(self._on_variable_toggled)
                self._checkboxes[var_name] = cb
                row.addWidget(cb, stretch=2)

                # Reference source combo
                combo = QComboBox()
                combo.setMinimumWidth(200)
                combo.setEnabled(False)  # Disabled until variable is checked
                self._populate_ref_combo(combo, var_name)
                combo.currentIndexChanged.connect(self._on_selection_changed)
                self._ref_combos[var_name] = combo
                row.addWidget(combo, stretch=3)

                group_layout.addLayout(row)

            group.setLayout(group_layout)
            self.content_layout.addWidget(group)

        # Select/deselect all buttons
        btn_layout = QHBoxLayout()
        select_all = QPushButton("Select All")
        select_all.clicked.connect(self._select_all)
        deselect_all = QPushButton("Deselect All")
        deselect_all.clicked.connect(self._deselect_all)
        btn_layout.addWidget(select_all)
        btn_layout.addWidget(deselect_all)
        btn_layout.addStretch()
        self.content_layout.addLayout(btn_layout)

    def _populate_ref_combo(self, combo, var_name):
        """Populate reference source combo from registry."""
        combo.clear()
        combo.addItem("-- Select Reference --", None)

        if self._registry:
            refs = self._registry.references_for_variable(var_name)
            for ref in sorted(refs, key=lambda r: r.name):
                label = f"{ref.name} ({ref.data_type}, {ref.tim_res})"
                combo.addItem(label, ref.name)

    def _on_variable_toggled(self):
        """Enable/disable reference combo when variable is toggled."""
        for var_name, cb in self._checkboxes.items():
            combo = self._ref_combos[var_name]
            combo.setEnabled(cb.isChecked())
            if not cb.isChecked():
                combo.setCurrentIndex(0)
        self.save_to_config()

    def _on_selection_changed(self):
        self.save_to_config()

    def _select_all(self):
        for cb in self._checkboxes.values():
            cb.setChecked(True)

    def _deselect_all(self):
        for cb in self._checkboxes.values():
            cb.setChecked(False)

    def load_from_config(self):
        """Load from config — check variables and set reference sources."""
        config = self.controller.config

        # Load evaluation items
        eval_items = config.get("evaluation_items", {})
        for var_name, cb in self._checkboxes.items():
            cb.setChecked(eval_items.get(var_name, False))

        # Load reference sources
        ref_config = config.get("reference", config.get("ref_data", {}))
        if isinstance(ref_config, dict):
            for var_name, combo in self._ref_combos.items():
                source = None
                # Handle both old format (nested) and new format (flat)
                if var_name in ref_config:
                    val = ref_config[var_name]
                    if isinstance(val, str):
                        source = val
                    elif isinstance(val, dict):
                        source = val.get("source")

                if source:
                    # Find the combo item matching this source
                    for i in range(combo.count()):
                        if combo.itemData(i) == source:
                            combo.setCurrentIndex(i)
                            break

    def save_to_config(self):
        """Save to config — evaluation items and reference mapping."""
        # Save evaluation items as boolean dict
        eval_items = {}
        for var_name, cb in self._checkboxes.items():
            if cb.isChecked():
                eval_items[var_name] = True

        self.controller.update_section("evaluation_items", eval_items)

        # Save reference mapping
        reference = {}
        for var_name, combo in self._ref_combos.items():
            if self._checkboxes[var_name].isChecked():
                source = combo.currentData()
                if source:
                    reference[var_name] = source

        self.controller.update_section("reference", reference)

    def validate(self) -> bool:
        """Check that at least one variable is selected with a reference."""
        self.save_to_config()
        eval_items = self.controller.config.get("evaluation_items", {})
        selected = [k for k, v in eval_items.items() if v]
        if not selected:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Validation", "Please select at least one evaluation variable.")
            return False

        # Check all selected variables have reference sources
        reference = self.controller.config.get("reference", {})
        missing = [v for v in selected if v not in reference]
        if missing:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Validation",
                f"Missing reference source for: {', '.join(missing[:5])}"
                + (f" and {len(missing)-5} more" if len(missing) > 5 else "")
            )
            return False

        return True
