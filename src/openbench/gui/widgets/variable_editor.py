# -*- coding: utf-8 -*-
"""
Dialog for editing a single variable mapping (model or reference).
"""

import logging
from typing import Optional

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QLineEdit,
    QComboBox,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QListWidget,
    QListWidgetItem,
)

logger = logging.getLogger(__name__)


def _collect_known_variables() -> list[str]:
    """Collect all known variable names from registry catalogs.

    Falls back to a hardcoded list if the registry is unavailable.
    """
    try:
        from openbench.data.registry.manager import get_registry

        mgr = get_registry()
        all_vars: set[str] = set()
        for ref in mgr.list_references():
            all_vars.update(ref.variables.keys())
        for model in mgr.list_models():
            all_vars.update(model.variables.keys())
        if all_vars:
            return sorted(all_vars)
    except Exception:
        logger.debug("Could not collect variables from registry, using fallback list")

    # Fallback: common OpenBench evaluation variable names
    return [
        "Canopy_Transpiration",
        "Ecosystem_Respiration",
        "Evapotranspiration",
        "Gross_Primary_Productivity",
        "Ground_Evaporation",
        "Ground_Heat",
        "Latent_Heat",
        "Leaf_Area_Index",
        "Net_Ecosystem_Exchange",
        "Net_Radiation",
        "Root_Zone_Soil_Moisture",
        "Sensible_Heat",
        "Snow_Depth",
        "Snow_Water_Equivalent",
        "Streamflow",
        "Subsurface_Runoff",
        "Surface_Runoff",
        "Surface_Soil_Moisture",
        "Surface_Soil_Temperature",
        "Surface_Upward_LW_Radiation",
        "Surface_Upward_SW_Radiation",
        "Terrestrial_Water_Storage_Change",
        "Total_Runoff",
        "Water_Table_Depth",
    ]


KNOWN_VARIABLES: list[str] = []  # Populated lazily on first dialog open


class VariableEditorDialog(QDialog):
    """Dialog for editing a single variable mapping.

    Parameters
    ----------
    mode : str
        ``"model"`` shows Variable name, NC varname, unit, compute, fallbacks.
        ``"reference"`` additionally shows sub_dir, prefix, suffix.
    variable_name : str
        The standard OpenBench variable name (e.g. ``Latent_Heat``).
    varname : str
        The NetCDF variable name in the data file.
    varunit : str
        Unit string for the variable.
    sub_dir, prefix, suffix : str
        File-location hints (reference mode only).
    compute : str
        Python expression to compute the variable (model mode).
    fallbacks : list[dict] | None
        List of fallback variable dicts, each with ``varname``, ``varunit``, ``convert``.
    parent : QWidget | None
        Parent widget.
    """

    def __init__(
        self,
        mode: str = "model",
        variable_name: str = "",
        varname: str = "",
        varunit: str = "",
        sub_dir: str = "",
        prefix: str = "",
        suffix: str = "",
        compute: str = "",
        fallbacks: Optional[list[dict]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._mode = mode
        self._fallbacks_data: list[dict] = list(fallbacks) if fallbacks else []

        self.setWindowTitle("Edit Variable Mapping")
        self.setMinimumWidth(480)
        self.setModal(True)

        # Ensure KNOWN_VARIABLES is populated
        global KNOWN_VARIABLES
        if not KNOWN_VARIABLES:
            KNOWN_VARIABLES = _collect_known_variables()

        self._setup_ui(variable_name, varname, varunit, sub_dir, prefix, suffix, compute)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(
        self,
        variable_name: str,
        varname: str,
        varunit: str,
        sub_dir: str,
        prefix: str,
        suffix: str,
        compute: str,
    ):
        layout = QVBoxLayout(self)

        # --- Core fields ---
        core_group = QGroupBox("Variable Mapping")
        form = QFormLayout(core_group)

        self.combo_variable = QComboBox()
        self.combo_variable.setEditable(True)
        self.combo_variable.addItems(KNOWN_VARIABLES)
        if variable_name:
            idx = self.combo_variable.findText(variable_name)
            if idx >= 0:
                self.combo_variable.setCurrentIndex(idx)
            else:
                self.combo_variable.setEditText(variable_name)
        form.addRow("Variable name:", self.combo_variable)

        self.edit_varname = QLineEdit(varname)
        self.edit_varname.setPlaceholderText("NC variable name (e.g. Qle)")
        form.addRow("NC varname:", self.edit_varname)

        self.edit_varunit = QLineEdit(varunit)
        self.edit_varunit.setPlaceholderText("Unit (e.g. W m-2)")
        form.addRow("Unit:", self.edit_varunit)

        if self._mode == "model":
            self.edit_compute = QLineEdit(compute)
            self.edit_compute.setPlaceholderText("Python expression (optional)")
            form.addRow("Compute:", self.edit_compute)
        else:
            self.edit_compute = None

        # Reference-only fields
        if self._mode == "reference":
            self.edit_sub_dir = QLineEdit(sub_dir)
            self.edit_sub_dir.setPlaceholderText("Subdirectory inside root_dir")
            form.addRow("sub_dir:", self.edit_sub_dir)

            self.edit_prefix = QLineEdit(prefix)
            self.edit_prefix.setPlaceholderText("File prefix")
            form.addRow("prefix:", self.edit_prefix)

            self.edit_suffix = QLineEdit(suffix)
            self.edit_suffix.setPlaceholderText("File suffix")
            form.addRow("suffix:", self.edit_suffix)
        else:
            self.edit_sub_dir = None
            self.edit_prefix = None
            self.edit_suffix = None

        layout.addWidget(core_group)

        # --- Fallbacks ---
        fb_group = QGroupBox("Fallback Variables")
        fb_layout = QVBoxLayout(fb_group)

        self.fallback_list = QListWidget()
        self.fallback_list.setMaximumHeight(120)
        self._refresh_fallback_list()
        fb_layout.addWidget(self.fallback_list)

        fb_btn_layout = QHBoxLayout()
        btn_add_fb = QPushButton("+ Add Fallback")
        btn_add_fb.clicked.connect(self._add_fallback)
        fb_btn_layout.addWidget(btn_add_fb)

        btn_remove_fb = QPushButton("Remove Selected")
        btn_remove_fb.clicked.connect(self._remove_fallback)
        fb_btn_layout.addWidget(btn_remove_fb)
        fb_btn_layout.addStretch()
        fb_layout.addLayout(fb_btn_layout)

        layout.addWidget(fb_group)

        # --- OK / Cancel ---
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    # ------------------------------------------------------------------
    # Fallback helpers
    # ------------------------------------------------------------------

    def _refresh_fallback_list(self):
        self.fallback_list.clear()
        for fb in self._fallbacks_data:
            label = fb.get("varname", "?")
            unit = fb.get("varunit", "")
            convert = fb.get("convert", "")
            display = f"{label} [{unit}]"
            if convert:
                display += f"  convert: {convert}"
            item = QListWidgetItem(display)
            self.fallback_list.addItem(item)

    def _add_fallback(self):
        """Add a new fallback entry via a small inline form."""
        dlg = _FallbackEntryDialog(parent=self)
        if dlg.exec() == QDialog.Accepted:
            self._fallbacks_data.append(dlg.get_data())
            self._refresh_fallback_list()

    def _remove_fallback(self):
        row = self.fallback_list.currentRow()
        if 0 <= row < len(self._fallbacks_data):
            self._fallbacks_data.pop(row)
            self._refresh_fallback_list()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_data(self) -> dict:
        """Return the edited variable mapping data.

        Returns
        -------
        dict
            Keys always include ``variable_name``, ``varname``, ``varunit``.
            Depending on mode may also include ``compute``, ``sub_dir``,
            ``prefix``, ``suffix``, and ``fallbacks``.
        """
        data: dict = {
            "variable_name": self.combo_variable.currentText().strip(),
            "varname": self.edit_varname.text().strip(),
            "varunit": self.edit_varunit.text().strip(),
        }
        if self.edit_compute is not None:
            data["compute"] = self.edit_compute.text().strip()
        if self.edit_sub_dir is not None:
            data["sub_dir"] = self.edit_sub_dir.text().strip()
        if self.edit_prefix is not None:
            data["prefix"] = self.edit_prefix.text().strip()
        if self.edit_suffix is not None:
            data["suffix"] = self.edit_suffix.text().strip()
        if self._fallbacks_data:
            data["fallbacks"] = list(self._fallbacks_data)
        return data


class _FallbackEntryDialog(QDialog):
    """Tiny dialog to enter one fallback variable."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Fallback Variable")
        self.setMinimumWidth(360)
        self.setModal(True)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.edit_varname = QLineEdit()
        self.edit_varname.setPlaceholderText("NC varname")
        form.addRow("varname:", self.edit_varname)

        self.edit_varunit = QLineEdit()
        self.edit_varunit.setPlaceholderText("Unit")
        form.addRow("varunit:", self.edit_varunit)

        self.edit_convert = QLineEdit()
        self.edit_convert.setPlaceholderText("e.g. value * 12.011")
        form.addRow("convert:", self.edit_convert)

        layout.addLayout(form)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_data(self) -> dict:
        d: dict = {"varname": self.edit_varname.text().strip()}
        unit = self.edit_varunit.text().strip()
        if unit:
            d["varunit"] = unit
        convert = self.edit_convert.text().strip()
        if convert:
            d["convert"] = convert
        return d
