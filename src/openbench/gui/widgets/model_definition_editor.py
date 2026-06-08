# -*- coding: utf-8 -*-
"""
Dialog for creating and editing model definition files.
"""

import base64
import logging
import os
import shlex
import yaml
from typing import Dict, Any, Optional

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QGroupBox,
    QDialogButtonBox,
    QFileDialog,
    QMessageBox,
    QHeaderView,
    QLabel,
)
from PySide6.QtCore import Qt

logger = logging.getLogger(__name__)


# Common evaluation variables
EVALUATION_VARIABLES = [
    "Sensible_Heat",
    "Latent_Heat",
    "Ground_Heat",
    "Net_Radiation",
    "Surface_Upward_SW_Radiation",
    "Surface_Upward_LW_Radiation",
    "Gross_Primary_Productivity",
    "Ecosystem_Respiration",
    "Leaf_Area_Index",
    "Evapotranspiration",
    "Canopy_Transpiration",
    "Ground_Evaporation",
    "Total_Runoff",
    "Surface_Runoff",
    "Subsurface_Runoff",
    "Snow_Water_Equivalent",
    "Snow_Depth",
    "Surface_Soil_Moisture",
    "Root_Zone_Soil_Moisture",
    "Surface_Soil_Temperature",
    "Streamflow",
    "Water_Table_Depth",
    "Terrestrial_Water_Storage_Change",
]


class ModelDefinitionEditor(QDialog):
    """Dialog for creating and editing model definition files."""

    def __init__(
        self,
        initial_data: Optional[Dict[str, Any]] = None,
        file_path: Optional[str] = None,
        ssh_manager=None,
        parent=None,
    ):
        super().__init__(parent)
        self.initial_data = initial_data or {}
        self.file_path = file_path  # Path to existing file (for editing)
        self._saved_path = file_path or ""
        self._ssh_manager = ssh_manager  # SSH manager for remote mode

        # Set title based on mode
        if file_path:
            self.setWindowTitle(f"Edit Model Definition - {os.path.basename(file_path)}")
        else:
            self.setWindowTitle("New Model Definition")

        self.setMinimumSize(700, 600)
        self.resize(800, 700)
        self.setModal(True)

        self._setup_ui()
        self._load_data()

    def _is_remote_mode(self) -> bool:
        """Check if we're in remote mode."""
        return self._ssh_manager is not None and self._ssh_manager.is_connected

    def _setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Model name
        name_group = QGroupBox("Model Information")
        name_layout = QFormLayout(name_group)

        self.model_name = QLineEdit()
        self.model_name.setPlaceholderText("e.g., CoLM, CLM5, Noah-MP")
        name_layout.addRow("Model Name:", self.model_name)

        layout.addWidget(name_group)

        # Variable mappings
        var_group = QGroupBox("Variable Mappings")
        var_layout = QVBoxLayout(var_group)

        hint_label = QLabel("Define variable names and units for each evaluation variable:")
        hint_label.setStyleSheet("color: #666; font-style: italic;")
        var_layout.addWidget(hint_label)

        # Table for variable mappings
        self.var_table = QTableWidget()
        self.var_table.setColumnCount(3)
        self.var_table.setHorizontalHeaderLabels(["Variable", "Variable Name in File", "Unit"])
        self.var_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.var_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.var_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)

        # Populate with common variables
        self.var_table.setRowCount(len(EVALUATION_VARIABLES))
        for i, var_name in enumerate(EVALUATION_VARIABLES):
            # Variable name (read-only)
            item = QTableWidgetItem(var_name.replace("_", " "))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            item.setData(Qt.UserRole, var_name)
            self.var_table.setItem(i, 0, item)

            # Variable name in file (editable)
            self.var_table.setItem(i, 1, QTableWidgetItem(""))

            # Unit (editable)
            self.var_table.setItem(i, 2, QTableWidgetItem(""))

        var_layout.addWidget(self.var_table)

        layout.addWidget(var_group, 1)

        # Dialog buttons
        btn_layout = QHBoxLayout()

        # Save button (only when editing existing file)
        if self.file_path:
            self.btn_save_inplace = QPushButton("Save")
            self.btn_save_inplace.clicked.connect(self._save_inplace)
            btn_layout.addWidget(self.btn_save_inplace)

        self.btn_save = QPushButton("Save As...")
        self.btn_save.clicked.connect(self._save_file)
        btn_layout.addWidget(self.btn_save)

        btn_layout.addStretch()

        btn_box = QDialogButtonBox(QDialogButtonBox.Cancel)
        btn_box.rejected.connect(self.reject)
        btn_layout.addWidget(btn_box)

        layout.addLayout(btn_layout)

    def _load_data(self):
        """Load initial data into form."""
        if not self.initial_data:
            return

        general = self.initial_data.get("general", {})
        if "model" in general:
            self.model_name.setText(general["model"])

        # Load variable mappings
        for i in range(self.var_table.rowCount()):
            var_item = self.var_table.item(i, 0)
            var_name = var_item.data(Qt.UserRole)

            if var_name in self.initial_data:
                var_data = self.initial_data[var_name]
                if "varname" in var_data:
                    self.var_table.item(i, 1).setText(str(var_data["varname"]))
                if "varunit" in var_data:
                    self.var_table.item(i, 2).setText(str(var_data["varunit"]))

    def get_data(self) -> Dict[str, Any]:
        """Get form data as dictionary."""
        data = {"general": {"model": self.model_name.text()}}

        # Collect variable mappings
        for i in range(self.var_table.rowCount()):
            var_item = self.var_table.item(i, 0)
            var_name = var_item.data(Qt.UserRole)

            item1 = self.var_table.item(i, 1)
            item2 = self.var_table.item(i, 2)
            varname = item1.text().strip() if item1 else ""
            varunit = item2.text().strip() if item2 else ""

            # Only include if at least varname is provided
            if varname or varunit:
                data[var_name] = {"varname": varname, "varunit": varunit}

        return data

    def _save_inplace(self):
        """Save model definition to the existing file."""
        if not self.file_path:
            return

        # Close any active cell editor to commit edits
        self.var_table.setCurrentCell(-1, -1)

        model_name = self.model_name.text().strip()
        if not model_name:
            QMessageBox.warning(self, "Error", "Please enter a model name.")
            return

        data = self.get_data()

        if self._is_remote_mode():
            # Save to remote server
            yaml_content = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=2)

            try:
                encoded = base64.b64encode(yaml_content.encode("utf-8")).decode("ascii")
                cmd = f"printf %s {shlex.quote(encoded)} | base64 -d > {shlex.quote(self.file_path)}"
                stdout, stderr, exit_code = self._ssh_manager.execute(cmd, timeout=30)

                if exit_code != 0:
                    QMessageBox.critical(self, "Error", f"Failed to save file on remote server:\n{stderr or stdout}")
                    return

                self._saved_path = self.file_path
                QMessageBox.information(self, "Success", f"Model definition saved to remote server:\n{self.file_path}")
                self.accept()

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file:\n{str(e)}")
        else:
            # Save to local file
            try:
                with open(self.file_path, "w", encoding="utf-8") as f:
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=2)

                self._saved_path = self.file_path
                QMessageBox.information(self, "Success", f"Model definition saved to:\n{self.file_path}")
                self.accept()

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file:\n{str(e)}")

    def _save_file(self):
        """Save model definition to a new file."""
        # Close any active cell editor to commit edits
        self.var_table.setCurrentCell(-1, -1)

        model_name = self.model_name.text().strip()
        if not model_name:
            QMessageBox.warning(self, "Error", "Please enter a model name.")
            return

        if self._is_remote_mode():
            self._save_file_remote(model_name)
        else:
            self._save_file_local(model_name)

    def _save_file_local(self, model_name: str):
        """Save model definition to a local file."""
        # Suggest default path without assuming any source-tree/v2 layout.
        candidates = [
            getattr(self, "_last_save_dir", None),
            os.getcwd(),
            os.path.expanduser("~"),
        ]
        default_dir = next(
            (c for c in candidates if c and os.path.isdir(c)),
            os.path.expanduser("~"),
        )
        default_path = os.path.join(default_dir, f"{model_name}.yaml")

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Model Definition", default_path, "YAML Files (*.yaml)")

        if not file_path:
            return

        # Generate and save YAML
        data = self.get_data()

        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=2)

            self._saved_path = file_path
            self._last_save_dir = os.path.dirname(file_path)
            QMessageBox.information(self, "Success", f"Model definition saved to:\n{file_path}")
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file:\n{str(e)}")

    def _save_file_remote(self, model_name: str):
        """Save model definition to remote server."""
        from openbench.gui.widgets.remote_config import RemoteFileBrowser

        # Get remote home directory as default
        try:
            home_dir = self._ssh_manager._get_home_dir()
        except Exception:
            home_dir = "/"

        # Let user select directory on remote server
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Save Location on Remote Server")
        dialog.resize(500, 400)

        layout = QVBoxLayout(dialog)

        # Add hint label
        hint = QLabel(f"Select directory to save '{model_name}.yaml':")
        layout.addWidget(hint)

        browser = RemoteFileBrowser(self._ssh_manager, home_dir, dialog, select_dirs=True)
        layout.addWidget(browser)

        selected_path = [None]  # Use list to allow modification in nested function

        def on_path_selected(path):
            selected_path[0] = path
            dialog.accept()

        browser.file_selected.connect(on_path_selected)

        if dialog.exec() != QDialog.Accepted or not selected_path[0]:
            return

        # Build full file path
        remote_dir = selected_path[0].rstrip("/")
        remote_file = f"{remote_dir}/{model_name}.yaml"

        # Generate YAML content
        data = self.get_data()
        yaml_content = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=2)

        # Save to remote server
        try:
            # Create directory if needed and write file. Encode the payload
            # so YAML content cannot terminate a here-doc or be interpreted
            # by the remote shell.
            encoded = base64.b64encode(yaml_content.encode("utf-8")).decode("ascii")
            cmd = (
                f"mkdir -p {shlex.quote(remote_dir)} && "
                f"printf %s {shlex.quote(encoded)} | base64 -d > {shlex.quote(remote_file)}"
            )
            stdout, stderr, exit_code = self._ssh_manager.execute(cmd, timeout=30)

            if exit_code != 0:
                QMessageBox.critical(self, "Error", f"Failed to save file on remote server:\n{stderr or stdout}")
                return

            self._saved_path = remote_file
            QMessageBox.information(self, "Success", f"Model definition saved to remote server:\n{remote_file}")
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file:\n{str(e)}")

    def get_saved_path(self) -> str:
        """Get the path where the file was saved."""
        return self._saved_path

    def _cleanup(self):
        """Clean up resources before dialog destruction."""
        # Clear SSH manager reference to break potential reference cycles
        self._ssh_manager = None

    def done(self, result):
        """Override done to ensure cleanup on dialog close."""
        self._cleanup()
        super().done(result)
