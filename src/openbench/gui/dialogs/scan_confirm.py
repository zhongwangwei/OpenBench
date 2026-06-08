# -*- coding: utf-8 -*-
"""
Scan confirmation dialog for simulation data.

Shows discovered cases with auto-detected model, lets user adjust
model per-case and confirm/cancel before populating the page.
"""

from typing import Any, Dict, List

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt


class ScanConfirmDialog(QDialog):
    """Confirm scan results: cases discovered, model detected, user adjusts and accepts."""

    def __init__(
        self,
        discovered: List[tuple],  # [(label, nc_dir, prefix), ...]
        model_names: List[str],  # available model names
        auto_model: str,  # auto-detected model name (or "")
        match_info: str,  # human-readable match summary
        nc_var_count: int,  # number of NC variables found
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Confirm Scan Results")
        self.setMinimumWidth(750)
        self.setMinimumHeight(400)

        self._discovered = discovered
        self._model_names = model_names
        self._auto_model = auto_model

        layout = QVBoxLayout(self)

        # --- Summary ---
        if auto_model:
            summary = QLabel(f"Found <b>{len(discovered)}</b> cases with <b>{nc_var_count}</b> NC variables.\n")
        else:
            summary = QLabel(
                f"Found <b>{len(discovered)}</b> cases with "
                f"<b>{nc_var_count}</b> NC variables. "
                f"<span style='color:orange'>No model auto-detected.</span>"
            )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        # --- Match info ---
        if match_info:
            match_label = QLabel(f"<pre>{match_info}</pre>")
            match_label.setStyleSheet("background: #f8f8f8; padding: 6px; border-radius: 4px;")
            layout.addWidget(match_label)

        # --- Case table ---
        layout.addWidget(QLabel("Select cases and assign models:"))

        self._table = QTableWidget(len(discovered), 4)
        self._table.setHorizontalHeaderLabels(["Include", "Case", "Path", "Model"])
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.Stretch)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)

        self._checkboxes: List[QCheckBox] = []
        self._model_combos: List[QComboBox] = []

        for i, (label, nc_dir, prefix) in enumerate(discovered):
            # Checkbox
            cb = QCheckBox()
            cb.setChecked(True)
            self._checkboxes.append(cb)
            cell_widget = QWidget()
            cell_layout = QHBoxLayout(cell_widget)
            cell_layout.addWidget(cb)
            cell_layout.setAlignment(Qt.AlignCenter)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            self._table.setCellWidget(i, 0, cell_widget)

            # Label
            self._table.setItem(i, 1, QTableWidgetItem(label))

            # Path (truncated)
            path_item = QTableWidgetItem(nc_dir)
            path_item.setToolTip(nc_dir)
            self._table.setItem(i, 2, path_item)

            # Model combo
            combo = QComboBox()
            for mn in model_names:
                combo.addItem(mn, mn)
            if auto_model:
                idx = combo.findData(auto_model)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            self._model_combos.append(combo)
            self._table.setCellWidget(i, 3, combo)

        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self._table, stretch=1)

        # --- Select all / none ---
        sel_row = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_all.clicked.connect(lambda: self._set_all(True))
        sel_row.addWidget(btn_all)
        btn_none = QPushButton("Deselect All")
        btn_none.clicked.connect(lambda: self._set_all(False))
        sel_row.addWidget(btn_none)

        # Register new model button
        self._btn_register = QPushButton("Register New Model...")
        self._btn_register.setToolTip("Open Data Registry to register a new model")
        sel_row.addWidget(self._btn_register)
        sel_row.addStretch()
        layout.addLayout(sel_row)

        # --- OK / Cancel ---
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def register_button(self) -> QPushButton:
        """Expose the register button so the caller can connect it."""
        return self._btn_register

    def _set_all(self, checked: bool):
        for cb in self._checkboxes:
            cb.setChecked(checked)

    def get_results(self) -> List[Dict[str, Any]]:
        """Return list of confirmed cases: {label, nc_dir, prefix, model}."""
        results = []
        for i, (label, nc_dir, prefix) in enumerate(self._discovered):
            if self._checkboxes[i].isChecked():
                model = self._model_combos[i].currentData() or ""
                results.append(
                    {
                        "label": label,
                        "nc_dir": nc_dir,
                        "prefix": prefix,
                        "model": model,
                    }
                )
        return results
