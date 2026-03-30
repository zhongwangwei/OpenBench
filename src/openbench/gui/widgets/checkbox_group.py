# -*- coding: utf-8 -*-
"""
Grouped checkbox widget with search and select all/none functionality.
"""

from typing import Dict, List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QCheckBox, QLineEdit, QPushButton, QLabel,
    QGroupBox, QScrollArea, QFrame
)
from PySide6.QtCore import Signal, Qt


class CheckboxGroup(QWidget):
    """Widget displaying grouped checkboxes with search and bulk selection."""

    selection_changed = Signal(dict)  # Emits {item_id: bool, ...}

    def __init__(
        self,
        items: Dict[str, Dict[str, List[str]]],
        parent=None
    ):
        """
        Initialize CheckboxGroup.

        Args:
            items: Nested dict of {group_name: {category: [item_names]}}
                   or flat dict {group_name: [item_names]}
            parent: Parent widget
        """
        super().__init__(parent)
        self.items = items
        self._checkboxes: Dict[str, QCheckBox] = {}
        self._setup_ui()

    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Toolbar: Search + Select All/None
        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)

        # Search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search...")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.textChanged.connect(self._filter_items)
        toolbar.addWidget(self.search_input, 1)

        # Select all button
        self.btn_select_all = QPushButton("Select All")
        self.btn_select_all.setProperty("secondary", True)
        self.btn_select_all.clicked.connect(self._select_all)
        toolbar.addWidget(self.btn_select_all)

        # Select none button
        self.btn_select_none = QPushButton("Select None")
        self.btn_select_none.setProperty("secondary", True)
        self.btn_select_none.clicked.connect(self._select_none)
        toolbar.addWidget(self.btn_select_none)

        layout.addLayout(toolbar)

        # Selection count
        self.count_label = QLabel("Selected: 0")
        self.count_label.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(self.count_label)

        # Scroll area for checkboxes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Container for groups
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(15)

        # Create checkbox groups
        for group_name, group_items in self.items.items():
            group_box = QGroupBox(group_name)
            group_layout = QGridLayout(group_box)
            group_layout.setSpacing(8)

            # Flatten if nested
            if isinstance(group_items, dict):
                flat_items = []
                for cat_items in group_items.values():
                    flat_items.extend(cat_items)
            else:
                flat_items = group_items

            # Create checkboxes in grid (3 columns)
            col_count = 3
            for i, item_name in enumerate(flat_items):
                cb = QCheckBox(item_name.replace("_", " "))
                cb.setProperty("item_id", item_name)
                cb.stateChanged.connect(self._on_checkbox_changed)
                self._checkboxes[item_name] = cb
                group_layout.addWidget(cb, i // col_count, i % col_count)

            container_layout.addWidget(group_box)  # No stretch - maintain natural size

        # Add stretch at end to push groups to top
        container_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll, 1)

    def _on_checkbox_changed(self, state):
        """Handle checkbox state change."""
        self._update_count()
        self.selection_changed.emit(self.get_selection())

    def _update_count(self):
        """Update the selection count label."""
        count = sum(1 for cb in self._checkboxes.values() if cb.isChecked())
        total = len(self._checkboxes)
        self.count_label.setText(f"Selected: {count} / {total}")

    def _filter_items(self, text: str):
        """Filter checkboxes by search text."""
        text = text.lower()
        for item_id, cb in self._checkboxes.items():
            visible = text in item_id.lower() or text in cb.text().lower()
            cb.setVisible(visible)

    def _select_all(self):
        """Select all visible checkboxes."""
        for cb in self._checkboxes.values():
            if cb.isVisible():
                cb.setChecked(True)

    def _select_none(self):
        """Deselect all visible checkboxes."""
        for cb in self._checkboxes.values():
            if cb.isVisible():
                cb.setChecked(False)

    def get_selection(self) -> Dict[str, bool]:
        """Get current selection state."""
        return {
            item_id: cb.isChecked()
            for item_id, cb in self._checkboxes.items()
        }

    def set_selection(self, selection: Dict[str, bool]):
        """Set selection state."""
        for item_id, checked in selection.items():
            if item_id in self._checkboxes:
                self._checkboxes[item_id].blockSignals(True)
                self._checkboxes[item_id].setChecked(checked)
                self._checkboxes[item_id].blockSignals(False)
        self._update_count()

    def get_selected_items(self) -> List[str]:
        """Get list of selected item IDs."""
        return [
            item_id for item_id, cb in self._checkboxes.items()
            if cb.isChecked()
        ]
