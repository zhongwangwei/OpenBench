# -*- coding: utf-8 -*-
"""
Data Registry management page.

Provides two tabs (Models / Datasets) with a list-on-the-left,
editor-on-the-right layout for browsing, creating, and editing
model profiles and reference dataset descriptors stored in the
OpenBench registry.
"""

import logging
from typing import Dict

from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QProgressDialog,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from openbench.gui.pages.base_page import BasePage
from openbench.gui.path_utils import browse_directory

logger = logging.getLogger(__name__)

_DETACHED_SCAN_WORKERS = []

# Custom item role for persisting fallback variable definitions on column 0
# of the variable tables. The table only exposes 4 (model) / 6 (reference) text
# columns, so fallbacks live on the cell's user-data instead of an extra column.
FALLBACKS_ROLE = Qt.UserRole + 1


def _set_row_fallbacks(item, fallbacks):
    """Attach a list of fallback dicts to a column-0 QTableWidgetItem."""
    if item is None:
        return
    item.setData(FALLBACKS_ROLE, list(fallbacks or []))


def _get_row_fallbacks(table, row) -> list:
    """Read fallback dict list stored on column 0 of the given row."""
    item = table.item(row, 0)
    if item is None:
        return []
    raw = item.data(FALLBACKS_ROLE)
    return list(raw) if isinstance(raw, list) else []


# ---------------------------------------------------------------------------
# Helper: lazy imports to avoid circular / heavy-import at module scope
# ---------------------------------------------------------------------------


def _get_registry():
    from openbench.data.registry.manager import get_registry

    return get_registry()


def _clear_cache():
    from openbench.data.registry.manager import clear_registry_cache

    clear_registry_cache()


def _schema():
    from openbench.data.registry import schema

    return schema


# ===================================================================
# PageRegistry
# ===================================================================


class PageRegistry(BasePage):
    """Data Registry management page."""

    PAGE_ID = "registry"
    PAGE_TITLE = "Data Registry"
    PAGE_SUBTITLE = "Browse and edit registered model profiles and reference datasets"
    CONTENT_EXPAND = True

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_content(self):
        self.tabs = QTabWidget()
        self.content_layout.addWidget(self.tabs)

        # --- Models tab ---
        models_widget = QWidget()
        self._setup_models_tab(models_widget)
        self.tabs.addTab(models_widget, "Models")

        # --- Reference Datasets tab ---
        datasets_widget = QWidget()
        self._setup_datasets_tab(datasets_widget)
        self.tabs.addTab(datasets_widget, "Reference Datasets")

        # Initial population
        self._refresh_model_list()
        self._refresh_dataset_list()

    # ==================================================================
    # MODELS TAB
    # ==================================================================

    def _setup_models_tab(self, parent: QWidget):
        layout = QHBoxLayout(parent)

        splitter = QSplitter(Qt.Horizontal)

        # ---- Left: list + buttons ----
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.model_list = QListWidget()
        self.model_list.currentRowChanged.connect(self._on_model_selected)
        left_layout.addWidget(self.model_list, stretch=1)

        btn_row = QHBoxLayout()
        btn_new_model = QPushButton("+ New Model")
        btn_new_model.clicked.connect(self._new_model)
        btn_row.addWidget(btn_new_model)

        btn_import_nc = QPushButton("Import")
        btn_import_nc.setToolTip("Import variables from a NetCDF file")
        btn_import_nc.clicked.connect(self._import_model_from_nc)
        btn_row.addWidget(btn_import_nc)

        btn_del_model = QPushButton("Delete")
        btn_del_model.clicked.connect(self._delete_model)
        btn_row.addWidget(btn_del_model)

        left_layout.addLayout(btn_row)
        splitter.addWidget(left)

        # ---- Right: editor ----
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        editor_group = QGroupBox("Model Profile")
        editor_form = QVBoxLayout(editor_group)

        # Top fields
        form_row = QHBoxLayout()
        form_row.addWidget(QLabel("Name:"))
        self.model_name = QLineEdit()
        form_row.addWidget(self.model_name, stretch=1)
        form_row.addWidget(QLabel("Description:"))
        self.model_desc = QLineEdit()
        form_row.addWidget(self.model_desc, stretch=2)
        editor_form.addLayout(form_row)

        form_row2 = QHBoxLayout()
        form_row2.addWidget(QLabel("data_type:"))
        self.model_data_type = QComboBox()
        self.model_data_type.addItems(["grid", "stn"])
        form_row2.addWidget(self.model_data_type)
        form_row2.addWidget(QLabel("grid_res:"))
        self.model_grid_res = QLineEdit()
        self.model_grid_res.setPlaceholderText("e.g. 0.5")
        self.model_grid_res.setMaximumWidth(100)
        form_row2.addWidget(self.model_grid_res)
        form_row2.addStretch()
        editor_form.addLayout(form_row2)

        # Variable table
        editor_form.addWidget(QLabel("Variables:"))
        self.model_var_table = QTableWidget(0, 4)
        self.model_var_table.setHorizontalHeaderLabels(["Variable", "varname", "varunit", "compute"])
        hdr = self.model_var_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.Stretch)
        self.model_var_table.setSelectionBehavior(QTableWidget.SelectRows)
        editor_form.addWidget(self.model_var_table, stretch=1)

        var_btn_row = QHBoxLayout()
        btn_add_var = QPushButton("+ Add Variable")
        btn_add_var.clicked.connect(self._model_add_variable)
        var_btn_row.addWidget(btn_add_var)

        btn_rm_var = QPushButton("Remove Selected")
        btn_rm_var.clicked.connect(self._model_remove_variable)
        var_btn_row.addWidget(btn_rm_var)

        btn_edit_var = QPushButton("Edit Variable...")
        btn_edit_var.clicked.connect(self._model_edit_variable)
        var_btn_row.addWidget(btn_edit_var)

        var_btn_row.addStretch()
        editor_form.addLayout(var_btn_row)

        # Save / Revert
        save_row = QHBoxLayout()
        save_row.addStretch()
        btn_revert = QPushButton("Revert")
        btn_revert.clicked.connect(self._model_revert)
        save_row.addWidget(btn_revert)
        btn_save = QPushButton("Save")
        btn_save.clicked.connect(self._model_save)
        save_row.addWidget(btn_save)
        editor_form.addLayout(save_row)

        right_layout.addWidget(editor_group)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)

    # ------------------------------------------------------------------
    # Model list helpers
    # ------------------------------------------------------------------

    def _refresh_model_list(self):
        self.model_list.clear()
        try:
            models = _get_registry().list_models()
            for m in models:
                n_vars = len(m.variables)
                item = QListWidgetItem(f"{m.name}  ({n_vars} vars)")
                item.setData(Qt.UserRole, m.name)
                self.model_list.addItem(item)
        except Exception as exc:
            logger.warning("Failed to list models: %s", exc)

    def _on_model_selected(self, row: int):
        if row < 0:
            return
        item = self.model_list.item(row)
        if item is None:
            return
        name = item.data(Qt.UserRole)
        model = _get_registry().get_model(name)
        if model is None:
            return
        self._populate_model_editor(model)

    def _populate_model_editor(self, model):
        self.model_name.setText(model.name)
        self.model_desc.setText(model.description)
        idx = self.model_data_type.findText(model.data_type)
        if idx >= 0:
            self.model_data_type.setCurrentIndex(idx)
        self.model_grid_res.setText(str(model.grid_res) if model.grid_res is not None else "")

        self.model_var_table.setRowCount(0)
        for var_name, vm in model.variables.items():
            row = self.model_var_table.rowCount()
            self.model_var_table.insertRow(row)
            name_item = QTableWidgetItem(var_name)
            _set_row_fallbacks(
                name_item,
                [fb.to_dict() for fb in (vm.fallbacks or [])],
            )
            self.model_var_table.setItem(row, 0, name_item)
            self.model_var_table.setItem(row, 1, QTableWidgetItem(str(vm.varname)))
            self.model_var_table.setItem(row, 2, QTableWidgetItem(vm.varunit))
            self.model_var_table.setItem(row, 3, QTableWidgetItem(vm.compute or ""))

    # ------------------------------------------------------------------
    # Model actions
    # ------------------------------------------------------------------

    def _new_model(self):
        self.model_name.clear()
        self.model_desc.clear()
        self.model_data_type.setCurrentIndex(0)
        self.model_grid_res.clear()
        self.model_var_table.setRowCount(0)
        self.model_list.clearSelection()

    def _import_model_from_nc(self):
        from openbench.gui.path_utils import remote_exec_context
        from openbench.gui.widgets.nc_importer import NCImporterDialog

        context = remote_exec_context(self.controller, self)
        if context is None:
            return
        kwargs = {"parent": self, **context}
        kwargs.pop("openbench_path", None)  # the importer inspects files, not the package

        dlg = NCImporterDialog(**kwargs)
        if dlg.exec():
            selected = dlg.get_selected_variables()
            if not selected:
                return
            # Append imported variables to the table
            for var in selected:
                row = self.model_var_table.rowCount()
                self.model_var_table.insertRow(row)
                self.model_var_table.setItem(row, 0, QTableWidgetItem(var["varname"]))
                self.model_var_table.setItem(row, 1, QTableWidgetItem(var["varname"]))
                self.model_var_table.setItem(row, 2, QTableWidgetItem(var.get("varunit", "")))
                self.model_var_table.setItem(row, 3, QTableWidgetItem(""))

    def _delete_model(self):
        item = self.model_list.currentItem()
        if item is None:
            return
        name = item.data(Qt.UserRole)
        reply = QMessageBox.question(
            self,
            "Delete Model",
            f"Delete model profile '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            try:
                _get_registry().delete_model(name)
                _clear_cache()
                self._refresh_model_list()
                self._new_model()
            except Exception as exc:
                QMessageBox.critical(self, "Error", f"Failed to delete model:\n{exc}")

    def _model_add_variable(self):
        from openbench.gui.widgets.variable_editor import VariableEditorDialog

        dlg = VariableEditorDialog(mode="model", parent=self)
        if dlg.exec():
            data = dlg.get_data()
            row = self.model_var_table.rowCount()
            self.model_var_table.insertRow(row)
            name_item = QTableWidgetItem(data.get("variable_name", ""))
            _set_row_fallbacks(name_item, data.get("fallbacks") or [])
            self.model_var_table.setItem(row, 0, name_item)
            self.model_var_table.setItem(row, 1, QTableWidgetItem(data.get("varname", "")))
            self.model_var_table.setItem(row, 2, QTableWidgetItem(data.get("varunit", "")))
            self.model_var_table.setItem(row, 3, QTableWidgetItem(data.get("compute", "")))

    def _model_remove_variable(self):
        rows = sorted({idx.row() for idx in self.model_var_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.model_var_table.removeRow(r)

    def _model_edit_variable(self):
        row = self.model_var_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "No Selection", "Select a variable row first.")
            return
        from openbench.gui.widgets.variable_editor import VariableEditorDialog

        variable_name = self.model_var_table.item(row, 0).text() if self.model_var_table.item(row, 0) else ""
        varname = self.model_var_table.item(row, 1).text() if self.model_var_table.item(row, 1) else ""
        varunit = self.model_var_table.item(row, 2).text() if self.model_var_table.item(row, 2) else ""
        compute = self.model_var_table.item(row, 3).text() if self.model_var_table.item(row, 3) else ""
        existing_fallbacks = _get_row_fallbacks(self.model_var_table, row)

        dlg = VariableEditorDialog(
            mode="model",
            variable_name=variable_name,
            varname=varname,
            varunit=varunit,
            compute=compute,
            fallbacks=existing_fallbacks,
            parent=self,
        )
        if dlg.exec():
            data = dlg.get_data()
            name_item = QTableWidgetItem(data.get("variable_name", ""))
            _set_row_fallbacks(name_item, data.get("fallbacks") or [])
            self.model_var_table.setItem(row, 0, name_item)
            self.model_var_table.setItem(row, 1, QTableWidgetItem(data.get("varname", "")))
            self.model_var_table.setItem(row, 2, QTableWidgetItem(data.get("varunit", "")))
            self.model_var_table.setItem(row, 3, QTableWidgetItem(data.get("compute", "")))

    def _model_revert(self):
        """Re-populate editor from registry for the currently selected model."""
        item = self.model_list.currentItem()
        if item is None:
            return
        name = item.data(Qt.UserRole)
        model = _get_registry().get_model(name)
        if model:
            self._populate_model_editor(model)

    def _model_save(self):
        """Build a ModelProfile from the editor and save it."""
        schema = _schema()
        name = self.model_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Name Required", "Please enter a model name.")
            return

        grid_res_text = self.model_grid_res.text().strip()
        if grid_res_text:
            try:
                grid_res = float(grid_res_text)
            except ValueError:
                QMessageBox.warning(
                    self,
                    "Invalid grid_res",
                    f"grid_res must be a number, got: {grid_res_text!r}",
                )
                return
        else:
            grid_res = None

        variables: dict = {}
        for row in range(self.model_var_table.rowCount()):
            var_key = self.model_var_table.item(row, 0).text().strip() if self.model_var_table.item(row, 0) else ""
            varname = self.model_var_table.item(row, 1).text().strip() if self.model_var_table.item(row, 1) else ""
            varunit = self.model_var_table.item(row, 2).text().strip() if self.model_var_table.item(row, 2) else ""
            compute = self.model_var_table.item(row, 3).text().strip() if self.model_var_table.item(row, 3) else ""
            if not var_key:
                continue
            fallback_dicts = _get_row_fallbacks(self.model_var_table, row)
            fallbacks = [
                schema.FallbackVar(
                    varname=fb.get("varname", ""),
                    varunit=fb.get("varunit", ""),
                    convert=fb.get("convert", ""),
                )
                for fb in fallback_dicts
                if fb.get("varname")
            ] or None
            vm = schema.VariableMapping(
                varname=varname,
                varunit=varunit,
                compute=compute or None,
                fallbacks=fallbacks,
            )
            variables[var_key] = vm

        profile = schema.ModelProfile(
            name=name,
            description=self.model_desc.text().strip(),
            data_type=self.model_data_type.currentText(),
            grid_res=grid_res,
            variables=variables,
        )

        try:
            _get_registry().save_model(name, profile)
            _clear_cache()
            self._refresh_model_list()
            QMessageBox.information(self, "Saved", f"Model '{name}' saved to registry.")
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", f"Failed to save model:\n{exc}")

    # ==================================================================
    # DATASETS TAB
    # ==================================================================

    def _setup_datasets_tab(self, parent: QWidget):
        layout = QHBoxLayout(parent)

        splitter = QSplitter(Qt.Horizontal)

        # ---- Left: list + buttons ----
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.dataset_list = QListWidget()
        self.dataset_list.currentRowChanged.connect(self._on_dataset_selected)
        left_layout.addWidget(self.dataset_list, stretch=1)

        btn_row = QHBoxLayout()
        btn_new_ds = QPushButton("+ New Dataset")
        btn_new_ds.clicked.connect(self._new_dataset)
        btn_row.addWidget(btn_new_ds)

        btn_scan = QPushButton("Scan Directory")
        btn_scan.clicked.connect(self._scan_directory)
        btn_row.addWidget(btn_scan)

        btn_del_ds = QPushButton("Delete")
        btn_del_ds.clicked.connect(self._delete_dataset)
        btn_row.addWidget(btn_del_ds)

        left_layout.addLayout(btn_row)
        splitter.addWidget(left)

        # ---- Right: editor ----
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        editor_group = QGroupBox("Reference Dataset")
        editor_form = QVBoxLayout(editor_group)

        # Row 1: Name, Description
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Name:"))
        self.ds_name = QLineEdit()
        row1.addWidget(self.ds_name, stretch=1)
        row1.addWidget(QLabel("Description:"))
        self.ds_desc = QLineEdit()
        row1.addWidget(self.ds_desc, stretch=2)
        editor_form.addLayout(row1)

        # Row 2: category, data_type, tim_res, grid_res
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("category:"))
        self.ds_category = QComboBox()
        self.ds_category.setEditable(True)
        self.ds_category.addItems(["Water", "Energy", "Bio", "Meteo", "Composite", "Heat"])
        row2.addWidget(self.ds_category)

        row2.addWidget(QLabel("data_type:"))
        self.ds_data_type = QComboBox()
        self.ds_data_type.addItems(["grid", "stn"])
        row2.addWidget(self.ds_data_type)

        row2.addWidget(QLabel("tim_res:"))
        self.ds_tim_res = QComboBox()
        self.ds_tim_res.setEditable(True)
        self.ds_tim_res.addItems(["Month", "Day", "Hour", "3Hour", "6Hour"])
        row2.addWidget(self.ds_tim_res)

        row2.addWidget(QLabel("grid_res:"))
        self.ds_grid_res = QLineEdit()
        self.ds_grid_res.setPlaceholderText("e.g. 0.25")
        self.ds_grid_res.setMaximumWidth(80)
        row2.addWidget(self.ds_grid_res)
        editor_form.addLayout(row2)

        # Row 3: root_dir, data_groupby, timezone
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("root_dir:"))
        self.ds_root_dir = QLineEdit()
        self.ds_root_dir.setPlaceholderText("/path/to/reference/data")
        row3.addWidget(self.ds_root_dir, stretch=1)
        btn_browse_root = QPushButton("Browse")
        btn_browse_root.clicked.connect(self._browse_ds_root)
        row3.addWidget(btn_browse_root)

        row3.addWidget(QLabel("data_groupby:"))
        self.ds_data_groupby = QComboBox()
        self.ds_data_groupby.setEditable(True)
        self.ds_data_groupby.addItems(["Year", "Month", "Single"])
        row3.addWidget(self.ds_data_groupby)

        row3.addWidget(QLabel("timezone:"))
        self.ds_timezone = QLineEdit("0")
        self.ds_timezone.setMaximumWidth(50)
        row3.addWidget(self.ds_timezone)
        editor_form.addLayout(row3)

        # Variable table
        editor_form.addWidget(QLabel("Variables:"))
        self.ds_var_table = QTableWidget(0, 6)
        self.ds_var_table.setHorizontalHeaderLabels(
            [
                "Variable",
                "varname",
                "varunit",
                "sub_dir",
                "prefix",
                "suffix",
            ]
        )
        hdr = self.ds_var_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.ds_var_table.setSelectionBehavior(QTableWidget.SelectRows)
        editor_form.addWidget(self.ds_var_table, stretch=1)

        var_btn_row = QHBoxLayout()
        btn_add_var = QPushButton("+ Add Variable")
        btn_add_var.clicked.connect(self._ds_add_variable)
        var_btn_row.addWidget(btn_add_var)

        btn_rm_var = QPushButton("Remove Selected")
        btn_rm_var.clicked.connect(self._ds_remove_variable)
        var_btn_row.addWidget(btn_rm_var)

        btn_edit_var = QPushButton("Edit Variable...")
        btn_edit_var.clicked.connect(self._ds_edit_variable)
        var_btn_row.addWidget(btn_edit_var)

        var_btn_row.addStretch()
        editor_form.addLayout(var_btn_row)

        # Save / Revert
        save_row = QHBoxLayout()
        save_row.addStretch()
        btn_revert = QPushButton("Revert")
        btn_revert.clicked.connect(self._ds_revert)
        save_row.addWidget(btn_revert)
        btn_save = QPushButton("Save")
        btn_save.clicked.connect(self._ds_save)
        save_row.addWidget(btn_save)
        editor_form.addLayout(save_row)

        right_layout.addWidget(editor_group)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)

    # ------------------------------------------------------------------
    # Dataset list helpers
    # ------------------------------------------------------------------

    _RES_SUFFIXES = ("_LowRes", "_MidRes", "_HigRes", "_HighRes")

    def _strip_res_suffix(self, name: str) -> tuple:
        """Return (base_name, suffix) — suffix is empty for standalone entries."""
        for suf in self._RES_SUFFIXES:
            if name.endswith(suf):
                return name[: -len(suf)], suf[1:]  # strip leading '_'
        return name, ""

    def _refresh_dataset_list(self):
        """Populate the dataset list, grouping resolution variants under base name."""
        self.dataset_list.clear()
        # _ds_group_map: base_name → [(full_name, suffix_label, ref)]
        self._ds_group_map: Dict[str, list] = {}
        try:
            refs = _get_registry().list_references()
            for r in refs:
                base, suffix = self._strip_res_suffix(r.name)
                self._ds_group_map.setdefault(base, []).append((r.name, suffix, r))

            for base in sorted(self._ds_group_map.keys()):
                variants = self._ds_group_map[base]
                n_vars = len(variants[0][2].variables)
                dtype = variants[0][2].data_type

                if len(variants) == 1 and not variants[0][1]:
                    # Standalone entry (no resolution suffix)
                    label = f"{base}  ({dtype}, {n_vars} vars)"
                elif len(variants) == 1:
                    # Single resolution variant
                    label = f"{base}  ({variants[0][1]} {dtype}, {n_vars} vars)"
                else:
                    # Multiple resolution variants
                    res_labels = " / ".join(v[1] for v in sorted(variants, key=lambda x: x[0]) if v[1])
                    label = f"{base}  ({res_labels}, {dtype}, {n_vars} vars)"

                item = QListWidgetItem(label)
                # Store the first variant's full name as default, plus base for group lookup
                item.setData(Qt.UserRole, variants[0][0])
                item.setData(Qt.UserRole + 1, base)
                self.dataset_list.addItem(item)
        except Exception as exc:
            logger.warning("Failed to list references: %s", exc)

    def _on_dataset_selected(self, row: int):
        if row < 0:
            return
        item = self.dataset_list.item(row)
        if item is None:
            return

        base = item.data(Qt.UserRole + 1) or ""
        variants = self._ds_group_map.get(base, [])

        if len(variants) > 1:
            # Let user pick which resolution variant to edit
            from PySide6.QtWidgets import QInputDialog

            names = [v[0] for v in variants]
            chosen, ok = QInputDialog.getItem(
                self,
                "Select Resolution",
                f"'{base}' has multiple resolutions.\nEdit which one?",
                names,
                0,
                False,
            )
            if not ok:
                return
            name = chosen
        else:
            name = item.data(Qt.UserRole)

        ref = _get_registry().get_reference(name)
        if ref is None:
            return
        self._populate_dataset_editor(ref)

    def _populate_dataset_editor(self, ref):
        self.ds_name.setText(ref.name)
        self.ds_desc.setText(ref.description)

        # category
        idx = self.ds_category.findText(ref.category)
        if idx >= 0:
            self.ds_category.setCurrentIndex(idx)
        else:
            self.ds_category.setEditText(ref.category)

        # data_type
        idx = self.ds_data_type.findText(ref.data_type)
        if idx >= 0:
            self.ds_data_type.setCurrentIndex(idx)

        # tim_res
        idx = self.ds_tim_res.findText(ref.tim_res)
        if idx >= 0:
            self.ds_tim_res.setCurrentIndex(idx)
        else:
            self.ds_tim_res.setEditText(ref.tim_res)

        self.ds_grid_res.setText(str(ref.grid_res) if ref.grid_res is not None else "")
        self.ds_root_dir.setText(ref.root_dir or "")

        # data_groupby
        idx = self.ds_data_groupby.findText(ref.data_groupby)
        if idx >= 0:
            self.ds_data_groupby.setCurrentIndex(idx)
        else:
            self.ds_data_groupby.setEditText(ref.data_groupby)

        self.ds_timezone.setText(str(ref.timezone))

        # Variables
        self.ds_var_table.setRowCount(0)
        for var_name, vm in ref.variables.items():
            row = self.ds_var_table.rowCount()
            self.ds_var_table.insertRow(row)
            name_item = QTableWidgetItem(var_name)
            _set_row_fallbacks(
                name_item,
                [fb.to_dict() for fb in (vm.fallbacks or [])],
            )
            self.ds_var_table.setItem(row, 0, name_item)
            self.ds_var_table.setItem(row, 1, QTableWidgetItem(str(vm.varname)))
            self.ds_var_table.setItem(row, 2, QTableWidgetItem(vm.varunit))
            self.ds_var_table.setItem(row, 3, QTableWidgetItem(vm.sub_dir or ""))
            self.ds_var_table.setItem(row, 4, QTableWidgetItem(vm.prefix))
            self.ds_var_table.setItem(row, 5, QTableWidgetItem(vm.suffix))

    # ------------------------------------------------------------------
    # Dataset actions
    # ------------------------------------------------------------------

    def _new_dataset(self):
        self.ds_name.clear()
        self.ds_desc.clear()
        self.ds_category.setCurrentIndex(0)
        self.ds_data_type.setCurrentIndex(0)
        self.ds_tim_res.setCurrentIndex(0)
        self.ds_grid_res.clear()
        self.ds_root_dir.clear()
        self.ds_data_groupby.setCurrentIndex(0)
        self.ds_timezone.setText("0")
        self.ds_var_table.setRowCount(0)
        self.dataset_list.clearSelection()

    def _scan_directory(self):
        if getattr(self, "_scan_worker", None) is not None:
            return
        path = browse_directory(self.controller, self, "Select Directory to Scan")
        if not path:
            return
        progress = QProgressDialog("Scanning reference datasets...", None, 0, 0, self)
        progress.setWindowTitle("Scanning")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)

        from openbench.gui.pages._scan_worker import FindDatasetsWorker
        from openbench.gui.path_utils import remote_exec_context

        worker_kwargs = remote_exec_context(self.controller, self)
        if worker_kwargs is None:
            progress.close()
            progress.deleteLater()
            return
        self._scan_was_remote = bool(worker_kwargs)

        worker = FindDatasetsWorker(path, **worker_kwargs)
        self._scan_worker = worker
        self._scan_progress = progress
        worker.finished_with_result.connect(self._on_scan_directory_finished)
        worker.failed.connect(self._on_scan_directory_failed)
        worker.finished.connect(worker.deleteLater)
        worker.start()
        progress.show()

    def _finish_scan_worker(self, cancel: bool = False):
        progress = getattr(self, "_scan_progress", None)
        if progress is not None:
            progress.close()
            progress.deleteLater()
        worker = getattr(self, "_scan_worker", None)
        if cancel and worker is not None and worker.isRunning():
            for signal, slot in (
                (worker.finished_with_result, self._on_scan_directory_finished),
                (worker.failed, self._on_scan_directory_failed),
            ):
                try:
                    signal.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
            worker.requestInterruption()
            worker.quit()
            worker.wait(3000)
        if worker is not None and worker.isRunning():
            from openbench.gui.widgets._task_worker import detach_worker

            detach_worker(worker, _DETACHED_SCAN_WORKERS)
        self._scan_progress = None
        self._scan_worker = None

    def closeEvent(self, event):
        self._finish_scan_worker(cancel=True)
        super().closeEvent(event)

    def _on_scan_directory_failed(self, message: str):
        self._finish_scan_worker()
        QMessageBox.critical(self, "Scan Failed", f"Error scanning:\n{message}")
        logger.error("Directory scan failed: %s", message)

    def _on_scan_directory_finished(self, new_groups):
        self._finish_scan_worker()
        try:
            from openbench.data.registry.scanner import register_scanned_datasets_batch

            if not new_groups:
                QMessageBox.information(
                    self, "Scan Complete", "No new datasets found. All datasets already registered."
                )
                return

            # Show discovery dialog for user to select which datasets to register
            try:
                from openbench.gui.dialogs.data_discovery import DataDiscoveryDialog, choose_nc_variable

                dlg = DataDiscoveryDialog(new_groups, parent=self)
                if not dlg.exec():
                    return
                selected = dlg.get_selected()
                if not selected:
                    return
                variants = [variant for _base, _res, variant in selected]
                multi_var_picker = lambda var_name, sub_dir, all_vars: choose_nc_variable(
                    self, var_name, sub_dir, all_vars
                )
            except ImportError:
                # Fallback: register all if dialog not available and choose the
                # first inspected data variable for multi-variable files.
                variants = []
                for group in new_groups:
                    for _res, variant in group.variants.items():
                        variants.append(variant)
                multi_var_picker = lambda _var_name, _sub_dir, all_vars: all_vars[0]["name"] if all_vars else None

            register_scanned_datasets_batch(
                variants,
                on_multi_var=multi_var_picker,
            )
            _clear_cache()
            self._refresh_dataset_list()
            message = f"Registered {len(variants)} new dataset(s)."
            if getattr(self, "_scan_was_remote", False):
                from openbench.gui.pages._scan_worker import remote_scan_caveats

                caveats = remote_scan_caveats(variants)
                if caveats:
                    message += f"\n\n{caveats}"
            QMessageBox.information(self, "Scan Complete", message)
        except Exception as exc:
            QMessageBox.critical(self, "Scan Failed", f"Error scanning:\n{exc}")
            logger.exception("Directory scan registration failed")

    def _delete_dataset(self):
        item = self.dataset_list.currentItem()
        if item is None:
            return
        name = item.data(Qt.UserRole)
        reply = QMessageBox.question(
            self,
            "Delete Dataset",
            f"Delete reference dataset '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            try:
                _get_registry().delete_reference(name)
                _clear_cache()
                self._refresh_dataset_list()
                self._new_dataset()
            except Exception as exc:
                QMessageBox.critical(self, "Error", f"Failed to delete dataset:\n{exc}")

    def _browse_ds_root(self):
        path = browse_directory(self.controller, self, "Select root_dir", self.ds_root_dir.text().strip())
        if path:
            self.ds_root_dir.setText(path)

    def _ds_add_variable(self):
        from openbench.gui.widgets.variable_editor import VariableEditorDialog

        dlg = VariableEditorDialog(mode="reference", parent=self)
        if dlg.exec():
            data = dlg.get_data()
            row = self.ds_var_table.rowCount()
            self.ds_var_table.insertRow(row)
            name_item = QTableWidgetItem(data.get("variable_name", ""))
            _set_row_fallbacks(name_item, data.get("fallbacks") or [])
            self.ds_var_table.setItem(row, 0, name_item)
            self.ds_var_table.setItem(row, 1, QTableWidgetItem(data.get("varname", "")))
            self.ds_var_table.setItem(row, 2, QTableWidgetItem(data.get("varunit", "")))
            self.ds_var_table.setItem(row, 3, QTableWidgetItem(data.get("sub_dir", "")))
            self.ds_var_table.setItem(row, 4, QTableWidgetItem(data.get("prefix", "")))
            self.ds_var_table.setItem(row, 5, QTableWidgetItem(data.get("suffix", "")))

    def _ds_remove_variable(self):
        rows = sorted({idx.row() for idx in self.ds_var_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.ds_var_table.removeRow(r)

    def _ds_edit_variable(self):
        row = self.ds_var_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "No Selection", "Select a variable row first.")
            return
        from openbench.gui.widgets.variable_editor import VariableEditorDialog

        def _cell(c):
            item = self.ds_var_table.item(row, c)
            return item.text() if item else ""

        existing_fallbacks = _get_row_fallbacks(self.ds_var_table, row)
        dlg = VariableEditorDialog(
            mode="reference",
            variable_name=_cell(0),
            varname=_cell(1),
            varunit=_cell(2),
            sub_dir=_cell(3),
            prefix=_cell(4),
            suffix=_cell(5),
            fallbacks=existing_fallbacks,
            parent=self,
        )
        if dlg.exec():
            data = dlg.get_data()
            name_item = QTableWidgetItem(data.get("variable_name", ""))
            _set_row_fallbacks(name_item, data.get("fallbacks") or [])
            self.ds_var_table.setItem(row, 0, name_item)
            self.ds_var_table.setItem(row, 1, QTableWidgetItem(data.get("varname", "")))
            self.ds_var_table.setItem(row, 2, QTableWidgetItem(data.get("varunit", "")))
            self.ds_var_table.setItem(row, 3, QTableWidgetItem(data.get("sub_dir", "")))
            self.ds_var_table.setItem(row, 4, QTableWidgetItem(data.get("prefix", "")))
            self.ds_var_table.setItem(row, 5, QTableWidgetItem(data.get("suffix", "")))

    def _ds_revert(self):
        item = self.dataset_list.currentItem()
        if item is None:
            return
        name = item.data(Qt.UserRole)
        ref = _get_registry().get_reference(name)
        if ref:
            self._populate_dataset_editor(ref)

    def _ds_save(self):
        """Build a ReferenceDataset from the editor and save it."""
        schema = _schema()
        name = self.ds_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Name Required", "Please enter a dataset name.")
            return

        tim_res = self.ds_tim_res.currentText().strip()
        if not tim_res:
            QMessageBox.warning(self, "tim_res Required", "Please select a time resolution.")
            return

        data_type = self.ds_data_type.currentText().strip()
        if not data_type:
            QMessageBox.warning(self, "data_type Required", "Please select a data type.")
            return

        grid_res_text = self.ds_grid_res.text().strip()
        if grid_res_text:
            try:
                grid_res = float(grid_res_text)
            except ValueError:
                QMessageBox.warning(
                    self,
                    "Invalid grid_res",
                    f"grid_res must be a number, got: {grid_res_text!r}",
                )
                return
        else:
            grid_res = None

        tz_text = self.ds_timezone.text().strip()
        try:
            timezone = int(tz_text) if tz_text else 0
        except ValueError:
            try:
                timezone = float(tz_text)
            except ValueError:
                timezone = 0

        variables: dict = {}
        for row in range(self.ds_var_table.rowCount()):
            # Bind `row` via default argument — without this, all closures
            # share the loop variable and read from the last row only.
            def _cell(c, _row=row):
                item = self.ds_var_table.item(_row, c)
                return item.text().strip() if item else ""

            var_key = _cell(0)
            if not var_key:
                continue
            fallback_dicts = _get_row_fallbacks(self.ds_var_table, row)
            fallbacks = [
                schema.FallbackVar(
                    varname=fb.get("varname", ""),
                    varunit=fb.get("varunit", ""),
                    convert=fb.get("convert", ""),
                )
                for fb in fallback_dicts
                if fb.get("varname")
            ] or None
            vm = schema.VariableMapping(
                varname=_cell(1),
                varunit=_cell(2),
                sub_dir=_cell(3) or None,
                prefix=_cell(4),
                suffix=_cell(5),
                fallbacks=fallbacks,
            )
            variables[var_key] = vm

        dataset = schema.ReferenceDataset(
            name=name,
            description=self.ds_desc.text().strip(),
            category=self.ds_category.currentText().strip(),
            data_type=data_type,
            tim_res=tim_res,
            data_groupby=self.ds_data_groupby.currentText().strip(),
            timezone=timezone,
            years=[],
            variables=variables,
            grid_res=grid_res,
            root_dir=self.ds_root_dir.text().strip() or None,
        )

        try:
            _get_registry().save_reference(name, dataset)
            _clear_cache()
            self._refresh_dataset_list()
            QMessageBox.information(self, "Saved", f"Dataset '{name}' saved to registry.")
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", f"Failed to save dataset:\n{exc}")

    # ==================================================================
    # BasePage interface (no-ops: this page does not use project config)
    # ==================================================================

    def load_from_config(self):
        """Refresh lists from registry (this page is registry-only, not config-driven)."""
        self._refresh_model_list()
        self._refresh_dataset_list()

    def save_to_config(self):
        """No-op: registry page does not write to the project config dict."""
        pass
