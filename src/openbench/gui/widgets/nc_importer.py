# -*- coding: utf-8 -*-
"""
Dialog that opens a NetCDF file and lets the user pick variables to import.
"""

import logging
import json
import os

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QLabel,
    QMessageBox,
    QFileDialog,
    QCheckBox,
)
from PySide6.QtCore import Qt

logger = logging.getLogger(__name__)

# Coordinate-like variable names that are auto-unchecked on import
_COORD_NAMES = frozenset(
    {
        "lat",
        "latitude",
        "lat_bnds",
        "latitude_bnds",
        "lon",
        "longitude",
        "lon_bnds",
        "longitude_bnds",
        "time",
        "time_bnds",
        "level",
        "lev",
        "depth",
        "height",
        "z",
        "x",
        "y",
        "nav_lat",
        "nav_lon",
        "bounds_lat",
        "bounds_lon",
        "crs",
        "spatial_ref",
    }
)


def _variable_rows(ds, coord_names=None):
    """Extract import-table rows from an xarray Dataset.

    Shared by the local table and the remote inspector script (embedded via
    its source) so both classify variables and coordinates identically.
    """
    coord_names = (coord_names or set()) | set(str(name) for name in ds.coords) | _COORD_NAMES
    rows = []
    for name, var in ds.data_vars.items():
        rows.append(
            {
                "name": str(name),
                "dtype": str(var.dtype),
                "dims": [[str(dim), int(size)] for dim, size in zip(var.dims, var.shape)],
                "units": str(var.attrs.get("units", var.attrs.get("unit", ""))),
                "is_coord": str(name).lower() in coord_names or str(name) in coord_names,
            }
        )
    for name, var in ds.coords.items():
        if name in ds.data_vars or len(var.dims) <= 1:
            continue
        rows.append(
            {
                "name": str(name),
                "dtype": str(var.dtype),
                "dims": [[str(dim), int(size)] for dim, size in zip(var.dims, var.shape)],
                "units": str(var.attrs.get("units", var.attrs.get("unit", ""))),
                "is_coord": True,
            }
        )
    return rows


class NCImporterDialog(QDialog):
    """Dialog to open a NetCDF file and select variables for import.

    After the user clicks *Import Selected*, call :meth:`get_selected_variables`
    to retrieve the chosen variables as a list of dicts.
    """

    def __init__(self, parent=None, ssh_manager=None, python_path: str = "", conda_env: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Import Variables from NetCDF")
        self.setMinimumSize(680, 480)
        self.setModal(True)

        self._selected: list[dict] = []
        self._ssh_manager = ssh_manager
        self._python_path = python_path
        self._conda_env = conda_env
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # --- File picker ---
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("NC file:"))
        self.edit_path = QLineEdit()
        self.edit_path.setPlaceholderText("Path to .nc file")
        file_layout.addWidget(self.edit_path, stretch=1)

        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse)
        file_layout.addWidget(btn_browse)

        btn_open = QPushButton("Open")
        btn_open.clicked.connect(self._open_file)
        file_layout.addWidget(btn_open)

        layout.addLayout(file_layout)

        # --- Info label ---
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.info_label)

        # --- Variable table ---
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["", "Variable", "dtype", "Dimensions", "Units"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.table, stretch=1)

        # --- Select / Deselect buttons ---
        sel_layout = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_all.clicked.connect(self._select_all)
        sel_layout.addWidget(btn_all)

        btn_none = QPushButton("Deselect All")
        btn_none.clicked.connect(self._deselect_all)
        sel_layout.addWidget(btn_none)
        sel_layout.addStretch()
        layout.addLayout(sel_layout)

        # --- Import / Cancel ---
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        btn_import = QPushButton("Import Selected")
        btn_import.clicked.connect(self._do_import)
        btn_layout.addWidget(btn_import)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)

        layout.addLayout(btn_layout)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _browse(self):
        if self._is_remote_mode():
            from openbench.gui.path_utils import pick_remote_path, remote_home_dir

            current = self.edit_path.text().strip()
            start = os.path.dirname(current) if current else remote_home_dir(self._ssh_manager)
            path = pick_remote_path(self._ssh_manager, self, "Select NetCDF File", start, select_dirs=False)
            if path:
                self.edit_path.setText(path)
                self._open_file()
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Select NetCDF File", "", "NetCDF files (*.nc *.nc4 *.hdf5);;All files (*)"
        )
        if path:
            self.edit_path.setText(path)
            self._open_file()

    def _open_file(self):
        if getattr(self, "_busy", False):
            return
        path = self.edit_path.text().strip()
        if not path:
            QMessageBox.warning(self, "No File", "Please enter or browse for a NetCDF file.")
            return

        if self._is_remote_mode():
            # The remote inspection spins a nested event loop (up to 60s);
            # keep the dialog inert so a second click cannot overwrite the
            # fresh result with a stale one.
            self._busy = True
            self.setEnabled(False)
            try:
                payload = self._open_remote_file(path)
                self.info_label.setText(
                    f"Opened: {payload.get('path', path)}  ({payload.get('data_var_count', 0)} data variables)"
                )
                self._populate_rows(payload.get("variables") or [])
            except Exception as exc:
                QMessageBox.critical(self, "Open Failed", f"Could not open remote file:\n{exc}")
            finally:
                self._busy = False
                self.setEnabled(True)
            return

        try:
            import xarray as xr
        except ImportError:
            QMessageBox.critical(
                self,
                "Missing Dependency",
                "xarray is required to open NetCDF files.\n\nInstall it with: pip install xarray netCDF4",
            )
            return

        try:
            # Use a with-block so the file handle is released even if
            # _populate_table raises mid-iteration.
            with xr.open_dataset(path) as ds:
                self.info_label.setText(f"Opened: {path}  ({len(ds.data_vars)} data variables)")
                self._populate_table(ds)
        except Exception as exc:
            QMessageBox.critical(self, "Open Failed", f"Could not open file:\n{exc}")
            return

    def _is_remote_mode(self) -> bool:
        return self._ssh_manager is not None and getattr(self._ssh_manager, "is_connected", False)

    def _open_remote_file(self, path: str) -> dict:
        import inspect
        import textwrap

        from openbench.gui.remote_python import run_remote_python_json

        # Embed the exact local extraction function so remote and local
        # imports classify variables identically.
        rows_src = textwrap.dedent(inspect.getsource(_variable_rows))
        coord_literal = json.dumps(sorted(_COORD_NAMES))

        script = f"""
import json
import xarray as xr

_COORD_NAMES = set({coord_literal})

{rows_src}

with xr.open_dataset({json.dumps(path)}) as ds:
    payload = {{
        "path": {json.dumps(path)},
        "data_var_count": len(ds.data_vars),
        "variables": _variable_rows(ds),
    }}
print(json.dumps(payload))
"""
        return run_remote_python_json(
            self._ssh_manager,
            script,
            python_path=self._python_path,
            conda_env=self._conda_env,
            timeout=60,
        )

    def _populate_table(self, ds):
        """Fill the table from an xarray Dataset."""
        self._populate_rows(_variable_rows(ds))

    def _populate_rows(self, rows: list[dict]):
        """Fill the table from serialized NetCDF variable metadata."""
        self.table.setRowCount(0)

        for row, record in enumerate(rows):
            self.table.insertRow(row)

            # Checkbox
            chk = QCheckBox()
            # Auto-uncheck coordinates and 1-D variables
            dims = record.get("dims") or []
            is_coord = bool(record.get("is_coord")) or str(record.get("name", "")).lower() in _COORD_NAMES
            is_1d = len(dims) <= 1
            chk.setChecked(not is_coord and not is_1d)
            self.table.setCellWidget(row, 0, chk)

            # Variable name
            item_name = QTableWidgetItem(str(record.get("name", "")))
            item_name.setFlags(item_name.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 1, item_name)

            # dtype
            item_dtype = QTableWidgetItem(str(record.get("dtype", "")))
            item_dtype.setFlags(item_dtype.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 2, item_dtype)

            # dimensions
            dims_str = ", ".join(f"{dim}({size})" for dim, size in dims)
            item_dims = QTableWidgetItem(dims_str)
            item_dims.setFlags(item_dims.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 3, item_dims)

            # units from attributes
            item_units = QTableWidgetItem(str(record.get("units", "")))
            item_units.setFlags(item_units.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 4, item_units)

    def _select_all(self):
        for row in range(self.table.rowCount()):
            chk = self.table.cellWidget(row, 0)
            if isinstance(chk, QCheckBox):
                chk.setChecked(True)

    def _deselect_all(self):
        for row in range(self.table.rowCount()):
            chk = self.table.cellWidget(row, 0)
            if isinstance(chk, QCheckBox):
                chk.setChecked(False)

    def _do_import(self):
        self._selected = self._gather_selected()
        if not self._selected:
            QMessageBox.information(self, "Nothing Selected", "No variables are checked.")
            return
        self.accept()

    def _gather_selected(self) -> list[dict]:
        result = []
        for row in range(self.table.rowCount()):
            chk = self.table.cellWidget(row, 0)
            if isinstance(chk, QCheckBox) and chk.isChecked():
                varname = self.table.item(row, 1).text()
                dims = self.table.item(row, 3).text()
                units = self.table.item(row, 4).text()
                result.append(
                    {
                        "varname": varname,
                        "varunit": units,
                        "dims": dims,
                    }
                )
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_selected_variables(self) -> list[dict]:
        """Return the list of selected variables.

        Each element is a dict with keys ``varname``, ``varunit``, ``dims``.
        """
        return list(self._selected)
