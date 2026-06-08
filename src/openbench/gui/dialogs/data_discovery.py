"""Auto-discovery dialog for reference datasets.

Shown on GUI startup when new datasets are found in the data directory.
Allows users to review discovered datasets and choose which to register.
"""

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QDialogButtonBox,
    QPushButton,
    QInputDialog,
)
from PySide6.QtCore import Qt


def choose_nc_variable(parent, var_name, sub_dir, all_vars):
    """Prompt for the NetCDF variable to register when a file has multiple data vars."""
    if not all_vars:
        return None

    labels = []
    label_to_name = {}
    for var in all_vars:
        name = str(var.get("name", ""))
        unit = str(var.get("unit", ""))
        dims = ", ".join(str(d) for d in var.get("dims", []))
        desc = var.get("long_name") or var.get("standard_name") or ""
        label = f"{name} [{unit}] ({dims})"
        if desc:
            label = f"{label} - {desc}"
        labels.append(label)
        label_to_name[label] = name

    selected, ok = QInputDialog.getItem(
        parent,
        "Multiple variables found",
        f"Select NetCDF variable for {var_name}\n{sub_dir}",
        labels,
        0,
        False,
    )
    if not ok:
        raise RuntimeError(f"Variable selection cancelled for {var_name} in {sub_dir}")
    return label_to_name.get(selected)


class DataDiscoveryDialog(QDialog):
    """Dialog showing newly discovered reference datasets for registration."""

    def __init__(self, new_groups, parent=None):
        """
        Args:
            new_groups: List of DatasetGroup from scanner.find_new_datasets().
        """
        super().__init__(parent)
        self.setWindowTitle("New Reference Datasets Found")
        self.setMinimumSize(800, 500)
        self._new_groups = new_groups
        self._checkboxes = {}  # (base_name, resolution) -> QCheckBox

        layout = QVBoxLayout(self)

        # Header
        header = QLabel(
            f"<b>{len(new_groups)} new dataset(s)</b> found in the reference data directory.\n"
            "Select which to register into the OpenBench registry."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        # Tree view
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Dataset", "Resolution", "Type", "Variables", "Files"])
        self.tree.setColumnWidth(0, 250)
        self.tree.setColumnWidth(1, 100)
        self.tree.setColumnWidth(2, 60)
        self.tree.setColumnWidth(3, 50)
        self.tree.setColumnWidth(4, 50)

        for group in new_groups:
            # Parent item: dataset base name
            parent_item = QTreeWidgetItem(
                [
                    group.base_name,
                    f"{len(group.variants)} variant(s)",
                    group.category,
                    "",
                    "",
                ]
            )
            parent_item.setFlags(parent_item.flags() | Qt.ItemIsUserCheckable)
            parent_item.setCheckState(0, Qt.Checked)

            for res_name, variant in sorted(group.variants.items()):
                child = QTreeWidgetItem(
                    [
                        "",
                        res_name,
                        variant.data_type,
                        str(len(variant.variables)),
                        str(variant.file_count),
                    ]
                )
                child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                child.setCheckState(0, Qt.Checked)
                child.setData(0, Qt.UserRole, (group.base_name, res_name))
                parent_item.addChild(child)

            self.tree.addTopLevelItem(parent_item)
            parent_item.setExpanded(True)

        layout.addWidget(self.tree)

        # Buttons
        btn_layout = QHBoxLayout()
        select_all = QPushButton("Select All")
        select_all.clicked.connect(self._select_all)
        deselect_all = QPushButton("Deselect All")
        deselect_all.clicked.connect(self._deselect_all)
        btn_layout.addWidget(select_all)
        btn_layout.addWidget(deselect_all)
        btn_layout.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Register Selected")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        btn_layout.addWidget(buttons)

        layout.addLayout(btn_layout)

    def _select_all(self):
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            item.setCheckState(0, Qt.Checked)
            for j in range(item.childCount()):
                item.child(j).setCheckState(0, Qt.Checked)

    def _deselect_all(self):
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            item.setCheckState(0, Qt.Unchecked)
            for j in range(item.childCount()):
                item.child(j).setCheckState(0, Qt.Unchecked)

    def get_selected(self):
        """Return list of (base_name, resolution, ScannedDataset) for checked items."""
        selected = []
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            group = self._new_groups[i]

            for j in range(parent.childCount()):
                child = parent.child(j)
                if child.checkState(0) == Qt.Checked:
                    data = child.data(0, Qt.UserRole)
                    if data:
                        base_name, res_name = data
                        variant = group.variants.get(res_name)
                        if variant:
                            selected.append((base_name, res_name, variant))

        return selected


class ResolutionPickerDialog(QDialog):
    """Dialog for choosing a resolution variant when multiple are available."""

    def __init__(self, base_name, variants, compatible=None, parent=None):
        """
        Args:
            base_name: Dataset base name (e.g., "GLEAM_v4.2a")
            variants: Dict[resolution_name -> ScannedDataset or registry entry]
            compatible: Optional list of compatible resolution names.
                If provided, incompatible resolutions are shown as disabled.
        """
        super().__init__(parent)
        self.setWindowTitle(f"Select Resolution — {base_name}")
        self.setMinimumWidth(500)
        self._selected = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"<b>{base_name}</b> is available at multiple resolutions:"))

        self._buttons = {}
        for res_name, variant in sorted(variants.items()):
            is_compatible = compatible is None or res_name in compatible

            if hasattr(variant, "tim_res"):
                # ScannedDataset
                info = f"{variant.data_type}, {variant.tim_res or '?'}, {len(variant.variables)} variables, {variant.file_count} files"
            else:
                # Registry entry
                info = f"{variant.data_type}, {variant.tim_res}, grid_res={variant.grid_res}"

            btn = QPushButton(f"{res_name}\n{info}")
            btn.setMinimumHeight(50)
            btn.setEnabled(is_compatible)
            if not is_compatible:
                btn.setToolTip("Not compatible — higher frequency data is available")
            btn.clicked.connect(lambda checked, r=res_name: self._pick(r))
            layout.addWidget(btn)
            self._buttons[res_name] = btn

        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        layout.addWidget(cancel)

    def _pick(self, resolution):
        self._selected = resolution
        self.accept()

    def selected_resolution(self):
        return self._selected
