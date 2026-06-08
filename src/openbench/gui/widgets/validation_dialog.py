# ui/widgets/validation_dialog.py
# -*- coding: utf-8 -*-
"""
Dialogs for data validation progress and results.
"""

from copy import deepcopy
from typing import Optional
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QDialogButtonBox,
    QFileDialog,
    QMessageBox,
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QColor

from openbench.gui.data_validator import DataValidator, DataValidationReport


class ValidationWorker(QThread):
    """Worker thread for running validation."""

    progress = Signal(int, int, str, str)  # current, total, var_name, source_name
    finished = Signal(object)  # DataValidationReport
    error = Signal(str)

    def __init__(self, validator: DataValidator, sources: dict, general_config: dict, parent=None):
        super().__init__(parent)
        self._validator = validator
        # Validate an immutable snapshot. GUI pages save opportunistically on
        # field changes/navigation, and those mutations can otherwise race the
        # worker thread while it is iterating nested source dictionaries.
        self._sources = deepcopy(sources)
        self._general_config = deepcopy(general_config)
        self._cancelled = False

    def run(self):
        """Run validation in background thread."""
        try:

            def progress_callback(current, total, var_name, source_name):
                if self._cancelled:
                    raise InterruptedError("Cancelled")
                self.progress.emit(current, total, var_name, source_name)

            report = self._validator.validate_all(self._sources, self._general_config, progress_callback)
            self.finished.emit(report)
        except InterruptedError:
            pass
        except Exception as e:
            self.error.emit(str(e))

    def cancel(self):
        """Request cancellation."""
        self._cancelled = True


class ValidationProgressDialog(QDialog):
    """Dialog showing validation progress."""

    def __init__(self, validator: DataValidator, sources: dict, general_config: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Validating Data...")
        self.setModal(True)
        self.setMinimumWidth(400)
        self.setMinimumHeight(150)

        self._report: Optional[DataValidationReport] = None
        self._closing = False
        self._worker: Optional[ValidationWorker] = None
        self._setup_ui()

        # Start worker - don't set parent to avoid Qt destruction issues
        self._worker = ValidationWorker(validator, sources, general_config, None)
        self._worker.progress.connect(self._on_progress, Qt.QueuedConnection)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.error.connect(self._on_error, Qt.QueuedConnection)
        self._worker.start()

    def _setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)

        # Progress label
        self.progress_label = QLabel("Preparing...")
        layout.addWidget(self.progress_label)

        # Current item label
        self.current_label = QLabel("")
        self.current_label.setStyleSheet("color: #666;")
        layout.addWidget(self.current_label)

        # Cancel button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

    def _on_progress(self, current: int, total: int, var_name: str, source_name: str):
        """Handle progress update."""
        if self._closing:
            return

        total = max(0, int(total))
        current = max(0, min(int(current), total))

        if total > 0:
            percent = int(current / total * 100)
            self.progress_bar.setValue(percent)
            self.progress_label.setText(f"{current}/{total}")

        if var_name and source_name:
            self.current_label.setText(f"Current: {var_name} / {source_name}")

    def _on_finished(self, report: DataValidationReport):
        """Handle validation finished."""
        if self._closing:
            return

        self._report = report
        self._cleanup_worker()
        self.accept()

    def _on_error(self, error: str):
        """Handle validation error."""
        if self._closing:
            return

        self._cleanup_worker()
        QMessageBox.warning(self, "Validation Error", f"Error during validation:\n{error}")
        self.reject()

    def _on_cancel(self):
        """Handle cancel button."""
        self._closing = True
        self._cleanup_worker()
        self.reject()

    def _cleanup_worker(self):
        """Clean up worker thread."""
        if self._worker is not None:
            # Disconnect signals first to prevent any more callbacks
            try:
                self._worker.progress.disconnect()
                self._worker.finished.disconnect()
                self._worker.error.disconnect()
            except RuntimeError:
                pass  # Already disconnected

            # Request cancellation and wait for thread to finish
            self._worker.cancel()
            if self._worker.isRunning():
                self._worker.wait(3000)
                if self._worker.isRunning():
                    # Force terminate if still running
                    self._worker.terminate()
                    self._worker.wait(1000)

            # Schedule deletion to ensure all events are processed
            self._worker.deleteLater()
            self._worker = None

    def closeEvent(self, event):
        """Handle dialog close event."""
        self._closing = True
        self._cleanup_worker()
        super().closeEvent(event)

    def get_report(self) -> Optional[DataValidationReport]:
        """Get validation report after dialog closes."""
        return self._report


class ValidationResultsDialog(QDialog):
    """Dialog showing validation results."""

    def __init__(self, report: DataValidationReport, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Validation Results")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)

        self._report = report
        self._setup_ui()

    def _setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Summary
        summary = QLabel(f"Validation complete: {self._report.passed_count} passed, {self._report.failed_count} failed")
        summary.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(summary)

        # Results tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Data Source", "Status"])
        self.tree.setColumnWidth(0, 400)
        self._populate_tree()
        layout.addWidget(self.tree)

        # Buttons
        btn_box = QDialogButtonBox()
        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self._export_results)
        btn_box.addButton(self.export_btn, QDialogButtonBox.ActionRole)
        btn_box.addButton(QDialogButtonBox.Ok)
        btn_box.accepted.connect(self.accept)
        layout.addWidget(btn_box)

    def _populate_tree(self):
        """Populate results tree."""
        for result in self._report.results:
            # Create source item
            status = "✓" if result.is_valid else "✗"
            item = QTreeWidgetItem([f"{result.var_name} / {result.source_name}", status])

            if result.is_valid:
                item.setForeground(1, QColor("#27ae60"))
            else:
                item.setForeground(1, QColor("#e74c3c"))

            # Add failed checks as children
            for check in result.failed_checks:
                child = QTreeWidgetItem([f"  └─ {check.message}", ""])
                child.setForeground(0, QColor("#666"))
                item.addChild(child)

            self.tree.addTopLevelItem(item)

        self.tree.expandAll()

    def _export_results(self):
        """Export results to text file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Validation Results", "validation_results.txt", "Text Files (*.txt)"
        )
        if not file_path:
            return

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("Data Validation Results\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Passed: {self._report.passed_count}\n")
                f.write(f"Failed: {self._report.failed_count}\n\n")

                for result in self._report.results:
                    status = "✓ Passed" if result.is_valid else "✗ Failed"
                    f.write(f"{result.var_name} / {result.source_name}: {status}\n")
                    for check in result.failed_checks:
                        f.write(f"  - {check.message}\n")
                    f.write("\n")

            QMessageBox.information(self, "Export Successful", f"Results exported to:\n{file_path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Failed", f"Failed to export results:\n{e}")
