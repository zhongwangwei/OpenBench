# ui/dialogs/project_selector.py
# -*- coding: utf-8 -*-
"""
Project selector dialog for startup.
"""

import os
from typing import Optional
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QStackedWidget, QWidget, QLineEdit, QMessageBox, QFileDialog, QGroupBox
)
from PySide6.QtCore import Qt, Signal

from openbench.remote.storage import ProjectStorage, LocalStorage, RemoteStorage
from openbench.remote.sync import SyncEngine
from openbench.gui.widgets.remote_config import RemoteConfigWidget, RemoteFileBrowser


class ProjectSelectorDialog(QDialog):
    """Dialog for selecting or creating a project."""

    # Signal emitted when project is selected: (storage, project_name)
    project_selected = Signal(object, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OpenBench Wizard - Select Project")
        self.setMinimumSize(600, 500)
        self._storage: Optional[ProjectStorage] = None
        self._project_name = ""

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Select Project Type")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)

        # Mode selection buttons
        mode_layout = QHBoxLayout()

        self._btn_local = QPushButton("Local Project")
        self._btn_local.setMinimumHeight(60)
        self._btn_local.setCheckable(True)
        self._btn_local.setChecked(True)
        self._btn_local.setStyleSheet(self._get_mode_button_style())
        mode_layout.addWidget(self._btn_local)

        self._btn_remote = QPushButton("Remote Project")
        self._btn_remote.setMinimumHeight(60)
        self._btn_remote.setCheckable(True)
        self._btn_remote.setStyleSheet(self._get_mode_button_style())
        mode_layout.addWidget(self._btn_remote)

        layout.addLayout(mode_layout)

        # Stacked widget for mode-specific content
        self._stack = QStackedWidget()
        layout.addWidget(self._stack, 1)

        # Local mode page
        self._local_page = self._create_local_page()
        self._stack.addWidget(self._local_page)

        # Remote mode page
        self._remote_page = self._create_remote_page()
        self._stack.addWidget(self._remote_page)

        # Bottom buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setMinimumWidth(100)
        btn_layout.addWidget(self._btn_cancel)

        self._btn_open = QPushButton("Open Project")
        self._btn_open.setMinimumWidth(120)
        self._btn_open.setDefault(True)
        btn_layout.addWidget(self._btn_open)

        layout.addLayout(btn_layout)

    def _get_mode_button_style(self) -> str:
        return """
            QPushButton {
                background-color: #f0f0f0;
                border: 2px solid #d0d0d0;
                border-radius: 8px;
                font-size: 14px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:checked {
                background-color: #3498db;
                color: white;
                border-color: #2980b9;
            }
        """

    def _create_local_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        # Project directory selection
        dir_group = QGroupBox("Project Directory")
        dir_layout = QVBoxLayout(dir_group)

        path_layout = QHBoxLayout()
        self._local_path = QLineEdit()
        self._local_path.setPlaceholderText("Select project directory containing nml/ folder")
        path_layout.addWidget(self._local_path)

        self._btn_browse_local = QPushButton("Browse...")
        self._btn_browse_local.setFixedWidth(100)
        path_layout.addWidget(self._btn_browse_local)

        dir_layout.addLayout(path_layout)

        # New project option
        new_layout = QHBoxLayout()
        new_layout.addWidget(QLabel("Or create new project:"))
        self._btn_new_local = QPushButton("New Project...")
        new_layout.addWidget(self._btn_new_local)
        new_layout.addStretch()
        dir_layout.addLayout(new_layout)

        layout.addWidget(dir_group)
        layout.addStretch()

        return page

    def _create_remote_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        # SSH Connection
        self._remote_config = RemoteConfigWidget()
        layout.addWidget(self._remote_config)

        # Remote project path
        path_group = QGroupBox("Remote Project Directory")
        path_layout = QVBoxLayout(path_group)

        remote_path_layout = QHBoxLayout()
        self._remote_path = QLineEdit()
        self._remote_path.setPlaceholderText("Select remote project directory")
        self._remote_path.setEnabled(False)
        remote_path_layout.addWidget(self._remote_path)

        self._btn_browse_remote = QPushButton("Browse...")
        self._btn_browse_remote.setFixedWidth(100)
        self._btn_browse_remote.setEnabled(False)
        remote_path_layout.addWidget(self._btn_browse_remote)

        path_layout.addLayout(remote_path_layout)

        # New project option
        new_layout = QHBoxLayout()
        new_layout.addWidget(QLabel("Or create new project:"))
        self._btn_new_remote = QPushButton("New Project...")
        self._btn_new_remote.setEnabled(False)
        new_layout.addWidget(self._btn_new_remote)
        new_layout.addStretch()
        path_layout.addLayout(new_layout)

        layout.addWidget(path_group)
        layout.addStretch()

        return page

    def _connect_signals(self):
        # Mode selection
        self._btn_local.clicked.connect(lambda: self._set_mode("local"))
        self._btn_remote.clicked.connect(lambda: self._set_mode("remote"))

        # Local mode
        self._btn_browse_local.clicked.connect(self._browse_local)
        self._btn_new_local.clicked.connect(self._new_local_project)

        # Remote mode
        self._remote_config.connection_status_changed.connect(self._on_remote_connection_changed)
        self._btn_browse_remote.clicked.connect(self._browse_remote)
        self._btn_new_remote.clicked.connect(self._new_remote_project)

        # Dialog buttons
        self._btn_cancel.clicked.connect(self.reject)
        self._btn_open.clicked.connect(self._open_project)

    def _set_mode(self, mode: str):
        """Set the current mode."""
        is_local = mode == "local"
        self._btn_local.setChecked(is_local)
        self._btn_remote.setChecked(not is_local)
        self._stack.setCurrentIndex(0 if is_local else 1)

    def _browse_local(self):
        """Browse for local project directory."""
        path = QFileDialog.getExistingDirectory(
            self, "Select Project Directory",
            os.path.expanduser("~")
        )
        if path:
            self._local_path.setText(path)

    def _new_local_project(self):
        """Create new local project."""
        path = QFileDialog.getExistingDirectory(
            self, "Select Directory for New Project",
            os.path.expanduser("~")
        )
        if path:
            # Create nml directory
            nml_dir = os.path.join(path, "nml")
            os.makedirs(nml_dir, exist_ok=True)
            self._local_path.setText(path)

    def _on_remote_connection_changed(self, connected: bool):
        """Handle remote connection state change."""
        self._remote_path.setEnabled(connected)
        self._btn_browse_remote.setEnabled(connected)
        self._btn_new_remote.setEnabled(connected)

    def _browse_remote(self):
        """Browse remote server for project directory."""
        ssh_manager = self._remote_config.get_ssh_manager()
        if not ssh_manager or not ssh_manager.is_connected:
            return

        # Get home directory as starting point
        try:
            home = ssh_manager._get_home_dir()
        except Exception:
            home = "/"

        dialog = QDialog(self)
        dialog.setWindowTitle("Select Remote Project Directory")
        dialog.resize(500, 400)

        layout = QVBoxLayout(dialog)
        browser = RemoteFileBrowser(ssh_manager, home, dialog, select_dirs=True)
        layout.addWidget(browser)

        selected = [None]

        def on_selected(path):
            selected[0] = path
            dialog.accept()

        browser.file_selected.connect(on_selected)

        if dialog.exec() and selected[0]:
            self._remote_path.setText(selected[0])

    def _new_remote_project(self):
        """Create new remote project."""
        ssh_manager = self._remote_config.get_ssh_manager()
        if not ssh_manager or not ssh_manager.is_connected:
            return

        # First browse for parent directory
        self._browse_remote()

        if self._remote_path.text():
            path = self._remote_path.text()
            # Create nml directory on remote
            nml_path = f"{path.rstrip('/')}/nml"
            ssh_manager.execute(f"mkdir -p '{nml_path}'", timeout=10)

    def _open_project(self):
        """Open the selected project."""
        if self._btn_local.isChecked():
            self._open_local_project()
        else:
            self._open_remote_project()

    def _open_local_project(self):
        """Open local project."""
        path = self._local_path.text().strip()
        if not path:
            QMessageBox.warning(self, "Error", "Please select a project directory.")
            return

        if not os.path.isdir(path):
            QMessageBox.warning(self, "Error", "Directory does not exist.")
            return

        # Check for nml directory
        nml_dir = os.path.join(path, "nml")
        if not os.path.isdir(nml_dir):
            reply = QMessageBox.question(
                self, "Create Project",
                "No nml/ directory found. Create new project?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                os.makedirs(nml_dir, exist_ok=True)
            else:
                return

        # Create storage
        self._storage = LocalStorage(path)
        self._project_name = os.path.basename(path)

        self.accept()

    def _open_remote_project(self):
        """Open remote project."""
        ssh_manager = self._remote_config.get_ssh_manager()
        if not ssh_manager or not ssh_manager.is_connected:
            QMessageBox.warning(self, "Error", "Please connect to remote server first.")
            return

        path = self._remote_path.text().strip()
        if not path:
            QMessageBox.warning(self, "Error", "Please select a remote project directory.")
            return

        # Check for nml directory
        stdout, stderr, exit_code = ssh_manager.execute(
            f"test -d '{path}/nml' && echo 'exists'", timeout=10
        )

        if 'exists' not in stdout:
            reply = QMessageBox.question(
                self, "Create Project",
                "No nml/ directory found. Create new project?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                ssh_manager.execute(f"mkdir -p '{path}/nml'", timeout=10)
            else:
                return

        # Create sync engine and storage
        sync_engine = SyncEngine(ssh_manager, path)
        self._storage = RemoteStorage(path, sync_engine)
        self._project_name = path.rstrip('/').split('/')[-1]

        # Start background sync
        sync_engine.start_background_sync()

        self.accept()

    def get_storage(self) -> Optional[ProjectStorage]:
        """Get the created storage instance."""
        return self._storage

    def get_project_name(self) -> str:
        """Get the project name."""
        return self._project_name

    def get_ssh_manager(self):
        """Get SSH manager if in remote mode."""
        if self._btn_remote.isChecked():
            return self._remote_config.get_ssh_manager()
        return None
