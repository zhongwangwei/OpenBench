# -*- coding: utf-8 -*-
"""
Remote server configuration widget.

Provides UI for configuring SSH connection to remote servers,
including authentication, compute node (multi-hop), and Python environment.

RemoteConfigWidget's long install/update flows (`git clone`, `git pull`,
and `conda env update` over slow links — up to 900s timeout) run through
`_ssh_worker.SshExecuteWorker` so the Qt main thread remains responsive.
RemoteFileBrowser's directory/listing calls have been migrated to
`_ssh_worker.execute_responsive`, which runs them on a worker thread
while keeping the event loop alive.

The SSH auth handshake (`_test_connection` / `_confirm_node_connection`)
also runs via `call_responsive`; the host-key confirmation dialog that
paramiko fires mid-handshake marshals itself back to the GUI thread via
`_ssh_worker.call_on_gui_thread`, so no remote operation blocks the Qt
main thread anymore.
"""

import logging
import os
import platform
import re
import shlex
from typing import Optional, Dict, Any, List


def _safe_remote_path(path: str) -> str:
    """Validate then shell-quote a user-supplied remote path.

    Rejects empty input and characters that have no business in a path
    (NUL, newline, carriage return) before quoting — defense in depth so
    the user gets a clear error rather than a half-escaped command.
    """
    if not path:
        raise ValueError("Remote path is empty.")
    if any(c in path for c in ("\x00", "\n", "\r")):
        raise ValueError("Remote path contains illegal characters.")
    # Expand a leading tilde to the remote $HOME: shlex.quote would otherwise
    # single-quote it and the shell would look for a literal '~' directory.
    # The codebase's remote convention defaults to ~/OpenBench, so this is a
    # mainstream input, not an edge case.
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME"' + shlex.quote(path[1:])
    return shlex.quote(path)


from PySide6.QtWidgets import (
    QDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QButtonGroup,
    QCheckBox,
    QLabel,
    QMessageBox,
    QFileDialog,
    QInputDialog,
)
from openbench.gui.widgets.no_scroll_widgets import NoScrollSpinBox, NoScrollComboBox
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QListWidgetItem

from openbench.gui.widgets._ssh_worker import (
    SshExecuteWorker,
    call_on_gui_thread,
    call_responsive,
    execute_responsive,
)
from openbench.gui.widgets._task_worker import CallableWorker
from openbench.remote.ssh import SSHManager, SSHConnectionError
from openbench.remote.credentials import CredentialManager, CredentialStorageError

logger = logging.getLogger(__name__)

_DETACHED_TASK_WORKERS = []


def parse_ssh_config() -> List[Dict[str, str]]:
    """Parse SSH config file and return list of hosts.

    Supports both Unix (~/.ssh/config) and Windows (%USERPROFILE%\\.ssh\\config).

    Returns:
        List of dicts with 'name', 'hostname', 'user', 'port', 'identity_file' keys
    """
    hosts = []

    # Determine SSH config path based on platform
    if platform.system() == "Windows":
        # Windows: %USERPROFILE%\.ssh\config
        user_profile = os.environ.get("USERPROFILE", "")
        config_paths = [
            os.path.join(user_profile, ".ssh", "config"),
        ]
    else:
        # Unix/macOS: ~/.ssh/config
        home = os.path.expanduser("~")
        config_paths = [
            os.path.join(home, ".ssh", "config"),
        ]

    for config_path in config_paths:
        if not os.path.exists(config_path):
            continue

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                current_host = None

                for line in f:
                    line = line.strip()

                    # Skip comments and empty lines
                    if not line or line.startswith("#"):
                        continue

                    # Parse key-value pairs
                    match = re.match(r"^(\S+)\s+(.+)$", line, re.IGNORECASE)
                    if not match:
                        continue

                    key = match.group(1).lower()
                    value = match.group(2).strip()

                    if key == "host":
                        # Skip wildcard hosts
                        if "*" in value or "?" in value:
                            current_host = None
                            continue
                        # New host block
                        current_host = {"name": value, "hostname": "", "user": "", "port": "22", "identity_file": ""}
                        hosts.append(current_host)
                    elif current_host is not None:
                        if key == "hostname":
                            current_host["hostname"] = value
                        elif key == "user":
                            current_host["user"] = value
                        elif key == "port":
                            current_host["port"] = value
                        elif key == "identityfile":
                            # Expand ~ in path
                            current_host["identity_file"] = os.path.expanduser(value)
        except Exception:
            continue

    return hosts


class ClickableLineEdit(QLineEdit):
    """QLineEdit that emits a signal when clicked."""

    clicked = Signal()

    def mousePressEvent(self, event):
        """Handle mouse press event."""
        super().mousePressEvent(event)
        if not self.text():  # Only show menu if empty
            self.clicked.emit()


# Delimits the resolve/ls/find sections of the combined listing command.
_SECTION_MARKER = "__OPENBENCH_SECTION__"


def _build_conda_create_task(ssh_manager, quoted_conda_exe, quoted_env_name, env_exists, interrupted):
    """Build the conda-create worker task with cooperative interruption.

    ``interrupted`` is probed before each blocking step so a detached worker
    stops at the next step boundary instead of running the full sequence.
    """

    def create_env_task():
        output_chunks = []
        if env_exists:
            if interrupted():
                return {"exit_code": 130, "output": "Interrupted before environment removal.\n", "envs": []}
            cmd = f"{quoted_conda_exe} env remove -n {quoted_env_name} -y 2>&1"
            stdout, stderr, exit_code = ssh_manager.execute(cmd, timeout=120)
            output_chunks.append(f"$ {cmd}\n{stdout}{stderr}\n")
            if exit_code != 0:
                return {"exit_code": exit_code, "output": "".join(output_chunks), "envs": []}
        if interrupted():
            output_chunks.append("\nInterrupted before environment creation.\n")
            return {"exit_code": 130, "output": "".join(output_chunks), "envs": []}
        cmd = f"{quoted_conda_exe} create -n {quoted_env_name} python=3.12 -y 2>&1"
        stdout, stderr, exit_code = ssh_manager.execute(cmd, timeout=300)
        output_chunks.append(f"$ {cmd}\n{stdout}{stderr}\n")
        envs = ssh_manager.detect_conda_envs() if exit_code == 0 and not interrupted() else []
        return {"exit_code": exit_code, "output": "".join(output_chunks), "envs": envs}

    return create_env_task


class _InstallProgressDialog(QDialog):
    """Progress dialog that ignores Esc/close while the install worker runs.

    A plain QDialog closes on Esc, leaving a 300-900s install running headless
    with no way to see its log or completion state again.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.allow_close = False

    def reject(self):
        if self.allow_close:
            super().reject()

    def closeEvent(self, event):
        if self.allow_close:
            super().closeEvent(event)
        else:
            event.ignore()


class RemoteFileBrowser(QWidget):
    """Dialog for browsing files on remote server."""

    file_selected = Signal(str)

    def __init__(self, ssh_manager, start_path: str = "/", parent=None, select_dirs: bool = False):
        """Initialize remote file browser.

        Args:
            ssh_manager: SSH manager for remote operations
            start_path: Initial directory path
            parent: Parent widget
            select_dirs: If True, allow selecting directories instead of files
        """
        super().__init__(parent)
        self._ssh_manager = ssh_manager
        self._current_path = start_path
        self._select_dirs = select_dirs
        self._loading = False
        self._setup_ui()
        self._load_directory(start_path)

    def _setup_ui(self):
        """Setup the browser UI."""
        from PySide6.QtWidgets import (
            QVBoxLayout,
            QHBoxLayout,
            QListWidget,
            QPushButton,
            QLineEdit,
            QLabel,
        )

        layout = QVBoxLayout(self)

        # Path bar
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Path:"))
        self.path_input = QLineEdit()
        self.path_input.returnPressed.connect(self._on_path_entered)
        path_layout.addWidget(self.path_input)
        self.btn_go = QPushButton("Go")
        self.btn_go.clicked.connect(self._on_path_entered)
        path_layout.addWidget(self.btn_go)
        layout.addLayout(path_layout)

        # File list
        self.file_list = QListWidget()
        self.file_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.file_list)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_new_folder = QPushButton("New Folder")
        self.btn_new_folder.clicked.connect(self._on_new_folder)
        btn_layout.addWidget(self.btn_new_folder)
        btn_layout.addStretch()
        self.btn_select = QPushButton("Select")
        self.btn_select.clicked.connect(self._on_select)
        btn_layout.addWidget(self.btn_select)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

    def _load_directory(self, path: str):
        """Load directory contents from remote server (UI stays responsive)."""
        if self._loading:
            return
        self._loading = True
        self.setEnabled(False)
        try:
            self._load_directory_now(path)
        finally:
            self._loading = False
            self.setEnabled(True)

    def _load_directory_now(self, path: str):
        """Fetch and render a directory listing; only called via _load_directory."""
        previous_path = getattr(self, "_current_path", "/")
        previous_input = self.path_input.text() if hasattr(self, "path_input") else previous_path

        try:
            # _safe_remote_path validates and shell-quotes user input —
            # single-quote wrapping alone was insecure (a quote in the path
            # could break out and execute arbitrary commands).
            try:
                quoted_path = _safe_remote_path(path)
            except ValueError as exc:
                QMessageBox.warning(self, "Remote Browser", str(exc))
                return

            # One round-trip for resolve + listing + bulk symlink probing.
            # The previous three sequential calls cost 3x RTT per navigation
            # on a high-latency link. Sections are delimited by a marker
            # line; ls/find run after cd so the resolved path applies, and a
            # find failure is tolerated (exit 0 forced) like before.
            list_cmd = (
                f"cd {quoted_path} 2>/dev/null || exit 21; pwd -P; echo {_SECTION_MARKER}; "
                f"ls -la 2>/dev/null || exit 22; echo {_SECTION_MARKER}; "
                f'find -L "$(pwd -P)" -maxdepth 1 -type d -print 2>/dev/null; exit 0'
            )
            stdout, _, exit_code = execute_responsive(self._ssh_manager, list_cmd, timeout=20)

            if exit_code != 0:
                self._current_path = previous_path
                self.path_input.setText(previous_input)
                QMessageBox.warning(self, "Remote Browser", f"Failed to list remote directory:\n{path}")
                return

            import posixpath as _pp

            sections: list[list[str]] = [[]]
            for line in stdout.splitlines():
                if line.strip() == _SECTION_MARKER:
                    sections.append([])
                else:
                    sections[-1].append(line)

            # `pwd -P` output is necessarily the LAST line before the first
            # marker; anything earlier is rc-file/motd noise from the remote
            # shell (bash under sshd sources ~/.bashrc non-interactively).
            resolved = next((line.strip() for line in reversed(sections[0]) if line.strip()), "")
            if resolved:
                path = resolved
            ls_text = "\n".join(sections[1]).strip() if len(sections) > 1 else ""
            find_lines = sections[2] if len(sections) > 2 else []
            symlink_dir_paths: set[str] = {_pp.normpath(line.strip()) for line in find_lines if line.strip()}

            self.file_list.clear()
            self._current_path = path
            self.path_input.setText(path)

            # Add parent directory entry
            if path != "/":
                item = QListWidgetItem("📁 ..")
                item.setData(Qt.UserRole, {"name": "..", "is_dir": True, "is_link": False})
                self.file_list.addItem(item)

            # Parse ls output. `line.split()` collapses runs of whitespace,
            # so a filename with multiple consecutive spaces or leading
            # spaces gets normalized to a single-space form here — display
            # will diverge from the real filename and downstream test/cd
            # commands won't find it. The 9th column is where the name
            # begins in `ls -la`, so re-split with maxsplit=8 to preserve
            # the original spacing in the filename portion.
            entries = []
            for line in ls_text.split("\n")[1:]:  # Skip total line
                if not line.strip():
                    continue
                parts = line.split(maxsplit=8)
                if len(parts) < 9:
                    continue

                perms = parts[0]
                name = parts[8]  # whole remaining string, spacing preserved

                # Handle symlink display (name -> target)
                is_link = perms.startswith("l")
                display_name = name
                if is_link and " -> " in name:
                    display_name = name.split(" -> ")[0]

                if display_name in [".", ".."]:
                    continue

                is_dir = perms.startswith("d")
                is_exec = "x" in perms and not is_dir and not is_link
                entries.append({"name": display_name, "is_dir": is_dir, "is_exec": is_exec, "is_link": is_link})

            # Symlink targets: prefer the bulk find result; if that lookup
            # failed (empty set), probe ALL symlinks in one compound round
            # trip instead of one test -d per entry.
            link_paths = [f"{path.rstrip('/')}/{e['name']}" for e in entries if e["is_link"]]
            if link_paths and not symlink_dir_paths:
                quoted_links = []
                for link_path in link_paths:
                    try:
                        quoted_links.append(_safe_remote_path(link_path))
                    except ValueError:
                        continue
                if quoted_links:
                    probe = (
                        "for p in " + " ".join(quoted_links) + '; do [ -d "$p" ] && printf \'%s\\n\' "$p"; done; exit 0'
                    )
                    try:
                        probe_stdout, _, _ = execute_responsive(self._ssh_manager, probe, timeout=10)
                        symlink_dir_paths = {
                            _pp.normpath(line.strip()) for line in probe_stdout.splitlines() if line.strip()
                        }
                    except Exception:
                        symlink_dir_paths = set()

            for entry in entries:
                is_dir = entry["is_dir"]
                if entry["is_link"]:
                    full_path = f"{path.rstrip('/')}/{entry['name']}"
                    if _pp.normpath(full_path) in symlink_dir_paths:
                        is_dir = True

                if is_dir:
                    icon = "🔗" if entry["is_link"] else "📁"
                elif entry["is_link"]:
                    icon = "🔗"
                elif entry["is_exec"]:
                    icon = "🐍" if "python" in entry["name"].lower() else "⚡"
                else:
                    icon = "📄"

                item = QListWidgetItem(f"{icon} {entry['name']}")
                item.setData(
                    Qt.UserRole,
                    {"name": entry["name"], "is_dir": is_dir, "is_exec": entry["is_exec"], "is_link": entry["is_link"]},
                )
                self.file_list.addItem(item)

        except Exception as e:
            self._current_path = previous_path
            self.path_input.setText(previous_input)
            QMessageBox.warning(self, "Remote Browser", f"Failed to load remote directory:\n{path}\n\n{e}")

    def _on_item_double_clicked(self, item):
        """Handle double-click on item."""
        if self._loading:
            return
        data = item.data(Qt.UserRole)
        if not data:
            return

        name = data["name"]
        is_link = data.get("is_link", False)

        if data["is_dir"]:
            # Navigate to directory (use forward slashes for remote Linux paths)
            if name == "..":
                # Get parent directory
                current = self._current_path.rstrip("/")
                new_path = "/".join(current.split("/")[:-1])
                if not new_path:
                    new_path = "/"
            else:
                # Join paths with forward slash
                new_path = f"{self._current_path.rstrip('/')}/{name}"
            self._load_directory(new_path)
        elif is_link and not self._select_dirs:
            # For symlinked files, resolve the target and select it
            full_path = f"{self._current_path.rstrip('/')}/{name}"
            try:
                # Resolve the symlink to get the actual file path
                try:
                    quoted_full = _safe_remote_path(full_path)
                except ValueError:
                    return
                resolve_cmd = f"readlink -f {quoted_full} 2>/dev/null"
                stdout, _, exit_code = execute_responsive(self._ssh_manager, resolve_cmd, timeout=5)
                if exit_code == 0 and stdout.strip():
                    resolved_path = stdout.strip()
                    # Emit the resolved path
                    self.file_selected.emit(resolved_path)
                    self.parent().accept() if hasattr(self.parent(), "accept") else None
                else:
                    QMessageBox.warning(self, "Broken Link", f"Failed to resolve remote symlink:\n{full_path}")
            except Exception as exc:
                QMessageBox.warning(self, "Broken Link", f"Failed to resolve remote symlink:\n{full_path}\n\n{exc}")
        else:
            # Regular file - ensure item is selected before calling _on_select
            self.file_list.setCurrentItem(item)
            self._on_select()

    def _on_path_entered(self):
        """Handle path entered in text box."""
        path = self.path_input.text().strip()
        if path:
            self._load_directory(path)

    def _on_select(self):
        """Handle select button click."""
        item = self.file_list.currentItem()
        if item:
            data = item.data(Qt.UserRole)
            if data:
                is_dir = data["is_dir"]
                # Allow selection based on mode
                if self._select_dirs:
                    # In directory mode, select directory or current path
                    if is_dir:
                        import posixpath

                        # The synthetic ".." row must resolve to the parent,
                        # not be emitted verbatim as "<current>/..".
                        full_path = posixpath.normpath(f"{self._current_path.rstrip('/')}/{data['name']}")
                    else:
                        # If file selected in dir mode, use current directory
                        full_path = self._current_path
                    self.file_selected.emit(full_path)
                    self.parent().accept() if hasattr(self.parent(), "accept") else None
                elif not is_dir:
                    # In file mode, only allow file selection
                    full_path = f"{self._current_path.rstrip('/')}/{data['name']}"
                    self.file_selected.emit(full_path)
                    self.parent().accept() if hasattr(self.parent(), "accept") else None
        elif self._select_dirs:
            # No item selected but in dir mode - select current directory
            self.file_selected.emit(self._current_path)
            self.parent().accept() if hasattr(self.parent(), "accept") else None

    def _on_cancel(self):
        """Handle cancel button click."""
        self.parent().reject() if hasattr(self.parent(), "reject") else None

    def _on_new_folder(self):
        """Create a new folder in current directory."""
        from PySide6.QtWidgets import QInputDialog, QMessageBox

        folder_name, ok = QInputDialog.getText(self, "New Folder", "Enter folder name:")

        if ok and folder_name:
            folder_name = folder_name.strip()
            if not folder_name:
                return

            new_path = f"{self._current_path.rstrip('/')}/{folder_name}"
            try:
                try:
                    quoted_new_path = _safe_remote_path(new_path)
                except ValueError as exc:
                    QMessageBox.warning(self, "Invalid Folder Name", str(exc))
                    return
                cmd = f"mkdir -p {quoted_new_path}"
                stdout, stderr, exit_code = execute_responsive(self._ssh_manager, cmd, timeout=10)

                if exit_code == 0:
                    # Refresh directory and navigate to new folder
                    self._load_directory(self._current_path)
                else:
                    QMessageBox.warning(self, "Error", f"Failed to create folder:\n{stderr}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to create folder: {e}")

    def get_selected_path(self) -> str:
        """Get the currently selected file path."""
        item = self.file_list.currentItem()
        if item:
            data = item.data(Qt.UserRole)
            if data:
                return f"{self._current_path.rstrip('/')}/{data['name']}"
        return ""


class RemoteConfigWidget(QWidget):
    """Widget for configuring remote server connection.

    Provides comprehensive UI for:
    - SSH host configuration with authentication options
    - Optional compute node (multi-hop) connection
    - Remote Python environment detection and selection

    Signals:
        connection_status_changed(bool): Emitted when connection state changes
        credentials_saved(str): Emitted when credentials are saved (host string)
        config_changed(): Emitted when any configuration value changes
    """

    # Signals
    connection_status_changed = Signal(bool)  # Connection state changed
    credentials_saved = Signal(str)  # Credentials saved for host
    config_changed = Signal()  # Configuration changed

    def __init__(self, parent=None):
        """Initialize RemoteConfigWidget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._ssh_manager: Optional[SSHManager] = None
        self._credential_manager = CredentialManager()
        self._conda_create_worker = None
        self._conda_create_dialog = None
        self._install_worker = None
        self._setup_ui()
        self.destroyed.connect(lambda *_: self._cleanup_conda_create_worker(detach=True))
        self.destroyed.connect(lambda *_: self._cleanup_install_worker(detach=True))

    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)

        # === Remote Server Group ===
        server_group = QGroupBox("Remote Server")
        server_layout = QFormLayout(server_group)
        server_layout.setSpacing(8)

        # Host input with Confirm button
        host_layout = QHBoxLayout()
        host_layout.setSpacing(8)

        # Load SSH config hosts
        self._ssh_config_hosts = parse_ssh_config()

        # Custom line edit that shows menu on click
        self.host_input = ClickableLineEdit()
        self.host_input.setPlaceholderText("Click to select or type user@host")
        self.host_input.textChanged.connect(self._on_config_changed)
        if self._ssh_config_hosts:
            self.host_input.clicked.connect(self._show_ssh_config_menu)
        host_layout.addWidget(self.host_input, 1)

        self.btn_test = QPushButton("Connect")
        self.btn_test.setFixedWidth(70)
        self.btn_test.setToolTip("Connect to SSH server")
        self.btn_test.clicked.connect(self._test_connection)
        host_layout.addWidget(self.btn_test)

        self.btn_disconnect = QPushButton("Disconnect")
        self.btn_disconnect.setFixedWidth(80)
        self.btn_disconnect.setToolTip("Disconnect from SSH server")
        self.btn_disconnect.clicked.connect(self._disconnect_server)
        self.btn_disconnect.setEnabled(False)
        host_layout.addWidget(self.btn_disconnect)

        server_layout.addRow("Host:", host_layout)

        # Authentication type radio buttons
        auth_layout = QHBoxLayout()
        auth_layout.setSpacing(15)
        self.auth_group = QButtonGroup(self)
        self.radio_password = QRadioButton("Password")
        self.radio_password.setChecked(True)
        self.radio_key = QRadioButton("SSH Key")
        self.auth_group.addButton(self.radio_password)
        self.auth_group.addButton(self.radio_key)
        self.radio_password.toggled.connect(self._on_auth_type_changed)
        auth_layout.addWidget(self.radio_password)
        auth_layout.addWidget(self.radio_key)
        auth_layout.addStretch()
        server_layout.addRow("Auth:", auth_layout)

        # Password input with Save checkbox
        pwd_layout = QHBoxLayout()
        pwd_layout.setSpacing(8)
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setPlaceholderText("Password")
        self.password_input.textChanged.connect(self._on_config_changed)
        pwd_layout.addWidget(self.password_input, 1)

        self.cb_save_password = QCheckBox("Save")
        self.cb_save_password.setToolTip("Save password (encrypted)")
        pwd_layout.addWidget(self.cb_save_password)

        self.password_row_widget = QWidget()
        self.password_row_widget.setLayout(pwd_layout)
        server_layout.addRow("", self.password_row_widget)

        # SSH Key input with Browse button
        key_layout = QHBoxLayout()
        key_layout.setSpacing(8)
        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("~/.ssh/id_rsa")
        self.key_input.textChanged.connect(self._on_config_changed)
        key_layout.addWidget(self.key_input, 1)

        self.btn_browse_key = QPushButton("Browse")
        self.btn_browse_key.setFixedWidth(60)
        self.btn_browse_key.clicked.connect(self._browse_key)
        key_layout.addWidget(self.btn_browse_key)

        self.key_row_widget = QWidget()
        self.key_row_widget.setLayout(key_layout)
        self.key_row_widget.hide()  # Hidden by default (password mode)
        server_layout.addRow("", self.key_row_widget)

        # Connection status indicator
        self.status_label = QLabel("Not connected")
        self.status_label.setStyleSheet("color: #999;")
        server_layout.addRow("Status:", self.status_label)

        layout.addWidget(server_group)

        # === Compute Node Group (Optional) ===
        node_group = QGroupBox("Compute Node (Optional)")
        node_group.setCheckable(True)
        node_group.setChecked(False)
        node_group.toggled.connect(self._on_config_changed)
        self.node_group = node_group
        node_layout = QFormLayout(node_group)
        node_layout.setSpacing(8)

        # Node name input with Confirm button
        node_input_layout = QHBoxLayout()
        node_input_layout.setSpacing(8)
        self.node_input = QLineEdit()
        self.node_input.setPlaceholderText("node110")
        self.node_input.textChanged.connect(self._on_config_changed)
        node_input_layout.addWidget(self.node_input, 1)

        self.btn_confirm_node = QPushButton("Connect")
        self.btn_confirm_node.setFixedWidth(70)
        self.btn_confirm_node.setToolTip("Connect to compute node via SSH")
        self.btn_confirm_node.clicked.connect(self._confirm_node_connection)
        node_input_layout.addWidget(self.btn_confirm_node)

        self.btn_disconnect_node = QPushButton("Disconnect")
        self.btn_disconnect_node.setFixedWidth(80)
        self.btn_disconnect_node.setToolTip("Disconnect from compute node")
        self.btn_disconnect_node.clicked.connect(self._disconnect_node)
        self.btn_disconnect_node.setEnabled(False)
        node_input_layout.addWidget(self.btn_disconnect_node)
        node_layout.addRow("Node:", node_input_layout)

        # Node authentication type
        node_auth_layout = QHBoxLayout()
        node_auth_layout.setSpacing(15)
        self.node_auth_group = QButtonGroup(self)
        self.radio_node_none = QRadioButton("None (internal trust)")
        self.radio_node_none.setChecked(True)
        self.radio_node_password = QRadioButton("Password")
        self.radio_node_key = QRadioButton("SSH Key")
        self.node_auth_group.addButton(self.radio_node_none)
        self.node_auth_group.addButton(self.radio_node_password)
        self.node_auth_group.addButton(self.radio_node_key)
        self.radio_node_password.toggled.connect(self._on_node_auth_changed)
        self.radio_node_key.toggled.connect(self._on_node_auth_changed)
        node_auth_layout.addWidget(self.radio_node_none)
        node_auth_layout.addWidget(self.radio_node_password)
        node_auth_layout.addWidget(self.radio_node_key)
        node_auth_layout.addStretch()
        node_layout.addRow("Auth:", node_auth_layout)

        # Node password input
        self.node_password_input = QLineEdit()
        self.node_password_input.setEchoMode(QLineEdit.Password)
        self.node_password_input.setPlaceholderText("Node password")
        self.node_password_input.textChanged.connect(self._on_config_changed)
        self.node_password_input.hide()
        node_layout.addRow("", self.node_password_input)

        # Node SSH key input
        node_key_layout = QHBoxLayout()
        node_key_layout.setSpacing(8)
        self.node_key_input = QLineEdit()
        self.node_key_input.setPlaceholderText("Path to SSH key for compute node")
        self.node_key_input.textChanged.connect(self._on_config_changed)
        node_key_layout.addWidget(self.node_key_input, 1)
        self.btn_browse_node_key = QPushButton("Browse")
        self.btn_browse_node_key.setFixedWidth(60)
        self.btn_browse_node_key.clicked.connect(self._browse_node_key)
        node_key_layout.addWidget(self.btn_browse_node_key)
        self.node_key_widget = QWidget()
        self.node_key_widget.setLayout(node_key_layout)
        self.node_key_widget.hide()
        node_layout.addRow("", self.node_key_widget)

        # Node connection status
        self.node_status_label = QLabel("Not connected")
        self.node_status_label.setStyleSheet("color: #999;")
        node_layout.addRow("Status:", self.node_status_label)

        layout.addWidget(node_group)

        # === Parallel Processing Group ===
        parallel_group = QGroupBox("Parallel Processing")
        parallel_layout = QFormLayout(parallel_group)
        parallel_layout.setSpacing(8)

        # Number of CPU cores
        cores_layout = QHBoxLayout()
        cores_layout.setSpacing(8)
        self.num_cores_spin = NoScrollSpinBox()
        self.num_cores_spin.setRange(1, 128)
        self.num_cores_spin.setValue(4)
        self.num_cores_spin.setMinimumWidth(80)
        self.num_cores_spin.setToolTip("Number of CPU cores to use for parallel processing")
        self.num_cores_spin.valueChanged.connect(self._on_config_changed)
        cores_layout.addWidget(self.num_cores_spin)
        self.cpu_available_label = QLabel("(Available: Connect to detect)")
        cores_layout.addWidget(self.cpu_available_label)
        cores_layout.addStretch()
        parallel_layout.addRow("CPU Cores:", cores_layout)

        layout.addWidget(parallel_group)

        # === Remote Python Environment Group ===
        env_group = QGroupBox("Remote Python Environment")
        env_layout = QFormLayout(env_group)
        env_layout.setSpacing(8)

        # Conda environment with Refresh button
        conda_layout = QHBoxLayout()
        conda_layout.setSpacing(8)
        self.conda_combo = NoScrollComboBox()
        self.conda_combo.addItem("(Not using conda environment)")
        self.conda_combo.currentIndexChanged.connect(self._on_conda_env_changed)
        self.conda_combo.currentTextChanged.connect(self._on_config_changed)
        conda_layout.addWidget(self.conda_combo, 1)

        self.btn_refresh_conda = QPushButton("Refresh")
        self.btn_refresh_conda.setFixedWidth(60)
        self.btn_refresh_conda.setToolTip("Refresh conda environments from remote server")
        self.btn_refresh_conda.clicked.connect(self._refresh_conda)
        conda_layout.addWidget(self.btn_refresh_conda)

        self.btn_new_conda = QPushButton("New")
        self.btn_new_conda.setFixedWidth(50)
        self.btn_new_conda.setToolTip("Create new OpenBench conda environment")
        self.btn_new_conda.clicked.connect(self._create_conda_env)
        conda_layout.addWidget(self.btn_new_conda)

        env_layout.addRow("Conda:", conda_layout)

        # Python path with Detect and Browse buttons
        python_layout = QHBoxLayout()
        python_layout.setSpacing(8)
        self.python_combo = NoScrollComboBox()
        self.python_combo.setEditable(True)
        self.python_combo.setMinimumWidth(250)
        self.python_combo.currentTextChanged.connect(self._on_config_changed)
        self.python_combo.currentTextChanged.connect(self._infer_conda_from_python)
        python_layout.addWidget(self.python_combo, 1)

        self.btn_detect_python = QPushButton("Detect")
        self.btn_detect_python.setFixedWidth(60)
        self.btn_detect_python.setToolTip("Detect Python interpreters on remote server")
        self.btn_detect_python.clicked.connect(self._detect_python)
        python_layout.addWidget(self.btn_detect_python)

        self.btn_browse_python = QPushButton("Browse")
        self.btn_browse_python.setFixedWidth(60)
        self.btn_browse_python.setToolTip("Enter Python path on remote server manually")
        self.btn_browse_python.clicked.connect(self._browse_python)
        python_layout.addWidget(self.btn_browse_python)

        env_layout.addRow("Python:", python_layout)

        # OpenBench path with Browse and Install buttons
        ob_layout = QHBoxLayout()
        ob_layout.setSpacing(8)
        self.openbench_input = QLineEdit()
        self.openbench_input.setPlaceholderText("/home/user/OpenBench")
        self.openbench_input.textChanged.connect(self._on_config_changed)
        ob_layout.addWidget(self.openbench_input, 1)

        self.btn_browse_ob = QPushButton("Browse")
        self.btn_browse_ob.setFixedWidth(60)
        self.btn_browse_ob.setToolTip("Browse remote server for OpenBench installation path")
        self.btn_browse_ob.clicked.connect(self._browse_openbench)
        ob_layout.addWidget(self.btn_browse_ob)

        self.btn_install_ob = QPushButton("Install")
        self.btn_install_ob.setFixedWidth(60)
        self.btn_install_ob.setToolTip("Install OpenBench on remote server")
        self.btn_install_ob.clicked.connect(self._install_openbench)
        ob_layout.addWidget(self.btn_install_ob)

        env_layout.addRow("OpenBench:", ob_layout)

        layout.addWidget(env_group)
        layout.addStretch()

    def _show_ssh_config_menu(self):
        """Show popup menu with SSH config hosts."""
        from PySide6.QtWidgets import QMenu
        from PySide6.QtCore import QPoint

        menu = QMenu(self)

        for host in self._ssh_config_hosts:
            name = host["name"]
            user = host.get("user", "")
            hostname = host.get("hostname", name)
            host.get("port", "22")

            if user and hostname:
                display = f"{name}  ({user}@{hostname})"
            elif hostname:
                display = f"{name}  ({hostname})"
            else:
                display = name

            action = menu.addAction(display)
            action.setData(host)

        # Show menu below the input field
        action = menu.exec(self.host_input.mapToGlobal(QPoint(0, self.host_input.height())))

        if action:
            host_data = action.data()
            self._apply_ssh_config_host(host_data)

    def _apply_ssh_config_host(self, host_data: Dict[str, str]):
        """Apply selected SSH config host to the form.

        Args:
            host_data: Host configuration dictionary
        """
        user = host_data.get("user", "")
        hostname = host_data.get("hostname", host_data["name"])
        port = host_data.get("port", "22")

        # Build host string
        if user:
            if port and port != "22":
                host_str = f"{user}@{hostname}:{port}"
            else:
                host_str = f"{user}@{hostname}"
        else:
            if port and port != "22":
                host_str = f"{hostname}:{port}"
            else:
                host_str = hostname

        self.host_input.setText(host_str)

        # Fill in identity file if specified
        identity_file = host_data.get("identity_file", "")
        if identity_file:
            self.radio_key.setChecked(True)
            self.key_input.setText(identity_file)

        self._on_config_changed()

    def _on_auth_type_changed(self, checked: bool):
        """Handle auth type radio button change.

        Args:
            checked: Whether password radio is checked
        """
        if self.radio_password.isChecked():
            self.password_row_widget.show()
            self.key_row_widget.hide()
        else:
            self.password_row_widget.hide()
            self.key_row_widget.show()
        self._on_config_changed()

    def _on_node_auth_changed(self, checked: bool):
        """Handle node auth type change."""
        self.node_password_input.setVisible(self.radio_node_password.isChecked())
        self.node_key_widget.setVisible(self.radio_node_key.isChecked())
        self._on_config_changed()

    def _browse_node_key(self):
        """Open file dialog to browse for node SSH key file."""
        start_path = os.path.expanduser("~/.ssh")
        path, _ = QFileDialog.getOpenFileName(self, "Select SSH Key for Compute Node", start_path, "All Files (*)")
        if path:
            self.node_key_input.setText(path)

    def _confirm_node_connection(self):
        """Connect to compute node via SSH."""
        if getattr(self, "_handshake_active", False):
            return  # the main-server handshake is mid-auth; don't race it
        if not self._ssh_manager or not self._ssh_manager.is_connected:
            QMessageBox.warning(self, "Error", "Please connect to the main server first using the Confirm button above")
            return

        node_name = self.node_input.text().strip()
        if not node_name:
            QMessageBox.warning(self, "Error", "Please enter the compute node name")
            return

        self._handshake_active = True
        try:
            self.btn_confirm_node.setEnabled(False)
            self.node_status_label.setText("Connecting...")
            self.node_status_label.setStyleSheet("color: orange;")

            # Get authentication details
            node_password = None
            node_key_file = None
            if self.radio_node_password.isChecked():
                node_password = self.node_password_input.text()
                if not node_password:
                    password, ok = QInputDialog.getText(
                        self, "Node Password", f"Enter password for {node_name}:", QLineEdit.Password
                    )
                    if ok and password:
                        node_password = password
                    else:
                        self.node_status_label.setText("Cancelled")
                        self.node_status_label.setStyleSheet("color: gray;")
                        return
            elif self.radio_node_key.isChecked():
                node_key_file = self.node_key_input.text().strip()
                if not node_key_file:
                    QMessageBox.warning(self, "Error", "Please specify SSH key file for compute node")
                    self.node_status_label.setText("No key file")
                    self.node_status_label.setStyleSheet("color: red;")
                    return

            # Connect to compute node through jump (handshake off the GUI
            # thread; host-key prompt marshals back to it)
            manager = self._ssh_manager
            call_responsive(
                lambda: manager.connect_with_jump(
                    main_host=node_name, main_password=node_password, main_key_file=node_key_file
                )
            )

            if self._ssh_manager.is_jump_connected:
                self.node_status_label.setText(f"✓ Connected to {node_name}")
                self.node_status_label.setStyleSheet("color: green; font-weight: bold;")
                # Toggle buttons - connected
                self.btn_confirm_node.setEnabled(False)
                self.btn_disconnect_node.setEnabled(True)
                # Update CPU count for compute node
                self._update_remote_cpu_count()
            else:
                self.node_status_label.setText("✗ Connection failed")
                self.node_status_label.setStyleSheet("color: red;")
                self.btn_confirm_node.setEnabled(True)

        except Exception as e:
            import shiboken6

            if not shiboken6.isValid(self):
                return  # widget destroyed while the handshake event loop ran
            self.node_status_label.setText(f"✗ {str(e)[:50]}")
            self.node_status_label.setStyleSheet("color: red;")
            self.btn_confirm_node.setEnabled(True)
            QMessageBox.warning(self, "Connection Failed", str(e))
        finally:
            self._handshake_active = False

    def _on_config_changed(self):
        """Handle any configuration change."""
        self.config_changed.emit()

    def _on_conda_env_changed(self, index: int):
        """Handle conda environment selection change.

        Updates the Python path to use the selected conda environment's Python.
        """
        if index <= 0:
            # "(Not using conda environment)" selected, don't change Python path
            return

        env_name = self.conda_combo.currentText()
        env_path = self.conda_combo.itemData(index)
        if not env_name or not env_path:
            return

        # Get Python path directly from conda
        if self._ssh_manager and self._ssh_manager.is_connected:
            # A combo change during the in-flight round trip supersedes this
            # query; the stale result must not be applied afterwards.
            seq = getattr(self, "_conda_env_sync_seq", 0) + 1
            self._conda_env_sync_seq = seq
            try:
                # Use conda run to get the actual Python path. Build the
                # inner bash -c argument with shlex.quote on env_name, then
                # shlex.quote the whole inner command for the outer shell.
                inner = f"conda run -n {shlex.quote(env_name)} which python 2>/dev/null"
                cmd = f"bash -i -l -c {shlex.quote(inner)}"
                stdout, _, exit_code = execute_responsive(self._ssh_manager, cmd, timeout=10)
                if seq != self._conda_env_sync_seq:
                    return  # superseded by a newer selection
                if exit_code == 0 and stdout.strip():
                    # Pick the last absolute path in the output; `conda run`
                    # under some shells appends an empty line or `(env) `
                    # prompt fragment, which `split("\n")[-1]` would return
                    # instead of the python path.
                    python_path = ""
                    for line in stdout.splitlines():
                        line = line.strip()
                        if line.startswith("/"):
                            python_path = line
                    if python_path and python_path.startswith("/"):
                        idx = self.python_combo.findText(python_path)
                        if idx < 0:
                            self.python_combo.addItem(python_path)
                        self.python_combo.setCurrentText(python_path)
                        return
            except Exception:
                pass
            if seq != self._conda_env_sync_seq:
                return  # a newer selection owns the fallback too

        # Fallback: construct path from env_path
        python_path = f"{env_path}/bin/python"
        idx = self.python_combo.findText(python_path)
        if idx < 0:
            self.python_combo.addItem(python_path)
        self.python_combo.setCurrentText(python_path)

    def _update_remote_cpu_count(self):
        """Query remote server for CPU count and update label."""
        if not self._ssh_manager or not self._ssh_manager.is_connected:
            return

        try:
            # Query CPU count on remote server (Linux: nproc, macOS: sysctl)
            stdout, stderr, exit_code = execute_responsive(
                self._ssh_manager, "nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null", timeout=10
            )
            if exit_code == 0 and stdout.strip():
                cpu_count = int(stdout.strip())
                self.cpu_available_label.setText(f"(Available on remote: {cpu_count})")
                # Update the max range
                self.num_cores_spin.setRange(1, max(128, cpu_count))
        except Exception:
            self.cpu_available_label.setText("(Available: Could not detect)")

    def _browse_key(self):
        """Open file dialog to browse for SSH key file."""
        start_path = os.path.expanduser("~/.ssh")
        path, _ = QFileDialog.getOpenFileName(self, "Select SSH Key", start_path, "All Files (*)")
        if path:
            self.key_input.setText(path)

    def _confirm_host_key(self, hostname: str, key_type: str, fingerprint: str) -> bool:
        """Show dialog to confirm unknown SSH host key.

        Args:
            hostname: The remote host
            key_type: Type of the key (e.g., ssh-ed25519)
            fingerprint: SHA256 fingerprint of the key

        Returns:
            True if user accepts the key, False otherwise
        """
        from PySide6.QtCore import QThread
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None and QThread.currentThread() != app.thread():
            # paramiko calls this mid-handshake on the call_responsive worker
            # thread; the QMessageBox below must run on the GUI thread.
            return bool(call_on_gui_thread(lambda: self._confirm_host_key(hostname, key_type, fingerprint)))

        msg = (
            f"The authenticity of host '{hostname}' can't be established.\n\n"
            f"Key type: {key_type}\n"
            f"Fingerprint: {fingerprint}\n\n"
            f"Are you sure you want to continue connecting?\n"
            f"The host key will be saved for future connections."
        )
        reply = QMessageBox.question(
            self,
            "Unknown Host Key",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _test_connection(self):
        """Test SSH connection with current settings."""
        host = self.host_input.text().strip()
        if not host:
            QMessageBox.warning(self, "Error", "Please enter host address")
            return
        if getattr(self, "_handshake_active", False):
            return  # another handshake is mid-auth; don't race it

        # Update status
        self.status_label.setText("Connecting...")
        self.status_label.setStyleSheet("color: #f39c12;")  # Orange
        self.btn_test.setEnabled(False)

        self._handshake_active = True
        try:
            self._ssh_manager = SSHManager(host_key_callback=self._confirm_host_key)
            manager = self._ssh_manager

            # Connect based on auth type. The handshake runs on a worker
            # thread (UI stays responsive); the host-key prompt marshals
            # itself back to the GUI thread.
            if self.radio_password.isChecked():
                password = self.password_input.text()
                call_responsive(lambda: manager.connect(host, password=password))
            else:
                key_file = os.path.expanduser(self.key_input.text().strip())
                call_responsive(lambda: manager.connect(host, key_file=key_file))

            # Test jump connection if enabled
            if self.node_group.isChecked():
                node = self.node_input.text().strip()
                if node:
                    node_password = None
                    if self.radio_node_password.isChecked():
                        node_password = self.node_password_input.text()
                    call_responsive(lambda: manager.connect_with_jump(main_host=node, main_password=node_password))

            import shiboken6

            if not shiboken6.isValid(self):
                return  # widget destroyed while the handshake event loop ran

            # Update status to connected
            self.status_label.setText("Connected")
            self.status_label.setStyleSheet("color: #27ae60;")  # Green
            self.connection_status_changed.emit(True)

            # Toggle buttons
            self.btn_test.setEnabled(False)
            self.btn_disconnect.setEnabled(True)

            # Detect remote CPU count
            self._update_remote_cpu_count()

            # Save credentials if requested
            if self.cb_save_password.isChecked():
                self._save_current_credentials()

            QMessageBox.information(self, "Success", "Connection successful!")

        except SSHConnectionError as e:
            import shiboken6

            if not shiboken6.isValid(self):
                return
            self.status_label.setText("Connection failed")
            self.status_label.setStyleSheet("color: #e74c3c;")  # Red
            self.connection_status_changed.emit(False)
            self.btn_test.setEnabled(True)
            QMessageBox.critical(self, "Connection Failed", str(e))
        except Exception as e:
            import shiboken6

            if not shiboken6.isValid(self):
                return
            self.status_label.setText("Error")
            self.status_label.setStyleSheet("color: #e74c3c;")  # Red
            self.connection_status_changed.emit(False)
            self.btn_test.setEnabled(True)
            QMessageBox.critical(self, "Error", f"Unexpected error: {e}")
        finally:
            self._handshake_active = False

    def _disconnect_server(self):
        """Disconnect from SSH server."""
        # First disconnect compute node if connected
        self._disconnect_node(silent=True)

        # Disconnect main server
        if self._ssh_manager:
            try:
                self._ssh_manager.disconnect()
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
            self._ssh_manager = None

        # Update UI
        self.status_label.setText("Not connected")
        self.status_label.setStyleSheet("color: #999;")
        self.btn_test.setEnabled(True)
        self.btn_disconnect.setEnabled(False)
        self.connection_status_changed.emit(False)

    def _disconnect_node(self, silent: bool = False):
        """Disconnect from compute node.

        Args:
            silent: If True, don't show message box
        """
        if not self._ssh_manager:
            return

        # Check if jump connection is active
        if hasattr(self._ssh_manager, "_jump_client") and self._ssh_manager._jump_client:
            try:
                self._ssh_manager.disconnect_jump()
            except Exception as e:
                logger.warning(f"Error during node disconnect: {e}")

        # Update UI
        self.node_status_label.setText("Not connected")
        self.node_status_label.setStyleSheet("color: #999;")
        self.btn_confirm_node.setEnabled(True)
        self.btn_disconnect_node.setEnabled(False)

        if not silent:
            logger.info("Disconnected from compute node")

    def _save_current_credentials(self):
        """Save current credentials using CredentialManager."""
        host = self.host_input.text().strip()
        if not host:
            return

        auth_type = "password" if self.radio_password.isChecked() else "key"
        password = self.password_input.text() if self.radio_password.isChecked() else None
        key_file = self.key_input.text().strip() if self.radio_key.isChecked() else None
        jump_node = self.node_input.text().strip() if self.node_group.isChecked() else None
        jump_auth = "password" if self.radio_node_password.isChecked() else "none"

        try:
            self._credential_manager.save_credential(
                host=host,
                auth_type=auth_type,
                password=password,
                key_file=key_file,
                jump_node=jump_node,
                jump_auth=jump_auth,
            )
        except CredentialStorageError as exc:
            QMessageBox.warning(self, "Credential Save Failed", str(exc))
            return
        self.credentials_saved.emit(host)

    def _browse_python(self):
        """Open remote file browser to select Python path."""
        from openbench.gui import path_utils

        if not self._ssh_manager or not self._ssh_manager.is_connected:
            QMessageBox.warning(self, "Error", "Please connect to server first using the Confirm button")
            return

        home = path_utils.remote_home_dir(self._ssh_manager)
        path = path_utils.pick_remote_path(
            self._ssh_manager, self, "Select Python on Remote Server", home, select_dirs=False
        )
        if not path:
            return
        # Add to combo if not already there
        if self.python_combo.findText(path) < 0:
            self.python_combo.addItem(path)
        self.python_combo.setCurrentText(path)

    def _detect_python(self):
        """Detect Python interpreters on remote server."""
        if getattr(self, "_handshake_active", False):
            return  # SSH auth in flight; don't open channels mid-handshake
        if not self._ssh_manager or not self._ssh_manager.is_connected:
            QMessageBox.warning(self, "Error", "Please connect to server first using the Confirm button")
            return

        try:
            self.btn_detect_python.setEnabled(False)

            # Debug: Also run which python directly and show result
            debug_info = []
            try:
                stdout, _, _ = execute_responsive(
                    self._ssh_manager, "bash -i -l -c 'which python3' 2>/dev/null", timeout=15
                )
                debug_info.append(f"bash -i -l python3: {stdout.strip()}")
            except Exception as e:
                logger.debug("Failed to detect python3 via bash: %s", e)
            try:
                stdout, _, _ = execute_responsive(
                    self._ssh_manager, "bash -i -l -c 'which python' 2>/dev/null", timeout=15
                )
                debug_info.append(f"bash -i -l python: {stdout.strip()}")
            except Exception as e:
                logger.debug("Failed to detect python via bash: %s", e)

            pythons = call_responsive(self._ssh_manager.detect_python_interpreters)
            detection_errors = list(getattr(self._ssh_manager, "last_detection_errors", ()))
            self.python_combo.clear()
            if pythons:
                for p in pythons:
                    self.python_combo.addItem(p)
                msg = f"Found {len(pythons)} Python interpreter(s):\n"
                msg += "\n".join(pythons)
                msg += "\n\nDebug info:\n" + "\n".join(debug_info)
                if detection_errors:
                    msg += "\n\nSome discovery probes failed:\n" + "\n".join(detection_errors)
                QMessageBox.information(self, "Detection Complete", msg)
            else:
                msg = "No Python interpreters found.\n\nDebug info:\n" + "\n".join(debug_info)
                if detection_errors:
                    msg += "\n\nDiscovery errors:\n" + "\n".join(detection_errors)
                    QMessageBox.warning(self, "Detection Failed", msg)
                else:
                    QMessageBox.information(self, "Detection Complete", msg)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to detect Python: {e}")
        finally:
            self.btn_detect_python.setEnabled(True)

    def _refresh_conda(self):
        """Refresh conda environments from remote server."""
        if getattr(self, "_handshake_active", False):
            return  # SSH auth in flight; don't open channels mid-handshake
        if not self._ssh_manager or not self._ssh_manager.is_connected:
            QMessageBox.warning(self, "Error", "Please connect to server first using the Test button")
            return

        try:
            self.btn_refresh_conda.setEnabled(False)

            envs = call_responsive(self._ssh_manager.detect_conda_envs)
            detection_errors = list(getattr(self._ssh_manager, "last_detection_errors", ()))
            self.conda_combo.clear()
            self.conda_combo.addItem("(Not using conda environment)")
            if envs:
                for name, path in envs:
                    self.conda_combo.addItem(name, path)
                msg = f"Found {len(envs)} conda environment(s)"
                if detection_errors:
                    msg += "\n\nSome discovery probes failed:\n" + "\n".join(detection_errors)
                QMessageBox.information(self, "Refresh Complete", msg)
            else:
                msg = "No conda environments found on remote server."
                if detection_errors:
                    msg += "\n\nDiscovery errors:\n" + "\n".join(detection_errors)
                    QMessageBox.warning(self, "Refresh Failed", msg)
                else:
                    QMessageBox.information(self, "Refresh Complete", msg)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to refresh conda environments: {e}")
        finally:
            self.btn_refresh_conda.setEnabled(True)

    def _create_conda_env(self):
        """Create OpenBench conda environment (re-entrancy-guarded entry point)."""
        if getattr(self, "_handshake_active", False):
            return  # SSH auth in flight; don't open channels mid-handshake
        if getattr(self, "_conda_create_worker", None) is not None:
            return
        if getattr(self, "_conda_create_flow_active", False):
            return
        self._conda_create_flow_active = True
        self.btn_new_conda.setEnabled(False)
        try:
            self._create_conda_env_flow()
        finally:
            self._conda_create_flow_active = False
            import shiboken6

            if shiboken6.isValid(self) and getattr(self, "_conda_create_worker", None) is None:
                self.btn_new_conda.setEnabled(True)

    def _create_conda_env_flow(self):
        """Create OpenBench conda environment."""
        from PySide6.QtWidgets import QVBoxLayout, QTextEdit

        if not self._ssh_manager or not self._ssh_manager.is_connected:
            QMessageBox.warning(self, "Error", "Please connect to server first using the Confirm button")
            return

        env_name = "openbench"

        # Check if environment already exists
        try:
            envs = call_responsive(self._ssh_manager.detect_conda_envs)
            env_exists = any(name.lower() == env_name for name, _ in envs)

            if env_exists:
                # Create custom message box with clear options
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Environment Exists")
                msg_box.setText(f"Conda environment '{env_name}' already exists.")
                msg_box.setInformativeText("Please choose an action:")

                msg_box.addButton("Delete and Recreate", QMessageBox.DestructiveRole)
                btn_use = msg_box.addButton("Use Existing", QMessageBox.AcceptRole)
                btn_cancel = msg_box.addButton("Cancel", QMessageBox.RejectRole)

                msg_box.exec()
                clicked = msg_box.clickedButton()

                if clicked == btn_cancel:
                    return
                elif clicked == btn_use:
                    # Use existing environment
                    self.conda_combo.clear()
                    self.conda_combo.addItem("(Not using conda environment)")
                    for name, path in envs:
                        self.conda_combo.addItem(name, path)
                    idx = self.conda_combo.findText(env_name)
                    if idx >= 0:
                        self.conda_combo.setCurrentIndex(idx)
                    QMessageBox.information(self, "Using Existing", f"Using existing '{env_name}' environment.")
                    return
                # else: Delete and recreate - continue below

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to check environments: {e}")
            return

        # Get Python path to find conda
        python_path = self.python_combo.currentText().strip()
        if not python_path:
            QMessageBox.warning(self, "Error", "Please detect or select a Python interpreter first")
            return

        # Determine conda path from python path
        import re

        conda_match = re.search(r"(.*/(miniconda|miniforge|anaconda|mambaforge)[^/]*)/bin/python", python_path)
        if not conda_match:
            QMessageBox.warning(
                self,
                "Error",
                "Cannot determine conda installation from Python path.\nPlease select a conda-based Python.",
            )
            return

        conda_base = conda_match.group(1)
        conda_exe = f"{conda_base}/bin/conda"

        # Progress dialog (Esc/close stays blocked until the worker finishes)
        progress_dialog = _InstallProgressDialog(self)
        progress_dialog.setWindowTitle("Creating Conda Environment")
        progress_dialog.resize(600, 400)
        progress_layout = QVBoxLayout(progress_dialog)

        status_label = QLabel("Creating environment...")
        progress_layout.addWidget(status_label)

        output_text = QTextEdit()
        output_text.setReadOnly(True)
        output_text.setStyleSheet("font-family: monospace;")
        progress_layout.addWidget(output_text)

        close_btn = QPushButton("Close")
        close_btn.setEnabled(False)
        close_btn.clicked.connect(progress_dialog.accept)
        progress_layout.addWidget(close_btn)

        self._conda_create_dialog = progress_dialog
        create_state = {"active": True}

        def mark_dialog_closed(*_):
            create_state["active"] = False

        progress_dialog.destroyed.connect(mark_dialog_closed)
        progress_dialog.show()

        quoted_conda_exe = _safe_remote_path(conda_exe)
        quoted_env_name = shlex.quote(env_name)

        worker_box = {}

        def _interrupted():
            worker = worker_box.get("worker")
            return worker is not None and worker.isInterruptionRequested()

        create_env_task = _build_conda_create_task(
            self._ssh_manager, quoted_conda_exe, quoted_env_name, env_exists, _interrupted
        )

        def finish_create_env(result):
            progress_dialog.allow_close = True
            if not create_state["active"]:
                self._conda_create_worker = None
                self._conda_create_dialog = None
                return
            output_text.append(result["output"])
            if result["exit_code"] == 0:
                status_label.setText("✓ Environment created successfully!")
                status_label.setStyleSheet("color: green; font-weight: bold;")
                self.conda_combo.clear()
                self.conda_combo.addItem("(Not using conda environment)")
                for name, path in result["envs"]:
                    self.conda_combo.addItem(name, path)
                idx = self.conda_combo.findText(env_name)
                if idx >= 0:
                    self.conda_combo.setCurrentIndex(idx)
            else:
                status_label.setText("✗ Failed to create environment")
                status_label.setStyleSheet("color: red; font-weight: bold;")
            self.btn_new_conda.setEnabled(True)
            close_btn.setEnabled(True)
            self._conda_create_worker = None
            self._conda_create_dialog = None

        def fail_create_env(message: str):
            progress_dialog.allow_close = True
            if not create_state["active"]:
                self._conda_create_worker = None
                self._conda_create_dialog = None
                return
            output_text.append(f"\nError: {message}")
            status_label.setText("✗ Error occurred!")
            status_label.setStyleSheet("color: red; font-weight: bold;")
            self.btn_new_conda.setEnabled(True)
            close_btn.setEnabled(True)
            self._conda_create_worker = None
            self._conda_create_dialog = None

        status_label.setText(f"Creating '{env_name}' environment with Python 3.12...")
        if env_exists:
            output_text.append(f"Will delete existing '{env_name}' before creating it.\n")

        worker = CallableWorker(create_env_task)
        worker_box["worker"] = worker
        self._conda_create_worker = worker
        worker.finished_with_result.connect(finish_create_env)
        worker.failed.connect(fail_create_env)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _cleanup_conda_create_worker(self, detach: bool = False):
        """Disconnect a long-running conda-create worker from this widget."""
        from openbench.gui.widgets._task_worker import safe_disconnect

        worker = getattr(self, "_conda_create_worker", None)
        if worker is not None:
            safe_disconnect(worker.finished_with_result, worker.failed)
            if worker.isRunning():
                worker.requestInterruption()
                if detach:
                    from openbench.gui.widgets._task_worker import detach_worker

                    detach_worker(worker, _DETACHED_TASK_WORKERS)
        self._conda_create_worker = None
        dialog = getattr(self, "_conda_create_dialog", None)
        self._conda_create_dialog = None
        if dialog is not None:
            try:
                dialog.close()
                dialog.deleteLater()
            except RuntimeError:
                pass

    def _cleanup_install_worker(self, detach: bool = False):
        """Disconnect a long-running OpenBench install/update worker from this widget."""
        from openbench.gui.widgets._task_worker import safe_disconnect

        worker = getattr(self, "_install_worker", None)
        if worker is not None:
            safe_disconnect(worker.line, worker.finished_with_result, worker.failed)
            if worker.isRunning():
                worker.requestInterruption()
                if detach:
                    from openbench.gui.widgets._task_worker import detach_worker

                    detach_worker(worker, _DETACHED_TASK_WORKERS)
        self._install_worker = None

    def closeEvent(self, event):
        self._cleanup_conda_create_worker(detach=True)
        self._cleanup_install_worker(detach=True)
        super().closeEvent(event)

    def _infer_conda_from_python(self, python_path: str):
        """Infer and display conda environment from Python path.

        Args:
            python_path: Path to Python interpreter
        """
        if not python_path:
            return

        # Check if this is a conda Python path
        # Pattern: /path/to/conda_install/bin/python -> base environment
        # Pattern: /path/to/conda_install/envs/ENV_NAME/bin/python -> ENV_NAME environment

        import re

        # Check for environment path: .../envs/ENV_NAME/bin/python
        env_match = re.search(r"/(miniconda|miniforge|anaconda|mambaforge)[^/]*/envs/([^/]+)/bin/python", python_path)
        if env_match:
            env_name = env_match.group(2)
            conda_type = env_match.group(1)
            # Update conda combo to show this environment
            self.conda_combo.clear()
            self.conda_combo.addItem(f"{env_name} ({conda_type} env)")
            self.conda_combo.setCurrentIndex(0)
            return

        # Check for base environment: .../miniconda*/bin/python
        base_match = re.search(r"/(miniconda|miniforge|anaconda|mambaforge)[^/]*/bin/python", python_path)
        if base_match:
            conda_type = base_match.group(1)
            # This is the base environment
            self.conda_combo.clear()
            self.conda_combo.addItem(f"base ({conda_type})")
            self.conda_combo.setCurrentIndex(0)
            return

        # Not a conda Python
        self.conda_combo.clear()
        self.conda_combo.addItem("(Not using conda environment)")
        self.conda_combo.setCurrentIndex(0)

    def _browse_openbench(self):
        """Open remote file browser to select OpenBench installation path."""
        from openbench.gui import path_utils

        if not self._ssh_manager or not self._ssh_manager.is_connected:
            QMessageBox.warning(self, "Error", "Please connect to server first using the Confirm button")
            return

        home = path_utils.remote_home_dir(self._ssh_manager)
        path = path_utils.pick_remote_path(
            self._ssh_manager, self, "Select OpenBench Directory on Remote Server", home, select_dirs=True
        )
        if path:
            self.openbench_input.setText(path)

    def _install_openbench(self):
        """Install OpenBench on remote server (re-entrancy-guarded entry point)."""
        if getattr(self, "_handshake_active", False):
            return  # SSH auth in flight; don't open channels mid-handshake
        if getattr(self, "_install_flow_active", False):
            return
        self._install_flow_active = True
        self.btn_install_ob.setEnabled(False)
        try:
            self._install_openbench_flow()
        finally:
            self._install_flow_active = False
            import shiboken6

            # The nested event loops can outlive the widget (app quit mid-install).
            if shiboken6.isValid(self) and getattr(self, "_install_worker", None) is None:
                self.btn_install_ob.setEnabled(True)

    def _install_openbench_flow(self):
        """Install OpenBench on remote server."""
        from PySide6.QtWidgets import (
            QDialog,
            QVBoxLayout,
            QHBoxLayout,
            QTextEdit,
            QPushButton,
            QLabel,
            QRadioButton,
            QButtonGroup,
        )

        if not self._ssh_manager or not self._ssh_manager.is_connected:
            QMessageBox.warning(self, "Error", "Please connect to server first using the Confirm button")
            return

        # Get installation path
        install_path = self.openbench_input.text().strip()
        if not install_path:
            try:
                home = call_responsive(self._ssh_manager._get_home_dir)
                install_path = f"{home}/OpenBench"
                self.openbench_input.setText(install_path)
            except Exception as e:
                logger.warning("Failed to get remote home directory for install path: %s", e)
                QMessageBox.warning(self, "Error", "Please specify an installation path for OpenBench")
                return

        # Check if git is available
        stdout, stderr, exit_code = execute_responsive(self._ssh_manager, "which git", timeout=10)
        if exit_code != 0:
            QMessageBox.warning(self, "Error", "Git is not installed on the remote server. Please install git first.")
            return

        # Check if path already exists
        is_update = False
        try:
            quoted_install_path = _safe_remote_path(install_path)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Path", str(exc))
            return
        stdout, stderr, exit_code = execute_responsive(
            self._ssh_manager, f"test -d {quoted_install_path} && echo exists", timeout=10
        )
        if "exists" in stdout:
            # Check if it's a git repository
            quoted_git_dir = _safe_remote_path(f"{install_path}/.git")
            stdout2, stderr2, exit_code2 = execute_responsive(
                self._ssh_manager, f"test -d {quoted_git_dir} && echo is_git", timeout=10
            )
            if "is_git" in stdout2:
                # It's a git repo, offer update
                reply = QMessageBox.question(
                    self,
                    "Directory Exists",
                    f"Directory {install_path} already exists and is a git repository.\n\nDo you want to update it with git pull?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if reply == QMessageBox.Yes:
                    is_update = True
                else:
                    return
            else:
                # Directory exists but not a git repo
                reply = QMessageBox.question(
                    self,
                    "Directory Exists",
                    f"Directory {install_path} already exists but is NOT a git repository.\n\nDo you want to DELETE it and install fresh?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply == QMessageBox.Yes:
                    # Delete the directory first (path already validated above)
                    stdout3, stderr3, exit_code3 = execute_responsive(
                        self._ssh_manager, f"rm -rf {quoted_install_path}", timeout=30
                    )
                    if exit_code3 != 0:
                        QMessageBox.warning(self, "Error", f"Failed to delete directory:\n{stderr3}")
                        return
                else:
                    return

        # Protocol selection dialog
        if not is_update:
            source_dialog = QDialog(self)
            source_dialog.setWindowTitle("Select Protocol")
            source_layout = QVBoxLayout(source_dialog)

            source_layout.addWidget(QLabel("Source: GitHub (github.com/zhongwangwei/OpenBench)"))
            source_layout.addWidget(QLabel("\nChoose protocol:"))

            protocol_group = QButtonGroup(source_dialog)
            radio_ssh = QRadioButton("SSH (git@github.com - Recommended if SSH key configured)")
            radio_https = QRadioButton("HTTPS (https://github.com)")
            radio_ssh.setChecked(True)
            protocol_group.addButton(radio_ssh)
            protocol_group.addButton(radio_https)
            source_layout.addWidget(radio_ssh)
            source_layout.addWidget(radio_https)

            btn_layout = QHBoxLayout()
            btn_layout.addStretch()
            btn_ok = QPushButton("OK")
            btn_ok.clicked.connect(source_dialog.accept)
            btn_cancel = QPushButton("Cancel")
            btn_cancel.clicked.connect(source_dialog.reject)
            btn_layout.addWidget(btn_ok)
            btn_layout.addWidget(btn_cancel)
            source_layout.addLayout(btn_layout)

            if source_dialog.exec() != QDialog.Accepted:
                return

            # Build repo URL based on protocol selection
            if radio_ssh.isChecked():
                repo_url = "git@github.com:zhongwangwei/OpenBench.git"
            else:
                repo_url = "https://github.com/zhongwangwei/OpenBench.git"
        else:
            repo_url = None  # Not needed for update

        # Progress dialog
        progress_dialog = _InstallProgressDialog(self)
        progress_dialog.setWindowTitle("Installing OpenBench" if not is_update else "Updating OpenBench")
        progress_dialog.resize(600, 400)
        progress_layout = QVBoxLayout(progress_dialog)

        status_label = QLabel("Starting..." if not is_update else "Updating...")
        progress_layout.addWidget(status_label)

        output_text = QTextEdit()
        output_text.setReadOnly(True)
        output_text.setStyleSheet("font-family: monospace;")
        progress_layout.addWidget(output_text)

        close_btn = QPushButton("Close")
        close_btn.setEnabled(False)
        close_btn.clicked.connect(progress_dialog.accept)
        progress_layout.addWidget(close_btn)

        progress_dialog.show()

        # Run installation
        self.btn_install_ob.setEnabled(False)

        if is_update:
            cmd = f"cd {quoted_install_path} && git pull --ff-only 2>&1"
            status_label.setText("Running git pull...")
        else:
            # repo_url comes from a fixed string (radio button choice),
            # but quote it anyway in case the source ever becomes
            # user-editable.
            quoted_repo_url = shlex.quote(repo_url)
            cmd = f"git clone --progress {quoted_repo_url} {quoted_install_path} 2>&1"
            status_label.setText(f"Cloning from {repo_url}...")

        def finish_install():
            self.btn_install_ob.setEnabled(True)
            close_btn.setEnabled(True)
            progress_dialog.allow_close = True
            self._install_worker = None

        def start_install_worker(command: str, timeout: int, on_done):
            output_text.append(f"$ {command}\n")
            # Deliberately unparented (like the conda-create worker): a child
            # QThread would be destroyed with the widget mid-run, aborting the
            # app. Lifetime is held by self._install_worker and, on detach,
            # by _DETACHED_TASK_WORKERS.
            worker = SshExecuteWorker(self._ssh_manager, command, timeout=timeout)
            self._install_worker = worker
            worker.line.connect(lambda line: output_text.append(str(line).rstrip("\n")))
            worker.finished_with_result.connect(on_done)
            worker.failed.connect(on_install_failed)
            worker.finished.connect(worker.deleteLater)
            worker.start()

        def on_install_failed(message: str):
            output_text.append(f"\nError: {message}")
            status_label.setText("✗ Error occurred!")
            status_label.setStyleSheet("color: red; font-weight: bold;")
            finish_install()

        def on_deps_done(exit_code: int, _stdout: str, _stderr: str):
            if exit_code == 0:
                status_label.setText("✓ Installation complete with all dependencies!")
                status_label.setStyleSheet("color: green; font-weight: bold;")
            else:
                status_label.setText("⚠ Repository ready but 'pip install -e' failed — see log above")
                status_label.setStyleSheet("color: orange; font-weight: bold;")
            finish_install()

        def on_git_done(exit_code: int, _stdout: str, _stderr: str):
            if exit_code != 0:
                status_label.setText("✗ Installation failed!" if not is_update else "✗ Update failed!")
                status_label.setStyleSheet("color: red; font-weight: bold;")
                finish_install()
                return

            status_label.setText("✓ Repository ready! Checking dependencies...")
            status_label.setStyleSheet("color: green;")

            try:
                # Install the package and its dependencies into the selected
                # interpreter's environment. The repo declares dependencies in
                # pyproject.toml (there is no requirements.yml), so
                # `pip install -e` is the one step that makes openbench
                # importable AND pulls xarray/netCDF4 for remote scans/runs.
                install_path = self.openbench_input.text().strip()
                python_path = self.python_combo.currentText().strip()

                if python_path:
                    output_text.append("\n\n=== Installing package and dependencies (pip install -e) ===\n")
                    status_label.setText("Installing dependencies with pip...")
                    pip_cmd = (
                        f"{_safe_remote_path(python_path)} -m pip install -e {_safe_remote_path(install_path)} 2>&1"
                    )
                    start_install_worker(pip_cmd, 900, on_deps_done)
                else:
                    output_text.append(
                        "\n\nNo Python environment selected — skipped dependency installation.\n"
                        "Configure Python/conda above, then run Install/Update again."
                    )
                    status_label.setText("✓ Clone successful! Select a Python environment to install dependencies.")
                    status_label.setStyleSheet("color: green; font-weight: bold;")
                    finish_install()
            except Exception as e:
                output_text.append(f"\nError: {e}")
                status_label.setText("✗ Error occurred!")
                status_label.setStyleSheet("color: red; font-weight: bold;")
                finish_install()

        # Reclaim the dialog once the user closes it (allow_close gates Esc
        # until the worker finishes); a plain deleteLater after exec() would
        # be swept early by nested event loops when exec is non-blocking.
        progress_dialog.finished.connect(progress_dialog.deleteLater)
        start_install_worker(cmd, 300, on_git_done)
        progress_dialog.exec()

    def get_ssh_manager(self) -> Optional[SSHManager]:
        """Get the current SSH manager instance.

        Returns:
            SSHManager instance if connected, None otherwise
        """
        return self._ssh_manager

    def is_connected(self) -> bool:
        """Check if currently connected to remote server.

        Returns:
            True if connected
        """
        return self._ssh_manager is not None and self._ssh_manager.is_connected

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary.

        Returns:
            Configuration dictionary with all settings
        """
        conda_env = ""
        conda_env_text = self.conda_combo.currentText()
        if conda_env_text and not conda_env_text.startswith("(Not"):
            # Extract env name from "envname (type)" format
            conda_env = conda_env_text.split()[0]

        return {
            "host": self.host_input.text().strip(),
            "auth_type": "password" if self.radio_password.isChecked() else "key",
            "key_file": self.key_input.text().strip(),
            "use_jump": self.node_group.isChecked(),
            "jump_node": self.node_input.text().strip(),
            "jump_auth": "password" if self.radio_node_password.isChecked() else "none",
            "num_cores": self.num_cores_spin.value(),
            "python_path": self.python_combo.currentText().strip(),
            "conda_env": conda_env,
            "openbench_path": self.openbench_input.text().strip(),
        }

    def set_config(self, config: Dict[str, Any]):
        """Set configuration from dictionary.

        Args:
            config: Configuration dictionary
        """
        # Block signals during batch update
        self.blockSignals(True)

        # Set host
        self.host_input.setText(config.get("host", ""))

        # Set auth type
        if config.get("auth_type") == "key":
            self.radio_key.setChecked(True)
        else:
            self.radio_password.setChecked(True)

        # Set key file
        self.key_input.setText(config.get("key_file", ""))

        # Set jump/compute node settings
        self.node_group.setChecked(config.get("use_jump", False))
        self.node_input.setText(config.get("jump_node", ""))

        if config.get("jump_auth") == "password":
            self.radio_node_password.setChecked(True)
        else:
            self.radio_node_none.setChecked(True)

        # Set num_cores
        self.num_cores_spin.setValue(config.get("num_cores", 4))

        # Set Python environment
        python_path = config.get("python_path", "")
        if python_path:
            idx = self.python_combo.findText(python_path)
            if idx >= 0:
                self.python_combo.setCurrentIndex(idx)
            else:
                self.python_combo.setCurrentText(python_path)

        # Set conda environment
        conda_env = config.get("conda_env", "")
        if conda_env:
            idx = self.conda_combo.findText(conda_env)
            if idx >= 0:
                self.conda_combo.setCurrentIndex(idx)
        else:
            self.conda_combo.setCurrentIndex(0)

        # Set OpenBench path
        self.openbench_input.setText(config.get("openbench_path", ""))

        # Restore signals
        self.blockSignals(False)

        # Try to load saved credentials for this host
        host = config.get("host", "")
        if host:
            self._load_saved_credentials(host)

    def _load_saved_credentials(self, host: str):
        """Load saved credentials for a host.

        Args:
            host: Host string to load credentials for
        """
        # Clear credential-derived fields first so switching to a host with
        # no saved credentials cannot leave a stale password/key visible.
        self.password_input.clear()
        self.cb_save_password.setChecked(False)
        self.key_input.clear()
        self.node_group.setChecked(False)
        self.node_input.clear()
        self.node_password_input.clear()
        self.node_key_input.clear()
        self.radio_node_none.setChecked(True)

        try:
            cred = self._credential_manager.get_credential(host)
        except CredentialStorageError as exc:
            QMessageBox.warning(self, "Credential Load Failed", str(exc))
            return

        if not cred:
            return

        auth_type = cred.get("auth_type")
        if auth_type == "key":
            self.radio_key.setChecked(True)
        else:
            self.radio_password.setChecked(True)

        # Load password if saved
        if cred.get("password"):
            self.password_input.setText(cred["password"])
            self.cb_save_password.setChecked(True)
        # Load key file if saved
        if cred.get("key_file"):
            self.key_input.setText(cred["key_file"])
        # Load jump node settings
        if cred.get("jump_node"):
            self.node_group.setChecked(True)
            self.node_input.setText(cred["jump_node"])
            if cred.get("jump_auth") == "password":
                self.radio_node_password.setChecked(True)

    def disconnect(self):
        """Disconnect from remote server."""
        # First disconnect compute node if connected
        self._disconnect_node(silent=True)

        # Disconnect main server
        if self._ssh_manager:
            try:
                self._ssh_manager.disconnect()
            except Exception:
                pass
            self._ssh_manager = None

        # Update UI - reset all button states
        self.status_label.setText("Not connected")
        self.status_label.setStyleSheet("color: #999;")
        self.btn_test.setEnabled(True)
        self.btn_disconnect.setEnabled(False)
        self.connection_status_changed.emit(False)

    def clear_credentials(self):
        """Clear all saved credentials."""
        self._credential_manager.clear_all()
        self.password_input.clear()
        self.cb_save_password.setChecked(False)

    def reset_to_defaults(self):
        """Reset all remote config fields to defaults."""
        # Host and authentication
        self.host_input.clear()
        self.password_input.clear()
        self.key_input.clear()
        self.radio_password.setChecked(True)
        self.cb_save_password.setChecked(False)

        # Compute node
        self.node_group.setChecked(False)
        self.node_input.clear()
        self.node_password_input.clear()
        self.node_key_input.clear()
        self.radio_node_password.setChecked(True)

        # Environment
        self.num_cores_spin.setValue(4)
        self.conda_combo.clear()
        self.conda_combo.addItem("(Not using conda environment)")
        self.conda_combo.setCurrentIndex(0)
        self.python_combo.clear()
        self.openbench_input.clear()
