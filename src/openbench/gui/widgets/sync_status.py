# -*- coding: utf-8 -*-
"""
Sync status indicator widget for remote mode.
"""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Signal, QTimer, Qt

from openbench.remote.sync import SyncStatus


class SyncStatusWidget(QWidget):
    """Widget showing sync status with retry button."""

    retry_clicked = Signal()
    # Thread-safe signal for updating status from background threads
    # Use Qt.QueuedConnection when connecting to ensure main thread execution
    status_update_requested = Signal(object, int)  # (SyncStatus, pending_count)

    STATUS_COLORS = {
        SyncStatus.SYNCED: "#27ae60",  # Green
        SyncStatus.PENDING: "#f39c12",  # Yellow/Orange
        SyncStatus.SYNCING: "#3498db",  # Blue
        SyncStatus.ERROR: "#e74c3c",  # Red
    }

    STATUS_TEXT = {
        SyncStatus.SYNCED: "Synced",
        SyncStatus.PENDING: "Pending...",
        SyncStatus.SYNCING: "Syncing...",
        SyncStatus.ERROR: "Sync Error",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._status = SyncStatus.SYNCED
        self._pending_count = 0
        self._setup_ui()

        # Animation timer for syncing state
        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._animate)
        self._animation_frame = 0

        # Connect thread-safe signal to set_status with queued connection
        # This ensures updates from background threads are executed in main thread
        self.status_update_requested.connect(self._on_status_update_requested, Qt.QueuedConnection)

    def _on_status_update_requested(self, status, pending_count):
        """Handle thread-safe status update request."""
        self.set_status(status, pending_count)

    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Status indicator (colored dot)
        self._dot = QLabel()
        self._dot.setFixedSize(12, 12)
        self._dot.setStyleSheet(self._get_dot_style(SyncStatus.SYNCED))
        layout.addWidget(self._dot)

        # Status text
        self._text = QLabel("Synced")
        self._text.setStyleSheet("color: #666666; font-size: 12px;")
        layout.addWidget(self._text)

        # Retry button (hidden by default)
        self._retry_btn = QPushButton("Retry")
        self._retry_btn.setFixedWidth(60)
        self._retry_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 3px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #c0392b; }
        """)
        self._retry_btn.clicked.connect(self.retry_clicked.emit)
        self._retry_btn.setVisible(False)
        layout.addWidget(self._retry_btn)

        layout.addStretch()

    def _get_dot_style(self, status: SyncStatus) -> str:
        """Get the stylesheet for the status dot."""
        color = self.STATUS_COLORS.get(status, "#666666")
        return f"""
            background-color: {color};
            border-radius: 6px;
        """

    def set_status(self, status: SyncStatus, pending_count: int = 0):
        """
        Update the sync status display.

        Args:
            status: The current sync status
            pending_count: Number of files pending sync
        """
        self._status = status
        self._pending_count = pending_count

        # Update dot color
        self._dot.setStyleSheet(self._get_dot_style(status))

        # Update text
        text = self.STATUS_TEXT.get(status, "Unknown")
        if status == SyncStatus.PENDING and pending_count > 0:
            text = f"Pending ({pending_count})"
        self._text.setText(text)

        # Show/hide retry button
        self._retry_btn.setVisible(status == SyncStatus.ERROR)

        # Start/stop animation
        if status == SyncStatus.SYNCING:
            self._animation_timer.start(200)
        else:
            self._animation_timer.stop()
            self._animation_frame = 0

    def _animate(self):
        """Animate the syncing indicator."""
        self._animation_frame = (self._animation_frame + 1) % 4
        dots = "." * (self._animation_frame + 1)
        self._text.setText(f"Syncing{dots}")

    def set_hidden_when_synced(self, hidden: bool):
        """
        Hide the widget when status is synced.

        Args:
            hidden: If True, hide when status is SYNCED
        """
        if hidden and self._status == SyncStatus.SYNCED:
            self.setVisible(False)
        else:
            self.setVisible(True)

    def get_status(self) -> SyncStatus:
        """Get the current sync status."""
        return self._status

    def get_pending_count(self) -> int:
        """Get the current pending count."""
        return self._pending_count

    def cleanup(self):
        """Clean up resources before destruction.

        Call this method before deleteLater() to ensure proper cleanup.
        """
        # Stop animation timer
        if hasattr(self, "_animation_timer") and self._animation_timer:
            self._animation_timer.stop()
            self._animation_timer.timeout.disconnect()

        # Disconnect our own signals to prevent callbacks after deletion
        try:
            self.status_update_requested.disconnect()
        except (RuntimeError, TypeError):
            # Already disconnected or never connected
            pass

        try:
            self.retry_clicked.disconnect()
        except (RuntimeError, TypeError):
            pass

    def closeEvent(self, event):
        """Handle widget close - stop animation timer."""
        self.cleanup()
        super().closeEvent(event)

    def __del__(self):
        """Destructor - ensure timer is stopped."""
        try:
            if hasattr(self, "_animation_timer") and self._animation_timer:
                self._animation_timer.stop()
        except RuntimeError:
            # Timer may already be deleted by Qt
            pass
