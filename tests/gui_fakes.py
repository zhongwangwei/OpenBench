"""Shared fake Qt widgets for GUI tests (no QApplication required).

Extend this module instead of hand-rolling another per-file fake when a
test needs a text-bearing widget stand-in.
"""


class FakeControllerBase:
    """Provides WizardController.remote_settings() semantics for controller fakes."""

    config: dict = {}

    def remote_settings(self) -> dict:
        general = self.config.get("general") or {}
        return general.get("remote") or {}


class FakeButton:
    """Stand-in for QPushButton state APIs: setEnabled/setVisible/setText."""

    def __init__(self):
        self.enabled = None
        self.visible = None
        self.text = None

    def setEnabled(self, value):
        self.enabled = value

    def setVisible(self, value):
        self.visible = value

    def setText(self, value):
        self.text = value


class FakeLineEdit:
    """Stand-in for QLineEdit/QComboBox text APIs: text()/setText()/currentText()."""

    def __init__(self, value: str = ""):
        self.value = value

    def text(self) -> str:
        return self.value

    def currentText(self) -> str:
        return self.value

    def setText(self, value: str) -> None:
        self.value = value
