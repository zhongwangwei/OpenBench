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
