"""OpenBench GUI application bootstrap."""

import os
import sys


def launch(config_path=None):
    """Launch the OpenBench GUI application.

    Args:
        config_path: Optional path to an openbench.yaml to load on startup.
    """
    from openbench.gui import _check_gui_deps

    _check_gui_deps()

    from PySide6.QtWidgets import QApplication

    # X11 forwarding support
    if sys.platform != "win32":
        os.environ.setdefault("LIBGL_ALWAYS_INDIRECT", "1")
        os.environ.setdefault("QT_QUICK_BACKEND", "software")

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("OpenBench")
    app.setOrganizationName("CoLM-SYSU")

    # Load stylesheet
    style_dir = os.path.join(os.path.dirname(__file__), "styles")
    theme_path = os.path.join(style_dir, "theme.qss")
    checkmark_path = os.path.join(style_dir, "checkmark.png")

    if os.path.exists(theme_path):
        with open(theme_path) as f:
            stylesheet = f.read()
        stylesheet = stylesheet.replace("CHECKMARK_PATH", checkmark_path.replace("\\", "/"))
        app.setStyleSheet(stylesheet)

    from openbench.gui.main_window import MainWindow

    window = MainWindow()
    if config_path:
        window._load_config_file(config_path)
    window.show()

    sys.exit(app.exec())
