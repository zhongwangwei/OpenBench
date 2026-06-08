"""OpenBench GUI application bootstrap."""

from contextlib import ExitStack
from importlib.resources import as_file, files
import os
import sys


def _prepare_stylesheet() -> tuple[str, ExitStack]:
    """Load the bundled GUI stylesheet in source, wheel, and zip layouts.

    Returns the stylesheet plus an ExitStack that owns any temporary files
    materialized by importlib.resources.as_file(). The caller must keep the
    stack alive for as long as Qt may resolve URLs inside the stylesheet.
    """
    resource_stack = ExitStack()
    style_dir = files("openbench.gui") / "styles"
    theme = style_dir / "theme.qss"
    if not theme.is_file():
        return "", resource_stack

    stylesheet = theme.read_text(encoding="utf-8")
    checkmark = style_dir / "checkmark.png"
    if checkmark.is_file():
        checkmark_path = resource_stack.enter_context(as_file(checkmark))
        stylesheet = stylesheet.replace("CHECKMARK_PATH", str(checkmark_path).replace("\\", "/"))
    return stylesheet, resource_stack


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

    stylesheet, stylesheet_resources = _prepare_stylesheet()
    try:
        if stylesheet:
            app.setStyleSheet(stylesheet)

        from openbench.gui.main_window import MainWindow

        window = MainWindow()
        if config_path:
            window._load_config_file(config_path)
        window.show()

        # No auto-scan on startup — user triggers scan via Reference Data page button

        sys.exit(app.exec())
    finally:
        stylesheet_resources.close()
