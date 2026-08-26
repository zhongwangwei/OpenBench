"""Wizard GUI (requires colm-openbench[gui]).

Use ``_check_gui_deps`` before importing GUI submodules so optional runtime
dependencies fail once with an actionable installation hint.
"""

from importlib import import_module


def _check_gui_deps():
    """Check that GUI dependencies are available."""
    missing = []
    for module in ("PySide6", "psutil", "paramiko", "cryptography"):
        try:
            import_module(module)
        except (ImportError, OSError):
            missing.append(module)
    if missing:
        raise ImportError(
            f"Missing or broken GUI dependencies: {', '.join(missing)}. "
            "Install them together with: pip install 'colm-openbench[gui]'"
        ) from None
