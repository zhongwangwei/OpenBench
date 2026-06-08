"""Wizard GUI (requires colm-openbench[gui]).

This package requires PySide6. If not installed, importing submodules
will raise ImportError with an installation hint.
"""


def _check_gui_deps():
    """Check that GUI dependencies are available."""
    try:
        import PySide6  # noqa: F401
    except ImportError:
        raise ImportError(
            "GUI requires PySide6. Install with: pip install 'colm-openbench[gui]' (or install PySide6 in your conda environment)"
        ) from None
