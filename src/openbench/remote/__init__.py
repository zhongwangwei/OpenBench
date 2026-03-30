"""SSH infrastructure for remote execution (requires openbench[remote]).

This package requires paramiko. If not installed, importing submodules
will raise ImportError with an installation hint.
"""


def _check_remote_deps():
    """Check that remote dependencies are available."""
    try:
        import paramiko  # noqa: F401
    except ImportError:
        raise ImportError(
            "Remote execution requires paramiko. Install with: pip install 'openbench[remote]'"
        ) from None
