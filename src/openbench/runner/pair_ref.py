"""Per-pair reference-file lifecycle helpers for the local runner."""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)


def cleanup_pair_ref_overrides(tasks: list[dict[str, Any]]) -> None:
    """Remove temporary per-pair reference copies created during preprocessing."""
    for task in tasks:
        pair_ref = task.get("ref_file_override")
        if pair_ref and os.path.lexists(pair_ref):
            try:
                os.remove(pair_ref)
            except OSError:
                logger.debug("Could not remove temporary per-pair ref file: %s", pair_ref)


def clone_or_link_ref_for_pair(
    src: str,
    dst: str,
    *,
    creators: Iterable[tuple[str, Callable[[str, str], bool]]] | None = None,
    copy2_fn: Callable[[str, str], object] | None = None,
) -> str:
    """Create a per-pair ref file using CoW/link strategies before full copy.

    The per-pair path is later rewritten via atomic ``os.replace`` by unified
    masking. That makes hardlinks/symlinks safe here: the write replaces the
    pair path itself rather than mutating the shared flat ref inode.

    Existing destinations are always removed first because per-pair files are
    temporary masked refs; reusing a stale file can silently over-mask a later
    run.

    Returns the strategy used: ``clonefile``, ``reflink``, ``hardlink``,
    ``symlink``, or ``copy2``.
    """
    if os.path.lexists(dst):
        remove_partial_pair_ref(dst)
    dst_dir = os.path.dirname(dst)
    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)

    if creators is None:
        creators = (
            ("clonefile", try_clonefile),
            ("reflink", try_reflink),
            ("hardlink", try_hardlink),
            ("symlink", try_symlink),
        )

    for name, creator in creators:
        try:
            if creator(src, dst):
                return name
        except OSError:
            remove_partial_pair_ref(dst)

    (copy2_fn or shutil.copy2)(src, dst)
    return "copy2"


def remove_partial_pair_ref(path: str) -> None:
    try:
        if os.path.lexists(path):
            os.remove(path)
    except OSError:
        logger.debug("Could not remove partial per-pair ref path: %s", path)


def try_clonefile(src: str, dst: str) -> bool:
    """Try macOS/APFS clonefile copy-on-write."""
    import ctypes
    import ctypes.util

    libc_path = ctypes.util.find_library("c")
    if not libc_path:
        return False
    libc = ctypes.CDLL(libc_path, use_errno=True)
    clonefile = getattr(libc, "clonefile", None)
    if clonefile is None:
        return False
    clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32]
    clonefile.restype = ctypes.c_int
    result = clonefile(os.fsencode(src), os.fsencode(dst), 0)
    if result == 0:
        return True
    errno_value = ctypes.get_errno()
    if errno_value:
        raise OSError(errno_value, os.strerror(errno_value), dst)
    return False


def try_reflink(src: str, dst: str) -> bool:
    """Try Linux FICLONE reflink copy-on-write."""
    if os.name != "posix":
        return False
    try:
        import fcntl
    except ImportError:  # pragma: no cover
        return False

    ficlone = 0x40049409
    with open(src, "rb") as src_fh, open(dst, "wb") as dst_fh:
        try:
            fcntl.ioctl(dst_fh.fileno(), ficlone, src_fh.fileno())
        except OSError:
            raise
    shutil.copystat(src, dst, follow_symlinks=True)
    return True


def try_hardlink(src: str, dst: str) -> bool:
    os.link(src, dst)
    return True


def try_symlink(src: str, dst: str) -> bool:
    os.symlink(os.path.abspath(src), dst)
    return True
