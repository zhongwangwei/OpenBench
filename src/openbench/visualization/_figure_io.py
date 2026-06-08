"""Safe figure output helpers for visualization renderers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure


def save_figure(fig: Figure, path: str | os.PathLike[str], **kwargs: Any) -> None:
    """Save *fig* to *path* via a same-directory temporary file.

    This makes standalone renderers create their output directory and avoids
    leaving a partially written final image if Matplotlib/Pillow fails midway.
    The temporary file keeps the final suffix so backends that infer behavior
    from the extension still work when callers omit an explicit format.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix or ".tmp"
    tmp_file = tempfile.NamedTemporaryFile(
        delete=False,
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=suffix,
    )
    tmp_path = Path(tmp_file.name)
    tmp_file.close()

    try:
        fig.savefig(tmp_path, **kwargs)
        os.replace(tmp_path, output_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
