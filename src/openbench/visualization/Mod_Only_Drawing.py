"""Compatibility alias for the read-only visualization renderers.

The implementation moved to :mod:`openbench.visualization.only_drawing` so the
legacy CamelCase module no longer carries a second 2k-line drawing path.  The
module object is aliased (not copied) to preserve existing monkeypatch/import
behavior for downstream callers that still import ``Mod_Only_Drawing``.
"""

from __future__ import annotations

import sys

from . import only_drawing as _impl

sys.modules[__name__] = _impl
