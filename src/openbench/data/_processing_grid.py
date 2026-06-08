"""Grid processing helper facade."""

from __future__ import annotations

from openbench.data._processing_grid_core import (
    GridProcessingCoreMixin,
    write_mfdataset_chunked_atomic as _write_mfdataset_chunked_atomic,
)
from openbench.data._processing_grid_regrid import GridRegridMixin


class GridProcessingMixin(GridProcessingCoreMixin, GridRegridMixin):
    """Grid dataset preparation, remapping, and station extraction helpers."""

    pass


def write_mfdataset_chunked_atomic(*args, **kwargs):
    """Compatibility facade for chunked multi-file NetCDF writes."""
    return _write_mfdataset_chunked_atomic(*args, **kwargs)
