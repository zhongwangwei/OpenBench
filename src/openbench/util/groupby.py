"""xarray group-by helpers shared by processing, scoring and statistics."""

from __future__ import annotations

import contextlib
import threading

import xarray as xr

_NO_FLOX_LOCK = threading.RLock()


@contextlib.contextmanager
def without_flox():
    """Use xarray's built-in group-by/resample reductions instead of flox.

    flox is not an OpenBench dependency, but xarray uses it whenever it is
    installed (common in existing environments). Without numbagg, flox moves the
    grouped axis last and aggregates through flattened index arrays; month/season
    group-bys and resamples measured 5-10x slower than xarray's slice-based path
    on gridded and multi-station data, with results equal up to float rounding.

    ``set_options`` is process-global, so the re-entrant lock keeps concurrent
    callers from restoring each other's state out of order while still allowing
    nested use. Dask graphs are built inside the block, so the choice also holds
    when they are computed later.
    """
    with _NO_FLOX_LOCK:
        try:
            options = xr.set_options(use_flox=False)
        except ValueError:  # xarray without flox integration
            options = contextlib.nullcontext()
        with options:
            yield
