"""Most-common-label regridding of categorical data (land cover, climate zones)."""

from collections.abc import Hashable
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .. import utils
from ._shared import (
    construct_intervals,
    reduce_data_to_new_domain,
    restore_properties,
)


def _bin_index(coord_values: np.ndarray, intervals: pd.IntervalIndex) -> np.ndarray:
    """Index of the left-closed target interval holding each value; -1 outside."""
    breaks = np.append(intervals.left.to_numpy(), intervals.right[-1])
    index = np.searchsorted(breaks, coord_values, side="right") - 1
    index[(index < 0) | (index >= len(intervals))] = -1
    return index


def compute_mode(
    data: xr.DataArray,
    target_ds: xr.Dataset,
    values: np.ndarray,
    time_dim: str | None,
    fill_value: None | Any = None,
) -> xr.DataArray:
    """Upsample the input data using a "most common label" (mode) approach.

    Source cells are binned into the target cells and labels are counted with a
    single ``np.bincount``. Labels absent from ``values`` are ignored; ties go to
    the label listed first in ``values``, which is also returned for a target
    cell without any counted source cell.

    Args:
        data: Input DataArray, with an integer data type. If your data does not consist
            of integer type values, you will have to encode them to integer types.
            It is assumed that the coordinates of this data are sorted.
        target_ds: Dataset which coordinates the input dataset should be regrid to.
        values: Numpy array containing all labels expected to be in the input
            data. For example, `np.array([0, 2, 4])`, if the data only contains the
            values 0, 2 and 4.
        time_dim: Name of the time dimension. Defaults to "time". Use `None` to force
            regridding over the time dimension.
        fill_value: What value to fill uncovered parts of the target grid. By default
            this will be NaN, and integer type data will be cast to float to accomodate
            this.

    Raises:
        ValueError: if the input data is not of an integer dtype.

    Returns:
        xarray.DataArray with regridded categorical data.
    """
    # Must be categorical data (integers)
    if not np.issubdtype(data.dtype, np.integer):
        msg = (
            "Your input data has to be of an integer datatype for this method.\n"
            f"    instead, your data is of type '{data.dtype}'."
            "You can convert the data with:\n        `dataset.astype(int)`."
        )
        raise ValueError(msg)

    coords: list[Hashable] = utils.common_coords(data, target_ds, remove_coord=time_dim)
    target_coords = xr.Dataset(target_ds.coords)  # stores coords for reindexing later
    sorted_target_coords = target_coords.sortby(coords)

    bounds = [construct_intervals(sorted_target_coords[coord].to_numpy()) for coord in coords]

    data = reduce_data_to_new_domain(data, sorted_target_coords, coords)
    for coord in coords:
        if data[coord].dims != (coord,):
            raise ValueError(f"Most-common regridding requires a 1-D dimension coordinate {coord!r}")

    labels = pd.Index(values.astype(data.dtype))
    shape = tuple(len(interval) for interval in bounds)
    target_size = int(np.prod(shape))
    # Target cell of every source grid point, independent of any leading dims.
    axis_index = [_bin_index(data[coord].to_numpy(), interval) for coord, interval in zip(coords, bounds)]
    grids = np.meshgrid(*axis_index, indexing="ij")
    inside = np.all([grid >= 0 for grid in grids], axis=0)
    cell = np.where(inside, np.ravel_multi_index([np.maximum(grid, 0) for grid in grids], shape), -1).ravel()

    def most_common(block: np.ndarray) -> np.ndarray:
        leading = block.shape[: block.ndim - len(coords)]
        flat = block.reshape(-1, cell.size)
        out = np.empty((flat.shape[0], target_size), dtype=data.dtype)
        # One slice at a time keeps the count table at target cells x labels.
        for row, source in enumerate(flat):
            label = labels.get_indexer(source)
            valid = (label >= 0) & (cell >= 0)
            counts = np.bincount(cell[valid] * len(labels) + label[valid], minlength=target_size * len(labels))
            out[row] = labels.to_numpy()[counts.reshape(target_size, len(labels)).argmax(axis=1)]
        return out.reshape(*leading, *shape)

    bin_dims = [f"{coord}_bins" for coord in coords]
    result = xr.apply_ufunc(
        most_common,
        data,
        input_core_dims=[coords],
        output_core_dims=[bin_dims],
        dask="parallelized",
        output_dtypes=[data.dtype],
        dask_gufunc_kwargs={"output_sizes": dict(zip(bin_dims, shape)), "allow_rechunk": True},
    )

    result = restore_properties(result, data, target_ds, coords, fill_value)
    result = result.reindex_like(target_coords, copy=False)
    return result
