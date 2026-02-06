"""Coordinate reference system conversion utilities."""

import numpy as np
from typing import Union
from pyproj import Transformer

Numeric = Union[float, np.ndarray]

_transformers = {}

def _get_transformer(from_crs: str) -> Transformer:
    if from_crs not in _transformers:
        _transformers[from_crs] = Transformer.from_crs(from_crs, "EPSG:4326", always_xy=True)
    return _transformers[from_crs]

def reproject_to_wgs84(x: Numeric, y: Numeric, from_crs: str) -> tuple:
    """Reproject coordinates to WGS84 (EPSG:4326). Returns (lon, lat)."""
    if from_crs.upper() in ("WGS84", "EPSG:4326"):
        return x, y
    transformer = _get_transformer(from_crs)
    lon, lat = transformer.transform(x, y)
    return lon, lat

def validate_wgs84(lon: float, lat: float) -> bool:
    """Check if coordinates are valid WGS84."""
    return -180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0
