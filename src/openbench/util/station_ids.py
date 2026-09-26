"""Station IDs are labels: read them as text and match them regardless of zero padding.

Lists written by different tools spell the same station differently
("0000000009463" vs "9463", or an integer NetCDF coordinate). Keeping the
written text preserves leading zeros for output and file names, while
``station_id_key`` lets the two spellings still match.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

_ID_COLUMN_NAMES = frozenset({"id", "site_id", "site", "station_id"})
_KEY_COLUMN = "_station_id_key"


def station_id_key(value: Any) -> str:
    """Comparison key: the stripped text, with leading zeros dropped from all-digit IDs."""
    text = str(value).strip()
    if text.isascii() and text.isdigit():
        return text.lstrip("0") or "0"
    return text


def read_station_csv(path, **kwargs) -> pd.DataFrame:
    """Read a station list or metadata CSV, keeping every ID-like column as text."""
    header_kwargs = {key: value for key, value in kwargs.items() if key != "dtype"}
    columns = pd.read_csv(path, nrows=0, **header_kwargs).columns
    dtype = {column: str for column in columns if str(column).strip().lower() in _ID_COLUMN_NAMES}
    extra = kwargs.pop("dtype", None) or {}
    return pd.read_csv(path, dtype={**dtype, **extra}, **kwargs)


def merge_on_station_id(left: pd.DataFrame, right: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Inner-join two station tables on ID, matching zero-padded and unpadded IDs.

    The result keeps the left table's ID text, like a plain merge on "ID".
    """
    left = left.assign(**{_KEY_COLUMN: left["ID"].map(station_id_key)})
    right = right.drop(columns="ID").assign(**{_KEY_COLUMN: right["ID"].map(station_id_key)})
    kwargs.setdefault("how", "inner")
    return pd.merge(left, right, on=_KEY_COLUMN, **kwargs).drop(columns=_KEY_COLUMN)
