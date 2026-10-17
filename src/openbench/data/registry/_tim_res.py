"""Time-resolution frequency ranks, kept import-light for the CLI checks."""

from __future__ import annotations

_TIM_RES_RANK = {
    "climatology-year": 0,
    "climatology_year": 0,
    "year": 0,
    "yearly": 0,
    "y": 0,
    "climatology-month": 1,
    "climatology_month": 1,
    "month": 1,
    "monthly": 1,
    "m": 1,
    "mon": 1,
    "8day": 2,
    "8daily": 2,
    "week": 2,
    "weekly": 2,
    "w": 2,
    "day": 3,
    "daily": 3,
    "d": 3,
    "6hour": 4,
    "6h": 4,
    "6hourly": 4,
    "3hour": 5,
    "3h": 5,
    "3hourly": 5,
    "hour": 6,
    "hourly": 6,
    "h": 6,
    "30min": 7,
    "30mins": 7,
    "30minute": 7,
    "30minutes": 7,
    "halfhour": 7,
    "half-hour": 7,
}


def _tim_res_rank(tim_res: str) -> int:
    """Return the frequency rank for a time resolution string."""
    return _TIM_RES_RANK.get(tim_res.lower().strip(), -1) if tim_res else -1
