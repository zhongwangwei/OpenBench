"""Custom station filter for the ResOpsUS dataset."""

import logging
from pathlib import Path

import pandas as pd


def _resolve_station_list(info) -> Path:
    """Return the station list path that matches the routing grid resolution."""
    try:
        sim_grid_res = float(info.sim_grid_res)
    except Exception as exc:  # pragma: no cover - defensive
        logging.error(f"Invalid sim_grid_res for ResOpsUS: {info.sim_grid_res} ({exc})")
        raise

    file_map = {
        0.25: "ResOpsUS_reservoir_matched_15min.csv",
        0.1: "ResOpsUS_reservoir_matched_06min.csv",
    }
    if sim_grid_res not in file_map:
        logging.error("ResOpsUS only supports sim_grid_res values of 0.25 or 0.1 degrees.")
        raise ValueError(f"Unsupported sim_grid_res: {sim_grid_res}")

    filename = file_map[sim_grid_res]
    candidates = [
        Path(info.ref_dir) / "list" / filename,
        Path(info.ref_dir).parent / "list" / filename,
    ]

    for station_list in candidates:
        if station_list.exists():  # pragma: no cover - IO guard
            return station_list

    logging.error(f"ResOpsUS station list not found. Tried: {', '.join(str(p) for p in candidates)}")
    raise FileNotFoundError(candidates[0])


def filter_ResOpsUS(info, ds=None):
    """Load and filter ResOpsUS station metadata based on simulation settings.
    
    Args:
        info: Configuration/info object with processing parameters
        ds: Optional xarray Dataset to filter (for data filtering mode)
        
    Returns:
        For data filtering mode: Tuple of (info, filtered_data)
        For initialization mode: None (modifies info in place)
    """
    # If ds is provided, we're in data filtering mode
    if ds is not None:
        # Return the first data variable or the dataset as-is
        data_vars = list(ds.data_vars)
        if data_vars:
            return info, ds[data_vars[0]]
        return info, ds
    
    # Initialization mode: load and filter station metadata
    station_list_path = _resolve_station_list(info)
    info.ref_fulllist = str(station_list_path)

    df = pd.read_csv(station_list_path)
    df.rename(
        columns={
            "SYEAR": "ref_syear",
            "EYEAR": "ref_eyear",
            "LON": "ref_lon",
            "LAT": "ref_lat",
            "DIR": "ref_dir",
        },
        inplace=True,
    )

    df["ref_syear"] = pd.to_numeric(df["ref_syear"], errors="coerce")
    df["ref_eyear"] = pd.to_numeric(df["ref_eyear"], errors="coerce")
    df["ref_lon"] = pd.to_numeric(df["ref_lon"], errors="coerce")
    df["ref_lat"] = pd.to_numeric(df["ref_lat"], errors="coerce")
    df["Flag"] = df.get("Flag", True).astype(bool)

    sim_syear = pd.Series([int(info.sim_syear)] * len(df))
    sim_eyear = pd.Series([int(info.sim_eyear)] * len(df))
    syear_series = pd.Series([int(info.syear)] * len(df))
    eyear_series = pd.Series([int(info.eyear)] * len(df))

    df["use_syear"] = (
        pd.concat([df["ref_syear"], sim_syear, syear_series], axis=1).max(axis=1).astype(int)
    )
    df["use_eyear"] = (
        pd.concat([df["ref_eyear"], sim_eyear, eyear_series], axis=1).min(axis=1).astype(int)
    )

    duration_ok = (df["use_eyear"] - df["use_syear"]) >= info.min_year
    lat_ok = (df["ref_lat"] >= info.min_lat) & (df["ref_lat"] <= info.max_lat)
    lon_ok = (df["ref_lon"] >= info.min_lon) & (df["ref_lon"] <= info.max_lon)
    df["Flag"] = df["Flag"] & duration_ok & lat_ok & lon_ok

    info.stn_list = df[df["Flag"]].copy()
    if info.stn_list.empty:
        logging.error("No ResOpsUS stations match the selection criteria.")
        raise ValueError("No ResOpsUS stations selected")

    info.use_syear = int(info.stn_list["use_syear"].min())
    info.use_eyear = int(info.stn_list["use_eyear"].max())

