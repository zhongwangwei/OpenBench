import logging
import math
import os

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


def _canonical_sim_grid_res(value):
    """Return the supported HydroWeb routing resolution matching a user value."""
    valid_resolutions = [0.25, 0.0167, 0.0833, 0.1, 0.05]
    for resolution in valid_resolutions:
        if math.isclose(float(value), resolution, rel_tol=1e-6, abs_tol=1e-8):
            return resolution
    raise ValueError(f"HydroWeb filter: sim_grid_res {value} not in valid set {valid_resolutions}")


def process_station(station, info):
    result = {
        "Flag": False,
        "use_syear": -9999,
        "use_eyear": -9999,
        "obs_syear": -9999,
        "obs_eyear": -9999,
        "ref_dir": "file",
    }
    # info.compare_tim_res arrives normalized via processing._normalize_frequency,
    # so "Day" → "D" → .lower() == "d". Previously checked "1d" which never
    # matched — caused all stations to fall through to Flag=False and the
    # initialization mode then sys.exit on "No stations selected".
    if info.compare_tim_res.lower() == "d":
        file_path = f"{info.ref_dir}/output/river/hydroprd_river_{station['ID']}.nc"
    else:
        return result
    if os.path.exists(file_path):
        result["ref_dir"] = file_path
        with xr.open_dataset(file_path) as df:
            if info.debug_mode:
                logger.info("Processing station %s...", int(station["ID"]))
            years = pd.to_datetime(df["time"].values).year
            result["obs_syear"] = int(years[0])
            result["obs_eyear"] = int(years[-1])
            result["use_syear"] = max(result["obs_syear"], int(info.sim_syear), int(info.syear))
            result["use_eyear"] = min(result["obs_eyear"], int(info.sim_eyear), int(info.eyear))
            if (
                (result["use_eyear"] - result["use_syear"] >= info.min_year)
                and (station["lon"] >= info.min_lon)
                and (station["lon"] <= info.max_lon)
                and (station["lat"] >= info.min_lat)
                and (station["lat"] <= info.max_lat)
            ):
                result["Flag"] = True
                if info.debug_mode:
                    logger.info("Station %s is selected", int(station["ID"]))
    return result


def filter_HydroWeb(info, ds=None):
    """Generate required station metadata for HydroWeb runs or filter dataset.

    Args:
        info: Configuration/info object with processing parameters
        ds: Optional xarray Dataset to filter (for data filtering mode)

    Returns:
        For data filtering mode: Tuple of (info, filtered_data)
        For initialization mode: None (modifies info in place)

    Raises:
        ValueError: For configuration errors that prevent station list generation.
            Filters must NOT call sys.exit — that kills the entire runner mid-run,
            making partial failures appear as silent dataset-not-found errors.
    """
    # If ds is provided, we're in data filtering mode
    if ds is not None:
        data_vars = list(ds.data_vars)
        if data_vars:
            return info, ds[data_vars[0]]
        return info, ds

    # Initialization mode
    if info.compare_tim_res.lower() != "d":
        raise ValueError(f"HydroWeb filter requires compare_tim_res=Day; got {info.compare_tim_res!r}")

    # Confirm the resolution of info.sim_grid_res
    if not hasattr(info, "sim_grid_res"):
        raise ValueError("HydroWeb filter requires info.sim_grid_res; not defined")

    if not isinstance(info.sim_grid_res, (int, float)):
        raise ValueError(f"HydroWeb filter: sim_grid_res must be a number, got {type(info.sim_grid_res).__name__}")

    logger.info(
        "The simulation grid resolution of routing model can be different from the LSM grid resolution. "
        "Using sim_grid_res = %s from configuration. "
        "Valid values: 0.25(15min), 0.0167(1min), 0.0833(5min), 0.1(6min), 0.05(3min)",
        info.sim_grid_res,
    )
    info.sim_grid_res = _canonical_sim_grid_res(info.sim_grid_res)

    if info.debug_mode:
        logger.info("Using simulation grid resolution: %s degrees", info.sim_grid_res)
    if info.sim_grid_res == 0.25:
        info.ref_fulllist = f"{info.ref_dir}/list/HydroWeb_alloc_15min.txt"
    elif info.sim_grid_res == 0.0167:
        info.ref_fulllist = f"{info.ref_dir}/list/HydroWeb_alloc_1min.txt"
    elif info.sim_grid_res == 0.0833:
        info.ref_fulllist = f"{info.ref_dir}/list/HydroWeb_alloc_5min.txt"
    elif info.sim_grid_res == 0.1:
        info.ref_fulllist = f"{info.ref_dir}/list/HydroWeb_alloc_6min.txt"
    elif info.sim_grid_res == 0.05:
        info.ref_fulllist = f"{info.ref_dir}/list/HydroWeb_alloc_3min.txt"
    station_list = pd.read_csv(f"{info.ref_fulllist}", delimiter=r"\s+", header=0)

    results = Parallel(n_jobs=-1)(delayed(process_station)(row, info) for _, row in station_list.iterrows())
    for i, result in enumerate(results):
        for key, value in result.items():
            station_list.at[i, key] = value

    ind = station_list[station_list["Flag"]].index
    data_select = station_list.loc[ind].copy()

    if info.sim_grid_res == 0.25:
        lat0 = np.arange(89.875, -90, -0.25)
        lon0 = np.arange(-179.875, 180, 0.25)
    elif info.sim_grid_res == 0.0167:  # 01min
        lat0 = np.arange(89.9916666666666600, -90, -0.0166666666666667)
        lon0 = np.arange(-179.9916666666666742, 180, 0.0166666666666667)
    elif info.sim_grid_res == 0.0833:  # 05min
        lat0 = np.arange(89.9583333333333286, -90, -0.0833333333333333)
        lon0 = np.arange(-179.9583333333333428, 180, 0.0833333333333333)
    elif info.sim_grid_res == 0.1:  # 06min
        lat0 = np.arange(89.95, -90, -0.1)
        lon0 = np.arange(-179.95, 180, 0.1)
    elif info.sim_grid_res == 0.05:  # 03min
        lat0 = np.arange(89.975, -90, -0.05)
        lon0 = np.arange(-179.975, 180, 0.05)
    data_select["lon_cama"] = -9999.0
    data_select["lat_cama"] = -9999.0
    for idx, row in data_select.iterrows():
        lon_cama = float(lon0[int(row["ix1"]) - 1])
        lat_cama = float(lat0[int(row["iy1"]) - 1])
        data_select.at[idx, "lon_cama"] = lon_cama
        data_select.at[idx, "lat_cama"] = lat_cama
        if abs(lat_cama - float(row["lat"])) > 1:
            logger.warning("ID %s lat does not match (cama vs station)", row["ID"])
        if abs(lon_cama - float(row["lon"])) > 1:
            logger.warning("ID %s lon does not match (cama vs station)", row["ID"])
    logger.info("HydroWeb filter selected %d stations", len(data_select["ID"]))
    if len(data_select["ID"]) == 0:
        raise ValueError(
            "HydroWeb filter selected zero stations. Check the station list and "
            "the min_year / min_lat / max_lat / min_lon / max_lon thresholds."
        )
    info.use_syear = data_select["use_syear"].min()
    info.use_eyear = data_select["use_eyear"].max()
    data_select["ref_lon"] = data_select["lon_cama"]
    data_select["ref_lat"] = data_select["lat_cama"]
    # all year here should be int
    data_select["use_syear"] = data_select["use_syear"].astype(int)
    data_select["use_eyear"] = data_select["use_eyear"].astype(int)
    data_select["obs_syear"] = data_select["obs_syear"].astype(int)
    data_select["obs_eyear"] = data_select["obs_eyear"].astype(int)
    data_select["ID"] = data_select["ID"]  # .astype(int)

    data_select.to_csv(f"{info.casedir}/stn_HydroWeb_{info.sim_source}_list.txt", index=False)
