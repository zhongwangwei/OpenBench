#!/usr/bin/env python3
"""
Download MERIT Hydro flow direction data and resample to 1-minute resolution

MERIT Hydro data source: http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/
The flow direction data is at 3 arc-second resolution (~90m)

For CaMa-Flood 1min maps, we need to resample from 3" to 1' (20x aggregation)

Flow direction encoding (D8):
    1 = East (E)
    2 = Southeast (SE)
    4 = South (S)
    8 = Southwest (SW)
   16 = West (W)
   32 = Northwest (NW)
   64 = North (N)
  128 = Northeast (NE)
    0 = River mouth / Ocean
   -1 = Inland pit

Note: MERIT Hydro uses power-of-2 encoding, while CaMa-Flood uses 1-8 encoding:
    MERIT:  64(N) 128(NE) 1(E) 2(SE) 4(S) 8(SW) 16(W) 32(NW)
    CaMa:    1(N)   2(NE) 3(E) 4(SE) 5(S) 6(SW)  7(W)  8(NW)
"""
import os
import sys
import numpy as np
import requests
from pathlib import Path

# MERIT Hydro download URL template
# Data is distributed in 5x5 degree tiles
MERIT_URL = "http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/distribute/v1.0.1/"

# Flow direction file naming: dir_{tile}.tar
# Where tile is like n30e120 (5x5 degree)

def merit_d8_to_cama_d8(merit_val):
    """Convert MERIT Hydro D8 encoding to CaMa-Flood encoding

    MERIT uses power-of-2:
        64=N, 128=NE, 1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 0=outlet, -1=inland

    CaMa uses 1-8:
        1=N, 2=NE, 3=E, 4=SE, 5=S, 6=SW, 7=W, 8=NW, 0=outlet, -9=invalid
    """
    mapping = {
        64: 1,    # North
        128: 2,   # Northeast
        1: 3,     # East
        2: 4,     # Southeast
        4: 5,     # South
        8: 6,     # Southwest
        16: 7,    # West
        32: 8,    # Northwest
        0: 0,     # River mouth / outlet
        -1: -9,   # Inland pit -> invalid
        247: -9,  # No data
        255: -9,  # No data (unsigned)
    }
    return mapping.get(merit_val, -9)


def resample_flwdir_3sec_to_1min(flwdir_3sec, method='mode'):
    """Resample 3 arc-second flow direction to 1 arc-minute

    Args:
        flwdir_3sec: 3" flow direction array (shape should be multiple of 20)
        method: 'mode' for most common direction, 'center' for center pixel

    Returns:
        1' flow direction array (20x smaller in each dimension)
    """
    ny_3sec, nx_3sec = flwdir_3sec.shape

    # 1' = 60", 3" -> 60"/3" = 20 pixels per 1'
    scale = 20

    ny_1min = ny_3sec // scale
    nx_1min = nx_3sec // scale

    flwdir_1min = np.zeros((ny_1min, nx_1min), dtype=np.int8)

    if method == 'center':
        # Use center pixel of each 20x20 block
        for iy in range(ny_1min):
            for ix in range(nx_1min):
                # Center pixel (10, 10) of the 20x20 block
                iy_3sec = iy * scale + scale // 2
                ix_3sec = ix * scale + scale // 2
                flwdir_1min[iy, ix] = flwdir_3sec[iy_3sec, ix_3sec]

    elif method == 'mode':
        # Use most common direction in each 20x20 block
        from scipy import stats

        for iy in range(ny_1min):
            for ix in range(nx_1min):
                block = flwdir_3sec[iy*scale:(iy+1)*scale, ix*scale:(ix+1)*scale]
                # Get valid values (not -9 or 0)
                valid = block[(block > 0) & (block <= 8)]
                if len(valid) > 0:
                    mode_result = stats.mode(valid, keepdims=True)
                    flwdir_1min[iy, ix] = mode_result.mode[0]
                else:
                    # No valid flow direction, check for outlet
                    if np.any(block == 0):
                        flwdir_1min[iy, ix] = 0
                    else:
                        flwdir_1min[iy, ix] = -9

    return flwdir_1min


def print_usage():
    """Print usage instructions for obtaining MERIT Hydro data"""
    print("""
================================================================================
How to Generate 1min.flwdir.bin from MERIT Hydro
================================================================================

The flow direction data needs to come from MERIT Hydro (3 arc-second resolution)
and be resampled to 1 arc-minute resolution.

OPTION 1: Download MERIT Hydro from official source
---------------------------------------------------------------------------
1. Go to: http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/
2. Register and download the flow direction (dir) tiles
3. Run this script to process them

OPTION 2: Use Google Earth Engine (faster for global data)
---------------------------------------------------------------------------
1. Use the Earth Engine dataset: MERIT/Hydro/v1_0_1
2. Export the 'dir' band to GeoTIFF
3. Run this script to process it

OPTION 3: Generate from CaMa-Flood map generation tools
---------------------------------------------------------------------------
If you have access to the CaMa-Flood map generation tools (map/src/),
the flow direction is generated during the map creation process.

================================================================================
""")


def check_merit_hydro_availability():
    """Check if MERIT Hydro data is available locally or needs download"""

    # Common locations to check
    possible_paths = [
        "/Volumes/Data01/MERIT_Hydro",
        "/Users/zhongwangwei/Data/MERIT_Hydro",
        os.path.expanduser("~/Data/MERIT_Hydro"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found MERIT Hydro data at: {path}")
            return path

    return None


if __name__ == "__main__":
    print("=" * 70)
    print("MERIT Hydro Flow Direction Data Preparation")
    print("=" * 70)

    # Check for existing MERIT Hydro data
    merit_path = check_merit_hydro_availability()

    if merit_path is None:
        print("\nMERIT Hydro data not found locally.")
        print_usage()
        sys.exit(1)

    # Process the data
    print(f"\nProcessing MERIT Hydro data from: {merit_path}")
    # TODO: Add actual processing logic
