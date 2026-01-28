#!/usr/bin/env python3
"""
Generate flwdir (D8 flow direction) from downxy (downstream XY coordinates)

The downxy.bin file contains two records:
- downXX: X coordinate of downstream cell (int16)
- downYY: Y coordinate of downstream cell (int16)

D8 direction encoding (same as CaMa-Flood):
    8 | 1 | 2
    --|---|--
    7 | 0 | 3
    --|---|--
    6 | 5 | 4

Where 0 means no movement (outlet or sink)
-9 means invalid (ocean, outside domain)
"""
import os
import numpy as np
import netCDF4 as nc

def generate_flwdir_from_downxy(cama_dir, map_name="glb_15min"):
    """Generate flwdir.bin from downxy.bin"""

    hires_path = f"{cama_dir}/{map_name}/1min"

    # Global 1min dimensions
    nx, ny = 21600, 10800
    shape = (ny, nx)

    print(f"Generating flwdir from downxy...")
    print(f"  Dimensions: {nx} x {ny}")

    # Read downxy
    downxy_file = f"{hires_path}/1min.downxy.bin"
    if not os.path.exists(downxy_file):
        print(f"Error: {downxy_file} not found")
        return False

    print(f"  Reading {downxy_file}...")
    with open(downxy_file, 'rb') as f:
        downXX = np.fromfile(f, dtype=np.int16, count=nx*ny).reshape(shape)
        downYY = np.fromfile(f, dtype=np.int16, count=nx*ny).reshape(shape)

    print(f"  downXX range: [{downXX.min()}, {downXX.max()}]")
    print(f"  downYY range: [{downYY.min()}, {downYY.max()}]")

    # Create coordinate arrays for current cell positions (1-based)
    # ix[j, i] = i+1, iy[j, i] = j+1
    ix = np.arange(1, nx+1, dtype=np.int16).reshape(1, nx)
    ix = np.broadcast_to(ix, shape)

    iy = np.arange(1, ny+1, dtype=np.int16).reshape(ny, 1)
    iy = np.broadcast_to(iy, shape)

    # Calculate dx and dy (downstream - current)
    dx = downXX - ix
    dy = downYY - iy

    print(f"  dx range: [{dx.min()}, {dx.max()}]")
    print(f"  dy range: [{dy.min()}, {dy.max()}]")

    # Initialize flwdir array with -9 (invalid/ocean)
    flwdir = np.full(shape, -9, dtype=np.int8)

    # D8 direction mapping based on (dx, dy):
    # (-1,-1)=8  (0,-1)=1  (1,-1)=2
    # (-1, 0)=7  (0, 0)=0  (1, 0)=3
    # (-1, 1)=6  (0, 1)=5  (1, 1)=4

    # Only process valid cells (downXX > 0 and downYY > 0)
    valid = (downXX > 0) & (downYY > 0)

    # Map (dx, dy) to D8 direction
    # North (0, -1) -> 1
    flwdir[(dx == 0) & (dy == -1) & valid] = 1
    # Northeast (1, -1) -> 2
    flwdir[(dx == 1) & (dy == -1) & valid] = 2
    # East (1, 0) -> 3
    flwdir[(dx == 1) & (dy == 0) & valid] = 3
    # Southeast (1, 1) -> 4
    flwdir[(dx == 1) & (dy == 1) & valid] = 4
    # South (0, 1) -> 5
    flwdir[(dx == 0) & (dy == 1) & valid] = 5
    # Southwest (-1, 1) -> 6
    flwdir[((dx == -1) & (dy == 1)) & valid] = 6
    # West (-1, 0) -> 7
    flwdir[(dx == -1) & (dy == 0) & valid] = 7
    # Northwest (-1, -1) -> 8
    flwdir[(dx == -1) & (dy == -1) & valid] = 8
    # No movement (outlet/sink, dx=0, dy=0) -> 0 (or could be ocean)
    flwdir[(dx == 0) & (dy == 0) & valid] = 0

    # Count direction distribution
    print("\n  D8 direction distribution:")
    for d in range(-9, 9):
        count = np.sum(flwdir == d)
        if count > 0:
            print(f"    {d:3d}: {count:>12,} ({100*count/(nx*ny):.2f}%)")

    # Write binary file
    flwdir_bin = f"{hires_path}/1min.flwdir.bin"
    print(f"\n  Writing {flwdir_bin}...")
    with open(flwdir_bin, 'wb') as f:
        flwdir.tofile(f)
    print(f"  Created {flwdir_bin} ({os.path.getsize(flwdir_bin):,} bytes)")

    # Also write NetCDF
    flwdir_nc = f"{hires_path}/1min.flwdir.nc"
    print(f"\n  Writing {flwdir_nc}...")

    # Remove existing nc file if exists
    if os.path.exists(flwdir_nc):
        os.remove(flwdir_nc)

    ds = nc.Dataset(flwdir_nc, 'w', format='NETCDF4')
    ds.createDimension('y', ny)
    ds.createDimension('x', nx)

    var = ds.createVariable('flwdir', 'i1', ('y', 'x'), zlib=True, complevel=4)
    var[:] = flwdir
    var.description = 'D8 flow direction (1-8 for 8 directions, 0 for outlet, -9 for invalid)'
    var.encoding = 'D8: 1=N, 2=NE, 3=E, 4=SE, 5=S, 6=SW, 7=W, 8=NW, 0=outlet, -9=invalid'

    ds.close()
    print(f"  Created {flwdir_nc}")

    print("\nDone!")
    return True


if __name__ == "__main__":
    # Use the symlink data directory
    cama_dir = "./data_for_wse/cama_maps"

    print("=" * 60)
    print("Generate flwdir from downxy")
    print("=" * 60)

    generate_flwdir_from_downxy(cama_dir, "glb_15min")
