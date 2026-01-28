#!/usr/bin/env python3
"""
Convert MERIT Hydro flow direction (3 arc-second) to CaMa-Flood 1min format

MERIT Hydro D8 encoding (power-of-2):
    64 = North (N)
   128 = Northeast (NE)
     1 = East (E)
     2 = Southeast (SE)
     4 = South (S)
     8 = Southwest (SW)
    16 = West (W)
    32 = Northwest (NW)
     0 = River mouth / Ocean
    -1 = Inland pit
   255 = No data (in uint8)

CaMa-Flood D8 encoding (1-8):
     1 = North (N)
     2 = Northeast (NE)
     3 = East (E)
     4 = Southeast (SE)
     5 = South (S)
     6 = Southwest (SW)
     7 = West (W)
     8 = Northwest (NW)
     0 = Outlet
    -9 = Invalid / Ocean

Usage:
    python convert_merit_flwdir.py <merit_flwdir.tif> <output_dir>

    Or for regional tiles:
    python convert_merit_flwdir.py <merit_tiles_dir> <output_dir> --tiles
"""
import os
import sys
import numpy as np
from pathlib import Path

# D8 encoding conversion: MERIT -> CaMa
MERIT_TO_CAMA = {
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
    247: -9,  # No data (signed)
    255: -9,  # No data (unsigned)
}


def convert_d8_encoding(merit_data):
    """Convert MERIT D8 encoding to CaMa D8 encoding (vectorized)"""
    result = np.full_like(merit_data, -9, dtype=np.int8)

    for merit_val, cama_val in MERIT_TO_CAMA.items():
        if merit_val >= 0:
            result[merit_data == merit_val] = cama_val
        else:
            # Handle signed comparison
            result[merit_data.astype(np.int16) == merit_val] = cama_val

    return result


def resample_flwdir_3sec_to_1min(flwdir_3sec, method='mode'):
    """
    Resample 3 arc-second flow direction to 1 arc-minute

    Args:
        flwdir_3sec: 3" flow direction array (already in CaMa encoding)
        method: 'mode' for most common direction, 'center' for center pixel

    Returns:
        1' flow direction array (20x smaller in each dimension)
    """
    ny_3sec, nx_3sec = flwdir_3sec.shape

    # 1' = 60", 3" -> 60"/3" = 20 pixels per 1'
    scale = 20

    ny_1min = ny_3sec // scale
    nx_1min = nx_3sec // scale

    print(f"  Resampling: {nx_3sec}x{ny_3sec} (3\") -> {nx_1min}x{ny_1min} (1')")

    if method == 'center':
        # Vectorized: take center pixel of each 20x20 block
        # Center is at offset (10, 10) within each block
        flwdir_1min = flwdir_3sec[10::scale, 10::scale].copy()
        # Ensure correct shape
        flwdir_1min = flwdir_1min[:ny_1min, :nx_1min].astype(np.int8)
        print(f"  Using center pixel method (vectorized)")

    elif method == 'mode':
        # Vectorized mode calculation using reshape
        print(f"  Using mode method (vectorized)...")

        # Reshape to (ny_1min, scale, nx_1min, scale) then to (ny_1min, nx_1min, scale*scale)
        # Trim to exact multiple of scale
        trimmed = flwdir_3sec[:ny_1min*scale, :nx_1min*scale]
        reshaped = trimmed.reshape(ny_1min, scale, nx_1min, scale)
        blocks = reshaped.transpose(0, 2, 1, 3).reshape(ny_1min, nx_1min, scale*scale)

        # Initialize output
        flwdir_1min = np.full((ny_1min, nx_1min), -9, dtype=np.int8)

        # Count occurrences of each direction (1-8) in each block
        print(f"    Counting direction frequencies...")
        for d in range(1, 9):
            counts_d = np.sum(blocks == d, axis=2)
            if d == 1:
                counts = counts_d[:, :, np.newaxis]
            else:
                counts = np.concatenate([counts, counts_d[:, :, np.newaxis]], axis=2)

        # Find mode (most frequent direction)
        max_counts = np.max(counts, axis=2)
        mode_dir = np.argmax(counts, axis=2) + 1  # +1 because directions are 1-8

        # Apply mode where valid directions exist
        has_valid = max_counts > 0
        flwdir_1min[has_valid] = mode_dir[has_valid]

        # Check for outlets where no valid direction
        has_outlet = np.any(blocks == 0, axis=2)
        flwdir_1min[~has_valid & has_outlet] = 0

        print(f"    Done!")

    elif method == 'outlet':
        # Outlet-priority method
        print(f"  Using outlet-priority method...")

        trimmed = flwdir_3sec[:ny_1min*scale, :nx_1min*scale]
        reshaped = trimmed.reshape(ny_1min, scale, nx_1min, scale)
        blocks = reshaped.transpose(0, 2, 1, 3).reshape(ny_1min, nx_1min, scale*scale)

        flwdir_1min = np.full((ny_1min, nx_1min), -9, dtype=np.int8)

        # First check for outlets
        has_outlet = np.any(blocks == 0, axis=2)
        flwdir_1min[has_outlet] = 0

        # Then compute mode for non-outlet blocks
        for d in range(1, 9):
            counts_d = np.sum(blocks == d, axis=2)
            if d == 1:
                counts = counts_d[:, :, np.newaxis]
            else:
                counts = np.concatenate([counts, counts_d[:, :, np.newaxis]], axis=2)

        max_counts = np.max(counts, axis=2)
        mode_dir = np.argmax(counts, axis=2) + 1

        has_valid = (max_counts > 0) & ~has_outlet
        flwdir_1min[has_valid] = mode_dir[has_valid]

    return flwdir_1min


def process_geotiff(input_file, output_dir, method='center'):
    """Process a single GeoTIFF file containing MERIT Hydro flow direction"""
    try:
        import rasterio
    except ImportError:
        print("Error: rasterio required. Install with: pip install rasterio")
        sys.exit(1)

    print(f"Processing: {input_file}")

    with rasterio.open(input_file) as src:
        merit_data = src.read(1)
        transform = src.transform
        crs = src.crs

        print(f"  Input shape: {merit_data.shape}")
        print(f"  Input dtype: {merit_data.dtype}")
        print(f"  Value range: [{merit_data.min()}, {merit_data.max()}]")

    # Step 1: Convert D8 encoding
    print("  Converting D8 encoding (MERIT -> CaMa)...")
    cama_data = convert_d8_encoding(merit_data)

    # Step 2: Resample from 3" to 1'
    print(f"  Resampling to 1' using '{method}' method...")
    flwdir_1min = resample_flwdir_3sec_to_1min(cama_data, method=method)

    # Print statistics
    print("\n  D8 direction distribution (1min):")
    for d in range(-9, 9):
        count = np.sum(flwdir_1min == d)
        if count > 0:
            pct = 100 * count / flwdir_1min.size
            print(f"    {d:3d}: {count:>12,} ({pct:.2f}%)")

    # Save output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Binary file
    bin_file = output_dir / "1min.flwdir.bin"
    print(f"\n  Writing {bin_file}...")
    with open(bin_file, 'wb') as f:
        flwdir_1min.tofile(f)
    print(f"  Created {bin_file} ({os.path.getsize(bin_file):,} bytes)")

    # NetCDF file
    try:
        import netCDF4 as nc
        nc_file = output_dir / "1min.flwdir.nc"
        print(f"  Writing {nc_file}...")

        if nc_file.exists():
            nc_file.unlink()

        ds = nc.Dataset(nc_file, 'w', format='NETCDF4')
        ny, nx = flwdir_1min.shape
        ds.createDimension('y', ny)
        ds.createDimension('x', nx)

        var = ds.createVariable('flwdir', 'i1', ('y', 'x'), zlib=True, complevel=4)
        var[:] = flwdir_1min
        var.description = 'D8 flow direction'
        var.encoding = '1=N, 2=NE, 3=E, 4=SE, 5=S, 6=SW, 7=W, 8=NW, 0=outlet, -9=invalid'
        var.source = 'Converted from MERIT Hydro'

        ds.close()
        print(f"  Created {nc_file}")
    except ImportError:
        print("  Warning: netCDF4 not available, skipping .nc output")

    return flwdir_1min


def process_merit_tiles(tiles_dir, output_dir, method='center'):
    """
    Process MERIT Hydro tiles (5x5 degree) and combine into global 1min

    MERIT Hydro tiles are named like: n30e120_dir.tif (or dir_n30e120.tif)
    Each tile is 5x5 degrees = 6000x6000 pixels at 3"
    """
    tiles_dir = Path(tiles_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Global 1min dimensions
    nx_1min, ny_1min = 21600, 10800  # 360*60, 180*60

    # Initialize global array
    global_flwdir = np.full((ny_1min, nx_1min), -9, dtype=np.int8)

    # Find all tile files
    tile_patterns = ['*_dir.tif', 'dir_*.tif', '*flwdir*.tif']
    tile_files = []
    for pattern in tile_patterns:
        tile_files.extend(tiles_dir.glob(pattern))

    if not tile_files:
        print(f"No MERIT Hydro tiles found in {tiles_dir}")
        print("Expected patterns: *_dir.tif, dir_*.tif, *flwdir*.tif")
        return None

    print(f"Found {len(tile_files)} tiles")

    try:
        import rasterio
    except ImportError:
        print("Error: rasterio required. Install with: pip install rasterio")
        sys.exit(1)

    for tile_file in sorted(tile_files):
        print(f"\nProcessing tile: {tile_file.name}")

        with rasterio.open(tile_file) as src:
            merit_data = src.read(1)
            bounds = src.bounds

            # Get tile bounds
            west = bounds.left
            east = bounds.right
            south = bounds.bottom
            north = bounds.top

            print(f"  Bounds: ({west}, {south}) to ({east}, {north})")
            print(f"  Shape: {merit_data.shape}")

        # Convert D8 encoding
        cama_data = convert_d8_encoding(merit_data)

        # Resample to 1min
        flwdir_1min = resample_flwdir_3sec_to_1min(cama_data, method=method)

        # Calculate position in global array
        # Global: west=-180, north=90
        ix_start = int((west - (-180)) * 60)
        iy_start = int((90 - north) * 60)

        ny_tile, nx_tile = flwdir_1min.shape

        print(f"  Global position: ix={ix_start}, iy={iy_start}")

        # Insert into global array
        global_flwdir[iy_start:iy_start+ny_tile, ix_start:ix_start+nx_tile] = flwdir_1min

    # Save global output
    print("\n" + "=" * 60)
    print("Saving global 1min.flwdir...")

    # Print statistics
    print("\nD8 direction distribution (global):")
    for d in range(-9, 9):
        count = np.sum(global_flwdir == d)
        if count > 0:
            pct = 100 * count / global_flwdir.size
            print(f"  {d:3d}: {count:>12,} ({pct:.2f}%)")

    # Binary file
    bin_file = output_dir / "1min.flwdir.bin"
    print(f"\nWriting {bin_file}...")
    with open(bin_file, 'wb') as f:
        global_flwdir.tofile(f)
    print(f"Created {bin_file} ({os.path.getsize(bin_file):,} bytes)")

    # NetCDF file
    try:
        import netCDF4 as nc
        nc_file = output_dir / "1min.flwdir.nc"
        print(f"Writing {nc_file}...")

        if nc_file.exists():
            nc_file.unlink()

        ds = nc.Dataset(nc_file, 'w', format='NETCDF4')
        ds.createDimension('y', ny_1min)
        ds.createDimension('x', nx_1min)

        var = ds.createVariable('flwdir', 'i1', ('y', 'x'), zlib=True, complevel=4)
        var[:] = global_flwdir
        var.description = 'D8 flow direction'
        var.encoding = '1=N, 2=NE, 3=E, 4=SE, 5=S, 6=SW, 7=W, 8=NW, 0=outlet, -9=invalid'
        var.source = 'Converted from MERIT Hydro tiles'

        ds.close()
        print(f"Created {nc_file}")
    except ImportError:
        print("Warning: netCDF4 not available, skipping .nc output")

    return global_flwdir


def print_usage():
    print("""
Usage:
  1. Process single GeoTIFF (global or regional):
     python convert_merit_flwdir.py <merit_flwdir.tif> <output_dir>

  2. Process MERIT Hydro tiles directory:
     python convert_merit_flwdir.py <tiles_dir> <output_dir> --tiles

  3. Use mode aggregation (slower but more accurate):
     python convert_merit_flwdir.py <input> <output_dir> --method mode

Options:
  --tiles     Process directory of 5x5 degree tiles
  --method    Aggregation method: 'center' (default, fast) or 'mode' (slow, accurate)

MERIT Hydro Download:
  1. Official source: http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/
  2. Google Earth Engine: ee.Image('MERIT/Hydro/v1_0_1').select('dir')
""")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print_usage()
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]

    # Parse options
    use_tiles = '--tiles' in sys.argv
    method = 'center'
    if '--method' in sys.argv:
        idx = sys.argv.index('--method')
        if idx + 1 < len(sys.argv):
            method = sys.argv[idx + 1]

    print("=" * 60)
    print("MERIT Hydro to CaMa-Flood Flow Direction Converter")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Method: {method}")
    print(f"Tiles mode: {use_tiles}")
    print("=" * 60)

    if use_tiles:
        process_merit_tiles(input_path, output_dir, method=method)
    else:
        process_geotiff(input_path, output_dir, method=method)

    print("\nDone!")
