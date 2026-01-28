#!/usr/bin/env python3
"""
Convert CaMa-Flood base binary files to NetCDF format
"""
import os
import numpy as np
import netCDF4 as nc

def convert_cama_base_files(cama_dir, map_name="glb_15min"):
    """Convert base CaMa binary files to NetCDF"""

    base_path = f"{cama_dir}/{map_name}"

    # Read params.txt
    params_file = f"{base_path}/params.txt"
    with open(params_file, 'r') as f:
        lines = f.readlines()
        nXX = int(lines[0].split('!!')[0].strip())
        nYY = int(lines[1].split('!!')[0].strip())
        gsize = float(lines[3].split('!!')[0].strip())
        west = float(lines[4].split('!!')[0].strip())
        east = float(lines[5].split('!!')[0].strip())
        south = float(lines[6].split('!!')[0].strip())
        north = float(lines[7].split('!!')[0].strip())

    print(f"CaMa parameters: nXX={nXX}, nYY={nYY}, gsize={gsize}")
    print(f"Bounds: west={west}, east={east}, south={south}, north={north}")

    shape = (nYY, nXX)

    # Files to convert
    files_to_convert = {
        'uparea': ('uparea.bin', np.float32, 'Upstream area', 'm2'),
        'basin': ('basin.bin', np.int32, 'Basin ID', '1'),
        'elevtn': ('elevtn.bin', np.float32, 'Elevation', 'm'),
        'nxtdst': ('nxtdst.bin', np.float32, 'Distance to next cell', 'm'),
        'biftag': ('biftag.bin', np.int32, 'Bifurcation tag', '1'),
    }

    # Convert nextxy separately (2 records)
    nextxy_file = f"{base_path}/nextxy.bin"
    nextxy_nc = f"{base_path}/nextxy.nc"

    if os.path.exists(nextxy_file) and not os.path.exists(nextxy_nc):
        print(f"Converting {nextxy_file}...")
        with open(nextxy_file, 'rb') as f:
            nextXX = np.fromfile(f, dtype=np.int32, count=nXX*nYY).reshape(shape)
            nextYY = np.fromfile(f, dtype=np.int32, count=nXX*nYY).reshape(shape)

        ds = nc.Dataset(nextxy_nc, 'w', format='NETCDF4')
        ds.createDimension('y', nYY)
        ds.createDimension('x', nXX)

        var = ds.createVariable('nextXX', 'i4', ('y', 'x'))
        var[:] = nextXX
        var.description = 'Next cell X index'

        var = ds.createVariable('nextYY', 'i4', ('y', 'x'))
        var[:] = nextYY
        var.description = 'Next cell Y index'

        ds.close()
        print(f"  Created {nextxy_nc}")

    # Convert other files
    for var_name, (bin_file, dtype, desc, units) in files_to_convert.items():
        bin_path = f"{base_path}/{bin_file}"
        nc_path = f"{base_path}/{var_name}.nc"

        if not os.path.exists(bin_path):
            print(f"Warning: {bin_path} not found, skipping")
            continue

        if os.path.exists(nc_path):
            print(f"  {nc_path} already exists, skipping")
            continue

        print(f"Converting {bin_path}...")

        with open(bin_path, 'rb') as f:
            data = np.fromfile(f, dtype=dtype).reshape(shape)

        ds = nc.Dataset(nc_path, 'w', format='NETCDF4')
        ds.createDimension('y', nYY)
        ds.createDimension('x', nXX)

        var = ds.createVariable(var_name, 'f4' if dtype == np.float32 else 'i4', ('y', 'x'))
        var[:] = data
        var.description = desc
        var.units = units

        ds.close()
        print(f"  Created {nc_path}")

    print("Base file conversion complete!")
    return True


def convert_hires_files(cama_dir, map_name="glb_15min", tag="1min"):
    """Convert high-resolution regional files to NetCDF"""

    base_path = f"{cama_dir}/{map_name}"
    hires_path = f"{base_path}/{tag}"

    if not os.path.exists(hires_path):
        print(f"High-res directory {hires_path} not found")
        return False

    # Determine dimensions based on tag
    # For global data: 1min = 21600 x 10800 (360*60 x 180*60)
    if tag == "1min":
        # Check if this is global or regional data
        test_file = f"{hires_path}/{tag}.catmzz.bin"
        if os.path.exists(test_file):
            file_size = os.path.getsize(test_file)
            if file_size == 21600 * 10800:  # Global 1min data (int8)
                nx, ny = 21600, 10800
                print(f"Detected global 1min data: {nx}x{ny}")
            else:
                nx = ny = 600  # Regional 10 degrees / (1/60) = 600
        else:
            nx = ny = 600
    elif tag == "3sec":
        nx = ny = 1200  # 10 degrees / (3/3600) = 1200
    elif tag == "15sec":
        nx = ny = 2400
    else:
        print(f"Unknown tag: {tag}")
        return False

    shape = (ny, nx)

    # Files to convert (global 1min uses prefix "1min.")
    prefix = f"{tag}."

    files_to_convert = {
        'catmzz': (f'{prefix}catmzz.bin', np.int8, 'CaMa Z mapping'),
        'flddif': (f'{prefix}flddif.bin', np.float32, 'Flood depth difference'),
        'hand': (f'{prefix}hand.bin', np.float32, 'Height above nearest drainage'),
        'elevtn': (f'{prefix}elevtn.bin', np.float32, 'Elevation'),
        'uparea': (f'{prefix}uparea.bin', np.float32, 'Upstream area'),
        'rivwth': (f'{prefix}rivwth.bin', np.float32, 'River width'),
        'visual': (f'{prefix}visual.bin', np.int8, 'Visual map'),
        'flwdir': (f'{prefix}flwdir.bin', np.int8, 'Flow direction'),
    }

    # Convert catmxy (2 records)
    catmxy_file = f"{hires_path}/{prefix}catmxy.bin"
    catmxy_nc = f"{hires_path}/{tag}.catmxy.nc"

    if os.path.exists(catmxy_file) and not os.path.exists(catmxy_nc):
        print(f"Converting {catmxy_file}...")
        with open(catmxy_file, 'rb') as f:
            catmXX = np.fromfile(f, dtype=np.int16, count=nx*ny).reshape(shape)
            catmYY = np.fromfile(f, dtype=np.int16, count=nx*ny).reshape(shape)

        ds = nc.Dataset(catmxy_nc, 'w', format='NETCDF4')
        ds.createDimension('y', ny)
        ds.createDimension('x', nx)

        var = ds.createVariable('catmXX', 'i2', ('y', 'x'))
        var[:] = catmXX

        var = ds.createVariable('catmYY', 'i2', ('y', 'x'))
        var[:] = catmYY

        ds.close()
        print(f"  Created {catmxy_nc}")

    # Convert other files
    for var_name, (bin_file, dtype, desc) in files_to_convert.items():
        bin_path = f"{hires_path}/{bin_file}"
        nc_path = f"{hires_path}/{tag}.{var_name}.nc"

        if not os.path.exists(bin_path):
            print(f"Warning: {bin_path} not found, skipping")
            continue

        if os.path.exists(nc_path):
            print(f"  {nc_path} already exists, skipping")
            continue

        print(f"Converting {bin_path}...")

        if dtype == np.int8:
            read_dtype = np.int8
            nc_dtype = 'i1'
        elif dtype == np.int16:
            read_dtype = np.int16
            nc_dtype = 'i2'
        else:
            read_dtype = np.float32
            nc_dtype = 'f4'

        with open(bin_path, 'rb') as f:
            data = np.fromfile(f, dtype=read_dtype).reshape(shape)

        ds = nc.Dataset(nc_path, 'w', format='NETCDF4')
        ds.createDimension('y', ny)
        ds.createDimension('x', nx)

        var = ds.createVariable(var_name, nc_dtype, ('y', 'x'))
        var[:] = data
        var.description = desc

        ds.close()
        print(f"  Created {nc_path}")

    print("High-res file conversion complete!")
    return True


def convert_global_1min_files(cama_dir, map_name="glb_15min"):
    """Convert global 1min high-resolution files to NetCDF"""

    base_path = f"{cama_dir}/{map_name}"
    hires_path = f"{base_path}/1min"

    if not os.path.exists(hires_path):
        print(f"High-res directory {hires_path} not found")
        return False

    # Global 1min dimensions: 21600 x 10800 (360*60 x 180*60)
    nx, ny = 21600, 10800
    shape = (ny, nx)

    print(f"Converting global 1min data: {nx} x {ny} = {nx*ny:,} pixels")

    prefix = "1min."

    # Convert catmxy (2 records, int16)
    catmxy_file = f"{hires_path}/{prefix}catmxy.bin"
    catmxy_nc = f"{hires_path}/{prefix}catmxy.nc"

    if os.path.exists(catmxy_file) and not os.path.exists(catmxy_nc):
        print(f"Converting {catmxy_file}...")
        print(f"  Reading catmXX and catmYY (int16)...")
        with open(catmxy_file, 'rb') as f:
            catmXX = np.fromfile(f, dtype=np.int16, count=nx*ny).reshape(shape)
            catmYY = np.fromfile(f, dtype=np.int16, count=nx*ny).reshape(shape)

        print(f"  Writing to NetCDF...")
        ds = nc.Dataset(catmxy_nc, 'w', format='NETCDF4')
        ds.createDimension('y', ny)
        ds.createDimension('x', nx)

        var = ds.createVariable('catmXX', 'i2', ('y', 'x'), zlib=True, complevel=4)
        var[:] = catmXX
        var.description = 'CaMa X index mapping'

        var = ds.createVariable('catmYY', 'i2', ('y', 'x'), zlib=True, complevel=4)
        var[:] = catmYY
        var.description = 'CaMa Y index mapping'

        ds.close()
        del catmXX, catmYY
        print(f"  Created {catmxy_nc}")
    elif os.path.exists(catmxy_nc):
        print(f"  {catmxy_nc} already exists, skipping")

    # Files to convert
    files_to_convert = {
        'catmzz': (np.int8, 'i1', 'CaMa Z mapping'),
        'flddif': (np.float32, 'f4', 'Flood depth difference'),
        'hand': (np.float32, 'f4', 'Height above nearest drainage'),
        'elevtn': (np.float32, 'f4', 'Elevation'),
        'uparea': (np.float32, 'f4', 'Upstream area'),
        'rivwth': (np.float32, 'f4', 'River width'),
        'visual': (np.int8, 'i1', 'Visual map'),
        'flwdir': (np.int8, 'i1', 'Flow direction'),
    }

    for var_name, (read_dtype, nc_dtype, desc) in files_to_convert.items():
        bin_file = f"{hires_path}/{prefix}{var_name}.bin"
        nc_file = f"{hires_path}/{prefix}{var_name}.nc"

        if not os.path.exists(bin_file):
            print(f"Warning: {bin_file} not found, skipping")
            continue

        if os.path.exists(nc_file):
            print(f"  {nc_file} already exists, skipping")
            continue

        print(f"Converting {bin_file}...")
        print(f"  Reading data ({read_dtype.__name__})...")

        with open(bin_file, 'rb') as f:
            data = np.fromfile(f, dtype=read_dtype).reshape(shape)

        print(f"  Data range: [{data.min()}, {data.max()}]")
        print(f"  Writing to NetCDF with compression...")

        ds = nc.Dataset(nc_file, 'w', format='NETCDF4')
        ds.createDimension('y', ny)
        ds.createDimension('x', nx)

        var = ds.createVariable(var_name, nc_dtype, ('y', 'x'), zlib=True, complevel=4)
        var[:] = data
        var.description = desc

        ds.close()
        del data
        print(f"  Created {nc_file}")

    print("\nGlobal 1min conversion complete!")
    return True


if __name__ == "__main__":
    cama_dir = "./data_for_wse/cama_maps"

    print("=" * 60)
    print("CaMa-Flood Data Conversion")
    print("=" * 60)

    print("\n[1/2] Converting CaMa base files...")
    convert_cama_base_files(cama_dir, "glb_15min")

    print("\n[2/2] Converting global 1min high-resolution files...")
    convert_global_1min_files(cama_dir, "glb_15min")
