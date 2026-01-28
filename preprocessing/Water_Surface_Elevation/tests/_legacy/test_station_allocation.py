#!/usr/bin/env python3
"""
Test station allocation with real CaMa data
Tests the core allocation logic even without flwdir data
"""
import os
import numpy as np
import netCDF4 as nc
import math

# CaMa data paths
CAMA_DIR = "./data_for_wse/cama_maps/glb_15min"
HIRES_DIR = f"{CAMA_DIR}/1min"

def load_netcdf_var(filepath, var_name):
    """Load a variable from NetCDF file"""
    with nc.Dataset(filepath, 'r') as ds:
        return ds.variables[var_name][:].filled(-9999)

def test_station_allocation():
    """Test station allocation with a sample station"""

    print("=" * 70)
    print("Station Allocation Test with Real CaMa Data")
    print("=" * 70)

    # Load CaMa base parameters
    params_file = f"{CAMA_DIR}/params.txt"
    with open(params_file, 'r') as f:
        lines = f.readlines()
        nXX = int(lines[0].split('!!')[0].strip())
        nYY = int(lines[1].split('!!')[0].strip())
        gsize = float(lines[3].split('!!')[0].strip())
        west = float(lines[4].split('!!')[0].strip())
        east = float(lines[5].split('!!')[0].strip())
        south = float(lines[6].split('!!')[0].strip())
        north = float(lines[7].split('!!')[0].strip())

    print(f"\nCaMa Grid: {nXX} x {nYY}, gsize={gsize}°")
    print(f"Bounds: ({west}, {south}) to ({east}, {north})")

    # High-res parameters (1min = 1/60 degree)
    hires = 60  # cells per degree
    csize = 1.0 / hires
    hires_nx, hires_ny = 21600, 10800

    print(f"\nHigh-res Grid: {hires_nx} x {hires_ny}, csize={csize}°")

    # Load essential data
    print("\nLoading data...")

    # CaMa base data
    uparea_cama = load_netcdf_var(f"{CAMA_DIR}/uparea.nc", 'uparea')
    elevtn_cama = load_netcdf_var(f"{CAMA_DIR}/elevtn.nc", 'elevtn')

    # High-res data
    visual = load_netcdf_var(f"{HIRES_DIR}/1min.visual.nc", 'visual')
    catmXX = load_netcdf_var(f"{HIRES_DIR}/1min.catmxy.nc", 'catmXX')
    catmYY = load_netcdf_var(f"{HIRES_DIR}/1min.catmxy.nc", 'catmYY')
    rivwth = load_netcdf_var(f"{HIRES_DIR}/1min.rivwth.nc", 'rivwth')
    ele1m = load_netcdf_var(f"{HIRES_DIR}/1min.elevtn.nc", 'elevtn')
    upa1m = load_netcdf_var(f"{HIRES_DIR}/1min.uparea.nc", 'uparea')

    print(f"  visual: {visual.shape}, unique values: {np.unique(visual)}")
    print(f"  catmXX range: [{catmXX.min()}, {catmXX.max()}]")
    print(f"  catmYY range: [{catmYY.min()}, {catmYY.max()}]")

    # Test stations from the sample file
    test_stations = [
        # (ID, station, lon, lat, expected_flag_type)
        ("0000000010311", "R_GALLEGOS_GALLEGOS_KM0016", -69.68, -51.69, "river"),
        ("0000000010702", "R_SANTA-CRUZ_SANTA-CRUZ_KM0047", -69.10, -50.11, "river"),
        ("0000000010314", "R_GRANDE_GRANDE_KM0014", -67.92, -53.82, "river"),
    ]

    print("\n" + "=" * 70)
    print("Testing Station Allocation")
    print("=" * 70)

    for station_id, station_name, lon0, lat0, expected_type in test_stations:
        print(f"\n--- Station: {station_name} ---")
        print(f"    Location: ({lon0}, {lat0})")

        # Calculate high-res index (1-based)
        # For global 1min: west=-180, north=90
        hires_west = -180.0
        hires_north = 90.0

        lon_term = lon0 - hires_west - csize/2.0
        lat_term = hires_north - csize/2.0 - lat0

        ix = int(lon_term * hires) + 1
        iy = int(lat_term * hires) + 1

        print(f"    High-res index (ix, iy): ({ix}, {iy})")

        # Check bounds
        if ix < 1 or ix > hires_nx or iy < 1 or iy > hires_ny:
            print(f"    ERROR: Out of bounds!")
            continue

        # Get visual value at this location
        vis_val = visual[iy-1, ix-1]
        riv_wth = rivwth[iy-1, ix-1]

        print(f"    Visual value: {vis_val}")
        print(f"    River width: {riv_wth}")

        # Determine station type based on visual
        if vis_val == 10:
            print(f"    Station is ON river centerline ✓")
            flag = 10
        elif vis_val == 20:
            print(f"    Station is at unit-catchment outlet ✓")
            flag = 12
        elif riv_wth > 0 and riv_wth != -9999:
            print(f"    Station is in river channel (width={riv_wth}m)")
            flag = 11
        else:
            print(f"    Station needs correction to river")
            flag = 20

        # Get CaMa grid mapping
        iXX = catmXX[iy-1, ix-1]
        iYY = catmYY[iy-1, ix-1]

        if iXX > 0 and iYY > 0:
            print(f"    CaMa grid (iXX, iYY): ({iXX}, {iYY})")

            # Get CaMa data at this location
            cama_uparea = uparea_cama[iYY-1, iXX-1]
            cama_elevtn = elevtn_cama[iYY-1, iXX-1]

            print(f"    CaMa upstream area: {cama_uparea/1e9:.2f} km² (×10⁹ m²)")
            print(f"    CaMa elevation: {cama_elevtn:.1f} m")

            # Calculate CaMa grid center coordinates
            lon_cama = west + (iXX - 0.5) * gsize
            lat_cama = north - (iYY - 0.5) * gsize

            print(f"    CaMa grid center: ({lon_cama:.2f}, {lat_cama:.2f})")
        else:
            print(f"    Warning: No valid CaMa mapping (catmXX={iXX}, catmYY={iYY})")

        # Test find_nearest_river logic (search range nn=60)
        if vis_val != 10 and vis_val != 20:
            print(f"\n    Testing find_nearest_river (nn=60)...")
            nn = 60
            kx, ky = ix, iy
            lag = 1e20

            for dy in range(-nn, nn+1):
                for dx in range(-nn, nn+1):
                    jx = ix + dx
                    jy = iy + dy

                    if jx < 1 or jx > hires_nx or jy < 1 or jy > hires_ny:
                        continue

                    if catmXX[jy-1, jx-1] <= 0 or catmYY[jy-1, jx-1] <= 0:
                        continue

                    lag_now = math.sqrt(dx**2 + dy**2)

                    if lag_now < lag:
                        if visual[jy-1, jx-1] == 10 or visual[jy-1, jx-1] == 20:
                            kx = jx
                            ky = jy
                            lag = lag_now

            if kx != ix or ky != iy:
                print(f"    Found nearest river at ({kx}, {ky}), distance={lag:.1f} pixels")
                print(f"    Visual at river: {visual[ky-1, kx-1]}")
            else:
                print(f"    No river found within search range")

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)

    return True


if __name__ == "__main__":
    test_station_allocation()
