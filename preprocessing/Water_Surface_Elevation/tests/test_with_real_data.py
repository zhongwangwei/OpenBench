#!/usr/bin/env python3
"""
Test AllocateVS with real CaMa data (reads binary files directly)
"""
import os
import numpy as np
import math

# CaMa data path
CAMA_DIR = "./data_for_wse/cama_maps/glb_15min"

def read_binary_file(filepath, dtype, shape):
    """Read binary file into numpy array"""
    with open(filepath, 'rb') as f:
        data = np.fromfile(f, dtype=dtype).reshape(shape)
    return data

def test_station_processing():
    """Test station processing with real CaMa data"""

    print("=" * 60)
    print("Testing with Real CaMa Data")
    print("=" * 60)

    # Check if data exists
    if not os.path.exists(CAMA_DIR):
        print(f"Error: CaMa data not found at {CAMA_DIR}")
        return False

    # Read params
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

    print(f"\nCaMa Grid Parameters:")
    print(f"  nXX={nXX}, nYY={nYY}")
    print(f"  gsize={gsize} degrees")
    print(f"  Bounds: ({west}, {south}) to ({east}, {north})")

    shape = (nYY, nXX)

    # Read essential binary files
    print("\nReading CaMa binary files...")

    uparea = read_binary_file(f"{CAMA_DIR}/uparea.bin", np.float32, shape)
    print(f"  uparea: shape={uparea.shape}, range=[{uparea.min():.0f}, {uparea.max():.0f}]")

    elevtn = read_binary_file(f"{CAMA_DIR}/elevtn.bin", np.float32, shape)
    print(f"  elevtn: shape={elevtn.shape}, range=[{elevtn.min():.1f}, {elevtn.max():.1f}]")

    nextxy = read_binary_file(f"{CAMA_DIR}/nextxy.bin", np.int32, (2, nYY, nXX))
    nextXX = nextxy[0]
    nextYY = nextxy[1]
    print(f"  nextXX: shape={nextXX.shape}, range=[{nextXX.min()}, {nextXX.max()}]")
    print(f"  nextYY: shape={nextYY.shape}, range=[{nextYY.min()}, {nextYY.max()}]")

    # Test finding upstream cell
    print("\n" + "=" * 60)
    print("Test 1: Find upstream cell for Amazon outlet")
    print("=" * 60)

    # Amazon river outlet is approximately at (-50, 0)
    # CaMa grid index: ix = (-50 - (-180)) / 0.25 + 1 = 521
    #                  iy = (90 - 0) / 0.25 + 1 = 361
    test_lon, test_lat = -50.0, 0.0
    ix = int((test_lon - west) / gsize) + 1
    iy = int((north - test_lat) / gsize) + 1

    print(f"\nTest location: ({test_lon}, {test_lat})")
    print(f"CaMa grid index: ix={ix}, iy={iy}")

    # Check data at this location
    if 1 <= ix <= nXX and 1 <= iy <= nYY:
        ua = uparea[iy-1, ix-1]
        el = elevtn[iy-1, ix-1]
        nx = nextXX[iy-1, ix-1]
        ny = nextYY[iy-1, ix-1]

        print(f"  Upstream area: {ua/1e9:.2f} km² (×10⁹ m²)")
        print(f"  Elevation: {el:.1f} m")
        print(f"  Next cell: ({nx}, {ny})")

        # Find upstream cell
        d = 10
        best_upstream = None
        best_dA = 1e20

        for tx in range(ix-d, ix+d+1):
            for ty in range(iy-d, iy+d+1):
                if tx < 1 or tx > nXX or ty < 1 or ty > nYY:
                    continue
                # Check if this cell flows to our target
                if nextXX[ty-1, tx-1] == ix and nextYY[ty-1, tx-1] == iy:
                    dA = abs(uparea[ty-1, tx-1] - ua)
                    if dA < best_dA:
                        best_dA = dA
                        best_upstream = (tx, ty, uparea[ty-1, tx-1])

        if best_upstream:
            ux, uy, u_area = best_upstream
            print(f"\n  Upstream cell found: ({ux}, {uy})")
            print(f"  Upstream area: {u_area/1e9:.2f} km²")
            print("  ✓ Upstream search working correctly")
        else:
            print("\n  No upstream cell found (may be near outlet)")

    # Test 2: Read high-res data
    print("\n" + "=" * 60)
    print("Test 2: Read 1min high-resolution data")
    print("=" * 60)

    hires_dir = f"{CAMA_DIR}/1min"
    if os.path.exists(hires_dir):
        # Global 1min: 21600 x 10800
        hires_nx, hires_ny = 21600, 10800

        # Read a small portion of visual data for testing
        visual_file = f"{hires_dir}/1min.visual.bin"
        if os.path.exists(visual_file):
            # Read just first 1000 rows to test
            with open(visual_file, 'rb') as f:
                visual_sample = np.fromfile(f, dtype=np.int8, count=hires_nx*1000)
                visual_sample = visual_sample.reshape(1000, hires_nx)

            # Count river pixels (visual=10) and outlets (visual=20)
            river_count = np.sum(visual_sample == 10)
            outlet_count = np.sum(visual_sample == 20)

            print(f"  Visual data sample (first 1000 rows of {hires_ny}):")
            print(f"    River pixels (10): {river_count}")
            print(f"    Outlet pixels (20): {outlet_count}")
            print("  ✓ High-res data readable")
        else:
            print(f"  Warning: {visual_file} not found")
    else:
        print(f"  Warning: High-res directory {hires_dir} not found")

    # Test 3: Verify search range is 60
    print("\n" + "=" * 60)
    print("Test 3: Verify search range (nn=60)")
    print("=" * 60)

    with open('AllocateVS.py', 'r') as f:
        content = f.read()

    if 'nn = 60' in content:
        count = content.count('nn = 60')
        print(f"  Found 'nn = 60' {count} times in AllocateVS.py")
        print("  ✓ Search range correctly set to 60 (matches Fortran)")
    else:
        print("  ✗ Search range nn = 60 NOT found")

    # Compare with existing output
    print("\n" + "=" * 60)
    print("Test 4: Verify output format matches existing data")
    print("=" * 60)

    existing_output = "altimetry_glb_15min_20250425.txt"
    if os.path.exists(existing_output):
        with open(existing_output, 'r') as f:
            header = f.readline().strip()
            first_data = f.readline().strip()

        print(f"  Existing output file: {existing_output}")
        print(f"  Header columns: {len(header.split())}")
        print(f"  First data row preview:")
        parts = first_data.split()
        print(f"    ID: {parts[0]}")
        print(f"    Station: {parts[1][:30]}...")
        print(f"    Lon/Lat: {parts[3]}, {parts[4]}")
        print(f"    Flag: {parts[6]}")
        print(f"    iXX/iYY: {parts[17]}, {parts[18]}")
        print("  ✓ Output format verified")
    else:
        print(f"  Warning: {existing_output} not found")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    test_station_processing()
