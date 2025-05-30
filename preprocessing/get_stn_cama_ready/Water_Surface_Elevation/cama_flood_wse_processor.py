#!/usr/bin/env python3
"""
CAMA-Flood Water Surface Elevation (WSE) Evaluation dataset preparation Script
This script processes water surface elevation data from HydroWeb and other sources,
performs bias correction, and prepares data for CAMA-Flood model evaluation.
the original script is from AltiMaP and in Fortran, but I modified it to be more general and flexible.
***Issue: when the station is very close to the edge of high resolution map, the program may not work***
***Reason: the program did not have the function to handle the case when it search to pixel to the next tile***
***This is also exist in AltiMaP, need to be fixed.***
"""


import os
import sys
import subprocess
import time
import datetime
from pathlib import Path

# Import the availability data function directly
from availability_data import check_data_availability
# Import the AllocateVS class
from AllocateVS import AllocateVS
# Import the set_name module
from set_name import set_name, get_region_boundaries, check_region_exists



# CaMa-Flood directory
CAMA_DIR = "./data_for_wse/cama_maps/"
# Map configuration
MAP = "glb_15min"
TAG = "3sec"
data = "HydroWeb"
# Output directory
OUTDIR = "./out"
os.makedirs(OUTDIR, exist_ok=True)

fname = "SampleStation_list_new.txt"

def check_netcdf_installed():
    """Check if netCDF4 is installed"""
    try:
        import netCDF4
        return True
    except ImportError:
        print("Warning: netCDF4 module not found. Please install it using 'pip install netCDF4'")
        return False

def main():
    print("Starting calculations........")
    
    # Check if netCDF4 is installed
    has_netcdf = check_netcdf_installed()
    
    # Create header for output file
    with open("tmp.txt", "w") as f:
        f.write(f"{'ID':13}{'station':64}{'dataname':12}{'lon':12}{'lat':10}{'satellite':17}{'flag':6}{'elevation':12}{'dist_to_mouth':15}{'kx1':10}{'ky1':8}{'kx2':8}{'ky2':8}{'dist1':14}{'dist2':12}{'rivwth':12}{'ix':10}{'iy':8}{'Lon_CaMa':12}{'Lat_CaMa':12}{'EGM08':12}{'EGM96':10}\n")

    # Process each region
    for south in range(-60, 90, 10):
        for west in range(-180, 180, 10):
            # Get region name directly using the imported function
            cname = set_name(west, south)
            #print(f"\nProcessing region: {cname} (west={west}, south={south})")
            # Check if region files exist
            if check_region_exists(CAMA_DIR, MAP, TAG, cname):
                # Check data availability directly using the imported function
                flag = check_data_availability(data, west, south,fname)
                if flag == 1:
                    print(f"Data available for region: {west} {south} {data}")
                    print(f"Parameters: west={west}, south={south}, data={data}, CAMA_DIR={CAMA_DIR}, MAP={MAP}, TAG={TAG}, OUTDIR={OUTDIR}")
                    
                    try:
                        # Initialize AllocateVS class
                        allocate_vs = AllocateVS(west, south, data, CAMA_DIR, MAP, TAG, OUTDIR,fname)
                        
                        # Save highres data to netCDF for inspection
                        if has_netcdf:
                            print(f"Saving region data to NetCDF for inspection: {cname}")
                            nc_file = allocate_vs.save_to_netcdf(None)
                            if nc_file:
                                print(f"Successfully saved data to {nc_file}")
                        
                        # Process stations
                        results = allocate_vs.process_stations()
                        if not results:
                        #    print(f"No results for region {cname}")
                            continue
                        
                        # Append results to the output file
                        with open("tmp.txt", "a") as f:
                            for result in results:
                                try:
                                    f.write(
                                        f"{result['id']:13s} {result['station']:60s} {result['dataname']:10s} "
                                        f"{result['lon']:10.2f} {result['lat']:10.2f} {result['sat']:15s} "
                                        f"{result['flag']:4d} {result['ele']:10.2f} {result['diffdist']:13.2f} "
                                        f"{result['kx1']:8d} {result['ky1']:8d} {result['kx2']:8d} {result['ky2']:8d} "
                                        f"{result['dist1']:12.2f} {result['dist2']:12.2f} {result['rivwth']:12.2f} "
                                        f"{result['iXX']:8d} {result['iYY']:8d} {result['lon_cama']:10.5f} {result['lat_cama']:10.5f} {result['egm08']:10.2f} {result['egm96']:10.2f}\n"
                                    )
                                except Exception as e:
                                    print(f"Error writing result: {e}")
                                    print(f"Problematic result: {result}")
                                    continue
                        
                    except Exception as e:
                        print(f"Error processing region {cname}: {e}")
                        continue
            else:
                pass

    # Save results
    try:
        day = datetime.datetime.now().strftime("%Y%m%d")
        output_file = f"{OUTDIR}/altimetry_{MAP}_{day}.txt"
        os.rename("tmp.txt", output_file)
        print("Saving ...")
        print(output_file)
    except Exception as e:
        print(f"Error saving results: {e}")
        if os.path.exists("tmp.txt"):
            print("tmp.txt still exists due to error")

if __name__ == "__main__":
    main() 