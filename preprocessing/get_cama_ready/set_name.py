#!/usr/bin/env python3
# ===============================================
import sys
import os
from pathlib import Path

def set_name(lon, lat):
    """
    Convert longitude and latitude to a region name string.
    
    Args:
        lon (float): Longitude value
        lat (float): Latitude value
        
    Returns:
        str: Region name in format 'nXXeYYY' or 'sXXwYYY'
    """
    # ===============================================
    # Convert longitude and latitude to string format
    if lon < 0:
        ew = 'w'
        clon = f"{int(-lon):03d}"
    else:
        ew = 'e'
        clon = f"{int(lon):03d}"

    if lat < 0:
        sn = 's'
        clat = f"{int(-lat):02d}"
    else:
        sn = 'n'
        clat = f"{int(lat):02d}"

    cname = f"{sn}{clat}{ew}{clon}"
    return cname

def get_region_boundaries(lon, lat, region_size=10):
    """
    Get the boundaries of a region based on longitude and latitude.
    
    Args:
        lon (float): Longitude value
        lat (float): Latitude value
        region_size (float): Size of the region in degrees
        
    Returns:
        tuple: (west, south, east, north) boundaries
    """
    # Calculate the nearest region boundaries
    west = int(lon / region_size) * region_size
    south = int(lat / region_size) * region_size
    east = west + region_size
    north = south + region_size
    
    return west, south, east, north

def check_region_exists(cama_dir, map_name, tag, region_name):
    """
    Check if a region's data files exist.
    
    Args:
        cama_dir (str): CaMa-Flood directory
        map_name (str): Map name (e.g., 'glb_15min')
        tag (str): Resolution tag (e.g., '3sec')
        region_name (str): Region name
        
    Returns:
        bool: True if region files exist, False otherwise
    """
    catmxy_file = f"{cama_dir}/data_for_wse/cama_maps/{map_name}/{tag}/{region_name}.catmxy.bin"
    return os.path.exists(catmxy_file) and os.path.getsize(catmxy_file) > 0

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python set_name.py <longitude> <latitude>")
        sys.exit(1)
    
    lon = float(sys.argv[1])
    lat = float(sys.argv[2])
    
    cname = set_name(lon, lat)
    print(cname) 