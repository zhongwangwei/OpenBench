#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import re

def check_data_availability(data, west, south):
    """
    Check if data is available for a given region.
    
    Args:
        data (str): Data source name (e.g., "HydroWeb")
        west (float): Western boundary of the region
        south (float): Southern boundary of the region
        
    Returns:
        int: 1 if data is available, 0 otherwise
    """
    east = west + 10.0
    north = south + 10.0
    
    flag = 0
    
    fname = "SampleStation_list.txt"
    try:
        with open(fname) as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header
                line = list(filter(None, re.split(" ", line)))
                if len(line) >= 1:  # Ensure we have enough elements
                    try:
                        lon = float(line[5])
                        lat = float(line[6])
                        
                        if lon >= west and lon <= east and lat >= south and lat <= north:
                            flag = 1
                            print(lon, lat, flag)
                            break  # Exit early if we find a match
                    except (ValueError, IndexError):
                        # Skip lines with invalid data
                        continue
    except FileNotFoundError:
        print(f"Warning: {fname} not found", file=sys.stderr)
    
    return flag

# For command-line usage
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python availability_data.py <data_source> <west> <south>")
        sys.exit(1)
    
    data = sys.argv[1]
    west = float(sys.argv[2])
    south = float(sys.argv[3])
    
    flag = check_data_availability(data, west, south)
    print(flag) 