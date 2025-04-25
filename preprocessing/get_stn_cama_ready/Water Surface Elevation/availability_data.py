#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import re
import pandas as pd

def check_data_availability(data, west, south,fname):
    """
    Check if data is available for a given region.
    
    Args:
        west (float): Western boundary of the region
        south (float): Southern boundary of the region
        
    Returns:
        int: 1 if data is available, 0 otherwise
    """
    east = west + 10.0
    north = south + 10.0
    
    flag = 0
    
    try:
        df = pd.read_csv(fname, sep=r'\s+', header=0, dtype={'ID': str})
        for index, row in df.iterrows():
            lon = float(row['lon'])
            lat = float(row['lat'])
            if lon >= west and lon <= east and lat >= south and lat <= north:
                flag = 1
                print(lon, lat, flag)
                break  # Exit early if we find a match
    except FileNotFoundError:
        print(f"Warning: {fname} not found", file=sys.stderr)
    
    return flag

# For command-line usage
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python availability_data.py <data_source> <west> <south> <fname>")
        sys.exit(1)
    data = sys.argv[1]
    west = float(sys.argv[2])
    south = float(sys.argv[3])
    fname = sys.argv[4]
    flag = check_data_availability(data, west, south,fname)
    print(flag) 