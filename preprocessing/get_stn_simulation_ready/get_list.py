import os
import glob
import xarray as xr
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Define root directory
root_dir = "/share/home/dq013/zhwei/OpenBench/20250120/data/simulation/single_point_test"  # Replace with actual path
output_dir = "./output"  # Directory for CSV file output

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print(f"Starting to process root directory: {root_dir}")
print(f"Output directory set to: {output_dir}")

# First get all possible scenarios
all_scenarios = set()
for site_id in os.listdir(root_dir):
    site_dir = os.path.join(root_dir, site_id)
    if os.path.isdir(site_dir):
        all_scenarios.update([s for s in os.listdir(site_dir) if os.path.isdir(os.path.join(site_dir, s))])

print(f"Found {len(all_scenarios)} different scenarios")

def process_site(args):
    site_id, site_dir, scenario, scenario_dir = args
    try:
        print(f"Processing site: {site_id}")  # Add site processing status
        # Check for history folder
        history_dir = os.path.join(scenario_dir, "history")
        if not os.path.isdir(history_dir):
            print(f"Skipping site {site_id}: History folder not found")
            return None

        # Get all history files
        history_files = sorted(glob.glob(os.path.join(history_dir, "*.nc")))
        if not history_files:
            print(f"Skipping site {site_id}: No NC files found")
            return None

        print(f"Site {site_id}: Found {len(history_files)} history files")

        # Only read the first file to get longitude and latitude information
        with xr.open_dataset(history_files[0]) as ds:
            lon = float(ds["lon"].values)
            lat = float(ds["lat"].values)

        # Use xr.open_mfdataset to read all files at once
        merged_ds = xr.open_mfdataset(history_files, combine='by_coords')
        
        # Extract year range
        years = [int(f.split("_")[-1].split(".")[0]) for f in history_files]
        syear, eyear = min(years), max(years)

        # Create merged filename and path
        merged_filename = f"sim_{site_id}_{syear}_{eyear}.nc"
        merged_filepath = os.path.join(scenario_dir, merged_filename)
        
        # Save merged file
        merged_ds.to_netcdf(merged_filepath)
        merged_ds.close()  # Explicitly close dataset

        return {
            "scenario": scenario,
            "ID": site_id,
            "SYEAR": syear,
            "EYEAR": eyear,
            "LON": lon,
            "LAT": lat,
            "DIR": merged_filepath,
        }
    except Exception as e:
        print(f"Error processing site {site_id}: {str(e)}")
        return None

# Process each scenario
for i, scenario in enumerate(sorted(all_scenarios), 1):
    print(f"\nProcessing scenario {i}/{len(all_scenarios)}: {scenario}")
    
    # Prepare parameters for parallel processing
    process_args = []
    valid_sites = 0
    for site_id in os.listdir(root_dir):
        site_dir = os.path.join(root_dir, site_id)
        if not os.path.isdir(site_dir):
            continue
        
        scenario_dir = os.path.join(site_dir, scenario)
        if not os.path.isdir(scenario_dir):
            continue
            
        process_args.append((site_id, site_dir, scenario, scenario_dir))
        valid_sites += 1
    
    print(f"Found {valid_sites} valid sites to process")

    # Use process pool for parallel processing
    scenario_data = []
    max_workers = multiprocessing.cpu_count()
    print(f"Using {max_workers} processes for parallel processing")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_site, process_args)
        
        # Collect results
        for result in results:
            if result is not None:
                scenario_data.append(result)

    # After processing all sites, save CSV file for this scenario
    if scenario_data:
        df = pd.DataFrame(scenario_data)
        output_csv = os.path.join(output_dir, f"{scenario}.csv")
        df.to_csv(output_csv, index=False)
        print(f"Generated {scenario}.csv containing {len(scenario_data)} sites")

print("\nAll scenarios processed successfully!")