import os
# Set OMP_NUM_THREADS to 1 to prevent thread explosion when using multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"

import glob
import xarray as xr
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Process site simulation data.")
parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers (default: 8)")
args = parser.parse_args()

# Define root directory
root_dir = "/tera09/luxj/OpenBench/cases/SiteTest20260106"  # Replace with actual path
nc_output_dir = "/tera11/zhwei/zhwei/Benchmark/OpenBench/dataset/Simulation/SiteTest20260106"
scenarios_dir = "/tera11/zhwei/zhwei/Benchmark/OpenBench/nml/nml-yaml/user/SiteTest20260106/senarios"
nml_dir = "/tera11/zhwei/zhwei/Benchmark/OpenBench/nml/nml-yaml/user/SiteTest20260106/nml"

# Create output directories if they don't exist
os.makedirs(nc_output_dir, exist_ok=True)
os.makedirs(scenarios_dir, exist_ok=True)
os.makedirs(nml_dir, exist_ok=True)

print(f"Starting to process root directory: {root_dir}", flush=True)
print(f"NC Output directory set to: {nc_output_dir}", flush=True)
print(f"Scenarios Output directory set to: {scenarios_dir}", flush=True)
print(f"NML Output directory set to: {nml_dir}", flush=True)

# First get all possible scenarios
all_scenarios = set()
for site_id in os.listdir(root_dir):
    site_dir = os.path.join(root_dir, site_id)
    if os.path.isdir(site_dir):
        all_scenarios.update([s for s in os.listdir(site_dir) if os.path.isdir(os.path.join(site_dir, s))])

print(f"Found {len(all_scenarios)} different scenarios", flush=True)

def process_site(args):
    site_id, site_dir, scenario, scenario_dir = args
    try:
        # print(f"Processing site: {site_id}")  # Reduced verbosity for parallel execution
        # Check for history folder
        history_dir = os.path.join(scenario_dir, "history")
        if not os.path.isdir(history_dir):
            # print(f"Skipping site {site_id}: History folder not found at {history_dir}")
            return None

        # Get all history files
        history_files = sorted(glob.glob(os.path.join(history_dir, "*.nc")))
        if not history_files:
            # print(f"Skipping site {site_id}: No NC files found in {history_dir}")
            return None

        # print(f"Site {site_id}: Found {len(history_files)} history files")

        # Only read the first file to get longitude and latitude information
        with xr.open_dataset(history_files[0]) as ds:
            lon = float(ds["lon"].values)
            lat = float(ds["lat"].values)

        # Use xr.open_mfdataset to read all files at once
        # Using parallel=False in open_mfdataset because we are already parallelizing across sites
        merged_ds = xr.open_mfdataset(history_files, combine='by_coords', data_vars='minimal', compat='override', coords='minimal', parallel=False)
        
        # Extract year range
        years = [int(f.split("_")[-1].split(".")[0]) for f in history_files]
        syear, eyear = min(years), max(years)

        # Create merged filename and path
        merged_filename = f"sim_{site_id}_{syear}_{eyear}.nc"
        merged_filepath = os.path.join(nc_output_dir, scenario, merged_filename)
        os.makedirs(os.path.dirname(merged_filepath), exist_ok=True)
        
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
        print(f"Error processing site {site_id}: {str(e)}", flush=True)
        return None

# Process each scenario
for i, scenario in enumerate(sorted(all_scenarios), 1):
    print(f"\nProcessing scenario {i}/{len(all_scenarios)}: {scenario}", flush=True)
    
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
    
    print(f"Found {valid_sites} valid sites to process", flush=True)

    # Use process pool for parallel processing
    scenario_data = []
    max_workers = args.workers
    print(f"Using {max_workers} processes for parallel processing", flush=True)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use as_completed to show progress and avoid blocking until all are done
        future_to_site = {executor.submit(process_site, arg): arg[0] for arg in process_args}
        
        completed_count = 0
        total_count = len(future_to_site)
        
        for future in as_completed(future_to_site):
            site_id = future_to_site[future]
            try:
                result = future.result()
                if result is not None:
                    scenario_data.append(result)
            except Exception as exc:
                print(f'{site_id} generated an exception: {exc}', flush=True)
            
            completed_count += 1
            if completed_count % 10 == 0 or completed_count == total_count:
                print(f"Progress: {completed_count}/{total_count} sites processed", end='\r', flush=True)
    print("") # New line after progress bar

    # After processing all sites, save CSV file for this scenario
    if scenario_data:
        df = pd.DataFrame(scenario_data)
        output_csv = os.path.join(scenarios_dir, f"{scenario}.csv")
        df.to_csv(output_csv, index=False)
        print(f"Generated {scenario}.csv containing {len(scenario_data)} sites")
        
        # Generate YAML namelist for this scenario
        # Calculate min syear and max eyear for the scenario
        scenario_syear = df['SYEAR'].min()
        scenario_eyear = df['EYEAR'].max()
        
        import textwrap
        yaml_content = textwrap.dedent(f"""\
            general:
              model_namelist: ./nml/nml-yaml/Mod_variables_definition/CoLM.yaml
              timezone: 0.0
              data_type: stn
              data_groupby: year
              fulllist: {os.path.abspath(output_csv)}
              tim_res: hour
              grid_res: 1
              syear: {scenario_syear}
              eyear: {scenario_eyear}
              prefix: ''
              suffix: .nc
              root_dir: {nc_output_dir}/
            """)
        yaml_file = os.path.join(nml_dir, f"{scenario}.yaml")
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        print(f"Generated YAML namelist for {scenario}: {yaml_file}")

print("\nAll scenarios processed successfully!")